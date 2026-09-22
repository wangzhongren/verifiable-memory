"""存储后端：SQLite（工作存储）与 JSON 文件（证据格式）。

分工（2026-09-22 定）：
- SQLite 是日常记忆库：ops 表一行一操作、只追加，哈希链住在行里；
  state 表是物化缓存（可丢弃、可重建）；写事务用 BEGIN IMMEDIATE
  串行化并发写者，WAL 允许并发读。落库前在事务内核对链头，被并发
  进程抢先则抛 StaleStateError，由会话层重载重试（乐观并发协议）。
- JSON 是证据格式：replay.py / verify.py 只吃导出的 canonical
  session.json（export_evidence），与存储引擎解耦——换后端核验层
  零改动。demo/ 冻结证据仍是 JSON。
- 两个后端的条目结构完全相同；同一操作序列在两后端下产出的证据
  文件逐字节一致（checks.py 有断言）。op/result/proof 以保序 JSON
  存储（不排序键），往返后与旧 JSON 后端逐字节一致。

不变量：日志是唯一事实来源，state/meta 皆为派生缓存；工作路径核对
当前缓存、独立检查点与尾部日志；历史段的完整审计由 replay/verify 负责。
"""

import json
import os
import sqlite3
from pathlib import Path

from .store import STATE_FORMAT, state_digest
from .audit import (AuditError, DB_FORMAT, SESSION_FORMAT, check_evidence,
                    log_genesis)

SQLITE_SUFFIXES = {'.db', '.sqlite', '.sqlite3'}

# 每 interval 条操作在同一事务中复制 state 到 checkpoint_state，
# 并记录状态哈希、日志哈希与全局序号。复制成本 O(N)，之后快照保持不变。
CHECKPOINT_INTERVAL = 512

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ops (
  seq INTEGER PRIMARY KEY,
  op_id TEXT NOT NULL UNIQUE,
  category TEXT NOT NULL,
  source TEXT NOT NULL,
  utterance TEXT NOT NULL,
  op TEXT NOT NULL,
  status TEXT NOT NULL,
  error TEXT,
  result TEXT,
  proof TEXT,
  hash_before TEXT NOT NULL,
  hash_after TEXT NOT NULL,
  prev_entry_hash TEXT NOT NULL,
  entry_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state (
  name TEXT PRIMARY KEY,
  record TEXT NOT NULL,
  seq_written INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoint_state (
  name TEXT PRIMARY KEY,
  record TEXT NOT NULL
);
"""

_SQLITE_COLS = ('seq', 'op_id', 'category', 'source', 'utterance', 'op', 'status',
                'error', 'result', 'proof', 'hash_before', 'hash_after',
                'prev_entry_hash', 'entry_hash')


class StorageError(Exception):
    """存储层失败（IO、损坏、数据库忙）。"""


class StaleStateError(StorageError):
    """链头与预期不符：会话已被并发进程推进（乐观并发协议）。"""


def is_sqlite_path(path):
    return Path(path).suffix in SQLITE_SUFFIXES


def _order_json(obj):
    """保序 JSON 文本（不排序键）：往返后与内存中的 dict 插入顺序
    一致，保证证据文件逐字节可复现。"""
    return json.dumps(obj, ensure_ascii=False)


def _maybe_text(obj):
    return None if obj is None else _order_json(obj)


def _entry_from_row(row):
    """一行 ops → 证据条目。字段顺序与旧 JSON 后端的条目字典逐键
    一致（error / result / proof 按有无插入），因此导出文件与旧后端
    逐字节相同。"""
    entry = {'op_id': row['op_id'], 'category': row['category'],
             'source': row['source'], 'utterance': row['utterance'],
             'op': json.loads(row['op']), 'status': row['status'],
             'state_hash_before': row['hash_before']}
    if row['error'] is not None:
        entry['error'] = row['error']
    if row['result'] is not None:
        entry['result'] = json.loads(row['result'])
    if row['proof'] is not None:
        entry['proof'] = json.loads(row['proof'])
    entry['state_hash_after'] = row['hash_after']
    entry['prev_entry_hash'] = row['prev_entry_hash']
    entry['entry_hash'] = row['entry_hash']
    return entry


class JsonStorage:
    """整文件原子重写。用于 demo 冻结证据与小型会话。

    已知限制：并发写互不合并（后写覆盖先写），并发场景必须用
    SQLite 后端——checks.py 的并发压力测试即为此设计。
    """

    def __init__(self, path, capacity):
        self.path = Path(path)
        self.capacity = capacity

    def _read(self):
        if not self.path.exists():
            raise StorageError(f'会话文件不存在：{self.path}')
        try:
            return json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError(f'会话文件无法解析：{exc}') from exc

    def meta(self):
        raw = self._read()
        if raw.get('format') != SESSION_FORMAT:
            raise StorageError(f"会话格式不符: {raw.get('format')!r}（期望 {SESSION_FORMAT}）")
        return {'format': raw['format'], 'capacity': raw.get('capacity')}

    def entries(self):
        return self._read().get('entries', [])

    def snapshot(self):
        """单次一致读（JSON 文件原子替换保证）。

        JSON 无检查点机制：全部条目即尾部，state_records 为 None
        （会话层据此走全量重放路径）。"""
        raw = self._read()
        if raw.get('format') != SESSION_FORMAT:
            raise StorageError(f"会话格式不符: {raw.get('format')!r}（期望 {SESSION_FORMAT}）")
        try:
            check_evidence(raw)
        except AuditError as exc:
            raise StorageError(str(exc)) from exc
        return {'capacity': raw['capacity'], 'entries': raw['entries'],
                'state_records': None, 'snapshot_seq': 0,
                'snapshot_hash': None, 'chain_head': None,
                'n_ops': raw['n_ops'], 'log_head': raw['log_head']}

    @classmethod
    def create(cls, path, capacity):
        path = Path(path)
        payload = {'format': SESSION_FORMAT, 'capacity': capacity, 'entries': [],
                   'n_ops': 0, 'log_head': log_genesis(capacity)}
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, path)
        return cls(path, capacity)

    def append_entry(self, entry, mutated, seq):
        raw = self._read()
        if raw.get('format') != SESSION_FORMAT:
            raise StorageError(f"会话格式不符: {raw.get('format')!r}")
        try:
            check_evidence(raw)
        except AuditError as exc:
            raise StorageError(str(exc)) from exc
        if raw['n_ops'] != seq - 1 or raw['log_head'] != entry['prev_entry_hash']:
            raise StaleStateError('JSON 会话已被推进')
        raw['entries'] = list(raw['entries']) + [entry]
        raw['n_ops'] = seq
        raw['log_head'] = entry['entry_hash']
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, self.path)


class SqliteStorage:
    """SQLite 工作存储。所有写走单个立即事务；并发安全由
    BEGIN IMMEDIATE + 链头核对提供。state 表是可重建缓存。"""

    def __init__(self, conn, path):
        self.conn = conn
        self.path = Path(path)

    # ---- 生命周期 ----

    @classmethod
    def create(cls, path, capacity):
        path = Path(path)
        conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.executescript(_SCHEMA)
        empty_head = state_digest({'format': STATE_FORMAT,
                                   'capacity': capacity, 'slots': {}})
        conn.execute('BEGIN IMMEDIATE')
        conn.execute("INSERT INTO meta(key, value) VALUES ('format', ?)", (DB_FORMAT,))
        conn.execute("INSERT INTO meta(key, value) VALUES ('capacity', ?)", (str(capacity),))
        conn.execute("INSERT INTO meta(key, value) VALUES ('chain_head', ?)", (empty_head,))
        conn.execute("INSERT INTO meta(key, value) VALUES ('snapshot_seq', ?)", ('0',))
        conn.execute("INSERT INTO meta(key, value) VALUES ('snapshot_hash', ?)", (empty_head,))
        conn.execute("INSERT INTO meta(key, value) VALUES ('n_ops', ?)", ('0',))
        conn.execute("INSERT INTO meta(key, value) VALUES ('log_head', ?)",
                     (log_genesis(capacity),))
        conn.execute("INSERT INTO meta(key, value) VALUES ('snapshot_log_hash', ?)",
                     (log_genesis(capacity),))
        conn.execute('COMMIT')
        return cls(conn, path)

    @classmethod
    def open(cls, path):
        path = Path(path)
        if not path.exists():
            raise StorageError(f'数据库不存在：{path}')
        conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        conn.execute('PRAGMA journal_mode=WAL')
        return cls(conn, path)

    def close(self):
        self.conn.close()

    # ---- 读 ----

    def _meta(self):
        rows = self.conn.execute('SELECT key, value FROM meta').fetchall()
        return dict(rows)

    def _entries(self):
        rows = self.conn.execute(
            f'SELECT {",".join(_SQLITE_COLS)} FROM ops ORDER BY seq').fetchall()
        return [_entry_from_row(dict(zip(_SQLITE_COLS, row))) for row in rows]

    def _state_cache(self):
        rows = self.conn.execute('SELECT name, record FROM state').fetchall()
        return {name: json.loads(record) for name, record in rows}

    def meta(self):
        raw = self._meta()
        if raw.get('format') != DB_FORMAT:
            raise StorageError(f"数据库格式不符: {raw.get('format')!r}（期望 {DB_FORMAT}）")
        return {'format': raw['format'], 'capacity': int(raw['capacity']),
                'chain_head': raw.get('chain_head'),
                'snapshot_seq': int(raw.get('snapshot_seq', 0)),
                'snapshot_hash': raw.get('snapshot_hash'),
                'n_ops': int(raw['n_ops']), 'log_head': raw['log_head']}

    def snapshot(self):
        """一次一致读（WAL 读事务快照）：meta + state 缓存 + 尾部日志。

        只取 seq > snapshot_seq 的尾部条目——checkpoint_state 是固定
        快照，state 是最新缓存；load 从快照重放尾部再比对缓存，全量
        重放核验交给离线的 replay/verify。几个读若分开做，并发写者
        可能在两次 SELECT 之间提交，造成假阳性——必须同处一个读事务。
        """
        self.conn.execute('BEGIN')
        try:
            raw = self._meta()
            if raw.get('format') != DB_FORMAT:
                raise StorageError(f"数据库格式不符: {raw.get('format')!r}（期望 {DB_FORMAT}）")
            snapshot_seq = int(raw.get('snapshot_seq', 0))
            rows = self.conn.execute(
                f'SELECT {",".join(_SQLITE_COLS)} FROM ops WHERE seq > ? ORDER BY seq',
                (snapshot_seq,)).fetchall()
            entries = [_entry_from_row(dict(zip(_SQLITE_COLS, row))) for row in rows]
            cache = self._state_cache()
            checkpoint = {name: json.loads(record) for name, record in
                          self.conn.execute('SELECT name, record FROM checkpoint_state')}
            max_seq = self.conn.execute('SELECT COALESCE(MAX(seq), 0) FROM ops').fetchone()[0]
            if max_seq != int(raw['n_ops']):
                raise StorageError('日志最大序号与 n_ops 不符')
            expected_seqs = list(range(snapshot_seq + 1, int(raw['n_ops']) + 1))
            if [row[0] for row in rows] != expected_seqs:
                raise StorageError('尾部日志 seq 不连续')
            if snapshot_seq:
                boundary = self.conn.execute(
                    'SELECT hash_after, entry_hash FROM ops WHERE seq = ?',
                    (snapshot_seq,)).fetchone()
                if boundary != (raw['snapshot_hash'], raw['snapshot_log_hash']):
                    raise StorageError('检查点锚与边界日志不符')
        finally:
            self.conn.execute('COMMIT')
        return {'capacity': int(raw['capacity']), 'entries': entries,
                'chain_head': raw.get('chain_head'),
                'snapshot_seq': snapshot_seq,
                'snapshot_hash': raw.get('snapshot_hash'),
                'n_ops': int(raw['n_ops']), 'log_head': raw['log_head'],
                'snapshot_log_hash': raw['snapshot_log_hash'],
                'checkpoint_records': checkpoint, 'state_records': cache}

    # ---- 写 ----

    def append_entry(self, entry, mutated, seq):
        """把一条条目落库（单立即事务）：核对链头 → 插 ops → 更新
        state 缓存与两种链头；到达检查点间隔时原子复制快照并推进
        snapshot_seq / snapshot_hash / snapshot_log_hash。被并发进程抢先抛
        StaleStateError。"""
        try:
            self.conn.execute('BEGIN IMMEDIATE')
            meta = self._meta()
            head = meta['chain_head']
            if (head != entry['state_hash_before']
                    or meta['log_head'] != entry['prev_entry_hash']
                    or int(meta['n_ops']) != seq - 1):
                raise StaleStateError(
                    f'链头被并发进程推进：{str(head)[:12]} ≠ 预期 '
                    f'{entry["state_hash_before"][:12]}')
            self.conn.execute(
                f'INSERT INTO ops ({",".join(_SQLITE_COLS)}) '
                f'VALUES ({",".join("?" * len(_SQLITE_COLS))})',
                (seq, entry['op_id'], entry['category'], entry['source'],
                 entry['utterance'], _order_json(entry['op']), entry['status'],
                 entry.get('error'), _maybe_text(entry.get('result')),
                 _maybe_text(entry.get('proof')),
                 entry['state_hash_before'], entry['state_hash_after'],
                 entry['prev_entry_hash'], entry['entry_hash']))
            if mutated is not None:
                name, record = mutated
                self.conn.execute(
                    'INSERT OR REPLACE INTO state(name, record, seq_written) '
                    'VALUES (?, ?, ?)', (name, _order_json(record), seq))
            self.conn.execute(
                "UPDATE meta SET value = ? WHERE key = 'chain_head'",
                (entry['state_hash_after'],))
            self.conn.execute("UPDATE meta SET value = ? WHERE key = 'log_head'",
                              (entry['entry_hash'],))
            self.conn.execute("UPDATE meta SET value = ? WHERE key = 'n_ops'",
                              (str(seq),))
            snapshot_seq = int(self.conn.execute(
                "SELECT value FROM meta WHERE key = 'snapshot_seq'").fetchone()[0])
            if seq - snapshot_seq >= CHECKPOINT_INTERVAL:
                self.conn.execute('DELETE FROM checkpoint_state')
                self.conn.execute('INSERT INTO checkpoint_state(name, record) '
                                  'SELECT name, record FROM state')
                self.conn.execute(
                    "UPDATE meta SET value = ? WHERE key = 'snapshot_log_hash'",
                    (entry['entry_hash'],))
                self.conn.execute(
                    "UPDATE meta SET value = ? WHERE key = 'snapshot_seq'", (str(seq),))
                self.conn.execute(
                    "UPDATE meta SET value = ? WHERE key = 'snapshot_hash'",
                    (entry['state_hash_after'],))
            self.conn.execute('COMMIT')
        except StaleStateError:
            self.conn.execute('ROLLBACK')
            raise
        except sqlite3.IntegrityError as exc:
            self.conn.execute('ROLLBACK')
            # 在立即事务中已用全局序号和事件链头排除了并发推进。
            raise StorageError(f'数据库约束失败：{exc}') from exc
        except sqlite3.OperationalError as exc:
            self._safe_rollback()
            raise StorageError(f'数据库忙/失败：{exc}') from exc
        except Exception:
            self._safe_rollback()
            raise

    def _safe_rollback(self):
        try:
            self.conn.execute('ROLLBACK')
        except sqlite3.OperationalError:
            pass


def export_evidence(db_path, out_path):
    """从 SQLite 导出 canonical 证据 session.json（原子写，拒绝覆盖
    由调用方负责）。返回导出的载荷。

    导出读**全量** ops 日志（与检查点无关）——证据是完整日志，尾部
    截取只属于工作路径的 load 优化。"""
    storage = SqliteStorage.open(db_path)
    try:
        storage.conn.execute('BEGIN')
        try:
            raw = storage._meta()
            if raw.get('format') != DB_FORMAT:
                raise StorageError(f"数据库格式不符: {raw.get('format')!r}（期望 {DB_FORMAT}）")
            entries = storage._entries()
        finally:
            storage.conn.execute('COMMIT')
    finally:
        storage.close()
    payload = {'format': SESSION_FORMAT, 'capacity': int(raw['capacity']),
               'entries': entries, 'n_ops': int(raw['n_ops']), 'log_head': raw['log_head']}
    out_path = Path(out_path)
    tmp = out_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, out_path)
    return payload
