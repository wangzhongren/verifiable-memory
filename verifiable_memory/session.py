"""会话编排：op 日志 + 哈希链 + 持久化（后端可插拔）。

日志是唯一事实来源；state 表/文件快照皆为派生缓存，load 时必须
通过"重放 vs 缓存"核对——对缓存或日志的任何一手改动在这里暴露。

两个后端（storage.py）：
- JSON 文件（demo 冻结证据、小会话）：整文件原子重写；
- SQLite（日常记忆库）：一行一 op 只追加、BEGIN IMMEDIATE 串行写。

并发协议（仅 SQLite）：落库前在立即事务内核对链头，被并发进程
抢先则重载全部日志、重建状态后重试（乐观并发）；重试耗尽抛
SessionError。

证据契约：entry[i].state_hash_before == entry[i-1].state_hash_after；
首条 before == 空状态哈希；错误条目 after == before（无副作用）。
replay.py / verify.py 只吃导出的证据 JSON（storage.export_evidence），
永不直读数据库——核验与存储引擎解耦。
"""

from pathlib import Path

from . import executor
from . import store
from .storage import (JsonStorage, SQLITE_SUFFIXES, SESSION_FORMAT,
                     SqliteStorage, StaleStateError, StorageError)
from .store import CAPACITY_DEFAULT, state_digest

CATEGORIES = {
    'teach': {'teach_fact', 'teach_rule'},
    'correct': {'correct_fact', 'correct_rule'},
    'ask': store.READ_OPS,
}


class SessionError(Exception):
    """会话缺失、被篡改、与实现不一致或并发重试耗尽。"""


def _empty_state_hash(capacity):
    return state_digest({'format': store.STATE_FORMAT,
                         'capacity': capacity, 'slots': {}})


def _structural_check(entries, capacity, anchor=None):
    """哈希链结构检查：首条锚定 anchor（默认空状态哈希）、逐条相扣、
    错误条目无副作用。快照路径下 anchor = 快照哈希。"""
    expected_before = anchor if anchor is not None else _empty_state_hash(capacity)
    for entry in entries:
        if entry.get('state_hash_before') != expected_before:
            raise SessionError(f"哈希链断裂于 {entry.get('op_id')}："
                               f"before={str(entry.get('state_hash_before'))[:12]} "
                               f"期望 {str(expected_before)[:12]}")
        status = entry.get('status')
        if status == 'error':
            if entry.get('state_hash_after') != entry.get('state_hash_before'):
                raise SessionError(f"{entry.get('op_id')} 是错误条目却有副作用")
        elif status != 'ok':
            raise SessionError(f"{entry.get('op_id')} 状态未知: {status}")
        expected_before = entry['state_hash_after']


def _rebuild(entries, capacity):
    """从日志重放所有成功的写操作，重建 Store（日志是唯一事实来源）。"""
    rebuilt = store.Store(capacity)
    for entry in entries:
        if entry.get('status') == 'ok' and entry['op']['op'] in store.WRITE_OPS:
            rebuilt.apply_write(entry['op'], op_id=entry['op_id'],
                                utterance=entry.get('utterance', ''))
    return rebuilt


class Session:
    """一次连续教学/问答会话。所有状态变化都留下日志与哈希链。

    SQLite 后端下 self.entries 只持有**尾部**（最近检查点之后的条
    目）——快照部分由 state 表 O(N) 重建、用 snapshot_hash 锚定；
    全量重放核验是 replay.py/verify.py 的离线职责，不在每条命令的
    热路径上。self.n_ops 为日志总条数（无检查点概念的后端 = 尾部长
    度）。"""

    def __init__(self, path, capacity, entries, store_obj, storage, n_ops=None):
        self.path = Path(path)
        self.capacity = capacity
        self.entries = entries
        self.store = store_obj
        self.storage = storage
        self.n_ops = n_ops if n_ops is not None else len(entries)

    # ---- 生命周期 ----

    @classmethod
    def create(cls, path, capacity=CAPACITY_DEFAULT):
        path = Path(path)
        if path.exists():
            raise SessionError(f'会话已存在，拒绝覆盖：{path}（如需重开请先 reset）')
        if path.suffix in SQLITE_SUFFIXES:
            storage = SqliteStorage.create(path, capacity)
        else:
            storage = JsonStorage.create(path, capacity)
        return cls(path, capacity, [], store.Store(capacity), storage)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise SessionError(f'会话不存在：{path}（用 teach/ask/correct 会自动创建）')
        if path.suffix in SQLITE_SUFFIXES:
            storage = SqliteStorage.open(path)
        else:
            storage = JsonStorage(path, capacity=None)
        try:
            snap = storage.snapshot()
        except StorageError as exc:
            raise SessionError(str(exc)) from exc
        capacity = snap['capacity']
        entries = snap['entries']
        if not isinstance(capacity, int) or not isinstance(entries, list):
            raise SessionError('会话缺少 capacity 或 entries')

        if snap.get('state_records') is not None and snap.get('snapshot_seq'):
            # 快照路径（SQLite，已到过检查点）：state 表 O(N) 重建 +
            # snapshot_hash 锚定 + 尾部重放。快照段内的日志篡改由此处
            # **有意放行**——它由离线的 replay/verify 全量重放负责（数
            # 据库工程的 WAL/checkpoint 分工）；快照缓存本身的篡改被
            # snapshot_hash 锚抓住。snapshot_seq=0（首个检查点前）时
            # state 缓存领先于快照锚，必须走全量重放。
            rebuilt = store.Store(capacity)
            rebuilt.slots.update(snap['state_records'])
            snapshot_hash = snap.get('snapshot_hash')
            if rebuilt.state_hash() != snapshot_hash:
                raise SessionError('state 快照与 snapshot_hash 不符——缓存被篡改或损坏')
            _structural_check(entries, capacity, anchor=snapshot_hash)
            for entry in entries:
                if entry.get('status') == 'ok' and entry['op']['op'] in store.WRITE_OPS:
                    try:
                        rebuilt.apply_write(entry['op'], op_id=entry['op_id'],
                                            utterance=entry.get('utterance', ''))
                    except store.StoreError as exc:
                        raise SessionError(f"{entry.get('op_id')} 尾部重放失败：{exc}") from exc
            last_after = entries[-1]['state_hash_after'] if entries else snapshot_hash
            if rebuilt.state_hash() != last_after:
                raise SessionError('尾部重放后的状态哈希与会话终态不符——日志可能被篡改')
            if snap.get('chain_head') != last_after:
                raise SessionError('meta.chain_head 与日志终态不符——元数据被篡改')
        else:
            # 全量重放路径（JSON 小会话；或 SQLite 首个检查点之前）
            _structural_check(entries, capacity)
            rebuilt = _rebuild(entries, capacity)
            last_after = entries[-1]['state_hash_after'] if entries else _empty_state_hash(capacity)
            if rebuilt.state_hash() != last_after:
                raise SessionError('重放后的状态哈希与会话终态不符——日志可能被篡改')
            # state 缓存仍是派生态：与重放结果比对（O(N) 字典比对，无哈希）
            if snap.get('state_records') is not None and snap['state_records'] != rebuilt.slots:
                raise SessionError('state 缓存与日志重放不符——缓存被篡改')

        return cls(path, capacity, entries, rebuilt, storage,
                   n_ops=snap.get('n_ops'))

    def close(self):
        if isinstance(self.storage, SqliteStorage):
            self.storage.close()

    # ---- 执行 ----

    def apply(self, op, *, category, source, utterance, max_attempts=8):
        """校验并执行一个结构化 op；无论成败都落日志、续哈希链。

        category 是 CLI 层的命令类别（teach/ask/correct），必须与 op
        类型一致——这是解析层与执行层之间的第二道锁。
        并发（SQLite 后端）：被抢先则重载重试，重试耗尽抛 SessionError。
        """
        for attempt in range(1, max_attempts + 1):
            entry, mutated = self._apply_once(op, category, source, utterance)
            try:
                self.storage.append_entry(entry, mutated, seq=len(self.entries) + 1)
            except StaleStateError:
                if attempt == max_attempts:
                    raise SessionError(f'并发冲突：重试 {max_attempts} 次未成功，请稍后重试')
                self._reload()
                continue
            except StorageError as exc:
                raise SessionError(f'持久化失败：{exc}') from exc
            self.entries.append(entry)
            return entry
        raise SessionError('apply：不可达状态')

    def _apply_once(self, op, category, source, utterance):
        """执行一次（可能被并发作废）：产出条目；写操作同时返回
        (name, record) 供缓存更新。"""
        allowed = CATEGORIES[category]
        op_id = f'op-{len(self.entries) + 1:03d}'
        entry = {'op_id': op_id, 'category': category, 'source': source,
                 'utterance': utterance, 'op': op, 'status': None,
                 'state_hash_before': self.store.state_hash()}
        mutated = None
        try:
            store.validate_op(op)
            if op['op'] not in allowed:
                raise store.StoreError(f"命令类别 {category} 不允许 op {op['op']}")
            if op['op'] in store.WRITE_OPS:
                cert = self.store.apply_write(op, op_id=op_id, utterance=utterance)
                entry['result'] = {'name': op['name'],
                                   'kind': 'rule' if op['op'].endswith('_rule') else 'fact',
                                   'revision': self.store.slots[op['name']]['revision'],
                                   'written_by': op_id}
                entry['proof'] = cert
                mutated = (op['name'], self.store.slots[op['name']])
            else:
                entry['result'] = self.store.read(op)
            entry['status'] = 'ok'
        except (store.StoreError, executor.ExecutorError, ValueError) as exc:
            entry['status'] = 'error'
            entry['error'] = str(exc)
        entry['state_hash_after'] = self.store.state_hash()
        return entry, mutated

    def _reload(self):
        """丢弃内存态，从一致快照重建（乐观并发冲突后的恢复路径）。

        与 load 相同的快照/尾部逻辑；仅在此处复核，避免两份漂移。"""
        fresh = self.load(self.path)
        self.entries = fresh.entries
        self.store = fresh.store
        self.n_ops = fresh.n_ops
        fresh.close()

    # ---- 报告 ----

    def terminal_state_hash(self):
        return self.store.state_hash()

    def ok_writes(self):
        return [e for e in self.entries
                if e.get('status') == 'ok' and e['op']['op'] in store.WRITE_OPS]

    def answers(self):
        """供 replay 比对：[(op_id, utterance, op, answer_or_error)]，按序。

        错误条目也计入（答案 = ERROR: 消息）——重放必须连错误都逐字
        复现，才算通过。
        """
        out = []
        for e in self.entries:
            if e.get('status') == 'ok' and e['op']['op'] in store.READ_OPS:
                out.append((e['op_id'], e['utterance'], e['op'], e['result']['answer']))
            elif e.get('status') == 'error':
                out.append((e['op_id'], e['utterance'], e['op'], f"ERROR: {e['error']}"))
        return out
