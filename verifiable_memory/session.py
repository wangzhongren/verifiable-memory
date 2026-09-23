"""会话编排：op 日志 + 哈希链 + 持久化（后端可插拔）。

日志是唯一事实来源；state 表/文件快照皆为派生缓存，load 时必须
核对检查点、尾部日志与当前缓存；历史段的全量审计由离线核验负责。

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
from .audit import AuditError, check_log, entry_digest, log_genesis
from .storage import (JsonStorage, SQLITE_SUFFIXES,
                     SqliteStorage, StaleStateError, StorageError)
from .store import CAPACITY_DEFAULT, state_digest

CATEGORIES = {
    'teach': {'teach_fact', 'teach_rule'},
    'correct': {'correct_fact', 'correct_rule'},
    'ask': store.READ_OPS,
    'vector': {'teach_entity', 'correct_entity', 'teach_vector_action',
               'correct_vector_action', 'link_entities',
               'apply_vector_action', 'derive_entities', 'query_record'},
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


class Session:
    """一次连续教学/问答会话。所有状态变化都留下日志与哈希链。

    SQLite 后端下 self.entries 只持有**尾部**（最近检查点之后的条
    目）——快照部分由 checkpoint_state 表 O(N) 重建、用哈希锚定；
    全量重放核验是 replay.py/verify.py 的离线职责，不在每条命令的
    热路径上。self.n_ops 为日志总条数（无检查点概念的后端 = 尾部长
    度）。"""

    def __init__(self, path, capacity, entries, store_obj, storage, n_ops=None,
                 log_head=None):
        self.path = Path(path)
        self.capacity = capacity
        self.entries = entries
        self.store = store_obj
        self.storage = storage
        self.n_ops = n_ops if n_ops is not None else len(entries)
        self.log_head = log_head if log_head is not None else log_genesis(capacity)

    # ---- 生命周期 ----

    @classmethod
    def create(cls, path, capacity=CAPACITY_DEFAULT):
        path = Path(path)
        if path.exists():
            raise SessionError(f'会话已存在，拒绝覆盖：{path}（如需重开请先 reset）')
        store_obj = store.Store(capacity)
        if path.suffix in SQLITE_SUFFIXES:
            storage = SqliteStorage.create(path, capacity)
        else:
            storage = JsonStorage.create(path, capacity)
        return cls(path, capacity, [], store_obj, storage)

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
            return cls._from_snapshot(path, storage, storage.snapshot())
        except (StorageError, AuditError, store.StoreError) as exc:
            if isinstance(storage, SqliteStorage):
                storage.close()
            raise SessionError(str(exc)) from exc
        except Exception:
            if isinstance(storage, SqliteStorage):
                storage.close()
            raise

    @classmethod
    def _from_snapshot(cls, path, storage, snap):
        capacity, entries = snap['capacity'], snap['entries']
        if type(capacity) is not int or capacity < 1 or not isinstance(entries, list):
            raise SessionError('会话 capacity / entries 非法')
        n_ops, snapshot_seq = snap['n_ops'], snap.get('snapshot_seq', 0)
        if (type(n_ops) is not int or type(snapshot_seq) is not int
                or not 0 <= snapshot_seq <= n_ops
                or snapshot_seq + len(entries) != n_ops):
            raise SessionError('日志条数或检查点序号不符')

        rebuilt = store.Store(capacity)
        anchor = _empty_state_hash(capacity)
        log_anchor = log_genesis(capacity)
        if snapshot_seq:
            # 检查点实体独立于最新 state 缓存，只在检查点事务中更新。
            rebuilt.slots.update(snap['checkpoint_records'])
            anchor = snap['snapshot_hash']
            log_anchor = snap['snapshot_log_hash']
            if rebuilt.state_hash() != anchor:
                raise SessionError('checkpoint_state 与 snapshot_hash 不符')
        elif snap.get('checkpoint_records'):
            raise SessionError('零序号检查点应为空')

        log_head = check_log(entries, capacity, anchor=log_anchor,
                             start_seq=snapshot_seq)
        if log_head != snap['log_head']:
            raise SessionError('log_head 与日志终态不符')
        _structural_check(entries, capacity, anchor=anchor)
        for entry in entries:
            if entry['status'] == 'ok' and entry['op']['op'] in store.WRITE_OPS:
                store.validate_op(entry['op'])
                rebuilt.apply_write(entry['op'], op_id=entry['op_id'],
                                    utterance=entry['utterance'])
            if rebuilt.state_hash() != entry['state_hash_after']:
                raise SessionError(f"{entry['op_id']}: 重放后的状态哈希不符")
        if snap.get('chain_head') is not None and snap['chain_head'] != rebuilt.state_hash():
            raise SessionError('meta.chain_head 与日志终态不符')
        if snap.get('state_records') is not None and snap['state_records'] != rebuilt.slots:
            raise SessionError('state 缓存与日志重放不符——缓存被篡改')
        return cls(path, capacity, entries, rebuilt, storage,
                   n_ops=n_ops, log_head=log_head)

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
            before_slots = self.store.slots.copy()
            try:
                entry, mutated = self._apply_once(op, category, source, utterance)
                self.storage.append_entry(entry, mutated, seq=self.n_ops + 1)
            except StaleStateError:
                self.store.slots = before_slots
                self._reload()
                if attempt == max_attempts:
                    raise SessionError(f'并发冲突：重试 {max_attempts} 次未成功，请稍后重试')
                continue
            except StorageError as exc:
                self.store.slots = before_slots
                raise SessionError(f'持久化失败：{exc}') from exc
            except Exception:
                self.store.slots = before_slots
                raise
            self.n_ops += 1
            self.log_head = entry['entry_hash']
            self.entries.append(entry)
            return entry
        raise SessionError('apply：不可达状态')

    def _apply_once(self, op, category, source, utterance):
        """执行一次（可能被并发作废）：产出条目；写操作同时返回
        (name, record) 供缓存更新。"""
        allowed = CATEGORIES[category]
        op_id = f'op-{self.n_ops + 1:03d}'
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
                record = self.store.slots[op['name']]
                entry['result'] = {'name': op['name'], 'kind': record['kind'],
                                   'revision': record['revision'],
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
        entry['prev_entry_hash'] = self.log_head
        entry['entry_hash'] = entry_digest(entry)
        return entry, mutated

    def _reload(self):
        """丢弃内存态，从一致快照重建（乐观并发冲突后的恢复路径）。

        与 load 相同的快照/尾部逻辑；仅在此处复核，避免两份漂移。"""
        fresh = self.load(self.path)
        self.entries = fresh.entries
        self.store = fresh.store
        self.n_ops = fresh.n_ops
        self.log_head = fresh.log_head
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
