"""v2 日志完整性：状态哈希之外，对每条完整事件串联 SHA-256。

没有外部可信锚时，无法阻止攻击者重写整条链及其元数据。
verify.py 刻意不复用本模块，以独立实现核验同一格式。
"""

from .store import state_digest

SESSION_FORMAT = 'verifiable_memory_01/session@v2'
DB_FORMAT = 'verifiable_memory_01/sqlite@v2'


class AuditError(ValueError):
    pass


def log_genesis(capacity):
    return state_digest({'format': 'verifiable_memory_01/log@v2',
                         'capacity': capacity})


def entry_digest(entry):
    return state_digest({k: v for k, v in entry.items() if k != 'entry_hash'})


def check_log(entries, capacity, *, anchor=None, start_seq=0):
    """验证完整日志或检查点尾部，返回最终事件哈希。"""
    previous = log_genesis(capacity) if anchor is None else anchor
    for seq, entry in enumerate(entries, start_seq + 1):
        if not isinstance(entry, dict) or entry.get('op_id') != f'op-{seq:03d}':
            raise AuditError(f'日志序号不连续：期望 op-{seq:03d}')
        if entry.get('prev_entry_hash') != previous:
            raise AuditError(f'{entry["op_id"]}: 日志前驱哈希不符')
        if entry.get('entry_hash') != entry_digest(entry):
            raise AuditError(f'{entry["op_id"]}: 完整日志条目哈希不符')
        previous = entry['entry_hash']
    return previous


def check_evidence(raw):
    if raw.get('format') != SESSION_FORMAT:
        raise AuditError('需要 session@v2 证据；旧 v1 文件请保留并用旧版本核验，'
                         '不得通过改格式标记升级')
    capacity, entries = raw.get('capacity'), raw.get('entries')
    if type(capacity) is not int or capacity < 1 or not isinstance(entries, list):
        raise AuditError('证据 capacity / entries 非法')
    if type(raw.get('n_ops')) is not int or raw['n_ops'] != len(entries):
        raise AuditError('证据日志条数与 n_ops 不符')
    head = check_log(entries, capacity)
    if raw.get('log_head') != head:
        raise AuditError('证据 log_head 与完整日志不符')
    return head
