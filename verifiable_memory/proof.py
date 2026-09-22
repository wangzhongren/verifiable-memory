"""证据与证明：零附带损害证书（v2 瘦身格式）。

v2（2026-09-22，为 32k 槽规模设计）：证书只存目标槽前后哈希与状态
哈希，O(1) 大小——v1 的全槽哈希表是 O(N)/写，32k 槽下每次写入膨胀
数 MB，不可行。

代价是证书不再"自带全表证明"：zero_collateral 变成**写入时的断言**
（由代码路径构造保证：apply_write 只赋值目标槽一个键）+ **离线的
独立证实**（verify.py 用第二套实现重放重建全槽 diff，对断言逐条
复核）。断言与证实的分工写进了证书格式本身。

v1（全表格式）仅存在于旧证据文件：verify.py / replay.py 双格式兼
容读取，新写入一律 v2。
"""


def write_certificate(*, target, created, zero_collateral, changed_slots,
                      target_hash_before, target_hash_after,
                      before_state_hash, after_state_hash):
    """构造 v2 证书。changed_slots 仅在断言不成立时落盘（异常取证）。"""
    cert = {
        'type': 'write_certificate_v2',
        'target': target,
        'created': created,
        'zero_collateral': zero_collateral,
        'target_hash_before': target_hash_before,
        'target_hash_after': target_hash_after,
        'before_state_hash': before_state_hash,
        'after_state_hash': after_state_hash,
    }
    if not zero_collateral:
        cert['changed_slots'] = sorted(changed_slots)
    return cert


def summarize_certificate(cert):
    """给人看的一行摘要（CLI/转录用）。双格式兼容。"""
    flag = '是' if cert.get('zero_collateral') else '否'
    if cert.get('type') == 'write_certificate':
        changed = '、'.join(cert['changed_slots'])
    else:
        changed = cert['target']
    return (f"零附带损害: {flag} · 变更槽: {changed}"
            f" · 状态 {str(cert.get('before_state_hash'))[:12]}"
            f"→{str(cert.get('after_state_hash'))[:12]}")
