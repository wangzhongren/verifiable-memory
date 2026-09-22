#!/usr/bin/env python3
"""独立进程重放：只读会话 op 日志，重建状态与全部输出。

重放器刻意"不知情"：不读 PROTOCOL 的期望答案，不看 CLI 的打印。
它只拿 entries 里的 op 逐个重跑，把结果与原条目逐字比对：

- 写操作：重建写前/写后逐槽哈希，必须与条目内证书完全一致；
- 读操作：重跑查询/执行，answer 与证据字段必须逐字一致；
- 错误条目：必须以同样的 StoreError 消息失败，且状态无副作用；
- 哈希链：首条 before == 空状态哈希，逐条相扣。

输出 demo/replayed.json + 终态哈希，供 verify.py 二次核验。
重放器允许 import store.py（确定性重放正是本系统的卖点），但独立
核验（verify.py）不得依赖 store 的执行逻辑。
"""

import argparse
import json
import sys
from pathlib import Path

from verifiable_memory import store
from verifiable_memory.audit import AuditError, check_evidence, check_log

HERE = Path(__file__).resolve().parent


def replay(entries, capacity):
    """重放 op 日志，返回 (checks, answers, terminal_hash)。

    checks: 每条的重放结论；answers: 供比对的答案序列。
    任何不一致都记为 failure 并继续（产出完整差异报告后统一失败）。
    """
    current = store.Store(capacity)
    empty_hash = current.state_hash()
    checks = []
    answers = []
    failures = 0
    expected_before = empty_hash

    try:
        check_log(entries, capacity)
    except AuditError as exc:
        return ([{'op_id': 'log', 'kind': 'error', 'ok': False,
                  'checks': [str(exc)]}], [], empty_hash, 1)

    for entry in entries:
        op = entry['op']
        note = {'op_id': entry['op_id'], 'kind': entry['status'], 'checks': []}
        ok = True

        # 哈希链扣合
        if entry.get('state_hash_before') != expected_before:
            note['checks'].append(f'链断裂: before {entry.get("state_hash_before")[:12]}'
                                  f' ≠ 期望 {expected_before[:12]}')
            ok = False

        try:
            store.validate_op(op)
            allowed = {'teach': {'teach_fact', 'teach_rule'},
                       'correct': {'correct_fact', 'correct_rule'},
                       'ask': {'query_record', 'apply_rule'}}
            if op['op'] not in allowed.get(entry['category'], set()):
                raise store.StoreError(
                    f"命令类别 {entry['category']} 不允许 op {op['op']}")
            if op['op'] in store.WRITE_OPS:
                if entry.get('status') == 'error':
                    # 错误写条目：重放必须以同样的消息失败（比对在外壳 except）
                    current.apply_write(op, op_id=entry['op_id'],
                                        utterance=entry.get('utterance', ''))
                    note['checks'].append('原条目为错误，但重放成功了')
                    ok = False
                else:
                    before_slots = current.slot_hashes()
                    before_hash = current.state_hash()
                    cert = entry.get('proof')
                    current.apply_write(op, op_id=entry['op_id'],
                                        utterance=entry.get('utterance', ''))
                    after_slots = current.slot_hashes()
                    after_hash = current.state_hash()
                    changed = sorted(n for n in set(before_slots) | set(after_slots)
                                     if before_slots.get(n) != after_slots.get(n))
                    if cert is not None and cert.get('type') == 'write_certificate':
                        # v1 全表格式（旧证据）：逐字比对含全槽哈希表
                        expected_cert = {
                            'type': 'write_certificate', 'target': op['name'],
                            'changed_slots': changed,
                            'zero_collateral': changed == [op['name']],
                            'before_state_hash': before_hash,
                            'after_state_hash': after_hash,
                            'before_slot_hashes': before_slots,
                            'after_slot_hashes': after_slots,
                        }
                    else:
                        # v2 瘦身格式
                        from verifiable_memory import proof
                        expected_cert = proof.write_certificate(
                            target=op['name'], created=(op['name'] not in before_slots),
                            zero_collateral=(changed == [op['name']]),
                            changed_slots=changed,
                            target_hash_before=before_slots.get(op['name']),
                            target_hash_after=after_slots.get(op['name']),
                            before_state_hash=before_hash,
                            after_state_hash=after_hash)
                    if cert != expected_cert:
                        diff = [k for k in set(cert) | set(expected_cert)
                                if cert.get(k) != expected_cert.get(k)]
                        note['checks'].append(f'证书不一致（字段 {sorted(diff)}）')
                        ok = False
                    else:
                        note['checks'].append('证书逐字一致')
                answers.append((entry['op_id'], entry['utterance'], op, None))
            else:
                if entry.get('status') == 'error':
                    try:
                        current.read(op)
                    except store.StoreError as exc:
                        if str(exc) != entry.get('error'):
                            note['checks'].append(f'错误消息不一致: {exc}')
                            ok = False
                        else:
                            note['checks'].append('错误消息逐字一致')
                    else:
                        note['checks'].append('原条目为错误，但重放成功了')
                        ok = False
                else:
                    rerun = current.read(op)
                    original = entry.get('result', {})
                    if rerun != original:
                        diff = [k for k in set(rerun) | set(original)
                                if rerun.get(k) != original.get(k)]
                        note['checks'].append(f'读结果不一致（字段 {sorted(diff)}）')
                        ok = False
                    else:
                        note['checks'].append('读结果与证据逐字一致')
                    answers.append((entry['op_id'], entry['utterance'], op,
                                    rerun.get('answer')))
        except store.StoreError as exc:
            if entry.get('status') == 'error':
                if str(exc) != entry.get('error'):
                    note['checks'].append(f'错误消息不一致: {exc}')
                    ok = False
                else:
                    note['checks'].append('错误消息逐字一致')
            else:
                note['checks'].append(f'原条目成功但重放失败: {exc}')
                ok = False

        # 状态一致性：错误必须无副作用；成功写后的哈希必须吻合
        if entry.get('status') == 'error':
            if current.state_hash() != expected_before:
                note['checks'].append('错误操作留下了副作用')
                ok = False
        else:
            if current.state_hash() != entry.get('state_hash_after'):
                note['checks'].append(f"after 哈希不符: 重放 {current.state_hash()[:12]}"
                                      f" ≠ 记录 {str(entry.get('state_hash_after'))[:12]}")
                ok = False
            expected_before = entry['state_hash_after']

        note['ok'] = ok
        failures += 0 if ok else 1
        checks.append(note)

    return checks, answers, current.state_hash(), failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--session', default=str(HERE / 'demo' / 'session.json'))
    ap.add_argument('--out', default=str(HERE / 'demo' / 'replayed.json'))
    args = ap.parse_args()

    session_path = Path(args.session)
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f'拒绝覆盖：{out_path}')

    raw = json.loads(session_path.read_text(encoding='utf-8'))
    try:
        check_evidence(raw)
    except AuditError as exc:
        raise SystemExit(f'证据完整性失败：{exc}') from exc
    entries, capacity = raw['entries'], raw['capacity']

    checks, answers, terminal_hash, failures = replay(entries, capacity)

    payload = {'source_session_sha256': __import__('hashlib').sha256(
        session_path.read_bytes()).hexdigest(),
        'terminal_state_hash': terminal_hash,
        'n_entries': len(entries), 'n_failures': failures,
        'answers': [list(a) for a in answers], 'checks': checks}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding='utf-8')

    for note in checks:
        mark = '✓' if note['ok'] else '✗'
        print(f"  {mark} {note['op_id']} ({note['kind']}): "
              f"{'；'.join(note['checks'])}")
    print(f"重放完成：{len(entries)} 条 · {failures} 条不一致 · "
          f"终态哈希 {terminal_hash[:12]} → {out_path}")
    sys.exit(0 if failures == 0 else 1)


if __name__ == '__main__':
    main()
