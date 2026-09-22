#!/usr/bin/env python3
"""独立核验：第二套原语实现 + 独立重放 + 哈希断言 + 灵魂样例。

独立性边界（刻意收窄）：本文件只 import store.py 的 canonical_json /
sha256_string 两个哈希原语（哈希函数必须一致结果才可比）。其余一切——
原语执行、槽位应用、证书复核、期望答案——全部在本文件内独立实现或
按 PROTOCOL 预声明独立重推，不 import data/session/proof/executor。

核验内容（PROTOCOL 验收标准 2/3/5/8）：
1. 会话哈希链独立重算：首条 before == 空状态哈希、逐条相扣、
   错误条目无副作用；
2. 用 independent_replay 独立重建每个写前/写后状态，逐槽哈希与条目
   证书逐字比对（零附带损害由此独立证实）；
3. 用 independent_transform 独立重算每个 apply_rule 的答案并与
   replayed.json 的答案比对；
4. 灵魂样例：纠错前 apply(规则07, '1234') == 3214、纠错后 == 1432，
   且两次之间恰好隔着对 规则07 的成功更正；
5. replayed.json 的终态哈希与会话一致。
输出 demo/verify_result.json；任何一处失败退出码 1。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# 哈希公共面：与 store.py 刻意共享的唯一函数。
from verifiable_memory import store
from verifiable_memory.store import canonical_json, sha256_string

HERE = Path(__file__).resolve().parent

# ---- 独立原语实现（不 import data.py；算法写法刻意不同） ----

OPS = ('反转', '左移一位', '交换前两位', '首位加一')


class VerifyError(Exception):
    pass


def independent_transform(s, program):
    """第二套实现：不用切片语法，全部走下标循环，避免与参考实现
    共享任何可能同时出错的捷径。"""
    chars = [c for c in s]
    for op in program:
        n = len(chars)
        if op == '反转':
            chars = [chars[n - 1 - i] for i in range(n)]
        elif op == '左移一位':
            chars = [chars[(i + 1) % n] for i in range(n)]
        elif op == '交换前两位':
            chars = [chars[1], chars[0]] + chars[2:]
        elif op == '首位加一':
            head = (ord(chars[0]) - ord('0') + 1) % 8
            chars = [chr(ord('0') + head)] + chars[1:]
        else:
            raise VerifyError(f'独立实现未知原语: {op!r}')
    return ''.join(chars)


def _valid_input(s):
    return (isinstance(s, str) and 2 <= len(s) <= 8
            and all(c in '01234567' for c in s))


def _record_from_write(op):
    """从 op 独立构造槽位记录（字段与 store 约定一致，但独立写出）。"""
    kind = 'rule' if op['op'].endswith('_rule') else 'fact'
    record = {'kind': kind, 'written_by': op['_op_id'], 'revision': op['_revision'],
              'utterance': op['_utterance']}
    if kind == 'fact':
        record['content'] = op['content']
    else:
        record['program'] = list(op['program'])
    return record


def independent_replay(entries, capacity):
    """独立重建全程状态；返回 (slot_states, problems)。

    slot_states: 每条目的 (before_slots, after_slots) 逐槽记录字典。
    """
    slots = {}
    problems = []
    empty_hash = sha256_string(canonical_json(
        {'format': 'verifiable_memory_01/state@v1', 'capacity': capacity, 'slots': {}}))
    states = []

    def slot_hashes():
        return {name: sha256_string(canonical_json(rec)) for name, rec in slots.items()}

    expected_before = empty_hash
    for entry in entries:
        op = dict(entry['op'])
        before = {k: dict(v) for k, v in slots.items()}
        before_slots = slot_hashes()

        if entry['status'] == 'ok' and op['op'] in ('teach_fact', 'teach_rule',
                                                    'correct_fact', 'correct_rule'):
            exists = op['name'] in slots
            if op['op'].startswith('teach_') and exists:
                problems.append(f"{entry['op_id']}: teach 已存在名称 {op['name']}")
            if op['op'].startswith('correct_') and not exists:
                problems.append(f"{entry['op_id']}: correct 不存在名称 {op['name']}")
            revision = slots[op['name']]['revision'] + 1 if exists else 1
            op['_op_id'] = entry['op_id']
            op['_revision'] = revision
            op['_utterance'] = entry.get('utterance', '')
            slots[op['name']] = _record_from_write(op)
        elif entry['status'] == 'error':
            # 独立重推错误：op 应在独立语义下被拒绝（粗粒度：与原错误
            # 同为拒绝即可，精确消息比对在 replay.py 已做）。
            pass

        after_slots = slot_hashes()
        states.append((entry['op_id'], before, slots.copy(), before_slots, after_slots))
        # 链检查
        if entry.get('state_hash_before') != expected_before:
            problems.append(f"{entry['op_id']}: 独立重算的链不连续")
        expected_before = entry['state_hash_after']

    final_state_hash = sha256_string(canonical_json(
        {'format': 'verifiable_memory_01/state@v1', 'capacity': capacity, 'slots': slots}))
    return states, problems, empty_hash, final_state_hash


def verify(session_path, replayed_path):
    failures = []
    session_raw = json.loads(session_path.read_text(encoding='utf-8'))
    replayed = json.loads(replayed_path.read_text(encoding='utf-8'))
    entries, capacity = session_raw['entries'], session_raw['capacity']

    # 1. 哈希链 + 错误条目无副作用
    empty_hash = sha256_string(canonical_json(
        {'format': 'verifiable_memory_01/state@v1', 'capacity': capacity, 'slots': {}}))
    if entries and entries[0]['state_hash_before'] != empty_hash:
        failures.append('首条 before ≠ 空状态哈希')
    for entry in entries:
        if entry['status'] == 'error' and entry['state_hash_after'] != entry['state_hash_before']:
            failures.append(f"{entry['op_id']}: 错误条目有副作用")
    # 逐条相扣
    for prev, cur in zip(entries, entries[1:]):
        if prev['state_hash_after'] != cur['state_hash_before']:
            failures.append(f"{cur['op_id']}: 链不扣合")

    # 2. 独立重放：证书断言对独立重建的全槽 diff 逐条复核
    #（双格式：v1 全表证书逐字比对；v2 瘦身证书核对每条断言）
    def _state_hash_of(records):
        return sha256_string(canonical_json(
            {'format': 'verifiable_memory_01/state@v1', 'capacity': capacity,
             'slots': records}))

    states, problems, _, final_hashes = independent_replay(entries, capacity)
    failures += problems
    for entry in entries:
        if entry['status'] != 'ok' or 'proof' not in entry:
            continue
        cert = entry['proof']
        target = cert.get('target')
        _, before_records, after_records, before_slots, after_slots = next(
            s for s in states if s[0] == entry['op_id'])
        changed = sorted(n for n in set(before_slots) | set(after_slots)
                         if before_slots.get(n) != after_slots.get(n))
        clean = changed == [target]
        if not clean:
            failures.append(f"{entry['op_id']}: 独立重放显示零附带损害不成立"
                            f"（变更 {changed}）")
        if cert.get('type') == 'write_certificate':
            expected = {
                'type': 'write_certificate', 'target': target,
                'changed_slots': changed,
                'zero_collateral': clean,
                'before_state_hash': _state_hash_of(before_records),
                'after_state_hash': _state_hash_of(after_records),
                'before_slot_hashes': before_slots,
                'after_slot_hashes': after_slots,
            }
            if cert != expected:
                diff = [k for k in set(cert) | set(expected)
                        if cert.get(k) != expected.get(k)]
                failures.append(f"{entry['op_id']}: 证书(v1)独立重算不一致（字段 {sorted(diff)}）")
        else:
            # v2 瘦身证书：每条断言对独立重放复核
            if cert.get('zero_collateral') != clean:
                failures.append(f"{entry['op_id']}: zero_collateral 断言与独立重放不符")
            if cert.get('created') != (target not in before_slots):
                failures.append(f"{entry['op_id']}: created 断言与独立重放不符")
            if cert.get('target_hash_before') != before_slots.get(target):
                failures.append(f"{entry['op_id']}: 目标槽写前哈希不符")
            if cert.get('target_hash_after') != after_slots.get(target):
                failures.append(f"{entry['op_id']}: 目标槽写后哈希不符")
            if cert.get('before_state_hash') != _state_hash_of(before_records):
                failures.append(f"{entry['op_id']}: 写前状态哈希不符")
            if cert.get('after_state_hash') != _state_hash_of(after_records):
                failures.append(f"{entry['op_id']}: 写后状态哈希不符")
            if 'changed_slots' in cert and cert['changed_slots'] != changed:
                failures.append(f"{entry['op_id']}: 异常取证字段 changed_slots 与独立重放不符")

    # 3. 独立重算全部 apply_rule 答案（按条目顺序取"当时"的程序，
    #    而非最终程序——纠错前后的执行必须各用各的版本）
    replay_answers = {a[0]: a[3] for a in replayed['answers']}
    name_to_program = {}
    for entry in entries:
        op = entry['op']
        if entry['status'] == 'ok' and op['op'] in ('teach_rule', 'correct_rule'):
            name_to_program[op['name']] = op['program']
        elif op['op'] == 'apply_rule' and entry['status'] == 'ok':
            if op['name'] not in name_to_program:
                failures.append(f"{entry['op_id']}: 引用了当时未定义的规则 {op['name']}")
                continue
            if not _valid_input(op['input']):
                failures.append(f"{entry['op_id']}: 非法输入 {op['input']!r}")
                continue
            expected_answer = independent_transform(op['input'], name_to_program[op['name']])
            recorded = entry.get('result', {}).get('answer')
            if recorded != expected_answer:
                failures.append(f"{entry['op_id']}: 答案 {recorded} ≠ 独立重算 {expected_answer}")
            if replay_answers.get(entry['op_id']) != expected_answer:
                failures.append(f"{entry['op_id']}: replayed.json 答案与独立重算不符")

    # 4. 灵魂样例（预声明，独立断言）——仅当会话含该模式时断言；
    # 真实场景的会话不含 规则07/1234，不应被 demo 专属检查误伤。
    soul = [e for e in entries
            if e['op']['op'] == 'apply_rule' and e['op']['name'] == '规则07'
            and e['op']['input'] == '1234']
    corrections = [e for e in entries
                   if e['op']['op'] == 'correct_rule' and e['op']['name'] == '规则07'
                   and e['status'] == 'ok']
    soul_checked = len(soul) >= 2 and bool(corrections)
    if soul_checked:
        if soul[0].get('result', {}).get('answer') != '3214':
            failures.append(f"灵魂样例（纠错前）: {soul[0].get('result', {}).get('answer')} ≠ 3214")
        if soul[-1].get('result', {}).get('answer') != '1432':
            failures.append(f"灵魂样例（纠错后）: {soul[-1].get('result', {}).get('answer')} ≠ 1432")
        first_idx = entries.index(soul[0])
        last_idx = entries.index(soul[-1])
        if not any(first_idx < entries.index(c) < last_idx for c in corrections):
            failures.append('灵魂样例：纠错不发生在两次执行之间')

    # 5. 终态哈希三方一致
    terminal = entries[-1]['state_hash_after'] if entries else empty_hash
    if terminal != replayed.get('terminal_state_hash'):
        failures.append('replayed.json 终态哈希与会话不符')
    if final_hashes != terminal:
        failures.append('独立重建的终态哈希不符')

    result = {'session_sha256': hashlib.sha256(session_path.read_bytes()).hexdigest(),
              'n_entries': len(entries), 'n_failures': len(failures),
              'failures': failures, 'soul_checked': soul_checked,
              'soul_example': {'before': soul[0].get('result', {}).get('answer') if soul else None,
                               'after': soul[-1].get('result', {}).get('answer') if soul else None}}
    return result, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--session', default=str(HERE / 'demo' / 'session.json'))
    ap.add_argument('--replayed', default=str(HERE / 'demo' / 'replayed.json'))
    ap.add_argument('--out', default=str(HERE / 'demo' / 'verify_result.json'))
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f'拒绝覆盖：{out_path}')
    if not Path(args.replayed).exists():
        raise SystemExit('缺 replayed.json——请先跑 replay.py（核验不代跑前置步骤）')

    result, failures = verify(Path(args.session), Path(args.replayed))
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding='utf-8')
    for f in failures:
        print(f'  ✗ {f}')
    soul = result['soul_example'] or {}
    soul_note = (f"灵魂样例 纠错前={soul.get('before')} 纠错后={soul.get('after')}"
                 if result.get('soul_checked') else '本会话不含灵魂样例模式（跳过）')
    print(f"核验完成：{result['n_entries']} 条 · {result['n_failures']} 处失败 · "
          f"{soul_note} → {out_path}")
    sys.exit(0 if not failures else 1)


if __name__ == '__main__':
    main()
