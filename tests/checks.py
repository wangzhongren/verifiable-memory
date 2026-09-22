#!/usr/bin/env python3
"""单元自检 + 端到端管线冒烟（子进程真跑 replay/verify）。

覆盖 PROTOCOL 验收的可单测部分：
- 参考实现断言（data.design_checks）与独立实现一致性；
- store 全部错误路径（teach 已存在 / correct 缺失 / 种类不符 / 槽满 /
  白名单拒绝）；
- 证书零附带损害（真/假两例）；
- 后备文法解析全部 SCENARIO 话语 == 预声明 op；
- 会话哈希链、重载一致性、篡改检测；
- 跨进程管线：临时目录里 teach→ask→correct→replay→verify 真进程跑通。
"""

import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifiable_memory import data, executor, proof, store
from verifiable_memory import parser as parser_module
from verifiable_memory import session as session_module
from verifiable_memory.storage import export_evidence

HERE = ROOT
FAILURES = []


def check(name, condition, detail=''):
    if condition:
        print(f'  ✓ {name}')
    else:
        FAILURES.append(name)
        print(f'  ✗ {name} {detail}')


def expect_store_error(name, fn):
    try:
        fn()
    except store.StoreError as exc:
        check(name, True)
        return str(exc)
    except Exception as exc:  # noqa: BLE001
        check(name, False, f'抛了别的异常: {type(exc).__name__}: {exc}')
        return ''
    check(name, False, '没有抛 StoreError')
    return ''


def test_primitives():
    print('原语实现：')
    data.design_checks()
    check('参考实现 design_checks 全过', True)
    ok = all(
        data.transform(s, p) == __import__('verify').independent_transform(s, p)
        for s in ('1234', '5670', '7777', '07', '6543210')
        for p in ([data.OPS[i]] for i in range(4)))
    check('独立实现与参考实现全原语一致', ok)


def test_store():
    print('store 错误路径：')
    expect_store_error('拒绝未知 op', lambda: store.validate_op({'op': 'nope'}))
    expect_store_error('拒绝非法程序词',
                       lambda: store.validate_op(data.teach_rule('规则X', ['反转', '乱写'])))
    expect_store_error('拒绝非法输入',
                       lambda: store.validate_op(data.apply_rule('规则X', '89')))
    s = store.Store(2)
    s.apply_write(data.teach_fact('实体07', '方向是向上'), op_id='op-001', utterance='u')
    expect_store_error('拒绝 teach 已存在名称',
                       lambda: s.apply_write(data.teach_fact('实体07', 'x'),
                                             op_id='op-002', utterance='u'))
    expect_store_error('拒绝 correct 缺失名称',
                       lambda: s.apply_write(data.correct_fact('实体99', 'x'),
                                             op_id='op-002', utterance='u'))
    expect_store_error('拒绝种类不符',
                       lambda: s.apply_write(data.correct_rule('实体07', ['反转']),
                                             op_id='op-002', utterance='u'))
    s.apply_write(data.teach_fact('实体11', '方向是向左'), op_id='op-002', utterance='u')
    expect_store_error('拒绝槽满', lambda: s.apply_write(
        data.teach_fact('实体12', 'x'), op_id='op-003', utterance='u'))


def test_proof_and_executor():
    print('证据与执行器：')
    s = store.Store()
    s.apply_write(data.teach_fact('实体07', '方向是向上'), op_id='op-001', utterance='u')
    s.apply_write(data.teach_fact('实体11', '方向是向左'), op_id='op-002', utterance='u')
    cert = s.apply_write(data.correct_fact('实体11', '方向是向下'),
                         op_id='op-003', utterance='u')
    check('v2 证书 O(1) 大小（不含全槽表）',
          cert['type'] == 'write_certificate_v2'
          and 'before_slot_hashes' not in cert and len(json.dumps(cert)) < 500)
    check('纠错证书断言齐全',
          cert['zero_collateral'] is True and cert['created'] is False
          and cert['target_hash_before'] == store.state_digest(
              {'kind': 'fact', 'content': '方向是向左', 'written_by': 'op-002',
               'revision': 1, 'utterance': 'u'})
          and cert['target_hash_after'] == store.state_digest(s.slots['实体11']))
    s2 = store.Store()
    cert2 = s2.apply_write(data.teach_fact('实体07', 'x'), op_id='op-001', utterance='u')
    check('teach 证书 created=True 且无写前哈希',
          cert2['created'] is True and cert2['target_hash_before'] is None)

    run = executor.execute(['反转', '左移一位'], '1234')
    check('执行器带逐步 trace',
          run['answer'] == '3214' and run['trace'][0]['after'] == '4321'
          and run['trace'][1]['after'] == '3214')
    try:
        executor.execute(['反转'], '9')
        check('执行器拒绝非法输入', False)
    except executor.ExecutorError:
        check('执行器拒绝非法输入', True)


def test_parser_fallback():
    print('后备文法：')
    ok = True
    for _category, utterance, expected_op, _answer in data.SCENARIO:
        got = parser_module.parse_fallback(utterance)
        if got != expected_op:
            ok = False
            print(f'    解析不符: {utterance!r} → {got}')
    check('SCENARIO 全部话语解析 == 预声明 op', ok)
    check('经典句式（对N应用M）',
          parser_module.parse_fallback('对1234应用规则07。') == data.apply_rule('规则07', '1234'))
    try:
        parser_module.parse_fallback('随便说点什么')
        check('无法解析时报 ParserError', False)
    except parser_module.ParserError:
        check('无法解析时报 ParserError', True)


def test_session(tmp):
    print('会话与哈希链：')
    path = Path(tmp) / 'session.json'
    s = session_module.Session.create(path)
    for category, utterance, op, _answer in data.SCENARIO[:14]:
        s.apply(op, category=category, source='fallback', utterance=utterance)
    chain_ok = all(s.entries[i]['state_hash_after'] == s.entries[i + 1]['state_hash_before']
                   for i in range(len(s.entries) - 1))
    check('哈希链逐条相扣', chain_ok)
    r = session_module.Session.load(path)
    check('重载后终态一致', r.terminal_state_hash() == s.terminal_state_hash())
    # 篡改检测
    raw = json.loads(path.read_text(encoding='utf-8'))
    raw['entries'][0]['op']['content'] = '被篡改的内容'
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
    try:
        session_module.Session.load(path)
        check('篡改被检测', False)
    except session_module.SessionError:
        check('篡改被检测', True)


def test_sqlite(tmp):
    print('SQLite 后端：')
    db = Path(tmp) / 'memory.db'
    s = session_module.Session.create(db)
    for category, utterance, op, _answer in data.SCENARIO[:14]:
        s.apply(op, category=category, source='fallback', utterance=utterance)
    r = session_module.Session.load(db)
    check('sqlite 重载终态一致', r.terminal_state_hash() == s.terminal_state_hash())
    check('sqlite state 缓存与重放一致', r.store.slots == s.store.slots)
    # 派生缓存篡改（state 表被手改 → load 时"重放 vs 缓存"核对必须抓住）
    conn = sqlite3.connect(str(db))
    fake = json.dumps({'kind': 'fact', 'content': '被篡改', 'written_by': 'op-001',
                       'revision': 1, 'utterance': 'x'}, ensure_ascii=False)
    conn.execute("UPDATE state SET record = ? WHERE name = '实体07'", (fake,))
    conn.commit()
    conn.close()
    try:
        session_module.Session.load(db)
        check('state 缓存篡改被检测', False)
    except session_module.SessionError:
        check('state 缓存篡改被检测', True)
    # 日志篡改（ops 行被手改 → 重放终态不符必须抓住）
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE ops SET op = replace(op, '方向是向上', '方向是篡改')"
                 " WHERE op_id = 'op-001'")
    conn.commit()
    conn.close()
    try:
        session_module.Session.load(db)
        check('ops 日志篡改被检测', False)
    except session_module.SessionError:
        check('ops 日志篡改被检测', True)


def test_evidence_equivalence(tmp):
    print('证据等价性与管线（SQLite 导出 = JSON 原生，逐字节）：')
    json_path = Path(tmp) / 'eq.json'
    db_path = Path(tmp) / 'eq.db'
    exported = Path(tmp) / 'eq_exported.json'
    sj = session_module.Session.create(json_path)
    sd = session_module.Session.create(db_path)
    for category, utterance, op, _answer in data.SCENARIO:
        sj.apply(op, category=category, source='fallback', utterance=utterance)
        sd.apply(op, category=category, source='fallback', utterance=utterance)
    export_evidence(db_path, exported)
    check('证据逐字节等价（JSON 后端 == SQLite 导出）',
          json_path.read_bytes() == exported.read_bytes())
    r5 = subprocess.run([sys.executable, str(HERE / 'replay.py'),
                         '--session', str(exported), '--out', str(Path(tmp) / 'rep.json')],
                        capture_output=True, text=True, cwd=str(HERE))
    check('SQLite 导出证据 → 独立重放通过', r5.returncode == 0, r5.stdout + r5.stderr)
    r6 = subprocess.run([sys.executable, str(HERE / 'verify.py'),
                         '--session', str(exported), '--replayed', str(Path(tmp) / 'rep.json'),
                         '--out', str(Path(tmp) / 'ver.json')],
                        capture_output=True, text=True, cwd=str(HERE))
    check('SQLite 导出证据 → 独立核验通过', r6.returncode == 0, r6.stdout + r6.stderr)


def test_checkpoint(tmp):
    print('快照+尾部重放（SQLite，间隔压到 3）：')
    from verifiable_memory import storage as storage_module
    db = Path(tmp) / 'cp.db'
    old_interval = storage_module.CHECKPOINT_INTERVAL
    storage_module.CHECKPOINT_INTERVAL = 3
    try:
        s = session_module.Session.create(db)
        for category, utterance, op, _answer in data.SCENARIO[:10]:
            s.apply(op, category=category, source='fallback', utterance=utterance)
        # 10 条 op，检查点应落在 3/6/9 → 快照锚定于 op-009，尾部仅 1 条
        r = session_module.Session.load(db)
        check('检查点后重载终态一致', r.terminal_state_hash() == s.terminal_state_hash())
        check('载入的是尾部而非全量', len(r.entries) == 1 and r.n_ops == 10,
              f'实际尾部 {len(r.entries)} 条 · n_ops={r.n_ops}')
        # 快照段内 ops 行篡改：load 有意放行（工作路径不背 O(M×N)），
        # 由离线 replay 全量重放负责抓住——这是架构分工，不是漏洞
        conn = sqlite3.connect(str(db))
        conn.execute("UPDATE ops SET op = replace(op, '方向是向上', '方向是篡改')"
                     " WHERE op_id = 'op-001'")
        conn.commit()
        conn.close()
        try:
            session_module.Session.load(db)
            check('快照段内日志篡改由 load 放行（设计分工）', True)
        except session_module.SessionError as exc:
            check('快照段内日志篡改由 load 放行（设计分工）', False, str(exc)[:120])
        out = Path(tmp) / 'cp_export.json'
        export_evidence(db, out)
        r5 = subprocess.run([sys.executable, str(HERE / 'replay.py'), '--session', str(out),
                             '--out', str(Path(tmp) / 'cp_rep.json')],
                            capture_output=True, text=True, cwd=str(HERE))
        check('快照段内篡改被离线重放抓住', r5.returncode != 0)
        # 快照缓存（state 表）篡改：load 的 snapshot_hash 锚必须抓住
        conn = sqlite3.connect(str(db))
        conn.execute("UPDATE state SET record = replace(record, '向上', '篡改')"
                     " WHERE name = '实体07'")
        conn.commit()
        conn.close()
        try:
            session_module.Session.load(db)
            check('快照缓存篡改被 snapshot_hash 锚抓住', False)
        except session_module.SessionError:
            check('快照缓存篡改被 snapshot_hash 锚抓住', True)
    finally:
        storage_module.CHECKPOINT_INTERVAL = old_interval


def test_import(tmp):
    print('批量导入：')
    db = Path(tmp) / 'imp.db'
    f = Path(tmp) / 'k.jsonl'
    f.write_text('\n'.join([
        json.dumps({'kind': 'fact', 'name': '事实甲', 'content': '内容一'}, ensure_ascii=False),
        json.dumps({'kind': 'fact', 'name': '事实乙', 'content': '内容二'}, ensure_ascii=False),
        json.dumps({'kind': 'rule', 'name': '规则甲', 'program': ['反转', '首位加一']},
                   ensure_ascii=False),
        '{"坏 json",',
        json.dumps({'kind': 'fact', 'name': '坏程序', 'program': ['乱写']}, ensure_ascii=False),
        json.dumps({'kind': 'fact', 'name': '事实甲', 'content': '重复'}, ensure_ascii=False),
    ]), encoding='utf-8')
    r = subprocess.run([sys.executable, str(HERE / 'cli.py'), '--session', str(db),
                        'import', str(f)], capture_output=True, text=True, cwd=str(HERE))
    check('导入统计：新增3·跳过1·失败2，退出码1',
          '新增 3' in r.stdout and '跳过 1' in r.stdout and '失败 2' in r.stdout
          and r.returncode == 1, r.stdout)
    r2 = subprocess.run([sys.executable, str(HERE / 'cli.py'), '--session', str(db),
                         'ask', '对35应用规则甲'], capture_output=True, text=True, cwd=str(HERE))
    check('导入的规则可执行（35→反转→53→首位加一→63）',
          r2.returncode == 0 and '63' in r2.stdout, r2.stdout)
    r3 = subprocess.run([sys.executable, str(HERE / 'cli.py'), '--session', str(db),
                         'import', str(f)], capture_output=True, text=True, cwd=str(HERE))
    check('重复导入幂等（跳过已存在）',
          '新增 0' in r3.stdout and '跳过 4' in r3.stdout, r3.stdout)
    s = session_module.Session.load(db)
    check('导入条目留痕（source=import:文件名）',
          all(e['source'] == 'import:k.jsonl' for e in s.entries
              if e.get('source', '').startswith('import'))
          and any(e.get('source', '').startswith('import') for e in s.entries))


def test_search(tmp):
    print('检索（确定性子串匹配）：')
    db = Path(tmp) / 'search.db'
    s = session_module.Session.create(db)
    for category, utterance, op, _answer in data.SCENARIO[:6]:
        s.apply(op, category=category, source='fallback', utterance=utterance)
    s.close()
    r = subprocess.run([sys.executable, str(HERE / 'cli.py'), '--session', str(db),
                        'search', '向上'], capture_output=True, text=True, cwd=str(HERE))
    check('按内容命中（向上 → 实体07）', r.returncode == 0 and '实体07' in r.stdout
          and '实体11' not in r.stdout, r.stdout)
    r2 = subprocess.run([sys.executable, str(HERE / 'cli.py'), '--session', str(db),
                         'search', '规则07'], capture_output=True, text=True, cwd=str(HERE))
    check('按名称命中（规则07）', r2.returncode == 0 and '规则07' in r2.stdout)
    r3 = subprocess.run([sys.executable, str(HERE / 'cli.py'), '--session', str(db),
                         'search', '不存在的词'], capture_output=True, text=True, cwd=str(HERE))
    check('无匹配如实报告', r3.returncode == 0 and '无匹配' in r3.stdout)


def test_concurrent(tmp):
    print('并发压力（5 进程 × 3 teach，SQLite，真子进程）：')
    db = Path(tmp) / 'stress.db'
    session_module.Session.create(db)
    procs = []
    for w in range(5):
        for k in range(3):
            name = f'并发{w}{k}实体'
            procs.append(subprocess.Popen(
                [sys.executable, str(HERE / 'cli.py'), '--session', str(db),
                 'teach', f'教事实 {name}：值{w}{k}'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(HERE)))
    outputs = [p.communicate() for p in procs]
    codes = [p.returncode for p in procs]
    check('全部并发写成功（乐观并发重试生效）', all(c == 0 for c in codes),
          str([(c, (out[0] or '') + (out[1] or ''))[:150]
               for c, out in zip(codes, outputs) if c != 0]))
    s = session_module.Session.load(db)
    names = [e['op']['name'] for e in s.entries if e['status'] == 'ok']
    check('15 条全部落库、无丢失', len(names) == 15, f'实际 {len(names)} 条')
    check('op_id 无冲突（严格线性链）',
          len({e['op_id'] for e in s.entries}) == len(s.entries))


def test_pipeline(tmp):
    print('跨进程管线（真子进程）：')
    demo = Path(tmp) / 'demo'
    demo.mkdir()
    session_path = demo / 'session.json'
    env_py = sys.executable

    def run_cli(*args):
        return subprocess.run([env_py, str(HERE / 'cli.py'), '--session', str(session_path),
                               *args], capture_output=True, text=True, cwd=str(HERE))

    r1 = run_cli('teach', '教规则 规则07：反转，左移一位')
    r2 = run_cli('ask', '对1234应用规则07。')
    r3 = run_cli('correct', '更正规则 规则07：左移一位，反转')
    r3b = run_cli('ask', '查询 不存在的实体')  # 错误条目也要过重放/核验管线
    r4 = run_cli('ask', '对1234应用规则07。')
    check('跨进程四步全成功',
          all(x.returncode == 0 for x in (r1, r2, r3, r4)),
          f'{[x.returncode for x in (r1, r2, r3, r4)]} {r1.stderr}')
    check('跨进程拒绝并留痕（错误条目进日志）',
          r3b.returncode == 1 and '名称不存在' in (r3b.stdout or ''))
    check('跨进程答案：纠错前 3214', '3214' in r2.stdout, r2.stdout)
    check('跨进程答案：纠错后 1432', '1432' in r4.stdout, r4.stdout)

    r5 = subprocess.run([env_py, str(HERE / 'replay.py'), '--session', str(session_path),
                         '--out', str(demo / 'replayed.json')],
                        capture_output=True, text=True, cwd=str(HERE))
    check('独立进程重放通过', r5.returncode == 0, r5.stdout + r5.stderr)
    r6 = subprocess.run([env_py, str(HERE / 'verify.py'), '--session', str(session_path),
                         '--replayed', str(demo / 'replayed.json'),
                         '--out', str(demo / 'verify_result.json')],
                        capture_output=True, text=True, cwd=str(HERE))
    check('独立核验通过', r6.returncode == 0, r6.stdout + r6.stderr)
    if r6.returncode == 0:
        result = json.loads((demo / 'verify_result.json').read_text(encoding='utf-8'))
        check('核验器报告灵魂样例', result['soul_example'] == {'before': '3214', 'after': '1432'})


def main():
    import verify  # noqa: F401  确认可导入且不产生副作用
    with tempfile.TemporaryDirectory() as tmp:
        test_primitives()
        test_store()
        test_proof_and_executor()
        test_parser_fallback()
        test_session(tmp)
        test_sqlite(tmp)
        test_evidence_equivalence(tmp)
        test_checkpoint(tmp)
        test_search(tmp)
        test_import(tmp)
        test_concurrent(tmp)
        test_pipeline(tmp)
    if FAILURES:
        print(f'\n自检失败 {len(FAILURES)} 项: {FAILURES}')
        sys.exit(1)
    print('\n自检全部通过。')


if __name__ == '__main__':
    main()
