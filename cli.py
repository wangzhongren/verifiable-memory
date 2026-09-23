#!/usr/bin/env python3
"""可验证的可教学记忆系统 v1 —— 命令行入口。

一命令一进程（跨进程持久化）+ repl 交互模式：
  python3 cli.py teach   "教事实 实体07：方向是向上"
  python3 cli.py ask     "对1234应用规则07。"
  python3 cli.py correct "更正规则 规则07：左移一位，反转"
  python3 cli.py repl
  python3 cli.py script          # PROTOCOL 固定场景（冻结后执行，拒绝覆盖）
  python3 cli.py status          # 列出记忆库现状
  python3 cli.py search 关键词    # 确定性检索（导航）
  python3 cli.py paraphrases --llm
  python3 cli.py reset

解析器：结构化命令本地执行；有本地配置时口语自动调用模型。
--llm 强制模型解析，--no-llm 禁用模型，--config 可指定配置文件。

日常会话写入 memory.db（默认）或 --session 指定路径；任何
失败都如实退出码 1，不静默兜底。
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from verifiable_memory import data
from verifiable_memory import parser as parser_module
from verifiable_memory import session as session_module
from verifiable_memory import store
from verifiable_memory import vectors
from verifiable_memory.llm import LLM, LLMError
from verifiable_memory.storage import SQLITE_SUFFIXES, export_evidence

HERE = Path(__file__).resolve().parent
DEMO = HERE / 'demo'
DEFAULT_SESSION = HERE / 'memory.db'

FREEZE_SOURCES = ['verifiable_memory/audit.py', 'verifiable_memory/data.py', 'verifiable_memory/store.py',
                  'verifiable_memory/executor.py', 'verifiable_memory/proof.py',
                  'verifiable_memory/vectors.py',
                  'verifiable_memory/llm.py', 'verifiable_memory/parser.py',
                  'verifiable_memory/session.py', 'verifiable_memory/storage.py',
                  'verifiable_memory/__init__.py', 'cli.py',
                  'replay.py', 'verify.py', 'tests/checks.py', 'tests/regressions.py',
                  'tests/llm_checks.py', 'tests/vector_checks.py']


def _make_llm(args, allow_local=False):
    if getattr(args, 'no_llm', False):
        return None
    config_path = getattr(args, 'config', None)
    if not args.llm and not (allow_local and LLM.has_local_config(config_path)):
        return None
    try:
        return LLM(config_path=config_path)
    except LLMError as exc:
        print(f'错误：{exc}', file=sys.stderr)
        sys.exit(1)


def _parse_utterance(args, utterance, known_slots, llm):
    """结构化命令留在本地；有本地配置时，口语才自动调用模型。"""
    try:
        return parser_module.parse(utterance, known_slots, llm)
    except parser_module.ParserError:
        if llm is not None:
            raise  # 模型失败不再换后备路径，保留原始错误
        configured = _make_llm(args, allow_local=True)
        if configured is None:
            raise
        return parser_module.parse(utterance, known_slots, configured)


def _load_or_create(path, capacity=None):
    if Path(path).exists():
        session = session_module.Session.load(path)
        if capacity is not None and capacity != session.capacity:
            print(f"注意：会话已存在，沿用创建时容量 {session.capacity}"
                  f"（--capacity {capacity} 被忽略）")
        return session
    return session_module.Session.create(
        path, capacity=store.CAPACITY_DEFAULT if capacity is None else capacity)


def _print_entry(entry, as_json):
    if as_json:
        print(json.dumps(entry, ensure_ascii=False, sort_keys=True, indent=1))
        return
    if entry['status'] == 'ok':
        op = entry['op']
        if op['op'] in store.WRITE_OPS:
            result = entry['result']
            cert = entry.get('proof')
            from verifiable_memory.proof import summarize_certificate
            print(f"✓ 已写入 {result['name']}（第{result['revision']}版） · "
                  f"{summarize_certificate(cert)}")
        else:
            result = entry['result']
            evidence = (f"记录 {result['name']} 第{result['revision']}版"
                        f"（写入于 {result['written_by']}，槽哈希 {result['slot_hash'][:12]}）")
            extra = ''
            if 'program' in result and result.get('trace'):
                steps = ' → '.join(f"{t['op']}:{t['before']}→{t['after']}" for t in result['trace'])
                extra = f' · 执行 [{steps}] · 后端 {result["backend"]}'
            elif 'program' in result:
                extra = f" · 程序 {'→'.join(result['program'])}"
            elif result.get('kind') == 'entity_derivation':
                path = ' → '.join(f"{t['source']} --{t['action']}[{t['edge']}]--> {t['target']}"
                                  for t in result['trace']) or '起点即终点'
                extra = f" · 推导 [{path}] · 向量 {result['vector']}"
            elif result.get('kind') == 'vector_action' and 'output_vector' in result:
                extra = (f" · 输出向量 {result['output_vector']}"
                         f" · 候选实体 {result['matches']}")
            print(f"✓ 答案：{result['answer']} · 证据：{evidence}{extra}")
    else:
        print(f"✗ 拒绝：{entry['error']}")


def cmd_teach(args):
    _run_single(args, 'teach')


def cmd_correct(args):
    _run_single(args, 'correct')


def cmd_ask(args):
    _run_single(args, 'ask')


def cmd_vector(args):
    """用结构化 JSON 操作实体向量与有向边，避免口语解析改变数值。"""
    try:
        op = json.loads(args.operation)
    except json.JSONDecodeError as exc:
        raise SystemExit(f'向量操作不是合法 JSON：{exc}') from exc
    session = _load_or_create(args.session, capacity=args.capacity)
    try:
        entry = session.apply(op, category='vector', source='vector-json',
                              utterance=args.operation)
    finally:
        session.close()
    _print_entry(entry, args.json)
    sys.exit(0 if entry['status'] == 'ok' else 1)


def _run_single(args, category):
    llm = _make_llm(args)
    session = _load_or_create(args.session)
    try:
        try:
            op, source = _parse_utterance(args, args.utterance, session.store.known_slots(), llm)
        except (parser_module.ParserError, store.StoreError, LLMError, ValueError) as exc:
            print(f'解析失败：{exc}', file=sys.stderr)
            sys.exit(1)
        try:
            entry = session.apply(op, category=category, source=source, utterance=args.utterance)
        except session_module.SessionError as exc:
            print(f'会话错误：{exc}', file=sys.stderr)
            sys.exit(1)
    finally:
        session.close()
    _print_entry(entry, args.json)
    sys.exit(0 if entry['status'] == 'ok' else 1)


def cmd_reset(args):
    path = Path(args.session)
    removed = False
    for suffix in ('', '-wal', '-shm', '-journal'):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()
            removed = True
    print(f'已删除会话：{path}' if removed else '没有可删除的会话')


def cmd_status(args):
    """列出记忆库现状：全部槽位（含内容与出处）、日志条数、终态哈希。"""
    session = _load_or_create(args.session, capacity=args.capacity)
    try:
        print(f"会话 {session.path}")
        tail_note = (f"（载入尾部 {len(session.entries)} 条）"
                     if session.n_ops != len(session.entries) else '')
        print(f"槽位 {len(session.store.slots)}/{session.capacity} · "
              f"日志 {session.n_ops} 条{tail_note} · "
              f"终态哈希 {session.terminal_state_hash()[:12]}")
        if not session.store.slots:
            print('  （记忆为空——用 teach 教第一条）')
        for slot in session.store.known_slots():
            rec = session.store.slots[slot['name']]
            if slot['kind'] == 'fact':
                print(f"  [事实] {slot['name']} · 第{rec['revision']}版 · "
                      f"{rec['written_by']} · {rec['content']}")
            elif slot['kind'] == 'rule':
                print(f"  [规则] {slot['name']} · 第{rec['revision']}版 · "
                      f"{rec['written_by']} · {'→'.join(rec['program'])}")
            elif slot['kind'] == 'entity':
                print(f"  [实体] {slot['name']} · 第{rec['revision']}版 · "
                      f"{rec['written_by']} · {len(rec['vector_units'])}维向量")
            elif slot['kind'] == 'vector_action':
                print(f"  [向量动作] {slot['name']} · 第{rec['revision']}版 · "
                      f"{rec['written_by']} · {len(rec['delta_units'])}维增量")
            else:
                active = '有效' if vectors.active_edge(slot['name'], rec, session.store.slots) else '已失效'
                print(f"  [有向边] {slot['name']} · 第{rec['revision']}版 · "
                      f"{rec['source']} --{rec['action']}--> {rec['target']} · {active}")
    finally:
        session.close()


def cmd_search(args):
    """确定性检索：名称与内容的子串匹配。

    检索是导航（同 ls），不落 op 日志；引用所选名称的 teach/ask/
    correct 操作才落日志——选择由后续 op 本身留痕。"""
    session = _load_or_create(args.session, capacity=args.capacity)
    try:
        q = args.query.lower()
        hits = []
        for name, rec in session.store.slots.items():
            text = name + ' ' + (
                rec['content'] if rec['kind'] == 'fact' else
                '→'.join(rec['program']) if rec['kind'] == 'rule' else
                f"{rec['source']} {rec['action']} {rec['target']}" if rec['kind'] == 'edge' else '')
            if q in text.lower():
                hits.append((name, rec))
        if not hits:
            print(f"无匹配：{args.query}")
            return
        for name, rec in hits:
            if rec['kind'] == 'fact':
                print(f"  [事实] {name} · 第{rec['revision']}版 · "
                      f"{rec['written_by']} · {rec['content']}")
            elif rec['kind'] == 'rule':
                print(f"  [规则] {name} · 第{rec['revision']}版 · "
                      f"{rec['written_by']} · {'→'.join(rec['program'])}")
            else:
                print(f"  [{rec['kind']}] {name} · 第{rec['revision']}版")
        print(f"共 {len(hits)} 处匹配")
    finally:
        session.close()


def cmd_import(args):
    """批量导入知识文件（JSONL，或 JSON 数组）为逐条 teach op。

    每行一个条目：{"kind": "fact"|"rule", "name": ..., "content": ...|"program": [...]}
    导入必须过验证边界：每条都是一个真实的、留痕的操作
    （source='import:<文件名>'），坏条目先验形、不落日志。"""
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f'导入文件不存在：{path}')
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        raise SystemExit(f'读取失败：{exc}')
    # JSONL：逐行解析，坏行按行报告、不拖垮整批；JSON 数组：整文档，解析失败即致命
    if text.lstrip().startswith('['):
        try:
            arr = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f'JSON 解析失败：{exc}')
        items = [(f'[{i}]', item) for i, item in enumerate(arr, start=1)]
    else:
        items = [(f'行{n}', line) for n, line in enumerate(text.splitlines(), start=1)
                 if line.strip()]

    def _as_item(raw):
        return raw if isinstance(raw, dict) else json.loads(raw)

    if args.dry_run:
        session = _load_or_create(args.session, capacity=args.capacity)
        problems, would, skipped = [], 0, 0
        try:
            for label, raw in items:
                name = None
                try:
                    item = _as_item(raw)
                    name = item.get('name')
                    kind = item.get('kind')
                    if kind not in ('fact', 'rule'):
                        raise store.StoreError('kind 必须是 fact 或 rule')
                    op = (data.teach_fact if kind == 'fact' else data.teach_rule)(
                        name, item.get('content') if kind == 'fact' else item.get('program'))
                    store.validate_op(op)
                    if name in session.store.slots:
                        skipped += 1
                    else:
                        would += 1
                except (store.StoreError, ValueError) as exc:
                    problems.append(f'  ✗ {label} {name}: {exc}')
        finally:
            session.close()
        print(f"预检：可新增 {would} · 已存在将跳过 {skipped} · "
              f"坏条目 {len(problems)}（未写入任何内容）")
        for line in problems:
            print(line)
        sys.exit(1 if problems else 0)

    session = _load_or_create(args.session, capacity=args.capacity)
    results = {'imported': [], 'skipped': [], 'failed': []}
    try:
        for label, raw in items:
            name = None
            try:
                item = _as_item(raw)
                name = item.get('name')
                kind = item.get('kind')
                if kind not in ('fact', 'rule'):
                    raise store.StoreError('kind 必须是 fact 或 rule')
                existing = name in session.store.slots
                if existing and args.on_conflict == 'skip':
                    results['skipped'].append((label, name, '已存在'))
                    continue
                use_correct = existing and args.on_conflict == 'correct'
                if kind == 'fact':
                    op = (data.correct_fact if use_correct else data.teach_fact)(
                        name, item.get('content'))
                else:
                    op = (data.correct_rule if use_correct else data.teach_rule)(
                        name, item.get('program'))
                store.validate_op(op)  # 先验形：坏条目不落日志
                entry = session.apply(
                    op, category='correct' if use_correct else 'teach',
                    source=f'import:{path.name}',
                    utterance=f'导入 {path.name} {label}')
                if entry['status'] == 'ok':
                    results['imported'].append((label, name, entry['result']['revision']))
                else:
                    results['failed'].append((label, name, entry['error']))
            except (store.StoreError, ValueError) as exc:
                results['failed'].append((label, name, str(exc)))
    finally:
        session.close()

    print(f"导入完成：新增 {len(results['imported'])} · "
          f"跳过 {len(results['skipped'])} · 失败 {len(results['failed'])}")
    for label, name, info in results['failed']:
        print(f'  ✗ {label} {name}: {info}')
    for label, name, info in results['skipped']:
        print(f'  - {label} {name}: {info}')
    sys.exit(1 if results['failed'] else 0)


def cmd_export(args):
    """把 SQLite 会话导出为 canonical 证据 session.json（replay/verify 吃它）。"""
    path = Path(args.session)
    if path.suffix not in SQLITE_SUFFIXES:
        raise SystemExit('export 只适用于 SQLite 会话（.db/.sqlite/.sqlite3）；'
                         'JSON 会话文件本身就是证据格式')
    out = Path(args.out) if args.out else path.with_name(path.stem + '.session.json')
    if out.exists():
        raise SystemExit(f'拒绝覆盖：{out}')
    payload = export_evidence(path, out)
    terminal = (payload['entries'][-1]['state_hash_after'][:12]
                if payload['entries'] else '（空）')
    print(f"证据已导出：{out} · {len(payload['entries'])} 条 · 终态哈希 {terminal}")
    print(f"复核：python3 replay.py --session {out} --out <replayed.json>")


def cmd_repl(args):
    llm = _make_llm(args)
    session = _load_or_create(args.session)
    print('可教学记忆系统 v1 — repl 模式（exit 退出）')
    print(f"会话: {session.path} · 槽位 {len(session.store.slots)}/{session.capacity}"
          f" · 解析器: {'LLM' if llm else '后备文法'}")
    try:
        _repl_loop(args, llm, session)
    finally:
        session.close()


def _repl_loop(args, llm, session):
    while True:
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ('exit', 'quit', '退出'):
            break
        try:
            op, source = _parse_utterance(args, line, session.store.known_slots(), llm)
        except (parser_module.ParserError, store.StoreError, LLMError, ValueError) as exc:
            print(f'解析失败：{exc}')
            continue
        category = {'teach_fact': 'teach', 'teach_rule': 'teach',
                    'correct_fact': 'correct', 'correct_rule': 'correct'}.get(op['op'], 'ask')
        try:
            entry = session.apply(op, category=category, source=source, utterance=line)
        except session_module.SessionError as exc:
            print(f'会话错误：{exc}')
            continue
        _print_entry(entry, args.json)


def _freeze(demo):
    """PROTOCOL + 全部源码冻结进 demo/frozen/，附 sha256 清单。"""
    frozen = demo / 'frozen'
    frozen.mkdir(parents=True)
    manifest_files = {}
    for name in ['docs/PROTOCOL.md'] + FREEZE_SOURCES:
        source = HERE / name
        if not source.exists():
            raise SystemExit(f'冻结失败：缺 {name}')
        dest = frozen / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        manifest_files[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    import platform
    manifest = {'files': manifest_files, 'python': sys.version.split()[0],
                'platform': platform.platform()}
    (frozen / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    return manifest


def cmd_script(_args):
    """PROTOCOL 固定场景：冻结 → 教学演示 → 转录与结果落盘。

    demo/ 已存在则拒绝（可验证性优先于便利性：不覆盖任何已产出证据）。
    """
    if DEMO.exists():
        raise SystemExit(f'拒绝执行：{DEMO} 已存在（复现请先删除或另开实验序号）')
    manifest = _freeze(DEMO)
    session = session_module.Session.create(DEMO / 'session.json')

    rows = []
    for category, utterance, expected_op, expected_answer in data.SCENARIO:
        op = parser_module.parse_fallback(utterance)
        source = 'fallback'
        parse_ok = (op == expected_op)
        entry = session.apply(op, category=category, source=source, utterance=utterance)
        answer = (entry['result'].get('answer') if entry['status'] == 'ok'
                  else f"ERROR: {entry['error']}")
        answer_ok = (expected_answer is None) or (answer == expected_answer)
        rows.append({'op_id': entry['op_id'], 'category': category,
                     'utterance': utterance, 'parse_ok': parse_ok,
                     'status': entry['status'], 'answer': answer,
                     'expected_answer': expected_answer,
                     'answer_ok': answer_ok,
                     'zero_collateral': entry.get('proof', {}).get('zero_collateral')})

    all_parse = all(r['parse_ok'] for r in rows)
    all_answer = all(r['answer_ok'] for r in rows)
    result = {'all_parse_ok': all_parse, 'all_answer_ok': all_answer,
              'n_steps': len(rows), 'terminal_state_hash': session.terminal_state_hash(),
              'session_file_sha256': hashlib.sha256(
                  (DEMO / 'session.json').read_bytes()).hexdigest(),
              'frozen_manifest': manifest, 'rows': rows}
    (DEMO / 'script_result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    _write_transcript(rows, result)

    print(f"场景完成：{len(rows)} 步 · 解析 {'全对' if all_parse else '有错'} · "
          f"答案 {'全对' if all_answer else '有错'}")
    print(f"证据：{DEMO}/session.json, script_result.json, CLI_TRANSCRIPT.md, frozen/")
    sys.exit(0 if (all_parse and all_answer) else 1)


def _write_transcript(rows, result):
    lines = ['# verifiable_memory_01 — 固定场景 CLI 转录', '',
             f"终态哈希：`{result['terminal_state_hash']}`", '',
             '| op_id | 类别 | 话语 | 解析 | 状态 | 答案 | 期望 | 判定 | 零附带损害 |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        verdict = '✓' if (r['parse_ok'] and r['answer_ok']) else '✗'
        lines.append(
            f"| {r['op_id']} | {r['category']} | {r['utterance']} "
            f"| {'✓' if r['parse_ok'] else '✗'} | {r['status']} "
            f"| {r['answer']} | {r['expected_answer']} | {verdict} "
            f"| {'—' if r['zero_collateral'] is None else ('是' if r['zero_collateral'] else '否')} |")
    lines += ['', '复核路径：新进程 replay.py 重放 → verify.py 独立核验（第二套原语实现）。',
              '']
    (DEMO / 'CLI_TRANSCRIPT.md').write_text('\n'.join(lines), encoding='utf-8')


def cmd_paraphrases(args):
    """改写基准 v2：显式前置状态，严格比较原始期望，不做语义放宽。"""
    out_arg = getattr(args, 'out', None)
    out = Path(out_arg) if out_arg else HERE / 'demo' / 'paraphrase_result.json'
    if out.exists():
        raise SystemExit(f'拒绝覆盖：{out}')
    if len(data.PARAPHRASES) != len(data.PARAPHRASE_KNOWN_SLOTS):
        raise SystemExit('改写用例与前置状态数量不一致')
    llm = _make_llm(args, allow_local=True)
    if llm is None:
        raise SystemExit('改写鲁棒性测试需要 --llm（无 key 时本测试不适用，'
                         '固定场景用后备文法已覆盖）')
    # 在调用前冻结本次输入、期望、提示词摘要；结果文件保留原始配置。
    cases = [{'utterance': utterance, 'expected': expected, 'known_slots': known}
             for (utterance, expected), known in
             zip(data.PARAPHRASES, data.PARAPHRASE_KNOWN_SLOTS)]
    prompt_hash = hashlib.sha256(parser_module.SYSTEM_PROMPT.encode('utf-8')).hexdigest()
    rows = []
    for case in cases:
        utterance, expected_op = case['utterance'], case['expected']
        try:
            op = parser_module.parse_with_llm(utterance, case['known_slots'], llm)
            rows.append({**case, 'got': op, 'match': op == expected_op, 'error': None})
        except (parser_module.ParserError, ValueError, LLMError) as exc:
            rows.append({**case, 'got': None, 'match': False, 'error': str(exc)})
    matches = sum(1 for r in rows if r['match'])
    print(f'改写解析：{matches}/{len(rows)} 与预声明 op 一致')
    for r in rows:
        mark = '✓' if r['match'] else '✗'
        got = json.dumps(r['got'], ensure_ascii=False) if r['got'] is not None else f"失败: {r['error']}"
        print(f"  {mark} {r['utterance']} → {got}")
    if out_arg or out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {'protocol': data.PARAPHRASE_PROTOCOL,
                  'parser_prompt_sha256': prompt_hash,
                  'model': llm.model, 'api_style': llm.api_style,
                  'matches': matches, 'n_cases': len(rows), 'rows': rows}
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'结果已写入 {out}')
    sys.exit(0 if matches == len(rows) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--session', default=str(DEFAULT_SESSION), help='会话文件路径')
    ap.add_argument('--capacity', type=int, default=None,
                    help='新建会话时的槽位容量（默认 256；已存在的会话沿用其容量）')
    ap.add_argument('--config', help='模型配置文件（默认 ~/.config/verifiable-memory/config.json）')
    llm_options = ap.add_mutually_exclusive_group()
    llm_options.add_argument('--llm', action='store_true', help='强制使用已配置模型解析')
    llm_options.add_argument('--no-llm', action='store_true', help='仅使用本地结构化文法，不调用模型')
    ap.add_argument('--json', action='store_true', help='输出原始条目 JSON')
    sub = ap.add_subparsers(dest='command', required=True)
    for name, fn, help_text in (
            ('teach', cmd_teach, '教学一条话语'),
            ('ask', cmd_ask, '提问/执行'),
            ('correct', cmd_correct, '纠错一条话语'),
            ('status', cmd_status, '列出记忆库现状'),
            ('search', cmd_search, '按关键词检索记忆'),
            ('repl', cmd_repl, '交互模式'),
            ('script', cmd_script, 'PROTOCOL 固定场景'),
            ('reset', cmd_reset, '删除会话'),
            ('import', cmd_import, '批量导入知识文件（JSONL）'),
            ('export', cmd_export, '导出 SQLite 会话为证据 JSON'),
            ('paraphrases', cmd_paraphrases, 'LLM 改写鲁棒性测试')):
        p = sub.add_parser(name, help=help_text)
        if name in ('teach', 'ask', 'correct'):
            p.add_argument('utterance', help='自然语言/结构化话语')
        if name == 'import':
            p.add_argument('file', help='JSONL 知识文件（或 JSON 数组）')
            p.add_argument('--on-conflict', choices=('skip', 'correct'), default='skip',
                           help='名称已存在时：跳过（默认）或转为更正')
            p.add_argument('--dry-run', action='store_true', help='只预检不写入')
        if name == 'export':
            p.add_argument('--out', default=None, help='证据输出路径（默认 <会话名>.session.json）')
        if name == 'paraphrases':
            p.add_argument('--out', default=None, help='保存带协议版本与逐例上下文的测试结果（拒绝覆盖）')
        if name == 'search':
            p.add_argument('query', help='关键词（名称与内容的子串匹配）')
        p.set_defaults(func=fn)
    p = sub.add_parser('vector', help='实体向量、有向动作及多步推导（JSON op）')
    p.add_argument('operation', help='结构化 JSON 操作')
    p.set_defaults(func=cmd_vector)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
