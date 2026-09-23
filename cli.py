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
import os
import shutil
import sys
from pathlib import Path

from verifiable_memory import data
from verifiable_memory import parser as parser_module
from verifiable_memory import policy as policy_module
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
                  'verifiable_memory/policy.py',
                  'verifiable_memory/llm.py', 'verifiable_memory/parser.py',
                  'verifiable_memory/session.py', 'verifiable_memory/storage.py',
                  'verifiable_memory/__init__.py', 'cli.py',
                  'replay.py', 'verify.py', 'tests/checks.py', 'tests/regressions.py',
                  'tests/llm_checks.py', 'tests/vector_checks.py',
                  'tests/policy_checks.py', 'tests/continual_checks.py',
                  'tests/self_study_checks.py']


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
            elif result.get('kind') == 'policy_route':
                path = ' → '.join(t['edge'] for t in result['trace']) or '未选择边'
                extra = (f" · 策略 {result['policy_sha256'][:12]} · 路径 [{path}]"
                         f" · {'弃权' if result['abstained'] else '完成'}: {result['reason']}")
            elif result.get('kind') == 'vector_action' and 'output_vector' in result:
                extra = (f" · 输出向量 {result['output_vector']}"
                         f" · 候选实体 {result['matches']}")
            answer = ('未判定' if result.get('kind') == 'policy_route'
                      and result['abstained'] else result['answer'])
            print(f"✓ 答案：{answer} · 证据：{evidence}{extra}")
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


def cmd_policy_train(args):
    """从带标签的分叉 JSONL 训练小策略模型；不改动记忆库。"""
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'拒绝覆盖模型文件：{out}')
    try:
        rows = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines()
                if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'训练数据读取失败：{exc}') from exc
    try:
        session = session_module.Session.load(args.session)
    except session_module.SessionError as exc:
        raise SystemExit(f'训练需要已有实体图：{exc}') from exc
    try:
        model = policy_module.train(rows, session.store.slots)
    except policy_module.PolicyError as exc:
        raise SystemExit(f'训练失败：{exc}') from exc
    finally:
        session.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = store.canonical_json(model) + '\n'
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as target:
        target.write(text)
    print(f'模型已训练：{len(rows)} 条标注 · 图指纹 {model["graph_signature"][:12]}'
          f' · 文件 {out}')


def cmd_policy_learn(args):
    """按反馈增量更新权重，保留旧模型和完整标注历史。"""
    model_out, data_out = Path(args.out), Path(args.data_out)
    source_paths = {Path(args.model).resolve(), Path(args.data).resolve(),
                    Path(args.feedback).resolve()}
    if args.eval:
        source_paths.add(Path(args.eval).resolve())
    if (model_out.resolve() in source_paths or data_out.resolve() in source_paths
            or model_out.resolve() == data_out.resolve()
            or model_out.exists() or data_out.exists()):
        raise SystemExit('输出文件必须为两个新的不同路径，且不能覆盖模型或标注输入')
    try:
        model_bytes = Path(args.model).read_bytes()
        previous = json.loads(model_bytes)

        def read_jsonl(path):
            return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
                    if line.strip()]

        old_rows = read_jsonl(args.data)
        feedback = read_jsonl(args.feedback)
        holdout = read_jsonl(args.eval) if args.eval else None
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'持续学习输入读取失败：{exc}') from exc
    try:
        session = session_module.Session.load(args.session)
    except session_module.SessionError as exc:
        raise SystemExit(f'持续学习需要已有实体图：{exc}') from exc
    try:
        parent_hash = hashlib.sha256(model_bytes).hexdigest()
        candidate, merged, stats = policy_module.learn(
            previous, old_rows, feedback, session.store.slots, parent_hash, holdout)
    except policy_module.PolicyError as exc:
        raise SystemExit(f'持续学习被拒绝：{exc}') from exc
    finally:
        session.close()

    _write_policy_pair(model_out, data_out, candidate, merged)
    report = {**stats, 'model_path': str(model_out), 'data_path': str(data_out)}
    print(json.dumps(report, ensure_ascii=False, indent=1))


def _write_policy_pair(model_out, data_out, model, rows):
    """只写新版本；第二个文件失败时清理本次已创建的文件。"""
    content = ((data_out, ''.join(store.canonical_json(row) + '\n' for row in rows)),
               (model_out, store.canonical_json(model) + '\n'))
    for path, _ in content:
        path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, text in content:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created.append(path)
            with os.fdopen(fd, 'w', encoding='utf-8') as target:
                target.write(text)
    except OSError as exc:
        for path in created:
            path.unlink(missing_ok=True)
        raise SystemExit(f'策略版本文件写入失败：{exc}') from exc


def cmd_policy_self_study(args):
    """模型作答→对照教材→写入精确纠错→尝试生成候选模型。"""
    model_out, data_out = Path(args.out), Path(args.data_out)
    sources = {Path(args.model).resolve(), Path(args.data).resolve(),
               Path(args.book).resolve()}
    if args.eval:
        sources.add(Path(args.eval).resolve())
    if (model_out.resolve() in sources or data_out.resolve() in sources
            or model_out.resolve() == data_out.resolve()
            or model_out.exists() or data_out.exists()):
        raise SystemExit('新模型与新数据必须使用不同的新路径，不能覆盖输入')
    try:
        model_bytes = Path(args.model).read_bytes()
        model = json.loads(model_bytes)

        def jsonl(path):
            return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
                    if line.strip()]

        old_rows = jsonl(args.data)
        book_bytes = Path(args.book).read_bytes()
        book_rows = [json.loads(line) for line in book_bytes.decode('utf-8').splitlines()
                     if line.strip()]
        holdout = jsonl(args.eval) if args.eval else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f'自测输入读取失败：{exc}') from exc
    try:
        session = session_module.Session.load(args.session)
    except session_module.SessionError as exc:
        raise SystemExit(f'自测需要已有实体图：{exc}') from exc
    try:
        slots = session.store.slots
        policy_module.validate_model(model, slots)
        if (model.get('generation', 1) > 1
                and model.get('promotion') != 'evaluated'
                and not args.allow_candidate):
            raise policy_module.PolicyError('候选模型未通过独立评估，实验性自测需 --allow-candidate')
        if (model.get('training_sha256') != policy_module.training_digest(old_rows)
                or model.get('n_examples') != len(old_rows)):
            raise policy_module.PolicyError('原标注文件与模型训练摘要不符')
        book = policy_module.check_book(book_rows, slots)
        inspected, model_misses, deployed_misses = [], [], []
        for item in book:
            source, query, gold = item['source'], item['query'], item['edge']
            raw = policy_module.route(model, query, source, slots,
                                      max_hops=1, use_overrides=False)
            deployed = policy_module.route(model, query, source, slots,
                                           max_hops=1, use_overrides=True)
            raw_choice = raw['decisions'][0]['choice'] if raw['decisions'] else None
            active_choice = (deployed['decisions'][0]['choice']
                             if deployed['decisions'] else None)
            inspected.append({'source': source, 'query': query, 'expected': gold,
                              'model_choice': raw_choice, 'active_choice': active_choice})
            if raw_choice != gold:
                model_misses.append(item)
            if active_choice != gold:
                deployed_misses.append(item)
        corrections = {store.override_name(item['source'], item['query']): item
                       for item in model_misses + deployed_misses}
        for name, item in corrections.items():
            prior = slots.get(name)
            if prior is not None and prior['kind'] != 'policy_override':
                raise policy_module.PolicyError('教材纠错名称与已有记录冲突')
            if (prior is not None and prior['enabled'] and prior['origin'] == 'manual'
                    and (prior['edge'] != item['edge']
                         or prior['stop_after'] != item['stop_after'])):
                raise policy_module.PolicyError('教材答案与人工确认纠错冲突，拒绝自动覆盖')
        parent_hash = hashlib.sha256(model_bytes).hexdigest()
        candidate = merged = stats = None
        regression = None
        if model_misses:
            feedback = [{'source': item['source'], 'query': item['query'],
                         'edge': item['edge']} for item in model_misses]
            try:
                candidate, merged, stats = policy_module.learn(
                    model, old_rows, feedback, slots, parent_hash, holdout)
            except policy_module.PolicyRegressionError as exc:
                regression = str(exc)
        book_hash = hashlib.sha256(book_bytes).hexdigest()
        written = []
        for name, item in corrections.items():
            prior = session.store.slots.get(name)
            if (prior is not None and policy_module.override_active(prior, session.store.slots)
                    and prior['edge'] == item['edge']
                    and prior['stop_after'] == item['stop_after']):
                continue
            original = next(row for row in inspected
                            if row['source'] == item['source'] and row['query'] == item['query'])
            before = original['model_choice']
            op = {'op': 'correct_override' if prior is not None else 'teach_override',
                  'name': name, 'source': item['source'], 'query': item['query'],
                  'edge': item['edge'], 'enabled': True,
                  'stop_after': item['stop_after'], 'origin': 'book',
                  'reason': f'教材核对：模型原选 {before}；正确答案 {item["edge"]}',
                  'judge': 'book:' + book_hash}
            entry = session.apply(op, category='policy', source='ai-self-check-book',
                                  utterance=item['query'])
            if entry['status'] != 'ok':
                raise policy_module.PolicyError(entry['error'])
            verified = policy_module.route(model, item['query'], item['source'],
                                           session.store.slots, max_hops=1)
            verified_choice = (verified['decisions'][0]['choice']
                               if verified['decisions'] else None)
            if verified_choice != item['edge']:
                raise policy_module.PolicyError('纠错已写入但复查结果不符')
            written.append(entry['op_id'])
    except policy_module.PolicyError as exc:
        raise SystemExit(f'自测失败：{exc}') from exc
    finally:
        session.close()
    if candidate is not None:
        _write_policy_pair(model_out, data_out, candidate, merged)
    report = {'checked': len(book), 'model_misses': len(model_misses),
              'active_misses': len(deployed_misses),
              'exact_corrections_written': len(written),
              'correction_op_ids': written, 'book_sha256': book_hash,
              'model_update': stats, 'model_update_rejected': regression,
              'new_model': str(model_out) if candidate is not None else None,
              'new_data': str(data_out) if candidate is not None else None,
              'answers': inspected}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(2 if regression else 0)


def cmd_policy_route(args):
    """逐节点自主选择分支，再让 Store 验证选中路径与向量运算。"""
    try:
        model_bytes = Path(args.model).read_bytes()
        model = json.loads(model_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'模型文件读取失败：{exc}') from exc
    try:
        session = session_module.Session.load(args.session)
    except session_module.SessionError as exc:
        raise SystemExit(f'策略路由需要已有实体图：{exc}') from exc
    try:
        if (model.get('generation', 1) > 1
                and model.get('promotion') != 'evaluated'
                and not args.allow_candidate):
            raise policy_module.PolicyError(
                '增量模型仍是候选版；先用独立评估通过的版本，'
                '实验性路由需显式 --allow-candidate')
        decision = policy_module.route(model, args.query, args.source,
                                       session.store.slots, args.max_hops)
        op = {'op': 'route_entities', 'name': decision['target'],
              'source': args.source, 'query': args.query,
              'path': decision['path'], 'decisions': decision['decisions'],
              'abstained': decision['abstained'], 'reason': decision['reason'],
              'max_hops': args.max_hops,
              'policy_sha256': hashlib.sha256(model_bytes).hexdigest()}
        entry = session.apply(op, category='policy', source='trained-policy',
                              utterance=args.query)
    except (policy_module.PolicyError, session_module.SessionError) as exc:
        raise SystemExit(f'策略路由失败：{exc}') from exc
    finally:
        session.close()
    _print_entry(entry, args.json)
    sys.exit(1 if entry['status'] == 'error' else 2 if decision['abstained'] else 0)


def cmd_policy_correct(args):
    """把明确反馈存入审计日志，作为同源实体/原话的精确优先路由。"""
    if args.disable:
        if args.edge is not None or args.abstain or args.continue_route:
            raise SystemExit('--disable 不与边选择、弃权或继续路由同时使用')
        edge, enabled = None, False
    else:
        if (args.edge is None) == (not args.abstain):
            raise SystemExit('必须且只能选择 --edge <边名> 或 --abstain')
        if args.abstain and args.continue_route:
            raise SystemExit('弃权不能再继续路由')
        edge, enabled = args.edge, True
    name = store.override_name(args.source, args.query)
    session = session_module.Session.load(args.session)
    try:
        if args.disable and name not in session.store.slots:
            raise SystemExit('无法停用不存在的纠错')
        op = {'op': 'correct_override' if name in session.store.slots else 'teach_override',
              'name': name, 'source': args.source, 'query': args.query,
              'edge': edge, 'enabled': enabled, 'stop_after': not args.continue_route}
        entry = session.apply(op, category='policy', source='explicit-feedback-cli',
                              utterance=f'{args.source}: {args.query} -> {edge}')
    finally:
        session.close()
    _print_entry(entry, args.json)
    sys.exit(0 if entry['status'] == 'ok' else 1)



def cmd_policy_export_feedback(args):
    """导出当前图上仍有效的教材核对纠错，供持续学习读取。"""
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'拒绝覆盖：{out}')
    try:
        session = session_module.Session.load(args.session)
    except session_module.SessionError as exc:
        raise SystemExit(f'导出 AI 反馈失败：{exc}') from exc
    try:
        rows = [{'source': rec['source'], 'query': rec['query'], 'edge': rec['edge']}
                for name, rec in sorted(session.store.slots.items())
                if rec['kind'] == 'policy_override' and rec['origin'] == 'book'
                and policy_module.override_active(rec, session.store.slots)]
    finally:
        session.close()
    if not rows:
        raise SystemExit('没有可导出的当前有效教材纠错')
    out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as target:
        for row in rows:
            target.write(store.canonical_json(row) + '\n')
    print(f'已导出 {len(rows)} 条当前有效教材反馈：{out}')


def cmd_policy_eval(args):
    """用独立标注集报告首步选边与弃权，不改动图或日志。"""
    try:
        model = json.loads(Path(args.model).read_text(encoding='utf-8'))
        rows = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines()
                if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'评估输入读取失败：{exc}') from exc
    try:
        session = session_module.Session.load(args.session)
    except session_module.SessionError as exc:
        raise SystemExit(f'评估需要已有实体图：{exc}') from exc
    try:
        result = policy_module.evaluate(model, rows, session.store.slots)
    except policy_module.PolicyError as exc:
        raise SystemExit(f'评估失败：{exc}') from exc
    finally:
        session.close()
    print(json.dumps(result, ensure_ascii=False, indent=1))


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
            elif slot['kind'] == 'edge':
                active = '有效' if vectors.active_edge(slot['name'], rec, session.store.slots) else '已失效'
                print(f"  [有向边] {slot['name']} · 第{rec['revision']}版 · "
                      f"{rec['source']} --{rec['action']}--> {rec['target']} · {active}")
            else:
                active = '有效' if policy_module.override_active(rec, session.store.slots) else '未启用/已过期'
                print(f"  [精确纠错] {slot['name']} · 第{rec['revision']}版 · "
                      f"{rec['source']} / {rec['query']} → {rec['edge']} · {active}"
                      f" · 来源 {rec['origin']}")
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
                f"{rec['source']} {rec['action']} {rec['target']}" if rec['kind'] == 'edge' else
                f"{rec['source']} {rec['query']} {rec['edge']}" if rec['kind'] == 'policy_override' else '')
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
    p = sub.add_parser('policy-train', help='用分叉标注 JSONL 训练本地控制器')
    p.add_argument('--data', required=True, help='每行 source、query、edge；edge=null 表示弃权')
    p.add_argument('--out', required=True, help='新模型文件路径（拒绝覆盖）')
    p.set_defaults(func=cmd_policy_train)
    p = sub.add_parser('policy-learn', help='用新标注从旧模型继续训练，输出新版本')
    p.add_argument('--model', required=True, help='上一版模型文件')
    p.add_argument('--data', required=True, help='与模型摘要一致的上一版标注 JSONL')
    p.add_argument('--feedback', required=True, help='本次新增或更正的标注 JSONL')
    p.add_argument('--out', required=True, help='新模型路径（拒绝覆盖）')
    p.add_argument('--data-out', required=True, help='合并后的新标注路径（拒绝覆盖）')
    p.add_argument('--eval', help='可选的独立评估 JSONL；指标退步时拒绝更新')
    p.set_defaults(func=cmd_policy_learn)
    p = sub.add_parser('policy-route', help='让已训练控制器沿有效有向边自主选路')
    p.add_argument('--model', required=True, help='训练生成的模型 JSON')
    p.add_argument('--source', required=True, help='起点实体名称')
    p.add_argument('--query', required=True, help='输入查询')
    p.add_argument('--max-hops', type=int, default=8, help='最大分叉步数 1–16')
    p.add_argument('--allow-candidate', action='store_true',
                   help='实验性使用尚未通过独立评估的反馈模型')
    p.set_defaults(func=cmd_policy_route)
    p = sub.add_parser('policy-correct', help='将人工确认的同实体/原话分叉写入记忆库')
    p.add_argument('--source', required=True, help='源实体名称')
    p.add_argument('--query', required=True, help='需要精确匹配的原话')
    override = p.add_mutually_exclusive_group()
    override.add_argument('--edge', help='确认的有向边名称')
    override.add_argument('--abstain', action='store_true', help='这条原话应当弃权')
    override.add_argument('--disable', action='store_true', help='停用现有纠错，回退到模型')
    p.add_argument('--continue-route', action='store_true', help='纠错一步后继续让模型处理后续路径')
    p.set_defaults(func=cmd_policy_correct)
    p = sub.add_parser('policy-export-feedback', help='导出可用于再训练的有效教材纠错')
    p.add_argument('--out', required=True, help='新 JSONL 文件路径（拒绝覆盖）')
    p.set_defaults(func=cmd_policy_export_feedback)
    p = sub.add_parser('policy-self-study', help='模型自测、核对教材、记录错题并继续训练')
    p.add_argument('--model', required=True, help='当前策略模型')
    p.add_argument('--data', required=True, help='与当前模型对应的旧训练标注')
    p.add_argument('--book', required=True, help='有标准答案的独立教材 JSONL')
    p.add_argument('--out', required=True, help='新模型路径；拒绝覆盖')
    p.add_argument('--data-out', required=True, help='累积标注路径；拒绝覆盖')
    p.add_argument('--eval', help='可选的独立保护集；退步时不生成新模型')
    p.add_argument('--allow-candidate', action='store_true', help='实验性自测候选模型')
    p.set_defaults(func=cmd_policy_self_study)
    p = sub.add_parser('policy-eval', help='在独立标注集上评估首步选边与弃权')
    p.add_argument('--model', required=True, help='训练生成的模型 JSON')
    p.add_argument('--data', required=True, help='独立评估集 JSONL')
    p.set_defaults(func=cmd_policy_eval)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
