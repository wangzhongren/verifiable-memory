"""小型可训练分支控制器；只提议路径，数值运算由向量内核核验。

标准库 softmax 分类器按当前实体过滤候选有向边，再对查询字、二元字组
和当前实体状态打分。模型不能创造新边，也不保证语义选择正确。
"""

import hashlib
import math
import random

from . import vectors
from .store import (canonical_json, normalize_override_query, override_name,
                    state_digest)

FORMAT = 'verifiable_memory/branch_policy@v1'
STOP = '__stop__'
EPOCHS = 120
UPDATE_EPOCHS = 200
LEARNING_RATE = 0.18


class PolicyError(ValueError):
    pass


class PolicyRegressionError(PolicyError):
    """候选训练未通过保留集或反馈的验收门槛。"""


def active_edges(slots, source):
    return sorted(name for name, rec in slots.items()
                  if rec['kind'] == 'edge' and rec['source'] == source
                  and vectors.active_edge(name, rec, slots))


def graph_signature(slots):
    """把可用边及其全部依赖记录绑定到训练模型。"""
    anchors = []
    for name, rec in sorted(slots.items()):
        if rec['kind'] == 'edge' and vectors.active_edge(name, rec, slots):
            anchors.append({'edge': name, 'edge_hash': state_digest(rec),
                            'source_hash': state_digest(slots[rec['source']]),
                            'action_hash': state_digest(slots[rec['action']]),
                            'target_hash': state_digest(slots[rec['target']])})
    return state_digest({'format': 'vector-graph@v1', 'anchors': anchors})


def override_active(record, slots):
    if record['kind'] != 'policy_override' or not record['enabled']:
        return False
    source = slots.get(record['source'])
    if source is None or source['kind'] != 'entity':
        return False
    if (source['revision'] != record['source_revision']
            or record['graph_signature'] != graph_signature(slots)):
        return False
    if record['edge'] is None:
        return True
    edge = slots.get(record['edge'])
    return (edge is not None and edge['kind'] == 'edge'
            and edge['source'] == record['source']
            and vectors.active_edge(record['edge'], edge, slots))


def features(query, source, slots):
    if not isinstance(query, str) or not 1 <= len(query) <= 500:
        raise PolicyError('query 必须是 1–500 字的字符串')
    entity = slots.get(source)
    if entity is None or entity['kind'] != 'entity':
        raise PolicyError(f'缺少源实体：{source}')
    cleaned = ''.join(query.lower().split())
    items = {'source:' + source}
    items.update('char:' + char for char in cleaned)
    items.update('pair:' + cleaned[i:i + 2] for i in range(len(cleaned) - 1))
    for index, value in enumerate(entity['vector_units'][:16]):
        sign = 'positive' if value > 0 else 'negative' if value < 0 else 'zero'
        items.add(f'vec:{index}:{sign}')
    return sorted(items)


def _choices(source, slots, visited=None):
    edges = active_edges(slots, source)
    if visited is not None:
        edges = [name for name in edges if slots[name]['target'] not in visited]
    return [STOP] + edges


def _probabilities(model, query, source, slots, visited=None):
    candidates = _choices(source, slots, visited)
    feats = features(query, source, slots)
    scores = {}
    for choice in candidates:
        row = model['weights'].get(choice, {})
        scores[choice] = model['bias'].get(choice, 0) + sum(
            row.get(feature, 0) for feature in feats) / max(1, len(feats))
    largest = max(scores.values())
    exps = {name: math.exp(score - largest) for name, score in scores.items()}
    denominator = sum(exps.values())
    return scores, {name: exps[name] / denominator for name in candidates}


def training_digest(rows):
    return hashlib.sha256(canonical_json(rows).encode('utf-8')).hexdigest()


def merge_feedback(rows, feedback):
    """同一源实体/原话的新标签替换旧标签，避免互相矛盾的监督。"""
    if not isinstance(feedback, list) or not feedback:
        raise PolicyError('反馈必须是非空 JSONL')
    merged, index = [], {}
    for position, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {'source', 'query', 'edge'}:
            raise PolicyError(f'旧标注第 {position} 行格式不符')
        if not isinstance(row['source'], str) or not isinstance(row['query'], str):
            raise PolicyError(f'旧标注第 {position} 行 source/query 必须是字符串')
        key = (row['source'], row['query'])
        if key in index:
            merged[index[key]] = dict(row)
        else:
            index[key] = len(merged)
            merged.append(dict(row))
    replaced = 0
    for position, row in enumerate(feedback, 1):
        if not isinstance(row, dict) or set(row) != {'source', 'query', 'edge'}:
            raise PolicyError(f'反馈第 {position} 行需要 source、query、edge 三个字段')
        if not isinstance(row['source'], str) or not isinstance(row['query'], str):
            raise PolicyError(f'反馈第 {position} 行 source/query 必须是字符串')
        key = (row['source'], row['query'])
        if key in index:
            merged[index[key]] = dict(row)
            replaced += 1
        else:
            index[key] = len(merged)
            merged.append(dict(row))
    return merged, replaced


def train(rows, slots, *, initial_model=None, parent_model_sha256=None):
    if not isinstance(rows, list) or not rows:
        raise PolicyError('训练集必须是非空 JSONL')
    if not any(rec['kind'] == 'edge' and vectors.active_edge(name, rec, slots)
               for name, rec in slots.items()):
        raise PolicyError('训练前至少需要一条有效的有向边')
    all_edges = sorted(name for name, rec in slots.items()
                       if rec['kind'] == 'edge' and vectors.active_edge(name, rec, slots))
    if STOP in all_edges:
        raise PolicyError(f'边名 {STOP} 为控制器保留名称')
    samples = []
    positive_edges, abstain_sources = set(), set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {'source', 'query', 'edge'}:
            raise PolicyError(f'第 {index} 行需要 source、query、edge 三个字段')
        source, query, target = row['source'], row['query'], row['edge']
        if not isinstance(source, str) or not isinstance(query, str):
            raise PolicyError(f'第 {index} 行的 source / query 必须是字符串')
        options = _choices(source, slots)
        label = STOP if target is None else target
        if label not in options:
            raise PolicyError(f'第 {index} 行的标注边不是源实体的有效出边：{target}')
        if label == STOP:
            abstain_sources.add(source)
        else:
            positive_edges.add(label)
        samples.append((options, label, features(query, source, slots)))
    untrained = set(all_edges) - positive_edges
    if untrained:
        raise PolicyError(f'训练集缺少这些有向边的正例：{sorted(untrained)}')
    sources = {slots[name]['source'] for name in all_edges}
    missing_abstain = sources - abstain_sources
    if missing_abstain:
        raise PolicyError(f'训练集缺少这些源实体的弃权样本：{sorted(missing_abstain)}')
    if initial_model is None:
        weights = {name: {} for name in [STOP] + all_edges}
        bias = {name: 0.0 for name in weights}
        epochs, generation = EPOCHS, 1
    else:
        validate_model(initial_model, slots)
        if not isinstance(parent_model_sha256, str) or len(parent_model_sha256) != 64:
            raise PolicyError('增量更新必须记录上一版模型 SHA-256')
        weights = {name: dict(initial_model['weights'][name]) for name in [STOP] + all_edges}
        bias = {name: float(initial_model['bias'][name]) for name in weights}
        epochs = UPDATE_EPOCHS
        generation = initial_model.get('generation', 1) + 1
    order = list(range(len(samples)))
    rng = random.Random(7 if initial_model is None else 7 + generation)
    for _ in range(epochs):
        rng.shuffle(order)
        for index in order:
            options, label, feats = samples[index]
            scores = {name: bias[name] + sum(weights[name].get(f, 0) for f in feats)
                      / max(1, len(feats)) for name in options}
            largest = max(scores.values())
            exps = {name: math.exp(score - largest) for name, score in scores.items()}
            total = sum(exps.values())
            for name in options:
                gradient = (1.0 if name == label else 0.0) - exps[name] / total
                bias[name] += LEARNING_RATE * gradient
                step = LEARNING_RATE * gradient / max(1, len(feats))
                for feature in feats:
                    weights[name][feature] = weights[name].get(feature, 0.0) + step
    model = {'format': FORMAT, 'graph_signature': graph_signature(slots),
             'training_sha256': training_digest(rows),
             'n_examples': len(rows), 'epochs': epochs,
             'generation': generation,
             'parent_model_sha256': parent_model_sha256,
             'promotion': 'initial' if initial_model is None else 'candidate',
             'minimum_probability': 0.55, 'minimum_margin': 0.08,
             'weights': {name: {key: round(value, 8) for key, value in row.items()
                                if abs(value) > 1e-8}
                         for name, row in weights.items()},
             'bias': {name: round(value, 8) for name, value in bias.items()}}
    return model


def validate_model(model, slots):
    if not isinstance(model, dict) or model.get('format') != FORMAT:
        raise PolicyError('模型格式不符')
    if model.get('graph_signature') != graph_signature(slots):
        raise PolicyError('实体/动作/有向边已变化，模型已过期，请重新训练')
    if not isinstance(model.get('weights'), dict) or not isinstance(model.get('bias'), dict):
        raise PolicyError('模型缺少参数')
    if 'generation' in model and (type(model['generation']) is not int
                                  or model['generation'] < 1):
        raise PolicyError('模型 generation 非法')
    if model.get('promotion', 'initial') not in ('initial', 'candidate', 'evaluated'):
        raise PolicyError('模型 promotion 非法')
    expected = {STOP} | {name for name, rec in slots.items()
                         if rec['kind'] == 'edge' and vectors.active_edge(name, rec, slots)}
    if set(model['weights']) != expected or set(model['bias']) != expected:
        raise PolicyError('模型动作类别与当前有向图不一致')
    for field in ('minimum_probability', 'minimum_margin'):
        value = model.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise PolicyError(f'模型 {field} 非法')
    for name, row in model['weights'].items():
        if not isinstance(row, dict):
            raise PolicyError(f'模型权重格式非法：{name}')
        if type(model['bias'][name]) not in (int, float) or not math.isfinite(model['bias'][name]):
            raise PolicyError(f'模型偏置非法：{name}')
        if any(not isinstance(key, str) or type(value) not in (int, float)
               or not math.isfinite(value) for key, value in row.items()):
            raise PolicyError(f'模型权重非法：{name}')


def route(model, query, source, slots, max_hops=8, *, use_overrides=True):
    validate_model(model, slots)
    if type(max_hops) is not int or not 1 <= max_hops <= 16:
        raise PolicyError('max_hops 必须是 1–16 的整数')
    current, visited, path, decisions = source, {source}, [], []
    features(query, source, slots)  # 即使没有候选边也验证输入和实体。
    for _ in range(max_hops):
        if use_overrides:
            name = override_name(current, query)
            correction = slots.get(name)
            if correction is not None:
                if (correction['kind'] != 'policy_override'
                        or correction['source'] != current
                        or correction['query'] != normalize_override_query(query)):
                    raise PolicyError('人工纠错名称冲突')
                if correction['enabled']:
                    if not override_active(correction, slots):
                        return {'target': current, 'path': path,
                                'decisions': decisions, 'abstained': True,
                                'reason': 'stale_override'}
                    choice = correction['edge']
                    if choice is not None and slots[choice]['target'] in visited:
                        choice = None
                        stop_reason = 'cycle_limit'
                    else:
                        stop_reason = 'confirmed_abstain'
                    decisions.append({'source': current, 'candidates': [],
                                      'choice': choice, 'confidence': 1.0,
                                      'override': name,
                                      'override_revision': correction['revision'],
                                      'override_written_by': correction['written_by'],
                                      'override_hash': state_digest(correction)})
                    if choice is None:
                        return {'target': current, 'path': path,
                                'decisions': decisions, 'abstained': True,
                                'reason': stop_reason}
                    path.append(choice)
                    current = slots[choice]['target']
                    visited.add(current)
                    if correction['stop_after']:
                        return {'target': current, 'path': path,
                                'decisions': decisions, 'abstained': False,
                                'reason': 'override_stop'}
                    continue
        all_outgoing = active_edges(slots, current)
        if not all_outgoing:
            reason = 'leaf' if path else 'no_edges'
            return {'target': current, 'path': path, 'decisions': decisions,
                    'abstained': not bool(path), 'reason': reason}
        if not any(slots[name]['target'] not in visited for name in all_outgoing):
            return {'target': current, 'path': path, 'decisions': decisions,
                    'abstained': True, 'reason': 'cycle_limit'}
        scores, probs = _probabilities(model, query, current, slots, visited)
        order = sorted(probs, key=lambda name: (-probs[name], name))
        best, runner_up = order[0], order[1] if len(order) > 1 else STOP
        confidence = probs[best]
        if (best == STOP or confidence < model['minimum_probability']
                or confidence - probs[runner_up] < model['minimum_margin']):
            selected = None
        else:
            selected = best
        decisions.append({'source': current,
                          'candidates': [{'edge': name, 'score': round(scores[name], 6),
                                          'probability': round(probs[name], 6)}
                                         for name in sorted(probs)],
                          'choice': selected, 'confidence': round(confidence, 6)})
        if selected is None:
            return {'target': current, 'path': path, 'decisions': decisions,
                    'abstained': True, 'reason': 'uncertain'}
        path.append(selected)
        current = slots[selected]['target']
        visited.add(current)
    return {'target': current, 'path': path, 'decisions': decisions,
            'abstained': bool(active_edges(slots, current)),
            'reason': 'max_hops' if active_edges(slots, current) else 'leaf'}


def evaluate(model, rows, slots):
    """用独立标注集测首步分支；不把训练正确率当成泛化能力。"""
    validate_model(model, slots)
    if not isinstance(rows, list) or not rows:
        raise PolicyError('评估集必须是非空 JSONL')
    details = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {'source', 'query', 'edge'}:
            raise PolicyError(f'评估第 {index} 行需要 source、query、edge 三个字段')
        prediction = route(model, row['query'], row['source'], slots,
                           max_hops=1, use_overrides=False)
        predicted = (prediction['decisions'][0]['choice']
                     if prediction['decisions'] else None)
        details.append({'source': row['source'], 'query': row['query'],
                        'expected': row['edge'], 'predicted': predicted,
                        'correct': predicted == row['edge']})
    return {'n_examples': len(details),
            'correct': sum(item['correct'] for item in details),
            'abstentions': sum(item['predicted'] is None for item in details),
            'rows': details}


def check_book(rows, slots):
    """教材给出首步标准答案；可另声明是否在该步停止。"""
    if not isinstance(rows, list) or not rows:
        raise PolicyError('教材必须是非空 JSONL')
    checked, seen = [], {}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) not in (
                {'source', 'query', 'edge'},
                {'source', 'query', 'edge', 'stop_after'}):
            raise PolicyError(f'教材第 {index} 行需要 source、query、edge 和可选 stop_after')
        if type(row.get('stop_after', True)) is not bool:
            raise PolicyError(f'教材第 {index} 行 stop_after 必须是布尔值')
        source, query, edge = row['source'], row['query'], row['edge']
        if not isinstance(source, str) or not isinstance(query, str):
            raise PolicyError(f'教材第 {index} 行 source/query 必须是字符串')
        features(query, source, slots)
        if edge is not None and edge not in active_edges(slots, source):
            raise PolicyError(f'教材第 {index} 行标注边无效：{edge}')
        if edge is None and row.get('stop_after') is False:
            raise PolicyError(f'教材第 {index} 行弃权不能继续路由')
        key = (source, normalize_override_query(query))
        label = (edge, row.get('stop_after', True))
        if key in seen:
            if seen[key] != label:
                raise PolicyError(f'教材第 {index} 行与前面的同一原话标签冲突')
            continue
        seen[key] = label
        checked.append({'source': source, 'query': query, 'edge': edge,
                        'stop_after': row.get('stop_after', True)})
    return checked


def learn(previous, old_rows, feedback, slots, parent_hash, holdout=None):
    """带历史回放和独立评估门槛的权重更新；不写文件或操作日志。"""
    validate_model(previous, slots)
    if (previous.get('training_sha256') != training_digest(old_rows)
            or previous.get('n_examples') != len(old_rows)):
        raise PolicyError('旧标注数据与模型记录的训练摘要不符')
    merged, replaced = merge_feedback(old_rows, feedback)
    candidate = train(merged, slots, initial_model=previous,
                      parent_model_sha256=parent_hash)
    feedback_labels, _ = merge_feedback([], feedback)
    changed_keys = {(row['source'], row['query']) for row in feedback_labels}
    retained = [row for row in old_rows
                if (row['source'], row['query']) not in changed_keys]
    old_retained = evaluate(previous, retained, slots)['correct'] if retained else 0
    new_retained = evaluate(candidate, retained, slots)['correct'] if retained else 0
    old_feedback = evaluate(previous, feedback_labels, slots)['correct']
    new_feedback = evaluate(candidate, feedback_labels, slots)['correct']
    if new_retained < old_retained:
        raise PolicyRegressionError(f'旧标注退步：{old_retained}→{new_retained} / '
                                    f'{len(retained)}，拒绝更新')
    if (new_feedback < old_feedback or
            (old_feedback < len(feedback_labels) and new_feedback == old_feedback)):
        raise PolicyRegressionError(f'反馈未改善或出现退步：{old_feedback}→{new_feedback} / '
                                    f'{len(feedback_labels)}，拒绝更新')
    eval_summary = None
    if holdout is not None:
        prior_eval, new_eval = evaluate(previous, holdout, slots), evaluate(candidate, holdout, slots)
        before, after = prior_eval['correct'], new_eval['correct']
        regressed = sum(first['correct'] and not second['correct']
                        for first, second in zip(prior_eval['rows'], new_eval['rows']))
        if regressed:
            raise PolicyRegressionError(
                f'独立评估退步：{regressed} 条原本正确的样本变错，'
                f'总体 {before}→{after} / {len(holdout)}，拒绝更新')
        eval_summary = {'before': before, 'after': after, 'n_examples': len(holdout)}
        candidate['promotion'] = 'evaluated'
        candidate['evaluation_sha256'] = training_digest(holdout)
    stats = {'generation': candidate['generation'],
             'parent_model_sha256': parent_hash,
             'n_examples': len(merged), 'feedback_rows': len(feedback_labels),
             'replaced_labels': replaced,
             'retained': {'before': old_retained, 'after': new_retained,
                          'n_examples': len(retained)},
             'feedback': {'before': old_feedback, 'after': new_feedback,
                          'n_examples': len(feedback_labels)},
             'independent_eval': eval_summary}
    return candidate, merged, stats
