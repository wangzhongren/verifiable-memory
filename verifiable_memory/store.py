"""记忆核心：命名槽位、精确寻址、结构化操作。

设计纪律（来自八轮实验的结论，见 PROTOCOL.md）：
- LLM 在验证边界上游：本模块只见已通过校验的结构化 op，不见自然
  语言，因此重放不依赖 LLM、逐位确定。
- 显式寻址：名称即槽位键，精确匹配；纠错 = 同槽覆盖。
  不做任何学习型软寻址。
- 语义：teach_* 只允许新名称；correct_* 只允许已存在的同名同种类
  记录；读操作不改变状态。

哈希公共面：verify.py 与 replay.py 复用本文件的 canonical_json /
sha256_string（哈希函数必须一致结果才可比，这是刻意保留的最小公
共面）；除此之外的一切核验逻辑在 verify.py 独立实现。
"""

import hashlib
import json

from . import data
from . import vectors

STATE_FORMAT = 'verifiable_memory_01/state@v1'
# 默认 256，建库时可指定更大的正整数；容量写入状态哈希，创建后固定。
# 实际规模受内存、磁盘、O(N) 状态哈希与解析上下文成本限制。
CAPACITY_DEFAULT = 256

WRITE_OPS = {'teach_fact', 'teach_rule', 'correct_fact', 'correct_rule',
             'teach_entity', 'correct_entity',
             'teach_vector_action', 'correct_vector_action', 'link_entities'}
READ_OPS = {'query_record', 'apply_rule', 'apply_vector_action', 'derive_entities'}
ALL_OPS = WRITE_OPS | READ_OPS

NAME_MAX = 24
CONTENT_MAX = 500
PROGRAM_MAX = 8


class StoreError(Exception):
    """结构化操作被拒绝（形状、白名单、存在性、容量或种类不符）。"""


def canonical_json(obj):
    """确定性序列化：键排序、无空白、保留中文。任何加入状态的对象
    都必须能经此序列化（即只含 JSON 原生类型）。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def sha256_string(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def state_digest(obj):
    return sha256_string(canonical_json(obj))


def validate_op(op, *, content_max=None):
    """校验 op 的形状与词表白名单（与状态无关的部分）。

    合法时原样返回 op；否则抛 StoreError。存在性、容量、种类等
    状态相关检查由 Store 在执行时做。content_max 仅供重放旧验证规则；
    所有新操作使用当前 CONTENT_MAX，操作内容不能自行指定上限。
    """
    if not isinstance(op, dict):
        raise StoreError('op 必须是对象')
    kind = op.get('op')
    if not isinstance(kind, str) or kind not in ALL_OPS:
        raise StoreError(f'未知操作类型: {kind!r}（允许: {sorted(ALL_OPS)}）')
    name = op.get('name')
    if not isinstance(name, str) or not (1 <= len(name) <= NAME_MAX):
        raise StoreError(f'name 必须是 1–{NAME_MAX} 字的字符串')
    if kind in ('teach_fact', 'correct_fact'):
        limit = CONTENT_MAX if content_max is None else content_max
        content = op.get('content')
        if not isinstance(content, str) or not (1 <= len(content) <= limit):
            raise StoreError(f'content 必须是 1–{limit} 字的字符串')
    elif kind in ('teach_rule', 'correct_rule'):
        program = op.get('program')
        if (not isinstance(program, list) or not (1 <= len(program) <= PROGRAM_MAX)
                or not all(step in data.OPS for step in program)):
            raise StoreError(f'program 必须是 1–{PROGRAM_MAX} 个原语名的列表（允许: {data.OPS}）')
    elif kind == 'apply_rule':
        if not data.is_input_string(op.get('input')):
            raise StoreError(f'input 必须是 {data.MIN_LEN}–{data.MAX_LEN} 位、字符属 0–7 的数字串')
    elif kind in ('teach_entity', 'correct_entity'):
        try:
            vectors.quantize(op.get('vector'))
        except vectors.VectorError as exc:
            raise StoreError(str(exc)) from exc
    elif kind in ('teach_vector_action', 'correct_vector_action'):
        try:
            vectors.quantize(op.get('delta'), 'delta')
        except vectors.VectorError as exc:
            raise StoreError(str(exc)) from exc
    elif kind == 'link_entities':
        for field in ('source', 'action', 'target'):
            value = op.get(field)
            if not isinstance(value, str) or not 1 <= len(value) <= NAME_MAX:
                raise StoreError(f'{field} 必须是 1–{NAME_MAX} 字的名称')
    elif kind in ('apply_vector_action', 'derive_entities'):
        if not isinstance(op.get('source'), str) or not 1 <= len(op['source']) <= NAME_MAX:
            raise StoreError(f'source 必须是 1–{NAME_MAX} 字的实体名称')
        if kind == 'derive_entities':
            hops = op.get('max_hops', 8)
            if type(hops) is not int or not 1 <= hops <= 16:
                raise StoreError('max_hops 必须是 1–16 的整数')
    return op


class Store:
    """容量可配置的命名槽位，默认 256。状态完全由槽位字典决定。"""

    def __init__(self, capacity=CAPACITY_DEFAULT):
        if not isinstance(capacity, int) or capacity < 1:
            raise StoreError('capacity 必须是正整数')
        self.capacity = capacity
        self.slots = {}  # name -> record dict；插入顺序即写入顺序

    # ---- 状态与哈希 ----

    def state(self):
        return {'format': STATE_FORMAT, 'capacity': self.capacity, 'slots': self.slots}

    def state_hash(self):
        return state_digest(self.state())

    def slot_hashes(self):
        return {name: state_digest(rec) for name, rec in self.slots.items()}

    def known_slots(self):
        return [{'name': n, 'kind': rec['kind']} for n, rec in self.slots.items()]

    # ---- 写 ----

    def apply_write(self, op, *, op_id, utterance):
        """执行写操作并返回零附带损害证书（v2，O(1) 大小）。要求 op
        已通过 validate_op。

        任何检查失败都在改动之前抛出 StoreError——错误操作绝不半途
        改状态，这是重放能逐位复现的前提。

        证书语义（v2）：zero_collateral 是**写入时的断言**——本方法
        只赋值 self.slots[name] 一个键，其余槽不可能被触碰，故断言由
        构造保证、O(1) 产出；它的独立证实由 verify.py 重放重建全槽
        diff 完成（v1 的写时全表计算已删除：32k 槽下 O(N)/写不可行）。
        """
        from . import proof
        kind = op['op']
        name = op['name']
        before_state_hash = self.state_hash()
        existing = self.slots.get(name)

        if kind.startswith('teach_') or kind == 'link_entities':
            if existing is not None:
                raise StoreError(f'名称已存在：{name}（更新请用 correct_*）')
            if len(self.slots) >= self.capacity:
                raise StoreError(f'槽位已满（容量 {self.capacity}），无法新增 {name}')
            revision = 1
            created = True
            target_hash_before = None
        else:
            if existing is None:
                known = '、'.join(sorted(self.slots)) or '（空）'
                raise StoreError(f'名称不存在：{name}（已知槽位: {known}）')
            want_kind = {'correct_fact': 'fact', 'correct_rule': 'rule',
                         'correct_entity': 'entity',
                         'correct_vector_action': 'vector_action'}[kind]
            if existing['kind'] != want_kind:
                raise StoreError(f'种类不符：{name} 是 {existing["kind"]}，不能用 {kind} 纠错')
            revision = existing['revision'] + 1
            created = False
            target_hash_before = state_digest(existing)

        record = {'kind': ('fact' if kind.endswith('_fact') else
                           'rule' if kind.endswith('_rule') else
                           'entity' if kind.endswith('_entity') else
                           'vector_action' if kind.endswith('_vector_action') else 'edge'),
                  'written_by': op_id, 'revision': revision, 'utterance': utterance}
        if record['kind'] == 'fact':
            record['content'] = op['content']
        elif record['kind'] == 'rule':
            record['program'] = list(op['program'])
        elif record['kind'] == 'entity':
            record['vector_units'] = vectors.quantize(op['vector'])
            record['vector_scale'] = vectors.SCALE
        elif record['kind'] == 'vector_action':
            record['delta_units'] = vectors.quantize(op['delta'], 'delta')
            record['vector_scale'] = vectors.SCALE
        else:
            refs = {}
            for key, want in (('source', 'entity'), ('action', 'vector_action'),
                              ('target', 'entity')):
                ref = self.slots.get(op[key])
                if ref is None or ref['kind'] != want:
                    raise StoreError(f'{key} 必须引用已有的 {want}：{op[key]}')
                refs[key + '_revision'] = ref['revision']
            a = self.slots[op['source']]['vector_units']
            delta = self.slots[op['action']]['delta_units']
            b = self.slots[op['target']]['vector_units']
            if len(a) != len(delta) or len(a) != len(b):
                raise StoreError('实体与动作向量维度不一致')
            if any(x + d != y for x, d, y in zip(a, delta, b)):
                raise StoreError('目标向量不等于源向量加动作增量')
            record.update({key: op[key] for key in ('source', 'action', 'target')})
            record.update(refs)
        self.slots[name] = record

        return proof.write_certificate(
            target=name, created=created, zero_collateral=True,
            changed_slots=[name], target_hash_before=target_hash_before,
            target_hash_after=state_digest(record),
            before_state_hash=before_state_hash,
            after_state_hash=self.state_hash())

    # ---- 读 ----

    def read(self, op):
        """执行读操作，返回带证据链的结果。不改变任何状态。"""
        kind = op['op']
        name = op['name']
        record = self.slots.get(name)
        if record is None:
            known = '、'.join(sorted(self.slots)) or '（空）'
            raise StoreError(f'名称不存在：{name}（已知槽位: {known}）')

        if kind == 'query_record':
            result = {'name': name, 'kind': record['kind'],
                      'written_by': record['written_by'], 'revision': record['revision']}
            if record['kind'] == 'fact':
                result['content'] = record['content']
                result['answer'] = record['content']
            elif record['kind'] == 'rule':
                result['program'] = record['program']
                result['answer'] = '→'.join(record['program'])
            elif record['kind'] == 'entity':
                result['vector'] = vectors.display(record['vector_units'])
                result['answer'] = result['vector']
            elif record['kind'] == 'vector_action':
                result['delta'] = vectors.display(record['delta_units'])
                result['answer'] = result['delta']
            else:
                result['source'], result['action'], result['target'] = (
                    record['source'], record['action'], record['target'])
                result['active'] = vectors.active_edge(name, record, self.slots)
                result['answer'] = f"{record['source']} --{record['action']}--> {record['target']}"
        elif kind == 'apply_rule':
            if record['kind'] != 'rule':
                raise StoreError(f'{name} 是事实记录，不能作为规则执行')
            from . import executor
            run = executor.execute(record['program'], op['input'])
            result = {'name': name, 'kind': 'rule', 'input': op['input'],
                      'written_by': record['written_by'], 'revision': record['revision'],
                      'program': record['program'], 'backend': run['backend'],
                      'trace': run['trace'], 'answer': run['answer']}
        elif kind == 'apply_vector_action':
            if record['kind'] != 'vector_action':
                raise StoreError(f'{name} 不是向量动作')
            entity = self.slots.get(op['source'])
            if entity is None or entity['kind'] != 'entity':
                raise StoreError(f'source 必须引用已有实体：{op["source"]}')
            source_units, delta = entity['vector_units'], record['delta_units']
            if len(source_units) != len(delta):
                raise StoreError('实体与动作向量维度不一致')
            output_units = [x + d for x, d in zip(source_units, delta)]
            matches = sorted(n for n, rec in self.slots.items()
                             if rec['kind'] == 'entity' and rec['vector_units'] == output_units)
            result = {'name': name, 'kind': 'vector_action',
                      'written_by': record['written_by'], 'revision': record['revision'],
                      'source': op['source'], 'source_revision': entity['revision'],
                      'input_vector': vectors.display(source_units),
                      'delta': vectors.display(delta),
                      'output_vector': vectors.display(output_units),
                      'matches': matches, 'answer': matches}
        else:  # derive_entities：name 为目标实体
            if record['kind'] != 'entity':
                raise StoreError(f'{name} 不是实体')
            source = self.slots.get(op['source'])
            if source is None or source['kind'] != 'entity':
                raise StoreError(f'source 必须引用已有实体：{op["source"]}')
            try:
                path = vectors.derive(op['source'], name, self.slots, op.get('max_hops', 8))
            except vectors.VectorError as exc:
                raise StoreError(str(exc)) from exc
            trace = []
            for edge_name in path:
                edge = self.slots[edge_name]
                left, action, right = (self.slots[edge['source']],
                                       self.slots[edge['action']],
                                       self.slots[edge['target']])
                trace.append({'edge': edge_name, 'edge_revision': edge['revision'],
                              'edge_written_by': edge['written_by'],
                              'edge_hash': state_digest(edge),
                              'source': edge['source'], 'source_revision': left['revision'],
                              'source_written_by': left['written_by'],
                              'source_hash': state_digest(left),
                              'action': edge['action'], 'action_revision': action['revision'],
                              'action_written_by': action['written_by'],
                              'action_hash': state_digest(action),
                              'target': edge['target'], 'target_revision': right['revision'],
                              'target_written_by': right['written_by'],
                              'target_hash': state_digest(right),
                              'before': vectors.display(left['vector_units']),
                              'delta': vectors.display(action['delta_units']),
                              'after': vectors.display(right['vector_units'])})
            result = {'name': name, 'kind': 'entity_derivation',
                      'written_by': record['written_by'], 'revision': record['revision'],
                      'source': op['source'], 'source_revision': source['revision'],
                      'trace': trace, 'vector': vectors.display(record['vector_units']),
                      'answer': name}

        result['slot_hash'] = state_digest(record)
        result['state_hash'] = self.state_hash()
        return result
