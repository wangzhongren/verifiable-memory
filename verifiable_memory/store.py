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

STATE_FORMAT = 'verifiable_memory_01/state@v1'
# 默认 256：容量烤进状态哈希，创建后不可改；16 是从 tiny_llm_memory_01
# 演示规模继承的旧默认。真成本在证书的全槽哈希表（O(N)/写，更大规模
# 需先做证书瘦身）与 LLM 解析的槽位上下文（索引/别名，梯子第 3 级）。
CAPACITY_DEFAULT = 256

WRITE_OPS = {'teach_fact', 'teach_rule', 'correct_fact', 'correct_rule'}
READ_OPS = {'query_record', 'apply_rule'}
ALL_OPS = WRITE_OPS | READ_OPS

NAME_MAX = 24
CONTENT_MAX = 200
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


def validate_op(op):
    """校验 op 的形状与词表白名单（与状态无关的部分）。

    合法时原样返回 op；否则抛 StoreError。存在性、容量、种类等
    状态相关检查由 Store 在执行时做。
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
        content = op.get('content')
        if not isinstance(content, str) or not (1 <= len(content) <= CONTENT_MAX):
            raise StoreError(f'content 必须是 1–{CONTENT_MAX} 字的字符串')
    elif kind in ('teach_rule', 'correct_rule'):
        program = op.get('program')
        if (not isinstance(program, list) or not (1 <= len(program) <= PROGRAM_MAX)
                or not all(step in data.OPS for step in program)):
            raise StoreError(f'program 必须是 1–{PROGRAM_MAX} 个原语名的列表（允许: {data.OPS}）')
    elif kind == 'apply_rule':
        if not data.is_input_string(op.get('input')):
            raise StoreError(f'input 必须是 {data.MIN_LEN}–{data.MAX_LEN} 位、字符属 0–7 的数字串')
    return op


class Store:
    """16（默认）个命名槽位。状态完全由槽位字典决定。"""

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

        if kind in ('teach_fact', 'teach_rule'):
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
            want_kind = 'fact' if kind == 'correct_fact' else 'rule'
            if existing['kind'] != want_kind:
                raise StoreError(f'种类不符：{name} 是 {existing["kind"]}，不能用 {kind} 纠错')
            revision = existing['revision'] + 1
            created = False
            target_hash_before = state_digest(existing)

        if kind in ('teach_fact', 'correct_fact'):
            record = {'kind': 'fact', 'content': op['content'], 'written_by': op_id,
                      'revision': revision, 'utterance': utterance}
        else:
            record = {'kind': 'rule', 'program': list(op['program']), 'written_by': op_id,
                      'revision': revision, 'utterance': utterance}
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
            else:
                result['program'] = record['program']
                result['answer'] = '→'.join(record['program'])
        else:  # apply_rule
            if record['kind'] != 'rule':
                raise StoreError(f'{name} 是事实记录，不能作为规则执行')
            from . import executor
            run = executor.execute(record['program'], op['input'])
            result = {'name': name, 'kind': 'rule', 'input': op['input'],
                      'written_by': record['written_by'], 'revision': record['revision'],
                      'program': record['program'], 'backend': run['backend'],
                      'trace': run['trace'], 'answer': run['answer']}

        result['slot_hash'] = state_digest(record)
        result['state_hash'] = self.state_hash()
        return result
