"""有向实体运算：向量量化、增量动作和版本锚定的路径推导。

每个坐标按 1e-6 固定点量化，执行与重放只做整数加法；输入的
浮点/整数不作为状态格式。近似向量相似度与自动语义匹配不在这里。
"""

from collections import deque
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN

SCALE = 1_000_000
MAX_DIM = 4096
MAX_ABS = 1_000_000


class VectorError(ValueError):
    pass


def quantize(values, field='vector'):
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_DIM:
        raise VectorError(f'{field} 必须是 1–{MAX_DIM} 维数值列表')
    units = []
    for value in values:
        if type(value) not in (int, float):
            raise VectorError(f'{field} 只允许有限的数值坐标')
        try:
            decimal = Decimal(str(value))
            if not decimal.is_finite() or abs(decimal) > MAX_ABS:
                raise VectorError(f'{field} 坐标必须为有限值且绝对值不超过 {MAX_ABS}')
            units.append(int((decimal * SCALE).to_integral_value(rounding=ROUND_HALF_EVEN)))
        except InvalidOperation as exc:
            raise VectorError(f'{field} 坐标无法量化') from exc
    return units


def display(units):
    """用户可读的十进制坐标，避免浮点算术参与推导。"""
    return [str(Decimal(value) / SCALE) for value in units]


def active_edge(name, edge, slots):
    if edge['kind'] != 'edge':
        return False
    for key, kind in (('source', 'entity'), ('action', 'vector_action'),
                      ('target', 'entity')):
        record = slots.get(edge[key])
        if record is None or record['kind'] != kind:
            return False
        if record['revision'] != edge[key + '_revision']:
            return False
    a, op, b = (slots[edge['source']]['vector_units'],
                slots[edge['action']]['delta_units'],
                slots[edge['target']]['vector_units'])
    return len(a) == len(op) == len(b) and all(x + d == y for x, d, y in zip(a, op, b))


def derive(source, target, slots, max_hops):
    """稳定的最短有向路径；同长度时按边名排序。只走当前有效的边。"""
    edges = sorted(((name, rec) for name, rec in slots.items() if rec['kind'] == 'edge'),
                   key=lambda item: item[0])
    queue = deque([(source, [])])
    seen = {source}
    while queue:
        current, path = queue.popleft()
        if current == target:
            return path
        if len(path) >= max_hops:
            continue
        for name, edge in edges:
            next_name = edge['target']
            if edge['source'] == current and next_name not in seen and active_edge(name, edge, slots):
                seen.add(next_name)
                queue.append((next_name, path + [name]))
    raise VectorError(f'没有可用的有向路径：{source}→{target}（边可能因修订过期）')
