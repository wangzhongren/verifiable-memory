"""执行器接口与符号后端。

执行器把已存程序（原语名称列表）真正跑到数字串上。接口刻意做薄：
execute(program, s) -> {"backend", "answer", "trace"}。

第二阶段将注册 neural 后端（加载 tiny_rule_learning_02_modular 的冻
结检查点，经第7轮控制器读出程序后执行）；本阶段只有 symbolic，
它直接调用 data.transform 参考实现，并保留逐步 trace 作为证据。
invalid 是失败语义：未知原语或非法输入一律抛 ExecutorError，宿主
不得代为修复——这与 tiny_rule_learning_07/engine.py 的契约一致。
"""

from . import data

BACKENDS = ('symbolic',)


class ExecutorError(Exception):
    """执行失败（未知原语、非法输入等）。失败不产出答案。"""


def symbolic_execute(program, s):
    if not data.is_input_string(s):
        raise ExecutorError(f'非法输入串: {s!r}（需 {data.MIN_LEN}–{data.MAX_LEN} 位、字符 0–7）')
    trace = []
    current = s
    for i, op in enumerate(program, start=1):
        before = current
        try:
            current = data.transform(before, [op])
        except ValueError as exc:
            raise ExecutorError(str(exc)) from exc
        trace.append({'step': i, 'op': op, 'before': before, 'after': current})
    return {'backend': 'symbolic', 'answer': current, 'trace': trace}


def execute(program, s, backend='symbolic'):
    if backend not in BACKENDS:
        raise ExecutorError(f'未知执行器后端: {backend!r}（已注册: {BACKENDS}；'
                            'neural 后端在第二阶段接入）')
    if not isinstance(program, list) or not program:
        raise ExecutorError('program 必须是非空的原语名列表')
    return symbolic_execute(program, s)
