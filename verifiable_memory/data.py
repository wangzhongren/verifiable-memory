"""四原语参考实现与预声明演示场景。

原语名称与顺序完全继承 tiny_rule_learning_01/data.py：
  OPS = ['反转', '左移一位', '交换前两位', '首位加一']
本文件是参考实现；verify.py 另有第二套独立实现 independent_transform
做交叉核验（两套实现必须一致，不一致即失败）。

SCENARIO 是 PROTOCOL.md 预声明的固定教学/查询/纠错序列，期望答案
在这里按参考实现手工算好并写进 PROTOCOL；checks.py 与 verify.py
都会独立重新验证这些期望值，不允许以本文件的输出为准。
"""

OPS = ['反转', '左移一位', '交换前两位', '首位加一']

DIGIT_CHARS = '01234567'
MIN_LEN = 2   # 交换前两位 需要至少两位
MAX_LEN = 8


def is_input_string(s):
    """合法的执行输入：2–8 位、字符均属 0–7（首位加一按模 8 定义）。"""
    return (isinstance(s, str) and MIN_LEN <= len(s) <= MAX_LEN
            and all(c in DIGIT_CHARS for c in s))


def transform(s, program):
    """把 program（原语名称列表）按顺序应用到数字串 s，返回新数字串。"""
    result = s
    for op in program:
        if op == '反转':
            result = result[::-1]
        elif op == '左移一位':
            result = result[1:] + result[:1]
        elif op == '交换前两位':
            result = result[1] + result[0] + result[2:]
        elif op == '首位加一':
            result = str((int(result[0]) + 1) % 8) + result[1:]
        else:
            raise ValueError(f'未知原语: {op!r}')
    return result


def design_checks():
    """参考实现的行为断言；任何一处失败都应中止一切后续流程。"""
    assert transform('1234', ['反转', '左移一位']) == '3214'
    assert transform('1234', ['左移一位', '反转']) == '1432'
    assert transform('1234', ['交换前两位', '首位加一']) == '3134'
    assert transform('7777', ['首位加一']) == '0777'
    assert transform('5670', ['反转', '左移一位']) == '7650'
    assert transform('5670', ['左移一位', '反转']) == '5076'
    assert transform('4321', ['交换前两位']) == '3421'
    assert is_input_string('1234') and not is_input_string('89') and not is_input_string('1')


def teach_fact(name, content):
    return {'op': 'teach_fact', 'name': name, 'content': content}


def teach_rule(name, program):
    return {'op': 'teach_rule', 'name': name, 'program': list(program)}


def correct_fact(name, content):
    return {'op': 'correct_fact', 'name': name, 'content': content}


def correct_rule(name, program):
    return {'op': 'correct_rule', 'name': name, 'program': list(program)}


def query_record(name):
    return {'op': 'query_record', 'name': name}


def apply_rule(name, input_str):
    return {'op': 'apply_rule', 'name': name, 'input': input_str}


# 预声明的固定演示场景：(类别, 话语, 期望op, 期望答案或None)。
# 期望答案按参考实现手工推得；PROTOCOL.md 冻结同一张表。
# 灵魂样例："对1234应用规则07。"——八轮实验中 text 模型 CLI 在同类
# 定义下答 1432（正确为 3214）；本系统必须在纠错前后分别给出
# 3214 与 1432，并各附完整证据链。
SCENARIO = [
    ('teach', '教事实 实体07：方向是向上', teach_fact('实体07', '方向是向上'), None),
    ('teach', '教事实 实体11：方向是向左', teach_fact('实体11', '方向是向左'), None),
    ('teach', '教事实 实体03：方向是向下', teach_fact('实体03', '方向是向下'), None),
    ('teach', '教事实 实体15：方向是向右', teach_fact('实体15', '方向是向右'), None),
    ('teach', '教规则 规则07：反转，左移一位', teach_rule('规则07', ['反转', '左移一位']), None),
    ('teach', '教规则 规则12：交换前两位，首位加一', teach_rule('规则12', ['交换前两位', '首位加一']), None),
    ('ask', '查询 实体07', query_record('实体07'), '方向是向上'),
    ('ask', '查询 实体11', query_record('实体11'), '方向是向左'),
    ('ask', '对1234应用规则07。', apply_rule('规则07', '1234'), '3214'),
    ('ask', '应用 规则07 5670', apply_rule('规则07', '5670'), '7650'),
    ('ask', '应用 规则12 1234', apply_rule('规则12', '1234'), '3134'),
    ('ask', '应用 规则12 7777', apply_rule('规则12', '7777'), '0777'),
    ('correct', '更正事实 实体11：方向是向下', correct_fact('实体11', '方向是向下'), None),
    ('correct', '更正规则 规则07：左移一位，反转', correct_rule('规则07', ['左移一位', '反转']), None),
    ('ask', '查询 实体11', query_record('实体11'), '方向是向下'),
    ('ask', '查询 实体07', query_record('实体07'), '方向是向上'),
    ('ask', '对1234应用规则07。', apply_rule('规则07', '1234'), '1432'),
    ('ask', '应用 规则07 5670', apply_rule('规则07', '5670'), '5076'),
]

# LLM 解析鲁棒性测试（PROTOCOL 场景7）：10 条同义改写 → 期望 op。
# 这是 parse-only 测试，不写入任何会话；解析上下文 = SCENARIO 前六条
# 教学完成后的六个已知槽位。期望 op 按预声明逐字比较，失败即如实
# 报告，不调参重试。
PARAPHRASES = [
    ('实体07的方向是向上', teach_fact('实体07', '方向是向上')),
    ('把实体11的方向改成向下', correct_fact('实体11', '方向是向下')),
    ('新规则：规则20，步骤是先反转再交换前两位', teach_rule('规则20', ['反转', '交换前两位'])),
    ('对5670应用规则07', apply_rule('规则07', '5670')),
    ('用规则07处理1234', apply_rule('规则07', '1234')),
    ('规则07现在应该改成：先左移一位，再反转', correct_rule('规则07', ['左移一位', '反转'])),
    ('实体03是什么方向？', query_record('实体03')),
    ('帮我记一下实体99朝右', teach_fact('实体99', '方向是朝右')),
    ('查一下规则12的内容', query_record('规则12')),
    ('把规则12改成：首位加一，交换前两位', correct_rule('规则12', ['首位加一', '交换前两位'])),
]
