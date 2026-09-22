# verifiable-memory

**可验证的可教学记忆系统——给 LLM agent 一个每次回答带证据链、每次纠错带证明的长期记忆。**

**A teachable, verifiable memory for LLM agents: every answer carries its evidence chain, every correction carries a proof, and the whole memory can be replayed bit-exactly in an independent process.**

---

## English TL;DR

- **What it is**: a small, pure-stdlib Python memory system for LLM agents. Natural language goes in at the edges; inside the verification boundary there are only six deterministic structured operations over named slots (facts + executable rules).
- **Why it's different**: LLM "memory" is usually a prompt or a vector DB — recall you cannot audit. Here the LLM is kept **upstream of a hard verification boundary**: it may mis-parse, but it can never cross the whitelist, and every error is logged, never silently fixed.
- **Guarantees**: SHA-256 hash-chained op log · **zero-collateral-write certificates** (editing slot A provably leaves slots B..N untouched) · **bit-exact replay** from the log alone · a **second independent implementation** of the primitives that must agree · tamper detection · SQLite working store + canonical JSON evidence export.
- **Requirements**: Python 3.10+ (no third-party packages). One command installs nothing; one file (SQLite) is the whole memory.

A true story that motivated this: in earlier experiments, a text-model CLI was asked to apply the rule `规则07 = [反转, 左移一位]` to `1234` and answered **1432 — wrong** (correct: 3214). This system answers 3214 with a step-by-step trace. Then one correction (`更正规则 规则07：左移一位，反转`) makes the **same utterance** return **1432 — now correct**, with full evidence of which revision changed and why. The old wrong answer becomes the new right answer, and nothing else in memory moves.

---

## 为什么做这个

LLM agent 的"记忆"通常是一条越来越长的 prompt，或一个向量库——召回对了没奖励、错了没记录、改一条不知道会不会碰坏别的。而任何被真正托付的系统（科研记录、个人台账、agent 的长期状态）需要的恰恰是反过来的性质：

1. **答案可溯源**——这个结论是谁、在什么时候、基于哪条记录给出的；
2. **纠错可审计**——旧值、新值、第几版，全都在历史里，且改 A 不会弄坏 B；
3. **状态可重建**——从一份日志出发，任何独立的一方都能逐位重放出同样的记忆。

本项目来自一条神经记忆研究线的工程终点：八轮对照实验（学习型软寻址 vs 显式寻址）一致显示，**显式寻址赢、学习型软寻址输**。于是这个系统把结论推到极致——**把"回忆"这个环节整个删掉**：名称即地址，精确匹配，没有相似度检索。回忆不存在了，所以回忆不会错；模糊性全部推到验证边界上游，由 LLM 消化并由白名单兜底。

## 架构

```
自然语言 ──→ [LLM / Claude 技能 解析层]──结构化op──→ │ 验证边界 │──→ [命名槽位 + 符号执行器]
              （边界上游，可以答错）                  └──────────┘      每步哈希链 · 证书 · op 日志
                                                                      ↓
                                                    答案 + 证据链（哪条记录、哪个op、零附带损害证明）
                                                                      ↓
                                            replay.py 独立进程重放 · verify.py 第二套实现独立核验
```

六种结构化操作（边界内只有这些）：`teach_fact / teach_rule / correct_fact / correct_rule / query_record / apply_rule`。规则由四个原语组成：`反转 / 左移一位 / 交换前两位 / 首位加一`（首位加一按模 8），作用于 2–8 位、每位 0–7 的数字串。

## 快速开始

```bash
git clone git@github.com:wangzhongren/verifiable-memory.git
cd verifiable-memory

# 18 步固定场景：冻结 PROTOCOL → 教学/查询/纠错 → 证据落盘（拒绝覆盖）
python3 cli.py script

# 独立重放 + 独立核验（新进程，只吃导出的证据文件）
python3 replay.py
python3 verify.py

# 49 项自检：两套原语实现互查、错误路径、并发压力、篡改检测……
python3 tests/checks.py
```

日常使用（一命令一进程，跨进程持久；默认记忆库 `memory.db`，SQLite）：

```bash
python3 cli.py teach   "教事实 实体07：方向是向上"
python3 cli.py teach   "教规则 规则07：反转，左移一位"
python3 cli.py ask     "对1234应用规则07。"            # → 3214，附逐步 trace
python3 cli.py correct "更正规则 规则07：左移一位，反转"  # → 第2版 + 零附带损害证书
python3 cli.py ask     "对1234应用规则07。"            # → 1432（同一条话语，规则已变）
python3 cli.py status                                  # 列出全部记忆
python3 cli.py search 对偶                              # 确定性子串检索
python3 cli.py reset                                   # 删除记忆（谨慎）
```

批量导入（先导数据，别处直接用）：

```bash
python3 cli.py import examples/import_knowledge.jsonl --dry-run   # 预检，不写入
python3 cli.py import examples/import_knowledge.jsonl             # 导入（每条都是留痕的 teach op）
python3 cli.py import knowledge.jsonl --on-conflict correct       # 已存在则转更正
```

JSONL 格式，每行一条：`{"kind":"fact","name":"...","content":"..."}` 或 `{"kind":"rule","name":"...","program":["反转",...]}`。坏行按行报告、不拖垮整批、不落日志。**导入必须过验证边界**：每条都是真实留痕的操作，没有任何绕过日志的直插。

LLM 解析模式（可选；无 key 时上述结构化文法即后备，功能完整）：

```bash
export VM_BASE_URL=https://api.deepseek.com/v1   # 任何 OpenAI 兼容接口
export VM_API_KEY=sk-...
export VM_MODEL=deepseek-chat
python3 cli.py paraphrases --llm                  # 预声明改写鲁棒性测试
```

## 命令一览

| 命令 | 作用 |
|---|---|
| `teach "教事实 <名称>：<内容>"` | 教新事实（名称已存在会被拒绝） |
| `teach "教规则 <名称>：<原语>，<原语>…"` | 教新规则（原语白名单） |
| `correct "更正事实/规则 <名称>：…"` | 纠错 = 同槽覆盖，版本 +1，附零附带损害证书 |
| `ask "查询 <名称>"` / `ask "对<数字串>应用<规则名>。"` | 查询 / 执行（答案带证据链与逐步 trace） |
| `status` | 列出全部槽位（内容、版本、写入 op）与终态哈希 |
| `search 关键词` | 名称与内容的确定性子串检索（导航，不落日志） |
| `import <jsonl> [--on-conflict] [--dry-run]` | 批量导入 |
| `export [--out]` | DB → canonical 证据 JSON（replay/verify 只吃它） |
| `script` | PROTOCOL 固定 18 步场景（先冻结后执行，拒绝覆盖） |
| `repl` / `reset` | 交互模式 / 删除会话 |

## 证据长什么样

一次执行的真实输出：

```
✓ 答案：3214 · 证据：记录 规则07 第1版（写入于 op-005，槽哈希 594551f4c62a）
  · 执行 [反转:1234→4321 → 左移一位:4321→3214] · 后端 symbolic
```

一次纠错的证书（v2，O(1) 大小）：目标槽写前/写后哈希、状态哈希、`created` 标志。`zero_collateral` 是写入时的构造断言，并由 verify.py 用**第二套独立实现**重放重建全槽 diff 逐条证实——断言与证实的分工写在格式里。

## 设计原则

1. **LLM 在验证边界上游**。它负责把人话翻译成六种操作；词表白名单、类别锁、存在性检查三道锁让翻译错误"被拒绝并留痕"，而不是静默通过。
2. **显式寻址**。名称即地址，纠错 = 同槽覆盖。没有向量、没有相似度——这是有意的设计立场，来自对照实验的结论（学习型软寻址在窄域全面落后）。
3. **日志是唯一事实来源**。state 表/快照皆是可重建缓存；任何对缓存或日志行的手改都会被"重放 vs 缓存"核对或离线全量重放抓住。
4. **独立性是核验的全部价值**。verify.py 只与 store.py 共享两个哈希原语（canonical JSON、sha256），原语执行、槽位应用、证书复核全部独立实现——两套实现不一致即整体失败。开发过程中这套互查真实抓到过 bug。
5. **失败语义**。执行器对非法输入/未知原语抛错，宿主不得代为修复；错误操作落日志、无副作用、哈希链不断。

## 信任模型（诚实版）

- **能抓**：手改数据库任意行、手改 state 缓存、手改导出文件——哈希链断裂、终态不符、证书断言复核失败，退出码 1。
- **不能抓**：能同时重写全部行并重算整条链的攻击者（没有外部锚的哈希链都如此）。工作路径对"检查点之前"的日志段有意只做轻校验（快照锚 + 尾部重放），全量重放核验是 `replay.py`/`verify.py` 的离线职责——数据库工程的 WAL/checkpoint 分工，不是漏洞。
- **并发**：多进程写入走乐观协议（落库前核对链头，被抢先则重载重试）；5 进程真并发压力测试在 `checks.py`。

## 规模化

默认 256 个命名槽位（`--capacity` 建库时可调，容量烤在哈希链里，改容量 = reset 后重建）。32k 槽规模的改造已就位：证书 v2 O(1)/写、快照+尾部重放（load 不再背全量日志）、确定性检索。再往上的备件（Merkle 增量哈希、FTS5）按需再上。

## 项目结构

```
├── cli.py / replay.py / verify.py    # 三个入口：日常命令 / 独立重放 / 独立核验
├── verifiable_memory/                # 核心包
│   ├── data.py       # 四原语参考实现 + 预声明场景
│   ├── store.py      # 命名槽位、精确寻址、op 白名单、canonical JSON + sha256
│   ├── executor.py   # 执行器接口 + 符号后端（逐步 trace，失败即抛错）
│   ├── proof.py      # 零附带损害证书（v2 O(1)；v1 全表格式兼容读取）
│   ├── parser.py     # NL→op：LLM 结构化输出（白名单）+ 无 key 后备文法
│   ├── llm.py        # OpenAI 兼容适配器（urllib；温度 0；密钥只从环境读）
│   ├── session.py    # op 日志 + 哈希链 + 乐观并发 + 快照/尾部加载
│   └── storage.py    # SQLite 工作存储（ops 只追加 + state 缓存 + 检查点）；JSON 证据格式
├── tests/checks.py                   # 49 项自检（含真子进程并发压力与篡改测试）
├── docs/PROTOCOL.md                  # 预声明的验收标准与期望值（先冻结后执行）
├── docs/REPORT.md                    # 研发记录：验收结果、机制事实、抓到的 bug
└── examples/import_knowledge.jsonl   # 示例知识文件（可直接导入试用）
```

`demo/` 由 `cli.py script` 运行时生成（PROTOCOL + 源码冻结快照、op 日志、转录、证据文件），不入库。

## 测试

```bash
python3 tests/checks.py    # 49 项，全绿为基线
```

覆盖：两套原语实现全原语一致、store 全部错误路径、证书 O(1) 与断言、后备文法解析全场景、哈希链与重载、SQLite 双向篡改检测、证据逐字节等价（JSON 后端 == SQLite 导出）、快照分工、检索、批量导入幂等、5 进程并发压力、跨进程四步管线 + 灵魂样例独立断言。

## 限制（如实）

- 执行器是符号的、域是四个数字串原语——答案 100% 正确是平凡事实；这个系统的全部难度在解析层与可验证性机制。
- 名称精确匹配，没有模糊寻址（有意的立场，不是缺失）。
- 命令文法是中文（槽位名称本身是任意字符串）。
- 记忆质量的上限 = 解析层（LLM）的质量；边界保证的是"错也留痕、可审计"，不是"不错"。

## 路线图

- [ ] 条件/循环原语（规则从线性序列走向程序）
- [ ] Merkle 增量状态哈希（写延迟敏感场景）
- [ ] FTS5 检索与别名/标签索引（万级以上槽位）
- [ ] 神经解析器/执行器接入（用训练出的模块替换符号后端，验证边界不动）

## License

[MIT](LICENSE) © 2026 wangzhongren
