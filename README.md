# verifiable-memory

**可验证的可教学记忆系统——给 LLM agent 一个每次回答带证据链、每次纠错带证明的长期记忆。**

**A teachable, verifiable memory for LLM agents: every answer carries its evidence chain, every correction carries a proof, and the whole memory can be replayed bit-exactly in an independent process.**

---

## English TL;DR

- **What it is**: a small, pure-stdlib Python memory system for LLM agents. The original six structured operations teach facts and executable rules; a new vector interface adds typed entities, reusable directed actions, revision-pinned edges, and independently checked path derivation.
- **Why it's different**: the LLM stays **upstream of a structured verification boundary**. Accepted operations are deterministic and auditable. Schema checks cannot establish factual truth or correct interpretation; parser failures occur before the operation log.
- **Guarantees**: SHA-256 hash-chained op log · **zero-collateral-write certificates** (editing slot A provably leaves slots B..N untouched) · **bit-exact replay** from the log alone · a **second independent implementation** of the primitives that must agree · tamper detection · SQLite working store + canonical JSON evidence export.
- **Requirements**: Python 3.10+ (no third-party packages). One command installs nothing; one file (SQLite) is the whole memory.

A true story that motivated this: in earlier experiments, a text-model CLI was asked to apply the rule `规则07 = [反转, 左移一位]` to `1234` and answered **1432 — wrong** (correct: 3214). This system answers 3214 with a step-by-step trace. Then one correction (`更正规则 规则07：左移一位，反转`) makes the **same utterance** return **1432 — now correct**, with full evidence of which revision changed and why. The old wrong answer becomes the new right answer, and nothing else in memory moves.

---

## 为什么做这个

LLM agent 的"记忆"通常是一条越来越长的 prompt，或一个向量库——召回对了没奖励、错了没记录、改一条不知道会不会碰坏别的。而任何被真正托付的系统（科研记录、个人台账、agent 的长期状态）需要的恰恰是反过来的性质：

1. **答案可溯源**——这个结论是谁、在什么时候、基于哪条记录给出的；
2. **纠错可审计**——旧值、新值、第几版，全都在历史里，且改 A 不会弄坏 B；
3. **状态可重建**——从一份日志出发，任何独立的一方都能逐位重放出同样的记忆。

本项目来自一条神经记忆研究线的工程终点：八轮对照实验（学习型软寻址 vs 显式寻址）一致显示，**显式寻址赢、学习型软寻址输**。于是这个系统把结论推到极致——**把"回忆"这个环节整个删掉**：名称即地址，精确匹配，没有相似度检索。按名称取值不再依赖相似度，但名称选择仍可能出错；模糊性移到验证边界上游。白名单约束操作形状，不能保证解析符合用户意图。

## 架构

```
自然语言 ──→ [LLM / Claude 技能 解析层]──结构化op──→ │ 验证边界 │──→ [命名槽位 + 符号执行器]
              （边界上游，可以答错）                  └──────────┘      每步哈希链 · 证书 · op 日志
                                                                      ↓
                                                    答案 + 证据链（哪条记录、哪个op、零附带损害证明）
                                                                      ↓
                                            replay.py 独立进程重放 · verify.py 第二套实现独立核验
```

原始场景的六种结构化操作：`teach_fact / teach_rule / correct_fact / correct_rule / query_record / apply_rule`。数字串规则由四个原语组成：`反转 / 左移一位 / 交换前两位 / 首位加一`（首位加一按模 8），作用于 2–8 位、每位 0–7 的数字串。实体向量与有向动作是独立的扩展，见下文。

## 快速开始

```bash
git clone git@github.com:wangzhongren/verifiable-memory.git
cd verifiable-memory

# 18 步固定场景：冻结 PROTOCOL → 教学/查询/纠错 → 证据落盘（拒绝覆盖）
python3 cli.py script

# 独立重放 + 独立核验（新进程，只吃导出的证据文件）
python3 replay.py
python3 verify.py

# 基础自检 + 检查点/审计回归测试（无需 API key）
python3 tests/checks.py
python3 tests/regressions.py
python3 tests/llm_checks.py
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

本地模型配置（可选）：把配置保存到 `~/.config/verifiable-memory/config.json`，此文件不在仓库里。例如 Anthropic 兼容服务：

```json
{
  "api_style": "anthropic",
  "base_url": "https://your-provider.example",
  "model": "your-model",
  "auth_token": "replace-with-your-token"
}
```

`auth_token` 使用 Bearer 认证；使用 API key 的服务改填 `api_key`，两者选一个。OpenAI 兼容服务使用 `api_style: "openai"`、`base_url: "https://your-provider.example/v1"` 和 `api_key`。文件权限建议设为 `600`。

配置一次后即可直接使用口语：

```bash
python3 cli.py teach "请帮我记一条事实，名称是项目主干，内容是main。"
python3 cli.py ask "项目主干是什么？"             # → main
python3 cli.py paraphrases --out paraphrases.json   # 改写基准 v2，拒绝覆盖
```

结构化文法能识别的命令仍在本地执行；口语在找到本地模型配置后自动调用服务。`--llm` 强制走模型，`--no-llm` 禁用模型，`--config <路径>` 指定另一份本地配置；这些全局选项放在子命令之前。模型返回错误不会静默改用后备解析。

本地配置优先于环境配置，不跨来源借用认证信息。没有本地配置时仍兼容原有 `VM_*` / `OPENAI_*` / `ANTHROPIC_*` 环境配置，需显式 `--llm`；`VM_API_STYLE` 可选择协议。Anthropic 的环境配置读取 `ANTHROPIC_BASE_URL`、`ANTHROPIC_MODEL`、`ANTHROPIC_AUTH_TOKEN`（或 `ANTHROPIC_API_KEY`）。

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

1. **LLM 在验证边界上游**。它负责把人话翻译成六种操作；词表白名单、类别锁、存在性检查约束结构化操作。合法但选错名称/内容的操作仍可能通过；可审计不等于事实真实或意图正确。
2. **显式寻址**。名称即地址，纠错 = 同槽覆盖。没有向量、没有相似度——这是有意的设计立场，来自对照实验的结论（学习型软寻址在窄域全面落后）。
3. **日志是唯一事实来源**。v2 将完整事件（包括来源、原话、类别、结果、证书、错误）与前驱事件哈希一起散列；状态哈希独立保留。工作路径核对检查点、尾部事件链和当前缓存，历史段由离线全量核验负责。
4. **独立性是核验的全部价值**。verify.py 只与 store.py 共享两个哈希原语（canonical JSON、sha256），原语执行、槽位应用、证书复核全部独立实现——两套实现不一致即整体失败。开发过程中这套互查真实抓到过 bug。
5. **失败语义**。执行器对非法输入/未知原语抛错，宿主不得代为修复；错误操作落日志、无副作用、哈希链不断。

## 信任模型（诚实版）

- **能抓**：不重算链的事件字段修改（包括查询的来源/原话/类别）、缺失哈希、条目乱序、未同步修改计数及链头的日志删减，以及 state/checkpoint_state 内容篡改。工作路径检查尾部，离线重放/核验检查全部事件。SQLite 的物理布局和未参与协议的辅助字段不属于事件证明。
- **不能抓**：能重写日志并重算链及元数据的攻击者；也无法识别完整旧版本的回滚或同时改写计数和链头的尾部截断（需要外部可信锚）。工作路径对"检查点之前"的日志段有意只做轻校验（快照锚 + 尾部重放），全量重放核验是 `replay.py`/`verify.py` 的离线职责——数据库工程的 WAL/checkpoint 分工，不是漏洞。
- **并发**：多进程写入走乐观协议（落库前核对链头，被抢先则重载重试）；5 进程真并发压力测试在 `checks.py`。

## 实体向量与有向动作

现在可以把实体存为向量，把动作存为可复用的向量增量，并记录 `A --动作--> B`。只有量化后 `v(B) = v(A) + Δ(动作)` 才能建边；多步推导返回每一步的实体、动作、向量和修订证据。实体或动作被更正时，引用旧修订的边自动退出当前推导，但历史记录不消失。CLI 使用结构化 JSON `vector` 子命令，Python 使用 `Session.apply(..., category="vector")`。这适用于明确定义了向量和动作的系统，不会自动学出“鸡与篮球指向蔡徐坤”这类语义关系。完整语义、操作示例和边界见 [docs/VECTOR_PROTOCOL.md](docs/VECTOR_PROTOCOL.md)。

对“根据输入自行决定分叉”的需求，新增可训练的本地控制器：用标注的 `(源实体, 查询, 选中边或弃权)` 训练，在当前有效出边中逐步选择路径；不确定时弃权，图变动后要求重新训练。选中的路径再交给独立核验器检查方向与向量运算。它是有限动作空间的监督学习原型，不是通用 LLM，也不会自动学出“鸡与篮球指向蔡徐坤”这类语义关系。数据格式、CLI 示例和证明边界见 [docs/POLICY_TRAINING.md](docs/POLICY_TRAINING.md)。

## 规模化

默认 256 个命名槽位，建库时可用 `--capacity` 指定更大的正整数，代码没有另设固定总量上限。例如 `python3 cli.py --session large-memory.db --capacity 32000 status` 创建一个 32k 容量的新库。容量写入状态哈希，现有库不会自动扩容；需要更大容量时，应保留原库和历史证据，再将知识导入新库。实际规模受内存、磁盘和性能限制，不等于无限存储。

每条事实内容上限为 **500 字符**，教学、纠错和批量导入使用同一限制；结构化内容超出会拒绝，存储层不会自动截断。按 Python `len()` 计 Unicode 码点，中文、英文、标点和空格均计入，组合 emoji 可能占多个码点。已有 v2 库不需要迁移；旧版因 200 字符上限被拒绝并留痕的操作，重放时仍按原约束核对，不把历史错误改成成功。

证书 v2 大小为 O(1)，但状态哈希仍为 O(N)；这不意味着写入耗时 O(1)。独立 checkpoint_state 每 512 条操作在事务内复制一次，成本 O(N)，其间从固定快照重放尾部并核对最新缓存。尚未提供 32k 槽性能基准。再往上的备件（Merkle 增量哈希、FTS5）按需再上。

## 项目结构

```
├── cli.py / replay.py / verify.py    # 三个入口：日常命令 / 独立重放 / 独立核验
├── verifiable_memory/                # 核心包
│   ├── audit.py      # v2 完整事件链、序号和证据封装校验
│   ├── data.py       # 四原语参考实现 + 预声明场景
│   ├── store.py      # 命名槽位、精确寻址、op 白名单、canonical JSON + sha256
│   ├── executor.py   # 执行器接口 + 符号后端（逐步 trace，失败即抛错）
│   ├── vectors.py    # 固定点向量、有效有向边和多步路径推导
│   ├── policy.py     # 监督学习分叉控制器（路径选择在验证边界上游）
│   ├── proof.py      # 零附带损害证书（v2 O(1)；v1 全表格式兼容读取）
│   ├── parser.py     # NL→op：LLM 结构化输出（白名单）+ 无 key 后备文法
│   ├── llm.py        # 本地配置 + OpenAI/Anthropic 调用（urllib；温度 0）
│   ├── session.py    # op 日志 + 哈希链 + 乐观并发 + 快照/尾部加载
│   └── storage.py    # SQLite 工作存储（ops 只追加 + state 缓存 + 检查点）；JSON 证据格式
├── tests/checks.py                   # 基础自检（含真子进程并发压力与篡改测试）
├── tests/regressions.py              # 检查点边界、完整事件链和事务回滚回归测试
├── tests/vector_checks.py            # 向量实体、方向、修订失效与独立推导测试
├── tests/policy_checks.py            # 训练、自动选路、弃权和策略版本失效测试
├── tests/llm_checks.py               # 本地 HTTP 服务验证协议、配置和 CLI；无需真实密钥
├── docs/PROTOCOL.md                  # 预声明的验收标准与期望值（先冻结后执行）
├── docs/REPORT.md                    # 研发记录：验收结果、机制事实、抓到的 bug
└── examples/import_knowledge.jsonl   # 示例知识文件（可直接导入试用）
```

`demo/` 由 `cli.py script` 运行时生成（PROTOCOL + 源码冻结快照、op 日志、转录、证据文件），不入库。

## 测试

```bash
python3 tests/checks.py
python3 tests/regressions.py
python3 tests/vector_checks.py
python3 tests/policy_checks.py
python3 tests/llm_checks.py
```

覆盖：两套原语实现全原语一致、store 全部错误路径、证书 O(1) 与断言、后备文法解析全场景、哈希链与重载、SQLite 双向篡改检测、证据逐字节等价（JSON 后端 == SQLite 导出）、快照分工、检索、批量导入幂等、5 进程并发压力、跨进程四步管线 + 灵魂样例独立断言。

## v2 格式与升级边界

此次修复使用 `session@v2` / `sqlite@v2`：每条事件新增 `prev_entry_hash` 和 `entry_hash`，证据封装新增 `n_ops` 和 `log_head`。两种后端生成相同的证据格式。`verify.py` 独立重算事件链，并检查重放报告的源文件摘要，拒绝沿用过期报告。

**旧 v1 数据不会自动迁移或覆盖。** 当前版本拒绝把缺少完整事件链的 v1 文件当作 v2 核验。请保留原库和原证据，用对应旧版本读取/核验；需要继续使用其内容时，先导出为知识 JSONL，再用 `--session <新文件.db> import <文件.jsonl>` 建立新库。重新导入只保留知识内容，不延续旧历史的证明，不能追溯证明旧来源字段未被修改。不要直接修改格式标记，也不必 reset 旧库。

存储修复见 [docs/FIXES.md](docs/FIXES.md)；本地配置、解析与真实模型复测见 [docs/LLM_FIXES.md](docs/LLM_FIXES.md)。GitHub Actions 配置在 Python 3.10 / 3.14 上执行基础自检、回归测试和完整演示链。

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
