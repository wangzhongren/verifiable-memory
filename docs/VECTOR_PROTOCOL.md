# 有向实体向量推导（第一版）

现有的事实文本、数字串规则与证据格式仍可使用。本扩展增加实体向量、可复用的增量动作、有向边和路径推导。原始 `PROTOCOL.md` 是早期六操作场景的冻结记录，不把新增功能冒充为旧验收结果。

## 语义

- 实体有名称和向量 `v(e)`；动作有名称和增量 `Δ(a)`。坐标可以是有限的 JSON 整数或浮点数，维度为 1–4096，绝对值不超过 1,000,000。
- 写入时，每个坐标以十进制四舍六入五成双量化到 `10⁻⁶`，状态和推导只使用定点整数。查询以十进制字符串返回坐标，避免把浮点舍入误差误称为精确相等。这是数值格式选择，不自动赋予向量语义。
- `link_entities` 表示有向边 `A --a--> B`。仅当 `v(B) = v(A) + Δ(a)` 在量化后逐坐标相等时允许写入；反向边不会自动产生。
- `apply_vector_action` 将动作作用于源实体，返回推得的向量以及向量完全相同的候选实体。候选匹配尚不是一条已确认的有向边。
- `derive_entities` 只沿已明确记录的有效有向边走最短路径；同长路径按边名称排序，每步附实体/动作/边的修订号、写入操作 ID 与槽位哈希，以及写前向量、增量和写后向量。`max_hops` 默认为 8，可指定 1–16。
- 更正实体或动作不会改写旧边；边记录了三方的修订号。一旦依赖修订号不同，旧边显示为失效，推导跳过它。确认新状态后可用新边名重新连接。
- 实体、动作和边各占一个命名槽位；新库默认容量为 256，可在创建时用 `--capacity` 增大。既有库的容量不会随新增向量功能自动改变。

## CLI 和 Python 接口

CLI 的 `vector` 子命令接收一个 JSON 对象，返回操作证据；全局选项放在子命令前。以下用一份新库演示 A 经两个动作到 C：

```bash
python3 cli.py --session graph.db --json vector '{"op":"teach_entity","name":"A","vector":[0,0]}'
python3 cli.py --session graph.db --json vector '{"op":"teach_entity","name":"B","vector":[1,0]}'
python3 cli.py --session graph.db --json vector '{"op":"teach_entity","name":"C","vector":[1,2]}'
python3 cli.py --session graph.db --json vector '{"op":"teach_vector_action","name":"向右","delta":[1,0]}'
python3 cli.py --session graph.db --json vector '{"op":"teach_vector_action","name":"向上","delta":[0,2]}'
python3 cli.py --session graph.db --json vector '{"op":"link_entities","name":"AB","source":"A","action":"向右","target":"B"}'
python3 cli.py --session graph.db --json vector '{"op":"link_entities","name":"BC","source":"B","action":"向上","target":"C"}'
python3 cli.py --session graph.db --json vector '{"op":"derive_entities","source":"A","name":"C"}'
```

`name` 是推导目标。单步候选查询为 `{"op":"apply_vector_action","name":"向右","source":"A"}`。读取实体、动作或边可用 `{"op":"query_record","name":"A"}`。更正用 `correct_entity` 加 `vector`，或 `correct_vector_action` 加 `delta`。Python 调用方可以把同样的字典交给 `Session.apply(op, category="vector", source=..., utterance=...)`。无需模型解析这些 JSON 参数。

日常数据存 SQLite，写入与错误进入原有操作日志和哈希链。`export` → `replay.py` → `verify.py` 可独立重建与复核实体记录、有向边、动作结果和推导路径。规则正确性在**声明的数值变换与边关系内**可核验；向量相似性、语义实体识别、自动学习动作、来源真实性都不由这套数值证明自动解决。例如“鸡 + 篮球联想到蔡徐坤”需要额外的语义映射或明确来源，不能靠任意两段普通 embedding 向量相加来保证。
