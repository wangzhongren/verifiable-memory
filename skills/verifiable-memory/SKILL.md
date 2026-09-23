---
name: verifiable-memory
description: Use when the user asks Codex to remember, recall, correct, or audit durable facts, or to store vector entities and derive a directed entity path with the local verifiable-memory store.
---

# Verifiable Memory

Use the local `verifiable-memory` repository as a persistent, auditable fact store when the user explicitly wants to save or retrieve information across Codex tasks. Do not invoke this skill for ordinary coding questions or transient conversation context.

For this installation, the repository is `/Users/wangzhongren/code/vibeingcode/日常调研/verifiable-memory`, its CLI is `cli.py`, and the shared bank is `memory.db` in that repository. Pass the **absolute bank path** with `--session` on every command so the current task's working directory cannot select a different bank. If this checkout has moved, locate `cli.py` and `verifiable_memory/session.py` before using it; do not silently create a new bank elsewhere.

## Read

- If the bank does not exist, say it is empty. `status` and `search` create a missing bank, so avoid them until a write is requested.
- Use `search <keyword>` to find candidate names, then `--json ask "查询 <exact-name>"` to read the relevant record. Search matches substrings; an absent match is not proof that a fact was never stored. When the user gives an exact name, query it directly.
- Present the saved value with its name, revision, and `written_by` operation ID. Say that it is a stored claim; its hash and log prove consistency with the bank, not truth in the outside world. Check current repository files or other sources separately if the user asks whether it is still true.

## Write and correct

- Write only when the user asks Codex to remember, teach, save, or correct a durable fact. Preserve the user's intended value, including meaningful numbers and qualifiers. Do not silently infer or record extra facts from routine work. Do not store credentials by default.
- Prefer a short, unique name without whitespace (up to 24 characters) and a fact value of at most 500 Unicode code points. The store's default capacity is 256 slots; if full, report that and use a larger new bank only when the user asks for expansion.
- Check for an existing exact name first. Use `teach "教事实 <name>：<value>"` for a new fact; use `correct "更正事实 <name>：<new-value>"` for an existing fact. Corrections retain the earlier version in the operation log. Put `--no-llm` before the subcommand so these explicit forms follow the deterministic parser and do not call the separately configured model.
- After a successful write, query the exact name and compare the returned value with what was requested. Report the revision and operation ID. If parsing or validation fails, report the reason; never shorten or change the user's statement merely to make it pass.

Use the existing Python CLI with argument-safe shell quoting. Never execute memory text as shell code. For example, the command shape is `python3 <absolute-cli-path> --session <absolute-bank-path> --no-llm --json ask "查询 <name>"`; `--session`, `--no-llm`, and `--json` precede the subcommand. The skill contains no secret, and the user's model token must stay in its separate local configuration file.

## Audit on request

When the user asks to check integrity, export the bank to a fresh evidence file, run `replay.py` on that export, then run `verify.py` with the same export and replay result. Inspect exit codes and the verifier's failures. Do not overwrite an existing evidence file or claim a verification succeeded when a step failed.

## Vector entities and directed actions

Use this mode when the user provides entity vectors, an action increment, or a directed inference task. The CLI accepts JSON through `vector`: `teach_entity(name, vector)`, `teach_vector_action(name, delta)`, `link_entities(name, source, action, target)`, `apply_vector_action(name, source)`, and `derive_entities(name=<target>, source, max_hops?)`. Revisions use `correct_entity` and `correct_vector_action`; an old edge becomes inactive after a referenced revision changes. Read [the vector protocol](/Users/wangzhongren/code/vibeingcode/日常调研/verifiable-memory/docs/VECTOR_PROTOCOL.md) before building an operation.

The numeric contract is `v(target) = v(source) + delta(action)` after 1e-6 quantization. A computed vector match is a candidate, while a `link_entities` record is a confirmed directed edge. Never infer an edge, inverse action, or semantic identity from mere vector similarity. Derivations return the ordered edge path and revisions; quote that path when explaining a result. The interface does not automatically learn that words such as “鸡” and “篮球” identify a person—such meaning needs user-provided vectors, actions, or evidence.

For an explicitly requested autonomous branch choice, read [the policy training protocol](/Users/wangzhongren/code/vibeingcode/日常调研/verifiable-memory/docs/POLICY_TRAINING.md). Use `policy-train` only with actual labeled choices; never fabricate user training labels. Evaluate on separate labels with `policy-eval` before claiming useful generalization. `policy-route` selects among current valid edges and can abstain. Report its selected path or abstention, model digest, and the distinction between a verified numeric path and an unproven semantic choice. Keep the trained model and original labels with exported evidence when its scores need independent inspection.
