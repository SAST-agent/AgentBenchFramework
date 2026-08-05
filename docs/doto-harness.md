# Codex-orchestrated DOTO benchmark

## What orchestrates the benchmark

AgentBenchFramework exposes deterministic atomic tools; it does not own an LLM
loop. **Codex decides** which training replay to inspect, what complete
`playerAI.cpp` change to make, which explicit parent to use, whether another
iteration is justified, and which complete training version enters the final
test. No iteration, token, or wall-time budget is imposed by the DOTO workflow.

The complete authority is stored in `DotoResults`. The unchanged
AgentBenchResults repository receives only the five-file **AgentBenchResults projection**
after the Run is sealed and validated.

Requirements: Python 3.11+, `uv`, `g++`, `make`, ZIP support, and:

```bash
uv sync --extra doto
```

## Four required Skills

Every Run snapshots and hashes exactly these packages:

- [`doto-benchmark-run`](../skills/doto-benchmark-run/SKILL.md): lifecycle,
  training selection, finalization, validation, and export.
- [`doto-game-rules`](../skills/doto-game-rules/SKILL.md): authoritative rules,
  observation, joint action, action mask, terminal, and strict KL support.
- [`doto-agent-authoring`](../skills/doto-agent-authoring/SKILL.md): fixed SDK and
  complete C++ candidate contract.
- [`doto-replay-reader`](../skills/doto-replay-reader/SKILL.md): replay/trace
  fields, events, evidence joins, and diagnosis.

Read rules and authoring before the baseline. Read replay-reader before drawing
any conclusion from a replay or trace. The workflow Skill governs every state
transition.

## One-prompt Codex launch

Open this directory as the Codex workspace; do not start inside only one of the
repositories:

```text
/home/six/Documents/THU/activities/SAST/AgentBenchmark
```

The workspace must contain `AgentBenchFramework`, `AgentBench`, `DotoResults`,
and `AgentBenchResults`. During review of PR #20, Codex uses
`AgentBenchFramework/.worktrees/feature-doto`; after merge it uses the normal
`AgentBenchFramework` checkout. Send exactly one task. No manual preparation,
Run initialization, finalization, validation, or export command is required:

```text
完成一次完整的 23_doto benchmark Run。不要让我手动执行任何命令，也不要在正常步骤之间询问确认；你负责从准备数据到最终导出全过程。

把启动时的当前目录记为 WORKSPACE_ROOT。优先将 FRAMEWORK_DIR 设为：
- AgentBenchFramework/.worktrees/feature-doto
如果该目录不存在，则使用：
- AgentBenchFramework

所有 Framework 命令都在 FRAMEWORK_DIR 中执行。下面的 Skill 和初始 Agent 路径相对于 FRAMEWORK_DIR；仓库和人类策略路径相对于 WORKSPACE_ROOT。不要混用主 checkout 与 PR worktree。

完整阅读并严格遵循以下四个 Skill：
- skills/doto-benchmark-run/SKILL.md
- skills/doto-game-rules/SKILL.md
- skills/doto-agent-authoring/SKILL.md
- skills/doto-replay-reader/SKILL.md

相关仓库和数据：
- 人类策略原始数据：AgentBench/backend_sources/corpus/23_doto
- 完整权威结果：DotoResults
- 兼容投影结果：AgentBenchResults
- 初始 Agent：examples/doto-initial-playerAI.cpp

你的任务如下：

1. 检查环境和依赖，验证43个人类策略身份。
2. 使用Framework提供的公开源码归档构建15策略训练 bundle。
3. 使用既有工具构建28策略 sealed test bundle。把测试集当作不可查看的黑盒：不得打开、搜索、阅读、总结或根据测试策略源码及历史测试结果修改 Agent。
4. 自动生成唯一 run-id，在 DotoResults 中初始化权威 Run，并快照四个 Skill、初始 Agent、训练集和测试集身份。
5. 完成 baseline 的“构建、30场训练评测、IG记录、关闭 iteration”。
6. 只依据15策略训练集的正常 replay、trace、score 和严格 IG 证据改进完整的 playerAI.cpp。每次都使用明确的已关闭父版本，保存具体分析，创建新 iteration，构建，完成30场双座位评测，记录严格 IG 或准确的缺失原因，然后关闭 iteration。
7. 由你判断是否还有证据支持继续改进以及应选择哪个完整训练版本。不要使用固定迭代次数、token预算或时间预算；但如果继续修改已没有具体训练证据支持，就停止。
8. 在查看任何测试结果之前选定一个已关闭、构建成功且30场训练评测完整的 candidate iteration。
9. 使用该候选版本执行且仅执行一次28策略、双座位、共56场的 sealed test。测试开始后不得更换候选版本，不得根据测试结果继续修改 Agent。中断时只恢复同一候选和同一测试集的缺失 cell。
10. 全程使用多进程调度，CPU目标为70%，不要人为设置过低的worker上限。
11. 所有失败、超时、崩溃、协议错误、缺失评测和无法计算IG的原因必须如实保存；不能删除失败记录，不能用其他指标冒充严格KL/IG。
12. 完成后验证 DotoResults，生成聚合结果、报告、score–iteration曲线和IG–iteration曲线。
13. 将结果导出到未经修改的 AgentBenchResults，只生成规定的五文件投影，并验证投影。
14. 不要修改 AgentBenchFramework、DotoResults 或 AgentBenchResults 的框架代码；本次只运行benchmark、修改候选 playerAI.cpp并写入正式结果。
15. 不要提前停止在“告诉我下一步怎么做”。你必须自行执行全部命令，直到Run完成、验证和导出结束，或者遇到确实无法自行解决的外部阻塞。

最终只向我报告：
- run-id和DotoResults绝对路径；
- 选择的candidate iteration及其版本/hash；
- 训练集和测试集分别击败的人类策略数量；
- score和IG曲线路径；
- AgentBenchResults投影路径；
- 所有失败、缺失或不完整项；
- 如果阻塞，给出已经完成的状态和唯一的真实阻塞原因。
```

This is behavioral sealed-test isolation inside one Codex task: Codex may invoke
the evaluator tooling but must treat the sealed pool as an opaque runtime input.
Candidate selection is fixed before any hidden result is observed, and hidden
evidence can never feed a new iteration. The commands below remain operator and
recovery references rather than required launch steps.

## Prepare human pools

Verify all 43 frozen complete snapshots, then build the 15-policy public train
bundle. The evaluator separately builds and owns the 28-policy sealed bundle.

```bash
uv run --extra doto python -m agentbench_frame.doto population verify \
  --manifest src/agentbench_frame/doto/population.toml \
  --source-root ../AgentBench/backend_sources/corpus/23_doto
uv run --extra doto python -m agentbench_frame.doto population build-train \
  --manifest src/agentbench_frame/doto/population.toml \
  --source-root src/agentbench_frame/doto/human_policies/train.tar.gz \
  --output-dir /workspace/doto-train-bundle
```

Require 43/43 identities verified and 15/15 public policies ready. Do not expose
sealed source, policy names, paths, binaries, or old test outcomes to Codex.

## Initialize an authoritative Run

The evaluator briefly opens the sealed bundle to capture its hash, creates the
Run, and closes that descriptor before Codex begins training:

```bash
uv run --extra doto python -m agentbench_frame.doto run init \
  --agent codex --initial-player-ai examples/doto-initial-playerAI.cpp \
  --doto-results ../DotoResults --run-id <run-id> \
  --train-bundle /workspace/doto-train-bundle \
  --test-bundle /proc/self/fd/<evaluator-fd> --skills-root skills
uv run --extra doto python -m agentbench_frame.doto run status \
  --run-dir ../DotoResults/runs/23_doto/codex/<run-id>
```

Initialization snapshots the initial source, four complete Skills, pool hashes,
and benchmark version. Never hand-edit the new Run.

## Complete one training iteration

Build the baseline, run/resume the full **30-cell** matrix (15 humans × both
candidate seats, seed 11), record baseline IG as missing, and close:

```bash
uv run --extra doto python -m agentbench_frame.doto iteration build \
  --run-dir <run-dir> --iteration 0
uv run --extra doto python -m agentbench_frame.doto iteration evaluate \
  --run-dir <run-dir> --iteration 0 --pool /workspace/doto-train-bundle \
  --cpu-target 70
uv run --extra doto python -m agentbench_frame.doto iteration compare \
  --run-dir <run-dir> --iteration 0
uv run --extra doto python -m agentbench_frame.doto iteration close \
  --run-dir <run-dir> --iteration 0
```

The adaptive multi-process scheduler targets **70%** aggregate CPU, grows below
65%, holds at 65–75%, and pauses new launches above 75%. `--workers` is an
optional hard ceiling. Running matches are never cancelled.

A formal score exists only after 30 normal cells. Failed build, timeout, crash,
protocol error, corrupt replay, and missing cell remain explicit and produce a
null formal score. Close failed/incomplete iterations; do not erase them.

## Diagnose, change, and create a child

Use `match` only for diagnostics. Use the official ZIP for events/scores and the
trace for observations/actions:

```bash
uv run --extra doto python -m agentbench_frame.doto match \
  --agent0 <candidate> --agent1 <training-opponent> --seed 11 \
  --output-dir /workspace/doto-diagnostics --tag candidate-seat0
uv run --extra doto python -m agentbench_frame.doto replay \
  --path /workspace/doto-diagnostics/candidate-seat0_seed11.zip \
  --jsonl /workspace/doto-diagnostics/candidate-seat0.events.jsonl
```

After evidence review, save a nonempty analysis and a non-identical complete
source, then create the explicit child:

```bash
uv run --extra doto python -m agentbench_frame.doto iteration begin \
  --run-dir <run-dir> --parent 0 --source /workspace/child-playerAI.cpp \
  --analysis-file /workspace/iteration-analysis.md
```

Build, evaluate all 30 cells, compare strict deterministic KL on recorded
observations, and close the returned iteration. Equal actions give strict KL 0;
different deterministic support gives infinity with numeric null; missing or
invalid output stays missing with a reason. Never call distance, change rate, or
score gain IG.

After each close, Codex inspects `run status`, aligned score/IG curves, failures,
and selected evidence. It either justifies another explicit child or stops and
selects among complete training versions. There is no automatic repetition.

## Finalize the hidden matrix once

The evaluator opens the sealed bundle only after training selection:

```bash
uv run --extra doto python -m agentbench_frame.doto run finalize \
  --run-dir <run-dir> --candidate-iteration <selected-training-iteration> \
  --sealed-bundle-fd <evaluator-fd> --cpu-target 70
```

Candidate and pool identities are durably fixed before the **56-cell** matrix
(28 policies × both seats) starts. An interruption resumes only missing cells
for the same identities. The final Run seals even if incomplete; final evidence
must never create another candidate.

A human policy counts as defeated only when both seats terminate normally and
their candidate-oriented mean score difference is strictly positive.

## Validate, report, and export

```bash
uv --directory ../DotoResults run doto-results validate \
  runs/23_doto/codex/<run-id>
uv --directory ../DotoResults run doto-results aggregate .
uv --directory ../DotoResults run doto-results build-report . --output _site
uv run --extra doto python -m agentbench_frame.doto run export \
  --run-dir ../DotoResults/runs/23_doto/codex/<run-id> \
  --agentbench-results ../AgentBenchResults
uv --directory ../DotoResults run doto-results check-projection \
  ../AgentBenchResults/runs/23_doto/codex/<run-id>
```

The projection contains exactly `run.toml`, `summary.json`, `score_curve.json`,
`ig_curve.json`, and `doto_results_ref.json`. Before publication, scan both
targets for credentials, hidden source fragments, absolute sealed paths, `NaN`,
and `Infinity`; exclude binaries, caches, locks, and unredacted sealed metadata.
