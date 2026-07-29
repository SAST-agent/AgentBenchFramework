# 24 Miracle 评测可复现性修复设计

## 背景

PR #3 为 `24_miracle` 提供外部 Judge 评测、矩阵运行、可恢复 session 和 replay-driven HL 迭代。当前实现依赖仓库外或未随提交提供的 protocol/roster 文件，导致从 PR 的干净提交树运行时在参数校验前直接抛出 `FileNotFoundError`。此外，Python 对手的冻结身份只检查目录存在，没有验证其实际可执行内容。

本修复只完善 PR #3 的执行与原始数据边界，不把评测指标或信息增益计算提前塞入 game adapter。

## 目标

1. 让 PR #3 的干净 checkout 具备可验证的控制面输入。
2. 让新建 session 和 resume 使用明确、可追溯的 protocol/roster 输入。
3. 让缺少外部运行资产时返回稳定、可读的 preflight 错误。
4. 对 Python 与 C++ 对手采用一致的冻结身份校验，并消除不确定的目录匹配。
5. 保持 framework 保存一手数据、CI 计算派生指标的职责边界。

## 非目标

- 不提交完整评测集、对局 replay 集或大型二进制。
- 不定义或计算信息增益、raw/evo/gain/AUC。
- 不实现 RL 训练或 RL/HL 的统一统计聚合。
- 不修改本地 `main`，也不改变 CI 页面职责。

## 设计

### 1. 控制面输入

将 `24_miracle_evaluation_protocol.v0.3.json` 与 `24_miracle_roster_manifest.json` 作为小型、版本化的控制面元数据纳入 game adapter。完整运行资产继续由运行环境提供。

`miracle_matrix.py` 为 protocol、roster 和运行资产提供 CLI 路径参数；默认值保持仓库内约定路径，以保留现有调用方式。程序启动后先执行 preflight，检查输入文件存在、JSON 可解析、协议 hash 与 manifest 结构满足要求；失败时打印稳定的错误信息并返回非零状态，不让底层 `Path.read_text()` 异常成为用户接口。

### 2. Session 与 resume

新建 session 时，在 session manifest 中记录 protocol/roster 的路径、SHA-256 和结构版本；保留现有事件和结果文件格式。resume 在任何写入前读取 session manifest，验证当前输入与原 session 身份一致，并继续执行既有的只读验证流程。验证失败时 session 目录保持不变。

本修复不改变历史事件字段含义；新增字段使用现有 envelope/前向兼容策略。

### 3. 对手身份

manifest 为每个 Python 对手提供唯一的冻结身份来源：优先校验归档文件 hash，并在解压后校验 manifest 指定的 runnable entry hash；C++ 继续校验 `main.exe` hash。对手目录解析改为 manifest 驱动的确定性路径，不使用 `next(glob)` 隐式选择。缺少匹配项或存在多个候选项时，preflight 直接失败。

### 4. 原始数据边界

每个 event 继续保存 `run_id`、session 标识、事件类型、时间、结果状态和 artifact 引用；usage 与过程退出信息保留在原始记录中。closure report 仅作为派生汇总，不作为 CI 的唯一数据源。后续 CI 可从这些记录计算 episode 曲线、性能分和信息增益。

## 测试策略

先添加回归测试并确认其在修复前失败，再实现代码。测试覆盖：

- 干净路径下 protocol/roster 的 preflight 成功与缺失报错；
- CLI 自定义 protocol/roster 路径；
- 新 session 保存输入 hash；
- resume 成功、输入被替换时拒绝且不写入 session；
- Python 对手 hash 正确、被修改、缺失和重复目录时的结果；
- `--dry-run` 不依赖完整评测集且不产生未声明的持久化副作用。

实现后运行相关测试、全量 pytest、`compileall` 和 `git diff --check`。只有从 PR 分支的实际 checkout 得到全量测试通过，才报告修复完成。

## 验收标准

1. PR #3 的实际提交树中全量测试无失败。
2. 无外部完整测试集时，CLI 能完成 preflight/dry-run 或给出明确的资产缺失说明。
3. resume 对 protocol、roster 或 Python runnable identity 的任何不一致都拒绝继续，并保持 session 不变。
4. 原始 event 和 session manifest 可被 CI 独立读取，adapter 不计算科研派生指标。
