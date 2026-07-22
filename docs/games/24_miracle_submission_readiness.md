# 24_miracle 提交就绪（Submission Readiness）

## 一、状态字段核验（权威）
- 协议 v0.3 磁盘状态：`PRE_REGISTERED_NOT_AUTHORIZED`（正确）。
- 笔误 `PRE_REGISTERED_NOT_AUTHORED`（曾出现在对话文字）**未写入任何权威 JSON/报告/源码**（已 grep 确认 NONE）。
- 说明：协议在赛前保持"未执行"状态；**正式授权由 execution manifest 独立记录**（`.smoke/matrix/<id>/manifest.json` 的 `auth_text`）；**完成状态由矩阵 progress/result 记录**。v0.3 事后**不被修改**。

## 二、Framework 提交范围（未来 commit，本轮不 git add/commit）

### 应提交
- `src/agentbench_frame/games/__init__.py`、`games/miracle/{__init__,result,driver,proctree,match_runner,entry,matrix,matrix_runner,smoke_audit,paths}.py`
- `tests/miracle/*.py` + `conftest.py`
- `vendor/miracle_local/run_match.py`、`vendor/results_local/aggregate.py`（含来源/哈希/补丁说明头部）
- `tools/{miracle_matrix,miracle_smoke,miracle_precheck,miracle_rank16_build}.py`
- `docs/games/24_miracle_*.md` + `*.json`（接入文档、roster、协议 v0.2/v0.3/draft、battle_record、evidence_audit、integration_report、submission_readiness、stage9A/9B 报告、assets）+ `docs/games/evidence/*.txt`
- 框架最小修复：`pyproject.toml`（`miracle` extra）、`src/agentbench_frame/tracking/run.py`（`_write_toml` 转义）

### 必须排除（已确认源码/测试无 `C:\Users\gongh` 硬编码，路径全部 env 驱动 via `paths.py`）
- `.smoke/`（全部 session：smoke/matrix/precheck/rank16build、Replay、构建产物、生成网页、临时日志、资产副本）
- `__pycache__/`、`.pytest_cache/`、uv 缓存
- 用户本机绝对路径（已通过 `paths.py` 参数化：`AGENTBENCH_ROOT`/`MIRACLE_IFELSE_DIR`/`AGENTBENCH_RESULTS`）
- 受保护策略副本、无关 DOTO 文件

### 硬编码审计结论
- `src/`、`vendor/`：**0 硬编码**。
- `tests/`、`tools/`：已全部改为 env 驱动（`paths.py`）；`test_driver.py` 的 Windows 路径 fixture 改为 `C:\Users\example\...`（非本机）。
- 双版本测试通过：**3.13 142 passed、3.11 142 passed**（设 `AGENTBENCH_RESULTS` env）。

## 三、Results 待提交载荷（独立 payload 目录，未复制到正式 Results 仓库）
位置：`docs/games/results_payload/`（仅 CI 必要文件，无本机路径/Replay/二进制/私有策略）。
- `runs/24_miracle/miracle_ifelse/<run_id>/{run.toml, summary.json, events.jsonl}`
- `MANIFEST.sha256`（载荷完整 SHA256 清单）
- `README.md`（载荷说明）
本地 schema/aggregate/report_builder 全通过；events 重算/summary/aggregate/web 均 10%。

## 四、最终提交计划（需用户授权的 git 动作）
### Framework
- 建议分支：`gongheng/24-miracle-adapter`（已存在）。
- 建议 commit message：`feat(24_miracle): external-Judge adapter + 32-game matrix (Plan A) + Windows portability`
- 文件清单：见第二节"应提交"。
### Results
- 建议分支：新分支 `gongheng/24-miracle-results`。
- 建议 commit message：`data(24_miracle): miracle_ifelse eval run (Plan A, 32 attempts, win_rate=0.1)`
- 载荷清单：`results_payload/` 内容（run.toml/summary.json/events.jsonl + MANIFEST）。
### 排除项及原因
`.smoke/`（本地运行产物/Replay/构建副本/生成网页）、本机绝对路径（env 驱动）、受保护策略副本、DOTO 无关文件——均不含评测契约所需数据，且含本机/私有信息。
### 本地复现/验证命令
```bash
# 测试（需 env）
AGENTBENCH_RESULTS=<results_repo> py -3.13 -m pytest tests/
AGENTBENCH_RESULTS=<results_repo> py -3.13 -m uv run --no-project --python 3.11 --with pytest --with psutil --with jinja2 python -m pytest tests/
# Results 四向验收（payload 目录）
py -3.13 -m agentbench_frame.cli data check --data-dir docs/games/results_payload
py -3.13 vendor/results_local/aggregate.py --data-dir docs/games/results_payload --output docs/games/results_payload/registry.toml
py -3.13 <results_repo>/scripts/report_builder.py --data-dir docs/games/results_payload --output <site>
```

## 五、尚需用户授权的动作
- `git add` / `commit`（Framework 工作仓库）
- `push` / 创建 PR
- 将 Results 载荷复制到正式 AgentBenchResults 仓库并上传
- 合并 main

**当前状态：仅差提交授权。** 所有离线审计、文档、载荷、本地 CI 验收均已完成；矩阵已 32/32 完成且四向一致；无重跑、无残留、无策略/Judge/协议修改。
