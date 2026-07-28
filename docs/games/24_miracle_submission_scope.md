# 24_miracle 提交范围（Submission Scope）

> 仅澄清未来提交边界；**本轮不 `git add`/commit/push/PR/上传**。.gitignore 当前仅含 `__pycache__/`（建议追加 `.smoke/` 等本地产物，但本轮不修改 .gitignore，留给用户决定）。

## 应进入未来提交（adapter + 测试 + vendor + 文档 + 框架最小修复）
- `src/agentbench_frame/games/__init__.py`
- `src/agentbench_frame/games/miracle/{__init__,result,driver,proctree,match_runner,runner,smoke_audit}.py`
- `tests/miracle/*.py`（含 `_fake_vendor.py`、`_proc_helper.py`）+ `conftest.py`
- `vendor/miracle_local/run_match.py`（vendor 补丁副本，含来源/哈希/修改摘要头部）
- `vendor/results_local/aggregate.py`（vendor 一行补丁副本，含上游 bug 说明）
- `tools/miracle_smoke.py`
- `docs/games/24_miracle_adapter_status.md`、`24_miracle_assets.json`、`24_miracle_roster_manifest.json`、`24_miracle_evaluation_protocol.draft.{json,md}`、`24_miracle_pre_matrix_audit.md`、`24_miracle_submission_scope.md`
- `docs/games/evidence/*`（红灯证据、双版本测试、vendor diff、smoke 三向验收 —— 建议提交为审计证据）
- 框架最小修复：`pyproject.toml`（`miracle` extra）、`src/agentbench_frame/tracking/run.py`（`_write_toml` 转义，含失败测试先行）

> 框架修改（`run.py`/`pyproject.toml`）属共享代码：已在报告中单列、最小、失败测试先行、未推送。提交时应附修改说明。

## 不应提交（本地运行产物 / 资产副本 / 生成网页 / 临时数据）
- `.smoke/sampleA/`、`.smoke/sampleB/`（sample AI 资产副本，本地 smoke 用）
- `.smoke/sessions/*/`（smoke session 全量产物：data/runs、work/trace+replay+stdout+stderr、site/ 生成网页、manifest、logs）
- 任何 `__pycache__/`、`.pytest_cache/`
- uv 托管 Python（`%APPDATA%/uv/...`，不在仓库内）

> 建议 `.gitignore` 追加：`.smoke/`、`.pytest_cache/`、`_site/`（已存在 `__pycache__/`）。本轮不修改 `.gitignore`。

## 未来上传到 AgentBenchResults（未授权不做）
- 目标目录契约：`runs/24_miracle/<agent>/<run_id>/{run.toml,summary.json,events.jsonl}` + 根 `registry.toml`（aggregate 产出）。
- 本轮 smoke 产物**不**上传：smoke 仅为基础设施验证（sample/ if-else vs sample），非正式评测。
- 正式矩阵产物上传需：用户授权 + 协议冻结 + Linux CI 上游 aggregate（非 Windows vendor 副本）重新聚合校验。

## Windows vendor 补丁 vs Linux CI 上游边界
- `vendor/results_local/aggregate.py`：仅本地 Windows 验收用（`as_posix` 一行）；**不是** Results 仓库上游代码。
- Linux 生产 CI 用上游 `aggregate.py`（正斜杠，无此 bug）；上传数据须能在 Linux CI 上游 aggregate + report_builder 下通过（schema/aggregate/web 一致）。
- `vendor/miracle_local/run_match.py`：仅本地 Windows 执行用；上游高翔 `run_match.py` 为 Linux 取向（`os.killpg`/`selectors`），Linux 环境应优先用上游或等价 Linux 适配。
