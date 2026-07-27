"""
Static site builder for AgentBench experiment reports.

Reads registry.toml + runs/*/summary.json, renders Jinja2 templates
into _site/.

The canonical implementation pattern is derived from the CI report builder
(see .worktrees/ci-report/scripts/report_builder.py).

Usage:
    builder = ReportBuilder(data_dir="./runs", output_dir="_site")
    builder.build()
"""

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional

from agentbench_frame.eval.curves import multi_axis_auc
from agentbench_frame.tracking.quality import inspect_event_file

try:
    import jinja2
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


class ReportBuilder:
    """Reads experiment data and builds a static HTML report site.

    The builder:
    1. Reads registry.toml (list of known games/agents)
    2. Walks runs/*/summary.json for experiment results
    3. Renders Jinja2 templates into _site/

    Usage:
        builder = ReportBuilder(data_dir="./experiments")
        builder.build()
    """

    def __init__(self,
                 data_dir: str = "./runs",
                 output_dir: str = "_site",
                 template_dir: Optional[str] = None):
        self.data_dir = os.path.abspath(data_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.template_dir = template_dir or os.path.join(
            os.path.dirname(__file__), "templates")

        self.registry: Dict[str, Any] = {}
        self.runs: List[Dict[str, Any]] = []
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.games: Dict[str, Dict[str, Any]] = {}

    def build(self):
        """Run the full build pipeline."""
        self._load_registry()
        self._load_runs()
        self._aggregate()

        if HAS_JINJA2:
            self._render()
        else:
            self._render_fallback()

    def _load_registry(self):
        """Load registry.toml if it exists."""
        registry_path = os.path.join(self.data_dir, "registry.toml")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r") as f:
                    content = f.read()
                self._parse_toml_registry(content)
            except Exception:
                pass

    def _parse_toml_registry(self, content: str):
        """Minimal TOML parser for registry format."""
        current_section = None
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                if current_section not in self.registry:
                    self.registry[current_section] = {}
                continue
            if "=" in line and current_section:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                self.registry[current_section][k] = v

    def _load_runs(self):
        """Walk runs/*/summary.json."""
        runs_dir = os.path.join(self.data_dir, "runs")
        if not os.path.isdir(runs_dir):
            runs_dir = self.data_dir

        for dirpath, dirnames, filenames in os.walk(runs_dir):
            for fname in filenames:
                if fname == "summary.json":
                    full_path = os.path.join(dirpath, fname)
                    try:
                        with open(full_path, "r") as f:
                            data = json.load(f)
                        run_dir = os.path.basename(dirpath)
                        data["_dir"] = run_dir
                        data["_path"] = full_path
                        events_path = os.path.join(dirpath, "events.jsonl")
                        events = []
                        if os.path.exists(events_path):
                            with open(events_path, encoding="utf-8") as event_file:
                                for line in event_file:
                                    try:
                                        value = json.loads(line)
                                    except (TypeError, json.JSONDecodeError):
                                        continue
                                    if isinstance(value, dict):
                                        events.append(value)
                        data["research"] = self._derive_research(data, events, events_path)
                        data.setdefault("win_rate", data["research"].get("benchmark_score", 0.0) or 0.0)
                        data.setdefault("total_episodes", 0)
                        data.setdefault("total_steps", 0)
                        data.setdefault("avg_reward_per_episode", 0.0)
                        data.setdefault("duration_s", 0.0)
                        self.runs.append(data)
                    except (json.JSONDecodeError, IOError):
                        continue

        # Sort by started_at (newest first)
        self.runs.sort(key=lambda r: r.get("started_at", 0), reverse=True)
        self._link_round2_learning_budgets()

    @staticmethod
    def _pair_dense_history(
        dense_history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        metrics = (
            "territory_share",
            "target_army",
            "army_margin",
            "coin_share",
            "net_main_pressure",
        )
        statistics = ("terminal", "time_average", "auc")
        grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for item in dense_history:
            if (
                item.get("phase") != "evaluation"
                or item.get("version") not in {"v1", "v2"}
                or not item.get("case_id")
            ):
                continue
            grouped.setdefault(
                str(item["case_id"]), {"v1": [], "v2": []}
            )[str(item["version"])].append(item)
        pairs = []
        for case_id in sorted(grouped):
            versions = grouped[case_id]
            v1_items = versions["v1"]
            v2_items = versions["v2"]
            v1 = v1_items[0] if len(v1_items) == 1 else None
            v2 = v2_items[0] if len(v2_items) == 1 else None
            if len(v1_items) > 1 or len(v2_items) > 1:
                status = "duplicate_version"
            elif v1 is None:
                status = "missing_v1"
            elif v2 is None:
                status = "missing_v2"
            else:
                status = "complete"
            survival_delta = None
            if status == "complete":
                before = v1.get("completed_rounds_survived")
                after = v2.get("completed_rounds_survived")
                if before is not None and after is not None:
                    survival_delta = float(after) - float(before)
            deltas: Dict[str, Dict[str, float | None]] = {}
            for metric in metrics:
                deltas[metric] = {}
                for statistic in statistics:
                    value = None
                    if status == "complete":
                        before = (v1.get(metric) or {}).get(statistic)
                        after = (v2.get(metric) or {}).get(statistic)
                        if before is not None and after is not None:
                            value = float(after) - float(before)
                    deltas[metric][statistic] = value
            pairs.append({
                "case_id": case_id,
                "status": status,
                "v1": v1,
                "v2": v2,
                "completed_rounds_survived_delta": survival_delta,
                "deltas": deltas,
            })
        return pairs

    @staticmethod
    def _sum_learning_budgets(
        budgets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        keys = (
            "learning_coding_agent_acts",
            "learning_episodes",
            "learning_env_steps",
            "learning_game_agent_decision_steps",
            "learning_primitive_commands",
            "learning_prompt_tokens",
            "learning_completion_tokens",
            "learning_total_tokens",
            "learning_time_s",
        )
        result = {}
        for key in keys:
            values = [budget.get(key) for budget in budgets]
            result[key] = (
                sum(values)
                if values and all(value is not None for value in values)
                else None
            )
        return result

    def _link_round2_learning_budgets(self) -> None:
        """Resolve cumulative global-curve coordinates across recovery runs."""
        by_id = {run.get("run_id"): run for run in self.runs}
        axis_keys = {
            "episode": "learning_episodes",
            "env_step": "learning_env_steps",
            "token": "learning_total_tokens",
            "time_s": "learning_time_s",
        }
        for run in self.runs:
            research = run.get("research", {})
            parent = run.get("parent_learning_budget")
            cumulative = run.get("cumulative_learning_budget")
            recovery_from = run.get("recovery_from_run_id")
            source = by_id.get(recovery_from, {}) if recovery_from else {}
            if recovery_from:
                parent = source.get("parent_learning_budget") or {}
                source_budget = dict(
                    run.get("source_learning_budget")
                    or source.get("budget")
                    or {}
                )
                cumulative = self._sum_learning_budgets(
                    [dict(parent), source_budget, dict(run.get("budget") or {})]
                )
                source_dense = [
                    {
                        **item,
                        "reused_from_run_id": recovery_from,
                    }
                    for item in source.get("research", {}).get(
                        "dense_history", []
                    )
                    if item.get("version") == "v1"
                    and item.get("phase") == "evaluation"
                ]
                current_dense = research.get("dense_history", [])
                current_keys = {
                    (item.get("case_id"), item.get("version"))
                    for item in current_dense
                }
                research["dense_history"] = [
                    item
                    for item in source_dense
                    if (item.get("case_id"), item.get("version"))
                    not in current_keys
                ] + current_dense
                research["dense_pairs"] = self._pair_dense_history(
                    research["dense_history"]
                )
                global_acts = cumulative.get(
                    "learning_coding_agent_acts"
                )
                if global_acts is not None:
                    failed_act = int(global_acts) - 1
                    research["score_history"] = [
                        {
                            "x": 0,
                            "score": run.get("raw_score"),
                            "act_id": None,
                        },
                        {
                            "x": 1,
                            "score": run.get("evo_score_1"),
                            "act_id": None,
                        },
                        {
                            "x": failed_act,
                            "score": None,
                            "act_id": None,
                            "status": "provider_failed",
                        },
                        {
                            "x": int(global_acts),
                            "score": run.get("evo_score_2"),
                            "act_id": None,
                        },
                    ]
            elif cumulative is None and parent:
                cumulative = self._sum_learning_budgets(
                    [dict(parent), dict(run.get("budget") or {})]
                )
            if cumulative is None:
                continue
            research["cumulative_learning_budget"] = cumulative
            scores = (
                run.get("raw_score"),
                run.get("evo_score_1"),
                run.get("evo_score_2"),
            )
            for axis, key in axis_keys.items():
                before = parent.get(key) if isinstance(parent, dict) else None
                after = cumulative.get(key)
                value = None
                if (
                    all(score is not None for score in scores)
                    and before is not None
                    and after is not None
                ):
                    raw, evo1, evo2 = (float(score) for score in scores)
                    if not recovery_from:
                        x1 = float(before)
                        x2 = float(after)
                        value = (
                            0.5 * (raw + evo1) * x1
                            + 0.5 * (evo1 + evo2) * (x2 - x1)
                        )
                research.setdefault("auc", {})[f"auc_{axis}"] = value
            if recovery_from:
                research.setdefault("auc", {})[
                    "auc_coding_agent_act"
                ] = None

    @staticmethod
    def _derive_research(summary: Dict[str, Any], events: List[Dict[str, Any]], events_path: str) -> Dict[str, Any]:
        """Add chart-ready fields while leaving summary/events untouched."""
        act_order = {}
        for event in events:
            if event.get("event_type", event.get("event")) == "coding_agent_act":
                act_id = event.get("act_id")
                if act_id and act_id not in act_order:
                    act_order[act_id] = len(act_order) + 1
        score_history = []
        auc_points = []
        budget = summary.get("budget") if isinstance(summary.get("budget"), dict) else {}
        for index, event in enumerate(events, start=1):
            if event.get("event_type", event.get("event")) not in {"act_evaluation", "evaluation", "benchmark_evaluation"}:
                continue
            score = next((event.get(key) for key in ("score", "benchmark_score", "evo_score")
                          if event.get(key) is not None), None)
            if score is None and event.get("evaluation_status") not in {"incomplete", "failed"}:
                continue
            phase = event.get("phase")
            explicit_act = event.get("global_coding_agent_act")
            if explicit_act is not None:
                act_x = explicit_act
            elif phase == "raw":
                act_x = 0
            elif phase == "evolved":
                act_x = budget.get("learning_coding_agent_acts", 1)
            else:
                act_x = act_order.get(event.get("act_id"), index)
            score_history.append({
                "x": act_x,
                "score": score,
                "act_id": event.get("act_id"),
            })
            auc_points.append({
                "coding_agent_act": event.get(
                    "global_coding_agent_act",
                    event.get("coding_agent_act", act_x),
                ),
                "episode": event.get("episode"),
                "env_step": event.get("env_step"),
                "token": event.get("token"),
                "time_s": event.get("time_s"),
                "score": score,
            })
        declared_score_history = summary.get("score_history")
        if isinstance(declared_score_history, list):
            score_history = [
                {
                    "x": index,
                    "score": score,
                    "act_id": None,
                    **({"status": "missing"} if score is None else {}),
                }
                for index, score in enumerate(declared_score_history)
            ]
        elif "evo_score_1" in summary or "evo_score_2" in summary:
            score_history = [
                {"x": 0, "score": summary.get("raw_score"), "act_id": None},
                {"x": 1, "score": summary.get("evo_score_1"), "act_id": None},
                {"x": 2, "score": summary.get("evo_score_2"), "act_id": None},
            ]
        if not score_history and summary.get("benchmark_score") is not None:
            score_history = [{
                "x": budget.get("learning_coding_agent_acts", 0),
                "score": summary.get("benchmark_score"),
                "act_id": None,
            }]
            auc_points = [{
                "coding_agent_act": budget.get("learning_coding_agent_acts", score_history[0]["x"]),
                "episode": budget.get("learning_episodes"),
                "env_step": budget.get("learning_env_steps"),
                "token": budget.get("learning_total_tokens"),
                "time_s": budget.get("learning_time_s"),
                "score": score_history[0]["score"],
            }]
        ig_history = []
        occupancy_history = []
        action_disagreement_history = []
        behavior_episode_history = []
        dense_history = []
        decision_class_history = []
        behavior_gate_event = None
        for index, event in enumerate(events, start=1):
            event_type = event.get("event_type", event.get("event"))
            if event_type == "policy_kl_trace":
                trace = event.get("trace")
                if isinstance(trace, list) and trace:
                    values = [float(value) for value in trace]
                    ig_history.append({"episode": event.get("episode", index),
                                       "ig": sum(values) / len(values),
                                       "decision_steps": event.get("decision_steps", len(values))})
            elif event_type == "occupancy":
                state_ids = event.get("state_ids")
                occupancy_history.append({
                    "episode": event.get("episode", index),
                    "shift": event.get("occupancy_shift"),
                    "state_count": event.get("state_count", len(state_ids) if isinstance(state_ids, list) else None),
                })
            elif event_type == "behavior_change":
                action_disagreement_history.append({
                    "version_before": event.get("version_before"),
                    "version_after": event.get("version_after"),
                    "mean": event.get("action_disagreement"),
                    "trace": event.get("action_disagreement_trace", []),
                    "policy_kl_status": event.get("policy_kl_status"),
                    "occupancy_shift": event.get("occupancy_shift"),
                    "episode_balanced_action_disagreement": event.get(
                        "episode_balanced_action_disagreement"
                    ),
                    "decision_balanced_action_disagreement": event.get(
                        "decision_balanced_action_disagreement",
                        event.get("action_disagreement"),
                    ),
                })
            elif event_type == "behavior_change_episode":
                behavior_episode_history.append({
                    "version_before": event.get("version_before"),
                    "version_after": event.get("version_after"),
                    "mean": event.get("action_disagreement"),
                    "trace": event.get("action_disagreement_trace", []),
                })
            elif event_type == "dense_episode_summary":
                dense_history.append({
                    "case_id": event.get("case_id"),
                    "version": event.get("version"),
                    "phase": event.get("phase"),
                    "suite": event.get("suite", "formal"),
                    "split": event.get("split"),
                    "outcome": event.get("outcome"),
                    "completed_rounds_survived": event.get(
                        "completed_rounds_survived"
                    ),
                    "territory_share": event.get("territory_share", {}),
                    "target_army": event.get("target_army", {}),
                    "army_margin": event.get("army_margin", {}),
                    "army_share": event.get("army_share", {}),
                    "coin_share": event.get("coin_share", {}),
                    "net_main_pressure": event.get(
                        "net_main_pressure", {}
                    ),
                    "artifact_ref": event.get("artifact_ref"),
                })
            elif event_type == "decision_class_summary":
                decision_class_history.append({
                    "version": event.get("version"),
                    "learning_id": event.get("learning_id"),
                    "total": event.get("total"),
                    "counts": event.get("counts", {}),
                    "rates": event.get("rates", {}),
                })
            elif event_type == "behavior_gate":
                behavior_gate_event = {
                    "passed": event.get("passed"),
                    "conditions": event.get("conditions", {}),
                    "improved_dense_metrics": event.get(
                        "improved_dense_metrics", []
                    ),
                    "dense_deltas": event.get("dense_deltas", {}),
                    "action_disagreement": event.get(
                        "action_disagreement"
                    ),
                }
        for aggregate in action_disagreement_history:
            episodes = [
                item
                for item in behavior_episode_history
                if item["version_before"] == aggregate["version_before"]
                and item["version_after"] == aggregate["version_after"]
            ]
            means = [
                float(item["mean"])
                for item in episodes
                if item["mean"] is not None
            ]
            decisions = [
                float(value)
                for item in episodes
                for value in item["trace"]
            ]
            if aggregate["episode_balanced_action_disagreement"] is None:
                aggregate["episode_balanced_action_disagreement"] = (
                    sum(means) / len(means) if means else None
                )
            if aggregate["decision_balanced_action_disagreement"] is None:
                aggregate["decision_balanced_action_disagreement"] = (
                    sum(decisions) / len(decisions) if decisions else None
                )
        raw_score = summary.get("raw_score")
        evo_score = summary.get(
            "evo_score_3",
            summary.get(
                "evo_score_2",
                summary.get(
                    "evo_score",
                    summary.get("benchmark_score"),
                ),
            ),
        )
        gain = summary.get(
            "gain_3",
            summary.get("gain_2", summary.get("gain")),
        )
        if gain is None and raw_score is not None and evo_score is not None:
            gain = float(evo_score) - float(raw_score)
        quality = inspect_event_file(events_path).to_dict() if os.path.exists(events_path) else {
            "total_lines": 0, "valid_events": 0, "warnings": []
        }
        auc = multi_axis_auc(auc_points) if auc_points else {}
        if summary.get("AUC_coding_agent_act") is not None:
            auc["auc_coding_agent_act"] = summary["AUC_coding_agent_act"]
        benchmark_score = summary.get("benchmark_score")
        if benchmark_score is None:
            benchmark_score = evo_score
        calibration = {
            "benchmark_id": summary.get("calibration_benchmark_id"),
            "status": summary.get("calibration_status"),
            "score": summary.get("calibration_score"),
            "wins": summary.get("calibration_wins"),
            "losses": summary.get("calibration_losses"),
            "draws": summary.get("calibration_draws"),
            "per_seat": summary.get("calibration_per_seat"),
            "target_range": summary.get("calibration_target_range"),
            "in_target_range": summary.get(
                "calibration_in_target_range"
            ),
        }
        return {
            "benchmark_score": benchmark_score,
            "raw_score": raw_score,
            "evo_score": evo_score,
            "gain": gain,
            "evaluation_status": summary.get(
                "evaluation_status"
            ) or summary.get(
                "status", "complete" if evo_score is not None else "unknown"
            ),
            "budget": budget,
            "score_history": score_history,
            "auc": auc,
            "ig_history": ig_history,
            "occupancy_history": occupancy_history,
            "action_disagreement_history": action_disagreement_history,
            "calibration": calibration,
            "dense_history": dense_history,
            "dense_pairs": ReportBuilder._pair_dense_history(dense_history),
            "decision_class_history": decision_class_history,
            "behavior_gate": (
                behavior_gate_event
                if behavior_gate_event is not None
                else summary.get("behavior_gate")
            ),
            "cumulative_learning_budget": summary.get(
                "cumulative_learning_budget"
            ),
            "benchmark_results": summary.get("benchmark_results", []),
            "event_counts": dict(Counter(event.get("event_type", event.get("event", "unknown")) for event in events)),
            "raw_event_count": len(events),
            "quality": quality,
        }

    def _aggregate(self):
        """Build per-agent and per-game aggregations."""
        for run in self.runs:
            game = run.get("game", "unknown")
            agent = run.get("agent", "unknown")

            if game not in self.games:
                self.games[game] = {"name": game, "runs": [], "agents": set()}
            self.games[game]["runs"].append(run)
            self.games[game]["agents"].add(agent)

            if agent not in self.agents:
                self.agents[agent] = {"name": agent, "runs": [], "games": set()}
            self.agents[agent]["runs"].append(run)
            self.agents[agent]["games"].add(game)

    def _render(self):
        """Render templates using Jinja2."""
        if not HAS_JINJA2:
            return

        os.makedirs(self.output_dir, exist_ok=True)

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(["html"]),
        )
        env.globals.update(
            fmt_num=lambda value: "—" if value is None else (f"{value:.2f}" if isinstance(value, float) else str(value)),
            fmt_pct=lambda value: "—" if value is None else f"{float(value) * 100:.1f}%",
            status_class=lambda value: {"complete": "success", "incomplete": "warning", "failed": "danger"}.get(value, "muted"),
        )

        ctx = self._build_context()
        pages = {
            "index.html": "index.html",
            "game.html": None,
            "agent.html": None,
            "compare.html": "compare.html",
        }

        for template_name, output_name in pages.items():
            try:
                template = env.get_template(template_name)
                html = template.render(**ctx)
                out_path = os.path.join(self.output_dir,
                                        output_name or template_name)
                with open(out_path, "w") as f:
                    f.write(html)
            except jinja2.TemplateNotFound:
                continue

        # Per-game pages
        for game_name, game_data in self.games.items():
            try:
                template = env.get_template("game.html")
                game_ctx = {**ctx, "game": game_data, "game_name": game_name}
                html = template.render(**game_ctx)
                out_path = os.path.join(self.output_dir,
                                        f"game_{game_name}.html")
                with open(out_path, "w") as f:
                    f.write(html)
            except (jinja2.TemplateNotFound, Exception):
                continue

        # Per-agent pages
        for agent_name, agent_data in self.agents.items():
            try:
                template = env.get_template("agent.html")
                agent_ctx = {**ctx, "agent": agent_data,
                             "agent_name": agent_name}
                html = template.render(**agent_ctx)
                safe_name = agent_name.replace("/", "_").replace(" ", "_")
                out_path = os.path.join(self.output_dir,
                                        f"agent_{safe_name}.html")
                with open(out_path, "w") as f:
                    f.write(html)
            except (jinja2.TemplateNotFound, Exception):
                continue

    def _render_fallback(self):
        """Fallback: generate simple HTML without Jinja2."""
        os.makedirs(self.output_dir, exist_ok=True)
        ctx = self._build_context()
        html = self._simple_html(ctx)
        out_path = os.path.join(self.output_dir, "index.html")
        with open(out_path, "w") as f:
            f.write(html)

    def _build_context(self) -> Dict[str, Any]:
        return {
            "title": "AgentBench Report",
            "registry": self.registry,
            "runs": self.runs,
            "games": self.games,
            "agents": self.agents,
            "num_runs": len(self.runs),
            "num_games": len(self.games),
            "num_agents": len(self.agents),
            "latest": self.runs[0] if self.runs else None,
            "latest_research": self.runs[0].get("research", {}) if self.runs else {},
        }

    def _simple_html(self, ctx: Dict[str, Any]) -> str:
        """Generate minimal HTML without Jinja2."""
        lines = ["<!DOCTYPE html>", "<html><head>",
                 "<meta charset='utf-8'>",
                 "<title>AgentBench Report</title>",
                 "<style>body{font-family:sans-serif;max-width:960px;"
                 "margin:0 auto;padding:20px}",
                 "table{border-collapse:collapse;width:100%}",
                 "th,td{border:1px solid #ccc;padding:8px;text-align:left}",
                 "th{background:#f5f5f5}</style>",
                 "</head><body>"]

        lines.append(f"<h1>{ctx['title']}</h1>")
        lines.append(f"<p>Runs: {ctx['num_runs']} | "
                     f"Games: {ctx['num_games']} | "
                     f"Agents: {ctx['num_agents']}</p>")

        if ctx["runs"]:
            lines.append("<h2>Recent Runs</h2>")
            lines.append("<table><tr><th>Run</th><th>Game</th><th>Agent</th>"
                         "<th>Episodes</th><th>Win Rate</th></tr>")
            for run in ctx["runs"][:50]:
                wr = run.get("win_rate", 0)
                lines.append(
                    f"<tr><td>{run.get('_dir','?')}</td>"
                    f"<td>{run.get('game','?')}</td>"
                    f"<td>{run.get('agent','?')}</td>"
                    f"<td>{run.get('total_episodes',0)}</td>"
                    f"<td>{wr:.1%}</td></tr>"
                )
            lines.append("</table>")

        lines.append("</body></html>")
        return "\n".join(lines)


def build_report(data_dir: str = "./runs", output_dir: str = "_site",
                 template_dir: Optional[str] = None):
    """Convenience function: build the report site."""
    builder = ReportBuilder(
        data_dir=data_dir,
        output_dir=output_dir,
        template_dir=template_dir,
    )
    builder.build()
    return builder.output_dir
