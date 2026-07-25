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
import math
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


def _strict_nonnegative_number(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


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
                                    except (TypeError, ValueError):
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
            act_x = act_order.get(event.get("act_id"), index)
            score_history.append({
                "x": act_x,
                "score": score,
                "act_id": event.get("act_id"),
            })
            auc_points.append({
                "coding_agent_act": event.get("coding_agent_act", act_x),
                "episode": event.get("episode"),
                "env_step": event.get("env_step"),
                "token": event.get("token"),
                "time_s": event.get("time_s"),
                "score": score,
            })
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
        for index, event in enumerate(events, start=1):
            event_type = event.get("event_type", event.get("event"))
            if event_type == "policy_kl_trace":
                trace = event.get("trace")
                values = []
                valid_trace = isinstance(trace, list) and bool(trace)
                if valid_trace:
                    for value in trace:
                        number = _strict_nonnegative_number(value)
                        if number is None:
                            valid_trace = False
                            break
                        values.append(number)
                declared_status = event.get("measurement_status")
                decision_steps = event.get(
                    "decision_steps",
                    len(trace) if isinstance(trace, list) else None,
                )
                decision_steps_present = "decision_steps" in event
                aligned = (
                    isinstance(decision_steps, int)
                    and not isinstance(decision_steps, bool)
                    and isinstance(trace, list)
                    and decision_steps == len(trace)
                )
                decisions_present = "decisions" in event
                decisions = event.get("decisions")
                if decisions_present:
                    aligned = (
                        aligned
                        and decision_steps_present
                        and isinstance(decisions, list)
                        and len(decisions) == len(trace)
                    )
                    if aligned:
                        for decision, value in zip(decisions, values):
                            local_value = (
                                decision.get("local_policy_kl")
                                if isinstance(decision, dict)
                                else None
                            )
                            local_number = _strict_nonnegative_number(
                                local_value
                            )
                            if (
                                local_number is None
                                or not math.isclose(
                                    local_number,
                                    value,
                                    rel_tol=0.0,
                                    abs_tol=1e-12,
                                )
                            ):
                                aligned = False
                                break
                status_allows_complete = (
                    declared_status == "complete"
                    if decisions_present
                    else declared_status in {None, "complete"}
                )
                complete = valid_trace and aligned and status_allows_complete
                trajectory_kl_episode = sum(values) if complete else None
                if (
                    trajectory_kl_episode is not None
                    and not math.isfinite(trajectory_kl_episode)
                ):
                    complete = False
                    trajectory_kl_episode = None
                if complete:
                    display_status = "complete"
                elif (
                    isinstance(declared_status, str)
                    and declared_status
                    and declared_status != "complete"
                ):
                    display_status = declared_status
                else:
                    display_status = "incomplete"
                mean_local_policy_kl = (
                    trajectory_kl_episode / len(values)
                    if trajectory_kl_episode is not None
                    else None
                )
                ig_history.append({
                    "episode": event.get("episode", index),
                    "trajectory_kl_episode": trajectory_kl_episode,
                    "mean_local_policy_kl": mean_local_policy_kl,
                    "ig": trajectory_kl_episode,
                    "decision_steps": decision_steps,
                    "status": display_status,
                    "estimand": event.get(
                        "estimand", "legacy_unspecified"
                    ),
                })
            elif event_type == "occupancy":
                state_ids = event.get("state_ids")
                occupancy_history.append({
                    "episode": event.get("episode", index),
                    "shift": event.get("occupancy_shift"),
                    "state_count": event.get("state_count", len(state_ids) if isinstance(state_ids, list) else None),
                })
        raw_score = summary.get("raw_score")
        evo_score = summary.get("evo_score", summary.get("benchmark_score"))
        gain = summary.get("gain")
        if gain is None and raw_score is not None and evo_score is not None:
            gain = float(evo_score) - float(raw_score)
        quality = inspect_event_file(events_path).to_dict() if os.path.exists(events_path) else {
            "total_lines": 0, "valid_events": 0, "warnings": []
        }
        return {
            "benchmark_score": summary.get("benchmark_score", evo_score),
            "raw_score": raw_score,
            "evo_score": evo_score,
            "gain": gain,
            "evaluation_status": summary.get("evaluation_status", "complete" if evo_score is not None else "unknown"),
            "budget": budget,
            "score_history": score_history,
            "auc": multi_axis_auc(auc_points) if auc_points else {},
            "ig_history": ig_history,
            "ig_chart": ReportBuilder._trajectory_kl_chart(ig_history),
            "occupancy_history": occupancy_history,
            "benchmark_results": summary.get("benchmark_results", []),
            "event_counts": dict(Counter(event.get("event_type", event.get("event", "unknown")) for event in events)),
            "raw_event_count": len(events),
            "quality": quality,
        }

    @staticmethod
    def _trajectory_kl_chart(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build dependency-free SVG coordinates while preserving missing gaps."""

        width = 560
        height = 180
        padding = 24
        complete_values = [
            float(point["trajectory_kl_episode"])
            for point in history
            if point.get("trajectory_kl_episode") is not None
        ]
        y_max = max(complete_values, default=0.0)
        scale_max = y_max if y_max > 0.0 else 1.0
        denominator = max(1, len(history) - 1)
        segments = []
        current = []
        for index, point in enumerate(history):
            value = point.get("trajectory_kl_episode")
            if value is None:
                if current:
                    segments.append(current)
                    current = []
                continue
            x = padding + index * (width - 2 * padding) / denominator
            y = height - padding - float(value) * (
                height - 2 * padding
            ) / scale_max
            current.append({
                "x": round(x, 2),
                "y": round(y, 2),
                "episode": point.get("episode"),
                "value": float(value),
            })
        if current:
            segments.append(current)
        return {
            "width": width,
            "height": height,
            "y_max": y_max,
            "segments": [
                {
                    "points": " ".join(
                        f"{node['x']},{node['y']}" for node in segment
                    ),
                    "nodes": segment,
                }
                for segment in segments
            ],
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

            research = ctx["latest_research"]
            lines.append("<h2>Information gain</h2>")
            lines.append(
                "<p>Trajectory KL (nats / episode); "
                "Mean local policy KL (nats / decision)</p>"
            )
            chart = research.get("ig_chart", {})
            segments = chart.get("segments", [])
            lines.append(
                '<svg aria-label="Trajectory KL by episode" '
                f'data-segment-count="{len(segments)}" '
                f'viewBox="0 0 {chart.get("width", 560)} {chart.get("height", 180)}">'
            )
            for segment in segments:
                lines.append(
                    f'<polyline fill="none" stroke="#277c68" '
                    f'points="{segment.get("points", "")}"/>'
                )
            lines.append("</svg>")
            lines.append(
                "<table><tr><th>Episode</th><th>Trajectory KL</th>"
                "<th>Mean local policy KL</th><th>Status</th></tr>"
            )
            for point in research.get("ig_history", []):
                trajectory_value = point.get("trajectory_kl_episode")
                mean_value = point.get("mean_local_policy_kl")
                trajectory_text = (
                    f"{trajectory_value:.2f}"
                    if trajectory_value is not None
                    else "missing"
                )
                mean_text = (
                    f"{mean_value:.2f}" if mean_value is not None else "missing"
                )
                lines.append(
                    f"<tr><td>{point.get('episode')}</td>"
                    f"<td>{trajectory_text}</td><td>{mean_text}</td>"
                    f"<td>{point.get('status')}</td></tr>"
                )
            lines.append("</table>")
            lines.append(
                f"<p>AUC / act: "
                f"{research.get('auc', {}).get('auc_coding_agent_act', '—')}</p>"
            )
            lines.append(
                f"<p>{research.get('raw_event_count', 0)} raw event records</p>"
            )
            for result in research.get("benchmark_results", []):
                lines.append(f"<p>{result.get('case_id', 'unknown case')}</p>")

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
