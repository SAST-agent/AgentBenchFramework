#!/usr/bin/env python3
"""Build a static research report from AgentBench run data.

The builder keeps ``summary.json`` as a convenient derived snapshot while
reading ``events.jsonl`` when available.  Unknown event types and malformed
individual lines are ignored so older and newer providers can share the CI
page without changing the raw data contract.
"""

import argparse
import json
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from markupsafe import Markup

try:
    import jinja2
except ImportError:
    jinja2 = None


@dataclass
class RunData:
    run_id: str
    game: str
    agent: str
    type: str
    path: str
    created: str = ""
    git_commit: str = ""
    metrics: dict = field(default_factory=dict)
    raw_summary: dict = field(default_factory=dict)
    raw_metadata: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    research: dict = field(init=False)

    def __post_init__(self):
        self.research = derive_research(self.raw_summary, self.metrics, self.events)

    def _metric(self, key: str):
        value = self.metrics.get(key)
        return value if value is not None else self.raw_summary.get(key)

    @property
    def best_elo(self): return self._metric("best_elo")
    @property
    def final_elo(self): return self._metric("final_elo")
    @property
    def wall_hours(self): return self._metric("wall_hours")
    @property
    def total_steps(self): return self._metric("total_steps")
    @property
    def win_rate(self): return self._metric("win_rate")
    @property
    def benchmark_score(self): return self.research["benchmark_score"]
    @property
    def raw_score(self): return self.research["raw_score"]
    @property
    def evo_score(self): return self.research["evo_score"]
    @property
    def gain(self): return self.research["gain"]
    @property
    def evaluation_status(self): return self.research["evaluation_status"]
    @property
    def elo_history(self): return self.raw_summary.get("elo_history", [])
    @property
    def h2h(self): return self.raw_summary.get("h2h", {})


@dataclass
class GameData:
    name: str
    agents: list = field(default_factory=list)
    runs: list = field(default_factory=list)


def _first(mapping: dict, *keys):
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def _load_events(run_path: Path) -> list[dict]:
    event_path = run_path / "events.jsonl"
    if not event_path.exists():
        return []
    events = []
    try:
        lines = event_path.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(record, dict):
            events.append(record)
    return events


def _normalise_score_history(raw_history: Any) -> list[dict]:
    if not isinstance(raw_history, list):
        return []
    normalised = []
    for index, item in enumerate(raw_history, start=1):
        if isinstance(item, dict):
            score = _first(item, "score", "benchmark_score", "evo_score", "y")
            x = _first(item, "x", "coding_agent_act", "act", "step")
        elif isinstance(item, (int, float)):
            score, x = item, index
        else:
            continue
        if score is not None:
            normalised.append({"x": x if x is not None else index, "score": score})
    return normalised


def _trapezoid_auc(points: list[dict]) -> float | None:
    """Integrate complete score points, breaking at explicit missing values."""
    total = 0.0
    previous = None
    complete_points = 0
    for point in points:
        score = point.get("score")
        x = point.get("x")
        if score is None or x is None:
            previous = None
            continue
        current = (float(x), float(score))
        complete_points += 1
        if previous is not None:
            dx = current[0] - previous[0]
            if dx < 0:
                raise ValueError("score history x values must be non-decreasing")
            total += dx * (current[1] + previous[1]) / 2.0
        previous = current
    return total if complete_points else None


def derive_research(summary: dict, metrics: dict, events: list[dict]) -> dict:
    """Derive chart-ready fields without replacing raw summary or events."""
    def metric(*keys):
        return _first(summary, *keys) if _first(summary, *keys) is not None else _first(metrics, *keys)

    budget = summary.get("budget")
    if not isinstance(budget, dict):
        budget = metrics.get("budget") if isinstance(metrics.get("budget"), dict) else {}
    budget = dict(budget)

    act_order = {}
    act_count = 0
    for event in events:
        if event.get("event_type", event.get("event")) != "coding_agent_act":
            continue
        act_id = event.get("act_id")
        if act_id and act_id not in act_order:
            act_count += 1
            act_order[act_id] = act_count

    score_history = []
    fallback_x = 0
    for event in events:
        event_type = event.get("event_type", event.get("event"))
        if event_type not in {"act_evaluation", "benchmark_evaluation", "evaluation"}:
            continue
        score = _first(event, "benchmark_score", "evo_score", "score", "evaluation_win_rate")
        fallback_x += 1
        act_id = event.get("act_id")
        x = act_order.get(act_id, fallback_x)
        if score is None:
            if event.get("evaluation_status") in {"incomplete", "failed"}:
                score_history.append({"x": x, "score": None, "act_id": act_id})
            continue
        score_history.append({
            "x": x,
            "score": score,
            "act_id": act_id,
        })
    if not score_history:
        score_history = _normalise_score_history(
            summary.get("score_history") or summary.get("benchmark_history")
        )
    if not score_history and metric("benchmark_score", "evo_score") is not None:
        score_history = [{
            "x": budget.get("learning_coding_agent_acts", 0),
            "score": metric("benchmark_score", "evo_score"),
            "act_id": None,
        }]
    auc_coding_agent_act = metric("auc_coding_agent_act")
    if auc_coding_agent_act is None:
        auc_coding_agent_act = _trapezoid_auc(score_history)

    ig_history = []
    for index, event in enumerate(events, start=1):
        if event.get("event_type", event.get("event")) != "policy_kl_trace":
            continue
        trace = event.get("trace")
        if not isinstance(trace, list):
            value = event.get("ig")
            steps = event.get("decision_steps", 0)
        else:
            values = [float(value) for value in trace if isinstance(value, (int, float))]
            if not values:
                continue
            value = sum(values) / len(values)
            steps = event.get("decision_steps", len(values))
        if value is None:
            continue
        ig_history.append({
            "episode": event.get("episode", index),
            "ig": value,
            "decision_steps": steps,
        })

    occupancy_history = []
    for index, event in enumerate(events, start=1):
        if event.get("event_type", event.get("event")) != "occupancy":
            continue
        shift = _first(event, "occupancy_shift", "shift", "kl")
        state_ids = event.get("state_ids")
        occupancy_history.append({
            "episode": event.get("episode", index),
            "shift": shift,
            "state_count": event.get("state_count", len(state_ids) if isinstance(state_ids, list) else None),
        })

    event_counts = Counter(event.get("event_type", event.get("event", "unknown")) for event in events)
    benchmark_score = metric("benchmark_score", "evo_score")
    raw_score = metric("raw_score", "performance_raw")
    evo_score = metric("evo_score", "performance_evo", "benchmark_score")
    gain = metric("gain", "performance_gain")
    if gain is None and raw_score is not None and evo_score is not None:
        gain = float(evo_score) - float(raw_score)
    status = metric("evaluation_status")
    if status is None:
        status = "complete" if benchmark_score is not None else "unknown"
    return {
        "benchmark_score": benchmark_score,
        "raw_score": raw_score,
        "evo_score": evo_score,
        "gain": gain,
        "auc_coding_agent_act": auc_coding_agent_act,
        "evaluation_status": status,
        "budget": budget,
        "score_history": score_history,
        "ig_history": ig_history,
        "occupancy_history": occupancy_history,
        "event_counts": dict(event_counts),
        "raw_event_count": len(events),
    }


def load_registry(data_dir: Path) -> list:
    path = data_dir / "registry.toml"
    if not path.exists():
        return []
    try:
        return tomllib.loads(path.read_text()).get("runs", [])
    except (OSError, tomllib.TOMLDecodeError):
        return []


def load_run(data_dir: Path, entry: dict) -> RunData:
    run_path = data_dir / entry["path"]
    raw_summary = {}
    summary_path = run_path / "summary.json"
    if summary_path.exists():
        try:
            raw_summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    raw_metadata = {}
    metadata_path = run_path / "run.toml"
    if metadata_path.exists():
        try:
            raw_metadata = tomllib.loads(metadata_path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            pass
    metrics = dict(entry.get("summary", {}))
    for key, value in raw_summary.items():
        if key not in {"budget", "elo_history", "h2h"}:
            metrics[key] = value
    return RunData(
        run_id=entry["run_id"],
        game=entry["game"],
        agent=entry["agent"],
        type=entry.get("type", "eval"),
        path=entry["path"],
        created=entry.get("created", raw_summary.get("created", "")),
        git_commit=entry.get("git_commit", raw_summary.get("git_commit", "")),
        metrics=metrics,
        raw_summary=raw_summary,
        raw_metadata=raw_metadata,
        events=_load_events(run_path),
    )


def load_compare(data_dir: Path) -> dict:
    path = data_dir / "reports" / "compare.toml"
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def group_by_game(runs: list[RunData]) -> dict[str, GameData]:
    games: dict[str, GameData] = {}
    for run in runs:
        game = games.setdefault(run.game, GameData(name=run.game))
        if run.agent not in game.agents:
            game.agents.append(run.agent)
        game.runs.append(run)
    return games


def _fmt_num(value, prec=2):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{prec}f}"
    return str(value)


def _fmt_pct(value):
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _status_class(status):
    return {"complete": "success", "incomplete": "warning", "failed": "danger"}.get(status, "muted")


def _dashboard(runs: list[RunData]) -> dict:
    latest = runs[0] if runs else None
    score_series = []
    ig_series = []
    for run in runs:
        scores = run.research["score_history"]
        if scores:
            score_series.append({
                "label": f"{run.agent} · {run.run_id[:8]}",
                "data": [{"x": p["x"], "y": p["score"]} for p in scores],
            })
        ig = run.research["ig_history"]
        if ig:
            ig_series.append({
                "label": f"{run.agent} · {run.run_id[:8]}",
                "data": [{"x": p["episode"], "y": p["ig"]} for p in ig],
            })
    complete = sum(run.evaluation_status == "complete" for run in runs)
    return {
        "latest": latest,
        "latest_research": latest.research if latest else {},
        "score_series": score_series,
        "ig_series": ig_series,
        "complete_runs": complete,
        "run_count": len(runs),
        "game_count": len({run.game for run in runs}),
        "agent_count": len({run.agent for run in runs}),
    }


def _public_run(run: RunData) -> dict:
    return {
        "run_id": run.run_id,
        "game": run.game,
        "agent": run.agent,
        "type": run.type,
        "created": run.created,
        "git_commit": run.git_commit,
        "metrics": run.metrics,
        "research": run.research,
        "elo_history": run.elo_history,
        "h2h": run.h2h,
    }


CHARTS_JS = r"""(function(){'use strict';
var palette=['#e6b86a','#77d4b5','#86a9f5','#d995c7','#e77b74','#84c8d8'];
function color(i){return palette[i%palette.length]}
function axes(){return{grid:{color:'rgba(145,163,181,.12)'},ticks:{color:'#8b9aad',font:{size:11}}}}
function tooltip(){return{backgroundColor:'#15202b',titleColor:'#f4f0e7',bodyColor:'#aab7c5',borderColor:'rgba(145,163,181,.25)',borderWidth:1,padding:11,cornerRadius:8}}
window.AgentBench=window.AgentBench||{};
window.AgentBench.lineChart=function(id,labels,datasets,extra){
 var node=document.getElementById(id);if(!node||typeof Chart==='undefined')return;
 var ds=datasets.map(function(d,i){var c=d.color||color(i);return{label:d.label,data:d.data,borderColor:c,backgroundColor:c+'22',borderWidth:2,pointRadius:3,pointHoverRadius:5,tension:.28,fill:false,spanGaps:false}});
 var options={responsive:true,maintainAspectRatio:false,animation:{duration:350},interaction:{intersect:false,mode:'index'},plugins:{legend:{position:'bottom',labels:{color:'#aab7c5',usePointStyle:true,padding:16}},tooltip:tooltip()},scales:{x:axes(),y:axes()}};
 if(extra)Object.assign(options,extra);new Chart(node.getContext('2d'),{type:'line',data:{labels:labels,datasets:ds},options:options});
};
window.AgentBench.scatterChart=function(id,datasets,extra){
 var node=document.getElementById(id);if(!node||typeof Chart==='undefined')return;
 var ds=datasets.map(function(d,i){var c=d.color||color(i);return{label:d.label,data:d.data,showLine:true,borderColor:c,backgroundColor:c,pointRadius:4,pointHoverRadius:6,borderWidth:2,tension:.25,spanGaps:false}});
 var options={responsive:true,maintainAspectRatio:false,animation:{duration:350},interaction:{intersect:false,mode:'nearest'},plugins:{legend:{position:'bottom',labels:{color:'#aab7c5',usePointStyle:true,padding:16}},tooltip:tooltip()},scales:{x:{...axes(),title:{display:true,text:'coding-agent act',color:'#8b9aad'}},y:{...axes(),min:0,max:1,title:{display:true,text:'benchmark score',color:'#8b9aad'}}}};
 if(extra)Object.assign(options,extra);new Chart(node.getContext('2d'),{type:'scatter',data:{datasets:ds},options:options});
};
})();"""


def build_site(data_dir, output, template_dir=None):
    if jinja2 is None:
        raise ImportError("jinja2 required: pip install jinja2")
    data_dir = Path(data_dir)
    output = Path(output)
    template_dir = Path(template_dir) if template_dir else Path(__file__).resolve().parent / "templates"
    registry = load_registry(data_dir)
    runs = sorted([load_run(data_dir, entry) for entry in registry], key=lambda run: run.created, reverse=True)
    games = group_by_game(runs)
    compare = load_compare(data_dir)
    dashboard = _dashboard(runs)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    def tojson(value):
        # Mark the JSON safe after escaping characters that can terminate a
        # script element.  A plain string would be HTML-escaped by Jinja and
        # produce ``&#34;`` inside executable chart configuration.
        encoded = json.dumps(value, ensure_ascii=False)
        encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        return Markup(encoded)

    env.globals.update(
        fmt_num=_fmt_num,
        fmt_pct=_fmt_pct,
        status_class=_status_class,
        tojson=tojson,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    (output / "assets" / "charts.js").write_text(CHARTS_JS)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def render(template_name, filename, depth=0, **extra):
        template = env.get_template(template_name)
        context = {
            "title": "AgentBench Research CI",
            "page": template_name.replace(".html", ""),
            "depth": depth,
            "gen_time": generated,
            "runs": runs,
            "games": games,
            "compare": compare,
            "dashboard": dashboard,
            **extra,
        }
        path = output / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.render(**context))

    render("index.html", "index.html")
    for game_name, game in sorted(games.items()):
        render("game.html", f"{game_name}/index.html", depth=1, game=game, game_runs=game.runs)
        for agent in sorted(game.agents):
            agent_runs = sorted([run for run in game.runs if run.agent == agent], key=lambda run: run.created)
            render("agent.html", f"{game_name}/{agent}/index.html", depth=2, game=game, agent=agent, agent_runs=agent_runs)
    render("compare.html", "compare.html")

    (output / "data.json").write_text(json.dumps({
        "runs": [_public_run(run) for run in runs],
        "games": {name: {"agents": game.agents, "run_count": len(game.runs)} for name, game in games.items()},
        "dashboard": {
            "complete_runs": dashboard["complete_runs"],
            "run_count": dashboard["run_count"],
            "score_series": dashboard["score_series"],
            "ig_series": dashboard["ig_series"],
        },
        "compare": compare,
    }, indent=2, ensure_ascii=False))
    print(f"Built: {len(runs)} runs, {len(games)} games -> {output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Build static research CI report")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output", default="./_site")
    parser.add_argument("--templates", default=None)
    args = parser.parse_args()
    build_site(args.data_dir, args.output, args.templates)


if __name__ == "__main__":
    main()
