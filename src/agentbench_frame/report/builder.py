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
import shutil
from typing import Any, Dict, List, Optional

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

        # Embed per-run figures (copied into the site) so templates can
        # reference them without depending on AGENTBENCH_DATA at serve time.
        # Must run before _render so run["_figures"] is set for templates.
        self._collect_figures()

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
                        data["_run_dir"] = dirpath
                        self.runs.append(data)
                    except (json.JSONDecodeError, IOError):
                        continue

        # Sort by started_at (newest first)
        self.runs.sort(key=lambda r: r.get("started_at", 0), reverse=True)

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

    def _collect_figures(self):
        """Copy each run's ``figures/*.png`` into the site.

        Site layout: ``_site/figures/{game}/{agent}/{run_id}/<name>.png``.
        Sets ``run["_figures"]`` to the site-relative paths (HTML-safe forward
        slashes) so templates can render ``<img>`` tags; runs without figures
        keep no key.
        """
        for run in self.runs:
            run_dir = run.get("_run_dir")
            if not run_dir:
                continue
            figs_dir = os.path.join(run_dir, "figures")
            if not os.path.isdir(figs_dir):
                continue
            game = run.get("game", "unknown")
            agent = run.get("agent", "unknown")
            run_id = run.get("_dir", "unknown")
            rel_root = os.path.join("figures", game, agent, run_id)
            dest = os.path.join(self.output_dir, rel_root)
            os.makedirs(dest, exist_ok=True)
            rel_figs = []
            for name in sorted(os.listdir(figs_dir)):
                if not name.lower().endswith(".png"):
                    continue
                shutil.copy2(os.path.join(figs_dir, name), os.path.join(dest, name))
                rel_figs.append(os.path.join(rel_root, name).replace("\\", "/"))
            if rel_figs:
                run["_figures"] = rel_figs

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
