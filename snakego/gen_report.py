import json, os, glob, html

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "agentbench_data", "runs", "26_snakego")
OUT = os.path.join(_HERE, "_site")

runs = []
for d in glob.glob(os.path.join(DATA, "*", "*")):
    if not os.path.isdir(d):
        continue
    sp = os.path.join(d, "summary.json")
    tp = os.path.join(d, "run.toml")
    if not os.path.exists(sp):
        continue
    with open(sp, encoding="utf-8") as f:
        s = json.load(f)
    s["_dir"] = os.path.basename(d)
    # read weights from run.toml [config]
    if os.path.exists(tp):
        cfg = {}
        in_cfg = False
        for line in open(tp, encoding="utf-8"):
            line = line.strip()
            if line == "[config]":
                in_cfg = True
                continue
            if line.startswith("[") and line.endswith("]"):
                in_cfg = False
                continue
            if in_cfg and "=" in line:
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip().strip('"')
        s["_cfg"] = cfg
    else:
        s["_cfg"] = {}
    runs.append(s)

os.makedirs(OUT, exist_ok=True)

agents = {}
for r in runs:
    a = r.get("agent", "?")
    agents.setdefault(a, []).append(r)

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
max-width:1100px;margin:0 auto;padding:20px;color:#333;background:#fff}
a{color:#0366d6;text-decoration:none}a:hover{text-decoration:underline}
h1{border-bottom:2px solid #eee;padding-bottom:10px}
table{border-collapse:collapse;width:100%;margin:10px 0}
th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left}
th{background:#f5f5f5;font-weight:600}
tr:nth-child(even){background:#fafafa}
.card{display:inline-block;margin:10px;padding:15px 25px;border:1px solid #dee2e6;
border-radius:6px;text-align:center}
.card .v{font-size:2em;font-weight:bold;color:#0366d6}
.card .l{font-size:.85em;color:#666;margin-top:5px}
.ig{color:#e8590c;font-weight:600}
.wr{font-weight:600}
.ok{color:#28a745}.lo{color:#dc3545}
nav{background:#f8f9fa;padding:10px 20px;border-radius:6px;margin-bottom:20px}
nav a{margin-right:15px}
pre{background:#f6f8fa;padding:12px;border-radius:6px;overflow-x:auto;font-size:.85em}
"""

def pct(v):
    try: return "%.1f%%" % (float(v) * 100)
    except: return str(v)

# ---- index.html ----
rows = ""
for r in sorted(runs, key=lambda x: x.get("_dir", "")):
    wr = r.get("win_rate", 0)
    cls = "ok" if wr >= 0.5 else "lo"
    ig = r.get("ig_kl")
    ig_str = "%.4f" % ig if ig is not None else "N/A"
    rows += '<tr><td><code>%s</code></td><td>%s</td><td><a href="agent_%s.html">%s</a></td><td>%s</td><td class="%s wr">%s</td><td>%d</td><td class="ig">%s</td></tr>\n' % (
        r.get("_dir",""), r.get("game",""), r.get("agent",""), r.get("agent",""),
        r.get("run_type",""), cls, pct(wr), r.get("total_steps",0), ig_str)

idx_html = '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SnakeGo Report</title><style>%s</style></head><body>\n<nav><a href="index.html">Home</a><a href="compare.html">Compare</a></nav>\n<h1>SnakeGo (26_snakego) Report</h1>\n<div><div class="card"><div class="v">%d</div><div class="l">Runs</div></div><div class="card"><div class="v">%d</div><div class="l">Agents</div></div><div class="card"><div class="v">%d</div><div class="l">Total Episodes</div></div></div>\n<h2>Runs</h2>\n<table><tr><th>Run ID</th><th>Game</th><th>Agent</th><th>Type</th><th>Win Rate</th><th>Steps</th><th>IG (KL)</th></tr>\n%s</table>\n</body></html>' % (
    CSS, len(runs), len(agents), sum(r.get("total_episodes",0) for r in runs), rows)

with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
    f.write(idx_html)

# ---- compare.html ----
cmp_rows = ""
for agent_name in sorted(agents.keys()):
    aruns = agents[agent_name]
    avg_wr = sum(r.get("win_rate",0) for r in aruns) / max(1,len(aruns))
    best_wr = max(r.get("win_rate",0) for r in aruns)
    total_ep = sum(r.get("total_episodes",0) for r in aruns)
    total_st = sum(r.get("total_steps",0) for r in aruns)
    ig_vals = [r.get("ig_kl") for r in aruns if r.get("ig_kl") is not None]
    avg_ig = "%.4f" % (sum(ig_vals)/len(ig_vals)) if ig_vals else "N/A"
    cmp_rows += '<tr><td><a href="agent_%s.html">%s</a></td><td>%d</td><td class="wr">%.1f%%</td><td>%.1f%%</td><td>%d</td><td>%d</td><td class="ig">%s</td></tr>\n' % (
        agent_name, agent_name, len(aruns), avg_wr*100, best_wr*100, total_ep, total_st, avg_ig)

cmp_html = '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Compare</title><style>%s</style></head><body>\n<nav><a href="index.html">Home</a><a href="compare.html">Compare</a></nav>\n<h1>Agent Comparison</h1>\n<table><tr><th>Agent</th><th>Runs</th><th>Avg Win Rate</th><th>Best Win Rate</th><th>Total Episodes</th><th>Total Steps</th><th>Avg IG (KL)</th></tr>\n%s</table>\n</body></html>' % (CSS, cmp_rows)

with open(os.path.join(OUT, "compare.html"), "w", encoding="utf-8") as f:
    f.write(cmp_html)

# ---- per-agent pages ----
for agent_name, aruns in agents.items():
    detail = ""
    for r in sorted(aruns, key=lambda x: x.get("_dir","")):
        wr = r.get("win_rate",0)
        cls = "ok" if wr >= 0.5 else "lo"
        ig = r.get("ig_kl")
        ig_str = "%.4f" % ig if ig is not None else "N/A"
        detail += '<tr><td><code>%s</code></td><td>%s</td><td class="%s">%s</td><td>%d</td><td>%d</td><td class="ig">%s</td></tr>\n' % (
            r.get("_dir",""), r.get("run_type",""), cls, pct(wr),
            r.get("total_episodes",0), r.get("total_steps",0), ig_str)

    # h2h table
    h2h = r.get("h2h", {}) if aruns else {}
    h2h_html = ""
    for src, targets in h2h.items():
        for tgt, val in targets.items():
            h2h_html += '<tr><td>%s</td><td>%s</td><td>%.1f%%</td></tr>\n' % (src, tgt, val*100)

    # weights from config
    cfg_html = ""
    for r2 in aruns:
        cfg = r2.get("_cfg", {})
        if cfg:
            cfg_html += '<h4>%s</h4><pre>' % r2.get("_dir","")
            for k in sorted(cfg.keys()):
                if k not in ("lesson","ig_status","ig_kl","source","player_rank","opponents","seeds","note"):
                    cfg_html += "%s = %s\n" % (k, cfg[k])
            cfg_html += '</pre>\n'

    page = '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>%s</title><style>%s</style></head><body>\n<nav><a href="index.html">Home</a><a href="compare.html">Compare</a></nav>\n<h1>Agent: %s</h1>\n<h2>Runs (%d)</h2>\n<table><tr><th>Run ID</th><th>Type</th><th>Win Rate</th><th>Episodes</th><th>Steps</th><th>IG (KL)</th></tr>\n%s</table>\n%s\n%s\n</body></html>' % (
        agent_name, CSS, agent_name, len(aruns), detail,
        ("<h2>Head-to-Head</h2>\n<table><tr><th>Agent</th><th>vs</th><th>Win Rate</th></tr>\n%s</table>" % h2h_html) if h2h_html else "",
        ("<h2>Weights / Config</h2>\n%s" % cfg_html) if cfg_html else "")

    safe = agent_name.replace("/", "_")
    with open(os.path.join(OUT, "agent_%s.html" % safe), "w", encoding="utf-8") as f:
        f.write(page)

# ---- game page ----
game_page = '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>26_snakego</title><style>%s</style></head><body>\n<nav><a href="index.html">Home</a><a href="compare.html">Compare</a></nav>\n<h1>Game: 26_snakego</h1>\n<h2>Agents</h2>\n<table><tr><th>Agent</th><th>Runs</th></tr>\n%s</table>\n</body></html>' % (
    CSS, "".join('<tr><td><a href="agent_%s.html">%s</a></td><td>%d</td></tr>' % (a, a, len(aruns)) for a, aruns in sorted(agents.items())))

with open(os.path.join(OUT, "game_26_snakego.html"), "w", encoding="utf-8") as f:
    f.write(game_page)

print("Report generated at: %s" % OUT)
print("Files: index.html, compare.html, game_26_snakego.html, + %d agent pages" % len(agents))
