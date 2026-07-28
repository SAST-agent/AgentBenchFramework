#!/usr/bin/env python3
"""Fail-closed 24_miracle research preflight.

The opt-in ``--legacy-smoke`` path is deprecated, non-authoritative historical
compatibility code.  It is never selected by the default parser or accepted as
research evidence.
"""
from __future__ import annotations

import argparse, difflib, hashlib, json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from agentbench_frame.games.miracle.iteration_protocol import (
    HumanAuthoredContentRequired,
    IterationBlockedError,
    LearningConfig,
    MatchConfig,
    default_bootstrap_template,
    preflight_learning,
    preflight_match,
    preflight_replay,
)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
        h.update(f.relative_to(p).as_posix().encode()); h.update(f.read_bytes())
    return h.hexdigest()

def copy(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst)

_NON_INFRA_INVALID = {"ai_crash", "ai_timeout"}


def run_pair(version: str, opponent: str, vdir: Path, odir: Path, judge: Path, vendor: Path, session: Path) -> dict:
    from agentbench_frame.games.miracle.runner import MiracleEvalRunner

    # Keep the framework Run agent path flat; version identity belongs in the
    # immutable version metadata, not in a path component.
    r = MiracleEvalRunner(agent=f"miracle_ifelse_{version}", opponent=opponent,
        evaluated_dir=vdir, opponent_dir=odir, n_games=2, data_dir=str(session / "data"),
        judge_dir=judge, vendor_script=vendor, framework_src=REPO / "src",
        work_dir=session / "work", timeout=10.0, wrapper_timeout_s=180.0,
        prefix=f"{version}_{opponent}", config={"version":version,"opponent":opponent})
    default_attempt = r._default_attempt_fn
    def guarded_attempt(**kwargs):
        attempt = default_attempt(**kwargs)
        # A game-level AI invalid is evidence and may proceed to the side swap.
        # Every other non-valid classification is infrastructure and must stop
        # before another game (and, in particular, before validation).
        if not attempt.valid and attempt.error_type not in _NON_INFRA_INVALID:
            raise RuntimeError(
                f"infrastructure anomaly in {attempt.game_id}: {attempt.error_type}"
            )
        return attempt
    r.attempt_fn = guarded_attempt
    summary = r.run()
    attempts=[]
    for a in r.attempts:
        attempts.append({"game_id":a.game_id,"valid":a.valid,"result":a.normalized_result,
            "steps":a.steps,"camp":a.evaluated_agent_camp,"error_type":a.error_type,
            "evidence_paths":a.evidence_paths,"process_cleanup":a.process_cleanup})
    return {"summary":summary,"attempts":attempts}

def totals(groups: dict) -> dict:
    atts=[a for g in groups.values() for a in g["attempts"]]
    valid=[a for a in atts if a["valid"]]; wins=[a for a in valid if a["result"]=="win"]
    return {"attempts":len(atts),"valid":len(valid),"invalid":len(atts)-len(valid),
      "wins":len(wins),"losses":sum(a["result"]=="loss" for a in valid),
      "draws":sum(a["result"]=="draw" for a in valid),
      "win_rate":(len(wins)/len(valid) if valid else None),"total_steps":sum(a["steps"] or 0 for a in valid)}

def _legacy_smoke_main(argv=None) -> int:
    from agentbench_frame.games.miracle.entry import resolve_ai_command
    from agentbench_frame.tracking import CodexProvider, Run
    from agentbench_frame.tracking.controller import CodingAgentController

    ap=argparse.ArgumentParser()
    ap.add_argument("--session-root",type=Path,required=True); ap.add_argument("--agentbench-root",type=Path,required=True)
    ap.add_argument("--ifelse-source",type=Path,required=True)
    ap.add_argument("--codex", nargs="+", required=True,
                    help="explicit provider argv; never fall back to PATH")
    a=ap.parse_args(argv); stamp=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S_%f")
    session=a.session_root / stamp
    if session.exists(): raise RuntimeError("refusing existing iteration session")
    judge=a.agentbench_root/"backend_sources/corpus/24_miracle/logic/judge_dev_logic"
    extracted=a.agentbench_root/"top_algorithms/corpus/24_miracle_final/extracted"
    r04=next(extracted.glob("rank04__*"),None); r09=next(extracted.glob("rank09__*"),None)
    if not ((judge/"main.py").exists() and r04 and r09 and (a.ifelse_source/"main.py").exists()):
        raise RuntimeError("required authority assets missing")
    session.mkdir(parents=True); (session/"work").mkdir(); (session/"data").mkdir()
    versions=session/"versions"; opponents=session/"opponents"; versions.mkdir(); opponents.mkdir()
    v0=versions/"v0"; copy(a.ifelse_source,v0); copy(r04,opponents/"rank04"); copy(r09,opponents/"rank09")
    # rank09 must be genuinely buildable/launchable before the first game.
    build=subprocess.run(["make"],cwd=opponents/"rank09",capture_output=True,text=True,timeout=180)
    if build.returncode or not (opponents/"rank09"/"main.exe").exists():
        (session/"precheck_rank09.json").write_text(json.dumps({"returncode":build.returncode,"stdout":build.stdout,"stderr":build.stderr}),encoding="utf-8")
        raise RuntimeError("rank09 compile precheck failed")
    for name in ("rank04","rank09"):
        cmd=resolve_ai_command(opponents/name); p=subprocess.Popen(cmd,cwd=opponents/name,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try: p.communicate(timeout=2)
        except subprocess.TimeoutExpired: p.kill(); p.communicate()
    history=a.agentbench_root/"24_miracle_results_migration_preview_v3/data/runs/24_miracle/miracle_ifelse/20260722-001929_52bd14/events.jsonl"
    hist=[json.loads(x) for x in history.read_text(encoding="utf-8").splitlines() if x.strip()]
    historic={f"rank{n:02d}":[x for x in hist if x.get("rank")==n and x.get("valid")] for n in (4,9)}
    if not all(len(x)>=2 for x in historic.values()): raise RuntimeError("rank04/rank09 historical-validity check failed")
    population={"rank04":{"role":"train","sha256":sha(opponents/"rank04")},"rank09":{"role":"validation","sha256":sha(opponents/"rank09")}}
    meta={"version_id":"miracle_ifelse/v0","parent":None,"source_sha256":sha(a.ifelse_source),"workspace_sha256":sha(v0),"entry_command":resolve_ai_command(v0),"created_at":datetime.now(timezone.utc).isoformat(),"immutable":True}
    (session/"v0.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); (session/"population.json").write_text(json.dumps(population,indent=2),encoding="utf-8")
    vendor=REPO/"vendor/miracle_local/run_match.py"
    g0={"train":run_pair("v0","rank04",v0,opponents/"rank04",judge,vendor,session),"validation":run_pair("v0","rank09",v0,opponents/"rank09",judge,vendor,session)}
    (session/"v0_evaluation.json").write_text(json.dumps({"groups":g0,"totals":totals(g0)},indent=2,default=str),encoding="utf-8")
    # Only training evidence is exposed to the provider.
    replay={"rules":"Score more than the opposing player; no deterministic seed.","allowed_for_strategy":True,
      "episodes":[{"game_id":x["game_id"],"camp":x["camp"],"valid":x["valid"],"result":x["result"],"steps":x["steps"],"error_type":x["error_type"]} for x in g0["train"]["attempts"]],
      "forbidden":"rank09 validation replay, result, summary, and all opponent source"}
    candidate=session/"workspace_for_agent"; candidate.mkdir(); shutil.copy2(v0/"main.py",candidate/"main.py"); (candidate/"TRAINING_REPLAY.json").write_text(json.dumps(replay,indent=2),encoding="utf-8")
    tracking=Run.start("24_miracle","miracle_ifelse",run_type="rule_iter",data_dir=str(session/"tracking"),config={"information_gain":None,"information_gain_status":"pending_authoritative_definition"})
    prompt="Read only TRAINING_REPLAY.json and main.py in this workspace. Make one minimal rule improvement in main.py only. Do not read validation data, network, or external files; do not change protocols, timeouts, framework, or opponents."
    controller=CodingAgentController(CodexProvider(executable=a.codex,timeout_s=1800),recorder=tracking._act_recorder,budget=tracking._budget,budget_phase="learning",raw_output_dir=str(session/"provider_output"))
    act=controller.run_act({"prompt":prompt,"workspace_root":str(candidate),"sandbox":"workspace-write"},workspace_root=str(candidate),version_before="v0")
    tracking.finish(); (session/"coding_agent_act.json").write_text(json.dumps(act.__dict__,indent=2,default=str),encoding="utf-8")
    if act.status != "completed": raise RuntimeError(f"provider status uncertain or failed: {act.status}")
    before=(v0/"main.py").read_text(encoding="utf-8").splitlines(keepends=True); after=(candidate/"main.py").read_text(encoding="utf-8").splitlines(keepends=True)
    diff="".join(difflib.unified_diff(before,after,fromfile="v0/main.py",tofile="v1/main.py")); (session/"workspace.diff").write_text(diff,encoding="utf-8")
    v1=versions/"v1"; copy(v0,v1); shutil.copy2(candidate/"main.py",v1/"main.py"); diff_hash=hashlib.sha256(diff.encode()).hexdigest(); m1={"version_id":"miracle_ifelse/v1","parent":"miracle_ifelse/v0","source_sha256":sha(v1),"workspace_sha256":sha(v1),"diff_sha256":diff_hash,"provider_invocation_id":act.act_id,"created_at":datetime.now(timezone.utc).isoformat(),"immutable":True}
    (session/"v1.json").write_text(json.dumps(m1,indent=2),encoding="utf-8")
    g1={"train":run_pair("v1","rank04",v1,opponents/"rank04",judge,vendor,session),"validation":run_pair("v1","rank09",v1,opponents/"rank09",judge,vendor,session)}
    t0,t1=totals(g0),totals(g1); report={"v0":t0,"v1":t1,"raw":t0["win_rate"],"evo":t1["win_rate"],"gain":(t1["win_rate"]-t0["win_rate"] if t0["win_rate"] is not None and t1["win_rate"] is not None else None),"small_sample_note":"Non-deterministic smoke only; not statistically significant; v1 degradation does not invalidate the loop.","agent_episode_read":2,"agent_step_read":sum(x["steps"] or 0 for x in g0["train"]["attempts"]),"information_gain":None,"information_gain_status":"pending_authoritative_definition"}
    (session/"final_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(session); return 0


def build_research_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a read-only fake-only 24_miracle iteration preflight stage. "
            "No Judge, opponent, provider, match, or session is started."
        )
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    def add_match_inputs(stage: argparse.ArgumentParser) -> None:
        stage.add_argument("--champion-manifest", type=Path, required=True)
        stage.add_argument("--human-replay-skill", type=Path, required=True)
        stage.add_argument("--match-plan", type=Path, required=True)

    plan = stages.add_parser("plan", help="validate the independently approved match plan")
    add_match_inputs(plan)
    replay = stages.add_parser(
        "replay", help="validate captured replay evidence and its independent approval"
    )
    add_match_inputs(replay)
    replay.add_argument("--replay-evidence-manifest", type=Path, required=True)
    learning = stages.add_parser(
        "learning", help="validate approved train evidence before an Agent may read it"
    )
    add_match_inputs(learning)
    learning.add_argument("--replay-evidence-manifest", type=Path, required=True)
    return parser


def _research_main(argv=None) -> int:
    args = build_research_parser().parse_args(argv)
    match_config = MatchConfig(
        game="24_miracle",
        bootstrap=default_bootstrap_template(),
        champion_manifest_path=args.champion_manifest,
        human_replay_skill_path=args.human_replay_skill,
        match_plan_path=args.match_plan,
    )
    if args.stage == "plan":
        context = preflight_match(match_config)
    else:
        learning_config = LearningConfig(
            match=match_config,
            replay_evidence_manifest_path=args.replay_evidence_manifest,
        )
        context = (
            preflight_replay(learning_config)
            if args.stage == "replay"
            else preflight_learning(learning_config)
        )
    print(
        json.dumps(
            {
                "status": "fake-only preflight complete",
                "stage": args.stage,
                "lifecycle_state": context.state.value,
                "authoritative_execution": "blocked",
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv=None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "--legacy-smoke":
        return _legacy_smoke_main(values[1:])
    try:
        return _research_main(values)
    except (HumanAuthoredContentRequired, IterationBlockedError, OSError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
