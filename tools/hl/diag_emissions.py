"""Diagnostic: probe v1 vs threshold-edited v2 over the recorded ν and dump
emissions per sample, to see whether the edit changes any emitted primitive
(and on which samples). Resolves whether real-run KL=0 is (a) the edit not
affecting reached states (honest) or (b) replay not reconstructing the model.
"""
from __future__ import annotations
import os, sys, shutil, tempfile
from pathlib import Path
os.environ.setdefault("PYTHONPATH", "src"); sys.path.insert(0, "src")

from agentbench_frame.hl.codebase import HLCodebase
from agentbench_frame.hl.adapter import candidate_command
from agentbench_frame.hl.reference import ReferenceStateSet
from agentbench_frame.hl.probe import ReferenceProbe
from agentbench_frame.hl.distribution import enumerate_legal_actions

NU = "agentbench_data/reference/nu-recorded.json"
V1 = Path("src/agentbench_frame/lostspace/candidates/v1")

def edit_threshold(workspace):
    # Skip the FIRST view_box call so v2's first emitted primitive becomes
    # view_box("Materials","Kit") instead of view_box("Box","Key") -> changes
    # the first action on EVERY reached decision point -> real KL>0.
    p = workspace / "agent.py"
    t = p.read_text(encoding="utf-8")
    n = t.replace(
        'if self.view_box("Box","Key")["success"]:',
        'if False and self.view_box("Box","Key")["success"]:')
    assert n != t, "view_box edit did not match"
    p.write_text(n, encoding="utf-8")

def make_codebase(tag, edit=None):
    root = Path(f".hl_codebase/diag-{tag}")
    if root.exists(): shutil.rmtree(root)
    store = root / "store"; ws = root / "workspace"
    store.mkdir(parents=True); ws.mkdir(parents=True)
    for p in V1.rglob("*"):
        if p.is_file():
            rel = p.relative_to(V1); dst = ws/rel
            dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(p.read_bytes())
    if edit: edit(ws)
    cb = HLCodebase(root=ws, store=store)
    return cb, cb.snapshot(parent_version_id=None, edit_type="initial")

def main():
    nu = ReferenceStateSet.load(NU)
    print(f"[diag] {len(nu)} reference samples")
    cb1, v1 = make_codebase("v1")
    cb2, v2 = make_codebase("v2", edit_threshold)
    # sanity: confirm the edit landed
    t1 = (cb1.root/"agent.py").read_text(encoding="utf-8")
    t2 = (cb2.root/"agent.py").read_text(encoding="utf-8")
    has_edit = "<= 0" in t2 and "<= 130" in t1
    print(f"[diag] edit landed in v2: {has_edit}")
    stage = Path(".hl_codebase/diag-stage")
    def probe(ver, cb):
        cmd, cwd = candidate_command(ver, store=cb.store, dest=stage/ver.version_id)
        pr = ReferenceProbe(cmd=cmd, cwd=cwd, timeout=15.0)
        try: return pr.probe_set(list(nu.samples))
        finally: pr.close()
    e1 = probe(v1, cb1); e2 = probe(v2, cb2)
    ndiff = 0; n_none = 0; examples = []
    for i,(a,b) in enumerate(zip(e1,e2)):
        pa = a.primitive if a else None; pb = b.primitive if b else None
        if a is None or b is None: n_none += 1
        if pa != pb:
            ndiff += 1
            if len(examples) < 10:
                las = enumerate_legal_actions(nu.samples[i].legal_actions,
                    status=nu.samples[i].status, inventory=nu.samples[i].inventory)
                examples.append((i, pa, pb, a.out_of_support if a else None,
                                 b.out_of_support if b else None, len(las)))
    print(f"[diag] n_samples={len(e1)} n_diff_emissions={ndiff} n_none={n_none}")
    for ex in examples:
        print(f"  sample#{ex[0]}: v1={ex[1]} v2={ex[2]} oos_v1={ex[3]} oos_v2={ex[4]} |A(s)|={ex[5]}")
    # also report a few same-emission samples + their primitives
    same = [(i, e1[i].primitive if e1[i] else None) for i in range(min(5,len(e1)))]
    print("[diag] first 5 v1 emissions:", same)

if __name__ == "__main__":
    main()
