"""High-level iteration loop with honest multi-opponent evaluation.

Each step() does the full HL cycle:
  1. play a rotating human (rank06/09/11/13/15 -- diverse, not one)
  2. save replay (deterministic; reconstructable via replay2)
  3. learn from replay -> update Experience (compression applied)
  4. evaluate against a FIXED pool (rank13 s1-s2, rank15 s1-s2)
  5. accept/reject weight changes based on aggregate territory_ratio
  6. after 3 rejects, random perturbation to escape local optima
  7. snapshot version (immutable; rollback via VersionStore)
  8. compute IG(version_t vs version_{t-1})
  9. log to score-iteration and IG-iteration series

Primary score is territory_ratio = my/(my+opp) averaged over the fixed
pool: 0.5 = equal territory, >0.5 = winning, <0.5 = losing.
"""
import copy
import json
import os
import random
import time

from snakego.host import run_match, socket_player, python_player
from snakego.strategy_core import Weights, make_decide
from snakego.experience import seed_experience, Lesson
from snakego.versions import VersionStore, Snapshot, dict_to_weights, weights_to_dict
from snakego import ig as IG
from snakego import replay2


DEFAULT_EVAL_GAMES = [
    ("rank13", 1), ("rank13", 2), ("rank13", 3),
    ("rank15", 1), ("rank15", 2), ("rank15", 3),
]

DEFAULT_LEARN_POOL = ["rank06", "rank09", "rank11", "rank13", "rank15"]

ACCEPT_TOLERANCE = 0.05


class Loop:
    def __init__(self, data_dir="runs",
                 learn_pool=None,
                 eval_games=None):
        self.data_dir = data_dir
        self.learn_pool = learn_pool or DEFAULT_LEARN_POOL
        self.eval_games = eval_games or DEFAULT_EVAL_GAMES
        self.store = VersionStore()
        self.experience = seed_experience()
        self.iteration = 0
        self.log = []
        self.baseline_ratio = None
        self.best_ratio = None
        self._consecutive_rejects = 0
        self._writable = self._check_writable()

    def _check_writable(self):
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            p = os.path.join(self.data_dir, "_probe")
            open(p, "w").write("x")
            os.remove(p)
            return True
        except Exception:
            return False

    def _play_one(self, opp_name, seed):
        w = dict_to_weights(self.experience.weights_snapshot)
        me = python_player("me", make_decide(w))
        opp = socket_player(opp_name, "bin_humans")
        return run_match(me, opp, seed=seed, time_limit_per_move=10.0)

    def play_and_record(self, human, seed, label):
        w = dict_to_weights(self.experience.weights_snapshot)
        me = python_player("iter%d" % self.iteration, make_decide(w))
        opp = socket_player(human, "bin_humans")
        r = run_match(me, opp, seed=seed, record=True, time_limit_per_move=10.0)
        rep = r.to_dict()
        rep["config"] = {"length": 16, "width": 16, "max_round": 512, "seed": seed}
        rep["names"] = ["iter%d" % self.iteration, human]
        rep["my_version"] = "iter%d" % self.iteration
        rep["my_side"] = 0
        rep["error"] = r.error
        path = None
        if self._writable:
            path = os.path.join(self.data_dir, "%s.json" % label)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rep, f)
        return rep, r, path

    def multiseed_eval(self):
        games = []
        for opp_name, seed in self.eval_games:
            r = self._play_one(opp_name, seed)
            my_s, opp_s = r.scores[0], r.scores[1]
            ratio = my_s / max(1, my_s + opp_s)
            games.append({
                "opponent": opp_name, "seed": seed,
                "my": my_s, "opp": opp_s,
                "won": 1 if r.winner == 0 else 0,
                "ratio": round(ratio, 4),
                "rounds": r.rounds,
                "error": r.error is not None,
            })
        n = len(games) or 1
        total_ratio = sum(g["ratio"] for g in games) / n
        wins = sum(g["won"] for g in games)
        return {
            "territory_ratio": round(total_ratio, 4),
            "avg_score": round(sum(g["my"] for g in games) / n, 1),
            "avg_opp_score": round(sum(g["opp"] for g in games) / n, 1),
            "wins": wins,
            "games": len(games),
            "detail": games,
        }

    def learn(self, replay):
        me, opp = 0, 1
        my_counts, opp_counts = {}, {}
        for r, p, sid, op in replay["ops"]:
            tgt = my_counts if p == me else opp_counts
            tgt[op] = tgt.get(op, 0) + 1
        my_splits = my_counts.get(6, 0)
        opp_splits = opp_counts.get(6, 0)
        my_total = sum(my_counts.values())

        deaths = []
        opp_big_gains = 0
        try:
            for step, fr in replay2.iter_frames(replay, include_initial=False):
                evs = fr["events"]
                if "snake_died" in evs and fr["player"] == me:
                    deaths.append((fr["round"], fr["snake_len"]))
                for ev in evs:
                    if "score_delta" in ev and fr["player"] == opp:
                        parts = ev.replace("score_delta_", "").split("_")
                        if len(parts) == 2:
                            d1 = int(parts[1])
                            if d1 > 4:
                                opp_big_gains += 1
        except Exception:
            pass

        insight_bits = []
        wd = {}
        my_score = replay["scores"][0]
        opp_score = replay["scores"][1]

        if opp_splits > my_splits and my_total > 10:
            insight_bits.append("opp split %dx vs my %dx -> split more" % (
                opp_splits, my_splits))
            split_deficit = opp_splits - my_splits
            wd["split_value"] = min(3.0, 1.0 * split_deficit)

        if deaths:
            avg_death = sum(d[0] for d in deaths) / len(deaths)
            max_death_len = max(d[1] for d in deaths)
            if avg_death < 80:
                insight_bits.append("died early (avg r=%.0f, len=%d) -> boost survival gate" % (
                    avg_death, max_death_len))
                wd["trap_penalty"] = -3.0
            elif max_death_len > 16:
                insight_bits.append("overgrew to len=%d then died -> cap growth harder" % max_death_len)
                wd["max_growth_length"] = -2.0

        if my_score < opp_score * 0.6 and replay["rounds"] > 40:
            insight_bits.append("territory deficit (%d vs %d) -> seal more" % (
                my_score, opp_score))
            sev = max(1.0, opp_score / max(1, my_score) - 1.0)
            wd["seal_area"] = round(min(3.0, sev), 2)

        if opp_big_gains > 2:
            insight_bits.append("opp sealed %d big regions -> learn their sealing" % opp_big_gains)
            wd["seal_area"] = wd.get("seal_area", 0) + round(min(1.5, 0.3 * opp_big_gains), 2)

        if not insight_bits:
            insight_bits.append("no clear deficit vs %s; holding weights" % replay["names"][1])
        lesson = Lesson(
            iteration=self.iteration,
            source="iter%d vs %s" % (self.iteration, replay["names"][1]),
            evidence="my=%d hu=%d rounds=%d my_splits=%d opp_splits=%d deaths=%d opp_big_gains=%d" % (
                my_score, opp_score, replay["rounds"],
                my_splits, opp_splits, len(deaths), opp_big_gains),
            insight="; ".join(insight_bits),
            weight_delta=wd,
        )
        self.experience.add(lesson)
        return lesson

    def compute_ig(self, prev_rep, cur_rep):
        prev_counts = {}
        cur_counts = {}
        for r, p, sid, op in (prev_rep or {}).get("ops", []):
            if p == 0:
                prev_counts[op] = prev_counts.get(op, 0) + 1
        for r, p, sid, op in cur_rep.get("ops", []):
            if p == 0:
                cur_counts[op] = cur_counts.get(op, 0) + 1
        if not prev_counts:
            return {"ig_kl": None, "status": "no_baseline"}
        return IG.ig_version_transition(prev_counts, cur_counts)

    def _random_perturbation(self):
        axes = [
            "seal_area", "split_value", "trap_penalty", "growth_item_delta",
            "enemy_proximity", "center_seeking", "claim_free_cell",
            "space_per_body_len", "loop_potential", "territory_contest",
        ]
        axis = random.choice(axes)
        delta = random.choice([-0.5, -0.3, 0.3, 0.5])
        cur = self.experience.weights_snapshot.get(axis, 0.0)
        self.experience.weights_snapshot[axis] = round(cur + delta, 4)
        lesson = Lesson(
            iteration=self.iteration,
            source="random_perturbation",
            evidence="stuck after %d rejects" % self._consecutive_rejects,
            insight="exploration: %s += %.1f" % (axis, delta),
            weight_delta={axis: delta},
        )
        self.experience.add(lesson)
        return lesson

    def step(self):
        label = "iter%d" % self.iteration

        if self.baseline_ratio is None:
            baseline = self.multiseed_eval()
            self.baseline_ratio = baseline["territory_ratio"]
            self.best_ratio = baseline["territory_ratio"]
            print("  [baseline] territory_ratio=%.4f  avg=%d vs opp=%d" % (
                self.baseline_ratio, baseline["avg_score"], baseline["avg_opp_score"]))

        prev_exp = copy.deepcopy(self.experience)

        opp = self.learn_pool[self.iteration % len(self.learn_pool)]
        seed = 1 + self.iteration
        rep, r, path = self.play_and_record(opp, seed, label + "_learn_vs_%s" % opp)
        lesson = self.learn(rep)

        eval_result = self.multiseed_eval()
        new_ratio = eval_result["territory_ratio"]

        accepted = True
        if self.best_ratio is not None and new_ratio < self.best_ratio - ACCEPT_TOLERANCE:
            self.experience = prev_exp
            accepted = False
            self._consecutive_rejects += 1
            eval_result = self.multiseed_eval()
        else:
            self._consecutive_rejects = 0
            if new_ratio > (self.best_ratio or 0):
                self.best_ratio = new_ratio

        if self._consecutive_rejects >= 3:
            pl = self._random_perturbation()
            self._consecutive_rejects = 0
            eval_result = self.multiseed_eval()
            new_ratio = eval_result["territory_ratio"]
            if new_ratio > (self.best_ratio or 0):
                self.best_ratio = new_ratio
            lesson = pl

        ig = self.compute_ig(
            self.log[-1]["learn_rep"] if self.log else None, rep)

        snap = Snapshot(
            iteration=self.iteration, label=label,
            weights=weights_to_dict(dict_to_weights(
                self.experience.weights_snapshot)),
            experience_lessons=len(self.experience.active_lessons()),
            eval_summary={
                "territory_ratio": eval_result["territory_ratio"],
                "avg_score": eval_result["avg_score"],
                "avg_opp_score": eval_result["avg_opp_score"],
                "wins": eval_result["wins"],
                "games": eval_result["games"],
                "learn_vs": opp,
                "accepted": accepted,
                "baseline_ratio": self.baseline_ratio,
                "best_ratio": self.best_ratio,
            },
            ig={"ig_kl": ig.get("ig_kl"), "status": ig.get("status")},
            note=lesson.insight + (
                " [ACCEPTED]" if accepted else " [REVERTED]"),
        )
        self.store.save(snap)
        entry = {
            "iter": self.iteration,
            "learn_rep": rep,
            "learn_vs": opp,
            "eval": snap.eval_summary,
            "ig": ig,
            "lesson": lesson.insight,
            "accepted": accepted,
            "replay_path": path,
            "eval_detail": eval_result["detail"],
        }
        self.log.append(entry)
        self.iteration += 1
        return entry

    def persist_all(self):
        if not self._writable:
            return False
        with open(os.path.join(self.data_dir, "versions.json"), "w", encoding="utf-8") as f:
            json.dump(self.store.export(), f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.data_dir, "experience.json"), "w", encoding="utf-8") as f:
            json.dump(self.experience.to_dict(), f, indent=2, ensure_ascii=False)

        curves = {
            "score_iteration": {"iter": [], "score": [], "ratio": []},
            "ig_iteration": {"iter": [], "ig_kl": []},
            "points": [],
        }

        if self.baseline_ratio is not None:
            curves["score_iteration"]["iter"].append(-1)
            curves["score_iteration"]["score"].append(0)
            curves["score_iteration"]["ratio"].append(self.baseline_ratio)

        for e in self.log:
            it = e["iter"]
            ev = e["eval"]
            curves["score_iteration"]["iter"].append(it)
            curves["score_iteration"]["score"].append(ev.get("avg_score", 0))
            curves["score_iteration"]["ratio"].append(ev.get("territory_ratio", 0))
            curves["ig_iteration"]["iter"].append(it)
            kl = e["ig"].get("ig_kl")
            curves["ig_iteration"]["ig_kl"].append(kl)
            curves["points"].append({
                "iter": it,
                "territory_ratio": ev.get("territory_ratio", 0),
                "avg_score": ev.get("avg_score", 0),
                "avg_opp_score": ev.get("avg_opp_score", 0),
                "wins": ev.get("wins", 0),
                "games": ev.get("games", 0),
                "learn_vs": ev.get("learn_vs", "?"),
                "accepted": ev.get("accepted", True),
                "ig_kl": kl,
                "ig_status": e["ig"].get("status", "ok"),
                "lesson": e["lesson"],
                "eval_detail": e.get("eval_detail", []),
            })
        with open(os.path.join(self.data_dir, "curves.json"), "w", encoding="utf-8") as f:
            json.dump(curves, f, indent=2, ensure_ascii=False)
        return True

    def series(self):
        ratios, igs, iters = [], [], []
        for e in self.log:
            iters.append(e["iter"])
            ratios.append(e["eval"].get("territory_ratio", 0))
            igs.append(e["ig"].get("ig_kl"))
        return {"iters": iters, "ratios": ratios, "ig_kl": igs}
