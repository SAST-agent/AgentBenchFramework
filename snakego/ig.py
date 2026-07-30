"""Information gain (IG) for strategy iteration.

Two complementary quantities, computed on the SAME action distribution defined
in decision_space (fixed support = {1..6}, Laplace-smoothed):

  * IG_action(version_t vs version_{t-1}) -- how much the BEHAVIOUR distribution
    changed between two strategy versions. Straight KL(p||q) over the smoothed
    support so it is always finite. Logged per version transition.

  * KL_to_human(version_t vs human opponent) -- KL between my action
    distribution and the human's (imitation distance). Recorded ONLY when it can
    be computed from a complete, clean replay. If the replay is incomplete
    (desync/crash, or fewer than MIN_MOVES recorded) we record
    reason="incomplete_replay" and do NOT substitute any other metric.
"""
import math
from snakego.decision_space import SUPPORT, action_distribution

MIN_MOVES_FOR_KL = 30
EPS = 1e-12


def kl_divergence(p, q):
    """KL(p || q) over the fixed SUPPORT. p, q are {op: prob}."""
    return sum(p[a] * math.log((p[a] + EPS) / (q[a] + EPS)) for a in SUPPORT)


def js_divergence(p, q):
    """Symmetric Jensen-Shannon divergence (bits). Bounded + symmetric, so
    cross-version comparison is safer. Headline IG metric is KL; JS is sanity."""
    m = {a: 0.5 * (p[a] + q[a]) for a in SUPPORT}
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def behaviour_dist_from_replay(replay, player):
    counts = {}
    for r, p, sid, op in replay["ops"]:
        if p == player:
            counts[op] = counts.get(op, 0) + 1
    return action_distribution(counts), sum(counts.values())


def ig_version_transition(prev_counts, cur_counts):
    """IG_action = KL(prev || cur) -- how much behaviour shifted this iteration."""
    p = action_distribution(prev_counts)
    q = action_distribution(cur_counts)
    return {
        "ig_kl": kl_divergence(p, q),
        "js_bits": js_divergence(p, q),
        "prev_dist": p,
        "cur_dist": q,
        "status": "ok",
    }


def kl_to_human(my_replay, human_player=None):
    """KL(my_behaviour || human_behaviour) from a single me-vs-human replay.

    status is 'ok' or 'incomplete_replay'. When incomplete, kl=None and we do
    NOT substitute any other metric -- missing stays missing.
    """
    err = my_replay.get("error")
    ops = my_replay.get("ops", [])
    if err and "timeout" not in str(err).lower() and len(ops) < 50:
        return {"status": "incomplete_replay", "reason": str(err)[:80],
                "kl": None, "js": None}
    me_player = 0 if my_replay.get("my_side", 0) == 0 else 1
    my_dist, my_n = behaviour_dist_from_replay(my_replay, me_player)
    hu_dist, hu_n = behaviour_dist_from_replay(my_replay, 1 - me_player)
    if my_n < MIN_MOVES_FOR_KL or hu_n < MIN_MOVES_FOR_KL:
        return {"status": "incomplete_replay",
                "reason": "too_few_moves(me=%d,human=%d)" % (my_n, hu_n),
                "kl": None, "js": None, "my_dist": my_dist, "hu_dist": hu_dist}
    return {
        "status": "ok",
        "kl": kl_divergence(my_dist, hu_dist),
        "js": js_divergence(my_dist, hu_dist),
        "my_dist": my_dist,
        "hu_dist": hu_dist,
        "my_moves": my_n,
        "hu_moves": hu_n,
    }


def aggregate_counts(replay_list, player_resolver):
    """Pool action counts across many replays for a single actor."""
    counts = {}
    for rep in replay_list:
        p = player_resolver(rep)
        for r, pp, sid, op in rep["ops"]:
            if pp == p:
                counts[op] = counts.get(op, 0) + 1
    return counts


# --------------------------------------------------------------------------
# Per-state epsilon-regularized KL (canonical IG, research doc 4.1 / 15)
# --------------------------------------------------------------------------
# The HL main line uses THIS metric, not the global Laplace KL above. The
# reason: deterministic HL strategies (if-else that always picks one action)
# produce p with 0 on every non-chosen action. Plain KL(p||q) is then 0*ln0
# which is 0 for agreement and +inf for disagreement -- not comparable. The
# epsilon-regularised version mixes each deterministic choice with uniform
# over the legal support at rate epsilon, yielding finite, meaningful KL.
#
# This is now the single source of truth for IG. The hl_sandbox scripts should
# import from here instead of inlining their own copy.

EPSILON = 0.05


def eps_regularized_kl(old_op, new_op, legal_mask, epsilon=EPSILON):
    """Per-state KL between two deterministic policies on the SAME state.

    legal_mask: {op: bool} from decision_space.compute_mask.
    old_op, new_op: the action each policy chose on this state.
    Returns KL(q || p) where p = eps-mix of old, q = eps-mix of new.
    Always finite even when one policy is deterministic.
    """
    legal = [a for a in SUPPORT if legal_mask.get(a, False)]
    if not legal:
        return 0.0
    n = len(legal)
    kl = 0.0
    for a in legal:
        p = (1 - epsilon) * (1.0 if a == old_op else 0.0) + epsilon / n
        q = (1 - epsilon) * (1.0 if a == new_op else 0.0) + epsilon / n
        kl += q * math.log((q + EPS) / (p + EPS))
    return kl


def per_state_ig(states, old_decide, new_decide, epsilon=EPSILON):
    """Mean per-state eps-reg KL over shared decision states (research doc 15).

    Each state is a deepcopy snapshot; both policies decide on their own copy
    so neither mutates the shared snapshot. Returns the canonical IG dict that
    curves.json / summary.json store as ig_kl.

    Returns:
        dict with ig_kl (mean nats/decision), trajectory_kl (sum),
        n_states, disagree_rate, status, epsilon.
    """
    import copy as _copy
    from snakego.decision_space import compute_mask as _compute_mask
    traces = []
    disagree = 0
    for snap in states:
        e_old = _copy.deepcopy(snap)
        e_new = _copy.deepcopy(snap)
        try:
            old_op = int(old_decide(e_old, 0))
        except Exception:
            old_op = 1
        try:
            new_op = int(new_decide(e_new, 0))
        except Exception:
            new_op = 1
        if not (1 <= old_op <= 6):
            old_op = 1
        if not (1 <= new_op <= 6):
            new_op = 1
        mask = _compute_mask(snap)
        if old_op != new_op:
            disagree += 1
        traces.append(eps_regularized_kl(old_op, new_op, mask.legal, epsilon))
    if not traces:
        return {"ig_kl": None, "status": "no_states", "disagree_rate": None,
                "epsilon": epsilon}
    return {
        "ig_kl": sum(traces) / len(traces),
        "trajectory_kl": sum(traces),
        "n_states": len(traces),
        "disagree_rate": disagree / len(traces),
        "status": "ok",
        "epsilon": epsilon,
    }