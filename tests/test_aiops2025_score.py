"""AIOps2025 scorer (scripts/aiops2025_score.py) on hand-made ground truth and answers: every rule of Cai et al. eq. (1)-(4) as read
in the scorer's docstring. No real data.

    python tests/test_aiops2025_score.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts import aiops2025_score as S   # noqa: E402

T = []


def test(fn):
    T.append(fn)
    return fn


def gt(uuid, cat, typ, level, inst, desc, obs=(), metrics=(), **extra):
    return {"uuid": uuid, "fault_category": cat, "fault_type": typ, "instance_type": level, "service": str(inst).rsplit("-", 1)[0],
            "instance": inst, "key_observations": list(obs), "key_metrics": list(metrics), "fault_description": list(desc), **extra}


def ans(uuid, comp, reason, obs=()):
    return {"uuid": uuid, "component": comp, "reason": reason,
            "reasoning_trace": [{"step": i + 1, "action": "LoadMetrics(x)", "observation": o} for i, o in enumerate(obs)]}


@test
def localisation_is_exact_and_pod_faults_need_the_pod():
    g = gt("a", "stress test", "cpu stress", "pod", "emailservice-0", ["cpu stress"])
    assert S.la_hit(ans("a", "emailservice-0", ""), g) == 1
    assert S.la_hit(ans("a", "emailservice", ""), g) == 0                 # the service is not the pod
    assert S.la_hit(ans("a", "EmailService-0", ""), g) == 0 and S.la_hit(ans("a", "", ""), g) == 0


@test
def network_attacks_accept_source_or_destination():
    g = gt("b", "network attack", "network delay", "service", "cartservice", ["network delay"], destination="redis-cart")
    assert S.la_hit(ans("b", "cartservice", ""), g) == 1 and S.la_hit(ans("b", "redis-cart", ""), g) == 1
    assert S.la_hit(ans("b", "frontend", ""), g) == 0
    g2 = gt("c", "stress test", "cpu stress", "service", "cartservice", ["cpu stress"], destination="redis-cart")
    assert S.la_hit(ans("c", "redis-cart", ""), g2) == 0                  # the relaxation is for network attacks only


@test
def type_accuracy_reads_only_the_first_twenty_words():
    g = gt("a", "stress test", "cpu stress", "service", "x", ["high CPU utilization", "cpu stress"])
    assert S.ta_hit(ans("a", "x", "cpu stress at x; x svc_cpu_rise"), g) == 1
    assert S.ta_hit(ans("a", "x", "CPU-stress at x"), g) == 1             # case, '-' and '_' are normalised
    assert S.ta_hit(ans("a", "x", "memory stress at x"), g) == 0
    assert S.ta_hit(ans("a", "x", " ".join(["w"] * 20) + " cpu stress"), g) == 0     # beyond word 20


@test
def explainability_counts_evidence_points_in_the_first_twenty_characters():
    g = gt("a", "stress test", "cpu stress", "service", "x", ["cpu stress"],
           obs=[{"type": "metric", "keyword": ["rrt", "rrt_max"]}, {"type": "metric", "keyword": ["pod_cpu_usage"]},
                {"type": "log", "keyword": ["OutOfMemory"]}], metrics=["pod_cpu_usage", "pod_processes"])
    pts = S.evidence_points(g)
    assert len(pts) == 4                                                  # 3 observations + pod_processes (pod_cpu_usage is listed)
    r = S.score_case(ans("a", "x", "cpu stress", ["pod_cpu_usage: baseline median=0.2", "rrt_max: baseline median=3",
                                                  "seen in the log: java.lang.OutOfMemoryError"]), g)
    assert (r["exp_hit"], r["exp_total"]) == (2, 4), r                    # cpu + rrt/rrt_max; the log keyword is past character 20
    r2 = S.score_case(ans("a", "x", "cpu stress", ["rrt_maximum: 3"]), g)
    assert r2["exp_hit"] == 0                                             # metric names match as whole tokens
    r3 = S.score_case(ans("a", "x", "cpu stress", ["outofmemory in pod x"]), g)
    assert r3["exp_hit"] == 1                                             # log keywords: case-insensitive substring


@test
def efficiency_uses_the_trace_length_of_the_localised_cases_and_final_follows_eq4():
    gts = [gt("a", "stress test", "cpu stress", "service", "x", ["cpu stress"], metrics=["m1"]),
           gt("b", "stress test", "cpu stress", "service", "y", ["cpu stress"], metrics=["m1", "m2"])]
    preds = [ans("a", "x", "cpu stress at x", ["m1: up"] * 10),           # LA 1, TA 1, Exp 1, trace 10
             ans("b", "z", "memory stress at z", ["m1: up"])]             # LA 0, TA 0, Exp 0.5, trace 1 (not in APL)
    d, s = S.score(preds, gts)
    eff = math.exp(-(10 - 5) / 5)
    assert abs(s["LA"] - 0.5) < 1e-9 and abs(s["TA"] - 0.5) < 1e-9 and abs(s["Exp"] - 0.75) < 1e-9 and abs(s["Eff"] - eff) < 1e-9
    assert abs(s["Final"] - 100 * (0.4 * 0.5 + 0.4 * 0.5 + 0.1 * 0.75 + 0.1 * eff)) < 1e-9
    assert abs(d["final_case"].mean() - s["Final"]) < 1e-9                # per-case Final averages to eq. (4)
    short = [ans("a", "x", "cpu stress at x", ["m1: up"] * 3), preds[1]]
    assert S.score(short, gts)[1]["Eff"] == 1.0                           # traces of at most 5 steps are not penalised
    assert S.score([ans("a", "q", "", []), ans("b", "q", "", [])], gts)[1]["Eff"] == 0.0    # nothing localised


@test
def an_unfinished_run_is_not_scored():
    gts = [gt("a", "stress test", "cpu stress", "service", "x", ["cpu stress"]),
           gt("b", "stress test", "cpu stress", "service", "y", ["cpu stress"])]
    try:
        S.score([ans("a", "x", "cpu stress")], gts)
        raise AssertionError("scored an unfinished run")
    except SystemExit:
        pass
    d, s = S.score([ans("a", "x", "cpu stress")], gts, partial=True)
    assert s["partial"] and s["n"] == 1 and s["n_missing"] == 1


@test
def a_missing_case_can_count_as_an_empty_answer_and_the_last_nonempty_answer_wins():
    gts = [gt("a", "stress test", "cpu stress", "service", "x", ["cpu stress"], metrics=["m1"]),
           gt("b", "stress test", "cpu stress", "service", "y", ["cpu stress"], metrics=["m1"])]
    preds = [ans("a", "", ""), ans("a", "x", "cpu stress at x", ["m1: up"])]            # empty line first (an earlier errored try)
    d, s = S.score(preds, gts, missing_as_empty=True)
    assert s["n_missing"] == 1 and not s["partial"] and s["n"] == 2
    assert abs(s["LA"] - 0.5) < 1e-9 and abs(s["TA"] - 0.5) < 1e-9


if __name__ == "__main__":
    ok = 0
    for fn in T:
        try:
            fn()
            print("ok  ", fn.__name__)
            ok += 1
        except Exception as e:      # noqa: BLE001
            print("FAIL", fn.__name__, repr(e)[:300])
    print(f"{ok}/{len(T)} passed")
    sys.exit(0 if ok == len(T) else 1)
