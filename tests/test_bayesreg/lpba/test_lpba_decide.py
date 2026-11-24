"""LP-C2 item 2 (contract C26.5-8, C26.10 T12; prereg §6, §12 item 13): bayesreg.eval.lpba_decide on synthetic records only.

T12: one synthetic case per rule of prereg §12 item 13 (N_PAIRS 17, MISSING_MIN_PAIRS 16, seed 20261119, PREREG_SHA256, families L1 = lpba54 and L2 = lpba56, no seg4 = G rule
and no F4, the suspicious threshold 2 with S5, the paired t p values, p^c = max, Holm ties in the main family's order, both p's 1 for a k not judged, S1's t on the complete
pairs (n_c < 2: p = 1), the guard max(U^B, U^t)), the two cases "bootstrap rejects, t does not" and "t rejects, bootstrap does not", and the comparison: with the t gate off
and the threshold 3 judge_family and sdlogj_guard_row equal holdout_decide's item for item."""
import json
import math

import numpy as np
import pytest
from scipy import stats

from bayesreg.eval import holdout_decide as HD
from bayesreg.eval import lpba_decide as D

MAIN = list(D.MAIN_ORDER)
TABLE = dict(main_family=MAIN, arms=[dict(name="method", family="method")]+[dict(name=k, family="main") for k in MAIN])
ROWS = [(i, f"S{2*i+3:02d}", f"S{2*i+4:02d}", f"LPBA40_S{2*i+3:02d}__LPBA40_S{2*i+4:02d}") for i in range(17)]


def official(value):
    return dict(dice=value, hd95=1., sdlogj=.05, dice_labels=[value, value], initial_dice_labels=[.5, .6])


def make_records(method, arms, *, sdlogj=None, statuses=None, outcomes=None, suspicious=None, drop=()):
    """{pair: record}: lpba54 = lpba56 = the given values; sdlogj per arm (default .05 everywhere); every cell an output unless outcomes[arm][i] is False."""
    sdlogj = sdlogj or {}
    records = {}
    for i, row in enumerate(ROWS):
        if i in drop:
            continue
        rec = dict(pair=row[3], index=i, complete=True, arms={}, _conflicts=[])
        rec["identity"] = dict(metrics=dict(lpba54=.4, lpba56=.41, sdlogj_official=0., dice_official=.4, hd95=9., sdlogj_brain=0., pct_J_lt_half_brain=0.,
                                            frac_abs_logj_gt_log2_brain=0.), official=official(.4))
        for name, values in [("method", method)]+list(arms.items()):
            out = (outcomes or {}).get(name, [True]*17)[i]
            metrics = dict(lpba54=float(values[i]), lpba56=float(values[i])+.01, sdlogj_official=float((sdlogj.get(name) or [.05]*17)[i]), dice_official=float(values[i]),
                           hd95=3., sdlogj_brain=.1, pct_J_lt_half_brain=0., frac_abs_logj_gt_log2_brain=0.)
            rec["arms"][name] = dict(outcome="output" if out else "no_output", metrics=metrics if out else None, official=official(float(values[i])) if out else None,
                                     no_output_class=None if out else "method", terminal_status=(statuses or {}).get(name, ["hx_certified" if name == "method" else "ok"]*17)[i],
                                     export_suspicious=bool((suspicious or {}).get(name, [False]*17)[i]))
        records[row[3]] = rec
    return records


def base_arms(delta_by_arm, method_level=.8):
    method = np.full(17, method_level)
    return method, {k: method-np.asarray(delta_by_arm.get(k, np.full(17, -.05))) for k in MAIN}


def family(records, idx=None, endpoint="lpba54", **kwargs):
    rows = ROWS
    values, imputed, suspicious, stopped, status, missing = HD.matrices(records, TABLE, rows, (endpoint, "sdlogj_official"))
    _, _, sus_all, _, _, _ = HD.matrices(records, TABLE, rows, ())
    idx = HD.draw_idx(seed=D.SEED, n=17) if idx is None else idx
    return D.judge_family(endpoint, values[endpoint], imputed, suspicious, stopped, TABLE, idx, sdlogj=values["sdlogj_official"], guard=D.SDLOGJ_GUARD,
                          missing=missing, suspicious_counts={a: int(v.sum()) for a, v in sus_all.items()}, **kwargs)


def test_t12_the_constants():
    from bayesreg.eval import lpba_eval as E
    assert (D.SEED, D.B, D.N_PAIRS, D.MISSING_MIN_PAIRS, D.SUSPICIOUS_STOP, D.SDLOGJ_GUARD) == (20261119, 10000, 17, 16, 2, .02)
    assert D.PREREG_SHA256 == E.PREREG_SHA256 == "06121278d8b229401afea7cb86c841f80ba02f23bc40f8c5f5594c7da8d95643"
    assert D.FAMILIES == (("L1", "lpba54"), ("L2", "lpba56"))
    assert not hasattr(D, "SEG4_G_RULE") and not hasattr(D, "decide_f4") and not hasattr(D, "PQ_BASELINES")
    assert (D.ROPE, D.EQ_BOUND, D.ALPHA) == (.005, .01, .025)
    assert HD.SEED == 20261107 and HD.N_PAIRS == 50 and HD.MISSING_MIN_PAIRS == 45      # holdout_decide's constants untouched
    idx = HD.draw_idx(seed=D.SEED, n=17, b=3)
    rng = np.random.default_rng(20261119)
    assert np.array_equal(idx, np.stack([rng.integers(0, 17, 17) for _ in range(3)]))


def test_t12_t_values_follow_the_preregistration():
    rng = np.random.default_rng(1)
    delta = rng.normal(.01, .01, 17)
    p = D.tvalues(delta)
    se = delta.std(ddof=1)/math.sqrt(17)
    F = lambda x: stats.t.cdf(x, 16)
    assert p["df"] == 16
    assert p["sup"] == pytest.approx(1-F((delta.mean()-.005)/se), abs=0, rel=1e-15)
    assert p["inf"] == pytest.approx(F((delta.mean()+.005)/se), rel=1e-15)
    assert p["eq_lo"] == pytest.approx(1-F((delta.mean()+.01)/se), rel=1e-15) and p["eq_hi"] == pytest.approx(F((delta.mean()-.01)/se), rel=1e-15)
    assert p["eq"] == max(p["eq_lo"], p["eq_hi"])
    same = D.tvalues(np.full(17, .02))                              # SE = 0: 0 on the alternative's side strictly, else 1
    assert (same["sup"], same["inf"], same["eq_lo"], same["eq_hi"]) == (0., 1., 0., 1.)
    edge = D.tvalues(np.full(17, .005))                             # numerator exactly 0: not strictly on the alternative's side
    assert edge["sup"] == 1. and edge["inf"] == 1.
    assert D.tvalues(np.full(17, -.005))["inf"] == 1. and D.tvalues(np.full(17, -.006))["inf"] == 0.
    assert D.tvalues(np.array([.3]))["sup"] == 1. and D.tvalues(np.array([]))["inf"] == 1.
    assert D.combine(dict(sup=.01, eq=.5, eq_lo=.5, eq_hi=.1, inf=.9), dict(sup=.02, eq=.4, eq_lo=.4, eq_hi=.2, inf=.8)) == dict(sup=.02, eq=.5, eq_lo=.5, eq_hi=.2, inf=.9)


def test_t12_against_holdout_decide_with_the_t_gate_off_and_threshold_three():
    """C26.10 T12 (revision 11a, review minor 1): t gate off, threshold 3 -> judge_family and sdlogj_guard_row equal holdout_decide's item for item."""
    rng = np.random.default_rng(7)
    method = rng.uniform(.7, .8, 17)
    arms = {k: method-rng.normal(m, .01, 17) for k, m in zip(MAIN, (.02, .0, -.02, .006, .012))}
    outcomes = {"ants_s": [i != 3 for i in range(17)], "method": [i != 5 for i in range(17)]}
    suspicious = {"fireants_syn": [i in (1, 2) for i in range(17)], "fireants_greedy": [i == 4 for i in range(17)]}
    sd = {k: list(rng.uniform(.03, .07, 17)) for k in ["method"]+MAIN}
    records = make_records(method, arms, sdlogj=sd, outcomes=outcomes, suspicious=suspicious)
    values, imputed, sus, stopped, status, missing = HD.matrices(records, TABLE, ROWS, ("lpba54", "sdlogj_official"))
    idx = HD.draw_idx(seed=D.SEED, n=17)
    counts = {a: int(v.sum()) for a, v in sus.items()}
    common = dict(sdlogj=values["sdlogj_official"], guard=.02, missing=missing, suspicious_counts=counts)
    ours = D.judge_family("lpba54", values["lpba54"], imputed, sus, stopped, TABLE, idx, t_gate=False, suspicious_stop=3, **common)
    theirs = HD.judge_family("lpba54", values["lpba54"], imputed, sus, stopped, TABLE, idx, **common)
    assert json.dumps(ours, sort_keys=True, default=str) == json.dumps(theirs, sort_keys=True, default=str)
    for k in MAIN:
        mine = D.sdlogj_guard_row(values["sdlogj_official"], imputed, missing, "method", k, np.arange(17), idx, .02, t_gate=False)
        assert json.dumps(mine, sort_keys=True) == json.dumps(HD.sdlogj_guard_row(values["sdlogj_official"], imputed, missing, "method", k, np.arange(17), idx, .02), sort_keys=True)
    gated = D.judge_family("lpba54", values["lpba54"], imputed, sus, stopped, TABLE, idx, **common)
    assert gated["arms"]["fireants_syn"]["not_judged"] is True and gated["arms"]["fireants_syn"]["verdict"] == D.NOT_JUDGED      # 2 suspicious pairs: stopped at 2
    assert theirs["arms"]["fireants_syn"]["not_judged"] is False                                                                 # holdout's 3: judged


def test_t12_bootstrap_rejects_and_t_does_not():
    """16 differences just above the ROPE and one large outlier: every bootstrap mean is above +.005 (p^B = 1/(B+1)), the t test's SE is large (p^t ~ .16): p^c = max -> not 优于."""
    delta = np.array([.0051]*16+[.2])
    method, arms = base_arms({"configc": delta})
    row = family(make_records(method, arms))["arms"]["configc"]
    assert row["p_boot"]["sup"] == pytest.approx(1/10001) and row["p_t"]["sup"] > .1
    assert row["p"]["sup"] == row["p_t"]["sup"] and not row["holm"]["sup"]["rejected"]
    assert row["verdict"] != D.SUP


def test_t12_t_rejects_and_bootstrap_does_not():
    """A clear t (differences .03 ± .001) and draws that all pick the one position below the ROPE: p^B = 1, p^t ~ 0: p^c = max -> not 优于 (the max takes the bootstrap)."""
    delta = np.full(17, .03)+np.linspace(-.001, .001, 17)
    delta[9] = .004
    method, arms = base_arms({"configc": delta})
    idx = np.full((10000, 17), 9, dtype=np.int64)
    row = family(make_records(method, arms), idx=idx)["arms"]["configc"]
    assert row["p_t"]["sup"] < 1e-6 and row["p_boot"]["sup"] == 1.
    assert row["p"]["sup"] == 1. and row["verdict"] != D.SUP


def test_t12_holm_on_the_combined_p_and_ties_in_the_main_family_order():
    clear = np.full(17, .05)+np.linspace(-.002, .002, 17)
    method, arms = base_arms({k: clear for k in MAIN})
    result = family(make_records(method, arms))
    assert [result["arms"][k]["verdict"] for k in MAIN] == [D.SUP]*5
    assert result["holm_order"]["sup"] == MAIN                          # equal p^c everywhere: the main family's order
    steps = [result["arms"][k]["holm"]["sup"]["step"] for k in MAIN]
    assert steps == [1, 2, 3, 4, 5]
    assert all(result["arms"][k]["holm"]["sup"]["raw"] == result["arms"][k]["p"]["sup"] == max(result["arms"][k]["p_boot"]["sup"], result["arms"][k]["p_t"]["sup"]) for k in MAIN)


def test_t12_verdict_order_inferior_and_equivalent():
    worse = np.full(17, -.05)+np.linspace(-.002, .002, 17)
    near = np.linspace(-.002, .002, 17)
    method, arms = base_arms({"configc": worse, "ants_s": near})
    result = family(make_records(method, arms))
    assert result["arms"]["configc"]["verdict"] == D.INF
    assert result["arms"]["ants_s"]["verdict"] == D.EQ


def test_t12_two_suspicious_pairs_stop_the_arm_and_one_is_s5():
    clear = np.full(17, .05)+np.linspace(-.002, .002, 17)
    method, arms = base_arms({k: clear for k in MAIN})
    suspicious = {"ants_s": [i in (0, 7) for i in range(17)], "fireants_syn": [i == 3 for i in range(17)]}
    result = family(make_records(method, arms, suspicious=suspicious))
    stopped = result["arms"]["ants_s"]
    assert stopped["not_judged"] and stopped["verdict"] == D.NOT_JUDGED
    assert all(stopped[k][c] == 1. for k in ("p", "p_boot", "p_t") for c in ("sup", "eq", "inf"))          # both p's of a k not judged are 1
    assert result["m"] == 5 and "ants_s" in result["holm_order"]["sup"]                                      # it stays in the family, m unchanged
    one = result["arms"]["fireants_syn"]
    assert not one["not_judged"] and one["s5"] is not None and one["s5"]["n"] == 16


def test_t12_s1_t_on_the_complete_pairs_and_n_c_below_two():
    clear = np.full(17, .05)+np.linspace(-.002, .002, 17)
    method, arms = base_arms({k: clear for k in MAIN})
    outcomes = {"configc": [i < 1 for i in range(17)], "ants_s": [i != 2 for i in range(17)]}         # configc: one complete pair only; ants_s: 16
    result = family(make_records(method, arms, outcomes=outcomes))
    few = result["arms"]["configc"]
    assert few["s1"]["n"] == 1 and few["s1"]["p"]["sup"] == 1. and few["s1"]["p_t"]["sup"] == 1. and few["s1"]["p_boot"]["sup"] == 1.
    assert few["verdict"] in (D.SUP_ITT, D.NONE, D.SUP_GUARD)
    if few["rejected_classes"][:1] == ["sup"]:
        assert few["itt_guard"]["holds"] is False
    many = result["arms"]["ants_s"]
    assert many["s1"]["p_t"]["df"] == 15 and many["s1"]["n"] == 16


def test_t12_the_guard_takes_the_larger_upper_end():
    clear = np.full(17, .05)+np.linspace(-.002, .002, 17)
    method, arms = base_arms({k: clear for k in MAIN})
    sd_method = np.full(17, .05)
    sd_method[0] = .32                                                 # one large SDlogJ difference: the bootstrap upper end below .02 for most draws? the t upper end high
    sd = {"method": list(sd_method)}
    result = family(make_records(method, arms, sdlogj=sd))
    guard = result["arms"]["configc"]["sdlogj_guard"]
    assert guard["upper"] == max(guard["upper_boot"], guard["upper_t"])
    t_up = (sd_method-.05).mean()+stats.t.ppf(.975, 16)*(sd_method-.05).std(ddof=1)/math.sqrt(17)
    assert guard["upper_t"] == pytest.approx(t_up, rel=1e-12)
    assert guard["holds"] is (guard["upper"] < .02)
    assert result["arms"]["configc"]["verdict"] == (D.SUP if guard["holds"] else D.SUP_GUARD)
    calm = family(make_records(method, arms))["arms"]["configc"]["sdlogj_guard"]
    assert calm["holds"] is True and calm["upper_t"] == 0.


def test_t12_the_guard_case_bootstrap_below_and_t_above():
    """U^B < .02 while U^t >= .02: the guard does not hold (max)."""
    clear = np.full(17, .05)+np.linspace(-.002, .002, 17)
    method, arms = base_arms({k: clear for k in MAIN})
    diff = np.zeros(17)
    diff[0] = .17                                                       # mean .01, the bootstrap draws without position 0 often: U^B well below; t's SE large
    idx = np.zeros((10000, 17), dtype=np.int64)+1                       # every draw on position 1 (difference 0): U^B = 0
    result = family(make_records(method, arms, sdlogj={"method": list(.05+diff)}), idx=idx)
    guard = result["arms"]["configc"]["sdlogj_guard"]
    assert guard["upper_boot"] == 0. and guard["upper_t"] >= .02 and guard["holds"] is False


def test_t12_missing_pairs_leave_every_arm_and_n_prime_below_16_is_described():
    clear = np.full(17, .05)+np.linspace(-.002, .002, 17)
    method, arms = base_arms({k: clear for k in MAIN})
    one = D.decide(make_records(method, arms, drop=(4,)), TABLE, ROWS)
    assert one["n_used"] == 16 and one["reading"] == "decided" and one["missing_pairs"] == [ROWS[4][3]]
    assert one["idx_sha256"] == HD.idx_sha256(HD.draw_idx(seed=D.SEED, n=16))
    assert one["L1"]["arms"]["configc"]["p_t"]["df"] == 15
    two = D.decide(make_records(method, arms, drop=(4, 9)), TABLE, ROWS)
    assert two["n_used"] == 15 and two["reading"] == D.DESCRIBED_FEW and two["L1"]["reading"] == D.DESCRIBED_FEW
    assert all(row["verdict"] is None for row in two["L1"]["arms"].values())


def test_t12_decide_two_families_and_no_f4():
    rng = np.random.default_rng(3)
    method = rng.uniform(.6, .7, 17)
    arms = {k: method-rng.normal(.01, .005, 17) for k in MAIN}
    out = D.decide(make_records(method, arms), TABLE, ROWS)
    assert out["L1"]["endpoint"] == "lpba54" and out["L2"]["endpoint"] == "lpba56"
    assert "F4" not in out and "F2" not in out and "ablation" not in out
    assert out["seed"] == 20261119 and out["idx_sha256"] == HD.idx_sha256(HD.draw_idx(seed=20261119, n=17))
    assert set(out["L3"]["endpoints"]) == set(HD.DESCRIBED)
    assert out["L3"]["identity"]["endpoints"]["sdlogj_official"]["mean"] == 0.
    assert all("t_interval" in row for row in out["L3"]["endpoints"]["hd95"].values())
    with pytest.raises(D.DecisionRefused):
        D.decide(make_records(method, arms), dict(TABLE, main_family=list(reversed(MAIN))), ROWS)


def test_minor1_all_equal_differences_are_judged_on_the_values():
    """Review minor 1: 17 equal differences whose float std is not exactly 0 still take the SE = 0 rule."""
    d = np.full(17, .1)
    p = D.tvalues(d)
    assert p['all_equal'] is True and (p['sup'], p['inf'], p['eq_lo'], p['eq_hi']) == (0., 1., 0., 1.)
    near = d.copy()
    near[3] = np.nextafter(.1, 1.)
    assert D.tvalues(near)['all_equal'] is False
