"""bayesreg.eval.holdout_decide: one synthetic case per rule of preregistration 4.3-4.7 (resampling, p values with ±.005 and ±.01, Holm, the verdicts and
their order, the ITT guard, the SDlogJ guard, P19, the seg4 = G rules, the missing-pair rules, F3, F4 of contract C22.H7).  Synthetic records only; the map
to the preregistration is runs/20261107-ev/实现说明.md §4 and §13."""
import json
from pathlib import Path

import numpy as np
import pytest

from bayesreg.eval import holdout as H
from bayesreg.eval import holdout_decide as D
from bayesreg.eval.metrics import aggregate

from .. import _p0_pins as P0

MAIN = ("configc", "ants", "fga", "fsyn", "cadam")
ARMS = ("method",)+MAIN+("desc", "t90")
N = 50
ROWS = tuple((i, f"OASIS_OAS1_{900+2*i:04d}_MR1", f"OASIS_OAS1_{901+2*i:04d}_MR1", f"p{i:02d}") for i in range(N))
SMALL = D.draw_idx(b=1000)
GUARD = .02                       # the bound of revision 2 (adopted 11-10), the code's constant since P0 (test_the_decision_codes_pins_at_p0)


@pytest.fixture(autouse=True)
def pinned_guard(monkeypatch):
    monkeypatch.setattr(D, "SDLOGJ_GUARD", GUARD)


def table():
    return dict(arms=[dict(name="method", family="method")]+[dict(name=k, family="main") for k in MAIN]+[dict(name="desc", family="descriptive"),
                                                                                                         dict(name="t90", family="tier_90s")],
                main_family=list(MAIN))


def metrics_of(seg35, seg4, sdlogj=.07):
    return dict(seg35=float(seg35), seg4=float(seg4), dice_official=float(seg35)-.01, hd95=2.-float(seg35), sdlogj_official=float(sdlogj), sdlogj_brain=.1,
                pct_J_lt_half_brain=.1, frac_abs_logj_gt_log2_brain=.01)


def official_of(rng, initial):
    final = np.clip(initial+rng.uniform(.1, .3, 35), 0, 1)
    final[3] = np.nan
    return dict(dice=float(np.nanmean(final)), hd95=2., sdlogj=.07, dice_labels=final.tolist(), initial_dice_labels=initial.tolist())


def build(values, *, identity=None, no_output=None, suspicious=None, stopped=(), status=None, seg4_bad=(), drop=None, sdlogj=None, pq=None):
    """values: {arm: (seg35 array, seg4 array)}; identity: (seg35, seg4) arrays of the d = 0 rows (SDlogJ 0); no_output: {arm: pair indices}; drop: {pair
    index: arms to leave out of the record (None: the whole record)}; sdlogj: {arm: official SDlogJ per pair (default .07)}; pq: {pair index: PQ section}."""
    rng = np.random.default_rng(5)
    identity = (np.full(N, .55), np.full(N, .7)) if identity is None else identity
    no_output, suspicious, drop, sdlogj, pq = no_output or {}, suspicious or {}, drop or {}, sdlogj or {}, pq or {}
    records = {}
    for i, row in enumerate(ROWS):
        initial = rng.uniform(.3, .6, 35)
        record = dict(pair=row[3], index=i, complete=True, identity=dict(metrics=metrics_of(identity[0][i], identity[1][i], 0.), official=official_of(rng, initial-.1)),
                      seg4_equals_G=dict(fixed=dict(equal=i not in seg4_bad), moving=dict(equal=True)), arms={})
        for arm in ARMS:
            seg35, seg4 = values[arm]
            if i in no_output.get(arm, ()):
                entry = dict(outcome="no_output", no_output_class="arm_stopped" if arm in stopped else "method", terminal_status="method_failed")
            else:
                sd = sdlogj[arm][i] if arm in sdlogj else .07
                entry = dict(outcome="output", metrics=metrics_of(seg35[i], seg4[i], sd), official=official_of(rng, initial), terminal_status="ok")
            if arm == "method" and entry["outcome"] == "output":
                entry["terminal_status"] = (status or ["hx_certified"]*N)[i]
            entry["export_suspicious"] = i in suspicious.get(arm, ())
            record["arms"][arm] = entry
        if i in pq:
            record["pq"] = pq[i]
        if i in drop:
            if drop[i] is None:
                continue
            record["complete"] = False
            for arm in drop[i]:
                del record["arms"][arm]
        records[row[3]] = record
    return records


def noisy(mean, sd, seed):
    return mean+np.random.default_rng(seed).normal(0, sd, N)


def base_values(**deltas):
    """method seg35 ~ .88; each arm = method - delta (+ small noise)."""
    method = noisy(.88, .01, 1)
    out = dict(method=(method, method+.04))
    for k, arm in enumerate(MAIN+("desc", "t90")):
        delta = deltas.get(arm, .03)
        values = method-delta+np.random.default_rng(10+k).normal(0, .002, N)
        out[arm] = (values, values+.04)
    return out


def family(values, idx=None, **kw):
    tab = table()
    v, imputed, suspicious, stopped, status, missing = D.matrices(build(values, **kw), tab, ROWS, ("seg35", "seg4", "sdlogj_official"))
    return D.judge_family("seg35", v["seg35"], imputed, suspicious, stopped, tab, D.draw_idx() if idx is None else idx, missing=missing,
                          sdlogj=v["sdlogj_official"], guard=GUARD)


# ---- 4.3 resampling and p values --------------------------------------------------------------------------------------------------------------------
def test_idx_is_the_preregistered_sequence():
    idx = D.draw_idx()
    rng = np.random.default_rng(20261107)
    assert idx.shape == (10000, 50) and idx.dtype == np.int64
    for b in range(10000):
        assert np.array_equal(idx[b], rng.integers(0, 50, 50))
    assert D.SEED == 20261107 and D.B == 10000 and D.ROPE == .005 and D.EQ_BOUND == .01 and D.ALPHA == .025


def test_boot_means_equal_the_row_by_row_mean():
    delta = np.random.default_rng(3).normal(.01, .02, N)
    idx = D.draw_idx()
    means = D.boot_means(delta, idx)
    assert np.array_equal(means, np.array([np.mean(delta[idx[b]]) for b in range(idx.shape[0])]))


def test_pvalues_plus_one_and_boundary_counts_with_two_bounds():
    means = np.array([-.01, -.005, -.005, 0., .004, .005, .005, .006, .01, np.nan])
    p = D.pvalues(means)
    n = 9
    assert p["sup"] == (1+7)/(n+1)                          # <= +.005 counts the two equal to +.005
    assert p["inf"] == (1+8)/(n+1)                          # >= -.005 counts the two equal to -.005
    assert p["eq_lo"] == (1+1)/(n+1) and p["eq_hi"] == (1+1)/(n+1) and p["eq"] == max(p["eq_lo"], p["eq_hi"])      # ±.01 for 相当, the boundary counts
    assert p["void"] == 1 and p["draws"] == 9 and p["rope"] == .005 and p["eq_bound"] == .01


# ---- 4.4 Holm and the verdicts ----------------------------------------------------------------------------------------------------------------------
def test_holm_adjusted_p_and_stops_at_the_first_failure():
    p = dict(configc=.004, ants=.0065, fga=.0066, fsyn=.5, cadam=.6)
    steps = D.holm(p, list(MAIN))
    assert steps["configc"]["rejected"] and steps["configc"]["adjusted"] == 5*.004
    assert not steps["ants"]["rejected"] and steps["ants"]["adjusted"] == pytest.approx(4*.0065)
    assert 3*.0066 <= .025 and not steps["fga"]["rejected"] and steps["fga"]["adjusted"] == steps["ants"]["adjusted"]      # would pass alone; Holm stopped
    assert [steps[k]["threshold"] for k in ("configc", "ants", "fga")] == [.025/5, .025/4, .025/3]


def test_holm_ties_follow_the_family_order():
    steps = D.holm(dict(configc=.01, ants=.01, fga=.001, fsyn=.5, cadam=.5), list(MAIN))
    assert [steps[k]["step"] for k in ("fga", "configc", "ants")] == [1, 2, 3]


def test_the_four_verdicts():
    values = base_values(configc=.03, ants=0., fga=-.03, fsyn=.005, cadam=.03)
    values["fsyn"] = (values["method"][0]-.005+np.random.default_rng(7).normal(0, .03, N), values["fsyn"][1])     # noisy around the ROPE edge
    f = family(values)
    verdicts = {k: f["arms"][k]["verdict"] for k in MAIN}
    assert verdicts == dict(configc=D.SUP, ants=D.EQ, fga=D.INF, fsyn=D.NONE, cadam=D.SUP)
    assert f["arms"]["ants"]["rejected_classes"] == ["eq"] and f["arms"]["fsyn"]["rejected_classes"] == []


def test_sup_and_eq_both_rejected_reads_sup_and_inf_and_eq_reads_inf():
    """With ±.01 for 相当 a difference of about ±.007 rejects both: the order 优于 → 劣于 → 相当 decides, both are recorded, nothing is refused (review B1)."""
    f = family(base_values(configc=.007, ants=-.007, fga=.03, fsyn=.03, cadam=.03))
    assert f["arms"]["configc"]["rejected_classes"] == ["sup", "eq"] and f["arms"]["configc"]["verdict"] == D.SUP
    assert f["arms"]["ants"]["rejected_classes"] == ["inf", "eq"] and f["arms"]["ants"]["verdict"] == D.INF


def test_holm_per_class_with_m5():
    f = family(base_values())
    for cls in ("sup", "eq", "inf"):
        assert sorted(f["arms"][k]["holm"][cls]["step"] for k in MAIN) == [1, 2, 3, 4, 5]
    assert f["m"] == 5 and f["eq_bound"] == .01 and f["rope"] == .005


def test_a_non_finite_difference_in_the_main_family_refuses():
    values = base_values()
    values["ants"][0][4] = np.nan
    with pytest.raises(D.DecisionRefused, match="non-finite"):
        family(values)


# ---- 4.5 ITT, the guards, P19 ----------------------------------------------------------------------------------------------------------------------------
def test_itt_imputes_the_identity_row():
    values = base_values()
    no_output = dict(ants={0, 1})
    records = build(values, no_output=no_output)
    v, imputed, *_ = D.matrices(records, table(), ROWS, ("seg35",))
    assert imputed["ants"][:2].all() and not imputed["ants"][2:].any()
    assert v["seg35"]["ants"][0] == .55 and v["seg35"]["ants"][2] == pytest.approx(values["ants"][0][2])


def test_sup_only_by_itt_is_downgraded_and_the_next_arm_judged_as_usual():
    values = base_values(configc=.004, ants=.03)                     # configc: on the complete pairs inside the ROPE
    f = family(values, no_output=dict(configc=set(range(20))))       # 20 imputed pairs at .55: Delta large under ITT
    assert f["arms"]["configc"]["verdict"] == D.SUP_ITT and f["arms"]["configc"]["itt_guard"]["holds"] is False
    assert f["arms"]["configc"]["holm"]["sup"]["rejected"]           # the step-down itself is not changed
    assert f["arms"]["ants"]["verdict"] == D.SUP and "itt_guard" not in f["arms"]["ants"]
    order = f["holm_order"]["sup"]
    assert all(f["arms"][k]["holm"]["sup"]["rejected"] for k in order[:order.index("ants")+1])


def test_sup_with_imputed_pairs_kept_when_s1_holds():
    f = family(base_values(configc=.03), no_output=dict(configc={0, 1}))
    assert f["arms"]["configc"]["verdict"] == D.SUP and f["arms"]["configc"]["itt_guard"]["holds"] is True


def test_inf_with_imputed_method_pairs_needs_s1():
    values = base_values(fga=-.004)
    f = family(values, identity=(np.full(N, .3), np.full(N, .5)), no_output=dict(method=set(range(20))))
    assert f["arms"]["fga"]["verdict"] == D.INF_ITT and f["arms"]["fga"]["itt_guard"]["holds"] is False


def test_eq_is_not_touched_by_imputation():
    values = base_values(ants=0.)
    identity = (values["method"][0].copy(), values["method"][1].copy())
    f = family(values, identity=identity, no_output=dict(ants={0, 1, 2}))
    assert f["arms"]["ants"]["verdict"] == D.EQ and "itt_guard" not in f["arms"]["ants"]


def test_sdlogj_guard_holds_and_fails():
    """优于 also needs the official SDlogJ difference (method - k) below the bound at the 95% interval's upper end (BZ 11-07 item (3), review B2)."""
    rng = np.random.default_rng(11)
    sd = dict(method=.07+rng.normal(0, .003, N), configc=.07+rng.normal(0, .003, N), ants=.07-.03+rng.normal(0, .003, N))     # ants .03 smoother
    f = family(base_values(configc=.03, ants=.03), sdlogj=sd)
    assert f["arms"]["configc"]["verdict"] == D.SUP and f["arms"]["configc"]["sdlogj_guard"]["holds"] is True
    assert f["arms"]["configc"]["sdlogj_guard"]["upper"] < GUARD
    ants = f["arms"]["ants"]
    assert ants["verdict"] == D.SUP_GUARD and ants["verdict_before_guard"] == D.SUP and ants["sdlogj_guard"]["upper"] >= GUARD
    assert ants["holm"]["sup"]["rejected"]                                          # only the verdict changes, not the step-down
    assert all(f["arms"][k]["verdict"] == D.SUP for k in ("fga", "fsyn", "cadam"))  # the arms after it are judged as usual


def test_sdlogj_guard_uses_the_complete_pairs_only():
    """An arm with many failures imputed as identity (SDlogJ 0) does not look smoother: the guard leaves imputed pairs out (the lead's ruling (2))."""
    sd = dict(method=np.full(N, .07), ants=np.full(N, .07))
    f = family(base_values(ants=.03), sdlogj=sd, no_output=dict(ants=set(range(25))))
    guard = f["arms"]["ants"]["sdlogj_guard"]
    assert guard["n"] == 25 and guard["mean"] == pytest.approx(0.) and guard["holds"] is True


def test_sdlogj_guard_on_an_itt_only_sup():
    """Review2 m8: the guard also reads an ITT-only 优于: 无法下结论（护栏）, with the ITT verdict kept beside it."""
    rng = np.random.default_rng(12)
    sd = dict(method=.07+.03+rng.normal(0, .002, N), configc=.07+rng.normal(0, .002, N))
    f = family(base_values(configc=.004, ants=.03), sdlogj=sd, no_output=dict(configc=set(range(20))))
    row = f["arms"]["configc"]
    assert row["verdict"] == D.SUP_GUARD and row["verdict_before_guard"] == D.SUP_ITT and row["sdlogj_guard"]["holds"] is False


def test_sdlogj_guard_upper_end_equal_to_the_bound_does_not_hold(monkeypatch):
    """Review2 m8: strictly below the bound (an exactly representable difference .0625 on every complete pair: every draw's mean is .0625)."""
    sd = dict(method=np.full(N, .125), ants=np.full(N, .0625))
    v, imputed, suspicious, stopped, status, missing = D.matrices(build(base_values(), sdlogj=sd), table(), ROWS, ("seg35", "sdlogj_official"))
    at = D.sdlogj_guard_row(v["sdlogj_official"], imputed, missing, "method", "ants", np.arange(N), SMALL, .0625)
    assert at["upper"] == .0625 and at["holds"] is False
    assert D.sdlogj_guard_row(v["sdlogj_official"], imputed, missing, "method", "ants", np.arange(N), SMALL, np.nextafter(.0625, 1.))["holds"] is True


def test_sdlogj_guard_in_f2_and_without_complete_pairs():
    """Review2 m8: F2 (seg4) goes through the guard too; an arm without one complete pair (every cell imputed) cannot hold it."""
    sd = dict(method=np.full(N, .12), ants=np.full(N, .07))
    out = D.decide(build(base_values(), sdlogj=sd), table(), ROWS, idx=SMALL)
    assert out["F2"]["arms"]["ants"]["verdict"] == D.SUP_GUARD and out["F1"]["arms"]["ants"]["verdict"] == D.SUP_GUARD
    f = family(base_values(fga=.03), no_output=dict(fga=set(range(N))))
    row = f["arms"]["fga"]
    assert row["sdlogj_guard"]["n"] == 0 and row["sdlogj_guard"]["holds"] is False and row["verdict"] == D.SUP_GUARD


def test_decide_refuses_without_the_pinned_guard(monkeypatch):
    monkeypatch.setattr(D, "SDLOGJ_GUARD", None)
    with pytest.raises(D.DecisionRefused, match="SDLOGJ_GUARD"):
        D.decide(build(base_values()), table(), ROWS, idx=SMALL)


def test_the_decision_codes_pins_at_p0():
    """P0: the decision code's own constants as written in holdout_decide.py (read from its source: the autouse fixture above patches SDLOGJ_GUARD) -- SDLOGJ_GUARD .02
    (preregistration revision 2, BZ 11-10) and PREREG_SHA256 the committed preregistration's sha256, the same pin as the entry's (main refuses a mismatch)."""
    import ast
    source = ast.parse(Path(D.__file__).read_text(encoding="utf-8"))
    written = {node.targets[0].id: ast.literal_eval(node.value) for node in source.body
               if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("SDLOGJ_GUARD", "PREREG_SHA256")}
    assert written == dict(SDLOGJ_GUARD=.02, PREREG_SHA256=D.PREREG_SHA256) and written["SDLOGJ_GUARD"] == GUARD
    P0.assert_pinned(D.PREREG_SHA256, H.PREREG_REL)
    assert D.PREREG_SHA256 == H.PREREG_SHA256


def test_two_suspicious_pairs_stay_and_s5_leaves_them_out():
    f = family(base_values(), suspicious=dict(ants={3, 4}))
    row = f["arms"]["ants"]
    assert row["verdict"] == D.SUP and row["n_suspicious"] == 2 and row["s5"]["n"] == 48


def test_third_suspicious_pair_is_not_judged_p1_and_m_unchanged():
    f = family(base_values(), suspicious=dict(ants={3, 4, 5}))
    row = f["arms"]["ants"]
    assert row["verdict"] == D.NOT_JUDGED and row["p"]["sup"] == row["p"]["eq"] == row["p"]["inf"] == 1.
    assert f["m"] == 5 and row["holm"]["sup"]["step"] == 5
    assert f["arms"]["configc"]["holm"]["sup"]["threshold"] in (.025/5, .025/4, .025/3, .025/2)


def test_suspicious_pairs_are_counted_over_the_whole_list():
    """Rule (乙) drops pair 5 (another arm lacks its record); ants' third suspicious pair is pair 5: ants stays 不判 (review m4)."""
    out = D.decide(build(base_values(), suspicious=dict(ants={3, 4, 5}), drop={5: ["fga"]}), table(), ROWS, idx=SMALL)
    assert out["n_used"] == 49 and out["F1"]["arms"]["ants"]["verdict"] == D.NOT_JUDGED and out["F1"]["arms"]["ants"]["n_suspicious"] == 3


def test_an_arm_stopped_by_p19_is_not_judged():
    f = family(base_values(), no_output=dict(cadam=set(range(40, 50))), stopped={"cadam"})
    assert f["arms"]["cadam"]["verdict"] == D.NOT_JUDGED


# ---- seg4 = G(seg35): rules (a) and (b) ----------------------------------------------------------------------------------------------------------------
def test_seg4_rule_a_describes_f2_and_leaves_f1():
    out = D.decide(build(base_values(), seg4_bad={7}), table(), ROWS, seg4_rule="a", idx=SMALL)
    assert out["F2"]["reading"] == D.F2_DESCRIBED and all(r["verdict"] is None for r in out["F2"]["arms"].values())
    assert all("holm" not in r and "rejected_classes" not in r for r in out["F2"]["arms"].values()) and "holm_order" not in out["F2"]     # review m13
    assert out["F1"]["arms"]["configc"]["verdict"] == D.SUP and out["seg4_rule"] == "a" == D.SEG4_G_RULE


def test_seg4_rule_b_drops_the_pairs_and_redraws_on_n_prime():
    out = D.decide(build(base_values(), seg4_bad={7}), table(), ROWS, seg4_rule="b", idx=SMALL)
    assert out["F2"]["n_prime"] == 49 and out["F2"]["idx_sha256"] == D.idx_sha256(D.draw_idx(n=49, b=1000))
    assert out["F2"]["arms"]["configc"]["verdict"] == D.SUP
    few = D.decide(build(base_values(), seg4_bad=set(range(6))), table(), ROWS, seg4_rule="b", idx=SMALL)
    assert few["F2"]["reading"] == D.F2_DESCRIBED and few["F2"]["n_prime"] == 44


# ---- pairs without an evaluation record after the reevaluation (N3) ---------------------------------------------------------------------------------
def test_missing_rule_jia_hands_over():
    out = D.decide(build(base_values(), drop={3: None}), table(), ROWS, missing_rule="jia", idx=SMALL)
    assert out["reading"] == D.HAND_OVER and out["F1"] is None and out["missing_pairs"] == ["p03"]


def test_missing_rule_yi_is_the_default_and_redraws_on_n_prime():
    assert D.MISSING_PAIR_RULE == "yi"
    out = D.decide(build(base_values(), drop={3: None, 9: ["ants"]}), table(), ROWS, idx=SMALL)
    assert out["n_used"] == 48 and out["idx_sha256"] == D.idx_sha256(D.draw_idx(n=48, b=1000))
    assert [p["pair"] for p in out["positions"]] == [r[3] for r in ROWS if r[0] not in (3, 9)]           # 𝒩 in tsv order
    assert out["F1"]["arms"]["configc"]["verdict"] == D.SUP and out["F1"]["arms"]["ants"]["n"] == 48
    few = D.decide(build(base_values(), drop={i: None for i in range(6)}), table(), ROWS, idx=SMALL)
    assert few["reading"] == D.DESCRIBED_FEW and all(r["verdict"] is None for r in few["F1"]["arms"].values())


def test_missing_rule_bing_only_the_arm_lacking_a_record():
    out = D.decide(build(base_values(), drop={9: ["ants"]}), table(), ROWS, missing_rule="bing", idx=SMALL)
    assert out["F1"]["arms"]["ants"]["verdict"] == D.NOT_JUDGED_MISSING and out["F1"]["arms"]["ants"]["p"]["sup"] == 1.
    assert out["F1"]["arms"]["configc"]["verdict"] == D.SUP and out["n_used"] == 50
    out = D.decide(build(base_values(), drop={9: ["method"]}), table(), ROWS, missing_rule="bing", idx=SMALL)
    assert all(r["verdict"] == D.NOT_JUDGED_MISSING for r in out["F1"]["arms"].values())


def write_records(folder, records):
    (folder/"pairs").mkdir(parents=True, exist_ok=True)
    for name, record in records.items():
        (folder/"pairs"/f"{name}.json").write_text(json.dumps(record))


def test_collect_merges_partial_first_records_with_the_reevaluation(tmp_path):
    records = build(base_values())
    first, second = tmp_path/"first", tmp_path/"re"
    partial = dict(records["p05"], complete=False, arms={k: records["p05"]["arms"][k] for k in ("method", "configc")}, error="OSError")
    write_records(first, dict(records, p05=partial))
    rest = dict(records["p05"], complete=False, arms={k: v for k, v in records["p05"]["arms"].items() if k not in ("method", "configc")}, error="killed")
    write_records(second, dict(p05=rest))
    merged = D.collect([first, second], ROWS)
    assert set(merged["p05"]["arms"]) == set(ARMS) and len(merged["p05"]["_sources"]) == 2 and merged["p05"]["_conflicts"] == []
    assert D.complete_mask(merged, table(), ROWS).all()
    (second/"pairs"/"p06.json").write_text(json.dumps(records["p06"]))
    with pytest.raises(D.DecisionRefused, match="two complete records"):
        D.collect([first, second], ROWS)


def test_collect_records_a_conflict_and_checks_the_rows(tmp_path):
    records = build(base_values())
    first, second = tmp_path/"first", tmp_path/"re"
    write_records(first, dict(records, p05=dict(records["p05"], complete=False, error="killed")))
    again = json.loads(json.dumps(records["p05"]))
    again["arms"]["ants"]["metrics"]["seg35"] += 1e-9                               # a different number for the same cell (review m11)
    again["arms"]["fga"]["seconds"] = 99.                                           # timings are left out of the comparison
    write_records(second, dict(p05=again))
    merged = D.collect([first, second], ROWS)
    assert [c["item"] for c in merged["p05"]["_conflicts"]] == ["arm ants"]
    out = D.decide(merged, table(), ROWS, idx=SMALL)
    assert list(out["merge_conflicts"]) == ["p05"]
    (second/"pairs"/"p07.json").write_text(json.dumps(dict(records["p07"], index=8, complete=False)))
    with pytest.raises(D.DecisionRefused, match="not a row of the list"):
        D.collect([first, second], ROWS)


# ---- F3 and the descriptions ------------------------------------------------------------------------------------------------------------------------
def test_f3_intervals_only_and_dsc30_recomputed_per_resample():
    records = build(base_values())
    out = D.decide(records, table(), ROWS, idx=SMALL)
    assert set(out["F3"]["endpoints"]) == set(D.DESCRIBED) and "verdict" not in json.dumps(out["F3"])
    assert out["F3"]["endpoints"]["hd95"]["ants"]["written"].startswith("arm - method")
    rows_off = D.official_rows(records, table(), ROWS)
    draws = D.dsc30_draws(rows_off["ants"], SMALL[:5])
    for b in range(5):
        assert draws[b] == aggregate([rows_off["ants"][i] for i in SMALL[b]])["dsc30_ranking"]
    assert out["F3"]["dsc30"]["desc"]["family"] == "descriptive"


def test_f3_gives_the_descriptive_arms_and_the_90s_tier_seg35_and_seg4():
    """Preregistration 4.4 F3 (review I9): method - arm on seg35 and seg4 for the descriptive arms and the 90 s tier, unadjusted intervals, the same idx."""
    out = D.decide(build(base_values(desc=.02, t90=.05)), table(), ROWS, idx=SMALL)
    for key in ("seg35", "seg4"):
        arms = out["F3"]["described_arms"][key]
        assert set(arms) == {"desc", "t90"} and arms["t90"]["family"] == "tier_90s" and arms["desc"]["written"] == "method - arm"
        assert arms["desc"]["mean"] == pytest.approx(.02, abs=.002) and arms["t90"]["mean"] == pytest.approx(.05, abs=.002)
        assert arms["t90"]["interval"][0] > 0


def test_f3_sdlogj_mean_with_itt_and_on_outputs_only():
    sd = dict(ants=np.full(N, .11))
    out = D.decide(build(base_values(), sdlogj=sd, no_output=dict(ants=set(range(30)))), table(), ROWS, idx=SMALL)
    means = out["F3"]["sdlogj_official_mean_le_010"]
    assert means["itt"]["ants"] is True and means["outputs_only"]["ants"] is False          # review m12: identity rows make it look smooth


def test_decide_with_the_default_resampling_uses_one_idx():
    out = D.decide(build(base_values()), table(), ROWS)
    assert out["B"] == 10000 and out["idx_sha256"] == D.idx_sha256(D.draw_idx()) and out["F4"]["reading"] == D.F4_NOT and out["F4"]["n_P"] == 0


# ---- F4 (C22.H7, contract revision 7, adopted 2026-11-09) ------------------------------------------------------------------------------------------
def pq_section(rng, *, q=.10, gap=.02, xlex=None, zero=False, state="present", complete=True, ev1=True, ev2=True, side="released", lock=True):
    ause = dict(q=q+rng.normal(0, .002))
    for k in D.PQ_BASELINES:
        ause[k] = ause["q"]+gap+rng.normal(0, .002)
    if xlex is not None:
        ause["X_lex"] = ause["q"]+xlex+rng.normal(0, .0005)
    acceptance = dict(n_errors=dict(omega_e=0 if zero else 1000), scores=dict(omega_e={k: dict(ause=v) for k, v in ause.items()}))
    return dict(method_side=side, state=state, reason=None if state == "present" else "test", complete=complete, ev1=dict(passed=ev1), ev2=dict(passed=ev2),
                input_lock=dict(passed=lock), acceptance=acceptance if ev2 and lock else None)


def f4_rows(n=N, *, special=None, method_none=(), method_missing=(), **kw):
    """n positions; special: {position: keyword changes of that pair's PQ section}."""
    rng = np.random.default_rng(9)
    rows = []
    for i in range(n):
        section = pq_section(rng, **dict(kw, **(special or {}).get(i, {})))
        rows.append(dict(position=i, pair=f"p{i:02d}", method_record=i not in method_missing, method_output=i not in method_none and i not in method_missing,
                         status="hx_certified", pq=section))
    return rows


def f4(rows, idx=None):
    idx = D.draw_idx(n=len(rows)) if idx is None else idx
    return D.decide_f4(rows, idx, n_prime=len(rows))


def test_f4_table_rows():
    assert f4(f4_rows(side="bz_ii"))["reading"] == D.F4_METHOD
    assert f4(f4_rows(special={4: dict(ev1=False)}))["reading"] == D.F4_EVAL
    out = f4(f4_rows(special={4: dict(ev2=False)}))
    assert out["reading"] == D.F4_EVAL and out["eval_side"]["Ev2"] == ["p04"]
    out = f4(f4_rows(special={6: dict(lock=False)}))                    # C22.H4-2 Hm2 after the step-8 input lock (the lead's ruling (10), 11-09)
    assert out["reading"] == D.F4_EVAL and out["eval_side"]["input_lock"] == ["p06"]
    assert f4(f4_rows(state="missing"))["reading"] == D.F4_NOT
    assert f4(f4_rows(special={4: dict(zero=True)}))["reading"] == D.F4_NOT
    good = f4(f4_rows())
    assert good["reading"] == D.F4_PASS and all(r["passed"] for r in good["baselines"].values()) and good["failed"] == [] and good["n_P"] == 50
    bad = f4(f4_rows(gap=-.001))
    assert bad["reading"] == D.F4_FAIL and set(bad["failed"]) == set(D.PQ_BASELINES)


def test_f4_has_no_row_for_an_unpinned_replay_gate():
    """C22.H7-3 revision 7a: the replay gate is checked at (0) of the entry; the decision has no such row (review I2 (b))."""
    import inspect
    assert "replay" not in inspect.signature(D.decide_f4).parameters and "replay" not in inspect.signature(D.decide).parameters


def test_f4_incomplete_records_leave_p_before_ev3():
    """C22.H7-1 revision 7d: a pair whose PQ evaluation record is still incomplete leaves 𝒫 first (listed), Ev1-Ev3 are then checked on 𝒫."""
    out = f4(f4_rows(special={4: dict(complete=False, ev1=False)}))
    assert out["reading"] == D.F4_PASS and out["n_P"] == 49 and [e["pair"] for e in out["excluded"]] == ["p04"] and out["eval_side"]["Ev1"] == []


def test_f4_only_x_lex_r_fails_with_a_positive_lb():
    out = f4(f4_rows(xlex=.003))                                   # R = .003/.103 < .05 but LB > 0
    assert out["reading"] == D.F4_FAIL and out["failed"] == ["X_lex"] and out["baselines"]["X_lex"]["LB"] > 0 and out["note"]


def test_f4_only_lb_fails():
    rng = np.random.default_rng(4)
    rows = f4_rows()
    for row in rows:                                               # X_lex: a large mean gain carried by a few pairs, many small losses (LB < 0, R >= .05)
        scores = row["pq"]["acceptance"]["scores"]["omega_e"]
        scores["X_lex"]["ause"] = scores["q"]["ause"]+(.25 if row["position"] < 3 else -.002+rng.normal(0, .0005))
    out = f4(rows)
    assert out["failed"] == ["X_lex"] and out["baselines"]["X_lex"]["LB"] <= 0 and out["baselines"]["X_lex"]["R"] >= .05 and out["note"] is None


def test_f4_p_below_45_pairs_is_only_described():
    """D125 (a): |𝒫| < 45 -> F4 only described; the numbers stay, no pass or fail (review I2 (a))."""
    out = f4(f4_rows(special={i: dict(state="pq_failed") for i in range(10)}))
    assert out["reading"] == D.F4_DESCRIBED and out["n_P"] == 40 and "failed" not in out and all("passed" not in r for r in out["baselines"].values())
    assert len(out["excluded"]) == 10 and out["excluded"][0]["reason"].startswith("缺 PQ 输出：pq_failed")
    few = f4(f4_rows(n=44))
    assert few["reading"] == D.F4_DESCRIBED and few["n_prime"] == 44


def test_f4_void_draws_and_zero_ause_sum():
    one = f4(f4_rows(special={i: dict(state="missing") for i in range(1, 50)}))
    assert one["reading"] == D.F4_DESCRIBED and one["baselines"]["h"]["void_draws"] > 0 and one["baselines"]["h"]["valid_draws"] < 10000
    rows = f4_rows()
    for row in rows:
        scores = row["pq"]["acceptance"]["scores"]["omega_e"]
        scores["h"]["ause"] = scores["q"]["ause"] = 0.
    out = f4(rows)
    assert out["baselines"]["h"]["R"] is None and not out["baselines"]["h"]["passed"] and "h" in out["failed"]


def test_f4_uses_the_same_idx_and_positions_as_f1():
    rng = np.random.default_rng(9)
    pq = {i: pq_section(rng) for i in range(N)}
    out = D.decide(build(base_values(), pq=pq, drop={3: None}), table(), ROWS)
    assert out["F4"]["idx_sha256"] == out["idx_sha256"] == D.idx_sha256(D.draw_idx(n=49)) and out["F4"]["n_prime"] == 49
    assert out["F4"]["reading"] == D.F4_PASS and [p["pair"] for p in out["F4"]["P"]] == [p["pair"] for p in out["positions"]]


def test_f4_reads_the_merged_records(tmp_path):
    """The PQ section of a reevaluation's complete record replaces the first run's incomplete one (collect)."""
    rng = np.random.default_rng(9)
    pq = {i: pq_section(rng) for i in range(N)}
    records = build(base_values(), pq=pq)
    first, second = tmp_path/"first", tmp_path/"re"
    write_records(first, dict(records, p05=dict(records["p05"], complete=False, pq=dict(pq[5], complete=False, acceptance=None))))
    write_records(second, dict(p05=records["p05"]))
    out = D.decide(D.collect([first, second], ROWS), table(), ROWS, idx=SMALL)
    assert out["F4"]["n_P"] == 50 and out["F4"]["reading"] == D.F4_PASS


def test_f4_problems_inside_f4_read_eval_side_not_a_refusal():
    """Review2 m5: records naming different method sides, or an AUSE missing on 𝒫, read 不判（评测侧） with the reason; F1-F3 are not stopped."""
    rows = f4_rows(special={3: dict(side="bz_ii")})
    out = f4(rows)
    assert out["reading"] == D.F4_EVAL and "method sides" in out["note"]
    rows = f4_rows()
    rows[5]["pq"]["acceptance"]["scores"]["omega_e"]["h"]["ause"] = None
    out = f4(rows)
    assert out["reading"] == D.F4_EVAL and "AUSE missing" in out["note"]


def test_f4_reasons_follow_revision_7d():
    """Review2 m10: the reasons a pair leaves 𝒫 are said apart -- the method arm without an evaluation record (rule 丙), the method without output, no PQ
    record, PQ's state, a PQ record still incomplete."""
    rows = f4_rows(method_missing={0}, method_none={1}, special={2: dict(state="pq_absent"), 3: dict(complete=False)})
    rows[4]["pq"] = None
    reasons = {e["pair"]: e["reason"] for e in f4(rows)["excluded"]}
    assert reasons["p00"].startswith("方法臂缺评测记录") and reasons["p01"].startswith("方法没有输出") and reasons["p02"].startswith("缺 PQ 输出：pq_absent")
    assert "不完整" in reasons["p03"] and reasons["p04"].startswith("缺 PQ 评测记录")


def test_f4_class_b_is_pq_b_class():
    """F4's class B on the hold-out is bayesreg.eval.holdout.pq_b_class (the function the replay gate checks) with the hold-out resampling."""
    from bayesreg.eval import holdout as H
    rows = f4_rows()
    idx = D.draw_idx(n=N)
    out = f4(rows, idx)
    P = list(range(N))
    table = {k: [r["pq"]["acceptance"]["scores"]["omega_e"][k]["ause"] for r in rows] for k in ("q",)+D.PQ_BASELINES}
    seen = []

    def interval(delta):
        seen.append(1)
        means = D.boot_means(np.asarray(delta), idx)
        return dict(lo=float(np.percentile(means, 2.5)))
    direct = H.pq_b_class(table, interval=interval)
    assert seen and all(out["baselines"][k]["LB"] == direct["baselines"][k]["LB"] and out["baselines"][k]["R"] == direct["baselines"][k]["R"]
                        for k in D.PQ_BASELINES)
    assert [p["position"] for p in out["P"]] == P


# ---- the command line: the runs come from the locks (review I8) ---------------------------------------------------------------------------------------
def test_main_refuses_without_the_pinned_preregistration_or_guard(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(D, "PREREG_SHA256", None)
    assert D.main(["--out", str(tmp_path/"d.json")]) == 2
    assert "PREREG_SHA256" in capsys.readouterr().err and not (tmp_path/"d.json").exists()
    monkeypatch.setattr(D, "PREREG_SHA256", "a"*64)
    monkeypatch.setattr(H, "PREREG_SHA256", "a"*64)
    monkeypatch.setattr(D, "SDLOGJ_GUARD", None)
    assert D.main(["--out", str(tmp_path/"d.json")]) == 2
    assert "SDLOGJ_GUARD" in capsys.readouterr().err and not (tmp_path/"d.json").exists()


def test_main_checks_rule_one_first(tmp_path, capsys, monkeypatch):
    """Contract revision 9 draft C24.3-7 (decision_log 2026-11-10 ~2:59 AM CST (7); A5b): holdout_decide checks rule one before it reads anything: on this tree there is no
    runs/holdout_B_v1/pinned_code.json, so it refuses (exit 2) before the preregistration, the arm table or a lock is read."""
    read = []
    monkeypatch.setattr(D, "PREREG_SHA256", "p"*64)
    monkeypatch.setattr(H, "PREREG_SHA256", "p"*64)
    monkeypatch.setattr(D, "SDLOGJ_GUARD", .02)
    def prereg():
        read.append("prereg")
        raise H.HoldoutRefused("prereg", "stub")
    monkeypatch.setattr(H, "check_prereg", prereg)
    monkeypatch.setattr(H, "run_lines", lambda: read.append("locks") or [])
    assert D.main(["--out", str(tmp_path/"d.json")]) == 2
    assert "rule one" in capsys.readouterr().err and read == [] and not (tmp_path/"d.json").exists()
    seen = []
    monkeypatch.setattr(H, "rule_one", lambda repo=None: seen.append(1) or dict(pinned="1"*40))
    assert D.main(["--out", str(tmp_path/"e.json")]) == 2 and seen == [1] and read == ["prereg"]       # then the rest in its order


def test_main_takes_the_runs_from_the_locks(tmp_path, capsys, monkeypatch):
    records = build(base_values())
    first, second = tmp_path/"first", tmp_path/"re"
    write_records(first, dict(records, p05=dict(records["p05"], complete=False, arms={}, error="killed")))
    write_records(second, dict(p05=records["p05"]))
    line = dict(prereg_sha256="p"*64, arm_table_sha256="t"*64, manifest_sha256="m"*64)
    monkeypatch.setattr(D, "PREREG_SHA256", "p"*64)
    monkeypatch.setattr(H, "PREREG_SHA256", "p"*64)
    monkeypatch.setattr(H, "check_prereg", lambda: (b"", dict(sha256="p"*64)))
    monkeypatch.setattr(H, "load_arm_table", lambda raw: (table(), dict(sha256="t"*64)))
    monkeypatch.setattr(H, "load_pairs", lambda: (ROWS, {}))
    monkeypatch.setattr(H, "run_lines", lambda: [(Path("master.lock"), dict(line, out_dir=str(first))), (Path("reeval_1.lock"), dict(line, out_dir=str(second)))])
    monkeypatch.setattr(H, "rule_one", lambda repo=None: dict(stub="rule one (tested in test_main_checks_rule_one_first)", pinned="1"*40))
    assert D.main(["--out", str(tmp_path/"x.json"), "--runs", str(second), str(first)]) == 2              # another order than the locks'
    assert "lock order" in capsys.readouterr().err and not (tmp_path/"x.json").exists()
    assert D.main(["--out", str(tmp_path/"y.json"), "--runs", str(first)]) == 2                          # a reevaluation left out
    assert D.main(["--out", str(tmp_path/"d.json")]) == 0
    document = json.loads((tmp_path/"d.json").read_text())
    assert document["runs"] == [str(first), str(second)] and document["n_used"] == 50 and document["F1"]["arms"]["configc"]["verdict"] == D.SUP
    monkeypatch.setattr(H, "run_lines", lambda: [(Path("master.lock"), dict(line, out_dir=str(first))),
                                                 (Path("reeval_1.lock"), dict(line, out_dir=str(second), arm_table_sha256="u"*64))])
    assert D.main(["--out", str(tmp_path/"z.json")]) == 2


def test_a_null_dice_of_the_entry_reads_back_as_nan_and_f3_is_unchanged():
    """A5b (review minor 7): the entry writes a non-finite Dice as JSON null (bayesreg.solve.export.dumps); holdout_decide's official_rows reads it back as NaN
    (_nulls_as_nan), so DSC30 and F3 are what the NaN rows give and nothing raises (np.isnan(None) did)."""
    import copy
    records = build(base_values())
    nulls = copy.deepcopy(records)
    n_null = 0
    for record in nulls.values():
        for entry in [record["identity"]]+list(record["arms"].values()):
            official = entry.get("official")
            if isinstance(official, dict):
                for key in ("dice_labels", "initial_dice_labels"):
                    n_null += sum(1 for v in official[key] if v != v)
                    official[key] = [None if v != v else v for v in official[key]]
    assert n_null > 0
    back = D._nulls_as_nan(dict(dice_labels=[.5, None], dice=None, hd95=2.))
    assert back["dice_labels"][0] == .5 and np.isnan(back["dice_labels"][1]) and np.isnan(back["dice"]) and back["hd95"] == 2.
    rows_nan, rows_null = D.official_rows(records, table(), ROWS), D.official_rows(nulls, table(), ROWS)
    assert D.dsc30_point(rows_null["method"]) == D.dsc30_point(rows_nan["method"])
    a, b = D.decide(records, table(), ROWS, idx=SMALL), D.decide(nulls, table(), ROWS, idx=SMALL)
    assert json.dumps(a["F3"], sort_keys=True, default=str) == json.dumps(b["F3"], sort_keys=True, default=str)
