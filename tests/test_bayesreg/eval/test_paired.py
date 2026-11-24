"""bayesreg.eval.paired: paired statistics on known examples."""
import csv
import json

import numpy as np
import pytest

from bayesreg.eval import paired as P


def pairs(n):
    return [f"S{k:02d}__S{k+1:02d}" for k in range(n)]


def test_known_example():
    names = pairs(5)
    a = {p: float(k+1) for k, p in enumerate(names)}
    b = {p: 0. for p in names}
    r = P.paired_compare(a, b, measure="seg35", block="pairs", resamples=2000)
    assert (r["n"], r["mean_diff"], r["median_diff"], r["n_positive"], r["n_negative"], r["n_zero"]) == (5, 3., 3., 5, 0, 0)
    assert r["wilcoxon"]["p"] == pytest.approx(.0625, abs=1e-12) and r["wilcoxon"]["statistic"] == 0.
    assert r["direction"] == 1 and r["n_a_better"] == 5
    lo, hi = r["bootstrap"]["ci95_mean"]
    assert 1. <= lo <= 3. <= hi <= 5.
    r = P.paired_compare(b, a, measure="hd95", block="pairs", resamples=100)
    assert r["n_positive"] == 0 and r["n_negative"] == 5 and r["n_a_better"] == 5      # smaller HD95 is better


def test_constant_difference_and_zero():
    names = pairs(6)
    a = {p: .8 for p in names}
    b = {p: .75 for p in names}
    for block in ("pairs", "subjects"):
        r = P.paired_compare(a, b, measure="seg4", block=block, resamples=500)
        c = .8-.75
        assert r["bootstrap"]["ci95_mean"] == [pytest.approx(c, abs=1e-15)]*2
        assert r["bootstrap"]["ci95_median"] == [c, c]
    r = P.paired_compare(a, dict(a), measure="seg4", resamples=50)
    assert r["wilcoxon"]["p"] == 1. and r["n_zero"] == 6 and r["mean_diff"] == 0.


def test_weighted_mean_median():
    d = np.array([1., 2., 10.])
    w = np.array([[1, 1, 1], [2, 0, 1], [0, 0, 1], [1, 1, 0]])
    mean, median = P.weighted_mean_median(d, w)
    assert np.allclose(mean, [13/3, 4., 10., 1.5], rtol=0, atol=1e-15)
    assert list(median) == [2., 1., 10., 1.5]
    rng = np.random.default_rng(1)
    for n in (5, 6):
        x = rng.normal(size=n)
        mean, median = P.weighted_mean_median(x, np.ones((1, n), int))
        assert median[0] == np.median(x) and mean[0] == pytest.approx(np.mean(x), rel=1e-15)
    with pytest.raises(ValueError):
        P.weighted_mean_median(d, np.zeros((1, 3), int))


def test_subject_block_bootstrap_is_seeded():
    names = [f"S{k:02d}__S{(k*7+3) % 20:02d}" for k in range(20)]       # 20 pairs over 20 subjects, shared
    rng = np.random.default_rng(3)
    a = {p: v for p, v in zip(names, rng.normal(.8, .02, 20))}
    b = {p: v for p, v in zip(names, rng.normal(.79, .02, 20))}
    r1 = P.paired_compare(a, b, measure="seg35", tag="x", resamples=400)
    r2 = P.paired_compare(a, b, measure="seg35", tag="x", resamples=400)
    r3 = P.paired_compare(a, b, measure="seg35", tag="y", resamples=400)
    assert r1["bootstrap"] == r2["bootstrap"] and r1["bootstrap"]["ci95_mean"] != r3["bootstrap"]["ci95_mean"]
    assert r1["bootstrap"]["block"] == "subjects" and r1["bootstrap"]["n_subjects"] == 20
    lo, hi = r1["bootstrap"]["ci95_mean"]
    assert lo <= r1["mean_diff"] <= hi
    with pytest.raises(ValueError):
        P.paired_compare({"nopair": 1.}, {"nopair": 0.}, resamples=10)


def test_long_csv_and_main(tmp_path, capsys):
    path = tmp_path/"long.csv"
    names = pairs(4)
    rows = []
    for k, p in enumerate(names):
        rows.append(dict(set="L2R_val", split="dev", method="bayesreg", arm="tissue", pair=p, measure="seg35",
                         value=.8+.01*k, run_id=f"r{k}", field_role="native_formal", run_status="converged_B"))
        if k < 3:
            rows.append(dict(set="L2R_val", split="dev", method="configC", arm="", pair=p, measure="seg35",
                             value=.79, run_id=f"c{k}", field_role="native_formal", run_status="x"))
    P.write_long(path, rows)
    rows = P.read_long(path)
    r = P.compare_long(rows, "bayesreg:tissue", "configC", ["seg35"], set_name="L2R_val", resamples=100)[0]
    assert r["n"] == 3 and r["missing_in_b"] == [names[3]] and r["n_positive"] == 3
    out = tmp_path/"paired.json"
    assert P.main([str(path), "--a", "bayesreg:tissue", "--b", "configC", "--measures", "seg35", "--out", str(out),
                   "--resamples", "50", "--block", "pairs"]) == 0
    document = json.loads(out.read_text())
    assert document["results"][0]["n"] == 3 and document["seed"] == P.DEFAULT_SEED
    assert document["roles"] == ["native_formal", "disp_only"]
    assert "missing in a 0, in b 1" in capsys.readouterr().out                 # review EV item 11: printed
    assert P.main([str(path), "--a", "bayesreg:tissue", "--b", "configC", "--out", str(out)]) == 2
    with open(path, "a", newline="") as stream:
        csv.writer(stream).writerow(["L2R_val", "dev", "configC", "", names[0], "seg35", ".7", "dup", "native_formal", ""])
    with pytest.raises(ValueError, match="duplicate"):
        P.values_by_pair(P.read_long(path), measure="seg35", method="configC")


def test_values_round_trip_and_legacy_files(tmp_path):
    for value in (0., -0., .1, 1/3, 1e-300, 7, True, np.float64(2.5), np.int64(4)):
        assert P.parse_value(P.format_value(value)) == float(value)
    assert [P.format_value(v) for v in (np.inf, -np.inf, np.nan)] == ["inf", "-inf", "nan"]
    assert np.isnan(P.parse_value("nan")) and P.parse_value("-inf") == -np.inf
    with pytest.raises(ValueError):
        P.parse_value("1e999")                    # non-finite only through the three strings
    legacy = tmp_path/"legacy.csv"
    with open(legacy, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(P.REQUIRED_COLUMNS)
        writer.writerow(["s", "dev", "m", "", "S00__S01", "seg35", ".8", "r"])
    rows = P.read_long(legacy)
    assert rows[0]["field_role"] == "" and rows[0]["value"] == .8
    assert rows[0]["structure_registry"] == "" and not P.is_staging(rows[0])        # older file: not staging
    assert P.select_values(rows, measure="seg35", method="m") == ({}, {"S00__S01": ""}, {})   # no role: not compared
    assert P.values_by_pair(rows, measure="seg35", method="m") == {"S00__S01": .8}


def test_staging_rows_are_excluded_by_default():
    """Review PS1_PS2 item d: rows of a run on the staging segmenter registry (structure_registry
    staging_unregistered) are left out and listed unless include_staging; registered, n/a and empty are admitted."""
    rows = [dict(set="s", split="dev", method="m", arm="", pair=f"S0{k}__S0{k+1}", measure="seg35", value=.8,
                 run_id=f"r{k}", field_role="native_formal", run_status="", structure_registry=label)
            for k, label in enumerate(("registered", "staging_unregistered", "n/a", ""))]
    values, by_role, staging = P.select_values(rows, measure="seg35", method="m")
    assert sorted(values) == ["S00__S01", "S02__S03", "S03__S04"] and by_role == {} and staging == {"S01__S02": "r1"}
    values, _, staging = P.select_values(rows, measure="seg35", method="m", include_staging=True)
    assert len(values) == 4 and staging == {}
    assert len(P.values_by_pair(rows, measure="seg35", method="m")) == 4                # the helper keeps every row
    assert len(P.values_by_pair(rows, measure="seg35", method="m", include_staging=False)) == 3


def test_nonfinite_pairs_are_listed_not_dropped():
    """Review EV item 3: a pair with HD95 = inf on one side is reported and counted, not silently missing."""
    names = pairs(6)
    a = {p: 2.+k for k, p in enumerate(names)}
    b = {p: 1. for p in names}
    a[names[1]] = np.inf
    b[names[4]] = np.nan
    r = P.paired_compare(a, b, measure="hd95", block="pairs", resamples=50)
    assert (r["n"], r["n_common"], r["n_nonfinite"], r["n_nonfinite_a"], r["n_nonfinite_b"]) == (4, 6, 2, 1, 1)
    assert r["nonfinite_pairs"] == {names[1]: dict(a="inf", b="1.0"), names[4]: dict(a="6.0", b="nan")}
    assert r["pairs"] == [names[k] for k in (0, 2, 3, 5)] and r["mean_diff"] == np.mean([1., 3., 4., 6.])
    assert r["missing_in_a"] == [] and r["missing_in_b"] == []


def test_roles_and_holm():
    names = pairs(5)
    rows = []
    for k, p in enumerate(names):
        for method, value, role in (("a", .8+.01*k, "native_formal"), ("b", .79-.002*k,
                                                                         "handover_only" if k == 0 else "native_formal")):
            rows.append(dict(set="s", split="dev", method=method, arm="", pair=p, measure="seg35", value=value,
                             run_id=f"{method}{k}", field_role=role, run_status=""))
            rows.append(dict(set="s", split="dev", method=method, arm="", pair=p, measure="seg4", value=value+.05,
                             run_id=f"{method}{k}", field_role=role, run_status=""))
    results = P.compare_long(rows, "a", "b", ["seg35", "seg4"], resamples=20, block="pairs")
    for r in results:
        assert r["n"] == 4 and r["excluded_by_role_b"] == {names[0]: "handover_only"} and r["missing_in_b"] == [names[0]]
    p = [r["wilcoxon"]["p"] for r in results]
    assert [r["wilcoxon"]["p_holm"] for r in results] == P.holm(p)
    assert all(r["wilcoxon"]["holm_family"] == ["seg35", "seg4"] for r in results)
    assert P.compare_long(rows, "a", "b", ["seg35"], resamples=20, block="pairs", roles=None)[0]["n"] == 5
    assert P.parse_roles("any") is None and P.parse_roles("native_formal, handover_only") == (
        "native_formal", "handover_only")
    # Holm on a known example: p = (.01, .04, .03, None) -> (.03, .06, .06, None)
    adjusted = P.holm([.01, .04, .03, None])
    assert adjusted[:3] == pytest.approx([.03, .06, .06], abs=1e-15) and adjusted[3] is None
    assert P.holm([.6, .5]) == [1., 1.]
