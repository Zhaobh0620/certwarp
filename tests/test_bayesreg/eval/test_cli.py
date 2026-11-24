"""bayesreg.eval.cli and truth: end to end on synthetic stages; the subject guard refuses before any label is read."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from jd_curl.three_d.nifti import file_sha256

from bayesreg.eval import cli
from bayesreg.eval import metrics as M
from bayesreg.eval import paired as P
from bayesreg.eval import truth as T
from bayesreg.eval.fields import u_to_d
from bayesreg.io.guard import SubjectSealed, load_dev_allowlist

from . import _synthetic as syn

P01 = f"{syn.DEV[0]}__{syn.DEV[1]}"
P23 = f"{syn.DEV[2]}__{syn.DEV[3]}"
HANDOVER_ROLE = "certified_native_initializer_NOT_native_outer_commit"


@pytest.fixture
def world(tmp_path):
    data = tmp_path/"data"
    arrays = {P01: syn.pair_arrays(seed=1), P23: syn.pair_arrays(seed=2)}
    syn.write_pair(data, P01, syn.DEV[0], syn.DEV[1], arrays[P01])
    syn.write_pair(data, P23, syn.DEV[2], syn.DEV[3], arrays[P23])
    allowlist = syn.write_allowlist(tmp_path/"dev_subjects.json")
    (tmp_path/"eval").mkdir()
    return dict(tmp=tmp_path, data=data, arrays=arrays, allowlist=allowlist)


def common(world, *extra):
    return ["--pairs-stage", syn.STAGE, "--data-root", str(world["data"]), "--allowlist", str(world["allowlist"]), *extra]


def read_long(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def pair_sha(world, pair):
    return file_sha256(world["data"]/syn.STAGE/pair/"pair.npz")


def test_end_to_end_bayesreg_configc_and_disp(world):
    tmp = world["tmp"]
    out = tmp/"eval"
    u1 = syn.d_to_u(syn.smooth_d())
    u2 = syn.d_to_u(syn.smooth_d(amplitude=(.5, .6, -.4), freq=2))
    prov = dict(pair_sha256=pair_sha(world, P01), fixed_subject=syn.DEV[0], moving_subject=syn.DEV[1])
    run1, _ = syn.write_bayesreg_run(tmp/"runs"/"b1", P01, u1, pair_provenance=prov)
    run2 = syn.write_configc_run(tmp/"runs"/"c2", P23, u2, input_sha256=pair_sha(world, P23), pair_dataset=syn.STAGE)
    code = cli.main([str(run1), str(run2), *common(world, "--out", "metrics.json", "--long", str(tmp/"long.csv"),
                                                    "--summary", str(tmp/"summary.json"), "--out-dir", str(out))])
    assert code == 0
    assert not (run1/"metrics.json").exists() and not (run2/"metrics.json").exists()      # never into a run dir
    rec1 = json.loads((out/"b1"/"metrics.json").read_text())
    rec2 = json.loads((out/"c2"/"metrics.json").read_text())
    assert (rec1["method"], rec1["arm"], rec1["kind"], rec1["pair"]) == ("bayesreg", "tissue", "bayesreg", P01)
    assert (rec2["method"], rec2["arm"], rec2["kind"]) == ("configC", "", "configC")
    assert (rec1["field_role"], rec2["field_role"], rec2["metrics_role"]) == (
        "native_formal", "native_formal", "formal_native_outer_commit")
    assert rec1["comparable_by_default"] and rec2["completion"]["exit_code"] == 0
    assert rec1["input_check"]["checked"] == dict(pair_sha256=True, fixed_subject=True, moving_subject=True)
    assert rec2["input_check"]["checked"] == dict(pair_sha256=True, stage=True)
    assert rec1["units"]["hd95_equals_mm"] and rec1["nonfinite_metrics"] == {}
    truth = dict(syn.pair_arrays(seed=1))
    truth = {k: truth[k] for k in ("fixed_seg35", "moving_seg35", "fixed_seg4", "moving_seg4")}
    truth["brain"] = truth["fixed_seg4"] > 0
    expected, _, _ = M.evaluate_field(u_to_d(u1, (1., 1., 1.)), truth, M.OASIS, (1., 1., 1.))
    for key in ("seg35", "seg4", "dice_official", "hd95", "sdlogj_official", "sdlogj_brain"):
        assert rec1["metrics"][key] == expected[key]
    for rec in (rec1, rec2):
        assert rec["certificate"]["mode"] == "full" and rec["certificate"]["passed"]
        assert rec["certificate"]["u_equals_run_u_bitwise"] and rec["certificate"]["run_certificate_state_hash_matches"]
        assert rec["certificate"]["cells"]["matches_map"]
        assert rec["field"]["roundtrip"]["passed"] and rec["truth"]["keys_read"] == [
            "fixed_seg35", "moving_seg35", "fixed_seg4", "moving_seg4"]
        assert rec["guard"]["unsealed"] is False and rec["guard"]["allowlist_sources_of"]["fixed"] == ["synthetic_dev"]
        assert {s["name"] for s in rec["guard"]["sealed_lists"]} == {"holdout_B_v1", "H1", "H4", "H3", "H2"}
        assert "Definitions" in rec["definitions"]["metrics"] and rec["code"]["git_head"]
        assert "bayesreg/eval/metrics.py" in rec["code"]["file_sha256"]
    rows = read_long(tmp/"long.csv")
    assert list(rows[0]) == list(cli.LONG_COLUMNS) == list(P.LONG_COLUMNS)
    measures = {(r["run_id"], r["measure"]) for r in rows}
    for measure in ("seg35", "seg4", "dice_official", "hd95", "sdlogj_official", "sdlogj_brain", "logj_q01_brain",
                    "frac_abs_logj_gt_log2_brain", "central_det_nonpositive", "cert_full_passed",
                    "cert_full_n_infeasible", "cert_n_infeasible_brain", "cert_n_uncertain_background", "seg35_label_01",
                    "seg4_label_04", "field_present"):
        assert ("b1", measure) in measures and ("c2", measure) in measures, measure
    assert all(r["field_role"] == "native_formal" for r in rows)
    assert {r["run_status"] for r in rows} == {"converged_B", "native_budget_exhausted"}
    seg35 = [r for r in rows if r["run_id"] == "b1" and r["measure"] == "seg35"][0]
    assert float(seg35["value"]) == rec1["metrics"]["seg35"] and seg35["set"] == syn.STAGE and seg35["split"] == "dev"
    summary = json.loads((tmp/"summary.json").read_text())
    assert {(g["method"], g["field_role"], g["runs"]) for g in summary["groups"]} == {
        ("bayesreg", "native_formal", 1), ("configC", "native_formal", 1)}
    group = summary["groups"][0]
    assert group["official_aggregate"]["cases"] == 1 and group["certificate_by_mode"] == {"full": dict(passed=1, evaluated=1)}
    assert group["value_counts"]["hd95"] == dict(n=1, n_nonfinite=0, nonfinite={})
    # a baseline's disp.npz: jacobian_only by default; faces need not be zero
    d = syn.smooth_d(amplitude=(.4, .3, .2))
    d[:, 0] = .1
    (tmp/"runs"/"base").mkdir()
    np.savez(tmp/"runs"/"base"/"disp.npz", d=d.astype(np.float32))
    args = [str(tmp/"runs"/"base"), *common(world, "--out", "m.json", "--long", str(tmp/"long2.csv"), "--pair", P01,
                                            "--out-dir", str(out))]
    assert cli.main(args) == 2                                   # --method required for a disp.npz directory
    assert cli.main(args+["--method", "baseline_x", "--arm", "default"]) == 0
    rec = json.loads((out/"base"/"m.json").read_text())
    assert rec["certificate"]["mode"] == "jacobian_only" and rec["certificate"]["passed"]
    assert rec["certificate"]["faces_max_abs_u_mm"] > 0 and rec["field"]["disp"]["stored_dtype"] == "float32"
    assert rec["field_role"] == "disp_only" and rec["input_check"]["recorded"] is False
    assert ("base", "cert_jacobian_only_passed") in {(r["run_id"], r["measure"]) for r in read_long(tmp/"long2.csv")}


def test_out_dir_is_required_and_never_inside_a_run(world):
    """Review EV item 1: no output in an evaluated run directory, by default or through --out-dir."""
    tmp = world["tmp"]
    run, _ = syn.write_bayesreg_run(tmp/"run", P01, syn.d_to_u(syn.smooth_d()))
    before = sorted(p.name for p in run.iterdir())
    assert cli.main([str(run), *common(world, "--out", "m.json", "--long", str(tmp/"l.csv"))]) == 2
    assert cli.main([str(run), *common(world, "--out", "m.json", "--long", str(tmp/"l.csv"),
                                       "--out-dir", str(run/"eval"))]) == 2
    assert cli.main([str(run), *common(world, "--out", "m.json", "--long", str(run/"l.csv"),
                                       "--out-dir", str(tmp/"eval"))]) == 2
    assert sorted(p.name for p in run.iterdir()) == before and not (tmp/"l.csv").exists()


def test_unfinished_configc_run_is_refused_before_anything(world, monkeypatch):
    tmp = world["tmp"]
    u = syn.d_to_u(syn.smooth_d())
    running = syn.write_configc_run(tmp/"running", P23, u, exit_code=None)
    good, _ = syn.write_bayesreg_run(tmp/"good", P01, u)
    monkeypatch.setattr(np, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("an npz file was opened")))
    assert cli.main([str(good), str(running), *common(world, "--out", "m.json", "--long", str(tmp/"l.csv"),
                                                       "--out-dir", str(tmp/"eval"))]) == 2
    assert not any((tmp/"eval").iterdir()) and not (tmp/"l.csv").exists()


def test_handover_runs_are_tagged_and_kept_apart(world):
    """Review EV item 2: handover_only states carry their role, form their own summary group and are not compared
    by default; a no_native_result run is recorded without metrics."""
    tmp = world["tmp"]
    out = tmp/"eval"
    u = syn.d_to_u(syn.smooth_d())
    native = syn.write_configc_run(tmp/"native", P01, u)
    handover = syn.write_configc_run(tmp/"handover", P23, u, metrics_role=HANDOVER_ROLE)
    none = syn.write_configc_run(tmp/"none", P23, u, metrics_role="no_native_result")
    for run, long_name in ((native, "a.csv"), (handover, "b.csv"), (none, "c.csv")):
        assert cli.main([str(run), *common(world, "--out", "m.json", "--long", str(tmp/long_name), "--out-dir",
                                           str(out), "--cert", "none", "--summary", str(tmp/(long_name+".json")))]) == 0
    rec = json.loads((out/"handover"/"m.json").read_text())
    assert rec["field_role"] == "handover_only" and not rec["comparable_by_default"] and "handover_only" in rec["note"]
    assert rec["metrics"]["seg35"] is not None
    rec = json.loads((out/"none"/"m.json").read_text())
    assert rec["field_role"] == "no_native_result" and rec["metrics"] is None and rec["field_present"] is False
    summary = json.loads((tmp/"b.csv.json").read_text())
    assert [(g["field_role"], g["comparable_by_default"]) for g in summary["groups"]] == [("handover_only", False)]
    rows = P.read_long([tmp/"a.csv", tmp/"b.csv"])
    assert {r["field_role"] for r in rows} == {"native_formal", "handover_only"}
    # both configC; give the handover one another arm name so that the two are a pair-wise comparison
    for r in rows:
        if r["field_role"] == "handover_only":
            r["arm"], r["pair"] = "h", P01
    result = P.compare_long(rows, "configC:h", "configC", ["seg35"], resamples=20)[0]
    assert result["n"] == 0 and result["excluded_by_role_a"] == {P01: "handover_only"}
    result = P.compare_long(rows, "configC:h", "configC", ["seg35"], resamples=20, roles=None)[0]
    assert result["n"] == 1 and result["excluded_by_role_a"] == {}


def test_staging_registry_runs_are_flagged(world, capsys):
    """Review PS item 2: a structure run made with the staging registry (未正式登记) carries the label in its record,
    on the printed line and in the summary counts; the metrics are computed as for any run."""
    tmp = world["tmp"]
    out = tmp/"eval"
    u = syn.d_to_u(syn.smooth_d())
    label = dict(status="staging_unregistered", path="/x/segmenters.json", staging_opt_in=True,
                 development_diagnostic_only=True)
    staging, _ = syn.write_bayesreg_run(tmp/"runs"/"s1", P01, u, channel_source="structure",
                                        status_extra=dict(structure_registry=label))
    tissue, _ = syn.write_bayesreg_run(tmp/"runs"/"t1", P23, u)
    capsys.readouterr()
    assert cli.main([str(staging), str(tissue), *common(world, "--out", "m.json", "--long", str(tmp/"l.csv"),
                                                        "--out-dir", str(out), "--cert", "none", "--summary",
                                                        str(tmp/"s.json"))]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert "[staging_unregistered]" in [line for line in printed if line.startswith("s1")][0]
    assert "[staging_unregistered]" not in [line for line in printed if line.startswith("t1")][0]
    rec = json.loads((out/"s1"/"m.json").read_text())
    assert rec["development_diagnostic_only"] is True and rec["structure_registry"]["status"] == "staging_unregistered"
    assert rec["metrics"]["seg35"] is not None
    rec = json.loads((out/"t1"/"m.json").read_text())
    assert rec["development_diagnostic_only"] is False and rec["structure_registry"] is None
    groups = {g["arm"]: g for g in json.loads((tmp/"s.json").read_text())["groups"]}
    assert groups["structure"]["staging_unregistered_runs"] == 1
    assert groups["structure"]["structure_registry_status"] == {"staging_unregistered": 1}
    assert groups["tissue"]["staging_unregistered_runs"] == 0 and groups["tissue"]["structure_registry_status"] == {"n/a": 1}
    long = read_long(tmp/"l.csv")
    assert list(long[0]) == list(cli.LONG_COLUMNS)
    # review PS1_PS2 item d: the long CSV carries the label (column structure_registry)
    assert {r["structure_registry"] for r in long if r["run_id"] == "s1"} == {"staging_unregistered"}
    assert {r["structure_registry"] for r in long if r["run_id"] == "t1"} == {"n/a"}


def test_staging_runs_are_grouped_apart_and_left_out_of_paired(world, tmp_path):
    """Review PS1_PS2 item d: a staging run and a registered run of the same (set, method, arm, field_role) form two
    summary groups (never averaged together); bayesreg.eval.paired leaves the staging rows out by default, lists them
    (excluded_staging_*) and admits them with --include-staging."""
    tmp = world["tmp"]
    out = tmp/"eval_d"
    u = syn.d_to_u(syn.smooth_d())
    staging_label = dict(status="staging_unregistered", path="/x/segmenters.json", staging_opt_in=True,
                         development_diagnostic_only=True)
    registered_label = dict(status="registered", path="/repo/configs/bayesreg/segmenters.json", staging_opt_in=False,
                            development_diagnostic_only=False)
    staging, _ = syn.write_bayesreg_run(tmp/"runs_d"/"s1", P01, u, channel_source="structure",
                                        status_extra=dict(structure_registry=staging_label))
    registered, _ = syn.write_bayesreg_run(tmp/"runs_d"/"r1", P23, u, channel_source="structure",
                                           status_extra=dict(structure_registry=registered_label))
    long_path, summary_path = tmp/"ld.csv", tmp/"sd.json"
    assert cli.main([str(staging), str(registered), *common(world, "--out", "m.json", "--long", str(long_path),
                                                            "--out-dir", str(out), "--cert", "none", "--summary",
                                                            str(summary_path))]) == 0
    groups = [g for g in json.loads(summary_path.read_text())["groups"] if g["arm"] == "structure"]
    assert sorted((g["structure_registry"], g["runs"], g["staging_unregistered_runs"]) for g in groups) == \
        [("registered", 1, 0), ("staging_unregistered", 1, 1)]
    assert all(len(g["structure_registry_status"]) == 1 for g in groups)
    rows = P.read_long(long_path)
    roles = sorted({r["field_role"] for r in rows})
    for r in rows:                                    # one method "x" with both runs, against itself as "y"
        r["method"], r["arm"], r["pair"] = "x", "", P01
    rows_b = [dict(r, method="y", structure_registry="registered") for r in rows if r["run_id"] == "r1"]
    result = P.compare_long(rows+rows_b, "x", "y", ["seg35"], resamples=20, roles=roles)[0]
    assert result["excluded_staging_a"] == {P01: "s1"} and result["excluded_staging_b"] == {}
    assert result["n"] == 1 and result["include_staging"] is False
    with pytest.raises(ValueError, match="duplicate"):                  # admitted, the staging run collides
        P.compare_long(rows+rows_b, "x", "y", ["seg35"], resamples=20, roles=roles, include_staging=True)
    paired_out = tmp_path/"paired.json"
    P.write_long(tmp_path/"x.csv", rows_b+[r for r in rows if r["run_id"] == "s1"])
    assert P.main([str(tmp_path/"x.csv"), "--a", "x", "--b", "y", "--measures", "seg35", "--roles", "any",
                   "--out", str(paired_out), "--resamples", "20"]) == 0
    document = json.loads(paired_out.read_text())
    assert document["include_staging"] is False and document["results"][0]["n"] == 0
    assert document["results"][0]["excluded_staging_a"] == {P01: "s1"}
    assert P.main([str(tmp_path/"x.csv"), "--a", "x", "--b", "y", "--measures", "seg35", "--roles", "any",
                   "--out", str(tmp_path/"paired2.json"), "--resamples", "20", "--include-staging"]) == 0
    document = json.loads((tmp_path/"paired2.json").read_text())
    assert document["include_staging"] is True and document["results"][0]["n"] == 1


def test_failed_run_and_evaluation_error_are_recorded(world):
    tmp = world["tmp"]
    out = tmp/"eval"
    u = syn.d_to_u(syn.smooth_d())
    failed, _ = syn.write_bayesreg_run(tmp/"failed", P01, u, with_disp=False)
    broken = syn.write_configc_run(tmp/"broken", P23, np.zeros((5, 5, 5, 3)))
    code = cli.main([str(failed), str(broken), *common(world, "--out", "metrics.json", "--long", str(tmp/"l.csv"),
                                                        "--out-dir", str(out))])
    assert code == 1
    rec = json.loads((out/"failed"/"metrics.json").read_text())
    assert rec["field_present"] is False and rec["metrics"] is None and "no field evaluated" in rec["note"]
    assert rec["field_role"] == "no_native_result"
    rec = json.loads((out/"broken"/"metrics.json").read_text())
    assert rec["error"].startswith("FieldError") and rec["metrics"] is None and "Traceback" in rec["traceback"]
    assert rec["field_role"] == "native_formal"
    rows = {(r["run_id"], r["measure"]): r["value"] for r in read_long(tmp/"l.csv")}
    assert rows[("failed", "field_present")] == "0" and rows[("broken", "evaluation_error")] == "1"


def test_outputs_are_never_overwritten(world, monkeypatch):
    tmp = world["tmp"]
    out = tmp/"eval"
    run, _ = syn.write_bayesreg_run(tmp/"run", P01, syn.d_to_u(syn.smooth_d()))
    (out/"run").mkdir()
    (out/"run"/"metrics.json").write_text("{}")
    assert cli.main([str(run), *common(world, "--out", "metrics.json", "--long", str(tmp/"l.csv"),
                                       "--out-dir", str(out))]) == 2
    assert (out/"run"/"metrics.json").read_text() == "{}" and not (tmp/"l.csv").exists()
    (tmp/"l.csv").write_text("x")
    assert cli.main([str(run), *common(world, "--out", "m2.json", "--long", str(tmp/"l.csv"), "--out-dir", str(out))]) == 2
    assert not (out/"run"/"m2.json").exists()
    assert cli.main([str(run), *common(world, "--out", "sub/m.json", "--long", str(tmp/"l2.csv"),
                                       "--out-dir", str(out))]) == 2
    # review EV item 5: the parents of --long / --summary must exist (checked in phase 1, nothing written)
    assert cli.main([str(run), *common(world, "--out", "m4.json", "--long", str(tmp/"nodir"/"l.csv"),
                                       "--out-dir", str(out))]) == 2
    assert cli.main([str(run), *common(world, "--out", "m4.json", "--long", str(tmp/"l4.csv"), "--summary",
                                       str(tmp/"nodir"/"s.json"), "--out-dir", str(out))]) == 2
    assert not (out/"run"/"m4.json").exists() and not (tmp/"l4.csv").exists()
    monkeypatch.setattr(cli, "READ_ONLY_ROOTS", (tmp.resolve(),))
    assert cli.main([str(run), *common(world, "--out", "m3.json", "--long", str(tmp/"x"/"l3.csv"),
                                       "--out-dir", str(out))]) == 2
    other = tmp.parent/(tmp.name+"_out")
    assert cli.main([str(run), *common(world, "--out", "m3.json", "--long", str(other/"l3.csv"),
                                       "--out-dir", str(out))]) == 2           # --out-dir under the read-only tree
    other.mkdir()
    assert cli.main([str(run), *common(world, "--out", "m3.json", "--long", str(other/"l3.csv"),
                                       "--out-dir", str(other/"eval"))]) == 0
    assert (other/"eval"/"run"/"m3.json").is_file() and not (run/"m3.json").exists()


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "..", ".hidden", "x y"])
def test_run_id_must_be_a_plain_name(world, run_id):
    """Review EV item 5: --run-id cannot leave --out-dir."""
    tmp = world["tmp"]
    run, _ = syn.write_bayesreg_run(tmp/"run", P01, syn.d_to_u(syn.smooth_d()))
    assert cli.main([str(run), *common(world, "--out", "m.json", "--long", str(tmp/"l.csv"), "--out-dir",
                                       str(tmp/"eval"), "--run-id", run_id)]) == 2
    assert not any((tmp/"eval").iterdir()) and not (tmp/"l.csv").exists() and not (tmp/"m.json").exists()


def test_run_id_from_the_run_directory(world):
    """Review EV item 7: a configuration C run's own run_id file names its output."""
    tmp = world["tmp"]
    run = syn.write_configc_run(tmp/P01, P01, syn.d_to_u(syn.smooth_d()), run_id="d0v1_A0_0901-0902_x")
    assert cli.main([str(run), *common(world, "--out", "m.json", "--long", str(tmp/"l.csv"), "--out-dir", str(tmp/"eval"),
                                       "--arm", "A0", "--cert", "none")]) == 0
    rec = json.loads((tmp/"eval"/"d0v1_A0_0901-0902_x"/"m.json").read_text())
    assert (rec["run_id"], rec["arm"]) == ("d0v1_A0_0901-0902_x", "A0") and not (tmp/"eval"/P01).exists()


def test_input_record_mismatch_is_refused(world):
    """Review EV item 8: the run's own input record must match the admitted pair."""
    tmp = world["tmp"]
    u = syn.d_to_u(syn.smooth_d())
    wrong_sha = syn.write_configc_run(tmp/"c", P01, u, input_sha256="f"*64)
    wrong_stage = syn.write_configc_run(tmp/"s", P01, u, input_sha256=pair_sha(world, P01), pair_dataset="stage_other_v1")
    wrong_subject, _ = syn.write_bayesreg_run(tmp/"b", P01, u, pair_provenance=dict(
        pair_sha256=pair_sha(world, P01), fixed_subject=syn.DEV[2], moving_subject=syn.DEV[1]))
    for run in (wrong_sha, wrong_stage, wrong_subject):
        assert cli.main([str(run), *common(world, "--out", "m.json", "--long", str(tmp/"l.csv"),
                                           "--out-dir", str(tmp/"eval"))]) == 2
    assert not any((tmp/"eval").iterdir()) and not (tmp/"l.csv").exists()


def test_stage_mismatch(world):
    tmp = world["tmp"]
    run, _ = syn.write_bayesreg_run(tmp/"run", P01, syn.d_to_u(syn.smooth_d()), stage="stage_other_v1")
    assert cli.main([str(run), *common(world, "--out", "m.json", "--long", str(tmp/"l.csv"),
                                       "--out-dir", str(tmp/"eval"))]) == 2


# ---- non-finite values in long.csv (review EV item 3) ---------------------------------------------------------------
def test_long_csv_keeps_nonfinite_values(tmp_path):
    base = dict(set="s", split="dev", method="m", arm="", pair=P01, run_id="r", field_role="native_formal",
                run_status="converged_B", field_present=True, error=None, truth=None, per_label=None, certificate=None)
    record = dict(base, metrics=dict(hd95=math.inf, dice_official=math.nan, seg35=.8125, sdlogj_brain=-math.inf,
                                     central_det_nonpositive=3, sdlogj_official=None))
    rows = cli.long_rows(record)
    assert {r["measure"] for r in rows} == {"field_present", "hd95", "dice_official", "seg35", "sdlogj_brain",
                                            "central_det_nonpositive"}
    P.write_long(tmp_path/"l.csv", rows)
    text = (tmp_path/"l.csv").read_text()
    assert ",inf," in text and ",nan," in text and ",-inf," in text and ",0.8125," in text
    back = {r["measure"]: r["value"] for r in P.read_long(tmp_path/"l.csv")}
    assert back["hd95"] == math.inf and math.isnan(back["dice_official"]) and back["sdlogj_brain"] == -math.inf
    assert back["seg35"] == .8125 and back["central_det_nonpositive"] == 3.
    assert cli.nonfinite_metrics(record["metrics"]) == dict(hd95="inf", dice_official="nan", sdlogj_brain="-inf")
    summary = cli.summarize([record, dict(record, run_id="r2", metrics=dict(hd95=2., seg35=.5))], [None, None])[0]
    assert summary["means_over_finite"]["hd95"] == 2. and summary["value_counts"]["hd95"] == dict(
        n=1, n_nonfinite=1, nonfinite={"inf": 1})
    assert summary["means_over_finite"]["dice_official"] is None
    with pytest.raises(FileExistsError):
        P.write_long(tmp_path/"l.csv", rows)


# ---- LPBA40 end to end (review EV item 12) --------------------------------------------------------------------------
def test_end_to_end_lpba40_disp(tmp_path):
    data = tmp_path/"data"
    pair = "LPBA40_S01__LPBA40_S02"
    syn.write_pair(data, pair, "S01", "S02", syn.lpba40_pair_arrays(seed=3),
                   label_sets=dict(lpba56=list(syn.LPBA56)))
    allowlist = syn.write_allowlist(tmp_path/"dev.json", lpba40=syn.DEV_LPBA40)
    (tmp_path/"base").mkdir()
    np.savez(tmp_path/"base"/"disp.npz", d=syn.smooth_d(amplitude=(.4, .3, .2)))
    code = cli.main([str(tmp_path/"base"), "--pairs-stage", syn.STAGE, "--data-root", str(data), "--allowlist",
                     str(allowlist), "--pair", pair, "--method", "baseline_x", "--out", "m.json", "--long",
                     str(tmp_path/"l.csv"), "--out-dir", str(tmp_path/"eval"), "--summary", str(tmp_path/"s.json")])
    assert code == 0
    rec = json.loads((tmp_path/"eval"/"base"/"m.json").read_text())
    assert rec["truth"]["scheme"] == "lpba40" and rec["truth"]["keys_read"] == [
        "fixed_lpba56", "moving_lpba56", "fixed_brain"]
    assert 0 < rec["metrics"]["lpba54"] <= 1 and 0 < rec["metrics"]["lpba56"] <= 1        # both averages reported
    l56, l54 = set(rec["per_label"]["lpba56_per_label"]), set(rec["per_label"]["lpba54_per_label"])
    assert l54 == l56-{"181", "182"} and l56 <= {str(i) for i in syn.LPBA56} and len(l54) > 40
    measures = {r["measure"] for r in read_long(tmp_path/"l.csv")}
    assert {"lpba56", "lpba54", "dice_official", "hd95", "cert_jacobian_only_passed"} <= measures
    assert any(m.startswith("lpba56_label_") for m in measures)
    assert not any(m.startswith("lpba54_label") for m in measures)          # per-label rows once per stem


# ---- the guard ------------------------------------------------------------------------------------------------------
def _no_npz(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("an npz file was opened")
    monkeypatch.setattr(np, "load", boom)


def _resolve(subject):
    """'<name>' -> the smallest OASIS id of that sealed list (read at run time; never written in this file)."""
    return syn.sealed_id(subject[1:-1]) if subject.startswith("<") else subject


def _sealed_case(world, monkeypatch, fixed, moving, *, allowlist_subjects=None, stage=syn.STAGE, write_manifest=True,
                 token=None, pair=None):
    tmp = world["tmp"]
    fixed, moving = _resolve(fixed), _resolve(moving)
    if allowlist_subjects is not None:
        allowlist_subjects = [_resolve(s) for s in allowlist_subjects]
    if pair is None:
        pair = f"{fixed}__{moving}" if fixed.startswith("OASIS") else f"LPBA40_{fixed}__LPBA40_{moving}"
    if write_manifest:
        syn.write_pair(world["data"], pair, fixed, moving, stage=stage)
    allowlist = world["allowlist"]
    if allowlist_subjects is not None:
        allowlist = syn.write_allowlist(tmp/"bad_allowlist.json", allowlist_subjects)
    good, _ = syn.write_bayesreg_run(tmp/"good", P01, syn.d_to_u(syn.smooth_d()))
    (tmp/"sealed").mkdir()
    np.savez(tmp/"sealed"/"disp.npz", d=syn.smooth_d())
    _no_npz(monkeypatch)
    args = [str(good), str(tmp/"sealed"), "--pairs-stage", stage, "--data-root", str(world["data"]), "--allowlist",
            str(allowlist), "--out", "metrics.json", "--long", str(tmp/"l.csv"), "--method", "x",
            "--out-dir", str(tmp/"eval")]
    if token:
        args += ["--unseal-token", token]
    # one disp directory per call needs --pair; here the sealed run is described by a bayesreg-style config instead
    (tmp/"sealed"/"config.json").write_text(json.dumps(dict(args=dict(pair=f"{stage}/{pair}"))))
    (tmp/"sealed"/"cmd.txt").write_text("x")
    (tmp/"sealed"/"status.json").write_text(json.dumps(dict(status="converged_B", exit_code=0, native=True)))
    return args, tmp


@pytest.mark.parametrize("fixed, moving, allowlist_subjects, write_manifest, pair", [
    (syn.DEV[0], "OASIS_OAS1_0905_MR1", None, True, None),                   # not on the allowlist
    (syn.DEV[0], "OASIS_OAS1_0905_MR1", None, True, "neutral_a"),            # the same; the name says nothing
    (syn.DEV[0], "<holdout_B_v1>", list(syn.DEV)+["<holdout_B_v1>"], True, "neutral_b"),   # B hold-out (manifest),
    #                                                                          even on a wrong allowlist
    (syn.DEV[0], "<holdout_B_v1>", None, False, None),                       # refused by name; manifest never read
    ("<H1>", syn.DEV[1], list(syn.DEV)+["<H1>"], True, "neutral_c"),         # H1
    ("<H4>", syn.DEV[1], list(syn.DEV)+["<H4>"], True, "neutral_d"),         # H4
    ("S01", "S37", None, True, None),                                        # H2 (LPBA40 test split), by name
    ("S01", "S37", None, True, "neutral_e"),                                 # H2, from the manifest
])
def test_sealed_subjects_stop_everything(world, monkeypatch, capsys, fixed, moving, allowlist_subjects, write_manifest,
                                         pair):
    args, tmp = _sealed_case(world, monkeypatch, fixed, moving, allowlist_subjects=allowlist_subjects,
                             write_manifest=write_manifest, pair=pair)
    assert cli.main(args) == 3
    assert "subject_sealed" in capsys.readouterr().err
    assert not any((tmp/"eval").iterdir()) and not (tmp/"good"/"metrics.json").exists()
    assert not (tmp/"l.csv").exists()


def test_unseal_token_is_recorded_but_admits_nothing(world, monkeypatch, capsys):
    args, tmp = _sealed_case(world, monkeypatch, syn.DEV[0], "<holdout_B_v1>",
                             allowlist_subjects=list(syn.DEV)+["<holdout_B_v1>"], token="BZ-token", pair="neutral")
    assert cli.main(args) == 3
    assert "unsealing is not implemented" in capsys.readouterr().err
    assert not (tmp/"l.csv").exists()


def test_hold_out_stages_and_ibsr_are_refused_by_name(world, monkeypatch):
    sealed = T.load_sealed_lists()
    for stage, pair in (("stage_H1_v1", P01), ("stage_H4_v1", P01), ("stage_IBSR18_v1", "IBSR_01__IBSR_02"),
                        (syn.STAGE, "IBSR_01__IBSR_06")):
        with pytest.raises(T.SubjectSealed):
            T.check_name(stage, pair, sealed)
    with pytest.raises(ValueError):
        T.check_name(syn.STAGE, "../x", sealed)
    # review EV item 6: H3 is pinned, and any id mentioning IBSR is H3 whatever its form
    assert all(sealed.which(s) == "H3" for s in sealed.subjects["H3"])
    assert sealed.which("IBSR18_IBSR_07") == "H3" and sealed.which("ibsr_07") == "H3"


def test_admit_then_truth(world):
    sealed = T.load_sealed_lists()
    allowlist = load_dev_allowlist(world["allowlist"])
    pair_dir, grid, _, guard = T.admit_pair(world["data"], syn.STAGE, P01, allowlist, sealed)
    assert grid.shape == syn.SHAPE and guard["fixed_subject"] == syn.DEV[0] and guard["unsealed"] is False
    truth, record = T.load_truth(pair_dir, T.scheme_for(guard, pair_dir), grid, guard)
    assert set(truth) == {"fixed_seg35", "moving_seg35", "fixed_seg4", "moving_seg4", "brain"}
    assert record["pair_sha256"] == guard["pair_sha256_manifest"]
    with pytest.raises(ValueError):
        T.load_truth(pair_dir, M.OASIS, grid, {})
    (pair_dir/"pair.npz").write_bytes((pair_dir/"pair.npz").read_bytes()+b"x")
    with pytest.raises(ValueError, match="sha256"):
        T.load_truth(pair_dir, M.OASIS, grid, guard)


def test_load_truth_does_not_trust_a_handmade_guard(world, monkeypatch):
    """Review EV item 4: a guard dict for another pair, a sealed subject or another allowlist is refused before
    pair.npz is opened."""
    sealed = T.load_sealed_lists()
    allowlist = load_dev_allowlist(world["allowlist"])
    pair_dir, grid, _, guard = T.admit_pair(world["data"], syn.STAGE, P01, allowlist, sealed)
    other_dir, _, _, other = T.admit_pair(world["data"], syn.STAGE, P23, allowlist, sealed)
    holdout = syn.sealed_id("holdout_B_v1")
    neutral = syn.write_pair(world["data"], "neutral_x", syn.DEV[0], holdout)
    _no_npz(monkeypatch)
    with pytest.raises(ValueError, match="manifest"):
        T.load_truth(pair_dir, M.OASIS, grid, other)                       # the record of another pair
    with pytest.raises(ValueError):
        T.load_truth(pair_dir, M.OASIS, grid, dict(guard, unsealed=None))
    with pytest.raises(ValueError, match="guard record without"):
        T.load_truth(pair_dir, M.OASIS, grid, dict(unsealed=False))
    forged = dict(guard, manifest_sha256=file_sha256(neutral/"manifest.json"), moving_subject=holdout,
                  pair_sha256_manifest=json.loads((neutral/"manifest.json").read_text())["pair_sha256"])
    with pytest.raises(SubjectSealed):
        T.load_truth(neutral, M.OASIS, grid, forged)                       # a sealed subject in a forged record
    wide = syn.write_allowlist(world["tmp"]/"wide.json", list(syn.DEV)+["OASIS_OAS1_0905_MR1"])
    with pytest.raises(ValueError, match="allowlist"):
        T.load_truth(pair_dir, M.OASIS, grid, dict(guard, allowlist_path=str(wide)))
    assert other_dir != pair_dir


def test_real_sealed_lists_and_allowlist_are_disjoint():
    """The committed allowlist (configs/bayesreg/dev_subjects.json) holds no subject of a sealed list."""
    sealed = T.load_sealed_lists()
    allowlist = load_dev_allowlist(T.DEFAULT_ALLOWLIST)
    listed = set().union(*allowlist.subjects.values())
    assert all(sealed.which(s) is None for s in listed)
    assert {r["name"]: r["subjects"] for r in sealed.record} == {"holdout_B_v1": 100, "H1": 100, "H4": 40, "H3": 18,
                                                                 "H2": 4}


def test_sealed_list_pins(tmp_path):
    with pytest.raises(T.SealedListError):
        T.load_sealed_lists(tmp_path)
    for spec in T.SEALED_LISTS:
        target = tmp_path/spec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((Path(T.REPO)/spec["path"]).read_bytes())
    holdout = syn.sealed_id("holdout_B_v1")
    assert T.load_sealed_lists(tmp_path).which(holdout) == "holdout_B_v1"
    target = tmp_path/T.SEALED_LISTS[0]["path"]
    target.write_text(target.read_text()+"OASIS_OAS1_0999_MR1\n")
    with pytest.raises(T.SealedListError, match="sha256"):
        T.load_sealed_lists(tmp_path)
