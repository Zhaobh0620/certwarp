"""experiments/m2v1/reeval_refs.py (M2V1: A0 and A4' re-evaluated with bayesreg.eval.cli), on the synthetic stage of
tests/test_bayesreg/eval/_synthetic.py (tmp_path only; no data/, no real run directory):

- the comparison and its Markdown from recorded evaluations (fast);
- slow (-m slow): end to end with child processes -- A0 configuration C run directories through the configC route,
  A4p runs whose own input record is a different (recoloured) pair.npz refused by bayesreg.eval.cli's input check and
  evaluated through the disp_from_final_native route (d bit for bit u_to_d(u), certificate full, derived disp.npz
  deleted afterwards); the pair id check against the original long.csv; bit-for-bit equality reported where the original
  value is the evaluator's own, differences where not; a restart evaluates nothing and leaves the comparison unchanged.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from jd_curl.three_d.nifti import file_sha256

from tests.test_bayesreg.eval import _synthetic as syn

ROOT = Path(__file__).resolve().parents[3]
P01 = f"{syn.DEV[0]}__{syn.DEV[1]}"
P23 = f"{syn.DEV[2]}__{syn.DEV[3]}"
METHOD = "configC_frozen_eb808be_90s"


@pytest.fixture(scope="module")
def rr():
    spec = importlib.util.spec_from_file_location("m2v1_reeval", ROOT/"experiments/m2v1/reeval_refs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_long(path, arm, values):
    with open(path, "w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(("set", "split", "method", "arm", "pair", "measure", "value", "run_id"))
        for pair, (rid, measures) in values.items():
            for key, value in measures.items():
                w.writerow(("L2R_val", "dev", METHOD, arm, pair, key, repr(value), rid))


def world(tmp_path):
    data = tmp_path/"data"
    arrays = {P01: syn.pair_arrays(seed=1), P23: syn.pair_arrays(seed=2)}
    for pair, (f, m) in ((P01, syn.DEV[:2]), (P23, syn.DEV[2:])):
        syn.write_pair(data, pair, f, m, arrays[pair])
    (data/syn.STAGE/"selection.json").write_text(json.dumps(dict(pairs=[dict(pair=P01), dict(pair=P23)])))
    allowlist = syn.write_allowlist(tmp_path/"dev_subjects.json")
    return data, allowlist, arrays


@pytest.mark.slow
def test_reeval_end_to_end(rr, tmp_path):
    data, allowlist, arrays = world(tmp_path)
    u = {P01: syn.d_to_u(syn.smooth_d(amplitude=(.6, -.4, .3))), P23: syn.d_to_u(syn.smooth_d(amplitude=(.3, .5, -.2)))}
    for arm, root in (("A0", tmp_path/"d0"), ("A4p", tmp_path/"a4")):
        for k, pair in enumerate((P01, P23)):
            sha = file_sha256(data/syn.STAGE/pair/"pair.npz") if arm == "A0" else "f"*64      # A4p: recoloured input
            syn.write_configc_run(root/"runs"/arm/pair, pair, u[pair], arrays[pair], input_sha256=sha,
                                  pair_dataset=syn.STAGE, run_id=f"r_{arm}_{k}")
    # original values: A0 0901-0902 the evaluator's own numbers (bit-for-bit expected), the rest made up
    from bayesreg.eval import cli
    probe = tmp_path/"probe"
    probe.mkdir()
    assert cli.main([str(tmp_path/"d0"/"runs"/"A0"/P01), "--pairs-stage", syn.STAGE, "--out", "m.json", "--long",
                     str(probe/"l.csv"), "--out-dir", str(probe), "--data-root", str(data), "--allowlist",
                     str(allowlist), "--run-id", "probe"]) == 0
    own = json.loads((probe/"probe"/"m.json").read_text())["metrics"]
    keys = ("seg35", "seg4", "hd95", "sdlogj_official", "sdlogj_brain")
    write_long(tmp_path/"d0.csv", "A0", {P01: ("r_A0_0", {k: own[k] for k in keys}),
                                         P23: ("r_A0_1", {k: .5 for k in keys})})
    write_long(tmp_path/"a4.csv", "A4p", {P01: ("r_A4p_0", {k: .5 for k in keys}), P23: ("r_A4p_1", {k: .5 for k in keys})})
    out = tmp_path/"ref_reeval"
    args = ["--out", str(out), "--code", str(ROOT), "--d0-root", str(tmp_path/"d0"), "--a4-root", str(tmp_path/"a4"),
            "--d0-long", str(tmp_path/"d0.csv"), "--a4-long", str(tmp_path/"a4.csv"), "--data-root", str(data),
            "--stage", syn.STAGE, "--allowlist", str(allowlist), "--min-mem-available-mib", "1", "--numba-threads", "1"]
    assert rr.main(args) == 0, (out/"reeval.log").read_text()[-3000:]
    compare = json.loads((out/"compare.json").read_text())
    rows = {(r["arm"], r["pair"]): r for r in compare["rows"]}
    assert all(x["equal_bitwise"] for x in rows[("A0", P01)]["measures"].values())
    assert not any(x["equal_bitwise"] for x in rows[("A0", P23)]["measures"].values())
    assert rows[("A0", P01)]["field_role"] == "native_formal" and rows[("A4p", P01)]["field_role"] == "disp_only"
    a4 = rows[("A4p", P23)]
    assert a4["route"] == "disp_from_final_native" and a4["evaluated_d_sha256_matches_derived"] is True
    assert a4["input_record_pair_sha256"] == "f"*64 and a4["stage_pair_sha256"] != "f"*64
    assert a4["derived"]["deleted_after_evaluation"] and not Path(a4["derived"]["disp_npz"]).exists()
    assert a4["certificate"]["reeval_status"] == "certified_feasible" and a4["pair_in_record"] == P23
    assert a4["run_id_in_record"] == "r_A4p_1"
    # the A4p field is the run's u: same metrics as the configC route of the same u (A0 run of the same pair)
    assert rows[("A4p", P01)]["measures"]["seg35"]["reeval"] == own["seg35"]
    done = json.loads((out/"A4p"/"0903-0904"/"done.json").read_text())
    assert done["derived"]["field_role"] == "native_formal" and done["script"]["sha256"]
    md = (out/"compare.md").read_text()
    assert "逐位相同" in md and "disp_from_final_native" in md
    # the A4p run itself is refused by the evaluator's input check (the reason for the route)
    refused = tmp_path/"refused"
    refused.mkdir()
    assert cli.main([str(tmp_path/"a4"/"runs"/"A4p"/P01), "--pairs-stage", syn.STAGE, "--out", "m.json", "--long",
                     str(refused/"l.csv"), "--out-dir", str(refused), "--data-root", str(data), "--allowlist",
                     str(allowlist)]) == 2
    # restart: nothing evaluated again, comparison unchanged
    assert rr.main(args) == 0
    assert "(unchanged)" in (out/"reeval.log").read_text().splitlines()[-1]
    # a run id that does not match the original long.csv stops before anything
    write_long(tmp_path/"bad.csv", "A0", {P01: ("other", {k: .5 for k in keys}), P23: ("r_A0_1", {k: .5 for k in keys})})
    bad = args[:]
    bad[bad.index("--out")+1], bad[bad.index("--d0-long")+1] = str(tmp_path/"bad_out"), str(tmp_path/"bad.csv")
    assert rr.main(bad+["--arms", "A0"]) == 8
    assert "differs from the original long.csv" in (tmp_path/"bad_out"/"reeval.log").read_text()


def test_compare_and_markdown_from_records(rr, tmp_path):
    data, allowlist, _ = world(tmp_path)
    out = tmp_path/"ref_reeval"
    keys = ("seg35", "seg4", "hd95", "sdlogj_official", "sdlogj_brain")
    for arm in ("A0", "A4p"):
        for k, pair in enumerate((P01, P23)):
            ev = out/arm/rr.summarize.short_name(pair)
            (ev/f"r_{arm}_{k}").mkdir(parents=True)
            (ev/f"r_{arm}_{k}"/"metrics.json").write_text(json.dumps(dict(
                pair=pair, run_id=f"r_{arm}_{k}", field_role="native_formal" if arm == "A0" else "disp_only",
                metrics={key: .5+.1*k for key in keys}, certificate=dict(status="certified_feasible", passed=True,
                                                                         min_jd_lower_bound=.1))))
            (ev/"done.json").write_text(json.dumps(dict(metrics=[str(ev/f"r_{arm}_{k}"/"metrics.json")])))
    originals = {arm: dict(path="x", sha256="y", pairs={P01: dict(run_id=f"r_{arm}_0", **{key: .5 for key in keys}),
                                                        P23: dict(run_id=f"r_{arm}_1", **{key: .7 for key in keys})})
                 for arm in ("A0", "A4p")}
    job = rr.Reeval(rr.parse_args(["--out", str(out), "--code", str(ROOT), "--data-root", str(data), "--stage",
                                   syn.STAGE]))
    job.arms, job.code, job.code_sha = ["A0", "A4p"], ROOT, "sha"
    path, state = job.compare([P01, P23], originals)
    doc = json.loads(path.read_text())
    assert state == "written" and doc["counts"] == dict(compared=20, bitwise_equal=10, differ=10, missing=0)
    row = next(r for r in doc["rows"] if r["arm"] == "A0" and r["pair"] == P23)
    assert row["measures"]["seg35"]["difference"] == pytest.approx(.6-.7)
    assert "逐位相同" in (out/"compare.md").read_text()
    assert job.compare([P01, P23], originals) == (path, "unchanged")
    np.testing.assert_equal(sorted(p.name for p in out.glob("compare*")), ["compare.json", "compare.md"])
