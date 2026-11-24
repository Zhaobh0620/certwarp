"""R3b's calibration tools (experiments/r3/r3b_*.py; runs/20261111-r3/calibration_plan.md) on the synthetic world of r3b_synthetic: CPU, tmp_path, tiny members.

No real subject, image, label, member or results root is opened: the world's ids are OASIS_OAS1_09xx_MR1, its roots are under tmp_path; the plan, the code and the
sealed lists the guard reads are the repository's.  No unit, no GPU.
"""
import dataclasses
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT/"experiments"/"r3") not in sys.path:
    sys.path.insert(0, str(ROOT/"experiments"/"r3"))
import r3b_calib as C  # noqa: E402
import r3b_common as rc  # noqa: E402
import r3b_decision as RD  # noqa: E402
import r3b_firewall as fw  # noqa: E402
import r3b_synthetic as S  # noqa: E402
from bayesreg.io import sealed  # noqa: E402
from bayesreg.seg import decode as D  # noqa: E402

QUIET = dict(log=lambda *_: None)


def _fresh():
    sealed._reset_cache()                 # each step is its own process in production; the spy wants the guard's first read under it


def _run_to_eval(ctx):
    _fresh()
    C.infer(ctx, device="cpu", **QUIET)
    _fresh()
    C.decode_step(ctx, **QUIET)
    return rc.sha256_file(Path(ctx.out)/rc.OPENED_DECODE)


@pytest.fixture
def world(tmp_path):
    return S.build(tmp_path/"w")


def _opened(path):
    return json.loads(Path(path).read_text())


def _paths(record):
    return [Path(f["path"]) for f in record["files"]]


def test_end_to_end_synthetic_and_what_each_unit_opened(world):
    sha = _run_to_eval(world)
    _fresh()
    document = C.evaluate(world, opened_files_sha256=sha, **QUIET)
    decision = C.decide_step(world, **QUIET)
    assert decision["verdict"] in ("pass", "not_pass") and decision["checks"] is not None
    assert document["subjects"] == world.subjects() and set(document["rows"]) == {"argmax", "decode", "ratio"}
    out = Path(world.out)
    infer_opened, decode_opened, eval_opened = (_opened(out/rc.OPENED_INFER), _opened(out/rc.OPENED_DECODE), _opened(out/rc.EVAL_DIR/rc.OPENED_EVAL))
    assert infer_opened["m6"] and decode_opened["m6"] and eval_opened["m6"]
    images = {Path(world.images).resolve()/s/"aligned_norm.nii.gz" for s in world.subjects()}
    assert images <= set(_paths(infer_opened))
    for record in (infer_opened, decode_opened):                     # no truth before eval
        assert not any("seg" in p.name and p.name.endswith(".nii.gz") for p in _paths(record))
    labels = {p for p in _paths(eval_opened) if p.name.endswith(".nii.gz") and "seg" in p.name}
    assert labels == {Path(world.images).resolve()/s/n for s in world.subjects() for n in ("aligned_seg35.nii.gz", "aligned_seg4.nii.gz")}
    others = {Path(world.images).resolve()/s for s in ("OASIS_OAS1_0901_MR1", "OASIS_OAS1_0907_MR1")}
    assert not any(p.parent in others for p in _paths(eval_opened))  # training subjects outside the calibration set and development subjects: nothing
    lock = [json.loads(line) for line in Path(world.lock).read_text().splitlines()]
    assert len(lock) == 1 and lock[0]["event"] == "first_read" and lock[0]["opened_files_decode_sha256"] == sha
    loaded = RD.load_calibration_decision(out/rc.ANALYSIS_DIR/rc.DECISION_FILE, allow_synthetic=True, require_pass=False)
    assert loaded["verdict"] == decision["verdict"] and RD.decode_kwargs(loaded) == dict(labels=rc.DECODE_LABELS, objective="expected_dice", candidate_floor=1e-3)
    here = rc.sha256_file(rc.decode_py_path())                       # review I1: the decode unit's decode.py, carried into the evaluation and the decision
    decode_record = json.loads((out/rc.DECODE_RECORD).read_text())
    assert loaded["decode_py_sha256"] == here == document["decode_py_sha256"] == decision["decode_py_sha256"] == decode_record["decode_py_sha256"]   # the key R3a reads
    assert document["implementation"]["decode_py_sha256"] == here and json.loads((out/rc.EVAL_DIR/rc.EVAL_FILE).read_text())["decode_py_sha256"] == here
    assert set(document["volumes"][world.subjects()[0]]) == {str(c) for c in rc.TARGET_LABELS}          # plan revision 2 item 8
    assert document["volume_bias"]["15"]["role"].startswith("other target") and document["volume_bias"]["4"]["role"].startswith("small six")
    decode_record = json.loads((out/rc.DECODE_RECORD).read_text())
    for subject in world.subjects():                                 # the decoded maps are bayesreg.seg.decode's on the stored read-back
        probs, _ = C._read_probs(world, subject, json.loads((out/rc.INFER_RECORD).read_text())["rows"][subject])
        with np.load(out/rc.DECODED_DIR/f"{subject}.npz") as npz:
            assert np.array_equal(npz["decode"], D.decode(probs)) and np.array_equal(npz["argmax"], np.argmax(probs, axis=0))
        assert decode_record["rows"][subject]["decode"]["decoder_id"] == D.DECODER_ID
    _fresh()
    with pytest.raises(C.Refused, match="read for R3b already"):         # read once
        C.evaluate(world, opened_files_sha256=sha, **QUIET)


def test_eval_without_the_released_opened_files_reads_no_truth(world):
    _run_to_eval(world)
    _fresh()
    with pytest.raises(C.Refused, match="opened-files-sha256"):
        C.evaluate(world, opened_files_sha256="0"*64, **QUIET)
    assert not Path(world.lock).exists() and not (Path(world.out)/rc.EVAL_DIR).exists()


def test_eval_refuses_a_spy_record_with_a_breach(world):
    _run_to_eval(world)
    path = Path(world.out)/rc.OPENED_INFER
    record = _opened(path)
    record["m6"] = False
    path.write_text(json.dumps(record))
    _fresh()
    with pytest.raises(C.Refused, match="m6"):
        C.evaluate(world, opened_files_sha256=rc.sha256_file(Path(world.out)/rc.OPENED_DECODE), **QUIET)
    assert not Path(world.lock).exists()


def test_steps_never_overwrite(world):
    _run_to_eval(world)
    _fresh()
    with pytest.raises(FileExistsError):
        C.infer(world, device="cpu", **QUIET)
    _fresh()
    with pytest.raises(FileExistsError):
        C.decode_step(world, **QUIET)


def test_a_member_that_trained_on_a_calibration_subject_is_refused_before_any_image(world):
    directory = Path(world.models)/"foldA_s1"
    record = json.loads((directory/"train_record.json").read_text())
    record["train_subjects"] = record["train_subjects"]+list(world.calib["A"])
    (directory/"train_record.json").write_text(json.dumps(record))
    member = json.loads((directory/"member.json").read_text())
    member["train_record_sha256"] = rc.sha256_file(directory/"train_record.json")
    (directory/"member.json").write_text(json.dumps(member))
    _fresh()
    with pytest.raises(C.Refused, match="not out of fold"):
        C.infer(world, device="cpu", **QUIET)
    opened = _opened(Path(world.out)/rc.OPENED_INFER)
    assert not any(p.name == "aligned_norm.nii.gz" for p in _paths(opened))


def test_registry_members_that_are_not_the_plans_are_refused(world):
    world.member_sha["B"] = ("1"*64, world.member_sha["B"][1])
    _fresh()
    with pytest.raises(C.Refused, match="registry's members"):
        C.infer(world, device="cpu", **QUIET)


def test_the_rules_of_each_unit(world):
    out, images = Path(world.out), Path(world.images)
    subject, other = world.subjects()[0], "OASIS_OAS1_0901_MR1"
    truth = [images/subject/"aligned_seg35.nii.gz"]
    common = dict(repo=world.repo, results=world.results, data=world.data_link, images=images, out=out, lock=world.lock)
    infer = fw.CalibRules(unit="infer", files=[images/subject/"aligned_norm.nii.gz"], own_dirs=[out/rc.PROBS_DIR], **common)
    evaluate = fw.CalibRules(unit="eval", files=[], own_dirs=[out/rc.EVAL_DIR], truth=truth, **common)
    judge = lambda rules, p: rules.judge(Path(p), Path(p).resolve())    # noqa: E731
    assert judge(infer, images/subject/"aligned_norm.nii.gz") is None
    assert "label file" in judge(infer, images/subject/"aligned_seg35.nii.gz")
    assert "outside" in judge(infer, images/other/"aligned_norm.nii.gz")
    assert "evaluation or analysis" in judge(infer, out/rc.EVAL_DIR/rc.EVAL_FILE)
    assert "read lock" in judge(infer, world.lock)
    assert "evaluation record" in judge(infer, Path(world.results)/"x"/"metrics_0600.json")
    assert "read lock" in judge(infer, Path(world.results)/"seg"/"uf_dev_read.lock")
    assert judge(infer, out/rc.PROBS_DIR/subject/"probs36.npz") is None
    assert judge(evaluate, truth[0]) is None and judge(evaluate, world.lock) is None and judge(evaluate, out/rc.EVAL_DIR/rc.EVAL_FILE) is None
    assert "label file" in judge(evaluate, images/subject/"aligned_seg4.nii.gz")
    assert "label file" in judge(evaluate, images/other/"aligned_seg35.nii.gz")
    assert "result document" in judge(evaluate, Path(world.repo)/"runs"/"20261110-r2"/"result.md")
    plan = judge(evaluate, Path(world.repo)/rc.PLAN_REL)                # the pinned plan is exempt from the result-document rule but is read before the spy:
    assert plan.startswith("outside") and "forbidden" not in plan     # not allowed inside it, and not forbidden


def _document(deltas_s6, deltas_35, deltas_4, n=16):
    subjects = [f"s{i}" for i in range(n)]
    paired = {k: dict(all=C.paired_summary(v)) for k, v in (("small_six", deltas_s6), ("seg35", deltas_35), ("seg4", deltas_4))}
    implementation = rc.implementation_record()
    return dict(subjects=subjects, paired=dict(decode=paired), decoder_spec=json.loads(json.dumps(rc.DECODE_SPEC)), plan=dict(sha256="x"),
                implementation=implementation, decode_py_sha256=implementation["decode_py_sha256"])


def test_decide_bounds_are_inclusive_and_written():
    s6 = [.005]*10+[.005]*6
    base = _document(s6, [0.]*16, [-.0005]*16)
    decision = C.decide(base)
    assert decision["verdict"] == "pass", decision
    assert C.decide(_document([.005]*9+[0.]*7, [0.]*16, [0.]*16))["verdict"] == "not_pass"                      # 9/16 positive (and mean below)
    assert C.decide(_document([.009]*9+[0.]*7, [0.]*16, [0.]*16))["checks"]["small6_positive"]["ok"] is False   # mean .0051 but 9 positive
    assert C.decide(_document(s6, [-1e-9]+[0.]*15, [0.]*16))["verdict"] == "not_pass"
    assert C.decide(_document(s6, [0.]*16, [-.0005-1e-9]*16))["verdict"] == "not_pass"
    assert C.decide(_document([.0049]*16, [0.]*16, [0.]*16))["verdict"] == "not_pass"
    assert C.decide(_document(s6[:15], [0.]*15, [0.]*15, n=15))["verdict"] == "undecidable"
    assert C.decide(_document([None]*16, [0.]*16, [0.]*16))["verdict"] == "undecidable"
    assert C.decide(_document([.01]*10+[0.]*6, [0.]*16, [0.]*16))["verdict"] == "pass"                       # review m12: exactly 10/16 positive passes
    assert C.decide(_document([.0099]*9+[0.]*6+[.0011], [0.]*16, [0.]*16))["checks"]["small6_positive"]["value"] == 10
    one_undefined = C.decide(_document([.01]*15+[None], [0.]*15+[None], [0.]*16))                            # plan revision 2 item 9 (review m11)
    assert one_undefined["verdict"] == "undecidable" and any("undefined for 1" in r for r in one_undefined["reasons"])
    unbound = _document(s6, [0.]*16, [0.]*16)
    unbound["decode_py_sha256"] = None
    assert C.decide(unbound)["verdict"] == "undecidable"                                                      # review I1: no decode.py sha256
    two = _document(s6, [0.]*16, [0.]*16)
    two["decode_py_sha256"] = "0"*64
    assert C.decide(two)["verdict"] == "undecidable"
    wrong = _document(s6, [0.]*16, [0.]*16)
    wrong["decoder_spec"] = dict(wrong["decoder_spec"], objective="ratio")
    assert C.decide(wrong)["verdict"] == "undecidable"


def test_decision_loader_for_r3a(tmp_path, monkeypatch):
    decision = C.decide(_document([.006]*16, [0.]*16, [0.]*16))
    decision["synthetic"] = False
    path = tmp_path/"decision_calib.json"
    path.write_text(json.dumps(decision))
    monkeypatch.setattr(rc, "PLAN_SHA256", "x")
    loaded = RD.load_calibration_decision(path)                                                   # review m3: require_pass by default
    assert loaded["verdict"] == "pass" and loaded["decision_sha256"] == rc.sha256_file(path) and loaded["decode_py_sha256"] == rc.sha256_file(rc.decode_py_path())
    monkeypatch.setattr(rc, "PLAN_SHA256", "y")
    with pytest.raises(ValueError, match="plan sha256"):
        RD.load_calibration_decision(path)
    monkeypatch.setattr(rc, "PLAN_SHA256", "x")
    failing = dict(decision, verdict="not_pass", checks=dict(decision["checks"], seg4_mean=dict(decision["checks"]["seg4_mean"], ok=False)))
    path.write_text(json.dumps(failing))
    with pytest.raises(RD.DecodeNotAdmitted):
        RD.load_calibration_decision(path)                                                        # default: not pass is refused
    assert RD.load_calibration_decision(path, require_pass=False)["verdict"] == "not_pass"        # only a descriptive use passes False
    for bad, words in ((dict(decision, verdict="not_pass"), "not_pass with every check"),
                       (dict(decision, checks=dict(decision["checks"], small6_positive=dict(decision["checks"]["small6_positive"], ok=False))), "pass with a check"),
                       (dict(decision, verdict="undecidable"), "undecidable without reasons"),
                       (dict(decision, criterion=dict(decision["criterion"], small6_mean_min=.001)), "criterion"),
                       (dict(decision, synthetic=True), "synthetic"),
                       (dict(decision, decoder_spec=dict(decision["decoder_spec"], candidate_floor=.01)), "decoder"),
                       (dict(decision, decode_py_sha256=None), "binds no decode.py")):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=words):
            RD.load_calibration_decision(path, require_pass=False)


def test_a_decode_py_one_byte_off_is_refused(tmp_path, monkeypatch):
    """Review I1: the record binds the decoder's file; a copy with one byte changed is not the decoder the reading judged."""
    monkeypatch.setattr(rc, "PLAN_SHA256", "x")
    decision = dict(C.decide(_document([.006]*16, [0.]*16, [0.]*16)), synthetic=False)
    path = tmp_path/"decision_calib.json"
    path.write_text(json.dumps(decision))
    copy = tmp_path/"decode_copy.py"
    raw = bytearray(rc.decode_py_path().read_bytes())
    copy.write_bytes(bytes(raw))
    assert RD.load_calibration_decision(path, decode_path=copy)["decode_py_sha256"] == rc.sha256_file(copy)
    raw[-2] = raw[-2] ^ 1
    copy.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="decode.py here has sha256"):
        RD.load_calibration_decision(path, decode_path=copy)


def test_eval_refuses_a_decode_record_of_another_decode_py(world):
    sha = _run_to_eval(world)
    path = Path(world.out)/rc.DECODE_RECORD
    record = json.loads(path.read_text())
    record["implementation"]["decode_py_sha256"] = "0"*64
    path.write_text(json.dumps(record))
    _fresh()
    with pytest.raises(C.Refused, match="not the file the decode unit ran"):
        C.evaluate(world, opened_files_sha256=sha, **QUIET)
    assert not Path(world.lock).exists()


def test_eval_refuses_an_opened_files_infer_that_is_not_the_recorded_one(world):
    """Review m7: the release chain covers both spy records."""
    sha = _run_to_eval(world)
    path = Path(world.out)/rc.OPENED_INFER
    record = _opened(path)
    record["note"] = "edited"
    path.write_text(json.dumps(record))
    _fresh()
    with pytest.raises(C.Refused, match="opened_files_infer.json is not the one"):
        C.evaluate(world, opened_files_sha256=sha, **QUIET)
    assert not Path(world.lock).exists()


def test_the_lock_stays_after_a_failure_past_it_and_bz_go(world):
    """Review m12: a failure after the lock keeps it; another read only with BZ's words (lock_check / lock_take), never on synthetic data."""
    sha = _run_to_eval(world)
    (Path(world.images)/world.subjects()[-1]/"aligned_seg4.nii.gz").unlink()                    # the last subject's truth cannot be read
    _fresh()
    with pytest.raises(Exception):
        C.evaluate(world, opened_files_sha256=sha, **QUIET)
    lines = Path(world.lock).read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["event"] == "first_read"
    with pytest.raises(C.Refused, match="read for R3b already"):
        C.lock_check(world)
    with pytest.raises(ValueError, match="BZ's words"):
        C.lock_check(world, bz_go="  ")
    C.lock_check(world, bz_go="BZ: 再读一次, 4:30 PM")
    line = C.lock_take(world, dict(out_dir="x"), bz_go="BZ: 再读一次, 4:30 PM")
    assert line["event"] == "bz_approved_reread" and len(Path(world.lock).read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="real data"):
        C.evaluate(world, opened_files_sha256=sha, bz_go="BZ: x", **QUIET)
    fresh = dataclasses.replace(world, lock=Path(world.lock).with_name("none.lock"))
    with pytest.raises(ValueError, match="nothing to unlock"):
        C.lock_check(fresh, bz_go="BZ: x")


def test_infer_behind_a_closed_gate_reads_nothing(world):
    """Review m4: the machine gate before anything is read; no wait."""
    _fresh()
    with pytest.raises(C.Refused, match="machine gate closed"):
        C.infer(world, device="cpu", gate=lambda: ["units active: tag-r3a-chain.service"], **QUIET)
    assert not Path(world.out).exists() and sealed.sealed_cache_is_empty()


def test_the_gate_is_r2s_with_the_c_floor(monkeypatch):
    sys.path.insert(0, str(ROOT/"experiments"/"ul"))
    import run_ul
    monkeypatch.setattr(run_ul, "gate_problems", lambda own_unit, **kw: [] if own_unit == "tag-r3b-infer" else ["wrong unit"])
    usage = lambda free: (lambda path: types.SimpleNamespace(free=free*2**30))                   # noqa: E731
    assert C.gate_problems(disk=usage(41.)) == []
    assert C.gate_problems(disk=usage(39.)) == ["/mnt/c free 39.0 GiB < 40 GiB"]


def test_a_member_json_temperature_unlike_member_pt_is_refused(world):
    """Review m6."""
    directory = Path(world.models)/"foldB_s1"
    member = json.loads((directory/"member.json").read_text())
    member["temperature"] = member["temperature"]+.25
    (directory/"member.json").write_text(json.dumps(member))
    _fresh()
    with pytest.raises(C.Refused, match="temperature"):
        C.infer(world, device="cpu", **QUIET)


def test_members_seed_files_must_be_distinct(world, monkeypatch):
    import shutil
    a0, a1 = Path(world.models)/"foldA_s0", Path(world.models)/"foldA_s1"
    shutil.rmtree(a1)
    shutil.copytree(a0, a1)
    member = json.loads((a1/"member.json").read_text())
    record = json.loads((a1/"train_record.json").read_text())
    record["seed"], member["seed"] = 1, 1
    (a1/"train_record.json").write_text(json.dumps(record))
    member["train_record_sha256"] = rc.sha256_file(a1/"train_record.json")
    (a1/"member.json").write_text(json.dumps(member))
    _fresh()
    with pytest.raises(C.Refused, match="not distinct"):
        C.infer(world, device="cpu", **QUIET)


def test_rehearsal_reads_infers_inputs_only_and_leaves_out_alone(world):
    """Review m5: infer's inputs under infer's rules and spy; no model, image or label; OUT untouched."""
    _fresh()
    result = C.rehearse(world, **QUIET)
    root = Path(result["root"])
    assert result["m6"] and root.parent == Path(world.results)/rc.REHEARSAL_ROOT.name and not Path(world.out).exists()
    paths = _paths(_opened(root/"opened_files.json"))
    assert any(p.name == "member.json" for p in paths) and any(p.name == "folds_full_v2.json" for p in paths)
    assert not any(p.name.endswith(".nii.gz") for p in paths)
    assert json.loads((root/"rehearsal.json").read_text())["error"] is None


def test_real_runs_need_the_pinned_plan(monkeypatch):
    monkeypatch.setattr(rc, "PLAN_SHA256", None)
    with pytest.raises(ValueError, match="no pinned sha256"):
        rc.plan_check(ROOT)
    monkeypatch.setattr(rc, "PLAN_SHA256", "0"*64)
    with pytest.raises(ValueError, match="not the pinned"):
        rc.plan_check(ROOT)


def test_the_plan_constants():
    assert rc.DECODE_SPEC == D.decoder_spec((4, 18, 19, 23, 34, 35), objective="expected_dice", candidate_floor=1e-3)
    assert rc.CRITERION == dict(n_subjects=16, small6_mean_min=.005, small6_positive_min=10, seg35_mean_min=0., seg4_mean_min=-.0005)
    assert len(rc.all_subjects()) == 16 and len(set(rc.all_subjects())) == 16 and not set(rc.CALIB_SUBJECTS["A"]) & set(rc.CALIB_SUBJECTS["B"])
    assert rc.READ_LOCK.parent == rc.RESULTS and rc.OUT == rc.RESULTS/"r3b_calib_v1"
    starts = C.start_strings("/x/code")
    assert all("pytest" not in s for s in starts.values()) and "--unit=tag-r3b-infer" in starts["infer"] and "CUDA_VISIBLE_DEVICES=" in starts["eval"]


def test_the_plan_in_the_tree_is_the_pinned_one():
    path, sha = rc.plan_check(ROOT)
    assert sha == rc.PLAN_SHA256 and path == ROOT/rc.PLAN_REL
