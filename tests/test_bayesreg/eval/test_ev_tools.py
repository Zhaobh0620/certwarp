"""experiments/ev/truth_gate.py (C23.12-10) and experiments/ev/entry_control.py (preregistration 5.1 item 8): pins, admission, the lock, and both readings on
synthetic development-like data (tmp_path; synthetic ids OASIS_OAS1_09xx_MR1).  Neither tool is run on real data here."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.nifti import file_sha256, save_scalar

from bayesreg.eval import holdout as H
from bayesreg.io.guard import SubjectSealed

from .. import _p0_pins as P0
from . import _synthetic as syn

EV = H.REPO/"experiments"/"ev"
if str(EV) not in sys.path:
    sys.path.insert(0, str(EV))
import entry_control as EC  # noqa: E402
import ev_common as ev  # noqa: E402
import truth_gate as TG  # noqa: E402

IDS = ("OASIS_OAS1_0901_MR1", "OASIS_OAS1_0902_MR1", "OASIS_OAS1_0903_MR1")
PAIRS = ((IDS[0], IDS[1]), (IDS[1], IDS[2]))
GRID = make_grid_3d(nx=syn.SHAPE[2], ny=syn.SHAPE[1], nz=syn.SHAPE[0], hx=1., hy=1., hz=1., unit="mm")


def no_units(argv, **kwargs):
    return subprocess.CompletedProcess(argv, 0, "", "")


def allowlist(path, subjects=IDS, *, never=False, extra=None):
    sources = [dict(name=H.DEV_STAGE, dataset="OASIS", path="x/stage.json", sha256="0"*64, never_for_training=never, subjects=list(subjects))]
    if extra:
        sources.append(dict(name="other", dataset="OASIS", path="x/o.json", sha256="1"*64, never_for_training=False, subjects=list(extra)))
    everyone = sorted(set(subjects) | set(extra or ()))
    Path(path).write_text(json.dumps(dict(schema="bayesreg.dev_subjects.v1", subjects=dict(OASIS=everyone), sources=sources)))
    return Path(path)


@pytest.fixture
def dev(tmp_path, monkeypatch):
    """Synthetic stage_L2R_val_v1 (pair.npz with labels, as the development stage), the label NIfTI files, configC run directories."""
    lookup = H.seg4_map()
    labels = {s: syn.labels(20+k) for k, s in enumerate(IDS)}
    data, nifti, runs = tmp_path/"data", tmp_path/"nifti", tmp_path/"runs"
    for subject, seg35 in labels.items():
        save_scalar(nifti/subject/H.LABEL_FILES["seg35"], seg35, GRID)
        save_scalar(nifti/subject/H.LABEL_FILES["seg4"], lookup[seg35.astype(int)].astype(np.float64), GRID)
    for k, (fixed, moving) in enumerate(PAIRS):
        name = H.PAIR_NAME.format(fixed=fixed, moving=moving)
        arrays = dict(fixed=np.zeros(syn.SHAPE), moving=np.zeros(syn.SHAPE), fixed_seg35=labels[fixed].astype(np.uint8), moving_seg35=labels[moving].astype(np.uint8),
                      fixed_seg4=lookup[labels[fixed].astype(int)].astype(np.uint8), moving_seg4=lookup[labels[moving].astype(int)].astype(np.uint8),
                      initial_u=np.zeros(syn.SHAPE+(3,)))
        folder = syn.write_pair(data, name, fixed, moving, arrays, stage=H.DEV_STAGE)
        syn.write_configc_run(runs/name, name, syn.d_to_u(syn.smooth_d(amplitude=(1., -.5+.2*k, .4))), arrays, input_sha256=file_sha256(folder/"pair.npz"),
                              pair_dataset=H.DEV_STAGE)
    monkeypatch.setattr(H, "DEV_SUBJECTS", IDS)
    monkeypatch.setattr(H, "DEV_PAIRS", PAIRS)
    monkeypatch.setattr(H, "DATA_ROOT", nifti)
    monkeypatch.setattr(H, "RESULTS", tmp_path/"results")
    monkeypatch.setattr(H, "TRUTH_GATE_OUT", tmp_path/"results"/"ev"/"truth_gate_v1")
    monkeypatch.setattr(H, "TRUTH_GATE_LOCK", tmp_path/"results"/"ev"/"truth_gate.lock")
    monkeypatch.setattr(TG, "FIELD_ROOT", runs)
    monkeypatch.setattr(ev, "code_state", lambda root=None: dict(head="0"*40, clean=True, status=""))
    return dict(tmp=tmp_path, data=data, nifti=nifti, runs=runs, allow=allowlist(tmp_path/"allow.json"), labels=labels)


# ---- the truth-reader gate ------------------------------------------------------------------------------------------------------------------------------
def test_gate_refuses_while_its_plan_or_field_list_is_not_pinned(monkeypatch):
    monkeypatch.setattr(H, "TRUTH_GATE_PLAN_SHA256", None)          # P0 pinned both (runs/20261114-p0/P0_pins.md); the unpinned state, set back here, still refuses
    monkeypatch.setattr(TG, "FIELDS_SHA256", None)
    assert H.TRUTH_GATE_PLAN_SHA256 is None and TG.FIELDS_SHA256 is None
    with pytest.raises(ev.ToolRefused) as info:
        TG.check_inputs()
    assert info.value.stage == "plan"
    monkeypatch.setattr(H, "TRUTH_GATE_PLAN_SHA256", file_sha256(H.REPO/H.TRUTH_GATE_PLAN_REL))
    with pytest.raises(ev.ToolRefused) as info:
        TG.check_inputs()
    assert info.value.stage == "fields"
    assert TG.main(["dry-run"]) == 2


def test_gate_pins_are_the_committed_plan_and_field_list():
    """P0: TRUTH_GATE_PLAN_SHA256 and FIELDS_SHA256 are the sha256 of the committed plan and field list; the gate's pin check accepts both (only these two code-tree
    files are read; check_inputs' hashing of the 171 listed fields is not run here)."""
    P0.assert_pinned(H.TRUTH_GATE_PLAN_SHA256, H.TRUTH_GATE_PLAN_REL)
    P0.assert_pinned(TG.FIELDS_SHA256, TG.FIELDS_REL)
    assert ev.pinned_file(H.TRUTH_GATE_PLAN_REL, H.TRUTH_GATE_PLAN_SHA256, "plan") == dict(path=H.TRUTH_GATE_PLAN_REL, sha256=H.TRUTH_GATE_PLAN_SHA256)
    assert ev.pinned_file(TG.FIELDS_REL, TG.FIELDS_SHA256, "fields") == dict(path=TG.FIELDS_REL, sha256=TG.FIELDS_SHA256)


def test_gate_field_list_names_the_19_configc_runs():
    listed = ev.sha256_list(H.REPO/TG.FIELDS_REL)
    dirs = {Path(p).relative_to(TG.FIELD_ROOT).parts[0] for p in listed}
    assert dirs == {H.PAIR_NAME.format(fixed=f, moving=m) for f, m in H.DEV_PAIRS} and len(listed) == 19*9
    assert all(not p.endswith(("metrics.json", "pair.npz")) for p in listed)            # no evaluation record, no label container


def test_gate_admission_refuses_t0_training_and_sealed(dev, tmp_path, monkeypatch):
    admission, rows = H.admit_truth_gate(data_root=dev["data"], allowlist_path=dev["allow"])
    assert [r["row"][3] for r in rows] == [H.PAIR_NAME.format(fixed=f, moving=m) for f, m in PAIRS]
    with pytest.raises(SubjectSealed, match="never_for_training"):
        H.admit_truth_gate(data_root=dev["data"], allowlist_path=allowlist(tmp_path/"t0.json", never=True))
    with pytest.raises(SubjectSealed, match="not only"):
        H.admit_truth_gate(data_root=dev["data"], allowlist_path=allowlist(tmp_path/"two.json", extra=(IDS[0],)))
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.seg.unet.data import load_folds
    training = sorted(load_folds(H.REPO/H.FOLDS_REL).training_subjects)[:1]
    sealed = sorted(load_sealed_lists().subjects["holdout_B_v1"])[:1]
    for bad in (training, sealed):
        monkeypatch.setattr(H, "DEV_SUBJECTS", tuple(bad))
        with pytest.raises(SubjectSealed):
            H.admit_truth_gate(data_root=dev["data"], allowlist_path=allowlist(tmp_path/f"{bad[0]}.json", subjects=bad))


def test_gate_items_bitwise_on_synthetic_data_and_the_lock_once(dev, monkeypatch):
    monkeypatch.setattr(TG, "check_inputs", lambda: (dict(path="plan", sha256="f"*64), dict(path="fields", sha256="e"*64), {}, dict(files=0, verified=True)))
    code = TG.run(run_unit=no_units, data_root=dev["data"], allowlist_path=dev["allow"])
    assert code == 8                                                  # 3 subjects and 2 pairs: identical, but not the 20/20 and 19/19 of the reading
    result = json.loads((H.TRUTH_GATE_OUT/"truth_gate.json").read_text())
    i, ii = result["item_i"], result["item_ii"]
    assert (i["n"], i["n_identical"]) == (3, 3) and (ii["n"], ii["n_identical"]) == (2, 2) and ii["cli_exit"] == 0 and ii["aggregates_identical"]
    rows = [r for items in i["subjects"].values() for r in items if "bitwise" in r]
    assert all(r["npz_dtype"] == "uint8" and r["nifti_stored_dtype"] == "float64" for r in rows)
    assert len(i["subjects"][IDS[1]]) == 2*2+2                        # the middle subject is compared at both pairs
    assert result["reading"] == H.GATE_READING_FAIL and Path(H.TRUTH_GATE_LOCK).is_file()
    assert "seg35" not in json.dumps(ii["pairs"]) or all(not p["differing_keys"] for p in ii["pairs"])
    with pytest.raises(ev.ToolRefused):
        TG.run(run_unit=no_units, data_root=dev["data"], allowlist_path=dev["allow"])            # the output directory exists
    import shutil
    shutil.rmtree(H.TRUTH_GATE_OUT)
    with pytest.raises(H.HoldoutRefused) as info:
        TG.run(run_unit=no_units, data_root=dev["data"], allowlist_path=dev["allow"])
    assert info.value.stage == "once"


def test_gate_reader_difference_is_caught(dev, monkeypatch):
    seg35 = dev["labels"][IDS[2]].copy()
    seg35[seg35 == 5] = 6
    path = dev["nifti"]/IDS[2]/H.LABEL_FILES["seg35"]
    path.unlink()
    save_scalar(path, seg35, GRID)
    monkeypatch.setattr(TG, "check_inputs", lambda: (dict(path="plan", sha256="f"*64), dict(path="fields", sha256="e"*64), {}, dict(files=0, verified=True)))
    TG.run(run_unit=no_units, data_root=dev["data"], allowlist_path=dev["allow"])
    result = json.loads((H.TRUTH_GATE_OUT/"truth_gate.json").read_text())
    assert result["item_i"]["n_identical"] == 2 and IDS[2] not in result["item_i"]["identical"]


def test_gate_reading_rule():
    good_i, good_ii = dict(n=20, n_identical=20), dict(n=19, n_identical=19, cli_exit=0)
    assert TG.reading(good_i, good_ii, True) == H.GATE_READING_PASS
    assert TG.reading(dict(n=20, n_identical=19), good_ii, True) == H.GATE_READING_FAIL
    assert TG.reading(good_i, dict(good_ii, n_identical=18), True) == H.GATE_READING_FAIL
    assert TG.reading(None, None, False) == H.GATE_READING_UNDECIDED


def test_gate_machine_gate_refuses_other_units(dev, monkeypatch):
    monkeypatch.setattr(TG, "check_inputs", lambda: (dict(path="plan", sha256="f"*64), dict(path="fields", sha256="e"*64), {}, dict(files=0, verified=True)))
    busy = lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "tag-hb-x-ants.service loaded active running x\n", "")    # noqa: E731
    with pytest.raises(H.HoldoutRefused) as info:
        TG.run(run_unit=busy, data_root=dev["data"], allowlist_path=dev["allow"])
    assert info.value.stage == "machine" and not Path(H.TRUTH_GATE_LOCK).exists()


def pinned_field_list(dev, monkeypatch):
    """The gate's two pins on a temporary code tree (ev.ROOT, TG.ROOT): the plan and a field list naming every file of the synthetic configC runs (FIELD_ROOT = dev runs) by sha256."""
    root = dev["tmp"]/"code"
    (root/"runs"/"20261107-ev").mkdir(parents=True)
    plan = root/H.TRUTH_GATE_PLAN_REL
    plan.write_text("# synthetic plan of the truth-reader gate\n", encoding="utf-8")
    files = sorted(p for p in Path(dev["runs"]).rglob("*") if p.is_file())
    (root/TG.FIELDS_REL).write_text("".join(f"{file_sha256(p)}  {p}\n" for p in files), encoding="utf-8")
    monkeypatch.setattr(ev, "ROOT", root)
    monkeypatch.setattr(TG, "ROOT", root)
    monkeypatch.setattr(H, "TRUTH_GATE_PLAN_SHA256", file_sha256(plan))
    monkeypatch.setattr(TG, "FIELDS_SHA256", file_sha256(root/TG.FIELDS_REL))
    return files


def test_gate_check_inputs_hashes_every_listed_file_for_real(dev, monkeypatch, capsys):
    """B4 review, 3rd plan, B1 (A5b item 2): check_inputs is not a stand-in here; it calls ev.verify_list on every listed file; the dry run says so truthfully."""
    files = pinned_field_list(dev, monkeypatch)
    calls = []
    real = ev.verify_list
    monkeypatch.setattr(ev, "verify_list", lambda listed: calls.append(sorted(listed)) or real(listed))
    plan, fields, listed, verified = TG.check_inputs()
    assert verified == dict(files=len(files), verified=True) and calls == [sorted(map(str, files))] and len(files) > 0
    assert TG.run(dry=True, data_root=dev["data"], allowlist_path=dev["allow"]) == 0
    assert f"({len(files)} files verified by sha256)" in capsys.readouterr().out and not Path(H.TRUTH_GATE_LOCK).exists()
    code = TG.run(run_unit=no_units, data_root=dev["data"], allowlist_path=dev["allow"])
    result = json.loads((H.TRUTH_GATE_OUT/"truth_gate.json").read_text())
    assert code == 8 and result["fields_verified"]["files"] == len(files) and result["fields_verified_before_ii"]["files"] == len(files)      # hashed before the lock and again before (ii)


def test_gate_a_listed_file_changed_by_one_byte_refuses_before_the_lock_and_reads_nothing(dev, monkeypatch):
    files = pinned_field_list(dev, monkeypatch)
    reads = []
    monkeypatch.setattr(H, "_load_label_volume", lambda path: reads.append(path))
    target = next(p for p in files if p.name == "summary.json" or p.suffix == ".json")
    raw = bytearray(target.read_bytes())
    raw[-2] = raw[-2] ^ 1
    target.write_bytes(bytes(raw))
    with pytest.raises(ev.ToolRefused) as info:
        TG.run(run_unit=no_units, data_root=dev["data"], allowlist_path=dev["allow"])
    assert info.value.stage == "list" and str(target) in str(info.value)
    assert reads == [] and not Path(H.TRUTH_GATE_LOCK).exists() and not Path(H.TRUTH_GATE_OUT).exists()


def test_gate_code_state_covers_every_code_directory_and_splits(tmp_path):
    """B4 review, 3rd plan m1: the clean check of the two EV tools is the entry's CLEAN_DIRS (experiments/ beyond ev/, scripts/ too)."""
    repo = tmp_path/"r"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo/"scripts").mkdir()
    (repo/"scripts"/"x.py").write_text("# untracked\n")
    state = ev.code_state(repo)
    assert state["clean"] is False and "scripts/" in state["status"] and state["dirs"] == list(H.CLEAN_DIRS)
    (repo/"scripts"/"x.py").unlink()
    (repo/"experiments"/"pq").mkdir(parents=True)
    (repo/"experiments"/"pq"/"y.py").write_text("# untracked\n")
    assert ev.code_state(repo)["clean"] is False


# ---- the evaluation-entry control ------------------------------------------------------------------------------------------------------------------------
def he_like(dev, tmp_path, *, alter=None):
    """An HE-like output (disp.npz) and an evaluation.json written the way HE's evaluate_main writes it (the bench's truth, json.dumps with NaN)."""
    from bayesreg.eval import fields, metrics
    from bayesreg.eval import truth as T
    from bayesreg.io.guard import load_dev_allowlist
    from bayesreg.solve.export import write_disp
    fixed, moving = PAIRS[0]
    name = H.PAIR_NAME.format(fixed=fixed, moving=moving)
    he = tmp_path/"he"/"0901-0902"
    (he/"output").mkdir(parents=True)
    u = syn.d_to_u(syn.smooth_d())
    write_disp(he/"output"/"disp.npz", u, GRID, state_hash="0"*64)
    pair_dir, grid, _, guard = T.admit_pair(dev["data"], H.DEV_STAGE, name, load_dev_allowlist(dev["allow"]), T.load_sealed_lists())
    truth, truth_record = T.load_truth(pair_dir, metrics.OASIS, grid, guard)
    d, disp_record = fields.read_disp(he/"output"/"disp.npz", grid.shape)
    scalars, per_label, official = metrics.evaluate_field(d, truth, metrics.OASIS, tuple(grid.spacing))
    if alter:
        scalars = dict(scalars, **alter)
    (he/"evaluation.json").write_text(json.dumps(dict(pair=name, metrics=scalars, per_label=per_label, official_row=official, disp=disp_record, truth=truth_record),
                                                  indent=1, default=float))
    return he, fixed, moving, name


def control(monkeypatch, dev, tmp_path, **kw):
    he, fixed, moving, name = he_like(dev, tmp_path, **kw)
    monkeypatch.setattr(EC, "PLAN_SHA256", file_sha256(H.REPO/EC.PLAN_REL))
    monkeypatch.setattr(EC, "FIXED", fixed)
    monkeypatch.setattr(EC, "MOVING", moving)
    monkeypatch.setattr(EC, "PAIR", name)
    monkeypatch.setattr(EC, "HE_PAIR_DIR", he)
    monkeypatch.setattr(EC, "HE_FILES", {str(he/"output"/"disp.npz"): file_sha256(he/"output"/"disp.npz"), str(he/"evaluation.json"): file_sha256(he/"evaluation.json")})
    monkeypatch.setattr(EC, "OUT", H.RESULTS/"ev"/"entry_control_v1")
    monkeypatch.setattr(EC, "LOCK", H.RESULTS/"ev"/"entry_control.lock")
    return EC.run(run_unit=no_units, data_root=dev["data"], allowlist_path=dev["allow"])


def test_control_calls_the_entrys_own_metric_function():
    assert EC.SCORE is H.score_cell


def test_control_refuses_while_its_plan_is_not_pinned(monkeypatch):
    monkeypatch.setattr(EC, "PLAN_SHA256", None)                    # P0 pinned it (runs/20261114-p0/P0_pins.md); the unpinned state, set back here, still refuses
    assert EC.PLAN_SHA256 is None
    with pytest.raises(ev.ToolRefused) as info:
        EC.run(dry=True)
    assert info.value.stage == "plan"


def test_control_pin_is_the_committed_plan():
    """P0: PLAN_SHA256 is the sha256 of the committed eval_control_plan.md; the control's pin check accepts it (only the plan is read)."""
    P0.assert_pinned(EC.PLAN_SHA256, EC.PLAN_REL)
    assert ev.pinned_file(EC.PLAN_REL, EC.PLAN_SHA256, "plan") == dict(path=EC.PLAN_REL, sha256=EC.PLAN_SHA256)


def test_control_number_for_number_and_once(dev, tmp_path, monkeypatch):
    assert control(monkeypatch, dev, tmp_path) == 0
    result = json.loads((EC.OUT/"entry_control.json").read_text())
    assert result["reading"] == EC.READ_SAME and all(result["identity"].values()) and result["score_is_entry_function"]
    assert not any(result["differing_keys"].values())
    import shutil
    shutil.rmtree(EC.OUT)
    with pytest.raises(H.HoldoutRefused) as info:
        EC.run(run_unit=no_units, data_root=dev["data"], allowlist_path=dev["allow"])
    assert info.value.stage == "once"


def test_control_catches_a_different_number(dev, tmp_path, monkeypatch):
    assert control(monkeypatch, dev, tmp_path, alter=dict(seg35=.5)) == 8
    result = json.loads((EC.OUT/"entry_control.json").read_text())
    assert result["reading"] == EC.READ_DIFFERENT and result["differing_keys"]["metrics"] == ["/seg35"]


def test_same_compares_bitwise_and_nan():
    assert ev.same(dict(a=[1., float("nan")]), dict(a=[1., float("nan")])) == []
    assert ev.same(dict(a=1.), dict(a=1.+2**-52)) == ["/a"]
    assert ev.same(dict(a=None), dict(a=float("nan")), null_is_nan=True) == [] and ev.same(dict(a=None), dict(a=float("nan"))) == ["/a"]
    assert ev.same(dict(a=1, seconds=3.), dict(a=1, seconds=4.), skip=("seconds",)) == []
