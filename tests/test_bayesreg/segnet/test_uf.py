"""Work package UF (runs/20261105-uf/任务书.md, plan.md): the recipe v3_large trained to the end of its schedule, four members matching v2 one to one, the nnU-Net probabilities, one evaluation, the reading. CPU, synthetic small volumes (ids OASIS_OAS1_09xx_MR1), one thread.

Covered: the checkpoint of the last step through train, calibrate and the member check (and the old paths unchanged); early stop off; the members against v2 (the plan, the records, every deviation); the 1/2-1/2 average; the reading's line and every branch of "not decidable"; the read lock; the gate of the nnU-Net
probabilities; the chain (order, once, hold, gate, time limit, the in-process member check); the start string; one synthetic run of the whole chain through the real code paths. Real repository and results files are read for ids, records and sha256 only: no image, no label, no prediction of a real subject.
No test starts a unit, uses a GPU or runs a real training; the real entry is called in a subprocess with a blocking systemd-run first on PATH and must stop at the snapshot check.
"""
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import types

import nibabel as nib
import numpy as np
import pytest
import torch

from jd_curl.validation import InputValidationError

from bayesreg.io.guard import SubjectSealed, load_dev_allowlist
from bayesreg.seg.channels import read_back_probabilities
from bayesreg.seg.unet import large
from bayesreg.seg.unet.calibrate import calibrate_member, selected_checkpoint
from bayesreg.seg.unet.data import LabelGuard, OasisFiles, load_full_folds
from bayesreg.seg.unet.infer import _save_npz
from bayesreg.seg.unet.large import TrainConfigLarge, train_large_member
from bayesreg.seg.unet.registry import MemberCheckFailed, verify_members
from bayesreg.seg.unet.snapshot import CODE_ROOT, SNAPSHOT_PATHS_V3

from ._helpers import A_CALIB, A_TRAIN, tiny_subject

UF = CODE_ROOT/"experiments"/"uf"
sys.path[:0] = [str(UF), str(CODE_ROOT/"experiments"/"unv2"), str(CODE_ROOT/"experiments"/"ul"), str(CODE_ROOT/"experiments"/"nn"), str(CODE_ROOT/"experiments"/"fl"), str(CODE_ROOT/"experiments"/"ef"), str(CODE_ROOT/"experiments"/"mc0")]
import uf_common as uc  # noqa: E402
import uf_lock  # noqa: E402
import uf_nn  # noqa: E402
import nx_seg  # noqa: E402


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = load_script("uf_run", UF/"run_uf.py")
EVAL = load_script("uf_eval", UF/"eval_uf.py")
DECIDE = load_script("uf_decide", UF/"decide_uf.py")
SEG = EVAL.eval_seg
LAST = dict(base_channels=4, depth=2, ds_levels=1, batch_size=2, patch_size=[8, 8, 8], max_iters=6, min_iters=6, patience=6, eval_every=2, warmup_iters=1, log_every=1, amp=False, flip_axis_check=False, checkpoint="last")
LIA = np.array([[-1., 0, 0, 90.], [0, 0, 1., -120.], [0, -1., 0, 110.], [0, 0, 0, 1.]])


def quiet(_):
    pass


def tiny_sets():
    return [tiny_subject(s, shape=(16, 16, 16), seed=i) for i, s in enumerate(A_TRAIN)], [tiny_subject(A_CALIB[0], shape=(16, 16, 16), seed=9)]


def falling_dice(monkeypatch):
    """The calib Dice falls at every evaluation: the best evaluation is the first, the last step is never the best."""
    calls = []
    real = large.evaluate

    def fake(*args, **kwargs):
        metrics = real(*args, **kwargs)
        calls.append(1)
        return dict(metrics, mean_dice35=1.-.1*len(calls))
    monkeypatch.setattr(large, "evaluate", fake)


def state_sha(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


# ---- the last step: train, calibrate, the member check ----------------------------------------------------------------------------------------------------------------

def test_checkpoint_choice_is_refused_unless_known(tmp_path):
    train, calib = tiny_sets()
    with pytest.raises(InputValidationError, match="checkpoint"):
        train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=dict(LAST, checkpoint="best"), device="cpu", log=quiet, subjects=(train, calib))
    assert TrainConfigLarge().checkpoint == "best_calib" and large.CHECKPOINT_CHOICES == ("best_calib", "last")


def test_the_default_writes_and_records_nothing_new(tmp_path):
    train, calib = tiny_sets()
    record = train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config={k: v for k, v in LAST.items() if k != "checkpoint"}, device="cpu", log=quiet, subjects=(train, calib))
    assert not (tmp_path/"foldA_s0"/"model_last.pt").exists() and not [k for k in record if k.startswith("checkpoint_")]
    calibrate_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, device="cpu", log=quiet, subjects=calib)
    blob = torch.load(tmp_path/"foldA_s0"/"member.pt", map_location="cpu", weights_only=True)
    assert not [k for k in blob if k.startswith("checkpoint")] and "checkpoint" not in json.loads((tmp_path/"foldA_s0"/"calibration.json").read_text())
    assert selected_checkpoint(record) == ("model_best.pt", record["model_best_sha256"], None)


def test_the_member_is_the_last_step_not_the_best(tmp_path, monkeypatch):
    falling_dice(monkeypatch)
    train, calib = tiny_sets()
    record = train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=dict(LAST), device="cpu", log=quiet, subjects=(train, calib))
    out = tmp_path/"foldA_s0"
    assert record["stopped"] == "max_iters" and record["iterations_run"] == 6 and record["best_iteration"] == 2           # early stop is off: the falling Dice did not stop it
    assert record["checkpoint_choice"] == "last" and record["checkpoint_file"] == "model_last.pt" and record["checkpoint_iteration"] == 6 and record["checkpoint_sha256"] == uc.sha256_file(out/"model_last.pt")
    assert record["checkpoint_calib_mean_dice35"] == record["eval_history"][-1]["mean_dice35"] and len(record["eval_history"]) == 3       # every calib Dice is recorded
    best = torch.load(out/"model_best.pt", map_location="cpu", weights_only=True)
    last = torch.load(out/"model_last.pt", map_location="cpu", weights_only=True)
    assert best["iteration"] == 2 and last["iteration"] == 6 and state_sha(best["state_dict"]) != state_sha(last["state_dict"]) and not [k for k in last["state_dict"] if k.startswith("aux_heads")]
    end = [json.loads(line) for line in (out/"train_log.jsonl").read_text().splitlines() if '"end"' in line][-1]
    assert end["checkpoint_iteration"] == 6
    calibrate_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, device="cpu", log=quiet, subjects=calib)
    member = torch.load(out/"member.pt", map_location="cpu", weights_only=True)
    assert state_sha(member["state_dict"]) == state_sha(last["state_dict"]) and member["best_iteration"] == 6
    assert {k: member[k] for k in ("checkpoint_choice", "checkpoint_file", "checkpoint_sha256", "checkpoint_iteration")} == {k: record[k] for k in ("checkpoint_choice", "checkpoint_file", "checkpoint_sha256", "checkpoint_iteration")}
    assert member["model_best_sha256"] == record["model_best_sha256"] and json.loads((out/"calibration.json").read_text())["checkpoint"]["checkpoint_iteration"] == 6


def test_the_same_falling_dice_would_have_stopped_u1s_settings(tmp_path, monkeypatch):
    falling_dice(monkeypatch)
    train, calib = tiny_sets()
    record = train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=dict(LAST, patience=1, min_iters=2), device="cpu", log=quiet, subjects=(train, calib))
    assert record["stopped"] == "early_stop" and record["iterations_run"] == 4 and record["checkpoint_iteration"] == 4      # so the plan's patience = min_iters = max_iters has teeth


def test_calibrate_refuses_a_changed_or_foreign_checkpoint(tmp_path):
    train, calib = tiny_sets()
    record = train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=dict(LAST), device="cpu", log=quiet, subjects=(train, calib))
    with (tmp_path/"foldA_s0"/"model_last.pt").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(InputValidationError, match="checkpoint_sha256"):
        calibrate_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, device="cpu", log=quiet, subjects=calib)
    for change in (dict(checkpoint_file="../model_best.pt"), dict(checkpoint_file="model_best.pt"), dict(checkpoint_choice="best_calib"), dict(checkpoint_iteration="6")):
        with pytest.raises(InputValidationError, match="names the checkpoint"):
            selected_checkpoint(dict(record, **change))


def test_a_record_of_the_last_step_that_lost_its_keys_does_not_fall_back(tmp_path):
    from bayesreg.seg.unet.calibrate import names_a_checkpoint
    train, calib = tiny_sets()
    record = train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=dict(LAST), device="cpu", log=quiet, subjects=(train, calib))
    stripped = {k: v for k, v in record.items() if not k.startswith("checkpoint_")}
    assert names_a_checkpoint(record) and names_a_checkpoint(stripped) and not names_a_checkpoint(dict(stripped, config=dict(stripped["config"], checkpoint="best_calib")))
    for broken in (stripped, dict(stripped, checkpoint_file="model_last.pt"), dict(stripped, checkpoint_choice="last", checkpoint_file="model_last.pt", checkpoint_iteration=6)):
        with pytest.raises(InputValidationError, match="names the checkpoint"):
            selected_checkpoint(broken)


def test_calibrate_requires_the_mark_of_the_last_step_inside_the_file(tmp_path):
    train, calib = tiny_sets()
    train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=dict(LAST), device="cpu", log=quiet, subjects=(train, calib))
    out = tmp_path/"foldA_s0"
    blob = torch.load(out/"model_last.pt", map_location="cpu", weights_only=True)
    del blob["checkpoint"]
    torch.save(blob, out/"model_last.pt")
    record = json.loads((out/"train_record.json").read_text())
    record["checkpoint_sha256"] = uc.sha256_file(out/"model_last.pt")
    (out/"train_record.json").write_text(json.dumps(record))
    with pytest.raises(InputValidationError, match="not the record's last step"):
        calibrate_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, device="cpu", log=quiet, subjects=calib)


def test_member_check_reads_the_last_checkpoint(tmp_path):
    import synthetic_tree
    work = synthetic_tree.build(tmp_path/"tree")
    allowlist, split = load_dev_allowlist(work/"dev_subjects.json"), load_full_folds(work/"folds_full_v2.json")
    for fold in ("A", "B"):
        train_large_member(files=OasisFiles(work/"data"), guard=LabelGuard(split, allowlist, training_source=synthetic_tree.SOURCE), fold=fold, seed=0, models_root=tmp_path/"m", config=dict(LAST, patch_size=[16, 16, 16]), device="cpu", log=quiet)
        calibrate_member(files=OasisFiles(work/"data"), guard=LabelGuard(split, allowlist, training_source=synthetic_tree.SOURCE), fold=fold, seed=0, models_root=tmp_path/"m", device="cpu", log=quiet)
    _, check = verify_members(tmp_path/"m", split=split, allowlist=allowlist, training_source=synthetic_tree.SOURCE)
    assert check["passed"] and all(m["checkpoint_choice"] == "last" and m["checkpoint_iteration"] == 6 for m in check["members"])
    with (tmp_path/"m"/"foldB_s0"/"model_last.pt").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(MemberCheckFailed, match="model_last.pt"):
        verify_members(tmp_path/"m", split=split, allowlist=allowlist, training_source=synthetic_tree.SOURCE)


# ---- the plan's constants and the members against v2 ---------------------------------------------------------------------------------------------------------------------

def test_the_plan_constants():
    assert uc.MEMBERS == (("A", 0), ("B", 0), ("A", 1), ("B", 1)) and uc.MEMBER_NAMES == ("foldA_s0", "foldB_s0", "foldA_s1", "foldB_s1")
    assert uc.V2_NAME == dict(foldA_s0="A0", foldB_s0="B0", foldA_s1="A1", foldB_s1="B1")
    assert uc.TRAIN_CONFIG == dict(max_iters=40000, min_iters=40000, patience=40000, eval_every=500, checkpoint="last", max_wall_seconds=14400.)
    assert uc.WORTH_BAR == .003 and uc.CONTROL_TOLERANCES == dict(v1_control=1e-9, v2_control=1e-9, U1_control=2e-4) and uc.NN_WEIGHT == .5 and uc.OUT == Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_full")
    assert uc.FIRST5_PAIRS[0] == ("OASIS_OAS1_0438_MR1", "OASIS_OAS1_0439_MR1") and uc.FIRST5_PAIRS[-1] == ("OASIS_OAS1_0442_MR1", "OASIS_OAS1_0443_MR1") and len(uc.FIRST5_PAIRS) == 5
    for root in (uc.V1_ROOT, uc.V2_ROOT, uc.U1_ROOT, uc.NN_OUT):
        assert root not in uc.OUT.parents and uc.OUT not in root.parents and root != uc.OUT


def test_the_recipe_is_v3_large_with_only_the_plans_changes():
    planned, recipe = uc.planned_config(), json.loads(json.dumps(asdict(TrainConfigLarge())))
    assert sorted(k for k in recipe if planned[k] != recipe[k]) == ["checkpoint", "max_iters", "max_wall_seconds", "min_iters", "patience"]
    assert planned["patch_size"] == [128, 160, 112] and planned["base_channels"] == 32 and planned["batch_size"] == 2 and planned["ds_levels"] == 2 and planned["window_gaussian_sigma"] == .125 and planned["flip_axis_check"] is True
    lr_last = large._lr_at(planned["max_iters"]-1, TrainConfigLarge(**uc.TRAIN_CONFIG))
    assert lr_last < 1e-7 and planned["min_iters"] >= planned["max_iters"] and planned["patience"] >= planned["max_iters"]       # the schedule ends; early stop cannot fire


def v2_world(tmp_path, *, contact, folds):
    """Records of a v2 in its layout: member.json, train_record.json, the registry. Ids only."""
    root = tmp_path/"v2"
    models = {}
    for fold, seed in uc.MEMBERS:
        name = uc.member_name(fold, seed)
        directory = root/name
        directory.mkdir(parents=True)
        record = dict(fold=fold, seed=seed, train_subjects=folds[fold]["train"], calib_subjects=folds[fold]["calib"], label_contact_subjects=contact)
        (directory/"train_record.json").write_text(json.dumps(record))
        sha = hashlib.sha256(name.encode()).hexdigest()
        (directory/"member.json").write_text(json.dumps(dict(fold=fold, seed=seed, member_sha256=sha, train_record_sha256=uc.sha256_file(directory/"train_record.json"))))
        models.setdefault(fold, []).append(sha)
    registry = tmp_path/"segmenters_v2.json"
    registry.write_text(json.dumps(dict(models={f"unet35_v2_full_fold{f}": dict(segmenter_id="unet35_v2_full", fold={"A": 0, "B": 1}[f], member_model_sha256=s) for f, s in models.items()})))
    return root, registry


FOLDS = dict(A=dict(train=["OASIS_OAS1_0902_MR1", "OASIS_OAS1_0901_MR1"], calib=["OASIS_OAS1_0903_MR1"]), B=dict(train=["OASIS_OAS1_0904_MR1"], calib=["OASIS_OAS1_0905_MR1"]))
CONTACT = sorted(["OASIS_OAS1_0901_MR1", "OASIS_OAS1_0902_MR1", "OASIS_OAS1_0903_MR1"])


def test_v2_member_lists_and_their_refusals(tmp_path):
    root, registry = v2_world(tmp_path, contact=CONTACT, folds=FOLDS)
    v2 = uc.v2_member_lists(root, registry=registry)
    assert sorted(v2) == sorted(uc.MEMBER_NAMES) and v2["foldA_s0"]["train"] == sorted(FOLDS["A"]["train"]) and v2["foldB_s1"]["seed"] == 1
    (root/"foldA_s1"/"train_record.json").write_text(json.dumps(dict(json.loads((root/"foldA_s1"/"train_record.json").read_text()), calib_subjects=[])))
    with pytest.raises(ValueError, match="sha256 differs"):
        uc.v2_member_lists(root, registry=registry)
    root, registry = v2_world(tmp_path/"b", contact=CONTACT, folds=FOLDS)
    registry.write_text(json.dumps(dict(models={})))
    with pytest.raises(ValueError, match="not registered"):
        uc.v2_member_lists(root, registry=registry)


def good_record(name, v2, config):
    fold, seed = name[4], int(name[-1])
    return dict(fold=fold, seed=seed, recipe="v3_large", config=config, stopped="max_iters", iterations_run=config["max_iters"], checkpoint_choice="last", checkpoint_file="model_last.pt", checkpoint_iteration=config["max_iters"], train_subjects=list(reversed(v2["train"])), calib_subjects=v2["calib"], label_contact_subjects=v2["contact"])


@pytest.mark.parametrize("change,match", [
    (dict(stopped="early_stop", iterations_run=30000), "not the 40000 steps"),
    (dict(stopped="wall_budget"), "not the 40000 steps"),
    (dict(iterations_run=39999), "not the 40000 steps"),
    (dict(checkpoint_choice=None, checkpoint_file=None, checkpoint_iteration=None), "not the last step"),
    (dict(checkpoint_iteration=39500), "not the last step"),
    (dict(recipe=None), "recipe"),
    (dict(seed=1), "the record is fold"),
    (dict(calib_subjects=["OASIS_OAS1_0905_MR1"]), "training or calibration subjects"),
    (dict(train_subjects=["OASIS_OAS1_0901_MR1"]), "training or calibration subjects"),
    (dict(label_contact_subjects=CONTACT[:2]), "label contact"),
])
def test_member_record_problems(tmp_path, change, match):
    root, registry = v2_world(tmp_path, contact=CONTACT, folds=FOLDS)
    v2 = uc.v2_member_lists(root, registry=registry)
    config = uc.planned_config()
    assert uc.member_record_problems(good_record("foldA_s0", v2["foldA_s0"], config), "foldA_s0", v2["foldA_s0"], expected_config=config) == []
    problems = uc.member_record_problems(dict(good_record("foldA_s0", v2["foldA_s0"], config), **change), "foldA_s0", v2["foldA_s0"], expected_config=config)
    assert problems and any(match in p for p in problems)


@pytest.mark.parametrize("key,value", [("patience", 20), ("min_iters", 10000), ("max_iters", 39000), ("checkpoint", "best_calib"), ("base_channels", 16), ("flip_probability", 0.)])
def test_member_record_problems_on_the_config(tmp_path, key, value):
    root, registry = v2_world(tmp_path, contact=CONTACT, folds=FOLDS)
    v2 = uc.v2_member_lists(root, registry=registry)
    config = uc.planned_config()
    problems = uc.member_record_problems(good_record("foldA_s0", v2["foldA_s0"], dict(config, **{key: value})), "foldA_s0", v2["foldA_s0"], expected_config=config)
    assert any(key in p for p in problems)
    if key in ("patience", "min_iters"):
        assert any("early stop not off" in p for p in problems)


def test_folds_problems_on_the_real_files():
    """The real folds file and v2's records (ids only): the four members get exactly the subjects of their v2 namesakes."""
    if not (uc.V2_ROOT/"foldA_s0"/"member.json").is_file():
        pytest.skip("unet35_v2_full is not on this machine")
    split = RUN.real_split()
    v2 = uc.v2_member_lists(uc.V2_ROOT, registry=CODE_ROOT/uc.V2_REGISTRY_REL)
    assert uc.folds_problems(split, v2) == []
    assert all(len(v2[n]["train"]) == 98 and len(v2[n]["calib"]) == 8 and len(v2[n]["contact"]) == 106 for n in uc.MEMBER_NAMES)
    assert v2["foldA_s0"]["calib"] == v2["foldA_s1"]["calib"] != v2["foldB_s0"]["calib"] == v2["foldB_s1"]["calib"]
    swapped = dict(v2, foldB_s1=dict(v2["foldB_s1"], calib=v2["foldA_s0"]["calib"]))
    assert uc.folds_problems(split, swapped) == ["foldB_s1: the folds file gives other subjects than v2's foldB_s1 trained on"]


def test_the_real_references_are_the_pinned_files():
    for path, pin in ((uc.REF_ENSEMBLES, uc.REF_ENSEMBLES_SHA256), (CODE_ROOT/uc.REF_MEMBERS_REL, uc.REF_MEMBERS_SHA256), (uc.REF_U1, uc.REF_U1_SHA256), (uc.U1_MEMBER/"member.pt", uc.U1_MEMBER_SHA256)):
        if not Path(path).is_file():
            pytest.skip(f"{path} is not on this machine")
        assert uc.sha256_file(path) == pin
    u1 = EVAL.read_u1_reference(uc.REF_U1, pinned_sha256=uc.REF_U1_SHA256)
    assert len(u1["subjects"]) == 20 and abs(u1["means"]["seg35"]-.92567) < 1e-5 and u1["member_sha256"] == uc.U1_MEMBER_SHA256


def test_a_reference_that_does_not_add_up_is_refused(tmp_path):
    rows = {"OASIS_OAS1_0907_MR1": dict(seg35=.9, seg4=.95, groups=dict(small_six=.8, big_six=.95, other_23=.9), per_label=[.9]*35, seg4_per_label=[.9]*4)}
    path = tmp_path/"u1.json"
    path.write_text(json.dumps(dict(subjects=list(rows), rows=dict(U1=rows), means=dict(U1=dict(seg35=.91, seg4=.95)), members=dict(U1=dict(member_sha256="0")))))
    with pytest.raises(ValueError, match="does not follow"):
        EVAL.read_u1_reference(path)
    with pytest.raises(ValueError, match="pinned"):
        EVAL.read_u1_reference(path, pinned_sha256="0"*64)


# ---- the 1/2-1/2 average and the float16 of UF4 -------------------------------------------------------------------------------------------------------------------------

def random_probabilities(rng, shape, n=36):
    p = rng.dirichlet(np.ones(n)*.3, size=shape).astype(np.float32)
    return np.moveaxis(p, -1, 0)


def test_half_and_half_is_the_plain_formula_by_slabs():
    rng = np.random.default_rng(0)
    p_uf = random_probabilities(rng, (11, 6, 5)).astype(np.float16)
    p_nn = random_probabilities(rng, (11, 6, 5)).astype(np.float64)
    p_nn /= p_nn.sum(axis=0)
    a = p_uf.astype(np.float64)
    want = (.5*(a/a.sum(axis=0))+.5*p_nn).argmax(axis=0)
    for slab in (1, 3, 8, 64):
        assert (EVAL.half_and_half(p_uf, p_nn, slab=slab) == want).all()
    with pytest.raises(ValueError, match="fixed"):
        EVAL.half_and_half(p_uf, p_nn, weight=.6)
    with pytest.raises(ValueError, match="float16"):
        EVAL.half_and_half(p_uf.astype(np.float32), p_nn)


def test_half_and_half_ties_go_to_the_smallest_label_and_each_family_counts_half():
    p_uf = np.zeros((36, 1, 1, 2), dtype=np.float16)
    p_nn = np.zeros((36, 1, 1, 2))
    p_uf[3, ..., 0], p_uf[5, ..., 0] = .6, .4                  # voxel 0: UF says 3, nnU-Net says 5 equally strongly: a tie, the smaller label
    p_nn[3, ..., 0], p_nn[5, ..., 0] = .4, .6
    p_uf[7, ..., 1], p_uf[2, ..., 1] = .75, .25                 # voxel 1: UF 7 at .75, nnU-Net 2 at 1.: (.375 + 0) against (.125 + .5)
    p_nn[2, ..., 1] = 1.
    assert EVAL.half_and_half(p_uf, p_nn).ravel().tolist() == [3, 2]


def test_the_argmax_of_the_float16_is_the_argmax_a_reader_gets_back(tmp_path):
    rng = np.random.default_rng(1)
    p = random_probabilities(rng, (9, 7, 6))
    p[:, 0, 0, 0] = 0.
    p[[4, 9], 0, 0, 0] = .5                                     # an exact tie
    p16 = p.astype(np.float16)
    _save_npz(tmp_path/"probs36.npz", p16)
    back, _ = read_back_probabilities(tmp_path/"probs36.npz")
    assert (back.argmax(axis=0) == p16.argmax(axis=0)).all() and p16.argmax(axis=0)[0, 0, 0] == 4


# ---- the nnU-Net step --------------------------------------------------------------------------------------------------------------------------------------------------

def test_infer_argv_is_nxs_with_this_root():
    train = dict(trainer="nnUNetTrainer_250epochs_NoMirroring")
    argv = uf_nn.infer_argv(train, out=Path("/x/out"))
    assert argv[:4] == [str(uf_nn.nc.NN_PY), "-I", "-B", str(CODE_ROOT/"experiments/fl/nx_infer_child.py")] and argv[4] == str(CODE_ROOT/"experiments/nn/infer_nn.py")
    assert argv[argv.index("--input-dir")+1] == "/x/out/nn/dev_input" and argv[argv.index("--out-dir")+1] == "/x/out/nn/raw" and argv[-1] == "--save-probabilities" and "--device" in argv
    assert argv[argv.index("--model-dir")+1] == str(uc.NN_OUT/"nnUNet_results/Dataset935_OASIS35/nnUNetTrainer_250epochs_NoMirroring__nnUNetPlans__3d_fullres")


def test_infer_fallbacks():
    text = "device cap 9 GiB\nOASIS_OAS1_0438_MR1: 1.2 s\nPrediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU\nOASIS_OAS1_0439_MR1: 9.0 s\n"
    assert uf_nn.infer_fallbacks(text) == dict(n_lines=1, fallback_subjects=["OASIS_OAS1_0439_MR1"])
    source = (CODE_ROOT/"experiments/fl/run_fx.py").read_text()
    body = source[source.index("def infer_fallbacks"):source.index("def nx_infer_one")]
    mine = (UF/"uf_nn.py").read_text()
    mine = mine[mine.index("def infer_fallbacks"):mine.index("def infer_problems")]
    strip = lambda text: [line.strip().replace("'", '"') for line in text.splitlines() if line.strip() and not line.strip().startswith('"""') and "FALLBACK" not in line]
    assert strip(body)[1:] == strip(mine)[1:]                   # the copy is run_fx's, but the docstring


def nn_training(tmp_path):
    nn_out = tmp_path/"nn_out"
    nn_out.mkdir()
    (nn_out/"checkpoint_final.pth").write_bytes(b"w")
    (nn_out/"train_record.json").write_text(json.dumps(dict(trainer="T", final_checkpoint=str(nn_out/"checkpoint_final.pth"), final_checkpoint_sha256=uc.sha256_file(nn_out/"checkpoint_final.pth"))))
    return nn_out


def test_infer_once_and_its_record(tmp_path):
    nn_out = nn_training(tmp_path)
    subjects = ["OASIS_OAS1_0907_MR1", "OASIS_OAS1_0908_MR1"]
    calls = []

    def fake(argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        raw = tmp_path/"out"/"nn"/"raw"
        raw.mkdir(parents=True)
        (raw/"infer_record.json").write_text(json.dumps(dict(checkpoint_sha256=uc.sha256_file(nn_out/"checkpoint_final.pth"), saved_probabilities=True, subjects=subjects, allowed_mirroring_axes=None, tta_views=1, gpu_peak_reserved_gib=3.)))
        return types.SimpleNamespace(returncode=0)
    check = uf_nn.infer(tmp_path/"out", subjects, nn_out=nn_out, data_root=tmp_path/"data", run=fake, env={"X": "1"})
    assert check["problems"] == [] and len(calls) == 1 and calls[0][1] == {"X": "1"} and "--save-probabilities" in calls[0][0]
    links = sorted((tmp_path/"out"/"nn"/"dev_input").iterdir())
    assert [p.name for p in links] == [f"{s}_0000.nii.gz" for s in subjects] and all(p.is_symlink() and os.readlink(p).endswith(f"{s}/aligned_norm.nii.gz") for p, s in zip(links, subjects))
    assert json.loads((tmp_path/"out"/"nn"/"infer_check.json").read_text())["checkpoint_sha256"] == uc.sha256_file(nn_out/"checkpoint_final.pth")
    with pytest.raises(uf_nn.NNStepFailed, match="never rerun"):
        uf_nn.infer(tmp_path/"out", subjects, nn_out=nn_out, data_root=tmp_path/"data", run=fake)


@pytest.mark.parametrize("change,match", [(dict(saved_probabilities=False), "no probabilities"), (dict(allowed_mirroring_axes=[0, 1]), "mirroring"), (dict(tta_views=8), "mirroring"), (dict(checkpoint_sha256="0"), "checkpoint_final"), (dict(subjects=["OASIS_OAS1_0907_MR1"]), "20 development"), (dict(gpu_peak_reserved_gib=9.5), "reserved peak")])
def test_infer_problems(change, match):
    train = dict(final_checkpoint_sha256="a")
    record = dict(checkpoint_sha256="a", saved_probabilities=True, subjects=["s1", "s2"], allowed_mirroring_axes=None, tta_views=1, gpu_peak_reserved_gib=3.)
    assert uf_nn.infer_problems(record, train, ["s2", "s1"]) == []
    assert any(match in p for p in uf_nn.infer_problems(dict(record, **change), train, ["s1", "s2"]))
    assert uf_nn.infer_problems({}, train, ["s1"]) == ["no infer_record.json"]


def write_raw(directory, subject, p, affine, *, labels=None, pred_dev=None):
    """nnU-Net's files for one subject: probabilities (classes, SimpleITK order) and label maps in the file's order, from arrays in load_scalar's order."""
    directory.mkdir(parents=True, exist_ok=True)
    native = np.stack([RUN.native_from_load_scalar(p[c], affine) for c in range(p.shape[0])])
    np.savez_compressed(directory/f"{subject}.npz", probabilities=native.transpose(0, 3, 2, 1))
    top = p.argmax(axis=0).astype(np.uint8) if labels is None else labels
    nib.save(nib.Nifti1Image(RUN.native_from_load_scalar(top, affine), affine), str(directory/f"{subject}.nii.gz"))
    if pred_dev is not None:
        pred_dev.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(RUN.native_from_load_scalar(top, affine), affine), str(pred_dev/f"{subject}.nii.gz"))


def image_on(path, shape_zyx, affine):
    """An image whose load_scalar shape is shape_zyx under this affine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(RUN.native_from_load_scalar(np.ones(shape_zyx, dtype=np.float32), affine), affine), str(path))


def test_the_inverse_orientation_of_the_synthetic_writer():
    rng = np.random.default_rng(2)
    values = rng.integers(0, 36, (5, 6, 7)).astype(np.uint8)
    for affine in (np.diag([1., 1, 1, 1]), LIA):
        assert (nx_seg.orient_native(RUN.native_from_load_scalar(values, affine), affine) == values).all()


def test_convert_passes_the_gate_and_the_reader_reads_the_argmax(tmp_path):
    rng = np.random.default_rng(3)
    subject = "OASIS_OAS1_0907_MR1"
    image_on(tmp_path/"data"/subject/"aligned_norm.nii.gz", (6, 7, 8), LIA)
    p = random_probabilities(rng, (6, 7, 8))
    write_raw(tmp_path/"raw", subject, p, LIA, pred_dev=tmp_path/"pred_dev")
    row = uf_nn.convert_subject(subject, raw_dir=tmp_path/"raw", pred_dev=tmp_path/"pred_dev", image_path=tmp_path/"data"/subject/"aligned_norm.nii.gz", out_root=tmp_path/"probs")
    back, _ = read_back_probabilities(tmp_path/"probs"/subject/"probs36.npz")
    assert row["gate"] == "passed" and row["n_differing"] == 0 and (back.argmax(axis=0) == p.argmax(axis=0)).all() and row["probs36_sha256"] == uc.sha256_file(tmp_path/"probs"/subject/"probs36.npz")
    with pytest.raises(FileExistsError):
        uf_nn.convert_subject(subject, raw_dir=tmp_path/"raw", pred_dev=tmp_path/"pred_dev", image_path=tmp_path/"data"/subject/"aligned_norm.nii.gz", out_root=tmp_path/"probs")


@pytest.mark.parametrize("break_it", ["flip", "other_subject"])
def test_the_gate_stops_and_leaves_nothing(tmp_path, break_it):
    rng = np.random.default_rng(4)
    subject = "OASIS_OAS1_0907_MR1"
    image_on(tmp_path/"data"/subject/"aligned_norm.nii.gz", (6, 7, 8), LIA)
    p = random_probabilities(rng, (6, 7, 8))
    labels = p.argmax(axis=0).astype(np.uint8)
    wrong = labels[:, :, ::-1].copy() if break_it == "flip" else random_probabilities(rng, (6, 7, 8)).argmax(axis=0).astype(np.uint8)
    write_raw(tmp_path/"raw", subject, p, LIA)
    write_raw(tmp_path/"dev_holder", subject, p, LIA, labels=wrong, pred_dev=tmp_path/"pred_dev")
    with pytest.raises(nx_seg.GateFailed, match="differs from pred_dev"):
        uf_nn.convert_subject(subject, raw_dir=tmp_path/"raw", pred_dev=tmp_path/"pred_dev", image_path=tmp_path/"data"/subject/"aligned_norm.nii.gz", out_root=tmp_path/"probs")
    assert not (tmp_path/"probs"/subject).exists()


def test_the_gate_line_is_fxs():
    assert nx_seg.GATE_TOLERANCE == 1e-4 and nx_seg.gate_passes(688, 6881280) and not nx_seg.gate_passes(689, 6881280)
    record = dict(gate="passed", tolerance=1e-4, subjects=["a", "b"], n_differing_by_subject=dict(a=0, b=3), n_voxels_by_subject=dict(a=10**6, b=10**6), probs36_sha256=dict(a="x", b="y"))
    assert uf_nn.convert_problems(record, ["a", "b"]) == []
    assert uf_nn.convert_problems(dict(record, probs36_sha256=dict(a="x")), ["a", "b"]) and uf_nn.convert_problems(dict(record, n_differing_by_subject=dict(a=0, b=101)), ["a", "b"]) and uf_nn.convert_problems({}, ["a", "b"])


@pytest.mark.parametrize("argv,match", [(["infer", "--synthetic"], "real data only"), (["convert", "--nn-out", "/tmp/x"], "real data"), (["convert", "--data-root", "/tmp/x"], "real data")])
def test_uf_nn_command_line_refuses(argv, match, capsys):
    with pytest.raises(SystemExit):
        uf_nn.main(argv)
    assert match in capsys.readouterr().err


# ---- the read lock ---------------------------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def lock(tmp_path, monkeypatch):
    path = tmp_path/"locks"/"uf_dev_read.lock"
    monkeypatch.setattr(uf_lock, "DEV_READ_LOCK", path)
    return path


def test_lock_path_is_fixed_and_outside_every_results_root():
    path = uf_lock.DEV_READ_LOCK
    assert path == Path("/home/rmcfz/seq_vp_tag_results/seg/uf_dev_read.lock") and path.parent == uc.OUT.parent and uc.OUT not in path.parents
    assert path.name not in ("ul_dev_read.lock", "nn_dev_read.lock")
    for script in ("eval_uf.py", "run_uf.py", "decide_uf.py", "uf_nn.py"):
        assert "--lock" not in (UF/script).read_text(encoding="utf-8")


def test_lock_functions(lock):
    uf_lock.check()
    with pytest.raises(ValueError, match="nothing to unlock"):
        uf_lock.check("BZ, some day")
    first = uf_lock.take(dict(out_dir="a"))
    assert first["event"] == "first_read" and lock.is_file()
    with pytest.raises(uf_lock.DevReadLocked, match="read once"):
        uf_lock.check()
    with pytest.raises(uf_lock.DevReadLocked, match="not read"):
        uf_lock.take(dict(out_dir="b"))
    for blank in ("", "  "):
        with pytest.raises(ValueError, match="words"):
            uf_lock.check(blank)
    uf_lock.check("BZ, 11-06 9 am: read it again")
    second = uf_lock.take(dict(out_dir="c"), bz_go="BZ, 11-06 9 am: read it again")
    assert second["event"] == "bz_approved_reread" and [r["out_dir"] for r in uf_lock.describe()] == ["a", "c"]


# ---- the reading ------------------------------------------------------------------------------------------------------------------------------------------------------

SUBJECTS = [f"OASIS_OAS1_{n:04d}_MR1" for n in range(438, 458)]


def row(seg35, seg4=.95):
    return dict(seg35=seg35, seg4=seg4, small_six=.8, big_six=.95, other_23=.94, cortex=.94, white_matter=.97, per_label=[seg35]*35, seg4_per_label=[seg4]*4)


def made_evaluation(*, a=.003, b=.0, v1=.926, n=20, control=0., u1_control=0., member_problems=(), gate=True, nn_weight=.5):
    """An eval_uf.json with every subject at v1's value plus the arm's offset; nothing of a real subject."""
    subjects = SUBJECTS[:n]
    base = {s: v1+.001*(i % 3-1) for i, s in enumerate(subjects)}
    rows = dict(UF4={s: row(base[s]+a) for s in subjects}, UF4_nn={s: row(base[s]+b) for s in subjects}, nn={s: row(base[s]) for s in subjects}, v1_control={s: row(base[s]+control) for s in subjects}, v2_control={s: row(base[s]+.0016) for s in subjects}, U1_control={s: row(base[s]+u1_control) for s in subjects},
                **{f"UF_{k}": {s: row(base[s]) for s in subjects} for k in ("A0", "A1", "B0", "B1")})
    references = dict(reference_v1={s: row(base[s]) for s in subjects}, reference_v2={s: row(base[s]+.0016) for s in subjects}, reference_U1={s: row(base[s]) for s in subjects}, **{f"reference_v2_{k}": {s: {m: row(base[s]-.004)[m] for m in ("seg35", "seg4", "small_six", "big_six", "other_23")} for s in subjects} for k in ("A0", "A1", "B0", "B1")})
    every = dict(rows, **references)
    means = {arm: {m: float(np.mean([r[s][m] for s in subjects])) for m in ("seg35", "seg4", "small_six", "big_six", "other_23") + (("cortex", "white_matter") if "cortex" in next(iter(r.values())) else ())} for arm, r in every.items()}
    controls = {arm: {m: means[arm][m]-means[ref][m] for m in ("seg35", "seg4")} for arm, ref in (("v1_control", "reference_v1"), ("v2_control", "reference_v2"), ("U1_control", "reference_U1"))}
    convert = dict(gate="passed" if gate else "failed", tolerance=1e-4, subjects=subjects, n_differing_by_subject={s: 0 for s in subjects}, n_voxels_by_subject={s: 6881280 for s in subjects}, probs36_sha256={s: "x" for s in subjects}, n_differing_total=0, n_voxels_total=6881280*n)
    members = {name: dict(stopped="max_iters", iterations_run=40000, checkpoint_iteration=40000, best_iteration=38000, best_calib_mean_dice35=.93, checkpoint_calib_mean_dice35=.929, wall_seconds=9000., temperature=1.3, memory={}) for name in EVAL.ENSEMBLE_ORDER}
    return dict(subjects=subjects, rows=rows, reference_rows=references, means=means, controls=dict(differences=controls, tolerance=dict(uc.CONTROL_TOLERANCES)), members=members, member_problems=list(member_problems), nn=dict(convert_record=convert), nn_weight=nn_weight,
                per_subject_extra={s: dict(voxels_b_differs_from_a=5) for s in subjects}, seconds={}, dev_read=dict(event="first_read"))


@pytest.mark.parametrize("delta,reading", [(.003, DECIDE.WORTH), (.0031, DECIDE.WORTH), (.0029999, DECIDE.NOT_CONNECT), (-.002, DECIDE.NOT_CONNECT), (float("nan"), DECIDE.NOT_DECIDABLE), (None, DECIDE.NOT_DECIDABLE)])
def test_reading_of(delta, reading):
    assert DECIDE.reading_of(delta) == reading


def test_the_primary_arm():
    assert DECIDE.primary_arm(dict(UF4=dict(seg35=.93), UF4_nn=dict(seg35=.931))) == "UF4_nn"
    assert DECIDE.primary_arm(dict(UF4=dict(seg35=.93), UF4_nn=dict(seg35=.93))) == "UF4"           # a tie goes to (a)
    assert DECIDE.primary_arm(dict(UF4=dict(seg35=.93), UF4_nn=dict(seg35=.929))) == "UF4"


@pytest.mark.parametrize("a,b,primary,reading", [(.0035, .001, "UF4", DECIDE.WORTH), (.001, .0035, "UF4_nn", DECIDE.WORTH), (.002, .0025, "UF4_nn", DECIDE.NOT_CONNECT), (.0025, .0025, "UF4", DECIDE.NOT_CONNECT), (.0032, -.01, "UF4", DECIDE.WORTH), (.0028, -.01, "UF4", DECIDE.NOT_CONNECT)])
def test_decide_each_branch(a, b, primary, reading):
    document = DECIDE.decide(made_evaluation(a=a, b=b))
    assert document["reading"] == reading and document["primary_arm"] == primary and document["delta_seg35"] == pytest.approx(max(a, b) if b > a else a, abs=1e-12)
    assert document["candidate"] == (primary if reading == DECIDE.WORTH else None)
    d = document["primary_minus_v1"]["seg35"]
    assert d["n"] == 20 and len(d["t95"]) == 2 and d["t95"][0] <= d["mean"] <= d["t95"][1] and d["bootstrap_draws"] == uc.BOOTSTRAP_DRAWS
    assert document["first5_pairs"]["reference_v1"]["seg35"] is not None and document["focus"]["UF4"]["subject"]["seg35"] is not None
    text = DECIDE.summary_markdown(DECIDE.uc_no_nan(document))
    assert DECIDE.READINGS[reading] in text and "seg4" in text and uc.FOCUS_SUBJECT in text


def test_the_delta_is_the_paired_mean_against_v1_not_v2():
    document = DECIDE.decide(made_evaluation(a=.004, b=0.))
    assert document["reading"] == DECIDE.WORTH and abs(document["delta_seg35"]-.004) < 1e-12 and abs(document["primary_minus_v2"]["seg35"]["mean"]-(.004-.0016)) < 1e-12
    assert DECIDE.decide(made_evaluation(a=.0045, b=0., v1=.926))["reading"] == DECIDE.WORTH and DECIDE.decide(made_evaluation(a=.0025, b=0.))["reading"] == DECIDE.NOT_CONNECT    # (a) - v2 would be +.0009: the line is on v1


@pytest.mark.parametrize("kwargs,match", [
    (dict(n=19), "19 名受试者"),
    (dict(control=3e-4), "控制不符：v1_control"),
    (dict(control=2e-9), "控制不符：v1_control"),                 # stored probabilities: 1e-9
    (dict(u1_control=3e-4), "控制不符：U1_control"),
    (dict(member_problems=["foldB_s1: stopped 'wall_budget'"]), "成员不是计划训的"),
    (dict(gate=False), "闸门没过"),
    (dict(nn_weight=.6), "权重"),
])
def test_decide_not_decidable(kwargs, match):
    document = DECIDE.decide(made_evaluation(**kwargs))
    assert document["reading"] == DECIDE.NOT_DECIDABLE and any(match in r for r in document["not_decidable_because"]) and document["delta_seg35"] is None


def test_a_missing_member_and_an_undefined_mean_are_not_decidable():
    evaluation = made_evaluation()
    del evaluation["members"]["foldB_s1"]
    assert any("成员缺失" in r for r in DECIDE.decide(evaluation)["not_decidable_because"])
    evaluation = made_evaluation()
    evaluation["means"]["UF4"]["seg35"] = float("nan")
    assert any("UF4 的 seg35 没有定义" in r for r in DECIDE.decide(evaluation)["not_decidable_because"])
    assert DECIDE.decide(None)["not_decidable_because"] == ["没有评测结果"]


def test_control_inside_its_tolerance_decides():
    assert DECIDE.decide(made_evaluation(control=5e-10))["reading"] != DECIDE.NOT_DECIDABLE and DECIDE.decide(made_evaluation(u1_control=1.9e-4))["reading"] != DECIDE.NOT_DECIDABLE
    assert DECIDE.decide(made_evaluation(control=1.9e-4))["reading"] == DECIDE.NOT_DECIDABLE                # what UL's 2e-4 would have let through


def test_the_tolerances_are_the_codes_and_every_control_is_required():
    evaluation = made_evaluation()
    evaluation["controls"]["tolerance"] = dict(v1_control=1., v2_control=1., U1_control=1.)
    evaluation["controls"]["differences"]["v1_control"]["seg35"] = 1e-3
    reasons = DECIDE.decide(evaluation)["not_decidable_because"]
    assert any("容差" in r and "不是本代码的" in r for r in reasons) and any("控制不符：v1_control 的 seg35" in r for r in reasons)
    evaluation = made_evaluation()
    del evaluation["controls"]["differences"]["U1_control"]["seg4"]
    del evaluation["controls"]["differences"]["v2_control"]
    reasons = DECIDE.decide(evaluation)["not_decidable_because"]
    assert "控制缺失：U1_control 的 seg4 没有差值" in reasons and "控制缺失：v2_control 的 seg35 没有差值" in reasons and "控制缺失：v2_control 的 seg4 没有差值" in reasons
    reasons = DECIDE.decide(made_evaluation(n=19, control=3e-4))["not_decidable_because"]
    assert any("19 名受试者" in r for r in reasons) and any("控制不符" in r for r in reasons)      # every reason, not only the first kind


def test_first5_is_the_mean_of_pair_means():
    evaluation = made_evaluation()
    source = evaluation["reference_rows"]["reference_v1"]
    want = np.mean([(source[a]["seg35"]+source[b]["seg35"])/2 for a, b in uc.FIRST5_PAIRS])
    assert DECIDE.first5(evaluation, "reference_v1") == pytest.approx(want, abs=1e-15)


def test_the_interval_is_the_t_formula_and_the_bootstrap_is_fixed():
    from scipy import stats
    x = np.random.default_rng(5).normal(.002, .002, 20)
    d = DECIDE.describe_difference(x)
    half = stats.t.ppf(.975, 19)*x.std(ddof=1)/np.sqrt(20)
    assert d["t95"] == pytest.approx([x.mean()-half, x.mean()+half]) and d["bootstrap95"] == DECIDE.describe_difference(x)["bootstrap95"] and d["n_positive"] == int((x > 0).sum())


def test_decision_files_never_overwritten_and_no_option_changes_the_reading(tmp_path, capsys):
    DECIDE.main(["--eval", str(tmp_path/"missing.json"), "--out-dir", str(tmp_path/"a")])
    assert json.loads((tmp_path/"a"/"decision_uf.json").read_text())["not_decidable_because"] == ["没有评测结果"]
    with pytest.raises(FileExistsError):
        DECIDE.main(["--eval", str(tmp_path/"missing.json"), "--out-dir", str(tmp_path/"a")])
    with pytest.raises(SystemExit):
        DECIDE.main(["--eval", str(tmp_path/"missing.json"), "--out-dir", str(tmp_path/"b"), "--reason", "x"])


def test_the_note_says_what_the_reading_does_not_say():
    assert "证伪不了什么" in DECIDE.NOTE and "分割级不等于配准后的 Dice" in DECIDE.NOTE and "这一组种子" in DECIDE.NOTE and "四个成员" in DECIDE.NOTE


# ---- the chain ---------------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_start_string():
    argv = RUN.launch_argv()
    text = " ".join(argv)
    assert argv[:4] == ["systemd-run", "--user", "--unit=tag-uf-chain", "--collect"] and "--property=MemoryMax=8G" in argv and f"--property=RuntimeMaxSec={RUN.runtime_max_seconds()}" in argv and "pytest" not in text
    assert f"--working-directory={uc.OUT}/snapshot" in argv and argv[-5:] == ["chain", "--out", str(uc.OUT), "--wait-minutes", "30"] and "seq-vp-" not in text
    assert RUN.runtime_max_seconds() == 4*(16200+1800)+3600+1800+7200+7200+7200+600+11*1800+1800


@pytest.mark.parametrize("change,match", [(lambda a: a+["pytest"], "test runner"), (lambda a: [x.replace("tag-uf-chain", "tag-ul-x") for x in a], "unit must be"), (lambda a: [x for x in a if not x.startswith("--property=RuntimeMaxSec")], "RuntimeMaxSec"), (lambda a: [x.replace("MemoryMax=8G", "MemoryMax=12G") for x in a], "MemoryMax"),
                                          (lambda a: [x.replace(f"--working-directory={uc.OUT}", "--working-directory=/tmp") for x in a], "working directory")])
def test_check_launch_refuses(change, match):
    with pytest.raises(ValueError, match=match):
        RUN.check_launch(change(RUN.launch_argv()))


def test_chain_order_and_contents():
    steps = RUN.chain_steps(RUN.Spec(out=uc.OUT, cwd=uc.OUT/"snapshot"))
    names = [n for n, _, _ in steps]
    assert names == ["control_check", "nn_infer", "nn_convert"]+[f"{k}_{m}" for m in ("foldA_s0", "foldB_s0", "foldA_s1", "foldB_s1") for k in ("train", "check", "calibrate")]+["check_members", "eval_uf", "decide_uf"]     # plan revision 1: nnU-Net before the training
    assert steps[0][2][2:5] == [str(uc.OUT/"snapshot/experiments/uf/eval_uf.py"), "--control-only", "--out-dir"] and "--bz-go" not in steps[0][2]
    train = " ".join(steps[3][2])
    assert steps[1][2][2:5] == [str(uc.OUT/"snapshot/experiments/uf/uf_nn.py"), "infer", "--out"] and steps[2][2][2:4] == [str(uc.OUT/"snapshot/experiments/uf/uf_nn.py"), "convert"]
    assert "--profile v3_large --fold A --seed 0 --out" in train and "--set max_iters=40000 --set min_iters=40000 --set patience=40000 --set eval_every=500" in train and "--set checkpoint=\"last\"" in train and "--set max_wall_seconds=14400.0" in train and "--synthetic" not in train and "--device cuda" in train
    assert " ".join(steps[12][2]).endswith("--profile v3_large --fold B --seed 1 --out /home/rmcfz/seq_vp_tag_results/seg/unet35_v4_full --device cuda --set max_iters=40000 --set min_iters=40000 --set patience=40000 --set eval_every=500 --set checkpoint=\"last\" --set max_wall_seconds=14400.0")
    kinds = {n: k for n, k, _ in steps}
    assert [n for n in names if kinds[n] in RUN.GPU_STEPS] == ["control_check", "nn_infer"]+[n for n in names if n.startswith(("train_", "calibrate_"))]+["eval_uf"]
    assert all(a is None for n, k, a in steps if k == "check") and "pytest" not in " ".join(" ".join(a) for _, _, a in steps if a)
    assert steps[-2][2][steps[-2][2].index("--out-dir")+1] == str(uc.OUT/"eval_uf") and "--bz-go" not in " ".join(steps[-2][2])


def test_run_step_once_and_a_started_step_is_not_rerun(tmp_path):
    calls = []
    RUN.run_step("a", lambda: calls.append(1) or 0, state=tmp_path/"s", log_path=tmp_path/"l")
    RUN.run_step("a", lambda: calls.append(1) or 0, state=tmp_path/"s", log_path=tmp_path/"l")
    assert calls == [1]
    with pytest.raises(RUN.Stop, match=r"failed \(exit 3\)"):
        RUN.run_step("b", lambda: 3, state=tmp_path/"s", log_path=tmp_path/"l")
    with pytest.raises(RUN.Stop, match="did not finish"):
        RUN.run_step("b", lambda: 0, state=tmp_path/"s", log_path=tmp_path/"l")
    assert (tmp_path/"s"/"b.started").is_file() and not (tmp_path/"s"/"b.done").exists()


def test_run_command_kills_the_group_on_the_time_limit(tmp_path):
    script = tmp_path/"s.sh"
    script.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {tmp_path/'child.pid'}\nsleep 60\n")
    started = time.monotonic()
    code, timed_out = RUN.run_command(["/bin/sh", str(script)], cwd=tmp_path, log_path=tmp_path/"log", timeout=1)
    assert timed_out and code != 0 and time.monotonic()-started < 30
    child = int((tmp_path/"child.pid").read_text())
    for _ in range(50):
        try:                                     # the test's own observation (A5b follow-up): the child may be reaped between a check and the read, so a missing /proc
            state = Path(f"/proc/{child}/stat").read_text().split()[2]       # entry at the read is "gone"; the assertion is the same: killed with its group, on time
        except (FileNotFoundError, ProcessLookupError):
            break
        if state == "Z":
            break
        time.sleep(.1)
    else:
        pytest.fail("the background child of the step survived the time limit")
    assert RUN.run_command(["/bin/sh", "-c", "exit 4"], cwd=tmp_path, log_path=tmp_path/"log", timeout=10) == (4, False)


def test_a_hold_file_put_while_the_gate_waits_stops_the_step(tmp_path, monkeypatch):
    spec = RUN.Spec(out=tmp_path, cwd=tmp_path)
    monkeypatch.setattr(RUN, "gate_problems", lambda: (tmp_path/"hold").write_text("x") and [])
    with pytest.raises(RUN.Stop, match="hold"):
        RUN.run_chain(spec)
    assert not (tmp_path/"chain_state").exists() or not list((tmp_path/"chain_state").iterdir())


def test_every_end_of_the_chain_is_recorded_with_the_raw_reminder(tmp_path):
    spec = RUN.Spec(out=tmp_path, cwd=tmp_path, gate=False)
    (tmp_path/"hold").write_text("x")
    with pytest.raises(RUN.Stop):
        RUN.run_chain(spec)
    first = json.loads((tmp_path/"queue_end.json").read_text())
    assert "hold" in first["stop"] and first["finished"] is False and first["nn_raw"] is None
    (tmp_path/"nn"/"raw").mkdir(parents=True)
    (tmp_path/"nn"/"raw"/"a.npz").write_bytes(b"x"*1000)
    with pytest.raises(RUN.Stop):
        RUN.run_chain(spec)
    later = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("queue_end_*.json"))]
    assert len(later) == 1 and later[0]["nn_raw"]["bytes"] == 1000 and "nnU-Net raw 待总负责人清理" in later[0]["nn_raw"]["text"] and first == json.loads((tmp_path/"queue_end.json").read_text())


def test_atomic_writes_never_overwrite_and_leave_no_tmp(tmp_path):
    uc.write_json_atomic(tmp_path/"a.json", dict(x=1))
    with pytest.raises(FileExistsError):
        uc.write_json_atomic(tmp_path/"a.json", dict(x=2))
    assert json.loads((tmp_path/"a.json").read_text()) == dict(x=1) and not list(tmp_path.glob("*.tmp"))


def test_the_hold_file_stops_before_the_next_step(tmp_path):
    spec = RUN.Spec(out=tmp_path, cwd=tmp_path, gate=False)
    (tmp_path/"hold").write_text("x")
    with pytest.raises(RUN.Stop, match="hold"):
        RUN.run_chain(spec)
    assert not (tmp_path/"chain_state").exists() or not list((tmp_path/"chain_state").iterdir())


def test_the_gate_is_uls_and_waits(monkeypatch):
    import run_ul
    seen = []
    monkeypatch.setattr(run_ul, "gate_problems", lambda unit, **kwargs: seen.append(unit) or [])
    assert RUN.gate_problems() == [] and seen == ["tag-uf-chain"]
    answers = [["x"], ["y"], []]
    naps = []
    RUN.wait_for_gate(5, problems=lambda: answers.pop(0), sleep=naps.append, poll=1)
    assert len(naps) == 2 and not answers
    with pytest.raises(RUN.Stop, match="stayed closed"):
        RUN.wait_for_gate(0, problems=lambda: ["busy"], sleep=naps.append, poll=1)


def test_a_gpu_step_waits_for_the_gate_and_a_closed_gate_starts_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, "gate_problems", lambda: ["GPU busy"])
    spec = RUN.Spec(out=tmp_path, cwd=tmp_path, wait_minutes=0)
    with pytest.raises(RUN.Stop, match="stayed closed"):
        RUN.run_chain(spec)
    assert not (tmp_path/"chain_state").exists() or not list((tmp_path/"chain_state").iterdir())


def test_the_estimate_from_a_measurement():
    measured = dict(step_seconds_median=.2, eval_seconds_median=2., data_ready_seconds=30., wall_per_step=.226)
    e = RUN.estimate(measured)
    assert e["per_member_hours_parts"] == pytest.approx((30+40000*.2+80*2.)/3600) and e["per_member_hours_whole"] == pytest.approx(40000*.226/3600) and e["per_member_hours"] == e["per_member_hours_whole"]
    assert e["wall_stop_over_estimate"] == pytest.approx(14400/(40000*.226))


def test_snapshot_paths_of_the_chain():
    assert set(RUN.SNAPSHOT_PATHS_UF)-set(SNAPSHOT_PATHS_V3) == set(uc.SNAPSHOT_EXTRA)
    assert all((CODE_ROOT/p).exists() for p in RUN.SNAPSHOT_PATHS_UF) and not [p for p in RUN.SNAPSHOT_PATHS_UF if "holdout" in p or p.startswith("splits/") and p not in ("splits/bayesreg_v1/folds.json", "splits/bayesreg_v1/train_subjects.txt", "splits/bayesreg_v1/folds_full_v2.json")]


@pytest.fixture
def blocked_env(tmp_path):
    bin_dir = tmp_path/"bin"
    bin_dir.mkdir()
    shim = bin_dir/"systemd-run"
    marker = tmp_path/"systemd_run_called"
    shim.write_text(f"#!/bin/sh\necho called >> {marker}\nexit 1\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"), marker


@pytest.mark.parametrize("command", [["chain"], ["members-plan"]])
def test_the_real_entry_stops_outside_a_snapshot(tmp_path, blocked_env, command):
    env, marker = blocked_env
    result = subprocess.run([sys.executable, "-B", str(UF/"run_uf.py"), *command, "--out", str(tmp_path/"out")], env=env, capture_output=True, text=True, timeout=120, cwd=str(CODE_ROOT))
    assert result.returncode == 8 and "STOP" in result.stdout and "snapshot" in result.stdout
    assert not marker.exists() and not (tmp_path/"out").exists()


def test_print_launch_starts_nothing(blocked_env):
    env, marker = blocked_env
    result = subprocess.run([sys.executable, "-B", str(UF/"run_uf.py"), "print-launch"], env=env, capture_output=True, text=True, timeout=120, cwd=str(CODE_ROOT))
    assert result.returncode == 0 and result.stdout.startswith("systemd-run --user --unit=tag-uf-chain") and not marker.exists()


def test_the_chain_refuses_to_start_when_the_lock_exists(tmp_path, lock, monkeypatch):
    monkeypatch.setattr(RUN, "require_snapshot", lambda: None)
    lock.parent.mkdir(parents=True)
    lock.write_text("{}\n")
    started = []
    monkeypatch.setattr(RUN, "run_chain", lambda spec: started.append(spec))
    with pytest.raises(uf_lock.DevReadLocked):
        RUN.cmd_chain(types.SimpleNamespace(out=tmp_path/"out", wait_minutes=30))
    assert not started


# ---- one synthetic world through the whole chain ------------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world(tmp_path_factory):
    torch.set_num_threads(1)
    work = tmp_path_factory.mktemp("uf_world")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    synthetic = RUN.build_synthetic(work, env)
    RUN.members_plan(synthetic.spec, load_full_folds(work/"folds_full_v2.json"))
    RUN.run_chain(synthetic.spec)
    return synthetic


def test_the_synthetic_chain_reads_like_the_plan(world):
    out = world.spec.out
    end = json.loads((out/"queue_end.json").read_text())
    assert end["stop"] is None and end["finished"] is True and end["reading"] == DECIDE.NOT_DECIDABLE and end["nn_raw"]["path"] == str(out/"nn"/"raw") and end["nn_raw"]["bytes"] > 0
    assert "nnU-Net raw 待总负责人清理" in (out/"analysis"/"summary_uf.md").read_text()
    order = [line.split()[2] for line in (out/"chain.log").read_text().splitlines() if " start " in line and line.split()[1] == "start"]
    assert order[:4] == ["control_check:", "nn_infer:", "nn_convert:", "train_foldA_s0:"]
    evaluation = json.loads((out/"eval_uf"/"eval_uf.json").read_text())
    decision = json.loads((out/"analysis"/"decision_uf.json").read_text())
    assert decision["reading"] == DECIDE.NOT_DECIDABLE and decision["not_decidable_because"] == ["2 名受试者，不是 20"]
    assert evaluation["member_problems"] == [] and sorted(evaluation["members"]) == sorted(uc.MEMBER_NAMES) and all(m["checkpoint_iteration"] == 4 and m["checkpoint_choice"] == "last" for m in evaluation["members"].values())
    assert all(v == 0. for d in evaluation["controls"]["differences"].values() for v in d.values()) and evaluation["nn"]["convert_record"]["gate"] == "passed"
    assert set(evaluation["rows"]) == set(EVAL.ARMS) and evaluation["ensemble_order"] == ["foldA_s0", "foldA_s1", "foldB_s0", "foldB_s1"] and evaluation["dev_read"] is None
    for name in uc.MEMBER_NAMES:
        assert json.loads((out/name/"uf_check.json").read_text())["problems"] == []


def control_argv(world, out_dir, ensembles=None):
    w = world
    return ["--control-only", "--synthetic", "--out-dir", str(out_dir), "--device", "cpu", "--nn-manifest", str(w.nn_out/"prepare_manifest.json"), "--v1-pred", str(w.work/"pred_v1"), "--v2-pred", str(w.v2_root/"pred"), "--u1-member", str(w.u1_root/"foldA_s0"),
            "--reference-ensembles", str(ensembles or w.work/"ref_ensembles"/"eval_seg.json"), "--reference-members", str(w.work/"ref_members.json"), "--reference-u1", str(w.work/"ref_u1.json"), "--allowlist", str(w.allowlist), "--folds", str(w.work/"folds.json"), "--dev-source", "synthetic_dev", "--data-root", str(w.data)]


def test_the_control_before_the_training_passed_and_took_no_lock(world, lock, tmp_path):
    document = json.loads((world.spec.out/"control_check"/"eval_control.json").read_text())
    assert document["passed"] is True and all(v == 0. for d in document["differences"].values() for v in d.values()) and sorted(document["differences"]) == sorted(EVAL.CONTROL_ARMS)
    lock.parent.mkdir(parents=True)
    lock.write_text("{}\n")
    assert EVAL.main(control_argv(world, tmp_path/"c")) == 0 and lock.read_text() == "{}\n"            # the control does not look at the lock: it adds nothing about UF
    with pytest.raises(SystemExit):
        EVAL.main(control_argv(world, tmp_path/"d")+["--bz-go", "BZ"])


def test_a_control_that_does_not_reproduce_stops_with_exit_8(world, tmp_path):
    document = json.loads((world.work/"ref_ensembles"/"eval_seg.json").read_text())
    subject = document["subjects"][0]
    document["rows"][subject]["v1"]["seg35"] += .01
    document["means"]["v1"]["seg35"] = float(np.mean([document["rows"][s]["v1"]["seg35"] for s in document["subjects"]]))
    tampered = tmp_path/"eval_seg.json"
    tampered.write_text(json.dumps(document))
    assert EVAL.main(control_argv(world, tmp_path/"c", ensembles=tampered)) == 8
    assert json.loads((tmp_path/"c"/"eval_control.json").read_text())["passed"] is False


def test_the_ensemble_is_v2s_average_of_the_members(world):
    """(a) on a synthetic subject equals the plain mean of softmax(z/T) over the four members, float16, argmax; (b) equals 1/2-1/2 with the stored nnU-Net probabilities."""
    import eval_u1
    from bayesreg.seg.unet.data import normalize_image, read_image
    from bayesreg.seg.unet.train import predict_logits
    subject = sorted(json.loads((world.spec.out/"eval_uf"/"eval_uf.json").read_text())["subjects"])[0]
    values, _ = read_image(OasisFiles(world.data), subject)
    total = None
    for name in EVAL.ENSEMBLE_ORDER:
        model, blob, _ = eval_u1.load_arm(world.spec.out/name, "cpu")
        image, _ = normalize_image(values, percentile=blob["percentile"])
        z = predict_logits(model, image, device="cpu", amp=False, window=tuple(blob["window"]), gaussian_sigma=blob["window_gaussian_sigma"])
        p = torch.softmax(z/blob["temperature"], dim=1)[0]
        total = p if total is None else total+p
    mean = (total/4)
    mean = (mean/mean.sum(dim=0, keepdim=True)).numpy().astype(np.float16)
    true35 = SEG.read_truth(world.data, subject, "seg35")[0]
    true4 = SEG.read_truth(world.data, subject, "seg4")[0]
    evaluation = json.loads((world.spec.out/"eval_uf"/"eval_uf.json").read_text())
    want = SEG.summarize_subject(mean.argmax(axis=0).astype(np.uint8), true35, true4, SEG.load_seg4_lookup())
    assert evaluation["rows"]["UF4"][subject]["seg35"] == want["seg35"]
    p_nn, _ = read_back_probabilities(world.spec.out/"nn"/"probs"/subject/"probs36.npz")
    a = mean.astype(np.float64)
    mix = (.5*a/a.sum(axis=0)+.5*p_nn).argmax(axis=0).astype(np.uint8)
    assert evaluation["rows"]["UF4_nn"][subject]["seg35"] == SEG.summarize_subject(mix, true35, true4, SEG.load_seg4_lookup())["seg35"]


def eval_kwargs(world, out_dir, **changes):
    w = world
    kwargs = dict(out_dir=out_dir, member_root=w.spec.out, nn_manifest=w.nn_out/"prepare_manifest.json", v1_pred=w.work/"pred_v1", v2_pred=w.v2_root/"pred", u1_member=w.u1_root/"foldA_s0", references=dict(ensembles=w.work/"ref_ensembles"/"eval_seg.json", members=w.work/"ref_members.json", u1=w.work/"ref_u1.json"),
                  data_root=w.data, allowlist_path=w.allowlist, folds_path=w.work/"folds.json", dev_source="synthetic_dev", v2_root=w.v2_root, v2_registry=w.work/"v2_registry.json", synthetic=True, train_overrides=dict(RUN.TINY_UF), device="cpu", log=quiet)
    kwargs.update(changes)
    return kwargs


@pytest.fixture
def real_like(world, lock, monkeypatch):
    """The real path of evaluate on the synthetic world: only the pins point at it. No real subject."""
    original = EVAL.read_inputs
    monkeypatch.setattr(EVAL, "read_inputs", lambda **kw: original(**dict(kw, synthetic=True)))
    monkeypatch.setattr(EVAL, "require_clean_code", lambda code: code)
    monkeypatch.setattr(uc, "U1_MEMBER_SHA256", EVAL.eval_u1.member_files(world.u1_root/"foldA_s0")[1])
    monkeypatch.setattr(uc, "TRAIN_CONFIG", dict(RUN.TINY_UF))
    monkeypatch.setattr(uc, "V2_ROOT", world.v2_root)
    monkeypatch.setattr(uc, "V2_REGISTRY_REL", str(world.work/"v2_registry.json"))       # absolute: REPO / it is it
    return lock


def real_kwargs(world, out_dir, **changes):
    return eval_kwargs(world, out_dir, **dict(dict(synthetic=False, train_overrides=None, v2_registry=None), **changes))


def test_the_synthetic_path_does_not_touch_the_lock(world, lock, tmp_path):
    EVAL.evaluate(**eval_kwargs(world, tmp_path/"e"))
    assert not lock.exists()
    with pytest.raises(ValueError, match="bz-go"):
        EVAL.evaluate(**eval_kwargs(world, tmp_path/"f", bz_go="BZ"))


def test_the_real_path_locks_before_the_first_truth_and_reads_once(world, real_like, tmp_path, monkeypatch):
    original = SEG.read_truth
    states = []
    monkeypatch.setattr(SEG, "read_truth", lambda *a, **k: (states.append(real_like.exists()), original(*a, **k))[1])
    document = EVAL.evaluate(**real_kwargs(world, tmp_path/"e1"))
    assert states and all(states) and document["dev_read"]["event"] == "first_read" and sorted(document["dev_read"]["members"]) == sorted(uc.MEMBER_NAMES)
    opened, image = [], EVAL.read_image
    monkeypatch.setattr(SEG, "read_truth", lambda *a, **k: opened.append(a))
    monkeypatch.setattr(EVAL, "read_image", lambda *a, **k: opened.append(a))
    with pytest.raises(uf_lock.DevReadLocked, match="read once"):
        EVAL.evaluate(**real_kwargs(world, tmp_path/"e2"))
    with pytest.raises(uf_lock.DevReadLocked, match="read once"):
        EVAL.evaluate(**real_kwargs(world, tmp_path/"e3", member_root=tmp_path/"another_root"))
    assert not opened and not (tmp_path/"e2").exists()
    monkeypatch.setattr(SEG, "read_truth", original)
    monkeypatch.setattr(EVAL, "read_image", image)
    again = EVAL.evaluate(**dict(real_kwargs(world, tmp_path/"e4"), bz_go="BZ, 11-06: read it again"))
    assert again["dev_read"]["event"] == "bz_approved_reread" and len(uf_lock.describe()) == 2


@pytest.mark.parametrize("damage,match", [("member", "not evaluated, the truth is not read"), ("gate", "not evaluated, the truth is not read"), ("probs", "not evaluated, the truth is not read"), ("last", "model_last.pt missing or its sha256"), ("record", "not the file member.json was written from")])
def test_a_wrong_member_or_gate_costs_no_read(world, real_like, tmp_path, monkeypatch, damage, match):
    opened = []
    monkeypatch.setattr(SEG, "read_truth", lambda *a, **k: opened.append(a))
    root = tmp_path/"root"
    import shutil
    shutil.copytree(world.spec.out, root, symlinks=True, ignore=shutil.ignore_patterns("eval_uf", "analysis"))
    if damage == "member":
        record = json.loads((root/"foldB_s1"/"train_record.json").read_text())
        record["stopped"] = "wall_budget"
        (root/"foldB_s1"/"train_record.json").write_text(json.dumps(record))
    elif damage == "last":
        with (root/"foldA_s1"/"model_last.pt").open("ab") as stream:
            stream.write(b"x")
    elif damage == "record":
        path = root/"foldB_s0"/"train_record.json"
        path.write_text(path.read_text()+" ")                       # the content as before, the file not
    elif damage == "gate":
        record = json.loads((root/"nn"/"convert_record.json").read_text())
        record["n_differing_by_subject"] = {s: 10**6 for s in record["subjects"]}
        (root/"nn"/"convert_record.json").write_text(json.dumps(record))
    else:
        path = sorted((root/"nn"/"probs").glob("*/probs36.npz"))[0]
        path.write_bytes(path.read_bytes()+b"x")
    with pytest.raises(ValueError, match=match):
        EVAL.evaluate(**real_kwargs(world, tmp_path/"e", member_root=root))
    assert not opened and not real_like.exists()


def test_a_subject_in_nnunets_label_contact_is_refused_before_any_file(world, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(EVAL, "read_image", lambda *a, **k: opened.append(a))
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps(dict(label_contact_subjects=["OASIS_OAS1_0907_MR1", "OASIS_OAS1_0901_MR1"])))
    with pytest.raises(SubjectSealed, match="label contact of nn"):
        EVAL.evaluate(**eval_kwargs(world, tmp_path/"e", nn_manifest=manifest))
    assert not opened


def test_real_evaluation_refuses_other_inputs(world, lock, tmp_path, monkeypatch):
    monkeypatch.setattr(EVAL, "require_clean_code", lambda code: code)
    with pytest.raises(ValueError, match="the plan's"):
        EVAL.evaluate(**eval_kwargs(world, tmp_path/"e", synthetic=False, train_overrides=None))
    assert not lock.exists()


def _lock_state(path):
    """(sha256 of the bytes, mtime in ns) of the real read lock, or None when there is none.  Only read here, never written."""
    path = Path(path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def test_the_dry_run_end_to_end(tmp_path, blocked_env):
    """Two branches on the state of the real read lock (docs/bayesreg/reviews/HI_工具_复核_20261107.md §6, suggestion (a)).  UF's evaluation wrote the lock once on 2026-11-06 by design, and the dry run rightly reports
    it as a problem (the real chain would be refused).  Lock there: exit 8, the problems line is exactly "the read lock exists" (every synthetic step of the dry run passed), and the lock's bytes and mtime are
    unchanged.  No lock: as before, exit 0, no problem, and still no lock afterwards.  In both: the start string, no systemd-run, nothing of pytest in the output."""
    env, marker = blocked_env
    before = _lock_state(uf_lock.DEV_READ_LOCK)
    result = subprocess.run([sys.executable, "-B", str(UF/"run_uf.py"), "--dry-run", "--scratch", str(tmp_path/"work")], env=env, capture_output=True, text=True, timeout=900, cwd=str(CODE_ROOT))
    problems_lines = [line for line in result.stdout.splitlines() if line.startswith("== dry run problems: ")]
    assert len(problems_lines) == 1, result.stdout[-4000:]+result.stderr[-3000:]
    if before is None:
        assert result.returncode == 0, result.stdout[-4000:]+result.stderr[-3000:]
        assert problems_lines[0] == f"== dry run problems: none; work {tmp_path/'work'}"
        assert not uf_lock.DEV_READ_LOCK.exists()
    else:
        assert result.returncode == 8, result.stdout[-4000:]+result.stderr[-3000:]
        assert problems_lines[0] == f"== dry run problems: the read lock exists; work {tmp_path/'work'}", problems_lines[0]
        assert f"== the read lock of the development truth, {uf_lock.DEV_READ_LOCK}: EXISTS" in result.stdout
        assert _lock_state(uf_lock.DEV_READ_LOCK) == before
    assert "--unit=tag-uf-chain" in result.stdout and "pytest" not in result.stdout.replace(str(tmp_path), "")
    assert not marker.exists()
