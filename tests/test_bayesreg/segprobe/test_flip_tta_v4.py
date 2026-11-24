"""TTA (runs/20261107-tta/plan.md, 任务书.md): the v4 option of experiments/segprobe/flip_tta.py, UF4 with the left-right flip. CPU, synthetic volumes and tiny made-up members in tmp_path only; the real repository files read
are the label table, the seg4 lookup and the plan's path (its sha256). No image, label, member or prediction of a results directory; the read lock is a file under tmp_path.

- the inference: the arm unflipped is hs_seg.uf4_probabilities bit by bit; each member's own amp, window, Gaussian window and temperature in both directions, the flip pass with the unflipped pass's parameters on the mirror;
- the flip premise from train_record.json (each refusal, before any image), the registered predictions' provenance;
- the control (mean, agreement, both, a missing prediction) and the decision 过门 / 不过门 / 不判;
- the read lock: there before the run, appearing before the first truth label; the truth only after it;
- one run end to end; the dry run; the v1 and v2 outputs and the command line as before.
"""
import json
from pathlib import Path
import types

import numpy as np
import pytest
import torch

import bayesreg.seg.unet.data as data_module
import bayesreg.seg.unet.train as train_module
from bayesreg.io.guard import load_dev_allowlist
from bayesreg.io.pair import REGISTRY_SCHEMA, load_segmenter_registry, subject_list_sha256
from bayesreg.seg.unet import synthetic
from bayesreg.seg.unet.calibrate import MEMBER_FORMAT
from bayesreg.seg.unet.data import IMAGE_FILE, LABEL_FILES, OasisFiles, normalize_image, read_image
from bayesreg.seg.unet.infer import load_member
from bayesreg.seg.unet.large import left_right_label_map
from bayesreg.seg.unet.model import build_model
from bayesreg.seg.unet.registry import find_member_files
from bayesreg.seg.unet.train import file_sha256
from jd_curl.three_d.nifti import load_scalar

from .test_flip_tta import SideOracle, _truth_from, blob, common, eval_seg, flip_tta, member_of

SUBJECTS = [synthetic.oasis(901), synthetic.oasis(902)]
CONTACT = [synthetic.oasis(910), synthetic.oasis(911), synthetic.oasis(912)]
SHAPE_XYZ = (12, 16, 16)                       # stored (nz, ny, nx) = (16, 16, 12): the flip axis x is the short one
TINY = dict(base_channels=4, depth=2)
# (fold, seed, temperature, window, Gaussian sigma, amp), in HS's order A0 A1 B0 B1: every member different, so that a parameter of the wrong member is seen
MEMBERS = (("A", 0, 1.3, [8, 8, 8], .125, True), ("A", 1, .7, [16, 8, 8], .25, False), ("B", 0, 1.1, [8, 16, 12], .125, False), ("B", 1, 2., [16, 16, 8], .5, True))
NAMES = [f"fold{f}_s{s}" for f, s, *_ in MEMBERS]
REAL_PREDICT = train_module.predict_logits


def quiet(*args, **kwargs):
    pass


def refuse(*args, **kwargs):
    raise AssertionError("must not be called")


def write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=1)+"\n", encoding="utf-8")


def write_member(root, fold, seed, temperature, window, sigma, amp):
    torch.manual_seed(100+2*"AB".index(fold)+seed)
    model = build_model(TINY)
    directory = Path(root)/f"fold{fold}_s{seed}"
    directory.mkdir(parents=True)
    torch.save(dict(format=MEMBER_FORMAT, state_dict=model.state_dict(), model_config=dict(TINY), temperature=temperature, fold=fold, seed=seed, best_iteration=1, label_contact_subjects=list(CONTACT),
                    label_contact_subjects_sha256=subject_list_sha256(CONTACT), model_best_sha256="0"*64, percentile=99.5, window=list(window), amp=amp, window_gaussian_sigma=sigma, checkpoint_choice="last",
                    checkpoint_iteration=1), directory/"member.pt")
    _, pairs, unpaired = left_right_label_map()
    write_json(directory/"train_record.json", dict(format="bayesreg.unet35.train_record.v1", recipe="v3_large", fold=fold, seed=seed, iterations_run=4, config=dict(flip_probability=.5, flip_axis_check=True, batch_size=2),
                                                    flip_check=dict(check="left labels at low x, x separates them most", n_subjects=3, smallest_x_separation_vox=20., min_separation_vox=10.),
                                                    left_right_pairs=[[int(a), int(b)] for a, b in pairs], unpaired_labels=[int(u) for u in unpaired]))
    (directory/"train_log.jsonl").write_text(json.dumps(dict(event="start"))+"\n"+json.dumps(dict(event="end", steps_done=4, flips=5))+"\n", encoding="utf-8")
    write_json(directory/"member.json", dict(format="bayesreg.unet35.member_record.v1", member_file="member.pt", member_sha256=file_sha256(directory/"member.pt"), train_record_sha256=file_sha256(directory/"train_record.json")))
    return file_sha256(directory/"member.pt")


def world(tmp_path, *, tamper=None, members=MEMBERS):
    """The synthetic UF4: two subjects' images, four tiny members of segmenter unet35_v4_ens in two fold models, their registry, a synthetic allowlist, and the registered predictions as HS's seg phase writes
    them (hs_seg.uf4_probabilities, float16, provenance with this registry and these members). tamper(member directory of foldB_s1) runs after the members are written."""
    data = synthetic.write_subjects(tmp_path/"data", SUBJECTS, shape_xyz=SHAPE_XYZ)
    models_root = tmp_path/"members"
    shas = {f"fold{f}_s{s}": write_member(models_root, f, s, t, w, g, a) for f, s, t, w, g, a in members}
    if tamper is not None:
        tamper(models_root/"foldB_s1")
    registry_path = tmp_path/"segmenters_tta_test.json"
    write_json(registry_path, dict(schema=REGISTRY_SCHEMA, comment="synthetic", models={f"unet35_v4_ens_fold{f}": dict(segmenter_id="unet35_v4_ens", fold=i, member_model_sha256=[shas[f"fold{f}_s0"], shas[f"fold{f}_s1"]],
                                                                                                                  label_contact_subjects=list(CONTACT), label_contact_subjects_sha256=subject_list_sha256(CONTACT)) for i, f in enumerate("AB")}))
    registry = load_segmenter_registry(registry_path)
    allowlist_path = tmp_path/"allowlist.json"
    write_json(allowlist_path, synthetic.allowlist_document([("synthetic_dev", SUBJECTS, False), ("synthetic_train", CONTACT, False)]))
    allowlist = load_dev_allowlist(allowlist_path)
    hs = flip_tta.hs_seg_module()
    available = find_member_files(models_root)
    plans = {s: hs.uf_plan(registry, "UF4", s, allowlist, available) for s in SUBJECTS}
    files = OasisFiles(root=data)
    pred_root = tmp_path/"pred"
    for subject in SUBJECTS:
        plan, models, _ = plans[subject]
        probs, meta, _, members = hs.uf4_probabilities(subject, files=files, registry=registry, models=models, plan=plan, device="cpu")
        directory = pred_root/subject
        directory.mkdir(parents=True)
        np.savez_compressed(directory/"probs36.npz", p=probs.astype(np.float16))
        write_json(directory/"provenance.json", dict(subject=subject, layer="structure", segmenter_id="unet35_v4_ens", policy="all_folds_mean", test_labels_seen=False, registry_sha256=registry.sha256,
                                                     members=[dict(member_sha256=m["member_sha256"]) for m in members], input_sha256={IMAGE_FILE: meta["sha256"]},
                                                     output_sha256={"probs36.npz": file_sha256(directory/"probs36.npz")}, environment=dict(cudnn_benchmark=False, cudnn_deterministic=False)))
    spec = dict(id="unet35_v4_ens", registry=registry_path, models_root=models_root, reference=(.9, .9), pred_root=pred_root, candidate="UF4")
    return types.SimpleNamespace(data=data, files=files, registry=registry, allowlist=allowlist, plans=plans, spec=spec, shas=shas, pred_root=pred_root)


@pytest.fixture
def w(tmp_path):
    return world(tmp_path)


def synthetic_truth(data_root):
    lookup = eval_seg.load_seg4_lookup()

    def read_truth(root, subject, kind):
        labels = load_scalar(Path(data_root)/subject/LABEL_FILES["seg35"], labels=True).values.astype(np.uint8)
        return (labels if kind == "seg35" else lookup[labels].astype(np.uint8)), "e"*64
    return read_truth


def patch_admission(monkeypatch, w):
    monkeypatch.setattr(common, "admit_dev_subjects", lambda **kwargs: (list(SUBJECTS), dict(code=dict(commit="c"*40, clean=True), dev_source="synthetic")))
    monkeypatch.setattr(flip_tta, "load_dev_allowlist", lambda path: w.allowlist)


def recorder(calls):
    def predict(model, image, **kwargs):
        calls.append(dict(image=np.array(image, copy=True), **kwargs))
        return REAL_PREDICT(model, image, **kwargs)
    return predict


# ---- the specification ---------------------------------------------------------------------------------------------------------------------------

def test_v4_is_uf4_as_hs_registered_it_and_v1_v2_are_unchanged():
    v4 = flip_tta.SEGMENTERS["v4"]
    assert v4["id"] == "unet35_v4_ens" and v4["candidate"] == "UF4" and v4["registry"] == flip_tta.REPO/"configs/bayesreg/segmenters_hs_v1.json"
    assert v4["models_root"] == Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_full") and v4["pred_root"] == Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_ens/pred")
    assert v4["reference"] == (0.9300092857879709, 0.9605822402722083)                  # eval_uf.json means.UF4 (任务书 范围 1)
    hs = flip_tta.hs_seg_module()
    assert hs.hc.CANDIDATES[v4["candidate"]]["segmenter_id"] == v4["id"] and hs.hc.REGISTRY_REL == "configs/bayesreg/segmenters_hs_v1.json"
    assert flip_tta.SEGMENTERS["v1"] == dict(id="unet35_v1_ens", registry=flip_tta.REPO/"configs/bayesreg/segmenters.json", models_root=Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v1"), reference=(.9259821660395682, .9583579541106246))
    assert flip_tta.SEGMENTERS["v2"] == dict(id="unet35_v2_full", registry=flip_tta.REPO/"configs/bayesreg/segmenters_v2.json", models_root=Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full"), reference=(.9275412471659505, .9590838826390028))
    assert (flip_tta.WORTH_GAIN, flip_tta.WORTH_POSITIVE, flip_tta.CONTROL_TOL, flip_tta.AGREEMENT_MIN, flip_tta.N_DEV) == (.002, 15, 5e-4, .9999, 20)
    assert flip_tta.TTA_DEV_READ_LOCK == Path("/home/rmcfz/seq_vp_tag_results/seg/tta_dev_read.lock") and flip_tta.V4_OUT == Path("/home/rmcfz/seq_vp_tag_results/segprobe_v1/tta_v4")
    assert flip_tta.METRICS_V4 == ("seg35", "seg4", "small_six", "big_six", "other_23", "cortex", "white_matter")


def test_the_old_paths_refuse_v4():
    with pytest.raises(ValueError, match="run_v4"):
        flip_tta.run(names=["v4"], out_dir="/nonexistent", device="cpu")
    with pytest.raises(ValueError, match="dry_run_v4"):
        flip_tta.dry_run(["v1", "v4"], [], allowlist=None, real_members=False)


# ---- the inference -------------------------------------------------------------------------------------------------------------------------------

def test_the_arm_unflipped_is_hs_segs_uf4_probabilities_bit_by_bit(w):
    hs = flip_tta.hs_seg_module()
    for subject in SUBJECTS:
        plan, models, _ = w.plans[subject]
        with torch.no_grad():
            mine, flip_only, tta, record = flip_tta.v4_probabilities(subject, files=w.files, registry=w.registry, models=models, plan=plan, device="cpu", lut=flip_tta.swap_table())
        theirs, meta, norm, members = hs.uf4_probabilities(subject, files=w.files, registry=w.registry, models=models, plan=plan, device="cpu")
        assert mine.dtype == np.float32 and mine.tobytes() == theirs.tobytes()
        assert flip_tta.top16(mine).tobytes() == theirs.astype(np.float16).argmax(axis=0).astype(np.uint8).tobytes()     # eval_uf's tops["UF4"], hs_seg's labels
        assert record["members"] == members and record["image_sha256"] == meta["sha256"] and record["normalization"] == norm
        assert record["n_forward"] == 8 and record["forward_s"] == pytest.approx(record["unflipped_s"]+record["flip_s"])
        assert not np.array_equal(mine, flip_only)                                   # random members are not mirror-symmetric: the flip pass is a different pass


def test_each_member_runs_with_its_own_parameters_and_the_flip_pass_with_the_same_on_the_mirror(w, monkeypatch):
    calls = []
    monkeypatch.setattr(train_module, "predict_logits", recorder(calls))             # hs_seg.uf4_probabilities imports it at call time
    monkeypatch.setattr(flip_tta, "predict_logits", recorder(calls))
    plan, models, _ = w.plans[SUBJECTS[0]]
    with torch.no_grad():
        flip_tta.v4_probabilities(SUBJECTS[0], files=w.files, registry=w.registry, models=models, plan=plan, device="cpu", lut=flip_tta.swap_table())
    assert [Path(p).parent.name for _, _, p in plan] == NAMES and len(calls) == 8
    for i, (_, _, _, window, sigma, amp) in enumerate(MEMBERS):
        unflipped, mirrored = calls[i], calls[4+i]
        assert unflipped["window"] == tuple(window) and unflipped["gaussian_sigma"] == sigma and unflipped["amp"] is amp and unflipped["device"] == "cpu"
        assert {k: v for k, v in mirrored.items() if k != "image"} == {k: v for k, v in unflipped.items() if k != "image"}
        assert np.array_equal(mirrored["image"], np.flip(unflipped["image"], axis=-1)) and np.array_equal(unflipped["image"], calls[0]["image"])


def test_the_arms_use_each_members_temperature(w):
    plan, models, _ = w.plans[SUBJECTS[0]]
    lut = flip_tta.swap_table()
    with torch.no_grad():
        unflipped, flip_only, tta, _ = flip_tta.v4_probabilities(SUBJECTS[0], files=w.files, registry=w.registry, models=models, plan=plan, device="cpu", lut=lut)
        values, _ = read_image(w.files, SUBJECTS[0])
        image, _ = normalize_image(values, percentile=99.5)
        z, zq = [], []
        for (model_id, sha, path), (_, _, temperature, window, sigma, amp) in zip(plan, MEMBERS):
            model, member = load_member(path, sha, "cpu", entry=models[model_id])
            assert member["temperature"] == temperature
            z.append(REAL_PREDICT(model, image, device="cpu", amp=amp, window=tuple(window), gaussian_sigma=sigma))
            zq.append(REAL_PREDICT(model, np.flip(image, -1).copy(), device="cpu", amp=amp, window=tuple(window), gaussian_sigma=sigma))

    def mean(logits, temperatures, back=False):
        total = sum((flip_tta.flip_back if back else (lambda q, _: q))(torch.softmax(x/t, dim=1)[0], lut) for x, t in zip(logits, temperatures))/len(logits)
        return total/total.sum(0, keepdim=True)
    temperatures = [m[2] for m in MEMBERS]
    p, q = mean(z, temperatures), mean(zq, temperatures, back=True)
    assert np.allclose(unflipped, p.numpy(), atol=1e-6) and np.allclose(flip_only, q.numpy(), atol=1e-6)
    assert np.allclose(tta, ((p+q)/(p+q).sum(0, keepdim=True)).numpy(), atol=1e-6)
    assert not np.allclose(unflipped, mean(z, [1.]*4).numpy(), atol=1e-4)           # one temperature for all would be seen
    assert not np.allclose(flip_only, mean(zq, [1.]*4, back=True).numpy(), atol=1e-4)


def test_the_flip_pass_refuses_parameters_other_than_the_unflipped_passs(w, monkeypatch):
    hs = flip_tta.hs_seg_module()
    real = hs.uf4_probabilities

    def other_window(*args, **kwargs):
        probs, meta, norm, members = real(*args, **kwargs)
        members[2] = dict(members[2], window=[4, 4, 4])
        return probs, meta, norm, members
    monkeypatch.setattr(hs, "uf4_probabilities", other_window)
    plan, models, _ = w.plans[SUBJECTS[0]]
    with pytest.raises(ValueError, match="window"):
        flip_tta.v4_probabilities(SUBJECTS[0], files=w.files, registry=w.registry, models=models, plan=plan, device="cpu", lut=flip_tta.swap_table())


ORACLE_MEMBERS = tuple((f, s, t, [16, 16, 12], g, False) for f, s, t, _, g, _ in MEMBERS)    # one window over the whole (16, 16, 12) volume: the oracle sees the image's own midline


def test_a_side_oracle_through_the_v4_path_gives_the_same_labels_in_both_directions(tmp_path, monkeypatch):
    """Review m6 (b): a member that reads the sides by position (SideOracle) through v4_probabilities, hs_seg.uf4_probabilities included: the mirror's answer, flipped back with the classes swapped, is the unflipped
    one. The members are re-hashed and checked as always; only the network is the oracle."""
    import bayesreg.seg.unet.infer as infer_module
    w = world(tmp_path, members=ORACLE_MEMBERS)
    real = infer_module.load_member

    def oracle(path, sha, device, *, entry):
        _, member = real(path, sha, device, entry=entry)
        return SideOracle(), member
    monkeypatch.setattr(infer_module, "load_member", oracle)                         # hs_seg.uf4_probabilities imports it at call time
    monkeypatch.setattr(flip_tta, "load_member", oracle)
    plan, models, _ = w.plans[SUBJECTS[0]]
    with torch.no_grad():
        unflipped, flip_only, tta, record = flip_tta.v4_probabilities(SUBJECTS[0], files=w.files, registry=w.registry, models=models, plan=plan, device="cpu", lut=flip_tta.swap_table())
    maps = flip_tta.tops_v4(unflipped, flip_only, tta)
    assert {0, 1, 20} <= set(np.unique(maps["top"]).tolist()) and record["n_forward"] == 8
    assert np.array_equal(maps["top"], maps["top_flip"]) and np.array_equal(maps["top"], maps["top_tta"])
    assert np.allclose(unflipped, flip_only, atol=1e-5)


def test_flip_back_undoes_the_left_right_flip_of_training():
    """Review m6 (b): v3_large trains on large.flip_left_right (the image mirrored, the labels lut[flip(y)]); a perfect answer on the mirror, flipped back, is y on every voxel. Without the swap it is not."""
    from bayesreg.seg.unet.large import flip_left_right
    rng = np.random.default_rng(3)
    y = torch.as_tensor(rng.integers(0, 36, size=(1, 6, 5, 8)), dtype=torch.int64)
    x = torch.rand(1, 1, 6, 5, 8)
    lut = flip_tta.swap_table()
    x_flipped, y_flipped = flip_left_right(x, y, lut)
    one_hot = torch.nn.functional.one_hot(y_flipped[0], 36).permute(3, 0, 1, 2).float()
    assert torch.equal(flip_tta.flip_back(one_hot, lut).argmax(0), y[0]) and torch.equal(torch.flip(x_flipped, (flip_tta.FLIP_DIM,)), x)
    assert not torch.equal(torch.flip(one_hot, (flip_tta.FLIP_DIM,)).argmax(0), y[0])


def test_the_argmax_is_of_the_float16_cast_ties_to_the_smallest_label():
    p = np.zeros((36, 1, 1, 2), dtype=np.float32)
    p[3, 0, 0, 0], p[7, 0, 0, 0] = .5, .5+1e-5                                      # one float16 value: a tie, label 3; float32 would say 7
    p[5, 0, 0, 1], p[9, 0, 0, 1] = .2, .8
    assert flip_tta.top16(p).tolist() == [[[3, 9]]] and flip_tta.top16(p).dtype == np.uint8
    assert p.argmax(axis=0).tolist() == [[[7, 9]]]


# ---- before any image: the flip premise, the registered predictions, the plan ----------------------------------------------------------------------

def edit_record(change, *, rehash=True):
    def tamper(directory):
        path = directory/"train_record.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        change(record)
        write_json(path, record)
        if rehash:
            member = json.loads((directory/"member.json").read_text(encoding="utf-8"))
            write_json(directory/"member.json", dict(member, train_record_sha256=file_sha256(path)))
    return tamper


def other_member_json(directory):
    member = json.loads((directory/"member.json").read_text(encoding="utf-8"))
    write_json(directory/"member.json", dict(member, member_sha256="0"*64))


@pytest.mark.parametrize("tamper, message", [
    (edit_record(lambda r: r["config"].update(flip_probability=0.)), "flip_probability"),
    (edit_record(lambda r: r["config"].pop("flip_probability")), "flip_probability"),
    (edit_record(lambda r: r["config"].update(flip_probability=True)), "flip_probability"),
    (edit_record(lambda r: r["config"].update(flip_axis_check=False)), "flip_axis_check"),
    (edit_record(lambda r: r.update(flip_check=dict(skipped=True, reason="flip_axis_check false (synthetic subjects only)"))), "axis check"),
    (edit_record(lambda r: r.update(recipe="v2_full")), "recipe"),
    (edit_record(lambda r: r.update(left_right_pairs=r["left_right_pairs"][1:])), "left-right pairs"),
    (edit_record(lambda r: r["config"].update(batch_size=3), rehash=False), "member.json names"),
    (other_member_json, "another member.pt"),
    (lambda d: (d/"train_record.json").unlink(), "foldB_s1")])
def test_a_member_without_the_flip_is_refused_before_any_image(tmp_path, monkeypatch, tamper, message):
    w = world(tmp_path, tamper=tamper)
    patch_admission(monkeypatch, w)
    for module in (flip_tta, data_module):
        monkeypatch.setattr(module, "read_image", refuse)
    monkeypatch.setattr(eval_seg, "read_truth", refuse)
    with pytest.raises(flip_tta.FlipPremiseRefused, match=message):
        flip_tta.run_v4(out_dir=tmp_path/"out", device="cpu", data_root=w.data, lock_path=tmp_path/"lock", spec=w.spec, require_clean=False, log=quiet)
    assert not (tmp_path/"out").exists() and not (tmp_path/"lock").exists()


def test_the_flip_premise_record_of_good_members(w):
    plan, _, _ = w.plans[SUBJECTS[0]]
    records = flip_tta.flip_premise(plan, lut=flip_tta.swap_table())
    assert list(records) == NAMES
    assert all(r["flip_probability"] == .5 and r["flip_axis_check"] is True and r["flips_logged"] == 5 and r["problems"] == [] and r["recipe"] == "v3_large" for r in records.values())
    with pytest.raises(flip_tta.FlipPremiseRefused, match="swap table"):
        flip_tta.flip_premise(plan, lut=torch.arange(36))


@pytest.mark.parametrize("change, message", [(dict(registry_sha256="f"*64), "registry"), (dict(segmenter_id="unet35_v2_full"), "segmenter"), (dict(members=[]), "members"), (dict(test_labels_seen=True), "test_labels_seen")])
def test_registered_predictions_of_another_registration_are_refused_before_any_image(w, monkeypatch, change, message):
    path = w.pred_root/SUBJECTS[1]/"provenance.json"
    write_json(path, dict(json.loads(path.read_text(encoding="utf-8")), **change))
    for module in (flip_tta, data_module):
        monkeypatch.setattr(module, "read_image", refuse)
    with pytest.raises(ValueError, match=message):
        flip_tta.v4_inputs(SUBJECTS, allowlist=w.allowlist, spec=w.spec)


def test_a_plan_out_of_hs_order_is_refused(w, monkeypatch):
    hs = flip_tta.hs_seg_module()
    real = hs.uf_plan

    def reversed_plan(*args, **kwargs):
        plan, models, policy = real(*args, **kwargs)
        return plan[::-1], models, policy
    monkeypatch.setattr(hs, "uf_plan", reversed_plan)
    with pytest.raises(ValueError, match="order"):
        flip_tta.v4_inputs(SUBJECTS, allowlist=w.allowlist, spec=w.spec)


def test_a_subset_of_the_subjects_is_refused(w, tmp_path, monkeypatch):
    patch_admission(monkeypatch, w)
    monkeypatch.setattr(flip_tta, "v4_inputs", refuse)
    with pytest.raises(ValueError, match="every admitted"):
        flip_tta.run_v4(out_dir=tmp_path/"out", device="cpu", requested=[SUBJECTS[0]], data_root=w.data, lock_path=tmp_path/"lock", spec=w.spec, require_clean=False, log=quiet)


def test_results_are_never_overwritten(w, tmp_path):
    (tmp_path/"out").mkdir()
    (tmp_path/"out"/"old").write_text("x")
    with pytest.raises(FileExistsError):
        flip_tta.run_v4(out_dir=tmp_path/"out", device="cpu", data_root=w.data, lock_path=tmp_path/"lock", spec=w.spec, require_clean=False, log=quiet)
    assert (tmp_path/"out"/"old").read_text() == "x"


# ---- the control and the decision ----------------------------------------------------------------------------------------------------------------

S20 = [f"OASIS_OAS1_{k:04d}_MR1" for k in range(438, 458)]


def row(value):
    return dict(seg35=value, seg4=value, groups={g: value for g in eval_seg.GROUPS}, seg4_per_label=[value]*4)


def table_of(gain, n_positive, *, subjects=S20, level=.93):
    rows = {"v4": {s: dict(unflipped=row(level), flip_only=row(level), tta=row(level+(gain if i < n_positive else 0.))) for i, s in enumerate(subjects)}}
    return flip_tta.tabulate(rows, subjects, {"v4": (level, level)}, metrics=flip_tta.METRICS_V4)["v4"]


def agreement_of(values, subjects=S20):
    return {s: dict(agreement=v) for s, v in zip(subjects, values)}


@pytest.mark.parametrize("reference, agreements, gain, n_positive, decision", [
    ((.93, .93), [1.]*20, .003, 20, "过门"),
    ((.93, .93), [.9999]*20, .0027, 15, "过门"),                                     # .9999 everywhere, 15/20 positive, mean .002025
    ((.9304, .9296), [1.]*20, .003, 20, "过门"),                                     # 4e-4 off the committed means: within 5e-4
    ((.93, .93), [1.]*20, .0019, 20, "不过门"),
    ((.93, .93), [1.]*20, .003, 14, "不过门"),
    ((.9306, .93), [1.]*20, .003, 20, "不判"),                                       # the mean of seg35 off by 6e-4
    ((.93, .9294), [1.]*20, .003, 20, "不判"),                                       # the mean of seg4
    ((.93, .93), [1.]*19+[.99989], .003, 20, "不判"),                                # one subject below .9999
    ((.9306, .9294), [.5]*20, .003, 20, "不判"),                                     # both
    ((.93, .93), [1.]*19+[None], .003, 20, "不判")])                                 # a registered prediction that could not be read
def test_the_control_and_the_decision_of_the_plan(reference, agreements, gain, n_positive, decision):
    table = table_of(gain, n_positive)
    control = flip_tta.v4_control(table, agreement_of(agreements), S20, reference)
    assert flip_tta.v4_decision(table, control) == decision
    assert control["passed"] is (decision != "不判")
    assert control["n_subjects"] == 20 and control["agreement_threshold"] == .9999 and control["tol"] == 5e-4


def test_the_control_names_which_part_failed():
    table = table_of(.003, 20)
    mean_only = flip_tta.v4_control(table, agreement_of([1.]*20), S20, (.9306, .93))
    agreement_only = flip_tta.v4_control(table, agreement_of([1.]*19+[.9]), S20, (.93, .93))
    both = flip_tta.v4_control(table, agreement_of([.9]*20), S20, (.9306, .9294))
    assert (mean_only["mean_passed"], mean_only["agreement_passed"]) == (False, True)
    assert (agreement_only["mean_passed"], agreement_only["agreement_passed"], agreement_only["n_below"], agreement_only["agreement_min"]) == (True, False, 1, .9)
    assert (both["mean_passed"], both["agreement_passed"], both["n_below"]) == (False, False, 20)


def test_fewer_than_20_subjects_are_not_judged():
    subjects = S20[:19]
    table = table_of(.003, 19, subjects=subjects)
    control = flip_tta.v4_control(table, agreement_of([1.]*19, subjects), subjects, (.93, .93))
    assert control["passed"] is False and flip_tta.v4_decision(table, control) == "不判"


def test_twenty_subjects_other_than_the_development_set_are_not_judged():
    other = [f"OASIS_OAS1_{k:04d}_MR1" for k in range(418, 438)]
    table = table_of(.003, 20, subjects=other)
    control = flip_tta.v4_control(table, agreement_of([1.]*20, other), other, (.93, .93))
    assert control["subjects_are_dev"] is False and control["passed"] is False and flip_tta.v4_decision(table, control) == "不判"
    assert flip_tta.DEV_SUBJECTS == tuple(S20)


def test_a_seg35_difference_that_is_not_finite_is_not_judged():
    """Review m2: 19 finite differences, all positive, mean .003, the control passed: without the count it would be 过门."""
    rows = {"v4": {s: dict(unflipped=row(.93), flip_only=row(.93), tta=row(float("nan") if i == 0 else .933)) for i, s in enumerate(S20)}}
    table = flip_tta.tabulate(rows, S20, {"v4": (.93, .93)}, metrics=flip_tta.METRICS_V4)["v4"]
    control = flip_tta.v4_control(table, agreement_of([1.]*20), S20, (.93, .93))
    assert control["passed"] is True and table["worth_using"] is True and table["tta_minus_unflipped"]["seg35"]["n"] == 19
    assert flip_tta.undecided_reasons(table, control) == ["seg35 的有限差值 19 人，不是 20 人"] and flip_tta.v4_decision(table, control) == "不判"


def test_the_reported_agreement_details_are_not_part_of_the_rule():
    table = table_of(.003, 20)
    records = {s: dict(agreement=1.-d/1e6, n_differ=d, agreement_foreground=1.-d/3e5, image_same_as_registered=True) for s, d in zip(S20, [0]*18+[5, 100])}
    control = flip_tta.v4_control(table, records, S20, (.93, .93))
    assert control["passed"] is True and flip_tta.v4_decision(table, control) == "过门"
    assert control["reported"]["n_differ_max"] == 100 and control["reported"]["agreement_foreground_min"] == pytest.approx(1.-100/3e5)
    assert control["reported"]["n_identical_to_registered"] == 18 and control["reported"]["n_image_same_as_registered"] == 20 and control["reported"]["n_unreadable"] == 0


# ---- the agreement with the registered predictions -----------------------------------------------------------------------------------------------

def test_the_agreement_counts_voxels_and_an_unreadable_prediction_is_recorded_not_raised(w, tmp_path):
    out = tmp_path/"out"
    (out/"v4").mkdir(parents=True)
    for subject in SUBJECTS:
        raw, _ = common.load_raw(w.pred_root, subject, expected_segmenter="unet35_v4_ens")
        top = common.argmax_of_mean([raw])
        if subject == SUBJECTS[0]:
            top.ravel()[:3] = (top.ravel()[:3]+1) % 36                               # three voxels differ
        np.savez_compressed(out/"v4"/f"{subject}.npz", top=top, top_flip=top, top_tta=top)
    with np.load(w.pred_root/SUBJECTS[1]/"probs36.npz") as npz:
        changed = npz["p"].copy()
    changed[0, 0, 0, 0] += np.float16(.001)
    np.savez_compressed(w.pred_root/SUBJECTS[1]/"probs36.npz", p=changed)            # its sha256 is not the provenance's any more
    agreement = flip_tta.registered_agreement(SUBJECTS, out_dir=out, pred_root=w.pred_root, expected_segmenter="unet35_v4_ens")
    n = 16*16*12
    assert agreement[SUBJECTS[0]]["n_differ"] == 3 and agreement[SUBJECTS[0]]["agreement"] == (n-3)/n and agreement[SUBJECTS[0]]["n_voxels"] == n
    assert agreement[SUBJECTS[1]]["agreement"] is None and "ProvenanceRejected" in agreement[SUBJECTS[1]]["error"]


# ---- the read lock -------------------------------------------------------------------------------------------------------------------------------

def test_a_lock_that_is_there_refuses_the_run_before_anything(tmp_path, monkeypatch):
    lock = tmp_path/"tta_dev_read.lock"
    lock.write_text(json.dumps(dict(event="first_read", utc="2026-11-07T08:00:00Z", out_dir="/x"))+"\n")
    monkeypatch.setattr(common, "admit_dev_subjects", refuse)
    monkeypatch.setattr(eval_seg, "read_truth", refuse)
    for module in (flip_tta, data_module):
        monkeypatch.setattr(module, "read_image", refuse)
    with pytest.raises(flip_tta.DevReadLocked, match="read once"):
        flip_tta.run_v4(out_dir=tmp_path/"out", device="cpu", data_root=tmp_path, lock_path=lock, spec=dict(flip_tta.SEGMENTERS["v4"]), require_clean=False, log=quiet)
    assert not (tmp_path/"out").exists() and lock.read_text().startswith('{"event": "first_read"')


def test_a_lock_that_appears_before_the_first_truth_label_stops_the_read(w, tmp_path, monkeypatch):
    lock = tmp_path/"tta_dev_read.lock"
    patch_admission(monkeypatch, w)
    monkeypatch.setattr(eval_seg, "read_truth", refuse)
    real = flip_tta.registered_agreement

    def then_another_run_takes_the_lock(*args, **kwargs):
        result = real(*args, **kwargs)
        lock.write_text("another run\n")
        return result
    monkeypatch.setattr(flip_tta, "registered_agreement", then_another_run_takes_the_lock)
    with pytest.raises(flip_tta.DevReadLocked, match="appeared"):
        flip_tta.run_v4(out_dir=tmp_path/"out", device="cpu", data_root=w.data, lock_path=lock, spec=w.spec, require_clean=False, log=quiet)
    assert lock.read_text() == "another run\n" and not (tmp_path/"out"/"flip_tta.json").exists()


@pytest.mark.parametrize("content", [b"not json\n", b"\xff\xfe\x00 garbage", b"[1, 2]\n", b""])
def test_a_lock_that_is_not_json_refuses_the_run_too(tmp_path, monkeypatch, content):
    """Review m3: whatever is in the file, the run is refused as DevReadLocked before anything is read, and the file is left as it is."""
    lock = tmp_path/"tta_dev_read.lock"
    lock.write_bytes(content)
    monkeypatch.setattr(common, "admit_dev_subjects", refuse)
    monkeypatch.setattr(eval_seg, "read_truth", refuse)
    for module in (flip_tta, data_module):
        monkeypatch.setattr(module, "read_image", refuse)
    with pytest.raises(flip_tta.DevReadLocked, match="read once"):
        flip_tta.run_v4(out_dir=tmp_path/"out", device="cpu", data_root=tmp_path, lock_path=lock, spec=dict(flip_tta.SEGMENTERS["v4"]), require_clean=False, log=quiet)
    assert lock.read_bytes() == content and not (tmp_path/"out").exists()


def test_another_plan_is_refused_before_anything(tmp_path, monkeypatch):
    """Review m4: plan.md is pinned; the check comes after the lock's and before the admission."""
    assert flip_tta.PLAN_SHA256 == "1098f388ff8a1c69abaec8f3041370d4d60c30bd511367775666b4da125bf183" and flip_tta.check_plan() == flip_tta.PLAN_SHA256      # the committed preregistration
    monkeypatch.setattr(flip_tta, "PLAN_SHA256", "0"*64)
    monkeypatch.setattr(common, "admit_dev_subjects", refuse)
    monkeypatch.setattr(eval_seg, "read_truth", refuse)
    with pytest.raises(flip_tta.PlanChanged, match="preregistration"):
        flip_tta.run_v4(out_dir=tmp_path/"out", device="cpu", data_root=tmp_path, lock_path=tmp_path/"lock", spec=dict(flip_tta.SEGMENTERS["v4"]), require_clean=False, log=quiet)
    assert not (tmp_path/"lock").exists() and not (tmp_path/"out").exists()


def test_the_lock_is_taken_once_before_the_first_read():
    taken, reads = [], []
    reader, state = flip_tta.locked_reader(lambda s, k: reads.append((s, k)) or k, lambda: taken.append(len(reads)) or "line")
    assert reader("a", "seg35") == "seg35" and reader("a", "seg4") == "seg4" and reader("b", "seg35") == "seg35"
    assert taken == [0] and state["line"] == "line" and len(reads) == 3


# ---- one run end to end, the dry run --------------------------------------------------------------------------------------------------------------

def test_a_run_end_to_end(w, tmp_path, monkeypatch):
    lock = tmp_path/"locks"/"tta_dev_read.lock"
    patch_admission(monkeypatch, w)
    truth, reads = synthetic_truth(w.data), []

    def read_truth(root, subject, kind):
        assert lock.exists(), "a truth label before the lock"
        reads.append((subject, kind))
        return truth(root, subject, kind)
    monkeypatch.setattr(eval_seg, "read_truth", read_truth)
    document = flip_tta.run_v4(out_dir=tmp_path/"out", device="cpu", data_root=w.data, lock_path=lock, spec=w.spec, require_clean=False, argv=["flip_tta.py", "--segmenters", "v4"], log=quiet)
    assert reads == [(s, k) for s in SUBJECTS for k in ("seg35", "seg4")]
    line = json.loads(lock.read_text(encoding="utf-8"))
    assert sorted(line) == sorted(["reading", "plan_sha256", "segmenter_id", "registry_sha256", "member_root", "members", "out_dir", "command", "code_commit", "event", "utc", "pid"])
    assert line["event"] == "first_read" and line["members"] == w.shas and list(line["members"]) == NAMES and line["code_commit"] == "c"*40 and line["reading"] == "runs/20261107-tta/plan.md"
    assert document["dev_read_lock"] == line and document["format"] == "bayesreg.segprobe.flip_tta_v4.v1"
    assert all(document["agreement"][s]["agreement"] == 1. and document["agreement"][s]["image_same_as_registered"] is True and document["agreement"][s]["provenance_unchanged"] is True for s in SUBJECTS)
    assert document["decision"] == "不判" and document["control"]["agreement_passed"] is False and document["control"]["n_subjects"] == 2     # two subjects: the reference means are over 20
    assert document["control"]["subjects_are_dev"] is False and document["undecided_reasons"][0] == "控制不过"
    assert "不解读" in document["interpretation"] and document["table"]["worth_using"] is None                       # a 不判 is not read (review m2)
    reported = document["control"]["reported"]
    assert (reported["n_differ_max"], reported["agreement_foreground_min"], reported["n_identical_to_registered"], reported["n_image_same_as_registered"], reported["n_unreadable"]) == (0, 1., 2, 2, 0)
    assert document["plan"]["sha256"] == document["plan"]["pinned"] == flip_tta.PLAN_SHA256 and line["plan_sha256"] == flip_tta.PLAN_SHA256
    assert {"torch", "cudnn", "cuda_build", "device"} <= set(document["environment"]) and document["host_peak_rss_gib"] > 0 and "forward_s" in document["seconds_note"]
    assert all(document["seconds"][s]["host_peak_rss_gib"] > 0 for s in SUBJECTS)
    assert all(rel in document["code_files"] and document["code_files"][rel] for rel in ("experiments/hs/hs_core.py", "experiments/unv2/eval_seg.py", "experiments/nn/eval_nn.py"))
    table = document["table"]
    assert sorted(table["means"]) == sorted(flip_tta.ARMS) and all(sorted(m) == sorted(flip_tta.METRICS_V4) for m in table["means"].values())
    assert sorted(table["tta_minus_unflipped_per_subject"]["seg35"]) == sorted(SUBJECTS) and table["tta_minus_unflipped"]["seg35"]["n"] == 2
    assert all(document["seconds"][s]["n_forward"] == 8 and document["seconds"][s]["gpu_peak_allocated_gib"] is None for s in SUBJECTS)
    assert [m["member_sha256"] for m in document["members"]] == [w.shas[n] for n in NAMES] and document["member_order"] == NAMES and list(document["flip_premise"]) == NAMES
    for subject in SUBJECTS:
        with np.load(tmp_path/"out"/"v4"/f"{subject}.npz") as npz:
            assert sorted(npz.files) == ["top", "top_flip", "top_tta"] and all(npz[k].dtype == np.uint8 and npz[k].shape == (16, 16, 12) for k in npz.files)
    stored = json.loads((tmp_path/"out"/"flip_tta.json").read_text(encoding="utf-8"))
    assert stored["decision"] == "不判" and stored["dev_read_lock"]["pid"] == line["pid"]
    header = (tmp_path/"out"/"per_subject.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "segmenter,subject,arm,seg35,seg4,small_six,big_six,other_23,cortex,white_matter,forward_s,agreement"
    summary = (tmp_path/"out"/"summary.md").read_text(encoding="utf-8")
    assert "判定：不判" in summary and "不解读" in summary and "与登记 argmax 不同的体素数最大 0" in summary and "argmax 与登记逐体素相同 2 人" in summary
    with pytest.raises(flip_tta.DevReadLocked):                                       # read once
        flip_tta.run_v4(out_dir=tmp_path/"out2", device="cpu", data_root=w.data, lock_path=lock, spec=w.spec, require_clean=False, log=quiet)


def test_the_dry_run_reads_no_subject_image_no_truth_and_writes_no_lock(w, tmp_path, monkeypatch):
    roots = []

    def watched(files, subject):
        roots.append((Path(files.root), subject))
        return read_image(files, subject)                                            # the function as imported above, not the patched attribute
    monkeypatch.setattr(data_module, "read_image", watched)
    monkeypatch.setattr(flip_tta, "read_image", watched)
    for target, name in ((eval_seg, "read_truth"), (flip_tta, "require_free_gpu"), (flip_tta, "take_lock"), (torch.cuda, "synchronize")):
        monkeypatch.setattr(target, name, refuse)
    lock, lines = tmp_path/"tta_dev_read.lock", []
    record = flip_tta.dry_run_v4(SUBJECTS, allowlist=w.allowlist, spec=w.spec, lock_path=lock, log=lines.append)
    assert not lock.exists() and record["lock_exists"] is False
    assert roots and all(subject == flip_tta.SYNTHETIC_SUBJECT and root != Path(w.data) for root, subject in roots)
    assert [m["member_sha256"] for m in record["members"]] == [w.shas[n] for n in NAMES] and [m["window"] for m in record["members"]] == [m[3] for m in MEMBERS]
    assert record["order"] == NAMES and sorted(record["registered"]) == sorted(SUBJECTS) and record["plan_pinned"] is True
    assert any("flip premise" in line for line in lines) and any("absent" in line and "the pinned preregistration" in line for line in lines)


@pytest.mark.parametrize("gain, positive, decision", [(-1., 0, "过门"), (1., 0, "不过门"), (-1., 3, "不过门")])
def test_run_v4_reaches_pass_and_fail_when_the_control_passes(w, tmp_path, monkeypatch, gain, positive, decision):
    """Review m6 (a): the whole path from run_v4 to the decision with a control that passes. The development set is these 2 subjects, the reference the measured unflipped means (the CPU repeats them bit by bit);
    the rule's gain and count are moved so that each decision is reached."""
    patch_admission(monkeypatch, w)
    monkeypatch.setattr(eval_seg, "read_truth", synthetic_truth(w.data))
    monkeypatch.setattr(flip_tta, "N_DEV", 2)
    monkeypatch.setattr(flip_tta, "DEV_SUBJECTS", tuple(SUBJECTS))
    first = flip_tta.run_v4(out_dir=tmp_path/"first", device="cpu", data_root=w.data, lock_path=tmp_path/"lock1", spec=w.spec, require_clean=False, log=quiet)
    measured = first["table"]["means"]["unflipped"]
    monkeypatch.setattr(flip_tta, "WORTH_GAIN", gain)
    monkeypatch.setattr(flip_tta, "WORTH_POSITIVE", positive)
    document = flip_tta.run_v4(out_dir=tmp_path/"second", device="cpu", data_root=w.data, lock_path=tmp_path/"lock2", spec=dict(w.spec, reference=(measured["seg35"], measured["seg4"])), require_clean=False, log=quiet)
    control = document["control"]
    assert control["passed"] is True and control["mean_difference"] == [0., 0.] and control["agreement_min"] == 1. and control["subjects_are_dev"] is True
    assert document["decision"] == decision and document["undecided_reasons"] == [] and document["interpretation"] is None and document["table"]["worth_using"] is (decision == "过门")
    summary = (tmp_path/"second"/"summary.md").read_text(encoding="utf-8")
    assert f"判定：{decision}**" in summary and "不解读" not in summary
    assert json.loads((tmp_path/"second"/"flip_tta.json").read_text(encoding="utf-8"))["decision"] == decision


# ---- v1 and v2 as before; the command line ------------------------------------------------------------------------------------------------------

def test_v1_and_v2_outputs_keep_their_keys(tmp_path, monkeypatch):
    volumes = {s: blob(seed=i) for i, s in enumerate(SUBJECTS)}
    lookup = eval_seg.load_seg4_lookup()
    truth = _truth_from(volumes, lookup)
    monkeypatch.setattr(common, "admit_dev_subjects", lambda **kwargs: (list(SUBJECTS), dict(dev_source="synthetic")))
    monkeypatch.setattr(flip_tta, "load_dev_allowlist", lambda path: None)
    monkeypatch.setattr(flip_tta, "load_members", lambda name, subjects, device, allowlist: [dict(member_of(SideOracle(), 1. if name == "v1" else 2.), model_id=f"{name}_m", fold=0, seed=0, sha256="0"*64)])
    monkeypatch.setattr(flip_tta, "read_image", lambda files, subject: (volumes[subject].astype(np.float64), dict(sha256="1"*64)))
    monkeypatch.setattr(eval_seg, "read_truth", lambda root, subject, kind: truth(subject, kind))
    monkeypatch.setattr(flip_tta, "hs_seg_module", refuse)                           # v1 and v2 never import hs_seg
    document = flip_tta.run(names=["v1", "v2"], out_dir=tmp_path/"out", device="cpu", log=quiet)
    assert sorted(document) == sorted(["format", "created_utc", "command", "admission", "subjects", "device", "gpu", "gpu_cap_gib", "members", "lut", "seconds", "tables", "truth_sha256", "rows", "note"])
    assert document["format"] == "bayesreg.segprobe.flip_tta.v1"
    for name in ("v1", "v2"):
        table = document["tables"][name]
        assert sorted(table) == sorted(["means", "tta_minus_unflipped", "rule", "worth_using", "control"]) and table["control"] is None
        assert sorted(table["means"]) == sorted(flip_tta.ARMS) and list(table["means"]["tta"]) == ["seg35", "seg4", "small_six", "big_six", "other_23"]
        assert list(table["tta_minus_unflipped"]) == ["seg35", "seg4", "small_six", "big_six", "other_23"]
        assert all(sorted(r) == sorted(["forward_s", "subject_s", "n_forward", "gpu_peak_allocated_gib", "image_sha256"]) for r in document["seconds"][name].values())
        assert sorted(document["members"][name][0]) == sorted(["model_id", "fold", "seed", "temperature", "amp", "window", "gaussian_sigma", "percentile", "sha256"])
        for subject in SUBJECTS:
            with np.load(tmp_path/"out"/name/f"{subject}.npz") as npz:
                assert sorted(npz.files) == ["top", "top_flip", "top_tta"]
    assert (tmp_path/"out"/"per_subject.csv").read_text(encoding="utf-8").splitlines()[0] == "segmenter,subject,arm,seg35,seg4,small_six,big_six,other_23,forward_s"
    assert "## v1" in (tmp_path/"out"/"summary.md").read_text(encoding="utf-8")


def test_the_command_line_defaults_and_v4_alone(monkeypatch):
    seen = {}
    monkeypatch.setattr(flip_tta, "run", lambda **kwargs: seen.update(run=kwargs) or dict(tables={}))
    monkeypatch.setattr(flip_tta, "dry_run", lambda names, subjects, **kwargs: seen.update(dry_run=names) or dict(subjects=subjects, synthetic_seconds=0.))
    monkeypatch.setattr(flip_tta, "run_v4", lambda **kwargs: seen.update(run_v4=kwargs) or dict(decision="不判", interpretation="不判（控制不过）：以下数字只作记录，不解读。", table=dict(means={}, tta_minus_unflipped=dict(seg35={}), worth_using=None),
                                                                                                  control=dict(passed=False, subjects_are_dev=True, mean_difference=[0, 0], agreement_min=None, n_below=0, reported=dict(n_differ_max=None, agreement_foreground_min=None, n_identical_to_registered=0))))
    monkeypatch.setattr(flip_tta, "dry_run_v4", refuse)
    monkeypatch.setattr(common, "admit_dev_subjects", lambda **kwargs: (list(S20), {}))
    monkeypatch.setattr(flip_tta, "load_dev_allowlist", lambda path: None)
    assert flip_tta.main([]) == 0
    assert seen["run"]["names"] == ["v1", "v2"] and seen["run"]["out_dir"] == common.OUT_ROOT/"p2"
    assert flip_tta.main(["--dry-run"]) == 0 and seen["dry_run"] == ["v1", "v2"]
    for argv in (["--segmenters", "v4"], ["--segmenters", "v4", "--dry-run", "--real-run"], ["--segmenters", "v1", "v4", "--real-run"], ["--real-run"]):
        with pytest.raises(SystemExit):                                                # v4 needs exactly one of --dry-run, --real-run (review m8); v4 alone; --real-run is for v4 only
            flip_tta.main(argv)
    assert "run_v4" not in seen
    assert flip_tta.main(["--segmenters", "v4", "--real-run"]) == 0 and seen["run_v4"]["out_dir"] == common.OUT_ROOT/"tta_v4" and seen["run_v4"]["requested"] is None
