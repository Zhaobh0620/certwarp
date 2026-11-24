"""D0 supplementary arm A4p (A4' of docs/bayesreg/05_D0协议.md §12): experiments/d0/make_inputs.py (A4p branch,
--write-a4-subjects) and experiments/d0/evaluate.py (decide_a4, summarize-a4), on synthetic files only (tmp_path; no
data/, no real image, label or prediction; the subject ids are the synthetic stand-ins of tests/test_bayesreg/io_seg).

- the A4p colouring is the A3 formula with the one-hot replaced by the posterior: with a one-hot posterior it equals the
  A3 colouring of the same labels (delta from the cerebral white matter, labels 1 and 20) to 1e-12; the written inputs
  equal an independent recomputation from probs36.npz; labels and initial_u are copied bitwise; no label reaches the
  colouring; the record carries policy all_folds_mean, the registered registry and the read-back raw sums;
- refusals (STOP, no d0_inputs.json): a float16 probs36.npz whose raw channel sum is off by more than 1e-3 on one voxel
  (provenance hashes updated, so only the read-back catches it), a probs36.npz changed after its provenance, a missing
  repository registry (no other registry is ever read), a model that saw a pair subject (D24 pair level), a subject
  listed in the folds file or a folds file without the pinned hash (D24, before anything is written), probs36
  disagreeing with structure_groups.npz, A4p under d0_v1/ or mixed with A1-A3;
- A1 is unchanged by the refactor (stage / allowlist parameters);
- --write-a4-subjects: the stage subjects after the admission, --only-missing, never overwrites;
- decide_a4: R4a, R4b, the middle region, the missing-pair rule (denominator 19, < 17 valid -> not decidable) and the
  descriptive recovery ratios; summarize-a4 merges A4p with A0-A3 of the D0 batch, reports seg35 and seg4, the launch
  load, and refuses an output directory under d0_v1/;
- run_queue.sh: bash -n, and the --a4 mode settings (d0_v1_a4, arm A4p, unit prefix, start checks, summarize-a4).
REG (reviews/A4_预登记复核_20261030.md):
- item 1: one link lock file shared by both modes;
- item 2: run_queue.sh --a4 on a synthetic tree (fake repository, results root, systemctl, nvidia-smi): the dry run lists
  the 19 pairs and creates nothing; each start check STOPs (training or D0 unit active, GPU busy, registry modified,
  inputs record from another registry or without a clean registry); all passing reaches the frozen-worktree check;
- item 3: the repository registry must be committed and clean (modified, staged, untracked, no work tree: refused before
  anything is written; the per-pair record is checked again);
- item 5: the label-free alignment QC, since protocol revision 2 (§12.4 item 5, before any A4p result)
  Dice({U-Net cerebral_wm > .5}, {Atropos WM > .5}) >= .80 per image: a synthetic gyral phantom passes aligned and is
  stopped by every flip and by shifts of 2, 4, 8 voxels along each axis; on disk a flipped or shifted U-Net prediction
  STOPs the arm; a posterior with a low revision-1 Dice({I > 0}, {P_0 < .5}) but agreeing white matter passes (that
  Dice is recorded, never stops); a missing Atropos output (one subject, no --seg-root, a missing / changed posterior
  file, provenance bound to another image) is refused; the WM file follows wm_index;
- item 6: the group-sum tolerance 2e-3; items 7, 8: the evaluator hashes in summary_a4.md, R4a worded as a
  single-channel diagnostic.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from jd_curl.three_d.nifti import file_sha256

from bayesreg.seg import structure_layer
from bayesreg.seg.structure_layer import GROUPS_FILE, PROVENANCE_FILE
from bayesreg.seg.unet.groups import group_probabilities, load_grouping

from ..io_seg._helpers import (DEV, FOLD0, FOLD1, SYN_FOLD0, SYN_FOLD1, allowlist_document, is_synthetic, source_sha, synthetic_fold_sources,
                               synthetic_folds, write_json)
from .test_run_pair_structure import SEG_ID, git_repo, registry_models, write_registry

ROOT = Path(__file__).resolve().parents[3]
SHAPE = (8, 9, 10)
GROUPING = load_grouping()
PROBS_FILE = "probs36.npz"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MI = load("d0_make_inputs", ROOT/"experiments"/"d0"/"make_inputs.py")
EV = load("d0_evaluate", ROOT/"experiments"/"d0"/"evaluate.py")


def number(subject):
    return int(subject.split("_")[2])


def image(subject):
    return np.random.default_rng(number(subject)).uniform(.1, 1., SHAPE)


def seg35(subject):
    return np.random.default_rng(1000+number(subject)).integers(0, 36, SHAPE)


def seg4(subject):
    return np.random.default_rng(2000+number(subject)).integers(0, 5, SHAPE)


def onehot(labels, n=36):
    return (np.arange(n)[:, None, None, None] == labels[None]).astype(np.float64)


def posterior(subject):
    """Soft 36-class posterior around the subject's synthetic seg35 (float32, as segment_subject computes it)."""
    rng = np.random.default_rng(3000+number(subject))
    noise = rng.uniform(0, 1, (36,)+SHAPE)
    p = .6*onehot(seg35(subject))+.4*noise/noise.sum(axis=0)
    return (p/p.sum(axis=0)).astype(np.float32)


def write_stage(root, pairs):
    stage = root/"data"/"three_d"/MI.STAGE_NAME
    rows = []
    for f, m in pairs:
        name = f"{f}__{m}"
        d = stage/name
        d.mkdir(parents=True)
        arrays = dict(fixed=image(f), fixed_seg4=seg4(f).astype(float), fixed_seg35=seg35(f).astype(float),
                      moving=image(m), moving_seg4=seg4(m).astype(float), moving_seg35=seg35(m).astype(float),
                      initial_u=np.zeros(SHAPE+(3,)))
        np.savez_compressed(d/"pair.npz", **{k: arrays[k] for k in MI.KEY_ORDER})
        manifest = dict(pair=name, fixed_subject=f, moving_subject=m, shape_zyx=list(SHAPE), spacing_xyz=[1., 1., 1.],
                        origin_xyz=[0., 0., 0.], unit="mm",
                        sources=dict(fixed=dict(sha256=source_sha(f)), moving=dict(sha256=source_sha(m))),
                        pair_sha256=file_sha256(d/"pair.npz"), baseline_native_mse=.01, preprocessing="synthetic")
        write_json(d/"manifest.json", manifest)
        rows.append(dict(pair=name))
    write_json(stage/"selection.json", dict(pairs=rows))
    write_json(stage/"completion.json", dict(done=True))
    return stage


def write_prediction(pred_root, subject, models, *, probs=None, groups=None):
    """A prediction directory as bayesreg.seg.unet.infer.segment_subject writes it (probs36 + groups + provenance)."""
    d = pred_root/subject
    d.mkdir(parents=True)
    probs = posterior(subject) if probs is None else probs
    groups = group_probabilities(probs, GROUPING, dtype=np.float32) if groups is None else groups
    np.savez_compressed(d/PROBS_FILE, p=probs.astype(np.float16))
    np.savez_compressed(d/GROUPS_FILE, p=groups.astype(np.float16))
    usable = sorted(m for m, e in models.items() if subject not in e["label_contact_subjects"])
    members = [dict(model_id=m, fold=models[m]["fold"], seed=k, member_sha256=sha, temperature=1.1, best_iteration=5)
               for m in usable for k, sha in enumerate(models[m]["member_model_sha256"])]
    prov = dict(layer="structure", segmenter_id=SEG_ID, model_ids=usable,
                model_sha256={m: list(models[m]["member_model_sha256"]) for m in usable},
                folds={m: models[m]["fold"] for m in usable}, trained_with_labels=True, test_labels_seen=False,
                subject=subject, policy="all_folds_mean" if len(usable) == len(models) else "cross_fit_other_folds",
                members=members, **GROUPING.record(),
                grouping_check=dict(grouping_status=GROUPING.status, grouping_status_admitted=True,
                                    allow_unverified_grouping=False),
                input_sha256={"aligned_norm.nii.gz": source_sha(subject)},
                output_sha256={PROBS_FILE: file_sha256(d/PROBS_FILE), GROUPS_FILE: file_sha256(d/GROUPS_FILE)},
                grid=dict(shape_zyx=list(SHAPE), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.]),
                registry_path="synthetic", registry_sha256="0"*64, environment=dict(code=dict(kind="synthetic")),
                runtime_seconds=0.)
    write_json(d/PROVENANCE_FILE, prov)
    return d


ATROPOS_FILES = [f"posteriors{k+1:02d}.nii.gz" for k in range(3)]


def atropos_posteriors(subject):
    """Synthetic Atropos (CSF, GM, WM) posteriors, zyx: WM .9 on the subject's synthetic cerebral white matter (seg35
    labels 1, 20 of the synthetic stand-in), where the U-Net cerebral_wm group of `posterior` is > .5; .05 elsewhere."""
    wm = np.where(np.isin(seg35(subject), MI.CEREBRAL_WM), .9, .05)
    csf = .3*(1-wm)
    return np.stack([csf, 1-wm-csf, wm])


def write_atropos(seg_root, subject, *, posteriors=None, class_names=("CSF", "GM", "WM"), wm_index=2, **changes):
    """An Atropos output directory as experiments/d0/atropos_seg.py writes it: posteriorsNN.nii.gz stored xyz (the reader
    transposes to zyx) and a tissue-layer provenance bound to the subject's synthetic source hash."""
    d = seg_root/subject
    d.mkdir(parents=True)
    P = atropos_posteriors(subject) if posteriors is None else posteriors
    for name, p in zip(ATROPOS_FILES, P):
        nib.Nifti1Image(np.ascontiguousarray(p.transpose(2, 1, 0)).astype(np.float32), np.eye(4)).to_filename(str(d/name))
    prov = dict(layer="tissue", segmenter_id="atropos", model_ids=["ANTs-2.6.5-synthetic-Atropos"],
                trained_with_labels=False, test_labels_seen=False, oracle=False, class_names=list(class_names),
                wm_index=wm_index, posterior_files=ATROPOS_FILES, posterior_file_index_base=1,
                input_sha256={"aligned_norm.nii.gz": source_sha(subject)},
                output_sha256={name: file_sha256(d/name) for name in ATROPOS_FILES})
    prov.update(changes)
    write_json(d/"provenance.json", prov)
    return d


def rehash(directory):
    """Update provenance output_sha256 after a deliberate change of an output file."""
    prov = json.loads((directory/PROVENANCE_FILE).read_text())
    prov["output_sha256"] = {name: file_sha256(directory/name) for name in prov["output_sha256"]}
    write_json(directory/PROVENANCE_FILE, prov)


PAIRS = [(DEV[0], DEV[1]), (DEV[1], DEV[2])]
SUBJECTS = [DEV[0], DEV[1], DEV[2]]


def commit_registry(registry, message="registry"):
    """Commit the test's repository registry in its git work tree (A4 review item 3: make_inputs needs it committed
    and clean)."""
    root = registry.parent.parent
    git = git_repo(root)                                    # `git init` again on an existing work tree is harmless
    git("add", str(registry.relative_to(root)))
    git("commit", "-q", "-m", message)


@pytest.fixture
def env(tmp_path, monkeypatch, request):
    """The A4 world.  With synthetic_folds("env"): the synthetic training draw (C23.9-13 (8), revision 6e); env["fold0"], env["fold1"] name the draw."""
    synthetic = is_synthetic(request)
    fold0, fold1 = (SYN_FOLD0, SYN_FOLD1) if synthetic else (FOLD0, FOLD1)
    allowlist = write_json(tmp_path/"dev_subjects.json", allowlist_document(synthetic_fold_sources() if synthetic else None))
    models = registry_models(fold0, fold1) if synthetic else registry_models()
    registry = write_registry(tmp_path/"repo"/"configs"/"segmenters.json", models)
    commit_registry(registry)
    monkeypatch.setattr(structure_layer, "REPO_REGISTRY", registry)          # the repository registry of this test
    folds = tmp_path/"folds.json"
    subjects = {s: dict(fold="A", role="calib" if k == 0 else "train") for k, s in enumerate(fold0)}
    subjects.update({s: dict(fold="B", role="calib" if k == 0 else "train") for k, s in enumerate(fold1)})
    write_json(folds, dict(seed=1, rule="synthetic", subjects=subjects))
    stage = write_stage(tmp_path, PAIRS)
    for s in SUBJECTS:
        write_prediction(tmp_path/"pred", s, models)
        write_atropos(tmp_path/"atropos", s)
    unet = MI.UNetSource(pred_root=tmp_path/"pred", folds=folds, folds_sha256=hashlib.sha256(folds.read_bytes()).hexdigest(),
                         models_root=tmp_path/"models", image_root=tmp_path/"images")
    return dict(root=tmp_path, allowlist=allowlist, registry=registry, models=models, folds=folds, stage=stage, unet=unet,
                seg_root=tmp_path/"atropos", fold0=list(fold0), fold1=list(fold1))


def build(env, arm="A4p", **changes):
    kwargs = dict(stage=env["stage"], allowlist=env["allowlist"], unet=env["unet"] if arm == "A4p" else None)
    seg_root = changes.pop("seg_root", env["seg_root"] if arm == "A4p" else None)
    kwargs.update(changes)
    MI.build_arm(arm, env["root"]/"inputs", seg_root, **kwargs)
    return env["root"]/"inputs"/arm


def reference_a4(I, probs16):
    """Independent recomputation of the §12.3 formula from the stored float16 posterior."""
    P = probs16.astype(np.float64)
    P /= P.sum(axis=0)
    mass = P.reshape(36, -1).sum(axis=1)
    c = (P*I).reshape(36, -1).sum(axis=1)/np.where(mass > 0, mass, 1)
    w = P[1]+P[20]
    delta = .05*(w*I).sum()/w.sum()
    c[1:] += delta*np.random.default_rng(20261030).uniform(-1, 1, 35)
    J = np.einsum("l,lzyx->zyx", c[1:], P[1:])
    return MI.smooth(J), delta


# ---- the colouring -----------------------------------------------------------------------------------------------

def test_one_hot_posterior_gives_the_a3_formula():
    I = image(DEV[0])
    lab = seg35(DEV[0])
    J4, rec = MI.colour("A4p", I, None, None, onehot(lab))
    wm = float(I[np.isin(lab, (1, 20))].mean())
    J3, consts = MI.colour_labels(I, lab, range(1, 36), .05*wm*MI.z_offsets())
    assert rec["delta"] == pytest.approx(.05*wm, rel=1e-13) and rec["wm_labels"] == [1, 20]
    assert np.allclose(J4, MI.smooth(J3), rtol=1e-12, atol=1e-14)
    assert all(rec["c_by_seg35_class"][l] == pytest.approx(consts[l], rel=1e-12) for l in range(1, 36))


def test_colouring_takes_no_label():
    I = image(DEV[0])
    with pytest.raises(ValueError, match="no label"):
        MI.colour("A4p", I, seg4(DEV[0]), None, onehot(seg35(DEV[0])))


def test_absent_class_is_skipped():
    I = image(DEV[0])
    lab = seg35(DEV[0])
    lab[lab == 7] = 8
    _, rec = MI.colour("A4p", I, None, None, onehot(lab))
    assert rec["c_by_seg35_class"][7] is None and rec["c_by_seg35_class"][8] is not None


# ---- make_inputs A4p ---------------------------------------------------------------------------------------------

@synthetic_folds("env")
def test_a4p_inputs_written_and_recomputed(env):
    arm_dir = build(env)
    rec = json.loads((arm_dir/"d0_inputs.json").read_text())
    assert rec["arm"] == "A4p" and rec["oracle"] is False and rec["seed"] == 20261030 and len(rec["z_offsets"]) == 35
    git = rec["structure_registry"].pop("git")
    assert rec["structure_registry"] == dict(path=str(env["registry"]), sha256=file_sha256(env["registry"]),
                                             status="registered")
    assert git["tracked"] is True and git["clean"] is True and len(git["last_commit"]) == 40
    assert rec["group_sum_tol"] == 2e-3 and rec["alignment_qc"]["threshold"] == .8
    assert rec["alignment_qc"]["min_dice"] == 1. and "cerebral_wm" in rec["alignment_qc"]["rule"]     # synthetic: same WM
    assert "never stops" in rec["support_dice_record"]["rule"] and rec["support_dice_record"]["min_dice"] > .9
    assert rec["support_dice_record"]["n_below_revision1_threshold"] == 0
    assert rec["atropos_root"] == str(env["seg_root"])
    assert rec["subject_admission"]["subjects"] == SUBJECTS
    assert rec["subject_admission"]["folds_sha256"] == env["unet"].folds_sha256
    for f, m in PAIRS:
        name = f"{f}__{m}"
        out = arm_dir/MI.STAGE_NAME/name
        src = env["stage"]/name
        with np.load(out/"pair.npz") as z, np.load(src/"pair.npz") as s:
            assert tuple(z.files) == MI.KEY_ORDER
            for key in ("fixed_seg4", "fixed_seg35", "moving_seg4", "moving_seg35", "initial_u"):
                assert np.array_equal(z[key], s[key]) and z[key].dtype == s[key].dtype
            for side, subject in (("fixed", f), ("moving", m)):
                with np.load(env["root"]/"pred"/subject/PROBS_FILE) as p:
                    want, delta = reference_a4(s[side], p["p"])
                assert np.allclose(z[side], want, rtol=1e-12, atol=1e-14)
                assert not np.array_equal(z[side], s[side])
        man = json.loads((out/"manifest.json").read_text())
        assert man["pair_sha256"] == file_sha256(out/"pair.npz") and "baseline_native_mse" not in man
        pd = man["preprocessing_d0"]
        assert pd["arm"] == "A4p" and pd["oracle"] is False and pd["source_baseline_native_mse"] == .01
        subject_of = dict(fixed=f, moving=m)
        for side in ("fixed", "moving"):
            c = pd["colour"][side]
            assert c["policy"] == "all_folds_mean" and c["model_ids"] == ["unet35_v1_foldA", "unet35_v1_foldB"]
            assert c["registry_status"] == "registered" and len(c["members"]) == 4
            assert c["read_back_probs36"]["stored_dtype"] == "float16"
            assert 0 < c["read_back_probs36"]["raw_max_abs_sum_deviation"] <= 1e-3
            assert c["group_sum_max_abs_deviation"] <= 2e-3 and c["group_sum_tol"] == 2e-3
            assert set(c["unet_output_sha256"]) == {PROBS_FILE, GROUPS_FILE}
            assert c["alignment_qc"]["passed"] is True and c["alignment_qc"]["dice"] == 1.
            assert c["alignment_qc"]["n_unet_wm"] == np.isin(seg35(subject_of[side]), (1, 20)).sum() > 0
            assert c["qc_wm_group"] == dict(file=GROUPS_FILE, name="cerebral_wm", channel=1, labels=[1, 20])
            assert c["support_dice_record"]["stops"] is False and c["support_dice_record"]["dice"] > .9
            a = c["atropos_qc"]
            assert a["wm_file"] == "posteriors03.nii.gz" and a["wm_index"] == 2 and a["wm_class_name"] == "WM"
            assert a["segmenter_id"] == "atropos" and a["output_sha256"] == {
                n: file_sha256(env["seg_root"]/subject_of[side]/n) for n in ATROPOS_FILES}
            assert a["provenance_sha256"] == file_sha256(env["seg_root"]/subject_of[side]/"provenance.json")
            assert c["registry_git"]["tracked"] is True and c["registry_git"]["clean"] is True
        assert rec["files_sha256"][f"{MI.STAGE_NAME}/{name}/pair.npz"] == man["pair_sha256"]
        assert rec["pairs"][name]["structure_layer"]["preflight"]["contact_union_contains_pair_subject"] is False


def _refused(env, *fragments, nothing_written=False, **changes):
    with pytest.raises(SystemExit) as exc:
        build(env, **changes)
    message = str(exc.value).replace(str(env["root"]), "<tmp>")                     # tmp_path carries the test name
    assert all(f in message for f in fragments), message
    assert not (env["root"]/"inputs"/"A4p"/"d0_inputs.json").exists()
    if nothing_written:
        assert not (env["root"]/"inputs"/"A4p").exists()


@synthetic_folds("env")
def test_raw_sum_off_by_more_than_float16_rounding_is_refused(env):
    d = env["root"]/"pred"/DEV[1]
    with np.load(d/PROBS_FILE) as z:
        p = z["p"].copy()
    p[:, 2, 3, 4] = (p[:, 2, 3, 4].astype(np.float32)*1.01).astype(np.float16)     # raw sum ~1.01 on one voxel
    np.savez_compressed(d/PROBS_FILE, p=p)
    rehash(d)                                                                       # hashes agree: only the read-back sees it
    _refused(env, f"A4p moving {DEV[1]}: probs36.npz refused by the read-back", "raw channel sum", "float16")


@synthetic_folds("env")
def test_changed_probs_file_is_refused_by_its_hash(env):
    d = env["root"]/"pred"/DEV[2]
    with np.load(d/PROBS_FILE) as z:
        p = z["p"][::-1].copy()
    np.savez_compressed(d/PROBS_FILE, p=p)
    _refused(env, "structure layer refused (ProvenanceRejected", "probs36.npz")


def test_only_the_repository_registry_is_used(env, monkeypatch):
    """make_inputs always asks StructureLayer for the repository registry: a registry elsewhere (e.g. the staging copy
    of the training chain) is never read; without a repository registry the arm is refused before anything is
    written."""
    monkeypatch.setattr(structure_layer, "REPO_REGISTRY", env["root"]/"elsewhere"/"segmenters.json")
    _refused(env, "A4p: no repository registry at <tmp>/elsewhere/segmenters.json", nothing_written=True)
    assert MI.pair_structure_layer.__code__.co_varnames.count("registry_path") == 0     # no registry argument exists


REGISTRY_GIT_CASES = {                     # A4 review item 3: case -> (change, fragment of the refusal)
    "modified after the commit": (lambda r: r.write_text(r.read_text()+"\n"), "clean False"),
    "staged, not committed": (lambda r: (r.write_text(r.read_text()+"\n"), git_repo(r.parent.parent)("add", "-A")),
                              "clean False"),
    "untracked": (lambda r: git_repo(r.parent.parent)("rm", "-q", "--cached", str(r.relative_to(r.parent.parent))),
                  "tracked False"),
    "no work tree": (lambda r: (r.parent.parent/".git").rename(r.parent.parent/"dot_git_moved"), "tracked False"),
}


@pytest.mark.parametrize("case", sorted(REGISTRY_GIT_CASES))
def test_registry_must_be_committed_and_clean(env, case):
    """A4 review item 3: besides its sha256, the repository registry must be tracked by git and clean (no staged,
    unstaged or untracked change); otherwise make_inputs stops before anything is written."""
    change, fragment = REGISTRY_GIT_CASES[case]
    change(env["registry"])
    _refused(env, "A4p: the repository registry <tmp>/repo/configs/segmenters.json must be committed and clean",
             fragment, nothing_written=True)


def test_registry_git_state_is_checked_again_per_pair(env, monkeypatch):
    """The per-pair StructureLayer record is checked too (the registry could change while make_inputs runs)."""
    real = structure_layer.registry_git_state
    calls = []

    def flaky(path):
        calls.append(path)
        state = real(path)
        return state if len(calls) == 1 else dict(state, clean=False, porcelain=" M configs/segmenters.json")
    monkeypatch.setattr(structure_layer, "registry_git_state", flaky)
    _refused(env, "A4p: the repository registry", "must be committed and clean", "clean False")
    assert len(calls) >= 2


def test_model_that_saw_a_pair_subject_is_refused(env):
    models = registry_models(fold0=FOLD0+[DEV[1]])
    write_registry(env["registry"], models)
    commit_registry(env["registry"], "leaky registry")
    _refused(env, "structure layer refused (ProvenanceRejected: D24 pair-level label contact", DEV[1])


def test_training_subject_is_refused_before_anything_is_written(env):
    doc = json.loads(env["folds"].read_text())
    doc["subjects"][DEV[2]] = dict(fold="B", role="train")
    write_json(env["folds"], doc)
    unet = MI.UNetSource(**dict(vars(env["unet"]), folds_sha256=hashlib.sha256(env["folds"].read_bytes()).hexdigest()))
    _refused(env, "A4p: subjects refused (D24)", DEV[2], "a B-line training subject (folds train/calib)",
             nothing_written=True, unet=unet)


def test_folds_file_must_have_the_pinned_hash(env):
    unet = MI.UNetSource(**dict(vars(env["unet"]), folds_sha256="0"*64))
    _refused(env, "A4p: <tmp>/folds.json sha256", "!= pinned " + "0"*64, nothing_written=True, unet=unet)


@synthetic_folds("env")
def test_probs_disagreeing_with_the_group_file_are_refused(env):
    d = env["root"]/"pred"/DEV[0]
    groups = group_probabilities(posterior(DEV[0]), GROUPING, dtype=np.float32)
    groups[[1, 2]] = groups[[2, 1]]                                                 # swap two group channels
    np.savez_compressed(d/GROUPS_FILE, p=groups.astype(np.float16))
    rehash(d)
    _refused(env, f"A4p fixed {DEV[0]}: probs36 summed by structure13_v1 differs from structure_groups.npz")


# ---- the alignment QC (protocol §12.4 item 5, revision 2) ----------------------------------------------------------

def gyral_phantom(noise_seed=None):
    """A white-matter-like posterior on (40, 44, 48): a gyral pattern of thin folds inside an ellipsoid off-centre in
    every axis (so no flip maps it onto itself); with `noise_seed` a second segmenter's noisy view of it. The
    posterior is > .5 exactly where the smoothed pattern is > .3."""
    shape = (40, 44, 48)
    z, y, x = np.meshgrid(*[np.arange(n, dtype=float) for n in shape], indexing="ij")
    inside = ((z-17)/14)**2+((y-24)/17)**2+((x-21)/18)**2 < 1
    field = np.cos(2*np.pi*x/7+.3)+np.cos(2*np.pi*y/9+.2*z)+.6*np.cos(2*np.pi*z/8)
    field[~inside] = -3.
    if noise_seed is not None:
        field = field+.4*np.random.default_rng(noise_seed).standard_normal(shape)
    return 1/(1+np.exp(-4*(gaussian_filter(field, 1.)-.3)))


NEGATIVE_CONTROLS = {"flip left-right (x)": lambda u: u[:, :, ::-1], "flip anterior-posterior (y)": lambda u: u[:, ::-1],
                     "flip superior-inferior (z)": lambda u: u[::-1],
                     **{f"shift {s} voxels along axis {k}": (lambda u, s=s, k=k: np.roll(u, s, axis=k))
                        for s in (2, 4, 8) for k in range(3)}}


def test_alignment_qc_passes_aligned_white_matter_and_stops_flips_and_shifts():
    """Dice({U-Net cerebral_wm > .5}, {Atropos WM > .5}) >= .80: two views of the same thin-fold white matter agree
    (Dice > .9); every flip and every shift of 2, 4 or 8 voxels along any axis drops below .7 and is stopped (the
    real-data separation, aligned >= .853 against flips and >= 2-voxel shifts <= .792 (1-voxel shifts not caught), is in runs/20261030-d0-a4/qc_survey_20261030/)."""
    atropos, unet = gyral_phantom(), gyral_phantom(noise_seed=0)
    aligned = MI.alignment_qc(unet, atropos)
    assert aligned["passed"] is True and .9 < aligned["dice"] < 1. and aligned["threshold"] == MI.ALIGN_QC_MIN == .8
    assert aligned["n_unet_wm"] == int((unet > .5).sum()) and aligned["n_atropos_wm"] == int((atropos > .5).sum())
    for name, control in NEGATIVE_CONTROLS.items():
        qc = MI.alignment_qc(control(unet), atropos)
        assert qc["passed"] is False and qc["dice"] < .7, (name, qc["dice"])
    with pytest.raises(SystemExit, match="differ"):                     # an axis-order error changes the shape here
        MI.alignment_qc(unet.transpose(2, 1, 0), atropos)


def test_alignment_qc_threshold_and_level():
    u = np.zeros((4, 5, 6))
    u[1:3] = .9
    a = u.copy()
    assert MI.alignment_qc(u, a) == dict(MI.alignment_qc(u, a), dice=1., passed=True, n_both=60, level=.5)
    a[0, 0, :] = .7                                                      # 6 extra WM voxels: 2*60/(60+66)
    assert MI.alignment_qc(u, a)["dice"] == pytest.approx(120/126)
    assert MI.alignment_qc(u, a, threshold=120/126)["passed"] and not MI.alignment_qc(u, a, threshold=.96)["passed"]
    a[0, 0, :] = .5                                                      # > .5 is strict
    assert MI.alignment_qc(u, a)["dice"] == 1.
    empty = MI.alignment_qc(np.zeros_like(u), np.zeros_like(u))
    assert empty["dice"] is None and empty["passed"] is False


def test_support_dice_is_recorded_and_never_stops():
    """Revision 1's Dice({I > 0}, {P_0 < .5}) is still computed per image, as a record only (no `passed`)."""
    I = np.zeros((4, 5, 6))
    I[1:3] = 1.
    p0 = np.ones_like(I)
    p0[1:3] = 0.
    rec = MI.support_dice_record(I, p0)
    assert rec == dict(rec, dice=1., stops=False, below_revision1_threshold=False, n_both=60, revision1_threshold=.8)
    assert "passed" not in rec
    shifted = np.ones_like(I)
    shifted[2:4] = 0.                                                    # half of the support overlaps
    rec = MI.support_dice_record(I, shifted)
    assert rec["dice"] == pytest.approx(.5) and rec["below_revision1_threshold"] is True and rec["stops"] is False
    empty = MI.support_dice_record(np.zeros_like(I), np.ones_like(I))
    assert empty["dice"] is None and empty["below_revision1_threshold"] is True


U_NET_MISALIGNMENTS = {"flip superior-inferior": lambda p: p[:, ::-1], "flip anterior-posterior": lambda p: p[:, :, ::-1],
                       "flip left-right": lambda p: p[:, :, :, ::-1], "shift 2 voxels": lambda p: np.roll(p, 2, axis=3)}


@pytest.mark.parametrize("case", sorted(U_NET_MISALIGNMENTS))
@synthetic_folds("env")
def test_misaligned_unet_prediction_is_refused_by_the_alignment_qc(env, case):
    """A U-Net prediction flipped or shifted against its image, written with consistent hashes (every StructureLayer
    check passes), has white matter disagreeing with Atropos: the pair is refused (STOP, no d0_inputs.json)."""
    shutil.rmtree(env["root"]/"pred"/DEV[1])
    probs = np.ascontiguousarray(U_NET_MISALIGNMENTS[case](posterior(DEV[1])))
    write_prediction(env["root"]/"pred", DEV[1], env["models"], probs=probs)
    _refused(env, f"A4p {PAIRS[0][0]}__{PAIRS[0][1]}:moving: alignment QC Dice({{U-Net cerebral_wm > .5}}, "
                  "{Atropos WM > .5})", "< 0.8", "revision 2; the pair is refused")


@synthetic_folds("env")
def test_low_support_dice_with_agreeing_white_matter_passes(env):
    """Revision 2: a posterior calling three quarters of the image background except its cerebral white matter has
    Dice({I > 0}, {P_0 < .5}) ~ .45 (revision 1 would have stopped) while its white matter agrees with Atropos: the arm
    is written, the low Dice recorded."""
    probs = posterior(DEV[1]).astype(np.float64)
    region = np.zeros(SHAPE, bool)
    region[:6] = True
    region &= ~np.isin(seg35(DEV[1]), MI.CEREBRAL_WM)
    probs[:, region] = 0.
    probs[0, region] = 1.
    shutil.rmtree(env["root"]/"pred"/DEV[1])
    write_prediction(env["root"]/"pred", DEV[1], env["models"], probs=probs.astype(np.float32))
    rec = json.loads((build(env)/"d0_inputs.json").read_text())
    for name, side in ((f"{PAIRS[0][0]}__{PAIRS[0][1]}", "moving"), (f"{PAIRS[1][0]}__{PAIRS[1][1]}", "fixed")):
        c = rec["pairs"][name][side]                                     # DEV[1] in both pairs
        assert c["support_dice_record"]["dice"] < .6 and c["support_dice_record"]["below_revision1_threshold"] is True
        assert c["support_dice_record"]["stops"] is False
        assert c["alignment_qc"]["passed"] is True and c["alignment_qc"]["dice"] == 1.
    assert rec["support_dice_record"]["n_below_revision1_threshold"] == 2 and rec["support_dice_record"]["min_dice"] < .6
    assert rec["alignment_qc"]["min_dice"] == 1.


def test_missing_atropos_output_is_refused_before_anything_is_written(env):
    shutil.rmtree(env["seg_root"]/DEV[2])
    _refused(env, f"A4p: no Atropos posteriors under <tmp>/atropos for ['{DEV[2]}']", "alignment QC",
             nothing_written=True)
    _refused(env, "A4p: no Atropos posteriors under None for", nothing_written=True, seg_root=None)


def _edit_provenance(directory, **changes):
    write_json(directory/"provenance.json", dict(json.loads((directory/"provenance.json").read_text()), **changes))


ATROPOS_DEFECTS = {                      # case -> (change of DEV[1]'s Atropos directory, fragment of the refusal)
    "posterior file missing": (lambda d: (d/"posteriors03.nii.gz").unlink(),
                               "ProvenanceRejected: posterior file 'posteriors03.nii.gz' missing"),
    "posterior file changed": (lambda d: shutil.copyfile(d/"posteriors01.nii.gz", d/"posteriors03.nii.gz"),
                               "'posteriors03.nii.gz': sha256 differs from output_sha256"),
    "bound to another image": (lambda d: _edit_provenance(d, input_sha256={"aligned_norm.nii.gz": source_sha(DEV[0])}),
                               "input_sha256 must be exactly the moving source image"),
    "one file fewer than classes": (lambda d: _edit_provenance(d, posterior_files=ATROPOS_FILES[:2]),
                                    "must be one per class"),
}


@pytest.mark.parametrize("case", sorted(ATROPOS_DEFECTS))
@synthetic_folds("env")
def test_defective_atropos_output_is_refused_per_image(env, case):
    change, fragment = ATROPOS_DEFECTS[case]
    change(env["seg_root"]/DEV[1])
    _refused(env, f"A4p moving {DEV[1]}: Atropos", fragment)


@synthetic_folds("env")
def test_the_white_matter_file_follows_wm_index(env):
    shutil.rmtree(env["seg_root"]/DEV[0])
    write_atropos(env["seg_root"], DEV[0], posteriors=atropos_posteriors(DEV[0])[::-1], class_names=("WM", "GM", "CSF"),
                  wm_index=0)
    rec = json.loads((build(env)/"d0_inputs.json").read_text())
    c = rec["pairs"][f"{PAIRS[0][0]}__{PAIRS[0][1]}"]["fixed"]
    assert c["atropos_qc"]["wm_file"] == "posteriors01.nii.gz" and c["atropos_qc"]["wm_index"] == 0
    assert c["alignment_qc"]["dice"] == 1.


def test_existing_arm_directory_is_never_overwritten(env):
    (env["root"]/"inputs"/"A4p").mkdir(parents=True)
    with pytest.raises(SystemExit, match="refusing to overwrite.*rename it first"):
        build(env)


def test_a4p_is_never_written_under_d0_v1_nor_mixed_with_other_arms():
    with pytest.raises(SystemExit, match="never written under"):
        MI.main(["--arms", "A4p", "--inputs", str(MI.D0V1/"inputs")])
    with pytest.raises(SystemExit, match="on its own"):
        MI.main(["--arms", "A1", "A4p"])
    with pytest.raises(SystemExit, match="never written under"):
        MI.main(["--write-a4-subjects", str(MI.D0V1/"a4_subjects.txt")])


def test_a1_unchanged_by_the_refactor(env):
    arm_dir = build(env, arm="A1")
    f, m = PAIRS[0]
    with np.load(arm_dir/MI.STAGE_NAME/f"{f}__{m}"/"pair.npz") as z:
        J, _ = MI.colour_labels(image(f), seg4(f).astype(float), range(1, 5))
        assert np.array_equal(z["fixed"], MI.smooth(J))
    rec = json.loads((arm_dir/"d0_inputs.json").read_text())
    assert rec["z_offsets"] is None and "structure_registry" not in rec


# ---- --write-a4-subjects -----------------------------------------------------------------------------------------

def test_write_a4_subjects(env, capsys):
    out = env["root"]/"lists"/"a4_subjects.txt"
    assert MI.write_a4_subjects(out, stage=env["stage"], allowlist=env["allowlist"], unet=env["unet"]) == SUBJECTS
    assert out.read_text().split() == SUBJECTS
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        MI.write_a4_subjects(out, stage=env["stage"], allowlist=env["allowlist"], unet=env["unet"])
    unet = MI.UNetSource(**dict(vars(env["unet"]), pred_root=env["root"]/"pred_partial"))
    write_prediction(unet.pred_root, DEV[1], env["models"])
    chosen = MI.write_a4_subjects(env["root"]/"missing.txt", stage=env["stage"], allowlist=env["allowlist"], unet=unet,
                                  only_missing=True)
    assert chosen == [DEV[0], DEV[2]]
    assert f'"already_predicted": [\n  "{DEV[1]}"\n ]' in capsys.readouterr().out


# ---- evaluate.py: decide_a4 and summarize-a4 ---------------------------------------------------------------------

PAIR_NAMES = [f"OASIS_OAS1_{438+i:04d}_MR1__OASIS_OAS1_{439+i:04d}_MR1" for i in range(19)]


def fake_runs(deltas, *, a1=.04, a3=.03):
    runs = {}
    for p, d in zip(PAIR_NAMES, deltas):
        runs[("A0", p)] = dict(metrics=dict(seg35=.80, seg4=.88))
        runs[("A1", p)] = dict(metrics=dict(seg35=.80+a1, seg4=.88+a1))
        runs[("A3", p)] = dict(metrics=dict(seg35=.80+a3, seg4=.88+a3))
        if d is not None:
            runs[("A4p", p)] = dict(metrics=dict(seg35=.80+d, seg4=.88+d/2))
    return runs


def test_decide_a4_rows():
    dec = EV.decide_a4(fake_runs([.02]*19), PAIR_NAMES)
    assert dec["rows"] == ["R4a"] and dec["A4p_minus_A0"]["positive"] == 19
    assert dec["conclusion"].startswith("R4a（单通道诊断）") and "不直接预测" in dec["conclusion"]   # A4 review item 8
    r = dec["recovery_descriptive"]["seg35"]
    assert r["recovery_vs_A1"] == pytest.approx(.5) and r["recovery_vs_A3"] == pytest.approx(2/3)
    assert dec["recovery_descriptive"]["seg4"]["recovery_vs_A1"] == pytest.approx(.25)
    assert EV.decide_a4(fake_runs([.001]*19), PAIR_NAMES)["rows"] == ["R4b"]
    middle = EV.decide_a4(fake_runs([.007]*19), PAIR_NAMES)
    assert middle["rows"] == [] and "中间区域" in middle["conclusion"]
    few_positive = EV.decide_a4(fake_runs([.05]*13+[-.01]*6), PAIR_NAMES)
    assert few_positive["A4p_minus_A0"]["median"] == pytest.approx(.05) and few_positive["rows"] == []


def test_decide_a4_missing_pairs():
    two_missing = EV.decide_a4(fake_runs([.02]*17+[None]*2), PAIR_NAMES)
    assert two_missing["rows"] == ["R4a"] and two_missing["A4p_minus_A0"]["n"] == 17
    not_enough_positive = EV.decide_a4(fake_runs([.02]*13+[-.01]*4+[None]*2), PAIR_NAMES)
    assert not_enough_positive["A4p_minus_A0"]["positive"] == 13 and "R4a" not in not_enough_positive["rows"]
    three_missing = EV.decide_a4(fake_runs([.02]*16+[None]*3), PAIR_NAMES)
    assert three_missing["rows"] == [] and three_missing["conclusion"].startswith("不可判")


def write_run(root, arm, pair, seg35_value, *, load=(1.2, 2.5, 2.0), commits=30):
    d = root/arm/pair
    d.mkdir(parents=True)
    metrics = {k: 0. for k in EV.LONG_MEASURES}
    metrics.update(seg35=seg35_value, seg4=seg35_value+.06, hd95=1.8, sdlogj_official=.08, sdlogj_brain=.6)
    write_json(d/"metrics.json", dict(run_id=f"{arm}_{pair[11:15]}", run_dir=str(d), arm=arm, pair=pair, metrics=metrics,
                                      certificate=dict(passed=True, min_jd_lower_bound=1e-6), self_check=dict(passed=True),
                                      native_commits=commits, e2e_seconds=88.5, success=False,
                                      evaluator=dict(sha256=f"sha-{arm}")))
    (d/"log.txt").write_text(f"2026-10-30T10:00:00Z launch tag-x-{arm}; loadavg {load[0]} {load[1]} {load[2]}; "
                             "MemAvailable 9000 MiB; gpu x\n")


def test_summarize_a4(tmp_path):
    stage = tmp_path/"stage"
    stage.mkdir()
    write_json(stage/"selection.json", dict(pairs=[dict(pair=p) for p in PAIR_NAMES]))
    d0, a4 = tmp_path/"d0_runs", tmp_path/"a4_runs"
    for i, p in enumerate(PAIR_NAMES):
        for arm, value in (("A0", .80), ("A1", .84), ("A2", .77), ("A3", .83)):
            write_run(d0, arm, p, value)
        write_run(a4, "A4p", p, .82 if i < 15 else .79, load=(.4, .8, 1.), commits=32)
    out = tmp_path/"analysis"
    assert EV.summarize_a4(a4, d0, out, stage=stage) == 0
    dec = json.loads((out/"decision_a4.json").read_text())
    assert dec["rows"] == ["R4a"] and dec["A4p_minus_A0"]["positive"] == 15
    assert dec["recovery_descriptive"]["seg35"]["recovery_vs_A1"] == pytest.approx(.5)
    assert dec["native_commits_A4p_minus_A0"]["median"] == 2
    assert dec["launch_loadavg_median"]["A4p"] == dict(one_min=.4, five_min=.8)
    assert dec["launch_loadavg_median"]["A0"] == dict(one_min=1.2, five_min=2.5)
    assert dec["sources"]["evaluator_sha256"]["A4p"] == ["sha-A4p"]
    text = (out/"summary_a4.md").read_text()
    assert "seg35" in text and "seg4" in text and "A0 不是同时段对照" in text and "A4′ − A0" in text
    assert "开发集" in text and "单次观测" in text
    # A4 review item 7: the evaluator hashes per arm, and that they differ between A0-A3 and A4'
    assert "评测脚本版本" in text and "A0 sha-A0" in text and "A4′ sha-A4p" in text and "evaluate_run 没有改" in text
    rows = (out/"long.csv").read_text().splitlines()
    assert any(",A4p," in r and ",seg4," in r for r in rows) and any("launch_loadavg_5min" in r for r in rows)
    with pytest.raises(SystemExit, match="never written under"):
        EV.summarize_a4(a4, d0, EV.D0/"analysis_a4", stage=stage)


def test_run_queue_syntax_and_a4_mode():
    import subprocess
    script = ROOT/"experiments"/"d0"/"run_queue.sh"
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    text = script.read_text()
    assert "D0=$RESULTS/d0_v1_a4; ARMS=(A4p); SELF_RE='^tag-d0a4-'" in text and "'tag-unet35-train*' 'tag-d0-*'" in text
    assert "summarize-a4 --a4-runs-root" in text
    # A4 review item 1: one lock file for the worktree link, the same in both modes (no per-mode .prepare.lock)
    assert ".prepare.lock" not in text and text.count('exec 9> "$LINK_LOCK"') == 1
    assert "LINK_LOCK=$RESULTS/.frozen_${FROZEN}_link.lock" in text


# ---- run_queue.sh --a4 on a synthetic tree (A4 review item 2: the start checks run, not only string matches) --------

QUEUE = ROOT/"experiments"/"d0"/"run_queue.sh"
FAKE_PAIRS = [f"OASIS_OAS1_{900+i:04d}_MR1__OASIS_OAS1_{901+i:04d}_MR1" for i in range(19)]


@pytest.fixture
def queue_tree(tmp_path):
    """A fake repository (git, committed registry, a 19-pair stage selection), results root, A4p inputs record,
    systemctl and nvidia-smi; nothing outside tmp_path is read or written by the queue script."""
    repo, results, fakebin = tmp_path/"repo", tmp_path/"results", tmp_path/"bin"
    stage = repo/"data"/"three_d"/"stage_L2R_val_v1"
    stage.mkdir(parents=True)
    write_json(stage/"selection.json", dict(pairs=[dict(pair=p) for p in FAKE_PAIRS]))
    registry = write_registry(repo/"configs"/"bayesreg"/"segmenters.json", registry_models(), ensemble=False)
    git = git_repo(repo)
    git("add", "configs/bayesreg/segmenters.json")
    git("commit", "-q", "-m", "registry")
    record = results/"d0_v1_a4"/"inputs"/"A4p"/"d0_inputs.json"
    record.parent.mkdir(parents=True)
    write_json(record, dict(structure_registry=dict(path=str(registry), sha256=file_sha256(registry), status="registered",
                                                    git=dict(tracked=True, clean=True))))
    fakebin.mkdir()
    (fakebin/"systemctl").write_text('#!/bin/bash\nprintf "%b" "${FAKE_UNITS:-}"\n')
    (fakebin/"nvidia-smi").write_text('#!/bin/bash\ncase "$*" in\n  *query-compute-apps*) printf "%b" "${FAKE_PIDS:-}";;\n'
                                      '  *utilization.gpu*) echo "${FAKE_UTIL:-0}";;\n  *) echo "fake gpu";;\nesac\n')
    for f in fakebin.iterdir():
        f.chmod(0o755)
    import os
    environment = dict(os.environ, PATH=f"{fakebin}:{os.environ['PATH']}", D0Q_REPO=str(repo), D0Q_RESULTS=str(results),
                       D0Q_SMI=str(fakebin/"nvidia-smi"), FAKE_UNITS="", FAKE_PIDS="", FAKE_UTIL="3")
    return dict(repo=repo, results=results, registry=registry, record=record, git=git, env=environment)


def run_queue(tree, *args, **env):
    import subprocess
    done = subprocess.run(["bash", str(QUEUE), *args], env=dict(tree["env"], **env), capture_output=True, text=True,
                          timeout=120)
    log = tree["results"]/"d0_v1_a4"/"queue.log"
    return done.returncode, done.stdout, log.read_text() if log.exists() else ""


def test_run_queue_a4_dry_run_lists_the_order(queue_tree):
    code, out, log = run_queue(queue_tree, "--a4", "--dry-run")
    assert code == 0 and log == ""
    assert out.splitlines() == [f"{i+1} A4p {p}" for i, p in enumerate(FAKE_PAIRS)]
    assert not (queue_tree["results"]/"d0_v1_a4"/"runs").exists()                    # a dry run creates nothing
    code, out, _ = run_queue(queue_tree, "--dry-run")                                 # D0 mode: order unchanged
    arms = ("A0", "A1", "A2", "A3")
    assert code == 0 and out.splitlines() == [f"{4*i+j+1} {arms[(i+j) % 4]} {p}" for i, p in enumerate(FAKE_PAIRS)
                                              for j in range(4)]
    assert not (queue_tree["results"]/"d0_v1").exists()


A4_START_STOPS = {                         # case: (environment / change, STOP fragment in queue.log)
    "segmenter training still active": (dict(FAKE_UNITS="tag-unet35-train.service loaded active running x\\n"),
                                         "STOP: units still active: tag-unet35-train.service"),
    "a D0 unit active": (dict(FAKE_UNITS="tag-d0-A0-0438-0439.service loaded active running x\\n"),
                         "STOP: units still active: tag-d0-A0-0438-0439.service"),
    "GPU compute process": (dict(FAKE_PIDS="4242\\n"), "STOP: GPU not idle at start (compute pids: 4242"),
    "GPU utilization": (dict(FAKE_UTIL="55"), "STOP: GPU not idle at start"),
    "registry modified, not committed": ("modify", "STOP: the repository registry"),
    "inputs record from another registry": ("record_sha", "STOP: the A4p inputs were not made with the current"),
    "inputs record without a clean registry": ("record_git", "STOP: the A4p inputs were not made with the current"),
}


@pytest.mark.parametrize("case", sorted(A4_START_STOPS))
def test_run_queue_a4_start_checks_stop(queue_tree, case):
    change, fragment = A4_START_STOPS[case]
    env = change if isinstance(change, dict) else {}
    if change == "modify":
        queue_tree["registry"].write_text(queue_tree["registry"].read_text()+"\n")
    elif change in ("record_sha", "record_git"):
        doc = json.loads(queue_tree["record"].read_text())
        if change == "record_sha":
            doc["structure_registry"]["sha256"] = "0"*64
        else:
            doc["structure_registry"]["git"]["clean"] = False
        write_json(queue_tree["record"], doc)
    code, _, log = run_queue(queue_tree, "--a4", **env)
    assert code == 1 and fragment in log, log
    assert "worktree HEAD" not in log and not list((queue_tree["results"]/"d0_v1_a4"/"runs").glob("*/*"))


def test_run_queue_a4_start_checks_pass_on_a_quiet_machine(queue_tree):
    """Every A4p start check passes; the queue then stops at the frozen-worktree check (there is no eb808be worktree in
    the synthetic tree), before any run directory or unit."""
    code, _, log = run_queue(queue_tree, "--a4")
    assert code == 1 and "STOP: worktree HEAD is not eb808be" in log, log
    assert "A4p registry check:" in log and "; ok" in log
    assert "units still active" not in log and "GPU not idle" not in log
    assert not list((queue_tree["results"]/"d0_v1_a4"/"runs").glob("*/*"))
    assert not (queue_tree["results"]/"d0_v1").exists()                               # nothing under d0_v1/
