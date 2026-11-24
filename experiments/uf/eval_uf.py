"""The one segmentation-level evaluation of work package UF (runs/20261105-uf/plan.md): the 20 development subjects, argmax against the truth, per label, seg35 = nanmean over labels 1..35, the small six, the big six, the other 23, seg4 through configs/bayesreg/seg4_of_seg35_v1.json, the cortex and the white matter
(seg4 labels 1 and 3), per subject (the functions of experiments/unv2/eval_seg.py and experiments/nn/eval_nn.py); OASIS_OAS1_0440_MR1 on its own in the reading.

Arms (one pass per subject; no prediction file is kept):
  UF4         (a) the four members of <member-root> (foldA_s0, foldA_s1, foldB_s0, foldB_s1): the mean over members of softmax(z_m / T_m) (bayesreg.seg.unet.infer.ensemble_probabilities, the order of v2's registry: A0 A1 B0 B1), each member on its sliding window with its Gaussian
              window (as trained), stored as v2's probs36 is (float16), argmax, ties to the smallest label: what a registered UF4 would give through eval_seg.
  UF4_nn      (b) 1/2 of UF4's probabilities (the float16 above, read back as every reader does: float64 divided by the channel sum) plus 1/2 of nnU-Net's (<out>/nn/probs/<subject>/probs36.npz, read_back_probabilities), argmax. The weights are fixed, never tuned.
  UF_A0 UF_A1 UF_B0 UF_B1   (d) each member alone: argmax of its logits (experiments/ul/eval_u1.py's path). Described.
  nn          nnU-Net alone, argmax of its stored probabilities. Described (NN read this model already: .9271 / .9590).
  v1_control, v2_control    (c) the stored probs36 of unet35_v1_ens and unet35_v2_full (eval_seg.read_prediction) through this same code: they must reproduce the committed rows (CONTROL_TOL_STORED, 1e-9: no network runs).
  U1_control  member U1 (runs/20261104-ul) through this same forward path (sliding window, Gaussian window, argmax of the logits): it must reproduce eval_u1.json's U1 (CONTROL_TOL_FORWARD, 2e-4: a GPU forward pass). The control of the forward path of (a) and (d); an addition of the implementer to the brief's (c).
References (pinned by sha256 on real data): unet35_v2_full/eval_seg/eval_seg.json (the v1 and v2 ensembles, the reading's bar), runs/20261103-unv2/v2_member_subsets.json (v2's single members), unet35_v3_large/eval_u1/eval_u1.json (U1).

Order of the checks: the read lock (uf_lock.py), first of all on real data; then, before any file of a subject is opened, every subject in the allowlist source stage_L2R_val_v1, not never_for_training, on no sealed list, no training subject, outside the label contact of each UF member, of U1 and of nnU-Net (prepare_manifest.json), and the reference's subjects;
then, before the lock: the four members as the plan trained them (uf_common.member_record_problems: recipe, the three changes, the last step, the subjects of their v2 namesakes), the gate of the nnU-Net probabilities (convert_record.json) and the stored files' hashes, the control predictions' provenance and files, the sha256 of every image against
the provenance of v1, v2 and the conversion. A failure there costs no read. The lock is taken immediately before the first truth label; a failure after it keeps the lock.
Real data: every input the plan's, clean code. --synthetic: synthetic ids and a non-real data root only, no pins. Outputs in --out-dir (never overwritten): eval_uf.json, per_subject.csv, per_label.csv.

    python -B experiments/uf/eval_uf.py --out-dir DIR [--device cuda] [--bz-go TEXT]
    python -B experiments/uf/eval_uf.py --control-only --out-dir DIR [--device cuda]       the chain's first step: v1_control, v2_control, U1_control alone; no lock (evaluate_control)
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(HERE), str(REPO/"experiments/unv2"), str(REPO/"experiments/ul"), str(REPO/"experiments/nn")]
import eval_nn  # noqa: E402
import eval_seg  # noqa: E402
import eval_u1  # noqa: E402
import uf_common as uc  # noqa: E402
import uf_lock  # noqa: E402
import uf_nn  # noqa: E402
from bayesreg.eval.truth import load_sealed_lists  # noqa: E402
from bayesreg.io.guard import SubjectSealed, load_dev_allowlist  # noqa: E402
from bayesreg.seg.channels import read_back_probabilities  # noqa: E402
from bayesreg.seg.unet.data import OasisFiles, load_folds, normalize_image, read_image  # noqa: E402
from bayesreg.seg.unet.infer import ensemble_probabilities  # noqa: E402
from bayesreg.seg.unet.large import GIB, cap_gpu_memory  # noqa: E402
from bayesreg.seg.unet.pins import check_synthetic_data_root, check_synthetic_subjects, load_pinned_allowlist, source_root_for  # noqa: E402
from bayesreg.seg.unet.snapshot import code_record, require_clean_code  # noqa: E402
from bayesreg.seg.unet.train import predict_logits  # noqa: E402

FORMAT = "bayesreg.uf.eval.v1"
ENSEMBLE_ARMS = ("UF4", "UF4_nn")
ENSEMBLE_ORDER = tuple(sorted(uc.MEMBER_NAMES))                       # foldA_s0, foldA_s1, foldB_s0, foldB_s1: v2's registry order (fold models sorted, seeds in order)
MEMBER_ARMS = tuple(f"UF_{uc.V2_NAME[n]}" for n in ENSEMBLE_ORDER)     # UF_A0, UF_A1, UF_B0, UF_B1
CONTROL_ARMS = ("v1_control", "v2_control", "U1_control")
ARMS = ENSEMBLE_ARMS+MEMBER_ARMS+("nn",)+CONTROL_ARMS
METRICS = eval_nn.METRICS                                             # seg35, seg4, small_six, big_six, other_23, cortex, white_matter
MEMBER_METRICS = ("seg35", "seg4")+tuple(eval_seg.GROUPS)              # what v2_member_subsets.json has
SEGMENTERS = dict(v1_control="unet35_v1_ens", v2_control="unet35_v2_full")
GPU_CAP_GIB = 20.
SLAB = 8                                                              # z slices per step of the half-and-half argmax (host memory)


def half_and_half(p16, p_nn, *, weight=uc.NN_WEIGHT, slab=SLAB):
    """argmax over classes of (1-weight) * P_UF + weight * P_nn, ties to the smallest label; P_UF = the float16 probabilities of UF4 read back (float64 divided by the channel sum), P_nn float64 as read_back_probabilities gives it. weight is the plan's 1/2 and nothing else. By slabs of z, the same numbers as at once."""
    if weight != uc.NN_WEIGHT:
        raise ValueError(f"the weight of arm (b) is fixed at {uc.NN_WEIGHT}")
    if p16.dtype != np.float16 or p16.shape != p_nn.shape:
        raise ValueError(f"UF4 float16 and nnU-Net probabilities of one shape required: {p16.dtype} {p16.shape}, {p_nn.shape}")
    out = np.empty(p16.shape[1:], dtype=np.uint8)
    for z0 in range(0, p16.shape[1], slab):
        part = p16[:, z0:z0+slab].astype(np.float64)
        part /= part.sum(axis=0)
        out[z0:z0+slab] = ((1-weight)*part+weight*p_nn[:, z0:z0+slab]).argmax(axis=0)
    return out


def read_u1_reference(path, *, pinned_sha256=None):
    """The rows of U1 in eval_u1.json, as eval_nn.metrics_of rows; ValueError if the file is not the pinned one or its means do not follow from its rows."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if pinned_sha256 is not None and digest != pinned_sha256:
        raise ValueError(f"{path}: sha256 {digest} is not the pinned {pinned_sha256}")
    document = json.loads(raw.decode("utf-8"))
    subjects = list(document["subjects"])
    rows = [eval_nn.metrics_of({k: document["rows"]["U1"][s][k] for k in ("seg35", "seg4", "groups", "seg4_per_label", "per_label")}) for s in subjects]
    means = {m: eval_nn.mean_of(rows, m) for m in METRICS}
    for metric in ("seg35", "seg4"):
        if abs(means[metric]-document["means"]["U1"][metric]) > 1e-12:
            raise ValueError(f"{path}: the mean of U1 {metric} does not follow from its rows")
    return dict(path=str(path), sha256=digest, subjects=subjects, rows=rows, means=means, member_sha256=document["members"]["U1"]["member_sha256"])


def read_inputs(*, references, data_root, allowlist_path, dev_source, folds_path, member_root, nn_root, nn_manifest, v1_pred, v2_pred, u1_member, synthetic):
    """(allowlist, its record, the references): pinned on real data; synthetic ids and a non-real data root otherwise."""
    if synthetic:
        check_synthetic_data_root(data_root)
        allowlist = load_dev_allowlist(allowlist_path)
        check_synthetic_subjects([s for ids in allowlist.subjects.values() for s in ids], "the allowlist")
        record = dict(mode="synthetic", allowlist_path=str(allowlist_path))
        pins = dict(ensembles=None, members=None, u1=None)
    else:
        plan = ((references["ensembles"], uc.REF_ENSEMBLES), (references["members"], REPO/uc.REF_MEMBERS_REL), (references["u1"], uc.REF_U1), (data_root, eval_seg.DATA_ROOT), (allowlist_path, eval_seg.ALLOWLIST), (folds_path, eval_seg.FOLDS), (member_root, uc.OUT), (nn_root, uf_nn.nn_dir(uc.OUT)),
                (nn_manifest, uc.NN_OUT/"prepare_manifest.json"), (v1_pred, uc.V1_PRED), (v2_pred, uc.V2_PRED), (u1_member, uc.U1_MEMBER))
        if dev_source != eval_seg.DEV_SOURCE or any(Path(given).resolve() != Path(want).resolve() for given, want in plan):
            raise ValueError("real data: the references, the data root, the allowlist, the folds, the members, the nnU-Net probabilities and manifest, the controls and the development source are the plan's; other values only with --synthetic")
        allowlist, record = load_pinned_allowlist(allowlist_path)
        record = {k: v for k, v in record.items() if k != "code"}
        pins = dict(ensembles=uc.REF_ENSEMBLES_SHA256, members=uc.REF_MEMBERS_SHA256, u1=uc.REF_U1_SHA256)
    refs = dict(ensembles=eval_nn.read_reference(references["ensembles"], pinned_sha256=pins["ensembles"]),
                members={uc.V2_NAME[n]: eval_u1.read_reference(references["members"], pinned_sha256=pins["members"], member=uc.V2_NAME[n]) for n in uc.MEMBER_NAMES},
                u1=read_u1_reference(references["u1"], pinned_sha256=pins["u1"]))
    return allowlist, record, refs


def load_members(member_root, device, *, v2, expected_config, max_iters):
    """{name: (model, blob, summary)} of the four UF members in ENSEMBLE_ORDER, through eval_u1.load_arm (member.pt re-hashed against member.json, what it carries against member.json); the problems of each against the plan (uf_common.member_record_problems)."""
    members, problems = {}, []
    for name in ENSEMBLE_ORDER:
        directory = Path(member_root)/name
        if not (directory/"member.json").is_file():
            problems.append(f"{name}: no member.json (member missing)")
            continue
        model, blob, record = eval_u1.load_arm(directory, device)
        train_record = json.loads((directory/"train_record.json").read_text(encoding="utf-8"))
        found = uc.member_record_problems(train_record, name, v2[name], expected_config=expected_config, max_iters=max_iters)
        if uc.sha256_file(directory/"train_record.json") != record.get("train_record_sha256"):         # review minor 6: this check does not lean on the earlier check_members
            found.append(f"{name}: train_record.json is not the file member.json was written from")
        last = directory/str(train_record.get("checkpoint_file"))
        if not last.is_file() or uc.sha256_file(last) != train_record.get("checkpoint_sha256"):
            found.append(f"{name}: {last.name} missing or its sha256 differs from train_record.json")
        if blob.get("checkpoint_choice") != "last" or blob.get("checkpoint_iteration") != max_iters or blob.get("checkpoint_sha256") != train_record.get("checkpoint_sha256"):
            found.append(f"{name}: member.pt is not the weights of the last step ({blob.get('checkpoint_choice')}, {blob.get('checkpoint_iteration')})")
        if blob.get("window_gaussian_sigma") is None or not blob.get("window"):
            found.append(f"{name}: member.pt has no sliding window with a Gaussian window")
        problems += found
        members[name] = (model, blob, dict(directory=str(directory), member_sha256=uc.sha256_file(directory/"member.pt"), fold=record["fold"], seed=record["seed"], temperature=blob["temperature"], checkpoint_choice=blob.get("checkpoint_choice"), checkpoint_iteration=blob.get("checkpoint_iteration"),
                                           checkpoint_sha256=blob.get("checkpoint_sha256"), window=list(blob["window"]) if blob.get("window") else None, window_gaussian_sigma=blob.get("window_gaussian_sigma"), recipe=train_record.get("recipe"), stopped=train_record.get("stopped"),
                                           iterations_run=train_record.get("iterations_run"), best_iteration=train_record.get("best_iteration"), best_calib_mean_dice35=train_record.get("best_calib_mean_dice35"), checkpoint_calib_mean_dice35=train_record.get("checkpoint_calib_mean_dice35"),
                                           wall_seconds=train_record.get("wall_seconds"), memory=train_record.get("memory"), label_contact_subjects_sha256=record["label_contact_subjects_sha256"], n_label_contact=len(record["label_contact_subjects"]), v2_namesake=uc.V2_NAME[name], problems=found))
    return members, problems


def image_problems(subjects, *, data_root, v1_pred, v2_pred, convert_record):
    """Before the lock: the sha256 of every image against what the stored predictions and the conversion were made from."""
    problems = []
    for subject in subjects:
        digest = uc.sha256_file(Path(data_root)/subject/eval_seg.IMAGE_FILE)
        for name, root in (("v1", v1_pred), ("v2", v2_pred)):
            provenance = json.loads((Path(root)/subject/"provenance.json").read_text(encoding="utf-8"))
            if provenance.get("input_sha256", {}).get(eval_seg.IMAGE_FILE) != digest:
                problems.append(f"{subject}: the {name} prediction was made from another image")
        if convert_record is not None and (convert_record.get("image_sha256") or {}).get(subject) != digest:
            problems.append(f"{subject}: the nnU-Net probabilities were made from another image")
    return problems


def nn_file_problems(subjects, nn_root, convert_record):
    """Before the lock: the gate of the conversion and the stored files against its hashes."""
    problems = uf_nn.convert_problems(convert_record, subjects)
    if problems:
        return problems
    for subject in subjects:
        path = Path(nn_root)/"probs"/subject/uf_nn.PROBS_FILE
        if not path.is_file() or uc.sha256_file(path) != convert_record["probs36_sha256"][subject]:
            problems.append(f"{path}: missing or not the file the gate passed")
    return problems


def evaluate_control(*, out_dir, v1_pred=uc.V1_PRED, v2_pred=uc.V2_PRED, u1_member=uc.U1_MEMBER, nn_manifest=uc.NN_OUT/"prepare_manifest.json", references=None, data_root=eval_seg.DATA_ROOT, allowlist_path=eval_seg.ALLOWLIST, folds_path=eval_seg.FOLDS, dev_source=eval_seg.DEV_SOURCE,
                     sealed_repo=None, synthetic=False, device="cuda", argv=None, log=print):
    """--control-only, the chain's first step (as UL's plan revision 1, R5): the three controls alone, v1_control, v2_control and U1_control, through the same functions as the evaluation, against the committed numbers, before 10 hours of training. These numbers were read long ago
    (v1, v2, U1): nothing of a UF member, no nnU-Net probability, no lock. The real path of the inputs, the provenance checks, the truth reading and U1's forward pass on the GPU are seen working; the members' and nnU-Net's paths are not. eval_control.json with `passed`."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists and is not empty; never overwritten")
    references = references or dict(ensembles=uc.REF_ENSEMBLES, members=REPO/uc.REF_MEMBERS_REL, u1=uc.REF_U1)
    code = code_record()
    if not synthetic:
        require_clean_code(code)
    allowlist, allowlist_record, refs = read_inputs(references=references, data_root=data_root, allowlist_path=allowlist_path, dev_source=dev_source, folds_path=folds_path, member_root=uc.OUT, nn_root=uf_nn.nn_dir(uc.OUT), nn_manifest=nn_manifest, v1_pred=v1_pred, v2_pred=v2_pred, u1_member=u1_member, synthetic=synthetic)
    u1_path, u1_sha, u1_record = eval_u1.member_files(u1_member)
    if not synthetic and (u1_sha != uc.U1_MEMBER_SHA256 or u1_sha != refs["u1"]["member_sha256"]):
        raise ValueError(f"{u1_path}: not U1's member")
    nn_contact, _ = eval_nn.label_contact(nn_manifest)
    sealed = load_sealed_lists(Path(sealed_repo) if sealed_repo is not None else source_root_for(code))
    folds = load_folds(folds_path)
    subjects = eval_u1.admit(allowlist, dev_source, sealed, folds.training_subjects, dict(U1=u1_record["label_contact_subjects"], nn=nn_contact))
    if subjects != refs["ensembles"]["subjects"] or subjects != refs["u1"]["subjects"]:
        raise ValueError("the subjects differ from the references'")
    problems = image_problems(subjects, data_root=data_root, v1_pred=v1_pred, v2_pred=v2_pred, convert_record=None)
    if problems:
        raise ValueError(f"not checked: {problems}")
    for arm, root in (("v1_control", v1_pred), ("v2_control", v2_pred)):
        eval_nn.check_control_files(root, subjects, SEGMENTERS[arm])
    device = torch.device(device)
    if device.type == "cuda":
        cap_gpu_memory(device, GPU_CAP_GIB)
    u1_model, u1_blob, _ = eval_u1.load_arm(u1_member, device)
    lookup = eval_seg.load_seg4_lookup()
    files = OasisFiles(Path(data_root))
    rows = {arm: [] for arm in CONTROL_ARMS}
    for subject in subjects:
        values, _ = read_image(files, subject)
        image, _ = normalize_image(values, percentile=u1_blob["percentile"])
        tops = dict(U1_control=eval_u1.timed_forward(u1_model, image, device=device, amp=u1_blob["amp"], window=tuple(u1_blob["window"]), sigma=u1_blob["window_gaussian_sigma"])[0])
        for arm in ("v1_control", "v2_control"):
            tops[arm], _ = eval_seg.read_prediction(v1_pred if arm == "v1_control" else v2_pred, subject, expected_segmenter=SEGMENTERS[arm])
        true35, _ = eval_seg.read_truth(data_root, subject, "seg35")
        true4, _ = eval_seg.read_truth(data_root, subject, "seg4")
        for arm in CONTROL_ARMS:
            rows[arm].append(eval_nn.metrics_of(eval_seg.summarize_subject(tops[arm], true35, true4, lookup)))
        log(f"{subject}: "+", ".join(f"{arm} {rows[arm][-1]['seg35']:.4f}" for arm in CONTROL_ARMS))
    reference = dict(v1_control=refs["ensembles"]["rows"]["v1"], v2_control=refs["ensembles"]["rows"]["v2"], U1_control=refs["u1"]["rows"])
    differences = {arm: {m: eval_nn.mean_of(rows[arm], m)-eval_nn.mean_of(reference[arm], m) for m in uc.CONTROL_METRICS} for arm in CONTROL_ARMS}
    document = dict(format=FORMAT+".control", created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=argv, code=code, synthetic=synthetic, subjects=subjects, differences=differences, tolerance=dict(uc.CONTROL_TOLERANCES), passed=all(abs(differences[arm][m]) <= uc.CONTROL_TOLERANCES[arm] for arm in CONTROL_ARMS for m in uc.CONTROL_METRICS),
                    means={arm: {m: eval_nn.mean_of(rows[arm], m) for m in ("seg35", "seg4")} for arm in CONTROL_ARMS}, allowlist=allowlist_record, device=str(device), note="只有 v1、v2 集成与 U1 三个控制，这些数早就读过，不增加任何 UF 的信息；不取读取锁。")
    out_dir.mkdir(parents=True, exist_ok=True)
    uc.write_json_atomic(out_dir/"eval_control.json", eval_seg.no_nan(document), allow_nan=False)
    return document


def evaluate(*, out_dir, member_root=uc.OUT, nn_root=None, nn_manifest=uc.NN_OUT/"prepare_manifest.json", v1_pred=uc.V1_PRED, v2_pred=uc.V2_PRED, u1_member=uc.U1_MEMBER, references=None, data_root=eval_seg.DATA_ROOT, allowlist_path=eval_seg.ALLOWLIST, folds_path=eval_seg.FOLDS,
             dev_source=eval_seg.DEV_SOURCE, sealed_repo=None, v2_root=uc.V2_ROOT, v2_registry=None, synthetic=False, train_overrides=None, device="cuda", argv=None, log=print, bz_go=None):
    if synthetic and bz_go is not None:
        raise ValueError("--bz-go is for the read lock of real data; synthetic runs do not touch it")
    if not synthetic:
        uf_lock.check(bz_go)                                    # the first thing on real data: no other option leads around it
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists and is not empty; the evaluation is never overwritten")
    nn_root = uf_nn.nn_dir(member_root) if nn_root is None else Path(nn_root)
    references = references or dict(ensembles=uc.REF_ENSEMBLES, members=REPO/uc.REF_MEMBERS_REL, u1=uc.REF_U1)
    code = code_record()
    if not synthetic:
        require_clean_code(code)
        if train_overrides is not None or v2_registry is not None or Path(v2_root).resolve() != uc.V2_ROOT.resolve():
            raise ValueError("real data: the training config, v2's members and their registry are the plan's")
    allowlist, allowlist_record, refs = read_inputs(references=references, data_root=data_root, allowlist_path=allowlist_path, dev_source=dev_source, folds_path=folds_path, member_root=member_root, nn_root=nn_root, nn_manifest=nn_manifest, v1_pred=v1_pred, v2_pred=v2_pred, u1_member=u1_member, synthetic=synthetic)
    expected_config = uc.planned_config(train_overrides)
    max_iters = expected_config["max_iters"]
    v2 = uc.v2_member_lists(v2_root, registry=Path(v2_registry) if v2_registry is not None else REPO/uc.V2_REGISTRY_REL)
    u1_path, u1_sha, u1_record = eval_u1.member_files(u1_member)
    if not synthetic and u1_sha != uc.U1_MEMBER_SHA256:
        raise ValueError(f"{u1_path}: sha256 {u1_sha} is not U1's")
    if not synthetic and u1_sha != refs["u1"]["member_sha256"]:
        raise ValueError("U1's member is not the one of eval_u1.json")
    member_contacts = {}
    for name in uc.MEMBER_NAMES:
        path = Path(member_root)/name/"member.json"
        if path.is_file():
            member_contacts[name] = json.loads(path.read_text(encoding="utf-8"))["label_contact_subjects"]
    nn_contact, nn_manifest_sha256 = eval_nn.label_contact(nn_manifest)
    sealed = load_sealed_lists(Path(sealed_repo) if sealed_repo is not None else source_root_for(code))
    folds = load_folds(folds_path)
    subjects = eval_u1.admit(allowlist, dev_source, sealed, folds.training_subjects, dict(member_contacts, U1=u1_record["label_contact_subjects"], nn=nn_contact))   # before any file of a subject
    if subjects != refs["ensembles"]["subjects"] or subjects != refs["u1"]["subjects"] or any(subjects != r["subjects"] for r in refs["members"].values()):
        raise ValueError("the subjects differ from the references'")
    device = torch.device(device)
    if device.type == "cuda":
        cap_gpu_memory(device, GPU_CAP_GIB)
    members, member_problems = load_members(member_root, device, v2=v2, expected_config=expected_config, max_iters=max_iters)
    convert_path = nn_root/"convert_record.json"
    convert_record = json.loads(convert_path.read_text(encoding="utf-8")) if convert_path.is_file() else {}
    before = member_problems+nn_file_problems(subjects, nn_root, convert_record)
    if not before:
        before += image_problems(subjects, data_root=data_root, v1_pred=v1_pred, v2_pred=v2_pred, convert_record=convert_record)
        for arm, root in (("v1_control", v1_pred), ("v2_control", v2_pred)):
            eval_nn.check_control_files(root, subjects, SEGMENTERS[arm])
    if before:
        raise ValueError(f"not evaluated, the truth is not read: {before}")
    u1_model, u1_blob, _ = eval_u1.load_arm(u1_member, device)
    percentiles = {blob["percentile"] for _, blob, _ in members.values()} | {u1_blob["percentile"]}
    if len(percentiles) != 1:
        raise ValueError(f"the members disagree on the normalization percentile: {percentiles}")
    percentile = percentiles.pop()
    lookup = eval_seg.load_seg4_lookup()
    files = OasisFiles(Path(data_root))
    rows, truth_sha, image_sha, seconds, extra, dev_read = {arm: [] for arm in ARMS}, {}, {}, {k: [] for k in ("UF4_forward", "U1_forward", "subject")}, {}, None
    for subject in subjects:
        started = time.perf_counter()
        values, meta = read_image(files, subject)
        image_sha[subject] = meta["sha256"]
        shape = tuple(meta["grid_shape_zyx"])
        image, _ = normalize_image(values, percentile=percentile)
        del values
        tops = {}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        clock = time.perf_counter()

        def member_logits():
            for name in ENSEMBLE_ORDER:
                model, blob, _ = members[name]
                z = predict_logits(model, image, device=device, amp=blob["amp"], window=tuple(blob["window"]), gaussian_sigma=blob["window_gaussian_sigma"])
                tops[f"UF_{uc.V2_NAME[name]}"] = z.argmax(1)[0].to(torch.uint8).cpu().numpy()
                yield z, blob["temperature"]
                del z

        p16 = ensemble_probabilities(member_logits()).cpu().numpy().astype(np.float16)     # as infer.segment_subject stores probs36
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds["UF4_forward"].append(time.perf_counter()-clock)
        tops["UF4"] = p16.argmax(axis=0).astype(np.uint8)           # = the argmax of the read-back (a common positive divisor keeps the order of distinct float16 values)
        p_nn, _ = read_back_probabilities(nn_root/"probs"/subject/uf_nn.PROBS_FILE, shape=(eval_seg.N_CLASSES,)+shape)
        tops["nn"] = p_nn.argmax(axis=0).astype(np.uint8)
        tops["UF4_nn"] = half_and_half(p16, p_nn)
        extra[subject] = dict(voxels_b_differs_from_a=int((tops["UF4_nn"] != tops["UF4"]).sum()))
        del p_nn, p16
        tops["U1_control"], u1_seconds = eval_u1.timed_forward(u1_model, image, device=device, amp=u1_blob["amp"], window=tuple(u1_blob["window"]), sigma=u1_blob["window_gaussian_sigma"])
        seconds["U1_forward"].append(u1_seconds)
        for arm in ("v1_control", "v2_control"):
            tops[arm], _ = eval_seg.read_prediction(v1_pred if arm == "v1_control" else v2_pred, subject, expected_segmenter=SEGMENTERS[arm])
        del image
        if not synthetic and dev_read is None:                  # the moment before the first truth label; a failure before it leaves no lock
            dev_read = uf_lock.take(dict(member_root=str(member_root), members={n: s["member_sha256"] for n, (_, _, s) in members.items()}, out_dir=str(out_dir), command=argv, code_commit=code.get("commit")), bz_go=bz_go)
        true35, sha35 = eval_seg.read_truth(data_root, subject, "seg35")
        true4, sha4 = eval_seg.read_truth(data_root, subject, "seg4")
        truth_sha[subject] = dict(seg35=sha35, seg4=sha4)
        for arm in ARMS:
            rows[arm].append(eval_nn.metrics_of(eval_seg.summarize_subject(tops[arm], true35, true4, lookup)))
        seconds["subject"].append(time.perf_counter()-started)
        log(f"{subject}: "+", ".join(f"{arm} {rows[arm][-1]['seg35']:.4f} / {rows[arm][-1]['seg4']:.4f}" for arm in ("UF4", "UF4_nn", "v1_control", "v2_control")))
    references_rows = dict(reference_v1=refs["ensembles"]["rows"]["v1"], reference_v2=refs["ensembles"]["rows"]["v2"], reference_U1=refs["u1"]["rows"], **{f"reference_v2_{k}": [dict(seg35=row["seg35"], seg4=row["seg4"], **row["groups"]) for row in r["rows"]] for k, r in refs["members"].items()})
    means = {arm: {m: eval_nn.mean_of(r, m) for m in (METRICS if not arm.startswith("reference_v2_") else MEMBER_METRICS)} for arm, r in dict(rows, **references_rows).items()}

    def column(source, metric):
        return [eval_nn.clean(row[metric]) for row in source]

    controls = {arm: {m: means[arm][m]-means[ref][m] for m in uc.CONTROL_METRICS} for arm, ref in (("v1_control", "reference_v1"), ("v2_control", "reference_v2"), ("U1_control", "reference_U1"))}
    paired = {}
    for arm in ENSEMBLE_ARMS+("nn",):
        for ref in ("reference_v1", "reference_v2"):
            paired[f"{arm}_minus_{ref}"] = {m: eval_seg.paired(column(rows[arm], m), column(references_rows[ref], m)) for m in METRICS}
    paired["UF4_nn_minus_UF4"] = {m: eval_seg.paired(column(rows["UF4_nn"], m), column(rows["UF4"], m)) for m in METRICS}
    for arm in MEMBER_ARMS:
        name = arm.removeprefix("UF_")
        paired[f"{arm}_minus_reference_v2_{name}"] = {m: eval_seg.paired(column(rows[arm], m), column(references_rows[f"reference_v2_{name}"], m)) for m in MEMBER_METRICS}
    paired["UF_A0_minus_reference_U1"] = {m: eval_seg.paired(column(rows["UF_A0"], m), column(references_rows["reference_U1"], m)) for m in METRICS}
    per_label = {arm: [eval_seg.nanmean([eval_nn.clean(row["per_label"][k]) for row in rows[arm]]) for k in range(eval_seg.N_CLASSES-1)] for arm in ARMS}
    document = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=argv, code=code, synthetic=synthetic, dev_source=dev_source, subjects=subjects, arms=list(ARMS), ensemble_order=list(ENSEMBLE_ORDER), nn_weight=uc.NN_WEIGHT,
                    means=means, controls=dict(differences=controls, tolerance=dict(uc.CONTROL_TOLERANCES)), paired=paired, per_label_mean=per_label, rows={arm: {s: r[i] for i, s in enumerate(subjects)} for arm, r in rows.items()},
                    reference_rows={arm: {s: r[i] for i, s in enumerate(subjects)} for arm, r in references_rows.items()}, references={k: dict(path=v["path"], sha256=v["sha256"]) for k, v in (("ensembles", refs["ensembles"]), ("u1", refs["u1"]))} | dict(members=dict(path=next(iter(refs["members"].values()))["path"], sha256=next(iter(refs["members"].values()))["sha256"])),
                    members={n: s for n, (_, _, s) in members.items()}, member_problems=member_problems, expected_config=expected_config, max_iters=max_iters, v2_members={n: {k: v[k] for k in ("fold", "seed", "member_sha256", "train_record_sha256")} for n, v in v2.items()},
                    U1=dict(directory=str(u1_member), member_sha256=u1_sha, window=list(u1_blob["window"]), window_gaussian_sigma=u1_blob.get("window_gaussian_sigma")), nn=dict(root=str(nn_root), convert_record=convert_record, manifest_sha256=nn_manifest_sha256, n_label_contact=len(nn_contact)),
                    per_subject_extra=extra, seconds={k: eval_u1.seconds_summary(v) for k, v in seconds.items() if v}, allowlist=allowlist_record, allowlist_sha256=allowlist.sha256, folds_sha256=folds.sha256, sealed_lists=sealed.record, seg4_map_sha256=eval_seg.sha_file(eval_seg.SEG4_MAP),
                    groups={k: list(v) for k, v in eval_seg.GROUPS.items()}, image_sha256=image_sha, truth_sha256=truth_sha, dev_read=dev_read, device=str(device), gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/GIB if device.type == "cuda" else None,
                    note="开发集，单次观测；分割级读数，不是配准结果，不是留出集结果。真值只在本评测读取。")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir/"per_subject.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "arm", *METRICS])
        for arm, source in dict(rows, **references_rows).items():
            for index, subject in enumerate(subjects):
                writer.writerow([subject, arm, *[eval_nn.clean(source[index].get(m)) for m in METRICS]])
    with (out_dir/"per_label.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "group", *[f"mean_{arm}" for arm in ARMS]])
        for index, label in enumerate(range(1, eval_seg.N_CLASSES)):
            group = next(name for name, labels in eval_seg.GROUPS.items() if label in labels)
            writer.writerow([label, group, *[per_label[arm][index] for arm in ARMS]])
    uc.write_json_atomic(out_dir/"eval_uf.json", eval_seg.no_nan(document), allow_nan=False)      # last: the file the reading needs exists only whole and only after everything else
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--member-root", type=Path, default=uc.OUT)
    parser.add_argument("--nn-root", type=Path, default=None, help="default: <member-root>/nn")
    parser.add_argument("--nn-manifest", type=Path, default=uc.NN_OUT/"prepare_manifest.json")
    parser.add_argument("--v1-pred", type=Path, default=uc.V1_PRED)
    parser.add_argument("--v2-pred", type=Path, default=uc.V2_PRED)
    parser.add_argument("--u1-member", type=Path, default=uc.U1_MEMBER)
    parser.add_argument("--reference-ensembles", type=Path, default=uc.REF_ENSEMBLES)
    parser.add_argument("--reference-members", type=Path, default=REPO/uc.REF_MEMBERS_REL)
    parser.add_argument("--reference-u1", type=Path, default=uc.REF_U1)
    parser.add_argument("--v2-root", type=Path, default=uc.V2_ROOT)
    parser.add_argument("--v2-registry", type=Path, default=None)
    parser.add_argument("--allowlist", type=Path, default=eval_seg.ALLOWLIST)
    parser.add_argument("--folds", type=Path, default=eval_seg.FOLDS)
    parser.add_argument("--dev-source", default=eval_seg.DEV_SOURCE)
    parser.add_argument("--data-root", type=Path, default=eval_seg.DATA_ROOT)
    parser.add_argument("--sealed-repo", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--train-overrides", type=Path, default=None, help="--synthetic only: a JSON file of the tiny training config the synthetic members were trained with")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bz-go", default=None, help="BZ's words and the time: approves another read of the development truth for UF (recorded in the lock file and in eval_uf.json)")
    parser.add_argument("--control-only", action="store_true", help="the three controls alone, before the training; no lock, no UF member")
    args = parser.parse_args(argv)
    if args.train_overrides is not None and not args.synthetic:
        parser.error("--train-overrides goes with --synthetic")
    overrides = json.loads(args.train_overrides.read_text(encoding="utf-8")) if args.train_overrides else None
    if args.control_only:
        if args.bz_go is not None:
            parser.error("--bz-go does not go with --control-only: the control takes no lock")
        document = evaluate_control(out_dir=args.out_dir, v1_pred=args.v1_pred, v2_pred=args.v2_pred, u1_member=args.u1_member, nn_manifest=args.nn_manifest, references=dict(ensembles=args.reference_ensembles, members=args.reference_members, u1=args.reference_u1), data_root=args.data_root,
                                    allowlist_path=args.allowlist, folds_path=args.folds, dev_source=args.dev_source, sealed_repo=args.sealed_repo, synthetic=args.synthetic, device=args.device, argv=sys.argv if argv is None else ["eval_uf.py", *argv])
        print(json.dumps({k: document[k] for k in ("passed", "differences", "tolerance")}, indent=1))
        return 0 if document["passed"] else 8
    try:
        document = evaluate(out_dir=args.out_dir, member_root=args.member_root, nn_root=args.nn_root, nn_manifest=args.nn_manifest, v1_pred=args.v1_pred, v2_pred=args.v2_pred, u1_member=args.u1_member, references=dict(ensembles=args.reference_ensembles, members=args.reference_members, u1=args.reference_u1),
                            data_root=args.data_root, allowlist_path=args.allowlist, folds_path=args.folds, dev_source=args.dev_source, sealed_repo=args.sealed_repo, v2_root=args.v2_root, v2_registry=args.v2_registry, synthetic=args.synthetic, train_overrides=overrides, device=args.device,
                            argv=sys.argv if argv is None else ["eval_uf.py", *argv], bz_go=args.bz_go)
    except uf_lock.DevReadLocked as exc:
        print(f"STOP: {exc}", flush=True)
        return 8
    print(json.dumps(dict(means={a: {m: document["means"][a][m] for m in ("seg35", "seg4")} for a in ENSEMBLE_ARMS+CONTROL_ARMS}, controls=document["controls"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
