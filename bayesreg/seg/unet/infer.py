"""Inference of the structure layer P_S (04_分割器.md §6) and the per-image provenance (contract C03.1-2/3).

For a subject s, only fold models whose registered label contact does not contain s are used:
a B-line training subject of fold A is segmented by the fold-B model only (and vice versa); every other
admitted subject (development, T0, and -- once BZ unseals them -- hold-out subjects) by all fold models
(contract D24: two-fold average). Within the chosen fold models every seed member is used; the
ensemble probability is the plain mean over members of softmax(z_m / T_m).

Order of checks, all before the image is read: canonical id, development allowlist and the sealed lists
(check_subject_admitted = bayesreg.io.guard.check_subject, C23.4; SubjectSealed otherwise), the grouping's status (groups.INFERENCE_STATUSES; an unverified grouping
such as tissue4_ref is refused unless allow_unverified_grouping, GroupingUnverified), registry models
of the segmenter, label-contact exclusion of s, every contact subject on the allowlist and none
never_for_training (as validate_segmentation_provenance requires), member files found by sha256 and
re-hashed before torch.load. After loading, the fold, label contact and its sha256 stored inside
member.pt must equal the registry entry that listed it (ModelSelectionRefused otherwise).

Outputs in <pred_root>/<subject>/ (never overwritten):
  probs36.npz          key "p": float16 (36, nz, ny, nx), class c = seg35 label c
  structure_groups.npz key "p": float16 (K_S + 1, nz, ny, nx), channel 0 = background, 1..K_S = groups
  provenance.json      the P_S provenance (layer "structure"; accepted by
                       bayesreg.io.validate_segmentation_provenance)
Arrays are in jd_curl.three_d.nifti.load_scalar storage (zyx, RAS), i.e. the grid of pair.npz.
Read them back with `read_probabilities`, which converts to float64, checks the raw channel sum on every voxel
(C03.2-4a, D39; bayesreg.seg.channels.read_back_probabilities) and renormalizes.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from jd_curl.validation import InputValidationError

from ... import __version__ as bayesreg_version
from ...io.guard import DevAllowlist, SubjectSealed, check_subject
from ...io.pair import SegmenterRegistry, verify_segmentation_files
from ..channels import read_back_probabilities
from .calibrate import MEMBER_FORMAT
from .data import FOLD_INDEX, IMAGE_FILE, normalize_image, read_image
from .groups import group_probabilities, require_inference_grouping
from .model import build_model
from .registry import SEGMENTER_ID, find_member_files
from .train import environment_record, file_sha256, predict_logits, write_json

PROBS_FILE, GROUPS_FILE, PROVENANCE_FILE = "probs36.npz", "structure_groups.npz", "provenance.json"
# provenance outputs.read_back (review PS1_PS2 item e): the raw-sum check comes before the division
READ_BACK = ("bayesreg.seg.channels.read_back_probabilities (also through bayesreg.seg.unet.infer.read_probabilities): "
             "the raw channel sum is checked on every voxel BEFORE the division (|sum_k p_k - 1| <= 1e-3 for this "
             "float16 storage, else refused; C03.2-4a, D39), then float64 divided by the channel sum")


class ModelSelectionRefused(Exception):
    """No admissible fold model for this subject; status `provenance_rejected`."""
    status = "provenance_rejected"


def select_fold_models(registry, subject, allowlist, *, segmenter_id=SEGMENTER_ID):
    """Registered fold models of `segmenter_id` admissible for segmenting `subject`, and the policy name."""
    if not isinstance(registry, SegmenterRegistry):
        raise InputValidationError("registry: a SegmenterRegistry from load_segmenter_registry is required")
    if not isinstance(allowlist, DevAllowlist):
        raise InputValidationError("allowlist: a DevAllowlist from load_dev_allowlist is required")
    models = {m: e for m, e in registry.models.items() if e["segmenter_id"] == segmenter_id}
    if not models:
        raise ModelSelectionRefused(f"no registered fold model of {segmenter_id!r}")
    for model_id, entry in models.items():
        contact = set(entry["label_contact_subjects"])
        sealed = sorted(s for s in contact if not allowlist.allows(s))
        if sealed:
            raise ModelSelectionRefused(f"{model_id} had label contact with subjects off the allowlist {sealed}")
        never = sorted(contact & allowlist.never_for_training)
        if never:
            raise ModelSelectionRefused(f"{model_id} had label contact with never_for_training subjects {never}")
    usable = sorted(m for m, e in models.items() if subject not in e["label_contact_subjects"])
    if not usable:
        raise ModelSelectionRefused(f"every fold model of {segmenter_id!r} saw the labels of {subject}")
    policy = "all_folds_mean" if len(usable) == len(models) else "cross_fit_other_folds"
    return {m: models[m] for m in usable}, policy


def check_subject_admitted(subject, allowlist):
    """The inference guard: bayesreg.io.guard.check_subject (C23.4-2, C23.4-5), the rule check_pair_subjects applies.
    Steps 1 and 2 refuse with the words they always had; a sealed subject passes only through an unseal object on the
    allowlist (bayesreg.io.unseal.bind_unseal).  Returns None; SubjectSealed otherwise.  Admission only: nothing of the
    inference's arithmetic is here (C23.8-1)."""
    verdict = check_subject(subject, allowlist)
    if verdict.step == "canonical":
        raise SubjectSealed(verdict.message)
    if verdict.step == "allowlist":
        raise SubjectSealed(f"{subject!r} is not on the development allowlist; not segmented (BZ unseals)")
    if verdict.step == "sealed":
        raise SubjectSealed(f"not segmented: {verdict.message}")


def load_member(path, expected_sha256, device, *, entry):
    """member.pt after re-hashing it against the registered sha256 and comparing what it carries (fold,
    label contact and its sha256) with the registry `entry` that lists it; ModelSelectionRefused otherwise."""
    if expected_sha256 not in tuple(entry["member_model_sha256"]):
        raise ModelSelectionRefused(f"{path}: {expected_sha256[:12]}... is not a member of the registry entry")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ModelSelectionRefused(f"{path}: sha256 differs from the registered member")
    blob = torch.load(io.BytesIO(raw), map_location=device, weights_only=True)
    if blob.get("format") != MEMBER_FORMAT:
        raise ModelSelectionRefused(f"{path}: not a {MEMBER_FORMAT} file")
    fold = FOLD_INDEX.get(blob.get("fold"))
    if fold is None or fold != entry["fold"]:
        raise ModelSelectionRefused(f"{path}: member fold {blob.get('fold')!r} != registered fold {entry['fold']}")
    if list(blob.get("label_contact_subjects") or []) != list(entry["label_contact_subjects"]) \
            or blob.get("label_contact_subjects_sha256") != entry["label_contact_subjects_sha256"]:
        raise ModelSelectionRefused(f"{path}: label contact inside member.pt differs from the registry entry")
    model = build_model(blob["model_config"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def ensemble_probabilities(members):
    """Mean over members of softmax(z_m / T_m) along the class axis, renormalized.

    `members`: iterable of (logits (1, C, ...) tensor, T_m > 0), consumed one at a time (streaming, so
    only one member's logits are alive). Returns a float32 tensor (C, ...).
    """
    total, n = None, 0
    for z, t in members:
        if not float(t) > 0:
            raise InputValidationError("temperatures must be positive")
        p = torch.softmax(z.float()/float(t), dim=1)[0]
        total = p if total is None else total.add_(p)
        n += 1
    if n == 0:
        raise InputValidationError("ensemble_probabilities: no member")
    total /= n
    return total/total.sum(dim=0, keepdim=True)


def _save_npz(path, array):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, p=array)
    tmp = Path(str(path)+".tmp")
    tmp.write_bytes(buffer.getvalue())
    tmp.replace(path)


def segment_subject(subject, *, files, registry, allowlist, models_root, grouping, pred_root, device="cuda",
                    segmenter_id=SEGMENTER_ID, command=None, log=print, allow_unverified_grouping=False):
    check_subject_admitted(subject, allowlist)
    grouping_check = require_inference_grouping(grouping, allow_unverified=allow_unverified_grouping)
    out = Path(pred_root)/subject
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; predictions are never overwritten")
    models, policy = select_fold_models(registry, subject, allowlist, segmenter_id=segmenter_id)
    available = find_member_files(models_root)
    plan = []
    for model_id, entry in models.items():
        for sha in entry["member_model_sha256"]:
            if sha not in available:
                raise ModelSelectionRefused(f"member {sha[:12]}... of {model_id} not found under {models_root}")
            plan.append((model_id, sha, available[sha]))
    wall = time.perf_counter()
    device = torch.device(device)
    values, meta = read_image(files, subject)
    members, state = [], {}

    def member_logits():
        for model_id, sha, path in plan:
            model, blob = load_member(path, sha, device, entry=models[model_id])
            if "norm" not in state:
                state["image"], state["norm"] = normalize_image(values, percentile=blob["percentile"])
            elif blob["percentile"] != state["norm"]["percentile"]:
                raise InputValidationError(f"{path}: members disagree on the normalization percentile")
            window = tuple(blob["window"]) if blob.get("window") else None
            z = predict_logits(model, state["image"], device=device, amp=blob["amp"], window=window,
                               gaussian_sigma=blob.get("window_gaussian_sigma"))
            members.append(dict(model_id=model_id, fold=blob["fold"], seed=blob["seed"], member_sha256=sha,
                                temperature=blob["temperature"], best_iteration=blob["best_iteration"]))
            del model
            yield z, blob["temperature"]

    probs = ensemble_probabilities(member_logits()).cpu().numpy()
    norm = state["norm"]
    groups = group_probabilities(probs, grouping, dtype=np.float32)
    out.mkdir(parents=True, exist_ok=True)
    _save_npz(out/PROBS_FILE, probs.astype(np.float16))
    _save_npz(out/GROUPS_FILE, groups.astype(np.float16))
    model_ids = sorted(models)
    provenance = dict(
        layer="structure", segmenter_id=segmenter_id, model_ids=model_ids,
        model_sha256={m: list(models[m]["member_model_sha256"]) for m in model_ids},
        folds={m: models[m]["fold"] for m in model_ids}, trained_with_labels=True, test_labels_seen=False,
        subject=subject, policy=policy, members=members, **grouping.record(), grouping_check=grouping_check,
        input_sha256={IMAGE_FILE: meta["sha256"]},
        output_sha256={PROBS_FILE: file_sha256(out/PROBS_FILE), GROUPS_FILE: file_sha256(out/GROUPS_FILE)},
        outputs=dict(probs36=dict(file=PROBS_FILE, key="p", dtype="float16", shape=list(probs.shape),
                                  classes="seg35 label c at channel c (0 = background)"),
                     groups=dict(file=GROUPS_FILE, key="p", dtype="float16", shape=list(groups.shape),
                                 channels="0 = background, k = class_names[k-1]"),
                     layout="(channels, nz, ny, nx), jd_curl.three_d.nifti.load_scalar storage (RAS, zyx)",
                     read_back=READ_BACK),
        grid=dict(shape_zyx=meta["grid_shape_zyx"], spacing_xyz=meta["spacing_xyz"], origin_xyz=meta["origin_xyz"]),
        normalization=norm, label_contact_check=dict(subject_in_any_used_model=False, image_level=True,
                                                     note="pair-level check runs in validate_segmentation_provenance"),
        registry_path=registry.path, registry_sha256=registry.sha256, allowlist_path=allowlist.path,
        allowlist_sha256=allowlist.sha256,
        version=dict(bayesreg=bayesreg_version, segmenter="unet35_v1" if segmenter_id == SEGMENTER_ID else segmenter_id),
        command=command if command is not None else sys.argv, environment=environment_record(device),
        runtime_seconds=time.perf_counter()-wall)
    write_json(out/PROVENANCE_FILE, provenance)
    log(f"{subject}: {policy} with {model_ids} ({len(members)} members), {provenance['runtime_seconds']:.1f} s")
    return provenance


def read_probabilities(directory, which="groups"):
    """(array float64 renormalized, provenance) after re-hashing the output files against provenance. The read-back
    is bayesreg.seg.channels.read_back_probabilities, shared by every structure-layer read-back path (C03.2-4a, D39):
    before the division by the channel sum, the raw sum on every voxel within 1e-3 of 1 for float16 storage, 1e-6 for
    float32 / float64, other storage types refused (InputValidationError)."""
    directory = Path(directory)
    provenance = json.loads((directory/PROVENANCE_FILE).read_text(encoding="utf-8"))
    verify_segmentation_files(provenance, directory)
    name = {"groups": GROUPS_FILE, "probs36": PROBS_FILE}[which]
    array, _ = read_back_probabilities(directory/name)
    return array, provenance


__all__ = ["GROUPS_FILE", "ModelSelectionRefused", "PROBS_FILE", "PROVENANCE_FILE", "check_subject_admitted",
           "ensemble_probabilities", "load_member", "read_probabilities", "segment_subject", "select_fold_models"]
