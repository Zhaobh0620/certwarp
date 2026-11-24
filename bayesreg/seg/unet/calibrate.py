"""Temperature calibration of one member (04_分割器.md §5).

A single temperature T > 0 per member, fitted on the brain voxels (aligned_norm > 0) of the member's
fold calib subjects by minimizing the mean negative log-likelihood of softmax(z/T). NLL(beta) with
beta = 1/T is convex (a log-sum-exp minus a linear term), so a bounded 1-D search on log T finds the
global minimum. Reported before/after: NLL and the top-label expected calibration error (ECE, 15
equal-width confidence bins).

Output <member dir>/member.pt = weights + model config + T + calibration record: the file whose
sha256 is the registered member_model_sha256 (so T is bound to the registered hash), plus
calibration.json and member.json (the record registry.py reads). Never overwritten.

The weights are model_best.pt, unless the train record names another checkpoint (large.py, checkpoint "last",
work package UF): then checkpoint_file (model_last.pt only), re-hashed against checkpoint_sha256, and member.pt and
calibration.json also carry checkpoint_choice, checkpoint_file, checkpoint_sha256, checkpoint_iteration. A record
without these keys (v1, v2, U1) is calibrated exactly as before.
"""
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
import torch
import torch.nn.functional as F

from jd_curl.validation import InputValidationError

from ...io.pair import subject_list_sha256
from .data import FOLD_INDEX, load_training_subjects
from .model import build_model
from .train import (atomic_torch_save, confusion_dice, environment_record, file_sha256, member_dir, nanmean_rows,
                    predict_logits, write_json)

MEMBER_FORMAT = "bayesreg.unet35.member.v1"
T_BOUNDS = (.05, 20.)
NAMED_CHECKPOINTS = ("model_last.pt",)          # the checkpoint_file a train record may name (large.LAST_CHECKPOINT)
CHECKPOINT_KEYS = ("checkpoint_choice", "checkpoint_file", "checkpoint_sha256", "checkpoint_iteration")


def names_a_checkpoint(train_record):
    """True when the record is of a member of the last step: any of CHECKPOINT_KEYS, or config.checkpoint == "last" (UF review minor 4: a record that lost its keys does not fall back to model_best.pt in silence)."""
    return any(k in train_record for k in CHECKPOINT_KEYS) or (train_record.get("config") or {}).get("checkpoint") == "last"


def selected_checkpoint(train_record):
    """(file name, sha256 it must have, the checkpoint keys to carry or None): model_best.pt and model_best_sha256 unless the record names a checkpoint; then all four keys are required."""
    if not names_a_checkpoint(train_record):
        return "model_best.pt", train_record["model_best_sha256"], None
    keys = {k: train_record.get(k) for k in CHECKPOINT_KEYS}
    if keys["checkpoint_file"] not in NAMED_CHECKPOINTS or keys["checkpoint_choice"] != "last" or not isinstance(keys["checkpoint_sha256"], str) or not isinstance(keys["checkpoint_iteration"], int):
        raise InputValidationError(f"train_record names the checkpoint {keys}: only checkpoint_choice 'last' with {NAMED_CHECKPOINTS}, its sha256 and iteration")
    return keys["checkpoint_file"], keys["checkpoint_sha256"], keys


def _chunks(n, size):
    for start in range(0, n, size):
        yield slice(start, min(n, start+size))


def nll(logits, labels, temperature, *, chunk=1 << 20):
    """Mean NLL of softmax(logits/T), accumulated in float64."""
    total = 0.
    for sl in _chunks(labels.shape[0], chunk):
        z = logits[sl].double()/temperature
        total += float(F.cross_entropy(z, labels[sl], reduction="sum"))
    return total/labels.shape[0]


def ece_bins(logits, labels, temperature, *, n_bins=15, chunk=1 << 20):
    """(counts, confidence sums, accuracy sums) per equal-width top-label confidence bin on (0, 1]."""
    counts, conf_sum, acc_sum = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for sl in _chunks(labels.shape[0], chunk):
        p = torch.softmax(logits[sl].double()/temperature, dim=1)
        conf, pred = p.max(dim=1)
        correct = (pred == labels[sl]).double()
        bins = torch.clamp((conf*n_bins).ceil().long()-1, 0, n_bins-1)
        counts += torch.bincount(bins, minlength=n_bins).double().cpu().numpy()
        conf_sum += torch.bincount(bins, weights=conf, minlength=n_bins).cpu().numpy()
        acc_sum += torch.bincount(bins, weights=correct, minlength=n_bins).cpu().numpy()
    return counts, conf_sum, acc_sum


def ece_from_bins(counts, conf_sum, acc_sum):
    mask = counts > 0
    return float(np.sum(np.abs(acc_sum[mask]-conf_sum[mask]))/counts.sum())


def ece(logits, labels, temperature, *, n_bins=15, chunk=1 << 20):
    """Top-label expected calibration error with equal-width bins on (0, 1]."""
    return ece_from_bins(*ece_bins(logits, labels, temperature, n_bins=n_bins, chunk=chunk))


def fit_temperature(logits, labels, *, bounds=T_BOUNDS, xatol=1e-4):
    """argmin_T NLL(T) on [bounds]; returns (T, record)."""
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0] or labels.shape[0] == 0:
        raise InputValidationError("fit_temperature: logits (N, C) and labels (N,) with N > 0 required")
    result = minimize_scalar(lambda log_t: nll(logits, labels, math.exp(log_t)), method="bounded",
                             bounds=(math.log(bounds[0]), math.log(bounds[1])), options=dict(xatol=xatol))
    temperature = float(math.exp(result.x))
    at_bound = min(abs(result.x-math.log(b)) for b in bounds) < 10*xatol
    record = dict(temperature=temperature, nll_before=nll(logits, labels, 1.), nll_after=float(result.fun),
                  ece_before=ece(logits, labels, 1.), ece_after=ece(logits, labels, temperature),
                  n_voxels=int(labels.shape[0]), bounds=list(bounds), at_bound=bool(at_bound),
                  optimizer=dict(method="scipy minimize_scalar bounded on log T", xatol=xatol, nfev=int(result.nfev),
                                 success=bool(result.success)))
    return temperature, record


def load_checkpoint(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=True)
    model = build_model(blob["model_config"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def collect_brain_logits(model, subjects, *, device, amp=True, window=None, max_voxels=None, seed=0, gaussian_sigma=None):
    """Logits (N, C) float32 and labels (N,) on the brain voxels of every subject (optionally subsampled)."""
    rng = np.random.default_rng(seed)
    zs, ys, dice = [], [], []
    for item in subjects:
        logits = predict_logits(model, item.image, device=device, amp=amp, window=window, gaussian_sigma=gaussian_sigma)[0]
        true = torch.as_tensor(item.labels.astype(np.int64), device=logits.device)
        dice.append(confusion_dice(logits.argmax(0), true, logits.shape[0]))
        brain = torch.as_tensor(item.image > 0, device=logits.device)
        zs.append(logits[:, brain].T.contiguous())
        ys.append(true[brain])
    z, y = torch.cat(zs), torch.cat(ys)
    if max_voxels is not None and y.shape[0] > max_voxels:
        keep = torch.as_tensor(np.sort(rng.choice(y.shape[0], max_voxels, replace=False)), device=y.device)
        z, y = z[keep], y[keep]
    return z, y, dice


def calibrate_member(*, files, guard, fold, seed, models_root, device="cuda", max_voxels=None, log=print,
                     command=None, subjects=None, inputs_record=None):
    """Fit T for member (fold, seed) on its fold's calib subjects and write member.pt / member.json.

    calibration.json records the label guard with the reads made here (`label_guard.label_reads`: the
    fold's calib subjects, once each); with `subjects` given (tests) it is null and registry.register
    refuses the member."""
    out = member_dir(models_root, fold, seed)
    record_path = out/"train_record.json"
    for path in (out/"member.pt", out/"member.json", out/"calibration.json"):
        if path.exists():
            raise FileExistsError(f"{path} exists; calibration is never redone in place")
    train_record = json.loads(record_path.read_text(encoding="utf-8"))
    if train_record["fold"] != fold or train_record["seed"] != seed:
        raise InputValidationError(f"{record_path}: fold/seed differ from the request")
    ckpt_name, ckpt_sha256, named = selected_checkpoint(train_record)
    ckpt_path = out/ckpt_name
    if file_sha256(ckpt_path) != ckpt_sha256:
        raise InputValidationError(f"{ckpt_path}: sha256 differs from train_record.{'model_best_sha256' if named is None else 'checkpoint_sha256'}")
    wall = time.perf_counter()
    device = torch.device(device)
    model, blob = load_checkpoint(ckpt_path, device)
    if named is not None and (blob.get("iteration") != named["checkpoint_iteration"] or blob.get("checkpoint") != "last"):
        raise InputValidationError(f"{ckpt_path}: iteration {blob.get('iteration')}, checkpoint {blob.get('checkpoint')!r}: not the record's last step {named['checkpoint_iteration']}")
    cfg = train_record["config"]
    label_guard = None
    if subjects is None:
        first_read = len(guard.reads)
        calib = load_training_subjects(files, guard, fold, "calib", percentile=cfg["percentile"], log=log)
        label_guard = dict(guard.record(), label_reads=list(guard.reads[first_read:]))
    else:
        calib = subjects
    if sorted(s.subject for s in calib) != sorted(train_record["calib_subjects"]):
        raise InputValidationError("calibration subjects differ from the member's calib subjects")
    window = tuple(cfg["patch_size"]) if cfg.get("patch_size") else None
    sigma = cfg.get("window_gaussian_sigma")                    # only in the configs of v3_large (large.py)
    z, y, dice = collect_brain_logits(model, calib, device=device, amp=cfg["amp"], window=window,
                                      max_voxels=max_voxels, seed=seed, gaussian_sigma=sigma)
    temperature, fit = fit_temperature(z, y)
    per_label = nanmean_rows(dice)
    calibration = dict(format="bayesreg.unet35.calibration.v1", fold=fold, seed=seed, temperature=temperature,
                       calib_subjects=[s.subject for s in calib], voxel_set="aligned_norm > 0 (brain)",
                       max_voxels=max_voxels, fit=fit,
                       calib_mean_dice35=float(np.mean([np.nanmean(d) for d in dice])),
                       calib_per_label_dice=per_label.tolist(), label_guard=label_guard, inputs=inputs_record,
                       environment=environment_record(device), command=command, wall_seconds=time.perf_counter()-wall,
                       **({} if named is None else dict(checkpoint=named)))
    contact = train_record["label_contact_subjects"]
    if subject_list_sha256(contact) != train_record["label_contact_subjects_sha256"]:
        raise InputValidationError("train_record: label_contact_subjects do not match their sha256")
    atomic_torch_save(dict(format=MEMBER_FORMAT, state_dict=blob["state_dict"], model_config=blob["model_config"],
                           temperature=temperature, fold=fold, seed=seed, best_iteration=blob["iteration"],
                           label_contact_subjects=list(contact),
                           label_contact_subjects_sha256=train_record["label_contact_subjects_sha256"],
                           model_best_sha256=train_record["model_best_sha256"], percentile=cfg["percentile"],
                           window=list(window) if window else None, amp=bool(cfg["amp"]),
                           **({} if sigma is None else dict(window_gaussian_sigma=float(sigma))),
                           **({} if named is None else named)), out/"member.pt")
    write_json(out/"calibration.json", calibration)
    member = dict(format="bayesreg.unet35.member_record.v1", member_file="member.pt",
                  member_sha256=file_sha256(out/"member.pt"), fold=fold, fold_index=FOLD_INDEX[fold], seed=seed,
                  temperature=temperature, label_contact_subjects=list(contact),
                  label_contact_subjects_sha256=train_record["label_contact_subjects_sha256"],
                  train_record_sha256=file_sha256(record_path), calibration_sha256=file_sha256(out/"calibration.json"),
                  folds_sha256=(train_record.get("label_guard") or {}).get("folds_sha256"))
    write_json(out/"member.json", member)
    log(f"fold {fold} seed {seed}: T = {temperature:.4f}, NLL {fit['nll_before']:.4f} -> {fit['nll_after']:.4f}, "
        f"ECE {fit['ece_before']:.4f} -> {fit['ece_after']:.4f} on {fit['n_voxels']} brain voxels")
    return member, calibration


__all__ = ["CHECKPOINT_KEYS", "MEMBER_FORMAT", "NAMED_CHECKPOINTS", "T_BOUNDS", "calibrate_member", "collect_brain_logits", "ece",
           "ece_bins", "ece_from_bins", "fit_temperature", "load_checkpoint", "names_a_checkpoint", "nll", "selected_checkpoint"]
