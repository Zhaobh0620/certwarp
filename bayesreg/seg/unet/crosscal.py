"""Cross-fitted check of the ensemble calibration (review 2026-10-30, item 8; 04_分割器.md §5).

Each member is temperature-scaled alone on its own fold's calib subjects, which also chose its
early-stopping iteration; so the members' calib NLL is optimistic and the calibration of the
*ensemble* (mean of the tempered softmaxes, infer.ensemble_probabilities) was never checked. Here the
fold-B ensemble predicts the fold-A calib subjects and vice versa -- subjects whose labels the
predicting models never saw. Reported per target fold, on the brain voxels (aligned_norm > 0):
  - NLL and top-label ECE (15 equal-width bins) of the ensemble, and of each member alone with its T;
  - for information only, the ensemble-level temperature T_ens minimizing the NLL of
    p_ens^(1/T) / sum (temperature scaling applied to log p_ens) on these voxels.
Nothing is changed: no ensemble temperature is applied (default). Adopting one is a separate decision;
fitting it on the other fold's calib subjects would add them to the label contact of the models it is
applied to, which breaks the cross-fitting -- it would have to be fitted per fold on that fold's own
calib subjects.

Labels read: the calib subjects of both folds (B-line training subjects; calibration use), through
the label guard, recorded in the output. Members are checked first (registry.verify_members) and
loaded with infer.load_member against an entry built from the verified members.
Output <models_root>/calibration_crossfit.json (never overwritten).
"""
from pathlib import Path
import time

import numpy as np
import torch

from jd_curl.validation import InputValidationError

from .calibrate import ece_bins, ece_from_bins, fit_temperature, nll
from .data import FOLD_INDEX, FOLD_NAMES, load_training_subjects
from .infer import load_member
from .registry import model_id_for, verify_members
from .train import environment_record, predict_logits, write_json

CROSSFIT_FILE = "calibration_crossfit.json"
CROSSFIT_FORMAT = "bayesreg.unet35.calibration_crossfit.v1"
LOG_FLOOR = 1e-300


class _Accumulator:
    """NLL and ECE bins over voxels, for log-probabilities or logits/T (both used at T = 1)."""

    def __init__(self, n_bins=15):
        self.n, self.nll_sum = 0, 0.
        self.bins = [np.zeros(n_bins) for _ in range(3)]
        self.n_bins = n_bins

    def add(self, logits, labels):
        count = int(labels.shape[0])
        self.nll_sum += nll(logits, labels, 1.)*count
        self.n += count
        for total, part in zip(self.bins, ece_bins(logits, labels, 1., n_bins=self.n_bins)):
            total += part

    def result(self):
        return dict(nll=self.nll_sum/self.n, ece=ece_from_bins(*self.bins), n_voxels=self.n)


def _other(fold):
    return FOLD_NAMES[1-FOLD_INDEX[fold]]


def evaluate_cross_fold_calibration(*, files, guard, models_root, split, allowlist, training_source, device="cuda",
                                    pinned_folds_sha256=None, max_voxels=2_000_000, seed=0, out_path=None, log=print,
                                    command=None, inputs_record=None):
    """Write and return the calibration_crossfit.json document (see the module docstring)."""
    out_path = Path(out_path) if out_path is not None else Path(models_root)/CROSSFIT_FILE
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; never overwritten")
    wall = time.perf_counter()
    members, check = verify_members(models_root, split=split, allowlist=allowlist, training_source=training_source,
                                    pinned_folds_sha256=pinned_folds_sha256)
    device = torch.device(device)
    rng = np.random.default_rng([seed, 20261030])
    first_read = len(guard.reads)
    results = {}
    for target in FOLD_NAMES:
        source = _other(target)
        group = sorted((m for m in members if m["fold"] == source), key=lambda m: m["seed"])
        if not group:
            raise InputValidationError(f"no calibrated member of fold {source}")
        contact = group[0]["label_contact_subjects"]
        entry = dict(fold=FOLD_INDEX[source], member_model_sha256=[m["member_sha256"] for m in group],
                     label_contact_subjects=contact, label_contact_subjects_sha256=group[0]["label_contact_subjects_sha256"])
        target_subjects = split.subjects(target, "calib")
        if set(target_subjects) & set(contact):
            raise InputValidationError(f"fold {source} models saw labels of fold {target} calib subjects")
        loaded = [load_member(m["path"], m["member_sha256"], device, entry=entry) for m in group]
        percentiles = {blob["percentile"] for _, blob in loaded}
        if len(percentiles) != 1:
            raise InputValidationError(f"fold {source} members disagree on the normalization percentile")
        calib = load_training_subjects(files, guard, target, "calib", percentile=percentiles.pop(), log=log)
        ensemble, alone = _Accumulator(), [_Accumulator() for _ in loaded]
        kept_z, kept_y, floored = [], [], 0
        for item in calib:
            brain = torch.as_tensor(item.image > 0, device=device)
            labels = torch.as_tensor(item.labels.astype(np.int64), device=device)[brain]
            total = None
            for k, (model, blob) in enumerate(loaded):
                window = tuple(blob["window"]) if blob.get("window") else None
                z = predict_logits(model, item.image, device=device, amp=blob["amp"], window=window)[0]
                scaled = z[:, brain].T.double()/float(blob["temperature"])
                del z
                alone[k].add(scaled, labels)
                p = torch.softmax(scaled, dim=1)
                total = p if total is None else total.add_(p)
            total /= len(loaded)
            total /= total.sum(dim=1, keepdim=True)
            floored += int((total < LOG_FLOOR).sum())
            log_p = torch.log(total.clamp_min(LOG_FLOOR))
            ensemble.add(log_p, labels)
            kept_z.append(log_p.float().cpu())
            kept_y.append(labels.cpu())
            del total, log_p
        z_all, y_all = torch.cat(kept_z), torch.cat(kept_y)
        del kept_z, kept_y
        if max_voxels is not None and y_all.shape[0] > max_voxels:
            keep = torch.as_tensor(np.sort(rng.choice(y_all.shape[0], max_voxels, replace=False)))
            z_all, y_all = z_all[keep], y_all[keep]
        t_ens, fit = fit_temperature(z_all, y_all)
        del z_all, y_all
        results[target] = dict(
            target_fold=target, target_subjects=target_subjects, models_fold=source, model_id=model_id_for(source),
            members=[dict(seed=m["seed"], member_sha256=m["member_sha256"], temperature=m["temperature"], **acc.result())
                     for m, acc in zip(group, alone)],
            ensemble=ensemble.result(), probabilities_floored=floored, log_floor=LOG_FLOOR,
            ensemble_temperature_information_only=dict(temperature=t_ens, nll_before=fit["nll_before"],
                                                       nll_after=fit["nll_after"], ece_before=fit["ece_before"],
                                                       ece_after=fit["ece_after"], at_bound=fit["at_bound"],
                                                       n_voxels=fit["n_voxels"], max_voxels=max_voxels))
        log(f"calib of fold {target} by fold {source} ensemble: NLL {results[target]['ensemble']['nll']:.4f}, "
            f"ECE {results[target]['ensemble']['ece']:.4f}; members NLL "
            f"{[round(r['nll'], 4) for r in results[target]['members']]}; T_ens (not applied) {t_ens:.3f}")
    document = dict(format=CROSSFIT_FORMAT, ensemble_temperature_applied=False,
                    note="information only: the ensemble temperature is not applied (default); see crosscal.py",
                    voxel_set="aligned_norm > 0 (brain)", results=results, member_check=check,
                    label_guard=dict(guard.record(), label_reads=list(guard.reads[first_read:])), inputs=inputs_record,
                    environment=environment_record(device), command=command, wall_seconds=time.perf_counter()-wall)
    write_json(out_path, document)
    return document


__all__ = ["CROSSFIT_FILE", "CROSSFIT_FORMAT", "evaluate_cross_fold_calibration"]
