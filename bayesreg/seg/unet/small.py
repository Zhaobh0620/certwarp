"""The recipe r2_small of work package R2 (runs/20261110-r2/plan.md): the recipe v3_large as UF trained it (large.py; runs/20261105-uf), with an emphasis on a set of target labels T.

T comes from the pp diagnosis of R2 (experiments/r2/pp_r2.py: the small six and at most three subcortical structures, by a rule fixed before the diagnosis ran) and is given as
`target_labels`; it is recorded in the member's config like every other field. Against v3_large only these change, through the hooks of large.train_large_member (the loop, the seeds, the
records, the member format and the checkpoint choice are large.py's):
  - the patches (target_patch): with probability p_target the patch is centred on a voxel of a target class drawn uniformly among the targets present in the sample (the way of
    class_balanced_patch, the patch shifted inside the volume); otherwise large.class_balanced_patch with the config's p_foreground, as v3_large;
  - the loss (small_loss): large.deep_supervision_loss plus target_dice_weight x (1 - the mean soft Dice over T) on each output, with the deep-supervision weights and the labels taken as
    deep_supervision_loss takes them (every 2**k voxels). The soft Dice of a class is dice_ce_loss's (train.py): (2 sum_{y=k} p_k + s)/(sum p_k + |{y=k}| + s) over the batch; one softmax per output
    gives both terms (dice_ce_target_loss), the first bit for bit dice_ce_loss's.
The member is an ordinary UNet3D member (calibrate.py, registry.py and infer.py read it unchanged). Real data: target_labels must be non-empty labels of 1..n_classes-1, sorted, distinct.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from jd_curl.validation import InputValidationError

from . import large as _large
from .large import TrainConfigLarge, class_balanced_patch, ds_weights

PROFILE_R2 = "r2_small"


@dataclass
class TrainConfigSmall(TrainConfigLarge):
    """TrainConfigLarge (the defaults of U1; UF's three changes are given by the caller) with the target emphasis of R2."""
    target_labels: tuple = ()                   # T: labels of 1..n_classes-1, sorted and distinct (experiments/r2 targets.json)
    p_target: float = .5                        # the share of patches centred on a target voxel
    target_dice_weight: float = 1.              # lambda_T of the extra term lambda_T x (1 - mean soft Dice over T)


def require_targets(cfg):
    """The target fields of cfg checked; InputValidationError otherwise. Returns T as a tuple of ints."""
    targets = cfg.target_labels
    if isinstance(targets, (str, bytes)) or not isinstance(targets, (list, tuple)) or not targets:
        raise InputValidationError(f"target_labels {targets!r}: a non-empty list of labels is required")
    if any(isinstance(t, bool) or not isinstance(t, (int, np.integer)) for t in targets):
        raise InputValidationError(f"target_labels {targets!r}: integers only")
    targets = tuple(int(t) for t in targets)
    if list(targets) != sorted(set(targets)) or not all(1 <= t < cfg.n_classes for t in targets):
        raise InputValidationError(f"target_labels {targets}: sorted, distinct, within 1..{cfg.n_classes-1}")
    if isinstance(cfg.p_target, bool) or not (isinstance(cfg.p_target, (int, float)) and 0 <= cfg.p_target <= 1):
        raise InputValidationError(f"p_target {cfg.p_target!r}: a probability")
    if isinstance(cfg.target_dice_weight, bool) or not (isinstance(cfg.target_dice_weight, (int, float)) and math.isfinite(cfg.target_dice_weight) and cfg.target_dice_weight >= 0):
        raise InputValidationError(f"target_dice_weight {cfg.target_dice_weight!r}: a finite number >= 0")
    return targets


def target_patch(image, labels, patch, rng, *, targets, p_target, p_foreground=1/3, n_classes=36):
    """(image patch, label patch, record) from (1, 1, D, H, W), (1, D, H, W) tensors. One draw u: u < p_target and a target present -> a target class uniformly among those present, a voxel of it
    uniformly, the patch centred there and shifted inside the volume; otherwise class_balanced_patch (its own draws, v3_large's patch). Patches are always inside the volume."""
    shape = tuple(image.shape[2:])
    if any(p > n for p, n in zip(patch, shape)):
        raise InputValidationError(f"patch {tuple(patch)} larger than volume {shape}")
    if rng.random() < p_target:
        counts = torch.bincount(labels.reshape(-1), minlength=n_classes)
        present = [t for t in targets if int(counts[t]) > 0]
        if present:
            label = present[int(rng.integers(len(present)))]
            where = (labels[0] == label).nonzero()
            centre = where[int(rng.integers(len(where)))].tolist()
            start = [min(max(c-p//2, 0), n-p) for c, p, n in zip(centre, patch, shape)]
            sl = tuple(slice(s, s+p) for s, p in zip(start, patch))
            return image[(slice(None), slice(None))+sl], labels[(slice(None),)+sl], dict(foreground=True, target=True, label=label, start=start)
        x, y, record = class_balanced_patch(image, labels, patch, rng, p_foreground=p_foreground, n_classes=n_classes)
        return x, y, dict(record, target=False, target_absent=True)
    x, y, record = class_balanced_patch(image, labels, patch, rng, p_foreground=p_foreground, n_classes=n_classes)
    return x, y, dict(record, target=False)


def dice_ce_target_loss(logits, labels, targets, *, n_classes=36, ce_weight=1., dice_weight=1., target_weight=1., smooth=1e-5):
    """(loss, parts): train.dice_ce_loss (CE + 1 - mean soft Dice over classes 1..n-1, float32) plus target_weight x (1 - mean soft Dice over `targets`), from one softmax.
    With target_weight 0 the loss is dice_ce_loss's bit for bit (the same operations in the same order, then + 0)."""
    logits = logits.float()
    ce = F.cross_entropy(logits, labels)
    probs = torch.softmax(logits, dim=1)
    flat = labels.reshape(-1)
    p_true = probs.gather(1, labels[:, None]).reshape(-1)
    inter = torch.zeros(n_classes, device=logits.device, dtype=probs.dtype).scatter_add(0, flat, p_true)
    count = torch.bincount(flat, minlength=n_classes).to(probs.dtype)
    psum = probs.sum(dim=tuple(d for d in range(probs.ndim) if d != 1))
    dice = (2*inter+smooth)/(psum+count+smooth)
    soft_dice = dice[1:].mean()
    index = torch.as_tensor(list(targets), device=logits.device, dtype=torch.long)
    target_dice = dice[index].mean()
    main = ce_weight*ce+dice_weight*(1-soft_dice)
    return main+target_weight*(1-target_dice), dict(ce=float(ce.detach()), soft_dice=float(soft_dice.detach()), target_soft_dice=float(target_dice.detach()), main=float(main.detach()))


def small_loss(outputs, labels, cfg):
    """large.deep_supervision_loss's sum over the outputs (weights ds_weights, labels every 2**k voxels) of dice_ce_target_loss. Returns (loss, parts of the full-resolution output with the loss of each level)."""
    targets = require_targets(cfg)
    weights = ds_weights(len(outputs))
    total, levels, target_levels, parts = 0., [], [], None
    for k, (output, weight) in enumerate(zip(outputs, weights)):
        step = 2**k
        target = labels[:, ::step, ::step, ::step]
        if tuple(target.shape[1:]) != tuple(output.shape[2:]):
            raise ValueError(f"level {k}: labels {tuple(target.shape[1:])} do not fit the output {tuple(output.shape[2:])}")
        loss, part = dice_ce_target_loss(output, target, targets, n_classes=cfg.n_classes, ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight, target_weight=cfg.target_dice_weight, smooth=cfg.dice_smooth)
        total = total+weight*loss
        levels.append(float(loss.detach()))
        target_levels.append(part["target_soft_dice"])
        parts = parts or part
    return total, dict(parts, level_losses=levels, level_target_soft_dice=target_levels)


def small_patch(x, y, patch, rng, cfg):
    return target_patch(x, y, patch, rng, targets=require_targets(cfg), p_target=cfg.p_target, p_foreground=cfg.p_foreground, n_classes=cfg.n_classes)


def train_small_member(*, files, guard, fold, seed, models_root, config=None, device="cuda", log=print, command=None, subjects=None, inputs_record=None):
    """large.train_large_member on TrainConfigSmall with this recipe's patches and loss; the record's recipe is r2_small. The targets are checked before anything is read."""
    cfg = config if isinstance(config, TrainConfigSmall) else TrainConfigSmall(**(config or {}))
    require_targets(cfg)
    return _large.train_large_member(files=files, guard=guard, fold=fold, seed=seed, models_root=models_root, config=cfg, device=device, log=log, command=command, subjects=subjects, inputs_record=inputs_record,
                              patch_fn=small_patch, loss_fn=small_loss, recipe=PROFILE_R2)


__all__ = ["PROFILE_R2", "TrainConfigSmall", "dice_ce_target_loss", "require_targets", "small_loss", "small_patch", "target_patch", "train_small_member"]
