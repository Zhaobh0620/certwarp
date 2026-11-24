"""The profile v3_large of the structure-layer segmenter: a larger U-Net on a recipe nearer to nnU-Net's (work package UL, runs/20261104-ul/plan.md, 配方差异.md; docs/bayesreg/04_分割器.md §15).

One arm, U1 = one member (fold A of folds_full_v2.json, seed 0). Compared with train.train_member, only these change; everything else is taken from train.py and data.py as they are (AdamW, poly schedule, Dice + CE, AMP, the augmentation
of data.augment, the label guard, the member and record formats):
  - base 32 instead of 16 (depth 4, widths 32 64 128 256 320; the same UNet3D);
  - training on patches of `patch_size` in batches of `batch_size` (one subject per sample), foreground patches centred on a voxel of a random class present in the sample (class_balanced_patch);
  - deep supervision: auxiliary heads on the decoder levels 1..ds_levels, used only in training (UNet3DDeepSupervision, deep_supervision_loss); the checkpoint holds the plain UNet3D, so calibrate.py, infer.py and build_model read it unchanged;
  - left-right flip with the labels swapped (flip_left_right), the pairing read from the seg35 label table (left_right_label_map), the axis checked on the training labels before the first step (check_flip_axis);
  - sliding-window inference with a Gaussian importance window (train.predict_logits, gaussian_sigma), recorded in the member (`window_gaussian_sigma`).

The old profiles never reach this module. The GPU budget is a hard limit of the process (gpu_cap_gib; an allocation above it raises, nothing is shared with another job), the wall budget a stop of the training loop (`stopped` = "wall_budget").

Which weights become the member (`checkpoint`, work package UF, runs/20261105-uf/plan.md): "best_calib" (the default, U1) is model_best.pt, the evaluation with the highest calib Dice35; "last" also writes model_last.pt, the weights after the last
step, and the train record names it (checkpoint_choice, checkpoint_file, checkpoint_sha256, checkpoint_iteration); calibrate.py then takes those weights. The evaluations, model_best.pt and the record of every calib Dice are the same under both;
under "best_calib" no file and no record key is added, except that the recorded config (train_record.json and the start event of train_log.jsonl) carries "checkpoint": "best_calib" (U1's records, written before the
field existed, do not). The bitwise fingerprint of test_ul.py covers v1 and v2_full only, not v3_large.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import resource
import time

import numpy as np
import torch
from torch import nn

from jd_curl.validation import InputValidationError

from ...io.pair import subject_list_sha256
from .data import FOLD_INDEX, FOLD_NAMES, augment, load_training_subjects
from .model import UNet3D, count_parameters
from .pins import check_synthetic_data_root
from .snapshot import CODE_ROOT
from .train import (TrainConfig, _json_default, _lr_at, atomic_torch_save, dice_ce_loss, environment_record, evaluate, file_sha256, member_dir, write_json)

SEGMENTER_ID_V3 = "unet35_v3_large"
MODEL_PREFIX_V3 = "unet35_v3_large"
PROFILE = "v3_large"
GROUPS_FILE = CODE_ROOT/"configs"/"bayesreg"/"structure_groups_v1.json"
FLIP_DIM = -1                               # the last array axis: jd_curl load_scalar stores (nz, ny, nx) in RAS, x runs from the left to the right of the head; check_flip_axis checks it on the data
GIB = 2**30
PATCH_U1 = (128, 160, 112)                  # (z, y, x); the aspect of the volume (192, 224, 160); estimate in runs/20261104-ul/estimate_20261104.out
CHECK_MIN_SEPARATION_VOX = 10.              # check_flip_axis: the centroids of a left and right cerebral structure at least this far apart along x
CHECKPOINT_CHOICES = ("best_calib", "last")   # TrainConfigLarge.checkpoint
LAST_CHECKPOINT = "model_last.pt"


# ---- the left-right pairing, from the label table -------------------------------------------------------------------------------------------------

def load_label_names(path=GROUPS_FILE):
    """{label: name} of the seg35 labels: `label_table.names` of structure_groups_v1.json (a copy of neurite-oasis seg35_labels.txt, whose sha256 it records)."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {int(k): v for k, v in document["label_table"]["names"].items()}
    if sorted(names) != list(range(document["n_network_classes"])):
        raise InputValidationError(f"{path}: the label table must name the labels 0..{document['n_network_classes']-1}")
    return names


def left_right_label_map(names=None):
    """(lut, pairs, unpaired): lut[l] = the label of the mirrored structure, uint8 array; pairs = sorted (left, right); unpaired = labels mapped to themselves (midline structures and background).
    Read from the names only: "Left-X" pairs with "Right-X". A name that mentions a side in any other form, or a side without its partner, is refused rather than guessed."""
    names = load_label_names() if names is None else {int(k): v for k, v in names.items()}
    by_name = {}
    for label, name in names.items():
        if name in by_name:
            raise InputValidationError(f"label table: the name {name!r} is listed twice")
        by_name[name] = label
    lut = np.arange(len(names), dtype=np.uint8)
    pairs, unpaired = [], []
    for label, name in sorted(names.items()):
        if name.startswith("Left-"):
            partner = "Right-"+name[len("Left-"):]
            if partner not in by_name:
                raise InputValidationError(f"label table: {name!r} has no partner {partner!r}")
            lut[label], lut[by_name[partner]] = by_name[partner], label
            pairs.append((label, by_name[partner]))
        elif name.startswith("Right-"):
            if "Left-"+name[len("Right-"):] not in by_name:
                raise InputValidationError(f"label table: {name!r} has no partner on the left")
        elif re.search(r"left|right|(^|[^a-z])[lr]h([^a-z]|$)", name, re.IGNORECASE):
            raise InputValidationError(f"label table: {name!r} names a side in a form this rule does not read")
        else:
            unpaired.append(label)
    if not np.array_equal(lut[lut], np.arange(len(names))):
        raise InputValidationError("label table: the left-right map is not an involution")
    return lut, pairs, unpaired


def flip_left_right(image, labels, lut):
    """Mirror along the left-right axis, labels swapped: image (..., D, H, W), labels (..., D, H, W) long, lut a long tensor on the same device."""
    return torch.flip(image, (FLIP_DIM,)), lut[torch.flip(labels, (FLIP_DIM,))]


def _centroids(labels, n_classes):
    """(n_classes, 3) mean index of each label along (z, y, x), every second voxel (nan where absent) and the counts."""
    sparse = np.asarray(labels)[::2, ::2, ::2].astype(np.int64)
    counts = np.bincount(sparse.ravel(), minlength=n_classes).astype(np.float64)
    means = np.full((n_classes, 3), np.nan)
    for axis, n in enumerate(sparse.shape):
        shape = [1, 1, 1]
        shape[axis] = n
        coordinate = np.broadcast_to(2.*np.arange(n).reshape(shape), sparse.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            means[:, axis] = np.bincount(sparse.ravel(), weights=coordinate.ravel(), minlength=n_classes)/counts
    return means, counts


def check_flip_axis(subjects, names=None, *, min_separation=CHECK_MIN_SEPARATION_VOX):
    """Before the first training step, on the training subjects' labels (already read through the guard): every left and right cerebral structure ("Cerebral" in the name) must lie on the expected side along x, the last array
    axis (RAS: the left at low index), and x must be the axis that separates them most. Any departure: InputValidationError, nothing is trained. Returns the record."""
    names = load_label_names() if names is None else names
    _, pairs, _ = left_right_label_map(names)
    required = [(a, b) for a, b in pairs if "Cerebral" in names[a]]
    if not required:
        raise InputValidationError("check_flip_axis: the label table has no left and right cerebral structure")
    worst = math.inf
    for item in subjects:
        means, counts = _centroids(item.labels, len(names))
        for left, right in required:
            if not counts[left] or not counts[right]:
                raise InputValidationError(f"check_flip_axis: {item.subject} has no voxel of {names[left]} or {names[right]}")
            separation = means[right]-means[left]
            if separation[2] < min_separation or abs(separation[2]) <= abs(separation[0]) or abs(separation[2]) <= abs(separation[1]):
                raise InputValidationError(f"check_flip_axis: {item.subject}, {names[left]} to {names[right]}: centroid difference (z, y, x) {separation.tolist()}; expected the left at low x, x the largest; not trained")
            worst = min(worst, float(separation[2]))
    return dict(check="left labels at low x, x separates them most", pairs=[[int(a), int(b)] for a, b in required], n_subjects=len(subjects), smallest_x_separation_vox=worst, min_separation_vox=min_separation)


# ---- the network with auxiliary heads ------------------------------------------------------------------------------------------------------------

class UNet3DDeepSupervision(UNet3D):
    """UNet3D plus 1x1x1 heads on the decoder levels 1..ds_levels. In training mode the forward returns [full resolution, level 1, ..., level ds_levels]; in eval mode exactly what UNet3D returns. The parent's modules are built first, so its
    initial weights are those of a UNet3D with the same seed. plain_state_dict() is a UNet3D checkpoint."""

    def __init__(self, in_channels=1, n_classes=36, base_channels=32, depth=4, max_channels=320, ds_levels=2):
        super().__init__(in_channels=in_channels, n_classes=n_classes, base_channels=base_channels, depth=depth, max_channels=max_channels)
        if not 0 <= ds_levels < depth:
            raise ValueError(f"ds_levels {ds_levels} must be in 0..{depth-1}")
        self.ds_levels = ds_levels
        self.aux_heads = nn.ModuleList([nn.Conv3d(self.widths[k], n_classes, 1) for k in range(1, ds_levels+1)])

    def forward(self, x):
        if not self.training:
            return super().forward(x)
        if any(n % self.multiple for n in x.shape[2:]):
            raise ValueError(f"spatial sizes {tuple(x.shape[2:])} must be multiples of {self.multiple}")
        skips = []
        for k, block in enumerate(self.encoder):
            x = block(x)
            if k < self.depth:
                skips.append(x)
        aux = [None]*self.ds_levels
        for k in reversed(range(self.depth)):
            x = self.decoder[k](torch.cat([self.up[k](x), skips[k]], dim=1))
            if 1 <= k <= self.ds_levels:
                aux[k-1] = self.aux_heads[k-1](x)
        return [self.head(x)]+aux

    def plain_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("aux_heads.")}


def ds_weights(n_outputs):
    """1, 1/2, 1/4, ... normalized to sum 1."""
    weights = np.array([.5**k for k in range(n_outputs)])
    return (weights/weights.sum()).tolist()


def deep_supervision_loss(outputs, labels, *, n_classes=36, ce_weight=1., dice_weight=1., smooth=1e-5):
    """Sum over the outputs of weight_k * dice_ce_loss(output_k, labels sampled every 2**k voxels). Level k of the decoder sits on the grid of the strided encoder (padding 1, stride 2: output i is centred on input 2i), so labels[::2**k]
    is the nearest label of each of its voxels. Returns (loss, parts of the full-resolution output, with the loss of each level)."""
    weights = ds_weights(len(outputs))
    total, levels, parts = 0., [], None
    for k, (output, weight) in enumerate(zip(outputs, weights)):
        step = 2**k
        target = labels[:, ::step, ::step, ::step]
        if tuple(target.shape[1:]) != tuple(output.shape[2:]):
            raise ValueError(f"level {k}: labels {tuple(target.shape[1:])} do not fit the output {tuple(output.shape[2:])}")
        loss, part = dice_ce_loss(output, target, n_classes=n_classes, ce_weight=ce_weight, dice_weight=dice_weight, smooth=smooth)
        total = total+weight*loss
        levels.append(float(loss.detach()))
        parts = parts or part
    return total, dict(parts, level_losses=levels)


# ---- patches ----------------------------------------------------------------------------------------------------------------------------------------

def class_balanced_patch(image, labels, patch, rng, *, p_foreground=1/3, n_classes=36):
    """(image patch, label patch, record) from (1, 1, D, H, W), (1, D, H, W) tensors. With probability p_foreground: a class drawn uniformly among the labels > 0 present in this sample, a voxel of it drawn uniformly, the patch centred on it and
    shifted inside the volume (data.random_patch draws the voxel over all labels > 0, so it almost never lands on a small structure). Otherwise a uniform position. Patches are always inside the volume."""
    shape = tuple(image.shape[2:])
    if any(p > n for p, n in zip(patch, shape)):
        raise InputValidationError(f"patch {tuple(patch)} larger than volume {shape}")
    record = dict(foreground=False)
    start = None
    if rng.random() < p_foreground:
        counts = torch.bincount(labels.reshape(-1), minlength=n_classes)
        present = (counts[1:] > 0).nonzero().reshape(-1).add(1).tolist()
        if present:
            label = present[int(rng.integers(len(present)))]
            where = (labels[0] == label).nonzero()
            centre = where[int(rng.integers(len(where)))].tolist()
            start = [min(max(c-p//2, 0), n-p) for c, p, n in zip(centre, patch, shape)]
            record.update(foreground=True, label=label)
    if start is None:
        start = [int(rng.integers(0, n-p+1)) for p, n in zip(patch, shape)]
    sl = tuple(slice(s, s+p) for s, p in zip(start, patch))
    return image[(slice(None), slice(None))+sl], labels[(slice(None),)+sl], dict(record, start=start)


def window_starts(n, window, overlap=.5):
    """The starts train.predict_logits uses along an axis of n voxels, for a window side <= n."""
    step = max(1, int(window*(1-overlap)))
    starts = list(range(0, n-window+1, step))
    return starts+([n-window] if starts[-1] != n-window else [])


def windows_per_volume(shape, window, overlap=.5):
    """How many forward passes train.predict_logits runs on a volume of `shape`, and the starts per axis."""
    starts = [window_starts(n, min(w, n), overlap) for n, w in zip(shape, window)]
    return len(starts[0])*len(starts[1])*len(starts[2]), starts


# ---- configuration ----------------------------------------------------------------------------------------------------------------------------------

@dataclass
class TrainConfigLarge(TrainConfig):
    """TrainConfig (whose defaults stay those of v1 and v2) with the defaults of U1. max_iters is set from the smoke run by experiments/ul/run_ul.py decide-steps."""
    base_channels: int = 32
    patch_size: tuple = PATCH_U1
    batch_size: int = 2
    ds_levels: int = 2
    flip_probability: float = .5
    p_foreground: float = 1/3
    window_gaussian_sigma: float = .125          # of the window's side, as nnU-Net's 1/8 (not checked against its source)
    max_iters: int = 30000
    patience: int = 20
    min_iters: int = 10000
    max_wall_seconds: float = None               # stop the loop after this many seconds from the start of train_large_member, data loading included
    gpu_cap_gib: float = 20.
    flip_axis_check: bool = True
    log_memory: bool = True
    checkpoint: str = "best_calib"               # CHECKPOINT_CHOICES: the weights of the member (module docstring); "last" for UF


def require_checkpoint_choice(cfg):
    if cfg.checkpoint not in CHECKPOINT_CHOICES:
        raise InputValidationError(f"checkpoint {cfg.checkpoint!r}: one of {CHECKPOINT_CHOICES}")
    return cfg.checkpoint


def require_patch(cfg):
    patch = tuple(cfg.patch_size) if cfg.patch_size else None
    if patch is None or len(patch) != 3 or any(p % 2**cfg.depth or p < 1 for p in patch):
        raise InputValidationError(f"patch_size {cfg.patch_size}: three positive multiples of {2**cfg.depth} required")
    if cfg.batch_size < 1:
        raise InputValidationError("batch_size must be positive")
    return patch


def require_flip_check(cfg, files, subjects):
    """flip_axis_check is True or False, nothing else, and False only for subjects handed in (tests) or a non-real data root: no value on the command line skips the check on real labels."""
    if cfg.flip_axis_check is True:
        return
    if cfg.flip_axis_check is not False:
        raise InputValidationError(f"flip_axis_check {cfg.flip_axis_check!r}: true or false")
    if subjects is None:
        check_synthetic_data_root(files.root)


def cap_gpu_memory(device, cap_gib, *, cuda=torch.cuda):
    """The hard limit of this process on a cuda device: an allocation above it raises. torch.cuda.set_per_process_memory_fraction refuses a device without an index ("cuda"), so the index is resolved here."""
    index = device.index if device.index is not None else cuda.current_device()
    total = cuda.get_device_properties(index).total_memory
    cuda.set_per_process_memory_fraction(min(1., float(cap_gib)*GIB/total), index)
    return index


def peak_host_gib():
    """Peak resident memory of this process so far (Linux ru_maxrss, KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20


def memory_entry(device):
    entry = dict(host_peak_rss_gib=peak_host_gib())
    if torch.device(device).type == "cuda":
        entry.update(gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/GIB, gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/GIB)
    return entry


# ---- training -----------------------------------------------------------------------------------------------------------------------------------------

def default_patch(x, y, patch, rng, cfg):
    """The patch of the recipe v3_large: class_balanced_patch with the config's p_foreground (the hook `patch_fn` of train_large_member; small.py gives another)."""
    return class_balanced_patch(x, y, patch, rng, p_foreground=cfg.p_foreground, n_classes=cfg.n_classes)


def default_loss(outputs, labels, cfg):
    """The loss of the recipe v3_large: deep_supervision_loss with the config's weights (the hook `loss_fn` of train_large_member; small.py gives another)."""
    return deep_supervision_loss(outputs, labels, n_classes=cfg.n_classes, ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight, smooth=cfg.dice_smooth)


def train_large_member(*, files, guard, fold, seed, models_root, config=None, device="cuda", log=print, command=None, subjects=None, inputs_record=None, patch_fn=default_patch, loss_fn=default_loss, recipe=PROFILE):
    """As train.train_member (same seeds, outputs and record format; only the fold's train subjects are trained on, through the guard, its calib subjects decide the best iteration), on the recipe of this module. Returns the train record.
    patch_fn(x, y, patch, rng, cfg), loss_fn(outputs, y, cfg) and recipe (the record's "recipe"): the hooks of another recipe on the same loop (small.py, work package R2, runs/20261110-r2/plan.md); the defaults are this module's, the same calls as before the hooks."""
    if fold not in FOLD_NAMES:
        raise InputValidationError(f"fold must be one of {FOLD_NAMES}")
    cfg = config if isinstance(config, TrainConfigLarge) else TrainConfigLarge(**(config or {}))
    patch = require_patch(cfg)
    require_flip_check(cfg, files, subjects)
    require_checkpoint_choice(cfg)
    if cfg.max_iters < 1 or cfg.eval_every < 1:
        raise InputValidationError("max_iters and eval_every must be positive")
    out = member_dir(models_root, fold, seed)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; members are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    started, wall = time.strftime("%Y-%m-%dT%H:%M:%S%z"), time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, FOLD_INDEX[fold], 20261030])
    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        cap_gpu_memory(device, cfg.gpu_cap_gib)
    label_guard = None
    if subjects is None:
        first_read = len(guard.reads)
        train_set = load_training_subjects(files, guard, fold, "train", percentile=cfg.percentile, log=log)
        calib_set = load_training_subjects(files, guard, fold, "calib", percentile=cfg.percentile, log=log)
        label_guard = dict(guard.record(), label_reads=list(guard.reads[first_read:]))
    else:
        train_set, calib_set = subjects
    if not train_set or not calib_set:
        raise InputValidationError("both train and calib subjects are required")
    names = load_label_names()
    lut_np, pairs, unpaired = left_right_label_map(names)
    flip_check = check_flip_axis(train_set, names) if cfg.flip_axis_check else dict(skipped=True, reason="flip_axis_check false (synthetic subjects only)")
    if cfg.flip_axis_check:
        log(f"flip axis checked on {flip_check['n_subjects']} subjects: smallest x separation {flip_check['smallest_x_separation_vox']:.1f} voxels")
    lut = torch.as_tensor(lut_np.astype(np.int64), device=device)
    contact = sorted({s.subject for s in train_set} | {s.subject for s in calib_set})
    model = UNet3DDeepSupervision(n_classes=cfg.n_classes, base_channels=cfg.base_channels, depth=cfg.depth, max_channels=cfg.max_channels, ds_levels=cfg.ds_levels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=tuple(cfg.betas))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    log_path = out/"train_log.jsonl"
    best, best_it, bad_evals, history, order, best_metrics = -math.inf, -1, 0, [], [], None
    stopped = "max_iters"
    steps_done, flips = 0, 0
    model.train()
    with log_path.open("a", encoding="utf-8") as stream:
        def emit(entry):
            stream.write(json.dumps(entry, default=_json_default)+"\n")
            stream.flush()
        emit(dict(event="start", fold=fold, seed=seed, n_train=len(train_set), n_calib=len(calib_set), parameters=count_parameters(model), config=asdict(cfg), started=started, recipe=recipe, flip_check=flip_check,
                  left_right_pairs=pairs, unpaired_labels=unpaired, ds_weights=ds_weights(cfg.ds_levels+1), data_ready_seconds=time.perf_counter()-wall, **memory_entry(device)))
        step_clock = time.perf_counter()
        for it in range(cfg.max_iters):
            for group in opt.param_groups:
                group["lr"] = _lr_at(it, cfg)
            xs, ys, batch_subjects, records = [], [], [], []
            for _ in range(cfg.batch_size):
                if not order:
                    order = list(rng.permutation(len(train_set)))
                item = train_set[order.pop()]
                x = torch.as_tensor(item.image, device=device)[None, None]
                y = torch.as_tensor(item.labels.astype(np.int64), device=device)[None]
                flipped = bool(rng.random() < cfg.flip_probability)
                if flipped:
                    x, y = flip_left_right(x, y, lut)
                    flips += 1
                x, y, aug = augment(x, y, rng, cfg.augment)
                x, y, where = patch_fn(x, y, patch, rng, cfg)
                xs.append(x)
                ys.append(y)
                batch_subjects.append(item.subject)
                records.append(dict(aug, flipped=flipped, **where))
            x, y = torch.cat(xs), torch.cat(ys)
            del xs, ys
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(x)
            loss, parts = loss_fn(outputs, y, cfg)
            if not torch.isfinite(loss):
                emit(dict(event="nonfinite_loss", it=it, subjects=batch_subjects, records=records))
                raise FloatingPointError(f"non-finite loss at iteration {it} ({batch_subjects}); log kept in {log_path}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            scaler.step(opt)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds = time.perf_counter()-step_clock
            steps_done = it+1
            if it % cfg.log_every == 0:
                emit(dict(event="train", it=it, loss=float(loss.detach()), lr=_lr_at(it, cfg), grad_norm=grad_norm, subjects=batch_subjects, elapsed=time.perf_counter()-wall, step_seconds=step_seconds, **parts,
                          **(memory_entry(device) if cfg.log_memory else {})))
            del outputs, loss, x, y                  # not alive during the next forward pass or the evaluation
            out_of_time = cfg.max_wall_seconds is not None and time.perf_counter()-wall >= cfg.max_wall_seconds
            last = it == cfg.max_iters-1 or out_of_time
            if (it+1) % cfg.eval_every == 0 or last:
                eval_clock = time.perf_counter()
                metrics = evaluate(model, calib_set, device=device, amp=cfg.amp, window=patch, n_classes=cfg.n_classes, gaussian_sigma=cfg.window_gaussian_sigma)
                improved = metrics["mean_dice35"] > best or best_metrics is None
                entry = dict(event="eval", it=it+1, improved=improved, elapsed=time.perf_counter()-wall, eval_seconds=time.perf_counter()-eval_clock, mean_dice35=metrics["mean_dice35"], mean_ce=metrics["mean_ce"],
                             per_subject_dice35=metrics["per_subject_dice35"], **memory_entry(device))
                if device.type == "cuda":
                    entry["max_memory_gb"] = torch.cuda.max_memory_allocated(device)/1e9
                emit(entry)
                history.append(entry)
                if improved:
                    best, best_it, bad_evals = metrics["mean_dice35"], it+1, 0
                    atomic_torch_save(dict(format="bayesreg.unet35.checkpoint.v1", state_dict=model.plain_state_dict(), model_config=cfg.model_config(), iteration=it+1, calib_mean_dice35=best, fold=fold, seed=seed), out/"model_best.pt")
                    best_metrics = metrics
                else:
                    bad_evals += 1
                    if bad_evals >= cfg.patience and it+1 >= cfg.min_iters:
                        stopped = "early_stop"
                        break
                if out_of_time and it != cfg.max_iters-1:
                    stopped = "wall_budget"
                    break
            step_clock = time.perf_counter()
        last_checkpoint = {}
        if cfg.checkpoint == "last":                 # the weights after the last step; the last step is always evaluated (`last` above, or the break of an early or wall stop inside the evaluation)
            atomic_torch_save(dict(format="bayesreg.unet35.checkpoint.v1", state_dict=model.plain_state_dict(), model_config=cfg.model_config(), iteration=steps_done, calib_mean_dice35=history[-1]["mean_dice35"], fold=fold, seed=seed, checkpoint="last"), out/LAST_CHECKPOINT)
            last_checkpoint = dict(checkpoint_choice="last", checkpoint_file=LAST_CHECKPOINT, checkpoint_sha256=file_sha256(out/LAST_CHECKPOINT), checkpoint_iteration=steps_done, checkpoint_calib_mean_dice35=history[-1]["mean_dice35"])
        emit(dict(event="end", stopped=stopped, best_iteration=best_it, best_mean_dice35=best, elapsed=time.perf_counter()-wall, steps_done=steps_done, flips=flips, **last_checkpoint, **memory_entry(device)))
    record = dict(format="bayesreg.unet35.train_record.v1", fold=fold, fold_index=FOLD_INDEX[fold], seed=seed, recipe=recipe, train_subjects=[s.subject for s in train_set], calib_subjects=[s.subject for s in calib_set],
                  label_contact_subjects=contact, label_contact_subjects_sha256=subject_list_sha256(contact), image_sha256={s.subject: s.image_sha256 for s in train_set+calib_set},
                  label_sha256={s.subject: s.label_sha256 for s in train_set+calib_set}, normalization={s.subject: s.normalization for s in train_set+calib_set}, config=asdict(cfg), parameters=count_parameters(model),
                  parameters_plain=sum(v.numel() for v in model.plain_state_dict().values()), stopped=stopped, iterations_run=steps_done, best_iteration=best_it, best_calib_mean_dice35=best,
                  best_calib_per_label_dice=best_metrics["per_label_dice"], eval_history=history, model_best_sha256=file_sha256(out/"model_best.pt"), label_guard=label_guard, inputs=inputs_record, flip_check=flip_check,
                  left_right_pairs=pairs, unpaired_labels=unpaired, memory=memory_entry(device), environment=environment_record(device), command=command, started=started, wall_seconds=time.perf_counter()-wall)
    record.update(last_checkpoint)                   # nothing under "best_calib" (only the config carries the new field)
    write_json(out/"train_record.json", record)
    log(f"fold {fold} seed {seed}: {stopped} after {steps_done} it, best calib Dice35 {best:.4f} at {best_it}, {record['wall_seconds']:.0f} s"+(f"; the member takes the last step ({steps_done})" if last_checkpoint else ""))
    return record


__all__ = ["CHECKPOINT_CHOICES", "CHECK_MIN_SEPARATION_VOX", "FLIP_DIM", "LAST_CHECKPOINT", "cap_gpu_memory", "default_loss", "default_patch", "require_checkpoint_choice", "require_flip_check", "GROUPS_FILE", "MODEL_PREFIX_V3", "PATCH_U1", "PROFILE", "SEGMENTER_ID_V3", "TrainConfigLarge", "UNet3DDeepSupervision", "check_flip_axis", "class_balanced_patch", "deep_supervision_loss",
           "ds_weights", "flip_left_right", "left_right_label_map", "load_label_names", "memory_entry", "peak_host_gib", "require_patch", "train_large_member", "window_starts", "windows_per_volume"]
