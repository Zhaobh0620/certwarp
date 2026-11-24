"""Training of one fold model member (04_分割器.md §4): Dice + cross-entropy, AdamW, poly schedule,
early stopping on the fold's calib subjects (mean Dice over the 35 labels, nanmean).

One member = (fold, seed). Only the fold's train subjects are trained on; its calib subjects are used
for early stopping (and later for the temperature, calibrate.py); both are the member's label contact.
Outputs in <models_root>/fold<F>_s<seed>/ (never overwritten): model_best.pt, train_log.jsonl,
train_record.json.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from jd_curl.validation import InputValidationError

from ...io.pair import subject_list_sha256
from .data import DEFAULT_AUGMENT, FOLD_INDEX, FOLD_NAMES, augment, load_training_subjects, random_patch
from .model import N_CLASSES, build_model, count_parameters, crop_to, pad_to_multiple
from .snapshot import code_record

MEMBER_DIR = "fold{fold}_s{seed}"


@dataclass
class TrainConfig:
    base_channels: int = 16
    depth: int = 4
    n_classes: int = N_CLASSES
    max_channels: int = 320
    lr: float = 5e-4
    weight_decay: float = 1e-4
    betas: tuple = (.9, .999)
    warmup_iters: int = 200
    max_iters: int = 20000
    poly_power: float = .9
    eval_every: int = 500
    patience: int = 10                  # evaluations without improvement before stopping
    min_iters: int = 4000               # never stop before this many iterations
    patch_size: tuple = None            # None = whole volume (padded to a multiple of 2**depth)
    amp: bool = True
    grad_clip: float = 12.
    ce_weight: float = 1.
    dice_weight: float = 1.
    dice_smooth: float = 1e-5
    percentile: float = 99.5
    log_every: int = 50
    cudnn_benchmark: bool = True
    augment: dict = field(default_factory=lambda: dict(DEFAULT_AUGMENT))

    def model_config(self):
        return dict(base_channels=self.base_channels, depth=self.depth, n_classes=self.n_classes,
                    max_channels=self.max_channels)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_record(device):
    """Versions (python, torch, numpy, scipy, cuDNN, CUDA), device and the code record
    (snapshot.code_record: the verified snapshot, or the git HEAD with a dirty flag)."""
    import scipy
    code = code_record()
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    record = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                  cudnn=cudnn, platform=platform.platform(), device=str(device), git_sha=code.get("commit"),
                  code=code, pid=os.getpid())
    if torch.device(device).type == "cuda":
        record.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(torch.device(device)))
    return record


def atomic_torch_save(obj, path):
    tmp = Path(str(path)+".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path, document):
    tmp = Path(str(path)+".tmp")
    tmp.write_text(json.dumps(document, indent=1, sort_keys=False, default=_json_default)+"\n", encoding="utf-8")
    os.replace(tmp, path)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


# ---- loss and metrics -----------------------------------------------------------------------------

def dice_ce_loss(logits, labels, *, n_classes=N_CLASSES, ce_weight=1., dice_weight=1., smooth=1e-5):
    """CE (mean over voxels) + (1 - mean soft Dice over classes 1..n-1), computed in float32.

    Soft Dice per class: (2 sum_{y=k} p_k + s)/(sum p_k + |{y=k}| + s) over the whole batch; the
    intersection uses scatter_add of p_y (no one-hot tensor).
    """
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
    return ce_weight*ce+dice_weight*(1-soft_dice), dict(ce=float(ce.detach()), soft_dice=float(soft_dice.detach()))


def nanmean_rows(rows):
    """Column nanmean of stacked per-subject Dice rows; all-nan columns stay nan without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(rows), axis=0)


def confusion_dice(pred, true, n_classes=N_CLASSES):
    """Hard Dice per label 1..n-1 (nan where a label is absent from both); numpy float64 array."""
    pred, true = torch.as_tensor(pred).reshape(-1).long(), torch.as_tensor(true).reshape(-1).long()
    conf = torch.bincount(true*n_classes+pred, minlength=n_classes*n_classes).reshape(n_classes, n_classes)
    conf = conf.double().cpu().numpy()
    inter, size_true, size_pred = np.diag(conf), conf.sum(axis=1), conf.sum(axis=0)
    denom = size_true+size_pred
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = np.where(denom > 0, 2*inter/denom, np.nan)
    return dice[1:]


def gaussian_window(window, sigma, device):
    """Importance window (D, H, W) float32: a product of 1-D Gaussians centred in the window, standard deviation
    sigma * side, peak at the centre (1 on an odd side), positive everywhere (profile v3_large, large.py)."""
    axes = [torch.exp(-.5*((torch.arange(n, dtype=torch.float32, device=device)-(n-1)/2)/(sigma*n))**2) for n in window]
    return axes[0][:, None, None]*axes[1][None, :, None]*axes[2][None, None, :]


def predict_logits(model, image, *, device, amp=True, window=None, overlap=.5, gaussian_sigma=None):
    """Logits (1, C, D, H, W) float32 for a (D, H, W) float32 numpy image (or (1,1,D,H,W) tensor).

    window None: one pass over the whole volume padded to a multiple of 2**depth. Otherwise a sliding
    window of that size (stride = window*(1-overlap)), logits averaged where windows overlap: plainly
    (gaussian_sigma None, v1 and v2), or weighted by gaussian_window (v3_large).
    """
    x = torch.as_tensor(image)
    if x.ndim == 3:
        x = x[None, None]
    x = x.to(device=device, dtype=torch.float32)
    x, shape = pad_to_multiple(x, model.multiple)
    device_type = torch.device(device).type
    use_amp = amp and device_type == "cuda"
    with torch.no_grad(), torch.autocast(device_type, dtype=torch.float16, enabled=use_amp):
        if window is None:
            out = model(x).float()
        else:
            window = [min(w, n) for w, n in zip(window, x.shape[2:])]
            if any(w % model.multiple for w in window):
                raise InputValidationError(f"window {window} must be multiples of {model.multiple}")
            out = torch.zeros((1, model.n_classes)+tuple(x.shape[2:]), device=x.device)
            hits = torch.zeros((1, 1)+tuple(x.shape[2:]), device=x.device)
            weight = None if gaussian_sigma is None else gaussian_window(window, gaussian_sigma, x.device)
            starts = []
            for n, w in zip(x.shape[2:], window):
                step = max(1, int(w*(1-overlap)))
                s = list(range(0, n-w+1, step))
                if s[-1] != n-w:
                    s.append(n-w)
                starts.append(s)
            for a in starts[0]:
                for b in starts[1]:
                    for c in starts[2]:
                        sl = (slice(None), slice(None), slice(a, a+window[0]), slice(b, b+window[1]),
                              slice(c, c+window[2]))
                        if weight is None:
                            out[sl] += model(x[sl]).float()
                            hits[sl] += 1
                        else:
                            out[sl] += model(x[sl]).float()*weight
                            hits[sl] += weight
            out = out/hits
    return crop_to(out, shape)


def evaluate(model, subjects, *, device, amp=True, window=None, n_classes=N_CLASSES, gaussian_sigma=None):
    """Mean over subjects of the nanmean Dice over labels 1..35, plus the mean CE (T = 1)."""
    model.eval()
    dices, ces = [], []
    for item in subjects:
        logits = predict_logits(model, item.image, device=device, amp=amp, window=window, gaussian_sigma=gaussian_sigma)
        true = torch.as_tensor(item.labels.astype(np.int64), device=logits.device)[None]
        ces.append(float(F.cross_entropy(logits, true)))
        dices.append(confusion_dice(logits.argmax(1), true, n_classes))
    model.train()
    per_label = nanmean_rows(dices)
    per_subject = [float(np.nanmean(d)) for d in dices]
    return dict(mean_dice35=float(np.mean(per_subject)), per_subject_dice35=per_subject, mean_ce=float(np.mean(ces)),
                per_label_dice=per_label.tolist())


# ---- training -------------------------------------------------------------------------------------

def _lr_at(it, cfg):
    if it < cfg.warmup_iters:
        return cfg.lr*(it+1)/cfg.warmup_iters
    frac = min(1., (it-cfg.warmup_iters)/max(1, cfg.max_iters-cfg.warmup_iters))
    return cfg.lr*(1-frac)**cfg.poly_power


def member_dir(models_root, fold, seed):
    return Path(models_root)/MEMBER_DIR.format(fold=fold, seed=seed)


def train_member(*, files, guard, fold, seed, models_root, config=None, device="cuda", log=print, command=None,
                 subjects=None, inputs_record=None):
    """Train one member; returns the train record. `subjects` = (train, calib) lists of Subject to skip
    reading (tests; such a member has no label-guard record and registry.register refuses it); otherwise
    they are read through the label guard. `inputs_record` = the pins check of cli (pins.py), recorded."""
    if fold not in FOLD_NAMES:
        raise InputValidationError(f"fold must be one of {FOLD_NAMES}")
    cfg = config if isinstance(config, TrainConfig) else TrainConfig(**(config or {}))
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
    label_guard = None
    if subjects is None:
        first_read = len(guard.reads)
        train_set = load_training_subjects(files, guard, fold, "train", percentile=cfg.percentile, log=log)
        calib_set = load_training_subjects(files, guard, fold, "calib", percentile=cfg.percentile, log=log)
        label_guard = dict(guard.record(), label_reads=list(guard.reads[first_read:]))     # this member's reads only
    else:
        train_set, calib_set = subjects
    if not train_set or not calib_set:
        raise InputValidationError("both train and calib subjects are required")
    contact = sorted({s.subject for s in train_set} | {s.subject for s in calib_set})
    model = build_model(cfg.model_config()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=tuple(cfg.betas))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    window = tuple(cfg.patch_size) if cfg.patch_size else None
    log_path = out/"train_log.jsonl"
    best, best_it, bad_evals, history, order, best_metrics = -math.inf, -1, 0, [], [], None
    stopped = "max_iters"
    if cfg.max_iters < 1 or cfg.eval_every < 1:
        raise InputValidationError("max_iters and eval_every must be positive")
    model.train()
    with log_path.open("a", encoding="utf-8") as stream:
        def emit(entry):
            stream.write(json.dumps(entry, default=_json_default)+"\n")
            stream.flush()
        emit(dict(event="start", fold=fold, seed=seed, n_train=len(train_set), n_calib=len(calib_set),
                  parameters=count_parameters(model), config=asdict(cfg), started=started))
        for it in range(cfg.max_iters):
            if not order:
                order = list(rng.permutation(len(train_set)))
            item = train_set[order.pop()]
            for group in opt.param_groups:
                group["lr"] = _lr_at(it, cfg)
            x = torch.as_tensor(item.image, device=device)[None, None]
            y = torch.as_tensor(item.labels.astype(np.int64), device=device)[None]
            x, y, aug = augment(x, y, rng, cfg.augment)
            x, shape = pad_to_multiple(x, model.multiple)
            y, _ = pad_to_multiple(y, model.multiple)
            if window is not None:
                x, y = random_patch(x, y, window, rng)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss, parts = dice_ce_loss(logits, y, n_classes=cfg.n_classes, ce_weight=cfg.ce_weight,
                                       dice_weight=cfg.dice_weight, smooth=cfg.dice_smooth)
            if not torch.isfinite(loss):
                emit(dict(event="nonfinite_loss", it=it, subject=item.subject, aug=aug))
                raise FloatingPointError(f"non-finite loss at iteration {it} ({item.subject}); log kept in {log_path}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            scaler.step(opt)
            scaler.update()
            if it % cfg.log_every == 0:
                emit(dict(event="train", it=it, loss=float(loss.detach()), lr=_lr_at(it, cfg), grad_norm=grad_norm,
                          subject=item.subject, elapsed=time.perf_counter()-wall, **parts))
            last = it == cfg.max_iters-1
            if (it+1) % cfg.eval_every == 0 or last:
                metrics = evaluate(model, calib_set, device=device, amp=cfg.amp, window=window,
                                   n_classes=cfg.n_classes)
                improved = metrics["mean_dice35"] > best or best_metrics is None
                entry = dict(event="eval", it=it+1, improved=improved, elapsed=time.perf_counter()-wall,
                             mean_dice35=metrics["mean_dice35"], mean_ce=metrics["mean_ce"],
                             per_subject_dice35=metrics["per_subject_dice35"])
                if device.type == "cuda":
                    entry["max_memory_gb"] = torch.cuda.max_memory_allocated(device)/1e9
                emit(entry)
                history.append(entry)
                if improved:
                    best, best_it, bad_evals = metrics["mean_dice35"], it+1, 0
                    atomic_torch_save(dict(format="bayesreg.unet35.checkpoint.v1", state_dict=model.state_dict(),
                                           model_config=cfg.model_config(), iteration=it+1,
                                           calib_mean_dice35=best, fold=fold, seed=seed), out/"model_best.pt")
                    best_metrics = metrics
                else:
                    bad_evals += 1
                    if bad_evals >= cfg.patience and it+1 >= cfg.min_iters:
                        stopped = "early_stop"
                        break
        emit(dict(event="end", stopped=stopped, best_iteration=best_it, best_mean_dice35=best,
                  elapsed=time.perf_counter()-wall))
    record = dict(format="bayesreg.unet35.train_record.v1", fold=fold, fold_index=FOLD_INDEX[fold], seed=seed,
                  train_subjects=[s.subject for s in train_set], calib_subjects=[s.subject for s in calib_set],
                  label_contact_subjects=contact, label_contact_subjects_sha256=subject_list_sha256(contact),
                  image_sha256={s.subject: s.image_sha256 for s in train_set+calib_set},
                  label_sha256={s.subject: s.label_sha256 for s in train_set+calib_set},
                  normalization={s.subject: s.normalization for s in train_set+calib_set},
                  config=asdict(cfg), parameters=count_parameters(model), stopped=stopped,
                  iterations_run=history[-1]["it"] if history else 0, best_iteration=best_it,
                  best_calib_mean_dice35=best, best_calib_per_label_dice=best_metrics["per_label_dice"],
                  eval_history=history, model_best_sha256=file_sha256(out/"model_best.pt"),
                  label_guard=label_guard,
                  inputs=inputs_record, environment=environment_record(device), command=command, started=started,
                  wall_seconds=time.perf_counter()-wall)
    write_json(out/"train_record.json", record)
    log(f"fold {fold} seed {seed}: {stopped} after {record['iterations_run']} it, best calib Dice35 {best:.4f} "
        f"at {best_it}, {record['wall_seconds']:.0f} s")
    return record


__all__ = ["MEMBER_DIR", "TrainConfig", "confusion_dice", "dice_ce_loss", "evaluate", "file_sha256", "gaussian_window", "member_dir",
           "nanmean_rows", "predict_logits", "train_member", "write_json"]
