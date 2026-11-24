"""SV, revision 1 (runs/20261104-sv/plan.md, docs/bayesreg/reviews/SV_tools_复核_20261104.md B1): the main rule of the label maps, a mix of the posterior and the truth.  No file, no GPU, no truth reader here: a posterior and a label map come in.

    L_f(x) = argmax_l [(1 - f) p_l(x) + f 1(l = truth(x))]

p: the (36, nz, ny, nx) posterior of unet35_v1_ens, all_folds_mean, calibrated: the same source as the argmax of A5P.

Ties: the smallest label, the rule of np.argmax and so of mc0p_core.predicted_labels.  With a = argmax p and t = truth(x), only two values can win: (1 - f) p_a and (1 - f) p_t + f, because a is the largest of the labels without the bonus (multiplying by 1 - f keeps the order).
Where a = t nothing changes for any f.  Where a != t, with d = p_a - p_t in [0, 1], the voxel takes the truth exactly when f > d / (1 + d): low-confidence errors go first, errors made with p_a near 1 and p_t near 0 go last.
So: f = 0 is the argmax bit for bit; f > 1/2 is the truth bit for bit; f = 1/2 is the truth except where d = 1 exactly (a tie, the smaller of a, t).  A reading arm takes f in (0, 1/2).  L_f never holds a third label.

reduce_posterior keeps what L_f needs, the two label maps and p_a, p_t at the voxels that disagree, so the stack (1.98 GB of float64 at 192 x 224 x 160) can go at once and a sweep over f costs one pass over those voxels.  mix is bit for bit mix_direct, the formula as written (a test).
"""
from dataclasses import dataclass

import numpy as np

import sdfmix

N_CLASSES = sdfmix.N_CLASSES
LABEL_DTYPE = sdfmix.LABEL_DTYPE
TIE_RULE = 'the smallest label, as np.argmax (mc0p_core.predicted_labels)'
MEASURES = ('seg35', 'seg4', 'small_six', 'big_six', 'other_23', 'cortex', 'white_matter')      # the measures whose error 1 - Dice is reduced


def check_f(f):
    """f as a float in [0, 1]."""
    if isinstance(f, bool) or not isinstance(f, (int, float, np.floating)):
        raise TypeError(f'f must be a number, not {f!r}')
    f = float(f)
    if not 0 <= f <= 1:
        raise ValueError(f'f must be in [0, 1], not {f}')
    return f


@dataclass
class Reduced:
    pred: np.ndarray                    # argmax p, int16
    truth: np.ndarray                   # int16
    where: np.ndarray                   # raveled indices of the voxels where they differ
    p_pred: np.ndarray                  # p_a at those voxels
    p_truth: np.ndarray                 # p_t at those voxels


def reduce_posterior(P, truth):
    """What L_f needs from a posterior and a truth label map (any integer-valued storage)."""
    P = np.asarray(P)
    truth = sdfmix.label_map(truth, 'truth')
    if P.ndim != 4 or P.shape != (N_CLASSES,)+truth.shape:
        raise ValueError(f'posterior {P.shape}, truth {truth.shape}: expected ({N_CLASSES}, ...) on the same grid')
    pred = np.argmax(P, axis=0).astype(LABEL_DTYPE)
    where = np.flatnonzero(pred != truth)
    zyx = np.unravel_index(where, truth.shape)
    a, t = pred.ravel()[where], truth.ravel()[where]
    return Reduced(pred, truth, where, np.array(P[(a,)+zyx], dtype=np.float64), np.array(P[(t,)+zyx], dtype=np.float64))


def mix(reduced, f):
    """L_f, int16, a new array."""
    f = check_f(f)
    out = reduced.pred.copy()
    flat = out.reshape(-1)
    value_truth, value_pred = (1-f)*reduced.p_truth+f, (1-f)*reduced.p_pred
    a, t = reduced.pred.ravel()[reduced.where], reduced.truth.ravel()[reduced.where]
    take, tie = value_truth > value_pred, value_truth == value_pred
    flat[reduced.where[take]] = t[take]
    flat[reduced.where[tie]] = np.minimum(a[tie], t[tie])
    return out


def mix_direct(P, truth, f):
    """The formula as written, for the tests: the whole stack."""
    f = check_f(f)
    truth = sdfmix.label_map(truth, 'truth')
    Q = (1-f)*np.asarray(P, dtype=np.float64)
    z, y, x = np.indices(truth.shape, sparse=True)
    Q[truth, z, y, x] += f
    return np.argmax(Q, axis=0).astype(LABEL_DTYPE)


def voxels(labels):
    """Voxels per label, 36 ints."""
    return np.bincount(np.asarray(labels).ravel(), minlength=N_CLASSES).tolist()


def vanished(counts, pred_counts, truth_counts):
    """The labels 1..35 that the prediction or the truth has and this map has not."""
    return [label for label in range(1, N_CLASSES) if counts[label] == 0 and (pred_counts[label] > 0 or truth_counts[label] > 0)]


def ratios(counts, reference):
    """Voxels against a reference, per label; None where the reference has none."""
    return [None if b == 0 else round(a/b, 4) for a, b in zip(counts, reference)]


def error_reduction(rows, base):
    """rows, base: {subject: {measure: Dice}} at some f and at f = 0.  1 - err(f) / err(0), err = 1 - Dice: per subject, and for the mean over the subjects of err (`mean`: the measure the arms are named by).  None where it is undefined."""
    def one(values, zero):
        return None if values is None or zero is None or zero == 0 else 1-values/zero

    def err(row, key):
        return None if row.get(key) is None else 1-row[key]
    per_subject = {s: {key: one(err(rows[s], key), err(base[s], key)) for key in MEASURES} for s in rows}
    mean = {}
    for key in MEASURES:
        a, b = [err(rows[s], key) for s in rows], [err(base[s], key) for s in rows]
        mean[key] = None if any(v is None for v in a+b) else one(float(np.mean(a)), float(np.mean(b)))
    return dict(mean=mean, per_subject=per_subject)
