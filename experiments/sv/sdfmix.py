"""SV (docs/bayesreg/notes/提速与换预分割器规划_20261104.md §7; runs/20261104-sv/plan.md): the SDF mix, a label map between a prediction and the truth.  No file, no GPU, no truth reader here: two label maps come in.

Since revision 1 of the plan this is NOT the rule of any arm: only the probe (probe_sv.py) computes it, for comparison; the arms use postmix.py.  The reason (docs/bayesreg/reviews/SV_tools_复核_20261104.md B1): on a thin layer the mix is a staircase, not linear.  On a flat boundary, with prediction and truth D voxels apart, the k-th voxel from the
prediction's side flips at f = k/(D + 1), a tie exactly there.  D = 1 flips only at f = 1/2, D = 2 at f = 1/3 and 2/3 (ties, decided by the label numbers); so f = 1/3 changes almost nothing and f = 2/3 almost everything when the errors are thin layers, and the part in between cannot be reached (tests: the table for D = 1..6).

For each label l, background included, on a grid of 1 mm (exact Euclidean distances between voxel centres, scipy.ndimage.distance_transform_edt):
    SDF_l(x) = dist(x, {L == l}) - dist(x, {L != l})        >= 1 outside the label, <= -1 inside it, so its zero level lies half way between two neighbouring voxels of different labels.
    L_f(x)   = argmin_l [(1 - f) SDF_l^pred(x) + f SDF_l^truth(x)]

The rules, fixed and without a random step:
  weights  f is an exact fraction n/d.  What is compared is (d - n) SDF^pred + n SDF^truth, d times the formula: the same argmin, and integer distances that tie in exact arithmetic tie in the floats too (irrational distances that are equal in exact arithmetic may differ by an ulp; deterministic all the same).  f = 0 reads only the prediction, f = 1 only the truth.
  ties     the smallest label wins, as the argmax of A5P (make_inputs.colour_from_table: "ties: the smallest l").  Labels are visited in ascending order and only a strictly smaller value replaces the one held.
  absent   a label absent from a map that has a positive weight has SDF = +infinity there: it cannot win for that f.  So a label that only the truth has appears only at f = 1, and one that only the prediction has only at f = 0.  Every such label is recorded.
  full     a label that fills the whole grid has no outside: SDF = -(diagonal of the grid), inside by more than any distance on the grid.

Why f = 0 is the prediction and f = 1 the truth, bit for bit: at any x exactly one label has SDF <= -1, its own, and every other label has SDF >= 1 or +infinity.  By the same inequality, where the prediction and the truth agree, L_f(x) is that label for every f.  L_f may also pick a third label where they disagree; the count is recorded.

Memory (shape 192 x 224 x 160, 6 881 280 voxels; arithmetic, measured values in runs/20261104-sv/sdf_bench_20261104.out): one distance_transform_edt call holds the int32 feature transform (82.6 MB), its indices (82.6 MB) and the float64 differences (165 MB), about 330 MB at its peak; two SDFs of one label
55 MB each; each f holds a float64 best value (55 MB) and an int16 label map (14 MB).  The labels are streamed: nothing is kept per label.  With 4 fractions and one worker this is about 0.75 GiB above the input maps; each further worker adds about 0.45 GiB.
Time: 4 distance transforms per label (2 maps, inside and outside), 36 labels: 144 transforms per subject.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
from scipy.ndimage import distance_transform_edt

N_CLASSES = 36
LABELS = tuple(range(N_CLASSES))
LABEL_DTYPE = np.int16
TIE_RULE = 'the smallest label'
ABSENT_RULE = 'a label absent from a map with a positive weight has SDF +infinity there and cannot win'
SAMPLING_MM = 1.


def as_fraction(f):
    """An exact fraction in [0, 1]; a float is refused: 1/3 must not depend on how it was rounded."""
    if isinstance(f, bool) or not isinstance(f, (int, Fraction)):
        raise TypeError(f'f must be an int or a Fraction, not {f!r}')
    f = Fraction(f)
    if not 0 <= f <= 1:
        raise ValueError(f'f must be in [0, 1], not {f}')
    return f


def weights(f):
    """The integer weights (of the prediction, of the truth) of f = n/d: (d - n, n)."""
    f = as_fraction(f)
    return f.denominator-f.numerator, f.numerator


def tag(f):
    """The name of f in file names and provenance ids: f0, f1_3, f2_3, f1."""
    f = as_fraction(f)
    return f'f{f.numerator}' if f.denominator == 1 else f'f{f.numerator}_{f.denominator}'


def label_map(values, name='label map'):
    """The map as int16, refusing a value it cannot keep or a value outside 0..35 (the truth members of pair.npz are stored as floats)."""
    values = np.asarray(values)
    out = values.astype(LABEL_DTYPE)
    if not np.array_equal(out, values):
        raise ValueError(f'{name}: values that are not int16')
    if out.size == 0 or out.min() < 0 or out.max() >= N_CLASSES:
        raise ValueError(f'{name}: values outside 0..{N_CLASSES-1}')
    return out


def signed_distance(region):
    """SDF of a boolean region, float64, or None when the region is empty."""
    region = np.asarray(region, dtype=bool)
    if not region.any():
        return None
    if region.all():
        return np.full(region.shape, -float(np.sqrt(sum(n*n for n in region.shape))))
    return distance_transform_edt(~region)-distance_transform_edt(region)


def _both(pred, truth, label, need_pred, need_truth):
    return (signed_distance(pred == label) if need_pred else None, signed_distance(truth == label) if need_truth else None)


def sdfmix(pred, truth, fractions, *, workers=1):
    """({f: L_f int16}, record).  pred, truth: label maps of the same shape.  fractions: exact fractions.  workers: threads over the labels; the values do not depend on it, the merge is in label order."""
    pred, truth = label_map(pred, 'prediction'), label_map(truth, 'truth')
    if pred.shape != truth.shape:
        raise ValueError(f'prediction {pred.shape} and truth {truth.shape} differ')
    fractions = [as_fraction(f) for f in fractions]
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError(f'fractions {fractions}: at least one, none twice')
    pairs = [weights(f) for f in fractions]
    need_pred, need_truth = any(a for a, _ in pairs), any(b for _, b in pairs)
    best = [np.full(pred.shape, np.inf) for _ in fractions]
    out = [np.zeros(pred.shape, dtype=LABEL_DTYPE) for _ in fractions]
    absent = dict(prediction=[], truth=[])

    def merge(label, a, b):
        if a is None and need_pred:
            absent['prediction'].append(label)
        if b is None and need_truth:
            absent['truth'].append(label)
        for k, (wa, wb) in enumerate(pairs):
            if (wa and a is None) or (wb and b is None):
                continue
            value = a if wb == 0 else b if wa == 0 else wa*a+wb*b     # zero weights never touch an array: f = 0 and f = 1 read one map
            better = value < best[k]
            best[k][better] = value[better]
            out[k][better] = label

    if workers <= 1:
        for label in LABELS:
            merge(label, *_both(pred, truth, label, need_pred, need_truth))
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='sv-sdf') as pool:
            for start in range(0, N_CLASSES, workers):      # a block at a time: at most `workers` labels' fields are in memory
                chunk = LABELS[start:start+workers]
                for label, (a, b) in zip(chunk, pool.map(lambda l: _both(pred, truth, l, need_pred, need_truth), chunk)):
                    merge(label, a, b)
    for f, values in zip(fractions, best):
        if not np.isfinite(values).all():
            raise ValueError(f'f = {f}: voxels without a label: no label is present in both maps')
    result = dict(zip(fractions, out))
    return result, dict(tie_rule=TIE_RULE, absent_rule=ABSENT_RULE, sampling_mm=SAMPLING_MM, absent_labels=absent, fractions=[str(f) for f in fractions], shape_zyx=list(pred.shape),
                        per_fraction={str(f): movement(labels, pred, truth) for f, labels in result.items()})


def movement(labels, pred, truth):
    """Voxel counts of one L_f: where the two maps disagree, how much of it went to the truth or stayed with the prediction, and what took a third label; agreeing voxels that changed (must be 0)."""
    disagree = pred != truth
    return dict(voxels=int(labels.size), disagree=int(np.count_nonzero(disagree)), equal_prediction=int(np.count_nonzero(disagree & (labels == pred))), equal_truth=int(np.count_nonzero(disagree & (labels == truth))),
                third_label=int(np.count_nonzero(disagree & (labels != pred) & (labels != truth))), agreeing_voxels_changed=int(np.count_nonzero(~disagree & (labels != pred))))


def synthetic_labels(shape, seed, *, shift=0.):
    """A label map of the kind the real ones are, for tests and the timing only: an ellipsoid cut into the 35 Voronoi cells of random seed points, background outside.  shift moves the seed points: a second map of the same seed with shift > 0 disagrees near the boundaries."""
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(.2, .8, (N_CLASSES-1, 3))*np.asarray(shape)+rng.normal(size=(N_CLASSES-1, 3))*shift
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    centre = (np.asarray(shape)-1)/2
    inside = ((z-centre[0])/(.42*shape[0]))**2+((y-centre[1])/(.42*shape[1]))**2+((x-centre[2])/(.42*shape[2]))**2 <= 1
    best, labels = np.full(tuple(shape), np.inf), np.zeros(tuple(shape), dtype=LABEL_DTYPE)
    for k, (a, b, c) in enumerate(seeds):
        d = (z-a)**2+(y-b)**2+(x-c)**2
        closer = d < best
        best[closer], labels[closer] = d[closer], k+1
    labels[~inside] = 0
    return labels
