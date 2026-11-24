"""Evaluation metrics of one displacement field (CLAUDE.md "评测约定"; C16: metrics only from the field).

Definitions (every number is computed here from d and the truth labels; nothing a method reports is used):
  field         d (3, nz, ny, nx) float64 voxels, pull-back warped(p) = moving(p + d(p)); u = h d in mm (fields.py)
  warping       labels warped by scipy map_coordinates(moving, identity + d, order=0), outside -> 0  [warp_labels]
  dice_official Learn2Reg Dice over the primary label set (OASIS 1..35; LPBA40 its 56 ids): per label
                2|A n B| / (|A| + |B|), NaN for a label missing in the fixed or in the UNWARPED moving labels;
                case value = nanmean  [dice_official]; per label: dice_official_labels
  hd95          Learn2Reg HD95 (voxel units): surface_distance.compute_robust_hausdorff(compute_surface_distances(
                fixed == l, warped == l, ones(3)), 95), same NaN rule, nanmean  [hd95_official]
  seg35, seg4   macro Dice over labels 1..35 / 1..4 (jd_curl.three_d.nifti.dice_labels: a label absent from both the
                fixed and the warped labels is skipped; the D0 / VAL10 convention); seg35_per_label, seg4_per_label.
                LPBA40: lpba56 over the manifest's 56 ids and lpba54 without 181, 182 (H2 amendment); both reported
  sdlogj_official  Learn2Reg SDlogJ: G = central differences of d (correlate with [-.5, 0, .5], zero padding),
                det = det(I + G), crop 2 voxels per side, std of log(clip(det + 3, 1e-9, 1e9)) (the "+3" is in the
                official code); num_foldings_official = count(clip(det + 3) <= 0) -- always 0, never evidence of
                no folding  [sdlogj_official]
  central_det_nonpositive  count of det <= 0 over the cropped volume (the same det, before "+3");
                central_det_nonpositive_fraction (= mean); central_det_nonpositive_brain = count over the brain
                voxels of the cropped volume
  brain cells   cells whose lower-corner node lies in the brain mask (OASIS: fixed seg4 > 0; LPBA40: fixed_brain > 0)
  J centre      det(I + grad u) at every cell centre of the Q1 map (grad = mean of the four parallel edge differences
                per axis; copy of eb808be jd_curl/three_d/smoothness.cell_jacobian_centres); log J = log(clip(J, 1e-12))
  sdlogj_brain  std of log J over brain cells; pct_J_lt_half_brain = 100 * fraction J < .5 over brain cells;
                min_J_centre and cell_centre_J_nonpositive over all cells; logj_q01_brain, logj_q99_brain = numpy
                quantile (linear interpolation) .01 / .99 of log J over brain cells; frac_abs_logj_gt_log2_brain =
                fraction of brain cells with |log J| > log 2
  aggregate     over cases (scripts/evaluate_l2r_style.aggregate): mean, std (ddof 0) and .3 quantile of
                dice_official; DSC30 = mean final Dice over the 30% (case, label) entries with the lowest initial
                (d = 0) Dice; hd95 mean and .3 quantile; sdlogj_official mean and .7 quantile
Bitwise identity with scripts/evaluate_l2r_style.py (warp_labels_l2r, dice_l2r, hd95_l2r, sdlogj_l2r, aggregate) and
with the archived smoothness.py is tested in tests/test_bayesreg/eval/test_metrics.py.
"""
from dataclasses import dataclass

import numpy as np
import scipy.ndimage
from scipy.ndimage import map_coordinates

from jd_curl.three_d.nifti import dice_labels

from .fields import d_to_u

SEG35 = tuple(range(1, 36))
SEG4 = tuple(range(1, 5))
LPBA40_EXCLUDED_54 = (181, 182)
DEFINITIONS = "Definitions" + __doc__.split("Definitions", 1)[1].rstrip()


@dataclass(frozen=True)
class LabelScheme:
    """name; the primary label stem (official Dice and HD95); macro-Dice sets {measure: (stem, labels)}; the
    brain mask's stem (nodes with fixed_<stem> > 0)."""
    name: str
    primary: str
    primary_labels: tuple
    macro_sets: tuple
    brain_stem: str

    def stems(self):
        return tuple(dict.fromkeys([self.primary]+[stem for _, stem, _ in self.macro_sets]+[self.brain_stem]))


OASIS = LabelScheme("oasis", "seg35", SEG35, (("seg35", "seg35", SEG35), ("seg4", "seg4", SEG4)), "seg4")


def lpba40_scheme(ids):
    ids = tuple(int(i) for i in ids)
    if len(ids) != 56 or len(set(ids)) != 56 or not all(i > 0 for i in ids):
        raise ValueError("LPBA40: 56 distinct positive label ids required")
    keep = tuple(i for i in ids if i not in LPBA40_EXCLUDED_54)
    if len(keep) != 54:
        raise ValueError("LPBA40: labels 181 and 182 must both be among the 56 ids")
    return LabelScheme("lpba40", "lpba56", ids, (("lpba56", "lpba56", ids), ("lpba54", "lpba56", keep)), "brain")


# ---- Learn2Reg definitions (re-implementations of scripts/evaluate_l2r_style.py, bitwise; tested) -----------------
def warp_labels(moving_seg, d):
    """map_coordinates(moving_seg, identity + d, order=0); outside the volume -> 0."""
    ident = np.meshgrid(*[np.arange(n) for n in moving_seg.shape], indexing="ij")
    return map_coordinates(moving_seg, ident+d, order=0)


def dice_official(fixed, moving, warped, labels=SEG35):
    out = []
    for label in labels:
        if (fixed == label).sum() == 0 or (moving == label).sum() == 0:
            out.append(np.nan)
        else:
            a, b = fixed == label, warped == label
            s = a.sum()+b.sum()
            out.append(2.*np.logical_and(a, b).sum()/s if s else np.nan)
    return float(np.nanmean(out)), out


def hd95_official(fixed, moving, warped, labels=SEG35):
    from surface_distance import compute_robust_hausdorff, compute_surface_distances
    out = []
    for label in labels:
        if (fixed == label).sum() == 0 or (moving == label).sum() == 0:
            out.append(np.nan)
        else:
            out.append(float(compute_robust_hausdorff(compute_surface_distances(fixed == label, warped == label,
                                                                                np.ones(3)), 95.)))
    return float(np.nanmean(out)), out


def central_det(d):
    """det(I + G) on the volume cropped by 2 voxels, G[a, c] = d d_c / d axis_a by central differences with zero
    padding; the same per-element operations as sdlogj_l2r, one entry at a time."""
    k = np.array([-.5, 0., .5])
    J = [[None]*3 for _ in range(3)]
    for axis in range(3):
        w = k.reshape([3 if a == axis else 1 for a in range(3)])
        for c in range(3):
            g = scipy.ndimage.correlate(d[c], w, mode="constant", cval=0.)
            J[axis][c] = (g+(1. if axis == c else 0.))[2:-2, 2:-2, 2:-2]
    return (J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1]) - J[1][0]*(J[0][1]*J[2][2]-J[0][2]*J[2][1])
            + J[2][0]*(J[0][1]*J[1][2]-J[0][2]*J[1][1]))


def sdlogj_official(d, det=None):
    """(SDlogJ, official folding count, fraction det <= 0) exactly as sdlogj_l2r returns them."""
    det = central_det(d) if det is None else det
    jac = np.clip(det+3., 1e-9, 1e9)
    return float(np.log(jac).std()), float((jac <= 0).sum()), float((det <= 0).mean())


def aggregate(rows):
    """scripts/evaluate_l2r_style.aggregate: rows carry dice, hd95, sdlogj, dice_labels, initial_dice_labels
    (the official values of each case)."""
    dice = np.array([r["dice"] for r in rows]); hd = np.array([r["hd95"] for r in rows])
    sd = np.array([r["sdlogj"] for r in rows])
    entries = [(r0, r1) for r in rows for r0, r1 in zip(r["initial_dice_labels"], r["dice_labels"])
               if not (np.isnan(r0) or np.isnan(r1))]
    entries.sort(key=lambda e: e[0])
    k = max(1, int(round(.3*len(entries))))
    return dict(cases=len(rows), dice_mean=float(dice.mean()), dice_std=float(dice.std()),
                dice30_quantile=float(np.quantile(dice, .3)), dsc30_ranking=float(np.mean([e[1] for e in entries[:k]])),
                hd95_mean=float(hd.mean()), hd95_30=float(np.quantile(hd, .3)), sdlogj_mean=float(sd.mean()),
                sdlogj_70=float(np.quantile(sd, .7)))


# ---- cell-centre Jacobian (copy of eb808be jd_curl/three_d/smoothness.cell_jacobian_centres, spacing tuple) --------
def cell_jacobian_centres(u, spacing):
    """det(I + grad u) at every cell centre; u is a (nz, ny, nx, 3) node field, components xyz."""
    u = np.asarray(u, dtype=np.float64)
    hx, hy, hz = spacing

    def d(axis, comp):
        a = u[..., comp]
        if axis == 0:
            diff = (a[:, :, 1:]-a[:, :, :-1])/hx
            return .25*(diff[:-1, :-1]+diff[1:, :-1]+diff[:-1, 1:]+diff[1:, 1:])
        if axis == 1:
            diff = (a[:, 1:, :]-a[:, :-1, :])/hy
            return .25*(diff[:-1, :, :-1]+diff[1:, :, :-1]+diff[:-1, :, 1:]+diff[1:, :, 1:])
        diff = (a[1:, :, :]-a[:-1, :, :])/hz
        return .25*(diff[:, :-1, :-1]+diff[:, 1:, :-1]+diff[:, :-1, 1:]+diff[:, 1:, 1:])
    with np.errstate(over="raise", invalid="raise"):
        F = [[(1. if r == j else 0.)+d(j, r) for j in range(3)] for r in range(3)]
        J = (F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1]) - F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
             + F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]))
    if not np.all(np.isfinite(J)):
        raise FloatingPointError("non-finite cell-centre Jacobian determinant")
    return J


def brain_jacobian_stats(u, spacing, brain_nodes):
    """The cell-centre measures of the module docstring; None for the brain ones when the mask has no cell."""
    J = cell_jacobian_centres(u, spacing)
    with np.errstate(divide="ignore", invalid="ignore"):
        lj = np.log(np.clip(J, 1e-12, None))
    cells = np.asarray(brain_nodes, dtype=bool)[:-1, :-1, :-1]
    out = dict(min_J_centre=float(J.min()), cell_centre_J_nonpositive=int((J <= 0).sum()), n_cells=int(J.size),
               n_brain_cells=int(cells.sum()))
    if not cells.any():
        return dict(out, sdlogj_brain=None, pct_J_lt_half_brain=None, logj_q01_brain=None, logj_q99_brain=None,
                    frac_abs_logj_gt_log2_brain=None)
    lb = lj[cells]
    q01, q99 = np.quantile(lb, [.01, .99])
    return dict(out, sdlogj_brain=float(lb.std()), pct_J_lt_half_brain=float(100*np.mean(J[cells] < .5)),
                logj_q01_brain=float(q01), logj_q99_brain=float(q99),
                frac_abs_logj_gt_log2_brain=float(np.mean(np.abs(lb) > np.log(2.))))


# ---- one case --------------------------------------------------------------------------------------------------------
def _per_label(result):
    return {str(r["label"]): r["dice"] for r in result["per_label"]}


def evaluate_field(d, truth, scheme, spacing_xyz):
    """All metrics of one case. `truth`: {"fixed_<stem>", "moving_<stem>" label arrays for the scheme's stems,
    "brain": node mask}. Returns (scalars, per-label lists, official rows for aggregate)."""
    d = np.asarray(d, dtype=np.float64)
    fixed_p, moving_p = truth["fixed_"+scheme.primary], truth["moving_"+scheme.primary]
    if d.shape != (3,)+fixed_p.shape:
        raise ValueError(f"d {d.shape} does not match the labels {fixed_p.shape}")
    labels = scheme.primary_labels
    warped = {scheme.primary: warp_labels(moving_p, d)}
    initial = warp_labels(moving_p, np.zeros_like(d))
    dice, dice_l = dice_official(fixed_p, moving_p, warped[scheme.primary], labels)
    dice0, dice0_l = dice_official(fixed_p, moving_p, initial, labels)
    hd, hd_l = hd95_official(fixed_p, moving_p, warped[scheme.primary], labels)
    scalars = dict(dice_official=dice, initial_dice_official=dice0, hd95=hd)
    per_label = dict(dice_official_labels=dice_l, initial_dice_official_labels=dice0_l, hd95_labels=hd_l)
    for measure, stem, ids in scheme.macro_sets:
        if stem not in warped:
            warped[stem] = warp_labels(truth["moving_"+stem], d)
        result = dice_labels(truth["fixed_"+stem], warped[stem].astype(np.float64), labels=list(ids))
        scalars[measure] = result["macro_dice"]
        per_label[measure+"_per_label"] = _per_label(result)
    det = central_det(d)
    sd, folds, neg = sdlogj_official(d, det)
    brain = np.asarray(truth["brain"], dtype=bool)
    nonpos = det <= 0
    scalars.update(sdlogj_official=sd, num_foldings_official=folds, central_det_nonpositive_fraction=neg,
                   central_det_nonpositive=int(nonpos.sum()),
                   central_det_nonpositive_brain=int(np.count_nonzero(nonpos & brain[2:-2, 2:-2, 2:-2])))
    scalars.update(brain_jacobian_stats(d_to_u(d, spacing_xyz), tuple(float(h) for h in spacing_xyz), brain))
    official_row = dict(dice=dice, hd95=hd, sdlogj=sd, dice_labels=dice_l, initial_dice_labels=dice0_l)
    return scalars, per_label, official_row


__all__ = ["DEFINITIONS", "LabelScheme", "OASIS", "SEG35", "SEG4", "aggregate", "brain_jacobian_stats",
           "cell_jacobian_centres", "central_det", "dice_official", "evaluate_field", "hd95_official", "lpba40_scheme",
           "sdlogj_official", "warp_labels"]
