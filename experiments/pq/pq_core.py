"""PQ (contract C22.Q2, runs/20261107-pq/plan.md sections 2, 3): the numbers of the PQ unit, pure numpy / scipy, no file is read here.

Notation (C22.Q2): Omega the native grid, labels c = 0..35 (0 background), K = 36; P_F, P_M the read-back posteriors (float64, divided by the channel sum); d the voxel displacement of the certified final
state (disp_voxels, pull-back: warped(x) = moving(x + d(x))).

iota      (C22.Q2-1) the source voxel of x, taken with the same call and the same coordinates as the evaluation's warp_labels_l2r (scripts/evaluate_l2r_style.py:35-37): map_coordinates(., ident + d,
          order = 0) of the array of linear voxel numbers + 1 (float64, exact below 2^53), so that the call is warp_labels_l2r itself with its own cval 0; 0 there means outside, iota = -1.
P~_M      (C22.Q2-2) P~_M^c(x) = P_M^c(iota(x)); outside, e_0 (all mass on the background), as warp_labels_l2r gives label 0 outside.
q         (C22.Q2-3) 1 - sum_{c=0}^{35} P_F^c P~_M^c (float64; the background is a label).
D^_c      (C22.Q2-4) 2 S^x_c / (S^F_c + S^M_c), the sums over the whole grid, S^M of the pulled-back posterior (the evaluation counts the warped side on Omega).
L^, L^off (C22.Q2-5) the defined sets: c in 1..35 with #{L_F = c} + #{L~_M = c} > 0 (the nanmean rule of dice_labels, with the predicted labels); the official one #{L_F = c} > 0 and #{L_M = c} > 0.
seg4      (C22.Q2-6) P^(4),g = sum_{c: G(c) = g} P^c, D^(4) and its defined set from G(L_F), G(L~_M); q^(4) = 1 - sum_g P_F^(4),g P~_M^(4),g (description).
H         (C22.Q2-7) the hard-label version: dice_labels(L_F, L~_M) (jd_curl.three_d.nifti, the evaluation's function), H35, H4; it is a baseline, not PQ.
eps-EM    (C22.Q2-8, D89 = (a): described only) on Omega_pred = ({L_F > 0} or {L~_M > 0}) dilated twice by the 6-neighbourhood; the floor 1e-4 and the ceiling .5 written in.
Baselines (plan.md 3.2, scores where larger means more likely wrong): h = 1[L_F != L~_M]; H_F the fixed posterior's entropy (natural log, 0 log 0 = 0); H2 = H_F + H_M o iota (0 outside); V_F, V2 the
          members' variance likewise; G+ = ||grad F|| by np.gradient with the array-axis spacings (s_z, s_y, s_x) = (spacing[2], spacing[1], spacing[0]) (C22.Q2-1 "轴序"), no smoothing; G- = -G+ (description);
          delta = min(delta_{L_F}, delta_{L~_M}), the distance (mm) to the nearest boundary voxel (a face neighbour inside the array with another label; label 0 is a label), B_d = -delta; X_lex the two keys
          (h, H2), never a weighted float.
Every sum is np.sum of a float64 array in channel order c = 0..35 (deterministic); the channels are visited one at a time so that P~_M is never held whole.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, generate_binary_structure

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO/'scripts') not in sys.path:
    sys.path.insert(0, str(REPO/'scripts'))

from evaluate_l2r_style import disp_voxels, warp_labels_l2r  # noqa: E402,F401  (the evaluation's own functions, not copied)

K = 36
LABELS = tuple(range(1, 36))
SEG4_LABELS = (1, 2, 3, 4)
EPS_FLOOR, EPS_CEILING = 1e-4, .5                   # C22.Q2-8: written in, never chosen on data
EPS_INIT, EPS_TOL, EPS_MAX_ITER = .05, 1e-8, 200
DILATION_ITERATIONS = 2                             # Omega_pred: 6-neighbourhood, twice


class PQRefused(Exception):
    """An input or a control of the PQ unit failed; nothing is written for the pair (plan.md section 8: m1-m6 stop before any truth)."""
    status = 'pq_refused'

    def __init__(self, stage, message):
        super().__init__(f'{stage}: {message}')
        self.stage = stage


# ---- the source voxel -------------------------------------------------------------------------------------------------------------------------------
def sample_index(d):
    """iota of C22.Q2-1: int64 (nz, ny, nx), the linear number of the source voxel of x, -1 where warp_labels_l2r gives its outside value.  d: (3, nz, ny, nx) voxel displacement (disp_voxels)."""
    d = np.asarray(d, dtype=np.float64)
    shape = d.shape[1:]
    n = int(np.prod(shape))
    if n >= 2**53:
        raise ValueError('more voxels than float64 numbers exactly')
    numbers = np.arange(1, n+1, dtype=np.float64).reshape(shape)
    picked = warp_labels_l2r(numbers, d)                # the evaluation's call: map_coordinates(numbers, ident + d, order=0), cval 0
    if not np.array_equal(picked, np.rint(picked)) or picked.min() < 0 or picked.max() > n:
        raise ValueError('order-0 sampling returned a value that is not a voxel number')
    return picked.astype(np.int64)-1


def take(volume, iota, outside):
    """volume(iota(x)), `outside` where iota = -1 (C22.Q2-2); the dtype of volume."""
    volume = np.asarray(volume)
    flat = volume.reshape(-1)
    out = flat[np.where(iota >= 0, iota, 0)].reshape(iota.shape)
    out[iota < 0] = outside
    return out


def argmax_smallest(P):
    """argmax over the channel axis, ties to the smallest label, int16 as the method's reader stores L (C22.Q1-1).  The channels one at a time with a strict '>' (the first maximum, np.argmax's rule for
    finite values; a test compares them bit for bit), so that no transposed copy of the (36, ...) float64 stack is made."""
    P = np.asarray(P)
    best = np.array(P[0], copy=True)
    top = np.zeros(P.shape[1:], dtype=np.int16)
    for c in range(1, P.shape[0]):
        better = P[c] > best
        top[better] = c
        np.maximum(best, P[c], out=best)
    return top


# ---- PQ ---------------------------------------------------------------------------------------------------------------------------------------------
def seg4_lookup_array(table):
    """{seg35 label: seg4 label} (configs/bayesreg/seg4_of_seg35_v1.json seg4_of_seg35) -> int array (36,), with 0 -> 0 required."""
    lookup = np.array([int(table[c]) for c in range(K)], dtype=np.int64)
    if lookup[0] != 0 or lookup.min() < 0 or lookup.max() > 4:
        raise ValueError('the seg4 table must send 0 to 0 and every label into 0..4')
    return lookup


def pq_maps(P_F, P_M, iota, lookup):
    """One pass over the 36 channels: q, q^(4), the sums S^x, S^F, S^M (36 and the 5 seg4 groups), argmax P~_M (ties to the smallest label, outside background), and the tail masses.
    P_F, P_M: float64 (36, nz, ny, nx); iota from sample_index; lookup from seg4_lookup_array.  P~_M is built one channel at a time and dropped."""
    P_F, P_M = np.asarray(P_F), np.asarray(P_M)
    if P_F.shape != P_M.shape or P_F.shape[0] != K or P_F.shape[1:] != iota.shape:
        raise ValueError(f'posteriors {P_F.shape}, {P_M.shape} and the index {iota.shape} do not fit')
    shape = iota.shape
    s = np.zeros(shape)
    F4, M4 = np.zeros((5,)+shape), np.zeros((5,)+shape)
    Sx, SF, SM = np.zeros(K), np.zeros(K), np.zeros(K)
    best, top = None, np.zeros(shape, dtype=np.int16)
    for c in range(K):
        pm = take(P_M[c], iota, 1. if c == 0 else 0.)
        product = P_F[c]*pm
        s += product
        Sx[c], SF[c], SM[c] = product.sum(), P_F[c].sum(), pm.sum()
        F4[lookup[c]] += P_F[c]
        M4[lookup[c]] += pm
        if best is None:
            best = pm
        else:
            better = pm > best
            top[better] = c
            np.maximum(best, pm, out=best)
        del pm, product
    q4 = _q4(F4, M4)
    Sx4 = np.array([(F4[g]*M4[g]).sum() for g in range(5)])
    SF4, SM4 = F4.reshape(5, -1).sum(axis=1), M4.reshape(5, -1).sum(axis=1)
    return dict(q=1.-s, q4=q4, Sx=Sx, SF=SF, SM=SM, Sx4=Sx4, SF4=SF4, SM4=SM4, top_tilde=top)


def _q4(F4, M4):
    """1 - sum_{g=0}^{4} F4_g M4_g, the products added in g order (as q's channels are)."""
    s = np.zeros(F4.shape[1:])
    for g in range(F4.shape[0]):
        s += F4[g]*M4[g]
    return 1.-s


def tail_masses(P, L, iota=None):
    """C22.Q2-4 (recorded only): per label c = 0..35, sum_x P^c(x) 1[L(x) != c].  With iota, P is pulled back first (P~_M with L~_M)."""
    out = np.zeros(K)
    for c in range(K):
        p = P[c] if iota is None else take(P[c], iota, 1. if c == 0 else 0.)
        out[c] = (p*(L != c)).sum()
    return out


def dice_ratio(Sx, SF, SM):
    """D^ = 2 S^x / (S^F + S^M); NaN where the denominator is 0 (outside the defined set it is not used)."""
    Sx, SF, SM = (np.asarray(a, dtype=np.float64) for a in (Sx, SF, SM))
    den = SF+SM
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, 2.*Sx/np.where(den > 0, den, 1.), np.nan)


def defined_set(labels_a, labels_b, candidates):
    """[c in candidates : #{a = c} + #{b = c} > 0] (dice_labels' rule: a label in neither image is skipped)."""
    counts = np.bincount(np.asarray(labels_a).ravel().astype(np.int64), minlength=K)+np.bincount(np.asarray(labels_b).ravel().astype(np.int64), minlength=K)
    return [int(c) for c in candidates if counts[c] > 0]


def official_set(L_F, L_M):
    """L^_off (C22.Q2-5): c with #{L_F = c} > 0 and #{L_M = c} > 0, L_M the moving labels NOT pulled back (dice_l2r's NaN rule)."""
    a = np.bincount(np.asarray(L_F).ravel().astype(np.int64), minlength=K)
    b = np.bincount(np.asarray(L_M).ravel().astype(np.int64), minlength=K)
    return [c for c in LABELS if a[c] > 0 and b[c] > 0]


def macro(values, defined):
    """Mean of values[c] over the defined labels; None when there is none."""
    picked = [float(values[c]) for c in defined]
    return float(np.mean(picked)) if picked else None


def hard_dice(L_F, L_tilde, lookup):
    """H (C22.Q2-7): dice_labels(L_F, L~_M) on 1..35 and on G(L_F), G(L~_M) on 1..4; jd_curl.three_d.nifti.dice_labels as the evaluation calls it (float64 label arrays)."""
    from jd_curl.three_d.nifti import dice_labels
    a, b = np.asarray(L_F).astype(np.float64), np.asarray(L_tilde).astype(np.float64)
    h35 = dice_labels(a, b, labels=list(LABELS))
    a4, b4 = lookup[np.asarray(L_F).astype(np.int64)].astype(np.float64), lookup[np.asarray(L_tilde).astype(np.int64)].astype(np.float64)
    h4 = dice_labels(a4, b4, labels=list(SEG4_LABELS))
    return dict(H35=h35['macro_dice'], H4=h4['macro_dice'], H_c={r['label']: r['dice'] for r in h35['per_label']}, H4_g={r['label']: r['dice'] for r in h4['per_label']},
                counts35={r['label']: (r['fixed_count'], r['warped_count']) for r in h35['per_label']}, counts4={r['label']: (r['fixed_count'], r['warped_count']) for r in h4['per_label']})


def pq_numbers(maps, L_F, L_tilde, L_M, lookup):
    """The per-pair numbers of pq.json (C22.Q5) from pq_maps: D^_c, the defined sets, S^35, S^4, the predicted official Dice35, and the hard version beside them."""
    D = dice_ratio(maps['Sx'], maps['SF'], maps['SM'])
    D4 = dice_ratio(maps['Sx4'], maps['SF4'], maps['SM4'])
    defined = defined_set(L_F, L_tilde, LABELS)
    defined4 = defined_set(lookup[np.asarray(L_F).astype(np.int64)], lookup[np.asarray(L_tilde).astype(np.int64)], SEG4_LABELS)
    off = official_set(L_F, L_M)
    for c in defined:
        if not maps['SF'][c]+maps['SM'][c] > 0:
            raise PQRefused('pq', f'label {c} is in the defined set but its denominator is not positive')
    hard = hard_dice(L_F, L_tilde, lookup)
    return dict(D_hat={c: float(D[c]) for c in LABELS}, Sx={c: float(maps['Sx'][c]) for c in range(K)}, SF={c: float(maps['SF'][c]) for c in range(K)}, SM={c: float(maps['SM'][c]) for c in range(K)},
                D4_hat={g: float(D4[g]) for g in SEG4_LABELS}, Sx4={g: float(maps['Sx4'][g]) for g in range(5)}, SF4={g: float(maps['SF4'][g]) for g in range(5)}, SM4={g: float(maps['SM4'][g]) for g in range(5)},
                defined=defined, defined4=defined4, defined_official=off, S35_hat=macro(D, defined), S4_hat=macro(D4, defined4), official_dice35_hat=macro(D, off), **hard)


# ---- eps-EM (D89 = (a): described) ------------------------------------------------------------------------------------------------------------------
def omega_pred(L_F, L_tilde):
    """C22.Q2-8: ({L_F > 0} or {L~_M > 0}) dilated DILATION_ITERATIONS times by the 6-neighbourhood; predictions only."""
    return binary_dilation((np.asarray(L_F) > 0) | (np.asarray(L_tilde) > 0), structure=generate_binary_structure(3, 1), iterations=DILATION_ITERATIONS)


def eps_log_likelihood(eps, s, k=K):
    return float(np.log((1.-eps)*s+eps/k).sum())


def eps_em(s, *, k=K, init=EPS_INIT, tol=EPS_TOL, max_iter=EPS_MAX_ITER, floor=EPS_FLOOR, ceiling=EPS_CEILING):
    """The eps of C22.Q2-8 by EM on s = 1 - q over Omega_pred (the values passed): gamma = (1 - eps) s / ((1 - eps) s + eps / K), eps <- 1 - mean gamma, cut to [floor, ceiling] after each step; stop at
    |delta eps| < tol or max_iter.  Returns eps, the iterations, at_floor / at_ceiling and the log-likelihood after each step (non-decreasing: EM on a concave function, the cut keeps it between the old and
    the new value)."""
    s = np.asarray(s, dtype=np.float64).ravel()
    if s.size == 0:
        return dict(eps=None, iterations=0, at_floor=False, at_ceiling=False, log_likelihood=[], n_voxels=0)
    eps, history, at_floor, at_ceiling = float(init), [eps_log_likelihood(init, s, k)], False, False
    iterations = 0
    for iterations in range(1, max_iter+1):
        gamma = (1.-eps)*s/((1.-eps)*s+eps/k)
        new = 1.-float(gamma.mean())
        at_floor, at_ceiling = new <= floor, new >= ceiling
        new = min(max(new, floor), ceiling)
        history.append(eps_log_likelihood(new, s, k))
        change, eps = abs(new-eps), new
        if change < tol:
            break
    return dict(eps=eps, iterations=iterations, at_floor=bool(at_floor), at_ceiling=bool(at_ceiling), log_likelihood=history, n_voxels=int(s.size))


def em_responsibility_complement(q, eps, k=K):
    """1 - gamma(x) at a fixed eps: (eps/K) / ((1 - eps)(1 - q) + eps/K), a strictly increasing function of q (C22.Q2-8: it orders the voxels as q does)."""
    s = 1.-np.asarray(q, dtype=np.float64)
    return (eps/k)/((1.-eps)*s+eps/k)


# ---- the baselines that need no label (plan.md 3.2) --------------------------------------------------------------------------------------------------
def entropy(P):
    """-sum_c P^c log P^c, natural log, 0 log 0 = 0; float64, channels in order."""
    P = np.asarray(P)
    out = np.zeros(P.shape[1:])
    for c in range(P.shape[0]):
        p = P[c]
        logp = np.zeros_like(p)
        np.log(p, out=logp, where=p > 0)
        out -= p*logp
    return out


def array_spacing(spacing_xyz):
    """(s_z, s_y, s_x) for the (z, y, x) arrays from the xyz spacing of grid.spacing / spacing_xyz (C22.Q2-1 "轴序")."""
    sx, sy, sz = (float(v) for v in spacing_xyz)
    return (sz, sy, sx)


def gradient_norm(F, spacing_xyz):
    """G+ = ||grad F||: np.gradient(F, s_z, s_y, s_x) (central differences inside, one-sided at the faces), float64, no smoothing."""
    gz, gy, gx = np.gradient(np.asarray(F, dtype=np.float64), *array_spacing(spacing_xyz))
    return np.sqrt(gz*gz+gy*gy+gx*gx)


def boundary_voxels(L):
    """d L of plan.md 3.2: x whose 6 face neighbours include one inside the array with another label (label 0 is a label; a neighbour outside the array does not count)."""
    L = np.asarray(L)
    edge = np.zeros(L.shape, dtype=bool)
    for axis in range(L.ndim):
        a = [slice(None)]*L.ndim
        b = [slice(None)]*L.ndim
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        differ = L[tuple(a)] != L[tuple(b)]
        edge[tuple(a)] |= differ
        edge[tuple(b)] |= differ
    return edge


def boundary_distance(L, spacing_xyz):
    """delta_L: distance_transform_edt(~d L, sampling = (s_z, s_y, s_x)), mm, 0 on the boundary voxels.  ValueError when L has no boundary voxel (the distance is not defined)."""
    edge = boundary_voxels(L)
    if not edge.any():
        raise ValueError('the label map has no boundary voxel')
    return distance_transform_edt(~edge, sampling=array_spacing(spacing_xyz))


def baseline_maps(*, L_F, L_tilde, H_F, H_M, V_F, V_M, F, iota, spacing_xyz):
    """The label-free baselines of plan.md 3.2 on the fixed grid, as stored (h uint8, the rest float32 after float64 arithmetic).  H_M, V_M on the moving grid, pulled back with iota (0 outside)."""
    H2 = np.asarray(H_F, dtype=np.float64)+take(np.asarray(H_M, dtype=np.float64), iota, 0.)
    V2 = np.asarray(V_F, dtype=np.float64)+take(np.asarray(V_M, dtype=np.float64), iota, 0.)
    G = gradient_norm(F, spacing_xyz)
    delta = np.minimum(boundary_distance(L_F, spacing_xyz), boundary_distance(L_tilde, spacing_xyz))
    return dict(h=(np.asarray(L_F) != np.asarray(L_tilde)).astype(np.uint8), H_F=np.asarray(H_F, dtype=np.float64).astype(np.float32), H2=H2.astype(np.float32),
                V_F=np.asarray(V_F, dtype=np.float64).astype(np.float32), V2=V2.astype(np.float32), G_plus=G.astype(np.float32), G_minus=(-G).astype(np.float32), delta=delta.astype(np.float32))


BASELINE_KEYS = ('h', 'H_F', 'H2', 'V_F', 'V2', 'G_plus', 'G_minus', 'delta')
