"""Expected-Dice decoding of selected structure labels from the ensemble posterior (work package R3b, runs/20261111-r3/任务书.md;
contract revision 10 draft C22.1-8 on branch r3-c22-draft, switch `decode`, default off; the setting judged by runs/20261111-r3/calibration_plan.md section 3).

This module is the interface R3a calls (task brief, "两包的接口").  It is a pure function of the posterior: no file, no label, no
subject id, no randomness.

Input
  probs   the per-voxel ensemble posterior over the 36 classes of probs36 (bayesreg/seg/unet/infer.py: class c = seg35 label c,
          0 = background), channel first: shape (36, *grid), float64 or float32, finite, >= 0, every voxel's channel sum within
          SUM_TOL of 1.  That is the READ-BACK array (bayesreg.seg.channels.read_back_probabilities, or
          bayesreg.seg.unet.infer.read_probabilities(directory, "probs36")), or the caller's own mean of read-back arrays
          renormalized by the channel sum; never the raw float16 storage (refused).
  labels  the seg35 labels to decode, a subset of 1..35 (background 0 is never decoded); default SMALL_SIX = (4, 18, 19, 23, 34, 35),
          the inferior lateral ventricles, the vessels and the choroid plexus (experiments/r2/r2_common.SMALL_SIX).
Output
  an int16 label map of shape grid, values in 0..35.

The rule (derivation and checks: runs/20261111-r3/实现说明_R3b.md, section "算法"; numerical checks
runs/20261111-r3/checks/decode_verify.py and tests/test_bayesreg/seg/test_decode.py):

1. Each decoded label c on its own.  p(x) = probs[c](x) on the whole grid Omega.  Candidates C = {x : p(x) >= candidate_floor},
   ordered by (-p, linear index).  R_tau = the first tau candidates, tau = 0..|C|.  Truth model: the voxels' labels "is c" are
   independent Bernoulli(p(x)), x in Omega (the voxels below the floor included); Gamma = their sum; Dice(R, T) = 2|R n T|/(|R|+|T|),
   with Dice(empty, empty) := 1.
   objective "expected_dice" (default): E(tau) = E[Dice(R_tau, T)], exactly under this model:
     tau = 0:  E(0) = P(Gamma = 0) = prod_x (1 - p(x));
     tau >= 1: E(tau) = sum_{i in R_tau} 2 p_i E[1/(tau + 1 + Gamma_{-i})]
                      = 2 int_0^1 (1-s)^tau Gt(s) sum_{i<=tau} p_i/(1 - p_i s) ds,   Gt(s) = prod_x (1 - p(x) s)
     (1/a = int_0^1 t^(a-1) dt for a >= 1, t = 1 - s; E[t^Gamma_{-i}] = prod_{j != i}(1 - p_j + p_j t)).  The integral by
     Gauss-Legendre (GL_NODES nodes) on the N_SEGMENTS geometric pieces [0, 2^-39], [2^-39, 2^-38], ..., [1/2, 1]; the voxels
     below the floor enter log Gt through their power sums, log Gt_tail(s) = -sum_{k=1..TAIL_TERMS} s^k m_k/k (remainder
     <= m_1 floor^K/((K+1)(1-floor))).
   objective "ratio" (description only): f(tau) = 2 sum_{i<=tau} p_i/(tau + sum_Omega p), f(0) = 0 (1 when sum_Omega p = 0); its
     optimum keeps the voxels with p > f*/2 (Lipton et al. 2014: the F1-optimal threshold is half the optimal F1).
   tau* = the smallest tau with the largest value (as computed: an exact tie in the mathematics is decided by the rounding of the
   quadrature); R_c = R_{tau*}.  For a fixed tau, the top-tau set is optimal over all sets of tau voxels: exchanging i in R for j not
   in R changes E by 2 (p_j - p_i) E[1/(tau + 1 + Gamma_{-ij})]; so R_c is the optimum over all subsets of C.  When f*/2 (ratio) or
   the expected-Dice threshold lies below the floor, R_c is all of C (the floor bounds the region, not the objective).
   Convention: E(0) counts Dice(empty, empty) as 1; the evaluation (experiments/unv2/eval_seg.label_dice) leaves such a label out
   of the nanmean.  It matters only where P(Gamma = 0) is not negligible (sum_Omega p of order 1); the small six have sums of
   order 10 to 10^3.
2. Several decoded labels on one voxel: A(x) = {c : x in R_c}.  A(x) not empty: the c in A(x) with the largest p_c(x), ties to the
   smallest label.  A(x) empty: the argmax over the labels NOT decoded (background included), ties to the smallest label.  So a label
   that is not decoded keeps its argmax voxels except those a decoded label takes, and receives the voxels a decoded label gives
   back.  One pass, no iteration.
With labels = () the output is the argmax bit for bit; a one-hot posterior is returned unchanged.

Signature (fixed for R3a):
  decode(probs, labels=SMALL_SIX, *, objective="expected_dice", candidate_floor=CANDIDATE_FLOOR) -> np.ndarray (int16)
  decode_with_record(probs, labels=SMALL_SIX, *, objective=..., candidate_floor=...) -> (np.ndarray, dict)
  decoder_spec(labels=SMALL_SIX, *, objective=..., candidate_floor=...) -> dict    the provenance block (no data)
  argmax_labels(probs) -> np.ndarray (int16)                                          the reference argmax
"""
import numpy as np

from jd_curl.validation import InputValidationError

DECODER_ID = "dice_decode_v1"
N_CLASSES = 36
SMALL_SIX = (4, 18, 19, 23, 34, 35)
OBJECTIVES = ("expected_dice", "ratio")
CANDIDATE_FLOOR = 1e-3
SUM_TOL = 1e-6
STUB = False
GL_NODES = 16                   # Gauss-Legendre nodes per piece
N_SEGMENTS = 40                 # pieces [0, 2^-39], [2^-39, 2^-38], ..., [1/2, 1] of s = 1 - t
TAIL_TERMS = 6                  # power sums of the voxels below the floor in log Gt
ROW_CHUNK = 2048                # candidates per block in the curve (memory ROW_CHUNK x GL_NODES*N_SEGMENTS doubles)


def _check_labels(labels):
    import numbers
    if isinstance(labels, (str, bytes)) or not hasattr(labels, "__iter__"):
        raise InputValidationError(f"decode: labels must be an iterable of integer labels, got {labels!r}")
    labels = tuple(labels)
    if any(isinstance(c, (bool, np.bool_)) or not isinstance(c, numbers.Integral) for c in labels):
        raise InputValidationError(f"decode: labels must be integers (not bool, float or str), got {labels!r}")
    labels = tuple(int(c) for c in labels)
    if len(set(labels)) != len(labels):
        raise InputValidationError(f"decode: labels {labels} repeat a label")
    bad = [c for c in labels if not 1 <= c <= N_CLASSES-1]
    if bad:
        raise InputValidationError(f"decode: labels {bad} outside 1..{N_CLASSES-1} (background 0 is never decoded)")
    return tuple(sorted(labels))


def _check_options(objective, candidate_floor):
    if objective not in OBJECTIVES:
        raise InputValidationError(f"decode: objective {objective!r} not in {OBJECTIVES}")
    floor = float(candidate_floor)
    if not 0 < floor <= 1e-2:
        raise InputValidationError(f"decode: candidate_floor {candidate_floor!r} outside (0, 1e-2]")
    return floor


def check_probs(probs):
    """The input checks (InputValidationError): ndarray, (36, *grid) with a non-empty grid, float64 or float32, finite, in [0, 1],
    every voxel's channel sum within SUM_TOL of 1.  (A value above 1 would make log1p(-p) NaN; the read-back divides by the channel
    sum, which is at least every addend, so it never gives one; a caller's own mixture must be renormalized the same way.)"""
    if not isinstance(probs, np.ndarray):
        raise InputValidationError("decode: probs must be a numpy array (the read-back posterior)")
    if probs.ndim < 2 or probs.shape[0] != N_CLASSES or probs[0].size == 0:
        raise InputValidationError(f"decode: probs of shape (36, *grid) required, got {probs.shape}")
    if probs.dtype not in (np.float64, np.float32):
        raise InputValidationError(f"decode: float64 or float32 read-back probabilities required, got {probs.dtype} "
                                   "(raw float16 storage goes through bayesreg.seg.channels.read_back_probabilities first)")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0) or np.any(probs > 1):
        raise InputValidationError("decode: non-finite, negative or above-1 probabilities")
    deviation = float(np.max(np.abs(probs.sum(axis=0, dtype=np.float64)-1)))
    if not deviation <= SUM_TOL:
        raise InputValidationError(f"decode: channel sum differs from 1 by {deviation:.3e} > {SUM_TOL:g} on some voxel "
                                   "(pass the read-back, renormalized posterior)")
    return deviation


def argmax_labels(probs):
    """argmax over the 36 classes, ties to the smallest label, int16 (the method's L, C22.1-4)."""
    return np.argmax(probs, axis=0).astype(np.int16)


def decoder_spec(labels=SMALL_SIX, *, objective="expected_dice", candidate_floor=CANDIDATE_FLOOR):
    """The provenance block of a decoding (no data): what R3a writes next to a decoded label map."""
    labels = _check_labels(labels)
    floor = _check_options(objective, candidate_floor)
    return dict(decoder_id=DECODER_ID, module="bayesreg.seg.decode", labels=list(labels), objective=objective,
                candidate_floor=floor, n_classes=N_CLASSES, stub=STUB,
                rule=("per decoded label: top-tau by (-p, index) among p >= floor, tau maximizing the objective (smallest tau on ties); "
                      "voxels claimed by several decoded labels go to the largest p (smallest label on ties); voxels no decoded label "
                      "claims go to the argmax over the labels not decoded"),
                quadrature=dict(gauss_legendre_nodes=GL_NODES, geometric_pieces=N_SEGMENTS, tail_power_sums=TAIL_TERMS)
                if objective == "expected_dice" else None)


# ---- one label ------------------------------------------------------------------------------------------------------------------------------
def quadrature():
    """(nodes s, weights) on [0, 1]: Gauss-Legendre with GL_NODES nodes on each piece [0, 2^-(N-1)], [2^-(N-1), 2^-(N-2)], ..., [1/2, 1]."""
    x, w = np.polynomial.legendre.leggauss(GL_NODES)
    edges = np.concatenate(([0.], 2.0**np.arange(-(N_SEGMENTS-1), 1)))
    a, b = edges[:-1, None], edges[1:, None]
    return ((b-a)/2*x+(b+a)/2).ravel(), ((b-a)/2*w).ravel()


_S, _W = quadrature()
_LOG1M_S = np.log1p(-_S)


def tail_moments(p_tail, terms=TAIL_TERMS):
    """m_k = sum p^k, k = 1..terms, of the voxels below the floor (float64)."""
    p_tail = np.asarray(p_tail, dtype=np.float64)
    out, power = [], np.ones_like(p_tail)
    for _ in range(terms):
        power = power*p_tail
        out.append(float(power.sum()))
    return out


def _log_g(p, moments):
    """(log Gt(s) at the nodes, log Gt(1) = log P(Gamma = 0)) for the voxels p (exact) and those given by their power sums."""
    log_g = np.zeros_like(_S)
    log_g0 = 0.
    for start in range(0, p.shape[0], ROW_CHUNK):
        block = p[start:start+ROW_CHUNK]
        with np.errstate(divide="ignore"):
            log_g += np.log1p(-np.outer(block, _S)).sum(axis=0)
            log_g0 += float(np.log1p(-block).sum())
    for k, m in enumerate(moments, start=1):
        log_g -= _S**k*m/k
        log_g0 -= m/k
    return log_g, log_g0


def expected_dice_curve(p_sorted, moments=()):
    """E(tau), tau = 0..n, for the candidates p_sorted (non-increasing) and the voxels below the floor given by their power sums
    `moments` (m_1, m_2, ...): the expected Dice of the top-tau set under independent Bernoulli voxels (module docstring, rule 1)."""
    p = np.asarray(p_sorted, dtype=np.float64)
    n = p.shape[0]
    log_g, log_g0 = _log_g(p, moments)
    curve = np.empty(n+1)
    curve[0] = float(np.exp(log_g0))
    running = np.zeros_like(_S)
    for start in range(0, n, ROW_CHUNK):
        block = p[start:start+ROW_CHUNK]
        cumulative = running+np.cumsum(block[:, None]/(1-block[:, None]*_S[None, :]), axis=0)
        taus = np.arange(start+1, start+1+block.shape[0], dtype=np.float64)
        weight = np.exp(taus[:, None]*_LOG1M_S[None, :]+log_g[None, :])*_W[None, :]
        curve[start+1:start+1+block.shape[0]] = 2*np.sum(weight*cumulative, axis=1)
        running = cumulative[-1]
    return curve


def expected_dice_of_set(p, region, *, candidate_floor=CANDIDATE_FLOOR):
    """E[Dice(region, T)] under the same model for ANY set `region` (linear indices into the flat p), e.g. the argmax set of a label
    (description; the decoding itself only needs the top-tau sets).  Same quadrature and the same tail by power sums."""
    p = np.asarray(p, dtype=np.float64).ravel()
    region = np.asarray(region, dtype=np.int64)
    exact = p[p >= candidate_floor]
    log_g, log_g0 = _log_g(exact, tail_moments(p[(p > 0) & (p < candidate_floor)]))
    tau = region.shape[0]
    if tau == 0:
        return float(np.exp(log_g0))
    inside = p[region]
    total = np.zeros_like(_S)
    for start in range(0, tau, ROW_CHUNK):
        block = inside[start:start+ROW_CHUNK]
        total += np.sum(block[:, None]/(1-block[:, None]*_S[None, :]), axis=0)
    return float(2*np.sum(np.exp(tau*_LOG1M_S+log_g)*_W*total))


def ratio_curve(p_sorted, total):
    """f(tau) = 2 sum_{i<=tau} p_i/(tau + total), tau = 0..n; f(0) = 0, or 1 when total = 0 (description only)."""
    p = np.asarray(p_sorted, dtype=np.float64)
    taus = np.arange(p.shape[0]+1, dtype=np.float64)
    head = np.concatenate(([0.], np.cumsum(p)))
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = 2*head/(taus+total)
    curve[0] = 1. if total == 0 else 0.
    return curve


def label_region(p, *, objective="expected_dice", candidate_floor=CANDIDATE_FLOOR):
    """(linear indices of R_c in the decoding order, record, the objective's curve over tau = 0..|C|) for one label's flat posterior p
    (rule 1)."""
    p = np.asarray(p, dtype=np.float64).ravel()
    candidates = np.flatnonzero(p >= candidate_floor)
    order = candidates[np.argsort(-p[candidates], kind="stable")]
    p_sorted = p[order]
    if objective == "expected_dice":
        below = p[(p > 0) & (p < candidate_floor)]
        curve = expected_dice_curve(p_sorted, tail_moments(below))
    else:
        curve = ratio_curve(p_sorted, float(p.sum()))
    if not np.all(np.isfinite(curve)):
        raise InputValidationError("decode: the objective is not finite (a probability outside [0, 1]?)")
    tau = int(np.argmax(curve))
    return order[:tau], dict(n_candidates=int(order.shape[0]), tau=tau, threshold=float(p_sorted[tau-1]) if tau else None,
                             objective_value=float(curve[tau]), expected_volume=float(p.sum())), curve


# ---- the label map --------------------------------------------------------------------------------------------------------------------------
def decode_with_record(probs, labels=SMALL_SIX, *, objective="expected_dice", candidate_floor=CANDIDATE_FLOOR):
    """(label map int16 of shape grid, record); see the module docstring."""
    spec = decoder_spec(labels, objective=objective, candidate_floor=candidate_floor)
    deviation = check_probs(probs)
    labels, floor = tuple(spec["labels"]), spec["candidate_floor"]
    grid = probs.shape[1:]
    flat = probs.reshape(N_CLASSES, -1)
    argmax = argmax_labels(flat)
    out = argmax.copy()
    best_label = np.full(argmax.shape, -1, dtype=np.int16)
    best_p = np.full(argmax.shape, -np.inf)
    claims = np.zeros(argmax.shape, dtype=np.int8)
    per_label = {}
    for c in labels:                                                 # ascending: a strictly larger p wins, so ties stay with the smaller label
        p = flat[c].astype(np.float64)
        region, rec, curve = label_region(p, objective=objective, candidate_floor=floor)
        argmax_volume = int(np.count_nonzero(argmax == c))
        rec.update(argmax_volume=argmax_volume, region_volume=int(region.shape[0]),
                   objective_top_k_at_argmax_volume=float(curve[min(argmax_volume, curve.shape[0]-1)]),
                   expected_dice_argmax_set=(expected_dice_of_set(p, np.flatnonzero(argmax == c), candidate_floor=floor)
                                             if objective == "expected_dice" else None))
        per_label[str(c)] = rec
        claims[region] += 1
        better = region[p[region] > best_p[region]]
        best_label[better] = c
        best_p[better] = p[better]
    claimed = best_label >= 0
    out[claimed] = best_label[claimed]
    given_back = np.flatnonzero(~claimed & np.isin(argmax, labels))
    if given_back.size:
        rest = np.ones(N_CLASSES, dtype=bool)
        rest[list(labels)] = False
        sub = flat[:, given_back].astype(np.float64)
        sub[~rest] = -np.inf
        out[given_back] = np.argmax(sub, axis=0).astype(np.int16)
    for c in labels:
        rec = per_label[str(c)]
        final, was = out == c, argmax == c
        rec.update(final_volume=int(np.count_nonzero(final)),                  # = the voxels of R_c that c won (rule 2)
                   n_lost_to_other_decoded=int(rec["region_volume"]-np.count_nonzero(final)),
                   n_taken=int(np.count_nonzero(final & ~was)), n_given_up=int(np.count_nonzero(was & ~final)))
    changed = out != argmax
    record = dict(spec, input_max_abs_sum_deviation=deviation, grid_shape=list(grid), per_label=per_label,
                  n_voxels_changed=int(np.count_nonzero(changed)), n_voxels_claimed_twice=int(np.count_nonzero(claims > 1)),
                  n_given_back=int(given_back.size))
    return out.reshape(grid), record


def decode(probs, labels=SMALL_SIX, *, objective="expected_dice", candidate_floor=CANDIDATE_FLOOR):
    """The decoded label map (int16, shape grid); see the module docstring."""
    return decode_with_record(probs, labels, objective=objective, candidate_floor=candidate_floor)[0]


__all__ = ["CANDIDATE_FLOOR", "DECODER_ID", "GL_NODES", "N_CLASSES", "N_SEGMENTS", "OBJECTIVES", "SMALL_SIX", "SUM_TOL", "TAIL_TERMS",
           "argmax_labels", "check_probs", "decode", "decode_with_record", "decoder_spec", "expected_dice_curve", "expected_dice_of_set",
           "label_region",
           "quadrature", "ratio_curve", "tail_moments"]
