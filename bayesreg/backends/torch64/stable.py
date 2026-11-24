"""GPU-path stable comparison of the torch FP64 backend (contract C08-8, C08-9; work package SG).

The CPU reference (C08-1..7, `bayesreg.model.stable_reduction`) stays the authority and the fallback; this module
implements the GPU path of C08-8 and the bookkeeping the solver needs around it.

(a) Elementwise.  For the three Gauss-point blocks of F_B the device computes, per Gauss point and in FP64,
    diffusion and curl: (r_t - r_b)(r_t + r_b) of the weighted residuals (C08-1, never E_t - E_b);
    log-volume: w_q l_q (log J_t + log J_b), w_q = omega w_g/(omega0 tau_q^2), l_q = log1p(x) for
    x = (J_t - J_b)/J_b >= -1/2, else log(J_t/J_b) (C08-2).
    The channel blocks are taken on the host from the downloaded channel residuals of both device evaluations,
    exactly as C08-1 (`bayesreg.model.energy._node_change_sum`, longdouble elementwise and summed).
    OPT1 C2a (bitwise neutral): J and log J of each state come from the J / log J its `evaluate` computed (the
    evaluation's Jacobian cache, `_compared_geometry`); only G = grad u_h is recomputed; a chunk without an entry is
    recomputed in full as before.
(b) Partial sums.  Per cell (fixed structure, no atomics): diffusion torch.sum over (r, j) (9 terms) then over the
    27 points; curl d_0 + d_1 + d_2 then over the 27 points; log-volume over the 27 points.  Per tile of
    state.tile_size cells: zero padding + reshape + torch.sum over the cells.  The (n_tiles, 3) FP64 tile sums are
    downloaded and accumulated in longdouble in ascending tile order; Delta F_B = longdouble sum of the halves in
    block order (channels, diffusion, logvol, curl) as in the reference; A_hat = float(-Delta).
    Worst-case serial depth of one tile sum (any summation order inside each torch.sum):
        m = max(8 + 26, 2 + 26, 26) + (tile_size - 1) = 8225 for tile_size 8192          (`reduction_depth`)
    C08-8(c)(ii) requires m <= 1.4e4; larger tiles are refused.
    Cell corners (C05-2a, logvol_points = "gauss27+corners"): the corner part of the log-volume block in the same
    log1p form, w_{K,v} = beta_c omega/(8 omega0 tau_{K,v}^2), summed over the 8 corners per cell into a fourth
    column of the tile sums; the host adds each tile's corner sum right after the tile's Gauss log-volume sum (the
    reference's order); a device J <= 0 or nonfinite value at a corner is a device anomaly like at a Gauss point.
    Per-cell depth 7 < 34, so the depth bound m is unchanged.
    Closed27 points (C05-2b, logvol_points = "gauss27+closed27"): the same with the 27 closed27 points per cell in the
    corners' place: w_{K,p} = omega s_p theta_S/(omega0 tau_{K,p}^2) (tau^2 the Q1 interpolation), summed per cell
    into the fourth column, added on the host right after the tile's Gauss log-volume sum; the Gauss log-volume
    weight is omega w_g theta_G (dstate.logvol_gauss_weights).  Per-cell depth 26 (as the Gauss log-volume sum),
    so m is unchanged (8225 for tile_size 8192 <= 1.4e4, C08-8(c)(ii)); the depth and the closed27 column are in
    `reduction_provenance`.  OPT1 C2a: the closed27 J come from the evaluation's cache ("closed27", chunk).
(c) Guard s_hat = 4096 eps64 max(1, E_b, E_t), certified = A_hat - s_hat (an engineering guard, not a rigorous
    error envelope; premise (i) is checked by the C08-9 gate, experiments/sg/l1_gate.py).
(d) Fallback (one rule for every comparison): r = c1 tau |g.p| (PG), 1e-4 pred (LM), 0 (outer E and F);
    |A_hat| <= 8 s_hat or |A_hat - r| <= 8 s_hat -> this comparison is redone on the CPU reference path (both states
    re-evaluated by bayesreg.model, C08-1..7, guard s_E = 16 eps max(E_b, E_t)); otherwise the GPU decision is the
    decision.  CPU evaluations are cached under the key
        (state_hash of u, ModelState identity, energy type "F_B" | "E_v", sha256 of the reference v or None)
    where the ModelState identity hashes the level grid, omega0, the tile size, sigma^2, w, eta = (alpha, nu, tau^2)
    and the fixed-side arrays (`model_identity`); a hit is used only for the very same ModelState object.  Two
    stores (CPUPAR): the base slot keeps the CPU evaluation of the base of the latest CPU comparison (and of the
    latest GF recheck point), whatever the trials in between, so a fallback re-evaluates only the candidate while
    the key stays the same; an LRU of `cache_size` entries keeps the other points.  The CPU Reduction of the latest
    fallback is memoized under the pair of keys and the identity of the two CPU Evaluations (held by weak
    references), so a second fallback on the same pair under the same key (e.g. PG and outer F_B on one candidate)
    does not recompute C08-1..7.  Any key component that changes means a recomputation.  The geometry of the base
    inside stable_reduction is recomputed per tile in every case (its per-Gauss-point residuals of a native volume
    do not fit in memory); the CPU reference runs in ModelState.cpu_threads threads (bitwise neutral).
(e) Device anomalies: a device evaluation that fails (any InvalidTrial, FloatingPointError, OOM or other
    RuntimeError) is recomputed on the CPU and the CPU result decides (`Torch64Backend.evaluate`); in the
    comparison a device J_q <= 0, a nonfinite value, an OOM or another device exception sends the comparison to the
    CPU path; the rejection reason is the CPU's (a trial the CPU cannot evaluate is rejected with the CPU's
    InvalidTrial reason; a base the CPU cannot evaluate raises FloatingPointError, as the reference would).
(f) GF_B: computed from the device gradient; when T_B/1.01 <= GF^GPU <= 1.01 T_B the CPU gradient decides
    (`Torch64Backend.confirm_gradient_norm`); G0 and the native level's final GF_B are CPU values (solver wiring).
(g) The inner energy (E_v under option (b)) and the outer F_B use their own s_hat: the guard is always formed from
    the two energies of the comparison at hand.
(h) Every comparison returns a record (A_hat, s_hat, r, path, fallback and its reason, the CPU values when computed,
    seconds, GF/T_B of the base); `ComparisonStats.summary()` gives counts, fallback ratio, reasons, per-context and
    per-GF_B/T_B-bin counts and the time spent on each path.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import time
import weakref

import numpy as np
import torch

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError

from ... import model as _model
from ...model import ModelState
from ...model.energy import GUARD_FACTOR, _node_change_sum
from .model import (TorchEvaluation, TorchModelState, _extra_geometry, _extra_scale, _extra_tau2, _extra_weights,
                    _geometry, _logvol_scale, _tile_reduce)
from .q1 import curl_of

LD = np.longdouble
EPS = float(np.finfo(np.float64).eps)
GPU_GUARD_FACTOR = 4096           # C08-8(c)
BAND_FACTOR = 8                   # C08-8(d)
NEAR_BAND_FACTOR = 32             # C08-9: near out-of-band candidates 8 s_hat < |A_hat - r| <= 32 s_hat
DEPTH_LIMIT = 14000               # C08-8(c)(ii): m <= 1.4e4
PREMISE_LIMIT = 4e-12             # C08-8(c)(i), C08-9
GF_BAND = 1.01                    # C08-8(f)
LM_RATIO = 1e-4                   # C09.3-11
CONTEXTS = ("pg", "lm", "outer_E", "outer_F", "unspecified")
ENERGIES = ("F_B", "E_v")
CELL_DEPTH = dict(diffusion=8+26, logvol=26, curl=2+26)
GF_BINS = (1.01, 2., 10., 1e2, 1e3, 1e4, 1e6)
GAUSS_NAMES = ("diffusion", "logvol", "curl")


class DeviceAnomaly(RuntimeError):
    """A device-side result that C08-8(e) sends to the CPU reference path."""


# ----------------------------------------------------------------------------------------------
# Guard, required decrease, fallback rule (C08-8(c)(d))
# ----------------------------------------------------------------------------------------------

def gpu_guard(E_b, E_t):
    """C08-8(c): s_hat = 4096 eps64 max(1, E_b, E_t)."""
    return float(GPU_GUARD_FACTOR*EPS*max(1., float(E_b), float(E_t)))


def required_decrease(context, *, c1=None, tau=None, slope=None, pred=None):
    """C08-8(d) r of one comparison: PG c1 tau |g.p| (C09.2-7, the expression of run_pg), LM 1e-4 pred
    (C09.3-11), outer E and F 0 (C10-4)."""
    if context == "pg":
        return c1*tau*abs(slope)
    if context == "lm":
        return LM_RATIO*pred
    if context in ("outer_E", "outer_F"):
        return 0.
    raise InputValidationError(f"context: one of pg, lm, outer_E, outer_F (got {context!r})")


@dataclass(frozen=True)
class FallbackDecision:
    fallback: bool
    reason: str | None            # None, "near_zero", "near_required", "near_zero+near_required", "nonfinite"


def fallback_rule(A_hat, s_hat, required):
    """C08-8(d): fall back to the CPU reference iff |A_hat| <= 8 s_hat or |A_hat - r| <= 8 s_hat (band edges
    included); a nonfinite input falls back too."""
    values = (A_hat, s_hat, required)
    if not all(np.isfinite(v) for v in values) or s_hat <= 0 or required < 0:
        return FallbackDecision(True, "nonfinite")
    band = BAND_FACTOR*s_hat
    near_zero = abs(A_hat) <= band
    near_required = abs(A_hat-required) <= band
    if near_zero and near_required:
        return FallbackDecision(True, "near_zero+near_required")
    if near_zero:
        return FallbackDecision(True, "near_zero")
    if near_required:
        return FallbackDecision(True, "near_required")
    return FallbackDecision(False, None)


def reduction_depth(tile_size):
    """Worst-case serial depth m of one device tile sum (module docstring (b)); refuses m > 1.4e4 (C08-8(c))."""
    if isinstance(tile_size, bool) or not isinstance(tile_size, (int, np.integer)) or tile_size < 1:
        raise InputValidationError("tile_size: positive integer required")
    m = max(CELL_DEPTH.values())+int(tile_size)-1
    if m > DEPTH_LIMIT:
        raise InputValidationError(f"C08-8(c)(ii): reduction depth {m} > {DEPTH_LIMIT} for tile_size {tile_size}")
    return m


def reduction_provenance(tile_size, logvol_points=_model.LOGVOL_DEFAULT):
    per_cell = dict(diffusion="torch.sum over (r, j) [9 terms], then torch.sum over 27 points",
                    curl="d0 + d1 + d2, then torch.sum over 27 points", logvol="torch.sum over 27 points")
    if logvol_points == "gauss27+closed27":
        per_cell["logvol_closed27"] = ("C05-2b: torch.sum over the 27 closed27 points (depth 26), weights omega s_p "
                                       "theta_S/(omega0 tau^2); its tile sums added in longdouble right after the "
                                       "tile's Gauss log-volume sum (weights omega w_g theta_G); depth bound m "
                                       "unchanged")
    elif logvol_points != "gauss27":
        per_cell["logvol_corners"] = ("C05-2a: torch.sum over the 8 corners; its tile sums added in longdouble right "
                                      "after the tile's Gauss log-volume sum")
    return dict(contract="C08-8", path="GPU FP64 elementwise (t-b)(t+b) / log1p form; per-cell then per-tile device "
                "sums; host longdouble over tiles in ascending order",
                per_cell=per_cell,
                per_tile=f"zero-pad, reshape (n_tiles, {int(tile_size)}), torch.sum over cells; no atomics",
                channel="host longdouble, C08-1 (bayesreg.model.energy._node_change_sum)",
                depth_bound=reduction_depth(tile_size), depth_limit=DEPTH_LIMIT, tile_size=int(tile_size),
                guard=f"{GPU_GUARD_FACTOR} eps64 max(1, E_b, E_t)", band=f"{BAND_FACTOR} s_hat",
                fallback="|A_hat| <= 8 s_hat or |A_hat - r| <= 8 s_hat, or a device anomaly -> CPU reference "
                "(C08-1..7)", cpu_guard=f"{GUARD_FACTOR} eps64 max(E_b, E_t)",
                gf_recheck=f"T_B/{GF_BAND} <= GF^GPU <= {GF_BAND} T_B -> CPU gradient")


# ----------------------------------------------------------------------------------------------
# Identity and hashing (C08-8(d) cache key)
# ----------------------------------------------------------------------------------------------

def array_sha256(array):
    """sha256 of the C-order bytes (= array.tobytes(order="C")); MEM: hashed through a buffer view, no copy of a
    C-contiguous array (model_identity hashes the (C, N) fixed stack: 771 MB at the native size)."""
    array = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(memoryview(array.reshape(-1).view(np.uint8))).hexdigest()


def state_hash(u):
    """sha256 of the C-order float64 bytes (C07-7; the digest of certify_map)."""
    return array_sha256(np.asarray(u, dtype=np.float64))


def model_identity(state):
    """The ModelState identity of the cache key: level grid, omega0, tile size, sigma^2, w, eta and the
    fixed-side arrays (fixed channels, mask, tissue simplex) plus the moving-side sampler ids; with the C05-2a corner
    term also the log-volume point set and beta_c, with the C05-2b closed27 part the point set, beta_S and the closed27
    rule (the "gauss27" identity -- the default before DEF, an explicit option since -- is that of the code
    before); under sigma_domain support / support_shared (C04-6, C08-8(d')) also sigma_domain, theta and the sha256 of
    the frozen support weights v (the brain identity is that of the code before SWLI)."""
    if not isinstance(state, ModelState):
        raise InputValidationError("state: ModelState required")
    digest = hashlib.sha256()
    parts = [state.grid.grid_id, repr(float(state.omega0)), repr(int(state.tile_size)),
             repr(float(state.alpha)), repr(float(state.nu)), array_sha256(state.sigma2), array_sha256(state.tau2),
             "gaussian" if state.weights is None else array_sha256(state.weights), array_sha256(state.fixed),
             array_sha256(state.mask), array_sha256(state.tissue), *[s.image_id for s in state.samplers]]
    if state.logvol_corners:
        parts += [state.logvol_points, repr(float(state.corner_weight))]
    elif state.logvol_closed27:
        parts += [state.logvol_points, repr(float(state.logvol_closed27_share)), state.closed27_weights]
    if state.has_support:                       # C08-8(d'): sigma_domain, theta and the sha256 of v (brain: unchanged)
        parts += [state.sigma_domain, repr(float(state.support_threshold)), state.support_v_sha256]
    for part in parts:
        digest.update(part.encode())
        digest.update(b"|")
    return digest.hexdigest()


# ----------------------------------------------------------------------------------------------
# Host-side evaluation handed to the solver
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class HostEvaluation:
    """F_B at one u as the solver sees it: u is the host read-only (nz, ny, nx, 3) array; `device` the
    TorchEvaluation (None when the device evaluation failed and `cpu`, the reference Evaluation, replaced it)."""
    state: ModelState
    u: np.ndarray
    total: np.longdouble
    energy: float
    energies: dict
    device: TorchEvaluation | None = None
    cpu: object = None
    anomaly: str | None = None
    _memo: dict = field(default_factory=dict, repr=False)

    @property
    def state_hash(self):
        if "hash" not in self._memo:
            self._memo["hash"] = state_hash(self.u)
        return self._memo["hash"]

    @property
    def path(self):
        return "device" if self.device is not None else "cpu"

    @classmethod
    def from_device(cls, state, u, tev):
        return cls(state, u, tev.total, tev.energy, dict(tev.energies), device=tev)

    @classmethod
    def from_cpu(cls, state, ev, anomaly):
        return cls(state, ev.u, ev.total, ev.energy, dict(ev.energies), cpu=ev, anomaly=anomaly)


@dataclass(frozen=True)
class ExtraBlocks:
    """Blocks of the compared energy beyond F_B, computed exactly on the host (the proximal blocks of E_v):
    `changes` the longdouble 1/2 sum (t - b)(t + b) per block in their fixed order, `base_total` /
    `trial_total` the longdouble block totals of the two states."""
    changes: dict
    base_total: np.longdouble
    trial_total: np.longdouble


@dataclass(frozen=True)
class GuardedReduction:
    """The result of one comparison with the interface of bayesreg.model.Reduction (ared, guard, certified,
    reason, delta, blocks) plus the C08-8(h) record.  `rejection` overrides `reason` when the CPU could not
    evaluate the trial (C08-8(e)): then ared, guard and certified are None."""
    ared: float | None
    guard: float | None
    delta: np.longdouble | None
    blocks: dict
    path: str                       # "gpu" or "cpu"
    info: dict
    rejection: str | None = None

    @property
    def certified(self):
        return None if self.ared is None else self.ared-self.guard

    @property
    def reason(self):
        if self.rejection is not None:
            return self.rejection
        if abs(self.ared) <= self.guard:
            return "ENERGY_RESOLUTION"
        if self.certified <= 0:
            return "NO_STRICT_DECREASE"
        return None

    def record(self):
        return dict(self.info, path=self.path, ared=self.ared, guard=self.guard,
                    certified=self.certified, reason=self.reason)


# ----------------------------------------------------------------------------------------------
# (a)(b) device elementwise changes and tile partial sums
# ----------------------------------------------------------------------------------------------

def _download(dstate, tensor):
    return dstate.device.download(tensor)


def _compared_geometry(dstate, ev, chunk, number, corners=False):
    """(G, A, J, log J) of one compared state on one chunk (corners=True: at the points of the second log-volume part,
    the cell corners of C05-2a or the closed27 points of C05-2b, cache key (dstate.logvol_extra, number)).
    OPT1 C2a: with a Jacobian-cache entry, J and log J are the tensors `evaluate` computed for this very u_cm and
    chunk, G = grad u_h is recomputed (gather + DGEMM, the first step of `_geometry`) and A is None (not recomputed);
    log J is torch.log(J) when only J was kept.  Without an entry the full geometry is recomputed as before OPT1
    (`_geometry`: G, F, A = cof F, J) and log J = torch.log(J).  Either way G, J and log J are bitwise those of the
    code before OPT1 (same functions on the same tensors)."""
    entry = ev.jacobian_entry((dstate.logvol_extra, number) if corners else number)
    if entry is None:
        G, _, A, J = _geometry(dstate, ev.u_cm, chunk, dstate.q1_extra if corners else None)
        return G, A, J, torch.log(J)
    J, logJ = entry
    G = (dstate.q1_extra if corners else dstate.q1).gradient(ev.u_cm, chunk)
    return G, None, J, torch.log(J) if logJ is None else logJ


def device_gauss_changes(dstate, base, trial):
    """C08-8(a)(b): (n_tiles, 3) host array of the device FP64 tile sums of the elementwise changes of the
    diffusion, log-volume and curl blocks (columns in that order; under C05-2a a fourth column, the corner part of
    the log-volume block).  Raises DeviceAnomaly on a device J <= 0 (Gauss point or corner), a nonfinite geometry
    or tile sum, or a device exception (OOM included).
    OPT1 C2a: J and log J of each state come from its evaluation's Jacobian cache when present (`_compared_geometry`);
    only G is recomputed.  Nonfinite check: G (recomputed), J (cached or recomputed) and, where recomputed, A.  A
    cached chunk's A is not recomputed; its finiteness, with that of G and J and J > 0 at every point of every chunk,
    was established by `evaluate` on the same tensors (a TorchEvaluation exists only when its per-tile nonfinite and
    J <= 0 counts are all zero).  G finite does not imply A finite in floating point (A has products of two entries
    of F = I + G, which overflow for |G| > ~1.3e154), which is why the check of A rests on evaluate and is not
    derived from G (tests/test_bayesreg/torch64/test_opt1_bitwise.py, overflow case).  Uses mark the two evaluations
    as recently used in the Jacobian-cache slots (trial first, then base)."""
    if not isinstance(dstate, TorchModelState):
        raise InputValidationError("dstate: TorchModelState required")
    for ev in (base, trial):
        if not isinstance(ev, TorchEvaluation) or ev.state is not dstate:
            raise InputValidationError("C08-8: two TorchEvaluations of this TorchModelState required")
    reduction_depth(dstate.state.tile_size)
    corners = bool(dstate.logvol_extra)                                 # C05-2a corners or C05-2b closed27 points
    dstate.jacobian_slots.touch(trial)
    dstate.jacobian_slots.touch(base)
    try:
        q1 = dstate.q1
        n_cells = dstate.grid.n_cells
        dev = base.u_cm.device
        cell_sums = torch.empty((4 if corners else 3, n_cells), dtype=torch.float64, device=dev)
        nonfinite = torch.zeros((), dtype=torch.int64, device=dev)
        nonpositive = torch.zeros((), dtype=torch.int64, device=dev)
        sD, sC = dstate.scale_diffusion[None, None, None, :], dstate.scale_curl
        omega0 = dstate.state.omega0
        for number, chunk in enumerate(q1.chunks):
            lo, hi = q1.cell_range(chunk)
            Gb, Ab, Jb, lJb = _compared_geometry(dstate, base, chunk, number)
            Gt, At, Jt, lJt = _compared_geometry(dstate, trial, chunk, number)
            for value in (Gb, Ab, Jb, Gt, At, Jt):
                if value is not None:
                    nonfinite += (~torch.isfinite(value)).sum()
            nonpositive += (Jb <= 0).sum()+(Jt <= 0).sum()
            rb, rt = sD*Gb, sD*Gt
            cell_sums[0, lo:hi] = ((rt-rb)*(rt+rb)).sum(dim=(0, 2)).sum(dim=1)
            weight = dstate.logvol_gauss_weights/(omega0*q1.interpolate(dstate.tau2_nodes, chunk))   # omega w_g
            #                                       (theta_G under C05-2b; else the tensor q1.weights) / (omega0 tau_q^2)
            x = (Jt-Jb)/Jb
            ell = torch.where(x >= -.5, torch.log1p(x), torch.log(Jt/Jb))
            cell_sums[1, lo:hi] = (weight*ell*(lJt+lJb)).sum(dim=1)
            curl = None
            for cb, ct in zip(curl_of(Gb), curl_of(Gt)):
                b, t = sC*cb, sC*ct
                d = (t-b)*(t+b)
                curl = d if curl is None else curl+d
            cell_sums[2, lo:hi] = curl.sum(dim=1)
            if corners:                                                 # C05-2a / C05-2b: the second log-volume part
                Gb, Ab, Jb, lJb = _compared_geometry(dstate, base, chunk, number, corners=True)
                Gt, At, Jt, lJt = _compared_geometry(dstate, trial, chunk, number, corners=True)
                for value in (Gb, Ab, Jb, Gt, At, Jt):
                    if value is not None:
                        nonfinite += (~torch.isfinite(value)).sum()
                nonpositive += (Jb <= 0).sum()+(Jt <= 0).sum()
                weight = _extra_weights(dstate)/(omega0*_extra_tau2(dstate, chunk))   # beta_c omega/(8 omega0 tau^2)
                #                                                        or omega s_p theta_S/(omega0 tau^2)
                x = (Jt-Jb)/Jb
                ell = torch.where(x >= -.5, torch.log1p(x), torch.log(Jt/Jb))
                cell_sums[3, lo:hi] = (weight*ell*(lJt+lJb)).sum(dim=1)
        sums = _tile_reduce(cell_sums, dstate.state.tile_size, dstate.n_tiles, n_cells)
        flags = _download(dstate, torch.stack((nonfinite, nonpositive, (~torch.isfinite(sums)).sum())))
        tiles = _download(dstate, sums).T
    except DeviceAnomaly:
        raise
    except (FloatingPointError, RuntimeError) as exc:                # torch.cuda.OutOfMemoryError included
        raise DeviceAnomaly(f"device exception {type(exc).__name__}: {exc}") from exc
    if flags[1]:
        where = {"corners": "Gauss points and cell corners (C05-2a)",
                 "closed27": "Gauss points and closed27 points (C05-2b)"}.get(dstate.logvol_extra, "Gauss points")
        raise DeviceAnomaly(f"device J_q <= 0 at {int(flags[1])} {where} of the compared states")
    if flags[0] or flags[2]:
        raise DeviceAnomaly(f"device nonfinite values ({int(flags[0])} geometry, {int(flags[2])} tile sums)")
    return np.ascontiguousarray(tiles)


def gpu_fb_change(dstate, base, trial):
    """Delta F_B of the GPU path (C08-8(a)(b)): (longdouble delta, blocks).  The channel blocks on the host from
    the downloaded channel residuals (C08-1), the Gauss-point blocks from `device_gauss_changes`."""
    state = dstate.state
    channel = [_node_change_sum(state, base.r_host[c], trial.r_host[c]) for c in range(state.n_channels)]
    tiles = device_gauss_changes(dstate, base, trial)
    corners = bool(dstate.logvol_extra)                                  # C05-2a corners / C05-2b closed27 points
    with np.errstate(over="raise", invalid="raise"):
        gauss = [LD(0) for _ in GAUSS_NAMES]
        for t in range(tiles.shape[0]):                                  # ascending tiles
            for k in range(3):
                gauss[k] += LD(tiles[t, k])
            if corners:
                gauss[1] += LD(tiles[t, 3])                              # right after the tile's Gauss sum
        half = LD(.5)
        changes = [half*s for s in channel]+[half*s for s in gauss]
        delta = LD(0)
        for change in changes:
            delta += change
    blocks = dict(channel=tuple(changes[:state.n_channels]),
                  **{name: change for name, change in zip(GAUSS_NAMES, changes[state.n_channels:])})
    return delta, blocks


# ----------------------------------------------------------------------------------------------
# (h) statistics
# ----------------------------------------------------------------------------------------------

def gf_bin(ratio):
    if ratio is None or not np.isfinite(ratio):
        return "n/a"
    lo = 0.
    for edge in GF_BINS:
        if ratio <= edge:
            return f"<={edge:g}" if lo == 0. else f"({lo:g},{edge:g}]"
        lo = edge
    return f">{GF_BINS[-1]:g}"


class ComparisonStats:
    """C08-8(h): every comparison's compact record and the run-end summary."""

    def __init__(self):
        self.records = []
        self.evaluate_anomalies = []
        self.gf_checks = []
        self.cpu_evaluations = 0
        self.cache_hits = 0
        self.base_hits = 0
        self.cpu_reductions = 0
        self.reduction_hits = 0

    def add(self, record):
        self.records.append(record)

    def summary(self):
        n = len(self.records)
        fallback = [r for r in self.records if r["fallback"]]

        def count(rows, key):
            out = {}
            for r in rows:
                out[r[key]] = out.get(r[key], 0)+1
            return dict(sorted(out.items(), key=lambda kv: str(kv[0])))

        by_bin, by_context = {}, {}
        for r in self.records:
            for table, key in ((by_bin, gf_bin(r.get("gf_ratio"))), (by_context, r["context"])):
                entry = table.setdefault(key, dict(comparisons=0, cpu_fallback=0))
                entry["comparisons"] += 1
                entry["cpu_fallback"] += int(r["fallback"])
        for table in (by_bin, by_context):
            for entry in table.values():
                entry["fallback_ratio"] = entry["cpu_fallback"]/entry["comparisons"]
        seconds = dict(gpu=float(sum(r["seconds_gpu"] for r in self.records)),
                       cpu_fallback=float(sum(r["seconds_cpu"] for r in self.records)))
        changed = [c for c in self.gf_checks if c["changed_decision"]]
        return dict(contract="C08-8(h)", comparisons=n, cpu_fallback=len(fallback),
                    fallback_ratio=len(fallback)/n if n else None,
                    fallback_reasons=count(fallback, "fallback_reason"), by_context=by_context,
                    by_gf_over_tb=dict(sorted(by_bin.items())), gf_bins=list(GF_BINS), seconds=seconds,
                    device_anomalies=dict(evaluate=len(self.evaluate_anomalies),
                                          compare=sum(1 for r in self.records
                                                      if str(r["fallback_reason"]).startswith("device"))),
                    evaluate_anomalies=self.evaluate_anomalies[:20],
                    cpu_evaluations=self.cpu_evaluations, cpu_cache_hits=self.cache_hits,
                    cpu_base_slot_hits=self.base_hits, cpu_reductions=self.cpu_reductions,
                    cpu_reduction_memo_hits=self.reduction_hits,
                    gf_rechecks=dict(count=len(self.gf_checks), changed_decision=len(changed),
                                     rows=self.gf_checks[:50]))


# ----------------------------------------------------------------------------------------------
# The comparator: GPU path, fallback, CPU reference with cache
# ----------------------------------------------------------------------------------------------

class StableComparator:
    """C08-8 for one backend.  `cpu_model` is the CPU reference module (bayesreg.model)."""

    def __init__(self, *, cache_size=2, cpu_model=_model):
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise InputValidationError("cache_size: integer >= 0 required")
        self.cache_size = cache_size
        self.cpu_model = cpu_model
        self.cache = OrderedDict()
        self.base = None              # (key, Evaluation | InvalidTrial): the base slot of the C08-8(d) cache
        self._reduction = None        # (key pair, weakref base Evaluation, weakref trial Evaluation, Reduction)
        self.stats = ComparisonStats()

    # ---- CPU reference evaluations (cached) ----
    @staticmethod
    def cache_key(point, *, model_id, energy="F_B", reference=None):
        """C08-8(d): (state_hash, ModelState identity, energy type, reference v hash)."""
        if energy not in ENERGIES:
            raise InputValidationError(f"energy: one of {ENERGIES}")
        return (point.state_hash, model_id, energy, reference)

    @staticmethod
    def _usable(value, state):
        return value is not None and (isinstance(value, InvalidTrial) or value.state is state)

    def cpu_point(self, state, point, *, model_id, energy="F_B", reference=None, base=False):
        """The CPU reference Evaluation of `point` (bayesreg.model.evaluate) or its InvalidTrial; cached under
        (state_hash, model_id, energy, reference) and reused only for the same ModelState object.  base=True also
        keeps it in the base slot (replacing the previous base), which trials never evict."""
        if energy not in ENERGIES:
            raise InputValidationError(f"energy: one of {ENERGIES}")
        if point.cpu is not None and point.cpu.state is state:
            if base:
                self.base = (self.cache_key(point, model_id=model_id, energy=energy, reference=reference), point.cpu)
            return point.cpu
        key = self.cache_key(point, model_id=model_id, energy=energy, reference=reference)
        if self.base is not None and self.base[0] == key and self._usable(self.base[1], state):
            value = self.base[1]
            self.stats.cache_hits += 1
            self.stats.base_hits += 1
        else:
            value = self.cache.get(key)
            if self._usable(value, state):
                self.cache.move_to_end(key)
                self.stats.cache_hits += 1
            else:
                value = self.cpu_model.evaluate(state, point.u)
                self.stats.cpu_evaluations += 1
                if self.cache_size:
                    self.cache[key] = value
                    self.cache.move_to_end(key)
                    while len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
        if base:
            self.base = (key, value)
        return value

    def cpu_reduction(self, state, b, t, keys):
        """stable_reduction(state, b, t) of two CPU Evaluations, memoized for the latest pair: reused only for the same
        pair of cache keys and the very same two Evaluation objects (so the same ModelState and u)."""
        memo = self._reduction
        if memo is not None and memo[0] == keys and memo[1]() is b and memo[2]() is t:
            self.stats.reduction_hits += 1
            return memo[3]
        red = self.cpu_model.stable_reduction(state, b, t)
        self.stats.cpu_reductions += 1
        self._reduction = (keys, weakref.ref(b), weakref.ref(t), red)
        return red

    def clear(self):
        self.cache.clear()
        self.base = None
        self._reduction = None

    def forget_other_states(self, state):
        """MEM (memory only): drop the cached CPU Evaluations of every ModelState other than `state` (the base slot
        included).  `_usable` serves an Evaluation only to its own ModelState object, so those entries can never be
        hit again once the solver has moved to a new (level, EM round) state; InvalidTrial entries (no state, tiny) are
        kept.  At the native size one CPU Evaluation holds rho, r (C, n) and grad (C, n, 3): 1.1 GiB with 14
        channels.  Returns the number of entries dropped."""
        stale = [key for key, value in self.cache.items()
                 if not isinstance(value, InvalidTrial) and value.state is not state]
        for key in stale:
            del self.cache[key]
        dropped = len(stale)
        if self.base is not None and not isinstance(self.base[1], InvalidTrial) and self.base[1].state is not state:
            self.base = None
            dropped += 1
        return dropped

    # ---- one comparison ----
    def compare(self, state, dstate, base, trial, *, model_id, required=0., energy="F_B", reference=None, extra=None,
                context="unspecified", gf_ratio=None):
        """C08-8 for E_b = energy(base), E_t = energy(trial): GPU path, fallback rule, CPU reference."""
        if context not in CONTEXTS:
            raise InputValidationError(f"context: one of {CONTEXTS}")
        if energy not in ENERGIES or (energy == "E_v") != (extra is not None):
            raise InputValidationError("energy 'E_v' needs the proximal ExtraBlocks, 'F_B' none")
        required = float(required)
        info = dict(context=context, energy=energy, required=required, gf_ratio=gf_ratio, A_hat=None, s_hat=None,
                    fallback=False, fallback_reason=None, A_cpu=None, s_cpu=None, seconds_gpu=0., seconds_cpu=0.)
        tick = time.perf_counter()
        anomaly = None
        if base.device is None or trial.device is None:
            anomaly = "device evaluation unavailable: " + "; ".join(
                f"{name} {p.anomaly}" for name, p in (("base", base), ("trial", trial)) if p.device is None)
        else:
            try:
                delta, blocks = gpu_fb_change(dstate, base.device, trial.device)
            except DeviceAnomaly as exc:
                anomaly = str(exc)
        result = None
        if anomaly is None:
            with np.errstate(over="raise", invalid="raise"):
                if extra is not None:
                    for change in extra.changes.values():
                        delta += change
                    blocks = dict(blocks, **extra.changes)
                    E_b = float(LD(base.total)+extra.base_total)
                    E_t = float(LD(trial.total)+extra.trial_total)
                else:
                    E_b, E_t = base.energy, trial.energy
            A_hat, s_hat = float(-delta), gpu_guard(E_b, E_t)
            decision = fallback_rule(A_hat, s_hat, required)
            info.update(A_hat=A_hat, s_hat=s_hat, E_b=E_b, E_t=E_t)
            if decision.fallback:
                info.update(fallback=True, fallback_reason=decision.reason)
            else:
                result = GuardedReduction(A_hat, s_hat, delta, blocks, "gpu", info)
        else:
            info.update(fallback=True, fallback_reason=f"device anomaly: {anomaly}")
        info["seconds_gpu"] = time.perf_counter()-tick
        if result is None:
            tick = time.perf_counter()
            result = self._cpu_compare(state, base, trial, model_id=model_id, energy=energy, reference=reference,
                                       extra=extra, info=info)
            info["seconds_cpu"] = time.perf_counter()-tick
        self.stats.add(info)
        return result

    def _cpu_compare(self, state, base, trial, *, model_id, energy, reference, extra, info):
        """C08-1..7 on the CPU for both states (the reference path of C08-8(d)(e)); the base from the base slot when
        its key is unchanged, so only the candidate is re-evaluated."""
        b = self.cpu_point(state, base, model_id=model_id, energy=energy, reference=reference, base=True)
        if isinstance(b, InvalidTrial):
            raise FloatingPointError(f"CPU reference: the base state is not evaluable ({b.reason}: {b.message})")
        t = self.cpu_point(state, trial, model_id=model_id, energy=energy, reference=reference)
        if isinstance(t, InvalidTrial):
            info.update(cpu_rejection=t.reason)
            return GuardedReduction(None, None, None, {}, "cpu", info, rejection=t.reason)
        keys = tuple(self.cache_key(p, model_id=model_id, energy=energy, reference=reference) for p in (base, trial))
        red = self.cpu_reduction(state, b, t, keys)
        if extra is None:
            ared, guard, delta, blocks = red.ared, red.guard, red.delta, red.blocks
        else:                                                    # bayesreg.solve.proximal.combined_reduction
            with np.errstate(over="raise", invalid="raise"):
                delta = LD(red.delta)
                for change in extra.changes.values():
                    delta += change
                E_b = float(LD(b.total)+extra.base_total)
                E_t = float(LD(t.total)+extra.trial_total)
            ared = float(-delta)
            guard = float(GUARD_FACTOR*EPS*max(E_b, E_t))
            blocks = dict(red.blocks, **extra.changes)
        if not (np.isfinite(ared) and np.isfinite(guard)):
            raise FloatingPointError("stable energy reduction: nonfinite numerical result")
        info.update(A_cpu=ared, s_cpu=guard)
        return GuardedReduction(ared, guard, delta, blocks, "cpu", info)

    # ---- (f) GF_B double-sided recheck ----
    def confirm_gradient_norm(self, state, point, G, target, *, model_id, energy="F_B", reference=None,
                              extra_gradient=None, context="unspecified"):
        """C08-8(f): G (from the device gradient) if it lies outside [T/1.01, 1.01 T], else the CPU value
        ||grad||/sqrt(omega_ell/omega0) of the CPU reference gradient (plus `extra_gradient()` for E_v)."""
        from ...solve.stopping import gf_b
        if G is None or not (target/GF_BAND <= G <= GF_BAND*target):
            return G, None
        tick = time.perf_counter()
        cpu = self.cpu_point(state, point, model_id=model_id, energy=energy, reference=reference, base=True)
        if isinstance(cpu, InvalidTrial):
            raise FloatingPointError(f"CPU reference: GF recheck state not evaluable ({cpu.reason})")
        g = self.cpu_model.gradient(state, cpu)
        if extra_gradient is not None:
            with np.errstate(over="raise", invalid="raise"):
                g = g+extra_gradient()
        G_cpu = gf_b(g, state.grid.omega, state.omega0)
        row = dict(context=context, energy=energy, G_gpu=G, G_cpu=G_cpu, target=target,
                   changed_decision=bool((G <= target) != (G_cpu <= target)), seconds=time.perf_counter()-tick)
        self.stats.gf_checks.append(row)
        return G_cpu, row


# ----------------------------------------------------------------------------------------------
# Device residuals of a cell range (C08-9 premise (i) on sampled tiles)
# ----------------------------------------------------------------------------------------------

def device_residuals(dstate, tev, lo, hi):
    """The weighted Gauss-point residuals of cells [lo, hi) from the device, in the CPU layouts of
    bayesreg.model.blocks: diffusion (cells, 27, 3, 3) [cell, q, r, j], logvol (cells, 27), curl (cells, 27, 3); under
    C05-2a also logvol_corners (cells, 8) (LogVolumeCorners), under C05-2b logvol_closed27 (cells, 27)
    (LogVolumeClosed27)."""
    q1 = dstate.q1
    extra_name = {"corners": "logvol_corners", "closed27": "logvol_closed27"}.get(dstate.logvol_extra)
    names = GAUSS_NAMES+((extra_name,) if extra_name else ())
    parts = {name: [] for name in names}
    sD, sC = dstate.scale_diffusion[None, None, None, :], dstate.scale_curl
    for chunk in q1.chunks:
        c_lo, c_hi = q1.cell_range(chunk)
        if c_hi <= lo or c_lo >= hi:
            continue
        a, b = max(lo, c_lo)-c_lo, min(hi, c_hi)-c_lo
        G, _, _, J = _geometry(dstate, tev.u_cm, chunk)
        parts["diffusion"].append((sD*G).permute(1, 3, 0, 2)[a:b])
        parts["logvol"].append((_logvol_scale(dstate, chunk)*torch.log(J))[a:b])
        parts["curl"].append(torch.stack([sC*c for c in curl_of(G)], dim=-1)[a:b])
        if extra_name:
            Jc = _extra_geometry(dstate, tev.u_cm, chunk)[3]
            parts[extra_name].append((_extra_scale(dstate, chunk)*torch.log(Jc))[a:b])
    return {name: _download(dstate, torch.cat(value)) for name, value in parts.items()}
