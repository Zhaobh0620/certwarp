"""sigma_domain: the frozen support weights v of the structure channels and the level-start sums (contract C04-6, C11-8;
D49, D50; work package SWLI; draft SWL 2026-10-31, default "brain" bitwise unchanged; BZ decides D49 / D50).

Values of the switch (C04-6):
  "brain"           (default) C04-2 as before: v_c = 1, N_c = N_q; nothing in this module runs.
  "support"         every structure channel c in S = {1..K_S} has its own sigma_c^2 on the frozen support S_c^ell.
  "support_shared"  the same frozen supports; one sigma_S^2 shared by the K_S structure channels.
The intensity channel (c = 0) is the same under all three (v_I = 1, count N_q).

Support (C04-6 "支撑的定义", D50 (1)): on level ell with seed u_ell (ell = 0: u = 0; ell > 0: the first certified scaled
prolongation, C12-2) and t_theta = np.sqrt(theta) (float64),
    S_c^ell = {a : m_a = 1, g_c(a) > t_theta or Ghat_{c,a} > t_theta},   Ghat_{c,a} = g_c(a) + rho_c(u_ell)_a (float64),
v_c(a) = 1[a in S_c^ell], stored on the mask nodes as one read-only (K_S, n) uint8 array (rows c = 1..K_S, columns in
node_index C-order; `ModelState.support_v`), hashed as sha256 of its C-order bytes.  rho_c(u_ell) is ALWAYS the CPU
reference's (C04-6 "后端", C17-1): `cpu_channel_rho` samples channel c by `sample_points(sampler_c, points[lo:hi],
strict=state.strict[lo:hi], with_gradient=False)` on chunks of SPLINE_CHUNK mask nodes starting at multiples of it (the
very calls of `blocks._sample_channels`), points[lo:hi] = points0[lo:hi] + u[node_index[lo:hi]] (the expression of
`channel_point`, chunk by chunk) and rho = value - g_c (one float64 subtraction); values only, no gradient; no (K_S, n)
float64 array is ever built.  So v depends only on the seed's bytes, the level's arrays and theta -- not on the backend.

Level-start sums (C11-8 "层首的算式", exact summation of the SWL review round 2): per channel, rho_c fills one float64
array of length n (mask nodes, node_index C-order); then
    N_c  = np.sum(q * v_c, dtype=longdouble) / longdouble(omega0)                      (q = node weights on the mask)
    S_c  = np.sum((q * v_c) * (rho_c * rho_c), dtype=longdouble) / longdouble(omega0)   (intensity: v = 1, i.e. q * rho^2)
one np.sum over the whole array each (not per chunk); N_q = np.sum(q, dtype=longdouble)/longdouble(omega0) (bitwise the
solver's n_q).  The products are formed in the contract's order: tmp = q * v_c (float64; exact, v in {0, 1}), N_c from
tmp, then rho_c *= rho_c in place and tmp *= rho_c.  support_shared adds S_c and N_c (longdouble) over the non-empty c in
ascending order (`SupportLevel.shared_sums`); each value is converted to float64 once.

Temporaries: rho_c and tmp (two float64 arrays of length n, 16 MiB each at the native OASIS size; tmp holds q, then
q * v_c, then the products -- q is written into it chunk by chunk, no length-n q is kept; review SWLI minor 5) and v; the
support bits of a channel are computed chunk by chunk; v is hashed through its buffer (no copy).
Threads (review SWLI minor 6): the spline chunks of one channel run in state.cpu_threads threads (`cpu_channel_rho`, the
chunk calls of the serial loop), the channels one after the other -- so the temporaries stay those of one channel and every
value (v, the sums, each one np.sum over a whole length-n array) is bitwise that of the serial pass.

Threshold theta (D50 (2); C04-6 "支撑阈值"): finite, 0 < theta < 1 (`check_theta_coarse`, at argument parsing) and
theta > eps/(1 + (K + 1) eps) with eps = FLOOR_EPS = 1e-3 (the C03.5-9 floor of `seg.channels.channel_pyramid`) and K the
number of likelihood channels after the intensity (K_S; K_T under channel_source tissue) (`check_theta`).
"""
from contextlib import closing
from dataclasses import dataclass
import hashlib
import time

import numpy as np

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError
from jd_curl.three_d.interpolation import check_map_domain, sample_points

from ..geom import node_field, ordered_map
from .blocks import SPLINE_CHUNK

LD = np.longdouble
BRAIN = "brain"
SUPPORT = "support"
SUPPORT_SHARED = "support_shared"
SIGMA_DOMAINS = (BRAIN, SUPPORT, SUPPORT_SHARED)
DEFAULT_SIGMA_DOMAIN = BRAIN
DEFAULT_SUPPORT_THRESHOLD = .05          # D50 (2)
SUPPORT_RULE = "union_seed_sqrt_gt"      # C04-6 "记录": union of both sides, level seed, square-root coordinates, ">"
FLOOR_EPS = 1e-3                         # C03.5-9 floor eps (seg.channels.channel_pyramid default)
NEAR_THRESHOLD = 1e-12                   # n_near_threshold: |Ghat - t_theta| <= 1e-12 (C04-6 "后端")
SIGMA2_PLACEHOLDER = 1.                  # internal sigma_c^2 of a channel without likelihood (beta_c = 0; never recorded)
_RAISE = dict(over="raise", invalid="raise", divide="raise")


def theta_lower_bound(k, eps=FLOOR_EPS):
    """eps/(1 + (K + 1) eps): the C03.5-9 floor of the K + 1 channels of one layer (C04-6 "下界")."""
    return eps/(1+(k+1)*eps)


def check_theta_coarse(theta):
    """C04-6 first step (argument parsing): theta a finite real with 0 < theta < 1; returns float(theta)."""
    if isinstance(theta, bool):
        raise InputValidationError("support_threshold: a number required")
    try:
        value = float(theta)
    except (TypeError, ValueError):
        raise InputValidationError(f"support_threshold: {theta!r} is not a number") from None
    if not np.isfinite(value) or not 0 < value < 1:
        raise InputValidationError(f"support_threshold: {value!r} must be finite with 0 < theta < 1 (C04-6, D50)")
    return value


def check_theta(theta, k, eps=FLOOR_EPS):
    """C04-6 second step: theta > eps/(1 + (K + 1) eps) (K = K_S, or K_T under channel_source tissue), after the coarse
    check.  At or below the floor every fixed-side node passes, i.e. the support is the whole mask (brain recorded as
    support)."""
    value = check_theta_coarse(theta)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise InputValidationError("support_threshold: K >= 1 likelihood channels after the intensity required")
    bound = theta_lower_bound(int(k), eps)
    if not value > bound:
        raise InputValidationError(f"support_threshold: theta = {value!r} <= eps/(1 + (K + 1) eps) = {bound!r} "
                                   f"(K = {int(k)}, eps = {eps:g}; C04-6, D50: the support would be the whole mask)")
    return value


def resolve_sigma_domain(sigma_domain=None, support_threshold=None):
    """(sigma_domain, theta) in effect: None -> "brain"; "brain" refuses an explicit theta (recorded null); the other
    values take theta (default .05, coarse check here; the exact lower bound needs K, `check_theta`)."""
    domain = DEFAULT_SIGMA_DOMAIN if sigma_domain is None else sigma_domain
    if domain not in SIGMA_DOMAINS:
        raise InputValidationError(f"sigma_domain: one of {SIGMA_DOMAINS} (C04-6)")
    if domain == BRAIN:
        if support_threshold is not None:
            raise InputValidationError("support_threshold: not with sigma_domain 'brain' (C04-6: refused so that the "
                                       "record does not mislead; recorded as null)")
        return domain, None
    return domain, check_theta_coarse(DEFAULT_SUPPORT_THRESHOLD if support_threshold is None else support_threshold)


def domain_record(sigma_domain, support_threshold):
    """The three record keys of C04-6 "记录" (config.json, provenance.json switches, hyper.json): brain -> theta and rule
    null."""
    brain = sigma_domain == BRAIN
    return dict(sigma_domain=sigma_domain, support_threshold=None if brain else float(support_threshold),
                support_rule=None if brain else SUPPORT_RULE)


def v_sha256(v):
    """sha256 of v as uint8 in C order (C04-6 "存储"); hashed through the buffer, no copy of a C-contiguous uint8 v (review
    SWLI minor 5)."""
    return hashlib.sha256(np.ascontiguousarray(v, dtype=np.uint8).data).hexdigest()


def cpu_channel_rho(state, u, c, out=None, threads=None):
    """rho_c(u) on the mask nodes (n,) float64 by the CPU reference, chunk by chunk and values only (module docstring),
    or the InvalidTrial of sample_points (the first chunk's, in chunk order).  u: a U0 node field already checked by
    check_map_domain.  threads (default state.cpu_threads; review SWLI minor 6): the SPLINE_CHUNK chunks of the channel
    run in that many threads (`geom.ordered_map`); every chunk is the very sample_points call of the serial loop and
    writes its own slice, so rho is bitwise the same for every thread count."""
    n = state.node_index.size
    rho = np.empty(n) if out is None else out
    flat = u.reshape(-1, 3)
    sampler, fixed = state.samplers[c], state.fixed_masked[c]
    chunks = [(lo, min(lo+SPLINE_CHUNK, n)) for lo in range(0, n, SPLINE_CHUNK)]

    def piece(chunk):
        lo, hi = chunk
        with np.errstate(**_RAISE):
            points = state.points0[lo:hi]+flat[state.node_index[lo:hi]]
        return sample_points(sampler, points, strict=state.strict[lo:hi], with_gradient=False)
    with closing(ordered_map(piece, chunks, state.cpu_threads if threads is None else threads)) as samples:
        for (lo, hi), sample in zip(chunks, samples):
            if isinstance(sample, InvalidTrial):
                return sample
            with np.errstate(**_RAISE):
                np.subtract(sample.value, fixed[lo:hi], out=rho[lo:hi])
    return rho


def _weight_parts(state):
    """((omega tz) ty) on the (nz, ny) plane, tx and nx: the factors of q = grid.node_weights() (jd_curl: omega * tz * ty
    * tx, left to right)."""
    grid = state.grid
    tx, ty, tz = grid.trapezoid
    return (grid.omega*tz[:, None]*ty[None, :]).ravel(), tx, grid.shape[2]


def _fill_weights(parts, index, out, row=None):
    """out[:] = q on the mask nodes (row None) or q * v_c (row, exact: v in {0, 1}), per SPLINE_CHUNK nodes; bitwise
    grid.node_weights().ravel()[index] (times row) without a full-grid or a second length-n array (review SWLI minor 5)."""
    plane, tx, nx = parts
    for lo in range(0, index.size, SPLINE_CHUNK):
        part = index[lo:lo+SPLINE_CHUNK]
        np.multiply(plane[part//nx], tx[part % nx], out=out[lo:lo+SPLINE_CHUNK])
        if row is not None:
            out[lo:lo+SPLINE_CHUNK] *= row[lo:lo+SPLINE_CHUNK]
    return out


def node_weights_masked(state):
    """q_a on the mask nodes (node_index order), float64: the weights of N_q, N_c and the level-start sums.  Bitwise
    grid.node_weights().ravel()[node_index] without the full-grid array (memory only)."""
    return _fill_weights(_weight_parts(state), state.node_index, np.empty(state.node_index.size))


@dataclass(frozen=True, eq=False)
class SupportLevel:
    """The frozen support of one level and the level-start sums (C04-6, C11-8).  v: (K_S, n) uint8 read-only;
    n_c_ld / s_ld: longdouble N_c (c = 1..K_S) and S_c (c = 0..K_S, the intensity first)."""
    level: int
    sigma_domain: str
    theta: float
    t_theta: float
    seed_state_hash: str | None
    v: np.ndarray
    v_sha256: str
    n_q_ld: np.longdouble
    n_c_ld: tuple
    s_ld: tuple
    support_nodes: tuple          # |S_c^ell| per structure channel (node count)
    fixed_only: tuple             # nodes in S_c only because of the fixed side
    moving_only: tuple            # nodes in S_c only because of the moving side
    near_threshold: tuple         # per channel |Ghat - t_theta| <= 1e-12
    seconds: float

    @property
    def k_s(self):
        return len(self.n_c_ld)

    @property
    def n_q(self):
        return float(self.n_q_ld)

    @property
    def n_c(self):
        """N_c (float64) per structure channel c = 1..K_S."""
        return tuple(float(x) for x in self.n_c_ld)

    @property
    def empty(self):
        """The structure channels (numbers 1..K_S) with an empty support on this level (C04-6 "空支撑")."""
        return tuple(c+1 for c, count in enumerate(self.support_nodes) if count == 0)

    @property
    def nonempty(self):
        return tuple(c+1 for c, count in enumerate(self.support_nodes) if count > 0)

    def shared_sums(self):
        """support_shared: (S_S, N_S) as float64 -- the longdouble S_c and N_c of the non-empty structure channels added
        in ascending c, each total converted once (C11-8)."""
        s, n = LD(0), LD(0)
        for c in self.nonempty:
            s += self.s_ld[c]
            n += self.n_c_ld[c-1]
        return float(s), float(n)

    def n_s(self):
        return self.shared_sums()[1]

    def record(self):
        """hyper.json per level (C04-6 "记录"; written only when sigma_domain != brain)."""
        n_q = self.n_q
        n_c = self.n_c
        out = dict(level=self.level, sigma_domain=self.sigma_domain, theta=self.theta, t_theta=self.t_theta,
                   support_rule=SUPPORT_RULE, seed_state_hash=self.seed_state_hash, N_q=n_q,
                   support_nodes=list(self.support_nodes), N_c=list(n_c),
                   N_c_over_N_q=[x/n_q for x in n_c], fixed_only_nodes=list(self.fixed_only),
                   moving_only_nodes=list(self.moving_only), empty_support_channels=list(self.empty),
                   n_near_threshold=int(sum(self.near_threshold)), n_near_threshold_per_channel=list(self.near_threshold),
                   near_threshold_tol=NEAR_THRESHOLD, v_sha256=self.v_sha256, v_shape=list(self.v.shape),
                   v_dtype="uint8", cpu_seconds=self.seconds,
                   source="CPU reference (bayesreg.model.support.cpu_channel_rho: sample_points per SPLINE_CHUNK, "
                          "values only) at the level seed (C04-6 后端)")
        if self.sigma_domain == SUPPORT_SHARED:
            out["N_S"] = self.n_s()
        return out


def support_counts(state):
    """N_c (c = 1..K_S) and N_S (support_shared: the non-empty N_c added in ascending c, longdouble) of a ModelState with
    a frozen v, by the formulas of `support_level` (bitwise its values for the same v); None under brain."""
    if state.support_v is None:
        return None
    parts, index = _weight_parts(state), state.node_index
    tmp = _fill_weights(parts, index, np.empty(index.size))
    omega0 = LD(state.omega0)
    n_q = np.sum(tmp, dtype=LD)/omega0
    n_c, n_s = [], LD(0)
    for row in state.support_v:
        value = np.sum(_fill_weights(parts, index, tmp, row), dtype=LD)/omega0
        n_c.append(float(value))
        if row.any():
            n_s += value
    return dict(sigma_domain=state.sigma_domain, support_threshold=state.support_threshold, N_q=float(n_q), N_c=n_c,
                N_S=float(n_s), empty_support_channels=list(state.support_empty), v_sha256=state.support_v_sha256)


def support_level(state, u, theta, *, sigma_domain, level=None, seed_state_hash=None, rows=None):
    """C04-6 / C11-8 at the level start: SupportLevel of ModelState `state` (the level template: grid, mask, channels,
    samplers; its sigma, w and own v are not read) at the seed `u`, or an InvalidTrial (domain / nonfinite spline).

    rows: optional callable c -> rho_c (n,) float64 replacing `cpu_channel_rho` (tests: the rows of channel_point's
    rho, C04-6 gate 9 (iii)); it must return a fresh writable array (it is overwritten)."""
    if sigma_domain not in (SUPPORT, SUPPORT_SHARED):
        raise InputValidationError(f"sigma_domain: {sigma_domain!r} has no support (C04-6)")
    k_s = state.n_channels-1
    theta = check_theta(theta, k_s)
    t = float(np.sqrt(np.float64(theta)))
    tick = time.perf_counter()
    u = node_field(u, state.grid)
    if rows is None:
        invalid = check_map_domain(u, state.grid)
        if invalid is not None:
            return invalid

        def rows(c):
            return cpu_channel_rho(state, u, c)
    n = state.node_index.size
    parts, index = _weight_parts(state), state.node_index
    tmp = np.empty(n)                                              # q, then q * v_c, then (q v_c)(rho rho)
    omega0 = LD(state.omega0)
    v = np.zeros((k_s, n), dtype=np.uint8)
    n_c, s, nodes, fixed_only, moving_only, near = [], [], [], [], [], []
    with np.errstate(**_RAISE):
        n_q = np.sum(_fill_weights(parts, index, tmp), dtype=LD)/omega0
        for c in range(state.n_channels):
            rho = rows(c)
            if isinstance(rho, InvalidTrial):
                return rho
            if c == 0:
                np.multiply(rho, rho, out=rho)
                _fill_weights(parts, index, tmp)
                np.multiply(tmp, rho, out=tmp)                     # (q * 1) * (rho * rho)
                s.append(np.sum(tmp, dtype=LD)/omega0)
                del rho
                continue
            g = state.fixed_masked[c]
            row = v[c-1]
            counts = [0, 0, 0]
            for lo in range(0, n, SPLINE_CHUNK):
                hi = min(lo+SPLINE_CHUNK, n)
                ghat = g[lo:hi]+rho[lo:hi]                         # Ghat = g + rho (one float64 addition)
                fixed_side = g[lo:hi] > t
                moving_side = ghat > t
                row[lo:hi] = fixed_side | moving_side
                counts[0] += int(np.count_nonzero(fixed_side & ~moving_side))
                counts[1] += int(np.count_nonzero(moving_side & ~fixed_side))
                counts[2] += int(np.count_nonzero(np.abs(ghat-t) <= NEAR_THRESHOLD))
            fixed_only.append(counts[0])
            moving_only.append(counts[1])
            near.append(counts[2])
            nodes.append(int(np.count_nonzero(row)))
            _fill_weights(parts, index, tmp, row)                  # q * v_c (float64; exact)
            n_c.append(np.sum(tmp, dtype=LD)/omega0)
            np.multiply(rho, rho, out=rho)
            np.multiply(tmp, rho, out=tmp)                         # (q * v_c) * (rho * rho)
            s.append(np.sum(tmp, dtype=LD)/omega0)
            del rho
    del tmp
    v.flags.writeable = False
    return SupportLevel(level=level, sigma_domain=sigma_domain, theta=theta, t_theta=t, seed_state_hash=seed_state_hash,
                        v=v, v_sha256=v_sha256(v), n_q_ld=n_q, n_c_ld=tuple(n_c), s_ld=tuple(s),
                        support_nodes=tuple(nodes), fixed_only=tuple(fixed_only), moving_only=tuple(moving_only),
                        near_threshold=tuple(near), seconds=time.perf_counter()-tick)
