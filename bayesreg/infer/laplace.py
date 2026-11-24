"""Laplace posterior N(u_hat, H^-1) by perturb-and-MAP (contract C14; theory B5, B6, B16).

H = sum_i K_i^T K_i is the GN Hessian of F_B at the output u_hat (C06-5): the channel blocks (sigma_used,
w_used) and the three prior blocks, never the proximal terms (lam, mu, nu2) of outer option (b) (C14-1).

Sampling (C14-2, B5-1).  Per sample, independent eps_i ~ N(0, I_{m_i}) per residual block and
    H delta = b,   b = sum_i B_i eps_i,   B_i B_i^T = K_i^T K_i,
so Cov(delta) = H^-1.  B_i = K_i^T for the channel, log-volume and curl blocks (their adjoints, never
assembled); for the diffusion block the default is the DST square root B_D = sqrt(alpha/omega0) K_Q^1/2,
K_Q^1/2 = S diag(sqrt(omega_ell Lambda_Q)) S with S = S^T = S^-1 the orthonormal DST-I of
`jd_curl.three_d.poisson` (eps_D in R^{3 N_int}); `diffusion_factor="explicit"` uses K_D^T with
eps_D in R^{9|Q|} instead (same covariance).  Block dimensions m_i: channel c: the mask nodes
(ModelState.node_index; the rows of r_c outside the mask are identically zero, B5 proof (1)); log-volume
|Q|; curl 3|Q|; diffusion 3 N_int (dst_sqrt) or 9|Q| (explicit).

Random numbers (C14-4).  `numpy.random.default_rng(seed).spawn(S)`: sample s draws from child s, in the
fixed order channel (C, n); diffusion (3 N_int,) [dst_sqrt]; then per cell tile of state.tile_size cells
in ascending order: diffusion (cells, 27, 3, 3) [explicit], logvol (cells, 27), curl (cells, 27, 3).  The
assignment of normals to residual rows therefore depends on the tile size (recorded).  The Gauss-point
noise is drawn tile by tile, so one sample never holds the 13 |Q| normals of a native volume at once.

PCG (C14-3, B5-3).  x0 = 0, atol = 0, rtol = 1e-3, maxiter = 200, preconditioner the C09.1 metric M
with gamma = 0 and lam_b = 0 (c_dat frozen at u_hat, b_P by `metric_mode`).  A column stops when its true
residual ||b - H x|| <= rtol ||b|| (one extra H application whenever the recursive residual meets the
tolerance; B5-3 is stated for the true residual).  A sample that does not reach rtol within maxiter is
`pcg_not_converged`, counted and excluded from every statistic.  The solve is an odd function of b
(B5-3, EAM), so E delta = 0; the covariance bias is O(rtol) (qc.json carries the note).

Posterior maps (C14-5).  A sample whose PCG did not reach rtol is `pcg_not_converged` (C14-3): counted as
`pcg_unconverged`, NOT certified, and in no statistic -- neither a certificate failure nor part of sigma_u,
the pass rate or the traces.  Every PCG-converged sample u_s = u_hat + delta_s is certified on the CPU in
FP64 with the C07 configuration (PrescreenCertificateConfig(delta=0.), max_depth 48, max_boxes 255 and the
run's recorded tile size T, checked field by field).
  sigma_u      all PCG-converged samples (certified or not), mean known: sigma_{u,d}^2 = (1/S_conv)
               sum_s delta_{s,d}^2 at the nodes, |u| norm sqrt(sum_d sigma_{u,d}^2), mm; faces 0.
  sigma_logJ   PCG converged AND certificate PASS AND every floating J_q > 0 (C07-5 iii; a sample with a
               J_q <= 0 after PASS is excluded with NONPOSITIVE_GAUSS_JD): Welford mean / M2 of log J_q
               over these S_pass samples (float64, sample order), ddof = 1 (NaN if S_pass < 2), then
               sigma_logJ(K) = sum_g w_g sigma(K, g) with w_g = ops.weights / omega_ell (sum 1).
  pass_rate    S_cert / S_conv: certificate PASS among the PCG-converged samples (C14-3 "not in any
               statistic", review M4 item 1); NaN when S_conv = 0 (then the flag is None, not True).
               pass_rate < .8 sets qc `posterior_cert_pass_rate_low` (C14-5, D20).  usable_fraction =
               S_pass / S is recorded next to it (the share of all samples that entered sigma_logJ).
  cell log J   per passed sample the cell average sum_g w_g log J_q on the cells with at least one mask
               vertex, float32 (D22), for bench's roi_logj.csv (labels never read here).
Memory (estimate from the array sizes, not measured; review M4 item 2).  |Q| = 27 n_cells Gauss points,
N_f = 3 N_int free unknowns; OASIS native 160 x 192 x 224 nodes: |Q| = 1.83e8 (8 |Q| = 1.46 GB),
N_f = 2.0e7 (8 N_f = 0.16 GB).  The log J of a sample is streamed tile by tile twice (pass 1: every
J_q > 0; pass 2: log J into Welford and the cell averages), u_hat's brain log J only lives until its
quantiles are taken, and sigma_logJ is finalized per tile in place, so during the sample loop
    peak ~ 16 |Q|                 Welford mean and M2 (2.9 GB, the whole loop)
         + 4 n_brain S_pass       float32 cell log J kept so far (n_brain <= n_cells = 6.8e6: <= 0.9 GB at S = 32)
         + 6 x 8 N_f              PCG vectors b, x, r, p, z, Hp of the next sample (~1.0 GB)
         + B_state + B_cert       ModelState / backend and the certificate's own working memory (not estimated)
         ~ 4.8 GB + B_state + B_cert at the native size;
the final np.stack of the cell log J (another 4 n_brain S_pass) runs after the Welford arrays are released.
The version reviewed in M4 (315e98c) also kept u_hat's log J (8 |Q|) for the whole loop and one sample's
log J list plus its np.concatenate (2 x 8 |Q|): 40 |Q| + the rest, about 7-8 GB.

Traces (C15-3, B6).  `sample_trace`: (1/S_conv) sum_s delta_s^T A delta_s over the converged samples (the
estimator C15 uses); `hutchinson_trace`: Rademacher probes z, PCG y = H^-1 z, mean z^T A y (the C14 check
only).  A = K_Q/omega0 (`stiffness_apply_scaled`) or C^T W C/omega0 (`curl_gram_apply`), W = omega w_g.

Backends.  The system uses only the backend protocol of `bayesreg.solve.backend` (prepare, evaluate, hvp,
c_dat, metric_bP, build_metric, certify) plus one OPTIONAL method:
    perturbation_rhs(state, ev, draw, *, diffusion) -> host float64 free vector b = sum_i B_i eps_i,
with draw(name, shape) the host noise source above (same order).  Without it the numpy reference
`perturbation_rhs` runs on the host ModelState and a host Evaluation of u_hat.  `pcg` works on host numpy
arrays (single right-hand side, or several as columns); a device backend may return host vectors from hvp
and Metric.solve (as the protocol requires) or bring its own solver.

Outputs (C16): sigma_u.npz, sigma_logj.npz, posterior_logj_cells.npz and qc.json via
`write_posterior_outputs` (npz rather than NIfTI: float64 without a header round trip, the grid record
inside; the run's NIfTI header lives with run_pair, which can export with nifti.save_scalar).
Nothing is overwritten (hard constraint 4): all four paths are checked before the first is written.
Each output carries u_hat_state_hash = sha256 of u_hat's float64 C-order bytes (the C07-7 / C16 state_hash
convention); run_laplace / posterior_maps take `expected_state_hash` (final_native.npz's state_hash) and
refuse a different u_hat.  ROI statistics are bench's (D22).

Hutchinson seeds: use `hutchinson_seed_from_run_id(run_id)`, a stream separate from the sampling seed
(the same integer would give probes from the same child streams as the samples' noise).

Prior-side samples (contract C15-9, eb_count = "gull_mackay"; theory B17).  `PriorSystem` is the same perturb-and-MAP
construction for Pi_0 = sum_i K_{i,0}^T K_{i,0} over the three prior blocks (diffusion, logvol, curl) at u = 0 under
the state's eta: no channel block, no proximal terms.  `prior_perturbation_rhs` draws, in this order, diffusion
(3 N_int,) -- always the DST square root, also when the C14 posterior samples use diffusion_factor "explicit" (C15-9)
-- then per tile ascending logvol (cells, 27), curl (cells, 27, 3); at u = 0 the log-volume block is
diag(sqrt(c_q)/tau_q) D_div (B13-2).  PCG as C14-3 (x0 = 0, rtol, maxiter, true-residual stop) with the C09.1 metric at
c_dat = lam_b = gamma = 0 (the DST inverse of (b_P/omega0) K_Q); by B17-5 its condition number is at most
max(a_max, b)/min(a_min, b), a_q = alpha + 1/tau_q^2, b = alpha + nu, independent of the grid.  Cov(delta0) = Pi_0^-1
(B5-1 with H replaced by Pi_0).

Log-volume point sets (contract C14-7, C15-10, C05-2b "C14, C15"; work package IMPL27b).  logvol_points "gauss27"
(C05-2; the default before DEF, an explicit option since): everything above, bitwise the code before C05-2b.
"gauss27+closed27" with closed27_weights "simpson" (C05-2b; the model default since DEF, 2026-10-31): H and Pi_0
contain K_{J,S}^T K_{J,S} (bayesreg.model.hvp, "logvol" = the whole block); the log-volume noise gets a second part
per tile, drawn right after the curl noise of that tile: logvol_closed27 (cells, 27), B = K_{J,S}^T
(the closed27 cotangent of bayesreg.model.LogVolumeClosed27, scattered with the closed27 operators right after the
tile's Gauss-point scatter).  Noise order (posterior) per tile: [diffusion explicit], logvol, curl, logvol_closed27;
(prior) logvol, curl, logvol_closed27 -- it changes the random-number assignment of later tiles, no correspondence
with the gauss27 sample stream is kept.  At u = 0 the closed27 block is diag(sqrt(c_p)/tau_p) D_{div,S}.  A sample
enters sigma_logJ only when every floating J > 0 at the Gauss AND the closed27 points (C07-5 iii under C05-2b);
sigma_logJ, the cell log J of bench's ROI statistics and the u_hat log J quantiles stay on the Gauss points (C14-5
unchanged); posterior_maps(keep_gauss_variance=True) also returns the ddof = 1 variance of log J at the closed27 points
(logj_var_closed27, for C15-10; another 16 bytes per closed27 point during the loop).  closed27_weights "uniform" and
logvol_points "gauss27+corners" (C05-2a) are refused: uniform is a scope limit of C14-7 / C15-10 (the premises of C14,
C15 were checked for the Simpson weights only), not a mathematical obstacle.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.fft import dstn, idstn

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError, as_float64_array, finite_real, require_finite_result
from jd_curl.three_d.boundary import inject_free
from jd_curl.three_d.cell_q1 import cell_tiles
from jd_curl.three_d.poisson import poisson_spectrum
from jd_curl.three_d.validation import integer_at_least

from .. import model as _model
from ..geom import finish, node_field, stiffness_apply, tile_geometry, tile_jvp, tile_vjp_accumulate, vjp_stream
from ..model import (Curl, Diffusion, Evaluation, LogVolume, LogVolumeClosed27, ModelState, NONPOSITIVE_GAUSS_JD,
                     channel_vjp, merge_cotangents)
from ..model.energy import METRIC_MODES
from ..solve.backend import CERT_TILE_SIZE, NumpyBackend, certificate_record, check_certificate_config
from ..solve.export import grid_record, write_json

S_NATIVE = 32                     # C14-4: native output
S_GROUP = 16                      # C14-4 / C15-3: group empirical Bayes
PCG_RTOL = 1e-3                   # C14-3 / D20
PCG_MAXITER = 200                 # C14-3
PASS_RATE_THRESHOLD = .8          # C14-5 / D20
DIFFUSION_FACTORS = ("dst_sqrt", "explicit")
NOISE_ORDER = ("channel (C, n)", "diffusion (3 N_int,) [dst_sqrt]",
               "per tile ascending: diffusion (cells, 27, 3, 3) [explicit], logvol (cells, 27), curl (cells, 27, 3)")
PCG_NOT_CONVERGED = "pcg_not_converged"
PRIOR_PCG_NOT_CONVERGED = "prior_pcg_not_converged"     # C15-9
PRIOR_BLOCKS = ("diffusion", "logvol", "curl")           # C15-9: Pi_0, the prior GN precision at u = 0
PRIOR_NOISE_ORDER = ("diffusion (3 N_int,) [dst_sqrt, always]", "per tile ascending: logvol (cells, 27), "
                     "curl (cells, 27, 3)")
NOISE_ORDER_CLOSED27 = ("channel (C, n)", "diffusion (3 N_int,) [dst_sqrt]",
                        "per tile ascending: diffusion (cells, 27, 3, 3) [explicit], logvol (cells, 27), curl (cells, "
                        "27, 3), logvol_closed27 (cells, 27)")                        # C14-7 (C05-2b)
PRIOR_NOISE_ORDER_CLOSED27 = ("diffusion (3 N_int,) [dst_sqrt, always]", "per tile ascending: logvol (cells, 27), "
                              "curl (cells, 27, 3), logvol_closed27 (cells, 27)")      # C15-10 (C05-2b)
POSTERIOR_FILES = ("sigma_u.npz", "sigma_logj.npz", "posterior_logj_cells.npz", "qc.json")
B5_3_NOTE = ("B5-3: PCG (x0 = 0, fixed preconditioner, stop on the true relative residual rtol) is odd in b, so "
             "E delta = 0; Cov(delta) - H^-1 = O(rtol) for fixed H, bounded by (2 rtol + rtol^2) ||H^-1||^2 tr(H), "
             "which is not informative at native size (tr H / lambda_min ~ 6e10); the actual bias is checked "
             "numerically (NB5).")
_RAISE = dict(over="raise", invalid="raise", divide="raise")


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LaplaceConfig:
    """C14-3/4/5 algorithm parameters (recorded; not model parameters)."""
    n_samples: int = S_NATIVE
    rtol: float = PCG_RTOL
    maxiter: int = PCG_MAXITER
    diffusion_factor: str = "dst_sqrt"
    metric_mode: str = "principal"
    pass_rate_threshold: float = PASS_RATE_THRESHOLD

    def __post_init__(self):
        object.__setattr__(self, "n_samples", integer_at_least(self.n_samples, "n_samples"))
        rtol = finite_real(self.rtol, "rtol", positive=True)
        if rtol >= 1:
            raise InputValidationError("rtol: must lie in (0, 1)")
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "maxiter", integer_at_least(self.maxiter, "maxiter"))
        if self.diffusion_factor not in DIFFUSION_FACTORS:
            raise InputValidationError(f"diffusion_factor: one of {DIFFUSION_FACTORS}")
        if self.metric_mode not in METRIC_MODES:
            raise InputValidationError(f"metric_mode: one of {METRIC_MODES}")
        threshold = finite_real(self.pass_rate_threshold, "pass_rate_threshold")
        if not 0 < threshold <= 1:
            raise InputValidationError("pass_rate_threshold: must lie in (0, 1]")
        object.__setattr__(self, "pass_rate_threshold", threshold)

    def record(self, state=None, *, logvol_points=None):
        """The record.  noise_order: that of `state`'s log-volume point set (`noise_order_of`, the source of
        LaplaceSystem's and PriorSystem's records: with logvol_closed27 under C05-2b, C14-7); without a state, that of
        the point set name `logvol_points` (`noise_order_of_points`; EBConfig.record, written before any state exists,
        passes the run's --logvol-points: FUP1, review MIN1 question 5); with neither, the C05-2 order NOISE_ORDER (the
        record before FUP1, byte for byte; review IMPL27b minor 1)."""
        if state is not None and logvol_points is not None:
            raise InputValidationError("LaplaceConfig.record: give state or logvol_points, not both")
        order = noise_order_of_points(logvol_points) if state is None else noise_order_of(state)
        return dict(contract="C14", n_samples=self.n_samples, pcg=dict(rtol=self.rtol, atol=0., maxiter=self.maxiter,
                    x0="zero", stop="true residual ||b - H x|| <= rtol ||b||",
                    preconditioner="C09.1 metric, gamma = 0, lam_b = 0, c_dat at u_hat"),
                    diffusion_factor=self.diffusion_factor, metric_mode=self.metric_mode,
                    pass_rate_threshold=self.pass_rate_threshold, noise_order=order,
                    hessian="GN H = sum_i K_i^T K_i: channels (sigma_used, w_used), diffusion, logvol, curl; "
                            "no proximal terms (C14-1)")


def seed_from_run_id(run_id):
    """C14-4: the sampling seed derived from the run id (first 8 bytes of sha256, 63 bits)."""
    if not isinstance(run_id, str) or not run_id:
        raise InputValidationError("run_id: nonempty string required")
    return int.from_bytes(hashlib.sha256(run_id.encode("utf-8")).digest()[:8], "little") >> 1


def hutchinson_seed_from_run_id(run_id):
    """The Hutchinson probe seed: seed_from_run_id(run_id + "/hutchinson"), a stream separate from the samples'
    (review M4 item 6)."""
    if not isinstance(run_id, str) or not run_id:
        raise InputValidationError("run_id: nonempty string required")
    return seed_from_run_id(run_id+"/hutchinson")


def state_hash(u):
    """sha256 of the float64 C-order bytes of a node field (C07-7 / C16 state_hash convention)."""
    return hashlib.sha256(np.ascontiguousarray(u, dtype=np.float64).tobytes(order="C")).hexdigest()


def noise_order_of(state, *, prior=False):
    """The noise order (C14-4; the prior side C15-9) of the state's log-volume point set, as a list: under C05-2b
    (state.logvol_closed27) the C14-7 / C15-10 order with logvol_closed27 last in every tile, else the C05-2 order.
    The one source of every noise_order record (LaplaceConfig.record, rng_record, logvol_noise_record, PriorSystem;
    review IMPL27b minor 1)."""
    if state.logvol_closed27:
        return list(PRIOR_NOISE_ORDER_CLOSED27 if prior else NOISE_ORDER_CLOSED27)
    return list(PRIOR_NOISE_ORDER if prior else NOISE_ORDER)


def noise_order_of_points(logvol_points=None, *, prior=False):
    """`noise_order_of` for a point set name before any state exists (EBConfig.record; FUP1, review MIN1 question 5):
    "gauss27+closed27" -> the C14-7 / C15-10 order (the state predicate logvol_closed27 is exactly this name); None or
    "gauss27" -> the C05-2 order.  Other names are refused: C14 / C15 do not cover them (_require_logvol_supported)."""
    if logvol_points == _model.LOGVOL_GAUSS27_CLOSED27:
        return list(PRIOR_NOISE_ORDER_CLOSED27 if prior else NOISE_ORDER_CLOSED27)
    if logvol_points in (None, _model.LOGVOL_GAUSS27):
        return list(PRIOR_NOISE_ORDER if prior else NOISE_ORDER)
    raise InputValidationError(f"logvol_points: None, 'gauss27' or 'gauss27+closed27' for a C14 noise order "
                               f"(got {logvol_points!r})")


def rng_record(rng, state=None, *, prior=False):
    """What provenance records of the random numbers (C14-4); validates rng.  noise_order: that of `state`
    (`noise_order_of`; prior: the C15-9 prior side), the C05-2 order NOISE_ORDER without a state."""
    order = list(NOISE_ORDER) if state is None else noise_order_of(state, prior=prior)
    versions = dict(numpy_version=np.__version__, noise_order=order)
    if isinstance(rng, np.random.Generator):
        return dict(seed=None, scheme="Generator.spawn(S): sample s uses child s",
                    bit_generator=type(rng.bit_generator).__name__, **versions)
    if isinstance(rng, (bool, np.bool_)) or not isinstance(rng, (int, np.integer)) or int(rng) < 0:
        raise InputValidationError("rng: a nonnegative integer seed or a numpy Generator required")
    return dict(seed=int(rng), scheme="numpy.random.default_rng(seed).spawn(S): sample s uses child s",
                bit_generator=type(np.random.default_rng(0).bit_generator).__name__, **versions)


def _generators(rng, count):
    """Per-sample child generators: default_rng(seed).spawn(count) or Generator.spawn(count)."""
    rng_record(rng)
    if isinstance(rng, np.random.Generator):
        return rng.spawn(count)
    return np.random.default_rng(int(rng)).spawn(count)


# ----------------------------------------------------------------------------------------------
# Square-root factor of the diffusion block and the trace operators
# ----------------------------------------------------------------------------------------------

def stiffness_sqrt_spectrum(grid):
    """sqrt(omega_ell Lambda_Q) on the interior DST-I modes (shape grid.interior_shape)."""
    with np.errstate(**_RAISE):
        root = np.sqrt(grid.omega*poisson_spectrum(grid, metric="q1"))
    require_finite_result(root, "K_Q square-root spectrum")
    return root


def stiffness_sqrt_apply(eps, grid, *, root=None):
    """K_Q^1/2 eps per component (free vector in, free vector out): S diag(sqrt(omega Lambda_Q)) S eps with S
    the orthonormal DST-I (symmetric, its own inverse), so (K_Q^1/2)^2 = K_Q = omega apply_poisson_operator."""
    theta = as_float64_array(eps, "eps", (3*grid.n_free_scalar,))
    root = stiffness_sqrt_spectrum(grid) if root is None else root
    fields = theta.reshape((3,)+grid.interior_shape)
    axes = (-3, -2, -1)
    with np.errstate(**_RAISE):
        out = idstn(root*dstn(fields, type=1, norm="ortho", axes=axes), type=1, norm="ortho", axes=axes)
    require_finite_result(out, "K_Q square root")
    return out.ravel(order="C")


def stiffness_apply_scaled(state, v):
    """K_Q v / omega0 (C15-1 A for alpha), free vector."""
    with np.errstate(**_RAISE):
        return stiffness_apply(v, state.grid)/state.omega0


def curl_gram_apply(state, v):
    """C^T W C v / omega0 with W = omega w_g at the Gauss27 points (C15-1 A for nu): nu times it is the curl
    block K_C^T K_C of H.  v: free vector or U0 field; returns the free vector."""
    grid, ops = state.grid, state.ops
    field_v = node_field(v, grid, name="v")
    out = np.zeros((3,)+grid.shape)
    with np.errstate(**_RAISE):
        weight = (ops.weights/state.omega0)[None, :, None]
    points = len(ops.points)
    for tile in cell_tiles(grid, state.tile_size):
        cells = tile.stop-tile.start
        _, dcurl, _ = tile_jvp(field_v, ops, tile=tile, A=np.zeros((cells, points, 3, 3)))
        with np.errstate(**_RAISE):
            gcurl = weight*dcurl
        tile_vjp_accumulate(out, ops, tile=tile, gcurl=gcurl)
    return finish(out)


def default_trace_operators(state):
    """The two C15-1 traces: A = K_Q/omega0 and C^T W C/omega0."""
    return {"KQ_over_omega0": lambda v: stiffness_apply_scaled(state, v),
            "CtWC_over_omega0": lambda v: curl_gram_apply(state, v)}


# ----------------------------------------------------------------------------------------------
# Perturbation right-hand side (numpy reference)
# ----------------------------------------------------------------------------------------------

def _require_logvol_supported(state):
    """The log-volume point sets C14 covers: 'gauss27' (C05-2) and 'gauss27+closed27' with the Simpson weights (C14-7,
    C05-2b).  Refused: closed27_weights = 'uniform' (a scope limit of C14-7 / C15-10: the premises of C14, C15 were
    checked for the Simpson weights only; uniform weights keep Pi_0 symmetric positive definite, but the closed27 rule
    is exact only on Q_{1,1,1}, so the class-constant fibre of C05-2b (iii) breaks by a quadrature artifact and the
    B17-5 spectral bound fails); 'gauss27+corners' (C05-2a: its noise and B_i are not in the contract, work package
    LVC)."""
    if state.logvol_points == "gauss27+closed27" and state.closed27_weights == "uniform":
        raise InputValidationError("Laplace (C14-7): closed27_weights = 'uniform' is refused (scope limit of C14-7 / "
                                   "C15-10: their premises were checked for the Simpson weights only; an ablation "
                                   "switch of the MAP solve, C05-2b (iv))")
    if state.logvol_points not in ("gauss27", "gauss27+closed27"):
        raise InputValidationError(f"Laplace (C14): logvol_points = {state.logvol_points!r} is not covered by the "
                                   "contract (C05-2a corner noise undefined); 'gauss27' or 'gauss27+closed27' "
                                   "(Simpson)")


def _closed27_stream(u, state, cotangents, draw):
    """The adjoint of the perturbation right-hand side under C05-2b (C14-7 / C15-10): tiles ascending; per tile the
    Gauss-point cotangents of cotangents(tile, geo) (which draw the tile's Gauss-point noise) scattered with state.ops,
    then the closed27 part K_{J,S}^T eps_S, eps_S = draw("logvol_closed27", (cells, 27)), scattered with
    state.closed27_ops (C06-7: the closed27 products right after the tile's Gauss-point products); the free vector."""
    grid = state.grid
    field = node_field(u, grid)
    ops_s = state.closed27_ops
    out = np.zeros((3,)+grid.shape)
    for tile in cell_tiles(grid, state.tile_size):
        geo = tile_geometry(field, state.ops, tile=tile)
        gJ, gcurl, ggrad = cotangents(tile, geo)
        tile_vjp_accumulate(out, state.ops, tile=tile, gJ=gJ, gcurl=gcurl, ggrad=ggrad, A=geo.A)
        geo_s = tile_geometry(field, ops_s, tile=tile)
        eps = _draw(draw, "logvol_closed27", geo_s.J.shape)
        gJ_s, _, _ = LogVolumeClosed27.cotangent(LogVolumeClosed27.scale(state, tile), geo_s, eps)
        tile_vjp_accumulate(out, ops_s, tile=tile, gJ=gJ_s, A=geo_s.A)
    return finish(out)


def _logvol_stream(u, state, cotangents, draw):
    """vjp_stream of the Gauss-point cotangents (C05-2: the code before C05-2b, bitwise) or, under C05-2b, with the
    closed27 part per tile (`_closed27_stream`)."""
    if state.logvol_closed27:
        return _closed27_stream(u, state, cotangents, draw)
    return vjp_stream(u, cotangents, state.ops, tile_size=state.tile_size)


def logvol_noise_record(state, *, prior=False):
    """What the system records of the log-volume point set: {} for 'gauss27' (records unchanged), else the point set,
    beta_S, the rule, the shares used and the noise order of C14-7 / C15-10."""
    if not state.logvol_closed27:
        return {}
    return dict(logvol_points=state.logvol_points, logvol_closed27_share=state.logvol_closed27_share,
                closed27_weights=state.closed27_weights, logvol_theta=list(state.logvol_theta),
                noise_order=noise_order_of(state, prior=prior))


def sigma_domain_noise_record(state):
    """C14 under sigma_domain support / support_shared (C04-6 "Laplace", C14 appended note; SWLI): H's channel blocks
    contain the native level's frozen support weights v (through beta); the channel noise keeps its N rows per channel
    and its order, and a row with v = 0 contributes exactly 0 to K_c^T eps; the posterior is conditional on v.  {} under
    brain (the record of the code before SWLI)."""
    if not state.has_support:
        return {}
    return dict(sigma_domain=state.sigma_domain, support_threshold=state.support_threshold,
                support_v_sha256=state.support_v_sha256, support_empty_channels=list(state.support_empty),
                channel_noise=("(C, n) per channel as under brain (same draws); rows with v_c = 0 have beta = 0 and add "
                               "exactly 0 (C04-6 Laplace); posterior conditional on v (B16 appended note)"))


def _draw(draw, name, shape):
    value = np.asarray(draw(name, tuple(int(s) for s in shape)))
    if value.dtype != np.float64 or value.shape != tuple(shape) or not np.all(np.isfinite(value)):
        raise InputValidationError(f"noise {name}: finite float64 array of shape {tuple(shape)} required")
    return value


def perturbation_rhs(state, ev, draw, *, diffusion="dst_sqrt", root=None):
    """C14-2: b = sum_i B_i eps_i at ev.u (the free vector), eps from draw(name, shape) in NOISE_ORDER.

    channel: K_c^T eps_c on the mask nodes (`channel_vjp`); diffusion: sqrt(alpha/omega0) K_Q^1/2 eps_D
    (dst_sqrt) or K_D^T eps_D (explicit); logvol, curl: K_J^T eps_J, K_C^T eps_C, one tile_vjp_accumulate
    per tile (the cotangents of bayesreg.model.blocks), never assembled."""
    if not isinstance(state, ModelState):
        raise InputValidationError("state: ModelState required")
    if not isinstance(ev, Evaluation) or ev.state is not state:
        raise InputValidationError("ev: an Evaluation of this ModelState required")
    _require_logvol_supported(state)
    if diffusion not in DIFFUSION_FACTORS:
        raise InputValidationError(f"diffusion: one of {DIFFUSION_FACTORS}")
    grid = state.grid
    b = channel_vjp(state, ev.channel, _draw(draw, "channel", ev.channel.r.shape))
    b_diffusion = None
    if diffusion == "dst_sqrt":
        eps = _draw(draw, "diffusion", (3*grid.n_free_scalar,))
        with np.errstate(**_RAISE):
            b_diffusion = np.sqrt(state.alpha/state.omega0)*stiffness_sqrt_apply(eps, grid, root=root)

    def cotangents(tile, geo):
        cells, points = geo.J.shape
        parts = []
        if diffusion == "explicit":
            eps = _draw(draw, "diffusion", (cells, points, 3, 3))
            parts.append(Diffusion.cotangent(Diffusion.scale(state, tile), geo, eps))
        eps = _draw(draw, "logvol", (cells, points))
        parts.append(LogVolume.cotangent(LogVolume.scale(state, tile), geo, eps))
        eps = _draw(draw, "curl", (cells, points, 3))
        parts.append(Curl.cotangent(Curl.scale(state, tile), geo, eps))
        return merge_cotangents(parts)

    gauss = _logvol_stream(ev.u, state, cotangents, draw)
    with np.errstate(**_RAISE):
        out = b+gauss if b_diffusion is None else b+b_diffusion+gauss
    require_finite_result(out, "perturbation right-hand side")
    return out


def prior_perturbation_rhs(state, ev, draw, *, root=None):
    """C15-9: b0 = sum_i K_{i,0}^T eps_i over the three prior blocks at u = 0 (ev: an Evaluation of `state` at
    u = 0), eps from draw(name, shape) in PRIOR_NOISE_ORDER: diffusion sqrt(alpha/omega0) K_Q^1/2 eps_D (the DST
    square root, always), then per tile logvol K_J^T eps_J, curl K_C^T eps_C (at u = 0: J = 1, cof F = I, so
    K_J = diag(sqrt(omega w_g/(omega0 tau_q^2))) D_div; under C05-2b with theta_G, and per tile after the curl noise the
    closed27 part logvol_closed27 (cells, 27) with K_{J,S} = diag(sqrt(c_p)/tau_p) D_{div,S}).  Cov(Pi_0^-1 b0) =
    Pi_0^-1."""
    if not isinstance(state, ModelState):
        raise InputValidationError("state: ModelState required")
    if not isinstance(ev, Evaluation) or ev.state is not state:
        raise InputValidationError("ev: an Evaluation of this ModelState required")
    _require_logvol_supported(state)
    if np.any(np.asarray(ev.u) != 0):
        raise InputValidationError("prior samples (C15-9): the evaluation must be at u = 0")
    grid = state.grid
    eps = _draw(draw, "diffusion", (3*grid.n_free_scalar,))
    with np.errstate(**_RAISE):
        b_diffusion = np.sqrt(state.alpha/state.omega0)*stiffness_sqrt_apply(eps, grid, root=root)

    def cotangents(tile, geo):
        cells, points = geo.J.shape
        parts = []
        eps = _draw(draw, "logvol", (cells, points))
        parts.append(LogVolume.cotangent(LogVolume.scale(state, tile), geo, eps))
        eps = _draw(draw, "curl", (cells, points, 3))
        parts.append(Curl.cotangent(Curl.scale(state, tile), geo, eps))
        return merge_cotangents(parts)

    gauss = _logvol_stream(ev.u, state, cotangents, draw)
    with np.errstate(**_RAISE):
        out = b_diffusion+gauss
    require_finite_result(out, "prior perturbation right-hand side")
    return out


# ----------------------------------------------------------------------------------------------
# PCG
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PCGResult:
    """x (n,) or (n, k); iterations (H applications of the recursion, the true-residual checks excluded),
    rel_residual (true ||b - H x|| / ||b||; 0 for b = 0) and converged, scalars or (k,) arrays."""
    x: np.ndarray
    iterations: object
    rel_residual: object
    converged: object


def _coldot(a, b):
    return np.einsum("ij,ij->j", a, b)


def _colnorm(a):
    return np.sqrt(_coldot(a, a))


def _selector(index, k):
    """Column selector: a view (slice) while every column is active, else the index array."""
    return slice(None) if index.size == k else index


def pcg(apply_H, apply_Minv, b, *, rtol=PCG_RTOL, maxiter=PCG_MAXITER):
    """C14-3 preconditioned CG from x0 = 0, atol = 0, for b (n,) or k right-hand sides as columns (n, k).

    apply_H, apply_Minv act on (n,) vectors for a 1-D b and on (n, j) column blocks for a 2-D b.  Each
    column stops when its true relative residual is <= rtol (one extra H application whenever the
    recursive residual meets rtol; if the true one does not, the column keeps iterating), or at maxiter.
    pHp <= 0 or a nonfinite scalar raises FloatingPointError (H must be SPD)."""
    rtol = finite_real(rtol, "rtol", positive=True)
    maxiter = integer_at_least(maxiter, "maxiter")
    b = as_float64_array(b, "b")
    if b.ndim not in (1, 2) or b.shape[0] < 1:
        raise InputValidationError("b: (n,) or (n, k) required")
    single = b.ndim == 1
    B = b[:, None] if single else b
    if single:
        H = lambda X: np.asarray(apply_H(X[:, 0]), dtype=np.float64)[:, None]      # noqa: E731
        Minv = lambda X: np.asarray(apply_Minv(X[:, 0]), dtype=np.float64)[:, None]    # noqa: E731
    else:
        H = lambda X: np.asarray(apply_H(X), dtype=np.float64)                       # noqa: E731
        Minv = lambda X: np.asarray(apply_Minv(X), dtype=np.float64)                  # noqa: E731
    n, k = B.shape
    with np.errstate(**_RAISE):
        bnorm = _colnorm(B)
        X = np.zeros((n, k))
        R = B.copy()
        P = np.zeros((n, k))
        rz = np.zeros(k)
        iterations = np.zeros(k, dtype=np.int64)
        rel = np.zeros(k)
        done = bnorm == 0
        active = np.flatnonzero(~done)
        if active.size:
            a = _selector(active, k)
            Z = Minv(R[:, a])
            P[:, a] = Z
            rz[a] = _coldot(R[:, a], Z)
        count = 0
        while active.size and count < maxiter:
            a = _selector(active, k)
            HP = H(P[:, a])
            pHp = _coldot(P[:, a], HP)
            if not np.all(np.isfinite(pHp)) or np.any(pHp <= 0):
                raise FloatingPointError("PCG: p^T H p <= 0 or nonfinite (H must be SPD, C06-5)")
            alpha = rz[a]/pHp
            X[:, a] += alpha*P[:, a]
            R[:, a] -= alpha*HP
            count += 1
            iterations[a] = count
            hit = _colnorm(R[:, a]) <= rtol*bnorm[a]
            if hit.any():
                cols = active[hit]
                true = _colnorm(B[:, cols]-H(X[:, cols]))/bnorm[cols]
                rel[cols] = true
                done[cols[true <= rtol]] = True
            still = active[~done[active]]
            if not still.size or count >= maxiter:
                active = still
                break
            s = _selector(still, k)
            Z = Minv(R[:, s])
            rz_new = _coldot(R[:, s], Z)
            beta = rz_new/rz[s]
            P[:, s] = Z+beta*P[:, s]
            rz[s] = rz_new
            active = still
        left = np.flatnonzero(~done)
        if left.size:
            rel[left] = _colnorm(B[:, left]-H(X[:, left]))/bnorm[left]
    require_finite_result(X, "PCG solution")
    if single:
        return PCGResult(X[:, 0], int(iterations[0]), float(rel[0]), bool(done[0]))
    return PCGResult(X, iterations, rel, done.copy())


# ----------------------------------------------------------------------------------------------
# The sampling system and the samples
# ----------------------------------------------------------------------------------------------

class LaplaceSystem:
    """H (GN, C06-5, no proximal terms), the C09.1 preconditioner (gamma = lam_b = 0) and the perturbation
    right-hand side at u_hat for one ModelState (C14-1/2/3)."""

    def __init__(self, state, u_hat, *, backend=None, config=None):
        if not isinstance(state, ModelState):
            raise InputValidationError("state: the host ModelState (sigma_used, w_used) required")
        _require_logvol_supported(state)
        self.config = LaplaceConfig() if config is None else config
        if not isinstance(self.config, LaplaceConfig):
            raise InputValidationError("config: LaplaceConfig required")
        self.backend = NumpyBackend() if backend is None else backend
        self.host = state
        self.u_hat = node_field(u_hat, state.grid, name="u_hat").copy()
        self.u_hat.flags.writeable = False
        self.state = self.backend.prepare(state)
        ev = self.backend.evaluate(self.state, self.u_hat)
        if isinstance(ev, InvalidTrial):
            raise InputValidationError(f"u_hat: F_B undefined ({ev.reason}); the output must be a certified state")
        self.ev = ev
        self.c_dat = float(self.backend.c_dat(self.state, ev))
        self.b_P = float(self.backend.metric_bP(self.state, mode=self.config.metric_mode))
        self.metric = self.backend.build_metric(self.state, c_dat=self.c_dat, b_P=self.b_P, lam_b=0., gamma=0.)
        self.n = 3*state.grid.n_free_scalar
        diffusion = self.config.diffusion_factor
        device_rhs = getattr(self.backend, "perturbation_rhs", None)
        if callable(device_rhs):
            self.rhs_source = "backend.perturbation_rhs"
            self._rhs = lambda draw: device_rhs(self.state, self.ev, draw, diffusion=diffusion)
        else:
            host_ev = ev if isinstance(ev, Evaluation) and ev.state is state else _model.evaluate(state, self.u_hat)
            if isinstance(host_ev, InvalidTrial):
                raise InputValidationError(f"u_hat: host F_B undefined ({host_ev.reason})")
            root = stiffness_sqrt_spectrum(state.grid) if diffusion == "dst_sqrt" else None
            self.rhs_source = "bayesreg.infer.laplace.perturbation_rhs (host numpy)"
            self._rhs = lambda draw: perturbation_rhs(state, host_ev, draw, diffusion=diffusion, root=root)

    def hvp(self, v):
        return np.asarray(self.backend.hvp(self.state, self.ev, v), dtype=np.float64)

    def precondition(self, r):
        return np.asarray(self.metric.solve(r), dtype=np.float64)

    def rhs(self, draw):
        out = np.asarray(self._rhs(draw), dtype=np.float64)
        if out.shape != (self.n,):
            raise InputValidationError(f"perturbation right-hand side: shape ({self.n},) required")
        return out

    def solve(self, b, *, rtol=None, maxiter=None):
        return pcg(self.hvp, self.precondition, b, rtol=self.config.rtol if rtol is None else rtol,
                   maxiter=self.config.maxiter if maxiter is None else maxiter)

    def record(self):
        backend_record = self.backend.record() if callable(getattr(self.backend, "record", None)) else {}
        return dict(contract="C14-1/2/3", c_dat=self.c_dat, b_P=self.b_P, metric_mode=self.config.metric_mode,
                    metric=dict(a=getattr(self.metric, "a", None), b=getattr(self.metric, "b", None), gamma=0.,
                                lam_b=0.), diffusion_factor=self.config.diffusion_factor,
                    rhs_source=self.rhs_source, n_free=self.n, tile_size=self.host.tile_size,
                    blocks=["channel"]*self.host.n_channels+["diffusion", "logvol", "curl"],
                    backend=backend_record.get("backend", type(self.backend).__name__),
                    **logvol_noise_record(self.host), **sigma_domain_noise_record(self.host))


class PriorSystem:
    """C15-9: Pi_0 (the GN precision of the three prior blocks at u = 0 under the state's eta), the C09.1
    preconditioner with c_dat = lam_b = gamma = 0 and the prior perturbation right-hand side; the same interface as
    LaplaceSystem for iter_perturbations / sample_perturbations (config.rtol / maxiter of C14-3)."""

    def __init__(self, state, *, backend=None, config=None):
        if not isinstance(state, ModelState):
            raise InputValidationError("state: the host ModelState of the pair (eta of this round) required")
        _require_logvol_supported(state)
        self.config = LaplaceConfig() if config is None else config
        if not isinstance(self.config, LaplaceConfig):
            raise InputValidationError("config: LaplaceConfig required")
        self.backend = NumpyBackend() if backend is None else backend
        self.host = state
        zero = np.zeros(state.grid.shape+(3,))
        zero.flags.writeable = False
        self.u0 = zero
        self.state = self.backend.prepare(state)
        ev = self.backend.evaluate(self.state, zero)
        if isinstance(ev, InvalidTrial):
            raise InputValidationError(f"u = 0: F_B undefined ({ev.reason})")
        self.ev = ev
        self.b_P = float(self.backend.metric_bP(self.state, mode=self.config.metric_mode))
        self.metric = self.backend.build_metric(self.state, c_dat=0., b_P=self.b_P, lam_b=0., gamma=0.)
        self.n = 3*state.grid.n_free_scalar
        host_ev = ev if isinstance(ev, Evaluation) and ev.state is state else _model.evaluate(state, zero)
        if isinstance(host_ev, InvalidTrial):
            raise InputValidationError(f"u = 0: host F_B undefined ({host_ev.reason})")
        root = stiffness_sqrt_spectrum(state.grid)
        self.rhs_source = "bayesreg.infer.laplace.prior_perturbation_rhs (host numpy)"
        self._rhs = lambda draw: prior_perturbation_rhs(state, host_ev, draw, root=root)

    def hvp(self, v):
        """Pi_0 v."""
        return np.asarray(self.backend.hvp(self.state, self.ev, v, blocks=PRIOR_BLOCKS), dtype=np.float64)

    def precondition(self, r):
        return np.asarray(self.metric.solve(r), dtype=np.float64)

    def rhs(self, draw):
        out = np.asarray(self._rhs(draw), dtype=np.float64)
        if out.shape != (self.n,):
            raise InputValidationError(f"prior perturbation right-hand side: shape ({self.n},) required")
        return out

    def solve(self, b, *, rtol=None, maxiter=None):
        return pcg(self.hvp, self.precondition, b, rtol=self.config.rtol if rtol is None else rtol,
                   maxiter=self.config.maxiter if maxiter is None else maxiter)

    def record(self):
        backend_record = self.backend.record() if callable(getattr(self.backend, "record", None)) else {}
        out = dict(contract="C15-9 prior side (Pi_0 at u = 0)", b_P=self.b_P, metric_mode=self.config.metric_mode,
                   metric=dict(a=getattr(self.metric, "a", None), b=getattr(self.metric, "b", None), c_dat=0.,
                               gamma=0., lam_b=0.), diffusion_factor="dst_sqrt (always for the prior side, C15-9)",
                   rhs_source=self.rhs_source, n_free=self.n, tile_size=self.host.tile_size,
                   blocks=list(PRIOR_BLOCKS), noise_order=noise_order_of(self.host, prior=True),
                   pcg=dict(rtol=self.config.rtol, atol=0., maxiter=self.config.maxiter, x0="zero",
                            stop="true residual ||b - Pi_0 x|| <= rtol ||b||"),
                   backend=backend_record.get("backend", type(self.backend).__name__))
        out.update(logvol_noise_record(self.host, prior=True))      # C15-10: the point set (same noise_order)
        return out


@dataclass(frozen=True)
class Perturbation:
    """One sample: delta (free vector, read-only) and its PCG record."""
    index: int
    delta: np.ndarray
    iterations: int
    rel_residual: float
    converged: bool
    rhs_norm: float
    seconds: float

    def record(self):
        return dict(index=self.index, pcg_iterations=self.iterations, pcg_rel_residual=self.rel_residual,
                    pcg_converged=self.converged, rhs_norm=self.rhs_norm, pcg_seconds=self.seconds)


def iter_perturbations(system, rng, n_samples=None):
    """Yield Perturbation s = 0..S-1 (one PCG each, C14-2/3); rng: an integer seed or a Generator."""
    count = system.config.n_samples if n_samples is None else integer_at_least(n_samples, "n_samples")
    for index, generator in enumerate(_generators(rng, count)):
        tick = time.perf_counter()
        b = system.rhs(lambda name, shape, g=generator: g.standard_normal(shape))
        result = system.solve(b)
        delta = np.array(result.x, dtype=np.float64, copy=True)
        delta.flags.writeable = False
        yield Perturbation(index, delta, result.iterations, result.rel_residual, result.converged,
                           float(np.linalg.norm(b)), time.perf_counter()-tick)


@dataclass(frozen=True)
class PerturbationSet:
    items: tuple
    system: dict
    rng: dict

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    @property
    def deltas(self):
        return np.stack([item.delta for item in self.items])

    @property
    def iterations(self):
        return np.array([item.iterations for item in self.items])

    @property
    def rel_residuals(self):
        return np.array([item.rel_residual for item in self.items])

    @property
    def converged(self):
        return np.array([item.converged for item in self.items])

    def record(self):
        it = self.iterations
        return dict(n_samples=len(self.items), n_converged=int(self.converged.sum()),
                    pcg_not_converged=int((~self.converged).sum()), iterations_min=int(it.min()),
                    iterations_median=float(np.median(it)), iterations_max=int(it.max()),
                    rel_residual_max=float(self.rel_residuals.max()), system=self.system, rng=self.rng,
                    samples=[item.record() for item in self.items])


def sample_perturbations(state, u_hat, rng, S=None, *, backend=None, config=None, system=None):
    """C14-2/3: S perturbations delta_s (H delta_s = sum_i B_i eps_i, PCG) at u_hat, with their PCG records.

    Holds all S deltas (S x 8 x 3 N_int bytes); at the native size stream iter_perturbations into
    posterior_maps instead (run_laplace does).
    The rng record (noise order, tile size) is taken from `system.host`, the ModelState the system's rhs draws the
    noise with (FUP1, review MIN1 minor 3): with a `system` built on another state object than `state`, the record
    still describes the draws (as system.record() does); with system=None, or a system built on `state`, it is the
    record as before."""
    system = LaplaceSystem(state, u_hat, backend=backend, config=config) if system is None else system
    count = system.config.n_samples if S is None else integer_at_least(S, "S")
    host = system.host
    record = dict(rng_record(rng, host, prior=isinstance(system, PriorSystem)), tile_size=host.tile_size)
    items = tuple(iter_perturbations(system, rng, count))
    return PerturbationSet(items, system.record(), record)


# ----------------------------------------------------------------------------------------------
# Trace estimators (C15-3 / B6)
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceEstimate:
    name: str
    method: str
    mean: float
    stderr: float
    n_used: int
    n_total: int
    values: tuple
    pcg: dict | None = None        # Hutchinson: the rtol / maxiter of its solves (review M4 item 6)

    def record(self):
        return dict(name=self.name, method=self.method, mean=self.mean, stderr=self.stderr, n_used=self.n_used,
                    n_total=self.n_total, pcg=self.pcg)


def _estimate(name, method, values, total, pcg=None):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values)) if values.size else float("nan")
    stderr = float(np.std(values, ddof=1)/np.sqrt(values.size)) if values.size >= 2 else float("nan")
    return TraceEstimate(name, method, mean, stderr, int(values.size), int(total), tuple(values.tolist()), pcg)


def sample_trace(perturbations, apply_A, *, name="A"):
    """B6 / C15-3: (1/S_conv) sum_s delta_s^T A delta_s over the PCG-converged samples."""
    values, total = [], 0
    for item in perturbations:
        total += 1
        if item.converged:
            values.append(float(item.delta @ np.asarray(apply_A(item.delta), dtype=np.float64)))
    return _estimate(name, "perturb_and_map_samples", values, total)


def hutchinson_trace(system, apply_A, rng, n_probes, *, name="A", rtol=None, maxiter=None):
    """C14 check of B6: tr(A H^-1) ~ mean z^T A H^-1 z over Rademacher probes z (PCG y = H^-1 z; probes whose
    PCG does not converge are excluded and counted).  Seed it with hutchinson_seed_from_run_id, not the sampling
    seed; the rtol / maxiter used (default: the system's) are recorded in the estimate."""
    count = integer_at_least(n_probes, "n_probes")
    used = dict(rtol=system.config.rtol if rtol is None else float(rtol),
                maxiter=system.config.maxiter if maxiter is None else int(maxiter))
    values = []
    for generator in _generators(rng, count):
        z = 2.*generator.integers(0, 2, size=system.n).astype(np.float64)-1.
        result = system.solve(z, rtol=used["rtol"], maxiter=used["maxiter"])
        if result.converged:
            values.append(float(z @ np.asarray(apply_A(result.x), dtype=np.float64)))
    return _estimate(name, "hutchinson_rademacher", values, count, pcg=dict(used, n_not_converged=count-len(values)))


# ----------------------------------------------------------------------------------------------
# Posterior maps (C14-5)
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PosteriorMaps:
    sigma_u: np.ndarray            # (nz, ny, nx, 3) mm, xyz
    sigma_u_mag: np.ndarray        # (nz, ny, nx) mm
    sigma_logj: np.ndarray         # cell grid (nz-1, ny-1, nx-1)
    logj_var_gauss: np.ndarray | None    # (n_cells, 27) ddof = 1 variance of log J_q over passed samples
    n_samples: int                 # S
    n_converged: int               # S_conv (PCG reached rtol)
    n_pcg_unconverged: int         # S - S_conv: not certified, in no statistic (C14-3)
    n_cert_passed: int             # S_cert: PCG converged and certificate PASS
    n_passed: int                  # S_pass: S_cert minus the NONPOSITIVE_GAUSS_JD samples (sigma_logJ)
    pass_rate: float               # S_cert / S_conv (NaN when S_conv = 0)
    pass_rate_low: bool | None     # pass_rate < threshold (None when S_conv = 0)
    samples: tuple                 # per-sample rows (PCG, certificate, inclusion)
    traces: dict                   # name -> TraceEstimate over the converged samples
    logj_cells: np.ndarray         # (n_passed, n_brain_cells) float32 cell-average log J
    logj_cell_index: np.ndarray    # (n_brain_cells,) flat C-order index into the cell grid
    logj_sample_index: np.ndarray  # (n_passed,) sample index
    u_hat_state_hash: str          # sha256 of u_hat's float64 C-order bytes
    qc: dict
    timing: dict
    logj_var_closed27: np.ndarray | None = None   # C05-2b (keep_gauss_variance): (n_cells, 27) at the closed27 points


def _brain_cells(mask):
    """Cells with at least one of their 8 vertices in the (node) mask, flat C-order of the cell grid."""
    m = np.asarray(mask, dtype=bool)
    out = np.zeros(tuple(s-1 for s in m.shape), dtype=bool)
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                out |= m[z:m.shape[0]-1+z, y:m.shape[1]-1+y, x:m.shape[2]-1+x]
    return out.ravel()


def _gauss_jacobians_positive(u, ops, tiles):
    """C07-5 (iii): every floating J_q > 0 at a node_field-validated u (streamed per tile; nothing kept)."""
    for tile in tiles:
        if not np.all(tile_geometry(u, ops, tile=tile).J > 0):
            return False
    return True


def _logvol_jacobians_positive(u, state, tiles):
    """C07-5 (iii) at the log-volume points of `state`: the Gauss points (C05-2: exactly _gauss_jacobians_positive)
    and, under C05-2b, the closed27 points.  Returns None when all are positive, else "gauss" or "closed27" (the
    first set with a floating J <= 0)."""
    if not _gauss_jacobians_positive(u, state.ops, tiles):
        return "gauss"
    if state.logvol_closed27 and not _gauss_jacobians_positive(u, state.closed27_ops, tiles):
        return "closed27"
    return None


def _iter_logj(u, ops, tiles):
    """(tile, log J_q of the tile), tile by tile; only after _gauss_jacobians_positive(u) holds."""
    for tile in tiles:
        J = tile_geometry(u, ops, tile=tile).J
        with np.errstate(**_RAISE):
            yield tile, np.log(J)


def _quantiles(values, qs):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.all(np.isfinite(values)):
        return [None for _ in qs]
    return [float(v) for v in np.quantile(values, qs)]


def _certificate_backend(backend, tile_size):
    """The backend's certificate configuration, checked against C07-2 with the run's recorded tile size T."""
    config = getattr(backend, "cert_config", None)
    if config is None or not callable(getattr(backend, "certify", None)):
        raise InputValidationError("backend: certify(u, grid) and cert_config (C07-2) required")
    check_certificate_config(config, integer_at_least(tile_size, "cert_tile_size"))
    return config


def posterior_maps(state, u_hat, perturbations, *, backend=None, config=None, trace_operators=None,
                   keep_cell_logj=True, keep_gauss_variance=False, log=None, expected_state_hash=None,
                   cert_tile_size=CERT_TILE_SIZE):
    """C14-5: certify every PCG-converged sample (an unconverged one is counted and skipped, C14-3), then
    sigma_u (converged samples), sigma_logJ and the cell log J (converged, certified, every J_q > 0), the pass
    rate S_cert / S_conv and its flag, sample traces (C15-3).  `expected_state_hash`: final_native.npz's
    state_hash, which u_hat must have; `cert_tile_size`: the run's recorded certificate tile size T (C07-2).
    Under C05-2b (C14-7) a sample needs every floating J > 0 at the closed27 points too; with keep_gauss_variance the
    ddof = 1 variance of log J at the closed27 points is returned as logj_var_closed27 (C15-10)."""
    if not isinstance(state, ModelState):
        raise InputValidationError("state: ModelState required")
    _require_logvol_supported(state)
    config = LaplaceConfig() if config is None else config
    backend = NumpyBackend() if backend is None else backend
    cert_config = _certificate_backend(backend, cert_tile_size)
    grid, ops = state.grid, state.ops
    u_hat = node_field(u_hat, grid, name="u_hat")
    digest = state_hash(u_hat)
    if expected_state_hash is not None and expected_state_hash != digest:
        raise InputValidationError(f"u_hat: state_hash {digest} differs from the expected {expected_state_hash} "
                                   "(the output state of the run is required, C14-1)")
    n_free = 3*grid.n_free_scalar
    points = len(ops.points)
    with np.errstate(**_RAISE):
        w_g = ops.weights/grid.omega
    tiles = list(cell_tiles(grid, state.tile_size))
    brain = _brain_cells(state.mask)
    brain_index = np.flatnonzero(brain)
    trace_operators = {} if trace_operators is None else dict(trace_operators)
    timing = dict(certificate_seconds=0., logj_seconds=0., rhs_pcg_seconds=0.)

    if _logvol_jacobians_positive(u_hat, state, tiles) is not None:
        raise InputValidationError("u_hat: a floating J <= 0 at a log-volume point (the output must be a certified "
                                   "state)")
    closed = state.logvol_closed27 and keep_gauss_variance        # C15-10: Var_s(log J) at the closed27 points
    hat_brain = [x[brain[tile]].ravel() for tile, x in _iter_logj(u_hat, ops, tiles)]
    hat_brain = np.concatenate(hat_brain) if hat_brain else np.zeros(0)
    q_logj = _quantiles(hat_brain, (.01, .99))
    del hat_brain                                   # not kept through the sample loop (memory, M4 item 2)

    sum_sq = np.zeros(n_free)
    mean = np.zeros((grid.n_cells, points))
    m2 = np.zeros((grid.n_cells, points))
    if closed:
        ops_s = state.closed27_ops
        mean_s = np.zeros((grid.n_cells, len(ops_s.points)))
        m2_s = np.zeros((grid.n_cells, len(ops_s.points)))
    trace_values = {name: [] for name in trace_operators}
    rows, cells, cell_samples = [], [], []
    n_total = n_conv = n_cert = n_pass = 0
    for item in perturbations:
        if not isinstance(item, Perturbation):
            raise InputValidationError("perturbations: Perturbation items required")
        delta = as_float64_array(item.delta, "delta", (n_free,))
        n_total += 1
        timing["rhs_pcg_seconds"] += item.seconds
        if not item.converged:
            # C14-3: counted, not certified, in no statistic (neither a certificate failure nor sigma_u)
            row = dict(item.record(), certificate_skipped=PCG_NOT_CONVERGED, cert_status=None, cert_reason=None,
                       cert_passed=None, cert_rejection=None, min_jd_lower_bound=None, n_uncertain=None,
                       n_infeasible=None, state_hash=None, cert_seconds=0., in_sigma_u=False, in_logj=False,
                       excluded=PCG_NOT_CONVERGED)
            rows.append(row)
            if log is not None:
                log(dict(event="posterior_sample", **row))
            continue
        n_conv += 1
        with np.errstate(**_RAISE):
            u_s = u_hat+inject_free(delta, grid)
            sum_sq += delta*delta
        for name, apply_A in trace_operators.items():
            trace_values[name].append(float(delta @ np.asarray(apply_A(delta), dtype=np.float64)))
        outcome = backend.certify(u_s, grid)
        timing["certificate_seconds"] += outcome.seconds
        n_cert += bool(outcome.passed)
        row = dict(item.record(), certificate_skipped=None, cert_status=outcome.status,
                   cert_reason=outcome.cert_reason, cert_passed=bool(outcome.passed), cert_rejection=outcome.rejection,
                   min_jd_lower_bound=outcome.min_jd_lower_bound, n_uncertain=outcome.n_uncertain,
                   n_infeasible=outcome.n_infeasible, state_hash=outcome.state_hash, cert_seconds=outcome.seconds,
                   in_sigma_u=True, in_logj=False, excluded=None)
        if not outcome.passed:
            row["excluded"] = f"logj: certificate {outcome.status} ({outcome.rejection})"
        else:
            tick = time.perf_counter()
            u_s = node_field(u_s, grid)
            failed = _logvol_jacobians_positive(u_s, state, tiles)
            if failed is not None:
                row["excluded"] = f"logj: {NONPOSITIVE_GAUSS_JD}" if failed == "gauss" else \
                    f"logj: {NONPOSITIVE_GAUSS_JD} (closed27 points, C05-2b)"
            else:
                n_pass += 1
                row["in_logj"] = True
                cell_parts = []
                with np.errstate(**_RAISE):
                    for tile, x in _iter_logj(u_s, ops, tiles):
                        d = x-mean[tile]
                        mean[tile] += d/n_pass
                        m2[tile] += d*(x-mean[tile])
                        if keep_cell_logj:
                            cell_parts.append((x@w_g)[brain[tile]])
                    if closed:
                        for tile, x in _iter_logj(u_s, ops_s, tiles):
                            d = x-mean_s[tile]
                            mean_s[tile] += d/n_pass
                            m2_s[tile] += d*(x-mean_s[tile])
                if keep_cell_logj:
                    part = np.concatenate(cell_parts) if cell_parts else np.zeros(0)
                    cells.append(part.astype(np.float32))
                    cell_samples.append(item.index)
            timing["logj_seconds"] += time.perf_counter()-tick
        rows.append(row)
        if log is not None:
            log(dict(event="posterior_sample", **row))
    if n_total == 0:
        raise InputValidationError("perturbations: at least one sample required")
    del mean
    var_s = None
    if closed:
        del mean_s
        with np.errstate(**_RAISE):
            if n_pass >= 2:
                m2_s /= n_pass-1
                var_s = m2_s
            else:
                var_s = np.full(m2_s.shape, np.nan)

    with np.errstate(**_RAISE):
        if n_conv:
            var = inject_free(sum_sq/n_conv, grid)
            sigma_u = np.sqrt(var)
            sigma_mag = np.sqrt(var[..., 0]+var[..., 1]+var[..., 2])
        else:
            sigma_u = np.full(grid.shape+(3,), np.nan)
            sigma_mag = np.full(grid.shape, np.nan)
        sigma_flat = np.full(grid.n_cells, np.nan)
        var_g = None
        if n_pass >= 2:
            for tile in tiles:                      # per tile: no full-size temporaries (memory, M4 item 2)
                sigma_flat[tile] = np.sqrt(m2[tile]/(n_pass-1))@w_g
            if keep_gauss_variance:
                m2 /= n_pass-1
                var_g = m2
        elif keep_gauss_variance:
            var_g = np.full(m2.shape, np.nan)
        sigma_logj = sigma_flat.reshape(grid.cell_shape)
    del m2
    n_unconv = n_total-n_conv
    pass_rate = n_cert/n_conv if n_conv else float("nan")
    low = bool(pass_rate < config.pass_rate_threshold) if n_conv else None
    traces = {name: _estimate(name, "perturb_and_map_samples", values, n_total)
              for name, values in trace_values.items()}
    mag_brain = sigma_mag[state.mask]
    q_mag = _quantiles(mag_brain, (.5, .99))
    qc = dict(contract="C14-5 / C16", pass_rate=pass_rate, pass_rate_threshold=config.pass_rate_threshold,
              posterior_cert_pass_rate_low=low, n_samples=n_total, S_conv=n_conv, S_cert=n_cert, S_pass=n_pass,
              n_certificate_passed=n_cert, pcg_unconverged=n_unconv, pcg_not_converged=n_unconv,
              n_nonpositive_gauss_jd=n_cert-n_pass, usable_fraction=n_pass/n_total,
              pass_rate_definition="S_cert / S_conv: certificate PASS among the PCG-converged samples; a sample "
                                   "whose PCG did not reach rtol is pcg_not_converged, counted (pcg_unconverged), "
                                   "not certified and in no statistic (C14-3); NaN and flag None when S_conv = 0",
              usable_fraction_definition="S_pass / S; S_pass = PCG converged, certificate PASS and every floating "
                                         "J_q > 0 (the samples of sigma_logJ)",
              u_hat_state_hash=digest, sigma_u_mag_brain_median=q_mag[0], sigma_u_mag_brain_q99=q_mag[1],
              sigma_u_definition="sqrt((1/S_conv) sum_s delta_s^2) over the PCG-converged samples "
                                 "(mean u_hat known), mm",
              sigma_logj_definition="ddof = 1 std of log J_q over the S_pass samples at each Gauss point, "
                                    "then sum_g w_g (cell grid); NaN if S_pass < 2",
              logj_hat_brain_q01=q_logj[0], logj_hat_brain_q99=q_logj[1],
              logj_hat_brain_definition="log J_q of u_hat at the Gauss points of the cells with >= 1 mask vertex",
              pcg_bias_note=B5_3_NOTE, semantics_note="B16: linearized Gaussian posterior; it ignores the support "
              "C and the nonlinearity of log J; the log J statistics are truncated to certified samples (B5-4) "
              "and biased; calibrated for predictive power, not frequentist coverage.",
              certificate=certificate_record(cert_config), config=config.record(state),
              traces={name: t.record() for name, t in traces.items()})
    if state.logvol_closed27:
        qc.update(logvol_points=state.logvol_points, logvol_closed27_share=state.logvol_closed27_share,
                  closed27_weights=state.closed27_weights,
                  logvol_closed27_note="C14-7: H and the noise contain the closed27 block; sigma_logJ, the cell log J "
                                       "and the u_hat log J quantiles stay on the Gauss points (C14-5); a sample "
                                       "enters them only with every floating J > 0 at the Gauss and closed27 points")
    if keep_cell_logj and cells:
        logj_cells = np.stack(cells)
    else:
        logj_cells = np.zeros((0, brain_index.size), dtype=np.float32)
    return PosteriorMaps(sigma_u=sigma_u, sigma_u_mag=sigma_mag, sigma_logj=sigma_logj, logj_var_gauss=var_g,
                         n_samples=n_total, n_converged=n_conv, n_pcg_unconverged=n_unconv, n_cert_passed=n_cert,
                         n_passed=n_pass, pass_rate=pass_rate, pass_rate_low=low, samples=tuple(rows), traces=traces,
                         logj_cells=logj_cells, logj_cell_index=brain_index.astype(np.int64),
                         logj_sample_index=np.asarray(cell_samples, dtype=np.int64), u_hat_state_hash=digest, qc=qc,
                         timing=timing, logj_var_closed27=var_s)


# ----------------------------------------------------------------------------------------------
# Output files (C16) and the one-call driver
# ----------------------------------------------------------------------------------------------

def write_posterior_outputs(out_dir, maps, grid, *, extra_qc=None):
    """sigma_u.npz, sigma_logj.npz, posterior_logj_cells.npz, qc.json in out_dir; none may exist (all four are
    checked before the first is written, review M4 item 7)."""
    out_dir = Path(out_dir)
    paths = {name: out_dir/name for name in POSTERIOR_FILES}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        raise FileExistsError(f"posterior outputs exist (nothing is overwritten): {existing}")
    qc = dict(maps.qc, samples=list(maps.samples), timing=maps.timing)
    if extra_qc:
        clash = set(extra_qc) & set(qc)
        if clash:
            raise InputValidationError(f"extra_qc: keys {sorted(clash)} are written by the posterior step")
        qc.update(extra_qc)
    record = json.dumps(grid_record(grid))
    digest = np.array(maps.u_hat_state_hash)
    np.savez(paths["sigma_u.npz"], sigma_u=maps.sigma_u, sigma_u_mag=maps.sigma_u_mag, grid=np.array(record),
             S_conv=np.array(maps.n_converged), u_hat_state_hash=digest,
             layout=np.array("(nz, ny, nx, 3) mm, components xyz; node grid"))
    np.savez(paths["sigma_logj.npz"], sigma_logj=maps.sigma_logj, grid=np.array(record),
             S_pass=np.array(maps.n_passed), u_hat_state_hash=digest,
             layout=np.array("cell grid (nz-1, ny-1, nx-1), cell centers at node origin + h/2"))
    np.savez(paths["posterior_logj_cells.npz"], logj=maps.logj_cells, cell_index=maps.logj_cell_index,
             sample_index=maps.logj_sample_index, grid=np.array(record), u_hat_state_hash=digest,
             layout=np.array("logj[s, k]: float32 sum_g w_g log J_q of cell cell_index[k] (flat C-order of the "
                             "cell grid, cells with >= 1 mask vertex) for the passed sample sample_index[s]"))
    write_json(paths["qc.json"], qc)
    return qc


def run_laplace(out_dir, state, u_hat, *, seed, backend=None, config=None, trace_operators=None, log=None,
                extra_qc=None, expected_state_hash=None, cert_tile_size=CERT_TILE_SIZE):
    """C14 end to end: system at u_hat, S streamed perturbations (seed per C14-4), posterior maps, outputs.
    `expected_state_hash`: final_native.npz's state_hash (u_hat must be that state); `cert_tile_size`: the run's
    recorded certificate tile size T."""
    config = LaplaceConfig() if config is None else config
    tick = time.perf_counter()
    record = dict(rng_record(seed, state), tile_size=state.tile_size)
    if record["seed"] is None:
        raise InputValidationError("seed: an integer (C14-4: derived from run_id and recorded) required")
    out_dir = Path(out_dir)
    existing = [name for name in POSTERIOR_FILES if (out_dir/name).exists()]
    if existing:
        raise FileExistsError(f"{out_dir}: posterior outputs {existing} exist (nothing is overwritten)")
    if expected_state_hash is not None and state_hash(node_field(u_hat, state.grid, name="u_hat")) != expected_state_hash:
        raise InputValidationError("u_hat: state_hash differs from the expected final_native state_hash (C14-1)")
    system = LaplaceSystem(state, u_hat, backend=backend, config=config)
    maps = posterior_maps(state, system.u_hat, iter_perturbations(system, seed, config.n_samples),
                          backend=system.backend, config=config, trace_operators=trace_operators, log=log,
                          expected_state_hash=expected_state_hash, cert_tile_size=cert_tile_size)
    maps = replace(maps, timing=dict(maps.timing, total_seconds=time.perf_counter()-tick))
    extra = dict(extra_qc or {}, laplace_system=system.record(), laplace_rng=record)
    write_posterior_outputs(out_dir, maps, state.grid, extra_qc=extra)
    return maps
