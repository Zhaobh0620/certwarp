"""ModelState: what F_B holds fixed during one inner solve (contract C01-4, C04, C05, C06-1, C08-7).

One level ell of one pair: the level grid and its Gauss27 operators, the native cell volume omega0,
the 1 + K_S likelihood channels stacked on the first axis (c = 0 the intensity I, c = 1..K_S the
structure channels s^k; fixed side as node arrays g_c, moving side as one not-a-knot cubic spline
G_c per channel built by the frozen `interpolation.build_image_sampler`, C03.5-10), the fixed brain
mask m, sigma_c^2, the Student-t weights w_c (None = Gaussian, w = 1), eta = (alpha, nu, tau_k^2)
and the floored tissue-layer posteriors Ptilde_{T,F} (K_T + 1 channels, background first) that give
tau^2 (C05-2).  sigma, w and eta are frozen for one (level, EM round); `replace` builds the next one
with the same samplers.
cpu_threads (default 4, geom.parallel.DEFAULT_CPU_THREADS) is the thread count of the CPU reference (bayesreg.geom.parallel, work package
CPUPAR): every value is bitwise the same for every count, so it is recorded but is not part of the model
identity of the C08-8(d) cache key.
logvol_points chooses the point set of the log-volume prior.  Default (work package DEF, 2026-10-31; contract C05-2b
first paragraph, D43, D48; the lead's decision after the GPU smoke v2, runs/20261030-m2-gpu-smoke-v2/result.md):
"gauss27+closed27" with beta_S = 1 and the Simpson rule (LOGVOL_DEFAULT).  "gauss27" (C05-2, the default before DEF)
and "gauss27+corners" (C05-2a; work package LVC) are explicit options, each bitwise the code before DEF.
"gauss27" = C05-2 bit for bit; "gauss27+corners" adds, for every cell K and corner v, the residual
sqrt(beta_c omega/(8 omega0)) log J_{K,v}/tau_{K,v} (J of K's own Q1 map at v, tau^2 the node value at v),
beta_c = corner_weight (default 1).  Both change F_B, so both are part of the model identity.
"gauss27+closed27" (C05-2b, BZ adopted (A') 10-30; work package IMPL27; the default since DEF) puts the
log-volume prior on Gauss27 U closed27 by shares: theta_G = 1/(1 + beta_S) of the Gauss27 rule, theta_S =
beta_S/(1 + beta_S) of a rule on the 27 points {0, 1/2, 1}^3 of every cell (J of K's own Q1 map there, shared points
not averaged; tau^2 the Q1 interpolation of tau2_nodes, as at the Gauss points); beta_S = logvol_closed27_share
(default 1), the closed27 rule closed27_weights = "simpson" (default, D48: s_p = sigma_i sigma_j sigma_k, sigma =
(1/6, 2/3, 1/6)) or "uniform" (1/27 per point, an explicit ablation switch; Laplace and EB refuse it: a scope limit
of C14-7 / C15-10).  Floating-point shares and weights (contract C05-2b "份额与权的浮点算式", AC27b; IMPL27b):
    theta_G = 1.0/(1.0 + beta_S),  theta_S = beta_S/(1.0 + beta_S)     one float64 division each (theta_S is not
                                                                       1 - theta_G)
    Gauss27 part   logvol_gauss_weights   = theta_G * ops.weights      (omega w_g theta_G, one multiplication each)
    closed27 part  closed27_point_weights = theta_S * W27              (omega s_p theta_S)
with W27 = geom.simpson27_weights(grid) (omega * x * y * z in jd_curl's product order) or, uniform,
CellQ1Operators3D(grid, "closed27_uniform").weights (omega/27).  theta_G and theta_S are stored (`logvol_theta`) and
recorded as the multipliers actually used.  For beta_S = 1, 3 theta_G = .5, .25 are powers of 2 (no rounding in w_G).
With "gauss27" and "gauss27+corners", logvol_gauss_weights IS ops.weights (the same array: bitwise the code before).
The three options are exclusive (one string); beta_S and closed27_weights are part of the model identity.

sigma_domain (contract C04-6, C11-8; work package SWLI; draft SWL 2026-10-31, D49/D50 for BZ).  "brain" (default):
support_threshold and support_v must be None and every array below is bitwise that of the code before SWLI.
"support" / "support_shared": support_v is the level's frozen support weight v, a (K_S, n) uint8 array of 0/1 on the mask
nodes (rows c = 1..K_S, node_index C-order; bayesreg.model.support), held as its own field -- never through the Student-t
`weights` (weights stays None, provenance noise "gaussian").  v joins beta at construction (C06-8): beta_c^2 = q m w v_c /
(omega0 sigma_c^2) for c >= 1, formed as the brain beta^2 times v in place (exact: v in {0, 1}); the c_dat weight is
w/sigma^2 times v (blocks.channel_cdat_weight).  A channel with an empty support has beta_c = 0 and no likelihood; its
sigma_c^2 is internal only (sigma2_record() gives null).  support_threshold theta (D50) must satisfy
eps/(1 + (K_S + 1) eps) < theta < 1 (support.check_theta, K_S = n_channels - 1).  sigma_domain, theta and the sha256 of v
are part of the model identity (C08-8(d')).

Derived once here (the mask nodes are the only nodes with nonzero channel weight):
  node_index (n,)   flat C-order indices of the mask nodes; every channel array is (C, n) on them;
  beta (C, n)       sqrt(q_a m_a w_c(a) / (omega0 sigma_c^2)) (C06-2), m_a = 1 on the mask;
  tau2_nodes        sum_k Ptilde_{T,F}^k tau_k^2 at the nodes.  Q1 interpolation is linear, so its
                    Gauss-point interpolation equals sum_k tau_k^2 Ptilde^k(x_q) of C05-2 (one
                    interpolation per tile instead of K_T + 1); it reads only the tissue layer.
"""
from contextvars import ContextVar
from dataclasses import dataclass, field, replace

import numpy as np

from jd_curl.validation import InputValidationError, as_float64_array, finite_real, readonly_copy
from jd_curl.three_d.grid import Grid3D
from jd_curl.three_d.interpolation import ImageSampler3D, build_image_sampler
from jd_curl.three_d.validation import integer_at_least

from ..readonly import adopt_readonly, freeze_new
from ..geom import TILE_SIZE, closed27, corners8, gauss27, scalar_field, simpson27_weights
from ..geom.parallel import DEFAULT_CPU_THREADS, cpu_threads as _cpu_threads, provenance as _parallel_provenance
from .support import BRAIN, SIGMA_DOMAINS, SUPPORT_RULE, SUPPORT_SHARED, check_theta, v_sha256

SIMPLEX_SUM_TOL = 1e-12        # C03 gate on Ptilde_T at the nodes
LOGVOL_GAUSS27 = "gauss27"                     # C05-2 (the default before DEF; explicit option, bitwise unchanged)
LOGVOL_GAUSS27_CORNERS = "gauss27+corners"     # C05-2a (explicit option; kept as the comparison arm)
LOGVOL_GAUSS27_CLOSED27 = "gauss27+closed27"   # C05-2b (BZ adopted (A') 10-30; the default since DEF, 2026-10-31)
LOGVOL_DEFAULT = LOGVOL_GAUSS27_CLOSED27       # DEF: C05-2b first paragraph, D43, D48 (beta_S 1, Simpson)
LOGVOL_POINTS = (LOGVOL_GAUSS27, LOGVOL_GAUSS27_CORNERS, LOGVOL_GAUSS27_CLOSED27)
CLOSED27_SIMPSON = "simpson"                   # D48 default
CLOSED27_UNIFORM = "uniform"                   # explicit ablation switch (Laplace / EB refuse it)
CLOSED27_WEIGHTS = (CLOSED27_SIMPSON, CLOSED27_UNIFORM)
LONGDOUBLE_MIN_NMANT = 63      # C08-7: x86-64 80-bit extended precision
_DONOR = ContextVar("bayesreg_model_state_donor", default=None)     # MEM: the state `replace` derives from


def require_extended_longdouble(nmant=None):
    """C08-7: refuse to run unless np.longdouble has >= 63 mantissa bits (no silent float64 fallback)."""
    nmant = np.finfo(np.longdouble).nmant if nmant is None else nmant
    if nmant < LONGDOUBLE_MIN_NMANT:
        raise InputValidationError(f"C08-7: np.longdouble has {nmant} mantissa bits < {LONGDOUBLE_MIN_NMANT}; "
                                   "the CPU reference refuses to run")


def build_samplers(moving, grid):
    """C03.5-10: one `build_image_sampler` per moving-side channel (moving: (C, nz, ny, nx))."""
    moving = as_float64_array(moving, "moving")
    if moving.ndim != 4 or moving.shape[1:] != grid.shape or moving.shape[0] < 1:
        raise InputValidationError(f"moving: expected (C>=1,) + {grid.shape}")
    return tuple(build_image_sampler(channel, grid) for channel in moving)


@dataclass(frozen=True, eq=False)
class ModelState:
    grid: Grid3D
    omega0: float                 # native cell volume (C01-4)
    fixed: np.ndarray             # (C, nz, ny, nx) g_c
    samplers: tuple               # C moving-side ImageSampler3D on `grid`
    mask: np.ndarray              # (nz, ny, nx) bool, m_a (C03.3, level mask)
    sigma2: np.ndarray            # (C,) sigma_c^2 > 0
    alpha: float                  # diffusion precision (C05-1)
    nu: float                     # curl precision (C05-3)
    tau2: np.ndarray              # (K_T + 1,) tau_k^2 > 0, k = 0 background (C05-2)
    tissue: np.ndarray            # (K_T + 1, nz, ny, nx) Ptilde_{T,F} on this level (C03.5/C03.6)
    weights: np.ndarray | None = None   # (C, nz, ny, nx) Student-t w_c (C04-3); None = Gaussian
    tile_size: int = TILE_SIZE    # cells per tile of the Gauss-point sums (C06-7, recorded)
    cpu_threads: int = DEFAULT_CPU_THREADS   # threads of the CPU reference (CPUPAR; bitwise neutral, recorded)
    logvol_points: str = LOGVOL_DEFAULT      # C05-2 / C05-2a / C05-2b point set of the log-volume prior (model
    #                                          identity); default gauss27+closed27 since DEF (C05-2b, D48)
    corner_weight: float = 1.             # beta_c of C05-2a (only with "gauss27+corners"; model identity)
    logvol_closed27_share: float = 1.     # beta_S of C05-2b (only with "gauss27+closed27"; model identity)
    closed27_weights: str = CLOSED27_SIMPSON   # D48 closed27 rule (only with "gauss27+closed27"; model identity)
    sigma_domain: str = BRAIN             # C04-6 (SWLI): "brain" (default) | "support" | "support_shared"
    support_threshold: float | None = None   # theta of C04-6 / D50 (None under "brain")
    support_v: np.ndarray | None = None   # (K_S, n) uint8 frozen support weights v (None under "brain")
    ops: object = field(init=False, repr=False)
    corner_ops: object = field(init=False, repr=False)       # corners8 operators (None unless "gauss27+corners")
    corner_weights: np.ndarray = field(init=False, repr=False)   # (8,) beta_c omega/8 (None unless corners)
    closed27_ops: object = field(init=False, repr=False)     # closed27 operators (None unless "gauss27+closed27")
    closed27_point_weights: np.ndarray = field(init=False, repr=False)   # (27,) omega s_p theta_S (None unless closed27)
    logvol_gauss_weights: np.ndarray = field(init=False, repr=False)     # (27,) omega w_g (x theta_G under closed27)
    logvol_theta: tuple = field(init=False, repr=False)       # (theta_G, theta_S) used (None unless closed27)
    node_index: np.ndarray = field(init=False, repr=False)
    strict: np.ndarray = field(init=False, repr=False)
    points0: np.ndarray = field(init=False, repr=False)
    fixed_masked: np.ndarray = field(init=False, repr=False)
    weights_masked: np.ndarray = field(init=False, repr=False)
    # (C, n) w_c on the mask nodes, read-only.  Gaussian (weights None): a read-only BROADCAST VIEW of one 1.0
    # (np.broadcast_to, strides (0, 0); MEM), not a contiguous array: elementwise use only (w / sigma2[:, None] etc.
    # allocate their own C-contiguous result); a new user that reduces over w directly (the memory layout differs from
    # a contiguous array's), needs contiguous memory (torch.from_numpy, numba C-layout signatures, .view) or writes in
    # place must take np.ascontiguousarray first (FUP1, review MEM minor 2).  Student-t: a C-contiguous read-only copy.
    beta: np.ndarray = field(init=False, repr=False)
    tau2_nodes: np.ndarray = field(init=False, repr=False)
    support_empty: tuple = field(init=False, repr=False)       # structure channels (1..K_S) with v_c = 0 (C04-6)
    support_v_sha256: str | None = field(init=False, repr=False)   # sha256 of v (None under "brain")

    def __post_init__(self):
        require_extended_longdouble()
        grid = self.grid
        if not isinstance(grid, Grid3D):
            raise InputValidationError("grid: Grid3D required")
        shape = grid.shape
        put = lambda name, value: object.__setattr__(self, name, value)   # noqa: E731
        put("omega0", finite_real(self.omega0, "omega0", positive=True))
        fixed = as_float64_array(self.fixed, "fixed")
        if fixed.ndim != 4 or fixed.shape[1:] != shape or fixed.shape[0] < 1:
            raise InputValidationError(f"fixed: expected (C>=1,) + {shape}")
        n_channels = fixed.shape[0]
        put("fixed", adopt_readonly(fixed))
        samplers = tuple(self.samplers)
        if len(samplers) != n_channels or not all(isinstance(s, ImageSampler3D) and s.grid.grid_id == grid.grid_id
                                                   for s in samplers):
            raise InputValidationError(f"samplers: {n_channels} ImageSampler3D on this grid required (build_samplers)")
        put("samplers", samplers)
        mask = np.asarray(self.mask)
        if mask.dtype != bool or mask.shape != shape or not mask.any():
            raise InputValidationError("mask: nonempty boolean array of the grid shape required")
        put("mask", adopt_readonly(mask, bool))
        sigma2 = as_float64_array(self.sigma2, "sigma2", (n_channels,))
        if np.any(sigma2 <= 0):
            raise InputValidationError("sigma2: every sigma_c^2 must be positive")
        put("sigma2", readonly_copy(sigma2))
        for name in ("alpha", "nu"):
            value = finite_real(getattr(self, name), name)
            if value < 0:
                raise InputValidationError(f"{name}: must be nonnegative")
            put(name, value)
        tau2 = as_float64_array(self.tau2, "tau2")
        if tau2.ndim != 1 or tau2.size < 2 or np.any(tau2 <= 0):
            raise InputValidationError("tau2: (K_T + 1 >= 2,) positive class variances required")
        put("tau2", readonly_copy(tau2))
        tissue = as_float64_array(self.tissue, "tissue", (tau2.size,) + shape)
        if np.any(tissue < 0) or float(np.max(np.abs(tissue.sum(axis=0)-1))) > SIMPLEX_SUM_TOL:
            raise InputValidationError(f"tissue: a (K_T + 1)-class simplex at every node (sum 1 within "
                                       f"{SIMPLEX_SUM_TOL:g}) required (C03 gate)")
        put("tissue", adopt_readonly(tissue))
        if self.weights is not None:
            weights = as_float64_array(self.weights, "weights", fixed.shape)
            if np.any(weights < 0):
                raise InputValidationError("weights: Student-t weights must be nonnegative")
            put("weights", adopt_readonly(weights))
        put("tile_size", integer_at_least(self.tile_size, "tile_size"))
        put("cpu_threads", _cpu_threads(self.cpu_threads))
        if self.logvol_points not in LOGVOL_POINTS:
            raise InputValidationError(f"logvol_points: one of {LOGVOL_POINTS} (C05-2, C05-2a, C05-2b)")
        put("corner_weight", finite_real(self.corner_weight, "corner_weight", positive=True))
        if self.logvol_points != LOGVOL_GAUSS27_CORNERS and self.corner_weight != 1.:
            raise InputValidationError("corner_weight: beta_c applies only to logvol_points = 'gauss27+corners' "
                                       "(C05-2a)")
        put("logvol_closed27_share", finite_real(self.logvol_closed27_share, "logvol_closed27_share", positive=True))
        if self.closed27_weights not in CLOSED27_WEIGHTS:
            raise InputValidationError(f"closed27_weights: one of {CLOSED27_WEIGHTS} (D48)")
        if self.logvol_points != LOGVOL_GAUSS27_CLOSED27 and (self.logvol_closed27_share != 1.
                                                              or self.closed27_weights != CLOSED27_SIMPSON):
            raise InputValidationError("logvol_closed27_share / closed27_weights apply only to logvol_points = "
                                       "'gauss27+closed27' (C05-2b)")

        put("ops", gauss27(grid))
        put("logvol_gauss_weights", self.ops.weights)             # the same array: bitwise the code before C05-2b
        put("corner_ops", None)
        put("corner_weights", None)
        put("closed27_ops", None)
        put("closed27_point_weights", None)
        put("logvol_theta", None)
        if self.logvol_points == LOGVOL_GAUSS27_CORNERS:
            corner_ops = corners8(grid)
            with np.errstate(over="raise", invalid="raise"):
                corner_weights = self.corner_weight*corner_ops.weights          # beta_c omega/8, (8,)
            put("corner_ops", corner_ops)
            put("corner_weights", readonly_copy(corner_weights))
        elif self.logvol_points == LOGVOL_GAUSS27_CLOSED27:
            closed27_ops = closed27(grid)
            share = self.logvol_closed27_share
            rule = simpson27_weights(grid) if self.closed27_weights == CLOSED27_SIMPSON else closed27_ops.weights
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                theta_G, theta_S = 1.0/(1.0+share), share/(1.0+share)         # C05-2b: one division each
                gauss = theta_G*self.ops.weights                              # omega w_g theta_G
                points = theta_S*rule                                         # omega s_p theta_S
            put("logvol_theta", (float(theta_G), float(theta_S)))
            put("closed27_ops", closed27_ops)
            put("closed27_point_weights", readonly_copy(points))
            put("logvol_gauss_weights", readonly_copy(gauss))
        # MEM (memory only): under `replace` the arrays that depend only on inputs shared with the donor state (the
        # same grid, mask, fixed, weights, tissue objects and equal tau2) are the donor's own read-only arrays -- the
        # very values the expressions below would compute again; beta (sigma2) is always computed.
        donor = _DONOR.get()
        same_nodes = donor is not None and donor.grid is grid and donor.mask is self.mask
        if same_nodes:
            index = donor.node_index
            put("node_index", index)
            put("strict", donor.strict)
            put("points0", donor.points0)
        else:
            index = np.flatnonzero(mask.ravel())
            interior = np.zeros(shape, dtype=bool)
            interior[1:-1, 1:-1, 1:-1] = True
            put("node_index", _readonly(index))
            put("strict", _readonly(interior.ravel()[index]))
            put("points0", freeze_new(grid.identity().reshape(-1, 3)[index]))
        if same_nodes and donor.fixed is self.fixed:
            put("fixed_masked", donor.fixed_masked)
        else:
            put("fixed_masked", freeze_new(self.fixed.reshape(n_channels, -1)[:, index]))
        if same_nodes and donor.weights is self.weights and donor.weights_masked.shape == (n_channels, index.size):
            w = donor.weights_masked
        elif self.weights is None:
            # MEM (memory only): the Gaussian w = 1 as a read-only broadcast view of one 1.0 instead of a (C, n) array
            # of ones (235 MB at the native size); every use (beta^2 = q w / (omega0 sigma^2), the c_dat weight
            # w / sigma^2) reads the same 1.0 values and allocates its own result
            w = np.broadcast_to(np.float64(1.), (n_channels, index.size))
        else:
            w = readonly_copy(self.weights.reshape(n_channels, -1)[:, index])
        put("weights_masked", w)
        self._check_sigma_domain(n_channels, index.size)
        q = grid.node_weights().ravel()[index]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            beta2 = q[None, :]*w/(self.omega0*self.sigma2[:, None])      # q_a m_a w_c(a) / (omega0 sigma_c^2)
            if self.support_v is not None:
                beta2[1:] *= self.support_v                               # C04-6 / C06-8: times v_c (exact, 0 or 1)
        put("beta", freeze_new(np.sqrt(beta2)))
        del beta2, q
        if (donor is not None and donor.grid is grid and donor.tissue is self.tissue
                and donor.tau2.tobytes() == self.tau2.tobytes()):
            put("tau2_nodes", donor.tau2_nodes)
        else:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                tau2_nodes = self.tissue[0]*self.tau2[0]
                for k in range(1, self.tau2.size):
                    tau2_nodes = tau2_nodes+self.tissue[k]*self.tau2[k]
            put("tau2_nodes", _readonly(scalar_field(tau2_nodes, grid, name="tau2_nodes")))

    def _check_sigma_domain(self, n_channels, n):
        """C04-6: the sigma_domain fields (brain: theta and v None; otherwise theta checked with K = n_channels - 1 and v
        a (K, n) uint8 array of 0/1, kept read-only, with its empty rows and sha256)."""
        put = lambda name, value: object.__setattr__(self, name, value)   # noqa: E731
        if self.sigma_domain not in SIGMA_DOMAINS:
            raise InputValidationError(f"sigma_domain: one of {SIGMA_DOMAINS} (C04-6)")
        if self.sigma_domain == BRAIN:
            if self.support_threshold is not None or self.support_v is not None:
                raise InputValidationError("sigma_domain 'brain': support_threshold and support_v must be None (C04-6)")
            put("support_empty", ())
            put("support_v_sha256", None)
            return
        if n_channels < 2:
            raise InputValidationError(f"sigma_domain {self.sigma_domain!r}: structure channels required (C04-6: "
                                       "channel_source none is refused)")
        put("support_threshold", check_theta(self.support_threshold, n_channels-1))
        v = self.support_v
        if (not isinstance(v, np.ndarray) or v.dtype != np.uint8 or v.shape != (n_channels-1, n)
                or (v.size and int(v.max()) > 1)):
            raise InputValidationError(f"support_v: a ({n_channels-1}, {n}) uint8 array of 0/1 on the mask nodes "
                                       "required (C04-6)")
        v = adopt_readonly(v, np.uint8)
        put("support_v", v)
        donor = _DONOR.get()
        if donor is not None and donor.support_v is v:                   # MEM-style reuse under `replace` (same v)
            put("support_empty", donor.support_empty)
            put("support_v_sha256", donor.support_v_sha256)
            return
        put("support_empty", tuple(c+1 for c in range(v.shape[0]) if not v[c].any()))
        put("support_v_sha256", v_sha256(v))

    @property
    def n_channels(self):
        return self.fixed.shape[0]

    @property
    def has_support(self):
        """True under sigma_domain support / support_shared (C04-6)."""
        return self.sigma_domain != BRAIN

    def sigma2_record(self):
        """sigma^2 as recorded (C04-6 "记录"): brain -> sigma2.tolist() (unchanged); otherwise null for every structure
        channel with an empty support (support: not estimated; support_shared: internally sigma_S^2, beta_c = 0)."""
        values = self.sigma2.tolist()
        for c in self.support_empty:
            values[c] = None
        return values

    def sigma_domain_record(self):
        """The C04-6 record of the model's sigma_domain (non-brain states only; brain gives {})."""
        if not self.has_support:
            return {}
        return dict(sigma_domain=self.sigma_domain, support_threshold=self.support_threshold, support_rule=SUPPORT_RULE,
                    support_v_sha256=self.support_v_sha256, support_empty_channels=list(self.support_empty),
                    support_shared=self.sigma_domain == SUPPORT_SHARED)

    @property
    def logvol_corners(self):
        """True when the log-volume prior includes the cell-corner term of C05-2a."""
        return self.logvol_points == LOGVOL_GAUSS27_CORNERS

    @property
    def logvol_closed27(self):
        """True when the log-volume prior is the C05-2b composite rule on Gauss27 U closed27."""
        return self.logvol_points == LOGVOL_GAUSS27_CLOSED27

    @property
    def logvol_extra(self):
        """The second point set of the log-volume block: None ("gauss27"), "corners" (C05-2a) or "closed27" (C05-2b)."""
        return {LOGVOL_GAUSS27_CORNERS: "corners", LOGVOL_GAUSS27_CLOSED27: "closed27"}.get(self.logvol_points)

    def replace(self, **changes):
        """The next (level, EM round): same grid, channels and samplers, new sigma2 / weights / eta.  MEM (memory
        only): the new state shares the read-only inputs and the derived node arrays that the change leaves equal
        (see __post_init__); every value is the one a fresh construction gives."""
        token = _DONOR.set(self)
        try:
            return replace(self, **changes)
        finally:
            _DONOR.reset(token)

    def provenance(self):
        return dict(contract="C01-4/C04/C05/C06", grid_id=self.grid.grid_id, omega0=self.omega0,
                    omega_level=self.grid.omega, tile_size=self.tile_size, n_channels=self.n_channels,
                    n_mask_nodes=int(self.node_index.size), sigma2=self.sigma2.tolist(),
                    noise="gaussian" if self.weights is None else "student_t_weights", alpha=self.alpha,
                    nu=self.nu, tau2=self.tau2.tolist(), n_tissue_classes=int(self.tau2.size-1),
                    tau2_at_gauss_points="Q1 interpolation of sum_k Ptilde_T^k tau_k^2 (C05-2)",
                    **self.logvol_record(), **_parallel_provenance(self.cpu_threads), **self.sigma_domain_record())

    def logvol_record(self):
        """The C05-2 / C05-2a / C05-2b point set of the log-volume prior, for provenance."""
        if self.logvol_closed27:
            share = self.logvol_closed27_share
            return dict(logvol_points=self.logvol_points, logvol_closed27_share=share,
                        closed27_weights=self.closed27_weights, theta_G=self.logvol_theta[0],
                        theta_S=self.logvol_theta[1],
                        logvol_closed27=(
                            "C05-2b: Pi_J = 1/2 [theta_G <(log J)^2/tau^2>_Q + theta_S <(log J)^2/tau^2>_S]; S the 27 "
                            "points {0, 1/2, 1}^3 of every cell (CellQ1Operators3D closed27_uniform points, basis, "
                            "coefficients; its uniform .weights not used); J_{K,p} of K's own Q1 map (shared points "
                            "not averaged); tau_{K,p}^2 the Q1 interpolation of tau2_nodes; per tile the closed27 sum "
                            "follows the tile's Gauss sum (C06-7)"),
                        closed27_rule=("Simpson s_p = sigma_i sigma_j sigma_k, sigma = (1/6, 2/3, 1/6)"
                                       if self.closed27_weights == CLOSED27_SIMPSON
                                       else "uniform 1/27 (ablation switch; Laplace and EB refuse it, a scope limit "
                                            "of C14-7 / C15-10: the Q_{2,2,2} exactness of C05-2b (iii) and the B17-5 "
                                            "spectral bound hold for the Simpson weights only; the B17 count identity "
                                            "holds for any weights)"),
                        weights_float=("theta_G = 1.0/(1.0 + beta_S), theta_S = beta_S/(1.0 + beta_S) (one float64 "
                                       "division each); Gauss: theta_G * ops.weights; closed27: theta_S * W27, W27 = "
                                       "omega * x * y * z over product(sigma, repeat=3) (Simpson) or omega/27 (uniform)"))
        if not self.logvol_corners:
            return dict(logvol_points=self.logvol_points)
        return dict(logvol_points=self.logvol_points, logvol_corner_weight=self.corner_weight,
                    logvol_corners=("C05-2a: + beta_c/2 sum_K sum_{v in corners(K)} omega/(8 omega0) (log J_{K,v})^2 / "
                                    "tau_{K,v}^2; J_{K,v} of K's own Q1 map at v (not averaged), tau_{K,v}^2 the node "
                                    "value at v; per tile the corner sum follows the tile's Gauss sum (C06-7)"))


def _readonly(array):
    array = np.array(array, copy=True, order="C")
    array.flags.writeable = False
    return array
