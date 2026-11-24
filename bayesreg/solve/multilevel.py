"""C12 multiresolution driver with C11 rounds and C13 stopping (one pair).

Levels ell = 0..3 on `four_level_grids(native)` (C12-1; not nested); each level's arrays come from the
C03.6 pyramid (`level_inputs` adapts `seg.channels.channel_pyramid`).  Per level:
  1. seed: ell = 0 -> u = 0, which must pass the certificate (C07-6; else initialization_failed);
     ell > 0 -> the previous level's last formal commit through `prolong_displacement` (Q1, six faces
     zero, no scaling or clipping), re-certified as s_k u_prolonged with s_k = .9^k, k = 0..6, each
     relative to the original seed; the first PASS is the new level's initial state; seven rejections
     -> transfer_rejected with all seven rows recorded (C12-2, C20 D17 default "end").
  2. sigma at the level start: `SigmaPolicy.level_start` at the seed (C11-3: the M step with w = 1);
  3. rounds r = 0..R_ell-1 (default R = (1, 1, 1, 3)): G0 = GF_B(u = 0) with this round's sigma (C13-2),
     T_B = 1e-8 + 1e-6 G0, one outer solve (option (a) or (b)), then either the run's final status
     (native level, last round; C13-3) or the C13-7 continuation decision; on continue the M step uses
     the round's last formal commit (`SigmaPolicy.m_step`), and after the level's last round the same
     commit is the transfer seed (C12-3a).  sigma changes only between inner solves.
EM (C11) is an interface here: `SigmaPolicy` gives sigma at the level start and after every round.
`GaussianMStep` is the Gaussian closed form of C11-2 with a = b = 0 (w = 1): sigma_c^2 = <rho_c^2>_B /
(N_q + 2), <rho_c^2>_B = 2 sigma_used^2 E_c (E_c = 1/2 sum r_c^2 of the channel block); `FixedSigma`
keeps given values (tests).  Student-t weights are not implemented (NotImplementedError).
`max_levels` < 4 stops after level max_levels - 1 with run status stopped_before_native (development
switch; no native output).
Exceptions (hard constraint 4, S1 review item 2): a CommitTracker follows the round's formal commits; an
ArithmeticError / RuntimeError / MemoryError anywhere in a round (prepare, G0, the outer solve) ends the run
level_failed with the round's last formal commit (its start state when it made none), the exception in the reason;
on the native level that commit is exported as for any level_failed (C13-7).  With jd_cell_estimate
certificate_bernstein the exception row also carries `jd_scaling`, the round's S~ ledger record up to the exception
(JDTI review minor 1: jd_scaling.j53_violations_total reads it; the other estimates' exception rows are unchanged).  A FloatingPointError inside the PG
loop (gradient, direction, stable comparison) already ends the inner solve nonfinite_evaluation at its last
accepted point.  The diagnostic sigma_post of the native last round (C11-6) never changes the ending: a failing
M step there leaves sigma2_post null with the error.
sigma monitor (C11-5, D15): `SigmaMonitor` records every sigma update and flags an adjacent-round drop of more
than 10x (`RunResult.sigma`; run_pair writes it to hyper.json and qc.json); a flag only.
G0 (C08-8(f)) is computed on the backend's CPU reference when it has one; the native level's final GF_B is
recomputed on the CPU (`GF_B_cpu` of the final round row).
Wall-clock budget (development switch --wall-seconds; C13-3 appendix, C20 D44, pending review; `budget`, a
stopping.WallBudget or None): checked before every transfer (before the next level's template is built; without a
budget the order is immaterial) and, inside a round, at the solver's safe points
(bayesreg.solve.pg / outer).  A round that ends budget_exhausted is the run's last round (row final = true, no M step,
continuation {"proceed": false}): on the native level the run ends budget_exhausted ("预算收尾") with the round's last
formal commit (exported and re-certified by the caller like any ending; GF_B_cpu recomputed); on a coarse level, or
when the budget is found used up before a transfer, the run ends stopped_before_native with that level's last formal
commit (no native output).  RunResult.budget carries the budget record (None without a budget).
Log-volume point set (model option of every level's ModelState, part of SolverConfig): logvol_points
"gauss27+closed27" (C05-2b; the default since DEF, 2026-10-31: C05-2b first paragraph, D43, D48), with
logvol_closed27_share beta_S (default 1) and closed27_weights "simpson" (default, D48) | "uniform" (ablation);
"gauss27" (C05-2, the default before DEF) and "gauss27+corners" (C05-2a) are explicit options; the model of either
is bitwise the code before DEF.  A SolverConfig with only logvol_points given is NOT the solve before DEF: DEF also
made "certificate" the default jd_cell_estimate of PGConfig / JDScalingConfig (C09.1-3a, D47; the default before DEF
was "gauss27+corners").  The solve before DEF needs both, SolverConfig(logvol_points="gauss27",
pg=PGConfig(jd=JDScalingConfig(cell_estimate="gauss27+corners"))); with metric_jd_scaling "off" the cell estimate is
not used and SolverConfig(logvol_points="gauss27", pg=PGConfig(metric_jd_scaling="off")) is the solve before DEF
with the metric off (FUP1; reviews DEF and MIN1; run_pair's module docstring gives the same rule for the command
line).  beta_S and the closed27 rule apply only with "gauss27+closed27" and are refused otherwise.
sigma_domain (contract C04-6, C11-8; work package SWLI; draft SWL 2026-10-31, D49/D50 for BZ; SolverConfig.sigma_domain,
support_threshold).  "brain" (default): every step above, bitwise the code before SWLI.  "support" / "support_shared"
(GaussianMStep only): at every level start, after the seed is certified and evaluated (an unevaluable seed ends the run
initialization_failed as before), `bayesreg.model.support.support_level` computes on the CPU reference the frozen support
v of the level and the level-start sums (C11-8 "层首的算式"); `SupportMStep.level_start` gives sigma from those sums (not
through 2 sigma^2 E_c); every round's ModelState is host.replace(sigma2, **level_fields) (sigma_domain, theta, v), so beta
contains v.  Every round's M step and sigma_post use `SupportMStep.m_step` ("浮点写法": 2 sigma_c^2 E_c of the round's
last evaluation, N_c of the level).  v is never recomputed within a level.  Records (non-brain only): a log.jsonl row of
kind "support" per level, RunResult.support (one SupportLevel.record() per level, with sigma2_start and sigma2_S_start),
the round rows' sigma2_used / sigma2_post with null for an empty-support channel, sigma2_S_used / sigma2_S_post
(support_shared) and sigma_tilde2 (the diagnostic <v_c rho_c^2>_B / N_c per structure channel), and the level timing's
support_cpu_seconds.  SigmaMonitor takes null entries (C04-6 gate 14).  Memory (work package SWLM, allocator only): the
support pass's freed temporaries are handed back to the OS (bayesreg.heap.release_free_heap) right after support_level,
before the level's round state and G0 -- the brain path makes no such call.
Per-level caps and coarse-level shares (work package SCHI; contract C13-4a, C13-5a, C12-3a SCH item, C16 SCH item; draft
SCH 2026-10-31, path only, defaults unchanged): every (level, round) of level ell runs with `config.caps.for_level(ell)`
(the Caps itself without per-level caps).  With `config.level_time_shares` (a stopping.LevelShares on the run's
WallBudget, required then) the three coarse levels get a stopping.LevelDeadline D_ell, checked by the solver at the PG
and option (b) outer iteration tops (bayesreg.solve.pg / outer).  A round that ends level_budget_reached is the level's
last round: the C13-7 table continues (as cap_reached), the M step is made on the round's last formal commit as usual
but is recorded as not used (D42: the next level re-estimates sigma; a FloatingPointError in it ends the run
level_failed exactly as for a coarse last round), the remaining rounds of the level are skipped (rounds_skipped) and the
transfer takes that commit as its seed.  A share is never checked on the native level.  On the last level of a
--max-levels < 4 run the share ending is that level's final round (stopped_before_native).  Records (C16 SCH item) only
when either switch is given (`SolverConfig.sch_records`): SolverConfig.record() gets max_accepted_per_level,
max_accepted_source, level_time_shares and level_deadlines_seconds; every round row cap_accepted, accepted_committed,
level_time_share, level_cumulative_share, level_deadline_seconds, level_elapsed_seconds, run_elapsed_seconds (both from
WallBudget readings at the level top -- before the transfer check -- and at the round end; null without a budget),
ended_by_level_share, level_share_hit and rounds_skipped; every timing level entry deadline_seconds and
ended_by_level_share; a log.jsonl row of kind "level_budget" when a share ends a level.  Without them nothing is added and
no extra clock reading is made.
Inner solver (work package LMI, 2026-11-01; contract C09.3, path only, default unchanged): `SolverConfig.inner` "pg"
(default: run_pg, bitwise the code before LMI) or "lm" (bayesreg.solve.lm.run_lm with `SolverConfig.lm`, an LMConfig;
options (a) and (b); numpy backend only, refused by solve_round otherwise).  Under "lm" the PG settings have no effect
(the JD scaling of C09.1-3a and its cell estimate change only the PG direction): `pg` must be the default PGConfig(), the
record's `pg` is null and `lm` records metric_jd_scaling and jd_cell_estimate as "none".  Records only under "lm":
SolverConfig.record() gets `inner` and `lm`; the round rows' metric carries `lm` (gamma0, damping and PCG summary) and
jd_scaling {"mode": "none"}; log.jsonl gets the LM rows (kind lm_start, candidate, accept; bayesreg.solve.lm).
Native-level warm start (work packages WST, WARM; contract C12-4a, C16 WST item, D59; path only, default off):
`SolverConfig.start_level` 3 with `init_seed` (a bayesreg.solve.init_field.InitSeed; both or neither) skips levels 0..2: the
level loop starts at the native level, whose entry state is the first of s_k u_seed, s_k = .9^k, k = 0..6 that passes the
certificate (C12-2 without the prolongation; seven rejections -> initialization_failed, u = 0 is never used; the rows go to
log.jsonl kind "init_field" and RunResult.init_field).  Everything after the entry state is the code of every level: the
sigma level start at the entry state (C11-3), the frozen support of sigma_domain != brain (C04-6), G0 at u = 0 (C13-2), the
rounds of rounds[3].  Records only under the switch: SolverConfig.record() start_level, levels_skipped and
levels_skipped_reason; RunResult.init_field.  With start_level 0 (default) the loop, the records and every value are the
code before WST.
Profile and checkpoints (C12-4a item 8; records only): solve_multilevel(recorder=) takes a checkpoints.ProfileRecorder (run_pair: with
--init-field or --checkpoint-*).  On the native level only: its first round evaluates the entry state once more for L, every accepted
step of option (a) PG goes through ProfileTracker (writing the checkpoints that are due), and each round end updates L at the end
state; RunResult.profile is its record.  `SolverConfig.checkpoint` (a checkpoints.CheckpointConfig, None by default) records the switch
and refuses option (b) and LM; the states and the decisions of the solve do not change (test_init_field).
"""
from dataclasses import dataclass, field
import time

import numpy as np

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError, finite_real, readonly_copy
from jd_curl.three_d.boundary import restrict_free
from jd_curl.three_d.validation import integer_at_least

from .. import memtrace
from ..heap import release_free_heap
from ..readonly import freeze_new
from ..geom.parallel import DEFAULT_CPU_THREADS, cpu_threads as _cpu_threads
from ..grids import FACTORS, prolong_displacement
from ..model import (CLOSED27_SIMPSON, CLOSED27_WEIGHTS, LOGVOL_DEFAULT, LOGVOL_GAUSS27_CLOSED27, LOGVOL_POINTS,
                     ModelState, build_samplers)
from ..model.support import (BRAIN, SIGMA2_PLACEHOLDER, SIGMA_DOMAINS, SUPPORT, SUPPORT_SHARED, check_theta,
                             domain_record, resolve_sigma_domain, support_level)
from .outer import OUTER_OPTIONS, Commit, CommitTracker, solve_round
from .checkpoints import CheckpointConfig
from .init_field import SKIP_REASON, InitSeed, status_summary
from .jd_scaling import CERTIFICATE_BERNSTEIN
from .lm import INNER_LM, INNER_PG, INNER_SOLVERS, LMConfig
from .pg import PGConfig
from .proximal import ProximalParams
from .stopping import (ABS_TOL, BUDGET_EXHAUSTED, INITIALIZATION_FAILED, LEVEL_BUDGET_REACHED, LEVEL_FAILED,
                       MAX_ACCEPTED, REL_TOL, RUN_WORDING, STOPPED_BEFORE_NATIVE, TRANSFER_REJECTED, Caps, LevelShares,
                       budget_expired, continuation, final_status, g0_of, threshold)

TRANSFER_ALPHAS = tuple(.9**k for k in range(7))       # C12-2 (mainline TRANSFER_BACKTRACK_ALPHAS)
DEFAULT_ROUNDS = (1, 1, 1, 3)                          # C11-3 / D15
CHANNEL_SOURCES = ("structure", "tissue", "none")
MAX_ACCEPTED_SOURCES = ("default", "max_accepted", "max_accepted_per_level")      # C16 SCH: max_accepted_source
TAU0_INITIAL = 1.0                                     # C05-7
TAU_K_INITIAL = .5                                     # C05-7
ALPHA_NUG_FACTOR = 1e-2                                # C05-6 / D9


# ---- inputs -----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LevelInput:
    """One pyramid level: channels first (c = 0 intensity, then the likelihood channels), the fixed brain
    mask and the floored tissue-layer simplex of the fixed image (K_T + 1 channels)."""
    level: int
    grid: object
    fixed: np.ndarray
    moving: np.ndarray
    mask: np.ndarray
    tissue: np.ndarray


def level_inputs(levels, channel_source):
    """C03.1-3: the ModelState channels of every ChannelLevel: [F~] + s_F^k (structure / tissue) or [F~]
    (none); the tissue layer always gives tau^2 and the mask."""
    if channel_source not in CHANNEL_SOURCES:
        raise InputValidationError(f"channel_source: one of {CHANNEL_SOURCES}")
    out = []
    for lv in levels:
        if channel_source == "none":
            fixed, moving = readonly_copy(lv.fixed[None]), readonly_copy(lv.moving[None])
        else:
            # MEM (memory only): the concatenations are new C-order float64 arrays owned here; they are made read-only
            # in place instead of copied once more (the values of readonly_copy, without the second (C, N) array)
            fixed = freeze_new(np.concatenate([lv.fixed[None], lv.s_fixed]))
            moving = freeze_new(np.concatenate([lv.moving[None], lv.s_moving]))
        out.append(LevelInput(lv.level, lv.grid, fixed, moving, lv.mask, lv.tissue_fixed))
    return tuple(out)


def grid_only(level_input):
    """MEM: the LevelInput of a released level -- level and grid kept, the arrays dropped (solve_multilevel
    release_inputs)."""
    return LevelInput(level_input.level, level_input.grid, None, None, None, None)


@dataclass(frozen=True)
class Eta:
    """Model parameters of the priors (C05): alpha, nu and tau_k^2 (k = 0 background)."""
    alpha: float
    nu: float
    tau2: tuple
    source: str = "given"

    def __post_init__(self):
        object.__setattr__(self, "alpha", finite_real(self.alpha, "alpha", positive=True))
        nu = finite_real(self.nu, "nu")
        if nu < 0:
            raise InputValidationError("nu: must be nonnegative")
        object.__setattr__(self, "nu", nu)
        tau2 = tuple(finite_real(t, "tau2", positive=True) for t in self.tau2)
        if len(tau2) < 2:
            raise InputValidationError("tau2: K_T + 1 >= 2 values required")
        object.__setattr__(self, "tau2", tau2)

    def record(self):
        return dict(alpha=self.alpha, nu=self.nu, tau2=list(self.tau2), tau=[float(np.sqrt(t)) for t in self.tau2],
                    source=self.source)


def eta_temporary(tissue, mask, grid, *, tau0=TAU0_INITIAL, tau_k=TAU_K_INITIAL):
    """C05-7 (P2, D9/D27): tau^(0) = (tau0, tau_k, ..., tau_k), nu_tmp = <1/tau^(0)2>_B/<1>_B over the brain
    (weights q_a m_a, tau_a^2 = sum_k Ptilde_T^k tau_k^2, native level), alpha_tmp = alpha_nug =
    1e-2 min_k (1/tau_k^2)."""
    tau2 = np.array([tau0**2]+[tau_k**2]*(tissue.shape[0]-1))
    q = grid.node_weights()[mask]
    tau2_nodes = np.tensordot(tau2, tissue, axes=1)[mask]
    nu = float(np.sum(q/tau2_nodes, dtype=np.longdouble)/np.sum(q, dtype=np.longdouble))
    alpha = ALPHA_NUG_FACTOR*float(np.min(1/tau2))
    return Eta(alpha, nu, tuple(float(t) for t in tau2), source="C05-7 temporary values (P2)")


# ---- sigma policies (C11 interface) -------------------------------------------------------------------

def n_q(state):
    """N_q = <1>_B = sum_a q_a m_a / omega0 (C04-2)."""
    return float(np.sum(state.grid.node_weights()[state.mask], dtype=np.longdouble)/np.longdouble(state.omega0))


def channel_mean_square(state, point):
    """<w_c rho_c^2>_B per channel from the channel energies: 2 sigma_c^2 E_c (E_c = 1/2 sum r_c^2)."""
    return np.array([2*s*e for s, e in zip(state.sigma2, point.energies["channel"])])


class GaussianMStep:
    """C11-2 Gaussian M step with a = b = 0 and w = 1: sigma_c^2 <- <rho_c^2>_B / (N_q + 2)."""
    name = "gaussian_m_step"
    weights = None

    def __init__(self, *, a=0., b=0., floor=0.):
        self.a, self.b, self.floor = finite_real(a, "a"), finite_real(b, "b"), finite_real(floor, "floor")

    def _update(self, state, point):
        value = (channel_mean_square(state, point)+2*self.b)/(n_q(state)+2*self.a+2)
        if not np.all(np.isfinite(value)) or np.any(value <= self.floor):
            raise FloatingPointError(f"M step: sigma^2 = {value.tolist()} is not positive (C11)")
        return value

    def level_start(self, state, point):
        return self._update(state, point)

    def m_step(self, state, point):
        return self._update(state, point)

    def record(self):
        return dict(policy=self.name, noise="gaussian", a=self.a, b=self.b, level_start="M step at the seed, w = 1")


class FixedSigma:
    """sigma_c^2 fixed for the whole run (tests and diagnostics)."""
    name = "fixed"
    weights = None

    def __init__(self, sigma2):
        self.sigma2 = np.asarray(sigma2, dtype=np.float64)

    def level_start(self, state, point):
        return self.sigma2

    def m_step(self, state, point):
        return self.sigma2

    def record(self):
        return dict(policy=self.name, noise="gaussian", sigma2=self.sigma2.tolist())


@dataclass(frozen=True)
class SupportSigma:
    """sigma^2 under sigma_domain support / support_shared: `internal` (C,) for the ModelState, `record` with null for an
    empty-support channel, sigma2_S the shared value (support_shared; None when absent or under support)."""
    internal: np.ndarray
    record: list
    sigma2_S: float | None


class SupportMStep:
    """C11-8 Gaussian M step under sigma_domain "support" / "support_shared" (a = b = 0 unless given, as GaussianMStep;
    the same (a, b) for every variance parameter: (a_I, b_I), (a_c, b_c) or (a_S, b_S)).  Variance parameters: I and
    every structure channel with a non-empty support (support), or I and S (support_shared; S absent when every
    structure channel is empty).

      level_start(level)           from the CPU level-start sums of `support_level` (C11-8 "层首的算式"):
                                   sigma_I^2 = (S_I + 2b)/(N_q + 2a + 2), sigma_c^2 = (S_c + 2b)/(N_c + 2a + 2),
                                   sigma_S^2 = (S_S + 2b)/(N_S + 2a + 2); S and N float64 (each converted once)
      m_step(state, point, level)  "浮点写法": <v_c rho_c^2>_B = 2 sigma_c^2 E_c of the evaluation (float64); N_c, N_S of
                                   the level; support_shared: S_S = float(longdouble sum over the non-empty c in ascending
                                   order of 2 sigma_S^2 E_c); N_q = n_q(state) for the intensity
    Both return a SupportSigma: the internal (C,) sigma^2 of the ModelState (an empty channel: SIGMA2_PLACEHOLDER under
    support, sigma_S^2 under support_shared, SIGMA2_PLACEHOLDER when S is absent -- beta_c = 0 either way) and the
    record (null for every empty channel; sigma2_S null when S is absent).  A variance <= floor (e.g. every residual on
    the support exactly 0, a = b = 0) raises FloatingPointError naming sigma_domain and the channel (S when shared), as
    GaussianMStep does (C04-6 "空支撑": level start / non-final M step -> level_failed; native sigma_post -> recorded)."""
    name = "gaussian_m_step"
    weights = None

    def __init__(self, sigma_domain, *, a=0., b=0., floor=0.):
        if sigma_domain not in (SUPPORT, SUPPORT_SHARED):
            raise InputValidationError(f"SupportMStep: sigma_domain support or support_shared, not {sigma_domain!r}")
        self.sigma_domain = sigma_domain
        self.a, self.b, self.floor = finite_real(a, "a"), finite_real(b, "b"), finite_real(floor, "floor")

    @classmethod
    def from_policy(cls, policy, sigma_domain):
        """The support M step with the (a, b, floor) of a GaussianMStep (the only policy admitted with
        sigma_domain != brain)."""
        if type(policy) is not GaussianMStep:
            raise InputValidationError(f"sigma_domain {sigma_domain!r}: the GaussianMStep policy is required (C11-8; "
                                       f"got {type(policy).__name__})")
        return cls(sigma_domain, a=policy.a, b=policy.b, floor=policy.floor)

    def _value(self, numerator, count, label):
        value = (numerator+2*self.b)/(count+2*self.a+2)
        if not np.isfinite(value) or value <= self.floor:
            raise FloatingPointError(f"M step ({self.sigma_domain}, channel {label}): sigma^2 = {value!r} is not "
                                     "positive (C11-8; C04-6 支撑上的残差全为 0)")
        return float(value)

    def _assemble(self, level, intensity, structure):
        """structure: {c: sigma_c^2} (support) or {"S": sigma_S^2} (support_shared; {} when S is absent)."""
        internal = np.full(level.k_s+1, SIGMA2_PLACEHOLDER)
        internal[0] = intensity
        record = [intensity]+[None]*level.k_s
        sigma_S = None
        if self.sigma_domain == SUPPORT:
            for c, value in structure.items():
                internal[c] = record[c] = value
        elif "S" in structure:
            sigma_S = structure["S"]
            internal[1:] = sigma_S
            for c in level.nonempty:
                record[c] = sigma_S
        return SupportSigma(internal, record, sigma_S)

    def level_start(self, level):
        s = [float(x) for x in level.s_ld]
        intensity = self._value(s[0], level.n_q, "I")
        structure = {}
        if self.sigma_domain == SUPPORT:
            n_c = level.n_c
            structure = {c: self._value(s[c], n_c[c-1], c) for c in level.nonempty}
        elif level.nonempty:
            s_S, n_S = level.shared_sums()
            structure["S"] = self._value(s_S, n_S, "S")
        return self._assemble(level, intensity, structure)

    def m_step(self, state, point, level):
        ms = channel_mean_square(state, point)             # 2 sigma_c^2 E_c (beta contains v)
        intensity = self._value(ms[0], n_q(state), "I")
        structure = {}
        if self.sigma_domain == SUPPORT:
            n_c = level.n_c
            structure = {c: self._value(ms[c], n_c[c-1], c) for c in level.nonempty}
        elif level.nonempty:
            total = np.longdouble(0)
            for c in level.nonempty:
                total += np.longdouble(ms[c])
            structure["S"] = self._value(float(total), level.n_s(), "S")
        return self._assemble(level, intensity, structure)

    @staticmethod
    def sigma_tilde2(state, point, level):
        """Diagnostic, not used by the solve (C04-6 "诊断"): sigma~_c^2 = <v_c w_c rho_c^2>_B / N_c per structure channel
        c = 1..K_S (null when the support is empty)."""
        ms = channel_mean_square(state, point)
        n_c = level.n_c
        return [None if n_c[c-1] == 0 else float(ms[c]/n_c[c-1]) for c in range(1, level.k_s+1)]

    def record(self):
        return dict(policy=self.name, noise="gaussian", a=self.a, b=self.b, sigma_domain=self.sigma_domain,
                    level_start="M step at the seed, w = 1, from the CPU level-start sums (C11-8 层首的算式)",
                    m_step="2 sigma^2 E_c of the round's last evaluation, N_c / N_S of the level (C11-8 浮点写法)")


# ---- configuration ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverConfig:
    outer: str = "a"
    rounds: tuple = DEFAULT_ROUNDS
    caps: Caps = field(default_factory=Caps)
    pg: PGConfig = field(default_factory=PGConfig)
    metric_mode: str = "principal"
    proximal: ProximalParams | None = None
    abs_tol: float = ABS_TOL
    rel_tol: float = REL_TOL
    max_levels: int = 4
    transfer_alphas: tuple = TRANSFER_ALPHAS
    tile_size: int = 8192
    cpu_threads: int = DEFAULT_CPU_THREADS     # threads of the CPU reference (CPUPAR; bitwise neutral)
    logvol_points: str = LOGVOL_DEFAULT        # C05-2 / C05-2a / C05-2b model option of every level's ModelState
    #                                            (default gauss27+closed27 since DEF)
    logvol_closed27_share: float = 1.          # beta_S of C05-2b (only with "gauss27+closed27")
    closed27_weights: str = CLOSED27_SIMPSON   # D48 closed27 rule (only with "gauss27+closed27")
    sigma_domain: str = BRAIN                  # C04-6 / C11-8 (SWLI): brain (default) | support | support_shared
    support_threshold: float | None = None     # theta (D50): None under brain (explicit refused); default .05 otherwise
    level_time_shares: LevelShares | None = None    # C13-5a (SCH, development switch): None = no share (default)
    max_accepted_source: str | None = None     # C16 SCH record: default | max_accepted | max_accepted_per_level (None:
    #                                            derived -- per-level caps given, else "default" for 500, else
    #                                            "max_accepted"); recorded only with sch_records
    inner: str = INNER_PG                      # C09 inner solver (LMI): "pg" (default) | "lm" (C09.3)
    lm: LMConfig | None = None                 # the LM parameters under inner "lm" (default LMConfig()); None under "pg"
    start_level: int = 0                       # C12-4a (WST): 0 (default) | 3 (with solve_multilevel's init_seed)
    checkpoint: CheckpointConfig | None = None  # C12-4a item 8 (WST): native-level checkpoints; None (default) = none

    def __post_init__(self):
        if self.checkpoint is not None:
            if not isinstance(self.checkpoint, CheckpointConfig):
                raise InputValidationError("checkpoint: a checkpoints.CheckpointConfig (C12-4a item 8)")
            if self.max_levels != len(FACTORS) or self.outer != "a" or self.inner != INNER_PG:
                raise InputValidationError("checkpoint: max_levels 4, option (a), inner pg (C12-4a item 8)")
        if self.start_level != 0:
            if self.start_level != len(FACTORS)-1:
                raise InputValidationError(f"start_level: 0 or {len(FACTORS)-1} (C12-4a: a native seed enters at the "
                                           "native level)")
            if self.max_levels != len(FACTORS) or self.outer != "a" or self.inner != INNER_PG \
                    or self.level_time_shares is not None:
                raise InputValidationError("start_level 3: max_levels 4, option (a), inner pg, no level_time_shares "
                                           "(C12-4a item 6)")
        if self.sigma_domain not in SIGMA_DOMAINS:
            raise InputValidationError(f"sigma_domain: one of {SIGMA_DOMAINS} (C04-6)")
        _, theta = resolve_sigma_domain(self.sigma_domain, self.support_threshold)
        object.__setattr__(self, "support_threshold", theta)
        if self.outer not in OUTER_OPTIONS:
            raise InputValidationError(f"outer: one of {OUTER_OPTIONS}")
        if self.logvol_points not in LOGVOL_POINTS:
            raise InputValidationError(f"logvol_points: one of {LOGVOL_POINTS} (C05-2, C05-2a, C05-2b)")
        object.__setattr__(self, "logvol_closed27_share",
                           finite_real(self.logvol_closed27_share, "logvol_closed27_share", positive=True))
        if self.closed27_weights not in CLOSED27_WEIGHTS:
            raise InputValidationError(f"closed27_weights: one of {CLOSED27_WEIGHTS} (D48)")
        if self.logvol_points != LOGVOL_GAUSS27_CLOSED27 and (self.logvol_closed27_share != 1.
                                                              or self.closed27_weights != CLOSED27_SIMPSON):
            raise InputValidationError("logvol_closed27_share / closed27_weights apply only to logvol_points = "
                                       "'gauss27+closed27' (C05-2b)")
        if self.outer == "b" and self.proximal is None:
            raise InputValidationError("outer option (b) needs ProximalParams")
        rounds = tuple(integer_at_least(r, "rounds") for r in self.rounds)
        if len(rounds) != len(FACTORS):
            raise InputValidationError("rounds: one entry per level (4)")
        object.__setattr__(self, "rounds", rounds)
        object.__setattr__(self, "max_levels", integer_at_least(self.max_levels, "max_levels"))
        object.__setattr__(self, "cpu_threads", _cpu_threads(self.cpu_threads))
        if self.max_levels > len(FACTORS):
            raise InputValidationError("max_levels: at most 4")
        if self.level_time_shares is not None and not isinstance(self.level_time_shares, LevelShares):
            raise InputValidationError("level_time_shares: a stopping.LevelShares (C13-5a)")
        if self.inner not in INNER_SOLVERS:
            raise InputValidationError(f"inner: one of {INNER_SOLVERS} (C09)")
        if self.inner == INNER_PG:
            if self.lm is not None:
                raise InputValidationError("lm: LM parameters only with inner 'lm' (C09.3)")
        else:
            if self.lm is None:
                object.__setattr__(self, "lm", LMConfig())
            elif not isinstance(self.lm, LMConfig):
                raise InputValidationError("lm: an LMConfig (C09.3)")
            if self.pg != PGConfig():
                raise InputValidationError("inner 'lm': the PG settings (metric_jd_scaling, jd_cell_estimate, BB2, "
                                           "Armijo) have no effect; leave pg at its default (recorded as none, LMI)")
        per_level = self.caps.max_accepted_per_level is not None
        if self.max_accepted_source is not None:
            if self.max_accepted_source not in MAX_ACCEPTED_SOURCES:
                raise InputValidationError(f"max_accepted_source: one of {MAX_ACCEPTED_SOURCES}")
            if per_level != (self.max_accepted_source == "max_accepted_per_level"):
                raise InputValidationError("max_accepted_source: max_accepted_per_level exactly when the caps are per "
                                           "level (C13-4a)")

    @property
    def sch_records(self):
        """C16 SCH: the new record keys are written only when either switch is given (per-level caps or shares)."""
        return self.caps.max_accepted_per_level is not None or self.level_time_shares is not None

    def sch_record(self):
        """The four solver-record keys of C16 SCH (config.json solver, provenance.json solver, hyper.json algorithm)."""
        caps = self.caps
        per_level = caps.max_accepted_per_level or (caps.max_accepted,)*len(FACTORS)
        source = self.max_accepted_source
        if source is None:
            source = ("max_accepted_per_level" if caps.max_accepted_per_level is not None
                      else "default" if caps.max_accepted == MAX_ACCEPTED else "max_accepted")
        shares = self.level_time_shares
        return dict(max_accepted_per_level=list(per_level), max_accepted_source=source,
                    level_time_shares=None if shares is None else list(shares.shares),
                    level_deadlines_seconds=None if shares is None else list(shares.deadlines))

    def logvol_options(self):
        """The ModelState keyword options of the log-volume point set ({} for "gauss27" and C05-2a: those
        ModelStates are built exactly as before C05-2b; beta_S and the closed27 rule for "gauss27+closed27")."""
        if self.logvol_points != LOGVOL_GAUSS27_CLOSED27:
            return {}
        return dict(logvol_closed27_share=self.logvol_closed27_share, closed27_weights=self.closed27_weights)

    def record(self):
        logvol = dict(logvol_points=self.logvol_points)
        if self.logvol_points == LOGVOL_GAUSS27_CLOSED27:
            logvol.update(logvol_closed27_share=self.logvol_closed27_share, closed27_weights=self.closed27_weights)
        lm = self.inner == INNER_LM
        return dict(outer=self.outer, rounds=list(self.rounds), caps=self.caps.record(),
                    pg=None if lm else self.pg.record(),
                    metric_mode=self.metric_mode, proximal=None if self.proximal is None else self.proximal.record(),
                    threshold=dict(abs_tol=self.abs_tol, rel_tol=self.rel_tol,
                                   formula="T_B = abs_tol + rel_tol G0, G0 = GF_B(u = 0) per (level, round)",
                                   gf_normalization="sqrt(omega_ell/omega0)"),
                    max_levels=self.max_levels, transfer_alphas=list(self.transfer_alphas),
                    transfer_all_rejected="transfer_rejected (D17)", gauss_tile_size=self.tile_size,
                    cpu_threads=self.cpu_threads, **logvol,
                    inner_forcing="off (C10-6)", continuation="C13-7 table",
                    **domain_record(self.sigma_domain, self.support_threshold),
                    **(self.sch_record() if self.sch_records else {}),
                    **(dict(inner=self.inner, lm=self.lm.record()) if lm else {}),
                    **(dict(start_level=self.start_level, levels_skipped=list(range(self.start_level)),
                            levels_skipped_reason=SKIP_REASON) if self.start_level else {}),
                    **({} if self.checkpoint is None else dict(checkpoint=self.checkpoint.record())))

    @property
    def lm_inner(self):
        """The LMConfig to pass to solve_round under inner "lm", else None (PG)."""
        return self.lm if self.inner == INNER_LM else None

    @property
    def metric_jd_scaling(self):
        """The metric_jd_scaling in effect: the PG config's, "none" under inner "lm" (LMI)."""
        return "none" if self.inner == INNER_LM else self.pg.metric_jd_scaling

    @property
    def jd_cell_estimate(self):
        """The jd_cell_estimate in effect: the PG config's, "none" under inner "lm" (LMI)."""
        return "none" if self.inner == INNER_LM else self.pg.jd_cell_estimate


@dataclass
class RunResult:
    status: str
    reason: str
    wording: str
    level: int                    # last level entered
    native: bool                  # the native level was reached (output allowed)
    commit: Commit | None         # last formal commit of the last level entered
    state: object                 # the ModelState of that commit (sigma_used)
    sigma_post: list | None
    rounds: list
    transfers: list
    timing: dict
    counters: dict
    sigma: dict | None = None     # SigmaMonitor.record(): trajectory and the C11-5 collapse flag
    budget: dict | None = None    # WallBudget.record() of the development wall-clock budget (None without one)
    support: list | None = None   # sigma_domain != brain: one SupportLevel.record() per level entered (C04-6); else None
    init_field: dict | None = None    # --init-field (C12-4a): the seed record with its entry rows; else None
    profile: dict | None = None       # C12-4a item 8: ProfileRecorder.record() of the native level; else None


def _template(level_input, eta, omega0, tile_size, n_channels, cpu_threads=DEFAULT_CPU_THREADS,
              logvol_points=LOGVOL_DEFAULT, **logvol_options):
    """The level's ModelState with sigma^2 = 1; logvol_options: logvol_closed27_share, closed27_weights (C05-2b)."""
    samplers = build_samplers(level_input.moving, level_input.grid)
    return ModelState(grid=level_input.grid, omega0=omega0, fixed=level_input.fixed, samplers=samplers,
                      mask=level_input.mask, sigma2=np.ones(n_channels), alpha=eta.alpha, nu=eta.nu,
                      tau2=np.asarray(eta.tau2), tissue=level_input.tissue, tile_size=tile_size,
                      cpu_threads=cpu_threads, logvol_points=logvol_points, **logvol_options)


def transfer(backend, u_coarse, source, target, alphas=TRANSFER_ALPHAS):
    """C12-2: prolong, then s_k u_prolonged for s_k in alphas until the first certificate PASS.
    Returns (seed or None, outcome or None, rows)."""
    return first_certified(backend, prolong_displacement(u_coarse, source, target), target, alphas)


def first_certified(backend, base, target, alphas=TRANSFER_ALPHAS):
    """The back-off of C12-2 (and of the native seed, C12-4a): s_k base for s_k in alphas until the first certificate PASS.
    Returns (seed or None, outcome or None, rows)."""
    rows = []
    for k, scale in enumerate(alphas):
        with np.errstate(over="raise", invalid="raise"):
            seed = freeze_new(scale*base)                         # MEM: new here, read-only in place (same values)
        outcome = backend.certify(seed, target)
        rows.append(dict(k=k, scale=scale, cert=outcome.record()))
        if outcome.passed:
            return seed, outcome, rows
    return None, None, rows


def _evaluate(backend, state, u):
    point = backend.evaluate(state, u)
    if isinstance(point, InvalidTrial):
        return None, f"{point.reason}: {point.message}"
    return point, None


class SigmaMonitor:
    """C11-5 (D15): the sigma_c^2 trajectory and the collapse flag -- a flag only, the solve is not changed.

    Every sigma update is recorded in run order with its ratio to the previous sigma value and sigma_k^2/sigma_I^2
    (k >= 1, channel 0 the intensity).  Collapse criterion (C11-5): between adjacent rounds of one level, i.e. the
    M step after round r against the sigma used in round r, some sigma_c^2 drops by more than COLLAPSE_FACTOR;
    counted when the new value is used (by round r + 1 of the same level, or as the native sigma_post).  The M
    step after a coarse level's last round (not used: the next level re-estimates at its start) and the level
    starts (a different grid) are recorded with their ratios but do not set the flag.

    sigma_domain support / support_shared (C04-6 gate 14, C11-8 "塌缩判据"; SWLI): `sigma2` is the record with null for
    an empty-support channel; a ratio with a null on either side is null (no error, no flag), so the flag looks at the
    non-empty channels only (support) or at sigma_S^2 and sigma_I^2 (support_shared: every non-empty structure channel
    carries sigma_S^2); support_shared rows also carry sigma2_S and its ratios (null when S is absent).  Brain: the code
    before SWLI."""
    COLLAPSE_FACTOR = 10.

    def __init__(self, sigma_domain=BRAIN):
        self.trace, self.events, self.unused_drops = [], [], []
        self._previous = None
        self._previous_S = None
        self.sigma_domain = sigma_domain

    def update(self, sigma2, *, level, round_, kind, used, sigma2_S=None):
        if self.sigma_domain != BRAIN:
            return self._update_nullable(sigma2, sigma2_S, level=level, round_=round_, kind=kind, used=used)
        sigma2 = np.asarray(sigma2, dtype=np.float64)
        ratio = None if self._previous is None else (sigma2/self._previous).tolist()
        row = dict(level=level, round=round_, kind=kind, used=used, sigma2=sigma2.tolist(), ratio_to_previous=ratio,
                   ratio_to_intensity=(sigma2[1:]/sigma2[0]).tolist(), collapse=False)
        if kind == "m_step" and ratio is not None:
            dropped = [c for c, q in enumerate(ratio) if q < 1/self.COLLAPSE_FACTOR]
            if dropped:
                event = dict(level=level, round=round_, channels=dropped, ratios=[ratio[c] for c in dropped],
                             sigma2_before=self._previous.tolist(), sigma2_after=sigma2.tolist())
                if used:
                    row["collapse"] = True
                    self.events.append(event)
                else:
                    self.unused_drops.append(event)
        self.trace.append(row)
        self._previous = sigma2

    @staticmethod
    def _ratio(a, b):
        return None if a is None or b is None else float(np.float64(a)/np.float64(b))

    def _update_nullable(self, sigma2, sigma2_S, *, level, round_, kind, used):
        values = [None if x is None else float(x) for x in sigma2]
        previous = self._previous
        ratio = None if previous is None else [self._ratio(a, b) for a, b in zip(values, previous)]
        row = dict(level=level, round=round_, kind=kind, used=used, sigma2=values, ratio_to_previous=ratio,
                   ratio_to_intensity=[self._ratio(x, values[0]) for x in values[1:]], collapse=False)
        if self.sigma_domain == SUPPORT_SHARED:
            row.update(sigma2_S=sigma2_S, sigma2_S_ratio_to_intensity=self._ratio(sigma2_S, values[0]),
                       sigma2_S_ratio_to_previous=None if previous is None else self._ratio(sigma2_S, self._previous_S))
        if kind == "m_step" and ratio is not None:
            dropped = [c for c, q in enumerate(ratio) if q is not None and q < 1/self.COLLAPSE_FACTOR]
            if dropped:
                event = dict(level=level, round=round_, channels=dropped, ratios=[ratio[c] for c in dropped],
                             sigma2_before=list(previous), sigma2_after=values)
                if used:
                    row["collapse"] = True
                    self.events.append(event)
                else:
                    self.unused_drops.append(event)
        self.trace.append(row)
        self._previous = values
        self._previous_S = sigma2_S

    def record(self):
        out = dict(contract="C11-5 / D15", sigma_collapse_suspected=bool(self.events), events=self.events,
                   criterion=f"an adjacent-round M step lowers some sigma_c^2 by more than {self.COLLAPSE_FACTOR:g}x "
                   "(flag only, no intervention)", unused_m_step_drops=self.unused_drops, trace=self.trace)
        if self.sigma_domain != BRAIN:
            out.update(sigma_domain=self.sigma_domain,
                       null_rule="C04-6 gate 14 / C11-8: a ratio with a null sigma^2 (empty support) is null and sets "
                                 "no flag; the flag looks at the non-empty channels (support) or sigma_S^2 "
                                 "(support_shared)")
        return out


ROUND_EXCEPTIONS = (ArithmeticError, RuntimeError, MemoryError)


def _sch_round_fields(config, ell, is_native, *, accepted_committed, share_end, deadline, rounds_skipped, level_top,
                      round_end):
    """C16 SCH: the round-row keys of the per-level caps and shares (written only with config.sch_records).
    level_top / round_end: WallBudget elapsed readings at the level top and at the round end (None without a budget)."""
    shares = config.level_time_shares
    coarse = shares is not None and not is_native
    return dict(cap_accepted=config.caps.accepted_cap(ell), accepted_committed=accepted_committed,
                level_time_share=shares.shares[ell] if coarse else None,
                level_cumulative_share=shares.cumulative[ell] if coarse else None,
                level_deadline_seconds=shares.deadlines[ell] if coarse else None,
                level_elapsed_seconds=None if round_end is None else round_end-level_top,
                run_elapsed_seconds=round_end, ended_by_level_share=share_end,
                level_share_hit=dict(deadline.hit) if share_end else None, rounds_skipped=rounds_skipped)


def solve_multilevel(backend, levels, *, eta, sigma, config=SolverConfig(), omega0=None, log=None, budget=None,
                     release_inputs=False, init_seed=None, recorder=None):
    """Run the C12 level loop on `levels` (LevelInput, coarse to native) and return a RunResult; `budget` is the
    development wall-clock budget (stopping.WallBudget) or None (module docstring).
    init_seed (C12-4a, WST): the InitSeed of config.start_level 3 (both or neither); module docstring.
    recorder (C12-4a item 8): a checkpoints.ProfileRecorder for the native level, or None; module docstring.
    release_inputs (MEM; memory only, every value unchanged): `levels` must then be a list that the caller hands over
    (it keeps no other reference to the LevelInputs); level ell's entry is replaced by `grid_only` (level and grid)
    as soon as the level's template (samplers and ModelState) is built, so the moving-side channel stack (771 MB at the
    native size with 14 channels) and the coarse levels' arrays are freed while the solve goes on; the ModelStates
    keep the fixed side, mask and tissue they share with the LevelInput."""
    if release_inputs:
        if not isinstance(levels, list):
            raise InputValidationError("release_inputs: levels must be a list handed over by the caller")
    else:
        levels = tuple(levels)
    if len(levels) != len(FACTORS):
        raise InputValidationError("levels: four LevelInput (coarse to native) required")
    shares = config.level_time_shares                  # C13-5a (SCH): None = no share (the code before SCH)
    if shares is not None and (budget is None or budget.seconds != shares.wall_seconds):
        raise InputValidationError("level_time_shares: the run's WallBudget with W = level_time_shares.wall_seconds "
                                   "is required (C13-5a: only with --wall-seconds)")
    sch = config.sch_records                           # C16 SCH: new record keys only when either switch is given
    grids = tuple(lv.grid for lv in levels)
    start_level = config.start_level                   # C12-4a (WST): 0 = the code before WST
    if (start_level != 0) != (init_seed is not None):
        raise InputValidationError("start_level 3 and init_seed go together (C12-4a)")
    if init_seed is not None and (not isinstance(init_seed, InitSeed)
                                  or init_seed.record.get("grid_id") != grids[start_level].grid_id):
        raise InputValidationError("init_seed: an init_field.InitSeed on the native grid (C12-4a)")
    init_record = None
    omega0 = grids[-1].omega if omega0 is None else finite_real(omega0, "omega0", positive=True)
    log = (lambda row: None) if log is None else log
    n_channels = levels[0].fixed.shape[0]
    rounds_out, transfers, timing = [], [], dict(levels=[])
    counters = dict(certificate_calls_before=getattr(backend, "certificate_calls", None))
    commit, state, sigma_post = None, None, None
    last_level = config.max_levels-1
    domain = config.sigma_domain
    support_policy = None                      # C04-6 / C11-8: the M step of sigma_domain support / support_shared
    support_rows = None                        # RunResult.support (None under brain)
    if domain != BRAIN:
        support_policy = SupportMStep.from_policy(sigma, domain)
        if n_channels < 2:
            raise InputValidationError(f"sigma_domain {domain!r}: structure channels required (C04-6: channel_source "
                                       "none is refused)")
        check_theta(config.support_threshold, n_channels-1)        # the exact lower bound with K = K_S (or K_T)
        support_rows = []
    monitor = SigmaMonitor(domain)
    tick_run = time.perf_counter()

    def finish(status, reason, level, native):
        timing["total_seconds"] = time.perf_counter()-tick_run
        counters["certificate_calls"] = getattr(backend, "certificate_calls", None)
        counters["certificate_seconds"] = getattr(backend, "certificate_seconds", None)
        return RunResult(status, reason, RUN_WORDING[status], level, native, commit, state, sigma_post,
                         rounds_out, transfers, timing, counters, monitor.record(),
                         None if budget is None else budget.record(), support=support_rows, init_field=init_record,
                         profile=None if recorder is None else recorder.record())

    if release_inputs:
        for skipped in range(start_level):
            levels[skipped] = grid_only(levels[skipped])     # C12-4a: the skipped levels' arrays go at once (memory only)
    for ell in range(start_level, last_level+1):
        tick = time.perf_counter()
        grid = grids[ell]
        is_native = ell == len(levels)-1
        # C16 SCH: the level top reading of level_elapsed_seconds (before the transfer's budget check); only with the
        # SCH records and a budget, so that no clock is read otherwise
        level_top = budget.elapsed() if sch and budget is not None else None
        if ell > start_level and budget_expired(budget, dict(level=ell, at="before_transfer")):
            # development switch: the previous level's last formal commit stays the run's commit (no native output);
            # checked before the level's template is built (GS review item 6: no template is built for nothing)
            log(dict(kind="budget", level=ell, at="before_transfer", **budget.record()))
            return finish(STOPPED_BEFORE_NATIVE, f"wall-clock budget {budget.seconds:g} s used up before the transfer "
                          f"{ell-1} -> {ell} (--wall-seconds; the last formal commit of level {ell-1} is kept)", ell-1,
                          False)
        memtrace.mark("level.start", level=ell)
        host = _template(levels[ell], eta, omega0, config.tile_size, n_channels, config.cpu_threads,
                         config.logvol_points, **config.logvol_options())
        if release_inputs:
            levels[ell] = grid_only(levels[ell])                    # MEM: the moving stack goes with it
        memtrace.mark("level.template_host", level=ell)
        template = backend.prepare(host)
        memtrace.mark("level.template_prepared", level=ell)
        if init_seed is not None and ell == start_level:
            # C12-4a item 4: the seed through the C12-2 back-off; all rejected: a run failure, u = 0 is never used
            seed, outcome, rows = first_certified(backend, init_seed.u, grid, config.transfer_alphas)
            init_record = dict(init_seed.record, entry_rows=rows, accepted_k=None if seed is None else len(rows)-1,
                               scale=None if seed is None else rows[-1]["scale"],
                               entry_state_hash=None if seed is None else outcome.state_hash,
                               levels_skipped=list(range(start_level)), levels_skipped_reason=SKIP_REASON)
            log(dict(kind="init_field", **init_record))
            if recorder is not None:
                recorder.tag = status_summary(init_record)     # C16: the entry's k, scale and hash in the disp.npz meta (records only)
            if release_inputs:
                init_seed.release()
            if seed is None:
                commit = None
                return finish(INITIALIZATION_FAILED, f"--init-field: all {len(rows)} scaled seeds rejected (C12-4a; "
                              "u = 0 is not used)", ell, False)
        elif ell == 0:
            seed = readonly_copy(np.zeros(grid.shape+(3,)))
            outcome = backend.certify(seed, grid)
            if not outcome.passed:
                commit = None
                return finish(INITIALIZATION_FAILED, f"u = 0 not certified ({outcome.cert_reason})", ell, False)
        else:
            seed, outcome, rows = transfer(backend, commit.u, grids[ell-1], grid, config.transfer_alphas)
            transfers.append(dict(from_level=ell-1, to_level=ell, rows=rows, accepted_k=None if seed is None
                                  else len(rows)-1, source_state_hash=commit.cert.state_hash))
            log(dict(kind="transfer", from_level=ell-1, to_level=ell, rows=rows))
            if seed is None:
                commit = None
                return finish(TRANSFER_REJECTED, f"all {len(rows)} scaled seeds rejected (C12-2)", ell, False)
        memtrace.mark("level.seed_certified", level=ell)
        seed_point, problem = _evaluate(backend, template, seed)
        memtrace.mark("level.seed_evaluated", level=ell)
        if seed_point is None:
            commit = None
            return finish(INITIALIZATION_FAILED, f"level {ell} seed: {problem}", ell, False)
        # MEM: run_pg adopts this read-only theta without copying it.  restrict_free returns a view (`.copy().ravel()`,
        # base not None), so freeze_new falls back to readonly_copy here: still one copy, moved from run_pg to this
        # line (FUP1, review MEM minor 6)
        commit = Commit(seed, freeze_new(restrict_free(seed, grid)), outcome)
        # level_fields: the ModelState fields of this level's frozen support (C04-6), passed to every round's
        # host.replace -- {} under brain (the call of the code before SWLI); no second resident host ModelState (review
        # SWLI important 1: a level-wide copy would keep one more (C, n) float64 beta alive)
        level_extra, level_support, current, level_fields = {}, None, None, {}
        if support_policy is None:
            try:
                sigma2 = sigma.level_start(template, seed_point)
            except FloatingPointError as exc:
                return finish(LEVEL_FAILED, f"level {ell} start: {exc}", ell, is_native)
            del seed_point                # MEM: only the level start reads it (host u and r, the device channel point)
            monitor.update(sigma2, level=ell, round_=0, kind="level_start", used=True)
        else:
            # C04-6 / C11-8: v and the level-start sums on the CPU reference at the seed (any backend); the backend's
            # seed evaluation above only decided that the seed is evaluable
            del seed_point
            level_support = support_level(host, seed, config.support_threshold, sigma_domain=domain, level=ell,
                                          seed_state_hash=outcome.state_hash)
            # SWLM (memory only): the pass's freed temporaries -- two length-n float64 in the main arena, the spline
            # chunks of the CPU workers in theirs -- go back to the OS before the round state and G0 (brain: no call)
            release_free_heap()
            memtrace.mark("level.support", level=ell)
            if isinstance(level_support, InvalidTrial):
                commit = None
                return finish(INITIALIZATION_FAILED, f"level {ell} seed: CPU reference support pass: "
                              f"{level_support.reason}: {level_support.message}", ell, False)
            support_record = level_support.record()
            support_rows.append(support_record)
            level_extra = dict(support_cpu_seconds=level_support.seconds)
            try:
                current = support_policy.level_start(level_support)
            except FloatingPointError as exc:
                log(dict(kind="support", **support_record))
                return finish(LEVEL_FAILED, f"level {ell} start: {exc}", ell, is_native)
            support_record.update(sigma2_start=current.record, sigma2_S_start=current.sigma2_S)
            log(dict(kind="support", **support_record))
            sigma2 = current.internal
            level_fields = dict(sigma_domain=domain, support_threshold=config.support_threshold,
                                support_v=level_support.v)
            monitor.update(current.record, level=ell, round_=0, kind="level_start", used=True,
                           sigma2_S=current.sigma2_S)
        # C13-4a / 5a (SCH): the level's caps (config.caps itself without per-level caps) and its share deadline (coarse
        # levels, shares given; None otherwise)
        level_caps = config.caps.for_level(ell)
        deadline = shares.deadline(budget, ell) if shares is not None and not is_native else None
        level_share_ended = False

        def level_entry(ended_by_share=False):
            entry = dict(level=ell, seconds=time.perf_counter()-tick, **level_extra)
            if sch:
                entry.update(deadline_seconds=None if deadline is None else deadline.deadline,
                             ended_by_level_share=ended_by_share)
            return entry
        for r in range(config.rounds[ell]):
            context = dict(level=ell, round=r)
            tick_round = time.perf_counter()
            profiled = recorder is not None and is_native
            tracker = recorder.tracker(commit) if profiled else CommitTracker(commit)
            try:
                state = backend.prepare(host.replace(sigma2=np.asarray(sigma2), **level_fields))
                memtrace.mark("round.state_prepared", level=ell, round=r)
                if profiled:              # records only: the entry state's L once, then the round's context
                    recorder.begin_round(backend, state, commit, level=ell, round_=r, n_q_value=n_q(state), mask=host.mask,
                                         level_support=level_support)
                g0 = g0_of(backend, state)
                release_free_heap()       # MEM: the threaded CPU G0 leaves freed arena memory; return it before the solve
                memtrace.mark("round.g0", level=ell, round=r)
                target = threshold(g0, abs_tol=config.abs_tol, rel_tol=config.rel_tol)
                result = solve_round(backend, state, commit, option=config.outer, target=target, caps=level_caps,
                                     params=config.proximal, pg=config.pg, metric_mode=config.metric_mode, log=log,
                                     context=context, tracker=tracker, budget=budget, level_deadline=deadline,
                                     lm=config.lm_inner)
            except ROUND_EXCEPTIONS as exc:
                # hard constraint 4: the round's formal commits are kept; the last one is the run's commit
                commit = tracker.commit
                text = f"{type(exc).__name__}: {exc}"
                row = dict(level=ell, round=r, status="exception", reason=text, commits=tracker.commits,
                           state_hash=commit.cert.state_hash, min_jd_lower_bound=commit.cert.min_jd_lower_bound,
                           seconds=time.perf_counter()-tick_round, final=ell == last_level)
                if sch:
                    row.update(_sch_round_fields(config, ell, is_native, accepted_committed=None, share_end=False,
                                                 deadline=deadline, rounds_skipped=0, level_top=level_top,
                                                 round_end=None if budget is None else budget.elapsed()))
                if config.pg.jd_cell_estimate == CERTIFICATE_BERNSTEIN and tracker.ledger is not None:
                    # JDTI review minor 1: the S~ computations the round made before the exception (their check-(b)
                    # violations count in jd_scaling.j53_violations_total); certificate_bernstein only, so that the
                    # other estimates keep the exception row of the code before
                    row["jd_scaling"] = tracker.ledger.record()
                rounds_out.append(row)
                log(dict(kind="round", **row))
                return finish(LEVEL_FAILED, f"level {ell} round {r}: exception {text} (last formal commit kept, "
                              f"{tracker.commits} commits in this round)", ell, is_native)
            commit = result.commit
            if profiled:
                recorder.end_round(commit, result.fb_point)
            memtrace.mark("round.end", level=ell, round=r)
            round_end = budget.elapsed() if sch and budget is not None else None
            budget_end = result.status == BUDGET_EXHAUSTED
            # C13-5a: a coarse level's share ended the round; it is the level's last round (C12-3a SCH item)
            share_end = result.status == LEVEL_BUDGET_REACHED
            if share_end:
                if deadline is None or deadline.hit is None:
                    raise AssertionError("level_budget_reached without a coarse-level deadline (C13-5a)")
                level_share_ended = True
                hit = deadline.hit
                log(dict(kind="level_budget", level=ell, round=r, safe_point=hit["where"].get("at"), where=hit["where"],
                         level_deadline_seconds=deadline.deadline, elapsed_seconds=hit["elapsed_seconds"],
                         check=hit["check"]))
            if budget_end:
                # development switch (D44): the run ends here; not a C13-7 row, no M step
                decision = None
                decision_record = dict(proceed=False, reason="budget_exhausted: wall-clock budget used up "
                                       "(--wall-seconds, development switch); the run ends with this round's last "
                                       "formal commit")
            else:
                decision = continuation(result.status, last_iteration=result.last_iteration,
                                        inner_status=result.inner_status, commit_u=commit.u, commit_cert=commit.cert,
                                        **({} if config.lm_inner is None else dict(inner=INNER_LM)))  # C09.3-13a (h)
                decision_record = decision.record()
            final = (ell == last_level and (r == config.rounds[ell]-1 or share_end)) or budget_end
            row = dict(level=ell, round=r, shape_zyx=list(grid.shape), spacing_xyz=list(grid.spacing),
                       sigma2_used=np.asarray(state.sigma2).tolist(), G0=g0, T_B=target, status=result.status,
                       reason=result.reason, inner_status=result.inner_status, accepted=result.accepted,
                       commits=result.commits, GF_B=result.gradient_norm,
                       GF_B_over_T_B=None if result.gradient_norm is None else result.gradient_norm/target,
                       cap=result.cap, F_B_start=result.energies[0], F_B_end=result.energies[-1],
                       F_B_trace=result.energies, GF_B_trace=result.gradient_norms, metric=result.metric,
                       counters=result.counters.record(), last_iteration=[list(t) for t in result.last_iteration],
                       continuation=decision_record, final=final, seconds=time.perf_counter()-tick_round,
                       state_hash=commit.cert.state_hash, min_jd_lower_bound=commit.cert.min_jd_lower_bound,
                       commit_refusal=result.commit_refusal)
            if support_policy is not None:
                # C04-6 "记录": null for an empty-support channel; sigma_S^2 (support_shared); the diagnostic sigma~_c^2
                row["sigma2_used"] = state.sigma2_record()
                if domain == SUPPORT_SHARED:
                    row["sigma2_S_used"] = current.sigma2_S
                row["sigma_tilde2"] = (None if result.fb_point is None
                                       else SupportMStep.sigma_tilde2(state, result.fb_point, level_support))
            if sch:
                row.update(_sch_round_fields(config, ell, is_native, accepted_committed=result.accepted_committed,
                                             share_end=share_end, deadline=deadline,
                                             rounds_skipped=config.rounds[ell]-r-1 if share_end else 0,
                                             level_top=level_top, round_end=round_end))
            if final and is_native:
                # C08-8(f): the native level's final GF_B on the CPU (the numpy backend's GF_B already is)
                cpu_gf = getattr(backend, "cpu_gradient_norm", None)
                try:
                    row["GF_B_cpu"] = result.gradient_norm if cpu_gf is None else cpu_gf(state, commit.u)
                except ROUND_EXCEPTIONS as exc:
                    row["GF_B_cpu"], row["GF_B_cpu_error"] = None, f"{type(exc).__name__}: {exc}"
                memtrace.mark("round.gf_cpu", level=ell, round=r)
            if budget_end:
                row.update(sigma2_post=None, sigma2_post_skipped="budget_exhausted (no M step)",
                           budget=budget.record() if budget is not None else None)
                if domain == SUPPORT_SHARED:
                    row["sigma2_S_post"] = None
            rounds_out.append(row)
            log(dict(kind="round", **{k: v for k, v in row.items() if k not in ("F_B_trace", "GF_B_trace")}))
            if budget_end:
                timing["levels"].append(level_entry())
                if is_native:
                    return finish(BUDGET_EXHAUSTED, f"native level, round {r}: {result.status} ({result.reason})",
                                  ell, True)
                return finish(STOPPED_BEFORE_NATIVE, f"level {ell} round {r}: {result.status} ({result.reason}); "
                              f"stopped before the native level (--wall-seconds; the last formal commit of level {ell} "
                              "is kept)", ell, False)
            if not decision.proceed:
                if final:
                    row["sigma2_post"] = None
                    if domain == SUPPORT_SHARED:
                        row["sigma2_S_post"] = None
                return finish(LEVEL_FAILED, f"level {ell} round {r}: {decision.reason} ({result.reason})", ell,
                              is_native)
            post_support = None
            try:
                if support_policy is None:
                    post = sigma.m_step(state, result.fb_point)
                else:                          # C11-8 浮点写法 with this level's v and N_c
                    post_support = support_policy.m_step(state, result.fb_point, level_support)
                    post = post_support.internal
            except FloatingPointError as exc:
                if not final:
                    return finish(LEVEL_FAILED, f"level {ell} round {r} M step: {exc}", ell, is_native)
                post = None                    # C11-6: sigma_post is a diagnostic; the ending is not changed
                row["sigma2_post_error"] = f"{type(exc).__name__}: {exc}"
            if support_policy is None:
                row["sigma2_post"] = None if post is None else np.asarray(post).tolist()
            else:
                row["sigma2_post"] = None if post_support is None else post_support.record
                if domain == SUPPORT_SHARED:
                    row["sigma2_S_post"] = None if post_support is None else post_support.sigma2_S
            if post is not None:
                # used by round r + 1 of this level, or the native sigma_post (C11-6); a coarse level's last M step
                # is not used (the next level re-estimates sigma at its start)
                # C13-5a: after a share ending the M step is a diagnostic only (D42: not used, no collapse flag)
                used = (r+1 < config.rounds[ell] and not share_end) or (final and is_native)
                if support_policy is None:
                    monitor.update(post, level=ell, round_=r, kind="m_step", used=used)
                else:
                    monitor.update(post_support.record, level=ell, round_=r, kind="m_step", used=used,
                                   sigma2_S=post_support.sigma2_S)
            if final:
                sigma_post = row["sigma2_post"]
                timing["levels"].append(level_entry(share_end))
                if not is_native:
                    why = f"max_levels = {config.max_levels}: stopped after level {ell}"
                    if share_end:
                        why += f" (round {r} ended level_budget_reached, C13-5a)"
                    return finish(STOPPED_BEFORE_NATIVE, why, ell, False)
                status = final_status(result.status, decision)
                return finish(status, f"native level, round {r}: {result.status} ({result.reason})", ell, True)
            sigma2 = post
            current = post_support
            result = None                 # MEM: the round's last evaluation is not needed by the next round or level
            if share_end:
                break                     # C13-5a / C12-3a: the level's remaining rounds are skipped
        timing["levels"].append(level_entry(level_share_ended))
    raise AssertionError("unreachable")
