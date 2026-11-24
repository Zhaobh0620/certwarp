"""Population empirical Bayes for the prior parameters eta (contract C15; parameterization C05-6 P1/P2/P3;
theory B6, B11, B12 (heuristic), B13).

One EB round (C15-6) on the admitted training pairs p = 1..n_T, eta frozen:
  1. MAP of every pair under eta (the multilevel solver, C09-C13; `multilevel_map`), EM on sigma with a = b = 0
     (C15-5: the training runs themselves have no sigma prior);
  2. at the native output u_hat_p (last formal commit, host ModelState with sigma_used, C11-6): the Laplace
     perturbations of C14 with S = 16 (C14-4), every sample certified (C14-5);
  3. the per-pair statistics (`pair_statistics`):
       kq_quadratic    u_hat^T K_Q u_hat / omega0         = sum_q (omega w_g/omega0) |grad u_h(x_q)|^2  (B7-1)
       curl_quadratic  ||C u_hat||_W^2 / omega0           = sum_q (omega w_g/omega0) |c_q(u_hat)|^2
       kq_trace        tr(K_Q H^-1)/omega0 ~ (1/S_conv) sum_s delta_s^T (K_Q/omega0) delta_s        (C15-3, B6)
       curl_trace      tr(C^T W C H^-1)/omega0, same samples
       tau_hat[k]      sum_q (omega w_g/omega0) Ptilde_T^k(x_q) (log J_q(u_hat))^2
       tau_var[k]      sum_q (omega w_g/omega0) Ptilde_T^k(x_q) Var_s(log J_q)    (PCG-converged AND certified
                       samples, ddof = 1; C15-3)
       tau_den[k]      sum_q (omega w_g/omega0) Ptilde_T^k(x_q)                    k = 0..K_T
     with N_int the interior nodes, n_free = 3 N_int and N_C = 3 N_int (B11, D21); Ptilde_T is the floored
     tissue layer of the fixed image on the native level (state.tissue; the structure channels are not read);
  4. the update (`eb_update`, C15-1/4):
       alpha <- sum_p 3N_int,p / sum_p (kq_quadratic_p + kq_trace_p)         (P1, P3)
       alpha <- alpha_nug(tau_new) = 1e-2 min_k 1/tau_k^2 (k = 0..K_T)        (P2: never learned, C05-6)
       nu    <- sum_p N_C,p / sum_p (curl_quadratic_p + curl_trace_p)
       tau_k^2 <- sum_p (tau_hat_p[k] + tau_var_p[k]) / sum_p tau_den_p[k]   (moment matching, not EM: B12-2)
     tau_0 is updated only when learned (P1; P2 with learn_tau0, D28); otherwise its numerator and denominator
     are recorded (diagnostic) and tau_0 stays bitwise unchanged (C15-4).
Rounds (C15-6): at most 3; the criterion max_j |eta_j^new - eta_j|/eta_j < 5% over the learned components
(alpha, nu and tau_k as standard deviations, the components of eta in C05) ends the iteration with
`eb_criterion_met`; otherwise the trajectory is reported as it is (`eb_round_cap`), nothing is forced.
B12: the block-wise normalization (N) gives larger alpha, nu than the joint form in one step ("over-smoothing");
C15-7 requires the development-set Dice comparison next to the EB eta (an experimental protocol, not done here).

Inclusion of pairs (not fixed by the contract; recorded per pair and per parameter): a pair whose MAP did
not end with a C13-3 ending on the native level (run failures, level_failed: C13-7 says no Laplace) gives no
statistics; a pair with S_conv = 0 gives no traces (alpha, nu); a pair with S_pass < 2 gives no Var_s(log J)
(tau).  Such a pair is left out of that parameter's sums, never filled in.  A learned parameter without any
usable pair ends the EB with `eb_failed`.

sigma prior (C15-5): from the last round's sigma_post of the usable pairs (C11-6), per channel c:
moment matching a_c = m_c^2/v_c + 2, b_c = m_c (a_c - 1) (v_c with ddof = 1, recorded; v_c = 0 or n_T < 3:
no prior, a = b = 0), with the pseudo-count form n0_c = 2 a_c + 2, sigma0_c^2 = b_c/(a_c + 1) (C04-4) and the
share n0_c/N_q (C11-5: about 1e-5 -- the moment-matched prior has practically no effect on the M step).
Option (D29, model change, BZ): pseudo counts n0_c = kappa0 median_p N_q,p, sigma0_c^2 = mean (or median) of
sigma_hat_c^2, a_c = n0_c/2 - 1, b_c = n0_c sigma0_c^2/2.
sigma_domain support / support_shared (C04-6, C15-5 appended note; work package SWLI; draft SWL, D49 / D50 for BZ): the
pairs' MAP and Laplace run with the same sigma_domain and theta (SolverConfig; learn CLI --sigma-domain); PairStatistics
carries `support` (N_q, N_c, N_S of the native state, model.support.support_counts), sigma_used / sigma_post with null for
an empty native support; `sigma_prior_support` matches the prior on the same domain (support: per structure channel on the
pairs with a non-empty support of it, count N_c; support_shared: one (a_S, b_S), count N_S; the intensity as before with
N_q) and records sigma_domain and theta; `check_sigma_prior_domain` admits a frozen prior only for the same domain and
theta (a prior without the keys reads as brain, theta null).  Brain: everything above, bitwise the code before SWLI.

Effective counts (C15-9, D45; theory B17).  EBConfig.eb_count = "gull_mackay" is the default with P2 (BZ 2026-10-30
about 1:50 pm CDT, D45 adopted; EBD).  eb_count = "dim" is the explicit option: the C15-1 update above, bitwise the code
before C15-9 (arrays and eta trajectory; provenance gains only the key eb_count; no prior samples); P1 and P3 need it
explicitly.  eb_count = "gull_mackay" (defined for P2 and P3; P1 is refused; P3 is refused too until a P3 gate and an
alpha lower bound exist, C15-9 / review R2 important 1, so only P2 runs) uses the traces of the joint prior precision
Pi_0 at u = 0 as counts, in the multi-block Gull-MacKay form:
       nu    <- sum_p (g_nu,p - nu t_nu,p) / sum_p q_nu,p          g_nu = nu (1/S0conv) sum_s d0_s^T A_nu d0_s
       alpha <- same with A_alpha (P3; under P2 alpha = alpha_nug(tau_new) as before)
       tau_k^2 <- tau_k^2 sum_p m_hat_k,p / sum_p (m_prior_k,p - m_var_k,p)   (learned k)
  with q, t = curl_quadratic / curl_trace (kq_* for alpha), d0_s = the C15-9 prior samples (bayesreg.infer.laplace.
  PriorSystem: S0 = n_samples PCG solves with Pi_0, seed `prior_round_seed`), rho_qk = P^k(x_q) (tau_k^2/tau_q^2)^2,
  m_prior_k = sum_q c_q rho_qk (1/S0conv) sum_s (div d0_s)_q^2, m_hat_k = sum_q c_q rho_qk (log J_q(u_hat))^2,
  m_var_k = sum_q c_q rho_qk Var_s(log J_q) (c_q = omega w_g/omega0).  A component whose summed Gull-MacKay numerator
  or denominator is <= 0 uses the ratio form of the same statistics (same fixed point, B17-4) and is marked
  `gm_fallback`: nu <- sum g_nu / sum (q_nu + t_nu), tau_k^2 <- tau_k^2 sum (m_hat + m_var) / sum m_prior.
  Pairs: nu (alpha) need usable_precisions and S0conv >= 1, tau needs usable_tau and S0conv >= 1.  Recorded per
  round: the Gull-MacKay, ratio and C15-1 (dim) values of every component, the form used, the z value of the
  difference of independent estimates (the nu / alpha numerator, the tau_k denominator; m_var's standard error by the
  B6 bound m_var sqrt(2/(S_pass - 1))), the P2 coupling flag alpha_nug_coupled (a learned tau_k >= every fixed tau, so
  that alpha_nug = 1e-2/max_k tau_k^2 moves with a learned component and the fixed point is only the alpha-held
  partial stationary point, B17 corollary 4.3; zero in the default case tau_0 = 1 largest), and the ratio of
  adjacent steps (the 5% criterion is a STEP criterion: "步长判据满足", never "convergence").

Log-volume point sets (C15-10, C05-2b "C14, C15"; work package IMPL27b).  Under logvol_points "gauss27+closed27"
with the Simpson weights the tau sums run over Gauss27 U closed27 with the weights c_x of C05-2b: c_q = omega w_g
theta_G/omega0 at the Gauss points (state.logvol_gauss_weights), c_p = omega s_p theta_S/omega0 at the closed27
points (state.closed27_point_weights); per tile the Gauss part, then the closed27 part (C06-7).  tau_hat, tau_var,
tau_den (dim) and m_hat, m_var, m_prior, rho (gull_mackay) all contain the closed27 block; Var_s(log J) at the closed27
points comes from the same samples (posterior_maps logj_var_closed27); the prior samples carry the closed27 noise
(PriorSystem).  tau_den still equals the class volume (both rules are exact on the Q1 Ptilde); the count identity
alpha a + nu c + sum_k m_k/tau_k^2 = d0^T Pi_0 d0 holds per sample (it uses only the -1 homogeneity of Pi_J in
tau^2).  kq_quadratic and curl_quadratic stay on the Gauss points with omega w_g (diffusion, curl unchanged).  With
"gauss27" the weights are ops.weights itself (bitwise the code before).  closed27_weights "uniform" is refused (scope
limit, C15-10), and so is "gauss27+corners" (C05-2a has no tau moment update).

Leak guard (C15-1, D23, D24): `admit_eb_pairs` accepts only pairs listed in splits/bayesreg_v1/eb_pairs.tsv
(ordered: fixed, moving, fold), both subjects in the folds file with that same fold, both in the allowlist
source `bayesreg_train_v1` (not never_for_training) and admitted by the allowlist; `check_structure_models`
requires every fold model behind an image's structure layer to be registered for the OTHER fold and to have
had no label contact with either subject of the pair.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time

import numpy as np

from jd_curl.validation import InputValidationError, finite_real
from jd_curl.three_d.boundary import inject_free
from jd_curl.three_d.cell_q1 import cell_tiles
from jd_curl.three_d.validation import integer_at_least

from ..geom import node_field, scalar_field, tile_geometry, tile_interpolate
from ..infer.laplace import (PRIOR_PCG_NOT_CONVERGED, S_GROUP, LaplaceConfig, LaplaceSystem, PriorSystem,
                             default_trace_operators, iter_perturbations, posterior_maps, seed_from_run_id)
from ..io.guard import CANONICAL_SUBJECT_PATTERNS, DevAllowlist
from ..io.pair import SegmenterRegistry
from ..model import LogVolume, LogVolumeClosed27, ModelState
from ..model.support import BRAIN, SUPPORT, support_counts
from ..seg.unet.data import FOLD_INDEX, FOLD_NAMES, FoldSplit
from ..solve.backend import NumpyBackend
from ..solve.multilevel import (ALPHA_NUG_FACTOR, Eta, GaussianMStep, SolverConfig, SupportMStep, n_q,
                                solve_multilevel)
from ..solve.export import write_json
from ..solve.jd_scaling import CERTIFICATE_BERNSTEIN, j53_violations_total
from ..solve.stopping import (CAP_REACHED, CONVERGED_B, LINE_SEARCH_FAILED, OUTER_INNER_FAILED, OUTER_NO_PROGRESS,
                              OUTER_STAGNATED, RUN_WORDING, STAGNATED, TRIAL_BUDGET_EXHAUSTED)

LD = np.longdouble
_RAISE = dict(over="raise", invalid="raise", divide="raise")

PARAMETRIZATIONS = ("P1", "P2", "P3")       # C05-6 / D9
MAX_ROUNDS = 3                               # C15-6
REL_TOL = .05                                # C15-6
SIGMA_PRIOR_RULES = ("moment_match", "pseudo_count", "none")    # C15-5 / D29
SIGMA0_STATISTICS = ("mean", "median")
TRAINING_SOURCE = "bayesreg_train_v1"        # allowlist source of the B-line training set
EB_PAIRS_HEADER = ("fixed", "moving", "fold")
# C13-3 endings on the native level: the output exists and C14 applies (run failures and level_failed: no
# Laplace, C13-7).
LAPLACE_ENDINGS = (CONVERGED_B, CAP_REACHED, STAGNATED, OUTER_STAGNATED, LINE_SEARCH_FAILED, TRIAL_BUDGET_EXHAUSTED,
                   OUTER_NO_PROGRESS, OUTER_INNER_FAILED)
EB_CRITERION_MET = "eb_criterion_met"
EB_ROUND_CAP = "eb_round_cap"
EB_FAILED = "eb_failed"
EB_STATUSES = (EB_CRITERION_MET, EB_ROUND_CAP, EB_FAILED)
EB_WORDING = {EB_CRITERION_MET: "EB 判据满足（学习分量的最大相对变化 < rel_tol）",
              EB_ROUND_CAP: "EB 轮数上限收尾（判据未满足；如实报告轨迹）",
              EB_FAILED: "EB 失败（某个学习分量没有可用的配对统计量）"}
B12_NOTE = ("B12 (heuristic): the alpha, nu updates are EM fixed points only under (L) Laplace and (N) the block-wise "
            "prior normalization; (N) gives larger alpha, nu than the joint normalization in one step (nu about "
            "1.5-2.2x in NB12); the tau_k^2 update is moment matching, not EM. C15-7: report the development-set "
            "Dice-optimal eta next to the EB eta.")
EB_COUNTS = ("dim", "gull_mackay")           # C15-9 / D45 (dim = C15-1, bitwise)
DEFAULT_EB_COUNT = "gull_mackay"             # C15-9 / D45: default with P2 (BZ 10-30, EBD)
DIM_HINT = ("eb_count defaults to 'gull_mackay' since BZ 10-30 (C15-9 / D45); pass eb_count='dim' (CLI --eb-count "
            "dim) for the C15-1 update")
GM_FORM = "gull_mackay"                      # C15-9 update form
GM_FALLBACK = "gm_fallback"                  # C15-9 ratio form (sample numerator or denominator <= 0)
EB_WORDING_GM = {EB_CRITERION_MET: "EB 步长判据满足（学习分量的最大相对变化 < rel_tol；这是步长判据，不是收敛：收缩率 ρ "
                                   "时剩余距离约 rel_tol·ρ/(1−ρ)，见相邻步长之比）",
                 EB_ROUND_CAP: "EB 轮数上限收尾（步长判据未满足；如实报告轨迹）",
                 EB_FAILED: "EB 失败（某个学习分量没有可用的配对统计量）"}
B17_NOTE = ("C15-9 / B17 (proof draft, R1 review passed, revisions pending R2): in the linear Gaussian model the "
            "fixed points of the Gull-MacKay update and of its ratio form are the stationary points of the evidence in "
            "the learned components; in the nonlinear model they are the stationary points of the joint Laplace "
            "evidence under (L), (L0) (a remark, not a theorem).  Not an EM M step: neither monotone evidence nor "
            "convergence is guaranteed; the 5% rule is a step criterion.")


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------

def default_learn_tau0(parametrization):
    """C15-4: P1 learns tau_0; P2 fixes it by default (D28); P3 fixes it."""
    return parametrization == "P1"


@dataclass(frozen=True)
class EBConfig:
    """C15 / C05-6 switches (all recorded).  learn_tau0 None = the parameterization's default."""
    parametrization: str = "P2"
    learn_tau0: bool | None = None
    max_rounds: int = MAX_ROUNDS
    rel_tol: float = REL_TOL
    n_samples: int = S_GROUP
    alpha_nug_factor: float = ALPHA_NUG_FACTOR
    sigma_prior: str = "moment_match"
    kappa0: float | None = None
    sigma0_statistic: str = "mean"
    variance_ddof: int = 1
    laplace: LaplaceConfig | None = None
    eb_count: str = DEFAULT_EB_COUNT

    def __post_init__(self):
        if self.parametrization not in PARAMETRIZATIONS:
            raise InputValidationError(f"parametrization: one of {PARAMETRIZATIONS}")
        if self.eb_count not in EB_COUNTS:
            raise InputValidationError(f"eb_count: one of {EB_COUNTS} (C15-9)")
        if self.eb_count == "gull_mackay" and self.parametrization == "P1":
            raise InputValidationError("eb_count 'gull_mackay' is defined for P2 and P3 only: under P1 the exact "
                                       "evidence is constant along the B13-3 (a) fibre for class-constant layouts "
                                       f"(no unique target; C15-9, B17 corollary 4.4). {DIM_HINT}")
        if self.eb_count == "gull_mackay" and self.parametrization == "P3":
            raise InputValidationError("eb_count 'gull_mackay' with P3 is not allowed yet (C15-9; review R2 important "
                                       "1): P3 has no linear-Gaussian gate, and on a toy the evidence supremum lies at "
                                       "alpha -> 0 (the iteration drives alpha to 3e-13, no convergence); it is "
                                       f"reopened once a P3 gate and an alpha lower bound exist. Use P2, or {DIM_HINT}")
        learn = default_learn_tau0(self.parametrization) if self.learn_tau0 is None else self.learn_tau0
        if not isinstance(learn, bool):
            raise InputValidationError("learn_tau0: bool or None")
        if self.parametrization == "P1" and not learn:
            raise InputValidationError("P1 learns every component including tau_0 (C15-4); use P3 to fix tau_0")
        if self.parametrization == "P3" and learn:
            raise InputValidationError("P3 fixes tau_0 (C05-6)")
        object.__setattr__(self, "learn_tau0", learn)
        object.__setattr__(self, "max_rounds", integer_at_least(self.max_rounds, "max_rounds"))
        rel = finite_real(self.rel_tol, "rel_tol", positive=True)
        object.__setattr__(self, "rel_tol", rel)
        object.__setattr__(self, "n_samples", integer_at_least(self.n_samples, "n_samples", 2))
        object.__setattr__(self, "alpha_nug_factor", finite_real(self.alpha_nug_factor, "alpha_nug_factor",
                                                                 positive=True))
        if self.sigma_prior not in SIGMA_PRIOR_RULES:
            raise InputValidationError(f"sigma_prior: one of {SIGMA_PRIOR_RULES}")
        if self.sigma_prior == "pseudo_count":
            if self.kappa0 is None:
                raise InputValidationError("sigma_prior 'pseudo_count' needs kappa0 (D29: a model change, BZ decides)")
            object.__setattr__(self, "kappa0", finite_real(self.kappa0, "kappa0", positive=True))
        elif self.kappa0 is not None:
            raise InputValidationError("kappa0 is used only by sigma_prior 'pseudo_count'")
        if self.sigma0_statistic not in SIGMA0_STATISTICS:
            raise InputValidationError(f"sigma0_statistic: one of {SIGMA0_STATISTICS}")
        if self.variance_ddof not in (0, 1):
            raise InputValidationError("variance_ddof: 0 or 1")
        laplace = LaplaceConfig(n_samples=self.n_samples) if self.laplace is None else self.laplace
        if not isinstance(laplace, LaplaceConfig) or laplace.n_samples != self.n_samples:
            raise InputValidationError("laplace: a LaplaceConfig with n_samples equal to EBConfig.n_samples")
        object.__setattr__(self, "laplace", laplace)

    def learned(self, n_tau):
        """Names of the learned components of eta (C15-4/6), n_tau = K_T + 1."""
        names = ["alpha"] if self.parametrization in ("P1", "P3") else []
        names.append("nu")
        names.extend(f"tau_{k}" for k in range(0 if self.learn_tau0 else 1, n_tau))
        return names

    @property
    def gull_mackay(self):
        return self.eb_count == "gull_mackay"

    def record(self, logvol_points=None):
        """The record.  logvol_points: the run's log-volume point set (learn CLI --logvol-points), which only sets
        laplace.noise_order (LaplaceConfig.record): "gauss27+closed27" -> the C14-7 order with logvol_closed27; None or
        "gauss27" -> the C05-2 order, the record byte for byte as before FUP1 (review MIN1 question 5; the EB goldens
        call it without an argument).  The sample stream of each pair is recorded in its posterior_qc.config and
        laplace_system either way."""
        out = dict(contract="C15 / C05-6", parametrization=self.parametrization, learn_tau0=self.learn_tau0,
                   tau0_rule="learned" if self.learn_tau0 else "fixed at eta0's tau_0 (C15-4, D28); numerator and "
                   "denominator recorded", alpha_rule="alpha_nug = alpha_nug_factor min_k 1/tau_k^2 (k = 0..K_T), "
                   "recomputed from the updated tau every round" if self.parametrization == "P2" else
                   "EM update sum 3N_int / sum (u^T K_Q u + tr(K_Q H^-1))/omega0",
                   alpha_nug_factor=self.alpha_nug_factor, max_rounds=self.max_rounds, rel_tol=self.rel_tol,
                   criterion="max_j |eta_j^new - eta_j| / eta_j < rel_tol over the learned components "
                             "(alpha, nu, tau_k as standard deviations)",
                   n_samples=self.n_samples, n_curl="N_C = 3 N_int (B11, D21)",
                   trace_estimator="(1/S_conv) sum_s delta_s^T A delta_s over the PCG-converged samples (C15-3, B6)",
                   logj_variance="ddof = 1 over the PCG-converged and certified samples (C15-3)",
                   sigma_prior=self.sigma_prior, kappa0=self.kappa0, sigma0_statistic=self.sigma0_statistic,
                   variance_ddof=self.variance_ddof, laplace=self.laplace.record(logvol_points=logvol_points),
                   note=B12_NOTE,
                   eb_count=self.eb_count)
        if self.gull_mackay:
            out["gull_mackay"] = dict(
                contract="C15-9 / D45 (default with P2, adopted by BZ 10-30)",
                counts="traces of the joint prior precision Pi_0 at u = 0 (S0 = n_samples prior samples, "
                       "bayesreg.infer.laplace.PriorSystem, DST square root for the diffusion block always)",
                update="nu <- sum(g_nu - nu t_nu)/sum q_nu; alpha the same (P3); tau_k^2 <- tau_k^2 sum m_hat / "
                       "sum (m_prior - m_var), rho_qk = P^k (tau_k^2/tau_q^2)^2",
                fallback="ratio form of the same statistics when a summed numerator or denominator <= 0 "
                         "(gm_fallback)",
                prior_seed="prior_round_seed(run_id, round, pair_id) = seed_from_run_id(run_id/eb_round<r>/<pair>"
                           "/prior)",
                criterion_wording="步长判据满足 (a step criterion, not convergence); adjacent step ratios recorded",
                note=B17_NOTE)
        return out


def alpha_nugget(tau2, factor=ALPHA_NUG_FACTOR):
    """C05-6 / D9: alpha_nug = factor min_k (1/tau_k^2), k = 0..K_T (as solve.multilevel.eta_temporary)."""
    tau2 = np.asarray(tau2, dtype=np.float64)
    with np.errstate(**_RAISE):
        return float(factor*np.min(1/tau2))


def eta_components(eta):
    """eta as the components of C05: alpha, nu, tau_k (standard deviations)."""
    out = dict(alpha=float(eta.alpha), nu=float(eta.nu))
    out.update({f"tau_{k}": float(np.sqrt(t)) for k, t in enumerate(eta.tau2)})
    return out


def relative_changes(old, new, names):
    """{name: |new - old| / old} over `names` and their maximum (C15-6)."""
    a, b = eta_components(old), eta_components(new)
    changes = {}
    for name in names:
        if a[name] <= 0:
            raise InputValidationError(f"eta component {name} = {a[name]} must be positive for a relative change")
        changes[name] = abs(b[name]-a[name])/a[name]
    return changes, max(changes.values())


# ----------------------------------------------------------------------------------------------
# Per-pair statistics
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GaussStatistics:
    """One pass over the Gauss points of u_hat (longdouble sums over tiles in ascending order, C06-7)."""
    kq_quadratic: float          # sum_q (omega w_g/omega0) |grad u|^2 = u^T K_Q u / omega0 (B7-1)
    curl_quadratic: float        # sum_q (omega w_g/omega0) |c_q|^2   = ||C u||_W^2 / omega0
    tau_hat: tuple               # (K_T+1,) sum_q W P^k (log J_q)^2
    tau_var: tuple | None        # (K_T+1,) sum_q W P^k Var_s(log J_q); None without the sample variance
    tau_den: tuple               # (K_T+1,) sum_q W P^k
    n_gauss: int
    rho_hat: tuple | None = None     # C15-9 (rho=True): (K_T+1,) sum_q W rho_qk (log J_q)^2   = m_hat
    rho_var: tuple | None = None     # C15-9: sum_q W rho_qk Var_s(log J_q) = m_var (None without the variance)
    rho_den: tuple | None = None     # C15-9: sum_q W rho_qk (diagnostic)
    n_closed27: int | None = None    # C15-10: closed27 points summed (None unless logvol_points "gauss27+closed27")


def require_eb_logvol(state):
    """The log-volume point sets C15 covers: "gauss27" (C05-2) and "gauss27+closed27" with the Simpson weights (C15-10,
    C05-2b).  Refused: closed27_weights "uniform" (a scope limit of C15-10: its premises were checked for the Simpson
    weights only -- uniform weights keep Pi_0 symmetric positive definite, but the class-constant fibre breaks by a
    quadrature artifact (rank K+2 -> K+3) and the B17-5 spectral bound fails) and "gauss27+corners" (C05-2a has no
    tau moment update)."""
    if state.logvol_points == "gauss27+closed27" and state.closed27_weights == "uniform":
        raise InputValidationError("EB (C15-10): closed27_weights = 'uniform' is refused (scope limit of C15-10: its "
                                   "premises were checked for the Simpson weights only; an ablation switch of the "
                                   "MAP solve, C05-2b (iv))")
    if state.logvol_points not in ("gauss27", "gauss27+closed27"):
        raise InputValidationError(f"EB (C15): logvol_points = {state.logvol_points!r} is not covered by the contract "
                                   "(the C05-2a corner term has no tau moment update); 'gauss27' or "
                                   "'gauss27+closed27' (Simpson)")


def rho_weights(state, tissue, weight, tile, block=LogVolume):
    """C15-9: c_x rho_xk = W P^k(x) (tau_k^2/tau_x^2)^2 on one tile, k = 0..K_T, as a list of (cells, P) arrays at the
    points of `block` (LogVolume: the Gauss points, C05-2; LogVolumeClosed27: the closed27 points, C15-10); tau_x^2 is
    the block's own interpolation (block.tau2) and tau_k^2 the state's eta; W the block's weights c_x (1, P).
    Returns (list of W P^k, list of W rho_k)."""
    tau2_q = block.tau2(state, tile)
    ops = block.operators(state)
    plain, rho = [], []
    with np.errstate(**_RAISE):
        for k, field_k in enumerate(tissue):
            wp = weight*tile_interpolate(field_k, ops, tile=tile)
            ratio = state.tau2[k]/tau2_q
            plain.append(wp)
            rho.append(wp*(ratio*ratio))
    return plain, rho


def gauss_point_statistics(state, u_hat, logj_var=None, *, rho=False, logj_var_closed27=None):
    """The u_hat terms of C15-1 at the Gauss27 points of `state`'s grid.

    W = omega w_g / omega0 (ops.weights / omega0); P^k = Q1 interpolation of state.tissue[k] (the floored
    tissue layer Ptilde_T of the fixed image, C05-2); log J_q of u_hat (every J_q > 0 required: u_hat is a
    certified state).  logj_var: (n_cells, 27) ddof = 1 sample variance of log J_q (posterior_maps
    keep_gauss_variance), or None.  Reads nothing of the likelihood channels.  The C15 tau update is written for
    the C05-2 log-volume prior: a state with the C05-2a corner term (logvol_points = 'gauss27+corners') is refused.
    rho=True (C15-9, eb_count "gull_mackay") also sums the same terms with the weights W rho_qk of `rho_weights`
    (rho_hat = m_hat, rho_var = m_var); the C15-1 sums are computed exactly as without it.
    C15-10 (logvol_points "gauss27+closed27", Simpson): the tau sums (plain and rho) run over Gauss27 U closed27 with
    c_q = state.logvol_gauss_weights/omega0 (theta_G) and c_p = state.closed27_point_weights/omega0, per tile the Gauss
    part then the closed27 part; logj_var_closed27 (n_cells, 27) is the sample variance at the closed27 points
    (posterior_maps logj_var_closed27; given together with logj_var).  kq_quadratic and curl_quadratic keep omega w_g.
    With "gauss27" the tau weights are ops.weights itself (bitwise the code before)."""
    if not isinstance(state, ModelState):
        raise InputValidationError("state: ModelState required")
    require_eb_logvol(state)
    grid, ops = state.grid, state.ops
    closed = state.logvol_closed27
    u = node_field(u_hat, grid, name="u_hat")
    n_tau = state.tissue.shape[0]
    tissue = [scalar_field(state.tissue[k], grid, name=f"tissue[{k}]") for k in range(n_tau)]
    if logj_var is not None:
        logj_var = np.asarray(logj_var, dtype=np.float64)
        if logj_var.shape != (grid.n_cells, len(ops.points)) or not np.all(np.isfinite(logj_var)) \
                or np.any(logj_var < 0):
            raise InputValidationError(f"logj_var: finite nonnegative ({grid.n_cells}, {len(ops.points)}) required")
    if closed:
        ops_s = state.closed27_ops
        if (logj_var is None) != (logj_var_closed27 is None):
            raise InputValidationError("logj_var_closed27: given together with logj_var under C05-2b (C15-10)")
        if logj_var_closed27 is not None:
            logj_var_closed27 = np.asarray(logj_var_closed27, dtype=np.float64)
            if logj_var_closed27.shape != (grid.n_cells, len(ops_s.points)) \
                    or not np.all(np.isfinite(logj_var_closed27)) or np.any(logj_var_closed27 < 0):
                raise InputValidationError(f"logj_var_closed27: finite nonnegative ({grid.n_cells}, "
                                           f"{len(ops_s.points)}) required")
    elif logj_var_closed27 is not None:
        raise InputValidationError("logj_var_closed27: only with logvol_points 'gauss27+closed27' (C15-10)")
    with np.errstate(**_RAISE):
        weight = (ops.weights/state.omega0)[None, :]
        wlog = (state.logvol_gauss_weights/state.omega0)[None, :]     # c_q (= weight's values under "gauss27")
        wlog_s = (state.closed27_point_weights/state.omega0)[None, :] if closed else None    # c_p (C15-10)
    grad_sq, curl_sq = LD(0), LD(0)
    hat, var, den = (np.zeros(n_tau, dtype=LD) for _ in range(3))
    r_hat, r_var, r_den = (np.zeros(n_tau, dtype=LD) for _ in range(3))
    for tile in cell_tiles(grid, state.tile_size):
        geo = tile_geometry(u, ops, tile=tile)
        if np.any(geo.J <= 0):
            raise InputValidationError("u_hat: a floating J_q <= 0 (the output must be a certified state)")
        with np.errstate(**_RAISE):
            logj2 = np.log(geo.J)**2
            grad_sq += np.sum(weight[..., None, None]*geo.grad**2, dtype=LD)
            curl_sq += np.sum(weight[..., None]*geo.curl**2, dtype=LD)
            for k in range(n_tau):
                wp = wlog*tile_interpolate(tissue[k], ops, tile=tile)
                den[k] += np.sum(wp, dtype=LD)
                hat[k] += np.sum(wp*logj2, dtype=LD)
                if logj_var is not None:
                    var[k] += np.sum(wp*logj_var[tile], dtype=LD)
            if rho:
                _, weights_rho = rho_weights(state, tissue, wlog, tile)
                for k, wr in enumerate(weights_rho):
                    r_den[k] += np.sum(wr, dtype=LD)
                    r_hat[k] += np.sum(wr*logj2, dtype=LD)
                    if logj_var is not None:
                        r_var[k] += np.sum(wr*logj_var[tile], dtype=LD)
        if closed:                              # C15-10: the closed27 part of the tile, right after its Gauss part
            geo_s = tile_geometry(u, ops_s, tile=tile)
            if np.any(geo_s.J <= 0):
                raise InputValidationError("u_hat: a floating J <= 0 at a closed27 point (the output must be a "
                                           "certified state)")
            with np.errstate(**_RAISE):
                logj2_s = np.log(geo_s.J)**2
                for k in range(n_tau):
                    wp = wlog_s*tile_interpolate(tissue[k], ops_s, tile=tile)
                    den[k] += np.sum(wp, dtype=LD)
                    hat[k] += np.sum(wp*logj2_s, dtype=LD)
                    if logj_var_closed27 is not None:
                        var[k] += np.sum(wp*logj_var_closed27[tile], dtype=LD)
                if rho:
                    _, weights_rho = rho_weights(state, tissue, wlog_s, tile, block=LogVolumeClosed27)
                    for k, wr in enumerate(weights_rho):
                        r_den[k] += np.sum(wr, dtype=LD)
                        r_hat[k] += np.sum(wr*logj2_s, dtype=LD)
                        if logj_var_closed27 is not None:
                            r_var[k] += np.sum(wr*logj_var_closed27[tile], dtype=LD)
    extra = {}
    if rho:
        extra = dict(rho_hat=tuple(float(x) for x in r_hat), rho_den=tuple(float(x) for x in r_den),
                     rho_var=None if logj_var is None else tuple(float(x) for x in r_var))
    if closed:
        extra["n_closed27"] = grid.n_cells*len(ops_s.points)
    return GaussStatistics(kq_quadratic=float(grad_sq), curl_quadratic=float(curl_sq),
                           tau_hat=tuple(float(x) for x in hat),
                           tau_var=None if logj_var is None else tuple(float(x) for x in var),
                           tau_den=tuple(float(x) for x in den), n_gauss=grid.n_cells*len(ops.points), **extra)


@dataclass(frozen=True)
class PairStatistics:
    """C15-1 statistics of one pair at its native output (see the module docstring)."""
    pair_id: str
    n_free: int                  # 3 N_int (the alpha count)
    n_curl: int                  # N_C = 3 N_int (B11, D21)
    n_q: float                   # N_q = <1>_B of the native level
    kq_quadratic: float
    curl_quadratic: float
    kq_trace: float              # NaN without PCG-converged samples
    kq_trace_stderr: float
    curl_trace: float
    curl_trace_stderr: float
    tau_hat: tuple
    tau_var: tuple | None        # None when S_pass < 2
    tau_den: tuple
    sigma_used: tuple
    sigma_post: tuple | None
    n_samples: int
    n_converged: int
    n_passed: int
    pass_rate: float
    pass_rate_low: bool
    gm: "GMStatistics | None" = None     # C15-9 (eb_count "gull_mackay") only
    support: dict | None = None          # sigma_domain != brain (C04-6, C15-5): model.support.support_counts of the
    #                                      native state (sigma_domain, theta, N_q, N_c, N_S); None under brain

    @property
    def sigma_domain(self):
        return BRAIN if self.support is None else self.support["sigma_domain"]

    @property
    def usable_precisions(self):
        return self.n_converged >= 1 and np.isfinite(self.kq_trace) and np.isfinite(self.curl_trace)

    @property
    def usable_tau(self):
        return self.n_passed >= 2 and self.tau_var is not None

    def record(self):
        out = dict(pair_id=self.pair_id, n_free=self.n_free, n_curl=self.n_curl, n_q=self.n_q,
                   kq_quadratic=self.kq_quadratic, curl_quadratic=self.curl_quadratic, kq_trace=self.kq_trace,
                   kq_trace_stderr=self.kq_trace_stderr, curl_trace=self.curl_trace,
                   curl_trace_stderr=self.curl_trace_stderr, tau_hat=list(self.tau_hat),
                   tau_var=None if self.tau_var is None else list(self.tau_var), tau_den=list(self.tau_den),
                   sigma_used=list(self.sigma_used), sigma_post=None if self.sigma_post is None else
                   list(self.sigma_post), n_samples=self.n_samples, S_conv=self.n_converged, S_pass=self.n_passed,
                   pass_rate=self.pass_rate, posterior_cert_pass_rate_low=self.pass_rate_low,
                   usable_precisions=bool(self.usable_precisions), usable_tau=bool(self.usable_tau))
        if self.gm is not None:
            out["gull_mackay"] = self.gm.record()
        if self.support is not None:
            out["support"] = dict(self.support)
        return out


def _mean_stderr(values):
    """Mean and standard error (ddof = 1; NaN for fewer than 2 values) of a list of floats."""
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values)) if values.size else float("nan")
    stderr = float(np.std(values, ddof=1)/np.sqrt(values.size)) if values.size >= 2 else float("nan")
    return mean, stderr


@dataclass(frozen=True)
class GMStatistics:
    """C15-9 statistics of one pair in one round (eb_count "gull_mackay").  Prior side (S0 samples of N(0, Pi_0^-1)
    at the round's eta): g_alpha = alpha mean d0^T A_alpha d0, g_nu = nu mean d0^T A_nu d0, m_prior[k] =
    sum_q c_q rho_qk mean (div d0)_q^2, with their sample standard errors; posterior side (the C14 samples of
    pair_statistics): m_hat[k] = sum_q c_q rho_qk (log J_q(u_hat))^2, m_var[k] = sum_q c_q rho_qk Var_s(log J_q)
    (None when S_pass < 2) and the B6 bound m_var sqrt(2/(S_pass - 1)) of its standard error.  eta: the (alpha, nu,
    tau2) the statistics were computed under (eb_update checks it against the round's eta)."""
    n_prior_samples: int
    n_prior_converged: int
    g_alpha: float
    g_alpha_stderr: float
    g_nu: float
    g_nu_stderr: float
    m_prior: tuple
    m_prior_stderr: tuple
    m_hat: tuple
    m_var: tuple | None
    m_var_stderr_bound: tuple | None
    alpha: float
    nu: float
    tau2: tuple
    count_check: float           # (g_alpha + g_nu + sum_k m_prior_k/tau_k^2) / 3N_int: expectation 1 (B17-3 (b))
    identity_max_rel: float      # max_s |sum_i g_i,s - d0_s^T Pi_0 d0_s| / d0_s^T Pi_0 d0_s (C15-9 gate (iii))
    seed: int

    @property
    def usable(self):
        return self.n_prior_converged >= 1 and np.isfinite(self.g_nu) and np.isfinite(self.g_alpha) \
            and all(np.isfinite(m) for m in self.m_prior)

    def record(self):
        return dict(contract="C15-9", S0=self.n_prior_samples, S0_conv=self.n_prior_converged,
                    g_alpha=self.g_alpha, g_alpha_stderr=self.g_alpha_stderr, g_nu=self.g_nu,
                    g_nu_stderr=self.g_nu_stderr, m_prior=list(self.m_prior), m_prior_stderr=list(self.m_prior_stderr),
                    m_hat=list(self.m_hat), m_var=None if self.m_var is None else list(self.m_var),
                    m_var_stderr_bound=None if self.m_var_stderr_bound is None else list(self.m_var_stderr_bound),
                    eta=dict(alpha=self.alpha, nu=self.nu, tau2=list(self.tau2)), count_check=self.count_check,
                    count_check_note="(g_alpha + g_nu + sum_k m_prior_k/tau_k^2)/3N_int; per sample = d0^T Pi_0 d0 / "
                                     "3N_int, expectation 1, relative sd sqrt(2/(3N_int S0)); recorded, never a "
                                     "trigger", identity_max_rel=self.identity_max_rel, seed=self.seed,
                    usable=bool(self.usable))


def prior_statistics(state, *, seed, config=None, backend=None, log=None):
    """C15-9 prior side of one pair in one round: S0 = config.n_samples samples d0_s = Pi_0^-1 b0_s (PriorSystem,
    PCG as C14-3; an unconverged sample is `prior_pcg_not_converged`, counted, in no statistic), then per converged
    sample a_s = d0^T A_alpha d0 (stiffness_apply_scaled), c_s = d0^T A_nu d0 (curl_gram_apply), m_s[k] =
    sum_q c_q rho_qk (div d0_s)_q^2 (longdouble per tile, ascending) and p_s = d0^T Pi_0 d0 (one Pi_0 product, for
    the identity alpha a_s + nu c_s + sum_k m_s[k]/tau_k^2 = p_s).  Returns (dict of the GMStatistics prior fields,
    record).  The weights of m_s are c_q = state.logvol_gauss_weights/omega0 (ops.weights itself under "gauss27");
    under C05-2b (C15-10) m_s also sums c_p rho_pk (div d0_s)_p^2 over the closed27 points, per tile after the Gauss
    part."""
    config = EBConfig() if config is None else config
    if not isinstance(state, ModelState):
        raise InputValidationError("state: ModelState required")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise InputValidationError("seed: an integer (prior_round_seed) required")
    tick = time.perf_counter()
    system = PriorSystem(state, backend=backend, config=config.laplace)
    operators = default_trace_operators(state)
    grid, ops = state.grid, state.ops
    n_tau = state.tissue.shape[0]
    tissue = [scalar_field(state.tissue[k], grid, name=f"tissue[{k}]") for k in range(n_tau)]
    tiles = list(cell_tiles(grid, state.tile_size))
    closed = state.logvol_closed27
    with np.errstate(**_RAISE):
        weight = (state.logvol_gauss_weights/state.omega0)[None, :]   # c_q (ops.weights itself under "gauss27")
        weight_s = (state.closed27_point_weights/state.omega0)[None, :] if closed else None    # c_p (C15-10)
    tau2 = np.asarray(state.tau2, dtype=np.float64)
    a_values, c_values, m_values, rows = [], [], [], []
    identity = 0.
    n_total = 0
    for item in iter_perturbations(system, int(seed), config.n_samples):
        n_total += 1
        row = item.record()
        if not item.converged:
            row.update(excluded=PRIOR_PCG_NOT_CONVERGED)
            rows.append(row)
            if log is not None:
                log(dict(event="prior_sample", **row))
            continue
        d = item.delta
        field_d = node_field(inject_free(d, grid), grid, name="prior sample")
        m = np.zeros(n_tau, dtype=LD)
        with np.errstate(**_RAISE):
            a = float(d @ np.asarray(operators["KQ_over_omega0"](d), dtype=np.float64))
            c = float(d @ np.asarray(operators["CtWC_over_omega0"](d), dtype=np.float64))
            for tile in tiles:
                div2 = tile_geometry(field_d, ops, tile=tile).div**2
                _, weights_rho = rho_weights(state, tissue, weight, tile)
                for k, wr in enumerate(weights_rho):
                    m[k] += np.sum(wr*div2, dtype=LD)
                if closed:
                    div2_s = tile_geometry(field_d, state.closed27_ops, tile=tile).div**2
                    _, weights_rho = rho_weights(state, tissue, weight_s, tile, block=LogVolumeClosed27)
                    for k, wr in enumerate(weights_rho):
                        m[k] += np.sum(wr*div2_s, dtype=LD)
            p = float(d @ system.hvp(d))
            total = state.alpha*a+state.nu*c+float(np.sum(m/tau2.astype(LD), dtype=LD))
            rel = abs(total-p)/p
        identity = max(identity, rel)
        a_values.append(a)
        c_values.append(c)
        m_values.append([float(x) for x in m])
        row.update(excluded=None, quad_alpha=a, quad_nu=c, m_div=[float(x) for x in m], pi0_quadratic=p,
                   identity_rel=rel)
        rows.append(row)
        if log is not None:
            log(dict(event="prior_sample", **row))
    n_conv = len(a_values)
    a_mean, a_se = _mean_stderr(a_values)
    c_mean, c_se = _mean_stderr(c_values)
    m_stats = [_mean_stderr([row[k] for row in m_values]) for k in range(n_tau)]
    g_alpha, g_alpha_se = state.alpha*a_mean, state.alpha*a_se
    g_nu, g_nu_se = state.nu*c_mean, state.nu*c_se
    m_prior = tuple(s[0] for s in m_stats)
    n_free = 3*grid.n_free_scalar
    with np.errstate(invalid="ignore"):
        count_check = (g_alpha+g_nu+sum(m/t for m, t in zip(m_prior, tau2)))/n_free if n_conv else float("nan")
    fields = dict(n_prior_samples=n_total, n_prior_converged=n_conv, g_alpha=g_alpha, g_alpha_stderr=g_alpha_se,
                  g_nu=g_nu, g_nu_stderr=g_nu_se, m_prior=m_prior, m_prior_stderr=tuple(s[1] for s in m_stats),
                  alpha=float(state.alpha), nu=float(state.nu), tau2=tuple(float(t) for t in tau2),
                  count_check=float(count_check), identity_max_rel=float(identity) if n_conv else float("nan"),
                  seed=int(seed))
    iterations = np.array([r["pcg_iterations"] for r in rows])
    residuals = np.array([r["pcg_rel_residual"] for r in rows])
    record = dict(contract="C15-9 prior side", system=system.record(), seed=int(seed), S0=n_total, S0_conv=n_conv,
                  prior_pcg_not_converged=n_total-n_conv, pcg_iterations_min=int(iterations.min()),
                  pcg_iterations_median=float(np.median(iterations)), pcg_iterations_max=int(iterations.max()),
                  pcg_rel_residual_max=float(residuals.max()), identity_max_rel=fields["identity_max_rel"],
                  count_check=fields["count_check"], samples=rows, seconds=time.perf_counter()-tick)
    return fields, record


def pair_statistics(state, u_hat, *, pair_id, config=None, seed, backend=None, sigma_post=None, log=None,
                    prior_seed=None):
    """Steps 2-3 of one EB round for one pair: C14 perturbations at u_hat (S = config.n_samples, PCG, every
    sample certified), the sample traces and Var_s(log J_q), then the u_hat terms.

    state: the host ModelState of the native output (sigma_used, C11-6; eta of this round); u_hat: its last
    formal commit; seed: an integer (C14-4; `run_population_eb` derives it from the run id).
    eb_count "gull_mackay" (C15-9) also needs prior_seed (`prior_round_seed`): the prior side (`prior_statistics`)
    and the rho-weighted posterior sums; the result carries them as PairStatistics.gm, the record as "prior".
    Returns (PairStatistics, record) -- record holds the posterior qc of C14-5 (pass rate etc.)."""
    config = EBConfig() if config is None else config
    if config.gull_mackay and prior_seed is None:
        raise InputValidationError("eb_count 'gull_mackay': prior_seed (prior_round_seed) required (C15-9)")
    if not config.gull_mackay and prior_seed is not None:
        raise InputValidationError("prior_seed is used only by eb_count 'gull_mackay'")
    tick = time.perf_counter()
    system = LaplaceSystem(state, u_hat, backend=backend, config=config.laplace)
    operators = default_trace_operators(state)
    maps = posterior_maps(state, system.u_hat, iter_perturbations(system, seed, config.n_samples),
                          backend=system.backend, config=config.laplace, trace_operators=operators,
                          keep_cell_logj=False, keep_gauss_variance=True, log=log)
    variance = maps.logj_var_gauss if maps.n_passed >= 2 else None
    variance_s = maps.logj_var_closed27 if maps.n_passed >= 2 else None       # C15-10 (None under "gauss27")
    if config.gull_mackay:
        gauss = gauss_point_statistics(state, system.u_hat, variance, rho=True, logj_var_closed27=variance_s)
    else:
        gauss = gauss_point_statistics(state, system.u_hat, variance, logj_var_closed27=variance_s)
    kq, curl = maps.traces["KQ_over_omega0"], maps.traces["CtWC_over_omega0"]
    n_free = 3*state.grid.n_free_scalar
    gm, prior_record = None, None
    if config.gull_mackay:
        prior_fields, prior_record = prior_statistics(state, seed=prior_seed, config=config, backend=system.backend,
                                                      log=log)
        bound = None
        if gauss.rho_var is not None:
            factor = np.sqrt(2/(maps.n_passed-1))
            bound = tuple(float(v*factor) for v in gauss.rho_var)
        gm = GMStatistics(m_hat=gauss.rho_hat, m_var=gauss.rho_var, m_var_stderr_bound=bound, **prior_fields)
    stats = PairStatistics(
        pair_id=str(pair_id), n_free=n_free, n_curl=n_free, n_q=n_q(state), kq_quadratic=gauss.kq_quadratic,
        curl_quadratic=gauss.curl_quadratic, kq_trace=kq.mean, kq_trace_stderr=kq.stderr, curl_trace=curl.mean,
        curl_trace_stderr=curl.stderr, tau_hat=gauss.tau_hat, tau_var=gauss.tau_var, tau_den=gauss.tau_den,
        sigma_used=(tuple(state.sigma2_record()) if state.has_support else tuple(float(s) for s in state.sigma2)),
        sigma_post=None if sigma_post is None else tuple(None if s is None else float(s) for s in sigma_post),
        n_samples=maps.n_samples, n_converged=maps.n_converged, n_passed=maps.n_passed, pass_rate=maps.pass_rate,
        pass_rate_low=maps.pass_rate_low, gm=gm, support=support_counts(state))
    qc = {k: v for k, v in maps.qc.items() if k not in ("pcg_bias_note", "semantics_note")}
    record = dict(statistics=stats.record(), posterior_qc=qc, samples=list(maps.samples),
                  laplace_system=system.record(),
                  seed=int(seed), timing=dict(maps.timing, total_seconds=time.perf_counter()-tick),
                  n_gauss=gauss.n_gauss)
    if prior_record is not None:
        record["prior"] = prior_record
        record["rho_den"] = list(gauss.rho_den)
    if gauss.n_closed27 is not None:
        record["n_closed27"] = gauss.n_closed27
    return stats, record


# ----------------------------------------------------------------------------------------------
# The update (C15-1/4) and the sigma prior (C15-5)
# ----------------------------------------------------------------------------------------------

class EBUpdateFailed(Exception):
    """A learned component has no usable pair (status eb_failed)."""


def precision_update(counts, quadratics, traces):
    """C15-1: sum counts / sum (quadratic + trace) in longdouble; returns (value, numerator, denominator)."""
    counts, quadratics, traces = (np.asarray(x, dtype=np.float64) for x in (counts, quadratics, traces))
    if counts.size == 0 or not (counts.shape == quadratics.shape == traces.shape):
        raise InputValidationError("precision_update: equal nonempty lists required")
    with np.errstate(**_RAISE):
        numerator = np.sum(counts, dtype=LD)
        denominator = np.sum(quadratics, dtype=LD)+np.sum(traces, dtype=LD)
    if not np.isfinite(float(denominator)) or denominator <= 0:
        raise EBUpdateFailed(f"precision update: denominator {float(denominator)} is not positive")
    return float(numerator/denominator), float(numerator), float(denominator)


def tau2_update(numerators, denominators):
    """C15-1: tau_k^2 = sum_p num_p / sum_p den_p (longdouble); returns (value, numerator, denominator)."""
    numerators, denominators = np.asarray(numerators, dtype=np.float64), np.asarray(denominators, dtype=np.float64)
    if numerators.size == 0 or numerators.shape != denominators.shape:
        raise InputValidationError("tau2_update: equal nonempty lists required")
    num, den = np.sum(numerators, dtype=LD), np.sum(denominators, dtype=LD)
    if not np.isfinite(float(den)) or den <= 0 or not np.isfinite(float(num)) or num <= 0:
        raise EBUpdateFailed(f"tau^2 update: numerator {float(num)} / denominator {float(den)} not positive")
    return float(num/den), float(num), float(den)


def eb_update(eta, statistics, config=None):
    """C15-1/4: the new eta from this round's per-pair statistics (None entries = pairs without statistics).
    Returns (Eta, record); raises EBUpdateFailed when a learned component has no usable pair.
    eb_count "gull_mackay": `gull_mackay_update` (C15-9) instead."""
    config = EBConfig() if config is None else config
    if config.gull_mackay:
        return gull_mackay_update(eta, statistics, config)
    if not isinstance(eta, Eta):
        raise InputValidationError("eta: Eta required")
    n_tau = len(eta.tau2)
    rows = [s for s in statistics if s is not None]
    for s in rows:
        if not isinstance(s, PairStatistics):
            raise InputValidationError("statistics: PairStatistics (or None) required")
        if len(s.tau_den) != n_tau:
            raise InputValidationError(f"pair {s.pair_id}: {len(s.tau_den)} tissue classes, eta has {n_tau}")
    learned = config.learned(n_tau)
    prec = [s for s in rows if s.usable_precisions]
    taus = [s for s in rows if s.usable_tau]
    record = dict(pairs_with_statistics=[s.pair_id for s in rows],
                  pairs_used_alpha_nu=[s.pair_id for s in prec], pairs_used_tau=[s.pair_id for s in taus],
                  pairs_excluded=[dict(pair_id=s.pair_id, alpha_nu=not s.usable_precisions, tau=not s.usable_tau,
                                       S_conv=s.n_converged, S_pass=s.n_passed) for s in rows
                                  if not (s.usable_precisions and s.usable_tau)])
    candidates, unavailable = {}, {}

    def candidate(name, compute):
        """A learned component's update must succeed; a diagnostic one (not learned) is recorded or skipped."""
        try:
            candidates[name] = compute()
        except EBUpdateFailed as exc:
            if name in learned:
                raise
            unavailable[name] = str(exc)

    def precision(counts, quadratics, traces):
        value, num, den = precision_update(counts, quadratics, traces)
        return dict(value=value, numerator=num, denominator=den, quadratic=float(np.sum(quadratics, dtype=LD)),
                    trace=float(np.sum(traces, dtype=LD)))

    def tau_k(k):
        value, num, den = tau2_update([s.tau_hat[k]+s.tau_var[k] for s in taus], [s.tau_den[k] for s in taus])
        return dict(tau2=value, tau=float(np.sqrt(value)), numerator=num, denominator=den,
                    hat_part=float(np.sum([s.tau_hat[k] for s in taus], dtype=LD)),
                    variance_part=float(np.sum([s.tau_var[k] for s in taus], dtype=LD)))
    if prec:
        candidate("alpha", lambda: precision([s.n_free for s in prec], [s.kq_quadratic for s in prec],
                                             [s.kq_trace for s in prec]))
        candidate("nu", lambda: precision([s.n_curl for s in prec], [s.curl_quadratic for s in prec],
                                          [s.curl_trace for s in prec]))
    if taus:
        for k in range(n_tau):
            candidate(f"tau_{k}", lambda k=k: tau_k(k))
    if not prec:
        raise EBUpdateFailed("no usable pair for nu" + (" and alpha" if "alpha" in learned else "") +
                             " (a pair needs S_conv >= 1)")
    if not taus:
        raise EBUpdateFailed("no usable pair for tau (a pair needs S_pass >= 2)")
    tau2 = list(eta.tau2)
    for k in range(n_tau):
        if f"tau_{k}" in learned:
            tau2[k] = candidates[f"tau_{k}"]["tau2"]
    nu = candidates["nu"]["value"]
    if config.parametrization == "P2":
        alpha = alpha_nugget(tau2, config.alpha_nug_factor)
        alpha_source = "alpha_nug(tau_new)"
    else:
        alpha = candidates["alpha"]["value"]
        alpha_source = "EM update"
    new = Eta(alpha, nu, tuple(tau2), source=f"EB update ({config.parametrization})")
    record.update(learned=learned, candidates=candidates, candidates_unavailable=unavailable,
                  alpha_source=alpha_source, alpha_nug_of_new_tau=alpha_nugget(tau2, config.alpha_nug_factor),
                  eta_in=eta.record(), eta_out=new.record(),
                  not_learned={name: value for name, value in eta_components(eta).items() if name not in learned})
    return new, record


def _positive(value):
    return np.isfinite(float(value)) and value > 0


def _z(difference, stderrs):
    """z value of a summed difference of independent estimates (NaN when a standard error is unavailable)."""
    stderrs = np.asarray(stderrs, dtype=np.float64)
    if stderrs.size == 0 or not np.all(np.isfinite(stderrs)):
        return float("nan")
    se = float(np.sqrt(np.sum(stderrs**2)))
    return float(difference)/se if se > 0 else float("nan")


def gm_precision(theta, g, g_stderr, t, t_stderr, q, counts):
    """C15-9 for nu (or alpha): Gull-MacKay value sum(g - theta t)/sum q; ratio form sum g / sum (q + t) when the
    Gull-MacKay numerator or denominator is <= 0 (gm_fallback); the C15-1 (dim) value sum counts / sum (q + t) and
    the z value of the numerator (a difference of independent estimates) recorded.  Longdouble sums.  Returns the
    candidate record; EBUpdateFailed when neither form is positive."""
    g, g_stderr, t, t_stderr, q, counts = (np.asarray(x, dtype=np.float64) for x in (g, g_stderr, t, t_stderr, q,
                                                                                    counts))
    theta = LD(theta)
    with np.errstate(**_RAISE):
        sum_g, sum_t, sum_q = np.sum(g, dtype=LD), np.sum(t, dtype=LD), np.sum(q, dtype=LD)
        numerator = sum_g-theta*sum_t
        gm_value = numerator/sum_q if _positive(sum_q) else LD("nan")
        ratio_den = sum_q+sum_t
        ratio_value = sum_g/ratio_den if _positive(ratio_den) else LD("nan")
        dim_value = np.sum(counts, dtype=LD)/ratio_den if _positive(ratio_den) else LD("nan")
    z = _z(numerator, np.concatenate([g_stderr, float(theta)*t_stderr]))
    out = dict(gull_mackay=float(gm_value), ratio=float(ratio_value), dim=float(dim_value), numerator=float(numerator),
               denominator=float(sum_q), ratio_numerator=float(sum_g), ratio_denominator=float(ratio_den),
               g_sum=float(sum_g), trace_sum=float(sum_t), quadratic_sum=float(sum_q), z_numerator=z)
    if _positive(numerator) and _positive(sum_q):
        out.update(value=float(gm_value), form=GM_FORM)
    elif _positive(sum_g) and _positive(ratio_den):
        out.update(value=float(ratio_value), form=GM_FALLBACK)
    else:
        raise EBUpdateFailed(f"C15-9 precision update: Gull-MacKay {float(numerator)}/{float(sum_q)} and ratio "
                             f"{float(sum_g)}/{float(ratio_den)} not positive")
    return out


def gm_tau2(tau2_k, m_hat, m_prior, m_prior_stderr, m_var, m_var_stderr, dim_numerators, dim_denominators):
    """C15-9 for tau_k^2: Gull-MacKay tau_k^2 sum m_hat / sum (m_prior - m_var); ratio form tau_k^2 sum (m_hat + m_var)
    / sum m_prior when the Gull-MacKay numerator or denominator is <= 0 (gm_fallback); the C15-1 (dim) value
    sum (tau_hat + tau_var) / sum tau_den and the z value of the denominator (a difference of independent
    estimates; m_var's standard error by the B6 bound) recorded.  Longdouble sums."""
    arrays = (m_hat, m_prior, m_prior_stderr, m_var, m_var_stderr, dim_numerators, dim_denominators)
    m_hat, m_prior, m_prior_stderr, m_var, m_var_stderr, dim_num, dim_den = (np.asarray(x, dtype=np.float64)
                                                                             for x in arrays)
    t2 = LD(tau2_k)
    with np.errstate(**_RAISE):
        sum_hat, sum_prior, sum_var = (np.sum(x, dtype=LD) for x in (m_hat, m_prior, m_var))
        denominator = sum_prior-sum_var
        gm_value = t2*sum_hat/denominator if _positive(denominator) else LD("nan")
        ratio_value = t2*(sum_hat+sum_var)/sum_prior if _positive(sum_prior) else LD("nan")
        n_dim, d_dim = np.sum(dim_num, dtype=LD), np.sum(dim_den, dtype=LD)
        dim_value = n_dim/d_dim if _positive(d_dim) else LD("nan")
    z = _z(denominator, np.concatenate([m_prior_stderr, m_var_stderr]))
    out = dict(gull_mackay=float(gm_value), ratio=float(ratio_value), dim=float(dim_value), numerator=float(sum_hat),
               denominator=float(denominator), m_hat_sum=float(sum_hat), m_prior_sum=float(sum_prior),
               m_var_sum=float(sum_var), dim_numerator=float(n_dim), dim_denominator=float(d_dim),
               z_denominator=z, tau2_in=float(tau2_k))
    if _positive(sum_hat) and _positive(denominator):
        value, form = float(gm_value), GM_FORM
    elif _positive(sum_hat+sum_var) and _positive(sum_prior):
        value, form = float(ratio_value), GM_FALLBACK
    else:
        raise EBUpdateFailed(f"C15-9 tau^2 update: Gull-MacKay {float(sum_hat)}/{float(denominator)} and ratio "
                             f"{float(sum_hat+sum_var)}/{float(sum_prior)} not positive")
    if not _positive(value):
        raise EBUpdateFailed(f"C15-9 tau^2 update: value {value} not positive")
    out.update(value=value, tau2=value, tau=float(np.sqrt(value)), form=form)
    return out


def alpha_nug_coupling(tau2, learned, parametrization):
    """C15-9 / B17 corollary 4.3 flag (P2): the learned tau_k with tau_k^2 >= every fixed tau^2 -- alpha_nug =
    factor/max_k tau_k^2 then moves with a learned component and the fixed point is only the alpha-held partial
    stationary point.  None under P3 (alpha learned)."""
    if parametrization != "P2":
        return dict(flag=None, components=[], rule="P3: alpha learned, no alpha_nug coupling")
    learned_tau = [k for k in range(len(tau2)) if f"tau_{k}" in learned]
    fixed = [tau2[k] for k in range(len(tau2)) if f"tau_{k}" not in learned]
    top = max(fixed) if fixed else -np.inf
    coupled = [k for k in learned_tau if tau2[k] >= top]
    return dict(flag=bool(coupled), components=coupled,
                rule="P2: a learned tau_k^2 >= every fixed tau^2 (default: tau_k >= tau_0; always with learned "
                     "tau_0); d alpha_nug/d tau_k^2 = -factor/tau_k^4 for the unique largest (B17 corollary 4.3)")


def gull_mackay_update(eta, statistics, config):
    """C15-9 (eb_count "gull_mackay"): the new eta from this round's per-pair statistics (each with .gm).
    Returns (Eta, record); raises EBUpdateFailed when a learned component has no usable pair or no positive form."""
    if not isinstance(config, EBConfig) or not config.gull_mackay:
        raise InputValidationError("config: an EBConfig with eb_count 'gull_mackay' required")
    if not isinstance(eta, Eta):
        raise InputValidationError("eta: Eta required")
    n_tau = len(eta.tau2)
    rows = [s for s in statistics if s is not None]
    for s in rows:
        if not isinstance(s, PairStatistics):
            raise InputValidationError("statistics: PairStatistics (or None) required")
        if len(s.tau_den) != n_tau:
            raise InputValidationError(f"pair {s.pair_id}: {len(s.tau_den)} tissue classes, eta has {n_tau}")
        if s.gm is not None:
            if not isinstance(s.gm, GMStatistics) or len(s.gm.m_prior) != n_tau or len(s.gm.m_hat) != n_tau:
                raise InputValidationError(f"pair {s.pair_id}: GMStatistics with {n_tau} tissue classes required")
            if (s.gm.alpha, s.gm.nu, tuple(s.gm.tau2)) != (eta.alpha, eta.nu, tuple(eta.tau2)):
                raise InputValidationError(f"pair {s.pair_id}: C15-9 statistics computed under eta "
                                           f"({s.gm.alpha}, {s.gm.nu}, {s.gm.tau2}), not this round's eta")
    learned = config.learned(n_tau)

    def gm_ok(s):
        return s.gm is not None and s.gm.usable
    prec = [s for s in rows if s.usable_precisions and gm_ok(s)]
    taus = [s for s in rows if s.usable_tau and gm_ok(s) and s.gm.m_var is not None]
    in_prec, in_taus = {id(s) for s in prec}, {id(s) for s in taus}
    record = dict(eb_count="gull_mackay", pairs_with_statistics=[s.pair_id for s in rows],
                  pairs_used_alpha_nu=[s.pair_id for s in prec], pairs_used_tau=[s.pair_id for s in taus],
                  pairs_excluded=[dict(pair_id=s.pair_id, alpha_nu=id(s) not in in_prec, tau=id(s) not in in_taus,
                                       S_conv=s.n_converged, S_pass=s.n_passed,
                                       S0_conv=None if s.gm is None else s.gm.n_prior_converged)
                                  for s in rows if id(s) not in in_prec or id(s) not in in_taus])
    candidates, unavailable = {}, {}

    def candidate(name, compute):
        try:
            candidates[name] = compute()
        except EBUpdateFailed as exc:
            if name in learned:
                raise
            unavailable[name] = str(exc)

    def precision(name, theta):
        g = "g_alpha" if name == "alpha" else "g_nu"
        t, q, c = ("kq_trace", "kq_quadratic", "n_free") if name == "alpha" else ("curl_trace", "curl_quadratic",
                                                                                 "n_curl")
        return gm_precision(theta, [getattr(s.gm, g) for s in prec], [getattr(s.gm, g+"_stderr") for s in prec],
                            [getattr(s, t) for s in prec], [getattr(s, t+"_stderr") for s in prec],
                            [getattr(s, q) for s in prec], [getattr(s, c) for s in prec])

    def tau_k(k):
        return gm_tau2(eta.tau2[k], [s.gm.m_hat[k] for s in taus], [s.gm.m_prior[k] for s in taus],
                       [s.gm.m_prior_stderr[k] for s in taus], [s.gm.m_var[k] for s in taus],
                       [s.gm.m_var_stderr_bound[k] for s in taus], [s.tau_hat[k]+s.tau_var[k] for s in taus],
                       [s.tau_den[k] for s in taus])
    if prec:
        candidate("alpha", lambda: precision("alpha", eta.alpha))
        candidate("nu", lambda: precision("nu", eta.nu))
    if taus:
        for k in range(n_tau):
            candidate(f"tau_{k}", lambda k=k: tau_k(k))
    if not prec:
        raise EBUpdateFailed("no usable pair for nu" + (" and alpha" if "alpha" in learned else "") +
                             " (a pair needs S_conv >= 1 and S0_conv >= 1)")
    if not taus:
        raise EBUpdateFailed("no usable pair for tau (a pair needs S_pass >= 2 and S0_conv >= 1)")
    tau2 = list(eta.tau2)
    for k in range(n_tau):
        if f"tau_{k}" in learned:
            tau2[k] = candidates[f"tau_{k}"]["tau2"]
    nu = candidates["nu"]["value"]
    if config.parametrization == "P2":
        alpha = alpha_nugget(tau2, config.alpha_nug_factor)
        alpha_source = "alpha_nug(tau_new)"
    else:
        alpha = candidates["alpha"]["value"]
        alpha_source = f"C15-9 update ({candidates['alpha']['form']})"
    new = Eta(alpha, nu, tuple(tau2), source=f"EB update ({config.parametrization}, gull_mackay)")
    forms = {name: candidates[name]["form"] for name in learned}
    record.update(learned=learned, candidates=candidates, candidates_unavailable=unavailable, forms=forms,
                  gm_fallback=[name for name, form in forms.items() if form == GM_FALLBACK],
                  alpha_source=alpha_source, alpha_nug_of_new_tau=alpha_nugget(tau2, config.alpha_nug_factor),
                  alpha_nug_coupling=alpha_nug_coupling(tau2, learned, config.parametrization),
                  count_checks={s.pair_id: s.gm.count_check for s in rows if s.gm is not None},
                  identity_max_rel={s.pair_id: s.gm.identity_max_rel for s in rows if s.gm is not None},
                  eta_in=eta.record(), eta_out=new.record(),
                  not_learned={name: value for name, value in eta_components(eta).items() if name not in learned})
    return new, record


def inverse_gamma_from_moments(mean, variance):
    """C15-5: InvGamma(a, b) with mean b/(a-1) = m and variance b^2/((a-1)^2 (a-2)) = v: a = m^2/v + 2,
    b = m (a - 1)."""
    m, v = finite_real(mean, "mean", positive=True), finite_real(variance, "variance", positive=True)
    a = m*m/v+2
    return a, m*(a-1)


def inverse_gamma_moments(a, b):
    """Mean b/(a-1) and variance b^2/((a-1)^2 (a-2)) of InvGamma(a, b), a > 2."""
    a, b = finite_real(a, "a"), finite_real(b, "b", positive=True)
    if a <= 2:
        raise InputValidationError("a > 2 required for a finite variance")
    return b/(a-1), b*b/((a-1)**2*(a-2))


def pseudo_count_form(a, b):
    """C04-4: n0 = 2a + 2, sigma0^2 = b/(a + 1) (prior mode)."""
    return 2*a+2, b/(a+1)


def inverse_gamma_from_pseudo_counts(n0, sigma0_sq):
    """C15-5 option / C04-4: a = n0/2 - 1, b = n0 sigma0^2/2."""
    n0, s = finite_real(n0, "n0", positive=True), finite_real(sigma0_sq, "sigma0_sq", positive=True)
    if n0 <= 2:
        raise InputValidationError("n0 > 2 required (a = n0/2 - 1 > 0)")
    return n0/2-1, n0*s/2


def sigma_prior(sigma_hat, n_q_values, config=None):
    """C15-5: the inverse-gamma prior (a_c, b_c) per channel from sigma_hat (n_T, C) = the last round's
    sigma_post of the usable pairs; n_q_values (n_T,) their N_q.  Returns the record (a, b, n0, sigma0^2, share
    n0/N_q per channel)."""
    config = EBConfig() if config is None else config
    sigma_hat = np.asarray(sigma_hat, dtype=np.float64)
    n_q_values = np.asarray(n_q_values, dtype=np.float64)
    if sigma_hat.ndim != 2 or sigma_hat.shape[0] != n_q_values.size or (sigma_hat.size and np.any(sigma_hat <= 0)):
        raise InputValidationError("sigma_hat: (n_T, C) positive values with one N_q per pair required")
    n_t = sigma_hat.shape[0]
    n_q_median = float(np.median(n_q_values)) if n_t else None
    channels = []
    for c in range(sigma_hat.shape[1] if sigma_hat.ndim == 2 else 0):
        values = sigma_hat[:, c]
        m = float(np.mean(values)) if n_t else None
        v = float(np.var(values, ddof=config.variance_ddof)) if n_t > config.variance_ddof else None
        row = dict(channel=c, n_T=n_t, mean=m, variance=v, median=float(np.median(values)) if n_t else None)
        a = b = 0.
        if config.sigma_prior == "none":
            row["rule"] = "none (a = b = 0)"
        elif config.sigma_prior == "moment_match":
            if n_t < 3 or not v:
                row["rule"] = f"moment_match not applicable (n_T = {n_t}, v = {v}): no prior, a = b = 0 (C15-5)"
            else:
                a, b = inverse_gamma_from_moments(m, v)
                row["rule"] = "moment_match: a = m^2/v + 2, b = m (a - 1)"
        else:
            if n_t < 1:
                row["rule"] = "pseudo_count not applicable (n_T = 0): a = b = 0"
            else:
                s0 = m if config.sigma0_statistic == "mean" else row["median"]
                a, b = inverse_gamma_from_pseudo_counts(config.kappa0*n_q_median, s0)
                row["rule"] = f"pseudo_count: n0 = kappa0 median N_q (kappa0 = {config.kappa0}), sigma0^2 = " \
                              f"{config.sigma0_statistic} (D29, model change)"
        n0, s02 = pseudo_count_form(a, b) if a > 0 else (0., None)
        row.update(a=a, b=b, n0=n0, sigma0_sq=s02, share_n0_over_Nq=n0/n_q_median if n_q_median else 0.)
        channels.append(row)
    return dict(contract="C15-5 / C04-4 / D29", source="sigma_post of the last EB round (C11-6), usable pairs",
                rule=config.sigma_prior, variance_ddof=config.variance_ddof, n_T=n_t, n_q_median=n_q_median,
                channels=channels,
                note="moment matching gives n0 = 2a + 2 of order tens against N_q ~ 1e6: the prior has practically "
                     "no effect on the M step (C11-5, C15-5); a real effect needs the pseudo-count option (D29).")


def _support_prior_row(parameter, values, counts, config):
    """One variance parameter of C15-5 under sigma_domain support / support_shared: the rule of `sigma_prior` on the
    values sigma_hat_p^2 of the pairs that have the parameter, with the count N (N_q, N_c or N_S) of those pairs in
    place of N_q (D29: n0 = kappa0 median N; share n0 / median N)."""
    values = np.asarray(values, dtype=np.float64)
    n_t = int(values.size)
    n_median = float(np.median(np.asarray(counts, dtype=np.float64))) if n_t else None
    m = float(np.mean(values)) if n_t else None
    v = float(np.var(values, ddof=config.variance_ddof)) if n_t > config.variance_ddof else None
    row = dict(parameter=parameter, n_T=n_t, mean=m, variance=v, median=float(np.median(values)) if n_t else None,
               count_median=n_median)
    a = b = 0.
    if config.sigma_prior == "none":
        row["rule"] = "none (a = b = 0)"
    elif config.sigma_prior == "moment_match":
        if n_t < 3 or not v:
            row["rule"] = f"moment_match not applicable (n_T = {n_t}, v = {v}): no prior, a = b = 0 (C15-5)"
        else:
            a, b = inverse_gamma_from_moments(m, v)
            row["rule"] = "moment_match: a = m^2/v + 2, b = m (a - 1)"
    else:
        if n_t < 1:
            row["rule"] = "pseudo_count not applicable (n_T = 0): a = b = 0"
        else:
            s0 = m if config.sigma0_statistic == "mean" else row["median"]
            a, b = inverse_gamma_from_pseudo_counts(config.kappa0*n_median, s0)
            row["rule"] = f"pseudo_count: n0 = kappa0 median N (kappa0 = {config.kappa0}; N = N_q / N_c / N_S of the " \
                          f"parameter), sigma0^2 = {config.sigma0_statistic} (D29, model change)"
    n0, s02 = pseudo_count_form(a, b) if a > 0 else (0., None)
    row.update(a=a, b=b, n0=n0, sigma0_sq=s02, share_n0_over_N=n0/n_median if n_median else 0.)
    return row


def sigma_prior_support(statistics, config=None):
    """C15-5 under sigma_domain support / support_shared (C04-6, C15-5 appended note; SWLI): the prior per variance
    parameter on the same estimation domain, from the last round's sigma_post (with null for an empty native support):
      I          every pair, count N_q (as `sigma_prior` channel 0);
      support    every structure channel c on the pairs whose native support of c is non-empty (sigma_post_c not null;
                 the others are left out of that channel's matching), count N_c;
      support_shared  one parameter S: sigma_hat_S^2 of every pair with a non-empty structure layer, count N_S
                 (one (a_S, b_S)).
    All pairs must share sigma_domain and theta.  The record carries sigma_domain and support_threshold (a frozen prior
    is admitted only by a run with the same domain and theta, `check_sigma_prior_domain`)."""
    config = EBConfig() if config is None else config
    stats = list(statistics)
    domains = {(s.sigma_domain, None if s.support is None else s.support["support_threshold"]) for s in stats}
    if len(domains) != 1 or next(iter(domains))[0] == BRAIN:
        raise InputValidationError(f"sigma_prior_support: the pairs of one non-brain sigma_domain and theta required "
                                   f"(got {sorted(domains, key=str)})")
    domain, theta = next(iter(domains))
    k_s = len(stats[0].sigma_post)-1
    if any(len(s.sigma_post) != k_s+1 for s in stats):
        raise InputValidationError("sigma_prior_support: every pair needs 1 + K_S sigma_post values")
    rows = [_support_prior_row("I", [s.sigma_post[0] for s in stats], [s.n_q for s in stats], config)]
    excluded = {}
    if domain == SUPPORT:
        for c in range(1, k_s+1):
            used = [s for s in stats if s.sigma_post[c] is not None]
            excluded[str(c)] = [s.pair_id for s in stats if s.sigma_post[c] is None]
            rows.append(dict(_support_prior_row(c, [s.sigma_post[c] for s in used],
                                                [s.support["N_c"][c-1] for s in used], config),
                             pairs=[s.pair_id for s in used]))
    else:
        used = [s for s in stats if any(x is not None for x in s.sigma_post[1:])]
        excluded["S"] = [s.pair_id for s in stats if s not in used]
        values = [next(x for x in s.sigma_post[1:] if x is not None) for s in used]
        rows.append(dict(_support_prior_row("S", values, [s.support["N_S"] for s in used], config),
                         pairs=[s.pair_id for s in used]))
    return dict(contract="C15-5 / C04-4 / D29 (sigma_domain appended note, C04-6)", sigma_domain=domain,
                support_threshold=theta, source="sigma_post of the last EB round (C11-6), usable pairs, on the "
                "run's estimation domain", rule=config.sigma_prior, variance_ddof=config.variance_ddof,
                n_T=len(stats), parameters=rows, excluded_empty_support=excluded,
                note="support: one (a_c, b_c) per structure channel (pairs with an empty native support of c left "
                     "out); support_shared: one (a_S, b_S); counts N_c / N_S in place of N_q (D29 option: "
                     "n0 = kappa0 median N)")


def check_sigma_prior_domain(prior, sigma_domain, support_threshold):
    """C15-5 appended note: a frozen sigma prior is admitted only by a run with the same sigma_domain and theta; a prior
    record without the keys is read as brain with theta null (brain bitwise unchanged).  InputValidationError
    otherwise.  (run_pair does not read sigma priors yet: this is the check its loader must call.)"""
    domain = prior.get("sigma_domain", BRAIN) if isinstance(prior, Mapping) else None
    theta = prior.get("support_threshold") if isinstance(prior, Mapping) else None
    if domain is None:
        raise InputValidationError("sigma prior: a record (mapping) required")
    if domain != sigma_domain or (theta is None) != (support_threshold is None) or (
            theta is not None and float(theta) != float(support_threshold)):
        raise InputValidationError(f"sigma prior: estimation domain {domain!r} with theta {theta!r} differs from the "
                                   f"run's {sigma_domain!r} with theta {support_threshold!r} (C15-5: refused)")
    return domain, theta


# ----------------------------------------------------------------------------------------------
# Leak guard (C15-1, D23, D24)
# ----------------------------------------------------------------------------------------------

class EBPairRefused(Exception):
    """The pair or its segmentation may not be used for EB; status `eb_pair_refused`."""
    status = "eb_pair_refused"


@dataclass(frozen=True)
class EBPair:
    """An admitted EB training pair (only `admit_eb_pairs` builds these)."""
    fixed: str
    moving: str
    fold: str
    admission: Mapping = field(default_factory=dict, compare=False, repr=False)

    @property
    def pair_id(self):
        return f"{self.fixed}__{self.moving}"

    def record(self):
        return dict(pair_id=self.pair_id, fixed=self.fixed, moving=self.moving, fold=self.fold, **dict(self.admission))


def _is_oasis(subject):
    return isinstance(subject, str) and CANONICAL_SUBJECT_PATTERNS["OASIS"].fullmatch(subject) is not None


def read_eb_pairs(path):
    """splits/bayesreg_v1/eb_pairs.tsv: header fixed<TAB>moving<TAB>fold; canonical OASIS ids, fold A/B, no
    repeated subject.  Returns (tuple of (fixed, moving, fold), sha256 of the file)."""
    raw = Path(path).read_bytes()
    lines = raw.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines or tuple(lines[0].split("\t")) != EB_PAIRS_HEADER:
        raise EBPairRefused(f"{path}: header must be {chr(9).join(EB_PAIRS_HEADER)!r}")
    rows, seen = [], set()
    for number, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != 3 or not _is_oasis(parts[0]) or not _is_oasis(parts[1]) or parts[2] not in FOLD_NAMES:
            raise EBPairRefused(f"{path}:{number}: expected <OASIS id>\\t<OASIS id>\\t<A|B>")
        if parts[0] == parts[1] or parts[0] in seen or parts[1] in seen:
            raise EBPairRefused(f"{path}:{number}: a subject is repeated (EB subjects are distinct)")
        seen.update(parts[:2])
        rows.append(tuple(parts))
    if not rows:
        raise EBPairRefused(f"{path}: no pairs")
    return tuple(rows), hashlib.sha256(raw).hexdigest()


def _parse_requested(requested):
    out = []
    for item in requested:
        if isinstance(item, str):
            parts = item.split("__")
            if len(parts) != 2:
                raise EBPairRefused(f"pair {item!r}: expected <fixed>__<moving>")
            out.append(tuple(parts))
        elif isinstance(item, (tuple, list)) and len(item) == 2 and all(isinstance(s, str) for s in item):
            out.append(tuple(item))
        else:
            raise EBPairRefused(f"pair {item!r}: a '<fixed>__<moving>' string or a (fixed, moving) pair required")
    if len(set(out)) != len(out):
        raise EBPairRefused("requested pairs: duplicates")
    return out


def admit_eb_pairs(requested=None, *, eb_pairs_path, split, allowlist, training_source=TRAINING_SOURCE):
    """C15-1 / D23 / D24 leak guard for the pair list (before any image is read).

    requested: None (every listed pair) or a list of '<fixed>__<moving>' / (fixed, moving).  Each must be a row
    of eb_pairs.tsv in that order; both subjects must be in the folds file (`split`, a FoldSplit) with the
    row's fold, in the allowlist source `training_source` (not never_for_training) and admitted by the
    allowlist.  Returns (tuple of EBPair, record); EBPairRefused otherwise."""
    if not isinstance(split, FoldSplit):
        raise InputValidationError("split: a FoldSplit from bayesreg.seg.unet.data.load_folds is required")
    if not isinstance(allowlist, DevAllowlist):
        raise InputValidationError("allowlist: a DevAllowlist from load_dev_allowlist is required")
    listed, listed_sha = read_eb_pairs(eb_pairs_path)
    by_pair = {(f, m): fold for f, m, fold in listed}
    sources = {source["name"]: source for source in allowlist.sources}
    source = sources.get(training_source)
    if source is None:
        raise EBPairRefused(f"the allowlist has no source {training_source!r}")
    if source["never_for_training"] or source["dataset"] != "OASIS":
        raise EBPairRefused(f"allowlist source {training_source!r} must be an OASIS training source")
    training = frozenset(source["subjects"])
    wanted = [(f, m) for f, m, _ in listed] if requested is None else _parse_requested(requested)
    if not wanted:
        raise EBPairRefused("no pair requested")
    admitted = []
    for fixed, moving in wanted:
        fold = by_pair.get((fixed, moving))
        if fold is None:
            raise EBPairRefused(f"{fixed}__{moving} is not a row of {eb_pairs_path} (fixed, moving in that order)")
        for subject in (fixed, moving):
            if subject not in split.assignment:
                raise EBPairRefused(f"{subject} is not a B-line training subject (folds file)")
            if subject not in training:
                raise EBPairRefused(f"{subject} is not in the allowlist source {training_source!r}")
            if not allowlist.allows(subject):
                raise EBPairRefused(f"{subject} is not on the development allowlist")
            if subject in allowlist.never_for_training:
                raise EBPairRefused(f"{subject} is never_for_training (T0)")
        folds = (split.fold_of(fixed), split.fold_of(moving))
        if folds[0] != folds[1]:
            raise EBPairRefused(f"{fixed}__{moving}: cross-fold pair ({folds[0]} / {folds[1]})")
        if folds[0] != fold:
            raise EBPairRefused(f"{fixed}__{moving}: eb_pairs.tsv says fold {fold}, the folds file {folds[0]}")
        admission = dict(check="C15-1/D23/D24 admit_eb_pairs", eb_pairs_sha256=listed_sha,
                         folds_sha256=split.sha256, allowlist_sha256=allowlist.sha256,
                         training_source=training_source, training_source_sha256=source["sha256"])
        admitted.append(EBPair(fixed, moving, fold, admission))
    record = dict(check="C15-1/D23/D24", eb_pairs_path=str(eb_pairs_path), eb_pairs_sha256=listed_sha,
                  n_listed=len(listed), folds_path=split.path, folds_sha256=split.sha256,
                  allowlist_path=allowlist.path, allowlist_sha256=allowlist.sha256, training_source=training_source,
                  training_source_sha256=source["sha256"], admitted=[p.pair_id for p in admitted])
    return tuple(admitted), record


def check_structure_models(accepted, pair, registry):
    """D24 / C15-1: the structure layer P_S of one image of an EB pair must come from fold models of the OTHER
    fold only.  accepted: the AcceptedProvenance of validate_segmentation_provenance(layer="structure") for
    this image (its `folds`: {model id: fold index}); pair: the EBPair; registry: the SegmenterRegistry.
    Every model must be registered with that fold index, differ from the pair's fold, and have had no label
    contact with either subject.  Returns the check record; EBPairRefused otherwise."""
    if not isinstance(pair, EBPair):
        raise InputValidationError("pair: an EBPair from admit_eb_pairs is required")
    if not isinstance(registry, SegmenterRegistry):
        raise InputValidationError("registry: a SegmenterRegistry is required")
    if not isinstance(accepted, Mapping) or accepted.get("layer") != "structure":
        raise EBPairRefused("structure provenance: an accepted structure-layer record is required")
    folds = accepted.get("folds")
    model_ids = accepted.get("model_ids")
    if not isinstance(folds, Mapping) or not folds or not isinstance(model_ids, list) or set(folds) != set(model_ids):
        raise EBPairRefused("structure provenance: folds {model id: fold index} for every model id required")
    if list(accepted.get("pair_subjects_checked", [])) != [pair.fixed, pair.moving]:
        raise EBPairRefused("structure provenance: checked against a different pair")
    own = FOLD_INDEX[pair.fold]
    for model_id in model_ids:
        entry = registry.models.get(model_id)
        if entry is None:
            raise EBPairRefused(f"model {model_id!r} is not registered")
        if entry["fold"] != folds[model_id]:
            raise EBPairRefused(f"model {model_id!r}: provenance fold {folds[model_id]} != registry fold {entry['fold']}")
        if entry["fold"] == own:
            raise EBPairRefused(f"model {model_id!r} is of the pair's own fold {pair.fold} (index {own}); EB pairs "
                                "must be segmented by the other fold's models (D24)")
        contact = set(entry["label_contact_subjects"])
        if pair.fixed in contact or pair.moving in contact:
            raise EBPairRefused(f"model {model_id!r} had label contact with a subject of {pair.pair_id}")
    return dict(check="D24 other-fold structure models", pair_id=pair.pair_id, pair_fold=pair.fold,
                pair_fold_index=own, models={m: folds[m] for m in model_ids}, registry_sha256=registry.sha256)


# ----------------------------------------------------------------------------------------------
# MAP of one pair and the EB driver
# ----------------------------------------------------------------------------------------------

@dataclass
class MapOutcome:
    """The native output of one pair's MAP under a fixed eta: u_hat (last formal commit, (nz, ny, nx, 3)) and
    the host ModelState with sigma_used when the run ended with a C13-3 ending on the native level."""
    status: str
    reason: str
    native: bool
    u_hat: np.ndarray | None
    state: ModelState | None
    sigma_post: list | None
    record: dict

    @property
    def usable(self):
        return (self.native and self.status in LAPLACE_ENDINGS and self.u_hat is not None
                and isinstance(self.state, ModelState))


def multilevel_map(levels, eta, *, solver_config=None, backend=None, sigma_policy=None, log=None):
    """Step 1 of a round for one pair: `solve_multilevel` (C09-C13) under eta with the Gaussian EM of C11
    (a = b = 0, C15-5).  Returns a MapOutcome (rounds without the energy traces; those go to `log`)."""
    backend = NumpyBackend() if backend is None else backend
    sigma_policy = GaussianMStep() if sigma_policy is None else sigma_policy
    solver_config = SolverConfig() if solver_config is None else solver_config
    result = solve_multilevel(backend, levels, eta=eta, sigma=sigma_policy, config=solver_config, log=log)
    rounds = [{k: v for k, v in row.items() if k not in ("F_B_trace", "GF_B_trace")} for row in result.rounds]
    commit = result.commit
    record = dict(status=result.status, wording=result.wording, reason=result.reason, native=result.native,
                  level=result.level, rounds=rounds, transfers=result.transfers, timing=result.timing,
                  counters=result.counters, sigma_used=None if result.state is None else
                  (result.state.sigma2_record() if result.state.has_support
                   else np.asarray(result.state.sigma2).tolist()), sigma_post=result.sigma_post,
                  state_hash=None if commit is None else commit.cert.state_hash,
                  min_jd_lower_bound=None if commit is None else commit.cert.min_jd_lower_bound,
                  solver=solver_config.record(), sigma_policy=sigma_policy.record(), eta=eta.record())
    if solver_config.pg.jd_cell_estimate == CERTIFICATE_BERNSTEIN:     # C09.1-3a check (b): the MAP's total (JDTI)
        record["bernstein_j53_violations_total"] = j53_violations_total(result.rounds)
    if result.support is not None:                 # sigma_domain != brain (C04-6 "记录"): the policy run and the levels
        record.update(sigma_policy=SupportMStep.from_policy(sigma_policy, solver_config.sigma_domain).record(),
                      support_levels=result.support)
    usable = result.native and commit is not None
    return MapOutcome(result.status, result.reason, result.native, commit.u if usable else None,
                      result.state if usable else None, result.sigma_post, record)


@dataclass
class EBResult:
    status: str
    wording: str
    eta: Eta
    trajectory: list             # eta records, eta0 first
    rounds: list                 # per-round records
    sigma_prior: dict | None
    record: dict


def round_seed(run_id, round_index, pair_id):
    """C14-4: the sampling seed of one pair in one EB round, derived from the run id."""
    return seed_from_run_id(f"{run_id}/eb_round{round_index}/{pair_id}")


def prior_round_seed(run_id, round_index, pair_id):
    """C15-9: the prior-sample seed of one pair in one EB round, seed_from_run_id(f"{run_id}/eb_round{r}/{pair}/prior")
    (a stream separate from round_seed's posterior samples)."""
    return seed_from_run_id(f"{run_id}/eb_round{round_index}/{pair_id}/prior")


def outcome_statistics(outcome, *, pair_id, config, seed, backend, prior_seed=None):
    """The default statistics step of `run_population_eb`: `pair_statistics` at the MapOutcome's native output
    (prior_seed: eb_count "gull_mackay" only, C15-9)."""
    return pair_statistics(outcome.state, outcome.u_hat, pair_id=pair_id, config=config, seed=seed, backend=backend,
                           sigma_post=outcome.sigma_post, prior_seed=prior_seed)


def _run_sigma_domain(stats, sigma_domain, support_threshold):
    """(sigma_domain, theta) of a non-brain EB run -- given by the caller (learn CLI) or carried by the pairs' statistics
    (PairStatistics.support); None for brain.  A given domain that the statistics contradict is refused."""
    carried = {(s.support["sigma_domain"], s.support["support_threshold"]) for s in stats if s.support is not None}
    given = None if sigma_domain in (None, BRAIN) else (sigma_domain, support_threshold)
    if given is not None and carried and carried != {given}:
        raise InputValidationError(f"EB: the run's sigma_domain {given} differs from the pairs' {sorted(carried)}")
    if given is not None:
        return given
    if len(carried) > 1:
        raise InputValidationError(f"EB: the pairs carry several sigma_domains {sorted(carried)}")
    return next(iter(carried)) if carried else None


def run_population_eb(pairs, solve_pair, *, eta0, config=None, run_id, out_dir=None, backend=None, log=None,
                      statistics=outcome_statistics, logvol_points=None, sigma_domain=None, support_threshold=None):
    """C15-6: at most config.max_rounds rounds of [MAP of every pair under eta (solve_pair) -> C14 samples
    (S = config.n_samples) -> pair statistics -> eb_update]; stop when the relative change of every learned
    component is < config.rel_tol.  Then the sigma prior (C15-5) from the last round.

    pairs: EBPair from admit_eb_pairs; solve_pair(pair, eta, *, round_index) -> MapOutcome; eta0: Eta (under P2
    its alpha must equal alpha_nug(tau0), C05-6); run_id: names the run and derives every sample seed (C14-4);
    out_dir: if given, rounds/round_<r>.json, pairs/round_<r>/<pair>.json, eb_result.json (never overwritten);
    statistics(outcome, *, pair_id, config, seed, backend) -> (PairStatistics, record): the C14 + C15-1 step
    (default `outcome_statistics`; tests inject cheap ones to exercise the round logic).
    eb_count "gull_mackay" (C15-9): statistics is also given prior_seed=prior_round_seed(run_id, r, pair_id) and
    must return PairStatistics with .gm; the update is `gull_mackay_update`; each round records step_ratio (its
    largest relative change over the previous round's) and alpha_nug_coupled; the result's wording is EB_WORDING_GM
    ("步长判据满足": a step criterion, not convergence) and its record adds eb_count, step_ratios,
    alpha_nug_coupled_rounds and gm_fallback_rounds.  Under "dim" nothing of this is added (C15-9: bitwise).
    logvol_points: the point set of the pairs' MAP solves (learn CLI --logvol-points), passed to config.record for the
    result's config.laplace.noise_order only (None: the record as before FUP1; review MIN1 question 5).  It is not
    checked against the pairs' ModelStates: the caller keeps it equal to the point set solve_pair uses (the learn CLI
    passes the same args.logvol_points to both; FUP1 review minor 3).
    sigma_domain, support_threshold (SWLI; learn CLI --sigma-domain): the estimation domain of the pairs' solves, used
    only for the sigma-prior record when no pair has a usable sigma_post (review SWLI minor 7); None or "brain": the
    record as before."""
    config = EBConfig() if config is None else config
    pairs = tuple(pairs)
    if not pairs or not all(isinstance(p, EBPair) and p.admission for p in pairs):
        raise InputValidationError("pairs: a nonempty list of EBPair from admit_eb_pairs is required")
    if len({p.pair_id for p in pairs}) != len(pairs):
        raise InputValidationError("pairs: duplicate pair ids")
    if not isinstance(eta0, Eta):
        raise InputValidationError("eta0: Eta required")
    if config.parametrization == "P2":
        nug = alpha_nugget(eta0.tau2, config.alpha_nug_factor)
        if abs(eta0.alpha-nug) > 1e-12*nug:
            raise InputValidationError(f"P2: eta0.alpha = {eta0.alpha} must be alpha_nug(tau0) = {nug} (C05-6)")
    if not isinstance(run_id, str) or not run_id:
        raise InputValidationError("run_id: nonempty string required")
    backend = NumpyBackend() if backend is None else backend
    log = (lambda row: None) if log is None else log
    out = None if out_dir is None else Path(out_dir)
    if out is not None:
        (out/"rounds").mkdir(parents=True, exist_ok=True)
    tick_run = time.perf_counter()
    eta = eta0
    trajectory = [eta0.record()]
    rounds = []
    learned = config.learned(len(eta0.tau2))
    status, last_stats = EB_ROUND_CAP, []
    for r in range(config.max_rounds):
        tick_round = time.perf_counter()
        pair_rows, stats_list = [], []
        if out is not None:
            (out/"pairs"/f"round_{r}").mkdir(parents=True, exist_ok=True)
        for pair in pairs:
            tick = time.perf_counter()
            outcome = solve_pair(pair, eta, round_index=r)
            if not isinstance(outcome, MapOutcome):
                raise InputValidationError("solve_pair must return a MapOutcome")
            row = dict(pair_id=pair.pair_id, fold=pair.fold, round=r, map=outcome.record,
                       map_seconds=time.perf_counter()-tick)
            stats = None
            if not outcome.usable:
                row["excluded"] = f"MAP ended {outcome.status} (native={outcome.native}): no Laplace (C13-7)"
            else:
                seed = round_seed(run_id, r, pair.pair_id)
                if config.gull_mackay:
                    stats, posterior = statistics(outcome, pair_id=pair.pair_id, config=config, seed=seed,
                                                  backend=backend,
                                                  prior_seed=prior_round_seed(run_id, r, pair.pair_id))
                else:
                    stats, posterior = statistics(outcome, pair_id=pair.pair_id, config=config, seed=seed,
                                                  backend=backend)
                if not isinstance(stats, PairStatistics):
                    raise InputValidationError("statistics must return (PairStatistics, record)")
                row["posterior"] = posterior
                row["excluded"] = None
            stats_list.append(stats)
            row["seconds"] = time.perf_counter()-tick
            log(dict(kind="eb_pair", round=r, pair_id=pair.pair_id, status=outcome.status,
                     statistics=None if stats is None else stats.record(), seconds=row["seconds"]))
            if out is not None:
                write_json(out/"pairs"/f"round_{r}"/f"{pair.pair_id}.json", row)
            pair_rows.append(dict(pair_id=pair.pair_id, map_status=outcome.status, excluded=row["excluded"],
                                  statistics=None if stats is None else stats.record(),
                                  pass_rate=None if stats is None else stats.pass_rate))
        round_record = dict(round=r, eta_in=eta.record(), pairs=pair_rows)
        try:
            new, update = eb_update(eta, stats_list, config)
        except EBUpdateFailed as exc:
            round_record.update(status=EB_FAILED, reason=str(exc), seconds=time.perf_counter()-tick_round)
            rounds.append(round_record)
            log(dict(kind="eb_round", **{k: v for k, v in round_record.items() if k != "pairs"}))
            if out is not None:
                write_json(out/"rounds"/f"round_{r}.json", round_record)
            status = EB_FAILED
            last_stats = [s for s in stats_list if s is not None]
            break
        changes, largest = relative_changes(eta, new, learned)
        met = largest < config.rel_tol
        round_record.update(update=update, eta_out=new.record(), relative_changes=changes, max_relative_change=largest,
                            criterion_met=met, seconds=time.perf_counter()-tick_round)
        if config.gull_mackay:
            previous = rounds[-1].get("max_relative_change") if rounds else None
            round_record.update(step_ratio=largest/previous if previous else None,
                                alpha_nug_coupled=update["alpha_nug_coupling"]["flag"],
                                forms=update["forms"])
        rounds.append(round_record)
        trajectory.append(new.record())
        log(dict(kind="eb_round", round=r, eta_in=eta.record(), eta_out=new.record(), relative_changes=changes,
                 criterion_met=met))
        if out is not None:
            write_json(out/"rounds"/f"round_{r}.json", round_record)
        eta = new
        last_stats = [s for s in stats_list if s is not None]
        if met:
            status = EB_CRITERION_MET
            break
    usable_sigma = [s for s in last_stats if s.sigma_post is not None]
    domain = _run_sigma_domain(last_stats, sigma_domain, support_threshold)
    if any(s.support is not None for s in usable_sigma):            # C15-5 appended note (SWLI): the same domain
        prior = sigma_prior_support(usable_sigma, config)
    else:
        prior = sigma_prior([s.sigma_post for s in usable_sigma], [s.n_q for s in usable_sigma], config) \
            if usable_sigma else sigma_prior(np.zeros((0, 1)), np.zeros(0), config)
        if domain is not None:
            # review SWLI minor 7: a non-brain run without any usable sigma_post falls back to the empty brain-form
            # record (no prior, a = b = 0); it names the run's estimation domain so that it is never read as brain
            prior.update(sigma_domain=domain[0], support_threshold=domain[1], sigma_domain_fallback=(
                "no pair with a usable sigma_post on the run's estimation domain: the brain-form record without a "
                "prior (a = b = 0) was written; sigma_domain / support_threshold are the run's (C15-5)"))
    prior["pairs"] = [s.pair_id for s in usable_sigma]
    prior["sigma_used_for_comparison"] = {s.pair_id: list(s.sigma_used) for s in usable_sigma}
    wording = EB_WORDING_GM[status] if config.gull_mackay else EB_WORDING[status]
    record = dict(contract="C15", status=status, wording=wording, run_id=run_id,
                  config=config.record(logvol_points=logvol_points),
                  pairs=[p.record() for p in pairs], learned=learned, eta0=eta0.record(), eta_final=eta.record(),
                  eta_final_components=eta_components(eta), trajectory=trajectory,
                  max_relative_change=[r.get("max_relative_change") for r in rounds],
                  rounds_run=len(rounds), sigma_prior=prior, seconds=time.perf_counter()-tick_run,
                  map_endings_used=list(LAPLACE_ENDINGS), map_wording={k: RUN_WORDING[k] for k in LAPLACE_ENDINGS},
                  frozen="not frozen here: configs/frozen/bayesreg_v1 and decision_log are the lead's step (C15-6)")
    if config.gull_mackay:
        record.update(contract="C15 / C15-9", eb_count="gull_mackay",
                      step_ratios=[r.get("step_ratio") for r in rounds],
                      step_criterion=dict(wording="步长判据满足 when met (a step criterion, never 'convergence')",
                                          note="contraction rate rho: a step of rel_tol leaves about "
                                               "rel_tol rho/(1 - rho) (rho ~ .85: ~28 %); step_ratios estimate rho"),
                      alpha_nug_coupled_rounds=[r["round"] for r in rounds if r.get("alpha_nug_coupled")],
                      gm_fallback_rounds={r["round"]: r["update"]["gm_fallback"] for r in rounds
                                          if r.get("update") and r["update"].get("gm_fallback")},
                      note_b17=B17_NOTE)
    if out is not None:
        write_json(out/"eb_result.json", record)
    return EBResult(status, wording, eta, trajectory, rounds, prior, record)


__all__ = ["B12_NOTE", "B17_NOTE", "DEFAULT_EB_COUNT", "EBConfig", "EBPair", "EBPairRefused", "EBResult",
           "EBUpdateFailed", "EB_COUNTS", "EB_CRITERION_MET", "EB_FAILED", "EB_ROUND_CAP", "EB_STATUSES", "EB_WORDING",
           "EB_WORDING_GM", "GMStatistics", "GM_FALLBACK", "GM_FORM", "GaussStatistics", "LAPLACE_ENDINGS", "MAX_ROUNDS",
           "MapOutcome", "PARAMETRIZATIONS", "PairStatistics", "REL_TOL", "TRAINING_SOURCE", "admit_eb_pairs", "alpha_nug_coupling",
           "alpha_nugget", "check_structure_models", "eb_update", "eta_components", "gauss_point_statistics",
           "gm_precision", "gm_tau2", "gull_mackay_update", "inverse_gamma_from_moments",
           "inverse_gamma_from_pseudo_counts", "inverse_gamma_moments", "multilevel_map", "outcome_statistics",
           "pair_statistics", "precision_update", "prior_round_seed", "prior_statistics", "pseudo_count_form",
           "read_eb_pairs", "relative_changes", "rho_weights", "round_seed", "run_population_eb", "sigma_prior",
           "check_sigma_prior_domain", "sigma_prior_support",
           "tau2_update"]
