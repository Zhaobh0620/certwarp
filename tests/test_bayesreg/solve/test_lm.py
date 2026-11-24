"""LM-PCG inner solver (work package LMI, 2026-11-01; contract C09.3, C09.4, C13-5a LM item, C13-7; the C09 gates).
Synthetic data and test doubles only (no data/), numpy backend, CPU.

Gates (task book LMI item 2):
  1. default unchanged: SolverConfig() / run_pair without --inner have no LM key, inner pg is the PG path (the existing
     goldens run in the full suite unchanged; here: --inner pg explicit has the golden def_explicit digest);
  2. quadratic toy (certificate always passes): one LM step has ratio in [1 - 1e-6, 1 + 1e-6] (option (a)); option (b)
     with the proximal blocks in H_E and pred (mu = 0, so E_v is exactly quadratic) likewise, and without them in pred
     the ratio leaves that interval; option (b)'s H_E HVP equals the explicitly assembled H + sum of the proximal K^T K to
     1e-12;
  3. damping update table and gamma reset (log: lm_start rows and the first attempt of every inner solve at gamma0, per
     (level, round) and per reference refresh);
  4. the true-residual check (one HVP after the PCG; a PCG that claims convergence with a wrong p is caught);
  5. rejections never move the base (theta, gradient, certificate, H_E's base point);
  6. acceptance order: certificate -> energy -> comparison -> ratio (no energy without a pass; no certificate for a
     failed direction);
  7. a certificate rejection increases gamma (x4);
  8. 12 attempts used up -> trial_budget_exhausted; C13-7 continues on safe reasons, terminates on TRUE_RESIDUAL_FAILED
     under the default table;
  9. stagnation and the endings of C09.2-8/9;
 10. SCH: per-level caps (cap_reached at a_ell) and the budget / shares checked at the LM iteration top only;
 11. descriptive: LM and PG iteration counts to T_B on a synthetic problem (no hard threshold), and the PCG iteration
     counts of the LM system at u = 0 on the four levels of the 26^3 synthetic case (C09 gate "PCG 迭代数随分辨率", layer
     1; recorded, D35 is a record only);
 12. run_pair: --inner lm end to end (records), and the refusals in parse_args.
Added after the review LMI_复核_20261101 and contract C09.3-13a (2026-11-01):
 13. (d) an indefinite or NaN H_E ends the solve nonfinite_evaluation after one attempt NONFINITE_LINEAR_SOLVE with
     detail indefinite / nonfinite, base untouched, C13-7 terminates (run_lm and solve_multilevel -> level_failed);
 14. (e) an attempt rejected at gamma = gamma_max ends the iteration trial_budget_exhausted;
 15. (h, narrowed after the second review round) the LM continuation table: a last iteration with a TRUE_RESIDUAL_FAILED
     after a PCG stopped at maxiter ("pcg_maxiter") among safe reasons continues under the LM inner (and still terminates
     under PG); a TRUE_RESIDUAL_FAILED whose recursive residual met rtol ("pcg_rtol_met", the claimed-convergence case)
     and NONFINITE_LINEAR_SOLVE terminate; both cases are produced by run_lm itself;
 16. option (b): the gamma reset follows an inner solve that ended away from gamma0; a share ending at the inner LM
     iteration top drops the uncommitted inner steps; natural endings win over the budget at an LM iteration top.
The explicit assembly of gate 2 (option (b)) takes the columns of H from the model's own GN HVP (its correctness is the
C06 gate); only the three proximal blocks are assembled independently, which is what the C09 gate asks for.
"""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
import pytest

from jd_curl.three_d.boundary import inject_free, restrict_free
from jd_curl.validation import InputValidationError

from bayesreg.cli import run_pair
from bayesreg.geom import gauss27, gauss_geometry
from bayesreg.model import Reduction
from bayesreg.solve import (Caps, Commit, DirectObjective, Eta, FixedSigma, NumpyBackend, PGConfig, ProximalObjective,
                            ProximalParams, SolverConfig, default_lambda, solve_multilevel)
from bayesreg.solve import lm as LM
from bayesreg.solve import proximal as prox
from bayesreg.solve.lm import LMConfig, LMMetric, run_lm
from bayesreg.solve.outer import solve_direct, solve_proximal, solve_round
from bayesreg.solve.stopping import (BUDGET_EXHAUSTED, CAP_REACHED, CONVERGED_GRADIENT, LEVEL_BUDGET_REACHED,
                                     MAX_ITERATIONS, STAGNATED, TRIAL_BUDGET_EXHAUSTED, LevelShares, WallBudget,
                                     continuation, g0_of, state_hash, threshold)

from . import _run_pair_digest as RD
from . import _synthetic as syn
from .test_def_defaults import GOLDEN_8A4D060_RUN_PAIR
from .test_pg import Point, passing, small_state
from .test_run_pair import read

CFG = LMConfig()
BITWISE = pytest.mark.skipif(np.__version__ != "2.0.2", reason="bitwise golden digests recorded with numpy 2.0.2")


# ---- test doubles -----------------------------------------------------------------------------------------------------

class ScaledIdentity:
    """A metric stand-in: M_gamma^-1 r = r / (d + gamma omega_ell/omega0) (SPD; any SPD preconditioner gives the same
    LM system)."""

    def __init__(self, d, damping=1., lam_b=0.):
        self.d, self.damping, self.lam_b, self.builds, self.gammas = d, damping, lam_b, 0, []

    def at(self, gamma):
        self.builds += 1
        self.gammas.append(gamma)
        scale = 1./(self.d+gamma*self.damping)
        return SimpleNamespace(solve=lambda r: scale*r)


def spd(n, low=1., high=100., seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (q*np.geomspace(low, high, n))@q.T


class QuadraticObjective:
    """E = 1/2 (theta - theta*)^T A (theta - theta*) on a grid's free vector; reduction E_b - E_t with s_E = 0 (a test
    double of C08); hvp = A (the exact Hessian, so pred is exact)."""
    name = "quadratic"

    def __init__(self, grid, A, theta_star):
        self.grid, self.A, self.theta_star = grid, A, theta_star
        self.state = SimpleNamespace(grid=grid, omega0=1.)
        self.evaluated, self.hvp_points = 0, []

    def evaluate(self, u):
        self.evaluated += 1
        d = restrict_free(u, self.grid)-self.theta_star
        return Point(u, .5*float(d@(self.A@d)))

    def gradient(self, point):
        return self.A@(restrict_free(point.u, self.grid)-self.theta_star)

    def hvp(self, point, v):
        self.hvp_points.append(point)
        return self.A@v

    def reduction(self, base, trial):
        a = base.energy-trial.energy
        return Reduction(a, 0., a, {})


@dataclass
class Red:
    ared: float
    guard: float

    @property
    def certified(self):
        return self.ared-self.guard

    @property
    def reason(self):
        if abs(self.ared) <= self.guard:
            return "ENERGY_RESOLUTION"
        if self.certified <= 0:
            return "NO_STRICT_DECREASE"
        return None


class ScriptedObjective:
    """E with a constant gradient g and H = I; compare() returns the scripted reductions in order (a callable gets the
    required decrease 1e-4 pred and returns a Red)."""
    name = "scripted"

    def __init__(self, grid, g, script):
        self.grid, self.g, self.script = grid, g, list(script)
        self.state = SimpleNamespace(grid=grid, omega0=1.)
        self.evaluated = 0

    def evaluate(self, u):
        self.evaluated += 1
        return Point(u, 1.)

    def gradient(self, point):
        return self.g.copy()

    def hvp(self, point, v):
        return np.array(v, dtype=np.float64, copy=True)

    def compare(self, base, trial, *, required, context, gf_ratio):
        assert context == "lm"
        entry = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return entry(required) if callable(entry) else entry


def exact(required):
    """certified = pred exactly (ratio 1)."""
    return Red(required/CFG.ratio_accept, 0.)


def toy(n=5, seed=0, scale=.05):
    grid = syn.cube_grid(n)
    m = 3*grid.n_free_scalar
    rng = np.random.default_rng(seed)
    A = spd(m, seed=seed)
    theta_star = scale*rng.standard_normal(m)
    return grid, A, theta_star


def start_of(objective, grid):
    zero = np.zeros(grid.shape+(3,))
    return objective.evaluate(zero), restrict_free(zero, grid), passing(zero, grid)


def kinds(rows, kind):
    return [r for r in rows if r["kind"] == kind]


# ---- gate 1: default unchanged ----------------------------------------------------------------------------------------

def test_default_solver_config_has_no_lm_record():
    plain = SolverConfig()
    record = plain.record()
    assert plain.inner == "pg" and plain.lm is None and plain.lm_inner is None
    assert "inner" not in record and "lm" not in record and record["pg"] == PGConfig().record()
    assert SolverConfig(inner="pg").record() == record
    assert plain.metric_jd_scaling == plain.pg.metric_jd_scaling and plain.jd_cell_estimate == plain.pg.jd_cell_estimate
    lm = SolverConfig(inner="lm")
    assert lm.lm == LMConfig() and lm.lm_inner is lm.lm
    rec = lm.record()
    assert rec["inner"] == "lm" and rec["lm"] == LMConfig().record() and rec["pg"] is None
    assert rec["lm"]["metric_jd_scaling"] == rec["lm"]["jd_cell_estimate"] == "none"
    assert lm.metric_jd_scaling == lm.jd_cell_estimate == "none"
    with pytest.raises(InputValidationError):
        SolverConfig(inner="lm", pg=PGConfig(metric_jd_scaling="off"))       # PG settings have no effect: refused
    with pytest.raises(InputValidationError):
        SolverConfig(lm=LMConfig())                                         # LM parameters only with inner lm
    with pytest.raises(InputValidationError):
        SolverConfig(inner="newton")


def test_lm_config_defaults_are_the_contract():
    """C09.3 / D13: rtol 1e-3, maxiter 200, gamma bounds 1e-12 / 1e12, x4 and /2, thresholds 1e-4 / .25 / .75, 12
    attempts; gamma0 = min(gamma_max, max(gamma_min, 1e-3 max(1, lam_b)))."""
    assert (CFG.rtol, CFG.maxiter, CFG.gamma_min, CFG.gamma_max, CFG.increase, CFG.decrease) == (
        1e-3, 200, 1e-12, 1e12, 4., .5)
    assert (CFG.ratio_accept, CFG.ratio_low, CFG.ratio_high, CFG.max_directions) == (1e-4, .25, .75, 12)
    assert (CFG.step_tol, CFG.stall_steps) == (1e-12, 5)
    assert CFG.gamma0(0.) == 1e-3 and CFG.gamma0(1e-5) == 1e-3 and CFG.gamma0(10.) == 1e-2
    assert CFG.gamma0(1e20) == 1e12
    assert LM.LM_INNER_STATUSES[-1] == TRIAL_BUDGET_EXHAUSTED and "line_search_failed" in LM.LM_INNER_STATUSES


# ---- gate 2: quadratic toy, option (a) ----------------------------------------------------------------------------------

def test_quadratic_toy_one_step_ratio_is_one():
    grid, A, theta_star = toy()
    objective = QuadraticObjective(grid, A, theta_star)
    base, theta, cert = start_of(objective, grid)
    rows = []
    result = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(float(np.mean(np.diag(A)))),
                    certify=passing, target=0., max_steps=1, log=rows.append)
    assert result.status == MAX_ITERATIONS and result.accepted == 1
    (accept,) = kinds(rows, "accept")
    assert 1-1e-6 <= accept["ratio"] <= 1+1e-6, accept["ratio"]
    (cand,) = kinds(rows, "candidate")
    assert cand["accepted"] and cand["ratio"] == accept["ratio"] and cand["true_residual"] <= 1e-3
    assert cand["gamma"] == 1e-3 and cand["gamma_next"] == 5e-4            # ratio > .75: gamma / 2
    # pred is the undamped model -g.p - 1/2 p.A p of the accepted p (recomputed independently)
    p = restrict_free(result.point.u, grid)-theta
    g = A@(theta-theta_star)
    assert abs(cand["pred"]-(-g@p-.5*p@(A@p))) <= 1e-12*abs(cand["pred"])


# ---- gate 2: option (b) -----------------------------------------------------------------------------------------------

class QuadraticBackend:
    """F_B replaced by the quadratic of `toy` on a real ModelState's grid (for the proximal module): evaluate gives the
    longdouble total, stable_reduction the C08 form with s_E = 16 eps max(E); hvp = A (exact Hessian)."""

    def __init__(self, grid, A, theta_star):
        self.grid, self.A, self.theta_star = grid, A, theta_star

    def evaluate(self, state, u):
        d = restrict_free(u, self.grid)-self.theta_star
        total = np.longdouble(.5)*np.longdouble(d@(self.A@d))
        return SimpleNamespace(u=u, total=total, energy=float(total), energies={})

    def gradient(self, state, ev):
        return self.A@(restrict_free(ev.u, self.grid)-self.theta_star)

    def hvp(self, state, ev, v):
        return self.A@np.asarray(v)

    def stable_reduction(self, state, base, trial):
        delta = trial.total-base.total
        guard = 16*np.finfo(float).eps*max(base.energy, trial.energy)
        return Reduction(float(-delta), guard, delta, {})


class NoProxModel(ProximalObjective):
    """E_v with H_E WITHOUT the proximal blocks (only to show that pred must contain them)."""

    def hvp(self, point, v):
        return np.asarray(self.backend.hvp(self.state, point.fb, v), dtype=np.float64)


def option_b_toy(objective_class):
    state = small_state(5)
    grid = state.grid
    m = 3*grid.n_free_scalar
    A = spd(m, low=.5, high=5., seed=3)
    theta_star = .05*np.random.default_rng(3).standard_normal(m)
    backend = QuadraticBackend(grid, A, theta_star)
    params = ProximalParams(lam=.5, mu=0., nu2=.7)                     # mu = 0: E_v exactly quadratic
    v = inject_free(.02*np.random.default_rng(4).standard_normal(m), grid)
    objective = objective_class(backend, state, v, params)
    base = objective.evaluate(v)
    rows = []
    result = run_lm(objective, base, theta=restrict_free(v, grid), cert=passing(v, grid),
                    metric=ScaledIdentity(2., grid.omega/state.omega0, lam_b=params.lam), certify=passing, target=0.,
                    max_steps=1, log=rows.append)
    return result, rows


def test_option_b_toy_ratio_with_and_without_the_proximal_blocks():
    result, rows = option_b_toy(ProximalObjective)
    assert result.accepted == 1
    (accept,) = kinds(rows, "accept")
    assert 1-1e-6 <= accept["ratio"] <= 1+1e-6, accept["ratio"]
    assert kinds(rows, "lm_start")[0]["gamma0"] == 1e-3 and kinds(rows, "lm_start")[0]["lam_b"] == .5
    _, rows_no = option_b_toy(NoProxModel)
    first = kinds(rows_no, "candidate")[0]
    assert first["ratio"] is not None and abs(first["ratio"]-1) > 1e-3, first["ratio"]


def test_option_b_hvp_equals_the_explicit_assembly():
    """C09 gate: H_E v of option (b) = (H + K_lam^T K_lam + K_mu^T K_mu + K_nu2^T K_nu2) v, assembled explicitly on a
    small grid: H by columns of the model's GN HVP, K_mu by exact differences of J_q (J is a cubic polynomial along a
    line, so the Richardson combination of central differences at h and 2h is exact), K_nu2 from the curl of the unit
    fields (linear), K_lam the node mass (lam q_a/omega0); 1e-12 relative."""
    state = small_state(5)
    grid = state.grid
    m = 3*grid.n_free_scalar
    backend = NumpyBackend()
    rng = np.random.default_rng(11)
    u = inject_free(.05*rng.standard_normal(m), grid)
    v_ref = inject_free(.03*rng.standard_normal(m), grid)
    params = ProximalParams(lam=.3, mu=.4, nu2=.6)
    objective = ProximalObjective(backend, state, v_ref, params)
    point = objective.evaluate(u)
    w = state.ops.weights[None, :]
    s_mu = np.sqrt(params.mu*w/state.omega0)
    s_nu2 = np.sqrt(params.nu2*w/state.omega0)[..., None]
    q = grid.node_weights()

    def J(field):
        return gauss_geometry(field, state.ops).J

    H = np.zeros((m, m))
    K_mu, K_nu2 = [], []
    lam_cols = np.zeros((m, m))
    h = .5
    for i in range(m):
        e = np.zeros(m)
        e[i] = 1.
        E = inject_free(e, grid)
        H[:, i] = backend.hvp(state, point.fb, e)
        d1 = (J(u+h*E)-J(u-h*E))/(2*h)
        d2 = (J(u+2*h*E)-J(u-2*h*E))/(4*h)
        K_mu.append((s_mu*(4*d1-d2)/3).ravel())
        K_nu2.append((s_nu2*gauss_geometry(E, state.ops).curl).ravel())
        lam_cols[:, i] = restrict_free((params.lam*q/state.omega0)[..., None]*E, grid)
    K_mu, K_nu2 = np.array(K_mu).T, np.array(K_nu2).T
    P = lam_cols+K_mu.T@K_mu+K_nu2.T@K_nu2
    for seed in range(3):
        x = np.random.default_rng(100+seed).standard_normal(m)
        got = prox.hvp(state, point.prox, x, params)
        want = P@x
        assert np.max(np.abs(got-want)) <= 1e-12*np.max(np.abs(want))
        full = objective.hvp(point, x)
        want_full = H@x+P@x
        assert np.max(np.abs(full-want_full)) <= 1e-12*np.max(np.abs(want_full))
    # P is symmetric positive semidefinite and zero curvature directions of K_mu are not lost (mu block present)
    assert np.allclose(P, P.T, atol=1e-13*np.max(np.abs(P)))
    assert np.max(np.abs(K_mu.T@K_mu)) > 0


# ---- gate 3: damping table and gamma reset ----------------------------------------------------------------------------

def test_damping_update_table():
    c = CFG
    assert c.updated(1e-3, accepted=True, ratio=.76) == 5e-4
    assert c.updated(1e-12, accepted=True, ratio=.9) == 1e-12                # floor gamma_min
    assert c.updated(1e-3, accepted=True, ratio=.75) == 1e-3                 # .25 <= ratio <= .75: unchanged
    assert c.updated(1e-3, accepted=True, ratio=.25) == 1e-3
    assert c.updated(1e-3, accepted=True, ratio=.2499) == 4e-3               # accepted, ratio < .25: x4
    assert c.updated(1e-3, accepted=False, ratio=None) == 4e-3               # any rejection / failed direction: x4
    assert c.updated(1e-3, accepted=False, ratio=.9) == 4e-3
    assert c.updated(2.5e11, accepted=False, ratio=None) == 1e12             # gamma >= gamma_max / 4 -> gamma_max
    assert c.updated(2.4e11, accepted=False, ratio=None) == 9.6e11
    assert c.updated(1e12, accepted=False, ratio=None) == 1e12


def first_attempts(rows):
    """gamma of the first candidate after every lm_start row, and the lm_start rows."""
    out, starts, pending = [], [], False
    for r in rows:
        if r["kind"] == "lm_start":
            starts.append(r)
            pending = True
        elif r["kind"] == "candidate" and pending:
            out.append(r["gamma"])
            pending = False
    return out, starts


def test_gamma_reset_per_round_and_per_reference_refresh():
    """C09.3-12 / C09 gate: gamma0 at every (level, EM round) start (option (a), two rounds on L0 of the synthetic case)
    and at every reference refresh (option (b)), never carried over from the previous inner solve (log check)."""
    _, levels, _ = syn.four_levels(26)
    config = SolverConfig(rounds=(2, 1, 1, 1), max_levels=1, caps=Caps(max_accepted=3), inner="lm")
    rows = []
    result = solve_multilevel(NumpyBackend(), list(levels), eta=Eta(1., 1., (1., 1., 1., 1.)),
                              sigma=FixedSigma([1.]), config=config, log=rows.append)
    gammas, starts = first_attempts(rows)
    assert len(starts) == 2 and [s["round"] for s in starts] == [0, 1]
    assert gammas == [1e-3, 1e-3] and all(s["gamma0"] == 1e-3 for s in starts)
    ends = [r["gamma_next"] for r in rows if r["kind"] == "candidate" and r.get("round") == 0]
    assert ends[-1] != 1e-3                                         # round 0 ended away from gamma0: a real reset
    assert [r["metric"]["lm"]["gamma0"] for r in result.rounds] == [1e-3, 1e-3]
    # option (b): one lm_start per reference refresh, gamma0 = min(gamma_max, max(gamma_min, 1e-3 max(1, lam)))
    state = small_state(9)
    backend = NumpyBackend()
    zero = np.zeros(state.grid.shape+(3,))
    start = Commit(zero, restrict_free(zero, state.grid), backend.certify(zero, state.grid))
    rows = []
    out = solve_proximal(backend, state, start, target=0., caps=Caps(max_accepted=6, max_outer=3),
                         params=ProximalParams(lam=default_lambda(state.grid), n_in=2), log=rows.append, lm=CFG)
    gammas, starts = first_attempts(rows)
    assert out.commits == 3 and len(starts) == 3 and [s["outer"] for s in starts] == [0, 1, 2]
    assert gammas == [1e-3]*3
    # every inner solve before a refresh ended away from gamma0, so the reset is real (LMI review minor 4)
    for k in (0, 1):
        last = [r for r in rows if r["kind"] == "candidate" and r["outer"] == k][-1]
        assert last["gamma_next"] != 1e-3, (k, last["gamma_next"])
    assert out.metric["lm"]["inner_solves"] == 3 and out.metric["jd_scaling"] == {"mode": "none"}
    # the round record comes from LMMetric.record() (no gamma-0 metric built): the values build_metric(gamma=0) gives
    metric = backend.build_metric(state, c_dat=out.metric["c_dat"], b_P=out.metric["b_P"], lam_b=out.metric["lam_b"])
    assert out.metric["a"] == metric.a and out.metric["shift"] == metric.shift and out.metric["gamma"] is None


# ---- gate 4: the true residual ------------------------------------------------------------------------------------------

def test_true_residual_failure_is_a_failed_direction():
    """maxiter 1 on an ill-conditioned toy: the PCG does not reach rtol, the one true HVP finds the residual above
    rtol -> TRUE_RESIDUAL_FAILED (no certificate, no energy), gamma x4, the next attempt from the same base; the logged
    true residual is ||A p + gamma p + g|| / ||g|| recomputed independently."""
    grid, A, theta_star = toy(seed=1)
    objective = QuadraticObjective(grid, A, theta_star)
    base, theta, cert = start_of(objective, grid)
    calls = []
    rows = []
    result = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.),
                    certify=lambda u, g: calls.append(1) or passing(u, g), target=0., max_steps=1,
                    config=LMConfig(maxiter=1), log=rows.append)
    cands = kinds(rows, "candidate")
    failed = [r for r in cands if r["reason"] == LM.TRUE_RESIDUAL_FAILED]
    assert failed and failed[0]["attempt"] == 0 and failed[0]["true_residual"] > 1e-3
    assert all(r["cert_status"] is None and r["energy"] is None and r["pred"] is None for r in failed)
    assert all(r["gamma_next"] == 4*r["gamma"] for r in failed)
    assert all(not r["pcg_converged"] and r["true_residual_case"] == "pcg_maxiter" for r in failed)
    assert len(calls) == sum(1 for r in cands if r["cert_status"] is not None)
    assert result.counters.true_residual_failed == len(failed)
    # independent recomputation of the first logged true residual: one PCG step from x0 = 0 with M^-1 = I
    g = A@(theta-theta_star)
    shift = 1e-3*grid.omega
    z = -g
    alpha = (g@g)/(z@(A@z)+shift*(z@z))
    p = alpha*z
    want = np.linalg.norm(A@p+shift*p+g)/np.linalg.norm(g)
    assert abs(failed[0]["true_residual"]-want) <= 1e-12*want


def test_true_residual_check_overrides_a_claimed_convergence(monkeypatch):
    """The PCG claims convergence with a wrong p (half the solution): the one true HVP after the solve decides."""
    grid, A, theta_star = toy(seed=2)
    objective = QuadraticObjective(grid, A, theta_star)
    base, theta, cert = start_of(objective, grid)
    original = LM.lm_pcg

    def lying(apply_A, apply_Minv, b, *, rtol, maxiter):
        out = original(apply_A, apply_Minv, b, rtol=rtol, maxiter=maxiter)
        return LM.PCGOutcome(.5*out.p, out.iterations, 0., True)
    monkeypatch.setattr(LM, "lm_pcg", lying)
    rows = []
    result = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(10.), certify=passing, target=0.,
                    max_steps=1, log=rows.append)
    cands = kinds(rows, "candidate")
    assert result.status == TRIAL_BUDGET_EXHAUSTED and len(cands) == 12
    assert all(r["reason"] == LM.TRUE_RESIDUAL_FAILED and r["pcg_converged"] and r["true_residual"] > .4
               for r in cands)
    assert objective.evaluated == 1                             # only the start point: no candidate evaluated
    # C09.3-13a (h), narrowed: this TRUE_RESIDUAL_FAILED (recursive residual met rtol) is not in the LM safe set
    assert result.last_iteration == [(LM.TRUE_RESIDUAL_FAILED, LM.PCG_RTOL_MET)]*12
    zero = np.zeros(grid.shape+(3,))
    for inner in ("pg", "lm"):
        decision = continuation(result.status, last_iteration=result.last_iteration, commit_u=zero,
                                commit_cert=passing(zero, grid), inner=inner)
        assert not decision.proceed and "TRUE_RESIDUAL_FAILED" in decision.reason


class RejectingQuadratic(QuadraticObjective):
    """The quadratic toy whose every compared candidate is NO_STRICT_DECREASE."""

    def reduction(self, base, trial):
        return Reduction(-1., 0., -1., {})


def test_maxiter_true_residual_failure_continues_under_lm():
    """C09.3-13a (h), narrowed: maxiter 2 on the ill-conditioned toy and every compared candidate rejected: the iteration
    ends trial_budget_exhausted with TRUE_RESIDUAL_FAILED / pcg_maxiter at small gamma and NO_STRICT_DECREASE once gamma
    is large enough for two PCG iterations; C13-7 continues under the LM inner and terminates under the default table."""
    grid, A, theta_star = toy(seed=1)
    objective = RejectingQuadratic(grid, A, theta_star)
    base, theta, cert = start_of(objective, grid)
    result = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=0.,
                    max_steps=1, config=LMConfig(maxiter=2))
    assert result.status == TRIAL_BUDGET_EXHAUSTED and len(result.last_iteration) == 12
    reasons = {r for r, _ in result.last_iteration}
    assert reasons == {LM.TRUE_RESIDUAL_FAILED, "NO_STRICT_DECREASE"}
    assert all(case == LM.PCG_MAXITER for r, case in result.last_iteration if r == LM.TRUE_RESIDUAL_FAILED)
    zero = np.zeros(grid.shape+(3,))
    lm = continuation(result.status, last_iteration=result.last_iteration, commit_u=zero,
                      commit_cert=passing(zero, grid), inner="lm")
    pg = continuation(result.status, last_iteration=result.last_iteration, commit_u=zero,
                      commit_cert=passing(zero, grid))
    assert lm.proceed and not pg.proceed


# ---- gates 5-8: rejections, acceptance order, certificate, attempt budget ------------------------------------------------

def test_rejections_never_move_the_base_and_twelve_attempts_end_the_iteration():
    grid = syn.cube_grid(5)
    g = np.full(3*grid.n_free_scalar, 1e-2)
    objective = ScriptedObjective(grid, g, [Red(-1., 1e-12)])           # NO_STRICT_DECREASE every time
    base, theta, cert = start_of(objective, grid)
    hvp_points = []
    original = objective.hvp
    objective.hvp = lambda point, v: hvp_points.append(point) or original(point, v)
    g0 = objective.gradient(base)
    rows = []
    result = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=0.,
                    max_steps=5, gradient=g0, log=rows.append)
    assert result.status == TRIAL_BUDGET_EXHAUSTED and result.accepted == 0
    assert result.theta.tobytes() == theta.tobytes() and result.gradient is g0 and result.cert is cert
    assert result.point is base and result.energies == [1.]
    assert result.last_iteration == [("NO_STRICT_DECREASE", None)]*12
    cands = kinds(rows, "candidate")
    assert len(cands) == 12 and [r["attempt"] for r in cands] == list(range(12))
    assert all(r["iteration"] == 1 for r in cands)
    assert [r["gamma"] for r in cands] == [1e-3*4**k for k in range(12)]
    assert all(point is base for point in hvp_points)                   # H_E frozen at the base for every attempt
    assert objective.evaluated == 13 and result.counters.candidates == 12 and result.counters.directions == 12
    # C13-7: trial_budget_exhausted continues on safe reasons; the default table (inner pg) terminates on
    # TRUE_RESIDUAL_FAILED, the LM table (C09.3-13a (h), narrowed) continues only for the pcg_maxiter case;
    # NONPOSITIVE_PREDICTION terminates in both
    zero = np.zeros(grid.shape+(3,))
    ok = continuation(TRIAL_BUDGET_EXHAUSTED, last_iteration=result.last_iteration, commit_u=zero,
                      commit_cert=passing(zero, grid))
    assert ok.proceed
    mixed = [("MODEL_RATIO_REJECTED", None), (LM.TRUE_RESIDUAL_FAILED, LM.PCG_MAXITER)]
    bad = continuation(TRIAL_BUDGET_EXHAUSTED, last_iteration=mixed, commit_u=zero, commit_cert=passing(zero, grid))
    assert not bad.proceed and LM.TRUE_RESIDUAL_FAILED in bad.reason
    assert continuation(TRIAL_BUDGET_EXHAUSTED, last_iteration=mixed, commit_u=zero, commit_cert=passing(zero, grid),
                        inner="lm").proceed
    for case in (LM.PCG_RTOL_MET, None):
        assert not continuation(TRIAL_BUDGET_EXHAUSTED, last_iteration=[("MODEL_RATIO_REJECTED", None),
                                                                         (LM.TRUE_RESIDUAL_FAILED, case)],
                                commit_u=zero, commit_cert=passing(zero, grid), inner="lm").proceed
    for inner in ("pg", "lm"):
        assert continuation(TRIAL_BUDGET_EXHAUSTED, last_iteration=[(LM.NONPOSITIVE_PREDICTION, None)], commit_u=zero,
                            commit_cert=passing(zero, grid), inner=inner).proceed is False


def test_model_ratio_rejection_and_damping_on_acceptance():
    """ratio < 1e-4 -> MODEL_RATIO_REJECTED (gamma x4); an accepted step with .25 <= ratio <= .75 keeps gamma, with
    ratio < .25 multiplies it by 4 (C09.3-11/12)."""
    grid = syn.cube_grid(5)
    g = np.full(3*grid.n_free_scalar, 1e-2)
    script = [lambda r: Red(.5*r, 0.),                  # ratio 5e-5: MODEL_RATIO_REJECTED
              lambda r: Red(.5e4*r, 0.),                # ratio .5: accepted, gamma unchanged
              lambda r: Red(.1e4*r, 0.),                # ratio .1: accepted, gamma x4
              exact]
    objective = ScriptedObjective(grid, g, script)
    base, theta, cert = start_of(objective, grid)
    rows = []
    run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=0., max_steps=3,
           log=rows.append)
    c = kinds(rows, "candidate")
    assert c[0]["reason"] == LM.MODEL_RATIO_REJECTED and abs(c[0]["ratio"]-5e-5) < 1e-12
    assert c[0]["gamma_next"] == 4e-3
    assert c[1]["accepted"] and c[1]["gamma"] == 4e-3 and c[1]["gamma_next"] == 4e-3
    assert c[2]["accepted"] and c[2]["gamma_next"] == 1.6e-2
    assert c[3]["accepted"] and c[3]["gamma_next"] == 8e-3
    a = kinds(rows, "accept")
    assert [r["gamma"] for r in a] == [4e-3, 4e-3, 1.6e-2] and [r["gamma_next"] for r in a] == [4e-3, 1.6e-2, 8e-3]


def folding_toy():
    """E = 1/2 |theta - theta*|^2 with theta* a smooth field that folds (min J < 0 at full size), the real CPU
    certificate (C07-2 configuration): the near-Newton LM step folds and is rejected by the certificate."""
    grid = syn.cube_grid(8)
    t = [(a-a[0])/(a[-1]-a[0]) for a in grid.axes]
    tz, ty, tx = np.meshgrid(t[2], t[1], t[0], indexing="ij")
    bump = np.sin(np.pi*tx)*np.sin(np.pi*ty)*np.sin(np.pi*tz)
    target = np.stack([4.*bump*np.sin(np.pi*tx), np.zeros_like(bump), np.zeros_like(bump)], axis=-1)
    target = inject_free(restrict_free(target, grid), grid)
    assert gauss_geometry(target, gauss27(grid)).J.min() < 0
    objective = QuadraticObjective(grid, np.eye(3*grid.n_free_scalar), restrict_free(target, grid))
    return grid, objective


def test_certificate_gates_the_energy_and_raises_gamma():
    """C07-5 / C09.4-14: a candidate that fails the certificate (INVALID_DOMAIN, JACOBIAN_VIOLATION) is never evaluated
    (energy, ared, ratio null), gamma x4 after it; a later attempt passes, is compared and accepted.  The accepted state
    is certified."""
    backend = NumpyBackend()
    grid, objective = folding_toy()
    zero = np.zeros(grid.shape+(3,))
    base = objective.evaluate(zero)
    cert = backend.certify(zero, grid)
    rows = []
    result = run_lm(objective, base, theta=restrict_free(zero, grid), cert=cert, metric=ScaledIdentity(1.),
                    certify=backend.certify, target=0., max_steps=1, log=rows.append)
    cands = kinds(rows, "candidate")
    gated = [r for r in cands if r["cert_status"] not in (None, "certified_feasible")]
    # the near-Newton steps leave the sampling box first (INVALID_DOMAIN), then fold (JACOBIAN_VIOLATION)
    assert gated and gated[0]["attempt"] == 0
    assert {r["reason"] for r in gated} == {"INVALID_DOMAIN", "JACOBIAN_VIOLATION"}
    assert all(r["energy"] is None and r["ared"] is None and r["ratio"] is None for r in gated)
    assert all(r["gamma_next"] == 4*r["gamma"] for r in gated)
    assert objective.evaluated == 1+sum(1 for r in cands if r["cert_status"] == "certified_feasible")
    assert result.status == MAX_ITERATIONS and result.accepted == 1 and cands[-1]["accepted"]
    assert result.cert.passed and result.cert.state_hash == state_hash(result.point.u)
    assert result.cert.min_jd_lower_bound > 0 and result.energies[1] < result.energies[0]


# ---- gate 9: stagnation and endings -----------------------------------------------------------------------------------

def test_endings_converged_max_iterations_and_stagnation():
    grid = syn.cube_grid(5)
    m = 3*grid.n_free_scalar
    # converged at the start: no attempt
    objective = ScriptedObjective(grid, np.full(m, 1e-3), [exact])
    base, theta, cert = start_of(objective, grid)
    rows = []
    r = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=1e9,
               max_steps=5, log=rows.append)
    assert r.status == CONVERGED_GRADIENT and r.accepted == 0 and not kinds(rows, "candidate")
    # max_iterations at max_steps
    r = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=0.,
               max_steps=3)
    assert r.status == MAX_ITERATIONS and r.accepted == 3
    # stagnated: 5 consecutive accepted steps with step_ratio <= 1e-12 (gradient 1e-20)
    tiny = ScriptedObjective(grid, np.full(m, 1e-20), [exact])
    base, theta, cert = start_of(tiny, grid)
    rows = []
    r = run_lm(tiny, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=0., max_steps=50,
               log=rows.append)
    assert r.status == STAGNATED and r.accepted == 5
    assert all(a["step_ratio"] <= 1e-12 for a in kinds(rows, "accept"))
    # max_steps 0: max_iterations before any attempt
    r = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=0.,
               max_steps=0)
    assert r.status == MAX_ITERATIONS and r.accepted == 0


# ---- C09.3-13a (d), (e), (h): linear-solve breakdown, gamma_max, the LM continuation table ------------------------------

class BrokenHVP(ScriptedObjective):
    """H_E v = -v (indefinite: p.(H_E + gamma I)p < 0 for gamma omega < 1) or NaN (nonfinite)."""

    def __init__(self, grid, g, kind):
        super().__init__(grid, g, [exact])
        self.kind = kind

    def hvp(self, point, v):
        v = np.asarray(v, dtype=np.float64)
        return -v if self.kind == "indefinite" else np.full_like(v, np.nan)


@pytest.mark.parametrize("kind", ("indefinite", "nonfinite"))
def test_linear_solve_breakdown_ends_the_solve(kind):
    """C09.3-13a (d) (LMI review important 2): an indefinite or NaN H_E ends the solve nonfinite_evaluation after one
    logged attempt (reason NONFINITE_LINEAR_SOLVE, detail indefinite / nonfinite; no certificate, no energy, no retry);
    the base is untouched; C13-7 terminates (also with inner lm)."""
    grid = syn.cube_grid(5)
    g = np.full(3*grid.n_free_scalar, 1e-2)
    objective = BrokenHVP(grid, g, kind)
    base, theta, cert = start_of(objective, grid)
    g0 = objective.gradient(base)
    calls, rows = [], []
    result = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.),
                    certify=lambda u, gr: calls.append(1) or passing(u, gr), target=0., max_steps=5, gradient=g0,
                    log=rows.append)
    assert result.status == "nonfinite_evaluation" and result.accepted == 0
    assert result.theta.tobytes() == theta.tobytes() and result.gradient is g0 and result.cert is cert
    assert result.point is base and result.energies == [1.] and not calls and objective.evaluated == 1
    (cand,) = kinds(rows, "candidate")
    assert cand["reason"] == LM.NONFINITE_LINEAR_SOLVE and cand["detail"] == kind and cand["attempt"] == 0
    assert cand["cert_status"] is None and cand["energy"] is None and not kinds(rows, "accept")
    assert result.last_iteration == [(LM.NONFINITE_LINEAR_SOLVE, None)]
    zero = np.zeros(grid.shape+(3,))
    for inner in ("pg", "lm"):
        assert not continuation(result.status, last_iteration=result.last_iteration, commit_u=zero,
                                commit_cert=passing(zero, grid), inner=inner).proceed
        assert not continuation(TRIAL_BUDGET_EXHAUSTED, last_iteration=result.last_iteration, commit_u=zero,
                                commit_cert=passing(zero, grid), inner=inner).proceed


def test_linear_solve_breakdown_in_the_solver_is_level_failed(levels):
    """The same through solve_multilevel (option (a), L0): the round ends nonfinite_evaluation with the round start as
    its commit, the run ends level_failed (C13-7 'other')."""
    class NaNHessian(NumpyBackend):
        def hvp(self, state, ev, v, **kwargs):
            out = np.asarray(super().hvp(state, ev, v, **kwargs))
            return np.full_like(out, np.nan)
    rows = []
    result = solve_multilevel(NaNHessian(), list(levels), eta=Eta(1., 1., (1., 1., 1., 1.)), sigma=FixedSigma([1.]),
                              config=SolverConfig(rounds=(1, 1, 1, 1), max_levels=2, inner="lm"), log=rows.append)
    (row,) = result.rounds
    assert row["status"] == "nonfinite_evaluation" and row["accepted"] == 0 and not row["continuation"]["proceed"]
    assert row["last_iteration"] == [[LM.NONFINITE_LINEAR_SOLVE, None]]
    assert result.status == "level_failed" and result.level == 0
    assert row["state_hash"] == state_hash(np.zeros(levels[0].grid.shape+(3,)))
    (cand,) = kinds(rows, "candidate")
    assert cand["detail"] == "nonfinite"


def test_rejection_at_gamma_max_ends_the_iteration():
    """C09.3-13a (e) (LMI review minor 1): gamma_max = .1, every attempt rejected: gamma 1e-3, 4e-3, 1.6e-2, 6.4e-2, then
    gamma_max (6.4e-2 >= gamma_max/4); the rejection at gamma_max ends the iteration trial_budget_exhausted after 5
    attempts (not 12 bit-for-bit repetitions)."""
    grid = syn.cube_grid(5)
    g = np.full(3*grid.n_free_scalar, 1e-2)
    objective = ScriptedObjective(grid, g, [Red(-1., 0.)])
    base, theta, cert = start_of(objective, grid)
    rows = []
    result = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=0.,
                    max_steps=5, config=LMConfig(gamma_max=.1), log=rows.append)
    cands = kinds(rows, "candidate")
    assert [r["gamma"] for r in cands] == [1e-3, 4e-3, 1.6e-2, 6.4e-2, .1]
    assert result.status == TRIAL_BUDGET_EXHAUSTED and result.reason.startswith("gamma_max")
    assert result.last_iteration == [("NO_STRICT_DECREASE", None)]*5 and result.theta.tobytes() == theta.tobytes()
    assert cands[-1]["gamma_next"] == .1
    # gamma_max reached in an earlier iteration: one attempt at gamma_max, then the end (first attempt accepted at .1)
    objective = ScriptedObjective(grid, g, [Red(-1., 0.)]*4+[lambda r: Red(.1e4*r, 0.), Red(-1., 0.)])
    rows = []
    result = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=0.,
                    max_steps=5, config=LMConfig(gamma_max=.1), log=rows.append)
    cands = kinds(rows, "candidate")
    assert result.accepted == 1 and result.status == TRIAL_BUDGET_EXHAUSTED
    assert [(r["iteration"], r["gamma"]) for r in cands[4:]] == [(1, .1), (2, .1)]


def test_lm_continuation_table_in_the_solver(levels, monkeypatch):
    """C09.3-13a (h), narrowed: a round ending trial_budget_exhausted whose last iteration has 11 certificate rejections
    and one TRUE_RESIDUAL_FAILED after a PCG stopped at maxiter continues under the LM inner (the transfer to L1
    happens); with the claimed-convergence case (pcg_rtol_met) or a NONFINITE_LINEAR_SOLVE instead it ends level_failed;
    the PG table (same crafted round, inner pg) still terminates on TRUE_RESIDUAL_FAILED / pcg_maxiter."""
    import dataclasses
    from bayesreg.solve import multilevel as ML
    original = ML.solve_round

    def crafted(entry):
        last = [("JACOBIAN_VIOLATION", None)]*11+[entry]

        def solve_round(*args, **kwargs):
            out = original(*args, **kwargs)
            if kwargs["context"]["level"] == 0:
                out = dataclasses.replace(out, status=TRIAL_BUDGET_EXHAUSTED, inner_status=TRIAL_BUDGET_EXHAUSTED,
                                          last_iteration=last)
            return out
        return solve_round

    def run(entry, inner):
        monkeypatch.setattr(ML, "solve_round", crafted(entry))
        return solve_multilevel(NumpyBackend(), list(levels), eta=Eta(1., 1., (1., 1., 1., 1.)),
                                sigma=FixedSigma([1.]), config=SolverConfig(rounds=(1, 1, 1, 1), max_levels=2,
                                                                          caps=Caps(max_accepted=1), inner=inner))
    ok = run((LM.TRUE_RESIDUAL_FAILED, LM.PCG_MAXITER), "lm")
    assert ok.rounds[0]["continuation"]["proceed"] and [r["level"] for r in ok.rounds] == [0, 1]
    assert ok.status == "stopped_before_native"
    for entry in ((LM.TRUE_RESIDUAL_FAILED, LM.PCG_RTOL_MET), (LM.NONFINITE_LINEAR_SOLVE, None)):
        bad = run(entry, "lm")
        assert not bad.rounds[0]["continuation"]["proceed"] and bad.status == "level_failed" and len(bad.rounds) == 1
    pg = run((LM.TRUE_RESIDUAL_FAILED, LM.PCG_MAXITER), "pg")
    assert not pg.rounds[0]["continuation"]["proceed"] and pg.status == "level_failed"


# ---- gate 10: SCH (per-level caps, budget and shares at the iteration top only) ----------------------------------------

class IncrementingClock:
    def __init__(self):
        self.t, self.calls = 0., 0

    def __call__(self):
        self.calls += 1
        self.t += 1.
        return self.t


def test_budget_checked_only_at_the_iteration_top():
    """C13-5a LM item: one budget check (one clock reading) per LM iteration top, none between direction attempts:
    iteration 1 has three rejected attempts before the accepted fourth, yet the budget is read once for it."""
    grid = syn.cube_grid(5)
    g = np.full(3*grid.n_free_scalar, 1e-2)
    script = [Red(-1., 0.)]*3+[exact]
    objective = ScriptedObjective(grid, g, script)
    base, theta, cert = start_of(objective, grid)
    clock = IncrementingClock()
    budget = WallBudget(1e9, clock=clock, started=0.)
    rows = []
    r = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=0.,
               max_steps=3, budget=budget, log=rows.append)
    assert r.status == MAX_ITERATIONS and r.accepted == 3
    assert len(kinds(rows, "candidate")) == 6 and kinds(rows, "candidate")[3]["attempt"] == 3
    assert budget.checks == 3 and clock.calls == 3                      # tops of iterations 1, 2, 3 only


def test_level_deadline_only_at_the_iteration_top():
    """The clock passes D_0 during iteration 1's attempts: the attempts go on to the acceptance; the share ends the
    solve at the next iteration top (level_budget_reached, at lm_iteration_top, iteration 2)."""
    grid = syn.cube_grid(5)
    g = np.full(3*grid.n_free_scalar, 1e-2)

    class Clock:
        t = 0.

        def __call__(self):
            return self.t
    clock = Clock()

    def late(required):
        clock.t = 50.                                                   # past D_0 = 10 in the middle of the attempts
        return Red(-1., 0.)
    objective = ScriptedObjective(grid, g, [late, Red(-1., 0.), exact])
    base, theta, cert = start_of(objective, grid)
    budget = WallBudget(100., clock=clock, started=0.)
    deadline = LevelShares((.1, .1, .1), 100.).deadline(budget, 0)
    r = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=0.,
               max_steps=5, budget=budget, level_deadline=deadline)
    assert r.status == LEVEL_BUDGET_REACHED and r.accepted == 1
    assert deadline.hit["where"]["at"] == "lm_iteration_top" and deadline.hit["where"]["iteration"] == 2
    assert deadline.hit["check"] == budget.checks == 2
    # the budget wins at the same check
    clock.t = 0.
    objective = ScriptedObjective(grid, g, [late, exact])
    budget = WallBudget(40., clock=clock, started=0.)
    deadline = LevelShares((.1, .1, .1), 40.).deadline(budget, 0)
    r = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=0.,
               max_steps=5, budget=budget, level_deadline=deadline)
    assert r.status == BUDGET_EXHAUSTED and deadline.hit is None and r.accepted == 1


def test_natural_endings_win_over_the_budget_at_the_iteration_top():
    """C13-3 / C13-5a: at one LM iteration top a natural ending (threshold, cap) wins over a budget already used up and
    over a passed share deadline; the budget is not even read there (LMI review minor 4)."""
    grid = syn.cube_grid(5)
    g = np.full(3*grid.n_free_scalar, 1e-2)
    objective = ScriptedObjective(grid, g, [exact])
    base, theta, cert = start_of(objective, grid)

    class Late:
        def __call__(self):
            return 1e6
    for target, max_steps, status in ((1e9, 5, CONVERGED_GRADIENT), (0., 0, MAX_ITERATIONS)):
        budget = WallBudget(100., clock=Late(), started=0.)
        deadline = LevelShares((.1, .1, .1), 100.).deadline(budget, 0)
        r = run_lm(objective, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=target,
                   max_steps=max_steps, budget=budget, level_deadline=deadline)
        assert r.status == status and budget.checks == 0 and budget.hit is None and deadline.hit is None
    # stagnation wins too: 5 tiny accepted steps, the budget used up after the fifth
    tiny = ScriptedObjective(grid, np.full(3*grid.n_free_scalar, 1e-20), [exact])
    base, theta, cert = start_of(tiny, grid)

    class Clock:
        t = 0.

        def __call__(self):
            return self.t
    clock = Clock()

    def log(row):
        if row["kind"] == "accept" and row["accepted_steps"] == 5:
            clock.t = 1e6
    budget = WallBudget(100., clock=clock, started=0.)
    r = run_lm(tiny, base, theta=theta, cert=cert, metric=ScaledIdentity(1.), certify=passing, target=0., max_steps=50,
               budget=budget, log=log)
    assert r.status == STAGNATED and r.accepted == 5 and budget.hit is None and budget.checks == 5


@pytest.fixture(scope="module")
def levels():
    return syn.four_levels(26)[1]


def test_share_at_the_option_b_inner_lm_iteration_top(levels):
    """C13-5a / SCH gate 5 for the LM inner (LMI review minor 4): the clock passes D_0 at the first accepted inner step
    of outer iteration 1; the inner LM ends at its next iteration top (level_budget_reached, at lm_iteration_top,
    objective E_v); its accepted steps are dropped and appear in no commit; the round keeps the outer commit and the
    transfer starts from it."""
    from bayesreg.solve.stopping import LEVEL_BUDGET_REACHED as LBR

    class Clock:
        t = 0.

        def __call__(self):
            return self.t
    clock = Clock()
    rows = []

    def log(row):
        rows.append(row)
        if (row["kind"] == "accept" and row.get("level") == 0 and row.get("outer") == 1
                and row["accepted_steps"] == 1):
            clock.t = 15.
    budget = WallBudget(100., clock=clock, started=0.)
    config = SolverConfig(outer="b", inner="lm", rounds=(1, 1, 1, 1), caps=Caps(max_accepted=40, max_outer=5),
                          max_levels=2, level_time_shares=LevelShares((.1, .1, .1), 100.),
                          proximal=ProximalParams(lam=default_lambda(levels[-1].grid)))
    result = solve_multilevel(NumpyBackend(), list(levels), eta=Eta(1., 1., (1., 1., 1., 1.)), sigma=FixedSigma([1.]),
                              config=config, log=log, budget=budget)
    row = result.rounds[0]
    assert row["status"] == LBR and row["inner_status"] == LBR and "uncommitted" in row["reason"], row["status"]
    hit = row["level_share_hit"]["where"]
    assert hit["at"] == "lm_iteration_top" and hit["outer"] == 1 and hit["objective"] == "E_v" and hit["iteration"] == 2
    commits = kinds([r for r in rows if r.get("level") == 0], "commit")
    assert len(commits) == row["commits"] == 1 and row["state_hash"] == commits[0]["state_hash"]
    dropped = [r for r in rows if r["kind"] == "accept" and r.get("level") == 0 and r.get("outer") == 1]
    assert len(dropped) == 1 and dropped[0]["state_hash"] not in {c["state_hash"] for c in kinds(rows, "commit")}
    assert row["accepted"]-row["accepted_committed"] == len(dropped)
    assert result.transfers[0]["source_state_hash"] == row["state_hash"]
    assert result.rounds[1]["level"] == 1


def test_per_level_caps_and_shares_in_the_solver(levels):
    """Caps.per_level with the LM inner: every (level, round) ends cap_reached at a_ell (option (a)); a share ends L0 at
    the LM iteration top (level_budget_reached, the transfer goes on)."""
    eta = Eta(1., 1., (1., 1., 1., 1.))
    config = SolverConfig(rounds=(1, 1, 1, 2), caps=Caps.per_level((2, 1, 1, 1)), inner="lm", max_levels=4)
    rows = []
    result = solve_multilevel(NumpyBackend(), list(levels), eta=eta, sigma=FixedSigma([1.]), config=config,
                              log=rows.append)
    assert all(r["status"] == CAP_REACHED for r in result.rounds), [r["status"] for r in result.rounds]
    assert [(r["level"], r["round"], r["accepted"], r["cap_accepted"]) for r in result.rounds] == [
        (0, 0, 2, 2), (1, 0, 1, 1), (2, 0, 1, 1), (3, 0, 1, 1), (3, 1, 1, 1)]
    assert all(r["metric"]["jd_scaling"] == {"mode": "none"} and r["metric"]["lm"]["gamma0"] == 1e-3
               for r in result.rounds)
    assert not kinds(rows, "jd_scaling")                                  # S~ never computed under LM

    class Clock:
        t = 0.

        def __call__(self):
            return self.t
    clock = Clock()

    def tick(row):
        rows2.append(row)
        if row["kind"] == "accept" and row.get("level") == 0 and row["accepted_steps"] == 1:
            clock.t = 15.
    rows2 = []
    budget = WallBudget(100., clock=clock, started=0.)
    config = SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(max_accepted=3), inner="lm", max_levels=2,
                          level_time_shares=LevelShares((.1, .1, .1), 100.))
    result = solve_multilevel(NumpyBackend(), list(levels), eta=eta, sigma=FixedSigma([1.]), config=config, log=tick,
                              budget=budget)
    row = result.rounds[0]
    assert row["status"] == LEVEL_BUDGET_REACHED and row["accepted"] == 1
    assert row["level_share_hit"]["where"]["at"] == "lm_iteration_top"
    assert result.rounds[1]["level"] == 1 and result.transfers[0]["source_state_hash"] == row["state_hash"]


def test_solve_round_refuses_a_device_backend():
    class Device(NumpyBackend):
        def stable_compare(self, *args, **kwargs):
            raise AssertionError
    with pytest.raises(InputValidationError):
        solve_round(Device(), None, None, option="a", target=1., caps=Caps(), lm=CFG)


# ---- gate 11: descriptive -------------------------------------------------------------------------------------------------

def test_lm_and_pg_iterations_to_the_threshold_are_recorded(capsys):
    """Descriptive only (no threshold): option (a) on a synthetic 11^3 problem, LM and PG to T_B with the real model and
    certificate; both reach T_B, their accepted steps and candidates are printed."""
    backend = NumpyBackend()
    state = small_state(11)
    zero = np.zeros(state.grid.shape+(3,))
    start = Commit(zero, restrict_free(zero, state.grid), backend.certify(zero, state.grid))
    target = threshold(g0_of(backend, state))
    out = {}
    for name, lm in (("pg", None), ("lm", CFG)):
        rows = []
        r = solve_direct(backend, state, start, target=target, caps=Caps(max_accepted=300), log=rows.append, lm=lm)
        assert r.status == CONVERGED_GRADIENT and r.commit.cert.passed
        out[name] = dict(accepted=r.accepted, candidates=len(kinds(rows, "candidate")), GF_B=r.gradient_norm,
                         T_B=target)
    out["lm"]["pcg_iterations_total"] = r.metric["lm"]["pcg_iterations_total"]
    print("LMI descriptive (11^3 synthetic, option (a)):", json.dumps(out))


def test_pcg_iterations_by_level_are_recorded(levels, capsys):
    """C09 gate "PCG 迭代数随分辨率" (layer 1, synthetic): the LM system (H + gamma0 omega_ell/omega0 I) p = -g at u = 0
    with the C09.1 M (gamma0) on the four levels of the 26^3 case; every solve reaches rtol; the counts are printed (D35:
    native <= 2 x L0 is a record, not a pass/fail criterion)."""
    from bayesreg.solve.multilevel import _template
    backend = NumpyBackend()
    eta = Eta(1., 1., (1., 1., 1., 1.))
    counts = []
    for level in levels:
        host = _template(level, eta, level.grid.omega, 8192, level.fixed.shape[0])
        state = backend.prepare(host.replace(sigma2=np.asarray([1.])))
        zero = np.zeros(level.grid.shape+(3,))
        point = backend.evaluate(state, zero)
        g = backend.gradient(state, point)
        metric = LMMetric(backend, state, c_dat=backend.c_dat(state, point), b_P=backend.metric_bP(state))
        shift = 1e-3*level.grid.omega/state.omega0
        M = metric.at(1e-3)
        out = LM.lm_pcg(lambda v: backend.hvp(state, point, v)+shift*v, M.solve, -g, rtol=1e-3, maxiter=200)
        Hp = backend.hvp(state, point, out.p)
        true = np.linalg.norm(Hp+shift*out.p+g)/np.linalg.norm(g)
        assert out.converged and true <= 1e-3
        counts.append(out.iterations)
    print("LMI PCG iterations at u = 0, levels 0..3:", counts, "native / L0 =", counts[-1]/counts[0])


# ---- gate 12: run_pair ------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("lmi_runs")
    return root, RD.write_inputs(root)


def test_run_pair_inner_lm_end_to_end(inputs):
    root, base = inputs
    out = root/"run_lm"
    code = run_pair.main([*base, "--out", str(out), *RD.COMMON, "--inner", "lm"])
    assert code == 0
    config = read(out/"config.json")
    assert config["args"]["inner"] == "lm" and config["args"]["metric_jd_scaling"] == "none"
    assert config["args"]["jd_cell_estimate"] == "none"
    assert config["solver"]["inner"] == "lm" and config["solver"]["pg"] is None
    assert config["solver"]["lm"] == LMConfig().record()
    provenance = read(out/"provenance.json")
    switches = provenance["switches"]
    assert switches["inner"] == "LM" and switches["metric_jd_scaling"] == "none" and switches["jd_cell_estimate"] == "none"
    assert provenance["solver"]["inner"] == "lm"
    hyper = read(out/"hyper.json")
    assert hyper["algorithm"]["inner"] == "lm"
    assert all(r["status"] == CAP_REACHED and r["accepted"] == 2 for r in hyper["rounds"])
    assert all(r["metric"]["lm"]["gamma0"] == 1e-3 and r["metric"]["jd_scaling"] == {"mode": "none"}
               for r in hyper["rounds"])
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    assert len(kinds(rows, "lm_start")) == 4 and not kinds(rows, "jd_scaling")
    cand = kinds(rows, "candidate")
    assert cand and all(k in cand[0] for k in ("gamma", "gamma_next", "pcg_iterations", "pcg_residual", "true_residual",
                                               "pred", "ratio", "witness_cell"))
    status = read(out/"status.json")
    assert status["status"] == CAP_REACHED and (out/"disp.npz").is_file() and read(out/"cert.json")["status"] == \
        "certified_feasible"


@BITWISE
def test_run_pair_inner_pg_explicit_is_the_golden_default(inputs):
    """--inner pg given: config.json args records it, every solve quantity is the golden default (def_explicit)."""
    root, base = inputs
    out = root/"run_pg"
    code = run_pair.main([*base, "--out", str(out), *RD.COMMON, "--inner", "pg"])
    assert code == 0
    assert RD.run_digest(out) == GOLDEN_8A4D060_RUN_PAIR["def_explicit"]
    config = read(out/"config.json")
    assert config["args"]["inner"] == "pg" and "inner" not in config["solver"] and config["solver"]["pg"] is not None
    assert read(out/"provenance.json")["switches"]["inner"] == "PG"


BAD = [("--inner", "lm", "--backend", "torch64"), ("--inner", "lm", "--metric-jd-scaling", "affine_v1"),
       ("--inner", "lm", "--metric-jd-scaling", "off"), ("--inner", "lm", "--jd-cell-estimate", "certificate"),
       ("--inner", "newton")]


@pytest.mark.parametrize("extra", BAD, ids=[" ".join(b) for b in BAD])
def test_refused_in_parse_args(tmp_path, extra):
    out = tmp_path/"never"
    argv = ["--pair", "s/p", "--channel-source", "tissue", "--out", str(out), "--data-root", str(tmp_path/"nodata"),
            *extra]
    with pytest.raises(SystemExit) as info:
        run_pair.parse_args(argv)
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        run_pair.main(argv)
    assert info.value.code == 2 and not out.exists()


def test_parse_args_records():
    base = ["--pair", "s/p", "--channel-source", "tissue", "--out", "x"]
    plain = run_pair.parse_args(base)
    assert not hasattr(plain, "inner") and plain.jd_cell_estimate == "certificate"
    assert run_pair.inner_options(plain) == dict(pg=run_pair.pg_config("affine_v1", "certificate"))
    off = run_pair.parse_args(base+["--metric-jd-scaling", "off"])          # the PG path's own rule, unchanged
    assert off.jd_cell_estimate == "none" and not hasattr(off, "inner")
    lm = run_pair.parse_args(base+["--inner", "lm", "--outer", "b", "--wall-seconds", "100", "--level-time-shares",
                                   "0.1,0.1,0.1"])
    assert lm.inner == "lm" and lm.metric_jd_scaling == "none" and lm.jd_cell_estimate == "none"
    assert run_pair.inner_options(lm) == dict(inner="lm", lm=LMConfig())
    pg = run_pair.parse_args(base+["--inner", "pg", "--jd-cell-estimate", "closed27"])
    assert pg.inner == "pg" and pg.jd_cell_estimate == "closed27"
