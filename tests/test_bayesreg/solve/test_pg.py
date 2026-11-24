"""C09.2 PG: BB2 rules, domain bound, Armijo table, rejections never move the base, certificate gating."""
from dataclasses import dataclass

import numpy as np

from jd_curl.three_d.boundary import restrict_free

from bayesreg.model import ModelState, build_samplers
from bayesreg.solve import DirectObjective, NumpyBackend, PGConfig, bb2_proposal, run_pg, sampling_step_bound
from bayesreg.solve.backend import CertOutcome
from bayesreg.solve.stopping import state_hash

from . import _synthetic as syn

CFG = PGConfig()


def test_bb2_rules():
    theta, g = np.array([1., 0.]), np.array([2., 0.])
    assert bb2_proposal(CFG, theta, g, -g/2, None)[:2] == (1., "initial")
    previous = (np.zeros(2), np.zeros(2), np.zeros(2), .3)
    tau, rule, sy, yy = bb2_proposal(CFG, theta, g, np.array([-1., 0.]), previous)     # M = 2I
    assert (tau, rule, sy, yy) == (1., "bb2", 2., 2.)
    tau, rule, _, _ = bb2_proposal(CFG, theta, -g, np.array([1., 0.]), previous)
    assert (tau, rule) == (.6, "fallback_nonpositive_curvature")                        # protected 2 tau_prev
    tau, rule, _, _ = bb2_proposal(CFG, theta, g, np.array([-1e-12, 0.]), previous)
    assert (tau, rule) == (10., "bb2")                                                  # clipped to step_max
    tau, rule, _, _ = bb2_proposal(CFG, np.array([1., 1e-20]), np.array([1e-20, 1.]), np.array([0., -1.]),
                                   (np.zeros(2), np.zeros(2), np.zeros(2), 8.))
    assert (tau, rule) == (10., "fallback_small_curvature")
    tau, rule, _, _ = bb2_proposal(CFG, theta, g, np.array([np.inf, 0.]), previous)
    assert rule == "fallback_nonfinite_secant" and tau == .6


def test_sampling_step_bound_is_exact():
    grid = syn.cube_grid(6, h=.5, origin=(1., 2., 3.))
    p = np.zeros(3*grid.n_free_scalar)
    m = grid.n_free_scalar
    p[0] = 2.                    # x component of interior node (1, 1, 1): x = 1.5, box ends at 3.5
    p[m+5] = -.25                # y component of interior node (1, 1, 4): y = 2.5, box starts at 2.0
    u = np.zeros(grid.shape+(3,))
    assert sampling_step_bound(grid, u, p) == min((3.5-1.5)/2., (2.0-2.5)/-.25)
    assert sampling_step_bound(grid, u, np.zeros_like(p)) == np.inf


# ---- fake objective: Armijo table and invariance of the base under rejections --------------------------

@dataclass
class Point:
    u: np.ndarray
    energy: float


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


class FakeObjective:
    name = "fake"

    def __init__(self, state, reductions):
        self.state, self.reductions, self.evaluated = state, list(reductions), 0

    def evaluate(self, u):
        self.evaluated += 1
        return Point(u, 1.)

    def gradient(self, point):
        return np.full(3*self.state.grid.n_free_scalar, 1e-3)

    def reduction(self, base, trial):
        return self.reductions.pop(0)


class Identity:
    @staticmethod
    def solve(g):
        return g.copy()


def passing(u, grid):
    return CertOutcome(True, "certified_feasible", "positive_enclosure_cover", None, .9, .9, 0, 0, grid.n_cells, None,
                       None, state_hash(u), "v", 0.)


def small_state(n=7, **kw):
    grid = syn.cube_grid(n)
    moving = syn.moving_image(grid.identity())
    fixed = syn.warp_with_spline(moving, grid, syn.solenoidal_displacement(grid, .3))
    options = dict(sigma2=np.array([.1]), alpha=.1, nu=.1)
    options.update(kw)
    return ModelState(grid=grid, omega0=grid.omega, fixed=fixed[None], samplers=build_samplers(moving[None], grid),
                      mask=syn.brain_box(grid, 1), tau2=np.full(4, 1.), tissue=syn.uniform_tissue(grid), **options)


def start(state):
    zero = np.zeros(state.grid.shape+(3,))
    return Point(zero, 1.), restrict_free(zero, state.grid), passing(zero, state.grid)


def test_armijo_table_and_resolution_branch():
    """C09 gate: accept iff |ared| > s_E, ared - s_E > 0 and ared - s_E >= c1 tau |g.p| (first candidate tau = 1,
    M = I, g = 1e-3 * ones of length m: c1 tau |g.p| = 1e-4 * 1e-6 m)."""
    state = small_state()
    g = np.full(3*state.grid.n_free_scalar, 1e-3)
    required = 1e-4*1.*abs(float(g@-g))                    # c1 tau |g.p| exactly as run_pg forms it
    guard = 1e-12
    cases = [([Red(required, 0.)], True, None),                                   # equality accepted (s_E = 0)
             ([Red(required*(1-1e-9)+guard, guard), Red(1., guard)], True, "ARMIJO_REJECTED"),
             ([Red(guard, guard), Red(1., guard)], True, "ENERGY_RESOLUTION"),     # |ared| <= s_E
             ([Red(-1., guard), Red(1., guard)], True, "NO_STRICT_DECREASE")]
    for reductions, accepted, first_reason in cases:
        objective = FakeObjective(state, reductions)
        base, theta, cert = start(state)
        rows = []
        result = run_pg(objective, base, theta=theta, cert=cert, metric=Identity(), certify=passing, target=0.,
                        max_steps=1, log=rows.append)
        assert result.status == "max_iterations" and result.accepted == 1
        candidates = [r for r in rows if r["kind"] == "candidate"]
        assert candidates[0]["reason"] == first_reason
        assert candidates[0]["tau"] == 1. and candidates[0]["required"] == required
        if first_reason is not None:
            assert candidates[1]["tau"] == candidates[0]["tau"]/2 and candidates[1]["accepted"]


def test_rejections_never_move_the_base():
    """C09 gate: a failed line search (26 candidates, all ENERGY_RESOLUTION) returns the base theta, gradient
    and certificate unchanged (bitwise) with line_search_failed, and every candidate's reason recorded."""
    state = small_state()
    objective = FakeObjective(state, [Red(0., 1.)]*26)
    base, theta, cert = start(state)
    g0 = objective.gradient(base)
    result = run_pg(objective, base, theta=theta, cert=cert, metric=Identity(), certify=passing, target=0.,
                    max_steps=5, gradient=g0)
    assert result.status == "line_search_failed" and result.accepted == 0
    assert result.theta.tobytes() == theta.tobytes() and result.gradient is g0 and result.cert is cert
    assert result.point is base and result.energies == [1.]
    assert result.last_iteration == [("ENERGY_RESOLUTION", None)]*26
    assert objective.evaluated == 26 and result.counters.candidates == 26


def test_certificate_gates_the_energy():
    """C07-5 / C09 gate: candidates that do not pass the certificate are never evaluated (energy null in the
    log); an oversized first step folds (JACOBIAN_VIOLATION) and halving reaches a certified decrease."""
    backend = NumpyBackend()
    state = small_state(9)
    objective = DirectObjective(backend, state)
    calls = []
    original = objective.evaluate
    objective.evaluate = lambda u: calls.append(1) or original(u)

    class Huge:
        @staticmethod
        def solve(g):
            return 1e4*g
    zero = np.zeros(state.grid.shape+(3,))
    cert = backend.certify(zero, state.grid)
    point = original(zero)
    rows = []
    result = run_pg(objective, point, theta=restrict_free(zero, state.grid), cert=cert, metric=Huge(),
                    certify=backend.certify, target=0., max_steps=1, log=rows.append)
    candidates = [r for r in rows if r["kind"] == "candidate"]
    assert result.status == "max_iterations" and result.accepted == 1
    assert candidates[0]["reason"] == "JACOBIAN_VIOLATION" and candidates[0]["cert_status"] == "infeasible"
    gated = [r for r in candidates if r["cert_status"] != "certified_feasible"]
    assert gated and all(r["energy"] is None and r["ared"] is None for r in gated)
    assert len(calls) == len(candidates)-len(gated)
    assert candidates[-1]["accepted"] and candidates[-1]["certified"] >= candidates[-1]["required"] > 0
    assert result.cert.passed and result.cert.state_hash == state_hash(result.point.u)
    assert result.energies[1] < result.energies[0]
