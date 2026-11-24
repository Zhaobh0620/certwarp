"""C09.1-3a JD affine-scaling metric (work package JDS; path only).

(i)   "off" is the C09.1 path bit for bit: golden digests of the pre-change code (commit 8f89645) on the near-fold
      synthetic problem below, options (a) and (b); and the affine metric with r = 1 everywhere reproduces the off
      trajectory bit for bit (1.0 * x is exact).
(ii)  S~ M^-1 S~ is symmetric positive definite, matvec is its inverse, p = -S~ M^-1 S~ g is a descent direction.
(iii) near-fold synthetic problem (9^3 nodes, u* of amplitude (1.6, -1.4, 1.2) voxels, alpha = nu = 1e-3,
      sigma^2 = 1e-2, option (a), 25 accepted steps allowed): without scaling the solve ends line_search_failed
      after 11 accepted steps, its last line search 26 candidates all JACOBIAN_VIOLATION (the mode of the
      development smoke run); with scaling the 25-step budget is used up (cap_reached), every accepted state is
      certified with a certified decrease, the refresh rule fired, there are fewer JACOBIAN_VIOLATION rejections
      and F_B ends lower.  A single synthetic observation of the path, not a convergence statement.
(iv)  refresh rule: 3 consecutive rejected candidates or > 13 JACOBIAN_VIOLATION among the most recent 26; the
      refresh happens at the next accepted state only (never after line_search_failed, nor when the solve ends
      there), restarts the statistics, is counted, and BB2 then uses the refreshed operator.
The J estimate: Gauss-point values equal bayesreg.geom.tile_geometry, corner values CellQ1Operators3D("corners8"),
the node minimum a brute-force loop, slabs do not change a bit.
DEF (2026-10-31; contract 8a4d060): the defaults became jd_cell_estimate "certificate" and logvol_points
"gauss27+closed27".  These tests are about the C09.1-3a mechanism with the "gauss27+corners" estimate on the C05-2
model (the defaults when they were written): both are given explicitly (G27, ON, logvol_points="gauss27"), so every
golden digest and observation here is that of the code before DEF.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from itertools import product

import numpy as np
import pytest

from jd_curl.three_d.boundary import inject_free, restrict_free
from jd_curl.three_d.cell_q1 import CellQ1Operators3D
from jd_curl.three_d.grid import make_grid_3d

from bayesreg.cli import run_pair
from bayesreg.geom import gauss27, node_field, tile_geometry
from bayesreg.model import ModelState, build_metric, build_samplers
from bayesreg.solve import (Caps, Commit, JDScaledMetric, JDScalingConfig, NumpyBackend, PGConfig, ProximalParams,
                            default_lambda, g0_of, node_jd_estimate, run_pg, scaled_metric, scaling_factors,
                            solve_direct, solve_proximal, threshold)
from bayesreg.solve import jd_scaling as jds
from bayesreg.solve.backend import CertOutcome
from bayesreg.solve.stopping import state_hash

from . import _synthetic as syn

OFF = PGConfig(metric_jd_scaling="off")
G27 = JDScalingConfig(cell_estimate="gauss27+corners")      # DEF: the estimate of these tests (the default before DEF)
ON = PGConfig(jd=G27)                                       # affine_v1 with that estimate (PGConfig() before DEF)


def wavy_field(grid, amplitude=.35, seed=0):
    """A smooth nonzero field with zero faces (components differ), J well inside (0, 2)."""
    rng = np.random.default_rng(seed)
    t = [(axis-axis[0])/(axis[-1]-axis[0]) for axis in grid.axes]
    tz, ty, tx = np.meshgrid(t[2], t[1], t[0], indexing="ij")
    bump = np.sin(np.pi*tx)*np.sin(np.pi*ty)*np.sin(np.pi*tz)
    u = np.stack([bump*np.cos(3*tx+c)*amplitude*h for c, h in zip(rng.uniform(0, 3, 3), grid.spacing)], axis=-1)
    u += .05*rng.standard_normal(u.shape)*bump[..., None]
    for face in (0, -1):
        u[face], u[:, face], u[:, :, face] = 0., 0., 0.
    return u


def aniso_grid():
    return make_grid_3d(nx=9, ny=8, nz=7, hx=1., hy=.8, hz=1.2, x0=.5, y0=-1., z0=2., unit="mm")


# ---- the J estimate --------------------------------------------------------------------------------------

def test_gauss_and_corner_jacobians_match_the_geometry():
    grid = aniso_grid()
    u = node_field(wavy_field(grid), grid)
    values = jds.slab_point_jacobians(u, grid, 0, grid.cell_shape[0])
    tile = slice(0, grid.n_cells)
    geo = tile_geometry(u, gauss27(grid), tile=tile)
    for g, point in enumerate(gauss27(grid).points):
        ours = values[tuple(float(v) for v in point)].ravel()
        assert np.max(np.abs(ours-geo.J[:, g])) <= 1e-13
    corners = CellQ1Operators3D(grid, location="corners8")
    grad = np.stack([np.stack([corners.derivative(np.ascontiguousarray(u[..., r]), axis, tile=tile)
                               for axis in "xyz"], axis=-1) for r in range(3)], axis=-2)    # (cells, 8, r, j)
    J = np.linalg.det(np.eye(3)+grad)
    for g, point in enumerate(corners.points):
        ours = values[tuple(float(v) for v in point)].ravel()
        assert np.max(np.abs(ours-J[:, g])) <= 1e-13
    assert len(values) == 35
    m = jds.cell_jd_estimate(u, grid)
    assert np.max(np.abs(m.ravel()-np.minimum(geo.J.min(axis=1), J.min(axis=1)))) <= 1e-13
    assert np.array_equal(jds.cell_jd_estimate(u, grid, slab_cells=1), m)          # one z layer per slab
    assert np.array_equal(jds.cell_jd_estimate(u, grid, slab_cells=10**9), m)      # one slab


def test_node_estimate_is_the_minimum_over_adjacent_cells():
    grid = aniso_grid()
    u = wavy_field(grid, amplitude=.6, seed=3)
    m = jds.cell_jd_estimate(u, grid)
    J = node_jd_estimate(u, grid)
    assert J.shape == grid.interior_shape
    nz, ny, nx = grid.shape
    for k, j, i in product(range(1, nz-1), range(1, ny-1), range(1, nx-1)):
        assert J[k-1, j-1, i-1] == m[k-1:k+1, j-1:j+1, i-1:i+1].min()


def test_scaling_factors_and_free_layout():
    grid = aniso_grid()
    u = wavy_field(grid, amplitude=1.2, seed=5)
    J = node_jd_estimate(u, grid)
    r, record = scaling_factors(u, grid)
    assert np.array_equal(r, np.minimum(1., np.maximum(J, 1e-12)/.25))
    assert 0 < record["fraction_r_lt_1"] < 1 and record["min_J_a"] == J.min() and record["min_r"] == r.min()
    assert record["n_r_lt_1"] == np.count_nonzero(J < .25)
    s = jds.free_scaling(r)
    field = inject_free(s, grid)
    for c in range(3):
        assert np.array_equal(field[1:-1, 1:-1, 1:-1, c], r)
    r1, _ = scaling_factors(np.zeros(grid.shape+(3,)), grid)
    assert np.all(r1 == 1.)
    with pytest.raises(Exception):
        JDScalingConfig(activation=1.5)
    with pytest.raises(Exception):
        PGConfig(metric_jd_scaling="legacy")


# ---- (ii) symmetric positive definite, inverse, descent ----------------------------------------------------

def small_state(n=6, **kw):
    grid = syn.cube_grid(n)
    moving = syn.moving_image(grid.identity())
    fixed = syn.warp_with_spline(moving, grid, syn.solenoidal_displacement(grid, .3))
    options = dict(sigma2=np.array([.1]), alpha=.1, nu=.1, logvol_points="gauss27")       # DEF: explicit C05-2
    options.update(kw)
    return ModelState(grid=grid, omega0=grid.omega, fixed=fixed[None], samplers=build_samplers(moving[None], grid),
                      mask=syn.brain_box(grid, 1), tau2=np.full(4, 1.), tissue=syn.uniform_tissue(grid), **options)


def test_scaled_metric_is_spd_and_gives_descent():
    state = small_state()
    grid = state.grid
    base = build_metric(state, c_dat=.3, b_P=2.)
    u = wavy_field(grid, amplitude=1.2, seed=1)                 # min J_a ~ .08 > 0: a quarter of the nodes damped
    metric = JDScaledMetric(base, grid, u, config=G27)
    s = metric.scale
    assert .25 < s.min() < 1 and s.max() == 1.
    n = 3*grid.n_free_scalar
    P = np.stack([metric.solve(e) for e in np.eye(n)], axis=1)
    B = np.stack([base.solve(e) for e in np.eye(n)], axis=1)
    assert np.max(np.abs(P-s[:, None]*B*s[None, :])) <= 1e-13*np.max(np.abs(B))
    assert np.max(np.abs(P-P.T)) <= 1e-12*np.max(np.abs(P))
    eig = np.linalg.eigvalsh((P+P.T)/2)
    assert eig.min() > 0
    np.linalg.cholesky((P+P.T)/2)
    rng = np.random.default_rng(7)
    for _ in range(5):
        g = rng.standard_normal(n)
        p = -metric.solve(g)
        assert g@p < 0
        assert np.linalg.norm(metric.matvec(metric.solve(g))-g) <= 1e-10*np.linalg.norm(g)
    assert metric.a == base.a and metric.shift == base.shift
    assert scaled_metric(base, grid, u, "off") is base


# ---- (i) off is bitwise the pre-change path -----------------------------------------------------------------

def near_fold_problem(n=9, amplitude=(1.6, -1.4, 1.2)):
    grid = syn.cube_grid(n)
    moving = syn.moving_image(grid.identity())
    fixed = syn.warp_with_spline(moving, grid, syn.true_displacement(grid, amplitude))
    state = ModelState(grid=grid, omega0=grid.omega, fixed=fixed[None], samplers=build_samplers(moving[None], grid),
                       mask=syn.brain_box(grid, 1), sigma2=np.array([1e-2]), alpha=1e-3, nu=1e-3,
                       tau2=np.full(4, 1.), tissue=syn.uniform_tissue(grid), logvol_points="gauss27")   # DEF: explicit
    return grid, state


def digest(rows, result):
    keep = []
    for r in rows:
        if r["kind"] == "candidate":
            keep.append(["c", r["iteration"], r["attempt"], float(r["tau"]).hex(), r["reason"],
                         None if r["energy"] is None else float(r["energy"]).hex(),
                         None if r["min_jd_lower_bound"] is None else float(r["min_jd_lower_bound"]).hex()])
        elif r["kind"] == "accept":
            keep.append(["a", r["iteration"], float(r["tau"]).hex(), float(r["energy"]).hex(), r["state_hash"]])
        elif r["kind"] in ("commit", "commit_check"):
            keep.append([r["kind"], r["outer"], r.get("state_hash")])
    text = json.dumps(dict(rows=keep, status=result.status, hash=result.commit.cert.state_hash,
                           energies=[float(e).hex() for e in result.energies]), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def run(option, pg, steps=25, problem=near_fold_problem):
    backend = NumpyBackend()
    grid, state = problem()
    zero = np.zeros(grid.shape+(3,))
    start = Commit(zero, restrict_free(zero, grid), backend.certify(zero, grid))
    target = threshold(g0_of(backend, state))
    rows = []
    if option == "a":
        result = solve_direct(backend, state, start, target=target, caps=Caps(max_accepted=steps), log=rows.append,
                              pg=pg)
    else:
        result = solve_proximal(backend, state, start, target=target, caps=Caps(max_accepted=steps, max_outer=4),
                                params=ProximalParams(lam=default_lambda(grid)), log=rows.append, pg=pg)
    return result, rows


# sha256 of the trajectory (every candidate's tau, reason, energy, certificate bound; every accepted / committed
# state hash; the status and F_B trace) of the code before C09.1-3a, commit 8f89645, numpy 2.0.2, this machine
GOLDEN_8F89645 = dict(a="ffbd319911d56b692542c85439e76e96948eefeb86aa1684c773d24147bbaf0e",
                      b="0f6f9169776480b407cb427da96adf2fedf0bed2400b12cad368885b2acddd56")


@pytest.mark.skipif(np.__version__ != "2.0.2", reason="bitwise golden digests recorded with numpy 2.0.2")
@pytest.mark.parametrize("option", ["a", "b"])
def test_off_is_bitwise_the_pre_change_path(option):
    result, rows = run(option, OFF)
    assert digest(rows, result) == GOLDEN_8F89645[option]
    assert result.metric["jd_scaling"] == dict(mode="off")
    assert not [r for r in rows if r["kind"] == "jd_scaling"]


def test_affine_with_unit_factors_is_bitwise_off():
    """r = 1 at every node (J >= .25 everywhere on a mild problem): S~ M^-1 S~ g = 1.0 * M^-1 (1.0 * g) bit for bit."""
    def mild():
        return near_fold_problem(amplitude=(.3, -.2, .25))
    off, rows_off = run("a", OFF, steps=8, problem=mild)
    on, rows_on = run("a", ON, steps=8, problem=mild)
    record = on.metric["jd_scaling"]
    assert all(c["fraction_r_lt_1"] == 0 for c in record["computations"])
    assert digest(rows_on, on) == digest(rows_off, off)
    assert on.commit.theta.tobytes() == off.commit.theta.tobytes()


# ---- (iii) the near-fold synthetic problem ------------------------------------------------------------------

def test_near_fold_problem_scaling_keeps_accepting():
    off, rows_off = run("a", OFF)
    on, rows_on = run("a", ON)
    count_off = Counter(r["reason"] for r in rows_off if r["kind"] == "candidate")
    count_on = Counter(r["reason"] for r in rows_on if r["kind"] == "candidate")
    # without scaling: the smoke-run mode (line_search_failed, the last line search all JACOBIAN_VIOLATION)
    assert off.status == "line_search_failed" and off.accepted < 25
    assert off.last_iteration == [("JACOBIAN_VIOLATION", "JACOBIAN_VIOLATION")]*26
    # with scaling: the budget is used up, every accepted state certified with a certified decrease
    assert on.status == "cap_reached" and on.accepted == 25
    accepted = [r for r in rows_on if r["kind"] == "candidate" and r["accepted"]]
    assert len(accepted) == 25 and all(r["cert_status"] == "certified_feasible" and r["certified"] > 0
                                       for r in accepted)
    assert np.all(np.diff(on.energies) < 0) and on.energies[-1] < off.energies[-1]
    assert count_on["JACOBIAN_VIOLATION"] < count_off["JACOBIAN_VIOLATION"]
    record = on.metric["jd_scaling"]
    assert record["jd_scaling_refresh"] >= 1 and record["computations"][0]["at"] == "round_start"
    logged = [r for r in rows_on if r["kind"] == "jd_scaling"]
    assert len(logged) == len(record["computations"]) == 1+record["jd_scaling_refresh"]
    assert all(0 <= c["fraction_r_lt_1"] <= 1 and c["min_J_a"] > 0 for c in record["computations"])


def test_option_b_records_every_reference_refresh():
    on, rows = run("b", ON)
    record = on.metric["jd_scaling"]
    at = Counter(c["at"] for c in record["computations"])
    inner_solves = {r["outer"] for r in rows if r["kind"] == "candidate"}
    assert at["reference_refresh"] == len(inner_solves) == on.commits and at["refresh"] == record["jd_scaling_refresh"]
    assert len([r for r in rows if r["kind"] == "jd_scaling"]) == len(record["computations"])


# ---- (iv) the refresh rule ------------------------------------------------------------------------------

JV = "JACOBIAN_VIOLATION"


def fresh_metric(config=G27):
    grid = aniso_grid()

    class Identity:
        a, shift = 0., 0.

        @staticmethod
        def solve(g):
            return g.copy()

        @staticmethod
        def matvec(v):
            return v.copy()
    return JDScaledMetric(Identity(), grid, np.zeros(grid.shape+(3,)), config=config), grid


def test_refresh_rule_consecutive_and_window():
    metric, grid = fresh_metric()
    for reason in (JV, "NO_STRICT_DECREASE"):
        metric.observe(reason)
    assert metric.refresh_due is None
    metric.observe("ARMIJO_REJECTED")                                     # any rejection reason counts
    assert metric.refresh_due == "3 consecutive rejected candidates"
    record = metric.refresh(np.zeros(grid.shape+(3,)), iteration=4)
    assert metric.refreshes == 1 and metric.refresh_due is None
    assert record["at"] == "refresh" and record["trigger"] == "3 consecutive rejected candidates"
    assert record["iteration"] == 4 and len(metric.computations) == 2
    # window: (JV, JV, accepted) never has 3 consecutive rejections; due once JV > 13 of the last 26
    seen = []
    for k in range(30):
        metric.observe(None if k % 3 == 2 else JV)
        seen.append(None if k % 3 == 2 else JV)
        if metric.refresh_due is not None:
            break
    assert sum(r == JV for r in seen[-26:]) == 14 and len(seen) == 20
    assert "14 of the most recent 20" in metric.refresh_due
    metric.refresh(np.zeros(grid.shape+(3,)))
    for _ in range(13):                                                   # statistics restarted: 13 JV of 26
        metric.observe(JV)
        metric.observe(None)
    assert metric.refresh_due is None and metric.refreshes == 2


@dataclass
class Point:
    u: np.ndarray
    energy: float


@dataclass
class Accept:
    ared: float = 1.
    guard: float = 0.
    certified: float = 1.
    reason: str = None


class Quadratic:
    """E = 1/2 |theta - theta*|^2 (gradient theta - theta*); every evaluated candidate is a certified decrease."""
    name = "quadratic"

    def __init__(self, state, theta_star):
        self.state, self.theta_star = state, theta_star

    def evaluate(self, u):
        return Point(u, 1.)

    def gradient(self, point):
        return restrict_free(point.u, self.state.grid)-self.theta_star

    def reduction(self, base, trial):
        return Accept()


@dataclass
class Holder:
    grid: object
    omega0: float = 1.


def outcome(u, grid, passed):
    if passed:
        return CertOutcome(True, "certified_feasible", "positive_enclosure_cover", None, .5, .5, 0, 0, grid.n_cells,
                           None, None, state_hash(u), "v", 0.)
    return CertOutcome(False, "infeasible", JV, JV, -1., -1., 0, 1, grid.n_cells, None, None, state_hash(u), "v", 0.)


def test_refresh_after_the_next_accepted_step_and_bb2_uses_the_new_operator():
    """Iteration 1: three JACOBIAN_VIOLATION rejections, then an accepted step -> S~ refreshed at that state (r < 1
    somewhere, activation 1); iteration 2: BB2's y.M^-1 y is y.(S~' M^-1 S~') y with the refreshed S~'."""
    metric, grid = fresh_metric(JDScalingConfig(activation=1., cell_estimate="gauss27+corners"))
    assert np.all(metric.scale == 1.)
    theta_star = .3*restrict_free(wavy_field(grid, amplitude=1., seed=4), grid)
    objective = Quadratic(Holder(grid), theta_star)
    calls = []

    def certify(u, grid_):
        calls.append(1)
        return outcome(u, grid_, len(calls) > 3)
    zero = np.zeros(grid.shape+(3,))
    rows = []
    result = run_pg(objective, Point(zero, 1.), theta=restrict_free(zero, grid), cert=outcome(zero, grid, True),
                    metric=metric, certify=certify, target=0., max_steps=2, log=rows.append)
    assert result.status == "max_iterations" and result.accepted == 2
    kinds = [r["kind"] for r in rows]
    assert kinds.index("jd_scaling") == kinds.index("accept")+1                  # right after the first accept
    assert metric.refreshes == 1 and metric.computations[1]["trigger"] == "3 consecutive rejected candidates"
    assert 0 < metric.scale.min() < 1
    assert result.counters.metric_solves == 3                                     # 2 directions + 1 BB2 recompute
    candidates = [r for r in rows if r["kind"] == "candidate"]
    assert [r["reason"] for r in candidates[:4]] == [JV, JV, JV, None]
    tau1 = candidates[3]["tau"]
    theta0, g0 = np.zeros_like(theta_star), -theta_star
    theta1 = theta0+tau1*theta_star                                                # p = -S M^-1 S g0 = theta_star
    g1 = theta1-theta_star
    s, y = theta1-theta0, g1-g0
    second = [r for r in candidates if r["iteration"] == 2][0]
    assert second["rule"] == "bb2"
    expected = float(y@(metric.scale*metric.scale*y))
    assert abs(second["bb2_yy"]-expected) <= 1e-12*abs(expected)
    assert abs(second["bb2_sy"]-float(s@y)) <= 1e-12*abs(float(s@y))


def test_no_refresh_when_the_solve_ends_after_the_accepted_step():
    """The rule fired in the last line search of the solve (max_steps = 1): S~ is not recomputed (it would only be
    used by a next direction)."""
    metric, grid = fresh_metric(JDScalingConfig(activation=1., cell_estimate="gauss27+corners"))
    objective = Quadratic(Holder(grid), .3*restrict_free(wavy_field(grid, amplitude=1., seed=4), grid))
    calls = []

    def certify(u, grid_):
        calls.append(1)
        return outcome(u, grid_, len(calls) > 3)
    zero = np.zeros(grid.shape+(3,))
    rows = []
    result = run_pg(objective, Point(zero, 1.), theta=restrict_free(zero, grid), cert=outcome(zero, grid, True),
                    metric=metric, certify=certify, target=0., max_steps=1, log=rows.append)
    assert result.status == "max_iterations" and metric.refresh_due is not None and metric.refreshes == 0
    assert not [r for r in rows if r["kind"] == "jd_scaling"] and result.counters.metric_solves == 1


def test_no_refresh_after_line_search_failed():
    metric, grid = fresh_metric()
    objective = Quadratic(Holder(grid), np.ones(3*grid.n_free_scalar))
    zero = np.zeros(grid.shape+(3,))
    rows = []
    result = run_pg(objective, Point(zero, 1.), theta=restrict_free(zero, grid), cert=outcome(zero, grid, True),
                    metric=metric, certify=lambda u, g: outcome(u, g, False), target=0., max_steps=3, log=rows.append)
    assert result.status == "line_search_failed" and metric.refresh_due is not None
    assert metric.refreshes == 0 and not [r for r in rows if r["kind"] == "jd_scaling"]


# ---- CLI --------------------------------------------------------------------------------------------------

def test_cli_switch():
    base = ["--pair", "s/p", "--channel-source", "none", "--out", "x"]
    assert run_pair.parse_args(base).metric_jd_scaling == "affine_v1"
    assert run_pair.parse_args(base+["--metric-jd-scaling", "off"]).metric_jd_scaling == "off"
    with pytest.raises(SystemExit):
        run_pair.parse_args(base+["--metric-jd-scaling", "legacy"])
    assert PGConfig().record()["metric_jd_scaling"] == "affine_v1"
    assert OFF.record()["jd_scaling"] is None and OFF.record()["direction"] == "p = -M^-1 g"
