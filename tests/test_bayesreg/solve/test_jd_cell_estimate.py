"""C09.1-3a jd_cell_estimate (work packages JDC and JDC1; path only; contract item of 2026-10-30, D47; review
docs/bayesreg/reviews/JDC_复核_20261030.md, its important items 1-3 fixed in JDC1).

DEF (2026-10-31; contract 8a4d060 C09.1-3a last item, D47): the default became "certificate"; "gauss27+corners" (the
default before DEF, called "default" below) is given explicitly (G27_PG, JDScalingConfig(cell_estimate=G27)), and the
synthetic F_B problems explicitly use the C05-2 model (logvol_points "gauss27", the model default before DEF).  With
metric_jd_scaling "off" an explicit value is refused and the unset option is recorded as "none".
(i)   "gauss27+corners" is bitwise the code before the option (commit a4b4deb): golden digests of every
      candidate / accept / jd_scaling row (S~ records without the timing), the status and the energy trace of a short
      run_pg on the edge-dip quadratic with the real certificate and of the near-fold problem (options (a) and (b));
      the configuration record differs only by the new key `cell_estimate`.
(ii)  closed27: J at the closed27 points equals CellQ1Operators3D("closed27_uniform") to rounding; the 35 default
      points are bitwise those of the default computation; m_K is the brute-force minimum over the 53 points.  Q1 J
      restricted to a cell edge is a quadratic in the edge parameter (why closed27 is only a heuristic).
(iii) the edge-dip cells (tests/.../_edge_dip.py): their Q1 J minimum lies inside an edge.  "mid" (dip at the edge
      midpoint, a closed27 point): the default estimate does not see it (m_K = .151, the 8 vertices r_a ~ .4), closed27
      and certificate do (r_a < .01).  "off" (dip at zeta = .25, between closed27 points; review important item 3):
      closed27 gives exactly the default's m_K and r_a (~ .4), only certificate sees it (r_a < .01).
(iv)  certificate: the minimum of the per-cell bounds equals certify_map's min_jd_lower_bound bit for bit, with the
      default and a small tile, on fields whose global minimum comes from each path of the per-cell certificate (review
      important item 1): edge hull, float prescreen, interval Bernstein fallback at the root box (no subdivision), and
      subdivision; the value of the minimizing cell checked against the jd_curl function of its path; the path counts
      equal certify_map's; every b_K is positive and at most J at the 53 points of its cell (review suggestion 3); the
      cache; every misuse raises (also a certificate of another configuration version, suggestion 2).
(v)   synthetic problems, written per option (single synthetic observations, not convergence statements):
      - the default estimate is blocked everywhere (line_search_failed, the last line search all JACOBIAN_VIOLATION);
      - certificate keeps descending to the step cap on all three problems;
      - closed27 reaches the cap on F_B of bayesreg.model (C08 comparison), but on the quadratic test double with the
        real certificate it is blocked as well: "mid" after 29 accepted steps of 30 (line_search_failed), "off" from
        the start, with the default's trajectory (it does not see that dip).
(vi)  wiring: refresh gets the accepted base's CertOutcome; the freeze points of options (a) and (b) get theirs; CLI.
(vii) torch64 (review important item 2): with Torch64Backend on the cpu device the three options give the numpy
      backend's step sequence (every candidate's outer round, iteration, attempt, decision and reason; tau and F_B to
      rounding) and the same S~: the freeze computation bit for bit (same host u), the later ones (at accepted states
      that differ by rounding between the backends) to 1e-9 with equal counts.  The cuda variant runs only on request
      (JDC_TEST_CUDA=1, GPU idle).
"""
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
from itertools import product
import os

import numpy as np
import pytest

from jd_curl.validation import InputValidationError
from jd_curl.three_d.boundary import restrict_free
from jd_curl.three_d.cell_q1 import VERTICES, CellQ1Operators3D
from jd_curl.three_d.certificate_prescreen import prescreen_lower_bounds
from jd_curl.three_d.compiled_enclosure import parallel_margins
from jd_curl.three_d.compiled_fallback import fallback_arrays
from jd_curl.three_d.feasibility import CertificateConfig, _node_indices_for, certificate_version, certify_map

from bayesreg.cli import run_pair
from bayesreg.eval.cert_eval import PROVEN_FALLBACK, PROVEN_HULL, PROVEN_PRESCREEN, cell_pass
from bayesreg.model import ModelState, build_samplers
from bayesreg.solve import (Caps, CertificateContractError, Commit, JDScaledMetric, JDScalingConfig, NumpyBackend,
                            PGConfig, ProximalParams, certificate_config, certify, default_lambda, g0_of, run_pg,
                            solve_direct, solve_proximal, threshold)
from bayesreg.solve import jd_scaling as jds
from bayesreg.geom import node_field

from . import _edge_dip as dip
from . import _synthetic as syn
from .test_jd_scaling import aniso_grid, near_fold_problem, wavy_field

G27, C27, CERT = jds.GAUSS_CORNERS, jds.CLOSED27, jds.CERTIFICATE
G27_PG = PGConfig(jd=JDScalingConfig(cell_estimate=G27))    # DEF: PGConfig() before DEF


@pytest.fixture(autouse=True)
def _empty_cache():
    jds.clear_cell_bounds_cache()
    yield
    jds.clear_cell_bounds_cache()


# ---- helpers ----------------------------------------------------------------------------------------------

def _hex(v):
    return None if v is None else float(v).hex()


def rows_digest(rows, status, energies):
    """sha256 of every candidate / accept / jd_scaling (without seconds) / commit row, the status and the energies."""
    keep = []
    for r in rows:
        kind = r["kind"]
        if kind == "candidate":
            keep.append(["c", r.get("outer"), r["iteration"], r["attempt"], _hex(r["tau"]), r["rule"], r["reason"],
                         r["cert_status"], r["cert_reason"], _hex(r["energy"]), _hex(r["min_jd_lower_bound"]),
                         _hex(r["ared"]), _hex(r["certified"]), _hex(r["required"])])
        elif kind == "accept":
            keep.append(["a", r.get("outer"), r["iteration"], _hex(r["tau"]), _hex(r["energy"]), r["state_hash"],
                         _hex(r["min_jd_lower_bound"]), _hex(r["gradient_norm"])])
        elif kind == "jd_scaling":
            keep.append(["j"]+[[k, _hex(v) if isinstance(v, float) else v] for k, v in sorted(r.items())
                               if k != "seconds"])
        elif kind in ("commit", "commit_check"):
            keep.append([kind, r["outer"], r.get("state_hash")])
    text = json.dumps(dict(rows=keep, status=status, energies=[_hex(e) for e in energies]), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class Point:
    u: np.ndarray
    energy: float


@dataclass
class Reduction:
    ared: float
    guard: float
    certified: float
    reason: str


class Identity:
    a, shift = 0., 0.

    @staticmethod
    def solve(g):
        return g.copy()

    @staticmethod
    def matvec(v):
        return v.copy()


class Quadratic:
    """E = 1/2 |theta - theta*|^2; reduction E_b - E_t (no guard): a test double of the C08 comparison."""
    name = "quadratic"

    def __init__(self, grid, theta_star):
        self.grid, self.theta_star = grid, theta_star
        self.state = type("Holder", (), dict(grid=grid, omega0=1.))()

    def evaluate(self, u):
        d = restrict_free(u, self.grid)-self.theta_star
        return Point(u, .5*float(d@d))

    def gradient(self, point):
        return restrict_free(point.u, self.grid)-self.theta_star

    def reduction(self, base, trial):
        a = base.energy-trial.energy
        return Reduction(a, 0., a, None if a > 0 else "NO_STRICT_DECREASE")


def dip_problem(n=8, push="gradient", kind="mid"):
    """(grid, cell, u0, w): the edge-dip field `kind` ("mid", "off") with the cell K = ((n-2)//2,)*3 and the push w
    (K's vertices: the unit direction lowering J at the dip fastest, or a constant -.2; elsewhere a smooth bump of
    amplitude .25)."""
    grid = syn.cube_grid(n)
    c = (n-2)//2
    cell = (c, c, c)
    u0 = dip.edge_dip_field(grid, cell, kind)
    t = [(a-a[0])/(a[-1]-a[0]) for a in grid.axes]
    tz, ty, tx = np.meshgrid(t[2], t[1], t[0], indexing="ij")
    bump = np.sin(np.pi*tx)*np.sin(np.pi*ty)*np.sin(np.pi*tz)
    w = np.stack([.25*bump*np.cos(2*tx+k) for k in range(3)], axis=-1)
    if push == "gradient":
        _, dJ = dip.cell_point_jacobian(u0, grid, cell, dip.dip_point(kind))
        dJ = -dJ/np.linalg.norm(dJ)
        for v, (x, y, z) in enumerate(VERTICES):
            w[c+z, c+y, c+x] = dJ[v]
    else:
        w[c:c+2, c:c+2, c:c+2] = -.2
    return grid, cell, u0, w


def run_quadratic(estimate, *, steps=30, push="gradient", rows=None, kind="mid"):
    backend = NumpyBackend()
    grid, cell, u0, w = dip_problem(push=push, kind=kind)
    cert0 = backend.certify(u0, grid)
    theta0 = restrict_free(u0, grid)
    objective = Quadratic(grid, theta0+restrict_free(w, grid))
    metric = JDScaledMetric(Identity(), grid, u0, config=JDScalingConfig(cell_estimate=estimate), at="freeze",
                            **({} if estimate == G27 else dict(cert=cert0, cert_config=backend.cert_config)))
    rows = [] if rows is None else rows
    rows.append(dict(kind="jd_scaling", **metric.computations[0]))
    base = objective.evaluate(u0)
    result = run_pg(objective, base, theta=theta0, cert=cert0, metric=metric, certify=backend.certify, target=0.,
                    max_steps=steps, log=rows.append)
    return result, rows, metric


def run_outer(option, pg, steps=25, problem=near_fold_problem, max_outer=4):
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
        result = solve_proximal(backend, state, start, target=target, caps=Caps(max_accepted=steps,
                                                                                max_outer=max_outer),
                                params=ProximalParams(lam=default_lambda(grid)), log=rows.append, pg=pg)
    return result, rows


# ---- (i) the default is bitwise the code before the option --------------------------------------------------------

# sha256 of rows_digest (and of the S~ records without timing) computed with the code of commit a4b4deb (before the
# option; the same functions run against `git archive a4b4deb`), numpy 2.0.2, this machine.
GOLDEN_A4B4DEB = dict(
    quadratic="522f3d56c67ae5fa44d3a2b7d89faad19809555a9c66bd202174d1bf3d40ea06",
    a=("72e3f9c269ad8f93c82a9b16b4363b10843c3f344ed83754b2110551227c970c",
       "3d06d901c46a7ee843acc7e6a9591f2366e3f1916248b6d393146f2e3a031660"),
    b=("14ce8be8fdae2343c0e2fb048818980395780f1e45d7180ee8db287dcf6ef437",
       "67c57bd0655ca901cee4d86189515458772abcae6d2f4a36de958411ef87a8da"),
    record="815a132030e3e955f840415c763914956e8fb55fb9375a8a91903a1926088a58")

BITWISE = pytest.mark.skipif(np.__version__ != "2.0.2", reason="bitwise golden digests recorded with numpy 2.0.2")


@BITWISE
def test_default_run_pg_is_bitwise_the_pre_option_code():
    result, rows, metric = run_quadratic(G27, push="constant")
    assert rows_digest(rows, result.status, result.energies) == GOLDEN_A4B4DEB["quadratic"]
    assert result.status == "line_search_failed" and result.accepted == 15
    assert all("cell_estimate" not in c for c in metric.computations)      # the S~ records keep their fields


@BITWISE
@pytest.mark.parametrize("option", ["a", "b"])
def test_default_outer_is_bitwise_the_pre_option_code(option):
    result, rows = run_outer(option, G27_PG)
    comps = [{k: v for k, v in c.items() if k != "seconds"} for c in result.metric["jd_scaling"]["computations"]]
    digest = hashlib.sha256(json.dumps(comps, sort_keys=True, default=_hex).encode()).hexdigest()
    assert (rows_digest(rows, result.status, result.energies), digest) == GOLDEN_A4B4DEB[option]
    assert result.metric["jd_scaling"]["cell_estimate"] == G27


def test_default_record_only_gains_the_option_key():
    assert JDScalingConfig().cell_estimate == CERT                          # DEF: the default since 2026-10-31
    record = JDScalingConfig(cell_estimate=G27).record()
    assert record.pop("cell_estimate") == G27 and record["estimate"] == jds.ESTIMATE
    assert hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() == GOLDEN_A4B4DEB["record"]
    assert PGConfig(metric_jd_scaling="off").record()["jd_scaling"] is None
    for estimate in (C27, CERT):
        rec = JDScalingConfig(cell_estimate=estimate).record()
        assert rec["cell_estimate"] == estimate and rec["estimate"] == jds.ESTIMATES[estimate] != jds.ESTIMATE


# ---- (ii) closed27 points ---------------------------------------------------------------------------------------

def test_closed27_jacobians_match_the_geometry_and_the_default_points_are_unchanged():
    grid = aniso_grid()
    u = node_field(wavy_field(grid, amplitude=.9, seed=2), grid)
    k1 = grid.cell_shape[0]
    default = jds.slab_point_jacobians(u, grid, 0, k1)
    closed = jds.slab_point_jacobians(u, grid, 0, k1, closed=True)
    assert len(default) == 35 and len(closed) == 53
    for point, value in default.items():                               # the 35 default points: bit for bit
        assert np.array_equal(closed[point], value)
    ops = CellQ1Operators3D(grid, location="closed27_uniform")
    tile = slice(0, grid.n_cells)
    grad = np.stack([np.stack([ops.derivative(np.ascontiguousarray(u[..., r]), axis, tile=tile) for axis in "xyz"],
                              axis=-1) for r in range(3)], axis=-2)      # (cells, 27, r, j)
    J = np.linalg.det(np.eye(3)+grad)
    for g, point in enumerate(ops.points):
        assert np.max(np.abs(closed[tuple(float(v) for v in point)].ravel()-J[:, g])) <= 1e-13
    m, m_default = jds.closed27_cell_estimate(u, grid)
    assert np.array_equal(m_default, jds.cell_jd_estimate(u, grid))
    brute = np.min(np.stack([v for v in closed.values()]), axis=0)
    assert np.array_equal(m, brute) and np.all(m <= m_default)
    assert np.array_equal(jds.closed27_cell_estimate(u, grid, slab_cells=1)[0], m)


def test_q1_jacobian_on_a_cell_edge_is_a_quadratic():
    """Along an edge the derivative in the edge direction is constant and the other two are linear in the edge
    parameter t, so J(t) is a quadratic: the parabola through t = 0, 1/2, 1 gives J everywhere on the edge.  Hence,
    with J(0) = A, J(1) = B and a zero minimum inside, J(1/2) = (sqrt B - sqrt A)^2 / 4: a dip off the midpoint can
    leave all closed27 points high (the "off" edge-dip cell: A = .1185, J(1/2) = .1195)."""
    grid = syn.cube_grid(6)
    u = node_field(wavy_field(grid, amplitude=1.2, seed=7), grid)
    t = np.linspace(0., 1., 41)
    for cell in ((1, 1, 1), (2, 3, 1), (3, 2, 2)):
        for axis in range(3):                                      # an x, a y and a z edge of the cell
            for corner in ((0., 0.), (1., 0.), (1., 1.)):
                def point(s):
                    p = list(corner)
                    p.insert(axis, s)
                    return tuple(p)
                J = np.array([dip.cell_point_jacobian(u, grid, cell, point(s))[0] for s in t])
                parabola = np.polyfit([0., .5, 1.], [J[0], J[20], J[-1]], 2)
                assert np.max(np.abs(np.polyval(parabola, t)-J)) <= 1e-12
    grid, cell, u, _ = dip_problem(kind="off")
    A, B, half = (dip.cell_point_jacobian(u, grid, cell, (0., 0., s))[0] for s in (0., 1., .5))
    assert .1 < A < half < 1.05*A and B > 1. and abs(half-(np.sqrt(B)-np.sqrt(A))**2/4) < .002


# ---- (iii) the edge-dip cells ----------------------------------------------------------------------------------

def test_edge_dip_cell_default_misses_it_closed27_and_certificate_see_it():
    backend = NumpyBackend()
    grid, cell, u, _ = dip_problem()
    cert = backend.certify(u, grid)
    assert cert.passed
    # the cell: J at the 35 default points >= .15, J at the edge midpoint .002, the minimum inside that edge
    J_mid, _ = dip.cell_point_jacobian(u, grid, cell, dip.EDGE_DIP_POINT)
    edge = [dip.cell_point_jacobian(u, grid, cell, (0., 0., t))[0] for t in np.linspace(0., 1., 101)]
    inner = [dip.cell_point_jacobian(u, grid, cell, p)[0] for p in product(np.linspace(0., 1., 9), repeat=3)]
    assert .0019 < J_mid < .0021 and 0 < min(edge) < J_mid and 30 < int(np.argmin(edge)) < 70
    assert min(edge) <= min(inner)
    m = jds.cell_jd_estimate(u, grid)
    assert m[cell] > .15
    m27, m_default = jds.closed27_cell_estimate(u, grid)
    assert np.array_equal(m_default, m) and abs(m27[cell]-J_mid) <= 1e-12
    # the root Bernstein minimum of the cell is negative: the certificate proves it by subdivision
    K = int(np.ravel_multi_index(cell, grid.cell_shape))
    _, coefficients, _ = fallback_arrays(u.reshape(-1, 3)[_node_indices_for(grid, np.array([K]))], grid.spacing, 0.,
                                         np.geterr())
    assert coefficients.lo.min() < 0
    bounds, _ = jds.certificate_cell_bounds(u, grid, cert, backend.cert_config)
    assert 0 < bounds[cell] < min(edge) and bounds[cell] == cert.min_jd_lower_bound
    # the eight vertices of the cell: frozen by closed27 and certificate, not by the default
    vertices = [(cell[0]+z-1, cell[1]+y-1, cell[2]+x-1) for x, y, z in VERTICES]     # interior-node indices
    r = {}
    for estimate in jds.CELL_ESTIMATES:
        r[estimate], record = jds.scaling_factors(u, grid, estimate=estimate, cert=cert,
                                                  cert_config=backend.cert_config)
        if estimate != G27:
            assert record["cell_estimate"] == estimate and record["n_cells_lowered"] >= 1
            assert record["n_nodes_r_lowered"] >= 8 and record["n_r_lt_1_default"] == np.count_nonzero(r[G27] < 1)
        else:
            assert "cell_estimate" not in record and "n_nodes_r_lowered" not in record
    assert min(r[G27][v] for v in vertices) > .39
    assert max(r[C27][v] for v in vertices) < .01 and max(r[CERT][v] for v in vertices) < .01
    for estimate in (C27, CERT):                    # only lowered: r never exceeds the default's
        assert np.all(r[estimate] <= r[G27])


def test_off_midpoint_edge_dip_closed27_misses_it_certificate_sees_it():
    """Review important item 3: a dip between closed27 points.  The cell's minimum (.00096) is at (0, 0, .25) on its
    z edge, J >= .1185 at all 53 points; closed27 equals the default there, the certificate bound is below the dip."""
    backend = NumpyBackend()
    grid, cell, u, _ = dip_problem(kind="off")
    cert = backend.certify(u, grid)
    assert cert.passed
    edge = [dip.cell_point_jacobian(u, grid, cell, (0., 0., t))[0] for t in np.linspace(0., 1., 401)]
    inner = [dip.cell_point_jacobian(u, grid, cell, p)[0] for p in product(np.linspace(0., 1., 17), repeat=3)]
    assert 0 < min(edge) < .001 and int(np.argmin(edge)) == 100 and min(edge) <= min(inner)   # t = .25
    m27, m_default = jds.closed27_cell_estimate(u, grid)
    assert m27[cell] == m_default[cell] > .118                     # closed27 adds nothing in K (a neighbour: .526 ->
    assert np.count_nonzero(m27 < m_default) == 1                  # .504 at a face centre, r_a = 1 either way)
    values = jds.slab_point_jacobians(u, grid, cell[0], cell[0]+1, closed=True)
    assert min(float(v[0, cell[1], cell[2]]) for v in values.values()) > .118
    bounds, _ = jds.certificate_cell_bounds(u, grid, cert, backend.cert_config)
    assert 0 < bounds[cell] < min(edge) and bounds[cell] == cert.min_jd_lower_bound
    vertices = [(cell[0]+z-1, cell[1]+y-1, cell[2]+x-1) for x, y, z in VERTICES]
    r = {estimate: jds.scaling_factors(u, grid, estimate=estimate, cert=cert, cert_config=backend.cert_config)[0]
         for estimate in jds.CELL_ESTIMATES}
    assert np.array_equal(r[C27], r[G27]) and min(r[G27][v] for v in vertices) > .39
    assert max(r[CERT][v] for v in vertices) < .01


# ---- (iv) certificate bounds: min equals certify_map bit for bit ----------------------------------------------------

FALLBACK_ROOT_SCALE = .8889089157      # "mid" edge dip times this: root Bernstein min of K +4.8e-12, prescreen -4e-12


def path_fields():
    """(name, grid, u, path of the global minimum): one field per path of the per-cell certificate.  Found by a scan
    (JDC1): wavy fields of amplitude .3-1.8 and seeds 0-5 (only amplitude 1.8, seed 4 has its minimum on a prescreen
    cell, .0105 against .0286 for the best hull cell), and the "mid" edge dip scaled until the root Bernstein minimum
    of its cell crosses 0 (bisection; the scale above leaves it inside the prescreen's rounding allowance, ~8.8e-12)."""
    grid = aniso_grid()
    yield "hull", grid, wavy_field(grid, amplitude=.5, seed=1), "hull"
    yield "prescreen", grid, wavy_field(grid, amplitude=1.8, seed=4), "prescreen"
    g, _, u, _ = dip_problem()
    yield "fallback_root", g, FALLBACK_ROOT_SCALE*u, "fallback_root"
    yield "subdivision", g, u, "subdivision"


def certified_fields():
    grid = aniso_grid()
    yield "zero", grid, np.zeros(grid.shape+(3,))
    for amplitude, seed in ((.5, 1), (1.2, 5), (1.6, 3)):
        yield f"wavy{amplitude}", grid, wavy_field(grid, amplitude=amplitude, seed=seed)
    g, _, u, _ = dip_problem()
    yield "edge_dip", g, u
    g, _, u, _ = dip_problem(n=9)
    yield "edge_dip9", g, u
    g, _, u, _ = dip_problem(kind="off")
    yield "edge_dip_off", g, u
    for name, g, u, _ in path_fields():
        if name in ("prescreen", "fallback_root"):
            yield name, g, u


def point_jacobians_53(u, grid):
    """J at the 53 points (Gauss27 and closed27) of every cell, (53, n_cells)."""
    values = jds.slab_point_jacobians(node_field(u, grid), grid, 0, grid.cell_shape[0], closed=True)
    return np.stack([v.ravel() for v in values.values()])


@pytest.mark.parametrize("tile", [8192, 16])
def test_certificate_cell_bounds_each_path_gives_the_minimum(tile):
    """Review important item 1: for each path a field whose global minimum lies on a cell of that path; the bound of
    that cell is the value of the jd_curl function of its path, and equals certify_map's min_jd_lower_bound."""
    config = certificate_config(tile)
    for name, grid, u, path in path_fields():
        result = certify_map(u, grid, config=config)
        outcome = certify(u, grid, config)
        assert outcome.passed, name
        code, lower = cell_pass(u, grid, config)
        k = int(np.nanargmin(lower))
        assert lower[k].hex() == float(result.min_jd_lower_bound).hex() == float(outcome.min_jd_lower_bound).hex()
        assert (result.n_fast_cells, result.n_prescreen_cells, result.n_fallback_cells) == tuple(
            int(np.count_nonzero(code == c)) for c in (PROVEN_HULL, PROVEN_PRESCREEN, PROVEN_FALLBACK)), name
        margin, flag = parallel_margins(np.ascontiguousarray(u), grid, 0.)
        table = CellQ1Operators3D(grid, location="closed27_uniform").coefficients
        pre = float(prescreen_lower_bounds(u, grid, table, np.array([k]))[0])
        _, coefficients, _ = fallback_arrays(u.reshape(-1, 3)[_node_indices_for(grid, np.array([k]))], grid.spacing,
                                             0., np.geterr())
        root = float(coefficients.lo.min())
        if path == "hull":
            assert code[k] == PROVEN_HULL and lower[k] == margin[k] > 0 and not flag.any(), name
        else:
            assert margin[k] <= 0 or flag[k], name                 # the hull did not prove it
        if path == "prescreen":
            assert code[k] == PROVEN_PRESCREEN and lower[k] == pre > 0, name
        if path in ("fallback_root", "subdivision"):
            assert code[k] == PROVEN_FALLBACK and not pre > 0, name
        if path == "fallback_root":
            assert lower[k] == root > 0, name                       # proven at the root box
        if path == "subdivision":
            assert root < 0 < lower[k], name                        # proven only after subdivision


@pytest.mark.parametrize("tile", [8192, 16])
def test_certificate_cell_bounds_minimum_equals_certify_map(tile):
    config = certificate_config(tile)
    paths = Counter()
    seen = 0
    for name, grid, u in certified_fields():
        result = certify_map(u, grid, config=config)
        outcome = certify(u, grid, config)
        if not outcome.passed:
            continue
        seen += 1
        jds.clear_cell_bounds_cache()
        bounds, record = jds.certificate_cell_bounds(u, grid, outcome, config)
        code, lower = cell_pass(u, grid, config)
        assert np.array_equal(bounds.ravel(), lower, equal_nan=True)
        lowest = float(np.nanmin(bounds))
        assert lowest.hex() == float(result.min_jd_lower_bound).hex() == float(outcome.min_jd_lower_bound).hex(), name
        assert record["cert_min_matches"] and record["min_cert_bound"] == lowest and not record["cell_bounds_cached"]
        assert record["cert_state_hash"] == outcome.state_hash
        assert not bounds.flags.writeable and bounds.shape == grid.cell_shape
        argmin = int(np.nanargmin(bounds))
        paths[int(code[argmin])] += 1                        # which path gave the global minimum
        assert set(np.unique(code)) <= {PROVEN_HULL, PROVEN_PRESCREEN, PROVEN_FALLBACK}
        # review suggestion 3: every bound is a positive lower bound of J at the 53 points of its cell
        assert np.all(np.isfinite(bounds)) and np.all(bounds > 0), name
        assert np.all(bounds.ravel() <= point_jacobians_53(u, grid).min(axis=0)), name
    assert seen >= 8
    assert paths[PROVEN_HULL] >= 1 and paths[PROVEN_PRESCREEN] >= 1 and paths[PROVEN_FALLBACK] >= 2


def test_certificate_cell_bounds_prescreen_path_and_cache():
    backend = NumpyBackend()
    grid, _, u, _ = dip_problem()
    cert = backend.certify(u, grid)
    code, _ = cell_pass(u, grid, backend.cert_config)
    assert set(np.unique(code)) == {PROVEN_HULL, PROVEN_PRESCREEN, PROVEN_FALLBACK}
    first, rec1 = jds.certificate_cell_bounds(u, grid, cert, backend.cert_config)
    second, rec2 = jds.certificate_cell_bounds(u.copy(), grid, cert, backend.cert_config)
    assert second is first and rec2["cell_bounds_cached"] and not rec1["cell_bounds_cached"]
    other = u.copy()
    other[1, 1, 1] += (.01, 0., 0.)
    cert2 = backend.certify(other, grid)
    third, rec3 = jds.certificate_cell_bounds(other, grid, cert2, backend.cert_config)
    assert third is not first and not rec3["cell_bounds_cached"]
    # another tile size is another cache key
    config16 = certificate_config(16)
    fourth, rec4 = jds.certificate_cell_bounds(other, grid, certify(other, grid, config16), config16)
    assert not rec4["cell_bounds_cached"]


def test_certificate_cell_bounds_misuse_raises():
    backend = NumpyBackend()
    grid, _, u, _ = dip_problem()
    cert = backend.certify(u, grid)
    with pytest.raises(InputValidationError):
        jds.certificate_cell_bounds(u, grid, None, backend.cert_config)
    with pytest.raises(InputValidationError):
        jds.certificate_cell_bounds(u, grid, replace(cert, passed=False), backend.cert_config)
    with pytest.raises(InputValidationError):
        jds.certificate_cell_bounds(u, grid, cert, None)
    with pytest.raises(InputValidationError):
        jds.certificate_cell_bounds(u, grid, cert, CertificateConfig())               # not the C07-2 configuration
    with pytest.raises(CertificateContractError):                                    # another state's certificate
        jds.certificate_cell_bounds(np.zeros_like(u), grid, cert, backend.cert_config)
    tampered = replace(cert, min_jd_lower_bound=float(np.nextafter(cert.min_jd_lower_bound, 1.)))
    with pytest.raises(CertificateContractError):
        jds.certificate_cell_bounds(u, grid, tampered, backend.cert_config)
    jds.certificate_cell_bounds(u, grid, cert, backend.cert_config)
    with pytest.raises(CertificateContractError):                                    # also on a cache hit
        jds.certificate_cell_bounds(u, grid, tampered, backend.cert_config)
    # review suggestion 2: a certificate of another configuration version (the version string does not depend on T)
    assert cert.certificate_version == certificate_version(backend.cert_config) == certificate_version(
        certificate_config(16))
    other_version = replace(cert, certificate_version=cert.certificate_version+"-other")
    with pytest.raises(CertificateContractError):                                    # also on a cache hit
        jds.certificate_cell_bounds(u, grid, other_version, backend.cert_config)
    jds.clear_cell_bounds_cache()
    with pytest.raises(CertificateContractError):
        jds.certificate_cell_bounds(u, grid, other_version, backend.cert_config)
    with pytest.raises(InputValidationError):
        JDScaledMetric(Identity(), grid, u, config=JDScalingConfig(cell_estimate=CERT), cert=cert)
    with pytest.raises(InputValidationError):
        JDScaledMetric(Identity(), grid, u, config=JDScalingConfig(cell_estimate=CERT),
                       cert_config=backend.cert_config)


# ---- (v) synthetic problems, per option ------------------------------------------------------------------------

BLOCKING = {"JACOBIAN_VIOLATION", "FEASIBILITY_UNCERTAIN"}


def test_quadratic_mid_dip_default_and_closed27_blocked_certificate_descends():
    """The "mid" edge dip on the quadratic: the default is blocked early; closed27 goes further but is blocked too
    (measured: 29 accepted steps of 30, then line_search_failed, the last line search rejected by the certificate);
    certificate uses up the 30 steps."""
    results = {}
    for estimate in jds.CELL_ESTIMATES:
        result, rows, metric = run_quadratic(estimate)
        accepted = [r for r in rows if r["kind"] == "candidate" and r["accepted"]]
        assert all(r["cert_status"] == "certified_feasible" for r in accepted)
        results[estimate] = (result, metric)
    default, _ = results[G27]
    assert default.status == "line_search_failed" and default.accepted < 20
    assert default.last_iteration == [("JACOBIAN_VIOLATION", "JACOBIAN_VIOLATION")]*26
    closed, metric = results[C27]
    assert closed.status == "line_search_failed" and default.accepted+10 < closed.accepted < 30
    assert len(closed.last_iteration) == 26 and {a for a, _ in closed.last_iteration} <= BLOCKING
    assert closed.energies[-1] < .5*default.energies[-1]
    assert all(c["cell_estimate"] == C27 for c in metric.computations)
    certificate, metric = results[CERT]
    assert certificate.status == "max_iterations" and certificate.accepted == 30
    assert certificate.energies[-1] < closed.energies[-1] < .5*default.energies[-1]
    assert all(c["cert_min_matches"] and c["cell_estimate"] == CERT for c in metric.computations)


@pytest.mark.parametrize("push", ["gradient", "constant"])
def test_quadratic_off_dip_closed27_is_the_default_certificate_descends(push):
    """The "off" edge dip (minimum at zeta = .25 on an edge): closed27 does not see it, so it runs the default's
    trajectory (same S~, same steps) and is blocked with it; certificate uses up the 30 steps."""
    # JDTI: the three options of JDC by name (CELL_ESTIMATES also holds certificate_bernstein since JDTI; its run on
    # this problem is in test_jd_cell_estimate_bernstein.py)
    out = {estimate: run_quadratic(estimate, push=push, kind="off") for estimate in (G27, C27, CERT)}
    (default, _, dm), (closed, _, cm), (certificate, _, tm) = (out[e] for e in (G27, C27, CERT))
    assert default.status == closed.status == "line_search_failed" and default.accepted == closed.accepted < 20
    assert closed.energies == default.energies and closed.last_iteration == default.last_iteration
    assert {a for a, _ in default.last_iteration} == {"JACOBIAN_VIOLATION"}
    assert [c["min_r"] for c in cm.computations] == [c["min_r"] for c in dm.computations]
    assert all(c["n_nodes_r_lowered"] == 0 for c in cm.computations)
    assert certificate.status == "max_iterations" and certificate.accepted == 30
    assert certificate.energies[-1] < .5*default.energies[-1]


def fb_dip_problem(n=8):
    """F_B (one intensity channel) with fixed = moving o (x + u*), u* = u0 + w folding the edge-dip cell."""
    grid, cell, u0, w = dip_problem(n)
    moving = syn.moving_image(grid.identity())
    fixed = syn.warp_with_spline(moving, grid, u0+w)
    state = ModelState(grid=grid, omega0=grid.omega, fixed=fixed[None], samplers=build_samplers(moving[None], grid),
                       mask=syn.brain_box(grid, 1), sigma2=np.array([1e-2]), alpha=1e-3, nu=1e-3, tau2=np.full(4, 1.),
                       tissue=syn.uniform_tissue(grid), logvol_points="gauss27")       # DEF: explicit C05-2
    return grid, state, u0


def test_fb_synthetic_default_blocked_new_estimates_descend():
    backend = NumpyBackend()
    grid, state, u0 = fb_dip_problem()
    start = Commit(u0, restrict_free(u0, grid), backend.certify(u0, grid))
    target = threshold(g0_of(backend, state))
    out = {}
    for estimate in jds.CELL_ESTIMATES:
        jds.clear_cell_bounds_cache()
        rows = []
        out[estimate] = solve_direct(backend, state, start, target=target, caps=Caps(max_accepted=30), log=rows.append,
                                     pg=PGConfig(jd=JDScalingConfig(cell_estimate=estimate)))
        record = out[estimate].metric["jd_scaling"]
        assert record["cell_estimate"] == estimate
        first = record["computations"][0]
        assert first["at"] == "round_start"
        if estimate == CERT:
            assert first["cert_state_hash"] == start.cert.state_hash
    default = out[G27]
    assert default.status == "line_search_failed" and default.accepted < 30
    for estimate in (C27, CERT):
        assert out[estimate].status == "cap_reached" and out[estimate].accepted == 30
        assert out[estimate].energies[-1] < .5*default.energies[-1]
        assert np.all(np.diff(out[estimate].energies) < 0)


# ---- (vi) wiring ---------------------------------------------------------------------------------------------

def test_refresh_uses_the_accepted_base_certificate():
    rows = []
    result, rows, metric = run_quadratic(CERT, rows=rows)
    accepts = {r["iteration"]: r for r in rows if r["kind"] == "accept"}
    refreshes = [r for r in rows if r["kind"] == "jd_scaling" and r.get("at") == "refresh"]
    assert refreshes, "the refresh rule did not fire"
    for r in refreshes:
        assert r["cert_state_hash"] == accepts[r["iteration"]-1]["state_hash"]
        assert r["min_cert_bound"] == accepts[r["iteration"]-1]["min_jd_lower_bound"]


def test_option_b_reference_refresh_gets_the_commit_certificate():
    pg = PGConfig(jd=JDScalingConfig(cell_estimate=CERT))
    result, rows = run_outer("b", pg, steps=10, max_outer=3)
    computations = result.metric["jd_scaling"]["computations"]
    references = [c for c in computations if c["at"] == "reference_refresh"]
    commits = {0: None}
    commits.update({r["outer"]: r["state_hash"] for r in rows if r["kind"] == "commit"})
    assert len(references) >= 2
    for c in references:
        if c["outer"] > 0:
            assert c["cert_state_hash"] == commits[c["outer"]]
        assert c["cert_min_matches"] and c["cell_estimate"] == CERT


def test_closed27_and_certificate_need_affine_scaling_and_cli_switch():
    """DEF (C09.1-3a last item): with metric_jd_scaling "off" an explicit cell estimate -- any value, the old default
    gauss27+corners included -- is refused; unset, it is recorded as "none".  Default certificate."""
    for estimate in jds.CELL_ESTIMATES:
        with pytest.raises(InputValidationError):
            PGConfig(metric_jd_scaling="off", jd=JDScalingConfig(cell_estimate=estimate))
        assert PGConfig(jd=JDScalingConfig(cell_estimate=estimate)).record()["jd_scaling"]["cell_estimate"] == estimate
    off = PGConfig(metric_jd_scaling="off")
    assert off.jd is None and off.jd_cell_estimate == "none" and off.record()["jd_cell_estimate"] == "none"
    assert PGConfig().jd == JDScalingConfig() and PGConfig().jd_cell_estimate == CERT
    with pytest.raises(InputValidationError):
        JDScalingConfig(cell_estimate="closed64")
    base = ["--pair", "s/p", "--channel-source", "none", "--out", "x"]
    assert run_pair.parse_args(base).jd_cell_estimate == CERT
    for estimate in jds.CELL_ESTIMATES:
        assert run_pair.parse_args(base+["--jd-cell-estimate", estimate]).jd_cell_estimate == estimate
        with pytest.raises(SystemExit):
            run_pair.parse_args(base+["--metric-jd-scaling", "off", "--jd-cell-estimate", estimate])
    with pytest.raises(SystemExit):
        run_pair.parse_args(base+["--jd-cell-estimate", "closed64"])
    assert run_pair.parse_args(base+["--metric-jd-scaling", "off"]).jd_cell_estimate == "none"


# ---- (vii) torch64 backend (review important item 2) ---------------------------------------------------------------

TORCH_CASES = ("fb_dip_a", "near_fold_b")       # F_B edge dip, option (a); near-fold, option (b) with S~ refreshes
INT_KEYS = ("n_r_lt_1", "n_nodes", "n_nodes_r_lowered", "n_r_lt_1_default", "n_cells_lowered", "n_cells_without_bound")
FLOAT_KEYS = ("min_J_a", "min_r", "fraction_r_lt_1", "min_m_K_default", "min_cert_bound")


def solve_case(backend, case, estimate):
    """solve_direct (fb_dip_a, 30 accepted steps) or solve_proximal (near_fold_b, 25 steps, 4 outer rounds) with the
    jd_cell_estimate `estimate`; (RoundResult, rows)."""
    jds.clear_cell_bounds_cache()
    pg = PGConfig(jd=JDScalingConfig(cell_estimate=estimate))
    rows = []
    if case == "fb_dip_a":
        grid, state, u0 = fb_dip_problem()
        state = backend.prepare(state)
        start = Commit(u0, restrict_free(u0, grid), backend.certify(u0, grid))
        target = threshold(g0_of(backend, state))
        return solve_direct(backend, state, start, target=target, caps=Caps(max_accepted=30), log=rows.append,
                            pg=pg), rows
    grid, state = near_fold_problem()
    state = backend.prepare(state)
    zero = np.zeros(grid.shape+(3,))
    start = Commit(zero, restrict_free(zero, grid), backend.certify(zero, grid))
    target = threshold(g0_of(backend, state))
    return solve_proximal(backend, state, start, target=target, caps=Caps(max_accepted=25, max_outer=4),
                          params=ProximalParams(lam=default_lambda(grid)), log=rows.append, pg=pg), rows


def compare_with_numpy(device, case, estimate, monkeypatch):
    pytest.importorskip("torch")
    from bayesreg.solve import Torch64Backend
    scales = []
    original = JDScaledMetric._compute

    def spy(self, u, **context):                        # every S~ as computed (free layout)
        record = original(self, u, **context)
        scales.append(self.scale.copy())
        return record
    monkeypatch.setattr(JDScaledMetric, "_compute", spy)
    runs = {}
    for name, backend in (("numpy", NumpyBackend()), ("torch64", Torch64Backend(device=device))):
        scales.clear()
        result, rows = solve_case(backend, case, estimate)
        runs[name] = (result, rows, list(scales), backend)
    (a, rows_a, s_a, _), (b, rows_b, s_b, torch_backend) = runs["numpy"], runs["torch64"]
    assert torch_backend.device == device and torch_backend.record()["model_module"] == "bayesreg.backends.torch64"
    # the step sequence
    decisions = [[(r.get("outer"), r["iteration"], r["attempt"], r["accepted"], r["reason"]) for r in rows
                  if r["kind"] == "candidate"] for rows in (rows_a, rows_b)]
    assert decisions[0] == decisions[1] and decisions[0]
    assert (a.status, a.accepted) == (b.status, b.accepted)
    tau = [np.array([r["tau"] for r in rows if r["kind"] == "candidate"]) for rows in (rows_a, rows_b)]
    assert np.allclose(tau[0], tau[1], rtol=1e-6, atol=0.)
    assert np.allclose(a.energies, b.energies, rtol=1e-10, atol=0.)
    assert np.max(np.abs(a.commit.u-b.commit.u)) < 1e-9
    # S~: the freeze computation bit for bit (the same host u), the later ones to rounding
    records = [r.metric["jd_scaling"]["computations"] for r in (a, b)]
    assert len(s_a) == len(s_b) == len(records[0]) == len(records[1]) >= 1
    assert np.array_equal(s_a[0], s_b[0])
    assert all(np.max(np.abs(x-y)) <= 1e-9 for x, y in zip(s_a, s_b))
    for x, y in zip(*records):
        assert all(x.get(key) == y.get(key) for key in ("at", "cell_estimate", "trigger"))
        assert all(x.get(key) == y.get(key) for key in INT_KEYS)
        assert all(abs(x[key]-y[key]) <= 1e-9*max(1., abs(x[key])) for key in FLOAT_KEYS if key in x)
    if estimate == CERT:
        assert records[0][0]["cert_state_hash"] == records[1][0]["cert_state_hash"]
        assert all(c["cert_min_matches"] for c in records[0]+records[1])
    return runs


@pytest.mark.parametrize("case", TORCH_CASES)
@pytest.mark.parametrize("estimate", jds.CELL_ESTIMATES)
def test_torch64_cpu_matches_numpy(estimate, case, monkeypatch):
    runs = compare_with_numpy("cpu", case, estimate, monkeypatch)
    result = runs["torch64"][0]
    if case == "near_fold_b":                         # the refresh rule fired on both backends
        assert {c["at"] for c in result.metric["jd_scaling"]["computations"]} == {"reference_refresh", "refresh"}
    elif estimate != G27:
        assert result.status == "cap_reached" and result.accepted == 30


@pytest.mark.skipif(os.environ.get("JDC_TEST_CUDA") != "1",
                    reason="cuda only on request (JDC_TEST_CUDA=1, GPU idle; the lead runs it)")
@pytest.mark.parametrize("case", TORCH_CASES)
@pytest.mark.parametrize("estimate", jds.CELL_ESTIMATES)
def test_torch64_cuda_matches_numpy(estimate, case, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    compare_with_numpy("cuda", case, estimate, monkeypatch)
