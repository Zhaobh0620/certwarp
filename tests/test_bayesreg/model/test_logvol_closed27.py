"""C05-2b: the log-volume prior on Gauss27 U closed27 (BZ adopted (A') 10-30; work package IMPL27), numpy reference.

(i)   the option: validation, exclusivity with C05-2a, model identity, provenance; the default "gauss27" unchanged
      (bitwise: the golden digests of test_logvol_corners / torch64 / solve, and LogVolume reads ops.weights itself);
      the closed27 tables are jd_curl's closed27_uniform points / basis / coefficients (corner rows one-hot); Simpson
      weights (D48) and the shares theta_G = 1/(1 + beta_S), theta_S = beta_S/(1 + beta_S); the floating-point
      shares and weights of AC27b (theta_G = 1.0/(1.0 + beta_S), theta_S = beta_S/(1.0 + beta_S), w_G = theta_G *
      ops.weights, w_S = theta_S * s27, bitwise; recorded thetas are the multipliers used) for beta_S = .3, .5, .7, 1,
      2, 3 (IMPL27b, review IMPL27 item 1).
(ii)  tau^2 at the closed27 points: the corners equal the node values bitwise, the other points sum_k tau_k^2 (Q1
      interpolation of Ptilde^k) to 1e-14; edge midpoints the mean of two node values.
(iii) derivatives of both parts (Gauss part with theta_G, closed27 part): central differences <= 1e-7, adjoint
      <= 1e-12; an independent dense oracle of both parts (residuals, jvp, energy, gradient, GN HVP <= 1e-12); HVP
      symmetric and positive semidefinite; energy directional derivatives.
(iv)  C06-7 order (per tile: Gauss sum then closed27 sum, bitwise), thread count bitwise neutral, tile size only
      rounding; the logvol_parts record (parts, minimum J over the closed27 points and its cell).
(v)   C08 stable comparison with the closed27 term against 110-digit Decimal; per-tile composition bitwise.
(vi)  B13'-2: u = 0, tau^2 constant per cell: the prior's GN Hessian equals that of C05-2 (1e-12) for Simpson weights
      (beta_S = 1, 3); the uniform rule does not (ablation).  Affine field with constant tau: Pi_J equals C05-2's.
      B13'-(2b) (IMPL27b, review IMPL27 item 3): u = 0, tau^2 the Q1 interpolation of multi-class node values
      (non-constant in tissue-boundary cells): the log-volume quadratic form lies between (1/max tau^2) and
      (1/min tau^2) times int (div v)^2 (dense, eigenvalue tolerance 1e-14 relative); it differs from C05-2's.
(vii) W1 family (edge-midpoint dip, B3'-2): the C05-2 and C05-2a penalties stay bounded, C05-2b's grows without bound;
      beyond b* a floating J <= 0 only at a closed27 point is refused (NONPOSITIVE_GAUSS_JD, message names closed27).
      W2 (53 points positive, edge interior negative): C05-2b's energy is finite (the certificate side is in
      tests/test_bayesreg/solve/test_witness.py).
(viii) Laplace (C14-7) and EB (C15-10) accept the Simpson rule and refuse the uniform rule (scope limit) and C05-2a
      (the closed27 blocks themselves: tests/test_bayesreg/infer/test_laplace_closed27.py,
      tests/test_bayesreg/learn/test_eb_closed27_dense.py).
Small grids, no data/, no certificate, no GPU.
DEF (2026-10-31): gauss27+closed27 (beta_S 1, Simpson) is the ModelState default since; "default" in this file means
the C05-2 model "gauss27" (the default before DEF), now built explicitly (_gauss27.make_state, D.pre_def_model()); the
closed27 states are built from it with replace(logvol_points="gauss27+closed27", ...) as before.
"""
from decimal import Decimal, localcontext
from itertools import product
import json

import numpy as np
import pytest

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError
from jd_curl.three_d.boundary import inject_free
from jd_curl.three_d.cell_q1 import CellQ1Operators3D, cell_tiles
from jd_curl.three_d.grid import make_grid_3d

from bayesreg.geom import gauss_geometry, node_field, simpson27_weights, tile_geometry, tile_interpolate
from bayesreg.model import (BLOCK_NAMES, LOGVOL_POINTS, Evaluation, LogVolume, LogVolumeClosed27, evaluate,
                            gauss_jvp, gauss_value, gauss_vjp, gradient, hvp, logvol_extra_block, stable_reduction)
from bayesreg.model.energy import _logvol_change_sum

from . import _logvol_digest as D
from ._gauss27 import make_state
from ._helpers import (aniso, cofactor, dense_model, dense_operators, feasible_field, free_random, relmax, selection,
                       theta_of)
from .test_energy import decimal_half_squares, relative

LD = np.longdouble
CLOSED = "gauss27+closed27"
SIGMA = (1/6, 2/3, 1/6)


def closed_setup(seed=0, student_t=True, share=1., weights="simpson", **kwargs):
    g = aniso()
    state = make_state(g, seed=seed, student_t=student_t, **kwargs).replace(
        logvol_points=CLOSED, logvol_closed27_share=share, closed27_weights=weights)
    rng = np.random.default_rng(seed+100)
    return g, state, rng, feasible_field(g, rng)


def thetas(state):
    beta = state.logvol_closed27_share
    return 1/(1+beta), beta/(1+beta)


# ----------------------------------------------------------------------------------------------
# Independent dense oracle of the closed27 rows
# ----------------------------------------------------------------------------------------------

def dense_closed27_operators(grid, rule="simpson"):
    """D (3, 27|K|, N), B (27|K|, N) and w (27|K|,) = omega s_p (Simpson) or omega/27 at the points {0, 1/2, 1}^3
    (z, y, x product, x fastest; cells in C order), from the scalar Q1 formula (not CellQ1Operators3D)."""
    nodes = (0., .5, 1.)
    points = [(px, py, pz) for pz, py, px in product(nodes, repeat=3)]
    one_d = SIGMA if rule == "simpson" else (1/3, 1/3, 1/3)
    pw = np.array([a*b*c for c, b, a in product(one_d, repeat=3)])
    P, N = len(points), int(np.prod(grid.shape))
    D = np.zeros((3, grid.n_cells*P, N))
    B = np.zeros((grid.n_cells*P, N))
    for c, (cz, cy, cx) in enumerate(np.ndindex(grid.cell_shape)):
        for vz, vy, vx in product((0, 1), repeat=3):
            node = np.ravel_multi_index((cz+vz, cy+vy, cx+vx), grid.shape)
            vertex = (vx, vy, vz)
            for k, point in enumerate(points):
                row = c*P+k
                B[row, node] = np.prod([point[t] if vertex[t] else 1-point[t] for t in range(3)])
                for j in range(3):
                    value = (2*vertex[j]-1)/grid.spacing[j]
                    for t in range(3):
                        if t != j:
                            value *= point[t] if vertex[t] else 1-point[t]
                    D[j, row, node] = value
    return D, B, np.tile(grid.omega*pw, grid.n_cells)


def dense_closed27(state, u):
    """(r, K, J, tau2) of the closed27 part of r_J: r = sqrt(theta_S omega s_p/(omega0 tau^2)) log J, rows (cell,
    point); tau^2 = sum_k tau_k^2 (B Ptilde^k); K (27|K|, 3 N_int) explicit."""
    g = state.grid
    D, B, w = dense_closed27_operators(g, state.closed27_weights)
    S = selection(g)
    U = np.asarray(u).reshape(-1, 3)
    F = np.einsum("jqn,nr->qrj", D, U)+np.eye(3)
    A, J = cofactor(F), np.linalg.det(F)
    tau2 = sum(t*(B@state.tissue[k].ravel()) for k, t in enumerate(state.tau2))
    s = np.sqrt(thetas(state)[1]*w/(state.omega0*tau2))
    DS = [D[j]@S for j in range(3)]
    L = np.concatenate([sum(A[:, r, j, None]*DS[j] for j in range(3)) for r in range(3)], axis=1)
    return s*np.log(J), (s/J)[:, None]*L, J, tau2


def dense_blocks(state, u):
    """All residual blocks of the dense oracle with the C05-2b log-volume block [sqrt(theta_G) r_Gauss; r_closed27]."""
    dense = dense_model(state, u)
    r_s, K_s, _, _ = dense_closed27(state, u)
    root = np.sqrt(thetas(state)[0])
    r_g, K_g = root*dense["logvol"][0], root*dense["logvol"][1]
    return dict(dense, logvol=(np.concatenate((r_g, r_s)), np.concatenate((K_g, K_s)))), (r_g, K_g), (r_s, K_s)


# ----------------------------------------------------------------------------------------------
# (i) the option
# ----------------------------------------------------------------------------------------------

def test_option_validation_identity_and_record():
    """Validation and exclusivity; the default state's Gauss weights are ops.weights itself; the closed27 tables are
    jd_curl's closed27_uniform tables (corner rows one-hot); Simpson weights and shares as D48; model identity and
    provenance record the option."""
    from bayesreg.backends.torch64.stable import model_identity
    state, _ = D.small_case(**D.pre_def_model())          # DEF: "gauss27" explicit
    assert LOGVOL_POINTS == ("gauss27", "gauss27+corners", CLOSED)
    assert state.logvol_gauss_weights is state.ops.weights and state.closed27_ops is None
    assert logvol_extra_block(state) is None and state.logvol_extra is None and not state.logvol_closed27
    for bad in (dict(logvol_closed27_share=2.), dict(closed27_weights="uniform"),
                dict(logvol_points="gauss27+corners", logvol_closed27_share=2.),
                dict(logvol_points="gauss27+corners", closed27_weights="uniform"),
                dict(logvol_points=CLOSED, logvol_closed27_share=0.), dict(logvol_points=CLOSED,
                                                                            logvol_closed27_share=-1.),
                dict(logvol_points=CLOSED, logvol_closed27_share=np.inf),
                dict(logvol_points=CLOSED, closed27_weights="gauss"), dict(logvol_points=CLOSED, corner_weight=2.),
                dict(logvol_points="closed27")):
        with pytest.raises(InputValidationError):
            state.replace(**bad)
    g = state.grid
    closed = state.replace(logvol_points=CLOSED)
    assert closed.logvol_closed27 and closed.logvol_extra == "closed27" and logvol_extra_block(closed) is LogVolumeClosed27
    reference = CellQ1Operators3D(g, location="closed27_uniform")
    ops = closed.closed27_ops
    for name in ("points", "basis", "coefficients"):
        assert np.array_equal(getattr(ops, name), getattr(reference, name))
    corner_rows = [k for k, p in enumerate(ops.points) if set(p.tolist()) <= {0., 1.}]
    assert len(corner_rows) == 8 and all(sorted(ops.basis[k].tolist()) == [0.]*7+[1.] for k in corner_rows)
    simpson = simpson27_weights(g)
    assert abs(simpson.sum()-g.omega) <= 1e-15*g.omega
    kinds = {}
    for p, w in zip(ops.points, simpson):
        kinds.setdefault(int(np.sum((p == 0) | (p == 1))), set()).add(round(w/g.omega*216, 12))
    assert kinds == {3: {1.}, 2: {4.}, 1: {16.}, 0: {64.}}
    assert np.array_equal(closed.closed27_point_weights, simpson/2)                # beta_S = 1: exact halves
    assert np.array_equal(closed.logvol_gauss_weights, state.ops.weights/2)
    three = state.replace(logvol_points=CLOSED, logvol_closed27_share=3.)
    assert np.array_equal(three.closed27_point_weights, .75*simpson)
    assert np.array_equal(three.logvol_gauss_weights, state.ops.weights/4.)
    uniform = state.replace(logvol_points=CLOSED, closed27_weights="uniform")
    assert np.array_equal(uniform.closed27_point_weights, .5*reference.weights)
    identities = {model_identity(s) for s in (state, closed, three, uniform, state.replace(
        logvol_points="gauss27+corners"))}
    assert len(identities) == 5
    record = three.provenance()
    assert record["logvol_points"] == CLOSED and record["logvol_closed27_share"] == 3.
    assert record["closed27_weights"] == "simpson" and record["theta_G"] == .25 and record["theta_S"] == .75
    assert "logvol_closed27_share" not in state.provenance() and state.provenance()["logvol_points"] == "gauss27"
    assert state.logvol_theta is None
    assert "scope limit" in uniform.provenance()["closed27_rule"] and "do not hold" not in \
        uniform.provenance()["closed27_rule"]
    with pytest.raises(InputValidationError):
        LogVolumeClosed27.operators(state)


@pytest.mark.parametrize("rule", ["simpson", "uniform"])
def test_share_floating_point_formula(rule, record_property):
    """AC27b (review IMPL27 item 1): theta_G = 1.0/(1.0 + beta_S), theta_S = beta_S/(1.0 + beta_S), one division each
    (theta_S is not 1 - theta_G); w_G = theta_G * ops.weights and w_S = theta_S * s27 bitwise; the recorded thetas are
    the multipliers used; for beta_S = 1, 3 w_G is exactly ops.weights times a power of 2.  The IMPL27 formula
    (ops.weights / (1 + beta_S), (beta_S * W27) / (1 + beta_S)) differs from it by 1 ulp in some element for some
    beta_S (power of the test); either way the relative difference is at most 2 ulp."""
    state, _ = D.small_case(**D.pre_def_model())          # DEF: "gauss27" explicit
    g = state.grid
    W27 = simpson27_weights(g) if rule == "simpson" else CellQ1Operators3D(g, location="closed27_uniform").weights
    differs = {}
    for beta in (.3, .5, .7, 1., 2., 3.):
        s = state.replace(logvol_points=CLOSED, logvol_closed27_share=beta, closed27_weights=rule)
        theta_G, theta_S = 1.0/(1.0+beta), beta/(1.0+beta)
        assert s.logvol_theta == (theta_G, theta_S)
        assert np.array_equal(s.logvol_gauss_weights, theta_G*state.ops.weights)
        assert np.array_equal(s.closed27_point_weights, theta_S*W27)
        record = s.provenance()
        assert (record["theta_G"], record["theta_S"]) == (theta_G, theta_S)
        old_g, old_s = state.ops.weights/(1+beta), (beta*W27)/(1+beta)
        differs[beta] = int(np.count_nonzero(old_g != s.logvol_gauss_weights)
                            + np.count_nonzero(old_s != s.closed27_point_weights))
        assert np.max(np.abs(old_g-s.logvol_gauss_weights)/old_g) <= 2*np.finfo(float).eps
        assert np.max(np.abs(old_s-s.closed27_point_weights)/old_s) <= 2*np.finfo(float).eps
        if beta in (1., 3.):
            assert np.array_equal(s.logvol_gauss_weights, state.ops.weights*{1.: .5, 3.: .25}[beta])
            assert differs[beta] == 0
    record_property("elements_differing_from_the_impl27_formula", json.dumps(differs))
    if rule == "simpson":
        assert sum(differs.values()) > 0, differs


# ----------------------------------------------------------------------------------------------
# (ii) tau^2 at the closed27 points
# ----------------------------------------------------------------------------------------------

def test_closed27_tau2():
    g, state, _, _ = closed_setup()
    tile = slice(0, g.n_cells)
    tau2 = LogVolumeClosed27.tau2(state, tile)
    ops = state.closed27_ops
    vertices = np.array([(x, y, z) for z, y, x in product((0, 1), repeat=3)], dtype=float)
    from jd_curl.three_d.cell_q1 import cell_node_indices
    nodes = state.tau2_nodes.ravel()[cell_node_indices(g, tile)]                # (cells, 8) VERTICES order
    for k, p in enumerate(ops.points):
        if set(p.tolist()) <= {0., 1.}:
            v = int(np.flatnonzero(np.all(vertices == p, axis=1))[0])
            assert np.array_equal(tau2[:, k], nodes[:, v])                       # corners: node values, bitwise
    direct = sum(t*tile_interpolate(np.ascontiguousarray(state.tissue[c]), ops, tile=tile)
                 for c, t in enumerate(state.tau2))
    assert np.max(np.abs(tau2-direct)) <= 1e-14*np.max(direct)
    mid = ops.points.tolist().index([0., 1., .5])                             # edge (x, y) = (0, 1), z midpoint
    a = int(np.flatnonzero(np.all(vertices == (0, 1, 0), axis=1))[0])
    b = int(np.flatnonzero(np.all(vertices == (0, 1, 1), axis=1))[0])
    assert np.max(np.abs(tau2[:, mid]-(nodes[:, a]+nodes[:, b])/2)) <= 1e-15
    assert tau2.min() >= state.tau2.min()-1e-15 and tau2.max() <= state.tau2.max()+1e-15


# ----------------------------------------------------------------------------------------------
# (iii) derivatives and the dense oracle
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("block", [LogVolume, LogVolumeClosed27], ids=["gauss_part", "closed27_part"])
def test_jvp_central_difference_and_adjoint(block, record_property):
    g, state, rng, u = closed_setup(seed=1, share=3.)
    errors = {}
    for trial in range(3):
        v = inject_free(free_random(g, rng), g)
        v *= np.max(np.abs(u))/np.max(np.abs(v))
        t = 1e-4*np.max(np.abs(u))/np.max(np.abs(v))
        fd = (gauss_value(block, state, u+t*v)-gauss_value(block, state, u-t*v))/(2*t)
        kv = gauss_jvp(block, state, u, v)
        errors[f"jvp{trial}"] = float(np.max(np.abs(fd-kv))/max(1., np.max(np.abs(kv))))
        r = rng.normal(size=kv.shape)
        lhs, rhs = float(np.sum(kv*r)), float(theta_of(v, g)@gauss_vjp(block, state, u, r))
        errors[f"adjoint{trial}"] = abs(lhs-rhs)/(np.linalg.norm(kv)*np.linalg.norm(r))
    record_property("errors", json.dumps(errors))
    assert max(v for k, v in errors.items() if k.startswith("jvp")) <= 1e-7, errors
    assert max(v for k, v in errors.items() if k.startswith("adjoint")) <= 1e-12, errors


@pytest.mark.parametrize("share,rule", [(1., "simpson"), (3., "simpson"), (1., "uniform")])
def test_dense_oracle_energy_gradient_hvp(share, rule, record_property):
    """Both parts against an independent dense assembly: residuals and Jacobians (1e-12); the log-volume energy is
    1/2 (|r_G|^2 + |r_S|^2); gradient and GN HVP of F_B and of the log-volume block (1e-12); H symmetric, PSD."""
    g, state, rng, u = closed_setup(seed=2, share=share, weights=rule)
    ev = evaluate(state, u)
    blocks, (r_g, K_g), (r_s, K_s) = dense_blocks(state, u)
    v, w = free_random(g, rng), free_random(g, rng)
    errors = dict(residual_gauss=relmax(gauss_value(LogVolume, state, u).ravel(), r_g),
                  jvp_gauss=relmax(gauss_jvp(LogVolume, state, u, v).ravel(), K_g@v),
                  residual_closed27=relmax(gauss_value(LogVolumeClosed27, state, u).ravel(), r_s),
                  jvp_closed27=relmax(gauss_jvp(LogVolumeClosed27, state, u, v).ravel(), K_s@v))
    r_J, K_J = blocks["logvol"]
    errors["logvol_energy"] = abs(ev.energies["logvol"]-.5*r_J@r_J)/(.5*r_J@r_J)
    parts = ev.energies["logvol_parts"]
    errors["part_gauss"] = abs(parts["gauss"]-.5*r_g@r_g)/(.5*r_g@r_g)
    errors["part_closed27"] = abs(parts["closed27"]-.5*r_s@r_s)/(.5*r_s@r_s)
    expected_g = sum(np.einsum("cnk,cn->k", blocks["channel"][1], blocks["channel"][0]) if name == "channel"
                     else blocks[name][1].T@blocks[name][0] for name in BLOCK_NAMES)
    expected_h = sum(np.einsum("cnk,cnj,j->k", blocks["channel"][1], blocks["channel"][1], v) if name == "channel"
                     else blocks[name][1].T@(blocks[name][1]@v) for name in BLOCK_NAMES)
    errors["gradient"] = relmax(gradient(state, ev), expected_g)
    errors["gradient_logvol"] = relmax(gradient(state, ev, blocks=("logvol",)), K_J.T@r_J)
    Hv, Hw = hvp(state, ev, v), hvp(state, ev, w)
    errors["hvp"] = relmax(Hv, expected_h)
    errors["hvp_logvol"] = relmax(hvp(state, ev, v, blocks=("logvol",)), K_J.T@(K_J@v))
    composed = gauss_vjp(LogVolume, state, u, gauss_jvp(LogVolume, state, u, v)) \
        + gauss_vjp(LogVolumeClosed27, state, u, gauss_jvp(LogVolumeClosed27, state, u, v))
    errors["hvp_logvol_composed"] = relmax(hvp(state, ev, v, blocks=("logvol",)), composed)
    errors["symmetry"] = abs(float(Hv@w)-float(v@Hw))/(np.linalg.norm(Hv)*np.linalg.norm(w))
    errors["blocks_sum"] = relmax(sum(gradient(state, ev, blocks=(b,)) for b in BLOCK_NAMES), gradient(state, ev))
    record_property("dense_errors", json.dumps(errors))
    assert max(errors.values()) <= 1e-12, errors
    for _ in range(5):
        x = free_random(g, rng)
        assert float(x@hvp(state, ev, x)) >= -1e-12*float(x@x)
        assert float(x@hvp(state, ev, x, blocks=("logvol",))) >= -1e-12*float(x@x)


@pytest.mark.parametrize("seed", [0, 5])
def test_energy_directional_derivative(seed, record_property):
    g, state, rng, u = closed_setup(seed=seed, student_t=bool(seed), share=2.)
    v = inject_free(free_random(g, rng), g)
    v *= np.max(np.abs(u))/np.max(np.abs(v))
    t = 1e-4
    ev, plus, minus = evaluate(state, u), evaluate(state, u+t*v), evaluate(state, u-t*v)
    errors = {}
    for name, pick in (("logvol", lambda e: e.energies["logvol"]), ("F_B", lambda e: e.energy)):
        fd = (pick(plus)-pick(minus))/(2*t)
        blocks = ("logvol",) if name == "logvol" else BLOCK_NAMES
        directional = float(gradient(state, ev, blocks=blocks)@theta_of(v, g))
        errors[name] = abs(fd-directional)/abs(directional)
    record_property("directional_errors", json.dumps(errors))
    assert max(errors.values()) <= 1e-7, errors


# ----------------------------------------------------------------------------------------------
# (iv) order, threads, tiles, record
# ----------------------------------------------------------------------------------------------

def test_accumulation_order_threads_tiles_and_record():
    """log-volume energy = float(1/2 (((G_0 + S_0) + G_1) + S_1 ...)) (longdouble, bitwise); parts G and S; the
    minimum J over the closed27 points and its cell / point equal a brute-force search (first minimum); cpu_threads 1,
    4 and 8 bitwise the same; another tile size changes only the rounding."""
    g, state, rng, u = closed_setup(seed=3, tile_size=7)
    ev = evaluate(state, u)
    total, gauss, closed = LD(0), LD(0), LD(0)
    for tile in cell_tiles(g, state.tile_size):
        rg = gauss_value(LogVolume, state, u, tile=tile)
        rs = gauss_value(LogVolumeClosed27, state, u, tile=tile)
        total += np.sum(rg*rg, dtype=LD)
        total += np.sum(rs*rs, dtype=LD)
        gauss += np.sum(rg*rg, dtype=LD)
        closed += np.sum(rs*rs, dtype=LD)
    assert ev.energies["logvol"] == float(LD(.5)*total)
    parts = ev.energies["logvol_parts"]
    assert parts["gauss"] == float(LD(.5)*gauss) and parts["closed27"] == float(LD(.5)*closed)
    J = gauss_geometry(u, state.closed27_ops).J
    flat = int(np.argmin(J))
    assert parts["closed27_min_J"] == float(J.ravel()[flat])
    assert (parts["closed27_min_cell"], parts["closed27_min_point"]) == divmod(flat, 27)
    assert parts["closed27_min_xyz"] == state.closed27_ops.points[flat % 27].tolist()
    assert set(evaluate(state.replace(logvol_points="gauss27", logvol_closed27_share=1.), u).energies) == \
        {"channel", "diffusion", "logvol", "curl"}                                # no record under C05-2
    v = free_random(g, rng)
    direction = inject_free(free_random(g, rng), g)
    trial_u = u+.3*direction*np.max(np.abs(u))/np.max(np.abs(direction))
    serial = state.replace(cpu_threads=1)
    a = evaluate(serial, u)
    for threaded in (state, state.replace(cpu_threads=8)):
        b = ev if threaded is state else evaluate(threaded, u)
        assert a.total == b.total and a.energies == b.energies
        assert np.array_equal(gradient(serial, a), gradient(threaded, b))
        assert np.array_equal(hvp(serial, a, v), hvp(threaded, b, v))
        ra = stable_reduction(serial, a, evaluate(serial, trial_u))
        rb = stable_reduction(threaded, b, evaluate(threaded, trial_u))
        assert ra.delta == rb.delta and ra.blocks == rb.blocks
    big = state.replace(tile_size=8192)
    assert abs(evaluate(big, u).energies["logvol"]-ev.energies["logvol"]) <= 1e-14*ev.energies["logvol"]


# ----------------------------------------------------------------------------------------------
# (v) stable comparison
# ----------------------------------------------------------------------------------------------

def decimal_change(state, base, trial):
    """Delta F_B at 110 digits from float64 residuals and J: node and linear blocks by half squares, both log-volume
    parts by 1/2 sum w (ln J_t^2 - ln J_b^2), w = (the state's float point weight)/(omega0 tau^2) in Decimal."""
    total = Decimal(0)
    for c in range(state.n_channels):
        total += decimal_half_squares(base.channel.r[c], trial.channel.r[c])
    from bayesreg.model import Curl, Diffusion
    for block in (Diffusion, Curl):
        total += decimal_half_squares(gauss_value(block, state, base.u), gauss_value(block, state, trial.u))
    tile = slice(0, state.grid.n_cells)
    for block in (LogVolume, LogVolumeClosed27):
        ops = block.operators(state)
        tau2 = block.tau2(state, tile)
        weights = np.broadcast_to(block.point_weights(state), tau2.shape)
        J_b = tile_geometry(node_field(base.u, state.grid), ops, tile=tile).J
        J_t = tile_geometry(node_field(trial.u, state.grid), ops, tile=tile).J
        for jb, jt, w, t2 in zip(J_b.ravel(), J_t.ravel(), weights.ravel(), tau2.ravel()):
            lb, lt = Decimal(float(jb)).ln(), Decimal(float(jt)).ln()
            total += Decimal(float(w))/(Decimal(state.omega0)*Decimal(float(t2)))*(lt-lb)*(lt+lb)/2
    return total


@pytest.mark.parametrize("share", [1., 3.])
def test_stable_reduction_with_closed27(share, record_property):
    """C08 with the closed27 term: equal states give 0 exactly; generic and cancellation steps match 110-digit
    Decimal (1e-15); the generic step agrees with the float64 difference within the guard; the log-volume block is
    the per-tile longdouble sum, Gauss term then closed27 term (bitwise); a state with J <= 0 at a closed27 point is
    refused."""
    g, state, rng, u = closed_setup(seed=9, student_t=False, share=share)
    base = evaluate(state, u)
    direction = inject_free(free_random(g, rng), g)
    direction *= np.max(np.abs(u))/np.max(np.abs(direction))
    same = stable_reduction(state, base, evaluate(state, u.copy()))
    assert same.delta == 0 and same.ared == 0 and same.reason == "ENERGY_RESOLUTION"
    errors = {}
    with localcontext() as ctx:
        ctx.prec = 110
        for name, step in (("generic", .3), ("cancellation", 1e-10)):
            trial = evaluate(state, u+step*direction)
            red = stable_reduction(state, base, trial)
            errors[name] = relative(red.delta, decimal_change(state, base, trial))
            if name == "generic":
                errors["generic_vs_float64"] = abs(red.ared-(base.energy-trial.energy))/red.guard
                generic = (trial, red)
    record_property("reduction_errors", json.dumps(errors))
    assert errors["generic"] <= 1e-15 and errors["cancellation"] <= 1e-15 and errors["generic_vs_float64"] <= 1., errors
    trial, red = generic
    omega0 = LD(state.omega0)
    total = LD(0)
    for tile in cell_tiles(g, state.tile_size):
        for block in (LogVolume, LogVolumeClosed27):
            ops = block.operators(state)
            jb, jt = (tile_geometry(node_field(p.u, g), ops, tile=tile).J for p in (base, trial))
            total += _logvol_change_sum(jb, jt, block.point_weights(state).astype(LD)
                                        / (omega0*block.tau2(state, tile).astype(LD)))
    assert red.blocks["logvol"] == LD(.5)*total


def test_c08_decimal_cases_on_the_closed27_part(record_property):
    """The four C08 Decimal cases (contract C05-2b gate) on the closed27 part with its weights w = omega s_p theta_S /
    (omega0 tau_{K,p}^2) (longdouble, as stable_reduction forms them): (i) x = (J_t - J_b)/J_b near +-1e-3, +-1e-2
    and -.6 (the log branch); (ii) J_b at 1 and its neighbouring floats; (iii) a state step below the resolution
    guard (ENERGY_RESOLUTION, the closed27 part of Delta E against Decimal); (iv) equal states give exactly 0;
    relative error <= 1e-15."""
    from .test_energy import decimal_logvol
    g, state, rng, u = closed_setup(seed=12, student_t=False, share=3.)
    tile = slice(0, g.n_cells)
    weight = (state.closed27_point_weights.astype(LD)/(LD(state.omega0)*LogVolumeClosed27.tau2(state, tile).astype(LD)))
    errors = {}
    with localcontext() as ctx:
        ctx.prec = 110
        for x in (1e-3, -1e-3, 1e-2, -1e-2, -.6):
            J_b = rng.uniform(.3, 2., weight.shape)
            J_t = J_b*(1+x)*(1+1e-6*rng.normal(size=weight.shape))
            errors[f"x={x:g}"] = relative(_logvol_change_sum(J_b, J_t, weight)/2, decimal_logvol(J_b, J_t, weight))
        w1 = weight[:1, :1]
        for name, (jb, jt) in dict(up=(1., np.nextafter(1., 2.)), down=(np.nextafter(1., 0.), 1.),
                                   across=(np.nextafter(1., 0.), np.nextafter(1., 2.))).items():
            J_b, J_t = np.full((1, 1), jb), np.full((1, 1), jt)
            errors["near_one_"+name] = relative(_logvol_change_sum(J_b, J_t, w1)/2, decimal_logvol(J_b, J_t, w1))
        base = evaluate(state, u)
        direction = inject_free(free_random(g, rng), g)
        direction *= np.max(np.abs(u))/np.max(np.abs(direction))
        tiny = evaluate(state, u+1e-15*direction)
        red = stable_reduction(state, base, tiny)
        assert red.reason == "ENERGY_RESOLUTION"
        ops = state.closed27_ops
        J_b = tile_geometry(node_field(base.u, g), ops, tile=tile).J
        J_t = tile_geometry(node_field(tiny.u, g), ops, tile=tile).J
        reference = decimal_logvol(J_b, J_t, weight)
        part = _logvol_change_sum(J_b, J_t, weight)/2
        errors["tiny_closed27_part"] = relative(part, reference) if reference != 0 else float(part)
    record_property("decimal_closed27", json.dumps(errors))
    assert max(errors.values()) <= 1e-15, errors
    same = evaluate(state, u.copy())
    assert stable_reduction(state, base, same).delta == 0
    assert _logvol_change_sum(J_b, J_b.copy(), weight) == 0


# ----------------------------------------------------------------------------------------------
# (vi) B13'-2 and the affine field
# ----------------------------------------------------------------------------------------------

def _prior_hessian_columns(state, k=12, seed=0):
    g = state.grid
    ev = evaluate(state, np.zeros(g.shape+(3,)))
    rng = np.random.default_rng(seed)
    V = [free_random(g, rng) for _ in range(k)]
    return {b: np.array([hvp(state, ev, v, blocks=b) for v in V]) for b in (("logvol",),
                                                                            ("diffusion", "logvol", "curl"))}


@pytest.mark.parametrize("share", [1., 3.])
def test_pi0_equals_c05_2_for_cellwise_constant_tau(share, record_property):
    """B13'-2(a): u = 0 and tau^2 constant (uniform tissue): the GN Hessian of the log-volume block and of the whole
    prior under C05-2b (Simpson) equals that of C05-2 to 1e-12; the uniform rule differs (> 1e-3, the quadrature
    artifact of B13'-3(c))."""
    g = aniso()
    state = make_state(g, seed=4, tissue=np.full((4,)+g.shape, .25))
    default = _prior_hessian_columns(state)
    closed = _prior_hessian_columns(state.replace(logvol_points=CLOSED, logvol_closed27_share=share))
    uniform = _prior_hessian_columns(state.replace(logvol_points=CLOSED, logvol_closed27_share=share,
                                                   closed27_weights="uniform"))
    errors = {"+".join(b): relmax(closed[b], default[b]) for b in default}
    record_property("pi0", json.dumps(errors))
    assert max(errors.values()) <= 1e-12, errors
    assert relmax(uniform[("logvol",)], default[("logvol",)]) > 1e-3


@pytest.mark.parametrize("share", [1., 3.])
def test_affine_field_constant_tau(share, record_property):
    """C05-2b gate: on an affine map every J equals det A; with tau constant Pi_J of C05-2b equals that of C05-2
    (1e-13; both rules have total weight omega per cell)."""
    g = aniso()
    base = make_state(g, seed=4, tissue=np.full((4,)+g.shape, .25))
    state = base.replace(logvol_points=CLOSED, logvol_closed27_share=share)
    Amat = np.array([[1.1, .05, -.02], [.03, .95, .04], [-.01, .02, 1.07]])
    X = g.identity()
    u = node_field((X-X[0, 0, 0])@(Amat-np.eye(3)).T, g, allow_nonzero_faces=True)
    tile = slice(0, g.n_cells)
    det = np.linalg.det(Amat)
    energies = {}
    for name, s, blocks in (("default", base, (LogVolume,)), ("closed27", state, (LogVolume, LogVolumeClosed27))):
        total = 0.
        for block in blocks:
            geo = tile_geometry(u, block.operators(s), tile=tile)
            assert np.max(np.abs(geo.J-det)) <= 1e-14
            r = block.residual(block.scale(s, tile), geo)
            total += .5*float(np.sum(r*r))
        energies[name] = total
    expected = .5*g.n_cells*g.omega/base.omega0*np.log(det)**2/float(np.mean(base.tau2))
    errors = dict(closed_vs_default=abs(energies["closed27"]-energies["default"])/energies["default"],
                  closed_vs_formula=abs(energies["closed27"]-expected)/expected)
    record_property("affine", json.dumps(errors))
    assert max(errors.values()) <= 1e-13, errors


def _dense_logvol_hessian(state):
    """The dense log-volume GN Hessian of `state` at u = 0 (columns hvp(e_j), blocks ("logvol",))."""
    g = state.grid
    ev = evaluate(state, np.zeros(g.shape+(3,)))
    n = 3*g.n_free_scalar
    eye = np.eye(n)
    H = np.stack([hvp(state, ev, eye[j], blocks=("logvol",)) for j in range(n)], axis=1)
    return .5*(H+H.T)


@pytest.mark.parametrize("share", [1., 3.])
def test_b13p_2b_partial_volume_sandwich(share, record_property):
    """B13'-(2b) (contract C05-2b gate; review IMPL27 item 3): u = 0, tau^2 the Q1 interpolation of multi-class node
    values (random simplex tissue: tau^2 not constant in the cells): with I_div = sum_q (omega w_g/omega0)
    D_div^T D_div (the exact int (div v)^2 dx/omega0: (div v)^2 is in Q_{2,2,2}, Gauss27 exact; independent dense
    oracle), the C05-2b log-volume GN Hessian Pi_J satisfies (1/max tau^2) I_div <= Pi_J <= (1/min tau^2) I_div --
    for the class range [min_k tau_k^2, max_k tau_k^2] and for the tighter range of the node values -- with eigenvalue
    tolerance 1e-14 relative; Pi_J is not that of C05-2 (relative Frobenius difference > 1e-4; review: 1.2e-2), and
    C05-2's Pi_J obeys the same sandwich."""
    g = aniso(5, 5, 5)
    base = make_state(g, seed=7)
    state = base.replace(logvol_points=CLOSED, logvol_closed27_share=share)
    tau2_nodes = state.tau2_nodes[state.tau2_nodes > 0]
    assert np.ptp(tau2_nodes) > .1                                       # tissue boundaries everywhere
    Dop, _, wq = dense_operators(g)
    S = selection(g)
    Ddiv = np.concatenate([Dop[j]@S for j in range(3)], axis=1)
    I_div = Ddiv.T@((wq/state.omega0)[:, None]*Ddiv)
    H = _dense_logvol_hessian(state)
    H0 = _dense_logvol_hessian(base)
    scale = float(np.max(np.linalg.eigvalsh(I_div)))/float(np.min(state.tau2))
    out = {}
    for name, (lo, hi) in (("classes", (float(np.min(state.tau2)), float(np.max(state.tau2)))),
                           ("nodes", (float(tau2_nodes.min()), float(tau2_nodes.max())))):
        for label, P in (("closed27", H), ("gauss27", H0)):
            lower = float(np.min(np.linalg.eigvalsh(P-I_div/hi)))
            upper = float(np.min(np.linalg.eigvalsh(I_div/lo-P)))
            out[f"{name}_{label}"] = dict(lower=lower/scale, upper=upper/scale)
            assert lower >= -1e-14*scale and upper >= -1e-14*scale, (name, label, lower/scale, upper/scale)
    out["frobenius_vs_c05_2"] = float(np.linalg.norm(H-H0)/np.linalg.norm(H0))
    record_property("b13p_2b", json.dumps(out))
    assert out["frobenius_vs_c05_2"] > 1e-4



# ----------------------------------------------------------------------------------------------
# (vii) W1 and W2 (02_理论 B3'-2)
# ----------------------------------------------------------------------------------------------

def w_grid(spacing):
    hx, hy, hz = spacing
    return make_grid_3d(nx=7, ny=7, nz=7, hx=hx, hy=hy, hz=hz, x0=-1., y0=.5, z0=-2., unit="mm")


def w1_field(grid, b):
    """W1(b) on K0 = cell (2, 2, 2): reference displacements (0, -b, 0) at local vertex (0, 1, 0), (-b, 0, 0) at
    (0, 0, 1), (b, 0, 0) at (0, 1, 1) (physical: times h_d).  On K0's edge (xi_1, xi_2) = (0, 1) J = 1 - b - b^2 t(1 -
    t): the midpoint 1 - b - b^2/4 vanishes at b* = 2 sqrt 2 - 2."""
    hx, hy, _ = grid.spacing
    u = np.zeros(grid.shape+(3,))
    u[2, 3, 2, 1] = -b*hy
    u[3, 2, 2, 0] = -b*hx
    u[3, 3, 2, 0] = b*hx
    return u


def w2_field(grid, a):
    """W2(a): (0, a, 0) at local vertex (1, 1, 0), (-a, 0, 0) at (0, 0, 1), (a, 0, 0) at (0, 1, 1); for a = 19/20 the
    54 sample values are >= 1/20 and the edge (0, 1, t) dips to -41/800 at t = 29/38."""
    hx, hy, _ = grid.spacing
    u = np.zeros(grid.shape+(3,))
    u[2, 3, 3, 1] = a*hy
    u[3, 2, 2, 0] = -a*hx
    u[3, 3, 2, 0] = a*hx
    return u


@pytest.mark.parametrize("spacing", [(1., 1., 1.), (.9, 1.1, 1.3)], ids=["isotropic", "anisotropic"])
def test_w1_edge_dip_visible_only_to_closed27(spacing, record_property):
    """Edge-midpoint dip (W1, b -> b*): the log-volume energies of C05-2 and C05-2a converge (k = 6 -> 8 relative
    change <= 1e-4), C05-2b's grows like (log J_mid)^2 (increments increase); the K0 edge midpoint J equals
    1 - b - b^2/4.  At b = 17/20 (beyond b*): C05-2 and C05-2a evaluate (finite), C05-2b refuses with
    NONPOSITIVE_GAUSS_JD: J <= 0 only at closed27 points (count = those points, the message names closed27)."""
    grid = w_grid(spacing)
    base = make_state(grid, seed=11)
    states = dict(gauss27=base, corners=base.replace(logvol_points="gauss27+corners"),
                  closed27=base.replace(logvol_points=CLOSED))
    k0 = int(np.ravel_multi_index((2, 2, 2), grid.cell_shape))
    mid = states["closed27"].closed27_ops.points.tolist().index([0., 1., .5])
    bstar = 2*np.sqrt(2)-2
    energy = {name: [] for name in states}
    for k in (2, 4, 6, 8):
        b = bstar*(1-10.**-k)
        u = w1_field(grid, b)
        J = gauss_geometry(u, states["closed27"].closed27_ops).J
        assert abs(J[k0, mid]-(1-b-b*b/4)) <= 1e-12
        assert gauss_geometry(u, base.ops).J.min() > .1 and gauss_geometry(
            u, states["corners"].corner_ops).J.min() > .1
        for name, state in states.items():
            energy[name].append(evaluate(state, u).energies["logvol"])
    record_property("w1_energies", json.dumps(energy))
    for name in ("gauss27", "corners"):
        assert abs(energy[name][3]-energy[name][2]) <= 1e-4*energy[name][3], (name, energy[name])
    s = energy["closed27"]
    increments = np.diff(s)
    assert np.all(increments > 0) and increments[2] > increments[1] > increments[0], s
    assert s[3]-s[1] > 100*max(abs(energy["gauss27"][3]-energy["gauss27"][1]),
                               abs(energy["corners"][3]-energy["corners"][1]))
    u = w1_field(grid, 17/20)
    for name in ("gauss27", "corners"):
        assert isinstance(evaluate(states[name], u), Evaluation)
    failure = evaluate(states["closed27"], u)
    J_closed = gauss_geometry(u, states["closed27"].closed27_ops).J
    assert gauss_geometry(u, base.ops).J.min() > 0
    assert isinstance(failure, InvalidTrial) and failure.reason == "NONPOSITIVE_GAUSS_JD"
    assert failure.invalid_count == int(np.count_nonzero(J_closed <= 0)) >= 1
    assert "0 Gauss point(s)" in failure.message and "closed27" in failure.message
    bad = gauss_value(LogVolumeClosed27, states["closed27"], u)
    assert isinstance(bad, InvalidTrial) and "closed27" in bad.message
    ok = evaluate(states["closed27"], np.zeros_like(u))
    forged = Evaluation(states["closed27"], u, ok.channel, ok.total, ok.energy, ok.energies)   # never from evaluate
    with pytest.raises(InputValidationError, match="J <= 0"):
        stable_reduction(states["closed27"], ok, forged)


def test_w2_finite_energy_with_53_positive_points():
    """W2(19/20): J >= 1/20 at every Gauss and closed27 point (equality at a corner of K0), the edge interior negative
    (-41/800 at t = 29/38, the closed form of B3'-2); C05-2b's energy is finite."""
    grid = w_grid((1., 1., 1.))
    state = make_state(grid, seed=11).replace(logvol_points=CLOSED)
    u = w2_field(grid, 19/20)
    assert gauss_geometry(u, state.ops).J.min() > .2
    assert abs(gauss_geometry(u, state.closed27_ops).J.min()-1/20) <= 1e-14
    from bayesreg.solve.witness import cell_J, cell_vertices
    k0 = int(np.ravel_multi_index((2, 2, 2), grid.cell_shape))
    assert abs(float(cell_J(cell_vertices(u, grid, k0), grid.spacing, [(0., 1., 29/38)])[0])+41/800) <= 1e-14
    ev = evaluate(state, u)
    assert isinstance(ev, Evaluation) and np.isfinite(ev.energy)
    assert abs(ev.energies["logvol_parts"]["closed27_min_J"]-1/20) <= 1e-14


# ----------------------------------------------------------------------------------------------
# (viii) Laplace and EB
# ----------------------------------------------------------------------------------------------

def test_laplace_and_eb_accept_simpson_refuse_uniform_and_corners():
    """C14-7 / C15-10 (IMPL27b): the Simpson rule runs (LaplaceSystem, PriorSystem, perturbation_rhs,
    gauss_point_statistics); closed27_weights = 'uniform' is refused as a scope limit (message says so); C05-2a corners
    stay refused."""
    from bayesreg.infer.laplace import LaplaceSystem, PriorSystem, perturbation_rhs
    from bayesreg.learn.population import gauss_point_statistics
    g, state, rng, u = closed_setup(seed=6)
    system = LaplaceSystem(state, u)
    assert system.record()["logvol_points"] == CLOSED and "logvol_closed27" in system.record()["noise_order"][-1]
    PriorSystem(state)
    b = perturbation_rhs(state, evaluate(state, u), lambda name, shape: np.zeros(shape))
    assert np.array_equal(b, np.zeros_like(b))
    stats = gauss_point_statistics(state, u)
    assert stats.n_closed27 == 27*g.n_cells
    uniform = state.replace(closed27_weights="uniform")
    for call in (lambda: LaplaceSystem(uniform, u), lambda: PriorSystem(uniform),
                 lambda: perturbation_rhs(uniform, evaluate(uniform, u), lambda name, shape: np.zeros(shape)),
                 lambda: gauss_point_statistics(uniform, u)):
        with pytest.raises(InputValidationError, match="scope limit"):
            call()
    corners = make_state(g, seed=6).replace(logvol_points="gauss27+corners")
    for call in (lambda: LaplaceSystem(corners, u), lambda: PriorSystem(corners),
                 lambda: gauss_point_statistics(corners, u)):
        with pytest.raises(InputValidationError, match="gauss27\\+corners"):
            call()


def test_default_digest_unchanged_by_the_option_machinery():
    """The default path's digest (the FB1 golden function) is that of the code before C05-2a, computed here again so
    that this file alone shows the C05-2b machinery leaves it unchanged (the constants live in
    test_logvol_corners.GOLDEN_481E33B)."""
    from .test_logvol_corners import GOLDEN_481E33B
    if np.__version__ != "2.0.2":
        pytest.skip("bitwise golden digests recorded with numpy 2.0.2")
    for name, build in (("small", D.small_case), ("box12", D.box12_case)):
        state, u = build(**D.pre_def_model())                      # DEF: "gauss27" explicit
        assert state.logvol_points == "gauss27"
        assert D.model_digest(state, u) == GOLDEN_481E33B[name]
