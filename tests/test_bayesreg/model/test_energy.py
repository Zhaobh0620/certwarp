"""bayesreg.model.energy against contract C06 (F_B, gradient, GN HVP, C06-7 sums), C07 (B3-2 counterexample
with the configured certificate), C08 (stable comparison) and C09.1 (metric); C13 gate on the prior
gradients at u = 0.

Small grids only; the certificate test triggers the numba compilation of certify_map (a few seconds).
DEF (2026-10-31): these tests are about the C05-2 model (the ModelState default before DEF); their states are
built with logvol_points "gauss27" explicitly (_gauss27.make_state).
"""
from decimal import Decimal, localcontext
import json

import numpy as np
import pytest

from jd_curl.validation import InputValidationError
from jd_curl.three_d.boundary import inject_free
from jd_curl.three_d.feasibility import PrescreenCertificateConfig, certificate_version, certify_map
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.interpolation import check_map_domain

from bayesreg.geom import gauss_geometry, gauss_gradient_energy, stiffness_apply
from bayesreg.model import (BLOCK_NAMES, Curl, Diffusion, Evaluation, LogVolume, brain_mean_inverse_tau2,
                            build_metric, c_dat, channel_jvp, channel_vjp, evaluate, gauss_jvp, gauss_value, gauss_vjp,
                            gn_diagonal, gradient, half_square_change, hvp, logvol_change, metric_bP,
                            stable_reduction)

from ._gauss27 import make_state
from ._helpers import aniso, dense_model, feasible_field, free_random, poly_gradients, relmax, theta_of

LD = np.longdouble
GAUSS = {"diffusion": Diffusion, "logvol": LogVolume, "curl": Curl}


def kron_stiffness(grid, dtype=np.float64):
    """Vector K_Q on U0 (3 N_int square) from the T10 Kronecker sum M_z(x)M_y(x)K_x + M_z(x)K_y(x)M_x +
    K_z(x)M_y(x)M_x, M_s = h_s tridiag(1,4,1)/6, K_s = tridiag(-1,2,-1)/h_s; independent of poisson.py."""
    mats = []
    for n, h in zip(grid.interior_shape[::-1], grid.spacing):            # x, y, z
        h = dtype(h)
        eye, off = np.eye(n, dtype=dtype), np.eye(n, k=1, dtype=dtype)+np.eye(n, k=-1, dtype=dtype)
        mats.append((h*(4*eye+off)/6, (2*eye-off)/h))
    (Mx, Kx), (My, Ky), (Mz, Kz) = mats
    scalar = np.kron(Mz, np.kron(My, Kx))+np.kron(Mz, np.kron(Ky, Mx))+np.kron(Kz, np.kron(My, Mx))
    m = scalar.shape[0]
    K = np.zeros((3*m, 3*m), dtype=dtype)
    for d in range(3):
        K[d*m:(d+1)*m, d*m:(d+1)*m] = scalar
    return K


def t10_eigenvalues(grid):
    factors = []
    for n, h in zip(grid.interior_shape[::-1], grid.spacing):
        theta = np.pi*np.arange(1, n+1)/(n+1)
        factors.append(((2*np.sin(theta/2)/h)**2, (2+np.cos(theta))/3))
    (lx, mx), (ly, my), (lz, mz) = factors
    lam = (lx[None, None, :]*my[None, :, None]*mz[:, None, None]+mx[None, None, :]*ly[None, :, None]*mz[:, None, None]
           + mx[None, None, :]*my[None, :, None]*lz[:, None, None])
    return grid.omega*lam.ravel()


def setup(seed=0, student_t=True, **kwargs):
    g = aniso()
    state = make_state(g, seed=seed, student_t=student_t, **kwargs)
    rng = np.random.default_rng(seed+100)
    u = feasible_field(g, rng)
    return g, state, rng, u


# ----------------------------------------------------------------------------------------------
# C06: F_B, gradient, HVP
# ----------------------------------------------------------------------------------------------

def test_energy_blocks_and_sums(record_property):
    """C06-1/2/7 and C05-1 gate: block energies equal 1/2 ||r_i||^2 of the dense oracle; the diffusion energy
    equals (alpha/2 omega0) u^T K_Q u (stiffness) and (alpha/2)<|grad u_h|^2>_Q to 1e-12; F_B = float of the
    longdouble sum of the parts in block order; another tile size changes only the rounding."""
    g, state, rng, u = setup()
    ev = evaluate(state, u)
    assert isinstance(ev, Evaluation) and not ev.u.flags.writeable
    dense = dense_model(state, u)
    theta = theta_of(u, g)
    errors = dict(channel=max(abs(e-.5*np.sum(r**2))/(.5*np.sum(r**2)) for e, r in zip(ev.energies["channel"],
                                                                                     dense["channel"][0])))
    for name in ("diffusion", "logvol", "curl"):
        expected = .5*np.sum(dense[name][0]**2)
        errors[name] = abs(ev.energies[name]-expected)/expected
    kq = .5*state.alpha/state.omega0*float(theta@stiffness_apply(theta, g))
    errors["diffusion_KQ"] = abs(ev.energies["diffusion"]-kq)/kq
    gauss = .5*state.alpha/state.omega0*gauss_gradient_energy(u, state.ops)
    errors["diffusion_gauss"] = abs(ev.energies["diffusion"]-gauss)/gauss
    record_property("energy_errors", json.dumps(errors))
    assert max(errors.values()) <= 1e-12, errors
    parts = [LD(e) for e in ev.energies["channel"]]+[LD(ev.energies[n]) for n in ("diffusion", "logvol", "curl")]
    assert abs(float(sum(parts))-ev.energy) <= 1e-15*ev.energy
    assert ev.energy == float(ev.total)
    tiled = evaluate(state.replace(tile_size=7), u)
    assert abs(tiled.energy-ev.energy) <= 1e-14*ev.energy
    for name in ("diffusion", "logvol", "curl"):
        assert abs(tiled.energies[name]-ev.energies[name]) <= 1e-14*ev.energies[name]
    with pytest.raises(InputValidationError):
        evaluate(state, u+1.)                           # faces must be zero (U0)


@pytest.mark.parametrize("seed", [0, 5])
def test_energy_directional_derivative(seed, record_property):
    """C06 gate: |(F(u + t v) - F(u - t v))/2t - <grad F, v>| <= 1e-7 |<grad F, v>|, for F_B and each block."""
    g, state, rng, u = setup(seed=seed, student_t=bool(seed))
    v = inject_free(free_random(g, rng), g)
    v *= np.max(np.abs(u))/np.max(np.abs(v))
    t = 1e-4
    ev, plus, minus = evaluate(state, u), evaluate(state, u+t*v), evaluate(state, u-t*v)
    errors = {}
    for name in BLOCK_NAMES:
        pick = (lambda e: sum(e.energies["channel"])) if name == "channel" else (lambda e, n=name: e.energies[n])
        fd = (pick(plus)-pick(minus))/(2*t)
        directional = float(gradient(state, ev, blocks=(name,))@theta_of(v, g))
        errors[name] = abs(fd-directional)/abs(directional)
    fd = (plus.energy-minus.energy)/(2*t)
    directional = float(gradient(state, ev)@theta_of(v, g))
    errors["F_B"] = abs(fd-directional)/abs(directional)
    record_property("directional_errors", json.dumps(errors))
    assert max(errors.values()) <= 1e-7, errors


def test_gradient_equals_sum_of_adjoints(record_property):
    """C06-4: grad F_B = sum_i K_i^T r_i (dense oracle, 1e-12); the diffusion term (alpha/omega0) K_Q u via
    stiffness_apply equals K_D^T r_D by the Gauss-point scatter (1e-13); per-block selection sums to the
    total."""
    g, state, rng, u = setup(seed=2)
    ev = evaluate(state, u)
    dense = dense_model(state, u)
    expected = sum(np.einsum("cnk,cn->k", dense["channel"][1], dense["channel"][0]) if name == "channel"
                   else dense[name][1].T@dense[name][0] for name in BLOCK_NAMES)
    total = gradient(state, ev)
    errors = dict(dense=relmax(total, expected))
    scatter = gauss_vjp(Diffusion, state, u, gauss_value(Diffusion, state, u))
    errors["diffusion_stiffness_vs_scatter"] = relmax(gradient(state, ev, blocks=("diffusion",)), scatter)
    parts = sum(gradient(state, ev, blocks=(name,)) for name in BLOCK_NAMES)
    errors["blocks_sum"] = relmax(parts, total)
    errors["channel_vjp"] = relmax(gradient(state, ev, blocks=("channel",)), channel_vjp(state, ev.channel, ev.channel.r))
    record_property("gradient_errors", json.dumps(errors))
    assert errors["dense"] <= 1e-12 and max(errors.values()) <= 1e-13, errors
    with pytest.raises(InputValidationError):
        gradient(state, ev, blocks=("channel", "barrier"))
    with pytest.raises(InputValidationError):
        gradient(state.replace(alpha=1.), ev)        # an Evaluation is bound to its ModelState


def test_hvp_symmetric_psd_and_dense(record_property):
    """C06-5 gate: H v = sum_i K_i^T K_i v equals the dense oracle (1e-12) and the composition of the block
    jvp/vjp (1e-13); |<Hv, w> - <v, Hw>| <= 1e-12 |Hv||w|; <v, Hv> >= -1e-12 |v|^2; with alpha > 0
    <v, Hv> >= (alpha/omega0) v^T K_Q v - 1e-12."""
    g, state, rng, u = setup(seed=3)
    ev = evaluate(state, u)
    dense = dense_model(state, u)
    v, w = free_random(g, rng), free_random(g, rng)
    Hv, Hw = hvp(state, ev, v), hvp(state, ev, w)
    expected = sum(np.einsum("cnk,cnj,j->k", dense["channel"][1], dense["channel"][1], v) if name == "channel"
                   else dense[name][1].T@(dense[name][1]@v) for name in BLOCK_NAMES)
    errors = dict(dense=relmax(Hv, expected))
    composed = channel_vjp(state, ev.channel, channel_jvp(state, ev.channel, v))
    errors["channel_composed"] = relmax(hvp(state, ev, v, blocks=("channel",)), composed)
    for name, block in GAUSS.items():
        composed = gauss_vjp(block, state, u, gauss_jvp(block, state, u, v))
        errors[name+"_composed"] = relmax(hvp(state, ev, v, blocks=(name,)), composed)
    errors["symmetry"] = abs(float(Hv@w)-float(v@Hw))/(np.linalg.norm(Hv)*np.linalg.norm(w))
    record_property("hvp_errors", json.dumps(errors))
    assert errors["dense"] <= 1e-12 and errors["symmetry"] <= 1e-12 and max(errors.values()) <= 1e-12, errors
    K = kron_stiffness(g)
    for _ in range(5):
        x = free_random(g, rng)
        quad = float(x@hvp(state, ev, x))
        assert quad >= -1e-12*float(x@x)
        assert quad >= state.alpha/state.omega0*float(x@K@x)-1e-12*max(1., abs(quad))


def test_prior_gradients_vanish_at_identity():
    """C13 gate: at u = 0 the diffusion, log-volume and curl gradients are zero (<= 1e-14 G0; exactly 0
    here), so G0 is the data term's gradient."""
    g, state, rng, _ = setup(seed=4)
    ev = evaluate(state, np.zeros(3*g.n_free_scalar))
    g0 = np.linalg.norm(gradient(state, ev))
    assert g0 > 0 and g0 == np.linalg.norm(gradient(state, ev, blocks=("channel",)))
    for name in ("diffusion", "logvol", "curl"):
        part = gradient(state, ev, blocks=(name,))
        assert np.linalg.norm(part) <= 1e-14*g0 and not np.any(part), name


def test_gn_diagonal_and_c_dat(record_property):
    """C06-6: the channel GN diagonal equals diag(sum_c K_c^T K_c) of the dense oracle (1e-12); C09.1 c_dat =
    (1/(3 N_B)) sum_mask sum_d sum_c (w_c/sigma_c^2)(d_d G_c)^2 with analytic gradients (1e-12)."""
    g, state, rng, u = setup(seed=6)
    ev = evaluate(state, u)
    K = dense_model(state, u)["channel"][1]
    errors = dict(diag=relmax(gn_diagonal(state, ev), np.einsum("cnk,cnk->k", K, K)))
    X = g.identity().reshape(-1, 3)[state.node_index]+u.reshape(-1, 3)[state.node_index]
    grads = poly_gradients(X)
    w = state.weights.reshape(3, -1)[:, state.node_index]
    expected = float(np.sum((w/state.sigma2[:, None])[..., None]*grads**2)/(3*state.node_index.size))
    errors["c_dat"] = abs(c_dat(state, ev)-expected)/expected
    record_property("gn_errors", json.dumps(errors))
    assert max(errors.values()) <= 1e-12, errors


# ----------------------------------------------------------------------------------------------
# C09.1 metric
# ----------------------------------------------------------------------------------------------

def test_metric_against_dense(record_property):
    """C09.1 gate: M^-1 = omega0 * build_poisson_inverse(a = c_dat + lam_b + gamma, b = b_P).apply equals the
    dense inverse of (b_P K_Q/omega0 + (omega_ell a/omega0) I) to 1e-10; matvec and solve are inverse to
    1e-12; the spectrum is (omega_ell/omega0) a + (b_P/omega0) omega_ell Lambda_Q; b_P "principal" and
    "methodology" follow C09.1; the methodology form (alpha K_Q/omega0 + c I)^-1 is the case b_P = alpha."""
    g, state, rng, u = setup(seed=7)
    ev = evaluate(state, u)
    K = kron_stiffness(g)
    cd = c_dat(state, ev)
    q = g.node_weights().ravel()[state.node_index]
    tau2 = sum(t*p for t, p in zip(state.tau2, state.tissue)).ravel()[state.node_index]
    mean = np.sum(q/tau2)/np.sum(q)
    assert abs(brain_mean_inverse_tau2(state)-mean) <= 1e-14*mean
    b_p = metric_bP(state)
    assert abs(b_p-(state.alpha+.5*(mean+state.nu))) <= 1e-15*b_p
    assert abs(metric_bP(state, mu=.03, nu2=.05)-(b_p+.04)) <= 1e-15*b_p
    assert metric_bP(state, mode="methodology") == state.alpha
    metric = build_metric(state, c_dat=cd, b_P=b_p, lam_b=.2, gamma=.05)
    assert metric.a == cd+.2+.05 and metric.shift == g.omega*metric.a/state.omega0
    dense = b_p/state.omega0*K+metric.shift*np.eye(K.shape[0])
    rhs = rng.normal(size=K.shape[0])
    solved = metric.solve(rhs)
    reference = np.linalg.solve(dense, rhs)
    errors = dict(solve=float(np.linalg.norm(solved-reference)/np.linalg.norm(reference)),
                  matvec=float(np.linalg.norm(metric.matvec(rhs)-dense@rhs)/np.linalg.norm(dense@rhs)),
                  roundtrip=float(np.linalg.norm(metric.matvec(solved)-rhs)/np.linalg.norm(rhs)))
    spectrum = np.sort(np.tile(g.omega*metric.a/state.omega0+b_p/state.omega0*t10_eigenvalues(g), 3))
    errors["spectrum"] = float(np.max(np.abs(np.linalg.eigvalsh(dense)-spectrum))/spectrum[-1])
    c = .4                                               # methodology form: (alpha K_Q/omega0 + c I)^-1
    methodology = build_metric(state, c_dat=c*state.omega0/g.omega, b_P=metric_bP(state, mode="methodology"))
    reference = np.linalg.solve(state.alpha/state.omega0*K+c*np.eye(K.shape[0]), rhs)
    errors["methodology"] = float(np.linalg.norm(methodology.solve(rhs)-reference)/np.linalg.norm(reference))
    record_property("metric_errors", json.dumps(errors))
    assert errors["solve"] <= 1e-10 and errors["methodology"] <= 1e-10, errors
    assert max(errors["matvec"], errors["roundtrip"], errors["spectrum"]) <= 1e-12, errors
    for kwargs in (dict(c_dat=-1., b_P=1.), dict(c_dat=1., b_P=0.), dict(c_dat=1., b_P=1., gamma=-1.)):
        with pytest.raises(InputValidationError):
            build_metric(state, **kwargs)
    for kwargs in (dict(mode="seven_point"), dict(mu=.1), dict(mode="methodology", mu=.1, nu2=.1)):
        with pytest.raises(InputValidationError):
            metric_bP(state, **kwargs)


# ----------------------------------------------------------------------------------------------
# C07: B3-2 counterexample against the configured certificate
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("spacing", [(1., 1., 1.), (.9, 1.1, 1.3)], ids=["isotropic", "anisotropic"])
def test_b3_2_counterexample_rejected_by_certificate(spacing, record_property):
    """C07 gates, theory B3(2): the configured certificate (PrescreenCertificateConfig(delta=0.): delta 0,
    max_depth 48, max_boxes 255, tile 8192) passes u = 0 with 0 < min J bound <= 1.  The two-node
    construction (K0 two cells from every face; local vertices (1,0,1), (1,1,1) moved by (0, 0, -1.05 h_z))
    has J = 1 - 1.05 xi_1 on K0, all 27|K| Gauss J_q > 0 (min 1 - 1.05 * .887298 = .0683), legal sampling and
    finite F_B, and the certificate does not pass it (infeasible, JACOBIAN_VIOLATION)."""
    hx, hy, hz = spacing
    grid = make_grid_3d(nx=7, ny=7, nz=7, hx=hx, hy=hy, hz=hz, x0=-1., y0=.5, z0=-2., unit="mm")
    config = PrescreenCertificateConfig(delta=0.)
    assert isinstance(config, PrescreenCertificateConfig)
    assert config.delta == 0. and config.max_depth == 48 and config.max_boxes == 255 and config.tile_size == 8192
    identity = certify_map(np.zeros(grid.shape+(3,)), grid, config=config)
    assert identity.status == "certified_feasible" and 0 < identity.min_jd_lower_bound <= 1

    u = np.zeros(grid.shape+(3,))
    u[3, 2, 3, 2] = u[3, 3, 3, 2] = -1.05*hz        # nodes (z, y, x) = (3, 2, 3), (3, 3, 3) of cell (2, 2, 2)
    state = make_state(grid, seed=11)
    geo = gauss_geometry(u, state.ops)
    k0 = int(np.ravel_multi_index((2, 2, 2), grid.cell_shape))
    np.testing.assert_allclose(geo.J[k0], 1-1.05*state.ops.points[:, 0], rtol=0, atol=1e-14)
    assert geo.J.min() > 0 and abs(geo.J.min()-(1-1.05*(.5+np.sqrt(15)/10))) <= 1e-14
    assert int(np.count_nonzero(np.any(geo.J != 1, axis=1))) == 12          # the 12 cells touching the nodes
    assert check_map_domain(u, grid) is None
    ev = evaluate(state, u)
    assert isinstance(ev, Evaluation) and np.isfinite(ev.energy) and ev.energies["logvol"] > 0
    result = certify_map(u, grid, config=config)
    record_property("certificate", json.dumps(dict(status=result.status, reason=result.reason,
                                                   min_jd_lower_bound=result.min_jd_lower_bound,
                                                   n_infeasible=result.n_infeasible, version=certificate_version(config))))
    assert result.status != "certified_feasible"
    assert result.status == "infeasible" and result.reason == "JACOBIAN_VIOLATION"
    assert result.min_jd_lower_bound is None or result.min_jd_lower_bound <= 0


# ----------------------------------------------------------------------------------------------
# C08: stable comparison
# ----------------------------------------------------------------------------------------------

def to_decimal(value):
    return Decimal(np.format_float_scientific(LD(value), unique=False, precision=24))


def decimal_logvol(J_b, J_t, weight):
    total = Decimal(0)
    for jb, jt, w in zip(np.ravel(J_b), np.ravel(J_t), np.ravel(weight)):
        lb, lt = Decimal(float(jb)).ln(), Decimal(float(jt)).ln()
        total += Decimal(float(w))*(lt-lb)*(lt+lb)
    return total/2


def decimal_half_squares(base, trial):
    total = Decimal(0)
    for b, t in zip(np.ravel(base), np.ravel(trial)):
        db, dt = Decimal(float(b)), Decimal(float(t))
        total += (dt-db)*(dt+db)
    return total/2


def relative(stable, reference):
    return float(abs(to_decimal(stable)-reference)/abs(reference))


def test_stable_differences_decimal(record_property):
    """C08 gate against 110-digit Decimal: (i) x = (J_t - J_b)/J_b near +-1e-3, +-1e-2 and -.6 (the log
    branch); (ii) J_b at 1 and its neighbouring floats; (iv) equal states give exactly 0; the generic
    half-square difference likewise; relative error <= 1e-15.  J <= 0 raises."""
    errors = {}
    with localcontext() as ctx:
        ctx.prec = 110
        rng = np.random.default_rng(0)
        for x in (1e-3, -1e-3, 1e-2, -1e-2, -.6):
            J_b = rng.uniform(.3, 2., 4)
            J_t = J_b*(1+x)*(1+1e-6*rng.normal(size=4))
            w = rng.uniform(.1, 2., 4)
            errors[f"x={x:g}"] = relative(logvol_change(J_b, J_t, w), decimal_logvol(J_b, J_t, w))
        one = np.array([1.])
        for name, (jb, jt) in dict(up=(1., np.nextafter(1., 2.)), down=(np.nextafter(1., 0.), 1.),
                                   across=(np.nextafter(1., 0.), np.nextafter(1., 2.))).items():
            J_b, J_t = one*jb, one*jt
            errors["near_one_"+name] = relative(logvol_change(J_b, J_t, one*.7), decimal_logvol(J_b, J_t, one*.7))
            errors["near_one_"+name+"_reversed"] = relative(logvol_change(J_t, J_b, one*.7),
                                                            decimal_logvol(J_t, J_b, one*.7))
        base = rng.normal(size=500)
        trial = base*(1+1e-9*rng.normal(size=500))
        errors["half_squares_cancellation"] = relative(half_square_change(base, trial),
                                                       decimal_half_squares(base, trial))
        naive = float(.5*(np.sum(trial**2)-np.sum(base**2)))
        errors["half_squares_naive_float64"] = float(abs(Decimal(naive)-decimal_half_squares(base, trial))
                                                     / abs(decimal_half_squares(base, trial)))
    record_property("decimal_errors", json.dumps(errors))
    assert max(v for k, v in errors.items() if "naive" not in k) <= 1e-15, errors
    assert errors["half_squares_naive_float64"] > 1e3*errors["half_squares_cancellation"]
    J = rng.uniform(.5, 2., 30)
    assert logvol_change(J, J.copy(), np.ones(30)) == 0 and half_square_change(J, J.copy()) == 0
    for bad in (np.array([1., 0.]), np.array([1., -.5])):
        with pytest.raises(InputValidationError):
            logvol_change(np.ones(2), bad, np.ones(2))
        with pytest.raises(InputValidationError):
            logvol_change(bad, np.ones(2), np.ones(2))


def decimal_model_change(state, base, trial):
    """Delta E of two Evaluations from their float64 residual values and J_q (110 digits): node blocks and
    the linear Gauss blocks by half squares, the log-volume block by 1/2 sum w (ln J_t^2 - ln J_b^2) with
    w = omega w_g/(omega0 tau_q^2) formed in Decimal."""
    total = Decimal(0)
    for c in range(state.n_channels):
        total += decimal_half_squares(base.channel.r[c], trial.channel.r[c])
    for block in (Diffusion, Curl):
        total += decimal_half_squares(gauss_value(block, state, base.u), gauss_value(block, state, trial.u))
    tau2 = LogVolume.tau2(state, slice(0, state.grid.n_cells))
    weights = np.broadcast_to(state.ops.weights, tau2.shape)
    J_b, J_t = gauss_geometry(base.u, state.ops).J, gauss_geometry(trial.u, state.ops).J
    for jb, jt, w, t2 in zip(J_b.ravel(), J_t.ravel(), weights.ravel(), tau2.ravel()):
        lb, lt = Decimal(float(jb)).ln(), Decimal(float(jt)).ln()
        total += Decimal(float(w))/(Decimal(state.omega0)*Decimal(float(t2)))*(lt-lb)*(lt+lb)/2
    return total


def test_stable_reduction_model(record_property):
    """C08 on F_B: (iii) a step below the resolution guard is ENERGY_RESOLUTION and its Delta E matches the
    Decimal value of the residual differences (1e-15 relative); equal states give Delta E = 0 exactly; a
    generic step agrees with the float64 difference F_b - F_t within the guard and its diffusion part with an
    independent longdouble (alpha/2 omega0)(u_t^T K_Q u_t - u_b^T K_Q u_b) (1e-13); a cancellation step
    (u_t - u_b ~ 1e-10 |u|) is resolved where the float64 difference is not; uphill is NO_STRICT_DECREASE,
    a descent step passes; Evaluations of another ModelState are refused."""
    g, state, rng, u = setup(seed=9, student_t=False)
    base = evaluate(state, u)
    direction = inject_free(free_random(g, rng), g)
    direction *= np.max(np.abs(u))/np.max(np.abs(direction))
    errors = {}

    same = stable_reduction(state, base, evaluate(state, u.copy()))
    assert same.delta == 0 and same.ared == 0 and same.reason == "ENERGY_RESOLUTION"

    tiny = evaluate(state, u+1e-15*direction)
    reduction = stable_reduction(state, base, tiny)
    assert abs(base.energy-tiny.energy) <= reduction.guard and reduction.reason == "ENERGY_RESOLUTION"
    with localcontext() as ctx:
        ctx.prec = 110
        reference = decimal_model_change(state, base, tiny)
        errors["tiny_vs_decimal"] = relative(reduction.delta, reference) if reference != 0 else float(reduction.delta)

        trial = evaluate(state, u+.3*direction)
        generic = stable_reduction(state, base, trial)
        errors["generic_vs_float64"] = abs(generic.ared-(base.energy-trial.energy))/generic.guard
        K = kron_stiffness(g, LD)
        tb, tt = theta_of(u, g).astype(LD), theta_of(u+.3*direction, g).astype(LD)
        expected = LD(state.alpha)/(2*LD(state.omega0))*(tt@(K@tt)-tb@(K@tb))
        errors["diffusion_vs_longdouble_KQ"] = float(abs(generic.blocks["diffusion"]-expected)/abs(expected))
        errors["generic_vs_decimal"] = relative(generic.delta, decimal_model_change(state, base, trial))

        close = evaluate(state, u+1e-10*direction)
        cancel = stable_reduction(state, base, close)
        reference = decimal_model_change(state, base, close)
        errors["cancellation_stable"] = relative(cancel.delta, reference)
        errors["cancellation_float64"] = float(abs(Decimal(close.energy)-Decimal(base.energy)-reference)/abs(reference))
    record_property("reduction_errors", json.dumps(errors))
    assert errors["tiny_vs_decimal"] <= 1e-15 and errors["generic_vs_decimal"] <= 1e-15, errors
    assert errors["cancellation_stable"] <= 1e-15, errors
    assert errors["cancellation_float64"] > 1e3*max(errors["cancellation_stable"], 1e-19), errors
    assert errors["generic_vs_float64"] <= 1. and errors["diffusion_vs_longdouble_KQ"] <= 1e-13, errors

    descent = -gradient(state, base)
    descent *= 1e-2*np.max(np.abs(u))/np.max(np.abs(descent))
    down = stable_reduction(state, base, evaluate(state, u+inject_free(descent, g)))
    up = stable_reduction(state, base, evaluate(state, u-inject_free(descent, g)))
    assert down.reason is None and down.certified > 0
    assert up.reason == "NO_STRICT_DECREASE" and up.ared < 0
    with pytest.raises(InputValidationError):
        stable_reduction(state.replace(nu=1.), base, trial)
