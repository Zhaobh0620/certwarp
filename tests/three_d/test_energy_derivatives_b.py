"""B protocol: independent dense and directional checks, no optimizer runs."""
from dataclasses import replace
from decimal import Decimal, localcontext
import json

import numpy as np
import pytest

from jd_curl.three_d.barrier import barrier_terms
from jd_curl.three_d.linearization import (energy_gradient, residual_jvp_blocks, residual_vjp_blocks,
                                          barrier_ggn_hvp, barrier_ggn_diagonal, build_linearization)
from jd_curl.three_d.residuals import evaluate_trial, promote_trial, residual_blocks, TrialState3D
from jd_curl.validation import InputValidationError
from .b_support import case, pack, oracle, dense_derivatives


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(1e-30, np.linalg.norm(b)))


@pytest.mark.parametrize("mu,nu", [(1., 1.), (.001, .001), (.7, .03)])
@pytest.mark.parametrize("seed", [0, 17, 91])
def test_dense_oracle_and_adjoint(mu, nu, seed, record_property):
    problem, state, lin = case(mu=mu, nu=nu)
    R, K, g, H = oracle(problem, state)
    rng = np.random.default_rng(seed)
    p, v = rng.normal(size=(2, len(g)))
    rb = list(residual_blocks(problem, state))
    tb = [replace(b, values=rng.normal(size=b.values.shape)) for b in rb]
    t = pack(tb)
    Hp = barrier_ggn_hvp(lin, p)
    errors = dict(residual=relative(pack(rb), R), jvp=relative(pack(residual_jvp_blocks(lin, p)), K@p),
                  vjp=relative(residual_vjp_blocks(lin, tb), K.T@t), gradient=relative(energy_gradient(lin), g),
                  hvp=relative(Hp, H@p), diagonal=relative(barrier_ggn_diagonal(lin), H.diagonal()))
    assert max(errors.values()) < 2e-11, errors
    kp = pack(residual_jvp_blocks(lin, p))
    kt = residual_vjp_blocks(lin, tb)
    adjoint = abs(kp@t-p@kt)/max(1e-30, np.linalg.norm(kp)*np.linalg.norm(t))
    Hv = barrier_ggn_hvp(lin, v)
    symmetry = abs(p@Hv-v@Hp)/max(1e-30, np.linalg.norm(p)*np.linalg.norm(Hv))
    assert adjoint < 2e-12 and symmetry < 2e-12
    floor = problem.weights.lam*problem.grid.omega
    assert p@Hp >= floor*(p@p)*(1-1e-12)
    assert np.linalg.eigvalsh(H)[0] >= floor*(1-1e-12)
    # The complete gradient includes the direct barrier, not K0.T R0 alone.
    assert np.linalg.norm(g-K.T@R) > 1e-6
    np.testing.assert_allclose(energy_gradient(lin), residual_vjp_blocks(lin, rb)+energy_gradient(lin, term="barrier"), atol=2e-16)
    assert len(R) == 4*np.prod(problem.grid.shape)+108*problem.grid.n_cells
    np.testing.assert_allclose(.5*(R@R), sum(getattr(state.energy, name) for name in ("image", "proximity", "jd", "curl")), rtol=2e-11)
    record_property("B_dense", json.dumps(dict(mu=mu, nu=nu, seed=seed, errors=errors,
                                               adjoint=adjoint, symmetry=symmetry, lambda_min=float(np.linalg.eigvalsh(H)[0]))))


def test_barrier_energy_uses_all_local_corners_with_physical_volume():
    problem, state, _ = case()
    D, weights = dense_derivatives(problem.grid, corners=True)
    F = np.einsum("jqn,nr->qrj", D, state.u.reshape(-1, 3))+np.eye(3)
    J = np.linalg.det(F).astype(np.longdouble)
    t = (J-np.longdouble(.01))/(1-np.longdouble(.01))
    expected = float(np.sum(problem.weights.kappa*weights*(t-1-np.log(t))))
    np.testing.assert_allclose(state.energy.barrier, expected, rtol=2e-11)


@pytest.mark.parametrize("mu", [1., .001])
@pytest.mark.parametrize("seed", [0, 17, 91])
def test_all_energy_directional_derivatives(mu, seed, record_property):
    problem, state, lin = case(mu=mu)
    direction = np.random.default_rng(seed).normal(size=state.theta.shape)
    direction /= np.linalg.norm(direction)
    terms = ("image", "proximity", "jd", "curl", "barrier", "outer", "total")
    exact = {k: energy_gradient(lin, term=k)@direction for k in terms}
    eps = [1e-2, 5e-3, 2.5e-3, 1e-4, 1e-5, 1e-6]
    errors = {k: [] for k in terms}
    taylor = []
    R = pack(residual_blocks(problem, state))
    kp = pack(residual_jvp_blocks(lin, direction))
    for e in eps:
        plus, minus = [evaluate_trial(problem, state.theta+sign*e*direction) for sign in (1, -1)]
        assert isinstance(plus, TrialState3D) and isinstance(minus, TrialState3D)
        for k in terms:
            fd = (getattr(plus.energy, k)-getattr(minus.energy, k))/(2*e)
            errors[k].append(abs(fd-exact[k])/max(1e-8, abs(exact[k])))
        taylor.append(float(np.linalg.norm(pack(residual_blocks(problem, plus))-R-e*kp)))
    assert max(min(v) for v in errors.values()) < 3e-6, errors
    # Proximity/curl are quadratic; other terms and residual Taylor are nonlinear.
    for k in ("image", "jd", "barrier", "outer", "total"):
        assert errors[k][0] > 3.5*errors[k][1] and errors[k][1] > 3.5*errors[k][2], (k, errors[k])
    assert taylor[0] > 3.5*taylor[1] and taylor[1] > 3.5*taylor[2]
    np.testing.assert_allclose(sum(energy_gradient(lin, term=k) for k in terms[:5]), energy_gradient(lin), atol=1e-16)
    record_property("B_directional", json.dumps(dict(mu=mu, seed=seed, eps=eps, errors=errors, residual_taylor=taylor)))


def test_ggn_is_not_exact_newton():
    problem, state, lin = case()
    p = np.random.default_rng(17).normal(size=len(state.theta))
    p /= np.linalg.norm(p)
    eps = 1e-5
    gradients = []
    for sign in (1, -1):
        s = promote_trial(problem, evaluate_trial(problem, state.theta+sign*eps*p))
        gradients.append(energy_gradient(build_linearization(problem, s)))
    newton_p = (gradients[0]-gradients[1])/(2*eps)
    assert relative(barrier_ggn_hvp(lin, p), newton_p) > 1e-4


@pytest.mark.parametrize("J", [1., np.nextafter(1., 2.), np.nextafter(1., 0.),
                                   1.+1e-8, 1.-1e-8, .99009, 1.00991, .4, 2.7,
                                   .01+1e-5, .01+1e-10, np.nextafter(.01, 1.)])
def test_barrier_high_precision(J, record_property):
    delta, weight = .01, .017*.024/8
    actual = barrier_terms(np.array([J]), delta=delta, weight=weight)
    with localcontext() as ctx:
        ctx.prec = 100
        j, d, w = map(Decimal.from_float, (J, delta, weight))
        t = (j-d)/(1-d)
        expected = [float(w*(t-1-t.ln())), float(w*(1/(1-d)-1/(j-d))), float(w/(j-d)**2)]
    errors = [abs(float(a[0])-b)/max(abs(b), 1e-300) for a, b in zip(actual, expected)]
    assert max(errors) <= 2e-12, errors
    record_property("B_barrier_decimal", json.dumps(dict(J=J, relative_errors=errors)))


@pytest.mark.parametrize("J,delta,weight", [(.01,.01,1), (0,.01,1), (1,-.01,1), (1,1,1),
                                            (1,.01,0), (1,.01,-1), (np.nan,.01,1)])
def test_invalid_barrier_arguments(J, delta, weight):
    with pytest.raises(InputValidationError):
        barrier_terms(np.array([J]), delta=delta, weight=weight)
