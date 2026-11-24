"""A synthetic population with a known eta for the bayesreg.learn tests (no data/ access).

Generative model (contract C15 test gate; theory B12, B13):
  * grid 8 x 8 x 8 nodes, h = 1 mm (3 N_int = 648); tissue layer Ptilde_T with K_T = 2 classes plus background:
    P^0 = .05 everywhere, P^1 = .95 s(x), P^2 = .95 (1 - s(x)), s a logistic step across x = 3.5 mm (a partial-
    volume band, B13-3 (b));
  * eta_true (P2): tau = (.15, .06, .10) (tau_0 the largest, so alpha_nug = 1e-2/.15^2 does not move when tau_1,
    tau_2 change), nu = 100, alpha = alpha_nug(tau);
  * u_p ~ N(0, Pi0^-1) with Pi0 the GN precision of the three prior blocks at u = 0 (B13-2: the linearized prior),
    assembled densely by the model's HVP and sampled by Cholesky; |grad u| ~ .1, so log J ~ div u;
  * three channels: textured moving images M_c (plane waves in different directions, wavelengths 4-6 mm),
    fixed F_c = spline(M_c)(x + u_p(x)) + sigma_true N(0, 1), sigma_true = .01 (data precision ~ 100 x the prior's).
The EB MAP here is native-only (one option (a) round of the solver's public `solve_round`, sigma fixed at the true
value; sigma_post is the Gaussian M step at the MAP): the four-level pyramid needs >= 5 nodes per axis on the
coarsest level (native >= ~33 nodes), too large for a test.  `multilevel_map` is exercised separately.

Oracle (linearized, data-dominated limit): the per-pair expectations E[u^T A u] = tr(A Sigma), Sigma = Pi0^-1,
with A = C^T W C / omega0 (nu) and A_k = D_div^T diag(W P^k) D_div (tau_k), and their variances 2 tr((A Sigma)^2).
The one-step EB update from eta_true in that limit is nu_inf = N_C / tr(A_C Sigma) and
tau_k,inf^2 = tr(A_k Sigma) / sum_q W P^k (the "oracle EB point" of the one-round test); by B12 nu_inf > nu_true
(the block-wise normalization (N) over-counts the curl degrees of freedom).  It is NOT a fixed point of the EB
iteration (review M5_EB section 4; corrected in FB1): a further round under the larger nu shrinks the posterior and
raises nu again (a probe run of three rounds from nu_true/5: 20 -> 93 -> 156 -> 188, past nu_inf ~ 162).
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from jd_curl.three_d.boundary import inject_free, restrict_free
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.interpolation import build_image_sampler, sample_points

from bayesreg import model
from bayesreg.geom import scalar_field, tile_interpolate, tile_jvp
from bayesreg.infer.laplace import curl_gram_apply
from bayesreg.learn import MapOutcome, alpha_nugget
from bayesreg.model import ModelState, build_samplers
from bayesreg.solve import Caps, Commit, Eta, GaussianMStep, NumpyBackend, solve_round
from bayesreg.solve.stopping import continuation, final_status, g0_of, threshold

N = 8
TAU_TRUE = (.15, .06, .10)
NU_TRUE = 100.
SIGMA_TRUE = .01
N_CHANNELS = 3
WAVES = (
    (((1., .3, .1), 5.), ((-.2, 1., .3), 4.5), ((.1, -.3, 1.), 6.)),
    (((.6, .7, -.3), 4.), ((-.5, .2, .8), 5.5), ((.9, -.4, .2), 5.)),
    (((.2, .9, .4), 5.), ((.8, .1, -.6), 4.2), ((-.3, -.6, .7), 6.)),
)


def grid8():
    return make_grid_3d(nx=N, ny=N, nz=N, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit="mm")


def eta_true():
    tau2 = tuple(t*t for t in TAU_TRUE)
    return Eta(alpha_nugget(tau2), NU_TRUE, tau2, source="synthetic truth")


def tissue_layer(grid):
    x = grid.identity()[..., 0]
    s = 1/(1+np.exp((x-3.5)/.6))
    p = np.stack([np.full(grid.shape, .05), .95*s, .95*(1-s)])
    return p/p.sum(axis=0)


def brain_mask(grid):
    mask = np.zeros(grid.shape, dtype=bool)
    mask[1:-1, 1:-1, 1:-1] = True
    return mask


def moving_channels(grid):
    X = grid.identity()
    out = []
    for c in range(N_CHANNELS):
        value = np.full(grid.shape, 1.)
        for k, (direction, wavelength) in enumerate(WAVES[c]):
            n = np.asarray(direction)/np.linalg.norm(direction)
            value = value+.4*np.sin(2*np.pi*(X@n)/wavelength+.9*k+c)
        out.append(value)
    return np.stack(out)


def warp(moving, grid, u):
    out = []
    points = (grid.identity()+u).reshape(-1, 3)
    for channel in moving:
        out.append(sample_points(build_image_sampler(channel, grid), points).value.reshape(grid.shape))
    return np.stack(out)


def make_state(grid, fixed, moving, eta, sigma2=SIGMA_TRUE**2, tissue=None, samplers=None):
    return ModelState(grid=grid, omega0=grid.omega, fixed=fixed,
                      samplers=build_samplers(moving, grid) if samplers is None else samplers,
                      mask=brain_mask(grid), sigma2=np.full(fixed.shape[0], sigma2), alpha=eta.alpha, nu=eta.nu,
                      tau2=np.asarray(eta.tau2), tissue=tissue_layer(grid) if tissue is None else tissue)


def dense_columns(apply, n):
    eye = np.eye(n)
    return np.stack([np.asarray(apply(eye[j])) for j in range(n)], axis=1)


@lru_cache(maxsize=1)
def prior_model():
    """(grid, Pi0, chol(Pi0), A_C, D_div, W P^k (2 classes + background), sum_q W P^k) at eta_true."""
    grid = grid8()
    moving = moving_channels(grid)
    state = make_state(grid, moving, moving, eta_true())
    ev = model.evaluate(state, np.zeros(grid.shape+(3,)))
    n = 3*grid.n_free_scalar
    pi0 = dense_columns(lambda v: model.hvp(state, ev, v, blocks=("diffusion", "logvol", "curl")), n)
    pi0 = .5*(pi0+pi0.T)
    chol = np.linalg.cholesky(pi0)
    a_curl = dense_columns(lambda v: curl_gram_apply(state, v), n)
    a_curl = .5*(a_curl+a_curl.T)
    ops = state.ops
    tile = slice(0, grid.n_cells)
    eye3 = np.broadcast_to(np.eye(3), (grid.n_cells, len(ops.points), 3, 3))
    div = np.empty((grid.n_cells*len(ops.points), n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.
        dJ, _, _ = tile_jvp(inject_free(e, grid), ops, tile=tile, A=eye3)
        div[:, j] = dJ.ravel()
    weight = np.tile(ops.weights/state.omega0, grid.n_cells)
    tissue = state.tissue
    wp = np.stack([weight*tile_interpolate(scalar_field(tissue[k], grid), ops, tile=tile).ravel()
                   for k in range(tissue.shape[0])])
    return grid, pi0, chol, a_curl, div, wp


@lru_cache(maxsize=1)
def oracle():
    """Linearized expectations under eta_true: per-pair tr(A Sigma) and Var(u^T A u) = 2 tr((A Sigma)^2) for
    A_C and the A_k, and the data-dominated one-step EB value from eta_true (nu_inf, tau_inf; not a fixed point)."""
    grid, pi0, chol, a_curl, div, wp = prior_model()
    n = pi0.shape[0]
    linv = np.linalg.solve(chol, np.eye(n))              # L^-1, Sigma = L^-T L^-1
    m_curl = linv@a_curl@linv.T
    out = dict(n_free=n, curl_mean=float(np.trace(m_curl)), curl_var=2*float(np.sum(m_curl*m_curl)))
    y = linv@div.T                                        # (n, |Q|): Sigma-whitened div rows
    tau_mean, tau_var, den = [], [], []
    for k in range(wp.shape[0]):
        m = (y*wp[k])@y.T
        tau_mean.append(float(np.trace(m)))
        tau_var.append(2*float(np.sum(m*m)))
        den.append(float(np.sum(wp[k])))
    out.update(tau_mean=tau_mean, tau_var=tau_var, tau_den=den)
    out["nu_inf"] = n/out["curl_mean"]
    out["tau_inf"] = [float(np.sqrt(t/d)) for t, d in zip(tau_mean, den)]
    return out


@dataclass(frozen=True)
class Pair:
    grid: object
    u_true: np.ndarray
    fixed: np.ndarray
    moving: np.ndarray
    samplers: tuple


def make_pairs(n_pairs, seed=20261030):
    grid, pi0, chol, *_ = prior_model()
    rng = np.random.default_rng(seed)
    moving = moving_channels(grid)
    samplers = build_samplers(moving, grid)
    pairs = []
    for _ in range(n_pairs):
        theta = np.linalg.solve(chol.T, rng.standard_normal(pi0.shape[0]))     # Cov = Pi0^-1
        u = inject_free(theta, grid)
        fixed = warp(moving, grid, u)+SIGMA_TRUE*rng.standard_normal((N_CHANNELS,)+grid.shape)
        pairs.append(Pair(grid, u, fixed, moving, samplers))
    return pairs


def native_map(pair, eta, *, caps=Caps(400, 60), backend=None):
    """One option (a) round on the native level under eta with sigma fixed at sigma_true (C11 EM is not the
    object here); sigma_post = the Gaussian M step at the MAP (C11-2, a = b = 0).  Returns a MapOutcome."""
    backend = NumpyBackend() if backend is None else backend
    grid = pair.grid
    state = make_state(grid, pair.fixed, pair.moving, eta, samplers=pair.samplers)
    zero = np.zeros(grid.shape+(3,))
    zero.flags.writeable = False
    start = Commit(zero, restrict_free(zero, grid), backend.certify(zero, grid))
    target = threshold(g0_of(backend, state))
    result = solve_round(backend, state, start, option="a", target=target, caps=caps)
    decision = continuation(result.status, last_iteration=result.last_iteration, inner_status=result.inner_status,
                            commit_u=result.commit.u, commit_cert=result.commit.cert)
    status = final_status(result.status, decision)
    post = GaussianMStep().m_step(state, result.fb_point)
    record = dict(status=status, round_status=result.status, accepted=result.accepted, GF_B=result.gradient_norm,
                  T_B=target, state_hash=result.commit.cert.state_hash)
    return MapOutcome(status, result.reason, True, result.commit.u, state, np.asarray(post).tolist(), record)
