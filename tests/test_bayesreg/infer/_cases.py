"""Small Laplace cases for the bayesreg.infer tests (no data/ access).

The model states are those of tests/test_bayesreg/model/_helpers.py (polynomial moving channels that
the not-a-knot spline reproduces exactly, so the independent dense oracle `dense_model` applies).
`scaled_state` multiplies every precision (alpha, nu, 1/tau_k^2, 1/sigma_c^2) by `factor`, i.e. H by
`factor`, so the posterior standard deviation scales by factor^-1/2 (used to set the certificate pass rate).
"""
import numpy as np

from bayesreg.solve.backend import CertOutcome, NumpyBackend

from ..model._helpers import SIGMA2, TAU2, aniso, dense_model, feasible_field, make_state


def scaled_state(grid, factor=1., **kwargs):
    return make_state(grid, alpha=.13*factor, nu=.07*factor, tau2=tuple(t/factor for t in TAU2),
                      sigma2=tuple(s/factor for s in SIGMA2), **kwargs)


def cube6(seed=1, factor=1., **kwargs):
    """6 x 6 x 6 nodes (3 N_int = 192, contract C14 gate), anisotropic spacing."""
    grid = aniso(6, 6, 6)
    return grid, scaled_state(grid, factor, seed=seed, **kwargs)


def u_hat_of(grid, seed=3, scale=.05):
    return feasible_field(grid, np.random.default_rng(seed), scale)


def dense_blocks(state, u):
    """Dense K_i (rows x 3 N_int) of the four block kinds at u from the independent oracle; the channel rows
    restricted to the mask nodes in (c, a) order (the noise layout (C, n))."""
    blocks = dense_model(state, u)
    K_ch = blocks["channel"][1][:, state.node_index, :]
    K = dict(channel=K_ch.reshape(-1, K_ch.shape[-1]), diffusion=blocks["diffusion"][1],
             logvol=blocks["logvol"][1], curl=blocks["curl"][1])
    H = sum(k.T@k for k in K.values())
    return K, H


def dense_operator(apply, n):
    """Columns apply(e_j), j < n."""
    eye = np.eye(n)
    return np.stack([np.asarray(apply(eye[j])) for j in range(n)], axis=1)


class RecordingDraw:
    """draw(name, shape) from one generator, recording (name, shape, array) in call order."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.calls = []

    def __call__(self, name, shape):
        value = self.rng.standard_normal(shape)
        self.calls.append((name, tuple(shape), value))
        return value

    def concatenated(self, name):
        return np.concatenate([value.ravel() for n, _, value in self.calls if n == name])


class ScriptedCertificate(NumpyBackend):
    """The numpy backend with certify replaced by a scripted outcome per call (calls in sample order):
    `passes[k]` decides call k.  The C07 configuration stays the real one (posterior_maps checks it)."""

    def __init__(self, passes):
        super().__init__()
        self.passes = list(passes)
        self.calls = 0

    def certify(self, u, grid):
        ok = bool(self.passes[self.calls])
        self.calls += 1
        return CertOutcome(passed=ok, status="certified_feasible" if ok else "infeasible",
                           cert_reason="positive_bernstein_cover" if ok else "JACOBIAN_VIOLATION",
                           rejection=None if ok else "JACOBIAN_VIOLATION", min_jd_lower_bound=.5 if ok else None,
                           min_margin_lower_bound=None, n_uncertain=0, n_infeasible=0 if ok else 1,
                           n_cells=grid.n_cells, witness_cell=None, witness_reference_point=None,
                           state_hash="scripted", certificate_version="scripted", seconds=0.)
