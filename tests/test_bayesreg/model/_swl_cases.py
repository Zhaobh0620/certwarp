"""Model states with structure channels for the sigma_domain tests (work package SWLI; contract C04-6).  No data/."""
import numpy as np

from bayesreg.model import ModelState, build_samplers, support_level

from ..solve import _swl_synthetic as S

TAU2 = np.array([1., .25, .36, .16])


def template(n=12, *, h=2., cpu_threads=1, **kwargs):
    """(grid, ModelState with sigma^2 = 1, LevelInput, u*) of solve/_swl_synthetic.single_level_structure: intensity +
    3 structure channels (blob 3 below theta = .05 on both sides: an empty support)."""
    grid, level, u_true = S.single_level_structure(n, h=h)
    state = ModelState(grid=grid, omega0=grid.omega, fixed=level.fixed, samplers=build_samplers(level.moving, grid),
                       mask=level.mask, sigma2=np.ones(level.fixed.shape[0]), alpha=.02, nu=.1, tau2=TAU2,
                       tissue=level.tissue, cpu_threads=cpu_threads, **kwargs)
    return grid, state, level, u_true


def with_support(state, u, theta=.05, sigma_domain="support", sigma2=None):
    """(SupportLevel at u, the state with that v and sigma2 (default: .04, .09, .02, .5))."""
    level = support_level(state, u, theta, sigma_domain=sigma_domain)
    sigma2 = np.array([.04, .09, .02, .5][:state.n_channels]) if sigma2 is None else np.asarray(sigma2)
    return level, state.replace(sigma2=sigma2, sigma_domain=sigma_domain, support_threshold=theta, support_v=level.v)


def weight_reference(state):
    """The C04-6 gate 2 reference: the same model with v written as float64 Student-t weights (intensity row 1)."""
    C = state.n_channels
    w = np.ones((C, int(np.prod(state.grid.shape))))
    w[1:, state.node_index] = state.support_v
    return ModelState(grid=state.grid, omega0=state.omega0, fixed=state.fixed, samplers=state.samplers, mask=state.mask,
                      sigma2=state.sigma2, alpha=state.alpha, nu=state.nu, tau2=state.tau2, tissue=state.tissue,
                      weights=w.reshape((C,)+state.grid.shape), tile_size=state.tile_size,
                      cpu_threads=state.cpu_threads, logvol_points=state.logvol_points)
