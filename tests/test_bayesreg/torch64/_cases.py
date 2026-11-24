"""Synthetic model states for the torch64 L0 tests (no data/ access).

small   7 x 6 x 5 nodes (the mainline grid3d fixture, anisotropic, off-origin), C = 3, Student-t weights,
        tile_size 7 (18 tiles), chunk_cells 40 (4 one-slab chunks).
box12   12 x 11 x 10 nodes, C = 3, Gaussian, default tile (1 tile), default chunk (1 chunk).
medium  40 x 38 x 36 nodes (50 505 cells, 1.36e6 Gauss points), C = 1 + 3, Student-t, default tile 8192
        (7 tiles), chunk_cells 4096 (2 slabs, 18 chunks); smooth textured channels, ellipsoid mask,
        smooth tissue simplex, smooth displacement of ~0.4 voxel.
"""
import numpy as np

from jd_curl.three_d.boundary import inject_free
from jd_curl.three_d.grid import make_grid_3d

from bayesreg.model import ModelState, build_samplers

from ..model._helpers import aniso, feasible_field, make_state


def _softmax(logits):
    e = np.exp(logits-logits.max(axis=0))
    return e/e.sum(axis=0)


def _medium_images(X, shift):
    x, y, z = np.moveaxis(X+shift, -1, 0)
    intensity = .5+.3*np.sin(.31*x+.2*y)*np.cos(.27*z)+.2*np.cos(.19*x-.23*y+.29*z)+.05*np.sin(.7*x*.3+.5*z)
    logits = np.stack((np.sin(.21*x)+np.cos(.17*y+.1*z), np.cos(.23*z-.11*x)+.5*np.sin(.3*y),
                       np.sin(.13*x+.19*y-.07*z)))
    return np.concatenate((intensity[None], np.sqrt(_softmax(logits))))


def medium_state(*, tile_size=8192):
    g = make_grid_3d(nx=40, ny=38, nz=36, hx=1., hy=1.1, hz=.9, x0=-3., y0=2., z0=1., unit="mm")
    X = g.identity()
    rng = np.random.default_rng(11)
    moving = _medium_images(X, 0.)
    fixed = _medium_images(X, np.array([.3, -.2, .25]))+.01*rng.standard_normal((4,)+g.shape)
    x, y, z = np.moveaxis(X, -1, 0)
    centre = [a.mean() for a in g.axes]
    extent = [a[-1]-a[0] for a in g.axes]
    mask = sum(((c-m)/(.4*e))**2 for c, m, e in zip((x, y, z), centre, extent)) <= 1.
    tissue = _softmax(np.stack((np.cos(.15*x)+np.sin(.12*y), np.sin(.2*z-.1*x), np.cos(.18*y+.05*z),
                                .5*np.sin(.1*(x+y+z)))))
    weights = rng.uniform(.3, 1.2, (4,)+g.shape)
    return ModelState(grid=g, omega0=g.omega, fixed=fixed, samplers=build_samplers(moving, g), mask=mask,
                      sigma2=np.array([.04, .09, .02, .05]), alpha=.13, nu=.07, tau2=np.array([1., .25, .36, .16]),
                      tissue=tissue, weights=weights, tile_size=tile_size)


def smooth_field(grid, amplitude, seed):
    """A smooth U0 node field: sine products vanishing at the faces, amplitude * min h at most."""
    rng = np.random.default_rng(seed)
    X = grid.identity()
    s = [(a-a[0])/(a[-1]-a[0]) for a in grid.axes]
    bump = np.sin(np.pi*s[2])[:, None, None]*np.sin(np.pi*s[1])[None, :, None]*np.sin(np.pi*s[0])[None, None, :]
    out = np.empty(grid.shape+(3,))
    for r in range(3):
        k = rng.uniform(.1, .4, 3)
        phase = rng.uniform(0, 2*np.pi)
        out[..., r] = amplitude*min(grid.spacing)*bump*np.cos(X@k+phase)
    theta = np.moveaxis(out[1:-1, 1:-1, 1:-1], -1, 0).ravel()
    return inject_free(theta, grid)


def case(name):
    """(state, u, torch-state options) of one named case."""
    if name == "small":
        g = aniso()
        state = make_state(g, seed=0, student_t=True, tile_size=7)
        u = feasible_field(g, np.random.default_rng(1))
        return state, u, dict(chunk_cells=40)
    if name == "box12":
        g = make_grid_3d(nx=12, ny=11, nz=10, hx=.5, hy=.45, hz=.6, x0=.1, y0=-.4, z0=.3, unit="mm")
        state = make_state(g, seed=3, student_t=False)
        u = feasible_field(g, np.random.default_rng(4), scale=.12)
        return state, u, {}
    if name == "medium":
        state = medium_state()
        return state, smooth_field(state.grid, .4, 5), dict(chunk_cells=4096)
    raise KeyError(name)
