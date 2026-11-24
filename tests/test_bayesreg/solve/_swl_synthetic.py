"""Synthetic problems with structure channels for the sigma_domain tests (work package SWLI; contract C04-6, C11-8).

No data/ access.  The intensity channel is that of solve/_synthetic.py (plane waves, fixed = G(x + u*)).  K_S structure
channels in square-root coordinates (C03.5): moving-side probabilities P_M^k(x) = A_k exp(-|x - c_k|^2 / (2 s_k^2)),
k = 1..K_S, background P^0 = 1 - sum_k P^k; fixed side P_F^k(x) = P_M^k(x + u*(x)) (the analytic blobs at the true map);
both floored (P + eps)/(1 + (K_S + 1) eps), eps = 1e-3, then sqrt.  A blob with a small amplitude (default the last one,
A = .01) stays below theta = .05 on both sides (an empty support).
"""
import numpy as np

from bayesreg.grids import four_level_grids
from bayesreg.solve import LevelInput

from ._synthetic import brain_box, cube_grid, moving_image, true_displacement, uniform_tissue, warp_with_spline

EPS = 1e-3
BLOBS = (((8., 9., 10.), 4., .6), ((16., 14., 12.), 3.5, .5), ((12., 17., 15.), 3., .01))   # (center xyz, s, A)


def blob_probabilities(points, blobs=BLOBS):
    """(K_S + 1, ...) floored probabilities (background first) of the analytic blobs at points (..., 3) xyz."""
    points = np.asarray(points)
    p = [a*np.exp(-np.sum((points-np.asarray(c))**2, axis=-1)/(2*s*s)) for c, s, a in blobs]
    p = np.stack([np.clip(1-np.sum(p, axis=0), 0., 1.)]+p)
    return (p+EPS)/(1+p.shape[0]*EPS)


def structure_level(ell, grid, u_true, blobs=BLOBS, margin=1):
    X = grid.identity()
    moving_int = moving_image(X)
    fixed_int = warp_with_spline(moving_int, grid, u_true)
    s_moving = np.sqrt(blob_probabilities(X, blobs))[1:]
    s_fixed = np.sqrt(blob_probabilities(X+u_true, blobs))[1:]
    fixed = np.concatenate([fixed_int[None], s_fixed])
    moving = np.concatenate([moving_int[None], s_moving])
    return LevelInput(level=ell, grid=grid, fixed=fixed, moving=moving, mask=brain_box(grid, margin),
                      tissue=uniform_tissue(grid))


def four_levels_structure(n=26, amplitude=(.7, -.6, .5), blobs=BLOBS):
    """Four LevelInput (coarse to native) with 1 + K_S channels; returns (native grid, levels, u* native)."""
    native = cube_grid(n)
    out = []
    for ell, grid in enumerate(four_level_grids(native)):
        out.append(structure_level(ell, grid, true_displacement(grid, amplitude), blobs, 2 if ell == 3 else 1))
    return native, tuple(out), true_displacement(native, amplitude)


def single_level_structure(n=12, amplitude=(.5, -.4, .3), blobs=None, margin=1, h=2.):
    """One level (grid, LevelInput, u*) with structure channels; blobs scaled to the grid size."""
    grid = cube_grid(n, h)
    if blobs is None:
        L = (n-1)*h
        blobs = (((.35*L, .4*L, .45*L), .18*L, .6), ((.65*L, .55*L, .5*L), .15*L, .5), ((.5*L, .7*L, .6*L), .1*L, .01))
    u = true_displacement(grid, amplitude)
    return grid, structure_level(3, grid, u, blobs, margin), u
