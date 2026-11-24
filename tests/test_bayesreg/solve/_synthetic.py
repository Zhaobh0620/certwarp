"""Synthetic registration problems for the bayesreg.solve tests (no data/ access).

Moving image: a sum of plane waves in several directions (wavelength 5-8 mm) under a smooth envelope,
so the local structure tensor is full rank over a few voxels (the aperture problem is only local: the
data term constrains each node along grad M only; the prior couples neighbours).  Known displacement
u*: a smooth bump per component, zero on the six faces, amplitude <= 1 voxel.  Fixed image
F(x_a) = G(x_a + u*(x_a)) with G the not-a-knot cubic spline of the sampled moving image -- the same
sampler the model uses -- so the noise-free data residual is exactly zero at u = u* (nodal values).
"""
import numpy as np

from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.interpolation import build_image_sampler, sample_points

from bayesreg.grids import four_level_grids
from bayesreg.solve import LevelInput

WAVES = (((1., .2, .1), 6.), ((-.3, 1., .25), 7.), ((.15, -.35, 1.), 5.5), ((.7, .7, -.4), 8.),
         ((-.6, .3, .75), 6.5))


def cube_grid(n, h=1., origin=(0., 0., 0.)):
    return make_grid_3d(nx=n, ny=n, nz=n, hx=h, hy=h, hz=h, x0=origin[0], y0=origin[1], z0=origin[2], unit="mm")


def moving_image(points):
    """sum of unit-amplitude plane waves (directions normalized), plus a constant offset; points (..., 3) xyz."""
    points = np.asarray(points)
    value = np.full(points.shape[:-1], 1.)
    for k, (direction, wavelength) in enumerate(WAVES):
        n = np.asarray(direction)/np.linalg.norm(direction)
        value = value+.25*np.sin(2*np.pi*(points@n)/wavelength+.7*k)
    return value


def true_displacement(grid, amplitude=(.7, -.6, .5)):
    """u* (nz, ny, nx, 3) mm: A_r sin(pi tx) sin(pi ty) sin(pi tz) times a component-dependent modulation;
    exactly zero on the faces."""
    t = [(axis-axis[0])/(axis[-1]-axis[0]) for axis in grid.axes]
    tz, ty, tx = np.meshgrid(t[2], t[1], t[0], indexing="ij")
    bump = np.sin(np.pi*tx)*np.sin(np.pi*ty)*np.sin(np.pi*tz)
    u = np.stack([amplitude[0]*bump*(1+.3*np.cos(np.pi*ty)), amplitude[1]*bump*(1+.3*np.sin(np.pi*tz)),
                  amplitude[2]*bump*(1+.3*np.cos(np.pi*tx))], axis=-1)
    for face in (0, -1):
        u[face], u[:, face], u[:, :, face] = 0., 0., 0.
    return u


def solenoidal_displacement(grid, amplitude=.5):
    """u* = curl psi (nz, ny, nx, 3) mm with psi_r = c_r b(x), b = prod_j sin^2(pi t_j) / max-normalized:
    divergence-free in the continuum (log J = O(|grad u|^2)), zero with zero derivative on the faces, so
    the log-volume prior hardly biases the minimizer; max |u*| = amplitude."""
    t = [(axis-axis[0])/(axis[-1]-axis[0]) for axis in grid.axes]
    tz, ty, tx = np.meshgrid(t[2], t[1], t[0], indexing="ij")
    lengths = [axis[-1]-axis[0] for axis in grid.axes]
    s = [np.sin(np.pi*v)**2 for v in (tx, ty, tz)]
    ds = [np.pi*np.sin(2*np.pi*v)/L for v, L in zip((tx, ty, tz), lengths)]       # d/dx_j of sin^2(pi t_j)
    grad_b = [ds[0]*s[1]*s[2], s[0]*ds[1]*s[2], s[0]*s[1]*ds[2]]
    c = (.6, -.8, 1.)                                                               # psi = c b
    u = np.stack([c[2]*grad_b[1]-c[1]*grad_b[2], c[0]*grad_b[2]-c[2]*grad_b[0], c[1]*grad_b[0]-c[0]*grad_b[1]], axis=-1)
    u *= amplitude/np.max(np.linalg.norm(u, axis=-1))
    for face in (0, -1):
        u[face], u[:, face], u[:, :, face] = 0., 0., 0.
    return u


def warp_with_spline(moving, grid, u):
    sampler = build_image_sampler(moving, grid)
    points = (grid.identity()+u).reshape(-1, 3)
    sample = sample_points(sampler, points)
    return sample.value.reshape(grid.shape)


def uniform_tissue(grid, n_classes=4):
    """A constant floored simplex (K_T + 1 = 4 channels): background .1, three tissue classes .3."""
    values = np.array([.1]+[.9/(n_classes-1)]*(n_classes-1))
    return np.broadcast_to(values[:, None, None, None], (n_classes,)+grid.shape).copy()


def brain_box(grid, margin=2):
    mask = np.zeros(grid.shape, dtype=bool)
    mask[margin:-margin, margin:-margin, margin:-margin] = True
    return mask


def single_level(n=17, h=1., amplitude=(.7, -.6, .5)):
    """(grid, LevelInput, u*) of one level: intensity-only."""
    grid = cube_grid(n, h)
    moving = moving_image(grid.identity())
    u_true = true_displacement(grid, amplitude)
    fixed = warp_with_spline(moving, grid, u_true)
    level = LevelInput(level=3, grid=grid, fixed=fixed[None], moving=moving[None], mask=brain_box(grid),
                       tissue=uniform_tissue(grid))
    return grid, level, u_true


def four_levels(n=26, amplitude=(.7, -.6, .5)):
    """Four LevelInput on four_level_grids(native): the analytic moving image and the analytic u* sampled
    on every level; the fixed image of each level is G_ell(x + u*(x)) with that level's spline."""
    native = cube_grid(n)
    out = []
    for ell, grid in enumerate(four_level_grids(native)):
        moving = moving_image(grid.identity())
        u_true = true_displacement(grid, amplitude)
        fixed = warp_with_spline(moving, grid, u_true)
        margin = 2 if ell == 3 else 1
        out.append(LevelInput(level=ell, grid=grid, fixed=fixed[None], moving=moving[None],
                              mask=brain_box(grid, margin), tissue=uniform_tissue(grid)))
    return native, tuple(out), true_displacement(native, amplitude)
