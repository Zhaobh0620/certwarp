"""Small synthetic model states and an independent dense oracle for the bayesreg.model tests.

No data/ access.  Moving channels are tensor cubic polynomials (degree <= 3 in each variable), which the
not-a-knot cubic spline reproduces exactly, so the oracle uses their analytic values and gradients.
The dense oracle follows tests/three_d/b_support.py: shape-function matrices from numpy's
Gauss-Legendre rule and the scalar Q1 formula, cofactor from signed 2x2 minors, explicit K_i;
nothing in it calls CellQ1Operators3D or bayesreg.
"""
from itertools import product

import numpy as np

from jd_curl.three_d.boundary import inject_free, restrict_free
from jd_curl.three_d.grid import make_grid_3d

from bayesreg.geom import TILE_SIZE
from bayesreg.model import ModelState, build_samplers

TAU2 = (1., .25, .36, .16)          # tau_k^2, k = 0 background (K_T = 3)
SIGMA2 = (.04, .09, .02)            # sigma_c^2, c = 0 intensity


def aniso(nx=7, ny=6, nz=5):
    """The mainline tests/three_d grid3d fixture (7x6x5 nodes, anisotropic, off-origin)."""
    return make_grid_3d(nx=nx, ny=ny, nz=nz, hx=.2, hy=.3, hz=.4, x0=-.3, y0=.2, z0=-.7, unit="mm")


def poly_images(points):
    x, y, z = np.moveaxis(np.asarray(points), -1, 0)
    return np.stack((.3+.2*x-.13*y+.17*z+.07*x*y*z+.04*x*x*y-.03*y*y*z+.02*z**3,
                     .5-.1*x+.25*y-.05*z+.03*x**3-.06*x*z*z+.05*x*y*z,
                     .2+.15*x+.1*y-.2*z+.04*y**3+.05*x*x*z-.02*x*y))


def poly_gradients(points):
    """(C, ..., 3) analytic xyz gradients of poly_images."""
    x, y, z = np.moveaxis(np.asarray(points), -1, 0)
    return np.stack((
        np.stack((.2+.07*y*z+.08*x*y, -.13+.07*x*z+.04*x*x-.06*y*z, .17+.07*x*y-.03*y*y+.06*z*z), axis=-1),
        np.stack((-.1+.09*x*x-.06*z*z+.05*y*z, .25+.05*x*z, -.05-.12*x*z+.05*x*y), axis=-1),
        np.stack((.15+.1*x*z-.02*y, .1+.12*y*y-.02*x, -.2+.05*x*x), axis=-1)))


def random_simplex(rng, n_classes, shape):
    p = rng.random((n_classes,)+shape)+.05
    return p/p.sum(axis=0)


def make_state(grid, *, seed=0, n_channels=3, omega0=None, alpha=.13, nu=.07, tau2=TAU2, sigma2=SIGMA2,
               student_t=False, mask=None, tile_size=TILE_SIZE, moving=None, fixed=None, tissue=None):
    rng = np.random.default_rng(seed)
    X = grid.identity()
    if moving is None:
        moving = poly_images(X)[:n_channels]
    if fixed is None:
        fixed = moving+.025+.01*np.cos(3*X[..., 0])[None]-.02*np.sin(2*X[..., 1]+X[..., 2])[None]
    if tissue is None:
        tissue = random_simplex(rng, len(tau2), grid.shape)
    if mask is None:
        mask = rng.random(grid.shape) < .75
        mask[2, 2, 2] = True
        mask[1, 1, 1] = False              # an interior node outside the mask
    weights = rng.uniform(.3, 1.2, (moving.shape[0],)+grid.shape) if student_t else None
    return ModelState(grid=grid, omega0=grid.omega/3 if omega0 is None else omega0, fixed=fixed,
                      samplers=build_samplers(moving, grid), mask=mask, sigma2=np.asarray(sigma2[:moving.shape[0]]),
                      alpha=alpha, nu=nu, tau2=np.asarray(tau2), tissue=tissue, weights=weights, tile_size=tile_size)


def free_random(grid, rng, scale=1.):
    return scale*rng.normal(size=3*grid.n_free_scalar)


def feasible_field(grid, rng, scale=.08):
    """A random U0 node field with |u| ~ scale * min h (Gauss J well above 0)."""
    return inject_free(free_random(grid, rng, scale*min(grid.spacing)), grid)


def relmax(a, b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b)))/max(1., np.max(np.abs(b))))


# ----------------------------------------------------------------------------------------------
# Dense oracle
# ----------------------------------------------------------------------------------------------

def dense_operators(grid):
    """D (3, |Q|, N) physical shape derivatives, B (|Q|, N) shape values and w (|Q|,) = omega w_g at the
    Gauss27 points (z, y, x product, x fastest; cells in C order), from numpy's Gauss-Legendre rule."""
    xi, wi = np.polynomial.legendre.leggauss(3)
    nodes, weights = (xi+1)/2, wi/2
    points = [(px, py, pz) for pz, py, px in product(nodes, repeat=3)]
    qw = np.array([grid.omega*a*b*c for c, b, a in product(weights, repeat=3)])
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
    return D, B, np.tile(qw, grid.n_cells)


def selection(grid):
    """S (N, N_int): interior node i of the free scalar layout -> node index."""
    N = int(np.prod(grid.shape))
    interior = np.arange(N).reshape(grid.shape)[1:-1, 1:-1, 1:-1].ravel()
    return np.eye(N)[:, interior]


def cofactor(F):
    A = np.empty_like(F)
    for r, j in product(range(3), repeat=2):
        rr, jj = [k for k in range(3) if k != r], [k for k in range(3) if k != j]
        A[:, r, j] = (-1.)**(r+j)*(F[:, rr[0], jj[0]]*F[:, rr[1], jj[1]]-F[:, rr[0], jj[1]]*F[:, rr[1], jj[0]])
    return A


def dense_model(state, u, operators=None):
    """Residuals r_i and explicit Jacobians K_i (rows in the module's layouts, columns the free vector):
    channel (C, N) and (C, N, 3m); diffusion (9|Q|,) rows (q, r, j); logvol (|Q|,); curl (3|Q|,) rows
    (q, i).  Uses the analytic poly_images / poly_gradients for the moving side."""
    g = state.grid
    D, B, wq = dense_operators(g) if operators is None else operators
    S = selection(g)
    m = g.n_free_scalar
    U = np.asarray(u).reshape(-1, 3)
    C = state.n_channels
    X = g.identity().reshape(-1, 3)+U
    values, grads = poly_images(X)[:C], poly_gradients(X)[:C]
    q = g.node_weights().ravel()
    w = np.ones((C, q.size)) if state.weights is None else state.weights.reshape(C, -1)
    beta = np.sqrt(q*state.mask.ravel()*w/(state.omega0*state.sigma2[:, None]))
    r_ch = beta*(values-state.fixed.reshape(C, -1))
    K_ch = np.zeros((C, q.size, 3*m))
    for d in range(3):
        K_ch[:, :, d*m:(d+1)*m] = (beta*grads[..., d])[:, :, None]*S[None]
    DS = [D[j]@S for j in range(3)]
    F = np.einsum("jqn,nr->qrj", D, U)+np.eye(3)
    A, J = cofactor(F), np.linalg.det(F)
    tau2 = sum(t*(B@state.tissue[k].ravel()) for k, t in enumerate(state.tau2))
    sD, sJ, sC = np.sqrt(state.alpha*wq/state.omega0), np.sqrt(wq/(state.omega0*tau2)), np.sqrt(state.nu*wq/state.omega0)
    nq = len(wq)
    K_D = np.zeros((nq, 3, 3, 3*m))
    for r, j in product(range(3), repeat=2):
        K_D[:, r, j, r*m:(r+1)*m] = sD[:, None]*DS[j]
    L = np.concatenate([sum(A[:, r, j, None]*DS[j] for j in range(3)) for r in range(3)], axis=1)
    zero = np.zeros_like(DS[0])
    LC = np.stack((np.concatenate((zero, -DS[2], DS[1]), axis=1), np.concatenate((DS[2], zero, -DS[0]), axis=1),
                   np.concatenate((-DS[1], DS[0], zero), axis=1)), axis=1)
    curl = np.stack((F[:, 2, 1]-F[:, 1, 2], F[:, 0, 2]-F[:, 2, 0], F[:, 1, 0]-F[:, 0, 1]), axis=-1)
    return dict(channel=(r_ch, K_ch),
                diffusion=((sD[:, None, None]*(F-np.eye(3))).ravel(), K_D.reshape(9*nq, 3*m)),
                logvol=(sJ*np.log(J), (sJ/J)[:, None]*L),
                curl=((sC[:, None]*curl).ravel(), (sC[:, None, None]*LC).reshape(3*nq, 3*m)),
                tau2=tau2, J=J, wq=wq, B=B)


def theta_of(u, grid):
    return restrict_free(u, grid)
