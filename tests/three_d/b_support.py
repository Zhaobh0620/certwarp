"""Independent small dense oracle for B; never imported by production code."""
from itertools import product

import numpy as np

from jd_curl.three_d.boundary import restrict_free
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.problem import make_problem, Weights3D, MemoryConfig
from jd_curl.three_d.residuals import evaluate_trial, promote_trial, AcceptedState3D
from jd_curl.three_d.linearization import build_linearization


def image(points):
    x, y, z = np.moveaxis(points, -1, 0)
    return .3+.2*x-.13*y+.17*z+.07*x*y*z+.04*x*x*y-.03*y*y*z+.02*z**3


def image_gradient(points):
    x, y, z = np.moveaxis(points, -1, 0)
    return np.stack((.2+.07*y*z+.08*x*y,
                     -.13+.07*x*z+.04*x*x-.06*y*z,
                     .17+.07*x*y-.03*y*y+.06*z*z), axis=-1)


def fields(grid):
    z, y, x = np.meshgrid(*(np.linspace(0, 1, n) for n in grid.shape), indexing="ij")
    s = np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
    ref = .014*s[..., None]*np.stack((1+.4*y, -.6+.3*z, .8-.2*x), axis=-1)
    u = ref+.011*s[..., None]*np.stack((.8+.2*z, .7-.4*x, -.5+.3*y), axis=-1)
    for a in (ref, u):
        a[[0, -1], :, :] = 0
        a[:, [0, -1], :] = 0
        a[:, :, [0, -1]] = 0
    return ref, u


def case(*, mu=1., nu=None, tile=64, chunk=31, shape=(5, 6, 7)):
    nz, ny, nx = shape
    grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=.2, hy=.3, hz=.4, x0=-.3, y0=.2, z0=-.7, unit="mm")
    ref, u = fields(grid)
    moving = image(grid.identity())
    fixed = moving+.025+.01*np.cos(grid.identity()[..., 0])
    problem = make_problem(grid=grid, moving_image=moving, fixed_image=fixed, u_ref=ref,
                           weights=Weights3D(.13, mu, mu if nu is None else nu, .017),
                           memory=MemoryConfig(tile, chunk))
    state = promote_trial(problem, evaluate_trial(problem, restrict_free(u, grid)))
    assert isinstance(state, AcceptedState3D)
    return problem, state, build_linearization(problem, state)


def pack(blocks):
    """Mathematical R0 order: image, prox xyz, JD, curl xyz, cell then point."""
    data = {k: [] for k in ("image", "proximity", "jd", "curl")}
    for b in blocks:
        data[b.name].append(b.values)
    return np.concatenate((data["image"][0].ravel(),
                           np.moveaxis(data["proximity"][0], -1, 0).ravel(),
                           np.concatenate(data["jd"]).ravel(),
                           np.moveaxis(np.concatenate(data["curl"]), -1, 0).ravel()))


def dense_derivatives(grid, *, corners=False):
    """Global physical shape matrices built from scalar Q1 formulas, not A ops."""
    if corners:
        nodes, weights = (0., 1.), (.5, .5)
    else:
        xi, wi = np.polynomial.legendre.leggauss(3)
        nodes, weights = (xi+1)/2, wi/2
    points = [(x, y, z) for z, y, x in product(nodes, repeat=3)]
    qw = np.array([grid.omega*x*y*z for z, y, x in product(weights, repeat=3)])
    D = np.zeros((3, grid.n_cells*len(points), np.prod(grid.shape)))
    for c, (cz, cy, cx) in enumerate(np.ndindex(grid.cell_shape)):
        for a, (az, ay, ax) in enumerate(product((0, 1), repeat=3)):
            node = np.ravel_multi_index((cz+az, cy+ay, cx+ax), grid.shape)
            vertex = (ax, ay, az)
            for k, point in enumerate(points):
                for j in range(3):
                    v = (2*vertex[j]-1)/grid.spacing[j]
                    for other in range(3):
                        if other != j:
                            v *= point[other] if vertex[other] else 1-point[other]
                    D[j, c*len(points)+k, node] = v
    return D, np.tile(qw, grid.n_cells)


def dense_geometry(D, u, S):
    F = np.einsum("jqn,nr->qrj", D, u.reshape(-1, 3))+np.eye(3)
    A = np.empty_like(F)
    for r, j in product(range(3), repeat=2):
        rr, jj = [k for k in range(3) if k != r], [k for k in range(3) if k != j]
        A[:, r, j] = (-1.)**(r+j)*(F[:, rr[0], jj[0]]*F[:, rr[1], jj[1]]-F[:, rr[0], jj[1]]*F[:, rr[1], jj[0]])
    DS = D@S
    L = np.concatenate([sum(A[:, r, j, None]*DS[j] for j in range(3)) for r in range(3)], axis=1)
    zero = np.zeros_like(DS[0])
    C = np.stack((F[:, 2, 1]-F[:, 1, 2], F[:, 0, 2]-F[:, 2, 0], F[:, 1, 0]-F[:, 0, 1]), axis=-1)
    LC = np.concatenate((np.concatenate((zero, -DS[2], DS[1]), axis=1),
                          np.concatenate((DS[2], zero, -DS[0]), axis=1),
                          np.concatenate((-DS[1], DS[0], zero), axis=1)))
    return np.linalg.det(F), C, L, LC


def oracle(problem, state):
    g, w = problem.grid, problem.weights
    n = int(np.prod(g.shape))
    interior = np.arange(n).reshape(g.shape)[1:-1, 1:-1, 1:-1].ravel()
    S = np.eye(n)[:, interior]
    q = np.ones(g.shape)
    for axis in range(3):
        sl = [slice(None)]*3
        sl[axis] = [0, -1]
        q[tuple(sl)] *= .5
    q = q.ravel()*g.omega
    b = image_gradient(g.identity()+state.u).reshape(-1, 3)
    D, qg = dense_derivatives(g)
    Dc, qc = dense_derivatives(g, corners=True)
    J, C, L, LC = dense_geometry(D, state.u, S)
    Jr, Cr, _, _ = dense_geometry(D, problem.controls.u_ref, S)
    Jb, _, Lb, _ = dense_geometry(Dc, state.u, S)
    KI = np.concatenate([np.sqrt(q)[:, None]*b[:, r, None]*S for r in range(3)], axis=1)
    KP = np.kron(np.eye(3), np.sqrt(w.lam*q)[:, None]*S)
    K = np.concatenate((KI, KP, np.sqrt(w.mu*qg)[:, None]*L,
                         np.tile(np.sqrt(w.nu*qg), 3)[:, None]*LC))
    R = np.concatenate((np.sqrt(q)*(image(g.identity()+state.u)-problem.fixed_image).ravel(),
                         (np.sqrt(w.lam*q)[None, :]*np.moveaxis(state.u-problem.controls.u_ref, -1, 0).reshape(3, n)).ravel(),
                         np.sqrt(w.mu*qg)*(J-Jr),
                         (np.sqrt(w.nu*qg)[None, :]*(C-Cr).T).ravel()))
    gap, scale = Jb-problem.certificate_config.delta, 1-problem.certificate_config.delta
    first = w.kappa*qc*(Jb-1)/(scale*gap)
    curvature = w.kappa*qc/gap**2
    H = K.T@K+Lb.T@(curvature[:, None]*Lb)
    gradient = K.T@R+Lb.T@first
    return R, K, gradient, H
