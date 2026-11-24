"""E3 of EF: the numerical check of runs/20261104-ef/repair_推导.md (verify-math procedure).  CPU, FP64, a small synthetic grid; no real image, label or field.  The output goes to runs/20261104-ef/check_repair_gradient.out.

    python -B experiments/ef/check_repair_gradient.py

1. J_q of repair.py against bayesreg.geom.tile_geometry (the model's J) at Gauss27 and closed27: largest difference.
2. The gradient of repair.Objective against central differences of an INDEPENDENT energy written with bayesreg.geom (not with repair.py's J): every free coordinate on a small mask, and random directions on a larger one; several rho, eps; with active and without active points; with extra lattice points (a third energy, from the Q1 formula).
3. The derivative of J alone: dJ_q/dv_(a,r) = sum_j A_rj D_jqa, one point, every coordinate, against central differences.
4. Sampling is not a proof: a cell whose J is >= .016 on all its 53 points and which the certificate's own classification calls infeasible.
5. J and dJ/dv of one Q1 cell against sympy, from the map itself.
6. Taylor: slope 2 of the first-order remainder of E.
Exit 0 only if every check passes.
"""
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1]))

from bayesreg.geom import closed27, gauss27, node_field, tile_geometry  # noqa: E402
from jd_curl.three_d.feasibility import PrescreenCertificateConfig, certify_map  # noqa: E402
from jd_curl.three_d.grid import make_grid_3d  # noqa: E402

import repair  # noqa: E402

FD_STEP = 1e-6
TOL_J, TOL_GRAD = 1e-13, 2e-6        # J: rounding; gradient: relative, central differences of a cubic-times-kink function
SEED = 20261104


def small_grid():
    return make_grid_3d(nx=9, ny=8, nz=7, hx=1., hy=1.25, hz=.75, unit='mm')


def rough_field(grid, amplitude, seed=SEED):
    """Random nodes with zero faces: large enough to fold some cells."""
    rng = np.random.default_rng(seed)
    u = np.zeros(grid.shape+(3,))
    u[1:-1, 1:-1, 1:-1] = amplitude*rng.normal(size=(grid.shape[0]-2, grid.shape[1]-2, grid.shape[2]-2, 3))
    return u


def geom_j(u, grid):
    """J at Gauss27 and closed27 without the body centre, every cell, through bayesreg.geom: shape (cells, 53)."""
    field = node_field(u, grid)
    all_cells = slice(0, grid.n_cells)
    a = tile_geometry(field, gauss27(grid), tile=all_cells).J
    ops = closed27(grid)
    keep = [i for i, p in enumerate(ops.points) if not np.all(p == .5)]
    return np.concatenate((a, tile_geometry(field, ops, tile=all_cells).J[:, keep]), axis=1)


def geom_energy(v, u, grid, free, cells, rho, eps):
    """E written with bayesreg.geom, independently of repair.py: the cells of the mask, the 53 points; x - u on the free nodes."""
    gap = np.clip(eps-geom_j(v, grid)[cells], 0., None)
    d = v.reshape(-1, 3)[free]-u.reshape(-1, 3)[free]
    return .5*float((d*d).sum())+.5*rho*float((gap*gap).sum())


def check_j(out):
    grid = small_grid()
    u = rough_field(grid, .3)
    _, table = repair.union53(grid)
    free = repair.mask_nodes(np.arange(grid.n_cells), grid, 0)
    objective = repair.Objective(torch, u, u, grid, free, table, device='cpu', tile_cells=64)
    X = objective.base.clone()
    V = X[objective.idx]
    mine, _ = repair._jacobian(torch, torch.matmul(objective.D, V[:, None]).transpose(-1, -2)+objective.eye)
    theirs = geom_j(u, grid)
    assert list(objective.cells) == list(range(grid.n_cells))
    worst = float(np.abs(mine.numpy()-theirs).max())
    out(f'1. J at 53 points, repair.py against bayesreg.geom: {theirs.shape}, largest difference {worst:.3e} (limit {TOL_J:g}), min J {theirs.min():.3f}, share of points below 0: {(theirs < 0).mean():.3f}')
    return worst <= TOL_J


def fd_gradient(fun, x, step=FD_STEP):
    out = np.empty(x.size)
    for i in range(x.size):
        e = np.zeros(x.size)
        e[i] = step
        out[i] = (fun(x+e)-fun(x-e))/(2*step)
    return out


def check_gradient(out):
    grid = small_grid()
    ok = True
    _, table = repair.union53(grid)
    for amplitude, rho, eps, seed_cells, label in ((.3, 1e2, .05, [120, 121], 'active, small mask'), (.3, 1e4, .01, [100], 'active, rho 1e4'), (.01, 1e2, .05, [120], 'smooth field: nothing active')):
        u = rough_field(grid, amplitude)
        free = repair.mask_nodes(np.array(seed_cells), grid, 0)
        start = u.copy().reshape(-1, 3)                                                 # the point of evaluation is not the anchor: both terms of the gradient are seen
        rng = np.random.default_rng(SEED+2)
        start[free] += .05*rng.normal(size=(len(free), 3))
        start = start.reshape(u.shape)
        objective = repair.Objective(torch, u, start, grid, free, table, device='cpu', tile_cells=7)
        x0 = objective.start
        energy, g, stats = objective.value_and_grad(x0, rho, eps)
        cells = objective.cells

        def as_field(flat):
            v = start.copy().reshape(-1, 3)
            v[free] = flat.reshape(-1, 3)
            return v.reshape(u.shape)

        def energy_geom(flat):
            return geom_energy(as_field(flat), u, grid, free, cells, rho, eps)
        same = abs(energy-energy_geom(x0.numpy().ravel()))/max(1., abs(energy))
        numeric = fd_gradient(energy_geom, x0.numpy().ravel())
        analytic = g.numpy().ravel()
        relative = float(np.abs(numeric-analytic).max()/max(1., np.abs(analytic).max()))
        passed = relative <= TOL_GRAD and same <= 1e-13
        ok = ok and passed
        out(f"2a. {label}: amplitude {amplitude}, rho {rho:g}, eps {eps}: {len(free)} free nodes, {len(cells)} cells, {stats['active']} active points, energy against geom {same:.2e}, gradient every coordinate: largest |FD - analytic| / max(1, |g|_inf) {relative:.2e} (limit {TOL_GRAD:g}); |g|_inf {np.abs(analytic).max():.3e}: {'ok' if passed else 'FAILED'}")
    # random directions on the larger mask, with extra lattice points; the energy from repair.py itself but a directional difference
    u = rough_field(grid, .3)
    free = repair.mask_nodes(np.arange(grid.n_cells), grid, 1)
    points, on_closed = repair._lattice()
    rng = np.random.default_rng(SEED+3)
    off = np.flatnonzero(~on_closed)
    extras = (rng.integers(0, grid.n_cells, 40), points[rng.choice(off, 40)])
    objective = repair.Objective(torch, u, u, grid, free, table, device='cpu', tile_cells=11, extras=extras)
    x0 = objective.start.clone()+.03*torch.as_tensor(rng.normal(size=(len(free), 3)))
    _, g, stats = objective.value_and_grad(x0, 1e3, .05)
    worst = 0.
    for k in range(20):
        d = torch.as_tensor(rng.normal(size=(len(free), 3)))
        plus = objective.value_and_grad(x0+FD_STEP*d, 1e3, .05, grad=False)[0]
        minus = objective.value_and_grad(x0-FD_STEP*d, 1e3, .05, grad=False)[0]
        slope = float((g*d).sum())
        worst = max(worst, abs((plus-minus)/(2*FD_STEP)-slope)/max(1., abs(slope)))
    passed = worst <= TOL_GRAD
    ok = ok and passed
    out(f"2b. with {len(extras[0])} extra lattice points, {len(free)} free nodes, {len(objective.cells)} cells, {stats['active']} active points, 20 random directions: largest relative difference {worst:.2e} (limit {TOL_GRAD:g}): {'ok' if passed else 'FAILED'}")
    # the extra points' energy from the Q1 formula, written separately (bayesreg.geom has no arbitrary points)
    table_e = repair.q1_derivative_table(extras[1], grid.spacing)
    nodes = repair.cell_nodes(grid, extras[0])
    flat = u.copy().reshape(-1, 3)
    flat[free] = x0.numpy()
    V = flat[nodes]
    F = np.eye(3)+np.einsum('jma,mar->mrj', table_e, V)
    direct = np.linalg.det(F)
    X = objective.base.clone()
    X[objective.free_pos] = x0
    mine, _ = repair._jacobian(torch, torch.matmul(objective.extra_D, X[objective.extra_idx]).transpose(-1, -2)+objective.eye)
    worst_j = float(np.abs(mine.numpy()-direct).max())
    closed = closed27(grid)
    table_ok = np.array_equal(repair.q1_derivative_table(closed.points, grid.spacing), closed.coefficients)
    passed = worst_j <= 1e-12 and table_ok
    ok = ok and passed
    out(f"2c. J at the extra points against numpy.linalg.det: largest difference {worst_j:.2e}; q1_derivative_table equals the closed27 table of jd_curl bit for bit: {table_ok}: {'ok' if passed else 'FAILED'}")
    return ok


def check_dj(out):
    """dJ_q/dv_(a,r) = sum_j A_rj D_jqa on one cell and one point, every (a, r)."""
    grid = small_grid()
    rng = np.random.default_rng(SEED+4)
    points, table = repair.union53(grid)
    worst = 0.
    for q in (0, 13, 40, 52):
        V = .2*rng.normal(size=(8, 3))
        D = table[:, q]                                                                 # (3, 8)
        F = np.eye(3)+np.einsum('ja,ar->rj', D, V)
        A = np.linalg.det(F)*np.linalg.inv(F).T
        for a in range(8):
            for r in range(3):
                e = np.zeros((8, 3))
                e[a, r] = FD_STEP
                numeric = (np.linalg.det(np.eye(3)+np.einsum('ja,ar->rj', D, V+e))-np.linalg.det(np.eye(3)+np.einsum('ja,ar->rj', D, V-e)))/(2*FD_STEP)
                worst = max(worst, abs(numeric-np.dot(A[r], D[:, a])))
    out(f'3. dJ_q/dv_(a,r) = sum_j A_rj D_jqa: four points, all 24 coordinates: largest difference {worst:.2e} (limit {TOL_GRAD:g})')
    return worst <= TOL_GRAD


def sampled_positive_cell():
    """(u, grid, cell): a field whose cell (0, 1, 1) has J = (1 + a z)(1 + b z) with a = -1/.52, b = -1/.86: on that cell's 53 points z is one of 0, .1127, .5, .8873, 1 and J >= .016; between z = .52 and z = .86 J < 0.  u_x = a x z, u_y = b y z on the cell.  The bottom nodes are on the face z = 0, where u = 0."""
    grid = make_grid_3d(nx=6, ny=6, nz=6, hx=1., hy=1., hz=1., unit='mm')
    a, b = -1/.52, -1/.86
    u = np.zeros(grid.shape+(3,))
    for y in (1, 2):
        for x in (1, 2):
            u[1, y, x, 0] = a*(x-1)
            u[1, y, x, 1] = b*(y-1)
    return u, grid, np.ravel_multi_index((0, 1, 1), grid.cell_shape)


def check_counterexample(out):
    """Positivity on the 53 points is not a proof: the cell above, by the certificate's own per-cell classification, and the whole field by certify_map."""
    from jd_curl.three_d import interval as iv
    from jd_curl.three_d.feasibility import _classify, _sample_jacobians, jacobian_intervals
    u, grid, cell = sampled_positive_cell()
    points, table = repair.union53(grid)
    V = u.reshape(-1, 3)[repair.cell_nodes(grid, np.array([cell]))]
    J53, _ = repair._jacobian(torch, torch.as_tensor(table.transpose(1, 0, 2)@V[0]).transpose(-1, -2)+torch.eye(3, dtype=torch.float64))
    lattice, _ = repair._lattice(33)
    Jl, _ = repair._jacobian(torch, torch.as_tensor(repair.q1_derivative_table(lattice, grid.spacing).transpose(1, 0, 2)@V[0]).transpose(-1, -2)+torch.eye(3, dtype=torch.float64))
    config = PrescreenCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=65536)
    p = jacobian_intervals(V, grid.spacing)
    verdict = _classify(iv.lagrange_to_bernstein(p)[0], p[0], 0., float(_sample_jacobians(V, grid.spacing).min()), config)
    whole = certify_map(u, grid, config=config)
    passed = float(J53.min()) >= .01 and float(Jl.min()) < 0 and verdict.status == 'infeasible' and whole.status != 'certified_feasible'
    out(f"4. a cell with min J over its 53 points {float(J53.min()):.4f} >= .01, min J on a 33^3 lattice {float(Jl.min()):.4f}: the certificate's classification of that cell says {verdict.status} ({verdict.reason}); certify_map on the field, whose other cells also fold: {whole.status}, {whole.n_infeasible} infeasible: {'as the derivation says, ok' if passed else 'FAILED'}")
    return passed


def check_sympy(out):
    """J and dJ of one Q1 cell from the map itself, symbolically (no D table, no cofactor): x(xi) = xi h + sum_a N_a(xi) v_a; F = dx/dX; J = det F; dJ/dv_(a,r) by sympy.diff; against q1_derivative_table, _jacobian and the flux of Objective._block at random vertices and points."""
    import sympy as sp
    h = (1., 1.25, .75)
    v = sp.symbols('v0:24')
    rng = np.random.default_rng(SEED+5)
    worst_j = worst_d = 0.
    for _ in range(6):
        point = rng.uniform(0, 1, 3)
        xi = sp.symbols('xi0 xi1 xi2')
        N = []
        for vertex in repair.VERTICES:
            term = sp.Integer(1)
            for j in range(3):
                term *= xi[j] if vertex[j] else 1-xi[j]
            N.append(term)
        u = [sum(N[a]*v[3*a+r] for a in range(8)) for r in range(3)]
        F = sp.Matrix(3, 3, lambda r, j: (1 if r == j else 0)+sp.diff(u[r], xi[j]).subs(dict(zip(xi, point)))/h[j])     # at this point: a cubic in the 24 v
        J = F.det(method='berkowitz')
        fn = sp.lambdify([v], [J]+[sp.diff(J, s) for s in v], 'numpy')
        V = .3*rng.normal(size=(8, 3))
        values = np.array(fn(V.ravel()), dtype=float)
        D = repair.q1_derivative_table(point[None], h)[:, 0]                            # (3, 8)
        F_t = torch.as_tensor(np.eye(3)+D@V).T.clone()                                  # (D V)[j, r] = d_j v_r, F[r, j]
        mine, A = repair._jacobian(torch, F_t)
        worst_j = max(worst_j, abs(float(mine)-float(values[0])))
        analytic = A.numpy()@D                                                          # sum_j A_rj D_ja: (3, 8)
        worst_d = max(worst_d, float(np.abs(analytic-values[1:].reshape(8, 3).T).max()))
    passed = worst_j <= 1e-12 and worst_d <= 1e-12
    out(f"5. sympy, one Q1 cell, spacing {h}, 6 random vertices and points: |J - J_sympy| {worst_j:.2e}, |sum_j A_rj D_ja - dJ/dv_sympy| {worst_d:.2e} (limit 1e-12): {'ok' if passed else 'FAILED'}")
    return passed


def check_taylor(out):
    """|E(x + t p) - E(x) - t <g, p>| against t: slope 2 where neither the truncation's cubic term nor rounding dominates.  E is C^1 and piecewise C^2 (the kink of [.]_+^2 is in the second derivative), so 2 is what to expect."""
    grid = small_grid()
    u = rough_field(grid, .3)
    _, table = repair.union53(grid)
    free = repair.mask_nodes(np.arange(grid.n_cells), grid, 1)
    objective = repair.Objective(torch, u, u, grid, free, table, device='cpu', tile_cells=64)
    rng = np.random.default_rng(SEED+6)
    x0 = objective.start.clone()+.02*torch.as_tensor(rng.normal(size=(len(free), 3)))
    p = torch.as_tensor(rng.normal(size=(len(free), 3)))
    ts = 10.**-np.arange(1, 8)
    ok = True
    for rho, eps in ((1e2, .05), (1e4, .01)):
        e0, g, _ = objective.value_and_grad(x0, rho, eps)
        d = float((g*p).sum())
        errs = np.array([abs(objective.value_and_grad(x0+t*p, rho, eps, grad=False)[0]-e0-t*d) for t in ts])
        slopes = np.diff(np.log(errs))/np.diff(np.log(ts))
        middle = slopes[2:4]                                                            # t = 1e-3 .. 1e-5
        passed = bool(np.all(np.abs(middle-2) < .1))
        ok = ok and passed
        out(f"6. Taylor, rho {rho:g}, eps {eps}: errors {[f'{e:.1e}' for e in errs]}, slopes {[f'{s:.2f}' for s in slopes]}: {'ok' if passed else 'FAILED'}")
    return ok


def main():
    torch.set_num_threads(1)

    def out(text):
        print(text, flush=True)
    print(f'EF check of the repair (numpy {np.__version__}, torch {torch.__version__}, CPU FP64, FD step {FD_STEP:g}, seed {SEED})')
    results = [check_j(out), check_gradient(out), check_dj(out), check_counterexample(out), check_sympy(out), check_taylor(out)]
    print(f'all passed: {all(results)}')
    return 0 if all(results) else 8


if __name__ == '__main__':
    sys.exit(main())
