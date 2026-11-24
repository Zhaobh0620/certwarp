"""E3 of EF (runs/20261104-ef/repair_推导.md, docs/bayesreg/notes/提速与换预分割器规划_20261104.md §6): local repair of a displacement field on the cells the certificate does not pass.

    min_v  1/2 ||v - u||^2 + (rho/2) sum_{c in C(M)} sum_{q in Q_c} [eps - J_{c,q}(v)]_+^2      v = u outside the free nodes M

Q_c = Gauss27 U closed27 (53 points; from the second round on also the points W of the dense lattice, see below).  No image and no label enters: this module sees a field and a grid.

The certificate (certify_map, passed in with its configuration) is the only verdict.  Nothing here ever says a field passes except by a certificate returned for exactly that array; positivity on the 53 points proves nothing (repair_推导.md §6).
Not a method and not in the contract: a probe of the first development pairs.  The certificate's semantics are not touched: jd_curl is only called.

The interface:

    cells = locate_cells(u, grid, certificate_config)                                      the cells the certificate's two fast proofs leave open: every cell it can fail is among them
    result = repair_field(u, grid, cells, RepairConfig(eps=.01), certify=certify_map, certificate_config=config, device='cuda')
    result.passed        True only if certify returned certified_feasible, bound > 0, no infeasible, no uncertain cell for result.u
    result.u             the certified field if passed; otherwise the input u itself (the caller goes back to the frozen .9^k rule on it)
    result.attempt_nodes, result.attempt_values    the free nodes and their values at the end, passed or not (the evidence of a failed attempt)
    result.rounds        one record per round (rho, radius, sizes, inner solver, certificate, seconds)

Layout as bayesreg.geom: node fields (nz, ny, nx, 3) mm, components (x, y, z); D[j, q, a] = d/dx_j of the Q1 basis function of vertex a at point q (jd_curl.three_d.cell_q1), F[r, j] = delta_rj + sum_a v[a, r] D[j, q, a].
torch only for the tensors; float64 on any device.  The arrays handed to certify are numpy float64 in C order; a GPU's IEEE float64 values cross unchanged.
"""
from dataclasses import asdict, dataclass, field
from itertools import product
from time import perf_counter

import numpy as np

from jd_curl.three_d.cell_q1 import VERTICES, CellQ1Operators3D, cell_tiles

LATTICE_N = 9                   # the dense lattice per axis: {0, 1/8, ..., 1}^3 holds closed27, not the Gauss points
ROLE = 'local repair of an uncertified field; only certify_map says it passes; not a method, not in any result table'


@dataclass(frozen=True)
class RepairConfig:
    """The numbers of repair_推导.md §4-5."""
    eps: float
    rho0: float = 1e2           # rho of the first round
    rho_factor: float = 10.     # rho_(r+1) = rho_factor rho_r
    radius0: int = 1            # the mask: every cell within this Chebyshev radius (in cells) of a seed cell has its interior nodes free
    radius_step: int = 1        # radius_(r+1) = radius_r + radius_step
    max_rounds: int = 5         # at the cap: failure, the input goes back to the frozen rule
    max_inner: int = 300        # L-BFGS iterations per round
    memory: int = 12            # L-BFGS pairs
    g_tol: float = 1e-3         # ||gradient||_inf in mm: stationary
    f_rtol: float = 1e-7        # relative decrease of E counted as no progress
    stall_iters: int = 5        # that many iterations in a row without progress: stalled
    wall_seconds: float = 120.  # after it no new inner iteration and no new round; the field reached still gets its certificate
    tile_cells: int = 32768     # cells per tile of a value and gradient evaluation

    def __post_init__(self):
        if not (self.eps > 0 and self.rho0 > 0 and self.rho_factor >= 1 and self.radius0 >= 0 and self.radius_step >= 0 and self.max_rounds >= 1):
            raise ValueError(f'RepairConfig: {self}')


@dataclass
class RepairResult:
    passed: bool
    u: np.ndarray               # certified, or the input unchanged (never a field that did not pass)
    reason: str
    certificate: dict           # the last certificate's record: of result.u if passed, of the last attempt otherwise
    initial_certificate: dict
    rounds: list = field(default_factory=list)
    attempt_nodes: np.ndarray = None
    attempt_values: np.ndarray = None
    seconds: float = 0.
    config: dict = None


def certificate_record(result):
    """What a round keeps of a certify_map result; the same fields as run_mc0.certify_ladder's attempts."""
    return dict(status=result.status, reason=result.reason, min_jd_lower_bound=result.min_jd_lower_bound, n_cells_checked=result.n_cells_checked, n_infeasible=result.n_infeasible, n_uncertain=result.n_uncertain,
                witness_cell=result.witness_cell, witness_reference_point=None if result.witness_reference_point is None else list(result.witness_reference_point), state_hash=result.state_hash, certificate_version=result.certificate_version)


def accepted(result):
    """The rule of the transfer ladder (run_mc0.certify_ladder): zero and undecided do not pass."""
    return result.status == 'certified_feasible' and result.min_jd_lower_bound is not None and result.min_jd_lower_bound > 0 and result.n_infeasible == 0 and result.n_uncertain == 0


# ---- points and tables ----------------------------------------------------------------------------------------------------------------------
def q1_derivative_table(points, spacing):
    """D (3, n, 8) at reference points (n, 3) ordered x, y, z: the formula of CellQ1Operators3D for any point (a test compares them at closed27)."""
    points = np.asarray(points, dtype=np.float64)
    out = np.empty((3, len(points), 8))
    for a, vertex in enumerate(VERTICES):
        factors = [points[:, j] if vertex[j] else 1-points[:, j] for j in range(3)]
        for j in range(3):
            others = [k for k in range(3) if k != j]
            out[j, :, a] = (1 if vertex[j] else -1)/spacing[j]*factors[others[0]]*factors[others[1]]
    return out


def union53(grid):
    """(points (53, 3), D (3, 53, 8)): Gauss27, then closed27 without its body centre, which is the middle Gauss point.  Each point once: the union of the brief, the set the witness record calls the 53 sample points."""
    gauss, closed = CellQ1Operators3D(grid, location='gauss27'), CellQ1Operators3D(grid, location='closed27_uniform')
    keep = [i for i, p in enumerate(closed.points) if not np.all(p == .5)]
    return np.concatenate((gauss.points, closed.points[keep])), np.concatenate((gauss.coefficients, closed.coefficients[:, keep]), axis=1)


def _lattice(n=LATTICE_N):
    """(points (n^3, 3) x, y, z, is_closed27 (n^3,)): x fastest, as ops.points."""
    t = np.linspace(0., 1., n)
    points = np.array([(x, y, z) for z, y, x in product(t, repeat=3)])
    return points, np.all(np.isin(points, (0., .5, 1.)), axis=1)


def cell_nodes(grid, cells):
    """(cells, 8) global node ids, the column order of cell_q1.VERTICES."""
    cz, cy, cx = np.unravel_index(np.asarray(cells, dtype=np.int64), grid.cell_shape)
    _, ny, nx = grid.shape
    return np.stack([((cz+z)*ny+cy+y)*nx+cx+x for x, y, z in VERTICES], axis=-1)


def mask_nodes(cells, grid, radius):
    """The free nodes (sorted ids): the nodes of the cells within Chebyshev radius `radius` of a cell of `cells`, faces left out (v = u = 0 there, so the certificate's boundary rule cannot be broken)."""
    cells = np.unique(np.asarray(cells, dtype=np.int64))
    if not len(cells):
        return np.empty(0, dtype=np.int64)
    nz, ny, nx = grid.shape
    offsets = np.arange(-radius, radius+2)
    dz, dy, dx = (a.ravel() for a in np.meshgrid(offsets, offsets, offsets, indexing='ij'))
    parts = []
    for start in range(0, len(cells), 4096):                                                # bounded memory, whatever the seed
        cz, cy, cx = np.unravel_index(cells[start:start+4096], grid.cell_shape)
        z, y, x = cz[:, None]+dz, cy[:, None]+dy, cx[:, None]+dx
        inside = (z >= 1) & (z <= nz-2) & (y >= 1) & (y <= ny-2) & (x >= 1) & (x <= nx-2)
        parts.append(np.unique(((z[inside]*ny+y[inside])*nx+x[inside]).astype(np.int64)))
    return np.unique(np.concatenate(parts))


def touched_cells(free, grid):
    """Every cell with a free node (sorted ids); a free node is interior, so all its 8 cells exist."""
    _, cy, cx = grid.cell_shape
    z, y, x = np.unravel_index(free, grid.shape)
    d = np.array([(a, b, c) for a in (-1, 0) for b in (-1, 0) for c in (-1, 0)])
    ids = ((z[:, None]+d[:, 0])*cy+y[:, None]+d[:, 1])*cx+x[:, None]+d[:, 2]
    return np.unique(ids.astype(np.int64))


def synthetic_folded_field(n=20, amplitude=4., sigma=1.2, seed=1):
    """(u, grid): a smooth random field tapered to zero faces on a small non-cubic, anisotropic grid, with some folded cells: the smoke test's and the tests' field.  No file."""
    from scipy.ndimage import gaussian_filter
    from jd_curl.three_d.grid import make_grid_3d
    grid = make_grid_3d(nx=n, ny=n+1, nz=n+2, hx=1., hy=1.25, hz=.75, unit='mm')
    rng = np.random.default_rng(seed)
    u = np.stack([gaussian_filter(rng.normal(size=grid.shape), sigma) for _ in range(3)], axis=-1)
    u *= amplitude/np.abs(u).max()
    edge = np.minimum.reduce(np.meshgrid(*[np.minimum(np.arange(s), s-1-np.arange(s)) for s in grid.shape], indexing='ij'))
    return np.minimum(edge/4., 1.)[..., None]*u, grid


# ---- the locator -----------------------------------------------------------------------------------------------------------------------------
def locate_cells(u, grid, certificate_config):
    """The ids of the cells the certificate's two fast proofs leave to its interval fallback, by the same kernels and the same rules (parallel_margins with its whole-tile rule, prescreen_lower_bounds).
    Every cell the certificate can call infeasible or uncertain is among them (a test checks the witness), some of the others pass after subdivision.  certify_map itself gives only the count and the first witness.  A locator, never a verdict."""
    from jd_curl.three_d.certificate_prescreen import prescreen_lower_bounds
    from jd_curl.three_d.compiled_enclosure import parallel_margins
    u = np.ascontiguousarray(np.asarray(u, dtype=np.float64))
    try:
        margin, flag = parallel_margins(u, grid, 0.)
    except FloatingPointError:
        margin = flag = None
    open_cells = []
    for tile in cell_tiles(grid, certificate_config.tile_size):
        if flag is None or bool(flag[tile].any()):
            open_cells.append(np.arange(tile.start, tile.stop))
        else:
            open_cells.append(tile.start+np.flatnonzero(~(margin[tile] > 0)))
    cells = np.concatenate(open_cells).astype(np.int64)
    if not len(cells):
        return cells
    table = CellQ1Operators3D(grid, location='closed27_uniform').coefficients
    lower = prescreen_lower_bounds(u, grid, table, cells)
    with np.errstate(invalid='ignore'):
        return cells[~(lower > 0)]


# ---- the objective -----------------------------------------------------------------------------------------------------------------------------
def _jacobian(torch, F):
    """(J, A): A = cof F, J = det F, by the mainline's expressions (geometry.cofactor; J = sum_c F_0c A_0c); F (..., 3, 3) float64."""
    f = F.reshape(F.shape[:-2]+(9,)).unbind(-1)
    a00, a01, a02 = f[4]*f[8]-f[5]*f[7], f[5]*f[6]-f[3]*f[8], f[3]*f[7]-f[4]*f[6]
    a10, a11, a12 = f[2]*f[7]-f[1]*f[8], f[0]*f[8]-f[2]*f[6], f[1]*f[6]-f[0]*f[7]
    a20, a21, a22 = f[1]*f[5]-f[2]*f[4], f[2]*f[3]-f[0]*f[5], f[0]*f[4]-f[1]*f[3]
    A = torch.stack((a00, a01, a02, a10, a11, a12, a20, a21, a22), dim=-1).reshape(F.shape)
    return f[0]*a00+f[1]*a01+f[2]*a02, A


class Objective:
    """E(x) of one round on the free nodes: x (N, 3) on the device.  Every value and gradient of this module goes through `value_and_grad`, which a finite-difference check tests against bayesreg.geom (check_repair_gradient.py)."""

    def __init__(self, torch, anchor, start, grid, free, points_table, *, device, tile_cells, extras=None):
        """anchor (u), start (the field the round starts from): (nz, ny, nx, 3) float64; free: sorted node ids; points_table: D (3, n, 8); extras: (cells, points (m, 3)), a point per row, or None."""
        self.torch, self.device, self.tile_cells = torch, torch.device(device), tile_cells
        kw = dict(device=self.device, dtype=torch.float64)
        self.free, self.cells = free, touched_cells(free, grid)
        nodes = cell_nodes(grid, self.cells)
        extra_nodes = None
        if extras is not None and len(extras[0]):
            extra_nodes = cell_nodes(grid, extras[0])
            local, inverse = np.unique(np.concatenate((nodes.ravel(), extra_nodes.ravel())), return_inverse=True)
            self.idx = torch.as_tensor(inverse[:nodes.size].reshape(nodes.shape), device=self.device, dtype=torch.long)
            self.extra_idx = torch.as_tensor(inverse[nodes.size:].reshape(extra_nodes.shape), device=self.device, dtype=torch.long)
            self.extra_D = torch.as_tensor(np.moveaxis(q1_derivative_table(extras[1], grid.spacing), 0, 1), **kw)        # (m, 3, 8)
        else:
            local, inverse = np.unique(nodes, return_inverse=True)
            self.idx = torch.as_tensor(inverse.reshape(nodes.shape), device=self.device, dtype=torch.long)
            self.extra_idx = self.extra_D = None
        self.free_pos = torch.as_tensor(np.searchsorted(local, free), device=self.device, dtype=torch.long)
        self.base = torch.as_tensor(np.asarray(start, dtype=np.float64).reshape(-1, 3)[local], **kw)                        # the nodes around the mask stay at start, which equals u there
        self.u = torch.as_tensor(np.asarray(anchor, dtype=np.float64).reshape(-1, 3)[free], **kw)
        self.start = torch.as_tensor(np.asarray(start, dtype=np.float64).reshape(-1, 3)[free], **kw)
        self.D = torch.as_tensor(np.moveaxis(points_table, 0, 1), **kw)                                                     # (q, 3, 8)
        self.eye = torch.eye(3, **kw)

    def _block(self, X, idx, D, rho, eps, out, stats, *, per_row=False):
        """One block: idx (c, 8), D (q, 3, 8), or (c, 3, 8) with per_row: one point per row.  Adds the penalty's gradient to out (L, 3); returns the sum of squared gaps."""
        torch = self.torch
        V = X[idx]                                                                          # (c, 8, 3)
        grad = torch.matmul(D, V) if per_row else torch.matmul(D, V[:, None])               # [..., j, r] = d_j v_r
        J, A = _jacobian(torch, grad.transpose(-1, -2)+self.eye)
        gap = (eps-J).clamp(min=0.)
        stats['min_j'] = min(stats['min_j'], float(J.min()))
        stats['max_gap'] = max(stats['max_gap'], float(gap.max()))
        active = gap > 0
        stats['active'] += int(active.sum())
        if out is not None and bool(active.any()):
            if per_row:
                flux = -(rho*gap)[..., None, None]*A                                        # dE/dF_rj of each point: (c, 3, 3)
                nodal = torch.matmul(flux, D).transpose(-1, -2)                             # (c, 8, 3): sum_j flux_rj D_ja
                out.index_add_(0, idx.reshape(-1), nodal.reshape(-1, 3))
            else:
                on = active.any(dim=1)
                flux = -(rho*gap[on])[..., None, None]*A[on]                                # (c', q, 3, 3)
                nodal = torch.matmul(flux, D).sum(dim=1).transpose(-1, -2)                  # (c', 8, 3)
                out.index_add_(0, idx[on].reshape(-1), nodal.reshape(-1, 3))
        return float((gap*gap).sum())

    def value_and_grad(self, x, rho, eps, *, grad=True):
        """(E, gradient (N, 3) or None, stats).  E = 1/2 ||x - u||^2 + (rho/2) sum_q gap_q^2; dE/dx_(a, r) = x_(a, r) - u_(a, r) - rho sum_q gap_q sum_j A_rj D_jqa."""
        torch = self.torch
        X = self.base.clone()
        X[self.free_pos] = x
        out = torch.zeros_like(X) if grad else None
        stats = dict(min_j=float('inf'), max_gap=0., active=0)
        squares = 0.
        for start in range(0, self.idx.shape[0], self.tile_cells):
            squares += self._block(X, self.idx[start:start+self.tile_cells], self.D, rho, eps, out, stats)
        if self.extra_idx is not None:
            squares += self._block(X, self.extra_idx, self.extra_D, rho, eps, out, stats, per_row=True)
        d = x-self.u
        energy = .5*float((d*d).sum())+.5*rho*squares
        return energy, None if out is None else d+out[self.free_pos], stats


def lbfgs(objective, x0, rho, eps, config, *, deadline, clock=perf_counter):
    """L-BFGS with Armijo backtracking (repair_推导.md §5).  Returns (x, record).  A pair is kept only if s.y > 0, so the two-loop direction stays a descent direction; a direction that is not one restarts from the gradient."""
    torch = objective.torch
    x = x0
    energy, g, stats = objective.value_and_grad(x, rho, eps)
    pairs, reason, iterations, stalled, evaluations = [], 'inner_cap', 0, 0, 1
    first_energy, first_stats = energy, dict(stats)
    while iterations < config.max_inner:
        norm = float(g.abs().max())
        if norm <= config.g_tol:
            reason = 'stationary'
            break
        if clock() >= deadline:
            reason = 'wall'
            break
        q = g.clone()
        alphas = []
        for s, y, r in reversed(pairs):
            a = r*float((s*q).sum())
            q -= a*y
            alphas.append(a)
        if pairs:
            s, y, r = pairs[-1]
            q *= 1/(r*float((y*y).sum()))
        for (s, y, r), a in zip(pairs, reversed(alphas)):
            q += (a-r*float((y*q).sum()))*s
        d = -q
        slope = float((g*d).sum())
        if not slope < 0:
            pairs, d, slope = [], -g, -float((g*g).sum())
        t = 1. if pairs else min(1., .1/norm)                # the first step moves no node by more than .1 mm in a component
        while True:
            trial = x+t*d
            value, gradient, trial_stats = objective.value_and_grad(trial, rho, eps)
            evaluations += 1
            if value <= energy+1e-4*t*slope:
                break
            t *= .5
            if t < 1e-12:
                gradient = None
                break
        if gradient is None:
            reason = 'line_search_failed'
            break
        s, y = trial-x, gradient-g
        curvature = float((s*y).sum())
        if curvature > 1e-12*float((y*y).sum()):
            pairs.append((s, y, 1/curvature))
            if len(pairs) > config.memory:
                pairs.pop(0)
        decrease = energy-value
        stalled = stalled+1 if decrease <= config.f_rtol*max(abs(value), 1e-12) else 0
        x, g, energy, stats = trial, gradient, value, trial_stats
        iterations += 1
        if stalled >= config.stall_iters:
            reason = 'stalled'
            break
    moved = (x-objective.u).norm(dim=1)
    return x, dict(iterations=iterations, evaluations=evaluations, stop=reason, energy_first=first_energy, energy=energy, g_inf=float(g.abs().max()), min_j_first=first_stats['min_j'], min_j=stats['min_j'], max_gap_first=first_stats['max_gap'], max_gap=stats['max_gap'],
                   active_points=stats['active'], max_move_mm=float(moved.max()), n_moved=int((moved > 0).sum()))


def lattice_points(u, grid, cells, eps, *, device, batch=4096):
    """(cells', points'): for each of `cells`, its smallest J on the dense lattice, if that is below eps and off closed27: the points the next round adds to Q_c (repair_推导.md §6).  An estimate for placing a penalty point, never a proof.
    (A Gauss point is not on the lattice, so a dip the 53 points see may come back once as a neighbouring lattice point.  A second point of the same cell costs nothing.)"""
    import torch
    cells = np.asarray(cells, dtype=np.int64)
    points, on_closed = _lattice()
    if not len(cells):
        return np.empty(0, dtype=np.int64), np.empty((0, 3))
    dev = torch.device(device)
    D = torch.as_tensor(np.moveaxis(q1_derivative_table(points, grid.spacing), 0, 1), device=dev, dtype=torch.float64)
    flat = np.asarray(u, dtype=np.float64).reshape(-1, 3)
    eye = torch.eye(3, device=dev, dtype=torch.float64)
    out_cells, out_points = [], []
    for start in range(0, len(cells), batch):
        part = cells[start:start+batch]
        V = torch.as_tensor(flat[cell_nodes(grid, part)], device=dev, dtype=torch.float64)
        J, _ = _jacobian(torch, torch.matmul(D, V[:, None]).transpose(-1, -2)+eye)
        low, where = J.min(dim=1)
        low, where = low.cpu().numpy(), where.cpu().numpy()
        keep = (low < eps) & ~on_closed[where]
        out_cells.append(part[keep])
        out_points.append(points[where[keep]])
    return np.concatenate(out_cells), np.concatenate(out_points)


# ---- the rounds --------------------------------------------------------------------------------------------------------------------------------
def repair_field(u, grid, cells, config, *, certify, certificate_config, device='cuda', locate=None, clock=perf_counter):
    """See the module docstring.  certify(field, grid, config=certificate_config) is called once on the input and once after every round; `locate` defaults to locate_cells (a test passes another)."""
    import torch
    t0 = clock()
    deadline = t0+config.wall_seconds
    u = np.ascontiguousarray(np.asarray(u, dtype=np.float64))
    if u.shape != grid.shape+(3,):
        raise ValueError(f'repair_field: u of shape {u.shape}, grid {grid.shape}')
    locate = locate_cells if locate is None else locate
    first = certify(u, grid, config=certificate_config)
    initial = certificate_record(first)
    if accepted(first):
        return RepairResult(True, u, 'already_certified', initial, initial, [], np.empty(0, dtype=np.int64), np.empty((0, 3)), clock()-t0, asdict(config))
    _, table = union53(grid)
    v = u.copy()
    seed = np.unique(np.asarray(cells, dtype=np.int64))
    extra_cells, extra_points = np.empty(0, dtype=np.int64), np.empty((0, 3))
    rounds, free, last, reason = [], np.empty(0, dtype=np.int64), initial, 'round_cap'
    for r in range(config.max_rounds):
        tick = clock()
        rho, radius = config.rho0*config.rho_factor**r, config.radius0+r*config.radius_step
        previous = free
        free = np.union1d(previous, mask_nodes(seed, grid, radius))     # never shrinks: a node moved earlier stays free, so everything outside the last mask is u
        if not len(free):
            reason = 'empty_mask'
            break
        objective = Objective(torch, u, v, grid, free, table, device=device, tile_cells=config.tile_cells, extras=(extra_cells, extra_points))
        x, inner = lbfgs(objective, objective.start, rho, config.eps, config, deadline=deadline, clock=clock)
        v.reshape(-1, 3)[free] = x.detach().cpu().numpy()
        solved = clock()
        result = certify(v, grid, config=certificate_config)
        last = certificate_record(result)
        checked = clock()
        rounds.append(dict(round=r, rho=rho, radius=radius, n_seed=len(seed), n_free=len(free), n_cells=len(objective.cells), n_extra_points=len(extra_cells), inner=inner, certificate=last, seconds_solve=solved-tick, seconds_certificate=checked-solved))
        if accepted(result):
            return RepairResult(True, v, 'certified', last, initial, rounds, free, v.reshape(-1, 3)[free].copy(), clock()-t0, asdict(config))
        if clock() >= deadline:
            reason = 'wall'
            break
        located = locate(v, grid, certificate_config)
        witness = np.array([] if result.witness_cell is None else [result.witness_cell], dtype=np.int64)
        seed = np.union1d(seed, np.union1d(located, witness))
        more_cells, more_points = lattice_points(v, grid, located, config.eps, device=device)
        extra_cells, extra_points = np.concatenate((extra_cells, more_cells)), np.concatenate((extra_points, more_points))
        rounds[-1].update(n_located=len(located), n_new_extra_points=len(more_cells), seconds_locate=clock()-checked)
    return RepairResult(False, u, reason, last, initial, rounds, free, v.reshape(-1, 3)[free].copy() if len(free) else np.empty((0, 3)), clock()-t0, asdict(config))


def change_record(before, after):
    """Computed from the two fields, whatever a run says of itself: the nodes that differ, their largest and mean change in mm (Euclidean norm per node), the bounding box of the change."""
    before, after = np.asarray(before), np.asarray(after)
    delta = after-before
    changed = np.any(after != before, axis=-1)
    n = int(np.count_nonzero(changed))
    if not n:
        return dict(n_changed_nodes=0, max_change_mm=0., mean_change_mm=0., rms_change_all_nodes_mm=0.)
    norm = np.sqrt((delta[changed]**2).sum(axis=-1))
    return dict(n_changed_nodes=n, max_change_mm=float(norm.max()), mean_change_mm=float(norm.mean()), rms_change_all_nodes_mm=float(np.sqrt((delta*delta).sum()/changed.size)))
