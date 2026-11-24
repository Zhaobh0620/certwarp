"""KKT (work package KKT): is the L0 GF_B plateau of CVG R1 a KKT point on the certificate boundary (H-KKT) or a
stall from ill-conditioning (H-cond)?  Diagnostic only: no solver run, bayesreg/ and jd_curl/ are not modified.

    CVG_CODE=<cvg snapshot> python -B experiments/kkt/kkt.py --run <cvg run dir> --state <npz> --label NAME
        --out /home/rmcfz/seq_vp_tag_results/kkt_v1 [--k K]

Plan: runs/20261031-kkt/plan.md (written before any computation).  At the stored state u of a CVG L0 run:

0. The model state is rebuilt as the run had it (as experiments/cvg/stiffness.py `setup` does at the CVG sha
   98d43f14..., copied here because the working-tree stiffness.py changed since): cache -> LevelInput ->
   multilevel._template with the run's options -> sigma2 = the round's sigma2_used; the frozen C09.1 metric from the
   run's c_dat and b_P.  GF_B is recomputed and compared with trace.csv at step K (state_hash too); relative
   difference > 1e-10 -> stop.
1. Per-cell certified lower bounds b_K: CaptureBackend.certify (C07-2 configuration), then
   jd_scaling.certificate_cell_bounds (cert_eval.cell_pass; bit-for-bit min check against the certificate).
2. Near-active cells b_K < t; per cell a 17^3 reference grid of Q1 J values, L-BFGS-B from the 3 lowest grid
   points -> x*, J_dense; the constraint normal n_K = dJ(x*; u)/du (dJ/du_{a,i} = sum_j cof(F)_ij d_j phi_a(x*)),
   boundary nodes dropped, mapped to the free vector theta (component-major, restrict_free).  By Danskin's theorem
   this is the gradient of min_{x in K} J(x; u) when the argmin is unique, otherwise one element of its (Clarke)
   subdifferential.  "Multi-point cone": up to 8 further grid points with J < 2 J_dense + 1e-6 (spacing >= 2 grid
   steps) per cell, descriptive only.
3. NNLS min_{lambda >= 0} ||g - N lambda||_2 on the rows N touches (other rows of g enter the residual as is);
   r in the GF_B normalization; the same in the M^-1 norm (M the run's frozen C09.1 metric, M^-1/2 by DST);
   the same cone applied to -g (sign control); the node-support upper bound.
4. F_B blocks (each channel, diffusion, log-volume Gauss / closed27 parts, curl) per brain node / brain cell;
   per-block gradient norms on the nodes of the near-active cells.
5. Quadrature blind spot: J at the 53 points (Gauss27 + closed27), J_dense and b_K on the near-active cells.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"cvg"))
import common  # noqa: E402  (inserts CVG_CODE at the front of sys.path)

import numpy as np  # noqa: E402
from scipy.fft import dstn, idstn  # noqa: E402
from scipy.optimize import minimize, nnls  # noqa: E402

from bayesreg.geom import node_field  # noqa: E402
from bayesreg.io import load_pair  # noqa: E402
from bayesreg.cli import run_pair as rp  # noqa: E402
from bayesreg.model import build_metric, evaluate, gradient, metric_bP  # noqa: E402
from bayesreg.solve.jd_scaling import (CERTIFICATE, J_ACT, certificate_cell_bounds, node_minimum,  # noqa: E402
                                       scaling_factors, slab_point_jacobians)
from bayesreg.solve.multilevel import _template  # noqa: E402
from bayesreg.solve.stopping import gf_b, state_hash  # noqa: E402

THRESHOLDS = (1e-2, 1e-3, 1e-4)          # pre-registered
DESCRIPTIVE = (1e-1, .25)                 # descriptive only (not in the decision)
PRIMARY_T = 1e-3
POSTHOC = (5e-2, 1e-2, 1e-3, 1e-4)        # post hoc: near-active by the dense minimum J_dense (not pre-registered)
NGRID = 17
RSS_LIMIT_MB = 3000
BLOCKS = ("channel", "diffusion", "logvol", "curl")


def rss_watchdog(limit_mb=RSS_LIMIT_MB):
    def run():
        while True:
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            mb = int(line.split()[1])/1024
                            if mb > limit_mb:
                                sys.stderr.write(f"RSS {mb:.0f} MB > {limit_mb} MB: abort\n")
                                os._exit(3)
            except OSError:
                pass
            time.sleep(.5)
    threading.Thread(target=run, daemon=True).start()


# ---- 0. model state as the run had it ---------------------------------------------------------------------------

def setup(run, level=0):
    config = json.loads((run/"config.json").read_text())
    result = json.loads((run/"result.json").read_text())
    args = config["args"]
    cache, meta = common.load_cache(args["cache"])
    pair = load_pair(rp.resolve_pair(common.DATA_ROOT, args["pair"]), allowlist=rp.DEFAULT_ALLOWLIST)
    grids = common.grids_of(pair, meta)
    inputs, kept = common.level_inputs_from_cache(cache, meta, grids, args["channel_source"])
    eta = common.eta_of(meta, args["nu"])
    if eta.record()["nu"] != config["eta"]["nu"]:
        raise RuntimeError("eta differs from the run")
    row = [r for r in result["rounds"] if r["level"] == level][0]
    solver = config["solver"]
    logvol = {k: solver[k] for k in ("logvol_closed27_share", "closed27_weights") if k in solver}
    host = _template(inputs[level], eta, inputs[-1].grid.omega, solver["gauss_tile_size"],
                     inputs[0].fixed.shape[0], args["cpu_threads"], solver["logvol_points"], **logvol)
    state = host.replace(sigma2=np.asarray(row["sigma2_used"]))
    b_P = metric_bP(state, mode="principal")
    if not np.isclose(b_P, row["metric"]["b_P"], rtol=1e-14, atol=0):
        raise RuntimeError(f"b_P {b_P} != run {row['metric']['b_P']}")
    metric = build_metric(state, c_dat=row["metric"]["c_dat"], b_P=b_P)
    return config, result, row, state, metric


def trace_row(run, k):
    lines = (run/"trace.csv").read_text().splitlines()
    head = lines[0].split(",")
    for line in lines[1:]:
        values = line.split(",")
        if int(values[1]) == k:
            return {h: float(v) if v else None for h, v in zip(head, values)}
    raise RuntimeError(f"step {k} not in trace.csv")


def accept_hash(run, k):
    with open(run/"log.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") == "accept" and int(row["accepted_steps"]) == k:
                return row["state_hash"]
    return None


# ---- Q1 J at arbitrary reference points of a set of cells --------------------------------------------------------

CORNERS = [(dz, dy, dx) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]       # node offsets (z, y, x)


def corner_values(u, cells, cell_shape):
    """(m, 8, 3) displacement of the 8 corners of each cell (flat C-order cell index), components xyz."""
    kz, jy, ix = np.unravel_index(cells, cell_shape)
    return np.stack([u[kz+dz, jy+dy, ix+dx] for dz, dy, dx in CORNERS], axis=1)


def basis_gradients(points, spacing):
    """(p, 8, 3) physical gradients d_j phi_a (j = x, y, z) of the 8 Q1 basis functions at reference points (p, 3) =
    (xi, eta, zeta) <-> (x, y, z)."""
    hx, hy, hz = spacing
    xi, et, ze = points[:, 0], points[:, 1], points[:, 2]
    out = np.empty((len(points), 8, 3))
    for a, (dz, dy, dx) in enumerate(CORNERS):
        lx = xi if dx else 1-xi
        ly = et if dy else 1-et
        lz = ze if dz else 1-ze
        sx, sy, sz = (1. if dx else -1.), (1. if dy else -1.), (1. if dz else -1.)
        out[:, a, 0] = sx*ly*lz/hx
        out[:, a, 1] = lx*sy*lz/hy
        out[:, a, 2] = lx*ly*sz/hz
    return out


def deformation(U, G):
    """F (m, p, 3, 3) with F[i, j] = delta_ij + d_j u_i from corner values U (m, 8, 3) and basis gradients G (p, 8, 3)."""
    F = np.einsum("mai,paj->mpij", U, G)
    F[..., 0, 0] += 1.
    F[..., 1, 1] += 1.
    F[..., 2, 2] += 1.
    return F


def det3(F):
    return (F[..., 0, 0]*(F[..., 1, 1]*F[..., 2, 2]-F[..., 1, 2]*F[..., 2, 1])
            - F[..., 0, 1]*(F[..., 1, 0]*F[..., 2, 2]-F[..., 1, 2]*F[..., 2, 0])
            + F[..., 0, 2]*(F[..., 1, 0]*F[..., 2, 1]-F[..., 1, 1]*F[..., 2, 0]))


def cofactor(F):
    """cof(F) with columns f1 x f2, f2 x f0, f0 x f1 (f_j the columns of F): d det / d F_ij = cof_ij."""
    f0, f1, f2 = F[..., :, 0], F[..., :, 1], F[..., :, 2]
    return np.stack([np.cross(f1, f2), np.cross(f2, f0), np.cross(f0, f1)], axis=-1)


def J_points(U, points, spacing):
    return det3(deformation(U, basis_gradients(points, spacing)))


def J_and_xi_gradient(Uc, x, spacing):
    """J and dJ/d(xi, eta, zeta) at one reference point x for one cell (corner values Uc (8, 3))."""
    hx, hy, hz = spacing
    G = basis_gradients(x[None], spacing)[0]                 # (8, 3)
    F = np.eye(3)+Uc.T@G                                     # F[i, j] = delta + sum_a U[a, i] G[a, j]
    C = cofactor(F)
    J = float(det3(F))
    xi, et, ze = x
    grad = np.zeros(3)
    for k in range(3):
        dG = np.zeros((8, 3))                                 # d/dx_k of d_j phi_a
        for a, (dz, dy, dx) in enumerate(CORNERS):
            l = [xi if dx else 1-xi, et if dy else 1-et, ze if dz else 1-ze]
            s = [1. if dx else -1., 1. if dy else -1., 1. if dz else -1.]
            h = [hx, hy, hz]
            for j in range(3):
                if j == k:
                    continue                                  # d_j phi_a is constant along x_j
                prod = s[j]/h[j]*s[k]
                for m in range(3):
                    if m not in (j, k):
                        prod *= l[m]
                dG[a, j] = prod
        dF = Uc.T@dG
        grad[k] = float(np.sum(C*dF))
    return J, grad


def normal_at(Uc, x, spacing):
    """dJ(x)/dU (8, 3) for one cell: dJ/dU[a, i] = sum_j cof(F)_ij G[a, j]."""
    G = basis_gradients(np.asarray(x)[None], spacing)[0]
    F = np.eye(3)+Uc.T@G
    C = cofactor(F)
    return G@C.T                                             # (8, 3): sum_j G[a, j] C[i, j]


def dense_minimum(Uc, spacing, grid_points, grid_J, extra_start=None):
    """x*, J(x*) by L-BFGS-B from the 3 lowest grid points (and extra_start, the lowest of the 53 points); never above
    the grid minimum."""
    order = np.argsort(grid_J)[:3]
    best_x, best_J = grid_points[order[0]].copy(), float(grid_J[order[0]])
    starts = [grid_points[i] for i in order]+([np.asarray(extra_start, float)] if extra_start is not None else [])
    for start in starts:
        res = minimize(lambda x: J_and_xi_gradient(Uc, x, spacing), start, jac=True, method="L-BFGS-B",
                       bounds=[(0., 1.)]*3, options=dict(ftol=1e-15, gtol=1e-13, maxiter=200))
        J = J_and_xi_gradient(Uc, np.clip(res.x, 0, 1), spacing)[0]
        if J < best_J:
            best_x, best_J = np.clip(res.x, 0, 1), J
    return best_x, best_J


def location_type(x, tol=1e-6):
    on = int(np.sum((np.abs(x) < tol) | (np.abs(x-1) < tol)))
    return ("interior", "face", "edge", "corner")[on]


# ---- free-vector mapping ------------------------------------------------------------------------------------------

def free_columns(cells, normals, grid):
    """Sparse columns (rows, values) of the free vector for each normal (m, 8, 3): theta index = i * n_int +
    C-order index of the interior node (z - 1, y - 1, x - 1); boundary nodes dropped."""
    nz, ny, nx = grid.shape
    iz, iy, ix = grid.interior_shape
    n_int = iz*iy*ix
    kz, jy, kx = np.unravel_index(cells, grid.cell_shape)
    rows, vals = [], []
    for m in range(len(cells)):
        r, v = [], []
        for a, (dz, dy, dx) in enumerate(CORNERS):
            z, y, x = kz[m]+dz, jy[m]+dy, kx[m]+dx
            if z in (0, nz-1) or y in (0, ny-1) or x in (0, nx-1):
                continue
            node = ((z-1)*iy+(y-1))*ix+(x-1)
            for i in range(3):
                r.append(i*n_int+node)
                v.append(normals[m, a, i])
        rows.append(np.array(r, dtype=np.int64))
        vals.append(np.array(v))
    return rows, vals


def dense_matrix(rows, vals, support):
    """(len(support), m) dense matrix on the rows `support` (sorted unique)."""
    where = {int(s): k for k, s in enumerate(support)}
    A = np.zeros((len(support), len(rows)))
    for j, (r, v) in enumerate(zip(rows, vals)):
        A[[where[int(x)] for x in r], j] = v
    return A


class MetricRoot:
    """M^-1/2 of the C09.1 metric by DST-I (as experiments/cvg/stiffness.py MetricRoot)."""

    def __init__(self, metric, grid):
        self.grid = grid
        self.factor = np.sqrt(metric.omega0/metric.inverse.denominator)

    def apply(self, v):
        f = v.reshape((3,)+self.grid.interior_shape)
        return idstn(dstn(f, type=1, norm="ortho", axes=(-3, -2, -1))*self.factor, type=1, norm="ortho",
                     axes=(-3, -2, -1)).ravel()


def nnls_fit(A, b, rest_sq=0.):
    """min_{lambda >= 0} ||b - A lambda||; returns lambda, residual norm (including rest_sq of rows outside A)."""
    if A.shape[1] == 0:
        return np.zeros(0), float(np.sqrt(b@b+rest_sq))
    lam, res = nnls(A, b, maxiter=50*A.shape[1])
    return lam, float(np.sqrt(res**2+rest_sq))


def summary(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    q = np.percentile(x, [0, 5, 25, 50, 75, 95, 100])
    return dict(n=int(x.size), min=q[0], p5=q[1], p25=q[2], median=q[3], p75=q[4], p95=q[5], max=q[6])


# ---- main -----------------------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, required=True, help="accepted step of the state (trace.csv row)")
    a = p.parse_args(argv)
    rss_watchdog()
    tick0 = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out/f"{a.label}.json"
    if target.exists():
        raise FileExistsError(target)
    run = Path(a.run)
    import bayesreg
    if not str(Path(bayesreg.__file__).resolve()).startswith(os.environ.get("CVG_CODE", "<unset>")):
        raise RuntimeError(f"bayesreg imported from {bayesreg.__file__}, not the CVG snapshot")
    config, result, row, state, metric = setup(run)
    grid = state.grid
    n_int = grid.n_free_scalar
    norm = np.sqrt(grid.omega/state.omega0)
    stored = np.load(a.state)
    u = np.array(stored["u"])
    digest = state_hash(u)
    if digest != str(stored["state_hash"]):
        raise RuntimeError("state file hash differs from its stored state_hash")
    logged_hash = accept_hash(run, a.k)
    trow = trace_row(run, a.k)
    doc = dict(label=a.label, run=str(run), state=a.state, k=a.k, state_hash=digest, code=common.code_record(),
               kkt_script_sha256=common.sha256_file(__file__), grid_shape=list(grid.shape), n_free=3*n_int,
               spacing=list(grid.spacing), omega_level=grid.omega, omega0=state.omega0,
               eta=dict(alpha=state.alpha, nu=state.nu, tau2=np.asarray(state.tau2).tolist()),
               channels=int(state.n_channels), G0=row["G0"], T_B=row["T_B"])
    # ---- 0. GF_B check
    point = evaluate(state, u)
    if not hasattr(point, "energy"):
        raise RuntimeError(f"F_B undefined at the state: {point.reason}")
    g = gradient(state, point)
    GF = gf_b(g, grid.omega, state.omega0)
    rel = abs(GF-trow["GF_B"])/trow["GF_B"]
    relF = abs(point.energy-trow["F_B"])/abs(trow["F_B"])
    doc["check"] = dict(GF_B=GF, GF_B_trace=trow["GF_B"], GF_B_rel_diff=rel, F_B=point.energy, F_B_trace=trow["F_B"],
                        F_B_rel_diff=relF, accept_row_state_hash=logged_hash,
                        hash_matches_accept_row=(logged_hash == digest) if logged_hash else None,
                        min_jd_lower_bound_trace=trow["min_jd_lower_bound"])
    print(f"[{a.label}] GF_B {GF!r} trace {trow['GF_B']!r} rel {rel:.2e}; F_B rel {relF:.2e}; hash "
          f"{doc['check']['hash_matches_accept_row']}", flush=True)
    if rel > 1e-10 or relF > 1e-10 or (logged_hash is not None and logged_hash != digest):
        doc["stopped"] = "GF_B / F_B / hash check failed: stop (plan)"
        common.write_json(target, doc)
        raise SystemExit(f"check failed: {doc['check']}")
    # ---- 1. per-cell bounds
    backend = common.CaptureBackend()
    cert = backend.certify(u, grid)
    if not cert.passed:
        raise RuntimeError("stored state does not pass the certificate")
    bounds, brec = certificate_cell_bounds(u, grid, cert, backend.cert_config)
    b = np.asarray(bounds).ravel()
    doc["certificate"] = dict(min_jd_lower_bound=cert.min_jd_lower_bound, state_hash=cert.state_hash,
                              certificate_version=getattr(cert, "certificate_version", None),
                              cell_bounds_record=brec, n_cells=int(b.size), n_without_bound=int(np.sum(~np.isfinite(b))),
                              min_bound_matches_trace=float(cert.min_jd_lower_bound) == trow["min_jd_lower_bound"])
    bfin = np.where(np.isfinite(b), b, np.inf)
    # 53-point minimum (Gauss27 + closed27) of every cell from bayesreg's slab evaluation
    nz = grid.shape[0]
    J53_all = slab_point_jacobians(node_field(u, grid), grid, 0, nz-1, closed=True)
    keys = list(J53_all)
    J53_stack = np.stack([J53_all[k].ravel() for k in keys])     # (53, n_cells)
    J53_min = J53_stack.min(axis=0)
    # brain cells: lower corner node in the mask (cert_eval convention)
    mask = np.asarray(state.mask, dtype=bool)
    brain_cell = mask[:-1, :-1, :-1].ravel()
    doc["counts"] = dict(brain_nodes=int(mask.sum()), brain_cells=int(brain_cell.sum()), cells=int(b.size),
                         cells_below={str(t): int(np.sum(bfin < t)) for t in THRESHOLDS+DESCRIPTIVE})
    # ---- 2. dense sampling of every cell with b_K < max threshold
    tmax = max(THRESHOLDS+DESCRIPTIVE)
    cells = np.flatnonzero(bfin < tmax)
    t1 = np.linspace(0., 1., NGRID)
    grid_points = np.stack(np.meshgrid(t1, t1, t1, indexing="ij"), -1).reshape(-1, 3)
    G_grid = basis_gradients(grid_points, grid.spacing)
    pts53 = np.array(keys, dtype=float)
    U_all = corner_values(u, cells, grid.cell_shape)
    xstar = np.zeros((len(cells), 3))
    Jdense = np.zeros(len(cells))
    J53_mine = np.zeros(len(cells))
    J_grid_min = np.zeros(len(cells))
    multi = []                                                   # per cell: list of extra grid points
    chunk = 64
    max53_rel = 0.
    for s in range(0, len(cells), chunk):
        U = U_all[s:s+chunk]
        Jg = det3(deformation(U, G_grid))                       # (m, p)
        J5 = J_points(U, pts53, grid.spacing)                    # (m, 53)
        ref = J53_stack[:, cells[s:s+chunk]].T
        max53_rel = max(max53_rel, float(np.max(np.abs(J5-ref)/np.maximum(np.abs(ref), 1e-300))))
        for m in range(U.shape[0]):
            i = s+m
            xs, js = dense_minimum(U[m], grid.spacing, grid_points, Jg[m], pts53[int(np.argmin(J5[m]))])
            xstar[i], Jdense[i] = xs, js
            J53_mine[i] = J5[m].min()
            J_grid_min[i] = Jg[m].min()
            # multi-point cone: grid points with J < 2 J_dense + 1e-6, increasing J, spacing >= 2 grid steps
            cand = np.flatnonzero(Jg[m] < 2*js+1e-6)
            cand = cand[np.argsort(Jg[m][cand])]
            idx3 = np.stack(np.unravel_index(cand, (NGRID,)*3), -1)
            chosen = []
            xs_idx = np.rint(xs*(NGRID-1))
            for c, ijk in zip(cand, idx3):
                if np.max(np.abs(ijk-xs_idx)) < 2:
                    continue
                if any(np.max(np.abs(ijk-o)) < 2 for o in chosen):
                    continue
                chosen.append(ijk)
                if len(chosen) == 8:
                    break
            multi.append([grid_points[np.ravel_multi_index(tuple(o), (NGRID,)*3)] for o in chosen])
    doc["implementation_checks"] = dict(J53_max_rel_diff_vs_slab_point_jacobians=max53_rel,
                                        lower_bound_violations=int(np.sum(bfin[cells] > Jdense*(1+1e-12)+1e-15)),
                                        dense_above_53=int(np.sum(Jdense > J53_mine+1e-15)))
    # normals and an FD check on a few cells
    normals = np.stack([normal_at(U_all[i], xstar[i], grid.spacing) for i in range(len(cells))]) if len(cells) \
        else np.zeros((0, 8, 3))
    fd = []
    rng = np.random.default_rng(0)
    for i in rng.choice(len(cells), size=min(5, len(cells)), replace=False) if len(cells) else []:
        Uc = U_all[i]
        for (aa, ii) in ((0, 0), (5, 1), (7, 2)):
            h = 1e-6
            Up, Um = Uc.copy(), Uc.copy()
            Up[aa, ii] += h
            Um[aa, ii] -= h
            d = (J_points(Up[None], xstar[i][None], grid.spacing)[0, 0]
                 - J_points(Um[None], xstar[i][None], grid.spacing)[0, 0])/(2*h)
            fd.append(abs(d-normals[i, aa, ii])/max(abs(d), 1e-12))
    doc["implementation_checks"]["normal_fd_max_rel_diff"] = float(max(fd)) if fd else None
    print(f"[{a.label}] cells sampled {len(cells)}; checks {doc['implementation_checks']}", flush=True)
    rows_n, vals_n = free_columns(cells, normals, grid)
    # extra normals (multi-point cone)
    extra_owner, extra_normals = [], []
    for i, pts in enumerate(multi):
        for x in pts:
            extra_owner.append(i)
            extra_normals.append(normal_at(U_all[i], x, grid.spacing))
    extra_owner = np.array(extra_owner, dtype=int)
    if extra_normals:
        rows_e, vals_e = free_columns(cells[extra_owner], np.stack(extra_normals), grid)
    else:
        rows_e, vals_e = [], []
    # ---- 3. NNLS per threshold
    root = MetricRoot(metric, grid)
    gM = root.apply(g)
    gsq = float(g@g)
    blocks_g = {bname: gradient(state, point, blocks=(bname,)) for bname in BLOCKS}
    doc["gradient_blocks_GF"] = {k: float(np.linalg.norm(v)/norm) for k, v in blocks_g.items()}
    nnls_doc, posthoc_doc = {}, {}
    jobs = [(nnls_doc, "b_K", t, np.flatnonzero(bfin[cells] < t), t in DESCRIPTIVE) for t in THRESHOLDS+DESCRIPTIVE]
    # post hoc (added after the first pass on R1-k500 showed b_K far below the true cell minimum; not pre-registered):
    # near-active by the dense-sampled minimum J_dense < t (cells with J_dense < .25 all have b_K < .25, so sampled)
    jobs += [(posthoc_doc, "J_dense", t, np.flatnonzero(Jdense < t), True) for t in POSTHOC]
    for target_doc, by, t, sel, descriptive in jobs:
        entry = dict(t=t, selected_by=by, n_cells=int(len(sel)), descriptive_only=descriptive)
        if len(sel) == 0:
            entry.update(r=GF, r_over_GF=1.0, explained_share=0.0, note="no near-active cell")
            target_doc[str(t)] = entry
            continue
        cols_r = [rows_n[i] for i in sel]
        cols_v = [vals_n[i] for i in sel]
        support = np.unique(np.concatenate(cols_r)) if cols_r else np.zeros(0, int)
        rest = np.ones(g.size, bool)
        rest[support] = False
        A = dense_matrix(cols_r, cols_v, support)
        for sign_name, sgn in (("g", 1.), ("minus_g", -1.)):
            lam, res = nnls_fit(A, sgn*g[support], float(g[rest]@g[rest]))
            key = "" if sign_name == "g" else "minus_g_"
            entry[key+"r"] = res/norm
            entry[key+"r_over_GF"] = res/np.sqrt(gsq)
            entry[key+"explained_share"] = 1-res**2/gsq
            if sign_name == "g":
                nz_ = lam > 0
                entry["r_over_T_B"] = res/norm/row["T_B"]
                entry["lambda"] = dict(n_nonzero=int(nz_.sum()), summary=summary(lam[nz_]),
                                       top=[dict(cell=int(cells[sel[j]]), lam=float(lam[j]), b_K=float(bfin[cells[sel[j]]]),
                                                 J_dense=float(Jdense[sel[j]]), J53=float(J53_mine[sel[j]]),
                                                 x_star=xstar[sel[j]].tolist(), where=location_type(xstar[sel[j]]))
                                            for j in np.argsort(-lam)[:10] if lam[j] > 0],
                                       b_K_of_nonzero=summary(bfin[cells[sel[nz_]]]))
                fitted = np.zeros_like(g)
                fitted[support] = A@lam
                entry["cos_g_fitted"] = float(g@fitted/max(np.sqrt(gsq)*np.linalg.norm(fitted), 1e-300))
        entry["support_rows"] = int(len(support))
        entry["support_upper_bound_share"] = float(g[support]@g[support]/gsq)
        # per-block gradient on the support nodes
        entry["blocks_on_support_GF"] = {k: float(np.linalg.norm(v[support])/norm) for k, v in blocks_g.items()}
        entry["total_on_support_GF"] = float(np.linalg.norm(g[support])/norm)
        # multi-point cone
        ex = np.flatnonzero(np.isin(extra_owner, sel))
        if len(ex):
            cr = cols_r+[rows_e[j] for j in ex]
            cv = cols_v+[vals_e[j] for j in ex]
            sup2 = np.unique(np.concatenate(cr))
            rest2 = np.ones(g.size, bool)
            rest2[sup2] = False
            A2 = dense_matrix(cr, cv, sup2)
            lam2, res2 = nnls_fit(A2, g[sup2], float(g[rest2]@g[rest2]))
            _, res2m = nnls_fit(A2, -g[sup2], float(g[rest2]@g[rest2]))
            entry["multi_point"] = dict(n_columns=len(cr), r_over_GF=res2/np.sqrt(gsq), explained_share=1-res2**2/gsq,
                                        minus_g_explained_share=1-res2m**2/gsq)
        # M^-1 norm (full rows; columns M^-1/2 n_K)
        if len(sel) <= 2000:
            AM = np.zeros((g.size, len(sel)))
            for j, (r_, v_) in enumerate(zip(cols_r, cols_v)):
                col = np.zeros(g.size)
                col[r_] = v_
                AM[:, j] = root.apply(col)
            lamM, resM = nnls_fit(AM, gM)
            gMsq = float(gM@gM)
            entry["M_inverse_norm"] = dict(r_over_gnorm=resM/np.sqrt(gMsq), explained_share=1-resM**2/gMsq,
                                           n_nonzero=int(np.sum(lamM > 0)))
            del AM
        target_doc[str(t)] = entry
        print(f"[{a.label}] {by}<{t:g}: cells {len(sel)} r/GF {entry['r_over_GF']:.4f} share {entry['explained_share']:.4f}"
              f" (-g {entry['minus_g_explained_share']:.4f}; support bound {entry['support_upper_bound_share']:.4f};"
              f" M {entry.get('M_inverse_norm', {}).get('r_over_gnorm')})", flush=True)
    doc["nnls"] = nnls_doc
    doc["nnls_true_J_posthoc"] = posthoc_doc
    # ---- 3b (post hoc). JD-scaling factors: certificate estimate (the run's), gauss27+corners, and the true minimum
    r_cert, rec_cert = scaling_factors(u, grid, estimate=CERTIFICATE, cert=cert, cert_config=backend.cert_config)
    r_def, _ = scaling_factors(u, grid)
    m_true = np.maximum(bfin, J_ACT)                         # unsampled cells: true min >= b_K >= .25 -> r = 1
    m_true[cells] = Jdense
    r_true = np.minimum(1., np.maximum(node_minimum(m_true.reshape(grid.cell_shape), grid), 1e-12)/J_ACT)
    node_g2 = (g.reshape(3, n_int)**2).sum(axis=0)
    rc, rd, rt = r_cert.ravel(), r_def.ravel(), r_true.ravel()
    classes = {}
    for name, r_ in (("certificate", rc), ("gauss27+corners", rd), ("true_min", rt)):
        for lim in (1., .1, .01):
            m_ = r_ < lim
            classes[f"{name}_r_lt_{lim:g}"] = dict(nodes=int(m_.sum()), share_g2=float(node_g2[m_].sum()/node_g2.sum()))
    loose = (rc < .01) & (rt >= .1)
    classes["cert_r_lt_0.01_but_true_r_ge_0.1"] = dict(nodes=int(loose.sum()),
                                                        share_g2=float(node_g2[loose].sum()/node_g2.sum()))
    doc["jd_scaling_posthoc"] = dict(record_certificate=rec_cert, classes=classes,
                                     note="r_a = min(1, max(J_a, 1e-12)/.25); true_min uses J_dense of the sampled cells")
    # ---- 4. energy blocks
    en = point.energies
    nb, nc = doc["counts"]["brain_nodes"], doc["counts"]["brain_cells"]
    blocks = {f"channel_{c}": v for c, v in enumerate(en["channel"])}
    blocks.update(diffusion=en["diffusion"], logvol=en["logvol"], curl=en["curl"])
    parts = en.get("logvol_parts")
    if parts:
        blocks["logvol_gauss"] = parts["gauss"]
        blocks["logvol_closed27"] = parts["closed27"]
    doc["energy"] = dict(F_B=point.energy, blocks=blocks,
                         data_total=float(sum(en["channel"])),
                         per_brain_node={k: v/nb for k, v in blocks.items()},
                         per_brain_cell={k: v/nc for k, v in blocks.items()},
                         logvol_parts_record=parts)
    # ---- 5. quadrature blind spot on the near-active cells (t = 1e-2) and on t = 1e-3
    blind = {}
    for t in (1e-2, 1e-3, .1):
        sel = np.flatnonzero(bfin[cells] < t)
        if len(sel) == 0:
            blind[str(t)] = None
            continue
        J53s, Jd, bK = J53_mine[sel], Jdense[sel], bfin[cells[sel]]
        where = [location_type(xstar[i]) for i in sel]
        blind[str(t)] = dict(
            n=int(len(sel)), J53=summary(J53s), J_dense=summary(Jd), b_K=summary(bK),
            ratio_J53_over_dense=summary(J53s/np.maximum(Jd, 1e-300)),
            ratio_bK_over_dense=summary(bK/np.maximum(Jd, 1e-300)),
            n_J53_gt_10x_dense=int(np.sum(J53s > 10*Jd)), n_J53_gt_2x_dense=int(np.sum(J53s > 2*Jd)),
            n_dense_le_0=int(np.sum(Jd <= 0)),
            argmin_location={k: where.count(k) for k in ("interior", "face", "edge", "corner")},
            n_brain_cells=int(np.sum(brain_cell[cells[sel]])),
            cells=[dict(cell=int(cells[i]), b_K=float(bfin[cells[i]]), J_dense=float(Jdense[i]), J53=float(J53_mine[i]),
                        J53_slab=float(J53_min[cells[i]]), x_star=xstar[i].tolist(), where=location_type(xstar[i]))
                   for i in sel[np.argsort(Jdense[sel])][:30]])
    doc["blind_spot"] = blind
    doc["seconds"] = time.perf_counter()-tick0
    doc["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    common.write_json(target, doc)
    print(f"[{a.label}] done {doc['seconds']:.1f} s, peak RSS {doc['peak_rss_mb']:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
