"""DIAG (work package DIAG, 2026-10-30): why the G3 arm of the m2 GPU smoke ended line_search_failed (受阻收尾).

Diagnosis only: no change to jd_curl/ or to the logic of bayesreg/; this script imports internal functions.

Inputs (read only): the G3 run directory (config.json, hyper.json, log.jsonl, cert.json, final_native.npz), the
development pair through the label firewall (bayesreg.io.load_pair: pair.npz members fixed / moving only, the
development allowlist), the Atropos tissue layer through run_pair.prepare_channels (channel_source tissue).

Rebuilt (native level, the run's last state): the native ModelState (eta of config.json, the native round's sigma2_used
of hyper.json, gauss27+corners, tile 8192, 4 CPU threads), the torch64 cuda backend, F_B and its gradient at u_final,
the frozen metric M of the native round (recorded c_dat and b_P), S~ at u_final (the solver's last refresh, refresh 28
at iteration 36, was computed at this very state), and the direction of the failed iteration 36:
    p = -S~ M^-1 S~ g.
Checks against the run record: F_B, GF_B, min J_a of the last refresh, the domain bound of iteration 36, and the
certificate min-J lower bound of every iteration-36 candidate (tau from log.jsonl).

Stages
  base   per-cell certificate bounds at u_final (the per-cell path of certify_map, edge hull -> float prescreen ->
         interval Bernstein fallback, replicated here; its minimum must equal cert.json), the critical cells (smallest
         certified lower bound), their true min J (dense sampling + bounded local refinement of the Q1 J), its
         location in the reference cell, J at the 8 corners / 27 Gauss points / closed27 points; the gradient by block
         (channel c = 0..3 separately, diffusion, logvol, curl) and the direction by block p_b = -S~ M^-1 S~ g_b; the
         top |p| nodes and the nodes of the critical cells with: position, cells, mask (boundary), channel values P and
         sqrt P (fixed node values, moving spline values at x + u) with gradients, block contributions to g and p,
         metric diagonal (M_aa, (M^-1)_aa), channel GN diagonal, r_a, J_a; dJ/dtau at the critical point along p and
         along every p_b.
  scan   certify(u + tau p) on the iteration-36 tau values and a log grid: status, reason, min J lower bound, witness
         cell, number of failing cells, true min J of the critical cell.
  cf     counterfactual metrics (diagnosis only; a path change of C09.1 / C09.1-3a that would need a contract revision
         and review):  CF1  S~ (M + D_GN)^-1 S~  (D_GN = channel GN diagonal at u_final, C06-6; PCG on the device with
         M^-1 as preconditioner);  CF1b  S~ (M - shift I + D_GN)^-1 S~ (the scalar c_dat replaced by the local GN
         diagonal);  CF2  S~' M^-1 S~' with S~' built from m'_K = min(m_K, certified per-cell lower bound);  CF3  CF1 +
         CF2.  For each: max |p|, p at the critical nodes, dJ/dtau, slope g.p, domain bound, and one PG step of
         bayesreg.solve.run_pg (max_steps 1, tau0 = 1 as for a fresh solve) with the real certificate, energy, stable
         comparison and Armijo; the same step for the baseline metric reproduces line_search_failed.
  short  run_pg from u_final with the listed counterfactual metrics for --short-steps accepted steps (rows saved).

Outputs (small files only) in --out: cmd.txt, git.txt, env.txt, diag.json, nodes.json, scan.json, cf.json,
short_<variant>.jsonl, log.txt.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jd_curl.three_d.boundary import inject_free, restrict_free  # noqa: E402
from jd_curl.three_d.cell_q1 import VERTICES, cell_tiles  # noqa: E402

from bayesreg.geom import finish  # noqa: E402
from bayesreg.model import channel_point as numpy_channel_point  # noqa: E402
from bayesreg.solve import (DirectObjective, Eta, JDScaledMetric, JDScalingConfig, PGConfig, make_backend,  # noqa: E402
                            run_pg, sampling_step_bound, scaling_factors)
from bayesreg.solve.jd_scaling import (GAUSS_NODES, cell_jd_estimate, free_scaling)  # noqa: E402
from bayesreg.solve.multilevel import _template  # noqa: E402
from bayesreg.solve.stopping import gf_b, state_hash  # noqa: E402

NATIVE = 3
DEFAULT_RUN = Path("/home/rmcfz/seq_vp_tag_results/m2_gpu_smoke/G3")
SCHEMA = "diag.g3_direction.v1"
CHANNEL_NAMES = ("intensity", "s1 (sqrt Ptilde_T^1)", "s2 (sqrt Ptilde_T^2)", "s3 (sqrt Ptilde_T^3)")
GAUSS_BLOCKS = ("diffusion", "logvol", "curl")
VERT = np.array(VERTICES, dtype=np.int64)          # (8, 3): (x, y, z) bits, cell_q1 order


# ----------------------------------------------------------------------------------------------
# bookkeeping
# ----------------------------------------------------------------------------------------------

def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=1, default=_json_default)+"\n", encoding="utf-8")


def git_record():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return dict(head=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")), branch=git("branch",
                                                                                                    "--show-current"))


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.items = {}

    def mark(self, name, since):
        self.items[name] = time.perf_counter()-since


# ----------------------------------------------------------------------------------------------
# run record
# ----------------------------------------------------------------------------------------------

def load_record(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir/"config.json").read_text())
    hyper = json.loads((run_dir/"hyper.json").read_text())
    cert = json.loads((run_dir/"cert.json").read_text())
    native = [r for r in hyper["rounds"] if r.get("level") == NATIVE][-1]
    rows = [json.loads(line) for line in (run_dir/"log.jsonl").open()]
    nat = [r for r in rows if r.get("level") == NATIVE]
    candidates = [r for r in nat if r.get("kind") == "candidate"]
    last_iteration = max(r["iteration"] for r in candidates)
    last = [r for r in candidates if r["iteration"] == last_iteration]
    accepts = [r for r in nat if r.get("kind") == "accept"]
    jd = [r for r in nat if r.get("kind") == "jd_scaling"]
    return dict(run_dir=str(run_dir), config=config, args=config["args"], eta=config["eta"],
                tile_size=config["solver"].get("gauss_tile_size", 8192),
                native=dict(sigma2_used=native["sigma2_used"], T_B=native["T_B"], G0=native["G0"],
                            c_dat=native["metric"]["c_dat"], b_P=native["metric"]["b_P"],
                            shift=native["metric"]["shift"], status=native["status"], accepted=native["accepted"],
                            GF_B=native["GF_B"], F_B_end=native["F_B_end"]),
                cert=cert, last_iteration=last_iteration, last_candidates=last,
                last_accept=accepts[-1] if accepts else None, last_jd=jd[-1] if jd else None,
                accepts=[dict(iteration=r["iteration"], tau=r["tau"], min_jd_lower_bound=r["min_jd_lower_bound"],
                              energy=r["energy"], step_ratio=r["step_ratio"]) for r in accepts])


def load_native(record, args, logger):
    """Native ModelState of the run and u_final (read only), through the label firewall (load_pair)."""
    from bayesreg.cli.run_pair import prepare_channels, resolve_pair
    from bayesreg.io import load_pair
    a = record["args"]
    pair = load_pair(resolve_pair(a["data_root"], a["pair"]), allowlist=a["allowlist"])
    if a["channel_source"] != "tissue":
        raise SystemExit("this diagnosis is written for channel_source tissue (G3)")
    levels, _, _, _ = prepare_channels(pair, a["channel_source"], a["tissue_root"], logger)
    native = levels[-1]
    del levels
    e = record["eta"]
    eta = Eta(e["alpha"], e["nu"], tuple(e["tau2"]), source=str(e.get("source")))
    host = _template(native, eta, native.grid.omega, int(record["tile_size"]), native.fixed.shape[0],
                     int(a.get("cpu_threads", 4)), a["logvol_points"])
    state = host.replace(sigma2=np.asarray(record["native"]["sigma2_used"], dtype=np.float64))
    with np.load(Path(record["run_dir"])/"final_native.npz") as data:
        u = np.array(data["u"], dtype=np.float64, order="C")
        stored = str(data["state_hash"])
    digest = state_hash(u)
    if digest != stored:
        raise SystemExit("final_native.npz: state_hash does not match the field")
    u.flags.writeable = False
    info = dict(pair=a["pair"], fixed_subject=pair.fixed_subject, moving_subject=pair.moving_subject,
                keys_read=pair.provenance.get("keys_read"), label_firewall=pair.provenance.get("label_firewall"),
                grid_shape_zyx=list(state.grid.shape), spacing_xyz=list(state.grid.spacing),
                n_mask_nodes=int(state.node_index.size), n_channels=state.n_channels,
                logvol_points=state.logvol_points, sigma2=state.sigma2.tolist(), eta=eta.record(), state_hash=digest)
    return state, native, u, info


# ----------------------------------------------------------------------------------------------
# Q1 cell geometry (diagnostic float64)
# ----------------------------------------------------------------------------------------------

def dshape(pts):
    """dN_v/dxi_j at reference points pts (P, 3) in (x, y, z): (P, 8, 3)."""
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    out = np.empty((pts.shape[0], 8, 3))
    for v, bits in enumerate(VERTICES):
        f = [pts[:, j] if bits[j] else 1-pts[:, j] for j in range(3)]
        d = [1. if bits[j] else -1. for j in range(3)]
        out[:, v, 0] = d[0]*f[1]*f[2]
        out[:, v, 1] = f[0]*d[1]*f[2]
        out[:, v, 2] = f[0]*f[1]*d[2]
    return out


def cell_grad(verts, spacing, pts):
    """grad[p, r, j] = d_j v_r of the Q1 field with vertex values verts (8, 3) (components x, y, z)."""
    dN = dshape(pts)/np.asarray(spacing, dtype=np.float64)[None, None, :]
    return np.einsum("vr,pvj->prj", verts, dN)


def cofactor(F):
    C = np.empty_like(F)
    C[..., 0, 0] = F[..., 1, 1]*F[..., 2, 2]-F[..., 1, 2]*F[..., 2, 1]
    C[..., 0, 1] = F[..., 1, 2]*F[..., 2, 0]-F[..., 1, 0]*F[..., 2, 2]
    C[..., 0, 2] = F[..., 1, 0]*F[..., 2, 1]-F[..., 1, 1]*F[..., 2, 0]
    C[..., 1, 0] = F[..., 0, 2]*F[..., 2, 1]-F[..., 0, 1]*F[..., 2, 2]
    C[..., 1, 1] = F[..., 0, 0]*F[..., 2, 2]-F[..., 0, 2]*F[..., 2, 0]
    C[..., 1, 2] = F[..., 0, 1]*F[..., 2, 0]-F[..., 0, 0]*F[..., 2, 1]
    C[..., 2, 0] = F[..., 0, 1]*F[..., 1, 2]-F[..., 0, 2]*F[..., 1, 1]
    C[..., 2, 1] = F[..., 0, 2]*F[..., 1, 0]-F[..., 0, 0]*F[..., 1, 2]
    C[..., 2, 2] = F[..., 0, 0]*F[..., 1, 1]-F[..., 0, 1]*F[..., 1, 0]
    return C


def cell_J(verts, spacing, pts):
    F = np.eye(3)+cell_grad(verts, spacing, pts)
    C = cofactor(F)
    return np.einsum("pj,pj->p", F[:, 0, :], C[:, 0, :])


def cell_dJ(verts, pverts, spacing, pts):
    """d/dtau J(u + tau p) at tau = 0: sum_rj cof(F)_rj d_j p_r."""
    F = np.eye(3)+cell_grad(verts, spacing, pts)
    return np.einsum("prj,prj->p", cofactor(F), cell_grad(pverts, spacing, pts))


def grid_points(n):
    t = np.linspace(0., 1., n)
    z, y, x = np.meshgrid(t, t, t, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)


GRID33 = grid_points(33)
CLOSED27 = grid_points(3)
CORNERS8 = VERT.astype(np.float64)
GAUSS27 = np.array([(x, y, z) for z in GAUSS_NODES for y in GAUSS_NODES for x in GAUSS_NODES])


def cell_min_J(verts, spacing):
    """(min J, point) over the reference cell: 33^3 samples, then bounded L-BFGS-B from the 4 best samples."""
    from scipy.optimize import minimize
    values = cell_J(verts, spacing, GRID33)
    best = (float(values.min()), GRID33[int(values.argmin())])
    for k in np.argsort(values)[:4]:
        res = minimize(lambda x: float(cell_J(verts, spacing, x[None])[0]), GRID33[k], method="L-BFGS-B",
                       bounds=[(0., 1.)]*3, options=dict(ftol=1e-15, gtol=1e-13, maxiter=500))
        if res.fun < best[0]:
            best = (float(res.fun), np.clip(res.x, 0., 1.))
    return best


def point_kind(pt, tol=1e-6):
    on = int(sum(1 for t in pt if t < tol or t > 1-tol))
    return {3: "corner", 2: "edge", 1: "face", 0: "interior"}[on]


def cell_zyx(grid, cell):
    return tuple(int(v) for v in np.unravel_index(int(cell), grid.cell_shape))


def cell_nodes(grid, cell):
    cz, cy, cx = cell_zyx(grid, cell)
    return [(cz+z, cy+y, cx+x) for x, y, z in VERTICES]


def node_field_values(field, nodes):
    return np.array([field[z, y, x] for z, y, x in nodes])


def free_view(vec, grid):
    return np.asarray(vec).reshape((3,)+grid.interior_shape)


def free_at(vec, grid, nodes):
    """(n, 3) x/y/z components of a free vector at full-grid nodes (0 on the faces)."""
    v = free_view(vec, grid)
    nz, ny, nx = grid.shape
    out = np.zeros((len(nodes), 3))
    for i, (z, y, x) in enumerate(nodes):
        if 0 < z < nz-1 and 0 < y < ny-1 and 0 < x < nx-1:
            out[i] = v[:, z-1, y-1, x-1]
    return out


def cells_of_node(grid, node):
    z, y, x = node
    cz, cy, cx = grid.cell_shape
    out = []
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                c = (z-dz, y-dy, x-dx)
                if 0 <= c[0] < cz and 0 <= c[1] < cy and 0 <= c[2] < cx:
                    out.append(int(np.ravel_multi_index(c, grid.cell_shape)))
    return out


# ----------------------------------------------------------------------------------------------
# per-cell certificate bounds (the per-cell path of certify_map, replicated read-only)
# ----------------------------------------------------------------------------------------------

HOW = {1: "edge_hull", 2: "prescreen", 3: "bernstein_root", 4: "subdivision_feasible", 5: "uncertain",
       6: "infeasible"}


def cert_cell_bounds(u, grid, config):
    """Per cell: the lower bound certify_map's margins list gets from that cell (edge hull, float prescreen,
    Bernstein root or subdivision; for uncertain / infeasible cells the root Bernstein minimum, which certify_map
    also puts in its margins), the path (HOW), and the root Bernstein minimum of every cell that reached the
    fallback.  Same order of tests as jd_curl.three_d.feasibility._certify_map_parallel (delta = 0)."""
    from jd_curl.three_d.cell_q1 import CellQ1Operators3D
    from jd_curl.three_d.certificate_prescreen import prescreen_lower_bounds
    from jd_curl.three_d.compiled_enclosure import parallel_margins
    from jd_curl.three_d.compiled_fallback import fallback_arrays
    from jd_curl.three_d.feasibility import _classify, _node_indices_for
    if config.delta != 0:
        raise SystemExit("cert_cell_bounds: delta 0 only")
    u = np.ascontiguousarray(u, dtype=np.float64)
    n = grid.n_cells
    bound = np.full(n, np.nan)
    how = np.zeros(n, dtype=np.int8)
    root = {}
    margin, flag = parallel_margins(u, grid, config.delta)
    unproven = []
    for tile in cell_tiles(grid, config.tile_size):
        ids = np.arange(tile.start, tile.stop)
        if bool(flag[tile].any()):
            proven = np.zeros(len(ids), dtype=bool)
        else:
            proven = margin[tile] > 0
        bound[ids[proven]] = margin[tile][proven]
        how[ids[proven]] = 1
        unproven.append(ids[~proven])
    cells = np.concatenate(unproven) if unproven else np.zeros(0, dtype=np.int64)
    if len(cells):
        table = CellQ1Operators3D(grid, location="closed27_uniform").coefficients
        lower = prescreen_lower_bounds(u, grid, table, cells)
        with np.errstate(invalid="ignore"):
            ok = lower > 0
        bound[cells[ok]] = lower[ok]
        how[cells[ok]] = 2
        rest = cells[~ok]
    else:
        rest = cells
    flat = u.reshape(-1, 3)
    errors = np.geterr()
    for lo in range(0, len(rest), 8192):
        chunk = rest[lo:lo+8192]
        arrays = fallback_arrays(flat[_node_indices_for(grid, chunk)], grid.spacing, config.delta, errors)
        if arrays is None:
            how[chunk] = 5
            continue
        p, coefficients, samples = arrays
        minima = coefficients.lo.min(axis=(-3, -2, -1))
        for i, c in enumerate(chunk):
            root[int(c)] = float(minima[i])
        safe = minima > 0
        bound[chunk[safe]] = minima[safe]
        how[chunk[safe]] = 3
        for local in np.flatnonzero(~safe):
            res = _classify(coefficients[local], p[local], config.delta, float(samples[local].min()), config)
            c = int(chunk[local])
            bound[c] = np.nan if res.min_margin_lower_bound is None else res.min_margin_lower_bound
            how[c] = {"certified_feasible": 4, "uncertain": 5, "infeasible": 6}[res.status]
    return bound, how, root


def node_min_of_cells(mcell, grid):
    """min over the (up to) 8 cells adjacent to every interior node (as jd_scaling.node_jd_estimate)."""
    nz, ny, nx = grid.interior_shape
    out = mcell[:nz, :ny, :nx].copy()
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                if dz or dy or dx:
                    np.minimum(out, mcell[dz:dz+nz, dy:dy+ny, dx:dx+nx], out=out)
    return out


def cert_scaling_factors(u, grid, cert_config, *, activation=.25, floor=1e-12):
    """CF2: r_a with m'_K = min(m_K, certified per-cell lower bound) (cells with no bound: m_K)."""
    tick = time.perf_counter()
    m = cell_jd_estimate(u, grid)
    bound, how, _ = cert_cell_bounds(u, grid, cert_config)
    b = bound.reshape(grid.cell_shape)
    with np.errstate(invalid="ignore"):
        m2 = np.where(np.isfinite(b), np.minimum(m, b), m)
    J = node_min_of_cells(m2, grid)
    r = np.minimum(1., np.maximum(J, floor)/activation)
    scaled = int(np.count_nonzero(r < 1))
    record = dict(min_J_a=float(J.min()), fraction_r_lt_1=scaled/r.size, n_r_lt_1=scaled, n_nodes=int(r.size),
                  min_r=float(r.min()), n_r_lt_1e_3=int(np.count_nonzero(r < 1e-3)), estimate="min(m_K, cert bound)",
                  seconds=time.perf_counter()-tick)
    return r, record


class CertJDScaledMetric(JDScaledMetric):
    """CF2 (diagnosis only): C09.1-3a with the node estimate built from min(m_K, certified per-cell bound)."""

    def __init__(self, metric, grid, u, *, cert_config, **kwargs):
        self.cert_config = cert_config
        super().__init__(metric, grid, u, **kwargs)

    def _compute(self, u, **context):
        r, record = cert_scaling_factors(u, self.grid, self.cert_config, activation=self.config.activation,
                                         floor=self.config.floor)
        self.r = r
        self.scale = free_scaling(r)
        record = dict(context, **record)
        self.computations.append(record)
        self._consecutive, self._due = 0, None
        self._window.clear()
        return record


# ----------------------------------------------------------------------------------------------
# CF1: M + D_GN by PCG on the device
# ----------------------------------------------------------------------------------------------

class GNMetric:
    """(M + D)^-1 or (M - shift I + D)^-1 (local=True) applied by PCG with M^-1 (DST) as preconditioner."""

    def __init__(self, host_metric, D, dev, *, local=False, tol=1e-10, maxit=1000, log=None):
        import torch
        self.torch = torch
        self.M = host_metric.device_metric
        self.dev = dev
        self.D = dev.upload(np.asarray(D, dtype=np.float64))
        if local:
            self.D = self.D-float(host_metric.shift)
        self.local, self.tol, self.maxit = local, tol, maxit
        self.a, self.shift = host_metric.a, host_metric.shift
        self.records = []
        self.log = log
        self._last = None

    def _apply(self, x):
        return self.M.matvec(x)+self.D*x

    def solve(self, g):
        g = np.asarray(g, dtype=np.float64)
        if self._last is not None and np.array_equal(self._last[0], g):
            return self._last[1].copy()                  # one-entry cache: run_pg repeats the direction solve
        out = self._solve(g)
        self._last = (g.copy(), out.copy())
        return out

    def _solve(self, g):
        torch = self.torch
        b = self.dev.upload(g)
        tick = time.perf_counter()
        x = torch.zeros_like(b)
        r = b.clone()
        z = self.M.solve(r)
        p = z.clone()
        rz = torch.dot(r, z)
        bnorm = float(torch.linalg.norm(b))
        rel, k = 1., 0
        for k in range(1, self.maxit+1):
            Ap = self._apply(p)
            alpha = rz/torch.dot(p, Ap)
            x = x+alpha*p
            r = r-alpha*Ap
            rel = float(torch.linalg.norm(r))/bnorm
            if rel <= self.tol:
                break
            z = self.M.solve(r)
            rz_new = torch.dot(r, z)
            p = z+(rz_new/rz)*p
            rz = rz_new
        true_rel = float(torch.linalg.norm(b-self._apply(x)))/bnorm
        rec = dict(iterations=k, rel_residual=rel, true_rel_residual=true_rel, seconds=time.perf_counter()-tick)
        self.records.append(rec)
        if self.log:
            self.log(f"PCG ({'local' if self.local else 'M + D'}): {rec}")
        out = self.dev.download(x)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("PCG: nonfinite solution")
        return out

    def matvec(self, v):
        return self.dev.download(self._apply(self.dev.upload(np.asarray(v, dtype=np.float64))))


# ----------------------------------------------------------------------------------------------
# node report
# ----------------------------------------------------------------------------------------------

def mask_boundary(mask, node):
    z, y, x = node
    nz, ny, nx = mask.shape
    sl = mask[max(z-1, 0):min(z+2, nz), max(y-1, 0):min(y+2, ny), max(x-1, 0):min(x+2, nx)]
    return bool(mask[z, y, x]) and not bool(sl.all()), bool(mask[z, y, x]) or bool(sl.any())


def fixed_gradient(arr, node, spacing):
    z, y, x = node
    hx, hy, hz = spacing
    nz, ny, nx = arr.shape

    def d(axis, h):
        idx = [z, y, x]
        n = arr.shape[axis]
        lo, hi = list(idx), list(idx)
        lo[axis] = max(idx[axis]-1, 0)
        hi[axis] = min(idx[axis]+1, n-1)
        return float((arr[tuple(hi)]-arr[tuple(lo)])/((hi[axis]-lo[axis])*h))
    return [d(2, hx), d(1, hy), d(0, hz)]


def moving_samples(state, u, nodes, identity):
    from jd_curl.three_d.interpolation import sample_points
    pts = np.array([identity[z, y, x]+u[z, y, x] for z, y, x in nodes])
    nz, ny, nx = state.grid.shape
    strict = np.array([0 < z < nz-1 and 0 < y < ny-1 and 0 < x < nx-1 for z, y, x in nodes], dtype=bool)
    out = []
    for sampler in state.samplers:
        s = sample_points(sampler, pts, strict=strict)
        out.append((np.asarray(s.value), np.asarray(s.gradient)))
    return out


def unit_diag(host_metric, grid, nodes):
    """M_aa and (M^-1)_aa (component x; the operator is the same scalar operator on each component)."""
    n = 3*grid.n_free_scalar
    out = []
    for z, y, x in nodes:
        nz, ny, nx = grid.shape
        if not (0 < z < nz-1 and 0 < y < ny-1 and 0 < x < nx-1):
            out.append((None, None))
            continue
        e = np.zeros(n)
        k = int(np.ravel_multi_index((z-1, y-1, x-1), grid.interior_shape))
        e[k] = 1.
        out.append((float(host_metric.matvec(e)[k]), float(host_metric.solve(e)[k])))
    return out


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", default=str(DEFAULT_RUN))
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stages", default="base,scan,cf,short")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--critical", type=int, default=12, help="critical cells reported (smallest certified bound)")
    p.add_argument("--short", default="cf2", help="variants of the short run (comma list of cf1,cf2,cf3,base)")
    p.add_argument("--short-steps", type=int, default=15)
    p.add_argument("--pcg-tol", type=float, default=1e-8)
    p.add_argument("--pcg-maxit", type=int, default=1500)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    stages = set(s.strip() for s in args.stages.split(",") if s.strip())
    logger = logging.getLogger("diag_g3")
    logger.setLevel(logging.INFO)
    for h in (logging.FileHandler(out/"log.txt"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    info = logger.info
    (out/"cmd.txt").write_text(" ".join([sys.executable, *sys.argv])+f"\n# cwd {os.getcwd()}\n")
    write_json(out/"git.txt", git_record())
    import torch
    env = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
               cuda=torch.version.cuda, threads={k: os.environ.get(k) for k in (
                   "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                   "NUMBA_CACHE_DIR")})
    write_json(out/"env.txt", env)
    T = Timer()
    diag = dict(schema=SCHEMA, run_dir=args.run_dir, git=git_record(), stages=sorted(stages))

    # ---- record, state, backend ----
    tick = time.perf_counter()
    record = load_record(args.run_dir)
    state, native_input, u, inputs = load_native(record, args, logger)
    grid = state.grid
    T.mark("load_inputs_state", tick)
    info(f"inputs: {inputs}")
    diag["inputs"] = inputs
    diag["record"] = {k: record[k] for k in ("native", "last_iteration", "last_accept", "last_jd")}
    theta = restrict_free(u, grid)
    backend = make_backend("torch64", device=args.device)
    cert_config = backend.cert_config
    tick = time.perf_counter()
    state = backend.prepare(state)
    ev = backend.evaluate(state, u)
    g = backend.gradient(state, ev)
    G = gf_b(g, grid.omega, state.omega0)
    T.mark("evaluate_gradient", tick)
    checks = dict(F_B=ev.energy, F_B_log=record["last_accept"]["energy"],
                  F_B_rel=abs(ev.energy-record["last_accept"]["energy"])/abs(record["last_accept"]["energy"]),
                  GF_B=G, GF_B_log=record["native"]["GF_B"], GF_B_rel=abs(G-record["native"]["GF_B"])/G,
                  energies=ev.energies, eval_path=ev.path)
    info(f"checks F_B/GF_B: {checks}")
    metric = backend.build_metric(state, c_dat=record["native"]["c_dat"], b_P=record["native"]["b_P"])
    bP_now = backend.metric_bP(state, mode="principal")
    checks.update(b_P_recomputed=bP_now, b_P_record=record["native"]["b_P"], shift=metric.shift,
                  shift_record=record["native"]["shift"])
    tick = time.perf_counter()
    r, rrec = scaling_factors(u, grid)
    T.mark("scaling_factors", tick)
    scale = free_scaling(r)
    checks.update(jd_record=rrec, jd_last_refresh=record["last_jd"],
                  min_J_a_equal=rrec["min_J_a"] == record["last_jd"]["min_J_a"])
    p = -scale*metric.solve(scale*g)
    slope = float(g@p)
    bound = sampling_step_bound(grid, u, p)
    lc = record["last_candidates"]
    checks.update(slope=slope, domain_bound=bound, domain_bound_log=lc[0]["domain_bound"],
                  domain_bound_rel=abs(bound-lc[0]["domain_bound"])/lc[0]["domain_bound"])
    info(f"checks metric/JD/direction: b_P {bP_now} vs {record['native']['b_P']}; min J_a {rrec['min_J_a']} vs "
         f"{record['last_jd']['min_J_a']}; domain bound {bound} vs {lc[0]['domain_bound']}; slope {slope}")
    diag["checks"] = checks
    pn = np.sqrt((free_view(p, grid)**2).sum(axis=0))
    gn = np.sqrt((free_view(g, grid)**2).sum(axis=0))
    diag["direction"] = dict(max_abs_p_node=float(pn.max()), p_node_quantiles={q: float(np.quantile(pn, q)) for q in (
        .5, .9, .99, .999, .9999)}, max_abs_g_node=float(gn.max()), norm_p=float(np.linalg.norm(p)),
        norm_g=float(np.linalg.norm(g)), tau_dom=bound, max_disp_at_tau_dom=float(pn.max()*bound))
    info(f"direction: {diag['direction']}")
    write_json(out/"diag.json", diag)

    # ---- per-cell certificate bounds, critical cells ----
    tick = time.perf_counter()
    cbound, chow, croot = cert_cell_bounds(u, grid, cert_config)
    T.mark("cert_cell_bounds", tick)
    finite = np.isfinite(cbound)
    order = np.argsort(np.where(finite, cbound, np.inf))
    mK = cell_jd_estimate(u, grid)
    mKf = mK.ravel()
    diag["cert_cells"] = dict(min_bound=float(np.nanmin(cbound)), cert_json_min=record["cert"]["min_jd_lower_bound"],
                              equal_cert_json=float(np.nanmin(cbound)) == record["cert"]["min_jd_lower_bound"],
                              path_counts={HOW[k]: int(np.count_nonzero(chow == k)) for k in HOW},
                              n_bound_lt={str(t): int(np.count_nonzero(cbound < t)) for t in (1e-8, 1e-6, 1e-4, 1e-2,
                                                                                               .05)},
                              n_fallback_cells=len(croot))
    info(f"cert cells: {diag['cert_cells']}")
    crit = []
    for c in order[:args.critical]:
        c = int(c)
        nodes = cell_nodes(grid, c)
        verts = node_field_values(u, nodes)
        jmin, pt = cell_min_J(verts, grid.spacing)
        jc = cell_J(verts, grid.spacing, CORNERS8)
        jg = cell_J(verts, grid.spacing, GAUSS27)
        j27 = cell_J(verts, grid.spacing, CLOSED27)
        mask_flags = [bool(native_input.mask[n]) for n in nodes]
        crit.append(dict(cell=c, cell_zyx=cell_zyx(grid, c), cert_bound=float(cbound[c]), cert_path=HOW[int(chow[c])],
                         root_bernstein=croot.get(c), m_K=float(mKf[c]), true_min_J=jmin, argmin_xyz=pt.tolist(),
                         argmin_kind=point_kind(pt), J_corners=jc.tolist(), J_gauss_min=float(jg.min()),
                         J_closed27=j27.tolist(), mask_nodes=mask_flags,
                         tissue_fixed_nodes=[native_input.tissue[:, z, y, x].tolist() for z, y, x in nodes]))
    diag["critical_cells"] = crit
    for c in crit[:5]:
        info(f"critical cell {c['cell']} {c['cell_zyx']}: cert {c['cert_bound']:.3e} ({c['cert_path']}), root "
             f"{c['root_bernstein']}, m_K {c['m_K']:.4f}, true min J {c['true_min_J']:.3e} at {c['argmin_xyz']} "
             f"({c['argmin_kind']}), corners min {min(c['J_corners']):.4f}, gauss min {c['J_gauss_min']:.4f}")
    write_json(out/"diag.json", diag)
    Kstar = crit[0]["cell"]
    Knodes = cell_nodes(grid, Kstar)
    Kverts = node_field_values(u, Knodes)
    Kpt = np.array(crit[0]["argmin_xyz"])[None]

    # ---- block gradients and block directions (streamed) ----
    if "base" in stages:
        tick = time.perf_counter()
        top = np.argsort(pn.ravel())[::-1][:args.top]
        top_nodes = [tuple(int(v)+1 for v in np.unravel_index(int(k), grid.interior_shape)) for k in top]
        report_nodes = list(dict.fromkeys(top_nodes+Knodes))
        blocks = {}
        host_cp = numpy_channel_point(state, u)
        nidx = state.node_index
        g_sum = np.zeros_like(g)
        p_sum = np.zeros_like(p)
        dJ = {}
        Kp_total = free_at(p, grid, Knodes)
        dJ["total"] = float(cell_dJ(Kverts, Kp_total, grid.spacing, Kpt)[0])
        names = [f"channel{c}" for c in range(state.n_channels)]+list(GAUSS_BLOCKS)
        for name in names:
            if name.startswith("channel"):
                c = int(name[-1])
                acc = np.zeros((3,)+grid.shape)
                coef = state.beta[c]*host_cp.r[c]
                acc.reshape(3, -1)[:, nidx] += (coef[:, None]*host_cp.grad[c]).T
                gb = finish(acc)
                del acc
            else:
                gb = backend.gradient(state, ev, blocks=(name,))
            pb = -scale*metric.solve(scale*gb)
            g_sum += gb
            p_sum += pb
            dJ[name] = float(cell_dJ(Kverts, free_at(pb, grid, Knodes), grid.spacing, Kpt)[0])
            blocks[name] = dict(g=free_at(gb, grid, report_nodes), p=free_at(pb, grid, report_nodes),
                                norm_g=float(np.linalg.norm(gb)), norm_p=float(np.linalg.norm(pb)),
                                slope=float(g@pb), max_p_node=float(np.sqrt((free_view(pb, grid)**2).sum(0)).max()))
            del gb, pb
        diag["block_sum_check"] = dict(g_rel=float(np.linalg.norm(g_sum-g)/np.linalg.norm(g)),
                                       p_rel=float(np.linalg.norm(p_sum-p)/np.linalg.norm(p)))
        del g_sum, p_sum
        info(f"block sums: {diag['block_sum_check']}")
        diag["blocks"] = {k: dict(norm_g=v["norm_g"], norm_p=v["norm_p"], slope=v["slope"], max_p_node=v["max_p_node"])
                          for k, v in blocks.items()}
        J0 = crit[0]["true_min_J"]
        diag["critical_dJ_dtau"] = dict(cell=Kstar, point=crit[0]["argmin_xyz"], J0=J0, dJ_dtau=dJ,
                                        tau_linear_zero=J0/(-dJ["total"]) if dJ["total"] < 0 else None)
        info(f"critical dJ/dtau: {diag['critical_dJ_dtau']}")
        # node reports
        from bayesreg.solve.jd_scaling import node_jd_estimate
        Jnode = node_jd_estimate(u, grid)
        identity = grid.identity()
        weights = grid.node_weights()
        moving = moving_samples(state, u, report_nodes, identity)
        diags = unit_diag(metric, grid, report_nodes)
        D = backend.gn_diagonal(state, ev)
        Dn = free_at(D, grid, report_nodes)
        pn_rep = free_at(p, grid, report_nodes)
        gn_rep = free_at(g, grid, report_nodes)
        nodes_out = []
        for i, node in enumerate(report_nodes):
            z, y, x = node
            boundary, near = mask_boundary(native_input.mask, node)
            interior = 0 < z < grid.shape[0]-1 and 0 < y < grid.shape[1]-1 and 0 < x < grid.shape[2]-1
            ch = []
            for c in range(state.n_channels):
                f = float(state.fixed[c, z, y, x])
                mv, mg = moving[c][0][i], moving[c][1][i]
                beta = float(np.sqrt(weights[z, y, x]/(state.omega0*state.sigma2[c])))
                ch.append(dict(name=CHANNEL_NAMES[c], fixed=f, fixed_P=f*f if c else None,
                               fixed_grad_xyz=fixed_gradient(state.fixed[c], node, grid.spacing),
                               moving=float(mv), moving_P=float(mv*mv) if c else None, moving_grad_xyz=mg.tolist(),
                               residual=float(mv-f), beta=beta if bool(native_input.mask[node]) else 0.,
                               g=blocks[f"channel{c}"]["g"][i].tolist(), p=blocks[f"channel{c}"]["p"][i].tolist()))
            adj = cells_of_node(grid, node)
            nodes_out.append(dict(
                node_zyx=list(node), xyz=identity[z, y, x].tolist(), interior=interior,
                in_top=node in top_nodes, in_critical_cell=node in Knodes, mask=bool(native_input.mask[node]),
                mask_boundary=boundary, u=u[z, y, x].tolist(), p=pn_rep[i].tolist(), abs_p=float(np.linalg.norm(
                    pn_rep[i])), g=gn_rep[i].tolist(),
                g_blocks={k: blocks[k]["g"][i].tolist() for k in GAUSS_BLOCKS},
                p_blocks={k: blocks[k]["p"][i].tolist() for k in GAUSS_BLOCKS}, channels=ch,
                tissue_fixed=native_input.tissue[:, z, y, x].tolist(), tau2_node=float(state.tau2_nodes[z, y, x]),
                r_a=float(r[z-1, y-1, x-1]) if interior else None, J_a=float(Jnode[z-1, y-1, x-1]) if interior else None,
                M_aa=diags[i][0], Minv_aa=diags[i][1], GN_diag=Dn[i].tolist(), shift=metric.shift,
                adjacent_cells=[dict(cell=c, m_K=float(mKf[c]), cert_bound=float(cbound[c]),
                                     cert_path=HOW[int(chow[c])]) for c in adj]))
        write_json(out/"nodes.json", dict(top_nodes=[list(n) for n in top_nodes], critical_nodes=[list(n) for n in Knodes],
                                          nodes=nodes_out))
        diag["D_GN"] = dict(max=float(D.max()), mean_mask=float(D[D > 0].mean()) if np.any(D > 0) else 0.,
                            q99=float(np.quantile(D, .99)), shift=metric.shift)
        del D, host_cp, identity, weights
        T.mark("base_stage", tick)
        write_json(out/"diag.json", diag)

    # ---- tau scan ----
    if "scan" in stages:
        tick = time.perf_counter()
        taus = [(r_["tau"], r_["min_jd_lower_bound"], r_["cert_status"]) for r_ in lc]
        grid_taus = [(float(t), None, None) for t in np.logspace(-12, 0, 25)]
        scan = []
        for tau, logged, logged_status in taus+grid_taus:
            cand = inject_free(theta+tau*p, grid)
            oc = backend.certify(cand, grid)
            kv = node_field_values(cand, Knodes)
            jk, _ = cell_min_J(kv, grid.spacing)
            row = dict(tau=tau, status=oc.status, reason=oc.cert_reason, min_jd_lower_bound=oc.min_jd_lower_bound,
                       witness_cell=oc.witness_cell, witness_point=oc.witness_reference_point,
                       n_infeasible=oc.n_infeasible, n_uncertain=oc.n_uncertain, Kstar_true_min_J=jk,
                       log_min_jd_lower_bound=logged, log_status=logged_status,
                       rel_vs_log=None if logged is None else abs(oc.min_jd_lower_bound-logged)/abs(logged))
            if oc.witness_cell is not None and oc.witness_cell != Kstar:
                wv = node_field_values(cand, cell_nodes(grid, oc.witness_cell))
                row["witness_true_min_J"] = cell_min_J(wv, grid.spacing)[0]
            scan.append(row)
            info(f"scan tau {tau:.4e}: {oc.status} {oc.cert_reason} LB {oc.min_jd_lower_bound} witness "
                 f"{oc.witness_cell} n_inf {oc.n_infeasible} K* minJ {jk:.4e}"
                 + ("" if logged is None else f" (log {logged})"))
            del cand
        write_json(out/"scan.json", scan)
        T.mark("scan_stage", tick)

    # ---- counterfactual metrics ----
    variants = {}
    cfout = {}
    # DEF (2026-10-31): explicit "gauss27+corners", the cell estimate this diagnostic (runs/20261030-diag-g3) used as
    # the default then; the JDScalingConfig default is "certificate" since DEF (CertJDScaledMetric is its own variant)
    pg = PGConfig(metric_jd_scaling="affine_v1", jd=JDScalingConfig(cell_estimate="gauss27+corners"))
    objective = DirectObjective(backend, state)
    base_cert = backend.certify(u, grid)
    target = record["native"]["T_B"]
    gn_metric = gn_local = None

    def build(name):
        nonlocal gn_metric, gn_local
        if name == "base":
            return JDScaledMetric(metric, grid, u, config=pg.jd, at="diag")
        if name in ("cf1", "cf3", "cf1b"):
            if gn_metric is None:
                D = backend.gn_diagonal(state, ev)
                gn_metric = GNMetric(metric, D, backend.dev, tol=args.pcg_tol, maxit=args.pcg_maxit, log=info)
                gn_local = GNMetric(metric, D, backend.dev, local=True, tol=args.pcg_tol, maxit=args.pcg_maxit,
                                    log=info)
                del D
            if name == "cf1":
                return JDScaledMetric(gn_metric, grid, u, config=pg.jd, at="diag")
            if name == "cf1b":
                return JDScaledMetric(gn_local, grid, u, config=pg.jd, at="diag")
            return CertJDScaledMetric(gn_metric, grid, u, cert_config=cert_config, config=pg.jd, at="diag")
        if name == "cf2":
            return CertJDScaledMetric(metric, grid, u, cert_config=cert_config, config=pg.jd, at="diag")
        raise SystemExit(f"unknown variant {name}")

    if "cf" in stages:
        tick = time.perf_counter()
        for name in ("base", "cf1", "cf1b", "cf2", "cf3"):
            t0 = time.perf_counter()
            pm = build(name)
            pv = -pm.solve(g)
            pvn = np.sqrt((free_view(pv, grid)**2).sum(axis=0))
            Kp = free_at(pv, grid, Knodes)
            dj = float(cell_dJ(Kverts, Kp, grid.spacing, Kpt)[0])
            top_v = [float(pvn[n[0]-1, n[1]-1, n[2]-1]) for n in (top_nodes if "base" in stages else [])]
            rows = []
            res = run_pg(objective, ev, theta=theta, cert=base_cert, metric=pm, certify=backend.certify, target=target,
                         max_steps=1, config=pg, log=rows.append, context=dict(variant=name), gradient=g)
            cands = [r_ for r_ in rows if r_.get("kind") == "candidate"]
            acc = [r_ for r_ in cands if r_.get("accepted")]
            J0 = crit[0]["true_min_J"]
            cfout[name] = dict(
                metric_record=pm.computations[-1] if hasattr(pm, "computations") else None,
                max_abs_p_node=float(pvn.max()), norm_p=float(np.linalg.norm(pv)), slope=float(g@pv),
                domain_bound=sampling_step_bound(grid, u, pv), Kstar_abs_p=np.linalg.norm(Kp, axis=1).tolist(),
                Kstar_dJ_dtau=dj, tau_linear_zero=J0/(-dj) if dj < 0 else None, top_nodes_abs_p=top_v,
                cos_to_base=None if name == "base" else float(pv@variants["base"]/np.linalg.norm(pv)/np.linalg.norm(
                    variants["base"])),
                step=dict(status=res.status, reason=res.reason, accepted=res.accepted, candidates=len(cands),
                          accepted_tau=acc[0]["tau"] if acc else None, energy_after=acc[0]["energy"] if acc else None,
                          ared=acc[0]["ared"] if acc else None, required=acc[0]["required"] if acc else None,
                          F_B_before=ev.energy,
                          rejections=[(r_["tau"], r_["reason"], r_["cert_reason"], r_["min_jd_lower_bound"])
                                      for r_ in cands if not r_.get("accepted")]),
                pcg=None if name in ("base", "cf2") else (gn_local if name == "cf1b" else gn_metric).records[-2:],
                seconds=time.perf_counter()-t0)
            if name == "base":
                variants["base"] = pv
            info(f"CF {name}: max|p| {cfout[name]['max_abs_p_node']:.4e}, K* |p| {cfout[name]['Kstar_abs_p']}, dJ/dtau "
                 f"{dj:.4e}, step {cfout[name]['step']['status']} tau {cfout[name]['step']['accepted_tau']} ared "
                 f"{cfout[name]['step']['ared']} ({len(cands)} candidates)")
            write_json(out/"cf.json", cfout)
            del pv, pvn
        T.mark("cf_stage", tick)

    # ---- short runs ----
    if "short" in stages:
        for name in [s.strip() for s in args.short.split(",") if s.strip()]:
            tick = time.perf_counter()
            pm = build(name)
            path = out/f"short_{name}.jsonl"
            with path.open("w") as fh:
                def log(row, fh=fh):
                    fh.write(json.dumps(row, default=_json_default)+"\n")
                    fh.flush()
                res = run_pg(objective, ev, theta=theta, cert=base_cert, metric=pm, certify=backend.certify,
                             target=target, max_steps=args.short_steps, config=pg, log=log, context=dict(variant=name),
                             gradient=g)
            rows = [json.loads(line) for line in path.open()]
            acc = [r_ for r_ in rows if r_.get("kind") == "accept"]
            cands = [r_ for r_ in rows if r_.get("kind") == "candidate"]
            diag.setdefault("short", {})[name] = dict(
                status=res.status, reason=res.reason, accepted=res.accepted, candidates=len(cands),
                F_B_start=ev.energy, F_B_end=res.energies[-1], GF_B_end=res.gradient_norm,
                taus=[r_["tau"] for r_ in acc], min_jd_lower_bounds=[r_["min_jd_lower_bound"] for r_ in acc],
                rejections={k: sum(1 for r_ in cands if r_.get("reason") == k) for k in set(
                    str(r_.get("reason")) for r_ in cands if not r_.get("accepted"))},
                jd_refreshes=getattr(pm, "refreshes", None), seconds=time.perf_counter()-tick)
            info(f"short {name}: {diag['short'][name]}")
            write_json(out/"diag.json", diag)

    diag["timing"] = T.items
    diag["peak_rss_mb"] = peak_rss_mb()
    write_json(out/"diag.json", diag)
    info(f"done; peak RSS {diag['peak_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
