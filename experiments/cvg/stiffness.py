"""CVG stiffness analysis at given states of one level (no solver run; diagnostic only).

    CVG_CODE=<snapshot> python -B experiments/cvg/stiffness.py --run <l0_run dir> --state <npz | zero> --label NAME
        --out DIR [--level 0] [--variants V0,V1,V2,V3,V4] [--per-block] [--tol 1e-3]

The model state is rebuilt exactly as the run had it on that level (cache, channel source, eta, logvol options from
the run's config.json; sigma_c^2 = the round's sigma2_used and the frozen metric's c_dat and b_P from result.json),
so the metric M is the run's round-start metric (C09.1, frozen for the whole L0 solve).  At the state u:

1. F_B blocks; gradient blocks: every channel c alone (K_c^T r_c), diffusion, log-volume, curl; norms in the GF_B
   normalization (/ sqrt(omega_ell/omega0)) and shares of ||g||^2; where the total gradient sits (nodes with JD factor
   r_a < 1, < .1, < .01; nodes whose data GN diagonal exceeds 10x / 100x its brain mean).
2. Generalized GN spectrum lambda(P H) for preconditioners P (H = GN Hessian of C06-5 at u, all blocks; P H is
   similar to the symmetric C^T H C with P = C C^T), extreme eigenvalues by Lanczos (scipy eigsh, which LA / SA):
   V0  P = M^-1                            the C09.1 metric of the run (no JD scaling)
   V1  P = S~ M^-1 S~                      the PG operator at u (S~ recomputed at u, certificate cell estimate)
   V2  P = B_full^-1                       block-Jacobi of the full GN (3x3 node blocks, exact by 8-colour probing)
   V3  P = (M - shift I + B_data)^-1       the metric with the data GN 3x3 node blocks in place of c_dat I
   V4  P = (M - shift I + B_full)^-1       the metric plus every block's 3x3 node diagonal of the GN
   V5  P = D M^-1 D, D = diag((1 + d_a/shift)^-1/2), d_a = trace(B_data,a)/3: a data-curvature affine scaling of the
       DST metric (as cheap as the JD scaling: a diagonal on both sides of M^-1)
   V6  P = S~ D M^-1 D S~                  V5 with the JD scaling on top
   V7  P = D' M'^-1 D'                     M' the C09.1 metric with c_dat replaced by the brain-median per-node data
       diagonal (a' = median_a d_a / (omega_ell/omega0)) and D' = diag((1 + d_a/shift')^-1/2): the same cheap
       diagonal-plus-DST form, with the scalar shift no longer set by the mean of a localized curvature
   (V3, V4 applied by CG preconditioned with M^-1, rtol 1e-11.)  lambda_max: Lanczos (eigsh LA); lambda_min: LOBPCG from
   the lowest DST modes (Ritz values are upper bounds: an unconverged lambda_min overestimates, kappa underestimates).  For V0 and V1: Rayleigh quotients of every block
   (each channel, diffusion, log-volume, curl) on the top and bottom Ritz vectors, and the share of the top vector's
   squared norm (in the u-coordinates C w) on the node classes of item 1.
3. --per-block: lambda_max(M^-1 H_b) of every block b alone (V0 metric).
Nothing here changes a solver path; the variants V2-V4 are candidate metrics evaluated as operators only.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
from scipy.fft import dstn, idstn  # noqa: E402
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, cg, eigsh  # noqa: E402

from bayesreg.io import load_pair  # noqa: E402
from bayesreg.cli import run_pair as rp  # noqa: E402
from bayesreg.geom import finish, node_field  # noqa: E402
from bayesreg.model import build_metric, evaluate, gradient, hvp, metric_bP  # noqa: E402
from bayesreg.solve.jd_scaling import CERTIFICATE, free_scaling, scaling_factors  # noqa: E402
from bayesreg.solve.multilevel import _template  # noqa: E402  (read-only use: the level's ModelState as the run built it)
from bayesreg.solve.stopping import gf_b, state_hash  # noqa: E402

PRIORS = ("diffusion", "logvol", "curl")
LEVEL_INPUT = {}


def setup(run, level):
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
    LEVEL_INPUT["moving"] = inputs[level].moving                       # for the channel support record only
    return config, result, row, state, metric


def channel_gradient(state, point, c):
    out = np.zeros((3,)+state.grid.shape)
    flat = out.reshape(3, -1)
    flat[:, state.node_index] += ((state.beta[c]*point.channel.r[c])[:, None]*point.channel.grad[c]).T
    return finish(out)


def channel_hvp(state, point, c, v):
    v = node_field(v, state.grid, name="v").reshape(-1, 3)[state.node_index]
    g = point.channel.grad[c]
    kv = state.beta[c]*(g[:, 0]*v[:, 0]+g[:, 1]*v[:, 1]+g[:, 2]*v[:, 2])
    out = np.zeros((3,)+state.grid.shape)
    out.reshape(3, -1)[:, state.node_index] += ((state.beta[c]*kv)[:, None]*g).T
    return finish(out)


def data_blocks(state, point):
    """(N_int, 3, 3) data GN node blocks sum_c beta_c^2 grad G_c grad G_c^T on the interior nodes (0 off the mask)."""
    grid = state.grid
    full = np.zeros((int(np.prod(grid.shape)), 3, 3))
    g = point.channel.grad
    b2 = state.beta**2
    full[state.node_index] = np.einsum("cn,cni,cnj->nij", b2, g, g)
    return full.reshape(grid.shape+(3, 3))[1:-1, 1:-1, 1:-1].reshape(-1, 3, 3)


def probed_blocks(H, grid):
    """(N_int, 3, 3) node blocks of H (free-vector operator) by 8-colour probing: two interior nodes of one colour
    (i, j, k mod 2) never share a cell, and every block of F_B couples only nodes of a common cell."""
    shape = grid.interior_shape
    n = int(np.prod(shape))
    idx = np.indices(shape)
    blocks = np.zeros((n, 3, 3))
    for colour in range(8):
        sel = ((idx[0] % 2) == (colour >> 2) & 1) & ((idx[1] % 2) == (colour >> 1) & 1) & ((idx[2] % 2) == colour & 1)
        sel = sel.ravel()
        for d in range(3):
            e = np.zeros((3, n))
            e[d, sel] = 1.
            y = H(e.ravel()).reshape(3, n)
            blocks[sel, :, d] = y[:, sel].T
    return blocks


def block_apply(blocks, v):
    n = blocks.shape[0]
    x = v.reshape(3, n).T
    return np.einsum("nij,nj->ni", blocks, x).T.ravel()


def block_power(blocks, power):
    w, q = np.linalg.eigh(blocks)
    if np.any(w <= 0):
        raise FloatingPointError(f"block not positive definite: min eig {w.min()}")
    return np.einsum("nij,nj,nkj->nik", q, w**power, q)


class MetricRoot:
    """M^-1/2 of the C09.1 metric by DST-I: M^-1 = omega0 idst(dst(v)/den), den = the metric's denominator."""

    def __init__(self, metric, grid):
        self.metric, self.grid = metric, grid
        self.factor = np.sqrt(metric.omega0/metric.inverse.denominator)

    def apply(self, v):
        f = v.reshape((3,)+self.grid.interior_shape)
        return idstn(dstn(f, type=1, norm="ortho", axes=(-3, -2, -1))*self.factor, type=1, norm="ortho",
                     axes=(-3, -2, -1)).ravel()


def smooth_start(grid, extra_random=2, seed=0):
    """Initial block for the smallest eigenvalues: the lowest DST-I modes (1,1,1), (2,1,1), (1,2,1), (1,1,2) of each
    component (orthonormal sine products on the interior nodes) plus a few random vectors."""
    nz, ny, nx = grid.interior_shape
    n_int = nz*ny*nx

    def mode(kz, ky, kx):
        z = np.sin(np.pi*kz*np.arange(1, nz+1)/(nz+1))
        y = np.sin(np.pi*ky*np.arange(1, ny+1)/(ny+1))
        x = np.sin(np.pi*kx*np.arange(1, nx+1)/(nx+1))
        m = np.einsum("i,j,k->ijk", z, y, x).ravel()
        return m/np.linalg.norm(m)
    cols = []
    for kk in ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)):
        m = mode(*kk)
        for d in range(3):
            v = np.zeros(3*n_int)
            v[d*n_int:(d+1)*n_int] = m
            cols.append(v)
    rng = np.random.default_rng(seed)
    for _ in range(extra_random):
        cols.append(rng.standard_normal(3*n_int))
    return np.array(cols).T


def smallest(op, n, X0, maxiter, tol, B=None):
    """Smallest eigenvalues by LOBPCG (scipy) from the block X0 (largest=False).  The Ritz values are upper bounds on
    the smallest eigenvalues (min-max), so a non-converged lambda_min is an upper estimate and kappa a lower estimate."""
    import warnings
    from scipy.sparse.linalg import lobpcg
    tick = time.perf_counter()
    count = [0]

    def mm(X):
        X = np.asarray(X)
        if X.ndim == 1:
            count[0] += 1
            return op(X)
        count[0] += X.shape[1]
        return np.column_stack([op(X[:, j]) for j in range(X.shape[1])])
    A = LinearOperator((n, n), matvec=lambda x: op(np.asarray(x).ravel()), matmat=mm, dtype=np.float64)
    Bl = None if B is None else LinearOperator((n, n), matvec=lambda x: B(np.asarray(x).ravel()),
                                               matmat=lambda X: np.column_stack([B(np.asarray(X)[:, j]) for j in
                                                                                 range(np.asarray(X).shape[1])]),
                                               dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vals, vecs, history = lobpcg(A, X0, B=Bl, largest=False, maxiter=maxiter, tol=tol, retLambdaHistory=True)
    order = np.argsort(vals)
    converged = not any("not reach" in str(w.message) or "convergence" in str(w.message).lower() for w in caught)
    return dict(values=[float(v) for v in np.asarray(vals)[order]], converged=converged, matvecs=count[0],
                seconds=time.perf_counter()-tick, method="lobpcg (smooth DST start)", maxiter=maxiter,
                first_iteration_values=[float(v) for v in np.sort(np.asarray(history[0]))[:3]] if history else None,
                warnings=[str(w.message)[:200] for w in caught][:3]), vecs[:, order]


def lanczos(op, n, which, k, tol, maxiter, ncv):
    tick = time.perf_counter()
    count = [0]

    def mv(x):
        count[0] += 1
        return op(np.asarray(x).ravel())
    A = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    try:
        vals, vecs = eigsh(A, k=k, which=which, tol=tol, maxiter=maxiter, ncv=ncv)
        converged = True
    except ArpackNoConvergence as exc:
        vals, vecs, converged = exc.eigenvalues, exc.eigenvectors, False
    order = np.argsort(vals)
    return dict(values=[float(v) for v in vals[order]], converged=converged, matvecs=count[0],
                seconds=time.perf_counter()-tick), (vecs[:, order] if vecs is not None and len(vals) else None)


def generalized(Hop, Bop, Binv, n, which, k, tol, maxiter, ncv):
    tick = time.perf_counter()
    count = [0]

    def mv(x):
        count[0] += 1
        return Hop(np.asarray(x).ravel())
    A = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    B = LinearOperator((n, n), matvec=lambda x: Bop(np.asarray(x).ravel()), dtype=np.float64)
    Bi = LinearOperator((n, n), matvec=lambda x: Binv(np.asarray(x).ravel()), dtype=np.float64)
    try:
        vals, vecs = eigsh(A, k=k, M=B, Minv=Bi, which=which, tol=tol, maxiter=maxiter, ncv=ncv)
        converged = True
    except ArpackNoConvergence as exc:
        vals, vecs, converged = exc.eigenvalues, exc.eigenvectors, False
    order = np.argsort(vals)
    return dict(values=[float(v) for v in vals[order]], converged=converged, matvecs=count[0],
                seconds=time.perf_counter()-tick), (vecs[:, order] if vecs is not None and len(vals) else None)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--state", required=True, help="npz with u (states/...) or 'zero'")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--variants", default="V0,V1,V2,V3,V4")
    p.add_argument("--per-block", action="store_true")
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--ncv", type=int, default=48)
    p.add_argument("--maxiter", type=int, default=4000)
    p.add_argument("--tol-small", type=float, default=1e-2, help="eigsh tol of the SA (smallest) runs")
    p.add_argument("--lobpcg-maxiter", type=int, default=60, help="LOBPCG iterations of the SA runs")
    p.add_argument("--maxiter-small", type=int, default=25, help="eigsh restarts of the SA runs (best Ritz values kept "
                   "when not converged: then lambda_min is an upper estimate and kappa a lower estimate)")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    tick0 = time.perf_counter()
    run = Path(a.run)
    config, result, row, state, metric = setup(run, a.level)
    grid = state.grid
    n_int = grid.n_free_scalar
    n = 3*n_int
    norm_factor = np.sqrt(grid.omega/state.omega0)
    if a.state == "zero":
        u = np.zeros(grid.shape+(3,))
    else:
        u = np.load(a.state)["u"]
    point = evaluate(state, u)
    if not hasattr(point, "energy"):
        raise RuntimeError(f"F_B not defined at the state: {point.reason}")
    backend = common.CaptureBackend()
    cert = backend.certify(u, grid)
    doc = dict(label=a.label, run=str(run), state=a.state, state_hash=state_hash(u), level=a.level,
               grid_shape=list(grid.shape), n_free=n, channels=int(state.n_channels),
               sigma2=np.asarray(state.sigma2).tolist(), eta=dict(alpha=state.alpha, nu=state.nu,
                                                                  tau2=np.asarray(state.tau2).tolist()),
               metric=dict(c_dat=metric.a, b_P=metric.b, shift=metric.shift,
                           M_spectrum=dict(min=float(metric.inverse.denominator.min()/state.omega0),
                                           max=float(metric.inverse.denominator.max()/state.omega0))),
               certificate=dict(passed=cert.passed, min_jd_lower_bound=cert.min_jd_lower_bound,
                                state_hash=cert.state_hash),
               F_B=point.energy, energies=point.energies, code=common.code_record())
    # ---- 1. gradient blocks -------------------------------------------------------------------------------
    g_total = gradient(state, point)
    parts = {f"channel_{c}": channel_gradient(state, point, c) for c in range(state.n_channels)}
    for b in PRIORS:
        parts[b] = gradient(state, point, blocks=(b,))
    total_sq = float(g_total@g_total)
    data = sum(parts[f"channel_{c}"] for c in range(state.n_channels))
    doc["gradient"] = dict(
        GF_B=gf_b(g_total, grid.omega, state.omega0),
        GF_B_data=float(np.linalg.norm(data)/norm_factor),
        GF_B_prior=float(np.linalg.norm(sum(parts[b] for b in PRIORS))/norm_factor),
        blocks={k: dict(GF=float(np.linalg.norm(v)/norm_factor), share_of_total_sq=float(v@v)/total_sq,
                        cos_with_total=float(v@g_total)/max(np.linalg.norm(v)*np.sqrt(total_sq), 1e-300))
                for k, v in parts.items()},
        sum_check=float(np.linalg.norm(sum(parts.values())-g_total)/np.sqrt(total_sq)))
    # node classes
    r, jd_record = scaling_factors(u, grid, estimate=CERTIFICATE, cert=cert, cert_config=backend.cert_config)
    r = r.ravel()
    Bdata = data_blocks(state, point)
    ddiag = np.einsum("nii->n", Bdata)
    on_mask = ddiag > 0
    mean_d = float(ddiag[on_mask].mean())
    node_g2 = (g_total.reshape(3, n_int)**2).sum(axis=0)
    classes = {"r_lt_1": r < 1, "r_lt_0.1": r < .1, "r_lt_0.01": r < .01,
               "data_diag_gt_10x_mean": ddiag > 10*mean_d, "data_diag_gt_100x_mean": ddiag > 100*mean_d,
               "off_mask": ~on_mask}
    doc["jd_scaling_at_state"] = jd_record
    doc["data_diagonal"] = dict(mean_on_mask=mean_d, shift=metric.shift,
                                percentiles_over_mean={str(q): float(np.percentile(ddiag[on_mask], q)/mean_d)
                                                       for q in (50, 90, 99, 99.9, 100)},
                                fraction_gt_10x=float(np.mean(ddiag[on_mask] > 10*mean_d)),
                                fraction_gt_100x=float(np.mean(ddiag[on_mask] > 100*mean_d)))
    # per-channel support on this level (C11 estimation domain question): brain nodes where the channel is not at its
    # floor on either side; sigma_c^2 is a mean over all N_q brain nodes, so sigma^2 / support is the mean square over
    # the support
    support = {}
    eps_floor = 1e-3
    for c in range(state.n_channels):
        fixed_c = state.fixed[c][state.mask]
        moving_c = LEVEL_INPUT["moving"][c][state.mask]
        thr = np.sqrt(.05) if c > 0 else None
        if c == 0:
            continue
        on = (fixed_c > thr) | (moving_c > thr)
        support[f"channel_{c}"] = dict(fraction_of_brain_nodes=float(on.mean()), sigma2=float(state.sigma2[c]),
                                       sigma2_over_support=float(state.sigma2[c]/max(on.mean(), 1e-12)))
    doc["channel_support"] = dict(rule="s > sqrt(.05) (P~ > .05) at a brain node of the level, fixed or moving side",
                                  channels=support)
    doc["gradient_location"] = {k: dict(nodes=int(m.sum()), share_g2=float(node_g2[m].sum()/node_g2.sum()))
                                for k, m in classes.items()}
    common.write_json(out/"stiffness.json", doc)
    # ---- 2. spectra ---------------------------------------------------------------------------------------
    H = lambda v: hvp(state, point, v)                                   # noqa: E731
    Hb = {f"channel_{c}": (lambda v, c=c: channel_hvp(state, point, c, v)) for c in range(state.n_channels)}
    for b in PRIORS:
        Hb[b] = (lambda v, b=b: hvp(state, point, v, blocks=(b,)))
    root = MetricRoot(metric, grid)
    check = np.random.default_rng(0).standard_normal(n)
    doc["metric_root_check"] = float(np.linalg.norm(root.apply(root.apply(check))-metric.solve(check))
                                     / np.linalg.norm(metric.solve(check)))
    S = free_scaling(r)
    dnode = ddiag/3                                                     # per-component mean data GN diagonal
    Dscale = np.tile((1+dnode/metric.shift)**-.5, 3)
    shift_low = float(np.median(dnode[on_mask]))
    metric_low = build_metric(state, c_dat=shift_low*state.omega0/grid.omega, b_P=metric.b)
    root_low = MetricRoot(metric_low, grid)
    Dlow = np.tile((1+dnode/shift_low)**-.5, 3)
    doc["V7_metric"] = dict(shift=metric_low.shift, c_dat=metric_low.a, b_P=metric_low.b,
                            D_min=float(Dlow.min()), D_median=float(np.median(Dlow[:n_int][on_mask])))
    doc["V5_scaling"] = dict(min=float(Dscale.min()), median=float(np.median(Dscale)),
                             fraction_lt_half=float(np.mean(Dscale[:n_int] < .5)))
    variants = [v for v in a.variants.split(",") if v]
    spectra = {}

    def rayleigh(C, w):
        z = C(w)
        quot = {k: float(z@f(z)) for k, f in Hb.items()}
        zz = (z.reshape(3, n_int)**2).sum(axis=0)
        where = {k: float(zz[m].sum()/zz.sum()) for k, m in classes.items()}
        return dict(blocks=quot, total=float(z@H(z)), location=where)

    # ---- 3 (run first). per-block lambda_max under V0 -------------------------------------------------------------------
    if a.per_block:
        per = {}
        for k, f in Hb.items():
            op = lambda x, f=f: root.apply(f(root.apply(x)))              # noqa: E731
            rec, _ = lanczos(op, n, "LA", 1, 1e-2, a.maxiter, 24)
            per[k] = rec
            print("per-block", k, rec["values"], flush=True)
        doc["per_block_lambda_max_V0"] = per
    common.write_json(out/"stiffness.json", doc)
    X0 = smooth_start(grid)
    # Rayleigh quotients of node-class indicator vectors (upper bounds on lambda_min of the PG operator V1 = S~ M^-1 S~
    # and of V0): z = 1 on every component of the nodes of the class; lambda_R = z.Hz / z.P^-1 z with P^-1 = S~^-1 M S~^-1
    # (V1) or M (V0).  The Lanczos/LOBPCG runs start from smooth modes and need not see these directions.
    frozen = {}
    for name in ("r_lt_1", "r_lt_0.1", "r_lt_0.01"):
        m = classes[name]
        if not m.any():
            continue
        z = np.tile(m.astype(float), 3)
        zHz = float(z@H(z))
        frozen[name] = dict(nodes=int(m.sum()), V0=zHz/float(z@metric.matvec(z)),
                            V1=zHz/float((z/S)@metric.matvec(z/S)))
    doc["indicator_rayleigh"] = frozen
    # Rayleigh quotients of two smooth test fields (upper bounds on lambda_min, usually tighter than the unconverged
    # LOBPCG values): the node gradient of the lowest sine mode phi (nearly curl-free) and (d_y psi, -d_x psi, 0) with
    # psi the same mode (nearly divergence-free), restricted to the interior nodes (faces zero, U0).
    zz_, yy_, xx_ = [np.linspace(0, 1, n_) for n_ in grid.shape]
    phi = np.einsum("i,j,k->ijk", np.sin(np.pi*zz_), np.sin(np.pi*yy_), np.sin(np.pi*xx_))
    gz, gy, gx = np.gradient(phi)
    tests = {"curl_free_lowest": (gx, gy, gz), "div_free_lowest": (gy, -gx, np.zeros_like(gx))}
    smooth = {}
    for name, (fx, fy, fz) in tests.items():
        z = np.concatenate([f[1:-1, 1:-1, 1:-1].ravel() for f in (fx, fy, fz)])
        z = z/np.linalg.norm(z)
        zHz = float(z@H(z))
        smooth[name] = dict(V0=zHz/float(z@metric.matvec(z)), V1=zHz/float((z/S)@metric.matvec(z/S)),
                            V5=zHz/float((z/Dscale)@metric.matvec(z/Dscale)),
                            V7=zHz/float((z/Dlow)@metric_low.matvec(z/Dlow)))
    doc["smooth_rayleigh"] = smooth
    common.write_json(out/"stiffness.json", doc)
    common.write_json(out/"stiffness.json", doc)
    Bfull = None
    if any(v in variants for v in ("V2", "V4")):
        tick = time.perf_counter()
        Bfull = probed_blocks(H, grid)
        doc["probe_seconds"] = time.perf_counter()-tick
    for v in variants:
        if v in ("V0", "V1", "V2", "V5", "V6", "V7"):
            if v == "V0":
                C = root.apply
            elif v == "V7":
                C = lambda x: Dlow*root_low.apply(x)                     # noqa: E731
            elif v in ("V1", "V5", "V6"):
                left = {"V1": S, "V5": Dscale, "V6": S*Dscale}[v]
                C = lambda x, left=left: left*root.apply(x)              # noqa: E731
            else:
                Bm12 = block_power(Bfull, -.5)
                C = lambda x: block_apply(Bm12, x)                       # noqa: E731
            if v in ("V1", "V5", "V6"):
                CT = lambda x, left=left: root.apply(left*x)              # noqa: E731  C^T (M^-1/2 symmetric)
            elif v == "V7":
                CT = lambda x: root_low.apply(Dlow*x)                     # noqa: E731
            else:
                CT = C                                                    # M^-1/2 and B^-1/2 symmetric
            op = lambda x: CT(H(C(x)))                                    # noqa: E731
            entry = {}
            for which in ("LA", "SA"):
                if which == "LA":
                    rec, vecs = lanczos(op, n, which, a.k, a.tol, a.maxiter, a.ncv)
                else:
                    rec, vecs = smallest(op, n, X0, a.lobpcg_maxiter, 1e-7)
                if vecs is not None and len(rec["values"]) and v in ("V0", "V1", "V5", "V6", "V7"):
                    rec["rayleigh_extreme"] = rayleigh(C, vecs[:, -1] if which == "LA" else vecs[:, 0])
                entry[which] = rec
                print(v, which, rec["values"], rec["converged"], rec["matvecs"], f"{rec['seconds']:.1f}s", flush=True)
        else:
            extra = Bdata if v == "V3" else Bfull
            shift = metric.shift
            Bop = lambda x: metric.matvec(x)-shift*x+block_apply(extra, x)       # noqa: E731
            iters = []

            def Binv(x, Bop=Bop):
                A = LinearOperator((n, n), matvec=Bop, dtype=np.float64)
                P = LinearOperator((n, n), matvec=metric.solve, dtype=np.float64)
                k = [0]
                sol, info = cg(A, x, rtol=1e-11, atol=0., maxiter=5000, M=P, callback=lambda _: k.__setitem__(0, k[0]+1))
                if info != 0:
                    raise FloatingPointError(f"CG info {info}")
                iters.append(k[0])
                return sol
            entry = {}
            for which in ("LA", "SA"):
                if which == "LA":
                    rec, _ = generalized(H, Bop, Binv, n, which, a.k, a.tol, a.maxiter, a.ncv)
                else:
                    rec, _ = smallest(H, n, X0, a.lobpcg_maxiter, 1e-7, B=Bop)
                rec["cg_iterations_mean"] = float(np.mean(iters)) if iters else None
                entry[which] = rec
                iters.clear()
                print(v, which, rec["values"], rec["converged"], rec["matvecs"], f"{rec['seconds']:.1f}s", flush=True)
        lmax = entry["LA"]["values"][-1] if entry["LA"]["values"] else None
        lmin = entry["SA"]["values"][0] if entry["SA"]["values"] else None
        entry["kappa"] = lmax/lmin if lmax is not None and lmin is not None and lmin > 0 else None
        spectra[v] = entry
        doc["spectra"] = spectra
        common.write_json(out/"stiffness.json", doc)
    doc["seconds"] = time.perf_counter()-tick0
    common.write_json(out/"stiffness.json", doc)
    print(f"done {doc['seconds']:.1f} s")


if __name__ == "__main__":
    main()
