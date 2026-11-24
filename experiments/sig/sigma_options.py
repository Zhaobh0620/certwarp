"""SIG: sigma_c^2 estimation options on the CVG L0 caches (analysis only; no solver, no certificate, no GPU).

    nice -n 19 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
        python -B experiments/sig/sigma_options.py [--out /home/rmcfz/seq_vp_tag_results/sig_v1] [--no-lanczos]

Inputs (read only): the CVG caches /home/rmcfz/seq_vp_tag_results/cvg_v1/cache/{0438-0439,0012-0013}.npz (L0 arrays,
already through the label firewall: fixed / moving intensity, structure and tissue simplices, mask) and accepted L0
states of the CVG runs (states/u_final_L0.npz).  The pair's grid comes from bayesreg.io.load_pair (keys fixed / moving
only; development allowlist).  Imports come from the CVG code snapshot be844cd (the code the caches and runs used).

For every (pair, state u) the channel residuals rho_c(u)_a and gradients grad G_c(x_a + u_a) on the L0 mask nodes are
taken from bayesreg.model.channel_point (sigma = 1, w = 1), node weights pi_a = q_a m_a / omega0.  Then, per option
(see OPTIONS below and docs/bayesreg/notes/sigma2_选项_20261031.md), sigma_c^2 and a frozen per-node channel weight
v_c(a) (1 unless the option says otherwise) are formed and these numbers computed with numpy:
  sigma2, ratio = sigma_I^2 / sigma_c^2 (weight relative to the intensity channel),
  c_dat = (1/(3 n)) sum_a sum_c v w |grad G_c|^2 / sigma_c^2 (C09.1, with v w in place of w),
  data gradient norm ||sum_c pi v w rho grad G / sigma^2|| / sqrt(omega_l/omega0) (= G0 of C13-2 at u = 0),
  per-channel shares of the data GN trace and of the data gradient^2, data energy per channel,
  node data-diagonal localization (max / mean, p99 / mean, fraction > 10 x mean).
At u = 0 only (with --lanczos, the default): lambda_max(M^-1 H) by Lanczos for the option's ModelState (sigma2, weights
= v w), H the full GN (data + priors, eta_tmp), M the C09.1 metric with the option's c_dat and b_P principal; and the
rigorous upper bound kappa(V0, u = 0) <= lambda_max (c_dat + b_P Lambda_min) / ((alpha + 1) Lambda_min) of the CVG
review (valid while 1/tau^2 >= 1 everywhere and nu >= 1; data GN is PSD for every option).
Nothing is written into bayesreg/, jd_curl/ or the CVG results.  Output: <out>/sigma_options.json and .md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

SNAPSHOT = Path("/home/rmcfz/seq_vp_tag_results/cvg_v1/code/be844cdf757fa62e74449c2879302f076f336b0c")
sys.path.insert(0, str(SNAPSHOT))

import numpy as np  # noqa: E402
from scipy.fft import dstn, idstn  # noqa: E402
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh  # noqa: E402

from jd_curl.validation import readonly_copy  # noqa: E402
from bayesreg.cli import run_pair as rp  # noqa: E402
from bayesreg.grids import four_level_grids  # noqa: E402
from bayesreg.io import load_pair  # noqa: E402
from bayesreg.model import build_metric, c_dat as model_c_dat, channel_point, evaluate, hvp, metric_bP  # noqa: E402
from bayesreg.solve import Eta, LevelInput  # noqa: E402
from bayesreg.solve.multilevel import _template  # noqa: E402
from bayesreg.solve.stopping import state_hash  # noqa: E402

CVG = Path("/home/rmcfz/seq_vp_tag_results/cvg_v1")
DATA_ROOT = Path("/home/rmcfz/code/python_projects/seq_vp_tag/data/three_d")
PAIRS = {
    "0438-0439": dict(cache=CVG/"cache/0438-0439.npz", reference_run="R1-struct", other="0012-0013",
                      states={"R1_final": "R1-struct", "R5_final": "R5-struct-nu874", "R4_final": "R4-struct-sigmaI"}),
    "0012-0013": dict(cache=CVG/"cache/0012-0013.npz", reference_run="R7-struct-0012", other="0438-0439",
                      states={"R7_final": "R7-struct-0012"}),
}
STATE_NOTES = {
    "u0": "u = 0 (L0 start; the level-start M step uses these residuals)",
    "R1_final": "CVG R1 final L0 state (structure, O0 sigma, nu_tmp; 4716 steps, near fold, budget end)",
    "R5_final": "CVG R5 final L0 state (structure, O0 sigma, nu = 874.2; L0 threshold reached, no near fold)",
    "R4_final": "CVG R4 final L0 state (diagnostic arm S: sigma_c^2 := sigma_I^2; budget end)",
    "R7_final": "CVG R7 final L0 state (T0 0012-0013, structure, O0 sigma, nu_tmp; 1306 steps, budget end)",
}
CLASS_NAMES = None          # filled from the cache (structure layer class_names, channels 1..K_S)
MERGES = {                  # O4 coarser groupings over the 13 structure13_v1 groups (1-based channel numbers)
    "K6": [[1], [2], [3, 4, 5], [6], [7, 8, 9, 10, 13], [11, 12]],
    "K4": [[1], [2], [3, 4, 5], [6, 7, 8, 9, 10, 11, 12, 13]],
}
NU_T = 4.0
LANCZOS = {"O0", "O1-union0-drop@0.05", "O1-fixed-drop@0.05", "O1-fixeddil1-drop@0.05", "O1-union0-plugin@0.05",
           "O1-twovar@0.05", "O1-soft-max", "O2-prior-sigI-k1", "O2-prior-sigI-k0.1", "O2-prior-popO0-k10",
           "O2-floor-0.3sigI", "O2-floor-1sigI", "O3-t-round1", "O3-t-fixedpoint", "O5-shared", "O5-shared-support",
           "O5-hellinger1", "O5-tied-lambda1"}
EPS = 1e-3
_RAISE = dict(over="raise", invalid="raise", divide="raise")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def level_input(cache, grid, merge=None):
    fixed, moving = cache["fixed_L0"], cache["moving_L0"]
    pf, pm = cache["structure_fixed_L0"], cache["structure_moving_L0"]
    if merge is not None:                       # sum of the floored P~ of the merged groups (O(eps) from re-flooring)
        pf = np.stack([pf[0]]+[pf[g].sum(axis=0) for g in merge])
        pm = np.stack([pm[0]]+[pm[g].sum(axis=0) for g in merge])
    f = readonly_copy(np.concatenate([fixed[None], np.sqrt(pf[1:])]))
    m = readonly_copy(np.concatenate([moving[None], np.sqrt(pm[1:])]))
    mask = cache["mask_L0"].astype(bool)
    mask.flags.writeable = False
    return LevelInput(0, grid, f, m, mask, readonly_copy(cache["tissue_fixed_L0"])), pf, pm


class MetricRoot:
    def __init__(self, metric, grid):
        self.grid = grid
        self.factor = np.sqrt(metric.omega0/metric.inverse.denominator)

    def apply(self, v):
        f = v.reshape((3,)+self.grid.interior_shape)
        return idstn(dstn(f, type=1, norm="ortho", axes=(-3, -2, -1))*self.factor, type=1, norm="ortho",
                     axes=(-3, -2, -1)).ravel()


def lanczos_max(state, ev, metric, grid, tol=1e-6):
    root = MetricRoot(metric, grid)
    n = 3*int(np.prod(grid.interior_shape))
    count = [0]

    def mv(x):
        count[0] += 1
        y = root.apply(np.asarray(x).ravel())
        return root.apply(hvp(state, ev, y))
    op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    tick = time.perf_counter()
    try:
        vals = eigsh(op, k=1, which="LA", tol=tol, maxiter=3000, ncv=24, return_eigenvectors=False)
        converged = True
    except ArpackNoConvergence as exc:
        vals, converged = exc.eigenvalues, False
    return dict(lambda_max=float(np.max(vals)) if len(vals) else None, converged=converged, matvecs=count[0],
                seconds=round(time.perf_counter()-tick, 2))


# ---------------------------------------------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------------------------------------------

def m_step(pi, rho2, v=None, extra_num=0., extra_den=0.):
    """(sum pi v rho^2 + extra_num) / (sum pi v + 2 + extra_den) per channel (C11-2 with a = b = 0 when v = 1)."""
    if v is None:
        num = rho2@pi
        den = np.full(rho2.shape[0], pi.sum())
    else:
        num = (v*rho2)@pi
        den = v@pi
    out = (num+extra_num)/(den+2.+extra_den)
    return np.where(den > 0, out, np.nan), den          # empty support: the channel has no likelihood (NaN sigma)


def support(s_f, s_m, theta):
    thr = np.sqrt(theta)
    return (s_f > thr) | (s_m > thr)


def dilate_nodes(on, mask_index, grid_shape, steps=1):
    """6-neighbour dilation by `steps` nodes on the level grid, restricted back to the mask nodes."""
    from scipy.ndimage import binary_dilation
    out = np.empty_like(on)
    for c in range(on.shape[0]):
        full = np.zeros(int(np.prod(grid_shape)), dtype=bool)
        full[mask_index] = on[c]
        full = binary_dilation(full.reshape(grid_shape), iterations=steps).ravel()
        out[c] = full[mask_index]
    return out


def student_t(pi, rho2, sigma2_start, iterations=2000, tol=1e-10):
    """C04-3/C11-2 at fixed u: E step w = (nu+1)/(nu + rho^2/sigma^2), M step sigma^2 = sum pi w rho^2/(N_q + 2).
    Returns (round-1 weights with the level-start sigma, fixed point (sigma2, w), iterations used)."""
    n_q = pi.sum()
    w1 = (NU_T+1)/(NU_T+rho2/sigma2_start[:, None])
    s = sigma2_start.copy()
    history = {}
    converged = False
    for it in range(iterations):
        w = (NU_T+1)/(NU_T+rho2/s[:, None])
        new = ((w*rho2)@pi)/(n_q+2.)
        if it+1 in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000):
            history[it+1] = new.tolist()
        if np.max(np.abs(new-s)/s) < tol:
            s = new
            converged = True
            break
        s = new
    w = (NU_T+1)/(NU_T+rho2/s[:, None])
    return w1, s, w, dict(iterations=it+1, converged=converged, history=history)


def options_for(pi, rho, s_f, s_m0, s_m, other_sigma, mask_index, grid_shape):
    """Every option's (sigma2 (C,), v (C, n) or None, w (C, n) or None, note).  Channel 0 is the intensity channel and
    keeps the O0 value in every option (dense channel)."""
    rho2 = rho*rho
    C, n = rho.shape
    n_q = float(pi.sum())
    out = {}
    s0, _ = m_step(pi, rho2)
    out["O0"] = dict(sigma2=s0, note="C11-2 global MLE over the brain (current contract)")
    sig_i = s0[0]

    # O1: support-domain variance; supports from P~ > theta on the fixed side and/or the moving side
    def o1(name, on, mode, note):
        v = on.astype(np.float64)
        v[0] = 1.
        s, den = m_step(pi, rho2, v)
        s[0] = s0[0]
        rec = dict(sigma2=s, note=note, support_fraction=(den/n_q).tolist())
        if mode == "drop":
            rec["v"] = v
        elif mode == "plugin":
            rec["note"] += "; sigma applied to every node (plug-in, not a likelihood: no v in the energy)"
        out[name] = rec
        return s, v

    for theta in (.01, .02, .05, .1, .2):
        on_u0 = support(s_f, s_m0, theta)
        tag = f"{theta:g}"
        o1(f"O1-union0-drop@{tag}", on_u0, "drop",
           f"S_c = P~_F > {tag} or P~_M(at level start) > {tag}, frozen; off-support residuals dropped (v = 0)")
    theta = .05
    on_u0 = support(s_f, s_m0, theta)
    on_f = support(s_f, s_f, theta)
    o1("O1-fixed-drop@0.05", on_f, "drop", "S_c = P~_F > .05 (fixed side only), off-support dropped")
    o1("O1-fixeddil1-drop@0.05", dilate_nodes(on_f, mask_index, grid_shape, 1), "drop",
       "S_c = P~_F > .05 dilated by 1 L0 node (~8 mm), off-support dropped")
    o1("O1-union0-plugin@0.05", on_u0, "plugin", "S_c = union at level start (theta .05)")
    on_cur = support(s_f, s_m, theta)
    o1("O1-unionU-drop@0.05", on_cur, "drop",
       "S_c = union with the moving side warped by the CURRENT u (u-dependent; shown only to measure the drift)")
    # two variances: in and out of the frozen union support
    v_in = on_u0.astype(np.float64)
    v_in[0] = 1.
    s_in, _ = m_step(pi, rho2, v_in)
    s_out, den_out = m_step(pi, rho2, 1.-v_in)
    s_in = np.where(np.arange(C) == 0, s0, s_in)
    w_two = np.where(v_in > 0, 1., (s_in/np.where(s_out > 0, s_out, np.inf))[:, None])   # per-node 1/sigma^2 as w/sigma_in^2
    w_two[0] = 1.
    out["O1-twovar@0.05"] = dict(sigma2=s_in, w=w_two, sigma2_out=s_out,
                                 note="sigma_in^2 on the frozen union support, sigma_out^2 off it (two-variance MLE); "
                                 "per node 1/sigma^2 carried as w / sigma_in^2",
                                 out_ratio=(sig_i/np.where(den_out > 0, s_out, np.nan)).tolist())
    # soft weights
    pf, pm0 = s_f**2, s_m0**2
    for name, v, note in (("O1-soft-max", np.maximum(pf, pm0), "v = max(P~_F, P~_M at level start), frozen"),
                          ("O1-soft-or", 1-(1-pf)*(1-pm0), "v = 1 - (1 - P~_F)(1 - P~_M at level start), frozen")):
        v = v.copy()
        v[0] = 1.
        s, den = m_step(pi, rho2, v)
        s[0] = s0[0]
        out[name] = dict(sigma2=s, v=v, note=note, support_fraction=(den/n_q).tolist())

    # O2: strong conjugate prior n0 = kappa0 N_q around sigma0^2; and a hard floor
    for kappa in (.01, .1, 1., 10.):
        n0 = kappa*n_q
        s = (rho2@pi+n0*sig_i)/(n_q+2.+n0)
        s[0] = s0[0]
        out[f"O2-prior-sigI-k{kappa:g}"] = dict(
            sigma2=s, note=f"InvGamma pseudo-count n0 = {kappa:g} N_q, sigma0^2 = sigma_I^2 of this pair (proxy only)")
    if other_sigma is not None:
        for kappa in (1., 10.):
            n0 = kappa*n_q
            s = (rho2@pi+n0*other_sigma)/(n_q+2.+n0)
            s[0] = s0[0]
            out[f"O2-prior-popO0-k{kappa:g}"] = dict(
                sigma2=s, note=f"n0 = {kappa:g} N_q, sigma0_c^2 = the OTHER pair's O0 u = 0 value (proxy for a "
                "population prior built from O0 sigma estimates)")
    n0 = 50.
    s = (rho2@pi+n0*sig_i)/(n_q+2.+n0)
    s[0] = s0[0]
    out["O2-prior-n0=50"] = dict(sigma2=s, note="n0 = 50 (moment-matching order, C11-5), sigma0^2 = sigma_I^2")
    for phi in (.1, .3, 1.):
        s = np.maximum(s0, phi*sig_i)
        out[f"O2-floor-{phi:g}sigI"] = dict(sigma2=s, note=f"sigma_c^2 = max(O0, {phi:g} sigma_I^2)")

    # O3: Student-t
    w1, st, wfix, iters = student_t(pi, rho2, s0)
    out["O3-t-round1"] = dict(sigma2=s0, w=w1, note="round 1 of L0: E step at the level-start sigma (O0), nu_t = 4, every channel incl. intensity")
    out["O3-t-fixedpoint"] = dict(sigma2=st, w=wfix, note=f"E/M iteration at this u ({iters['iterations']} iterations, "
                                  f"converged {iters['converged']}), nu_t = 4, intensity channel also Student-t",
                                  iterations=iters)

    # O5: shared structure-layer variance / fixed relative weight
    ks = C-1
    s = s0.copy()
    s[1:] = (rho2[1:]@pi).sum()/(ks*n_q+2.)
    out["O5-shared"] = dict(sigma2=s, note="one sigma_S^2 for all structure channels (K_S N_q observations)")
    v = on_u0.astype(np.float64)
    num = sum((v[c]*rho2[c])@pi for c in range(1, C))
    den = sum(v[c]@pi for c in range(1, C))
    s = s0.copy()
    s[1:] = num/(den+2.)
    out["O5-shared-support"] = dict(sigma2=s, v=np.where(np.arange(C)[:, None] == 0, 1., v),
                                    note="one sigma_S^2 pooled over the frozen union supports (theta .05), off-support "
                                    "dropped")
    s = s0.copy()
    s[1:] = (rho2[1:]@pi).sum()/(n_q+2.)
    out["O5-hellinger1"] = dict(sigma2=s, note="one sigma_H^2, count N_q (one Hellinger observation per node; tempered "
                                "likelihood, not a K_S-dim Gaussian)")
    s = s0.copy()
    s[1:] = sig_i
    out["O5-tied-lambda1"] = dict(sigma2=s, note="sigma_c^2 := sigma_I^2 / lambda_S, lambda_S = 1 (= CVG diagnostic arm "
                                  "S)")
    return out


def metrics(pi, rho, grad, sigma2, v=None, w=None, omega_ratio=1.):
    C, n = rho.shape
    vw = np.ones_like(rho) if v is None else np.array(v, dtype=np.float64)
    if w is not None:
        vw = vw*w
    g2 = np.einsum("cni,cni->cn", grad, grad)                        # |grad G_c|^2 at the nodes
    inv = np.where(np.isfinite(sigma2) & (sigma2 > 0), 1/np.where(sigma2 > 0, sigma2, 1.), 0.)   # NaN sigma: no term
    curv = vw*g2*inv[:, None]                                        # per node per channel (no pi)
    c_dat = float(curv.sum()/(3*n))
    coef = pi[None, :]*vw*rho*inv[:, None]
    gfield = np.einsum("cn,cni->ni", coef, grad)
    gnorm = float(np.sqrt(np.sum(gfield**2))/np.sqrt(omega_ratio))
    chan_g2 = [float(np.sum((coef[c][:, None]*grad[c])**2)) for c in range(C)]
    tot_g2 = float(np.sum(gfield**2))
    trace = (curv*pi[None, :]).sum(axis=1)
    d = curv.sum(axis=0)/3
    energy = 0.5*(vw*rho*rho*pi[None, :]).sum(axis=1)*inv
    return dict(c_dat=c_dat, data_grad_norm=gnorm,
                share_trace=(trace/trace.sum()).tolist(),
                share_grad2_alone=[x/tot_g2 for x in chan_g2],
                energy=energy.tolist(),
                diag_max_over_mean=float(d.max()/d.mean()), diag_p99_over_mean=float(np.percentile(d, 99)/d.mean()),
                diag_frac_gt10=float(np.mean(d > 10*d.mean())))


def effective_ratio(sigma2, w, grad, pi, v=None):
    """Weight relative to intensity: curvature-weighted mean of v w / sigma_c^2 over the nodes, / the intensity's."""
    g2 = np.einsum("cni,cni->cn", grad, grad)*pi[None, :]
    vw = np.ones_like(g2) if v is None else np.array(v, dtype=np.float64)
    if w is not None:
        vw = vw*w
    inv = np.where(np.isfinite(sigma2) & (sigma2 > 0), 1/np.where(sigma2 > 0, sigma2, 1.), 0.)
    eff = (vw*g2).sum(axis=1)/g2.sum(axis=1)*inv
    return (eff/eff[0]).tolist(), eff.tolist()


def _ratio(sigma2):
    return [float(sigma2[0]/s) if np.isfinite(s) and s > 0 else None for s in sigma2]


def full_weights(state, v_nodes):
    """(C, n) node values on the mask -> (C, nz, ny, nx) weights array (1 outside the mask; unused there)."""
    C = v_nodes.shape[0]
    out = np.ones((C, int(np.prod(state.grid.shape))))
    out[:, state.node_index] = v_nodes
    return out.reshape((C,)+state.grid.shape)


def run_pair(name, spec, args, sigma_other):
    cache = np.load(spec["cache"], allow_pickle=False)
    meta = json.loads(str(cache["meta_json"]))
    global CLASS_NAMES
    CLASS_NAMES = meta["structure_fixed"]["accepted"]["class_names"]
    pair = load_pair(rp.resolve_pair(DATA_ROOT, meta["pair"]), allowlist=rp.DEFAULT_ALLOWLIST)
    grids = four_level_grids(pair.grid)
    grid = grids[0]
    if grid.grid_id != meta["grid_ids"]["0"]:
        raise RuntimeError("L0 grid id differs from the cache")
    native_omega = pair.grid.omega
    del pair
    ref = json.loads((CVG/"runs"/spec["reference_run"]/"result.json").read_text())
    ref_cfg = json.loads((CVG/"runs"/spec["reference_run"]/"config.json").read_text())
    row = ref["rounds"][0]
    e = meta["eta_tmp"]
    eta = Eta(e["alpha"], e["nu"], tuple(e["tau2"]), source=e["source"])
    solver = ref_cfg["solver"]
    logvol = dict(logvol_closed27_share=solver["logvol_closed27_share"], closed27_weights=solver["closed27_weights"])
    li, pf, pm = level_input(cache, grid)
    template = _template(li, eta, native_omega, solver["gauss_tile_size"], li.fixed.shape[0], 1,
                         solver["logvol_points"], **logvol)
    if not np.all(template.strict):
        raise RuntimeError("a mask node lies on the grid boundary")
    pi = template.beta[0]**2                                          # q_a m_a / omega0 (sigma = 1, w = 1)
    omega_ratio = grid.omega/native_omega
    zero = np.zeros(grid.shape+(3,))
    p0 = channel_point(template, zero)
    s_f = np.asarray(template.fixed_masked)
    s_m0 = np.asarray(p0.rho)+s_f
    # Lambda_min of K_Q/omega: metric with a = 0 has den = omega_l b Lambda
    bP = metric_bP(template)
    m0 = build_metric(template, c_dat=0., b_P=bP)
    lam_min = float(np.min(m0.inverse.denominator)/(grid.omega*bP))
    doc = dict(pair=meta["pair"], cache=str(spec["cache"]), cache_sha256=sha256_file(spec["cache"]),
               grid_id=grid.grid_id, shape=list(grid.shape), omega_ratio=omega_ratio, n_mask_nodes=int(pi.size),
               N_q=float(pi.sum()), eta_tmp=e, b_P=bP, Lambda_min=lam_min, class_names=["intensity"]+CLASS_NAMES,
               reference=dict(run=spec["reference_run"], sigma2_used=row["sigma2_used"], G0=row["G0"],
                              c_dat=row["metric"]["c_dat"]), states={})
    states = {"u0": None}
    states.update(spec["states"])
    merged = {}
    for key, groups in MERGES.items():
        lim, _, _ = level_input(cache, grid, groups)
        tm = _template(lim, eta, native_omega, solver["gauss_tile_size"], lim.fixed.shape[0], 1,
                       solver["logvol_points"], **logvol)
        merged[key] = (tm, groups)
    for sname, run in states.items():
        if run is None:
            u = zero
            uh = None
        else:
            f = np.load(CVG/"runs"/run/"states"/"u_final_L0.npz")
            u = np.asarray(f["u"])
            uh = str(f["state_hash"])
            if state_hash(u) != uh:
                raise RuntimeError(f"{run}: state hash")
        point = p0 if run is None else channel_point(template, u)
        rho, grad = np.asarray(point.rho), np.asarray(point.grad)
        s_m = rho+s_f
        opts = options_for(pi, rho, s_f, s_m0, s_m, sigma_other, template.node_index, grid.shape)
        srec = dict(note=STATE_NOTES[sname], state_hash=uh, options={})
        for oname, o in opts.items():
            sigma2 = np.asarray(o["sigma2"], dtype=np.float64)
            v, w = o.get("v"), o.get("w")
            met = metrics(pi, rho, grad, sigma2, v, w, omega_ratio)
            ratio_eff, eff = effective_ratio(sigma2, w, grad, pi, v)
            rec = dict(note=o["note"], sigma2=sigma2.tolist(), ratio_sigmaI_over_sigma=_ratio(sigma2),
                       ratio_effective=ratio_eff, **met)
            for k in ("support_fraction", "sigma2_out", "out_ratio", "iterations"):
                if k in o:
                    val = o[k]
                    rec[k] = val.tolist() if isinstance(val, np.ndarray) else val
            if w is not None:
                ww = np.asarray(w)
                rec["w_stats"] = dict(mean=[float(x) for x in (ww*pi).sum(axis=1)/pi.sum()],
                                      min=[float(x) for x in ww.min(axis=1)])
            if args.lanczos and run is None and oname in LANCZOS:
                rec["spectrum"] = spectrum(template, sigma2, v, w, eta, grid, lam_min)
            srec["options"][oname] = rec
        # O4: merged groupings (O0 and O1 union-drop@.05 on the merged channels)
        for key, (tm, groups) in merged.items():
            pt = channel_point(tm, u)
            rho_m, grad_m = np.asarray(pt.rho), np.asarray(pt.grad)
            sf_m = np.asarray(tm.fixed_masked)
            pt0 = channel_point(tm, zero)
            sm0_m = np.asarray(pt0.rho)+sf_m
            rho2 = rho_m*rho_m
            s0, _ = m_step(pi, rho2)
            on = support(sf_m, sm0_m, .05).astype(np.float64)
            on[0] = 1.
            s1, den = m_step(pi, rho2, on)
            s1[0] = s0[0]
            for oname, sigma2, v, note in ((f"O4-{key}-O0", s0, None, f"groups {groups}, O0 global MLE"),
                                           (f"O4-{key}-O1drop@0.05", s1, on, f"groups {groups}, O1 union0-drop@.05")):
                met = metrics(pi, rho_m, grad_m, sigma2, v, None, omega_ratio)
                ratio_eff, eff = effective_ratio(sigma2, None, grad_m, pi, v)
                rec = dict(note=note, groups=groups, sigma2=sigma2.tolist(),
                           ratio_sigmaI_over_sigma=_ratio(sigma2), ratio_effective=ratio_eff, **met)
                if v is not None:
                    rec["support_fraction"] = (den/pi.sum()).tolist()
                if args.lanczos and run is None:
                    rec["spectrum"] = spectrum(tm, sigma2, v, None, eta, grid, lam_min)
                srec["options"][oname] = rec
        # support drift of the u-dependent union (only meaningful at u != 0)
        on0 = support(s_f, s_m0, .05)
        onu = support(s_f, s_m, .05)
        srec["support_drift_theta_0.05"] = dict(
            entered=[int(x) for x in (onu & ~on0).sum(axis=1)], left=[int(x) for x in (on0 & ~onu).sum(axis=1)],
            size_u0=[int(x) for x in on0.sum(axis=1)])
        doc["states"][sname] = srec
        print(f"{name} {sname}: done ({time.perf_counter()-T0:.1f} s)", flush=True)
    return doc


def spectrum(template, sigma2, v, w, eta, grid, lam_min):
    vw = None
    if v is not None or w is not None:
        vw = np.ones((template.n_channels, template.node_index.size))
        if v is not None:
            vw = vw*np.asarray(v)
        if w is not None:
            vw = vw*np.asarray(w)
        vw = full_weights(template, vw)
    state = template.replace(sigma2=np.asarray(sigma2), weights=vw)
    ev = evaluate(state, np.zeros(grid.shape+(3,)))
    cd = model_c_dat(state, ev)
    bP = metric_bP(state)
    metric = build_metric(state, c_dat=cd, b_P=bP)
    lz = lanczos_max(state, ev, metric, grid)
    alpha = eta.alpha
    bound = None
    if lz["lambda_max"] is not None and min(1/np.asarray(eta.tau2)) >= 1 and eta.nu >= 1:
        lam_lo = (alpha+1)*lam_min/(cd+bP*lam_min)
        bound = dict(lambda_min_lower=lam_lo, kappa_upper=lz["lambda_max"]/lam_lo)
    return dict(c_dat_model=cd, b_P=bP, **lz, kappa_bound=bound)


T0 = time.perf_counter()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/home/rmcfz/seq_vp_tag_results/sig_v1")
    parser.add_argument("--no-lanczos", dest="lanczos", action="store_false")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out/"sigma_options.json"
    if target.exists():
        raise FileExistsError(f"{target} exists (nothing is overwritten)")
    # the O0 u = 0 sigma of each pair (for the cross-pair "population" proxy of O2)
    sig0 = {}
    for name in PAIRS:
        ref = json.loads((CVG/"runs"/PAIRS[name]["reference_run"]/"result.json").read_text())
        sig0[name] = np.asarray(ref["rounds"][0]["sigma2_used"])
    docs = {}
    for name, spec in PAIRS.items():
        docs[name] = run_pair(name, spec, args, sig0[spec["other"]])
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    result = dict(work_package="SIG", note="analysis only (开发集与 T0 各一对，单次，诊断；不是方法结果)",
                  snapshot=str(SNAPSHOT), script=str(Path(__file__).resolve()),
                  script_sha256=sha256_file(__file__), pairs=docs, peak_rss_mb=peak,
                  seconds=time.perf_counter()-T0)
    target.write_text(json.dumps(result, indent=1)+"\n", encoding="utf-8")
    print(f"wrote {target}; peak RSS {peak:.0f} MB; {time.perf_counter()-T0:.1f} s")


if __name__ == "__main__":
    main()
