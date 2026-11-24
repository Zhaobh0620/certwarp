"""JDT (contract draft C09.1-3a jd_cell_estimate "certificate_bernstein"): L0 diagnostic on stored CVG states.

Diagnostic only (no solver run; bayesreg/ and jd_curl/ unchanged; the draft's option is NOT implemented here, its
definition is evaluated on stored states with the existing read-only functions):

    CVG_CODE=<cvg snapshot> python -B experiments/jdt/jdt_l0.py --run <cvg run dir> --state <npz> --k K
        --label NAME --out /home/rmcfz/seq_vp_tag_results/jdt_v1

At the accepted, certified state u of a CVG L0 run (model state rebuilt as experiments/kkt/kkt.py `setup`, GF_B / F_B
checked against trace.csv, state_hash against the accept row):

1. b_K, code_K: the per-cell certified lower bounds and paths (bayesreg.eval.cert_eval.cell_pass, C07-2 configuration);
   jd_scaling.certificate_cell_bounds on the same state (bitwise min check with the certificate).
2. l_K = jd_curl prescreen_lower_bounds (min of the float Bernstein coefficients minus the T75 a-priori error bound B,
   a rigorous lower bound of J over the cell) for every cell (timed) and for S = {edge-hull cells with b_K < J_act}
   (timed); l_K == b_K bitwise on prescreen cells, l_K <= 0 or NaN on fallback cells.
3. The draft's estimate: b~_K = max(b_K, l_K) on S (l_K finite), b_K elsewhere; m''_K = min(m_K, b~_K) (cells without
   bound keep m_K); the unrestricted variant max(b_K, l_K) on every cell must give the same r bitwise.
4. Checks: b~_K <= min of J at the 53 points (Gauss27 + closed27) of every cell; r'' >= r_certificate everywhere;
   r'' <= r of the 53-point minimum everywhere.
5. Node factors r = min(1, max(J_a, 1e-12)/.25) for gauss27+corners, certificate, certificate_bernstein, the 53-point
   minimum and a dense minimum (17^3 grid + L-BFGS-B, kkt.dense_minimum) on every cell with b~_K < .25 or b_K < .25 or
   J53 < .25: node counts r < 1, .1, .01 and their share of ||g||^2 (g = grad F_B at the state, free layout).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"kkt"))
import kkt  # noqa: E402  (inserts experiments/cvg and CVG_CODE at the front of sys.path)
import common  # noqa: E402

import numpy as np  # noqa: E402

from bayesreg.geom import node_field  # noqa: E402
from bayesreg.model import evaluate, gradient  # noqa: E402
from bayesreg.eval.cert_eval import PROVEN_FALLBACK, PROVEN_HULL, PROVEN_PRESCREEN, cell_pass  # noqa: E402
from bayesreg.solve.jd_scaling import (CERTIFICATE, J_ACT, J_FLOOR, cell_jd_estimate, certificate_cell_bounds,  # noqa: E402
                                       clear_cell_bounds_cache, closed27_cell_estimate, node_minimum, scaling_factors)
from bayesreg.solve.stopping import gf_b, state_hash  # noqa: E402
from jd_curl.three_d.cell_q1 import CellQ1Operators3D  # noqa: E402
from jd_curl.three_d.certificate_prescreen import prescreen_lower_bounds  # noqa: E402

LIMITS = (1., .1, .01)
REPEATS = 5


def factors(m, grid):
    return np.minimum(1., np.maximum(node_minimum(m.reshape(grid.cell_shape), grid), J_FLOOR)/J_ACT).ravel()


def timed(fn, repeats=REPEATS):
    out, seconds = None, []
    for _ in range(repeats):
        tick = time.perf_counter()
        out = fn()
        seconds.append(time.perf_counter()-tick)
    return out, dict(median=float(np.median(seconds)), min=float(min(seconds)), max=float(max(seconds)), n=repeats)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, required=True)
    a = p.parse_args(argv)
    kkt.rss_watchdog(1800)
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
    config, result, row, state, metric = kkt.setup(run)
    grid = state.grid
    n_int = grid.n_free_scalar
    stored = np.load(a.state)
    u = np.array(stored["u"])
    digest = state_hash(u)
    if digest != str(stored["state_hash"]):
        raise RuntimeError("state file hash differs from its stored state_hash")
    logged = kkt.accept_hash(run, a.k)
    trow = kkt.trace_row(run, a.k)
    doc = dict(label=a.label, run=str(run), state=a.state, k=a.k, state_hash=digest, code=common.code_record(),
               script_sha256=common.sha256_file(__file__), kkt_sha256=common.sha256_file(kkt.__file__),
               grid_shape=list(grid.shape), n_cells=int(grid.n_cells), channel_source=config["args"]["channel_source"],
               threads=dict(NUMBA_NUM_THREADS=os.environ.get("NUMBA_NUM_THREADS"),
                            OPENBLAS_NUM_THREADS=os.environ.get("OPENBLAS_NUM_THREADS")),
               loadavg_before=list(os.getloadavg()))
    point = evaluate(state, u)
    g = gradient(state, point)
    GF = gf_b(g, grid.omega, state.omega0)
    rel = abs(GF-trow["GF_B"])/trow["GF_B"]
    relF = abs(point.energy-trow["F_B"])/abs(trow["F_B"])
    doc["check"] = dict(GF_B=GF, GF_B_rel_diff=rel, F_B_rel_diff=relF,
                        hash_matches_accept_row=(logged == digest) if logged else None)
    if rel > 1e-10 or relF > 1e-10 or (logged is not None and logged != digest):
        raise SystemExit(f"state check failed: {doc['check']}")
    # ---- 1. certificate and per-cell bounds
    backend = common.CaptureBackend()
    cert = backend.certify(u, grid)
    if not cert.passed:
        raise RuntimeError("stored state does not pass the certificate")
    cfg = backend.cert_config
    (code, lower), t_pass = timed(lambda: cell_pass(u, grid, cfg))
    clear_cell_bounds_cache()
    bounds, brec = certificate_cell_bounds(u, grid, cert, cfg)
    b = np.asarray(bounds).ravel()
    if not np.array_equal(b, lower, equal_nan=True):
        raise RuntimeError("certificate_cell_bounds differs from cell_pass")
    hull, pre, fb = code == PROVEN_HULL, code == PROVEN_PRESCREEN, code == PROVEN_FALLBACK
    # ---- 2. prescreen lower bounds
    table = CellQ1Operators3D(grid, location="closed27_uniform").coefficients
    all_cells = np.arange(grid.n_cells, dtype=np.int64)
    ell_all, t_all = timed(lambda: prescreen_lower_bounds(u, grid, table, all_cells))
    S = np.flatnonzero(hull & (b < J_ACT)).astype(np.int64)
    ell_S, t_S = timed(lambda: prescreen_lower_bounds(u, grid, table, S))
    _, t_table = timed(lambda: CellQ1Operators3D(grid, location="closed27_uniform").coefficients)
    checks = dict(
        ell_subset_equals_all_bitwise=bool(np.array_equal(ell_S, ell_all[S], equal_nan=True)),
        ell_equals_b_on_prescreen_cells_bitwise=bool(np.array_equal(ell_all[pre], b[pre], equal_nan=True)),
        n_fallback_cells=int(fb.sum()),
        fallback_ell_not_positive=bool(np.all(~(ell_all[fb] > 0))),
    )
    # ---- 3. the draft's estimate
    bt = b.copy()
    raise_S = np.isfinite(ell_S) & (ell_S > b[S])
    bt[S[raise_S]] = ell_S[raise_S]
    bt_full = np.where(np.isfinite(ell_all) & (ell_all > b), ell_all, b)
    m_def = cell_jd_estimate(u, grid).ravel()
    m53_cells, m_def2 = closed27_cell_estimate(u, grid)
    J53 = m53_cells.ravel()
    checks["closed27_default_equals_cell_jd_estimate"] = bool(np.array_equal(m_def2.ravel(), m_def))

    def with_bound(bb):
        has = np.isfinite(bb)
        return np.where(has & (bb < m_def), bb, m_def)

    m_cert, m_new, m_new_full = with_bound(b), with_bound(bt), with_bound(bt_full)
    r_def, r_cert, r_new, r_new_full, r_53 = (factors(m, grid) for m in (m_def, m_cert, m_new, m_new_full, J53))
    r_cert_lib, rec_cert = scaling_factors(u, grid, estimate=CERTIFICATE, cert=cert, cert_config=cfg)
    checks.update(
        r_certificate_equals_library_bitwise=bool(np.array_equal(r_cert, r_cert_lib.ravel())),
        r_restricted_equals_unrestricted_bitwise=bool(np.array_equal(r_new, r_new_full)),
        n_cells_bt_above_J53=int(np.sum(bt > J53)),
        n_cells_bt_full_above_J53=int(np.sum(bt_full > J53)),
        min_J53_minus_bt_over_S=float(np.min(J53[S]-bt[S])) if S.size else None,
        r_new_ge_r_cert_everywhere=bool(np.all(r_new >= r_cert)),
        r_new_le_r_53_everywhere=bool(np.all(r_new <= r_53)),
        all_bt_positive=bool(np.all(bt[np.isfinite(bt)] > 0)),
    )
    # ---- dense minimum on the low cells
    low = np.flatnonzero((bt < J_ACT) | (b < J_ACT) | (J53 < J_ACT))
    t1 = np.linspace(0., 1., kkt.NGRID)
    gp = np.stack(np.meshgrid(t1, t1, t1, indexing="ij"), -1).reshape(-1, 3)
    G_grid = kkt.basis_gradients(gp, grid.spacing)
    Ulow = kkt.corner_values(u, low, grid.cell_shape)
    Jd = np.empty(len(low))
    for s in range(0, len(low), 64):
        Jg = kkt.det3(kkt.deformation(Ulow[s:s+64], G_grid))
        for m in range(Jg.shape[0]):
            Jd[s+m] = min(kkt.dense_minimum(Ulow[s+m], grid.spacing, gp, Jg[m])[1], float(J53[low[s+m]]))
    m_true = np.maximum(J53, J_ACT)          # cells outside `low`: J53 >= .25 and b~, b >= .25; r = 1 from them
    m_true[low] = Jd
    r_true = factors(m_true, grid)
    checks["n_low_cells_bt_above_dense"] = int(np.sum(bt[low] > Jd))
    checks["n_low_cells_b_above_dense"] = int(np.sum(b[low] > Jd))
    # ---- 4. counts
    node_g2 = (g.reshape(3, n_int)**2).sum(axis=0)
    tot = float(node_g2.sum())
    classes = {}
    for name, r in (("gauss27+corners", r_def), ("certificate", r_cert), ("certificate_bernstein", r_new),
                    ("closed27_53pt", r_53), ("dense_min", r_true)):
        for lim in LIMITS:
            sel = r < lim
            classes[f"{name}_r_lt_{lim:g}"] = dict(nodes=int(sel.sum()), share_g2=float(node_g2[sel].sum()/tot))
    for name, r in (("certificate", r_cert), ("certificate_bernstein", r_new)):
        sel = (r < .01) & (r_true >= .1)
        classes[f"{name}_r_lt_0.01_but_dense_r_ge_0.1"] = dict(nodes=int(sel.sum()), share_g2=float(node_g2[sel].sum()/tot))
        classes[f"{name}_n_nodes_r_lowered_vs_default"] = int(np.sum(r < r_def))
    sel = r_new > r_cert
    classes["n_nodes_r_raised_new_vs_certificate"] = int(sel.sum())
    hS = S
    ratio_b = b[hS]/Jd[np.searchsorted(low, hS)] if hS.size else np.zeros(0)
    ratio_bt = bt[hS]/Jd[np.searchsorted(low, hS)] if hS.size else np.zeros(0)
    doc["cells"] = dict(
        n_hull=int(hull.sum()), n_prescreen=int(pre.sum()), n_fallback=int(fb.sum()),
        n_hull_below_act=int(S.size), n_raised=int(raise_S.sum()),
        n_hull_below_act_ell_not_positive=int(np.sum(~(ell_S > 0))),
        n_hull_below_act_ell_below_b=int(np.sum(np.isfinite(ell_S) & (ell_S < b[S]))),
        n_hull_all_ell_below_b=int(np.sum(hull & np.isfinite(ell_all) & (ell_all < b))),
        n_low_cells_dense=int(len(low)),
        ratio_b_over_dense_on_S=kkt.summary(ratio_b), ratio_bt_over_dense_on_S=kkt.summary(ratio_bt),
        n_S_bt_over_dense_ge_0p9=int(np.sum(ratio_bt >= .9)), n_S_b_over_dense_lt_0p1=int(np.sum(ratio_b < .1)),
        min_b=float(np.nanmin(b)), min_bt=float(np.nanmin(bt)), min_J53=float(J53.min()), min_dense=float(m_true.min()),
        min_J_a=dict(default=float(node_minimum(m_def.reshape(grid.cell_shape), grid).min()),
                     certificate=float(node_minimum(m_cert.reshape(grid.cell_shape), grid).min()),
                     certificate_bernstein=float(node_minimum(m_new.reshape(grid.cell_shape), grid).min())),
    )
    doc["record_certificate_library"] = rec_cert
    doc["classes"] = classes
    doc["checks"] = checks
    doc["timing"] = dict(cell_pass=t_pass, prescreen_all_cells=t_all, prescreen_S=t_S, closed27_table=t_table,
                         per_cell_prescreen_all_us=1e6*t_all["median"]/grid.n_cells,
                         per_cell_prescreen_S_us=(1e6*t_S["median"]/S.size) if S.size else None,
                         note="L0 (13440 cells), one numba thread, nice 19, shared machine (load in loadavg)")
    doc["seconds"] = time.perf_counter()-tick0
    doc["loadavg_after"] = list(os.getloadavg())
    import resource
    doc["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    common.write_json(target, doc)
    c = classes
    print(f"[{a.label}] hull {doc['cells']['n_hull']} S {S.size} raised {raise_S.sum()} | r<.01 nodes/share: "
          f"def {c['gauss27+corners_r_lt_0.01']} cert {c['certificate_r_lt_0.01']} new {c['certificate_bernstein_r_lt_0.01']} "
          f"dense {c['dense_min_r_lt_0.01']} | checks {checks}", flush=True)


if __name__ == "__main__":
    main()
