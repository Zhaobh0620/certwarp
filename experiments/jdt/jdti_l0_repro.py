"""JDTI: the implemented jd_cell_estimate "certificate_bernstein" against the JDT prototype on the 12 stored L0 states.

Two passes, two processes (no solver run, no data/ access, CPU only; reads the CVG cache metadata and states only):

    # 1. prototype: the estimate exactly as experiments/jdt/jdt_l0.py computed it (sections 1-3), on the CVG snapshot
    CVG_CODE=<cvg snapshot> python -B experiments/jdt/jdti_l0_repro.py --pass prototype --work DIR
    # 2. implementation: bayesreg.solve.jd_scaling.scaling_factors(estimate="certificate_bernstein") of THIS tree,
    #    compared with pass 1 (r bit for bit) and with the JDT diagnostic outputs (jdt_v1/<label>.json counts)
    python -B experiments/jdt/jdti_l0_repro.py --pass implementation --work DIR --out FILE

Grid: the L0 grid of the CVG cache (shape and grid_id from its metadata) rebuilt as four_level_grids of the native OASIS
grid 160 x 224 x 192 (xyz), 1 mm, origin (-79, -112, -95) mm (the native_grid record of run_pair provenance.json for
both pairs, e.g. m2_gpu_smoke_v2/H1 and eta_sweep_v3/probe/0012-0013) -- checked against the cached grid_id (refused
otherwise).  The state's
certificate: NumpyBackend().certify (C07-2 configuration), as jdt_l0.py.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

CVG = Path("/home/rmcfz/seq_vp_tag_results/cvg_v1")
JDT = Path("/home/rmcfz/seq_vp_tag_results/jdt_v1")
STATES = (("R1-struct", "u_L0_k500.npz", "R1-k500"), ("R1-struct", "u_L0_k1000.npz", "R1-k1000"),
          ("R1-struct", "u_L0_k2000.npz", "R1-k2000"), ("R1-struct", "u_final_L0.npz", "R1-final"),
          ("R2-tissue", "u_L0_k50.npz", "R2-k50"), ("R2-tissue", "u_L0_k200.npz", "R2-k200"),
          ("R2-tissue", "u_L0_k500.npz", "R2-k500"), ("R2-tissue", "u_L0_k1000.npz", "R2-k1000"),
          ("R2-tissue", "u_L0_k2000.npz", "R2-k2000"), ("R2-tissue", "u_final_L0.npz", "R2-final"),
          ("R7-struct-0012", "u_L0_k1000.npz", "R7-k1000"), ("R7-struct-0012", "u_final_L0.npz", "R7-final"))
CACHES = {"R1-struct": "0438-0439", "R2-tissue": "0438-0439", "R7-struct-0012": "0012-0013"}


def l0_grid(cache_name):
    import numpy as np
    from jd_curl.three_d.grid import make_grid_3d
    from bayesreg.grids import four_level_grids
    meta = json.loads(str(np.load(CVG/"cache"/f"{cache_name}.npz", allow_pickle=False)["meta_json"]))
    native = make_grid_3d(nx=160, ny=224, nz=192, hx=1., hy=1., hz=1., x0=-79., y0=-112., z0=-95., unit="mm")
    grid = four_level_grids(native)[0]
    if grid.grid_id != meta["grid_ids"]["0"] or list(grid.shape) != meta["shapes"]["0"]:
        raise SystemExit(f"{cache_name}: rebuilt L0 grid differs from the cache ({grid.grid_id[:12]})")
    return grid


def load_state(run, name):
    import numpy as np
    from bayesreg.solve.stopping import state_hash
    stored = np.load(CVG/"runs"/run/"states"/name)
    u = np.array(stored["u"])
    if state_hash(u) != str(stored["state_hash"]):
        raise SystemExit(f"{run}/{name}: state_hash differs")
    return u


def prototype(work):
    """jdt_l0.py sections 1-3 verbatim (b, code, S, l_S, b~, m'' and the factors), on the snapshot's code."""
    import numpy as np
    import bayesreg
    if not str(Path(bayesreg.__file__).resolve()).startswith(os.environ.get("CVG_CODE", "<unset>")):
        raise SystemExit(f"bayesreg imported from {bayesreg.__file__}, not the CVG snapshot")
    from bayesreg.eval.cert_eval import PROVEN_HULL, cell_pass
    from bayesreg.solve import NumpyBackend
    from bayesreg.solve.jd_scaling import (J_ACT, J_FLOOR, cell_jd_estimate, certificate_cell_bounds,
                                           clear_cell_bounds_cache, node_minimum)
    from jd_curl.three_d.cell_q1 import CellQ1Operators3D
    from jd_curl.three_d.certificate_prescreen import prescreen_lower_bounds

    def factors(m, grid):
        return np.minimum(1., np.maximum(node_minimum(m.reshape(grid.cell_shape), grid), J_FLOOR)/J_ACT).ravel()

    for run, name, label in STATES:
        grid = l0_grid(CACHES[run])
        u = load_state(run, name)
        backend = NumpyBackend()
        cert = backend.certify(u, grid)
        cfg = backend.cert_config
        code, lower = cell_pass(u, grid, cfg)
        clear_cell_bounds_cache()
        bounds, _ = certificate_cell_bounds(u, grid, cert, cfg)
        b = np.asarray(bounds).ravel()
        if not np.array_equal(b, lower, equal_nan=True):
            raise SystemExit("certificate_cell_bounds differs from cell_pass")
        hull = code == PROVEN_HULL
        table = CellQ1Operators3D(grid, location="closed27_uniform").coefficients
        S = np.flatnonzero(hull & (b < J_ACT)).astype(np.int64)
        ell_S = prescreen_lower_bounds(u, grid, table, S)
        bt = b.copy()
        raise_S = np.isfinite(ell_S) & (ell_S > b[S])
        bt[S[raise_S]] = ell_S[raise_S]
        m_def = cell_jd_estimate(u, grid).ravel()

        def with_bound(bb):
            has = np.isfinite(bb)
            return np.where(has & (bb < m_def), bb, m_def)

        r_cert, r_new = factors(with_bound(b), grid), factors(with_bound(bt), grid)
        np.savez(Path(work)/f"{label}.npz", r_new=r_new, r_cert=r_cert, S=S, ell_S=ell_S, bt=bt, b=b,
                 state_hash=cert.state_hash, min_jd_lower_bound=cert.min_jd_lower_bound,
                 bayesreg=str(Path(bayesreg.__file__).parent))
        print(f"[prototype {label}] S {S.size} raised {int(raise_S.sum())} r<.01 {int((r_new < .01).sum())}",
              flush=True)


def implementation(work, out):
    import numpy as np
    import bayesreg
    from bayesreg.solve import JDScaledMetric, JDScalingConfig, NumpyBackend
    from bayesreg.solve import jd_scaling as jds
    root = Path(__file__).resolve().parents[2]
    if not str(Path(bayesreg.__file__).resolve()).startswith(str(root)):
        raise SystemExit(f"bayesreg imported from {bayesreg.__file__}, not this tree")

    class Identity:
        a, shift = 0., 0.

        @staticmethod
        def solve(g):
            return g.copy()

        @staticmethod
        def matvec(v):
            return v.copy()

    rows = []
    for run, name, label in STATES:
        grid = l0_grid(CACHES[run])
        u = load_state(run, name)
        proto = np.load(Path(work)/f"{label}.npz")
        backend = NumpyBackend()
        cert = backend.certify(u, grid)
        cfg = backend.cert_config
        jds.clear_cell_bounds_cache()
        tick = time.perf_counter()
        r, rec = jds.scaling_factors(u, grid, estimate=jds.CERTIFICATE_BERNSTEIN, cert=cert, cert_config=cfg)
        seconds = time.perf_counter()-tick
        r_cert, _ = jds.scaling_factors(u, grid, estimate=jds.CERTIFICATE, cert=cert, cert_config=cfg)
        bt, _, _ = jds.bernstein_cell_bounds(u, grid, cert, cfg)
        metric = JDScaledMetric(Identity(), grid, u, config=JDScalingConfig(cell_estimate=jds.CERTIFICATE_BERNSTEIN),
                                cert=cert, cert_config=cfg)
        diag = json.loads((JDT/f"{label}.json").read_text())
        classes, cells = diag["classes"], diag["cells"]
        expected = dict(n_cells_bernstein=cells["n_hull_below_act"], n_cells_raised=cells["n_raised"],
                        n_cells_hull=cells["n_hull"],
                        n_r_lt_1=classes["certificate_bernstein_r_lt_1"]["nodes"],
                        n_r_lt_0p01=classes["certificate_bernstein_r_lt_0.01"]["nodes"],
                        n_r_lt_1_certificate=classes["certificate_r_lt_1"]["nodes"],
                        n_r_lt_0p01_certificate=classes["certificate_r_lt_0.01"]["nodes"],
                        n_r_lt_1_default=classes["gauss27+corners_r_lt_1"]["nodes"],
                        n_r_lt_0p01_default=classes["gauss27+corners_r_lt_0.01"]["nodes"],
                        n_nodes_r_raised_vs_certificate=classes["n_nodes_r_raised_new_vs_certificate"],
                        n_nodes_r_lowered=classes["certificate_bernstein_n_nodes_r_lowered_vs_default"],
                        n_nodes_r_lowered_certificate=classes["certificate_n_nodes_r_lowered_vs_default"])
        mismatched = {k: (rec[k], v) for k, v in expected.items() if rec[k] != v}
        row = dict(label=label, state_hash=cert.state_hash,
                   state_hash_matches_prototype=cert.state_hash == str(proto["state_hash"]),
                   r_bitwise_equal_prototype=bool(np.array_equal(r.ravel(), proto["r_new"])),
                   r_certificate_bitwise_equal_prototype=bool(np.array_equal(r_cert.ravel(), proto["r_cert"])),
                   bt_bitwise_equal_prototype=bool(np.array_equal(bt, proto["bt"], equal_nan=True)),
                   S_equal_prototype=bool(np.array_equal(jds._BERNSTEIN_CACHE["S"], proto["S"])),
                   ell_bitwise_equal_prototype=bool(np.array_equal(jds._BERNSTEIN_CACHE["ell"], proto["ell_S"],
                                                                   equal_nan=True)),
                   metric_r_bitwise_equal=bool(np.array_equal(metric.r, r)),
                   min_J_a_equals_diagnostic=rec["min_J_a"] == diag["cells"]["min_J_a"]["certificate_bernstein"],
                   min_bernstein_bound_equals_diagnostic_min_bt=rec["min_bernstein_bound"] == cells["min_bt"],
                   counts_equal_diagnostic=not mismatched, count_mismatches=mismatched,
                   record={k: v for k, v in rec.items() if not k.endswith("seconds")}, seconds=seconds)
        rows.append(row)
        print(f"[implementation {label}] r==proto {row['r_bitwise_equal_prototype']} bt==proto "
              f"{row['bt_bitwise_equal_prototype']} counts==diag {row['counts_equal_diagnostic']} "
              f"S {rec['n_cells_bernstein']} raised {rec['n_cells_raised']} r<.01 {rec['n_r_lt_0p01']} "
              f"(cert {rec['n_r_lt_0p01_certificate']}) viol {rec['n_bernstein_j53_violations']} "
              f"margin {rec['bernstein_j53_min_margin']:.3g}", flush=True)
    keys = [k for k in rows[0] if isinstance(rows[0][k], bool)]
    doc = dict(purpose="JDTI L0 reproduction: implementation vs JDT prototype (not a method result; no solver run)",
               script=str(Path(__file__).resolve()), bayesreg=str(Path(bayesreg.__file__).parent),
               threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
               all_true={k: all(r[k] for r in rows) for k in keys}, states=rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=1)
    print(json.dumps(doc["all_true"]))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pass", dest="which", required=True, choices=("prototype", "implementation"))
    p.add_argument("--work", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    Path(a.work).mkdir(parents=True, exist_ok=True)
    if a.which == "prototype":
        cvg = os.environ.get("CVG_CODE")
        if not cvg:
            raise SystemExit("CVG_CODE (the CVG snapshot) required for the prototype pass")
        sys.path.insert(0, cvg)
        prototype(a.work)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        implementation(a.work, a.out)


if __name__ == "__main__":
    main()
