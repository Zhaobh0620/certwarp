"""KKT supplement (descriptive, not pre-registered): how each gradient block sits relative to the cone of constraint
normals of the truly near-active cells (dense minimum J_dense < t).  For each block b (channel, logvol, curl) and its
negative: the NNLS share of ||g_b restricted to the support||^2 explained by the cone, and the cosine of g_b with the
sum of the unit normals.  Same model, normals and mapping as kkt.py.

    CVG_CODE=<cvg snapshot> python -B experiments/kkt/block_cone.py --run <cvg run> --state <npz> --label L --out DIR
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kkt  # noqa: E402

import numpy as np  # noqa: E402
from bayesreg.model import evaluate, gradient  # noqa: E402
from bayesreg.solve.jd_scaling import certificate_cell_bounds  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    config, result, row, state, metric = kkt.setup(Path(a.run))
    grid = state.grid
    u = np.array(np.load(a.state)["u"])
    point = evaluate(state, u)
    blocks = {b: gradient(state, point, blocks=(b,)) for b in kkt.BLOCKS}
    blocks["total"] = gradient(state, point)
    backend = kkt.common.CaptureBackend()
    cert = backend.certify(u, grid)
    bounds, _ = certificate_cell_bounds(u, grid, cert, backend.cert_config)
    b = np.where(np.isfinite(bounds.ravel()), bounds.ravel(), np.inf)
    cand = np.flatnonzero(b < .25)
    t1 = np.linspace(0., 1., kkt.NGRID)
    gp = np.stack(np.meshgrid(t1, t1, t1, indexing="ij"), -1).reshape(-1, 3)
    G = kkt.basis_gradients(gp, grid.spacing)
    U = kkt.corner_values(u, cand, grid.cell_shape)
    Jg = kkt.det3(kkt.deformation(U, G))
    res = [kkt.dense_minimum(U[m], grid.spacing, gp, Jg[m]) for m in range(len(cand))]
    Jd = np.array([r[1] for r in res])
    out = dict(label=a.label, state=a.state, by_threshold={})
    for t in (5e-2, 1e-2):
        sel = np.flatnonzero(Jd < t)
        if len(sel) == 0:
            continue
        normals = np.stack([kkt.normal_at(U[m], res[m][0], grid.spacing) for m in sel])
        rows, vals = kkt.free_columns(cand[sel], normals, grid)
        support = np.unique(np.concatenate(rows))
        A = kkt.dense_matrix(rows, vals, support)
        unit_sum = (A/np.linalg.norm(A, axis=0)).sum(axis=1)
        entry = dict(n_cells=int(len(sel)), J_dense=Jd[sel].tolist(), blocks={})
        for name, gb in blocks.items():
            gs = gb[support]
            ss = float(gs@gs)
            rec = dict(norm_on_support_GF=float(np.sqrt(ss)/np.sqrt(grid.omega/state.omega0)),
                       cos_with_unit_normal_sum=float(gs@unit_sum/max(np.linalg.norm(gs)*np.linalg.norm(unit_sum), 1e-300)))
            for sign, s in (("plus", 1.), ("minus", -1.)):
                lam, r = kkt.nnls_fit(A, s*gs)
                rec[f"{sign}_explained_share_on_support"] = 1-r**2/ss if ss > 0 else None
            entry["blocks"][name] = rec
        out["by_threshold"][str(t)] = entry
        print(a.label, t, len(sel), {k: (round(v["norm_on_support_GF"], 1), round(v["cos_with_unit_normal_sum"], 3),
                                         round(v["plus_explained_share_on_support"], 3),
                                         round(v["minus_explained_share_on_support"], 3))
                                     for k, v in entry["blocks"].items()}, flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out)/f"block_cone_{a.label}.json").write_text(json.dumps(out, indent=1)+"\n")


if __name__ == "__main__":
    main()
