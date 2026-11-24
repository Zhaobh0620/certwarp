"""KKT implementation checks (run after kkt.py's own checks): (1) the free-vector mapping of the constraint normals,
by a central difference of J(x*) along the injected free direction d = n_K (expected n_K . d = ||n_K||^2); (2) NNLS
recovery on a synthetic gradient g_syn = sum lambda_K n_K + noise (lambda_K > 0), which the fit must explain.

    CVG_CODE=<cvg snapshot> python -B experiments/kkt/check_mapping.py --state <npz> --out <json>
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kkt  # noqa: E402  (imports common, which puts CVG_CODE first on sys.path)

import numpy as np  # noqa: E402
from jd_curl.three_d.boundary import inject_free  # noqa: E402
from bayesreg.solve.jd_scaling import certificate_cell_bounds  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    config, result, row, state, metric = kkt.setup(Path(a.run))
    grid = state.grid
    u = np.array(np.load(a.state)["u"])
    backend = kkt.common.CaptureBackend()
    cert = backend.certify(u, grid)
    bounds, _ = certificate_cell_bounds(u, grid, cert, backend.cert_config)
    b = np.where(np.isfinite(bounds.ravel()), bounds.ravel(), np.inf)
    cells = np.flatnonzero(b < 1e-2)
    t1 = np.linspace(0., 1., kkt.NGRID)
    gp = np.stack(np.meshgrid(t1, t1, t1, indexing="ij"), -1).reshape(-1, 3)
    G = kkt.basis_gradients(gp, grid.spacing)
    U = kkt.corner_values(u, cells, grid.cell_shape)
    Jg = kkt.det3(kkt.deformation(U, G))
    xs = np.array([kkt.dense_minimum(U[m], grid.spacing, gp, Jg[m])[0] for m in range(len(cells))])
    normals = np.stack([kkt.normal_at(U[m], xs[m], grid.spacing) for m in range(len(cells))])
    rows, vals = kkt.free_columns(cells, normals, grid)
    n = 3*grid.n_free_scalar
    out = dict(state=a.state, cells=cells.tolist(), mapping=[])
    for m in range(len(cells)):
        d = np.zeros(n)
        d[rows[m]] = vals[m]
        field = inject_free(d, grid)
        eps = 1e-7
        Jp = kkt.J_points(kkt.corner_values(u+eps*field, cells[m:m+1], grid.cell_shape), xs[m][None], grid.spacing)[0, 0]
        Jm = kkt.J_points(kkt.corner_values(u-eps*field, cells[m:m+1], grid.cell_shape), xs[m][None], grid.spacing)[0, 0]
        fd = (Jp-Jm)/(2*eps)
        expect = float(vals[m]@vals[m])
        out["mapping"].append(dict(cell=int(cells[m]), fd=fd, expected=expect, rel=abs(fd-expect)/expect))
    # synthetic NNLS recovery
    rng = np.random.default_rng(1)
    lam_true = rng.uniform(.5, 2., len(cells))
    g_syn = np.zeros(n)
    for m in range(len(cells)):
        g_syn[rows[m]] += lam_true[m]*vals[m]
    noise = rng.standard_normal(n)
    noise *= .01*np.linalg.norm(g_syn)/np.linalg.norm(noise)
    g_syn += noise
    support = np.unique(np.concatenate(rows))
    rest = np.ones(n, bool)
    rest[support] = False
    A = kkt.dense_matrix(rows, vals, support)
    lam, res = kkt.nnls_fit(A, g_syn[support], float(g_syn[rest]@g_syn[rest]))
    out["synthetic"] = dict(n_cells=len(cells), noise_rel=.01, r_over_norm=res/np.linalg.norm(g_syn),
                            explained_share=1-res**2/float(g_syn@g_syn),
                            lambda_rel_err_max=float(np.max(np.abs(lam-lam_true)/lam_true)))
    lam_m, res_m = kkt.nnls_fit(A, -g_syn[support], float(g_syn[rest]@g_syn[rest]))
    out["synthetic"]["minus_g_explained_share"] = 1-res_m**2/float(g_syn@g_syn)
    Path(a.out).write_text(json.dumps(out, indent=1)+"\n")
    print(json.dumps(dict(max_mapping_rel=max(x["rel"] for x in out["mapping"]), **out["synthetic"]), indent=1))


if __name__ == "__main__":
    main()
