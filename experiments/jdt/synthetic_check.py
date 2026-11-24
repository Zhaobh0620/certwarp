"""JDT: the draft's estimate "certificate_bernstein" on the synthetic fields of the existing jd_cell_estimate tests
(tests/test_bayesreg/solve/test_jd_cell_estimate.py certified_fields / path_fields, the "mid" and "off" edge dips).
Diagnostic for the draft's gates; nothing in bayesreg/ or jd_curl/ changes, the option is evaluated with existing
read-only functions.  No data/ access.

    python -B experiments/jdt/synthetic_check.py [--out FILE]
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from jd_curl.three_d.cell_q1 import VERTICES, CellQ1Operators3D  # noqa: E402
from jd_curl.three_d.certificate_prescreen import prescreen_lower_bounds  # noqa: E402

from bayesreg.eval.cert_eval import PROVEN_FALLBACK, PROVEN_HULL, PROVEN_PRESCREEN, cell_pass  # noqa: E402
from bayesreg.solve import NumpyBackend  # noqa: E402
from bayesreg.solve import jd_scaling as jds  # noqa: E402
from tests.test_bayesreg.solve import test_jd_cell_estimate as T  # noqa: E402


def factors(m, grid):
    return np.minimum(1., np.maximum(jds.node_minimum(m.reshape(grid.cell_shape), grid), jds.J_FLOOR)/jds.J_ACT).ravel()


def evaluate(name, grid, u, backend):
    cert = backend.certify(u, grid)
    if not cert.passed:
        return dict(name=name, passed=False)
    cfg = backend.cert_config
    jds.clear_cell_bounds_cache()
    bounds, _ = jds.certificate_cell_bounds(u, grid, cert, cfg)
    b = np.asarray(bounds).ravel()
    code, _ = cell_pass(u, grid, cfg)
    table = CellQ1Operators3D(grid, location="closed27_uniform").coefficients
    ell = prescreen_lower_bounds(u, grid, table, np.arange(grid.n_cells, dtype=np.int64))
    hull = code == PROVEN_HULL
    S = np.flatnonzero(hull & (b < jds.J_ACT))
    bt = b.copy()
    up = np.isfinite(ell[S]) & (ell[S] > b[S])
    bt[S[up]] = ell[S][up]
    bt_full = np.where(np.isfinite(ell) & (ell > b), ell, b)
    m53, m_def = jds.closed27_cell_estimate(u, grid)
    J53, m_def = m53.ravel(), m_def.ravel()

    def mm(bb):
        return np.where(np.isfinite(bb) & (bb < m_def), bb, m_def)

    r_def, r_cert, r_new, r_full, r_53 = (factors(x, grid) for x in (m_def, mm(b), mm(bt), mm(bt_full), J53))
    pre, fb = code == PROVEN_PRESCREEN, code == PROVEN_FALLBACK
    return dict(name=name, passed=True, cells=int(grid.n_cells), hull=int(hull.sum()), prescreen=int(pre.sum()),
                fallback=int(fb.sum()), S=int(S.size), raised=int(up.sum()),
                nodes_r_raised_vs_cert=int(np.sum(r_new > r_cert)),
                frozen=dict(default=int(np.sum(r_def < .01)), certificate=int(np.sum(r_cert < .01)),
                            certificate_bernstein=int(np.sum(r_new < .01)), J53=int(np.sum(r_53 < .01))),
                checks=dict(r_new_ge_r_cert=bool(np.all(r_new >= r_cert)), r_new_le_r_53=bool(np.all(r_new <= r_53)),
                            restricted_equals_unrestricted=bool(np.array_equal(r_new, r_full)),
                            bt_le_J53=bool(np.all(bt <= J53)), bt_positive=bool(np.all(bt > 0)),
                            ell_equals_b_on_prescreen=bool(np.array_equal(ell[pre], b[pre])),
                            fallback_ell_not_positive=bool(np.all(~(ell[fb] > 0)))),
                min_b=float(b.min()), min_bt=float(bt.min()), min_J53=float(J53.min()),
                max_ratio_bt_over_b_on_S=float(np.max(bt[S]/b[S])) if S.size else None), (r_cert, r_new, code, b, bt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/home/rmcfz/seq_vp_tag_results/jdt_v1/synthetic.json")
    a = p.parse_args()
    backend = NumpyBackend()
    rows = []
    for name, grid, u in T.certified_fields():
        rows.append(evaluate(name, grid, u, backend)[0])
    for kind in ("mid", "off"):
        grid, cell, u, _ = T.dip_problem(kind=kind)
        row, (r_cert, r_new, code, b, bt) = evaluate(f"dip_{kind}", grid, u, backend)
        K = int(np.ravel_multi_index(cell, grid.cell_shape))
        vertices = [np.ravel_multi_index((cell[0]+z-1, cell[1]+y-1, cell[2]+x-1), grid.interior_shape)
                    for x, y, z in VERTICES]
        row["dip_cell"] = dict(code=int(code[K]), b=float(b[K]), bt=float(bt[K]), bt_equals_b=bool(bt[K] == b[K]),
                               max_r_vertices_certificate=float(max(r_cert[v] for v in vertices)),
                               max_r_vertices_certificate_bernstein=float(max(r_new[v] for v in vertices)))
        rows.append(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1)+"\n", encoding="utf-8")
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
