"""Independent recomputation of the smallest corner J of the method's 67 outputs (reviewer, 2026-11-23).

A second implementation of the corner values behind posthoc_minj.py / minj_by_position.csv, written without
bayesreg.geom, jd_curl or any repository module: numpy only (plus the standard library).

Principle.  Inside a cell the Q1 (trilinear) interpolant of the node field u is, along each coordinate
direction, linear between the two nodes of an edge.  At a corner of the cell the partial derivative along x is
therefore the one-sided difference along the x-edge that leaves that corner inside the cell (and likewise for
y and z).  So at the corner (oz, oy, ox) in {0,1}^3 of the cell (k, j, i):
    d u / d x = (u[k+oz, j+oy, i+1] - u[k+oz, j+oy, i]) / hx
    d u / d y = (u[k+oz, j+1, i+ox] - u[k+oz, j, i+ox]) / hy
    d u / d z = (u[k+1, j+oy, i+ox] - u[k, j+oy, i+ox]) / hz
and J = det(I + G), G[c][a] = d u_c / d x_a, for each of the 8 corners of every cell (equivalently: 8 corner
determinants per node, one for each adjacent cell).  The per-pair minimum over all cells and all 8 corners is
compared with the sampled minimum recorded in minj_by_position.csv (by position number).

Conventions (as in the evaluation, bayesreg/eval/fields.py d_to_u: u[..., 2 - k] = d[k] h_zyx[k]): u is a
(nz, ny, nx, 3) node array in mm, last axis = (x, y, z) components; array axes = (z, y, x); spacing from the
grid metadata stored in the output field itself (output/final.npz key "grid", spacing_xyz).

Reads only: the two decision.json files (position order; sha256 prefix pinned) and the 67 output fields
output/final.npz (keys u, grid, state_hash, status).  No truth, image, label, evaluation record or opponent field.
Fixed paths only, no directory listing.  Output carries position numbers only (guarded).

    python -I -B independent_cornerj.py [--limit N] [--out DIR]
"""
import csv
import hashlib
import json
import os
import re
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/home/rmcfz/seq_vp_tag_results"
SETS = {
    "oasis": dict(name="OASIS hold-out", n=50, decision=RES + "/holdout_B_v1/eval/decision.json", pin="a0dbdcf9",
                  field="/mnt/d/seq_vp_tag_holdout_B_v1/method/{}/output/final.npz"),
    "lpba": dict(name="LPBA40 second test set", n=17, decision=RES + "/lpba40_test_v1/eval/decision.json", pin="a0596c5f",
                 field=RES + "/lpba40_test_v1/work/method/{}/output/final.npz"),
}
THRESHOLDS = (.01, .001)

out = []


def P(*a):
    out.append(" ".join(str(x) for x in a))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corner_minimum(u, h_xyz):
    """Per-cell minimum over the 8 corners of det(I + grad u) of the trilinear interpolant; float64."""
    nz, ny, nx, _ = u.shape
    cz, cy, cx = nz - 1, ny - 1, nx - 1
    hx, hy, hz = h_xyz
    # one-sided edge differences, per component c (0 = x, 1 = y, 2 = z)
    Dx = [(u[:, :, 1:, c] - u[:, :, :-1, c]) / hx for c in range(3)]      # (nz, ny, nx-1)
    Dy = [(u[:, 1:, :, c] - u[:, :-1, :, c]) / hy for c in range(3)]      # (nz, ny-1, nx)
    Dz = [(u[1:, :, :, c] - u[:-1, :, :, c]) / hz for c in range(3)]      # (nz-1, ny, nx)
    cell_min = np.full((cz, cy, cx), np.inf)
    for oz in (0, 1):
        for oy in (0, 1):
            for ox in (0, 1):
                gx = [D[oz:oz + cz, oy:oy + cy, :] for D in Dx]
                gy = [D[oz:oz + cz, :, ox:ox + cx] for D in Dy]
                gz = [D[:, oy:oy + cy, ox:ox + cx] for D in Dz]
                # A[c][a] = delta_ca + d u_c / d x_a, a = (x, y, z)
                a11, a12, a13 = 1. + gx[0], gy[0], gz[0]
                a21, a22, a23 = gx[1], 1. + gy[1], gz[1]
                a31, a32, a33 = gx[2], gy[2], 1. + gz[2]
                J = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
                np.minimum(cell_min, J, out=cell_min)
                del a11, a12, a13, a21, a22, a23, a31, a32, a33, J
    return cell_min


def one_field(path):
    with np.load(path) as z:
        u = np.ascontiguousarray(z["u"], dtype=np.float64)
        grid = json.loads(str(z["grid"]))
        state_hash = str(z["state_hash"])
        status = str(z["status"])
    assert u.shape == tuple(grid["shape_zyx"]) + (3,), "shape"
    assert np.isfinite(u).all(), "finite"
    faces_zero = all(not np.any(s) for s in (u[0], u[-1], u[:, 0], u[:, -1], u[:, :, 0], u[:, :, -1]))
    hash_ok = hashlib.sha256(u.tobytes()).hexdigest() == state_hash
    cm = corner_minimum(u, grid["spacing_xyz"])
    k = int(cm.argmin())
    return dict(corner_min=float(cm.flat[k]), n_cells=int(cm.size), below={t: int((cm < t).sum()) for t in THRESHOLDS},
                spacing=tuple(grid["spacing_xyz"]), faces_zero=faces_zero, state_hash_equal=hash_ok, status=status)


def g(x):
    return f"{x:.4g}"


def Q(name, xs):
    xs = np.asarray(xs, dtype=float)
    q = np.percentile(xs, [0, 25, 50, 75, 100])
    P(f"  {name}: n {len(xs)}; min {g(q[0])}; q25 {g(q[1])}; median {g(q[2])}; q75 {g(q[3])}; max {g(q[4])}")


limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
target = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else HERE
with open(os.path.join(HERE, "minj_by_position.csv")) as f:
    ref = {(r["set"], int(r["position"])): r for r in csv.DictReader(f)}
tokens = set()
t_all = time.perf_counter()
allres = {}
for key, S in SETS.items():
    if sha256_file(S["decision"])[:8] != S["pin"]:
        sys.exit("decision.json pin mismatch")
    with open(S["decision"]) as f:
        positions = sorted(json.load(f)["positions"], key=lambda e: e["position"])
    # the pair list is the one of the pinned decision.json (sha256 prefix checked above), nothing else
    assert len(positions) == S["n"] and [e["position"] for e in positions] == list(range(S["n"]))
    tokens |= {e["pair"] for e in positions}
    res = []
    for e in positions[:limit] if limit else positions:
        t0 = time.perf_counter()
        x = one_field(S["field"].format(e["pair"]))
        x["seconds"] = time.perf_counter() - t0
        x["position"] = e["position"]
        r = ref[(key, e["position"])]
        x["csv_sampled"], x["csv_closed27"] = float(r["sampled_min_J"]), float(r["sampled_min_J_closed27"])
        x["csv_point"] = r["point_of_sampled_min"]
        x["csv_below"] = {.01: int(r["cells_sampled_min_lt_0.01"]), .001: int(r["cells_sampled_min_lt_0.001"])}
        res.append(x)
    allres[key] = res

P("Independent corner-J recomputation (numpy only; no bayesreg / jd_curl module imported)")
P("per position: set, position, corner minimum here, sampled minimum in minj_by_position.csv, |difference|, relative difference,"
  " cells with corner J < .01 here / cells with sampled J < .01 in the csv, same for < .001")
for key, res in allres.items():
    for x in res:
        d = abs(x["corner_min"] - x["csv_sampled"])
        P(f"  {key} {x['position']:2d}  {x['corner_min']!r}  {x['csv_sampled']!r}  {d:.3g}  {d / abs(x['csv_sampled']):.3g}"
          f"  {x['below'][.01]}/{x['csv_below'][.01]}  {x['below'][.001]}/{x['csv_below'][.001]}")
for key, res in allres.items():
    S = SETS[key]
    n = len(res)
    diffs = [abs(x["corner_min"] - x["csv_sampled"]) for x in res]
    rels = [d / abs(x["csv_sampled"]) for d, x in zip(diffs, res)]
    P(f"\n== {S['name']}: n {n} fields; seconds per field median {np.median([x['seconds'] for x in res]):.1f} (max {max(x['seconds'] for x in res):.1f})")
    P("  field checks: status hx_certified", sum(x["status"] == "hx_certified" for x in res), "/", n,
      "| six faces exactly zero", sum(x["faces_zero"] for x in res), "/", n,
      "| sha256(u) equals the stored state_hash", sum(x["state_hash_equal"] for x in res), "/", n,
      "| spacing_xyz", sorted({x["spacing"] for x in res}), "| cells per field", sorted({x["n_cells"] for x in res}))
    P(f"  corner minimum vs csv sampled_min_J: largest |difference| {max(diffs):.3g}; largest relative difference {max(rels):.3g}")
    P(f"  corner minimum vs csv sampled_min_J_closed27: largest |difference| {max(abs(x['corner_min'] - x['csv_closed27']) for x in res):.3g}")
    P("  csv point_of_sampled_min:", {k: sum(x["csv_point"] == k for x in res) for k in sorted({x["csv_point"] for x in res})})
    Q("corner minimum here", [x["corner_min"] for x in res])
    Q("sampled minimum in the csv", [x["csv_sampled"] for x in res])
    P("  fields with corner minimum < 1e-3:", sum(x["corner_min"] < 1e-3 for x in res), "/", n,
      "| < 1e-4:", sum(x["corner_min"] < 1e-4 for x in res), "| <= 0:", sum(x["corner_min"] <= 0 for x in res))
    for t in THRESHOLDS:
        P(f"  cells with corner J < {t}: corner-only count <= csv union count in", sum(x["below"][t] <= x["csv_below"][t] for x in res), "/", n,
          "| equal in", sum(x["below"][t] == x["csv_below"][t] for x in res), "/", n)
        Q(f"cells with corner J < {t} (count per field)", [x["below"][t] for x in res])
if len(allres) == 2:
    allx = allres["oasis"] + allres["lpba"]
    P(f"\n== both sets: n {len(allx)}; fields with corner minimum < 1e-3: {sum(x['corner_min'] < 1e-3 for x in allx)};"
      f" largest |difference| {max(abs(x['corner_min'] - x['csv_sampled']) for x in allx):.3g}")
mods = sorted(m for m in sys.modules if m.split(".")[0] in ("bayesreg", "jd_curl", "numba", "torch"))
P(f"\ntotal seconds {time.perf_counter() - t_all:.0f}; peak RSS GiB {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f}")
P("repository / numba / torch modules imported:", mods if mods else "none")
P("numpy", np.__version__, "| python", sys.version.split()[0])
P("script sha256", sha256_file(os.path.abspath(__file__)))

text = "\n".join(out) + "\n"
if any(t in text for t in tokens if len(t) >= 3) or re.search(r"OAS[0-9_]|OASIS_|LPBA40_|(?<![A-Za-z])S\d\d(?!\d)|(?<!\d)\d{4}[-_]\d{4}(?!\d)", text):
    sys.exit("guard: an identifier in the output; nothing written")
name = "independent_cornerj.out" if limit is None else f"independent_cornerj_limit{limit}.out"
with open(os.path.join(target, name), "w") as fh:
    fh.write(text)
print("written:", os.path.join(target, name), "| lines:", len(out))
