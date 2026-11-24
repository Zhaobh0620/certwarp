"""MS-D4, package B, amendment 3, item 2: the smallest SAMPLED J of the method's 67 outputs (post hoc, descriptive).

Per pair: the method's output field (output/final.npz, key u) and JSON records (decision.json, the pair's evaluation
record, output/line.json, output/cert.json, repair/repair.json).  No truth label, no image, no segmentation.  CPU only,
one field in memory at a time.

J is the Q1 map's det(I + grad u) at the Gauss27 points and at the closed27 points {0, 1/2, 1}^3 (corners, edge
midpoints, face centres, body centre) of every cell, computed by the repository's own kernels:
bayesreg.geom.q1.tile_geometry(u, ops, tile).J with ops = q1.gauss27(grid) and q1.closed27(grid), over
jd_curl.three_d.cell_q1.cell_tiles(grid, TILE).  At a point shared by several cells each cell's own value is taken.
The sampled minimum is an upper bound of the true minimum of J over the closed cells; the certificate's
min_jd_lower_bound is a lower bound of it.

    python -I -B posthoc_minj.py [--out DIR] [--limit N] [--cache DIR]
"""
import csv
import hashlib
import io
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import numpy as np                                             # noqa: E402

from jd_curl.three_d.cell_q1 import cell_tiles                 # noqa: E402
from jd_curl.three_d.grid import make_grid_3d                  # noqa: E402
from bayesreg.geom import q1                                   # noqa: E402

RES = "/home/rmcfz/seq_vp_tag_results"
SETS = {
    "oasis": dict(name="OASIS hold-out", n=50, decision=RES + "/holdout_B_v1/eval/decision.json", pin="a0dbdcf9",
                  rec=RES + "/holdout_B_v1/eval/v1/pairs/{}.json", cell="/mnt/d/seq_vp_tag_holdout_B_v1/method/{}"),
    "lpba": dict(name="LPBA40 second test set", n=17, decision=RES + "/lpba40_test_v1/eval/decision.json", pin="a0596c5f",
                 rec=RES + "/lpba40_test_v1/eval/run_1/pairs/{}.json", cell=RES + "/lpba40_test_v1/work/method/{}"),
}
TILE = 32768
THRESHOLDS = (.05, .01, .001)
KINDS = {0: "corner", 1: "edge midpoint", 2: "face centre", 3: "body centre"}


def sha(path, n=64):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def J_(path):
    with open(path) as f:
        return json.load(f)


out = []


def P(*a):
    out.append(" ".join(str(x) for x in a))


def g(x):
    return f"{x:.4g}"


def Q(name, xs):
    xs = np.asarray(xs, dtype=float)
    q = np.percentile(xs, [0, 25, 50, 75, 100])
    P(f"  {name}: n {len(xs)}; min {g(q[0])}; q25 {g(q[1])}; median {g(q[2])}; q75 {g(q[3])}; max {g(q[4])}")


def one_pair(S, p):
    r = J_(S["rec"].format(p))
    assert r["pair"] == p and r["complete"]
    cell = S["cell"].format(p)
    ln, cert, rep = J_(cell + "/output/line.json"), J_(cell + "/output/cert.json"), J_(cell + "/repair/repair.json")
    assert ln["line_status"] == "hx_certified" and cert["status"] == "certified_feasible" and rep["pair"] == p
    assert sha(cell + "/output/final.npz") == ln["output"]["final"]["sha256"]
    nz, ny, nx = ln["grid"]["shape_zyx"]
    hx, hy, hz = ln["grid"]["spacing_xyz"]
    x0, y0, z0 = ln["grid"]["origin_xyz"]
    grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=hx, hy=hy, hz=hz, x0=x0, y0=y0, z0=z0, unit=ln["grid"]["unit"])
    assert grid.grid_id == ln["grid"]["grid_id"], "grid id"
    with np.load(cell + "/output/final.npz") as z:
        u = z["u"]
    u = q1.node_field(u, grid)              # also checks: finite, shape, exactly zero on the six faces
    sets = (("gauss27", q1.gauss27(grid)), ("closed27", q1.closed27(grid)))
    kind = [int(sum(c == .5 for c in pt)) for pt in sets[1][1].points]
    assert sorted(kind).count(0) == 8 and kind[13] == 3
    best = {name: (np.inf, -1) for name, _ in sets}
    centre, below = np.inf, {t: 0 for t in THRESHOLDS}
    for tile in cell_tiles(grid, TILE):
        cell_min = None
        for name, ops in sets:
            Jt = q1.tile_geometry(u, ops, tile=tile).J
            k = int(Jt.argmin())
            if Jt.flat[k] < best[name][0]:
                best[name] = (float(Jt.flat[k]), k % Jt.shape[1])
            m = Jt.min(axis=1)
            cell_min = m if cell_min is None else np.minimum(cell_min, m)
            if name == "closed27":
                centre = min(centre, float(Jt[:, 13].min()))
        for t in THRESHOLDS:
            below[t] += int((cell_min < t).sum())
    m = r["arms"]["method"]["metrics"]
    gmin, cmin = best["gauss27"][0], best["closed27"][0]
    return dict(lower=cert["min_jd_lower_bound"], gauss=gmin, closed=cmin, sampled=min(gmin, cmin),
                where="Gauss27 point" if gmin < cmin else KINDS[kind[best["closed27"][1]]], centre_here=centre, centre_rec=m["min_J_centre"],
                n_cells=grid.n_cells, below={str(t): below[t] for t in THRESHOLDS},
                repair_min_j=rep["rounds"][-1]["inner"]["min_j"], eps=rep["eps"], pair_hash=hashlib.sha256(p.encode()).hexdigest())


limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
cache = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else None
tokens, rows = set(), [["set", "position", "certificate_lower_bound", "sampled_min_J_gauss27", "sampled_min_J_closed27", "sampled_min_J", "point_of_sampled_min",
                        "min_J_cell_centre_recorded", "cells_sampled_min_lt_0.05", "cells_sampled_min_lt_0.01", "cells_sampled_min_lt_0.001"]]
t_all = time.perf_counter()
for key, S in SETS.items():
    if sha(S["decision"], 8) != S["pin"]:
        sys.exit("pin mismatch")
    pairs = [e["pair"] for e in sorted(J_(S["decision"])["positions"], key=lambda e: e["position"])]
    assert len(pairs) == S["n"]
    if key == "lpba":
        assert not any(re.search(r"S(01|02|37|38|39|40)(?!\d)", p) for p in pairs), "an LPBA pair outside the test list"
    res, secs = [], []
    for i, p in enumerate(pairs[:limit] if limit else pairs):
        t0 = time.perf_counter()
        r0 = J_(S["rec"].format(p))
        tokens |= {p, str(r0["fixed"]), str(r0["moving"])}
        f = os.path.join(cache, f"{key}_p{i}.json") if cache else None      # a scratch file by position (no identifier in it), never committed
        if f and os.path.isfile(f):
            x = J_(f)
            assert x["pair_hash"] == hashlib.sha256(p.encode()).hexdigest()
        else:
            x = one_pair(S, p)
            secs.append(time.perf_counter() - t0)
            if f:
                with open(f + ".tmp", "w") as fh:
                    json.dump(x, fh)
                os.replace(f + ".tmp", f)
        res.append(x)
        rows.append([key, i, repr(x["lower"]), repr(x["gauss"]), repr(x["closed"]), repr(x["sampled"]), x["where"], repr(x["centre_rec"])] + [x["below"][str(t)] for t in THRESHOLDS])
    n = len(res)
    P(f"\n== {S['name']}: n {n} pairs" + (f"; seconds per pair computed here median {np.median(secs):.0f} (max {max(secs):.0f}), {len(secs)} computed in this run" if secs else "; all from the scratch cache of an earlier run of this script"))
    P("  checks: certificate lower bound <= sampled minimum:", sum(x["lower"] <= x["sampled"] for x in res), "/", n,
      "| sampled minimum <= recorded min_J_centre:", sum(x["sampled"] <= x["centre_rec"] for x in res), "/", n,
      "| body-centre minimum computed here vs recorded min_J_centre, largest |difference|:", g(max(abs(x["centre_here"] - x["centre_rec"]) for x in res)))
    Q("certificate lower bound (cert.json min_jd_lower_bound)", [x["lower"] for x in res])
    Q("sampled minimum of J over Gauss27 U closed27 (54 distinct points per cell)", [x["sampled"] for x in res])
    Q("  over the Gauss27 points only", [x["gauss"] for x in res])
    Q("  over the closed27 points only", [x["closed"] for x in res])
    Q("smallest cell-centre J (evaluation record min_J_centre)", [x["centre_rec"] for x in res])
    Q("sampled minimum / certificate lower bound", [x["sampled"] / x["lower"] for x in res])
    Q("sampled minimum - certificate lower bound (width of the bracket of the true minimum)", [x["sampled"] - x["lower"] for x in res])
    P("  point of the sampled minimum:", {k: sum(x["where"] == k for x in res) for k in sorted(set(x["where"] for x in res))})
    P("  repair target eps:", sorted(set(x["eps"] for x in res)), "| pairs with sampled minimum < eps:", sum(x["sampled"] < x["eps"] for x in res), "/", n,
      "| < .04:", sum(x["sampled"] < .04 for x in res), "| < .01:", sum(x["sampled"] < .01 for x in res), "| <= 0:", sum(x["sampled"] <= 0 for x in res))
    P("  last repair round's inner.min_j (53 points + added points, over the cells of the repair mask only) equals the sampled minimum to 1e-9:",
      sum(abs(x["repair_min_j"] - x["sampled"]) < 1e-9 for x in res), "/", n, "| is larger:", sum(x["repair_min_j"] > x["sampled"] + 1e-9 for x in res),
      "| is smaller:", sum(x["repair_min_j"] < x["sampled"] - 1e-9 for x in res))
    Q("last repair round's inner.min_j", [x["repair_min_j"] for x in res])
    for t in THRESHOLDS:
        Q(f"cells whose sampled minimum is < {t} (count per pair, of {res[0]['n_cells']} cells)", [x["below"][str(t)] for x in res])
P(f"\ntotal seconds {time.perf_counter() - t_all:.0f}; peak RSS GiB {__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss / 2**20:.2f}; tile {TILE} cells")
P("numba imported:", any(m == "numba" or m.startswith("numba.") for m in sys.modules))
P("script sha256", sha(os.path.abspath(__file__)))

b = io.StringIO()
csv.writer(b, lineterminator="\n").writerows(rows)
files = {"posthoc_minj.out": "\n".join(out) + "\n", "minj_by_position.csv": b.getvalue()}
tokens = {t for t in tokens if len(t) >= 3}
for name, text in files.items():
    if any(t in text for t in tokens) or re.search(r"OAS[0-9_]|OASIS_|(?<![A-Za-z])S\d\d(?!\d)|(?<!\d)\d{4}[-_]\d{4}(?!\d)", text):
        sys.exit(f"guard: an identifier in {name}; nothing written")
target = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else HERE
for name, text in files.items():
    with open(os.path.join(target, name), "w") as fh:
        fh.write(text)
print("written:", ", ".join(sorted(files)), "| lines:", len(out))
