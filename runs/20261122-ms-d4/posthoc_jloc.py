"""MS-D4, package B, amendment 2: where the cells with J < .5 are (post hoc, descriptive, outside both preregistrations).

Reads, per pair of the two test sets: the method's output field (output/final.npz, key u), the structure-layer
PREDICTION of the fixed subject (HI: pred/<fixed>/probs36.npz, key p; argmax over the 36 classes -- a prediction
of the method's own segmenter, not a truth label), and JSON records (decision.json, the pair's evaluation record,
output/line.json).  No truth label, no image, no moving-side file.  CPU only; one field in memory at a time.

J is the evaluation's cell-centre J (bayesreg/eval/metrics.py cell_jacobian_centres, copied below word for word);
a cell takes the class of its lower-corner node (the evaluation's convention for its brain mask, metrics.py
brain_jacobian_stats).  The evaluation's brain mask is not read here (OASIS: the truth seg4 > 0; LPBA40: the fixed
image > 0): the mask of this script is "predicted label > 0", and the two are compared through the recorded numbers.

    python -I -B posthoc_jloc.py [--out DIR] [--limit N]
"""
import csv
import hashlib
import io
import json
import os
import re
import sys
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/home/rmcfz/seq_vp_tag_results"
SETS = {
    "oasis": dict(name="OASIS hold-out", n=50, decision=RES + "/holdout_B_v1/eval/decision.json", pin="a0dbdcf9",
                  rec=RES + "/holdout_B_v1/eval/v1/pairs/{}.json", cell="/mnt/d/seq_vp_tag_holdout_B_v1/method/{}",
                  pred=RES + "/seg/unet35_v4_ens_holdout_B_v1/pred/{}/probs36.npz"),
    "lpba": dict(name="LPBA40 second test set", n=17, decision=RES + "/lpba40_test_v1/eval/decision.json", pin="a0596c5f",
                 rec=RES + "/lpba40_test_v1/eval/run_1/pairs/{}.json", cell=RES + "/lpba40_test_v1/work/method/{}",
                 pred=RES + "/seg/unet35_v4_ens_lpba40_test_v1/pred/{}/probs36.npz"),
}
# seg35 classes of the segmenter (configs/bayesreg/structure_groups_v1.json label_table), grouped here
CLASSES = [("cerebral cortex", (2, 21)), ("cerebral white matter", (1, 20)),
           ("subcortical grey (thalamus, caudate, putamen, pallidum, hippocampus, amygdala, accumbens, ventral DC)", (7, 8, 9, 10, 14, 15, 16, 17, 26, 27, 28, 29, 30, 31, 32, 33)),
           ("ventricles (lateral, inferior lateral, 3rd, 4th)", (3, 4, 11, 12, 22, 23)), ("choroid plexus and vessel", (18, 19, 34, 35)),
           ("cerebellum (cortex and white matter)", (5, 6, 24, 25)), ("brain stem", (13,))]
assert sorted(i for _, ids in CLASSES for i in ids) == list(range(1, 36))
BINS = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 1e9)]      # mm from the nearest node outside the predicted mask, (lo, hi]


def cell_jacobian_centres(u, spacing):
    """det(I + grad u) at every cell centre; u is a (nz, ny, nx, 3) node field, components xyz.  (metrics.py:145-166)"""
    u = np.asarray(u, dtype=np.float64)
    hx, hy, hz = spacing

    def d(axis, comp):
        a = u[..., comp]
        if axis == 0:
            diff = (a[:, :, 1:]-a[:, :, :-1])/hx
            return .25*(diff[:-1, :-1]+diff[1:, :-1]+diff[:-1, 1:]+diff[1:, 1:])
        if axis == 1:
            diff = (a[:, 1:, :]-a[:, :-1, :])/hy
            return .25*(diff[:-1, :, :-1]+diff[1:, :, :-1]+diff[:-1, :, 1:]+diff[1:, :, 1:])
        diff = (a[1:, :, :]-a[:-1, :, :])/hz
        return .25*(diff[:, :-1, :-1]+diff[:, 1:, :-1]+diff[:, :-1, 1:]+diff[:, 1:, 1:])
    with np.errstate(over="raise", invalid="raise"):
        F = [[(1. if r == j else 0.)+d(j, r) for j in range(3)] for r in range(3)]
        J = (F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1]) - F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
             + F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]))
    if not np.all(np.isfinite(J)):
        raise FloatingPointError("non-finite cell-centre Jacobian determinant")
    return J


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


def q3(xs, f="{:.4g}"):
    xs = np.asarray(xs, dtype=float)
    return f"{f.format(np.median(xs))} ({f.format(xs.min())}..{f.format(xs.max())})"


limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
tokens, rows = set(), [["set", "group", "n_pairs", "share_of_J_lt_half_median", "share_min", "share_max", "rate_J_lt_half_median", "rate_min", "rate_max",
                        "enrichment_median", "enrichment_min", "enrichment_max", "share_of_cells_median", "rate_J_gt_2_median"]]
t_all = time.perf_counter()
for key, S in SETS.items():
    if sha(S["decision"], 8) != S["pin"]:
        sys.exit("pin mismatch")
    pairs = [e["pair"] for e in sorted(J_(S["decision"])["positions"], key=lambda e: e["position"])]
    assert len(pairs) == S["n"]
    if key == "lpba":
        assert not any(re.search(r"S(01|02|37|38|39|40)(?!\d)", p) for p in pairs), "an LPBA pair outside the test list"
    if limit:
        pairs = pairs[:limit]
    res, secs = [], []
    for p in pairs:
        t0 = time.perf_counter()
        r = J_(S["rec"].format(p))
        assert r["pair"] == p and r["complete"]
        tokens |= {p, str(r["fixed"]), str(r["moving"])}
        cell = S["cell"].format(p)
        ln = J_(cell + "/output/line.json")
        assert ln["line_status"] == "hx_certified" and sha(cell + "/output/final.npz") == ln["output"]["final"]["sha256"]
        with np.load(cell + "/output/final.npz") as z:
            u = z["u"]
        assert list(u.shape[:3]) == r["grid"]["shape_zyx"] and u.dtype == np.float64
        Jc = cell_jacobian_centres(u, r["grid"]["spacing_xyz"])
        del u
        m = r["arms"]["method"]["metrics"]
        chk = dict(min_J=float(Jc.min()) == m["min_J_centre"], nonpos=int((Jc <= 0).sum()) == m["cell_centre_J_nonpositive"], n_cells=Jc.size == m["n_cells"])
        with np.load(S["pred"].format(r["fixed"])) as z:
            prob = z["p"]
        assert prob.shape == (36,) + tuple(r["grid"]["shape_zyx"])
        lab = prob.argmax(axis=0).astype(np.uint8)
        del prob
        inside = lab > 0
        dist = distance_transform_edt(inside, sampling=r["grid"]["spacing_xyz"][::-1]).astype(np.float32)[:-1, :-1, :-1]
        lc = lab[:-1, :-1, :-1]
        del lab
        lt, gt = Jc < .5, Jc > 2.
        brain = lc > 0
        n_lt_all, n_lt_brain, n_brain = int(lt.sum()), int((lt & brain).sum()), int(brain.sum())
        rate_brain = n_lt_brain / n_brain
        cnt = np.bincount(lc.ravel(), minlength=36)
        cnt_lt = np.bincount(lc.ravel(), weights=lt.ravel(), minlength=36)
        cnt_gt = np.bincount(lc.ravel(), weights=gt.ravel(), minlength=36)
        one = dict(chk=chk, pct_pred=100 * rate_brain, pct_rec=m["pct_J_lt_half_brain"], n_brain_pred=n_brain, n_brain_rec=m["n_brain_cells"],
                   outside_share=(n_lt_all - n_lt_brain) / n_lt_all, outside_rate=(n_lt_all - n_lt_brain) / (Jc.size - n_brain), rate_all=n_lt_all / Jc.size, groups={}, bins={},
                   mean_dist_lt=float(dist[lt & brain].mean()), mean_dist_brain=float(dist[brain].mean()))
        for name, ids in CLASSES:
            ids = list(ids)
            n, a, b = cnt[ids].sum(), cnt_lt[ids].sum(), cnt_gt[ids].sum()
            one["groups"][name] = dict(share=a / n_lt_brain, rate=a / n, enrich=(a / n) / rate_brain, cells=n / n_brain, gt2=b / n)
        for lo, hi in BINS:
            sel = brain & (dist > lo) & (dist <= hi)
            n, a = int(sel.sum()), int((sel & lt).sum())
            one["bins"][(lo, hi)] = dict(share=a / n_lt_brain, rate=a / n if n else float("nan"), enrich=(a / n) / rate_brain if n else float("nan"), cells=n / n_brain)
        res.append(one)
        secs.append(time.perf_counter() - t0)
        del Jc, lc, dist, lt, gt, brain
    n = len(res)
    P(f"\n== {S['name']}: n {n} pairs; seconds per pair median {np.median(secs):.1f} (max {max(secs):.1f}), total {sum(secs):.0f}")
    P("  definition check against the evaluation records (whole grid, exact): min_J_centre equal", sum(x["chk"]["min_J"] for x in res), "/", n,
      "| cell_centre_J_nonpositive equal", sum(x["chk"]["nonpos"] for x in res), "/", n, "| n_cells equal", sum(x["chk"]["n_cells"] for x in res), "/", n)
    d = np.array([x["pct_pred"] - x["pct_rec"] for x in res])
    P(f"  % of brain cells with J < .5: this script (mask = predicted label > 0) mean {np.mean([x['pct_pred'] for x in res]):.2f}, median (range) {q3([x['pct_pred'] for x in res])};",
      f"evaluation record (its own mask) mean {np.mean([x['pct_rec'] for x in res]):.2f}, median (range) {q3([x['pct_rec'] for x in res])}")
    P(f"    per pair, this script minus the record (percentage points): median (range) {q3(d, '{:+.3f}')}; mean |difference| {np.abs(d).mean():.3f}; Pearson r {np.corrcoef([x['pct_pred'] for x in res], [x['pct_rec'] for x in res])[0, 1]:.4f}" if n > 2 else "")
    P(f"    brain cells: predicted mask / recorded mask, ratio per pair median (range) {q3([x['n_brain_pred'] / x['n_brain_rec'] for x in res])}")
    P(f"  cells with J < .5 outside the predicted mask (background, sulcal CSF included): share of all J < .5 cells of the grid, median (range) {q3([x['outside_share'] for x in res])};",
      f"rate among the cells outside {q3([x['outside_rate'] for x in res])}; rate over the whole grid {q3([x['rate_all'] for x in res])}")
    P("  by predicted class (cells inside the predicted mask; share = of the J < .5 cells inside the mask; rate = J < .5 within the class; enrichment = rate / rate inside the whole mask):")
    P("    class | share of J < .5, median (range) | rate within the class | enrichment | share of the mask's cells (median) | rate of J > 2 within the class (median)")
    for name, _ in CLASSES:
        gsh, grt, gen = ([x["groups"][name][k] for x in res] for k in ("share", "rate", "enrich"))
        cells, gt2 = np.median([x["groups"][name]["cells"] for x in res]), np.median([x["groups"][name]["gt2"] for x in res])
        P(f"    {name} | {q3(gsh)} | {q3(grt)} | {q3(gen)} | {cells:.4g} | {gt2:.4g}")
        rows.append([key, name, n] + [f"{f(v):.5f}" for v in (gsh, grt, gen) for f in (np.median, np.min, np.max)] + [f"{cells:.5f}", f"{gt2:.5f}"])
    P("    pairs in which the class has the highest rate:", {name.split(" (")[0]: sum(max(x["groups"], key=lambda g: x["groups"][g]["rate"]) == name for x in res) for name, _ in CLASSES})
    P("    pairs in which the class holds the largest share:", {name.split(" (")[0]: sum(max(x["groups"], key=lambda g: x["groups"][g]["share"]) == name for x in res) for name, _ in CLASSES})
    P("  by distance from the boundary of the predicted mask (mm, Euclidean, from the cell's lower-corner node to the nearest node outside the mask):")
    P("    bin (lo, hi] | share of J < .5, median (range) | rate within the bin | enrichment | share of the mask's cells (median)")
    for lo, hi in BINS:
        b = [x["bins"][(lo, hi)] for x in res]
        label = f"({lo}, {hi}]" if hi < 1e8 else f"> {lo}"
        P(f"    {label} | {q3([x['share'] for x in b])} | {q3([x['rate'] for x in b])} | {q3([x['enrich'] for x in b])} | {np.median([x['cells'] for x in b]):.4g}")
        rows.append([key, "distance " + label + " mm", n] + [f"{f([x[k] for x in b]):.5f}" for k in ("share", "rate", "enrich") for f in (np.median, np.min, np.max)]
                    + [f"{np.median([x['cells'] for x in b]):.5f}", ""])
    P(f"    mean distance of the J < .5 cells {q3([x['mean_dist_lt'] for x in res])} mm; of all cells of the mask {q3([x['mean_dist_brain'] for x in res])} mm")
P(f"\ntotal seconds {time.perf_counter() - t_all:.0f}; peak RSS GiB {__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss / 2**20:.2f}")
P("script sha256", sha(os.path.abspath(__file__)))

b = io.StringIO()
csv.writer(b, lineterminator="\n").writerows(rows)
files = {"posthoc_jloc.out": "\n".join(x for x in out if x != "") + "\n", "jloc_groups.csv": b.getvalue()}
tokens = {t for t in tokens if len(t) >= 3}
for name, text in files.items():
    if any(t in text for t in tokens) or re.search(r"OAS[0-9_]|OASIS_|(?<![A-Za-z])S\d\d(?!\d)|(?<!\d)\d{4}[-_]\d{4}(?!\d)", text):
        sys.exit(f"guard: an identifier in {name}; nothing written")
target = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else HERE
for name, text in files.items():
    with open(os.path.join(target, name), "w") as fh:
        fh.write(text)
print("written:", ", ".join(sorted(files)), "| lines:", len(out))
