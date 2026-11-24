"""MS-D4, package B: post hoc descriptive readouts for the manuscript (outside both preregistrations).

Reads only JSON run records and evaluation records of the two test sets (no image, no label, no field array, no
truth file), the committed summary csv of the hold-out result, and the development-set HS evaluation records.
Pairs are taken from the two decision.json files (position order); nothing is globbed under the hold-out roots
or /mnt/d.  No subject or pair identifier is printed: every output passes a guard before it is written.

    python -I -B posthoc.py            # writes posthoc.out, scatter_input.csv, lpba_per_label_delta.csv,
                                       # timing_main_family.csv next to this file
    python -I -B posthoc.py --out DIR  # the same into DIR (a trial run)
"""
import csv
import glob
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = "/home/rmcfz/seq_vp_tag_results"
H = RES + "/holdout_B_v1"                       # OASIS hold-out: records
HD = "/mnt/d/seq_vp_tag_holdout_B_v1"           # OASIS hold-out: method cells after step 5' (JSON records read only)
L = RES + "/lpba40_test_v1"                     # LPBA40 second test set
HI = {"oasis": RES + "/seg/unet35_v4_ens_holdout_B_v1/hi_record.json",
      "lpba": RES + "/seg/unet35_v4_ens_lpba40_test_v1/hi_record.json"}
DEV = RES + "/hs_v1/runs"                       # development set (19 pairs), HS evaluation records
PINS = {H + "/eval/decision.json": "a0dbdcf9", L + "/eval/decision.json": "a0596c5f"}   # first 8 hex, as the result documents quote them
MAIN = ["configc", "ants_s", "fireants_greedy", "fireants_syn", "convexadam_v4seg"]
XB = ["ants_s", "fireants_greedy", "fireants_syn", "convexadam_v4seg"]
LIDS = (list(range(21, 35)) + list(range(41, 51)) + list(range(61, 69)) + list(range(81, 93)) + [101, 102, 121, 122]
        + list(range(161, 167)) + [181, 182])
SEGMENTS = [("21-34", range(21, 35)), ("41-50", range(41, 51)), ("61-68", range(61, 69)), ("81-92", range(81, 93)),
            ("101-102", (101, 102)), ("121-122", (121, 122)), ("161-166", range(161, 167)), ("181-182", (181, 182))]
GROUPS = [("ids 21..92 (44 labels)", [i for i in LIDS if i <= 92]), ("ids 101..166 (10 labels)", [i for i in LIDS if 101 <= i <= 166]),
          ("ids 181,182 (2 labels)", [181, 182])]
# LPBA40 arm table as printed in runs/20261122-lpba-result/result.md section 2 and 2.1 (lpba54, lpba56, official SDlogJ, BL1 passed of 17)
LPBA_DOC = {"identity": (.5303, .5398, .0000, None), "method": (.6747, .6804, .0812, 17), "configc": (.7022, .7074, .0558, 17),
            "ants_s": (.7237, .7281, .0448, 2), "fireants_greedy": (.6923, .6971, .0658, 17), "fireants_syn": (.6664, .6720, .0533, 17),
            "convexadam_v4seg": (.6749, .6806, .0432, 0)}


def sha(path, n=64):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def J(path):
    with open(path) as f:
        return json.load(f)


out = []


def P(*a):
    out.append(" ".join(str(x) for x in a))


def g(x):
    return "None" if x is None else f"{x:.4g}"


def Q(name, xs, fmt=g):
    xs = np.asarray([x for x in xs if x is not None], dtype=float)
    q = np.percentile(xs, [0, 25, 50, 75, 100])
    P(f"  {name}: n {len(xs)}; min {fmt(q[0])}; q25 {fmt(q[1])}; median {fmt(q[2])}; q75 {fmt(q[3])}; max {fmt(q[4])}; mean {fmt(xs.mean())}")
    return q


def counts(xs):
    return {str(k): xs.count(k) for k in sorted(set(xs), key=str)}


for p, want in PINS.items():
    if sha(p, 8) != want:
        sys.exit("pin mismatch: " + os.path.basename(os.path.dirname(os.path.dirname(p))))
P("pins ok: both decision.json (first 8 hex as quoted in the two result documents)")

dec = {"oasis": J(H + "/eval/decision.json"), "lpba": J(L + "/eval/decision.json")}
pairs = {k: [e["pair"] for e in sorted(d["positions"], key=lambda e: e["position"])] for k, d in dec.items()}
assert len(pairs["oasis"]) == 50 and len(pairs["lpba"]) == 17
assert not any(re.search(r"S(01|02|37|38|39|40)(?!\d)", p) for p in pairs["lpba"]), "an LPBA pair outside the test list (feasibility or H2 subject)"
recs = {"oasis": [J(f"{H}/eval/v1/pairs/{p}.json") for p in pairs["oasis"]], "lpba": [J(f"{L}/eval/run_1/pairs/{p}.json") for p in pairs["lpba"]]}
assert all(r["pair"] == p and r["complete"] for k in recs for r, p in zip(recs[k], pairs[k]))
CELL = {"oasis": lambda p: f"{HD}/method/{p}", "lpba": lambda p: f"{L}/work/method/{p}"}
NAME = {"oasis": "OASIS hold-out (50 pairs)", "lpba": "LPBA40 second test set (17 pairs)"}

# ---- item 1: before the repair ------------------------------------------------------------------------------------------
P("\n== item 1: the pre-optimisation output before the repair (repair/repair.json of every method cell; output/line.json)")
for k in ("oasis", "lpba"):
    rp = [J(CELL[k](p) + "/repair/repair.json") for p in pairs[k]]
    ln = [J(CELL[k](p) + "/output/line.json") for p in pairs[k]]
    assert all(r["pair"] == p for r, p in zip(rp, pairs[k]))
    P(f"-- {NAME[k]}: n {len(rp)}")
    P("  repair status", counts([r["status"] for r in rp]), "| reason", counts([r["reason"] for r in rp]), "| passed", sum(bool(r["passed"]) for r in rp),
      "| wall_hit", sum(bool(r["wall_hit"]) for r in rp), "| line_status", counts([x["line_status"] for x in ln]))
    ini = [r["initial_certificate"] for r in rp]
    ok0 = [i["status"] == "certified_feasible" and i["min_jd_lower_bound"] is not None and i["min_jd_lower_bound"] > 0 and i["n_infeasible"] == 0 and i["n_uncertain"] == 0 for i in ini]
    P("  initial certificate (configuration C, on the tapered T88 field, before any repair): status", counts([i["status"] for i in ini]),
      "| passing (rule of C22.3-6)", sum(ok0), "| not passing", len(ok0) - sum(ok0))
    shape = [x["grid"]["shape_zyx"] for x in ln]
    ncell = [(s[0] - 1) * (s[1] - 1) * (s[2] - 1) for s in shape]
    nnode = [s[0] * s[1] * s[2] for s in shape]
    P("  grid cells per pair", sorted(set(ncell)), "| nodes", sorted(set(nnode)), "| initial n_cells_checked == all cells:", sum(i["n_cells_checked"] == c for i, c in zip(ini, ncell)), "/", len(ini))
    Q("initial n_infeasible (cells with a proven J <= 0 point)", [i["n_infeasible"] for i in ini])
    Q("initial n_uncertain (cells undecided)", [i["n_uncertain"] for i in ini])
    Q("initial n_infeasible + n_uncertain (cells not proven)", [i["n_infeasible"] + i["n_uncertain"] for i in ini])
    Q("  as a fraction of all cells", [(i["n_infeasible"] + i["n_uncertain"]) / c for i, c in zip(ini, ncell)])
    P("  pairs with initial n_uncertain > 0:", sum(i["n_uncertain"] > 0 for i in ini), "| pairs with initial n_infeasible > 0:", sum(i["n_infeasible"] > 0 for i in ini))
    Q("initial min_jd_lower_bound", [i["min_jd_lower_bound"] for i in ini])
    Q("n_located (cells the two fast proofs of locate_cells do not prove; the repair's seed, a superset, not a verdict)", [r["n_located"] for r in rp])
    nr = [len(r["rounds"]) for r in rp]
    P("  repair rounds", counts(nr), "| line.json repair.rounds the same:", sum(a == x["repair"]["rounds"] for a, x in zip(nr, ln)), "/", len(nr))
    last = [r["rounds"][-1] for r in rp if r["rounds"]]
    Q("free nodes in the last round (n_free; every other node is bitwise the input)", [x["n_free"] for x in last])
    Q("  as a fraction of all nodes", [x["n_free"] / n for x, n in zip(last, nnode)])
    P("  attempt.n_free_nodes == last n_free:", sum(r["attempt"]["n_free_nodes"] == r["rounds"][-1]["n_free"] for r in rp if r["rounds"]), "/", len(last))
    Q("nodes moved in the last round (inner.n_moved: free nodes with |v - u| > 0)", [x["inner"]["n_moved"] for x in last])
    Q("cells touched in the last round (n_cells)", [x["n_cells"] for x in last])
    Q("largest node displacement change of the output against the repair input, mm (last round inner.max_move_mm)", [x["inner"]["max_move_mm"] for x in last])
    Q("  largest over all rounds of a pair, mm", [max(x["inner"]["max_move_mm"] for x in r["rounds"]) for r in rp if r["rounds"]])
    P("  inner stop reasons over all rounds", counts([x["inner"]["stop"] for r in rp for x in r["rounds"]]),
      "| inner iterations per round: median", g(np.median([x["inner"]["iterations"] for r in rp for x in r["rounds"]])))
    Q("seconds_repair (first locate + repair_field: its certificates included; warm-up excluded)", [r["seconds_repair"] for r in rp])
    Q("  of which seconds_first_locate", [r["seconds_first_locate"] for r in rp])
    Q("  of which L-BFGS solves (sum of rounds' seconds_solve)", [sum(x["seconds_solve"] for x in r["rounds"]) for r in rp])
    Q("  of which certificates after rounds (sum of rounds' seconds_certificate)", [sum(x["seconds_certificate"] for x in r["rounds"]) for r in rp])
    Q("seconds_warmup (numba + device, not in seconds_repair)", [r["seconds_warmup"] for r in rp])
    Q("repair child seconds_total", [r["seconds_total"] for r in rp])
    Q("final certificate min_jd_lower_bound (repair.json certificate)", [r["certificate"]["min_jd_lower_bound"] for r in rp])
    for n in sorted(set(nr)):
        sel = [r for r in rp if len(r["rounds"]) == n]
        P(f"  pairs with {n} round(s): n {len(sel)}; median initial not-proven cells {g(np.median([r['initial_certificate']['n_infeasible'] + r['initial_certificate']['n_uncertain'] for r in sel]))};",
          f"median seconds_repair {g(np.median([r['seconds_repair'] for r in sel]))}; median last n_free {g(np.median([r['rounds'][-1]['n_free'] for r in sel]))}")

# development set: the change record computed from the two fields by the HS evaluation (already on record)
dv = [J(f) for arm in ("RSHx24_c600", "RSHx24_rest600") for f in sorted(glob.glob(f"{DEV}/{arm}/*/repair_metrics.json"))]
P(f"-- development set (HS, {len(dv)} pairs, single run; repair_metrics.json): the change record computed from the two fields")
Q("before.n_infeasible_at_1", [m["before"]["n_infeasible_at_1"] for m in dv])
Q("change.n_changed_nodes", [m["change"]["n_changed_nodes"] for m in dv])
Q("change.max_change_mm", [m["change"]["max_change_mm"] for m in dv])

# ---- item 2: LPBA40 per label --------------------------------------------------------------------------------------------
P("\n== item 2: LPBA40, method minus opponent per label and per ID group (pair records: arms.<arm>.per_label.lpba56_per_label)")
R = recs["lpba"]
ARMS_L = ["method"] + MAIN


def lab(arm, i):
    return np.array([r["arms"][arm]["per_label"]["lpba56_per_label"][str(i)] for r in R])


def ini_lab(i):
    return np.array([r["identity"]["per_label"]["dice_official_labels"][LIDS.index(i)] for r in R])


for a in ARMS_L:
    m56 = np.array([np.mean([r["arms"][a]["per_label"]["lpba56_per_label"][str(i)] for i in LIDS]) for r in R])
    m54 = np.array([np.mean([r["arms"][a]["per_label"]["lpba56_per_label"][str(i)] for i in LIDS if i < 181]) for r in R])
    rec56 = np.array([r["arms"][a]["metrics"]["lpba56"] for r in R])
    rec54 = np.array([r["arms"][a]["metrics"]["lpba54"] for r in R])
    P(f"  check {a}: mean of per-label = recorded lpba56 / lpba54 to {max(abs(m56 - rec56).max(), abs(m54 - rec54).max()):.1e}; lpba54 {rec54.mean():.4f} lpba56 {rec56.mean():.4f}")
rows = [["label_id", "identity_mean", "method_mean"] + [f"{x}_{k}" for k in MAIN for x in ("delta_mean", "wins_of_17")]]
for i in LIDS:
    m = lab("method", i)
    row = [i, f"{ini_lab(i).mean():.4f}", f"{m.mean():.4f}"]
    for k in MAIN:
        d = m - lab(k, i)
        row += [f"{d.mean():+.4f}", int((d > 0).sum())]
    rows.append(row)
P("  per label (56 rows) in lpba_per_label_delta.csv")
P("  opponent | lpba54 delta | lpba56 delta | labels with mean delta < 0 (of 56 / of 54) | most negative 5 labels (id: delta) | most positive 5 labels")
for k in MAIN:
    d = {i: (lab("method", i) - lab(k, i)).mean() for i in LIDS}
    d54 = np.mean([np.mean([r["arms"]["method"]["metrics"]["lpba54"] - r["arms"][k]["metrics"]["lpba54"]]) for r in R])
    d56 = np.mean([r["arms"]["method"]["metrics"]["lpba56"] - r["arms"][k]["metrics"]["lpba56"] for r in R])
    order = sorted(LIDS, key=lambda i: d[i])
    P(f"  {k} | {d54:+.4f} | {d56:+.4f} | {sum(v < 0 for v in d.values())} / {sum(d[i] < 0 for i in LIDS if i < 181)} |",
      ", ".join(f"{i}: {d[i]:+.3f}" for i in order[:5]), "|", ", ".join(f"{i}: {d[i]:+.3f}" for i in order[-5:][::-1]))
P("  ID group | n labels | identity | method | " + " | ".join(f"method - {k} (pairs won of 17)" for k in MAIN))
for name, ids in SEGMENTS + GROUPS:
    ids = list(ids)
    mm = np.mean([lab("method", i) for i in ids], axis=0)
    cells = []
    for k in MAIN:
        d = mm - np.mean([lab(k, i) for i in ids], axis=0)
        cells.append(f"{d.mean():+.4f} ({int((d > 0).sum())})")
    P(f"  {name} | {len(ids)} | {np.mean([ini_lab(i) for i in ids]):.4f} | {mm.mean():.4f} | " + " | ".join(cells))
P("  rank correlation over the 56 labels (Spearman, no test): method-minus-opponent mean delta against the label's unregistered (identity) Dice")
for k in MAIN:
    x = np.array([ini_lab(i).mean() for i in LIDS])
    y = np.array([(lab("method", i) - lab(k, i)).mean() for i in LIDS])
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    P(f"    {k}: rho {np.corrcoef(rx, ry)[0, 1]:+.2f}")

# ---- item 3: lower bounds ------------------------------------------------------------------------------------------------
P("\n== item 3: certificate lower bounds (decision certificate: method cell output/cert.json; BL1: pair records arms.<arm>.certificate)")
for k in ("oasis", "lpba"):
    ce = [J(CELL[k](p) + "/output/cert.json") for p in pairs[k]]
    P(f"-- {NAME[k]}")
    P("  decision certificate (configuration C, full mode): status", counts([c["status"] for c in ce]), "| n_infeasible sum", sum(c["n_infeasible"] for c in ce), "| n_uncertain sum", sum(c["n_uncertain"] for c in ce))
    lb = Q("method, decision certificate min_jd_lower_bound (one number per pair: the certified lower bound of min J over all closed cells)", [c["min_jd_lower_bound"] for c in ce])
    xs = np.array([c["min_jd_lower_bound"] for c in ce])
    P("  pairs with the bound below 1e-8 / 1e-7 / 1e-6 / 1e-5 / 1e-4 / 1e-3 / 1e-2:", " / ".join(str(int((xs < t).sum())) for t in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)))
    Q("method, min_margin_lower_bound (cert.json)", [c["min_margin_lower_bound"] for c in ce])
    ev = [r["arms"]["method"]["decision_certificate"]["certificate"]["min_jd_lower_bound"] for r in recs[k]]
    P("  evaluation's recomputation equals cert.json:", sum(a == b for a, b in zip(ev, xs)), "/", len(ev))
    Q("method, min_J_centre (smallest cell-centre J on the whole grid; pair records metrics)", [r["arms"]["method"]["metrics"]["min_J_centre"] for r in recs[k]])
    Q("method, pct_J_lt_half_brain (% of brain cells with centre J < .5)", [r["arms"]["method"]["metrics"]["pct_J_lt_half_brain"] for r in recs[k]])
    Q("method, logj_q01_brain", [r["arms"]["method"]["metrics"]["logj_q01_brain"] for r in recs[k]])
    Q("method, logj_q99_brain", [r["arms"]["method"]["metrics"]["logj_q99_brain"] for r in recs[k]])
    arms = [a for a in recs[k][0]["arms"]]
    P("  BL1 (jacobian_only) min_jd_lower_bound per arm: passed | min | q25 | median | q75 | max")
    for a in arms:
        c = [r["arms"][a]["certificate"] for r in recs[k]]
        q = np.percentile([x["min_jd_lower_bound"] for x in c], [0, 25, 50, 75, 100])
        P(f"    {a}: {sum(bool(x['passed']) for x in c)}/{len(c)} | " + " | ".join(g(v) for v in q))
P("  where the cells with J < .5 are: NOT computed. The records hold one number per pair (pct_J_lt_half_brain) and no location.")

# ---- item 4: seconds ------------------------------------------------------------------------------------------------------
P("\n== item 4: seconds per arm, median (min..max); method: output/line.json; external arms and configuration C: <arm>/<pair>/result.json")
trow = [["set", "arm", "n", "quantity", "median", "min", "max"]]


def T(k, arm, name, xs):
    xs = np.asarray(xs, dtype=float)
    P(f"  {arm} | {name}: median {np.median(xs):.2f} ({xs.min():.2f}..{xs.max():.2f}); n {len(xs)}")
    trow.append([k, arm, len(xs), name, f"{np.median(xs):.3f}", f"{xs.min():.3f}", f"{xs.max():.3f}"])


for k, root in (("oasis", H), ("lpba", L)):
    P(f"-- {NAME[k]}")
    ln = [J(CELL[k](p) + "/output/line.json") for p in pairs[k]]
    T(k, "method", "task clock line.json seconds_total (supervisor: T88 + repair + decision; no U-Net)", [x["seconds_total"] for x in ln])
    T(k, "method", "t88.seconds_task", [x["t88"]["seconds_task"] for x in ln])
    T(k, "method", "repair.seconds_repair", [x["repair"]["seconds_repair"] for x in ln])
    T(k, "method", "repair.seconds_warmup", [x["repair"]["seconds_warmup"] for x in ln])
    T(k, "method", "seconds_total - t88.seconds_task - repair.seconds_repair (the rest: start-up, channels, warm-up, checks)",
      [x["seconds_total"] - x["t88"]["seconds_task"] - x["repair"]["seconds_repair"] for x in ln])
    P(f"  method | pairs with task clock <= 90 s: {sum(x['seconds_total'] <= 90 for x in ln)}/{len(ln)}; <= 120 s: {sum(x['seconds_total'] <= 120 for x in ln)}/{len(ln)}")
    hi = J(HI[k])["rows"]
    T(k, "method", "HI (U-Net, per subject, a separate unit) seconds_total", [v["seconds_total"] for v in hi.values()])
    T(k, "method", "HI seconds_forward", [v["seconds_forward"] for v in hi.values()])
    cc = [J(f"{root}/configc/{p}/result.json") for p in pairs[k]]
    f = "%Y-%m-%dT%H:%M:%SZ"
    T(k, "configc", "result.json utc_end - utc_start (whole seconds; the driver's span, the 90 s native budget inside)",
      [(datetime.strptime(c["utc_end"], f) - datetime.strptime(c["utc_start"], f)).total_seconds() for c in cc])
    for a in XB:
        x = [J(f"{root}/{a}/{p}/result.json") for p in pairs[k]]
        assert all(r["status"] == "ok" for r in x)
        T(k, a, "method.seconds_registration (cold; the method's own report)", [r["method"]["seconds_registration"] for r in x])
        if all(r["method"].get("seconds_total") is not None for r in x):
            T(k, a, "method.seconds_total (method process start to output written)", [r["method"]["seconds_total"] for r in x])
        T(k, a, "seconds_driver_wall", [r["seconds_driver_wall"] for r in x])
        P(f"    {a} device: {counts([str(r['method'].get('device')) for r in x])}")

# ---- item 5: development set, official Dice -----------------------------------------------------------------------------
P("\n== item 5: development set (19 Learn2Reg validation pairs; HS, single run; not an independent test): after.metrics of repair_metrics.json")
am = [m["after"]["metrics"] for m in dv]
assert len(am) == 19 and all(m["oracle"] is False for m in dv)
for key in ("seg35", "seg4", "dice_official", "hd95", "sdlogj_official", "sdlogj_brain"):
    xs = np.array([m[key] for m in am], dtype=float)
    P(f"  {key}: mean {np.nanmean(xs):.4f}; sd (ddof 0) {np.nanstd(xs):.4f}; min {np.nanmin(xs):.4f}; max {np.nanmax(xs):.4f}; nonfinite {int((~np.isfinite(xs)).sum())}")
P("  pairs with seg35 != dice_official:", sum(m["seg35"] != m["dice_official"] for m in am), "| max |difference|", g(max(abs(m["seg35"] - m["dice_official"]) for m in am)))
P("  after.certificate status_at_1 certified_feasible with accepted_alpha 1 and n_infeasible_at_1 0:", sum(bool(m["after"]["certificate"].get("accepted_alpha") == 1.0 and m["after"]["certificate"].get("n_infeasible_at_1") == 0) for m in dv), "/ 19; arms", counts([m["arm"] for m in dv]), "| eps", counts([m["eps"] for m in dv]))

# ---- item 6: scatter input ---------------------------------------------------------------------------------------------------
P("\n== item 6: input table of the scatter figure (scatter_input.csv)")
srow = [["set", "arm", "family", "n", "dice_a_name", "dice_a", "dice_b_name", "dice_b", "sdlogj_official", "cert_name", "cert_passed", "cert_of", "source"]]
with open(os.path.join(REPO, "runs/20261119-hb-result/arms_absolute.csv")) as fh:
    rd = list(csv.reader(fh))
hd = rd[0]
ix = {n: [i for i, h in enumerate(hd) if h == n][0] for n in ("arm", "family", "n", "seg35_mean", "seg4_mean", "sdlogj_official_mean", "bl1_passed")}
for r in rd[1:]:
    arm = r[ix["arm"]]
    passed = "" if arm == "identity" else int(r[ix["bl1_passed"]])
    srow.append(["oasis", arm, r[ix["family"]], r[ix["n"]], "seg35", f"{float(r[ix['seg35_mean']]):.4f}", "seg4", f"{float(r[ix['seg4_mean']]):.4f}",
                 f"{float(r[ix['sdlogj_official_mean']]):.4f}", "BL1 jacobian_only", passed, 50, "runs/20261119-hb-result/arms_absolute.csv"])
    if arm != "identity":      # the same from the pair records
        a = [x["arms"][arm] for x in recs["oasis"]]
        chk = (np.mean([x["metrics"]["seg35"] for x in a]), np.mean([x["metrics"]["seg4"] for x in a]), np.mean([x["metrics"]["sdlogj_official"] for x in a]), sum(bool(x["certificate"]["passed"]) for x in a))
        assert abs(chk[0] - float(r[ix["seg35_mean"]])) < 1e-12 and abs(chk[1] - float(r[ix["seg4_mean"]])) < 1e-12 and abs(chk[2] - float(r[ix["sdlogj_official_mean"]])) < 1e-12 and chk[3] == passed, arm
P("  OASIS: 16 arms + identity from arms_absolute.csv (seg35_mean, seg4_mean, sdlogj_official_mean, bl1_passed); each equal to the pair records' means (1e-12) and pass counts")
for arm, (d54, d56, sd, bl1) in LPBA_DOC.items():
    fam = "reference" if arm == "identity" else "method" if arm == "method" else "main"
    srow.append(["lpba", arm, fam, 17, "lpba54", f"{d54:.4f}", "lpba56", f"{d56:.4f}", f"{sd:.4f}", "BL1 jacobian_only", "" if bl1 is None else bl1, 17,
                 "runs/20261122-lpba-result/result.md section 2 and 2.1"])
    if arm != "identity":
        a = [x["arms"][arm] for x in recs["lpba"]]
        chk = (np.mean([x["metrics"]["lpba54"] for x in a]), np.mean([x["metrics"]["lpba56"] for x in a]), np.mean([x["metrics"]["sdlogj_official"] for x in a]), sum(bool(x["certificate"]["passed"]) for x in a))
        assert abs(chk[0] - d54) < 5.0001e-5 and abs(chk[1] - d56) < 5.0001e-5 and abs(chk[2] - sd) < 5.0001e-5 and chk[3] == bl1, arm
P("  LPBA40: 6 arms + identity as printed in the result document (section 2, 2.1); each equal to the pair records' means after rounding to 4 places, and the pass counts")
P("  method's decision certificate (configuration C, full mode) passed: OASIS", sum(bool(r["arms"]["method"]["decision_certificate"]["certificate"]["passed"]) for r in recs["oasis"]), "/ 50; LPBA40",
  sum(bool(r["arms"]["method"]["decision_certificate"]["certificate"]["passed"]) for r in recs["lpba"]), "/ 17")
for r in srow[1:]:
    P("  " + " | ".join(str(x) for x in r[:12]))


# ---- the guard and the files -------------------------------------------------------------------------------------------------
def table(rows):
    b = io.StringIO()
    csv.writer(b, lineterminator="\n").writerows(rows)
    return b.getvalue()


files = {"scatter_input.csv": table(srow), "lpba_per_label_delta.csv": table(rows), "timing_main_family.csv": table(trow)}
P("\nscript sha256", sha(os.path.abspath(__file__)))
files["posthoc.out"] = "\n".join(out) + "\n"
tokens = set()
for k in pairs:
    for p, r in zip(pairs[k], recs[k]):
        tokens |= {p, str(r["fixed"]), str(r["moving"])}
tokens |= set(J(HI["oasis"])["rows"]) | set(J(HI["lpba"])["rows"]) | {m["pair"] for m in dv}
tokens = {t for t in tokens if len(t) >= 3}
for name, text in files.items():
    if any(t in text for t in tokens) or re.search(r"OAS[0-9_]|OASIS_|(?<![A-Za-z])S\d\d(?!\d)|(?<!\d)\d{4}[-_]\d{4}(?!\d)", text):
        hit = [t for t in tokens if t in text]
        m = re.search(r"OAS[0-9_]|OASIS_|(?<![A-Za-z])S\d\d(?!\d)|(?<!\d)\d{4}[-_]\d{4}(?!\d)", text)
        sys.exit(f"guard: an identifier in {name}; nothing written (token hits {len(hit)}, shapes {sorted(set(re.sub('[0-9]', '#', re.sub('[A-Za-z]', 'a', t)) for t in hit))}; pattern hit {m is not None}"
                 + (f", in a line starting {text[text.rfind(chr(10), 0, m.start()) + 1:][:40]!r}" if m else "") + ")")
target = HERE
if "--out" in sys.argv:
    target = sys.argv[sys.argv.index("--out") + 1]
for name, text in files.items():
    with open(os.path.join(target, name), "w") as fh:
        fh.write(text)
print("written:", ", ".join(sorted(files)), "| lines of posthoc.out:", len(out))
