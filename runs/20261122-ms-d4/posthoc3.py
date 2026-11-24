"""MS-D4, package B, amendment 3, items 1, 3, 4 (post hoc, descriptive, outside both preregistrations).

Reads only JSON records (decision.json, the pairs' evaluation records, the method cells' repair/repair.json and
output/line.json), the committed csv files of this directory and of the hold-out result, and two metadata files of
the LPBA40 distribution that hold label NAMES only (no label volume): raw/delineation_space/lpba40.label.xml and
notes/label_id_map.md.  No image, no label volume, no truth, no field.

  item 1  per arm: the Jacobian-only (BL1) certificate against the central-difference folding count of the same field
  item 3  LPBA40 label names per id; the six labels caudate / putamen / hippocampus against the five comparators
  item 4  the inputs of a before/after-repair table, one column per test set

    python -I -B posthoc3.py [--out DIR]
"""
import csv
import hashlib
import io
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = "/home/rmcfz/seq_vp_tag_results"
H, HD, L = RES + "/holdout_B_v1", "/mnt/d/seq_vp_tag_holdout_B_v1", RES + "/lpba40_test_v1"
PINS = {H + "/eval/decision.json": "a0dbdcf9", L + "/eval/decision.json": "a0596c5f"}
NAMES_XML = "/mnt/d/data/LPBA40/raw/delineation_space/lpba40.label.xml"
NAMES_MD = "/mnt/d/data/LPBA40/notes/label_id_map.md"
NAME_PINS = {NAMES_XML: "3b3f1602a8e784993c6a81eb40897dd0766dd305c74d2bf47507276706b0bc05", NAMES_MD: "608d0c3bf9b8200fc8a12c439846c5a61a831f591398d9f8ebcecbc211817a84"}
# arm key -> (name in the manuscript's Table 1 / text where it has one, name in the OASIS preregistration's table of section 2.3, family)
ARMS = [("method", "CertWarp", "method (Hx24-RF05-F1200-v4)", "method"),
        ("configc", "Frozen solver [R-P1]", "configuration C (paper 1's frozen solver)", "main (not external)"),
        ("ants_s", "ANTs SyN", "ANTs SyN `-t s`", "main"), ("fireants_greedy", "FireANTs Greedy", "FireANTs Greedy", "main"),
        ("fireants_syn", "FireANTs SyN", "FireANTs SyN", "main"), ("convexadam_v4seg", "ConvexAdam-Seg", "ConvexAdam-v4seg", "main"),
        ("ants_so", "", "ANTs SyN `-t so`", "descriptive"), ("niftyreg", "", "NiftyReg", "descriptive"), ("deeds", "", "deedsBCV", "descriptive"),
        ("convexadam", "", "ConvexAdam (MIND-SSC)", "descriptive"), ("unigradicon_io", "", "uniGradICON + IO", "descriptive"),
        ("unigradicon_noio", "", "uniGradICON without IO", "descriptive"), ("voxelmorph", "", "VoxelMorph", "descriptive"),
        ("fireants_greedy_eq", "", "FireANTs Greedy, 90 s tier", "90 s tier"), ("fireants_syn_eq", "", "FireANTs SyN, 90 s tier", "90 s tier"),
        ("unigradicon_io_eq", "", "uniGradICON + IO, 90 s tier", "90 s tier")]
EXTERNAL = [a[0] for a in ARMS[2:]]
OPP = ["configc", "ants_s", "fireants_greedy", "fireants_syn", "convexadam_v4seg"]


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
    return f"{x:.4g}"


def mr(xs, f=g):
    xs = np.asarray(xs, dtype=float)
    return f"{f(np.median(xs))} ({f(xs.min())}..{f(xs.max())})"


def table(rows):
    b = io.StringIO()
    csv.writer(b, lineterminator="\n").writerows(rows)
    return b.getvalue()


for p, want in PINS.items():
    if sha(p, 8) != want:
        sys.exit("pin mismatch")
dec = {"oasis": J(H + "/eval/decision.json"), "lpba": J(L + "/eval/decision.json")}
pairs = {k: [e["pair"] for e in sorted(d["positions"], key=lambda e: e["position"])] for k, d in dec.items()}
assert len(pairs["oasis"]) == 50 and len(pairs["lpba"]) == 17
assert not any(re.search(r"S(01|02|37|38|39|40)(?!\d)", p) for p in pairs["lpba"]), "an LPBA pair outside the test list"
recs = {"oasis": [J(f"{H}/eval/v1/pairs/{p}.json") for p in pairs["oasis"]], "lpba": [J(f"{L}/eval/run_1/pairs/{p}.json") for p in pairs["lpba"]]}
assert all(r["pair"] == p and r["complete"] for k in recs for r, p in zip(recs[k], pairs[k]))
NAME = {"oasis": "OASIS hold-out (50 pairs)", "lpba": "LPBA40 second test set (17 pairs)"}

# ---- item 1 ---------------------------------------------------------------------------------------------------------------
P("== item 1: per arm, the Jacobian-only certificate (BL1; pair records arms.<arm>.certificate) against the central-difference count of the same exported field")
P("   (arms.<arm>.metrics.central_det_nonpositive: voxels with det(I + G) <= 0, G = central differences of d, on the volume cropped by 2 voxels per side;")
P("    metrics.cell_centre_J_nonpositive: cells with J <= 0 at the cell centre of the Q1 map, whole grid; certificate.cells.regions.brain.infeasible: cells of the evaluation's brain mask with a certified J <= 0 point)")
hdr = ["set", "arm", "name_in_manuscript", "name_in_preregistration_of_the_hold_out", "family", "n_fields", "bl1_passed", "bl1_failed", "failed_n_uncertain_sum",
       "failed_and_central_diff_zero", "failed_and_cell_centre_zero", "failed_and_both_zero", "failed_central_diff_zero_and_infeasible_in_brain",
       "failed_central_diff_zero_and_brain_central_diff_zero_and_infeasible_in_brain", "infeasible_cells_in_failed_central_zero_fields_median", "infeasible_cells_in_failed_central_zero_fields_max",
       "passed_and_central_diff_positive", "official_num_foldings_sum"]
arows = [hdr]
for k in ("oasis", "lpba"):
    P(f"-- {NAME[k]}")
    P("  arm | name | family | passed | failed | failed with central-difference count 0 | failed with cell-centre count 0 | failed with both 0 | failed, central 0, infeasible cells in the brain mask | infeasible cells per failed central-0 field, median (max) | passed with central count > 0")
    tot = np.zeros(8, dtype=int)
    for arm, ms, pre, fam in ARMS:
        if arm not in recs[k][0]["arms"]:
            continue
        A = [r["arms"][arm] for r in recs[k]]
        cert = [a["certificate"] for a in A]
        passed = np.array([bool(c["passed"]) for c in cert])
        assert all((c["status"] == "certified_feasible") == bool(c["passed"]) for c in cert)
        central = np.array([a["metrics"]["central_det_nonpositive"] for a in A])
        central_b = np.array([a["metrics"]["central_det_nonpositive_brain"] for a in A])
        centre = np.array([a["metrics"]["cell_centre_J_nonpositive"] for a in A])
        ninf = np.array([c["n_infeasible"] for c in cert])
        nunc = np.array([c["n_uncertain"] for c in cert])
        brain_inf = np.array([c["cells"]["regions"]["brain"]["infeasible"] for c in cert])      # every record has the cell split (asserted below)
        folds = sum(a["metrics"]["num_foldings_official"] for a in A)
        fail = ~passed
        assert all(c["cells"]["regions"]["all"]["infeasible"] == c["n_infeasible"] for c in cert)
        assert all(ninf[fail] > 0) and all(ninf[passed] == 0)
        f0 = fail & (central == 0)
        v = [int(passed.sum()), int(fail.sum()), int(f0.sum()), int((fail & (centre == 0)).sum()), int((f0 & (centre == 0)).sum()), int((f0 & (brain_inf > 0)).sum()),
             int((f0 & (central_b == 0) & (brain_inf > 0)).sum()), int((passed & (central > 0)).sum())]
        med = f"{np.median(ninf[f0]):.0f} ({ninf[f0].max()})" if f0.any() else "-"
        P(f"  {arm} | {ms or pre} | {fam} | {v[0]}/{len(A)} | {v[1]} | {v[2]} | {v[3]} | {v[4]} | {v[5]} | {med} | {v[7]}")
        arows.append([k, arm, ms, pre, fam, len(A), v[0], v[1], int(nunc[fail].sum()), v[2], v[3], v[4], v[5], v[6], f"{np.median(ninf[f0]):.0f}" if f0.any() else "", int(ninf[f0].max()) if f0.any() else "", v[7], folds])
        if arm in EXTERNAL:
            tot += np.array(v)
    ne = sum(1 for a in EXTERNAL if a in recs[k][0]["arms"])
    n_f = ne * len(recs[k])
    P(f"  TOTAL over the {ne} external arms: fields {n_f}; passed {tot[0]}; failed {tot[1]}; of the failed: central-difference count 0 in {tot[2]} ({100 * tot[2] / tot[1]:.1f}%);",
      f"cell-centre count 0 in {tot[3]}; both 0 in {tot[4]}; central 0 and infeasible cells inside the brain mask in {tot[5]} (and brain central count 0: {tot[6]}); passed with central count > 0: {tot[7]}")
    arows.append([k, "TOTAL_external", "", f"{ne} external arms", "", n_f, tot[0], tot[1], "", tot[2], tot[3], tot[4], tot[5], tot[6], "", "", tot[7], ""])

# ---- item 3 ---------------------------------------------------------------------------------------------------------------
P("\n== item 3: LPBA40 label names (the distribution's own label file; names only)")
for p, want in NAME_PINS.items():
    if sha(p) != want:
        sys.exit("name table changed: " + os.path.basename(p))
xml_names = {int(e.attrib["id"]): e.attrib["fullname"] for e in ET.parse(NAMES_XML).getroot().iter("label")}
md_names = {}
with open(NAMES_MD) as fh:
    for line in fh:
        m = re.match(r"\|\s*(\d+)\s*\|[^|]*\|\s*(.+?)\s*\|\s*$", line)
        if m:
            md_names[int(m.group(1))] = m.group(2)
LIDS = sorted(i for i in xml_names if i != 0)
P("  lpba40.label.xml: labels", len(xml_names), "(with id 0 background); non-zero ids", len(LIDS), "| notes/label_id_map.md equal to the xml (id and full name):", md_names == xml_names)
rec_ids = sorted(int(i) for i in recs["lpba"][0]["arms"]["method"]["per_label"]["lpba56_per_label"])
P("  the xml's non-zero ids equal the 56 ids of the evaluation records:", LIDS == rec_ids)
nrows = [["label_id", "full_name", "hemisphere", "structure", "id_group", "in_lpba54", "name_contains_gyrus"]]
for i in LIDS:
    nm = xml_names[i]
    hemi = {"L ": "left", "R ": "right"}.get(nm[:2], "none")
    st = nm[2:] if hemi != "none" else nm
    grp = "21-92" if i <= 92 else "101-166" if i <= 166 else "181-182"
    nrows.append([i, nm, hemi, st, grp, int(i < 181), int("gyrus" in nm)])
    assert hemi == "none" or (hemi == "left") == (i % 2 == 1)
for lo, hi in ((21, 92), (101, 166), (181, 182)):
    ids = [i for i in LIDS if lo <= i <= hi]
    sts = []
    for i in ids:
        s = xml_names[i][2:] if xml_names[i][:2] in ("L ", "R ") else xml_names[i]
        if s not in sts:
            sts.append(s)
    P(f"  ids {lo}..{hi}: {len(ids)} labels, {len(sts)} distinct structure names; names with the word 'gyrus': {sum('gyrus' in xml_names[i] for i in ids)} labels;",
      "without it:", sorted(set(s for s in sts if "gyrus" not in s)))
    P("    " + "; ".join(sts))
P("  per id (id: full name):")
for i in LIDS:
    P(f"    {i}: {xml_names[i]}")

R = recs["lpba"]


def lab(arm, i):
    return np.array([r["arms"][arm]["per_label"]["lpba56_per_label"][str(i)] for r in R])


with open(os.path.join(HERE, "lpba_per_label_delta.csv")) as fh:
    prev = {int(r["label_id"]): r for r in csv.DictReader(fh)}
P("  the six labels caudate / putamen / hippocampus: method minus comparator, mean over 17 pairs (pairs with the method higher); from lpba_per_label_delta.csv, recomputed here from the pair records")
P("    id: name | method mean | " + " | ".join(OPP))
srows = [["label_or_group", "name", "n_labels", "method_mean"] + [f"{x}_{k}" for k in OPP for x in ("delta_mean", "wins_of_17")]]
ok = True
for i in range(161, 167):
    m = lab("method", i)
    cells, row = [], [i, xml_names[i], 1, f"{m.mean():.4f}"]
    for k in OPP:
        d = m - lab(k, i)
        cells.append(f"{d.mean():+.4f} ({int((d > 0).sum())})")
        row += [f"{d.mean():+.4f}", int((d > 0).sum())]
        ok &= prev[i][f"delta_mean_{k}"] == f"{d.mean():+.4f}" and int(prev[i][f"wins_of_17_{k}"]) == int((d > 0).sum())
    P(f"    {i}: {xml_names[i]} | {m.mean():.4f} | " + " | ".join(cells))
    srows.append(row)
P("    equal to lpba_per_label_delta.csv for the six labels and five comparators:", ok)
NAMED = [("caudate (161, 162)", (161, 162)), ("putamen (163, 164)", (163, 164)), ("hippocampus (165, 166)", (165, 166)), ("caudate + putamen + hippocampus (161-166)", range(161, 167)),
         ("caudate + putamen (161-164; the 4 non-cortical structures of this label set by name)", range(161, 165)),
         ("insular cortex (101, 102)", (101, 102)), ("cingulate gyrus (121, 122)", (121, 122)), ("insular cortex + cingulate gyrus (101-122)", (101, 102, 121, 122)),
         ("ids 21-34: 7 structures x L/R (superior, middle, inferior frontal, precentral, middle and lateral orbitofrontal gyri, gyrus rectus)", range(21, 35)),
         ("ids 41-50: 5 x L/R (postcentral, superior parietal, supramarginal, angular gyri, precuneus)", range(41, 51)),
         ("ids 61-68: 4 x L/R (superior, middle, inferior occipital gyri, cuneus)", range(61, 69)),
         ("ids 81-92: 6 x L/R (superior, middle, inferior temporal, parahippocampal, lingual, fusiform gyri)", range(81, 93)),
         ("ids 21-92: all 44 gyral labels", [i for i in LIDS if i <= 92]),
         ("ids 21-92 without parahippocampal gyrus (87, 88): 42 labels", [i for i in LIDS if i <= 92 and i not in (87, 88)]),
         ("ids 21-122: the 44 gyral labels + insular cortex + cingulate gyrus (48 labels)", [i for i in LIDS if i <= 122])]
P("  named groups: per pair the mean over the group's labels, then over 17 pairs; method minus comparator (pairs with the method higher)")
P("    group | n labels | method | " + " | ".join(OPP))
for name, ids in NAMED:
    ids = list(ids)
    mm = np.mean([lab("method", i) for i in ids], axis=0)
    cells, row = [], [name.split(":")[0], name, len(ids), f"{mm.mean():.4f}"]
    for k in OPP:
        d = mm - np.mean([lab(k, i) for i in ids], axis=0)
        cells.append(f"{d.mean():+.4f} ({int((d > 0).sum())})")
        row += [f"{d.mean():+.4f}", int((d > 0).sum())]
    P(f"    {name} | {len(ids)} | {mm.mean():.4f} | " + " | ".join(cells))
    srows.append(row)

# ---- item 4 ---------------------------------------------------------------------------------------------------------------
P("\n== item 4: inputs of the before/after-repair table (repair/repair.json of every method cell; runs/20261119-hb-result/ablation.csv)")
CELL = {"oasis": lambda p: f"{HD}/method/{p}", "lpba": lambda p: f"{L}/work/method/{p}"}
col = {}
for k in ("oasis", "lpba"):
    rp = [J(CELL[k](p) + "/repair/repair.json") for p in pairs[k]]
    ln = [J(CELL[k](p) + "/output/line.json") for p in pairs[k]]
    assert all(r["pair"] == p for r, p in zip(rp, pairs[k]))
    ini = [r["initial_certificate"] for r in rp]
    ok0 = sum(i["status"] == "certified_feasible" and (i["min_jd_lower_bound"] or 0) > 0 and i["n_infeasible"] == 0 and i["n_uncertain"] == 0 for i in ini)
    nr = [len(r["rounds"]) for r in rp]
    last = [r["rounds"][-1] for r in rp]
    n_node = [x["grid"]["shape_zyx"][0] * x["grid"]["shape_zyx"][1] * x["grid"]["shape_zyx"][2] for x in ln]
    n_cell = [(x["grid"]["shape_zyx"][0] - 1) * (x["grid"]["shape_zyx"][1] - 1) * (x["grid"]["shape_zyx"][2] - 1) for x in ln]
    i0 = lambda v: f"{v:,.0f}"
    col[k] = {
        "pairs": len(rp),
        "pairs passing the certificate before the repair (initial_certificate)": f"{ok0}/{len(rp)}",
        "cells with a certified J <= 0 point before the repair, median (range) (initial_certificate.n_infeasible)": mr([i["n_infeasible"] for i in ini], i0),
        "  as % of all cells, median (range)": mr([100 * i["n_infeasible"] / c for i, c in zip(ini, n_cell)], lambda v: f"{v:.2f}"),
        "undecided cells before the repair (n_uncertain), sum": sum(i["n_uncertain"] for i in ini),
        "certificate lower bound before the repair, median (range)": mr([i["min_jd_lower_bound"] for i in ini], lambda v: f"{v:.2f}"),
        "repair rounds: pairs with 1 / 2 / 3 rounds": " / ".join(str(nr.count(n)) for n in (1, 2, 3)),
        "nodes changed, median (range) (last round inner.n_moved)": mr([x["inner"]["n_moved"] for x in last], i0),
        "  as % of all nodes, median (range)": mr([100 * x["inner"]["n_moved"] / n for x, n in zip(last, n_node)], lambda v: f"{v:.2f}"),
        "free nodes of the last round, median (range) (n_free)": mr([x["n_free"] for x in last], i0),
        "largest change of a node's displacement, mm, median (range) (last round inner.max_move_mm)": mr([x["inner"]["max_move_mm"] for x in last], lambda v: f"{v:.2f}"),
        "repair seconds, median (range) (seconds_repair)": mr([r["seconds_repair"] for r in rp], lambda v: f"{v:.1f}"),
        "pairs passing the certificate after the repair": f"{sum(bool(r['passed']) for r in rp)}/{len(rp)}",
        "certificate lower bound after the repair, median (range)": mr([r["certificate"]["min_jd_lower_bound"] for r in rp], lambda v: f"{v:.3g}"),
        "pairs at the 1200 s fuse (wall_hit)": sum(bool(r["wall_hit"]) for r in rp),
        "pairs on a fallback (line_status other than hx_certified)": sum(x["line_status"] != "hx_certified" for x in ln),
    }
with open(os.path.join(REPO, "runs/20261119-hb-result/ablation.csv")) as fh:
    ab = {r["endpoint"]: r for r in csv.DictReader(fh) if r["table"] == "main"}
for e, nm, nd in (("seg35", "seg35", 5), ("seg4", "seg4", 5), ("sdlogj_official", "official SDlogJ", 5)):
    a = ab[e]
    assert a["n"] == "50"
    col["oasis"][f"{nm}: before -> after the repair; after - before, mean [95% interval]; pairs lower / higher (ablation.csv)"] = (
        f"{float(a['mean_before']):.{nd}f} -> {float(a['mean_after']):.{nd}f}; {float(a['mean_diff_after_minus_before']):+.{nd}f} [{float(a['ci95_lo']):+.{nd}f}, {float(a['ci95_hi']):+.{nd}f}]; {a['pairs_after_lt_before']} / {a['pairs_after_gt_before']}")
    col["lpba"][f"{nm}: before -> after the repair; after - before, mean [95% interval]; pairs lower / higher (ablation.csv)"] = "none (no repair ablation was run on LPBA40)"
rrows = [["quantity", "OASIS hold-out", "LPBA40"]]
for q in col["oasis"]:
    P(f"  {q} | OASIS {col['oasis'][q]} | LPBA40 {col['lpba'][q]}")
    rrows.append([q.strip(), col["oasis"][q], col["lpba"][q]])

# ---- guard and files ------------------------------------------------------------------------------------------------------
P("\nscript sha256", sha(os.path.abspath(__file__)))
files = {"posthoc3.out": "\n".join(out) + "\n", "external_arms_cert.csv": table(arows), "lpba_label_names.csv": table(nrows), "lpba_named_groups_delta.csv": table(srows), "repair_table.csv": table(rrows)}
tokens = set()
for k in pairs:
    for p, r in zip(pairs[k], recs[k]):
        tokens |= {p, str(r["fixed"]), str(r["moving"])}
tokens = {t for t in tokens if len(t) >= 3}
for name, text in files.items():
    if any(t in text for t in tokens) or re.search(r"OAS[0-9_]|OASIS_|(?<![A-Za-z])S\d\d(?!\d)|(?<!\d)\d{4}[-_]\d{4}(?!\d)", text):
        sys.exit(f"guard: an identifier in {name}; nothing written")
target = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else HERE
for name, text in files.items():
    with open(os.path.join(target, name), "w") as fh:
        fh.write(text)
print("written:", ", ".join(sorted(files)), "| lines:", len(out))
