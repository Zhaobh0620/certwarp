"""Summary of gate A, X2 (experiments/gate_a/run_x2.py; runs/20261102-gate-a/plan.md, revisions 1 and 2).  Development set, the first five pairs,
single observations: every reading is a label for the lead, not a conclusion; n = 5 gives direction only.

    python -B experiments/gate_a/summarize_x2.py --root DIR

Reads $ROOT/batch.json, runs/<arm>/<short>/, eval/<arm>/<short>/ and, read only, the references named in batch.json: A0
(<a0_runs_root>/<pair>/metrics.json, D0 batch) and P0n (<pilot_root>/eval/p0n/<short>/pilot_p0n_<short>/metrics.json, path pilot).
Rows come from experiments/m2v1/summarize.py pair_row and experiments/pilot/summarize.py derive (evaluator metrics and
certificate, config.json args and eta checks, per-level records: accepted steps, rejections by reason, blocked levels, GF_B/T_B);
the 53-point minimum J is not computed here.  Added: the cortex and white matter Dice (seg4 labels 1 and 3), the six large seg35
structures and the split "six / other 29" against A0 and P0n; the accepted and rejected candidates over all levels and on the native
level; the 1-minute load at each launch.

Readings (the plan, revisions 1 and 2; per arm, separate lines, on the means over the comparable pairs).  Each line says "reached" only with
5/5 comparable pairs (a missing pair is more likely a hard one; review I5); 4/5 or fewer: described only.
  seg4 >= .85                                seg4 达到
  cortex >= .7191 and white matter >= .8366  目标函数改动在现行求解器上把皮层、白质带回了 A0 与 P0n 差距的一半以上 (the bars are only the midpoints of A0 and P0n over these 5 pairs)
  seg35 >= .80                               seg35 达到 (its own line, not part of the large-structure label; the bar lies within the MDE of
                                             both references)
  otherwise                                  未达到（900 s，现行路径）: never evidence against the objective (review I7)
Reproduction check (X2a only): seg4 >= .8445 on 0438 <- 0439 (G3 - .02) -> 复现 G3; otherwise 未复现 G3, and the readings of the other arms
(X2b, X2c, X2d: the same solver path) carry that qualifier.
X2d (support_shared; revision 2) has the same lines, and each of them carries 诊断点，不是合同选出的 η: C04-6 does not carry an eta selected
under brain to support_shared, so its nu is not an eta of the contract; a good result calls for a new probe and sweep.  Also for X2d: X2d -
X2c (the sigma estimation domain, same nu and tau), X2d - P4 (<pilot_root>/eval/p4/<short>/pilot_p4_<short>/metrics.json, same sigma
domain; P4 also has the per-level caps and certificate_bernstein, and code ffc6844), and the support records of hyper.json; a run of a
non-brain arm that ended without them is counted and named at the top of summary.md (the config.json check looks at the arguments only).
Side references, not in the readings: A0 and P0n on the same pairs (paired differences; a line of their means over the arm's own pairs),
NM = VAL4C (19 pairs, .8173 / .8779, archive; these 5 pairs are harder, A0 -.0134 / -.0128: not to be compared directly), G3 on
0438 <- 0439 (.8121 / .8645, cortex .783, white matter .887; 2700 s, gauss27+corners, GPU shared with training).
Also arm b - arm a (the weights, same channels) and arm c - arm b (the channels, same weights), described only.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


psum = _load("_gax2_pilot_summarize", "experiments/pilot/summarize.py")
ms, lt = psum.ms, psum.lt

SCHEMA = "bayesreg.gate_a_x2.summary.v2"
REQUIRED_N = 5                   # review I5: "reached" needs every pair
SEG4_BAR = .85
SEG35_BAR = .80
CORTEX_BAR = .7191               # (A0 .8168 + P0n .6213)/2 over the 5 pairs, from the 4-decimal means (exact .71904)
WM_BAR = .8366                   # (A0 .9033 + P0n .7698)/2 over the 5 pairs, from the 4-decimal means (exact .83654)
BAR_TOLERANCE = 1e-4             # the pins against the midpoints recomputed from the references
G3_SEG4_BAR = .8445              # G3 - .02 on 0438 <- 0439
P0N_CODE = "ffc6844"
NM = dict(seg35=.8173, seg4=.8779, note="VAL4C，19 对，无 T88；归档结果，跨版本与预算，只作旁证（GDR 复核 E4）。这 5 对偏难（A0 差 −.0134 / −.0128），不要直接比")
G3_REF = dict(pair="OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1", seg35=.8121, seg4=.8645, cortex=.783, white_matter=.887, native_accepted=35,
              note="m2_gpu_smoke G3：ν 3.8705、gauss27+corners、2700 s、与分割器训练共用 GPU、受阻收尾；单对")
CORTEX, WM = "1", "3"            # seg4 labels (/mnt/d/data/seg4_labels.txt, experiments/d0/check_seg4_map.py)
SEG35_LARGE = {2: "左大脑皮层", 21: "右大脑皮层", 1: "左大脑白质", 20: "右大脑白质", 5: "左小脑白质", 24: "右小脑白质"}   # GDR, gap decomp §2
MEASURES = (("seg35", "seg35"), ("seg4", "seg4"), ("cortex", "皮层"), ("white_matter", "白质"), ("hd95", "HD95"),
            ("sdlogj_official", "SDlogJ 官方"), ("sdlogj_brain", "SDlogJ 脑内"))
READING_LARGE = "目标函数改动在现行求解器上把皮层、白质带回了 A0 与 P0n 差距的一半以上"
READING_SEG4 = "seg4 达到"
READING_SEG35 = "seg35 达到"
READING_NO = "未达到（900 s，现行路径）"
REPRO_YES = "复现 G3"
REPRO_NO = "未复现 G3"
QUALIFIER = "X2a 未复现 G3，这一读数要带这个限定（先核对计划列出的差别）"
BRAIN = "brain"
DIAGNOSTIC = "诊断点，不是合同选出的 η"
DIAGNOSTIC_NOTE = ("诊断点：C04-6 规定 brain 下选出的 η 不直接用于 support_shared，换选项要重做探针与扫描；ν 1.37、τ (1,1,1,1) 不是合同选出的 η，"
                   "结果好的话要重做探针与扫描，不能直接当作这个模型的表现")
P4_NOTE = ("P4 是 path_pilot_v1 已有的记录（探索臂，代码 ffc6844，不重跑，非同时段）：P3 加 support_shared，ν 21.855、τ (1,.5,.5,.5)，"
           "另有按层上限 500,500,100,500 与 certificate_bernstein 单元估计，X2d 都没有。所以 X2d − P4 同 σ 估计域，但不只差 ν、τ，还差这两项和代码版本；只描述")
GF_NOTE = "T_B 依赖 σ，GF_B/T_B 在不同的 σ 估计域之间不可比（runs/20261031-path-pilot/result.md）；上表 X2d 这一列不要与其他臂比"
SUPPORT_MISSING = "非 brain 臂有运行结束了却没有 support 记录，hyper.json 没有确认它的 σ 估计域；先核对 config.json 的 solver.sigma_domain，这些运行的读数交总负责人"
LOADAVG = re.compile(r"loadavg ([0-9.]+) ([0-9.]+) ([0-9.]+);")
NOTE = ("开发集（stage_L2R_val_v1 前 5 对），单次观测。n = 5 的最小可检测效应约 .009–.020，只有方向信息；读数不是结论。每次运行以开发开关 "
        "--wall-seconds 的“预算收尾”或其他 C13-3 收尾结束，不是“B 线收敛”，除非状态写明。A0、P0n 都不是同时段对照。"
        "“未达到”只说明在 900 s 与现行路径下没到，不能用来否定目标函数：900 s 下原生层约为 G3 的一半〔推断〕，原生接受步明显少于 G3 的 35 步时尤其如此。")


class SummaryError(Exception):
    """The summary cannot be written as specified."""


read_json, finite, short_name, utc, fmt = ms.read_json, ms.finite, ms.short_name, ms.utc, ms.fmt


def seg_labels(doc, measure):
    """{label: Dice} from an evaluation record: bayesreg.eval per_label.<measure>_per_label or D0 metrics.<measure>_per_label."""
    key = f"{measure}_per_label"
    return ((doc or {}).get("per_label") or {}).get(key) or ((doc or {}).get("metrics") or {}).get(key) or {}


def per_label(doc, label):
    value = seg_labels(doc, "seg4").get(label)
    return value if finite(value) else None


def structures(doc):
    return dict(cortex=per_label(doc, CORTEX), white_matter=per_label(doc, WM))


def counts_of(row):
    """Accepted steps and rejected candidates over all levels and on the native level (level_table's records)."""
    levels = (row.get("levels") or {}).get("levels") or []

    def add(rows):
        accepted = [r.get("accepted") for r in rows if isinstance(r.get("accepted"), int)]
        rejected = [sum((r.get("rejections") or {}).values()) for r in rows]
        return dict(accepted=sum(accepted) if rows else None, rejected=sum(rejected) if rows else None)
    native = [r for r in levels if r.get("level") == psum.NATIVE]
    return dict(all=add(levels), native=add(native),
                rejections_by_reason={k: sum((r.get("rejections") or {}).get(k, 0) for r in levels)
                                      for k in sorted({k for r in levels for k in (r.get("rejections") or {})})},
                native_ending=native[-1].get("ending") if native else None)


def launch_load(path):
    """(1, 5, 15)-minute load at a launch, from the launch record's "before" line; None when absent (review I8)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = LOADAVG.search(text)
    return None if match is None else [float(x) for x in match.groups()]


def reference_row(doc, pair):
    if doc is None or not doc.get("metrics"):
        return dict(pair=pair, short=short_name(pair), comparable=False, metrics={}, seg35_labels={})
    m = doc["metrics"]
    metrics = {k: (m.get(k) if finite(m.get(k)) else None) for k in ("seg35", "seg4", "hd95", "sdlogj_official", "sdlogj_brain")}
    metrics.update(structures(doc))
    cert = doc.get("certificate") or {}
    return dict(pair=pair, short=short_name(pair), comparable=cert.get("passed") is True, metrics=metrics, seg35_labels=seg_labels(doc, "seg35"),
                run_id=doc.get("run_id"), min_jd_lower_bound=cert.get("min_jd_lower_bound"))


def references(batch):
    """{"A0": rows, "P0n": rows} in the batch's pair order (absent references: not comparable)."""
    a0, pilot = Path(batch["a0_runs_root"]), Path(batch["pilot_root"])
    rows = dict(A0=[], P0n=[])
    for pair in batch["pairs"]:
        short = short_name(pair)
        rows["A0"].append(reference_row(read_json(a0/pair/"metrics.json"), pair))
        rows["P0n"].append(reference_row(read_json(pilot/"eval"/"p0n"/short/f"pilot_p0n_{short}"/"metrics.json"), pair))
    return rows


def p4_rows(batch):
    """P4 of the path pilot in the batch's pair order (the side reference of X2d, revision 2; an absent record: not comparable)."""
    pilot = Path(batch["pilot_root"])
    return [reference_row(read_json(pilot/"eval"/"p4"/short_name(p)/f"pilot_p4_{short_name(p)}"/"metrics.json"), p) for p in batch["pairs"]]


def arm_row(root, aid, pair, batch):
    row = psum.derive(ms.pair_row(root, aid, pair, dict(batch, eta=batch["arms"][aid]["eta"])), j53=False)
    doc = read_json(row["eval_metrics"]) if row.get("eval_metrics") else None
    row["metrics"] = dict(row["metrics"], **structures(doc))
    row["seg35_labels"] = seg_labels(doc, "seg35")
    row["counts"] = counts_of(row)
    row["launch_loadavg"] = launch_load(row["launches"][0]) if row.get("launches") else None
    return row


def paired(rows_x, rows_y, key):
    per_pair = []
    for x, y in zip(rows_x, rows_y):
        a, b = x["metrics"].get(key), y["metrics"].get(key)
        if x.get("comparable") and y.get("comparable") and finite(a) and finite(b):
            per_pair.append(dict(pair=x["pair"], x=a, y=b, d=a-b))
    d = [p["d"] for p in per_pair]
    return dict(n=len(d), median=statistics.median(d) if d else None, mean=statistics.fmean(d) if d else None,
                positive=sum(v > 0 for v in d), negative=sum(v < 0 for v in d), per_pair=per_pair)


def compare(rows_x, rows_y):
    return {key: paired(rows_x, rows_y, key) for key, _ in MEASURES}


def means(rows):
    ok = [r for r in rows if r.get("comparable")]
    out = dict(n=len(ok))
    for key, _ in MEASURES:
        values = [r["metrics"].get(key) for r in ok if finite(r["metrics"].get(key))]
        out[key] = statistics.fmean(values) if values else None
    return out


def same_pairs_means(rows, reference_rows):
    """The reference's means over the pairs where this arm is comparable (review I5: the line next to the arm's mean)."""
    wanted = {r["pair"] for r in rows if r.get("comparable")}
    return means([r for r in reference_rows if r["pair"] in wanted])


def label_means(rows, labels=SEG35_LARGE):
    """Mean Dice of the seg35 labels over the comparable rows."""
    out = {}
    for label in labels:
        values = [r["seg35_labels"].get(str(label)) for r in rows if r.get("comparable") and finite(r["seg35_labels"].get(str(label)))]
        out[label] = statistics.fmean(values) if values else None
    return out


def seg35_split(rows_x, rows_y):
    """The mean seg35 difference split as in runs/20261102-gap-decomp/result.md §2: the six large structures and the other 29, each the
    sum over its labels of the mean per-label difference / 35, over the pairs where both rows are comparable."""
    diffs = {}
    for label in range(1, 36):
        values = []
        for x, y in zip(rows_x, rows_y):
            a, b = x["seg35_labels"].get(str(label)), y["seg35_labels"].get(str(label))
            if x.get("comparable") and y.get("comparable") and finite(a) and finite(b):
                values.append(a-b)
        diffs[label] = statistics.fmean(values) if values else 0.
    rest = [v for k, v in diffs.items() if k not in SEG35_LARGE]
    return dict(large_six=sum(diffs[k] for k in SEG35_LARGE)/35, other_29=sum(rest)/35, n_other_above=sum(v > 0 for v in rest))


def row_reading(mean, bars, yes):
    """One line of the plan's readings: `yes` with 5/5 comparable pairs and every bar reached; READING_NO with 5/5 and a bar missed;
    otherwise described only."""
    if mean["n"] < REQUIRED_N:
        return f"只描述（可比 {mean['n']}/{REQUIRED_N}；标“达到”要 {REQUIRED_N}/{REQUIRED_N}）"
    return yes if all(finite(mean[k]) and mean[k] >= bar for k, bar in bars.items()) else READING_NO


def readings_of(mean, qualifier=None):
    """The three lines of one arm; `qualifier` (X2d: DIAGNOSTIC) goes into the text of every line, whatever the line says."""
    lines = dict(seg4=row_reading(mean, dict(seg4=SEG4_BAR), READING_SEG4),
                 large_structures=row_reading(mean, dict(cortex=CORTEX_BAR, white_matter=WM_BAR), READING_LARGE),
                 seg35=row_reading(mean, dict(seg35=SEG35_BAR), READING_SEG35))
    return lines if qualifier is None else {k: f"{v}（{qualifier}）" for k, v in lines.items()}


def arm_qualifier(arm_doc):
    """DIAGNOSTIC for an arm outside the brain domain (C04-6: its eta is not an eta of the contract), else None."""
    return DIAGNOSTIC if arm_doc["sigma_domain"] != BRAIN else None


def support_ratio(entry):
    """N_S/N_q of one support record (the count of C11-8 against N_q); None when either is missing."""
    n_s, n_q = entry.get("N_S"), entry.get("N_q")
    return n_s/n_q if finite(n_s) and finite(n_q) and n_q > 0 else None


def reproduction(rows, pairs):
    """X2a's check on 0438 <- 0439 (the plan, revision 1): seg4 >= .8445, else 未复现 G3 (a run that is not comparable there too); not
    decided when the batch has no such pair or no arm a (dry runs)."""
    pair = G3_REF["pair"]
    record = dict(pair=pair, bar=G3_SEG4_BAR, g3=G3_REF["seg4"])
    if "a" not in rows or pair not in pairs:
        return dict(record, state="未判（这一批没有 X2a 的这一对）", value=None, qualifier=None)
    row = rows["a"][pairs.index(pair)]
    value = row["metrics"].get("seg4")
    if row.get("comparable") and finite(value) and value >= G3_SEG4_BAR:
        return dict(record, state=REPRO_YES, value=value, qualifier=None)
    return dict(record, state=REPRO_NO, value=value, qualifier=QUALIFIER)


def bars_check(ref_means):
    """The pinned bars against the midpoints of the references over the batch's pairs."""
    out = {}
    for key, bar in (("cortex", CORTEX_BAR), ("white_matter", WM_BAR)):
        a, b = ref_means["A0"].get(key), ref_means["P0n"].get(key)
        midpoint = None if a is None or b is None else (a+b)/2
        out[key] = dict(bar=bar, midpoint=midpoint, ok=midpoint is not None and abs(midpoint-bar) <= BAR_TOLERANCE)
    return out


def summary(root):
    root = Path(root)
    batch = read_json(root/"batch.json")
    if batch is None:
        raise SummaryError(f"{root}/batch.json missing (the driver writes it at its start)")
    pairs, order = batch["pairs"], batch["arm_order"]
    rows = {aid: [arm_row(root, aid, p, batch) for p in pairs] for aid in order}
    refs = references(batch)
    arm_means = {aid: means(rows[aid]) for aid in order}
    ref_means = {name: means(r) for name, r in refs.items()}
    comparisons = {aid: {name: compare(rows[aid], r) for name, r in refs.items()} for aid in order}
    splits = {aid: {name: seg35_split(rows[aid], r) for name, r in refs.items()} for aid in order}
    same_pairs = {aid: {name: same_pairs_means(rows[aid], r) for name, r in refs.items()} for aid in order}
    large = {**{aid: label_means(rows[aid]) for aid in order}, **{name: label_means(r) for name, r in refs.items()}}
    extra = {}
    for a, b, title in (("b", "a", "权重的效应（同为 tissue）"), ("c", "b", "通道的效应（同为 ν 1.37、τ 1）"),
                        ("d", "c", "σ 估计域的效应（同为 structure、ν 1.37、τ 1：support_shared 对 brain）")):
        if a in rows and b in rows:
            extra[f"{a}_vs_{b}"] = dict(title=title, first=batch["arms"][a]["label"], second=batch["arms"][b]["label"],
                                        comparison=compare(rows[a], rows[b]), seg35_split=seg35_split(rows[a], rows[b]))
    p4 = None
    if "d" in rows:                 # revision 2: not in the readings, and only against X2d
        p4_list = p4_rows(batch)
        large["P4"] = label_means(p4_list)
        p4 = dict(rows=p4_list, means=means(p4_list), note=P4_NOTE, code=P0N_CODE)
        extra["d_vs_P4"] = dict(title="同 σ 估计域；ν 21.855、τ (1,.5,.5,.5) 改为 ν 1.37、τ (1,1,1,1)；P4 另有按层上限、certificate_bernstein 与不同的代码，非同时段",
                                first=batch["arms"]["d"]["label"], second="P4", comparison=compare(rows["d"], p4_list),
                                seg35_split=seg35_split(rows["d"], p4_list))
    flat = [r for aid in order for r in rows[aid]]
    # A run that ended (status.json) under a non-brain domain must have support records; killed runs are failures, counted above
    no_support = [f"{aid}-{r['short']}" for aid in order if batch["arms"][aid]["sigma_domain"] != BRAIN for r in rows[aid]
                  if r["state"] == "ran" and not r.get("support")]
    counts = dict(ran=sum(r["state"] == "ran" for r in flat), not_run=sum(r["state"] == "not_run" for r in flat),
                  failed_no_status=sum(r["state"] == "failed_no_status" for r in flat),
                  evaluator_certificate_passed=sum((r["certificate"] or {}).get("passed") is True for r in flat),
                  args_mismatch=sum(bool(r["args_differ"]) for r in flat), eta_mismatch=sum(r["eta_ok"] is False for r in flat),
                  non_brain_without_support=len(no_support))
    repro = reproduction(rows, pairs)
    return dict(schema=SCHEMA, note=NOTE, written_utc=utc(), root=str(root), stage=batch["stage"], pairs=pairs, arm_order=order,
                arms={a: {k: batch["arms"][a][k] for k in ("label", "note", "channel_source", "sigma_domain", "support_threshold", "eta_role", "eta")}
                      for a in order},
                code_sha=batch["code_sha"], p0n_code=P0N_CODE, wall_seconds=batch["wall_seconds"], dry_run=bool(batch.get("dry_run")),
                bars=dict(seg4=SEG4_BAR, seg35=SEG35_BAR, cortex=CORTEX_BAR, white_matter=WM_BAR, required_n=REQUIRED_N, g3_seg4=G3_SEG4_BAR),
                bars_check=bars_check(ref_means), side_references=dict(NM=NM, G3=G3_REF), counts=counts, non_brain_without_support=no_support,
                rows=rows, references=refs,
                arm_means=arm_means, reference_means=ref_means, same_pairs_reference_means=same_pairs, seg35_large_means=large, seg35_split=splits,
                comparisons=comparisons, extra_comparisons=extra, p4=p4,
                readings={a: readings_of(arm_means[a], arm_qualifier(batch["arms"][a])) for a in order}, reproduction=repro)


def label_of(doc, aid):
    return doc["arms"][aid]["label"]


def markdown(doc):
    out = [f"# 关口 A X2 汇总（{doc['root']}）", "", f"**{doc['note']}**", ""]
    if doc["non_brain_without_support"]:
        out += [f"**{SUPPORT_MISSING}：{'、'.join(doc['non_brain_without_support'])}。**", ""]
    out += [f"代码 {doc['code_sha']}；P0n 的代码是 {doc['p0n_code']}，其间 run_pair、lm、multilevel、outer、pg、proximal、stopping 有改动，默认路径由各包复核确认逐位不变，这里没有重新验证。"
            f"每次 {doc['wall_seconds']:g} s。", "", "## 各臂", "", "| 臂 | 通道 | σ 估计域 | α | ν | τ | 说明 |", "|---|---|---|---|---|---|---|"]
    for aid in doc["arm_order"]:
        a = doc["arms"][aid]
        domain = a["sigma_domain"] if a["support_threshold"] is None else f"{a['sigma_domain']}（θ {a['support_threshold']:g}）"
        out.append(f"| {a['label']} | {a['channel_source']} | {domain} | {a['eta']['alpha']:g} | {a['eta']['nu']:g} | {a['eta']['tau']} | {a['note']} |")
    if doc["p4"]:
        out += ["", f"**X2d 是{DIAGNOSTIC_NOTE}。**"]
    c = doc["counts"]
    out += ["", f"运行 {c['ran']}，未运行 {c['not_run']}，失败（无 status.json）{c['failed_no_status']}，评测证书通过 "
            f"{c['evaluator_certificate_passed']}，args 不符 {c['args_mismatch']}，η 不符 {c['eta_mismatch']}，非 brain 臂没有 support 记录的运行 "
            f"{c['non_brain_without_support']}。"]
    bad = [f"{k}：门槛 {v['bar']}，中点 {fmt(v['midpoint'])}" for k, v in doc["bars_check"].items() if not v["ok"]]
    if bad and not doc["dry_run"]:
        out += ["", "**门槛与 A0、P0n 同批对的中点不符，读数要交总负责人**：" + "；".join(bad)]
    out += ["", "## 均值（seg4 与 seg35 同时报；可比的配对）", "", "| 臂 | 可比对数 | seg35 | seg4 | 皮层 | 白质 | HD95 | SDlogJ 官方 | SDlogJ 脑内 |", "|---|---|---|---|---|---|---|---|---|"]
    table = [(label_of(doc, a), doc["arm_means"][a]) for a in doc["arm_order"]]+list(doc["reference_means"].items())
    if doc["p4"]:
        table.append((f"P4（旁证，只对 X2d；代码 {doc['p4']['code']}）", doc["p4"]["means"]))
    for aid in doc["arm_order"]:
        if doc["arm_means"][aid]["n"] < len(doc["pairs"]):      # the references over the arm's own pairs, only when they differ from the lines above
            table += [(f"{name}（{label_of(doc, aid)} 的同批对）", doc["same_pairs_reference_means"][aid][name]) for name in doc["reference_means"]]
    for name, m in table:
        out.append(f"| {name} | {m['n']}/{len(doc['pairs'])} | {fmt(m['seg35'])} | {fmt(m['seg4'])} | {fmt(m['cortex'])} | {fmt(m['white_matter'])} | "
                   f"{fmt(m['hd95'], 3)} | {fmt(m['sdlogj_official'], 3)} | {fmt(m['sdlogj_brain'], 3)} |")
    nm, g3 = doc["side_references"]["NM"], doc["side_references"]["G3"]
    out += ["", f"旁证，不进读数：NM {nm['seg35']} / {nm['seg4']}（{nm['note']}）；G3 只有 {short_name(g3['pair'])}：{g3['seg35']} / {g3['seg4']}，皮层 "
            f"{g3['cortex']}，白质 {g3['white_matter']}，原生接受 {g3['native_accepted']} 步（{g3['note']}）。"]
    side = [*doc["reference_means"], *(["P4"] if doc["p4"] else [])]
    if doc["p4"]:
        out += ["", f"{doc['p4']['note']}。"]
    out += ["", "## seg35 六个大结构与其余 29 个（可比的配对）", "", "| seg35 标签 | " + " | ".join([label_of(doc, a) for a in doc["arm_order"]]+side) + " |",
            "|---|"+"---|"*(len(doc["arm_order"])+len(side))]
    for label, text in SEG35_LARGE.items():
        cells = [fmt(doc["seg35_large_means"][k][label]) for k in [*doc["arm_order"], *side]]
        out.append(f"| {label} {text} | " + " | ".join(cells) + " |")
    out += ["", "| 比较 | 六个大结构（标签均值差之和 / 35） | 其余 29 个 | 其余 29 个里为正的结构数 |", "|---|---|---|---|"]
    for aid in doc["arm_order"]:
        for name, s in doc["seg35_split"][aid].items():
            out.append(f"| {label_of(doc, aid)} − {name} | {fmt(s['large_six'])} | {fmt(s['other_29'])} | {s['n_other_above']} |")
    out += ["", "## 逐对逐臂", "", "| 对 | 臂 | 状态 | seg35 | seg4 | 皮层 | 白质 | 证书 / 下界 | 接受步（全部 / 原生） | 拒绝（全部 / 原生） | 受阻层 | 原生收尾 | 原生末 GF_B/T_B | 启动负载（1 分钟） |", "|"+"---|"*14]
    for k, pair in enumerate(doc["pairs"]):
        for aid in doc["arm_order"]:
            r = doc["rows"][aid][k]
            m, cert, n = r["metrics"], r["certificate"] or {}, r["counts"]
            load = r.get("launch_loadavg")
            out.append(f"| {short_name(pair)} | {label_of(doc, aid)} | {r['state']} / {r['run_status']} | {fmt(m.get('seg35'))} | {fmt(m.get('seg4'))} | "
                       f"{fmt(m.get('cortex'))} | {fmt(m.get('white_matter'))} | {cert.get('passed')} / {fmt(cert.get('min_jd_lower_bound'), 3)} | "
                       f"{fmt(n['all']['accepted'])} / {fmt(n['native']['accepted'])} | {fmt(n['all']['rejected'])} / {fmt(n['native']['rejected'])} | "
                       f"{fmt(r['n_blocked'])} | {n['native_ending']} | {fmt(r['native_gf_over_tb'], 3)} | {'—' if not load else fmt(load[0], 2)} |")
    if any(doc["arms"][aid]["sigma_domain"] != BRAIN for aid in doc["arm_order"]):
        out += ["", f"{GF_NOTE}。"]
    head =["", "| 量 | n | 中位差 | 均值差 | 正 | 负 | 逐对 |", "|---|---|---|---|---|---|---|"]

    def lines(comp):
        for key, name in MEASURES:
            s = comp[key]
            per = "；".join(f"{short_name(x['pair'])} {x['d']:+.4g}" for x in s["per_pair"])
            yield (f"| {name} | {s['n']} | {fmt(s['median'])} | {fmt(s['mean'])} | {s['positive']} | {s['negative']} | {per} |")
    for aid in doc["arm_order"]:
        for name in doc["reference_means"]:
            out += ["", f"## {label_of(doc, aid)} − {name} 的配对差（只描述；非同时段）", *head, *lines(doc["comparisons"][aid][name])]
    for item in doc["extra_comparisons"].values():
        s = item["seg35_split"]
        out += ["", f"## {item['first']} − {item['second']} 的配对差（{item['title']}；只描述）", *head, *lines(item["comparison"]), "",
                f"seg35 拆分：六个大结构 {fmt(s['large_six'])}，其余 29 个 {fmt(s['other_29'])}（{s['n_other_above']} 个为正）。"]
    rep = doc["reproduction"]
    out += ["", "## 读数（预登记，修订 1、2；不是结论；每行分别判，标“达到”要 5/5）", "",
            f"- **X2a 的复现核对**（{short_name(rep['pair'])} 上 seg4 ≥ {rep['bar']}，G3 − .02）：seg4 {fmt(rep['value'])}：{rep['state']}。", ""]
    for aid in doc["arm_order"]:
        m, r = doc["arm_means"][aid], doc["readings"][aid]
        native = [x["counts"]["native"]["accepted"] for x in doc["rows"][aid] if x.get("comparable") and isinstance(x["counts"]["native"]["accepted"], int)]
        out.append(f"- **{label_of(doc, aid)}**（可比 {m['n']}/{len(doc['pairs'])}，原生接受步中位 {fmt(statistics.median(native) if native else None)}，G3 为 {G3_REF['native_accepted']}）：")
        out.append(f"  - seg4 均值 {fmt(m['seg4'])}（≥ {SEG4_BAR}）：{r['seg4']}；")
        out.append(f"  - 皮层 {fmt(m['cortex'])}（≥ {CORTEX_BAR}）、白质 {fmt(m['white_matter'])}（≥ {WM_BAR}）：{r['large_structures']}；")
        out.append(f"  - seg35 均值 {fmt(m['seg35'])}（≥ {SEG35_BAR}；门槛离 A0、P0n 都在 MDE 之内，单独一行）：{r['seg35']}。")
        if aid != "a" and rep["qualifier"]:
            out.append(f"  - {rep['qualifier']}。")
        if doc["arms"][aid]["sigma_domain"] != BRAIN:
            out.append(f"  - {DIAGNOSTIC_NOTE}。")
    support = [(label_of(doc, aid), r) for aid in doc["arm_order"] for r in doc["rows"][aid] if r.get("support")]
    if support:
        out += ["", "## σ 估计域的支撑记录（非 brain 的运行；hyper.json support_levels）", "",
                "| 运行 | 层 | N_c/N_q | 空支撑通道数 | N_S/N_q | σ_S²（层首） | σ_S²（轮用 / 轮后） |", "|---|---|---|---|---|---|---|"]
        for name, r in support:
            for e in r["support"]:
                ratio = "，".join(fmt(x, 3) for x in e["N_c_over_N_q"] or [])
                used = "；".join(f"{fmt(a, 3)} / {fmt(b, 3)}" for a, b in zip(e["sigma2_S_used"], e["sigma2_S_post"]))
                out.append(f"| {name}-{r['short']} | {e['level']} | {ratio} | {e['n_empty']} | {fmt(support_ratio(e), 3)} | {fmt(e['sigma2_S_start'], 3)} | {used or '—'} |")
    records = [dict(r["levels"], label=f"{label_of(doc, aid)}-{r['short']}") for aid in doc["arm_order"] for r in doc["rows"][aid] if r.get("levels")]
    out += ["", lt.markdown(records, title="各层读数（level_table：接受步、拒绝原因、受阻收尾、GF_B/T_B、认证下界）")]
    return "\n".join(out)+"\n"


def strip_volatile(doc):
    return {k: v for k, v in doc.items() if k != "written_utc"}


def write(root):
    """summary.json / summary.md once; a different recomputation goes to summary_<UTC>.*; returns the paths and the state."""
    root = Path(root)
    doc = summary(root)
    text = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False)+"\n"
    stored = read_json(root/"summary.json")
    if stored is None:
        target_json, target_md = root/"summary.json", root/"summary.md"
    elif json.loads(json.dumps(strip_volatile(doc))) == strip_volatile(stored):
        return root/"summary.json", root/"summary.md", "unchanged"
    else:
        s = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target_json, target_md = root/f"summary_{s}.json", root/f"summary_{s}.md"
    with open(target_json, "x", encoding="utf-8") as stream:
        stream.write(text)
    with open(target_md, "x", encoding="utf-8") as stream:
        stream.write(markdown(doc))
    return target_json, target_md, "new" if stored is None else "differs (new file; the earlier one kept)"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -B experiments/gate_a/summarize_x2.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        js, md, state = write(args.root)
    except SummaryError as exc:
        print(f"summary: {exc}", file=sys.stderr)
        return 8
    print(f"summary: {js} {md} ({state})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
