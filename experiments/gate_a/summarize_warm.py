"""Summary of WARM (experiments/gate_a/run_warm.py; runs/20261103-warm/plan.md).  Development set, single observations: every reading is a label for the
lead, not a conclusion; n = 5 gives direction only.

    python -B experiments/gate_a/summarize_warm.py --root DIR --phase probe|pairs

The probe's run is 0441-0442's result (the lead's revision before any run: 900 s), so the summary of the phase pairs covers all five pairs of batch_pairs.json summary_pairs.
Reads $ROOT/batch_<phase>.json, runs/<arm>/<short>/ (hyper.json profile: L, F_B, RMS, the checkpoints), eval/<arm>/<short>/ (the final field),
eval_fields/<arm>/<short>/<field>/ (the entry and each checkpoint) and, read only, the side references named in the batch: A1Ph (<seed_root>/<pair>/metrics.json) and
X2d from u = 0 (<x2_root>/eval/d/<short>/gax2_d_<short>/metrics.json).  Rows of the final field come from experiments/gate_a/summarize_x2.py arm_row.

Reported at the entry, at every checkpoint and at the end: seg35, seg4, cortex (seg4 label 1), white matter (seg4 label 3), the small six {4, 23, 18, 34, 19, 35}, the large six
{2, 21, 1, 20, 5, 24}, the other 23, HD95, SDlogJ (official, brain); L and F_B (along one run's trajectory only), accepted steps, seconds, RMS(u - u_0) over all nodes
and over the brain nodes (voxels: refused unless the spacing is 1 mm); accepted and rejected candidates; the JD scaling fractions; the 1-minute load at launch.

Probe verdict (BZ's accepted text; the plan): accepted < 10 or RMS < .1 voxel: 停滞, stop and hand to BZ; accepted 3-9 and RMS >= .03 too, with the words 动了但没到门槛,
BZ decides whether to go on; otherwise go.  A failed run: stop and hand to BZ.  The entry not evaluated with a passed certificate: 不可判, never go, stop and hand to the lead (review WARM_tools M2:
the lines of the five pairs read the end against the entry).  RMS in the bars is over all nodes, the measure of the bar's reference .40-.53 (the lead's revision
before any run; review X2 section 6.3, T4); the brain-node value stands next to it and decides nothing.

Readings of the five pairs (the plan's table, review X2 section 6.3; each line on its own; "reached" only with 5/5 comparable pairs, the entry and the end both evaluated with a
passed certificate, otherwise described only, review GATEA_tools I5; and with no stagnant pair: 1-2 stagnant pairs are also described only, because their differences near 0 would pull the
means inside the bars, review WARM_tools I1).  Means over the pairs; drops and gains are end against entry:
  at least 3 of the 5 pairs with accepted < 10 or RMS < .1 voxel     停滞：未决 (nothing else is read)
  cortex drop >= .063 or seg4 drop >= .026, that measure's drop > 0 on 5/5     F_B 下降把高 Dice 拉回去了：这个盆地里目标函数与 Dice 不一致
  cortex and white matter >= entry - .01, L_end < L_entry on 5/5     这个盆地里目标函数与 Dice 相容；从 u = 0 的路径到不了这里，路径是原因之一
  seg35 gain >= .01 with gain > 0 on 5/5, each large structure >= entry - .01     两段路径能同时拿到大结构与其余 29 个 (descriptive; goes to BZ as a route candidate)
  none of them     中间带，只描述
The bars (A1Ph - X2d)/2 are pinned (seg4 .026, cortex .063) and recomputed from the references (bars_check).
Not comparable: under support_shared sigma and v follow the entry state, so F_B and L of these runs are not the functions of X2d from u = 0: never compared across seeds
(C12-4a items 5, 8; plan I10).  Along one run they are.
"""
import argparse
from datetime import datetime, timezone
import glob
import importlib.util
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sx2 = _load("_warm_summarize_x2", "experiments/gate_a/summarize_x2.py")
ms, psum = sx2.ms, sx2.psum
read_json, finite, short_name, utc, fmt = sx2.read_json, sx2.finite, sx2.short_name, sx2.utc, sx2.fmt
COMPARABLE_ROLES = ms.COMPARABLE_ROLES

SCHEMA = "bayesreg.warm.summary.v1"
REQUIRED_N = 5                   # review GATEA_tools I5: "reached" needs every pair
STAGNANT_PAIRS = 3               # the plan: at least 3 of the 5 pairs
ACCEPTED_BAR, RMS_BAR = 10, .1
MOVED_ACCEPTED, MOVED_RMS = 3, .03
SEG4_DROP, CORTEX_DROP = .026, .063     # (A1Ph - X2d)/2 over the 5 pairs: .02600, .06285 from the stored metrics (review X2 section 6.3)
HOLD = .01
SEG35_GAIN = .01
BAR_TOLERANCE = 5e-4
SMALL_SIX = (4, 23, 18, 34, 19, 35)
LARGE_SIX = tuple(sx2.SEG35_LARGE)
OTHER_23 = tuple(k for k in range(1, 36) if k not in SMALL_SIX+LARGE_SIX)
GROUPS = dict(small_six=SMALL_SIX, large_six=LARGE_SIX, other_23=OTHER_23)
MEASURES = (("seg35", "seg35"), ("seg4", "seg4"), ("cortex", "皮层"), ("white_matter", "白质"), ("small_six", "小六个"), ("large_six", "大六个"), ("other_23", "其余 23 个"),
            ("hd95", "HD95"), ("sdlogj_official", "SDlogJ 官方"), ("sdlogj_brain", "SDlogJ 脑内"))
DICE = ("seg35", "seg4", "cortex", "white_matter", "small_six", "large_six", "other_23")
READING_STAGNANT = "停滞：未决"
READING_PULLED = "F_B 下降把高 Dice 拉回去了：这个盆地里目标函数与 Dice 不一致"
READING_COMPATIBLE = "这个盆地里目标函数与 Dice 相容；从 u = 0 的路径到不了这里，路径是原因之一"
READING_BOTH = "两段路径能同时拿到大结构与其余 29 个（描述；BZ 已接受种子进方法，这条读数交 BZ 作路线候选）"
READING_MIDDLE = "中间带，只描述"
READING_FEW = "只描述（可比配对不足 5/5）"
READING_SOME_STAGNANT = "只描述（{k}/{n} 对停滞）"
PROBE_GO = "动起来了：可以跑 5 对"
PROBE_STAGNANT = "停滞：停下交 BZ"
PROBE_MOVED = "停滞（动了但没到门槛）：停下交 BZ，由 BZ 决定是否继续"
PROBE_FAILED = "运行失败或没有评测：停下交 BZ"
PROBE_UNDECIDABLE = "不可判（入口没有评测到）：停下交总负责人"
NOTE = ("开发集（stage_L2R_val_v1），单次观测；n = 5 只有方向信息；读数不是结论。目标函数是 X2d 的诊断点，不是合同选出的 η（C04-6）；结果好要重做探针与扫描。种子是 A1Ph 的终态场（冻结配置 C + U-Net 预测上色，"
        "不读标签）。每次运行以开发开关 --wall-seconds 的“预算收尾”或其他 C13-3 收尾结束，不是“B 线收敛”，除非状态写明。"
        "support_shared 下 σ、v 由入口状态定，这组运行的 F_B、L 与 X2d（u = 0）不是同一个函数，不跨种子比（C12-4a 第 5、8 项）；沿同一次运行的轨迹可以比。")


class SummaryError(Exception):
    """The summary cannot be written as specified."""


def summary_name(phase, suffix):
    return f"summary_{phase}.{suffix}"


def measures_of(doc):
    """The measures of one evaluation record (bayesreg.eval or the D0 evaluator)."""
    m = (doc or {}).get("metrics") or {}
    out = {k: (m.get(k) if finite(m.get(k)) else None) for k in ("seg35", "seg4", "hd95", "sdlogj_official", "sdlogj_brain")}
    out.update(sx2.structures(doc))
    labels = sx2.seg_labels(doc, "seg35")
    for name, group in GROUPS.items():
        values = [labels.get(str(k)) for k in group]
        values = [v for v in values if finite(v)]
        out[name] = statistics.fmean(values) if values else None
    return out


def field_doc(root, aid, short, field):
    """The evaluator's record of an entry or checkpoint field, or None."""
    ev = Path(root)/"eval_fields"/aid/short/field
    found = sorted(glob.glob(str(ev/"*"/"metrics.json")))
    return read_json(found[0]) if (ev/"done.json").is_file() and len(found) == 1 else None


def state_row(name, doc, record, *, accepted, seconds, rms_all=None, rms_brain=None):
    """One state of a run's trajectory: the evaluation (None: not evaluated) with the run's own record of L, F_B and the certificate."""
    cert = (doc or {}).get("certificate") or {}
    comparable = bool(doc is not None and not doc.get("error") and doc.get("field_role") in COMPARABLE_ROLES and cert.get("passed") is True)
    record = record or {}
    return dict(state=name, evaluated=doc is not None, comparable=comparable, field_role=(doc or {}).get("field_role"), certificate=cert.get("status"),
                min_jd_lower_bound=cert.get("min_jd_lower_bound"), metrics=measures_of(doc), seg35_labels=sx2.seg_labels(doc, "seg35"), run_id=(doc or {}).get("run_id"),
                accepted=accepted, seconds=seconds, rms_all_voxels=rms_all, rms_brain_voxels=rms_brain, L=record.get("L"), F_B=record.get("F_B"), L_undefined=record.get("L_undefined"),
                lower_bound=record.get("min_jd_lower_bound"), state_hash=record.get("state_hash"))


def jd_scaling_of(hyper):
    """The JD scaling fractions of the native round (C09.1-3a), as level_table reads them."""
    rounds = (hyper or {}).get("rounds") or []
    js = ((rounds[-1].get("metric") or {}).get("jd_scaling") or {}) if rounds else {}
    return {k: js.get(k) for k in ("cell_estimate", "jd_scaling_refresh", "min_J_a", "max_fraction_r_lt_1")}


def run_row(root, aid, pair, batch):
    """The arm_row of summarize_x2 (final field, levels, counts) with the states of the run."""
    root, short = Path(root), short_name(pair)
    row = sx2.arm_row(root, aid, pair, batch)
    run = root/"runs"/aid/short
    hyper = read_json(run/"hyper.json") or {}
    profile = hyper.get("profile") or {}
    seed = hyper.get("init_field") or {}
    rounds = hyper.get("rounds") or []
    spacing = rounds[-1].get("spacing_xyz") if rounds else None
    entry, end = profile.get("entry"), profile.get("end")
    states = [state_row("entry", field_doc(root, aid, short, "entry"), entry, accepted=0, seconds=0., rms_all=0., rms_brain=0.)]
    for c in profile.get("checkpoints") or []:
        states.append(state_row(f"ckpt_{c['k']:03d}", field_doc(root, aid, short, f"ckpt_{c['k']:03d}"), c, accepted=c.get("accepted"), seconds=c.get("seconds_from_entry"),
                                rms_all=c.get("rms_from_entry_all_mm"), rms_brain=c.get("rms_from_entry_mm")))
    final_doc = read_json(row["eval_metrics"]) if row.get("eval_metrics") else None
    states.append(state_row("final", final_doc, end, accepted=(end or {}).get("accepted"), seconds=None, rms_all=(end or {}).get("rms_from_entry_all_mm"),
                            rms_brain=(end or {}).get("rms_from_entry_mm")))
    row.update(states=states, accepted_k=seed.get("accepted_k"), scale=seed.get("scale"), entry_state_hash=seed.get("entry_state_hash"), seed_state_hash=seed.get("state_hash"),
               seed_file_sha256=seed.get("file_sha256"), seed_oracle=seed.get("oracle"), profile_errors=profile.get("errors") or [], n_checkpoints=profile.get("n_checkpoints"),
               sigma2_used=(rounds[-1].get("sigma2_used") if rounds else None), sigma2_S_used=(rounds[-1].get("sigma2_S_used") if rounds else None),
               F_B_trace=(rounds[-1].get("F_B_trace") if rounds else None), jd_scaling=jd_scaling_of(hyper), spacing_xyz=spacing)
    if spacing is not None and any(s != 1. for s in spacing):
        raise SummaryError(f"{run}: spacing {spacing}: the RMS bar is in voxels and the records in mm (the plan's pairs have 1 mm)")
    row.update(moved=moved_of(row), differences=differences_of(states))
    return row


def moved_of(row):
    """Accepted steps and RMS over all nodes of one run against the bars (the brain-node RMS stands next to it, it decides nothing)."""
    end = next((s for s in row["states"] if s["state"] == "final"), {})
    accepted, rms = row.get("native_accepted"), end.get("rms_all_voxels")
    known = isinstance(accepted, int) and finite(rms)
    return dict(accepted=accepted, rms_all_voxels=rms, rms_brain_voxels=end.get("rms_brain_voxels"), known=known,
                stagnant=None if not known else bool(accepted < ACCEPTED_BAR or rms < RMS_BAR))


def differences_of(states):
    """End minus entry of every measure; None unless both are comparable."""
    entry, final = states[0], states[-1]
    if not (entry["comparable"] and final["comparable"]):
        return None
    out = {}
    for key, _ in MEASURES:
        a, b = entry["metrics"].get(key), final["metrics"].get(key)
        out[key] = b-a if finite(a) and finite(b) else None
    large = {}
    for label in LARGE_SIX:
        a, b = entry["seg35_labels"].get(str(label)), final["seg35_labels"].get(str(label))
        large[str(label)] = dict(entry=a, final=b) if finite(a) and finite(b) else None
    return dict(out, large_labels=large, L_entry=entry["L"], L_end=final["L"], F_B_entry=entry["F_B"], F_B_end=final["F_B"])


def probe_verdict(row):
    """The probe's verdict of its one run."""
    moved = row["moved"]
    record = dict(pair=row["pair"], accepted=moved["accepted"], rms_all_voxels=moved["rms_all_voxels"], rms_brain_voxels=moved["rms_brain_voxels"], run_status=row.get("run_status"),
                  bars=dict(accepted=ACCEPTED_BAR, rms_all_voxels=RMS_BAR, moved_accepted=[MOVED_ACCEPTED, ACCEPTED_BAR-1], moved_rms_all_voxels=MOVED_RMS))
    if row["state"] != "ran" or row.get("exit_code") != 0 or not moved["known"] or not row["states"][-1]["evaluated"]:
        return dict(record, label=PROBE_FAILED, go=False, moved_below_bar=False)
    if not row["states"][0].get("comparable"):        # review WARM_tools M2: the table reads the end against the entry
        return dict(record, label=PROBE_UNDECIDABLE, go=False, moved_below_bar=False)
    if not moved["stagnant"]:
        return dict(record, label=PROBE_GO, go=True, moved_below_bar=False)
    below = MOVED_ACCEPTED <= moved["accepted"] < ACCEPTED_BAR and moved["rms_all_voxels"] >= MOVED_RMS
    return dict(record, label=PROBE_MOVED if below else PROBE_STAGNANT, go=False, moved_below_bar=below)


def readings(rows):
    """The plan's table on the rows of the five pairs: each line with its numbers; `labels` the lines reached, in the table's order."""
    stagnant = [r["short"] for r in rows if r["moved"]["stagnant"]]
    lines = dict(stagnation=dict(reached=len(stagnant) >= STAGNANT_PAIRS, pairs=stagnant, n=len(stagnant), of=len(rows), rule=f"至少 {STAGNANT_PAIRS} 对：接受步 < {ACCEPTED_BAR} 或全节点 RMS < {RMS_BAR} 体素"))
    comparable = [r for r in rows if r["differences"] is not None]
    n = len(comparable)
    complete = n == REQUIRED_N == len(rows) and not stagnant      # review WARM_tools I1: a stagnant pair's difference near 0 must not dilute the means
    diffs = [r["differences"] for r in comparable]

    def mean_of(key):
        values = [d[key] for d in diffs if finite(d.get(key))]
        return statistics.fmean(values) if len(values) == n and n else None

    def count(key, sign):
        return sum(1 for d in diffs if finite(d.get(key)) and sign*d[key] > 0)
    cortex, seg4, wm, seg35 = (mean_of(k) for k in ("cortex", "seg4", "white_matter", "seg35"))
    falls = dict(cortex=dict(drop=None if cortex is None else -cortex, bar=CORTEX_DROP, same_sign=count("cortex", -1)),
                 seg4=dict(drop=None if seg4 is None else -seg4, bar=SEG4_DROP, same_sign=count("seg4", -1)))
    pulled = complete and any(v["drop"] is not None and v["drop"] >= v["bar"] and v["same_sign"] == n for v in falls.values())
    l_down = sum(1 for d in diffs if finite(d.get("L_entry")) and finite(d.get("L_end")) and d["L_end"] < d["L_entry"])
    compatible = complete and cortex is not None and wm is not None and cortex >= -HOLD and wm >= -HOLD and l_down == n
    large = {}
    for label in LARGE_SIX:
        pairs = [d["large_labels"].get(str(label)) for d in diffs]
        pairs = [p for p in pairs if p]
        large[str(label)] = dict(name=sx2.SEG35_LARGE[label], gain=statistics.fmean(p["final"]-p["entry"] for p in pairs) if len(pairs) == n and n else None)
    large_hold = all(v["gain"] is not None and v["gain"] >= -HOLD for v in large.values())
    both = complete and seg35 is not None and seg35 >= SEG35_GAIN and count("seg35", 1) == n and large_hold
    lines.update(pull_back=dict(reached=pulled, falls=falls, n=n),
                 compatible=dict(reached=compatible, cortex_gain=cortex, white_matter_gain=wm, bar=-HOLD, L_down=l_down, n=n),
                 both_paths=dict(reached=both, seg35_gain=seg35, seg35_bar=SEG35_GAIN, seg35_positive=count("seg35", 1), large=large, large_bar=-HOLD, n=n))
    if lines["stagnation"]["reached"]:
        labels = [READING_STAGNANT]
    elif stagnant:
        labels = [READING_SOME_STAGNANT.format(k=len(stagnant), n=len(rows))]
    elif not complete:
        labels = [READING_FEW]
    else:
        labels = [text for key, text in (("pull_back", READING_PULLED), ("compatible", READING_COMPATIBLE), ("both_paths", READING_BOTH)) if lines[key]["reached"]] or [READING_MIDDLE]
    return dict(lines=lines, labels=labels, comparable=n, required=REQUIRED_N)


def state_means(rows):
    """Means of the measures by state over the pairs that have it: entry, ckpt_kkk, final, each with its n."""
    names = []
    for r in rows:
        for s in r["states"]:
            if s["state"] not in names:
                names.append(s["state"])
    names.sort(key=lambda s: (s == "final", s != "entry", s))
    out = {}
    for name in names:
        states = [s for r in rows for s in r["states"] if s["state"] == name and s["comparable"]]
        out[name] = dict(n=len(states), accepted=_mean(s["accepted"] for s in states), seconds=_mean(s["seconds"] for s in states), rms_all_voxels=_mean(s["rms_all_voxels"] for s in states),
                         rms_brain_voxels=_mean(s["rms_brain_voxels"] for s in states),
                         **{k: _mean(s["metrics"].get(k) for s in states) for k, _ in MEASURES})
    return out


def _mean(values):
    values = [v for v in values if finite(v)]
    return statistics.fmean(values) if values else None


def reference_rows(batch):
    """A1Ph (the seed's own record) and X2d from u = 0 on the batch's pairs: Dice only, read only."""
    seeds, x2_root = Path(batch["seed_root"]), Path(batch["x2_root"])
    rows = dict(A1Ph=[], X2d=[])
    for pair in batch["summary_pairs"]:
        short = short_name(pair)
        rows["A1Ph"].append(dict(sx2.reference_row(read_json(seeds/pair/"metrics.json"), pair), measures=measures_of(read_json(seeds/pair/"metrics.json"))))
        doc = read_json(x2_root/"eval"/"d"/short/f"gax2_d_{short}"/"metrics.json")
        rows["X2d"].append(dict(sx2.reference_row(doc, pair), measures=measures_of(doc)))
    return rows


def reference_means(rows):
    return {k: _mean(r["measures"].get(k) for r in rows if r["comparable"]) for k, _ in MEASURES}


def bars_check(refs):
    """The pinned bars against (A1Ph - X2d)/2 over the batch's pairs (only meaningful for the five pairs)."""
    a, b = reference_means(refs["A1Ph"]), reference_means(refs["X2d"])
    out = {}
    for key, bar in (("seg4", SEG4_DROP), ("cortex", CORTEX_DROP)):
        half = (a[key]-b[key])/2 if finite(a[key]) and finite(b[key]) else None
        out[key] = dict(pinned=bar, recomputed=half, agrees=None if half is None else abs(half-bar) <= BAR_TOLERANCE)
    return out


def seed_check(rows, refs):
    """The entry's Dice against the seed's own record (accepted_k 0: the same field; two evaluators, one definition): the largest difference; only a check."""
    out = []
    for row, ref in zip(rows, refs["A1Ph"]):
        entry = row["states"][0]["metrics"]
        gaps = {k: abs(entry[k]-ref["measures"][k]) for k in ("seg35", "seg4") if finite(entry.get(k)) and finite(ref["measures"].get(k))}
        out.append(dict(short=row["short"], accepted_k=row.get("accepted_k"), scale=row.get("scale"), max_gap=max(gaps.values()) if gaps else None,
                        entry_is_seed=row.get("entry_state_hash") == row.get("seed_state_hash") if row.get("entry_state_hash") else None))
    return out


def summary(root, phase):
    root = Path(root)
    batch = read_json(root/f"batch_{phase}.json")
    if batch is None:
        raise SummaryError(f"{root}/batch_{phase}.json missing (the driver writes it at its start)")
    aid = batch["arm_order"][0]
    rows = [run_row(root, aid, p, batch) for p in batch["summary_pairs"]]
    refs = reference_rows(batch)
    doc = dict(schema=SCHEMA, note=NOTE, written_utc=utc(), root=str(root), phase=phase, stage=batch["stage"], pairs=batch["summary_pairs"], launched_by_this_phase=batch["pairs"], probe_pair=batch["probe_pair"], arm=aid, arm_record={
                   k: batch["arms"][aid][k] for k in ("label", "note", "channel_source", "sigma_domain", "support_threshold", "eta_role", "eta", "eta_file")},
               code_sha=batch["code_sha"], wall_seconds=batch["wall_seconds"], checkpoint_seconds=batch["checkpoint_seconds"], dry_run=bool(batch.get("dry_run")),
               probe_gate=batch.get("probe_gate"), bars=dict(accepted=ACCEPTED_BAR, rms_all_voxels=RMS_BAR, moved_accepted=MOVED_ACCEPTED, moved_rms_all_voxels=MOVED_RMS, seg4_drop=SEG4_DROP,
                                                               cortex_drop=CORTEX_DROP, hold=HOLD, seg35_gain=SEG35_GAIN, required_n=REQUIRED_N, stagnant_pairs=STAGNANT_PAIRS),
               rows=rows, state_means=state_means(rows), references=refs, reference_means={k: reference_means(v) for k, v in refs.items()}, seed_check=seed_check(rows, refs),
               counts=dict(ran=sum(r["state"] == "ran" for r in rows), not_run=sum(r["state"] == "not_run" for r in rows), failed_no_status=sum(r["state"] == "failed_no_status" for r in rows),
                           evaluator_certificate_passed=sum((r["certificate"] or {}).get("passed") is True for r in rows), args_mismatch=sum(bool(r["args_differ"]) for r in rows),
                           eta_mismatch=sum(r["eta_ok"] is False for r in rows), record_errors=sum(bool(r["profile_errors"]) for r in rows),
                           fields_not_comparable=[f"{r['short']}/{s['state']}" for r in rows for s in r["states"] if not s["comparable"]]))
    if phase == "probe":
        doc["probe"] = probe_verdict(rows[0]) if rows else dict(label=PROBE_FAILED, go=False, moved_below_bar=False, accepted=None, rms_all_voxels=None, rms_brain_voxels=None)
    else:
        doc["bars_check"] = bars_check(refs)
        doc["readings"] = readings(rows)
    return doc


def _reading_rows(doc):
    lines = doc["readings"]["lines"]
    stag, pull, comp, both = lines["stagnation"], lines["pull_back"], lines["compatible"], lines["both_paths"]
    falls = pull["falls"]
    yes = (lambda v: "是" if v else "否")
    return [f"| 接受步 < {ACCEPTED_BAR} 或全节点 RMS < {RMS_BAR} 体素，至少 {STAGNANT_PAIRS} 对 | {stag['n']}/{stag['of']}（{', '.join(stag['pairs']) or '无'}） | {yes(stag['reached'])} | {READING_STAGNANT} |",
            f"| 皮层回落 ≥ {CORTEX_DROP} 或 seg4 回落 ≥ {SEG4_DROP}（5/5 同向） | 皮层 {fmt(falls['cortex']['drop'])}（{falls['cortex']['same_sign']}/{pull['n']} 回落），seg4 {fmt(falls['seg4']['drop'])}（{falls['seg4']['same_sign']}/{pull['n']} 回落） | {yes(pull['reached'])} | {READING_PULLED} |",
            f"| 皮层、白质 ≥ 入口 − {HOLD}，L 继续下降（5/5） | 皮层 {fmt(comp['cortex_gain'])}，白质 {fmt(comp['white_matter_gain'])}，L 下降 {comp['L_down']}/{comp['n']} | {yes(comp['reached'])} | {READING_COMPATIBLE} |",
            f"| seg35 ≥ 入口 + {SEG35_GAIN}（5/5），六个大结构各 ≥ 入口 − {HOLD} | seg35 {fmt(both['seg35_gain'])}（{both['seg35_positive']}/{both['n']} 为正），大结构最低 {fmt(min((v['gain'] for v in both['large'].values() if finite(v['gain'])), default=None))} | {yes(both['reached'])} | {READING_BOTH} |"]


def markdown(doc):
    lines = [f"# WARM 汇总：{'探针' if doc['phase'] == 'probe' else '前 5 对'}", "", doc["note"], "",
             f"代码 {doc['code_sha']}；wall {doc['wall_seconds']:g} s，检查点 {doc['checkpoint_seconds']:g} s；目标：{doc['arm_record']['note']}。η {doc['arm_record']['eta']}（{doc['arm_record']['eta_file']}）。", ""]
    if doc["phase"] == "probe":
        v = doc["probe"]
        lines += ["## 探针判定", "", f"**{v['label']}**（配对 {short_name(v['pair']) if v.get('pair') else '—'}：接受步 {v.get('accepted')}，全节点 RMS {fmt(v.get('rms_all_voxels'))} 体素；脑内节点 {fmt(v.get('rms_brain_voxels'))}，不进判定）。", "",
                  f"门槛：接受步 < {ACCEPTED_BAR} 或 RMS < {RMS_BAR} 体素记停滞；接受步 {MOVED_ACCEPTED}–{ACCEPTED_BAR-1} 且 RMS ≥ {MOVED_RMS} 单独写明\"动了但没到门槛\"，由 BZ 决定是否继续。RMS 是全节点，与门槛 .1 的参考 .40–.53 同一口径（复核 T4）。这次运行就是这一对的结果，`--phase pairs` 不重跑它。", ""]
    else:
        r = doc["readings"]
        lines += ["## 判定（读数，不是结论；n = 5）", "", f"**{'；'.join(r['labels'])}**（可比配对 {r['comparable']}/{r['required']}；{short_name(doc['probe_pair'])} 是探针那次运行，不重跑）", "", "| 条件 | 实测 | 成立 | 记为 |", "|---|---|---|---|", *_reading_rows(doc), ""]
        check = doc["bars_check"]
        lines += [f"回落门槛 = (A1Ph − X2d)/2：seg4 {check['seg4']['pinned']}（重算 {fmt(check['seg4']['recomputed'])}），皮层 {check['cortex']['pinned']}（重算 {fmt(check['cortex']['recomputed'])}）。", ""]
    header = "| 状态 | n | 接受步 | 秒 | RMS 全节点（体素） | RMS 脑内（体素） | " + " | ".join(t for _, t in MEASURES) + " |"
    lines += ["## 入口、检查点与终态（可比配对的均值）", "", header, "|" + "---|"*(6+len(MEASURES))]
    for name, row in doc["state_means"].items():
        lines.append(f"| {name} | {row['n']} | {fmt(row['accepted'])} | {fmt(row['seconds'])} | {fmt(row['rms_all_voxels'])} | {fmt(row['rms_brain_voxels'])} | " + " | ".join(fmt(row[k]) for k, _ in MEASURES) + " |")
    lines += ["", "## 逐对", "", "| 配对 | 收尾 | 原生接受 | 原生拒绝 | k | 检查点 | RMS 全节点 | RMS 脑内 | 入口 L | 末 L | 入口 F_B | 末 F_B | r < 1 的比例（最大） | 启动负载 |", "|" + "---|"*14]
    for r in doc["rows"]:
        entry, final = r["states"][0], r["states"][-1]
        counts = r["counts"]["native"]
        load = r.get("launch_loadavg")
        lines.append(f"| {r['short']} | {r.get('wording')} | {counts['accepted']} | {counts['rejected']} | {r.get('accepted_k')} | {r.get('n_checkpoints')} | {fmt(r['moved']['rms_all_voxels'])} | "
                     f"{fmt(r['moved']['rms_brain_voxels'])} | {fmt(entry['L'])} | {fmt(final['L'])} | {fmt(entry['F_B'])} | {fmt(final['F_B'])} | {fmt(r['jd_scaling'].get('max_fraction_r_lt_1'))} | {fmt(load[0]) if load else '—'} |")
    lines += ["", "沿同一次运行才比 L、F_B；不跨种子，也不与 X2d（u = 0）比。", "", "## 每个状态", "", "| 配对 | 状态 | 证书 | 接受步 | 秒 | RMS 全节点 | L | F_B | " + " | ".join(t for _, t in MEASURES) + " |", "|" + "---|"*(8+len(MEASURES))]
    for r in doc["rows"]:
        for s in r["states"]:
            lines.append(f"| {r['short']} | {s['state']} | {s['certificate'] or '未评测'} | {fmt(s['accepted'])} | {fmt(s['seconds'])} | {fmt(s['rms_all_voxels'])} | {fmt(s['L'])} | {fmt(s['F_B'])} | " + " | ".join(fmt(s['metrics'].get(k)) for k, _ in MEASURES) + " |")
    means = doc["reference_means"]
    lines += ["", "## 旁证（只比 Dice，均值，本批配对）", "", "| 来源 | " + " | ".join(t for _, t in MEASURES) + " |", "|" + "---|"*(1+len(MEASURES))]
    for name in ("A1Ph", "X2d"):
        lines.append(f"| {name} | " + " | ".join(fmt(means[name][k]) for k, _ in MEASURES) + " |")
    lines += ["", "入口与种子自己的记录（k = 0 时是同一个场，两个评测入口同一个定义；只是核对）：" + "；".join(f"{c['short']} k = {c['accepted_k']}，最大差 {fmt(c['max_gap'])}" for c in doc["seed_check"]) + "。", ""]
    c = doc["counts"]
    lines += ["## 核对", "", f"已运行 {c['ran']}，未运行 {c['not_run']}，失败（无 status.json）{c['failed_no_status']}；评测证书通过 {c['evaluator_certificate_passed']}；参数不符 {c['args_mismatch']}，η 不符 {c['eta_mismatch']}；记录出错的运行 {c['record_errors']}；不可比的状态：{', '.join(c['fields_not_comparable']) or '无'}。", ""]
    for r in doc["rows"]:
        if r["profile_errors"]:
            lines.append(f"- {r['short']} 记录错误（求解不受影响）：{r['profile_errors']}")
    return "\n".join(lines)+"\n"


def strip_volatile(doc):
    return {k: v for k, v in doc.items() if k != "written_utc"}


def write(root, phase):
    """summary_<phase>.json / .md once; a different recomputation goes to summary_<phase>_<UTC>.*; returns the paths and the state."""
    root = Path(root)
    doc = summary(root, phase)
    text = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False)+"\n"
    stored = read_json(root/summary_name(phase, "json"))
    if stored is None:
        target_json, target_md = root/summary_name(phase, "json"), root/summary_name(phase, "md")
    elif json.loads(json.dumps(strip_volatile(doc))) == strip_volatile(stored):
        return root/summary_name(phase, "json"), root/summary_name(phase, "md"), "unchanged"
    else:
        s = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target_json, target_md = root/f"summary_{phase}_{s}.json", root/f"summary_{phase}_{s}.md"
    with open(target_json, "x", encoding="utf-8") as stream:
        stream.write(text)
    with open(target_md, "x", encoding="utf-8") as stream:
        stream.write(markdown(doc))
    return target_json, target_md, "new" if stored is None else "differs (new file; the earlier one kept)"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -B experiments/gate_a/summarize_warm.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("probe", "pairs"))
    args = parser.parse_args(argv)
    try:
        js, md, state = write(args.root, args.phase)
    except SummaryError as exc:
        print(f"summary: {exc}", file=sys.stderr)
        return 8
    print(f"summary: {js} {md} ({state})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
