"""SEGP P1: the mixed ensemble of unet35_v1_ens and unet35_v2_full (runs/20261105-segprobe/任务书.md). Segmentation level, 20 development subjects, single observation.

Arms per subject: v1, v2, mix = argmax of the equal-weight mean of the two posteriors (no search over weights). Evaluated by experiments/unv2/eval_seg.py summarize_subject: seg35,
seg4, the small six, the big six, the other 23; cortex and white matter both as seg4 classes and as seg35 labels. Control: v1 and v2 recomputed here, subject by subject and metric by
metric, must equal the committed eval_seg.json (tolerance 1e-12, any departure stops the run before the mix is reported). The truth is read for this evaluation only.

    python -B experiments/segprobe/p1_mix.py --out-dir DIR
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segprobe_common as common  # noqa: E402
from segprobe_common import eval_seg  # noqa: E402

FORMAT = "bayesreg.segprobe.p1.v1"
ARMS = ("v1", "v2", "mix")
CONTROL_TOL = 1e-12
GROUP_METRICS = tuple(eval_seg.GROUPS)
TISSUE_METRICS = ("seg4_cortex", "seg4_wm", "seg35_cortex_L", "seg35_cortex_R", "seg35_wm_L", "seg35_wm_R")
METRICS = ("seg35", "seg4")+GROUP_METRICS+TISSUE_METRICS


def row_of(pred35, true35, true4, lookup):
    """eval_seg.summarize_subject plus the two tissues read from the same per-label arrays (labels 1..n are at index l - 1)."""
    row = eval_seg.summarize_subject(pred35, true35, true4, lookup)
    row["seg4_cortex"], row["seg4_wm"] = row["seg4_per_label"][common.SEG4_CORTEX-1], row["seg4_per_label"][common.SEG4_WM-1]
    row["seg35_cortex_L"], row["seg35_cortex_R"] = (row["per_label"][l-1] for l in common.CORTEX_LABELS)
    row["seg35_wm_L"], row["seg35_wm_R"] = (row["per_label"][l-1] for l in common.WM_LABELS)
    return row


def value(row, metric):
    return row["groups"][metric] if metric in GROUP_METRICS else row[metric]


def control(rows, reference):
    """v1 and v2 of this run against the committed eval_seg.json: the largest absolute difference over subjects and metrics; ValueError above CONTROL_TOL."""
    worst = 0.
    for subject, row in rows.items():
        for arm in ("v1", "v2"):
            old = reference["rows"][subject][arm]
            pairs = [(row[arm]["seg35"], old["seg35"]), (row[arm]["seg4"], old["seg4"])]
            pairs += [(row[arm]["groups"][g], old["groups"][g]) for g in GROUP_METRICS]
            pairs += list(zip(row[arm]["per_label"], old["per_label"]))
            for new, committed in pairs:
                if new is None or committed is None or not np.isfinite(new) or not np.isfinite(committed):
                    if (new is None or not np.isfinite(new)) != (committed is None or not np.isfinite(committed)):
                        raise ValueError(f"{subject} {arm}: defined here and undefined in the reference, or the reverse")
                    continue
                worst = max(worst, abs(new-committed))
    if not worst <= CONTROL_TOL:
        raise ValueError(f"the control failed: v1 or v2 recomputed differs from {common.REFERENCE} by {worst}")
    return worst


def summary_markdown(document):
    means, paired = document["means"], document["paired"]
    f = eval_seg.fmt
    lines = ["# SEGP P1：v1 + v2 混合（开发集 20 人，单次观测；分割级，不是配准结果）", "", "等权平均后 argmax，没有搜权重。这个臂看过全部 20 人，带选择效应。", "",
             "| 臂 | "+" | ".join(METRICS)+" |", "|---|"+"---|"*len(METRICS)]
    lines += [f"| {arm} | "+" | ".join(f(means[arm][m]) for m in METRICS)+" |" for arm in ARMS]
    lines += ["", "| 比较 | 指标 | 均值差 | 中位差 | 正 | 负 | 零 |", "|---|---|---|---|---|---|---|"]
    for name, comparison in paired.items():
        lines += [f"| {name} | {m} | {f(p['mean'], 5)} | {f(p['median'], 5)} | {p['n_positive']} | {p['n_negative']} | {p['n_zero']} |" for m, p in comparison.items()]
    lines += ["", f"对照：v1、v2 重算与 {document['reference']} 的最大绝对差 {document['control_worst_abs_diff']:.3e}。", ""]
    return "\n".join(lines)


def run(*, out_dir, argv=None, log=print, v1_root=common.V1_ROOT, v2_root=common.V2_ROOT, reference_path=common.REFERENCE, data_root=eval_seg.DATA_ROOT, admit_kw=None):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists and is not empty; results are never overwritten")
    subjects, admission = common.admit_dev_subjects(data_root=data_root, **(admit_kw or {}))        # before any file of a subject is opened
    reference = json.loads(Path(reference_path).read_text())
    lookup = eval_seg.load_seg4_lookup()
    rows, records, disagreement = {}, {}, {}
    started = time.perf_counter()
    for subject in subjects:
        raw1, record1 = common.load_raw(v1_root, subject, expected_segmenter=common.V1_ID)
        raw2, record2 = common.load_raw(v2_root, subject, expected_segmenter=common.V2_ID)
        if record1["image_sha256"] != record2["image_sha256"]:
            raise ValueError(f"{subject}: the predictions were made from different images")
        predictions = dict(v1=common.argmax_of_mean([raw1]), v2=common.argmax_of_mean([raw2]), mix=common.argmax_of_mean([raw1, raw2]))
        del raw1, raw2
        true35, sha35 = eval_seg.read_truth(data_root, subject, "seg35")
        true4, sha4 = eval_seg.read_truth(data_root, subject, "seg4")
        rows[subject] = {arm: row_of(predictions[arm], true35, true4, lookup) for arm in ARMS}
        records[subject] = dict(v1=record1, v2=record2, truth_sha256=dict(seg35=sha35, seg4=sha4))
        foreground = (true35 > 0) | (predictions["v1"] > 0) | (predictions["v2"] > 0)
        disagreement[subject] = dict(voxels_v1_ne_v2=int((predictions["v1"] != predictions["v2"]).sum()), foreground_voxels=int(foreground.sum()),
                                     fraction_of_foreground=float((predictions["v1"] != predictions["v2"]).sum()/foreground.sum()))
        log(f"{subject}: "+", ".join(f"{a} {rows[subject][a]['seg35']:.4f} / {rows[subject][a]['seg4']:.4f}" for a in ARMS)+f", peak RSS {common.peak_rss_gib():.2f} GiB, {time.perf_counter()-started:.0f} s", flush=True)
    worst = control(rows, reference)
    log(f"control passed: largest |difference| {worst:.3e}")

    def column(arm, metric):
        return [value(rows[s][arm], metric) for s in subjects]

    means = {arm: {m: eval_seg.nanmean(column(arm, m)) for m in METRICS} for arm in ARMS}
    comparisons = dict(mix_minus_v1=("mix", "v1"), mix_minus_v2=("mix", "v2"), v2_minus_v1=("v2", "v1"))
    paired = {name: {m: eval_seg.paired(column(a, m), column(b, m)) for m in METRICS} for name, (a, b) in comparisons.items()}
    document = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=argv, admission=admission, subjects=subjects, roots=dict(v1=str(v1_root), v2=str(v2_root)),
                    reference=str(reference_path), reference_sha256=eval_seg.sha_file(reference_path), control_worst_abs_diff=worst, control_tol=CONTROL_TOL, means=means, paired=paired,
                    per_label_mean={a: [eval_seg.nanmean([rows[s][a]["per_label"][k] for s in subjects]) for k in range(eval_seg.N_CLASSES-1)] for a in ARMS},
                    disagreement=disagreement, records=records, rows=rows, seconds=time.perf_counter()-started, peak_rss_gib=common.peak_rss_gib(),
                    note="开发集，单次观测；分割级读数，不是配准结果。真值只在本评测读取。这个臂看过全部 20 人，带选择效应。")
    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(out_dir/"p1.json", document)
    with (out_dir/"per_subject.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "arm", *METRICS])
        for subject in subjects:
            for arm in ARMS:
                writer.writerow([subject, arm]+[value(rows[subject][arm], m) for m in METRICS])
    with (out_dir/"per_label.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "group", *[f"mean_{a}" for a in ARMS], "mix_minus_v1_mean", "mix_minus_v1_n_positive", "mix_minus_v2_mean", "mix_minus_v2_n_positive"])
        for index, label in enumerate(range(1, eval_seg.N_CLASSES)):
            group = next(name for name, labels in eval_seg.GROUPS.items() if label in labels)
            diff = {b: eval_seg.paired([rows[s]["mix"]["per_label"][index] for s in subjects], [rows[s][b]["per_label"][index] for s in subjects]) for b in ("v1", "v2")}
            writer.writerow([label, group, *[document["per_label_mean"][a][index] for a in ARMS], diff["v1"]["mean"], diff["v1"]["n_positive"], diff["v2"]["mean"], diff["v2"]["n_positive"]])
    (out_dir/"summary.md").write_text(summary_markdown(document), encoding="utf-8")
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=common.OUT_ROOT/"p1")
    args = parser.parse_args(argv)
    document = run(out_dir=args.out_dir, argv=sys.argv if argv is None else ["p1_mix.py", *argv])
    print(json.dumps(dict(means=document["means"], mix_minus_v1=document["paired"]["mix_minus_v1"]["seg35"], mix_minus_v2=document["paired"]["mix_minus_v2"]["seg35"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
