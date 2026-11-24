"""The reading of work package NN (runs/20261105-nn/plan.md, by code; a label for the lead and BZ, not a conclusion): decision_nn.json and summary_nn.md, never overwritten.

Delta = the mean over the 20 development subjects of the nnU-Net model (one fold-all model) minus the ensemble of v2 (.9275 / .9591 in the brief; the committed per-subject numbers, unrounded), seg35. The bars are included.
  Delta >= +.005            clearly better: connect it to the registration path (it must be registered first, hard rule 7); after 250 epochs also worth considering the full training
  Delta <= +.002            not better than the present ensemble
  otherwise                 middle band
Not decidable: no evaluation, not 20 subjects, the evaluator's control does not reproduce the reference (|mean difference| > the tolerance of eval_nn), a metric undefined, no training or inference record, a model that is not fold all, 3d_fullres, a trainer of this plan, checkpoint_final, or an inference that was not one view (allowed_mirroring_axes not None, tta_views not 1; plan.md, revision 1).
Everything else is described: seg4, the small six, the big six, the other 23, cortex, white matter; against v1 and U1 of UL; the place in the table of SV ([model]: .87 needs about .947, .88 about .96-.965); subject 0440; seconds per subject; the training time. One model, no mirror TTA, one evaluation on the development set, recipe not changed after seeing results; not a hold-out result, not a registration result.

    python -B experiments/nn/decide_nn.py --eval FILE --train-record FILE --infer-record FILE --out-dir DIR [--u1-eval FILE]
"""
import argparse
import json
import math
from pathlib import Path
import sys

import nn_common

SCHEMA = "bayesreg.nn.decision.v1"
BETTER_BAR, NOT_BETTER_BAR = .005, .002
N_SUBJECTS = 20
BETTER, NOT_BETTER, MIDDLE, NOT_DECIDABLE = "clearly_better", "not_better", "middle", "not_decidable"
READINGS = {BETTER: "明显更好，接到配准路径（要先登记，硬约束 7）", NOT_BETTER: "不比现在的集成好", MIDDLE: "中间带", NOT_DECIDABLE: "不可判"}
FULL_TRAINING = "；250 epoch，值得考虑训满"
SV_BARS = ((.87, .947, .947), (.88, .96, .965))         # runs/20261104-sv/result.md, [model]: the 19-pair seg35 goal, the segmentation-level seg35 it needs (low, high)
METRICS = ("seg35", "seg4", "small_six", "big_six", "other_23", "cortex", "white_matter")
NAMES = dict(seg35="seg35", seg4="seg4", small_six="小六个", big_six="大六个", other_23="其余 23 个", cortex="皮层（seg4）", white_matter="白质（seg4）")
NOTE = ("开发集 20 名受试者，单个 fold all 模型（106 名训练受试者全部参与训练，没有留标定集），单次观测，分割级读数，不是配准结果，也不是留出集结果。Δ 是单个模型对四成员集成。没有镜像 TTA：trainer 不做镜像增强（32/35 个标签分左右，见 plan.md）。开发集只读这一次，不看结果后调配方。"
        "它证伪不了什么：不说明训满或集成能到多少；分割级不等于配准后的 Dice（斜率约 .66–.70，runs/20261104-sv/result.md）。")


def reading_of(delta):
    """The label of a delta of seg35 (bars included)."""
    if delta is None or not math.isfinite(delta):
        return NOT_DECIDABLE
    if delta >= BETTER_BAR:
        return BETTER
    if delta <= NOT_BETTER_BAR:
        return NOT_BETTER
    return MIDDLE


def reading_text(reading, epochs):
    return READINGS[reading]+(FULL_TRAINING if reading == BETTER and epochs == nn_common.EPOCHS_LONG else "")


def epochs_of(trainer):
    return next((epochs for epochs, name in nn_common.TRAINERS.items() if name == trainer), None)


def not_decidable_reasons(evaluation, train_record, infer_record):
    """Why the rule cannot be read (empty: it can). All of it follows from the three files: there is no option that adds a reason."""
    out = []
    if evaluation is None:
        return out+["没有评测结果"]
    if train_record is None:
        out.append("没有 train_record.json")
    elif train_record.get("trainer") not in nn_common.TRAINERS.values():
        out.append(f"训练记录的 trainer 是 {train_record.get('trainer')!r}，不是本计划的")
    if infer_record is None:
        out.append("没有 infer_record.json")
    else:
        if infer_record.get("trainer") not in nn_common.TRAINERS.values():
            out.append(f"推断记录的 trainer 是 {infer_record.get('trainer')!r}，不是本计划的")
        if train_record is not None and infer_record.get("trainer") != train_record.get("trainer"):
            out.append("训练与推断的 trainer 不同")
        if infer_record.get("fold") != nn_common.FOLD or infer_record.get("checkpoint") != nn_common.CHECKPOINT or infer_record.get("configuration") != nn_common.CONFIGURATION:
            out.append(f"模型不是 {nn_common.CONFIGURATION}、fold {nn_common.FOLD}、{nn_common.CHECKPOINT}")
        if train_record is not None and train_record.get("final_checkpoint_sha256") != infer_record.get("checkpoint_sha256"):
            out.append("推断用的权重与训练记录的 checkpoint_final 不同")
        if infer_record.get("allowed_mirroring_axes") is not None or infer_record.get("tta_views") != 1:
            out.append(f"推断不是一次前向：镜像轴 {infer_record.get('allowed_mirroring_axes')}，tta_views {infer_record.get('tta_views')}；本计划没有 TTA")
    if len(evaluation.get("subjects", [])) != N_SUBJECTS:
        out.append(f"{len(evaluation.get('subjects', []))} 名受试者，不是 {N_SUBJECTS}")
    means = evaluation.get("means", {})
    for arm in ("nn", "reference_v2"):
        for metric in ("seg35", "seg4"):
            value = means.get(arm, {}).get(metric)
            if value is None or not math.isfinite(value):
                out.append(f"{arm} 的 {metric} 没有定义")
    control = evaluation.get("control", {})
    if control.get("passed") is not True:
        out.append(f"控制不符：v2 集成经本评测与已提交的数差 {control.get('differences')}，容差 {control.get('tolerance')}")
    return out


def sv_place(seg35):
    """Where a segmentation-level seg35 stands in the table of SV. Described only, [model]."""
    return [dict(goal=goal, needs_low=low, needs_high=high, at_or_above_low=seg35 >= low, at_or_above_high=seg35 >= high) for goal, low, high in SV_BARS]


def decide(evaluation, train_record, infer_record, *, u1_means=None):
    problems = not_decidable_reasons(evaluation, train_record, infer_record)
    trainer = (train_record or {}).get("trainer")
    epochs = epochs_of(trainer)
    training = {k: (train_record or {}).get(k) for k in ("trainer", "epochs", "wall_seconds", "epochs_logged", "epoch_seconds_median", "host_peak_pss_gib", "gpu_peak_used_mib", "gpu_mean_utilization_percent", "stopped_before_final_validation", "resumed_runs", "runs_without_record")}
    inference = {k: (infer_record or {}).get(k) for k in ("trainer", "allowed_mirroring_axes", "use_mirroring", "tta_views", "seconds_median", "seconds_max", "seconds_first", "gpu_peak_allocated_gib", "patch_size")}
    document = dict(schema=SCHEMA, rule=dict(clearly_better=f"delta seg35 >= +{BETTER_BAR}", not_better=f"delta seg35 <= +{NOT_BETTER_BAR}", middle="otherwise", reference="unet35_v2_full ensemble, eval_seg.json of the UNV2 chain"), training=training, inference=inference, dev_read=(evaluation or {}).get("dev_read"), note=NOTE)
    if problems:
        return dict(document, reading=NOT_DECIDABLE, reading_text=READINGS[NOT_DECIDABLE], not_decidable_because=problems, delta_seg35=None, delta_seg4=None)
    means = evaluation["means"]
    nn, v2, v1 = means["nn"], means["reference_v2"], means["reference_v1"]
    delta = nn["seg35"]-v2["seg35"]
    reading = reading_of(delta)
    document.update(reading=reading, reading_text=reading_text(reading, epochs), not_decidable_because=[], delta_seg35=delta, delta_seg4=nn["seg4"]-v2["seg4"], means=means, paired=evaluation["paired"], control=evaluation["control"], focus=evaluation.get("focus"), against_v1={m: nn[m]-v1[m] for m in METRICS}, against_v2={m: nn[m]-v2[m] for m in METRICS},
                    against_u1=None if u1_means is None else dict(means=u1_means, seg35=nn["seg35"]-u1_means["seg35"], seg4=nn["seg4"]-u1_means["seg4"]), sv_place=sv_place(nn["seg35"]))
    return document


def fmt(value, digits=4):
    return "n/a" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.{digits}f}"


def signed(value, digits=4):
    return "n/a" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:+.{digits}f}"


def summary_markdown(document):
    lines = ["# NN：nnU-Net v2（一个 fold all 模型）对 unet35_v2_full 集成（开发集，单次观测，分割级）", "", f"**读数（由代码判）：{document['reading_text']}**", ""]
    training, inference = document["training"], document["inference"]
    if document["reading"] == NOT_DECIDABLE:
        lines += ["不可判的原因：", ""]+[f"- {reason}" for reason in document["not_decidable_because"]]+[""]
    else:
        lines += [f"Δ = nnU-Net − v2 集成：seg35 {signed(document['delta_seg35'], 5)}，seg4 {signed(document['delta_seg4'], 5)}。规则：seg35 ≥ +{BETTER_BAR} 明显更好；≤ +{NOT_BETTER_BAR} 不比现在的集成好；其余中间带。", "",
                  "| 臂 | "+" | ".join(NAMES[m] for m in METRICS)+" |", "|---|"+"---|"*len(METRICS)]
        for arm, name in (("nn", "nnU-Net"), ("v2_control", "v2 集成经本评测（控制）"), ("reference_v2", "v2 集成（已提交的数）"), ("reference_v1", "v1 集成（已提交的数）")):
            lines.append(f"| {name} | "+" | ".join(fmt(document["means"][arm][m]) for m in METRICS)+" |")
        lines += ["", "逐受试者配对差（nnU-Net − v2 集成；均值、中位、正 / 负 / 零）：", "", "| 指标 | 均值差 | 中位差 | 正 | 负 | 零 |", "|---|---|---|---|---|---|"]
        for metric in METRICS:
            p = document["paired"]["nn_minus_reference_v2"][metric]
            lines.append(f"| {NAMES[metric]} | {signed(p['mean'], 5)} | {signed(p['median'], 5)} | {p['n_positive']} | {p['n_negative']} | {p['n_zero']} |")
        control = document["control"]
        lines += ["", f"- 控制：v2 集成经本评测与已提交的数差 seg35 {signed(control['differences']['seg35'], 10)}，seg4 {signed(control['differences']['seg4'], 10)}（容差 {control['tolerance']}）。",
                  f"- 对 v1 集成：{signed(document['against_v1']['seg35'])} / {signed(document['against_v1']['seg4'])}（seg35 / seg4）。"]
        u1 = document.get("against_u1")
        lines.append(f"- 对 UL 的 U1（单成员，fold A）：{signed(u1['seg35'])} / {signed(u1['seg4'])}。" if u1 else "- 对 UL 的 U1：没有 eval_u1.json，没有比较。")
        for place in document["sv_place"]:
            needs = fmt(place["needs_low"], 3) if place["needs_low"] == place["needs_high"] else f"{fmt(place['needs_low'], 3)}–{fmt(place['needs_high'], 3)}"
            lines.append(f"- 〔模型〕SV 过线表：19 对 seg35 {place['goal']} 约要分割级 {needs}（只描述）；这里 {fmt(document['means']['nn']['seg35'])}，{'已到' if place['at_or_above_high'] else ('已到下沿' if place['at_or_above_low'] else '未到')}。")
        focus = document.get("focus") or {}
        if focus:
            row, others = focus["row"], focus["others_mean"]
            lines.append(f"- {focus['subject']}：nnU-Net {fmt(row['nn']['seg35'])} / {fmt(row['nn']['seg4'])}，v2 集成 {fmt(row['reference_v2']['seg35'])} / {fmt(row['reference_v2']['seg4'])}；其余 19 人 nnU-Net {fmt(others['nn']['seg35'])} / {fmt(others['nn']['seg4'])}，v2 集成 {fmt(others['reference_v2']['seg35'])} / {fmt(others['reference_v2']['seg4'])}。")
    lines += ["", f"训练：{training.get('trainer')}，{training.get('epochs')} epoch，{fmt(None if training.get('wall_seconds') is None else training['wall_seconds']/3600, 2)} 小时，每 epoch 中位 {fmt(training.get('epoch_seconds_median'), 1)} s；主机峰值（PSS 之和）{fmt(training.get('host_peak_pss_gib'), 1)} GiB，GPU 已用峰值 {training.get('gpu_peak_used_mib')} MiB。",
              f"推断：每人中位 {fmt(inference.get('seconds_median'), 1)} s，最长 {fmt(inference.get('seconds_max'), 1)} s（含读图、预处理、滑窗、写文件），镜像轴 {inference.get('allowed_mirroring_axes')}，每窗 {inference.get('tta_views')} 次前向，GPU 分配峰值 {fmt(inference.get('gpu_peak_allocated_gib'), 1)} GiB。"]
    if training.get("runs_without_record"):
        lines += ["", f"**训练有 {training['runs_without_record']} 次运行没有写完记录**（被 RuntimeMaxSec 或 OOM 杀掉）；时长与 epoch 数来自 training_log 的时间戳，主机与 GPU 峰值只来自有记录的运行。"]
    if (document.get("dev_read") or {}).get("event") == "bz_approved_reread":
        lines += ["", f"**这是 BZ 批准的再次读取**：{document['dev_read'].get('bz_go')}。"]
    lines += ["", document["note"], ""]
    return "\n".join(lines)


def read_json(path):
    path = None if path is None else Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path is not None and path.is_file() else None


def read_u1(path):
    """The means of the arm U1 of eval_u1.json, or None."""
    document = read_json(path)
    return None if document is None else document.get("means", {}).get("U1")


def no_nan(node):
    """The document for JSON: an undefined value is null, never the non-standard NaN."""
    if isinstance(node, dict):
        return {k: no_nan(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [no_nan(v) for v in node]
    return None if isinstance(node, float) and not math.isfinite(node) else node


def write(out_dir, document):
    out_dir = Path(out_dir)
    targets = (out_dir/"decision_nn.json", out_dir/"summary_nn.md")
    for path in targets:
        if path.exists():
            raise FileExistsError(f"{path} exists; never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    document = no_nan(document)
    targets[0].write_text(json.dumps(document, indent=1, allow_nan=False, ensure_ascii=False)+"\n", encoding="utf-8")
    targets[1].write_text(summary_markdown(document), encoding="utf-8")
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=Path, required=True, help="eval_nn.json; a missing file is read as no evaluation")
    parser.add_argument("--train-record", type=Path, required=True)
    parser.add_argument("--infer-record", type=Path, required=True)
    parser.add_argument("--u1-eval", type=Path, default=None, help="eval_u1.json of work package UL; described only")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    document = decide(read_json(args.eval), read_json(args.train_record), read_json(args.infer_record), u1_means=read_u1(args.u1_eval))
    write(args.out_dir, document)
    print(json.dumps({k: document.get(k) for k in ("reading", "reading_text", "delta_seg35", "delta_seg4", "not_decidable_because")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
