"""The reading of work package UL (runs/20261104-ul/plan.md, by code; a label for the lead and BZ, not a conclusion): decision_ul.json and summary_ul.md, never overwritten.

Delta = the mean over the 20 development subjects of U1 minus member A0 of unet35_v2_full (fold A, seed 0: the same fold and seed as U1), seg35, unrounded, on the committed per-subject numbers of runs/20261103-unv2/v2_member_subsets.json. The bars are included.
  Delta >= +.005            worth training the ensemble
  Delta <= +.002            the recipe is not the main limit: hand to BZ to decide on nnU-Net
  otherwise                 middle band
Not decidable: no evaluation (the training or the evaluation failed: the reasons are written down), not 20 subjects, the evaluator's control does not reproduce the reference (|mean difference| > CONTROL_TOL, seg35 or seg4), the member is not U1 (recipe v3_large, fold A, seed 0), or a metric undefined.
Budgets (GPU peak <= 20 GiB, host peak <= 8 GiB, <= 5 hours) are reported; they do not change the reading. Everything else is described: seg4, the small six, the big six, the other 23, the same delta against the mean of the four single members (.9230 / .9566 in the brief) with the reading it would give,
the ensemble of v2, the Gaussian window (U1_uniform), the forward seconds. One member, one seed, one evaluation on the development set; not a hold-out result, not a comparison to an external method.

    python -B experiments/ul/decide_ul.py --eval FILE --member DIR --out-dir DIR [--steps FILE] [--reason TEXT]
"""
import argparse
import json
import math
from pathlib import Path
import sys

SCHEMA = "bayesreg.ul.decision.v1"
WORTH_BAR, NOT_LIMIT_BAR = .005, .002
CONTROL_TOL = 2e-4                  # the same member, the same image: AMP and cuDNN may move a few voxels, not this much
N_SUBJECTS = 20
GPU_LIMIT_GIB, HOST_LIMIT_GIB, WALL_LIMIT_HOURS = 20., 8., 5.
WORTH, NOT_LIMIT, MIDDLE, NOT_DECIDABLE = "worth_ensemble", "recipe_not_main_limit", "middle", "not_decidable"
READINGS = {WORTH: "值得训完集成", NOT_LIMIT: "配方不是主要限制，交 BZ 定 nnU-Net", MIDDLE: "中间带", NOT_DECIDABLE: "不可判"}
NOTE = ("开发集 20 名受试者，单个成员、单次观测，分割级读数，不是配准结果，也不是留出集结果。标签只用训练集（fold A，98 + 8 人），开发受试者不在 label_contact 里。v2 对照是同折同种子的成员 A0。"
        "U1 同时改了网络宽度、patch 训练、批大小、深监督、翻转、类别均衡采样、滑窗权重和早停参数（步数也不同），读数不能拆给其中任何一项。开发集只读这一次，不看结果后调配方。"
        "读数只说这个配方、这个步数、这一个种子：\u201c配方不是主要限制\u201d没有测 nnU-Net 本身（更多步数、更大的增强、SGD 都没有测），\u201c值得训完集成\u201d不说明传到配准有多少。")
SEG4_AND_GROUPS = ("seg4", "small_six", "big_six", "other_23")


def reading_of(delta):
    """The label of a delta of seg35 (bars included)."""
    if delta is None or not math.isfinite(delta):
        return NOT_DECIDABLE
    if delta >= WORTH_BAR:
        return WORTH
    if delta <= NOT_LIMIT_BAR:
        return NOT_LIMIT
    return MIDDLE


def not_decidable_reasons(evaluation, train_record, reasons=()):
    """Why the rule cannot be read (empty: it can)."""
    out = list(reasons)
    if evaluation is None:
        return out+["没有评测结果"]
    if train_record is None:
        out.append("没有 train_record.json")
    else:
        if train_record.get("recipe") != "v3_large":
            out.append(f"成员的 recipe 是 {train_record.get('recipe')!r}，不是 v3_large")
        if (train_record.get("fold"), train_record.get("seed")) != ("A", 0):
            out.append(f"成员是 {train_record.get('fold')}{train_record.get('seed')}，不是 A0")
    members = evaluation.get("members", {})
    for name in ("U1", "v2_A0_control"):
        if (members.get(name, {}).get("fold"), members.get(name, {}).get("seed")) != ("A", 0):
            out.append(f"{name} 不是 fold A、seed 0")
    if evaluation.get("reference", {}).get("member") != "A0":
        out.append("参照不是成员 A0")
    if len(evaluation.get("subjects", [])) != N_SUBJECTS:
        out.append(f"{len(evaluation.get('subjects', []))} 名受试者，不是 {N_SUBJECTS}")
    means = evaluation.get("means", {})
    for arm, metric in (("U1", "seg35"), ("U1", "seg4"), ("v2_A0_control", "seg35"), ("v2_A0_control", "seg4")):
        value = means.get(arm, {}).get(metric)
        if value is None or not math.isfinite(value):
            out.append(f"{arm} 的 {metric} 没有定义")
    if not out:
        for metric in ("seg35", "seg4"):
            difference = means["v2_A0_control"][metric]-evaluation["reference"]["means"][metric]
            if abs(difference) > CONTROL_TOL:
                out.append(f"控制不符：v2 A0 经本评测的 {metric} 与已提交的数差 {difference:+.6f}，超过 {CONTROL_TOL}")
    return out


def diff(a, b):
    """a - b, None when either is undefined."""
    return None if a is None or b is None else a-b


def no_nan(node):
    """The document for JSON: an undefined value is null, never the non-standard NaN."""
    if isinstance(node, dict):
        return {k: no_nan(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [no_nan(v) for v in node]
    return None if isinstance(node, float) and not math.isfinite(node) else node


def budget_flags(train_record):
    """The budgets, reported; missing readings are None."""
    memory = (train_record or {}).get("memory", {})
    wall = (train_record or {}).get("wall_seconds")
    hours = None if wall is None else wall/3600
    gpu, host = memory.get("gpu_peak_allocated_gib"), memory.get("host_peak_rss_gib")
    return dict(gpu_peak_allocated_gib=gpu, gpu_within_limit=None if gpu is None else gpu <= GPU_LIMIT_GIB, host_peak_rss_gib=host, host_within_limit=None if host is None else host <= HOST_LIMIT_GIB, training_hours=hours,
                training_within_limit=None if hours is None else hours <= WALL_LIMIT_HOURS, limits=dict(gpu_gib=GPU_LIMIT_GIB, host_gib=HOST_LIMIT_GIB, hours=WALL_LIMIT_HOURS))


def schedule_at_stop(train_record):
    """How far the poly schedule went: the share of max_iters and the learning rate of the last step (the formula of train._lr_at, from the config in the record). None when the record has no config."""
    record = train_record or {}
    config, done = record.get("config") or {}, record.get("iterations_run")
    if not done or not all(k in config for k in ("max_iters", "warmup_iters", "lr", "poly_power")):
        return None
    it, total, warmup = done-1, config["max_iters"], config["warmup_iters"]
    lr = config["lr"]*(it+1)/warmup if it < warmup else config["lr"]*(1-min(1., (it-warmup)/max(1, total-warmup)))**config["poly_power"]
    return dict(share_of_max_iters=done/total, lr_at_stop=lr, max_iters=total)


def decide(evaluation, train_record, *, steps=None, reasons=()):
    problems = not_decidable_reasons(evaluation, train_record, reasons)
    training = {k: (train_record or {}).get(k) for k in ("stopped", "iterations_run", "best_iteration", "best_calib_mean_dice35", "wall_seconds", "parameters", "parameters_plain")}
    training.update(schedule=schedule_at_stop(train_record), truncated_by_time=training["stopped"] == "wall_budget")
    document = dict(schema=SCHEMA, rule=dict(worth_ensemble=f"delta seg35 >= +{WORTH_BAR}", recipe_not_main_limit=f"delta seg35 <= +{NOT_LIMIT_BAR}", middle="otherwise", reference="member A0 of unet35_v2_full (fold A, seed 0), runs/20261103-unv2/v2_member_subsets.json"),
                    budgets=budget_flags(train_record), training=training, steps_decision=steps, dev_read=(evaluation or {}).get("dev_read"), note=NOTE)
    if problems:
        return dict(document, reading=NOT_DECIDABLE, reading_text=READINGS[NOT_DECIDABLE], not_decidable_because=problems, delta_seg35=None, delta_seg4=None)
    means, reference = evaluation["means"], evaluation["reference"]
    delta = means["U1"]["seg35"]-reference["means"]["seg35"]
    comparisons = dict(same_member_A0={m: diff(means["U1"].get(m), reference["means"].get(m)) for m in ("seg35",)+SEG4_AND_GROUPS})
    four = reference.get("single_member_mean") or {}
    ensemble = reference.get("ensemble") or {}
    four_delta = None if four.get("seg35") is None else means["U1"]["seg35"]-four["seg35"]
    document.update(reading=reading_of(delta), reading_text=READINGS[reading_of(delta)], not_decidable_because=[], delta_seg35=delta, delta_seg4=comparisons["same_member_A0"]["seg4"], means=means, reference_means=reference["means"],
                    comparisons=comparisons, paired=evaluation["paired"], control=dict(seg35=means["v2_A0_control"]["seg35"]-reference["means"]["seg35"], seg4=means["v2_A0_control"]["seg4"]-reference["means"]["seg4"], tolerance=CONTROL_TOL),
                    against_four_single_members=dict(reference=four, delta_seg35=four_delta, delta_seg4=None if four.get("seg4") is None else means["U1"]["seg4"]-four["seg4"], reading_it_would_give=reading_of(four_delta), reading_differs=reading_of(four_delta) != reading_of(delta)),
                    against_v2_ensemble=dict(reference=ensemble, delta_seg35=None if ensemble.get("seg35") is None else means["U1"]["seg35"]-ensemble["seg35"], delta_seg4=None if ensemble.get("seg4") is None else means["U1"]["seg4"]-ensemble["seg4"]),
                    gaussian_window=dict(U1_minus_uniform_seg35=means["U1"]["seg35"]-means["U1_uniform"]["seg35"], U1_minus_uniform_seg4=means["U1"]["seg4"]-means["U1_uniform"]["seg4"]),
                    seconds={a: {k: v for k, v in s.items() if k != "per_subject"} for a, s in evaluation["seconds"].items()}, window=evaluation["window"], gpu_peak_inference_gib=evaluation.get("gpu_peak_allocated_gib"))
    return document


def fmt(value, digits=4):
    return "n/a" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.{digits}f}"


def signed(value, digits=4):
    return "n/a" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:+.{digits}f}"


def summary_markdown(document):
    lines = ["# UL：U1（v3_large 单成员）对 v2 单成员（开发集，单次观测，分割级）", "", f"**读数（由代码判）：{document['reading_text']}**", ""]
    budgets, training = document["budgets"], document["training"]
    if document["reading"] == NOT_DECIDABLE:
        lines += ["不可判的原因：", ""]+[f"- {reason}" for reason in document["not_decidable_because"]]+[""]
    else:
        lines += [f"Δ = U1 − A0：seg35 {signed(document['delta_seg35'], 5)}，seg4 {signed(document['delta_seg4'], 5)}。规则：seg35 ≥ +{WORTH_BAR} 值得训完集成；≤ +{NOT_LIMIT_BAR} 配方不是主要限制，交 BZ 定 nnU-Net；其余中间带。", "",
                  "| 臂 | seg35 | seg4 | 小六个 | 大六个 | 其余 23 个 | 每人前向秒（中位） |", "|---|---|---|---|---|---|---|"]
        names = dict(U1="U1（高斯窗，读数用它）", U1_uniform="U1（平均窗，只描述）", v2_A0_control="v2 A0 经本评测（控制）")
        for arm, name in names.items():
            m = document["means"][arm]
            lines.append(f"| {name} | {fmt(m['seg35'])} | {fmt(m['seg4'])} | {fmt(m['small_six'])} | {fmt(m['big_six'])} | {fmt(m['other_23'])} | {fmt(document['seconds'][arm]['median'], 2)} |")
        r = document["reference_means"]
        lines += [f"| v2 A0（已提交的数，参照） | {fmt(r['seg35'])} | {fmt(r['seg4'])} | {fmt(r['small_six'])} | {fmt(r['big_six'])} | {fmt(r['other_23'])} | |", "", "逐受试者配对差（U1 − A0；均值、中位、正 / 负 / 零）：", "",
                  "| 指标 | 均值差 | 中位差 | 正 | 负 | 零 |", "|---|---|---|---|---|---|"]
        for metric, p in document["paired"]["U1_minus_reference"].items():
            lines.append(f"| {metric} | {signed(p['mean'], 5)} | {signed(p['median'], 5)} | {p['n_positive']} | {p['n_negative']} | {p['n_zero']} |")
        four, ensemble, gauss, window = document["against_four_single_members"], document["against_v2_ensemble"], document["gaussian_window"], document["window"]
        lines += ["", f"- 控制：v2 A0 经本评测与已提交的数差 seg35 {signed(document['control']['seg35'], 6)}，seg4 {signed(document['control']['seg4'], 6)}（容差 {document['control']['tolerance']}）。",
                  f"- 对四个 v2 单成员的均值（{fmt(four['reference'].get('seg35'))} / {fmt(four['reference'].get('seg4'))}）：{signed(four['delta_seg35'])} / {signed(four['delta_seg4'])}，同一规则会读成"
                  f"“{READINGS[four['reading_it_would_give']]}”{'（与上面的读数不同，以同折同种子为准）' if four['reading_differs'] else ''}。",
                  f"- 对 v2 四成员集成（{fmt(ensemble['reference'].get('seg35'))} / {fmt(ensemble['reference'].get('seg4'))}）：{signed(ensemble['delta_seg35'])} / {signed(ensemble['delta_seg4'])}。这是单成员对集成，只作描述。",
                  f"- 高斯窗相对平均窗：{signed(gauss['U1_minus_uniform_seg35'], 5)} / {signed(gauss['U1_minus_uniform_seg4'], 5)}。",
                  f"- 滑窗：窗 {window['window']}，overlap {window['overlap']}，每人 {window['n_windows']} 个窗；推断 GPU 峰值 {fmt(document.get('gpu_peak_inference_gib'), 1)} GiB。"]
    if training.get("truncated_by_time"):
        schedule = training.get("schedule") or {}
        lines += ["", f"**训练被时间截断**：走到调度的 {fmt(schedule.get('share_of_max_iters'), 3)}，停止时 lr {fmt(schedule.get('lr_at_stop'), 6)}。poly 没有走完，分数可能偏低，读成\u201c中间带\u201d或\u201c配方不是主要限制\u201d时要带这个限定。规则本身不变。"]
    elif training.get("stopped") == "early_stop":
        lines += ["", f"**训练早停**（走到调度的 {fmt((training.get('schedule') or {}).get('share_of_max_iters'), 3)}）。规则本身不变。"]
    if (document.get("dev_read") or {}).get("event") == "bz_approved_reread":
        lines += ["", f"**这是 BZ 批准的再次读取**：{document['dev_read'].get('bz_go')}。"]
    lines += ["", f"训练：{training.get('stopped')}，{training.get('iterations_run')} 步，最好 {training.get('best_iteration')} 步，标定集 Dice35 {fmt(training.get('best_calib_mean_dice35'))}，{fmt(None if training.get('wall_seconds') is None else training['wall_seconds']/3600, 2)} 小时；"
              f"GPU 峰值 {fmt(budgets['gpu_peak_allocated_gib'], 1)} GiB（上限 {budgets['limits']['gpu_gib']:g}），主机峰值 {fmt(budgets['host_peak_rss_gib'], 1)} GiB（上限 {budgets['limits']['host_gib']:g}）。", "", document["note"], ""]
    return "\n".join(lines)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write(out_dir, document):
    out_dir = Path(out_dir)
    targets = (out_dir/"decision_ul.json", out_dir/"summary_ul.md")
    for path in targets:
        if path.exists():
            raise FileExistsError(f"{path} exists; never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    document = no_nan(document)
    targets[0].write_text(json.dumps(document, indent=1, allow_nan=False)+"\n", encoding="utf-8")
    targets[1].write_text(summary_markdown(document), encoding="utf-8")
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=Path, required=True, help="eval_u1.json; a missing file is read as no evaluation")
    parser.add_argument("--member", type=Path, required=True, help="the member directory of U1")
    parser.add_argument("--steps", type=Path, default=None, help="steps_decision.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reason", action="append", default=[], help="an added reason why it is not decidable")
    args = parser.parse_args(argv)
    document = decide(read_json(args.eval), read_json(args.member/"train_record.json"), steps=read_json(args.steps) if args.steps else None, reasons=args.reason)
    write(args.out_dir, document)
    print(json.dumps({k: document.get(k) for k in ("reading", "reading_text", "delta_seg35", "delta_seg4", "not_decidable_because")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
