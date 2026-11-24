"""The reading of work package UF (runs/20261105-uf/plan.md, by code; a label for the lead and BZ, not a conclusion): analysis/decision_uf.json and summary_uf.md, never overwritten.

The primary arm: of (a) UF4 and (b) UF4_nn the one with the higher mean seg35 over the 20 development subjects; a tie goes to (a). Delta = the mean over the 20 subjects of that arm minus the v1 ensemble (the committed rows of unet35_v2_full/eval_seg/eval_seg.json; the Hx line runs on v1), seg35, unrounded, paired.
  Delta >= +.003        worth connecting to Hx24; the candidate is the primary arm (its registration and the 19 pairs of Hx24 are another brief)
  Delta <  +.003        not connected: stop, hand to BZ
Not decidable (the reasons are written, no reading): no evaluation; not 20 subjects; a member missing or not as the plan trained it (eval_uf's member_problems); the gate of the nnU-Net probabilities not passed for the 20 subjects; a control that does not reproduce its committed numbers (|mean difference| over its tolerance, seg35 or seg4: 1e-9 for v1_control and
v2_control, which read stored probabilities, 2e-4 for U1_control, a GPU forward pass; the tolerances are this code's constants, the file's are only checked equal), a control or a metric of it missing; a mean of the reading undefined. Every reason is listed, not only the first kind.
Described, not read: the t interval (95 %) and the bootstrap over subjects (BOOTSTRAP_DRAWS draws, seed BOOTSTRAP_SEED) of Delta and the count of positive subjects; the same against the v2 ensemble; seg4, the groups, the cortex, the white matter; (b) - (a); the six subjects of the first 5 pairs (a pair = the mean of its two subjects, as the .9172 of OX);
OASIS_OAS1_0440_MR1 and the other 19; each member alone against its v2 namesake and foldA_s0 against U1; nnU-Net alone; the seconds; the training of each member.

    python -B experiments/uf/decide_uf.py --eval FILE --out-dir DIR

No option can turn a reading into "not decidable" after the evaluation (as experiments/nn/decide_nn.py, NN plan revision 1 R9): the reading follows from eval_uf.json alone; a missing file reads as no evaluation.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import uf_common as uc  # noqa: E402
import uf_nn  # noqa: E402

SCHEMA = "bayesreg.uf.decision.v1"
WORTH, NOT_CONNECT, NOT_DECIDABLE = "worth_hx24", "not_connected", "not_decidable"
READINGS = {WORTH: "值得接进 Hx24", NOT_CONNECT: "不接，停下交 BZ", NOT_DECIDABLE: "不可判"}
READING_ARMS = ("UF4", "UF4_nn")
CONTROLS = ("v1_control", "v2_control", "U1_control")
DESCRIBED_METRICS = ("seg35", "seg4", "small_six", "big_six", "other_23", "cortex", "white_matter")
NAMES = dict(UF4="(a) UF4 四成员集成", UF4_nn="(b) UF4 + nnU-Net 各 1/2", nn="nnU-Net 单模型（只描述）", reference_v1="v1 集成（已提交的数）", reference_v2="v2 集成（已提交的数）")
NOTE = ("开发集 20 名受试者，单次观测，分割级读数，不是配准结果，也不是留出集结果，不是对外比较。UF 的四个成员用训练集（每个 98 + 8 人，和 v2 同名成员一一对应）的标签训练，开发受试者不在 label_contact 里。开发集只读这一次（seg/uf_dev_read.lock），不看结果后调配方。"
        "它证伪不了什么：只说 v3_large 这个配方训满（4 万步，poly 走完，取最后一步）、四个成员、这一组种子（0、1，两折同种子同初值）；(b) 的 1/2 是事先定死的一个权重。分割级不等于配准后的 Dice（SV 的斜率约 .66–.70，OX 约 .89，都是〔模型〕）。"
        "主读数取 (a)、(b) 在同一批 20 人上较高的那个再和 v1 比，Δ 带一点偏乐观的选择效应（任务书定的规则，线不改）。"
        "“值得接进 Hx24”不说明接进去之后能到多少；“不接”也不说明 nnU-Net 训满、五折或别的配方不行。")


def finite(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def reading_of(delta):
    """The label of Delta seg35 (the bar included)."""
    if not finite(delta):
        return NOT_DECIDABLE
    return WORTH if delta >= uc.WORTH_BAR else NOT_CONNECT


def primary_arm(means):
    """(b) when its mean seg35 is higher than (a)'s, else (a): a tie goes to (a)."""
    return "UF4_nn" if means["UF4_nn"]["seg35"] > means["UF4"]["seg35"] else "UF4"


def column(evaluation, arm, metric):
    source = evaluation["rows"].get(arm) or evaluation["reference_rows"][arm]
    return np.array([np.nan if source[s].get(metric) is None else source[s][metric] for s in evaluation["subjects"]], dtype=np.float64)


def describe_difference(x, *, draws=uc.BOOTSTRAP_DRAWS, seed=uc.BOOTSTRAP_SEED):
    """Mean, median, the t interval (95 %), the bootstrap over subjects (percentile, 95 %), counts. Undefined subjects are left out and counted."""
    from scipy import stats
    x = np.asarray(x, dtype=np.float64)
    ok = x[np.isfinite(x)]
    n = int(ok.size)
    out = dict(mean=float(ok.mean()) if n else None, median=float(np.median(ok)) if n else None, n=n, n_undefined=int(x.size-n), n_positive=int((ok > 0).sum()), n_negative=int((ok < 0).sum()), n_zero=int((ok == 0).sum()))
    if n >= 2:
        half = float(stats.t.ppf(.975, n-1)*ok.std(ddof=1)/math.sqrt(n))
        rng = np.random.default_rng(seed)
        boot = ok[rng.integers(0, n, (draws, n))].mean(axis=1)
        out.update(t95=[out["mean"]-half, out["mean"]+half], bootstrap95=[float(v) for v in np.percentile(boot, [2.5, 97.5])], bootstrap_draws=draws, bootstrap_seed=seed)
    return out


def not_decidable_reasons(evaluation, reasons=()):
    out = list(reasons)
    if evaluation is None:
        return out+["没有评测结果"]
    subjects = evaluation.get("subjects") or []
    if len(subjects) != uc.N_SUBJECTS:
        out.append(f"{len(subjects)} 名受试者，不是 {uc.N_SUBJECTS}")
    if sorted(evaluation.get("members") or {}) != sorted(uc.MEMBER_NAMES):
        out.append(f"成员缺失：{sorted(set(uc.MEMBER_NAMES)-set(evaluation.get('members') or {}))}")
    for problem in evaluation.get("member_problems") or []:
        out.append(f"成员不是计划训的：{problem}")
    for problem in uf_nn.convert_problems((evaluation.get("nn") or {}).get("convert_record") or {}, subjects):
        out.append(f"闸门没过：{problem}")
    if evaluation.get("nn_weight") != uc.NN_WEIGHT:
        out.append(f"(b) 的权重是 {evaluation.get('nn_weight')}，不是 {uc.NN_WEIGHT}")
    means = evaluation.get("means") or {}
    for arm in READING_ARMS+("reference_v1",)+CONTROLS+("reference_v2", "reference_U1"):
        for metric in ("seg35", "seg4"):
            if not finite((means.get(arm) or {}).get(metric)):
                out.append(f"{arm} 的 {metric} 没有定义")
    controls = evaluation.get("controls") or {}
    if controls.get("tolerance") != uc.CONTROL_TOLERANCES:
        out.append(f"评测记下的控制容差 {controls.get('tolerance')} 不是本代码的 {uc.CONTROL_TOLERANCES}")
    for arm in CONTROLS:                                       # review minor 2: the code's tolerances, every control and metric required, listed whatever else is wrong
        tol = uc.CONTROL_TOLERANCES[arm]
        for metric in uc.CONTROL_METRICS:
            difference = ((controls.get("differences") or {}).get(arm) or {}).get(metric)
            if not finite(difference):
                out.append(f"控制缺失：{arm} 的 {metric} 没有差值")
            elif abs(difference) > tol:
                out.append(f"控制不符：{arm} 的 {metric} 与已提交的数差 {difference:+.3e}，超过 {tol:g}")
    return out


def nn_raw(evaluation):
    """The raw directory of the nnU-Net inference (npz, pkl, nii.gz) and its size: kept by the tools, cleaned by the lead after the reading (plan revision 1, review minor 9)."""
    root = ((evaluation or {}).get("nn") or {}).get("root")
    if not root:
        return None
    path = Path(root)/"raw"
    return dict(path=str(path), bytes=uc.directory_bytes(path), note="nnU-Net raw 待总负责人清理")


def raw_line(document):
    raw = document.get("nn_raw")
    return f"**提醒**：nnU-Net raw 待总负责人清理：{raw['path']}，{raw['bytes']/1e9:.2f} GB（工具不删）。" if raw else None


def first5(evaluation, arm, metric="seg35"):
    """The mean over the first 5 development pairs of the mean of the pair's two subjects (the .9172 of OX: v1 on these six subjects)."""
    source = evaluation["rows"].get(arm) or evaluation["reference_rows"][arm]
    values = [(source[a][metric]+source[b][metric])/2 for a, b in uc.FIRST5_PAIRS if a in source and b in source]
    return float(np.mean(values)) if len(values) == len(uc.FIRST5_PAIRS) else None


def focus(evaluation, arms):
    subjects = evaluation["subjects"]
    if uc.FOCUS_SUBJECT not in subjects:
        return None
    others = [s for s in subjects if s != uc.FOCUS_SUBJECT]
    out = {}
    for arm in arms:
        source = evaluation["rows"].get(arm) or evaluation["reference_rows"][arm]
        out[arm] = dict(subject={m: source[uc.FOCUS_SUBJECT].get(m) for m in ("seg35", "seg4", "small_six")}, others_mean={m: float(np.nanmean([np.nan if source[s].get(m) is None else source[s][m] for s in others])) for m in ("seg35", "seg4", "small_six")})
    return out


def decide(evaluation, *, reasons=()):
    problems = not_decidable_reasons(evaluation, reasons)
    document = dict(schema=SCHEMA, rule=dict(primary="of (a) UF4 and (b) UF4_nn the higher mean seg35; a tie goes to (a)", delta="primary minus the v1 ensemble, seg35, the mean over the 20 paired subjects (the committed rows of unet35_v2_full/eval_seg/eval_seg.json)",
                                             worth_hx24=f"delta >= +{uc.WORTH_BAR}", not_connected=f"delta < +{uc.WORTH_BAR}", control_tolerance=dict(uc.CONTROL_TOLERANCES), nn_weight=uc.NN_WEIGHT), dev_read=(evaluation or {}).get("dev_read"), nn_raw=nn_raw(evaluation), note=NOTE)
    if problems:
        return dict(document, reading=NOT_DECIDABLE, reading_text=READINGS[NOT_DECIDABLE], not_decidable_because=problems, primary_arm=None, candidate=None, delta_seg35=None)
    means = evaluation["means"]
    primary = primary_arm(means)
    delta = float(np.mean(column(evaluation, primary, "seg35")-column(evaluation, "reference_v1", "seg35")))
    reading = reading_of(delta)
    comparisons = {}
    for arm in READING_ARMS+("nn",):
        for reference in ("reference_v1", "reference_v2"):
            comparisons[f"{arm}_minus_{reference}"] = {m: describe_difference(column(evaluation, arm, m)-column(evaluation, reference, m)) for m in DESCRIBED_METRICS}
    comparisons["UF4_nn_minus_UF4"] = {m: describe_difference(column(evaluation, "UF4_nn", m)-column(evaluation, "UF4", m)) for m in DESCRIBED_METRICS}
    members = {}
    for name in uc.MEMBER_NAMES:
        short = uc.V2_NAME[name]
        members[f"UF_{short}"] = dict(means={m: means[f"UF_{short}"][m] for m in ("seg35", "seg4", "small_six", "big_six", "other_23")}, v2_namesake={m: means[f"reference_v2_{short}"][m] for m in ("seg35", "seg4", "small_six", "big_six", "other_23")},
                                      minus_v2_namesake={m: describe_difference(column(evaluation, f"UF_{short}", m)-column(evaluation, f"reference_v2_{short}", m)) for m in ("seg35", "seg4")})
    members["UF_A0"]["minus_U1"] = {m: describe_difference(column(evaluation, "UF_A0", m)-column(evaluation, "reference_U1", m)) for m in ("seg35", "seg4", "small_six")}
    training = {n: {k: s.get(k) for k in ("stopped", "iterations_run", "checkpoint_iteration", "best_iteration", "best_calib_mean_dice35", "checkpoint_calib_mean_dice35", "wall_seconds", "temperature", "memory")} for n, s in evaluation["members"].items()}
    convert = evaluation["nn"]["convert_record"]
    document.update(reading=reading, reading_text=READINGS[reading], not_decidable_because=[], primary_arm=primary, candidate=primary if reading == WORTH else None, delta_seg35=delta,
                    delta_seg4=float(np.mean(column(evaluation, primary, "seg4")-column(evaluation, "reference_v1", "seg4"))), primary_minus_v1=comparisons[f"{primary}_minus_reference_v1"], primary_minus_v2=comparisons[f"{primary}_minus_reference_v2"],
                    means={arm: {m: means[arm].get(m) for m in DESCRIBED_METRICS} for arm in READING_ARMS+("nn", "reference_v1", "reference_v2")}, comparisons=comparisons, members=members, training=training,
                    first5_pairs={arm: dict(seg35=first5(evaluation, arm), seg4=first5(evaluation, arm, "seg4")) for arm in READING_ARMS+("nn", "reference_v1", "reference_v2")}, focus=focus(evaluation, READING_ARMS+("nn", "reference_v1", "reference_v2")),
                    controls=evaluation["controls"], gate=dict(n_differing_total=convert.get("n_differing_total"), n_voxels_total=convert.get("n_voxels_total"), tolerance=convert.get("tolerance")),
                    voxels_b_differs_from_a={s: v["voxels_b_differs_from_a"] for s, v in (evaluation.get("per_subject_extra") or {}).items()}, seconds=evaluation.get("seconds"))
    return document


def fmt(value, digits=4):
    return "n/a" if not finite(value) else f"{value:.{digits}f}"


def signed(value, digits=4):
    return "n/a" if not finite(value) else f"{value:+.{digits}f}"


def interval(d):
    if not d or "t95" not in d:
        return "n/a"
    return f"{signed(d['mean'], 5)}（{d['n_positive']}/{d['n']} 为正；t [{signed(d['t95'][0], 4)}, {signed(d['t95'][1], 4)}]，bootstrap [{signed(d['bootstrap95'][0], 4)}, {signed(d['bootstrap95'][1], 4)}]）"


def summary_markdown(document):
    lines = ["# UF：v3_large 训满的四成员集成（开发集 20 人，单次观测，分割级）", "", f"**读数（由代码判）：{document['reading_text']}**", ""]
    if document["reading"] == NOT_DECIDABLE:
        lines += ["不可判的原因：", ""]+[f"- {r}" for r in document["not_decidable_because"]]+[""]+([raw_line(document), ""] if raw_line(document) else [])+[document["note"], ""]
        return "\n".join(lines)
    lines += [f"主臂（(a)、(b) 中 seg35 较高的那个，并列取 (a)）：**{NAMES[document['primary_arm']]}**。Δ = 主臂 − v1 集成：seg35 {signed(document['delta_seg35'], 5)}，seg4 {signed(document['delta_seg4'], 5)}。线：Δ ≥ +{uc.WORTH_BAR} 读“值得接进 Hx24”，候选是主臂；否则“不接”，交 BZ。", "",
              f"- 主臂 − v1：seg35 {interval(document['primary_minus_v1']['seg35'])}；seg4 {interval(document['primary_minus_v1']['seg4'])}",
              f"- 主臂 − v2：seg35 {interval(document['primary_minus_v2']['seg35'])}；seg4 {interval(document['primary_minus_v2']['seg4'])}", "",
              "| 臂 | seg35 | seg4 | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | 前 5 对（6 人）seg35 / seg4 |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, m in document["means"].items():
        f5 = document["first5_pairs"][arm]
        lines.append(f"| {NAMES[arm]} | {fmt(m['seg35'])} | {fmt(m['seg4'])} | {fmt(m['small_six'])} | {fmt(m['big_six'])} | {fmt(m['other_23'])} | {fmt(m['cortex'])} | {fmt(m['white_matter'])} | {fmt(f5['seg35'])} / {fmt(f5['seg4'])} |")
    comparisons = document["comparisons"]
    lines += ["", f"- (a) − v1：seg35 {interval(comparisons['UF4_minus_reference_v1']['seg35'])}；(b) − v1：seg35 {interval(comparisons['UF4_nn_minus_reference_v1']['seg35'])}",
              f"- (a) − v2：seg35 {interval(comparisons['UF4_minus_reference_v2']['seg35'])}；(b) − v2：seg35 {interval(comparisons['UF4_nn_minus_reference_v2']['seg35'])}",
              f"- (b) − (a)：seg35 {interval(comparisons['UF4_nn_minus_UF4']['seg35'])}；seg4 {interval(comparisons['UF4_nn_minus_UF4']['seg4'])}", "", "单成员（只描述）：", "", "| 成员 | seg35 | seg4 | 小六个 | v2 同名成员 seg35 / seg4 | 差 seg35（为正） |", "|---|---|---|---|---|---|"]
    for arm, m in document["members"].items():
        d = m["minus_v2_namesake"]["seg35"]
        lines.append(f"| {arm} | {fmt(m['means']['seg35'])} | {fmt(m['means']['seg4'])} | {fmt(m['means']['small_six'])} | {fmt(m['v2_namesake']['seg35'])} / {fmt(m['v2_namesake']['seg4'])} | {signed(d['mean'], 5)}（{d['n_positive']}/{d['n']}） |")
    lines += ["", f"- foldA_s0 − U1：seg35 {interval(document['members']['UF_A0']['minus_U1']['seg35'])}；seg4 {interval(document['members']['UF_A0']['minus_U1']['seg4'])}"]
    if document.get("focus"):
        lines += ["", f"{uc.FOCUS_SUBJECT}（单列）：", ""]+[f"- {NAMES[a]}：{fmt(f['subject']['seg35'])} / {fmt(f['subject']['seg4'])}，其余 19 人 {fmt(f['others_mean']['seg35'])} / {fmt(f['others_mean']['seg4'])}" for a, f in document["focus"].items()]
    controls = document["controls"]
    lines += ["", "控制：" + "；".join(f"{a} seg35 {d['seg35']:+.2e}、seg4 {d['seg4']:+.2e}（容差 {uc.CONTROL_TOLERANCES[a]:g}）" for a, d in controls["differences"].items())+"。",
              f"闸门：nnU-Net 存下的概率的 argmax 对 pred_dev，20 人合计 {document['gate']['n_differing_total']} / {document['gate']['n_voxels_total']} 体素不同（每人容差 {document['gate']['tolerance']}）。", "", "训练（每个成员）：", ""]
    for name, t in document["training"].items():
        lines.append(f"- {name}：{t['stopped']}，{t['iterations_run']} 步，取第 {t['checkpoint_iteration']} 步（标定集 Dice35 {fmt(t['checkpoint_calib_mean_dice35'])}；最好 {t['best_iteration']} 步 {fmt(t['best_calib_mean_dice35'])}，只记录不选），{fmt(None if t['wall_seconds'] is None else t['wall_seconds']/3600, 2)} 小时，T {fmt(t['temperature'], 3)}")
    if (document.get("dev_read") or {}).get("event") == "bz_approved_reread":
        lines += ["", f"**这是 BZ 批准的再次读取**：{document['dev_read'].get('bz_go')}。"]
    lines += ([""]+[raw_line(document)] if raw_line(document) else [])+["", document["note"], ""]
    return "\n".join(lines)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write(out_dir, document):
    out_dir = Path(out_dir)
    targets = (out_dir/"decision_uf.json", out_dir/"summary_uf.md")
    for path in targets:
        if path.exists():
            raise FileExistsError(f"{path} exists; never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    document = uc_no_nan(document)
    uc.write_text_atomic(targets[1], summary_markdown(document))
    uc.write_json_atomic(targets[0], document, ensure_ascii=False, allow_nan=False)
    return targets


def uc_no_nan(node):
    if isinstance(node, dict):
        return {k: uc_no_nan(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [uc_no_nan(v) for v in node]
    return None if isinstance(node, float) and not math.isfinite(node) else node


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=Path, required=True, help="eval_uf.json; a missing file is read as no evaluation")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    document = decide(read_json(args.eval))
    write(args.out_dir, document)
    print(json.dumps({k: document.get(k) for k in ("reading", "reading_text", "primary_arm", "delta_seg35", "not_decidable_because")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
