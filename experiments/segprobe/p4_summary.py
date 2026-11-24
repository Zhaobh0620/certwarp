"""SEGP P4: the tables of runs/20261105-segprobe/result.md from features.json and confusion.npz (p4_0440.py). 0440 against the other 19: value, their median and range, a robust z, the rank among the 20. Written next to the features as summary.md and printed.

    python -B experiments/segprobe/p4_summary.py --features DIR/features.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segprobe_common as common  # noqa: E402

TARGET = "OASIS_OAS1_0440_MR1"
NAMES = {int(k): v for k, v in json.loads((common.REPO/"configs/bayesreg/structure_groups_v1.json").read_text())["label_table"]["names"].items()}
SECTIONS = {
    "错误的几何（v1）": ["v1.errors.error_rate", "v1.errors.n_wrong", "v1.errors.share_on_boundary", "v1.errors.share_within_1", "v1.errors.share_within_2", "v1.errors.share_beyond_2", "v1.errors.mean_distance", "v1.errors.share_in_thick_parts", "v1.errors.n_components", "v1.errors.largest_component_share", "v1.errors.top10_components_share", "v1.errors.share_low_x"],
    "错误的置信度（v1）": ["v1.confidence.conf_on_wrong_mean", "v1.confidence.conf_on_right_mean", "v1.confidence.wrong_with_conf_above_90", "v1.confidence.wrong_with_conf_above_70", "v1.confidence.ptrue_on_wrong_mean", "v1.confidence.top2_hit_on_wrong", "v1.confidence.accuracy_foreground", "v1.confidence.confidence_foreground", "v1.confidence.ece_foreground"],
    "v1 与 v2 的错误": ["v1_v2_errors.jaccard", "v1_v2_errors.v1_errors_also_v2", "v1_v2_errors.v2_errors_also_v1", "v1_v2_errors.same_wrong_label_share_of_both", "v2.errors.error_rate", "v2.errors.share_beyond_2", "v2.confidence.wrong_with_conf_above_90"],
    "强度": ["intensity.q50", "intensity.wm_median", "intensity.gm_median", "intensity.csf_median", "intensity.subcortical_median", "intensity.wm_sd", "intensity.gm_sd", "intensity.contrast_wm_gm", "intensity.cnr_wm_gm", "intensity.auc_truth_wm_vs_cortex", "intensity.auc_pred_wm_vs_cortex", "intensity.bright_cortex_share", "intensity.dark_wm_share", "intensity.dark_wm_core_share", "intensity.dark_wm_core_near_ventricle", "intensity.dark_wm_core_largest_component", "intensity.noise_wm_high_pass_sd_over_wm", "intensity.wm_block_cv_norm", "intensity.wm_block_cv_orig", "intensity.orig_wm_median"],
    "解剖": ["anatomy.n_brain", "anatomy.ventricles", "anatomy.lateral_ventricles", "anatomy.ventricle_share", "anatomy.n_cortex", "anatomy.n_wm", "anatomy.cortex_share", "anatomy.wm_share", "anatomy.cortex_over_wm", "anatomy.thickness_proxy", "anatomy.image_positive_truth_background", "anatomy.truth_foreground_image_zero", "anatomy.native_volume_mm3", "anatomy.linear_scale_of_alignment"],
    "对齐": ["alignment.extent_z", "alignment.extent_y", "alignment.extent_x", "alignment.roll_deg", "alignment.yaw_deg", "leave_one_out.ncc_image", "leave_one_out.seg4_dice_with_majority_of_others", "leave_one_out.seg4_dice_per_class.cortex", "leave_one_out.seg4_dice_per_class.subcortical", "leave_one_out.seg4_dice_per_class.white_matter", "leave_one_out.seg4_dice_per_class.csf"],
    "标签与图像": ["labels_v1.truth_crossing", "labels_v1.truth_slope_over_span", "labels_v1.pred_crossing", "labels_v1.pred_slope_over_span", "labels_v1.roughness.truth.isolated_share", "labels_v1.roughness.truth.at_most_1_neighbour_share", "labels_v1.roughness.truth.surface_share", "labels_v1.roughness.pred.at_most_1_neighbour_share", "labels_v1.roughness.pred.surface_share", "labels_v1.cortex_components_truth", "labels_v1.cortex_components_pred", "labels_v1.seg4_macro_with_truth_shifted.none"],
    "混淆体素的强度（(I − 皮层中位) / (白质中位 − 皮层中位)）": ["labels_v1.confusions_normalized_intensity.truth_cortex_pred_wm.n", "labels_v1.confusions_normalized_intensity.truth_cortex_pred_wm.median", "labels_v1.confusions_normalized_intensity.truth_wm_pred_cortex.n", "labels_v1.confusions_normalized_intensity.truth_wm_pred_cortex.median", "labels_v1.confusions_normalized_intensity.truth_cortex_pred_background.n", "labels_v1.confusions_normalized_intensity.truth_cortex_pred_background.median", "labels_v1.confusions_normalized_intensity.truth_background_pred_cortex.n", "labels_v1.confusions_normalized_intensity.truth_background_pred_cortex.median", "labels_v1.confusions_normalized_intensity.agree_wm.median", "labels_v1.confusions_normalized_intensity.agree_cortex.median"],
}


def num(value, digits=4):
    if value is None:
        return "n/a"
    return f"{value:.0f}" if abs(value) >= 1000 else f"{value:.{digits}f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=common.OUT_ROOT/"p4/features.json")
    args = parser.parse_args(argv)
    document = json.loads(args.features.read_text())
    stats, rows, subjects = document["stats"], document["rows"], document["subjects"]
    lines = ["# SEGP P4：0440 对其余 19 人（开发集，单次观测；只读描述）", "", "z 是稳健 z（其余 19 人的中位数与 MAD），名次从低到高，共 20。特征很多，单个 z 的极端值要带多重比较的眼光看。", ""]
    for title, keys in SECTIONS.items():
        lines += [f"## {title}", "", "| 特征 | 0440 | 其余中位 | 其余范围 | z | 名次 |", "|---|---:|---:|---|---:|---:|"]
        for key in keys:
            s = stats.get(key)
            if s is None:
                lines.append(f"| {key} | 未定义 | | | | |")
                continue
            z = "n/a" if s["robust_z"] is None else f"{s['robust_z']:+.1f}"
            lines.append(f"| {key} | {num(s['value'])} | {num(s['others_median'])} | {num(s['others_min'])} 到 {num(s['others_max'])} | {z} | {s['rank_of_20_from_low']} |")
        lines.append("")
    # the largest departures among all the scalar features
    ranked = sorted((k for k, s in stats.items() if s["robust_z"] is not None and (s["rank_of_20_from_low"] in (1, 20))), key=lambda k: -abs(stats[k]["robust_z"]))
    lines += ["## 0440 在 20 人里排第一或最后、稳健 z 最大的 30 个特征", "", "| 特征 | 0440 | 其余中位 | 其余范围 | z |", "|---|---:|---:|---|---:|"]
    lines += [f"| {k} | {num(stats[k]['value'])} | {num(stats[k]['others_median'])} | {num(stats[k]['others_min'])} 到 {num(stats[k]['others_max'])} | {stats[k]['robust_z']:+.1f} |" for k in ranked[:30]]
    lines.append("")
    # shifts
    lines += ["## seg4 macro：预测对平移一个体素的真值", "", "| 受试者 | 不平移 | 最好的平移 | 那个平移的 seg4 macro |", "|---|---:|---|---:|"]
    for subject in subjects:
        shifts = rows[subject]["labels_v1"]["seg4_macro_with_truth_shifted"]
        best = rows[subject]["labels_v1"]["best_shift"]
        lines.append(f"| {subject[11:15]} | {shifts['none']:.4f} | {best} | {shifts[best]:.4f} |")
    lines.append("")
    shifts = rows[TARGET]["labels_v1"]["seg4_macro_with_truth_shifted"]
    lines += ["0440 的六个平移："+"，".join(f"{k} {v:.4f}" for k, v in shifts.items()), ""]
    # the location
    target_location = rows[TARGET]["v1"]["location"]
    others = [s for s in subjects if s != TARGET]
    lines += ["## 错误率的位置（v1）", "", "| 八分区 | 0440 | 其余中位 | 比 |", "|---|---:|---:|---:|"]
    for octant, value in target_location["octant_rate"].items():
        median = float(np.median([rows[s]["v1"]["location"]["octant_rate"][octant] for s in others]))
        lines.append(f"| {octant} | {value:.4f} | {median:.4f} | {value/median:.2f} |")
    lines.append("")
    median_share = float(np.median([rows[s]["v1"]["location"]["share_in_densest_10pct_of_blocks"] for s in others]))
    lines += [f"最密 10% 的 16 体素块占错误的比例：0440 {target_location['share_in_densest_10pct_of_blocks']:.4f}，其余中位 {median_share:.4f}。", ""]
    # the confusions
    with np.load(args.features.parent/"confusion.npz") as npz:
        confusion, names = npz["v1"].astype(np.float64), [str(s) for s in npz["subjects"]]
    index = names.index(TARGET)
    matrix = confusion[index].copy()
    others_mean = np.delete(confusion, index, axis=0).mean(axis=0)
    np.fill_diagonal(matrix, 0)
    np.fill_diagonal(others_mean, 0)
    excess = matrix-others_mean
    order = np.dstack(np.unravel_index(np.argsort(-excess, axis=None), excess.shape))[0][:24]
    lines += ["## 比其余 19 人平均多出最多的混淆（真值 → v1 预测，体素数）", "", "| 真值 | 预测 | 0440 | 其余平均 | 多出 | 占 0440 总多出的比例 |", "|---|---|---:|---:|---:|---:|"]
    total_excess = float(matrix.sum()-others_mean.sum())
    for a, b in order:
        lines.append(f"| {NAMES[int(a)]} | {NAMES[int(b)]} | {matrix[a, b]:.0f} | {others_mean[a, b]:.0f} | {excess[a, b]:.0f} | {excess[a, b]/total_excess:.3f} |")
    lines += ["", f"错误体素总数：0440 {matrix.sum():.0f}，其余平均 {others_mean.sum():.0f}。", ""]
    # false negatives and positives by label
    lines += ["## 逐标签：Dice、体积比（预测 / 真值）、真值体积", "", "| 标签 | 名称 | Dice 0440 | 其余均值 | 其余最小 | 体积比 0440 | 其余中位 | 真值体积 0440 | 其余中位 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    dice = np.array([rows[s]["v1"]["per_label"] for s in subjects], dtype=np.float64)
    ratio = np.array([rows[s]["v1"]["volume_ratio"] for s in subjects], dtype=np.float64)
    true_volume = np.array([rows[s]["v1"]["true_volume"] for s in subjects], dtype=np.float64)
    t = subjects.index(TARGET)
    keep = [i for i in range(len(subjects)) if i != t]
    gap = dice[t]-dice[keep].mean(axis=0)
    for k in np.argsort(gap):
        label = int(k)+1
        lines.append(f"| {label} | {NAMES[label]} | {dice[t, k]:.4f} | {dice[keep, k].mean():.4f} | {dice[keep, k].min():.4f} | {ratio[t, label]:.3f} | {np.median(ratio[keep, label]):.3f} | {true_volume[t, label]:.0f} | {np.median(true_volume[keep, label]):.0f} |")
    lines.append("")
    text = "\n".join(lines)
    (args.features.parent/"summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
