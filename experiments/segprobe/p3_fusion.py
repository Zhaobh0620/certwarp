"""SEGP P3: pair fusion on the final fields of the arm S_P (runs/20261105-segprobe/任务书.md). Segmentation level, the first 5 development pairs, single observation; a diagnostic.

For a pair (fixed F, moving M) with the certified final field of S_P (seed_v1): the posterior of M (unet35_v1_ens, probs36) pulled back to the grid of F, channel by channel, trilinear (map_coordinates order 1,
mode nearest, so that the channels still sum to 1 near the boundary), warped_c(p) = P_M,c(p + d(p)); then argmax_c (P_F,c + warped_c)/2. That is the only fusion. Evaluated on F by eval_seg.summarize_subject.

The field, before anything is reported: final_native.npz key u ((z, y, x, 3) mm, xyz components); its sha256 against final_native.json, the state hash of the certificate against u, certificate certified_feasible with
n_infeasible = n_uncertain = 0. The direction warped(p) = moving(p + d(p)), d = disp_voxels(u): the truth of M warped with this d (order 0, the evaluator's warp_labels_l2r) against the truth of F must reproduce metrics.json
seg35 and seg4 (tolerance 1e-12); -d is computed too and must give a lower seg35. Any departure stops the run.

Reduction of error: 1 - (1 - Dice fused)/(1 - Dice fixed only), per pair and for the means over the pairs (the ratio of the means and the mean of the ratios). The reading of the brief: >= 10% is worth considering for the path.
The field was driven by these same two predictions (S_P entry: the K = 35 channels of unet35_v1_ens; native: the colouring from it), so their errors may be correlated: a diagnostic on development pairs.

    python -B experiments/segprobe/p3_fusion.py --out-dir DIR
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segprobe_common as common  # noqa: E402
from segprobe_common import REPO, eval_seg  # noqa: E402

sys.path.insert(0, str(REPO/"scripts"))
from evaluate_l2r_style import disp_voxels, warp_labels_l2r  # noqa: E402
from jd_curl.three_d.nifti import dice_labels  # noqa: E402

FORMAT = "bayesreg.segprobe.p3.v1"
RUNS = Path("/home/rmcfz/seq_vp_tag_results/seed_v1/runs/S_P")
PAIRS = tuple((f"OASIS_OAS1_{a:04d}_MR1", f"OASIS_OAS1_{a+1:04d}_MR1") for a in range(438, 443))     # (fixed, moving), experiments/seed: the first 5 development pairs
TASK = "pg_weak__four__330s"
ARMS = ("fixed", "fused", "moving_warped")
METRICS = ("seg35", "seg4")+tuple(eval_seg.GROUPS)+("seg4_cortex", "seg4_wm")
READING_BAR = .10
DIRECTION_TOL = 1e-12


def check_field(run_dir):
    """(u, spacing, record): the file, the certificate and its binding to u; ValueError otherwise."""
    run_dir = Path(run_dir)
    task = run_dir/TASK
    final = json.loads((task/"final_native.json").read_text())
    npz_sha = eval_seg.sha_file(task/"final_native.npz")
    if final.get("sha256") != npz_sha:
        raise ValueError(f"{task}/final_native.npz: sha256 differs from final_native.json")
    with np.load(task/"final_native.npz") as z:
        files = list(z.files)
        u = np.array(z["u"])
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError(f"{task}: u has shape {u.shape}, expected (z, y, x, 3)")
    certificate = final.get("certificate") or {}
    state = hashlib.sha256(np.ascontiguousarray(u, dtype=np.float64).view(np.uint8)).hexdigest()
    passed = certificate.get("status") == "certified_feasible" and certificate.get("min_jd_lower_bound", 0) > 0 and certificate.get("n_infeasible") == 0 and certificate.get("n_uncertain") == 0 and certificate.get("state_hash") == state
    if not passed or final.get("role") != "formal_native_outer_commit":
        raise ValueError(f"{task}: certificate not passed or not bound to u: {certificate.get('status')}, role {final.get('role')}")
    pair = run_dir.name
    manifest = json.loads((run_dir/"inputs"/pair/"manifest.json").read_text())
    return u, tuple(manifest["spacing_xyz"]), dict(file=str(task/"final_native.npz"), keys=files, sha256=npz_sha, u_shape=list(u.shape), u_dtype=str(u.dtype), certificate_status=certificate["status"],
                                                  min_jd_lower_bound=certificate["min_jd_lower_bound"], state_hash_matches_u=True, spacing_xyz=list(manifest["spacing_xyz"]))


def macro(fixed, warped, n):
    return dice_labels(fixed, warped.astype(np.float64), labels=list(range(1, n)))["macro_dice"]


def check_direction(d, true_f35, true_f4, true_m35, true_m4, metrics):
    """The evaluator's warp of the truth with d must reproduce metrics.json; -d is the wrong direction and must be worse. Returns the record; ValueError otherwise."""
    forward35, forward4 = macro(true_f35, warp_labels_l2r(true_m35, d), 36), macro(true_f4, warp_labels_l2r(true_m4, d), 5)
    backward35 = macro(true_f35, warp_labels_l2r(true_m35, -d), 36)
    record = dict(seg35_recomputed=forward35, seg4_recomputed=forward4, seg35_metrics_json=metrics["seg35"], seg4_metrics_json=metrics["seg4"], seg35_with_minus_d=backward35,
                  max_abs_diff=max(abs(forward35-metrics["seg35"]), abs(forward4-metrics["seg4"])))
    if not record["max_abs_diff"] <= DIRECTION_TOL:
        raise ValueError(f"the field with warped(p) = moving(p + d(p)) does not reproduce metrics.json: {record}")
    if not backward35 < forward35:
        raise ValueError(f"-d is not worse than d: {record}")
    return record


def pull_back_coordinates(d):
    """(3, Z, Y, X) float64: p + d(p), in voxels, with the number of voxels whose point lies outside the grid."""
    shape = d.shape[1:]
    grid = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij", sparse=True)
    coordinates = np.stack([grid[axis]+d[axis] for axis in range(3)])
    inside = np.ones(shape, dtype=bool)
    for axis, n in enumerate(shape):
        inside &= (coordinates[axis] >= 0) & (coordinates[axis] <= n-1)
    return coordinates, int(inside.size-inside.sum())


def pull_back_channels(raw_moving, coordinates, *, order=1):
    """Yields (c, warped channel float64) for every channel: read-back normalized, then interpolated at `coordinates` (mode nearest: the channels keep their sum)."""
    total = np.zeros(raw_moving.shape[1:], dtype=np.float64)
    for c in range(raw_moving.shape[0]):
        total += raw_moving[c].astype(np.float64)
    for c in range(raw_moving.shape[0]):
        yield c, map_coordinates(raw_moving[c].astype(np.float64)/total, coordinates, order=order, mode="nearest", output=np.float64)


def fuse(raw_fixed, raw_moving, d):
    """(fused argmax, moving_warped argmax, record). Running maxima: a strictly larger channel replaces, so ties go to the smallest label as np.argmax. raw_fixed normalized as the read-back."""
    shape = raw_fixed.shape[1:]
    coordinates, outside = pull_back_coordinates(d)
    total_fixed = np.zeros(shape, dtype=np.float64)
    for c in range(raw_fixed.shape[0]):
        total_fixed += raw_fixed[c].astype(np.float64)
    best_mix, best_warp = np.full(shape, -1., dtype=np.float64), np.full(shape, -1., dtype=np.float64)
    top_mix, top_warp = np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8)
    warped_sum = np.zeros(shape, dtype=np.float64)
    for c, warped in pull_back_channels(raw_moving, coordinates):
        warped_sum += warped
        mix = .5*(raw_fixed[c].astype(np.float64)/total_fixed+warped)
        better = mix > best_mix
        best_mix[better], top_mix[better] = mix[better], c
        better = warped > best_warp
        best_warp[better], top_warp[better] = warped[better], c
    return top_mix, top_warp, dict(warped_channel_sum_max_abs_dev=float(np.max(np.abs(warped_sum-1))), voxels_pulled_from_outside_grid=outside)


def reduction(fused, base):
    """1 - (1 - fused)/(1 - base); nan for a perfect base."""
    return float("nan") if base >= 1 else 1-(1-fused)/(1-base)


def row_of(pred, true35, true4, lookup):
    row = eval_seg.summarize_subject(pred, true35, true4, lookup)
    row["seg4_cortex"], row["seg4_wm"] = row["seg4_per_label"][common.SEG4_CORTEX-1], row["seg4_per_label"][common.SEG4_WM-1]
    return row


def value(row, metric):
    return row["groups"][metric] if metric in eval_seg.GROUPS else row[metric]


def error_flow(top_fixed, top_fused, top_warp, truth):
    """Voxel counts on the union of foreground (truth or a prediction non-zero): what the fusion corrected and what it broke; how often the pulled-back moving repeats the wrong label of the fixed prediction."""
    foreground = (truth > 0) | (top_fixed > 0) | (top_fused > 0)
    wrong_fixed = (top_fixed != truth) & foreground
    wrong_fused = (top_fused != truth) & foreground
    corrected, broken = int((wrong_fixed & ~wrong_fused).sum()), int((~wrong_fixed & wrong_fused).sum())
    n_wrong = int(wrong_fixed.sum())
    same_wrong = int((wrong_fixed & (top_warp == top_fixed)).sum())
    warp_right = int((wrong_fixed & (top_warp == truth)).sum())
    return dict(foreground=int(foreground.sum()), wrong_fixed=n_wrong, wrong_fused=int(wrong_fused.sum()), corrected=corrected, broken=broken, net_voxels=corrected-broken,
                voxel_reduction=(corrected-broken)/n_wrong if n_wrong else float("nan"), warped_repeats_fixed_label_at_fixed_errors=same_wrong/n_wrong if n_wrong else float("nan"),
                warped_right_at_fixed_errors=warp_right/n_wrong if n_wrong else float("nan"))


def summary_markdown(document):
    f = eval_seg.fmt
    lines = ["# SEGP P3：配对融合（开发集前 5 对，单次观测；分割级诊断，不是配准结果）", "", "场由这两份预测驱动，错误可能相关。只有一种融合。", "",
             "| 对（fixed←moving） | 只用 fixed seg35 / seg4 | 融合 seg35 / seg4 | 差 | 减少比例 seg35 / seg4 | moving 拉回 seg35 / seg4 |", "|---|---|---|---|---|---|"]
    for row in document["pairs"]:
        a, b, c = row["rows"]["fixed"], row["rows"]["fused"], row["rows"]["moving_warped"]
        lines.append(f"| {row['fixed'][11:15]}←{row['moving'][11:15]} | {f(a['seg35'])} / {f(a['seg4'])} | {f(b['seg35'])} / {f(b['seg4'])} | {b['seg35']-a['seg35']:+.4f} / {b['seg4']-a['seg4']:+.4f} | "
                     f"{100*row['reduction']['seg35']:.1f}% / {100*row['reduction']['seg4']:.1f}% | {f(c['seg35'])} / {f(c['seg4'])} |")
    r = document["reading"]
    lines += ["", f"均值：只用 fixed {f(document['means']['fixed']['seg35'])} / {f(document['means']['fixed']['seg4'])}，融合 {f(document['means']['fused']['seg35'])} / {f(document['means']['fused']['seg4'])}。",
              f"减少比例（seg35）：均值的比 {100*r['seg35_ratio_of_means']:.1f}%，比的均值 {100*r['seg35_mean_of_ratios']:.1f}%；（seg4）{100*r['seg4_ratio_of_means']:.1f}%，{100*r['seg4_mean_of_ratios']:.1f}%。线 {100*READING_BAR:.0f}%：{r['text']}。", ""]
    return "\n".join(lines)


def run(*, out_dir, argv=None, log=print, data_root=eval_seg.DATA_ROOT):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists and is not empty; results are never overwritten")
    subjects, admission = common.admit_dev_subjects(data_root=data_root)
    for pair in PAIRS:
        for subject in pair:
            if subject not in subjects:
                raise eval_seg.SubjectSealed(f"{subject} is not an admitted development subject")
    lookup = eval_seg.load_seg4_lookup()
    results = []
    started = time.perf_counter()
    for fixed, moving in PAIRS:
        run_dir = RUNS/f"{fixed}__{moving}"
        metrics = json.loads((run_dir/"metrics.json").read_text())
        if metrics.get("arm") != "S_P" or metrics.get("pair") != run_dir.name or metrics.get("classification", {}).get("class") != "ok":
            raise ValueError(f"{run_dir}: not an ok S_P run of this pair")
        u, spacing, field = check_field(run_dir)
        d = disp_voxels(u, spacing).astype(np.float64)
        del u
        true_f35, sha_f35 = eval_seg.read_truth(data_root, fixed, "seg35")
        true_f4, sha_f4 = eval_seg.read_truth(data_root, fixed, "seg4")
        true_m35, sha_m35 = eval_seg.read_truth(data_root, moving, "seg35")
        true_m4, sha_m4 = eval_seg.read_truth(data_root, moving, "seg4")
        direction = check_direction(d, true_f35, true_f4, true_m35, true_m4, metrics["metrics"])
        del true_m35, true_m4
        raw_f, record_f = common.load_raw(common.V1_ROOT, fixed, expected_segmenter=common.V1_ID)
        raw_m, record_m = common.load_raw(common.V1_ROOT, moving, expected_segmenter=common.V1_ID)
        top_f = common.argmax_of_mean([raw_f])
        top_fused, top_warp, warp_record = fuse(raw_f, raw_m, d)
        del raw_f, raw_m, d
        rows = dict(fixed=row_of(top_f, true_f35, true_f4, lookup), fused=row_of(top_fused, true_f35, true_f4, lookup), moving_warped=row_of(top_warp, true_f35, true_f4, lookup))
        reductions = {m: reduction(value(rows["fused"], m), value(rows["fixed"], m)) for m in METRICS}
        flow = error_flow(top_f, top_fused, top_warp, true_f35)
        results.append(dict(fixed=fixed, moving=moving, run_id=metrics.get("run_id"), field=field, direction=direction, warp=warp_record, predictions=dict(fixed=record_f, moving=record_m),
                            truth_sha256=dict(fixed_seg35=sha_f35, fixed_seg4=sha_f4, moving_seg35=sha_m35, moving_seg4=sha_m4), rows=rows, reduction=reductions, error_flow=flow))
        log(f"{fixed[11:15]}<-{moving[11:15]}: direction ok (diff {direction['max_abs_diff']:.1e}, seg35 with -d {direction['seg35_with_minus_d']:.4f}); fixed {rows['fixed']['seg35']:.4f} / {rows['fixed']['seg4']:.4f}, "
            f"fused {rows['fused']['seg35']:.4f} / {rows['fused']['seg4']:.4f}, warped {rows['moving_warped']['seg35']:.4f} / {rows['moving_warped']['seg4']:.4f}, reduction {100*reductions['seg35']:.1f}% / {100*reductions['seg4']:.1f}%, "
            f"peak RSS {common.peak_rss_gib():.2f} GiB, {time.perf_counter()-started:.0f} s", flush=True)
    means = {arm: {m: eval_seg.nanmean([value(r["rows"][arm], m) for r in results]) for m in METRICS} for arm in ARMS}
    paired = {m: eval_seg.paired([value(r["rows"]["fused"], m) for r in results], [value(r["rows"]["fixed"], m) for r in results]) for m in METRICS}
    ratio_of_means = {m: reduction(means["fused"][m], means["fixed"][m]) for m in METRICS}
    mean_of_ratios = {m: eval_seg.nanmean([r["reduction"][m] for r in results]) for m in METRICS}
    both = min(ratio_of_means["seg35"], mean_of_ratios["seg35"]) >= READING_BAR
    either = max(ratio_of_means["seg35"], mean_of_ratios["seg35"]) >= READING_BAR
    reading = dict(bar=READING_BAR, seg35_ratio_of_means=ratio_of_means["seg35"], seg35_mean_of_ratios=mean_of_ratios["seg35"], seg4_ratio_of_means=ratio_of_means["seg4"], seg4_mean_of_ratios=mean_of_ratios["seg4"],
                   text="seg35 的两个读法都过线，交总负责人" if both else ("seg35 只有一个读法过线，贴线，交总负责人" if either else "seg35 没有过线"))
    document = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=argv, admission=admission, means=means, paired=paired, reduction_ratio_of_means=ratio_of_means, reduction_mean_of_ratios=mean_of_ratios,
                    reading=reading, pairs=results, seconds=time.perf_counter()-started, peak_rss_gib=common.peak_rss_gib(),
                    note="开发集前 5 对，单次观测；分割级诊断，不是配准结果。相邻对共用受试者。场由这两份预测驱动，错误可能相关。")
    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(out_dir/"p3.json", document)
    with (out_dir/"per_pair.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fixed", "moving", "run_id", "arm", *METRICS])
        for r in results:
            for arm in ARMS:
                writer.writerow([r["fixed"], r["moving"], r["run_id"], arm]+[value(r["rows"][arm], m) for m in METRICS])
            writer.writerow([r["fixed"], r["moving"], r["run_id"], "reduction"]+[r["reduction"][m] for m in METRICS])
    (out_dir/"summary.md").write_text(summary_markdown(document), encoding="utf-8")
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=common.OUT_ROOT/"p3")
    args = parser.parse_args(argv)
    document = run(out_dir=args.out_dir, argv=sys.argv if argv is None else ["p3_fusion.py", *argv])
    print(json.dumps(dict(means=document["means"], reading=document["reading"]), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
