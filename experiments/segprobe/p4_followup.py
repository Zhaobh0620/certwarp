"""SEGP P4, the follow-up after the first read of features.json (runs/20261105-segprobe/result.md): cheap checks of the three candidate causes, the same features for all 20 development subjects. Read only.

  arbiter    Atropos (seg/atropos_v1/<subject>/labels.nii.gz, three classes CSF, GM, WM, EM on the image alone, no label seen) on the voxels where the v1 prediction and the truth disagree: whose side is it on?
             Agreement of Atropos GM and WM with the truth and with the prediction on the voxels inside the Atropos mask. It is an intensity model, not a truth; a reading of the same kind on all 20.
  surfaces   the intensity profile across the outer surface of the parenchyma (seg4 classes 1, 2, 3) of the truth and of the prediction: where the image falls to half of the cortex median; the thickness proxy of the prediction
  volumes    the foreground and the cortex of the prediction against the truth
  sharpness  the share of the spectral power above .2, .3, .4 cycles per voxel in the aligned image and in the scanner-space norm.nii.gz (a blurred image has less), inside a box around the brain
  residue    bright voxels outside the truth's foreground (non-brain tissue the skull stripping left): their share, how far from the brain
  across19   Spearman, across the other 19 only, of seg35 of v1 against native volume, cortex share, thickness, ventricles, scale, and the linear extrapolation of the 19 to the value of 0440

Outputs in --out-dir (never overwritten): followup.json, summary.md.

    python -B experiments/segprobe/p4_followup.py --features DIR/features.json --out-dir DIR
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage, stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segprobe_common as common  # noqa: E402
from segprobe_common import eval_seg  # noqa: E402
from jd_curl.three_d.nifti import load_scalar  # noqa: E402
from p4_0440 import TARGET, crossing, group_stats, wm_profile  # noqa: E402

FORMAT = "bayesreg.segprobe.p4_followup.v1"
ATROPOS = Path("/home/rmcfz/seq_vp_tag_results/seg/atropos_v1")
CSF, GM, WM = 1, 2, 3                       # Atropos labels: class_names ["CSF", "GM", "WM"], file index base 1 (provenance.json)
CUTS = (.2, .3, .4)
ACROSS = ("anatomy.native_volume_mm3", "anatomy.cortex_share", "anatomy.cortex_over_wm", "anatomy.thickness_proxy", "anatomy.ventricle_share", "anatomy.linear_scale_of_alignment", "intensity.cnr_wm_gm", "intensity.noise_wm_high_pass_sd_over_wm")


def votes(mask, atropos, for_a, for_b):
    """On `mask`: the counts of Atropos classes, and the share of the voxels with a vote for side a or b that side with a."""
    n = int(mask.sum())
    a, b = int(np.isin(atropos[mask], for_a).sum()), int(np.isin(atropos[mask], for_b).sum())
    return dict(n=n, for_a=a, for_b=b, share_for_a=a/(a+b) if a+b else float("nan"))


def arbiter(atropos, truth4, pred4):
    """seg4 classes: 1 cortex, 2 subcortical, 3 WM, 4 CSF. In each disagreement `a` is the prediction and `b` the truth."""
    within = atropos > 0
    out = dict(pred_cortex_truth_wm=votes((pred4 == 1) & (truth4 == 3) & within, atropos, (GM,), (WM,)), pred_wm_truth_cortex=votes((pred4 == 3) & (truth4 == 1) & within, atropos, (WM,), (GM,)),
               pred_cortex_truth_background=votes((pred4 == 1) & (truth4 == 0) & within, atropos, (GM,), (CSF,)), pred_background_truth_cortex=votes((pred4 == 0) & (truth4 == 1) & within, atropos, (CSF,), (GM,)))

    def dice(a, b):
        return float(2*(a & b).sum()/max(1, a.sum()+b.sum()))

    gm_a, wm_a, csf_a = atropos == GM, atropos == WM, atropos == CSF
    for name, labels in (("truth", truth4), ("pred", pred4)):
        out[f"dice_atropos_gm_{name}"] = dice(gm_a & within, np.isin(labels, (1, 2)) & within)
        out[f"dice_atropos_cortex_{name}"] = dice(gm_a & within, (labels == 1) & within)
        out[f"dice_atropos_wm_{name}"] = dice(wm_a & within, (labels == 3) & within)
    out["share_of_atropos_csf_in_truth_foreground"] = float((csf_a & (truth4 > 0)).sum()/max(1, csf_a.sum()))
    return out


def thickness_proxy(labels4):
    cortex, wm = labels4 == 1, labels4 == 3
    contact = cortex & ndimage.binary_dilation(wm, structure=ndimage.generate_binary_structure(3, 1))
    return float(cortex.sum()/contact.sum())


def surfaces(image, truth4, pred4, gm_median):
    half = .5*gm_median
    out = {}
    for name, labels in (("truth", truth4), ("pred", pred4)):
        inside = np.isin(labels, (1, 2, 3))
        profile = wm_profile(image, inside, ~inside)
        position, slope = crossing(profile, half)
        out[f"outer_crossing_{name}"] = position                # positive: the image falls to half of the cortex median inside this surface; negative: outside it
        out[f"outer_slope_over_gm_{name}"] = slope/gm_median
        out[f"outer_profile_{name}"] = {str(k): v/gm_median for k, v in sorted(profile.items())}
        out[f"thickness_{name}"] = thickness_proxy(labels)
    return out


def spectral_shares(volume, mask=None):
    """The share of the spectral power above each cut (cycles per voxel), in a box around the non-zero part, with its mean removed. `mask` is optional and defines the box."""
    where = np.nonzero(volume > 0 if mask is None else mask)
    box = tuple(slice(int(i.min()), int(i.max())+1) for i in where)
    part = np.asarray(volume[box], dtype=np.float64)
    part = part-part.mean()
    power = np.abs(np.fft.rfftn(part))**2
    freq = [np.fft.fftfreq(n) for n in part.shape[:-1]]+[np.fft.rfftfreq(part.shape[-1])]
    radius2 = freq[0][:, None, None]**2+freq[1][None, :, None]**2+freq[2][None, None, :]**2
    total = power.sum()
    return {f"above_{cut}": float(power[radius2 > cut*cut].sum()/total) for cut in CUTS}


def residue(image, truth, wm_median):
    foreground = truth > 0
    bright = (image > 1.15*wm_median) & ~foreground
    distance = ndimage.distance_transform_edt(~foreground)
    n = int(foreground.sum())
    components, n_components = ndimage.label(bright, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(components.ravel())[1:] if n_components else np.array([0])
    tissue_outside = (image > .5*wm_median) & ~foreground
    return dict(bright_outside_share=float(bright.sum()/n), bright_outside_far_share=float((bright & (distance > 2)).sum()/n), bright_outside_largest=int(sizes.max()), tissue_outside_share=float(tissue_outside.sum()/n),
                tissue_outside_far_share=float((tissue_outside & (distance > 2)).sum()/n), truth_background_positive_far_share=float(((image > 0) & (distance > 2)).sum()/n))


def read_path(row, path):
    node = row
    for key in path.split("."):
        node = node[key]
    return node


def across_others(rows):
    """seg35 of v1 against each feature, across the other 19 only: Spearman and the line of the 19 evaluated at 0440."""
    others = [s for s in rows if s != TARGET]
    y = np.array([rows[s]["v1"]["seg35"] for s in others])
    out = {}
    for path in ACROSS:
        x = np.array([read_path(rows[s], path) for s in others], dtype=np.float64)
        rho, p = scipy_stats.spearmanr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        at = float(read_path(rows[TARGET], path))
        out[path] = dict(spearman=float(rho), p=float(p), slope=float(slope), value_0440=at, line_at_0440=float(slope*at+intercept), others_mean_seg35=float(y.mean()), seg35_0440=rows[TARGET]["v1"]["seg35"])
    return out


def describe(follow):
    table = {}

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path+(key,))
        elif isinstance(node, (int, float)) and not isinstance(node, bool) and np.isfinite(node):
            table[".".join(path)] = group_stats(follow, path)

    walk(follow[TARGET], ())
    return table


def summary_markdown(document):
    stats = document["stats"]
    keys = [k for k in stats if "outer_profile" not in k]
    lines = ["# SEGP P4 后续：0440 对其余 19 人（开发集，单次观测；只读描述）", "", "| 特征 | 0440 | 其余中位 | 其余范围 | z | 名次（从低到高，共 20） |", "|---|---:|---:|---|---:|---:|"]
    for key in keys:
        s = stats[key]
        z = "n/a" if s["robust_z"] is None else f"{s['robust_z']:+.1f}"
        lines.append(f"| {key} | {s['value']:.5g} | {s['others_median']:.5g} | {s['others_min']:.5g} 到 {s['others_max']:.5g} | {z} | {s['rank_of_20_from_low']} |")
    lines += ["", "## 其余 19 人的 seg35（v1）对特征：Spearman 与直线外推到 0440", "", "| 特征 | Spearman | p | 0440 的值 | 直线在 0440 处 | 0440 实际 | 其余均值 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for path, row in document["across_others"].items():
        lines.append(f"| {path} | {row['spearman']:+.3f} | {row['p']:.3f} | {row['value_0440']:.5g} | {row['line_at_0440']:.4f} | {row['seg35_0440']:.4f} | {row['others_mean_seg35']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def run(*, features, out_dir, argv=None, log=print, data_root=eval_seg.DATA_ROOT):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists and is not empty; results are never overwritten")
    subjects, admission = common.admit_dev_subjects(data_root=data_root)
    lookup = eval_seg.load_seg4_lookup()
    first = json.loads(Path(features).read_text())
    if first["subjects"] != subjects:
        raise ValueError("features.json is of other subjects")
    started = time.perf_counter()
    follow = {}
    for subject in subjects:
        directory = Path(data_root)/subject
        image = load_scalar(directory/eval_seg.IMAGE_FILE).values
        truth, _ = eval_seg.read_truth(data_root, subject, "seg35")
        truth4, _ = eval_seg.read_truth(data_root, subject, "seg4")
        raw, _ = common.load_raw(common.V1_ROOT, subject, expected_segmenter=common.V1_ID)
        top = common.argmax_of_mean([raw])
        del raw
        pred4 = lookup[top].astype(np.uint8)
        atropos_volume = load_scalar(ATROPOS/subject/"labels.nii.gz", labels=True)
        if atropos_volume.values.shape != image.shape:
            raise ValueError(f"{subject}: the Atropos labels have the shape {atropos_volume.values.shape}, not {image.shape}")
        atropos = atropos_volume.values.astype(np.uint8)
        intensity = first["rows"][subject]["intensity"]
        foreground = truth > 0
        native = np.asarray(nib.load(str(directory/"norm.nii.gz")).dataobj, dtype=np.float64)
        follow[subject] = dict(arbiter=arbiter(atropos, truth4, pred4), surfaces=surfaces(image, truth4, pred4, intensity["gm_median"]),
                               volumes=dict(foreground_pred_over_truth=float((top > 0).sum()/foreground.sum()), cortex_pred_over_truth=float((pred4 == 1).sum()/(truth4 == 1).sum()), wm_pred_over_truth=float((pred4 == 3).sum()/(truth4 == 3).sum())),
                               sharpness=dict(aligned=spectral_shares(image, foreground), native=spectral_shares(native)), residue=residue(image, truth, intensity["wm_median"]))
        del native, image
        arb = follow[subject]["arbiter"]
        log(f"{subject}: Atropos with the prediction: cortex/WM {arb['pred_cortex_truth_wm']['share_for_a']:.3f}, WM/cortex {arb['pred_wm_truth_cortex']['share_for_a']:.3f}, cortex/background {arb['pred_cortex_truth_background']['share_for_a']:.3f}; "
            f"outer crossing truth {follow[subject]['surfaces']['outer_crossing_truth']:.3f}, pred {follow[subject]['surfaces']['outer_crossing_pred']:.3f}; peak RSS {common.peak_rss_gib():.2f} GiB, {time.perf_counter()-started:.0f} s", flush=True)
    document = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=argv, admission=admission, features=str(features), features_sha256=eval_seg.sha_file(features), subjects=subjects, stats=describe(follow),
                    across_others=across_others(first["rows"]), rows=follow, seconds=time.perf_counter()-started, peak_rss_gib=common.peak_rss_gib(), note="开发集 20 人的描述，单次观测；只读。Atropos 是只看强度的模型，不是真值。这些检查是看过 features.json 之后加的。")
    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(out_dir/"followup.json", document)
    (out_dir/"summary.md").write_text(summary_markdown(document), encoding="utf-8")
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=common.OUT_ROOT/"p4/features.json")
    parser.add_argument("--out-dir", type=Path, default=common.OUT_ROOT/"p4_followup")
    args = parser.parse_args(argv)
    document = run(features=args.features, out_dir=args.out_dir, argv=sys.argv if argv is None else ["p4_followup.py", *argv])
    print(f"done, {document['seconds']:.0f} s, peak RSS {document['peak_rss_gib']:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
