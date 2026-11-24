"""SEGP P4: is the truth of 0440 displaced against its image? (runs/20261105-segprobe/result.md). The integer shifts of features.json show no better shift; a sub-voxel translation, a small scale or rotation would not show there.

For all 20 development subjects: the centroid of each of the 35 labels in the truth and in the v1 prediction (which is read from the image alone); the offset truth - prediction per label; the mean offset; a least-squares affine map from the predicted to the true centroids,
over the labels (weighted by the true volume, labels with fewer than 200 voxels left out): its translation at the brain centre, its linear part minus the identity, its scale and the rest of the offsets. A displacement of the labels against the image would show as an affine with a non-trivial part
that the other 19 do not have. Read only.

    python -B experiments/segprobe/p4_label_offset.py --out DIR/label_offset.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segprobe_common as common  # noqa: E402
from segprobe_common import eval_seg  # noqa: E402
from p4_0440 import NAMES, TARGET, group_stats  # noqa: E402

MIN_VOXELS = 200


def centroids(labels, n=36):
    """(n, 3) mean index along (z, y, x), counts."""
    labels = np.asarray(labels).astype(np.int64)
    counts = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    out = np.full((n, 3), np.nan)
    for axis, size in enumerate(labels.shape):
        shape = [1, 1, 1]
        shape[axis] = size
        coordinate = np.broadcast_to(np.arange(size, dtype=np.float64).reshape(shape), labels.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, axis] = np.bincount(labels.ravel(), weights=coordinate.ravel(), minlength=n)/counts
    return out, counts


def fit_affine(source, target, weights):
    """Weighted least squares target ~ A (source - centre) + t, centre = the weighted mean of the source. Returns (A, t, residuals, centre)."""
    w = weights/weights.sum()
    centre = (w[:, None]*source).sum(axis=0)
    design = np.hstack([source-centre, np.ones((len(source), 1))])
    root = np.sqrt(w)[:, None]
    solution, *_ = np.linalg.lstsq(design*root, target*root, rcond=None)
    return solution[:3].T, solution[3], target-design@solution, centre


def subject_record(truth, top):
    true_c, true_n = centroids(truth)
    pred_c, pred_n = centroids(top)
    keep = [l for l in range(1, 36) if true_n[l] >= MIN_VOXELS and pred_n[l] >= MIN_VOXELS]
    offsets = true_c[keep]-pred_c[keep]
    weights = true_n[keep]
    linear, translation, residual, centre = fit_affine(pred_c[keep], true_c[keep], weights)
    w = weights/weights.sum()
    return dict(n_labels=len(keep), mean_offset_zyx_voxels=(w[:, None]*offsets).sum(axis=0).tolist(), unweighted_mean_offset_zyx_voxels=offsets.mean(axis=0).tolist(), median_label_offset_norm=float(np.median(np.linalg.norm(offsets, axis=1))),
                affine=dict(translation_at_centre_zyx=(translation-centre).tolist(), linear_minus_identity_fro=float(np.linalg.norm(linear-np.eye(3))), scale=float(np.linalg.det(linear)**(1/3)), residual_rms_voxels=float(np.sqrt((w*(residual**2).sum(axis=1)).sum()))),
                per_label_offset_zyx={NAMES[l]: offsets[k].tolist() for k, l in enumerate(keep)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=common.OUT_ROOT/"p4_followup/label_offset.json")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; results are never overwritten")
    subjects, _ = common.admit_dev_subjects()
    rows = {}
    for subject in subjects:
        truth, _ = eval_seg.read_truth(eval_seg.DATA_ROOT, subject, "seg35")
        raw, _ = common.load_raw(common.V1_ROOT, subject, expected_segmenter=common.V1_ID)
        top = common.argmax_of_mean([raw])
        del raw
        rows[subject] = subject_record(truth, top)
        r = rows[subject]
        print(f"{subject[11:15]}: mean offset (z, y, x) {np.round(r['mean_offset_zyx_voxels'], 3).tolist()}, median label offset {r['median_label_offset_norm']:.3f}, affine: scale {r['affine']['scale']:.4f}, |A - I| {r['affine']['linear_minus_identity_fro']:.4f}, residual {r['affine']['residual_rms_voxels']:.3f}", flush=True)
    flat = {s: dict(median_label_offset_norm=r["median_label_offset_norm"], mean_offset_norm=float(np.linalg.norm(r["mean_offset_zyx_voxels"])), scale=r["affine"]["scale"], linear_minus_identity_fro=r["affine"]["linear_minus_identity_fro"], residual_rms_voxels=r["affine"]["residual_rms_voxels"],
                    mean_offset_z=r["mean_offset_zyx_voxels"][0], mean_offset_y=r["mean_offset_zyx_voxels"][1], mean_offset_x=r["mean_offset_zyx_voxels"][2]) for s, r in rows.items()}
    stats = {k: group_stats(flat, (k,)) for k in flat[TARGET]}
    target = rows[TARGET]["per_label_offset_zyx"]
    largest = sorted(target, key=lambda name: -np.linalg.norm(target[name]))[:10]
    common.write_json(args.out, dict(subjects=subjects, stats=stats, rows=rows, largest_offsets_0440={name: target[name] for name in largest}, note="开发集 20 人，只读。预测是只看图像的 v1 集成，不是真值；这里看的是真值相对图像是否整体偏移。"))
    for key, s in stats.items():
        print(f"{key}: 0440 {s['value']:.4f}, others median {s['others_median']:.4f}, range {s['others_min']:.4f} to {s['others_max']:.4f}, z {s['robust_z'] if s['robust_z'] is None else round(s['robust_z'], 1)}, rank {s['rank_of_20_from_low']}")
    print("largest label offsets of 0440:", {name: np.round(target[name], 2).tolist() for name in largest})
    return 0


if __name__ == "__main__":
    sys.exit(main())
