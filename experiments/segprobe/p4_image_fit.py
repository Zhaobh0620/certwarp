"""SEGP P4: do the labels of 0440 sit on the image? A check on the image alone, no network in the score (runs/20261105-segprobe/result.md). Written after the centroids of label_offset.json showed a z offset that grows toward the inferior brain.

For a label map L (the truth, or the v1 prediction) and a subject's image I: the model M = the mean intensity of I per label of L, piecewise constant. In three boxes (the inferior, middle and superior third of the extent of the image along z, on voxels with I > 0): the normalized correlation of I with M shifted by s voxels, s = -3..3 along each axis, and the position of its maximum, refined by a parabola through the three best points. A label map that sits on the image has its maximum near s = 0; a label map displaced against the image has it away from 0, whichever map the score comes from. Also the correlation at s = 0.
All 20 development subjects, read only. roll: a voxel shifted out of the volume re-enters on the other side, where the volume is background.

    python -B experiments/segprobe/p4_image_fit.py --out DIR/image_fit.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segprobe_common as common  # noqa: E402
from segprobe_common import eval_seg  # noqa: E402
from jd_curl.three_d.nifti import load_scalar  # noqa: E402
from p4_0440 import TARGET, group_stats  # noqa: E402

SHIFTS = tuple(range(-3, 4))
BOXES = ("inferior", "middle", "superior")


def model_of(image, labels, n=36):
    """The mean intensity of the image per label, as a volume."""
    labels = np.asarray(labels).astype(np.int64)
    counts = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    totals = np.bincount(labels.ravel(), weights=image.ravel(), minlength=n)
    means = np.divide(totals, counts, out=np.zeros(n), where=counts > 0)
    return means[labels]


def ncc(a, b):
    a, b = a-a.mean(), b-b.mean()
    denominator = np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/denominator) if denominator > 0 else float("nan")


def refine(scores):
    """The position of the maximum over SHIFTS, refined by a parabola through the three points around it; at the end of the range, the end."""
    best = int(np.argmax(scores))
    if 0 < best < len(scores)-1:
        left, middle, right = scores[best-1], scores[best], scores[best+1]
        curvature = left-2*middle+right
        return float(SHIFTS[best]+(.5*(left-right)/curvature if curvature < 0 else 0.))
    return float(SHIFTS[best])


def best_shifts(image, model, mask):
    """{box: {axis: position of the maximum}, ncc at 0}."""
    z_any = np.nonzero(mask.any(axis=(1, 2)))[0]
    edges = np.linspace(z_any[0], z_any[-1]+1, len(BOXES)+1).astype(int)
    out = {}
    for name, a, b in zip(BOXES, edges[:-1], edges[1:]):
        where = np.zeros_like(mask)
        where[a:b] = True
        where &= mask
        image_part = image[where]
        record = dict(ncc_at_0=ncc(image_part, model[where]))
        for axis, axis_name in enumerate("zyx"):
            scores = [ncc(image_part, np.roll(model, s, axis=axis)[where]) for s in SHIFTS]
            record[f"best_{axis_name}"] = refine(scores)
        out[name] = record
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=common.OUT_ROOT/"p4_followup/image_fit.json")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; results are never overwritten")
    subjects, _ = common.admit_dev_subjects()
    rows = {}
    for subject in subjects:
        image = load_scalar(Path(eval_seg.DATA_ROOT)/subject/eval_seg.IMAGE_FILE).values
        truth, _ = eval_seg.read_truth(eval_seg.DATA_ROOT, subject, "seg35")
        raw, _ = common.load_raw(common.V1_ROOT, subject, expected_segmenter=common.V1_ID)
        top = common.argmax_of_mean([raw])
        del raw
        mask = image > 0
        rows[subject] = dict(truth=best_shifts(image, model_of(image, truth), mask), pred=best_shifts(image, model_of(image, top), mask))
        r = rows[subject]
        print(subject[11:15], " ".join(f"{b}: truth z {r['truth'][b]['best_z']:+.2f} y {r['truth'][b]['best_y']:+.2f} x {r['truth'][b]['best_x']:+.2f} ncc {r['truth'][b]['ncc_at_0']:.4f}; pred z {r['pred'][b]['best_z']:+.2f} y {r['pred'][b]['best_y']:+.2f} x {r['pred'][b]['best_x']:+.2f} ncc {r['pred'][b]['ncc_at_0']:.4f} |" for b in BOXES), f"RSS {common.peak_rss_gib():.2f}", flush=True)
    stats = {}
    for labels in ("truth", "pred"):
        for box in BOXES:
            for key in ("ncc_at_0", "best_z", "best_y", "best_x"):
                flat = {s: {"v": rows[s][labels][box][key]} for s in subjects}
                stats[f"{labels}.{box}.{key}"] = group_stats(flat, ("v",))
    gap = {s: {f"{box}.pred_minus_truth_ncc": rows[s]["pred"][box]["ncc_at_0"]-rows[s]["truth"][box]["ncc_at_0"] for box in BOXES} for s in subjects}
    for box in BOXES:
        stats[f"gap.{box}.pred_minus_truth_ncc"] = group_stats(gap, (f"{box}.pred_minus_truth_ncc",))
    common.write_json(args.out, dict(subjects=subjects, target=TARGET, shifts=list(SHIFTS), stats=stats, rows=rows, note="开发集 20 人，只读。这项检查是看过 label_offset 之后加的。分数里没有网络，模型是逐标签平均强度。roll 的回绕落在背景里。"))
    for key, s in stats.items():
        print(f"{key}: 0440 {s['value']:.4f}, others median {s['others_median']:.4f}, range {s['others_min']:.4f} to {s['others_max']:.4f}, rank {s['rank_of_20_from_low']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
