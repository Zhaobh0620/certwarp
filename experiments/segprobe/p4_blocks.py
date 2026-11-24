"""SEGP P4: the block errors of 0440, what they are and where (runs/20261105-segprobe/result.md). The wrong voxels of v1 that survive an opening by a 3x3x3 cube, connected (26): a table of the largest parts (size, centroid, the truth and predicted labels,
the confidence, the intensity, the Atropos class) and a figure with crops through the three largest. For comparison the same table of the subject with the median seg35 among the other 19 is not needed there: they have almost no such voxels (features.json, v1.errors.share_in_thick_parts).
Read only.

    python -B experiments/segprobe/p4_blocks.py --features DIR/features.json --out-dir DIR
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segprobe_common as common  # noqa: E402
from segprobe_common import eval_seg  # noqa: E402
from jd_curl.three_d.nifti import load_scalar  # noqa: E402
from p4_0440 import CUBE, NAMES, TARGET, top_conf_ptrue  # noqa: E402
from p4_figures import BLUE, INK, MUTED, ORANGE, SEG4_COLOURS, SURFACE  # noqa: E402
from p4_followup import ATROPOS  # noqa: E402

CROP = 36


def modal(values):
    counts = np.bincount(values.astype(np.int64))
    return int(np.argmax(counts)), float(counts.max()/counts.sum())


def crop_of(volume, centre, axis):
    index = centre[axis]
    plane = np.take(volume, index, axis=axis)
    kept = [c for a, c in enumerate(centre) if a != axis]
    slices = tuple(slice(max(c-CROP, 0), c+CROP) for c in kept)
    return plane[slices]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=common.OUT_ROOT/"p4/features.json")
    parser.add_argument("--out-dir", type=Path, default=common.OUT_ROOT/"p4_blocks")
    args = parser.parse_args(argv)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"{args.out_dir} exists and is not empty; results are never overwritten")
    subjects, _ = common.admit_dev_subjects()
    if TARGET not in subjects:
        raise eval_seg.SubjectSealed(TARGET)
    intensity = json.loads(args.features.read_text())["rows"][TARGET]["intensity"]
    lookup = eval_seg.load_seg4_lookup()
    image = load_scalar(Path(eval_seg.DATA_ROOT)/TARGET/eval_seg.IMAGE_FILE).values
    truth, _ = eval_seg.read_truth(eval_seg.DATA_ROOT, TARGET, "seg35")
    raw, _ = common.load_raw(common.V1_ROOT, TARGET, expected_segmenter=common.V1_ID)
    top, _, conf, _ = top_conf_ptrue(raw, truth)
    del raw
    atropos = load_scalar(ATROPOS/TARGET/"labels.nii.gz", labels=True).values.astype(np.uint8)
    foreground = (top > 0) | (truth > 0)
    wrong = (top != truth) & foreground
    thick = ndimage.binary_opening(wrong, structure=CUBE)
    components, n = ndimage.label(thick, structure=CUBE)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    order = np.argsort(-sizes)[:12]
    span = intensity["wm_median"]-intensity["gm_median"]
    rows = []
    for k in order:
        if sizes[k] == 0:
            continue
        part = components == k
        centre = [int(np.round(i.mean())) for i in np.nonzero(part)]
        t, ts = modal(truth[part])
        p, ps = modal(top[part])
        rows.append(dict(size=int(sizes[k]), centroid_zyx=centre, truth_label=NAMES[t], truth_share=ts, pred_label=NAMES[p], pred_share=ps, conf_mean=float(conf[part].mean()), intensity_over_gm=float(image[part].mean()/intensity["gm_median"]),
                         normalized_intensity=float((image[part].mean()-intensity["gm_median"])/span), atropos_counts={c: int((atropos[part] == i).sum()) for i, c in enumerate(("outside_mask", "CSF", "GM", "WM"))}))
    total = int(thick.sum())
    pairs = np.bincount(truth[thick].astype(np.int64)*36+top[thick].astype(np.int64), minlength=36*36)
    composition = [dict(truth=NAMES[int(i)//36], pred=NAMES[int(i) % 36], voxels=int(pairs[i]), share=float(pairs[i]/total)) for i in np.argsort(-pairs)[:12] if pairs[i] > 0]
    by_class = {c: int((atropos[thick] == i).sum()) for i, c in enumerate(("outside_mask", "CSF", "GM", "WM"))}
    document = dict(subject=TARGET, n_wrong=int(wrong.sum()), n_thick=total, n_parts=int(n), composition=composition, atropos_on_thick=by_class, conf_on_thick_mean=float(conf[thick].mean()), normalized_intensity_on_thick_median=float(np.median((image[thick]-intensity["gm_median"])/span)), largest_parts=rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(args.out_dir/"blocks.json", document)
    print(json.dumps({k: document[k] for k in ("n_wrong", "n_thick", "n_parts", "composition", "atropos_on_thick", "conf_on_thick_mean", "normalized_intensity_on_thick_median")}, indent=0, ensure_ascii=False))
    for row in rows:
        print(row)
    vmax = float(np.percentile(image[image > 0], 99.5))
    fig, axes = plt.subplots(min(4, len(rows)), 4, figsize=(12, 3*min(4, len(rows))), facecolor=SURFACE, squeeze=False)
    for line, row in enumerate(rows[:4]):
        centre, axis = row["centroid_zyx"], 0
        i, t, p, c = crop_of(image, centre, axis), crop_of(truth, centre, axis), crop_of(top, centre, axis), crop_of(conf, centre, axis)
        panels = (("image", None), ("truth (seg4 colours)", lookup[t]), ("v1 (seg4 colours)", lookup[p]), ("v1 confidence", None))
        for column, (title, labels) in enumerate(panels):
            ax = axes[line, column]
            ax.imshow(i, cmap="gray", vmin=0, vmax=vmax, origin="lower", interpolation="nearest")
            if labels is not None:
                ax.imshow(labels, cmap=SEG4_COLOURS, vmin=0, vmax=4, origin="lower", interpolation="nearest", alpha=(labels > 0)*.6)
                ax.contour(crop_of(thick, centre, axis), levels=[.5], colors=BLUE if column == 1 else ORANGE, linewidths=1.)
            elif column == 3:
                ax.imshow(c, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if line == 0:
                ax.set_title(title, fontsize=9, color=INK)
            if column == 0:
                ax.set_ylabel(f"part {line+1}: {row['size']} voxels\nz = {centre[0]}", fontsize=8, color=MUTED)
    fig.suptitle("0440: axial crops through the largest thick error parts of v1 (contour: the part; development subject)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(args.out_dir/"blocks_0440.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print(f"peak RSS {common.peak_rss_gib():.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
