"""SEGP P4, the figures of subject 0440 and of a comparison subject (the one with the median v1 seg35 among the other 19), from the same files as p4_0440.py. Read only; the figures go to the results directory, not to git.

  slices_<id>.png        three planes through the brain centre, image / truth / v1 prediction / the errors of v1 and v2 (orange: v1 only, blue: v2 only, black: both)
  worst_<id>.png         the same on the planes where the error rate of v1 is highest
  zoom_<id>.png          a coronal crop: the WM surface of the truth (blue) and of the v1 prediction (orange) on the image
  group.png              intensity histograms of the brain, the profile across the WM surface, the error rate along z, y, x: 0440 against each of the other 19

    python -B experiments/segprobe/p4_figures.py --features DIR/features.json --out-dir DIR
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segprobe_common as common  # noqa: E402
from segprobe_common import eval_seg  # noqa: E402
from jd_curl.three_d.nifti import load_scalar  # noqa: E402

TARGET = "OASIS_OAS1_0440_MR1"
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#b5b4ae"
SEG4_COLOURS = ListedColormap(["#000000", ORANGE, AQUA, "#f0efec", BLUE])       # 0 background, 1 cortex, 2 subcortical, 3 white matter, 4 CSF
PLANES = (("axial z", 0), ("coronal y", 1), ("sagittal x", 2))


def plane(volume, axis, index):
    """A slice shown with the superior or anterior side up."""
    part = np.take(volume, index, axis=axis)
    if axis == 0:                                        # (y, x)
        return part
    return part                                          # (z, x) or (z, y): origin lower puts z upward


def load_subject(subject, data_root=eval_seg.DATA_ROOT):
    image = load_scalar(Path(data_root)/subject/eval_seg.IMAGE_FILE).values
    truth, _ = eval_seg.read_truth(data_root, subject, "seg35")
    raw1, _ = common.load_raw(common.V1_ROOT, subject, expected_segmenter=common.V1_ID)
    top1 = common.argmax_of_mean([raw1])
    del raw1
    raw2, _ = common.load_raw(common.V2_ROOT, subject, expected_segmenter=common.V2_ID)
    top2 = common.argmax_of_mean([raw2])
    del raw2
    return image, truth, top1, top2


def panels(image, truth, top1, top2, lookup, indices, title, path):
    fig, axes = plt.subplots(3, 4, figsize=(13, 10.5), facecolor=SURFACE)
    vmax = float(np.percentile(image[image > 0], 99.5))
    for row, ((name, axis), index) in enumerate(zip(PLANES, indices)):
        slices = [plane(a, axis, index) for a in (image, truth, top1, top2)]
        wrong1, wrong2 = slices[2] != slices[1], slices[3] != slices[1]
        columns = [("image", None), ("truth (seg4 colours)", lookup[slices[1]]), ("v1 (seg4 colours)", lookup[slices[2]]), ("errors on seg35: orange v1 only, blue v2 only, black both", None)]
        for column, (label, labels) in enumerate(columns):
            ax = axes[row, column]
            ax.imshow(slices[0], cmap="gray", vmin=0, vmax=vmax, origin="lower", interpolation="nearest")
            if labels is not None:
                ax.imshow(labels, cmap=SEG4_COLOURS, vmin=0, vmax=4, origin="lower", interpolation="nearest", alpha=(labels > 0)*.75)
            elif column == 3:
                overlay = np.zeros(slices[0].shape+(4,))
                overlay[wrong1 & ~wrong2] = matplotlib.colors.to_rgba(ORANGE)
                overlay[wrong2 & ~wrong1] = matplotlib.colors.to_rgba(BLUE)
                overlay[wrong1 & wrong2] = (0, 0, 0, 1)
                background = np.ones(slices[0].shape)
                ax.imshow(background*0, cmap="gray", vmin=-1, vmax=0, origin="lower")
                ax.imshow(slices[0], cmap="gray", vmin=-vmax*.3, vmax=vmax*1.8, origin="lower", interpolation="nearest")
                ax.imshow(overlay, origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(label, fontsize=9, color=INK)
            if column == 0:
                ax.set_ylabel(f"{name} = {index}", fontsize=10, color=MUTED)
    fig.suptitle(title, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def wm(labels):
    return np.isin(labels, (1, 20))


def zoom(image, truth, top1, index, crop, title, path):
    (a, b), (c, d) = crop
    part = np.take(image, index, axis=1)[a:b, c:d]
    true_wm, pred_wm = np.take(wm(truth), index, axis=1)[a:b, c:d], np.take(wm(top1), index, axis=1)[a:b, c:d]
    vmax = float(np.percentile(image[image > 0], 99.5))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor=SURFACE)
    axes[0].imshow(part, cmap="gray", vmin=0, vmax=vmax, origin="lower", interpolation="nearest")
    axes[0].set_title("image", fontsize=10, color=INK)
    axes[1].imshow(part, cmap="gray", vmin=0, vmax=vmax, origin="lower", interpolation="nearest")
    axes[1].contour(true_wm, levels=[.5], colors=BLUE, linewidths=1.2)
    axes[1].set_title("WM surface of the truth", fontsize=10, color=INK)
    axes[2].imshow(part, cmap="gray", vmin=0, vmax=vmax, origin="lower", interpolation="nearest")
    axes[2].contour(pred_wm, levels=[.5], colors=ORANGE, linewidths=1.2)
    axes[2].set_title("WM surface of the v1 prediction", fontsize=10, color=INK)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def group_figure(rows, subjects, histograms, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), facecolor=SURFACE)
    ax = axes[0, 0]
    for subject, (centres, density) in histograms.items():
        if subject != TARGET:
            ax.plot(centres, density, color=GRAY, linewidth=1)
    ax.plot(*histograms[TARGET], color=ORANGE, linewidth=2)
    ax.set_title("brain intensity, divided by the WM median of the subject", fontsize=10, color=INK)
    ax.set_xlabel("I / WM median", fontsize=9, color=MUTED)
    ax.set_ylabel("density", fontsize=9, color=MUTED)
    ax.text(.02, .95, "orange: 0440; gray: the other 19", transform=ax.transAxes, fontsize=9, color=MUTED, va="top")

    ax = axes[0, 1]
    for subject in subjects:
        profile = rows[subject]["labels_v1"]["profile_truth"]
        ds = sorted(int(k) for k in profile)
        span = rows[subject]["intensity"]["wm_median"]-rows[subject]["intensity"]["gm_median"]
        values = [(profile[str(k)]-rows[subject]["intensity"]["gm_median"])/span for k in ds]
        if subject == TARGET:
            target_line = (ds, values)
        else:
            ax.plot(ds, values, color=GRAY, linewidth=1)
    ax.plot(*target_line, color=ORANGE, linewidth=2)
    ax.axhline(.5, color=MUTED, linewidth=.8, linestyle=":")
    ax.set_title("across the WM surface of the truth (0 cortex median, 1 WM median)", fontsize=10, color=INK)
    ax.set_xlabel("signed layer (+ inside WM, - in cortex)", fontsize=9, color=MUTED)

    ax = axes[0, 2]
    for subject in subjects:
        profile = rows[subject]["labels_v1"]["profile_pred"]
        ds = sorted(int(k) for k in profile)
        span = rows[subject]["intensity"]["wm_median"]-rows[subject]["intensity"]["gm_median"]
        values = [(profile[str(k)]-rows[subject]["intensity"]["gm_median"])/span for k in ds]
        if subject == TARGET:
            target_line = (ds, values)
        else:
            ax.plot(ds, values, color=GRAY, linewidth=1)
    ax.plot(*target_line, color=ORANGE, linewidth=2)
    ax.axhline(.5, color=MUTED, linewidth=.8, linestyle=":")
    ax.set_title("across the WM surface of the v1 prediction", fontsize=10, color=INK)
    ax.set_xlabel("signed layer (+ inside WM, - in cortex)", fontsize=9, color=MUTED)

    for ax, name, label in zip(axes[1], "zyx", ("along z (inferior to superior)", "along y (posterior to anterior)", "along x (low to high)")):
        for subject in subjects:
            values = rows[subject]["v1"]["location"][f"rate_along_{name}"]
            if subject == TARGET:
                target_values = values
            else:
                ax.plot(values, color=GRAY, linewidth=1)
        ax.plot(target_values, color=ORANGE, linewidth=2)
        ax.set_title(f"error rate of v1 {label}", fontsize=10, color=INK)
        ax.set_xlabel("slab (16 equal slabs of the volume)", fontsize=9, color=MUTED)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRAY)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=common.OUT_ROOT/"p4/features.json")
    parser.add_argument("--out-dir", type=Path, default=common.OUT_ROOT/"p4/figures")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    document = json.loads(args.features.read_text())
    rows, subjects = document["rows"], document["subjects"]
    others = sorted((rows[s]["v1"]["seg35"], s) for s in subjects if s != TARGET)
    comparison = others[len(others)//2][1]
    lookup = eval_seg.load_seg4_lookup()
    print(f"comparison subject: {comparison}", flush=True)
    for subject in (TARGET, comparison):
        image, truth, top1, top2 = load_subject(subject)
        brain = np.nonzero(truth > 0)
        centre = [int(np.round(i.mean())) for i in brain]
        short = subject[11:15]
        title = f"{short}: v1 {rows[subject]['v1']['seg35']:.4f} / {rows[subject]['v1']['seg4']:.4f} (seg35 / seg4), development subject, single observation"
        panels(image, truth, top1, top2, lookup, centre, title+"; through the brain centre", args.out_dir/f"slices_{short}.png")
        foreground = (truth > 0) | (top1 > 0)
        wrong = (top1 != truth) & foreground
        worst = []
        for axis in range(3):
            others_axes = tuple(a for a in range(3) if a != axis)
            n_wrong, n_foreground = wrong.sum(axis=others_axes).astype(np.float64), foreground.sum(axis=others_axes).astype(np.float64)
            rate = np.where(n_foreground > .25*n_foreground.max(), n_wrong/np.maximum(1, n_foreground), 0)
            worst.append(int(np.argmax(rate)))
        panels(image, truth, top1, top2, lookup, worst, title+"; planes with the highest error rate", args.out_dir/f"worst_{short}.png")
        x = np.nonzero((truth > 0).any(axis=(0, 1)))[0]
        z = np.nonzero((truth > 0).any(axis=(1, 2)))[0]
        crop = ((max(z[-1]-90, 0), z[-1]+3), (x[0]-3, x[0]+77))
        zoom(image, truth, top1, centre[1], crop, f"{short}: coronal crop at y = {centre[1]}, the upper part of the low-x hemisphere", args.out_dir/f"zoom_{short}.png")
        print(f"{subject}: done, peak RSS {common.peak_rss_gib():.2f} GiB", flush=True)
    histograms = {}
    edges = np.linspace(0, 1.8, 181)
    for subject in subjects:
        image = load_scalar(Path(eval_seg.DATA_ROOT)/subject/eval_seg.IMAGE_FILE).values
        truth, _ = eval_seg.read_truth(eval_seg.DATA_ROOT, subject, "seg35")
        density, _ = np.histogram(image[truth > 0]/rows[subject]["intensity"]["wm_median"], bins=edges, density=True)
        histograms[subject] = (.5*(edges[:-1]+edges[1:]), density)
    group_figure(rows, subjects, histograms, args.out_dir/"group.png")
    print("group figure done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
