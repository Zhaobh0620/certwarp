"""SEGP P4: why subject 0440 is the low outlier of the 20 development subjects (runs/20261105-segprobe/任务书.md). Read only; a description, no data change, no exclusion.

The same features for all 20 subjects, one subject at a time, so that 0440 is read against the other 19 (median, range, a robust z). For the predictions of v1 and v2 (no member-level file exists):
  errors      the wrong voxels on the union of the foregrounds: distance to the truth boundary (boundary layer or block), thick parts (surviving an erosion by a 3x3x3 cube), components, confusion matrix, volumes, side, coarse location
  confidence  the top probability on wrong and right voxels, the probability of the true label, top-2 hits, the calibration on the foreground
  v1 and v2   how far their errors coincide
For the image and the truth, independently of any network:
  intensity   quantiles, tissue medians, contrast and CNR of white matter and cortex, AUC of the two tissues under the truth and under the prediction, noise, WM inhomogeneity (also on aligned_orig), dark WM
  anatomy     brain volume, ventricles, cortex and WM volumes, a thickness proxy; the native volume (seg4.nii.gz, scanner space) and the scale of the affine alignment that it implies
  alignment   brain centroid, extents, the tilt of the hemispheres, NCC of the image and Dice of seg4 with the mean of the other 19
  the labels  intensity profile across the WM surface of the truth and of the prediction (the crossing of the WM/cortex midpoint, the slope), roughness, the Dice against the prediction with the truth shifted by one voxel, the intensity at the confused voxels

Only the 20 development subjects (segprobe_common.admit_dev_subjects). The truth is read for this description only. Outputs in --out-dir (never overwritten): features.json, confusion.npz, maps_0440.npz, summary.md.

    python -B experiments/segprobe/p4_0440.py --out-dir DIR
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segprobe_common as common  # noqa: E402
from segprobe_common import eval_seg  # noqa: E402
from jd_curl.three_d.nifti import load_scalar  # noqa: E402

FORMAT = "bayesreg.segprobe.p4.v1"
TARGET = "OASIS_OAS1_0440_MR1"
NAMES = {int(k): v for k, v in json.loads((common.REPO/"configs/bayesreg/structure_groups_v1.json").read_text())["label_table"]["names"].items()}
CUBE = np.ones((3, 3, 3), dtype=bool)
BLOCK = 16
ECE_BINS = 15
VENTRICLES = (3, 22, 4, 23)
ALL_CSF = (3, 22, 4, 23, 11, 12)
SEG4_NAMES = {1: "cortex", 2: "subcortical", 3: "white_matter", 4: "csf"}


def boundary_of(labels):
    """True on voxels with a 6-neighbour of another label."""
    out = np.zeros(labels.shape, dtype=bool)
    for axis in range(3):
        differs = np.diff(labels, axis=axis) != 0
        low = [slice(None)]*3
        high = [slice(None)]*3
        low[axis], high[axis] = slice(None, -1), slice(1, None)
        out[tuple(low)] |= differs
        out[tuple(high)] |= differs
    return out


def error_geometry(pred, truth, truth_boundary_distance):
    foreground = (pred > 0) | (truth > 0)
    wrong = (pred != truth) & foreground
    n_wrong = int(wrong.sum())
    distance = truth_boundary_distance[wrong]
    thick = ndimage.binary_opening(wrong, structure=CUBE)
    components, n_components = ndimage.label(wrong, structure=CUBE)
    sizes = np.bincount(components.ravel())[1:]
    sizes_sorted = np.sort(sizes)[::-1]
    side = wrong.shape[2]//2
    return dict(wrong=wrong, record=dict(
        n_foreground=int(foreground.sum()), n_wrong=n_wrong, error_rate=n_wrong/int(foreground.sum()),
        share_on_boundary=float((distance == 0).mean()), share_within_1=float((distance <= 1).mean()), share_within_2=float((distance <= 2).mean()), share_beyond_2=float((distance > 2).mean()),
        mean_distance=float(distance.mean()), share_in_thick_parts=float(thick.sum()/n_wrong), n_components=int(n_components), largest_component_share=float(sizes_sorted[0]/n_wrong),
        top10_components_share=float(sizes_sorted[:10].sum()/n_wrong), share_low_x=float(wrong[:, :, :side].sum()/n_wrong)))


def location(wrong, foreground):
    """Error rate along each axis (16 bins), by octant, and how concentrated the errors are among the 16-voxel blocks."""
    out = {}
    for axis, name in enumerate("zyx"):
        others = tuple(a for a in range(3) if a != axis)
        n_wrong, n_foreground = wrong.sum(axis=others), foreground.sum(axis=others)
        edges = np.linspace(0, wrong.shape[axis], 17).astype(int)
        out[f"rate_along_{name}"] = [float(n_wrong[a:b].sum()/max(1, n_foreground[a:b].sum())) for a, b in zip(edges[:-1], edges[1:])]
    centre = [int(np.round(np.nonzero(foreground.any(axis=tuple(a for a in range(3) if a != axis)))[0].mean())) for axis in range(3)]
    octants = {}
    for a, z in (("inferior", slice(None, centre[0])), ("superior", slice(centre[0], None))):
        for b, y in (("posterior", slice(None, centre[1])), ("anterior", slice(centre[1], None))):
            for c, x in (("lowx", slice(None, centre[2])), ("highx", slice(centre[2], None))):
                octants[f"{a}_{b}_{c}"] = float(wrong[z, y, x].sum()/max(1, foreground[z, y, x].sum()))
    out["octant_rate"] = octants
    starts = [np.arange(0, n, BLOCK) for n in wrong.shape]
    counts = wrong.astype(np.int64)
    for axis, s in enumerate(starts):
        counts = np.add.reduceat(counts, s, axis=axis)
    fg_counts = foreground.astype(np.int64)
    for axis, s in enumerate(starts):
        fg_counts = np.add.reduceat(fg_counts, s, axis=axis)
    values = np.sort(counts[fg_counts > 0].astype(np.float64))[::-1]
    out["share_in_densest_10pct_of_blocks"] = float(values[:max(1, len(values)//10)].sum()/max(1., values.sum()))
    return out


def top_conf_ptrue(raw, truth, *, slab=common.SLAB):
    """float32 maps: the top probability, the probability of the true label, the second label (uint8). The first label comes from the same normalized arrays as argmax_of_mean."""
    shape = raw.shape[1:]
    top, second, conf, ptrue = np.empty(shape, np.uint8), np.empty(shape, np.uint8), np.empty(shape, np.float32), np.empty(shape, np.float32)
    grid = np.ogrid[:shape[1], :shape[2]]
    for start in range(0, shape[0], slab):
        stop = min(start+slab, shape[0])
        part = common.normalized_slab(raw, start, stop)
        best = np.argmax(part, axis=0)
        top[start:stop] = best
        z = np.arange(stop-start)[:, None, None]
        conf[start:stop] = part[best, z, grid[0], grid[1]]
        ptrue[start:stop] = part[truth[start:stop].astype(np.int64), z, grid[0], grid[1]]
        part[best, z, grid[0], grid[1]] = -1.
        second[start:stop] = np.argmax(part, axis=0)
    return top, second, conf, ptrue


def confidence(top, second, conf, ptrue, truth):
    foreground = (top > 0) | (truth > 0)
    right = (top == truth) & foreground
    wrong = (top != truth) & foreground
    edges = np.linspace(0, 1, ECE_BINS+1)
    binned = np.clip(np.digitize(conf[foreground], edges)-1, 0, ECE_BINS-1)
    n_bin = np.bincount(binned, minlength=ECE_BINS).astype(np.float64)
    hits = np.bincount(binned, weights=right[foreground].astype(np.float64), minlength=ECE_BINS)
    confs = np.bincount(binned, weights=conf[foreground].astype(np.float64), minlength=ECE_BINS)
    ece = float(np.abs(hits-confs).sum()/n_bin.sum())
    return dict(conf_on_wrong_mean=float(conf[wrong].mean()), conf_on_right_mean=float(conf[right].mean()), conf_on_wrong_q50=float(np.median(conf[wrong])), wrong_with_conf_above_90=float((conf[wrong] > .9).mean()),
                wrong_with_conf_above_70=float((conf[wrong] > .7).mean()), ptrue_on_wrong_mean=float(ptrue[wrong].mean()), top2_hit_on_wrong=float((second[wrong] == truth[wrong]).mean()),
                accuracy_foreground=float(right.sum()/foreground.sum()), confidence_foreground=float(conf[foreground].mean()), ece_foreground=ece)


def volumes_ratio(top, truth):
    predicted, true = np.bincount(top.ravel(), minlength=36).astype(np.float64), np.bincount(truth.ravel(), minlength=36).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(true > 0, predicted/true, np.nan)
    return ratio.tolist(), true.tolist()


def robust_sd(values):
    return float(1.4826*np.median(np.abs(values-np.median(values))))


def auc_from_hist(high, low, edges):
    """P(high > low) + P(=)/2 from the histograms of two samples."""
    a, _ = np.histogram(high, bins=edges)
    b, _ = np.histogram(low, bins=edges)
    a, b = a/a.sum(), b/b.sum()
    return float((a*(np.cumsum(b)-b)).sum()+.5*(a*b).sum())


def intensity(image, truth, seg4, top, orig):
    brain = truth > 0
    wm, cortex = np.isin(truth, (1, 20)), np.isin(truth, (2, 21))
    wm_core = ndimage.binary_erosion(wm, structure=ndimage.generate_binary_structure(3, 1), iterations=2)
    wm_median, gm_median = float(np.median(image[wm_core])), float(np.median(image[cortex]))
    wm_sd, gm_sd = robust_sd(image[wm_core]), robust_sd(image[cortex])
    midpoint = .5*(wm_median+gm_median)
    edges = np.linspace(0, float(np.percentile(image[brain], 99.9))+1, 513)
    pred_wm, pred_cortex = np.isin(top, (1, 20)), np.isin(top, (2, 21))
    wm_dark = wm_core & (image < midpoint)
    near_ventricle = ndimage.distance_transform_edt(~np.isin(truth, VENTRICLES)) <= 5
    components, n_components = ndimage.label(wm_dark, structure=CUBE)
    sizes = np.bincount(components.ravel())[1:] if n_components else np.array([0])
    high_pass = image-ndimage.gaussian_filter(image, 1.)
    starts = [np.arange(0, n, 32) for n in image.shape]

    def block_cv(values):
        medians = []
        for a, b in zip(starts[0], list(starts[0][1:])+[image.shape[0]]):
            for c, d in zip(starts[1], list(starts[1][1:])+[image.shape[1]]):
                for e, f in zip(starts[2], list(starts[2][1:])+[image.shape[2]]):
                    mask = wm_core[a:b, c:d, e:f]
                    if mask.sum() >= 200:
                        medians.append(float(np.median(values[a:b, c:d, e:f][mask])))
        medians = np.array(medians)
        return float(medians.std()/medians.mean()), len(medians)

    cv_norm, n_blocks = block_cv(image)
    cv_orig, _ = block_cv(orig)
    quantiles = {f"q{q}": float(np.percentile(image[brain], q)) for q in (1, 5, 25, 50, 75, 95, 99)}
    return dict(**quantiles, wm_median=wm_median, gm_median=gm_median, wm_sd=wm_sd, gm_sd=gm_sd, csf_median=float(np.median(image[np.isin(truth, ALL_CSF)])), subcortical_median=float(np.median(image[seg4 == 2])),
                contrast_wm_gm=(wm_median-gm_median)/wm_median, cnr_wm_gm=(wm_median-gm_median)/np.hypot(wm_sd, gm_sd), auc_truth_wm_vs_cortex=auc_from_hist(image[wm], image[cortex], edges), auc_pred_wm_vs_cortex=auc_from_hist(image[pred_wm], image[pred_cortex], edges),
                bright_cortex_share=float((image[cortex] > midpoint).mean()), dark_wm_share=float((image[wm] < midpoint).mean()), dark_wm_core_share=float(wm_dark.sum()/wm_core.sum()), dark_wm_core_near_ventricle=float((wm_dark & near_ventricle).sum()/max(1, wm_dark.sum())),
                dark_wm_core_largest_component=int(sizes.max()), noise_wm_high_pass_sd_over_wm=robust_sd(high_pass[wm_core])/wm_median, wm_block_cv_norm=cv_norm, wm_block_cv_orig=cv_orig, wm_blocks=n_blocks,
                midpoint=midpoint, orig_wm_median=float(np.median(orig[wm_core]))), dict(wm_median=wm_median, gm_median=gm_median, midpoint=midpoint, edges=edges)


def anatomy(truth, seg4, image, native_seg4_path):
    brain = truth > 0
    n_brain = int(brain.sum())
    wm, cortex = np.isin(truth, (1, 20)), np.isin(truth, (2, 21))
    contact = cortex & ndimage.binary_dilation(wm, structure=ndimage.generate_binary_structure(3, 1))
    native = nib.load(str(native_seg4_path))                 # the scanner space: only the voxel count and the voxel size are used, so no reorientation
    spacing = [float(v) for v in native.header.get_zooms()[:3]]
    native_voxel_mm3 = float(np.prod(spacing))
    n_native = int((np.asanyarray(native.dataobj) > 0).sum())
    return dict(n_brain=n_brain, ventricles=int(np.isin(truth, ALL_CSF).sum()), lateral_ventricles=int(np.isin(truth, (3, 22)).sum()), ventricle_share=float(np.isin(truth, ALL_CSF).sum()/n_brain), n_cortex=int(cortex.sum()), n_wm=int(wm.sum()),
                cortex_share=float(cortex.sum()/n_brain), wm_share=float(wm.sum()/n_brain), cortex_over_wm=float(cortex.sum()/wm.sum()), thickness_proxy=float(cortex.sum()/contact.sum()),
                image_positive_truth_background=float(((image > 0) & ~brain).sum()/n_brain), truth_foreground_image_zero=float((brain & (image <= 0)).sum()/n_brain),
                native_volume_mm3=n_native*native_voxel_mm3, native_spacing=spacing, native_shape=[int(v) for v in native.shape[:3]], aligned_over_native_volume=n_brain/(n_native*native_voxel_mm3), linear_scale_of_alignment=float((n_brain/(n_native*native_voxel_mm3))**(1/3)))


def alignment(truth):
    brain = truth > 0
    indices = np.nonzero(brain)
    centroid = [float(i.mean()) for i in indices]
    extents = {f"extent_{n}": float(np.percentile(i, 99.5)-np.percentile(i, .5)) for n, i in zip("zyx", indices)}
    left, right = np.nonzero(np.isin(truth, (1, 2))), np.nonzero(np.isin(truth, (20, 21)))
    between = np.array([r.mean()-l.mean() for l, r in zip(left, right)])
    tilt = dict(hemisphere_vector_zyx=between.tolist(), roll_deg=float(np.degrees(np.arctan2(between[0], between[2]))), yaw_deg=float(np.degrees(np.arctan2(between[1], between[2]))))
    return dict(centroid_zyx=centroid, **extents, **tilt)


def wm_profile(image, mask_wm, outside, *, reach=5):
    """Mean and sd of the intensity against the signed distance to the surface of mask_wm: d = +1 the outermost WM layer, -1 the first layer of `outside`. Bins of width 1 (rounded distance)."""
    inside_distance = ndimage.distance_transform_edt(mask_wm)
    outside_distance = ndimage.distance_transform_edt(~mask_wm)
    profile = {}
    for k in range(1, reach+1):
        a = mask_wm & (np.round(inside_distance) == k)
        b = outside & (np.round(outside_distance) == k)
        profile[k] = float(image[a].mean()) if a.any() else float("nan")
        profile[-k] = float(image[b].mean()) if b.any() else float("nan")
    return profile


def crossing(profile, midpoint):
    """Where the profile crosses the midpoint and the slope there, per voxel. The layers +1 and -1 are adjacent voxels on the two sides of the surface, so a layer d stands at the position d - sign(d)/2, in voxels from the surface
    (positive: inside the WM of this surface). nan if it does not cross between the layers -5 and +5."""
    ds = sorted(profile)
    at = {d: d-.5*np.sign(d) for d in ds}
    for low, high in zip(ds[:-1], ds[1:]):
        a, b = profile[low], profile[high]
        if np.isfinite(a) and np.isfinite(b) and a <= midpoint <= b and b > a:
            return float(at[low]+(midpoint-a)*(at[high]-at[low])/(b-a)), float((b-a)/(at[high]-at[low]))
    return float("nan"), float("nan")


def label_quality(image, truth, seg4, top, reference, lookup):
    wm_t, cortex_t = np.isin(truth, (1, 20)), np.isin(truth, (2, 21))
    wm_p, cortex_p = np.isin(top, (1, 20)), np.isin(top, (2, 21))
    wm_mid, gm_mid, midpoint = reference["wm_median"], reference["gm_median"], reference["midpoint"]
    p_t, p_p = wm_profile(image, wm_t, cortex_t), wm_profile(image, wm_p, cortex_p)
    cross_t, slope_t = crossing(p_t, midpoint)
    cross_p, slope_p = crossing(p_p, midpoint)
    span = wm_mid-gm_mid
    isolated = {}
    for name, labels in (("truth", truth), ("pred", top)):
        same = np.zeros(labels.shape, dtype=np.int8)
        for axis in range(3):
            equal = np.diff(labels, axis=axis) == 0
            low, high = [slice(None)]*3, [slice(None)]*3
            low[axis], high[axis] = slice(None, -1), slice(1, None)
            same[tuple(low)] += equal
            same[tuple(high)] += equal
        foreground = labels > 0
        isolated[name] = dict(isolated_share=float(((same == 0) & foreground).sum()/foreground.sum()), at_most_1_neighbour_share=float(((same <= 1) & foreground).sum()/foreground.sum()),
                              surface_share=float(((same < 6) & foreground).sum()/foreground.sum()))
    cc_t, cc_p = ndimage.label(cortex_t, structure=CUBE)[1], ndimage.label(cortex_p, structure=CUBE)[1]
    pred4 = lookup[top]
    shifts = {}
    for axis, name in enumerate("zyx"):
        for step in (-1, 1):
            shifted = np.roll(seg4, step, axis=axis)
            shifts[f"{name}{step:+d}"] = eval_seg.nanmean(eval_seg.label_dice(pred4, shifted, eval_seg.N_SEG4))
    shifts["none"] = eval_seg.nanmean(eval_seg.label_dice(pred4, seg4, eval_seg.N_SEG4))

    def normalized_intensity(mask):
        return dict(n=int(mask.sum()), mean=float(((image[mask]-gm_mid)/span).mean()) if mask.any() else float("nan"), median=float(np.median((image[mask]-gm_mid)/span)) if mask.any() else float("nan"))

    confusions = dict(truth_cortex_pred_wm=normalized_intensity(cortex_t & wm_p), truth_wm_pred_cortex=normalized_intensity(wm_t & cortex_p), truth_cortex_pred_background=normalized_intensity(cortex_t & (top == 0)),
                      truth_background_pred_cortex=normalized_intensity((truth == 0) & cortex_p), agree_wm=normalized_intensity(wm_t & wm_p), agree_cortex=normalized_intensity(cortex_t & cortex_p))
    return dict(profile_truth={str(k): v for k, v in sorted(p_t.items())}, profile_pred={str(k): v for k, v in sorted(p_p.items())}, truth_crossing=cross_t, truth_slope_over_span=slope_t/span, pred_crossing=cross_p, pred_slope_over_span=slope_p/span,
                roughness=isolated, cortex_components_truth=int(cc_t), cortex_components_pred=int(cc_p), seg4_macro_with_truth_shifted=shifts, best_shift=max(shifts, key=shifts.get), confusions_normalized_intensity=confusions)


def error_overlap(wrong1, wrong2, top1, top2):
    union, both = wrong1 | wrong2, wrong1 & wrong2
    return dict(jaccard=float(both.sum()/union.sum()), v1_errors_also_v2=float(both.sum()/wrong1.sum()), v2_errors_also_v1=float(both.sum()/wrong2.sum()), same_wrong_label_share_of_both=float((top1[both] == top2[both]).sum()/both.sum()),
                n_only_v1=int((wrong1 & ~wrong2).sum()), n_only_v2=int((wrong2 & ~wrong1).sum()), n_both=int(both.sum()))


def ncc(a, b, mask):
    x, y = a[mask].astype(np.float64), b[mask].astype(np.float64)
    x, y = x-x.mean(), y-y.mean()
    return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()))


def group_stats(rows, path, target=TARGET):
    """Reads one number per subject along `path`: 0440's value, the median, min, max of the other 19, a robust z, its rank among the 20."""
    def read(row):
        node = row
        for key in path:
            node = node[key]
        return node
    target_value = read(rows[target])
    others = np.array([read(r) for s, r in rows.items() if s != target], dtype=np.float64)
    median, mad = float(np.median(others)), float(1.4826*np.median(np.abs(others-np.median(others))))
    rank = int((others < target_value).sum())+1
    return dict(value=target_value, others_median=median, others_min=float(others.min()), others_max=float(others.max()), robust_z=(target_value-median)/mad if mad > 0 else float("nan"), rank_of_20_from_low=rank)


def describe(rows):
    """The stats over every scalar feature found under each subject's row."""
    table = {}

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path+(key,))
        elif isinstance(node, (int, float)) and not isinstance(node, bool) and np.isfinite(node):
            table[".".join(path)] = group_stats(rows, path)

    walk(rows[TARGET], ())
    return table


def run(*, out_dir, argv=None, log=print, data_root=eval_seg.DATA_ROOT):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists and is not empty; results are never overwritten")
    subjects, admission = common.admit_dev_subjects(data_root=data_root)
    if TARGET not in subjects:
        raise eval_seg.SubjectSealed(f"{TARGET} is not an admitted development subject")
    lookup = eval_seg.load_seg4_lookup()
    started = time.perf_counter()
    # the first pass: the mean image and the seg4 counts of the 20, for the leave-one-out comparisons
    image_sum, counts4 = None, None
    for subject in subjects:
        image = load_scalar(Path(data_root)/subject/eval_seg.IMAGE_FILE).values
        seg4, _ = eval_seg.read_truth(data_root, subject, "seg4")
        scale = float(np.median(image[image > 0]))
        image_sum = image/scale if image_sum is None else image_sum+image/scale
        if counts4 is None:
            counts4 = np.zeros((eval_seg.N_SEG4,)+seg4.shape, dtype=np.int16)
        for k in range(eval_seg.N_SEG4):
            counts4[k] += seg4 == k
    log(f"first pass done, {time.perf_counter()-started:.0f} s", flush=True)
    rows, confusion, error_sum = {}, dict(v1=[], v2=[]), None
    maps_target = {}
    for subject in subjects:
        directory = Path(data_root)/subject
        image = load_scalar(directory/eval_seg.IMAGE_FILE).values
        orig = load_scalar(directory/"aligned_orig.nii.gz").values
        truth, sha35 = eval_seg.read_truth(data_root, subject, "seg35")
        seg4, sha4 = eval_seg.read_truth(data_root, subject, "seg4")
        distance = ndimage.distance_transform_edt(~boundary_of(truth))
        foreground_mask = truth > 0
        row = dict(truth_sha256=dict(seg35=sha35, seg4=sha4))
        tops, wrongs = {}, {}
        for arm, root, segmenter in (("v1", common.V1_ROOT, common.V1_ID), ("v2", common.V2_ROOT, common.V2_ID)):
            raw, record = common.load_raw(root, subject, expected_segmenter=segmenter)
            top, second, conf, ptrue = top_conf_ptrue(raw, truth)
            del raw
            geometry = error_geometry(top, truth, distance)
            fg = (top > 0) | (truth > 0)
            summary = eval_seg.summarize_subject(top, truth, seg4, lookup)
            ratio, true_volume = volumes_ratio(top, truth)
            row[arm] = dict(seg35=summary["seg35"], seg4=summary["seg4"], groups=summary["groups"], errors=geometry["record"], location=location(geometry["wrong"], fg), confidence=confidence(top, second, conf, ptrue, truth),
                            per_label=summary["per_label"], seg4_per_label=summary["seg4_per_label"], volume_ratio=ratio, true_volume=true_volume)
            confusion[arm].append(np.bincount(truth.astype(np.int64).ravel()*36+top.astype(np.int64).ravel(), minlength=36*36).reshape(36, 36))
            tops[arm], wrongs[arm] = top, geometry["wrong"]
            if subject == TARGET:
                maps_target[f"conf_{arm}"] = conf.astype(np.float16)
            del second, conf, ptrue
        row["v1_v2_errors"] = error_overlap(wrongs["v1"], wrongs["v2"], tops["v1"], tops["v2"])
        error_sum = wrongs["v1"].astype(np.int16) if error_sum is None else error_sum+wrongs["v1"]
        row["intensity"], reference = intensity(image, truth, seg4, tops["v1"], orig)
        del orig
        row["anatomy"] = anatomy(truth, seg4, image, directory/"seg4.nii.gz")
        row["alignment"] = alignment(truth)
        row["labels_v1"] = label_quality(image, truth, seg4, tops["v1"], reference, lookup)
        scale = float(np.median(image[image > 0]))
        loo_image = (image_sum-image/scale)/(len(subjects)-1)
        loo_counts = counts4.copy()
        for k in range(eval_seg.N_SEG4):
            loo_counts[k] -= seg4 == k
        atlas = np.argmax(loo_counts, axis=0)
        row["leave_one_out"] = dict(ncc_image=ncc(image/scale, loo_image, foreground_mask), seg4_dice_with_majority_of_others=eval_seg.nanmean(eval_seg.label_dice(atlas, seg4, eval_seg.N_SEG4)),
                                    seg4_dice_per_class={SEG4_NAMES[k]: float(v) for k, v in zip(range(1, 5), eval_seg.label_dice(atlas, seg4, eval_seg.N_SEG4))})
        del loo_image, loo_counts, atlas
        rows[subject] = row
        if subject == TARGET:
            maps_target.update(top_v1=tops["v1"], top_v2=tops["v2"], distance_to_truth_boundary=distance.astype(np.float16))
        log(f"{subject}: v1 {row['v1']['seg35']:.4f} / {row['v1']['seg4']:.4f}, wrong {row['v1']['errors']['n_wrong']}, beyond 2 voxels {row['v1']['errors']['share_beyond_2']:.3f}, peak RSS {common.peak_rss_gib():.2f} GiB, {time.perf_counter()-started:.0f} s", flush=True)
        del tops, wrongs, distance, truth, seg4, image
    out_dir.mkdir(parents=True, exist_ok=True)
    scalar_rows = {s: {k: v for k, v in r.items() if k not in ("truth_sha256",)} for s, r in rows.items()}
    stats = describe({s: {k: v for k, v in r.items() if k in ("intensity", "anatomy", "alignment", "labels_v1", "leave_one_out", "v1_v2_errors")} | {a: {k: v for k, v in r[a].items() if k in ("errors", "confidence", "seg35", "seg4", "groups")} for a in ("v1", "v2")} for s, r in scalar_rows.items()})
    document = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=argv, admission=admission, subjects=subjects, target=TARGET, stats=stats, rows=rows, seconds=time.perf_counter()-started, peak_rss_gib=common.peak_rss_gib(),
                    note="开发集 20 人的描述，单次观测；只读，不改数据，不提议剔除。robust_z 用其余 19 人的中位数与 MAD。特征很多，单个 z 的极端值要带多重比较的眼光看。")
    common.write_json(out_dir/"features.json", document)
    np.savez_compressed(out_dir/"confusion.npz", subjects=np.array(subjects), v1=np.stack(confusion["v1"]).astype(np.int64), v2=np.stack(confusion["v2"]).astype(np.int64))
    np.savez_compressed(out_dir/"maps_0440.npz", error_count_v1_over_20=error_sum, mean_image_over_20=(image_sum/len(subjects)).astype(np.float32), **maps_target)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=common.OUT_ROOT/"p4")
    args = parser.parse_args(argv)
    document = run(out_dir=args.out_dir, argv=sys.argv if argv is None else ["p4_0440.py", *argv])
    print(f"done, {document['seconds']:.0f} s, peak RSS {document['peak_rss_gib']:.2f} GiB, {len(document['stats'])} features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
