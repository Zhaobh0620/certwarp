"""SEGP P3 and P4 (runs/20261105-segprobe/任务书.md): the pull-back and the fusion of experiments/segprobe/p3_fusion.py, the features of experiments/segprobe/p4_0440.py. Synthetic arrays only."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
import pytest

REPO = Path(__file__).resolve().parents[3]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"segprobe_{name}", REPO/"experiments/segprobe"/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


p3 = _load("p3_fusion")
p4 = _load("p4_0440")


def posterior(labels, n=36, on=.9):
    """float16 (n, *shape): `on` on the label, the rest spread evenly."""
    p = np.full((n,)+labels.shape, (1-on)/(n-1), dtype=np.float64)
    grid = np.ogrid[tuple(slice(0, s) for s in labels.shape)]
    p[(labels.astype(np.int64),)+tuple(grid)] = on
    return p.astype(np.float16)


def cube_labels(shape=(12, 14, 16), low=(4, 4, 4), high=(8, 10, 12), label=7):
    labels = np.zeros(shape, dtype=np.uint8)
    labels[low[0]:high[0], low[1]:high[1], low[2]:high[2]] = label
    return labels


# ---- P3 ------------------------------------------------------------------------------------------------------------------------------------------

def test_pull_back_follows_warped_of_p_equals_moving_of_p_plus_d():
    """A constant integer displacement: warped(p) = moving(p + d), so the structure of the moving image appears shifted by -d. The other sign would land it on +d."""
    moving = cube_labels()
    raw = posterior(moving)
    d = np.zeros((3,)+moving.shape)
    d[2] = 2.                                  # x: warped(z, y, x) = moving(z, y, x + 2)
    coordinates, _ = p3.pull_back_coordinates(d)
    channels = dict(p3.pull_back_channels(raw, coordinates))
    expected = np.zeros_like(moving)
    expected[4:8, 4:10, 2:10] = 7              # the moving cube is at x 4..11: the warped one at x 2..9
    assert np.array_equal(channels[7] > .5, expected == 7)


def test_pull_back_agrees_with_the_evaluators_label_warp_on_integer_displacements():
    moving = cube_labels()
    d = np.zeros((3,)+moving.shape)
    d[0], d[1], d[2] = 1., -2., 3.
    warped_labels = p3.warp_labels_l2r(moving, d)
    coordinates, _ = p3.pull_back_coordinates(d)
    channels = dict(p3.pull_back_channels(posterior(moving), coordinates))
    assert np.array_equal(channels[7] > .5, warped_labels == 7)


def test_pull_back_is_trilinear_and_keeps_the_channel_sum():
    moving = cube_labels()
    raw = posterior(moving)
    d = np.zeros((3,)+moving.shape)
    d[2] = .5
    coordinates, outside = p3.pull_back_coordinates(d)
    channels = dict(p3.pull_back_channels(raw, coordinates))
    total = raw.astype(np.float64).sum(axis=0)
    normalized = raw[7].astype(np.float64)/total
    assert channels[7][6, 6, 3] == pytest.approx(.5*(normalized[6, 6, 3]+normalized[6, 6, 4]))
    assert np.abs(sum(channels.values())-1).max() < 1e-12
    assert outside == int(np.prod(moving.shape[:2]))     # only the last x column looks beyond the grid (x + .5 > n - 1)


def test_fusion_is_the_argmax_of_the_mean_at_zero_displacement():
    rng = np.random.default_rng(0)
    a, b = rng.random((36, 5, 6, 7)), rng.random((36, 5, 6, 7))
    a, b = (a/a.sum(axis=0)).astype(np.float16), (b/b.sum(axis=0)).astype(np.float16)
    top_mix, top_warp, record = p3.fuse(a, b, np.zeros((3, 5, 6, 7)))
    na, nb = a.astype(np.float64)/a.astype(np.float64).sum(axis=0), b.astype(np.float64)/b.astype(np.float64).sum(axis=0)
    assert np.array_equal(top_mix, np.argmax(.5*(na+nb), axis=0))
    assert np.array_equal(top_warp, np.argmax(nb, axis=0))
    assert record["voxels_pulled_from_outside_grid"] == 0


def test_fusion_ties_go_to_the_smallest_label():
    a = np.zeros((36, 1, 1, 1), dtype=np.float16)
    a[9], a[4] = .5, .5
    top_mix, top_warp, _ = p3.fuse(a, a.copy(), np.zeros((3, 1, 1, 1)))
    assert top_mix.item() == 4 and top_warp.item() == 4


def test_reduction_of_error():
    assert p3.reduction(.95, .90) == pytest.approx(.5)
    assert p3.reduction(.90, .90) == pytest.approx(0.)
    assert p3.reduction(.89, .90) == pytest.approx(-.1)
    assert np.isnan(p3.reduction(1., 1.))


def test_error_flow_counts_what_was_corrected_and_broken():
    truth = np.array([[[1, 1, 2, 2, 0]]], dtype=np.uint8)
    fixed = np.array([[[1, 2, 2, 1, 0]]], dtype=np.uint8)        # wrong at 1, 3
    fused = np.array([[[1, 1, 1, 1, 0]]], dtype=np.uint8)        # corrects 1, breaks 2, keeps 3 wrong
    warp = np.array([[[1, 1, 1, 1, 0]]], dtype=np.uint8)
    flow = p3.error_flow(fixed, fused, warp, truth)
    assert (flow["wrong_fixed"], flow["wrong_fused"], flow["corrected"], flow["broken"], flow["net_voxels"]) == (2, 2, 1, 1, 0)
    assert flow["warped_right_at_fixed_errors"] == pytest.approx(.5)
    assert flow["warped_repeats_fixed_label_at_fixed_errors"] == pytest.approx(.5)


def test_direction_check_refuses_a_field_that_does_not_reproduce_the_metrics():
    fixed = cube_labels()
    moving = np.roll(fixed, 2, axis=2)                           # warped(p) = moving(p + d) with d_x = 2 brings it back
    d = np.zeros((3,)+fixed.shape)
    d[2] = 2.
    four = (fixed > 0).astype(np.uint8)
    four_moving = (moving > 0).astype(np.uint8)
    good = dict(seg35=1., seg4=1.)
    assert p3.check_direction(d, fixed, four, moving, four_moving, good)["max_abs_diff"] == 0.
    with pytest.raises(ValueError, match="does not reproduce"):
        p3.check_direction(-d, fixed, four, moving, four_moving, good)
    with pytest.raises(ValueError, match="does not reproduce"):
        p3.check_direction(d, fixed, four, moving, four_moving, dict(seg35=.9, seg4=1.))


# ---- P4 ------------------------------------------------------------------------------------------------------------------------------------------

def test_boundary_of_marks_both_sides():
    labels = np.zeros((5, 5, 5), dtype=np.uint8)
    labels[2, 2, 2] = 3
    marked = p4.boundary_of(labels)
    assert marked[2, 2, 2] and marked[1, 2, 2] and marked[2, 3, 2] and not marked[0, 0, 0] and marked.sum() == 7


def _geometry(pred, truth):
    distance = ndimage.distance_transform_edt(~p4.boundary_of(truth))
    return p4.error_geometry(pred, truth, distance)["record"]


def test_a_shifted_boundary_is_a_boundary_error_and_a_wrong_block_is_not():
    truth = cube_labels(shape=(40, 40, 40), low=(8, 8, 8), high=(32, 32, 32))
    shifted = np.roll(truth, 1, axis=2)                          # one layer on each of two faces
    near = _geometry(shifted, truth)
    assert near["share_within_1"] == pytest.approx(1.) and near["share_beyond_2"] == 0. and near["share_in_thick_parts"] == 0.
    block = truth.copy()
    block[14:26, 14:26, 14:26] = 9                               # a block deep inside the structure
    far = _geometry(block, truth)
    assert far["share_beyond_2"] > .9 and far["share_in_thick_parts"] == pytest.approx(1.) and far["n_components"] == 1


def test_crossing_finds_the_midpoint_and_the_slope():
    profile = {-5: 10., -4: 10., -3: 10., -2: 10., -1: 20., 1: 40., 2: 50., 3: 50., 4: 50., 5: 50.}
    position, slope = p4.crossing(profile, 30.)
    assert position == pytest.approx(0.) and slope == pytest.approx(20.)     # the layers -1 and +1 are adjacent voxels, one voxel apart
    position, slope = p4.crossing(profile, 45.)
    assert position == pytest.approx(1.) and slope == pytest.approx(10.)     # between the layers +1 (at .5) and +2 (at 1.5)
    position, _ = p4.crossing(profile, 15.)
    assert position == pytest.approx(-1.)
    assert np.isnan(p4.crossing(profile, 500.)[0])


def test_wm_profile_reads_the_layers_on_both_sides():
    image = np.zeros((30, 30, 30))
    mask = np.zeros((30, 30, 30), dtype=bool)
    mask[:, :, 15:] = True
    image[:, :, 15:] = 100.                                      # WM on the high side
    image[:, :, :15] = 50.
    profile = p4.wm_profile(image, mask, ~mask)
    assert profile[1] == 100. and profile[5] == 100. and profile[-1] == 50. and profile[-5] == 50.


def test_top_conf_ptrue_reads_the_same_normalized_posterior():
    rng = np.random.default_rng(1)
    p = rng.random((36, 20, 4, 5))
    raw = (p/p.sum(axis=0)).astype(np.float16)
    truth = rng.integers(0, 36, (20, 4, 5)).astype(np.uint8)
    top, second, conf, ptrue = p4.top_conf_ptrue(raw, truth, slab=7)
    normalized = raw.astype(np.float64)/raw.astype(np.float64).sum(axis=0)
    assert np.array_equal(top, np.argmax(normalized, axis=0))
    assert np.allclose(conf, normalized.max(axis=0), atol=1e-6)
    assert np.allclose(ptrue, np.take_along_axis(normalized, truth[None].astype(np.int64), axis=0)[0], atol=1e-6)
    # float16 storage has ties: the second label is read by its value (the second largest), and it is never the first
    assert np.array_equal(np.take_along_axis(normalized, second[None].astype(np.int64), axis=0)[0], np.sort(normalized, axis=0)[-2])
    assert not np.any(second == top)


def test_confidence_on_a_perfect_and_a_wrong_prediction():
    truth = np.array([[[1, 1, 2, 2, 0, 0]]], dtype=np.uint8)
    top = np.array([[[1, 1, 2, 3, 0, 0]]], dtype=np.uint8)
    second = np.array([[[2, 2, 3, 2, 1, 1]]], dtype=np.uint8)
    conf = np.array([[[.9, .9, .8, .95, .9, .9]]], dtype=np.float32)
    ptrue = np.array([[[.9, .9, .8, .04, .9, .9]]], dtype=np.float32)
    record = p4.confidence(top, second, conf, ptrue, truth)
    assert record["conf_on_wrong_mean"] == pytest.approx(.95) and record["wrong_with_conf_above_90"] == 1. and record["top2_hit_on_wrong"] == 1.
    assert record["accuracy_foreground"] == pytest.approx(3/4)


def test_group_stats_reads_the_target_against_the_other_19():
    rows = {f"s{i}": dict(a=dict(b=float(i))) for i in range(19)}
    rows[p4.TARGET] = dict(a=dict(b=-5.))
    stats = p4.group_stats(rows, ("a", "b"))
    assert stats["value"] == -5. and stats["others_median"] == 9. and stats["others_min"] == 0. and stats["others_max"] == 18. and stats["rank_of_20_from_low"] == 1 and stats["robust_z"] < 0


def test_error_overlap_of_two_arms():
    w1, w2 = np.array([True, True, False, False]), np.array([True, False, True, False])
    t1, t2 = np.array([5, 1, 0, 0]), np.array([5, 0, 2, 0])
    record = p4.error_overlap(w1, w2, t1, t2)
    assert record["jaccard"] == pytest.approx(1/3) and record["v1_errors_also_v2"] == pytest.approx(.5) and record["same_wrong_label_share_of_both"] == 1.


# ---- the follow-up checks ------------------------------------------------------------------------------------------------------------------------

label_offset = _load("p4_label_offset")
image_fit = _load("p4_image_fit")
leverage = _load("p4_leverage")
followup = _load("p4_followup")


def test_fit_affine_recovers_a_known_map_and_its_scale():
    rng = np.random.default_rng(3)
    source = rng.random((35, 3))*100
    linear = np.array([[1.02, .01, 0.], [0., .98, .02], [-.01, 0., 1.]])
    target = source@linear.T+np.array([1.5, -2., .5])
    weights = rng.random(35)+.1
    found, translation, residual, centre = label_offset.fit_affine(source, target, weights)
    assert np.allclose(found, linear, atol=1e-9) and np.abs(residual).max() < 1e-9
    assert np.allclose(translation, centre@linear.T+np.array([1.5, -2., .5]), atol=1e-9)


def test_centroids_of_labels():
    labels = np.zeros((4, 4, 4), dtype=np.uint8)
    labels[1, 2, 3] = 5
    labels[3, 2, 3] = 5
    means, counts = label_offset.centroids(labels, n=6)
    assert counts[5] == 2 and means[5].tolist() == [2., 2., 3.] and np.isnan(means[1]).all()


def test_the_subject_record_of_a_uniform_shift_is_that_shift():
    truth = np.zeros((60, 60, 60), dtype=np.uint8)
    for label, (a, b, c) in {1: (10, 10, 10), 2: (10, 30, 40), 3: (35, 12, 35), 4: (30, 35, 12), 5: (40, 40, 40)}.items():
        truth[a:a+10, b:b+10, c:c+10] = label
    top = np.roll(truth, -2, axis=0)                 # the truth lies 2 voxels above the prediction
    record = label_offset.subject_record(truth, top)
    assert np.allclose(record["mean_offset_zyx_voxels"], [2., 0., 0.]) and abs(record["affine"]["scale"]-1) < 1e-9 and record["affine"]["residual_rms_voxels"] < 1e-9
    assert record["affine"]["linear_minus_identity_fro"] < 1e-9


def _image_and_labels(shape=(60, 60, 60), seed=4):
    rng = np.random.default_rng(seed)
    coarse = rng.integers(1, 6, (6, 6, 6))
    labels = ndimage.zoom(coarse, 10, order=0).astype(np.uint8)
    labels = ndimage.median_filter(labels, size=3)
    image = (20.*labels+rng.normal(0, 1., shape))
    return image, labels


def test_a_displaced_label_map_shows_in_the_score_with_the_sign_of_the_shift():
    image, labels = _image_and_labels()
    mask = image > 0
    on_image = image_fit.best_shifts(image, image_fit.model_of(image, labels), mask)
    assert all(abs(on_image[box][f"best_{a}"]) < .3 for box in image_fit.BOXES for a in "zyx")
    displaced = np.roll(labels, 2, axis=0)            # the labels 2 voxels too high: shifting the model by -2 brings it back
    off_image = image_fit.best_shifts(image, image_fit.model_of(image, displaced), mask)
    assert all(abs(off_image[box]["best_z"]+2) < .3 for box in image_fit.BOXES)
    assert all(abs(off_image[box]["best_x"]) < .3 for box in image_fit.BOXES)
    assert all(off_image[box]["ncc_at_0"] < on_image[box]["ncc_at_0"] for box in image_fit.BOXES)


def test_refine_is_a_parabola_and_stops_at_the_ends():
    position = 1.3
    scores = [-(s-position)**2 for s in image_fit.SHIFTS]
    assert image_fit.refine(scores) == pytest.approx(position)
    assert image_fit.refine([5, 4, 3, 2, 1, 0, -1]) == image_fit.SHIFTS[0]
    assert image_fit.refine([-1, 0, 1, 2, 3, 4, 5]) == image_fit.SHIFTS[-1]


def test_counted_right_leaves_the_labels_untouched_and_raises_the_dice():
    lookup = np.array([0, 3, 1, 4, 4, 3, 2, 2, 2, 2, 2, 4, 4, 3, 2, 2, 2, 3, 4, 4, 3, 1, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4])
    truth = np.zeros((8, 8, 8), dtype=np.uint8)
    truth[2:6, 2:6, 2:6] = 2
    truth4 = lookup[truth].astype(np.uint8)
    top = truth.copy()
    top[2:6, 2:6, 6:8] = 2                            # a block the truth calls background
    keep35, keep4 = truth.copy(), truth4.copy()
    mask = (top != truth)
    base = leverage.eval_seg.summarize_subject(top, truth, truth4, lookup)
    counted = leverage.counted_right(top, truth, truth4, lookup, mask)
    assert counted["seg35"] == pytest.approx(1.) and base["seg35"] < 1.
    assert np.array_equal(truth, keep35) and np.array_equal(truth4, keep4)


def test_votes_and_the_arbiter_read_the_sides_the_right_way_round():
    atropos = np.array([[[followup.GM, followup.GM, followup.WM, followup.CSF, 0]]], dtype=np.uint8)
    pred4 = np.array([[[1, 1, 1, 1, 1]]], dtype=np.uint8)
    truth4 = np.array([[[3, 3, 3, 3, 3]]], dtype=np.uint8)
    record = followup.arbiter(atropos, truth4, pred4)["pred_cortex_truth_wm"]
    assert (record["n"], record["for_a"], record["for_b"]) == (4, 2, 1) and record["share_for_a"] == pytest.approx(2/3)


def test_spectral_shares_fall_for_a_smooth_volume():
    rng = np.random.default_rng(5)
    noise = rng.normal(size=(40, 40, 40))+5
    smooth = ndimage.gaussian_filter(noise, 2.)+5
    sharp, blurred = followup.spectral_shares(noise), followup.spectral_shares(smooth)
    # white noise keeps most of its power above .2 cycles per voxel; a Gaussian of sigma 2 removes nearly all of it. What stays above .4 is leakage from the faces of the box (no window, the same on every subject: the measure is for comparing subjects)
    assert blurred["above_0.2"] < .2 < .8 < sharp["above_0.2"] and blurred["above_0.4"] < .05 < .1 < sharp["above_0.4"]
    assert sharp["above_0.2"] >= sharp["above_0.3"] >= sharp["above_0.4"]
