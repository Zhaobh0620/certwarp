"""bayesreg.eval.metrics: bitwise identity with scripts/evaluate_l2r_style.py and the archived smoothness.py, and
known values on synthetic labels and fields."""
import numpy as np
import pytest

from jd_curl.three_d.grid import make_grid_3d

from bayesreg.eval import metrics as M
from bayesreg.eval.fields import u_to_d

from . import _synthetic as syn

PAIR = f"{syn.DEV[0]}__{syn.DEV[1]}"


def same(a, b):
    """Bitwise equality of floats / float lists (NaN == NaN)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def pinned_l2r_script(tmp_path, monkeypatch):
    """C23.9-13 (11), revision 6e: the L2R script with its allowlist pinned (C23.9 总则, monkeypatch) to a synthetic one holding syn.DEV, so that
    the synthetic ids written into the run manifests go through C23.4-2 the normal way (the script has no synthetic exception)."""
    from bayesreg.seg.unet import synthetic
    script = syn.l2r_script()
    pinned = synthetic.write_json(tmp_path/"pinned_allowlist.json", synthetic.allowlist_document([("dev", list(syn.DEV), False)]))
    monkeypatch.setattr(script, "ALLOWLIST", pinned)
    return script


def truth_of(arrays):
    return dict(fixed_seg35=arrays["fixed_seg35"], moving_seg35=arrays["moving_seg35"], fixed_seg4=arrays["fixed_seg4"],
                moving_seg4=arrays["moving_seg4"], brain=arrays["fixed_seg4"] > 0)


@pytest.mark.parametrize("make_d", [syn.smooth_d, syn.folding_d])
def test_bitwise_equal_to_evaluate_l2r_style(tmp_path, monkeypatch, make_d):
    script = pinned_l2r_script(tmp_path, monkeypatch)
    arrays = syn.pair_arrays(seed=3)
    u = syn.d_to_u(make_d())
    run = syn.write_configc_run(tmp_path/"run", PAIR, u, arrays=arrays, subjects=syn.DEV[:2])        # C23.9-13 (11): the pair's synthetic ids in the manifest
    reference = script.evaluate_run(run)
    scalars, per_label, official = M.evaluate_field(u_to_d(u, (1., 1., 1.)), truth_of(arrays), M.OASIS, (1., 1., 1.))
    assert same(reference["dice"], scalars["dice_official"]) and same(reference["dice_labels"], per_label["dice_official_labels"])
    assert same(reference["initial_dice"], scalars["initial_dice_official"])
    assert same(reference["initial_dice_labels"], per_label["initial_dice_official_labels"])
    assert same(reference["hd95"], scalars["hd95"]) and same(reference["sdlogj"], scalars["sdlogj_official"])
    assert same(reference["num_foldings"], scalars["num_foldings_official"]) and scalars["num_foldings_official"] == 0
    assert same(reference["central_det_nonpositive_fraction"], scalars["central_det_nonpositive_fraction"])
    assert (scalars["central_det_nonpositive"] > 0) == (make_d is syn.folding_d)
    assert np.isnan(per_label["dice_official_labels"][6]) and np.isnan(per_label["dice_official_labels"][32])
    for key in ("dice", "hd95", "sdlogj", "dice_labels", "initial_dice_labels"):
        assert same(official[key], reference[key])


def test_aggregate_bitwise_equal(tmp_path, monkeypatch):
    script = pinned_l2r_script(tmp_path, monkeypatch)
    refs, ours = [], []
    for k, make_d in enumerate((syn.smooth_d, syn.folding_d, lambda: syn.smooth_d(amplitude=(.3, .4, -.5), freq=2))):
        arrays = syn.pair_arrays(seed=10+k)
        u = syn.d_to_u(make_d())
        refs.append(script.evaluate_run(syn.write_configc_run(tmp_path/f"run{k}", PAIR, u, arrays=arrays, subjects=syn.DEV[:2])))     # C23.9-13 (11)
        ours.append(M.evaluate_field(u_to_d(u, (1., 1., 1.)), truth_of(arrays), M.OASIS, (1., 1., 1.))[2])
    a, b = script.aggregate(refs), M.aggregate(ours)
    assert a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)


def test_jd_curl_dice_labels_convention_and_nan_rule():
    """seg35/seg4 (dice_labels): absent from both -> skipped; missing in moving -> Dice 0; official: NaN for both."""
    arrays = syn.pair_arrays(seed=5)
    d = np.zeros((3,)+syn.SHAPE)
    scalars, per_label, _ = M.evaluate_field(d, truth_of(arrays), M.OASIS, (1., 1., 1.))
    seg35 = per_label["seg35_per_label"]
    assert seg35["33"] is None and seg35["7"] == 0.
    defined = [v for v in seg35.values() if v is not None]
    assert scalars["seg35"] == pytest.approx(np.mean(defined), abs=0, rel=1e-15)
    assert set(per_label["seg4_per_label"]) == {"1", "2", "3", "4"}
    assert np.isnan(per_label["dice_official_labels"][6]) and np.isnan(per_label["dice_official_labels"][32])
    assert scalars["dice_official"] == pytest.approx(np.nanmean(per_label["dice_official_labels"]), abs=0)


def test_known_values_identity_and_translation():
    fixed = np.zeros(syn.SHAPE)
    fixed[4:10, 5:12, 6:14] = 1
    fixed[11:15, 3:9, 4:9] = 2
    truth = dict(fixed_seg35=fixed, moving_seg35=fixed.copy(), fixed_seg4=(fixed > 0)*1., moving_seg4=(fixed > 0)*1.,
                 brain=fixed > 0)
    scalars, per_label, _ = M.evaluate_field(np.zeros((3,)+syn.SHAPE), truth, M.OASIS, (1., 1., 1.))
    assert scalars["dice_official"] == 1. and scalars["seg35"] == 1. and scalars["seg4"] == 1. and scalars["hd95"] == 0.
    # std of a constant array is 0 up to the rounding of its mean
    assert scalars["sdlogj_official"] == pytest.approx(0., abs=1e-15) and scalars["central_det_nonpositive"] == 0
    assert scalars["sdlogj_brain"] == 0. and scalars["logj_q01_brain"] == 0. and scalars["logj_q99_brain"] == 0.
    assert scalars["frac_abs_logj_gt_log2_brain"] == 0. and scalars["min_J_centre"] == 1.
    # warped(p) = moving(p + d(p)): moving shifted by +2 voxels in x, d_x = +2 recovers the fixed labels
    moving = np.roll(fixed, 2, axis=2)
    truth.update(moving_seg35=moving, moving_seg4=(moving > 0)*1.)
    d = np.zeros((3,)+syn.SHAPE)
    d[2] = 2.
    scalars, _, _ = M.evaluate_field(d, truth, M.OASIS, (1., 1., 1.))
    assert scalars["dice_official"] == 1. and scalars["seg35"] == 1. and scalars["hd95"] == 0.
    assert scalars["sdlogj_official"] == pytest.approx(0., abs=1e-15) and scalars["sdlogj_brain"] == 0.
    d[2] = -2.
    scalars, _, _ = M.evaluate_field(d, truth, M.OASIS, (1., 1., 1.))
    assert scalars["dice_official"] < 1. and scalars["hd95"] > 0.


def test_brain_log_j_measures_on_a_uniform_stretch():
    """u_x = 1.5 x: J = 2.5 in every cell; log J = log 2.5 > log 2."""
    nz, ny, nx = syn.SHAPE
    u = np.zeros(syn.SHAPE+(3,))
    u[..., 0] = 1.5*np.arange(nx)
    brain = np.zeros(syn.SHAPE, bool)
    brain[3:9, 3:9, 3:9] = True
    stats = M.brain_jacobian_stats(u, (1., 1., 1.), brain)
    assert stats["n_brain_cells"] == 216 and stats["min_J_centre"] == 2.5
    assert stats["logj_q01_brain"] == np.log(2.5) == stats["logj_q99_brain"]
    assert stats["sdlogj_brain"] == pytest.approx(0., abs=1e-15)
    assert stats["frac_abs_logj_gt_log2_brain"] == 1. and stats["pct_J_lt_half_brain"] == 0.
    u[..., 0] = -.75*np.arange(nx)                       # J = .25 -> log J = log .25, |log J| > log 2
    stats = M.brain_jacobian_stats(u, (1., 1., 1.), brain)
    assert stats["pct_J_lt_half_brain"] == 100. and stats["frac_abs_logj_gt_log2_brain"] == 1.
    empty = M.brain_jacobian_stats(u, (1., 1., 1.), np.zeros(syn.SHAPE, bool))
    assert empty["sdlogj_brain"] is None and empty["n_brain_cells"] == 0


def test_cell_centre_jacobian_bitwise_equal_to_archived_smoothness():
    smoothness = syn.archived_smoothness()
    spacing = (1.2, .9, 1.1)
    grid = make_grid_3d(nx=syn.SHAPE[2], ny=syn.SHAPE[1], nz=syn.SHAPE[0], hx=spacing[0], hy=spacing[1], hz=spacing[2],
                        unit="mm")
    u = syn.d_to_u(syn.smooth_d(amplitude=(1.5, -1.1, .9), freq=2), spacing)
    brain = syn.pair_arrays(seed=2)["fixed_seg4"] > 0
    assert same(M.cell_jacobian_centres(u, spacing), smoothness.cell_jacobian_centres(u, grid))
    ours, theirs = M.brain_jacobian_stats(u, spacing, brain), smoothness.smoothness_report(u, grid, brain)
    for key in ("sdlogj_brain", "pct_J_lt_half_brain", "min_J_centre", "n_brain_cells"):
        assert same(ours[key], theirs[key])


def test_lpba40_scheme():
    ids = list(range(21, 75))+[181, 182]
    scheme = M.lpba40_scheme(ids)
    assert dict((m, len(l)) for m, _, l in scheme.macro_sets) == {"lpba56": 56, "lpba54": 54}
    assert 181 not in dict((m, l) for m, _, l in scheme.macro_sets)["lpba54"]
    with pytest.raises(ValueError):
        M.lpba40_scheme(ids[:-1])
    base = syn.labels(4, n_labels=56).astype(int)
    lut = np.array([0]+ids)
    fixed = lut[base].astype(float)
    moving = np.roll(fixed, 1, axis=1)
    truth = dict(fixed_lpba56=fixed, moving_lpba56=moving, fixed_brain=(fixed > 0)*1., brain=fixed > 0)
    scalars, per_label, _ = M.evaluate_field(np.zeros((3,)+syn.SHAPE), truth, scheme, (1., 1., 1.))
    rows = per_label["lpba56_per_label"]
    defined54 = [v for k, v in rows.items() if v is not None and int(k) not in (181, 182)]
    assert scalars["lpba54"] == pytest.approx(np.mean(defined54), rel=1e-15)
    assert scalars["lpba56"] == pytest.approx(np.mean([v for v in rows.values() if v is not None]), rel=1e-15)
    assert len(per_label["dice_official_labels"]) == 56
