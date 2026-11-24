"""C03.2-C03.6 gates on synthetic arrays: simplex, floor bound, smoothing keeps the sum, brain mask,
white-matter normalization with fallback, and the four-level pyramid."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from bayesreg.grids import four_level_grids
from bayesreg.io import AcceptedProvenance
from bayesreg.seg import (brain_mask, channel_pyramid, floor_channels, normalize_intensity, segmentation_qc,
                          simplex_from_background_class, simplex_with_background, smooth_channels, tissue_channels)
from jd_curl.three_d.cell_q1 import CellQ1Operators3D, cell_tiles
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.interpolation import build_image_sampler, sample_points
from jd_curl.validation import InputValidationError

EPS = 1e-3


def random_simplex(n_channels, shape, seed):
    p = np.random.default_rng(seed).gamma(.5, size=(n_channels,)+shape)
    return p/p.sum(axis=0)


def test_simplex_with_background():
    shape = (8, 9, 10)
    rng = np.random.default_rng(1)
    p = random_simplex(3, shape, 2)*rng.uniform(1-5e-7, 1+5e-7, shape)      # segmenter roundoff
    p = np.minimum(p, 1.)
    p[:, :2] = 0                                                             # outside Omega_S
    p[:, 2, :3] *= .4                                                        # partial support is rejected below
    with pytest.raises(InputValidationError, match="Omega_S"):
        simplex_with_background(p)
    p[:, 2, :3] /= .4
    pbar, provenance = simplex_with_background(p)
    assert pbar.shape == (4,)+shape and not pbar.flags.writeable
    assert np.max(np.abs(pbar.sum(axis=0)-1)) <= 1e-12 and pbar.min() >= 0
    assert np.all(pbar[0, :2] == 1) and np.all(pbar[1:, :2] == 0)
    total = p.sum(axis=0)
    expected = np.concatenate([np.clip(1-total, 0, 1)[None], p])
    assert np.max(np.abs(pbar-expected/expected.sum(axis=0))) == 0
    assert provenance["n_classes"] == 3 and provenance["sum_tol"] == 1e-6
    assert 0 < provenance["max_abs_input_sum_deviation"] <= 1e-6


@pytest.mark.parametrize("bad", ["above_one", "negative", "ndim"])
def test_simplex_rejects_out_of_range(bad):
    p = random_simplex(3, (6, 6, 6), 3)
    if bad == "above_one":
        p[0, 0, 0, 0] = 1+1e-12
    elif bad == "negative":
        p[1, 1, 1, 1] = -1e-300
    else:
        p = p[0]
    with pytest.raises(InputValidationError):
        simplex_with_background(p)


def test_smoothing_keeps_the_sum_and_matches_scipy():
    pbar = random_simplex(4, (11, 12, 13), 4)
    phat = smooth_channels(pbar)
    assert np.max(np.abs(phat.sum(axis=0)-1)) <= 1e-12 and phat.min() >= 0
    for k in range(4):
        assert np.array_equal(phat[k], gaussian_filter(pbar[k], sigma=1., mode="nearest", truncate=4.))


def test_floor():
    pbar = random_simplex(4, (9, 10, 11), 5)
    pbar[3] = 0                                                  # an absent class reaches the floor exactly
    pbar /= pbar.sum(axis=0)
    ptilde, provenance = tissue_channels(pbar)
    bound = EPS/(1+4*EPS)
    assert np.max(np.abs(ptilde.sum(axis=0)-1)) <= 1e-12
    assert ptilde.min() >= bound-1e-15 and abs(ptilde[3].min()-bound) <= 1e-18
    assert np.array_equal(ptilde, floor_channels(smooth_channels(pbar)))
    assert provenance["order"] == "smooth-floor (sqrt in channel_pyramid)" and provenance["eps"] == EPS
    assert (provenance["sigma_voxels"], provenance["truncate"], provenance["mode"]) == (1., 4., "nearest")
    assert provenance["floor_denominator"] == 1+4*EPS and provenance["floor_lower_bound"] == bound
    with pytest.raises(TypeError):
        tissue_channels(pbar, background_channel=True)          # the switch lives in channel_pyramid only


def test_simplex_from_background_class():
    """C03.2-4a (D39): the segmenter's own background group is P^0; sum checked on every voxel, then renormalized;
    the floor afterwards uses 1 + (K_S + 1) eps."""
    shape = (7, 8, 9)
    rng = np.random.default_rng(11)
    exact = random_simplex(14, shape, 12)                                     # K_S = 13 groups + background
    p = np.minimum(exact*rng.uniform(1-4e-7, 1+4e-7, exact.shape), 1.)       # float roundoff of the segmenter
    pbar, provenance = simplex_from_background_class(p)
    assert pbar.shape == (14,)+shape and not pbar.flags.writeable
    assert np.max(np.abs(pbar.sum(axis=0)-1)) <= 1e-12 and np.array_equal(pbar, p/p.sum(axis=0))
    assert provenance["n_classes"] == 13 and provenance["renormalized_over_channels"] == 14
    assert 0 < provenance["max_abs_input_sum_deviation"] <= 1e-6 and "D39" in provenance["contract"]
    ptilde, floor = tissue_channels(pbar)
    assert floor["floor_denominator"] == 1+14*EPS and ptilde.min() >= EPS/(1+14*EPS)-1e-15
    # the clip(1 - sum) entry refuses the K_S group channels alone (they do not sum to 1): why 4a exists
    with pytest.raises(InputValidationError, match="Omega_S"):
        simplex_with_background(exact[1:])


@pytest.mark.parametrize("bad", ["sum_one_voxel", "zero_voxel", "above_one", "negative", "one_channel"])
def test_simplex_from_background_class_rejects(bad):
    p = random_simplex(4, (5, 5, 5), 13)
    if bad == "sum_one_voxel":
        p[:, 2, 3, 4] *= 1+2e-6
    elif bad == "zero_voxel":
        p[:, 0, 0, 0] = 0                                                     # no Omega_S exemption here
    elif bad == "above_one":
        p[0, 1, 1, 1] = 1+1e-12
    elif bad == "negative":
        p[2, 1, 1, 1] = -1e-300
    else:
        p = p[:1]/p[:1]
    with pytest.raises(InputValidationError):
        simplex_from_background_class(p)


def blocks_pbar(shape, tissue):
    """Two-class simplex: tissue probability `tissue` in class 1, background takes the rest."""
    pbar = np.zeros((3,)+shape)
    pbar[1] = tissue
    pbar[0] = 1-tissue
    return pbar


def test_brain_mask_keeps_largest_6_connected_component():
    shape = (12, 12, 12)
    tissue = np.zeros(shape)
    tissue[2:6, 2:6, 2:6] = .9                                   # 64 voxels: kept
    tissue[6, 6, 6] = .9                                         # touches the big cube only at a corner (26-nbr)
    tissue[6, 6, 3] = .9                                         # shares only an edge with (5, 5, 3) (18-nbr)
    tissue[5, 5, 6] = .5                                         # not > 1/2
    tissue[8:11, 8:11, 8:11] = .8                                # 27 voxels: second component
    mask, provenance = brain_mask(blocks_pbar(shape, tissue))
    expected = np.zeros(shape, dtype=bool)
    expected[2:6, 2:6, 2:6] = True
    assert mask.dtype == bool and not mask.flags.writeable and np.array_equal(mask, expected)
    assert provenance["n_components"] == 4 and provenance["n_voxels"] == 64 and provenance["connectivity"] == 6
    assert provenance["mask_mode"] == "hard"
    with pytest.raises(InputValidationError):
        brain_mask(blocks_pbar(shape, np.full(shape, .5)))
    with pytest.raises(NotImplementedError):
        brain_mask(blocks_pbar(shape, tissue), dilation_radius=1)
    for mode in ("soft", "t80", None):
        with pytest.raises(NotImplementedError):
            brain_mask(blocks_pbar(shape, tissue), mask_mode=mode)


def test_mask_test_distinguishes_connectivities():
    """The fixture above is not vacuous: 18-connectivity would absorb the edge voxel, 26 the corner one."""
    from scipy.ndimage import generate_binary_structure, label as nd_label
    shape = (12, 12, 12)
    candidate = np.zeros(shape, dtype=bool)
    candidate[2:6, 2:6, 2:6] = candidate[6, 6, 6] = candidate[6, 6, 3] = True
    sizes = {}
    for rank in (1, 2, 3):
        labels, n = nd_label(candidate, structure=generate_binary_structure(3, rank))
        sizes[rank] = (n, int(np.bincount(labels.ravel())[1:].max()))
    assert sizes == {1: (3, 64), 2: (2, 65), 3: (1, 66)}


def wm_case(wm_block, shape=(16, 16, 16), seed=6):
    """Class 1 = white matter (wm_index 0) with probability .95 on `wm_block`, else .2 (tissue = class 2)."""
    rng = np.random.default_rng(seed)
    pbar = np.zeros((3,)+shape)
    pbar[1] = .2
    pbar[1][wm_block] = .95
    pbar[2] = .8
    pbar[2][wm_block] = .05
    mask = np.zeros(shape, dtype=bool)
    mask[1:15, 1:15, 1:15] = True
    pbar[1][~mask] = pbar[2][~mask] = 0
    pbar[0] = 1-pbar[1]-pbar[2]
    return rng.uniform(50., 150., shape), pbar, mask


def tissue_record(class_names, wm_index):
    """Stand-in for an accepted tissue record (test_pair.py feeds the real validator's output). A null
    wm_index is refused for Atropos by the validator; the fallback branch is still part of C03.4-7."""
    return AcceptedProvenance(layer="tissue", segmenter_id="atropos", class_names=list(class_names),
                              wm_index=wm_index, output_sha256={"posteriors01.nii.gz": "0"*64})


WM_FIRST = tissue_record(["WM", "GM"], 0)


@pytest.mark.parametrize("block, count", [((slice(2, 13),)*3, 1331),
                                          ((slice(2, 14), slice(2, 13), slice(2, 12)), 1320)])
def test_wm_median_normalization(block, count):
    image, pbar, mask = wm_case(block)
    normalized, provenance = normalize_intensity(image, pbar, mask, WM_FIRST)
    reference = pbar[1] > .9
    assert reference.sum() == count and provenance["reference_voxels"] == count
    assert provenance["normalization"] == "wm_median" and provenance["fallback_reason"] is None
    assert provenance["wm_index"] == 0 and provenance["wm_class_name"] == "WM"
    assert abs(np.median(normalized[reference])-1) <= 1e-15
    assert provenance["median"] == np.median(image[reference]) and not normalized.flags.writeable


def test_normalization_fallbacks_are_recorded():
    image, pbar, mask = wm_case((slice(3, 12),)*3)                 # 729 < 1000 reference voxels
    normalized, provenance = normalize_intensity(image, pbar, mask, WM_FIRST)
    assert provenance["normalization"] == "brain_median_fallback" and provenance["fallback_reason"] == "|R| = 729 < 1000"
    assert provenance["reference_voxels"] == mask.sum() and abs(np.median(normalized[mask])-1) <= 1e-15
    normalized, provenance = normalize_intensity(image, pbar, mask, tissue_record(["WM", "GM"], None))
    assert provenance["normalization"] == "brain_median_fallback" and provenance["fallback_reason"] == "wm_index is null"
    pbar[1][pbar[1] > .9] = .9                                     # threshold is strict: P^wm = .9 is not in R
    _, provenance = normalize_intensity(image, pbar, mask, WM_FIRST, min_reference_voxels=1)
    assert provenance["normalization"] == "brain_median_fallback"


@pytest.mark.parametrize("record", [
    tissue_record(["WM", "GM"], 2), tissue_record(["WM", "GM"], 1), tissue_record(["WM", "GM", "CSF"], 0),
    tissue_record(["WM", "GM"], True), dict(WM_FIRST), AcceptedProvenance(WM_FIRST, layer="structure"), 0, None])
def test_normalization_requires_a_consistent_tissue_record(record):
    image, pbar, mask = wm_case((slice(2, 13),)*3)
    with pytest.raises(InputValidationError):
        normalize_intensity(image, pbar, mask, record)


def test_segmentation_qc_records_only():
    image, pbar, mask = wm_case((slice(2, 13),)*3)
    qc = segmentation_qc(pbar, mask)
    assert qc["qc_gate"] == "recorded_only" and qc["segmentation_rejected"] is False
    assert len(qc["class_volume_fractions_in_mask"]) == 2 and abs(sum(qc["class_volume_fractions_in_mask"])-1) <= 1e-12


# ---- C03.6 pyramid, tissue layer (K_T = 3) and structure layer (K_S = 5) ---------------------------
# four_level_grids needs >= 5 nodes per axis on the factor-8 level, i.e. n >= 26: the smallest legal
# native grid is used here (26 x 27 x 28 nodes).
K_T, K_S = 3, 5


def synthetic_side(grid, center, seed):
    """Segmenter-like outputs on the support ball r <= 11 mm, zero outside: tissue layer = three soft
    concentric classes (wm, gm, csf), structure layer = five soft shells; each sums to 1 on the ball.

    On this minimum-size grid the factor-8 level has 5 nodes per axis and its N_q swings with the
    ball radius (a scan of r = 8.5..12 mm gave |N_q - N_B|/N_B from .006 to .51); r = 11 mm keeps
    all levels inside the 10 % gate (-.082, -.062, +.009)."""
    r = np.linalg.norm(grid.identity()-np.asarray(center), axis=-1)
    edge = lambda radius: 1/(1+np.exp(-(radius-r)/.6))    # noqa: E731
    support = r <= 11.
    wm, inner = edge(5.), edge(8.)
    tissue = np.stack([wm, inner-wm, 1-inner])*support
    shells = [edge(radius) for radius in (3., 5., 7., 9.)]
    structure = np.stack([shells[0]]+[b-a for a, b in zip(shells, shells[1:])]+[1-shells[-1]])*support
    image = 40+60*wm+30*(inner-wm)+np.random.default_rng(seed).normal(0, 1, grid.shape)
    return image, tissue, structure


@pytest.fixture(scope="module")
def pyramid_case():
    native = make_grid_3d(nx=26, ny=27, nz=28, hx=1., hy=1.1, hz=.9, x0=-3., y0=2., z0=1.5, unit="mm")
    center = np.array([a[0]+(a[-1]-a[0])/2 for a in native.axes])
    inputs = {}
    for side, shift, seed in (("fixed", (0., 0., 0.), 8), ("moving", (.7, -.4, .3), 9)):
        image, tissue, structure = synthetic_side(native, center+shift, seed)
        pbar_t, _ = simplex_with_background(tissue)
        pbar_s, _ = simplex_with_background(structure)
        mask, _ = brain_mask(pbar_t)
        inputs[side], _ = normalize_intensity(image, pbar_t, mask, tissue_record(["WM", "GM", "CSF"], 0))
        inputs["tissue_"+side], _ = tissue_channels(pbar_t)
        inputs["structure_"+side], _ = tissue_channels(pbar_s)
        if side == "fixed":
            inputs["mask"] = mask
    levels, provenance = channel_pyramid(native, **inputs)
    return native, levels, provenance, inputs


def test_pyramid_grids_and_native_level(pyramid_case):
    native, levels, provenance, inputs = pyramid_case
    reference = four_level_grids(native)
    assert [lv.grid.grid_id for lv in levels] == [g.grid_id for g in reference]
    assert levels[3].grid is native and [lv.level for lv in levels] == [0, 1, 2, 3]
    for name in ("fixed", "moving", "tissue_fixed", "tissue_moving", "structure_fixed", "structure_moving", "mask"):
        assert np.array_equal(getattr(levels[3], name), inputs[name])
    assert [row["grid_id"] for row in provenance["levels"]] == [g.grid_id for g in reference]
    assert provenance["levels"][3]["downsample"] is None and provenance["levels"][0]["downsample"]["labels"] is False
    assert (provenance["n_tissue_classes"], provenance["n_structure_classes"]) == (K_T, K_S)


def test_pyramid_simplex_floor_and_sqrt_on_every_level(pyramid_case):
    _, levels, _, _ = pyramid_case
    for lv in levels:
        for name, k in (("tissue_fixed", K_T), ("tissue_moving", K_T), ("structure_fixed", K_S), ("structure_moving", K_S)):
            ptilde = getattr(lv, name)
            assert ptilde.shape == (k+1,)+lv.grid.shape and not ptilde.flags.writeable
            assert np.max(np.abs(ptilde.sum(axis=0)-1)) <= 1e-12
            assert ptilde.min() >= EPS/(1+(k+1)*EPS)-1e-15
        for ptilde, s in ((lv.structure_fixed, lv.s_fixed), (lv.structure_moving, lv.s_moving)):
            assert s.shape == (K_S,)+lv.grid.shape and np.max(np.abs(s**2-ptilde[1:])) <= 1e-15
        assert lv.mask.dtype == bool and lv.mask.shape == lv.grid.shape and not lv.mask.flags.writeable


def test_pyramid_simplex_at_gauss_points(pyramid_case):
    _, levels, _, _ = pyramid_case
    for lv in levels:
        ops = CellQ1Operators3D(lv.grid, location="gauss27")
        for tile in cell_tiles(lv.grid):
            for ptilde in (lv.tissue_fixed, lv.structure_fixed):
                total = sum(ops.interpolate(channel, tile=tile) for channel in ptilde)
                assert np.max(np.abs(total-1)) <= 1e-12


def test_pyramid_level_volume_consistency(pyramid_case):
    native, levels, provenance, inputs = pyramid_case
    n_b = int(inputs["mask"].sum())
    assert provenance["native_n_b"] == n_b and provenance["omega0"] == native.omega
    for lv, row in zip(levels, provenance["levels"]):
        n_q = float(np.sum(lv.grid.node_weights()*lv.mask)/native.omega)
        assert row["n_q"] == n_q and abs(n_q-n_b)/n_b <= .1


def test_pyramid_background_channel_and_rejections(pyramid_case):
    native, default_levels, default_provenance, inputs = pyramid_case
    assert default_provenance["background_channel"] is False and default_provenance["likelihood_channels"] == [1, 2, 3, 4, 5]
    levels, provenance = channel_pyramid(native, **inputs, background_channel=True)
    assert provenance["background_channel"] is True and provenance["likelihood_channels"] == list(range(K_S+1))
    for lv, default in zip(levels, default_levels):
        assert lv.s_fixed.shape[0] == K_S+1 and np.array_equal(lv.s_fixed[1:], default.s_fixed)
        assert np.max(np.abs(lv.s_moving**2-lv.structure_moving)) <= 1e-15
    tiny = np.zeros(native.shape, dtype=bool)
    tiny[14, 13, 13] = True
    with pytest.raises(InputValidationError, match="N_q"):
        channel_pyramid(native, **dict(inputs, mask=tiny))
    with pytest.raises(InputValidationError, match="same number of classes"):
        channel_pyramid(native, **dict(inputs, structure_moving=inputs["tissue_moving"]))
    with pytest.raises(InputValidationError, match="background_channel"):
        channel_pyramid(native, **inputs, background_channel=1)


@pytest.mark.parametrize("rtol", [None, 0, -.1, .1000001, .5, float("inf"), float("nan"), True, "0.1"])
def test_level_volume_gate_cannot_be_switched_off(pyramid_case, rtol):
    native, _, _, inputs = pyramid_case
    with pytest.raises(InputValidationError, match="level_volume_rtol"):
        channel_pyramid(native, **inputs, level_volume_rtol=rtol)


def test_pyramid_level_volume_gate_may_be_tightened(pyramid_case):
    native, _, provenance, inputs = pyramid_case
    worst = max(row["n_q_relative_to_native_n_b"] for row in provenance["levels"])
    channel_pyramid(native, **inputs, level_volume_rtol=min(.1, 1.01*worst))
    with pytest.raises(InputValidationError, match="N_q"):
        channel_pyramid(native, **inputs, level_volume_rtol=.99*worst)


@pytest.mark.parametrize("case", ["sum", "unfloored", "eps_mismatch", "below_floor"])
def test_pyramid_native_entry_gates(pyramid_case, case):
    native, _, _, inputs = pyramid_case
    structure = np.array(inputs["structure_fixed"])
    kwargs = {}
    if case == "sum":
        structure[2, 3, 4, 5] += 2e-12                              # |sum - 1| > 1e-12 at one node
    elif case == "unfloored":
        structure = structure*(1+(K_S+1)*EPS)-EPS                   # undo the floor: absent classes reach ~0
    elif case == "eps_mismatch":
        kwargs = dict(eps=2*EPS)                                   # ptilde floored with eps, checked with 2 eps
    else:                                                          # one channel below the floor, sum kept
        moved = structure[1, 3, 4, 5]-EPS/(1+(K_S+1)*EPS)/2
        structure[1, 3, 4, 5] -= moved
        structure[2, 3, 4, 5] += moved
    expected = "tissue_fixed: a channel is below the floor" if case == "eps_mismatch" else "structure_fixed"
    with pytest.raises(InputValidationError, match=expected):
        channel_pyramid(native, **dict(inputs, structure_fixed=structure), **kwargs)


@pytest.mark.parametrize("level", [0, 3])
def test_moving_channel_spline_interpolates_nodes(pyramid_case, level):
    _, levels, _, _ = pyramid_case
    lv = levels[level]
    for channel in lv.s_moving:
        sampled = sample_points(build_image_sampler(channel, lv.grid), lv.grid.identity(), with_gradient=False)
        assert np.max(np.abs(sampled.value-channel)) <= 1e-12

