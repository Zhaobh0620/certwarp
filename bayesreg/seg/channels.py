"""C03.2-C03.6 tissue channels: background class and simplex, brain mask, white-matter median
normalization, smooth -> floor -> square root, and the four-level pyramid.

Pure numpy/scipy. Every default of the contract (C20 D2-D6 included) is an explicit keyword
argument, and every function returns a provenance dict next to its arrays.

Two segmentation layers (contract C03.1): the tissue layer P_T (K_T classes, Atropos EM, no labels)
feeds tau^2, the brain mask (`brain_mask`) and the white-matter normalization (`normalize_intensity`,
which takes the accepted tissue provenance record); the structure layer P_S (K_S classes, registered
self-trained fold models) feeds only the likelihood channels s^k. Both go through
`simplex_with_background` (or, for a structure segmenter that outputs its own background class,
`simplex_from_background_class`, C03.2-4a) and `tissue_channels` separately (eps and the floor
denominator use that layer's own K), and `channel_pyramid` carries both. The square root s^k and the background-channel
switch live in `channel_pyramid` only (every level, the native one included).

Layout: scalar arrays are (nz, ny, nx); posteriors are channel-first, (K, nz, ny, nx) for the
segmenter's K classes and (K+1, nz, ny, nx) from C03.2 on, background P^0 at channel 0.
`wm_index` (in the tissue record) is the 0-based index into the K_T `class_names`, so white matter
is channel wm_index + 1 of a (K_T+1)-channel array. Atropos numbers its posterior files from 1; the
0-based conversion is done when the files are read, before C03.2.

Stored structure-layer probabilities (npz key "p") are read back by `read_back_probabilities` only (C03.2-4a,
C20 D39: raw channel sum checked on every voxel before the division by it); every read-back path calls it
(bayesreg.seg.structure_layer.read_structure_groups, bayesreg.seg.unet.infer.read_probabilities,
bayesreg.learn.cli.load_structure_layer).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label

from jd_curl.validation import InputValidationError, as_float64_array, readonly_copy
from jd_curl.three_d.grid import Grid3D
from jd_curl.three_d.nifti import downsample_array

from ..grids import FACTORS, four_level_grids
from ..readonly import adopt_readonly, freeze_new
from ..io.pair import AcceptedProvenance, wm_index_problem

_RAISE = dict(over="raise", invalid="raise", divide="raise")
RAW_SUM_TOL = {"float16": 1e-3, "float32": 1e-6, "float64": 1e-6}      # C03.2-4a, D39: raw |sum - 1| per voxel


def _readonly_mask(mask):
    mask = np.array(mask, dtype=bool, copy=True, order="C")
    mask.flags.writeable = False
    return mask


def _channels(values, name, n_min=1):
    array = as_float64_array(values, name)
    if array.ndim != 4 or array.shape[0] < n_min:
        raise InputValidationError(f"{name}: expected (channels>={n_min}, nz, ny, nx)")
    return array


def simplex_with_background(posteriors, *, sum_tol=1e-6):
    """C03.2 (C20 D2 default): P^0 = clip(1 - sum_k P^k, 0, 1), then divide all K+1 channels by their sum.

    Input: the segmenter's K posteriors (K, nz, ny, nx), values in [0, 1], summing to 1 within
    `sum_tol` on Omega_S = {sum_k P^k > 0} (C03.1-3). Returns (pbar, provenance).
    """
    p = _channels(posteriors, "posteriors")
    if np.any(p < 0) or np.any(p > 1):
        raise InputValidationError("posteriors: C03.1-3 values must lie in [0, 1]")
    total = p.sum(axis=0)
    support = total > 0
    deviation = float(np.max(np.abs(total[support]-1))) if np.any(support) else 0.
    if deviation > sum_tol:
        raise InputValidationError(f"posteriors: |sum_k P^k - 1| = {deviation:.3e} > {sum_tol:g} on Omega_S")
    with np.errstate(**_RAISE):
        pbar = np.concatenate([np.clip(1-total, 0., 1.)[None], p])
        pbar = pbar/pbar.sum(axis=0)
    provenance = dict(contract="C03.2", n_classes=p.shape[0], background="clip(1 - sum_k P^k, 0, 1) at channel 0",
                      renormalized_over_channels=p.shape[0]+1, sum_tol=sum_tol,
                      max_abs_input_sum_deviation=deviation, support_voxels=int(support.sum()))
    return freeze_new(pbar), provenance                  # MEM: pbar is new here, read-only in place (same values)


def simplex_from_background_class(probabilities, *, sum_tol=1e-6):
    """C03.2 item 4a (【待复核】, C20 D39): a structure-layer segmenter with its own background class.

    Input: (K_S+1, nz, ny, nx), channel 0 = the segmenter's background group (P^0), channels 1..K_S =
    the K_S likelihood groups (e.g. bayesreg.seg.unet.infer.read_probabilities(dir, "groups")); values
    in [0, 1] and, on EVERY voxel, |sum_{k=0}^{K_S} P^k - 1| <= `sum_tol`, else InputValidationError
    (no clip(1 - sum) is formed). Then all K_S+1 channels are divided by their sum. Smoothing, floor
    (denominator 1 + (K_S+1) eps), square root and pyramid follow unchanged (tissue_channels,
    channel_pyramid); whether P^0 is a likelihood channel stays C20 D2. Not for the tissue layer P_T.
    Returns (pbar, provenance).
    """
    p = _channels(probabilities, "probabilities", 2)
    if np.any(p < 0) or np.any(p > 1):
        raise InputValidationError("probabilities: C03.1-3 values must lie in [0, 1]")
    total = p.sum(axis=0)
    deviation = float(np.max(np.abs(total-1)))
    if deviation > sum_tol:
        raise InputValidationError(f"probabilities: |sum_k P^k - 1| = {deviation:.3e} > {sum_tol:g} "
                                   "(background class included; every voxel)")
    with np.errstate(**_RAISE):
        pbar = p/total
    provenance = dict(contract="C03.2-4a (D39, pending review)", n_classes=p.shape[0]-1,
                      background="segmenter background class at channel 0 (no clip(1 - sum))",
                      renormalized_over_channels=p.shape[0], sum_tol=sum_tol, max_abs_input_sum_deviation=deviation,
                      support_voxels=int(total.size))
    return freeze_new(pbar), provenance                  # MEM: pbar is new here, read-only in place (same values)


def read_back_probabilities(path, *, shape=None):
    """C03.2 item 4a (C20 D39): the read-back of stored structure-layer probabilities, shared by every read-back
    path (bayesreg.seg.structure_layer.read_structure_groups, bayesreg.seg.unet.infer.read_probabilities,
    bayesreg.learn.cli.load_structure_layer); pure numpy, no torch.

    `path`: an npz with key "p", channel-first (channels, nz, ny, nx); `shape`, when given, the required full shape.
    Converted to float64. Refused with InputValidationError: missing key, a storage type outside RAW_SUM_TOL, a
    shape other than `shape`, non-finite or negative values, a voxel with zero total, and -- BEFORE the division --
    a voxel whose raw channel sum differs from 1 by more than RAW_SUM_TOL[storage type] (1e-3 for float16 storage,
    unit roundoff 2^-11 ~ 4.9e-4; 1e-6 for float32 / float64). Then divided by the channel sum.
    Returns (array, record).
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        if "p" not in npz.files:
            raise InputValidationError(f"{path.name}: key 'p' missing")
        raw = npz["p"]
    dtype = str(raw.dtype)
    if raw.dtype.kind != "f" or dtype not in RAW_SUM_TOL:
        raise InputValidationError(f"{path.name}: probabilities stored as one of {sorted(RAW_SUM_TOL)} required, "
                                   f"got {raw.dtype}")
    if shape is not None and raw.shape != tuple(shape):
        raise InputValidationError(f"{path.name}: shape {raw.shape} != (channels,) + grid shape = {tuple(shape)}")
    array = raw.astype(np.float64)
    del raw
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise InputValidationError(f"{path.name}: non-finite or negative probabilities")
    total = array.sum(axis=0)
    if np.any(total <= 0):
        raise InputValidationError(f"{path.name}: a voxel with zero total probability")
    deviation = float(np.max(np.abs(total-1)))
    tol = RAW_SUM_TOL[dtype]
    if not deviation <= tol:
        raise InputValidationError(f"{path.name}: raw channel sum |sum_k P^k - 1| = {deviation:.3e} > {tol:g} "
                                   f"({dtype} storage) on some voxel, before the read-back normalization (C03.2-4a)")
    with np.errstate(**_RAISE):
        array /= total
    return array, dict(file=path.name, stored_dtype=dtype, raw_max_abs_sum_deviation=deviation, raw_sum_tol=tol,
                       raw_sum_checked="every voxel, before the division by the channel sum (C03.2-4a, D39)",
                       read_back="float64 divided by the channel sum (bayesreg.seg.channels.read_back_probabilities)")


def brain_mask(pbar, *, threshold=.5, dilation_radius=0, mask_mode="hard"):
    """C03.3 (C20 D3 default): largest 6-connected component of {sum_{k>=1} Pbar^k > threshold}.

    `pbar` is the fixed image's tissue-layer (K_T+1)-channel simplex from C03.2. Ties between equally
    large components go to the lowest scipy label. Dilation (d_B > 0) and the soft T80 weights
    (`mask_mode` other than "hard") are pending BZ (C20 D3) and raise NotImplementedError.
    Returns (mask bool, provenance).
    """
    pbar = _channels(pbar, "pbar", 2)
    if mask_mode != "hard":
        raise NotImplementedError("C20 D3: only mask_mode = 'hard' is defined (soft T80 weights pending BZ)")
    if dilation_radius != 0:
        raise NotImplementedError("C20 D3: only the default d_B = 0 is defined")
    candidate = pbar[1:].sum(axis=0) > threshold
    labels, n_components = label(candidate)            # default structure = 6-connectivity in 3D
    if n_components == 0:
        raise InputValidationError("brain mask: no voxel with tissue probability above threshold")
    sizes = np.bincount(labels.ravel())[1:]
    mask = labels == 1+int(np.argmax(sizes))
    provenance = dict(contract="C03.3", threshold=threshold, connectivity=6, n_components=int(n_components),
                      candidate_voxels=int(candidate.sum()), n_voxels=int(mask.sum()),
                      dilation_radius=dilation_radius, mask_mode=mask_mode)
    return _readonly_mask(mask), provenance


def normalize_intensity(image, pbar, mask, tissue, *, wm_threshold=.9, min_reference_voxels=1000):
    """C03.4-7 (C20 D4 default): divide by the median over R = {Pbar^wm > wm_threshold}.

    `pbar` and `mask` are this image's own tissue-layer simplex (unsmoothed, unfloored) and brain mask.
    `tissue` is this image's tissue-layer record as returned by `validate_segmentation_provenance`
    (an AcceptedProvenance with layer "tissue"); its `class_names` must have K_T = pbar.shape[0] - 1
    entries and its 0-based `wm_index` selects white matter (channel wm_index + 1 of pbar). If
    `wm_index` is None or |R| < min_reference_voxels, the median over the mask is used and the
    provenance says normalization = "brain_median_fallback" with the reason. Returns
    (normalized image, provenance).
    """
    if not isinstance(tissue, AcceptedProvenance) or tissue.get("layer") != "tissue":
        raise InputValidationError("tissue: the accepted tissue-layer record of validate_segmentation_provenance required")
    pbar = _channels(pbar, "pbar", 2)
    image = as_float64_array(image, "image", pbar.shape[1:])
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != image.shape:
        raise InputValidationError("mask: boolean array of the image shape required")
    class_names, wm_index = tissue.get("class_names"), tissue.get("wm_index")
    problem = wm_index_problem(class_names, wm_index)
    if problem:
        raise InputValidationError(f"tissue record: {problem}")
    if len(class_names) != pbar.shape[0]-1:
        raise InputValidationError(f"tissue record has {len(class_names)} classes, pbar has {pbar.shape[0]-1}")
    reason = None
    if wm_index is None:
        reason = "wm_index is null"
    else:
        reference = pbar[wm_index+1] > wm_threshold
        if int(reference.sum()) < min_reference_voxels:
            reason = f"|R| = {int(reference.sum())} < {min_reference_voxels}"
    if reason is not None:
        reference = mask
    n_reference = int(reference.sum())
    if n_reference == 0:
        raise InputValidationError("normalization: empty reference set")
    median = float(np.median(image[reference]))
    if not np.isfinite(median) or median <= 0:
        raise InputValidationError(f"normalization: reference median {median} is not positive")
    with np.errstate(**_RAISE):
        normalized = image/median
    provenance = dict(contract="C03.4", normalization="wm_median" if reason is None else "brain_median_fallback",
                      fallback_reason=reason, reference_voxels=n_reference, median=median,
                      wm_index=None if wm_index is None else int(wm_index),
                      wm_class_name=None if wm_index is None else class_names[wm_index],
                      wm_threshold=wm_threshold, min_reference_voxels=min_reference_voxels,
                      tissue_segmenter_id=tissue.get("segmenter_id"), tissue_output_sha256=tissue.get("output_sha256"))
    return freeze_new(normalized), provenance            # MEM: new here, read-only in place (same values)


def segmentation_qc(pbar, mask):
    """C03.4-8 (C20 D5 default): record tissue-layer class volume fractions inside the mask; the gate
    does not trigger before M5."""
    pbar = _channels(pbar, "pbar", 2)
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != pbar.shape[1:]:
        raise InputValidationError("mask: boolean array of the image shape required")
    n = int(mask.sum())
    if n == 0:
        raise InputValidationError("segmentation QC: empty mask")
    fractions = [float(pbar[k][mask].sum()/n) for k in range(1, pbar.shape[0])]
    return dict(contract="C03.4-8", qc_gate="recorded_only", segmentation_rejected=False,
                class_volume_fractions_in_mask=fractions)


def smooth_channels(pbar, *, sigma_voxels=1., truncate=4., mode="nearest"):
    """C03.5-9 smoothing: the same Gaussian (sigma in index units on every axis) on all K+1 channels."""
    pbar = _channels(pbar, "pbar", 2)
    # MEM: the stack is new here, read-only in place instead of copied once more (same values)
    return freeze_new(np.stack([gaussian_filter(channel, sigma=sigma_voxels, mode=mode, truncate=truncate)
                                for channel in pbar]))


def floor_channels(phat, *, eps=1e-3):
    """C03.5-9 floor: (Phat^k + eps) / (1 + (K+1) eps) over all K+1 channels; the sum stays 1."""
    phat = _channels(phat, "phat", 2)
    with np.errstate(**_RAISE):
        return freeze_new((phat+eps)/(1+phat.shape[0]*eps))    # MEM: new here, read-only in place


def tissue_channels(pbar, *, sigma_voxels=1., truncate=4., mode="nearest", eps=1e-3):
    """C03.5-9: smooth -> floor, for one layer (eps and K are that layer's).

    Returns (ptilde, provenance): ptilde is the floored (K+1)-channel simplex. For the tissue layer
    it feeds tau^2; for the structure layer the square root s^k = sqrt(ptilde^k) is taken in
    `channel_pyramid` on every level, where the background-channel switch (C20 D2) lives.
    """
    ptilde = floor_channels(smooth_channels(pbar, sigma_voxels=sigma_voxels, truncate=truncate, mode=mode), eps=eps)
    provenance = dict(contract="C03.5", order="smooth-floor (sqrt in channel_pyramid)", sigma_voxels=sigma_voxels,
                      truncate=truncate, mode=mode, eps=eps, floor_denominator=1+ptilde.shape[0]*eps,
                      floor_lower_bound=eps/(1+ptilde.shape[0]*eps))
    return ptilde, provenance


@dataclass(frozen=True)
class ChannelLevel:
    """One pyramid level ell (3 = native); arrays read-only, scalar (nz, ny, nx), channels first.

    tissue_* = floored P_T (K_T+1 channels; tau^2), structure_* = floored P_S (K_S+1 channels),
    s_* = sqrt of the structure likelihood channels, mask = m_ell of the fixed image.
    """
    level: int
    grid: Grid3D
    fixed: np.ndarray
    moving: np.ndarray
    tissue_fixed: np.ndarray
    tissue_moving: np.ndarray
    structure_fixed: np.ndarray
    structure_moving: np.ndarray
    s_fixed: np.ndarray
    s_moving: np.ndarray
    mask: np.ndarray


SIMPLEX_FIELDS = ("tissue_fixed", "tissue_moving", "structure_fixed", "structure_moving")
SIMPLEX_SUM_TOL = 1e-12              # C03 gate: |sum_k Ptilde^k - 1| on every node
FLOOR_TOL = 1e-15                    # C03 gate: min Ptilde^k >= eps/(1 + (K+1) eps) - 1e-15
MAX_LEVEL_VOLUME_RTOL = .1           # C03/C12 gate; may be tightened, not loosened or switched off


def channel_pyramid(native_grid, *, fixed, moving, tissue_fixed, tissue_moving, structure_fixed, structure_moving,
                    mask, eps=1e-3, background_channel=False, mask_threshold=.5, level_volume_rtol=.1):
    """C03.6 (C20 D6 default): four levels on `four_level_grids(native_grid)`, both layers.

    Inputs: normalized images (C03.4), floored tissue and structure simplices of both images
    (C03.5, ptilde of `tissue_channels` with this `eps`) and the fixed brain mask (C03.3). The native
    simplices must already pass the C03 gates (sum 1 within 1e-12 on every node, every channel
    >= eps/(1 + (K+1) eps) - 1e-15), else InputValidationError. Native level: these arrays as given.
    Level ell < 3: each array through `nifti.downsample_array(..., labels=False)` (Gaussian +
    linear), every simplex renormalized over its own K+1 channels, mask_ell = 1[downsampled mask >=
    mask_threshold]. On every level s = sqrt of the structure channels k = 1..K_S, or k = 0..K_S
    when `background_channel` is True (C20 D2, default False; the only place of that switch). The
    coarse grid of every call must equal the level grid (grid_id). C03/C12 gate: N_q,ell =
    sum q_ell m_ell / omega0 within `level_volume_rtol` (0 < rtol <= .1; cannot be switched off) of the
    native N_B, otherwise InputValidationError. Returns (levels, provenance).
    """
    if isinstance(level_volume_rtol, bool) or not isinstance(level_volume_rtol, (int, float)) \
            or not 0 < level_volume_rtol <= MAX_LEVEL_VOLUME_RTOL:
        raise InputValidationError(f"level_volume_rtol: a number in (0, {MAX_LEVEL_VOLUME_RTOL}] is required")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not 0 < eps < 1:
        raise InputValidationError("eps: a number in (0, 1) is required")
    if not isinstance(background_channel, bool):
        raise InputValidationError("background_channel: bool required")
    shape = native_grid.shape
    native = dict(fixed=as_float64_array(fixed, "fixed", shape), moving=as_float64_array(moving, "moving", shape))
    for name, value in zip(SIMPLEX_FIELDS, (tissue_fixed, tissue_moving, structure_fixed, structure_moving)):
        native[name] = _channels(value, name, 2)
        if native[name].shape[1:] != shape:
            raise InputValidationError(f"{name}: (K+1, nz, ny, nx) on the native grid required")
        deviation = float(np.max(np.abs(native[name].sum(axis=0)-1)))
        if deviation > SIMPLEX_SUM_TOL:
            raise InputValidationError(f"{name}: |sum_k - 1| = {deviation:.3e} > {SIMPLEX_SUM_TOL:g} at the native level")
        bound = eps/(1+native[name].shape[0]*eps)
        if float(native[name].min()) < bound-FLOOR_TOL:
            raise InputValidationError(f"{name}: a channel is below the floor eps/(1 + (K+1) eps) = {bound:.6e}")
    for layer in ("tissue", "structure"):
        if native[layer+"_fixed"].shape != native[layer+"_moving"].shape:
            raise InputValidationError(f"{layer}: fixed and moving need the same number of classes")
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != shape:
        raise InputValidationError("mask: boolean array of the native shape required")
    omega0, n_b = native_grid.omega, int(mask.sum())
    if n_b == 0:
        raise InputValidationError("mask: empty brain mask")
    first = 0 if background_channel else 1
    levels, rows = [], []
    for ell, grid in enumerate(four_level_grids(native_grid)):
        if grid is native_grid:
            values, level_mask, downsample_meta = native, mask, None
        else:
            metas = []

            def down(array):
                out, coarse, meta = downsample_array(array, native_grid, grid.shape, labels=False)
                if coarse.grid_id != grid.grid_id:
                    raise InputValidationError(f"level {ell}: downsample grid differs from four_level_grids")
                metas.append(meta)
                return out
            values = dict(fixed=down(native["fixed"]), moving=down(native["moving"]))
            for name in SIMPLEX_FIELDS:
                stacked = np.stack([down(channel) for channel in native[name]])
                with np.errstate(**_RAISE):
                    values[name] = stacked/stacked.sum(axis=0)
            level_mask = down(mask.astype(np.float64)) >= mask_threshold
            downsample_meta = metas[0]
        with np.errstate(**_RAISE):
            n_q = float(np.sum(grid.node_weights()*level_mask)/omega0)
        relative = abs(n_q-n_b)/n_b
        if relative > level_volume_rtol:
            raise InputValidationError(f"level {ell}: |N_q - N_B|/N_B = {relative:.3f} > {level_volume_rtol}")
        # MEM (memory only, same values): an input that is already an owned read-only float64 array (the native level:
        # the callers' normalize_intensity / tissue_channels outputs) is shared instead of copied; the coarse simplices
        # and the square roots are new here and made read-only in place
        if grid is native_grid:
            arrays = {name: adopt_readonly(values[name]) for name in ("fixed", "moving")+SIMPLEX_FIELDS}
        else:
            arrays = dict({name: readonly_copy(values[name]) for name in ("fixed", "moving")},
                          **{name: freeze_new(values[name]) for name in SIMPLEX_FIELDS})
        levels.append(ChannelLevel(
            level=ell, grid=grid, **arrays, mask=_readonly_mask(level_mask),
            s_fixed=freeze_new(np.sqrt(values["structure_fixed"][first:])),
            s_moving=freeze_new(np.sqrt(values["structure_moving"][first:]))))
        del values, arrays
        rows.append(dict(level=ell, factor=FACTORS[ell], shape_zyx=list(grid.shape), spacing_xyz=list(grid.spacing),
                         grid_id=grid.grid_id, n_q=n_q, n_q_relative_to_native_n_b=relative,
                         downsample=downsample_meta))
    provenance = dict(contract="C03.6", grids="four_level_grids (copied from eb808be)", factors=list(FACTORS),
                      omega0=omega0, native_n_b=n_b, mask_threshold=mask_threshold, level_volume_rtol=level_volume_rtol,
                      eps=eps, native_simplex_sum_tol=SIMPLEX_SUM_TOL, floor_tol=FLOOR_TOL,
                      n_tissue_classes=native["tissue_fixed"].shape[0]-1,
                      n_structure_classes=native["structure_fixed"].shape[0]-1,
                      simplex_renormalized_on_coarse_levels=True, background_channel=background_channel,
                      likelihood_channels=list(range(first, native["structure_fixed"].shape[0])),
                      likelihood_layer="structure", levels=rows)
    return tuple(levels), provenance
