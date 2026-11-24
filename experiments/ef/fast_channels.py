"""E2 of EF (runs/20261104-ef/e2_profile.out, plan.md): the channels of Q2 and their pyramid in less time, bit for bit the same numbers.  Only the implementation changes; experiments/mc0's entries do not use this module.

The profile (one real development pair, predictions only, CPU, one thread): most of the 21 s of `seconds_channels` is not the channels but getting L = argmax P out of probs36.npz: float16 -> float64, division by the channel sum, np.argmax along 36 channels.  Then Q0, then the two uploads of the scale rule.

Why each replacement is the same number:
  labels   L = argmax_l P_l with P_l = p_l / sum_k p_k, ties to the smallest l.  p_l are float16.  Non-negative float16 compare as their uint16 bit patterns; two different values differ by at least 2^-11 relatively, and an IEEE division by the same positive number keeps them in strict order, so the argmax of the raw
           patterns (first strict maximum) is the argmax of P.  A pattern with the sign bit (negative or -0.0), NaN or Inf goes to the old path, which decides.  The checks stay: every voxel's raw sum in float64 (exact: at most 36 values with 2^-24 granularity), the same tolerance, zero total, the group sums against structure_groups.npz.
  one-hot  G_.75 of a mask: zero outside the mask's box grown by the kernel radius.  The filter on that box does the same additions on the same values, and adding +0.0 changes nothing.  Mode nearest: at a face of the volume the box ends there too; elsewhere the box ends in zeros and the volume has zeros there.
  Q0       the A5 colouring by a lookup table instead of 36 boolean assignments.
  pyramid  downsample_array of F: gaussian_filter, then map_coordinates, order 1.  Same box argument.  The coordinates are shifted by an integer: exact, the fractional part is unchanged, so are the weights.  Outside the box every coefficient is +0.0.
Tests: tests/test_bayesreg/solve/test_ef_tools.py against the old functions, bit for bit; in the batch, the field of F600 against mc0p's Q2 fine field.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, map_coordinates

import mc0_core as core
import mc0p_core as mc

SIGMA, TRUNCATE = .75, 4.0                      # make_inputs.smooth, a test compares
RADIUS = int(TRUNCATE*SIGMA+.5)                 # scipy's kernel radius for it
LABELS = core.LABELS_35
N_CLASSES = 36
SIGN_OR_SPECIAL = 0x7c00                        # a float16 bit pattern at or above it: Inf, NaN or a sign bit


def _half_table():
    table = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float64)
    table.flags.writeable = False
    return table


HALF = _half_table()


def _map(fn, items, workers):
    return core._pool_map(fn, items, workers)


# ---- the label map ----------------------------------------------------------------------------------------------------------------------------
def raw_labels(path, shape, groups, group_labels, workers=1):
    """(L, p0, read-back record, group-sum deviation) from the npz at path, the checks of read_back_probabilities and of make_inputs.unet_posterior on the raw patterns.  None when the old path must decide: another storage or shape, a sign bit, NaN or Inf (it refuses them or reads them).
    SystemExit on the refusals that are certain: zero total, raw sum out of tolerance.  groups: the (K+1, ...) read-back of structure_groups.npz; group_labels: the labels of each of its channels."""
    from bayesreg.seg.channels import RAW_SUM_TOL
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        if 'p' not in npz.files:
            return None
        raw = npz['p']
    if raw.dtype != np.float16 or raw.shape != tuple(shape):
        return None
    bits = raw.view(np.uint16)
    nz = shape[1]
    chunks = max(1, workers*2)
    edges = np.linspace(0, nz, chunks+1).astype(int)
    L = np.empty(shape[1:], dtype=np.int16)
    p0 = np.empty(shape[1:], dtype=np.float64)

    def slab(k):
        z0, z1 = edges[k], edges[k+1]
        if z0 == z1:
            return dict(special=False, zero=False, raw=0., group=0.)
        part = bits[:, z0:z1]
        if max(int(part[c].max()) for c in range(N_CLASSES)) >= SIGN_OR_SPECIAL:
            return dict(special=True, zero=False, raw=0., group=0.)
        total = np.zeros(part.shape[1:])
        best = part[0].copy()
        labels = np.zeros(part.shape[1:], dtype=np.int16)
        for c in range(N_CLASSES):
            total += HALF[part[c]]
            if c:
                better = part[c] > best
                labels[better] = c
                np.maximum(best, part[c], out=best)
        L[z0:z1] = labels
        if np.any(total <= 0):
            return dict(special=False, zero=True, raw=0., group=0.)
        worst = 0.
        for k_group, members in enumerate(group_labels):
            s = None
            for label in members:
                p = HALF[part[label]]/total
                s = p if s is None else s+p
            worst = max(worst, float(np.max(np.abs(s-groups[k_group, z0:z1]))))
        p0[z0:z1] = HALF[part[0]]/total
        return dict(special=False, zero=False, raw=float(np.max(np.abs(total-1))), group=worst)
    rows = _map(slab, range(chunks), workers)
    if any(r['special'] for r in rows):
        return None
    if any(r['zero'] for r in rows):
        raise SystemExit(f'{path.name}: a voxel with zero total probability')
    deviation = max(r['raw'] for r in rows)
    tol = RAW_SUM_TOL['float16']
    if not deviation <= tol:
        raise SystemExit(f'{path.name}: raw channel sum |sum_k P^k - 1| = {deviation:.3e} > {tol:g} (float16 storage) on some voxel, before the read-back normalization (C03.2-4a)')
    record = dict(file=path.name, stored_dtype='float16', raw_max_abs_sum_deviation=deviation, raw_sum_tol=tol, raw_sum_checked='every voxel, before the division by the channel sum (C03.2-4a, D39)', read_back='float16 bit patterns, the sum in float64 (experiments/ef/fast_channels.py); no float64 stack')
    return L, p0, record, max(r['group'] for r in rows)


def read_side(layer, pin, side, seg_root, workers=1):
    """(L, the side's record): make_inputs.unet_posterior and the per-side part of mc0p_core.read_predicted, the same refusals in the same order, on raw_labels; the old path when raw_labels says so."""
    from jd_curl.validation import InputValidationError
    from bayesreg.io.pair import ProvenanceRejected
    mi = core.make_inputs()
    subject = layer.subjects[side]
    ids, policy = layer.expected[side]
    if policy != 'all_folds_mean':
        raise SystemExit(f'A4p {side} {subject}: policy {policy} != all_folds_mean (D24: a development subject is in no fold model\'s label contact)')
    try:
        groups, accepted, rec = layer.load(side)
    except (ProvenanceRejected, InputValidationError) as exc:
        raise SystemExit(f'A4p {side} {subject}: structure layer refused ({type(exc).__name__}: {exc})')
    directory = Path(rec['directory'])
    if 'probs36.npz' not in rec['verified_outputs']:
        raise SystemExit(f'A4p {side} {subject}: probs36.npz is not among the hash-verified outputs')
    g = layer.grouping
    if mi.UNET_WM_GROUP not in g.group_names or tuple(g.group_labels[g.group_names.index(mi.UNET_WM_GROUP)]) != mi.CEREBRAL_WM:
        raise SystemExit(f'A4p {side} {subject}: grouping {g.name} has no group {mi.UNET_WM_GROUP} of seg35 labels {mi.CEREBRAL_WM} (alignment QC, protocol §12.4 item 5)')
    wm_channel = 1+g.group_names.index(mi.UNET_WM_GROUP)
    fast = raw_labels(directory/'probs36.npz', (N_CLASSES,)+groups.shape[1:], groups, (g.background_labels,)+tuple(g.group_labels), workers)
    if fast is None:                                # the old path decides, and reads everything again
        P, record, u_wm = mi.unet_posterior(layer, side)
        labels, p0 = mc.predicted_labels(P), np.array(P[0])
        del P
    else:
        labels, p0, read_back, deviation = fast
        u_wm = np.array(groups[wm_channel])
        if not deviation <= mi.GROUP_TOL:
            raise SystemExit(f'A4p {side} {subject}: probs36 summed by {g.name} differs from structure_groups.npz by {deviation:.3e} > {mi.GROUP_TOL:g}')
        record = dict(unet_provenance_sha256=rec['provenance_sha256'], unet_output_sha256=rec['verified_outputs'], unet_directory=str(directory), segmenter_id=accepted['segmenter_id'], model_ids=list(ids),
                      policy=policy, members=rec['models']['members'], registry_path=layer.registry.path, registry_sha256=layer.registry.sha256, registry_status=layer.registry_record['status'],
                      registry_git={k: (layer.registry_record.get('git') or {}).get(k) for k in ('tracked', 'clean', 'last_commit')}, read_back_probs36=read_back, read_back_groups=rec['read_back'],
                      group_sum_max_abs_deviation=deviation, group_sum_tol=mi.GROUP_TOL, grouping=rec['grouping'], prediction_code=rec['prediction_code'],
                      qc_wm_group=dict(file='structure_groups.npz', name=mi.UNET_WM_GROUP, channel=wm_channel, labels=list(mi.CEREBRAL_WM)))
    del groups
    support = mi.support_dice_record(getattr(pin, side), p0)
    del p0
    bound = getattr(layer, 'allowlist', None)                   # HB (C23.4-8): a hold-out pair's Atropos provenance is rechecked against the bound allowlist load_pair admitted it through
    a_wm, atropos = mi.atropos_wm(seg_root, pin, side, allowlist=bound) if mi.unsealed_allowlist(bound) else mi.atropos_wm(seg_root, pin, side)
    qc = mi.alignment_qc(u_wm, a_wm)
    del u_wm, a_wm
    if not qc['passed']:
        raise SystemExit(f"EF {pin.fixed_subject} <- {pin.moving_subject}:{side}: alignment QC {qc['dice']} < {qc['threshold']} (protocol §12.4 item 5)")
    return labels, dict(record, subject=getattr(pin, side+'_subject'), alignment_qc=qc, support_dice_record=support, atropos_qc=atropos, read_path='old' if fast is None else 'fast')


def read_predicted(pin, *, unet=None, seg_root=None, allowlist=None, workers=1):
    """mc0p_core.Predicted, soft None: the labels of mc0p_core.read_predicted bit for bit, after the same checks, the two images in parallel.  It takes the PairInput, which holds no label."""
    mi = core.make_inputs()
    unet = mc.default_unet() if unet is None else unet
    seg_root = mc.SEG_ROOT if seg_root is None else seg_root
    allowlist = core.ALLOWLIST if allowlist is None else allowlist
    admission = mi.a4_admission([pin.fixed_subject, pin.moving_subject], allowlist, unet)
    layer = mi.pair_structure_layer(pin, allowlist, unet)
    each = max(1, workers//2)
    rows = _map(lambda side: read_side(layer, pin, side, seg_root, each), ('fixed', 'moving'), 2 if workers > 1 else 1)
    labels = dict(fixed=rows[0][0], moving=rows[1][0])
    sides = dict(fixed=rows[0][1], moving=rows[1][1])
    keys_read = list(pin.provenance['keys_read'])
    return mc.Predicted(labels, None, dict(**mc.oracle_mark(keys_read, sides), pair_keys_read=keys_read, label_firewall=pin.provenance['label_firewall'], segmenter=unet.record(), atropos_root=str(seg_root),
                                           subject_admission=admission, structure_layer=layer.record(), sides=sides))


# ---- the channels -----------------------------------------------------------------------------------------------------------------------------
def _grown(box, shape, radius):
    return tuple(slice(max(s.start-r, 0), min(s.stop+r, n)) for s, n, r in zip(box, shape, radius))


def onehot_channels(L, workers=1):
    """core.channels('P2', image, L) bit for bit: (35, nz, ny, nx) float64, G_.75 of each label's mask, 0 in the rest."""
    L = np.asarray(L)
    boxes = find_objects(L.astype(np.int32), max_label=len(LABELS))
    out = np.zeros((len(LABELS),)+L.shape, dtype=np.float64)

    def fill(c):
        box = boxes[c]
        if box is not None:
            crop = _grown(box, L.shape, (RADIUS,)*3)
            out[c][crop] = gaussian_filter((L[crop] == LABELS[c]).astype(np.float64), sigma=SIGMA, mode='nearest', truncate=TRUNCATE)
    _map(fill, range(len(LABELS)), workers)
    return out


def q0_image(I, L, table):
    """core.channels('P0', I, L, table=table) bit for bit, with make_inputs.colour_from_table's refusals: (1, nz, ny, nx)."""
    mi = core.make_inputs()
    L = np.asarray(L)
    if L.shape != I.shape:
        raise SystemExit(f'colour_from_table: label map shape {L.shape} != image shape {I.shape}')
    if L.min() < 0 or L.max() >= N_CLASSES:
        raise SystemExit(f'colour_from_table: label map has values outside 0..{N_CLASSES-1}')
    white = (L == mi.CEREBRAL_WM[0]) | (L == mi.CEREBRAL_WM[1])
    if not white.any():
        raise SystemExit('colour_from_table: no cerebral white matter in the label map (w undefined)')
    w = float(I[white].mean())
    del white
    lookup = np.zeros(N_CLASSES, dtype=I.dtype)
    for label in range(1, N_CLASSES):
        lookup[label] = table[label]*w
    return mi.smooth(lookup[L.astype(np.intp)])[None]


def scale_in_place(stack, factor, workers=1):
    def one(c):
        stack[c] *= factor
    _map(one, range(len(stack)), workers)


def prepare_stacks(spec, pin, predicted, table, device, workers=1):
    """([F, M], record): the arm's two stacks scaled by the rule of MC0 (||g_D|| at u = 0, native level, equal to Q0's).  spec['channels']: 'onehot' (Q2, bit for bit mc0p_core.prepare_stacks) or 'colour8'.
    The second gradient norm of the old path, its check of the scale, is taken by the caller from optimize_k's record of the native level: no second upload."""
    import vector_colour as vc
    labels = predicted.labels
    q0 = _map(lambda side: q0_image(getattr(pin, side), labels[side], table), ('fixed', 'moving'), 2 if workers > 1 else 1)
    reference = core.data_grad_norm(q0[0], q0[1], device)
    del q0
    record = dict(D_factor=1., g_norm_Q0=reference, channel_weights=None, fixed_voxels=None, images_equal_a5p_inputs='not compared here: the arm Q2 of the same pair compares them; g_norm_Q0 and D_multiplier tie the two (decide_ef.controls)')
    if spec['channels'] == 'onehot':
        each = max(1, workers//2)
        F, M = _map(lambda side: onehot_channels(labels[side], each), ('fixed', 'moving'), 2 if workers > 1 else 1)
    elif spec['channels'] == 'colour8':
        colours, colouring = vc.choose_colouring(labels['fixed'], labels['moving'], K=spec['k'], workers=workers)
        record['colouring'] = colouring
        mi = core.make_inputs()
        each = max(1, workers//2)
        F, M = _map(lambda side: vc.vector_channels(labels[side], colours, spec['k'], smooth=mi.smooth, workers=each), ('fixed', 'moving'), 2 if workers > 1 else 1)
    else:
        raise SystemExit(f"no channels {spec['channels']}")
    unscaled = core.data_grad_norm(F, M, device)
    s = core.scale_factor(reference, unscaled)*record['D_factor']
    root = np.sqrt(s)
    scale_in_place(F, root, workers)
    scale_in_place(M, root, workers)
    return [F, M], dict(record, D_multiplier=s, channel_multiplier=float(root), g_norm_unscaled=unscaled, g_norm_scaled=None)


# ---- the pyramid ------------------------------------------------------------------------------------------------------------------------------
class FastDownsample:
    """nifti.downsample_array(values, source_grid, shape_zyx) bit for bit on an array mostly 0: (values, grid, None).  The box of a channel is computed once for its three levels (keyed by the data pointer: only while the stacks live, one instance per pyramid)."""

    def __init__(self):
        self.boxes = {}

    def box(self, values):
        key = values.__array_interface__['data'][0]
        if key not in self.boxes:
            nonzero = values != 0
            self.boxes[key] = None if not nonzero.any() else [(int(np.argmax(p)), len(p)-1-int(np.argmax(p[::-1]))) for p in (nonzero.any(axis=(1, 2)), nonzero.any(axis=(0, 2)), nonzero.any(axis=(0, 1)))]
        return self.boxes[key]

    def __call__(self, values, source_grid, shape_zyx):
        from jd_curl.three_d.nifti import coarse_grid
        grid = coarse_grid(source_grid, shape_zyx)
        old = np.asarray(source_grid.shape)
        ratios = (old-1)/(np.asarray(shape_zyx)-1)
        sigma = .5*np.sqrt(ratios**2-1)
        out = np.zeros(tuple(shape_zyx))
        box = self.box(values)
        if box is None:
            return out, grid, None
        axes = [np.linspace(0, n-1, new) for n, new in zip(source_grid.shape, shape_zyx)]
        radius = [int(TRUNCATE*s+.5) for s in sigma]                                    # kernel radius; one more below so that the interpolation's stencil stays inside
        crop = tuple(slice(max(lo-r-1, 0), min(hi+r+2, n)) for (lo, hi), r, n in zip(box, radius, old))
        smooth = gaussian_filter(values[crop], sigma=sigma, mode='nearest', truncate=TRUNCATE)
        chosen, shifted = [], []
        for a, (lo, hi), r, c in zip(axes, box, radius, crop):
            f = np.floor(a)
            keep = np.flatnonzero((f+1 >= lo-r) & (f <= hi+r))
            chosen.append(keep)
            shifted.append(a[keep]-c.start)
        if min(len(k) for k in chosen):
            coords = np.array(np.meshgrid(*shifted, indexing='ij'))
            out[np.ix_(*chosen)] = map_coordinates(smooth, coords, order=1, mode='nearest', prefilter=False)
        return out, grid, None
