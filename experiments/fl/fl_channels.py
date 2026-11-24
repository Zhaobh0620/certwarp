"""FL's channels of predicted labels (runs/20261105-fl/plan.md): ef_v1's fast path, bit for bit, with the two things it does not have: the weights of Q3h with a D factor (arms H, Hx), the image as a channel (arm I).  It takes the PairInput, which holds no label, and mc0p_core.Predicted.

Arms F, M, S, T, X: the numbers of fast_channels.prepare_stacks (a test compares).  H, Hx: the numbers of mc0p_core.prepare_stacks for Q3h, with Q3x3's D factor rule at factor 2 (a test compares).
I: K = 36, channel 0 the image as P1 and Q1 take it (mc0_core.channels: unsmoothed, in the image's own intensity), channels 1..35 the G_.75 one-hot; no weight of its own: one scale factor for all 36, by MC0's rule.
"""
import numpy as np
from scipy.ndimage import find_objects, gaussian_filter

import fast_channels as fast
import mc0_core as core
import mc0p_core as mc

SIDES = ('fixed', 'moving')


def onehot_with_intensity(image, L, workers=1):
    """(36, nz, ny, nx) float64: out[0] = image; out[1:] are fast.onehot_channels(L), written in place (a test compares): the host never holds a second copy of the stack."""
    L = np.asarray(L)
    boxes = find_objects(L.astype(np.int32), max_label=len(fast.LABELS))
    out = np.zeros((len(fast.LABELS)+1,)+L.shape, dtype=np.float64)
    out[0] = image

    def fill(c):
        box = boxes[c]
        if box is not None:
            crop = fast._grown(box, L.shape, (fast.RADIUS,)*3)
            out[c+1][crop] = gaussian_filter((L[crop] == fast.LABELS[c]).astype(np.float64), sigma=fast.SIGMA, mode='nearest', truncate=fast.TRUNCATE)
    core._pool_map(fill, range(len(fast.LABELS)), workers)
    return out


def weigh_in_place(stack, root, workers=1):
    def one(c):
        stack[c] *= root[c]
    core._pool_map(one, range(len(stack)), workers)


def prepare_stacks(spec, pin, predicted, table, device, workers=1):
    """([F, M], record): the arm's two stacks, weighted, scaled by the rule of MC0 (||g_D|| at u = 0, native level, equal to Q0's), then the arm's D factor.  The check of the scale is taken by the caller from optimize_k's record of the native level (scale_check_of): no second upload, as fast_channels."""
    labels = predicted.labels
    pool = 2 if workers > 1 else 1
    each = max(1, workers//2)
    q0 = core._pool_map(lambda side: fast.q0_image(getattr(pin, side), labels[side], table), SIDES, pool)
    reference = core.data_grad_norm(q0[0], q0[1], device)
    del q0
    record = dict(D_factor=spec['D_factor'], g_norm_Q0=reference, channel_weights=None, fixed_voxels=None, images_equal_a5p_inputs='not compared here: ef_v1 compared the same reader; g_norm_Q0 and D_multiplier tie the two (decide_fl.controls)')
    if spec['channels'] == 'onehot':
        F, M = core._pool_map(lambda side: fast.onehot_channels(labels[side], each), SIDES, pool)
    elif spec['channels'] == 'onehot+intensity':
        F, M = core._pool_map(lambda side: onehot_with_intensity(getattr(pin, side), labels[side], each), SIDES, pool)
    else:
        raise SystemExit(f"FL's predicted channels have no kind {spec['channels']}")
    if spec['weight'] is not None:
        w, voxels = mc.channel_weights(spec['weight'], labels['fixed'])
        roots = np.sqrt(w)
        weigh_in_place(F, roots, workers)
        weigh_in_place(M, roots, workers)
        record.update(channel_weights=w.tolist(), fixed_voxels=voxels.tolist())
    unscaled = core.data_grad_norm(F, M, device)
    s = core.scale_factor(reference, unscaled)*spec['D_factor']
    root = np.sqrt(s)
    fast.scale_in_place(F, root, workers)
    fast.scale_in_place(M, root, workers)
    return [F, M], dict(record, D_multiplier=s, channel_multiplier=float(root), g_norm_unscaled=unscaled, g_norm_scaled=None)


def image_channel_norm(stacks, device):
    """Arm I: ||g_D|| at u = 0 of channel 0 alone, after the scale: how much the image weighs against g_norm_Q0 (norms do not add).  A record for telling "the image did not help" from "the image weighed nothing": the caller keeps it out of the task clock."""
    return core.data_grad_norm(stacks[0][:1], stacks[1][:1], device)


def scale_check_of(record, scale):
    """The norm of the native tensors the loop ran on (optimize_k's record) against D_factor x Q0's.  A wall cap on a coarse level never reached them: passed None."""
    norms = record['level_g_norm']
    target = scale['g_norm_Q0']*scale['D_factor']
    if len(norms) < 4:
        return dict(g_norm_scaled=None, target=target, tol=core.SCALE_TOL, passed=None)
    scaled = norms[3]['raw']
    return dict(g_norm_scaled=scaled, target=target, tol=core.SCALE_TOL, passed=abs(scaled/target-1) <= core.SCALE_TOL)
