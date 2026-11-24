"""E2 of EF: the profile of the Q2 channel path and the bit-for-bit comparison of the faster one, on real development pairs: predictions only, no truth label, no optimizer, no unit.  CPU, one thread, nice 19 next to whatever else runs on the machine (the seconds are for comparing the steps, not the seconds of a run: the queue's records have those).

    PYTHONPATH=<frozen>:<code> python -B experiments/ef/profile_e2.py > runs/20261104-ef/e2_profile.out

Old path: mc0p_core.read_predicted, core.channels, nifti.downsample_array.  New path: fast_channels.py.  The colourings of the 5 pairs at the end (vector_colour.py).
"""
import cProfile
import hashlib
from pathlib import Path
import pstats
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent/'mc0'))

import numpy as np  # noqa: E402

import fast_channels as fast  # noqa: E402
import mc0_core as core  # noqa: E402
import mc0p_core as mc  # noqa: E402
import vector_colour as vc  # noqa: E402


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def timed(name, fn, rows):
    t = perf_counter()
    value = fn()
    rows.append((name, perf_counter()-t))
    print(f'{name}: {rows[-1][1]:.2f} s', flush=True)
    return value


def needed(graph):
    """The fewest colours DSATUR's tight packing fits on this graph."""
    return next(K for K in range(1, 36) if vc.dsatur(graph, K) is not None)


def main():
    fz = core.frozen_api()
    pair = core.PAIRS[0]
    table = core.load_table()
    print(f'pair {pair}; CPU, one thread, nice 19; the machine also runs other CPU jobs: {Path("/proc/loadavg").read_text().split()[:3]}')
    pin = mc.read_images(pair)
    print(f'grid {pin.grid.shape}, spacing {pin.grid.spacing}')
    rows = []
    profile = cProfile.Profile()
    profile.enable()
    old = timed('old read_predicted, both images', lambda: mc.read_predicted(pin, workers=1), rows)
    profile.disable()
    pstats.Stats(profile, stream=sys.stdout).sort_stats('tottime').print_stats(10)
    new1 = timed('fast read_predicted, both images, one thread', lambda: fast.read_predicted(pin, workers=1), rows)
    new8 = timed('fast read_predicted, both images, 7 workers (the machine is shared: an upper bound)', lambda: fast.read_predicted(pin, workers=7), rows)
    labels_equal = all(old.labels[side].tobytes() == new1.labels[side].tobytes() == new8.labels[side].tobytes() for side in ('fixed', 'moving'))
    print(f"labels bit for bit, old against fast (one thread, 7 workers): {labels_equal}; dtype {old.labels['fixed'].dtype} {new1.labels['fixed'].dtype}")
    for side in ('fixed', 'moving'):
        a, b = old.record['sides'][side], new1.record['sides'][side]
        same = {k: a[k] == b[k] for k in a if k not in ('read_back_probs36', 'group_sum_max_abs_deviation') and k in b}
        print(f"record {side}: keys only in the old {sorted(set(a)-set(b))}, only in the fast {sorted(set(b)-set(a))}; differing values {[k for k, v in same.items() if not v]}; raw deviation {a['read_back_probs36']['raw_max_abs_sum_deviation']} / {b['read_back_probs36']['raw_max_abs_sum_deviation']}; group deviation {a['group_sum_max_abs_deviation']} / {b['group_sum_max_abs_deviation']}")
    print(f"oracle {old.record['oracle']} / {new1.record['oracle']}, members of pair.npz read {old.record['pair_keys_read']} / {new1.record['pair_keys_read']}")
    del new8
    all_equal = labels_equal
    for side in ('fixed', 'moving'):
        image, L = getattr(pin, side), new1.labels[side]
        q_old = timed(f'old Q0 {side}', lambda: core.channels('P0', image, L, table=table), rows)
        q_new = timed(f'fast Q0 {side}', lambda: fast.q0_image(image, L, table), rows)
        same = q_old.tobytes() == q_new.tobytes()
        all_equal = all_equal and same
        print(f'Q0 {side} bit for bit: {same}  {sha(q_new)}')
        del q_old, q_new
        one_old = timed(f'old one-hot channels {side}', lambda: core.channels('P2', image, L), rows)
        one_new = timed(f'fast one-hot channels {side}', lambda: fast.onehot_channels(L), rows)
        same = one_old.tobytes() == one_new.tobytes()
        all_equal = all_equal and same
        print(f'35 channels {side} bit for bit: {same}  {sha(one_new)}')
        del one_old
        grids = fz.four_level_grids(pin.grid)
        downsample = fast.FastDownsample()
        spent_old, spent_new, compared = 0., 0., 0
        for c in (0, 10, 16, 19):                       # the two largest labels of the first and the second hemisphere's table rows, and two small ones: all three levels
            for level in range(3):
                t = perf_counter()
                a, _, _ = fz.downsample_array(one_new[c], pin.grid, grids[level].shape)
                spent_old += perf_counter()-t
                t = perf_counter()
                b, got, _ = downsample(one_new[c], pin.grid, grids[level].shape)
                spent_new += perf_counter()-t
                same = a.tobytes() == b.tobytes() and got.grid_id == grids[level].grid_id
                all_equal = all_equal and same
                compared += 1
                if not same:
                    print(f'pyramid DIFFERS {side} channel {c} level {level}: largest difference {np.abs(a-b).max()}')
        print(f'pyramid {side}: {compared} channel-levels bit for bit against nifti.downsample_array so far true: {all_equal}; {spent_old:.2f} s old, {spent_new:.2f} s fast (these channels only)')
        t = perf_counter()
        fast_all = [downsample(one_new[c], pin.grid, grids[level].shape) for c in range(len(core.LABELS_35)) for level in range(3)]
        print(f'pyramid {side}: all 35 channels, 3 levels, fast, one thread: {perf_counter()-t:.2f} s')
        del fast_all, one_new
    print('--- the K = 8 colourings of the 5 pairs (predictions only)')
    for pair in core.PAIRS:
        pin = mc.read_images(pair)
        predicted = fast.read_predicted(pin, workers=1)
        t = perf_counter()
        colours, record = vc.choose_colouring(predicted.labels['fixed'], predicted.labels['moving'], K=8)
        seconds = perf_counter()-t
        cliques = vc.contact_matrix(predicted.labels['fixed'], 1) | vc.contact_matrix(predicted.labels['moving'], 1)
        print(f"{core.short_name(pair)} K 8: {record['mode']}, {record['hard_edges']} face edges, {record['soft_edges']} edges within {record['soft_reach']} voxels, same colour within that reach: {record['same_colour_within_soft_reach']}, members {record['members']}, {seconds:.1f} s; DSATUR tight packing needs {needed(vc.face_contact(predicted.labels['fixed']) | vc.face_contact(predicted.labels['moving']))} colours on the face graph, {needed(cliques)} on the Chebyshev-1 graph; smallest code distance between labels with a common face: {vc.check_separated(predicted.labels['fixed'], colours):.3f} (fixed), {vc.check_separated(predicted.labels['moving'], colours):.3f} (moving)", flush=True)
    print(f'everything compared bit for bit: {all_equal}')
    return 0 if all_equal else 8


if __name__ == '__main__':
    sys.exit(main())
