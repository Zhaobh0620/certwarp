"""E1 of EF: the K = 8 vector colouring of the 35 labels.  Each label gets one of K unit vectors, background the zero vector, so that two labels sharing a voxel face have orthogonal codes, as in the K = 35 one-hot (runs/20261104-ef/plan.md "K = 8 的选择").

    X_c = G_.75 * sum_{l: colour(l) = c} 1[L = l],   c = 1..K

Why a colouring and not a random orthogonal projection: at a boundary between two labels the code difference is e_a - e_b, exactly as with one-hot, so the local data force is the K = 35 one until two labels of one colour come within the reach of the blur or the displacement.  A projection of 35 onto 8 dimensions
has no such guarantee: ||P(e_l - e_m)|| varies pair by pair and can be near 0, and every channel holds every label's tail; it would need a seed and a rejection step.  A colouring needs neither.

The graph (measured on the predictions of the 5 pairs, runs/20261104-ef/e2_profile.out): the hard edges are the labels 1..35 that share a voxel face in the fixed OR the moving predicted label map: 7 or 8 colours are enough on all 5 pairs.  With diagonal contact too (Chebyshev distance 1) the union has cliques of 9 on two pairs: no 8 colours exist.
So diagonal and near contact are a preference only: among the colours a hard edge allows, the one with the fewest labels of that colour within SOFT_REACH voxels (the radius of G_.75), then the least used, then the smallest.  If that order runs out of colours: the tighter packing (smallest allowed colour).  Deterministic, no seed.  Neither an image nor a truth label enters.
"""
import numpy as np
from scipy.ndimage import find_objects, maximum_filter

N_LABELS = 36                                    # 0 = background, labels 1..35
K_DEFAULT = 8
SOFT_REACH = 3                                   # voxels: the kernel radius of G_.75


class ColouringFailed(Exception):
    """K colours do not keep the labels with a common face apart: the arm has no channels for this pair."""


def face_contact(L):
    """(36, 36) symmetric bool: labels l != m in 1..35 sharing a voxel face in L.  Background has no row: its code is the zero vector, at distance 1 from every label's."""
    L = np.asarray(L, dtype=np.intp)
    out = np.zeros((N_LABELS, N_LABELS), dtype=bool)
    for axis in range(3):
        a, b = np.moveaxis(L, axis, 0)[:-1], np.moveaxis(L, axis, 0)[1:]
        differ = a != b
        out[a[differ], b[differ]] = True
    out = out | out.T
    out[0], out[:, 0] = False, False
    return out


def contact_matrix(L, reach=1, workers=1):
    """(36, 36) symmetric bool: labels l != m in 1..35 with voxels at Chebyshev distance <= reach in L."""
    from concurrent.futures import ThreadPoolExecutor
    L = np.asarray(L)
    boxes = find_objects(L.astype(np.int32), max_label=N_LABELS-1)

    def one(label):
        box = boxes[label-1]
        if box is None:
            return label, None
        grown = tuple(slice(max(s.start-reach, 0), min(s.stop+reach, n)) for s, n in zip(box, L.shape))
        part = L[grown]
        near = maximum_filter((part == label).astype(np.uint8), size=2*reach+1, mode='constant', cval=0).astype(bool)
        return label, np.unique(part[near])
    labels = range(1, N_LABELS)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ef-contact') as pool:
            rows = list(pool.map(one, labels))
    else:
        rows = [one(label) for label in labels]
    out = np.zeros((N_LABELS, N_LABELS), dtype=bool)
    for label, others in rows:
        if others is not None:
            out[label, others[others > 0]] = True
    out[np.arange(N_LABELS), np.arange(N_LABELS)] = False
    return out | out.T


def dsatur(hard, K, *, soft=None):
    """{label: colour in 0..K-1}, or None if K colours are not enough in this order.  DSATUR: next the uncoloured label with the most colours among its hard neighbours, ties by degree, then the smallest label.
    soft given: among the allowed colours, the fewest soft neighbours of that colour, then the least used, then the smallest.  soft None: the smallest allowed colour (the tighter packing)."""
    labels = list(range(1, N_LABELS))
    degree = {l: int(hard[l].sum()) for l in labels}
    colours, counts = {}, [0]*K
    while len(colours) < len(labels):
        def seen(l):
            return {colours[m] for m in labels if hard[l, m] and m in colours}
        label = min((l for l in labels if l not in colours), key=lambda l: (-len(seen(l)), -degree[l], l))
        allowed = [c for c in range(K) if c not in seen(label)]
        if not allowed:
            return None
        if soft is None:
            colour = allowed[0]
        else:
            colour = min(allowed, key=lambda c: (sum(1 for m in labels if soft[label, m] and colours.get(m) == c), counts[c], c))
        colours[label] = colour
        counts[colour] += 1
    return colours


def choose_colouring(fixed, moving, *, K=K_DEFAULT, soft_reach=SOFT_REACH, workers=1):
    """(colours {label: colour}, record).  ColouringFailed if K colours do not fit the face graph of the two label maps in either order."""
    hard = face_contact(fixed) | face_contact(moving)
    soft = contact_matrix(fixed, soft_reach, workers) | contact_matrix(moving, soft_reach, workers)
    for mode, preference in (('soft preference', soft), ('tight packing', None)):
        colours = dsatur(hard, K, soft=preference)
        if colours is not None:
            check_separated(fixed, colours, K)
            check_separated(moving, colours, K)
            members = [[l for l, c in sorted(colours.items()) if c == k] for k in range(K)]
            near = sorted((l, m) for l in colours for m in colours if l < m and soft[l, m] and colours[l] == colours[m])
            return colours, dict(K=K, mode=mode, hard_edges=int(hard.sum())//2, soft_reach=soft_reach, soft_edges=int(soft.sum())//2, same_colour_within_soft_reach=[list(p) for p in near], colours={str(l): c for l, c in sorted(colours.items())}, members=members,
                                 rule='hard: a common voxel face in the fixed or moving predicted labels; DSATUR, deterministic; background: zero vector')
    raise ColouringFailed(f'{K} colours do not fit the face graph of the two label maps ({int(hard.sum())//2} edges)')


def codes(colours, K=K_DEFAULT):
    """(36, K): row l = the unit vector of colour(l); row 0, background, all zero."""
    out = np.zeros((N_LABELS, K))
    for label, colour in colours.items():
        out[label, colour] = 1.
    return out


def check_separated(L, colours, K=K_DEFAULT):
    """The smallest code distance between two labels with a common face in L.  STOP (ColouringFailed) if two such labels have the same code."""
    graph = face_contact(L)
    table = codes(colours, K)
    distance = np.sqrt(((table[:, None]-table[None])**2).sum(axis=-1))
    bad = np.argwhere(graph & (distance == 0))
    if len(bad):
        raise ColouringFailed(f'labels with a common face and the same code: {[tuple(map(int, b)) for b in bad if b[0] < b[1]]}')
    return float(distance[graph].min()) if graph.any() else None


def vector_channels(L, colours, K=K_DEFAULT, *, smooth, workers=1):
    """(K, nz, ny, nx) float64: G_.75 of each colour's mask.  smooth: make_inputs.smooth, the G_.75 of the arms."""
    from concurrent.futures import ThreadPoolExecutor
    lookup = np.full(N_LABELS, -1, dtype=np.int16)
    for label, colour in colours.items():
        lookup[label] = colour
    code = lookup[np.asarray(L, dtype=np.intp)]
    out = np.empty((K,)+tuple(np.shape(L)), dtype=np.float64)

    def fill(c):
        out[c] = smooth((code == c).astype(np.float64))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ef-colour') as pool:
            list(pool.map(fill, range(K)))
    else:
        for c in range(K):
            fill(c)
    return out
