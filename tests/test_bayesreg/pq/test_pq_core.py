"""PQ (contract C22.Q7 "PQ 单元"; plan.md sections 2, 3.2): experiments/pq/pq_core.py on small synthetic volumes (CPU, no file).

- the nearest neighbour: argmax P~_M (outside: background) bit for bit warp_labels_l2r(argmax P_M, d), with d covering points outside, coordinates within +-.5 of each face and exactly on .5;
- the one-hot case: q = 1[L_F != L~_M], D^_c = H_c, S^35 = H35; the identity field: P~_M = P_M;
- a hand example of three voxels and three classes: D^, the defined set, the missing rule, positive denominators;
- seg4: sampling then summing by group is bit for bit summing then sampling;
- eps-EM: the log-likelihood never decreases, the floor and the ceiling, 1 - gamma ordered as q;
- the baselines: the boundary and its distance on a hand example (6 faces, label 0 a label, mm, the two sides' minimum), ||grad F|| with the array-axis spacings, the entropy.
"""
import numpy as np
import pytest

from . import _world as W

core = W.core
io_ = W.io_


def random_posterior(rng, shape, alpha=.4):
    p = rng.gamma(alpha, 1., (core.K,)+shape)
    return p/p.sum(axis=0)


def edge_displacement(shape, rng):
    """(3, nz, ny, nx): random, plus points outside, within +-.5 of the faces and exactly on .5 (C22.Q7, first item)."""
    d = rng.uniform(-2.5, 2.5, (3,)+shape)
    flat = d.reshape(3, -1)
    idx = np.indices(shape).reshape(3, -1)
    for axis in range(3):
        n = shape[axis]
        specials = [-.5, -.49, -.51, -.5000001, -.4999999, n-.5, n-.51, n-.49, n-1.5, n-1+.5, 1.5, 2.5, .5, n+3, -4.]     # the faces at -.5 and n - .5 of the voxel centres 0 .. n-1
        for k, target in enumerate(specials):
            j = (5*k+11*axis) % flat.shape[1]
            flat[axis, j] = target-idx[axis, j]
    return d


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_nearest_neighbour_is_the_evaluations_warp(seed):
    rng = np.random.default_rng(seed)
    shape = (5, 6, 4)
    P_M = random_posterior(rng, shape)
    P_F = random_posterior(rng, shape)
    L_M = core.argmax_smallest(P_M)
    d = edge_displacement(shape, rng)
    iota = core.sample_index(d)
    assert (iota < 0).any() and (iota >= 0).any()
    warped = core.warp_labels_l2r(L_M, d)
    maps = core.pq_maps(P_F, P_M, iota, W.LOOKUP)
    assert maps['top_tilde'].dtype == warped.dtype == np.int16 and maps['top_tilde'].tobytes() == warped.tobytes()
    assert io_.nearest_top(P_M, iota).tobytes() == warped.tobytes()
    assert np.all(warped[iota < 0] == 0)                                             # outside: the evaluation's label 0, our background
    for c in (0, 3, 17):                                                             # each channel through the evaluation's own call
        channel = core.take(P_M[c], iota, 1. if c == 0 else 0.)
        theirs = core.warp_labels_l2r(P_M[c], d)
        inside = iota >= 0
        assert channel[inside].tobytes() == theirs[inside].tobytes()


def test_argmax_is_numpys_first_maximum_bit_for_bit():
    rng = np.random.default_rng(11)
    P = np.round(rng.uniform(size=(core.K, 6, 7, 5)), 1)                            # many ties
    P[:, 0, 0, 0] = .5                                                               # all equal: label 0
    L = core.argmax_smallest(P)
    assert L.dtype == np.int16 and L.tobytes() == np.argmax(P, axis=0).astype(np.int16).tobytes() and L[0, 0, 0] == 0


def test_sample_index_identity_and_outside():
    shape = (3, 4, 5)
    iota = core.sample_index(np.zeros((3,)+shape))
    assert np.array_equal(iota.ravel(), np.arange(np.prod(shape)))
    d = np.zeros((3,)+shape)
    d[0, 0, 0, 0] = -1.                                                              # z = -1: outside
    d[2, 1, 1, 4] = 1.                                                               # x = 5 of 5: outside
    iota = core.sample_index(d)
    assert iota[0, 0, 0] == -1 and iota[1, 1, 4] == -1 and (iota >= 0).sum() == np.prod(shape)-2


def test_one_hot_posteriors_give_the_hard_version():
    rng = np.random.default_rng(3)
    shape = (6, 5, 7)
    L_F = rng.integers(0, 36, shape).astype(np.int16)
    L_M = rng.integers(0, 36, shape).astype(np.int16)
    onehot = lambda L: (np.arange(core.K)[:, None, None, None] == L[None]).astype(np.float64)     # noqa: E731
    d = rng.uniform(-1.5, 1.5, (3,)+shape)
    iota = core.sample_index(d)
    maps = core.pq_maps(onehot(L_F), onehot(L_M), iota, W.LOOKUP)
    L_tilde = core.warp_labels_l2r(L_M, d)
    assert maps['q'].tobytes() == (L_F != L_tilde).astype(np.float64).tobytes()
    numbers = core.pq_numbers(maps, L_F, L_tilde, L_M, W.LOOKUP)
    for c in numbers['defined']:
        assert abs(numbers['D_hat'][c]-numbers['H_c'][c]) <= 1e-12*max(1., abs(numbers['H_c'][c]))
    assert abs(numbers['S35_hat']-numbers['H35']) <= 1e-12*numbers['H35']
    assert abs(numbers['S4_hat']-numbers['H4']) <= 1e-12*numbers['H4']
    assert sorted(numbers['defined']) == sorted(c for c in range(1, 36) if numbers['H_c'][c] is not None)


def test_identity_field_keeps_the_moving_posterior():
    rng = np.random.default_rng(4)
    shape = (4, 5, 6)
    P_M = random_posterior(rng, shape)
    iota = core.sample_index(np.zeros((3,)+shape))
    for c in range(core.K):
        assert core.take(P_M[c], iota, 1. if c == 0 else 0.).tobytes() == P_M[c].tobytes()


def test_hand_example_three_voxels_three_classes():
    """Three voxels in a row, the mass on labels 0, 1, 2 only; d = 0.  Hand: S^x_1 = .6*.5 + .7*.2 + 0*.1 = .44, S^F_1 = 1.3, S^M_1 = .8, D^_1 = .88 / 2.1;
    label 2: S^x = .1*.1 + .1*.2 + .2*.3 = .09, S^F = .4, S^M = .6, D^_2 = .18; argmax: L_F = (1, 1, 0), L_M = (1, 0, 0), so label 2 is in neither map: not defined though it has mass."""
    shape = (1, 1, 3)
    P_F = np.zeros((core.K,)+shape)
    P_M = np.zeros((core.K,)+shape)
    P_F[:3, 0, 0, :] = np.array([[.3, .2, .8], [.6, .7, .0], [.1, .1, .2]])
    P_M[:3, 0, 0, :] = np.array([[.4, .6, .6], [.5, .2, .1], [.1, .2, .3]])
    iota = core.sample_index(np.zeros((3,)+shape))
    maps = core.pq_maps(P_F, P_M, iota, W.LOOKUP)
    L_F, L_M = core.argmax_smallest(P_F), core.argmax_smallest(P_M)
    assert L_F.ravel().tolist() == [1, 1, 0] and L_M.ravel().tolist() == [1, 0, 0]
    numbers = core.pq_numbers(maps, L_F, L_M, L_M, W.LOOKUP)
    sx1 = .6*.5+.7*.2+0*.1
    assert numbers['Sx'][1] == pytest.approx(sx1, abs=1e-15) and numbers['SF'][1] == pytest.approx(1.3) and numbers['SM'][1] == pytest.approx(.8)
    assert numbers['D_hat'][1] == pytest.approx(2*sx1/2.1, rel=1e-14)
    assert numbers['D_hat'][2] == pytest.approx(2*(.1*.1+.1*.2+.2*.3)/(.4+.6), rel=1e-14)
    assert numbers['defined'] == [1] and numbers['S35_hat'] == pytest.approx(numbers['D_hat'][1], rel=1e-15)
    q = 1-(P_F[:3]*P_M[:3]).sum(axis=0)
    assert np.allclose(maps['q'], q, rtol=0, atol=1e-15)
    assert all(numbers['SF'][c]+numbers['SM'][c] > 0 for c in numbers['defined'])
    assert numbers['defined_official'] == [1]                                        # label 1 in L_F and in the unwarped L_M
    for c in range(3, 36):
        assert np.isnan(numbers['D_hat'][c])                                         # no mass: no ratio (it is not used)


def test_seg4_sampling_then_grouping_equals_grouping_then_sampling():
    rng = np.random.default_rng(5)
    shape = (5, 4, 6)
    P_F, P_M = random_posterior(rng, shape), random_posterior(rng, shape)
    d = edge_displacement(shape, rng)
    iota = core.sample_index(d)
    maps = core.pq_maps(P_F, P_M, iota, W.LOOKUP)
    F4, M4 = np.zeros((5,)+shape), np.zeros((5,)+shape)
    for c in range(core.K):                                                          # group first (the same additions in the same order) ...
        F4[W.LOOKUP[c]] += P_F[c]
        M4[W.LOOKUP[c]] += P_M[c]
    M4s = np.stack([core.take(M4[g], iota, 1. if g == 0 else 0.) for g in range(5)])     # ... then sample
    q4 = np.zeros(shape)
    for g in range(5):
        q4 += F4[g]*M4s[g]
    assert maps['q4'].tobytes() == (1.-q4).tobytes()
    assert maps['SM4'].tobytes() == M4s.reshape(5, -1).sum(axis=1).tobytes()


def test_eps_em_never_decreases_and_its_floor_and_ceiling():
    rng = np.random.default_rng(6)
    s = np.concatenate([rng.beta(20, 1, 4500), rng.uniform(0, .02, 500)])            # about a tenth of the voxels disagree: an interior maximum
    out = core.eps_em(s)
    ll = np.array(out['log_likelihood'])
    assert np.all(np.diff(ll) >= -1e-9*np.abs(ll[:-1])) and not out['at_floor'] and not out['at_ceiling'] and core.EPS_FLOOR < out['eps'] < core.EPS_CEILING
    floor = core.eps_em(np.ones(1000))                                               # every voxel agrees: eps goes to 0, cut at the floor
    assert floor['at_floor'] and floor['eps'] == core.EPS_FLOOR and np.all(np.diff(floor['log_likelihood']) >= 0)
    ceiling = core.eps_em(np.full(1000, 1e-4))                                       # every voxel disagrees: eps goes to 1, cut at the ceiling
    assert ceiling['at_ceiling'] and ceiling['eps'] == core.EPS_CEILING and np.all(np.diff(ceiling['log_likelihood']) >= 0)


def test_eps_em_symbolic_concavity_fixed_point_and_order():
    """verify-math procedure for C22.Q2-8 (sympy, independent of the code): l''(eps) = -sum (s - 1/K)^2 / ((1 - eps) s + eps/K)^2 <= 0 (concave); the EM fixed point eps = 1 - mean gamma is a zero of l'(eps);
    d(1 - gamma)/ds < 0 for 0 < eps < 1, so 1 - gamma increases with q = 1 - s.  Then the code's EM against the root of l' found by bisection on random s."""
    sympy = pytest.importorskip('sympy')
    eps, s, K = sympy.symbols('epsilon s K', positive=True)
    term = sympy.log((1-eps)*s+eps/K)
    second = sympy.simplify(sympy.diff(term, eps, 2))
    assert sympy.simplify(second+(s-1/K)**2/((1-eps)*s+eps/K)**2) == 0
    gamma = (1-eps)*s/((1-eps)*s+eps/K)
    first = sympy.diff(term, eps)
    assert sympy.simplify(first-((1-gamma)/eps-gamma/(1-eps))) == 0               # so sum l' = N [(1 - mean gamma)/eps - mean gamma/(1 - eps)] = 0 iff mean gamma = 1 - eps
    slope = sympy.simplify(sympy.diff(1-gamma, s))
    assert sympy.simplify(slope+(eps/K)*(1-eps)/((1-eps)*s+eps/K)**2) == 0
    rng = np.random.default_rng(12)
    values = np.concatenate([rng.beta(30, 1, 3000), rng.uniform(0, .002, 300)])        # a tenth of the voxels well below 1/K: an interior root
    out = core.eps_em(values, tol=1e-14, max_iter=5000)
    derivative = lambda e: float(np.sum((1/core.K-values)/((1-e)*values+e/core.K)))      # noqa: E731
    lo, hi = core.EPS_FLOOR, core.EPS_CEILING
    assert derivative(lo) > 0 > derivative(hi)
    for _ in range(200):
        mid = .5*(lo+hi)
        lo, hi = (mid, hi) if derivative(mid) > 0 else (lo, mid)
    assert out['eps'] == pytest.approx(.5*(lo+hi), rel=1e-6)


def test_one_minus_gamma_orders_the_voxels_as_q():
    rng = np.random.default_rng(7)
    q = np.round(rng.uniform(0, 1, 2000), 3)
    for eps in (1e-4, .05, .5):
        r = core.em_responsibility_complement(q, eps)
        order_q, order_r = np.argsort(q, kind='stable'), np.argsort(r, kind='stable')
        assert np.array_equal(order_q, order_r)
        assert np.array_equal(np.unique(q, return_inverse=True)[1], np.unique(r, return_inverse=True)[1])


def test_boundary_and_distance_hand_example():
    L = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int16).reshape(1, 1, 7)
    edge = core.boundary_voxels(L)
    assert edge.ravel().tolist() == [False, True, True, False, True, True, False]       # label 0 is a label; the array's ends do not count
    delta = core.boundary_distance(L, (2., 1., 1.))                                 # x spacing 2 mm
    assert delta.ravel().tolist() == [2., 0., 0., 2., 0., 0., 2.]
    other = np.array([3, 3, 3, 3, 3, 3, 4], dtype=np.int16).reshape(1, 1, 7)
    both = np.minimum(core.boundary_distance(L, (2., 1., 1.)), core.boundary_distance(other, (2., 1., 1.)))
    assert both.ravel().tolist() == [2., 0., 0., 2., 0., 0., 0.]
    L3 = np.zeros((3, 3, 3), dtype=np.int16)
    L3[1, 1, 1] = 5
    edge3 = core.boundary_voxels(L3)
    assert edge3.sum() == 7 and edge3[1, 1, 1] and edge3[0, 1, 1] and not edge3[0, 0, 0]     # the centre and its 6 face neighbours; no corner
    d3 = core.boundary_distance(L3, (1., 2., 3.))                                    # spacing x 1, y 2, z 3 mm: (s_z, s_y, s_x) = (3, 2, 1)
    assert d3[0, 0, 1] == pytest.approx(2.) and d3[1, 0, 0] == pytest.approx(1.) and d3[0, 0, 0] == pytest.approx(np.hypot(2., 1.))
    with pytest.raises(ValueError):
        core.boundary_distance(np.zeros((2, 2, 2)), (1., 1., 1.))


def test_gradient_norm_uses_the_array_axis_spacings():
    sx, sy, sz = .5, 1.5, 2.
    z, y, x = np.meshgrid(np.arange(5)*sz, np.arange(6)*sy, np.arange(7)*sx, indexing='ij')
    F = 3.*z+2.*y+5.*x
    G = core.gradient_norm(F, (sx, sy, sz))
    assert np.allclose(G, np.sqrt(38.), rtol=0, atol=1e-12)
    assert not np.allclose(core.gradient_norm(F, (sz, sy, sx)), np.sqrt(38.))           # the other order would be wrong


def test_entropy_and_tail_mass():
    P = np.zeros((core.K, 1, 1, 3))
    P[:, 0, 0, 0] = 1./core.K
    P[0, 0, 0, 1] = 1.
    P[[1, 2], 0, 0, 2] = .5
    H = core.entropy(P)
    assert H[0, 0, 0] == pytest.approx(np.log(core.K)) and H[0, 0, 1] == 0. and H[0, 0, 2] == pytest.approx(np.log(2.))
    L = core.argmax_smallest(P)
    assert L.ravel().tolist() == [0, 0, 1]
    tail = core.tail_masses(P, L)
    assert tail[2] == pytest.approx(1./core.K+.5) and tail[0] == pytest.approx(0.) and tail[1] == pytest.approx(1./core.K)


def test_baseline_maps_dtypes_and_definitions():
    rng = np.random.default_rng(8)
    shape = (5, 6, 7)
    P_F, P_M = random_posterior(rng, shape), random_posterior(rng, shape)
    L_F, L_M = core.argmax_smallest(P_F), core.argmax_smallest(P_M)
    d = rng.uniform(-1, 1, (3,)+shape)
    iota = core.sample_index(d)
    L_tilde = core.warp_labels_l2r(L_M, d)
    H_F, H_M = core.entropy(P_F), core.entropy(P_M)
    V_F, V_M = rng.uniform(0, .1, shape).astype(np.float32), rng.uniform(0, .1, shape).astype(np.float32)
    F = rng.uniform(size=shape)
    base = core.baseline_maps(L_F=L_F, L_tilde=L_tilde, H_F=H_F, H_M=H_M, V_F=V_F, V_M=V_M, F=F, iota=iota, spacing_xyz=(1., 1., 1.))
    assert base['h'].dtype == np.uint8 and all(base[k].dtype == np.float32 for k in core.BASELINE_KEYS if k != 'h')
    assert np.array_equal(base['h'], (L_F != L_tilde).astype(np.uint8))
    H2 = H_F+np.where(iota >= 0, H_M.ravel()[np.maximum(iota, 0)], 0.)
    assert base['H2'].tobytes() == H2.astype(np.float32).tobytes()
    V2 = V_F.astype(np.float64)+np.where(iota >= 0, V_M.astype(np.float64).ravel()[np.maximum(iota, 0)], 0.)
    assert base['V2'].tobytes() == V2.astype(np.float32).tobytes()
    assert base['G_minus'].tobytes() == (-base['G_plus']).tobytes()
