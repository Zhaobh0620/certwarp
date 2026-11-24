"""T46 independent extrema/rounding checks; small arrays, not speed claims."""
from dataclasses import asdict
from itertools import product
import re

import numpy as np
import pytest
pytest.importorskip('numba')

from jd_curl.three_d import feasibility as f, compiled_enclosure as c
from jd_curl.three_d.interval import ArithmeticUncertain, Interval
from jd_curl.three_d.work_control import WorkBudgetExceeded, work_session
from jd_curl.validation import InputValidationError
from .test_interval_certificate import VERTICES, exact_jacobian, contains, counterexample
from .test_enclosure_e11 import grid
from .test_compiled_certificate_e32 import normalized


def config(**kw):
    return f.CompiledEnclosureCertificateConfig(kernel=c.HOISTED_KERNEL, **kw)


def hoisted(u, h):
    return f.compiled_jacobian_enclosure(u, h, kernel=c.HOISTED_KERNEL)


def bits_equal(a, b):
    np.testing.assert_array_equal(np.asarray(a).view('u8'), np.asarray(b).view('u8'))


@pytest.mark.parametrize('seed', [0, 7, 89])
@pytest.mark.parametrize('exponent', [-1000, -500, -40, 0, 40, 500, 1000])
def test_endpoints_equal_both_old_kernels_and_fraction_inclusion(seed, exponent):
    rng = np.random.default_rng(seed)
    scale = 2.**exponent
    u = rng.normal(scale=.17, size=(9, 8, 3))*scale
    h = np.array([.2, .3, .7])*scale
    out = hoisted(u, h)
    for old in (f.jacobian_enclosure(u, h), f.compiled_jacobian_enclosure(u, h)):
        bits_equal(out.lo, old.lo); bits_equal(out.hi, old.hi)
    for p in list(product((0., 1.), repeat=3))+list(rng.uniform(size=(3, 3))):
        contains(out[0], exact_jacobian(u[0], h, p))


@pytest.mark.parametrize('j', [0, 1, 2])
@pytest.mark.parametrize('h', [2.**-1000, .3, 1., 2.**1000])
def test_raw_extrema_match_original_four_intervals_including_overflow(j, h):
    # Explicit independent Interval oracle, not the new determinant or hull.
    rng = np.random.default_rng(181+j)
    raw = rng.integers(0, np.iinfo('u8').max, size=(64, 8), dtype='u8').view('f8')
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    tiny = np.nextafter(0., 1.)
    special = np.array([[0., -0., tiny, -tiny, 1., -1., .5, -.5],
                        [np.finfo(float).max, 0., 1., -1., 0., 1., 2., 3.],
                        [tiny, 0., -tiny, -0., np.finfo(float).tiny, 0., -np.finfo(float).tiny, 0.]])
    for row in np.concatenate((raw, special)):
        u = np.zeros((8, 3)); u[:, 0] = row
        try:
            intervals = [(Interval.point(u[a | (1 << j), 0])-Interval.point(u[a, 0]))/h
                         for a in range(8) if not a & (1 << j)]
        except ArithmeticUncertain:
            with pytest.raises(ValueError): c._edge_hull(u, j, 0, h)
        else:
            got = c._edge_hull(u, j, 0, h)
            expected = (min(float(x.lo) for x in intervals), max(float(x.hi) for x in intervals))
            bits_equal(got, expected)


@pytest.mark.parametrize('jd', [-.1, 0., 2.**-30, .005, .01, .5, 1.])
@pytest.mark.parametrize('delta', [0., .01])
def test_strict_jd_threshold_matches_old_certificate(jd, delta):
    u = VERTICES*np.array([jd-1., 0., 0.])
    assert f.certify_cell(u, (1,1,1), config=config(delta=delta)) == f.certify_cell(
        u, (1,1,1), config=f.EnclosureCertificateConfig(delta=delta))


@pytest.mark.parametrize('tile', [1, 7, 8192])
@pytest.mark.parametrize('mode', ['identity', 'smooth', 'fold', 'nan', 'boundary', 'domain'])
def test_map_guards_full_coverage_and_distinct_policy_identity(tile, mode):
    g = grid(); u = np.zeros(g.shape+(3,))
    if mode == 'smooth': u[1:-1, 1:-1, 1:-1] = [.007, -.004, .005]
    elif mode == 'fold': u[1, 2, 3, 0] = -.35
    elif mode == 'nan': u[1, 1, 1, 0] = np.nan
    elif mode == 'boundary': u[0, 0, 0, 0] = .1
    elif mode == 'domain': u[1, 1, 1, 0] = 100.
    cfg = f.EnclosureCertificateConfig(delta=0, tile_size=tile)
    a = f.certify_map(u, g, config=cfg)
    b = f.certify_map(u, g, config=config(**asdict(cfg)))
    assert normalized(a) == normalized(b)
    assert b.certificate_version == 'q1-3d-hoisted-edge-hull-bernstein-v1'
    if mode in ('identity', 'smooth', 'fold'):
        assert b.n_fast_cells+b.n_fallback_cells == b.n_cells_checked == g.n_cells


@pytest.mark.parametrize('depth,boxes', [(0, 255), (48, 1), (48, 255)])
def test_refinement_and_budget_not_replaced_by_samples(depth, boxes):
    x, y, z = VERTICES.T
    hidden = np.stack((x, y+2*(x-.25)*z, -2*(x-.25)*y-.0625*z), axis=-1)-VERTICES
    assert min(exact_jacobian(hidden, (1,1,1), p) for p in product((0., .5, 1.), repeat=3)) > 0
    opts = dict(delta=0, max_depth=depth, max_boxes=boxes)
    for u in (hidden, counterexample(.1), counterexample(-.1)):
        assert f.certify_cell(u, (1,1,1), config=config(**opts)) == f.certify_cell(
            u, (1,1,1), config=f.EnclosureCertificateConfig(**opts))


@pytest.mark.parametrize('mode', ['subtract', 'divide', 'determinant', 'outward'])
def test_overflow_at_every_stage_still_uncertain(mode):
    u = np.zeros((2, 8, 3)); h = np.ones(3)
    if mode == 'subtract': u[:, ::2] = -1e308; u[:, 1::2] = 1e308
    elif mode == 'divide': u[:] = VERTICES; h[:] = np.nextafter(0., 1.)
    elif mode == 'determinant': u[:] = VERTICES*1e200
    else: u[1, 7, 0] = np.finfo(float).max
    for fn in (f.jacobian_enclosure, f.compiled_jacobian_enclosure, hoisted):
        with pytest.raises(ArithmeticUncertain): fn(u, h)


def test_entire_tile_fallback_on_error_and_unsupported_arithmetic(monkeypatch):
    g = grid(); u = np.zeros(g.shape+(3,))
    def fail(*args): raise ValueError('injected arithmetic uncertainty')
    monkeypatch.setattr(c, '_hoisted_bounds', fail)
    out = f.certify_map(u, g, config=config(delta=0))
    assert out.status == 'certified_feasible' and out.n_fast_cells == 0
    assert out.n_fallback_cells == g.n_cells
    monkeypatch.setattr(c, 'arithmetic_supported', lambda x: False)
    assert f.certify_map(u, g, config=config(delta=0)).n_fallback_cells == g.n_cells
    monkeypatch.setattr(f.iv, 'arithmetic_supported', lambda: False)
    assert f.certify_map(u, g, config=config(delta=0)).status == 'uncertain'


def test_deadline_and_observer_keep_original_contract():
    seen=[]
    out = f.certify_cell(np.zeros((8,3)), (1,1,1), config=config(delta=0),
                         point_observer=lambda *a: seen.append(a))
    assert len(seen) == 8 and out.sampled_min_jd == 1.
    with work_session(lambda: True), pytest.raises(WorkBudgetExceeded):
        f.certify_map(np.zeros(grid().shape+(3,)), grid(), config=config(delta=0, tile_size=7))


@pytest.mark.parametrize('readonly', [False, True])
def test_strided_inputs_not_mutated(readonly):
    u = np.random.default_rng(96).normal(scale=.1, size=(7, 16, 3))[:, ::2, :]
    before = u.copy(); u.flags.writeable = not readonly
    a, b = f.jacobian_enclosure(u, (.2,.3,.7)), hoisted(u, (.2,.3,.7))
    bits_equal(a.lo, b.lo); bits_equal(a.hi, b.hi); bits_equal(u, before)
    assert u.flags.writeable is not readonly


def test_bad_config_inputs_and_no_fastmath_ir():
    for bad in ('fastmath', '', None):
        with pytest.raises(InputValidationError): c.runtime_info(bad)
        with pytest.raises(InputValidationError): c.enclosure(np.zeros((1,8,3)), (1,1,1), kernel=bad)
    for u,h in ((np.zeros((8,3)),(1,1,1)), (np.zeros((1,8,3)),(1,0,1)),
                (np.full((1,8,3),np.nan),(1,1,1))):
        with pytest.raises(InputValidationError): hoisted(u,h)
    hoisted(np.zeros((1,8,3)), (1,1,1))
    assert c._hoisted_bounds.nopython_signatures
    for ir in c._hoisted_bounds.inspect_llvm().values():
        assert not re.search(r'\b(fadd|fsub|fmul|fdiv)\s+(fast|reassoc|nnan|ninf|nsz|arcp|contract|afn)\b', ir)
        assert 'llvm.fma.' not in ir and 'llvm.fmuladd.' not in ir
    info=c.runtime_info(c.HOISTED_KERNEL)
    assert not info['fastmath'] and not info['parallel'] and not info['disk_cache']
