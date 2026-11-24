"""Independent small-array proof checks; no native timing workload."""
from dataclasses import asdict
from itertools import product
import os
import re
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('numba')
from jd_curl.three_d import feasibility as f, compiled_enclosure as compiled
from jd_curl.three_d.interval import ArithmeticUncertain
from jd_curl.three_d.work_control import work_session, WorkBudgetExceeded
from jd_curl.validation import InputValidationError
from .test_interval_certificate import VERTICES, exact_jacobian, contains, counterexample
from .test_enclosure_e11 import grid


def normalized(proof):
    result = asdict(proof)
    result.pop('config', None); result.pop('certificate_version', None)
    return result


@pytest.mark.parametrize('seed', [0, 7, 89])
@pytest.mark.parametrize('scale', [2.**-500, 2.**-40, 1., 2.**40, 2.**500])
def test_bitwise_endpoints_and_independent_rational_inclusion(seed, scale):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(9, 8, 3))*.17*scale
    spacing = np.array([.2, .3, .7])*scale
    old = f.jacobian_enclosure(u, spacing)
    new = f.compiled_jacobian_enclosure(u, spacing)
    for a, b in ((old.lo, new.lo), (old.hi, new.hi)):
        np.testing.assert_array_equal(a.view('u8'), b.view('u8'))
    for point in list(product((0., 1.), repeat=3))+list(rng.uniform(size=(3, 3))):
        contains(new[0], exact_jacobian(u[0], spacing, point))


@pytest.mark.parametrize('value', [0., -0., np.nextafter(0., 1.), -np.nextafter(0., 1.), np.finfo(float).tiny])
def test_zero_and_subnormal_no_identity_shortcuts(value):
    u = np.full((2, 8, 3), value)
    u[1, 1::2] *= -1
    for spacing in ((1., 1., 1.), (.2, .3, .7)):
        old, new = f.jacobian_enclosure(u, spacing), f.compiled_jacobian_enclosure(u, spacing)
        np.testing.assert_array_equal(old.lo.view('u8'), new.lo.view('u8'))
        np.testing.assert_array_equal(old.hi.view('u8'), new.hi.view('u8'))


@pytest.mark.parametrize('delta', [0., .01])
@pytest.mark.parametrize('jd', [0., -.1, 2.**-30, .005, .5, 1.])
def test_strict_threshold_no_clipping(delta, jd):
    u = VERTICES*np.array([jd-1., 0., 0.])
    a = f.certify_cell(u, (1., 1., 1.), config=f.EnclosureCertificateConfig(delta=delta))
    b = f.certify_cell(u, (1., 1., 1.), config=f.CompiledEnclosureCertificateConfig(delta=delta))
    assert a == b
    assert (b.status == 'certified_feasible') == (jd > delta)


@pytest.mark.parametrize('depth,boxes', [(0, 255), (48, 1), (48, 255)])
def test_hull_unproven_keeps_original_refinement_and_budget(depth, boxes):
    for v in (counterexample(.1), counterexample(-.1)):
        options = dict(max_depth=depth, max_boxes=boxes)
        a = f.certify_cell(v, (1, 1, 1), config=f.EnclosureCertificateConfig(**options))
        b = f.certify_cell(v, (1, 1, 1), config=f.CompiledEnclosureCertificateConfig(**options))
        assert a == b


def test_all_27_samples_positive_but_interior_negative():
    x, y, z = VERTICES.T
    u = np.stack((x, y+2*(x-.25)*z, -2*(x-.25)*y-.0625*z), axis=-1)-VERTICES
    assert min(exact_jacobian(u, (1,1,1), p) for p in product((0., .5, 1.), repeat=3)) > 0
    out = f.certify_cell(u, (1,1,1), config=f.CompiledEnclosureCertificateConfig(delta=0))
    assert out.status == 'infeasible'
    assert exact_jacobian(u, (1,1,1), out.witness_reference_point) <= 0


@pytest.mark.parametrize('tile', [1, 7, 8192])
@pytest.mark.parametrize('mode', ['identity', 'smooth', 'fold', 'nan', 'boundary', 'domain'])
def test_whole_map_same_proof_coverage_witness_and_guards(tile, mode):
    g = grid(); u = np.zeros(g.shape+(3,))
    if mode == 'smooth': u[1:-1, 1:-1, 1:-1] = [.007, -.004, .005]
    elif mode == 'fold': u[1, 2, 3, 0] = -.35
    elif mode == 'nan': u[1, 1, 1, 0] = np.nan
    elif mode == 'boundary': u[0, 0, 0, 0] = .1
    elif mode == 'domain': u[1, 1, 1, 0] = 100.
    cfg = f.EnclosureCertificateConfig(delta=0, tile_size=tile)
    a = f.certify_map(u, g, config=cfg)
    b = f.certify_map(u, g, config=f.CompiledEnclosureCertificateConfig(**asdict(cfg)))
    assert normalized(a) == normalized(b)
    assert b.certificate_version == f.certificate_version(b.config) != a.certificate_version
    if mode in ('identity', 'smooth', 'fold'):
        assert b.n_fast_cells+b.n_fallback_cells == b.n_cells_checked == g.n_cells


@pytest.mark.parametrize('kind', ['subtract', 'divide', 'determinant'])
def test_overflow_never_returns_a_finite_enclosure(kind):
    u = np.zeros((2, 8, 3)); spacing = np.ones(3)
    if kind == 'subtract': u[:, ::2] = -1e308; u[:, 1::2] = 1e308
    elif kind == 'divide': u[:] = VERTICES; spacing[:] = np.nextafter(0., 1.)
    else: u[:] = VERTICES*1e200
    for fn in (f.jacobian_enclosure, f.compiled_jacobian_enclosure):
        with pytest.raises(ArithmeticUncertain): fn(u, spacing)


def test_prepass_error_falls_back_full_tile_and_runtime_unsupported(monkeypatch):
    g = grid(); u = np.zeros(g.shape+(3,)); cfg = f.CompiledEnclosureCertificateConfig(delta=0)
    monkeypatch.setattr(compiled, 'arithmetic_supported', lambda x: False)
    got = f.certify_map(u, g, config=cfg)
    assert got.status == 'certified_feasible' and got.n_fast_cells == 0
    assert got.n_fallback_cells == g.n_cells and got.sampled_min_jd == 1.
    monkeypatch.setattr(f.iv, 'arithmetic_supported', lambda: False)
    assert f.certify_map(u, g, config=cfg).status == 'uncertain'


def test_observer_uses_full_path_and_deadline_cannot_return_partial():
    observed = []
    cfg = f.CompiledEnclosureCertificateConfig(delta=0, tile_size=7)
    out = f.certify_cell(np.zeros((8,3)), (1,1,1), config=cfg,
                         point_observer=lambda *args: observed.append(args))
    assert len(observed) == 8 and out.sampled_min_jd == 1.
    calls = [0]
    def stop(): calls[0] += 1; return calls[0] > 1
    with work_session(stop), pytest.raises(WorkBudgetExceeded):
        f.certify_map(np.zeros(grid().shape+(3,)), grid(), config=cfg)


def test_llvm_no_fastmath_contraction_parallel_or_object_mode():
    assert compiled.arithmetic_supported(np.float64(0.))
    f.compiled_jacobian_enclosure(np.zeros((1,8,3)), (1,1,1))
    assert compiled._bounds.nopython_signatures
    for ir in compiled._bounds.inspect_llvm().values():
        assert not re.search(r'\b(fadd|fsub|fmul|fdiv)\s+(fast|reassoc|nnan|ninf|nsz|arcp|contract|afn)\b', ir)
        assert 'llvm.fma.' not in ir and 'llvm.fmuladd.' not in ir
    assert compiled.runtime_info()['fastmath'] is False
    assert compiled.runtime_info()['parallel'] is False
    assert compiled.runtime_info()['disk_cache'] is False


def test_legacy_schema_and_new_config_validation():
    assert 'kernel' not in asdict(f.EnclosureCertificateConfig())
    with pytest.raises(InputValidationError): f.CompiledEnclosureCertificateConfig(kernel='fastmath')
    for u, h in ((np.zeros((8,3)), (1,1,1)), (np.zeros((1,8,3)), (1,0,1)),
                 (np.full((1,8,3), np.nan), (1,1,1))):
        with pytest.raises(InputValidationError): f.compiled_jacobian_enclosure(u, h)


@pytest.mark.parametrize('readonly',[False,True])
def test_noncontiguous_readonly_input_keeps_ownership_and_bits(readonly):
    storage=np.random.default_rng(96).normal(scale=.1,size=(7,16,3))
    u=storage[:,::2,:];u.flags.writeable=not readonly
    snapshot=u.copy()
    old=f.jacobian_enclosure(u,(.2,.3,.7));new=f.compiled_jacobian_enclosure(u,(.2,.3,.7))
    np.testing.assert_array_equal(old.lo.view('u8'),new.lo.view('u8'))
    np.testing.assert_array_equal(old.hi.view('u8'),new.hi.view('u8'))
    np.testing.assert_array_equal(u,snapshot)
    assert u.flags.writeable is not readonly


@pytest.mark.parametrize('mode',['dependency_missing','jit_disabled'])
def test_optional_import_failure_never_silently_labels_python_as_compiled(mode):
    script = '''
import builtins,sys,numpy as np
from jd_curl.three_d import feasibility as f
from jd_curl.validation import InputValidationError
if sys.argv[1]=='dependency_missing':
    original=builtins.__import__
    def block(name,*args,**kwargs):
        if name=='numba':raise ImportError('injected unavailable dependency')
        return original(name,*args,**kwargs)
    builtins.__import__=block
f.certify_cell(np.zeros((8,3)),(1,1,1),config=f.EnclosureCertificateConfig())
assert 'numba' not in sys.modules
try:f.compiled_jacobian_enclosure(np.zeros((1,8,3)),(1,1,1))
except InputValidationError as error:assert 'numba' in str(error).lower()
else:raise AssertionError('silently accepted unavailable compiled path')
'''
    env=dict(os.environ,NUMBA_DISABLE_JIT='1' if mode=='jit_disabled' else '0')
    subprocess.run([sys.executable,'-B','-c',script,mode],env=env,check=True,capture_output=True,text=True,timeout=30)
