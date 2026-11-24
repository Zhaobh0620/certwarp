"""P1 (runs/20261105-p1/任务书.md, plan.md): experiments/p1/{p1_core,run_p1}.py.  CPU only, synthetic small volumes in tmp_path; no real image, label,
prediction or field is read; nothing here starts a unit.

- control (ii): the numpy formulas against mc0_core.pre_energy_k (torch, CPU, float64) to 1e-12, and the check can fail;
- control (i): bit for bit against the records, and each way it fails;
- the channels: onehot_channel against fast_channels.onehot_channels bit for bit; the warp against grid_sample, beyond the faces too;
- the EM weights on the three domains against hand computations; beta and R^2; the group ratios;
- the cell terms slab by slab, the cell centre, the tissue moments, the block bootstrap;
- the reading lines of revision 1, every branch and boundary: d_A for A (beta only described), CSF/WM and GM/WM for C with both directions, control (iii)'s 5 %
  flag holding the readings; the small, unfolded case of control (ii); the moments inside the brain mask and on field_0600;
- the firewall: a member with seg in its name is refused, the spy sees only the images, no metrics file is read, the source;
- the start: the unit command (no 'pytest'), every refusal of start_problems (inside the unit, UF's chain state, the imported modules, the frozen tree,
  active or activating units), the dry run;
- a synthetic pipeline: compute_pair against independent computations, run_all to the summary, a stop after a failed control; once in a subprocess with the
  frozen tree's boundary_taper.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
import torch

from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.nifti import save_scalar

from bayesreg.seg import structure_layer

from ..io_seg._helpers import DEV, FOLD0, FOLD1, SYN_FOLD0, SYN_FOLD1, allowlist_document, is_synthetic, synthetic_fold_sources, synthetic_folds, write_json
from ..solve import test_d0_a4_tools as a4
from ..solve.test_run_pair_structure import registry_models, write_registry

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT/'experiments/p1/run_p1.py'
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
needs_frozen = pytest.mark.skipif(not (FROZEN/'jd_curl/three_d/preopt.py').is_file(), reason='the frozen worktree is not on this machine')
torch.set_num_threads(1)


def _load():
    spec = importlib.util.spec_from_file_location('_p1_run', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules['_p1_run'] = module
    spec.loader.exec_module(module)
    return module


RUN = _load()
p1, core, mc, fast, fx = RUN.p1, RUN.core, RUN.mc, RUN.fast, RUN.fx
SHAPE = (14, 16, 18)
SUBJECT_PAIRS = [(DEV[0], DEV[1]), (DEV[1], DEV[2])]
TWO = tuple(f'{f}__{m}' for f, m in SUBJECT_PAIRS)


def frozen_formula_taper(u, grid, L=4):
    """boundary_taper of the frozen preopt.py, the same operations (the subprocess test runs the frozen one on the same records)."""
    rs = []
    for n in grid.shape:
        k = np.arange(n)
        rs.append(np.minimum(np.minimum(k, n-1-k)/float(L), 1.))
    w = rs[0][:, None, None]*rs[1][None, :, None]*rs[2][None, None, :]
    return w[..., None]*np.asarray(u, dtype=np.float64), w


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ---- control (ii) and the formulas -------------------------------------------------------------------------------------------------------------
def test_control_ii_passes_against_pre_energy_k():
    record = p1.control_ii(core.pre_energy_k)
    assert record['passed'] is True and record['max_relative'] <= 1e-12 and len(record['cases']) == sum(len(c[6]) for c in p1.CONTROL_II_CASES) == 12
    assert all(r['floor_active'] for r in record['cases'])
    assert any(r['torch']['min_det'] < 0 for r in record['cases'])                      # folded cells: det itself is compared, not only near 1
    small = [r for r in record['cases'] if r['seed'] == 14]                             # revision 1 (review minor 5): the size of a T88 field, no fold
    assert len(small) == 3 and all(r['folded'] is False and .5 < r['torch']['min_det'] < 1 and r['floor_active'] for r in small)
    u_int = p1.synthetic_case(*p1.CONTROL_II_CASES[3][:6])[0]
    det, _ = p1.cell_terms(np.pad(u_int, ((0, 0), (1, 1), (1, 1), (1, 1))), p1.CONTROL_II_CASES[3][2])
    assert np.abs(det-1).max() < .5


def test_control_ii_fails_when_a_case_does_not_fold_as_it_is_meant_to(monkeypatch):
    cases = list(p1.CONTROL_II_CASES)
    cases[3] = cases[3][:7]+(True,)
    monkeypatch.setattr(p1, 'CONTROL_II_CASES', tuple(cases))
    assert p1.control_ii(core.pre_energy_k)['passed'] is False


def test_control_ii_can_fail(monkeypatch):
    """A det off by 1e-9 relative, a difference without the averaging, and a warp with zero padding instead of the border are caught."""
    real_terms, real_warp = p1.cell_terms, p1.warp

    def perturbed(u, spacing):
        det, curl = real_terms(u, spacing)
        return det*(1+1e-9), curl
    monkeypatch.setattr(p1, 'cell_terms', perturbed)
    assert p1.control_ii(core.pre_energy_k)['passed'] is False

    def unaveraged(u, spacing):                                       # forward differences at the corner node, no averaging: another stencil
        def d(axis):
            g = np.diff(u, axis=1+axis)
            for ax in range(3):
                if ax != axis:
                    g = np.take(g, np.arange(g.shape[1+ax]-1), axis=1+ax)
            return g
        h = tuple(float(s) for s in spacing)
        gz_, gy_, gx_ = d(0), d(1), d(2)
        F = [[(1. if c == dd else 0.)+(gx_, gy_, gz_)[dd][c]*(h[c]/h[dd]) for dd in range(3)] for c in range(3)]
        det = F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1])-F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])+F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0])
        return det, np.stack((F[2][1]-F[1][2], F[0][2]-F[2][0], F[1][0]-F[0][1]))
    monkeypatch.setattr(p1, 'cell_terms', unaveraged)
    assert p1.control_ii(core.pre_energy_k)['passed'] is False
    monkeypatch.setattr(p1, 'cell_terms', real_terms)
    from scipy.ndimage import map_coordinates
    monkeypatch.setattr(p1, 'warp', lambda M, coords: M if coords is None else map_coordinates(M, coords, order=1, mode='constant', cval=0., prefilter=False))
    assert p1.control_ii(core.pre_energy_k)['passed'] is False
    monkeypatch.setattr(p1, 'warp', real_warp)
    assert p1.control_ii(core.pre_energy_k)['passed'] is True


def test_the_warp_is_grid_sample_with_the_border_beyond_the_faces():
    rng = np.random.default_rng(5)
    shape = (7, 8, 9)
    M = rng.uniform(size=shape)
    u = rng.normal(scale=3., size=(3,)+shape)
    ours = p1.warp(M, p1.sample_coordinates(u))
    import torch.nn.functional as Fn
    D, H, W = shape
    zz, yy, xx = np.meshgrid(np.arange(D, dtype=float), np.arange(H, dtype=float), np.arange(W, dtype=float), indexing='ij')
    grid = torch.tensor(np.stack((2*(xx+u[0])/(W-1)-1, 2*(yy+u[1])/(H-1)-1, 2*(zz+u[2])/(D-1)-1), axis=-1))[None]
    theirs = Fn.grid_sample(torch.tensor(M)[None, None], grid, mode='bilinear', padding_mode='border', align_corners=True)[0, 0].numpy()
    outside = (np.abs(u) > 2).any(axis=0)
    assert outside.sum() > 50 and np.max(np.abs(ours-theirs)) <= 1e-12
    assert p1.warp(M, None) is M


@pytest.mark.parametrize('seed', (0, 1))
def test_onehot_channel_bit_for_bit(seed):
    L = np.random.default_rng(seed).integers(0, 36, (12, 13, 14)).astype(np.int16)
    L[L == 7] = 0                                                                      # an absent label
    stack = fast.onehot_channels(L)
    boxes = p1.label_boxes(L)
    for i in range(35):
        assert p1.onehot_channel(L, i, boxes).tobytes() == stack[i].tobytes()


def test_the_voxel_field_is_the_u_of_the_optimizer():
    u_mm = np.random.default_rng(2).normal(size=(5, 6, 7, 3))
    u = p1.voxel_field(u_mm, (1., 1.25, .75))
    assert u.shape == (3, 5, 6, 7) and np.array_equal(u[1], u_mm[..., 1]/1.25) and np.array_equal(u[2], u_mm[..., 2]/.75)


def test_cell_terms_slab_by_slab_are_the_whole_array():
    rng = np.random.default_rng(3)
    u = np.zeros((3, 11, 9, 10))
    u[:, 1:-1, 1:-1, 1:-1] = rng.normal(size=(3, 9, 7, 8))
    spacing = (1., 1.25, .75)
    det, curl = p1.cell_terms(u, spacing)
    tissue = rng.uniform(size=(4, 11, 9, 10))
    whole = p1.geometry_sums(u, spacing, tissue=tissue, slab=10**6)
    for slab in (1, 3, 4):
        parts = p1.geometry_sums(u, spacing, tissue=tissue, slab=slab)
        assert parts['n_cells'] == det.size and parts['min_det'] == float(det.min())
        for key in ('sum_e', 'sum_q', 'sum_floor0'):
            assert parts[key] == pytest.approx(whole[key], rel=1e-13)
        assert np.allclose(parts['sum_pe'], whole['sum_pe'], rtol=1e-13, atol=0)
    for z0, z1 in p1.slabs(10, 3):
        d, c = p1.cell_terms(u[:, z0:z1+1], spacing)
        assert d.tobytes() == det[z0:z1].tobytes() and c.tobytes() == curl[:, z0:z1].tobytes()
    centre = p1.cell_centre(tissue)
    assert np.allclose(centre[:, 2, 3, 4], tissue[:, 2:4, 3:5, 4:6].mean(axis=(1, 2, 3)), rtol=1e-14)
    e = (det-1)**2
    assert whole['sum_pe'][1] == pytest.approx(float((centre[1]*e).sum()), rel=1e-12)


def test_the_geometry_of_a_field_against_pre_energy_k():
    """sum (det - 1)^2 + sum |curl|^2 and the floor of geometry_sums are pre_energy_k's with mu = 2, kappa = 1."""
    rng = np.random.default_rng(4)
    u_int = gaussian_filter(rng.normal(size=(3, 10, 12, 9)), (0, .8, .8, .8))*3
    u = np.pad(u_int, ((0, 0), (1, 1), (1, 1), (1, 1)))
    F = np.zeros((1, 12, 14, 11))
    for spacing in ((1., 1., 1.), (1.1, .9, 1.3)):
        g = p1.geometry_sums(u, spacing, j0=.6, slab=4)
        _, theirs = core.pre_energy_k(torch.tensor(u_int), torch.tensor(F), torch.tensor(F), spacing, 2., 0., 1., .6)
        assert g['sum_e']+g['sum_q'] == pytest.approx(theirs['geometry'], rel=1e-12) and g['sum_floor0'] == pytest.approx(theirs['floor'], rel=1e-12) and g['min_det'] == theirs['min_det']


# ---- the M step and its reading ----------------------------------------------------------------------------------------------------------------------
def sums_of(R_brain, R_support, N):
    return [dict(R_grid=float(b)+1., R_brain=float(b), R_support=float(s), N_support=int(n)) for b, s, n in zip(R_brain, R_support, N)]


def test_em_weights_on_the_three_domains_by_hand():
    rng = np.random.default_rng(6)
    V = rng.integers(50, 5000, 35).astype(float)
    V[[3, 9]] = 0                                                       # absent in the fixed image
    R_B = rng.uniform(1, 9, 35)
    R_S = R_B*rng.uniform(.5, 1., 35)
    N = rng.integers(10, 900, 35)
    R_B[12] = 0.                                                        # no residual on the brain: no finite precision
    N[20] = 0                                                           # an empty support
    sums = sums_of(R_B, R_S, N)
    present = V > 0
    w, sigma2, left = p1.em_weights(sums, V, 10**5, 'brain')
    ok = present & (R_B > 0)
    expected = (1/R_B[ok])/np.mean(1/R_B[ok])
    assert np.allclose(w[ok], expected, rtol=1e-14) and np.isnan(w[~ok]).all() and w[ok].mean() == pytest.approx(1., rel=1e-14)
    assert np.allclose(sigma2[ok], R_B[ok]/10**5, rtol=1e-15) and left == [13]
    w, sigma2, left = p1.em_weights(sums, V, 10**5, 'support')
    ok = present & (N > 0)
    expected = (N[ok]/R_S[ok])/np.mean(N[ok]/R_S[ok])
    assert np.allclose(w[ok], expected, rtol=1e-14) and np.isnan(w[~ok]).all() and left == [21]
    w, sigma2, left = p1.em_weights(sums, V, 10**5, 'support_shared')
    assert (w[ok] == 1.).all() and np.isnan(w[~ok]).all() and sigma2 == pytest.approx(R_S[ok].sum()/N[ok].sum(), rel=1e-14) and left == [21]
    with pytest.raises(ValueError):
        p1.em_weights(sums, V, 10**5, 'grid')


def test_channel_sums_brute_force():
    rng = np.random.default_rng(7)
    F, W = rng.uniform(size=(2, 6, 7, 8))
    brain = rng.uniform(size=(6, 7, 8)) > .3
    s = p1.channel_sums(F, W, brain, theta=.4)
    rho2 = (W-F)**2
    support = brain & ((F > .4) | (W > .4))
    assert s['R_grid'] == pytest.approx(rho2.sum(), rel=1e-14) and s['R_brain'] == pytest.approx(rho2[brain].sum(), rel=1e-14)
    assert s['R_support'] == pytest.approx(rho2[support].sum(), rel=1e-14) and s['N_support'] == support.sum()


def test_q3h_weights_are_mc0p_core_s():
    L = np.random.default_rng(8).integers(0, 36, (10, 11, 12)).astype(np.int16)
    L[L == 5] = 0
    w, voxels = mc.channel_weights('Q3h', L)
    assert np.array_equal(p1.q3h_weights(voxels), w)


def test_beta_and_r2():
    V = np.geomspace(30, 3e5, 35)
    for beta in (-.5, -2/3, -1., .3):
        w = 3.*V**beta
        fit = p1.fit_beta(w, V)
        assert fit['beta'] == pytest.approx(beta, abs=1e-12) and fit['r2'] == pytest.approx(1., abs=1e-12) and fit['n'] == 35
    assert p1.fit_beta(p1.q3h_weights(V), V)['beta'] == pytest.approx(-.5, abs=1e-12)
    noisy = V**-.5*np.exp(np.random.default_rng(9).normal(scale=.5, size=35))
    fit = p1.fit_beta(noisy, V)
    assert 0 < fit['r2'] < 1
    flat = p1.fit_beta(np.ones(35), V)
    assert flat['beta'] == 0. and flat['r2'] is None
    w = V**-.5
    w[:33] = np.nan
    assert p1.fit_beta(w, V)['beta'] is None


def test_group_ratios():
    w_q = np.ones(35)
    w_em = np.ones(35)
    for label in p1.SMALL_SIX:
        w_em[label-1] = 3.
    w_em[p1.CORTEX[0]-1] = np.nan                                     # a label not defined under EM leaves the group sum on both sides
    r = p1.group_ratios(w_em, w_q)
    assert r['small_six'] == 3. and r['cortex'] == 1. and r['white_matter'] == 1. and r['large_six'] == 1.
    assert p1.group_ratios(np.full(35, np.nan), w_q)['small_six'] is None


def test_tissue_moments():
    sums = dict(sum_p=[10., 4., 5., 6.], sum_pe=[1., 2., 1., .6], sum_pq=[2., 4., 3., 1.2], sum_e=8., sum_q=4., n_cells=100)
    m = p1.tissue_moments(sums, ['CSF', 'GM', 'WM'], 2)
    assert m['classes'] == ['background', 'CSF', 'GM', 'WM'] and m['wm_class'] == 'WM'
    assert m['tau2']['CSF'] == .5 and m['ratio_to_wm']['CSF'] == pytest.approx(5.) and m['ratio_to_wm']['WM'] == 1. and m['ratio_to_wm']['GM'] == pytest.approx(2.)
    assert m['curl_to_det'] == .5 and m['b_per_component'] == 6. and m['b_helmholtz'] == 4. and m['n_cells'] == 100
    sums.update(sum_p_brain=[1., 2., 2., 4.], sum_pe_brain=[.1, 1., .4, .4], sum_pq_brain=[1., 1., 1., 1.], sum_e_brain=2., sum_q_brain=2., n_cells_brain=10)
    mb = p1.tissue_moments(sums, ['CSF', 'GM', 'WM'], 2, '_brain')
    assert mb['ratio_to_wm']['CSF'] == pytest.approx(5.) and mb['ratio_to_wm']['GM'] == pytest.approx(2.) and mb['b_per_component'] == pytest.approx(3.) and mb['n_cells'] == 10


def test_the_brain_cells_of_geometry_sums():
    rng = np.random.default_rng(15)
    u = np.zeros((3, 12, 10, 11))
    u[:, 1:-1, 1:-1, 1:-1] = rng.normal(scale=.3, size=(3, 10, 8, 9))
    tissue = rng.uniform(size=(4, 12, 10, 11))
    brain = rng.uniform(size=(12, 10, 11)) > .2
    g = p1.geometry_sums(u, (1., 1., 1.), tissue=tissue, brain=brain, slab=3)
    det, curl = p1.cell_terms(u, (1., 1., 1.))
    keep = np.ones(det.shape, dtype=bool)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                keep &= brain[dz:dz+11, dy:dy+9, dx:dx+10]
    e, q = (det-1)**2, (curl**2).sum(axis=0)
    centre = p1.cell_centre(tissue)
    assert g['n_cells_brain'] == keep.sum() and 0 < keep.sum() < det.size and g['n_cells'] == det.size
    assert g['sum_e_brain'] == pytest.approx(e[keep].sum(), rel=1e-12) and g['sum_q_brain'] == pytest.approx(q[keep].sum(), rel=1e-12) and g['sum_e'] == pytest.approx(e.sum(), rel=1e-12)
    assert np.allclose(g['sum_pe_brain'], [(c[keep]*e[keep]).sum() for c in centre], rtol=1e-12) and np.allclose(g['sum_p_brain'], [c[keep].sum() for c in centre], rtol=1e-12)
    assert 'sum_e_brain' not in p1.geometry_sums(u, (1., 1., 1.), tissue=tissue)


def test_a_field_that_moves_csf_more_gives_a_csf_ratio_above_one():
    rng = np.random.default_rng(10)
    shape = (16, 14, 15)
    tissue = np.zeros((4,)+shape)
    tissue[1, :8], tissue[3, 8:] = 1., 1.                               # CSF in the lower half, WM in the upper
    tissue = (tissue+1e-3)/(1+4e-3)
    u = np.zeros((3,)+shape)
    u[:, 1:7, 1:-1, 1:-1] = gaussian_filter(rng.normal(size=(3, 6, 12, 13)), (0, .7, .7, .7))*2
    u[:, 9:-1, 1:-1, 1:-1] = gaussian_filter(rng.normal(size=(3, 6, 12, 13)), (0, .7, .7, .7))*.3
    m = p1.tissue_moments(p1.geometry_sums(u, (1., 1., 1.), tissue=tissue), ['CSF', 'GM', 'WM'], 2)
    assert m['ratio_to_wm']['CSF'] > 5


def test_block_bootstrap():
    index = np.arange(19)
    x = np.random.default_rng(11).uniform(.5, 2., 19)
    a, b = p1.block_interval(x, index), p1.block_interval(x, index)
    assert a == b and a['lo'] < x.mean() < a['hi']
    # runs/20261105-fd2/recompute.py block, line for line
    rng = np.random.default_rng(20261105)
    out = []
    while len(out) < 10000:
        counts = np.bincount(rng.integers(0, 20, 20), minlength=20)
        w = counts[index]*counts[index+1]
        if w.sum():
            out.append(float((w*x).sum()/w.sum()))
    lo, hi = np.percentile(out, [2.5, 97.5])
    assert (a['lo'], a['hi']) == (float(lo), float(hi))
    g = p1.block_interval(x, index, log=True)
    assert g['centre'] == pytest.approx(np.exp(np.log(x).mean())) and g['lo'] < g['centre'] < g['hi'] and g['scale'].startswith('log')
    flat = p1.block_interval(np.full(19, 1.1), index, log=True)
    assert flat['lo'] == pytest.approx(1.1) and flat['hi'] == pytest.approx(1.1)


# ---- the reading lines -------------------------------------------------------------------------------------------------------------------------------
def test_the_distance_to_q3h():
    V = np.geomspace(30, 3e5, 35)
    w_q = p1.q3h_weights(V)
    assert p1.distance_to_q3h(w_q, w_q) == (0., 35)
    r = np.random.default_rng(16).normal(scale=.3, size=35)
    w = w_q*np.exp(r)
    w[[2, 9]] = np.nan                                                                  # labels without a finite EM weight are left out
    keep = np.isfinite(w)
    d, n = p1.distance_to_q3h(w, w_q)
    assert n == 33 and d == pytest.approx(float(np.sqrt(np.mean(r[keep]**2))), rel=1e-12)
    assert p1.distance_to_q3h(np.full(35, 1.25), np.ones(35))[0] == pytest.approx(p1.A_NULL, rel=1e-15)
    assert p1.distance_to_q3h(np.full(35, np.nan), w_q) == (None, 0)


@pytest.mark.parametrize('d, reading, to_p2', ((0., 'A 是空操作', False), (np.log(1.25), 'A 是空操作', False), (np.nextafter(np.log(1.25), 1.), 'A 不是空操作', True), (.9, 'A 不是空操作', True)))
def test_the_line_of_a(d, reading, to_p2):
    r = p1.read_a(d, -.5)
    assert r['reading'] == reading and r['to_p2'] is to_p2 and r['beta_band'] == '体积指数接近 −½'
    assert p1.A_NULL == pytest.approx(.2231435513, abs=1e-10)


@pytest.mark.parametrize('beta, band', ((-.61, '体积指数比 −½ 陡'), (-.6, '体积指数接近 −½'), (-.5, '体积指数接近 −½'), (-.4, '体积指数接近 −½'), (-.39, '体积指数比 −½ 平'), (None, None)))
def test_the_beta_band_is_only_described(beta, band):
    assert p1.beta_band(beta) == band
    for d in (.1, .5):                                                                  # the band never changes the reading of A
        assert p1.read_a(d, beta)['to_p2'] == p1.read_a(d, -.5)['to_p2']


def iv(lo, hi):
    return dict(lo=lo, hi=hi)


@pytest.mark.parametrize('csf, gm, m_csf, m_gm, reading, to_p2', (
    (iv(.8, 1.25), iv(.9, 1.1), 1., 1., 'C 是空操作', False),
    (iv(.81, 1.2), iv(.85, 1.24), 1.6, 1., 'C 是空操作', False),            # both intervals inside: null comes first
    (iv(.9, 1.1), iv(.9, 1.26), 1., 1.1, 'C 是弱改动', True),                # GM/WM out of the band: no longer null (revision 1)
    (iv(.79, 1.2), iv(.9, 1.1), 1., 1., 'C 是弱改动', True),
    (iv(1.3, 2.), iv(.9, 1.1), 1.5, 1., 'C 是实质改动', True),
    (iv(1.3, 2.), iv(.9, 1.1), 1.49, 1., 'C 是弱改动', True),
    (iv(.3, .6), iv(.9, 1.1), .5, 1., 'C 是实质改动', True),                 # CSF tighter than WM: the band added by revision 1
    (iv(.3, .7), iv(.9, 1.1), 1/1.5, 1., 'C 是实质改动', True),
    (iv(.9, 1.1), iv(1.4, 1.9), 1., 1.6, 'C 是实质改动', True),              # GM alone
))
def test_the_line_of_c(csf, gm, m_csf, m_gm, reading, to_p2):
    r = p1.read_c(dict(CSF=csf, GM=gm), dict(CSF=m_csf, GM=m_gm))
    assert r['reading'] == reading and r['to_p2'] is to_p2
    if reading == 'C 是实质改动':
        assert r['beyond'] and all(x['cls'] in ('CSF', 'GM') for x in r['beyond'])
    if m_csf <= 1/1.5:
        assert '与设计预期的方向相反' in r['beyond'][0]['direction']


def test_control_iii_flags_beyond_five_percent():
    rows = [dict(pair='a', ratio=dict(data=1.0499, geometry=.9501)), dict(pair='b', ratio=dict(data=1.0501, geometry=1.)), dict(pair='c', ratio=dict(data=1., geometry=.9)), dict(pair='d', ratio=dict(data=None, geometry=1.))]
    flagged = p1.iii_abnormal(rows)
    assert [x['pair'] for x in flagged] == ['b', 'c', 'd'] and flagged[1]['terms'] == ['geometry'] and flagged[2]['terms'] == ['data']
    assert p1.iii_abnormal([dict(pair='e', ratio=dict(data=1.04, geometry=.97, floor=3., min_det=-9.))]) == []        # floor and min det only described


def test_decide():
    null_c = (dict(CSF=iv(.9, 1.1), GM=iv(.9, 1.1)), dict(CSF=1., GM=1.))
    stopped = p1.decide(False, .5, -.5, *null_c)
    assert stopped['stopped'] is True and stopped['to_p2'] == []
    d = p1.decide(True, .5, -.5, *null_c)
    assert d['stopped'] is False and d['held'] is False and d['to_p2'] == ['A'] and d['C']['reading'] == 'C 是空操作'
    d = p1.decide(True, .1, -.7, dict(CSF=iv(1.4, 2.), GM=iv(.9, 1.1)), dict(CSF=1.7, GM=1.))
    assert d['to_p2'] == ['C'] and d['A']['reading'] == 'A 是空操作' and d['A']['beta_band'] == '体积指数比 −½ 陡' and d['C']['reading'] == 'C 是实质改动'
    assert p1.decide(True, .1, -.5, *null_c)['to_p2'] == []
    held = p1.decide(True, .5, -.5, *null_c, abnormal=[dict(pair='0438-0439', terms=['data'])])
    assert held['held'] is True and held['to_p2'] is None and held['A']['reading'] == 'A 不是空操作' and '暂缓' in held['text']
    assert p1.read_a(None)['to_p2'] is None and p1.read_c(None, None)['to_p2'] is None and p1.read_c(dict(CSF=iv(.9, 1.1)), dict(CSF=1.))['to_p2'] is None
    assert p1.BETA_LINE == (-.6, -.4) and p1.NULL_BAND == (.8, 1.25) and p1.SUBSTANTIAL == 1.5 and p1.THETA == .05 and p1.C_CLASSES == ('CSF', 'GM') and p1.III_TOL == .05


# ---- the firewall --------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('keys', (['fixed', 'moving', 'initial_u', 'fixed_seg35'], ['fixed', 'moving_seg4', 'moving'], ['fixed'], ['fixed', 'moving', 'labels']))
def test_a_member_other_than_the_images_is_refused(keys):
    with pytest.raises(SystemExit, match='only'):
        p1.check_keys_read(keys)
    assert p1.check_keys_read(['fixed', 'moving', 'initial_u']) == ['fixed', 'moving', 'initial_u'] and p1.check_keys_read(['fixed', 'moving']) == ['fixed', 'moving']


@pytest.mark.parametrize('name', ('metrics_0600.json', 'repair_metrics.json', 'metrics.json', 'long.csv', 'summary_rw.md'))
def test_no_evaluation_file_is_read(tmp_path, name):
    (tmp_path/name).write_text('{}')
    with pytest.raises(SystemExit, match='reads no evaluation'):
        RUN.read_record(tmp_path/name)


def test_the_source_reads_no_label_and_no_evaluation():
    for path in (ROOT/'experiments/p1/p1_core.py', ROOT/'experiments/p1/run_p1.py'):
        text = path.read_text()
        for word in ('fixed_seg', 'moving_seg', "seg35'", "seg4'", 'read_truth', 'TRUTH_MEMBERS', 'metrics_0', 'repair_metrics', 'long.csv', '_evaluate', 'labs['):
            assert word not in text, (path.name, word)
        assert 'np.load(world.stage' not in text and "z['fixed" not in text
    assert set(fx.ALL_PAIRS) == set(RUN.PAIRS) and len(RUN.PAIRS) == 19 and RUN.PAIRS[0].startswith('OASIS_OAS1_0438') and RUN.PAIRS[-1].endswith('0457_MR1')


def test_only_the_19_pairs():
    for pair in ('OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1', 'OASIS_OAS1_0457_MR1__OASIS_OAS1_0458_MR1', 'S37__S38'):
        with pytest.raises(SystemExit, match='only the 19'):
            RUN.require_pair(pair)
    assert [RUN.PAIRS.index(p) for p in RUN.PAIRS] == list(range(19))
    assert all(p.split('__')[0].endswith(f'{438+i:04d}_MR1') and p.split('__')[1].endswith(f'{439+i:04d}_MR1') for i, p in enumerate(RUN.PAIRS))


# ---- the start -------------------------------------------------------------------------------------------------------------------------------------
GOOD_ENV = dict(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def good_state():
    return dict(root=str(core.MAIN_REPO), head='a'*40, tracked_clean=True, dirty=[], files={}, digest='d'*64)


def good_o():
    return SimpleNamespace(root=str(RUN.DEFAULT_ROOT), commit='a'*40, code_digest='d'*64)


def test_the_unit_command():
    lines = RUN.unit_command('a'*40, 'd'*64)
    text = '\n'.join(lines)
    assert 'pytest' not in text and len(lines) == 2 and lines[0] == f'mkdir -p {RUN.DEFAULT_ROOT}'
    for piece in ('--unit=tag-p1', '--property=MemoryMax=4G', '--nice=19', '--property=CPUQuota=100%', '--setenv=CUDA_VISIBLE_DEVICES= ', f'--setenv=PYTHONPATH={FROZEN}:{core.MAIN_REPO}',
                  f'--working-directory={core.MAIN_REPO}', f'{core.MAIN_REPO}/experiments/p1/run_p1.py run --commit {"a"*40} --code-digest {"d"*64}', '--setenv=OPENBLAS_NUM_THREADS=1'):
        assert piece in lines[1], piece
    assert RUN.UNIT.startswith('tag-') and not RUN.UNIT.startswith('seq-vp')


@pytest.mark.parametrize('change, fragment', (
    (dict(o=dict(root='/tmp/elsewhere')), 'writes only to'),
    (dict(state=dict(root='/tmp/a_worktree')), 'not the main checkout'),
    (dict(state=dict(head='b'*40)), 'is not --commit'),
    (dict(state=dict(tracked_clean=False, dirty=[' M experiments/p1/p1_core.py'])), 'not clean'),
    (dict(state=dict(digest='e'*64)), 'code digest'),
    (dict(busy=['tag-rp-rhx24-c600-0438-0439.service']), 'may not run beside'),
    (dict(environ=dict(CUDA_VISIBLE_DEVICES='0')), 'CUDA_VISIBLE_DEVICES'),
    (dict(environ=dict(OMP_NUM_THREADS='8')), 'OMP_NUM_THREADS'),
    (dict(kw=dict(inside_unit=False)), 'not inside the unit'),
    (dict(kw=dict(uf=["UF's chain is at ['calibrate_foldA_s0'], not a member in training"])), 'not a member in training'),
    (dict(kw=dict(modules=['mc0_core is imported from /x/experiments/p1/mc0_core.py, not /y'])), 'imported from'),
    (dict(kw=dict(frozen=['the frozen tree: jd_curl comes from elsewhere'])), 'frozen tree'),
))
def test_every_refusal_of_the_start(tmp_path, change, fragment):
    o, state, environ = good_o(), good_state(), dict(GOOD_ENV)
    assert RUN.start_problems(o, tmp_path, state, [], environ) == []
    for key, value in change.get('o', {}).items():
        setattr(o, key, value)
    state.update(change.get('state', {}))
    environ.update(change.get('environ', {}))
    problems = RUN.start_problems(o, tmp_path, state, change.get('busy', []), environ, **change.get('kw', {}))
    assert len(problems) == 1 and fragment in problems[0]


def test_a_root_that_started_is_never_rerun(tmp_path):
    (tmp_path/'started.json').write_text('{}')
    assert any('never rerun' in p for p in RUN.start_problems(good_o(), tmp_path, good_state(), [], GOOD_ENV))


def test_busy_units_reads_systemctl_with_activating(tmp_path, monkeypatch):
    shim = tmp_path/'bin'
    shim.mkdir()
    args = tmp_path/'args.txt'
    (shim/'systemctl').write_text(f'#!/bin/sh\necho "$@" > {args}\necho "tag-rp-queue.service loaded active running RP"\n')
    (shim/'systemctl').chmod(0o755)
    monkeypatch.setenv('PATH', f"{shim}:{os.environ['PATH']}")
    assert RUN.busy_units() == ['tag-rp-queue.service'] and '--state=active,activating' in args.read_text() and 'tag-rp*' in args.read_text()
    RUN.busy_units(RUN.UF_PATTERNS)
    assert 'tag-uf*' in args.read_text()
    (shim/'systemctl').write_text('#!/bin/sh\nexit 1\n')
    with pytest.raises(SystemExit, match='not known to be free'):
        RUN.busy_units()


def chain_state(tmp_path, steps):
    """A UF chain_state directory as run_uf.run_step writes it: {name: (started utc, done or None)}."""
    d = tmp_path/'chain_state'
    d.mkdir()
    for name, (started, done) in steps.items():
        (d/f'{name}.started').write_text(started+'\n')
        if done:
            (d/f'{name}.done').write_text(done+'\n')
    return d


NOW = RUN.datetime(2026, 11, 6, 6, 0, 0, tzinfo=RUN.timezone.utc)
BEFORE = dict(control_check=('2026-11-06T01:00:00Z', '2026-11-06T01:10:00Z'), nn_infer=('2026-11-06T01:10:00Z', '2026-11-06T02:00:00Z'), nn_convert=('2026-11-06T02:00:00Z', '2026-11-06T02:30:00Z'))


@pytest.mark.parametrize('steps, ok, fragment', (
    (dict(BEFORE, train_foldA_s0=('2026-11-06T05:30:00Z', None)), True, None),                                   # a member training for 30 min
    (dict(BEFORE, train_foldA_s0=('2026-11-06T04:30:00Z', None)), False, 'more than 60'),                        # 90 min: too close to its end
    (dict(BEFORE, train_foldA_s0=('2026-11-06T03:00:00Z', '2026-11-06T05:40:00Z')), False, 'a gate between steps'),   # between steps: the next one waits on a gate
    (dict(BEFORE, train_foldA_s0=('2026-11-06T03:00:00Z', '2026-11-06T05:40:00Z'), check_foldA_s0=('2026-11-06T05:40:00Z', '2026-11-06T05:40:01Z'),
          calibrate_foldA_s0=('2026-11-06T05:41:00Z', None)), False, 'not a member in training'),
    (dict(control_check=('2026-11-06T05:50:00Z', None)), False, 'not a member in training'),
    (dict(BEFORE, train_foldB_s1=('2026-11-06T05:01:00Z', None)), True, None),
    (dict(BEFORE, train_foldA_s0=('yesterday', None)), False, 'not a UTC time'),
))
def test_uf_may_run_beside_p1_only_while_a_member_trains(tmp_path, steps, ok, fragment):
    d = chain_state(tmp_path, steps)
    problems, record = RUN.uf_status(['tag-uf-chain.service'], d, now=NOW)
    assert (problems == []) is ok
    if fragment:
        assert fragment in problems[0]
    assert RUN.uf_status([], d, now=NOW) == ([], dict(uf_units=[], state_dir=str(d), running=None, started_utc=None, minutes=None))     # no UF unit: between RP and UF


def test_uf_without_a_chain_state_cannot_be_told(tmp_path):
    problems, _ = RUN.uf_status(['tag-uf-chain.service'], tmp_path/'none', now=NOW)
    assert 'cannot be told' in problems[0]
    assert RUN.UF_TRAIN_STEPS == ('train_foldA_s0', 'train_foldB_s0', 'train_foldA_s1', 'train_foldB_s1') and RUN.UF_STATE_DIR == Path('/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_full/chain_state')


def test_inside_the_unit(tmp_path):
    cgroup = tmp_path/'cgroup'
    cgroup.write_text('0::/user.slice/user-1000.slice/user@1000.service/app.slice/tag-p1.service\n')
    assert RUN.in_unit(dict(INVOCATION_ID='abc'), cgroup) is True
    assert RUN.in_unit({}, cgroup) is False
    cgroup.write_text('0::/user.slice/user-1000.slice/session-3.scope\n')
    assert RUN.in_unit(dict(INVOCATION_ID='abc'), cgroup) is False
    cgroup.write_text('0::/user.slice/user@1000.service/app.slice/tag-p10.service\n')
    assert RUN.in_unit(dict(INVOCATION_ID='abc'), cgroup) is False


def test_the_imported_modules_are_the_digested_files():
    files = RUN.module_files()
    assert set(files) == set(RUN.IMPORTED) | {'run_p1'} and RUN.module_problems(files, ROOT) == []
    assert {rel for rel in RUN.IMPORTED.values()} | {RUN.SCRIPT} <= set(RUN.CODE_FILES)
    shadowed = dict(files, mc0_core=str(ROOT/'experiments/p1/mc0_core.py'))
    problems = RUN.module_problems(shadowed, ROOT)
    assert len(problems) == 1 and problems[0].startswith('mc0_core is imported from')


def test_the_code_digest():
    files = RUN.code_files(ROOT)
    assert set(files) == set(RUN.CODE_FILES) and RUN.code_digest(files) == RUN.code_digest(dict(reversed(list(files.items()))))
    changed = dict(files, **{RUN.CODE_FILES[0]: '0'*64})
    assert RUN.code_digest(changed) != RUN.code_digest(files)


# ---- a synthetic world ----------------------------------------------------------------------------------------------------------------------
def write_records(world, pair, seed):
    """fd2_v1's run.json and field_0600.npz, rw_v1's repair.json and repair.npz of one synthetic pair, as those batches write them; returns its pin."""
    pin = RUN.read_images(pair, world)
    predicted = fast.read_predicted(pin, unet=world.unet, seg_root=world.seg_root, allowlist=world.allowlist)
    labels = predicted.labels
    w, voxels = mc.channel_weights('Q3h', labels['fixed'])
    grid = pin.grid
    rng = np.random.default_rng(seed)
    raw = np.zeros(grid.shape+(3,))
    raw[1:-1, 1:-1, 1:-1] = gaussian_filter(rng.normal(size=tuple(n-2 for n in grid.shape)+(3,)), (1., 1., 1., 0))*4.
    tapered = np.ascontiguousarray(frozen_formula_taper(raw, grid)[0])
    interior = np.flatnonzero(np.pad(np.ones(tuple(n-2 for n in grid.shape), dtype=bool), 1).ravel())
    nodes = np.sort(rng.choice(interior, 12, replace=False))
    values = tapered.reshape(-1, 3)[nodes]+rng.normal(scale=.1, size=(12, 3))
    v = tapered.copy()
    v.reshape(-1, 3)[nodes] = values
    fd2_dir, rw_dir = world.fd2_dir(pair), world.rw_dir(pair)
    fd2_dir.mkdir(parents=True)
    rw_dir.mkdir(parents=True)
    np.savez(fd2_dir/'field_0600.npz', u=raw)
    np.savez(rw_dir/'repair.npz', nodes=nodes, values=values)
    sides = {}
    for side in ('fixed', 'moving'):
        subject = getattr(pin, side+'_subject')
        prov = json.loads((world.seg_root/subject/'provenance.json').read_text())
        sides[side] = dict(unet_output_sha256={'probs36.npz': core.sha_file(Path(world.unet.pred_root)/subject/'probs36.npz')}, atropos_qc=dict(output_sha256=prov['output_sha256']))
    terms = dict(data=1234.5, geometry=987.6, floor=12.3, curvature=0., min_det=-.1, total=2234.4)
    run = dict(schema='bayesreg.fl.run.v1', run_id=f'fd2_Hx24_x_{core.short_name(pair)}', arm='Hx24', tier='first5', pair=pair, k=35, channels='onehot', weight='Q3h', D_factor=24., oracle=False, status='done',
               config=dict(iterations=[50, 50, 100, 600], mu=[1., 1., .01, .003], kappa_pre=10., j0=.2, sigma_field=.5),
               scale=dict(D_factor=24., channel_weights=w.tolist(), fixed_voxels=voxels.tolist(), D_multiplier=4.25),
               labels_sha256={side: sha(labels[side]) for side in ('fixed', 'moving')},
               channel_sources=dict(oracle=False, truth_labels_read=False, pair_keys_read=['fixed', 'moving', 'initial_u'], sides=sides),
               checkpoints={'600': dict(step=600, file='field_0600.npz', sha256=core.sha_file(fd2_dir/'field_0600.npz'), u_sha256=sha(raw), terms=terms)},
               data=dict(pair_sha256=pin.provenance['pair_sha256'], fixed_subject=pin.fixed_subject, moving_subject=pin.moving_subject))
    write_json(fd2_dir/'run.json', run)
    repair = dict(schema='bayesreg.fl.repair.v1', pair=pair, arm='RHx24', status='certified', passed=True, oracle=False,
                  certificate=dict(status='certified_feasible', n_infeasible=0, n_uncertain=0, state_hash=sha(v)), certificate_config=dict(delta=0., max_depth=48, max_boxes=255, tile_size=65536),
                  source=dict(directory=str(fd2_dir), file='field_0600.npz', field_sha256=run['checkpoints']['600']['sha256'], tapered_u_sha256=sha(tapered)),
                  attempt=dict(file='repair.npz', sha256=core.sha_file(rw_dir/'repair.npz'), n_free_nodes=12), data=dict(pair_sha256=pin.provenance['pair_sha256']))
    write_json(rw_dir/'repair.json', repair)
    return dict(field_file='field_0600.npz', field_sha256=run['checkpoints']['600']['sha256'], u_sha256=sha(raw), tapered_u_sha256=sha(tapered)), dict(raw=raw, v=v, labels=labels)


def make_world(tmp_path, monkeypatch, synthetic=False):
    """synthetic: the synthetic training draw (C23.9-13 (8), revision 6e)."""
    fold0, fold1 = (SYN_FOLD0, SYN_FOLD1) if synthetic else (FOLD0, FOLD1)
    monkeypatch.setattr(a4, 'SHAPE', SHAPE)
    allowlist = write_json(tmp_path/'dev_subjects.json', allowlist_document(synthetic_fold_sources() if synthetic else None))
    models = registry_models(fold0, fold1) if synthetic else registry_models()
    registry = write_registry(tmp_path/'repo'/'configs'/'segmenters.json', models)
    a4.commit_registry(registry)
    monkeypatch.setattr(structure_layer, 'REPO_REGISTRY', registry)
    folds = tmp_path/'folds.json'
    subjects = {s: dict(fold='A', role='calib' if k == 0 else 'train') for k, s in enumerate(fold0)}
    subjects.update({s: dict(fold='B', role='calib' if k == 0 else 'train') for k, s in enumerate(fold1)})
    write_json(folds, dict(seed=1, rule='synthetic', subjects=subjects))
    stage = a4.write_stage(tmp_path, SUBJECT_PAIRS)
    for s in sorted({s for p in SUBJECT_PAIRS for s in p}):
        a4.write_prediction(tmp_path/'pred', s, models)
        directory = a4.write_atropos(tmp_path/'atropos', s)
        grid = make_grid_3d(nx=SHAPE[2], ny=SHAPE[1], nz=SHAPE[0], hx=1., hy=1., hz=1., unit='mm')
        for name, p in zip(a4.ATROPOS_FILES, a4.atropos_posteriors(s)):   # as atropos_seg.py writes them: units mm, qform and sform (load_scalar, the contract's reader, requires them)
            (directory/name).unlink()
            save_scalar(directory/name, p, grid)
        a4.rehash(directory)
    unet = a4.MI.UNetSource(pred_root=tmp_path/'pred', folds=folds, folds_sha256=hashlib.sha256(folds.read_bytes()).hexdigest(), models_root=tmp_path/'models', image_root=tmp_path/'images')
    world = RUN.World(results=tmp_path/'results', sources_file=tmp_path/'rw_sources.json', stage=stage, allowlist=allowlist, seg_root=tmp_path/'atropos', unet=unet, taper=frozen_formula_taper)
    pins, truth = {}, {}
    for k, pair in enumerate(TWO):
        pins[core.short_name(pair)], truth[pair] = write_records(world, pair, k)
    write_json(world.sources_file, dict(schema='bayesreg.rw.sources.v1', pairs=pins))
    world.extra.update(truth=truth, registry=str(registry), folds=str(folds))
    return world


@pytest.fixture
def world(tmp_path, monkeypatch, request):
    assert TWO == tuple(core.PAIRS[:2])
    return make_world(tmp_path, monkeypatch, synthetic=is_synthetic(request))


def spy_np_load(monkeypatch):
    """np.load with every member asked for recorded; a member with seg in its name raises."""
    seen, real = {}, np.load

    class Spy:
        def __init__(self, path, inner):
            self.name, self.inner = Path(path).name, inner

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.inner.close()

        def close(self):
            self.inner.close()

        def __contains__(self, key):
            return key in self.inner

        def __getitem__(self, key):
            seen.setdefault(self.name, set()).add(key)
            if 'seg' in key:
                raise AssertionError(f'a label member was read: {self.name}:{key}')
            return self.inner[key]

        @property
        def files(self):
            if self.name == 'pair.npz':
                raise AssertionError('the member list of pair.npz')
            return self.inner.files
    monkeypatch.setattr(np, 'load', lambda path, **kwargs: Spy(path, real(path, **kwargs)))
    return seen


def independent_R(F, M, v, spacing):
    """2 x data_term_k of one channel at the field v (torch float64): R_grid by another path."""
    u = torch.tensor(p1.voxel_field(v, spacing))
    return 2*float(core.data_term_k(u, torch.tensor(F)[None], torch.tensor(M)[None]))


@synthetic_folds('world')
def test_compute_pair_on_a_synthetic_pair(world, monkeypatch):
    seen = spy_np_load(monkeypatch)
    pair = TWO[0]
    record = RUN.compute_pair(pair, world, log=lambda *a: None)
    assert seen['pair.npz'] == {'fixed', 'moving', 'initial_u'} and seen['probs36.npz'] == {'p'} and seen['field_0600.npz'] == {'u'} and seen['repair.npz'] == {'nodes', 'values'}
    assert record['control_i']['passed'] is True and record['oracle'] is False and record['inputs']['keys_read'] == ['fixed', 'moving', 'initial_u']
    truth = world.extra['truth'][pair]
    labels, v, raw = truth['labels'], truth['v'], truth['raw']
    stack_f, stack_m = fast.onehot_channels(labels['fixed']), fast.onehot_channels(labels['moving'])
    spacing = tuple(record['spacing'])
    for i in (0, 1, 19, 34):
        assert record['sums']['uhat'][i]['R_grid'] == pytest.approx(independent_R(stack_f[i], stack_m[i], v, spacing), rel=1e-11)
        assert record['sums']['f600'][i]['R_grid'] == pytest.approx(independent_R(stack_f[i], stack_m[i], raw, spacing), rel=1e-11)
        assert record['sums']['u0'][i]['R_grid'] == pytest.approx(float(((stack_m[i]-stack_f[i])**2).sum()), rel=1e-14)
    # the brain mask and the tissue layer of the contract, independently
    from bayesreg.cli.run_pair import load_tissue_layer
    from bayesreg.seg.channels import brain_mask, simplex_with_background, tissue_channels
    pin = RUN.read_images(pair, world)
    P, accepted, _ = load_tissue_layer(pin, 'fixed', world.seg_root)
    pbar = simplex_with_background(P)[0]
    brain = brain_mask(pbar)[0]
    ptilde = tissue_channels(pbar)[0]
    assert record['n_brain'] == int(brain.sum())
    W = p1.warp(stack_m[1], p1.sample_coordinates(p1.voxel_field(v, spacing)))
    support = brain & ((stack_f[1] > .05) | (W > .05))
    assert record['sums']['uhat'][1]['N_support'] == int(support.sum()) and record['sums']['uhat'][1]['R_brain'] == pytest.approx(float(((W-stack_f[1])**2)[brain].sum()), rel=1e-12)
    # P1b: the cell terms on the whole array at once, the tissue at the cell centres
    det, curl = p1.cell_terms(p1.voxel_field(v, spacing), spacing)
    centre = p1.cell_centre(ptilde)
    e, q = (det-1)**2, (curl**2).sum(axis=0)
    tau2 = (centre*e).reshape(4, -1).sum(axis=1)/centre.reshape(4, -1).sum(axis=1)
    m = record['p1b']['moments']
    assert m['tau2']['CSF'] == pytest.approx(tau2[1], rel=1e-12) and m['ratio_to_wm']['CSF'] == pytest.approx(tau2[1]/tau2[3], rel=1e-12) and m['b_per_component'] == pytest.approx(3*e.mean()/q.mean(), rel=1e-12)
    assert m['ratio_to_wm']['GM'] == pytest.approx(tau2[2]/tau2[3], rel=1e-12) and m['b_helmholtz'] == pytest.approx(2*e.mean()/q.mean(), rel=1e-12)
    described = record['p1b']['described']
    assert set(described) == {'uhat_brain_cells', 'f600_all_cells', 'f600_brain_cells'} and described['uhat_brain_cells']['n_cells'] == int(p1.brain_cells(brain).sum())
    det6, _ = p1.cell_terms(p1.voxel_field(raw, spacing), spacing)
    tau6 = (centre*(det6-1)**2).reshape(4, -1).sum(axis=1)/centre.reshape(4, -1).sum(axis=1)
    assert described['f600_all_cells']['ratio_to_wm']['CSF'] == pytest.approx(tau6[1]/tau6[3], rel=1e-12)
    # A's distance on u_hat, brain: the RMS of the log ratio to Q3h
    w_brain = np.array([np.nan if x is None else x for x in record['p1a']['uhat']['brain']['w']])
    r = np.log(w_brain[np.isfinite(w_brain)]/np.asarray(record['w_q3h'])[np.isfinite(w_brain)])
    assert record['p1a']['uhat']['brain']['distance']['d'] == pytest.approx(float(np.sqrt(np.mean(r*r))), rel=1e-12)
    # P1a: the brain weights from the sums, the Q3h weights from the records, beta of Q3h
    present = np.asarray(record['fixed_voxels']) > 0
    R_B = np.array([s['R_brain'] for s in record['sums']['uhat']])
    w = np.array([np.nan if x is None else x for x in record['p1a']['uhat']['brain']['w']])
    assert np.allclose(w[present], (1/R_B[present])/np.mean(1/R_B[present]), rtol=1e-13)
    assert record['w_q3h'] == json.loads((world.fd2_dir(pair)/'run.json').read_text())['scale']['channel_weights']
    assert record['p1a']['uhat']['support_shared']['fit']['beta'] == 0. and record['p1a']['uhat']['support_shared']['fit']['r2'] is None
    # control (iii) is written beside the record
    iii = record['control_iii']
    assert iii['recorded']['data'] == 1234.5 and iii['recomputed']['geometry'] == pytest.approx(.5*.003*(iii['geometry_600']['sum_e']+iii['geometry_600']['sum_q']))
    assert iii['recomputed']['data'] == pytest.approx(.5*4.25*sum(record['w_q3h'][i]*record['sums']['f600'][i]['R_grid'] for i in range(35)), rel=1e-12)
    json.dumps(record, default=RUN._plain)


@pytest.mark.parametrize('breaking', ('labels', 'voxels', 'weights'))
@synthetic_folds('world')
def test_control_i_fails_on_any_difference(world, breaking):
    pair = TWO[0]
    path = world.fd2_dir(pair)/'run.json'
    run = json.loads(path.read_text())
    if breaking == 'labels':
        run['labels_sha256']['moving'] = '0'*64
    elif breaking == 'voxels':
        run['scale']['fixed_voxels'][4] += 1.
    else:
        run['scale']['channel_weights'][0] = np.nextafter(run['scale']['channel_weights'][0], 1.)
    write_json(path, run)
    record = RUN.compute_pair(pair, world, log=lambda *a: None)
    assert record['control_i']['passed'] is False and 'p1a' not in record and 'sums' not in record


@pytest.mark.parametrize('breaking, fragment', (
    ('field_file', 'sha256'), ('repair_file', 'sha256'), ('state', 'certified state'), ('taper', 'the repair started from'), ('status', 'status'), ('oracle', 'oracle'),
    ('pin', 'pin'), ('directory', 'source directory'), ('config', 'configuration is not C'), ('arm', 'is not Hx24'),
))
@synthetic_folds('world')
def test_an_input_that_is_not_its_record_stops(world, monkeypatch, breaking, fragment):
    pair = TWO[0]
    fd2_dir, rw_dir = world.fd2_dir(pair), world.rw_dir(pair)
    run, repair = json.loads((fd2_dir/'run.json').read_text()), json.loads((rw_dir/'repair.json').read_text())
    if breaking == 'field_file':
        z = dict(np.load(fd2_dir/'field_0600.npz'))
        z['u'][3, 3, 3, 0] += 1e-9
        np.savez(fd2_dir/'field_0600.npz', **z)
    elif breaking == 'repair_file':
        np.savez(rw_dir/'repair.npz', nodes=np.array([5]), values=np.zeros((1, 3)))
    elif breaking == 'state':
        repair['certificate']['state_hash'] = '1'*64
    elif breaking == 'taper':
        world.taper = lambda u, grid, L: (np.asarray(u)*.5, None)
    elif breaking == 'status':
        repair['status'] = 'not_certified_fallback_rule_not_run'
    elif breaking == 'oracle':
        run['oracle'] = True
    elif breaking == 'pin':
        pins = json.loads(world.sources_file.read_text())
        pins['pairs'][core.short_name(pair)]['field_sha256'] = '2'*64
        write_json(world.sources_file, pins)
    elif breaking == 'directory':
        repair['source']['directory'] = '/somewhere/else'
    elif breaking == 'config':
        repair['certificate_config']['max_boxes'] = 64
    elif breaking == 'arm':
        run['D_factor'] = 16.
    write_json(fd2_dir/'run.json', run)
    write_json(rw_dir/'repair.json', repair)
    with pytest.raises(SystemExit, match=fragment):
        RUN.compute_pair(pair, world, log=lambda *a: None)


@synthetic_folds('world')
def test_a_label_member_read_by_the_image_reader_is_refused(world, monkeypatch):
    """The counterexample: a reader that reports a label member among the members read stops the pair before any number."""
    from bayesreg.io import pair as pair_module
    real = pair_module.load_pair

    def leaky(*args, **kwargs):
        pin = real(*args, **kwargs)
        pin.provenance['keys_read'].append('fixed_seg35')
        return pin
    monkeypatch.setattr(pair_module, 'load_pair', leaky)
    with pytest.raises(SystemExit, match='only'):
        RUN.compute_pair(TWO[0], world, log=lambda *a: None)


def machine(**changes):
    """The start's checks all passing (a test changes one)."""
    return dict(dict(state=good_state(), busy=[], environ=GOOD_ENV, uf=[], inside_unit=True), **changes)


@synthetic_folds('world')
def test_run_all_writes_the_summary_and_the_reading(world, tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, 'PAIRS', TWO)
    out = tmp_path/'p1_out'
    lines = []
    code = RUN.run_all(good_o(), world, root=out, log=lines.append, **machine())
    assert code == 0, lines
    end = json.loads((out/'end.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    started = json.loads((out/'started.json').read_text())
    assert started['machine']['inside_unit'] is True and set(started['modules']) == set(RUN.IMPORTED) | {'run_p1'}
    assert end['status'] == 'done' and summary['n_pairs'] == 2 and summary['controls']['i'] is True and summary['controls']['ii'] is True
    assert sorted(p.name for p in (out/'pairs').iterdir()) == ['0438-0439.json', '0439-0440.json']
    assert set(summary['p1a']) == {'u0', 'uhat'} and set(summary['p1a']['uhat']) == {'brain', 'support', 'support_shared', 'beta_brain_minus_support'}
    records = [json.loads((out/'pairs'/f'{core.short_name(p)}.json').read_text()) for p in TWO]
    assert summary['p1a']['uhat']['brain']['beta']['median'] == pytest.approx(float(np.median([r['p1a']['uhat']['brain']['fit']['beta'] for r in records])))
    assert summary['p1a']['uhat']['brain']['distance']['median'] == pytest.approx(float(np.median([r['p1a']['uhat']['brain']['distance']['d'] for r in records])))
    # the synthetic records hold made-up T88 terms: control (iii) flags both pairs and the readings are held, nothing stops
    reading = summary['reading']
    assert [x['pair'] for x in summary['controls']['iii_abnormal']] == ['0438-0439', '0439-0440'] and reading['held'] is True and reading['to_p2'] is None and reading['stopped'] is False
    assert reading['A']['reading'] in ('A 是空操作', 'A 不是空操作') and reading['C']['reading'] in ('C 是空操作', 'C 是实质改动', 'C 是弱改动')
    assert {'uhat', 'uhat_brain_cells', 'f600_all_cells', 'f600_brain_cells'} == set(summary['p1b']) and 'block_log' in summary['p1b']['uhat']['GM']
    md = (out/'summary.md').read_text()
    assert '开发集 19 对单次' in md and '不是 Dice' in md and '它证伪不了什么' in md and '暂缓' in md and 'Helmholtz' in md
    rows = (out/'per_channel.csv').read_text().splitlines()
    assert len(rows) == 1+2*3*35 and rows[0].startswith('pair,field,label')
    assert all(not p.name.endswith('.npz') for p in out.rglob('*'))                      # no field is written
    with pytest.raises(SystemExit, match='never rerun'):
        RUN.run_all(good_o(), world, root=out, log=lines.append, **machine())
    # the recorded terms set to what was recomputed: nothing flagged, the reading is given
    for p, r in zip(TWO, records):
        path = world.fd2_dir(p)/'run.json'
        run = json.loads(path.read_text())
        run['checkpoints']['600']['terms'].update(data=r['control_iii']['recomputed']['data'], geometry=r['control_iii']['recomputed']['geometry']*1.04)
        write_json(path, run)
    again = tmp_path/'p1_again'
    assert RUN.run_all(good_o(), world, root=again, log=lines.append, **machine()) == 0
    summary = json.loads((again/'summary.json').read_text())
    assert summary['controls']['iii_abnormal'] == [] and summary['reading']['held'] is False and isinstance(summary['reading']['to_p2'], list)
    assert '暂缓' not in (again/'summary.md').read_text()


@synthetic_folds('world')
def test_run_all_stops_after_a_failed_control_and_reads_nothing_more(world, tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, 'PAIRS', TWO)
    path = world.fd2_dir(TWO[0])/'run.json'
    run = json.loads(path.read_text())
    run['scale']['fixed_voxels'][0] += 1.
    write_json(path, run)
    called = []
    real = RUN.compute_pair
    monkeypatch.setattr(RUN, 'compute_pair', lambda pair, world, log: called.append(pair) or real(pair, world, log))
    out = tmp_path/'p1_out'
    assert RUN.run_all(good_o(), world, root=out, log=lambda *a: None, **machine()) == 8
    end = json.loads((out/'end.json').read_text())
    assert end['status'] == 'stopped' and 'control (i)' in end['reason'] and called == [TWO[0]] and not (out/'summary.json').exists()
    assert json.loads((out/'pairs'/'0438-0439.json').read_text())['control_i']['passed'] is False


@synthetic_folds('world')
def test_run_all_stops_when_control_ii_fails(world, tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, 'PAIRS', TWO)
    monkeypatch.setattr(p1, 'control_ii', lambda f: dict(control='ii', passed=False, max_relative=1., cases=[]))
    out = tmp_path/'p1_out'
    assert RUN.run_all(good_o(), world, root=out, log=lambda *a: None, **machine()) == 8
    assert 'control (ii)' in json.loads((out/'end.json').read_text())['reason'] and not (out/'pairs'/'0438-0439.json').exists()


@pytest.mark.parametrize('changes', (dict(state=dict(good_state(), tracked_clean=False, dirty=['x'])), dict(inside_unit=False), dict(uf=['not a member in training']), dict(busy=['tag-rp-queue.service'])))
@synthetic_folds('world')
def test_a_refused_start_writes_nothing(world, tmp_path, changes):
    out = tmp_path/'p1_out'
    with pytest.raises(SystemExit, match='does not start'):
        RUN.run_all(good_o(), world, root=out, log=lambda *a: None, **machine(**changes))
    assert not out.exists()


@synthetic_folds('world')
def test_a_start_without_the_frozen_tree_is_refused(world, tmp_path):
    """In this process jd_curl is the repository's: with the real taper asked for, the start is refused before anything is written."""
    world.taper = None
    out = tmp_path/'p1_out'
    with pytest.raises(SystemExit, match='frozen tree'):
        RUN.run_all(good_o(), world, root=out, log=lambda *a: None, **machine())
    assert not out.exists()


@needs_frozen
@synthetic_folds('world')
def test_the_dry_run_on_the_synthetic_world(world, monkeypatch):
    monkeypatch.setattr(RUN, 'DEFAULT_ROOT', Path(__import__('tempfile').gettempdir())/'p1_dry_run_never_written'/'p1_v1')     # first: the machine's own p1_v1 (P1 has run there) is never read; nothing is written there
    assert not RUN.DEFAULT_ROOT.exists()
    monkeypatch.setattr(RUN, 'PAIRS', TWO)
    monkeypatch.setattr(RUN, 'code_state', lambda root: good_state())
    monkeypatch.setattr(RUN, 'busy_units', lambda patterns=RUN.BUSY_PATTERNS: ['tag-uf-chain.service'] if patterns == RUN.UF_PATTERNS else [])
    monkeypatch.setattr(RUN, 'UF_STATE_DIR', world.results/'no_chain_state')
    lines = []
    assert RUN.dry_run(world, out=lines.append) == [] and lines[-1] == 'problems: none'
    assert any(line.startswith('machine now') and 'tag-uf-chain.service' in line for line in lines) and any(line.startswith('note: P1 would not start now') for line in lines)
    assert any(line.startswith('start: systemd-run') and 'pytest' not in line for line in lines) and any('estimate〔估算〕' in line for line in lines)
    probs = Path(world.unet.pred_root)/DEV[1]/'probs36.npz'
    probs.write_bytes(probs.read_bytes()+b'x')
    lines = []
    problems = RUN.dry_run(world, out=lines.append)
    assert len(problems) == 2 and all('probs36' in p for p in problems) and lines[-3] == 'problems: '


@needs_frozen
@synthetic_folds('world')
def test_the_pipeline_in_a_subprocess_with_the_frozen_taper(world, tmp_path):
    """run_all as the unit runs it, but on the synthetic world: PYTHONPATH=<frozen>:<code>, the frozen boundary_taper, the records written above."""
    script = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import run_p1 as RUN
from bayesreg.seg import structure_layer
w = json.loads(sys.argv[2])
structure_layer.REPO_REGISTRY = Path(w['registry'])
mi = RUN.core.make_inputs()
unet = mi.UNetSource(pred_root=Path(w['pred']), folds=Path(w['folds']), folds_sha256=w['folds_sha256'], models_root=Path(w['models']), image_root=Path(w['images']))
world = RUN.World(results=w['results'], sources_file=w['sources'], stage=w['stage'], allowlist=w['allowlist'], seg_root=w['seg_root'], unet=unet)
assert world.taper is None
RUN.PAIRS = tuple(w['pairs'])
state = dict(root=str(RUN.core.MAIN_REPO), head='a'*40, tracked_clean=True, dirty=[], files={}, digest='d'*64)
o = SimpleNamespace(root=str(RUN.DEFAULT_ROOT), commit='a'*40, code_digest='d'*64)
env = dict(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
sys.exit(RUN.run_all(o, world, root=Path(w['out']), log=print, state=state, busy=[], environ=env, uf=[], inside_unit=True))
"""
    w = dict(registry=world.extra['registry'], folds=world.extra['folds'], folds_sha256=world.unet.folds_sha256, pred=str(world.unet.pred_root), models=str(world.unet.models_root), images=str(world.unet.image_root),
             results=str(world.results), sources=str(world.sources_file), stage=str(world.stage), allowlist=str(world.allowlist), seg_root=str(world.seg_root), pairs=list(TWO), out=str(tmp_path/'p1_sub'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{FROZEN}:{ROOT}')
    done = subprocess.run([PY, '-B', '-c', script, str(ROOT/'experiments/p1'), json.dumps(w)], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:]+done.stderr[-3000:]
    end = json.loads((tmp_path/'p1_sub'/'end.json').read_text())
    assert end['status'] == 'done' and json.loads((tmp_path/'p1_sub'/'summary.json').read_text())['controls']['i'] is True
