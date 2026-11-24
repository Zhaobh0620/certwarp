"""P2 / P3 (runs/20261106-p2/任务书.md, plan.md): experiments/p2/{p2_core,decide_p2,run_p2}.py.  CPU only, synthetic small volumes in tmp_path; no real image, label, prediction
or field is read (the reference test reads JSON records of fd2_v1 and rw_v1 only, as the dry run does); nothing here starts a unit.

- the frozen path: pre_energy_p2 and run_level_p2 without their arguments, and with a = 1, are pre_energy_k and run_level_k bit for bit; optimize_p2 with hooks that do nothing
  (EM without M steps, a = None) and with a = 1 on every level is optimize_k bit for bit (a subprocess with the frozen tree); the comparison can fail;
- A: the M step against C11-2 by hand, at u = 0 and at a field (P1's channel_sums and warp), the normalization, the labels left out, the Gram norm against data_grad_norm,
  the scale rule after the M steps (the native checks), the rescaling and its rebuild path, the collapse flag;
- C: the moment ratios against p1_core and against a brute-force sum, the cell weights' normalization on Omega_B cells, the coarse levels of C03.6-12, the class index;
- the readings: every branch of P2's gate and order and of P3's lines, with their boundaries; the method's value with the fallback; decide_p2 on synthetic records;
- the firewall: the source, and a synthetic pipeline of the three arms under a spy on np.load, C's pass 0 against optimize_k itself;
- the reference: its file, its loader's refusals, HS's reading, C's control; the cells, the plan, the start strings.
Revision 2 (the review of 11-06 and the lead's rulings): every row of a reference checked item by item, a wrong copy of each refused; a fallback row of HS entering with HS's value; the
fallback unit certifying again without a label (a fake certificate: no numba compile); the collapse flag within one level; the restart's checks; P3's arm from P2's reading computed again;
the mark for the lead's rejudgment; P3 undecidable on a missing value; C's control whatever the status, its field kept; the native marks of Adam's rounds.
"""
import ast
from dataclasses import dataclass
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

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT/'experiments/p2/run_p2.py'
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
needs_frozen = pytest.mark.skipif(not (FROZEN/'jd_curl/three_d/preopt.py').is_file(), reason='the frozen worktree is not on this machine')
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{FROZEN}:{ROOT}')
torch.set_num_threads(1)


def _load():
    spec = importlib.util.spec_from_file_location('_p2_run', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules['_p2_run'] = module
    spec.loader.exec_module(module)
    return module


RUN = _load()
p2, core, mc, fast, decision, p1 = RUN.p2, RUN.core, RUN.mc, RUN.fast, RUN.decision, RUN.p2.p1
SHAPE = (12, 14, 16)
SPACING = (1., 1.25, .75)                           # (hx, hy, hz)


@pytest.fixture(autouse=True)
def no_real_machine(monkeypatch):
    """A test never starts a process through the queue's machine."""
    def forbidden(*args, **kwargs):
        raise AssertionError('the real System.run was called')
    monkeypatch.setattr(RUN.run.System, 'run', forbidden)


# ---- synthetic channels ----------------------------------------------------------------------------------------------------------------------------
def label_maps(shape=SHAPE):
    L = np.zeros(shape, dtype=np.int16)
    L[2:-2, 2:-2, 2:-2] = 1
    for k, label in enumerate((20, 2, 21, 7, 11)):
        L[2+k:4+k, 3:-3, 3:-3] = label
    L[4:6, 4:6, 4:7] = 35
    return L, np.roll(L, 1, axis=1)


def brain_of(shape=SHAPE):
    b = np.zeros(shape, dtype=bool)
    b[1:-1, 1:-2, 2:-1] = True
    return b


def prepared(L, M, D=24., g_q0=.7):
    """fl_channels.prepare_stacks' numbers on one-hot channels: Q3h, then the scale rule with factor D (the Q0 reference norm given)."""
    F0, M0 = fast.onehot_channels(L), fast.onehot_channels(M)
    w, voxels = mc.channel_weights('Q3h', L)
    root = np.sqrt(w)[:, None, None, None]
    Fw, Mw = F0*root, M0*root
    unscaled = core.data_grad_norm(Fw, Mw, 'cpu')
    s0 = core.scale_factor(g_q0, unscaled)*D
    host = [Fw*np.sqrt(s0), Mw*np.sqrt(s0)]
    return F0, M0, host, dict(channel_weights=w.tolist(), D_multiplier=s0, g_norm_Q0=g_q0, fixed_voxels=voxels.tolist(), g_norm_unscaled=unscaled, D_factor=D)


def smooth_field(shape=SHAPE, amplitude=.8, seed=5):
    rng = np.random.default_rng(seed)
    u = np.stack([gaussian_filter(rng.normal(size=shape), 1.5) for _ in range(3)], axis=-1)
    u *= amplitude/np.abs(u).max()
    u[0], u[-1], u[:, 0], u[:, -1], u[:, :, 0], u[:, :, -1] = 0, 0, 0, 0, 0, 0
    return u


def hand_weights(F0, M0, brain, present, warped=None):
    R = np.array([float((((M0[c] if warped is None else warped[c])-F0[c])**2)[brain].sum()) for c in range(len(F0))])
    ok = present & (R > 0)
    precision = np.zeros(len(R))
    precision[ok] = (brain.sum()+2)/R[ok]
    w = np.zeros(len(R))
    w[ok] = precision[ok]/precision[ok].mean()
    return w, R


# ---- the frozen path ---------------------------------------------------------------------------------------------------------------------------------
def test_pre_energy_p2_is_pre_energy_k_bit_for_bit_without_a_and_with_a_one():
    rng = np.random.default_rng(1)
    for dtype in (torch.float64, torch.float32):
        F = torch.tensor(gaussian_filter(rng.uniform(size=(3, 9, 10, 11)), (0, 1, 1, 1)), dtype=dtype)
        M = torch.tensor(gaussian_filter(rng.uniform(size=(3, 9, 10, 11)), (0, 1, 1, 1)), dtype=dtype)
        u0 = rng.normal(size=(3, 7, 8, 9))*.6
        for alpha in (0., 1e-3):
            results = []
            for weight in ('frozen', None, 'ones'):
                u = torch.tensor(u0, dtype=dtype, requires_grad=True)
                if weight == 'frozen':
                    total, terms = core.pre_energy_k(u, F, M, (1., 1.25, .75), .003, alpha, 10., .2)
                else:
                    a = None if weight is None else torch.ones((8, 9, 10), dtype=dtype)
                    total, terms = p2.pre_energy_p2(u, F, M, (1., 1.25, .75), .003, alpha, 10., .2, a)
                grad, = torch.autograd.grad(total, u)
                results.append((total.detach().numpy().tobytes(), terms, grad.numpy().tobytes()))
            assert results[0] == results[1] == results[2]


def test_pre_energy_p2_weighs_det_cell_by_cell():
    rng = np.random.default_rng(2)
    F = torch.tensor(rng.uniform(size=(2, 8, 9, 10)))
    M = torch.tensor(rng.uniform(size=(2, 8, 9, 10)))
    u = rng.normal(size=(3, 6, 7, 8))*.3
    a = rng.uniform(.5, 2., size=(7, 8, 9))
    _, terms = p2.pre_energy_p2(torch.tensor(u), F, M, SPACING, .5, 0., 10., .2, torch.tensor(a))
    det, curl = p1.cell_terms(np.pad(u, ((0, 0), (1, 1), (1, 1), (1, 1))), SPACING)
    assert terms['geometry'] == pytest.approx(.25*((a*(det-1)**2).sum()+(curl**2).sum()), rel=1e-12)
    _, plain = p2.pre_energy_p2(torch.tensor(u), F, M, SPACING, .5, 0., 10., .2)
    assert plain['geometry'] != terms['geometry'] and plain['data'] == terms['data'] and plain['floor'] == terms['floor']


def test_run_level_p2_is_run_level_k_bit_for_bit_and_calls_between():
    rng = np.random.default_rng(3)
    F = torch.tensor(gaussian_filter(rng.uniform(size=(4, 10, 11, 12)), (0, 1, 1, 1)), dtype=torch.float32)
    M = torch.tensor(gaussian_filter(rng.uniform(size=(4, 10, 11, 12)), (0, 1, 1, 1)), dtype=torch.float32)
    config = SimpleNamespace(iterations=(6,), seconds=1e9, mu=(.003,), alpha=(1e-3,), kappa_pre=10., j0=.2, sigma_grad=1., sigma_field=.5, beta1=.9, beta2=.99, lr=.5, cap=.5, eps=1e-8)
    out, calls = [], []
    for kind in ('frozen', 'none', 'ones'):
        u = torch.tensor(rng.normal(size=(3, 8, 9, 10))*0+.01, dtype=torch.float32, requires_grad=True)
        m, v = torch.zeros_like(u), torch.zeros_like(u)
        counters = dict(cap_hits=0)
        if kind == 'frozen':
            first, last, done, stopped = core.run_level_k(u, m, v, F, M, (1., 1., 1.), 0, config, 0., None, counters)
        else:
            a = None if kind == 'none' else torch.ones((9, 10, 11), dtype=torch.float32)
            first, last, done, stopped = p2.run_level_p2(u, m, v, F, M, (1., 1., 1.), 0, config, 0., None, counters, a, calls.append if kind == 'ones' else None)
        out.append((u.detach().numpy().tobytes(), m.numpy().tobytes(), v.numpy().tobytes(), first, last, done, stopped, counters['cap_hits']))
    assert out[0] == out[1] == out[2]
    assert calls == list(range(6))


OPTIMIZE = """
import sys, hashlib
sys.path.insert(0, sys.argv[1])
import numpy as np, torch
import p2_core as p2
core = p2.core
fz = core.frozen_api()
import fast_channels as fast, mc0p_core as mc
from jd_curl.three_d.grid import make_grid_3d
from bayesreg.seg.channels import brain_mask, simplex_with_background, tissue_channels
from scipy.ndimage import gaussian_filter
grid = make_grid_3d(nx=30, ny=28, nz=26, hx=1., hy=1.25, hz=.75, unit='mm')
L = np.zeros(grid.shape, dtype=np.int16); L[5:20, 6:22, 7:24] = 1; L[8:14, 9:15, 10:16] = 20; L[3:6, 3:5, 3:9] = 7; L[15:18, 10:14, 12:16] = 2
Mv = np.roll(L, 2, axis=1)
w, voxels = mc.channel_weights('Q3h', L)
def stacks():
    F, M = fast.onehot_channels(L), fast.onehot_channels(Mv)
    root = np.sqrt(w)[:, None, None, None]
    F *= root; M *= root
    return [F, M]
base = stacks()
g_q0 = .9
unscaled = core.data_grad_norm(base[0], base[1], 'cpu')
s0 = core.scale_factor(g_q0, unscaled)*24.
def scaled():
    s = stacks()
    s[0] *= np.sqrt(s0); s[1] *= np.sqrt(s0)
    return s
scale = dict(channel_weights=w.tolist(), D_multiplier=s0, g_norm_Q0=g_q0, fixed_voxels=voxels.tolist(), g_norm_unscaled=unscaled)
rng = np.random.default_rng(4)
P = gaussian_filter(rng.uniform(size=(3,)+grid.shape), (0, 2, 2, 2)); P /= P.sum(axis=0)
inside = np.zeros(grid.shape, dtype=bool); inside[4:-4, 4:-4, 4:-4] = True
P *= inside
pbar, _ = simplex_with_background(P)
brain, _ = brain_mask(pbar)
ptilde, _ = tissue_channels(pbar)
config = fz.PreoptConfig(**dict(core.PRE_C, iterations=(2, 2, 2, 6), seconds=600., sigma_field=.5))
ref, rec = core.optimize_k(scaled(), grid, config, device='cpu', workers=1, record_level_norms=True)
# hooks that do nothing: the EM without an M step, a = None
s = scaled()
em = p2.EMWeights(s, scale, brain, grid.spacing, 'cpu', schedule=dict(heads=(), native=()))
mine, rec2 = p2.optimize_p2(s, grid, config, device='cpu', workers=1, record_level_norms=True, hooks=p2.Hooks(em, None))
assert mine.tobytes() == ref.tobytes() and rec2['level_g_norm'] == rec['level_g_norm'] and [l['last'] for l in rec2['levels']] == [l['last'] for l in rec['levels']]
assert em.gram_check['passed'] and em.history == []
# a = 1 on every level
class Ones(p2.Hooks):
    def cell_weight(self, level, dev, dtype):
        n = fz.four_level_grids(grid)[level].shape
        return torch.ones(tuple(k-1 for k in n), device=dev, dtype=dtype)
seen = []
class Seen(Ones):
    def between(self, level, F, M, u_int, level_grid, h):
        return seen.append
ones, _ = p2.optimize_p2(scaled(), grid, config, device='cpu', workers=1, hooks=Seen())
assert ones.tobytes() == ref.tobytes() and len(seen) == 12
direct, _ = p2.optimize_p2(scaled(), grid, config, device='cpu', workers=1)
assert direct.tobytes() == ref.tobytes()
# the comparison can fail: a = 1.5
class Half(p2.Hooks):
    def cell_weight(self, level, dev, dtype):
        n = fz.four_level_grids(grid)[level].shape
        return torch.full(tuple(k-1 for k in n), 1.5, device=dev, dtype=dtype)
other, _ = p2.optimize_p2(scaled(), grid, config, device='cpu', workers=1, hooks=Half())
assert other.tobytes() != ref.tobytes()
# A with its M steps: the heads of the four levels and native steps 2, 4; the scale rule holds on the native tensors after each
s = scaled()
em = p2.EMWeights(s, scale, brain, grid.spacing, 'cpu', schedule=dict(heads=(0, 1, 2, 3), native=(2, 4)))
u, rec3 = p2.optimize_p2(s, grid, config, device='cpu', workers=1, record_level_norms=True, hooks=p2.Hooks(em, None))
assert [(h['where']['level'], h['where']['step']) for h in em.history] == [(0, 0), (1, 0), (2, 0), (3, 0), (3, 2), (3, 4)]
assert len(em.scale_checks) == 3 and all(c['passed'] for c in em.scale_checks), em.scale_checks
marks = em.native_marks
assert [m['step'] for m in marks] == [0, 2, 4] and all(m['cap_hits'] is not None for m in marks) and all(m['energy_before']['data'] != m['energy_after']['data'] and m['energy_before']['geometry'] == m['energy_after']['geometry'] for m in marks[1:])
assert all(h['collapse_flagged'] == [] for h in em.history[:4])                       # across level heads: recorded, never flagged
# the follow-up's minor 8: with the records of native_marks off, the field is the same, bit for bit
real_energy, real_cap = p2._energy_record, p2.Hooks.cap_hits
p2._energy_record = lambda *args: None
p2.Hooks.cap_hits = lambda self: None
s = scaled()
quiet = p2.EMWeights(s, scale, brain, grid.spacing, 'cpu', schedule=dict(heads=(0, 1, 2, 3), native=(2, 4)))
u_quiet, _ = p2.optimize_p2(s, grid, config, device='cpu', workers=1, record_level_norms=True, hooks=p2.Hooks(quiet, None))
p2._energy_record, p2.Hooks.cap_hits = real_energy, real_cap
assert u_quiet.tobytes() == u.tobytes() and all(m['energy_before'] is None and m['cap_hits'] is None for m in quiet.native_marks[1:])
assert abs(rec3['level_g_norm'][3]['raw']/(24*g_q0)-1) <= 1e-3 and u.tobytes() != ref.tobytes()
assert em.history[0]['w'] != em.history[-1]['w'] and abs(np.mean([x for x, v in zip(em.history[-1]['w'], voxels) if v > 0])-1) < 1e-12
# C with ratios: every level weighted, mean 1 on its Omega_B cells
index = p2.class_index(['CSF', 'GM', 'WM'], 2)
levels = p2.tissue_levels(ptilde, brain, grid, fz.four_level_grids(grid), fz.downsample_array, index)
tw = p2.TissueWeights(levels, dict(CSF=2., GM=.5))
uc, _ = p2.optimize_p2(scaled(), grid, config, device='cpu', workers=1, hooks=p2.Hooks(None, tw))
assert len(tw.records) == 4 and all(abs(r['mean_brain']-1) < 1e-12 for r in tw.records) and uc.tobytes() != ref.tobytes()
print('same')
"""


@needs_frozen
def test_optimize_p2_without_effect_is_optimize_k_bit_for_bit_and_the_hooks_act(tmp_path):
    done = subprocess.run([PY, '-B', '-c', OPTIMIZE, str(ROOT/'experiments/p2')], env=ENV, cwd=tmp_path, capture_output=True, text=True, timeout=280)
    assert done.returncode == 0 and 'same' in done.stdout, done.stdout[-2000:]+done.stderr[-3000:]


def test_the_copies_change_only_their_marked_lines():
    """pre_energy_p2 and run_level_p2 against the functions they copy: the statements outside the marked additions are the same, line for line (AST).  optimize_p2 is compared by its
    behaviour only, bit for bit in the subprocess above (the review's minor 5: no AST comparison of it)."""
    def body(path, name):
        tree = ast.parse(Path(path).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        return [ast.unparse(s) for s in node.body[1:]]                                   # without the docstring
    mc0 = ROOT/'experiments/mc0/mc0_core.py'
    mine = body(ROOT/'experiments/p2/p2_core.py', 'pre_energy_p2')
    theirs = body(mc0, 'pre_energy_k')
    assert [s.replace('core.data_term_k', 'data_term_k') for s in mine if not s.startswith('if cell_weight is None')] == [s for s in theirs if not s.startswith('geometry =')]
    mine = body(ROOT/'experiments/p2/p2_core.py', 'run_level_p2')
    theirs = body(mc0, 'run_level_k')
    assert len(mine) == len(theirs) and [s for s in mine if not s.startswith('for ')] == [s for s in theirs if not s.startswith('for ')]
    loop_m, loop_t = next(s for s in mine if s.startswith('for ')), next(s for s in theirs if s.startswith('for '))
    assert loop_m.replace('core._smooth', '_smooth').replace('pre_energy_p2', 'pre_energy_k').replace(', cell_weight)', ')').replace('\n    if between is not None:\n        between(it)', '') == loop_t


# ---- A: the M step, the weights, the scale ---------------------------------------------------------------------------------------------------------------------
def test_em_at_u0_is_c11_2_by_hand_and_normalized():
    L, Mv = label_maps()
    F0, M0, host, scale = prepared(L, Mv)
    brain = brain_of()
    em = p2.EMWeights(host, scale, brain, SPACING, 'cpu')
    row = em.m_step('head', None)
    present = np.asarray(scale['fixed_voxels']) > 0
    w, R = hand_weights(F0, M0, brain, present)
    assert np.allclose(row['R'], R, rtol=1e-12, atol=0) and np.allclose(row['w'], w, rtol=1e-12, atol=0)
    assert abs(np.mean(np.asarray(row['w'])[present])-1) < 1e-12 and all(row['w'][i] == 0 for i in np.flatnonzero(~present))
    assert row['sigma2'][0] == pytest.approx(R[0]/(brain.sum()+2), rel=1e-12) and row['n_brain'] == int(brain.sum())
    assert em.gram_check['passed'] and row['used'] and em.s == row['s']


def test_em_at_a_field_is_p1s_channel_sums():
    L, Mv = label_maps()
    F0, M0, host, scale = prepared(L, Mv)
    brain = brain_of()
    u = smooth_field()
    em = p2.EMWeights(host, scale, brain, SPACING, 'cpu')
    row = em.m_step('native', u)
    coords = p1.sample_coordinates(p1.voxel_field(u, SPACING))
    warped = np.stack([p1.warp(M0[c], coords) for c in range(35)])
    R = np.array([p1.channel_sums(F0[c], warped[c], brain)['R_brain'] for c in range(35)])
    assert np.allclose(row['R'], R, rtol=1e-12, atol=1e-300)
    w, _ = hand_weights(F0, M0, brain, np.asarray(scale['fixed_voxels']) > 0, warped)
    assert np.allclose(row['w'], w, rtol=1e-12, atol=0) and row['w'] != em.m_step('head', None, used=False)['w']


def test_em_weights_left_out_and_refusal():
    present = np.array([True, True, False, True])
    w, sigma2, left = p2.em_weights(np.array([2., 0., 5., 4.]), present, 98)
    assert left == [2] and w[1] == 0 and w[2] == 0 and np.isnan(sigma2[1]) and np.isnan(sigma2[2])
    assert w[0] == pytest.approx(2*(1/2)/((1/2)+(1/4)), rel=1e-15) and abs(w[[0, 3]].mean()-1) < 1e-15
    assert sigma2[0] == 2./100
    with pytest.raises(ValueError):
        p2.em_weights(np.array([0., 0.]), np.array([True, True]), 10)
    with pytest.raises(ValueError):
        p2.em_weights(np.array([1., 2.]), np.array([False, False]), 10)


def test_gram_norm_is_data_grad_norm_of_the_weighted_channels():
    L, Mv = label_maps()
    F0, M0, host, scale = prepared(L, Mv)
    em = p2.EMWeights(host, scale, brain_of(), SPACING, 'cpu')
    rng = np.random.default_rng(7)
    present = np.asarray(scale['fixed_voxels']) > 0
    for _ in range(3):
        w = np.where(present, rng.uniform(.1, 3., 35), 0.)
        root = np.sqrt(w)[:, None, None, None]
        assert em.g_norm(w) == pytest.approx(core.data_grad_norm(F0*root, M0*root, 'cpu'), rel=1e-5)
    assert em.gram_check['relative'] < 1e-5


def test_the_scale_rule_holds_on_the_rescaled_tensors_and_a_zero_channel_is_rebuilt():
    L, Mv = label_maps()
    F0, M0, host, scale = prepared(L, Mv)
    em = p2.EMWeights(host, scale, brain_of(), SPACING, 'cpu')
    F, M = (core.stack_tensor(side, torch.device('cpu'), torch.float32) for side in host)
    assert em.native_check(F, M, 'before')['passed']                                   # Q3h, s0: the frozen rule
    em.m_step('head', None)
    em.apply(F, M)
    assert em.native_check(F, M, 'head')['passed']
    old = em.r.copy()
    i = int(np.flatnonzero(np.asarray(scale['fixed_voxels']) > 0)[0])
    em.r[i] = 0.
    em.apply(F, M, old)
    assert float(F[i].abs().max()) == 0
    zero = em.r.copy()
    em.r = old
    em.apply(F, M, zero)
    expected = torch.tensor(host[0][i], dtype=torch.float32)*float(old[i])
    assert torch.equal(F[i], expected) and em.native_check(F, M, 'again')['passed']


def test_collapse_flags_compare_used_steps():
    assert p2.collapse_flags(np.array([1., 1., np.nan]), np.array([.05, .2, 1.])) == [1]
    assert p2.collapse_flags(np.array([1., 1.]), np.array([.1, .2])) == []                # exactly 10x is not more than 10x


def test_hooks_without_m_steps_upload_the_frozen_tensors():
    L, Mv = label_maps()
    _, _, host, scale = prepared(L, Mv)
    em = p2.EMWeights(host, scale, brain_of(), SPACING, 'cpu', schedule=dict(heads=(), native=()))
    hooks = p2.Hooks(em, None)
    F, M = hooks.upload(3, host, np.zeros(SHAPE+(3,)), [None]*4, torch.device('cpu'), torch.float32)
    G, N = (core.stack_tensor(side, torch.device('cpu'), torch.float32) for side in host)
    assert torch.equal(F, G) and torch.equal(M, N) and hooks.between(3, F, M, None, None, None) is None and hooks.cell_weight(0, None, None) is None


# ---- C: the moments and the cell weights -----------------------------------------------------------------------------------------------------------------------
def tissue(shape=SHAPE, seed=9):
    from bayesreg.seg.channels import brain_mask, simplex_with_background, tissue_channels
    rng = np.random.default_rng(seed)
    P = gaussian_filter(rng.uniform(size=(3,)+shape), (0, 1.5, 1.5, 1.5))
    P /= P.sum(axis=0)
    inside = np.zeros(shape, dtype=bool)
    inside[2:-2, 2:-2, 3:-2] = True
    P *= inside
    pbar, _ = simplex_with_background(P)
    brain, _ = brain_mask(pbar)
    ptilde, _ = tissue_channels(pbar)
    return ptilde, brain


def test_moment_ratios_are_p1s_and_a_brute_force_sum():
    ptilde, brain = tissue()
    u = smooth_field(amplitude=1.)
    ratios, record = p2.moment_ratios(u, SPACING, ptilde, brain, ['CSF', 'GM', 'WM'], 2, .2)
    sums = p1.geometry_sums(p1.voxel_field(u, SPACING), SPACING, tissue=ptilde, brain=brain, j0=.2)
    moments = p1.tissue_moments(sums, ['CSF', 'GM', 'WM'], 2)
    assert ratios == {k: moments['ratio_to_wm'][k] for k in ('CSF', 'GM')} and record['moments'] == moments
    det, _ = p1.cell_terms(p1.voxel_field(u, SPACING), SPACING)
    P = p1.cell_centre(ptilde)
    tau = [(P[k]*(det-1)**2).sum()/P[k].sum() for k in range(4)]
    assert ratios['CSF'] == pytest.approx(tau[1]/tau[3], rel=1e-10) and ratios['GM'] == pytest.approx(tau[2]/tau[3], rel=1e-10)
    for names, wm in ((['CSF', 'GM', 'WM'], 2), (['GM', 'WM', 'CSF'], 1)):
        index = p2.class_index(names, wm)
        assert index['WM'] == 1+wm and index['CSF'] == 1+names.index('CSF')
    for names, wm in ((['CSF', 'XX', 'WM'], 2), (['CSF', 'GM', 'WM'], 0)):
        with pytest.raises(ValueError):
            p2.class_index(names, wm)


def test_cell_weights_mean_one_on_brain_cells_and_ratios_one_give_one():
    ptilde, brain = tissue()
    index = p2.class_index(['CSF', 'GM', 'WM'], 2)
    from jd_curl.three_d.grid import make_grid_3d
    grid = make_grid_3d(nx=SHAPE[2], ny=SHAPE[1], nz=SHAPE[0], hx=1., hy=1.25, hz=.75, unit='mm')
    level = p2.tissue_levels(ptilde, brain, grid, [grid], None, index)[0]
    a, rec = p2.cell_weights(level, dict(CSF=3., GM=.4))
    assert abs(a[level.brain_cells].mean()-1) < 1e-12 and rec['mean_brain'] == pytest.approx(1, abs=1e-12) and a.shape == tuple(n-1 for n in SHAPE)
    t = level.base+3.*level.csf+.4*level.gm
    assert np.allclose(a*t, rec['m'], rtol=1e-13)                                         # a = m / t
    ones, _ = p2.cell_weights(level, dict(CSF=1., GM=1.))
    assert np.allclose(ones, 1., atol=1e-12)                                              # P~ sums to 1: no change
    csf = level.csf/(level.base+level.csf+level.gm)
    loose, _ = p2.cell_weights(level, dict(CSF=4., GM=1.))
    assert np.corrcoef(loose[level.brain_cells], csf[level.brain_cells])[0, 1] < 0      # a looser CSF weighs CSF cells less
    for bad in (dict(CSF=0., GM=1.), dict(CSF=np.nan, GM=1.)):
        with pytest.raises(ValueError):
            p2.cell_weights(level, bad)


def test_coarse_tissue_levels_follow_c03_6():
    ptilde, brain = tissue(shape=(34, 36, 38))
    from jd_curl.three_d.grid import make_grid_3d
    from jd_curl.three_d.nifti import downsample_array
    from bayesreg.grids import four_level_grids
    grid = make_grid_3d(nx=38, ny=36, nz=34, hx=1., hy=1., hz=1., unit='mm')
    grids = four_level_grids(grid)
    index = p2.class_index(['CSF', 'GM', 'WM'], 2)
    levels = p2.tissue_levels(ptilde, brain, grid, grids, downsample_array, index)
    level = levels[2]
    parts = [downsample_array(ptilde[k], grid, grids[2].shape, labels=False)[0] for k in range(4)]
    P = np.stack(parts)/np.stack(parts).sum(axis=0)
    cells = p1.cell_centre(P)
    mask = downsample_array(brain.astype(np.float64), grid, grids[2].shape, labels=False)[0] >= .5
    assert np.array_equal(level.csf, cells[1]) and np.array_equal(level.gm, cells[2]) and np.array_equal(level.base, cells[0]+cells[3])
    assert np.array_equal(level.brain_cells, p1.brain_cells(mask)) and levels[3].shape_cells == tuple(n-1 for n in grid.shape)


# ---- the readings ---------------------------------------------------------------------------------------------------------------------------------------------
MEASURES = decision.MEASURES


def metrics_of(seg35, sdlogj=.07, seg4=.9):
    return dict({m: .5 for m in MEASURES}, seg35=seg35, seg4=seg4, sdlogj_official=sdlogj)


def row(pair, kind=decision.CERTIFIED, seg35=.86, sdlogj=.07, wall=False, why=None):
    repair = None if kind == decision.NOT_READ else dict(after=metrics_of(seg35, sdlogj), wall_hit=wall, run_id=f'r{pair}', rounds=1)
    return dict(pair=pair, short=core.short_name(pair), arm='A', kind=kind, why=why, field=None, repair=repair, bayes={}, task_clock=100.)


def refs(pairs, seg35=.86):
    return {p: dict(pair=p, short=core.short_name(p), kind=decision.CERTIFIED, why=None, repair=dict(after=metrics_of(seg35)), value=metrics_of(seg35), source=decision.REPAIR, line_status=p2.HX_CERTIFIED, field_u_sha256='x', directory='d', pinned=True) for p in pairs}


def test_p2_gate_every_branch_and_boundary():
    pairs = core.PAIRS
    ref = refs(pairs)
    status, _, s = decision.arm_gate([row(p) for p in pairs], ref)
    assert status == decision.THROUGH and s['delta']['seg35']['mean'] == 0
    status, _, s = decision.arm_gate([row(p, seg35=0.) for p in pairs], refs(pairs, seg35=.004))
    assert s['delta']['seg35']['mean'] == -.004 and status == decision.THROUGH          # -.004 is in (at or above)
    status, why, _ = decision.arm_gate([row(p, seg35=.86-.0041) for p in pairs], ref)
    assert status == decision.NOT_THROUGH and 'delta' in why
    status, _, _ = decision.arm_gate([row(p, sdlogj=.10) for p in pairs], ref)
    assert status == decision.THROUGH
    status, why, _ = decision.arm_gate([row(p, sdlogj=.1001) for p in pairs], ref)
    assert status == decision.NOT_THROUGH and 'sdlogj' in why
    for kind in decision.FAILURES:
        rows = [row(p) for p in pairs[:4]]+[row(pairs[4], kind=kind, why='x')]
        assert decision.arm_gate(rows, ref)[0] == decision.NOT_THROUGH
    rows = [row(p) for p in pairs[:4]]+[row(pairs[4], kind=decision.NOT_READ, why='没有运行')]
    assert decision.arm_gate(rows, ref)[0] == decision.UNDECIDABLE
    rows = [row(p) for p in pairs[:3]]+[row(pairs[3], kind=decision.NOT_READ, why='x'), row(pairs[4], kind=decision.REPAIR_FAILURE, why='y')]
    assert decision.arm_gate(rows, ref)[0] == decision.NOT_THROUGH                         # a failure decides even beside a pair not read
    bad = dict(ref)
    bad[pairs[0]] = dict(bad[pairs[0]], kind=decision.NOT_READ, why='不是钉住的那份', value=None)
    assert decision.arm_gate([row(p) for p in pairs], bad)[0] == decision.UNDECIDABLE


def test_p2_order_and_reading():
    T, N, U = (decision.THROUGH, None, {}), (decision.NOT_THROUGH, 'x', {}), (decision.UNDECIDABLE, 'y', {})
    assert decision.read_p2(dict(AC=T, A=T, C=T))[:2] == (decision.READ_CANDIDATE, 'AC')
    assert decision.read_p2(dict(AC=N, A=T, C=T))[:2] == (decision.READ_CANDIDATE, 'A')
    assert decision.read_p2(dict(AC=N, A=N, C=T))[:2] == (decision.READ_CANDIDATE, 'C')
    assert decision.read_p2(dict(AC=N, A=N, C=N))[:2] == (decision.READ_NONE, None)
    assert decision.read_p2(dict(AC=U, A=T, C=T))[:2] == (decision.UNDECIDABLE, None)
    assert decision.read_p2(dict(AC=N, A=U, C=T))[:2] == (decision.UNDECIDABLE, None)
    assert decision.read_p2(dict(AC=T, A=U, C=U))[:2] == (decision.READ_CANDIDATE, 'AC')
    assert p2.ORDER == ('AC', 'A', 'C') and (p2.P2_DELTA_MIN, p2.SDLOGJ_MAX, p2.P3_LO, p2.P3_MEAN, p2.SUPERIOR_LO) == (-.004, .10, -.003, -.001, 0.)


def method(values, sdlogj=.07):
    return {p: ('repair', metrics_of(v, sdlogj), None) for p, v in zip(p2.ALL_PAIRS, values)}


def test_p3_lines_every_branch():
    ref = method([.87]*19)
    reading, note, n = decision.read_p3(method([.87]*19), ref)
    assert reading == decision.READ_GAIN and note is None and n['mean_delta_seg35'] == 0 and not n['superior']
    reading, note, n = decision.read_p3(method([.872]*19), ref)
    assert reading == decision.READ_GAIN and note == decision.SUPERIOR and n['block_seg35']['lo'] > 0
    reading, note, n = decision.read_p3(method([.87-.0011]*19), ref)
    assert reading == decision.READ_NO_GAIN and 'mean' in note
    reading, note, n = decision.read_p3(method([.87-.001]*19), ref)
    assert n['mean_delta_seg35'] == pytest.approx(-.001) and reading == (decision.READ_GAIN if n['mean_delta_seg35'] >= -.001 else decision.READ_NO_GAIN)
    rng = np.random.default_rng(11)
    noisy = .87+rng.normal(0, .008, 19)
    reading, note, n = decision.read_p3(method(noisy-noisy.mean()+.87), ref)
    assert n['block_seg35']['lo'] < -.003 and reading == decision.READ_NO_GAIN and 'lower' in note
    reading, note, n = decision.read_p3(method([.87]*19, sdlogj=.11), ref)
    assert reading == decision.READ_NO_GAIN and 'sdlogj' in note
    missing = method([.87]*19)
    missing[p2.ALL_PAIRS[3]] = (None, None, 'x')
    assert decision.read_p3(missing, ref)[0] == decision.UNDECIDABLE
    assert decision.read_p3(method([.87]*19), ref, problems=['参照 0438-0439 没有运行'])[0] == decision.UNDECIDABLE
    expected = p1.block_interval(np.zeros(19), list(range(19)))
    assert decision.read_p3(method([.87]*19), ref)[2]['block_seg35'] == expected and expected['n'] == 10000 and expected['seed'] == 20261105


def test_method_value_and_classify():
    good = row(p2.ALL_PAIRS[0])
    a0 = dict(kind=decision.A0, a0_metrics=metrics_of(.8), why=None)
    ident = dict(kind=decision.IDENTITY, a0_metrics=None, why='no pin')
    nothing = dict(kind=None, a0_metrics=None, why='the identity did not pass either')
    idy = dict(usable=True, metrics=metrics_of(.5), why=None)
    nid = dict(usable=False, metrics=None, why='none')
    assert decision.method_value(good, None, nid)[0] == decision.REPAIR
    for kind in decision.FAILURES:
        bad = row(p2.ALL_PAIRS[0], kind=kind, why='x')
        assert decision.method_value(bad, a0, nid)[:2] == (decision.A0, a0['a0_metrics'])        # the fallback unit's decision (D75), not d0_v1's records alone
        assert decision.method_value(bad, ident, idy)[:2] == (decision.IDENTITY, idy['metrics'])
        assert decision.method_value(bad, ident, nid)[:2] == (None, None)
        assert decision.method_value(bad, nothing, idy)[:2] == (None, None)
        assert decision.method_value(bad, None, idy)[:2] == (None, None)                       # no fallback.json: not decided, no value
    assert decision.method_value(row(p2.ALL_PAIRS[0], kind=decision.NOT_READ, why='x'), a0, idy)[:2] == (None, None)
    field = dict(status='done', metrics={'seg35': .8})
    rep = dict(metrics=True, certified_at_1=True, wall_hit=False)
    assert decision.classify(None, rep, True)[0] == decision.NOT_READ
    assert decision.classify(dict(status='time_cap', metrics=None), None, False)[0] == decision.ENTRY_FAILURE
    assert decision.classify(dict(status='failed', metrics=None), None, False)[0] == decision.ENTRY_FAILURE
    assert decision.classify(dict(status='done', metrics=None), None, False)[0] == decision.NOT_READ
    assert decision.classify(field, None, False)[0] == decision.NOT_READ
    assert decision.classify(field, dict(status='failed'), False)[0] == decision.REPAIR_FAILURE
    assert decision.classify(field, dict(metrics=False), True)[0] == decision.NOT_READ
    assert decision.classify(field, rep, True) == (decision.CERTIFIED, None)
    assert decision.classify(field, dict(rep, wall_hit=True), True)[0] == decision.CERTIFIED     # D74: a fuse hit, then the certificate: passed
    assert decision.classify(field, dict(rep, certified_at_1=False), True)[0] == decision.REPAIR_FAILURE


# ---- decide_p2 on synthetic records --------------------------------------------------------------------------------------------------------------------------
def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def certificate(alpha=1., n=0):
    return dict(accepted_alpha=alpha, n_infeasible_at_1=n, attempts=[dict(min_jd_lower_bound=1e-3 if alpha == 1. else -1.)])


DEFAULT_PRED = str(mc.default_unet().pred_root)


def write_field(d, cell, status='done', seg35=.86, passes=None, segmenter=p2.DEFAULT_SEGMENTER, sha='f', u='u', pred_root=DEFAULT_PRED):
    """A field run's records as run_p2.run_field writes them: the keys reference_pair_problems reads (pair, oracle, the segmenter, its pred_root and policy on both sides, the frozen Hx24's
    whole configuration, the channels, the taper, the checkpoint)."""
    write_json(d/'run.json', dict(run_id=cell.run_id, pair=cell.pair, status=status, oracle=False, k=35, weight='Q3h', D_factor=24., config=p2.frozen_t88_config(), channels='onehot', taper=4,
                                  channel_sources=dict(segmenter=dict(pred_root=pred_root), sides={s: dict(segmenter_id=segmenter, policy='all_folds_mean') for s in ('fixed', 'moving')}),
                                  checkpoints={'600': dict(file='field_0600.npz', sha256=sha, u_sha256=u, seconds_task=60., seconds_io=1., terms={})}, optimizer=dict(levels=[]), passes=passes or [], seconds_channels=3.))
    write_json(d/'queue_done.json', dict(state=status, unit_seconds=70.))
    if status == 'done':
        write_json(d/'metrics_0600.json', dict(certificate=certificate(.81, 5), metrics=metrics_of(seg35-.01), tapered_u_sha256='t'))


def write_repair(d, cell, seg35=.86, certified=True, wall=False, attempt='a', repaired='r', field_dir=None, field_sha='f'):
    rounds = [dict(inner=dict(stop='wall' if wall else 'converged'), n_free=10)]
    write_json(d/'repair.json', dict(run_id=cell.run_id, pair=cell.pair, status='certified' if certified else 'not_certified_fallback_rule_not_run', passed=certified, oracle=False, reason='certified' if certified else 'round_cap', rounds=rounds,
                                     source=dict(seconds_task=60., field_sha256=field_sha, directory=None if field_dir is None else str(field_dir)), seconds_repair=80., seconds_warmup=2., attempt=dict(sha256=attempt), eps=.05, config=p2.frozen_repair_config()))
    write_json(d/'repair_metrics.json', dict(repaired_u_sha256=repaired, before=dict(metrics=metrics_of(seg35-.01), n_infeasible_at_1=5), after=dict(metrics=metrics_of(seg35), certificate=certificate(1. if certified else .9, 0 if certified else 3)), change={}))
    write_json(d/'queue_done.json', dict(state='certified', unit_seconds=90.))


def write_reference_row(base, pair, seg35=.86, segmenter=p2.DEFAULT_SEGMENTER, pred_root=DEFAULT_PRED):
    """One certified row of a reference: its T88 field run and its repair, both right; returns the pin."""
    short = core.short_name(pair)
    tier = 'c600' if pair in core.PAIRS else 'rest600'
    fdir, rdir = Path(base)/short/'field', Path(base)/short/'repair'
    write_field(fdir, p2.RefCell('Hx24', 'first5' if pair in core.PAIRS else 'rest600', pair), segmenter=segmenter, sha=f'f{short}', u=f'u{short}', pred_root=pred_root)
    write_repair(rdir, p2.RefCell('RHx24', tier, pair), seg35=seg35, attempt=f'a{short}', repaired=f'r{short}', field_dir=fdir, field_sha=f'f{short}')
    return dict(tier=tier, repair_dir=str(rdir), attempt_sha256=f'a{short}', repaired_u_sha256=f'r{short}', field_dir=str(fdir), field_sha256=f'f{short}', field_u_sha256=f'u{short}')


def reference_world(tmp_path, monkeypatch, seg35=.86):
    """A reference file of the default segmenter pointing at synthetic field and repair directories, and the batch's pins."""
    doc = dict(schema=p2.REFERENCE_SCHEMA, segmenter_id=p2.DEFAULT_SEGMENTER, channels={}, p2=dict(source='synthetic', pairs={}), p3=dict(source='synthetic', pairs={}))
    for pair in p2.ALL_PAIRS:
        short = core.short_name(pair)
        doc['p3']['pairs'][short] = write_reference_row(tmp_path/'ref', pair, seg35)
        if pair in core.PAIRS:                                                              # one reference row per pair (plan.md §3.1)
            doc['p2']['pairs'][short] = doc['p3']['pairs'][short]
    path = write_json(tmp_path/'reference.json', doc)
    monkeypatch.setattr(p2, 'reference_path', lambda segmenter: path)
    monkeypatch.setattr(p2, 'HS_DECISION', tmp_path/'no_hs/decision_hs.json')
    return doc, path


def edit_json(path, change):
    value = json.loads(Path(path).read_text())
    change(value)
    Path(path).write_text(json.dumps(value))


MUTATIONS = {                                                                               # the review's I1: every item, one wrong copy each
    'run pair': ('field', lambda r: r.update(pair='OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1')),
    'run oracle': ('field', lambda r: r.update(oracle=True)),
    'run segmenter': ('field', lambda r: r['channel_sources']['sides']['moving'].update(segmenter_id='other')),
    'run status': ('field', lambda r: r.update(status='time_cap')),
    'run k': ('field', lambda r: r.update(k=14)),
    'run weight': ('field', lambda r: r.update(weight=None)),
    'run D factor': ('field', lambda r: r.update(D_factor=6.)),
    'run sigma_field': ('field', lambda r: r['config'].update(sigma_field=.35)),
    'run iterations': ('field', lambda r: r['config'].update(iterations=[50, 50, 100, 1200])),
    'run lr': ('field', lambda r: r['config'].update(lr=.25)),                             # revision 3 (the follow-up's minor 4): T88's every key
    'run mu': ('field', lambda r: r['config'].update(mu=[1., 1., .01, .01])),
    'run seconds': ('field', lambda r: r['config'].update(seconds=330.)),
    'run channels': ('field', lambda r: r.update(channels='onehot+intensity')),
    'run taper': ('field', lambda r: r.update(taper=0)),
    'run policy': ('field', lambda r: r['channel_sources']['sides']['fixed'].update(policy='cross_fit_other_folds')),
    'run pred_root': ('field', lambda r: r['channel_sources']['segmenter'].update(pred_root='/elsewhere/pred')),
    'repair rho0': ('repair', lambda r: r['config'].update(rho0=10.)),                     # RF05's every parameter
    'repair max_rounds': ('repair', lambda r: r['config'].update(max_rounds=3)),
    'run field sha256': ('field', lambda r: r['checkpoints']['600'].update(sha256='other')),
    'run field u sha256': ('field', lambda r: r['checkpoints']['600'].update(u_sha256='other')),
    'repair pair': ('repair', lambda r: r.update(pair='OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1')),
    'repair oracle': ('repair', lambda r: r.update(oracle=None)),
    'repair eps': ('repair', lambda r: r.update(eps=.01)),
    'repair wall': ('repair', lambda r: r['config'].update(wall_seconds=120.)),
    'repair source field': ('repair', lambda r: r['source'].update(field_sha256='other')),
    'repair source directory': ('repair', lambda r: r['source'].update(directory='/elsewhere')),
    'repair attempt': ('repair', lambda r: r['attempt'].update(sha256='other')),
    'repaired field': ('metrics', lambda r: r.update(repaired_u_sha256='other')),
    'certificate': ('metrics', lambda r: r['after']['certificate'].update(accepted_alpha=.9)),
}


@pytest.mark.parametrize('name', sorted(MUTATIONS))
def test_a_reference_row_copied_wrong_in_one_item_is_refused(tmp_path, name):
    pair = core.PAIRS[1]
    pin = write_reference_row(tmp_path, pair)
    assert p2.reference_pair_problems(pin, pair, p2.DEFAULT_SEGMENTER, DEFAULT_PRED) == ([], p2.HX_CERTIFIED)
    where, change = MUTATIONS[name]
    path = dict(field=Path(pin['field_dir'])/'run.json', repair=Path(pin['repair_dir'])/'repair.json', metrics=Path(pin['repair_dir'])/'repair_metrics.json')[where]
    edit_json(path, change)
    problems, _ = p2.reference_pair_problems(pin, pair, p2.DEFAULT_SEGMENTER, DEFAULT_PRED)
    assert problems, name
    rows = decision.reference_rows(dict(p2=dict(pairs={core.short_name(p): (pin if p == pair else write_reference_row(tmp_path/'others', p)) for p in core.PAIRS})), 'p2', p2.DEFAULT_SEGMENTER, hs_decision=tmp_path/'none.json')
    assert rows[pair]['kind'] == decision.NOT_READ and rows[pair]['value'] is None and all(r['kind'] == decision.CERTIFIED for p, r in rows.items() if p != pair)


def test_reference_rows_refuse_missing_records_another_segmenter_and_status_conflicts(tmp_path):
    pair = core.PAIRS[0]
    pin = write_reference_row(tmp_path, pair)
    assert p2.reference_pair_problems(pin, pair, 'seg_x')[0]                                # the sides name unet35_v1_ens, the reference another
    assert p2.reference_pair_problems(dict(pin, field_dir=str(tmp_path/'nothing')), pair, p2.DEFAULT_SEGMENTER)[0]
    assert p2.reference_pair_problems(dict(pin, repair_dir=str(tmp_path/'nothing')), pair, p2.DEFAULT_SEGMENTER)[0]
    problems, status = p2.reference_pair_problems(dict(pin, line_status=p2.HX_CERTIFIED, hs_line_status=p2.HX_A0), pair, p2.DEFAULT_SEGMENTER)
    assert problems and status is None
    problems, status = p2.reference_pair_problems(dict(pin, hs_line_status=p2.HX_REJECTED), pair, p2.DEFAULT_SEGMENTER)
    assert problems and status == p2.HX_REJECTED
    assert p2.reference_pair_problems(dict(pin, hs_line_status=p2.HX_CERTIFIED), pair, p2.DEFAULT_SEGMENTER) == ([], p2.HX_CERTIFIED)


SEG_X = dict(pred_root='/hs/pred', models_root='/hs/models', image_root='/hs/images', registry_rel='configs/bayesreg/segmenters_x.json')


def hs_world(tmp_path, statuses):
    """A reference of segmenter seg_x on the first 5 pairs and HS's decision_hs.json behind it.  statuses {pair index: (line_status, method value)}: those rows are fallbacks (an entry
    failure of HS: the field stopped before it read its channels, no repair, no T88 field)."""
    seg = 'seg_x'
    pins = {core.short_name(p): write_reference_row(tmp_path, p, segmenter=seg, pred_root=SEG_X['pred_root']) for p in core.PAIRS}
    for i, (status, value) in statuses.items():
        k = core.short_name(core.PAIRS[i])
        edit_json(Path(pins[k]['field_dir'])/'run.json', lambda r: r.update(status='cuda_oom', channel_sources=None, oracle=None, checkpoints={}))     # the follow-up's minor 3
        pins[k] = dict(pins[k], repair_dir=None, attempt_sha256=None, repaired_u_sha256=None, field_sha256=None, field_u_sha256=None, hs_line_status=status, method_metrics=value)
    hs = write_json(tmp_path/'decision_hs.json', dict(reading=p2.HS_ADOPT, segmenter_id=seg, per_pair=[dict(pair=k, line_status=p2.pin_line_status(v), metrics=v.get('method_metrics') or metrics_of(.86)) for k, v in pins.items()]))
    reference = dict(segmenter_id=seg, channels=SEG_X, hs=dict(decision_sha256=core.sha_file(hs)), p2=dict(pairs=pins), p3=dict(pairs={}))
    return seg, pins, hs, reference


def test_a_reference_fallback_enters_with_hss_value(tmp_path):
    """The lead on the review's I2, (b): the reference by the candidates' rule; an entry failure of HS (stopped before it read its channels: channel_sources and oracle None) has no repair
    and no T88 field; its value is the one decision_hs.json recorded.  Revision 3: every row's line_status bound to decision_hs.json; in P2 such a pair does not enter Delta (R1, (a))."""
    value = metrics_of(.84, .08)
    seg, pins, hs, reference = hs_world(tmp_path, {2: (p2.HX_IDENTITY, value)})
    bad = core.short_name(core.PAIRS[2])
    assert p2.reference_pair_problems(pins[bad], core.PAIRS[2], seg, SEG_X['pred_root']) == ([], p2.HX_IDENTITY)        # oracle None accepted on a fallback row (minor 3)
    assert p2.reference_pair_problems(dict(pins[bad], method_metrics=dict(value, sdlogj_official=None)), core.PAIRS[2], seg)[0]
    edit_json(Path(pins[bad]['field_dir'])/'run.json', lambda r: r.update(oracle=True))
    assert p2.reference_pair_problems(pins[bad], core.PAIRS[2], seg)[0]                                                  # oracle True never
    edit_json(Path(pins[bad]['field_dir'])/'run.json', lambda r: r.update(oracle=None))
    decided = json.loads(hs.read_text())
    assert p2.hs_reference_problems(reference, decided, core.sha_file(hs), seg) == []
    assert p2.hs_reference_problems(reference, None, None, seg)[0].startswith('HS:') and p2.hs_reference_problems(dict(reference, hs=dict(decision_sha256='other')), decided, core.sha_file(hs), seg)
    for change in (lambda d: d['per_pair'][2].update(metrics=metrics_of(.85, .08)), lambda d: d['per_pair'][0].update(line_status=p2.HX_A0), lambda d: d.update(reading='不采用'), lambda d: d.update(segmenter_id='seg_y')):
        other = json.loads(hs.read_text())
        change(other)
        assert p2.hs_reference_problems(reference, other, core.sha_file(hs), seg)                                     # minor 6: every row, the reading, the segmenter
    rows = decision.reference_rows(reference, 'p2', seg, hs_decision=hs)
    assert rows[core.PAIRS[2]]['kind'] == decision.FALLBACK and rows[core.PAIRS[2]]['value'] == value and all(rows[p]['kind'] == decision.CERTIFIED for p in core.PAIRS if p != core.PAIRS[2])
    status, why, s = decision.arm_gate([row(p, seg35=.86) for p in core.PAIRS], rows)
    assert status == decision.THROUGH and s['delta_excluded'] == [bad] and s['n_delta'] == 4 and s['delta']['seg35']['mean'] == 0 and bad in why and s['reference_fallbacks'] == [dict(pair=bad, line_status=p2.HX_IDENTITY)]
    # revision 4 (minor 3): the reference's mean and the arm's on the pairs of Delta only; the pair left out has no Delta in the table
    assert s['reference']['seg35'] == pytest.approx(.86) and s['after_delta_pairs']['seg35'] == pytest.approx(.86) and bad not in s['delta_pairs'] and len(s['delta_pairs']) == 4
    lines = {p: decision.pair_line_record(dict(row(p, seg35=.9), bayes={}), rows[p], exclude_fallback=True) for p in core.PAIRS}
    assert lines[core.PAIRS[2]]['excluded_from_delta'] and lines[core.PAIRS[2]]['delta_seg35'] is None and lines[core.PAIRS[2]]['reference_line_status'] == p2.HX_IDENTITY
    assert all(lines[p]['delta_seg35'] == pytest.approx(.04) and not lines[p]['excluded_from_delta'] for p in core.PAIRS if p != core.PAIRS[2])
    assert decision.pair_line_record(dict(row(core.PAIRS[2], seg35=.9), bayes={}), rows[core.PAIRS[2]])['delta_seg35'] == pytest.approx(.9-.84)     # P3: symmetric, it enters
    rows = decision.reference_rows(reference, 'p2', seg, hs_decision=tmp_path/'missing.json')
    assert all(r['kind'] == decision.NOT_READ for r in rows.values()) and decision.arm_gate([row(p) for p in core.PAIRS], rows)[0] == decision.UNDECIDABLE


def test_p2_leaves_out_the_reference_fallbacks_and_needs_four_pairs(tmp_path):
    """The lead on the follow-up's R1, (a): a pair whose reference row is a fallback does not enter Delta; with 4 left the gate reads, with 3 it is 不可判."""
    _, _, hs, reference = hs_world(tmp_path, {1: (p2.HX_IDENTITY, metrics_of(.5)), 3: (p2.HX_IDENTITY, metrics_of(.5))})
    rows = decision.reference_rows(reference, 'p2', 'seg_x', hs_decision=hs)
    status, why, s = decision.arm_gate([row(p, seg35=.86) for p in core.PAIRS], rows)
    assert status == decision.UNDECIDABLE and '只剩 3 对' in why and s['n_delta'] == 3
    status, why, _ = decision.arm_gate([row(p, kind=decision.REPAIR_FAILURE, why='x') if p == core.PAIRS[0] else row(p) for p in core.PAIRS], rows)
    assert status == decision.NOT_THROUGH                                                   # the arm's own failure decides first
    status, why, s = decision.arm_gate([row(p, seg35=.86-.005 if p == core.PAIRS[1] else .86) for p in core.PAIRS], refs(core.PAIRS))
    assert s['n_delta'] == 5 and s['delta_excluded'] == [] and why is None


def test_a_reference_row_decided_by_a_third_certificate_is_certified(tmp_path):
    """HS revision 1b's review, new minor 2: the repair process passes, the evaluation's review certificate does not, a third certificate in a process that reads no label passes; by
    C22.3-9 (d) the row is certified and outputs the repaired field.  Accepted only with that record, its evidence of no label, and its field the row's."""
    pair = core.PAIRS[3]
    short = core.short_name(pair)
    pin = write_reference_row(tmp_path, pair)
    edit_json(Path(pin['repair_dir'])/'repair_metrics.json', lambda r: r['after']['certificate'].update(accepted_alpha=None))
    assert p2.reference_pair_problems(pin, pair, p2.DEFAULT_SEGMENTER)[0]                    # no third certificate: refused
    import rp_core as rp
    record = dict(schema='bayesreg.hs.fallback.v1', pair=short, kind='disagree', reads_labels=False, certificate_config=dict(rp.CERTIFICATE_C), output='repair', keys_read=['fixed', 'moving', 'initial_u'], unit='tag-hs-fallback',
                  third_certificate=dict(passed=True, certificate=dict(accepted_alpha=1.), repaired_u_sha256=pin['repaired_u_sha256']))
    third = write_json(tmp_path/'fallback.json', record)
    pin = dict(pin, third_certificate_file=str(third), third_certificate_sha256=core.sha_file(third))
    assert p2.reference_pair_problems(pin, pair, p2.DEFAULT_SEGMENTER) == ([], p2.HX_CERTIFIED)
    rows = decision.reference_rows(dict(p2=dict(pairs={core.short_name(p): (pin if p == pair else write_reference_row(tmp_path/'others', p)) for p in core.PAIRS})), 'p2', p2.DEFAULT_SEGMENTER, hs_decision=tmp_path/'none.json')
    assert rows[pair]['kind'] == decision.CERTIFIED and rows[pair]['value']['seg35'] == .86 and 'C22.3-9 (d)' in rows[pair]['why']
    for change in (dict(reads_labels=True), dict(reads_labels=None), dict(output='A0'), dict(pair='0438-0439'), dict(third_certificate=dict(record['third_certificate'], passed=False)),
                   dict(third_certificate=dict(record['third_certificate'], repaired_u_sha256='other')), dict(certificate_config=dict(rp.CERTIFICATE_C, max_depth=8)),
                   dict(keys_read=None), dict(keys_read=['fixed', 'moving', 'fixed_seg35']), dict(keys_read=['initial_u']), dict(unit=None), dict(unit='tag-hs-eval-r'), dict(unit='hs-fallback')):     # revision 4: the evidence, the unit
        bad = write_json(tmp_path/'bad.json', dict(record, **change))
        assert p2.reference_pair_problems(dict(pin, third_certificate_file=str(bad), third_certificate_sha256=core.sha_file(bad)), pair, p2.DEFAULT_SEGMENTER)[0], change
    assert p2.reference_pair_problems(dict(pin, third_certificate_sha256='other'), pair, p2.DEFAULT_SEGMENTER)[0]
    edit_json(Path(pin['repair_dir'])/'repair.json', lambda r: r.update(passed=False))                 # revision 4: the premise, the deciding certificate passed
    assert 'third certificate: premise' in p2.reference_pair_problems(pin, pair, p2.DEFAULT_SEGMENTER)[0]


def test_an_a0_reference_row_is_bound_to_the_d0_record(tmp_path, monkeypatch):
    """The follow-up's minor 5: a reference row of configuration C takes the number the candidates' A0 takes (d0_v1's record of the pair)."""
    record = dict(qualified=True, why=None, metrics=metrics_of(.83, .07))
    monkeypatch.setattr(decision.rp, 'fallback_status', lambda pair, results=None: record)
    seg, pins, hs, reference = hs_world(tmp_path, {4: (p2.HX_A0, metrics_of(.83, .07))})
    k = core.short_name(core.PAIRS[4])
    assert p2.reference_pair_problems(pins[k], core.PAIRS[4], seg) == ([], p2.HX_A0)
    record['metrics'] = metrics_of(.82, .07)
    assert p2.reference_pair_problems(pins[k], core.PAIRS[4], seg)[0]
    record.update(metrics=metrics_of(.83, .07), qualified=False)
    assert p2.reference_pair_problems(pins[k], core.PAIRS[4], seg)[0]


def test_decide_p2_on_records(tmp_path, monkeypatch):
    doc, path = reference_world(tmp_path, monkeypatch)
    root = tmp_path/'p2_v1'
    write_json(root/'batch.json', dict(code_sha='c', segmenter_id=p2.DEFAULT_SEGMENTER, reference=dict(sha256=core.sha_file(path))))
    batch = p2.P2
    values = dict(AC=.858, A=.861, C=.862)
    for arm in p2.ORDER:
        for pair in core.PAIRS:
            f, r = batch.field_cell(arm, pair), batch.repair_cell(arm, pair)
            passes = [dict(u_sha256=f'u{f.short}' if arm == 'C' else 'z', em=None, tissue=None)]
            write_field(batch.run_dir(root, f), f, passes=passes)
            failed = arm == 'AC' and pair == core.PAIRS[2]
            write_repair(batch.run_dir(root, r), r, seg35=values[arm], certified=not failed, wall=arm == 'A' and pair == core.PAIRS[1])
    d, rows = decision.decide_p2(root)
    assert d['arms']['AC']['status'] == decision.NOT_THROUGH and d['arms']['A']['status'] == decision.THROUGH and d['reading'] == decision.READ_CANDIDATE and d['candidate'] == 'A'
    assert d['arms']['A']['summary']['fuse_hits'] == [core.short_name(core.PAIRS[1])] and all(c['same'] for c in d['control_c_pass0'])
    assert d['arms']['A']['summary']['delta']['seg35']['mean'] == pytest.approx(.001)
    out = decision.write_p2(root)
    assert (root/'analysis/decision_p2.json').is_file() and '有候选' in (root/'analysis/summary_p2.md').read_text() and out['candidate'] == 'A'
    # a reference that is not the pinned one: nothing written
    write_json(root/'batch.json', dict(code_sha='c', segmenter_id=p2.DEFAULT_SEGMENTER, reference=dict(sha256='other')))
    with pytest.raises(SystemExit):
        decision.decide_p2(root)
    # P3's choice from this reading, computed again from the records (the review's minor 7)
    write_json(root/'batch.json', dict(code_sha='c', segmenter_id=p2.DEFAULT_SEGMENTER, reference=dict(sha256=core.sha_file(path))))
    choice, why = decision.p3_arm_choice(root)
    assert choice['arm'] == 'A' and choice['p2_code_sha'] == 'c' and choice['recomputed'] and why is None
    edit_json(root/'analysis/decision_p2.json', lambda d: d.update(candidate='C'))
    choice, why = decision.p3_arm_choice(root)
    assert choice is None and 'now read' in why
    edit_json(root/'analysis/decision_p2.json', lambda d: d.update(candidate='A', code=dict(d['code'], batch_code_sha='d')))
    choice, why = decision.p3_arm_choice(root)
    assert choice is None and 'written for the commit' in why
    # an entry failure without run.json: marked for the lead, the log's last lines in the reason (the review's minor 2)
    f = batch.field_cell('AC', core.PAIRS[4])
    (batch.run_dir(root, f)/'run.json').unlink()
    edit_json(batch.run_dir(root, f)/'queue_done.json', lambda d: d.update(state='failed'))
    (batch.run_dir(root, f)/'log.txt').write_text('a\nSystemExit: structure layer refused (ProvenanceRejected: no registry)\n')
    d, _ = decision.decide_p2(root)
    assert d['needs_lead_rejudgment'][0]['pair'] == core.short_name(core.PAIRS[4]) and 'ProvenanceRejected' in d['needs_lead_rejudgment'][0]['why'] and '要总负责人改判' in decision.summary_p2(d)
    # revision 4 (minor 2): a candidate whose repair process says passed and whose review certificate says not: a repair failure as before, and marked for the lead
    r = batch.repair_cell('A', core.PAIRS[3])
    write_repair(batch.run_dir(root, r), r, seg35=.861, certified=False)
    edit_json(batch.run_dir(root, r)/'repair.json', lambda x: x.update(passed=True))
    d, _ = decision.decide_p2(root)
    marked = [x for x in d['needs_lead_rejudgment'] if x['arm'] == 'A']
    assert d['arms']['A']['status'] == decision.NOT_THROUGH and len(marked) == 1 and marked[0]['pair'] == core.short_name(core.PAIRS[3]) and 'C22.3-9 (d)' in marked[0]['why']
    assert 'C22.3-9 (d)' in decision.summary_p2(d).split('\n## ')[0]


def test_decide_p3_on_records_with_a_fallback(tmp_path, monkeypatch):
    doc, path = reference_world(tmp_path, monkeypatch, seg35=.87)
    p2_root, root = tmp_path/'p2_v1', tmp_path/'p3_v1'
    write_json(p2_root/'batch.json', dict(code_sha='c', segmenter_id=p2.DEFAULT_SEGMENTER))
    write_json(p2_root/'analysis/decision_p2.json', dict(reading=decision.READ_CANDIDATE, candidate='C'))
    monkeypatch.setattr(p2.P3, 'DEFAULT_ROOT', root)
    choice, _ = decision.ensure_p3_arm(root, p2_root, recompute=False)                  # the recomputation has its own test (decide_p2 on records)
    assert choice['arm'] == 'C' and p2.P3.arm_of(root) == 'C'
    write_json(root/'batch.json', dict(code_sha='c', segmenter_id=p2.DEFAULT_SEGMENTER, reference=dict(sha256=core.sha_file(path))))
    batch = p2.P3
    for pair in p2.ALL_PAIRS:
        f, r = batch.field_cell('C', pair), batch.repair_cell('C', pair)
        write_field(batch.run_dir(root, f), f, passes=[dict(u_sha256='u'), dict(tissue=dict(ratios_used=dict(CSF=.8, GM=1.4))), dict(tissue=dict(ratios_used=dict(CSF=.82, GM=1.38)))])
        write_repair(batch.run_dir(root, r), r, seg35=.871, certified=pair != p2.ALL_PAIRS[7])
    failed = core.short_name(p2.ALL_PAIRS[7])
    d, _ = decision.decide_p3(root)
    assert d['reading'] == decision.UNDECIDABLE and '没有 fallback.json' in d['reason']    # the fallback unit has not decided yet
    fallback = root/decision.FALLBACK_DIR/failed/'fallback.json'
    write_json(fallback, dict(kind=decision.A0, a0_metrics=metrics_of(.84), why=None))
    d, rows = decision.decide_p3(root)
    assert [f['pair'] for f in d['fallbacks']] == [failed] and d['sources'][failed] == decision.A0
    n = d['numbers']
    assert n['mean_delta_seg35'] == pytest.approx((18*.001+(.84-.87))/19) and d['reading'] == decision.READ_NO_GAIN
    write_json(fallback, dict(kind=decision.IDENTITY, a0_metrics=None, why='no'))
    d, _ = decision.decide_p3(root)
    assert d['reading'] == decision.UNDECIDABLE                                          # the identity, its metrics not evaluated yet
    write_json(root/decision.ID_DIR/failed/'identity_metrics.json', dict(certified_at_1=True, oracle=False, metrics=metrics_of(.6)))
    d, _ = decision.decide_p3(root)
    assert d['sources'][failed] == decision.IDENTITY and d['reading'] == decision.READ_NO_GAIN
    write_json(root/decision.ID_DIR/failed/'identity_metrics.json', dict(certified_at_1=True, oracle=False, metrics=dict(metrics_of(.6), sdlogj_official=None)))
    d, _ = decision.decide_p3(root)
    assert d['reading'] == decision.UNDECIDABLE and 'sdlogj_official' in d['reason']   # the review's minor 8: no silent mean over 18 pairs
    with pytest.raises(p2.fl.SelectionChanged):
        (root/'analysis'/p2.P3.ARM_FILE).write_text('{"arm": "A"}')
        p2.P3.arm_of(root)


def test_p3_needs_a_candidate(tmp_path):
    assert decision.p3_arm_choice(tmp_path)[0] is None
    write_json(tmp_path/'batch.json', dict(code_sha='c'))
    for reading in (decision.READ_NONE, decision.UNDECIDABLE):
        write_json(tmp_path/'analysis/decision_p2.json', dict(reading=reading, candidate=None))
        choice, why = decision.p3_arm_choice(tmp_path)
        assert choice is None and reading in why


# ---- the firewall, and a synthetic pipeline of the three arms ------------------------------------------------------------------------------------------------
def test_no_label_member_in_the_source():
    for name in ('p2_core.py', 'run_p2.py', 'decide_p2.py'):
        text = (ROOT/'experiments/p2'/name).read_text()
        assert 'fixed_seg' not in text and 'moving_seg' not in text                          # the measure names seg35, seg4 are the evaluation's numbers
    tree = ast.parse((ROOT/'experiments/p2/run_p2.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    callers = {n.name for n in functions if 'read_truth' in ast.unparse(n)}
    assert callers == {'evaluate'}                                                       # the truth only in the evaluation unit
    loaders = {n.name for n in functions if 'np.load(' in ast.unparse(n)}
    assert loaders == {'fallback_one'} and "z['u']" in ast.unparse(next(n for n in functions if n.name == 'fallback_one'))     # only configuration C's field, its member u
    assert 'np.load(' not in (ROOT/'experiments/p2/p2_core.py').read_text() and 'np.load(' not in (ROOT/'experiments/p2/decide_p2.py').read_text()


@dataclass
class FakeConfig:
    delta: float = 0.


def fake_certify(passes):
    def certify(u, grid, config=None):
        ok = passes(u)
        return SimpleNamespace(status='certified_feasible' if ok else 'infeasible', reason=None, min_jd_lower_bound=1e-3 if ok else -1., n_cells_checked=8, n_infeasible=0 if ok else 3, n_uncertain=0, certificate_version='fake', config=FakeConfig())
    return certify


def test_the_fallback_unit_certifies_again_without_a_label(tmp_path, monkeypatch):
    """D75 as C22.5-2 says it (the lead on the review's minor 1): configuration C's field qualified by d0_v1's records, certified again here; else the identity, certified here too."""
    root = tmp_path/'p3_v1'
    u = smooth_field()
    field = tmp_path/'a0.npz'
    np.savez(field, u=u)
    grid = SimpleNamespace(shape=SHAPE)
    pin = SimpleNamespace(grid=grid, provenance=dict(keys_read=['fixed', 'moving', 'initial_u']))
    monkeypatch.setattr(p2.P3, 'read_images', lambda cell: pin)
    monkeypatch.setattr(RUN.rp.RP, 'a0_field', lambda pair, results=None: field)
    pinned = {'u_sha256': p2.sha_bytes(u)}
    monkeypatch.setattr(RUN.rp.RP, 'fallback_pins', lambda path=None: {core.short_name(p): pinned for p in p2.ALL_PAIRS})
    status = dict(qualified=True, why=None, metrics=metrics_of(.84))
    monkeypatch.setattr(RUN.rp, 'fallback_status', lambda pair, results=None: status)
    pairs = p2.ALL_PAIRS[:5]
    assert RUN.fallback_one(root, 'A', pairs[0], fake_certify(lambda x: True), None) == decision.A0
    record = json.loads((root/'fallback/0438-0439/fallback.json').read_text())
    assert record['kind'] == decision.A0 and record['a0_metrics']['seg35'] == .84 and record['a0']['certificate']['accepted_alpha'] == 1. and record['keys_read'] == ['fixed', 'moving', 'initial_u'] and 'identity' not in record
    assert RUN.fallback_one(root, 'A', pairs[0], fake_certify(lambda x: True), None) is None                      # once
    assert RUN.fallback_one(root, 'A', pairs[1], fake_certify(lambda x: not np.any(x)), None) == decision.IDENTITY   # the field does not pass again: the identity
    pinned['u_sha256'] = 'other'
    assert RUN.fallback_one(root, 'A', pairs[2], fake_certify(lambda x: True), None) == decision.IDENTITY          # not the pinned field
    status.update(qualified=False, why='no D0 record')
    assert RUN.fallback_one(root, 'A', pairs[3], fake_certify(lambda x: True), None) == decision.IDENTITY
    assert RUN.fallback_one(root, 'A', pairs[4], fake_certify(lambda x: False), None) is None
    assert json.loads((root/'fallback'/core.short_name(pairs[4])/'fallback.json').read_text())['kind'] is None
    assert RUN.identity_pairs(root) == list(pairs[1:4])


PIPELINE = """
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
import run_p2 as run
p2, core, mc, fast, channels = run.p2, run.core, run.mc, run.fast, run.channels
fz = core.frozen_api()
world = json.loads(sys.argv[2])
root = Path(world['root'])
p2.P2.DEFAULT_ROOT = root
core.STAGE = Path(world['stage'])
core.SETUP_WORKERS = 2
mc.SEG_ROOT = Path(world['seg_root'])
mi = core.make_inputs()
mc.default_unet = lambda: mi.UNetSource(pred_root=Path(world['pred']))
p2.NATIVE = 4
for spec in p2.ARMS.values():
    spec.update(native=4, checkpoints=(4,))
p2.P2.COARSE_ITERATIONS = (2, 2, 2)
p2.EM_SCHEDULE = dict(heads=(0, 1, 2, 3), native=(2, 3))
seen, real_load = set(), np.load
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
        seen.add(self.name+':'+key)
        if 'seg' in key:
            raise AssertionError('a label member was read: '+key)
        return self.inner[key]
    @property
    def files(self):
        return self.inner.files
np.load = lambda path, **kwargs: Spy(path, real_load(path, **kwargs))
pair = core.PAIRS[0]
for arm in ('A', 'C', 'AC'):
    cell = p2.P2.field_cell(arm, pair)
    assert run.main(['p2', 'run-one', '--arm', arm, '--tier', cell.tier, '--pair', pair, '--out', str(p2.P2.run_dir(root, cell)), '--device', 'cpu', '--code-sha', 'x']) == 0
np.load = real_load
assert {s for s in seen if s.startswith('pair.npz:')} <= {'pair.npz:fixed', 'pair.npz:moving', 'pair.npz:initial_u'}, seen
records = {arm: json.loads((p2.P2.run_dir(root, p2.P2.field_cell(arm, pair))/'run.json').read_text()) for arm in ('A', 'C', 'AC')}
for arm, r in records.items():
    assert r['status'] == 'done' and r['oracle'] is False and r['final_equals_last_checkpoint'] is True and r['scale_check']['passed'], (arm, r['status'], r['reason'])
    assert set(r['data']['keys_read']) <= {'fixed', 'moving', 'initial_u'} and r['segmenter_id'] == 'unet35_v1_ens'
assert len(records['A']['passes']) == 1 and len(records['C']['passes']) == 3 and len(records['AC']['passes']) == 3
em = records['A']['passes'][0]['em']
assert [(s['where']['level'], s['where']['step'], s['used']) for s in em['steps']] == [(0, 0, True), (1, 0, True), (2, 0, True), (3, 0, True), (3, 2, True), (3, 3, True), (3, 600, False)]
assert em['gram_check']['passed'] and all(c['passed'] for c in em['native_scale_checks'])
c = records['C']['passes']
assert c[0]['tissue']['ratios_used'] is None and c[1]['tissue']['ratios_used'] == c[0]['estimate']['ratios'] and c[2]['tissue']['ratios_used'] == c[1]['estimate']['ratios'] and 'estimate' in c[2]
assert all(abs(l['mean_brain']-1) < 1e-12 for l in c[1]['tissue']['levels']) and len(c[1]['tissue']['levels']) == 4
assert records['AC']['passes'][0]['em'] and records['AC']['passes'][2]['tissue']['ratios_used'] == records['AC']['passes'][1]['estimate']['ratios']
# C's pass 0 is the frozen Hx24 path itself, bit for bit
cell = p2.P2.field_cell('C', pair)
pin = p2.P2.read_images(cell)
predicted = fast.read_predicted(pin, workers=2)
stacks, scale = channels.prepare_stacks(p2.ARMS['C'], pin, predicted, core.load_table(), 'cpu', workers=2)
config = fz.PreoptConfig(**p2.P2.arm_config(cell, .5))
u, _ = core.optimize_k(stacks, pin.grid, config, device='cpu', workers=2, record_level_norms=True, downsample=fast.FastDownsample())
assert records['C']['passes'][0]['u_sha256'] == hashlib.sha256(u.tobytes()).hexdigest()
assert records['AC']['passes'][0]['u_sha256'] != records['C']['passes'][0]['u_sha256'] != records['C']['passes'][2]['u_sha256']
# the control of pass 0 in the unit (the review's minor 9): this synthetic pair is not the reference's real field, so it says so and keeps the field
control = records['C']['passes'][0]['control']
assert control['same'] is False and control['field_kept'] is True and (p2.P2.run_dir(root, cell)/'field_pass0.npz').is_file()
with np.load(p2.P2.run_dir(root, cell)/'field_pass0.npz') as z:
    assert hashlib.sha256(np.ascontiguousarray(z['u']).tobytes()).hexdigest() == records['C']['passes'][0]['u_sha256']
assert 'field_pass0.npz' in run.c_control_problem(p2.P2, cell, records['C'])
assert [m['step'] for m in records['A']['passes'][0]['em']['native_marks']] == [0, 2, 3]
# a run is never rerun
try:
    run.main(['p2', 'run-one', '--arm', 'A', '--tier', 'first5', '--pair', pair, '--out', str(p2.P2.run_dir(root, p2.P2.field_cell('A', pair))), '--device', 'cpu'])
    raise AssertionError('a run ran twice')
except SystemExit as stop:
    assert 'never rerun' in str(stop)
print('done')
"""


@needs_frozen
def test_three_arms_through_the_field_unit(tmp_path, monkeypatch):
    import nibabel as nib
    from jd_curl.three_d.grid import make_grid_3d
    from jd_curl.three_d.nifti import save_scalar
    from ..solve import test_d0_a4_tools as a4
    from ..solve.test_ef_tools import pipeline_world
    world = pipeline_world(tmp_path, monkeypatch)
    world['root'] = str(tmp_path/'p2_v1')
    grid = make_grid_3d(nx=30, ny=28, nz=26, hx=1., hy=1., hz=1., unit='mm')
    for subject in core.PAIRS[0].split('__'):                                           # as atropos_seg.py writes them: units mm, qform and sform (the tissue layer's reader requires them; P1's test does the same)
        directory = Path(world['seg_root'])/subject
        for name in a4.ATROPOS_FILES:
            p = np.asarray(nib.load(str(directory/name)).get_fdata()).transpose(2, 1, 0)
            (directory/name).unlink()
            save_scalar(directory/name, p, grid)
        a4.rehash(directory)
    done = subprocess.run([PY, '-B', '-c', PIPELINE, str(ROOT/'experiments/p2'), json.dumps(world)], env=ENV, cwd=tmp_path, capture_output=True, text=True, timeout=280)
    assert done.returncode == 0 and 'done' in done.stdout, done.stdout[-3000:]+done.stderr[-4000:]


# ---- the reference, HS, C's control -------------------------------------------------------------------------------------------------------------------------------
def test_the_committed_reference_is_what_the_records_say():
    if not (p2.RESULTS/'fd2_v1').is_dir() or not (p2.RESULTS/'rw_v1').is_dir():
        pytest.skip('fd2_v1, rw_v1 not on this machine')
    document = p2.load_reference(p2.DEFAULT_SEGMENTER)
    assert RUN.v1_reference() == document
    assert all(pin['certified_at_1'] and pin['field_D_factor'] == 24. for stage in ('p2', 'p3') for pin in document[stage]['pairs'].values())
    assert {pin['wall_seconds'] for stage in ('p2', 'p3') for pin in document[stage]['pairs'].values()} == {1200.}               # the fuse of the three arms (the lead, 11-06)
    assert all(document['p2']['pairs'][k] == document['p3']['pairs'][k] and '/rw_v1/' in document['p2']['pairs'][k]['repair_dir'] for k in document['p2']['pairs'])


def test_load_reference_refusals(tmp_path):
    good = dict(schema=p2.REFERENCE_SCHEMA, segmenter_id='seg_x', channels=dict(pred_root='p', models_root='m', image_root='i', registry_rel='configs/x.json'),
                p2=dict(pairs={core.short_name(p): {} for p in core.PAIRS}), p3=dict(pairs={core.short_name(p): {} for p in p2.ALL_PAIRS}))
    path = write_json(tmp_path/'r.json', good)
    assert p2.load_reference('seg_x', path)['segmenter_id'] == 'seg_x'
    with pytest.raises(SystemExit):
        p2.load_reference('seg_y', path)
    with pytest.raises(SystemExit):
        p2.load_reference('seg_x', tmp_path/'missing.json')
    first = core.short_name(core.PAIRS[0])
    for change in (dict(channels={}), dict(p2=dict(pairs={})), dict(schema='other'), dict(p2=dict(pairs=dict(good['p2']['pairs'], **{first: dict(tier='c600')})))):
        with pytest.raises(SystemExit):
            p2.load_reference('seg_x', write_json(tmp_path/'bad.json', dict(good, **change)))
    batch = p2.P2Batch()
    batch.use_segmenter('seg_x', p2.load_reference('seg_x', path))
    assert batch.channels.segmenter_id == 'seg_x' and not batch.channels.is_default and batch.reference['segmenter_id'] == 'seg_x'


def test_channel_set_marks_its_own_segmenter():
    sides = {s: dict(segmenter_id='seg_x', policy='all_folds_mean') for s in ('fixed', 'moving')}
    keys = ['fixed', 'moving', 'initial_u']
    assert p2.ChannelSet('seg_x').mark(keys, sides) == dict(oracle=False, truth_labels_read=False)
    assert p2.ChannelSet(p2.DEFAULT_SEGMENTER).mark(keys, sides)['oracle'] is True
    assert p2.ChannelSet('seg_x').mark(keys+['fixed_seg35'], sides) == dict(oracle=True, truth_labels_read=True)
    record = dict(oracle=False, channel_sources=dict(pair_keys_read=keys, sides=sides))
    assert p2.ChannelSet('seg_x').oracle_of_run(record) is False
    with pytest.raises(ValueError):
        p2.ChannelSet(p2.DEFAULT_SEGMENTER).oracle_of_run(record)


def test_hs_reading_names_the_reference_configuration(tmp_path):
    path = tmp_path/'hs/analysis/decision_hs.json'
    assert RUN.hs_problem(p2.DEFAULT_SEGMENTER, path) is None
    assert RUN.hs_problem('seg_x', path) is not None
    write_json(path, dict(reading='不采用', segmenter_id='seg_x'))
    assert RUN.hs_problem(p2.DEFAULT_SEGMENTER, path) is None
    write_json(path, dict(reading=RUN.HS_ADOPT, segmenter_id='seg_x'))
    assert RUN.hs_problem('seg_x', path) is None and RUN.hs_problem(p2.DEFAULT_SEGMENTER, path) is not None and RUN.hs_problem('seg_y', path) is not None
    write_json(tmp_path/'hs/analysis/other.json', dict(reading=RUN.HS_ADOPT, segmenter_id='seg_y'))       # the lead on I2: only decision_hs.json is read
    assert RUN.hs_problem('seg_x', path) is None
    assert p2.HS_DECISION == p2.RESULTS/'hs_v1/analysis/decision_hs.json'


def test_the_collapse_flag_compares_rounds_of_one_level_only():
    """C11-5, D42 (the review's minor 3): across a level head the ratio is recorded, not flagged."""
    L, Mv = label_maps()
    _, _, host, scale = prepared(L, Mv)
    em = p2.EMWeights(host, scale, brain_of(), SPACING, 'cpu')
    first = em.m_step(dict(level=0, step=0, at='head'), None)
    sigma2 = np.array([np.nan if v is None else v for v in first['sigma2']])
    em.sigma2 = sigma2*100
    row = em.m_step(dict(level=1, step=0, at='head'), None)
    assert row['collapse_flagged'] == [] and row['max_drop_from_previous'] == pytest.approx(100)
    em.sigma2, em.level = sigma2*100, 3
    row = em.m_step(dict(level=3, step=200, at='native'), None)
    assert len(row['collapse_flagged']) == int(np.isfinite(sigma2).sum())


def test_a_restart_with_another_segmenter_or_reference_stops(tmp_path, monkeypatch):
    batch = p2.P2Batch()
    path = write_json(tmp_path/'ref.json', dict(x=1))
    monkeypatch.setattr(p2, 'reference_path', lambda segmenter: path)
    o = SimpleNamespace(gap_seconds=5)
    gate = RUN.BASES['p2'].GATE
    for existing in (dict(code_sha='c', plan=[], segmenter_id='seg_x', reference=dict(sha256=core.sha_file(path))), dict(code_sha='c', plan=[], segmenter_id=p2.DEFAULT_SEGMENTER, reference=dict(sha256='other'))):
        write_json(tmp_path/'p2_v1/batch.json', existing)
        with pytest.raises(RUN.run.Stop) as stop:
            RUN.write_batch(RUN.BASES['p2'], batch, o, tmp_path/'p2_v1', 'c', gate, [])
        assert 'this start has' in str(stop.value)


def test_c_control_problem():
    batch = p2.P2Batch()
    batch.reference = dict(p2=dict(pairs={core.short_name(p): dict(field_u_sha256='abc') for p in core.PAIRS}))
    cell = batch.field_cell('C', core.PAIRS[0])
    assert RUN.c_control_problem(batch, cell, dict(status='done', passes=[dict(u_sha256='abc')])) is None
    assert 'not the reference' in RUN.c_control_problem(batch, cell, dict(status='done', passes=[dict(u_sha256='abd')]))
    assert RUN.c_control_problem(batch, cell, dict(status='done', passes=[])) is not None
    assert RUN.c_control_problem(batch, cell, dict(status='time_cap')) is None
    assert 'not the reference' in RUN.c_control_problem(batch, cell, dict(status='time_cap', passes=[dict(u_sha256='abd')]))     # the review's minor 9: whatever the status
    assert 'field_pass0.npz' in RUN.c_control_problem(batch, cell, dict(status='estimate_failed', passes=[dict(u_sha256='abd', control=dict(field_kept=True))]))
    # revision 3: pass 0 cut by the wall (the arm's own K = 35 rule stops C), a reference row without a T88 field: recorded, the queue goes on
    assert RUN.c_control_problem(batch, cell, dict(status='time_cap', passes=[dict(u_sha256='abd', stopped=dict(level=3, iteration=10))])) is None
    assert RUN.pass0_control(batch, cell, dict(u_sha256='abd', stopped=dict(level=3)))['why'] == RUN.PASS0_TRUNCATED
    batch.reference['p2']['pairs'][cell.short] = dict(field_u_sha256=None, hs_line_status=p2.HX_IDENTITY)
    assert RUN.c_control_problem(batch, cell, dict(status='done', passes=[dict(u_sha256='abd')])) is None
    assert RUN.pass0_control(batch, cell, dict(u_sha256='abd', stopped=None))['why'] == RUN.NO_REFERENCE_FIELD
    batch.reference['p2']['pairs'][cell.short] = dict(field_u_sha256=None)                  # a certified row without its field: the start refuses it; here it is not checkable as the same
    assert RUN.c_control_problem(batch, cell, dict(status='done', passes=[dict(u_sha256='abd')])) is not None
    assert RUN.c_control_problem(batch, batch.field_cell('A', core.PAIRS[0]), dict(status='done', passes=[dict(u_sha256='x')])) is None
    assert RUN.c_control_problem(batch, batch.repair_cell('C', core.PAIRS[0]), dict(status='done')) is None


# ---- the cells, the plan, the start strings -------------------------------------------------------------------------------------------------------------
def test_cells_and_plans():
    P2, P3 = p2.P2, p2.P3
    fields = P2.plan_fields()
    assert len(fields) == 15 and {c.arm for c in fields} == set(p2.ORDER) and [c.arm for c in fields[:3]] == ['AC', 'A', 'C'] and [c.arm for c in fields[3:6]] == ['A', 'C', 'AC']
    assert all(c.pair in core.PAIRS and c.tier == 'first5' for c in fields) and fields[0].unit == 'tag-p2-ac-first5-0438-0439' and fields[0].run_id == 'p2_AC_first5_0438-0439'
    repairs = P2.plan_repairs()
    assert [(c.arm, c.tier) for c in repairs[:2]] == [('RAC', 'c600'), ('RA', 'c600')] and all(c.is_repair and c.base in p2.ORDER for c in repairs)
    for arm, tier, pair in (('A', 'first5', p2.ALL_PAIRS[6]), ('A', 'rest600', core.PAIRS[0]), ('Hx24', 'first5', core.PAIRS[0]), ('RA', 'first5', core.PAIRS[0]), ('A', 'first5', 'OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1')):
        with pytest.raises(SystemExit):
            P2.require_cell(arm, tier, pair)
    assert P3.require_cell('C', 'rest600', p2.ALL_PAIRS[10]).tier == 'rest600' and P3.require_cell('RC', 'c600', core.PAIRS[1]).base == 'C'
    with pytest.raises(SystemExit):
        P3.require_cell('C', 'first5', p2.ALL_PAIRS[10])
    assert len(P3.plan_fields_of('A')) == 19 and P3.plan_repairs_of('A')[-1].tier == 'rest600'
    assert P2.steps_of(fields[0]) == 600 and P2.steps_of(repairs[0]) == 600 and P3.steps_of(P3.repair_cell('A', p2.ALL_PAIRS[-1])) == 600
    config = P2.arm_config(fields[0], .5)
    assert config['iterations'] == (50, 50, 100, 600) and config['sigma_field'] == .5 and config['seconds'] == 900.


def test_p3_all_cells_follow_the_choice(tmp_path):
    batch = p2.P3Batch()
    assert batch.all_cells(tmp_path) == [] and batch.chosen_repairs(tmp_path) == []
    batch.write_selection(tmp_path, batch.ARM_FILE, dict(arm='AC'))
    cells = batch.all_cells(tmp_path)
    assert len(cells) == 38 and {c.base for c in cells} == {'AC'}


def test_start_strings_and_roots(tmp_path):
    for name in ('p2', 'p3'):
        base, batch = RUN.BASES[name], p2.BATCHES[name]
        cells = batch.plan_fields() if name == 'p2' else batch.plan_fields_of('A')
        argv = base.run_argv(cells[0], Path('/c'), batch.DEFAULT_ROOT, 's')
        text = ' '.join(map(str, argv))
        assert 'pytest' not in text and f'{RUN.SCRIPT} {name} run-one' in text and f'--unit={cells[0].unit}' in text and 'PYTHONPATH=/home/rmcfz/seq_vp_tag_results/frozen_eb808be:/c' in text
        repair = base.run_argv(batch.repair_cell(cells[0].arm, cells[0].pair), Path('/c'), batch.DEFAULT_ROOT, 's')
        assert 'RuntimeMaxSec=1800' in ' '.join(repair) and 'repair-one' in repair
        assert f'tag-{name}-eval' in ' '.join(base.eval_argv(Path('/c'), batch.DEFAULT_ROOT)) and 'CUDA_VISIBLE_DEVICES=' in ' '.join(base.eval_argv(Path('/c'), batch.DEFAULT_ROOT))
        assert 'pytest' not in ' '.join(RUN.queue_launch(name))
        assert batch.root_problem(tmp_path/f'{name}_v1') is not None and batch.root_problem(batch.DEFAULT_ROOT) is None
        o = SimpleNamespace(root=str(tmp_path/f'{name}_v1'))
        assert RUN.queue(name, o) == 2 and not (tmp_path/f'{name}_v1').exists()
    assert RUN.BASES['p2'].repair.RepairConfig(eps=.05).wall_seconds == 1200.


def test_the_queue_refuses_p3_without_p2_and_at_another_commit(tmp_path, monkeypatch):
    class Machine:
        def code_checkout(self, root, commit):
            raise AssertionError('checked out')
    monkeypatch.setattr(p2.P2, 'DEFAULT_ROOT', tmp_path/'p2_v1')
    logs = []
    o = SimpleNamespace(root=str(tmp_path/'p3_v1'), commit='HEAD', segmenter=None, min_host_gib=15., gap_seconds=5, wait_minutes=1)
    with pytest.raises(RUN.run.Stop) as stop:
        RUN.p3_body(o, Machine(), tmp_path/'p3_v1', logs.append)
    assert '等 P2' in str(stop.value)
    write_json(tmp_path/'p2_v1/batch.json', dict(code_sha='0'*40, segmenter_id=p2.DEFAULT_SEGMENTER))
    write_json(tmp_path/'p2_v1/analysis/decision_p2.json', dict(reading=decision.READ_CANDIDATE, candidate='A', code=dict(batch_code_sha='0'*40)))
    with pytest.raises(RUN.run.Stop) as stop:
        RUN.p3_body(o, Machine(), tmp_path/'p3_v1', logs.append)
    assert "the reading's rules would not be the batch's" in str(stop.value)              # P2's reading is computed again with this process's files: they must be P2's commit
    monkeypatch.setattr(RUN, 'reading_code_problem', lambda base, root: None)
    monkeypatch.setattr(decision, 'decide_p2', lambda root, stops=None: (dict(reading=decision.READ_CANDIDATE, candidate='A'), None))
    with pytest.raises(RUN.run.Stop) as stop:
        RUN.p3_body(o, Machine(), tmp_path/'p3_v1', logs.append)
    assert "P2's commit" in str(stop.value)
    monkeypatch.setattr(decision, 'decide_p2', lambda root, stops=None: (dict(reading=decision.READ_NONE, candidate=None), None))
    with pytest.raises(RUN.run.Stop) as stop:
        RUN.p3_body(o, Machine(), tmp_path/'p3_v1', logs.append)
    assert 'now read' in str(stop.value)
