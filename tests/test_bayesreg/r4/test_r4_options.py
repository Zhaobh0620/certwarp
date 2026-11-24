"""R4 的 T88 选项（合同 C25.1–C25.5，C25.10-2、-3）：设定、核表、A 的掩膜与 run_level_k_masked、B 的软通道、关着时逐位相同；复制的函数做 AST 对照。
CPU，合成小体积；需要冻结树的部分（mc0_core.frozen_api）在子进程里跑，PYTHONPATH = <frozen>:<repo>。"""
import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, gaussian_filter

ROOT = Path(__file__).resolve().parents[3]
for _p in ('r4', 'he', 'mc0', 'ef', 'fl', 'r2'):
    p = str(ROOT/'experiments'/_p)
    if p not in sys.path:
        sys.path.insert(0, p)

import r4_common as rc  # noqa: E402
import r4_options as r4o  # noqa: E402

PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')


def function_def(module, name):
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(name)


def body_dump(node, replace=None):
    """函数体的 AST（文档串之后），replace(stmt) 返回 True 的语句换成占位。"""
    class Swap(ast.NodeTransformer):
        def generic_visit(self, n):
            n = super().generic_visit(n)
            return ast.Expr(ast.Constant('<replaced>')) if replace is not None and isinstance(n, ast.stmt) and replace(n) else n
    body = [Swap().visit(ast.parse(ast.unparse(s)).body[0]) for s in node.body]
    return ast.dump(ast.Module(body, []))


# ---- 设定（C25.1-1、C25.4、C25.5） ----------------------------------------------------------------------------------------------------------------
def test_constants_equal_the_code_they_come_from():
    import fx_core
    from importlib import util
    spec = util.spec_from_file_location('_pq_stats_r4', ROOT/'experiments/pq/pq_stats.py')
    pq = util.module_from_spec(spec)
    spec.loader.exec_module(pq)
    assert rc.SPEC == fx_core.SPEC and rc.SMALL_SIX == pq.SMALL_SIX == (4, 23, 18, 34, 19, 35)
    import r2_common
    assert (rc.HS_REFERENCE_REL, rc.HS_REFERENCE_SHA256, rc.HS_DECISION_SHA256) == (r2_common.HS_REFERENCE_REL, r2_common.HS_REFERENCE_SHA256, r2_common.HS_DECISION_SHA256)
    import he_core
    assert rc.HS_REGISTRY_SHA256 == he_core.CONTRACT_SEGMENTER['registry_sha256'] and (rc.HX, rc.FALLBACK_C, rc.FALLBACK_ID, rc.REJECTED, rc.IMPLEMENTATION) == he_core.LINE_STATUSES


def test_arm_specs():
    assert rc.arm_spec('CTL') == rc.SPEC
    assert rc.arm_spec('A1')['sigma_small'] == .42 and rc.arm_spec('A2')['sigma_small'] == .35 and rc.arm_spec('A1')['field_mask'] == 'small6'
    assert rc.arm_spec('B')['channels'] == 'soft' and rc.arm_spec('C1')['D_factor'] == 32. and rc.arm_spec('C3')['sigma_field'] == .65
    c2 = rc.arm_spec('C2')
    assert c2['native'] == 1200 and c2['checkpoints'] == (1200,) and rc.field_file(c2) == 'field_1200.npz' and rc.field_file(rc.SPEC) == 'field_0600.npz'
    combo = rc.arm_spec('COMBO-C1-C2')
    assert combo['D_factor'] == 32. and combo['native'] == 1200
    assert rc.options_of(rc.arm_spec('COMBO-A1-C3')) == ['A1', 'C3'] and rc.options_of(rc.SPEC) == []
    assert rc.spec_sha256(rc.arm_spec('B')) == rc.spec_sha256(dict(rc.arm_spec('B'))) != rc.spec_sha256(rc.SPEC)
    with pytest.raises(ValueError):
        rc.arm_spec('COMBO-A2-A1')
    with pytest.raises(ValueError):
        rc.arm_spec('D')


def test_kernel_table_is_the_contracts_and_mc0s_kernel():
    import torch
    import mc0_core
    table = r4o.kernel_table()
    want = {.25: (1, .9993, .00067, .9987), .35: (2, .9673, .03266, .9347), .42: (2, .8948, .10523, .7897), .5: (2, .7866, .21501, .5742), .65: (2, .6135, .41889, .2485)}
    for s, (radius, centre, variance, gain) in want.items():
        row = table[s]
        assert row['radius'] == radius and abs(row['centre']-centre) < 5e-5 and abs(row['variance']-variance) < 5e-6 and abs(row['checkerboard']-gain) < 5e-5
        k = mc0_core._gaussian_1d(s, 'cpu', torch.float64).numpy()
        assert len(k) == 2*radius+1 and abs(k[radius]-row['centre']) < 1e-12


# ---- A：掩膜（C25.2-1、-2；C25.10-3） -------------------------------------------------------------------------------------------------------------
def labels(shape=(34, 36, 38), seed=0, with_small=True):
    rng = np.random.default_rng(seed)
    L = rng.choice([0, 2, 3, 41 % 36, 17], size=shape).astype(np.int16)
    if with_small:
        L[10:12, 10:13, 9:11] = 18
        L[20, 20, 20] = 4
    return L


def test_ball_has_33_offsets_and_the_mask_matches_a_numpy_reference():
    assert int(r4o.ball().sum()) == 33
    F, M = labels(seed=1), labels(seed=2, with_small=False)
    M[5, 6, 7] = 35
    r = np.arange(-2, 3)
    z, y, x = np.meshgrid(r, r, r, indexing='ij')
    B = np.isin(F, (4, 23, 18, 34, 19, 35)) | np.isin(M, (4, 23, 18, 34, 19, 35))
    ref = np.clip(gaussian_filter(binary_dilation(B, structure=(z*z+y*y+x*x) <= 4, iterations=1, border_value=0).astype(np.float64), sigma=1.0, mode='nearest', truncate=4.0), 0, 1)
    got = r4o.native_mask(F, M)
    assert got.dtype == np.float64 and np.array_equal(got, ref)
    assert np.array_equal(r4o.native_mask(labels(with_small=False), labels(seed=3, with_small=False)), np.zeros((34, 36, 38)))     # 没有小六个：掩膜 ≡ 0，不是错误


def test_mask_function_takes_only_two_label_arrays():
    assert list(inspect.signature(r4o.native_mask).parameters) == ['fixed_labels', 'moving_labels']
    with pytest.raises(r4o.MaskError):
        r4o.native_mask(np.zeros((4, 4, 4)), np.zeros((4, 4, 4), dtype=np.int16))           # 类型不对：实现错误
    with pytest.raises(r4o.MaskError):
        r4o.native_mask(np.zeros((4, 4, 4), dtype=np.int16), np.zeros((4, 4, 5), dtype=np.int16))


def pyramid():
    """(原生网格, 四层)：粗层由 nifti.coarse_grid 给（冻结树的 four_level_grids 在子进程的测试里用）。"""
    from jd_curl.three_d.grid import make_grid_3d
    from jd_curl.three_d.nifti import coarse_grid
    grid = make_grid_3d(nx=38, ny=36, nz=34, hx=1., hy=1., hz=1., unit='mm')
    return grid, [coarse_grid(grid, s) for s in ((5, 5, 6), (9, 10, 10), (17, 18, 19))]+[grid]


def test_level_masks_are_downsample_array_then_clip_and_interior_shapes():
    from jd_curl.three_d.nifti import downsample_array
    grid, grids = pyramid()
    m3 = r4o.native_mask(labels(seed=1), labels(seed=2))
    full, interior, record = r4o.level_masks(m3, grid, grids)
    for level in range(3):
        assert np.array_equal(full[level], np.clip(downsample_array(m3, grid, grids[level].shape)[0], 0, 1))
    for level in range(4):
        assert interior[level].dtype == np.float32 and interior[level].shape == tuple(n-2 for n in grids[level].shape)
    assert record['n_mask'] == int((m3[1:-1, 1:-1, 1:-1] > 0).sum()) and record['dilation']['offsets'] == 33


def test_fast_downsample_equals_downsample_array_on_a_dense_soft_channel():
    import fast_channels as fast
    from jd_curl.three_d.nifti import downsample_array
    grid, grids = pyramid()
    dense = np.random.default_rng(5).uniform(1e-3, 1, (34, 36, 38))
    for g in grids[:3]:
        assert np.array_equal(fast.FastDownsample()(dense, grid, g.shape)[0], downsample_array(dense, grid, g.shape)[0])


def test_the_mixture_is_the_contracts_torch_expression():
    import torch
    import mc0_core
    rng = np.random.default_rng(3)
    u = torch.tensor(rng.normal(size=(3, 6, 7, 8)), dtype=torch.float32)
    mask = torch.tensor(np.clip(rng.uniform(-.2, 1.2, (6, 7, 8)), 0, 1), dtype=torch.float32)
    runner = r4o.masked_level_runner(mc0_core, {0: mask.numpy()}, .42)
    a, b = mc0_core._smooth(u, .5), mc0_core._smooth(u, .42)
    want = (1-mask)*a+mask*b
    ref64 = (1-mask.double())*a.double()+mask.double()*b.double()
    assert torch.allclose(want.double(), ref64, rtol=0, atol=1e-6)
    # m ≡ 1 → S_σs；m ≡ 0 → S_σf（数学核对）
    one, zero = torch.ones((6, 7, 8)), torch.zeros((6, 7, 8))
    assert torch.equal((1-one)*a+one*b, b) and torch.equal((1-zero)*a+zero*b, a)
    assert runner.__name__ == 'run_level_k_masked'


def test_run_level_k_masked_differs_from_run_level_k_only_in_the_field_smoothing():
    import mc0_core
    original = function_def(mc0_core, 'run_level_k')
    copy = function_def(r4o, 'run_level_k_masked')
    is_smoothing = lambda s: isinstance(s, ast.If) and ast.unparse(s.test) == 'config.sigma_field > 0'      # noqa: E731
    assert [a.arg for a in copy.args.args] == [a.arg for a in original.args.args]
    assert body_dump(copy, is_smoothing) == body_dump(original, is_smoothing)
    replaced = [s for s in ast.walk(copy) if isinstance(s, ast.If) and is_smoothing(s)][0]
    assert [ast.unparse(s) for s in replaced.body] == ['(mask, keep) = mask_of(level, u_int)', 'a = _smooth(u_int, config.sigma_field)', 'b = _smooth(u_int, sigma_small)',
                                                       'u_int.copy_(keep * a + mask * b)']
    # tests/test_bayesreg/solve/test_mc0_tools.py:294 keeps run_level_k = the frozen loop; this package never edits mc0_core
    assert 'def run_level_k(u_int, m, v, F, M, h, level, config, started, observer, counters)' in Path(mc0_core.__file__).read_text()


def test_load_copy_and_replace_names_are_r2s():
    for name in ('load_copy', 'replace_names'):
        theirs = function_def(_r2f(), name)
        assert body_dump(function_def(r4o, name)) == body_dump(theirs)


def _r2f():
    class M:
        __file__ = str(ROOT/'experiments/r2/r2f_core.py')
    return M


def test_core_for_installs_nothing_when_a_is_off():
    import mc0_core
    assert r4o.core_for(rc.arm_spec('CTL')) is mc0_core and r4o.core_for(rc.arm_spec('B')) is mc0_core and r4o.core_for(rc.arm_spec('C3')) is mc0_core
    with pytest.raises(ValueError):
        r4o.core_for(dict(rc.SPEC, sigma_small=.42))
    masks = {k: np.zeros((2, 2, 2), dtype=np.float32) for k in range(4)}
    copy = r4o.core_for(rc.arm_spec('A1'), masks)
    assert copy is not mc0_core and copy.run_level_k.__name__ == 'run_level_k_masked' and mc0_core.run_level_k.__name__ == 'run_level_k'
    with pytest.raises(ValueError):
        r4o.core_for(dict(rc.arm_spec('A1'), sigma_small=.25), masks)


# ---- B：软通道（C25.3；C25.10-3） ----------------------------------------------------------------------------------------------------------------
def probs_file(tmp_path, L, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, .1, (36,)+L.shape)
    p[(L.astype(int),)+tuple(np.indices(L.shape))] += 1.
    p /= p.sum(axis=0)
    path = tmp_path/'probs36.npz'
    np.savez_compressed(path, p=p.astype(np.float16))
    return path


def test_soft_side_checks_the_sha_and_the_argmax_and_is_mc0ps_soft_channels(tmp_path):
    import mc0p_core as mc
    from bayesreg.seg.channels import read_back_probabilities
    L = np.random.default_rng(1).integers(0, 36, (8, 9, 10)).astype(np.int16)
    path = probs_file(tmp_path, L)
    P, _ = read_back_probabilities(path, shape=(36, 8, 9, 10))
    fast_L = mc.predicted_labels(P)
    stack, record = r4o.soft_side(path, rc.sha256_file(path), (8, 9, 10), fast_L)
    P2, _ = read_back_probabilities(path, shape=(36, 8, 9, 10))
    assert np.array_equal(stack, mc.soft_channels(P2)) and record['argmax_consistent'] is True
    wrong = fast_L.copy()
    wrong[0, 0, 0] = (wrong[0, 0, 0]+1) % 36
    with pytest.raises(r4o.SoftArgmaxMismatch):
        r4o.soft_side(path, rc.sha256_file(path), (8, 9, 10), wrong)
    with pytest.raises(r4o.SoftArgmaxMismatch):
        r4o.soft_side(path, '0'*64, (8, 9, 10), fast_L)


class Pin:
    def __init__(self, fixed, moving):
        self.fixed, self.moving = fixed, moving


class Predicted:
    def __init__(self, labels):
        self.labels, self.record = labels, {}


def test_soft_branch_keeps_q0_weights_and_scale_of_the_onehot_path():
    import fast_channels as fast
    import fl_channels
    import mc0_core
    import mc0p_core as mc
    rng = np.random.default_rng(2)
    shape = (12, 13, 14)
    L = {s: rng.choice([0, 2, 3, 41, 17, 18], size=shape).astype(np.int16) % 36 for s in ('fixed', 'moving')}
    for s in L:
        L[s][3:6, 3:6, 3:6] = 1                                                               # cerebral white matter, seg35 label 1 (Q0 needs it)
    pin = Pin(rng.uniform(.1, 1, shape), rng.uniform(.1, 1, shape))
    table = mc0_core.load_table()
    spec = dict(rc.SPEC)
    ref, ref_rec = fl_channels.prepare_stacks(spec, pin, Predicted(L), table, 'cpu')
    soft = {s: fast.onehot_channels(L[s]) for s in L}
    got, rec = r4o.prepare_stacks_soft(dict(spec, channels='soft'), pin, Predicted(L), table, 'cpu', soft)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))
    assert {k: v for k, v in rec.items()} == {k: v for k, v in ref_rec.items()}
    w, voxels = mc.channel_weights('Q3h', L['fixed'])
    assert rec['channel_weights'] == w.tolist() and rec['fixed_voxels'] == voxels.tolist()
    assert r4o.prepare(spec, pin, Predicted(L), table, 'cpu')[1]['D_multiplier'] == ref_rec['D_multiplier']


# ---- 关着时逐位相同，CPU 合成小体积（C25.10-2 第一条） -------------------------------------------------------------------------------------------------
T88_SCRIPT = r'''
import json, sys
import numpy as np
sys.path[:0] = [sys.argv[1]+'/experiments/r4', sys.argv[1]+'/experiments/mc0', sys.argv[1]+'/experiments/ef', sys.argv[1]+'/experiments/fl']
import mc0_core as core
fz = core.frozen_api()
import r4_options as r4o
import r4_common as rc
import fast_channels as fast
from jd_curl.three_d.grid import make_grid_3d
grid = make_grid_3d(nx=38, ny=36, nz=34, hx=1., hy=1., hz=1., unit='mm')
rng = np.random.default_rng(0)
from scipy.ndimage import gaussian_filter
def stacks():
    r = np.random.default_rng(0)
    return [np.stack([gaussian_filter(r.uniform(0, 1, grid.shape), 2.) for _ in range(2)]) for _ in range(2)]
pre = dict(core.PRE_C)
pre.update(iterations=[50, 50, 100, 600], sigma_field=.5, seconds=900.)
out = {}
for name, spec in (('ctl', rc.arm_spec('CTL')), ('a1', rc.arm_spec('A1'))):
    small = dict(spec, native=3)
    config, _ = r4o.preopt_config(fz, pre, small)
    config = config.__class__(**dict(config.__dict__, iterations=(2, 2, 2, 3)))
    if name == 'ctl':
        engine = r4o.core_for(spec)
        assert engine is core
        u0, _ = core.optimize_k(stacks(), grid, config, device='cpu', downsample=fast.FastDownsample())
        u1, _ = engine.optimize_k(stacks(), grid, config, device='cpu', downsample=fast.FastDownsample())
        out['ctl_same'] = bool(np.array_equal(u0, u1))
        out['ref'] = u0
    else:
        L = np.zeros(grid.shape, dtype=np.int16)
        _, zero, rec = r4o.level_masks(r4o.native_mask(L, L), grid, fz.four_level_grids(grid))
        uz, _ = r4o.core_for(spec, zero).optimize_k(stacks(), grid, config, device='cpu', downsample=fast.FastDownsample())
        out['zero_mask_same'] = bool(np.array_equal(uz, out['ref']))
        L[10:14, 10:14, 10:14] = 18
        _, masks, rec = r4o.level_masks(r4o.native_mask(L, L), grid, fz.four_level_grids(grid))
        um, _ = r4o.core_for(spec, masks).optimize_k(stacks(), grid, config, device='cpu', downsample=fast.FastDownsample())
        out['mask_changes'] = bool(not np.array_equal(um, out['ref']))
        out['n_mask'] = rec['n_mask']
out.pop('ref')
print(json.dumps(out))
'''


@pytest.mark.skipif(not FROZEN.is_dir(), reason='the frozen tree is not on this machine')
def test_off_is_the_current_path_bit_for_bit_and_a_zero_mask_changes_nothing(tmp_path):
    env = dict(os.environ, PYTHONPATH=f'{FROZEN}:{ROOT}', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    done = subprocess.run([PY, '-B', '-c', T88_SCRIPT, str(ROOT)], capture_output=True, text=True, env=env, timeout=900, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-3000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out == dict(ctl_same=True, zero_mask_same=True, mask_changes=True, n_mask=out['n_mask']) and out['n_mask'] > 0


SETTINGS_SCRIPT = r'''
import json, sys
import numpy as np
sys.path[:0] = [sys.argv[1]+'/experiments/'+d for d in ('r4', 'he', 'mc0', 'ef', 'fl')]
import mc0_core as core
fz = core.frozen_api()
import fast_channels as fast
import fl_channels
import fx_core
import he_core as he
import r4_chain as ch
import r4_options as r4o
from jd_curl.three_d.grid import make_grid_3d
grid = make_grid_3d(nx=38, ny=36, nz=34, hx=1., hy=1., hz=1., unit='mm')
s = ch.settings_for('CTL', 'bench')
spec = ch.spec_of(dict(settings=s))
mine, pre = r4o.preopt_config(fz, s['t88']['pre'], spec)
ref = fz.PreoptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in he.frozen_settings()['t88']['pre'].items()})
rng = np.random.default_rng(0)
L = {side: rng.choice([0, 2, 3, 17], size=grid.shape).astype(np.int16) for side in ('fixed', 'moving')}
for side in L:
    L[side][5:20, 5:20, 5:20] = 1
pin = type('Pin', (), dict(fixed=rng.uniform(.1, 1, grid.shape), moving=rng.uniform(.1, 1, grid.shape)))()
pred = type('Pred', (), dict(labels=L, record={}))()
table = core.load_table()
a, ra = r4o.prepare(spec, pin, pred, table, 'cpu')
b, rb_ = fl_channels.prepare_stacks(fx_core.SPEC, pin, pred, table, 'cpu')
same_stacks = all(np.array_equal(x, y) for x, y in zip(a, b)) and ra == rb_
small = mine.__class__(**dict(mine.__dict__, iterations=(2, 2, 2, 3)))
u1, _ = r4o.core_for(spec).optimize_k(a, grid, small, device='cpu', downsample=fast.FastDownsample())
u2, _ = core.optimize_k(b, grid, small, device='cpu', downsample=fast.FastDownsample())
print(json.dumps(dict(same_config=mine == ref, same_stacks=bool(same_stacks), same_field=bool(np.array_equal(u1, u2)), spec_is_fx=spec == fx_core.SPEC)))
'''


@pytest.mark.skipif(not FROZEN.is_dir(), reason='the frozen tree is not on this machine')
def test_off_through_the_real_settings_path_is_the_current_method(tmp_path):
    """复核次要 8：关着时逐位相同走 settings_for → preopt_config → r4o.prepare → core_for 的真实路径，对照 HE 的冻结设定、fx_core.SPEC、fl_channels.prepare_stacks、mc0_core.optimize_k。"""
    env = dict(os.environ, PYTHONPATH=f'{FROZEN}:{ROOT}', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    done = subprocess.run([PY, '-B', '-c', SETTINGS_SCRIPT, str(ROOT)], capture_output=True, text=True, env=env, timeout=900, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-3000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == dict(same_config=True, same_stacks=True, same_field=True, spec_is_fx=True)
