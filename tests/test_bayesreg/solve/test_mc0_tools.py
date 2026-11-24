"""MC0 (runs/20261103-mc0/任务书.md, plan.md): experiments/mc0/, CPU only, small synthetic volumes; no real image, label or field is read.

- K = 1 against the frozen optimizer, bit for bit, in a subprocess with PYTHONPATH=<frozen>:<repo> (the repository's jd_curl has no preopt: one process never mixes them);
- the K-channel data term and the whole pre-objective against finite differences; K channels against K single channels; sqrt(w) on the images is w on the term;
- the channels of each arm against an independent recomputation; the scale rule; the colour table's hash;
- the reading: every branch, on made-up numbers;
- only the first 5 development pairs; the plan's 45 runs and order; the queue's guards on a fake machine; the start strings;
- one synthetic run through run-one, evaluate and the reading, in a subprocess.
"""
import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT/'experiments/mc0/run_mc0.py'
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'


def _load():
    spec = importlib.util.spec_from_file_location('_mc0_run', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules['_mc0_run'] = module
    spec.loader.exec_module(module)
    return module


RUN = _load()
core, decision = RUN.core, RUN.decision
BRIEF_PAIRS = ('0438-0439', '0439-0440', '0440-0441', '0441-0442', '0442-0443')


def frozen_env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{FROZEN}:{ROOT}')


needs_frozen = pytest.mark.skipif(not (FROZEN/'jd_curl/three_d/preopt.py').is_file(), reason='the frozen worktree is not on this machine')


# ---- K = 1 against the frozen optimizer -----------------------------------------------------------------------------------------------
@needs_frozen
def test_k1_is_bitwise_the_frozen_default_path(tmp_path):
    started = time.time()-1
    out = tmp_path/'k1.json'
    done = subprocess.run([PY, '-B', str(SCRIPT), 'self-check-k1', '--device', 'cpu', '--out', str(out)], env=frozen_env(), capture_output=True, text=True)
    assert done.returncode == 0, done.stdout+done.stderr
    result = json.loads(out.read_text())
    assert result['bitwise'] is True and result['device'] == 'cpu'
    for row in result['rows']:
        assert all(row['same'].values()), row            # field, every level's first and last terms, cap hits, stop; the negative control differs; the field moved; recording the norms changes no bit
        assert row['device_repeatable'] is True
    assert 'analytic_grad' in result['not_compared']
    written = [p for p in (FROZEN/'jd_curl').rglob('*') if p.stat().st_mtime > started]
    assert not written, written                          # the frozen tree stays read only (no bytecode either)


def test_the_frozen_tree_is_required():
    """Under the repository's jd_curl the frozen functions refuse, instead of mixing the two."""
    done = subprocess.run([PY, '-B', '-c', f"import sys; sys.path.insert(0, {str(ROOT/'experiments/mc0')!r}); import mc0_core; mc0_core.frozen_api()"],
                          env=dict(frozen_env(), PYTHONPATH=str(ROOT)), cwd='/', capture_output=True, text=True)
    assert done.returncode != 0 and 'PYTHONPATH' in done.stderr


# ---- the data term and its gradient ------------------------------------------------------------------------------------------------------
def synthetic(K, shape=(7, 8, 9), seed=1, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((K,)+shape, generator=g, dtype=dtype), torch.rand((K,)+shape, generator=g, dtype=dtype)


def interior(shape, seed=2, dtype=torch.float64, low=.2, high=.3):
    g = torch.Generator().manual_seed(seed)
    D, H, W = shape
    return (low+(high-low)*torch.rand((3, D-2, H-2, W-2), generator=g, dtype=dtype)).requires_grad_(True)


def check_against_differences(energy, u_int, seed=3, n=14, eps=1e-6, tol=1e-6):
    grad, = torch.autograd.grad(energy(u_int), u_int)
    rng = np.random.default_rng(seed)
    flat = u_int.detach().clone()
    for index in rng.integers(0, flat.numel(), size=n):
        plus, minus = flat.clone(), flat.clone()
        plus.view(-1)[index] += eps
        minus.view(-1)[index] -= eps
        fd = (float(energy(plus))-float(energy(minus)))/(2*eps)
        assert abs(fd-float(grad.view(-1)[index])) <= tol*max(1., abs(fd))
    return grad


def test_data_term_gradient_against_finite_differences():
    F, M = synthetic(3)
    grad = check_against_differences(lambda u: core.data_term_k(Fn.pad(u, (1, 1, 1, 1, 1, 1)), F, M), interior(F.shape[1:]))
    assert float(grad.abs().max()) > 0


def test_whole_pre_objective_gradient_against_finite_differences():
    F, M = synthetic(4)
    spacing = (1., 1.25, .75)
    grad = check_against_differences(lambda u: core.pre_energy_k(u, F, M, spacing, .1, 1e-3, 10., .2)[0], interior(F.shape[1:], low=-.3, high=.3))
    assert float(grad.abs().max()) > 0


def test_k_channels_equal_the_sum_of_single_channels():
    F, M = synthetic(4)
    u_int = interior(F.shape[1:])
    u = Fn.pad(u_int, (1, 1, 1, 1, 1, 1))
    together = core.data_term_k(u, F, M)
    single = [core.data_term_k(u, F[c:c+1], M[c:c+1]) for c in range(4)]
    assert float(together) == pytest.approx(float(sum(single)), rel=1e-13)
    g, = torch.autograd.grad(together, u_int, retain_graph=True)
    gs = [torch.autograd.grad(term, u_int, retain_graph=True)[0] for term in single]
    assert float((g-sum(gs)).abs().max()) <= 1e-12


def test_square_root_of_the_weight_on_the_images_is_the_weight_on_the_term():
    F, M = synthetic(3)
    w = torch.tensor([.5, 2., 7.], dtype=torch.float64)
    root = w.sqrt()[:, None, None, None]
    u_int = interior(F.shape[1:])
    u = Fn.pad(u_int, (1, 1, 1, 1, 1, 1))
    weighted = core.data_term_k(u, F*root, M*root)
    single = sum(w[c]*core.data_term_k(u, F[c:c+1], M[c:c+1]) for c in range(3))
    assert float(weighted) == pytest.approx(float(single), rel=1e-13)


def test_k1_term_is_the_frozen_expression_bit_for_bit():
    """The frozen data term (preopt.py lines 169-172), written out here, in float32."""
    F, M = synthetic(1, dtype=torch.float32)
    u_int = interior(F.shape[1:], dtype=torch.float32)
    u = Fn.pad(u_int, (1, 1, 1, 1, 1, 1))
    D, H, W = F.shape[1:]
    zz, yy, xx = torch.meshgrid(torch.arange(D, dtype=u.dtype), torch.arange(H, dtype=u.dtype), torch.arange(W, dtype=u.dtype), indexing='ij')
    grid = torch.stack((2*(xx+u[0])/(W-1)-1, 2*(yy+u[1])/(H-1)-1, 2*(zz+u[2])/(D-1)-1), dim=-1)[None]
    warped = Fn.grid_sample(M[0][None, None], grid, mode='bilinear', padding_mode='border', align_corners=True)[0, 0]
    frozen = .5*((warped-F[0])**2).sum()
    ours = core.data_term_k(u, F, M)
    assert float(frozen) == float(ours)
    a, = torch.autograd.grad(frozen, u_int, retain_graph=True)
    b, = torch.autograd.grad(ours, u_int)
    assert a.numpy().tobytes() == b.numpy().tobytes()


# ---- channels and the scale rule --------------------------------------------------------------------------------------------------------
SHAPE = (10, 11, 12)


def labels_and_image(seed=5, shift=0):
    n = int(np.prod(SHAPE))
    labels = (np.arange(n) % 36).reshape(SHAPE).astype(np.int16)
    labels = np.roll(labels, shift, axis=0)
    image = np.random.default_rng(seed).uniform(.1, .7, SHAPE)
    return image, labels


def g75(x):
    return gaussian_filter(x, sigma=.75, mode='nearest', truncate=4.0)


def test_p0_is_the_a5_colouring():
    image, labels = labels_and_image()
    table = core.load_table()
    w = image[(labels == 1) | (labels == 20)].mean()
    J = np.zeros(SHAPE)
    for label, f in table.items():
        J[labels == label] = f*w
    got = core.channels('P0', image, labels, table=table)
    assert got.shape == (1,)+SHAPE and got[0].tobytes() == g75(J).tobytes()


def test_p1_is_the_intensity_and_13_group_masks():
    image, labels = labels_and_image()
    groups = core.load_groups()
    assert len(groups) == 13 and sorted(sum(groups, ())) == list(range(1, 36))
    got = core.channels('P1', image, labels, groups=groups)
    assert got.shape == (14,)+SHAPE
    assert got[0].tobytes() == image.tobytes()           # the intensity is not smoothed
    for c, members in enumerate(groups):
        assert got[c+1].tobytes() == g75(np.isin(labels, members).astype(np.float64)).tobytes()


def test_p2_is_one_hot_of_35_labels_and_a_pool_changes_nothing():
    image, labels = labels_and_image()
    got = core.channels('P2', image, labels)
    assert got.shape == (35,)+SHAPE
    for label in (1, 18, 35):
        assert got[label-1].tobytes() == g75((labels == label).astype(np.float64)).tobytes()
    assert core.channels('P2', image, labels, workers=2).tobytes() == got.tobytes()


def test_p3_weights_are_inverse_fixed_volume():
    _, labels = labels_and_image()
    labels = labels.copy()
    labels[labels == 7] = 0                              # label 7 absent from the fixed image
    w, voxels = core.channel_weights('P3', labels)
    assert voxels[6] == 0 and w[6] == 0
    present = voxels > 0
    assert np.allclose(w[present]*voxels[present], (w[present]*voxels[present])[0])
    assert w[present].mean() == pytest.approx(1.)
    assert core.channel_weights('P2', labels) == (None, None)
    with pytest.raises(SystemExit):
        core.channel_weights('P3', np.zeros(SHAPE))


def test_scale_factor_refuses_what_it_cannot_scale():
    assert core.scale_factor(3., 2.) == 1.5
    for bad in ((0., 1.), (1., 0.), (float('nan'), 1.), (1., float('inf'))):
        with pytest.raises(SystemExit):
            core.scale_factor(*bad)


@pytest.mark.parametrize('arm,k', (('P1', 14), ('P2', 35), ('P3', 35)))
def test_scale_rule_gives_the_gradient_norm_of_p0(arm, k):
    (If, Lf), (Im, Lm) = labels_and_image(5), labels_and_image(6, shift=2)
    stacks, record = core.prepare_stacks(arm, If, Im, Lf, Lm, 'cpu')
    F, M = stacks
    assert F.shape == M.shape == (k,)+SHAPE
    reference = core.data_grad_norm(core.channels('P0', If, Lf), core.channels('P0', Im, Lm), 'cpu')
    assert reference == record['g_norm_P0']
    assert core.data_grad_norm(F, M, 'cpu') == pytest.approx(reference, rel=1e-4)
    assert record['g_norm_scaled'] == pytest.approx(reference, rel=1e-4)
    assert record['D_multiplier'] == pytest.approx(reference/record['g_norm_unscaled'])
    assert record['channel_multiplier'] == pytest.approx(np.sqrt(record['D_multiplier']))
    assert record['D_multiplier'] != pytest.approx(1., abs=1e-3)      # the rule did something here
    if arm == 'P3':                                       # sqrt(w_l s) on each one-hot channel
        w = np.array(record['channel_weights'])
        expected = g75((Lf == 9).astype(np.float64))*np.sqrt(w[8]*record['D_multiplier'])
        assert np.allclose(F[8], expected, rtol=1e-12, atol=0)


def test_scale_rule_leaves_p0_exactly_as_it_is():
    (If, Lf), (Im, Lm) = labels_and_image(5), labels_and_image(6, shift=2)
    stacks, record = core.prepare_stacks('P0', If, Im, Lf, Lm, 'cpu')
    assert record['D_multiplier'] == 1. and record['channel_multiplier'] == 1.
    assert stacks[0].tobytes() == core.channels('P0', If, Lf).tobytes() and stacks[1].tobytes() == core.channels('P0', Im, Lm).tobytes()


def test_p0_images_are_compared_with_the_a5_inputs(tmp_path):
    (If, Lf), (Im, Lm) = labels_and_image(5), labels_and_image(6, shift=2)
    F, M = core.channels('P0', If, Lf)[0], core.channels('P0', Im, Lm)[0]
    assert core.equals_a5_input(core.PAIRS[0], F, M, inputs=tmp_path) is None
    (tmp_path/core.PAIRS[0]).mkdir()
    np.savez(tmp_path/core.PAIRS[0]/'pair.npz', fixed=F, moving=M, fixed_seg35=Lf)
    assert core.equals_a5_input(core.PAIRS[0], F, M, inputs=tmp_path) is True
    assert core.equals_a5_input(core.PAIRS[0], F, M+1e-12, inputs=tmp_path) is False


def test_colour_table_hash_is_pinned(tmp_path):
    mi = core.make_inputs()
    assert core.sha_file(mi.COLOUR_TABLE_A5) == core.COLOUR_TABLE_SHA256 == mi.COLOUR_TABLE_A5_SHA256
    assert len(core.load_table()) == 35
    changed = tmp_path/'table.json'
    changed.write_text(Path(mi.COLOUR_TABLE_A5).read_text().replace('0.62', '0.63'))
    with pytest.raises(SystemExit):
        core.load_table(changed)


def test_tiers_are_the_pre_of_configuration_c():
    reference = json.loads((ROOT/'experiments/d0/configC_reference.json').read_text())['reference_protocol']['pre_optimizer']
    short = core.tier_config('short')
    for key, value in reference.items():
        if key == 'policy':
            assert value == 'fp32-preopt-v1'
        elif key == 'taper':
            assert value == core.TAPER
        elif key == 'seconds':
            assert short['seconds'] == core.RUN_CAP_SECONDS == 900.    # the one difference, written in the plan
        else:
            assert list(np.atleast_1d(short[key])) == list(np.atleast_1d(value)), key
    assert core.tier_config('long') == dict(short, iterations=(50, 50, 100, 600))
    assert core.tier_config('fine') == dict(short, iterations=(50, 50, 100, 600), sigma_field=.5)


def _functions(path):
    tree = ast.parse(Path(path).read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


@needs_frozen
@pytest.mark.parametrize('ours,theirs', (('_gaussian_1d', '_gaussian_1d'), ('_smooth', '_smooth'), ('_identity_mesh', '_identity_mesh'), ('run_level_k', '_run_level_default')))
def test_copies_are_verbatim_up_to_the_energy_name(ours, theirs):
    """The copied functions against the frozen source, statement by statement (docstrings and comments aside); the frozen file is parsed, not imported."""
    mine, frozen = _functions(core.__file__)[ours], _functions(FROZEN/'jd_curl/three_d/preopt.py')[theirs]
    body = (lambda node: [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))])
    assert ast.dump(mine.args) == ast.dump(frozen.args)
    assert [ast.dump(s) for s in body(mine)] == [ast.dump(s).replace("'_pre_energy'", "'pre_energy_k'") for s in body(frozen)]


@needs_frozen
def test_frozen_preopt_hash_is_the_reference():
    reference = json.loads((ROOT/'experiments/d0/configC_reference.json').read_text())['freeze_source_py_sha256']['jd_curl/three_d/preopt.py']
    assert reference == core.FROZEN_PREOPT_SHA256 == core.sha_file(FROZEN/'jd_curl/three_d/preopt.py')


# ---- the reading ----------------------------------------------------------------------------------------------------------------------------
BASE = .85


def row(seg35=BASE, sd=.07, status='done'):
    metrics = {m: .5 for m in decision.MEASURES}
    metrics.update(seg35=seg35, seg4=seg35+.05, sdlogj_official=sd)
    return dict(status=status, metrics=metrics, run_id='x')


def table(p1, *, p2=None, p3=None, p0_sd=.07):
    """Differences against P0 per pair: p1 {tier: differences}, p2, p3 the same (absent: the arm is below every other)."""
    rows = {}
    for tier in core.ALL_TIERS:
        for i, pair in enumerate(core.PAIRS):
            rows[('P0', tier, pair)] = row(sd=p0_sd)
            rows[('P1', tier, pair)] = row(BASE+p1[tier][i])
            if tier != 'short':
                rows[('P2', tier, pair)] = row(BASE+(p2[tier][i] if p2 else -.1))
            if tier == 'long':
                rows[('P3', tier, pair)] = row(BASE+(p3[tier][i] if p3 else -.2))
    return rows


def flat(d):
    return {t: [d]*5 for t in core.ALL_TIERS}


def test_no_headroom_small_delta_in_every_tier():
    result = decision.decide(table(flat(.0049)))
    assert result['reading'] == decision.READ_NONE and result['is_conclusion'] is False and result['reading_basis'] == 'effect' and result['scope_note'] is None
    assert all(r['comparable'] and r['best_arm'] == 'P1' and r['no_headroom'] and not r['headroom'] for r in result['tiers'].values())
    assert result['tiers']['long']['delta_seg35'] == pytest.approx(.0049)
    assert result['tiers']['long']['delta_seg4'] == pytest.approx(.0049)   # seg4 always reported


def test_no_headroom_large_delta_but_fewer_than_4_positive():
    result = decision.decide(table({t: [.05, .05, .05, -.01, -.01] for t in core.ALL_TIERS}))
    assert result['reading'] == decision.READ_NONE
    assert result['tiers']['short']['positive_seg35'] == 3 and result['tiers']['short']['delta_seg35'] > .005


def test_headroom_in_one_tier_is_enough():
    deltas = flat(.001)
    deltas['fine'] = [.0101]*5
    result = decision.decide(table(deltas))
    assert result['reading'] == decision.READ_SEE and result['tiers']['fine']['headroom'] and not result['tiers']['long']['headroom']


@pytest.mark.parametrize('differences', ([.0099]*5, [.0051]*5, [.03, .03, .03, .03, -.01]))
def test_middle_band(differences):
    deltas = flat(.001)
    deltas['long'] = differences
    result = decision.decide(table(deltas))
    assert result['reading'] == decision.READ_MIDDLE and result['reading_basis'] == 'effect' and result['scope_note'] is None    # everything read: the effect itself is in the band
    assert not result['tiers']['long']['no_headroom'] and not result['tiers']['long']['headroom']


def test_the_best_arm_is_chosen_per_tier_among_the_eligible():
    p2 = {t: [.02]*5 for t in core.NOT_SHORT}
    p3 = dict(long=[.03]*5)
    result = decision.decide(table(flat(.001), p2=p2, p3=p3))
    assert result['tiers']['short']['best_arm'] == 'P1' and result['tiers']['long']['best_arm'] == 'P3' and result['tiers']['fine']['best_arm'] == 'P2'
    assert result['reading'] == decision.READ_SEE


def over_the_bar(rows, arms=('P2', 'P3'), tiers=core.NOT_SHORT):
    for key in rows:
        if key[0] in arms and key[1] in tiers:
            rows[key]['metrics']['sdlogj_official'] = .1001


def test_an_arm_over_the_sdlogj_bar_cannot_make_headroom_and_blocks_no_headroom():
    """Review 阻断 1, the test that fixed the error: P2 +.02, P3 +.03 at SDlogJ .1001 is not "看不到" (nor "看得到": an arm not read makes nothing)."""
    rows = table(flat(.001), p2={t: [.02]*5 for t in core.NOT_SHORT}, p3=dict(long=[.03]*5))
    over_the_bar(rows)
    result = decision.decide(rows)
    for tier in core.NOT_SHORT:
        reading = result['tiers'][tier]
        assert reading['best_arm'] == 'P1' and 'SDlogJ' in reading['excluded']['P2'] and reading['predicate_on_read_arms'] is True and reading['no_headroom'] is False and not reading['headroom']
    assert result['tiers']['short']['no_headroom'] is True
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ and result['reading_basis'] == 'not_read' and 'P3' in result['scope_note'] and 'SDlogJ' in result['scope_note']


def drop(rows, arms, tiers=core.NOT_SHORT, pairs=core.PAIRS):
    for key in list(rows):
        if key[0] in arms and key[1] in tiers and key[2] in pairs:
            del rows[key]


def test_arms_not_finished_block_no_headroom():
    """Review 阻断 1, the branch that had no test: P2, P3 did not finish, P1 shows no headroom."""
    rows = table(flat(.001))
    drop(rows, ('P2', 'P3'))
    result = decision.decide(rows)
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ and result['reading_basis'] == 'not_read' and result['arms_incomplete'] == ['P2', 'P3']
    assert result['tiers']['long']['comparable'] and result['tiers']['long']['best_arm'] == 'P1' and result['tiers']['long']['no_headroom'] is False
    assert result['tiers']['short']['no_headroom'] is True and 'P2' in result['scope_note']


@pytest.mark.parametrize('how', ('missing', 'time_cap', 'vram_stop', 'failed', 'not evaluated'))
def test_one_arm_short_of_a_pair_blocks_no_headroom_in_its_tier(how):
    """The budget case: only the last pair's P3 is not there, whatever the way."""
    rows = table(flat(.001))
    key = ('P3', 'long', core.PAIRS[4])
    if how == 'missing':
        del rows[key]
    elif how == 'not evaluated':
        rows[key]['metrics'] = None
    else:
        rows[key]['status'] = how
    result = decision.decide(rows)
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ and result['reading_basis'] == 'not_read' and result['tiers']['long']['excluded'] == dict(P3=decision.INCOMPLETE)
    assert result['tiers']['fine']['no_headroom'] is True and result['tiers']['long']['no_headroom'] is False


def test_an_arm_not_read_does_not_undo_headroom_of_a_read_arm():
    deltas = flat(.001)
    deltas['long'] = [.02]*5
    rows = table(deltas)
    drop(rows, ('P2', 'P3'))
    result = decision.decide(rows)
    assert result['tiers']['long']['headroom'] is True and result['reading'] == decision.READ_SEE and result['reading_basis'] == 'effect'
    assert 'P2' in result['scope_note']


def p0_over(rows, tiers, sd=.11):
    for key in rows:
        if key[0] == 'P0' and key[1] in tiers:
            rows[key]['metrics']['sdlogj_official'] = sd


def test_p0_over_the_bar_in_the_fine_tier_is_not_read_as_an_effect():
    """Review of the fix round, 重要 1: the fine tier was not compared; the first line must not say the effect fell in the band, and the scope note names the tier and its arms."""
    rows = table(flat(.001))
    p0_over(rows, ('fine',))
    result = decision.decide(rows)
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ == '中间带（有档或臂没读到，不一定是效应落在 .005–.01，见 scope_note）' and result['reading_basis'] == 'not_read'
    assert result['tiers']['short']['no_headroom'] and result['tiers']['long']['no_headroom'] and not result['tiers']['fine']['comparable']
    assert result['tiers']['fine']['excluded'] == {} and result['arms_incomplete'] == []          # nothing else says it
    assert list(result['tiers_not_comparable']) == ['fine'] and result['tiers_not_comparable']['fine']['k_arms_not_compared'] == ['P1', 'P2']
    note = result['scope_note']
    assert 'fine：P0 官方 SDlogJ 均值 0.1100 > 0.1' in note and '因此没比的 K > 1 臂：P1、P2' in note
    text = decision.summary_text(result)
    assert note in text and result['reading'] in text and 'reading_basis：not_read' in text


def test_p0_over_the_bar_in_long_and_fine_names_every_k35_arm():
    """重要 1, the second case: no K = 35 cell was compared; only the short tier's K = 14 arm can read, and the note says so at the top."""
    rows = table(flat(.001))
    p0_over(rows, core.NOT_SHORT)
    result = decision.decide(rows)
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ and result['reading_basis'] == 'not_read' and result['comparable_tiers'] == ['short']
    note = result['scope_note']
    assert '因此没比的 K > 1 臂：P1、P2、P3' in note and '因此没比的 K > 1 臂：P1、P2' in note
    assert result['tiers_not_comparable']['long']['k_arms_not_compared'] == ['P1', 'P2', 'P3'] and result['tiers_not_comparable']['fine']['k_arms_not_compared'] == ['P1', 'P2']
    deltas = flat(.001)                                  # the same with headroom on the K = 14 arm of the short tier: it reads, and the note still says what it is not about
    deltas['short'] = [.02]*5
    rows = table(deltas)
    p0_over(rows, core.NOT_SHORT)
    result = decision.decide(rows)
    assert result['reading'] == decision.READ_SEE and result['reading_basis'] == 'effect' and '因此没比的 K > 1 臂：P1、P2、P3' in result['scope_note']


def test_the_tier_section_shows_the_predicate_on_the_read_arms():
    rows = table(flat(.001))
    p0_over(rows, ('fine',))
    drop(rows, ('P3',))
    result = decision.decide(rows)
    text = decision.summary_text(result)
    sections = {tier: text.split(f'## 档 {tier}')[1].split('## 档')[0] for tier in core.ALL_TIERS}
    assert 'predicate_on_read_arms（只看读到的臂时，每个臂都没有余量）：True；no_headroom（这一档算没有余量）：True' in sections['short']
    assert 'predicate_on_read_arms（只看读到的臂时，每个臂都没有余量）：True；no_headroom（这一档算没有余量）：False' in sections['long']     # P3 not read
    assert 'predicate_on_read_arms（只看读到的臂时，每个臂都没有余量）：—；no_headroom（这一档算没有余量）：False' in sections['fine']       # not compared


def test_an_arm_read_in_the_band_is_named_when_the_first_line_speaks_of_what_was_not_read():
    """A gap and an effect in the band at once: the reading line is the plan's; the note keeps the reader from taking the whole band for the gap."""
    deltas = flat(.001)
    deltas['short'] = [.008]*5
    rows = table(deltas)
    drop(rows, ('P3',))
    result = decision.decide(rows)
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ and result['reading_basis'] == 'not_read'
    assert '读到的臂本身也有不满足"没有余量"的（这部分不是没读到造成的）：short：P1（Δ seg35 +0.0080，5/5 为正）' in result['scope_note']


@pytest.mark.parametrize('p2_long,at_bar', (([.015]*5, ['P2']), ([.007, .007, .007, .007, -.001], [])))
def test_no_headroom_asks_every_read_arm_not_only_the_best(p2_long, at_bar):
    """Review of the fix round, 次要 1: P3 long .06 .06 .06 -.02 -.02 has the largest mean at 3 of 5, the arm below it does not meet the predicate.  Before revision 2 this read "看不到"."""
    p3 = dict(long=[.06, .06, .06, -.02, -.02])
    result = decision.decide(table(flat(.001), p2=dict(long=p2_long, fine=[-.1]*5), p3=p3))
    long = result['tiers']['long']
    assert long['best_arm'] == 'P3' and long['positive_seg35'] == 3 and long['delta_seg35'] == pytest.approx(.028)
    assert long['arm_effects']['P3']['no_headroom'] is True and long['arm_effects']['P2']['no_headroom'] is False
    assert long['predicate_on_read_arms'] is False and long['no_headroom'] is False and long['headroom'] is False
    assert result['reading'] == decision.READ_MIDDLE and result['reading_basis'] == 'effect'      # all read
    assert long['other_arms_at_headroom_bar'] == at_bar


def test_another_arm_at_the_bar_does_not_make_headroom_and_is_shown():
    """The rule of "看得到" stays on the best arm (the plan); what the reader sees is that another arm is at the bar."""
    p3 = dict(long=[.06, .06, .06, -.02, -.02])
    result = decision.decide(table(flat(.001), p2=dict(long=[.015]*5, fine=[-.1]*5), p3=p3))
    assert result['reading'] != decision.READ_SEE and not result['tiers']['long']['headroom'] and result['tiers']['long']['arm_effects']['P2']['at_headroom_bar'] is True
    shown = 'long：P2（Δ seg35 +0.0150，5/5 为正）达到"看得到"的门槛'
    assert shown in result['scope_note'] and '最好的臂 P3（Δ seg35 +0.0280，3/5 为正）没有达到' in result['scope_note'] and '只判均值最高的臂' in result['scope_note']
    assert shown in decision.summary_text(result).split('## 档 long')[1].split('## 档 fine')[0]


def test_a_best_arm_at_the_bar_names_no_other_arm():
    p3 = dict(long=[.03]*5)
    result = decision.decide(table(flat(.001), p2=dict(long=[.015]*5, fine=[-.1]*5), p3=p3))
    assert result['tiers']['long']['headroom'] and result['tiers']['long']['other_arms_at_headroom_bar'] == [] and result['reading'] == decision.READ_SEE and result['scope_note'] is None


def test_the_inference_on_p3_is_written_as_one():
    assert '〔推断〕P3 可能在粗层上吃亏，看 r_ℓ' in decision.NOTE and 'P3 在粗层上吃亏' not in decision.NOTE


def test_the_marks_of_an_oracle_field():
    rows = table(flat(.001))
    rows[('P1', 'long', core.PAIRS[0])].update(n_infeasible_at_1=0, status_at_1='infeasible')     # a domain failure: no cell counted, not a clean field
    rows[('P1', 'long', core.PAIRS[1])].update(n_infeasible_at_1=12, status_at_1='infeasible')
    text = decision.summary_text(decision.decide(rows))
    assert '0（infeasible）；12（infeasible）' in text
    assert '| α = 1 不可行单元（status） | 接受 α 下 seg35 / seg4（oracle，不是方法） |' in text and '| 接受 α 下 seg35 / seg4 |' not in text


def test_the_working_point_is_reported_with_delta():
    rows = table(flat(.0049))
    for key in rows:
        if key[0] == 'P1':
            rows[key]['metrics'].update(sdlogj_official=.05, sdlogj_brain=.2)
        elif key[0] == 'P0':
            rows[key]['metrics'].update(sdlogj_official=.07, sdlogj_brain=.3)
    result = decision.decide(rows)
    reading = result['tiers']['long']
    assert reading['delta_sdlogj_official'] == pytest.approx(-.02)
    assert reading['working_point']['sdlogj_official'] == dict(best=pytest.approx(.05), P0=pytest.approx(.07), delta=pytest.approx(-.02))
    assert reading['working_point']['sdlogj_brain'] == dict(best=pytest.approx(.2), P0=pytest.approx(.3), delta=pytest.approx(-.1))
    text = decision.summary_text(result)
    assert '工作点' in text and '官方 SDlogJ 0.0500 / 0.0700 / -0.0200' in text
    assert '不同工作点' in result['note'] and '不同工作点' in text


def test_the_selection_effect_is_in_the_reading_and_counts_the_revised_plan():
    result = decision.decide(table(flat(.001)))
    effect = result['selection_effect']
    assert (effect['tiers'], effect['most_arms_in_a_tier'], effect['arm_tier_cells']) == (3, 3, 6)   # short P1; long P1, P2, P3; fine P1, P2
    assert effect['per_cell_5_of_5_under_zero'] == 1/32 and effect['rough_upper_bound'] == 6/32
    assert '选择效应' in effect['text'] and '6 个臂' in effect['text'] and effect['text'] in decision.summary_text(result)


def test_level_ratios_and_the_accepted_alpha_reach_the_tables():
    rows = table(flat(.001))
    for i, pair in enumerate(core.PAIRS):
        rows[('P3', 'long', pair)].update(level_ratio_raw=[.1, .2, .5, 1.], level_ratio_smoothed=[.2, .3, .6, 1.1], at_accepted_alpha=dict(alpha=.9, seg35=.8, seg4=.85))
    rows[('P3', 'long', core.PAIRS[0])]['level_ratio_raw'] = [.1, .2]               # a run cut on a coarse level: not averaged
    table_row = decision.decide(rows)['tables']['long']['P3']
    assert table_row['level_ratio_raw'] == pytest.approx([.1, .2, .5, 1.]) and table_row['level_ratio_smoothed'] == pytest.approx([.2, .3, .6, 1.1])
    assert table_row['at_accepted_alpha'] == dict(seg35=pytest.approx(.8), seg4=pytest.approx(.85))
    text = decision.summary_text(decision.decide(rows))
    assert '0.100 / 0.200 / 0.500 / 1.000' in text and 'r_ℓ' in text
    assert core.level_norm_record([dict(raw=2., smoothed=1.)], [dict(raw=4., smoothed=0.)]) == dict(arm=[dict(raw=2., smoothed=1.)], P0=[dict(raw=4., smoothed=0.)], ratio_raw=[.5], ratio_smoothed=[None])


def test_a_run_cut_by_the_wall_cap_is_not_read():
    rows = table(flat(.001), p2={t: [.02]*5 for t in core.NOT_SHORT})
    rows[('P2', 'long', core.PAIRS[2])]['status'] = 'time_cap'
    result = decision.decide(rows)
    assert result['tiers']['long']['best_arm'] == 'P1' and result['tiers']['long']['excluded']['P2'] == decision.INCOMPLETE
    assert result['arms_incomplete'] == ['P2'] and 'P2' in result['scope_note'] and result['tiers']['long']['no_headroom'] is False
    assert result['tables']['long']['P2']['n_evaluated'] == 5 and result['tables']['long']['P2']['n_read'] == 4
    assert result['tiers']['fine']['headroom'] and result['reading'] == decision.READ_SEE


def test_a_tier_without_comparison_keeps_no_headroom_from_being_read():
    rows = table(flat(.001))
    for pair in core.PAIRS:
        for arm in ('P1', 'P2'):
            rows[(arm, 'fine', pair)]['metrics']['sdlogj_official'] = .12
    result = decision.decide(rows)
    assert not result['tiers']['fine']['comparable'] and result['comparable_tiers'] == ['short', 'long']
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ and result['reading_basis'] == 'not_read' and '因此没比的 K > 1 臂：P1、P2' in result['scope_note']


def test_p0_over_the_sdlogj_bar_makes_the_tier_not_comparable():
    result = decision.decide(table(flat(.05), p0_sd=.11))
    assert not any(r['comparable'] for r in result['tiers'].values()) and result['reading'] == decision.READ_UNDECIDABLE and result['reading_basis'] == 'not_read'
    assert all(t in result['scope_note'] for t in core.ALL_TIERS)


def test_nothing_comparable_is_undecidable():
    assert decision.decide({})['reading'] == decision.READ_UNDECIDABLE
    rows = table(flat(.05))
    del rows[('P0', 'short', core.PAIRS[0])]
    result = decision.decide(rows)
    assert not result['tiers']['short']['comparable'] and result['tiers']['long']['headroom'] and result['reading'] == decision.READ_SEE


def test_script_magnitude_check_and_descriptive_tables():
    rows = table(flat(.001))
    result = decision.decide(rows, stops=dict(P2=dict(reason='r')))
    assert result['script_magnitude_check']['below_a5_final'] is True and result['arms_stopped'] == dict(P2=dict(reason='r'))
    assert set(result['tables']['long']) == {'P0', 'P1', 'P2', 'P3'} and set(result['tables']['fine']) == {'P0', 'P1', 'P2'} and set(result['tables']['short']) == {'P0', 'P1'}
    for measure in decision.MEASURES:
        assert measure in result['tables']['long']['P1']['means'] and measure in result['tables']['long']['P1']['against_P0']
    for pair in core.PAIRS:
        rows[('P0', 'short', pair)]['metrics']['seg35'] = .90
    assert decision.decide(rows)['script_magnitude_check']['below_a5_final'] is False


def test_written_analysis_keeps_the_earlier_version(tmp_path):
    root = tmp_path/'mc0_v1'
    d = core.run_dir(root, core.Cell('P0', 'short', core.PAIRS[0]))
    d.mkdir(parents=True)
    RUN.write_json(d/'run.json', dict(status='done', scale=dict(D_multiplier=1.)))
    RUN.write_json(d/'metrics.json', dict(metrics=row()['metrics'], certificate=dict(accepted_alpha=.9, n_infeasible_at_1=3, status_at_1='infeasible')))
    first = decision.write_analysis(root)
    assert first['reading'] == decision.READ_UNDECIDABLE
    with open(root/'analysis/long.csv', newline='') as f:
        long_rows = list(csv.DictReader(f))
    assert long_rows and {(r['oracle'], r['field']) for r in long_rows} == {('true', 'uncertified_alpha1')}
    assert first['tables']['short']['P0']['status_at_1'] == ['infeasible'] and first['tables']['short']['P0']['n_infeasible_at_1'] == [3]
    assert json.loads((root/'analysis/decision_mc0.json').read_text())['schema'] == decision.SCHEMA
    text = (root/'analysis/summary_mc0.md').read_text(encoding='utf-8')
    assert '不是结论' in text and '阴性不是逻辑证明' in text and 'seg4' in text
    assert 'mc0_P0_short_0438-0439' in (root/'analysis/long.csv').read_text()
    time.sleep(1.1)
    decision.write_analysis(root)
    assert len(list((root/'analysis').glob('previous_*/decision_mc0.json'))) == 1


# ---- only the first 5 development pairs ---------------------------------------------------------------------------------------------------
def test_pairs_are_the_briefs_five():
    assert tuple(core.short_name(p) for p in core.PAIRS) == BRIEF_PAIRS


@pytest.mark.parametrize('pair', ('OASIS_OAS1_0443_MR1__OASIS_OAS1_0444_MR1', 'OASIS_OAS1_0020_MR1__OASIS_OAS1_0021_MR1', 'S37__S38', '../x'))
def test_other_pairs_are_refused_before_any_file_is_opened(pair, monkeypatch):
    monkeypatch.setattr(core, 'STAGE', Path('/nonexistent'))
    with pytest.raises(SystemExit, match='first 5'):
        RUN.read_pair(pair, ())
    with pytest.raises(SystemExit, match='first 5'):
        core.require_cell('P0', 'short', pair)


def test_unknown_cells_are_refused():
    with pytest.raises(SystemExit):
        core.require_cell('P2', 'short', core.PAIRS[0])      # P2, P3 have no short tier
    with pytest.raises(SystemExit):
        core.require_cell('P9', 'long', core.PAIRS[0])


def test_subjects_are_on_the_allowlist_and_off_the_training_and_holdout_lists():
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    allowlist = load_dev_allowlist(core.ALLOWLIST)
    subjects = set()
    for pair in core.PAIRS:
        fixed, moving = pair.split('__')
        check_pair_subjects(fixed, moving, allowlist)
        subjects |= {fixed, moving}
    assert len(subjects) == 6
    for name in ('train_subjects.txt', 'holdout_B_subjects.txt'):
        listed = set((ROOT/'splits/bayesreg_v1'/name).read_text().split())
        assert listed and not subjects & listed, name


# ---- the plan and the start strings ------------------------------------------------------------------------------------------------------
def test_plan_has_the_45_runs_pair_major():
    plan = core.plan_runs()
    assert len(plan) == 45 == 5*(3+3+2+1) and len(set(plan)) == 45
    assert [c.pair for c in plan] == [p for p in core.PAIRS for _ in range(9)]
    assert {c.tier for c in plan if c.arm == 'P3'} == {'long'}                       # plan revision 1
    for arm, tiers in (('P0', 3), ('P1', 3), ('P2', 2), ('P3', 1)):
        assert sum(c.arm == arm for c in plan) == 5*tiers
    assert not any(c.arm in ('P2', 'P3') and c.tier == 'short' for c in plan)
    assert [c.arm for c in plan[:9]][0] == 'P0' and [c.arm for c in plan[9:18]][0] == 'P1'     # the arm order rotates with the pair
    assert core.plan_runs()[0].unit == 'tag-mc0-p0-short-0438-0439' and core.plan_runs()[0].run_id == 'mc0_P0_short_0438-0439'


def test_start_strings():
    code, root = Path('/home/rmcfz/seq_vp_tag_results/mc0_v1/code'), RUN.DEFAULT_ROOT
    strings = [RUN.queue_launch(), RUN.k1_argv(code, root), RUN.eval_argv(code, root)]+[RUN.run_argv(c, code, root) for c in core.plan_runs()]
    for argv in strings:
        text = shlex.join(argv)
        assert 'py'+'test' not in text
        assert text.startswith('systemd-run --user --unit=tag-mc0-') and 'seq-vp' not in text and '--property=MemoryMax=9G' in text
    for argv in strings[1:]:
        text = shlex.join(argv)
        assert f'--setenv=PYTHONPATH={core.FROZEN}:{code}' in text and '--wait' in text and '--collect' in text and PY+' -B' in text
    assert '--setenv=CUDA_VISIBLE_DEVICES=' in shlex.join(RUN.eval_argv(code, root))      # the evaluation stays off the GPU
    assert 'CUDA_VISIBLE_DEVICES' not in shlex.join(RUN.run_argv(core.plan_runs()[0], code, root))
    assert str(root) == '/home/rmcfz/seq_vp_tag_results/mc0_v1'


def test_a_start_string_with_the_word_is_refused():
    with pytest.raises(ValueError):
        RUN.unit_argv('tag-mc0-x', Path('/tmp/py'+'test-of-x/code'), [], log='/tmp/x')


# ---- the queue on a fake machine ----------------------------------------------------------------------------------------------------------
class FakeSystem:
    def __init__(self, behaviour=None, k1=True):
        self.launched, self.clock, self.behaviour, self.k1 = [], 0., behaviour or (lambda cell: (0, dict(status='done'), 10.)), k1
        self.load, self.other, self.free_gpu, self.free_home, self.evaluate = .1, [], 20000., 500., None

    def load1(self):
        return self.load

    def units(self):
        return [f'{RUN.QUEUE_UNIT}.service']+list(self.other)

    def gpu_free_mib(self):
        return self.free_gpu

    def home_free_gb(self):
        return self.free_home

    def now(self):
        return self.clock

    def sleep(self, seconds):
        self.clock += seconds

    def code_checkout(self, root, commit):
        return Path(root)/'code', 'fakesha'

    def run(self, argv, log):
        unit = next(a for a in argv if a.startswith('--unit=')).split('=', 1)[1]
        self.launched.append(unit)
        if unit == RUN.K1_UNIT:
            RUN.write_json(argv[argv.index('--out')+1], dict(bitwise=self.k1))
            return 0
        if unit == RUN.EVAL_UNIT:
            return self.evaluate(argv) if self.evaluate else 0
        cell = core.require_cell(argv[argv.index('--arm')+1], argv[argv.index('--tier')+1], argv[argv.index('--pair')+1])
        rc, run, seconds = self.behaviour(cell)
        self.clock += seconds
        if run is not None:
            RUN.write_json(Path(argv[argv.index('--out')+1])/'run.json', run)
        return rc


@pytest.fixture
def root():
    base = tempfile.mkdtemp(prefix='mc0q_')             # not the test runner's tmp_path: its name would fail the start-string check
    yield Path(base)/'mc0_v1'
    shutil.rmtree(base)


def options(root):
    return RUN.argparse.Namespace(root=str(root), commit=None, wait_minutes=30)


RUN_UNITS = {c.unit for c in core.plan_runs()}


def test_queue_runs_the_plan_once_and_a_restart_reruns_nothing(root):
    system = FakeSystem()
    assert RUN.queue(options(root), system) == 0
    assert system.launched[0] == RUN.K1_UNIT and system.launched[-1] == RUN.EVAL_UNIT
    assert system.launched[1:-1] == [c.unit for c in core.plan_runs()]
    assert (root/'analysis/decision_mc0.json').is_file() and (root/'batch.json').is_file() and (root/'queue_end.json').is_file()
    first = core.plan_runs()[0]
    assert (core.run_dir(root, first)/'cmd.txt').is_file() and json.loads((core.run_dir(root, first)/'queue_done.json').read_text())['state'] == 'done'
    again = FakeSystem()
    assert RUN.queue(options(root), again) == 0
    assert again.launched == [RUN.EVAL_UNIT]             # no run again; the K = 1 check is not repeated either


@pytest.mark.parametrize('how', ('metrics_error', 'unit_limit'))
def test_a_failed_evaluation_still_writes_the_reading_and_returns_8(root, how):
    """Review of the fix round, 次要 6: one field has no metrics: it counts as not read, the others are read, the exit code still says the evaluation failed."""
    missing = core.Cell('P3', 'long', core.PAIRS[4])

    def evaluate(argv):
        assert argv[argv.index('--root')+1] == str(root)
        for cell in core.plan_runs():
            d = core.run_dir(root, cell)
            if cell != missing:
                RUN.write_json(d/'metrics.json', dict(metrics=row(BASE+(.001 if cell.arm != 'P0' else 0.))['metrics'], certificate={}))
            elif how == 'metrics_error':
                RUN.write_json(d/'metrics_error.json', dict(error='x'))
        return 8 if how == 'metrics_error' else 1
    system = FakeSystem()
    system.evaluate = evaluate
    assert RUN.queue(options(root), system) == 8
    result = json.loads((root/'analysis/decision_mc0.json').read_text())
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ and result['reading_basis'] == 'not_read'
    assert result['tiers']['long']['excluded'] == dict(P3=decision.INCOMPLETE) and result['tiers']['fine']['no_headroom'] is True and result['tiers']['long']['no_headroom'] is False
    assert (root/'analysis/summary_mc0.md').is_file() and (root/'analysis/long.csv').is_file()
    stop = json.loads((root/'queue_stop.json').read_text())['reason']
    assert RUN.EVAL_UNIT in stop and 'the reading is written' in stop
    assert json.loads((root/'queue_end.json').read_text())['eval_rc'] == (8 if how == 'metrics_error' else 1)


def test_an_evaluation_that_did_nothing_gives_the_reading_undecidable_and_returns_8(root):
    system = FakeSystem()
    system.evaluate = lambda argv: 1
    assert RUN.queue(options(root), system) == 8
    assert json.loads((root/'analysis/decision_mc0.json').read_text())['reading'] == decision.READ_UNDECIDABLE and (root/'queue_stop.json').is_file()


@pytest.mark.parametrize('out', ('[N/A]\n', 'Failed to initialize NVML: Unknown Error\n', '', 'nan\n'))
def test_a_gpu_reading_that_is_not_a_number_keeps_the_gate_closed_and_stops_with_a_record(root, monkeypatch, out):
    """Review of the fix round, 次要 9: it was a ValueError, in the journal only, with no queue_stop.json."""
    monkeypatch.setattr(RUN.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr=''))
    assert RUN.System().gpu_free_mib() == 0.

    class Machine(FakeSystem):
        gpu_free_mib = RUN.System.gpu_free_mib
    system = Machine()
    assert RUN.queue(RUN.argparse.Namespace(root=str(root), commit=None, wait_minutes=1), system) == 8
    assert system.launched == [] and 'GPU free' in json.loads((root/'queue_stop.json').read_text())['reason']


def test_a_gpu_reading_that_is_a_number_is_that_number(monkeypatch):
    monkeypatch.setattr(RUN.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout='24000\n', stderr=''))
    assert RUN.System().gpu_free_mib() == 24000.


def test_p0_images_differing_from_a5_stop_at_the_first_run(root):
    system = FakeSystem(lambda cell: (0, dict(status='done', scale=dict(images_equal_a5_inputs=False)), 5.))
    assert RUN.queue(options(root), system) == 8
    assert [u for u in system.launched if u in RUN_UNITS] == [core.plan_runs()[0].unit] and 'A5' in json.loads((root/'queue_stop.json').read_text())['reason']
    assert RUN.EVAL_UNIT not in system.launched
    missing = tempfile.mkdtemp(prefix='mc0q_')                                       # no A5 file to compare with (None): the queue goes on
    system = FakeSystem(lambda cell: (0, dict(status='done', scale=dict(images_equal_a5_inputs=None)), 5.))
    assert RUN.queue(options(Path(missing)/'mc0_v1'), system) == 0
    shutil.rmtree(missing)


def test_a_failing_systemctl_keeps_the_gate_closed(root, monkeypatch):
    root.mkdir(parents=True)
    monkeypatch.setattr(RUN.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout='', stderr='no bus'))
    system = RUN.System()
    assert system.units() == ['systemctl failed with exit 1']
    assert 'systemctl failed' in RUN.busy_reason(system, root, False)


def test_k1_mismatch_stops_before_any_run(root):
    system = FakeSystem(k1=False)
    assert RUN.queue(options(root), system) == 8
    assert system.launched == [RUN.K1_UNIT] and (root/'queue_stop.json').is_file()


def test_vram_stop_stops_only_that_arm(root):
    def behaviour(cell):
        return 0, dict(status='vram_stop', reason='r') if cell.arm == 'P2' else dict(status='done'), 10.
    system = FakeSystem(behaviour)
    assert RUN.queue(options(root), system) == 0
    p2 = [u for u in system.launched if '-p2-' in u]
    assert p2 == ['tag-mc0-p2-long-0438-0439']
    assert sum('-p3-' in u for u in system.launched) == 5 and sum('-p0-' in u for u in system.launched) == 15
    assert json.loads((root/'analysis/decision_mc0.json').read_text())['arms_stopped']['P2']['run_id'] == 'mc0_P2_long_0438-0439'


def test_a_device_peak_over_the_limit_stops_any_arm(root):
    def behaviour(cell):
        return 0, dict(status='done', device_peak_reserved_gib=14.01 if cell.arm == 'P1' else 3.), 10.
    system = FakeSystem(behaviour)
    assert RUN.queue(options(root), system) == 0
    assert sum('-p1-' in u for u in system.launched) == 1 and sum('-p0-' in u for u in system.launched) == 15


def test_fifteen_minutes_stop_a_k35_arm_but_not_k14(root):
    def behaviour(cell):
        return 0, dict(status='done'), (901. if cell.arm == 'P3' or (cell.arm == 'P1' and cell.tier == 'short') else 5.)     # one K = 14 tier: stays under the total budget
    system = FakeSystem(behaviour)
    assert RUN.queue(options(root), system) == 0
    assert sum('-p3-' in u for u in system.launched) == 1 and sum('-p1-' in u for u in system.launched) == 15


def test_the_run_unit_budget_starts_no_new_run_and_the_rest_is_still_evaluated(root):
    system = FakeSystem(lambda cell: (0, dict(status='done'), 1000.))
    assert RUN.queue(options(root), system) == 0
    assert sum(u in RUN_UNITS for u in system.launched) == 9     # 9 x 1000 s reach 2.5 h
    assert system.launched[-1] == RUN.EVAL_UNIT and 'no new run' in json.loads((root/'queue_end.json').read_text())['end']


def test_failures_are_kept_not_rerun_and_three_in_a_row_stop(root):
    system = FakeSystem(lambda cell: (1, None, 5.))
    assert RUN.queue(options(root), system) == 8
    assert sum(u in RUN_UNITS for u in system.launched) == 3
    assert 'failures in a row' in json.loads((root/'queue_stop.json').read_text())['reason']
    state = RUN.ledger(root)
    assert set(state['states'].values()) == {'failed'}
    again = FakeSystem()
    assert RUN.queue(options(root), again) == 0
    assert not set(again.launched) & set(system.launched)-{RUN.K1_UNIT, RUN.EVAL_UNIT}


def test_a_k35_run_failing_without_a_record_stops_its_arm_and_the_others_go_on(root):
    def behaviour(cell):
        return (137, None, 60.) if core.ARMS[cell.arm]['k'] == 35 else (0, dict(status='done'), 5.)
    system = FakeSystem(behaviour)
    assert RUN.queue(options(root), system) == 0
    assert sum('-p2-' in u for u in system.launched) == 1 and sum('-p3-' in u for u in system.launched) == 1
    assert sum('-p0-' in u for u in system.launched) == 15 and sum('-p1-' in u for u in system.launched) == 15


def test_a_lost_run_is_never_rerun_and_counts_at_the_wall_cap(root):
    first = core.plan_runs()[0]
    core.run_dir(root, first).mkdir(parents=True)        # started, the driver died: no queue_done.json
    assert RUN.ledger(root)['states'] == {first.run_id: 'interrupted'} and RUN.ledger(root)['run_unit_seconds'] == core.RUN_CAP_SECONDS
    system = FakeSystem()
    assert RUN.queue(options(root), system) == 0
    assert first.unit not in system.launched


@pytest.mark.parametrize('field,value,word', (('load', 1.5, 'load'), ('other', ['tag-other.service'], 'other units'), ('other', ['seq-vp-x.service'], 'other units'),
                                              ('free_gpu', 100., 'GPU'), ('free_home', 99., '/home')))
def test_gate_refuses_a_busy_machine_then_stops(root, field, value, word):
    root.mkdir(parents=True)
    system = FakeSystem()
    assert RUN.busy_reason(system, root, True) is None   # its own queue unit does not count
    setattr(system, field, value)
    assert word in RUN.busy_reason(system, root, True)
    assert RUN.queue(RUN.argparse.Namespace(root=str(root), commit=None, wait_minutes=1), system) == 8
    assert system.launched == [] and 'gate closed' in json.loads((root/'queue_stop.json').read_text())['reason']


def test_gate_hold_file_and_a_cpu_unit_ignores_the_gpu(root):
    root.mkdir(parents=True)
    system = FakeSystem()
    system.free_gpu = 0.
    assert RUN.busy_reason(system, root, False) is None
    (root/'hold').write_text('')
    assert RUN.busy_reason(system, root, False) == 'hold file'


def test_code_checkout_pins_a_clean_commit_in_a_detached_worktree(tmp_path, monkeypatch):
    repo = tmp_path/'repo'
    repo.mkdir()

    def git(*args):
        return subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    git('init', '-q')
    (repo/'a.txt').write_text('1')
    git('add', 'a.txt')
    git('commit', '-qm', 'one')
    monkeypatch.setattr(core, 'ROOT', repo)
    root = tmp_path/'mc0_v1'
    root.mkdir()
    code, sha = RUN.System().code_checkout(root, None)
    assert code == root/'code' and sha == git('rev-parse', 'HEAD') and (code/'a.txt').read_text() == '1'
    assert RUN.System().code_checkout(root, None) == (code, sha)     # a restart at the same commit
    (repo/'a.txt').write_text('2')
    with pytest.raises(RUN.Stop, match='not clean'):
        RUN.System().code_checkout(root, None)
    git('add', 'a.txt')
    git('commit', '-qm', 'two')
    with pytest.raises(RUN.Stop, match='is not at'):
        RUN.System().code_checkout(root, None)                       # the worktree stays at the batch's commit
    git('worktree', 'remove', '--force', str(code))


def test_the_root_must_be_this_probes(tmp_path):
    assert RUN.queue(RUN.argparse.Namespace(root=str(tmp_path/'d0_v1'), commit=None, wait_minutes=1), FakeSystem()) == 2
    assert not (tmp_path/'d0_v1').exists()


def test_vram_guard_stops_at_the_limit():
    class Cuda:
        peak = 0

        @classmethod
        def max_memory_reserved(cls):
            return cls.peak

    class FakeTorch:
        cuda = Cuda
        device = staticmethod(torch.device)
    observer = RUN.vram_guard(FakeTorch, 'cuda')
    Cuda.peak = RUN.VRAM_LIMIT_BYTES
    observer(dict(level=3, iteration=1))
    Cuda.peak = RUN.VRAM_LIMIT_BYTES+1
    with pytest.raises(RUN.VramLimit):
        observer(dict(level=3, iteration=2))
    assert RUN.vram_guard(FakeTorch, 'cpu') is None


# ---- the evaluation --------------------------------------------------------------------------------------------------------------------------
def test_certify_ladder_records_the_first_certified_alpha():
    class R:
        def __init__(self, ok):
            self.status, self.reason, self.min_jd_lower_bound, self.n_cells_checked, self.n_infeasible, self.n_uncertain = (
                ('certified_feasible', 'x', .1, 8, 0, 0) if ok else ('infeasible', 'JACOBIAN_VIOLATION', -.1, 8, 3, 1))
            self.certificate_version, self.config = 'the version', RUN.certificate_config()
    calls = []

    def certify(u, grid, *, config):
        calls.append(float(u.max()))
        return R(len(calls) == 3)
    alphas = tuple(.9**k for k in range(7))
    ladder = RUN.certify_ladder(np.ones((2, 2)), None, alphas, certify, None)
    assert calls == pytest.approx([1., .9, .81]) and ladder['accepted_alpha'] == alphas[2]
    assert ladder['n_infeasible_at_1'] == 3 and ladder['n_uncertain_at_1'] == 1 and len(ladder['attempts']) == 3 and ladder['status_at_1'] == 'infeasible'
    for attempt in ladder['attempts']:                   # review of the fix round, 次要 2: which certificate gave this, in the record
        assert attempt['certificate_version'] == 'the version' and attempt['config']['tile_size'] == 65536 and attempt['config']['max_depth'] == 48
    json.dumps(ladder)
    calls.clear()
    assert RUN.certify_ladder(np.ones((2, 2)), None, alphas, lambda u, g, config: R(False), None)['accepted_alpha'] is None


def test_certificate_configuration_is_c():
    config = RUN.certificate_config()
    assert type(config).__name__ == 'PrescreenCertificateConfig' and (config.delta, config.max_depth, config.max_boxes, config.tile_size) == (0., 48, 255, 65536)


@needs_frozen
def test_the_certificate_tile_size_is_the_one_of_the_frozen_driver():
    """The driver's THROUGHPUT_POLICY, parsed and not imported; every later throughput policy derives from it, and A5's records say 65536."""
    tree = ast.parse((FROZEN/'scripts/run_stage_3d_e19_mri.py').read_text())
    policy = next(node.value for node in tree.body if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'THROUGHPUT_POLICY')
    keywords = {k.arg: k.value for k in policy.keywords}
    assert ast.literal_eval(keywords['certificate_tile_size']) == RUN.certificate_config().tile_size == 65536


def test_field_metrics_of_the_identity():
    labels = (np.arange(int(np.prod(SHAPE))) % 36).reshape(SHAPE).astype(np.int16)
    seg4 = (labels % 5).astype(np.int16)
    labs = dict(fixed_seg35=labels, moving_seg35=labels, fixed_seg4=seg4, moving_seg4=seg4)
    ev = core.load_module('_mc0_evaluate', 'experiments/d0/evaluate.py')
    metrics = RUN.field_metrics(np.zeros(SHAPE+(3,)), labs, (1., 1., 1.), ev)
    for key in ('seg35', 'seg4', 'small_six', 'large_six', 'other_23', 'cortex', 'white_matter'):
        assert metrics[key] == 1., key
    assert metrics['sdlogj_official'] == 0. and metrics['sdlogj_brain'] == 0.
    assert set(decision.MEASURES) <= set(metrics)
    json.dumps(metrics)


def test_group_means_use_the_three_groups_of_evaluate():
    ev = core.load_module('_mc0_evaluate', 'experiments/d0/evaluate.py')
    assert sorted(ev.A5_GROUPS['small_six']) == [4, 18, 19, 23, 34, 35] and sorted(ev.A5_GROUPS['large_six']) == [1, 2, 5, 20, 21, 24] and len(ev.A5_GROUPS['other_23']) == 23
    per35 = {k: float(k in ev.A5_GROUPS['small_six']) for k in range(1, 36) if k != 4}
    out = RUN.group_means(per35, {1: .7, 3: .8}, ev.A5_GROUPS)
    assert out['small_six'] == 1. and out['large_six'] == 0. and out['cortex'] == .7 and out['white_matter'] == .8
    per35[18] = None                                                                 # dice_labels: a label in neither image
    assert RUN.group_means(per35, {}, ev.A5_GROUPS)['small_six'] == 1.


# ---- the dry run --------------------------------------------------------------------------------------------------------------------------------
def fake_stage(tmp_path, *, missing=None, grid=(26, 28, 30)):
    from jd_curl.three_d.nifti import file_sha256
    stage = tmp_path/'stage_L2R_val_v1'
    stage.mkdir()
    (stage/'selection.json').write_text(json.dumps(dict(pairs=[dict(pair=p) for p in core.PAIRS]+[dict(pair='later')])))
    n = int(np.prod(grid))
    labels = (np.arange(n) // 6 % 36).reshape(grid).astype(np.int16)
    for i, pair in enumerate(core.PAIRS):
        d = stage/pair
        d.mkdir()
        rng = np.random.default_rng(i)
        members = dict(fixed=gaussian_filter(rng.uniform(.1, .7, grid), 1.), moving=gaussian_filter(rng.uniform(.1, .7, grid), 1.), fixed_seg4=(labels % 5).astype(np.int16),
                       fixed_seg35=labels, moving_seg4=np.roll(labels % 5, 1, axis=0).astype(np.int16), moving_seg35=np.roll(labels, 1, axis=0), initial_u=np.zeros(grid+(3,)))
        if missing:
            del members[missing]
        np.savez(d/'pair.npz', **members)
        fixed, moving = pair.split('__')
        (d/'manifest.json').write_text(json.dumps(dict(shape_zyx=list(grid), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm', pair_sha256=file_sha256(d/'pair.npz'),
                                                       fixed_subject=fixed, moving_subject=moving, sources=dict(fixed=dict(sha256='a'*64), moving=dict(sha256='b'*64)))))
    return stage


def test_dry_run_prints_all_45_and_checks_the_members(tmp_path):
    lines = []
    assert RUN.dry_run(stage=fake_stage(tmp_path), out=lines.append) == 0
    text = '\n'.join(lines)
    for cell in core.plan_runs():
        assert cell.run_id in text
    assert 'fixed_seg35 (26, 28, 30)' in text and 'problems: none' in text and 'py'+'test' not in text.replace(str(tmp_path), '')
    assert 'selection.json first 5' in text


def test_dry_run_reads_only_the_header_and_stops_on_a_missing_member(tmp_path):
    stage = fake_stage(tmp_path, missing='moving_seg35')
    with pytest.raises(KeyError):
        RUN.dry_run(stage=stage, out=lambda line: None)
    assert RUN.npz_header(stage/core.PAIRS[0]/'pair.npz', 'fixed_seg4') == ((26, 28, 30), 'int16')


# ---- one synthetic run through everything ------------------------------------------------------------------------------------------------------
PIPELINE = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import mc0_core as core
core.frozen_api()
import run_mc0 as run
stage, root = Path(sys.argv[2]), Path(sys.argv[3])
core.STAGE = stage
core.SETUP_WORKERS = 1
core.TIERS.update(short=dict(iterations=(2, 2, 2, 3)), long=dict(iterations=(2, 2, 2, 4)))
pair = core.PAIRS[0]
for arm, tier in (('P0', 'short'), ('P1', 'long'), ('P3', 'long')):
    cell = core.Cell(arm, tier, pair)
    assert run.main(['run-one', '--arm', arm, '--tier', tier, '--pair', pair, '--out', str(core.run_dir(root, cell)), '--device', 'cpu', '--code-sha', 'x']) == 0
assert run.main(['evaluate', '--root', str(root)]) == 0
assert run.main(['decide', '--root', str(root)]) == 0
"""


@needs_frozen
def test_one_synthetic_run_through_everything(tmp_path):
    stage, root = fake_stage(tmp_path), tmp_path/'mc0_v1'
    done = subprocess.run([PY, '-B', '-c', PIPELINE, str(ROOT/'experiments/mc0'), str(stage), str(root)], env=frozen_env(), cwd=tmp_path, capture_output=True, text=True)    # -c: sys.path[0] is the cwd
    assert done.returncode == 0, done.stdout+done.stderr
    for arm, tier, k in (('P0', 'short', 1), ('P1', 'long', 14), ('P3', 'long', 35)):
        d = core.run_dir(root, core.Cell(arm, tier, core.PAIRS[0]))
        run, metrics = json.loads((d/'run.json').read_text()), json.loads((d/'metrics.json').read_text())
        assert run['status'] == 'done' and run['k'] == k == run['optimizer']['channels'] and run['oracle'] is True and run['scale']['g_norm_scaled'] > 0
        assert run['optimizer']['stopped'] is None and run['field']['sha256'] == core.sha_file(d/'field.npz')
        assert (run['scale']['D_multiplier'] == 1.) == (arm == 'P0')
        norms = run['level_norms']                                                    # r_l: the scale rule fixes the native level, the coarse ones are recorded
        assert len(norms['ratio_raw']) == len(norms['ratio_smoothed']) == len(norms['arm']) == len(norms['P0']) == 4
        assert norms['ratio_raw'][3] == pytest.approx(1., abs=1e-3)
        if arm == 'P0':
            assert norms['ratio_raw'] == [1.]*4 and norms['ratio_smoothed'] == [1.]*4
        else:
            assert any(abs(r-1) > 1e-3 for r in norms['ratio_raw'][:3])
        assert 'level_g_norm' in run['optimizer'] and all('level_g_norm' not in level for level in run['optimizer']['levels'])
        alpha = metrics['certificate']['accepted_alpha']
        assert alpha is None and metrics['at_accepted_alpha'] is None or metrics['at_accepted_alpha']['alpha'] == alpha
        assert set(decision.MEASURES) <= set(metrics['metrics'])
        assert metrics['certificate']['attempts'] and metrics['certificate']['attempts'][0]['alpha'] == 1.
        assert metrics['oracle'] is True and metrics['certificate']['status_at_1'] == metrics['certificate']['attempts'][0]['status']
        first = metrics['certificate']['attempts'][0]
        assert first['config']['tile_size'] == 65536 and first['config']['max_boxes'] == 255 and first['certificate_version'].startswith('q1-3d-parallel-hoisted-edge-hull-float-bernstein-prescreen')
        with np.load(d/'field.npz') as z:
            assert z['u'].shape == (26, 28, 30, 3) and np.abs(z['u']).max() > 0
    assert json.loads((root/'analysis/decision_mc0.json').read_text())['reading'] == decision.READ_UNDECIDABLE


def test_a_run_refuses_to_start_twice(tmp_path):
    (tmp_path/'started.json').write_text('{}')
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import mc0_core as core; core.frozen_api(); import run_mc0 as run; "
            "sys.exit(run.main(['run-one', '--arm', 'P0', '--tier', 'short', '--pair', core.PAIRS[0], '--out', sys.argv[2], '--device', 'cpu']))")
    done = subprocess.run([PY, '-B', '-c', code, str(ROOT/'experiments/mc0'), str(tmp_path)], env=frozen_env(), cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode != 0 and 'never rerun' in done.stderr
