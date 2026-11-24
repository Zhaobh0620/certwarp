"""MC0P (runs/20261104-mc0p/任务书.md, plan.md): experiments/mc0/{mc0p_core,run_mc0p,decide_mc0p}.py and the options they pass to MC0's code; CPU only, synthetic small volumes; no real image, label,
prediction or field is read (the subject ids are synthetic stand-ins, tests/test_bayesreg/io_seg).

- the arms' channels against independent recomputations (argmax and soft, synthetic probabilities); the scale rule, the factor of Q3x3; V of Q3, Q3h from the prediction only;
- the firewall: a spy on np.load sees only fixed, moving, initial_u of pair.npz and p of the predictions; the same spy does stop MC0's oracle reader (the counterexample); a stage without label
  members gives the same bytes and MC0's reader fails on it; changed truth labels change nothing; the source of the channel path names no label member;
- Q0 against the inputs written by make_inputs' A5P, bit for bit; a different or missing file stops the queue;
- the relaxed gate only at this entry: MC0's gate and start strings as they were; a tag-* unit outside the list still blocks; the hold for the chain; the device cap;
- the marks: oracle computed from the inputs, run.json to metrics.json and long.csv; the manual entries refuse a root that is not MC0P's; the gate closed for the last member and past the deadline (plan revision 1);
- the reading, every branch, on made-up numbers; the realization ratios; Q2s - Q2; only the first 5 pairs; the 45 runs; the queue on a fake machine; the start strings;
- the dry run; one synthetic run per arm through run-one, evaluate and the reading, in a subprocess, through the real structure-layer checks.
"""
import ast
import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from jd_curl.three_d.nifti import file_sha256

from bayesreg.io import pair as pair_module
from bayesreg.seg import structure_layer

from . import test_d0_a4_tools as a4
from ..io_seg._helpers import DEV, source_sha, write_json
from ..io_seg._helpers import synthetic_folds  # noqa: E402   (C23.9-13 (8), revision 6e)
from .test_d0_a4_tools import MI, PAIRS, SHAPE, env, seg35  # noqa: F401
from .test_d0_a5_tools import relabel_stage

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT/'experiments/mc0/run_mc0p.py'
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'


def _load():
    spec = importlib.util.spec_from_file_location('_mc0p_run', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules['_mc0p_run'] = module
    spec.loader.exec_module(module)
    return module


RUN = _load()
core, mc, decision, MC0 = RUN.core, RUN.mc, RUN.decision, RUN.run
REAL_CLOCK = RUN.wall_clock
EARLY = datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc)      # 3:00 am CST, before the chain's deadline; a test must not depend on the hour it runs at
needs_frozen = pytest.mark.skipif(not (FROZEN/'jd_curl/three_d/preopt.py').is_file(), reason='the frozen worktree is not on this machine')


@pytest.fixture(autouse=True)
def no_real_results(monkeypatch):
    """Nothing here reads the real chain or MC0's real results."""
    monkeypatch.setattr(RUN, 'UNV2_OUT', Path('/nonexistent/unv2'))
    monkeypatch.setattr(RUN, 'wall_clock', lambda: EARLY)
    monkeypatch.setattr(mc, 'MC0_ROOT', Path('/nonexistent/mc0_v1'))


def g75(x):
    return gaussian_filter(x, sigma=.75, mode='nearest', truncate=4.0)


# ---- the plan ----------------------------------------------------------------------------------------------------------------------------
def test_the_arms_are_the_briefs():
    want = dict(Q0=(1, ('long', 'fine')), Q1=(14, ('long',)), Q2=(35, ('long', 'fine')), Q2s=(35, ('long',)), Q3=(35, ('long',)), Q3x3=(35, ('long',)), Q3h=(35, ('long',)))
    assert {arm: (spec['k'], spec['tiers']) for arm, spec in mc.ARMS.items()} == want
    assert [a for a, s in mc.ARMS.items() if s['decides']] == ['Q1', 'Q2', 'Q2s']
    assert mc.deciding_arms('long') == ['Q1', 'Q2', 'Q2s'] and mc.deciding_arms('fine') == ['Q2']
    assert mc.ARMS['Q3x3']['D_factor'] == 3. and all(s['D_factor'] == 1. for a, s in mc.ARMS.items() if a != 'Q3x3')
    assert set(mc.TIER_NAMES) <= set(core.TIERS) and core.tier_config('long')['iterations'] == (50, 50, 100, 600) and core.tier_config('fine')['sigma_field'] == .5     # MC0's tiers
    assert mc.VRAM_LIMIT_BYTES == 9*2**30


def test_plan_has_45_runs_the_deciding_arms_first():
    plan = mc.plan_runs()
    assert len(plan) == len(set(plan)) == 45 and {c.pair for c in plan} == set(core.PAIRS)
    assert sum(c.arm == 'Q0' for c in plan) == 10 and sum(c.arm == 'Q1' for c in plan) == 5 and sum(mc.ARMS[c.arm]['k'] == 35 for c in plan) == 30
    assert {c.arm for c in plan[:30]} == set(mc.FIRST_PHASE) and {c.arm for c in plan[30:]} == set(mc.SECOND_PHASE)
    assert (plan[0].arm, plan[0].tier, plan[0].pair) == ('Q0', 'long', core.PAIRS[0])       # the first run checks the images
    assert plan[0].run_id == 'mc0p_Q0_long_0438-0439' and plan[0].unit == 'tag-mc0p-q0-long-0438-0439'
    assert mc.Cell('Q3x3', 'long', core.PAIRS[1]).unit == 'tag-mc0p-q3x3-long-0439-0440'
    assert core.run_dir('/r', plan[0]) == Path('/r/runs/Q0_long/0438-0439')
    second = [c.arm for c in plan if c.pair == core.PAIRS[1]][:4]
    assert second[0] == 'Q1'                                                                # the order rotates with the pair


@pytest.mark.parametrize('pair', ('OASIS_OAS1_0443_MR1__OASIS_OAS1_0444_MR1', 'OASIS_OAS1_0020_MR1__OASIS_OAS1_0021_MR1', 'S37__S38', '../x'))
def test_only_the_first_five_pairs(pair, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('a file was opened')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(SystemExit, match='first 5'):
        mc.read_images(pair)
    with pytest.raises(SystemExit, match='first 5'):
        mc.require_cell('Q2', 'long', pair)
    args = RUN.argparse.Namespace(arm='Q2', tier='long', pair=pair, out='/nonexistent/x', device='cpu', code_sha='n/a')
    with pytest.raises(SystemExit, match='first 5'):
        RUN.run_one(args)


@pytest.mark.parametrize('arm,tier', (('Q2s', 'fine'), ('Q1', 'fine'), ('Q3', 'fine'), ('Q3x3', 'fine'), ('Q3h', 'fine'), ('P2', 'long'), ('Q9', 'long'), ('Q2', 'short')))
def test_unknown_cells_are_refused(arm, tier):
    with pytest.raises(SystemExit, match='no cell'):
        mc.require_cell(arm, tier, core.PAIRS[0])


# ---- the channels ------------------------------------------------------------------------------------------------------------------------
GRID = (10, 11, 12)


def probabilities(seed=1, shift=0):
    """A soft (36, ...) posterior with a known argmax: .6 on the map's label, the rest spread."""
    n = int(np.prod(GRID))
    labels = np.roll((np.arange(n) % 36).reshape(GRID), shift, axis=0)
    noise = np.random.default_rng(seed).uniform(0, 1, (36,)+GRID)
    P = .6*(np.arange(36)[:, None, None, None] == labels[None])+.4*noise/noise.sum(axis=0)
    return P/P.sum(axis=0), labels


def predicted_world(soft=True):
    (Pf, Lf), (Pm, Lm) = probabilities(1), probabilities(2, shift=2)
    pin = type('Pin', (), dict(fixed=np.random.default_rng(5).uniform(.1, .7, GRID), moving=np.random.default_rng(6).uniform(.1, .7, GRID)))()
    labels = dict(fixed=mc.predicted_labels(Pf), moving=mc.predicted_labels(Pm))
    assert (labels['fixed'] == Lf).all() and (labels['moving'] == Lm).all()
    return pin, mc.Predicted(labels, dict(fixed=mc.soft_channels(Pf.copy()), moving=mc.soft_channels(Pm.copy())) if soft else None, {}), (Pf, Pm)


def test_argmax_ties_go_to_the_smaller_label():
    P = np.zeros((36, 1, 1, 2))
    P[3], P[7] = .5, .5
    assert mc.predicted_labels(P).tolist() == [[[3, 3]]]


def test_q1_is_the_intensity_and_the_predicted_group_masks():
    pin, predicted, _ = predicted_world(False)
    groups = core.load_groups()
    F, M = mc.arm_channels('Q1', pin, predicted, groups=groups)
    assert F.shape == M.shape == (14,)+GRID
    assert F[0].tobytes() == pin.fixed.tobytes() and M[0].tobytes() == pin.moving.tobytes()
    for c, members in enumerate(groups):
        assert F[c+1].tobytes() == g75(np.isin(predicted.labels['fixed'], members).astype(np.float64)).tobytes()
        assert M[c+1].tobytes() == g75(np.isin(predicted.labels['moving'], members).astype(np.float64)).tobytes()


def test_q2_is_the_one_hot_of_the_argmax():
    pin, predicted, _ = predicted_world(False)
    F, M = mc.arm_channels('Q2', pin, predicted)
    assert F.shape == M.shape == (35,)+GRID
    for label in (1, 18, 35):
        assert F[label-1].tobytes() == g75((predicted.labels['fixed'] == label).astype(np.float64)).tobytes()
        assert M[label-1].tobytes() == g75((predicted.labels['moving'] == label).astype(np.float64)).tobytes()


def test_q2s_is_the_smoothed_posterior_not_the_argmax():
    pin, predicted, (Pf, Pm) = predicted_world()
    F, M = mc.arm_channels('Q2s', pin, predicted)
    assert F.shape == M.shape == (35,)+GRID
    for label in (1, 18, 35):
        assert F[label-1].tobytes() == g75(Pf[label]).tobytes() and M[label-1].tobytes() == g75(Pm[label]).tobytes()
    hard = mc.arm_channels('Q2', pin, predicted)[0]
    assert np.abs(F-hard).max() > .01
    assert mc.soft_channels(Pf.copy(), workers=2).tobytes() == F.tobytes()              # a pool changes nothing
    onehot = (np.arange(36)[:, None, None, None] == predicted.labels['fixed'][None]).astype(np.float64)
    assert mc.soft_channels(onehot).tobytes() == hard.tobytes()                         # a one-hot posterior is Q2 bit for bit


def test_q2s_without_soft_channels_is_refused():
    pin, predicted, _ = predicted_world(False)
    with pytest.raises(SystemExit, match='soft'):
        mc.arm_channels('Q2s', pin, predicted)


def test_q3_weights_are_inverse_volume_and_q3h_inverse_square_root_of_the_given_map():
    _, labels = probabilities()
    labels = labels.copy()
    labels[labels == 7] = 0                                                              # label 7 absent from the fixed image's prediction
    w, voxels = mc.channel_weights('Q3', labels)
    assert voxels[6] == 0 and w[6] == 0 and w[voxels > 0].mean() == pytest.approx(1.)
    assert np.allclose(w[voxels > 0]*voxels[voxels > 0], (w[voxels > 0]*voxels[voxels > 0])[0])
    assert w.tolist() == core.channel_weights('P3', labels)[0].tolist()                   # the rule of P3
    h, voxels_h = mc.channel_weights('Q3h', labels)
    assert voxels_h.tolist() == voxels.tolist() and h[6] == 0 and h[voxels > 0].mean() == pytest.approx(1.)
    assert np.allclose(h[voxels > 0]*np.sqrt(voxels[voxels > 0]), (h[voxels > 0]*np.sqrt(voxels[voxels > 0]))[0])
    assert mc.channel_weights('Q3x3', labels)[0].tolist() == w.tolist()
    for arm in ('Q0', 'Q1', 'Q2', 'Q2s'):
        assert mc.channel_weights(arm, labels) == (None, None)
    with pytest.raises(SystemExit):
        mc.channel_weights('Q3h', np.zeros(GRID))


@pytest.mark.parametrize('arm,k', (('Q1', 14), ('Q2', 35), ('Q2s', 35), ('Q3', 35), ('Q3h', 35)))
def test_scale_rule_gives_the_gradient_norm_of_q0(arm, k):
    pin, predicted, _ = predicted_world()
    table = core.load_table()
    q0 = mc.q0_channels(pin, predicted, table)
    (F, M), record = mc.prepare_stacks(arm, pin, predicted, q0, 'cpu')
    reference = core.data_grad_norm(*q0, 'cpu')
    assert F.shape == M.shape == (k,)+GRID and record['g_norm_Q0'] == reference
    assert core.data_grad_norm(F, M, 'cpu') == pytest.approx(reference, rel=1e-4) and record['g_norm_scaled'] == pytest.approx(reference, rel=1e-4)
    assert record['D_multiplier'] == pytest.approx(reference/record['g_norm_unscaled']) and record['channel_multiplier'] == pytest.approx(np.sqrt(record['D_multiplier']))
    assert record['D_multiplier'] != pytest.approx(1., abs=1e-3) and record['D_factor'] == 1.
    assert (record['channel_weights'] is None) == (arm not in ('Q3', 'Q3h')) and (record['fixed_voxels'] is None) == (arm not in ('Q3', 'Q3h'))


def test_the_factor_three_is_on_top_of_the_scale_rule():
    pin, predicted, _ = predicted_world(False)
    q0 = mc.q0_channels(pin, predicted, core.load_table())
    (F1, M1), r1 = mc.prepare_stacks('Q3', pin, predicted, q0, 'cpu')
    (F3, M3), r3 = mc.prepare_stacks('Q3x3', pin, predicted, q0, 'cpu')
    assert r3['D_factor'] == 3. and r3['D_multiplier'] == pytest.approx(3*r1['D_multiplier']) and r3['g_norm_unscaled'] == r1['g_norm_unscaled']
    assert r3['g_norm_scaled'] == pytest.approx(3*r1['g_norm_Q0'], rel=1e-4)
    assert np.allclose(F3, F1*np.sqrt(3.)) and np.allclose(M3, M1*np.sqrt(3.))
    assert r3['channel_weights'] == r1['channel_weights']


def test_q0_is_never_multiplied():
    pin, predicted, _ = predicted_world(False)
    q0 = mc.q0_channels(pin, predicted, core.load_table())
    stacks, record = mc.prepare_stacks('Q0', pin, predicted, q0, 'cpu')
    assert stacks[0] is q0[0] and stacks[1] is q0[1]
    assert record['D_multiplier'] == record['channel_multiplier'] == record['D_factor'] == 1. and record['channel_weights'] is None


def test_a_broken_scale_stops(monkeypatch):
    pin, predicted, _ = predicted_world(False)
    q0 = mc.q0_channels(pin, predicted, core.load_table())
    real, calls = core.data_grad_norm, []

    def off(F, M, device):
        calls.append(1)
        return real(F, M, device)*(1.01 if len(calls) == 3 else 1.)
    monkeypatch.setattr(core, 'data_grad_norm', off)
    with pytest.raises(SystemExit, match='after scaling'):
        mc.prepare_stacks('Q2', pin, predicted, q0, 'cpu')


def test_level_norm_record_names_the_baseline():
    arm, q0 = [dict(raw=2., smoothed=4.)]*4, [dict(raw=1., smoothed=1.)]*4
    record = mc.level_norm_record(arm, q0)
    assert 'P0' not in record and record['Q0'] == q0 and record['ratio_raw'] == [2.]*4 and record['ratio_smoothed'] == [4.]*4


# ---- the firewall ------------------------------------------------------------------------------------------------------------------------
class LabelRead(Exception):
    """A channel path asked for a label member."""


def spy_on_load(monkeypatch):
    """np.load, with every member asked for recorded {file name: keys}; a member with seg in its name raises.  It sees load_pair's reads, the read-back's, equals_a5_input's."""
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
                raise LabelRead(f'{self.name}: {key}')
            return self.inner[key]

        @property
        def files(self):
            if self.name == 'pair.npz':                  # the firewall never lists that archive's members
                raise LabelRead(f'{self.name}: the member list')
            return self.inner.files

    monkeypatch.setattr(np, 'load', lambda path, **kwargs: Spy(path, real(path, **kwargs)))
    return seen


def world_of(env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(core, 'STAGE', env['stage'])
    monkeypatch.setattr(core, 'ALLOWLIST', env['allowlist'])
    monkeypatch.setattr(mc, 'SEG_ROOT', env['seg_root'])
    monkeypatch.setattr(mc, 'default_unet', lambda: env['unet'])
    assert core.PAIRS[0] == f'{PAIRS[0][0]}__{PAIRS[0][1]}'                             # the stand-in pair has the name of the first pair of the plan
    return core.PAIRS[0]


def channels_of_every_arm(pair):
    """Every arm's two stacks through the entry functions run_one uses, without any optimizer."""
    out = {}
    table = core.load_table()
    for arm in mc.ARMS:
        pin = mc.read_images(pair)
        predicted = mc.read_predicted(pin, soft=arm == 'Q2s')
        q0 = mc.q0_channels(pin, predicted, table)
        stacks, record = mc.prepare_stacks(arm, pin, predicted, q0, 'cpu')
        out[arm] = (stacks[0].tobytes(), stacks[1].tobytes(), record, predicted.record)
    return out


@synthetic_folds('env')
def test_channel_path_reads_only_the_images_and_the_predictions(env, monkeypatch):  # noqa: F811
    pair = world_of(env, monkeypatch)
    MI.build_arm('A5P', env['root']/'inputs', env['seg_root'], stage=env['stage'], allowlist=env['allowlist'], unet=env['unet'])     # its pair.npz holds copies of the labels too
    seen = spy_on_load(monkeypatch)
    out = channels_of_every_arm(pair)
    pin = mc.read_images(pair)
    q0 = mc.q0_channels(pin, mc.read_predicted(pin), core.load_table())
    assert core.equals_a5_input(pair, q0[0][0], q0[1][0], inputs=env['root']/'inputs/A5P'/MI.STAGE_NAME) is True
    assert seen['pair.npz'] == {'fixed', 'moving', 'initial_u'}
    assert seen['probs36.npz'] == {'p'} and seen['structure_groups.npz'] == {'p'}
    for *_, record in out.values():
        assert record['oracle'] is False and record['truth_labels_read'] is False and record['pair_keys_read'] == ['fixed', 'moving', 'initial_u'] and record['label_firewall'] == 'whitelist-v2'
        assert {s['policy'] for s in record['sides'].values()} == {'all_folds_mean'} and all(s['alignment_qc']['passed'] for s in record['sides'].values())
        assert {s['subject'] for s in record['sides'].values()} == set(PAIRS[0])


def test_the_counterexample_the_spy_stops_a_reader_of_labels(env, monkeypatch):  # noqa: F811
    """The check above can fail: on the same files, MC0's oracle reader (same stage, same allowlist) trips it."""
    pair = world_of(env, monkeypatch)
    spy_on_load(monkeypatch)
    with pytest.raises(LabelRead, match='fixed_seg35'):
        MC0.read_pair(pair, ('fixed_seg35',))
    with pytest.raises(LabelRead, match='fixed_seg4'):
        MC0.read_pair(pair, RUN.TRUTH_MEMBERS)               # what the evaluation reads: it does read labels, and only there


def drop_labels(env):  # noqa: F811
    d = env['stage']/core.PAIRS[0]
    with np.load(d/'pair.npz') as z:
        arrays = {k: np.array(z[k]) for k in ('fixed', 'moving', 'initial_u')}
    np.savez_compressed(d/'pair.npz', **arrays)
    manifest = json.loads((d/'manifest.json').read_text())
    manifest['pair_sha256'] = file_sha256(d/'pair.npz')
    write_json(d/'manifest.json', manifest)


@synthetic_folds('env')
def test_a_stage_without_label_members_gives_the_same_channels_and_mc0s_reader_fails_on_it(env, monkeypatch):  # noqa: F811
    pair = world_of(env, monkeypatch)
    before = channels_of_every_arm(pair)
    drop_labels(env)
    after = channels_of_every_arm(pair)
    assert {arm: v[:2] for arm, v in before.items()} == {arm: v[:2] for arm, v in after.items()}
    with pytest.raises(KeyError):
        MC0.read_pair(pair, ('fixed_seg35',))                                            # the counterexample: an oracle path cannot run here


@synthetic_folds('env')
def test_changed_truth_labels_change_no_channel_and_v_comes_from_the_prediction(env, monkeypatch):  # noqa: F811
    pair = world_of(env, monkeypatch)
    before = channels_of_every_arm(pair)
    with np.load(env['stage']/pair/'pair.npz') as z:
        truth = np.array(z['fixed_seg35'])
    relabel_stage(env)
    with np.load(env['stage']/pair/'pair.npz') as z:
        changed = np.array(z['fixed_seg35'])
    assert truth.tobytes() != changed.tobytes()
    after = channels_of_every_arm(pair)
    assert {arm: v[:2] for arm, v in before.items()} == {arm: v[:2] for arm, v in after.items()}
    pin = mc.read_images(pair)
    predicted = mc.read_predicted(pin)
    want = [float(np.count_nonzero(predicted.labels['fixed'] == label)) for label in core.LABELS_35]
    assert before['Q3'][2]['fixed_voxels'] == before['Q3h'][2]['fixed_voxels'] == want


@synthetic_folds('env')
def test_v_is_the_predictions_count_when_prediction_and_truth_differ(env, monkeypatch):  # noqa: F811
    """A prediction whose label counts differ from the truth's: V follows the prediction."""
    pair = world_of(env, monkeypatch)
    subject = PAIRS[0][0]
    d = env['unet'].pred_root/subject
    with np.load(d/'probs36.npz') as z:
        probs = np.array(z['p']).astype(np.float64)
    L = np.argmax(probs, axis=0)
    moved = np.argwhere(L == 2)[:3]                                                      # three voxels of label 2 become label 3: not white matter, the alignment QC does not move
    for z_, y_, x_ in moved:
        probs[:, z_, y_, x_] = 0.
        probs[3, z_, y_, x_] = 1.
    shutil.rmtree(d)
    a4.write_prediction(env['unet'].pred_root, subject, env['models'], probs=(probs/probs.sum(axis=0)).astype(np.float32))
    pin = mc.read_images(pair)
    predicted = mc.read_predicted(pin)
    q0 = mc.q0_channels(pin, predicted, core.load_table())
    record = mc.prepare_stacks('Q3', pin, predicted, q0, 'cpu')[1]
    truth = seg35(subject)
    assert record['fixed_voxels'] == [float(np.count_nonzero(predicted.labels['fixed'] == label)) for label in core.LABELS_35]
    assert record['fixed_voxels'] != [float(np.count_nonzero(truth == label)) for label in core.LABELS_35]


GOOD_SIDES = dict(fixed=dict(segmenter_id='unet35_v1_ens', policy='all_folds_mean'), moving=dict(segmenter_id='unet35_v1_ens', policy='all_folds_mean'))


@pytest.mark.parametrize('keys,sides,oracle,truth', ((['fixed', 'moving', 'initial_u'], GOOD_SIDES, False, False), (['fixed', 'moving'], GOOD_SIDES, False, False),
                                                     (['fixed', 'moving', 'initial_u', 'fixed_seg35'], GOOD_SIDES, True, True), (['fixed', 'moving', 'moving_seg4'], GOOD_SIDES, True, True),
                                                     (['fixed'], GOOD_SIDES, True, False), ([], GOOD_SIDES, True, False), (['fixed', 'moving'], {}, True, False),
                                                     (['fixed', 'moving'], dict(fixed=GOOD_SIDES['fixed']), True, False),
                                                     (['fixed', 'moving'], dict(GOOD_SIDES, moving=dict(segmenter_id='unet35_v2_full', policy='all_folds_mean')), True, False),
                                                     (['fixed', 'moving'], dict(GOOD_SIDES, fixed=dict(segmenter_id='unet35_v1_ens', policy='cross_fit_other_folds')), True, False)))
def test_the_oracle_mark_is_computed_from_the_inputs(keys, sides, oracle, truth):
    assert mc.oracle_mark(keys, sides) == dict(oracle=oracle, truth_labels_read=truth)


@synthetic_folds('env')
def test_the_record_of_the_channels_follows_what_was_read(env, monkeypatch):  # noqa: F811
    pair = world_of(env, monkeypatch)
    pin = mc.read_images(pair)
    record = mc.read_predicted(pin).record
    assert record['oracle'] is False and record['truth_labels_read'] is False
    pin.provenance['keys_read'].append('fixed_seg35')
    record = mc.read_predicted(pin).record
    assert record['oracle'] is True and record['truth_labels_read'] is True and 'fixed_seg35' in record['pair_keys_read']


def test_oracle_of_run_checks_the_record_against_its_sources():
    sources = dict(pair_keys_read=['fixed', 'moving', 'initial_u'], sides=GOOD_SIDES)
    assert mc.oracle_of_run(dict(oracle=False, channel_sources=sources)) is False
    for bad in (dict(oracle=True, channel_sources=sources), dict(oracle=None, channel_sources=sources), dict(channel_sources=sources), dict(oracle=False), dict(oracle=False, channel_sources=None), {},
                dict(oracle=False, channel_sources=dict(sources, pair_keys_read=['fixed', 'moving', 'fixed_seg35'])), dict(oracle=False, channel_sources=dict(sources, sides=dict(GOOD_SIDES, fixed=dict(segmenter_id='x', policy='all_folds_mean'))))):
        with pytest.raises(ValueError):
            mc.oracle_of_run(bad)


def test_the_evaluation_takes_oracle_from_run_json_and_refuses_otherwise(tmp_path, monkeypatch):
    root = tmp_path/'mc0p_v1'
    sources = dict(pair_keys_read=['fixed', 'moving', 'initial_u'], sides=GOOD_SIDES)
    good, bad = mc.Cell('Q2', 'long', core.PAIRS[0]), mc.Cell('Q1', 'long', core.PAIRS[0])
    for cell, oracle, channel in ((good, False, sources), (bad, False, dict(sources, pair_keys_read=['fixed', 'moving', 'fixed_seg35']))):
        d = core.run_dir(root, cell)
        d.mkdir(parents=True)
        write_json(d/'run.json', dict(oracle=oracle, channel_sources=channel))
        (d/'field.npz').write_bytes(b'x')
    calls = []
    monkeypatch.setattr(core, 'frozen_api', lambda: None)
    monkeypatch.setattr(core, 'load_module', lambda *a: None)
    monkeypatch.setattr(MC0, 'read_pair', lambda pair, members: ('pin', members))
    monkeypatch.setattr(MC0, 'evaluate_one', lambda d, cell, labs, pin, fz, ev, certify, config, *, oracle, schema: calls.append((cell, oracle, schema, config.fallback_workers, labs)))
    assert RUN.evaluate_all(RUN.argparse.Namespace(root=str(root))) == 8
    assert calls == [(good, False, 'bayesreg.mc0p.metrics.v1', 4, RUN.TRUTH_MEMBERS)]
    assert 'channel_sources' in (core.run_dir(root, bad)/'metrics_error.json').read_text() and not (core.run_dir(root, good)/'metrics_error.json').exists()


def test_the_evaluations_certificate_is_c_but_for_the_pool():
    ours, theirs = asdict(RUN.certificate_config()), asdict(MC0.certificate_config())
    assert theirs['fallback_workers'] == 8 and ours.pop('fallback_workers') == RUN.EVAL_POOL <= 4
    theirs.pop('fallback_workers')
    assert ours == theirs and ours['delta'] == 0. and ours['max_depth'] == 48 and ours['max_boxes'] == 255 and ours['tile_size'] == 65536
    assert type(RUN.certificate_config()) is type(MC0.certificate_config())


def _source_tree(path):
    return ast.parse(Path(path).read_text())


def test_the_source_of_the_channel_path_names_no_label_member():
    tree = _source_tree(SCRIPT)
    run_one = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'run_one')
    core_tree = _source_tree(ROOT/'experiments/mc0/mc0p_core.py')
    for name, node in (('run_one', run_one), ('mc0p_core', core_tree)):
        words = {x.value for x in ast.walk(node) if isinstance(x, ast.Constant) and isinstance(x.value, str)} | {x.attr for x in ast.walk(node) if isinstance(x, ast.Attribute)} | {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}
        for bad in ('read_pair', 'PAIR_MEMBERS', 'TRUTH_MEMBERS', 'fixed_seg4', 'fixed_seg35', 'moving_seg4', 'moving_seg35'):
            assert bad not in words, (name, bad)
    users = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and any(isinstance(x, ast.Name) and x.id == 'TRUTH_MEMBERS' for x in ast.walk(n))]
    assert users == ['evaluate_all']                                                     # the truth labels enter only there


@synthetic_folds('env')
def test_a_refused_prediction_stops(env, monkeypatch):  # noqa: F811
    """The checks are A5P's: a changed file, a flipped prediction, a subject not admitted."""
    pair = world_of(env, monkeypatch)
    pin = mc.read_images(pair)
    d = env['unet'].pred_root/PAIRS[0][0]
    keep = (d/'probs36.npz').read_bytes()
    (d/'probs36.npz').write_bytes(keep+b'x')
    with pytest.raises(SystemExit):
        mc.read_predicted(pin)
    (d/'probs36.npz').write_bytes(keep)
    shutil.rmtree(env['seg_root']/PAIRS[0][1])
    with pytest.raises(SystemExit, match='Atropos'):
        mc.read_predicted(pin)


@synthetic_folds('env')
def test_a_flipped_prediction_fails_the_alignment_qc(env, monkeypatch):  # noqa: F811
    pair = world_of(env, monkeypatch)
    subject = PAIRS[0][0]
    d = env['unet'].pred_root/subject
    with np.load(d/'probs36.npz') as z:
        probs = np.array(z['p']).astype(np.float32)
    shutil.rmtree(d)
    a4.write_prediction(env['unet'].pred_root, subject, env['models'], probs=probs[:, ::-1, ::-1, ::-1])
    with pytest.raises(SystemExit, match='lignment'):
        mc.read_predicted(mc.read_images(pair))


# ---- Q0 against A5P's inputs ---------------------------------------------------------------------------------------------------------------
@synthetic_folds('env')
def test_q0_is_a5p_input_bit_for_bit_and_a_change_is_seen(env, monkeypatch):  # noqa: F811
    pair = world_of(env, monkeypatch)
    MI.build_arm('A5P', env['root']/'inputs', env['seg_root'], stage=env['stage'], allowlist=env['allowlist'], unet=env['unet'])
    relabel_stage(env)                                                                   # the truth now differs from the prediction
    MI.build_arm('A5', env['root']/'inputs', None, stage=env['stage'], allowlist=env['allowlist'])
    a5p, a5 = env['root']/'inputs/A5P'/MI.STAGE_NAME, env['root']/'inputs/A5'/MI.STAGE_NAME
    pin = mc.read_images(pair)
    q0 = mc.q0_channels(pin, mc.read_predicted(pin), core.load_table())
    assert q0[0].shape == (1,)+SHAPE
    assert core.equals_a5_input(pair, q0[0][0], q0[1][0], inputs=a5p) is True
    assert core.equals_a5_input(pair, q0[0][0], q0[1][0], inputs=a5) is False            # the oracle's images are not the predicted ones
    changed = q0[0][0].copy()
    changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], 2.)
    assert core.equals_a5_input(pair, changed, q0[1][0], inputs=a5p) is False            # one bit
    changed = q0[1][0].copy()
    changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], 2.)
    assert core.equals_a5_input(pair, q0[0][0], changed, inputs=a5p) is False            # one bit of the moving image alone
    assert core.equals_a5_input(pair, q0[0][0], q0[1][0], inputs=env['root']/'nowhere') is None


# ---- the gate ------------------------------------------------------------------------------------------------------------------------------
class Machine:
    def __init__(self):
        self.load, self.other, self.free_gpu, self.free_home, self.avail = .1, [], 20000., 500., 12.

    def load1(self):
        return self.load

    def units(self):
        return list(self.other)

    def gpu_free_mib(self):
        return self.free_gpu

    def home_free_gb(self):
        return self.free_home

    def mem_available_gib(self):
        return self.avail

    def now(self):
        return 0.


class NoAvail(Machine):
    def mem_available_gib(self):
        raise AssertionError("MC0's gate does not ask for MemAvailable")


def next_to_training():
    m = Machine()
    m.load, m.other, m.free_gpu, m.avail = 7.9, ['tag-unv2-chain.service', 'tag-mc0p-queue.service'], 9.5*1024., 7.
    return m


def test_mc0p_gate_runs_next_to_the_training(tmp_path):
    assert MC0.busy_reason(next_to_training(), tmp_path, True, RUN.GATE) is None
    assert RUN.GATE.load_limit == 8. and RUN.GATE.gpu_free_mib == 9728 > mc.VRAM_LIMIT_BYTES/2**20 and RUN.GATE.mem_available_gib == 7.


@pytest.mark.parametrize('field,value,word', (('load', 8., 'load'), ('free_gpu', 9.5*1024-1., 'GPU free'), ('avail', 6.99, 'MemAvailable'), ('free_home', 99., '/home'),
                                              ('other', ['tag-unv2-chain.service', 'tag-other.service'], 'other units'), ('other', ['tag-unv2-chain.service', 'seq-vp-x.service'], 'other units'),
                                              ('other', ['tag-mc0-queue.service'], 'other units'), ('other', ['tag-unv2-chain2.service'], 'other units'), ('other', ['tag-unv2-chain.scope'], 'other units'),
                                              ('other', ['tag-seed-queue.service'], 'other units'), ('other', ['systemctl failed with exit 1'], 'other units')))
def test_mc0p_gate_still_blocks(tmp_path, field, value, word):
    m = next_to_training()
    setattr(m, field, value)
    assert word in MC0.busy_reason(m, tmp_path, True, RUN.GATE)


def test_mc0p_gate_hold_file_and_the_cpu_unit_does_not_ask_for_the_gpu(tmp_path):
    m = next_to_training()
    m.free_gpu = 0.
    assert MC0.busy_reason(m, tmp_path, False, RUN.GATE) is None
    (tmp_path/'hold').write_text('')
    assert MC0.busy_reason(m, tmp_path, False, RUN.GATE) == 'hold file'


@pytest.mark.parametrize('field,value,word', (('load', 1.5, 'load'), ('free_gpu', 15*1024-1., 'GPU free'), ('free_home', 99., '/home'), ('other', ['tag-unv2-chain.service'], 'other units'),
                                              ('other', ['tag-mc0p-queue.service'], 'other units'), ('other', ['seq-vp-x.service'], 'other units')))
def test_mc0s_entry_keeps_its_gate(tmp_path, field, value, word):
    m = NoAvail()
    setattr(m, field, value)
    assert word in MC0.busy_reason(m, tmp_path, True)
    assert MC0.busy_reason(NoAvail(), tmp_path, True) is None
    m = NoAvail()
    m.other = ['tag-mc0-queue.service']
    assert MC0.busy_reason(m, tmp_path, True) is None


def test_mc0s_defaults_are_its_numbers():
    gate = MC0.Gate()
    assert gate == MC0.MC0_GATE and gate.own_units == ('tag-mc0-queue.service',) and gate.load_limit == 1.5 and gate.gpu_free_mib == 15*1024 and gate.mem_available_gib is None and gate.gpu_hold is None
    assert MC0.LOAD_LIMIT == 1.5 and MC0.GPU_FREE_MIB == 15*1024 and MC0.VRAM_LIMIT_BYTES == 14*2**30 and MC0.GPU_CAP_SECONDS == 2.5*3600


def test_only_this_entry_passes_the_relaxed_gate():
    """No call of busy_reason or wait_for_gate in run_mc0.py passes a gate of its own, except the forwarding one."""
    tree = _source_tree(ROOT/'experiments/mc0/run_mc0.py')
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'wait_for_gate']
    assert calls and all(len(c.args) == 5 and not c.keywords for c in calls)              # queue_body: (system, root, gpu, minutes, log), the default
    text = SCRIPT.read_text()
    assert text.count('run.wait_for_gate(') == 3 and text.count(', gate)') == 3           # plan revision 2: queue_body's gate, the first batch's GATE unless it is the retry
    assert 'gate, k1_unit, eval_unit = (GATE, K1_UNIT, EVAL_UNIT) if retry_of is None else (RETRY_GATE, RETRY_K1_UNIT, RETRY_EVAL_UNIT)' in text


def test_the_chain_has_one_unit_and_no_child_unit():
    """The allow list against what chain.sh and the cli use (the brief asks for this check)."""
    chain = (ROOT/'experiments/unv2/chain.sh').read_text()
    assert 'UNIT=tag-unv2-chain\n' in chain
    code_lines = [line for line in chain.splitlines() if line.lstrip().startswith('systemd-run')]
    assert len(code_lines) == 1 and '--unit $UNIT' in code_lines[0]
    assert 'systemd-run' not in (ROOT/'bayesreg/seg/unet/cli.py').read_text()
    assert RUN.UNV2_UNITS == ('tag-unv2-chain.service',)


def chain_files(tmp_path, monkeypatch, *, log=None, state=()):
    monkeypatch.setattr(RUN, 'UNV2_OUT', tmp_path)
    (tmp_path/'chain_state').mkdir()
    for name in state:
        (tmp_path/'chain_state'/name).write_text('x')
    if log is not None:
        (tmp_path/'chain.log').write_text(log)


@pytest.mark.parametrize('log,state,closed', ((None, (), False),
                                              ('12:00:00 loaded OASIS_OAS1_0107_MR1 (A/calib)\n', ('train_A0.started',), False),
                                              ('2026-11-04T10:00:00Z waiting (member): GPU utilization 94% > 20%\n', ('train_A0.done', 'calibrate_A0.done'), True),
                                              ('x\n2026-11-04T10:00:00Z waiting (member): GPU free 1 MiB < 14336 MiB\n\n', (), True),
                                              ('2026-11-04T10:00:00Z waiting (member): x\n2026-11-04T10:01:00Z free after 1 polls\n2026-11-04T10:01:00Z GPU: y\n', (), False),
                                              ('z\n', ('register.started',), True), ('z\n', ('register.done',), True), ('z\n', ('register.done', 'infer.started'), True),
                                              ('z\n', ('register.done', 'infer.started', 'infer.done'), False)))
def test_the_gate_leaves_the_gpu_to_a_waiting_chain(tmp_path, monkeypatch, log, state, closed):
    chain_files(tmp_path, monkeypatch, log=log, state=state)
    reason = RUN.unv2_hold_reason()
    assert (reason is not None) == closed
    m = next_to_training()
    got = MC0.busy_reason(m, tmp_path, True, RUN.GATE)
    assert (got is not None) == closed and (not closed or 'UNV2' in got)
    assert MC0.busy_reason(m, tmp_path, False, RUN.GATE) is None                          # the evaluation does not use the GPU


@pytest.mark.parametrize('state,now,word', ((('train_B1.started',), EARLY, 'train_B1.started'), (('train_B1.started', 'train_B1.done', 'register.done', 'infer.done'), EARLY, 'train_B1.started'),
                                             ((), datetime(2026, 11, 4, 14, 15, tzinfo=timezone.utc), 'past 2026-11-04T08:15:00-06:00'), ((), datetime(2026, 11, 4, 8, 15, tzinfo=timezone(timedelta(hours=-6))), 'past 2026-11-04T08:15:00-06:00'),
                                             (('infer.done',), datetime(2026, 11, 5, 1, 0, tzinfo=timezone.utc), 'past 2026-11-04T08:15:00-06:00')))
def test_no_new_gpu_run_once_the_last_member_started_or_past_the_deadline(tmp_path, monkeypatch, state, now, word):
    chain_files(tmp_path, monkeypatch, log='z\n', state=state)
    monkeypatch.setattr(RUN, 'wall_clock', lambda: now)
    assert word in RUN.unv2_hold_reason()
    m = next_to_training()
    assert word in MC0.busy_reason(m, tmp_path, True, RUN.GATE)
    assert MC0.busy_reason(m, tmp_path, False, RUN.GATE) is None                          # GPU units only: the evaluation still runs


@pytest.mark.parametrize('state', ((), ('train_A0.started', 'train_A0.done', 'train_B0.started'), ('train_A1.started', 'calibrate_A1.done')))
def test_the_earlier_members_and_a_minute_before_the_deadline_leave_it_open(tmp_path, monkeypatch, state):
    chain_files(tmp_path, monkeypatch, log='z\n', state=state)
    monkeypatch.setattr(RUN, 'wall_clock', lambda: datetime(2026, 11, 4, 14, 14, 59, tzinfo=timezone.utc))
    assert RUN.unv2_hold_reason() is None


def test_the_deadline_and_the_mark_are_the_chains():
    chain = (ROOT/'experiments/unv2/chain.sh').read_text()
    members = next(line for line in chain.splitlines() if line.startswith('MEMBERS=('))
    assert RUN.LAST_MEMBER_MARK == f"train_{members[len('MEMBERS=('):-1].split()[-1]}.started" and 'step "train_$m"' in chain
    assert RUN.NO_NEW_GPU_RUN_AFTER == datetime(2026, 11, 4, 14, 15, tzinfo=timezone.utc)   # the running chain's --no-start-after 2026-11-04T08:15:00-06:00 (8:15 am CST)
    assert REAL_CLOCK().tzinfo is not None


def test_mc0_does_not_ask_the_chain(tmp_path, monkeypatch):
    chain_files(tmp_path, monkeypatch, log='z\n', state=('train_B1.started',))
    assert MC0.busy_reason(NoAvail(), tmp_path, True) is None


def test_the_chains_words_are_the_ones_the_hold_reads():
    chain = (ROOT/'experiments/unv2/chain.sh').read_text()
    assert 'log "waiting ($1): $reason"' in chain and 'wait_until_free member' in chain and 'step register' in chain and 'step infer' in chain and 'utc > "$STATE/$name.started"' in chain and 'utc > "$STATE/$name.done"' in chain


def test_mem_available_reads_meminfo_and_no_reading_closes(monkeypatch):
    assert MC0.System().mem_available_gib() > 0
    monkeypatch.setattr(Path, 'read_text', lambda self, *a, **k: 'MemTotal: 1 kB\n')
    assert MC0.System().mem_available_gib() == 0.


# ---- the device cap ------------------------------------------------------------------------------------------------------------------------
class FakeTorch:
    def __init__(self, total=24564*2**20):
        self.calls, self.total = [], total
        self.cuda = self

    device = staticmethod(lambda name: __import__('torch').device(name))

    def get_device_properties(self, index):
        return type('Props', (), dict(total_memory=self.total))()

    def set_per_process_memory_fraction(self, fraction, index):
        self.calls.append((fraction, index))


def test_device_cap_is_nine_gib_and_off_the_gpu_nothing():
    torch = FakeTorch()
    record = RUN.limit_device_memory(torch, 'cuda')
    assert torch.calls == [(9*2**30/torch.total, 0)] and record['limit_bytes'] == 9*2**30 and record['fraction'] == pytest.approx(9*1024/24564)
    cpu = FakeTorch()
    assert RUN.limit_device_memory(cpu, 'cpu') is None and cpu.calls == []
    small = FakeTorch(total=8*2**30)
    assert RUN.limit_device_memory(small, 'cuda')['fraction'] == 1.


def test_the_cap_is_set_before_any_channel_is_read():
    tree = _source_tree(SCRIPT)
    run_one = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'run_one')
    names = [n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(run_one) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))]
    lines = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr: n.lineno for n in ast.walk(run_one) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    assert 'limit_device_memory' in names and lines['limit_device_memory'] < lines['read_predicted'] < lines['optimize_k']


# ---- start strings -------------------------------------------------------------------------------------------------------------------------
def test_start_strings():
    code, root = Path('/home/rmcfz/seq_vp_tag_results/mc0p_v1/code'), RUN.DEFAULT_ROOT
    strings = [RUN.queue_launch(), RUN.k1_argv(code, root), RUN.eval_argv(code, root)]+[RUN.run_argv(c, code, root) for c in mc.plan_runs()]
    for argv in strings:
        text = shlex.join(argv)
        assert 'py'+'test' not in text
        assert text.startswith('systemd-run --user --unit=tag-mc0p-') and 'seq-vp' not in text and '--property=MemoryMax=8G' in text and '--nice=5' in text and 'run_mc0p.py' in text
    for argv in strings[1:]:
        text = shlex.join(argv)
        assert f'--setenv=PYTHONPATH={core.FROZEN}:{code}' in text and '--wait' in text and '--collect' in text and PY+' -B' in text and '--property=MemorySwapMax=4G' in text
    evaluation = shlex.join(RUN.eval_argv(code, root))
    assert '--setenv=CUDA_VISIBLE_DEVICES=' in evaluation and '--setenv=NUMBA_NUM_THREADS=1' in evaluation and '--setenv=OMP_NUM_THREADS=1' in evaluation and '--property=CPUQuota=100%' in evaluation
    assert 'CUDA_VISIBLE_DEVICES' not in shlex.join(RUN.run_argv(mc.plan_runs()[0], code, root))
    assert str(root) == '/home/rmcfz/seq_vp_tag_results/mc0p_v1'


def test_mc0s_start_string_has_not_changed():
    got = MC0.unit_argv('tag-mc0-x', Path('/c'), ['a'], log='/l')
    assert got == ['systemd-run', '--user', '--unit=tag-mc0-x', '--wait', '--collect', '--working-directory=/c', '--property=MemoryMax=9G', '--property=MemorySwapMax=4G', '--property=CPUQuota=800%', '--property=RuntimeMaxSec=1200',
                   '--property=StandardOutput=append:/l', '--property=StandardError=append:/l', '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=NUMEXPR_NUM_THREADS=1',
                   '--setenv=NUMBA_NUM_THREADS=1', '--setenv=PYTHONDONTWRITEBYTECODE=1', f'--setenv=PYTHONPATH={core.FROZEN}:/c', MC0.PY, '-B', '/c/experiments/mc0/run_mc0.py', 'a']


def test_a_start_string_with_the_word_is_refused():
    with pytest.raises(ValueError):
        RUN.unit_argv('tag-mc0p-x', Path('/tmp/py'+'test-of-x/code'), [], log='/tmp/x')


def test_nothing_here_stops_or_signals_a_unit():
    """Neither kill nor pause the training: the source has no such word."""
    for name in ('run_mc0p.py', 'mc0p_core.py', 'decide_mc0p.py'):
        text = (ROOT/'experiments/mc0'/name).read_text()
        for word in ('systemctl', 'kill', 'SIGSTOP', 'freeze', 'os.kill'):
            assert word not in text, (name, word)


# ---- the reading ---------------------------------------------------------------------------------------------------------------------------
BASE = .85


def row(seg35=BASE, sd=.07, status='done'):  # noqa: F811
    metrics = {m: .5 for m in decision.MEASURES}
    metrics.update(seg35=seg35, seg4=seg35+.05, sdlogj_official=sd)
    return dict(status=status, metrics=metrics, run_id='x', D_factor=1., oracle=False)


def table(q1=None, q2=None, q2s=None, q2_fine=None, q0_sd=.07, q0_fine_sd=None):
    """Differences against Q0 per pair; an arm left out is well below Q0."""
    rows = {}
    low = [-.1]*5
    for i, pair in enumerate(core.PAIRS):
        rows[('Q0', 'long', pair)] = row(sd=q0_sd)
        rows[('Q0', 'fine', pair)] = row(sd=q0_sd if q0_fine_sd is None else q0_fine_sd)
        rows[('Q1', 'long', pair)] = row(BASE+(q1 or low)[i])
        rows[('Q2', 'long', pair)] = row(BASE+(q2 or low)[i])
        rows[('Q2s', 'long', pair)] = row(BASE+(q2s or low)[i])
        rows[('Q2', 'fine', pair)] = row(BASE+(q2_fine or low)[i])
    return rows


def test_no_headroom_small_deltas_in_both_tiers():
    result = decision.decide(table(q1=[.0048]*5, q2=[.003]*5, q2s=[.0049]*5, q2_fine=[.0049]*5))
    assert result['reading'] == decision.READ_NONE == '预测标签上看不到通道余量' and result['reading_basis'] == 'effect' and result['is_conclusion'] is False and result['scope_note'] is None and result['oracle'] is False
    assert result['tiers']['long']['best_arm'] == 'Q2s' and result['tiers']['fine']['best_arm'] == 'Q2'
    assert all(r['no_headroom'] and not r['headroom'] for r in result['tiers'].values())


def test_the_best_arm_ties_go_to_the_smaller_name():
    assert decision.decide(table(q1=[.0049]*5, q2s=[.0049]*5))['tiers']['long']['best_arm'] == 'Q1'


def test_no_headroom_large_delta_but_fewer_than_4_positive():
    d = [.05, .05, .05, -.01, -.01]
    assert decision.decide(table(q2=d, q2_fine=d))['reading'] == decision.READ_NONE


def test_headroom_in_one_tier_is_enough():
    result = decision.decide(table(q2_fine=[.01]*5))
    assert result['reading'] == decision.READ_SEE == '预测标签上看得到通道余量' and result['tiers']['fine']['headroom'] and not result['tiers']['long']['headroom']
    result = decision.decide(table(q1=[.02]*5))
    assert result['reading'] == decision.READ_SEE and result['tiers']['long']['best_arm'] == 'Q1'


@pytest.mark.parametrize('differences', ([.0099]*5, [.0051]*5, [.03, .03, .03, .03, -.01]))
def test_middle_band(differences):
    result = decision.decide(table(q2=differences))
    assert result['reading'] == decision.READ_MIDDLE and result['reading_basis'] == 'effect'


def test_no_headroom_asks_every_read_arm_and_headroom_only_the_best():
    d = table(q1=[.06, .06, .06, -.02, -.02], q2=[.015]*5)
    result = decision.decide(d)
    assert result['tiers']['long']['best_arm'] == 'Q1' and not result['tiers']['long']['headroom'] and result['tiers']['long']['other_arms_at_headroom_bar'] == ['Q2']
    assert result['reading'] == decision.READ_MIDDLE and 'Q2' in result['scope_note']


def over(rows, arm, tier='long', sd=.1001):
    for pair in core.PAIRS:
        rows[(arm, tier, pair)]['metrics']['sdlogj_official'] = sd
    return rows


def test_an_arm_over_the_sdlogj_bar_blocks_no_headroom_and_cannot_make_headroom():
    result = decision.decide(over(table(q2=[.03]*5), 'Q2'))
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ and result['reading_basis'] == 'not_read' and 'Q2' in result['tiers']['long']['excluded'] and not result['tiers']['long']['headroom']
    assert 'Q2' in result['scope_note']


@pytest.mark.parametrize('how', ('missing', 'time_cap', 'vram_stop', 'failed', 'not evaluated'))
def test_one_arm_short_of_a_pair_blocks_no_headroom(how):
    rows = table()
    key = ('Q2s', 'long', core.PAIRS[4])
    if how == 'missing':
        del rows[key]
    elif how == 'not evaluated':
        rows[key]['metrics'] = None
    else:
        rows[key]['status'] = how
    result = decision.decide(rows)
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ and result['tiers']['long']['predicate_on_read_arms'] is True and not result['tiers']['long']['no_headroom']
    assert result['arms_incomplete'] == ['Q2s'] and 'Q2s' in result['scope_note']


def test_an_arm_not_read_does_not_undo_headroom_of_a_read_arm():
    rows = table(q2=[.03]*5)
    del rows[('Q2s', 'long', core.PAIRS[0])]
    assert decision.decide(rows)['reading'] == decision.READ_SEE


def test_q0_over_the_bar_is_not_comparable():
    result = decision.decide(table(q0_fine_sd=.11))
    assert not result['tiers']['fine']['comparable'] and 'Q0' in result['tiers']['fine']['reason'] and result['tiers_not_comparable']['fine']['k_arms_not_compared'] == ['Q2']
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ and result['reading_basis'] == 'not_read' and 'Q2' in result['scope_note']


def test_nothing_comparable_is_undecidable():
    result = decision.decide(table(q0_sd=.2))
    assert result['reading'] == decision.READ_UNDECIDABLE and result['reading_basis'] == 'not_read'
    assert decision.decide({})['reading'] == decision.READ_UNDECIDABLE


def test_the_arms_that_only_describe_never_decide():
    rows = table()
    for arm in ('Q3', 'Q3x3', 'Q3h'):
        for pair in core.PAIRS:
            rows[(arm, 'long', pair)] = row(BASE+.2)
    result = decision.decide(rows)
    assert result['reading'] == decision.READ_NONE and result['tiers']['long']['best_arm'] not in ('Q3', 'Q3x3', 'Q3h')
    assert set(result['tiers']['long']['arm_effects']) == {'Q1', 'Q2', 'Q2s'}
    assert result['tables']['long']['Q3']['against_Q0']['seg35']['mean'] == pytest.approx(.2) and result['described']['arms'] == ['Q3', 'Q3x3', 'Q3h']   # but they are described


def test_the_selection_effect_counts_this_plan():
    effect = decision.decide(table())['selection_effect']
    assert effect['tiers'] == 2 and effect['arm_tier_cells'] == 4 and effect['most_arms_in_a_tier'] == 3 and effect['rough_upper_bound'] == pytest.approx(4/32) and '〔推断〕' in effect['text'] and '0.125' in effect['text']


def test_the_working_point_is_named_after_q0():
    point = decision.decide(table(q2=[.02]*5))['tiers']['long']['working_point']['sdlogj_official']
    assert set(point) == {'best', 'Q0', 'delta'}


def mc0_rows(deltas, seg4_deltas=None):
    """Delta_P per (arm, tier): MC0's P arms against P0."""
    rows = {}
    for (arm, tier), d in deltas.items():
        for i, pair in enumerate(core.PAIRS):
            rows[('P0', tier, pair)] = dict(metrics=dict(seg35=.80, seg4=.85))
            rows[(arm, tier, pair)] = dict(metrics=dict(seg35=.80+d[i], seg4=.85+(seg4_deltas or d)[i]))
    return rows


def test_realization_ratios_pair_by_pair():
    q = table(q1=[.01]*5, q2=[.02, .02, .02, .02, -.01], q2_fine=[.03]*5)
    p = mc0_rows({('P1', 'long'): [.04]*5, ('P2', 'long'): [.04, .04, 0., -.01, .04], ('P2', 'fine'): [.06]*5})
    result = decision.decide(q, mc0_rows=p)
    rows = {(r['q_arm'], r['tier']): r for r in result['described']['realization']}
    assert list(rows) == [('Q1', 'long'), ('Q2', 'long'), ('Q2', 'fine')] and [r['p_arm'] for r in rows.values()] == ['P1', 'P2', 'P2']
    assert rows[('Q1', 'long')]['seg35']['ratio_of_means'] == pytest.approx(.25) and rows[('Q2', 'fine')]['seg35']['ratio_of_means'] == pytest.approx(.5)
    assert rows[('Q2', 'fine')]['seg35']['per_pair']['0438-0439']['ratio'] == pytest.approx(.5)
    per = rows[('Q2', 'long')]['seg35']['per_pair']
    assert per['0438-0439']['ratio'] == pytest.approx(.5) and per['0442-0443']['ratio'] == pytest.approx(-.25)
    assert per['0440-0441']['ratio'] is None and per['0441-0442']['ratio'] is None         # Delta_P not positive: nothing to realize
    assert 'seg4' in rows[('Q1', 'long')] and result['described']['role'] == '只描述，不进判据'
    assert decision.decide(q, mc0_rows=p)['reading'] == decision.decide(q)['reading']       # no part of the reading


def test_realization_without_mc0_says_so():
    described = decision.decide(table())['described']
    assert described['realization'][0]['seg35']['n'] == 0 and described['realization'][0]['seg35']['ratio_of_means'] is None and '没有读到 MC0' in described['realization_note']


def test_the_reading_says_oracle_from_the_rows():
    rows = table()
    assert decision.decide(rows)['oracle'] is False
    rows[('Q2', 'long', core.PAIRS[0])]['oracle'] = True
    assert decision.decide(rows)['oracle'] is True


def test_soft_minus_hard():
    described = decision.decide(table(q2=[.01]*5, q2s=[.03, .03, .03, .03, 0.]))['described']['q2s_minus_q2']
    assert described['seg35']['mean'] == pytest.approx(.014) and described['seg35']['positive'] == 4 and described['seg35']['n'] == 5 and described['seg35']['per_pair']['0442-0443'] == pytest.approx(-.01)


def test_written_analysis_keeps_the_earlier_version_and_marks_the_fields(tmp_path, monkeypatch):
    root = tmp_path/'mc0p_v1'
    cell = mc.Cell('Q2', 'long', core.PAIRS[0])
    d = core.run_dir(root, cell)
    d.mkdir(parents=True)
    write_json(d/'run.json', dict(status='done', oracle=False, scale=dict(D_multiplier=2., D_factor=1., images_equal_a5p_inputs=True)))
    metrics = {m: .5 for m in decision.MEASURES}
    write_json(d/'metrics.json', dict(metrics=metrics, certificate=dict(accepted_alpha=1., n_infeasible_at_1=0, status_at_1='certified_feasible')))
    first = decision.write_analysis(root)
    assert first['reading'] == decision.READ_UNDECIDABLE
    with open(root/'analysis/long.csv') as f:
        rows = list(csv.DictReader(f))
    assert rows and all(r['oracle'] == 'false' and r['field'] == 'uncertified_alpha1' and r['K'] == '35' and r['run_id'] == 'mc0p_Q2_long_0438-0439' and r['D_factor'] == '1.0' for r in rows)
    text = (root/'analysis/summary_mc0p.md').read_text()
    assert '不可判' in text and 'oracle: false' in text and '.8290' in text and '只描述，不进判据' in text and '| Q2 | 是 | ×1 |' in text
    assert json.loads((root/'analysis/decision_mc0p.json').read_text())['oracle'] is False
    decision.write_analysis(root)
    assert len(list((root/'analysis').glob('previous_*/decision_mc0p.json'))) == 1


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob('*')) if p.is_file()}


def test_a_run_with_metrics_that_is_not_oracle_false_writes_nothing(tmp_path):
    root = tmp_path/'mc0p_v1'
    d = core.run_dir(root, mc.Cell('Q2', 'long', core.PAIRS[0]))
    d.mkdir(parents=True)
    write_json(d/'run.json', dict(status='done', oracle=True, scale=dict(D_multiplier=2., D_factor=1.)))
    write_json(d/'metrics.json', dict(metrics={m: .5 for m in decision.MEASURES}, certificate={}))
    (root/'analysis').mkdir()
    (root/'analysis/long.csv').write_text('earlier\n')
    before = snapshot(root)
    with pytest.raises(SystemExit, match='not oracle false'):
        decision.write_analysis(root)
    assert snapshot(root) == before


@pytest.mark.parametrize('command', ('decide', 'evaluate'))
@pytest.mark.parametrize('name,batch', (('mc0_v1', None), ('mc0_v1', dict(schema='bayesreg.mc0.batch.v1')), ('mc0p_v1', dict(schema='bayesreg.mc0.batch.v1')), ('other', None)))
def test_the_manual_entries_refuse_a_root_that_is_not_mc0ps(tmp_path, capsys, command, name, batch):
    root = tmp_path/name
    (root/'analysis').mkdir(parents=True)
    (root/'analysis/long.csv').write_text('run_id,arm\nmc0_P2_long_0438-0439,P2\n')
    (root/'analysis/decision_mc0p.json').write_text('{}\n')
    if batch:
        write_json(root/'batch.json', batch)
    before = snapshot(root)
    assert RUN.main([command, '--root', str(root)]) == 2
    assert snapshot(root) == before and not list(root.rglob('previous_*'))                 # nothing moved
    assert 'MC0P' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        decision.write_analysis(root)
    assert snapshot(root) == before


def test_mc0s_real_root_is_refused_by_its_name_alone():
    assert 'mc0p' in decision.root_problem('/home/rmcfz/seq_vp_tag_results/mc0_v1')        # the name comes first: nothing under it is opened


def test_this_probes_root_passes(tmp_path):
    root = tmp_path/'mc0p_v1'
    root.mkdir()
    assert decision.root_problem(root) is None
    write_json(root/'batch.json', dict(schema=mc.BATCH_SCHEMA))
    assert decision.root_problem(root) is None
    assert RUN.main(['decide', '--root', str(root)]) == 0 and (root/'analysis/decision_mc0p.json').is_file()


def test_the_note_says_what_the_brief_says_it_cannot_show():
    for words in ('oracle: false', '不是方法', 'T88 没有原生层', '可认证', '不是同一个东西', '.8290 / .8894'):
        assert words in decision.NOTE


# ---- the queue on a fake machine -------------------------------------------------------------------------------------------------------------
GOOD = dict(status='done', oracle=False, scale=dict(images_equal_a5p_inputs=True))


class FakeSystem(Machine):
    def __init__(self, behaviour=None, k1=True):
        super().__init__()
        self.launched, self.clock, self.behaviour, self.k1, self.evaluate = [], 0., behaviour or (lambda cell: (0, dict(GOOD), 10.)), k1, None
        self.other = ['tag-unv2-chain.service']
        self.load, self.free_gpu = 5., 10000.

    def units(self):
        return [f'{RUN.QUEUE_UNIT}.service']+list(self.other)

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
            MC0.write_json(argv[argv.index('--out')+1], dict(bitwise=self.k1))
            return 0
        if unit == RUN.EVAL_UNIT:
            return self.evaluate(argv) if self.evaluate else 0
        cell = mc.require_cell(argv[argv.index('--arm')+1], argv[argv.index('--tier')+1], argv[argv.index('--pair')+1])
        rc, record, seconds = self.behaviour(cell)
        self.clock += seconds
        if record is not None:
            MC0.write_json(Path(argv[argv.index('--out')+1])/'run.json', record)
        return rc


@pytest.fixture
def root():
    base = tempfile.mkdtemp(prefix='mc0pq_')             # not the test runner's tmp_path: its name would fail the start-string check
    yield Path(base)/'mc0p_v1'
    shutil.rmtree(base)


def options(root):  # noqa: F811
    return RUN.argparse.Namespace(root=str(root), commit=None, wait_minutes=30)


RUN_UNITS = {c.unit for c in mc.plan_runs()}


def test_queue_runs_the_plan_next_to_the_training_and_a_restart_reruns_nothing(root):
    system = FakeSystem()
    assert RUN.queue(options(root), system) == 0
    assert system.launched[0] == RUN.K1_UNIT and system.launched[-1] == RUN.EVAL_UNIT and system.launched[1:-1] == [c.unit for c in mc.plan_runs()]
    assert not any('unv2' in unit for unit in system.launched)
    assert (root/'analysis/decision_mc0p.json').is_file() and (root/'queue_end.json').is_file()
    batch = json.loads((root/'batch.json').read_text())
    assert batch['schema'] == mc.BATCH_SCHEMA and batch['caps']['device_gib'] == 9. and batch['gate']['own_units'] == ['tag-mc0p-queue.service', 'tag-unv2-chain.service'] and len(batch['plan']) == 45
    assert batch['gate']['chain_hold']['last_member_mark'] == 'train_B1.started' and batch['gate']['chain_hold']['no_new_gpu_run_after'] == '2026-11-04T08:15:00-06:00' and batch['gate']['gpu_free_mib'] == 9728
    assert batch['evaluation'] == dict(numba_threads=1, cpu_quota_percent=100, fallback_workers=4)
    first = core.run_dir(root, mc.plan_runs()[0])
    assert (first/'cmd.txt').is_file() and json.loads((first/'queue_done.json').read_text())['state'] == 'done'
    again = FakeSystem()
    assert RUN.queue(options(root), again) == 0
    assert again.launched == [RUN.EVAL_UNIT]


def test_mc0s_queue_does_not_run_next_to_the_training():
    """The same machine: MC0's entry waits and stops."""
    base = Path(tempfile.mkdtemp(prefix='mc0q_'))
    try:
        system = FakeSystem()
        system.units = lambda: [f'{MC0.QUEUE_UNIT}.service', 'tag-unv2-chain.service']
        assert MC0.queue(options(base/'mc0_v1'), system) == 8
        assert system.launched == [] and 'other units' in json.loads((base/'mc0_v1/queue_stop.json').read_text())['reason']
    finally:
        shutil.rmtree(base)


def test_the_root_must_be_this_probes(tmp_path, capsys):
    assert RUN.queue(options(tmp_path/'mc0_v1'), FakeSystem()) == 2
    assert 'mc0p*' in capsys.readouterr().err


@pytest.mark.parametrize('equal', (False, None))
def test_q0_not_a5p_stops_the_queue_at_the_first_run(root, equal):
    system = FakeSystem(lambda cell: (0, dict(status='input_mismatch', reason='x', scale=dict(images_equal_a5p_inputs=equal)), 5.))
    assert RUN.queue(options(root), system) == 8
    assert system.launched == [RUN.K1_UNIT, mc.plan_runs()[0].unit]
    assert 'A5P' in json.loads((root/'queue_stop.json').read_text())['reason'] and not (root/'analysis').exists()


def test_a_k_gt_1_run_with_a_mismatched_reference_stops_too(root):
    def behaviour(cell):
        return 0, dict(status='input_mismatch', reason='x', scale=dict(images_equal_a5p_inputs=False)) if cell.arm == 'Q2s' else dict(GOOD), 5.
    system = FakeSystem(behaviour)
    assert RUN.queue(options(root), system) == 8
    assert system.launched[-1].startswith('tag-mc0p-q2s-') and RUN.EVAL_UNIT not in system.launched


def test_a_run_whose_record_is_not_oracle_false_stops_the_queue(root):
    system = FakeSystem(lambda cell: (0, dict(status='not_predicted_channels', reason='x', oracle=True, scale=dict(images_equal_a5p_inputs=True)), 5.))
    assert RUN.queue(options(root), system) == 8
    assert system.launched == [RUN.K1_UNIT, mc.plan_runs()[0].unit] and 'oracle True' in json.loads((root/'queue_stop.json').read_text())['reason'] and not (root/'analysis').exists()


def test_k1_mismatch_stops_before_any_run(root):
    system = FakeSystem(k1=False)
    assert RUN.queue(options(root), system) == 8
    assert system.launched == [RUN.K1_UNIT]


def test_a_device_oom_stops_only_that_arm(root):
    def behaviour(cell):
        return 0, dict(status='cuda_oom', reason='oom', oracle=False, scale=dict(images_equal_a5p_inputs=True)) if cell.arm == 'Q2s' else dict(GOOD), 10.
    system = FakeSystem(behaviour)
    assert RUN.queue(options(root), system) == 0
    assert sum(u.startswith('tag-mc0p-q2s-') for u in system.launched) == 1 and sum(u.startswith('tag-mc0p-q2-') for u in system.launched) == 10
    stopped = json.loads((root/'queue_end.json').read_text())['ledger']['arms_stopped']
    assert list(stopped) == ['Q2s'] and 'cuda_oom' in stopped['Q2s']['reason']


def test_a_device_peak_over_nine_gib_stops_any_arm(root):
    def behaviour(cell):
        return 0, dict(GOOD, device_peak_reserved_gib=9.5 if cell.arm == 'Q0' else 9.), 10.
    system = FakeSystem(behaviour)
    assert RUN.queue(options(root), system) == 0
    assert sum(u.startswith('tag-mc0p-q0-') for u in system.launched) == 1 and sum(u.startswith('tag-mc0p-q1-') for u in system.launched) == 5


def test_fifteen_minutes_stop_a_k35_arm_but_not_k14(root):
    def behaviour(cell):
        return 0, dict(GOOD), (950. if cell.arm in ('Q1', 'Q3h') else 10.)
    system = FakeSystem(behaviour)
    assert RUN.queue(options(root), system) == 0
    assert sum(u.startswith('tag-mc0p-q3h-') for u in system.launched) == 1 and sum(u.startswith('tag-mc0p-q1-') for u in system.launched) == 5


def test_the_run_unit_budget_starts_no_new_run_and_the_rest_is_still_evaluated(root):
    system = FakeSystem(lambda cell: (0, dict(GOOD), 400.))
    assert RUN.queue(options(root), system) == 0
    assert len(system.launched)-2 == 23 and system.launched[-1] == RUN.EVAL_UNIT and 'cap of' in json.loads((root/'queue_end.json').read_text())['end']
    assert all(u.split('-')[2] in ('q0', 'q1', 'q2', 'q2s') for u in system.launched[1:-1])          # the cap costs description first


def test_failures_are_kept_not_rerun_and_three_in_a_row_stop(root):
    system = FakeSystem(lambda cell: (1, None, 5.))
    assert RUN.queue(options(root), system) == 8
    assert len(system.launched) == 1+3 and '3 failures' in json.loads((root/'queue_stop.json').read_text())['reason']
    again = FakeSystem()
    assert RUN.queue(options(root), again) == 0
    assert RUN.K1_UNIT not in again.launched and not set(system.launched[1:]) & set(again.launched)


def test_a_k35_run_failing_without_a_record_stops_its_arm_and_the_others_go_on(root):
    def behaviour(cell):
        return (1, None, 5.) if cell.arm == 'Q2' else (0, dict(GOOD), 5.)
    system = FakeSystem(behaviour)
    assert RUN.queue(options(root), system) == 0
    assert sum(u.startswith('tag-mc0p-q2-') for u in system.launched) == 1 and sum(u.startswith('tag-mc0p-q2s-') for u in system.launched) == 5


@pytest.mark.parametrize('field,value,word', (('load', 8., 'load'), ('avail', 6., 'MemAvailable'), ('free_gpu', 8000., 'GPU free'), ('other', ['tag-unv2-chain.service', 'tag-other.service'], 'other units')))
def test_a_gate_closed_for_30_minutes_stops_with_a_record_and_touches_nothing(root, field, value, word):
    system = FakeSystem()
    setattr(system, field, value)
    assert RUN.queue(options(root), system) == 8
    assert system.launched == [] and word in json.loads((root/'queue_stop.json').read_text())['reason'] and system.clock >= 30*60


def test_a_hold_file_stops_the_queue(root):
    root.mkdir(parents=True)
    (root/'hold').write_text('')
    system = FakeSystem()
    assert RUN.queue(options(root), system) == 8
    assert 'hold file' in json.loads((root/'queue_stop.json').read_text())['reason']


def test_the_queue_waits_while_the_chain_waits_then_goes_on(root, tmp_path, monkeypatch):
    chain_files(tmp_path, monkeypatch, log='x\n2026-11-04T10:00:00Z waiting (member): GPU utilization 94% > 20%\n')
    system = FakeSystem()
    system.sleep = lambda seconds: (setattr(system, 'clock', system.clock+seconds), (tmp_path/'chain.log').write_text('y\nstart train_B0\n'))
    assert RUN.queue(options(root), system) == 0
    assert 'UNV2 chain waits' in (root/'queue.log').read_text() and system.launched[0] == RUN.K1_UNIT


def test_the_run_that_has_started_finishes_and_no_new_one_starts_once_the_last_member_started(root, tmp_path, monkeypatch):
    chain_files(tmp_path, monkeypatch, log='z\n')
    plan = mc.plan_runs()

    def behaviour(cell):
        if cell == plan[3]:                                # B1 starts while this run is on the GPU
            (tmp_path/'chain_state/train_B1.started').write_text('x')
        return 0, dict(GOOD), 10.
    system = FakeSystem(behaviour)
    assert RUN.queue(options(root), system) == 8
    assert system.launched[1:] == [c.unit for c in plan[:4]]
    assert json.loads((core.run_dir(root, plan[3])/'queue_done.json').read_text())['state'] == 'done'
    assert 'train_B1.started' in json.loads((root/'queue_stop.json').read_text())['reason'] and system.clock >= 30*60
    assert not (root/'analysis').exists() and not core.run_dir(root, plan[4]).exists()


def test_past_the_deadline_the_queue_starts_no_gpu_unit(root, monkeypatch):
    monkeypatch.setattr(RUN, 'wall_clock', lambda: datetime(2026, 11, 4, 14, 15, tzinfo=timezone.utc))
    system = FakeSystem()
    assert RUN.queue(options(root), system) == 8
    assert system.launched == [] and 'past 2026-11-04T08:15:00-06:00' in json.loads((root/'queue_stop.json').read_text())['reason']


def test_a_failed_evaluation_still_writes_the_reading_and_returns_8(root):
    system = FakeSystem()
    system.evaluate = lambda argv: 8
    assert RUN.queue(options(root), system) == 8
    assert (root/'analysis/decision_mc0p.json').is_file() and json.loads((root/'queue_end.json').read_text())['eval_rc'] == 8


# ---- the dry run ---------------------------------------------------------------------------------------------------------------------------
def dry_world(tmp_path, monkeypatch):
    """Five pairs of stand-ins with no label member anywhere: a reader of labels fails on them."""
    subjects = [f'OASIS_OAS1_{n:04d}_MR1' for n in range(438, 444)]
    stage = tmp_path/'stage_L2R_val_v1'
    stage.mkdir()
    write_json(stage/'selection.json', dict(pairs=[dict(pair=p) for p in core.PAIRS]+[dict(pair='later')]))
    a5p = tmp_path/'A5P'
    grid = (6, 7, 8)
    for pair in core.PAIRS:
        fixed, moving = pair.split('__')
        for directory in (stage/pair, a5p/pair):
            directory.mkdir(parents=True)
            np.savez(directory/'pair.npz', fixed=np.zeros(grid), moving=np.zeros(grid))
        write_json(stage/pair/'manifest.json', dict(shape_zyx=list(grid), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm', pair_sha256=file_sha256(stage/pair/'pair.npz'),
                                                     fixed_subject=fixed, moving_subject=moving, sources=dict(fixed=dict(sha256='a'*64), moving=dict(sha256='b'*64))))
    for subject in subjects:
        p = tmp_path/'pred'/subject
        p.mkdir(parents=True)
        np.savez(p/'probs36.npz', p=np.zeros((36,)+grid, dtype=np.float16))
        (p/'structure_groups.npz').write_bytes(b'x')
        write_json(p/'provenance.json', dict(segmenter_id='unet35_v1_ens', policy='all_folds_mean', test_labels_seen=False))
        (tmp_path/'atropos'/subject).mkdir(parents=True)
        write_json(tmp_path/'atropos'/subject/'provenance.json', {})
    mc0 = tmp_path/'mc0_v1'
    for arm, tier in (('P0', 'long'), ('P0', 'fine'), ('P1', 'long'), ('P2', 'long'), ('P2', 'fine')):
        for pair in core.PAIRS:
            d = core.run_dir(mc0, core.Cell(arm, tier, pair))
            d.mkdir(parents=True)
            write_json(d/'metrics.json', {})
    mi = core.make_inputs()
    return dict(stage=stage, a5p=a5p, seg_root=tmp_path/'atropos', mc0_root=mc0, unet=mi.UNetSource(pred_root=tmp_path/'pred'))


def test_dry_run_prints_all_45_and_reads_no_label(tmp_path, monkeypatch):
    world = dry_world(tmp_path, monkeypatch)
    lines = []
    assert RUN.dry_run(out=lines.append, **world) == 0
    text = '\n'.join(lines)
    for cell in mc.plan_runs():
        assert cell.run_id in text
    assert 'problems: none' in text and 'selection.json first 5' in text and 'fixed (6, 7, 8)' in text and 'all_folds_mean' in text
    assert 'seg35' not in text and 'seg4' not in text and 'py'+'test' not in text.replace(str(tmp_path), '')


@pytest.mark.parametrize('what', ('a5p', 'prediction', 'atropos', 'mc0', 'policy'))
def test_dry_run_names_what_is_missing(tmp_path, monkeypatch, what):
    world = dry_world(tmp_path, monkeypatch)
    if what == 'a5p':
        (world['a5p']/core.PAIRS[2]/'pair.npz').unlink()
    elif what == 'prediction':
        (tmp_path/'pred'/DEV[0]/'probs36.npz').unlink()
    elif what == 'atropos':
        (world['seg_root']/DEV[1]/'provenance.json').unlink()
    elif what == 'mc0':
        (core.run_dir(world['mc0_root'], core.Cell('P2', 'fine', core.PAIRS[0]))/'metrics.json').unlink()
    else:
        write_json(tmp_path/'pred'/DEV[0]/'provenance.json', dict(segmenter_id='unet35_v1_ens', policy='cross_fit_other_folds', test_labels_seen=False))
    lines = []
    assert RUN.dry_run(out=lines.append, **world) == 8
    assert 'problems: none' not in lines[-1]


# ---- one synthetic run per arm through everything --------------------------------------------------------------------------------------------
PIPELINE = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
import mc0_core as core
core.frozen_api()
import mc0p_core as mc
import run_mc0p as run
world = json.loads(sys.argv[2])
seen, real_load = set(), np.load
class Spy:                                              # the whole of run-one under the spy: a member with seg in its name raises, and the run fails
    def __init__(self, path, inner):
        self.name, self.inner = Path(path).name, inner
    def __enter__(self):
        return self
    def __exit__(self, *exc):
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
root = Path(world['root'])
core.STAGE = Path(world['stage'])
core.SETUP_WORKERS = 1
core.TIERS.update(long=dict(iterations=(2, 2, 2, 4)), fine=dict(iterations=(2, 2, 2, 4), sigma_field=.5))
mc.SEG_ROOT, mc.A5P_INPUTS, mc.MC0_ROOT = Path(world['seg_root']), Path(world['a5p']), Path(world['mc0_root'])
mi = core.make_inputs()
mc.default_unet = lambda: mi.UNetSource(pred_root=Path(world['pred']))
pair = core.PAIRS[0]
np.load = lambda path, **kwargs: Spy(path, real_load(path, **kwargs))
for arm, tier in (('Q0', 'long'), ('Q0', 'fine'), ('Q1', 'long'), ('Q2', 'long'), ('Q2s', 'long'), ('Q3x3', 'long'), ('Q3h', 'long')):
    cell = mc.Cell(arm, tier, pair)
    assert run.main(['run-one', '--arm', arm, '--tier', tier, '--pair', pair, '--out', str(core.run_dir(root, cell)), '--device', 'cpu', '--code-sha', 'x']) == 0
np.load = real_load
(root/'seen_by_runs.json').write_text(json.dumps(sorted(seen)))
assert run.main(['evaluate', '--root', str(root)]) == 0
assert run.main(['decide', '--root', str(root)]) == 0
"""


def pipeline_world(tmp_path, monkeypatch):
    """A stage of the first planned pair on a grid large enough for four levels; predictions under the repository's registry, Atropos, A5P's inputs written by make_inputs itself."""
    grid = (26, 28, 30)
    monkeypatch.setattr(a4, 'SHAPE', grid)                 # write_prediction's grid record
    pair = core.PAIRS[0]
    fixed, moving = pair.split('__')
    n = int(np.prod(grid))
    block = (np.arange(n) // 6 % 36).reshape(grid)
    maps = {fixed: block, moving: np.roll(block, 1, axis=0)}
    stage = tmp_path/'stage_L2R_val_v1'
    (stage/pair).mkdir(parents=True)
    rng = np.random.default_rng(3)
    truth = {s: np.roll(m, 1, axis=2).astype(np.int16) for s, m in maps.items()}          # the truth differs from the prediction
    arrays = dict(fixed=gaussian_filter(rng.uniform(.1, .7, grid), 1.), fixed_seg4=(truth[fixed] % 5).astype(np.int16), fixed_seg35=truth[fixed], moving=gaussian_filter(rng.uniform(.1, .7, grid), 1.),
                  moving_seg4=(truth[moving] % 5).astype(np.int16), moving_seg35=truth[moving], initial_u=np.zeros(grid+(3,)))
    np.savez_compressed(stage/pair/'pair.npz', **{k: arrays[k] for k in MI.KEY_ORDER})
    write_json(stage/pair/'manifest.json', dict(pair=pair, fixed_subject=fixed, moving_subject=moving, shape_zyx=list(grid), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm',
                                                sources=dict(fixed=dict(sha256=source_sha(fixed)), moving=dict(sha256=source_sha(moving))), pair_sha256=file_sha256(stage/pair/'pair.npz'), baseline_native_mse=.01, preprocessing='synthetic'))
    write_json(stage/'selection.json', dict(pairs=[dict(pair=pair)]))
    write_json(stage/'completion.json', dict(done=True))
    models = pair_module.load_segmenter_registry(structure_layer.REPO_REGISTRY).models
    for subject, L in maps.items():
        noise = np.random.default_rng(int(subject.split('_')[2])).uniform(0, 1, (36,)+grid)
        probs = .6*(np.arange(36)[:, None, None, None] == L[None])+.4*noise/noise.sum(axis=0)
        a4.write_prediction(tmp_path/'pred', subject, models, probs=(probs/probs.sum(axis=0)).astype(np.float32))
        wm = np.where(np.isin(L, MI.CEREBRAL_WM), .9, .05)
        csf = .3*(1-wm)
        a4.write_atropos(tmp_path/'atropos', subject, posteriors=np.stack([csf, 1-wm-csf, wm]))
    unet = MI.UNetSource(pred_root=tmp_path/'pred', models_root=tmp_path/'models', image_root=tmp_path/'images')
    MI.build_arm('A5P', tmp_path/'inputs', tmp_path/'atropos', stage=stage, allowlist=core.ALLOWLIST, unet=unet)
    return dict(root=str(tmp_path/'mc0p_v1'), stage=str(stage), seg_root=str(tmp_path/'atropos'), pred=str(tmp_path/'pred'), a5p=str(tmp_path/'inputs/A5P'/MI.STAGE_NAME), mc0_root=str(tmp_path/'no_mc0'))


@needs_frozen
def test_every_arm_through_everything(tmp_path, monkeypatch):
    world = pipeline_world(tmp_path, monkeypatch)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{FROZEN}:{ROOT}')
    done = subprocess.run([PY, '-B', '-c', PIPELINE, str(ROOT/'experiments/mc0'), json.dumps(world)], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout+done.stderr
    root = Path(world['root'])
    for arm, tier, k in (('Q0', 'long', 1), ('Q0', 'fine', 1), ('Q1', 'long', 14), ('Q2', 'long', 35), ('Q2s', 'long', 35), ('Q3x3', 'long', 35), ('Q3h', 'long', 35)):
        d = core.run_dir(root, mc.Cell(arm, tier, core.PAIRS[0]))
        run, metrics = json.loads((d/'run.json').read_text()), json.loads((d/'metrics.json').read_text())
        assert run['status'] == 'done' and run['k'] == k == run['optimizer']['channels'] and run['oracle'] is False and metrics['oracle'] is False and metrics['schema'] == 'bayesreg.mc0p.metrics.v1'
        # what SEED's entry check reads (cf7887f, experiments/seed/run_seed.py): arm, tier, k, oracle, run_id, pair, data, field, tapered_u_sha256
        assert (run['arm'], run['tier'], run['pair']) == (arm, tier, core.PAIRS[0]) and run['data']['pair_sha256'] == file_sha256(Path(world['stage'])/core.PAIRS[0]/'pair.npz')
        assert run['data']['manifest_sha256'] == file_sha256(Path(world['stage'])/core.PAIRS[0]/'manifest.json') and (run['data']['fixed_subject'], run['data']['moving_subject']) == tuple(core.PAIRS[0].split('__'))
        assert run['field']['file'] == 'field.npz' and len(run['field']['u_sha256']) == 64 and len(metrics['tapered_u_sha256']) == 64 and run['code_sha'] == 'x' and run['seconds_total'] > 0
        assert mc.oracle_of_run(run) is False
        assert run['run_id'] == f"mc0p_{arm}_{tier}_0438-0439" and run['optimizer']['role'] == mc.ROLE and 'oracle_probe' not in run['optimizer']['role']
        assert run['scale']['images_equal_a5p_inputs'] is True and run['field']['sha256'] == core.sha_file(d/'field.npz')       # Q0's images are what make_inputs wrote for A5P, in every arm
        sources = run['channel_sources']
        assert sources['oracle'] is False and sources['truth_labels_read'] is False and sources['pair_keys_read'] == ['fixed', 'moving', 'initial_u'] and set(sources['sides']) == {'fixed', 'moving'}
        assert sources['sides']['fixed']['segmenter_id'] == 'unet35_v1_ens' and sources['sides']['moving']['policy'] == 'all_folds_mean'
        assert (run['scale']['D_multiplier'] == 1.) == (arm == 'Q0')
        norms = run['level_norms']
        assert 'P0' not in norms and len(norms['Q0']) == 4
        assert norms['ratio_raw'][3] == pytest.approx(3. if arm == 'Q3x3' else 1., abs=1e-3)
        first = metrics['certificate']['attempts'][0]
        assert first['alpha'] == 1. and first['config']['tile_size'] == 65536 and first['config']['fallback_workers'] == 4 and set(decision.MEASURES) <= set(metrics['metrics'])
        with np.load(d/'field.npz') as z:
            assert z['u'].shape == (26, 28, 30, 3) and np.abs(z['u']).max() > 0
    result = json.loads((root/'analysis/decision_mc0p.json').read_text())
    assert result['reading'] == decision.READ_UNDECIDABLE and result['oracle'] is False and result['inputs_check']['q0_images_equal_a5p_inputs'] == ['True']
    with open(root/'analysis/long.csv') as f:
        assert {r['oracle'] for r in csv.DictReader(f)} == {'false'}
    assert json.loads((root/'seen_by_runs.json').read_text()) == ['pair.npz:fixed', 'pair.npz:initial_u', 'pair.npz:moving', 'probs36.npz:p', 'structure_groups.npz:p']     # every run of every arm


def test_a_run_refuses_to_start_twice(tmp_path):
    (tmp_path/'started.json').write_text('{}')
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import mc0_core as core; core.frozen_api(); import run_mc0p as run; "
            "sys.exit(run.main(['run-one', '--arm', 'Q0', '--tier', 'long', '--pair', core.PAIRS[0], '--out', sys.argv[2], '--device', 'cpu']))")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{FROZEN}:{ROOT}')
    done = subprocess.run([PY, '-B', '-c', code, str(ROOT/'experiments/mc0'), str(tmp_path)], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode != 0 and 'never rerun' in done.stderr
