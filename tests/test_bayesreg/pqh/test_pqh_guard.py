"""PQH's write guard in the test sandbox (review I7; the lead's ruling (8) of 11-09) and the pins that must equal the development package's (review m2).

pqh_inputs.guard_write refuses, before anything is written, a path under one of PQH's real roots (pqh_pins.REAL_WRITE_ROOTS: he_smoke_v3, pq_replay_v1, seg/pq_holdout_members.lock, holdout_B_v1,
D:'s hold-out directory, pq_v1) or holdout_common's, in either of its forms (a link in tmp_path that leads into a real root too); every write of the units, the smoke, the replay gate and the
launcher goes through it.  No real root is touched: the guard refuses before the write (the conftest's tripwire lists their files before and after every test)."""
import os
from pathlib import Path

import numpy as np
import pytest

from . import _world as W

hp, hin, hu, io_, hc = W.hp, W.hin, W.hu, W.io_, W.hc
import pq_core  # noqa: E402
import pq_stats  # noqa: E402
import run_pqh  # noqa: E402

REAL = [Path(p) for p in ('/home/rmcfz/seq_vp_tag_results/he_smoke_v3/pq/x.json', '/home/rmcfz/seq_vp_tag_results/pq_replay_v1/pq_replay_gate.lock',
                          '/home/rmcfz/seq_vp_tag_results/seg/pq_holdout_members.lock', '/home/rmcfz/seq_vp_tag_results/seg/pq_holdout_members.lock.rerun',
                          '/home/rmcfz/seq_vp_tag_results/seg/pq_holdout_members.lock.resume', '/home/rmcfz/seq_vp_tag_results/holdout_B_v1/pq/x.json',
                          '/mnt/d/seq_vp_tag_holdout_B_v1/pq/x/pq.npz', '/home/rmcfz/seq_vp_tag_results/pq_v1/runs/x.json', '/home/rmcfz/seq_vp_tag_results/xb_dev_v1/x.json')]


def test_the_real_roots_are_spelled_out_and_not_moved_by_the_tests():
    assert Path('/home/rmcfz/seq_vp_tag_results/seg/pq_holdout_members.lock') in hp.REAL_WRITE_ROOTS and Path('/home/rmcfz/seq_vp_tag_results/he_smoke_v3') in hp.REAL_WRITE_ROOTS
    assert Path(hp.MEMBERS_LOCK) not in hp.REAL_WRITE_ROOTS and Path(hp.SMOKE_ROOT) not in hp.REAL_WRITE_ROOTS        # the tests moved the pins, not the guard's roots


@pytest.mark.parametrize('path', REAL)
def test_the_guard_refuses_every_real_root_in_the_sandbox(path):
    assert os.environ.get(hc.TEST_SANDBOX_ENV)
    with pytest.raises(hc.RealRootInTest):
        hin.guard_write(path)


def test_the_guard_passes_tmp_path_and_does_nothing_outside_the_sandbox(tmp_path, monkeypatch):
    assert hin.guard_write(tmp_path/'x.json') == tmp_path/'x.json'
    monkeypatch.delenv(hc.TEST_SANDBOX_ENV)
    assert hin.guard_write(REAL[0]) == REAL[0]


def test_a_link_into_a_real_root_is_refused(tmp_path):
    link = tmp_path/'link'
    link.symlink_to('/home/rmcfz/seq_vp_tag_results/he_smoke_v3')                         # the target is only named, never made or opened
    with pytest.raises(hc.RealRootInTest):
        hin.guard_write(link/'pq'/'x.json')


@pytest.mark.parametrize('write', ['write_once', 'save_npz_once', 'write_json_once', 'make_dir', 'take_lock', 'keep_attempt', 'write_record', 'write_refusal', 'launch'])
def test_every_write_of_pqh_goes_through_the_guard(write, tmp_path, monkeypatch):
    target = Path('/home/rmcfz/seq_vp_tag_results/he_smoke_v3/pq/guard_test.json')
    calls = dict(write_once=lambda: hu.write_once(target, dict(a=1)), save_npz_once=lambda: hu.save_npz_once(target.with_suffix('.npz'), a=np.zeros(1)),
                 write_json_once=lambda: hu.write_json_once(target, dict(a=1)), make_dir=lambda: hu.make_dir(target.parent/'d'),
                 take_lock=lambda: hu.take_lock(dict(lock=Path('/home/rmcfz/seq_vp_tag_results/seg/pq_holdout_members.lock'), kind='first_run', authorization=None), dict(tool='pqh')),
                 keep_attempt=lambda: hu.keep_attempt(target.parent/'pair'), write_record=lambda: hu.write_record(target, dict(a=1)),
                 write_refusal=lambda: hu.write_refusal(W.namespace(batch_root=target.parent/'batches', pair_set='dev'), 'out', dict(stage='test'), W.quiet))
    if write == 'launch':
        monkeypatch.setattr(hp, 'REPLAY_ROOT', Path('/home/rmcfz/seq_vp_tag_results/pq_replay_v1'))
        calls['launch'] = lambda: run_pqh.launch('replay-gate', system=object(), log=W.quiet)
    with pytest.raises(hc.RealRootInTest):
        calls[write]()


def test_m2_the_pinned_baselines_are_the_development_packages():
    """pqh_pins.py's comments said "a test compares": here it is (review m2)."""
    assert hp.JUDGED_BASELINES == tuple(pq_stats.JUDGED_BASELINES) and hp.DESCRIBED_SCORES == tuple(pq_stats.DESCRIBED_SCORES)
    assert hp.BASELINE_KEYS == tuple(pq_core.BASELINE_KEYS)
    settings = hp.frozen_settings()
    assert settings['judged_baselines'] == list(pq_stats.JUDGED_BASELINES) and settings['baseline_keys'] == list(pq_core.BASELINE_KEYS)
