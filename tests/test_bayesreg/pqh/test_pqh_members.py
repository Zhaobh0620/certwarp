"""PQH, the members unit (contract C22.H5; C22.H8 冒烟 (i) in its dev mode) on the PQ package's synthetic world (CPU, tmp_path).  The inference itself is pq_members.mean_and_variance, tested
by the PQ package (tests/test_bayesreg/pq/test_pq_members.py); here it is replaced by the registered prediction and the development V, so that the unit's own steps are what is tested:
the order before the lock, Hm5 with pq_members.subject_record, the records and the files written once, the dev reference of segment (i) (V bit for bit after the reference file is bound to its record),
a Hm5 failure stopping the unit with pq_refused.json, and the once rule."""
from pathlib import Path

import numpy as np
import pytest
import torch

from . import _world as W

hp, hin, hu, io_, PW = W.hp, W.hin, W.hu, W.io_, W.PW
SUBJECTS = W.PAIR.split('__')


@pytest.fixture
def dev(tmp_path, monkeypatch):
    w, layout, entry = W.dev_world(tmp_path, monkeypatch)
    import pq_members
    from bayesreg.io.guard import load_dev_allowlist
    calls = []

    def mean_and_variance(subject, *, files, models, plan, device):
        calls.append(subject)
        with np.load(w.ctx.pred_root/subject/'probs36.npz') as z:
            mean = np.array(z['p']).astype(np.float32)
        with np.load(w.ctx.pq_root/'members'/subject/io_.VAR_FILE) as z:
            V = np.array(z['V'])
        return mean, V, dict(sha256=PW.source_sha(subject)), dict(percentile=99.5), [dict(member_sha256='m')]
    monkeypatch.setattr(pq_members, 'mean_and_variance', mean_and_variance)
    monkeypatch.setattr(pq_members, 'member_files', lambda root, members=None: {'m': Path(root)/'m'/'member.pt'})
    monkeypatch.setattr(pq_members, 'plan_for', lambda registry, subject, allowlist, available: ([('model', 'm', 'path')], {'model': {}}, 'all_folds_mean'))
    import hi_core
    monkeypatch.setattr(hi_core, 'check_registry', lambda registry, anchor: dict(path=str(registry.path), sha256=registry.sha256))
    allowlist = load_dev_allowlist(w.repo/W.pins.ALLOWLIST_REL)
    deps = W.deps(allowlist=lambda: (allowlist, dict(allowlist_sha256=allowlist.sha256)), files=lambda: object(), device=lambda device, free: torch.device('cpu'),
                  environment=lambda device, decision: dict(device=str(device), note='test'))
    return W.namespace(w=w, layout=layout, deps=deps, calls=calls)


def run(dev, **kwargs):
    W.XW.sealed_module._reset_cache()
    return hu.members_unit(dev.layout, device='cpu', deps=dev.deps, **kwargs)


def test_the_members_unit_in_dev_mode(dev):
    small, big = hin.run_dirs(dev.layout, 'members', 'first_run')
    import shutil
    shutil.rmtree(small)
    shutil.rmtree(big)
    assert run(dev) == 0
    record = io_.read_json(small/hin.MEMBERS_RECORD)
    assert record['m5_passed'] is True and record['mode'] == 'dev' and sorted(record['subjects']) == sorted(SUBJECTS) and record['settings_sha256'] == hp.settings_sha256()
    assert record['dev_reference']['equal'] is True and all(r['reference_bound'] for r in record['dev_reference']['subjects'].values())
    for s in SUBJECTS:
        assert (big/s/io_.VAR_FILE).is_file() and (big/s/'member.json').is_file() and record['subjects'][s]['var_sha256'] == io_.sha_file(big/s/io_.VAR_FILE)
    opened = io_.read_json(small/'opened_files_members.json')
    assert opened['m6'] is True and not any(f['path'].endswith('aligned_seg35.nii.gz') for f in opened['files'])
    assert dev.calls == SUBJECTS
    rows, _ = hin.members_view(dev.layout, SUBJECTS)
    assert sorted(rows) == sorted(SUBJECTS)
    assert run(dev) == hu.EXIT['refused']                                       # once: the lock
    assert io_.read_json(W.refusals(dev.layout)[-1])['stage'] == 'once'


def test_a_hm5_failure_stops_the_unit(dev, monkeypatch):
    small, big = hin.run_dirs(dev.layout, 'members', 'first_run')
    import shutil
    shutil.rmtree(small)
    shutil.rmtree(big)
    import pq_members
    original = pq_members.mean_and_variance

    def shuffled(subject, **kwargs):
        mean, V, meta, norm, rows = original(subject, **kwargs)
        return mean[::-1].copy(), V, meta, norm, rows                         # another argmax almost everywhere
    monkeypatch.setattr(pq_members, 'mean_and_variance', shuffled)
    assert run(dev) == hu.EXIT['stopped']
    stop = io_.read_json(small/hu.STOPPED)
    assert stop['stage'] == 'hm5' and io_.read_json(small/hin.MEMBERS_RECORD)['m5_passed'] is False
    assert len(dev.calls) == 1                                                   # the first subject failed: the unit stopped there
    assert hu.previous_stop_records(dev.layout, 'members')
    with pytest.raises(W.core.PQRefused, match='Hm5 false|without a record'):
        hin.members_view(dev.layout, SUBJECTS)


def test_an_image_other_than_the_provenances_fails_hm5(dev, monkeypatch):
    small, big = hin.run_dirs(dev.layout, 'members', 'first_run')
    import shutil
    shutil.rmtree(small)
    shutil.rmtree(big)
    import pq_members
    original = pq_members.mean_and_variance
    monkeypatch.setattr(pq_members, 'mean_and_variance', lambda subject, **kw: (lambda r: (r[0], r[1], dict(sha256='0'*64), r[3], r[4]))(original(subject, **kw)))
    assert run(dev) == hu.EXIT['stopped']
    assert io_.read_json(small/hu.STOPPED)['stage'] == 'hm5'


def test_an_existing_members_directory_refuses(dev):
    assert run(dev) == hu.EXIT['refused']                                        # dev_world wrote a members run already
    assert io_.read_json(W.refusals(dev.layout)[-1])['stage'] == 'once'
    assert dev.calls == []


def test_the_dev_members_unit_needs_no_finalization_record(dev):
    """Smoke (i) runs before the review unit of segment (ii) writes its record (C22.H8): the dev members unit does not read it; the hold-out one does (the checkout's commit)."""
    small, big = hin.run_dirs(dev.layout, 'members', 'first_run')
    import shutil
    shutil.rmtree(small)
    shutil.rmtree(big)
    dev.layout.finalized.unlink()
    assert run(dev) == 0
    assert io_.read_json(small/hin.MEMBERS_RECORD)['checks']['finalized']['path'] is None
