"""xb_drive.run_cell end to end on synthetic pairs with a fake method step: the order (code and preregistration, rerun authorization, admission, checks, lock, derived inputs, method, export check,
cleanup, move, terminal record), the statuses, P19, the refusal records and the committed authorizations (resume, rerun, no_resume), the dry run.  Prereg §2.3 (a)-(h); the lead's additions of
2026-11-07; the XB review's I1-I5, I7 and m3, m7, m8, m10 (revision 2, 2026-11-09)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from .. import _p0_pins as P0
from . import _world as W

hc, xc, xb_drive = W.hc, W.xc, W.xb_drive
QUIET = dict(max=0, oom=0, oom_kill=0)
OOM = dict(max=3, oom=1, oom_kill=1)


def pin_prereg(monkeypatch):
    """I5: a hold-out run needs PREREG_SHA256; the tests pin it to the file as it is (the code keeps None until BZ adopts revision 2)."""
    monkeypatch.setattr(xc, 'PREREG_SHA256', xc.sha256(xc.REPO/xc.PREREG_REL))


def refusal_docs(pair_set='dev'):
    return [json.loads(r.read_text()) for r in W.refusals(pair_set)]


def dev_world(monkeypatch, *, seg=True):
    W.fake_checks(monkeypatch)
    original = W.write_stage_pair(hc.DEV_STAGE, W.DEV_PAIR, seg=seg)
    return original


def result(arm, pair_set='dev', pair=W.DEV_PAIR, resume=False):
    paths = xc.cell_paths(arm, pair_set, pair)
    return json.loads((paths['resume_result'] if resume else paths['result']).read_text())


def test_a_development_run_writes_the_terminal_record(monkeypatch):
    original = dev_world(monkeypatch)
    method = W.FakeMethod()
    assert W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=W.deps(step=method)) == 0
    doc = result('fireants_greedy')
    assert doc['status'] == 'ok' and doc['export_check']['check']['corr'] > .999 and doc['export_check']['suspicious'] is False
    assert doc['disp']['sha256'] == xc.sha256(doc['disp']['file']) and doc['lock']['line']['arm'] == 'fireants_greedy' and doc['resume'] is None
    assert doc['spies']['driver']['passed'] and doc['spies']['driver']['original_opens_inside_reader'] >= 2      # the container hash and the reader
    assert doc['spy_passed'] and doc['derived_inputs']['keys'] == ['fixed', 'moving']
    assert set(doc['harness']['files']) >= {'xb_drive.py', 'xb_core.py', 'harness/m_fireants.py'} and doc['harness']['bl2_protocol']['sha256'] == xc.BL2_PROTOCOL_SHA256
    assert doc['unit']['expected'] == list(xc.UNIT_PROPERTIES) and 'sanity' not in doc and 'attempts' not in doc and 'attempt' not in doc    # (b), (c)
    # (a): the method process never sees the original's path; its argv holds the derived npz under spyrun
    call = method.calls[0]
    assert str(original) not in ' '.join(call['cmd']) and str(original) not in json.dumps(call['env']) and str(os.path.realpath(original)) not in ' '.join(call['cmd'])
    assert call['cmd'][3].endswith('spy/xb_spyrun.py') and call['cmd'][6].endswith('harness/m_fireants.py') and call['cmd'][7:9] == ['greedy', str(xc.cell_paths('fireants_greedy', 'dev', W.DEV_PAIR)['work']/'inputs.npz')]
    assert 'BL2_ATTEMPT' not in call['env']
    # the driver deletes nothing: the intermediates are listed with their sha256 for the reviewed deletion step (the lead's ruling); the kept files stay (a development run keeps them on /home)
    work = Path(doc['store']['work'])
    assert (work/'fixed.nii.gz').exists() and (work/'inputs.npz').exists() and (work/'disp.npz').exists() and doc['cleanup']['deleted'] is False
    lines = Path(doc['cleanup']['manifest']).read_text().splitlines()
    assert lines[0] == 'path\tsize_bytes\tsha256' and {Path(l.split('\t')[0]).name for l in lines[1:]} == {'fixed.nii.gz', 'inputs.npz'}
    assert all(xc.sha256(l.split('\t')[0]) == l.split('\t')[2] for l in lines[1:]) and doc['cleanup']['manifest_sha256'] == xc.sha256(doc['cleanup']['manifest'])
    # C23.6-5: the sealed lists were read under the spy, every read recorded and allowed
    driver = doc['spies']['driver']
    assert driver['n_sealed_list_reads'] >= 8 and all(r['cls'] == 'allowed' for r in driver['sealed_list_reads'])


def test_a_second_run_is_refused_and_reads_nothing(monkeypatch):
    dev_world(monkeypatch)
    assert W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=W.deps()) == 0
    method = W.FakeMethod()
    assert W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=W.deps(step=method)) == xc.EXIT['refused']
    assert method.calls == []


def test_derived_inputs_are_the_same_from_a_pair_without_label_members(monkeypatch, tmp_path):
    """m6: the hold-out stage's pair.npz has no seg member; the derived npz is the same bytes, and nothing asks for a label key (sanity() is gone)."""
    W.fake_checks(monkeypatch)
    with_seg = W.write_stage_pair(tmp_path/'a', W.DEV_PAIR, seg=True)
    without = W.write_stage_pair(tmp_path/'b', W.DEV_PAIR, seg=False)
    za, ra = xc.derive_inputs(with_seg, tmp_path/'a.npz', shape=W.SHAPE)
    zb, rb = xc.derive_inputs(without, tmp_path/'b.npz', shape=W.SHAPE)
    for k in ('fixed', 'moving'):
        assert za[k].tobytes() == zb[k].tobytes() and za[k].dtype == zb[k].dtype
    assert ra['members'] == rb['members']
    W.write_stage_pair(hc.DEV_STAGE, W.DEV_PAIR, seg=False)
    assert W.run_cell('fireants_syn', 'dev', W.DEV_PAIR, deps=W.deps()) == 0
    assert result('fireants_syn')['status'] == 'ok'


def test_the_reader_asks_for_fixed_and_moving_only(monkeypatch, tmp_path):
    from bayesreg.io import pair as pair_module
    original = W.write_stage_pair(tmp_path, W.DEV_PAIR)
    asked = []
    real = pair_module._open_npz

    class Guard:
        def __init__(self, path):
            self.inner = real(path)

        def __getitem__(self, key):
            asked.append(key)
            if key not in ('fixed', 'moving'):
                raise AssertionError(f'asked for {key}')
            return self.inner[key]

        def __getattr__(self, name):
            raise AssertionError(f'attribute {name} of the original archive')

        def __iter__(self):
            raise AssertionError('iteration over the original archive')

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.inner.close()
    monkeypatch.setattr(pair_module, '_open_npz', Guard)
    xc.read_image_members(original)
    assert asked == ['fixed', 'moving']


def test_holdout_without_sg_is_refused_at_admission_with_a_refusal_record_and_nothing_in_the_cell(monkeypatch):
    """I1: the refusal is a record in the batch directory (here <record root>/batches/direct: no chain), never the cell's result.json; the cell keeps no lock, no record directory."""
    W.fake_checks(monkeypatch)
    pin_prereg(monkeypatch)
    row = W.holdout_rows()[0]
    pair = hc.pair_name(*row)
    W.write_stage_pair(hc.STAGE_ROOT, pair, seg=False)
    method = W.FakeMethod()
    assert W.run_cell('ants_s', 'holdout', pair, deps=W.deps(step=method)) == xc.EXIT['admission_rejected']
    paths = xc.cell_paths('ants_s', 'holdout', pair)
    assert not paths['result'].exists() and not paths['record'].exists() and not paths['lock'].exists() and method.calls == []
    [path] = W.refusals('holdout')
    doc = json.loads(path.read_text())
    assert path.parent.parent == hc.batch_root(hc.HOLDOUT_ROOT)/'direct' and doc['schema'] == 'bayesreg.holdout.refusal.v1'
    assert doc['stage'] == 'unseal' and doc['refusal_class'] == 'refusal' and doc['planned'] is True and doc['exit'] == xc.EXIT['admission_rejected']
    assert {'code', 'preregistration', 'rerun_authorization'} <= set(doc['checks_run']) and 'admission' not in doc['checks_run']        # the code tree before the admission
    assert doc['spy']['original_opens_inside_reader'] == 0 and doc['spy']['breaches'] == []


def test_the_dry_run_on_a_holdout_pair_is_refused_at_admission_and_writes_only_its_out(monkeypatch, tmp_path):
    W.fake_checks(monkeypatch)
    monkeypatch.setattr(xc, 'PREREG_SHA256', None)                  # P0 pinned it (runs/20261114-p0/P0_pins.md); the unpinned state, set back here, as before
    pair = hc.pair_name(*W.holdout_rows()[0])
    out = tmp_path/'dry'
    out.mkdir()
    assert xb_drive.main(['dry-run', '--arm', 'fireants_greedy', '--pair-set', 'holdout', '--pair', pair, '--out', str(out)]) == xc.EXIT['admission_rejected']
    doc = json.loads((out/'dry_run.json').read_text())
    assert doc['refused']['stage'] == 'unseal' and doc['spy']['breaches'] == []
    assert 'preregistration' in [p['stage'] for p in doc['problems_by_stage']]          # I5: PREREG_SHA256 is None in the code: a dry run reports it and goes on
    assert sorted(e['kind'] for e in doc['expected']) == ['no_unseal_record', 'rule_one_unpinned'] and 'preregistration' in ' '.join(e['problem'] for e in doc['unexpected'])   # A5a review m8
    assert doc['passed'] is False and len(doc['problems']) == len(doc['expected'])+len(doc['unexpected'])          # the real tree: no pinned_code.json, no unseal record
    assert W.refusals('holdout') == []                                          # a dry run writes no refusal record
    assert not hc.RESULTS.exists() or not any(hc.RESULTS.rglob('*.json'))
    assert xb_drive.dry_out_problem(xc.REPO/'runs') is not None


def test_the_dry_run_on_a_holdout_pair_at_p0_has_only_the_two_expected_refusals(monkeypatch, tmp_path):
    """P0 (runs/20261114-p0/d1_expected.md section 4): with PREREG_SHA256 as committed the dry run of a hold-out pair is refused at the unseal record; its problems are
    exactly rule_one_unpinned and no_unseal_record (p0_ok), the preregistration check has no problem; only dry_run.json is written.  The tree's cleanliness is stood in
    as clean (a P0 checkout is; this test is about the preregistration's pin, and a work tree being edited would add a `code` problem); rule one is the real one."""
    P0.assert_pinned(xc.PREREG_SHA256, xc.PREREG_REL)
    W.fake_checks(monkeypatch)
    monkeypatch.setattr(hc, 'code_identity', lambda run=None, repo=None: dict(commit='c'*40, clean=True, status='', untracked=[]))
    pair = hc.pair_name(*W.holdout_rows()[0])
    out = tmp_path/'dry'
    out.mkdir()
    assert xb_drive.main(['dry-run', '--arm', 'fireants_greedy', '--pair-set', 'holdout', '--pair', pair, '--out', str(out)]) == xc.EXIT['admission_rejected']
    doc = json.loads((out/'dry_run.json').read_text())
    assert doc['refused']['stage'] == 'unseal' and doc['spy']['breaches'] == []
    assert 'preregistration' not in [p['stage'] for p in doc['problems_by_stage']] and doc['checks']['preregistration']['problem'] is None
    assert sorted(e['kind'] for e in doc['expected']) == ['no_unseal_record', 'rule_one_unpinned'] and doc['unexpected'] == [] and doc['p0_ok'] is True
    assert doc['passed'] is False and len(doc['problems']) == len(doc['expected'])
    assert W.refusals('holdout') == []
    assert not hc.RESULTS.exists() or not any(hc.RESULTS.rglob('*.json'))


def holdout_world(tmp_path, monkeypatch, row=0):
    W.fake_checks(monkeypatch)
    pin_prereg(monkeypatch)
    fixed, moving = W.holdout_rows()[row]
    pair = hc.pair_name(fixed, moving)
    original = W.write_stage_pair(hc.STAGE_ROOT, pair, seg=False)
    sim = W.simulated_unseal(tmp_path, monkeypatch)
    return pair, original, sim


def test_a_holdout_run_through_the_real_loader_moves_the_kept_files_to_d(tmp_path, monkeypatch):
    pair, original, sim = holdout_world(tmp_path, monkeypatch)
    assert W.run_cell('fireants_greedy', 'holdout', pair, deps=W.deps()) == 0
    doc = result('fireants_greedy', 'holdout', pair)
    store = xc.cell_paths('fireants_greedy', 'holdout', pair)['store']
    assert doc['status'] == 'ok' and doc['disp']['file'] == str(store/'disp.npz') and (store/'disp.npz').exists()
    assert doc['move']['passed'] and not (Path(doc['store']['work'])/'disp.npz').exists()
    assert doc['checks']['admission']['pairs_tsv']['row'] == 1 and doc['lock']['line']['unseal_sha256'] == doc['checks']['admission']['unseal']['sha256']
    assert doc['checks']['pair_input']['guard']['unsealed'] is True and doc['checks']['pair_input']['guard']['sealed_guard']['unseal']['sha256'] == doc['checks']['admission']['unseal']['sha256']
    assert doc['spies']['driver']['passed'] and doc['spies']['driver']['n_sealed_list_reads'] >= 8


@pytest.mark.parametrize('git', [dict(status=' M experiments/xb/xb_core.py'), dict(untracked='experiments/hi/new_module.py')])
def test_a_holdout_run_on_a_dirty_tree_is_refused_before_the_admission(tmp_path, monkeypatch, git):
    """I1: the code tree is checked before the admission (a dirty tree never reaches the unseal record); m9: an untracked module under experiments/ (hi, he) counts."""
    pair, original, sim = holdout_world(tmp_path, monkeypatch)
    d = W.deps()
    d.code_run = W.fake_git(**git)
    assert W.run_cell('fireants_greedy', 'holdout', pair, deps=d) == xc.EXIT['refused']
    assert not xc.cell_paths('fireants_greedy', 'holdout', pair)['lock'].exists()
    [doc] = refusal_docs('holdout')
    assert doc['stage'] == 'code' and doc['checks_run'] == ['code'] and doc['spy']['n_sealed_list_reads'] == 0


def test_the_preregistration_must_be_pinned_for_a_holdout_run(tmp_path, monkeypatch):
    """I5: PREREG_SHA256 None (the code as committed) or another sha256 refuses a hold-out run before the admission; a dry run reports it as a problem and checks the rest."""
    pair, original, sim = holdout_world(tmp_path, monkeypatch)
    monkeypatch.setattr(xc, 'PREREG_SHA256', None)
    out = tmp_path/'dry'
    out.mkdir()
    assert W.run_cell('fireants_greedy', 'holdout', pair, dry_run_out=out, deps=W.deps()) == xc.EXIT['refused']
    dry = json.loads((out/'dry_run.json').read_text())
    assert [p['stage'] for p in dry['problems_by_stage']] == ['preregistration'] and dry['checks']['admission']['pairs_tsv']['row'] == 1 and dry['would_lock']
    assert W.refusals('holdout') == []
    method = W.FakeMethod()
    assert W.run_cell('fireants_greedy', 'holdout', pair, deps=W.deps(step=method)) == xc.EXIT['refused'] and method.calls == []
    monkeypatch.setattr(xc, 'PREREG_SHA256', '0'*64)
    assert xc.prereg_problem() and 'is not the pinned' in xc.prereg_problem()
    assert [d['stage'] for d in refusal_docs('holdout')] == ['preregistration']
    assert not xc.cell_paths('fireants_greedy', 'holdout', pair)['lock'].exists()


def test_a_development_run_records_the_preregistration_without_refusing(monkeypatch):
    dev_world(monkeypatch)
    monkeypatch.setattr(xc, 'PREREG_SHA256', None)
    assert W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=W.deps()) == 0
    doc = result('fireants_greedy')
    assert doc['checks']['preregistration']['problem'] and doc['lock']['line']['preregistration_sha256'] == xc.sha256(xc.REPO/xc.PREREG_REL)


def test_a_cell_refused_at_admission_runs_again_only_with_a_committed_rerun_authorization(tmp_path, monkeypatch):
    """I1 (C23.12-6, D107): refused (a refusal record, nothing in the cell) -> run again without an authorization: refused (stage rerun), nothing read -> an uncommitted authorization: refused ->
    the committed authorization that lists the original refusal record and its sha256: the run goes through, its lock line names the authorization; every refusal record stays."""
    W.fake_checks(monkeypatch)
    pin_prereg(monkeypatch)
    pair = hc.pair_name(*W.holdout_rows()[0])
    W.write_stage_pair(hc.STAGE_ROOT, pair, seg=False)
    assert W.run_cell('fireants_greedy', 'holdout', pair, deps=W.deps()) == xc.EXIT['admission_rejected']           # no unseal record yet
    [first] = W.refusals('holdout')
    W.simulated_unseal(tmp_path, monkeypatch)
    method = W.FakeMethod()
    assert W.run_cell('fireants_greedy', 'holdout', pair, deps=W.deps(step=method)) == xc.EXIT['refused'] and method.calls == []
    assert [d['stage'] for d in refusal_docs('holdout')].count('rerun') == 1
    path = W.authorize('rerun', 'fireants_greedy', 'holdout', pair, commit=False)
    assert path.parent == hc.AUTH_ROOT/'splits'/'bayesreg_v1' and path.name == 'holdout_B_v1_xb_rerun_1.json'
    assert json.loads(path.read_text())['refusals'] == [dict(path=str(first), sha256=xc.sha256(first))]        # the stage-rerun refusals are not listed
    assert W.run_cell('fireants_greedy', 'holdout', pair, deps=W.deps(step=method)) == xc.EXIT['refused'] and method.calls == []      # not committed: refused
    W.git(W.auth_repo(), 'add', '-A')
    W.git(W.auth_repo(), 'commit', '-q', '-m', 'the rerun authorization')
    assert W.run_cell('fireants_greedy', 'holdout', pair, deps=W.deps(step=method)) == 0 and len(method.calls) == 1
    doc = result('fireants_greedy', 'holdout', pair)
    assert doc['status'] == 'ok' and doc['lock']['line']['rerun_authorization'] == dict(path=str(path), sha256=xc.sha256(path))
    assert len(W.refusals('holdout')) == 3 and first.exists()


def test_a_rerun_authorization_covers_one_set_of_refusals(monkeypatch):
    """I1: a further refusal (other than a refusal for lacking the authorization) makes the authorization stale; it never covers a set it did not list."""
    dev_world(monkeypatch)
    paths = xc.cell_paths('fireants_greedy', 'dev', W.DEV_PAIR)
    d = W.deps()
    d.gate = lambda: (_ for _ in ()).throw(hc.InfrastructureFailure('precheck_timeout', 'busy'))
    assert W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=d) == xc.EXIT['infra_failure']
    W.authorize('rerun', 'fireants_greedy', 'dev', W.DEV_PAIR)
    assert W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=d) == xc.EXIT['infra_failure']                     # authorized, refused again (the machine): a second record
    assert W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=W.deps()) == xc.EXIT['refused']                     # the authorization lists one of the two
    assert [x['stage'] for x in refusal_docs()] == ['precheck_timeout', 'precheck_timeout', 'rerun'] and not paths['lock'].exists()
    path = W.authorize('rerun', 'fireants_greedy', 'dev', W.DEV_PAIR)
    assert path.name == 'dev_xb_rerun_2.json' and path.parent == hc.AUTH_ROOT/'runs'/'20261107-xb'/'authorizations'        # m15: a development authorization is named dev_
    assert W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=W.deps()) == 0


@pytest.mark.parametrize('method, status', [
    (W.FakeMethod(returncode=1), 'method_failure'),
    (W.FakeMethod(returncode=-9, events=(dict(max=0, oom=0, oom_kill=0), dict(max=3, oom=1, oom_kill=1))), 'infra_failure'),           # rule (乙): an exit -9 with the unit's OOM is infrastructure
    (W.FakeMethod(returncode=-9, events=(dict(max=0, oom=0, oom_kill=0), dict(max=0, oom=0, oom_kill=0))), 'infra_failure'),
    (W.FakeMethod(returncode=-9, events=(None, None)), 'infra_failure'),              # an exit -9 is OOM evidence even when memory.events are unread
    (W.FakeMethod(returncode=-9, memory_guard=True, events=(dict(max=0), dict(max=5, oom_kill=1))), 'infra_failure'),
    (W.FakeMethod(returncode=97), 'infra_failure'),
    (W.FakeMethod(returncode=0, events=(QUIET, OOM)), 'infra_failure'),                 # I2: an exit 0 with the unit's OOM during the step (an OOM-killed child the script did not report)
    (W.FakeMethod(returncode=1, events=(QUIET, dict(max=0, oom=0, oom_kill=1))), 'infra_failure'),     # I2: a non-zero exit with an oom_kill is not a method failure
])
def test_failure_classes(monkeypatch, method, status):
    dev_world(monkeypatch)
    rc = W.run_cell('unigradicon_io', 'dev', W.DEV_PAIR, deps=W.deps(step=method))
    doc = result('unigradicon_io')
    assert doc['status'] == status and rc == xc.EXIT[status]
    assert doc['disp'] is None and doc['lock'] is not None and (Path(doc['store']['work'])/'inputs.npz').exists()      # a failure keeps everything


def test_a_wrapper_error_and_its_policy(monkeypatch):
    dev_world(monkeypatch)

    def broken(*a, **k):
        raise ValueError('a bug in our conversion')
    monkeypatch.setitem(xc.ARM_FUNCTIONS, 'py', broken)
    assert W.run_cell('convexadam', 'dev', W.DEV_PAIR, deps=W.deps()) == xc.EXIT['wrapper_error']
    doc = result('convexadam')
    assert doc['status'] == 'wrapper_error' and doc['wrapper_error_policy'] == 'stop' and not xc.complete(doc)
    assert xc.complete(dict(doc, wrapper_error_policy='identity'))
    assert xc.WRAPPER_ERROR_POLICY in xc.WRAPPER_ERROR_POLICIES


def test_p19_stops_the_arm_at_the_third_suspicious_pair(monkeypatch):
    W.fake_checks(monkeypatch)
    pairs = hc.DEV_PAIRS[:4]
    for i, pair in enumerate(pairs):
        W.write_stage_pair(hc.DEV_STAGE, pair, seed=i)
    for pair in pairs[:3]:
        assert W.run_cell('voxelmorph', 'dev', pair, deps=W.deps(step=W.FakeMethod(own_scale=1.))) == 0
        assert result('voxelmorph', pair=pair)['export_check']['suspicious'] is True
    stop = xc.cell_paths('voxelmorph', 'dev', pairs[3])['arm_stop']
    assert stop.exists() and json.loads(stop.read_text())['suspicious_pairs'] == list(pairs[:3])
    method = W.FakeMethod()
    assert W.run_cell('voxelmorph', 'dev', pairs[3], deps=W.deps(step=method)) == xc.EXIT['refused'] and method.calls == []


def test_convexadam_v4seg_maps_from_the_predictions(monkeypatch):
    dev_world(monkeypatch)
    fixed, moving = W.DEV_PAIR.split('__')
    rng = np.random.default_rng(5)
    labels = {s: rng.integers(0, 36, W.SHAPE) for s in (fixed, moving)}
    for s in (fixed, moving):
        W.write_prediction(xc.HI_PRED_ROOTS['dev'], s, labels[s])
    run = W.method_run('dev', W.DEV_PAIR, labels)
    assert run == xc.HS_RUNS/'SHx24_first5'/'0438-0439'/'run.json'
    method = W.FakeMethod()
    assert W.run_cell('convexadam_v4seg', 'dev', W.DEV_PAIR, method_run=run, deps=W.deps(step=method)) == 0
    doc = result('convexadam_v4seg')
    assert doc['status'] == 'ok' and doc['maps']['sides']['fixed']['probs36_sha256'] and doc['maps']['method_labels_check']['equal']
    assert doc['lock']['line']['method_run'] == dict(path=str(run), sha256=xc.sha256(run)) and doc['checks']['method_run']['labels_sha256'] == doc['maps']['labels_sha256']
    with np.load(Path(doc['store']['work'])/'maps.npz') as z:
        assert np.array_equal(z['fixed_map'], labels[fixed]) and z['fixed_map'].dtype == np.int16
    cmd = method.calls[0]['cmd']
    assert cmd[6].endswith('m_convexadam_v4seg.py') and cmd[7].endswith('inputs.npz') and cmd[8].endswith('maps.npz')


def test_convexadam_v4seg_refuses_a_prediction_off_the_anchor_before_the_lock(monkeypatch):
    dev_world(monkeypatch)
    fixed, moving = W.DEV_PAIR.split('__')
    W.write_prediction(xc.HI_PRED_ROOTS['dev'], fixed, np.zeros(W.SHAPE, int))
    W.write_prediction(xc.HI_PRED_ROOTS['dev'], moving, np.zeros(W.SHAPE, int), override=dict(test_labels_seen=True))
    W.method_run('dev', W.DEV_PAIR, {fixed: np.zeros(W.SHAPE), moving: np.zeros(W.SHAPE)})
    method = W.FakeMethod()
    assert W.run_cell('convexadam_v4seg', 'dev', W.DEV_PAIR, method_run=xc.method_t88_run('dev', W.DEV_PAIR), deps=W.deps(step=method)) == xc.EXIT['refused']
    assert not xc.cell_paths('convexadam_v4seg', 'dev', W.DEV_PAIR)['lock'].exists() and method.calls == []
    assert [d['stage'] for d in refusal_docs()] == ['provenance']


def v4seg_world(monkeypatch):
    dev_world(monkeypatch)
    fixed, moving = W.DEV_PAIR.split('__')
    labels = {s: np.random.default_rng(i).integers(0, 36, W.SHAPE) for i, s in enumerate((fixed, moving))}
    for s in (fixed, moving):
        W.write_prediction(xc.HI_PRED_ROOTS['dev'], s, labels[s])
    return labels


@pytest.mark.parametrize('case', ['missing', 'another file', 'no labels', 'not a v4seg arm', 'holdout'])
def test_convexadam_v4seg_needs_the_methods_t88_run_before_the_lock(tmp_path, monkeypatch, case):
    """I4, m10: --method-run is required for ConvexAdam-v4seg, must be the pinned path (xb_core.method_t88_run), clear of the spy's name rules and hold labels_sha256; another arm takes none;
    a hold-out pair whose pinned path (HB: the method arm's cell on D:) is not written yet: refused.  A refusal record, nothing in the cell."""
    labels = v4seg_world(monkeypatch)
    run = W.method_run('dev', W.DEV_PAIR, labels)
    arm, pair_set, pair, given = 'convexadam_v4seg', 'dev', W.DEV_PAIR, None
    if case == 'another file':
        given = W.write_json(tmp_path/'elsewhere'/'run.json', json.loads(run.read_text()))
    elif case == 'no labels':
        W.write_json(run, dict(status='ok'))
        given = run
    elif case == 'not a v4seg arm':
        arm, given = 'convexadam', run
    elif case == 'holdout':
        pin_prereg(monkeypatch)
        pair = hc.pair_name(*W.holdout_rows()[0])
        W.write_stage_pair(hc.STAGE_ROOT, pair, seg=False)
        W.simulated_unseal(tmp_path, monkeypatch)
        pair_set = 'holdout'
        expected = xc.method_t88_run('holdout', pair)                  # HB's finalized and move records name it (HB review m13); neither is written here, so the cell is refused
        assert expected is None and xc.method_t88_sha256('holdout', pair) is None
        given = hc.big_dir('method', pair)/'t88'/'run.json'
    method = W.FakeMethod()
    assert W.run_cell(arm, pair_set, pair, method_run=given, deps=W.deps(step=method)) == xc.EXIT['refused'] and method.calls == []
    [doc] = refusal_docs(pair_set)
    assert doc['stage'] == 'method_run' and not xc.cell_paths(arm, pair_set, pair)['lock'].exists()


def test_convexadam_v4seg_whose_maps_are_not_the_methods_labels_is_infrastructure(monkeypatch):
    labels = v4seg_world(monkeypatch)
    fixed, moving = W.DEV_PAIR.split('__')
    run = W.method_run('dev', W.DEV_PAIR, {fixed: labels[moving], moving: labels[fixed]})
    method = W.FakeMethod()
    assert W.run_cell('convexadam_v4seg', 'dev', W.DEV_PAIR, method_run=run, deps=W.deps(step=method)) == xc.EXIT['infra_failure'] and method.calls == []
    doc = result('convexadam_v4seg')
    assert doc['status'] == 'infra_failure' and doc['failure']['kind'] == 'input_rejected' and "not the method's labels" in doc['failure']['reason']


def test_a_resume_needs_its_committed_authorization_and_the_same_frozen_configuration(monkeypatch):
    dev_world(monkeypatch)
    assert W.run_cell('niftyreg', 'dev', W.DEV_PAIR, deps=W.deps(step=W.FakeMethod(returncode=-9, events=(QUIET, QUIET)))) == 20
    first = result('niftyreg')
    assert first['status'] == 'infra_failure' and not xc.complete(first)
    paths = xc.cell_paths('niftyreg', 'dev', W.DEV_PAIR)
    ok = W.FakeMethod()
    path = W.authorize('resume', 'niftyreg', 'dev', W.DEV_PAIR, commit=False)
    rel = str(path.relative_to(hc.AUTH_ROOT))
    assert path.name == 'dev_xb_resume_1.json' and json.loads(path.read_text())['previous_record'] == str(paths['result'])
    assert W.run_cell('niftyreg', 'dev', W.DEV_PAIR, resume_rel=rel, deps=W.deps(step=ok)) == xc.EXIT['refused'] and ok.calls == []          # not committed
    W.git(W.auth_repo(), 'add', '-A')
    W.git(W.auth_repo(), 'commit', '-q', '-m', 'resume')
    W.authorize('rerun', 'niftyreg', 'dev', W.DEV_PAIR)                       # the refused attempt above is a refusal record: the next attempt needs its rerun authorization too (I1)
    # the frozen configuration changed since the first run: refused
    monkeypatch.setattr(xc, 'check_frozen', lambda arm, **kw: dict(config=dict(sha256='0'*64), passed=True))
    assert W.run_cell('niftyreg', 'dev', W.DEV_PAIR, resume_rel=rel, deps=W.deps(step=ok)) == xc.EXIT['refused'] and ok.calls == []
    W.authorize('rerun', 'niftyreg', 'dev', W.DEV_PAIR)
    W.fake_checks(monkeypatch)
    monkeypatch.setitem(xc.ARM_FUNCTIONS, 'niftyreg', xc.ARM_FUNCTIONS['py'])
    monkeypatch.setitem(xc.ARMS, 'niftyreg', dict(xc.ARMS['niftyreg'], script='harness/m_convexadam.py', args=[], post_args=[]))
    assert W.run_cell('niftyreg', 'dev', W.DEV_PAIR, resume_rel=rel, deps=W.deps(step=ok)) == 0
    second = result('niftyreg', resume=True)
    assert second['status'] == 'ok' and second['resume']['path'] == str(path) and second['lock']['path'] == str(hc.resume_lock_path(paths['lock']))
    assert Path(second['store']['work']).name.endswith('.resume') and result('niftyreg')['status'] == 'infra_failure'     # the first record stays
    assert W.run_cell('niftyreg', 'dev', W.DEV_PAIR, resume_rel=rel, deps=W.deps(step=W.FakeMethod())) == xc.EXIT['refused']       # at most one resume


def test_a_committed_no_resume_authorization_makes_an_infrastructure_failure_final(monkeypatch):
    """I3: 未批补跑 is a committed authorization (BZ, or the lead on BZ's behalf with BZ's words; reason, cell, the record's sha256, time); written but not committed, or changed after the commit,
    it does not count."""
    import xb_chain
    dev_world(monkeypatch)
    W.run_cell('deeds', 'dev', W.DEV_PAIR, deps=W.deps(step=W.FakeMethod(returncode=-9, events=(QUIET, QUIET))))
    paths = xc.cell_paths('deeds', 'dev', W.DEV_PAIR)
    assert xb_chain.cell_state('deeds', 'dev', W.DEV_PAIR)[0] == 'needs_resume'
    path = W.authorize('no_resume', 'deeds', 'dev', W.DEV_PAIR, commit=False)
    doc = json.loads(path.read_text())
    assert doc['note'] == '未批补跑' and doc['record'] == str(paths['result']) and doc['record_sha256'] == xc.sha256(paths['result']) and doc['reason'] and doc['words'] and doc['time_local']
    assert xb_chain.cell_state('deeds', 'dev', W.DEV_PAIR)[0] == 'needs_resume'                     # not committed
    W.git(W.auth_repo(), 'add', '-A')
    W.git(W.auth_repo(), 'commit', '-q', '-m', 'no resume')
    assert xb_chain.cell_state('deeds', 'dev', W.DEV_PAIR)[0] == 'final'
    path.write_text(path.read_text().replace('"test"', '"edited"'))
    assert xb_chain.cell_state('deeds', 'dev', W.DEV_PAIR)[0] == 'needs_resume'                     # changed after the commit
    with pytest.raises(hc.Refused):
        xb_drive.authorization_draft('no_resume', 'deeds', 'dev', W.DEV_PAIR, written_by='someone', words='x', time_local='y', reason='z')


def test_a_spy_breach_before_the_lock_is_a_controlled_refusal(monkeypatch):
    """m7: the driver's spy stops an open of a forbidden path before the lock: a refusal record (infrastructure), exit 20, no lock, no traceback."""
    dev_world(monkeypatch)

    def peeking(arm, **kw):
        try:
            open(hc.RESULTS/'other_tool'/'peek.json')
        except FileNotFoundError:
            pass
        return dict(passed=True)
    monkeypatch.setattr(xc, 'check_environment', peeking)
    method = W.FakeMethod()
    assert W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=W.deps(step=method)) == xc.EXIT['infra_failure'] and method.calls == []
    [doc] = refusal_docs()
    assert doc['stage'] == 'spy_breach' and doc['refusal_class'] == 'infrastructure' and doc['spy']['breaches'][0]['path'].endswith('peek.json')
    assert not xc.cell_paths('fireants_greedy', 'dev', W.DEV_PAIR)['lock'].exists()


def test_a_breach_swallowed_after_the_lock_changes_the_status(monkeypatch):
    """m7: a breach some code swallows after the lock leaves the driver's spy not passed: the record is infra_failure (spy_not_passed), whatever the method returned."""
    dev_world(monkeypatch)

    class Swallowing(W.FakeMethod):
        def __call__(self, *args, **kwargs):
            try:
                open(hc.RESULTS/'other_tool'/'peek.json')
            except Exception:                       # noqa: BLE001 - the swallowing library of the test
                pass
            return super().__call__(*args, **kwargs)
    assert W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=W.deps(step=Swallowing())) == xc.EXIT['infra_failure']
    doc = result('fireants_greedy')
    assert doc['status'] == 'infra_failure' and doc['failure']['kind'] == 'spy_not_passed' and doc['failure']['superseded']['status'] == 'ok' and doc['spy_passed'] is False


def test_the_precheck_runs_before_the_lock(monkeypatch):
    """m8: the arm's precheck (MemAvailable, the GPU idle) before the lock: a GPU that stays busy is a refusal record (infrastructure), no lock taken."""
    dev_world(monkeypatch)
    d = W.deps()
    seen = []

    def busy(gpu, log=None):
        seen.append(gpu)
        raise hc.InfrastructureFailure('precheck', 'the GPU not idle after 900 s')
    d.precheck = busy
    assert W.run_cell('unigradicon_io', 'dev', W.DEV_PAIR, deps=d) == xc.EXIT['infra_failure'] and seen == [True]
    [doc] = refusal_docs()
    assert doc['stage'] == 'precheck' and not xc.cell_paths('unigradicon_io', 'dev', W.DEV_PAIR)['lock'].exists()


def test_a_real_root_is_refused_in_the_test_sandbox(monkeypatch):
    """I7: under XB_TEST_SANDBOX holdout_common's root functions refuse the real roots: a test that forgot to move a root stops before it writes anything."""
    assert os.environ.get(hc.TEST_SANDBOX_ENV)
    before = W.real_roots_present()
    for target in (hc.REAL_RESULTS/'holdout_B_v1'/'x.json', hc.REAL_RESULTS/'xb_dev_v1'/'batches'/'b'/'x.json', hc.REAL_RESULTS/'xb_limit_v1'/'x.json', hc.REAL_BIG/'a'/'x.json'):
        with pytest.raises(hc.RealRootInTest):
            hc.write_json_once(target, {})
    assert hc.real_root_problem(hc.REAL_RESULTS/'seg'/'x') is None and hc.real_root_problem(hc.RESULTS/'holdout_B_v1') is None
    monkeypatch.setattr(hc, 'HOLDOUT_ROOT', hc.REAL_RESULTS/'holdout_B_v1')                # a test that forgot to move the roots
    monkeypatch.setattr(hc, 'BIG_ROOT', hc.REAL_BIG)
    monkeypatch.setattr(xc, 'DEV_ROOT', hc.REAL_RESULTS/'xb_dev_v1')
    with pytest.raises(hc.RealRootInTest):
        hc.arm_dir('ants_s')
    with pytest.raises(hc.RealRootInTest):
        hc.take_lock(hc.REAL_RESULTS/'holdout_B_v1'/'locks'/'x.lock', dict(arm='x'))
    with pytest.raises(hc.RealRootInTest):
        xc.cell_paths('ants_s', 'dev', W.DEV_PAIR)
    import xb_regress
    import xb_limit_experiment
    with pytest.raises(hc.RealRootInTest):
        xb_regress.root()
    monkeypatch.setattr(hc, 'RESULTS', hc.REAL_RESULTS)
    with pytest.raises(hc.RealRootInTest):
        xb_limit_experiment.root()
    dev_world(monkeypatch)
    with pytest.raises(hc.RealRootInTest):
        W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=W.deps())                # the driver itself, before it writes anything
    assert W.real_roots_present() == before
    monkeypatch.delenv(hc.TEST_SANDBOX_ENV)
    assert hc.guard_root(hc.REAL_BIG/'a') == hc.REAL_BIG/'a'                             # outside the tests: the path, unchanged


UNIT = 'tag-hb-x-ants_s-0438-0439'
MEMCG = f'/user.slice/user-1000.slice/user@1000.service/app.slice/{UNIT}.service'


@pytest.mark.parametrize('kernel, events', [
    (None, (QUIET, OOM)),                                                                                           # memory.events of the unit
    (f'[ 20.000000] oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=/,mems_allowed=0,oom_memcg={MEMCG},task_memcg={MEMCG},task=antsRegistration,pid=5,uid=1000\n', (QUIET, QUIET)),
])
def test_ants_exit_0_after_an_oom_kill_is_infrastructure_not_a_method_failure(monkeypatch, kernel, events):
    """I2 (the review's case): antsRegistrationSyN.sh exits 0 after its antsRegistration was OOM-killed; the step's OOM evidence (memory.events, or the kernel's line of this unit, read on every exit)
    makes it infra_failure, not method_failure (no Warp)."""
    dev_world(monkeypatch)
    monkeypatch.setenv('XB_UNIT', UNIT)
    lines = None if kernel is None else hc.parse_oom(kernel, 0.)
    method = W.FakeMethod(returncode=0, events=events, kernel=lines)
    assert W.run_cell('ants_s', 'dev', W.DEV_PAIR, deps=W.deps(step=method)) == xc.EXIT['infra_failure']
    doc = result('ants_s')
    assert doc['status'] == 'infra_failure' and doc['failure']['kind'] == 'oom' and 'exit 0' in doc['failure']['reason']


def test_cli_arms_get_the_derived_images_as_nifti(monkeypatch):
    dev_world(monkeypatch)
    method = W.FakeMethod(returncode=1)
    assert W.run_cell('ants_s', 'dev', W.DEV_PAIR, deps=W.deps(step=method)) == 1
    call = method.calls[0]
    assert call['cmd'][0].endswith('antsRegistrationSyN.sh') and call['cmd'][-4:] == ['-t', 's', '-n', '8'] and 'ANTS_RANDOM_SEED' not in call['env']
    assert call['env']['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] == '8' and (Path(call['cwd'])/'fixed.nii.gz').exists()
    assert result('ants_s')['status'] == 'method_failure'


def test_rule_jia_reads_the_units_own_limit_as_a_unit_limit(monkeypatch):
    dev_world(monkeypatch)
    monkeypatch.setattr(hc, 'LIMIT_RULE', 'jia')
    method = W.FakeMethod(returncode=-9, events=(dict(max=0, oom=0, oom_kill=0), dict(max=3, oom=1, oom_kill=1)))
    assert W.run_cell('fireants_syn', 'dev', W.DEV_PAIR, deps=W.deps(step=method)) == xc.EXIT['unit_limit']
    doc = result('fireants_syn')
    assert doc['status'] == 'unit_limit' and xc.complete(doc)


def test_a_populated_sealed_list_cache_refuses_before_the_spy(monkeypatch):
    """C23.6-5: a driver whose process already holds the guard's copy of the sealed lists would let the guard skip the read the spy must see: refused, nothing read, no record."""
    dev_world(monkeypatch)
    W.sealed_module.guard_sealed_lists()
    method = W.FakeMethod()
    assert xb_drive.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=W.deps(step=method)) == xc.EXIT['refused']        # not W.run_cell: the cache is left populated
    assert method.calls == [] and not xc.cell_paths('fireants_greedy', 'dev', W.DEV_PAIR)['lock'].exists() and not xc.cell_paths('fireants_greedy', 'dev', W.DEV_PAIR)['result'].exists()


# ---- A5a: rule one of a hold-out cell (contract C24.3; decision_log 2026-11-10 ~2:28 AM CST n8, ~2:59 AM CST (1)) --------------------------------------
def test_a_holdout_cell_checks_rule_one_under_its_spy_and_its_lock_names_p1(tmp_path, monkeypatch):
    """The real rule one (holdout_common.pinned_code_problems) on a temporary tree whose pinned record names P1, inside the driver's spy (git only: no open the spy refuses): the run goes
    through, the lock line's code_commit is P1 and its code_head the HEAD of the code record; the refusal records carry their sequence number (P20's "newest")."""
    pair, original, sim = holdout_world(tmp_path, monkeypatch)
    repo, p1 = W.pinned_repo(tmp_path/'pinned_tree')
    assert W.run_cell('fireants_greedy', 'holdout', pair, deps=W.deps(pinned=lambda: hc.pinned_code_problems(repo=repo))) == 0
    doc = result('fireants_greedy', 'holdout', pair)
    assert doc['status'] == 'ok' and doc['spies']['driver']['passed'] and doc['checks']['pinned_code']['pinned'] == p1 and doc['checks']['pinned_code']['passed'] is True
    assert doc['lock']['line']['code_commit'] == p1 and doc['lock']['line']['code_head'] == 'c'*40


def test_a_holdout_cell_whose_code_is_not_the_pinned_commits_is_refused_before_the_admission(tmp_path, monkeypatch):
    """No pinned record (main today): a real run is refused at stage code before the admission (a refusal record with its seq, nothing in the cell); a dry run lists the problem and checks the rest."""
    pair, original, sim = holdout_world(tmp_path, monkeypatch)
    bare = tmp_path/'bare'
    bare.mkdir()
    W.git(bare, 'init', '-q')
    (bare/'x.md').write_text('x\n')
    W.git(bare, 'add', '-A')
    W.git(bare, 'commit', '-q', '-m', 'no pinned record')
    pinned = lambda: hc.pinned_code_problems(repo=bare)            # noqa: E731
    method = W.FakeMethod()
    assert W.run_cell('fireants_greedy', 'holdout', pair, deps=W.deps(step=method, pinned=pinned)) == xc.EXIT['refused'] and method.calls == []
    [doc] = refusal_docs('holdout')
    assert doc['stage'] == 'code' and 'rule one' in doc['reason'] and doc['seq'] == 1 and 'admission' not in doc['checks_run']
    assert not xc.cell_paths('fireants_greedy', 'holdout', pair)['lock'].exists()
    out = tmp_path/'dry'
    out.mkdir()
    assert W.run_cell('fireants_greedy', 'holdout', pair, dry_run_out=out, deps=W.deps(pinned=pinned)) == xc.EXIT['refused']
    dry = json.loads((out/'dry_run.json').read_text())
    assert [p['stage'] for p in dry['problems_by_stage']] == ['code', 'rerun'] and dry['checks']['admission']['pairs_tsv']['row'] == 1 and dry['checks']['pinned_code']['passed'] is False   # rerun: the refusal above (I1)
    assert [p['kind'] for p in dry['problems_by_stage']] == ['rule_one_unpinned', 'problem'] and [e['kind'] for e in dry['expected']] == ['rule_one_unpinned']     # A5a review m8
    assert [e['problem'].split(':')[0] for e in dry['unexpected']] == ['rerun'] and dry['p0_ok'] is False and dry['passed'] is False
