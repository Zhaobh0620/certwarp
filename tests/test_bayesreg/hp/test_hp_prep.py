"""HP data preparation (experiments/holdout/prep_stage.py; runs/20261107-hp/任务书.md item 1, 2): the stage holds no label member and no label-derived field
(C23.12-3, C23.12-7 "数据准备", prereg revision 1 item 1), the spy shows that no label file is opened, the admission refuses without SG's unseal object, for a
subject off the record, a pair off the list or reversed, H1-H4 -- and reads nothing then; the lock, the output, the grid check; the development check's comparison."""
import json
from pathlib import Path

import numpy as np
import pytest

from .. import _p0_pins as P0
from . import _world as W

hc, core, prep, S = W.hc, W.core, W.prep, W.S


def run(w, **kwargs):
    W.S._reset_cache()
    return prep.run_holdout(unseal_path=kwargs.pop('unseal_path', w.record_path), code=W.CODE, log=W.quiet, **kwargs)


def refused(w, stage, match=None, **kwargs):
    with pytest.raises(hc.Refused) as info:
        run(w, **kwargs)
    assert info.value.stage == stage, info.value
    if match:
        assert match in str(info.value), info.value
    return info.value


def nothing_read(exc, w):
    """No file of a subject was opened (the tool's own spy, attached to the refusal), no stage, no lock."""
    events = exc.hp_spy['events']
    data = str(Path(w.data).resolve())
    assert not [e for e in events if e['resolved'].startswith(data)], events
    assert not Path(hc.STAGE_ROOT).exists()
    assert not (Path(hc.LOCK_ROOT)/prep.LOCK_NAME).exists()


# ---- the stage ------------------------------------------------------------------------------------------------------------------------------------
def test_the_hold_out_stage_has_no_label_member_and_no_label_file_is_opened(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    spy = W.OpenSpy(monkeypatch)
    record = run(w)
    stage = Path(hc.STAGE_ROOT)
    assert record['passed'] and record['spy']['passed'] and not record['spy']['breaches']
    assert spy.labels() == [] and W.label_files(w.data)                     # the synthetic aligned_seg35 files are there and were never opened
    for fixed, moving in W.PAIRS:
        d = stage/f'{fixed}__{moving}'
        with np.load(d/'pair.npz') as npz:                                  # the test lists the synthetic archive: exactly these members, in this order
            assert npz.files == ['fixed', 'moving', 'initial_u']
            assert not [k for k in npz.files if 'seg' in k or 'label' in k]
            u = npz['initial_u']
            assert u.dtype == np.float64 and u.shape == W.SHAPE_ZYX+(3,) and not u.any()
            for side, subject in (('fixed', fixed), ('moving', moving)):
                v = W.load_scalar(w.data/subject/'aligned_norm.nii.gz')
                a = npz[side]
                assert a.dtype == np.float64 and a.tobytes() == np.asarray(v.values).tobytes()
        text = (d/'manifest.json').read_text()
        assert 'seg' not in text.lower() and 'label' not in text.lower()
        manifest = json.loads(text)
        assert sorted(manifest['sources']) == ['fixed', 'moving']
        assert manifest['pair_sha256'] == W.file_sha256(d/'pair.npz')
        assert manifest['shape_zyx'] == list(W.SHAPE_ZYX) and manifest['spacing_xyz'] == [1., 1., 1.]
    for name in ('selection.json', 'completion.json', prep.SUMS):
        text = (stage/name).read_text()
        assert 'seg' not in text.lower() and 'label' not in text.lower(), name
    sums = dict(reversed(line.split('  ', 1)) for line in (stage/prep.SUMS).read_text().splitlines())
    assert sorted(sums) == sorted(prep.stage_files(W.PAIRS)) and all(W.file_sha256(stage/n) == s for n, s in sums.items())
    images = {str((w.data/s/'aligned_norm.nii.gz').resolve()) for p in W.PAIRS for s in p}
    assert set(record['spy']['expected_reads']) >= images and all(n >= 1 for n in record['spy']['expected_reads'].values())
    assert all(n >= 1 for n in record['spy']['sealed_list_reads'].values())    # C23.6-5: the sealed lists were read under the spy and recorded
    assert {m['member'] for m in record['spy']['members']} <= set(core.PAIR_MEMBERS)
    assert json.loads((stage/prep.RECORD).read_text())['admission']['unseal']['sha256'] == W.file_sha256(w.record_path)


def test_the_loader_reads_the_stage_and_its_manifest_fields_are_the_whitelist(tmp_path, monkeypatch):
    from bayesreg.io.pair import load_pair
    from bayesreg.io.unseal import bind_unseal, load_unseal
    w = W.make_world(tmp_path, monkeypatch)
    run(w)
    W.S._reset_cache()
    bound = bind_unseal(w.allowlist, load_unseal(w.record_path))
    pair = load_pair(Path(hc.STAGE_ROOT)/f'{W.PAIRS[0][0]}__{W.PAIRS[0][1]}'/'pair.npz', allowlist=bound)
    assert pair.provenance['keys_read'] == ['fixed', 'moving', 'initial_u'] and pair.provenance['subject_guard']['unsealed'] is True
    manifest = json.loads((Path(hc.STAGE_ROOT)/pair.provenance['pair_path'].split('/')[-2]/'manifest.json').read_text())
    dropped = {'l2r_fixed', 'l2r_moving', 'baseline_label_metrics', 'labels_policy'}
    assert not dropped & set(manifest)
    assert set(manifest) == {'dataset', 'pair', 'fixed_subject', 'moving_subject', 'mapping', 'shape_zyx', 'spacing_xyz', 'origin_xyz', 'unit', 'preprocessing', 'reader',
                             'sources', 'pair_sha256', 'baseline_native_mse', 'weights', 'holdout_row', 'pair_list'}
    assert manifest['holdout_row'] == 1 and manifest['pair_list'] == dict(path=hc.PAIRS_TSV_REL, sha256=hc.PAIRS_TSV_SHA256)
    assert 'labels' not in manifest['sources']['fixed'] and manifest['sources']['fixed']['sha256'] == W.file_sha256(w.data/W.PAIRS[0][0]/'aligned_norm.nii.gz')


def test_a_label_file_opened_stops_the_run_and_the_record_stays(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    real = prep.write_pair

    def peeking(out, fixed, moving, files, **kwargs):
        (Path(files.root)/fixed/'aligned_seg35.nii.gz').read_bytes()
        return real(out, fixed, moving, files, **kwargs)
    monkeypatch.setattr(prep, 'write_pair', peeking)
    with pytest.raises(core.Breach, match='label or evaluation file'):
        run(w)
    stopped = json.loads((Path(hc.STAGE_ROOT)/prep.STOPPED).read_text())
    assert 'Breach' in stopped['stopped'] and stopped['spy']['breaches']
    assert (Path(hc.LOCK_ROOT)/prep.LOCK_NAME).exists()
    assert not (Path(hc.STAGE_ROOT)/prep.RECORD).exists()


def test_a_file_outside_the_policy_stops_the_run(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    other = W.write_text(tmp_path/'elsewhere.txt', 'x')
    real = prep.write_pair

    def peeking(out, fixed, moving, files, **kwargs):
        other.read_text()
        return real(out, fixed, moving, files, **kwargs)
    monkeypatch.setattr(prep, 'write_pair', peeking)
    with pytest.raises(core.Breach, match='unlisted'):
        run(w)


def test_another_subjects_image_is_not_allowed(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    real = prep.write_pair

    def peeking(out, fixed, moving, files, **kwargs):
        (Path(files.root)/W.DEV[0]/'aligned_norm.nii.gz').read_bytes()
        return real(out, fixed, moving, files, **kwargs)
    monkeypatch.setattr(prep, 'write_pair', peeking)
    monkeypatch.setattr(core, 'DATA_ROOTS', (w.data,))                        # the synthetic root plays /mnt/d/data
    with pytest.raises(core.Breach, match='a data root'):
        run(w)


# ---- the admission: refused, nothing read ---------------------------------------------------------------------------------------------------------
def test_no_unseal_object_is_refused_and_nothing_is_read(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch, record=False)
    spy = W.OpenSpy(monkeypatch)
    exc = refused(w, 'unseal', match='no valid unseal object', unseal_path=None)
    nothing_read(exc, w)
    assert spy.images() == []
    exc = refused(w, 'unseal', unseal_path=w.record_path)              # the pinned path, but no record there
    nothing_read(exc, w)


def test_an_uncommitted_record_is_refused(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    W.write_json(w.record_path, dict(w.document, purpose='changed after the commit'))
    nothing_read(refused(w, 'unseal', match='not a committed, unchanged file'), w)


@pytest.mark.parametrize('case', ['record_not_the_list', 'not_in_tsv', 'reversed'])
def test_a_pair_or_subject_off_the_record_or_the_list_is_refused(tmp_path, monkeypatch, case):
    if case == 'record_not_the_list':
        # a fifth hold-out subject is on the sealed list and in the pair list, but the record names only four (its subjects_sha256 and n_subjects): bayesreg's loader
        # refuses the record (C23.3-2 (vi)), so no unseal object exists for anyone
        extra = W.oasis(905)
        rows = W.PAIRS+[(extra, W.HOLDOUT[0])]
        w = W.make_world(tmp_path, monkeypatch, rows=rows, unsealed_extra=[extra], subjects_sha256=W.subject_list_sha256(sorted(W.HOLDOUT)), n_subjects=len(W.HOLDOUT))
        nothing_read(refused(w, 'unseal'), w)
        return
    w = W.make_world(tmp_path, monkeypatch)
    W.S._reset_cache()
    pairs = [(W.HOLDOUT[0], W.HOLDOUT[2])] if case == 'not_in_tsv' else [(W.HOLDOUT[1], W.HOLDOUT[0])]
    with pytest.raises(hc.Refused) as info:
        core.admit_holdout(pairs, unseal_path=w.record_path, data_root=w.data)
    assert info.value.stage == 'admission'
    assert ('the other way round' in str(info.value)) == (case == 'reversed')
    assert ('is not a row' in str(info.value)) == (case == 'not_in_tsv')


def test_a_subject_not_unsealed_is_refused_even_on_the_allowlist(tmp_path, monkeypatch):
    """A development, T0 or training subject in the pair list (on no sealed list: the guard admits it as a development subject), or an id on no list at all: the
    hold-out admission refuses it -- the subject is not in the unseal object's list."""
    for group in ('dev', 't0', 'train', 'unknown'):
        subject = dict(dev=W.DEV[0], t0=W.T0[0], train=W.TRAIN[0], unknown=W.oasis(950))[group]
        sub = tmp_path/group
        sub.mkdir()
        with monkeypatch.context() as m:
            w = W.make_world(sub, m, rows=W.PAIRS+[(subject, W.HOLDOUT[0])])
            m.setattr(prep, 'DATA_ROOT', w.data)
            exc = refused(w, 'admission', match='sealed lists')
            nothing_read(exc, w)


def test_the_guard_layer_refuses_on_its_own(tmp_path, monkeypatch):
    """With the id step (hp_core._ids_problem) switched off, SG's check_subject on the bound allowlist still refuses a development subject (not through the unseal
    object), a wrongly allowlisted H1 subject (H1-H4 item) and an id off the allowlist."""
    extra = [('h1_wrongly', [W.H1[0]], False)]
    for subject, words in ((W.DEV[0], 'not admitted through the unseal object'), (W.H1[0], 'paper 1'), (W.oasis(950), 'not on the development allowlist')):
        sub = tmp_path/subject
        sub.mkdir()
        with monkeypatch.context() as m:
            w = W.make_world(sub, m, rows=W.PAIRS+[(W.HOLDOUT[0], subject)], extra_sources=extra)
            m.setattr(core, '_ids_problem', lambda subject, side: None)
            exc = refused(w, 'admission', match=words)
            nothing_read(exc, w)


@pytest.mark.parametrize('name', ['H1', 'H4', 'H2', 'H3'])
def test_h1_to_h4_are_refused_whatever_else_holds(tmp_path, monkeypatch, name):
    subject = dict(H1=W.H1[0], H4=W.H4[0], H2='S37', H3=W.H3[0])[name]
    extra = [('h1_wrongly', [W.H1[0]], False), ('h4_wrongly', [W.H4[0]], False)]          # H1 and H4 wrongly on the allowlist too
    w = W.make_world(tmp_path, monkeypatch, rows=W.PAIRS+[(W.HOLDOUT[0], subject)], extra_sources=extra)
    exc = refused(w, 'never', match=name)
    nothing_read(exc, w)


def test_the_pair_list_must_be_the_pinned_one(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    monkeypatch.setattr(hc, 'PAIRS_TSV_SHA256', '0'*64)
    nothing_read(refused(w, 'pairs', match='is not the pinned'), w)


def test_the_real_ids_on_a_simulated_unseal_branch(tmp_path, monkeypatch):
    """The real sealed lists and the real pair list (ids only, copied byte for byte into a temporary tree that holds a synthetic record unsealing holdout_B_v1): the
    admission passes every real row in its direction, refuses a real row reversed, a real H1 subject and a training subject; nothing of a subject is opened."""
    import bayesreg.io.unseal as U
    from bayesreg.io.guard import load_dev_allowlist
    repo = tmp_path/'sim'
    repo.mkdir()
    W.git(repo, 'init', '-q')
    for spec in W.REAL_PINS['SEALED_LISTS']:
        (repo/spec['path']).parent.mkdir(parents=True, exist_ok=True)
        (repo/spec['path']).write_bytes((W.ROOT/spec['path']).read_bytes())
    (repo/hc.PAIRS_TSV_REL).write_bytes((W.ROOT/hc.PAIRS_TSV_REL).read_bytes())
    holdout = sorted(S.load_sealed_lists(repo).subjects['holdout_B_v1'])
    document = json.loads((W.ROOT/U.ALLOWLIST_REL).read_text(encoding='utf-8'))
    spec = {s['name']: s for s in W.REAL_PINS['SEALED_LISTS']}['holdout_B_v1']
    document['sources'].append(dict(name='holdout_B_v1_unsealed', dataset='OASIS', path=spec['path'], sha256=spec['sha256'], never_for_training=True, subjects=holdout))
    document['subjects']['OASIS'] = list(document['subjects']['OASIS'])+holdout
    allowlist = load_dev_allowlist(W.write_json(repo/U.ALLOWLIST_REL, document))
    record = dict(schema=U.UNSEAL_SCHEMA, written_by="lead on BZ's behalf", bz_words=f'（测试）{U.DELEGATION_WORDS}，模拟解封分支', time_local='2026-11-07 6:00 AM CST',
                  utc='2026-11-07T12:00:00Z', sealed_list='holdout_B_v1', subjects_file=spec['path'], subjects_file_sha256=spec['sha256'],
                  subjects_sha256=W.subject_list_sha256(holdout), n_subjects=len(holdout), allowlist_source='holdout_B_v1_unsealed', allowlist_sha256=allowlist.sha256,
                  segmenter_id='unet35_v4_ens', registry_sha256='b'*64, purpose='test: the real ids')
    W.write_json(repo/U.UNSEAL_REL, record)
    W.log_line(repo, repo/U.UNSEAL_REL, record['bz_words'])
    W.commit_all(repo, 'simulated unseal branch')
    W.pin(monkeypatch, repo, W.REAL_PINS['SEALED_LISTS'])
    monkeypatch.setattr(hc, 'PAIRS_TSV_SHA256', W.REAL_HC['PAIRS_TSV_SHA256'])
    monkeypatch.setattr(hc, 'N_PAIRS', W.REAL_HC['N_PAIRS'])
    data = tmp_path/'no_data'                                                   # never read: bind_files only names the root
    admission = core.admit_holdout(None, unseal_path=repo/U.UNSEAL_REL, data_root=data)
    assert len(admission.pairs) == 50 and admission.row_numbers == list(range(1, 51)) and admission.files.root == data
    assert not data.exists()
    fixed, moving = admission.rows[0]
    with pytest.raises(hc.Refused, match='the other way round'):
        core.admit_holdout([(moving, fixed)], unseal_path=repo/U.UNSEAL_REL)
    h1 = sorted(S.load_sealed_lists(repo).subjects['H1'])[0]
    with pytest.raises(hc.Refused) as info:
        core.admit_holdout([(fixed, h1)], unseal_path=repo/U.UNSEAL_REL)
    assert info.value.stage == 'never'
    train = W.ROOT/'splits/bayesreg_v1/train_subjects.txt'
    trainee = [s for s in train.read_text().split() if s.startswith('OASIS')][0]
    with pytest.raises(hc.Refused, match='sealed lists'):
        core.admit_holdout([(fixed, trainee)], unseal_path=repo/U.UNSEAL_REL)


# ---- once, the output, the grid --------------------------------------------------------------------------------------------------------------------
def test_the_second_run_is_refused_and_the_first_stays(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    run(w)
    stage = Path(hc.STAGE_ROOT)
    before = {p: W.file_sha256(p) for p in stage.rglob('*') if p.is_file()}
    refused(w, 'once', match='resume record')                               # the lock first (holdout_common.check_once)
    lock = Path(hc.LOCK_ROOT)/prep.LOCK_NAME
    lock.rename(lock.with_name('moved.lock'))                                # synthetic: without the lock the stage root still refuses
    refused(w, 'output', match='written once')
    assert {p: W.file_sha256(p) for p in stage.rglob('*') if p.is_file()} == before


def test_the_lock_is_taken_after_the_machine_gate(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)

    def busy(**kwargs):
        raise hc.Refused('machine', 'other units are active')
    monkeypatch.setattr(hc, 'machine_gate', busy)
    exc = refused(w, 'machine')
    nothing_read(exc, w)


def test_a_grid_other_than_the_expected_one_stops(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    monkeypatch.setattr(prep, 'EXPECTED_SHAPE_ZYX', (192, 224, 160))
    with pytest.raises(prep.StageCheckFailed, match='expected'):
        run(w)
    assert json.loads((Path(hc.STAGE_ROOT)/prep.STOPPED).read_text())['stopped'].startswith('StageCheckFailed')


def test_two_images_on_different_grids_stop(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    W.synthetic.write_subjects(tmp_path/'other', [W.HOLDOUT[1]], shape_xyz=(10, 12, 9))
    (w.data/W.HOLDOUT[1]/'aligned_norm.nii.gz').write_bytes((tmp_path/'other'/W.HOLDOUT[1]/'aligned_norm.nii.gz').read_bytes())
    with pytest.raises(prep.StageCheckFailed, match='same canonical grid'):          # review m9: a check of the original script, exit 8
        run(w)
    assert json.loads((Path(hc.STAGE_ROOT)/prep.STOPPED).read_text())['stopped'].startswith('StageCheckFailed')


def test_the_sealed_cache_must_be_empty_when_the_spy_is_installed(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    S.guard_sealed_lists()
    with pytest.raises(hc.Refused) as info:
        prep.run_holdout(unseal_path=w.record_path, code=W.CODE, log=W.quiet)
    assert info.value.stage == 'spy'
    assert not Path(hc.STAGE_ROOT).exists()


def test_the_dry_run_on_the_real_tree_is_refused_at_the_admission(tmp_path, monkeypatch, capsys):
    """main without an unseal object, on this code tree's real pins (the real pair list, the real sealed lists, no record): refused at the admission (stage unseal),
    the record written into --out only."""
    W.real_pins(monkeypatch)
    out = tmp_path/'dry'
    code = prep.main(['run', '--dry-run', '--out', str(out)], code=dict(W.CODE, root=str(W.ROOT)), out=W.quiet)
    assert code == 8
    record = json.loads((out/'dry_run.json').read_text())
    assert record['refused']['stage'] == 'unseal'
    assert 'no_unseal_record' in [e['kind'] for e in record['expected']] and record['passed'] is False and len(record['problems']) == len(record['expected'])+len(record['unexpected'])   # A5a review m8
    assert not [e for e in record['spy']['events'] if '/mnt/d' in e['resolved'] or 'aligned' in e['resolved']]
    assert any(e['resolved'].endswith('holdout_B_pairs.tsv') for e in record['spy']['events'])
    assert not Path(hc.STAGE_ROOT).exists() and not Path(hc.LOCK_ROOT).exists()
    assert prep.main(['run', '--dry-run', '--out', str(hc.RESULTS/'x')], code=W.CODE, out=W.quiet) == 2


def test_the_dry_run_passes_the_admission_with_a_record(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    out = tmp_path/'dry'
    W.S._reset_cache()
    monkeypatch.setattr(core, 'check_code', lambda code, require_clean=True: dict(checked='test'))
    assert prep.main(['run', '--dry-run', '--out', str(out), '--unseal', str(w.record_path)], code=W.CODE, out=W.quiet) == 0
    record = json.loads((out/'dry_run.json').read_text())
    assert 'tag-hb-prep' in record['launch'] and record['images']['n'] == 4
    assert record['p0_ok'] is True and record['unexpected'] == [] and set(e['kind'] for e in record['expected']) <= set(hc.P0_EXPECTED)        # A5a review m8
    assert not Path(hc.STAGE_ROOT).exists() and not Path(hc.LOCK_ROOT).exists()


# ---- the code check ---------------------------------------------------------------------------------------------------------------------------------
def test_the_code_check_wants_one_clean_tree(monkeypatch):
    W.real_pins(monkeypatch)
    good = dict(kind='git_worktree', clean=True, root=str(W.ROOT), imports_inside_root=True, imported_from={})
    assert core.check_code(good)['problems'] == []
    with pytest.raises(hc.Refused, match='code'):
        core.check_code(dict(good, clean=False, n_dirty=1, dirty_paths=['x']))
    with pytest.raises(hc.Refused):
        core.check_code(dict(good, root='/tmp'))
    assert core.check_code(dict(good, clean=False), require_clean=False)['problems']


# ---- the development check --------------------------------------------------------------------------------------------------------------------------
def devcheck_world(tmp_path, monkeypatch, *, perturb=None):
    w = W.make_world(tmp_path, monkeypatch, decided=False)
    monkeypatch.setattr(hc, 'DEV_PAIRS', W.DEV_PAIRS)
    W.write_old_stage(hc.DEV_STAGE, w.data, W.DEV_PAIRS, perturb=perturb)
    monkeypatch.setattr(prep, 'PREP_DEVCHECK_PLAN_SHA256', W.plan_file(w, prep.DEVCHECK_PLAN_REL))
    return w


def test_the_development_check_reads_bitwise_on_an_equal_stage(tmp_path, monkeypatch):
    w = devcheck_world(tmp_path, monkeypatch)
    W.S._reset_cache()
    record = prep.run_devcheck(code=W.CODE, log=W.quiet)
    assert record['reading'] == '逐位相同 2/2' and record['n_equal'] == 2
    row = record['pairs'][W.DEV_PAIRS[0]]
    assert all(m['bitwise'] for m in row['members'].values()) and row['load_pair']['keys_equal']
    asked = {m['member'] for m in record['spy']['members']}
    assert asked <= set(core.PAIR_MEMBERS)                                     # the old stage's label members were never asked
    assert json.loads((prep.devcheck_root()/'devcheck_prep.json').read_text())['reading'] == '逐位相同 2/2'


def test_the_development_check_reads_a_difference(tmp_path, monkeypatch):
    w = devcheck_world(tmp_path, monkeypatch, perturb=(W.DEV_PAIRS[1], 'moving'))
    W.S._reset_cache()
    record = prep.run_devcheck(code=W.CODE, log=W.quiet)
    assert record['reading'] == prep.READ_DIFFERENT and record['n_equal'] == 1
    assert record['pairs'][W.DEV_PAIRS[1]]['members']['moving']['bitwise'] is False


def test_the_development_checks_plan_pin_is_the_committed_plan():
    """P0: PREP_DEVCHECK_PLAN_SHA256 is the sha256 of the committed prep_devcheck_plan.md; check_plan accepts it on this tree (only the plan is read).  The unpinned and
    the wrong value are refused in the test below and in test_hp_review."""
    P0.assert_pinned(prep.PREP_DEVCHECK_PLAN_SHA256, prep.DEVCHECK_PLAN_REL)
    assert prep.check_plan(P0.REPO) == dict(path=prep.DEVCHECK_PLAN_REL, sha256=prep.PREP_DEVCHECK_PLAN_SHA256)


def test_the_development_check_refuses_an_unpinned_plan_and_a_sealed_subject(tmp_path, monkeypatch):
    w = devcheck_world(tmp_path, monkeypatch)
    monkeypatch.setattr(prep, 'PREP_DEVCHECK_PLAN_SHA256', None)
    W.S._reset_cache()
    with pytest.raises(hc.Refused, match='draft'):
        prep.run_devcheck(code=W.CODE, log=W.quiet)
    W.S._reset_cache()
    with pytest.raises(hc.Refused, match='not one of the'):
        core.admit_dev([(W.HOLDOUT[0], W.HOLDOUT[1])])
    monkeypatch.setattr(hc, 'DEV_PAIRS', (f'{W.HOLDOUT[0]}__{W.HOLDOUT[1]}',))
    with pytest.raises(hc.Refused, match='not a development subject'):
        core.admit_dev([(W.HOLDOUT[0], W.HOLDOUT[1])])
    with pytest.raises(hc.Refused, match='never_for_training'):
        core.admit_dev(subjects=[W.T0[0]])
