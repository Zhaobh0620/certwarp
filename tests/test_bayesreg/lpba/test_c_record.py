"""LP-T0 item 3: experiments/lpba/c_record.py (c and W_dev, preregistration §2.2) on a synthetic development set only: the definitions, a subject's two arrays bit for bit,
the lock, the spy (label names, pair.npz members, unlisted files), the Atropos binding, the dry run; no real data and no real root is touched."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments'/'lpba'))
import c_record as cr  # noqa: E402

SHAPE = (6, 7, 5)                                   # zyx; load_scalar wants every axis >= 5
ORIGIN_XYZ = [-2., -3., -1.]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def subject_array(subject):
    rng = np.random.default_rng(int(subject[11:15]))
    v = rng.uniform(0., .9, SHAPE)
    v[0] = 0.                                       # a background slab: v > 0 is not everything
    v[1, 2, 3] = 0.
    return v


def posteriors(v):
    """CSF, GM, WM posteriors that sum to one, WM > .5 on the brighter voxels."""
    wm = np.where(v > .55, .8, .1)
    gm = np.where(v > .55, .15, .6)
    return [1.-wm-gm, gm, wm]


def save_xyz(path, zyx):
    affine = np.eye(4)
    affine[:3, 3] = ORIGIN_XYZ
    image = nib.Nifti1Image(np.ascontiguousarray(np.asarray(zyx).transpose(2, 1, 0)).astype(np.float32), affine)
    image.set_qform(affine, code=1)                 # load_scalar wants both codes (ITK's output, Atropos's, has them)
    image.set_sform(affine, code=1)
    image.header.set_xyzt_units('mm')
    nib.save(image, str(path))


def make_dev(tmp, *, tamper_pair=None, bad_binding=None, bad_posterior=None, edit_prov=None):
    """A synthetic development set: allowlist, stage (selection.json, 19 pair.npz with seg members and initial_u), Atropos root (provenance, three posteriors, decoys),
    a preregistration file.  tamper_pair: index of the pair whose fixed array is changed by one ulp (its subject's second appearance); edit_prov(subject, prov) returns
    the provenance document written for that subject."""
    stage, atropos = tmp/'stage', tmp/'atropos'
    stage.mkdir()
    atropos.mkdir()
    rows = [dict(pair=p, fixed=f'/mnt/d/data/{p.split("__")[0]}', moving=f'/mnt/d/data/{p.split("__")[1]}') for p in cr.DEV_PAIRS]
    selection = json.dumps(dict(rule='synthetic', pairs=rows, no_metric_based_selection=True), indent=1).encode()
    (stage/'selection.json').write_bytes(selection)
    allowlist = dict(schema='bayesreg.dev_subjects.v1', subjects=dict(OASIS=list(cr.DEV_SUBJECTS)),
                     sources=[dict(name='stage_L2R_val_v1', dataset='OASIS', path=cr.SELECTION_REL, sha256=sha(selection), never_for_training=False, subjects=list(cr.DEV_SUBJECTS))])
    (tmp/'dev_subjects.json').write_text(json.dumps(allowlist), encoding='utf-8')
    for k, pair in enumerate(cr.DEV_PAIRS):
        fixed, moving = pair.split('__')
        f, m = subject_array(fixed), subject_array(moving)
        if tamper_pair == k:
            f = f.copy()
            f[3, 3, 3] = np.nextafter(f[3, 3, 3], 1.)
        (stage/pair).mkdir()
        np.savez_compressed(stage/pair/'pair.npz', fixed=f, moving=m, initial_u=np.zeros(SHAPE+(3,)), fixed_seg35=np.ones(SHAPE, np.int16),
                            moving_seg35=np.ones(SHAPE, np.int16), fixed_seg4=np.ones(SHAPE, np.int16))
        (stage/pair/'manifest.json').write_text('{"label_sets": "never read"}', encoding='utf-8')
    for subject in cr.DEV_SUBJECTS:
        v = subject_array(subject)
        d = atropos/subject
        d.mkdir()
        for j, p in enumerate(posteriors(v)):
            save_xyz(d/f'posteriors{j+1:02d}.nii.gz', p)
        save_xyz(d/'labels.nii.gz', np.argmax(np.stack(posteriors(v)), 0))
        outputs = {f'posteriors{j:02d}.nii.gz': sha((d/f'posteriors{j:02d}.nii.gz').read_bytes()) for j in (1, 2, 3)}
        if bad_posterior == subject:
            outputs['posteriors03.nii.gz'] = '0'*64
        prov = dict(schema='bayesreg.tissue_provenance.v1', layer='tissue', segmenter_id='atropos', trained_with_labels=False, test_labels_seen=False, oracle=False,
                    class_names=['CSF', 'GM', 'WM'], wm_index=2, posterior_files=[f'posteriors{j:02d}.nii.gz' for j in (1, 2, 3)], posterior_file_index_base=1,
                    output_sha256=outputs, subject=subject, dataset='OASIS',
                    image_source=dict(stage='stage_L2R_val_v1', array_sha256_float64='1'*64 if bad_binding == subject else cr.array_sha256(v)),
                    grid=dict(shape_zyx=list(SHAPE), spacing_xyz=[1., 1., 1.], origin_xyz=ORIGIN_XYZ, unit='mm'))
        if edit_prov is not None:
            prov = edit_prov(subject, prov)
        (d/'provenance.json').write_text(json.dumps(prov), encoding='utf-8')
    (atropos/'dice_vs_truth_seg4.json').write_text('{}', encoding='utf-8')
    prereg = tmp/'prereg.md'
    prereg.write_text('synthetic preregistration\n', encoding='utf-8')
    return dict(stage=stage, atropos_root=atropos, allowlist=tmp/'dev_subjects.json', prereg=prereg, prereg_sha256=sha(prereg.read_bytes()))


def paths(tmp):
    """(OUT, LOCK) in tmp laid out as the real ones: C_ROOT/record and C_ROOT/c_record.lock."""
    return tmp/'croot'/'record', tmp/'croot'/'c_record.lock'


def run(tmp, dev, **kw):
    out, lock = paths(tmp)
    return cr.run(repo=ROOT, out=out, lock=lock, require_clean=False, **dev, **kw)


def test_the_values_are_the_definitions_word_for_word(tmp_path):
    dev = make_dev(tmp_path)
    record = run(tmp_path, dev)
    p_values, means = [], []
    for subject in cr.DEV_SUBJECTS:
        v = subject_array(subject)
        p_values.append(np.percentile(v[v > 0], 99.5))
        means.append(v[posteriors(v)[2] > .5].mean())
    c, w = float(np.median(p_values)), float(np.median(means))
    assert record['c'] == dict(decimal=repr(c), hex=c.hex()) and record['W_dev'] == dict(decimal=repr(w), hex=w.hex())
    for s, p, m in zip(record['subjects'], p_values, means):
        assert s['p99_5']['hex'] == float(p).hex() and s['wm']['mean']['hex'] == float(m).hex() and s['wm']['wm_index'] == 2
    on_disk = json.loads((paths(tmp_path)[0]/cr.RECORD_NAME).read_text(encoding='utf-8'))
    assert on_disk['c'] == record['c'] and on_disk['W_dev'] == record['W_dev'] and on_disk['n_subjects'] == 20
    occurrences = {s['subject']: len(s['occurrences']) for s in record['subjects']}
    assert occurrences[cr.DEV_SUBJECTS[0]] == occurrences[cr.DEV_SUBJECTS[-1]] == 1 and all(occurrences[s] == 2 for s in cr.DEV_SUBJECTS[1:-1])


def test_the_record_lists_every_file_read_with_its_sha256_and_the_spy_saw_only_those(tmp_path):
    dev = make_dev(tmp_path)
    record = run(tmp_path, dev)
    roles = [f['role'] for f in record['files_read']]
    assert roles.count('pair.npz') == 19 and roles.count('atropos provenance') == 20 and roles.count('atropos WM posterior') == 20 and len(roles) == 62
    for f in record['files_read']:
        assert f['sha256'] == sha(Path(f['resolved']).read_bytes()), f
    spy = record['spy']
    assert spy['passed'] and not spy['breaches'] and {m['member'] for m in spy['members']} == {'fixed', 'moving'} and len(spy['members']) == 38
    read = {Path(e['resolved']).name for e in spy['events'] if not e['write'] and not e['resolved'].endswith(cr.CODE_SUFFIXES)}
    assert not {n for n in read if any(part in n.lower() for part in ('label', 'seg', 'dice', 'truth', 'manifest'))}, read
    assert {'posteriors01.nii.gz', 'posteriors02.nii.gz'} & read == set()               # only the WM file of each subject
    assert sorted(spy['data_reads']) == sorted({f['resolved'] for f in record['files_read']})
    assert record['inputs']['members_read'] == ['fixed', 'moving'] and record['inputs']['manifest_read'] is False
    line = json.loads(paths(tmp_path)[1].read_text(encoding='utf-8'))
    assert line == record['lock']['line'] and line['prereg_sha256'] == dev['prereg_sha256'] and len(line['commit']) == 40


def test_the_percentile_as_written_is_the_linear_one():
    x = np.random.default_rng(0).uniform(0, 1, 1001)
    assert np.percentile(x, 99.5) == np.percentile(x, 99.5, method='linear')


def test_a_subject_whose_two_arrays_differ_by_one_ulp_stops_the_run(tmp_path):
    dev = make_dev(tmp_path, tamper_pair=3)                                     # OASIS_OAS1_0441_MR1: moving of pair 2, fixed of pair 3
    with pytest.raises(cr.Stopped, match='OASIS_OAS1_0441_MR1.*bit for bit'):
        run(tmp_path, dev)
    stopped = json.loads((paths(tmp_path)[0]/cr.STOPPED_NAME).read_text(encoding='utf-8'))
    assert 'bit for bit' in stopped['stopped'] and not (paths(tmp_path)[0]/cr.RECORD_NAME).exists() and paths(tmp_path)[1].is_file()
    with pytest.raises(cr.Refused, match='once'):                               # no automatic rerun: the lock refuses a second run
        run(tmp_path, dev)


def test_the_lock_makes_the_record_once_and_a_refusal_reads_nothing(tmp_path, monkeypatch):
    dev = make_dev(tmp_path)
    run(tmp_path, dev)
    before = (paths(tmp_path)[0]/cr.RECORD_NAME).read_bytes()
    monkeypatch.setattr(cr, 'admission', lambda *a, **k: pytest.fail('admission reached'))
    with pytest.raises(cr.Refused, match='once'):
        run(tmp_path, dev)
    assert (paths(tmp_path)[0]/cr.RECORD_NAME).read_bytes() == before
    other_out, other_lock = paths(tmp_path/'second')
    other_lock.parent.mkdir(parents=True)
    other_lock.write_text('{}\n', encoding='utf-8')                             # the lock alone is enough
    with pytest.raises(cr.Refused, match='once'):
        cr.run(repo=ROOT, out=other_out, lock=other_lock, require_clean=False, **dev)
    assert not other_out.exists()


def test_cr_take_lock_is_o_excl(tmp_path):
    cr.take_lock(tmp_path/'l.lock', dict(a=1))
    with pytest.raises(cr.Refused, match='once'):
        cr.take_lock(tmp_path/'l.lock', dict(a=2))
    assert json.loads((tmp_path/'l.lock').read_text(encoding='utf-8')) == dict(a=1)


def test_the_spy_refuses_label_and_evaluation_names_even_when_listed(tmp_path):
    names = ['aligned_seg35.nii.gz', 'S03_label.nii.gz', 'labels.nii.gz', 'dice_vs_truth_seg4.json', 'x_seg4.npz', 'metrics.json', 'evaluation.json', 'truth.json']
    for n in names+['ok.json', 'unlisted.json']:
        (tmp_path/n).write_text('{}', encoding='utf-8')
    spy = cr.Spy(read_exact=[tmp_path/n for n in names+['ok.json']])
    with cr.spying(spy):
        assert (tmp_path/'ok.json').read_text(encoding='utf-8') == '{}'
        for n in names:
            with pytest.raises(cr.Breach, match='label or evaluation file'):
                open(tmp_path/n)
        with pytest.raises(cr.Breach, match='unlisted'):
            open(tmp_path/'unlisted.json')
        with pytest.raises(cr.Breach, match='unlisted'):
            open(tmp_path/'ok.json', 'a')                                       # a listed input is read only
    assert len(spy.breaches) == len(names)+2 and spy.record()['passed'] is False
    assert (tmp_path/'labels.nii.gz').read_text(encoding='utf-8') == '{}'      # removed: nothing watched after the block


def test_the_spy_refuses_every_pair_member_but_fixed_and_moving(tmp_path):
    np.savez_compressed(tmp_path/'pair.npz', fixed=np.zeros(3), moving=np.ones(3), initial_u=np.zeros(3), fixed_seg35=np.ones(3), moving_seg4=np.ones(3))
    spy = cr.Spy(read_exact=[tmp_path/'pair.npz'])
    with cr.spying(spy):
        with np.load(tmp_path/'pair.npz') as npz:
            assert npz['fixed'].sum() == 0 and npz['moving'].sum() == 3
            for key in ('fixed_seg35', 'moving_seg4', 'initial_u'):
                with pytest.raises(cr.Breach, match='member'):
                    npz[key]
    assert [m['member'] for m in spy.members] == ['fixed', 'moving', 'fixed_seg35', 'moving_seg4', 'initial_u'] and len(spy.breaches) == 3
    with np.load(tmp_path/'pair.npz') as npz:                                  # the wrapper is gone after the block
        assert npz['fixed_seg35'].sum() == 3


def test_the_spy_refuses_the_real_roots_and_their_listings(tmp_path):
    spy = cr.Spy()
    for path in ('/home/rmcfz/seq_vp_tag_results/holdout_B_v1/x.json', '/mnt/d/seq_vp_tag_holdout_B_v1/x.npz', str(cr.MAIN_REPO/'data'/'x'), '/mnt/d/data/OASIS_OAS1_0438_MR1/x'):
        assert spy.classify(path, False)[0] in ('forbidden', 'unlisted') and spy.classify(path, False)[0] != 'allowed', path
    assert spy.classify(str(cr.RESULTS/'lpba40_test_v1'/'c_record'/'x.json'), True)[0] == 'forbidden'
    with cr.spying(spy):
        with pytest.raises(cr.Breach, match='listing'):
            spy.see_listing(str(cr.RESULTS))
    assert spy.classify(str(ROOT/'experiments'/'lpba'/'c_record.py'), False) == ('allowed', 'code')


def test_a_seg_member_asked_by_the_reader_stops_the_run(tmp_path, monkeypatch):
    dev = make_dev(tmp_path)

    def greedy(path):
        with np.load(path) as npz:
            return dict(fixed=npz['fixed_seg35'], moving=npz['moving'])
    monkeypatch.setattr(cr, 'read_pair', greedy)
    with pytest.raises(cr.Stopped, match='member'):
        run(tmp_path, dev)
    stopped = json.loads((paths(tmp_path)[0]/cr.STOPPED_NAME).read_text(encoding='utf-8'))
    assert stopped['spy']['breaches'][0]['member'] == 'fixed_seg35' and not (paths(tmp_path)[0]/cr.RECORD_NAME).exists()


def test_a_label_file_opened_by_the_reader_stops_the_run(tmp_path, monkeypatch):
    dev = make_dev(tmp_path)
    real = cr.read_wm

    def peeking(atropos_root, subject, v, spy=None):
        (Path(atropos_root)/subject/'labels.nii.gz').read_bytes()
        return real(atropos_root, subject, v, spy)
    monkeypatch.setattr(cr, 'read_wm', peeking)
    with pytest.raises(cr.Stopped, match='label or evaluation file'):
        run(tmp_path, dev)
    stopped = json.loads((paths(tmp_path)[0]/cr.STOPPED_NAME).read_text(encoding='utf-8'))
    assert stopped['spy']['breaches'][0]['path'].endswith('labels.nii.gz')


@pytest.mark.parametrize('kind', ['bad_binding', 'bad_posterior'])
def test_the_atropos_posterior_must_be_bound_to_the_array_and_hashed(tmp_path, kind):
    dev = make_dev(tmp_path, **{kind: cr.DEV_SUBJECTS[5]})
    with pytest.raises(cr.Stopped, match='array_sha256_float64' if kind == 'bad_binding' else 'sha256 .* is not the provenance'):
        run(tmp_path, dev)
    assert not (paths(tmp_path)[0]/cr.RECORD_NAME).exists()


def test_the_preregistration_must_be_the_pinned_one(tmp_path):
    dev = make_dev(tmp_path)
    dev['prereg_sha256'] = '0'*64
    with pytest.raises(cr.Refused, match='prereg'):
        run(tmp_path, dev)
    assert not paths(tmp_path)[1].exists() and not paths(tmp_path)[0].exists()


def test_the_admission_refuses_another_stage(tmp_path):
    dev = make_dev(tmp_path)
    selection = json.loads((dev['stage']/'selection.json').read_text(encoding='utf-8'))
    selection['pairs'] = selection['pairs'][:-1]
    (dev['stage']/'selection.json').write_text(json.dumps(selection), encoding='utf-8')
    with pytest.raises(cr.Refused, match='sha256'):
        run(tmp_path, dev)
    allowlist = json.loads(dev['allowlist'].read_text(encoding='utf-8'))
    allowlist['sources'][0]['sha256'] = sha((dev['stage']/'selection.json').read_bytes())
    dev['allowlist'].write_text(json.dumps(allowlist), encoding='utf-8')
    with pytest.raises(cr.Refused, match='19 development pairs'):
        run(tmp_path, dev)
    assert not paths(tmp_path)[1].exists()


def test_the_dry_run_lists_the_files_and_reads_no_array(tmp_path):
    dev = make_dev(tmp_path)
    out = tmp_path/'dry'
    document = cr.dry_run(repo=ROOT, out=out, lock=paths(tmp_path)[1], record_dir=paths(tmp_path)[0], **dev)
    assert document['ready'] and document['will_read']['data_files'] == 59 and document['exists'] == dict(pairs=19, provenance=20, subjects_with_three_posteriors=20)
    reads = {Path(r).name for r in document['spy']['data_reads']}
    assert reads == {'dev_subjects.json', 'selection.json', 'prereg.md'} and document['spy']['members'] == []
    assert not any(e['resolved'].endswith(('.npz', '.nii.gz', 'provenance.json')) for e in document['spy']['events'])
    assert json.loads((out/cr.DRY_NAME).read_text(encoding='utf-8'))['ready'] is True
    assert not (tmp_path/'croot').exists()
    with pytest.raises(cr.Refused, match='out'):
        cr.dry_run(repo=ROOT, out=cr.RESULTS/'lpba40_test_v1'/'dry', **dev)


def test_the_real_roots_are_refused_while_pytest_runs(tmp_path, monkeypatch):
    """The real OUT and LOCK, refused by sandbox_problem; two more guards behind it (diff review: no --run behind one check only): a wrong preregistration pin
    refuses before any read, and the lock is never taken."""
    dev = make_dev(tmp_path)
    dev['prereg_sha256'] = '0'*64
    monkeypatch.setattr(cr, 'take_lock', lambda *a, **k: pytest.fail('the lock was reached'))
    with pytest.raises(cr.Refused, match='sandbox'):
        cr.run(repo=ROOT, require_clean=False, **dev)                             # OUT and LOCK, the real ones
    assert cr.C_ROOT == Path('/home/rmcfz/seq_vp_tag_results/lpba40_c_record_v1')                 # the lead's ruling 11-19 ~1:01 PM CST (2): its own root
    assert cr.OUT == cr.C_ROOT/'record' and cr.LOCK == cr.C_ROOT/'c_record.lock' and 'lpba40_test_v1' not in str(cr.OUT)+str(cr.LOCK)


def test_main_needs_exactly_one_mode(monkeypatch):
    monkeypatch.setattr(cr, 'run', lambda **kw: pytest.fail('run reached'))
    monkeypatch.setattr(cr, 'dry_run', lambda **kw: pytest.fail('dry_run reached'))
    with pytest.raises(SystemExit) as caught:
        cr.main([])
    assert caught.value.code == 2
    with pytest.raises(SystemExit):
        cr.main(['--dry-run', '--run'])
    with pytest.raises(SystemExit):
        cr.main(['--run', '--out', '/tmp/x'])


def _assigned(path, name):
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and [getattr(t, 'id', None) for t in node.targets] == [name]:
            return ast.unparse(node.value)
    raise AssertionError(f'{path}: no {name}')


def test_the_constants_are_the_development_sources_the_method_uses():
    """ATROPOS_ROOT is HE's development atropos_root (run_hx.default_paths: mc0p_core.SEG_ROOT); the 20 subjects and the selection path are the allowlist source's; the 19 pairs
    are the evaluation entry's DEV_PAIRS; the preregistration pin is the file's."""
    assert _assigned(ROOT/'experiments'/'mc0'/'mc0p_core.py', 'SEG_ROOT') == f"Path('{cr.ATROPOS_ROOT}')"
    assert 'atropos_root=str(mc.SEG_ROOT)' in (ROOT/'experiments'/'he'/'run_hx.py').read_text(encoding='utf-8')
    allowlist = json.loads((ROOT/cr.ALLOWLIST_REL).read_text(encoding='utf-8'))
    source = next(s for s in allowlist['sources'] if s['name'] == cr.DEV_STAGE_NAME)
    assert source['subjects'] == list(cr.DEV_SUBJECTS) and source['path'] == cr.SELECTION_REL and source['never_for_training'] is False
    from bayesreg.eval import holdout
    assert cr.DEV_PAIRS == tuple(f'{f}__{m}' for f, m in holdout.DEV_PAIRS) and cr.DEV_STAGE_NAME == holdout.DEV_STAGE
    assert sha((ROOT/cr.PREREG_REL).read_bytes()) == cr.PREREG_SHA256
    assert cr.array_sha256(np.arange(6.).reshape(2, 3)[:, ::-1]) == sha(np.ascontiguousarray(np.arange(6.).reshape(2, 3)[:, ::-1]).tobytes())


# ---- LP-T0 revision (review 0/2/5, the lead's rulings of 2026-11-19 ~1:01 PM CST) ------------------------------------------------------------------
def stopped_record(tmp):
    out, lock = paths(tmp)
    found = [q for q in (out/cr.STOPPED_NAME, lock.parent/cr.STOPPED_NAME) if q.is_file()]
    assert len(found) == 1, found
    return found[0], json.loads(found[0].read_text(encoding='utf-8'))


def test_a_corrupt_pair_npz_after_the_lock_leaves_a_stopped_record(tmp_path):
    """Review important 1: an exception no reader names (zipfile.BadZipFile here) still stops the run with its record."""
    dev = make_dev(tmp_path)
    npz = dev['stage']/cr.DEV_PAIRS[4]/'pair.npz'
    npz.write_bytes(npz.read_bytes()[:200])                                     # the zip magic kept, the archive cut: np.load opens it as an npz
    with pytest.raises(cr.Stopped, match='BadZipFile'):
        run(tmp_path, dev)
    where, stopped = stopped_record(tmp_path)
    assert where == paths(tmp_path)[0]/cr.STOPPED_NAME and stopped['kind'] == 'BadZipFile' and stopped['out_made'] is True and 'Traceback' in stopped['traceback']
    assert not (paths(tmp_path)[0]/cr.RECORD_NAME).exists() and paths(tmp_path)[1].is_file()


def test_a_provenance_that_is_a_list_stops_the_run(tmp_path):
    dev = make_dev(tmp_path, edit_prov=lambda subject, prov: [prov] if subject == cr.DEV_SUBJECTS[7] else prov)
    with pytest.raises(cr.Stopped, match='JSON object'):
        run(tmp_path, dev)
    assert stopped_record(tmp_path)[1]['kind'] == 'Stopped'


def test_any_exception_after_the_lock_is_a_stop(tmp_path, monkeypatch):
    dev = make_dev(tmp_path)
    monkeypatch.setattr(cr, 'compute', lambda *a, **k: (_ for _ in ()).throw(AttributeError('an unexpected attribute')))
    with pytest.raises(cr.Stopped, match='AttributeError'):
        run(tmp_path, dev)
    assert stopped_record(tmp_path)[1]['kind'] == 'AttributeError'


def test_a_base_exception_is_recorded_then_raised_as_it_was(tmp_path, monkeypatch):
    dev = make_dev(tmp_path)
    monkeypatch.setattr(cr, 'compute', lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, dev)
    assert stopped_record(tmp_path)[1]['kind'] == 'KeyboardInterrupt'


def test_when_out_cannot_be_made_the_stopped_record_lies_beside_the_lock(tmp_path):
    dev = make_dev(tmp_path)
    lock = tmp_path/'croot'/'c_record.lock'
    (tmp_path/'croot').mkdir()
    (tmp_path/'croot'/'blocker').write_text('a file, not a directory\n', encoding='utf-8')
    out = tmp_path/'croot'/'blocker'/'record'
    with pytest.raises(cr.Stopped):
        cr.run(repo=ROOT, out=out, lock=lock, require_clean=False, **dev)
    stopped = json.loads((lock.parent/cr.STOPPED_NAME).read_text(encoding='utf-8'))
    assert stopped['out_made'] is False and stopped['kind'] in ('NotADirectoryError', 'FileExistsError') and lock.is_file()


def _set(key, value):
    def edit(subject, prov):
        if subject == cr.DEV_SUBJECTS[2]:
            target = prov
            for k in key[:-1]:
                target = target[k]
            target[key[-1]] = value
        return prov
    return edit


PROVENANCE_CASES = [
    ('subject', _set(['subject'], 'OASIS_OAS1_0001_MR1'), 'subject'),
    ('segmenter', _set(['segmenter_id'], 'fast'), 'segmenter'),
    ('layer', _set(['layer'], 'structure'), 'layer'),
    ('test_labels_seen', _set(['test_labels_seen'], True), 'test_labels_seen'),
    ('trained_with_labels', _set(['trained_with_labels'], True), 'trained_with_labels'),
    ('oracle', _set(['oracle'], True), 'oracle'),
    ('no WM', _set(['class_names'], ['CSF', 'GM', 'white']), 'class_names'),
    ('wm_index', _set(['wm_index'], 1), 'class_names'),
    ('posterior file', _set(['posterior_files'], ['posteriors01.nii.gz', 'posteriors02.nii.gz', 'posteriors04.nii.gz']), 'posterior_files'),
    ('index base', _set(['posterior_file_index_base'], 0), 'posterior_files'),
    ('no sha256', _set(['output_sha256', 'posteriors03.nii.gz'], 'xyz'), 'posterior_files'),
    ('stage', _set(['image_source', 'stage'], 'stage_T0_tune_v1'), 'image_source'),
    ('grid shape', _set(['grid', 'shape_zyx'], [6, 7, 6]), 'grid shape_zyx'),
    ('grid spacing', _set(['grid', 'spacing_xyz'], [1., 1., 2.]), "not on the array's grid"),
    ('grid origin', _set(['grid', 'origin_xyz'], [0., 0., 0.]), "not on the array's grid"),
]


@pytest.mark.parametrize('name,edit,match', PROVENANCE_CASES, ids=[c[0] for c in PROVENANCE_CASES])
def test_every_field_of_the_provenance_is_checked(tmp_path, name, edit, match):
    """Review minor 3: each field read_wm checks, changed on one subject, stops the run before its value is used."""
    dev = make_dev(tmp_path, edit_prov=edit)
    with pytest.raises(cr.Stopped, match=match):
        run(tmp_path, dev)
    assert not (paths(tmp_path)[0]/cr.RECORD_NAME).exists()


def test_a_breach_before_the_lock_is_a_refusal(tmp_path, monkeypatch):
    """Review minor 1: nothing of a subject read, nothing written: exit 8, not 9."""
    dev = make_dev(tmp_path)
    real = cr.admission

    def peeking(allowlist, stage):
        (Path(stage)/cr.DEV_PAIRS[0]/'manifest.json').read_bytes()
        return real(allowlist, stage)
    monkeypatch.setattr(cr, 'admission', peeking)
    with pytest.raises(cr.Refused, match='before the lock') as caught:
        run(tmp_path, dev)
    assert caught.value.stage == 'spy' and not (tmp_path/'croot').exists()
    with pytest.raises(cr.Refused, match='the dry run'):
        cr.dry_run(repo=ROOT, lock=paths(tmp_path)[1], record_dir=paths(tmp_path)[0], **dev)


@pytest.mark.parametrize('raised,code', [(cr.Refused('once', 'x'), 8), (cr.Stopped('x'), 9), (cr.Breach('x'), 9)])
def test_main_exit_codes(monkeypatch, raised, code):
    """Review minor 3: main maps a refusal to 8, a stop to 9 (the real roots are never reached: run and dry_run are replaced)."""
    def raise_it(**kw):
        raise raised
    monkeypatch.setattr(cr, 'run', raise_it)
    monkeypatch.setattr(cr, 'dry_run', raise_it)
    monkeypatch.setattr(cr, 'bytecode_problem', lambda flags=None: None)          # diff review minor 4: not dependent on running under -B
    assert cr.main(['--run']) == code and cr.main(['--dry-run']) == code


def test_main_refuses_when_python_writes_bytecode(monkeypatch, tmp_path):
    """Review minor 2: without -B (or PYTHONDONTWRITEBYTECODE) the entry refuses with 8 before anything; the module never writes bytecode itself."""
    from types import SimpleNamespace
    assert cr.bytecode_problem(SimpleNamespace(dont_write_bytecode=False)) and cr.bytecode_problem(SimpleNamespace(dont_write_bytecode=True)) is None
    monkeypatch.setattr(cr, 'run', lambda **kw: pytest.fail('run reached'))
    monkeypatch.setattr(cr, 'bytecode_problem', lambda flags=None: 'bytecode')
    assert cr.main(['--run']) == 8
    import subprocess
    cache = ROOT/'experiments'/'lpba'/'__pycache__'
    before = cache.exists()
    # diff review (important): the child runs --dry-run (it writes nothing without --out), with PYTEST_CURRENT_TEST so that a real root would be refused even if the
    # check under test fell through
    done = subprocess.run([sys.executable, '-I', str(ROOT/'experiments'/'lpba'/'c_record.py'), '--dry-run'], capture_output=True, text=True, timeout=120,
                          env=dict(PATH='/usr/bin:/bin', HOME=str(tmp_path), PYTEST_CURRENT_TEST='1'))
    assert done.returncode == 8 and 'refused (bytecode)' in done.stderr, (done.returncode, done.stderr[-500:])
    assert cache.exists() == before


# ---- LP-T0 revision 2 (diff review 0/1/5) ---------------------------------------------------------------------------------------------------------
def test_a_record_whose_spy_did_not_pass_is_kept_in_the_stopped_record(tmp_path, monkeypatch):
    """Diff review minor 2: an expected input never opened -> the spy does not pass -> the stopped record carries the whole record (its spy with the expected reads)."""
    dev = make_dev(tmp_path)
    real = cr.compute
    ghost = tmp_path/'never_opened.json'
    ghost.write_text('{}', encoding='utf-8')

    def with_ghost(*a, **k):
        subjects, files, c, w = real(*a, **k)
        return subjects, files+[dict(role='ghost', path=str(ghost), resolved=str(ghost), sha256='0'*64)], c, w
    monkeypatch.setattr(cr, 'compute', with_ghost)
    with pytest.raises(cr.Stopped, match='spy did not pass'):
        run(tmp_path, dev)
    stopped = stopped_record(tmp_path)[1]
    assert stopped['record']['spy']['passed'] is False and stopped['record']['spy']['expected_reads'][str(ghost.resolve())] == 0
    assert stopped['record']['c'] and not (paths(tmp_path)[0]/cr.RECORD_NAME).exists()


@pytest.mark.parametrize('error', [KeyError('pairs'), ValueError('bad JSON'), AttributeError('x')])
def test_any_failure_before_the_lock_is_a_refusal(tmp_path, monkeypatch, error):
    """Diff review minor 3: before the lock any exception is Refused (exit 8), nothing written."""
    dev = make_dev(tmp_path)

    def failing(*a, **k):
        raise error
    monkeypatch.setattr(cr, 'admission', failing)
    with pytest.raises(cr.Refused, match='before the lock') as caught:
        run(tmp_path, dev)
    assert caught.value.stage == 'input' and not (tmp_path/'croot').exists()
    with pytest.raises(cr.Refused, match='the dry run'):
        cr.dry_run(repo=ROOT, lock=paths(tmp_path)[1], record_dir=paths(tmp_path)[0], **dev)


def test_a_lock_that_cannot_be_made_is_a_refusal(tmp_path, monkeypatch):
    dev = make_dev(tmp_path)

    def denied(path, line):
        raise PermissionError(f'{path}: denied')
    monkeypatch.setattr(cr, 'take_lock', denied)
    with pytest.raises(cr.Refused, match='lock could not be taken') as caught:
        run(tmp_path, dev)
    assert caught.value.stage == 'lock' and not paths(tmp_path)[0].exists() and not paths(tmp_path)[1].exists()
