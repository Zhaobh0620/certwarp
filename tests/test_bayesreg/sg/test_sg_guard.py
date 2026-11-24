"""C23.9-1, -2, -3, -4, -7, -8, -10, -15, -16, -17 (SG): the sealed guard of bayesreg (check_subject behind check_pair_subjects and check_subject_admitted,
DevAllowlist.allows, OasisFiles.image_path / label_path, load_pair and the three readers that take a PairInput).  Ids only and synthetic files in tmp_path; the
real sealed lists are read for their ids (C23.9-1); no image or label of any real subject is opened."""
import dataclasses
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

import bayesreg.io.pair as pair_module
import bayesreg.io.sealed as S
from bayesreg.eval import truth
from bayesreg.io import unseal as U
from bayesreg.io.guard import DevAllowlist, SubjectSealed, check_pair_subjects, check_subject, load_dev_allowlist, refuse_sealed
from bayesreg.io.pair import PairInput, ProvenanceRejected, load_pair, load_segmenter_registry, subject_list_sha256, validate_segmentation_provenance
from bayesreg.seg.unet import synthetic
from bayesreg.seg.unet.data import LabelAccessRefused, LabelGuard, OasisFiles, load_folds
from bayesreg.seg.unet.infer import ModelSelectionRefused, check_subject_admitted, select_fold_models

from . import _sg_world as W


def real_ids(n=3):
    """A few ids of each real OASIS sealed list, read at run time (ids only; review EV item 12: never written into the test source)."""
    lists = S.load_sealed_lists()
    return {name: sorted(s for s in lists.subjects[name] if s.startswith('OASIS_'))[:n] for name in ('holdout_B_v1', 'H1', 'H4')}


def write_registry(path, contact, segmenter_id='seg_sg_test'):
    document = dict(schema=pair_module.REGISTRY_SCHEMA, models={
        f'{segmenter_id}_f{fold}': dict(segmenter_id=segmenter_id, fold=fold, member_model_sha256=[f'{fold}'*64], label_contact_subjects=list(ids),
                                        label_contact_subjects_sha256=subject_list_sha256(list(ids))) for fold, ids in enumerate(contact)})
    return load_segmenter_registry(W.write_json(path, document))


def write_pair(directory, fixed, moving, *, image=True):
    """manifest.json (ids, grid, hashes) and, with image, a pair.npz of zeros; synthetic bytes only."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shape = (5, 6, 7)
    if image:
        np.savez_compressed(directory/'pair.npz', fixed=np.zeros(shape), moving=np.zeros(shape))
    manifest = dict(fixed_subject=fixed, moving_subject=moving, shape_zyx=list(shape), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm',
                    pair_sha256=W.file_sha256(directory/'pair.npz') if image else 'f'*64, sources=dict(fixed=dict(sha256='1'*64), moving=dict(sha256='2'*64)))
    W.write_json(directory/'manifest.json', manifest)
    return directory/'pair.npz'


# ---- C23.9-1: a wrongly allowlisted sealed subject is refused everywhere -----------------------------------------------------------------------------
@pytest.mark.parametrize('name', ['holdout_B_v1', 'H1', 'H4'])
def test_a_wrongly_allowlisted_sealed_subject_is_refused_without_an_unseal_object(tmp_path, monkeypatch, name):
    ids = real_ids(5)[name]
    allowlist = load_dev_allowlist(W.write_json(tmp_path/'mistaken.json', synthetic.allowlist_document([('dev', W.DEV, False), ('mistaken', ids, False)])))
    for subject in ids:
        assert allowlist.lists(subject) and not allowlist.allows(subject)
        with pytest.raises(SubjectSealed, match='without an unseal record'):
            check_subject_admitted(subject, allowlist)
        with pytest.raises(SubjectSealed, match='moving: .*without an unseal record'):
            check_pair_subjects(W.DEV[0], subject, allowlist)
    # load_pair: the manifest holds only ids; pair.npz is neither hashed nor opened
    calls = []
    monkeypatch.setattr(pair_module, 'file_sha256', lambda *a, **k: calls.append(a) or '0'*64)
    monkeypatch.setattr(pair_module, '_open_npz', lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError('opened')))
    with pytest.raises(SubjectSealed):
        load_pair(write_pair(tmp_path/'pair', W.DEV[0], ids[0], image=False), allowlist=allowlist)
    assert calls == []
    monkeypatch.undo()
    # segment_subject: no image read, no directory made
    import bayesreg.seg.unet.infer as infer
    reads = []
    monkeypatch.setattr(infer, 'read_image', lambda *a, **k: reads.append(a))
    from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping
    registry = write_registry(tmp_path/'registry.json', [W.TRAIN, W.TRAIN])
    with pytest.raises(SubjectSealed):
        infer.segment_subject(ids[0], files=OasisFiles(root=tmp_path/'data'), registry=registry, allowlist=allowlist, models_root=tmp_path, grouping=load_grouping(DEFAULT_GROUPS_PATH, 'structure13_v1'),
                              pred_root=tmp_path/'pred', device='cpu', log=lambda *a: None)
    assert reads == [] and not (tmp_path/'pred').exists()
    # label contact and training: select_fold_models, LabelGuard, admit_eb_pairs, validate_segmentation_provenance, a4_admission
    with pytest.raises(ModelSelectionRefused, match='off the allowlist'):
        select_fold_models(write_registry(tmp_path/'contact.json', [ids, ids]), W.DEV[0], allowlist, segmenter_id='seg_sg_test')
    folds = W.write_json(tmp_path/'folds.json', synthetic.folds_document(ids[:2], ids[2:3], ids[3:4], ids[4:5]))
    split = load_folds(folds)
    with pytest.raises(LabelAccessRefused, match='not on the development allowlist'):
        LabelGuard(split, allowlist, training_source='mistaken')
    from bayesreg.learn.population import EBPairRefused, admit_eb_pairs
    eb = W.write_text(tmp_path/'eb_pairs.tsv', f'fixed\tmoving\tfold\n{ids[0]}\t{ids[1]}\tA\n')
    with pytest.raises(EBPairRefused, match='not on the development allowlist'):
        admit_eb_pairs(eb_pairs_path=eb, split=split, allowlist=allowlist, training_source='mistaken')
    dev_pair = load_pair(write_pair(tmp_path/'dev_pair', W.DEV[0], W.DEV[1]), allowlist=allowlist)
    prov = dict(layer='structure', segmenter_id='seg_sg_test', model_ids=['seg_sg_test_f0'], model_sha256={'seg_sg_test_f0': ['0'*64]}, trained_with_labels=True, test_labels_seen=False,
                class_names=['a', 'b'], input_sha256={'aligned_norm.nii.gz': '1'*64}, output_sha256={'g.npz': '3'*64})
    with pytest.raises(ProvenanceRejected, match='off the allowlist'):
        validate_segmentation_provenance(prov, dev_pair, side='fixed', layer='structure', registry=write_registry(tmp_path/'contact2.json', [ids, ids]), allowlist=allowlist)
    sys.path.insert(0, str(W.ROOT/'experiments/d0'))
    try:
        import make_inputs as MI
    finally:
        sys.path.remove(str(W.ROOT/'experiments/d0'))
    unet = MI.UNetSource(folds=folds, folds_sha256=split.sha256)
    with pytest.raises(SystemExit, match='not on the development allowlist'):
        MI.a4_admission([ids[0]], tmp_path/'mistaken.json', unet)


# ---- C23.9-2, -3: a valid unseal object admits exactly its subjects -----------------------------------------------------------------------------------
@pytest.fixture
def w(tmp_path, monkeypatch):
    return W.make_world(tmp_path, monkeypatch, sources=W.sources_default([('also_h1', W.H1+W.H4, False)]))


def test_a_valid_unseal_object_admits_its_subjects_through_both_guards(w):
    unseal, bound = W.bound(w)
    for subject in W.HOLDOUT:
        check_subject_admitted(subject, bound)
        verdict = check_subject(subject, bound)
        assert verdict.admitted and verdict.admitted_by_unseal and verdict.sealed_lists == ('holdout_B_v1',)
        with pytest.raises(SubjectSealed, match='without an unseal record'):
            check_subject_admitted(subject, w.allowlist)                       # the same allowlist unbound
    record = check_pair_subjects(W.HOLDOUT[0], W.DEV[0], bound)
    assert record['unsealed'] is True and record['sealed_guard']['fixed'] == dict(sealed_list='holdout_B_v1', admitted_by_unseal=True)
    assert record['sealed_guard']['moving'] == dict(sealed_list=None, admitted_by_unseal=False) and record['sealed_guard']['version'] == 'sealed-guard-v1'
    assert record['sealed_guard']['unseal'] == unseal.record() and record['sealed_guard']['unseal']['sha256'] == unseal.sha256
    assert [r['name'] for r in record['sealed_guard']['sealed_lists']] == ['holdout_B_v1', 'H1', 'H4', 'H3', 'H2']
    assert record['allowlist_sha256'] == w.allowlist.sha256 and not bound.allows(W.HOLDOUT[0])       # D96: never for training, bound or not
    plain = check_pair_subjects(W.DEV[0], W.DEV[1], bound)                    # the allowlist carries the object: the record says so (D102 (b))
    assert plain['unsealed'] is False and plain['sealed_guard']['fixed'] == dict(sealed_list=None, admitted_by_unseal=False)
    assert 'sealed_guard' not in check_pair_subjects(W.DEV[0], W.DEV[1], w.allowlist)


def test_a_subject_not_in_the_record_or_on_another_list_is_refused(w, monkeypatch):
    unseal, bound = W.bound(w)
    for subject in W.H1+W.H4:                                                  # on the (bound) allowlist, on a paper-1 list: refused whatever the record
        verdict = check_subject(subject, bound)
        assert verdict.step == 'sealed' and verdict.item == 'H1-H4', verdict
        with pytest.raises(SubjectSealed, match='of paper 1'):
            check_pair_subjects(subject, W.DEV[0], bound)
    # the list grows after the record was loaded: the new subject is on holdout_B_v1 but not one of the record's
    extra = W.oasis(963)
    W.write_text(w.repo/'lists/b.txt', ''.join(f'{s}\n' for s in W.HOLDOUT+[extra]))
    specs = (dict(w.specs[0], sha256=W.file_sha256(w.repo/'lists/b.txt'), count=4),)+w.specs[1:]
    W.pin(monkeypatch, w.repo, specs)
    other = dataclasses.replace(w.allowlist, subjects={'OASIS': w.allowlist.subjects['OASIS'] | {extra}},
                                sources=w.allowlist.sources+({'name': 'late', 'dataset': 'OASIS', 'path': 'x', 'sha256': '9'*64, 'never_for_training': True, 'subjects': (extra,)},))
    bound_other = dataclasses.replace(other, unseal=unseal)
    U._BOUND[id(bound_other)] = bound_other               # white box: as if bind_unseal had bound an allowlist of the same sha256 with another content
    verdict = check_subject(extra, bound_other)
    assert verdict.step == 'sealed' and verdict.item == 'not in the record', verdict
    # a subject of the record held by another source too (the same white box)
    shared = dataclasses.replace(w.allowlist, sources=w.allowlist.sources+({'name': 'dev2', 'dataset': 'OASIS', 'path': 'y', 'sha256': '8'*64, 'never_for_training': False,
                                                                             'subjects': (W.HOLDOUT[0],)},), unseal=unseal)
    U._BOUND[id(shared)] = shared
    verdict = check_subject(W.HOLDOUT[0], shared)
    assert verdict.step == 'sealed' and verdict.item == 'allowlist source', verdict


def test_h2_and_h3_are_refused_whatever_the_form(w):
    unseal, bound = W.bound(w)
    for subject in ('S37', 'S38', 'S39', 'S40'):
        assert check_subject(subject, bound).step == 'allowlist'              # never on an allowlist (guard.NEVER_ALLOWLISTED)
        with pytest.raises(SubjectSealed, match=r"\['H2'\]"):
            refuse_sealed(subject, 'test')
    for subject in W.H3+['ibsr_03', 'OASIS_IBSR_01']:
        assert check_subject(subject, bound).step == 'canonical'              # no canonical IBSR id
        with pytest.raises(SubjectSealed, match=r"\['H3'\]"):
            refuse_sealed(subject, 'test')
    from bayesreg.io.guard import NEVER_ALLOWLISTED
    assert S.SEALED_LPBA40 is truth.SEALED_LPBA40 and S.SEALED_LPBA40 == NEVER_ALLOWLISTED['LPBA40']


def test_d100_a_bad_unseal_object_refuses_a_subject_on_no_list(w):
    unseal, bound = W.bound(w)
    assert check_subject(W.DEV[0], bound).admitted
    w.path.write_bytes(w.path.read_bytes()+b'\n')
    for subject in (W.DEV[0], W.HOLDOUT[0]):
        verdict = check_subject(subject, bound)
        assert verdict.step == 'sealed' and verdict.item == 'unseal object'
        with pytest.raises(SubjectSealed, match='D100'):
            check_pair_subjects(W.DEV[1], subject, bound)


def test_the_unseal_token_still_admits_nobody_and_the_unseal_object_decides(w):
    """C23.9-8 (D101): the token is recorded only; with a token and the unseal object, the object alone decides."""
    unseal, bound = W.bound(w)
    with pytest.raises(SubjectSealed, match='without an unseal record.*unsealing is not implemented'):
        check_pair_subjects(W.HOLDOUT[0], W.DEV[0], w.allowlist, unseal_token='a token')
    record = check_pair_subjects(W.HOLDOUT[0], W.DEV[0], bound, unseal_token='a token')
    assert record['unsealed'] is True and record['unseal_token_presented'] is True and record['unseal_token_sha256']
    with pytest.raises(SubjectSealed, match='of paper 1'):
        check_pair_subjects(W.H1[0], W.DEV[0], bound, unseal_token='a token')


# ---- C23.9-10: the lists --------------------------------------------------------------------------------------------------------------------------------
def test_the_real_lists_are_read_with_their_pins_and_truth_reexports_the_same_objects():
    lists = S.load_sealed_lists()
    assert {r['name']: r['subjects'] for r in lists.record} == {'holdout_B_v1': 100, 'H1': 100, 'H4': 40, 'H3': 18, 'H2': 4}
    for spec in S.SEALED_LISTS:
        assert W.file_sha256(S.REPO/spec['path']) == spec['sha256']
        assert subprocess.run(['git', '-C', str(W.ROOT), 'ls-files', '--error-unmatch', spec['path']], capture_output=True).returncode == 0
    for name in ('SEALED_LISTS', 'SEALED_LPBA40', 'SealedListError', 'SealedLists', 'load_sealed_lists'):
        assert getattr(truth, name) is getattr(S, name), name
    assert S.REPO == truth.REPO and S.guard_sealed_lists().subjects == lists.subjects
    assert not set(lists.subjects['holdout_B_v1']) & set(lists.subjects['H1']) and not set(lists.subjects['holdout_B_v1']) & set(lists.subjects['H4'])


@pytest.mark.parametrize('how', ['missing', 'one_byte'])
def test_a_missing_or_changed_list_refuses_every_subject(tmp_path, monkeypatch, how):
    repo = tmp_path/'tree'
    for spec in W.REAL['SEALED_LISTS']:
        target = repo/spec['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((W.ROOT/spec['path']).read_bytes())
    victim = repo/W.REAL['SEALED_LISTS'][2]['path']
    victim.unlink() if how == 'missing' else victim.write_bytes(victim.read_bytes()+b' ')
    W.pin(monkeypatch, repo, W.REAL['SEALED_LISTS'])
    allowlist = load_dev_allowlist(W.ROOT/'configs/bayesreg/dev_subjects.json')
    with pytest.raises(S.SealedListError):
        check_pair_subjects('OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0439_MR1', allowlist)
    with pytest.raises(S.SealedListError):
        check_subject_admitted('OASIS_OAS1_0438_MR1', allowlist)
    with pytest.raises(S.SealedListError):
        allowlist.allows('OASIS_OAS1_0438_MR1')
    with pytest.raises(SubjectSealed, match='not on the development allowlist'):
        check_pair_subjects('OASIS_OAS1_9999_MR1', 'OASIS_OAS1_0439_MR1', allowlist)        # steps 1-2 first, unchanged: no list is needed to refuse
    assert allowlist.allows('OASIS_OAS1_9999_MR1') is False


def test_the_cache_is_keyed_by_the_root_and_the_pins(tmp_path, monkeypatch):
    first = S.guard_sealed_lists()
    assert S.guard_sealed_lists() is first                                    # read once, then cached
    world = W.make_world(tmp_path, monkeypatch)                               # other pins (and the cache reset, as the tests do)
    synthetic_lists = S.guard_sealed_lists()
    assert synthetic_lists.subjects['holdout_B_v1'] == frozenset(W.HOLDOUT) and S.guard_sealed_lists() is synthetic_lists
    monkeypatch.setattr(S, 'REPO', W.REAL['REPO'])
    monkeypatch.setattr(S, 'SEALED_LISTS', W.REAL['SEALED_LISTS'])
    assert S.guard_sealed_lists().subjects == first.subjects and S.guard_sealed_lists() is not synthetic_lists and world.repo.is_dir()
    assert not [name for name in vars(S) if name.startswith(('set_', 'put_', 'write_'))]       # no public way to write the cache


def test_the_cached_lists_are_read_only_and_each_record_gets_a_copy(tmp_path, monkeypatch):
    """Review m4: the guard's cached copy cannot be changed through its fields, and the record a guard verdict carries is a copy of it."""
    from bayesreg.io import unseal as U
    w = W.make_world(tmp_path, monkeypatch)
    lists = S.guard_sealed_lists()
    with pytest.raises(TypeError):
        lists.subjects['holdout_B_v1'] = frozenset()
    with pytest.raises(TypeError):
        lists.record[0]['name'] = 'changed'
    with pytest.raises(AttributeError):
        lists.record.append({})
    assert isinstance(lists.subjects['holdout_B_v1'], frozenset) and lists.which(W.HOLDOUT[0]) == 'holdout_B_v1'
    _, bound = W.bound(w)
    record = check_pair_subjects(W.HOLDOUT[0], W.DEV[0], bound)
    copied = record['sealed_guard']['sealed_lists']
    assert copied == [dict(r) for r in lists.record] and all(type(r) is dict for r in copied)
    copied[0]['name'] = 'changed'
    assert S.guard_sealed_lists() is lists and lists.record[0]['name'] == 'holdout_B_v1'
    assert U.bind_unseal is not None


def test_the_read_only_cache_query(tmp_path, monkeypatch):
    """C23.2-3 (revision 6g): sealed_cache_is_empty tells whether a copy is cached; it reads no file (the pinned root may not even exist) and changes nothing."""
    monkeypatch.setattr(S, 'REPO', tmp_path/'no_such_tree')
    S._reset_cache()
    assert S.sealed_cache_is_empty() is True and S.sealed_cache_is_empty() is True and S._CACHE == {}       # nothing read, nothing cached, no SealedListError
    world = W.make_world(tmp_path/'w', monkeypatch)                                                       # other pins (and the cache reset)
    assert S.sealed_cache_is_empty() is True
    lists = S.guard_sealed_lists()
    before = dict(S._CACHE)
    assert S.sealed_cache_is_empty() is False and S._CACHE == before and S.guard_sealed_lists() is lists and world.repo.is_dir()
    S._reset_cache()
    assert S.sealed_cache_is_empty() is True and 'sealed_cache_is_empty' in S.__all__


# ---- C23.9-15: snapshots do not carry the lists; the guard there refuses ---------------------------------------------------------------------------
def test_in_a_tree_without_the_lists_the_guard_refuses_and_reads_nothing(tmp_path):
    """D95 (a'): a copy of bayesreg, jd_curl and configs/bayesreg as a snapshot holds them (no splits/, no archive/): the new guard raises SealedListError, before any
    file of a subject; snapshot.py's "the hold-out lists of the same directory are not copied" stands."""
    snap = tmp_path/'snapshot'
    for part in ('bayesreg', 'jd_curl', 'configs/bayesreg'):
        shutil.copytree(W.ROOT/part, snap/part, ignore=shutil.ignore_patterns('__pycache__'))
    code = ('import sys\nfrom bayesreg.io.guard import check_pair_subjects, load_dev_allowlist\nfrom bayesreg.io.sealed import SealedListError, REPO\n'
            'assert str(REPO).startswith(sys.argv[1]), REPO\n'
            'try:\n    check_pair_subjects("OASIS_OAS1_0438_MR1", "OASIS_OAS1_0439_MR1", load_dev_allowlist(sys.argv[1]+"/configs/bayesreg/dev_subjects.json"))\n'
            'except SealedListError as exc:\n    print("SealedListError", exc)\nelse:\n    print("ADMITTED")\n')
    done = subprocess.run([sys.executable, '-B', '-c', code, str(snap)], cwd=tmp_path, env=dict(W.GIT_ENV, PYTHONPATH=str(snap)), capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and done.stdout.startswith('SealedListError') and 'missing' in done.stdout, done.stdout+done.stderr
    assert 'are not copied' in (W.ROOT/'bayesreg/seg/unet/snapshot.py').read_text(encoding='utf-8')


# ---- C23.9-16: the readers that take a PairInput recheck the pair and its bound allowlist by identity ---------------------------------------------------
def tissue_provenance(pair, side):
    return dict(layer='tissue', segmenter_id='atropos', model_ids=['ANTs-2.6.5'], trained_with_labels=False, test_labels_seen=False, class_names=['CSF', 'GM', 'WM'], wm_index=2,
                input_sha256={'aligned_norm.nii.gz': pair.source_sha256[side]}, output_sha256={'posteriors01.nii.gz': '4'*64})


@pytest.fixture
def sealed_pair(w):
    unseal, bound = W.bound(w)
    pair = load_pair(write_pair(w.tmp/'sealed_pair', W.HOLDOUT[0], W.DEV[0]), allowlist=bound)
    return pair, bound, unseal


def test_the_pair_is_registered_with_its_bound_allowlist_and_the_readers_take_only_that_object(w, sealed_pair, tmp_path):
    pair, bound, unseal = sealed_pair
    assert pair.provenance['subject_guard']['sealed_guard']['unseal']['sha256'] == unseal.sha256 and pair_module.pair_allowlist(pair) is bound
    for side in ('fixed', 'moving'):
        assert validate_segmentation_provenance(tissue_provenance(pair, side), pair, side=side, layer='tissue', allowlist=bound)['layer'] == 'tissue'
    _, other = W.bound(w)                                                      # another bound allowlist, the same content
    for allowlist, kind in ((None, ProvenanceRejected), (w.allowlist, SubjectSealed), (other, ProvenanceRejected)):
        with pytest.raises(kind):
            validate_segmentation_provenance(tissue_provenance(pair, 'fixed'), pair, side='fixed', layer='tissue', allowlist=allowlist)
        with pytest.raises(kind):
            pair_module.recheck_sealed_pair(pair, allowlist, where='test')
    forged = PairInput(pair.fixed, pair.moving, pair.grid, pair.grid_metadata, pair.fixed_subject, pair.moving_subject, pair.source_sha256, dict(pair.provenance))
    with pytest.raises(ProvenanceRejected, match='not made by load_pair'):
        validate_segmentation_provenance(tissue_provenance(forged, 'fixed'), forged, side='fixed', layer='tissue', allowlist=bound)
    from bayesreg.cli.run_pair import load_tissue_layer
    with pytest.raises(ProvenanceRejected, match='no tissue-layer provenance'):           # through the recheck, then the (absent) file
        load_tissue_layer(pair, 'fixed', tmp_path/'tissue', allowlist=bound)
    for allowlist in (None, other):
        with pytest.raises(ProvenanceRejected, match='C23.4-8'):
            load_tissue_layer(pair, 'fixed', tmp_path/'tissue', allowlist=allowlist)


def test_the_structure_provenance_rechecks_the_sealed_pair_too(w, sealed_pair, tmp_path):
    """Review m10 (C23.9-16, both layers): validate_segmentation_provenance(layer='structure') on the sealed pair takes only the pair's bound allowlist; with it,
    the recheck passes and the structure checks follow (here they accept the provenance)."""
    pair, bound, _ = sealed_pair
    registry = write_registry(tmp_path/'registry.json', [W.TRAIN, W.TRAIN])
    model = sorted(registry.models)[0]
    prov = dict(layer='structure', segmenter_id=registry.models[model]['segmenter_id'], model_ids=[model], model_sha256={model: list(registry.models[model]['member_model_sha256'])},
                trained_with_labels=True, test_labels_seen=False, class_names=['a', 'b'], input_sha256={'aligned_norm.nii.gz': pair.source_sha256['fixed']},
                output_sha256={'g.npz': '3'*64})
    _, other = W.bound(w)
    for allowlist, kind in ((None, ProvenanceRejected), (w.allowlist, SubjectSealed), (other, ProvenanceRejected)):
        with pytest.raises(kind) as refused:
            validate_segmentation_provenance(prov, pair, side='fixed', layer='structure', registry=registry, allowlist=allowlist)
        assert kind is SubjectSealed or 'C23.4-8' in str(refused.value), refused.value
    try:
        record = validate_segmentation_provenance(prov, pair, side='fixed', layer='structure', registry=registry, allowlist=bound)
    except ProvenanceRejected as exc:                                       # a structure check of its own may refuse this made-up provenance; never the recheck
        assert 'C23.4-8' not in str(exc), exc
    else:
        assert record['layer'] == 'structure'


def test_the_structure_layer_takes_only_the_bound_allowlist_of_the_pair(w, sealed_pair, tmp_path):
    from bayesreg.seg.structure_layer import StructureLayer, StructureSource
    pair, bound, unseal = sealed_pair
    registry_path = tmp_path/'staging'/'segmenters.json'
    registry = write_registry(registry_path, [W.TRAIN, W.TRAIN])
    W.write_json(registry_path.parent/'ensemble.json', dict(registry_sha256=registry.sha256, member_check=dict(passed=True, format='test')))
    source = StructureSource(root=tmp_path/'pred', registry_path=registry_path, allowlist_path=w.repo/U.ALLOWLIST_REL, models_root=tmp_path, image_root=tmp_path,
                             segmenter_id='seg_sg_test', allow_staging=True)
    layer = StructureLayer(pair, source, allowlist=bound)
    assert layer.allowlist is bound and layer.record()['allowlist_sha256'] == bound.sha256
    _, other = W.bound(w)
    for allowlist, kind in ((None, ProvenanceRejected), (w.allowlist, SubjectSealed), (other, ProvenanceRejected)):
        with pytest.raises(kind):
            StructureLayer(pair, source, allowlist=allowlist)
    dev = load_pair(write_pair(tmp_path/'dev_pair', W.DEV[0], W.DEV[1]), allowlist=w.allowlist)
    assert StructureLayer(dev, source).allowlist.sha256 == w.allowlist.sha256       # a pair on no list: as before, without an allowlist


def test_a_pair_on_no_list_needs_no_allowlist_in_the_tissue_layer_as_before(tmp_path):
    allowlist = load_dev_allowlist(W.write_json(tmp_path/'allowlist.json', synthetic.allowlist_document([('dev', W.DEV, False)])))
    pair = load_pair(write_pair(tmp_path/'pair', W.DEV[0], W.DEV[1]), allowlist=allowlist)
    assert pair_module.pair_allowlist(pair) is None and pair_module.recheck_sealed_pair(pair, None, where='test') is None
    assert 'sealed_guard' not in pair.provenance['subject_guard']
    assert validate_segmentation_provenance(tissue_provenance(pair, 'fixed'), pair, side='fixed', layer='tissue')['wm_index'] == 2


# ---- C23.9-17: allows is training and label contact; the pair admission is check_subject ---------------------------------------------------------------
def test_allows_is_false_for_a_sealed_subject_bound_or_not_and_the_old_pair_admissions_still_refuse(w):
    unseal, bound = W.bound(w)
    for allowlist in (w.allowlist, bound):
        assert [allowlist.allows(s) for s in W.HOLDOUT] == [False]*3 and allowlist.allows(W.DEV[0]) and [allowlist.lists(s) for s in W.HOLDOUT] == [True]*3
    assert check_subject(W.HOLDOUT[0], bound).admitted and not check_subject(W.HOLDOUT[0], w.allowlist).admitted
    sys.path.insert(0, str(W.ROOT/'experiments/unv2'))
    try:
        import eval_seg
    finally:
        sys.path.remove(str(W.ROOT/'experiments/unv2'))
    with pytest.raises(SubjectSealed, match='sealed list'):
        eval_seg.admit_subjects(bound, 'unsealed_B', S.guard_sealed_lists(), [])
    with pytest.raises(SubjectSealed, match='sealed list'):
        eval_seg.read_truth(w.tmp/'data', W.HOLDOUT[0], 'seg35')            # C23.6-2: the reader refuses by itself, before opening anything


# ---- C23.6-1: OasisFiles ----------------------------------------------------------------------------------------------------------------------------
def test_oasis_files_answer_for_a_sealed_subject_only_when_bound_and_never_for_its_labels(w, tmp_path):
    unseal, bound = W.bound(w)
    plain = OasisFiles(root=tmp_path)
    assert plain.image_path(W.DEV[0]) == tmp_path/W.DEV[0]/'aligned_norm.nii.gz' and plain.label_path(W.DEV[0]).name == 'aligned_seg35.nii.gz'
    with pytest.raises(SubjectSealed, match='bind_files'):
        plain.image_path(W.HOLDOUT[0])
    files = U.bind_files(plain, bound)
    assert files.image_path(W.HOLDOUT[0]) == tmp_path/W.HOLDOUT[0]/'aligned_norm.nii.gz' and files.image_path(W.DEV[0]) == plain.image_path(W.DEV[0])
    for f in (plain, files):
        for kind in ('seg35', 'seg4'):
            with pytest.raises(SubjectSealed, match='labels of a sealed subject'):
                f.label_path(W.HOLDOUT[0], kind)
    for subject in W.H1:
        with pytest.raises(SubjectSealed, match='of paper 1'):
            files.image_path(subject)
    assert files == plain and hash(files) == hash(plain)                      # the binding takes no part in comparison (compare=False)


def test_write_pair_takes_synthetic_ids_and_a_synthetic_root_only(tmp_path):
    from bayesreg.seg.unet.pins import PinnedInputRefused
    from bayesreg.seg.unet.synthetic import write_pair as synthetic_write_pair, write_subjects
    data = write_subjects(tmp_path/'data', [W.DEV[0], W.DEV[1]], shape_xyz=(6, 6, 6))
    assert synthetic_write_pair(tmp_path/'p', data, W.DEV[0], W.DEV[1]).is_file()
    with pytest.raises(PinnedInputRefused, match='non-synthetic'):
        synthetic_write_pair(tmp_path/'q', data, 'OASIS_OAS1_0438_MR1', W.DEV[1])
    with pytest.raises(PinnedInputRefused, match='real data root'):
        synthetic_write_pair(tmp_path/'r', Path('/mnt/d/data'), W.DEV[0], W.DEV[1])
    assert not (tmp_path/'q').exists() and not (tmp_path/'r').exists()
