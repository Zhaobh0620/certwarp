"""LP-A item 1 (contract C26.3; C26.10 T1-T7, T17): the LPBA40 rejection constant of bayesreg.io.guard.check_subject, its one exception (the pinned LPBA allowlist), the
name-only refusal shared by truth.check_name and load_pair, and the records.  Ids and synthetic files only: the real code-tree lists and allowlists are read (ids), the
pairs are synthetic files in tmp_path; no LPBA40 image or label is opened."""
import ast
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

import bayesreg.io.guard as G
import bayesreg.io.pair as P
import bayesreg.io.sealed as S
from bayesreg.eval import truth as T
from bayesreg.io.guard import SubjectSealed, check_pair_subjects, check_subject, load_dev_allowlist
from bayesreg.seg.unet.infer import check_subject_admitted

from ..io_seg._helpers import write_pair
from ..sg import _sg_world as W

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments'/'lpba'))
import make_lpba40_allowlist as MK  # noqa: E402

BASE = '5c183cc5efcd1b286139483abe082612f142d4b7'          # main before LP-A (the guard before C26.3)
TEST34 = [f'S{n:02d}' for n in range(3, 37)]
DEV_ALLOWLIST = ROOT/'configs/bayesreg/dev_subjects.json'
ANCHOR = 'OASIS_OAS1_0438_MR1'


@pytest.fixture(autouse=True)
def fresh_cache():
    S._reset_cache()
    yield
    S._reset_cache()


@pytest.fixture(scope='module')
def dev():
    return load_dev_allowlist(DEV_ALLOWLIST)


@pytest.fixture(scope='module')
def lpba():
    return load_dev_allowlist(G.lpba40_allowlist_path())


@pytest.fixture(scope='module')
def old():
    """The guard of BASE (before C26.3), a module of its own, resolving its relative imports in bayesreg.io (the same sealed lists)."""
    text = subprocess.run(['git', '-C', str(ROOT), 'show', f'{BASE}:bayesreg/io/guard.py'], capture_output=True, text=True, check=True).stdout
    module = types.ModuleType('old_guard_lp_a')
    module.__package__ = 'bayesreg.io'
    exec(compile(text, 'old_guard_lp_a.py', 'exec'), module.__dict__)
    return module


def outcome(call):
    try:
        return ('ok', json.dumps(call(), sort_keys=False, default=str))
    except Exception as exc:                          # noqa: BLE001 -- the class (by name) and the message are compared
        return (type(exc).__name__, str(exc))


class Opened:
    """Records what load_pair opened: the manifest decoder and the archive opener."""

    def __init__(self, monkeypatch):
        self.manifests, self.archives = [], []
        real_read, real_open = P._read_manifest, P._open_npz
        monkeypatch.setattr(P, '_read_manifest', lambda path: (self.manifests.append(str(path)), real_read(path))[1])
        monkeypatch.setattr(P, '_open_npz', lambda path: (self.archives.append(str(path)), real_open(path))[1])


# ---- T4: the constant ---------------------------------------------------------------------------------------------------------------------------
def test_t4_the_constant_is_the_34_test_subjects_and_its_sha256_is_pinned():
    listed = [line for line in (ROOT/'splits/lpba40_test_v1/test_subjects.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(G.LPBA40_TEST_REJECT) == 34 and isinstance(G.LPBA40_TEST_REJECT, frozenset)
    assert G.LPBA40_TEST_REJECT == set(listed) == set(TEST34)
    assert G.lpba40_reject_sha256() == G.LPBA40_TEST_REJECT_SHA256
    assert G.lpba40_reject_text().encode('utf-8') == (ROOT/'splits/lpba40_test_v1/test_subjects.txt').read_bytes()
    assert G.LPBA40_TEST_REJECT_SHA256 == hashlib.sha256(''.join(f'{s}\n' for s in sorted(TEST34)).encode()).hexdigest()


def test_the_lpba_allowlist_is_the_generators_and_pinned():
    path = G.lpba40_allowlist_path()
    assert MK.OUT_REL == G.LPBA40_ALLOWLIST_REL and path == ROOT/G.LPBA40_ALLOWLIST_REL
    assert path.read_bytes() == MK.text(ROOT).encode('utf-8')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == G.LPBA40_ALLOWLIST_SHA256
    allowlist, dev_doc = load_dev_allowlist(path), json.loads(DEV_ALLOWLIST.read_text(encoding='utf-8'))
    doc = json.loads(path.read_text(encoding='utf-8'))
    assert doc['sources'][:-1] == dev_doc['sources']                       # every development source copied item by item (C26.3-5)
    test = doc['sources'][-1]
    assert test == dict(name='lpba40_test_v1', dataset='LPBA40', path='splits/lpba40_test_v1/test_subjects.txt',
                        sha256=hashlib.sha256((ROOT/'splits/lpba40_test_v1/test_subjects.txt').read_bytes()).hexdigest(), never_for_training=False, subjects=TEST34)
    train = {s['name']: s for s in doc['sources']}['bayesreg_train_v1']
    assert train['sha256'].startswith('81aefd1a') and len(train['subjects']) == 106
    assert not set(TEST34) & allowlist.never_for_training                 # D143 false
    assert sorted(allowlist.subjects['LPBA40']) == [f'S{n:02d}' for n in range(1, 37)]


def test_the_name_pattern_is_truths():
    assert G.LPBA40_IN_NAME.pattern == T._LPBA40_IN_NAME.pattern and G.LPBA40_IN_NAME.flags == T._LPBA40_IN_NAME.flags


# ---- T1: the development allowlist refuses the 34 everywhere ----------------------------------------------------------------------------------------
@pytest.mark.parametrize('subject', TEST34)
def test_t1_check_subject_refuses_the_34_as_sealed_on_the_development_allowlist(dev, subject):
    verdict = check_subject(subject, dev)
    assert (verdict.step, verdict.item, verdict.dataset, verdict.sealed_lists, verdict.admitted_by_lpba40) == ('sealed', 'lpba40 test set', 'LPBA40', (), False)
    assert 'LPBA40_TEST_REJECT' in verdict.message and '(i) path' in verdict.message
    with pytest.raises(SubjectSealed, match='LPBA40_TEST_REJECT'):
        check_pair_subjects(subject, ANCHOR, dev)
    with pytest.raises(SubjectSealed, match='LPBA40_TEST_REJECT'):
        check_pair_subjects('S01', subject, dev)
    with pytest.raises(SubjectSealed, match='not segmented'):
        check_subject_admitted(subject, dev)
    assert dev.allows(subject) == dev.lists(subject)                     # allows is not changed (C26.3-7)


def test_t1_s25_to_s36_are_sealed_not_allowlist(dev):
    for subject in TEST34[22:]:
        assert not dev.lists(subject) and check_subject(subject, dev).step == 'sealed'


@pytest.mark.parametrize('pair', ['LPBA40_S03__LPBA40_S04', 'LPBA40_S01__LPBA40_S36', 'LPBA40_S25__LPBA40_S02'])
def test_t1_truth_and_load_pair_refuse_by_name_before_any_file(tmp_path, monkeypatch, dev, pair):
    opened = Opened(monkeypatch)
    sealed = T.load_sealed_lists()
    with pytest.raises(SubjectSealed, match='LPBA40_TEST_REJECT'):
        T.check_name('stage_x', pair, sealed)
    with pytest.raises(SubjectSealed, match='LPBA40_TEST_REJECT'):
        T.admit_pair(tmp_path, 'stage_x', pair, dev, sealed)
    fixed, moving = (s.removeprefix('LPBA40_') for s in pair.split('__'))
    path, _ = write_pair(tmp_path/'stage'/pair, fixed_subject=fixed, moving_subject=moving)
    with pytest.raises(SubjectSealed, match='LPBA40_TEST_REJECT'):
        P.load_pair(path, allowlist=dev)
    record = dict(unsealed=False, manifest_sha256='0'*64, fixed_subject=fixed, moving_subject=moving, pair_sha256_manifest='0'*64, allowlist_path=str(DEV_ALLOWLIST),
                  allowlist_sha256=dev.sha256)
    with pytest.raises(SubjectSealed, match='LPBA40_TEST_REJECT'):
        T.recheck_guard(path.parent, record)
    with pytest.raises(SubjectSealed, match='LPBA40_TEST_REJECT'):
        T.load_truth(path.parent, None, None, record)
    assert opened.manifests == [] and opened.archives == []


# ---- T2: the LPBA allowlist admits them; truth never -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('subject', TEST34)
def test_t2_the_lpba_allowlist_admits_the_34(lpba, subject):
    verdict = check_subject(subject, lpba)
    assert verdict.admitted and verdict.admitted_by_lpba40 and verdict.sealed_lists == () and verdict.dataset == 'LPBA40'
    check_subject_admitted(subject, lpba)
    record = check_pair_subjects(subject, 'S01', lpba)
    assert record['lpba40_guard'] == dict(version='lpba40-guard-v1', fixed=dict(admitted_by_lpba40=True), moving=dict(admitted_by_lpba40=False),
                                          allowlist_rel=G.LPBA40_ALLOWLIST_REL, allowlist_sha256=G.LPBA40_ALLOWLIST_SHA256,
                                          reject_constant_sha256=G.LPBA40_TEST_REJECT_SHA256, reject_item='lpba40 test set', source='lpba40_test_v1')
    assert record['unsealed'] is False and 'sealed_guard' not in record


def test_t2_load_pair_admits_an_lpba_test_pair_and_truth_refuses_it(tmp_path, monkeypatch, lpba):
    opened = Opened(monkeypatch)
    path, _ = write_pair(tmp_path/'stage'/'LPBA40_S05__LPBA40_S06', fixed_subject='S05', moving_subject='S06')
    pair = P.load_pair(path, allowlist=lpba)
    guard = pair.provenance['subject_guard']
    assert guard['lpba40_guard']['fixed'] == guard['lpba40_guard']['moving'] == dict(admitted_by_lpba40=True)
    assert len(opened.manifests) == 1 and len(opened.archives) == 1
    sealed = T.load_sealed_lists()
    with pytest.raises(SubjectSealed, match='no exception here'):
        T.check_name('stage', 'LPBA40_S05__LPBA40_S06', sealed)
    with pytest.raises(SubjectSealed, match='no exception here'):
        T.admit_pair(tmp_path, 'stage', 'LPBA40_S05__LPBA40_S06', lpba, sealed)
    record = dict(guard, unsealed=False, manifest_sha256=pair.provenance['manifest_sha256'], pair_sha256_manifest=pair.provenance['pair_sha256'],
                  allowlist_path=str(G.lpba40_allowlist_path()))
    with pytest.raises(SubjectSealed, match='no exception here'):
        T.load_truth(path.parent, None, pair.grid, record)
    assert len(opened.archives) == 1                                    # truth opened nothing


def test_t2_the_exception_does_not_need_the_name(tmp_path, lpba, dev):
    """A neutral directory name: the refusal (dev) is after the manifest's whitelisted decoding and before pair.npz; the LPBA allowlist admits."""
    path, _ = write_pair(tmp_path/'neutral', fixed_subject='S07', moving_subject='S08')
    assert P.load_pair(path, allowlist=lpba).provenance['subject_guard']['lpba40_guard']['fixed']['admitted_by_lpba40']
    with pytest.raises(SubjectSealed, match='LPBA40_TEST_REJECT'):
        P.load_pair(path, allowlist=dev)


# ---- T3: the counterexamples of the exception (a synthetic tree pinned by monkeypatch) ---------------------------------------------------------------
def lpba_document(test_subjects=tuple(TEST34), *, test_source=True, never=False):
    sources = [dict(name='dev', dataset='OASIS', path='lists/dev.txt', sha256='a'*64, never_for_training=False, subjects=list(W.DEV)),
               dict(name='lpba_tune', dataset='LPBA40', path='lists/tune.txt', sha256='b'*64, never_for_training=False, subjects=[f'S{n:02d}' for n in range(1, 25)])]
    if test_source:
        sources.append(dict(name='lpba40_test_v1', dataset='LPBA40', path='splits/lpba40_test_v1/test_subjects.txt', sha256='c'*64, never_for_training=never,
                            subjects=list(test_subjects)))
    lpba_ids = sorted({s for src in sources if src['dataset'] == 'LPBA40' for s in src['subjects']})
    return dict(schema=G.ALLOWLIST_SCHEMA, subjects=dict(OASIS=list(W.DEV), LPBA40=lpba_ids), sources=sources)


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch, record=False)
    path = W.write_json(w.repo/G.LPBA40_ALLOWLIST_REL, lpba_document())
    monkeypatch.setattr(G, 'LPBA40_ALLOWLIST_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    w.lpba_path = path
    w.lpba = load_dev_allowlist(path)
    return w


def test_t3_the_synthetic_pinned_file_admits(world):
    assert check_subject('S20', world.lpba).admitted_by_lpba40
    assert check_subject('S01', world.lpba).admitted and not check_subject('S01', world.lpba).admitted_by_lpba40


def refused_as(subject, allowlist, words):
    verdict = check_subject(subject, allowlist)
    assert (verdict.step, verdict.item) == ('sealed', 'lpba40 test set') and words in verdict.message, verdict
    with pytest.raises(SubjectSealed):
        check_pair_subjects(subject, 'S01', allowlist)
    with pytest.raises(SubjectSealed):
        G.check_lpba40_names([f'LPBA40_{subject}__LPBA40_S01'], allowlist)


def test_t3_one_byte_changed_in_the_pinned_file(world):
    world.lpba_path.write_bytes(world.lpba_path.read_bytes()+b' ')
    refused_as('S20', world.lpba, '(ii) sha256')


def test_t3_the_same_content_at_another_path(world):
    other = world.repo/'configs/bayesreg/copy.json'
    other.write_bytes(world.lpba_path.read_bytes())
    refused_as('S20', load_dev_allowlist(other), '(i) path')


def test_t3_right_path_and_sha256_but_other_content(world):
    forged = dataclasses.replace(world.lpba, subjects=dict(world.lpba.subjects, LPBA40=frozenset(world.lpba.subjects['LPBA40'] - {'S24'})))
    assert (forged.path, forged.sha256) == (world.lpba.path, world.lpba.sha256)
    refused_as('S20', forged, '(ii) content')
    sources = tuple(s if s['name'] != 'lpba_tune' else dict(s, subjects=('S01',)) for s in world.lpba.sources)
    refused_as('S20', dataclasses.replace(world.lpba, sources=sources), '(ii) content')
    refused_as('S20', dataclasses.replace(world.lpba, never_for_training=frozenset({'S20'})), '(ii) content')


def test_t3_the_development_allowlist_and_another_allowlist(world):
    refused_as('S20', world.allowlist, '(i) path')
    with pytest.raises(SubjectSealed):
        check_subject_admitted('S20', world.allowlist)


def test_t3_no_test_source_or_the_subject_not_in_it(world, monkeypatch):
    path = W.write_json(world.lpba_path, lpba_document(test_source=False))
    monkeypatch.setattr(G, 'LPBA40_ALLOWLIST_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    refused_as('S03', load_dev_allowlist(path), '(iii) source')
    path = W.write_json(world.lpba_path, lpba_document(test_subjects=[f'S{n:02d}' for n in range(3, 11)]))
    monkeypatch.setattr(G, 'LPBA40_ALLOWLIST_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    allowlist = load_dev_allowlist(path)
    assert check_subject('S05', allowlist).admitted_by_lpba40
    refused_as('S20', allowlist, '(iii) source')                       # on the allowlist (lpba_tune), not in lpba40_test_v1


def test_t3_the_pinned_file_cannot_be_read(world):
    allowlist = world.lpba
    world.lpba_path.write_text('{not json', encoding='utf-8')
    refused_as('S20', allowlist, 'cannot be read again')
    world.lpba_path.unlink()
    refused_as('S20', allowlist, 'cannot be read again')


def test_t3_the_pinned_sha256_is_another(world, monkeypatch):
    monkeypatch.setattr(G, 'LPBA40_ALLOWLIST_SHA256', 'f'*64)
    refused_as('S20', world.lpba, '(ii) sha256')


def test_t3_an_allowlist_with_an_unseal_object_is_refused(world):
    forged = dataclasses.replace(world.lpba, unseal=object())
    verdict = check_subject('S20', forged)
    assert verdict.step == 'sealed' and not verdict.admitted


# ---- T5: everyone else as before --------------------------------------------------------------------------------------------------------------------
def test_t5_the_development_allowlist_records_are_byte_for_byte_as_before(old, dev):
    old_dev = old.load_dev_allowlist(DEV_ALLOWLIST)
    others = sorted(s for ids in dev.subjects.values() for s in ids if s not in G.LPBA40_TEST_REJECT)
    assert len(others) == 162
    for subject in others+['S37', 'S40', 'OASIS_OAS1_0901_MR1', 'oasis_0438', 'S41']:
        for fixed, moving in ((subject, ANCHOR), (ANCHOR, subject), (subject, 'S01')):
            assert outcome(lambda: check_pair_subjects(fixed, moving, dev)) == outcome(lambda: old.check_pair_subjects(fixed, moving, old_dev)), (fixed, moving)
        new, before = check_subject(subject, dev), old.check_subject(subject, old_dev)
        assert (new.step, new.item, new.message, new.dataset, new.sealed_lists, new.admitted_by_unseal) == \
            (before.step, before.item, before.message, before.dataset, before.sealed_lists, before.admitted_by_unseal)


def test_t5_the_lpba_allowlist_adds_no_key_for_the_others(old, lpba):
    old_lpba = old.load_dev_allowlist(G.lpba40_allowlist_path())
    for fixed, moving in (('S01', 'S02'), ('S02', 'S01'), (ANCHOR, 'S01'), ('OASIS_OAS1_0439_MR1', ANCHOR)):
        new, before = check_pair_subjects(fixed, moving, lpba), old.check_pair_subjects(fixed, moving, old_lpba)
        assert json.dumps(new) == json.dumps(before) and 'lpba40_guard' not in new


# ---- T6: the order, and the one shared function -----------------------------------------------------------------------------------------------------
def test_t6_the_shared_name_function(lpba, dev):
    assert G.check_lpba40_names(['LPBA40_S01__LPBA40_S02', 'stage_LPBA40_v1', 'OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1']) == ['S01', 'S02']
    assert G.check_lpba40_names(['LPBA40_S37__LPBA40_S38']) == ['S37', 'S38']      # H2 is the sealed lists' business, not this constant's
    for allowlist in (None, dev):
        with pytest.raises(SubjectSealed, match='S04'):
            G.check_lpba40_names(['LPBA40_S01__LPBA40_S04'], allowlist)
    assert G.check_lpba40_names(['LPBA40_S03__LPBA40_S04', 'LPBA40_S35__LPBA40_S36'], lpba) == ['S03', 'S04', 'S35', 'S36']


def calls_of(function):
    tree = ast.parse(inspect.getsource(function).lstrip())
    return [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]


def test_t6_truth_and_load_pair_call_the_same_function():
    assert T.check_lpba40_names is G.check_lpba40_names and P.check_lpba40_names is G.check_lpba40_names
    assert 'check_lpba40_names' in calls_of(T.check_name) and 'check_lpba40_names' in calls_of(P.load_pair) and 'check_lpba40_names' in calls_of(T.recheck_guard)
    load = calls_of(P.load_pair)
    assert load.index('check_lpba40_names') < load.index('_read_manifest')


def test_t6_the_callers_really_call_it(tmp_path, monkeypatch, lpba):
    seen = []

    def spy(names, allowlist=None):
        seen.append((list(names), allowlist))
        return G.check_lpba40_names(names, allowlist)
    monkeypatch.setattr(T, 'check_lpba40_names', spy)
    monkeypatch.setattr(P, 'check_lpba40_names', spy)
    T.check_name('stage', 'OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1', T.load_sealed_lists())
    path, _ = write_pair(tmp_path/'LPBA40_S09__LPBA40_S10', fixed_subject='S09', moving_subject='S10')
    P.load_pair(path, allowlist=lpba)
    assert seen == [(['OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1'], None), (['LPBA40_S09__LPBA40_S10', 'LPBA40_S09__LPBA40_S10'], lpba)]     # the name and the resolved name (minor 1)


def test_t6_a_neutral_name_is_refused_after_the_manifest_before_the_archive(tmp_path, monkeypatch, dev):
    opened = Opened(monkeypatch)
    path, _ = write_pair(tmp_path/'neutral', fixed_subject='S11', moving_subject='S12')
    with pytest.raises(SubjectSealed, match='LPBA40_TEST_REJECT'):
        P.load_pair(path, allowlist=dev)
    assert opened.manifests == [str(path.parent/'manifest.json')] and opened.archives == []


# ---- T7, T17 -----------------------------------------------------------------------------------------------------------------------------------------
def test_t7_recheck_sealed_pair_returns_none_for_the_34(tmp_path, lpba):
    path, _ = write_pair(tmp_path/'LPBA40_S13__LPBA40_S14', fixed_subject='S13', moving_subject='S14')
    pair = P.load_pair(path, allowlist=lpba)
    assert P.recheck_sealed_pair(pair, lpba, where='t7') is None
    assert P.recheck_sealed_pair(pair, None, where='t7') is None
    assert not S.lists_of(S.guard_sealed_lists(), 'S13')


def test_t17_the_manifest_directory_is_named_too(tmp_path, monkeypatch, lpba, dev):
    opened = Opened(monkeypatch)
    path, _ = write_pair(tmp_path/'neutral', fixed_subject='S01', moving_subject='S02')
    elsewhere, _ = write_pair(tmp_path/'LPBA40_S05__LPBA40_S06', fixed_subject='S01', moving_subject='S02')
    with pytest.raises(SubjectSealed, match='LPBA40_S05__LPBA40_S06'):
        P.load_pair(path, allowlist=dev, manifest_path=elsewhere.parent/'manifest.json')
    path2, _ = write_pair(tmp_path/'LPBA40_S07__LPBA40_S08', fixed_subject='S01', moving_subject='S02')
    with pytest.raises(SubjectSealed, match='LPBA40_S07__LPBA40_S08'):
        P.load_pair(path2, allowlist=dev, manifest_path=path.parent/'manifest.json')
    assert opened.manifests == [] and opened.archives == []
    assert P.load_pair(path, allowlist=lpba, manifest_path=elsewhere.parent/'manifest.json').fixed_subject == 'S01'     # the exception holds for both names



# ---- revision 11c-1 (review I1): truth refuses by the manifest's subjects and by the resolved name, on any allowlist; review minor 1 for load_pair ---------
class NpLoads:
    def __init__(self, monkeypatch):
        self.calls = []
        real = T.np.load
        monkeypatch.setattr(T.np, 'load', lambda *a, **k: (self.calls.append(a[0]), real(*a, **k))[1])


def forged_record(pair, path):
    return dict(pair.provenance['subject_guard'], unsealed=False, manifest_sha256=pair.provenance['manifest_sha256'], pair_sha256_manifest=pair.provenance['pair_sha256'],
                allowlist_path=str(G.lpba40_allowlist_path()))


@pytest.mark.parametrize('which', ['lpba', 'dev'])
def test_i1_a_neutral_name_whose_manifest_names_the_34_is_refused_by_truth(tmp_path, monkeypatch, lpba, dev, which):
    allowlist = dict(lpba=lpba, dev=dev)[which]
    path, _ = write_pair(tmp_path/'stage'/'neutral_pair', fixed_subject='S05', moving_subject='S06')
    loads = NpLoads(monkeypatch)
    with pytest.raises(SubjectSealed, match='manifest names'):
        T.admit_pair(tmp_path, 'stage', 'neutral_pair', allowlist, T.load_sealed_lists())
    pair = P.load_pair(path, allowlist=lpba)                             # the LPBA allowlist admits the pair for the entries (the exception), never for truth
    loads.calls.clear()
    with pytest.raises(SubjectSealed, match='manifest names'):
        T.load_truth(path.parent, None, pair.grid, forged_record(pair, path))
    assert loads.calls == []                                             # no member of pair.npz read


def test_i1_a_link_to_a_named_directory_is_refused_by_truth(tmp_path, monkeypatch, lpba):
    real_dir = tmp_path/'real'/'LPBA40_S05__LPBA40_S06'
    path, _ = write_pair(real_dir, fixed_subject='S05', moving_subject='S06')
    (tmp_path/'stage').mkdir()
    (tmp_path/'stage'/'neutral_link').symlink_to(real_dir, target_is_directory=True)
    pair = P.load_pair(path, allowlist=lpba)
    loads = NpLoads(monkeypatch)
    for stage_pair in ('neutral_link',):
        with pytest.raises(SubjectSealed, match='LPBA40_S05__LPBA40_S06'):
            T.admit_pair(tmp_path, 'stage', stage_pair, lpba, T.load_sealed_lists())
        with pytest.raises(SubjectSealed, match='LPBA40_S05__LPBA40_S06'):
            T.load_truth(tmp_path/'stage'/stage_pair, None, pair.grid, forged_record(pair, path))
    assert loads.calls == []


def test_i1_truth_still_admits_the_others(tmp_path, lpba, dev):
    write_pair(tmp_path/'stage'/'LPBA40_S01__LPBA40_S02', fixed_subject='S01', moving_subject='S02')
    for allowlist in (lpba, dev):
        pair_dir, _, _, guard = T.admit_pair(tmp_path, 'stage', 'LPBA40_S01__LPBA40_S02', allowlist, T.load_sealed_lists())
        T.recheck_guard(pair_dir, guard)
        assert 'lpba40_guard' not in guard


def test_minor1_load_pair_names_the_resolved_directory(tmp_path, monkeypatch, dev, lpba):
    opened = Opened(monkeypatch)
    path, _ = write_pair(tmp_path/'LPBA40_S05__LPBA40_S06', fixed_subject='S01', moving_subject='S02')
    monkeypatch.chdir(path.parent)
    with pytest.raises(SubjectSealed, match='LPBA40_S05__LPBA40_S06'):
        P.load_pair('pair.npz', allowlist=dev)                          # Path('pair.npz').parent.name is ''
    (tmp_path/'neutral').symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(SubjectSealed, match='LPBA40_S05__LPBA40_S06'):
        P.load_pair(tmp_path/'neutral'/'pair.npz', allowlist=dev)
    with pytest.raises(SubjectSealed, match='LPBA40_S05__LPBA40_S06'):
        P.load_pair(tmp_path/'neutral'/'pair.npz', allowlist=dev, manifest_path='manifest.json')
    assert opened.manifests == [] and opened.archives == []
    assert P.load_pair('pair.npz', allowlist=lpba).fixed_subject == 'S01'      # the exception holds for the resolved name too
