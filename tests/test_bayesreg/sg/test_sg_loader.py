"""C23.9-6 and C23.9-5 (SG): bayesreg.io.unseal.load_unseal, item by item (C23.3-2 (i)-(ix)), and the unseal object and its bindings cannot be forged
(copy, replace, pickle, by hand, a dict, a namedtuple, HI's old object, another process, a second import).  Synthetic tree in tmp_path (_sg_world.py)."""
import builtins
import collections
import copy
import dataclasses
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys

import pytest

import bayesreg.io.sealed as S
from bayesreg.io import unseal as U
from bayesreg.io.guard import DevAllowlist, SubjectSealed, check_subject
from bayesreg.seg.unet.data import OasisFiles

from . import _sg_world as W


@pytest.fixture
def w(tmp_path, monkeypatch):
    return W.make_world(tmp_path, monkeypatch)


class Opened:
    def __init__(self, monkeypatch):
        self.paths = []

        def opener(real):
            def wrapped(file, *args, **kwargs):
                if isinstance(file, (str, bytes, os.PathLike)):
                    self.paths.append(os.fsdecode(file))
                return real(file, *args, **kwargs)
            return wrapped
        monkeypatch.setattr(builtins, 'open', opener(builtins.open))
        monkeypatch.setattr(io, 'open', opener(io.open))
        import pathlib
        accessor = getattr(pathlib, '_NormalAccessor', None)
        if accessor is not None and 'open' in vars(accessor):
            monkeypatch.setattr(accessor, 'open', staticmethod(opener(vars(accessor)['open'])))


# ---- (i)-(ix): one counter-example each, and both writers -----------------------------------------------------------------------------------------
def test_a_committed_record_is_loaded_and_issued(w):
    unseal = U.load_unseal(w.path)
    assert type(unseal) is U.Unseal and U.unseal_problem(unseal) is None and unseal.subjects == tuple(sorted(W.HOLDOUT)) and unseal.sealed_list == 'holdout_B_v1'
    assert unseal.allowlist_sha256 == w.allowlist.sha256 and unseal.written_by == 'BZ' and unseal.git['record']['path'] == U.UNSEAL_REL
    assert unseal.git['allowlist']['path'] == U.ALLOWLIST_REL and unseal.git['decision_log']['path'] == U.DECISION_LOG_REL
    assert U.load_unseal(w.path) is not unseal                    # every call makes its own object


@pytest.mark.parametrize('where', ['elsewhere_in_the_tree', 'outside_the_tree', 'missing', 'none'])
def test_i_the_path(w, where):
    if where == 'elsewhere_in_the_tree':
        path = W.write_json(w.repo/'splits/bayesreg_v1/unseal_other.json', w.document)
        W.commit_all(w.repo)
    elif where == 'outside_the_tree':
        path = W.write_json(w.tmp/'outside'/U.UNSEAL_REL, w.document)
    elif where == 'missing':
        w.path.unlink()
        path = w.path
    else:
        path = None
    with pytest.raises(U.UnsealRefused) as refused:
        U.load_unseal(path)
    assert refused.value.item == 'path' and refused.value.status == 'subject_sealed' and isinstance(refused.value, SubjectSealed)


def untrack(w):
    W.git(w.repo, 'rm', '-q', '--cached', U.UNSEAL_REL)
    W.git(w.repo, 'commit', '-q', '-m', 'untrack')


GIT_CASES = dict(
    untracked=untrack,
    ignored=lambda w: (untrack(w), W.write_text(w.repo/'.gitignore', U.UNSEAL_REL+'\n'), W.git(w.repo, 'add', '.gitignore'), W.git(w.repo, 'commit', '-q', '-m', 'ignore')),
    staged_only=lambda w: (untrack(w), W.git(w.repo, 'add', U.UNSEAL_REL)),
    modified=lambda w: w.path.write_text(w.path.read_text(encoding='utf-8')+' ', encoding='utf-8'),
)


@pytest.mark.parametrize('case', sorted(GIT_CASES))
def test_ii_git_before_a_byte_is_read(w, monkeypatch, case):
    """HI review I2's cases: committed and unchanged passes (above); untracked, ignored, only staged, modified are refused before the record is opened."""
    GIT_CASES[case](w)
    opened = Opened(monkeypatch)
    with pytest.raises(U.UnsealRefused, match='not a committed, unchanged file') as refused:
        U.load_unseal(w.path)
    assert refused.value.item == 'git' and not [p for p in opened.paths if p.endswith('unseal_holdout_B_v1.json')]


CONTENT = dict(
    not_json=(lambda w: W.write_text(w.path, '{"schema": '), 'json'),
    duplicate_key=(lambda w: W.write_text(w.path, w.path.read_text(encoding='utf-8').replace('"purpose"', '"schema": "x", "purpose"', 1)), 'json'),
    extra_key=(lambda w: W.write_json(w.path, dict(w.document, note='x')), 'keys'),
    missing_key=(lambda w: W.write_json(w.path, {k: v for k, v in w.document.items() if k != 'purpose'}), 'keys'),
    schema=(lambda w: W.write_json(w.path, dict(w.document, schema='bayesreg.hi.unseal.v0')), 'schema'),
    written_by=(lambda w: W.write_json(w.path, dict(w.document, written_by='lead')), 'written_by'),
    written_by_bz_lowercase=(lambda w: W.write_json(w.path, dict(w.document, written_by='bz')), 'written_by'),
    lead_without_bz_words=(lambda w: W.write_json(w.path, dict(w.document, written_by="lead on BZ's behalf")), 'bz_words'),
    bz_words_empty=(lambda w: W.write_json(w.path, dict(w.document, bz_words='  ')), 'bz_words'),
    bz_words_newline=(lambda w: W.write_json(w.path, dict(w.document, bz_words='a\nb')), 'bz_words'),
    bz_words_bar=(lambda w: W.write_json(w.path, dict(w.document, bz_words='a | b')), 'bz_words'),
    bz_words_not_a_string=(lambda w: W.write_json(w.path, dict(w.document, bz_words=['a'])), 'bz_words'),
    time_local=(lambda w: W.write_json(w.path, dict(w.document, time_local='')), 'time_local'),
    purpose=(lambda w: W.write_json(w.path, dict(w.document, purpose=None)), 'purpose'),
    utc_form=(lambda w: W.write_json(w.path, dict(w.document, utc='2026-11-07 12:00')), 'utc'),
    utc_before_the_draw=(lambda w: W.write_json(w.path, dict(w.document, utc='2026-10-29T23:59:59Z')), 'utc'),
    utc_future=(lambda w: W.write_json(w.path, dict(w.document, utc='2031-01-01T00:00:00Z')), 'utc'),
    sealed_list_H1=(lambda w: W.write_json(w.path, dict(w.document, sealed_list='H1')), 'sealed_list'),
    sealed_list_H2=(lambda w: W.write_json(w.path, dict(w.document, sealed_list='H2')), 'sealed_list'),
    sealed_list_H3=(lambda w: W.write_json(w.path, dict(w.document, sealed_list='H3')), 'sealed_list'),
    sealed_list_H4=(lambda w: W.write_json(w.path, dict(w.document, sealed_list='H4')), 'sealed_list'),
    sealed_list_unknown=(lambda w: W.write_json(w.path, dict(w.document, sealed_list='holdout_C')), 'sealed_list'),
    subjects_file=(lambda w: W.write_json(w.path, dict(w.document, subjects_file='lists/h1.txt')), 'subjects_file'),
    subjects_file_sha256=(lambda w: W.write_json(w.path, dict(w.document, subjects_file_sha256='1'*64)), 'subjects_file_sha256'),
    subjects_sha256=(lambda w: W.write_json(w.path, dict(w.document, subjects_sha256='2'*64)), 'subjects_sha256'),
    n_subjects_wrong=(lambda w: W.write_json(w.path, dict(w.document, n_subjects=4)), 'n_subjects'),
    n_subjects_float=(lambda w: W.write_json(w.path, dict(w.document, n_subjects=3.0)), 'n_subjects'),
    n_subjects_bool=(lambda w: W.write_json(w.path, dict(w.document, n_subjects=True)), 'n_subjects'),
    allowlist_sha256=(lambda w: W.write_json(w.path, dict(w.document, allowlist_sha256='3'*64)), 'allowlist'),
    allowlist_source_missing=(lambda w: W.write_json(w.path, dict(w.document, allowlist_source='nowhere')), 'allowlist'),
    allowlist_source_not_exact=(lambda w: W.write_json(w.path, dict(w.document, allowlist_source='dev')), 'allowlist'),
    segmenter_id=(lambda w: W.write_json(w.path, dict(w.document, segmenter_id='a b')), 'segmenter_id'),
    registry_sha256=(lambda w: W.write_json(w.path, dict(w.document, registry_sha256='XYZ')), 'registry_sha256'),
)


@pytest.mark.parametrize('name', sorted(CONTENT))
def test_iii_to_viii_every_departure_of_the_content(w, name):
    """Each record committed and logged with its words on one line, so the refusal is the item's."""
    change, item = CONTENT[name]
    change(w)
    if name not in ('not_json', 'duplicate_key'):
        w.document = json.loads(w.path.read_text(encoding='utf-8'))
    words = w.document.get('bz_words') if isinstance(w.document, dict) and isinstance(w.document.get('bz_words'), str) else W.WORDS
    W.log_line(w.repo, w.path, words.replace('\n', ' '))
    W.commit_all(w.repo)
    with pytest.raises(U.UnsealRefused) as refused:
        U.load_unseal(w.path)
    assert refused.value.item == item, str(refused.value)


@pytest.mark.parametrize('writer, words, passes', [('BZ', W.WORDS, True), ("lead on BZ's behalf", '（合成测试）BZ 原话"解封记录你代我写"；过目：同意', True),
                                                   ("lead on BZ's behalf", '（合成测试）解封记录你替我写', False)])
def test_iv_the_two_writers_d98(w, writer, words, passes):
    W.rewrite(w, written_by=writer, bz_words=words)
    if passes:
        assert U.load_unseal(w.path).written_by == writer
    else:
        with pytest.raises(U.UnsealRefused, match='must contain') as refused:
            U.load_unseal(w.path)
        assert refused.value.item == 'bz_words'


def test_vi_a_list_changed_on_disk_is_refused(w):
    (w.repo/'lists/b.txt').write_text(f'{W.HOLDOUT[0]}\n', encoding='utf-8')
    with pytest.raises(U.UnsealRefused, match='not the pinned sha256') as refused:
        U.load_unseal(w.path)
    assert refused.value.item == 'subjects_file_sha256'


def test_vi_another_list_that_cannot_be_read_is_an_unseal_refusal(w):
    """Review m2: the guard's lists (read in (vi)) cannot be read -> UnsealRefused (a SubjectSealed), not SealedListError; its item is 'sealed_lists' (follow-up
    review n5: here H1's list is gone, not the record's file)."""
    (w.repo/'lists/h1.txt').unlink()
    with pytest.raises(U.UnsealRefused, match='the sealed lists cannot be read .SealedListError') as refused:
        U.load_unseal(w.path)
    assert refused.value.item == 'sealed_lists' and isinstance(refused.value, SubjectSealed)


def test_vii_a_malformed_allowlist_is_an_unseal_refusal(tmp_path, monkeypatch):
    """Review m2: a committed, unchanged allowlist that load_dev_allowlist cannot load -> UnsealRefused('allowlist'), not InputValidationError."""
    w = W.make_world(tmp_path, monkeypatch)
    (w.repo/U.ALLOWLIST_REL).write_text('{"schema": ', encoding='utf-8')
    W.commit_all(w.repo, 'a broken allowlist, committed')
    with pytest.raises(U.UnsealRefused, match='cannot be loaded .InputValidationError') as refused:
        U.load_unseal(w.path)
    assert refused.value.item == 'allowlist' and isinstance(refused.value, SubjectSealed)


@pytest.mark.parametrize('case', ['trainable', 'shared', 'untracked', 'modified'])
def test_vii_the_allowlist(tmp_path, monkeypatch, case):
    if case == 'trainable':
        w = W.make_world(tmp_path, monkeypatch, sources=[('unsealed_B', W.HOLDOUT, False), ('dev', W.DEV, False)])
        words = 'must be never_for_training'
    elif case == 'shared':
        w = W.make_world(tmp_path, monkeypatch, sources=[('unsealed_B', W.HOLDOUT, True), ('dev', W.DEV+W.HOLDOUT[:1], False)])
        words = 'also in other allowlist sources'
    else:
        w = W.make_world(tmp_path, monkeypatch)
        words = 'not a committed, unchanged file'
        if case == 'untracked':
            W.git(w.repo, 'rm', '-q', '--cached', U.ALLOWLIST_REL)
            W.git(w.repo, 'commit', '-q', '-m', 'untrack')
        else:
            path = w.repo/U.ALLOWLIST_REL
            path.write_text(path.read_text(encoding='utf-8')+'\n', encoding='utf-8')
    with pytest.raises(U.UnsealRefused, match=words) as refused:
        U.load_unseal(w.path)
    assert refused.value.item == 'allowlist'


@pytest.mark.parametrize('case', ['no_sha', 'words_on_another_line', 'missing', 'untracked', 'modified'])
def test_ix_the_decision_log(tmp_path, monkeypatch, case):
    w = W.make_world(tmp_path, monkeypatch)
    log = w.repo/U.DECISION_LOG_REL
    if case == 'no_sha':
        log.write_text('| nothing here |\n', encoding='utf-8')
        W.commit_all(w.repo)
        words = 'does not name the record'
    elif case == 'words_on_another_line':
        sha = U.load_unseal(w.path).sha256
        log.write_text(f'| 2026-11-07 | the unseal record {sha} |\n| {W.WORDS} |\n', encoding='utf-8')
        W.commit_all(w.repo)
        words = 'holds both'
    elif case == 'missing':
        W.git(w.repo, 'rm', '-q', U.DECISION_LOG_REL)
        W.git(w.repo, 'commit', '-q', '-m', 'no log')
        words = 'no such file'
    elif case == 'untracked':
        W.git(w.repo, 'rm', '-q', '--cached', U.DECISION_LOG_REL)
        W.git(w.repo, 'commit', '-q', '-m', 'untrack')
        words = 'not a committed, unchanged file'
    else:
        log.write_text(log.read_text(encoding='utf-8')+'x\n', encoding='utf-8')
        words = 'not a committed, unchanged file'
    with pytest.raises(U.UnsealRefused, match=words) as refused:
        U.load_unseal(w.path)
    assert refused.value.item == 'decision_log'


def test_unsealable_is_defined_once_and_names_only_the_b_line():
    """C23.3-2 (v): UNSEALABLE is ('holdout_B_v1',), defined in bayesreg.io.unseal; hi_core imports that very object."""
    sys.path.insert(0, str(W.ROOT/'experiments/hi'))
    try:
        import hi_core
    finally:
        sys.path.remove(str(W.ROOT/'experiments/hi'))
    assert U.UNSEALABLE == ('holdout_B_v1',) and hi_core.UNSEALABLE is U.UNSEALABLE and not set(U.UNSEALABLE) & S.PAPER1_LISTS
    assert hi_core.UNSEAL_KEYS is U.UNSEAL_KEYS and hi_core.UNSEAL_REL == U.UNSEAL_REL == 'splits/bayesreg_v1/unseal_holdout_B_v1.json'
    text = (W.ROOT/'experiments/hi/hi_core.py').read_text(encoding='utf-8')
    assert "UNSEALABLE = ('holdout_B_v1',)" not in text and 'UNSEALABLE = bayes_unseal.UNSEALABLE' in text


# ---- C23.9-5: forgery -----------------------------------------------------------------------------------------------------------------------------
def forged_objects(unseal):
    fields = {f.name: getattr(unseal, f.name) for f in dataclasses.fields(unseal)}
    by_hand = U.Unseal(**fields)
    bare = object.__new__(U.Unseal)
    for name, value in fields.items():
        object.__setattr__(bare, name, value)
    Named = collections.namedtuple('Unseal', list(fields))

    class Sub(U.Unseal):
        pass
    sys.path.insert(0, str(W.ROOT/'experiments/hi'))
    try:
        import hi_core
    finally:
        sys.path.remove(str(W.ROOT/'experiments/hi'))
    old_hi = hi_core.Unseal(path=unseal.path, sha256=unseal.sha256, sealed_list=unseal.sealed_list, subjects=unseal.subjects, allowlist_source=unseal.allowlist_source,
                            allowlist_sha256=unseal.allowlist_sha256, document=unseal.document, git=unseal.git, base=None)
    return dict(by_hand=by_hand, new_then_setattr=bare, copy=copy.copy(unseal), replace=dataclasses.replace(unseal), dict=dict(fields), namedtuple=Named(**fields),
                subclass=Sub(**fields), hi_core_unseal=old_hi)


def test_the_unseal_object_compares_by_identity(w):
    unseal = U.load_unseal(w.path)
    twin = copy.copy(unseal)
    assert twin is not unseal and twin != unseal and unseal == unseal and dataclasses.fields(twin) == dataclasses.fields(unseal)
    assert all(getattr(twin, f.name) == getattr(unseal, f.name) for f in dataclasses.fields(unseal))      # equal field by field, still refused below


def test_deepcopy_and_pickle_cannot_duplicate_the_objects(w, tmp_path):
    """C23.3-5 (ii): the unseal object, the bound allowlist and the bound OasisFiles hold mapping proxies, which neither deepcopy nor pickle can reproduce: none of
    them can be duplicated that way, nor sent to another process by pickling (spawn, multiprocessing, subprocess); a process has to load and bind its own (D99)."""
    unseal, good = W.bound(w)
    files = U.bind_files(OasisFiles(root=tmp_path), good)
    for obj in (unseal, good, files):
        with pytest.raises((TypeError, pickle.PicklingError)):
            pickle.dumps(obj)
        with pytest.raises((TypeError, pickle.PicklingError)):
            copy.deepcopy(obj)


@pytest.mark.parametrize('kind', ['by_hand', 'new_then_setattr', 'copy', 'replace', 'dict', 'namedtuple', 'subclass', 'hi_core_unseal'])
def test_a_forged_unseal_object_is_refused_by_bind_and_by_the_guard(w, kind):
    unseal, good = W.bound(w)
    forged = forged_objects(unseal)[kind]
    assert U.unseal_problem(forged) is not None
    with pytest.raises(U.UnsealRefused) as refused:
        U.bind_unseal(w.allowlist, forged)
    assert refused.value.item == 'bind'
    hung = dataclasses.replace(w.allowlist, unseal=forged)                    # an allowlist with the forged object set by hand
    for subject in (W.HOLDOUT[0], W.DEV[0]):                                    # D100: a bad object refuses a subject on no list too
        verdict = check_subject(subject, hung)
        assert verdict.step == 'sealed' and verdict.item == 'unseal object', verdict
    assert check_subject(W.HOLDOUT[0], good).admitted_by_unseal is True


@pytest.mark.parametrize('how', ['replace', 'copy', 'by_hand'])
def test_a_bound_allowlist_or_bound_files_not_made_by_bind_is_refused(w, tmp_path, how):
    unseal, good = W.bound(w)
    files = U.bind_files(OasisFiles(root=tmp_path/'data'), good)
    make = dict(replace=lambda o: dataclasses.replace(o), copy=copy.copy,
                by_hand=lambda o: dataclasses.replace(w.allowlist, unseal=unseal) if isinstance(o, DevAllowlist) else OasisFiles(root=o.root, allowlist=good))[how]
    other_allowlist, other_files = make(good), make(files)
    assert other_allowlist is not good and other_files is not files
    if how == 'replace':                                                       # equal field by field and still refused (identity, C23.3-5 (ii))
        assert other_allowlist.unseal is unseal and other_allowlist.sha256 == good.sha256
    for subject in (W.HOLDOUT[0], W.DEV[0]):
        verdict = check_subject(subject, other_allowlist)
        assert verdict.step == 'sealed' and verdict.item == 'unseal object', verdict
        with pytest.raises(SubjectSealed, match='bind_files|unseal object|not the one|does not hold'):
            other_files.image_path(subject)
    assert files.image_path(W.HOLDOUT[0]) == tmp_path/'data'/W.HOLDOUT[0]/'aligned_norm.nii.gz'


def test_bind_takes_only_an_unbound_allowlist_of_the_records_sha256_and_bind_files_only_a_bound_one(w, tmp_path):
    unseal, good = W.bound(w)
    other = W.write_json(tmp_path/'other.json', json.loads((w.repo/U.ALLOWLIST_REL).read_text(encoding='utf-8')) | dict(comment='another file, another sha256'))
    from bayesreg.io.guard import load_dev_allowlist
    with pytest.raises(U.UnsealRefused, match='is not the unseal record'):
        U.bind_unseal(load_dev_allowlist(other), unseal)
    with pytest.raises(U.UnsealRefused, match='without an unseal object'):
        U.bind_unseal(good, unseal)                                           # bound already
    with pytest.raises(U.UnsealRefused, match='not an unbound one'):
        U.bind_files(OasisFiles(root=tmp_path), w.allowlist)
    with pytest.raises(U.UnsealRefused, match='without a bound allowlist'):
        U.bind_files(U.bind_files(OasisFiles(root=tmp_path), good), good)
    with pytest.raises(U.UnsealRefused, match='bind_files takes an OasisFiles'):
        U.bind_files(object(), good)


def test_bind_takes_only_the_allowlist_of_the_pinned_file_with_its_content(w, tmp_path):
    """Review m3 (C23.4-1): the same bytes at another path, and an object of the right sha256 with other content (dataclasses.replace), are refused."""
    import shutil
    from bayesreg.io.guard import load_dev_allowlist
    unseal = U.load_unseal(w.path)
    copy = tmp_path/'elsewhere'/'dev_subjects.json'
    copy.parent.mkdir()
    shutil.copyfile(w.repo/U.ALLOWLIST_REL, copy)
    elsewhere = load_dev_allowlist(copy)
    assert elsewhere.sha256 == w.allowlist.sha256
    with pytest.raises(U.UnsealRefused, match='is not configs/bayesreg/dev_subjects.json of the code tree'):
        U.bind_unseal(elsewhere, unseal)
    more = dataclasses.replace(w.allowlist, subjects={'OASIS': w.allowlist.subjects['OASIS'] | {W.oasis(963)}})
    fewer = dataclasses.replace(w.allowlist, sources=w.allowlist.sources[:1])
    for other in (more, fewer):
        with pytest.raises(U.UnsealRefused, match="content is not that of") as refused:
            U.bind_unseal(other, unseal)
        assert refused.value.item == 'bind' and isinstance(refused.value, SubjectSealed)
    assert U.bind_unseal(w.allowlist, unseal).unseal is unseal


def test_bind_files_keeps_the_class_of_the_files(w, tmp_path):
    """C23.4-1 (diff review minor 3): dataclasses.replace inside bind_files keeps a subclass (HI's Spy), entered by identity."""
    unseal, good = W.bound(w)
    asked = []

    class Spy(OasisFiles):
        def image_path(self, subject):
            asked.append(subject)
            return super().image_path(subject)
    files = U.bind_files(Spy(root=tmp_path), good)
    assert type(files) is Spy and U.files_problem(files) is None
    files.image_path(W.HOLDOUT[0])
    assert asked == [W.HOLDOUT[0]]


@pytest.mark.parametrize('change', ['one_byte', 'deleted'])
def test_a_record_changed_or_deleted_after_loading_is_refused_at_the_next_use(w, tmp_path, change):
    unseal, good = W.bound(w)
    files = U.bind_files(OasisFiles(root=tmp_path), good)
    assert check_subject(W.HOLDOUT[0], good).admitted
    if change == 'one_byte':
        w.path.write_bytes(w.path.read_bytes()+b' ')
    else:
        w.path.unlink()
    verdict = check_subject(W.HOLDOUT[0], good)
    assert verdict.step == 'sealed' and 'changed or is gone' in verdict.message
    assert check_subject(W.DEV[0], good).step == 'sealed'                     # D100
    with pytest.raises(SubjectSealed, match='changed or is gone'):
        files.image_path(W.HOLDOUT[0])
    with pytest.raises(U.UnsealRefused, match='changed or is gone'):
        U.bind_unseal(w.allowlist, unseal)


# ---- across processes (C23.3-5 (ii), diff review minor 7) -----------------------------------------------------------------------------------------
def _spawned_child(unseal, bound, subject):          # never runs: the objects cannot be pickled into it
    from bayesreg.io.guard import check_subject as child_check
    return child_check(subject, bound)


def test_a_spawned_process_cannot_be_handed_the_objects(w):
    """C23.3-5 (ii): spawn pickles the arguments of the child; the unseal object and the bound allowlist cannot be pickled, so the child never gets them (a process
    has to load_unseal and bind_unseal itself, D99).  Nothing starts."""
    unseal, good = W.bound(w)
    child = multiprocessing.get_context('spawn').Process(target=_spawned_child, args=(unseal, good, W.HOLDOUT[0]))
    with pytest.raises((TypeError, pickle.PicklingError)):
        child.start()
    assert child.pid is None or not child.is_alive()


def _in_a_forked_child(bound, subject, record_path, queue):
    from bayesreg.io.guard import check_subject as child_check
    first = child_check(subject, bound)
    Path(record_path).write_bytes(Path(record_path).read_bytes()+b' ')
    second = child_check(subject, bound)
    queue.put(((first.step, first.admitted_by_unseal), (second.step, second.item)))


@pytest.mark.skipif('fork' not in multiprocessing.get_all_start_methods(), reason='fork is not available on this platform')
def test_a_forked_child_inherits_the_tables_and_still_rechecks_the_record(w):
    unseal, good = W.bound(w)
    context = multiprocessing.get_context('fork')
    queue = context.Queue()
    child = context.Process(target=_in_a_forked_child, args=(good, W.HOLDOUT[0], str(w.path), queue))
    child.start()
    first, second = queue.get(timeout=120)
    child.join(timeout=120)
    assert first == (None, True) and second == ('sealed', 'unseal object')


def test_a_second_import_of_the_module_has_its_own_class_and_tables(w):
    """C23.3-5 (ii): bayesreg/io/unseal.py imported under another name (as if a snapshot and the tree were both on sys.path): its objects are refused."""
    spec = importlib.util.spec_from_file_location('bayesreg.io._unseal_twin', W.ROOT/'bayesreg/io/unseal.py')
    twin = importlib.util.module_from_spec(spec)
    sys.modules['bayesreg.io._unseal_twin'] = twin
    try:
        spec.loader.exec_module(twin)
        other = twin.load_unseal(w.path)
        assert type(other) is twin.Unseal and twin.Unseal is not U.Unseal and twin.unseal_problem(other) is None
        with pytest.raises(U.UnsealRefused, match='is not the unseal object'):
            U.bind_unseal(w.allowlist, other)
        with pytest.raises(twin.UnsealRefused):
            twin.bind_unseal(w.allowlist, U.load_unseal(w.path))
    finally:
        del sys.modules['bayesreg.io._unseal_twin']
