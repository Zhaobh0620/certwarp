"""C23.9-9 (SG, C23.7): without an unseal object, the guards admit and refuse exactly as before.  The old guard is c027447's bayesreg/io/guard.py (git show), imported
as a module of its own, and the old check_subject_admitted is c027447's function text from bayesreg/seg/unet/infer.py, executed with the old guard's names.  For every
id of the repository's allowlist (160 OASIS ids, the LPBA40 source) and for refused groups (holdout_B_v1, H1, H4, H2, H3 ids, ids off the allowlist, non-canonical
ids), both guards give the same record byte for byte, or the same exception class (by name) and message; DevAllowlist.allows gives the same answer.  Ids only.
SG code review I1 (contract C23 revision 6h): the assertions follow RECORD, as the HI tests do (diff review n4).  Main has no unseal record: the allowlist is main's
(184 ids, 160 OASIS).  The dedicated unseal branch has BZ's record and an allowlist that also holds the record's source: those subjects are taken out of
"allowlisted" (main's 184 stay) and checked on their own -- the old guard admitted them, the new guard refuses them unbound and admits them bound to the record
(check_unsealed; the same function is run on the synthetic unseal branch of _sg_world below, so that the branch's code is exercised on main too)."""
import ast
import json
import subprocess
import types

import pytest

from bayesreg.io import unseal as U
from bayesreg.io.guard import LPBA40_TEST_REJECT, check_pair_subjects, check_subject, load_dev_allowlist
from bayesreg.io.sealed import load_sealed_lists
from bayesreg.seg.unet.infer import check_subject_admitted

from . import _sg_world as W

OLD = 'c0274476b4a61ef82952eff5eed459baabb15ced'          # main before SG (the contract's c027447; bayesreg/ has no diff from it to 854c708)
ANCHOR = 'OASIS_OAS1_0438_MR1'
RECORD = (W.ROOT/U.UNSEAL_REL).exists()           # review I1: the unseal branch has BZ's record, main has none (as tests/test_bayesreg/hi/_world.py RECORD)


def git_show(path):
    return subprocess.run(['git', '-C', str(W.ROOT), 'show', f'{OLD}:{path}'], capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope='module')
def old():
    guard = types.ModuleType('old_guard_c027447')
    exec(compile(git_show('bayesreg/io/guard.py'), 'old_guard_c027447.py', 'exec'), guard.__dict__)
    infer_text = git_show('bayesreg/seg/unet/infer.py')
    function = next(n for n in ast.parse(infer_text).body if isinstance(n, ast.FunctionDef) and n.name == 'check_subject_admitted')
    namespace = dict(subject_dataset=guard.subject_dataset, SubjectSealed=guard.SubjectSealed)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'old_infer_c027447.py', 'exec'), namespace)
    path = W.ROOT/'configs/bayesreg/dev_subjects.json'
    return types.SimpleNamespace(guard=guard, admitted=namespace['check_subject_admitted'], allowlist=guard.load_dev_allowlist(path), new_allowlist=load_dev_allowlist(path))


def unsealed_subjects():
    """On the unseal branch: the subjects of BZ's record (the real loader); on main: none."""
    return sorted(U.load_unseal(W.ROOT/U.UNSEAL_REL).subjects) if RECORD else []


def ids():
    lists = load_sealed_lists()
    allowlist = load_dev_allowlist(W.ROOT/'configs/bayesreg/dev_subjects.json')
    unsealed = set(unsealed_subjects())
    listed = sorted(s for values in allowlist.subjects.values() for s in values if s not in unsealed)       # main's allowlist (the record's source taken out)
    sealed = {name: sorted(lists.subjects[name])[:12] for name in ('holdout_B_v1', 'H1', 'H4', 'H3')}
    on_lists = {s for values in lists.subjects.values() for s in values}
    off = [s for s in (f'OASIS_OAS1_{n:04d}_MR1' for n in range(1, 458)) if s not in on_lists and not allowlist.lists(s)][:12]
    groups = dict(allowlisted=listed, **sealed, H2=['S37', 'S38', 'S39', 'S40'], ibsr=['IBSR_01', 'ibsr_18'], off=off+['OASIS_OAS1_0901_MR1', 'OASIS_OAS1_9999_MR1'],
                  noncanonical=['oasis_0438', 'OASIS_OAS1_438_MR1', 'S41', 'S1', '', ' OASIS_OAS1_0438_MR1', None, 438], unsealed=sorted(unsealed))
    assert len(listed) == 184 and len([s for s in listed if s.startswith('OASIS_')]) == 160, len(listed)
    if RECORD:                                                    # the record's source is exactly the record's subjects, all of them on the allowlist of the branch
        assert sorted(s for values in allowlist.subjects.values() for s in values if s in unsealed) == sorted(unsealed) == sorted(lists.subjects['holdout_B_v1'])
    return groups


def check_unsealed(old_guard, old_allowlist, new_allowlist, bound, subjects, anchor, token):
    """The record branch, for subjects that are both on a sealed list and on the branch's allowlist (the record's source): the old guard admitted them on the
    allowlist alone; the new guard refuses them without the record (C23.4-2 step 3) and admits them through the allowlist bound to the record (C23.4-1)."""
    assert subjects
    for subject in subjects:
        for fixed, moving in ((subject, anchor), (anchor, subject)):
            assert outcome(lambda: old_guard.check_pair_subjects(fixed, moving, old_allowlist))[0] == 'ok', (fixed, moving)
            unbound = outcome(lambda: check_pair_subjects(fixed, moving, new_allowlist, unseal_token=token))
            assert unbound[0] == 'SubjectSealed' and 'without an unseal record' in unbound[1], (fixed, moving, unbound)
            through = outcome(lambda: check_pair_subjects(fixed, moving, bound, unseal_token=token))
            assert through[0] == 'ok' and 'sealed_guard' in json.loads(through[1]), (fixed, moving, through)
        # SG follow-up diff review n11 (A5b): the second general guard too -- check_subject_admitted refuses the sealed subject on the branch's allowlist without the record, admits it
        # through the allowlist bound to the record
        unbound = outcome(lambda: check_subject_admitted(subject, new_allowlist))
        assert unbound[0] == 'SubjectSealed' and 'not segmented' in unbound[1], (subject, unbound)
        assert outcome(lambda: check_subject_admitted(subject, bound)) == ('ok', 'null'), subject
        assert old_allowlist.allows(subject) is True and new_allowlist.allows(subject) is False and bound.allows(subject) is False, subject       # D96: allows is False for every sealed subject, bound or not


def outcome(call):
    try:
        return ('ok', json.dumps(call(), sort_keys=False, default=str))
    except Exception as exc:                          # noqa: BLE001 -- the class (by name) and the message are what is compared
        return (type(exc).__name__, str(exc))


@pytest.mark.parametrize('group', ['allowlisted', 'holdout_B_v1', 'H1', 'H4', 'H3', 'H2', 'ibsr', 'off', 'noncanonical']+(['unsealed'] if RECORD else []))
@pytest.mark.parametrize('token', [None, 'a token'])
def test_the_guards_admit_and_refuse_as_before(old, group, token):
    subjects = ids()[group]
    if RECORD and group in ('holdout_B_v1', 'unsealed'):          # review I1: on the unseal branch these are the record's subjects, on the branch's allowlist
        unseal = U.load_unseal(W.ROOT/U.UNSEAL_REL)
        check_unsealed(old.guard, old.allowlist, old.new_allowlist, U.bind_unseal(old.new_allowlist, unseal), subjects, ANCHOR, token)
        return
    if group == 'allowlisted':                  # contract C26 revision 11c (decision_log line 558): the expected tightening -- LPBA40_TEST_REJECT taken out of "as before"
        tightened = [s for s in subjects if s in LPBA40_TEST_REJECT]
        subjects = [s for s in subjects if s not in LPBA40_TEST_REJECT]
        assert len(tightened) == 22, tightened                                  # revision 11c: S03-S24 of the allowlist, refused now by step "sealed"
        for subject in tightened:
            assert (check_subject(subject, old.new_allowlist).step, check_subject(subject, old.new_allowlist).item) == ('sealed', 'lpba40 test set'), subject
            for fixed, moving in ((subject, ANCHOR), (ANCHOR, subject)):     # revision 11c (review minor 5): both directions
                assert outcome(lambda: old.guard.check_pair_subjects(fixed, moving, old.allowlist, unseal_token=token))[0] == 'ok', (fixed, moving)     # the old guard admitted them
                assert outcome(lambda: check_pair_subjects(fixed, moving, old.new_allowlist, unseal_token=token))[0] == 'SubjectSealed', (fixed, moving)
            assert outcome(lambda: check_subject_admitted(subject, old.new_allowlist))[0] == 'SubjectSealed', subject
    for subject in subjects:
        for fixed, moving in ((subject, ANCHOR), (ANCHOR, subject)):
            before = outcome(lambda: old.guard.check_pair_subjects(fixed, moving, old.allowlist, unseal_token=token))
            after = outcome(lambda: check_pair_subjects(fixed, moving, old.new_allowlist, unseal_token=token))
            assert after == before, (fixed, moving, before, after)
        assert outcome(lambda: check_subject_admitted(subject, old.new_allowlist)) == outcome(lambda: old.admitted(subject, old.allowlist)), subject
        assert old.new_allowlist.allows(subject) == old.allowlist.allows(subject), subject
    if group == 'allowlisted':
        assert all(outcome(lambda: check_pair_subjects(s, ANCHOR, old.new_allowlist))[0] == 'ok' for s in subjects)
    else:
        assert all(outcome(lambda: check_pair_subjects(s, ANCHOR, old.new_allowlist))[0] == 'SubjectSealed' for s in subjects)


@pytest.mark.parametrize('token', [None, 'a token'])
def test_the_record_branch_on_the_synthetic_unseal_branch(old, tmp_path, monkeypatch, token):
    """Review I1: check_unsealed (what the test above runs on the dedicated unseal branch) on _sg_world's synthetic branch -- its allowlist holds the synthetic
    holdout_B_v1 as the record's source, the real loader issues the record, bind_unseal binds it; the old guard is c027447's on the same allowlist file."""
    w = W.make_world(tmp_path, monkeypatch)
    unseal, bound = W.bound(w)
    old_allowlist = old.guard.load_dev_allowlist(w.repo/U.ALLOWLIST_REL)
    check_unsealed(old.guard, old_allowlist, w.allowlist, bound, sorted(W.HOLDOUT), W.DEV[0], token)
    assert sorted(unseal.subjects) == sorted(W.HOLDOUT)


def test_the_old_guard_is_the_one_before_sg():
    text = git_show('bayesreg/io/guard.py')
    assert 'check_subject(' not in text and 'unseal' not in text.replace('unseal_token', '').replace('Unsealing', '').replace('unsealing', '').replace('unsealed', '')
    assert subprocess.run(['git', '-C', str(W.ROOT), 'diff', '--quiet', OLD, '854c7080e12cbde577a549d84a6a5613dbdfd880', '--', 'bayesreg', 'jd_curl', 'experiments/hs', 'experiments/fl/nx_seg.py',
                           'experiments/ef/fast_channels.py'], capture_output=True).returncode == 0


def test_check_subject_admitted_refuses_an_unbound_sealed_subject_and_admits_a_bound_one(tmp_path, monkeypatch):
    """SG follow-up diff review n11 (A5b): what check_unsealed adds on the unseal branch, run here on the synthetic tree of _sg_world (main has no record): the second general guard,
    check_subject_admitted, refuses each subject of the record on the allowlist alone and admits it through the allowlist bound to the record."""
    w = W.make_world(tmp_path, monkeypatch)
    _, bound_allowlist = W.bound(w)
    for subject in W.HOLDOUT:
        unbound = outcome(lambda: check_subject_admitted(subject, w.allowlist))
        assert unbound[0] == 'SubjectSealed' and 'not segmented' in unbound[1], (subject, unbound)
        assert outcome(lambda: check_subject_admitted(subject, bound_allowlist)) == ('ok', 'null'), subject
