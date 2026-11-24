"""A5a (runs/20261110-a5/任务书.md supplements 2-6; B4 review docs/bayesreg/reviews/开发集核对计划_复核_20261110.md items 1 and 2 I1, §1.3; contract C24.3 (revision 9 draft): rule one, rule two):
  - the two decision constants of HP carry code_commit, the commit the development check ran on (the record's code and its lock line's), and hp_core.check_decision applies rule two from it
    (bayesreg.io.code_identity.gate_identity_problems: only the four decision constants' assignments and the allowlist's inclusion may change after it);
  - the constants can be filled as plain dict literals (C24.3-6 (a): rule two takes only a literal; decision_log 2026-11-10 ~3:10 AM CST (1)) and are read through a read-only copy;
  - a hold-out run of HP checks rule one in main (hp_core.check_pinned: P1 from the committed runs/holdout_B_v1/pinned_code.json); run_holdout refuses a code record without it; a dry run records
    the problems and goes on.
Temporary git repositories; the HP sandbox (conftest) replaces both identities by default, these tests call the real ones (W.REAL_CORE)."""
import json
from pathlib import Path
import subprocess
from types import MappingProxyType

import pytest

from bayesreg.io import code_identity as ci

from .. import _p1_decisions as P1
from . import _world as W

hc, core, prep, ah = W.hc, W.core, W.prep, W.ah
REAL_GATE, REAL_PINNED = W.REAL_CORE['gate_identity'], W.REAL_CORE['pinned_identity']


def run_prep(w, **kwargs):
    return prep.run_holdout(unseal_path=kwargs.pop('unseal_path', w.record_path), code=kwargs.pop('code', W.CODE), log=W.quiet, **kwargs)


# ---- the decision's code_commit ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('case', ['not_hex', 'record_code', 'record_not_clean', 'lock_line', 'rule_two', 'missing_key'])
def test_a_decision_whose_code_is_not_its_checks_is_refused(tmp_path, monkeypatch, case):
    w = W.make_world(tmp_path, monkeypatch)
    decision = dict(w.decisions['prep'])
    if case == 'not_hex':
        decision['code_commit'] = 'HEAD'
    elif case == 'record_code':
        decision['code_commit'] = 'e'*40                                           # the record and the lock line name 'd'*40
    elif case == 'record_not_clean':
        path = W.fake_devcheck(prep, reading=decision['reading'], record_extra=dict(n=len(hc.DEV_PAIRS), n_equal=len(hc.DEV_PAIRS)),
                               code=dict(W.DEV_CODE, check=dict(problems=['code at x is not a clean git worktree'])))
        decision = W.decision_for(path, reading=decision['reading'])
    elif case == 'lock_line':
        line = json.loads(prep.devcheck_lock().read_text().splitlines()[0])
        path = W.fake_devcheck(prep, reading=decision['reading'], record_extra=dict(n=len(hc.DEV_PAIRS), n_equal=len(hc.DEV_PAIRS)))
        prep.devcheck_lock().write_text(json.dumps(dict(line, code_commit='e'*40))+'\n')
        doc = json.loads(path.read_text())
        doc['lock_line'] = dict(line, code_commit='e'*40)                          # the record agrees with its lock; both name another commit than the decision
        W.write_json(path, doc)
        decision = W.decision_for(path, reading=decision['reading'])
    elif case == 'rule_two':
        monkeypatch.setattr(core, 'gate_identity', lambda commit, **kw: (dict(passed=False), ['M experiments/holdout/hp_core.py: not one of the decision constants\' files']))
    else:
        decision.pop('code_commit')
    monkeypatch.setattr(prep, 'PREP_DEVCHECK_DECISION', decision)
    with pytest.raises(hc.Refused) as info:
        run_prep(w)
    assert info.value.stage == 'decision'
    word = dict(not_hex='40 lowercase hex', record_code='ran on the code', record_not_clean='not clean', lock_line='lock line', rule_two='rule two', missing_key='exactly the keys')[case]
    assert word in str(info.value), str(info.value)
    assert not Path(hc.STAGE_ROOT).exists()


def test_a_decision_that_holds_names_its_commit_to_rule_two_and_is_read_through_a_copy(tmp_path, monkeypatch, hp_sandbox):
    w = W.make_world(tmp_path, monkeypatch)
    checked = prep.check_decision()
    assert checked['code_commit'] == W.DEV_CODE['commit'] and hp_sandbox.gates == [W.DEV_CODE['commit']] and checked['code_identity']['gate'] == W.DEV_CODE['commit']
    holder = dict(w.decisions['atropos'])
    monkeypatch.setattr(ah, 'ATROPOS_DEVCHECK_DECISION', holder)
    assert ah.check_decision()['code_commit'] == holder['code_commit']
    reading = holder['reading']
    monkeypatch.setattr(core, 'gate_identity', lambda commit, **kw: (holder.__setitem__('reading', 'changed during the check'), (dict(passed=True), []))[1])
    assert ah.check_decision()['reading'] == reading                                # the reader works on its own read-only copy (C24.3-6 (a): the constant is a plain literal)
    with pytest.raises(TypeError):
        MappingProxyType(dict(holder))['reading'] = 'x'


# ---- P1: the two constants as filled (runs/20261115-p1/实现说明_P1.md) ---------------------------------------------------------------------------------
HP_DECISIONS = dict(prep=(prep, 'PREP_DEVCHECK_DECISION', 'experiments/holdout/prep_stage.py', P1.PREP, P1.PREP_LOCK),
                    atropos=(ah, 'ATROPOS_DEVCHECK_DECISION', 'experiments/holdout/atropos_holdout.py', P1.ATROPOS, P1.ATROPOS_LOCK))


@pytest.mark.parametrize('module', ['prep', 'atropos'])
def test_the_p1_decision_is_the_development_check_that_ran_on_p0pp(module):
    """P1 (prereg revision 2 item 9 step 5; runs/20261115-d58/result_C1.md, result_C2.md): the constant is a plain dict literal with exactly hp_core.DECISION_KEYS and the values of the check
    of 2026-11-15 on P0'' (written again in tests/test_bayesreg/_p1_decisions.py): the tool's pinned plan, the reading the tool accepts without BZ's words (bz_words None), the record and the
    lock at the tool's fixed paths under the real results root; the file is the one of P0'' but for this constant (rule two's AST check on the file)."""
    mod, name, rel, values, (lock, _) = HP_DECISIONS[module]
    decision = getattr(mod, name)
    P1.assert_values(decision, core.DECISION_KEYS, values)
    plan, reading = ((prep.PREP_DEVCHECK_PLAN_SHA256, prep.READ_EQUAL.format(n=len(hc.DEV_PAIRS))) if mod is prep else
                     (ah.ATROPOS_DEVCHECK_PLAN_SHA256, ah.READ_EQUAL.format(n=len(ah.DEV_SUBJECTS))))
    assert decision['plan_sha256'] == plan and decision['reading'] == reading and decision['bz_words'] is None
    real = Path(hc.REAL_RESULTS)/core.DEVCHECK_DIR
    assert Path(decision['devcheck_json']) == real/module/mod.DEVCHECK_RECORD and lock == real/'locks'/f'{module}_devcheck.lock'
    now, gate = (W.ROOT/rel).read_text(encoding='utf-8'), P1.blob_at(P1.P0PP, rel)
    assert P1.at_gate(now, name) == gate and ci.decision_assignment_problem(gate, now, rel) is None


@pytest.mark.parametrize('module', ['prep', 'atropos'])
def test_the_p1_decision_holds_against_the_real_development_check(monkeypatch, hp_sandbox, module):
    """P1: the constant passes the tool's own check_decision against the real record and lock (read only: their sha256 before and after; the results root set back to the real one for this
    read, nothing written), the record's reading, plan, code (commit P0'', clean) and lock line those of the constant; rule two is asked about P0'' (replaced here by the sandbox's recorder:
    the range P0''..HEAD is tested in tests/test_bayesreg/io_seg/test_code_identity.py)."""
    mod, name, _, values, (lock, lock_sha) = HP_DECISIONS[module]
    before = P1.real_evidence(values['devcheck_json'], lock)
    assert before == (values['devcheck_sha256'], lock_sha)
    record = json.loads(Path(values['devcheck_json']).read_text(encoding='utf-8'))
    assert record['reading'] == values['reading'] and record['plan']['sha256'] == values['plan_sha256'] and record['schema'] == mod.SCHEMA+'.devcheck'
    assert record['code']['commit'] == P1.P0PP and record['code']['check']['problems'] == [] and record['lock_line'] == P1.lock_line(lock) and record['lock_line']['code_commit'] == P1.P0PP
    monkeypatch.setattr(hc, 'RESULTS', hc.REAL_RESULTS)
    checked = mod.check_decision()
    assert hp_sandbox.gates == [P1.P0PP] and checked['reading'] == values['reading'] and checked['code_commit'] == P1.P0PP
    assert (P1.sha256_of(values['devcheck_json']), P1.sha256_of(lock)) == before


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True, env=W.GIT_ENV)


def gate_repo(tmp_path):
    """A temporary code tree holding the real prep_stage.py and atropos_holdout.py (their bytes as at the gate: P1 filled the two constants, P1.at_gate sets them back to None) and one other
    code file; its first commit is the gate's (P0)."""
    repo = tmp_path/'tree'
    for rel, name in (('experiments/holdout/prep_stage.py', 'PREP_DEVCHECK_DECISION'), ('experiments/holdout/atropos_holdout.py', 'ATROPOS_DEVCHECK_DECISION')):
        W.write_text(repo/rel, P1.at_gate((W.ROOT/rel).read_text(encoding='utf-8'), name))
    W.write_text(repo/'experiments/holdout/hp_core.py', 'X = 1\n')
    W.write_text(repo/'docs/decision_log.md', 'log\n')
    git(repo, 'init', '-q')
    W.commit_all(repo, 'P0')
    return repo, git(repo, 'rev-parse', 'HEAD').stdout.strip()


def literal(keys=None, **values):
    """A decision constant as the filling commit writes it: a dict literal over several lines (C24.3-6 (a))."""
    keys = core.DECISION_KEYS if keys is None else keys
    body = ''.join(f'    {k!r}: {values.get(k, "x" if k != "bz_words" else None)!r},\n' for k in keys)
    return '{\n'+body+'}'


def fill(repo, rel, name, value):
    path = repo/rel
    text = path.read_text(encoding='utf-8')
    assert text.count(f'\n{name} = None\n') == 1
    path.write_text(text.replace(f'\n{name} = None\n', f'\n{name} = {value}\n'), encoding='utf-8')


def test_rule_two_from_the_checks_commit_on_a_real_repository(tmp_path):
    """The real hp_core.gate_identity (bayesreg.io.code_identity.gate_identity_problems) on a temporary tree: filling both constants with dict literals after the gate passes; any other change of the
    code (hp_core.py), a constant written as a call (MappingProxyType(...)), or a statement smuggled after the literal is refused."""
    repo, p0 = gate_repo(tmp_path)
    fill(repo, 'experiments/holdout/prep_stage.py', 'PREP_DEVCHECK_DECISION', literal(code_commit=p0))
    fill(repo, 'experiments/holdout/atropos_holdout.py', 'ATROPOS_DEVCHECK_DECISION', literal(code_commit=p0))
    W.commit_all(repo, 'P1: the decision constants')
    record, problems = REAL_GATE(p0, repo=repo)
    assert problems == [] and record['gate'] == p0 and [c['kind'] for c in record['changes']] == ['decision_constants', 'decision_constants']
    W.write_text(repo/'experiments/holdout/hp_core.py', 'X = 2\n')
    W.commit_all(repo, 'another code change')
    record, problems = REAL_GATE(p0, repo=repo)
    assert problems and any('hp_core.py' in p for p in problems)
    repo2, q0 = gate_repo(tmp_path/'second')
    fill(repo2, 'experiments/holdout/prep_stage.py', 'PREP_DEVCHECK_DECISION', f'MappingProxyType(dict({literal(code_commit=q0)}))')
    W.commit_all(repo2, 'a call, not a literal')
    assert REAL_GATE(q0, repo=repo2)[1]
    repo3, r0 = gate_repo(tmp_path/'third')
    fill(repo3, 'experiments/holdout/prep_stage.py', 'PREP_DEVCHECK_DECISION', literal(code_commit=r0)+'; import os')
    W.commit_all(repo3, 'a statement after the literal')
    assert REAL_GATE(r0, repo=repo3)[1]


def test_rule_two_runs_inside_the_hold_out_runs_spy(tmp_path, monkeypatch):
    """check_decision runs under the hold-out run's spy (hp_core.Spy): the real rule two (git only) opens nothing the spy refuses; the decision whose code_commit is the gate of a temporary tree
    where only the constants changed lets the run go through, and the same decision against a tree with another code change stops it (stage decision)."""
    w = W.make_world(tmp_path, monkeypatch)
    repo, p0 = gate_repo(tmp_path/'gate')
    fill(repo, 'experiments/holdout/prep_stage.py', 'PREP_DEVCHECK_DECISION', literal(code_commit=p0))
    W.commit_all(repo, 'P1')
    monkeypatch.setattr(core, 'gate_identity', lambda commit, **kw: REAL_GATE(commit, repo=repo))
    code = dict(W.DEV_CODE, commit=p0)
    path = W.fake_devcheck(prep, reading=prep.READ_EQUAL.format(n=len(hc.DEV_PAIRS)), record_extra=dict(n=len(hc.DEV_PAIRS), n_equal=len(hc.DEV_PAIRS)), code=code)
    monkeypatch.setattr(prep, 'PREP_DEVCHECK_DECISION', W.decision_for(path, reading=prep.READ_EQUAL.format(n=len(hc.DEV_PAIRS)), code_commit=p0))
    W.S._reset_cache()
    record = run_prep(w)
    assert record['passed'] and record['decision']['code_identity']['passed'] is True and record['decision']['code_commit'] == p0
    assert not record['spy']['breaches']
    W.write_text(repo/'experiments/holdout/hp_core.py', 'X = 3\n')
    W.commit_all(repo, 'a code change after the gate')
    import shutil
    shutil.rmtree(hc.STAGE_ROOT)
    shutil.rmtree(hc.LOCK_ROOT)
    W.S._reset_cache()
    with pytest.raises(hc.Refused) as info:
        run_prep(w)
    assert info.value.stage == 'decision' and 'rule two' in str(info.value)


@pytest.mark.parametrize('rel, name', [('experiments/holdout/prep_stage.py', 'PREP_DEVCHECK_DECISION'), ('experiments/holdout/atropos_holdout.py', 'ATROPOS_DEVCHECK_DECISION')])
def test_the_constants_can_be_filled_as_plain_dict_literals(rel, name):
    """C24.3-6 (a) on the files as at the gate (P1 filled the constants: P1.at_gate sets them back to None, P0's bytes): replacing `NAME = None` by a dict literal with exactly
    hp_core.DECISION_KEYS (code_commit among them) is a change rule two allows (decision_assignment_problem with the constant's own file); the literal is what check_decision's key check
    takes.  P1: the file as it is now is such a change."""
    now = (W.ROOT/rel).read_text(encoding='utf-8')
    old = P1.at_gate(now, name)
    assert ci.decision_assignment_problem(old, now, rel) is None
    assert f'\n{name} = None\n' in old and name in ci.DECISION_CONSTANTS and ci.DECISION_HOMES[name] == rel
    value = literal(code_commit='d'*40)
    new = old.replace(f'\n{name} = None\n', f'\n{name} = {value}\n')
    assert ci.decision_assignment_problem(old, new, rel) is None
    namespace = {}
    exec(f'{name} = {value}', namespace)                    # noqa: S102 - the literal itself, nothing else
    assert sorted(namespace[name]) == sorted(core.DECISION_KEYS) and 'code_commit' in core.DECISION_KEYS
    assert ci.decision_assignment_problem(old, old.replace(f'\n{name} = None\n', f'\n{name} = dict(a=1)\n'), rel) is not None


# ---- rule one of a hold-out run ----------------------------------------------------------------------------------------------------------------------
def test_a_holdout_run_without_rule_one_is_refused_before_anything(tmp_path, monkeypatch):
    """run_holdout takes only a code record whose rule-one check passed and names its commit as P1; main refuses a real run whose check fails (stage code) and records it for a dry run."""
    w = W.make_world(tmp_path, monkeypatch)
    for code in (dict(W.CODE, pinned_code=None), dict(W.CODE, pinned_code=dict(passed=False, pinned=None, problems=['no record'])),
                 dict(W.CODE, commit='e'*40), {k: v for k, v in W.CODE.items() if k != 'pinned_code'}):
        with pytest.raises(hc.Refused) as info:
            run_prep(w, code=code)
        assert info.value.stage == 'code'
        with pytest.raises(hc.Refused) as info:
            ah.run_holdout(unseal_path=w.record_path, code=code, log=W.quiet)
        assert info.value.stage == 'code'
    assert not Path(hc.STAGE_ROOT).exists() and not Path(hc.LOCK_ROOT).exists()
    monkeypatch.setattr(core, 'check_code', lambda code, require_clean=True: dict(checked='test', problems=[]))
    monkeypatch.setattr(core, 'pinned_identity', lambda: (dict(passed=False, pinned=None), ['runs/holdout_B_v1/pinned_code.json is not committed at HEAD (test)']))
    said = []
    assert prep.main(['run', '--unseal', str(w.record_path)], code=dict(W.CODE, root=str(W.ROOT)), out=said.append) == 8
    assert 'refused (code)' in said[-1] and 'rule one' in said[-1]
    assert ah.main(['run', '--unseal', str(w.record_path)], code=dict(W.CODE, root=str(W.ROOT)), out=said.append) == 8 and 'rule one' in said[-1]
    out = tmp_path/'dry'
    code = prep.main(['run', '--dry-run', '--out', str(out), '--unseal', str(w.record_path)], code=dict(W.CODE, root=str(W.ROOT)), out=said.append)
    doc = json.loads((out/'dry_run.json').read_text())
    assert code == 0 and doc['code']['pinned_code']['problems'] and doc['code']['commit'] == W.CODE['commit'] and doc['admission']        # recorded; the rest checked
    assert [e['kind'] for e in doc['expected']] == ['rule_one_unpinned'] and doc['unexpected'] == [] and doc['p0_ok'] is True and doc['passed'] is False    # A5a review m8: the record, not the exit code
    assert not Path(hc.STAGE_ROOT).exists()


UNPINNED = 'runs/holdout_B_v1/pinned_code.json is not committed at HEAD (test): no pinned commit'


def checked(*problems):
    return lambda code, require_clean=True: dict(checked='test', problems=list(problems))


@pytest.mark.parametrize('module', ['prep', 'atropos'])
def test_the_dry_run_record_on_p0_holds_the_code_check_and_the_pending_decision(tmp_path, monkeypatch, module):
    """A5a diff review n1: on the real pins (no unseal record: the dry run stops at the unseal, as on P0) with the decision constant None, the record of `run --dry-run` holds rule one without the
    pinned record, no_unseal_record and decision_pending -- p0_ok; the same on a checkout whose code check has a problem (an unclean tree) holds it as kind code, unexpected -- not p0_ok."""
    mod = prep if module == 'prep' else ah
    W.real_pins(monkeypatch)
    monkeypatch.setattr(mod, 'PREP_DEVCHECK_DECISION' if mod is prep else 'ATROPOS_DEVCHECK_DECISION', None)
    monkeypatch.setattr(core, 'pinned_identity', lambda: (dict(passed=False, pinned=None), [UNPINNED]))
    monkeypatch.setattr(core, 'check_code', checked())
    out = tmp_path/'clean'
    assert mod.main(['run', '--dry-run', '--out', str(out)], code=dict(W.CODE, root=str(W.ROOT)), out=W.quiet) == 8
    doc = json.loads((out/'dry_run.json').read_text())
    assert doc['refused']['stage'] == 'unseal' and sorted(e['kind'] for e in doc['expected']) == ['decision_pending', 'no_unseal_record', 'rule_one_unpinned']
    assert doc['unexpected'] == [] and doc['p0_ok'] is True and doc['passed'] is False
    monkeypatch.setattr(core, 'check_code', checked('the code tree is not clean (test)'))
    out = tmp_path/'dirty'
    W.S._reset_cache()                              # a dry run is a process of its own: the guard's cache starts empty (C23.6-5)
    assert mod.main(['run', '--dry-run', '--out', str(out)], code=dict(W.CODE, root=str(W.ROOT)), out=W.quiet) == 8
    doc = json.loads((out/'dry_run.json').read_text())
    assert [e['kind'] for e in doc['unexpected']] == ['code'] and 'not clean' in doc['unexpected'][0]['problem'] and doc['p0_ok'] is False
    assert sorted(e['kind'] for e in doc['expected']) == ['decision_pending', 'no_unseal_record', 'rule_one_unpinned']


@pytest.mark.parametrize('module', ['prep', 'atropos'])
@pytest.mark.parametrize('decision', ['passes', 'refused'])
def test_the_dry_run_record_on_p1_holds_no_pending_decision(tmp_path, monkeypatch, module, decision):
    """P1 review minor 3: on the real pins (no unseal record: the dry run stops at the unseal) with the decision constant as P1 filled it (the module's own value, not replaced), the record of
    `run --dry-run` holds rule one without the pinned record and no_unseal_record, and no decision_pending -- p0_ok.  So whether the tool's check_decision passes (as on the P1 checkout, where it
    reads the real development check: test_the_p1_decision_holds_against_the_real_development_check; stood in here) or refuses (as here, unreplaced: the sandbox's results root is not the
    record's): a dry run stopped at the unseal records the decision by decision_none only (prep_stage.main, atropos_holdout.main), never the run's record['decision']."""
    mod, name, _, values, _ = HP_DECISIONS[module]
    filled = getattr(mod, name)
    assert isinstance(filled, dict) and filled['devcheck_sha256'] == values['devcheck_sha256']        # the P1 state: the constant as filled
    W.real_pins(monkeypatch)
    monkeypatch.setattr(core, 'pinned_identity', lambda: (dict(passed=False, pinned=None), [UNPINNED]))
    monkeypatch.setattr(core, 'check_code', checked())
    calls, original = [], mod.check_decision

    def check_decision():
        if decision == 'passes':
            calls.append('passed')
            return dict(filled, code_identity=dict(stub='test'))
        try:
            return original()
        except hc.Refused as exc:
            calls.append(exc.stage)
            raise
    monkeypatch.setattr(mod, 'check_decision', check_decision)
    out = tmp_path/'p1'
    assert mod.main(['run', '--dry-run', '--out', str(out)], code=dict(W.CODE, root=str(W.ROOT)), out=W.quiet) == 8
    doc = json.loads((out/'dry_run.json').read_text())
    assert calls == ['passed' if decision == 'passes' else 'decision']                                 # the run asked its decision first, under the dry run
    assert doc['refused']['stage'] == 'unseal' and sorted(e['kind'] for e in doc['expected']) == ['no_unseal_record', 'rule_one_unpinned']
    assert doc['unexpected'] == [] and doc['p0_ok'] is True and doc['passed'] is False and 'decision' not in doc
    assert not any(e['kind'].startswith('decision') for e in doc['expected']+doc['unexpected'])


def test_the_dry_run_record_past_the_admission_holds_the_code_check(tmp_path, monkeypatch):
    """A5a diff review n1: a dry run that passes the admission (a record, the decision filled) on a checkout whose code check has a problem is not p0_ok (kind code); with a clean one it is."""
    w = W.make_world(tmp_path, monkeypatch)
    monkeypatch.setattr(core, 'pinned_identity', lambda: (dict(passed=False, pinned=None), [UNPINNED]))
    for problems, ok in (((), True), (('bayesreg is not imported from the code root (test)',), False)):
        monkeypatch.setattr(core, 'check_code', checked(*problems))
        out = tmp_path/f'dry_{ok}'
        W.S._reset_cache()
        assert prep.main(['run', '--dry-run', '--out', str(out), '--unseal', str(w.record_path)], code=dict(W.CODE, root=str(W.ROOT)), out=W.quiet) == 0
        doc = json.loads((out/'dry_run.json').read_text())
        assert doc['p0_ok'] is ok and [e['kind'] for e in doc['unexpected']] == ([] if ok else ['code']) and [e['kind'] for e in doc['expected']] == ['rule_one_unpinned']


def test_dry_run_fields_takes_decision_none_only_as_given():
    """A5a diff review n1: the pending decision of a stopped dry run comes from decision_none, which main gives for `run` only (devcheck does not read the constant); a decision in the record wins."""
    code = dict(W.CODE, pinned_code=dict(passed=True, pinned='c'*40, problems=[]))
    kinds = lambda fields: [e['kind'] for e in fields['expected']+fields['unexpected']]       # noqa: E731
    assert kinds(core.dry_run_fields(code, refusal=('spy', 'x'), decision_none=False)) == ['refused_spy']
    assert kinds(core.dry_run_fields(code, refusal=('spy', 'x'), decision_none=True)) == ['decision_pending', 'refused_spy']
    assert kinds(core.dry_run_fields(code, record=dict(decision=dict(devcheck_sha256='d'*64)), decision_none=True)) == []
    assert kinds(core.dry_run_fields(code, record=dict(decision=dict(refused='None')), decision_none=True)) == ['decision_pending']
    assert kinds(core.dry_run_fields(code, code_check=dict(problems=['x']))) == ['code'] and kinds(core.dry_run_fields(code, code_check=dict(problems=[]))) == []


def test_check_pinned_names_p1_and_keeps_head():
    good = core.check_pinned(dict(commit='a'*40, clean=True), identity=lambda: (dict(passed=True, pinned='b'*40), []))
    assert good['commit'] == 'b'*40 and good['head'] == 'a'*40 and core.pinned_code_problem(good) is None
    with pytest.raises(hc.Refused, match='rule one'):
        core.check_pinned(dict(commit='a'*40), identity=lambda: (dict(passed=False, pinned=None), ['x']))
    dry = core.check_pinned(dict(commit='a'*40), dry_run=True, identity=lambda: (dict(passed=False, pinned=None), ['x']))
    assert dry['commit'] == 'a'*40 and dry['pinned_code']['problems'] == ['x'] and core.pinned_code_problem(dry)


def test_the_real_rule_one_refuses_this_tree_without_a_pinned_record():
    """main has no runs/holdout_B_v1/pinned_code.json: the real check (git of this code tree, read only) gives the expected refusal, never HEAD by default."""
    if (W.ROOT/ci.PINNED_RECORD_REL).exists():
        pytest.skip('a tree with a pinned record (the dedicated branch)')
    record, problems = REAL_PINNED()
    assert problems and record['pinned'] is None and any('never HEAD' in p for p in problems)
