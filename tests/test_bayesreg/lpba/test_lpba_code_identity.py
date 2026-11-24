"""LP-C2 item 4 (contract C26.9, C26.10 T13; D148, D149): rule one with the LPBA40 constants (bayesreg.io.code_identity.LPBA40) and the choice of P_L (pl_choice_problems), every
branch on a temporary git repository (git's global and system configuration kept out).  The hold-out's rule one (profile None) is tested where it was (holdout/test_pinned_code.py,
io_seg/test_code_identity.py) and must keep passing unchanged."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from bayesreg.io import code_identity as ci

ROOT = Path(__file__).resolve().parents[3]
GIT_ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_AUTHOR_NAME='lp-test', GIT_AUTHOR_EMAIL='lp-test@example.invalid', GIT_COMMITTER_NAME='lp-test',
               GIT_COMMITTER_EMAIL='lp-test@example.invalid')
CODE = ('bayesreg/a.py', 'experiments/lpba/x.py', 'jd_curl/b.py', 'configs/bayesreg/lpba40_test_v1_allowlist.json', 'configs/bayesreg/dev_subjects.json', 'scripts/d.py')
OTHER = ('splits/lpba40_test_v1/test_pairs.tsv', 'runs/lpba40_test_v1/README.md', 'docs/decision_log.md', 'docs/other.md', 'README.md')
RECORD = ci.LPBA40_PINNED_RECORD_REL


def git(r, *args):
    return subprocess.run(['git', '-C', str(r), *args], check=True, capture_output=True, text=True, env=GIT_ENV)


def write(r, rel, text='x\n'):
    (r/rel).parent.mkdir(parents=True, exist_ok=True)
    (r/rel).write_text(text, encoding='utf-8')


def commit(r, message='later'):
    git(r, 'add', '-A')
    git(r, 'commit', '-q', '-m', message)


def head(r):
    return git(r, 'rev-parse', 'HEAD').stdout.strip()


def pin_record(r, pin, *, log='same', **override):
    """runs/lpba40_test_v1/pinned_code.json and its decision log line in one commit (C26.9-5; C24.3-9)."""
    line = override.pop('line', f'| 2026-11-20 | P_L = {pin} (test) |')
    doc = dict(schema=ci.LPBA40_PINNED_RECORD_SCHEMA, commit=pin, decision_log_line=line, written_by="lead on BZ's behalf", time_local='2026-11-20 1:00 AM CST',
               utc='2026-11-20T07:00:00Z', reason='test: P_L after the feasibility pairs')
    doc.update(override)
    write(r, RECORD, json.dumps(doc))
    if log == 'same':
        with open(r/'docs/decision_log.md', 'a', encoding='utf-8') as stream:
            stream.write(line+'\n')
    commit(r, 'P_L')
    if log == 'later':
        with open(r/'docs/decision_log.md', 'a', encoding='utf-8') as stream:
            stream.write(line+'\n')
        commit(r, 'the line after')


@pytest.fixture
def repo(tmp_path):
    r = tmp_path/'code'
    for rel in CODE+OTHER:
        write(r, rel, rel+'\n')
    git(r, 'init', '-q')
    commit(r, 'the feasibility commit')
    pin = head(r)
    pin_record(r, pin)
    return r, pin


def rule(r, commit=None):
    record, problems = ci.unit_identity_problems(commit, repo=r, profile=ci.LPBA40)
    assert record['passed'] is (not problems) and record['problems'] == problems and record['profile'] == 'lpba40_test_v1'
    return record, problems


def test_t13_the_constants():
    assert ci.LPBA40.pinned_record_rel == 'runs/lpba40_test_v1/pinned_code.json' and ci.LPBA40.later_records == 'runs/lpba40_test_v1/'
    assert ci.LPBA40.allowlist_exception is False and ci.holdout_profile().allowlist_exception is True
    assert ci.holdout_profile().pinned_record_rel == ci.PINNED_RECORD_REL == 'runs/holdout_B_v1/pinned_code.json'      # the hold-out's unchanged
    for name in ('splits/lpba40_test_v1/lpba40_test_v1_hb_resume_1.json', 'splits/lpba40_test_v1/lpba40_test_v1_xb_no_resume_12.json',
                 'splits/lpba40_test_v1/lpba40_test_v1_hp_prep_rerun_3.json', 'splits/lpba40_test_v1/lpba40_test_v1_reeval_2.json'):
        assert ci.LPBA40_LATER_AUTHORIZATION.fullmatch(name), name
        assert not ci.LATER_AUTHORIZATION.fullmatch(name)
    for name in ('splits/lpba40_test_v1/test_pairs.tsv', 'splits/bayesreg_v1/holdout_B_v1_hb_resume_1.json', 'splits/lpba40_test_v1/lpba40_test_v1_hb_resume_1234.json',
                 'splits/lpba40_test_v1/lpba40_test_v1_hb_other_1.json'):
        assert not ci.LPBA40_LATER_AUTHORIZATION.fullmatch(name), name
    sys.path.insert(0, str(ROOT/'experiments'/'holdout'))
    import holdout_common as hc
    for tool in ('hb', 'xb', 'hp_prep'):
        for kind in hc.AUTH_KINDS:
            assert ci.LPBA40_LATER_AUTHORIZATION.fullmatch(f"{hc.AUTH_DIRS['lpba40_test']}/{hc.AUTH_PREFIX['lpba40_test']}{tool}_{kind}_7.json")
    from bayesreg.eval import lpba_eval as E
    assert ci.LPBA40_LATER_AUTHORIZATION.fullmatch(f'{E.REEVAL_DIR_REL}/lpba40_test_v1_reeval_5.json') and E.REEVAL_NAME.fullmatch('lpba40_test_v1_reeval_5.json')
    assert E.MANIFEST_REL.startswith(ci.LPBA40_LATER_RECORDS) and E.RECORDS_REL.startswith(ci.LPBA40_LATER_RECORDS)


def test_t13_rule_one_passes_and_takes_p_l_from_the_record(repo):
    r, pin = repo
    record, problems = rule(r)
    assert problems == [] and record['pinned'] == pin
    write(r, 'splits/lpba40_test_v1/lpba40_test_v1_hb_resume_1.json', '{}\n')
    write(r, 'runs/lpba40_test_v1/eval_manifest.txt', 'm\n')
    with open(r/'docs/decision_log.md', 'a', encoding='utf-8') as stream:
        stream.write('| appended |\n')
    commit(r, 'an authorization, a record, an append')
    assert rule(r)[1] == []
    assert rule(r, pin)[1] == []


def refusal(r, pin, case):
    """One change of the range (or of the tree) that rule one refuses; returns the expected words."""
    if case == 'record_missing':
        git(r, 'rm', '-q', RECORD)
        commit(r, 'record removed')
        return 'not committed'
    if case == 'record_changed':
        doc = json.loads((r/RECORD).read_text())
        write(r, RECORD, json.dumps(dict(doc, reason='changed')))
        commit(r, 'changed')
        return 'only added, once'
    if case == 'record_keys':
        write(r, RECORD, json.dumps(dict(schema=ci.LPBA40_PINNED_RECORD_SCHEMA, commit=pin)))
        commit(r, 'keys')
        return 'exactly the keys'
    if case == 'record_holdout_schema':
        doc = json.loads((r/RECORD).read_text())
        write(r, RECORD, json.dumps(dict(doc, schema=ci.PINNED_RECORD_SCHEMA)))
        commit(r, 'schema')
        return 'schema'
    if case == 'no_log_line':
        git(r, 'checkout', '-q', '-b', 'side')
        write(r, 'docs/decision_log.md', 'rewritten\n')
        commit(r, 'log rewritten')
        return 'not a line of'
    if case == 'log_old_line_changed':
        text = (r/'docs/decision_log.md').read_text()
        write(r, 'docs/decision_log.md', 'X'+text)
        commit(r, 'old line changed')
        return 'not an append'
    if case == 'given_commit':
        return None
    if case == 'merge':
        base = head(r)
        git(r, 'checkout', '-q', '-b', 'side')
        write(r, 'runs/lpba40_test_v1/a.md', 'a\n')
        commit(r, 'side')
        git(r, 'checkout', '-q', '-')
        write(r, 'runs/lpba40_test_v1/b.md', 'b\n')
        commit(r, 'main')
        git(r, 'merge', '-q', '--no-ff', '-m', 'merge', 'side')
        return 'merge commit'
    if case == 'code_changed':
        write(r, 'experiments/lpba/x.py', 'changed\n')
        commit(r, 'code')
        return 'the code changed'
    if case == 'allowlist_changed':
        write(r, 'configs/bayesreg/dev_subjects.json', '{"x": 1}\n')
        commit(r, 'allowlist')
        return 'the code changed'
    if case == 'authorization_changed':
        write(r, 'splits/lpba40_test_v1/lpba40_test_v1_xb_resume_1.json', '{}\n')
        commit(r, 'added')
        write(r, 'splits/lpba40_test_v1/lpba40_test_v1_xb_resume_1.json', '{"x": 1}\n')
        commit(r, 'changed')
        return 'only added, once'
    if case == 'record_deleted':
        write(r, 'runs/lpba40_test_v1/c.md', 'c\n')
        commit(r, 'added')
        git(r, 'rm', '-q', 'runs/lpba40_test_v1/c.md')
        commit(r, 'deleted')
        return 'only added, once'
    if case == 'holdout_authorization':
        write(r, 'splits/bayesreg_v1/holdout_B_v1_hb_resume_1.json', '{}\n')
        commit(r, 'the hold-out kind')
        return 'not an authorization under splits/lpba40_test_v1/'
    if case == 'other_file':
        write(r, 'docs/other.md', 'changed\n')
        commit(r, 'docs')
        return 'not an authorization'
    if case == 'dirty':
        write(r, 'README.md', 'dirty\n')
        return 'tracked files modified'
    if case == 'untracked_code':
        write(r, 'bayesreg/new.py', 'x = 1\n')
        return 'untracked or ignored files in the code directories'
    raise AssertionError(case)


CASES = ('record_missing', 'record_changed', 'record_keys', 'record_holdout_schema', 'no_log_line', 'log_old_line_changed', 'given_commit', 'merge', 'code_changed',
         'allowlist_changed', 'authorization_changed', 'record_deleted', 'holdout_authorization', 'other_file', 'dirty', 'untracked_code')


@pytest.mark.parametrize('case', CASES)
def test_t13_rule_one_refuses(repo, case):
    r, pin = repo
    words = refusal(r, pin, case)
    if case == 'given_commit':
        record, problems = rule(r, 'f'*40)
        assert problems and any('only a check' in p for p in problems)
        return
    record, problems = rule(r)
    assert problems, case
    assert any(words in p for p in problems), (case, problems)


def test_t13_the_decision_log_line_in_another_commit_is_refused(tmp_path):
    r = tmp_path/'code'
    for rel in CODE+OTHER:
        write(r, rel, rel+'\n')
    git(r, 'init', '-q')
    commit(r, 'feasibility')
    pin = head(r)
    pin_record(r, pin, log='later')
    record, problems = rule(r)
    assert any('one commit' in p for p in problems)


def test_t13_the_holdout_profile_is_unchanged_on_an_lpba_record(repo):
    """A tree with only the LPBA40 record: the hold-out's rule one (profile None) does not see it as its P1."""
    r, _ = repo
    record, problems = ci.unit_identity_problems(repo=r)
    assert problems and 'runs/holdout_B_v1/pinned_code.json is not committed' in problems[0] and 'profile' not in record


def test_t13_p_l_against_the_feasibility_commit(repo):
    r, feas = repo
    record, problems = ci.pl_choice_problems(feas, feas, repo=r)
    assert problems == [] and record['passed']
    pl = head(r)                                                     # the record commit: runs/ and docs/ only
    assert ci.pl_choice_problems(feas, pl, repo=r)[1] == []
    write(r, 'docs/notes.md', 'n\n')
    commit(r, 'docs')
    assert ci.pl_choice_problems(feas, head(r), repo=r)[1] == []
    write(r, 'experiments/lpba/x.py', 'changed after the feasibility\n')
    commit(r, 'code')
    record, problems = ci.pl_choice_problems(feas, head(r), repo=r)
    assert problems and record['outside_docs_runs'] == ['experiments/lpba/x.py']
    record, problems = ci.pl_choice_problems(head(r), feas, repo=r)
    assert problems and record['feasibility_is_ancestor'] is False
    assert ci.pl_choice_problems('nope', feas, repo=r)[1]


def test_t13_lpba_common_rule_one_on_a_tree(repo, monkeypatch):
    r, pin = repo
    sys.path.insert(0, str(ROOT/'experiments'/'lpba'))
    import lpba_common as lc
    assert lc.rule_one_problem(repo=r) is None
    record, problems = lc.rule_one(repo=r)
    assert record['pinned'] == pin and problems == []
    write(r, 'jd_curl/b.py', 'changed\n')
    commit(r, 'code')
    problem = lc.rule_one_problem(repo=r)
    assert problem.startswith('rule one for LPBA40') and 'the code changed' in problem
    monkeypatch.setattr(lc, 'CODE_REPO', r)
    monkeypatch.setattr(lc.hp, 'check_code', lambda code, require_clean=True: dict(problems=[]))
    with pytest.raises(lc.hc.Refused, match='rule one for LPBA40'):
        lc.check_code(dict(commit='c'*40, clean=True), 'lpba40_test')
    git(r, 'reset', '-q', '--hard', 'HEAD~1')
    code = lc.check_code(dict(commit='c'*40, clean=True), 'lpba40_test')
    assert code['rule_one']['passed'] is True and code['rule_one']['record']['pinned'] == pin
    assert lc.formal_code_problem(code, 'lpba40_test') is None
    assert lc.formal_code_problem(dict(rule_one=dict(problem=None)), 'lpba40_test') is not None


def test_t13_the_evaluation_entrys_rule_one(repo):
    from bayesreg.eval import holdout as H
    from bayesreg.eval import lpba_eval as E
    r, pin = repo
    record = E.rule_one(r)
    assert record['pinned'] == pin
    write(r, 'scripts/d.py', 'changed\n')
    commit(r, 'code')
    with pytest.raises(H.HoldoutRefused, match='code_identity'):
        E.rule_one(r)


# ---- LP-B's hook (experiments/lpba/lpba_arms.pinned_code_problems; lpba_common.pinned_code_problems) --------------------------------------------------------------------
def test_lp_bs_hook_is_rule_one_with_the_lpba40_constants(repo):
    r, pin = repo
    sys.path.insert(0, str(ROOT/'experiments'/'lpba'))
    import lpba_arms
    import lpba_common as lc
    for hook in (lpba_arms.pinned_code_problems, lc.pinned_code_problems):
        record, problems = hook(repo=r)
        assert problems == [] and record['pinned'] == pin and record['profile'] == 'lpba40_test_v1'
        assert hook(pin, repo=r)[1] == []
        assert any('only a check' in p for p in hook('f'*40, repo=r)[1])
    write(r, 'runs/lpba40_test_v1/hb_input_lock.json', '{"lock": 1}\n')           # LP-B's step-7 input lock: added once after P_L
    commit(r, 'the input lock')
    assert lpba_arms.pinned_code_problems(repo=r)[1] == []
    write(r, 'runs/lpba40_test_v1/hb_input_lock.json', '{"lock": 2}\n')           # changed after: refused
    commit(r, 'the input lock changed')
    assert any('only added, once' in p for p in lpba_arms.pinned_code_problems(repo=r)[1])


def test_lp_bs_hook_refuses_a_tree_without_p_l(tmp_path):
    r = tmp_path/'code'
    for rel in CODE+OTHER:
        write(r, rel, rel+'\n')
    git(r, 'init', '-q')
    commit(r, 'no P_L')
    sys.path.insert(0, str(ROOT/'experiments'/'lpba'))
    import lpba_arms
    record, problems = lpba_arms.pinned_code_problems(repo=r)
    assert record['pinned'] is None and problems and 'runs/lpba40_test_v1/pinned_code.json is not committed' in problems[0]


def test_a_text_only_runner_gives_problems_not_an_exception(repo):
    """LP-B's queues hand their text-mode git stand-in to the rule-one hook in tests; rule one reads some files raw (git show): a runner answering in text is read as bytes and
    the result is the same as with git itself (C24.3: a git failure is a problem, never an exception)."""
    r, pin = repo

    def text_run(args, capture_output=True, text=False, timeout=None):
        done = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=GIT_ENV)
        return done
    record, problems = ci.unit_identity_problems(repo=r, run=text_run, profile=ci.LPBA40)
    assert problems == [] and record['pinned'] == pin
