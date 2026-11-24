"""bayesreg/io/code_identity.py, rule two (the gates; the lead's item 14 of HB's revision round; decision_log 2026-11-10 N3 and ~2:59 AM CST (2), (3); prereg revision 2 items 9 and 18): from the
gate's commit the code (the five directories, the root's importable files, new top-level directories) changes only by the four decision constants' values, judged on the AST, and by the allowlist's
allowance.  Rule one is tested through holdout_common in tests/test_bayesreg/holdout/test_pinned_code.py.  Temporary git repositories only; git's global and system configuration kept out."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from bayesreg.io import code_identity as ci

GIT_ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_AUTHOR_NAME='ci-test', GIT_AUTHOR_EMAIL='ci-test@example.invalid', GIT_COMMITTER_NAME='ci-test',
               GIT_COMMITTER_EMAIL='ci-test@example.invalid')
HOLDOUT_PY = '''"""The evaluation entry (a stand-in)."""
SEEDS = (1, 2)
TRUTH_READER_DECISION = None
OTHER_DECISION = None


def gate():
    return TRUTH_READER_DECISION
'''
PINS_PY = 'PQ_REPLAY_DECISION = None   # pinned after the replay\nNAME = "x"\n'
PREP_PY = 'import os\n\nPREP_DEVCHECK_DECISION = None\n'
ATROPOS_PY = 'import os\n\nATROPOS_DEVCHECK_DECISION = None\n'
FILES = {'bayesreg/__init__.py': '', 'bayesreg/eval/holdout.py': HOLDOUT_PY, 'experiments/pq/pqh_pins.py': PINS_PY, 'experiments/holdout/prep_stage.py': PREP_PY, 'experiments/holdout/atropos_holdout.py': ATROPOS_PY,
         'jd_curl/a.py': 'A = 1\n',
         'scripts/s.py': 'S = 1\n', ci.ALLOWLIST_REL: '{"included": []}\n', 'configs/other.json': '{}\n', 'docs/decision_log.md': 'log\n', 'splits/bayesreg_v1/holdout_B_pairs.tsv': 'x\n',
         'runs/holdout_B_v1/README.md': 'records\n', 'STATUS.md': 'status\n', 'README.md': 'readme\n'}
DECISION = 'dict(\n    passed=True,\n)'                          # not a literal: refused
LITERAL = "{\n    'passed': True,\n    'sha256': 'ab',\n    'pairs': [1, 2],\n}"           # a block of several lines: a literal


def git(r, *args):
    return subprocess.run(['git', '-C', str(r), *args], check=True, capture_output=True, text=True, env=GIT_ENV)


def write(r, rel, text):
    (r/rel).parent.mkdir(parents=True, exist_ok=True)
    (r/rel).write_text(text)


def commit(r, message='later'):
    git(r, 'add', '-A')
    git(r, 'commit', '-q', '-m', message)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path/'code'
    for rel, text in FILES.items():
        write(r, rel, text)
    git(r, 'init', '-q')
    commit(r, 'the gate')
    return r, git(r, 'rev-parse', 'HEAD').stdout.strip()


def gate(r, pin, run=None):
    record, problems = ci.gate_identity_problems(pin, repo=r, run=run)
    assert record['passed'] is (not problems) and record['problems'] == problems
    return record, problems


def include(r, *, off=False, pinned_record=True, log_line=True):
    """The allowlist's inclusion with the unseal record, runs/holdout_B_v1/pinned_code.json and the decision log's line in one commit (C23.3-6; contract 9b C24.3-5 exception B; C24.3-9)."""
    text = '{"included": ["OASIS_OAS1_0001_MR1"]}\n'
    write(r, ci.ALLOWLIST_REL, text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    if off:
        digest = digest[:-1]+('0' if digest[-1] != '0' else '1')
    write(r, ci.UNSEAL_RECORD_REL, json.dumps(dict(schema='bayesreg.unseal.v1', allowlist_sha256=digest)))
    line = '| 2026-11-10 | P1 = '+'0'*40+' (test) |'
    if pinned_record:
        write(r, ci.PINNED_RECORD_REL, json.dumps(dict(schema=ci.PINNED_RECORD_SCHEMA, commit='0'*40, decision_log_line=line)))
        if log_line:
            (r/'docs').mkdir(exist_ok=True)
            with open(r/ci.LATER_DECISION_LOG, 'a') as stream:
                stream.write(line+'\n')
    commit(r, 'inclusion')
    return line


def test_the_names_and_the_package():
    """The names are the unseal module's; the module imports nothing of experiments/ (bayesreg may not)."""
    from bayesreg.io import unseal
    assert ci.ALLOWLIST_REL == unseal.ALLOWLIST_REL and ci.UNSEAL_RECORD_REL == unseal.UNSEAL_REL and ci.REPO == Path(unseal.__file__).resolve().parents[2]
    text = Path(ci.__file__).read_text()
    assert 'experiments' not in [line.split()[1].split('.')[0] for line in text.splitlines() if line.startswith(('import ', 'from '))]
    assert ci.DECISION_CONSTANTS == ('TRUTH_READER_DECISION', 'PQ_REPLAY_DECISION', 'PREP_DEVCHECK_DECISION', 'ATROPOS_DEVCHECK_DECISION')


def test_the_gate_itself_and_the_constants_assignments_pass(repo):
    r, pin = repo
    record, problems = gate(r, pin)
    assert problems == [] and record['changes'] == [] and record['gate'] == record['head'] == pin
    write(r, 'bayesreg/eval/holdout.py', HOLDOUT_PY.replace('TRUTH_READER_DECISION = None', f'TRUTH_READER_DECISION = {LITERAL}'))
    write(r, 'experiments/pq/pqh_pins.py', PINS_PY.replace('PQ_REPLAY_DECISION = None', "PQ_REPLAY_DECISION = 'passed'"))
    write(r, 'experiments/holdout/prep_stage.py', PREP_PY.replace('PREP_DEVCHECK_DECISION = None', 'PREP_DEVCHECK_DECISION = {"rows": [1, 2.5, None, True]}'))
    write(r, 'experiments/holdout/atropos_holdout.py', ATROPOS_PY.replace('ATROPOS_DEVCHECK_DECISION = None', 'ATROPOS_DEVCHECK_DECISION = {  # a comment in the block\n    "ok": 1,\n}'))
    commit(r, 'P1: the four decisions')
    write(r, 'docs/decision_log.md', 'other\n')                         # outside the code: not checked (even an old line changed)
    write(r, 'runs/holdout_B_v1/x.md', 'anything\n')
    write(r, 'STATUS.md', 'status\n')
    commit(r, 'documents')
    include(r)                                                          # the allowlist and the unseal record in one commit
    record, problems = gate(r, pin)
    assert problems == [], problems
    kinds = sorted((c['kind'], c['path']) for c in record['changes'])
    assert kinds == [('allowlist', ci.ALLOWLIST_REL), ('decision_constants', 'bayesreg/eval/holdout.py'), ('decision_constants', 'experiments/holdout/atropos_holdout.py'),
                     ('decision_constants', 'experiments/holdout/prep_stage.py'), ('decision_constants', 'experiments/pq/pqh_pins.py')]
    assert record['allowlist_inclusion']['passed'] is True


BAD = {
    'statement_on_the_line': ('bayesreg/eval/holdout.py', ('TRUTH_READER_DECISION = None', "TRUTH_READER_DECISION = 'x'; import os")),
    'not_a_literal': ('bayesreg/eval/holdout.py', ('TRUTH_READER_DECISION = None', f'TRUTH_READER_DECISION = {DECISION}')),
    'call_in_a_block': ('bayesreg/eval/holdout.py', ('TRUTH_READER_DECISION = None', "TRUTH_READER_DECISION = {'a': __import__('os').getcwd()}")),
    'fifth_constant': ('bayesreg/eval/holdout.py', ('OTHER_DECISION = None', 'OTHER_DECISION = 1')),
    'other_statement': ('bayesreg/eval/holdout.py', ('SEEDS = (1, 2)', 'SEEDS = (1, 3)')),
    'statement_added': ('bayesreg/eval/holdout.py', ('def gate():', 'X = 1\n\n\ndef gate():')),
    'second_assignment': ('experiments/holdout/prep_stage.py', ('PREP_DEVCHECK_DECISION = None\n', 'PREP_DEVCHECK_DECISION = None\nPREP_DEVCHECK_DECISION = 1\n')),
    'augmented': ('experiments/holdout/prep_stage.py', ('PREP_DEVCHECK_DECISION = None\n', 'PREP_DEVCHECK_DECISION = None\nPREP_DEVCHECK_DECISION += 1\n')),
    'tuple_target': ('experiments/holdout/prep_stage.py', ('PREP_DEVCHECK_DECISION = None', 'PREP_DEVCHECK_DECISION, Y = None, 1')),
    'other_code_file': ('jd_curl/a.py', ('A = 1', 'A = 2')),
    'syntax_error': ('scripts/s.py', ('S = 1', 'S = (')),
    'multiline_last_line': ('bayesreg/eval/holdout.py', ('TRUTH_READER_DECISION = None', "TRUTH_READER_DECISION = {\n    'a': 1,\n}; import os")),
    'renamed_constant': ('bayesreg/eval/holdout.py', ('TRUTH_READER_DECISION = None\n', 'TRUTH_READER_DECISION_X = None\n')),
    'annotated': ('experiments/holdout/atropos_holdout.py', ('ATROPOS_DEVCHECK_DECISION = None', 'ATROPOS_DEVCHECK_DECISION: dict = {"ok": 1}')),
    'bound_by_import': ('experiments/holdout/prep_stage.py', ('import os\n', 'import os\nfrom os import sep as PREP_DEVCHECK_DECISION_X\nfrom os import sep as PQ_REPLAY_DECISION\n')),
    'constant_in_another_home': ('experiments/pq/pqh_pins.py', ('NAME = "x"', 'NAME = "x"\nTRUTH_READER_DECISION = 1')),
    'constant_in_a_code_file': ('jd_curl/a.py', ('A = 1', 'A = 1\nPQ_REPLAY_DECISION = 1')),
    'moved_to_another_home': ('experiments/holdout/prep_stage.py', ('PREP_DEVCHECK_DECISION = None', 'ATROPOS_DEVCHECK_DECISION = None')),
    'comment_then_statement': ('bayesreg/eval/holdout.py', ('TRUTH_READER_DECISION = None\n', 'TRUTH_READER_DECISION = 1  # x\nimport os\n')),
}


@pytest.mark.parametrize('case', sorted(BAD)+['added_file', 'deleted_file', 'allowlist_sha_off', 'allowlist_without_record', 'other_config', 'worktree_change', 'untracked_module',
                                            'not_ancestor', 'not_a_commit', 'root_sitecustomize', 'root_module', 'new_top_level_package', 'root_untracked_pth',
                                            'allowlist_without_pinned_record', 'allowlist_log_line_a_commit_later', 'root_link_committed', 'root_link_untracked',
                                            'root_link_untracked_dir', 'root_package_untracked_through_a_link'])
def test_anything_else_in_the_code_directories_is_refused(repo, case):
    r, pin = repo
    if case in BAD:
        rel, (old, new) = BAD[case]
        text = (r/rel).read_text()
        assert old in text
        write(r, rel, text.replace(old, new, 1))
        commit(r)
    elif case == 'added_file':
        write(r, 'experiments/new_tool.py', 'X = 1\n')
        commit(r)
    elif case == 'deleted_file':
        (r/'scripts/s.py').unlink()
        commit(r)
    elif case == 'allowlist_sha_off':
        include(r, off=True)
    elif case == 'allowlist_without_record':
        write(r, ci.ALLOWLIST_REL, '{"included": ["x"]}\n')
        commit(r)
    elif case == 'allowlist_without_pinned_record':                  # exception B: the pinned record is added in the same commit too
        include(r, pinned_record=False)
        write(r, ci.PINNED_RECORD_REL, '{}')
        commit(r, 'the pinned record a commit later')
    elif case == 'other_config':
        write(r, 'configs/other.json', '{"changed": 1}\n')
        commit(r)
    elif case == 'worktree_change':
        write(r, 'bayesreg/eval/holdout.py', HOLDOUT_PY.replace('TRUTH_READER_DECISION = None', "TRUTH_READER_DECISION = 'not committed'"))
    elif case == 'untracked_module':
        write(r, 'experiments/shadow.py', 'X = 1\n')
    elif case == 'not_ancestor':
        git(r, 'checkout', '-q', '-b', 'side')
        write(r, 'docs/decision_log.md', 'side\n')
        commit(r)
        pin = git(r, 'rev-parse', 'HEAD').stdout.strip()
        git(r, 'checkout', '-q', '-')
    elif case == 'not_a_commit':
        pin = 'f'*40
    elif case == 'root_sitecustomize':                  # the root is on every unit's PYTHONPATH: its importable files are code (decision_log ~2:59 AM CST (3))
        write(r, 'sitecustomize.py', 'import os\n')
        commit(r, 'a document commit that also adds sitecustomize.py')
    elif case == 'root_module':
        write(r, 'numpy.py', 'x = 1\n')
        commit(r)
    elif case == 'new_top_level_package':
        write(r, 'shadowpkg/__init__.py', 'x = 1\n')
        commit(r)
    elif case == 'root_untracked_pth':
        write(r, 'x.pth', 'import os\n')
    elif case == 'allowlist_log_line_a_commit_later':  # HB diff review n6 (C24.3-9): the decision log's line in the inclusion commit
        line = include(r, log_line=False)
        (r/'docs').mkdir(exist_ok=True)
        with open(r/ci.LATER_DECISION_LOG, 'a') as stream:
            stream.write(line+'\n')
        commit(r, 'the line a commit later')
    elif case == 'root_link_committed':                 # HB diff review n1: a root symbolic link is code (committed after the gate: mode 120000)
        (r/'numpy').symlink_to('experiments/holdout')
        commit(r, 'a link at the root')
    elif case == 'root_link_untracked':                 # HB diff review 2 r1: a link to a package directory (bayesreg/ holds __init__.py) is code
        (r/'json').symlink_to('bayesreg')
    elif case == 'root_link_untracked_dir':
        (r/'other_tree').mkdir()
        write(r, 'other_tree/pkg/__init__.py', 'x = 1\n')
        (r/'pkg').symlink_to('other_tree/pkg')        # untracked, git lists it without a trailing '/'
    elif case == 'root_package_untracked_through_a_link':
        (r/'pkg').symlink_to('bayesreg')
    record, problems = gate(r, pin)
    assert problems, record
    if case in ('root_link_untracked', 'root_link_untracked_dir', 'root_package_untracked_through_a_link'):
        assert record['importable_at_the_top'] and set(record['importable_at_the_top']) <= {'json', 'pkg', 'other_tree'}, record['importable_at_the_top']
    if case == 'root_link_committed':
        assert [(c['kind'], c['path']) for c in record['changes']] == [('root_link', 'numpy')], record['changes']
    if case == 'allowlist_log_line_a_commit_later':
        assert record['allowlist_inclusion']['decision_log_line_same_commit'] is False


@pytest.mark.parametrize('link', ['data_ignored', 'data_untracked', 'dangling', 'to_a_file', 'importable_name'])
def test_a_root_link_is_code_only_when_python_could_import_it(repo, tmp_path, link):
    """HB diff review 2 r1 (the lead's ruling (a), 2026-11-10 ~10:04 AM CST; C24.3-3 as written): an untracked or ignored root link is code only when it points at a directory holding __init__.py
    or its name is importable.  The main checkout's ignored data link (to a directory without __init__.py), a dangling link and a link to a file under a plain name pass rule two; shadow.py ->
    anything is refused."""
    r, pin = repo
    if link == 'data_ignored':
        write(r, '.gitignore', '/data\n')
        commit(r, 'outside the code: not checked')
    target = tmp_path/'d_root'
    (target/'three_d').mkdir(parents=True)
    name = 'shadow.py' if link == 'importable_name' else 'data'
    (r/name).symlink_to({'dangling': tmp_path/'nowhere', 'to_a_file': r/'README.md'}.get(link, target))
    record, problems = gate(r, pin)
    if link == 'importable_name':
        assert problems and record['importable_at_the_top'] == ['shadow.py'], record
    else:
        assert problems == [] and record['importable_at_the_top'] == [], (problems, record['importable_at_the_top'])


def test_a_compiled_file_beside_its_source_is_no_problem(repo):
    r, pin = repo
    write(r, 'bayesreg/eval/__pycache__/holdout.cpython-310.pyc', 'x')
    assert gate(r, pin)[1] == []
    write(r, 'bayesreg/eval/__pycache__/gone.cpython-310.pyc', 'x')
    assert gate(r, pin)[1]


STEPS = [('rev-parse', '--verify', '--quiet', 'HEAD^{commit}'), ('merge-base',), ('diff', '--name-status'), ('show',), ('status',), ('ls-tree', '-z', '--name-only'), ('ls-tree', '-r'),
         ('ls-files', '-z', '--others', '--', 'bayesreg'), ('ls-files', '-z', '--others', '--directory')]


@pytest.mark.parametrize('step', STEPS, ids=['-'.join(s[:2]) for s in STEPS])
def test_a_failing_git_call_is_a_problem_never_an_exception(repo, step):
    r, pin = repo
    write(r, 'bayesreg/eval/holdout.py', HOLDOUT_PY.replace('TRUTH_READER_DECISION = None', "TRUTH_READER_DECISION = 'x'"))
    commit(r)

    def run(args, **kwargs):
        if tuple(args[6:6+len(step)]) == step:
            return subprocess.CompletedProcess(args, 128, '' if kwargs.get('text') else b'', 'test: git failed' if kwargs.get('text') else b'test: git failed')
        return subprocess.run(args, **kwargs)
    record, problems = gate(r, pin, run=run)
    assert problems and any('test: git failed' in p for p in problems), problems


def test_an_oserror_of_git_is_a_problem(repo):
    r, pin = repo

    def run(args, **kwargs):
        raise FileNotFoundError('git')
    record, problems = gate(r, pin, run=run)
    assert problems and record['gate'] is None
    record, problems = ci.allowlist_inclusion_problems(pin, repo=r, run=run)
    assert problems and record['passed'] is False


def test_constants_only_problem_on_texts():
    assert ci.constants_only_problem(HOLDOUT_PY, HOLDOUT_PY) is None
    assert ci.constants_only_problem(HOLDOUT_PY, HOLDOUT_PY.replace('TRUTH_READER_DECISION = None', "TRUTH_READER_DECISION = ('a', 1.0, -2, b'x', None)")) is None
    assert ci.constants_only_problem(HOLDOUT_PY, HOLDOUT_PY.replace('TRUTH_READER_DECISION = None', 'TRUTH_READER_DECISION = None  # x')) is None     # a comment on its own lines: no statement
    assert ci.constants_only_problem(HOLDOUT_PY, HOLDOUT_PY.replace('TRUTH_READER_DECISION = None', 'TRUTH_READER_DECISION = SEEDS'))
    assert ci.constants_only_problem(HOLDOUT_PY, HOLDOUT_PY.replace('\n\n\ndef gate', '\nTRUTH_READER_DECISION = 2\n\n\ndef gate'))
    assert ci.decision_assignment_problem(HOLDOUT_PY, HOLDOUT_PY.replace('TRUTH_READER_DECISION = None', 'TRUTH_READER_DECISION = (\n    1,\n    2,\n)')) is None
    assert ci.decision_assignment_problem(HOLDOUT_PY, HOLDOUT_PY.replace('TRUTH_READER_DECISION = None', 'TRUTH_READER_DECISION = (\n    1,\n); X = 3'))
    assert ci.decision_assignment_problem(HOLDOUT_PY, HOLDOUT_PY.replace('SEEDS = (1, 2)\n', 'SEEDS = (1, 2)  # changed comment outside\n'))


def test_a_comment_on_the_constants_line_may_change_but_not_one_elsewhere(repo):
    """The AST rule (decision_log ~2:59 AM CST (2)): the bytes outside the assignment's lines are the same; inside them only the statement (whose value may change) and a comment."""
    r, pin = repo
    write(r, 'experiments/pq/pqh_pins.py', PINS_PY.replace('PQ_REPLAY_DECISION = None   # pinned after the replay', "PQ_REPLAY_DECISION = 'passed'   # the replay passed"))
    commit(r)
    assert gate(r, pin)[1] == []
    write(r, 'experiments/pq/pqh_pins.py', (r/'experiments/pq/pqh_pins.py').read_text().replace('NAME = "x"', 'NAME = "x"  # a new comment elsewhere'))
    commit(r)
    assert gate(r, pin)[1]


def test_a_gates_range_with_a_merge_counts_the_allowlists_commit_once(repo):
    """Contract 9b C24.3-5 exception B: a gate's range on main may hold merges; the allowlist's commit is counted with git log --full-history --no-merges, once."""
    r, pin = repo
    git(r, 'checkout', '-q', '-b', 'side')
    include(r)
    git(r, 'checkout', '-q', '-')
    write(r, 'docs/other.md', 'main moved on\n')
    commit(r)
    git(r, 'merge', '-q', '--no-edit', 'side')
    record, problems = gate(r, pin)
    assert problems == [], problems
    assert record['allowlist_inclusion']['allowlist_commits'] and len(record['allowlist_inclusion']['allowlist_commits']) == 1


def test_the_homes_are_the_codes():
    """DECISION_HOMES names the files the constants are assigned in now (PQ_REPLAY_DECISION's pqh_pins.py comes with PQH)."""
    for name, rel in ci.DECISION_HOMES.items():
        path = ci.REPO/rel
        if path.is_file():
            assert f'\n{name} = ' in '\n'+path.read_text(), (name, rel)
    assert sorted(ci.DECISION_HOMES) == sorted(ci.DECISION_CONSTANTS)


def test_status_z_is_read_with_a_renames_old_path():
    """HB diff review n7: git status --porcelain=v1 -z writes a rename or a copy as 'XY NEW\\0OLD\\0' (the old path without 'XY ')."""
    assert ci._status_paths('R  docs/y.md\0docs/x.md\0 M a.py\0C  b.py\0c.py\0RM d/e.py\0f.py\0') == ['docs/y.md', 'docs/x.md', 'a.py', 'b.py', 'c.py', 'd/e.py', 'f.py']
    assert ci._status_paths('') == [] and ci._status_paths('?? x\0') == ['x']


def test_a_staged_rename_is_two_paths_in_the_working_tree_check(tmp_path):
    """n7 on a repository: a staged rename of a document is no change of the code (its old path was read as a new top-level directory 's/x.md'); a staged rename of a root module away is (its old
    path, a root importable file, was read as 'y')."""
    r = tmp_path/'code'
    for rel, text in {**FILES, 'a.py': 'x = 1\n', 'docs/x.md': 'x\n'}.items():
        write(r, rel, text)
    git(r, 'init', '-q')
    commit(r, 'the gate')
    pin = git(r, 'rev-parse', 'HEAD').stdout.strip()
    git(r, 'mv', 'docs/x.md', 'docs/y.md')
    record, problems = gate(r, pin)
    assert problems == [] and record['worktree_changes'] == [], (problems, record['worktree_changes'])
    git(r, 'mv', 'a.py', 'docs/a.py')
    record, problems = gate(r, pin)
    assert problems and record['worktree_changes'] == ['a.py'], (problems, record['worktree_changes'])


def test_files_outside_the_code_are_not_checked(repo):
    r, pin = repo
    for rel in ('docs/decision_log.md', 'STATUS.md', 'runs/holdout_B_v1/x.md', 'splits/bayesreg_v1/holdout_B_pairs.tsv', 'README.md'):
        write(r, rel, 'changed\n')
    commit(r, 'documents and records')
    record, problems = gate(r, pin)
    assert problems == [] and record['changes'] == []


def test_p1_the_range_from_the_gates_commit_to_p1_is_rule_two():
    """P1 (runs/20261115-p1/实现说明_P1.md): the gates ran on P0'' (318e72ad; every record and lock names it, and the four constants' code_commit with it: tests/test_bayesreg/_p1_decisions.py),
    an ancestor of P1 (3005a57e, the commit the hold-out ran on); from P0'' to P1 the committed code changes pass rule two (gate_identity_problems on this repository): exactly the four
    constants' files, each modified only by its constant -- no allowlist row (the allowlist changed only in the unseal commit, on the unseal branch, under rule one's exception B) and nothing
    else.  The working tree is the gates' own check at run time, not this test's (a local edit does not fail it).  Reads git only.

    Re-pinned 2026-11-20 (the lead's brief; tests only, no code changed).  Until then this was test_p1_the_range_from_the_gates_commit_to_head_is_rule_two and checked P0''..HEAD: that was the
    gates' range while main stood at P1's code, but after the hold-out run main added code that rule two refuses by design (T24's bayesreg/eval/holdout_tet24.py, C24.12; the LPBA40 tools,
    C26), and the test failed on that, not on the gates (rule_two_record says more).  The range is now the fixed one the lead recorded as passed (decision_log 2026-11-15 ~10:05 PM CST:
    gate_identity_problems('318e72ad', head='3005a57e') passed).  Every old assertion is kept with `head` P1 in place of HEAD; the old bounds on the changed paths (the four files <= the
    paths <= the four files and the allowlist) are now an equality, and each row's status, kind and problem are asserted: stronger, not weaker -- on this fixed history an allowlist row, an
    added file or a row with a problem would fail.  That main keeps the hold-out's frozen files is the next test's."""
    from .. import _p1_decisions as P1
    record, problems = P1.rule_two_record(P1.P0PP, head=P1.P1_COMMIT)
    assert problems == [], problems
    assert record['gate'] == P1.P0PP and record['gate_is_ancestor'] is True and record['head'] == P1.P1_COMMIT and record['head_given'] == P1.P1_COMMIT
    assert sorted(P1.HOMES) == sorted(ci.DECISION_HOMES.values())
    assert sorted(r['path'] for r in record['changes']) == sorted(P1.HOMES)       # P1 review minor 1: P1 changed all four files (an empty range does not pass); no allowlist row, nothing else
    assert all(r['status'] == 'M' and r['kind'] == 'decision_constants' and r['problem'] is None for r in record['changes'])
    assert record['allowlist_inclusion'] is None and record['passed'] is True
    assert record['worktree_changes'] == [] and record['files_in_code_dirs'] == [] and record['importable_at_the_top'] == []      # step 3 answered as a clean tree (rule_two_record)


FROZEN_SINCE_P1 = ('jd_curl', 'bayesreg/eval/holdout.py', 'bayesreg/eval/holdout_decide.py', 'tests/test_bayesreg/eval/test_holdout_decide.py',
                   'configs/bayesreg/dev_subjects.json', 'configs/frozen/holdout_B_v1_arm_table.json', 'scripts/folding_metrics.py', 'bayesreg/eval/metrics.py',
                   'bayesreg/eval/fields.py')
T24_MERGE = '52de5375d02f56dd7e7c449ac66bb1225621b3f7'      # the merge that brought T24 into main (2026-11-19 ~12:37 PM CST; contract C24.12; decision_log line 527)
FROZEN_SINCE_T24 = ('bayesreg/eval/holdout_tet24.py', 'tests/test_bayesreg/eval/test_holdout_tet24.py')


def test_after_p1_main_keeps_the_hold_outs_frozen_files():
    """Added 2026-11-20 with the re-pin above (the lead's brief: guard that main does not change the hold-out's frozen code, now that the range of the test above no longer runs to HEAD);
    widened the same day (the p1-repin review's m1, m2).  For each base -- P1 (3005a57e) for FROZEN_SINCE_P1, T24's merge into main (52de5375) for FROZEN_SINCE_T24, whose two files
    P1 does not have -- the base is an ancestor of HEAD, and each of its paths is at HEAD exactly as at the base: `git diff --name-status --no-renames <base> HEAD -- <paths>` is empty (no
    file modified, added, deleted or renamed), and each path is in the base's tree (so an empty diff is not a typo's).  Committed history only, like the test above: a local edit does not
    fail it.  Reads git only.

    Why these and not all of P1's code: after the hold-out run main changed P1's code on purpose under contract C26 (LP-A, LP-B: bayesreg/io/guard.py, pair.py, eval/truth.py,
    experiments/hb, he, xb -- the B line's behaviour kept, the 34 LPBA40 test subjects refused more; LP-C2: bayesreg/io/code_identity.py's profiles), so P1's code as a whole is not
    frozen on main.  Among what the contract freezes on main, these:
      - jd_curl/: hard constraint 2 (the 37 modules byte for byte as at eb808be; P1's jd_curl/ is that, `git diff eb808be 3005a57e -- jd_curl` only deletions into archive/) and C26.1-5;
      - bayesreg/eval/holdout.py: the B line's evaluation entry, the only reader of the hold-out truth, and the home of TRUTH_READER_DECISION -- C26.7-1 "B 线留出集的评测入口（C23.12）
        不改", C26.1-5;
      - bayesreg/eval/holdout_decide.py and its tests (tests/test_bayesreg/eval/test_holdout_decide.py): D146 (C26.5-8, C26.12) "holdout_decide.py 与它的常量、测试都不改，B 线判定代码在
        main 上保持冻结";
      - configs/bayesreg/dev_subjects.json: C26.1-5 (its sha256 is cited by the segmenter registries and many records);
      - configs/frozen/holdout_B_v1_arm_table.json: the hold-out's arm table (sha256 69cb1ddf..., prereg revision 2); configs/frozen/ is add-only (project rules, git rules);
      - scripts/folding_metrics.py: C24.12-1 ("jd_curl/ 与 scripts/folding_metrics.py 一行不改") and C26.7-5 (sha256 859fe08f...);
      - bayesreg/eval/metrics.py and bayesreg/eval/fields.py: the metric and field code the hold-out evaluation imported at P1 (holdout.py's `from . import cert_eval, metrics` and
        `from .fields import ...`), shared with the LPBA40 entry by C26.7-4 (metrics sha256 36828cfb...), and fields' functions are the ones T24 calls as P1's (C24.12-1, -2);
      - bayesreg/eval/holdout_tet24.py and tests/test_bayesreg/eval/test_holdout_tet24.py, against 52de5375: C26.7-5 "不改 bayesreg/eval/holdout_tet24.py" (C24.12-2's
        --reviewed-commit wants main's blob of the module to be the reviewed one), and the tests that went into main with it.
    The hold-out numbers (seg35, seg4, the F1/F2/F4 verdicts, the 24-tet proxy) come from these files, and the manuscript cites them by SHA.  A deliberate change of one of them on main
    needs the lead's (or BZ's) decision and a contract revision first; then this test is changed in the same reviewed package, naming the file and the decision_log line -- it is not
    skipped or deleted."""
    from .. import _p1_decisions as P1
    git, _ = ci.git_runner(P1.REPO)
    for base, paths in ((P1.P1_COMMIT, FROZEN_SINCE_P1), (T24_MERGE, FROZEN_SINCE_T24)):
        rc_a, _, err_a = git('merge-base', '--is-ancestor', base, 'HEAD')
        assert rc_a == 0, f'{base} is not an ancestor of HEAD (exit {rc_a} {err_a})'
        for path in paths:
            rc_t, listed, err_t = git('ls-tree', '-r', '--name-only', '-z', base, '--', path)
            assert rc_t == 0 and [p for p in listed.split('\0') if p], f'{path} is not in the tree of {base} (exit {rc_t} {err_t})'
        rc_d, out, err_d = git('diff', '--name-status', '--no-renames', '-z', base, 'HEAD', '--', *paths)
        fields = [f for f in out.split('\0') if f]
        assert rc_d == 0, f'git diff --name-status {base} HEAD: exit {rc_d} {err_d}'
        assert fields == [], f'the hold-out\'s frozen files changed on main since {base} (status, path): {list(zip(fields[0::2], fields[1::2]))[:20]}'
