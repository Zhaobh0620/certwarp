"""Rule one of the code identity (bayesreg/io/code_identity.unit_identity_problems through holdout_common.pinned_code_problems; decision_log 2026-11-09 df8541d, 2026-11-10 I1, N3 and ~2:59 AM
CST (1)-(4); the lead's items 7, 12, 14 of HB's revision rounds; PQH review I6): P1 from the committed runs/holdout_B_v1/pinned_code.json (never HEAD; a given commit only compared); every commit of
P1..HEAD on its own: authorizations and records only added once, the decision log only appended to, the allowlist modified once with the unseal record; the code (the five directories, the root's
importable files, new top-level directories) unchanged; a clean working tree.  Every branch on a temporary git repository."""
import ast
import hashlib
import json
import re
import subprocess

import pytest

from ..xb import _world as W

hc = W.hc
CODE = ('bayesreg/a.py', 'experiments/he/run_hx.py', 'jd_curl/b.py', 'configs/c.json', 'configs/bayesreg/dev_subjects.json', 'scripts/d.py')
OTHER = ('splits/bayesreg_v1/holdout_B_pairs.tsv', 'splits/bayesreg_v1/holdout_B_v1_hb_resume_1.json', 'runs/holdout_B_v1/README.md', 'docs/decision_log.md', 'docs/other.md', 'README.md')
ALLOW = 'configs/bayesreg/dev_subjects.json'
UNSEAL = 'splits/bayesreg_v1/unseal_holdout_B_v1.json'
RECORD = 'runs/holdout_B_v1/pinned_code.json'


def write(r, rel, text='x\n'):
    (r/rel).parent.mkdir(parents=True, exist_ok=True)
    (r/rel).write_text(text)


def commit(r, message='later'):
    W.git(r, 'add', '-A')
    W.git(r, 'commit', '-q', '-m', message)


def pin_record(r, pin, log='same', **override):
    """The inclusion commit (C23.3-6): the allowlist modified, the unseal record and runs/holdout_B_v1/pinned_code.json added, the decision log's line appended -- in the same commit (C24.3-9;
    log 'earlier' or 'later': in a commit of its own before or after, HB diff review n6)."""
    text = '{"included": ["OASIS_OAS1_0001_MR1"]}\n'
    line = override.pop('line', f'| 2026-11-10 | P1 = {pin} (test) |')
    if log == 'earlier':
        with open(r/'docs/decision_log.md', 'a') as stream:
            stream.write(line+'\n')
        commit(r, 'the decision log line first')
    write(r, ALLOW, text)
    write(r, UNSEAL, json.dumps(dict(schema='bayesreg.unseal.v1', allowlist_sha256=override.pop('allowlist_sha256', hashlib.sha256(text.encode()).hexdigest()))))
    doc = dict(schema=hc.PINNED_RECORD_SCHEMA, commit=pin, decision_log_line=line, written_by='lead', time_local='2026-11-10 3:10 AM CST', utc='2026-11-10T09:10:00Z', reason='test')
    doc.update(override)
    write(r, RECORD, json.dumps(doc))
    if log == 'same':
        with open(r/'docs/decision_log.md', 'a') as stream:
            stream.write(line+'\n')
    commit(r, 'inclusion and the pinned record')
    if log == 'later':
        with open(r/'docs/decision_log.md', 'a') as stream:
            stream.write(line+'\n')
        commit(r, 'the decision log line after')


def repo(tmp_path, *, files=CODE+OTHER, ignore=None, record=True):
    r = tmp_path/'code'
    for rel in files:
        write(r, rel, rel+'\n')
    if ignore:
        write(r, '.gitignore', ignore)
    W.git(r, 'init', '-q')
    commit(r, 'P1')
    pin = W.git(r, 'rev-parse', 'HEAD').stdout.strip()
    if record:
        pin_record(r, pin)
    return r, pin


def problems(r, pinned=None, run=None):
    record, out = hc.pinned_code_problems(pinned, repo=r, run=run)
    assert record['passed'] is (not out) and record['problems'] == out
    return record, out


LATER_NAMES = ('splits/bayesreg_v1/holdout_B_v1_hb_resume_2.json', 'splits/bayesreg_v1/holdout_B_v1_xb_no_resume_12.json', 'splits/bayesreg_v1/holdout_B_v1_pqh_rerun_1.json',
               'splits/bayesreg_v1/unseal_holdout_B_v1_resume_1.json', 'splits/bayesreg_v1/holdout_B_v1_reeval_3.json', 'runs/holdout_B_v1/pq_release.md',
               'runs/holdout_B_v1/记录_非ASCII 名.json', 'runs/holdout_B_v1/rejudgments/configc/p_1.json')


def test_p1_comes_from_the_record_and_the_allowed_later_commits_pass(tmp_path):
    r, pin = repo(tmp_path, ignore='__pycache__/\n*.pyc\n')
    record, out = problems(r)
    assert out == [] and record['pinned'] == pin and record['head'] != pin and record['pinned_record']['document']['commit'] == pin
    assert record['code_dirs'] == ['bayesreg', 'experiments', 'jd_curl', 'configs', 'scripts'] and record['code_dirs_differ'] == [ALLOW]
    for rel in LATER_NAMES:
        write(r, rel, '{}\n')
        commit(r, f'add {rel}')                        # one commit each: every commit is checked on its own
    with open(r/'docs/decision_log.md', 'a') as stream:
        stream.write('| 2026-11-10 | appended |\n')
    commit(r)
    for rel in ('experiments/he/__pycache__/run_hx.cpython-310.pyc', 'scripts/__pycache__/d.cpython-310.opt-1.pyc'):
        write(r, rel)
    record, out = problems(r, pin[:10])                 # a given commit (short) is only compared with the record's
    assert out == [] and record['pinned'] == pin and len(record['commits']) == len(LATER_NAMES)+2
    assert all(c['problem'] is None for row in record['commits'] for c in row['changes'])
    assert sum(f['problem'] is None for f in record['files_in_code_dirs']) == 2


@pytest.mark.parametrize('case', ['no_record', 'commit_mismatch', 'record_keys', 'record_line_missing', 'record_line_without_commit', 'record_writer', 'not_ancestor'])
def test_the_pinned_record_is_the_one_source_of_p1(tmp_path, case):
    """decision_log 2026-11-10 ~2:59 AM CST (1): no record -> refused (never HEAD by default); --commit only a check; the record's keys, its decision log line (holding the commit, a line of the
    log), its writer; P1 an ancestor of HEAD."""
    if case == 'no_record':
        r, pin = repo(tmp_path, record=False)
        record, out = problems(r)
        assert out and record['pinned'] is None and any('never HEAD' in p for p in out)
        return
    r, pin = repo(tmp_path, record=case in ('commit_mismatch', 'not_ancestor'))
    if case == 'commit_mismatch':
        record, out = problems(r, 'f'*40)
        assert out and any('only a check' in p for p in out)
        return
    if case == 'not_ancestor':
        W.git(r, 'checkout', '-q', '-b', 'side', pin)
        write(r, 'docs/other.md', 'side\n')
        commit(r)
        side = W.git(r, 'rev-parse', 'HEAD').stdout.strip()
        W.git(r, 'checkout', '-q', '-')
        doc = json.loads((r/RECORD).read_text())
        W.git(r, 'rm', '-q', RECORD)                    # a record pointing at a commit off HEAD's line (the deletion itself is refused as well)
        commit(r)
        write(r, RECORD, json.dumps(dict(doc, commit=side, decision_log_line=f'| P1 = {side} |')))
        with open(r/'docs/decision_log.md', 'a') as stream:
            stream.write(f'| P1 = {side} |\n')
        commit(r)
        record, out = problems(r)
        assert out
        return
    override = dict(record_keys=dict(extra=1), record_line_missing=dict(line='| not in the log |'), record_line_without_commit=dict(line='| P1 = (no sha) |'),
                    record_writer=dict(written_by='someone'))[case]
    if case == 'record_line_missing':
        text = '{"included": ["OASIS_OAS1_0001_MR1"]}\n'
        write(r, ALLOW, text)
        write(r, UNSEAL, json.dumps(dict(allowlist_sha256=hashlib.sha256(text.encode()).hexdigest())))
        write(r, RECORD, json.dumps(dict(schema=hc.PINNED_RECORD_SCHEMA, commit=pin, decision_log_line=f'| P1 = {pin} (not written to the log) |', written_by='lead', time_local='x', utc='x',
                                         reason='x')))
        commit(r)
    else:
        pin_record(r, pin, **override)
    record, out = problems(r)
    assert out and record['pinned'] is None


def later(r, case, pin):
    """One case of a commit (or a working tree change) after the inclusion commit."""
    if case == 'code_committed':
        write(r, 'experiments/he/run_hx.py', 'changed\n')
        commit(r)
    elif case == 'code_deleted':
        (r/'jd_curl/b.py').unlink()
        commit(r)
    elif case == 'code_uncommitted':
        write(r, 'configs/c.json', 'changed\n')
    elif case == 'untracked_module':
        write(r, 'experiments/he/shadow.py', 'x = 1\n')
    elif case == 'untracked_in_scripts':
        write(r, 'scripts/new_tool.py', 'x = 1\n')
    elif case == 'ignored_module':
        write(r, '.gitignore', 'local_*.py\n')
        write(r, 'runs/holdout_B_v1/ignore_note.md', '.gitignore is not a record: refused too\n')
        write(r, 'bayesreg/local_patch.py', 'x = 1\n')
    elif case == 'sourceless_pyc':
        write(r, 'experiments/he/he_core.pyc', 'x')
    elif case == 'cache_pyc_without_source':
        write(r, 'experiments/he/__pycache__/gone.cpython-310.pyc', 'x')
    elif case == 'cache_other_file':
        write(r, 'jd_curl/__pycache__/b.py', 'x = 1\n')
    elif case == 'other_split_file':
        write(r, 'splits/bayesreg_v1/unseal_other.json', '{}')
        commit(r)
    elif case == 'reeval_elsewhere':
        write(r, 'splits/holdout_B_v1_reeval_1.json', '{}')
        commit(r)
    elif case == 'authorization_modified':
        write(r, 'splits/bayesreg_v1/holdout_B_v1_hb_resume_1.json', 'changed\n')
        commit(r)
    elif case == 'authorization_added_then_modified':
        write(r, 'splits/bayesreg_v1/holdout_B_v1_hb_resume_2.json', '{}')
        commit(r)
        write(r, 'splits/bayesreg_v1/holdout_B_v1_hb_resume_2.json', '{"changed": 1}')
        commit(r)
    elif case == 'record_added_then_modified':
        write(r, 'runs/holdout_B_v1/x.json', '{}')
        commit(r)
        write(r, 'runs/holdout_B_v1/x.json', '{"changed": 1}')
        commit(r)
    elif case == 'record_added_then_deleted':
        write(r, 'runs/holdout_B_v1/x.json', '{}')
        commit(r)
        (r/'runs/holdout_B_v1/x.json').unlink()
        commit(r)
    elif case == 'record_modified':
        write(r, 'runs/holdout_B_v1/README.md', 'changed\n')
        commit(r)
    elif case == 'record_renamed':
        W.git(r, 'mv', 'runs/holdout_B_v1/README.md', 'runs/holdout_B_v1/README2.md')
        commit(r)
    elif case == 'unseal_record_modified':
        write(r, UNSEAL, '{"changed": 1}')
        commit(r)
    elif case == 'unseal_record_deleted':              # HB diff review n8: an authorization deleted after it was added
        (r/UNSEAL).unlink()
        commit(r)
    elif case == 'pinned_record_deleted':
        (r/RECORD).unlink()
        commit(r)
    elif case == 'pinned_record_modified':
        doc = json.loads((r/RECORD).read_text())
        write(r, RECORD, json.dumps(dict(doc, reason='changed')))
        commit(r)
    elif case == 'decision_log_old_line_changed':
        text = (r/'docs/decision_log.md').read_text()
        write(r, 'docs/decision_log.md', text.replace('docs/decision_log.md', 'docs/decision_log.md (edited)', 1)+'| appended too |\n')
        commit(r)
    elif case == 'decision_log_deleted':
        (r/'docs/decision_log.md').unlink()
        commit(r)
    elif case == 'allowlist_again':
        write(r, ALLOW, '{"included": ["OASIS_OAS1_0002_MR1"]}\n')
        commit(r)
    elif case == 'other_config':
        write(r, 'configs/c.json', '{"changed": 1}\n')
        commit(r)
    elif case == 'other_path':
        write(r, 'docs/other.md', 'changed\n')
        commit(r)
    elif case == 'root_sitecustomize':
        write(r, 'sitecustomize.py', 'import os\n')
        commit(r)
    elif case == 'root_pth':
        write(r, 'x.pth', 'import os\n')
        commit(r)
    elif case == 'new_top_level_package':
        write(r, 'shadowpkg/__init__.py', 'x = 1\n')
        commit(r)
    elif case == 'new_top_level_dir':
        write(r, 'tools/notes.md', 'x\n')
        commit(r)
    elif case in ('top_level_module', 'top_level_pth', 'sitecustomize_untracked', 'top_level_package'):
        name = dict(top_level_module='numpy.py', top_level_pth='y.pth', sitecustomize_untracked='sitecustomize.py', top_level_package='pkg2/__init__.py')[case]
        write(r, name, 'x = 1\n')
    elif case == 'root_link_committed':                 # HB diff review n1: a symbolic link at the root (to a package) is code
        write(r, 'experiments/pkg/__init__.py', 'x = 1\n')
        (r/'numpy').symlink_to('experiments/pkg')
        commit(r)
    elif case == 'root_link_ignored':
        write(r, '.gitignore', 'json\nother_tree/\n')    # an ignored link too (git ls-files --others lists both)
        commit(r)
        write(r, 'other_tree/pkg/__init__.py', 'x = 1\n')   # HB diff review 2 r1: a link is code when it points at a package directory
        (r/'json').symlink_to('other_tree/pkg')
    elif case == 'root_link_untracked':
        write(r, 'other_tree/pkg/__init__.py', 'x = 1\n')   # other_tree/ itself holds no __init__.py: not code; the link to its package is
        (r/'json').symlink_to('other_tree/pkg')
    elif case == 'root_link_importable_name':          # r1: a link under an importable name is code whatever it points at
        (r/'shadow.py').symlink_to('docs/other.md')
    elif case == 'staged_rename':                       # HB diff review n7: a rename's two paths
        W.git(r, 'mv', 'docs/other.md', 'docs/other2.md')
    elif case == 'tracked_dirty':
        write(r, 'docs/other.md', 'changed, not committed\n')
    elif case == 'staged_not_committed':
        write(r, 'runs/holdout_B_v1/new.json', '{}')
        W.git(r, 'add', 'runs/holdout_B_v1/new.json')
    elif case == 'merge_commit':
        W.git(r, 'checkout', '-q', '-b', 'side')
        write(r, 'runs/holdout_B_v1/side.json', '{}')
        commit(r)
        W.git(r, 'checkout', '-q', '-')
        write(r, 'runs/holdout_B_v1/main.json', '{}')
        commit(r)
        W.git(r, 'merge', '-q', '--no-edit', 'side')
    else:
        raise AssertionError(case)


CASES = ['code_committed', 'code_deleted', 'code_uncommitted', 'untracked_module', 'untracked_in_scripts', 'ignored_module', 'sourceless_pyc', 'cache_pyc_without_source', 'cache_other_file',
         'other_split_file', 'reeval_elsewhere', 'authorization_modified', 'authorization_added_then_modified', 'record_added_then_modified', 'record_added_then_deleted', 'record_modified',
         'record_renamed', 'unseal_record_modified', 'unseal_record_deleted', 'pinned_record_modified', 'pinned_record_deleted', 'decision_log_old_line_changed', 'decision_log_deleted',
         'allowlist_again', 'other_config', 'other_path', 'root_sitecustomize', 'root_pth', 'new_top_level_package', 'new_top_level_dir', 'top_level_module', 'top_level_pth',
         'sitecustomize_untracked', 'top_level_package', 'root_link_committed', 'root_link_ignored', 'root_link_untracked', 'root_link_importable_name', 'staged_rename', 'tracked_dirty',
         'staged_not_committed', 'merge_commit']


@pytest.mark.parametrize('case', CASES)
def test_anything_else_after_p1_is_a_problem(tmp_path, case):
    r, pin = repo(tmp_path)
    later(r, case, pin)
    record, out = problems(r)
    assert out, record
    if case in ('authorization_added_then_modified', 'record_added_then_modified', 'record_added_then_deleted'):
        net = W.git(r, 'diff', '--name-status', pin, 'HEAD').stdout
        assert 'x.json' not in net or '\tA' in net or 'A\t' in net          # the net difference alone would not show it: the commits are checked one by one
    if case in ('top_level_module', 'top_level_pth', 'sitecustomize_untracked', 'top_level_package'):
        assert record['importable_at_the_top']
    if case in ('root_link_ignored', 'root_link_untracked'):
        assert record['importable_at_the_top'] == ['json'], record['importable_at_the_top']
    if case == 'root_link_importable_name':
        assert record['importable_at_the_top'] == ['shadow.py'], record['importable_at_the_top']
    if case == 'root_link_committed':
        assert any('numpy' in p and 'root_link' in p for p in out), out
    if case == 'unseal_record_deleted':
        assert any(f'D {UNSEAL}' in p for p in out), out
    if case == 'staged_rename':
        assert sorted(record['tracked_changes']) == ['docs/other.md', 'docs/other2.md'], record['tracked_changes']


@pytest.mark.parametrize('link', ['data_ignored', 'data_untracked', 'dangling', 'to_a_file'])
def test_a_root_link_that_imports_nothing_is_not_code(tmp_path, link):
    """HB diff review 2 r1 (the lead's ruling (a), 2026-11-10 ~10:04 AM CST; contract C24.3-3 as written): the main checkout's ignored data -> <a directory without __init__.py> link, an untracked one,
    a dangling link and a link to a file under a name Python cannot import are not code: rule one passes with them there (the expectation written down, as the review asked)."""
    r, pin = repo(tmp_path, ignore='/data\n' if link == 'data_ignored' else None)
    (tmp_path/'d_root'/'three_d').mkdir(parents=True)
    if link in ('data_ignored', 'data_untracked'):
        (r/'data').symlink_to(tmp_path/'d_root')
    elif link == 'dangling':
        (r/'data').symlink_to(tmp_path/'nowhere')
    else:
        (r/'data').symlink_to(r/'docs'/'other.md')
    record, out = problems(r)
    assert out == [] and record['importable_at_the_top'] == [], (out, record['importable_at_the_top'])


@pytest.mark.parametrize('log', ['earlier', 'later'])
def test_the_decision_log_line_is_added_in_the_inclusion_commit(tmp_path, log):
    """HB diff review n6 (C24.3-9): the allowlist's inclusion, the unseal record, pinned_code.json and the decision log's line in one commit; the line a commit earlier or later is refused (rule one,
    and the gates' allowlist_inclusion_problems)."""
    r, pin = repo(tmp_path, record=False)
    pin_record(r, pin, log=log)
    record, out = problems(r)
    assert out and any('C24.3-9' in p for p in out), out
    found, inclusion = hc.allowlist_inclusion_problems(pin, repo=r)
    assert found['same_commit'] is True and found['decision_log_line_same_commit'] is False and any('C24.3-9' in p for p in inclusion), inclusion


@pytest.mark.parametrize('case', ['sha_one_digit', 'no_record', 'separate_commits', 'pinned_record_later'])
def test_the_allowlists_change_is_held_to_the_inclusion_commit(tmp_path, case):
    r = tmp_path/'code'
    for rel in CODE+OTHER:
        write(r, rel, rel+'\n')
    W.git(r, 'init', '-q')
    commit(r, 'P1')
    pin = W.git(r, 'rev-parse', 'HEAD').stdout.strip()
    text = '{"included": ["OASIS_OAS1_0001_MR1"]}\n'
    digest = hashlib.sha256(text.encode()).hexdigest()
    write(r, ALLOW, text)
    if case == 'separate_commits':
        commit(r, 'allowlist')
    if case != 'no_record':
        write(r, UNSEAL, json.dumps(dict(allowlist_sha256=digest[:-1]+('0' if digest[-1] != '0' else '1') if case == 'sha_one_digit' else digest)))
    if case == 'pinned_record_later':                   # exception B: the pinned record in the same commit as the allowlist and the unseal record
        commit(r, 'allowlist and unseal record')
    line = f'| P1 = {pin} |'
    write(r, RECORD, json.dumps(dict(schema=hc.PINNED_RECORD_SCHEMA, commit=pin, decision_log_line=line, written_by='lead', time_local='x', utc='x', reason='x')))
    with open(r/'docs/decision_log.md', 'a') as stream:
        stream.write(line+'\n')
    commit(r, 'inclusion')
    record, out = problems(r)
    assert out and record['code_dirs_differ'] == [ALLOW]


def test_the_allowlist_inclusion_function_on_its_own(tmp_path):
    """allowlist_inclusion_problems (the gates' half): the inclusion commit passes; nothing changed: nothing to allow."""
    r, pin = repo(tmp_path)
    found, out = hc.allowlist_inclusion_problems(pin, repo=r)
    assert out == [] and found['applies'] is True and found['same_commit'] is True and found['decision_log_line_same_commit'] is True
    found, out = hc.allowlist_inclusion_problems('HEAD', repo=r)
    assert out == [] and found['applies'] is False


def test_the_names_are_the_shared_modules():
    from bayesreg.io import code_identity as ci
    from bayesreg.io import unseal
    assert hc.LATER_AUTHORIZATION is ci.LATER_AUTHORIZATION and hc.PINNED_CODE_DIRS == ci.CODE_DIRS and hc.PINNED_RECORD_REL == 'runs/holdout_B_v1/pinned_code.json'
    assert ci.ALLOWLIST_REL == unseal.ALLOWLIST_REL == hc.ALLOWLIST_REL and ci.UNSEAL_RECORD_REL == unseal.UNSEAL_REL
    for kind in hc.AUTH_KINDS:
        name = 'holdout_B_v1_pqh_' + kind + '_7.json'
        assert re.fullmatch(hc.authorization_name('holdout', 'pqh', kind), name) and hc.LATER_AUTHORIZATION.fullmatch('splits/bayesreg_v1/'+name)
    assert hc.hi.RESUME_NAME.fullmatch('unseal_holdout_B_v1_resume_12.json') and hc.LATER_AUTHORIZATION.fullmatch('splits/bayesreg_v1/unseal_holdout_B_v1.json')
    for bad in ('splits/bayesreg_v1/holdout_B_v1_reeval_1234.json', 'splits/bayesreg_v1/unseal_holdout_B_v1.json.bak', 'splits/holdout_B_v1_reeval_1.json', 'splits/bayesreg_v1/dev_hb_resume_1.json'):
        assert not hc.LATER_AUTHORIZATION.fullmatch(bad), bad


# ---- AF (runs/20261115-af): every tool's authorization names are authorizations of rule one, HP's two (hp_prep, hp_atropos: a '_' in the tool's name) among them -------------------------------
AUTH_TOOL_FILES = {'experiments/hb/hb_core.py': 'hb', 'experiments/xb/xb_drive.py': 'xb', 'experiments/pq/pqh_pins.py': 'pqh', 'experiments/holdout/prep_stage.py': 'hp_prep',
                   'experiments/holdout/atropos_holdout.py': 'hp_atropos'}
AUTH_TOOLS = tuple(AUTH_TOOL_FILES.values())
HP_TOOLS = ('hp_prep', 'hp_atropos')


def module_tools(path):
    """The module-level string constants named TOOL of a source file (TOOL = '...', TOOL, ARM = '...', '...' or, AF review m2, TOOL: str = '...'), read on the AST: nothing is imported."""
    found = []
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'TOOL':
            found += [node.value.value] if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) else []
            continue
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            pairs = zip(target.elts, node.value.elts) if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple) else [(target, node.value)]
            found += [v.value for t, v in pairs if isinstance(t, ast.Name) and t.id == 'TOOL' and isinstance(v, ast.Constant) and isinstance(v.value, str)]
    return found


def test_every_tool_of_the_code_is_the_one_known_and_its_authorizations_are_rule_ones():
    """The tools whose authorizations holdout_common names (holdout_common.authorization_name('holdout', TOOL, kind)): read from the code itself (each file's TOOL; every name-like TOOL of
    experiments/ -- a new one fails here until it is added above; the arm table's authorization tools), and for each of them every kind and k of 1-3 digits is an authorization of rule one."""
    for rel, tool in AUTH_TOOL_FILES.items():
        assert module_tools(W.ROOT/rel) == [tool], rel
    scanned = {v for p in sorted((W.ROOT/'experiments').rglob('*.py')) for v in module_tools(p) if re.fullmatch(r'[A-Za-z0-9_]+', v)}
    assert scanned == set(AUTH_TOOLS), scanned
    table = json.loads((W.ROOT/'configs/frozen/holdout_B_v1_arm_table.json').read_text(encoding='utf-8'))
    assert {a['authorization']['tool'] for a in table['arms']} <= set(AUTH_TOOLS)
    for tool in AUTH_TOOLS:
        for kind in hc.AUTH_KINDS:
            for k in ('0', '1', '7', '42', '999'):
                name = f'holdout_B_v1_{tool}_{kind}_{k}.json'
                assert re.fullmatch(hc.authorization_name('holdout', tool, kind), name), name
                assert hc.LATER_AUTHORIZATION.fullmatch(f'{hc.AUTH_DIRS["holdout"]}/{name}'), name


AUTH_CALLS = ('load_resume', 'find_authorizations', 'rerun_authorization', 'no_resume_authorization', 'declined_refusal')      # holdout_common's readers of a tool's authorizations


def tool_arguments(path):
    """Every call of AUTH_CALLS (by name or as an attribute) in a source file that passes tool= (or a ** expansion that could), read on the AST: (line, the call's name, the form of its tool
    argument): 'TOOL' (a name or an attribute TOOL: a module's constant, scanned by module_tools), 'spec["tool"]' (the evaluation entry: the pinned arm table's authorization.tool), 'parameter'
    (a parameter of the enclosing function: its own callers are scanned), '**' or 'other'.  Nothing is imported."""
    out = []

    def walk(node, params):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            params = params | {x.arg for x in a.posonlyargs+a.args+a.kwonlyargs+[v for v in (a.vararg, a.kwarg) if v is not None]}
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
            if name in AUTH_CALLS:
                for kw in node.keywords:
                    if kw.arg is None:
                        out.append((node.lineno, name, '**'))
                    elif kw.arg == 'tool':
                        v = kw.value
                        if (isinstance(v, ast.Name) and v.id == 'TOOL') or (isinstance(v, ast.Attribute) and v.attr == 'TOOL'):
                            form = 'TOOL'
                        elif isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name) and v.value.id == 'spec' and isinstance(v.slice, ast.Constant) and v.slice.value == 'tool':
                            form = 'spec["tool"]'
                        elif isinstance(v, ast.Name) and v.id in params:
                            form = 'parameter'
                        else:
                            form = 'other'
                        out.append((node.lineno, name, form))
        for child in ast.iter_child_nodes(node):
            walk(child, params)
    walk(ast.parse(path.read_text(encoding='utf-8')), frozenset())
    return out


def test_every_authorization_reader_is_given_a_tool_of_the_code():
    """AF review m2: module_tools reads the tools' constants; a tool passed as a literal (tool='...') to holdout_common's readers of authorizations would name authorizations of a tool no
    constant declares.  Every call of AUTH_CALLS with tool= in bayesreg/, experiments/ and scripts/ passes a TOOL constant, the arm table's spec['tool'] or a parameter of its own function;
    no call hides it in a ** expansion.  Each reader is called so at least once (the scan is not empty)."""
    found = {str(p.relative_to(W.ROOT)): tool_arguments(p) for d in ('bayesreg', 'experiments', 'scripts') for p in sorted((W.ROOT/d).rglob('*.py'))}
    bad = [(rel, line, name, form) for rel, rows in found.items() for line, name, form in rows if form not in ('TOOL', 'spec["tool"]', 'parameter')]
    assert bad == [], bad
    called = {name for rows in found.values() for _, name, _ in rows}
    assert called == set(AUTH_CALLS), called
    forms = {rel: sorted({form for _, _, form in rows}) for rel, rows in found.items() if rows}
    assert forms['bayesreg/eval/holdout.py'] == ['spec["tool"]'] and forms['experiments/holdout/holdout_common.py'] == ['parameter'] and forms['experiments/xb/xb_drive.py'] == ['TOOL'], forms


LATER_YES = ('holdout_B_v1_hp_prep_resume_1.json', 'holdout_B_v1_hp_prep_rerun_12.json', 'holdout_B_v1_hp_prep_no_resume_123.json', 'holdout_B_v1_hp_atropos_resume_9.json',
             'holdout_B_v1_hp_atropos_rerun_1.json', 'holdout_B_v1_hp_atropos_no_resume_77.json', 'holdout_B_v1_hb_resume_1.json', 'holdout_B_v1_xb_rerun_2.json',
             'holdout_B_v1_pqh_no_resume_3.json',
             'holdout_B_v1_ab_cd_ef_resume_1.json', 'holdout_B_v1_a1_b2_c3_no_resume_42.json',              # AF review m1: a synthetic tool of three segments (none of the code's)
             'unseal_holdout_B_v1.json', 'unseal_holdout_B_v1_resume_1.json', 'unseal_holdout_B_v1_resume_123.json', 'holdout_B_v1_reeval_1.json', 'holdout_B_v1_reeval_999.json')
LATER_NO = ('splits/bayesreg_v1/holdout_B_v1_hp_prep_7.json',                          # no kind
            'splits/bayesreg_v1/holdout_B_v1_hp_prep__7.json',
            'splits/bayesreg_v1/holdout_B_v1_hp_prep_retry_7.json',                    # not a kind
            'splits/bayesreg_v1/holdout_B_v1_hp_prep_resume.json',                     # no k
            'splits/bayesreg_v1/holdout_B_v1_hp_prep_resume_.json',
            'splits/bayesreg_v1/holdout_B_v1_hp_prep_resume_1234.json',                # k of 4 digits
            'splits/bayesreg_v1/holdout_B_v1_hp_atropos_no_resume_1000.json',
            'splits/bayesreg_v1/holdout_B_v1_hp_prep_resume_1_extra.json',             # a suffix more
            'splits/bayesreg_v1/holdout_B_v1_hp_prep_resume_1.json.bak',
            'splits/bayesreg_v1/holdout_B_v1_hp_atropos_rerun_1.jsonl',
            'splits/bayesreg_v1/holdout_B_v1_hp_prep_resume_1.json\n',
            'splits/bayesreg_v1/holdout_B_v1_hp_prep_resume_1',
            'splits/bayesreg_v1/holdout_B_v1_hp_prep_resume_1xjson',                   # AF review m1: '.' before json is a dot (the pattern's \.json), in every branch
            'splits/bayesreg_v1/unseal_holdout_B_v1_resume_1xjson',
            'splits/bayesreg_v1/unseal_holdout_B_v1xjson',
            'splits/bayesreg_v1/holdout_B_v1_reeval_1xjson',
            'splits/bayesreg_v1/holdout_B_v1__hp_prep_resume_1.json',                  # an empty segment of the tool's name: authorization_name gives none of these for a tool of the code
            'splits/bayesreg_v1/holdout_B_v1_hp__prep_resume_1.json',
            'splits/bayesreg_v1/holdout_B_v1_hp_prep__resume_1.json',
            'splits/bayesreg_v1/holdout_B_v1_HP_prep_resume_1.json',                   # not lower case
            'splits/bayesreg_v1/holdout_B_v1_hp-prep_resume_1.json',
            'splits/bayesreg_v1/holdout_B_v1_hp.prep_resume_1.json',
            'splits/bayesreg_v1/holdout_B_v1_hp/prep_resume_1.json',                   # another directory
            'splits/bayesreg_v1/sub/holdout_B_v1_hp_prep_resume_1.json',
            'splits/holdout_B_v1_hp_prep_resume_1.json',
            'splits/bayesreg_v2/holdout_B_v1_hp_prep_resume_1.json',
            'runs/20261107-xb/authorizations/holdout_B_v1_hp_prep_resume_1.json',
            'x/splits/bayesreg_v1/holdout_B_v1_hp_prep_resume_1.json',
            'splits/bayesreg_v1/dev_hp_prep_resume_1.json',                            # the development's prefix
            'splits/bayesreg_v1/holdout_B_v2_hp_prep_resume_1.json',
            'splits/bayesreg_v1/unseal_holdout_B_v1_hp_prep_resume_1.json',            # the other branches unchanged
            'splits/bayesreg_v1/unseal_holdout_B_v1_rerun_1.json',
            'splits/bayesreg_v1/unseal_holdout_B_v1_resume_1234.json',
            'splits/bayesreg_v1/unseal_holdout_B_v1_resume.json',
            'splits/bayesreg_v1/holdout_B_v1_reeval_1234.json',
            'splits/bayesreg_v1/holdout_B_v1_reeval.json',
            'splits/bayesreg_v1/holdout_B_v1.json',
            'splits/bayesreg_v1/holdout_B_pairs.tsv', 'splits/bayesreg_v1/holdout_B_subjects.txt', 'splits/bayesreg_v1/SHA256SUMS', 'splits/bayesreg_v1/folds.json')


@pytest.mark.parametrize('name', LATER_YES)
def test_these_names_are_authorizations_of_rule_one(name):
    assert hc.LATER_AUTHORIZATION.fullmatch('splits/bayesreg_v1/'+name), name


@pytest.mark.parametrize('path', LATER_NO, ids=[repr(p) for p in LATER_NO])
def test_these_paths_are_not_authorizations_of_rule_one(path):
    assert not hc.LATER_AUTHORIZATION.fullmatch(path), path


@pytest.mark.parametrize('kind', ('resume', 'rerun', 'no_resume'))
def test_a_tool_of_three_segments_is_an_authorization_of_rule_one(kind):
    """AF review m1: a tool's name of any number of lower-case segments (synthetic tools of three, none of the code's) is taken by holdout_common.authorization_name's pattern and by rule
    one alike (a tightening of rule one to the code's tools or to two segments fails here; AF review R3, R4)."""
    assert kind in hc.AUTH_KINDS
    for tool in ('ab_cd_ef', 'hp_prep_x', 'a1_b2_c3'):
        for k in ('1', '42', '999'):
            name = f'holdout_B_v1_{tool}_{kind}_{k}.json'
            assert re.fullmatch(hc.authorization_name('holdout', tool, kind), name), name
            assert hc.LATER_AUTHORIZATION.fullmatch(f'{hc.AUTH_DIRS["holdout"]}/{name}'), name


def test_the_other_branches_are_their_own_modules_patterns():
    """HI's resume record (hi_core.RESUME_NAME) and EV's reevaluation record (bayesreg.eval.holdout.REEVAL_NAME): the same names in and out as rule one's branches for them."""
    from bayesreg.eval import holdout as ev
    names = [f'unseal_holdout_B_v1_resume_{k}.json' for k in ('', '0', '1', '12', '123', '1234', 'x')] + [f'holdout_B_v1_reeval_{k}.json' for k in ('', '0', '1', '12', '123', '1234', 'x')]
    for name in names:
        theirs = bool(hc.hi.RESUME_NAME.fullmatch(name) or ev.REEVAL_NAME.fullmatch(name))
        assert bool(hc.LATER_AUTHORIZATION.fullmatch('splits/bayesreg_v1/'+name)) == theirs, name


@pytest.mark.parametrize('tool', HP_TOOLS)
@pytest.mark.parametrize('kind', ('resume', 'rerun', 'no_resume'))
def test_an_hp_authorization_added_after_p1_passes_rule_one(tmp_path, tool, kind):
    """The blocking case of AF: HP's authorization committed after P1 (its name from holdout_common.authorization_name) is an authorization, so rule one still holds; added once only."""
    assert kind in hc.AUTH_KINDS
    r, pin = repo(tmp_path)
    rel = f'splits/bayesreg_v1/holdout_B_v1_{tool}_{kind}_1.json'
    assert re.fullmatch(hc.authorization_name('holdout', tool, kind), rel.rsplit('/', 1)[1])
    write(r, rel, '{}\n')
    commit(r, f'add {rel}')
    record, out = problems(r)
    assert out == [], out
    assert [c for row in record['commits'] for c in row['changes'] if c['path'] == rel] == [dict(status='A', path=rel, problem=None)]
    write(r, rel, '{"changed": 1}\n')
    commit(r, f'change {rel}')
    record, out = problems(r)
    assert len(out) == 1 and f'M {rel}: authorizations and records are only added, once' in out[0], out


@pytest.mark.parametrize('rel', ['splits/bayesreg_v1/holdout_B_v1_hp_prep_resume_1_extra.json', 'splits/bayesreg_v1/holdout_B_v1_hp_atropos_resume_1234.json',
                                 'splits/bayesreg_v1/holdout_B_v1_hp__prep_resume_1.json',
                                 'splits/bayesreg_v1/holdout_B_v1_hp_prep_resume_1.json.bak'])     # AF review m1: a name that begins as an authorization: rule one asks fullmatch, not match
def test_a_near_miss_of_an_hp_authorization_after_p1_is_a_problem(tmp_path, rel):
    r, pin = repo(tmp_path)
    write(r, rel, '{}\n')
    commit(r, f'add {rel}')
    record, out = problems(r)
    assert len(out) == 1 and f'A {rel}: not an authorization under splits/bayesreg_v1/' in out[0], out


GIT_STEPS = [('show', 'HEAD:runs/holdout_B_v1/pinned_code.json'), ('rev-parse', '--verify', '--quiet', 'HEAD^{commit}'), ('merge-base',), ('rev-list',), ('diff-tree',), ('diff', '--name-only'),
             ('ls-tree', '-r'), ('ls-files', '-z', '--others', '--', 'bayesreg'), ('ls-files', '-z', '--others', '--directory'), ('status',)]


@pytest.mark.parametrize('step', GIT_STEPS, ids=['-'.join(s[:2]) for s in GIT_STEPS])
def test_a_failing_git_call_is_a_problem_never_an_exception(tmp_path, step):
    r, pin = repo(tmp_path)
    write(r, 'runs/holdout_B_v1/x.json', '{}')
    commit(r)

    def run(args, **kwargs):
        if tuple(args[6:6+len(step)]) == step:
            return subprocess.CompletedProcess(args, 128, '' if kwargs.get('text') else b'', 'test: git failed' if kwargs.get('text') else b'test: git failed')
        return subprocess.run(args, **kwargs)
    record, out = problems(r, run=run)
    assert out and any('test: git failed' in p for p in out), out


def test_an_oserror_of_git_is_a_problem(tmp_path):
    r, pin = repo(tmp_path)

    def run(args, **kwargs):
        raise FileNotFoundError('git')
    record, out = problems(r, run=run)
    assert out and record['head'] is None and record['pinned'] is None
