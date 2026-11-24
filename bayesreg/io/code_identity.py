"""The code identity of the hold-out: every rule in one place (decision_log 2026-11-10 about 2:45 AM CST N3, about 2:59 AM CST (1)-(4), (7); the lead's item 14 of HB's revision round; contract C24.3
(revision 9 draft); prereg revision 2 items 9 and 18).  The standard library and git only; nothing of experiments/ is imported, so that bayesreg (the evaluation entry, holdout_decide) and the
experiments' tools (HB, HP, PQH, XB through holdout_common) call the same functions.

The code: the five code directories (CODE_DIRS), the code tree's root files Python could import (ROOT_IMPORTABLE: *.py, *.pyc, *.pth, *.so, sitecustomize*, usercustomize*), the root's committed
symbolic links (mode 120000; HB diff review n1) and any top-level directory the base commit does not have (every unit runs with the code tree's root on PYTHONPATH); of the untracked or ignored
root entries, an importable name and a directory, or a link to one, holding __init__.py -- a link to anything else (the main checkout's ignored data/ link) is not code (HB diff review 2 r1,
C24.3-3).

Rule one (a unit; unit_identity_problems): the pinned commit P1 is read from the committed record runs/holdout_B_v1/pinned_code.json at HEAD (PINNED_RECORD_REL; never HEAD by default; a commit
given by the caller is only compared with it); then every commit of P1..HEAD, one by one (not only the net difference):
  - nothing of the code changes, but configs/bayesreg/dev_subjects.json once, modified (M), in the commit that adds the unseal record and runs/holdout_B_v1/pinned_code.json, its sha256 the
    unseal record's allowlist_sha256 (C23.3-6; contract 9b C24.3-5 exception B);
  - the commit that adds runs/holdout_B_v1/pinned_code.json appends the record's decision_log_line to docs/decision_log.md (C24.3-9: one commit; HB diff review n6);
  - an authorization file under splits/bayesreg_v1/ (LATER_AUTHORIZATION, the unseal record among them) and a file under runs/holdout_B_v1/ are only added, once, never changed or deleted after;
  - docs/decision_log.md is only appended to: its old bytes a prefix of its new bytes;
  - nothing else changes; a merge commit in the range is refused (checked by hand);
and the working tree: no tracked change at all (git status -z parsed with a rename's old path as a field of its own, n7), no untracked or ignored file in the code directories but a compiled file
beside its source, no untracked or ignored importable file at the root, the code directories the pinned commit's but the allowlist (its one allowed change).
Rule two (a gate; gate_identity_problems): from the gate's commit to HEAD the code changes only by the assignment statements of the four decision constants (DECISION_CONSTANTS) and by the allowlist
under allowlist_inclusion_problems; a constant's assignment is judged on the AST (decision_assignment_problem); nothing outside the code is checked.
git is read only (git show for a commit's files); a git failure is a problem, never an exception.

The LPBA40 second test set (contract C26.9, D149; LP-C2): the same functions with another set of constants, a Profile (C24.3-7: one set of functions, the constants apart).
`profile=None` is the hold-out's (HOLDOUT: this module's constants, read at call time; its behaviour and tests unchanged); LPBA40 is P_L from runs/lpba40_test_v1/pinned_code.json
(schema bayesreg.lpba40.pinned_code.v1), the authorizations splits/lpba40_test_v1/lpba40_test_v1_<tool>_<resume|rerun|no_resume>_<k>.json and
lpba40_test_v1_reeval_<k>.json (D148), the records runs/lpba40_test_v1/, and no allowlist exception (exception B is the unsealing's: LPBA40 has no unseal record, so its allowlist
is code like any other file).  pl_choice_problems is C26.9-3's check of P_L against the feasibility commit (only docs/ and runs/ may change between them).
"""
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[2]                     # the code tree this bayesreg package is in
CODE_DIRS = ('bayesreg', 'experiments', 'jd_curl', 'configs', 'scripts')
PINNED_CODE_DIRS = CODE_DIRS
ROOT_IMPORTABLE = re.compile(r'[^/]+\.(py|pyc|pth|so)|(sitecustomize|usercustomize)(\..*)?', re.ASCII)
ALLOWLIST_REL = 'configs/bayesreg/dev_subjects.json'            # bayesreg.io.unseal.ALLOWLIST_REL (a test compares)
UNSEAL_RECORD_REL = 'splits/bayesreg_v1/unseal_holdout_B_v1.json'     # bayesreg.io.unseal.UNSEAL_REL (a test compares)
PINNED_RECORD_REL = 'runs/holdout_B_v1/pinned_code.json'        # the one source of P1 (decision_log 2026-11-10 ~2:59 AM (1))
PINNED_RECORD_SCHEMA = 'bayesreg.holdout.pinned_code.v1'
PINNED_RECORD_KEYS = ('schema', 'commit', 'decision_log_line', 'written_by', 'time_local', 'utc', 'reason')
PINNED_RECORD_WRITERS = ('lead', 'BZ', "lead on BZ's behalf")
LATER_AUTHORIZATION = re.compile(r'splits/bayesreg_v1/(holdout_B_v1_[a-z0-9]+(?:_[a-z0-9]+)*_(resume|rerun|no_resume)_[0-9]{1,3}'      # holdout_common.authorization_name('holdout', <any tool>, <AUTH_KINDS>); a tool's name is lower-case segments joined by '_' (hb, xb, pqh, hp_prep, hp_atropos: AF, runs/20261115-af; only fullmatch is asked, no group read)
                                 r'|unseal_holdout_B_v1_resume_[0-9]{1,3}'                                              # HI's resume record (hi_core.RESUME_NAME)
                                 r'|unseal_holdout_B_v1'                                                                # the unseal record itself (added once)
                                 r'|holdout_B_v1_reeval_[0-9]{1,3})\.json', re.ASCII)                                 # EV's reevaluation record (bayesreg.eval.holdout.REEVAL_NAME)
LATER_RECORDS = 'runs/holdout_B_v1/'
LATER_DECISION_LOG = 'docs/decision_log.md'
DECISION_CONSTANTS = ('TRUTH_READER_DECISION', 'PQ_REPLAY_DECISION', 'PREP_DEVCHECK_DECISION', 'ATROPOS_DEVCHECK_DECISION')
DECISION_HOMES = {'TRUTH_READER_DECISION': 'bayesreg/eval/holdout.py', 'PQ_REPLAY_DECISION': 'experiments/pq/pqh_pins.py',          # each constant only in its own file (contract 9b C24.3-6 (a))
                  'PREP_DEVCHECK_DECISION': 'experiments/holdout/prep_stage.py', 'ATROPOS_DEVCHECK_DECISION': 'experiments/holdout/atropos_holdout.py'}
CACHE_FILE = re.compile(r'(?P<dir>(?:[^/]+/)*)__pycache__/(?P<module>[A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z0-9_-]+(?:\.opt-[12])?\.pyc')
SHA = re.compile(r'[0-9a-f]{40}')

# ---- the LPBA40 second test set's constants (contract C26.9-2, -4; D148, D149) -----------------------------------------------------------------------
LPBA40_PINNED_RECORD_REL = 'runs/lpba40_test_v1/pinned_code.json'        # the one source of P_L (C26.9-2)
LPBA40_PINNED_RECORD_SCHEMA = 'bayesreg.lpba40.pinned_code.v1'
LPBA40_LATER_AUTHORIZATION = re.compile(r'splits/lpba40_test_v1/lpba40_test_v1_(?:[a-z0-9]+(?:_[a-z0-9]+)*_(?:resume|rerun|no_resume)|reeval)_[0-9]{1,3}\.json',
                                        re.ASCII)       # holdout_common.authorization_name('lpba40_test', <tool>, <kind>) and the evaluation's reeval (bayesreg.eval.lpba_eval)
LPBA40_LATER_RECORDS = 'runs/lpba40_test_v1/'
PL_CHOICE_PREFIXES = ('docs/', 'runs/')                                    # C26.9-3: P_L is the feasibility commit or one after it changing only these


@dataclass(frozen=True)
class Profile:
    """The constants of rule one for one test set (C24.3-7: the functions shared, the constants apart).  allowlist_exception: exception B of C24.3-5 (the allowlist modified once,
    with the unseal record) -- the hold-out's only."""
    name: str
    pinned_record_rel: str
    pinned_record_schema: str
    later_authorization: object
    later_records: str
    allowlist_exception: bool
    authorizations: str


LPBA40 = Profile('lpba40_test_v1', LPBA40_PINNED_RECORD_REL, LPBA40_PINNED_RECORD_SCHEMA, LPBA40_LATER_AUTHORIZATION, LPBA40_LATER_RECORDS, False, 'splits/lpba40_test_v1/')


def holdout_profile():
    """The hold-out's profile from this module's constants, read at call time (profile=None everywhere)."""
    return Profile('holdout_B_v1', PINNED_RECORD_REL, PINNED_RECORD_SCHEMA, LATER_AUTHORIZATION, LATER_RECORDS, True, 'splits/bayesreg_v1/')


def _profile(profile):
    return holdout_profile() if profile is None else profile


# ---- small parts ------------------------------------------------------------------------------------------------------------------------------------
def git_runner(repo, run=None):
    """(text, raw): git in `repo` with --literal-pathspecs and core.quotepath=off; text -> (rc, stdout str, stderr), raw -> (rc, stdout bytes, stderr); never raises."""
    run = subprocess.run if run is None else run

    def call(args, text):
        try:
            done = run(['git', '--literal-pathspecs', '-c', 'core.quotepath=off', '-C', str(repo), *args], capture_output=True, text=text, timeout=120)
            out, err = done.stdout, done.stderr
            if not text:                                # LP-C2: a runner that answers in text (a test's stand-in for git) is read as bytes, never an exception
                out = out.encode('utf-8') if isinstance(out, str) else (out or b'')
                err = err.decode('utf-8', 'replace') if isinstance(err, bytes) else (err or '')
            return done.returncode, out, (err or '').strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return 127, '' if text else b'', f'{type(exc).__name__}: {exc}'
    return (lambda *a: call(a, True)), (lambda *a: call(a, False))


def strict_pairs(pairs):
    """json's object_pairs_hook refusing duplicate keys."""
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f'duplicate keys {sorted({k for k in keys if keys.count(k) > 1})}')
    return dict(pairs)


def _paths(out):
    return [p for p in out.split('\0') if p]


def _commit(git, ref):
    rc, out, err = git('rev-parse', '--verify', '--quiet', f'{ref}^{{commit}}') if ref else (1, '', 'no commit given')
    return (out.strip() if rc == 0 else None), err


def code_dir_file_problem(path, sources):
    """None for a compiled file <dir>/__pycache__/<module>.<tag>[.opt-N].pyc whose source <dir>/<module>.py is a file of the commit (`sources`); the reason for every other untracked or ignored file of
    a code directory (a module there, or a compiled file without its source, would run without a commit)."""
    match = CACHE_FILE.fullmatch(path)
    if match is None:
        return f'{path}: an untracked or ignored file in a code directory'
    source = f"{match['dir']}{match['module']}.py"
    return None if source in sources else f'{path}: a compiled file whose source {source} is not a file of the commit'


def code_kind(path, base_top, links=()):
    """'dir' (in a code directory), 'root' (a root file Python could import), 'root_link' (a symbolic link at the root, `links`: it may point at a package; HB diff review n1), 'new_top' (under a
    top-level directory the base commit does not have), or None (not code)."""
    top = path.split('/', 1)[0]
    if top in CODE_DIRS:
        return 'dir'
    if '/' not in path:
        return 'root' if ROOT_IMPORTABLE.fullmatch(path) else 'root_link' if path in links else None
    return 'new_top' if top not in base_top else None


def _top_entries(git, commit):
    rc, out, err = git('ls-tree', '-z', '--name-only', commit)
    return set(_paths(out)), (None if rc == 0 else f'git ls-tree {commit}: exit {rc} {err}')


def _root_links(git, commit):
    """The root entries of `commit` that are symbolic links (mode 120000), (set, problem or None)."""
    rc, out, err = git('ls-tree', '-z', commit)
    links = {entry.split('\t', 1)[1] for entry in _paths(out) if '\t' in entry and entry.split(' ', 1)[0] == '120000'}
    return links, (None if rc == 0 else f'git ls-tree {commit}: exit {rc} {err}')


def _status_paths(out):
    """The paths of `git status --porcelain=v1 -z` (HB diff review n7): an entry is 'XY PATH\\0'; a rename or copy (R or C in XY) is 'XY NEW\\0OLD\\0', its old path a field of its own (no 'XY ')."""
    fields, paths, i = out.split('\0'), [], 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if len(entry) < 4:
            continue
        paths.append(entry[3:])
        if 'R' in entry[:2] or 'C' in entry[:2]:
            if i < len(fields) and fields[i]:
                paths.append(fields[i])
            i += 1
    return paths


def _untracked_importables(repo, out):
    """Rule one's and rule two's root check (HB diff review n1; HB diff review 2 r1, the lead's ruling (a) of 2026-11-10 ~10:04 AM CST, contract C24.3-3 as written): of `git ls-files -z
    --others --directory` (untracked and ignored), the root entries Python could import: an importable name (ROOT_IMPORTABLE), or a directory -- followed through a symbolic link, not judged by
    git's trailing '/' -- holding __init__.py.  A symbolic link is code only so: one to a package directory (or under an importable name); a link to a directory without __init__.py (the main
    checkout's ignored data/ link), to a file under a name Python cannot import, or a dangling link is not (a directory without __init__.py imports only as a namespace package, which shadows
    no installed regular package: C24.3-3)."""
    found = []
    for entry in _paths(out):
        name = entry.rstrip('/')
        if '/' in name:
            continue
        path = Path(repo)/name
        if ROOT_IMPORTABLE.fullmatch(name) or (path.is_dir() and (path/'__init__.py').is_file()):
            found.append(name)
    return found


def _decision_line_problem(raw, commit, profile=None):
    """C24.3-9 (HB diff review n6): the commit that adds runs/holdout_B_v1/pinned_code.json appends its decision_log_line to docs/decision_log.md in the same commit (the log's old bytes a prefix of
    the new, the line one of the appended lines).  None or the problem."""
    rel = _profile(profile).pinned_record_rel
    rc_r, data, err_r = raw('show', f'{commit}:{rel}')
    try:
        line = json.loads(data.decode('utf-8'), object_pairs_hook=strict_pairs).get('decision_log_line') if rc_r == 0 else None
    except (UnicodeDecodeError, ValueError, AttributeError):
        line = None
    if not isinstance(line, str) or not line.strip():
        return f'{commit}: {rel} has no decision_log_line ({err_r})'
    rc_o, old, _ = raw('show', f'{commit}^:{LATER_DECISION_LOG}')
    rc_n, new, err_n = raw('show', f'{commit}:{LATER_DECISION_LOG}')
    old = old if rc_o == 0 else b''
    if rc_n or not new.startswith(old) or line not in new[len(old):].decode('utf-8', 'replace').splitlines():
        return f'{commit}: the decision log line of {rel} is not appended to {LATER_DECISION_LOG} in the commit that adds the record (C24.3-9: one commit) ({err_n})'
    return None


# ---- the allowlist's allowance (C23.3-6) ---------------------------------------------------------------------------------------------------------
def allowlist_inclusion_problems(pinned, *, repo=None, run=None, head='HEAD'):
    """(record, problems): the one narrow allowance (decision_log 2026-11-10 I1, N3; contract C23.3-6).  From `pinned` to `head`:
      - configs/bayesreg/dev_subjects.json (ALLOWLIST_REL) unchanged: nothing to allow (record applies false, problems []);
      - else it is allowed only when: its change is a modification (M; not A, not D); exactly one commit of the range touches it (git log --full-history --no-merges: a gate's range on main
        may hold merges, contract 9b C24.3-5 exception B); that commit adds (A) the unseal record (UNSEAL_RECORD_REL) and runs/holdout_B_v1/pinned_code.json (PINNED_RECORD_REL) and appends the
        pinned record's decision_log_line to docs/decision_log.md (C24.3-9: one commit; HB diff review n6), and the range adds both records (A); the allowlist's sha256 at `head` equals the
        record's allowlist_sha256 (the record read at `head`, strict JSON); no other file under configs/ changed in the range.
    Reads git only; never raises.  record keys: rule, allowlist, unseal_record, pinned_given, head_given, pinned, head, applies, allowlist_status, unseal_record_status, other_configs_changes,
    pinned_record_status, allowlist_commits, same_commit, decision_log_line_same_commit, allowlist_sha256, record_allowlist_sha256, passed, problems."""
    repo = Path(REPO if repo is None else repo)
    git, raw = git_runner(repo, run)
    record = dict(rule='decision_log 2026-11-10 I1, C23.3-6: the allowlist modified once, in the commit that adds the unseal record, its sha256 the record\'s allowlist_sha256; nothing else under configs/',
                  allowlist=ALLOWLIST_REL, unseal_record=UNSEAL_RECORD_REL, pinned_given=pinned, head_given=head, applies=False)
    problems = []
    (pin, err_p), (top, err_h) = _commit(git, pinned), _commit(git, head)
    record.update(pinned=pin, head=top)
    if pin is None or top is None:
        problems.append(f'git rev-parse: pinned {pinned!r} -> {pin} ({err_p}), head {head!r} -> {top} ({err_h})')
        return dict(record, passed=False, problems=problems), problems
    rc_d, out, err_d = git('diff', '--name-status', '--no-renames', '-z', pin, top)
    fields = _paths(out)
    if rc_d or len(fields) % 2:
        problems.append(f'git diff --name-status {pin} {top}: exit {rc_d} {err_d}')
        return dict(record, passed=False, problems=problems), problems
    changes = {fields[i+1]: fields[i] for i in range(0, len(fields), 2)}
    status = changes.get(ALLOWLIST_REL)
    record['allowlist_status'] = status
    if status is None:
        return dict(record, passed=True, problems=[]), []
    record['applies'] = True
    if status != 'M':
        problems.append(f'{status} {ALLOWLIST_REL}: only a modification of the allowlist is allowed')
    record['unseal_record_status'] = changes.get(UNSEAL_RECORD_REL)
    record['pinned_record_status'] = changes.get(PINNED_RECORD_REL)
    for rel in (UNSEAL_RECORD_REL, PINNED_RECORD_REL):
        if changes.get(rel) != 'A':
            problems.append(f'{rel}: not added in {pin}..{top} ({changes.get(rel)}): the allowlist changes only with the unseal record and the pinned record (C23.3-6; C24.3-5 exception B)')
    others = sorted(f'{st} {path}' for path, st in changes.items() if path.startswith('configs/') and path != ALLOWLIST_REL)
    record['other_configs_changes'] = others
    if others:
        problems.append(f'other changes under configs/: {others}')
    rc_l, out_l, err_l = git('log', '--full-history', '--no-merges', '--format=%H', f'{pin}..{top}', '--', ALLOWLIST_REL)
    commits = out_l.split()
    record['allowlist_commits'] = commits
    if rc_l or len(commits) != 1:
        problems.append(f'{ALLOWLIST_REL}: {len(commits)} commits of the range touch it (exactly one is allowed) {err_l}')
    else:
        rc_c, out_c, err_c = git('show', '--name-status', '--no-renames', '-z', '--format=', commits[0], '--', UNSEAL_RECORD_REL, PINNED_RECORD_REL)
        same = _paths(out_c)
        added = {same[i+1]: same[i] for i in range(0, len(same)-1, 2)}
        record['same_commit'] = rc_c == 0 and added.get(UNSEAL_RECORD_REL) == 'A' and added.get(PINNED_RECORD_REL) == 'A'
        if not record['same_commit']:
            problems.append(f'{commits[0]} modifies the allowlist but does not add both {UNSEAL_RECORD_REL} and {PINNED_RECORD_REL} ({same} {err_c}): one commit (C23.3-6; exception B)')
        else:
            line_problem = _decision_line_problem(raw, commits[0])
            record['decision_log_line_same_commit'] = line_problem is None
            if line_problem:
                problems.append(line_problem)
    problem, shas = _allowlist_sha_problem(raw, top)
    record.update(allowlist_sha256=shas[0], record_allowlist_sha256=shas[1])
    if problem:
        problems.append(problem)
    return dict(record, passed=not problems, problems=problems), problems


def _allowlist_sha_problem(raw, commit):
    """(problem or None, (the allowlist's sha256 at commit, the unseal record's allowlist_sha256 at commit))."""
    rc_a, blob, err_a = raw('show', f'{commit}:{ALLOWLIST_REL}')
    rc_r, text, err_r = raw('show', f'{commit}:{UNSEAL_RECORD_REL}')
    sha = hashlib.sha256(blob).hexdigest() if rc_a == 0 else None
    want = None
    if rc_r == 0:
        try:
            doc = json.loads(text.decode('utf-8'), object_pairs_hook=strict_pairs)
            want = doc.get('allowlist_sha256') if isinstance(doc, dict) else None
        except (UnicodeDecodeError, ValueError) as exc:
            return f'{UNSEAL_RECORD_REL} at {commit}: not strict JSON ({exc})', (sha, None)
    if sha is None or want is None or sha != want:
        return f'{ALLOWLIST_REL} at {commit} has sha256 {sha} ({err_a}); the unseal record names {want} ({err_r})', (sha, want)
    return None, (sha, want)


# ---- rule one: P1 from the committed record; every commit of P1..HEAD ------------------------------------------------------------------------------
def pinned_record(*, repo=None, run=None, head='HEAD', profile=None):
    """(commit, record, problems): runs/holdout_B_v1/pinned_code.json at `head` (git show; strict JSON; exactly PINNED_RECORD_KEYS; schema PINNED_RECORD_SCHEMA; commit 40 hex and a commit of the
    tree; decision_log_line a non-empty line that holds the commit and is a line of docs/decision_log.md at `head`; written_by one of PINNED_RECORD_WRITERS; time_local, utc, reason non-empty).
    No record: (None, ..., [why]) -- never HEAD by default."""
    repo = Path(REPO if repo is None else repo)
    git, raw = git_runner(repo, run)
    p = _profile(profile)
    rel = p.pinned_record_rel
    rc, data, err = raw('show', f'{head}:{rel}')
    if rc:
        return None, None, [f'{rel} is not committed at {head} ({err}): no pinned commit (it is never HEAD by default)']
    try:
        doc = json.loads(data.decode('utf-8'), object_pairs_hook=strict_pairs)
    except (UnicodeDecodeError, ValueError) as exc:
        return None, None, [f'{rel}: not strict JSON ({exc})']
    if not isinstance(doc, dict) or sorted(doc) != sorted(PINNED_RECORD_KEYS):
        return None, doc, [f'{rel}: exactly the keys {list(PINNED_RECORD_KEYS)} are required']
    problems = []
    if doc['schema'] != p.pinned_record_schema:
        problems.append(f"schema {doc['schema']!r}")
    commit = doc['commit'] if isinstance(doc['commit'], str) and SHA.fullmatch(doc['commit']) else None
    if commit is None or _commit(git, commit)[0] != commit:
        problems.append(f"commit {doc['commit']!r}: 40 lowercase hex, a commit of the tree")
    if doc['written_by'] not in PINNED_RECORD_WRITERS:
        problems.append(f"written_by {doc['written_by']!r}")
    problems += [f'{k}: non-empty' for k in ('time_local', 'utc', 'reason') if not isinstance(doc[k], str) or not doc[k].strip()]
    line = doc['decision_log_line']
    rc_l, log, err_l = raw('show', f'{head}:{LATER_DECISION_LOG}')
    if not (isinstance(line, str) and line.strip() and '\n' not in line and commit and commit in line):
        problems.append('decision_log_line: one non-empty line holding the commit')
    elif rc_l or line not in log.decode('utf-8', 'replace').splitlines():
        problems.append(f'decision_log_line: not a line of {LATER_DECISION_LOG} at {head} ({err_l})')
    return (None if problems else commit), doc, problems


def _commit_changes(git, commit, parent):
    rc, out, err = git('diff-tree', '-r', '-z', '--no-renames', '--name-status', parent, commit)
    fields = _paths(out)
    if rc or len(fields) % 2:
        return None, f'git diff-tree {parent} {commit}: exit {rc} {err}'
    return [(fields[i], fields[i+1]) for i in range(0, len(fields), 2)], None


def later_commits_problems(pinned, head, *, repo=None, run=None, profile=None):
    """(rows, problems): rule one's commit by commit check of pinned..head (module docstring).  rows: per commit {commit, changes: [{status, path, problem}]}, and decision_log_line {path, problem}
    in the commit that adds the pinned record (C24.3-9)."""
    repo = Path(REPO if repo is None else repo)
    git, raw = git_runner(repo, run)
    p = _profile(profile)
    problems, rows = [], []
    rc, out, err = git('rev-list', '--reverse', '--parents', f'{pinned}..{head}')
    if rc:
        return rows, [f'git rev-list {pinned}..{head}: exit {rc} {err}']
    base_top, err_t = _top_entries(git, pinned)
    if err_t:
        problems.append(err_t)
    seen, allowlist_done = set(), False
    for line in out.split('\n'):
        if not line.strip():
            continue
        commit, *parents = line.split()
        row = dict(commit=commit, changes=[])
        rows.append(row)
        if len(parents) != 1:
            problems.append(f'{commit}: a merge commit (or a root) in {pinned}..{head}: refused, to be checked by hand')
            continue
        parent = parents[0]
        changes, err_c = _commit_changes(git, commit, parent)
        if err_c:
            problems.append(err_c)
            continue
        added = {path for status, path in changes if status == 'A'}
        links, err_l = (set(), None) if all('/' in path for _, path in changes) else _root_links(git, commit)
        if err_l:
            problems.append(err_l)
        if p.pinned_record_rel in added:                # HB diff review n6 (C24.3-9): the decision log's line in the very commit that adds the pinned record
            line_problem = _decision_line_problem(raw, commit, p)
            row['decision_log_line'] = dict(path=LATER_DECISION_LOG, problem=line_problem)
            if line_problem:
                problems.append(f'{commit[:12]}: {line_problem}')
        for status, path in changes:
            problem = None
            kind = code_kind(path, base_top, links)
            if path == ALLOWLIST_REL and p.allowlist_exception:
                if status != 'M' or allowlist_done:
                    problem = f'{status} {path}: the allowlist changes once, modified'
                elif UNSEAL_RECORD_REL not in added or PINNED_RECORD_REL not in added:
                    problem = f'{path}: modified in {commit} without both the unseal record and {PINNED_RECORD_REL} added in the same commit (C23.3-6; exception B)'
                else:
                    sha_problem, _ = _allowlist_sha_problem(raw, commit)
                    problem = None if sha_problem is None else f'{commit}: {sha_problem}'
                allowlist_done = True
            elif kind is not None:
                problem = f'{status} {path}: the code changed after the pinned commit ({kind})'
            elif p.later_authorization.fullmatch(path) or path.startswith(p.later_records):
                if status != 'A' or path in seen:
                    problem = f'{status} {path}: authorizations and records are only added, once, never changed or deleted after'
                seen.add(path)
            elif path == LATER_DECISION_LOG:
                if status == 'D':
                    problem = f'D {path}: the decision log is only appended to'
                elif status == 'M':
                    rc_o, old, _ = raw('show', f'{parent}:{path}')
                    rc_n, new, _ = raw('show', f'{commit}:{path}')
                    if rc_o or rc_n or not new.startswith(old):
                        problem = f'M {path} in {commit}: not an append (its old bytes are not a prefix of the new)'
                elif status != 'A':
                    problem = f'{status} {path}: the decision log is only appended to'
            else:
                problem = f'{status} {path}: not an authorization under {p.authorizations}, a new record under {p.later_records} or an append to {LATER_DECISION_LOG}'
            row['changes'].append(dict(status=status, path=path, problem=problem))
            if problem:
                problems.append(f'{commit[:12]}: {problem}')
    return rows, problems


def unit_identity_problems(commit=None, *, repo=None, run=None, profile=None):
    """(record, problems): rule one for one process (module docstring).  `commit`: the caller's pinned commit, only compared with the record's (None: the record's).  record keys: rule, repo,
    commit_given, pinned_record (path, document), pinned, head, pinned_is_ancestor, commits (per commit changes), code_dirs_differ, code_dirs_equal_to_pinned, files_in_code_dirs,
    importable_at_the_top, tracked_changes, tracked_clean, passed, problems."""
    repo = Path(REPO if repo is None else repo)
    git, raw = git_runner(repo, run)
    p = _profile(profile)
    if profile is None:
        rule = ('rule one (contract C24.3; decision_log 2026-11-10 ~2:59 AM CST (1)-(4)): P1 from runs/holdout_B_v1/pinned_code.json; every commit of P1..HEAD only adds authorizations '
                'and records, appends to the decision log and changes the allowlist once with the unseal record; the code (five directories, root importables, new top-level directories) '
                'unchanged; a clean working tree')
    else:
        rule = (f'rule one ({p.name}; contract C26.9-4, C24.3): P from {p.pinned_record_rel}; every commit of P..HEAD only adds authorizations ({p.authorizations}) and records '
                f'({p.later_records}) once and appends to the decision log; the code (five directories, root importables, new top-level directories) unchanged; a clean working tree')
    record = dict(rule=rule, repo=str(repo), commit_given=commit, code_dirs=list(CODE_DIRS))
    if profile is not None:
        record['profile'] = p.name
    problems = []
    head, err_h = _commit(git, 'HEAD')
    pin, doc, record_problems = pinned_record(repo=repo, run=run, profile=profile)
    record.update(head=head, pinned=pin, pinned_record=dict(path=p.pinned_record_rel, document=doc))
    problems += record_problems
    if head is None:
        problems.append(f'git rev-parse HEAD: {err_h}')
    if commit is not None and pin is not None:
        given, _ = _commit(git, commit)
        if given != pin:
            problems.append(f'the given commit {commit!r} ({given}) is not the pinned commit of {p.pinned_record_rel} ({pin}): --commit is only a check')
    if pin is None or head is None:
        return dict(record, passed=False, problems=problems), problems
    rc_a, _, err_a = git('merge-base', '--is-ancestor', pin, head)
    record['pinned_is_ancestor'] = rc_a == 0
    if rc_a != 0:
        problems.append(f'the pinned commit {pin} is not an ancestor of HEAD {head} ({err_a or "not an ancestor"})')
    record['commits'], commit_problems = later_commits_problems(pin, head, repo=repo, run=run, profile=profile)
    problems += commit_problems
    rc_w, out_w, err_w = git('diff', '--name-only', '-z', '--no-ext-diff', pin, '--', *CODE_DIRS)
    differ = sorted(_paths(out_w))
    allowed = p.allowlist_exception and any(c['path'] == ALLOWLIST_REL and c['problem'] is None for row in record['commits'] for c in row['changes'])
    record['code_dirs_differ'] = differ
    record['code_dirs_equal_to_pinned'] = rc_w == 0 and (not differ or (allowed and differ == [ALLOWLIST_REL]))
    if not record['code_dirs_equal_to_pinned']:
        problems.append(f'the working tree\'s code directories differ from the pinned commit {pin}: {differ[:10]} (git diff: exit {rc_w} {err_w})')
    rc_t, out_t, err_t = git('ls-tree', '-r', '-z', '--name-only', pin, '--', *CODE_DIRS)
    sources = set(_paths(out_t))
    rc_u, out_u, err_u = git('ls-files', '-z', '--others', '--', *CODE_DIRS)
    record['files_in_code_dirs'] = [dict(path=p, problem=code_dir_file_problem(p, sources)) for p in _paths(out_u)]
    if rc_t or rc_u:
        problems.append(f'git ls-tree / ls-files: exit {rc_t} {err_t}; exit {rc_u} {err_u}')
    bad = [f['problem'] for f in record['files_in_code_dirs'] if f['problem']]
    if bad:
        problems.append(f'{len(bad)} untracked or ignored files in the code directories: {bad[:10]}')
    rc_o, out_o, err_o = git('ls-files', '-z', '--others', '--directory')
    record['importable_at_the_top'] = _untracked_importables(repo, out_o)
    if rc_o:
        problems.append(f'git ls-files --others at the top: exit {rc_o} {err_o}')
    if record['importable_at_the_top']:
        problems.append(f'untracked or ignored importable files at the code tree\'s top level: {record["importable_at_the_top"][:10]}')
    rc_s, out_s, err_s = git('status', '--porcelain=v1', '-z', '--untracked-files=no')
    record['tracked_changes'] = _status_paths(out_s)
    record['tracked_clean'] = rc_s == 0 and not record['tracked_changes']
    if not record['tracked_clean']:
        problems.append(f'tracked files modified or staged in the working tree: {record["tracked_changes"][:10]} (git status: exit {rc_s} {err_s})')
    return dict(record, passed=not problems, problems=problems), problems


def pl_choice_problems(feasibility_commit, pinned, *, repo=None, run=None):
    """(record, problems): C26.9-3, the choice of P_L (LPBA40): `pinned` is the feasibility commit itself or a descendant of it, and `git diff --name-only` between them names only
    files under docs/ or runs/ (PL_CHOICE_PREFIXES).  Reads git only; never raises."""
    repo = Path(REPO if repo is None else repo)
    git, _ = git_runner(repo, run)
    record = dict(rule='C26.9-3: P_L = the feasibility commit, or a later commit that changes only docs/ and runs/ (git diff --name-only)', feasibility_given=feasibility_commit,
                  pinned_given=pinned, prefixes=list(PL_CHOICE_PREFIXES))
    problems = []
    (feas, err_f), (pin, err_p) = _commit(git, feasibility_commit), _commit(git, pinned)
    record.update(feasibility=feas, pinned=pin)
    if feas is None or pin is None:
        problems.append(f'git rev-parse: feasibility {feasibility_commit!r} -> {feas} ({err_f}), P_L {pinned!r} -> {pin} ({err_p})')
        return dict(record, passed=False, problems=problems), problems
    rc_a, _, err_a = git('merge-base', '--is-ancestor', feas, pin)
    record['feasibility_is_ancestor'] = rc_a == 0
    if rc_a != 0:
        problems.append(f'the feasibility commit {feas} is not P_L {pin} or an ancestor of it ({err_a or "not an ancestor"})')
    rc_d, out, err_d = git('diff', '--name-only', '-z', '--no-renames', feas, pin)
    changed = sorted(_paths(out))
    record['changed'] = changed
    if rc_d:
        problems.append(f'git diff --name-only {feas} {pin}: exit {rc_d} {err_d}')
    others = [path for path in changed if not path.startswith(PL_CHOICE_PREFIXES)]
    record['outside_docs_runs'] = others
    if others:
        problems.append(f'P_L {pin} changes files outside {list(PL_CHOICE_PREFIXES)} after the feasibility commit {feas}: {others[:10]} (C26.9-3: start again from prereg §8.1 item 8)')
    return dict(record, passed=not problems, problems=problems), problems


# ---- rule two: a gate's commit to HEAD ------------------------------------------------------------------------------------------------------------
def _decision_nodes(tree):
    """{name: [statement nodes binding it]} of the decision constants, anywhere in the module (an Assign or AnnAssign or AugAssign target, a def, a class, an import alias, a for or with target)."""
    found = {}
    for node in ast.walk(tree):
        names = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                names += [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names += [(a.asname or a.name).split('.')[0] for a in node.names]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            names += [n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            names += [n.id for item in node.items if item.optional_vars is not None for n in ast.walk(item.optional_vars) if isinstance(n, ast.Name)]
        for n in names:
            if n in DECISION_CONSTANTS:
                found.setdefault(n, []).append(node)
    return found


def _masked_dump(tree, nodes):
    """ast.dump of the module with the values of `nodes` (the decision assignments) replaced by one placeholder."""
    keep = {id(n): n.value for n in nodes}
    for n in nodes:
        n.value = ast.Constant(value='<decision>')
    try:
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    finally:
        for n in nodes:
            n.value = keep[id(n)]


def decision_assignment_problem(old, new, path=None):
    """None when `new` differs from `old` (one Python file, bytes or str) only by the values of the decision constants' assignments, judged on the AST (decision_log 2026-11-10 ~2:59 AM CST (2)):
      - each decision constant bound in `new` is bound exactly once, by a top-level ast.Assign with one ast.Name target, and the same holds in `old` for each one bound there;
      - the two modules' ASTs are the same but for those assignments' values (ast.dump without positions);
      - each such assignment has its lines to itself: no other statement of the module has a line in [lineno, end_lineno] (a `;` on its line, or after the last line of a multi-line literal, is
        another statement there);
      - its new value is a literal (ast.literal_eval);
      - the bytes outside those line ranges are the same, line for line.
      - `path` given (the gate): each decision constant bound in either version is bound in its own file (DECISION_HOMES; contract 9b C24.3-6 (a)): an assignment of the same name in any
        other file is refused.
    The reason otherwise."""
    old = old.encode('utf-8') if isinstance(old, str) else old
    new = new.encode('utf-8') if isinstance(new, str) else new
    try:
        trees = [ast.parse(old.decode('utf-8')), ast.parse(new.decode('utf-8'))]
    except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
        return f'not parsed as Python: {type(exc).__name__}: {exc}'
    if path is not None:
        strays = sorted({n for tree in trees for n in _decision_nodes(tree) if DECISION_HOMES[n] != path})
        if strays:
            return f'{strays} bound in {path}: each decision constant changes only in its own file ({ {n: DECISION_HOMES[n] for n in strays} })'
        if not any(_decision_nodes(tree) for tree in trees):
            return f'{path} holds no decision constant: a code file other than the four constants\' own files may not change'
    picked = []
    for label, tree in zip(('old', 'new'), trees):
        nodes = []
        for name, binds in _decision_nodes(tree).items():
            node = binds[0]
            if len(binds) != 1 or node not in tree.body or not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                return f'{label}: {name} is bound {len(binds)} times or not by one top-level `{name} = <value>` statement'
            nodes.append(node)
        picked.append(sorted(nodes, key=lambda n: n.lineno))
    if _masked_dump(trees[0], picked[0]) != _masked_dump(trees[1], picked[1]):
        return 'the module differs outside the decision constants\' values (the AST)'
    for node in picked[1]:
        try:
            ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
            return f'line {node.lineno}: {node.targets[0].id}: the value is not a literal ({type(exc).__name__})'
    for label, tree, nodes in (('old', trees[0], picked[0]), ('new', trees[1], picked[1])):
        statements = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)]
        for node in nodes:
            for other in statements:
                if other is not node and other.lineno <= node.end_lineno and (other.end_lineno or other.lineno) >= node.lineno:
                    return f'{label}: line {node.lineno}-{node.end_lineno}: {node.targets[0].id} does not have its lines to itself (another statement there)'

    def outside(data, nodes):
        lines = data.splitlines(keepends=True)
        out, last = [], 0
        for node in nodes:
            out += lines[last:node.lineno-1]+[f'<{node.targets[0].id}>\n'.encode()]
            last = node.end_lineno
        return out+lines[last:]
    if outside(old, picked[0]) != outside(new, picked[1]):
        return 'the bytes outside the decision constants\' lines differ'
    return None


constants_only_problem = decision_assignment_problem         # the name the first version had


def gate_identity_problems(gate_commit, *, repo=None, run=None, head='HEAD'):
    """(record, problems): rule two (module docstring).  From `gate_commit` to `head`:
      1. both commits, the gate's an ancestor of head;
      2. every change of the code (code_kind: the five code directories, the root's importable files, a new top-level directory; git diff --name-status --no-renames -z) is a modified (M) .py
         file whose difference is only the decision constants' values (decision_assignment_problem on `git show <gate>:<path>` and `git show <head>:<path>`), or the allowlist under
         allowlist_inclusion_problems(gate, head); anything else is a problem;
      3. the working tree's code is head's: no tracked change in the code directories or at the root, no untracked or ignored file in the code directories but a compiled file beside its source,
         no untracked or ignored importable file at the root.
    Outside the code nothing is checked (documents, records, reviews, tests).  Reads git only; never raises.  record keys: rule, repo, gate_given, head_given, code_dirs, decision_constants, gate,
    head, gate_is_ancestor, changes (status, path, kind, problem), allowlist_inclusion, worktree_changes, files_in_code_dirs, importable_at_the_top, passed, problems."""
    repo = Path(REPO if repo is None else repo)
    git, raw = git_runner(repo, run)
    record = dict(rule='rule two (contract C24.3-5; prereg revision 2 item 9, B1; decision_log 2026-11-10 ~2:59 AM CST (2), (3)): from the gate\'s commit the code changes only by the four decision '
                       'constants\' values and the allowlist\'s allowance', repo=str(repo), gate_given=gate_commit, head_given=head, code_dirs=list(CODE_DIRS),
                  decision_constants=list(DECISION_CONSTANTS))
    problems = []
    (gate, err_g), (top, err_h) = _commit(git, gate_commit), _commit(git, head)
    record.update(gate=gate, head=top)
    if gate is None or top is None:
        problems.append(f'git rev-parse: gate {gate_commit!r} -> {gate} ({err_g}), head {head!r} -> {top} ({err_h})')
        return dict(record, passed=False, problems=problems), problems
    rc_a, _, err_a = git('merge-base', '--is-ancestor', gate, top)
    record['gate_is_ancestor'] = rc_a == 0
    if rc_a != 0:
        problems.append(f'the gate\'s commit {gate} is not an ancestor of {top} ({err_a or "not an ancestor"})')
    base_top, err_t = _top_entries(git, gate)
    if err_t:
        problems.append(err_t)
    links = set()
    for commit in (gate, top):                          # HB diff review n1: a root symbolic link of either commit is code
        found, err_l = _root_links(git, commit)
        links |= found
        if err_l:
            problems.append(err_l)
    rc_d, out, err_d = git('diff', '--name-status', '--no-renames', '-z', gate, top)
    fields = _paths(out)
    if rc_d or len(fields) % 2:
        problems.append(f'git diff --name-status {gate} {top}: exit {rc_d} {err_d}')
        fields = []
    changes, inclusion = [], None
    for i in range(0, len(fields), 2):
        status, path = fields[i], fields[i+1]
        kind = code_kind(path, base_top, links)
        if kind is None:
            continue                                    # outside the code: not checked
        row = dict(status=status, path=path, kind=kind, problem=None)
        if path == ALLOWLIST_REL:
            inclusion, inclusion_problems = allowlist_inclusion_problems(gate, repo=repo, run=run, head=top)
            row.update(kind='allowlist', problem=None if inclusion['applies'] and not inclusion_problems else f'the allowlist\'s allowance does not hold: {inclusion_problems}')
        elif kind == 'dir' and status == 'M' and path.endswith('.py'):
            rc_o, old, err_o = raw('show', f'{gate}:{path}')
            rc_n, new, err_n = raw('show', f'{top}:{path}')
            if rc_o or rc_n:
                row.update(kind='decision_constants', problem=f'git show: exit {rc_o} {err_o}; exit {rc_n} {err_n}')
            else:
                problem = decision_assignment_problem(old, new, path) if path in DECISION_HOMES.values() else (
                    f'not one of the decision constants\' files {sorted(DECISION_HOMES.values())}: any change refused')
                row.update(kind='decision_constants', problem=None if problem is None else f'{path}: {problem}')
        else:
            row['problem'] = f'{status} {path}: only the decision constants\' values and the allowlist may change in the code ({kind})'
        changes.append(row)
    record['changes'] = changes
    record['allowlist_inclusion'] = inclusion
    problems += [r['problem'] for r in changes if r['problem']]
    rc_w, out_w, err_w = git('status', '--porcelain=v1', '-z', '--untracked-files=no')
    entries = _status_paths(out_w)                      # HB diff review n7: a rename's old path parsed as a path of its own
    wt_links = links | {e for e in entries if '/' not in e and (repo/e).is_symlink()}
    record['worktree_changes'] = [e for e in entries if code_kind(e, base_top, wt_links) is not None]
    if rc_w or record['worktree_changes']:
        problems.append(f'tracked changes of the code in the working tree: {record["worktree_changes"][:10]} (git status: exit {rc_w} {err_w})')
    rc_t, out_t, err_t = git('ls-tree', '-r', '-z', '--name-only', top, '--', *CODE_DIRS)
    sources = set(_paths(out_t))
    rc_u, out_u, err_u = git('ls-files', '-z', '--others', '--', *CODE_DIRS)
    record['files_in_code_dirs'] = [dict(path=p, problem=code_dir_file_problem(p, sources)) for p in _paths(out_u)]
    if rc_t or rc_u:
        problems.append(f'git ls-tree / ls-files: exit {rc_t} {err_t}; exit {rc_u} {err_u}')
    bad = [f['problem'] for f in record['files_in_code_dirs'] if f['problem']]
    if bad:
        problems.append(f'{len(bad)} untracked or ignored files in the code directories: {bad[:10]}')
    rc_o, out_o, err_o = git('ls-files', '-z', '--others', '--directory')
    record['importable_at_the_top'] = _untracked_importables(repo, out_o)
    if rc_o:
        problems.append(f'git ls-files --others at the top: exit {rc_o} {err_o}')
    if record['importable_at_the_top']:
        problems.append(f'untracked or ignored importable files at the code tree\'s top level: {record["importable_at_the_top"][:10]}')
    return dict(record, passed=not problems, problems=problems), problems


__all__ = ['ALLOWLIST_REL', 'CACHE_FILE', 'CODE_DIRS', 'DECISION_CONSTANTS', 'DECISION_HOMES', 'LATER_AUTHORIZATION', 'LATER_DECISION_LOG', 'LATER_RECORDS', 'LPBA40',
           'LPBA40_LATER_AUTHORIZATION', 'LPBA40_LATER_RECORDS', 'LPBA40_PINNED_RECORD_REL', 'LPBA40_PINNED_RECORD_SCHEMA', 'PINNED_CODE_DIRS', 'PINNED_RECORD_KEYS',
           'PINNED_RECORD_REL', 'PINNED_RECORD_SCHEMA', 'PINNED_RECORD_WRITERS', 'PL_CHOICE_PREFIXES', 'Profile', 'REPO', 'ROOT_IMPORTABLE', 'UNSEAL_RECORD_REL',
           'allowlist_inclusion_problems', 'code_dir_file_problem', 'code_kind', 'constants_only_problem', 'decision_assignment_problem', 'gate_identity_problems', 'git_runner',
           'holdout_profile', 'later_commits_problems', 'pinned_record', 'pl_choice_problems', 'strict_pairs', 'unit_identity_problems']
