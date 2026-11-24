"""HP (runs/20261107-hp/任务书.md): what the hold-out data preparation (prep_stage.py) and the hold-out Atropos entry (atropos_holdout.py) share -- the admission through
SG's real functions, the decision of the development checks, the resume, the spy, the test-sandbox write guard, the code check.  Contract C23 (SG, revision 6-6h):
C23.3 (load_unseal), C23.4-1 (bind_unseal, bind_files), C23.4-2 (check_subject), C23.6-5 (the sealed lists are allowed reads of a spy, recorded every time; the
guard's cache is empty when the spy is installed), C23.9-12 (the call sites), C23.12-2 (a) (the pair list pinned by sha256, a row and its direction), C23.12-3 (the
stage holds no label); prereg docs/bayesreg/notes/留出集预登记_20261107.md revision 1 item 1, revision 1a item 18, §5.4 (resume).  The shared hold-out module is XB's
experiments/holdout/holdout_common.py (fcad7a5): the roots, the pair list, the machine gate, the lock, the resume authorization, the test-sandbox guard.

The hold-out admission (admit_holdout), in this order; nothing of a subject is opened before it has passed for every pair of the run:
  1. the pair list: splits/bayesreg_v1/holdout_B_pairs.tsv of bayesreg's code tree (bayesreg.io.sealed.REPO, the tree of the record and of the guard's lists), its
     sha256 the pinned 8088916b..., 50 distinct rows (holdout_common.holdout_pairs);
  2. every subject of every pair: on none of H1, H2, H3, H4 (bayesreg.io.sealed: the guard's own copy of the lists; H2 and H3 also by their spelling) -- refused
     whatever else holds (stage never); a canonical OASIS id (bayesreg.io.guard.subject_dataset); on exactly the list holdout_B_v1;
  3. BZ's unseal record through bayesreg.io.unseal.load_unseal (C23.3-2 (i)-(ix); no record -> refused, stage unseal);
  4. the pinned allowlist configs/bayesreg/dev_subjects.json of that tree, bound to the unseal object by bayesreg.io.unseal.bind_unseal; the object unseals holdout_B_v1;
  5. every subject through bayesreg.io.guard.check_subject on the bound allowlist: admitted, and admitted through the unseal object (a subject on no sealed list -- a
     development, T0 or training subject -- is refused here: it is not a hold-out subject), and one of the record's subjects;
  6. every pair a row of the pair list, fixed and moving in that row's direction (a reversed pair is refused and named as such);
  7. (data preparation only) the OasisFiles of the data root bound by bayesreg.io.unseal.bind_files, through which alone aligned_norm is read (C23.4-1, C23.12-3).
A development run (the two development checks) is admitted by admit_dev: the 19 development pairs (or the two development subjects), each subject on no sealed list,
through check_subject on the unbound pinned allowlist, in the allowlist source stage_L2R_val_v1 (not T0, not the training set).

The decision of a development check (check_decision; review I1, the lead's ruling of 2026-11-09): a hold-out run refuses (stage decision) unless its tool's decision
constant -- filled by a reviewed commit after the check ran and was read (P1), as a plain dict literal (contract C24.3-6 (a)) -- names the check's record by path and sha256,
the pinned plan, a reading the tool accepts, the check's lock line, and (A5a; B4 review I1) the commit the check ran on (code_commit: the record's code and its lock line's),
from which to HEAD the code changed only as rule two allows (bayesreg.io.code_identity.gate_identity_problems: the four decision constants and the allowlist's inclusion).
The code of a hold-out run (A5a; contract C24.3 rule one): besides one clean tree (check_code), the pinned commit's (check_pinned in main: P1 from the committed
runs/holdout_B_v1/pinned_code.json; a dry run records the problems); run_holdout refuses a code record without that check (pinned_code_problem); the lock line's code_commit is P1.

The resume (prereg §5.4; holdout_common.load_resume, the committed authorization splits/bayesreg_v1/holdout_B_v1_<tool>_resume_<k>.json): only for a run whose lock is
there and whose record is not complete; at most once (holdout_common.resume_lock_path).  The failed attempt's output directory is kept whole: it is renamed to
<root>.attempt1 (never deleted or overwritten) before the new attempt writes the root again, and the rename is recorded.

The spy (Spy): a Python audit hook (installed once per process, inert while no spy is active; an audit hook cannot be removed) judges the events 'open' (builtins.open,
io.open, os.open, pathlib, np.load, gzip, nibabel's gzip reader), 'os.listdir' and 'os.scandir', and the members asked of a pair.npz (numpy's NpzFile.__getitem__,
patched while the spy is active).  Each open is judged against the run's policy (Policy) and recorded every time, not deduplicated (C23.6-5 as corrected in revision 6f):
an allowed read of an exact file (the pair list, the sealed lists at the paths bayesreg.io.sealed reads, the unseal record, the allowlist, the decision log, the decision's
records, the images and stage files of the run), a read or write inside the run's own output directory and of its lock file, a read of the code (.py, .pyc, .so, .pth)
and of the Python environment and system; anything else -- a label or evaluation file by its name (aligned_seg*, seg4, seg35, dice, metrics, evaluation, truth; code
files of the code tree and of the environment are exempt from the name rule), a pair.npz member other than fixed, moving, initial_u, a file of the data roots, of
seq_vp_3d or of the results root that is not allowed one by one, a write into a real root while the test sandbox is set, any other path -- is a breach: the open raises
Breach and the run stops.  The spy is installed only when bayesreg.io.sealed.sealed_cache_is_empty() (C23.2-3, C23.6-5): the guard's first read of the sealed lists then
happens under it.
Limits (review m5): (1) a subprocess (git inside load_unseal, systemctl, nvidia-smi, the ANTs binaries) is not seen; (2) a C extension that opens a file without Python's
io is not seen -- nibabel opens .gz through Python's gzip only while indexed_gzip is absent (preload refuses when nibabel._compression.HAVE_INDEXED_GZIP is True);
(3) only the three events above are judged: mkdir, rename/replace, link, unlink, chmod are not (the tools' own ones stay in their own directories); (4) directory
listings are judged only under the data roots, seq_vp_3d and the results root; (5) a module imported lazily after the lock reads code files: allowed, but an unexpected
read stops the run after the lock -- preload imports what the runs call.
The code tree inside the results root (XF, 2026-11-15; runs/20261115-d1/实现说明_XF.md): the hold-out's detached checkout is <results>/holdout_B_v1/code (C22.H0-4) and the P0
checkouts are <results>/p0_code_<sha>.  A read of a code file (CODE_SUFFIXES) whose two forms lie in a code root, and a listing of a directory whose two forms lie in one (the
import machinery's FileFinder lists a package directory), are judged after the data roots, seq_vp_3d and the code trees' data/ (forbidden_roots: they still win) and BEFORE the
results root -- so (5) holds wherever the checkout lies.  Before XF the results root was judged first and every code read under it was a breach (XB's driver stopped so on D1);
with the code tree outside the results root nothing changes.  A write into the code tree is never allowed; a non-code file of the code tree is still allowed only one by one.

The test sandbox (review I2; holdout_common.TEST_SANDBOX_ENV, set by the tests' conftest): guard_write refuses a real root (holdout_common.real_root_problem: the
results root's holdout_B_v1 and xb_*, the D: directory; plus HP's own hp_devcheck_v1) before anything is written, at the start of every run.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys
import threading
from types import MappingProxyType

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import holdout_common as hc  # noqa: E402

CODE_TREE = HERE.parents[1]                     # the code tree this module runs from (= bayesreg.io.sealed.REPO in production; check_code)
UNSEALED_LIST = 'holdout_B_v1'
ALLOWLIST_REL = 'configs/bayesreg/dev_subjects.json'
DEV_SOURCE = 'stage_L2R_val_v1'                 # the allowlist source of the development pairs and subjects
PAIR_MEMBERS = ('fixed', 'moving', 'initial_u')
SEQ_VP_3D = Path('/home/rmcfz/code/python_projects/seq_vp_3d')
DATA_ROOTS = (Path('/mnt/d/data'),)             # the neurite-oasis subject root (bayesreg.seg.unet.data.DEFAULT_DATA_ROOT)
DEVCHECK_DIR = 'hp_devcheck_v1'                 # <results root>/hp_devcheck_v1: the two development checks (a real root for the test sandbox)
FORBIDDEN_NAME_PARTS = ('aligned_seg', 'seg4', 'seg35', 'dice', 'metrics', 'evaluation', 'truth')
SYSTEM_READ = ('/usr', '/lib', '/lib64', '/bin', '/sbin', '/etc', '/proc', '/sys', '/dev', '/run')
CODE_SUFFIXES = ('.py', '.pyc', '.so', '.pth')
DECISION_KEYS = ('devcheck_json', 'devcheck_sha256', 'reading', 'plan_sha256', 'code_commit', 'decided_by', 'review', 'bz_words')     # code_commit: A5a (B4 review I1; contract C24.3-6)
COMMIT = re.compile(r'[0-9a-f]{40}', re.ASCII)
PAIR_SET = 'holdout'                            # holdout_common's authorization pair set
WHOLE_RUN = 'all'                               # the "pair" of a resume authorization: these tools run every pair (every subject) as one cell
ATTEMPT_SUFFIX = '.attempt1'


class Breach(AssertionError):
    """The run opened what its policy does not allow; it stops at that open."""


# ---- the test sandbox (review I2) -----------------------------------------------------------------------------------------------------------------
def real_root_problem(path):
    """holdout_common.real_root_problem, plus HP's development checks' root (<real results root>/hp_devcheck_v1)."""
    problem = hc.real_root_problem(path)
    if problem:
        return problem
    try:
        target = Path(os.path.abspath(path))
    except TypeError:
        return None
    root = Path(hc.REAL_RESULTS)/DEVCHECK_DIR
    if target == root or root in target.parents:
        return f'{path} is under the real {root}'
    return None


def guard_write(*paths):
    """In the test sandbox (holdout_common.TEST_SANDBOX_ENV set), refuse every real root among `paths` (holdout_common.RealRootInTest) before anything is written;
    otherwise nothing.  Every run calls it first, on its output root, its lock and the resume lock."""
    if os.environ.get(hc.TEST_SANDBOX_ENV):
        for path in paths:
            problem = real_root_problem(path)
            if problem:
                raise hc.RealRootInTest(f'test sandbox: {problem}; monkeypatch the roots into tmp_path')
    return paths


def dry_run_fields(code, *, code_check=None, record=None, refusal=None, decision_none=False):
    """A5a review m8: holdout_common.dry_run_summary for HP's dry-run record (prep_stage.py and atropos_holdout.py): the code tree's check (check_code's problems: a tree that is not clean, bayesreg
    not imported from the code root -- kind 'code', never expected; A5a diff review n1), rule one's problems (the code record of check_pinned; none for devcheck), the development check's decision
    (decision_pending while the constant is None: from the record when the dry run got that far, else -- a dry run that stopped on a refusal, as every dry run on P0 does at the unseal -- from
    `decision_none` itself, which the caller passes for `run` only; n1), and the refusal or stop (stage, message) that ended the dry run -- at the unseal, with no record there, the expected
    no_unseal_record."""
    rows = []
    checked = (code_check or {}).get('problems') if isinstance(code_check, dict) else None
    if checked:
        rows.append(('code', f'code: {list(checked)}'))
    pinned = code.get('pinned_code') if isinstance(code, dict) else None
    if isinstance(pinned, dict) and pinned.get('problems'):
        rows.append((hc.rule_one_kind(pinned['problems']), f"rule one: {list(pinned['problems'])}"))
    decision = (record or {}).get('decision') if isinstance(record, dict) else None
    if isinstance(decision, dict) and 'refused' in decision:
        rows.append(('decision_pending' if decision_none else 'decision', f"decision: {decision['refused']}"))
    elif decision is None and decision_none:
        rows.append(('decision_pending', 'decision: the development check\'s decision constant is None (a run refuses at stage decision until a reviewed commit fills it)'))
    if refusal is not None:
        stage, message = refusal
        rows.append(('no_unseal_record' if stage == 'unseal' and not hc.unseal_record_exists() else f'refused_{stage}', f'refused ({stage}): {message}'))
    return hc.dry_run_summary(rows)


def dry_run_out_problem(out):
    """Review m7: a dry run's --out must be a new or empty directory outside the results root, the D: directory, the data roots, the repository's data link and the
    code trees (prereg §5.3: a root a unit started in is not used again).  None, or the reason."""
    target = Path(os.path.abspath(out))
    resolved = target.resolve()
    roots = [hc.RESULTS, hc.REAL_RESULTS, hc.BIG_ROOT, hc.REAL_BIG, *DATA_ROOTS, CODE_TREE, hc.MAIN_REPO, SEQ_VP_3D]
    for root in roots:
        for form in (Path(os.path.abspath(root)), Path(root).resolve()):
            if target == form or form in target.parents or resolved == form or form in resolved.parents:
                return f'--out {out} is inside {root}'
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        return f'--out {out} exists and is not an empty directory'
    return None


# ---- the code -------------------------------------------------------------------------------------------------------------------------------------
def code_identity():
    """bayesreg.seg.unet.snapshot.code_record of the tree bayesreg runs from (git HEAD, clean, where bayesreg and jd_curl were imported from)."""
    from bayesreg.seg.unet.snapshot import code_record
    return code_record()


def check_code(code, *, require_clean=True):
    """A real run: bayesreg and jd_curl imported from inside the code root, a clean git worktree (snapshot.require_clean_code), and one tree for this module, for
    bayesreg (the code record's root) and for bayesreg.io.sealed.REPO (the tree of the record, the pair list, the allowlist and the sealed lists).  Refused (stage code)
    otherwise; nothing of a subject is read.  require_clean False (the dry run): recorded only."""
    import bayesreg.io.sealed as sealed
    from bayesreg.seg.unet.snapshot import CodeNotClean, require_clean_code
    problems = []
    root = Path(code.get('root', '')).resolve() if isinstance(code, dict) and code.get('root') else None
    for name, tree in (('this module', CODE_TREE), ('bayesreg.io.sealed.REPO', Path(sealed.REPO))):
        if root is None or tree.resolve() != root:
            problems.append(f'{name} is in {tree}, the code record is of {root}')
    try:
        require_clean_code(code)
    except CodeNotClean as exc:
        problems.append(str(exc))
    if problems and require_clean:
        raise hc.Refused('code', f'{problems}')
    return dict(root=str(root), checked='one code tree for hp, bayesreg and bayesreg.io.sealed.REPO; clean (snapshot.require_clean_code)', problems=problems)


def pinned_identity():
    """Rule one on this code tree (holdout_common.pinned_code_problems: bayesreg.io.code_identity.unit_identity_problems, P1 from the committed runs/holdout_B_v1/pinned_code.json, never HEAD).
    (record, problems).  The tests replace it (its own tests: tests/test_bayesreg/holdout/test_pinned_code.py)."""
    return hc.pinned_code_problems(repo=CODE_TREE)


def check_pinned(code, *, dry_run=False, identity=None):
    """A5a (contract C24.3 rule one; decision_log 2026-11-09 df8541d, 2026-11-10 ~2:59 AM CST (1), B4 review §1.3): a hold-out run's code is the pinned commit's.  Returns the code record with
    `head` (HEAD), `commit` (P1; HEAD when a dry run found problems) and `pinned_code` (the check's record).  A real run with a problem: Refused (stage code), before anything of the hold-out is read;
    a dry run records the problems and goes on (before BZ unseals there is no pinned record: the expected refusal, C24.3-2)."""
    record, problems = (pinned_identity if identity is None else identity)()
    if problems and not dry_run:
        raise hc.Refused('code', f'rule one (the pinned commit of runs/holdout_B_v1/pinned_code.json): {problems}')
    pinned = record.get('pinned') if isinstance(record, dict) else None
    return dict(code, head=code.get('commit'), commit=pinned if pinned and not problems else code.get('commit'), pinned_code=dict(record, problems=list(problems)) if isinstance(record, dict) else
                dict(record=record, problems=list(problems)))


def pinned_code_problem(code):
    """A hold-out run (run_holdout) takes only a code record whose rule one passed (check_pinned, in main) and whose commit is that P1: None, or why not (fail closed for a direct caller)."""
    pinned = (code or {}).get('pinned_code') if isinstance(code, dict) else None
    if not isinstance(pinned, dict) or pinned.get('passed') is not True or pinned.get('problems') or not pinned.get('pinned') or code.get('commit') != pinned.get('pinned'):
        return f'the code record carries no passing rule-one check naming its commit as P1 (hp_core.check_pinned): {pinned!r}'
    return None


def gate_identity(code_commit, *, repo=None, run=None):
    """Rule two for a development check's decision (contract C24.3-6; decision_log 2026-11-10 ~2:45 AM CST N3, B4 review I1): bayesreg.io.code_identity.gate_identity_problems from the commit the
    check ran on to HEAD of this code tree -- only the four decision constants' assignments (each in its own file, judged on the AST) and the allowlist's inclusion may differ.  (record,
    problems).  The tests replace it where a decision is a fixture (its own tests: tests/test_bayesreg/io_seg/test_code_identity.py)."""
    from bayesreg.io import code_identity
    return code_identity.gate_identity_problems(code_commit, repo=CODE_TREE if repo is None else repo, run=run)


# ---- the decision of a development check (review I1) ----------------------------------------------------------------------------------------------
def lock_first_line(path):
    try:
        lines = [line for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
        return json.loads(lines[0]) if lines else None
    except (OSError, ValueError):
        return None


def check_decision(decision, *, constant, record_path, lock_path, plan_sha256, schema, accepted, accepted_with_bz=(), extra=None, identity=None):
    """The development check was run, read and decided: `decision` (the tool's constant, named `constant`, a plain dict literal in its file -- contract C24.3-6 (a): read here through a
    read-only copy) has exactly DECISION_KEYS; its devcheck_json is `record_path` with that sha256; the record's schema, its plan's sha256 (= the decision's = the tool's pinned plan constant
    `plan_sha256`), its reading (= the decision's), its lock line (= the first line of `lock_path`); the reading one of `accepted`, or one of `accepted_with_bz` with BZ's words (bz_words, a
    non-empty line) -- for a reading the plan leaves to the lead and BZ; `extra(record)` the tool's own checks (a list of problems).
    The code (A5a; B4 review I1; contract C24.3-6; decision_log 2026-11-10 ~2:06 AM CST): code_commit is 40 hex, the commit the check ran on -- the record's code.commit and its lock line's
    code_commit -- with no problem in the record's code check (a clean tree), and from it to HEAD the code changed only as rule two allows (`identity`, default gate_identity: the four decision
    constants' assignments and the allowlist's inclusion).  Refused (stage decision) otherwise; reads two small records of the development check and git, nothing of a subject."""
    if decision is None:
        raise hc.Refused('decision', f'{constant} is None: the development check has not been run, read and decided (a reviewed commit fills it); the hold-out run does not start')
    if not isinstance(decision, (dict, MappingProxyType)) or sorted(decision) != sorted(DECISION_KEYS):
        raise hc.Refused('decision', f'{constant}: exactly the keys {list(DECISION_KEYS)} are required')
    decision = MappingProxyType(dict(decision))         # the constant is a plain literal (C24.3-6 (a)); the reader keeps it read-only (decision_log 2026-11-10 ~3:10 AM CST (1))
    problems = []
    if plan_sha256 is None or decision['plan_sha256'] != plan_sha256:
        problems.append(f"plan_sha256 {decision['plan_sha256']!r} is not the pinned plan {plan_sha256!r}")
    reading, words = decision['reading'], decision['bz_words']
    if reading in accepted:
        pass
    elif reading in accepted_with_bz:
        if not (isinstance(words, str) and words.strip() and '\n' not in words):
            problems.append(f'reading {reading!r} is accepted only with BZ\'s words (bz_words)')
    else:
        problems.append(f'reading {reading!r} does not let the hold-out run start (accepted: {list(accepted)}; with BZ\'s words: {list(accepted_with_bz)})')
    path = Path(str(decision['devcheck_json']))
    if path.resolve() != Path(record_path).resolve():
        raise hc.Refused('decision', f'{constant}: {path} is not the one development check record {record_path}')
    if not path.is_file() or hc.sha256_file(path) != decision['devcheck_sha256']:
        raise hc.Refused('decision', f"{constant}: {path} is missing or not the sha256 {decision['devcheck_sha256']!r}")
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except ValueError:
        raise hc.Refused('decision', f'{constant}: {path} is not JSON') from None
    if record.get('schema') != schema:
        problems.append(f"{path}: schema {record.get('schema')!r}, not {schema!r}")
    if (record.get('plan') or {}).get('sha256') != decision['plan_sha256']:
        problems.append(f"the check ran under the plan {(record.get('plan') or {}).get('sha256')!r}, not {decision['plan_sha256']!r}")
    if record.get('reading') != reading:
        problems.append(f"reading {reading!r} is not the check's {record.get('reading')!r}")
    line = lock_first_line(lock_path)
    if line is None or record.get('lock_line') != line:
        problems.append(f"the check's lock line is not the first line of {lock_path}")
    problems += list(extra(record) if extra else [])
    commit = decision['code_commit']
    code = record.get('code') if isinstance(record.get('code'), dict) else {}
    identity_record = None
    if not (isinstance(commit, str) and COMMIT.fullmatch(commit)):
        problems.append(f'code_commit {commit!r}: 40 lowercase hex, the commit the development check ran on')
    else:
        if code.get('commit') != commit:
            problems.append(f"the check ran on the code {code.get('commit')!r}, not code_commit {commit}")
        if (code.get('check') or {}).get('problems') != []:
            problems.append(f"the check's code check is not clean: {(code.get('check') or {}).get('problems')!r}")
        if (line or {}).get('code_commit') != commit:
            problems.append(f"the check's lock line names the code {(line or {}).get('code_commit')!r}, not code_commit {commit}")
        identity_record, identity_problems = (gate_identity if identity is None else identity)(commit)
        if identity_problems:
            problems.append(f'rule two from code_commit {commit} to HEAD (only the four decision constants and the allowlist may change): {identity_problems}')
    if problems:
        raise hc.Refused('decision', f'{constant}: {problems}')
    return dict({k: decision[k] for k in DECISION_KEYS}, code_identity=identity_record,
                checked='the record by path and sha256, its schema, plan, reading and lock line; the reading accepted; code_commit the commit the check ran on, rule two from it to HEAD')


# ---- the resume (prereg §5.4) ---------------------------------------------------------------------------------------------------------------------
def load_resume(resume_path, *, tool, arm, lock, previous_record, complete):
    """holdout_common.load_resume for the whole run (pair 'all', pair set 'holdout'): the committed authorization, the first run's lock, the previous record (or null),
    the run not complete, the resume lock not there.  None when resume_path is None.  Refused (stage resume) otherwise."""
    if resume_path is None:
        return None
    return hc.load_resume(resume_path, tool=tool, arm=arm, pair=WHOLE_RUN, pair_set=PAIR_SET, lock=lock, previous_record=previous_record, complete=complete)


def attempt_path(root):
    root = Path(root)
    return root.with_name(root.name+ATTEMPT_SUFFIX)


def keep_failed_attempt(root):
    """The failed attempt's output directory renamed to <root>.attempt1, kept whole (never deleted or overwritten).  Returns the record of the rename (files and sizes),
    or None when the attempt left no directory."""
    root, target = Path(root), attempt_path(root)
    if not root.exists():
        return None
    if target.exists():
        raise hc.Refused('resume', f'{target} exists: a failed attempt was kept there already (one resume per run)')
    files = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())
    os.rename(root, target)
    return dict(src=str(root), dst=str(target), files=files, rule='the failed attempt kept whole by a rename (prereg §5.4; hard rule 4)')


# ---- the admission --------------------------------------------------------------------------------------------------------------------------------
def _sealed():
    from bayesreg.io import sealed
    return sealed


def _ids_problem(subject, side):
    """Step 2 of the module docstring for one id: None, or (stage, message)."""
    from bayesreg.io.guard import SubjectSealed, subject_dataset
    sealed = _sealed()
    lists = sealed.lists_of(sealed.guard_sealed_lists(), subject)
    never = sorted(set(lists) & sealed.PAPER1_LISTS)
    if never:
        return 'never', f'{side} subject {subject!r} is on {never}: the paper-1 hold-out sets are never unsealed by the B line (C23.4-7), whatever else holds'
    try:
        dataset = subject_dataset(subject)
    except SubjectSealed as exc:
        return 'admission', f'{side}: {exc}'
    if dataset != 'OASIS':
        return 'admission', f'{side} subject {subject!r}: an OASIS id is required (dataset {dataset})'
    if list(lists) != [UNSEALED_LIST]:
        return 'admission', f'{side} subject {subject} is on the sealed lists {list(lists)}, not exactly [{UNSEALED_LIST!r}]: not a B-line hold-out subject'
    return None


def admit_holdout(pairs, *, unseal_path, data_root=None):
    """Steps 1-7 of the module docstring for `pairs` [(fixed, moving), ...] (None: every row of the pair list, in its order).  Returns a namespace (rows, pairs,
    row_numbers, unseal, allowlist (bound), files (bound; data_root given), record).  holdout_common.Refused otherwise; nothing of a subject is opened (the pair list,
    the sealed lists, the record, the allowlist and the decision log are id lists and records)."""
    from types import SimpleNamespace
    from bayesreg.io.guard import SubjectSealed, check_subject, load_dev_allowlist
    from bayesreg.io.sealed import SealedListError
    from bayesreg.io.unseal import bind_files, bind_unseal, load_unseal
    sealed = _sealed()
    tree = Path(sealed.REPO)
    rows = hc.holdout_pairs(tree)
    pairs = list(rows) if pairs is None else [tuple(p) for p in pairs]
    if not pairs or any(len(p) != 2 for p in pairs):
        raise hc.Refused('admission', f'pairs: (fixed, moving) pairs required, got {pairs[:3]}')
    record = dict(pair_set=UNSEALED_LIST, n_pairs=len(pairs), tree=str(tree), rule='hp_core.admit_holdout: the pair list -> ids, H1-H4 -> load_unseal -> bind_unseal -> '
                                                                                      'check_subject -> the row and its direction -> bind_files')
    try:
        for fixed, moving in pairs:
            for side, subject in (('fixed', fixed), ('moving', moving)):
                problem = _ids_problem(subject, side)
                if problem:
                    raise hc.Refused(*problem)
        record['sealed_lists'] = hc.sealed_record(sealed.guard_sealed_lists())        # SG follow-up review n6: a copy, not the cache's mapping proxies
    except SealedListError as exc:
        raise hc.Refused('sealed', f'the sealed lists cannot be read as pinned: {exc}') from None
    try:
        unseal = load_unseal(unseal_path)
    except SubjectSealed as exc:
        raise hc.Refused('unseal', f'no valid unseal object ({type(exc).__name__}: {exc}); the hold-out is sealed until BZ unseals it') from None
    except (SealedListError, OSError, ValueError) as exc:
        raise hc.Refused('unseal', f'the unseal record could not be checked ({type(exc).__name__}: {exc})') from None
    try:
        allowlist = load_dev_allowlist(tree/ALLOWLIST_REL)
        bound = bind_unseal(allowlist, unseal)
    except SubjectSealed as exc:
        raise hc.Refused('unseal', f'bind_unseal refused ({exc})') from None
    except (OSError, ValueError) as exc:
        raise hc.Refused('unseal', f'the pinned allowlist could not be read ({type(exc).__name__}: {exc})') from None
    if unseal.sealed_list != UNSEALED_LIST:
        raise hc.Refused('unseal', f'the unseal object unseals {unseal.sealed_list!r}, not {UNSEALED_LIST!r}')
    checks = {}
    try:
        for fixed, moving in pairs:
            for side, subject in (('fixed', fixed), ('moving', moving)):
                verdict = check_subject(subject, bound)
                if not verdict.admitted:
                    raise hc.Refused('admission', f"{side}: {verdict.message or f'{subject!r} is not on the development allowlist'} (check_subject step {verdict.step})")
                if not verdict.admitted_by_unseal:
                    raise hc.Refused('admission', f'{side} subject {subject} was not admitted through the unseal object: not a hold-out subject')
                if subject not in unseal.subjects:
                    raise hc.Refused('admission', f'{side} subject {subject} is not one of the {len(unseal.subjects)} subjects of the unseal record')
                checks[subject] = dict(sealed_lists=list(verdict.sealed_lists), admitted_by_unseal=verdict.admitted_by_unseal)
    except SealedListError as exc:
        raise hc.Refused('sealed', f'the guard could not read the sealed lists: {exc}') from None
    index = {row: i for i, row in enumerate(rows)}
    rows_of = []
    for fixed, moving in pairs:
        if (fixed, moving) not in index:
            if (moving, fixed) in index:
                raise hc.Refused('admission', f'{hc.pair_name(fixed, moving)}: the pair list has it the other way round ({moving} fixed, {fixed} moving); the direction is part of '
                                              'the row (C23.12-2 (a))')
            raise hc.Refused('admission', f'{hc.pair_name(fixed, moving)} is not a row of {hc.PAIRS_TSV_REL}')
        rows_of.append(index[(fixed, moving)]+1)
    files = None
    if data_root is not None:
        from bayesreg.seg.unet.data import OasisFiles
        try:
            files = bind_files(OasisFiles(root=Path(data_root)), bound)
        except SubjectSealed as exc:
            raise hc.Refused('unseal', f'bind_files refused ({exc})') from None
    record.update(unseal=unseal.record(), allowlist_sha256=bound.sha256, allowlist_source=unseal.allowlist_source, check_subject=checks,
                  pairs_tsv=dict(path=hc.PAIRS_TSV_REL, sha256=hc.PAIRS_TSV_SHA256, rows=rows_of, direction='fixed, moving as in the row'),
                  files=None if files is None else dict(root=str(files.root), bound='bayesreg.io.unseal.bind_files'))
    return SimpleNamespace(rows=rows, pairs=pairs, row_numbers=rows_of, unseal=unseal, allowlist=bound, files=files, record=record)


def admit_dev(pairs=None, *, subjects=None):
    """A development run: `pairs` [(fixed, moving)] each one of holdout_common.DEV_PAIRS, or `subjects`; every subject on no sealed list, admitted by check_subject on
    the unbound pinned allowlist, in its source stage_L2R_val_v1 and in no never_for_training source.  Returns (allowlist, record); Refused otherwise."""
    from bayesreg.io.guard import check_subject, load_dev_allowlist
    from bayesreg.io.sealed import SealedListError
    sealed = _sealed()
    tree = Path(sealed.REPO)
    names = []
    if pairs is not None:
        for fixed, moving in pairs:
            if hc.pair_name(fixed, moving) not in hc.DEV_PAIRS:
                raise hc.Refused('admission', f'{hc.pair_name(fixed, moving)} is not one of the {len(hc.DEV_PAIRS)} development pairs')
            names += [fixed, moving]
    names += list(subjects or ())
    if not names:
        raise hc.Refused('admission', 'no pair and no subject')
    allowlist = load_dev_allowlist(tree/ALLOWLIST_REL)
    if allowlist.unseal is not None:
        raise hc.Refused('admission', 'a development run takes the unbound allowlist')
    sources = {s['name']: s for s in allowlist.sources}
    if DEV_SOURCE not in sources:
        raise hc.Refused('admission', f'the allowlist has no source {DEV_SOURCE!r}')
    checks = {}
    try:
        for subject in dict.fromkeys(names):
            on = sealed.lists_of(sealed.guard_sealed_lists(), subject)
            if on:
                raise hc.Refused('never' if set(on) & sealed.PAPER1_LISTS else 'admission', f'{subject} is on the sealed lists {list(on)}: not a development subject')
            verdict = check_subject(subject, allowlist)
            if not verdict.admitted:
                raise hc.Refused('admission', f"{verdict.message or f'{subject!r} is not on the development allowlist'} (check_subject step {verdict.step})")
            holding = sorted(s['name'] for s in allowlist.sources if subject in s['subjects'])
            never = sorted(s['name'] for s in allowlist.sources if subject in s['subjects'] and s['never_for_training'])
            if DEV_SOURCE not in holding or never:
                raise hc.Refused('admission', f'{subject} is in the allowlist sources {holding} (never_for_training {never}): only {DEV_SOURCE} subjects are development subjects here')
            checks[subject] = dict(sources=holding)
    except SealedListError as exc:
        raise hc.Refused('sealed', f'the sealed lists cannot be read as pinned: {exc}') from None
    return allowlist, dict(pair_set='dev', tree=str(tree), allowlist_sha256=allowlist.sha256, subjects=checks, sealed_lists=hc.sealed_record(sealed.guard_sealed_lists()),
                           rule='hp_core.admit_dev: the 19 development pairs, on no sealed list, check_subject on the unbound pinned allowlist, source stage_L2R_val_v1')


# ---- the spy -------------------------------------------------------------------------------------------------------------------------------------
_ACTIVE = None
_INSTALLED = False
_LOCAL = threading.local()


def _hook(event, args):
    spy = _ACTIVE
    if spy is None or getattr(_LOCAL, 'busy', False):
        return
    if event not in ('open', 'os.listdir', 'os.scandir'):
        return
    _LOCAL.busy = True
    try:
        if event == 'open':
            spy.see_open(args)
        else:
            spy.see_listing(args[0] if args else None)
    finally:
        _LOCAL.busy = False


def _forms(path):
    """(as written: absolute, '..' folded, links kept; resolved) of a path, or (None, None) for a descriptor or a path that cannot be made."""
    if path is None or isinstance(path, int):
        return None, None
    try:
        raw = Path(os.path.abspath(os.fsdecode(path)))
        return raw, raw.resolve()
    except (TypeError, ValueError, OSError):
        return None, None


def _under(path, roots):
    return any(path == root or path.is_relative_to(root) for root in roots)


def _is_write(mode, flags):
    if isinstance(mode, str) and any(c in mode for c in 'wax+'):
        return True
    return bool(isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))


def sealed_list_paths():
    """The files bayesreg.io.sealed.load_sealed_lists reads (REPO/<path> of each pinned list), at the values of now: the ones whose reads the spy's record counts."""
    sealed = _sealed()
    return [Path(sealed.REPO)/spec['path'] for spec in sealed.SEALED_LISTS]


def allowed_list_paths():
    """C23.6-5: the sealed lists as allowed reads, one by one, no directory -- holdout_common.sealed_list_paths (under bayesreg's code tree sealed.REPO only,
    contract C23 revision 6i, SG follow-up review n9; the tree of holdout_common in production)."""
    return [Path(p) for p in hc.sealed_list_paths()]


class Policy:
    """What one run may open.  read_exact: files, judged in their resolved form; write_inside: the run's own directories (read and write); write_exact: single files
    (the locks); data_roots, results: judged one by one (only the exact files above, and the write directories, are allowed there)."""

    def __init__(self, *, name, read_exact=(), write_inside=(), write_exact=(), results=None, data_roots=None, code_roots=(CODE_TREE,)):
        r = lambda p: Path(p).resolve()     # noqa: E731
        data_roots = DATA_ROOTS if data_roots is None else data_roots
        self.name = name
        self.read_exact = {r(p) for p in read_exact}
        self.write_inside = [r(p) for p in write_inside]
        self.write_exact = {r(p) for p in write_exact}
        roots = [Path(p) for p in data_roots]+[SEQ_VP_3D]
        for code in code_roots:
            roots.append(Path(code)/'data')
        roots.append(hc.MAIN_REPO/'data')
        self.forbidden_roots = sorted({x for p in roots for x in (Path(os.path.abspath(p)), r(p))})
        self.results = None if results is None else sorted({Path(os.path.abspath(results)), r(results)})
        self.code_roots = sorted({r(p) for p in code_roots})
        self.code_root_forms = sorted({x for p in code_roots for x in (Path(os.path.abspath(p)), r(p))})     # XF: a raw path is judged against both forms
        prefixes = {sys.prefix, sys.base_prefix, sys.exec_prefix, *SYSTEM_READ}
        self.system_roots = sorted({r(p) for p in prefixes})

    def allow_read(self, paths):
        self.read_exact |= {Path(p).resolve() for p in paths}

    def _code(self, resolved):
        return resolved.suffix in CODE_SUFFIXES and (_under(resolved, self.code_roots) or _under(resolved, self.system_roots))

    def _in_code_tree(self, raw, resolved):
        """XF: both forms lie in a code root (a link out of the code tree is judged where it points)."""
        return _under(raw, self.code_root_forms) and _under(resolved, self.code_roots)

    def classify(self, path, write):
        """('allowed' | 'forbidden' | 'unlisted', why) of one open."""
        raw, resolved = _forms(path)
        if raw is None:
            return 'allowed', 'not a path (a descriptor)'
        if write and os.environ.get(hc.TEST_SANDBOX_ENV) and (real_root_problem(raw) or real_root_problem(resolved)):
            return 'forbidden', 'a real root written while the test sandbox is set'
        if not (not write and self._code(resolved)):
            for form in (raw, resolved):
                low = form.name.lower()
                if any(part in low for part in FORBIDDEN_NAME_PARTS):
                    return 'forbidden', f'a label or evaluation file by its name ({form.name})'
        if resolved in self.write_exact:
            return 'allowed', 'the run\'s lock'
        if _under(resolved, self.write_inside):
            return 'allowed', 'the run\'s own output directory'
        if not write and resolved in self.read_exact:
            return 'allowed', 'an input of the run, allowed one by one'
        if _under(raw, self.forbidden_roots) or _under(resolved, self.forbidden_roots):
            return 'forbidden', 'a data root or the seq_vp_3d tree, not allowed one by one'
        if not write and resolved.suffix in CODE_SUFFIXES and self._in_code_tree(raw, resolved):
            return 'allowed', 'code (the running code tree, wherever it lies: XF)'
        if self.results and (_under(raw, self.results) or _under(resolved, self.results)):
            return 'forbidden', 'the results root outside the run\'s own files'
        if not write and self._code(resolved) and _under(resolved, self.code_roots):
            return 'allowed', 'code'
        if not write and _under(resolved, self.system_roots):
            return 'allowed', 'the Python environment or the system'
        if write and resolved == Path('/dev/null'):
            return 'allowed', '/dev/null'
        return 'unlisted', 'not in the policy'

    def classify_listing(self, path):
        raw, resolved = _forms(path)
        if raw is None:
            return 'allowed', 'not a path'
        if _under(resolved, self.write_inside):
            return 'allowed', 'the run\'s own output directory'
        if _under(raw, self.forbidden_roots) or _under(resolved, self.forbidden_roots):
            return 'forbidden', 'a listing of a data root, the seq_vp_3d tree or the results root'
        if self._in_code_tree(raw, resolved):
            return 'allowed', 'a directory of the running code tree, wherever it lies (XF)'
        if self.results and (_under(raw, self.results) or _under(resolved, self.results)):
            return 'forbidden', 'a listing of a data root, the seq_vp_3d tree or the results root'
        return 'allowed', 'not watched'

    def document(self):
        return dict(name=self.name, read_exact=sorted(map(str, self.read_exact)), write_inside=sorted(map(str, self.write_inside)), write_exact=sorted(map(str, self.write_exact)),
                    forbidden_roots=list(map(str, self.forbidden_roots)), results=None if self.results is None else list(map(str, self.results)),
                    code_roots=list(map(str, self.code_roots)), system_roots=list(map(str, self.system_roots)), forbidden_name_parts=list(FORBIDDEN_NAME_PARTS),
                    pair_members=list(PAIR_MEMBERS))


class Spy:
    """The audit-hook spy of the module docstring for one run."""

    def __init__(self, policy):
        self.policy = policy
        self.events, self.breaches, self.members = [], [], []
        self._real_getitem = None

    def see_open(self, args):
        path = args[0] if args else None
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        raw, resolved = _forms(path)
        if raw is None:
            return
        write = _is_write(mode, flags)
        cls, why = self.policy.classify(path, write)
        self.events.append(dict(path=str(raw), resolved=str(resolved), write=write, cls=cls))
        if cls != 'allowed':
            self.breaches.append(dict(path=str(raw), resolved=str(resolved), write=write, cls=cls, why=why))
            raise Breach(f'hp spy ({self.policy.name}): {cls} open of {raw} ({why})')

    def see_listing(self, path):
        cls, why = self.policy.classify_listing(path)
        if cls != 'allowed':
            raw, resolved = _forms(path)
            self.breaches.append(dict(path=str(raw), resolved=str(resolved), write=False, cls=cls, why=why, event='listing'))
            raise Breach(f'hp spy ({self.policy.name}): {cls} listing of {raw} ({why})')

    def see_member(self, npz, key):
        source = getattr(getattr(npz, 'fid', None), 'name', None)
        self.members.append(dict(file=str(source), member=str(key)))
        if key not in PAIR_MEMBERS:
            self.breaches.append(dict(path=str(source), member=str(key), cls='forbidden', why='a pair.npz member other than fixed, moving, initial_u'))
            raise Breach(f'hp spy ({self.policy.name}): the member {key!r} of {source} was asked')

    def install(self):
        global _ACTIVE, _INSTALLED
        if _ACTIVE is not None and _ACTIVE is not self:
            raise RuntimeError('another hp spy is active in this process')
        if not _INSTALLED:
            sys.addaudithook(_hook)
            _INSTALLED = True
        from numpy.lib.npyio import NpzFile
        spy = self
        real = NpzFile.__getitem__

        def getitem(npz, key):
            if _ACTIVE is spy:
                spy.see_member(npz, key)
            return real(npz, key)
        self._real_getitem = real
        NpzFile.__getitem__ = getitem
        _ACTIVE = self
        return self

    def remove(self):
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None
        if self._real_getitem is not None:
            from numpy.lib.npyio import NpzFile
            NpzFile.__getitem__ = self._real_getitem
            self._real_getitem = None

    def opened(self, path, *, write=False):
        target = str(Path(path).resolve())
        return sum(1 for e in self.events if e['resolved'] == target and e['write'] == write)

    def record(self, *, expect=()):
        """passed: no breach, every file of `expect` read at least once, and the spy saw opens at all (a spy that saw no open does not pass)."""
        expected = {str(Path(p).resolve()): self.opened(p) for p in expect}
        sealed_reads = {str(p): self.opened(p) for p in sealed_list_paths()}
        positive = bool(self.events) and all(n >= 1 for n in expected.values())
        return dict(spy='hp_core.Spy (audit hook: open, os.listdir, os.scandir; NpzFile.__getitem__)', policy=self.policy.document(), n_events=len(self.events),
                    events=self.events, members=self.members, breaches=list(self.breaches), expected_reads=expected, sealed_list_reads=sealed_reads,
                    positive_condition=positive, passed=positive and not self.breaches,
                    rule='every open judged and recorded (not deduplicated); a breach stops the run; every expected input read at least once; a spy that saw no open does '
                         'not pass')


@contextmanager
def spying(policy):
    """Install a Spy for `policy` only when the guard's cache of the sealed lists is empty (C23.6-5): Refused (stage spy) otherwise.  Yields the spy; removed after.
    An exception leaving the block carries the spy's record as `hp_spy` (what was opened before the refusal or the stop)."""
    from bayesreg.io.sealed import sealed_cache_is_empty
    if not sealed_cache_is_empty():
        raise hc.Refused('spy', 'the guard\'s cache of the sealed lists is populated before the spy was installed: the guard would read the lists unseen (C23.6-5); '
                                'run the entry in a process of its own')
    spy = Spy(policy).install()
    try:
        yield spy
    except BaseException as exc:
        try:
            exc.hp_spy = spy.record()
        except Exception:                       # noqa: BLE001 - the record is a convenience; the exception itself is what counts
            pass
        raise
    finally:
        spy.remove()


def preload():
    """Import, before the spy is installed, the modules a run calls (an import reads files; they are allowed as code, but a run should record its reading, not its
    imports), and refuse (stage spy) when nibabel would open .gz files with indexed_gzip (C code the spy cannot see; review m5).  holdout_common's hi_core (the
    machine gate and the lock) brings hs_seg and torch with it.  Returns the record of the check."""
    import gzip  # noqa: F401
    import zipfile  # noqa: F401
    import encodings.cp437  # noqa: F401   (zipfile decodes member names with it on the first read)
    import nibabel  # noqa: F401
    import nibabel.nifti1  # noqa: F401
    from nibabel import _compression
    import numpy.lib.npyio  # noqa: F401
    import bayesreg.io.guard  # noqa: F401
    import bayesreg.io.pair  # noqa: F401
    import bayesreg.io.sealed  # noqa: F401
    import bayesreg.io.unseal  # noqa: F401
    import bayesreg.seg.unet.data  # noqa: F401
    import bayesreg.seg.unet.snapshot  # noqa: F401
    import jd_curl.three_d.nifti  # noqa: F401
    import jd_curl.three_d.grid  # noqa: F401
    hc._hi()
    if getattr(_compression, 'HAVE_INDEXED_GZIP', False):
        raise hc.Refused('spy', 'nibabel would open .gz files with indexed_gzip (C code): the spy could not see the NIfTI reads (review m5)')
    return dict(nibabel=nibabel.__version__, have_indexed_gzip=False)


def launch_line(*, unit, script, args, runtime_s, memory_max='8G', cpu_quota=None, code=None, log=None):
    """The start string the lead may use (printed by the dry run; nothing here starts a unit).  Units are tag-hb-*: never seq-vp-*.  The unit's output is appended
    to `log` (default <results root>/<unit>_unit.log: outside every output directory, which must be new; the results root exists, systemd does not make directories)."""
    if not unit.startswith('tag-'):
        raise ValueError(f'{unit}: a unit of this project is named tag-*')
    code = str(CODE_TREE if code is None else code)
    log = Path(hc.RESULTS)/f'{unit}_unit.log' if log is None else log
    env = dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, NUMBA_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=code)
    props = [f'--property=MemoryMax={memory_max}', '--property=MemorySwapMax=0', f'--property=RuntimeMaxSec={runtime_s}']
    if cpu_quota:
        props.append(f'--property=CPUQuota={cpu_quota}')
    props += [f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}']
    return ' '.join(['systemd-run', '--user', f'--unit={unit}', '--wait', '--collect', f'--working-directory={code}', '--nice=10', *props,
                     *[f'--setenv={k}={v}' for k, v in env.items()], '/home/rmcfz/miniconda3/envs/sci310/bin/python', '-B', f'{code}/{script}', *args])
