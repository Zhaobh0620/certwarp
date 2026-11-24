"""B-line hold-out: what every hold-out tool shares (runs/20261107-xb/任务书.md item 5; preregistration docs/bayesreg/notes/留出集预登记_20261107.md §5.3, §5.4, §5.6 and its revision 1;
contract C23).  Written for the external baseline runner (experiments/xb/); the data preparation, the Atropos entry, HE's batch queue and the evaluation entry are to reuse it (none of them is written yet).

Why holdout_common.py and not common.py: the experiments/ modules are imported by flat names from their own directories (sys.path), so a module called `common` would shadow, or be shadowed by, any other
`common` on the path (experiments/segprobe/segprobe_common.py is the precedent).

Storage (the lead's choice, BZ may change it; prereg §5.3 and project rules 机器 BZ 11-07):
  records and small files    /home/rmcfz/seq_vp_tag_results/holdout_B_v1/<arm>/<pair>/
  working directories        /home/rmcfz/seq_vp_tag_results/holdout_B_v1/work/<arm>/<pair>/      every timed run writes on ext4 (prereg §5.3 "计时运行在 /home 上跑")
  big files kept on D:       /mnt/d/seq_vp_tag_holdout_B_v1/<arm>/<pair>/                         moved from the working directory by move_files (sha256 before and after)
  stage, Atropos             /home/rmcfz/seq_vp_tag_results/holdout_B_v1/stage/, .../atropos/    (the tools that write them are not written yet)
  locks                      /home/rmcfz/seq_vp_tag_results/holdout_B_v1/locks/
  batch records              /home/rmcfz/seq_vp_tag_results/holdout_B_v1/batches/
The names work, stage, atropos, locks, batches are reserved: no arm may be called so.

Admission (C23.3, C23.4-2, C23.10; the brief item 6).  A hold-out pair is admitted only if, in this order, nothing of a subject being opened before the last step passed:
  1. both ids are canonical OASIS ids (guard.subject_dataset);
  2. neither is on H1, H2, H3 or H4 (bayesreg.eval.truth's pinned sealed lists): refused whatever else holds; and both are on holdout_B_v1 and on no other list;
  3. bayesreg.io.unseal.load_unseal gives the unseal object (the pinned record splits/bayesreg_v1/unseal_holdout_B_v1.json of bayesreg's code tree; it checks C23.3-2 (i)-(ix)); without a record (main
     today) every hold-out pair is refused here;
  4. bind_unseal binds it to the pinned allowlist configs/bayesreg/dev_subjects.json of the same tree;
  5. its list is holdout_B_v1, and bayesreg.io.guard.check_subject admits each subject with the bound allowlist through the unseal object (C23.4-2 in full: canonical id, allowlist, sealed lists, the
     record's subjects and allowlist source);
  6. the pair is a row of splits/bayesreg_v1/holdout_B_pairs.tsv (sha256 pinned, 8088916b…), fixed and moving in that direction (C23.12-2 (a)).
A development pair: one of the 19 development pairs, neither subject on any sealed list, both admitted by guard.check_pair_subjects with the pinned allowlist.

What this module never does: open a subject's file.  The admission reads id lists only (the sealed lists, the pair list, the allowlist, the unseal record through SG).
The sealed lists are the guard's own cached copy (bayesreg.io.sealed.guard_sealed_lists); an entry with a spy checks the cache is empty before installing it (C23.6-5) and lists the four files as allowed
reads (sealed_list_paths).  load_unseal and bind_unseal are called in admit_holdout_pair only (pinned by tests/test_bayesreg/sg/test_sg_census.py CALL_SITES); bind_files is not called: no tool here reads
a subject's image through OasisFiles.

Unit limits against machine failures (prereg revision 1c items 1, 2; the lead's message of 2026-11-07): classify_unit_end below, shared by XB and the later HB runners (external baselines,
ConvexAdam-v4seg, configuration C; not the method arm).  The evidence of one unit (or one killed step inside it): systemd's Result (systemd-run --wait's "Finished with result", kept as a file), the
in-unit end probe's memory.events (experiments/he/unit_probe.py, read before the unit ends), the kernel's oom-kill lines in the run's window (dmesg), the unit's journal, a step's exit -9.
  OOM evidence of this unit: a dmesg oom-kill line in the window whose killed task belongs to this unit (task_memcg is the unit's cgroup or below it; any constraint: MEMCG of the unit or of a parent,
  CPUSET, NONE), memory.events oom or oom_kill > 0, "killed by the OOM killer" in the journal, systemd's Result oom-kill (decision_log 2026-11-09), a step's exit -9 (not one the supervisor itself
  sent after a SIGTERM).  A step is read on every exit, not only on -9 (I2: antsRegistrationSyN.sh exits 0 after its antsRegistration was killed).
  Rule (乙), the default (LIMIT_RULE = 'yi'): OOM evidence -> infra_failure (the batch stops; one resume allowed).  Then: no OOM evidence and Result timeout -> unit_limit (RuntimeMaxSec: a method
  failure of class (3), not rerun).  A source that could not be read (dmesg, the end probe, the Result) and no OOM evidence -> needs_lead_rejudgment (stop): never read as 0.  OOM is looked at first.
  Rule (甲) (LIMIT_RULE = 'jia', the alternative): with OOM evidence, a CONSTRAINT_NONE line (the machine) -> infra_failure; the unit's own limit -- a CONSTRAINT_MEMCG line whose oom_memcg is this
  unit's cgroup, or memory.events oom > 0 -> unit_limit (MemoryMax); any other OOM evidence (a parent's memcg, a cpuset, a -9 or a journal line alone) -> infra_failure.
  Both rules: a CONSTRAINT_NONE line of any task in the window is a machine event -> infra_failure (BL2 stopped on every kernel OOM line; adopted by the lead, decision_log 2026-11-09); the driver's
  memory guard -> infra_failure.
  The window's clock (decision_log 2026-11-10 ~3:16 AM CST; runs/20261107-xb/limit_experiment_result.md revision 1 item 2, the result review's I2): the window starts at /proc/uptime read just
  before the unit (parse_oom's `since`), but on this machine dmesg's time stamps run about 3.6-4.1 s ahead of /proc/uptime, so the window in fact reaches back about 4 s before the start: a line the
  kernel wrote up to ~4 s before the unit started (the previous unit's) can fall into it.  A task_memcg of another unit is filtered out of this unit's OOM evidence; a CONSTRAINT_NONE line is not
  (it reads infra_failure: the conservative side -- the batch stops, nothing is read as a method failure).  The window is not changed (the lead's ruling); instead machine_gate refuses while the
  previous tag- unit of this project ended less than UNIT_GAP_SECONDS (10 s) ago, for every queue that uses it: by the user journal (which must be readable: journal_alive) and by the
  queue's own record of the last unit it ran (note_unit_end), the later one, on the monotonic clock (A5a review I1, m1).
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import MappingProxyType

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))


def _hi():
    """experiments/hi/hi_core.py, imported when first needed (the machine gate, the lock, git): the limit discriminator below is imported by the in-unit supervisor, which needs none of it."""
    if str(EXPERIMENTS/'hi') not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS/'hi'))
    import hi_core
    return hi_core


def __getattr__(name):
    if name == 'hi':
        return _hi()
    if name in CODE_IDENTITY_NAMES:
        return getattr(_ci(), name)
    raise AttributeError(name)

MAIN_REPO = Path('/home/rmcfz/code/python_projects/seq_vp_tag')     # data/ is an untracked link of the main checkout
RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
HOLDOUT_ROOT = RESULTS/'holdout_B_v1'
BIG_ROOT = Path('/mnt/d/seq_vp_tag_holdout_B_v1')
RESERVED = ('work', 'stage', 'atropos', 'locks', 'batches')
STAGE_ROOT = HOLDOUT_ROOT/'stage'               # <pair>/pair.npz and manifest.json (fixed, moving, an all-zero initial_u; no label member: prereg revision 1 item 1)
ATROPOS_ROOT = HOLDOUT_ROOT/'atropos'
LOCK_ROOT = HOLDOUT_ROOT/'locks'
BATCH_ROOT = HOLDOUT_ROOT/'batches'
PAIR_SET = 'holdout_B_v1'

PAIRS_TSV_REL = 'splits/bayesreg_v1/holdout_B_pairs.tsv'
PAIRS_TSV_SHA256 = '8088916b103a4ddd8b5589164a62ea6048bac1c1799fd367f1deec154151a9ee'    # C23.12-2 (a); splits/bayesreg_v1/SHA256SUMS line 5
N_PAIRS = 50
UNSEAL_REL = 'splits/bayesreg_v1/unseal_holdout_B_v1.json'          # C23.3-1: the one path of the record
ALLOWLIST_REL = 'configs/bayesreg/dev_subjects.json'
UNSEALED_LIST = 'holdout_B_v1'
NEVER = ('H1', 'H2', 'H3', 'H4')                # C23.3-2 (v), C23.4-7: never unsealed by the B line

DEV_STAGE = MAIN_REPO/'data'/'three_d'/'stage_L2R_val_v1'
DEV_PAIRS = tuple(f'OASIS_OAS1_{n:04d}_MR1__OASIS_OAS1_{n+1:04d}_MR1' for n in range(438, 457))    # the 19 development pairs (experiments/pq/pq_pins.py SHORTS)

D_FLOOR_GIB = 40.                               # prereg §5.3: D: keeps 40 GiB free after the expected output
C_FLOOR_GIB = 20.                               # prereg §5.3: C: the top of the 15-20 GiB band
D_MOUNT, C_MOUNT, HOME_MOUNT = Path('/mnt/d'), Path('/mnt/c'), Path('/home')
MEM_AVAILABLE_MIN_GIB = 6.                      # prereg §5.3; drive.py MEM_MIN_GB (MemAvailable kB / 1024**2)
LOAD_LIMIT = 1.5                                # HE's gate (experiments/mc0/run_mc0.py LOAD_LIMIT, smoke_he GATE)
UNIT_GAP_SECONDS = 10.                          # decision_log 2026-11-10 ~3:16 AM CST (I2): the previous tag- unit ended at least this long ago (dmesg runs ~4 s ahead of /proc/uptime here)
JOURNAL_SPAN_SECONDS = 600                      # the user journal's window read by the gate (wall clock, only to select entries; the age is on the monotonic clock: A5a review m1)
PROJECT_UNIT_PREFIX = 'tag-'                    # this project's units (project rules: never seq-vp-*)
GATE_WAIT_SECONDS = 15*60                       # prereg §5.4: a precheck that does not pass in 15 minutes is an infrastructure failure
RESUME_WRITERS = ('BZ', "lead on BZ's behalf")  # prereg §5.4
# the committed authorizations (prereg §5.4, C23.12-6, D107; the lead's rulings of 2026-11-09 on the XB review I1-I3): a resume record, a rerun authorization after a refusal before the lock,
# a "no resume" (未批补跑) record.  The hold-out's in splits/bayesreg_v1/ (as HI's), the development runs' in runs/20261107-xb/authorizations/ (not a split file).  AUTH_ROOT is the code tree (tests
# point it to a temporary git repository: the records must be committed there).
AUTH_ROOT = REPO
AUTH_DIRS = dict(holdout='splits/bayesreg_v1', dev='runs/20261107-xb/authorizations', lpba40_test='splits/lpba40_test_v1')
AUTH_PREFIX = dict(holdout='holdout_B_v1_', dev='dev_', lpba40_test='lpba40_test_v1_')
AUTH_KINDS = ('resume', 'rerun', 'no_resume')
# The LPBA40 second test set (contract C26.8-3, D148; LP-C2): its authorizations in splits/lpba40_test_v1/ named lpba40_test_v1_<tool>_<kind>_<k>.json, pair set 'lpba40_test' (the
# formal mode only: the feasibility mode has none, C26.6-2).  A resume (补跑) written by "lead on BZ's behalf" carries BZ's words of decision_log line 520 word for word
# (LPBA40_BZ_WORDS, as bayesreg.io.unseal.DELEGATION_WORDS); the lead's ruling on the LP-C2 review, minor 8: only the resume (and the evaluation's reevaluation, bayesreg.eval.lpba_eval)
# -- a no_resume or a rerun is BZ's decision afterwards: its words are BZ's own words of that time (non-empty, as every pair set) and its reason names where they are.
LPBA40_PAIR_SET = 'lpba40_test'
LPBA40_BZ_WORDS = '预登记写好、复核过后批准你直接跑, 不用等我'      # decision_log line 520 (main 36a40941), BZ's words that cover the LPBA40 runs (C26.8-3)
# I7 (the lead, 2026-11-09): while the tests run (the conftest sets XB_TEST_SANDBOX), every root function and every writer of this module refuses the real roots, whatever the module constants say
TEST_SANDBOX_ENV = 'XB_TEST_SANDBOX'
REAL_RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
REAL_BIG = Path('/mnt/d/seq_vp_tag_holdout_B_v1')
SAFE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,200}', re.ASCII)
UTC_FORMAT = '%Y-%m-%dT%H:%M:%SZ'
GIB = 2**30


class Refused(Exception):
    """Refused before anything of a subject was opened; `stage` says which check (admission, never, unseal, once, machine, disk, frozen, environment, resume, arm_stopped, ...)."""
    status = 'holdout_refused'

    def __init__(self, stage, message):
        super().__init__(f'{stage}: {message}')
        self.stage = stage


class InfrastructureFailure(Exception):
    """Prereg §5.4: the batch stops, the evidence stays, BZ decides.  `kind` names it (disk_guard, move_mismatch, ...)."""

    def __init__(self, kind, message):
        super().__init__(f'{kind}: {message}')
        self.kind = kind


def utc():
    return datetime.now(timezone.utc).strftime(UTC_FORMAT)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, (Path, MappingProxyType)):
        return str(value) if isinstance(value, Path) else dict(value)
    return str(value)


def write_json_once(path, document):
    """A record written once: a temporary file in the same directory, then a hard link (FileExistsError if the record is there), then the temporary removed."""
    path = Path(guard_root(path))
    if path.exists():
        raise FileExistsError(f'{path} exists: records are never overwritten')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(document, indent=1, ensure_ascii=False, default=json_default)+'\n', encoding='utf-8')
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()
    return path


class RealRootInTest(RuntimeError):
    """A test reached one of the real roots (I7): it must monkeypatch the roots into its temporary directory."""


def real_root_problem(path):
    """Why `path` is a real root of the hold-out tools (the results root's holdout_B_v1 or xb_*, or the D: directory), or None."""
    try:
        target = Path(os.path.abspath(path))
    except TypeError:
        return None
    for root in (REAL_RESULTS, REAL_BIG):
        if target == root or root in target.parents:
            if root == REAL_BIG:
                return f'{path} is under {REAL_BIG}'
            rel = target.relative_to(root).parts
            if rel and (rel[0] == 'holdout_B_v1' or rel[0].startswith('xb_')):
                return f'{path} is under the real {root/rel[0]}'
    return None


def guard_root(path):
    """In the test sandbox (XB_TEST_SANDBOX set), refuse a real root (RealRootInTest); otherwise return the path unchanged."""
    if os.environ.get(TEST_SANDBOX_ENV):
        problem = real_root_problem(path)
        if problem:
            raise RealRootInTest(f'test sandbox: {problem}; monkeypatch the roots into tmp_path')
    return path


def safe_name(value, what):
    if not isinstance(value, str) or not SAFE.fullmatch(value) or value in RESERVED:
        raise Refused('admission', f'{what} {value!r} is not a plain name (or is reserved: {RESERVED})')
    return value


# ---- paths --------------------------------------------------------------------------------------------------------------------------------------
def pair_name(fixed, moving):
    return f'{fixed}__{moving}'


def split_pair(name):
    parts = str(name).split('__')
    if len(parts) != 2 or not all(parts):
        raise Refused('admission', f'{name!r} is not <fixed>__<moving>')
    return parts[0], parts[1]


def short_pair(name):
    """OASIS_OAS1_0423_MR1__OASIS_OAS1_0099_MR1 -> 0423-0099 (unit names)."""
    m = re.fullmatch(r'OASIS_OAS1_([0-9]{4})_MR1__OASIS_OAS1_([0-9]{4})_MR1', str(name))
    if not m:
        raise Refused('admission', f'{name!r} is not a pair of OASIS ids')
    return f'{m.group(1)}-{m.group(2)}'


def arm_dir(arm, root=None):
    return guard_root(Path(HOLDOUT_ROOT if root is None else root)/safe_name(arm, 'arm'))


def record_dir(arm, pair, root=None):
    return guard_root(arm_dir(arm, root)/safe_name(pair, 'pair'))


def work_dir(arm, pair, root=None):
    return guard_root(Path(HOLDOUT_ROOT if root is None else root)/'work'/safe_name(arm, 'arm')/safe_name(pair, 'pair'))


def big_dir(arm, pair, big=None):
    return guard_root(Path(BIG_ROOT if big is None else big)/safe_name(arm, 'arm')/safe_name(pair, 'pair'))


def stage_pair_path(pair, stage=None):
    return guard_root(Path(STAGE_ROOT if stage is None else stage)/safe_name(pair, 'pair')/'pair.npz')


def batch_root(root=None):
    return guard_root(Path(HOLDOUT_ROOT if root is None else root)/'batches')


# ---- the id lists -------------------------------------------------------------------------------------------------------------------------------
def holdout_pairs(repo=None):
    """[(fixed, moving), ...] of splits/bayesreg_v1/holdout_B_pairs.tsv in its row order, after its sha256 is checked against the pinned value (ids only)."""
    path = Path(REPO if repo is None else repo)/PAIRS_TSV_REL
    if not path.is_file():
        raise Refused('pairs', f'{PAIRS_TSV_REL} is missing')
    digest = sha256_file(path)
    if digest != PAIRS_TSV_SHA256:
        raise Refused('pairs', f'{PAIRS_TSV_REL}: sha256 {digest} is not the pinned {PAIRS_TSV_SHA256}')
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not lines or lines[0].split('\t') != ['fixed', 'moving']:
        raise Refused('pairs', f'{PAIRS_TSV_REL}: the header is not fixed<TAB>moving')
    rows = [tuple(line.split('\t')) for line in lines[1:]]
    if len(rows) != N_PAIRS or any(len(r) != 2 for r in rows) or len(set(rows)) != N_PAIRS:
        raise Refused('pairs', f'{PAIRS_TSV_REL}: {len(rows)} rows, {N_PAIRS} distinct two-column rows expected')
    return rows


def sealed_lists():
    """The guard's own copy of the sealed lists (bayesreg.io.sealed.guard_sealed_lists: read from bayesreg's code tree with the pinned sha256, cached per process; C23.2-3).  An entry with a spy
    checks the cache is empty before it installs the spy (C23.6-5), so this first read happens under the spy.  Fail closed: a list missing or changed refuses."""
    from bayesreg.io.sealed import SealedListError, guard_sealed_lists
    try:
        return guard_sealed_lists()
    except SealedListError as exc:
        raise Refused('sealed', f'the sealed lists cannot be read as pinned: {exc}') from None


def sealed_record(sealed):
    """SG follow-up review n6 (contract C23 revision 6j, last item; C24.7): the record of the guard's cached sealed lists as a plain copy, a list of dicts (the cache holds a tuple of mapping
    proxies, SG code review m4): what a record of this module holds, as bayesreg.io.guard._sealed_guard_record does (guard.py:334).  Its JSON is the same bytes."""
    return [dict(r) for r in sealed.record]


def lists_of(subject, sealed):
    """The sealed lists holding `subject` (bayesreg.io.sealed.lists_of: H2 and H3 also by their code rules)."""
    from bayesreg.io.sealed import lists_of as sealed_lists_of
    return list(sealed_lists_of(sealed, subject))


def sealed_list_paths():
    """The files the guard reads for the sealed lists (C23.2; C23.6-5: allowed reads of every spy, listed one by one): SEALED_LISTS under bayesreg's code tree
    sealed.REPO only -- the one root the sealed module reads (contract C23 revision 6i; SG follow-up review n9, as m6 did for pq_firewall).  In production it is
    this tree too."""
    from bayesreg.io import sealed as sealed_module
    return sorted({str((Path(sealed_module.REPO)/spec['path']).resolve()) for spec in sealed_module.SEALED_LISTS})


def canonical(subject, side):
    from bayesreg.io.guard import SubjectSealed, subject_dataset
    try:
        dataset = subject_dataset(subject)
    except SubjectSealed as exc:
        raise Refused('admission', f'{side}: {exc}') from None
    if dataset != 'OASIS':
        raise Refused('admission', f'{side} {subject!r}: an OASIS id is required (dataset {dataset})')
    return subject


# ---- SG (contract C23): bayesreg.io.unseal, bayesreg.io.guard.check_subject -------------------------------------------------------------------
def admit_holdout_pair(fixed, moving):
    """The module docstring's steps 1-6 for one hold-out pair.  Returns (record, bound allowlist).  Refused otherwise; nothing of either subject is opened here.
    The record and the allowlist are those of bayesreg's code tree (bayesreg.io.sealed.REPO, read at call time; the tree of this module in production): load_unseal takes only its pinned path."""
    from bayesreg.io import sealed as sealed_module
    from bayesreg.io import unseal as unseal_module
    from bayesreg.io.guard import SubjectSealed, check_subject, load_dev_allowlist
    record = dict(pair_set=PAIR_SET, fixed_subject=fixed, moving_subject=moving, rule='holdout_common.admit_holdout_pair: ids -> H1-H4 -> load_unseal, bind_unseal, check_subject (C23.4-2) -> the pinned pair list')
    for side, subject in (('fixed', fixed), ('moving', moving)):
        canonical(subject, side)
    sealed = sealed_lists()
    for side, subject in (('fixed', fixed), ('moving', moving)):
        on = lists_of(subject, sealed)
        never = [name for name in on if name in NEVER]
        if never:
            raise Refused('never', f'{side} subject {subject} is on {never}: the paper-1 hold-out sets are never unsealed by the B line (C23.4-7), whatever else holds')
        if on != [UNSEALED_LIST]:
            raise Refused('admission', f'{side} subject {subject} is on the sealed lists {on}, not exactly [{UNSEALED_LIST!r}]: not a B-line hold-out subject')
    record['sealed_lists'] = sealed_record(sealed)
    tree = Path(sealed_module.REPO)
    try:
        unseal = unseal_module.load_unseal(tree/unseal_module.UNSEAL_REL)
    except SubjectSealed as exc:                    # UnsealRefused: no record, or a record that does not hold (C23.3-2)
        raise Refused('unseal', f'no valid unseal object ({exc}); the hold-out is sealed until BZ unseals it') from None
    allowlist = load_dev_allowlist(tree/unseal_module.ALLOWLIST_REL)
    try:
        bound = unseal_module.bind_unseal(allowlist, unseal)
    except SubjectSealed as exc:
        raise Refused('unseal', f'bind_unseal refused ({exc})') from None
    if unseal.sealed_list != UNSEALED_LIST:
        raise Refused('unseal', f'the unseal object unseals {unseal.sealed_list!r}, not {UNSEALED_LIST!r}')
    checks = {}
    for side, subject in (('fixed', fixed), ('moving', moving)):
        try:
            check = check_subject(subject, bound)
        except Exception as exc:                    # noqa: BLE001 - SealedListError and the like: fail closed
            raise Refused('admission', f'{side}: the sealed guard could not decide ({type(exc).__name__}: {exc})') from None
        if not check.admitted:
            raise Refused('admission', f'{side} subject {subject}: refused by bayesreg.io.guard.check_subject at step {check.step} ({check.item}): {check.message or "not on the allowlist"}')
        if not check.admitted_by_unseal:
            raise Refused('admission', f'{side} subject {subject}: admitted without the unseal object (on no sealed list?): not a hold-out subject')
        checks[side] = dict(step=check.step, sealed_lists=list(check.sealed_lists), admitted_by_unseal=check.admitted_by_unseal)
    rows = holdout_pairs()
    if (fixed, moving) not in rows:
        if (moving, fixed) in rows:
            raise Refused('admission', f'{pair_name(fixed, moving)}: the pair list has it the other way round ({moving} fixed, {fixed} moving); the direction is part of the row (C23.12-2 (a))')
        raise Refused('admission', f'{pair_name(fixed, moving)} is not a row of {PAIRS_TSV_REL}')
    record.update(unseal=unseal.record(), allowlist_sha256=bound.sha256, sg='bayesreg.io.unseal (load_unseal, bind_unseal), bayesreg.io.guard.check_subject', check_subject=checks,
                  pairs_tsv=dict(path=PAIRS_TSV_REL, sha256=PAIRS_TSV_SHA256, row=rows.index((fixed, moving))+1, direction='fixed, moving as in the row'))
    return record, bound


def admit_dev_pair(fixed, moving, *, dev_pairs=DEV_PAIRS):
    """A development pair: one of `dev_pairs`, on no sealed list, both on the pinned allowlist (guard.check_pair_subjects).  Returns (record, allowlist); Refused otherwise."""
    from bayesreg.io.guard import SubjectSealed, check_pair_subjects, load_dev_allowlist
    name = pair_name(fixed, moving)
    if name not in dev_pairs:
        raise Refused('admission', f'{name} is not one of the {len(dev_pairs)} development pairs')
    for side, subject in (('fixed', fixed), ('moving', moving)):
        canonical(subject, side)
    sealed = sealed_lists()
    for side, subject in (('fixed', fixed), ('moving', moving)):
        on = lists_of(subject, sealed)
        if on:
            raise Refused('never' if any(n in NEVER for n in on) else 'admission', f'{side} subject {subject} is on the sealed lists {on}: not a development subject')
    allowlist = load_dev_allowlist(REPO/ALLOWLIST_REL)
    try:
        guard = check_pair_subjects(fixed, moving, allowlist)
    except SubjectSealed as exc:
        raise Refused('admission', str(exc)) from None
    return dict(pair_set='dev', fixed_subject=fixed, moving_subject=moving, sealed_lists=sealed_record(sealed), guard=_plain(guard), rule='holdout_common.admit_dev_pair'), allowlist


def _plain(value):
    try:
        return json.loads(json.dumps(value, default=json_default))
    except (TypeError, ValueError):
        return str(value)


# ---- disks --------------------------------------------------------------------------------------------------------------------------------------
def disk_reading(usage=None):
    """GiB free on /mnt/d, /mnt/c and /home (None when a mount cannot be read)."""
    usage = shutil.disk_usage if usage is None else usage
    out = {}
    for name, mount in (('d', D_MOUNT), ('c', C_MOUNT), ('home', HOME_MOUNT)):
        try:
            out[name] = usage(str(mount)).free/GIB
        except OSError:
            out[name] = None
    return out


def disk_guard(*, need_d_gib, need_home_gib, usage=None):
    """Prereg §5.3, both sides, before a batch and before each unit:
      D: side   need_d + 40 GiB <= D: free;
      C: side   need_home + 20 GiB <= C: free, and the same against /home (ext4 inside the VHDX, which lives on C:).
    The C: side counts no VHDX reuse (the prereg's formula allows adding it): the reusable part of ext4.vhdx is not visible from inside WSL as a number this process can read, and a guess on the
    permissive side could fill C:, the machine's real limit (project rules 机器).  So this guard is stricter than the formula.  Returns the record; InfrastructureFailure(disk_guard) otherwise
    (a reading that cannot be made fails it)."""
    reading = disk_reading(usage)
    record = dict(rule='D: need + 40 GiB <= free(/mnt/d); C: need + 20 GiB <= free(/mnt/c) and free(/home); no VHDX reuse counted', need_d_gib=need_d_gib, need_home_gib=need_home_gib,
                  free_gib=reading, d_floor_gib=D_FLOOR_GIB, c_floor_gib=C_FLOOR_GIB, utc=utc())
    problems = []
    if need_d_gib and (reading['d'] is None or need_d_gib+D_FLOOR_GIB > reading['d']):
        problems.append(f"D: free {reading['d']} GiB < need {need_d_gib} + {D_FLOOR_GIB}")
    for side in ('c', 'home'):
        if reading[side] is None or need_home_gib+C_FLOOR_GIB > reading[side]:
            problems.append(f"{side} free {reading[side]} GiB < need {need_home_gib} + {C_FLOOR_GIB}")
    record['passed'] = not problems
    if problems:
        record['problems'] = problems
        raise InfrastructureFailure('disk_guard', f'{problems} (record {json.dumps(record, default=json_default)})')
    return record


# ---- the machine --------------------------------------------------------------------------------------------------------------------------------
def mem_available_gib(meminfo='/proc/meminfo'):
    try:
        for line in Path(meminfo).read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])/1024**2
    except (OSError, ValueError, IndexError):
        pass
    return 0.


def load1(loadavg='/proc/loadavg'):
    try:
        return float(Path(loadavg).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return float('inf')


JOURNAL_FIELDS = ('__REALTIME_TIMESTAMP', '__MONOTONIC_TIMESTAMP', '_BOOT_ID', 'USER_UNIT', '_SYSTEMD_USER_UNIT')     # A5a review m6: nothing else of an entry (no MESSAGE) is read
JOURNAL_SOURCE = ('--user',)                   # the user journal (the tests point journalctl at a directory: ('--directory', D))
_local_end = None                               # A5a review I1: this process's last unit end (note_unit_end), on the monotonic clock


def boot_id(path='/proc/sys/kernel/random/boot_id'):
    """This boot's id as the journal writes it (_BOOT_ID: 32 hex, no dashes), or None."""
    try:
        text = Path(path).read_text(encoding='utf-8').strip().replace('-', '')
    except OSError:
        return None
    return text or None


def note_unit_end(unit, *, monotonic=None):
    """A queue records the end of a unit it ran (systemd-run --wait returned), on the monotonic clock (A5a review I1: the gate's local lower bound, which holds when journald is slow to take the
    manager's last lines in).  Called by xb_chain.run_one and hb_core.run_unit right after the unit."""
    global _local_end
    _local_end = dict(unit=str(unit), monotonic=time.monotonic() if monotonic is None else float(monotonic), boot_id=boot_id())
    return dict(_local_end)


def local_unit_end():
    """The last unit end this process recorded (note_unit_end), or None."""
    return None if _local_end is None else dict(_local_end)


def _journal(run, *args):
    argv = ['journalctl', *JOURNAL_SOURCE, '-o', 'json', '--no-pager', f'--output-fields={",".join(JOURNAL_FIELDS)}', *args]
    try:
        done = run(argv, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise Refused('machine', f'the user journal could not be read ({type(exc).__name__}: {exc}): the gap since the previous unit is not known') from None
    if getattr(done, 'returncode', 1) != 0:
        raise Refused('machine', f'journalctl exited {getattr(done, "returncode", None)} ({(getattr(done, "stderr", "") or "").strip()[:300]}): the gap since the previous unit is not known')
    entries = []
    for line in (done.stdout or '').splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            entry = None
        if not isinstance(entry, dict):
            raise Refused('machine', f'journalctl -o json gave a line that is not a JSON object ({line[:120]!r}): the gap since the previous unit is not known')
        entries.append(entry)
    return argv, entries


def _monotonic_of(entry):
    """(seconds on the monotonic clock, boot id) of a journal entry; Refused when either is missing (__MONOTONIC_TIMESTAMP is CLOCK_MONOTONIC in microseconds; A5a review m1)."""
    try:
        mono = int(entry['__MONOTONIC_TIMESTAMP'])/1e6
    except (KeyError, TypeError, ValueError):
        raise Refused('machine', f"a journal entry has no __MONOTONIC_TIMESTAMP ({entry.get('__MONOTONIC_TIMESTAMP')!r}): the gap since the previous unit is not known") from None
    boot = entry.get('_BOOT_ID')
    if not isinstance(boot, str) or not boot:
        raise Refused('machine', 'a journal entry has no _BOOT_ID: the gap since the previous unit is not known')
    return mono, boot


def journal_alive(*, run=None):
    """A5a review I1: the user journal is readable -- `journalctl --user -o json -n 1` gives exactly one entry with its monotonic stamp and boot id.  journalctl exits 0 with no output when it finds no
    journal file at all (an empty directory, a user journal never kept): that is refused here, never read as "no unit".  Returns the entry's stamps; Refused (stage machine) otherwise."""
    run = subprocess.run if run is None else run
    argv, entries = _journal(run, '-n', '1')
    if len(entries) != 1:
        raise Refused('machine', f'{" ".join(argv)} gave {len(entries)} entries, not one: the user journal cannot be read, so the gap since the previous unit is not known')
    mono, boot = _monotonic_of(entries[0])
    return dict(query=argv, newest_monotonic=mono, newest_boot_id=boot)


def last_unit_end(*, run=None, exclude=(), now=None, boot=None, span=JOURNAL_SPAN_SECONDS):
    """The newest entry of the user journal in the last `span` seconds that belongs to a unit of this project (PROJECT_UNIT_PREFIX) other than `exclude` (names with or without .service): the
    manager's lines about the unit (USER_UNIT: started, its result, deactivated, its CPU time -- the last of them is written when the unit ends) and the unit's own output (_SYSTEMD_USER_UNIT).
    A unit that is no longer active ended no later than its newest entry, so "newest entry >= UNIT_GAP_SECONDS ago" implies "ended >= UNIT_GAP_SECONDS ago" (the gate checks first that no other
    unit is active).  The age is on the monotonic clock (`now`: time.monotonic(); the entry's __MONOTONIC_TIMESTAMP of this boot, `boot`; A5a review m1); the window is chosen on the wall clock
    (--since @<time.time() - span>) and only selects the entries; an entry of another boot ended before this boot.  Only the fields JOURNAL_FIELDS are asked (m6).  The journal must be readable first
    (journal_alive).  Returns dict(unit, monotonic, age_seconds, boot_id, query, alive) or None (none in the window); Refused (stage machine) when the journal cannot be read."""
    run = subprocess.run if run is None else run
    now = time.monotonic() if now is None else float(now)
    boot = boot_id() if boot is None else boot
    alive = journal_alive(run=run)
    argv, entries = _journal(run, '--since', f'@{int(time.time()-span)}')
    skip = {str(u).removesuffix('.service') for u in exclude}
    newest = None
    for entry in entries:
        units = {str(entry.get(k)).removesuffix('.service') for k in ('USER_UNIT', '_SYSTEMD_USER_UNIT') if isinstance(entry.get(k), str)}
        units = sorted(u for u in units if u.startswith(PROJECT_UNIT_PREFIX) and u not in skip)
        if not units:
            continue
        mono, entry_boot = _monotonic_of(entry)
        if boot is not None and entry_boot != boot:
            continue                                # another boot: that unit ended before this boot began
        if newest is None or mono > newest['monotonic']:
            newest = dict(unit=units[0], monotonic=mono, age_seconds=round(now-mono, 3), boot_id=entry_boot)
    return None if newest is None else dict(newest, query=argv, alive=alive)


def machine_gate(*, own_units=(), parent_units=(), run=None, meminfo='/proc/meminfo', loadavg='/proc/loadavg', mem_min_gib=MEM_AVAILABLE_MIN_GIB, load_limit=LOAD_LIMIT, now=None, boot=None,
                 cgroup=None):
    """hi_core.machine_gate (no active tag-* or seq-vp-* unit but this process's own, which counts only when /proc/self/cgroup shows this process in it; no GPU compute process; utilization <= 20 %),
    then MemAvailable >= 6 GiB and the 1-minute load < 1.5 (HE's gate), then the previous unit of this project ended at least UNIT_GAP_SECONDS ago (decision_log 2026-11-10 ~3:16 AM CST: dmesg runs
    ~4 s ahead of /proc/uptime here, the module docstring), by two readings, the later one counting (A5a review I1):
      - the journal: the user journal readable (journal_alive) and the newest entry of a tag- unit other than this process's own and `parent_units` (last_unit_end);
      - this process's own record of the last unit it ran (note_unit_end; the lower bound when journald has not yet taken the manager's last lines in).
    Both on the monotonic clock (`now`: time.monotonic() unless the tests give one; `boot`: this boot's id; A5a review m1).  `parent_units`: the units allowed to be active besides, by exact name --
    the chain's unit while it waits for the driver's unit (hi_core's gate cannot know it: the driver does not run in it).  Returns the record; Refused (stage machine) otherwise."""
    run = subprocess.run if run is None else run
    parents = tuple(parent_units)

    def filtered(argv, **kwargs):
        done = run(argv, **kwargs)
        if parents and list(argv[:2]) == ['systemctl', '--user'] and getattr(done, 'returncode', 1) == 0:
            kept = [line for line in done.stdout.splitlines() if not (line.split() and line.split()[0].removesuffix('.service') in parents)]
            done = subprocess.CompletedProcess(done.args, done.returncode, ''.join(f'{line}\n' for line in kept), done.stderr)
        return done
    try:
        record = _hi().machine_gate(run=filtered, own_units=tuple(own_units), **({} if cgroup is None else dict(cgroup=cgroup)))     # cgroup: the tests' /proc/self/cgroup (A5a review m2)
    except _hi().HIRefused as exc:
        raise Refused('machine', str(exc)) from None
    mem, load = mem_available_gib(meminfo), load1(loadavg)
    record = dict(record, parent_units=list(parents), mem_available_gib=round(mem, 2), mem_min_gib=mem_min_gib, load1=load, load_limit=load_limit)
    if mem < mem_min_gib:
        raise Refused('machine', f'MemAvailable {mem:.2f} GiB < {mem_min_gib} GiB')
    if not load < load_limit:
        raise Refused('machine', f'1-minute load {load} >= {load_limit}')
    t = time.monotonic() if now is None else float(now)
    boot = boot_id() if boot is None else boot
    previous = last_unit_end(run=run, exclude=(*(record.get('own_units') or ()), *parents), now=t, boot=boot)
    local = local_unit_end()
    record.update(previous_unit=previous, local_unit_end=local, unit_gap_seconds=UNIT_GAP_SECONDS,
                  unit_gap_rule='the previous tag- unit of this project ended >= UNIT_GAP_SECONDS ago: its newest journal entry and this process\'s own record of its last unit, the later one, on '
                                'the monotonic clock; dmesg runs ~4 s ahead of /proc/uptime on this machine (decision_log 2026-11-10 ~3:16 AM CST; A5a review I1, m1)')
    if previous is not None and t-previous['monotonic'] < UNIT_GAP_SECONDS:
        raise Refused('machine', f"the previous unit {previous['unit']} wrote its last journal entry {t-previous['monotonic']:.1f} s ago (< {UNIT_GAP_SECONDS:.0f} s): the next unit waits, so that "
                                 'the kernel lines of the previous one (dmesg ~4 s ahead of /proc/uptime) stay out of its window')
    if local is not None and (boot is None or local.get('boot_id') in (None, boot)) and t-local['monotonic'] < UNIT_GAP_SECONDS:
        raise Refused('machine', f"this queue's previous unit {local['unit']} ended {t-local['monotonic']:.1f} s ago (< {UNIT_GAP_SECONDS:.0f} s; its own record: the journal may not have "
                                 'taken the end in yet): the next unit waits')
    return record


# ---- the one-shot lock (hi_core's take_lock; after every check) --------------------------------------------------------------------------------
def check_once(lock, *, resume=None):
    """No lock (a first run) or, with a resume record that load_resume accepted, the first run's lock there and the resume's own lock not.  Refused (stage once) otherwise; nothing read."""
    lock = Path(lock)
    if resume is None and lock.exists():
        raise Refused('once', f'{lock} exists: this cell ran; it runs once (prereg §5.4: a rerun only through an explicit, committed resume record)')
    if resume is not None:
        if not lock.exists():
            raise Refused('once', f'{lock} does not exist: there is no run to resume')
        if Path(resume.lock_path).exists():
            raise Refused('once', f'{resume.lock_path} exists: the run was resumed already (at most once)')


def take_lock(path, line):
    """hi_core.take_lock: O_EXCL, written once, never edited or deleted.  Refused (stage once) if it appeared."""
    guard_root(path)
    try:
        return _hi().take_lock(path, line)
    except _hi().HIRefused as exc:
        raise Refused('once', str(exc)) from None


def lock_line(**fields):
    return dict(fields, utc=utc(), **_hi().process_identity())


def resume_lock_path(lock):
    lock = Path(lock)
    return lock.with_name(f'{lock.name}.resume')


@dataclass(frozen=True)
class Authorization:
    kind: str
    path: str
    sha256: str
    document: MappingProxyType


@dataclass(frozen=True)
class Resume:
    path: str
    sha256: str
    lock_path: str
    previous_record: str
    document: MappingProxyType


COMMON_AUTH_KEYS = ('schema', 'written_by', 'words', 'time_local', 'utc', 'reason', 'tool', 'arm', 'pair', 'pair_set')
AUTH_KEYS = dict(resume=COMMON_AUTH_KEYS+('lock_sha256', 'previous_record', 'previous_record_sha256'),
                 rerun=COMMON_AUTH_KEYS+('refusals',),
                 no_resume=COMMON_AUTH_KEYS+('record', 'record_sha256', 'note'))
NO_RESUME_NOTE = '未批补跑'


def authorization_dir(pair_set):
    return Path(AUTH_ROOT)/AUTH_DIRS[pair_set]


def authorization_name(pair_set, tool, kind):
    return rf'{re.escape(AUTH_PREFIX[pair_set])}{tool}_{kind}_[0-9]{{1,3}}\.json'


def load_authorization(path, *, kind, tool, arm, pair, pair_set):
    """A committed authorization of `kind` for this cell: in the pair set's authorization directory of the code tree (AUTH_ROOT) under its name pattern; committed and unchanged (hi_core.tracked_unchanged,
    literal pathspecs); strict JSON with exactly AUTH_KEYS[kind]; schema bayesreg.<tool>.<kind>.v1; written by BZ or "lead on BZ's behalf" with BZ's words (non-empty), the local time, the reason;
    a real UTC time; this tool, arm, pair and pair set.  The kind's own checks are the caller's.  Refused (stage <kind>) otherwise."""
    root = Path(AUTH_ROOT)
    path = Path(path)
    if not path.is_absolute():
        path = root/path
    if path.resolve().parent != authorization_dir(pair_set).resolve() or not re.fullmatch(authorization_name(pair_set, tool, kind), path.name):
        raise Refused(kind, f'{path} is not {AUTH_DIRS[pair_set]}/{AUTH_PREFIX[pair_set]}{tool}_{kind}_<k>.json in the code tree {root}')
    if not path.is_file():
        raise Refused(kind, f'{path}: no such file')
    try:
        tracked = _hi().tracked_unchanged(path, root, stage=kind)
    except _hi().HIRefused as exc:
        raise Refused(kind, f'not committed and unchanged: {exc}') from None
    raw = path.read_bytes()
    try:
        from bayesreg.io.guard import strict_json_object
        document = json.loads(raw.decode('utf-8'), object_pairs_hook=strict_json_object)
    except Exception as exc:                        # noqa: BLE001
        raise Refused(kind, f'{path}: not strict JSON ({exc})') from None
    keys = AUTH_KEYS[kind]
    if not isinstance(document, dict) or sorted(document) != sorted(keys):
        raise Refused(kind, f'{path}: exactly the keys {list(keys)} are required')
    problems = []
    if document['schema'] != f'bayesreg.{tool}.{kind}.v1':
        problems.append(f"schema {document['schema']!r}")
    if document['written_by'] not in RESUME_WRITERS:
        problems.append(f"written_by {document['written_by']!r} is not one of {RESUME_WRITERS}")
    for key in ('words', 'time_local', 'reason'):
        if not (isinstance(document[key], str) and document[key].strip()):
            problems.append(f'{key}: a non-empty string is required')
    if pair_set == LPBA40_PAIR_SET and kind == 'resume' and document['written_by'] == RESUME_WRITERS[1] and document['words'] != LPBA40_BZ_WORDS:
        problems.append(f"words {document['words']!r}: a resume written on BZ's behalf quotes BZ's words of decision_log line 520 word for word ({LPBA40_BZ_WORDS!r}; C26.8-3)")
    problem = _hi().utc_problem(document['utc'], 'utc')
    if problem:
        problems.append(problem)
    if (document['tool'], document['arm'], document['pair'], document['pair_set']) != (tool, arm, pair, pair_set):
        problems.append(f"tool/arm/pair/pair_set {document['tool']}/{document['arm']}/{document['pair']}/{document['pair_set']} are not {tool}/{arm}/{pair}/{pair_set}")
    if problems:
        raise Refused(kind, f'{path}: {problems}')
    return Authorization(kind=kind, path=str(path), sha256=hashlib.sha256(raw).hexdigest(), document=MappingProxyType(dict(document, git=tracked)))


def find_authorizations(*, kind, tool, arm, pair, pair_set):
    """(valid, refused): every file of the kind's name pattern in the authorization directory whose arm, pair and pair set are this cell's, checked by load_authorization."""
    directory = authorization_dir(pair_set)
    valid, refused = [], []
    if not directory.is_dir():
        return valid, refused
    for path in sorted(directory.iterdir()):
        if not re.fullmatch(authorization_name(pair_set, tool, kind), path.name):
            continue
        try:
            doc = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            refused.append(dict(path=str(path), problem='not JSON'))
            continue
        if not isinstance(doc, dict) or (doc.get('arm'), doc.get('pair'), doc.get('pair_set')) != (arm, pair, pair_set):
            continue
        try:
            valid.append(load_authorization(path, kind=kind, tool=tool, arm=arm, pair=pair, pair_set=pair_set))
        except Refused as exc:
            refused.append(dict(path=str(path), problem=str(exc)))
    return valid, refused


def load_resume(path, *, tool, arm, pair, pair_set, lock, previous_record, complete):
    """Prereg §5.4's resume record for one cell (load_authorization, kind resume), and: the first run's lock (its sha256), the previous record (its path and sha256, or null when the run left none),
    which is not complete (`complete()` False: only cells without a complete record are resumed -- an infrastructure failure, a wrapper error under the stop policy, needs_lead_rejudgment, or a record
    the unit's evidence overrode), and the resume's own lock not there (one resume per failed run).  Refused (stage resume) otherwise."""
    auth = load_authorization(path, kind='resume', tool=tool, arm=arm, pair=pair, pair_set=pair_set)
    document, lock, previous_record = auth.document, Path(lock), Path(previous_record)
    problems = []
    if not lock.is_file() or document['lock_sha256'] != sha256_file(lock):
        problems.append(f"lock_sha256 {document['lock_sha256']!r} is not the sha256 of {lock}")
    if previous_record.is_file():
        if document['previous_record'] != str(previous_record) or document['previous_record_sha256'] != sha256_file(previous_record):
            problems.append(f"previous_record {document['previous_record']!r} / {document['previous_record_sha256']!r} are not {previous_record} and its sha256")
    elif document['previous_record_sha256'] is not None:
        problems.append(f'previous_record_sha256 is given but {previous_record} is not there')
    if complete():
        problems.append(f'{previous_record} is a complete record and no unit evidence overrode it: nothing to resume (a method failure is not rerun)')
    own = resume_lock_path(lock)
    if own.exists():
        problems.append(f'{own} exists: this cell was resumed already (at most once)')
    if problems:
        raise Refused('resume', f'{path}: {problems}')
    return Resume(path=auth.path, sha256=auth.sha256, lock_path=str(own), previous_record=str(previous_record), document=auth.document)


def refusal_dir(batch_dir):
    return guard_root(Path(batch_dir)/'refusals')


def write_refusal(batch_dir, document):
    """I1: a refusal before the lock is recorded beside the cell, never in it: one new file in <batch>/refusals/ per refusal (written once, never overwritten; the evidence stays)."""
    directory = refusal_dir(batch_dir)
    cell = re.sub(r'[^A-Za-z0-9._-]', '_', f"{document['arm']}__{document['pair']}")[:160]          # an unknown arm or a malformed pair is refused too: its name is made plain
    name = f"{utc().replace(':', '').replace('-', '')}_{os.getpid()}_{cell}.json"
    path = directory/name
    k = 1
    while path.exists():
        k += 1
        path = directory/name.replace('.json', f'_{k}.json')
    write_json_once(path, dict(document, schema='bayesreg.holdout.refusal.v1', utc=utc(), pid=os.getpid()))
    return path


RERUN_STAGE = 'rerun'


def cell_refusals(record_root, *, arm, pair, pair_set, counted=True):
    """Every refusal record of this cell under <record root>/batches/*/refusals/ (path and sha256), in name order.  counted=True leaves out the refusals of stage `rerun` (an attempt refused because it
    had no authorization: counting them would make every unauthorized attempt void the authorization BZ is writing)."""
    out = []
    for path in sorted(Path(record_root).glob('batches/*/refusals/*.json')):
        try:
            doc = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if (doc.get('arm'), doc.get('pair'), doc.get('pair_set')) == (arm, pair, pair_set) and not (counted and doc.get('stage') == RERUN_STAGE):
            out.append(dict(path=str(path), sha256=sha256_file(path), stage=doc.get('stage')))
    return out


# ---- P20: refused before the lock after a rerun, BZ's no_resume (prereg revision 2 item 17 (f); decision_log 2026-11-10 ~2:52 and ~2:53 AM CST) ---------------
# A cell refused before its lock (a refusal record in <batch>/refusals/, write_refusal) that is refused again after a rerun authorization is written -- no output, cause P20 -- by BZ's committed
# no_resume record whose record is the newest refusal record of the cell.  The running side copies every refusal record of the cell into the cell's record directory (the evaluation entry reads that
# directory only): refusals/<batch>__<name>, byte for byte, with <copy>.sha256 ("<sha256>  <source>").  The format of experiments/hb/hb_configc.keep_refusals (the configuration C arm, the lead's item 15
# of HB's revision round); the XB chain uses these functions.  "Newest" is by the records' sequence number `seq` (HB diff review n4: never the files' mtime or names).
REFUSAL_COPIES = 'refusals'


def refusal_seq(path):
    """The sequence number `seq` of a refusal record (an int >= 1), or None (none, not a number, the file unreadable)."""
    try:
        doc = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    seq = doc.get('seq') if isinstance(doc, dict) else None
    return seq if isinstance(seq, int) and not isinstance(seq, bool) and seq >= 1 else None


def next_refusal_seq(record_root, *, arm, pair, pair_set):
    """The sequence number of the cell's next refusal record: one more than the largest of its refusal records (those of stage rerun included), 1 for the first; None when they cannot be listed (the
    record is written all the same; refusals_by_seq then refuses to name the newest, to the lead)."""
    try:
        rows = cell_refusals(record_root, arm=arm, pair=pair, pair_set=pair_set, counted=False)
    except OSError:
        return None
    return 1+max((q for q in (refusal_seq(r['path']) for r in rows) if q is not None), default=0)


def refusals_by_seq(paths):
    """[(seq, path)] by the records' sequence numbers, oldest first; Refused (stage refusal_record) when one has none or two share one: the newest is then not known (to the lead).  The ordering is
    bayesreg.io.rejudgment.by_seq (A5b review minor 6: one ordering for the XB chain, the configuration C queue and the evaluation entry)."""
    from bayesreg.io.rejudgment import by_seq
    rows, problem = by_seq(paths, seq_of=refusal_seq)
    if problem:
        raise Refused('refusal_record', f'refusal {problem}')
    return rows


def refusal_copy_path(record_dir, source):
    """<record_dir>/refusals/<batch>__<name> of a refusal record <record root>/batches/<batch>/refusals/<name>."""
    source = Path(source)
    return Path(record_dir)/REFUSAL_COPIES/f'{source.parent.parent.name}__{source.name}'


def keep_refusals(record_root, record_dir, *, arm, pair, pair_set):
    """Every refusal record of the cell (cell_refusals, those of stage rerun included) copied once into its record directory (refusal_copy_path), byte for byte, with <copy>.sha256 written once
    ("<sha256>  <source>"); a copy already there is checked, not rewritten.  A copy whose sha256 is not its source's refuses (stage refusal_copy).  Returns [dict(path, sha256, source, seq)] in the
    sources' name order (seq None when a record has none: copying does not need it, naming the newest does)."""
    out = []
    for row in cell_refusals(record_root, arm=arm, pair=pair, pair_set=pair_set, counted=False):
        src = Path(row['path'])
        copy = Path(guard_root(refusal_copy_path(record_dir, src)))
        if not copy.exists():
            copy.parent.mkdir(parents=True, exist_ok=True)
            with open(src, 'rb') as stream:
                data = stream.read()
            with open(copy, 'xb') as stream:
                stream.write(data)
            with open(copy.with_name(copy.name+'.sha256'), 'x', encoding='utf-8') as stream:
                stream.write(f'{sha256_file(copy)}  {src}\n')
        if sha256_file(copy) != row['sha256']:
            raise Refused('refusal_copy', f'{copy} is not the refusal record {src} (sha256)')
        out.append(dict(path=str(copy), sha256=row['sha256'], source=str(src), seq=refusal_seq(copy)))
    return out


def refusal_copies(record_dir):
    """The copies of the cell's refusal records in its record directory (refusals/*.json, the .sha256 files left out), in name order."""
    directory = Path(record_dir)/REFUSAL_COPIES
    return sorted(directory.glob('*.json')) if directory.is_dir() else []


def newest_refusal(record_dir):
    """(the newest copy by seq, its source named in <copy>.sha256 or None), or None without copies; Refused (refusals_by_seq) when the copies' numbers do not tell."""
    copies = refusal_copies(record_dir)
    if not copies:
        return None
    newest = refusals_by_seq(copies)[-1][1]
    side = newest.with_name(newest.name+'.sha256')
    source = (side.read_text(encoding='utf-8').split('  ', 1)+[''])[1].strip() if side.is_file() else ''
    return newest, (Path(source) if source else None)


def declined_refusal(record_dir, *, tool, arm, pair, pair_set):
    """P20 (prereg revision 2 item 17 (f)): BZ's committed no_resume record (no_resume_authorization: tool, arm, pair, pair set, note 未批补跑) whose record is the newest refusal record of the cell
    as its copy in this cell's record directory (refusals/<batch>__<name>), with the copy's sha256, and whose reason names P20 (bayesreg.io.rejudgment.p20_reason_problem: A5b review I2, the
    evaluation entry's criterion; a record without it is not taken, the batch stops there as before).  Only the copy (A5a review m4, the lead's ruling ~12:29 PM CST 11-10): a no_resume naming the
    source it was copied from is not taken, as the evaluation entry (A5b) reads the record directory only.  The authorization, or None (no copy, or no such record); Refused when the copies' sequence
    numbers do not tell the newest."""
    found = newest_refusal(record_dir)
    if found is None:
        return None
    newest, _source = found
    digest = sha256_file(newest)
    from bayesreg.io.rejudgment import p20_reason_problem
    return no_resume_authorization(tool=tool, arm=arm, pair=pair, pair_set=pair_set, record=newest, complete=lambda: False,
                                   accept=lambda a: a.document.get('record_sha256') == digest and p20_reason_problem(a.document.get('reason')) is None)


def rerun_authorization(record_root, *, tool, arm, pair, pair_set):
    """I1 (C23.12-6, D107): a cell refused before its lock runs again only with a committed rerun authorization whose `refusals` are exactly this cell's refusal records (path and sha256), all of them
    but those of stage rerun (cell_refusals).  An authorization so covers one set: a further refusal makes it stale, and the next attempt needs a new one.  None when the cell has no refusal record;
    Refused (stage rerun) when it has and no authorization covers them."""
    prior = cell_refusals(record_root, arm=arm, pair=pair, pair_set=pair_set)
    if not prior:
        return None
    want = sorted((r['path'], r['sha256']) for r in prior)
    valid, refused = find_authorizations(kind='rerun', tool=tool, arm=arm, pair=pair, pair_set=pair_set)
    for auth in valid:
        listed = auth.document['refusals']
        if isinstance(listed, list) and all(isinstance(r, dict) for r in listed) and sorted((r.get('path'), r.get('sha256')) for r in listed) == want:
            return dict(path=auth.path, sha256=auth.sha256, refusals=[dict(path=p, sha256=h) for p, h in want])
    raise Refused(RERUN_STAGE, f'{arm} {pair} ({pair_set}) was refused before its lock {len(prior)} time(s) ({[r["stage"] for r in prior]}); a rerun needs a committed authorization listing exactly these '
                           f'refusal records {want} (C23.12-6, D107); found {[a.path for a in valid]}, refused {refused}')


def no_resume_authorization(*, tool, arm, pair, pair_set, record, complete, accept=None):
    """I3: the committed "no resume" (未批补跑) record of a cell: load_authorization (kind no_resume), its note 未批补跑, its record the cell's terminal (or overriding) record with that sha256 now,
    and that record not complete; `accept(auth)` (optional, HB: the reason of BZ's (甲) names the case of C22.5-6) true.  The first such authorization in name order, or None when there is none."""
    valid, _ = find_authorizations(kind='no_resume', tool=tool, arm=arm, pair=pair, pair_set=pair_set)
    for auth in valid:
        doc = auth.document
        path = Path(doc['record'])
        if doc['note'] == NO_RESUME_NOTE and path == Path(record) and path.is_file() and sha256_file(path) == doc['record_sha256'] and not complete() and (accept is None or accept(auth)):
            return auth
    return None


# ---- the rejudgment records (prereg revision 2 item 18; decision_log 2026-11-10 NI2, NI3, N1, N2; the lead's item 13 of HB's revision round) --------
# One format, one reader: the XB chain (A5a), the evaluation entry (A5b) and HB's configuration C queue call `rejudgment` -- the core in bayesreg/io/rejudgment.py (A5b item 6), wrapped here; the
# constants below are the core's (a test compares).  Only the external arms' and the configuration C arm's
# needs_lead_rejudgment are judged this way; the method arm uses holdout_common's resume / no_resume records, the review unit's are a trace only (N1).
REJUDGMENT_DIR_REL = 'runs/holdout_B_v1/rejudgments'          # <arm>/<pair>_<k>.json, k from 1, added only (a later commit may add files under runs/holdout_B_v1/)
REJUDGMENT_SCHEMA = 'bayesreg.holdout.rejudgment.v1'
REJUDGMENT_KEYS = ('schema', 'arm', 'pair', 'pair_set', 'status', 'record', 'evidence', 'judgment', 'reason', 'written_by', 'words', 'time_local', 'utc')     # no resume key (N1)
JUDGMENTS = ('infrastructure', 'unit_limit', 'no_output')
LEAD_WRITERS = ('lead',)+RESUME_WRITERS                         # infrastructure and unit_limit: the lead judges from the evidence; no_output and every P20 cell: BZ's (RESUME_WRITERS)
REJUDGMENT_NAME = re.compile(r'(?P<pair>.+)_(?P<k>[1-9][0-9]{0,2})\.json', re.ASCII)


def rejudgment(arm, pair, *, record, pair_set='holdout', root=None):
    """(found, problems): the cell's rejudgment record (prereg revision 2 item 18), the highest k of <root>/runs/holdout_B_v1/rejudgments/<arm>/<pair>_<k>.json (root: AUTH_ROOT, the code tree).
    The reader's core is bayesreg/io/rejudgment.py's `rejudgment` (A5b item 6: one format, one reader -- the evaluation entry, which imports nothing of experiments/, calls the core; this name wraps it
    for the XB chain and HB's configuration C queue, which keep calling holdout_common.rejudgment).  It holds when: committed and unchanged (git ls-files and git status of that path, hi_core.
    tracked_unchanged's rule); strict JSON with exactly REJUDGMENT_KEYS; schema REJUDGMENT_SCHEMA; arm, pair, pair_set this cell's; status a non-empty string; record {path, sha256} with path the
    given `record` (the terminal or stop record it judges) and sha256 that file's now (NI3: the record it judged); evidence a list of {path, sha256}; judgment one of JUDGMENTS; reason, time_local, utc
    non-empty; written_by: 'lead' or BZ's for infrastructure and unit_limit, BZ's (BZ, "lead on BZ's behalf") for no_output and for a status hx_provenance_rejected (P20), with BZ's words non-empty
    then (words may be empty otherwise).  found: dict(path, sha256, k, document) or None (none there, or the highest does not hold: problems say why).  An arm that is not a plain name refuses
    (safe_name, as before); otherwise never raises.  A rerun is never authorized here: it needs holdout_common's resume record (prereg §5.4)."""
    from bayesreg.io import rejudgment as core
    found, problems = core.rejudgment(safe_name(arm, 'arm'), pair, record=record, pair_set=pair_set, root=Path(AUTH_ROOT if root is None else root))
    if found is None:
        return None, problems
    return {k: found[k] for k in ('path', 'sha256', 'k', 'document')}, problems


# ---- moving to D: -------------------------------------------------------------------------------------------------------------------------------
def move_files(src_dir, dst_dir, record_path, *, names=None):
    """Move files from a working directory to D: (prereg §5.3): for each file (all regular files of src_dir, or `names`), sha256 before, copy, sha256 of the copy; equal -> the source removed.
    A file already at the destination is never overwritten.  The move record is written once as JSON, also when the move stops half way (m4: an OSError, e.g. D: full): the files moved so far and the
    error are in it.  A difference or an OSError keeps both copies of the file at hand and raises InfrastructureFailure (move_mismatch, move_error)."""
    src_dir, dst_dir = Path(src_dir), Path(guard_root(dst_dir))
    rows, bad, error = [], [], None
    try:
        files = sorted(p for p in src_dir.rglob('*') if p.is_file()) if names is None else [src_dir/n for n in names]
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            rel = src.relative_to(src_dir)
            dst = dst_dir/rel
            if dst.exists():
                bad.append(f'{dst} exists')
                rows.append(dict(file=str(rel), src=str(src), dst=str(dst), ok=False, problem='destination exists'))
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            before = sha256_file(src)
            row = dict(file=str(rel), src=str(src), dst=str(dst), bytes=src.stat().st_size, sha256_before=before, ok=False, problem='copying')
            rows.append(row)
            shutil.copyfile(src, dst)
            with open(dst, 'rb') as stream:
                os.fsync(stream.fileno())
            after = sha256_file(dst)
            row.update(sha256_after=after, ok=before == after, problem=None if before == after else 'sha256 differs')
            if row['ok']:
                src.unlink()
            else:
                bad.append(f'{rel}: {before} -> {after}')
    except OSError as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        record = dict(schema='bayesreg.holdout.move.v1', src_dir=str(src_dir), dst_dir=str(dst_dir), utc=utc(), files=rows, passed=not bad and error is None, problems=bad, error=error,
                      rule='sha256 before and after; equal -> source removed; a difference or an error keeps both and stops (prereg §5.3)')
        write_json_once(record_path, record)
    if error is not None:
        raise InfrastructureFailure('move_error', f'{error}; the files moved so far and the rest are listed in {record_path}')
    if bad:
        raise InfrastructureFailure('move_mismatch', f'{bad}; both copies kept; record {record_path}')
    return record


# ---- the code -----------------------------------------------------------------------------------------------------------------------------------
def code_identity(repo=None, run=None):
    """git HEAD and whether the tracked files are clean (git status --porcelain --untracked-files=no), plus the untracked files under experiments/ (the driver imports hi_core, he/unit_probe and their
    imports), bayesreg/, jd_curl/ and configs/ (a module there would run without a commit).  Recorded; the runner refuses a dirty tree for a hold-out run."""
    repo = Path(REPO if repo is None else repo)
    run = subprocess.run if run is None else run

    def git(*args):
        try:
            done = run(['git', '--literal-pathspecs', '-C', str(repo), *args], capture_output=True, text=True, timeout=60)
            return done.returncode, done.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return 127, f'{type(exc).__name__}: {exc}'
    rc_head, head = git('rev-parse', 'HEAD')
    rc_status, status = git('status', '--porcelain', '--untracked-files=no')
    rc_untracked, untracked = git('ls-files', '--others', '--exclude-standard', '--', 'experiments', 'bayesreg', 'jd_curl', 'configs')      # m9: hi_core, he/unit_probe and their imports too
    clean = rc_head == 0 and rc_status == 0 and status == '' and rc_untracked == 0 and untracked == ''
    return dict(root=str(repo), commit=head if rc_head == 0 else None, clean=clean, status=status, untracked=untracked.splitlines() if untracked else [])


# The code identity of every hold-out unit (decision_log 2026-11-09 df8541d; 2026-11-10 ~2:45 AM CST N3, ~2:59 AM CST (1)-(4), (7); the lead's item 14 of HB's revision round): the rules live in
# bayesreg/io/code_identity.py (the evaluation entry calls the same functions); this module only wraps them.  Its names (CODE_DIRS, PINNED_CODE_DIRS, LATER_AUTHORIZATION, LATER_RECORDS,
# LATER_DECISION_LOG, ROOT_IMPORTABLE, PINNED_RECORD_REL, ...) are read through this module's __getattr__.
CODE_IDENTITY_NAMES = frozenset(('CODE_DIRS', 'PINNED_CODE_DIRS', 'LATER_AUTHORIZATION', 'LATER_RECORDS', 'LATER_DECISION_LOG', 'ROOT_IMPORTABLE', 'PINNED_RECORD_REL', 'PINNED_RECORD_SCHEMA',
                                 'PINNED_RECORD_KEYS', 'PINNED_RECORD_WRITERS', 'UNSEAL_RECORD_REL', 'DECISION_CONSTANTS', 'CACHE_FILE'))


def _ci():
    """bayesreg/io/code_identity.py, imported when first needed (the in-unit supervisor imports this module and needs none of it)."""
    from bayesreg.io import code_identity
    return code_identity


def code_dir_file_problem(path, sources):
    return _ci().code_dir_file_problem(path, sources)


def allowlist_inclusion_problems(pinned, *, repo=None, run=None, head='HEAD'):
    """bayesreg.io.code_identity.allowlist_inclusion_problems (the code tree `repo` defaults to this module's, REPO)."""
    return _ci().allowlist_inclusion_problems(pinned, repo=REPO if repo is None else repo, run=run, head=head)


DRY_RUN_SCHEMA = 'bayesreg.holdout.dry_run_summary.v1'
P0_EXPECTED = dict(                                 # A5a review m8: the refusals a hold-out entry's dry run on P0 (before BZ unseals) is expected to meet; nothing else
    rule_one_unpinned='rule one fails only because runs/holdout_B_v1/pinned_code.json is not committed yet (contract C24.3-2: the lead commits it after the unseal, P1)',
    no_unseal_record='the unseal record splits/bayesreg_v1/unseal_holdout_B_v1.json is not there yet (BZ writes it at the unseal, C23.3)',
    decision_pending="a development check's decision constant is still None (HP: the check runs on P0, its constant is filled in P1; prereg revision 2 item 9 steps 4-5)")


def rule_one_kind(problems):
    """A5a review m8: 'rule_one_unpinned' when every problem of rule one is that runs/holdout_B_v1/pinned_code.json is not committed (bayesreg.io.code_identity.pinned_record's words), else
    'rule_one'."""
    head = f'{_ci().PINNED_RECORD_REL} is not committed at '
    problems = [str(p) for p in problems]
    return 'rule_one_unpinned' if problems and all(p.startswith(head) for p in problems) else 'rule_one'


def unseal_record_exists(tree=None):
    """Whether bayesreg's code tree (bayesreg.io.sealed.REPO, read at call time) holds the unseal record at its one path."""
    from bayesreg.io import sealed as sealed_module
    return (Path(sealed_module.REPO if tree is None else tree)/UNSEAL_REL).exists()


def dry_run_summary(rows):
    """A5a review m8 (the lead's ruling ~12:29 PM CST 11-10): the fields every hold-out entry's dry-run record carries, the same way (HP's two entries, HI, XB's driver): `rows` are
    (kind, message), one per problem the dry run met (rule one's under rule_one_kind, a refusal at the unseal under 'no_unseal_record' when the record is not there, ...).  Returns passed (no problem
    at all), problems (every message), expected (the problems of a kind in P0_EXPECTED, with the kind), unexpected (the rest), p0_ok (unexpected == []).  The checklist (prereg revision 2 items 6
    and 14) reads these fields of the record -- on P0, p0_ok; after the unseal and P1, passed -- never the exit code."""
    rows = [(str(k), str(m)) for k, m in rows]
    unexpected = [dict(kind=k, problem=m) for k, m in rows if k not in P0_EXPECTED]
    return dict(passed=not rows, problems=[m for _, m in rows], expected=[dict(kind=k, problem=m) for k, m in rows if k in P0_EXPECTED], unexpected=unexpected, p0_ok=not unexpected,
                summary_schema=DRY_RUN_SCHEMA, p0_expected_kinds=sorted(P0_EXPECTED))


def pinned_code_problems(pinned=None, *, repo=None, run=None):
    """Rule one for a hold-out process: bayesreg.io.code_identity.unit_identity_problems (P1 from the committed runs/holdout_B_v1/pinned_code.json, never HEAD; `pinned`, when given, only compared
    with it; every commit of P1..HEAD checked one by one; the code: the five directories, the root's importable files, new top-level directories).  (record, problems); record['pinned'] is P1."""
    return _ci().unit_identity_problems(pinned, repo=REPO if repo is None else repo, run=run)


def _strict_pairs(pairs):
    return _ci().strict_pairs(pairs)


def wait_for(check, *, seconds=GATE_WAIT_SECONDS, poll=30., now=time.monotonic, sleep=time.sleep, log=print):
    """Call `check` until it returns (a record); Refused or InfrastructureFailure from it are retried until `seconds` have passed, then raised as InfrastructureFailure(precheck_timeout)
    (prereg §5.4: a precheck that does not pass in 15 minutes)."""
    started = now()
    while True:
        try:
            return check()
        except (Refused, InfrastructureFailure) as exc:
            if now()-started >= seconds:
                raise InfrastructureFailure('precheck_timeout', f'not passed in {seconds:.0f} s: {exc}') from None
            log(f'waiting: {exc}')
            sleep(poll)


# ---- unit limits against machine failures (prereg revision 1c items 1, 2) ---------------------------------------------------------------------
LIMIT_RULES = ('yi', 'jia')
LIMIT_RULE = 'yi'                               # (乙) by default; (甲) is the alternative (the lead and BZ decide)
OOM_LINE = re.compile(r'^\[\s*([0-9.]+)\]\s*oom-kill:(\S+)')
KERNEL_TIME = re.compile(r'^\[\s*([0-9.]+)\]')
DXG = 'create_allocation failed'
JOURNAL_OOM = 'killed by the OOM killer'


def parse_oom(text, since=0.):
    """The kernel's oom-kill lines at or after uptime `since`: [{t, constraint, oom_memcg, task_memcg, task, pid, global_oom}]; None when dmesg could not be read (text None)."""
    if text is None:
        return None
    rows = []
    for line in text.splitlines():
        m = OOM_LINE.match(line)
        if not m or float(m.group(1)) < since:
            continue
        fields = {}
        for part in m.group(2).split(','):
            key, eq, value = part.partition('=')
            fields[key] = value if eq else True
        rows.append(dict(t=float(m.group(1)), constraint=fields.get('constraint'), oom_memcg=fields.get('oom_memcg'), task_memcg=fields.get('task_memcg'), task=fields.get('task'),
                         pid=fields.get('pid'), global_oom=bool(fields.get('global_oom'))))
    return rows


def dxg_lines(text, since=0.):
    if text is None:
        return None
    return sum(1 for line in text.splitlines() if (m := KERNEL_TIME.match(line)) and float(m.group(1)) >= since and DXG in line)


def events_delta(before, after):
    """memory.events counts during a step (after - before); None when either was not read (never read as 0)."""
    if after is None or before is None:
        return None
    return {k: after.get(k, 0)-before.get(k, 0) for k in set(before) | set(after)}


def in_unit(cgroup, unit, unit_cgroup=None):
    """Whether a cgroup path is the unit's or below it (by the unit's cgroup path when the probe read it, else by the unit's name)."""
    if not isinstance(cgroup, str):
        return False
    if unit_cgroup:
        root = unit_cgroup.rstrip('/')
        return cgroup.rstrip('/') == root or cgroup.startswith(root+'/')
    return bool(unit) and (cgroup.rstrip('/').endswith(f'/{unit}.service') or f'/{unit}.service/' in cgroup)


def is_unit_cgroup(cgroup, unit, unit_cgroup=None):
    """Whether a cgroup path is exactly the unit's (rule (甲): the unit's own memory limit)."""
    if not isinstance(cgroup, str):
        return False
    if unit_cgroup:
        return cgroup.rstrip('/') == unit_cgroup.rstrip('/')
    return bool(unit) and cgroup.rstrip('/').endswith(f'/{unit}.service')


def oom_evidence(*, unit, unit_cgroup=None, oom_lines=None, events=None, journal=None, step_kill=False, systemd_result=None):
    """The items of OOM evidence of this unit (the module docstring's list; systemd's Result oom-kill too, the lead's ruling of 2026-11-09)."""
    items = []
    if systemd_result == 'oom-kill':
        items.append(dict(source='systemd Result oom-kill', own_limit=False))
    for r in oom_lines or []:
        if in_unit(r.get('task_memcg'), unit, unit_cgroup):
            items.append(dict(source='dmesg', constraint=r.get('constraint'), oom_memcg=r.get('oom_memcg'), task_memcg=r.get('task_memcg'), task=r.get('task'), pid=r.get('pid'),
                              own_limit=r.get('constraint') == 'CONSTRAINT_MEMCG' and is_unit_cgroup(r.get('oom_memcg'), unit, unit_cgroup)))
    if events is not None and (events.get('oom', 0) > 0 or events.get('oom_kill', 0) > 0):
        items.append(dict(source='memory.events', oom=events.get('oom', 0), oom_kill=events.get('oom_kill', 0), own_limit=events.get('oom', 0) > 0))
    if journal is not None and JOURNAL_OOM in journal:
        items.append(dict(source='journal', own_limit=False))
    if step_kill:
        items.append(dict(source='exit -9', own_limit=False))
    return items


def classify_unit_end(*, unit, systemd_result, oom_lines, events, journal=None, step_kill=False, memory_guard=False, unit_cgroup=None, rule=None, need=('dmesg', 'end_probe', 'result')):
    """(status, kind, evidence) of the module docstring's rules; status None when the unit ended without a limit, a kill or an OOM (the driver's terminal record decides).
    `oom_lines`: parse_oom of the window (None: dmesg unreadable); `events`: the end probe's memory.events (None: no end probe; for a step, the counts during it); `systemd_result`: the unit's Result
    (None: unknown); `need`: the sources that must be readable when there is no OOM evidence (a step inside a running unit has no Result yet: need=('dmesg', 'end_probe'))."""
    rule = LIMIT_RULE if rule is None else rule
    if rule not in LIMIT_RULES:
        raise ValueError(f'limit rule {rule!r}: one of {LIMIT_RULES}')
    readable = dict(dmesg=oom_lines is not None, end_probe=events is not None, result=systemd_result is not None)
    items = oom_evidence(unit=unit, unit_cgroup=unit_cgroup, oom_lines=oom_lines, events=events, journal=journal, step_kill=step_kill, systemd_result=systemd_result)
    machine = [r for r in oom_lines or [] if r.get('constraint') == 'CONSTRAINT_NONE' or r.get('global_oom')]
    evidence = dict(rule=rule, unit=unit, unit_cgroup=unit_cgroup, systemd_result=systemd_result, readable=readable, oom_evidence=items, machine_oom_lines=machine, memory_guard=memory_guard,
                    memory_events=events)
    if memory_guard:
        return 'infra_failure', 'memory_guard', evidence
    if machine:
        return 'infra_failure', 'machine_oom', evidence
    if items:
        if rule == 'jia':
            if any(i['own_limit'] for i in items):
                return 'unit_limit', 'MemoryMax', evidence
            return 'infra_failure', 'oom_not_the_units_own_limit', evidence
        return 'infra_failure', 'oom', evidence
    missing = [k for k in need if not readable[k]]
    if missing:
        return 'needs_lead_rejudgment', f'unreadable:{",".join(missing)}', evidence
    if systemd_result == 'timeout':
        return 'unit_limit', 'RuntimeMaxSec', evidence
    return None, None, evidence
