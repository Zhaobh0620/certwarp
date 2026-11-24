"""RP (runs/20261105-rp/任务书.md, plan.md): the repeatability of RW's local repair.  The same 19 Hx24@600 fields of fd2_v1 (pinned by rw_sources_v1.json), RW's configuration (RF05, wall_seconds 1200 as a fuse), RW's code; repaired twice more, R2 and R3.  R1 is rw_v1, read only.  Not a method: no F_B, no certified descent, only the final field passes the certificate; development pairs only, in no result table.

Root rp_v1: the queue, its code checkout, batch.json, the reading.  rp_v1/r2 and rp_v1/r3: each stands where run_fl.py expects the root of a batch, with RW's cells (Cell('RHx24', 'c600' or 'rest600', pair)) under its own prefix rp2, rp3, in RW's order (the control 0439-0440, the 5 pairs that hit the 120 s wall, the other 13).
The fallback of 2c(i), two tiers (the lead's ruling, about 9 pm CST 11-05, after the C22 draft; D74, D75 for BZ): first configuration C's field of d0_v1 (arm A0, final_native.npz), pinned by rp_fallback_v1.json, qualified only by d0_v1's own records (fallback_status: the file there and the pinned one, its certificate passed under configuration C, D0's evaluation passed, class ok; no certificate is computed again); where it does not qualify, the identity u = 0 (J = 1), evaluated into rp_v1/id.
rw_v1, fd2_v1, d0_v1 are only read.  No function of this module reads a label.

The code is RW's: every tracked file under IDENTITY_PATHS that is at 85fc80d (RW's commit, rw_v1/batch.json) must be byte for byte the same at the batch's commit; the only files added there may be this package's own (OWN_FILES) or files that are not Python modules (none under a code directory, scripts or the repository's root); and the frozen tree the units import jd_curl from must be eb808be's blobs, file by file (revision 1, important 2).  identity says why not, before anything is checked out.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess

import fx_core as fx
import mc0_core as core
from fx_core import ALL_PAIRS  # noqa: F401  (run_rp.py takes it from here)

HERE = Path(__file__).resolve().parent
RESULTS = fx.RESULTS
ROOT_NAME = 'rp_v1'
REPEATS = (2, 3)
R1_ROOT_NAME = fx.RW.ROOT_NAME                  # rw_v1, read only
D0_ROOT_NAME = 'd0_v1'                          # read only: the fallback's fields
A0_ARM, A0_TASK, A0_FILE = 'A0', 'pg_weak__four__90s', 'final_native.npz'
ID_DIR = 'id'
FALLBACK_MEASURES = ('seg35', 'seg4', 'sdlogj_official', 'hd95', 'sdlogj_brain', 'small_six', 'large_six', 'other_23', 'cortex', 'white_matter')     # what the method's means use; D0's records hold the first five
BASE_COMMIT = '85fc80df02f5bff742ef36968db75031c671a3ea'     # rw_v1/batch.json code_sha: RW's code
QUEUE_UNIT = 'tag-rp-queue'
ID_EVAL_UNIT = 'tag-rp-eval-id'
D0_CLASS_OK = 'ok'                              # experiments/d0/classify_run.py; the others (method_failure, no_certified_map, ...) do not qualify
FALLBACK_FILE = HERE/'rp_fallback_v1.json'
FALLBACK_REL = 'experiments/fl/rp_fallback_v1.json'
OWN_FILES = ('experiments/fl/rp_core.py', 'experiments/fl/decide_rp.py', 'experiments/fl/run_rp.py', FALLBACK_REL)
IDENTITY_PATHS = ('bayesreg', 'jd_curl', 'experiments/fl', 'experiments/ef', 'experiments/mc0', 'experiments/nn', 'experiments/d0', 'scripts', 'configs/bayesreg', 'splits')
CODE_DIRS = ('bayesreg', 'jd_curl', 'experiments', 'scripts')     # a Python file added under these (or at the repository's root: evaluate.py puts the root and scripts first on sys.path) could be imported or shadow a module: refused unless it is this package's
FROZEN_PATHS = ('jd_curl',)                     # what the units import from the frozen tree (a test lists it); plus the root's tracked .py files
D0_EVALUATOR_SHA256 = 'd066e28c0aa21f02cf197fc072ba0d5929035ccbf84e49d03cf3fbb6c367d272'     # experiments/d0/evaluate.py as D0 evaluated A0 (its metrics.json); the function-by-function comparison with today's: the RP review, item 3
CERTIFICATE_C = dict(delta=0., max_depth=48, max_boxes=255, tile_size=65536)     # run_mc0.certificate_config(): configuration C as the evaluation passes it
ROLE = 'RP: RW repeated twice on the same fd2_v1 Hx24@600 fields, the same configuration and code (wall_seconds 1200, a fuse); predicted channels: a probe, NOT oracle, NOT a method, NOT in any result table'


@dataclass(frozen=True)
class RP2Cell(fx.RWCell):
    PREFIX = 'rp2'


@dataclass(frozen=True)
class RP3Cell(fx.RWCell):
    PREFIX = 'rp3'


class RPBatch(fx.RWBatch):
    """One repetition: RW's batch with its own root and prefix, every one of the 19 pairs in the plan (no control branch: RW's control is the reason RW repaired all 19)."""
    ROLE = ROLE

    def __init__(self, rep):
        self.REP = rep
        self.PREFIX = f'rp{rep}'
        self.ROOT_NAME = f'{ROOT_NAME}/r{rep}'
        self.Cell = {2: RP2Cell, 3: RP3Cell}[rep]
        super().__init__()

    def plan(self):
        """RW's order: the control, the 5 pairs that hit the 120 s wall, the other 13."""
        return self.plan_control()+self.plan_wall()+self.plan_reused()

    def all_cells(self, root):
        return self.plan()

    def control_choice(self, root):
        return None


RP2, RP3 = RPBatch(2), RPBatch(3)
BATCHES = {2: RP2, 3: RP3}


class Root:
    """rp_v1: the queue's root; the repetitions' roots under it."""

    def __init__(self):
        self.RESULTS = RESULTS
        self.DEFAULT_ROOT = RESULTS/ROOT_NAME

    def root_problem(self, root):
        return None if Path(root).resolve() == Path(self.DEFAULT_ROOT).resolve() else f'{root}: RP writes only to {self.DEFAULT_ROOT}; nothing read, nothing moved'

    def rep_root(self, root, rep):
        return Path(root)/f'r{rep}'

    def id_root(self, root=None):
        return Path(self.DEFAULT_ROOT if root is None else root)/ID_DIR

    def id_problem(self, root):
        """Why the evaluation of the identity may not write at this directory, or None: only rp_v1/id."""
        return None if Path(root).resolve() == self.id_root().resolve() else f'{root}: the identity is evaluated only into {self.id_root()}'

    def id_dir(self, root, pair):
        return Path(root)/core.short_name(pair)

    def r1_root(self, results=None):
        return Path(self.RESULTS if results is None else results)/R1_ROOT_NAME

    def d0_root(self, results=None):
        return Path(self.RESULTS if results is None else results)/D0_ROOT_NAME

    def a0_run_dir(self, pair, results=None):
        return self.d0_root(results)/'runs'/A0_ARM/pair

    def a0_field(self, pair, results=None):
        """d0_v1/runs/A0/<pair>/pg_weak__four__90s/final_native.npz: the brief's path."""
        return self.a0_run_dir(pair, results)/A0_TASK/A0_FILE

    def fallback_document(self, path=None):
        """rp_fallback_v1.json: {'pairs': {short name: the pins of its configuration C field}, 'evaluation_certificate': the configuration RW's evaluation recorded}."""
        return json.loads(Path(FALLBACK_FILE if path is None else path).read_text())

    def fallback_pins(self, path=None):
        """{short name of a pair: the pins of its configuration C field}."""
        return self.fallback_document(path)['pairs']

    def plan(self):
        return [c for rep in REPEATS for c in BATCHES[rep].plan()]


RP = Root()


# ---- the code is RW's ----------------------------------------------------------------------------------------------------------------------
def git(*args, cwd=None):
    done = subprocess.run(['git', *args], cwd=core.ROOT if cwd is None else cwd, capture_output=True, text=True)
    return done.returncode, done.stdout, done.stderr.strip()


def _in(path, prefixes):
    return any(path == x or path.startswith(x+'/') for x in prefixes)


def _blob(data):
    """git's blob hash of these bytes."""
    return hashlib.sha1(b'blob %d\0' % len(data)+data).hexdigest()


def frozen_identity(frozen=None, commit=None, cwd=None):
    """(record, problems): the frozen tree the units put first on PYTHONPATH, against `commit`'s blobs.  Every tracked file under FROZEN_PATHS and every tracked .py at the root: there, the same blob (symlinks: the same target); no .py under FROZEN_PATHS or at the root that the commit does not track (__pycache__ aside).  Reads the files' bytes only."""
    frozen = Path(core.FROZEN if frozen is None else frozen)
    commit = core.FROZEN_COMMIT if commit is None else commit
    rc, out, err = git('ls-tree', '-r', '-z', commit, '--', *FROZEN_PATHS, cwd=cwd)
    rc_root, out_root, err_root = git('ls-tree', '-z', commit, cwd=cwd)
    if rc or rc_root:
        return dict(frozen=str(frozen), commit=commit, error=err or err_root), [f'git ls-tree {commit}: {err or err_root}']
    tracked = {}
    for item in [x for x in out.split('\0') if x]+[x for x in out_root.split('\0') if x and x.split('\t', 1)[-1].endswith('.py') and x.split()[1] == 'blob']:
        meta, path = item.split('\t', 1)
        mode, _, blob = meta.split()
        tracked[path] = (mode, blob)
    mismatched, missing = [], []
    for path, (mode, blob) in sorted(tracked.items()):
        f = frozen/path
        if mode == '120000':
            if not f.is_symlink() or _blob(os.readlink(f).encode()) != blob:
                mismatched.append(path)
        elif not f.is_file() or f.is_symlink():
            missing.append(path)
        elif _blob(f.read_bytes()) != blob:
            mismatched.append(path)
    extra = sorted(str(f.relative_to(frozen)) for top in FROZEN_PATHS if (frozen/top).is_dir() for f in (frozen/top).rglob('*.py') if '__pycache__' not in f.parts and str(f.relative_to(frozen)) not in tracked)
    extra += sorted(f.name for f in frozen.glob('*.py') if f.name not in tracked)
    problems = []
    if missing or mismatched or extra:
        problems.append(f'the frozen tree {frozen} is not {commit[:7]}: missing {missing[:10]}, not the same blob {mismatched[:10]}, untracked .py {extra[:10]}')
    record = dict(frozen=str(frozen), commit=commit, paths=list(FROZEN_PATHS)+['*.py at the root'], n_files=len(tracked), missing=missing, mismatched=mismatched, untracked_py=extra, same=not problems,
                  rule="every tracked file of the commit under these paths has the commit's blob in the frozen tree; no untracked .py there")
    return record, problems


def identity(sha, base=BASE_COMMIT, cwd=None, frozen=None, frozen_commit=None):
    """What differs between RW's commit and `sha` (git diff --no-renames --name-status -z, so a name is compared as it is), and why it may not run: (record, problems).  Under IDENTITY_PATHS, and for .py files at the repository's root: a file of RW's commit that is changed, removed or of another type is a problem; an added Python file under a code directory (CODE_DIRS) or at the root that is not this package's is a problem; this package's own files must be there; other added files are listed.  Then the frozen tree (frozen_identity)."""
    problems = []
    rc, out, err = git('diff', '--no-renames', '--name-status', '-z', base, sha, cwd=cwd)
    if rc:
        return dict(base=base, sha=sha, error=err), [f'git diff {base} {sha}: {err}']
    tokens = out.split('\0')
    entries = [(tokens[k], tokens[k+1]) for k in range(0, len(tokens)-1, 2) if tokens[k]]
    scoped = [(status, path) for status, path in entries if _in(path, IDENTITY_PATHS) or ('/' not in path and path.endswith('.py'))]
    rc_files, files, err_files = git('ls-tree', '-r', '-z', '--name-only', base, '--', *IDENTITY_PATHS, cwd=cwd)
    n_base = len([f for f in files.split('\0') if f]) if rc_files == 0 else None
    if rc_files:
        problems.append(f'git ls-tree {base}: {err_files}')
    added_own, added_other = [], []
    for status, path in scoped:
        if status != 'A':
            problems.append(f'{status} {path}: a file of RW\'s commit {base[:7]} is not byte for byte the same at {sha[:7]}')
        elif path in OWN_FILES:
            added_own.append(path)
        elif path.endswith('.py') and ('/' not in path or path.split('/')[0] in CODE_DIRS):
            problems.append(f'A {path}: a Python file added under a code directory or at the root, not this package\'s (it could be imported or shadow a module)')
        else:
            added_other.append(path)
    missing = [x for x in OWN_FILES if x not in added_own]
    if missing:
        problems.append(f'{sha[:7]} does not hold {missing}: commit RP\'s files before the queue')
    frozen_record, frozen_problems = frozen_identity(frozen, frozen_commit, cwd)
    problems += frozen_problems
    record = dict(base=base, sha=sha, paths=list(IDENTITY_PATHS)+['*.py at the root'], n_files_at_base=n_base, n_files_changed=sum(status != 'A' for status, _ in scoped), entries=[f'{status}\t{path}' for status, path in scoped], added_own=added_own, added_other=added_other, frozen=frozen_record, same=not problems,
                  rule='every tracked file under these paths at RW\'s commit is byte for byte the same at this commit (git diff --name-status lists none of them); only this package\'s files, or files that are not Python modules, are added; the frozen tree is eb808be\'s')
    return record, problems


# ---- the fallback's first tier, from d0_v1's records ---------------------------------------------------------------------------------------------
def _json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except ValueError:
        return None


def fallback_status(pair, *, results=None, document=None, stage=None):
    """Configuration C's field of one pair as d0_v1's own records say: dict(qualified, why, metrics, ...).  Qualified: a pin; final_native.npz there, its sha256 the pin's and final_native.json's; u a float64 field of the image's shape (npy headers); the run's own certificate certified_feasible, lower bound > 0, no infeasible or uncertain cell, its state hash the pinned u, its whole configuration and its version those RW's evaluation recorded (the pins' evaluation_certificate); D0's evaluation passed with the hash of u matched and of the file the pinned one (final_npz_sha256, final_npz_sha256_matches), by the pinned evaluate.py (D0_EVALUATOR_SHA256), class ok, its metrics there.  The certificate is D0's run's own (the frozen configuration C, CPU FP64), not recomputed here.  Reads JSON records and npy headers and hashes the file; never the field's values, never a label; no certificate is computed (the lead, about 9 pm CST 11-05)."""
    import run_mc0 as run
    short = core.short_name(pair)
    document = RP.fallback_document() if document is None else document
    pins, expected = document.get('pairs') or {}, document.get('evaluation_certificate') or {}
    stage = core.STAGE if stage is None else Path(stage)
    pin, path = pins.get(short), RP.a0_field(pair, results)
    final, d0 = _json(path.parent/'final_native.json'), _json(RP.a0_run_dir(pair, results)/'metrics.json')
    why, sha, head = [], None, None
    if pin is None:
        why.append(f'{FALLBACK_FILE.name} has no pin')
        pin = {}
    if final is None:
        why.append('no final_native.json')
    if d0 is None:
        why.append('no D0 metrics.json')
    if not path.is_file():
        why.append('no final_native.npz')
    else:
        sha = core.sha_file(path)
        if sha != pin.get('file_sha256') or sha != (final or {}).get('sha256'):
            why.append('final_native.npz is not the pinned file or not the one final_native.json names')
        try:
            head, image = run.npz_header(path, 'u'), run.npz_header(stage/pair/'pair.npz', 'fixed')
            if tuple(head[0]) != tuple(image[0])+(3,) or head[1] != 'float64':
                why.append(f'u is {head[0]} {head[1]}, not a float64 field of the image grid {image[0]}')
        except (OSError, KeyError, ValueError) as error:
            why.append(f'npy header: {type(error).__name__}: {error}')
    cert = (final or {}).get('certificate') or {}
    config = cert.get('config') or {}
    if not (cert.get('status') == 'certified_feasible' and (cert.get('min_jd_lower_bound') or 0.) > 0 and cert.get('n_infeasible') == 0 and cert.get('n_uncertain') == 0):
        why.append(f"the run's certificate {cert.get('status')}, lower bound {cert.get('min_jd_lower_bound')}, infeasible {cert.get('n_infeasible')}, uncertain {cert.get('n_uncertain')}")
    if any(config.get(k) != v for k, v in CERTIFICATE_C.items()):
        why.append(f"the run's certificate is not configuration C: {dict((k, config.get(k)) for k in CERTIFICATE_C)}")
    if not expected.get('config') or config != expected.get('config') or cert.get('certificate_version') != expected.get('certificate_version'):
        why.append(f"the run's certificate configuration or version is not RW's evaluation's: {cert.get('certificate_version')}")
    if cert.get('state_hash') is None or cert.get('state_hash') != pin.get('u_sha256'):
        why.append("the run's certificate is not of the pinned u")
    d0_cert, d0_class = (d0 or {}).get('certificate') or {}, ((d0 or {}).get('classification') or {}).get('class')
    if d0 is not None and not (d0_cert.get('passed') is True and d0_cert.get('state_hash_matches_u') is True and d0_cert.get('state_hash') == pin.get('u_sha256') and d0.get('pair') == pair and d0.get('arm') == A0_ARM):
        why.append(f"D0's evaluation: passed {d0_cert.get('passed')}, state hash matches u {d0_cert.get('state_hash_matches_u')}, pair and arm {d0.get('pair') == pair and d0.get('arm') == A0_ARM}")
    if d0 is not None and not (d0_cert.get('final_npz_sha256') == pin.get('file_sha256') and d0_cert.get('final_npz_sha256_matches') is True):
        why.append("D0's evaluation is not of the pinned file (final_npz_sha256)")
    evaluator = ((d0 or {}).get('evaluator') or {}).get('sha256')
    if d0 is not None and not (evaluator == D0_EVALUATOR_SHA256 == (pin.get('d0_evaluation') or {}).get('evaluator_sha256')):
        why.append(f"D0's evaluator {evaluator} is not the pinned evaluate.py")
    if d0 is not None and d0_class != D0_CLASS_OK:
        why.append(f"D0's class {d0_class}")
    metrics = (d0 or {}).get('metrics') or {}
    if d0 is not None and any(metrics.get(m) is None for m in ('seg35', 'seg4', 'sdlogj_official')):
        why.append("D0's metrics are not there")
    return dict(pair=short, field=str(path), qualified=not why, why='; '.join(why) or None, file_sha256=sha, u_header=None if head is None else [list(head[0]), head[1]], run_id=(d0 or {}).get('run_id'), d0_class=d0_class, d0_passed=d0_cert.get('passed'),
                run_certificate=dict(status=cert.get('status'), min_jd_lower_bound=cert.get('min_jd_lower_bound'), n_infeasible=cert.get('n_infeasible'), n_uncertain=cert.get('n_uncertain'), config={k: config.get(k) for k in CERTIFICATE_C}),
                metrics={m: metrics.get(m) for m in FALLBACK_MEASURES}, evaluator=(d0 or {}).get('evaluator'), source="d0_v1 records; the certificate is D0's run's own (frozen configuration C, CPU FP64), not computed again")
