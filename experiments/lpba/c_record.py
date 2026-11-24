"""LP-T0 item 3 (runs/20261119-lpba/任务书_T0_名单臂表与c.md): c and W_dev of the LPBA40 second test set, from the development set, written once.

Preregistration docs/bayesreg/notes/LPBA40第二测试集预登记_20261119.md (final, revision 3, sha256 06121278...) §2.2 (ruling B1), §8.1 items 5-6, §12 item 3.  The definitions,
word for word:
  c      = the median (np.median) over the 20 subjects of the development L2R validation stage (data/three_d/stage_L2R_val_v1) of each subject's
           np.percentile(v[v > 0], 99.5) (linear), v the subject's fixed or moving array of the stage pair.npz.  Only these two members are read, no seg key.
  W_dev  = the median of the 20 means of v over the voxels where the subject's Atropos WM posterior > .5; the Atropos posteriors are the tissue layer the method uses on
           the development set (HE's atropos_root: ATROPOS_ROOT).
  A subject in two pairs: the two arrays must be bit for bit the same, else the run stops (a tool or data error).
The record: per subject the p99.5 and the WM mean, c and W_dev, each as decimal (repr) and float.hex; the path and sha256 of every file read; the code commit.  No truth.

What is read, and nothing else (the spy below stops any other open):
  configs/bayesreg/dev_subjects.json        the allowlist source stage_L2R_val_v1: its 20 subjects must be DEV_SUBJECTS; its sha256 of the stage's selection.json
  <stage>/selection.json                    sha256 = the allowlist source's; its 19 pairs = DEV_PAIRS in order; the subjects from its fixed / moving entries (names only)
  <stage>/<pair>/pair.npz (19)              np.load (allow_pickle False), then npz['fixed'] and npz['moving'] only (no seg key, not initial_u; manifest.json is not read);
                                            its file sha256
  <atropos>/<subject>/provenance.json (20)  segmenter atropos, layer tissue, no label seen; class_names with WM at wm_index; image_source.stage stage_L2R_val_v1 and
                                            image_source.array_sha256_float64 = the sha256 of v (the posterior is bound to this very array); grid shape = v's
  <atropos>/<subject>/posteriors0k.nii.gz   only the WM file, posterior_files[wm_index] = posteriors{wm_index+1:02d}.nii.gz (bayesreg.cli.run_pair.load_tissue_layer's rule),
                                            read with jd_curl.three_d.nifti.load_scalar (load_tissue_layer's reader); its sha256 = provenance output_sha256; on v's grid
  the preregistration                       its sha256 = PREREG_SHA256
ATROPOS_ROOT is experiments/mc0/mc0p_core.py SEG_ROOT, which experiments/he/run_hx.py default_paths() gives as atropos_root (HE's development context; a test compares).  The
Atropos root also holds dice_vs_truth_seg4.json and each subject labels.nii.gz: never opened (the spy refuses those names).  bayesreg.io.guard is not called: it would read the
sealed lists; the 20 subjects are code constants checked against the allowlist source instead.

The spy (as experiments/holdout/hp_core.Spy, an allow list as the preregistration §8.5 item 7 asks): an audit hook on open, os.listdir and os.scandir, and a wrapper of
numpy's NpzFile.__getitem__.  An open is allowed when it is a code read (.py, .pyc, .so, .pth) of the code tree or the Python environment, one of the files above (judged
resolved), the lock, a file in the output directory, or a read of the system (/usr, /lib, /proc, /dev, ...); any name containing a FORBIDDEN_NAME_PARTS word (label, seg4,
seg35, aligned_seg, dice, truth, metric, evaluation) is refused before that, and so is anything else under the data roots, the seq_vp_3d tree, D: or the results root.  A
pair.npz member other than fixed and moving is refused.  A refusal raises Breach inside the open; after the lock it stops the run (c_record_stopped.json).

Once, in its own root C_ROOT = /home/rmcfz/seq_vp_tag_results/lpba40_c_record_v1 (the lead's ruling, decision_log 2026-11-19 ~1:01 PM CST (2): not the formal run root
lpba40_test_v1, which is the preregistered entries'): the lock LOCK = C_ROOT/c_record.lock (a fixed path, O_EXCL, never edited or deleted) is taken after every check and
before the first array; the record directory OUT = C_ROOT/record must not exist before and is made after the lock.  Failures before the lock are refusals (exit 8, nothing
written; a spy breach or any other exception there too, LP-T0 review minor 1, diff review minor 3); after the lock every exception stops the run (LP-T0 review important 1): c_record_stopped.json in OUT, or beside
the lock when OUT could not be made, exit 9 (a BaseException such as KeyboardInterrupt is re-raised after the record is written); kept, not rerun (CLAUDE.md hard rule 4).
The formal run needs a clean code tree (git status --porcelain --untracked-files=no empty); the record names its HEAD.  The record is then committed to git by the lead
(§2.2: "提交进 git"; its sha256 into the decision log; the data preparation pins it).  The entry refuses to run (exit 8) unless Python writes no bytecode
(sys.flags.dont_write_bytecode: python -B or PYTHONDONTWRITEBYTECODE; review minor 2); the module also sets sys.dont_write_bytecode before any other import.

    python -B experiments/lpba/c_record.py --dry-run [--out DIR]
        the allowlist, selection.json and the preregistration only; the planned files listed and their existence checked by stat (no array, no provenance opened); the
        lock and OUT checked to be absent.  DIR (new or empty, not under the results root or a data root) receives dry_run.json; without it the summary is printed only.
    python -B experiments/lpba/c_record.py --run
        the formal run (once): OUT/c_record.json.
"""
import sys
sys.dont_write_bytecode = True                  # review minor 2: before any other import (the code tree gets no __pycache__ from this tool)

import argparse  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import nibabel  # noqa: E402
from jd_curl.three_d.nifti import load_scalar  # noqa: E402

SCHEMA = 'bayesreg.lpba40.c_record.v1'
DRY_SCHEMA = 'bayesreg.lpba40.c_record_dry_run.v1'
SCRIPT_REL = 'experiments/lpba/c_record.py'
PREREG_REL = 'docs/bayesreg/notes/LPBA40第二测试集预登记_20261119.md'
PREREG_SHA256 = '06121278d8b229401afea7cb86c841f80ba02f23bc40f8c5f5594c7da8d95643'
MAIN_REPO = Path('/home/rmcfz/code/python_projects/seq_vp_tag')           # data/ is an untracked link of the main checkout (a worktree has none)
DEV_STAGE_NAME = 'stage_L2R_val_v1'
DEV_STAGE = MAIN_REPO/'data'/'three_d'/DEV_STAGE_NAME
ALLOWLIST_REL = 'configs/bayesreg/dev_subjects.json'
SELECTION_REL = f'data/three_d/{DEV_STAGE_NAME}/selection.json'           # the allowlist source's path of the stage
ATROPOS_ROOT = Path('/home/rmcfz/seq_vp_tag_results/seg/atropos_v1')     # = experiments/mc0/mc0p_core.SEG_ROOT = run_hx.default_paths()['atropos_root'] (HE, development)
ATROPOS_ORIGIN = 'experiments/he/run_hx.py default_paths()["atropos_root"] = experiments/mc0/mc0p_core.py SEG_ROOT (HE development context)'
DEV_SUBJECTS = tuple(f'OASIS_OAS1_{n:04d}_MR1' for n in range(438, 458))
DEV_PAIRS = tuple(f'{DEV_SUBJECTS[i]}__{DEV_SUBJECTS[i+1]}' for i in range(19))
MEMBERS = ('fixed', 'moving')
PAIR_FILE = 'pair.npz'
PROVENANCE = 'provenance.json'
WM = 'WM'
WM_LEVEL = .5
PERCENTILE = 99.5
RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
C_ROOT = RESULTS/'lpba40_c_record_v1'           # the lead's ruling of 2026-11-19 ~1:01 PM CST (2): its own root; lpba40_test_v1 is left to the preregistered entries
OUT = C_ROOT/'record'
LOCK = C_ROOT/'c_record.lock'
RECORD_NAME, STOPPED_NAME, DRY_NAME = 'c_record.json', 'c_record_stopped.json', 'dry_run.json'
FORBIDDEN_NAME_PARTS = ('aligned_seg', 'seg4', 'seg35', 'label', 'dice', 'truth', 'metric', 'evaluation')
FORBIDDEN_ROOTS = (MAIN_REPO/'data', Path('/mnt/d/data'), Path('/mnt/d'), Path('/home/rmcfz/code/python_projects/seq_vp_3d'), RESULTS)
SYSTEM_READ = ('/usr', '/lib', '/lib64', '/bin', '/sbin', '/etc', '/proc', '/sys', '/dev', '/run')
CODE_SUFFIXES = ('.py', '.pyc', '.so', '.pth')
EXIT = dict(ok=0, usage=2, refused=8, stopped=9)


class Refused(Exception):
    """Before the lock: nothing of a subject read, nothing written under the results root."""

    def __init__(self, stage, message):
        super().__init__(f'{stage}: {message}')
        self.stage = stage


class Stopped(Exception):
    """After the lock: the run stops (c_record_stopped.json)."""


class Breach(AssertionError):
    """The spy refused an open, a listing or a pair.npz member."""


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(a):
    """he_core.array_sha256 (the preregistration P7's convention): sha256 of the C-order bytes."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def number(x):
    x = float(x)
    return dict(decimal=repr(x), hex=x.hex())


def _hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


# ---- the spy ---------------------------------------------------------------------------------------------------------------------------------------
_ACTIVE = None
_INSTALLED = False
_LOCAL = threading.local()


def _hook(event, args):
    spy = _ACTIVE
    if spy is None or event not in ('open', 'os.listdir', 'os.scandir') or getattr(_LOCAL, 'busy', False):
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


class Spy:
    """The allow-list spy of the module docstring for one run."""

    def __init__(self, *, read_exact=(), write_inside=(), write_exact=(), code_roots=(REPO,), forbidden_roots=FORBIDDEN_ROOTS, members=MEMBERS):
        r = lambda p: Path(p).resolve()     # noqa: E731
        self.read_exact = {r(p) for p in read_exact}
        self.write_inside = [r(p) for p in write_inside]
        self.write_exact = {r(p) for p in write_exact}
        self.code_roots = sorted({r(p) for p in code_roots})
        self.forbidden_roots = sorted({x for p in forbidden_roots for x in (Path(os.path.abspath(p)), r(p))})
        self.system_roots = sorted({r(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, *SYSTEM_READ)})
        self.members_allowed = tuple(members)
        self.events, self.breaches, self.members = [], [], []
        self._real_getitem = None

    def allow_read(self, paths):
        self.read_exact |= {Path(p).resolve() for p in paths}

    def classify(self, path, write):
        """('allowed' | 'forbidden' | 'unlisted', why) of one open."""
        raw, resolved = _forms(path)
        if raw is None:
            return 'allowed', 'a descriptor'
        if not write and resolved.suffix in CODE_SUFFIXES and (_under(resolved, self.code_roots) or _under(resolved, self.system_roots)):
            return 'allowed', 'code'
        for form in (raw, resolved):
            low = form.name.lower()
            if any(part in low for part in FORBIDDEN_NAME_PARTS):
                return 'forbidden', f'a label or evaluation file by its name ({form.name})'
        if resolved in self.write_exact:
            return 'allowed', 'the lock'
        if _under(resolved, self.write_inside):
            return 'allowed', 'the output directory'
        if not write and resolved in self.read_exact:
            return 'allowed', 'an input, allowed one by one'
        if _under(raw, self.forbidden_roots) or _under(resolved, self.forbidden_roots):
            return 'forbidden', 'a data root, D:, the seq_vp_3d tree or the results root, not allowed one by one'
        if not write and _under(resolved, self.system_roots):
            return 'allowed', 'the Python environment or the system'
        if write and resolved == Path('/dev/null'):
            return 'allowed', '/dev/null'
        return 'unlisted', 'not in the allow list'

    def see_open(self, args):
        path = args[0] if args else None
        write = _is_write(args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else 0)
        raw, resolved = _forms(path)
        if raw is None:
            return
        cls, why = self.classify(path, write)
        self.events.append(dict(path=str(raw), resolved=str(resolved), write=write, cls=cls))
        if cls != 'allowed':
            self.breaches.append(dict(path=str(raw), resolved=str(resolved), write=write, cls=cls, why=why))
            raise Breach(f'c_record spy: {cls} open of {raw} ({why})')

    def see_listing(self, path):
        raw, resolved = _forms(path)
        if raw is None or _under(resolved, self.write_inside):
            return
        if _under(raw, self.forbidden_roots) or _under(resolved, self.forbidden_roots):
            self.breaches.append(dict(path=str(raw), resolved=str(resolved), write=False, cls='forbidden', why='a listing of a watched root', event='listing'))
            raise Breach(f'c_record spy: listing of {raw} (a watched root)')

    def see_member(self, npz, key):
        source = getattr(getattr(npz, 'fid', None), 'name', None)
        self.members.append(dict(file=str(source), member=str(key)))
        if key not in self.members_allowed:
            self.breaches.append(dict(path=str(source), member=str(key), cls='forbidden', why=f'a pair.npz member other than {list(self.members_allowed)}'))
            raise Breach(f'c_record spy: the member {key!r} of {source} was asked')

    def install(self):
        global _ACTIVE, _INSTALLED
        if _ACTIVE is not None:
            raise RuntimeError('another c_record spy is active in this process')
        if not _INSTALLED:
            sys.addaudithook(_hook)
            _INSTALLED = True
        from numpy.lib.npyio import NpzFile
        spy, real = self, NpzFile.__getitem__

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

    def opened(self, path):
        target = str(Path(path).resolve())
        return sum(1 for e in self.events if e['resolved'] == target and not e['write'])

    def record(self, *, expect=()):
        """passed: no breach, every expected input read at least once, and opens were seen at all."""
        expected = {str(Path(p).resolve()): self.opened(p) for p in expect}
        positive = bool(self.events) and all(n >= 1 for n in expected.values())
        data_reads = sorted({e['resolved'] for e in self.events if not e['write'] and e['cls'] == 'allowed' and Path(e['resolved']) in self.read_exact})
        return dict(spy='experiments/lpba/c_record.Spy (audit hook: open, os.listdir, os.scandir; NpzFile.__getitem__)', n_events=len(self.events), events=self.events,
                    data_reads=data_reads, members=self.members, breaches=list(self.breaches), expected_reads=expected, positive_condition=positive,
                    passed=positive and not self.breaches, forbidden_name_parts=list(FORBIDDEN_NAME_PARTS), members_allowed=list(self.members_allowed),
                    forbidden_roots=list(map(str, self.forbidden_roots)), rule='every open judged and recorded; a breach stops the run; every expected input read')


@contextmanager
def spying(spy):
    spy.install()
    try:
        yield spy
    finally:
        spy.remove()


# ---- code, lock, writes ----------------------------------------------------------------------------------------------------------------------------
def git(repo, *args):
    done = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, timeout=60)
    if done.returncode:
        raise Refused('code', f'git {" ".join(args)}: {done.stderr.strip()[:300]}')
    return done.stdout


def code_record(repo):
    """The code tree: HEAD, whether the tracked files are clean, this script's sha256 (read as code)."""
    status = git(repo, 'status', '--porcelain', '--untracked-files=no')
    return dict(repo=str(repo), commit=git(repo, 'rev-parse', 'HEAD').strip(), clean=status == '', status=status[:2000], script=SCRIPT_REL,
                script_sha256=file_sha256(Path(repo)/SCRIPT_REL) if (Path(repo)/SCRIPT_REL).is_file() else None)


def take_lock(path, line):
    """O_EXCL, written once, never edited or deleted; Refused('once') if it is there."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise Refused('once', f'{path} is there: the record is made once; nothing read') from None
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        stream.write(json.dumps(line, ensure_ascii=False)+'\n')
    return line


def write_once(path, document):
    with open(path, 'x', encoding='utf-8') as stream:
        stream.write(json.dumps(document, ensure_ascii=False, indent=1)+'\n')
    return path


def write_stopped(out, lock, document):
    """c_record_stopped.json in OUT, or beside the lock when OUT could not be made (review important 1).  Returns its path."""
    target = (Path(out) if Path(out).is_dir() else Path(lock).parent)/STOPPED_NAME
    return write_once(target, document)


def bytecode_problem(flags=None):
    """Review minor 2: the entry runs only with sys.flags.dont_write_bytecode (python -B, or PYTHONDONTWRITEBYTECODE)."""
    flags = sys.flags if flags is None else flags
    return None if flags.dont_write_bytecode else 'Python writes bytecode here: run with python -B (or PYTHONDONTWRITEBYTECODE=1); nothing read'


def sandbox_problem(*paths):
    """While pytest runs, the real results root is never written (the tests pass their own roots)."""
    if os.environ.get('PYTEST_CURRENT_TEST'):
        bad = [str(p) for p in paths if _under(Path(os.path.abspath(p)), [RESULTS])]
        if bad:
            return f'a test may not write the real results root: {bad}'
    return None


# ---- the inputs ------------------------------------------------------------------------------------------------------------------------------------
def admission(allowlist_path, stage):
    """The allowlist source stage_L2R_val_v1 and the stage's selection.json: ([(pair, fixed subject, moving subject)], record).  Refused otherwise."""
    allowlist_path, stage = Path(allowlist_path), Path(stage)
    raw = allowlist_path.read_bytes()
    document = json.loads(raw)
    sources = [s for s in document.get('sources', []) if isinstance(s, dict) and s.get('name') == DEV_STAGE_NAME]
    if len(sources) != 1:
        raise Refused('admission', f'{allowlist_path}: one source {DEV_STAGE_NAME} required, found {len(sources)}')
    source = sources[0]
    if source.get('subjects') != list(DEV_SUBJECTS) or source.get('path') != SELECTION_REL or not _hex64(source.get('sha256')) or source.get('dataset') != 'OASIS':
        raise Refused('admission', f'{allowlist_path}: the source {DEV_STAGE_NAME} is not the 20 development subjects of {SELECTION_REL} with a sha256')
    selection_path = stage/'selection.json'
    selection_raw = selection_path.read_bytes()
    if hashlib.sha256(selection_raw).hexdigest() != source['sha256']:
        raise Refused('admission', f'{selection_path}: sha256 is not the allowlist source\'s {source["sha256"]}')
    rows = []
    for k, row in enumerate(json.loads(selection_raw)['pairs']):
        fixed, moving = Path(row['fixed']).name, Path(row['moving']).name
        rows.append((row['pair'], fixed, moving))
    if [r[0] for r in rows] != list(DEV_PAIRS) or any(r[0] != f'{r[1]}__{r[2]}' for r in rows):
        raise Refused('admission', f'{selection_path}: the pairs are not the 19 development pairs {DEV_PAIRS[0]} ... {DEV_PAIRS[-1]} in order')
    if sorted({s for r in rows for s in r[1:]}) != list(DEV_SUBJECTS):
        raise Refused('admission', f'{selection_path}: the subjects are not the 20 development subjects')
    return rows, dict(allowlist=dict(path=str(allowlist_path), sha256=hashlib.sha256(raw).hexdigest(), source=DEV_STAGE_NAME, selection_sha256_pinned=source['sha256']),
                      selection=dict(path=str(selection_path), resolved=str(selection_path.resolve()), sha256=hashlib.sha256(selection_raw).hexdigest(), pairs=len(rows)))


def plan(rows, stage, atropos_root):
    """The data files the run will read: 19 pair.npz, 20 provenance.json; the WM posterior of each subject is named by its provenance (one of posteriors01-03)."""
    stage, atropos_root = Path(stage), Path(atropos_root)
    pairs = [dict(pair=p, path=str(stage/p/PAIR_FILE)) for p, _, _ in rows]
    subjects = [dict(subject=s, provenance=str(atropos_root/s/PROVENANCE), posterior_candidates=[str(atropos_root/s/f'posteriors{k:02d}.nii.gz') for k in (1, 2, 3)])
                for s in DEV_SUBJECTS]
    return dict(pairs=pairs, subjects=subjects)


def read_pair(path):
    """{side: array} of the two members fixed and moving (np.load, then by key), each float64, 3-D, finite."""
    out = {}
    with np.load(path, allow_pickle=False) as npz:
        for side in MEMBERS:
            a = npz[side]
            if a.dtype != np.float64 or a.ndim != 3 or not np.isfinite(a).all():
                raise Stopped(f'{path}: {side} is not a finite float64 3-D array ({a.dtype}, {a.shape})')
            out[side] = np.ascontiguousarray(a)
    return out


def read_wm(atropos_root, subject, v, spy=None):
    """(WM posterior (nz, ny, nx) float64, record) of one subject, bound to the array v (module docstring).  Stopped otherwise."""
    directory = Path(atropos_root)/subject
    path = directory/PROVENANCE
    raw = path.read_bytes()
    prov = json.loads(raw)
    if not isinstance(prov, dict):
        raise Stopped(f'{path}: a JSON object required, got {type(prov).__name__}')
    names = prov.get('class_names')
    problems = []
    if prov.get('subject') != subject or prov.get('segmenter_id') != 'atropos' or prov.get('layer') != 'tissue':
        problems.append(f"subject {prov.get('subject')!r}, segmenter {prov.get('segmenter_id')!r}, layer {prov.get('layer')!r}")
    if prov.get('test_labels_seen') is not False or prov.get('trained_with_labels') is not False or prov.get('oracle') is not False:
        problems.append('test_labels_seen, trained_with_labels and oracle must be false')
    if not isinstance(names, list) or names.count(WM) != 1 or prov.get('wm_index') != names.index(WM):
        problems.append(f"class_names {names!r} with one {WM} at wm_index {prov.get('wm_index')!r}")
    if problems:
        raise Stopped(f'{path}: {problems}')
    k = names.index(WM)
    name = f'posteriors{k+1:02d}.nii.gz'
    files, outputs, source, grid = prov.get('posterior_files'), prov.get('output_sha256'), prov.get('image_source'), prov.get('grid')
    if not isinstance(files, list) or len(files) != len(names) or files[k] != name or prov.get('posterior_file_index_base') != 1 \
            or not isinstance(outputs, dict) or not _hex64(outputs.get(name)):
        raise Stopped(f'{path}: posterior_files[{k}] must be {name} (index base 1) with its sha256 in output_sha256')
    sha_v = array_sha256(v)
    if not isinstance(source, dict) or source.get('stage') != DEV_STAGE_NAME or source.get('array_sha256_float64') != sha_v:
        raise Stopped(f'{path}: image_source must be {DEV_STAGE_NAME} with array_sha256_float64 = the stage array\'s {sha_v}')
    if not isinstance(grid, dict) or grid.get('shape_zyx') != list(v.shape):
        raise Stopped(f'{path}: grid shape_zyx {None if not isinstance(grid, dict) else grid.get("shape_zyx")} is not the array\'s {list(v.shape)}')
    posterior = directory/name
    if spy is not None:
        spy.allow_read([posterior])
    volume = load_scalar(posterior)
    meta = volume.metadata
    if meta['sha256'] != outputs[name]:
        raise Stopped(f'{posterior}: sha256 {meta["sha256"]} is not the provenance\'s {outputs[name]}')
    if volume.values.shape != v.shape or meta['spacing_xyz'] != list(grid.get('spacing_xyz', [])) \
            or [float(a[0]) for a in volume.grid.axes] != [float(x) for x in grid.get('origin_xyz', [])]:
        raise Stopped(f'{posterior}: not on the array\'s grid (shape {volume.values.shape} / {v.shape}, spacing {meta["spacing_xyz"]}, origin '
                      f'{[float(a[0]) for a in volume.grid.axes]} / {grid.get("origin_xyz")})')
    record = dict(provenance=dict(path=str(path), resolved=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest()),
                  posterior=dict(path=str(posterior), resolved=meta['source'], sha256=meta['sha256'], file=name), wm_index=k, class_names=list(names),
                  posterior_min=meta['min'], posterior_max=meta['max'], bound_to_array_sha256=sha_v)
    return np.asarray(volume.values, dtype=np.float64), record


def subject_values(v, wm):
    """§2.2: (p99.5, number of voxels > 0, WM mean, number of WM voxels) of one subject, the definitions word for word."""
    positive = v[v > 0]
    if positive.size == 0:
        raise Stopped('no voxel > 0')
    p = np.percentile(v[v > 0], 99.5)
    mask = wm > WM_LEVEL
    if not mask.any():
        raise Stopped(f'no voxel with the WM posterior > {WM_LEVEL}')
    return float(p), int(positive.size), float(v[mask].mean()), int(mask.sum())


def c_and_w(p_values, wm_means):
    """§2.2: c = np.median of the 20 p99.5, W_dev = np.median of the 20 WM means (in DEV_SUBJECTS order)."""
    return float(np.median(np.asarray(p_values, dtype=np.float64))), float(np.median(np.asarray(wm_means, dtype=np.float64)))


def compute(rows, stage, atropos_root, spy=None):
    """Every subject at its first appearance (pair order of selection.json; fixed before moving); at a second appearance the array must be bit for bit the first one's."""
    stage = Path(stage)
    seen, per_subject, files = {}, {}, []
    for pair, fixed, moving in rows:
        path = stage/pair/PAIR_FILE
        arrays = read_pair(path)
        files.append(dict(role=PAIR_FILE, pair=pair, path=str(path), resolved=str(path.resolve()), sha256=file_sha256(path), members_read=list(MEMBERS)))
        for side, subject in zip(MEMBERS, (fixed, moving)):
            v = arrays[side]
            digest = array_sha256(v)
            if subject in seen:
                first = seen[subject]
                if (first['dtype'], first['shape'], first['sha256']) != (str(v.dtype), list(v.shape), digest):
                    raise Stopped(f'{subject}: {pair}:{side} differs from {first["at"]} (sha256 {digest} / {first["sha256"]}): the two arrays must be bit for bit the same')
                per_subject[subject]['occurrences'].append(f'{pair}:{side}')
                continue
            seen[subject] = dict(dtype=str(v.dtype), shape=list(v.shape), sha256=digest, at=f'{pair}:{side}')
            wm, wm_record = read_wm(atropos_root, subject, v, spy)
            p, n_positive, mean, n_wm = subject_values(v, wm)
            del wm
            files += [dict(role='atropos provenance', subject=subject, **{k: wm_record['provenance'][k] for k in ('path', 'resolved', 'sha256')}),
                      dict(role='atropos WM posterior', subject=subject, **{k: wm_record['posterior'][k] for k in ('path', 'resolved', 'sha256')})]
            per_subject[subject] = dict(subject=subject, occurrences=[f'{pair}:{side}'], array=dict(sha256=digest, dtype=str(v.dtype), shape_zyx=list(v.shape)),
                                        n_positive=n_positive, p99_5=number(p), wm=dict(wm_record, n_voxels=n_wm, mean=number(mean)))
        del arrays
    if sorted(per_subject) != list(DEV_SUBJECTS):
        raise Stopped(f'subjects read {sorted(per_subject)} are not the 20 development subjects')
    subjects = [per_subject[s] for s in DEV_SUBJECTS]
    c, w = c_and_w([float.fromhex(s['p99_5']['hex']) for s in subjects], [float.fromhex(s['wm']['mean']['hex']) for s in subjects])
    return subjects, files, number(c), number(w)


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, nibabel=nibabel.__version__)


DEFINITIONS = dict(
    c='np.median over the 20 subjects of np.percentile(v[v > 0], 99.5) (linear), v = the subject\'s fixed or moving array of the stage pair.npz (§2.2)',
    W_dev='np.median over the 20 subjects of the mean of v over the voxels where the Atropos WM posterior > .5 (§2.2)',
    twice='a subject in two pairs: both arrays bit for bit the same, else stop (§2.2, review minor 9)')


# ---- the two entries -------------------------------------------------------------------------------------------------------------------------------
def dry_run(*, repo=REPO, stage=DEV_STAGE, allowlist=None, atropos_root=ATROPOS_ROOT, prereg=None, prereg_sha256=PREREG_SHA256, out=None, lock=LOCK, record_dir=OUT):
    """The dry run of the module docstring.  Returns the document (and writes `record_dir`-independent DIR/dry_run.json when `out` is given)."""
    allowlist = Path(repo)/ALLOWLIST_REL if allowlist is None else Path(allowlist)
    prereg = Path(repo)/PREREG_REL if prereg is None else Path(prereg)
    if out is not None:
        out = Path(out)
        if any(_under(Path(os.path.abspath(out)), [Path(os.path.abspath(r))]) or _under(out.resolve(), [Path(r).resolve()]) for r in FORBIDDEN_ROOTS) \
                or (out.exists() and any(out.iterdir())):
            raise Refused('out', f'{out}: new or empty, not under the results root, D: or a data root')
    code = code_record(repo)
    spy = Spy(read_exact=[allowlist, Path(stage)/'selection.json', prereg], write_inside=[out] if out is not None else [])
    with spying(spy):
        try:
            prereg_sha = file_sha256(prereg)
            rows, admitted = admission(allowlist, stage)
            planned = plan(rows, stage, atropos_root)
        except Refused:
            raise
        except Breach as exc:
            raise Refused('spy', f'the dry run: {exc}') from exc
        except Exception as exc:                    # diff review minor 3: as in run, a failure of these reads is a refusal (exit 8)
            raise Refused('input', f'the dry run: {type(exc).__name__}: {exc}') from exc
    exists = dict(pairs=sum(Path(p['path']).is_file() for p in planned['pairs']),
                  provenance=sum(Path(s['provenance']).is_file() for s in planned['subjects']),
                  subjects_with_three_posteriors=sum(all(Path(c).is_file() for c in s['posterior_candidates']) for s in planned['subjects']))
    n_data = len(planned['pairs'])+2*len(planned['subjects'])
    document = dict(schema=DRY_SCHEMA, utc=utc(), code=code, prereg=dict(path=str(prereg), sha256=prereg_sha, pinned=prereg_sha256, equal=prereg_sha == prereg_sha256),
                    admission=admitted, atropos_root=str(atropos_root), atropos_origin=ATROPOS_ORIGIN, plan=planned, exists=exists,
                    will_read=dict(pair_npz=len(planned['pairs']), provenance=len(planned['subjects']), wm_posteriors=len(planned['subjects']), data_files=n_data,
                                   repository_files=[ALLOWLIST_REL, PREREG_REL], stage_files=['selection.json'], members_per_pair=list(MEMBERS)),
                    lock=dict(path=str(lock), exists=Path(lock).exists()), out=dict(path=str(record_dir), exists=Path(record_dir).exists()),
                    spy=spy.record(expect=[allowlist, Path(stage)/'selection.json', prereg]))
    document['ready'] = bool(document['prereg']['equal'] and exists['pairs'] == 19 and exists['provenance'] == 20 and exists['subjects_with_three_posteriors'] == 20
                             and not document['lock']['exists'] and not document['out']['exists'] and document['spy']['passed'])
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        write_once(out/DRY_NAME, document)
    return document


def run(*, repo=REPO, stage=DEV_STAGE, allowlist=None, atropos_root=ATROPOS_ROOT, prereg=None, prereg_sha256=PREREG_SHA256, out=OUT, lock=LOCK, require_clean=True):
    """The formal run (once).  Returns the record; Refused before the lock, Stopped after it (c_record_stopped.json written)."""
    allowlist = Path(repo)/ALLOWLIST_REL if allowlist is None else Path(allowlist)
    prereg = Path(repo)/PREREG_REL if prereg is None else Path(prereg)
    out, lock = Path(out), Path(lock)
    problem = sandbox_problem(out, lock)
    if problem:
        raise Refused('sandbox', problem)
    code = code_record(repo)
    if require_clean and not code['clean']:
        raise Refused('code', f'the code tree {repo} is not clean: {code["status"][:300]}')
    if lock.exists() or out.exists():
        raise Refused('once', f'{lock} or {out} is there: the record is made once; nothing read')
    spy = Spy(read_exact=[allowlist, Path(stage)/'selection.json', prereg], write_inside=[out], write_exact=[lock, lock.parent/STOPPED_NAME])
    started = utc()
    with spying(spy):
        try:
            prereg_sha = file_sha256(prereg)
            if prereg_sha != prereg_sha256:
                raise Refused('prereg', f'{prereg}: sha256 {prereg_sha} is not the pinned {prereg_sha256}')
            rows, admitted = admission(allowlist, stage)
            planned = plan(rows, stage, atropos_root)
            spy.allow_read([p['path'] for p in planned['pairs']]+[s['provenance'] for s in planned['subjects']])
            line = dict(tool=SCRIPT_REL, schema=SCHEMA, utc=utc(), commit=code['commit'], prereg_sha256=prereg_sha, pid=os.getpid(), out=str(out))
        except Refused:
            raise
        except Breach as exc:                       # review minor 1: before the lock a breach is a refusal (nothing of a subject read, nothing written)
            raise Refused('spy', f'before the lock: {exc}') from exc
        except Exception as exc:                    # diff review minor 3: any other failure before the lock is a refusal too (exit 8)
            raise Refused('input', f'before the lock: {type(exc).__name__}: {exc}') from exc
        try:
            take_lock(lock, line)
        except Refused:                             # the lock is there (another run): nothing of this run is written
            raise
        except Exception as exc:
            if not lock.exists():                   # nothing was made: a refusal (diff review minor 3)
                raise Refused('lock', f'the lock could not be taken: {type(exc).__name__}: {exc}') from exc
            write_stopped(out, lock, dict(schema=SCHEMA, stopped=f'{type(exc).__name__}: {exc}', kind=type(exc).__name__, utc_start=started, utc_end=utc(), code=code,
                                          lock=dict(path=str(lock), line=line), out=str(out), out_made=False, traceback=traceback.format_exc()[-4000:], spy=spy.record()))
            raise Stopped(f'the lock was made but not written whole: {type(exc).__name__}: {exc}') from exc
        record = None
        try:                                        # review important 1: after the lock every exception leaves a stopped record
            out.mkdir(parents=True)
            subjects, files, c, w = compute(rows, stage, atropos_root, spy)
            expect = [allowlist, Path(stage)/'selection.json', prereg]+[f['path'] for f in files]
            record = dict(schema=SCHEMA, prereg=dict(path=PREREG_REL, sha256=prereg_sha, sections='§2.2, §8.1 items 5-6, §12 item 3'), definitions=DEFINITIONS,
                          c=c, W_dev=w, n_subjects=len(subjects), subjects=subjects,
                          inputs=dict(stage=dict(path=str(stage), resolved=str(Path(stage).resolve()), name=DEV_STAGE_NAME), atropos_root=str(atropos_root),
                                      atropos_origin=ATROPOS_ORIGIN, admission=admitted, members_read=list(MEMBERS), manifest_read=False),
                          files_read=[dict(role='allowlist', path=str(allowlist), resolved=str(allowlist.resolve()), sha256=admitted['allowlist']['sha256']),
                                      dict(role='selection.json', path=admitted['selection']['path'], resolved=admitted['selection']['resolved'],
                                           sha256=admitted['selection']['sha256']),
                                      dict(role='preregistration', path=str(prereg), resolved=str(prereg.resolve()), sha256=prereg_sha)]+files,
                          code=code, environment=environment(), lock=dict(path=str(lock), line=line), utc_start=started, utc_end=utc(),
                          spy=spy.record(expect=expect))
            if not record['spy']['passed']:
                raise Stopped(f"the spy did not pass (expected reads {record['spy']['expected_reads']})")
            write_once(out/RECORD_NAME, record)
        except BaseException as exc:
            stopped = dict(schema=SCHEMA, stopped=f'{type(exc).__name__}: {exc}', kind=type(exc).__name__, utc_start=started, utc_end=utc(), code=code,
                           lock=dict(path=str(lock), line=line), out=str(out), out_made=out.is_dir(), traceback=traceback.format_exc()[-4000:], spy=spy.record(),
                           record=record)              # diff review minor 2: the whole record (its spy with the expected reads) when it was made, e.g. the spy did not pass
            write_stopped(out, lock, stopped)
            if isinstance(exc, Exception):
                raise Stopped(stopped['stopped']) from exc
            raise                                   # KeyboardInterrupt, SystemExit: the record first, then as it was
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--out', help='--dry-run only: a new or empty directory for dry_run.json')
    args = parser.parse_args(argv)
    if args.run and args.out:
        parser.error('--out is for --dry-run only; the formal run writes OUT')
    problem = bytecode_problem()
    if problem:
        print(f'refused (bytecode): {problem}', file=sys.stderr)
        return EXIT['refused']
    try:
        if args.dry_run:
            d = dry_run(out=args.out)
            w = d['will_read']
            print(f"dry run: code {d['code']['commit']} clean {d['code']['clean']}; prereg sha256 equal {d['prereg']['equal']}; admission: {d['admission']['selection']['pairs']} pairs")
            print(f"will read: {w['pair_npz']} pair.npz (members {w['members_per_pair']}), {w['provenance']} provenance.json, {w['wm_posteriors']} WM posteriors = "
                  f"{w['data_files']} data files; plus {w['repository_files']} and {w['stage_files']}")
            print(f"exist now: {d['exists']}; lock {d['lock']}; out {d['out']}; spy passed {d['spy']['passed']} ({d['spy']['n_events']} opens, data reads "
                  f"{len(d['spy']['data_reads'])}); ready {d['ready']}")
            return EXIT['ok']
        record = run()
        print(f"c = {record['c']['decimal']} ({record['c']['hex']}); W_dev = {record['W_dev']['decimal']} ({record['W_dev']['hex']}); {OUT/RECORD_NAME}")
        return EXIT['ok']
    except Refused as exc:
        print(f'refused ({exc.stage}): {exc}', file=sys.stderr)
        return EXIT['refused']
    except (Stopped, Breach) as exc:
        print(f'stopped: {exc}', file=sys.stderr)
        return EXIT['stopped']


if __name__ == '__main__':
    sys.exit(main())
