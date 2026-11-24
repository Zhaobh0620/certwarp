"""XB's spies (runs/20261107-xb/任务书.md item 1 (e)): what the driver and the method processes open is checked against a policy (spy/xb_policy.py); a breach stops the run.

Three spies, each with its own limit (runs/20261107-xb/实现说明.md §spy):
  DriverSpy   in the driver's own process (sci310): a Python audit hook (installed once per process, inert while no policy is active) sees every Python-level open, os.open and import.  The original pair.npz
              is forbidden except inside `reading()`, which the driver's reader and the container hash enter (xb_core.read_image_members, the sha256 against the manifest); while inside, the reader asks the
              archive for fixed and moving only (the members are counted at the loader's hook bayesreg.io.pair._open_npz, as tests replace it).  The positive condition: the original was opened, and only
              inside reading(); a spy that saw no open at all does not pass.
  spyrun      in each method process (spy/xb_spyrun.py, the same policy logic): the Python arms' m_*.py and the ConvexAdam-v4seg wrapper run under it.  Its log is read back here (method_spy_record).
  FdSampler   for every arm, the CLI tools included: once a second (the driver's monitor tick, as BL2's Monitor) the open file descriptors (/proc/<pid>/fd) of every process of the method's session; a
              forbidden or unlisted path kills the session and is a breach.  It sees only what is open at a sample; a file opened and closed between two samples is not seen; every
              descriptor is judged as a read (/proc/<pid>/fd gives no mode: a descriptor open for writing in the code tree is not refused by it, XF review minor 2).
The method process is never given the original's path: its argv holds the derived npz (or NIfTI files the driver wrote), its working directory is the run's work directory, its environment is drive.py's
base_env.  The policy file it reads names directories, never the original file.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import threading

HERE = Path(__file__).resolve().parent
if str(HERE/'spy') not in sys.path:
    sys.path.insert(0, str(HERE/'spy'))
import xb_policy  # noqa: E402

SPYRUN = HERE/'spy'/'xb_spyrun.py'
SPYRUN_BREACH_EXIT = 97


class Breach(Exception):
    """A process of the runner opened what its policy forbids or does not list."""


# ---- the policies -------------------------------------------------------------------------------------------------------------------------------
SYSTEM_READ = ('/usr', '/lib', '/lib64', '/bin', '/sbin', '/etc', '/proc', '/sys', '/dev', '/run', '/opt/conda')
SYSTEM_WRITE = ('/dev/null', '/dev/shm', '/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/dxg', '/proc/self')
HOME = Path('/home/rmcfz')
CACHES = tuple(str(HOME/p) for p in ('.cache', '.nv', '.triton', '.keras', '.config/matplotlib', '.cupy', '.nvcache'))
CONDA = HOME/'miniconda3'
BASELINES = HOME/'opt'/'baselines'
SEQ_VP_3D = HOME/'code'/'python_projects'/'seq_vp_3d'
SUBJECT_DATA = Path('/mnt/d/data')


def forbidden_roots(*, results, main_repo, holdout_root):
    roots = [str(SEQ_VP_3D), str(SUBJECT_DATA), str(Path(main_repo)/'data'), str(results), str(Path(holdout_root)/'stage'), str(Path(holdout_root)/'atropos')]
    try:
        roots.append(os.path.realpath(Path(main_repo)/'data'))        # the link's target (seq_vp_3d/data today)
    except OSError:
        pass
    return roots


def code_data_root(code_root):
    """XF revision 2 (review I-1): the running code tree's own data/ as a forbidden root.  In production main_repo (hc.MAIN_REPO, the main checkout) is not the code root (a P0 checkout,
    <results>/holdout_B_v1/code), so forbidden_roots does not hold the checkout's data/; listed here it lies at or below the code root, and xb_policy rule 3a then never reads through it --
    a link (Policy resolves it: seq_vp_3d/data) or a real directory alike (hp_core lists <code root>/data the same way; hb_review's hold-out deny holds hb.REPO/data)."""
    return str(Path(code_root)/'data')


def method_policy(*, name, work, code_root, env_prefix, baselines, results, main_repo, holdout_root, tmp='/tmp'):
    """The policy of a method process: read the code (the harness and the spy), the system, conda (its environment and the base), its baselines' directory, the caches and its work directory; write only
    the work directory, the caches, /tmp and the device files.  Forbidden: the stages, the subjects' data, the seq_vp_3d tree, the whole results root but its own work directory.
    code_roots (XF, xb_policy rule 3a): the code tree is read even where it lies inside the results root (the hold-out's checkout <results>/holdout_B_v1/code, a P0 checkout), but not its
    data/ (code_data_root, a forbidden root); the spyrun hook refuses a write into it (classify with write=True).  FdSampler judges every descriptor it samples as a read (/proc/<pid>/fd gives
    no mode): a descriptor of the code tree open for writing is not refused by the sampler, as with the code tree outside the results root (a read root there)."""
    work = str(work)
    return dict(name=name, allowed_exact=[], allowed_inside=[work],
                forbidden_roots=[*forbidden_roots(results=results, main_repo=main_repo, holdout_root=holdout_root), code_data_root(code_root)], code_roots=[str(code_root)],
                read_roots=[str(code_root), *SYSTEM_READ, str(CONDA), *(str(b) for b in baselines), str(tmp)],
                write_roots=[work, str(tmp), *SYSTEM_WRITE, *CACHES],
                name_exempt_roots=[str(code_root), *SYSTEM_READ, str(CONDA), *(str(b) for b in baselines)])


def driver_policy(*, work, record_dir, lock_dir, code_root, results, main_repo, holdout_root, extra_exact=(), extra_write=(), extra_read=()):
    """The driver's policy: the code tree, conda, the baselines (their sha256), the system; write its record directory (the arm's: P19 reads the arm's other records), its lock directory, its work
    directory and `extra_write` (the D: store, a dry run's output); the pair's manifest.json and the predicted maps' files named one by one (`extra_exact`).  No /tmp (the driver writes no temporary
    file there).  The original pair.npz is forbidden by name (pair.npz) except inside DriverSpy.reading().
    code_roots (XF, xb_policy rule 3a): the code tree is read even where it lies inside the results root (the hold-out's checkout <results>/holdout_B_v1/code, a P0 checkout: D1 stopped on a
    lazy import there), but not its data/ (code_data_root, a forbidden root); DriverSpy refuses a write into it."""
    return dict(name='xb driver', allowed_exact=[os.path.realpath(p) for p in extra_exact], allowed_inside=[str(work), str(record_dir), str(lock_dir), *(str(p) for p in extra_write)],
                forbidden_roots=[*forbidden_roots(results=results, main_repo=main_repo, holdout_root=holdout_root), code_data_root(code_root)], code_roots=[str(code_root)],
                read_roots=[str(code_root), *SYSTEM_READ, str(CONDA), str(BASELINES), *(str(p) for p in extra_read)],
                write_roots=[str(work), str(record_dir), str(lock_dir), *(str(p) for p in extra_write), *SYSTEM_WRITE, *CACHES],
                name_exempt_roots=[str(code_root), *SYSTEM_READ, str(CONDA), str(BASELINES)])


# ---- the driver's own spy -----------------------------------------------------------------------------------------------------------------------
_ACTIVE = None                  # the DriverSpy that owns the hook now, or None (inert)
_INSTALLED = False
_LOCAL = threading.local()


def _hook(event, args):
    spy = _ACTIVE
    if spy is None or event != 'open' or getattr(_LOCAL, 'busy', False):
        return
    _LOCAL.busy = True
    try:
        spy.see(args)
    finally:
        _LOCAL.busy = False


class DriverSpy:
    """`counted`: paths every open of which is recorded, not only the first (C23.6-5: the sealed lists of C23.2, allowed reads, recorded each time, any number of times)."""

    def __init__(self, policy, original=None, counted=()):
        self.policy = xb_policy.Policy(policy)
        self.original = None if original is None else os.path.realpath(original)
        self.counted = frozenset(os.path.realpath(p) for p in counted)
        self.events, self.breaches, self.original_opens, self.counted_reads = [], [], [], []
        self.inside = 0
        self._seen = set()

    def see(self, args):
        path = xb_policy.norm(args[0])
        if path is None:
            return
        write = xb_policy.is_write(args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else 0)
        if self.original is not None and path == self.original:
            if self.inside and not write:
                self.original_opens.append('inside reading()')
                return
            self.original_opens.append('outside reading()' if not write else 'write')
            self.breaches.append(dict(path=path, write=write, cls='forbidden', why='the original pair.npz outside the reader'))
            raise Breach(f'xb spy: the original pair.npz was opened outside the reader ({path})')
        cls, why = self.policy.classify(path, write)
        if path in self.counted:
            self.counted_reads.append(dict(path=path, write=write, cls=cls))
        if (path, write) not in self._seen:
            self._seen.add((path, write))
            self.events.append(dict(path=path, write=write, cls=cls))
        if cls != 'allowed':
            self.breaches.append(dict(path=path, write=write, cls=cls, why=why))
            raise Breach(f'xb spy: {cls} open of {path} by the driver ({why})')

    @contextmanager
    def reading(self):
        self.inside += 1
        try:
            yield
        finally:
            self.inside -= 1

    def install(self):
        global _ACTIVE, _INSTALLED
        if _ACTIVE is not None and _ACTIVE is not self:
            raise RuntimeError('another DriverSpy is active in this process')
        if not _INSTALLED:
            sys.addaudithook(_hook)                  # an audit hook cannot be removed: it stays installed and inert (_ACTIVE None) between spies
            _INSTALLED = True
        _ACTIVE = self
        return self

    def remove(self):
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False

    def record(self, *, expect_original):
        """passed: no breach and, when the original was to be read, it was opened only inside reading() (>= 1); otherwise not at all."""
        outside = [o for o in self.original_opens if o != 'inside reading()']
        inside = len(self.original_opens)-len(outside)
        positive = (inside >= 1 if expect_original else inside == 0) and not outside and (len(self.events) > 0 or inside > 0)
        return dict(spy='DriverSpy (audit hook, experiments/xb/xb_spy.py)', policy=self.policy.document.get('name'), n_paths=len(self.events), breaches=list(self.breaches),
                    original_opens_inside_reader=inside, original_opens_outside=outside, expect_original=expect_original, positive_condition=positive,
                    sealed_list_reads=list(self.counted_reads), n_sealed_list_reads=len(self.counted_reads),
                    passed=positive and not self.breaches,
                    rule='no forbidden or unlisted open; the original pair.npz opened only by the reader and the container hash, inside reading(), at least once when it is read, never otherwise; '
                         'a spy that saw no open does not pass')


def active():
    return _ACTIVE


def original_context():
    """The active DriverSpy's reading() (a no-op context when none is active, as in the tests that call the reader directly)."""
    spy = _ACTIVE
    if spy is None:
        @contextmanager
        def nothing():
            yield
        return nothing()
    return spy.reading()


# ---- the method process's spy (spyrun) ----------------------------------------------------------------------------------------------------------
def spyrun_argv(python, policy_path, log_path, script, args):
    return [str(python), '-I', '-B', str(SPYRUN), str(policy_path), str(log_path), str(script), *[str(a) for a in args]]


def method_spy_record(log_path, *, expect=()):
    """The record of a spyrun log: the paths with their class, the breaches, the positive condition (the log's start and end lines are there and each of `expect` -- the input files -- was opened)."""
    log_path = Path(log_path)
    rows = []
    if log_path.is_file():
        for line in log_path.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                rows.append(dict(event='unparsed', text=line[:200]))
    opens = [r for r in rows if 'event' not in r]
    breaches = [r for r in rows if r.get('event') == 'breach']
    started = any(r.get('event') == 'start' for r in rows)
    ended = any(r.get('event') == 'end' for r in rows)
    seen = {r.get('path') for r in opens}
    expected = {str(os.path.realpath(p)): os.path.realpath(p) in seen for p in expect}
    positive = started and bool(opens) and all(expected.values())
    return dict(spy='spyrun (audit hook in the method process, experiments/xb/spy/xb_spyrun.py)', log=str(log_path), n_paths=len(seen), breaches=breaches, started=started, ended=ended,
                expected_inputs_opened=expected, positive_condition=positive, passed=positive and not breaches,
                rule='no forbidden or unlisted open in the method process; it started under the hook and opened its inputs (a spy that saw nothing does not pass)')


# ---- the /proc sampler (every arm; the CLI tools' only spy) -------------------------------------------------------------------------------------
def session_pids(sid, proc='/proc'):
    """The pids whose session id (/proc/<pid>/stat field 6) is `sid` (the method was started with start_new_session=True: its pid is its session's id)."""
    out = []
    for entry in os.listdir(proc):
        if not entry.isdigit():
            continue
        try:
            stat = Path(proc, entry, 'stat').read_text()
            fields = stat[stat.rindex(')')+2:].split()
            if int(fields[3]) == sid:
                out.append(int(entry))
        except (OSError, ValueError, IndexError):
            continue
    return out


class FdSampler:
    def __init__(self, policy, sid, proc='/proc'):
        self.policy = xb_policy.Policy(policy)
        self.sid, self.proc = sid, proc
        self.samples, self.paths, self.breaches = 0, set(), []

    def sample(self):
        """One sample: classify each open file of the session (reads only /proc).  Returns the new breaches."""
        self.samples += 1
        new = []
        for pid in session_pids(self.sid, self.proc):
            fd_dir = Path(self.proc, str(pid), 'fd')
            try:
                fds = os.listdir(fd_dir)
            except OSError:
                continue
            for fd in fds:
                try:
                    target = os.readlink(fd_dir/fd)
                except OSError:
                    continue
                if not target.startswith('/'):
                    continue                          # pipe:[...], socket:[...], anon_inode:...
                target = target.removesuffix(' (deleted)')
                path = os.path.realpath(target)
                if path in self.paths:
                    continue
                self.paths.add(path)
                cls, why = self.policy.classify(path, write=False)
                if cls != 'allowed':
                    row = dict(pid=pid, path=path, cls=cls, why=why)
                    self.breaches.append(row)
                    new.append(row)
        return new

    def kill(self):
        try:
            os.killpg(self.sid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def record(self):
        return dict(spy='FdSampler (/proc/<pid>/fd of the method session, once per monitor tick)', samples=self.samples, n_paths=len(self.paths), breaches=list(self.breaches),
                    positive_condition=self.samples >= 1, passed=self.samples >= 1 and not self.breaches,
                    rule='every open file of the method session at each sample is allowed by the policy (reads); at least one sample taken; files opened and closed between samples are not seen')
