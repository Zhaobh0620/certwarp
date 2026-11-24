"""LP-D (runs/20261119-lpba/任务书_D_开发集逐位对照.md): what the three development bitwise checks of the LPBA40 preregistration share -- §12 item 4 (a), (b), (c); §8.1 items 9-10;
contract C26.5-11 (revision 11b item 4: (c) through the development pair set), C26.9-1 (they run before P_L, from a clean detached checkout, code_commit and clean recorded; rule one does
not apply), C26.4 L38, L39; C26.5-12 N55, N56.

  lpdev_a.py   (a) the new stage's S01 <- S02 (the feasibility stage of prep_lpba.py) against the old stage_LPBA40_v1/LPBA40_S01__LPBA40_S02/pair.npz: CPU, after the feasibility's data preparation
  lpdev_b.py   (b) HI's path from a stage array (hi_lpba.stage_reader, hi_lpba.predict: hs_a) on the 20 development subjects of stage_L2R_val_v1, bitwise against the registered predictions
               seg/unet35_v4_ens/pred: GPU (about 20 minutes)
  lpdev_c.py   (c) the method entry in HE's development mode (run_hx.py pair: the development pair set, sealed_admission and the development allowlist) on 0438 <- 0439 -- the run's start string and
               gate; then the comparison: t88/field_0600.npz's u bitwise against HS's T88 field of the pair (hs_v1), and the line status hx_certified

Each tool has a one-shot lock (O_EXCL, after every check, never edited or deleted), a record (the files read with their sha256, the code commit and its clean flag, the spy's record, the conclusion
passed / not passed) and a dry run (no array, no image, no GPU; the planned files stat'ed; nothing written but DIR/dry_run.json).  Exit codes (EXIT): 0 the check ran and passed (or a dry run
passed), 2 usage, 3 the check ran and did not pass (its record written), 8 refused before the lock -- any exception there, a disk guard's and the spy's included (guarded; review minors 4, 6);
nothing written, a dry run writes its refusal into DIR --, 9 stopped after the lock: everything after the lock runs inside after_lock, so any exception there writes the stopped record (review
minor 5; kept, not rerun: project rules, hard rule 4).

The roots (code constants, read at call time from lpba_common.RESULTS, which the tests point into tmp_path; new roots, never written twice):
  (a)  <RESULTS>/lpba40_feas_v1/devcheck_a/ and the lock <RESULTS>/lpba40_feas_v1/locks/lpba_devcheck_a.lock: inside the feasibility root, since (a) is part of the feasibility (prereg §8.1
       item 10) and a rerun of the feasibility renames that root whole (C26.2-3);
  (b)  <RESULTS>/lpba40_devcheck_v1/b/ (its record, and pred/<s>/ of the stage-array inference) and the lock <RESULTS>/lpba40_devcheck_v1/locks/devcheck_b.lock;
  (c)  the run's output <RESULTS>/lpba40_devcheck_v1/c_run/0438-0439/ (run_hx.py pair --out; its cache c_run/cache), the comparison's record <RESULTS>/lpba40_devcheck_v1/c/ and the lock
       <RESULTS>/lpba40_devcheck_v1/locks/devcheck_c.lock.
The hold file <RESULTS>/lpba40_test_v1/hold (lpba_common.check_hold) pauses every one of them.

The spy is lpba_common's (hp_core's audit-hook spy, the name rule widened to 'label'), with one more allowance (Policy.npz_members): the members of an npz file named one by one -- the registered
predictions' probs36.npz and structure_groups.npz (member p), HS's and the run's field_0600.npz (member u); a pair.npz anywhere answers fixed, moving and initial_u only (hp_core.PAIR_MEMBERS:
bayesreg.io.pair.load_pair reads the all-zero initial_u when the archive has it; no label member is ever asked).
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
LPBA = HERE.parent
EXPERIMENTS = LPBA.parent
for _p in (LPBA, EXPERIMENTS/'hi', EXPERIMENTS/'holdout', EXPERIMENTS/'he'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import lpba_common as lc  # noqa: E402

hc, hp = lc.hc, lc.hp

DEVCHECK_DIR = 'lpba40_devcheck_v1'
EXIT = dict(passed=0, usage=2, not_passed=3, refused=8, stopped=9)
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
TASK = 'runs/20261119-lpba/任务书_D_开发集逐位对照.md'
CONTRACT = 'C26.5-11 (revision 11b item 4), C26.9-1; prereg §12 item 4, §8.1 items 9-10'


class Stopped(Exception):
    """After the lock: the check stops (its stopped record kept, not rerun).  original: the exception that stopped it; record_error: the exception of the stopped record's own write when
    that failed too (LP-D diff review minor, LP-C2 integration item 6), else None."""

    def __init__(self, message, *, original=None, record_error=None):
        super().__init__(message)
        self.original, self.record_error = original, record_error


def utc():
    return lc.utc()


def roots():
    """The development checks' roots (module docstring), from lpba_common.RESULTS at call time."""
    base = Path(lc.RESULTS)/DEVCHECK_DIR
    feas = lc.roots('lpba40_feas')
    return SimpleNamespace(base=base, locks=base/'locks', b=base/'b', c=base/'c', c_run=base/'c_run', a=feas.root/'devcheck_a', a_lock=feas.locks/'lpba_devcheck_a.lock',
                           b_lock=base/'locks'/'devcheck_b.lock', c_lock=base/'locks'/'devcheck_c.lock', feas=feas)


def file_record(path):
    """{path, sha256} of a file read (its bytes, as it is now)."""
    return dict(path=str(path), sha256=hc.sha256_file(path))


def arrays_equal(mine, theirs):
    """Bitwise: the same dtype, the same shape, the same C-order bytes.  With the number of elements that differ (bit patterns, then value) and the largest absolute difference when they
    are comparable (same shape)."""
    import numpy as np
    record = dict(dtype=str(mine.dtype), reference_dtype=str(theirs.dtype), shape=list(mine.shape), reference_shape=list(theirs.shape),
                  sha256=lc.array_sha256(mine), reference_sha256=lc.array_sha256(theirs))
    same_layout = mine.dtype == theirs.dtype and mine.shape == theirs.shape
    record['bitwise_equal'] = bool(same_layout and np.ascontiguousarray(mine).tobytes() == np.ascontiguousarray(theirs).tobytes())
    if mine.shape == theirs.shape:
        a, b = np.asarray(mine, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
        differ = a != b
        record.update(n_elements=int(a.size), n_elements_differ=int(differ.sum()), max_abs_difference=float(np.max(np.abs(a-b))) if differ.any() else 0.)
    return record


def check_code(code, *, dry_run):
    """One clean code tree (hp_core.check_code: this module's tree, bayesreg's and bayesreg.io.sealed.REPO are one, git clean); a dry run records the problems only.  C26.9-1: these checks
    run before P_L and are not under rule one; the record names the commit and its clean flag."""
    checked = hp.check_code(code, require_clean=not dry_run)
    return dict(code, check=checked, rule_one='not applied: the development checks run before P_L (C26.9-1)')


def bytecode_problem():
    """None when Python writes no bytecode (python -B or PYTHONDONTWRITEBYTECODE): a real run's code tree gets no __pycache__."""
    return None if sys.flags.dont_write_bytecode else 'run with python -B (or PYTHONDONTWRITEBYTECODE): the code tree gets no __pycache__'


# ---- the spy ------------------------------------------------------------------------------------------------------------------------------------
class Policy(lc.Policy):
    """lpba_common.Policy with npz_members: {file: member names} -- the npz files (other than a pair.npz) whose named members this check may ask (read only; the file itself must be
    allowed by read_exact or write_inside as usual)."""

    def __init__(self, *, npz_members=None, **kwargs):
        super().__init__(**kwargs)
        self.npz_members = {Path(p).resolve(): tuple(keys) for p, keys in (npz_members or {}).items()}

    def document(self):
        return dict(super().document(), npz_members=[dict(file=str(p), members=list(k)) for p, k in sorted(self.npz_members.items())])


class Spy(lc.Spy):
    """lpba_common.Spy (its own outputs' npz members allowed) plus Policy.npz_members; every other npz member goes to hp_core's rule (a pair.npz: fixed, moving, initial_u)."""

    def see_member(self, npz, key):
        source = getattr(getattr(npz, 'fid', None), 'name', None)
        _, resolved = hp._forms(source)
        allowed = getattr(self.policy, 'npz_members', {}).get(resolved) if resolved is not None else None
        if allowed is not None:
            self.members.append(dict(file=str(source), member=str(key), allowed=list(allowed)))
            if key not in allowed:
                self.breaches.append(dict(path=str(source), member=str(key), cls='forbidden', why=f'a member other than {list(allowed)} of a file named in npz_members'))
                raise hp.Breach(f'spy ({self.policy.name}): the member {key!r} of {source} was asked (allowed: {list(allowed)})')
            return
        super().see_member(npz, key)


@contextmanager
def spying(policy):
    """lpba_common.spying with this module's Spy: installed only when the guard's cache of the sealed lists is empty (C23.6-5); removed after; an exception leaving the block carries the
    spy's record as `hp_spy`."""
    from bayesreg.io.sealed import sealed_cache_is_empty
    if not sealed_cache_is_empty():
        raise lc.Refused('spy', 'the guard\'s cache of the sealed lists is populated before the spy was installed (C23.6-5): run the check in a process of its own')
    spy = Spy(policy).install()
    try:
        yield spy
    except BaseException as exc:
        try:
            exc.hp_spy = spy.record()
        except Exception:                       # noqa: BLE001 - the record is a convenience
            pass
        raise
    finally:
        spy.remove()


# ---- records --------------------------------------------------------------------------------------------------------------------------------------
def planned(paths):
    """A dry run's view of the files a run would read: stat only (exists, size)."""
    out = []
    for path in paths:
        try:
            st = os.stat(path)
            out.append(dict(path=str(path), exists=True, size=st.st_size))
        except OSError:
            out.append(dict(path=str(path), exists=False))
    return out


def write_dry_run(out, record):
    Path(out).mkdir(parents=True, exist_ok=True)
    return hc.write_json_once(Path(out)/'dry_run.json', record)


def stop(record, exc, spy, target_dir, lock, name):
    """After the lock: the stopped record (in target_dir, or beside the lock when target_dir could not be made), kept; the spy removed first (its record taken)."""
    record.update(stopped=f'{type(exc).__name__}: {exc}'[:4000], spy=spy.record(), ended_utc=utc(), passed=False,
                  note='kept, not rerun (project rules, hard rule 4); the lead decides')
    spy.remove()
    where = Path(target_dir) if Path(target_dir).is_dir() else Path(lock).parent
    return hc.write_json_once(where/name, record)


def guarded(call, *, dry, out_dir, schema, out):
    """Run `call` (a tool's run or compare) and map what leaves it to (record or None, exit code or None) -- the one mapping of the three tools (review minors 4, 6):
      Refused, InfrastructureFailure (the disk guard), Breach -- all before the lock (after it the tools raise Stopped) -> EXIT refused;
      Stopped -> EXIT stopped (the tool wrote its stopped record);
      any other Exception -- also before the lock -> EXIT refused, not a traceback's 1.
    A dry run's refusal is written to DIR/dry_run.json.  KeyboardInterrupt and SystemExit pass."""
    try:
        return call(), None
    except Stopped as exc:
        out(f'stopped after the lock (record kept): {exc}')
        return None, EXIT['stopped']
    except Exception as exc:                    # noqa: BLE001 - before the lock: a refusal, whatever it is (review minor 6)
        if isinstance(exc, hc.Refused):
            stage = exc.stage
        elif isinstance(exc, hc.InfrastructureFailure):
            stage = f'infrastructure ({exc.kind})'
        elif isinstance(exc, hp.Breach):
            stage = 'spy'
        else:
            stage = f'error before the lock ({type(exc).__name__})'
        out(f'refused ({stage}): {exc}')
        if dry:
            write_dry_run(out_dir, dict(schema=schema+'.dry_run', refused=dict(stage=stage, message=str(exc)), utc=utc(), spy=getattr(exc, 'hp_spy', None)))
        return None, EXIT['refused']


def after_lock(record, spy, target_dir, lock, name):
    """The context of everything a tool does after its lock (review minor 5): any exception there writes the stopped record (stop) and leaves as Stopped (a BaseException such as
    KeyboardInterrupt as itself, after the record).  When the stopped record's own write fails (LP-D diff review minor; LP-C2 integration item 6), the context still leaves as Stopped,
    carrying both: its message names both, .original the first exception (also its __cause__), .record_error the write's (also its __context__) -- whatever the first one was."""
    return _AfterLock(record, spy, target_dir, lock, name)


class _AfterLock:
    def __init__(self, record, spy, target_dir, lock, name):
        self.args = (record, spy, target_dir, lock, name)

    def __enter__(self):
        return self

    def __exit__(self, kind, exc, tb):
        if exc is None:
            return False
        record, spy, target_dir, lock, name = self.args
        try:
            stop(record, exc, spy, target_dir, lock, name)
        except Exception as failed:                     # noqa: BLE001 - the stopped record not written: still Stopped, with both exceptions
            raise Stopped(f'{type(exc).__name__}: {exc}; and the stopped record {name} could not be written: {type(failed).__name__}: {failed}', original=exc,
                          record_error=failed) from exc
        if isinstance(exc, Exception) and not isinstance(exc, Stopped):
            raise Stopped(f'{type(exc).__name__}: {exc}', original=exc) from exc
        return False


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


__all__ = ['CONTRACT', 'DEVCHECK_DIR', 'EXIT', 'Policy', 'Spy', 'Stopped', 'TASK', 'after_lock', 'arrays_equal', 'bytecode_problem', 'check_code', 'file_record', 'guarded', 'planned',
           'read_json', 'roots', 'spying', 'stop', 'utc', 'write_dry_run']
