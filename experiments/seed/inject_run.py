"""SEED (runs/20261104-seed/任务书.md, plan.md; docs/bayesreg/05_D0协议.md §16): one run of the frozen solver with a given entry field in place of its pre-optimizer.

    python -B <run>/seed_inject.py <run>        (the queue copies this file into the run directory; the unit runs that copy)

What it does, in this order, and nothing else:
  1. reads <run>/seed_entry.json (written by run_seed.py) and the entry field.npz: file sha256, member `u` float64 (D, H, W, 3), bytes sha256, finite, zero faces.  Anything else: exit 3, before any
     frozen code, before the task clock (no started.json; classify_run.py calls that unsure and the queue stops);
  2. imports jd_curl.three_d.preopt from <run>/snapshot and replaces the module attribute `preoptimize` (never another name; never a file);
  3. hands over to the frozen entry: runpy.run_path(<run>/snapshot/scripts/run_stage_3d_e48_pg.py, run_name='__main__') with argv [script, 'run', <run>] and the sys.path of `python script`.

Why the attribute reaches the solver (F = /home/rmcfz/seq_vp_tag_results/frozen_eb808be, eb808be):
  F/scripts/run_stage_3d_e19_mri.py:1890-1895  the worker's callable looks the name up at call time: __import__('jd_curl.three_d.preopt', fromlist=['preoptimize']).preoptimize(fixed, moving, grid, preopt_config,
                                               device=device, execution=preopt_execution); the first-use dry pass takes the other branch of that expression (a synthetic seed) and never calls it;
  F/jd_curl/three_d/multiresolution.py:392-400  raw, record = preoptimizer(fixed, moving, grid); float64 (D, H, W, 3), finite, zero faces; seed = boundary_taper(raw, grid, 4), the one taper of the path;
                                               record + raw_hash, seed_hash; event preoptimizer_done;
  F/jd_curl/three_d/multiresolution.py:430-441  prepare_transfer_problem certifies .9^k * seed (k = 0..6) with the frozen certificate: the entry is uncertified until then, as T88's field is.
The frozen tests inject the same way (archive/seq_vp_3d/tests/three_d/test_native_transfer_alphas_ap2.py, _fake_preoptimizer).

The field: F/jd_curl/three_d/preopt.py:598-601, 614 and its copy experiments/mc0/mc0_core.py optimize_k return float64 (D, H, W, 3) in mm, components x, y, z, zero faces, NOT tapered; run_mc0.py:173 saves exactly that as
`u`.  So nothing is tapered here and the frozen path tapers once.  run_seed.py compares the seed_hash of the run with the taper of the field computed before the clock (seed_entry.json), and with the
tapered_u_sha256 that the MC0 evaluation recorded.

The time: charge_seconds is what computing the entry costs (run_seed.py ENTRY_CHARGE_SECONDS).  The injected function holds the task clock that long, so that the entry is paid out of the same 330 s as
T88's 2 s is in B1TM (the frozen budgets have no 285 s: run_stage_3d_e48_pg.py:24, 405-406, 925).  charge_seconds 0 is for the bitwise control only.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys
from time import perf_counter, sleep
import zipfile

POLICY = 'seed-injected-entry-v1'
ROLE = 'uncertified_given_entry_field_NOT_outer_commit'
ENTRY_KEYS = ('arm', 'pair', 'field_path', 'field_sha256', 'u_sha256', 'source_run_id', 'oracle', 'charge_seconds', 'shape_zyx')
REFUSED = 3


class Refused(Exception):
    """The entry is not the one the queue named: nothing frozen has run."""


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def read_entry(path):
    doc = json.loads(Path(path).read_text())
    missing = [k for k in ENTRY_KEYS if k not in doc]
    if missing:
        raise Refused(f'{path}: missing {missing}')
    return doc


def load_field(entry):
    """The field named by the record, or Refused: file sha256, one member u, float64, the grid's (D, H, W, 3), bytes sha256, finite, zero faces.  Read only: the array comes back not writeable."""
    import numpy as np
    path = Path(entry['field_path'])
    if not path.is_file():
        raise Refused(f'{path}: no such file')
    got = sha_file(path)
    if got != entry['field_sha256']:
        raise Refused(f"{path}: sha256 {got} != {entry['field_sha256']}")
    with zipfile.ZipFile(path) as z:
        if z.namelist() != ['u.npy']:
            raise Refused(f'{path}: members {z.namelist()}, expected only u')
    with np.load(path) as z:
        u = z['u']
    want = tuple(entry['shape_zyx'])+(3,)
    if u.dtype != np.float64 or u.shape != want:
        raise Refused(f'{path}: {u.dtype} {u.shape}, expected float64 {want}')
    u = np.ascontiguousarray(u)
    got = hashlib.sha256(u.tobytes()).hexdigest()
    if got != entry['u_sha256']:
        raise Refused(f"{path}: bytes sha256 {got} != {entry['u_sha256']}")
    if not np.all(np.isfinite(u)):
        raise Refused(f'{path}: not finite')
    if any(np.any(side) for side in (u[0], u[-1], u[:, 0], u[:, -1], u[:, :, 0], u[:, :, -1])):
        raise Refused(f'{path}: the faces are not zero')
    u.setflags(write=False)
    return u


def make_injected(entry, u, *, clock=perf_counter, wait=sleep):
    """The frozen signature of preoptimize; returns (the given field, a record) and holds the clock for charge_seconds.  One call only (a second one means the path is not the one read)."""
    calls = []
    held = [u]                                           # given away at the call: the frozen path keeps its own read-only copy

    def preoptimize(fixed_image, moving_image, grid, config=None, device='cuda', observer=None, execution=None):
        started = clock()
        if calls:
            raise RuntimeError('seed: the pre-optimizer was called twice')
        if observer is not None:
            raise RuntimeError('seed: the frozen path passes no observer here')
        if tuple(grid.shape) != tuple(held[0].shape[:3]):
            raise RuntimeError(f'seed: grid {tuple(grid.shape)} and field {held[0].shape} differ')
        calls.append(1)
        left = float(entry['charge_seconds'])-(clock()-started)
        if left > 0:
            wait(left)
        return held.pop(), dict(policy=POLICY, role=ROLE, arm=entry['arm'], pair=entry['pair'], source_run_id=entry['source_run_id'], oracle=entry['oracle'], field_path=str(entry['field_path']), field_sha256=entry['field_sha256'],
                       u_sha256=entry['u_sha256'], charge_seconds=float(entry['charge_seconds']), seconds=clock()-started, dtype='float64', replaced='jd_curl.three_d.preopt.preoptimize',
                       not_used=dict(config=None if config is None else 'the T88 configuration of the protocol', device=str(device), execution=None if execution is None else 'the profile switches'))
    preoptimize.calls = calls
    return preoptimize


def install(preopt_module, entry, u, **kwargs):
    """Replace the one attribute; returns the function put there."""
    injected = make_injected(entry, u, **kwargs)
    preopt_module.preoptimize = injected
    return injected


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1:
        print('usage: inject_run.py RUN_DIR', file=sys.stderr)
        return 2
    run = Path(argv[0]).resolve()
    snapshot = run/'snapshot'
    script = snapshot/'scripts'/'run_stage_3d_e48_pg.py'
    try:
        entry = read_entry(run/'seed_entry.json')
        u = load_field(entry)
        if not script.is_file():
            raise Refused(f'{script}: no such file')
    except Refused as refused:
        print(f'seed: refused: {refused}', flush=True)
        return REFUSED
    sys.path[0] = str(snapshot/'scripts')                # the sys.path of `python script` (this file's own directory is not on it)
    sys.path.insert(0, str(snapshot))
    from jd_curl.three_d import preopt                   # preopt.py imports numpy and the two validation modules; jd_curl/__init__.py and three_d/__init__.py import nothing
    sys.path.pop(0)                                      # the frozen script inserts its own root
    if snapshot.resolve() not in Path(preopt.__file__).resolve().parents:
        print(f'seed: refused: jd_curl.three_d.preopt comes from {preopt.__file__}', flush=True)
        return REFUSED
    injected = install(preopt, entry, u)
    del u
    print(f"seed: entry {entry['field_path']} sha256 {entry['field_sha256']} source {entry['source_run_id']} oracle {entry['oracle']} charge {entry['charge_seconds']} s; the frozen entry starts", flush=True)
    sys.argv = [str(script), 'run', str(run)]
    runpy.run_path(str(script), run_name='__main__')
    print(f'seed: the frozen entry ended; calls of the injected function: {len(injected.calls)}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
