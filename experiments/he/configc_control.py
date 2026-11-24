"""HE's control of fallback one (runs/20261106-he/任务书.md item 4; contract C22.5-2 "门槛分两处做", second half): on the CPU, on a small synthetic volume, with a fixed number of outer steps, the frozen worker of
configuration C run through the production path of experiments/he/firewall_c.py (the derived pair, the private copy's slot, the real frozen `prepare`, check_prepared, the run's own snapshot, classify_run):

    A  original    the synthetic pair.npz WITH its label members, as D0 hands A0 its pair (labels in the archive the worker decodes);
    B  derived     firewall_c.derive_pair of the same pair: fixed, moving, initial_u byte for byte, the label members zero;
    C  missing     B without the four label members: how the frozen worker behaves when they are absent;
    D  changed     B of a pair whose moving image differs by 1e-9 at one voxel: the comparison can fail.
A and B must be the same bit for bit: final_native.npz:u, every commit, the history and inner attempts without timings and the input hashes, the transfer attempts, the summary's status, commits and role.  The label
members differ (real against zero) and the field does not.  C must fail after the solve (the frozen label evaluation reads the members) and B's labels.json carries no Dice.  D must differ.

    python -B experiments/he/configc_control.py --scratch DIR --out JSON [--cache DIR]       (no PYTHONPATH; DIR new and empty; removed at the end)
    python -B experiments/he/configc_control.py driver RUN --outer-steps N                    (the child: one run directory, patched as below)

The patches, in the driver process only, on the frozen module objects (no file changes; the snapshot keeps the freeze's hashes), as experiments/seed/e2e_control.py:
  e48.worker          device 'cuda' (F:scripts/run_stage_3d_e48_pg.py:821, hard-coded) -> 'cpu'
  e19.OUTER_STEPS     N instead of 1000, so that no wall clock decides anything
  e48.prewarm_jit, e48.first_use_prewarm     empty (GPU only, before the task clock)
Not reached: the GPU path, the two prewarms on a device, systemd-run, a budget that ends the run, real images.  No real image, label or field is read.
Fixed steps: the patch fixes OUTER_STEPS only; the frozen worker's level deadlines (the level shares of its 90 s) and the T88 pre-optimizer's 60 s are wall clocks.  The record asserts, from levels.json and
level_results.json, that the pre-optimizer was not stopped and every level ended at outer_max_iterations (control_established); otherwise the reading is "control not established", not "different".
The comparison leaves out the timing keys by exact name (TIMING_KEYS) and records what it removed from each run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
OUTER_STEPS = 3
CORES = 2                                       # the driver's CPU affinity: a control, not a timing
SHAPE_XYZ = (38, 36, 34)                        # at least 34 nodes per axis: the 8x level of the frozen pyramid needs 5 (experiments/seed/k1_control.py)
SUBJECTS = ('OASIS_OAS1_9001_MR1', 'OASIS_OAS1_9002_MR1')     # synthetic: no data of these names exists; on the control's own allowlist only
# the timing keys of the compared records, by exact name (revision 1, review minor 7: not by substring, which also took `timeout`, `runtime`); read from a kept control run 2026-11-06 (commits.json,
# history.json, inner_attempts.json, levels.json, final_native.json).  The input hashes are not left out: those records carry none that differ.  What each comparison removed is in the record.
TIMING_KEYS = frozenset(('solver_commit_seconds', 'visible_seconds', 'elapsed_seconds', 'inner_elapsed_seconds', 'node_measure_seconds', 'seconds', 'prepare_seconds', 'seconds_total', 'wait_seconds'))
PATCHES = ("e48.worker device 'cuda' -> 'cpu' (F:scripts/run_stage_3d_e48_pg.py:821)", 'e19.OUTER_STEPS = N', 'e48.prewarm_jit, e48.first_use_prewarm: empty (GPU only, before the task clock)')


# ---- the child -----------------------------------------------------------------------------------------------------------------------------------
def driver(run, outer_steps):
    """A new process: e48.run(run) on the frozen module of the run's snapshot, with the three patches."""
    run = Path(run).resolve()
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:CORES]))     # before the libraries start their pools: the frozen worker sets its own thread counts (experiments/seed/k1_control.py), the machine may be training
    sys.path.insert(0, str(run/'snapshot'))
    import scripts.run_stage_3d_e48_pg as e48
    import scripts.run_stage_3d_e19_mri as e19
    import jd_curl
    if (run/'snapshot').resolve() not in Path(jd_curl.__file__).resolve().parents:
        raise SystemExit(f'jd_curl comes from {jd_curl.__file__}, not the snapshot')
    e19.OUTER_STEPS = outer_steps
    original = e48.worker

    def worker(task, out, data, *, device='cuda', **options):
        return original(task, out, data, device='cpu', **options)
    e48.worker = worker
    e48.prewarm_jit = lambda *a, **k: 0.
    e48.first_use_prewarm = lambda output, cfg: (None, None)
    e48.run(run)
    return 0


class CpuControlLauncher:
    """firewall_c's launcher interface: the inner command of a unit, with this module's driver for the frozen entry.  Controls and tests only; the method's launchers run the frozen entry unpatched."""
    kind = 'cpu_control'

    def __init__(self, cache, outer_steps=OUTER_STEPS, threads=1):
        self.cache, self.outer_steps, self.threads = Path(cache), outer_steps, threads

    def argv(self, run, short):
        import firewall_c as fc
        return fc.check_launch(['/bin/bash', '-c', fc.inner_command(run, [PY, '-B', str(Path(__file__).resolve()), 'driver', str(run), '--outer-steps', str(self.outer_steps)])])

    def __call__(self, run, short):
        import firewall_c as fc
        argv = self.argv(run, short)
        (Path(run)/'cmd.txt').write_text(' '.join(argv)+'\n')
        env = dict(fc.frozen_env(self.cache), PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=os.environ.get('HOME', ''), CUDA_VISIBLE_DEVICES='', MALLOC_ARENA_MAX='1')
        env.update({k: str(self.threads) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')})
        done = subprocess.run(argv, cwd=Path(run)/'snapshot', env=env, capture_output=True, text=True)
        return dict(kind=self.kind, argv=argv, returncode=done.returncode, stderr_tail=done.stderr[-500:], patches=list(PATCHES), outer_steps=self.outer_steps)


# ---- the synthetic pair ----------------------------------------------------------------------------------------------------------------------------
def allowlist_document(subjects):
    return dict(schema='bayesreg.dev_subjects.v1', subjects=dict(OASIS=list(subjects)),
                sources=[dict(name='he_control', dataset='OASIS', path='synthetic/he_control.txt', sha256='0'*64, never_for_training=False, subjects=list(subjects))])


def write_pair(stage, pair, *, moving_change=0., seed=20261106):
    """stage/pair/{pair.npz, manifest.json} as the frozen prepare scripts write them: images, label members, initial_u; the manifest with the frozen weights of the grid and the baseline Dice of the labels."""
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from jd_curl.three_d.grid import make_grid_3d
    from jd_curl.three_d.nifti import dice_labels, file_sha256
    import firewall_c as fc
    nx, ny, nz = SHAPE_XYZ
    grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')
    rng = np.random.default_rng(seed)
    fixed = gaussian_filter(rng.uniform(size=grid.shape), 2.)
    moving = np.roll(fixed, 1, axis=2)
    fixed, moving = ((a-a.min())/(a.max()-a.min()) for a in (fixed, moving))
    moving[nz//2, ny//2, nx//2] += moving_change
    seg35 = (np.arange(np.prod(grid.shape)).reshape(grid.shape) % 7).astype(float)
    seg4 = (seg35 % 5).astype(float)
    arrays = dict(fixed=fixed, fixed_seg4=seg4, fixed_seg35=seg35, moving=moving, moving_seg4=np.roll(seg4, 1, axis=0), moving_seg35=np.roll(seg35, 1, axis=0), initial_u=np.zeros(grid.shape+(3,)))
    d = Path(stage)/pair
    d.mkdir(parents=True)
    np.savez_compressed(d/'pair.npz', **arrays)
    baseline = {k: dice_labels(arrays['fixed_'+k], arrays['moving_'+k], labels=list(range(1, n+1))) for k, n in (('seg4', 4), ('seg35', 35))}
    fs, ms = pair.split('__')
    manifest = dict(pair=pair, fixed_subject=fs, moving_subject=ms, shape_zyx=list(grid.shape), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm',
                    sources=dict(fixed=dict(sha256=hashlib.sha256(fs.encode()).hexdigest()), moving=dict(sha256=hashlib.sha256(ms.encode()).hexdigest())), pair_sha256=file_sha256(d/'pair.npz'),
                    weights=fc.frozen_weights(grid), baseline_label_metrics=baseline, preprocessing='synthetic (HE control)')
    (d/'manifest.json').write_text(json.dumps(manifest, default=float))
    return d


# ---- the comparison ---------------------------------------------------------------------------------------------------------------------------------
def strip(value, removed=None, where=''):
    """value without the keys of TIMING_KEYS (exact names, revision 1, review minor 7); `removed` counts what was taken out, per key."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in TIMING_KEYS:
                if removed is not None:
                    removed[k] = removed.get(k, 0)+1
                continue
            out[k] = strip(v, removed, where)
        return out
    if isinstance(value, list):
        return [strip(v, removed, where) for v in value]
    return value


def fixed_steps(levels, level_results):
    """Whether every level ended by its step limit, not by a wall-clock deadline: the frozen worker's level deadlines (F:scripts/run_stage_3d_e19_mri.py level shares of the 90 s) and the T88 pre-optimizer's 60 s are
    wall clocks; the patch fixes only OUTER_STEPS.  A level cut by a deadline would make A and B stop at different steps: the control is then not established (not a difference).  (passed, record)."""
    pre = [((e.get('preoptimizer') or {}).get('stopped')) for e in levels if e.get('event') == 'preoptimizer_done']
    ends = [(e.get('level'), e.get('status')) for e in levels if e.get('event') == 'level_completed']
    results = [(e.get('level'), e.get('status')) for e in level_results or []]
    passed = bool(pre) and all(p is None for p in pre) and bool(ends) and all(s == 'outer_max_iterations' for _, s in ends) and all(s in ('outer_max_iterations', 'skipped_by_preoptimizer') for _, s in results)
    return passed, dict(preoptimizer_stopped=pre, level_completed=ends, level_results=results, rule='the pre-optimizer not stopped by its wall; every level completed at outer_max_iterations (or skipped by the pre-optimizer)')


def summary_of(run):
    import numpy as np
    import firewall_c as fc
    task = fc.task_dir(run)

    def read(name):
        p = task/name
        return json.loads(p.read_text()) if p.is_file() else None
    final = None
    if (task/fc.FINAL).is_file():
        with np.load(task/fc.FINAL) as z:
            final = hashlib.sha256(np.array(z['u']).tobytes()).hexdigest()
    levels = read('levels.json') or []
    s = read('summary.json') or {}
    labels = read('labels.json')
    removed = {}
    steps_fixed, steps = fixed_steps(levels, read('level_results.json'))
    record = dict(final=final, final_record=strip(read('final_native.json'), removed), commits=strip(read('commits.json'), removed), history=strip(read('history.json'), removed), inner=strip(read('inner_attempts.json'), removed),
                  levels=strip(levels, removed), attempts=[strip({k: e.get(k) for k in ('alpha', 'certificate_passed', 'certificate')}) for e in levels if e.get('event') == 'transfer_attempt'],
                  status=s.get('status'), native=s.get('native_commits'), role=s.get('metrics_role'), labels_final=None if labels is None else labels.get('final'), smoothness=read('smoothness.json'))
    return dict(record, timing_keys_removed=removed, steps_fixed=steps_fixed, steps=steps)


def one(base, name, original, *, derive, drop_labels=False, launcher, allowlist):
    """prepare, check, launch, classify: the production path of firewall_c, on one variant."""
    import numpy as np
    import firewall_c as fc
    from bayesreg.io.pair import load_pair
    pair = original.name
    variant = base/name
    variant.mkdir()
    if derive:
        pin = load_pair(original/'pair.npz', allowlist=allowlist)
        firewall = fc.derive_pair(pin, variant/'derived', pair=pair)
        source = variant/'derived'
        if drop_labels:
            with np.load(source/'pair.npz') as z:
                kept = {k: np.array(z[k]) for k in z.files if k not in fc.SEG_KEYS}
            np.savez(source/'pair.npz', **kept)
            manifest = json.loads((source/'manifest.json').read_text())
            manifest['pair_sha256'] = fc.sha_file(source/'pair.npz')
            (source/'manifest.json').write_text(json.dumps(manifest, indent=1, default=float)+'\n')
    else:
        firewall, source = None, original
    run = variant/'run'
    prepared = fc.prepare(source, run, pair=pair, scratch=variant)
    gate = fc.firewall_check(pin, source, run, pair=pair)[0] if derive and not drop_labels else None
    tick = time.perf_counter()
    launched = launcher(run, name)
    seconds = time.perf_counter()-tick
    classification = fc.classify(run, dxg=0, oom=0)
    return dict(name=name, firewall=firewall, gate=gate, check_prepared_ok=prepared['check']['ok'], launch=dict(kind=launched['kind'], returncode=launched['returncode'], seconds=seconds), classification=classification,
                exit_code=(run/'exit_code').read_text().strip() if (run/'exit_code').is_file() else None, summary=summary_of(run), run=str(run), last_log=((run/'run.log').read_text(errors='replace').strip().splitlines() or [''])[-1][:300])


def control(base, cache):
    sys.path.insert(0, str(HERE))
    sys.path.insert(1, str(ROOT))
    import firewall_c as fc
    from bayesreg.io.guard import load_dev_allowlist
    tick = time.perf_counter()
    pair = '__'.join(SUBJECTS)
    (base/'allowlist.json').write_text(json.dumps(allowlist_document(SUBJECTS)))
    allowlist = load_dev_allowlist(base/'allowlist.json')
    stage = base/'stage'
    original = write_pair(stage, pair)
    changed = write_pair(base/'stage_changed', pair, moving_change=1e-9)
    launcher = CpuControlLauncher(cache)
    rows = dict(A=one(base, 'A_original', original, derive=False, launcher=launcher, allowlist=allowlist),
                B=one(base, 'B_derived', original, derive=True, launcher=launcher, allowlist=allowlist),
                C=one(base, 'C_missing', original, derive=True, drop_labels=True, launcher=launcher, allowlist=allowlist),
                D=one(base, 'D_changed', changed, derive=True, launcher=launcher, allowlist=allowlist))
    a, b, c, d = (rows[k]['summary'] for k in 'ABCD')
    keys = ('final', 'final_record', 'commits', 'history', 'inner', 'levels', 'attempts', 'status', 'native', 'role')
    labels_b = b['labels_final'] or {}
    checks = dict(
        original_and_derived_both_ok=rows['A']['classification']['class'] == 'ok' and rows['B']['classification']['class'] == 'ok',
        derived_bitwise_equal_to_original={k: a[k] == b[k] for k in keys},
        field_moved=a['native'] is not None and a['native'] >= 1 and a['final'] is not None,
        gate_on_the_derived_pair_passes=rows['B']['gate']['passed'] is True,
        derived_labels_json_has_no_dice=all((labels_b.get(k) or {}).get('macro_dice') is None for k in ('seg4', 'seg35')) and bool(labels_b),
        derived_smoothness_has_no_brain_numbers=((b['smoothness'] or {}).get('final') or {}).get('sdlogj_brain') is None,
        original_labels_json_has_dice=all(((a['labels_final'] or {}).get(k) or {}).get('macro_dice') is not None for k in ('seg4', 'seg35')),
        missing_members_fail_after_the_solve=rows['C']['exit_code'] not in (None, '0') and c['final'] is not None and 'KeyError' in rows['C']['last_log'],
        missing_members_class=rows['C']['classification']['class'],
        negative_control_differs=d['final'] is not None and d['final'] != b['final'])
    established = {k: rows[k]['summary']['steps_fixed'] for k in 'ABD'}          # every level of the compared runs ended at its step limit, not at a wall-clock deadline
    flat = [all(v.values()) if isinstance(v, dict) else v for k, v in checks.items() if k != 'missing_members_class']
    return dict(role='control evidence on a synthetic volume (CPU, fixed outer steps), not a run of the method', shape_xyz=list(SHAPE_XYZ), outer_steps=OUTER_STEPS, patches=list(PATCHES),
                not_reached=['the GPU path', 'the two prewarms on a device', 'systemd-run', 'a budget that ends the run', 'real images'], checks=checks, control_established=established,
                passed=all(established.values()) and all(flat), reading=None if all(established.values()) else 'control not established: a level ended at a wall-clock deadline, not at its step limit (not a difference)',
                timing_keys_removed={k: rows[k]['summary']['timing_keys_removed'] for k in 'ABCD'}, steps={k: rows[k]['summary']['steps'] for k in 'ABCD'},
                rows={k: {x: v[x] for x in ('name', 'firewall', 'gate', 'check_prepared_ok', 'launch', 'classification', 'exit_code', 'last_log')} for k, v in rows.items()},
                finals={k: rows[k]['summary']['final'] for k in 'ABCD'}, seconds=time.perf_counter()-tick)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['driver']:
        ap = argparse.ArgumentParser()
        ap.add_argument('run', type=Path)
        ap.add_argument('--outer-steps', type=int, required=True)
        a = ap.parse_args(argv[1:])
        return driver(a.run, a.outer_steps)
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--cache', type=Path, default=None)
    ap.add_argument('--keep', action='store_true', help='keep the scratch directory (inspection)')
    a = ap.parse_args(argv)
    if a.scratch.exists():
        raise SystemExit(f'{a.scratch} exists (nothing is overwritten)')
    a.scratch.mkdir(parents=True)
    try:
        result = control(a.scratch, a.scratch/'cache' if a.cache is None else a.cache)
    finally:
        if not a.keep:
            shutil.rmtree(a.scratch, ignore_errors=True)
    text = json.dumps(result, indent=1, default=float, ensure_ascii=False)
    print(text)
    if a.out:
        a.out.write_text(text+'\n')
    return 0 if result['passed'] else 8


if __name__ == '__main__':
    sys.exit(main())
