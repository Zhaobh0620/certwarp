"""SEED's end-to-end control on the CPU (control evidence only, not a run of the plan; docs/bayesreg/reviews/SEED_tools_复核_20261104.md, question 4 and M1).  The independent reviewer wrote it as four scratch scripts (e2e_setup.py, e2e_field.py, e2e_driver2.py,
e2e_compare.py); they are merged into this file by the implementer, with the same steps and the same patches.  What it adds to k1_control.py: the frozen pre-optimizer never ran in the injected process, and the path goes through the real `prepare`, `check_prepared`, `e48.run` and `_run_pair`.

    python -B experiments/seed/e2e_control.py --scratch DIR --out JSON      (no PYTHONPATH; a new, empty DIR; deleted at the end)

Steps: a scratch copy of the frozen source, checked against the freeze hashes; a synthetic pair (38 x 36 x 34, no real image or label) at the place of its data link; the real frozen `prepare` with SEED's options (--seconds 330, this geometry, v41) and the real `check_prepared` for three
run directories; the frozen pre-optimizer's field computed in a process of its own (the worker's arguments, execution parallel_setup + analytic_grad); then three new processes, each `e48.run(run)`: plain; injected, hold 0 s; injected, hold 47 s.  Finally the real seed_inject.py on the real frozen
entry, without a GPU: it must get through the hand-over and stop at the first GPU call, before the task clock.

The patches of the three runs, the only ones, all on the module objects in the driver process (no file changes; the snapshot's hashes stay the freeze's):
  e48.worker          its device='cuda' (F/scripts/run_stage_3d_e48_pg.py:821, hard-coded) becomes 'cpu'
  e19.OUTER_STEPS     3 instead of 1000, so that no wall clock decides anything
  e48.prewarm_jit, e48.first_use_prewarm     empty functions (:850-855: GPU only, before the task clock)
The injection in these three runs is inject_run.install, called by the driver; the hand-over by runpy is checked by the last step and by the tests with a small entry script.

What it does not reach: the GPU path (T84, the fused kernels, Triton; every real run checks policy, raw_hash and seed_hash instead); prewarm_jit and first_use_prewarm with a real device; systemd-run and the watchdog; a wall budget that ends the run (outer_gate); real images.  Allocator: MALLOC_ARENA_MAX=1 in the
children, only to stay under the memory limit of this machine; it changes no number.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import secrets
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAIR = 'OASIS_OAS1_9001_MR1__OASIS_OAS1_9002_MR1'       # a synthetic name: no data of this name exists
SYN_FIXED, SYN_MOVING = 'SYN1', 'SYN2'                    # the ids pair_files writes into the synthetic manifest (contract C23.6-3 S class, revisions 6f, 6g)
STAGE_REL = 'data/three_d/stage_L2R_val_v1'               # where setup puts the synthetic pair inside the scratch copy
MARKER = 'e2e_control_scratch.json'                       # written by setup at the scratch root (revision 6g): the constants and a random nonce
REAL_DATA_ROOTS = (ROOT/'data', Path('/mnt/d/data'), Path('/home/rmcfz/code/python_projects/seq_vp_3d'))     # data/, /mnt/d/data, seq_vp_3d (revision 6g)
SHAPE_XYZ = (38, 36, 34)
OUTER_STEPS = 3
HOLD = 47.
TIMING = ('seconds', 'utc', 'elapsed', 'time', 'tick', 'duration')
RUNS = ('plain', 'injected', 'injected_hold')
PATCHES = ("e48.worker device 'cuda' -> 'cpu' (F/scripts/run_stage_3d_e48_pg.py:821)", f'e19.OUTER_STEPS = {OUTER_STEPS}', 'e48.prewarm_jit and e48.first_use_prewarm: empty functions (:850-855, GPU only, before the task clock)')
NOT_REACHED = ('the GPU path', 'prewarm_jit and first_use_prewarm with a real device', 'systemd-run and the watchdog', 'a wall budget that ends the run (outer_gate)', 'real images')


def peak():
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    return dict(peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, rss_anon_end_gib=int(status['RssAnon'].split()[0])/2**20)


def child_env(cache):
    return dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', MALLOC_ARENA_MAX='1',
                JD_CURL_JIT_CACHE=str(cache/'jit'), JD_CURL_FIRST_USE_CACHE=str(cache/'first'), NUMBA_CACHE_DIR=str(cache/'numba'))


def driver(mode, run):
    """A new process: e48.run(run) on the frozen module of the snapshot."""
    run = Path(run).resolve()
    snapshot = run/'snapshot'
    sys.path.insert(0, str(snapshot))
    import scripts.run_stage_3d_e48_pg as e48
    import scripts.run_stage_3d_e19_mri as e19
    e19.OUTER_STEPS = OUTER_STEPS
    original = e48.worker

    def worker(task, out, data, *, device='cuda', **options):
        return original(task, out, data, device='cpu', **options)
    e48.worker = worker
    e48.prewarm_jit = lambda *a, **k: 0.
    e48.first_use_prewarm = lambda output, cfg: (None, None)
    calls = None
    if mode != 'plain':
        import importlib.util
        spec = importlib.util.spec_from_file_location('seed_inject_copy', run/'seed_inject.py')
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        entry = wrapper.read_entry(run/'seed_entry.json')
        field = wrapper.load_field(entry)
        from jd_curl.three_d import preopt
        if snapshot.resolve() not in Path(preopt.__file__).resolve().parents:
            raise SystemExit(f'jd_curl.three_d.preopt comes from {preopt.__file__}')
        calls = wrapper.install(preopt, entry, field).calls
        del field
    e48.run(run)
    print(json.dumps(dict(mode=mode, calls=None if calls is None else len(calls), **peak())))


def scratch_problems(base, run, pair, document, nonce):
    """The narrow synthetic rule of contract C23.6-3 (S class; revisions 6f, 6g) for a manifest whose ids are SYN_FIXED, SYN_MOVING: [] when the run is this control's
    own, or what does not hold.  Own: the marker setup wrote at the scratch root is there and holds PAIR, the two ids and this control's nonce; the run is <base>/plain;
    the task pair, the manifest's pair and PAIR agree; no directory from the scratch down to the pair (the scratch copy's data/ and the run's inputs/) is a symbolic
    link or lies in a real data root (data/, /mnt/d/data, seq_vp_3d), and the manifest is the one pair_files wrote into this scratch, byte for byte; the two
    pair.npz files are no links, in the scratch and in no real data root (review m11).  Reads the marker and the two manifests; only when every condition above
    holds, it reads and hashes the run's own pair.npz against the manifest's pair_sha256 (review m11; follow-up review n12)."""
    problems = []
    marker = base/MARKER
    try:
        mark = json.loads(marker.read_text())
    except (OSError, ValueError):
        mark = None
    want = dict(pair=PAIR, fixed_subject=SYN_FIXED, moving_subject=SYN_MOVING)
    if not isinstance(mark, dict) or {k: mark.get(k) for k in want} != want or not isinstance(nonce, str) or mark.get('nonce') != nonce:
        problems.append(f'the scratch marker {marker} is missing or does not hold this control\'s pair, ids and nonce')
    if run != base/'plain':
        problems.append(f'the run {run} is not {base/"plain"}')
    if not (pair == PAIR == document.get('pair')):
        problems.append(f"the task pair {pair}, the manifest's pair {document.get('pair')} and this control's pair {PAIR} differ")
    real = [r.resolve() for r in REAL_DATA_ROOTS if r.exists()]
    for directory in (base/'frozen_copy'/'data', base/'frozen_copy'/STAGE_REL, base/'frozen_copy'/STAGE_REL/PAIR, run/'inputs', run/'inputs'/pair):
        resolved = directory.resolve()
        if directory.is_symlink() or not resolved.is_relative_to(base) or any(resolved == r or r in resolved.parents for r in real):
            problems.append(f'{directory} is a link or lies outside the scratch / in a real data root')
    written = base/'frozen_copy'/STAGE_REL/PAIR/'manifest.json'
    manifest = run/'inputs'/pair/'manifest.json'
    if not written.is_file() or manifest.is_symlink() or written.read_bytes() != manifest.read_bytes():
        problems.append(f'the manifest is not the one pair_files wrote into this scratch ({written})')
    # SG code review m11: the two pair.npz files themselves -- not links, inside the scratch, in no real data root; and, only when everything above holds
    # (so that nothing but this scratch's own file is read), the run's copy is the file the manifest names (pair_sha256)
    archives = (base/'frozen_copy'/STAGE_REL/PAIR/'pair.npz', run/'inputs'/pair/'pair.npz')
    for archive in archives:
        resolved = archive.resolve()
        if not archive.is_file() or archive.is_symlink() or not resolved.is_relative_to(base) or any(resolved == r or r in resolved.parents for r in real):
            problems.append(f'{archive} is missing, a link, or outside the scratch / in a real data root')
    if not problems and hashlib.sha256(archives[1].read_bytes()).hexdigest() != document.get('pair_sha256'):
        problems.append(f"{archives[1]} is not the pair.npz of the manifest's pair_sha256")
    return problems


def check_run_subjects(base, run, nonce=None):
    """Contract C23.6-2 / C23.6-3 (SG; revisions 6f, 6g: field_process is S class, a narrow relaxation of the C23.6-2 row), before pair.npz of the run directory is
    read.  The manifest's ids are the module constants (SYN_FIXED, SYN_MOVING) -> admitted only when scratch_problems finds nothing (this control's own scratch, its
    marker and nonce, no link to real data); refused otherwise.  Any other ids go through the bayesreg guard (bayesreg.io.guard.check_pair_subjects, C23.4-2, without
    an unseal object, on the repository's development allowlist) in a process of its own, so that this process's jd_curl stays the snapshot's (field_process checks
    where preopt comes from).  A manifest without the two ids is refused.  SystemExit for any refusal, before pair.npz is loaded; reads the protocol, the marker
    and the manifests, and for the synthetic pair -- only when every other condition holds -- hashes the run's own pair.npz (scratch_problems; follow-up review
    n12).  Returns the record of the rule that admitted the run."""
    base, run = Path(base).resolve(), Path(run).resolve()
    protocol = json.loads((run/'protocol.json').read_text())
    pair = protocol['task']['pair']
    manifest = run/'inputs'/pair/'manifest.json'
    document = json.loads(manifest.read_text())
    ids = [document.get('fixed_subject'), document.get('moving_subject')] if isinstance(document, dict) else [None, None]
    if not all(isinstance(s, str) for s in ids):
        raise SystemExit(f'{manifest}: refused (C23.6-2): the manifest has no fixed_subject / moving_subject')
    if ids == [SYN_FIXED, SYN_MOVING]:
        problems = scratch_problems(base, run, pair, document, nonce)
        if problems:
            raise SystemExit(f'{manifest}: the synthetic ids {ids} refused (C23.6-3 S, e2e_control, revision 6g): {"; ".join(problems)}')
        return dict(rule="C23.6-3 S (e2e_control, revisions 6f, 6g): this control's own synthetic pair in its own scratch", subjects=ids, manifest=str(manifest))
    code = ('import sys\nfrom bayesreg.io.guard import check_pair_subjects, load_dev_allowlist\n'
            'check_pair_subjects(sys.argv[1], sys.argv[2], load_dev_allowlist(sys.argv[3]))\n')
    done = subprocess.run([sys.executable, '-B', '-c', code, ids[0], ids[1], str(ROOT/'configs/bayesreg/dev_subjects.json')], env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True,
                          text=True)
    if done.returncode != 0:
        raise SystemExit(f'{manifest}: the subjects are refused by the bayesreg guard (C23.6-2): {done.stderr.strip()[-600:]}')
    return dict(rule='C23.4-2 (bayesreg.io.guard.check_pair_subjects in a process of its own, no unseal object, the repository allowlist)', subjects=ids, manifest=str(manifest))


def field_process(base, nonce=None):
    """The frozen pre-optimizer's field, with the worker's arguments, in a process of its own: written for the injection, with each run's entry record.  The subjects
    are checked first (check_run_subjects with this control's nonce; C23.6-3 S, revision 6g): nothing is read before."""
    base = Path(base)
    run = base/'plain'
    check_run_subjects(base, run, nonce)
    sys.path.insert(0, str(run/'snapshot'))
    import numpy as np
    from jd_curl.three_d import preopt
    from jd_curl.three_d.grid import make_grid_3d
    protocol = json.loads((run/'protocol.json').read_text())
    pair = protocol['task']['pair']
    with np.load(run/'inputs'/pair/'pair.npz') as z:
        fixed, moving = z['fixed'], z['moving']
    manifest = json.loads((run/'inputs'/pair/'manifest.json').read_text())
    nz, ny, nx = manifest['shape_zyx']
    grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')
    pre = protocol['pre_optimizer']
    config = preopt.PreoptConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pre.items() if k not in ('policy', 'taper')})
    raw, record = preopt.preoptimize(fixed, moving, grid, config, device='cpu', execution=preopt.PreoptExecution(parallel_setup=True, analytic_grad=True))
    np.savez(base/'field.npz', u=np.array(raw))
    sys.path.insert(1, str(HERE))
    import run_seed as RUN
    shape = tuple(manifest['shape_zyx'])
    sha = hashlib.sha256(np.asarray(raw).tobytes()).hexdigest()
    for name, hold in (('injected', 0.), ('injected_hold', HOLD), ('handover', 0.)):
        target = base/name
        (target/'seed_entry.json').write_text(json.dumps(dict(arm='S_P', pair=pair, field_path=str(base/'field.npz'), field_sha256=RUN.sha_file(base/'field.npz'), u_sha256=sha, source_run_id='synthetic', oracle=False, charge_seconds=hold, shape_zyx=list(shape),
                                                              tapered_sha256=RUN.boundary_taper_sha256(raw, shape), entry_metrics=None)))
        (target/'seed_inject.py').write_bytes((HERE/'inject_run.py').read_bytes())
    print(json.dumps(dict(dtype=str(raw.dtype), shape=list(raw.shape), max_abs=float(np.abs(raw).max()), seconds=record['seconds'], **peak())))


def setup(base, cache, nonce):
    """The scratch marker (C23.6-3 S, revision 6g: the constants and this control's nonce, at the scratch root), the scratch copy, the synthetic pair, prepare and
    check_prepared for each run directory."""
    sys.path.insert(0, str(HERE))
    import run_seed as RUN
    (base/MARKER).write_text(json.dumps(dict(pair=PAIR, fixed_subject=SYN_FIXED, moving_subject=SYN_MOVING, nonce=nonce)))
    copy = base/'frozen_copy'
    copy.mkdir()
    n_files = RUN.frozen_copy(copy)
    stage = copy/'data/three_d/stage_L2R_val_v1'
    (stage/PAIR).mkdir(parents=True)
    env = child_env(cache)
    made = subprocess.run([RUN.PY, '-B', str(Path(__file__).resolve()), '--pair-files', str(copy), str(stage/PAIR)], env=env, capture_output=True, text=True)
    if made.returncode:
        raise SystemExit(made.stdout+made.stderr)
    (stage/'selection.json').write_text(json.dumps(dict(pairs=[dict(pair=PAIR)])))
    spec = __import__('importlib.util').util.spec_from_file_location('_seed_check_prepared', ROOT/'experiments/d0/check_prepared.py')
    check_prepared = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(check_prepared)
    protocols = {}
    for name in RUNS+('handover',):
        run = base/name
        done = subprocess.run([RUN.PY, '-B', str(copy/'scripts/run_stage_3d_e48_pg.py'), 'prepare', str(run), *RUN.OPTS, '--stage', 'SEED-S_P', '--pair', PAIR], cwd=base, env=env, capture_output=True, text=True)
        if done.returncode:
            raise SystemExit(f'prepare {name}: {done.stderr[-800:]}')
        checked = check_prepared.check(run, PAIR, 'SEED-S_P', RUN.sha_file(stage/PAIR/'pair.npz'), RUN.sha_file(stage/PAIR/'manifest.json'), b1_entry=RUN.b1_entry('S_P'))
        if not checked['ok']:
            raise SystemExit(f"check_prepared {name}: {checked['problems']}")
        protocols[name] = json.loads((run/'protocol.json').read_text())
    first = protocols['plain']
    return dict(frozen_files=n_files, seconds=first['task']['wall_seconds'], geometry=first['geometry_weight_levels'], throughput=first['throughput'], pre_optimizer_iterations=first['pre_optimizer']['iterations'], check_prepared='ok for all')


def pair_files(copy, target):
    """A process with the frozen copy on its path: the synthetic pair.npz and manifest.json."""
    sys.path.insert(0, copy)
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from jd_curl.three_d.grid import make_grid_3d
    from jd_curl.three_d.nifti import dice_labels, file_sha256
    target = Path(target)
    nx, ny, nz = SHAPE_XYZ
    grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')
    rng = np.random.default_rng(20261104)
    fixed = gaussian_filter(rng.uniform(size=grid.shape), 2.)
    moving = np.roll(fixed, 1, axis=2)
    fixed, moving = ((a-a.min())/(a.max()-a.min()) for a in (fixed, moving))
    labels = (np.arange(np.prod(grid.shape)).reshape(grid.shape) % 5).astype(float)
    arrays = dict(fixed=fixed, moving=moving, initial_u=np.zeros(grid.shape+(3,)))
    for kind in ('seg4', 'seg35'):
        arrays['fixed_'+kind] = labels
        arrays['moving_'+kind] = np.roll(labels, 1, axis=0)
    np.savez(target/'pair.npz', **arrays)
    baseline = {k: dice_labels(arrays['fixed_'+k], arrays['moving_'+k], labels=list(range(1, n+1))) for k, n in (('seg4', 4), ('seg35', 35))}
    (target/'manifest.json').write_text(json.dumps(dict(pair=PAIR, shape_zyx=list(grid.shape), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], weights={'lam': .001}, pair_sha256=file_sha256(target/'pair.npz'), fixed_subject=SYN_FIXED, moving_subject=SYN_MOVING,
                                                    baseline_label_metrics=baseline, preprocessing='synthetic'), default=float))


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if not any(t in k for t in TIMING)}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def summary(run):
    import numpy as np
    protocol = json.loads((run/'protocol.json').read_text())
    task = run/protocol['task']['name']

    def read(name):
        return json.loads((task/name).read_text())
    with np.load(task/'final_native.npz') as z:
        final = hashlib.sha256(z['u'].tobytes()).hexdigest()
    levels = read('levels.json')
    done = next(e for e in levels if e['event'] == 'preoptimizer_done')
    s = read('summary.json')
    return dict(final=final, commits=strip(read('commits.json')), history=strip(read('history.json')), inner=strip(read('inner_attempts.json')), attempts=[strip({k: e[k] for k in ('alpha', 'certificate_passed', 'certificate')}) for e in levels if e['event'] == 'transfer_attempt'],
                raw=done['preoptimizer']['raw_hash'], seed=done['preoptimizer']['seed_hash'], taper=done['preoptimizer']['taper_L'], native=s['native_commits'], status=s['status'], role=s['metrics_role'], visible=done['visible_seconds'], policy=done['preoptimizer']['policy'], e2e=s['e2e_seconds'],
                record_seconds=done['preoptimizer'].get('seconds'))


def control(base, cache):
    sys.path.insert(0, str(HERE))
    import run_seed as RUN
    tick = time.perf_counter()
    nonce = secrets.token_hex(16)                                   # C23.6-3 S (revision 6g): ties the field process to the scratch this control wrote
    result = dict(role='control evidence on a synthetic volume, not a run of the plan', pair=PAIR, shape_xyz=list(SHAPE_XYZ), outer_steps=OUTER_STEPS, hold_seconds=HOLD, patches=list(PATCHES), not_reached=list(NOT_REACHED), setup=setup(base, cache, nonce))
    env = child_env(cache)
    me = str(Path(__file__).resolve())
    done = subprocess.run([RUN.PY, '-B', me, '--field', str(base), nonce], cwd=base, env=env, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f'the field: {done.stdout[-1000:]} {done.stderr[-1500:]}')
    result['field'] = json.loads(done.stdout.strip().splitlines()[-1])
    processes = {}
    for name, mode in zip(RUNS, ('plain', 'injected', 'injected')):
        started = time.perf_counter()
        done = subprocess.run([RUN.PY, '-B', me, '--driver', mode, str(base/name)], cwd=base/name/'snapshot', env=env, capture_output=True, text=True)
        (base/f'{name}.log').write_text(done.stdout+done.stderr)
        if done.returncode:
            raise SystemExit(f'{name}: {done.stdout[-1000:]} {done.stderr[-1500:]}')
        processes[name] = dict(json.loads(done.stdout.strip().splitlines()[-1]), seconds=time.perf_counter()-started)
    result['processes'] = processes
    a, b, c = (summary(base/n) for n in RUNS)
    keys = ('final', 'commits', 'history', 'inner', 'attempts', 'raw', 'seed', 'taper', 'native', 'status', 'role')
    traces = {}
    for name in RUNS[1:]:
        trace, problems = RUN.injection_trace(base/name)
        traces[name] = dict(passed=trace['passed'], problems=problems, accepted_alpha=trace['accepted_alpha'], entry_D_at_accepted_alpha=trace['entry_D_at_accepted_alpha'], transfer_status=trace['transfer_status'])
    shutil.copyfile(base/'injected/seed_entry.json', base/'plain/seed_entry.json')
    plain_trace = RUN.injection_trace(base/'plain')[1]
    classified = {}
    for name in ('plain', 'injected_hold'):
        run = base/name
        (run/'exit_code').write_text('0\n')
        (run/'supervisor_started_utc.txt').write_text('2026-11-04T00:00:00Z\n')
        (run/'supervisor_finished_utc.txt').write_text('2026-11-04T00:01:00Z\n')
        subprocess.run([RUN.PY, '-B', str(ROOT/'experiments/d0/classify_run.py'), str(run), '--watchdog', '1100', '--dxg', '0', '--oom', '0', '--out', str(run/'classification.json')], capture_output=True, text=True)
        record = json.loads((run/'classification.json').read_text())
        classified[name] = dict(cls=record['class'], kind=record['kind'])
    handover = subprocess.run([RUN.PY, '-B', str(base/'handover/seed_inject.py'), str(base/'handover')], cwd=base/'handover/snapshot', env=env, capture_output=True, text=True)
    (base/'handover/run.log').write_text(handover.stdout+handover.stderr)
    (base/'handover/exit_code').write_text(f'{handover.returncode}\n')
    subprocess.run([RUN.PY, '-B', str(ROOT/'experiments/d0/classify_run.py'), str(base/'handover'), '--watchdog', '1100', '--dxg', '0', '--oom', '0', '--out', str(base/'handover/classification.json')], capture_output=True, text=True)
    record = json.loads((base/'handover/classification.json').read_text())
    result.update(checks=dict(
        plain_equals_injected={k: a[k] == b[k] for k in keys}, plain_equals_injected_hold={k: a[k] == c[k] for k in keys},
        field_is_the_plain_runs=a['raw'] == json.loads((base/'injected/seed_entry.json').read_text())['u_sha256'],
        seed_is_one_taper=a['seed'] == json.loads((base/'injected/seed_entry.json').read_text())['tapered_sha256'],
        policies=[a['policy'], b['policy'], c['policy']] == ['fp32-preopt-v1', RUN.inject_run.POLICY, RUN.inject_run.POLICY],
        injected_once=processes['injected']['calls'] == 1 and processes['injected_hold']['calls'] == 1,
        traces_passed=all(t['passed'] for t in traces.values()), plain_fails_the_trace=any('did not run' in p for p in plain_trace),
        hold_is_in_the_clock=c['visible']-b['visible'] >= HOLD-1. and c['e2e']-b['e2e'] >= HOLD-1. and c['record_seconds'] >= HOLD,
        classified_ok=all(v['cls'] == 'ok' for v in classified.values()),
        handover_reaches_the_frozen_entry_and_stops_before_the_clock=handover.returncode != 0 and 'the frozen entry starts' in handover.stdout and not (base/'handover/started.json').exists() and record['class'] == 'unsure' and record['kind'] == 'before_task_clock'),
        native_commits=a['native'], status=a['status'], metrics_role=a['role'], traces=traces, plain_trace=plain_trace, classified=classified, visible_seconds_of_preoptimizer_done=dict(zip(RUNS, (a['visible'], b['visible'], c['visible']))), e2e_seconds=dict(zip(RUNS, (a['e2e'], b['e2e'], c['e2e']))),
        injected_record_seconds=dict(injected=b['record_seconds'], injected_hold=c['record_seconds']), handover=dict(exit=handover.returncode, cls=record['class'], kind=record['kind'], last_line=(handover.stderr.strip().splitlines() or [''])[-1][:300]), seconds=time.perf_counter()-tick)
    flat = [all(v.values()) if isinstance(v, dict) else bool(v) for v in result['checks'].values()]
    result['passed'] = all(flat)
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['--driver']:
        return driver(argv[1], argv[2])
    if argv[:1] == ['--field']:
        return field_process(argv[1], argv[2] if len(argv) > 2 else None)
    if argv[:1] == ['--pair-files']:
        return pair_files(argv[1], argv[2])
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--cache', type=Path, default=None)
    a = ap.parse_args(argv)
    if a.scratch.exists():
        raise SystemExit(f'{a.scratch} exists (nothing is overwritten)')
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})        # one core, the children too; the machine has real runs on it
    a.scratch.mkdir(parents=True)
    try:
        result = control(a.scratch, a.scratch/'cache' if a.cache is None else a.cache)
    finally:
        shutil.rmtree(a.scratch, ignore_errors=True)
    text = json.dumps(result, indent=1, default=float, ensure_ascii=False)
    print(text)
    if a.out:
        a.out.write_text(text+'\n')
    return 0 if result['passed'] else 8


if __name__ == '__main__':
    sys.exit(main())
