"""MC0 (runs/20261103-mc0/任务书.md, plan.md): the driver of the uncertified multi-channel ceiling probe.  An oracle probe on the first 5 development pairs;
its fields are not a method and enter no result table.  Channels and optimizer: mc0_core.py.  Reading: decide_mc0.py.

    python -B experiments/mc0/run_mc0.py --dry-run
        prints all 45 runs and checks the inputs: real files are read up to the shapes of the pair.npz members; no optimization, nothing written.
    python -B experiments/mc0/run_mc0.py queue [--root DIR] [--commit SHA] [--wait-minutes 30]
        the lead's start, in the unit printed by the dry run.  Code: a detached worktree $ROOT/code of --commit (default HEAD; tracked files clean).  First the bitwise K = 1
        check on the GPU (unit tag-mc0-k1check; a mismatch stops the queue).  Then every run in plan order, one unit tag-mc0-<arm>-<tier>-<ffff>-<mmmm> at a time (MemoryMax 9G),
        each behind the gate: no $ROOT/hold, no other tag-* or seq-vp-* unit, 1-minute load < 1.5, /home free >= 100 GB, GPU free >= 15 GiB; waited for --wait-minutes, then STOP.
        A run directory that exists is never rerun, whatever its state; failures are kept.  An arm stops on a device peak above 14 GiB, a device out-of-memory error, or a K = 35 run
        over 15 minutes; its finished runs stay.  At 2.5 h of run units no new run starts.  3 failures in a row, or a P0 whose images differ from A5's inputs: STOP (no evaluation, no reading:
        by hand `evaluate` with PYTHONPATH=<frozen>:<code>, then `decide`).  Finally one CPU unit tag-mc0-eval, and the reading; an evaluation that failed or was cut still gets the reading
        (a field without metrics.json counts as not read), then STOP with exit 8.
    run_mc0.py run-one --arm P1 --tier long --pair PAIR --out DIR [--device cuda] [--code-sha SHA]         one run: field.npz, run.json
    run_mc0.py evaluate --root DIR       every field without metrics.json: taper, .9^k certificate (CPU FP64, a record only), the evaluate.py metrics (one process: numba compiles once)
    run_mc0.py decide --root DIR         analysis/decision_mc0.json, long.csv, summary_mc0.md
    run_mc0.py self-check-k1 --device cpu|cuda [--out JSON]                                                the copied loop against the frozen one on a small synthetic volume

run-one, evaluate and self-check-k1 need PYTHONPATH=<frozen>:<code> (the unit sets it).  Exit codes: 0 done, 2 usage or another driver on the root, 8 STOP or a failed check.
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import decide_mc0 as decision  # noqa: E402
import mc0_core as core  # noqa: E402

PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
SMI = '/usr/lib/wsl/lib/nvidia-smi'
DEFAULT_ROOT = Path('/home/rmcfz/seq_vp_tag_results/mc0_v1')
QUEUE_UNIT, K1_UNIT, EVAL_UNIT = 'tag-mc0-queue', 'tag-mc0-k1check', 'tag-mc0-eval'
VRAM_LIMIT_BYTES = 14*2**30
GPU_CAP_SECONDS = 2.5*3600
RUN_RUNTIME_MAX = 1200                          # the unit's own limit; the wall cap of the loop is core.RUN_CAP_SECONDS
EVAL_RUNTIME_MAX = 3*3600
LOAD_LIMIT, HOME_FLOOR_GB, GPU_FREE_MIB = 1.5, 100, 15*1024
POLL_SECONDS = 30
MAX_CONSECUTIVE_FAILURES = 3
K35 = 35


class Stop(Exception):
    """The queue stops here; everything so far stays."""


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False)+'\n')
    tmp.replace(path)


read_json = decision.read_json


# ---- start strings ------------------------------------------------------------------------------------------------------------------------
def check_launch(argv):
    """The start string of a process never holds the word of the test runner (the machine's gates look for it; gate-12 incident)."""
    if 'pytest' in ' '.join(str(a) for a in argv):
        raise ValueError(f'start string with a forbidden word: {argv}')
    return argv


def unit_argv(unit, code, inner, *, log, gpu=True, numba=1, quota=800, runtime=RUN_RUNTIME_MAX, script='experiments/mc0/run_mc0.py', memory_max='9G', nice=None):
    """One unit of the queue.  script, memory_max, nice: MC0P's entry passes its own (run_mc0p.py); the defaults are MC0's string, unchanged."""
    env = dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, NUMBA_NUM_THREADS=numba, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=f'{core.FROZEN}:{code}')
    if not gpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return check_launch(['systemd-run', '--user', f'--unit={unit}', '--wait', '--collect', f'--working-directory={code}', f'--property=MemoryMax={memory_max}', '--property=MemorySwapMax=4G',
                         *([] if nice is None else [f'--nice={nice}']), f'--property=CPUQuota={quota}%', f'--property=RuntimeMaxSec={runtime}', f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}',
                         *[f'--setenv={k}={v}' for k, v in env.items()], PY, '-B', f'{code}/{script}', *inner])


def run_argv(cell, code, root, sha='n/a'):
    d = core.run_dir(root, cell)
    return unit_argv(cell.unit, code, ['run-one', '--arm', cell.arm, '--tier', cell.tier, '--pair', cell.pair, '--out', str(d), '--code-sha', sha], log=d/'log.txt')


def k1_argv(code, root):
    return unit_argv(K1_UNIT, code, ['self-check-k1', '--device', 'cuda', '--out', f'{root}/k1_check_gpu.json'], log=f'{root}/k1_check_gpu.log', runtime=600)


def eval_argv(code, root):
    return unit_argv(EVAL_UNIT, code, ['evaluate', '--root', str(root)], log=f'{root}/eval.log', gpu=False, numba=8, runtime=EVAL_RUNTIME_MAX)


def queue_launch(root=DEFAULT_ROOT, repo=core.MAIN_REPO):
    """The lead's start string of the queue."""
    return check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT}', '--collect', f'--working-directory={repo}', '--property=MemoryMax=9G', '--setenv=PYTHONDONTWRITEBYTECODE=1',
                         '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', PY, '-B', 'experiments/mc0/run_mc0.py', 'queue', '--root', str(root)])


# ---- one run ---------------------------------------------------------------------------------------------------------------------------------
def read_pair(pair, labels):
    """The pair through bayesreg.io.pair.load_pair (allowlist guard, manifest, sha256, firewall view) and, this being an oracle probe, the named truth label members."""
    core.require_pair(pair)
    from bayesreg.io.pair import load_pair
    path = core.STAGE/pair/'pair.npz'
    pin = load_pair(path, allowlist=core.ALLOWLIST)
    with np.load(path) as z:
        return pin, {k: np.array(z[k]) for k in labels}


class VramLimit(Exception):
    """The device peak went over the limit (the arm stops)."""


def vram_guard(torch, device, limit=VRAM_LIMIT_BYTES):
    """The loop's observer: checks the reserved peak after every iteration (no device sync)."""
    if torch.device(device).type != 'cuda':
        return None

    def observer(row):
        peak = torch.cuda.max_memory_reserved()
        if peak > limit:
            raise VramLimit(f"device peak {peak/2**30:.2f} GiB > {limit/2**30:.0f} GiB at level {row['level']} iteration {row['iteration']}")
    return observer


def run_one(o):
    from time import perf_counter
    t0 = perf_counter()
    cell = core.require_cell(o.arm, o.tier, o.pair)
    fz = core.frozen_api()
    import torch
    out = Path(o.out)
    if (out/'started.json').exists():
        raise SystemExit(f'{out}: this run has started before; a run is never rerun')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'started.json', dict(run_id=cell.run_id, utc=utc(), argv=sys.argv, pid=os.getpid()))
    pin, labs = read_pair(cell.pair, ('fixed_seg35', 'moving_seg35'))
    config = fz.PreoptConfig(**core.tier_config(cell.tier))
    status, reason, u, record, scale, seconds_channels, level_norms = 'done', None, None, None, None, None, None
    try:
        stacks, scale = core.prepare_stacks(cell.arm, pin.fixed, pin.moving, labs['fixed_seg35'], labs['moving_seg35'], o.device, workers=core.SETUP_WORKERS)
        reference = None if cell.arm == 'P0' else core.p0_level_norms(pin.fixed, pin.moving, labs['fixed_seg35'], labs['moving_seg35'], pin.grid, o.device, config.sigma_grad, workers=core.SETUP_WORKERS)
        del labs
        if cell.arm == 'P0':
            scale['images_equal_a5_inputs'] = core.equals_a5_input(cell.pair, stacks[0][0], stacks[1][0])
        seconds_channels = perf_counter()-t0
        u, record = core.optimize_k(stacks, pin.grid, config, device=o.device, started=t0, observer=vram_guard(torch, o.device), workers=core.SETUP_WORKERS, record_level_norms=True)
        del stacks
        level_norms = core.level_norm_record(record['level_g_norm'], reference)
        if record['stopped'] is not None:
            status, reason = 'time_cap', f"the wall cap {core.RUN_CAP_SECONDS} s at {record['stopped']}"
    except VramLimit as error:
        status, reason = 'vram_stop', str(error)
    except torch.cuda.OutOfMemoryError as error:
        status, reason = 'cuda_oom', str(error)[:500]
    field = None
    if u is not None:
        np.savez(out/'field.npz', u=u)
        field = dict(file='field.npz', sha256=core.sha_file(out/'field.npz'), u_sha256=hashlib.sha256(u.tobytes()).hexdigest())
    cuda = torch.device(o.device).type == 'cuda'
    write_json(out/'run.json', dict(
        schema='bayesreg.mc0.run.v1', run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, k=core.ARMS[cell.arm]['k'], oracle=True, status=status, reason=reason, role='not a method, not in any result table',
        config=core.tier_config(cell.tier), scale=scale, level_norms=level_norms, optimizer=record, field=field, taper=core.TAPER, code_sha=o.code_sha, frozen_commit=core.FROZEN_COMMIT, random_seed='none: no random step',
        data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], fixed_subject=pin.fixed_subject, moving_subject=pin.moving_subject, stage=str(core.STAGE)),
        env=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if cuda else 'cpu'),
        device_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None, device_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if cuda else None,
        host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, seconds_channels=seconds_channels, seconds_total=perf_counter()-t0, ended_utc=utc()))
    return 0


# ---- evaluation --------------------------------------------------------------------------------------------------------------------------------
def group_means(seg35_per_label, seg4_per_label, groups):
    """The label groups' mean seg35 over the labels present, the cortex (seg4 label 1) and the white matter (seg4 label 3)."""
    out = {}
    for name, labels in groups.items():
        values = [seg35_per_label[k] for k in labels if seg35_per_label.get(k) is not None]     # dice_labels gives None for a label in neither image
        out[name] = float(np.mean(values)) if values else None
    return dict(out, cortex=seg4_per_label.get(1), white_matter=seg4_per_label.get(3))


def field_metrics(u, labs, spacing, ev):
    """The same functions as experiments/d0/evaluate.py evaluate_run (no HD95: not asked for, the slow one)."""
    d = ev.disp_voxels(u, spacing)
    w35, w4 = ev.warp_labels_l2r(labs['moving_seg35'], d), ev.warp_labels_l2r(labs['moving_seg4'], d)
    seg35 = ev.dice_labels(labs['fixed_seg35'], w35.astype(np.float64), labels=list(range(1, 36)))
    seg4 = ev.dice_labels(labs['fixed_seg4'], w4.astype(np.float64), labels=list(range(1, 5)))
    dice, _ = ev.dice_l2r(labs['fixed_seg35'], labs['moving_seg35'], w35)
    sd, folds, neg = ev.sdlogj_l2r(d)
    sm = ev.brain_smoothness(u, spacing, labs['fixed_seg4'] > 0)
    per35, per4 = {r['label']: r['dice'] for r in seg35['per_label']}, {r['label']: r['dice'] for r in seg4['per_label']}
    return dict(seg35=seg35['macro_dice'], seg4=seg4['macro_dice'], dice_official=dice, sdlogj_official=sd, num_foldings_official=folds, central_det_nonpositive_fraction=neg, **sm,
                **group_means(per35, per4, ev.A5_GROUPS), seg4_per_label=per4, seg35_per_label=per35)


def certify_ladder(u, grid, alphas, certify, config):
    """The transfer ladder as a record: the first alpha of alpha * u that certifies, and the infeasible cells at alpha = 1.  It decides nothing here."""
    attempts, accepted = [], None
    for alpha in alphas:
        r = certify(u if alpha == 1. else alpha*u, grid, config=config)
        attempts.append(dict(alpha=alpha, status=r.status, reason=r.reason, min_jd_lower_bound=r.min_jd_lower_bound, n_cells_checked=r.n_cells_checked, n_infeasible=r.n_infeasible, n_uncertain=r.n_uncertain,
                             certificate_version=r.certificate_version, config=asdict(r.config)))
        if r.status == 'certified_feasible' and r.min_jd_lower_bound is not None and r.min_jd_lower_bound > 0 and r.n_infeasible == 0 and r.n_uncertain == 0:
            accepted = alpha
            break
    return dict(accepted_alpha=accepted, status_at_1=attempts[0]['status'], n_infeasible_at_1=attempts[0]['n_infeasible'], n_uncertain_at_1=attempts[0]['n_uncertain'], attempts=attempts, role='record only')


def certificate_config():
    """The main line's configuration C, passed explicitly: certify_map's own default is not this.  tile_size: the frozen driver's THROUGHPUT_POLICY (a test reads it), as in A5's records."""
    from jd_curl.three_d.feasibility import PrescreenCertificateConfig
    return PrescreenCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=65536)


def evaluate_one(d, cell, labs, pin, fz, ev, certify, config, *, oracle=True, schema='bayesreg.mc0.metrics.v1'):
    run = read_json(d/'run.json')
    if run is None or run.get('field') is None or core.sha_file(d/'field.npz') != run['field']['sha256']:
        raise ValueError('run.json has no field, or field.npz differs from it')
    with np.load(d/'field.npz') as z:
        raw = np.array(z['u'])
    u, _ = fz.boundary_taper(raw, pin.grid, core.TAPER)
    ladder = certify_ladder(u, pin.grid, fz.alphas, certify, config)
    metrics = field_metrics(np.asarray(u), labs, tuple(pin.grid.spacing), ev)
    alpha = ladder['accepted_alpha']
    other = metrics if alpha == 1. else None if alpha is None else field_metrics(alpha*np.asarray(u), labs, tuple(pin.grid.spacing), ev)
    write_json(d/'metrics.json', dict(schema=schema, run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, oracle=oracle, status=run['status'], evaluated_utc=utc(),
                                      field='raw field after the taper of width 4, alpha = 1', tapered_u_sha256=hashlib.sha256(np.asarray(u).tobytes()).hexdigest(),
                                      certificate=ladder, at_accepted_alpha=None if other is None else dict(alpha=alpha, seg35=other['seg35'], seg4=other['seg4'], role='descriptive'),
                                      metrics=metrics, evaluator=dict(script='experiments/d0/evaluate.py', sha256=core.sha_file(core.ROOT/'experiments/d0/evaluate.py'))))


def evaluate_all(o):
    fz = core.frozen_api()
    ev = core.load_module('_mc0_evaluate', 'experiments/d0/evaluate.py')
    from jd_curl.three_d.feasibility import certify_map
    config = certificate_config()
    failures, loaded = 0, {}
    for cell in core.plan_runs():
        d = core.run_dir(o.root, cell)
        if not (d/'field.npz').is_file() or (d/'metrics.json').exists() or (d/'metrics_error.json').exists():
            continue
        try:
            if cell.pair not in loaded:
                loaded = {cell.pair: read_pair(cell.pair, core.PAIR_MEMBERS[2:])}      # pair-major: one pair in memory
            evaluate_one(d, cell, loaded[cell.pair][1], loaded[cell.pair][0], fz, ev, certify_map, config)
        except Exception as error:                     # kept; the others go on; not retried
            write_json(d/'metrics_error.json', dict(run_id=cell.run_id, error=repr(error)[:2000], utc=utc()))
            failures += 1
    return 8 if failures else 0


# ---- self-check ---------------------------------------------------------------------------------------------------------------------------------
def self_check_k1(o):
    """K = 1 of the copy against the frozen default path, on a small synthetic volume: the field's bytes and every level's record but the seconds."""
    fz = core.frozen_api()
    import torch
    from scipy.ndimage import gaussian_filter
    from jd_curl.three_d.grid import make_grid_3d
    grid = make_grid_3d(nx=30, ny=28, nz=26, hx=1., hy=1.25, hz=.75, unit='mm')
    rng = np.random.default_rng(20261103)
    fixed, moving = (gaussian_filter(rng.uniform(size=grid.shape), 1.5) for _ in range(2))
    rows, passed = [], True
    for sigma_field in (1., .5):
        settings = dict(core.PRE_C, iterations=(3, 3, 4, 6), seconds=600., sigma_field=sigma_field)
        config = fz.PreoptConfig(**settings)
        a, ra = fz.preoptimize(fixed, moving, grid, config, device=o.device)
        again, _ = fz.preoptimize(fixed, moving, grid, config, device=o.device)     # frozen against frozen: on a GPU mismatch, tells a wrong copy from an unrepeatable device
        b, rb = core.optimize_k([fixed[None], moving[None]], grid, config, device=o.device, workers=1, record_level_norms=True)
        control, _ = core.optimize_k([fixed[None], moving[None]], grid, fz.PreoptConfig(**dict(settings, lr=settings['lr']*(1+1e-3))), device=o.device, workers=1)

        def strip(record):
            return [{k: v for k, v in level.items() if k != 'seconds'} for level in record['levels']]
        same = dict(field=a.tobytes() == b.tobytes(), levels=strip(ra) == strip(rb), cap_hits=ra['cap_hits'] == rb['cap_hits'], stopped=ra['stopped'] == rb['stopped'],
                    control_differs=control.tobytes() != b.tobytes(), field_moved=bool(np.abs(b).max() > .1),     # the comparison can fail, and the field is not trivial
                    level_norms_recorded=len(rb['level_g_norm']) == 4)                                            # recording them changes no bit
        passed = passed and all(same.values())
        rows.append(dict(sigma_field=sigma_field, same=same, device_repeatable=a.tobytes() == again.tobytes(), u_sha256=hashlib.sha256(b.tobytes()).hexdigest(), max_abs_u=float(np.abs(b).max())))
    result = dict(bitwise=passed, device=o.device, torch=torch.__version__, level='bit for bit against the frozen preoptimize(execution=None), the default path', rows=rows,
                  not_compared='the production path of configuration C (throughput v41, analytic_grad: rounding level by the frozen docstring)', utc=utc())
    print(json.dumps(result, indent=1))
    if o.out:
        write_json(o.out, result)
    return 0 if passed else 8


# ---- the queue ------------------------------------------------------------------------------------------------------------------------------------
class System:
    """What the queue reads from and does to the machine (tests replace it)."""

    def load1(self):
        return float(Path('/proc/loadavg').read_text().split()[0])

    def units(self):
        done = subprocess.run(['systemctl', '--user', 'list-units', '--state=active,activating', '--no-legend', '--plain', 'tag-*', 'seq-vp-*'], capture_output=True, text=True)
        if done.returncode:
            return [f'systemctl failed with exit {done.returncode}']     # the gate stays closed: an error is not an empty machine
        return [line.split()[0] for line in done.stdout.splitlines() if line.strip()]

    def gpu_free_mib(self):
        out = subprocess.run([SMI, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
        try:
            free = float(out.split()[0])
        except (IndexError, ValueError):                # empty, "[N/A]", an error text: no reading; the gate stays closed and, at its time, STOPs with queue_stop.json
            return 0.
        return free if math.isfinite(free) else 0.

    def home_free_gb(self):
        return shutil.disk_usage('/home').free/2**30

    def mem_available_gib(self):
        """MemAvailable of /proc/meminfo; no reading counts as 0: the gate stays closed.  Only a Gate with mem_available_gib asks for it (MC0P)."""
        try:
            kib = next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')).split()[1]
            return float(kib)/2**20
        except (OSError, StopIteration, IndexError, ValueError):
            return 0.

    def host_free_gib(self):
        """/mnt/c free; no reading counts as 0: the gate stays closed.  Only a Gate with host_floor_gib asks for it (SEED19)."""
        try:
            return shutil.disk_usage('/mnt/c').free/2**30
        except OSError:
            return 0.

    def now(self):
        return time.monotonic()

    def sleep(self, seconds):
        time.sleep(seconds)

    def run(self, argv, log):
        with open(log, 'ab') as f:
            return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT).returncode

    def code_checkout(self, root, commit):
        def git(*args, cwd=core.ROOT):
            done = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
            if done.returncode:
                raise Stop(f'git {args}: {done.stderr.strip()}')
            return done.stdout.strip()
        if git('status', '--porcelain', '--untracked-files=no'):
            raise Stop('tracked files are not clean')
        sha = git('rev-parse', '--verify', f"{commit or 'HEAD'}^{{commit}}")
        code = Path(root)/'code'
        if not code.exists():
            git('worktree', 'add', '--detach', str(code), sha)
        if git('rev-parse', 'HEAD', cwd=code) != sha:
            raise Stop(f'{code} is not at {sha}')
        return code, sha


@dataclass(frozen=True)
class Gate:
    """What busy_reason asks of the machine.  The defaults are MC0's gate (plan rule 18); only MC0P's entry passes another one (run_mc0p.GATE, runs/20261104-mc0p/任务书.md)."""
    own_units: tuple = (f'{QUEUE_UNIT}.service',)   # the only tag-* or seq-vp-* units that may be active
    load_limit: float = LOAD_LIMIT
    gpu_free_mib: float = GPU_FREE_MIB
    mem_available_gib: float = None                 # None: not asked
    gpu_hold: object = None                         # a callable: why the GPU must be left alone now, or None; asked for GPU units only
    host_floor_gib: float = None                    # None: not asked.  SEED19's mc0p_v1c: /mnt/c free (run_mc0pc.py --min-host-gib)


MC0_GATE = Gate()


def busy_reason(system, root, gpu, gate=MC0_GATE):
    """Why no unit may start now, or None."""
    if (Path(root)/'hold').exists():
        return 'hold file'
    others = [u for u in system.units() if u not in gate.own_units]
    if others:
        return f'other units: {others}'
    load = system.load1()
    if load >= gate.load_limit:
        return f'1-minute load {load} >= {gate.load_limit}'
    if system.home_free_gb() < HOME_FLOOR_GB:
        return f'/home free < {HOME_FLOOR_GB} GB'
    if gate.mem_available_gib is not None and system.mem_available_gib() < gate.mem_available_gib:
        return f'MemAvailable < {gate.mem_available_gib} GiB'
    if gate.host_floor_gib is not None and system.host_free_gib() < gate.host_floor_gib:
        return f'/mnt/c free < {gate.host_floor_gib} GiB'
    if gpu and system.gpu_free_mib() < gate.gpu_free_mib:
        return f'GPU free < {gate.gpu_free_mib} MiB'
    if gpu and gate.gpu_hold is not None:
        return gate.gpu_hold()
    return None


def wait_for_gate(system, root, gpu, minutes, log, gate=MC0_GATE):
    started = system.now()
    while True:
        reason = busy_reason(system, root, gpu, gate)
        if reason is None:
            return
        if system.now()-started >= minutes*60:
            raise Stop(f'gate closed for {minutes} minutes: {reason}')
        log(f'waiting: {reason}')
        system.sleep(POLL_SECONDS)


def arm_stop_reason(cell, run, unit_seconds, *, k=None, vram_limit=VRAM_LIMIT_BYTES):
    """The brief's guards: a device peak over 14 GiB (any arm), a device out-of-memory error, a K = 35 run over 15 minutes; and a K = 35 run that failed without
    a record (MemoryMax, say: the next pair would fail the same way, and 3 failures in a row would stop the K = 1, 14 arms too; the plan's tool rule).
    k, vram_limit: MC0P's entry passes its arm's K and its 9 GiB; the defaults are MC0's."""
    k = core.ARMS[cell.arm]['k'] if k is None else k
    if run is None and k == K35:
        return f'K = 35 run failed without a record, after {unit_seconds:.0f} s'
    run = run or {}
    peak = run.get('device_peak_reserved_gib')
    if run.get('status') in ('vram_stop', 'cuda_oom'):
        return f"{run['status']}: {run.get('reason')}"
    if peak is not None and peak*2**30 > vram_limit:
        return f'device peak {peak:.2f} GiB'
    if k == K35 and (run.get('status') == 'time_cap' or unit_seconds > core.RUN_CAP_SECONDS):
        return f'K = 35 run: {unit_seconds:.0f} s, status {run.get("status")}'
    return None


def ledger(root, plan=None):
    """What the run directories say: run-unit seconds, stopped arms, the state of each run.  A directory without queue_done.json is a run the driver lost: counted at the wall cap, never rerun."""
    seconds, stopped, states = 0., {}, {}
    for cell in (core.plan_runs() if plan is None else plan):
        d = core.run_dir(root, cell)
        if not d.exists():
            continue
        done = read_json(d/'queue_done.json')
        if done is None:
            seconds += core.RUN_CAP_SECONDS
            states[cell.run_id] = 'interrupted'
            continue
        seconds += done['unit_seconds']
        states[cell.run_id] = done['state']
        if done.get('arm_stop') and cell.arm not in stopped:
            stopped[cell.arm] = dict(run_id=cell.run_id, reason=done['arm_stop'])
    return dict(run_unit_seconds=seconds, arms_stopped=stopped, states=states)


def queue_body(o, system, root, log):
    code, sha = system.code_checkout(root, o.commit)
    batch = read_json(root/'batch.json')
    if batch is None:
        write_json(root/'batch.json', dict(schema='bayesreg.mc0.batch.v1', code_sha=sha, frozen_commit=core.FROZEN_COMMIT, started_utc=utc(), plan=[c.run_id for c in core.plan_runs()],
                                           caps=dict(run_seconds=core.RUN_CAP_SECONDS, run_units_seconds=GPU_CAP_SECONDS, device_gib=VRAM_LIMIT_BYTES/2**30)))
    elif batch['code_sha'] != sha:
        raise Stop(f"batch.json is at {batch['code_sha']}, the code at {sha}")
    k1 = root/'k1_check_gpu.json'
    if not k1.is_file():
        wait_for_gate(system, root, True, o.wait_minutes, log)
        argv = k1_argv(code, root)
        log(f'launch {K1_UNIT}')
        system.run(argv, root/'k1_check_gpu.systemd_run.log')
    check = read_json(k1)
    if check is None or check.get('bitwise') is not True:
        raise Stop(f'the K = 1 check on the GPU did not pass: {check}')
    failures, end = 0, 'all runs handled'
    for cell in core.plan_runs():
        d = core.run_dir(root, cell)
        if d.exists():
            continue                                    # never rerun
        state = ledger(root)
        if cell.arm in state['arms_stopped']:
            log(f"skip {cell.run_id}: arm stopped ({state['arms_stopped'][cell.arm]['reason']})")
            continue
        if state['run_unit_seconds'] >= GPU_CAP_SECONDS:
            end = f"the cap of {GPU_CAP_SECONDS} s of run units reached ({state['run_unit_seconds']:.0f} s); no new run"
            log(end)
            break
        wait_for_gate(system, root, True, o.wait_minutes, log)
        d.mkdir(parents=True)
        argv = run_argv(cell, code, root, sha)
        (d/'cmd.txt').write_text(shlex.join(argv)+'\n')
        log(f'launch {cell.unit}; load {system.load1()}')
        tick = system.now()
        rc = system.run(argv, d/'systemd_run.log')
        seconds = system.now()-tick
        run = read_json(d/'run.json')
        failed = rc != 0 or run is None
        stop = arm_stop_reason(cell, run, seconds)
        write_json(d/'queue_done.json', dict(run_id=cell.run_id, unit=cell.unit, rc=rc, unit_seconds=seconds, state='failed' if failed else run['status'], arm_stop=stop, ended_utc=utc()))
        log(f"ended {cell.run_id}: rc {rc}, {seconds:.0f} s, {'failed' if failed else run['status']}" + (f'; arm stops: {stop}' if stop else ''))
        if cell.arm == 'P0' and run is not None and (run.get('scale') or {}).get('images_equal_a5_inputs') is False:
            raise Stop(f"{cell.run_id}: P0's images differ from A5's inputs; the comparison would not be against A5's table")
        failures = failures+1 if failed else 0
        if failures >= MAX_CONSECUTIVE_FAILURES:
            raise Stop(f'{failures} failures in a row')
    wait_for_gate(system, root, False, o.wait_minutes, log)
    log(f'launch {EVAL_UNIT}')
    rc = system.run(eval_argv(code, root), root/'eval.systemd_run.log')
    state = ledger(root)
    result = decision.write_analysis(root, stops=state['arms_stopped'])     # also after a failed evaluation: decide_mc0.series counts a field without metrics.json as not read
    write_json(root/'queue_end.json', dict(end=end, eval_rc=rc, reading=result['reading'], reading_basis=result['reading_basis'], ledger=state, utc=utc()))
    log(f"reading: {result['reading']}. Handed to BZ: no next step starts here")
    if rc != 0:
        raise Stop(f'{EVAL_UNIT}: exit {rc} (metrics_error.json in the run directories, or the {EVAL_RUNTIME_MAX} s limit of the unit); the reading is written, the fields without metrics.json count as not read')
    return 0


def queue(o, system=None, *, body=None, prefix='mc0'):
    """body, prefix: MC0P's entry passes its own; the defaults are MC0's."""
    system = System() if system is None else system
    body = queue_body if body is None else body
    root = Path(o.root)
    if not root.name.startswith(prefix):
        print(f'{root}: the root of this probe is named {prefix}*', file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root/'queue.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f'another driver is on {root}', file=sys.stderr)
        return 2

    def log(text):
        with open(root/'queue.log', 'a') as f:
            f.write(f'{utc()} {text}\n')
    try:
        return body(o, system, root, log)
    except Stop as stop:
        log(f'STOP: {stop}')
        write_json(root/'queue_stop.json', dict(reason=str(stop), utc=utc()))
        return 8


# ---- dry run --------------------------------------------------------------------------------------------------------------------------------------
def npz_header(path, member):
    """(shape, dtype) of an npz member, reading only the npy header."""
    with zipfile.ZipFile(path) as z, z.open(member+'.npy') as f:
        version = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = read(f)
    return shape, str(dtype)


def dry_run(root=DEFAULT_ROOT, stage=core.STAGE, out=print):
    plan = core.plan_runs()
    out(f'MC0 dry run ({utc()}): no optimization, nothing written; real files read up to the npy headers of pair.npz.')
    out(f'{len(plan)} runs, root {root}; cap per run {core.RUN_CAP_SECONDS} s, run units {GPU_CAP_SECONDS} s, device {VRAM_LIMIT_BYTES/2**30} GiB')
    out('  # | arm | K | tier | iterations | sigma_field | pair | unit | run_id')
    for i, c in enumerate(plan, 1):
        config = core.tier_config(c.tier)
        out(f"{i:3d} | {c.arm} | {core.ARMS[c.arm]['k']:2d} | {c.tier:5s} | {'/'.join(map(str, config['iterations']))} | {config['sigma_field']} | {c.short} | {c.unit} | {c.run_id}")
    problems = []
    if len(plan) != 45:
        problems.append(f'{len(plan)} runs, not 45')
    selection = json.loads((stage/'selection.json').read_text())
    first = tuple(row['pair'] for row in selection['pairs'][:5])
    out(f"selection.json first 5 == the plan's pairs: {first == core.PAIRS}")
    if first != core.PAIRS:
        problems.append('the first 5 pairs of selection.json differ')
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    allowlist = load_dev_allowlist(core.ALLOWLIST)
    for pair in core.PAIRS:
        d = stage/core.require_pair(pair)
        manifest = json.loads((d/'manifest.json').read_text())
        check_pair_subjects(manifest['fixed_subject'], manifest['moving_subject'], allowlist)
        heads = {m: npz_header(d/'pair.npz', m) for m in core.PAIR_MEMBERS}
        shapes = {tuple(h[0]) for h in heads.values()}
        out(f"{core.short_name(pair)}: {d/'pair.npz'}; subjects on the allowlist ({manifest['fixed_subject']}, {manifest['moving_subject']}); spacing {manifest['spacing_xyz']}; "
            + ', '.join(f'{m} {h[0]} {h[1]}' for m, h in heads.items()))
        if len(shapes) != 1:
            problems.append(f'{pair}: member shapes differ')
    table = core.load_table()
    groups = core.load_groups()
    out(f'colour table sha256 {core.COLOUR_TABLE_SHA256} pinned and read: {len(table)} labels; structure13_v1: {len(groups)} groups; P1 K = {len(groups)+1}, P2, P3 K = {len(core.LABELS_35)}')
    out(f"frozen {core.FROZEN}: preopt.py sha256 equal to the pin: {core.sha_file(core.FROZEN/'jd_curl/three_d/preopt.py') == core.FROZEN_PREOPT_SHA256}; the units' PYTHONPATH puts it first")
    out(f'/home free {shutil.disk_usage("/home").free/2**30:.0f} GB')
    code = root/'code'
    out('--- start strings (the queue: the lead; the others: the queue, one per run; the code at $ROOT/code)')
    out(shlex.join(queue_launch(root)))
    out(shlex.join(k1_argv(code, root)))
    for c in plan[:10]:
        out(shlex.join(run_argv(c, code, root, '<sha>')))
    out(f'... (the other {len(plan)-10} differ only in --arm, --tier, --pair, --out, the unit and the log)')
    out(shlex.join(eval_argv(code, root)))
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv == ['--dry-run']:
        return dry_run()
    p = argparse.ArgumentParser(description='MC0: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(DEFAULT_ROOT))
    q.add_argument('--commit', default=None)
    q.add_argument('--wait-minutes', type=int, default=30)
    r = sub.add_parser('run-one')
    r.add_argument('--arm', required=True)
    r.add_argument('--tier', required=True)
    r.add_argument('--pair', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--device', default='cuda')
    r.add_argument('--code-sha', default='n/a')
    for name in ('evaluate', 'decide'):
        sub.add_parser(name).add_argument('--root', default=str(DEFAULT_ROOT))
    s = sub.add_parser('self-check-k1')
    s.add_argument('--device', default='cpu')
    s.add_argument('--out', default=None)
    o = p.parse_args(argv)
    if o.command == 'queue':
        return queue(o)
    if o.command == 'run-one':
        return run_one(o)
    if o.command == 'evaluate':
        return evaluate_all(o)
    if o.command == 'decide':
        decision.write_analysis(o.root, stops=ledger(o.root)['arms_stopped'])
        return 0
    return self_check_k1(o)


if __name__ == '__main__':
    sys.exit(main())
