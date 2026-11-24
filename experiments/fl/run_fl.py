"""FL (runs/20261105-fl/任务书.md, plan.md): the driver of K-channel T88, more steps, finer sigma_field, local repair, and no scalar native level.  `oracle: false` but for arm O, isolated; not a method: no F_B, no certified descent, only the final field passes the certificate; development pairs only, in no result table.
Arms and guards: fl_core.py.  Channels: fl_channels.py on experiments/ef/fast_channels.py; arm O: fl_oracle.py.  Repair: experiments/ef/repair.py, unchanged.  Reading and the code's choices: decide_fl.py.  Optimizer, evaluation functions, queue skeleton: MC0's, MC0P's and EF's, called; mc0_core.optimize_k has one optional argument for the checkpoints.

    python -B experiments/fl/run_fl.py --dry-run [--pairs rest14 --arm S --steps 2400]
        prints the plan, the estimates of time and disk, checks the inputs: selection.json, the npy headers of fixed, moving, the predictions, the fields of ef_v1 that RF150, RF300 start from (sha256 against their run.json), the records of the controls, the start strings.  No optimization, no truth label, nothing written.
    python -B experiments/fl/run_fl.py queue [--commit SHA] [--wait-minutes 30] [--gap-seconds 5] [--min-host-gib 15] [--through fields|repair|all]
        the lead's start, in the unit the dry run prints.  Code: a detached worktree $ROOT/code of --commit (default HEAD; tracked files clean).  First the checks of the dry run: anything missing stops before batch.json and before any run.
        Phases.  1 every field of the first phase in plan order, one unit tag-fl-<arm>-first5-<ffff>-<mmmm> at a time (MemoryMax 8G, nice 5), each behind the gate: no $ROOT/hold, no other tag-* or seq-vp-* unit, 1-minute load < 1.5, GPU free >= 15 GiB, MemAvailable >= 7 GiB, /home >= 100 GB, /mnt/c >= --min-host-gib; closed for --wait-minutes: STOP.
        Then one CPU unit tag-fl-eval (numba 8), the reading.  2 the repairs the code chose and RF150, RF300, their evaluation, the reading.  3 X with the sigma the code chose, its evaluation, its repair, the evaluation, the reading.  --through stops after a phase.
        A run directory that exists is never rerun, failures are kept.  An arm stops on the 9 GiB device limit, a device out-of-memory error or a run over 15 minutes.  3 failures in a row, a record whose oracle mark is wrong, F's 600 steps not bit for bit ef_v1's F600: STOP.  A STOP in phase 1: no evaluation.  From the first reading on a STOP only ends the later phases: what exists is evaluated, the reading is written again, exit 8.
        An evaluation unit that failed: exit 8 and no choice is written after it (a choice is written once); a restart evaluates what is left (a unit killed half way) and chooses then.  A checkpoint that wrote its metrics error file is not retried, failures are kept: it stays not read, and the choice's candidates say so (n_read).  A choice that is not the file the code wrote, or a driver whose decide_fl.py or fl_core.py is not the batch's commit: STOP.
    python -B experiments/fl/run_fl.py queue --pairs rest14 --arm S --steps 2400 --commit SHA
        the lead's separate start for the other 14 pairs, after he read the first 5: one field per pair (no intermediate checkpoint), the evaluation, a repair per pair, the evaluation, the reading again.  Not part of the default queue.
    run_fl.py run-one --arm F --tier first5 --pair PAIR --out DIR [--device cuda] [--code-sha SHA]      one field run: field_<step>.npz per checkpoint, run.json
    run_fl.py repair-one --arm RF --tier c2400 --pair PAIR --out DIR [--device cuda] [--code-sha SHA]   one repair: repair.npz, repair.json.  No image reaches the solve and no label is read anywhere.
    run_fl.py evaluate                   every checkpoint and repair without metrics: taper, certificate at alpha = 1 (no ladder), the metrics with HD95; with arm O's own reader, the only place the truth labels are read
    run_fl.py decide                     analysis/decision_fl.json, long.csv, summary_fl.md

The only root is /home/rmcfz/seq_vp_tag_results/fl_v1: every entry that writes refuses any other.  run-one, repair-one and evaluate need PYTHONPATH=<frozen>:<code> (the unit sets it).  Exit codes: 0 done, 2 usage, another driver on the root, another root, 8 STOP or a failed check.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent/'ef'))
sys.path.insert(2, str(HERE.parent/'mc0'))

import numpy as np  # noqa: E402

import decide_fl as decision  # noqa: E402
import fast_channels as fast  # noqa: E402
import fl_channels as channels  # noqa: E402
import fl_core as fl  # noqa: E402
import fl_oracle as oracle_arm  # noqa: E402
import mc0_core as core  # noqa: E402
import mc0p_core as mc  # noqa: E402
import repair  # noqa: E402
import run_ef as efrun  # noqa: E402
import run_mc0 as run  # noqa: E402
import run_mc0p as mc0p  # noqa: E402

SCRIPT = 'experiments/fl/run_fl.py'
QUEUE_UNIT, EVAL_UNIT, EVAL_R1_UNIT, EVAL_X_UNIT, EVAL_R2_UNIT, EVAL_REST_UNIT, EVAL_REST_R_UNIT = 'tag-fl-queue', 'tag-fl-eval', 'tag-fl-eval-r1', 'tag-fl-eval-x', 'tag-fl-eval-r2', 'tag-fl-eval-rest', 'tag-fl-eval-rest-r'
MEMORY_MAX, NICE = '8G', 5                      # as EF's: its timings are the ones compared with.  Arm I: the host peak of K = 35 was 5.6-5.7 GiB in ef_v1, K = 36 about 5.8 GiB〔估算〕
EVAL_THREADS, EVAL_QUOTA = 8, 800               # the machine is free here (the gate); a thread count cannot change a certificate
REPAIR_THREADS, REPAIR_QUOTA = 8, 800           # the certificate inside the repair, as EF's
RUN_THREADS = 1                                 # the field runs' numba: no certificate in the unit
GAP_SECONDS = 5                                 # EF's
DEFAULT_MIN_HOST_GIB = 15.                      # the brief
RUN_UNITS_CAP_SECONDS = 5*3600                  # the safety net of the run units over the whole root (MC0's 2.5 h would cut X); the plan: about 2.4 h for the first 5 pairs, about 40 minutes per arm of the other 14
K_STOP = run.K35                                # every arm of this batch stops as K = 35 does, also I with K = 36
GATE = replace(mc0p.RETRY_GATE, own_units=(f'{QUEUE_UNIT}.service',), host_floor_gib=DEFAULT_MIN_HOST_GIB)     # load < 1.5, GPU free >= 15 GiB, MemAvailable >= 7 GiB, /home >= 100 GB
TRUTH_MEMBERS = core.PAIR_MEMBERS[2:]


# ---- start strings ------------------------------------------------------------------------------------------------------------------------
def unit_argv(unit, code, inner, *, log, gpu=True, numba=RUN_THREADS, quota=800, runtime=run.RUN_RUNTIME_MAX):
    return run.unit_argv(unit, code, inner, log=log, gpu=gpu, numba=numba, quota=quota, runtime=runtime, script=SCRIPT, memory_max=MEMORY_MAX, nice=NICE)


def run_argv(cell, code, root, sha='n/a'):
    d = fl.run_dir(root, cell)
    command = 'repair-one' if cell.is_repair else 'run-one'
    return unit_argv(cell.unit, code, [command, '--arm', cell.arm, '--tier', cell.tier, '--pair', cell.pair, '--out', str(d), '--code-sha', sha], log=d/'log.txt', numba=REPAIR_THREADS if cell.is_repair else RUN_THREADS, quota=REPAIR_QUOTA)


def eval_argv(code, root, unit=EVAL_UNIT):
    return unit_argv(unit, code, ['evaluate', '--root', str(root)], log=f'{root}/eval.log', gpu=False, numba=EVAL_THREADS, quota=EVAL_QUOTA, runtime=run.EVAL_RUNTIME_MAX)


def queue_launch(root=None, repo=core.MAIN_REPO, gap=GAP_SECONDS, min_host_gib=DEFAULT_MIN_HOST_GIB, extra=()):
    """The lead's start string of a queue."""
    root = fl.DEFAULT_ROOT if root is None else root
    return run.check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT}', '--collect', f'--working-directory={repo}', f'--property=MemoryMax={MEMORY_MAX}', f'--nice={NICE}', '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1',
                             run.PY, '-B', SCRIPT, 'queue', '--root', str(root), '--gap-seconds', str(gap), '--min-host-gib', f'{min_host_gib:g}', *extra])


def rest_launch(arm, steps, commit, root=None):
    return queue_launch(root, extra=('--pairs', 'rest14', '--arm', arm, '--steps', str(steps), '--commit', commit))


# ---- one field run ------------------------------------------------------------------------------------------------------------------------------
def node_field(u_int, level_grid, h):
    """u float64 (D, H, W, 3) in mm with zero faces: the lines of optimize_k's export (a test compares the last checkpoint with what it returns, bit for bit; so does every run)."""
    import torch
    with torch.no_grad():
        out = np.zeros(level_grid.shape+(3,))
        out[1:-1, 1:-1, 1:-1] = np.moveaxis(u_int.detach().double().cpu().numpy(), 0, -1)*np.array(h)
    return out


class CheckpointKeeper:
    """optimize_k's checkpoint: at the native level's planned steps, field_<step>.npz and its record.  It only reads u_int.
    seconds_task: the task clock at this checkpoint, as one run to this step would have had it: from the run's start to the field on the host, without the writing and hashing of this and the export, writing and hashing of the earlier checkpoints (seconds_io), nor anything else added to `spent` (arm I's record of the image channel's norm): costs of the probe, not of the task."""

    def __init__(self, out, steps, t0, clock=perf_counter):
        self.out, self.steps, self.t0, self.clock = Path(out), tuple(steps), t0, clock
        self.records, self.spent = {}, 0.

    def __call__(self, level, done, u_int, level_grid, h, row):
        if level != fl.NATIVE_LEVEL or done not in self.steps:
            return
        tick = self.clock()
        u = node_field(u_int, level_grid, h)
        exported = self.clock()
        seconds_task = exported-self.t0-self.spent
        name = f'field_{done:04d}.npz'
        np.savez(self.out/name, u=u)
        record = dict(step=done, file=name, sha256=core.sha_file(self.out/name), u_sha256=hashlib.sha256(u.tobytes()).hexdigest(), seconds_task=seconds_task, terms={k: row[k] for k in ('data', 'geometry', 'floor', 'curvature', 'min_det', 'total')}, terms_note='the energy before this step (the loop evaluates, then steps)')
        finished = self.clock()
        record['seconds_io'] = finished-tick
        self.spent += finished-tick
        self.records[str(done)] = record


def run_field(o, cell):
    """One field run.  The flow, the records and the stops of run_ef.run_fast, with the checkpoints; arm O: fl_oracle.prepare instead of the predictions."""
    t0 = perf_counter()
    spec = fl.ARMS[cell.arm]
    fz = core.frozen_api()
    import torch
    out = Path(o.out)
    if (out/'started.json').exists():
        raise SystemExit(f'{out}: this run has started before; a run is never rerun')
    sigma = fl.sigma_of(cell, fl.DEFAULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    run.write_json(out/'started.json', dict(run_id=cell.run_id, utc=run.utc(), argv=sys.argv, pid=os.getpid()))
    device_cap = mc0p.limit_device_memory(torch, o.device)
    config = fz.PreoptConfig(**fl.arm_config(cell, sigma))
    steps = fl.checkpoints_of(cell)
    keeper = CheckpointKeeper(out, steps, t0)
    seconds_image_norm = None
    status, reason, u, record, scale, seconds_labels, seconds_channels, channel_sources, digest, scale_check, final_same = 'done', None, None, None, None, None, None, None, None, None, None
    pin = None if cell.is_oracle else fl.read_images(cell)     # the firewall view: no label member is decoded, here or below; arm O reads the truth by itself
    try:
        stacks = None
        if cell.is_oracle:
            pin, stacks, scale, channel_sources, digest = oracle_arm.prepare(cell.pair, o.device, workers=core.SETUP_WORKERS)
            seconds_labels = perf_counter()-t0
        else:
            predicted = fast.read_predicted(pin, workers=core.SETUP_WORKERS)
            seconds_labels = perf_counter()-t0
            channel_sources = predicted.record
            digest = efrun.labels_digest(predicted.labels)
            if channel_sources['oracle'] is not False:
                status, reason = 'not_predicted_channels', f"channel sources: oracle {channel_sources['oracle']}, members read {channel_sources['pair_keys_read']}; no optimization"
            else:
                stacks, scale = channels.prepare_stacks(spec, pin, predicted, core.load_table(), o.device, workers=core.SETUP_WORKERS)
            del predicted
        if stacks is not None:
            seconds_channels = perf_counter()-t0
            if spec['channels'] == 'onehot+intensity':          # a record only, on the scaled stacks; not in the task clock
                tick = perf_counter()
                scale['g_norm_image_channel'] = channels.image_channel_norm(stacks, o.device)
                seconds_image_norm = perf_counter()-tick
                keeper.spent += seconds_image_norm
            u, record = core.optimize_k(stacks, pin.grid, config, device=o.device, started=t0, observer=run.vram_guard(torch, o.device, mc.VRAM_LIMIT_BYTES), workers=core.SETUP_WORKERS, record_level_norms=True, downsample=None if cell.is_oracle else fast.FastDownsample(), checkpoint=keeper)
            del stacks
            record['role'] = fl.ORACLE_ROLE if cell.is_oracle else fl.ROLE
            last = keeper.records.get(str(steps[-1]))
            final_same = None if last is None else hashlib.sha256(u.tobytes()).hexdigest() == last['u_sha256']
            scale_check = 'core.prepare_stacks checked it' if cell.is_oracle else channels.scale_check_of(record, scale)
            if record['stopped'] is not None:
                status, reason = 'time_cap', f"the wall cap {fl.RUN_CAP_SECONDS} s at {record['stopped']}"
                np.savez(out/'field_stopped.npz', u=u)                                              # the evidence of a run cut short; never read
            elif final_same is not True:
                status, reason = 'checkpoint_mismatch', 'the last checkpoint is not the field optimize_k returned'
            elif not cell.is_oracle and not scale_check['passed']:
                status, reason = 'scale_mismatch', f"||g_D|| of the native tensors {scale_check['g_norm_scaled']} differs from {scale_check['target']}"
    except run.VramLimit as error:
        status, reason = 'vram_stop', str(error)
    except torch.cuda.OutOfMemoryError as error:
        status, reason = 'cuda_oom', str(error)[:500]
    except FloatingPointError as error:                 # run_level_k: a nonfinite objective; the checkpoints written so far stay on record, a status that is not done is not read
        status, reason = 'nonfinite', str(error)[:500]
    last = keeper.records.get(str(steps[-1]))
    cuda = torch.device(o.device).type == 'cuda'
    run.write_json(out/'run.json', dict(
        schema='bayesreg.fl.run.v1', run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, k=spec['k'], native_iterations=fl.steps_of(cell), checkpoint_steps=list(steps), channels=spec['channels'], weight=spec['weight'], D_factor=spec['D_factor'], oracle=None if channel_sources is None else channel_sources['oracle'], status=status, reason=reason, role=fl.ORACLE_ROLE if cell.is_oracle else fl.ROLE,
        config=fl.arm_config(cell, sigma), scale=scale, scale_check=scale_check, labels_sha256=digest, channel_sources=channel_sources, optimizer=record, checkpoints=keeper.records, seconds_checkpoint_io=sum(r['seconds_io'] for r in keeper.records.values()), seconds_image_norm=seconds_image_norm, final_equals_last_checkpoint=final_same, field=None if last is None else dict(file=last['file'], sha256=last['sha256'], u_sha256=last['u_sha256']), taper=core.TAPER, code_sha=o.code_sha, frozen_commit=core.FROZEN_COMMIT, random_seed='none: no random step',
        data=None if pin is None else dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], fixed_subject=pin.fixed_subject, moving_subject=pin.moving_subject, stage=str(core.STAGE)),
        env=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if cuda else 'cpu'), device_cap=device_cap,
        device_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None, device_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if cuda else None,
        host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, seconds_labels=seconds_labels, seconds_channels=seconds_channels, seconds_total=perf_counter()-t0, ended_utc=run.utc()))
    return 0


def run_one(o):
    cell = fl.require_cell(o.arm, o.tier, o.pair)
    if cell.is_repair:
        raise SystemExit(f'{cell.run_id}: a repair arm: repair-one')
    problem = fl.out_problem(o.out, cell)
    if problem:
        raise SystemExit(problem)
    return run_field(o, cell)


# ---- one repair -------------------------------------------------------------------------------------------------------------------------------------
def enable_jit(root):
    """The frozen tree's content-addressed cache of the Numba kernels, before any kernel compiles: a cached entry is the machine code an identical compile produced (jit_cache.py; a test runs a repair with and without it).  Its own directory under this root, not the default that other lines share."""
    from jd_curl.three_d import jit_cache
    return jit_cache.enable(Path(root)/'cache/jit')


def source_of_repair(cell, root=None, results=None):
    """(directory, file name, expected sha256, metrics name, run.json, seconds of the task up to that field): where a repair's field is, read only."""
    root = fl.DEFAULT_ROOT if root is None else root
    results = fl.RESULTS if results is None else results
    if cell.arm in fl.EF_REPAIRS:
        d = fl.ef_dir(results, cell.base, cell.pair)
        record = run.read_json(d/'run.json')
        if record is None:
            raise SystemExit(f'{d}: no run.json')
        if (record.get('arm'), record.get('tier')) != (cell.base, 'fine'):
            raise SystemExit(f"{d}: not {cell.base} fine: {record.get('arm')} {record.get('tier')}")
        return d, 'field.npz', record['field']['sha256'], 'metrics.json', record, record['seconds_total']
    source = fl.Cell(cell.base, fl.FIRST if cell.tier.startswith('c') else cell.tier, cell.pair)
    d = fl.run_dir(root, source)
    record = run.read_json(d/'run.json')
    step = fl.steps_of(cell)
    checkpoint = ((record or {}).get('checkpoints') or {}).get(str(step))
    if checkpoint is None:
        raise SystemExit(f'{d}: no checkpoint at {step} steps')
    return d, checkpoint['file'], checkpoint['sha256'], f'metrics_{step:04d}.json', record, checkpoint['seconds_task']


def source_checked(cell, fz, grid, root=None, results=None):
    """(tapered u, record of the source): the field a repair starts from, after every check.  The file against its run.json, the taper of the evaluation, its hash against that evaluation's metrics.  STOP otherwise."""
    d, name, sha, metrics_name, record, seconds_task = source_of_repair(cell, root, results)
    metrics = run.read_json(d/metrics_name)
    if metrics is None:
        raise SystemExit(f'{d}: no {metrics_name}')
    if record.get('status') != 'done' or record.get('oracle') is not False:
        raise SystemExit(f"{d}: not a finished field with oracle false: {record.get('status')}, {record.get('oracle')}")
    if core.sha_file(d/name) != sha:
        raise SystemExit(f'{d}/{name} differs from its run.json')
    with np.load(d/name) as z:
        raw = np.array(z['u'])
    tick = perf_counter()
    u, _ = fz.boundary_taper(raw, grid, core.TAPER)
    u = np.ascontiguousarray(u)
    seconds_taper = perf_counter()-tick
    digest = hashlib.sha256(u.tobytes()).hexdigest()
    if digest != metrics['tapered_u_sha256']:
        raise SystemExit(f"{d}/{name}: the tapered field {digest} is not the one {metrics_name} evaluated, {metrics['tapered_u_sha256']}")
    return u, dict(directory=str(d), file=name, run_id=record['run_id'], field_sha256=sha, tapered_u_sha256=digest, code_sha=record.get('code_sha'), metrics_file=metrics_name, accepted_alpha=metrics['certificate']['accepted_alpha'], n_infeasible_at_1=metrics['certificate']['n_infeasible_at_1'], seconds_taper=seconds_taper, seconds_task=seconds_task)


def repair_allowed(cell, root):
    """RF150, RF300 and the other 14 pairs are in their plans; a checkpoint of the first 5 pairs only when the code chose it."""
    if cell.arm in fl.EF_REPAIRS or cell.tier.startswith('rest'):
        return True
    return any(f"R{c['arm']}" == cell.arm and f"c{c['step']}" == cell.tier for c in fl.chosen_repairs(root))


def repair_one(o):
    """run_ef.repair_one on FL's sources.  No image, no label reaches the repair: read_images gives the grid and its firewall; the field, the grid and eps are all repair_field sees.  An existing repair.npz is never redone."""
    cell = fl.require_cell(o.arm, o.tier, o.pair)
    if not cell.is_repair:
        raise SystemExit(f'{cell.run_id}: not a repair arm: run-one')
    problem = fl.out_problem(o.out, cell)
    if problem:
        raise SystemExit(problem)
    if not repair_allowed(cell, fl.DEFAULT_ROOT):
        raise SystemExit(f'{cell.run_id}: the code did not choose this checkpoint (analysis/repair_selection_*.json)')
    t0 = perf_counter()
    fz = core.frozen_api()
    jit = enable_jit(fl.DEFAULT_ROOT)
    import torch
    from jd_curl.three_d.feasibility import certify_map
    out = Path(o.out)
    if (out/'started.json').exists():
        raise SystemExit(f'{out}: this run has started before; a run is never rerun')
    out.mkdir(parents=True, exist_ok=True)
    run.write_json(out/'started.json', dict(run_id=cell.run_id, utc=run.utc(), argv=sys.argv, pid=os.getpid()))
    device_cap = mc0p.limit_device_memory(torch, o.device)
    pin = fl.read_images(cell)
    u, source = source_checked(cell, fz, pin.grid)
    config = run.certificate_config()                   # configuration C, explicitly: delta 0, depth 48, 255 boxes, the frozen driver's tile
    tick = perf_counter()
    efrun.tiny_warmup(certify_map, config, repair.locate_cells)
    seconds_warmup_numba = perf_counter()-tick
    tick = perf_counter()
    warmed = efrun.device_warmup(o.device, certify_map, config)
    seconds_warmup_device = perf_counter()-tick
    settings = repair.RepairConfig(eps=fl.REPAIR_EPS)
    tick = perf_counter()
    cells = repair.locate_cells(u, pin.grid, config)
    seconds_first_locate = perf_counter()-tick
    tick = perf_counter()
    result = repair.repair_field(u, pin.grid, cells, settings, certify=certify_map, certificate_config=config, device=o.device)
    seconds_repair = seconds_first_locate+perf_counter()-tick    # locate, solve and every round's certificate: what a repair costs on a compiled process
    np.savez(out/'repair.npz', nodes=result.attempt_nodes, values=result.attempt_values)
    cuda = torch.device(o.device).type == 'cuda'
    from jd_curl.three_d import jit_cache
    run.write_json(out/'repair.json', dict(
        schema='bayesreg.fl.repair.v1', run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, oracle=False, role=repair.ROLE, status='certified' if result.passed else 'failed_back_to_frozen_rule', passed=result.passed, reason=result.reason, eps=settings.eps, config=asdict(settings), source=source, n_located=len(cells),
        initial_certificate=result.initial_certificate, certificate=result.certificate, rounds=result.rounds, seconds_warmup=seconds_warmup_numba+seconds_warmup_device, seconds_warmup_numba=seconds_warmup_numba, seconds_warmup_device=seconds_warmup_device, device_warmup=warmed, seconds_first_locate=seconds_first_locate, seconds_repair=seconds_repair, attempt=dict(file='repair.npz', sha256=core.sha_file(out/'repair.npz'), n_free_nodes=len(result.attempt_nodes)),
        jit_cache_enabled=jit, jit_cache=jit_cache.record(), certificate_config=asdict(config), code_sha=o.code_sha, frozen_commit=core.FROZEN_COMMIT, data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], fixed_subject=pin.fixed_subject, moving_subject=pin.moving_subject, keys_read=list(pin.provenance['keys_read'])),
        env=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if cuda else 'cpu'), device_cap=device_cap, device_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None, device_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if cuda else None,
        host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, seconds_total=perf_counter()-t0, ended_utc=run.utc()))
    return 0


# ---- evaluation --------------------------------------------------------------------------------------------------------------------------------
def field_metrics(u, labs, spacing, ev):
    """run_mc0.field_metrics, plus experiments/d0/evaluate.py's HD95: the same function, the same labels."""
    metrics = run.field_metrics(u, labs, spacing, ev)
    warped = ev.warp_labels_l2r(labs['moving_seg35'], ev.disp_voxels(u, spacing))
    metrics['hd95'], _ = ev.hd95_l2r(labs['fixed_seg35'], labs['moving_seg35'], warped)
    return metrics


def oracle_mark_checked(cell, record):
    """The mark metrics_<step>.json takes from run.json, after checking it; ValueError otherwise: the field is not read."""
    if cell.is_oracle:
        if record.get('oracle') is not True:
            raise ValueError(f"the oracle arm's run.json says oracle {record.get('oracle')}")
        return True
    return mc.oracle_of_run(record)


def evaluate_field(d, cell, step, record, labs, pin, fz, ev, certify, config):
    """One checkpoint: its file against run.json, the taper, the certificate at alpha = 1 (no ladder), the metrics."""
    checkpoint = record['checkpoints'][str(step)]
    if core.sha_file(d/checkpoint['file']) != checkpoint['sha256']:
        raise ValueError(f"{checkpoint['file']} differs from run.json")
    with np.load(d/checkpoint['file']) as z:
        raw = np.array(z['u'])
    u, _ = fz.boundary_taper(raw, pin.grid, core.TAPER)
    tick = perf_counter()
    ladder = run.certify_ladder(u, pin.grid, (1.,), certify, config)
    seconds_certificate = perf_counter()-tick
    tick = perf_counter()
    metrics = field_metrics(np.asarray(u), labs, tuple(pin.grid.spacing), ev)
    run.write_json(d/f'metrics_{step:04d}.json', dict(schema='bayesreg.fl.metrics.v1', run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, step=step, oracle=oracle_mark_checked(cell, record), status=record['status'], evaluated_utc=run.utc(), field='the checkpoint after the taper of width 4, alpha = 1', tapered_u_sha256=hashlib.sha256(np.asarray(u).tobytes()).hexdigest(),
                                                      certificate=ladder, seconds_certificate=seconds_certificate, seconds_metrics=perf_counter()-tick, metrics=metrics, evaluator=dict(script='experiments/d0/evaluate.py', sha256=core.sha_file(core.ROOT/'experiments/d0/evaluate.py'))))


def face_distance(nodes, grid):
    """The smallest number of nodes between a free node and a face of the volume; None when no node is free.  A record for whoever reinjects the field: a repaired node inside the taper's band (< core.TAPER) does not survive a second taper."""
    nodes = np.asarray(nodes, dtype=np.int64)
    if not nodes.size:
        return None
    return int(min(min(int(c.min()), int((n-1-c).min())) for c, n in zip(np.unravel_index(nodes, grid.shape), grid.shape)))


def evaluate_repair(d, cell, labs, pin, fz, ev, certify, config):
    """As run_ef.evaluate_repair.  After: the repaired field, certificate at alpha = 1 (the evaluation's own), metrics.  Before: the source's metrics, for the same tapered field.  Written from the fields, nothing from the run's opinion."""
    record = run.read_json(d/'repair.json')
    if record is None:
        raise ValueError('no repair.json')
    u, source = source_checked(cell, fz, pin.grid)
    if source['tapered_u_sha256'] != record['source']['tapered_u_sha256']:
        raise ValueError('the source field changed since the repair')
    source_metrics = run.read_json(Path(source['directory'])/source['metrics_file'])
    v = efrun.repaired_field(u, d, record)
    tick = perf_counter()
    ladder = run.certify_ladder(v, pin.grid, (1.,), certify, config)
    seconds = perf_counter()-tick
    metrics = field_metrics(v, labs, tuple(pin.grid.spacing), ev)
    with np.load(d/'repair.npz') as z:
        min_face_distance = face_distance(z['nodes'], pin.grid)
    run.write_json(d/'repair_metrics.json', dict(schema='bayesreg.fl.repair_metrics.v1', min_face_distance=min_face_distance, taper_width=core.TAPER, run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, eps=fl.REPAIR_EPS, oracle=False, evaluated_utc=run.utc(), field='the repaired field (no second taper)', repaired_u_sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(),
                                                 before=dict(source=source['directory'], file=source['file'], tapered_u_sha256=source['tapered_u_sha256'], metrics=source_metrics['metrics'], n_infeasible_at_1=source_metrics['certificate']['n_infeasible_at_1']),
                                                 after=dict(metrics=metrics, certificate=ladder, seconds_certificate=seconds), change=repair.change_record(u, v), evaluator=dict(script='experiments/d0/evaluate.py', sha256=core.sha_file(core.ROOT/'experiments/d0/evaluate.py'))))


def evaluate_all(o, *, cells=None):
    """As run_ef.evaluate_all.  The truth labels enter here, for the metrics only (arm O has its own reader for its channels)."""
    root = Path(o.root)
    fz = core.frozen_api()
    enable_jit(root)
    ev = core.load_module('_mc0_evaluate', 'experiments/d0/evaluate.py')
    from jd_curl.three_d.feasibility import certify_map
    config = run.certificate_config()
    failures, loaded = 0, {}

    def truth(pair):
        if pair not in loaded:
            loaded.clear()
            loaded[pair] = fl.read_truth(pair, TRUTH_MEMBERS)     # one pair in memory
        return loaded[pair]
    for cell in (fl.all_cells(root) if cells is None else cells):
        d = fl.run_dir(root, cell)
        if cell.is_repair:
            if not (d/'repair.json').is_file() or (d/'repair_metrics.json').exists() or (d/'metrics_error.json').exists():
                continue
            try:
                pin, labs = truth(cell.pair)
                evaluate_repair(d, cell, labs, pin, fz, ev, certify_map, config)
            except Exception as error:                  # kept; the others go on; not retried
                run.write_json(d/'metrics_error.json', dict(run_id=cell.run_id, error=repr(error)[:2000], utc=run.utc()))
                failures += 1
            continue
        record = run.read_json(d/'run.json')
        for step in (fl.checkpoints_of(cell) if record is not None else ()):
            if str(step) not in (record.get('checkpoints') or {}) or (d/f'metrics_{step:04d}.json').exists() or (d/f'metrics_{step:04d}_error.json').exists():
                continue
            try:
                pin, labs = truth(cell.pair)
                evaluate_field(d, cell, step, record, labs, pin, fz, ev, certify_map, config)
            except Exception as error:                  # kept; the others go on; not retried
                run.write_json(d/f'metrics_{step:04d}_error.json', dict(run_id=cell.run_id, step=step, error=repr(error)[:2000], utc=run.utc()))
                failures += 1
    return 8 if failures else 0


# ---- the queue ------------------------------------------------------------------------------------------------------------------------------------
def ledger(root):
    """run_mc0.ledger on this root's cells, wherever they are written: a directory without queue_done.json is a run the driver lost, counted at the wall cap, never rerun."""
    seconds, stopped, states = 0., {}, {}
    for cell in fl.all_cells(root):
        d = fl.run_dir(root, cell)
        if not d.exists():
            continue
        done = run.read_json(d/'queue_done.json')
        if done is None:
            seconds += fl.RUN_CAP_SECONDS
            states[cell.run_id] = 'interrupted'
            continue
        seconds += done['unit_seconds']
        states[cell.run_id] = done['state']
        if done.get('arm_stop') and cell.arm not in stopped:
            stopped[cell.arm] = dict(run_id=cell.run_id, reason=done['arm_stop'])
    return dict(run_unit_seconds=seconds, arms_stopped=stopped, states=states)


def control_problem(cell, record, results=None):
    """Arm F at 600 steps against ef_v1's F600, u bit for bit (the path does not depend on the total steps, plan.md).  A reason when it does not hold or cannot be checked, None otherwise.  Only a finished run of F is asked."""
    if cell.arm != 'F' or cell.tier != fl.FIRST or record is None or record.get('status') != 'done':
        return None
    results = fl.RESULTS if results is None else results
    mine = ((record.get('checkpoints') or {}).get('600') or {}).get('u_sha256')
    theirs = ((run.read_json(fl.ef_dir(results, 'F600', cell.pair)/'run.json') or {}).get('field') or {}).get('u_sha256')
    if mine is None or theirs is None or mine != theirs:
        return f'{cell.run_id}: the field at 600 steps {mine} is not ef_v1 F600 {theirs}'
    return None


def run_phase(o, system, root, log, code, sha, cells, gate):
    """One phase of the plan, in order, as run_ef.run_phase.  Returns how it ended; run.Stop on 3 failures in a row, a closed gate, a wrong oracle mark, the control of F."""
    failures, end, first = 0, 'all runs handled', True
    for cell in cells:
        d = fl.run_dir(root, cell)
        if d.exists():
            continue                                    # never rerun
        state = ledger(root)
        if cell.arm in state['arms_stopped']:
            log(f"skip {cell.run_id}: arm stopped ({state['arms_stopped'][cell.arm]['reason']})")
            continue
        if state['run_unit_seconds'] >= RUN_UNITS_CAP_SECONDS:
            end = f"the cap of {RUN_UNITS_CAP_SECONDS} s of run units reached ({state['run_unit_seconds']:.0f} s); no new run"
            log(end)
            break
        if not first:
            system.sleep(o.gap_seconds)
        first = False
        run.wait_for_gate(system, root, True, o.wait_minutes, log, gate)
        d.mkdir(parents=True)
        argv = run_argv(cell, code, root, sha)
        (d/'cmd.txt').write_text(shlex.join(argv)+'\n')
        log(f'launch {cell.unit}; load {system.load1()}; /mnt/c free {system.host_free_gib():.0f} GiB')
        tick = system.now()
        rc = system.run(argv, d/'systemd_run.log')
        seconds = system.now()-tick
        record = run.read_json(d/('repair.json' if cell.is_repair else 'run.json'))
        failed = rc != 0 or record is None
        stop = run.arm_stop_reason(cell, record, seconds, k=1 if cell.is_repair else K_STOP, vram_limit=mc.VRAM_LIMIT_BYTES)
        state_name = 'failed' if failed else record['status']
        run.write_json(d/'queue_done.json', dict(run_id=cell.run_id, unit=cell.unit, rc=rc, unit_seconds=seconds, state=state_name, arm_stop=stop, ended_utc=run.utc()))
        log(f"ended {cell.run_id}: rc {rc}, {seconds:.0f} s, {state_name}" + (f'; arm stops: {stop}' if stop else ''))
        never_read_its_channels = record is not None and record.get('channel_sources') is None and record.get('status') in ('cuda_oom', 'vram_stop')     # the arm's stop above is the rule for it: no mark to check
        if not cell.is_repair and record is not None and not never_read_its_channels and record.get('oracle') is not cell.is_oracle:
            raise run.Stop(f"{cell.run_id}: run.json oracle {record.get('oracle')}: the channels are not what this arm is about: {record.get('reason')}")
        problem = control_problem(cell, record)
        if problem:
            raise run.Stop(problem)
        failures = failures+1 if failed else 0
        if failures >= run.MAX_CONSECUTIVE_FAILURES:
            raise run.Stop(f'{failures} failures in a row')
    return end


def phase_and_evaluation(o, system, root, log, code, sha, cells, gate, eval_unit, rcs):
    """A later phase: a STOP among its runs still gets the evaluation of what exists; then it is raised."""
    stopped = None
    try:
        run_phase(o, system, root, log, code, sha, cells, gate)
    except run.Stop as stop:
        stopped = stop
        log(f'STOP: {stop}')
    try:
        run.wait_for_gate(system, root, False, o.wait_minutes, log, gate)
        log(f'launch {eval_unit}')
        rcs[eval_unit] = system.run(eval_argv(code, root, eval_unit), root/f'{eval_unit}.systemd_run.log')
    except run.Stop as stop:
        log(f'{eval_unit} not started: {stop}')
        stopped = stopped or stop
    if stopped is not None:
        raise stopped


def evaluated(rcs, unit, what):
    """A choice of the code is written once: never after an evaluation unit that failed, whose fields would count as not read for good (FL_tools_复核 important 1).  A restart evaluates what is left and chooses then; a checkpoint with a metrics error file is not retried, so a restart's rc is 0 and the choice is written with that checkpoint not read (n_read in its candidates)."""
    if rcs.get(unit, 0) != 0:
        raise run.Stop(f'{unit} exit {rcs[unit]}: {what} not chosen; the evaluation first, by a restart')


def log_choice(log, root, name, text):
    log(f'{text}; {name} sha256 {fl.selection_hashes(root).get(name)}')


def later_phases(o, system, root, log, code, sha, gate, rcs):
    """Phase 2 and 3: nothing here can take away the first reading.  Returns how they ended."""
    evaluated(rcs, EVAL_UNIT, 'the repairs are')
    selection = decision.ensure_repair_selection(root, 1)
    log_choice(log, root, 'repair_selection_1.json', f"the repairs the code chose: {selection['chosen'] or 'none'}; and RF150, RF300")
    phase_and_evaluation(o, system, root, log, code, sha, fl.plan_ef_repairs()+fl.repair_cells(selection['chosen']), gate, EVAL_R1_UNIT, rcs)
    decision.write_analysis(root, stops=ledger(root)['arms_stopped'])
    if o.through == 'repair':
        return 'stopped by --through repair'
    evaluated(rcs, EVAL_R1_UNIT, "X's sigma is")
    x = decision.ensure_x_selection(root)
    if x['sigma_field'] is None:
        log_choice(log, root, 'x_selection.json', 'X: no arm qualifies; phase 3 does not run')
        return 'no sigma for X'
    log_choice(log, root, 'x_selection.json', f"X: sigma_field {x['sigma_field']} of arm {x['arm']}")
    phase_and_evaluation(o, system, root, log, code, sha, fl.plan_x(), gate, EVAL_X_UNIT, rcs)
    evaluated(rcs, EVAL_X_UNIT, 'the repair of X is')
    selection = decision.ensure_repair_selection(root, 2)
    log_choice(log, root, 'repair_selection_2.json', f"the repairs of X the code chose: {selection['chosen'] or 'none'}")
    decision.write_analysis(root, stops=ledger(root)['arms_stopped'])
    phase_and_evaluation(o, system, root, log, code, sha, fl.repair_cells(selection['chosen']), gate, EVAL_R2_UNIT, rcs)
    return 'all phases handled'


def keep_old_stops(root):
    """Evidence, kept under its time."""
    for name in ('queue_stop', 'queue_stop_later'):
        if (root/f'{name}.json').exists():
            (root/f'{name}.json').rename(root/f"{name}_{run.utc().replace(':', '').replace('-', '')}.json")


def committed_bytes(sha, rel):
    """The bytes of a tracked file at a commit, None when git cannot say."""
    done = subprocess.run(['git', 'show', f'{sha}:{rel}'], cwd=core.ROOT, capture_output=True)
    return done.stdout if done.returncode == 0 else None


def reading_code_problem(root):
    """Why this process may not write this root's reading, or None.  The units run from $ROOT/code, the batch's commit; the driver and a hand's decide from wherever they were started (FL_tools_复核 important 4): the rules of the reading must be the batch's, byte for byte.  No batch.json: nothing has run."""
    batch = run.read_json(Path(root)/'batch.json')
    if batch is None:
        return None
    for module, rel in ((decision, 'experiments/fl/decide_fl.py'), (fl, 'experiments/fl/fl_core.py')):
        theirs = committed_bytes(batch['code_sha'], rel)
        if theirs is None or theirs != Path(module.__file__).read_bytes():
            return f"{module.__file__} is not {rel} of the batch's commit {batch['code_sha']}: the reading's rules would not be the batch's; run the driver from $ROOT/code"
    return None


def write_batch(root, name, value, sha):
    """batch records: written once; a restart must be at the same code and the same plan."""
    batch = run.read_json(root/name)
    if batch is None:
        run.write_json(root/name, dict(value, schema=fl.BATCH_SCHEMA, code_sha=sha, frozen_commit=core.FROZEN_COMMIT, started_utc=run.utc()))
    elif batch['code_sha'] != sha:
        raise run.Stop(f"{name} is at {batch['code_sha']}, the code at {sha}")
    elif batch['plan'] != value['plan']:
        raise run.Stop(f'{name} has another plan')


def rest_body(o, system, root, log, code, sha, gate):
    """The lead's separate queue of the other 14 pairs, after the first reading."""
    if not (root/'analysis/decision_fl.json').is_file():
        raise run.Stop('no reading of the first 5 pairs yet: the other 14 pairs start after it')
    fields, repairs = fl.plan_rest(o.arm, o.steps)
    batch = run.read_json(root/'batch.json') or {}
    if batch.get('code_sha') != sha:
        raise run.Stop(f"the other 14 pairs run at the first batch's commit {batch.get('code_sha')}, not at {sha}")
    problem = reading_code_problem(root)
    if problem:
        raise run.Stop(problem)
    problems = safe_input_problems(pairs='rest14')
    if o.arm == 'X' and fl.x_sigma(root) is None:
        problems.append('X: no sigma_field in analysis/x_selection.json')
    if problems:
        raise run.Stop(f'the inputs, before any run: {problems}')
    name = f'batch_rest14_{o.arm}_{o.steps}.json'
    write_batch(root, name, dict(plan=[c.run_id for c in fields+repairs], arm=o.arm, steps=o.steps, gap_seconds=o.gap_seconds), sha)
    log(f'rest14 start: {o.arm} at {o.steps} steps, {len(fields)} fields, then {len(repairs)} repairs')
    rcs, stop = {}, None
    try:
        phase_and_evaluation(o, system, root, log, code, sha, fields, gate, EVAL_REST_UNIT, rcs)
        phase_and_evaluation(o, system, root, log, code, sha, repairs, gate, EVAL_REST_R_UNIT, rcs)
    except run.Stop as error:
        stop = str(error)
    state = ledger(root)
    result = decision.write_analysis(root, stops=state['arms_stopped'])
    run.write_json(root/f'queue_end_rest14_{o.arm}_{o.steps}.json', dict(stop=stop, eval_rc=rcs, rest14=[dict(arm=r['arm'], step=r['step'], seg35=r['repairs']['seg35'], seg4=r['repairs']['seg4'], n_certified_at_1=r['repairs']['n_certified_at_1']) for r in result['rest14']], ledger=state, utc=run.utc()))
    log('rest14 ended.  Handed to the lead: no next step starts here')
    if stop is not None or any(rc != 0 for rc in rcs.values()):
        raise run.Stop(f'the rest14 queue: stop {stop}, evaluation units {rcs}')
    return 0


def queue_body(o, system, root, log):
    """Three phases.  A STOP in phase 1 or at its evaluation's gate ends the queue: no evaluation, no reading.  From the first reading on, nothing that happens later can take it away."""
    keep_old_stops(root)
    code, sha = system.code_checkout(root, o.commit)
    gate = replace(GATE, host_floor_gib=o.min_host_gib)
    if o.pairs == 'rest14':
        return rest_body(o, system, root, log, code, sha, gate)
    problems = safe_input_problems()
    if problems:
        raise run.Stop(f'the inputs, before any run: {problems}')
    fields = fl.plan_fields()
    write_batch(root, 'batch.json', dict(plan=[c.run_id for c in fields], later='repairs of the checkpoints the code chooses, RF150, RF300, X, its repair: analysis/repair_selection_*.json, x_selection.json', gap_seconds=o.gap_seconds, through=o.through,
                                         caps=dict(run_seconds=fl.RUN_CAP_SECONDS, run_units_seconds=RUN_UNITS_CAP_SECONDS, device_gib=mc.VRAM_LIMIT_BYTES/2**30, memory_max=MEMORY_MAX, nice=NICE, task_budget_seconds=fl.TASK_BUDGET_SECONDS),
                                         gate=dict(own_units=list(gate.own_units), load_limit=gate.load_limit, gpu_free_mib=gate.gpu_free_mib, mem_available_gib=gate.mem_available_gib, home_floor_gb=run.HOME_FLOOR_GB, host_floor_gib=gate.host_floor_gib),
                                         evaluation=dict(numba_threads=EVAL_THREADS, cpu_quota_percent=EVAL_QUOTA)), sha)
    problem = reading_code_problem(root)
    if problem:
        raise run.Stop(problem)
    log(f'start: {len(fields)} field runs; the inputs checked; /mnt/c floor {o.min_host_gib:g} GiB; {o.gap_seconds} s between runs; through {o.through}')
    end = run_phase(o, system, root, log, code, sha, fields, gate)
    run.wait_for_gate(system, root, False, o.wait_minutes, log, gate)
    log(f'launch {EVAL_UNIT}')
    rcs = {EVAL_UNIT: system.run(eval_argv(code, root), root/f'{EVAL_UNIT}.systemd_run.log')}
    state = ledger(root)
    result = decision.write_analysis(root, stops=state['arms_stopped'])          # also after a failed evaluation: a checkpoint without metrics counts as not read
    run.write_json(root/'queue_first_end.json', dict(end=end, eval_rc=rcs[EVAL_UNIT], reading=result['reading'], reason=result['reason'], role='interim: nothing is repaired yet, so no checkpoint can satisfy the premise', ledger=state, utc=run.utc()))
    log(f"the fields are evaluated (interim record: {result['reason']}); a failure of the later phases leaves their records as they are")
    stop_later, end_later = None, None
    if o.through != 'fields':
        try:
            end_later = later_phases(o, system, root, log, code, sha, gate, rcs)
        except run.Stop as stop:
            stop_later, end_later = str(stop), f'STOP in a later phase: {stop}'
            log(end_later)
            run.write_json(root/'queue_stop_later.json', dict(reason=str(stop), utc=run.utc()))
    state = ledger(root)
    result = decision.write_analysis(root, stops=state['arms_stopped'])
    run.write_json(root/'queue_end.json', dict(end=end, end_later=end_later, stop_later=stop_later, eval_rc=rcs, reading=result['reading'], best=result['best'], selections=fl.selection_hashes(root), ledger=state, utc=run.utc()))
    log(f"reading: {result['reading']}, best {result['best']}.  Handed to the lead: no next step starts here")
    failed = {unit: rc for unit, rc in rcs.items() if rc != 0}
    if failed:
        raise run.Stop(f'an evaluation unit failed: {failed} (metrics errors in the run directories, or the {run.EVAL_RUNTIME_MAX} s limit of the unit); the reading is written, a checkpoint without metrics counts as not read')
    if stop_later is not None:
        raise run.Stop(f'a later phase stopped: {stop_later}')
    return 0


def guarded_body(o, system, root, log):
    """A choice that is not the file the code wrote is a STOP with its record, wherever it is met."""
    try:
        return queue_body(o, system, root, log)
    except fl.SelectionChanged as changed:
        raise run.Stop(str(changed))


def queue(o, system=None):
    problem = fl.root_problem(o.root)
    if problem:                                         # before the lock file: nothing is created or touched
        print(problem, file=sys.stderr)
        return 2
    return run.queue(o, system, body=guarded_body, prefix='fl')


# ---- the inputs ------------------------------------------------------------------------------------------------------------------------------------
def input_problems(out=lambda text: None, *, pairs='first5', stage=None, seg_root=None, allowlist=None, unet=None, results=None):
    """Why this batch cannot run, as a list; the dry run prints what it read, the queue stops on any (before batch.json, before any run).  Real files only up to the npy headers of `fixed`, `moving` and of the predictions' `p`, the manifest through the firewall's reader, the predictions' provenance.json, the fields of ef_v1 that RF150, RF300 start from, the control records.  No truth label, no optimization, nothing written; no frozen PYTHONPATH needed."""
    stage = core.STAGE if stage is None else stage
    seg_root = mc.SEG_ROOT if seg_root is None else seg_root
    allowlist = core.ALLOWLIST if allowlist is None else allowlist
    results = fl.RESULTS if results is None else results
    problems = []
    selection = json.loads((Path(stage)/'selection.json').read_text())
    listed = [row['pair'] for row in selection['pairs']]
    first_same, rest_same = tuple(listed[:5]) == core.PAIRS, tuple(listed[5:19]) == fl.REST_PAIRS
    out(f"selection.json first 5 == the plan's pairs: {first_same}; pairs 6..19 == the other 14 of runs/20261104-seed19: {rest_same}; {len(listed)} pairs listed")
    if not first_same:
        problems.append('the first 5 pairs of selection.json differ')
    if not rest_same:
        problems.append('pairs 6..19 of selection.json differ from the other 14')
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    from bayesreg.io.pair import _read_manifest                 # the whitelisting reader: label-derived records are dropped while the JSON is decoded
    guard = load_dev_allowlist(allowlist)
    mi = core.make_inputs()
    unet = mc.default_unet() if unet is None else unet
    from bayesreg.seg.structure_layer import PROVENANCE_FILE, registry_git_state
    git = registry_git_state(mi._registry_of(unet))
    out(f"registry {mi._registry_of(unet)}: tracked {git.get('tracked')}, clean {git.get('clean')}")
    if not (git.get('tracked') is True and git.get('clean') is True):
        problems.append('the repository registry is not committed and clean')
    subjects = {}
    for pair in (core.PAIRS if pairs == 'first5' else fl.REST_PAIRS):
        d = Path(stage)/pair
        _, metadata, manifest = _read_manifest(d/'manifest.json')
        check_pair_subjects(manifest['fixed_subject'], manifest['moving_subject'], guard)
        heads = {m: run.npz_header(d/'pair.npz', m) for m in ('fixed', 'moving')}
        out(f"{core.short_name(pair)}: {d/'pair.npz'}; subjects on the allowlist ({manifest['fixed_subject']}, {manifest['moving_subject']}); spacing {metadata['spacing_xyz']}; " + ', '.join(f'{m} {h[0]} {h[1]}' for m, h in heads.items()))
        if len({tuple(h[0]) for h in heads.values()}) != 1:
            problems.append(f'{pair}: image shapes differ')
        subjects.update({manifest['fixed_subject']: None, manifest['moving_subject']: None})
    admission = mi.a4_admission(list(subjects), allowlist, unet)
    out(f"D24 admission of the {len(admission['subjects'])} subjects (ids only): folds sha256 {admission['folds_sha256']}")
    for subject in subjects:
        p = Path(unet.pred_root)/subject
        missing = [name for name in (PROVENANCE_FILE, 'probs36.npz', 'structure_groups.npz') if not (p/name).is_file()]
        missing += [] if (Path(seg_root)/subject/'provenance.json').is_file() else [f'Atropos {Path(seg_root)/subject}']
        if missing:
            problems.append(f'{subject}: missing {missing}')
            continue
        prov = json.loads((p/PROVENANCE_FILE).read_text())
        head = run.npz_header(p/'probs36.npz', 'p')
        out(f"{subject}: prediction {p}: segmenter {prov.get('segmenter_id')}, policy {prov.get('policy')}, test_labels_seen {prov.get('test_labels_seen')}, probs36 p {head[0]} {head[1]}; Atropos there")
        if prov.get('policy') != 'all_folds_mean' or prov.get('segmenter_id') != mi.RECIPE_SEGMENTER or prov.get('test_labels_seen') is not False:
            problems.append(f'{subject}: prediction not unet35_v1_ens all_folds_mean')
    if pairs == 'first5':
        for arm, source in fl.EF_REPAIRS.items():
            for pair in core.PAIRS:
                d = fl.ef_dir(results, source, pair)
                record, metrics = run.read_json(d/'run.json') or {}, run.read_json(d/'metrics.json') or {}
                same = (d/'field.npz').is_file() and core.sha_file(d/'field.npz') == (record.get('field') or {}).get('sha256')
                certificate = metrics.get('certificate') or {}
                out(f"{arm} source {core.short_name(pair)}: {d}: status {record.get('status')}, oracle {record.get('oracle')}, field.npz equal to its run.json: {same}; alpha = 1: {certificate.get('n_infeasible_at_1')} infeasible cells; seg35 / seg4 {(metrics.get('metrics') or {}).get('seg35')} / {(metrics.get('metrics') or {}).get('seg4')}")
                if record.get('status') != 'done' or record.get('oracle') is not False or not same or metrics.get('tapered_u_sha256') is None:
                    problems.append(f'{arm} source {core.short_name(pair)} is not usable')
        for pair in core.PAIRS:
            f600 = (run.read_json(fl.ef_dir(results, 'F600', pair)/'run.json') or {}).get('field') or {}
            p2 = (run.read_json(Path(results)/fl.MC0_ROOT/'runs/P2_fine'/core.short_name(pair)/'run.json') or {}).get('field') or {}
            out(f"controls {core.short_name(pair)}: ef_v1 F600 u_sha256 {f600.get('u_sha256')}; mc0_v1 P2 fine u_sha256 {p2.get('u_sha256')} (a record, its absence stops nothing)")
            if f600.get('u_sha256') is None:
                problems.append(f'the control of F: no ef_v1 F600 record for {core.short_name(pair)}')
    out(f"colour table sha256 {core.COLOUR_TABLE_SHA256} pinned and read: {len(core.load_table())} labels")
    preopt_same = core.sha_file(core.FROZEN/'jd_curl/three_d/preopt.py') == core.FROZEN_PREOPT_SHA256
    jit = (core.FROZEN/'jd_curl/three_d/jit_cache.py').is_file()
    out(f"frozen {core.FROZEN}: preopt.py sha256 equal to the pin: {preopt_same}; jit_cache.py there: {jit}; the units' PYTHONPATH puts it first")
    if not preopt_same:
        problems.append('the frozen preopt.py differs from the pin')
    if not jit:
        problems.append('the frozen tree has no jit_cache.py')
    return problems


def safe_input_problems(out=lambda text: None, **kwargs):
    """input_problems, with a missing file or a refused input as a problem and not a traceback: the queue says what it was, in queue_stop.json."""
    try:
        return input_problems(out, **kwargs)
    except (Exception, SystemExit) as error:            # noqa: BLE001 - any of them means the inputs are not there
        return [f'{type(error).__name__}: {error}'[:1000]]


# ---- estimates --------------------------------------------------------------------------------------------------------------------------------------
# 〔估算〕 HD95_SECONDS: one field, 35 labels: 2.3 s on a synthetic Voronoi map of the real shape, 5.9 s on the reviewer's brain-shaped map, one thread, next to a training (plan.md); real cortex surfaces may be longer, so 6 s.  EVAL_BASE_SECONDS: ef_v1's evaluation, 35 cells in 342 s with the .9^k ladder and two sets of metrics.  REPAIR_SECONDS: ef_v1's RF05, 26.6-30.4 s a process; more steps mean more infeasible cells.  UNIT_OVERHEAD_SECONDS: unit seconds minus seconds_total in ef_v1 (.7-.9 s).  IO_SECONDS: writing and hashing a checkpoint.
HD95_SECONDS, EVAL_BASE_SECONDS, REPAIR_SECONDS, UNIT_OVERHEAD_SECONDS, IO_SECONDS = 6., 6., 45., 1., 2.


def estimates(root=None, results=None):
    """Seconds and bytes of the plan from ef_v1's own records (F600: seconds per native step, the rest of the run, the size of a field).  An estimate: nothing here runs anything."""
    results = fl.RESULTS if results is None else results
    per_step, rest, size = [], [], []
    for pair in core.PAIRS:
        d = fl.ef_dir(results, 'F600', pair)
        record = run.read_json(d/'run.json')
        if record is not None:
            per_step.append(record['optimizer']['levels'][3]['seconds']/600)
            rest.append(record['seconds_total']-record['optimizer']['levels'][3]['seconds'])
            size.append((d/'field.npz').stat().st_size if (d/'field.npz').is_file() else 165150976)
    step, other, field = (float(np.mean(per_step)), float(np.mean(rest)), float(np.mean(size))) if per_step else (.0545, 11., 165150976.)

    def field_run(cell):
        k = fl.ARMS[cell.arm]['k']
        return other+UNIT_OVERHEAD_SECONDS+fl.steps_of(cell)*step*k/35+len(fl.checkpoints_of(cell))*IO_SECONDS

    def n_fields(cells):
        return sum(len(fl.checkpoints_of(c)) for c in cells)
    first, x = fl.plan_fields(), fl.plan_x()
    ef_repairs = fl.plan_ef_repairs()
    chosen = fl.repair_cells([dict(arm='F', step=2400), dict(arm='M', step=2400), dict(arm='S', step=2400)])
    x_repairs = fl.repair_cells([dict(arm='X', step=4800)])
    evaluation = EVAL_BASE_SECONDS+HD95_SECONDS
    phases = dict(fields=sum(field_run(c) for c in first), fields_evaluation=n_fields(first)*evaluation,
                  repairs=len(ef_repairs+chosen)*REPAIR_SECONDS, repairs_evaluation=len(ef_repairs+chosen)*evaluation,
                  x=sum(field_run(c) for c in x), x_evaluation=n_fields(x)*evaluation, x_repairs=len(x_repairs)*REPAIR_SECONDS, x_repairs_evaluation=len(x_repairs)*evaluation)
    rest_fields, rest_repairs = fl.plan_rest('S', 2400)
    rest_arm = sum(field_run(c) for c in rest_fields)+len(rest_fields)*evaluation+len(rest_repairs)*(REPAIR_SECONDS+evaluation)
    return dict(seconds_per_native_step=step, seconds_rest_of_a_run=other, field_bytes=field, n_field_files=n_fields(first)+n_fields(x), field_files_gib=(n_fields(first)+n_fields(x))*field/2**30, rest14_field_files_gib=len(rest_fields)*field/2**30,
                x_task_seconds=other+4800*step, seconds=phases, seconds_total=sum(phases.values()), seconds_through_repair=sum(phases[k] for k in ('fields', 'fields_evaluation', 'repairs', 'repairs_evaluation')), rest14_arm_seconds=rest_arm)


# ---- dry run --------------------------------------------------------------------------------------------------------------------------------------
def print_cell(out, i, c, phase):
    if c.is_repair:
        out(f"{i:3d} | {phase} | {c.arm} | repair of {c.base}{'' if fl.steps_of(c) is None else f' at {fl.steps_of(c)} steps'} | eps {fl.REPAIR_EPS} | {c.short} | {c.unit} | {c.run_id}")
    else:
        spec = fl.ARMS[c.arm]
        out(f"{i:3d} | {phase} | {c.arm} | K {spec['k']} | sigma {spec['sigma_field'] if spec['sigma_field'] is not None else 'by the code'} | native {fl.steps_of(c)} | checkpoints {'/'.join(map(str, fl.checkpoints_of(c)))} | {spec['channels']}{'' if spec['weight'] is None else ', weight '+spec['weight']}, D factor {spec['D_factor']} | {c.short} | {c.unit} | {c.run_id}")


def dry_run(root=None, stage=None, out=print, unet=None, seg_root=None, allowlist=None, results=None, pairs='first5', arm=None, steps=None):
    """The plan, the estimates, the checks of the queue (input_problems), the start strings.  No truth label, no optimization, nothing written."""
    root = fl.DEFAULT_ROOT if root is None else root
    gate = GATE
    out(f'FL dry run ({run.utc()}): no optimization, no truth label, nothing written; real files read up to the npy headers of fixed, moving and p, the fields of ef_v1 that RF150, RF300 start from, the control records.')
    out(f'root {root}; root exists: {Path(root).exists()}; cap per run {fl.RUN_CAP_SECONDS} s, run units {RUN_UNITS_CAP_SECONDS} s, device {mc.VRAM_LIMIT_BYTES/2**30} GiB; MemoryMax {MEMORY_MAX}, nice {NICE}; {GAP_SECONDS} s between runs (--gap-seconds); task budget {fl.TASK_BUDGET_SECONDS} s (a premise of the reading, not a stop)')
    out(f'gate: own units {gate.own_units}, load < {gate.load_limit}, GPU free >= {gate.gpu_free_mib} MiB, MemAvailable >= {gate.mem_available_gib} GiB, /home >= {run.HOME_FLOOR_GB} GB, /mnt/c >= {gate.host_floor_gib} GiB')
    if pairs == 'first5':
        plan = [(c, 'phase 1') for c in fl.plan_fields()]+[(c, 'phase 2') for c in fl.plan_ef_repairs()]+[(c, 'phase 3') for c in fl.plan_x()]
        out('  # | phase | arm | ...  (phase 2 also: the repairs of up to 3 checkpoints the code chooses, 5 pairs each, cells R<arm> c<steps>; phase 3 also: the repair of X at 4800 steps if it passes)')
    else:
        fields, repairs = fl.plan_rest(arm, steps)
        plan = [(c, 'rest14 fields') for c in fields]+[(c, 'rest14 repairs') for c in repairs]
    for i, (c, phase) in enumerate(plan, 1):
        print_cell(out, i, c, phase)
    e = estimates(root, results)
    s = e['seconds']
    out(f"estimates〔估算, from ef_v1's F600 records and the constants in run_fl.py〕: {e['seconds_per_native_step']:.4f} s per native step, {e['seconds_rest_of_a_run']:.1f} s the rest of a run; phase 1 fields {s['fields']/3600:.2f} h, their evaluation {s['fields_evaluation']/3600:.2f} h; phase 2 with 3 chosen checkpoints {s['repairs']/3600:.2f} h, evaluation {s['repairs_evaluation']/3600:.2f} h; phase 3 X {s['x']/3600:.2f} h, evaluation {s['x_evaluation']/3600:.2f} h, repair {s['x_repairs']/3600:.2f} h, evaluation {s['x_repairs_evaluation']/3600:.2f} h")
    out(f"estimates: through phase 2 {e['seconds_through_repair']/3600:.2f} h; all {e['seconds_total']/3600:.2f} h; X's task clock at 4800 steps about {e['x_task_seconds']:.0f} s before its repair; one arm of the other 14 pairs about {e['rest14_arm_seconds']/3600:.2f} h")
    out(f"estimates: {e['n_field_files']} field files of {e['field_bytes']/2**20:.0f} MiB (np.savez, as ef_v1; not compressed: a write in the T88 loop should stay short) = {e['field_files_gib']:.1f} GiB on /home; one arm of the other 14 pairs {e['rest14_field_files_gib']:.1f} GiB; repair.npz and records small; the JIT cache some MiB")
    problems = safe_input_problems(out, pairs=pairs, stage=stage, seg_root=seg_root, allowlist=allowlist, unet=unet, results=results)
    out(f'/home free {shutil.disk_usage("/home").free/2**30:.0f} GB; /mnt/c free {shutil.disk_usage("/mnt/c").free/2**30:.0f} GiB')
    code = Path(root)/'code'
    out('--- start strings (the queue: the lead; the others: the queue, one per run; the code at $ROOT/code)')
    if pairs == 'first5':
        out(shlex.join(queue_launch(root)))
        for c, _ in plan:
            out(shlex.join(run_argv(c, code, root, '<sha>')))
        for unit in (EVAL_UNIT, EVAL_R1_UNIT, EVAL_X_UNIT, EVAL_R2_UNIT):
            out(shlex.join(eval_argv(code, root, unit)))
        out('--- the other 14 pairs: only the lead, after reading the first 5, at the same commit (example: arm S, 2400 steps)')
        out(shlex.join(rest_launch('S', 2400, '<sha of batch.json>', root)))
    else:
        out(shlex.join(rest_launch(arm, steps, '<sha of batch.json>', root)))
        for c, _ in plan:
            out(shlex.join(run_argv(c, code, root, '<sha>')))
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['--dry-run']:
        d = argparse.ArgumentParser(description='FL: the dry run')
        d.add_argument('--dry-run', action='store_true')
        d.add_argument('--pairs', choices=('first5', 'rest14'), default='first5')
        d.add_argument('--arm', default=None)
        d.add_argument('--steps', type=int, default=None)
        o = d.parse_args(argv)
        if o.pairs == 'rest14' and (o.arm is None or o.steps is None):
            print('usage: --dry-run --pairs rest14 --arm A --steps N', file=sys.stderr)
            return 2
        return dry_run(pairs=o.pairs, arm=o.arm, steps=o.steps)
    p = argparse.ArgumentParser(description='FL: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(fl.DEFAULT_ROOT))
    q.add_argument('--commit', default=None)
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--gap-seconds', type=int, default=GAP_SECONDS)
    q.add_argument('--min-host-gib', type=float, default=DEFAULT_MIN_HOST_GIB)
    q.add_argument('--through', choices=('fields', 'repair', 'all'), default='all')
    q.add_argument('--pairs', choices=('first5', 'rest14'), default='first5')
    q.add_argument('--arm', default=None)
    q.add_argument('--steps', type=int, default=None)
    for name in ('run-one', 'repair-one'):
        r = sub.add_parser(name)
        r.add_argument('--arm', required=True)
        r.add_argument('--tier', required=True)
        r.add_argument('--pair', required=True)
        r.add_argument('--out', required=True)
        r.add_argument('--device', default='cuda')
        r.add_argument('--code-sha', default='n/a')
    for name in ('evaluate', 'decide'):
        sub.add_parser(name).add_argument('--root', default=str(fl.DEFAULT_ROOT))
    o = p.parse_args(argv)
    if o.command == 'queue':
        if o.pairs == 'rest14' and (o.arm is None or o.steps is None or o.commit is None):
            print('--pairs rest14 needs --arm, --steps and --commit (the commit of batch.json)', file=sys.stderr)
            return 2
        if o.pairs == 'first5' and (o.arm is not None or o.steps is not None):
            print('--arm, --steps belong to --pairs rest14', file=sys.stderr)
            return 2
        return queue(o)
    if o.command == 'run-one':
        return run_one(o)
    if o.command == 'repair-one':
        return repair_one(o)
    problem = fl.root_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if o.command == 'evaluate':
        return evaluate_all(o)
    problem = reading_code_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 8
    decision.write_analysis(o.root, stops=ledger(o.root)['arms_stopped'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
