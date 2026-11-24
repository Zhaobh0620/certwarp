"""MC0P (runs/20261104-mc0p/任务书.md, plan.md): the driver of the multi-channel probe on PREDICTED labels.  `oracle: false`, but not a method: uncertified fields of the first 5 development
pairs, in no result table.  Channels: mc0p_core.py.  Reading: decide_mc0p.py.  Optimizer, scale norms, evaluation, certificate record, queue skeleton: MC0's (mc0_core.py, run_mc0.py).

    python -B experiments/mc0/run_mc0p.py --dry-run
        prints all 45 runs and checks the inputs: the npy headers of `fixed`, `moving` of the pair.npz and of A5P's pair.npz (no label member), the manifest through the firewall's reader,
        that each prediction exists, the `p` header of its probs36.npz; no optimization, nothing written.
    python -B experiments/mc0/run_mc0p.py queue [--root DIR] [--commit SHA] [--wait-minutes 30]
        the lead's start, in the unit printed by the dry run.  Code: a detached worktree $ROOT/code of --commit (default HEAD; tracked files clean).  First the bitwise K = 1 check on the GPU
        (unit tag-mc0p-k1check; a mismatch stops the queue).  Then every run in plan order, one unit tag-mc0p-<arm>-<tier>-<ffff>-<mmmm> at a time (MemoryMax 8G, nice 5), each behind GATE, the
        gate of this entry only, which runs next to the training tag-unv2-chain: no $ROOT/hold, no other tag-* or seq-vp-* unit, 1-minute load < 8, /home free >= 100 GB, MemAvailable >= 7 GiB,
        GPU free >= 9.5 GiB, and for a GPU unit unv2_hold_reason: the last member not started, before 8:15 am CST, the chain not waiting for the GPU; waited for --wait-minutes, then STOP with
        queue_stop.json.  A run that has started finishes.  It never touches the training.
        A run directory that exists is never rerun.  An arm stops on the device limit of 9 GiB (per-process cap), a device out-of-memory error, or a K = 35 run over 15 minutes.  At 2.5 h of run
        units no new run starts.  3 failures in a row, or a run whose Q0 images are not bit for bit A5P's inputs (or A5P's file is missing): STOP (no evaluation, no reading: by hand `evaluate`
        with PYTHONPATH=<frozen>:<code>, then `decide`).  Finally one CPU unit tag-mc0p-eval (CPUQuota 100 %, at most 4 threads) and the reading; an evaluation that failed or was cut still gets the reading, then STOP with exit 8.
    python -B experiments/mc0/run_mc0p.py queue --root /home/rmcfz/seq_vp_tag_results/mc0p_v1b --retry-of /home/rmcfz/seq_vp_tag_results/mc0p_v1
        the one retry the lead approved (plan revision 2), after the UNV2 chain, not next to any training; only these two paths.  Plan: the control Q2 long 0438-0439, then every cell not done in the first batch (from its records),
        the arms of the reading first.  Units tag-mc0pb-*, the strict RETRY_GATE (no other tag-* or seq-vp-* unit, load < 1.5, GPU free >= 15 GiB, MemAvailable >= 7 GiB, /home >= 100 GB; no allow list, no hold for the chain).
        Caps and arm stops as before.  Then the evaluation of these fields and the merged reading.  The first batch is read only.
    python -B experiments/mc0/run_mc0p.py --dry-run --retry-of /home/rmcfz/seq_vp_tag_results/mc0p_v1      prints that plan
    run_mc0p.py run-one --arm Q2 --tier long --pair PAIR --out DIR [--device cuda] [--code-sha SHA] [--retry-of V1]     one run: field.npz, run.json
    run_mc0p.py evaluate --root DIR       every field without metrics.json: taper, .9^k certificate (CPU FP64, a record only), the evaluate.py metrics; the only place the truth labels are read
    run_mc0p.py decide --root DIR         analysis/decision_mc0p.json, long.csv, summary_mc0p.md
    run_mc0p.py decide --root mc0p_v1b --merge mc0p_v1         analysis/decision_mc0p_merged.json, long.csv, summary_mc0p_merged.md under mc0p_v1b
    run_mc0p.py self-check-k1 --device cpu|cuda [--out JSON]    run_mc0.self_check_k1

run-one, evaluate and self-check-k1 need PYTHONPATH=<frozen>:<code> (the unit sets it).  Exit codes: 0 done, 2 usage, another driver on the root, or a root that is not MC0P's, 8 STOP or a failed check.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import decide_mc0p as decision  # noqa: E402
import mc0_core as core  # noqa: E402
import mc0p_core as mc  # noqa: E402
import run_mc0 as run  # noqa: E402

SCRIPT = 'experiments/mc0/run_mc0p.py'
DEFAULT_ROOT = Path('/home/rmcfz/seq_vp_tag_results/mc0p_v1')
QUEUE_UNIT, K1_UNIT, EVAL_UNIT = 'tag-mc0p-queue', 'tag-mc0p-k1check', 'tag-mc0p-eval'
RETRY_QUEUE_UNIT, RETRY_K1_UNIT, RETRY_EVAL_UNIT = 'tag-mc0pb-queue', 'tag-mc0pb-k1check', 'tag-mc0pb-eval'
SEED19_ROOT_NAME = 'mc0p_v1c'                    # SEED19's root (run_mc0pc.py): this entry's queue, evaluate and decide refuse it
RETRY_APPROVAL = '总负责人，2026-11-04 约 5 am CST（相当于 §6 的 RETRY_APPROVED）：补跑一次，UNV2 全部结束之后，不与训练同跑'
MEMORY_MAX, NICE = '8G', 5
EVAL_THREADS, EVAL_QUOTA, EVAL_POOL = 1, 100, 4   # next to the training: numba and BLAS one thread, the certificate's fallback pool at most 4 (8 by default), all of it held to one core by CPUQuota: not literally single-threaded
UNV2_OUT = Path('/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full')
UNV2_UNITS = ('tag-unv2-chain.service',)         # experiments/unv2/chain.sh: its one unit; members, calibration and inference run inside it, no child unit (a test reads chain.sh)
TRUTH_MEMBERS = core.PAIR_MEMBERS[2:]            # the evaluation's, and only the evaluation's
LAST_MEMBER_MARK = 'train_B1.started'             # chain.sh: MEMBERS=(A0 B0 A1 B1), `step "train_$m"` writes $STATE/$name.started; after it the chain's next GPU step is the inference
NO_NEW_GPU_RUN_AFTER = datetime.fromisoformat('2026-11-04T08:15:00-06:00')     # the chain's --no-start-after (the running unit's command line): after it no member starts, so the inference is what comes next


def wall_clock():
    return datetime.now(timezone.utc)


def last_line(path, tail=4096):
    """The last non-empty line of a file; '' when there is none."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(f.tell()-tail, 0))
            lines = f.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return ''
    return next((line for line in reversed(lines) if line.strip()), '')


def unv2_hold_reason():
    """Why the GPU must be left to the chain now, or None.  Its member gate asks for utilization <= 20 % and 14 GiB free; its inference gate refuses any GPU compute process and stops the chain, whatever the
    utilization of that moment.  So no new run once the last member has started or the chain's deadline has passed (plan revision 1, in force also after the inference is done: a later GPU run is the lead's decision),
    while the chain's last log line says it waits at the member gate, nor from its registration until the inference is done.  Read only, this closes the gate and can never open it; a run that has started finishes."""
    state = UNV2_OUT/'chain_state'
    if (state/LAST_MEMBER_MARK).exists():
        return f'the last UNV2 member has started ({state/LAST_MEMBER_MARK}): no new GPU run before the chain reaches its inference'
    if wall_clock() >= NO_NEW_GPU_RUN_AFTER:
        return f'past {NO_NEW_GPU_RUN_AFTER.isoformat()}, the deadline of the UNV2 chain: no new GPU run'
    if ((state/'register.started').exists() or (state/'register.done').exists()) and not (state/'infer.done').exists():
        return 'the UNV2 chain is at its registration or inference'
    line = last_line(UNV2_OUT/'chain.log')
    return f'the UNV2 chain waits for the GPU: {line}' if ' waiting (member):' in line else None


GATE = run.Gate(own_units=(f'{QUEUE_UNIT}.service',)+UNV2_UNITS, load_limit=8., gpu_free_mib=9.5*1024, mem_available_gib=7., gpu_hold=unv2_hold_reason)     # 9.5 GiB: above the 9 GiB cap (plan revision 1; the brief had 8)


RETRY_GATE = run.Gate(own_units=(f'{RETRY_QUEUE_UNIT}.service',), load_limit=1.5, gpu_free_mib=15*1024, mem_available_gib=7.)     # the strict gate: no allow list, no hold for the chain (plan revision 2)


def certificate_config():
    """MC0's (configuration C), with the fallback pool at EVAL_POOL threads: a thread count cannot change a certificate (tests/three_d/test_parallel_certificate_e50.py); it is recorded in each attempt's config."""
    return replace(run.certificate_config(), fallback_workers=EVAL_POOL)


# ---- start strings ------------------------------------------------------------------------------------------------------------------------
def unit_argv(unit, code, inner, *, log, gpu=True, numba=1, quota=800, runtime=run.RUN_RUNTIME_MAX):
    return run.unit_argv(unit, code, inner, log=log, gpu=gpu, numba=numba, quota=quota, runtime=runtime, script=SCRIPT, memory_max=MEMORY_MAX, nice=NICE)


def run_argv(cell, code, root, sha='n/a', retry_of=None):
    d = core.run_dir(root, cell)
    return unit_argv(cell.unit, code, ['run-one', '--arm', cell.arm, '--tier', cell.tier, '--pair', cell.pair, '--out', str(d), '--code-sha', sha, *([] if retry_of is None else ['--retry-of', str(retry_of)])], log=d/'log.txt')


def k1_argv(code, root, unit=K1_UNIT):
    return unit_argv(unit, code, ['self-check-k1', '--device', 'cuda', '--out', f'{root}/k1_check_gpu.json'], log=f'{root}/k1_check_gpu.log', runtime=600)


def eval_argv(code, root, unit=EVAL_UNIT):
    return unit_argv(unit, code, ['evaluate', '--root', str(root)], log=f'{root}/eval.log', gpu=False, numba=EVAL_THREADS, quota=EVAL_QUOTA, runtime=run.EVAL_RUNTIME_MAX)


def queue_launch(root=DEFAULT_ROOT, repo=core.MAIN_REPO, retry_of=None):
    """The lead's start string of the queue."""
    return run.check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT if retry_of is None else RETRY_QUEUE_UNIT}', '--collect', f'--working-directory={repo}', f'--property=MemoryMax={MEMORY_MAX}', f'--nice={NICE}', '--setenv=PYTHONDONTWRITEBYTECODE=1',
                             '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', run.PY, '-B', SCRIPT, 'queue', '--root', str(root), *([] if retry_of is None else ['--retry-of', str(retry_of)])])


# ---- one run ---------------------------------------------------------------------------------------------------------------------------------
def limit_device_memory(torch, device, limit=mc.VRAM_LIMIT_BYTES):
    """The per-process cap of the brief, set before any allocation: above it torch raises its out-of-memory error, which stops the arm.  None off the GPU."""
    if torch.device(device).type != 'cuda':
        return None
    index = torch.device(device).index or 0
    total = torch.cuda.get_device_properties(index).total_memory
    fraction = min(limit/total, 1.)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    return dict(limit_bytes=limit, total_bytes=total, fraction=fraction)


def run_one(o, *, cell=None, images=None):
    """cell, images: SEED19's entry (run_mc0pc.py) passes its own checked cell and its reader of the images; the defaults are MC0P's, first 5 pairs only."""
    from time import perf_counter
    t0 = perf_counter()
    cell = mc.require_cell(o.arm, o.tier, o.pair) if cell is None else cell
    retry_of = getattr(o, 'retry_of', None)
    if retry_of is not None:
        problem = mc.retry_problem(Path(o.out).parents[2], retry_of)      # <root>/runs/<arm>_<tier>/<ffff>-<mmmm>
        if problem or (cell.arm, cell.tier, cell.pair) not in {(c.arm, c.tier, c.pair) for c in mc.retry_plan(retry_of)}:
            raise SystemExit(problem or f'{cell.run_id} is done in {retry_of} and is not the control: not run again')
    fz = core.frozen_api()
    import torch
    out = Path(o.out)
    if (out/'started.json').exists():
        raise SystemExit(f'{out}: this run has started before; a run is never rerun')
    out.mkdir(parents=True, exist_ok=True)
    run.write_json(out/'started.json', dict(run_id=cell.run_id, utc=run.utc(), argv=sys.argv, pid=os.getpid()))
    device_cap = limit_device_memory(torch, o.device)
    pin = (mc.read_images if images is None else images)(cell.pair)     # the firewall view: no label member is decoded, here or below
    config = fz.PreoptConfig(**core.tier_config(cell.tier))
    status, reason, u, record, scale, seconds_channels, level_norms, channel_sources = 'done', None, None, None, None, None, None, None
    try:
        predicted = mc.read_predicted(pin, soft=cell.arm == 'Q2s', workers=core.SETUP_WORKERS)
        channel_sources = predicted.record
        table = core.load_table()
        q0 = mc.q0_channels(pin, predicted, table)
        equal = core.equals_a5_input(cell.pair, q0[0][0], q0[1][0], inputs=mc.A5P_INPUTS)
        scale = dict(images_equal_a5p_inputs=equal)     # every arm's reference, not only Q0's own run; True or the queue stops
        if equal is not True:
            status, reason = 'input_mismatch', f'Q0 images against {mc.A5P_INPUTS}: {equal} (None: the file is missing); no optimization'
        elif channel_sources['oracle'] is not False:
            status, reason = 'not_predicted_channels', f"channel sources: oracle {channel_sources['oracle']}, members read {channel_sources['pair_keys_read']}; no optimization"
        else:
            stacks, prepared = mc.prepare_stacks(cell.arm, pin, predicted, q0, o.device, workers=core.SETUP_WORKERS)
            scale.update(prepared)
            reference = None if cell.arm == mc.BASELINE else core.p0_level_norms(pin.fixed, pin.moving, predicted.labels['fixed'], predicted.labels['moving'], pin.grid, o.device, config.sigma_grad, table=table, workers=core.SETUP_WORKERS)
            del q0, predicted                           # the stacks are the only channels left: optimize_k empties them at the native level
            seconds_channels = perf_counter()-t0
            u, record = core.optimize_k(stacks, pin.grid, config, device=o.device, started=t0, observer=run.vram_guard(torch, o.device, mc.VRAM_LIMIT_BYTES), workers=core.SETUP_WORKERS, record_level_norms=True)
            del stacks
            record['role'] = mc.ROLE
            level_norms = mc.level_norm_record(record['level_g_norm'], reference)
            if record['stopped'] is not None:
                status, reason = 'time_cap', f"the wall cap {core.RUN_CAP_SECONDS} s at {record['stopped']}"
    except run.VramLimit as error:
        status, reason = 'vram_stop', str(error)
    except torch.cuda.OutOfMemoryError as error:
        status, reason = 'cuda_oom', str(error)[:500]
    field = None
    if u is not None:
        np.savez(out/'field.npz', u=u)
        field = dict(file='field.npz', sha256=core.sha_file(out/'field.npz'), u_sha256=hashlib.sha256(u.tobytes()).hexdigest())
    cuda = torch.device(o.device).type == 'cuda'
    run.write_json(out/'run.json', dict(
        **({} if retry_of is None else dict(retry_of=str(mc.RETRY_OF_ROOT))),     # the absolute path, a string: SEED's entry check looks for the path component mc0p_v1 (86ca3b4, S_P_ROOTS)
        schema='bayesreg.mc0p.run.v1', run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, k=mc.ARMS[cell.arm]['k'], oracle=None if channel_sources is None else channel_sources['oracle'], decides=mc.ARMS[cell.arm]['decides'], status=status, reason=reason, role='not a method, not in any result table',
        config=core.tier_config(cell.tier), scale=scale, channel_sources=channel_sources, level_norms=level_norms, optimizer=record, field=field, taper=core.TAPER, code_sha=o.code_sha, frozen_commit=core.FROZEN_COMMIT, random_seed='none: no random step',
        data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], fixed_subject=pin.fixed_subject, moving_subject=pin.moving_subject, stage=str(core.STAGE)),
        env=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if cuda else 'cpu'), device_cap=device_cap,
        device_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None, device_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if cuda else None,
        host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, seconds_channels=seconds_channels, seconds_total=perf_counter()-t0, ended_utc=run.utc()))
    return 0


# ---- evaluation --------------------------------------------------------------------------------------------------------------------------------
def evaluate_all(o, *, plan=None, read_pair=None):
    """As run_mc0.evaluate_all, with this plan; `oracle` comes from each run.json, checked against its channel_sources.  The truth labels enter here, for the metrics only.
    plan, read_pair: SEED19's entry (run_mc0pc.py) passes its own; the defaults are MC0P's."""
    fz = core.frozen_api()
    ev = core.load_module('_mc0_evaluate', 'experiments/d0/evaluate.py')
    from jd_curl.three_d.feasibility import certify_map
    config = certificate_config()
    failures, loaded = 0, {}
    for cell in (mc.plan_runs() if plan is None else plan):
        d = core.run_dir(o.root, cell)
        if not (d/'field.npz').is_file() or (d/'metrics.json').exists() or (d/'metrics_error.json').exists():
            continue
        try:
            oracle = mc.oracle_of_run(run.read_json(d/'run.json') or {})
            if cell.pair not in loaded:
                loaded = {cell.pair: (run.read_pair if read_pair is None else read_pair)(cell.pair, TRUTH_MEMBERS)}      # one pair in memory
            run.evaluate_one(d, cell, loaded[cell.pair][1], loaded[cell.pair][0], fz, ev, certify_map, config, oracle=oracle, schema='bayesreg.mc0p.metrics.v1')
        except Exception as error:                     # kept; the others go on; not retried
            run.write_json(d/'metrics_error.json', dict(run_id=cell.run_id, error=repr(error)[:2000], utc=run.utc()))
            failures += 1
    return 8 if failures else 0


# ---- the queue ------------------------------------------------------------------------------------------------------------------------------------
def write_merged(root, retry_of):
    """The merged reading with the arms stopped in each batch; the first batch's records are only read."""
    stops = dict(v1=run.ledger(retry_of, mc.plan_runs())['arms_stopped'], v1b=run.ledger(root, mc.retry_plan(retry_of))['arms_stopped'])
    return decision.write_merged(root, retry_of, stops=stops)


def queue_body(o, system, root, log, retry_of=None):
    """retry_of None: the first batch.  Otherwise the approved retry: its plan, its strict gate, its units, the merged reading at the end."""
    gate, k1_unit, eval_unit = (GATE, K1_UNIT, EVAL_UNIT) if retry_of is None else (RETRY_GATE, RETRY_K1_UNIT, RETRY_EVAL_UNIT)
    code, sha = system.code_checkout(root, o.commit)
    plan = mc.plan_runs() if retry_of is None else mc.retry_plan(retry_of)
    batch = run.read_json(root/'batch.json')
    if batch is None:
        extra = {} if retry_of is None else dict(retry_of=str(retry_of), approval=RETRY_APPROVAL, control=mc.CONTROL.run_id, not_done_in_first_batch=len(plan)-1)
        run.write_json(root/'batch.json', dict(**extra, schema=mc.BATCH_SCHEMA, code_sha=sha, frozen_commit=core.FROZEN_COMMIT, started_utc=run.utc(), plan=[c.run_id for c in plan],
                                               caps=dict(run_seconds=core.RUN_CAP_SECONDS, run_units_seconds=run.GPU_CAP_SECONDS, device_gib=mc.VRAM_LIMIT_BYTES/2**30, memory_max=MEMORY_MAX, nice=NICE),
                                               gate=dict(own_units=list(gate.own_units), load_limit=gate.load_limit, gpu_free_mib=gate.gpu_free_mib, mem_available_gib=gate.mem_available_gib, home_floor_gb=run.HOME_FLOOR_GB,
                                                         chain_hold=None if gate.gpu_hold is None else dict(root=str(UNV2_OUT), last_member_mark=LAST_MEMBER_MARK, no_new_gpu_run_after=NO_NEW_GPU_RUN_AFTER.isoformat(), log_line=' waiting (member):', markers=['register.started', 'register.done', 'infer.done'])),
                                               evaluation=dict(numba_threads=EVAL_THREADS, cpu_quota_percent=EVAL_QUOTA, fallback_workers=EVAL_POOL)))
    elif batch['code_sha'] != sha:
        raise run.Stop(f"batch.json is at {batch['code_sha']}, the code at {sha}")
    elif batch['plan'] != [c.run_id for c in plan]:
        raise run.Stop('batch.json has another plan')
    if retry_of is not None:
        log(f'retry of {retry_of}: {len(plan)-1} cells not done there, and the control {mc.CONTROL.run_id}: {[c.run_id for c in plan]}')
    k1 = root/'k1_check_gpu.json'
    if not k1.is_file():
        run.wait_for_gate(system, root, True, o.wait_minutes, log, gate)
        argv = k1_argv(code, root, k1_unit)
        log(f'launch {k1_unit}')
        system.run(argv, root/'k1_check_gpu.systemd_run.log')
    check = run.read_json(k1)
    if check is None or check.get('bitwise') is not True:
        raise run.Stop(f'the K = 1 check on the GPU did not pass: {check}')
    failures, end = 0, 'all runs handled'
    for cell in plan:
        d = core.run_dir(root, cell)
        if d.exists():
            continue                                    # never rerun
        state = run.ledger(root, plan)
        if cell.arm in state['arms_stopped']:
            log(f"skip {cell.run_id}: arm stopped ({state['arms_stopped'][cell.arm]['reason']})")
            continue
        if state['run_unit_seconds'] >= run.GPU_CAP_SECONDS:
            end = f"the cap of {run.GPU_CAP_SECONDS} s of run units reached ({state['run_unit_seconds']:.0f} s); no new run"
            log(end)
            break
        run.wait_for_gate(system, root, True, o.wait_minutes, log, gate)
        d.mkdir(parents=True)
        argv = run_argv(cell, code, root, sha, retry_of)
        (d/'cmd.txt').write_text(shlex.join(argv)+'\n')
        log(f'launch {cell.unit}; load {system.load1()}')
        tick = system.now()
        rc = system.run(argv, d/'systemd_run.log')
        seconds = system.now()-tick
        record = run.read_json(d/'run.json')
        failed = rc != 0 or record is None
        stop = run.arm_stop_reason(cell, record, seconds, k=mc.ARMS[cell.arm]['k'], vram_limit=mc.VRAM_LIMIT_BYTES)
        control_stop = None
        if retry_of is not None and cell == mc.CONTROL:     # a record only: its failure stops no arm (plan revision 3); if K = 35 really cannot run, the next Q2 cell stops the arm
            stop, control_stop = None, stop
        run.write_json(d/'queue_done.json', dict(run_id=cell.run_id, unit=cell.unit, rc=rc, unit_seconds=seconds, state='failed' if failed else record['status'], arm_stop=stop, **({} if control_stop is None else dict(control_stop=control_stop)), ended_utc=run.utc()))
        log(f"ended {cell.run_id}: rc {rc}, {seconds:.0f} s, {'failed' if failed else record['status']}" + (f'; arm stops: {stop}' if stop else '') + (f'; the control, no arm stops: {control_stop}' if control_stop else ''))
        if record is not None and (record.get('scale') or {}).get('images_equal_a5p_inputs') is not True:
            raise run.Stop(f"{cell.run_id}: Q0's images are not A5P's inputs bit for bit, or the check did not finish: {record.get('reason')}")
        if record is not None and record.get('oracle') is not False:
            raise run.Stop(f"{cell.run_id}: run.json oracle {record.get('oracle')}: the channels are not what this probe is about: {record.get('reason')}")
        if retry_of is not None and cell == mc.CONTROL:
            control = decision.control_record(retry_of, root)
            run.write_json(root/'control.json', control)
            log(f"control_bitwise {control['control_bitwise']}: recorded, no failure, nothing in the reading changes")
        failures = failures+1 if failed else 0
        if failures >= run.MAX_CONSECUTIVE_FAILURES:
            raise run.Stop(f'{failures} failures in a row')
    run.wait_for_gate(system, root, False, o.wait_minutes, log, gate)
    log(f'launch {eval_unit}')
    rc = system.run(eval_argv(code, root, eval_unit), root/'eval.systemd_run.log')
    state = run.ledger(root, plan)
    # also after a failed evaluation: a field without metrics.json counts as not read
    if retry_of is None:
        result = decision.write_analysis(root, stops=state['arms_stopped'])
    else:
        try:
            result = write_merged(root, retry_of)
        except SystemExit as error:                     # the merged reading refuses: say so in queue_stop.json, not only in the unit's exit code (plan revision 3)
            raise run.Stop(f'the merged reading: {error}') from None
    run.write_json(root/'queue_end.json', dict(end=end, eval_rc=rc, reading=result['reading'], reading_basis=result['reading_basis'], ledger=state, utc=run.utc()))
    log(f"reading: {result['reading']}. Handed to BZ: no next step starts here")
    if rc != 0:
        raise run.Stop(f'{eval_unit}: exit {rc} (metrics_error.json in the run directories, or the {run.EVAL_RUNTIME_MAX} s limit of the unit); the reading is written, the fields without metrics.json count as not read')
    return 0


def queue(o, system=None):
    retry_of = getattr(o, 'retry_of', None)
    problem = mc.v1_write_problem(o.root) or (mc.retry_problem(o.root, retry_of) if retry_of is not None else (f'{o.root}: the root of the retry, only with --retry-of' if mc.under(o.root, mc.RETRY_ROOT) else None))
    if problem:                                         # before the lock file: nothing is created or touched
        print(problem, file=sys.stderr)
        return 2
    if retry_of is None:
        return run.queue(o, system, body=queue_body, prefix='mc0p')
    return run.queue(o, system, body=lambda options, machine, root, log: queue_body(options, machine, root, log, retry_of=Path(retry_of)), prefix='mc0p')


# ---- dry run --------------------------------------------------------------------------------------------------------------------------------------
def dry_run(root=DEFAULT_ROOT, stage=None, out=print, unet=None, seg_root=None, a5p=None, mc0_root=None, allowlist=None, retry_of=None):
    """Real files only up to the npy headers of `fixed`, `moving` and of the predictions' `p`, the manifest through the firewall's reader, the predictions' provenance.json.  No truth label."""
    stage = core.STAGE if stage is None else stage
    seg_root = mc.SEG_ROOT if seg_root is None else seg_root
    a5p = mc.A5P_INPUTS if a5p is None else a5p
    mc0_root = mc.MC0_ROOT if mc0_root is None else mc0_root
    allowlist = core.ALLOWLIST if allowlist is None else allowlist
    plan = mc.plan_runs() if retry_of is None else mc.retry_plan(retry_of)
    gate, k1_unit, eval_unit = (GATE, K1_UNIT, EVAL_UNIT) if retry_of is None else (RETRY_GATE, RETRY_K1_UNIT, RETRY_EVAL_UNIT)
    out(f'MC0P dry run ({run.utc()}): no optimization, nothing written; real files read up to the npy headers of fixed, moving and p; no label member.')
    out(f'{len(plan)} runs, root {root}; oracle computed per run from its inputs; cap per run {core.RUN_CAP_SECONDS} s, run units {run.GPU_CAP_SECONDS} s, device {mc.VRAM_LIMIT_BYTES/2**30} GiB; MemoryMax {MEMORY_MAX}, nice {NICE}')
    out(f'gate: own units {gate.own_units}, load < {gate.load_limit}, GPU free >= {gate.gpu_free_mib} MiB, MemAvailable >= {gate.mem_available_gib} GiB, /home >= {run.HOME_FLOOR_GB} GB; '
        +('no allow list, no hold for the chain' if retry_of is not None else f'GPU units: no {LAST_MEMBER_MARK}, before {NO_NEW_GPU_RUN_AFTER.isoformat()}, chain not waiting, not between registration and inference'))
    out(f'evaluation: NUMBA_NUM_THREADS {EVAL_THREADS}, CPUQuota {EVAL_QUOTA}%, the certificate fallback pool {EVAL_POOL} threads')
    out('  # | arm | K | tier | D factor | decides | iterations | sigma_field | pair | unit | run_id')
    for i, c in enumerate(plan, 1):
        config = core.tier_config(c.tier)
        spec = mc.ARMS[c.arm]
        out(f"{i:3d} | {c.arm:4s} | {spec['k']:2d} | {c.tier:5s} | x{spec['D_factor']:g} | {'yes' if spec['decides'] else 'base' if c.arm == mc.BASELINE else 'no '} | {'/'.join(map(str, config['iterations']))} | {config['sigma_field']} | {c.short} | {c.unit} | {c.run_id}")
    problems = []
    if retry_of is None and len(plan) != 45:
        problems.append(f'{len(plan)} runs, not 45')
    if retry_of is not None:
        statuses = [mc.v1_status(retry_of, c) for c in mc.plan_runs()]
        control = run.read_json(core.run_dir(retry_of, mc.CONTROL)/'run.json')
        reading = decision.first_batch_reading(retry_of)
        out(f"retry of {retry_of} (read only): {statuses.count('done')} done, {statuses.count('cuda_oom')} cuda_oom, {statuses.count(None)} never run, {len(statuses)-statuses.count('done')-statuses.count('cuda_oom')-statuses.count(None)} other; {len(plan)-1} cells to run (the lead counted {mc.EXPECTED_NOT_DONE}), and the control {mc.CONTROL.run_id}, u sha256 there {control['field']['u_sha256']}")
        out(f"its own reading, quoted at the head of the merged summary: {reading['reading']}; approval: {RETRY_APPROVAL}")
        if len(plan)-1 != mc.EXPECTED_NOT_DONE:
            problems.append(f'{len(plan)-1} cells not done, the lead counted {mc.EXPECTED_NOT_DONE}')
    selection = json.loads((stage/'selection.json').read_text())
    first = tuple(row['pair'] for row in selection['pairs'][:5])
    out(f"selection.json first 5 == the plan's pairs: {first == core.PAIRS}")
    if first != core.PAIRS:
        problems.append('the first 5 pairs of selection.json differ')
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
    for pair in core.PAIRS:
        d = stage/core.require_pair(pair)
        _, metadata, manifest = _read_manifest(d/'manifest.json')
        check_pair_subjects(manifest['fixed_subject'], manifest['moving_subject'], guard)
        subjects.update({manifest['fixed_subject']: None, manifest['moving_subject']: None})
        heads = {m: run.npz_header(d/'pair.npz', m) for m in ('fixed', 'moving')}
        theirs = {m: run.npz_header(Path(a5p)/pair/'pair.npz', m) for m in ('fixed', 'moving')} if (Path(a5p)/pair/'pair.npz').is_file() else None
        mc0_there = all((core.run_dir(mc0_root, core.Cell(p, tier, pair))/'metrics.json').is_file() for p, tier in (('P0', 'long'), ('P0', 'fine'), ('P1', 'long'), ('P2', 'long'), ('P2', 'fine')))
        out(f"{core.short_name(pair)}: {d/'pair.npz'}; subjects on the allowlist ({manifest['fixed_subject']}, {manifest['moving_subject']}); spacing {metadata['spacing_xyz']}; "
            + ', '.join(f'{m} {h[0]} {h[1]}' for m, h in heads.items())+f"; A5P's images {theirs}; MC0 metrics for the ratios there: {mc0_there}")
        if len({tuple(h[0]) for h in heads.values()}) != 1:
            problems.append(f'{pair}: image shapes differ')
        if theirs != heads:
            problems.append(f"{pair}: A5P's input is missing or has other shapes")
        if not mc0_there:
            problems.append(f'{pair}: MC0 metrics missing (the ratios would be empty)')
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
    table = core.load_table()
    groups = core.load_groups()
    out(f'colour table sha256 {core.COLOUR_TABLE_SHA256} pinned and read: {len(table)} labels; structure13_v1: {len(groups)} groups; Q1 K = {len(groups)+1}, Q2, Q2s, Q3, Q3x3, Q3h K = {len(core.LABELS_35)}')
    out(f"frozen {core.FROZEN}: preopt.py sha256 equal to the pin: {core.sha_file(core.FROZEN/'jd_curl/three_d/preopt.py') == core.FROZEN_PREOPT_SHA256}; the units' PYTHONPATH puts it first")
    out(f'/home free {shutil.disk_usage("/home").free/2**30:.0f} GB')
    code = root/'code'
    out('--- start strings (the queue: the lead; the others: the queue, one per run; the code at $ROOT/code)')
    out(shlex.join(queue_launch(root, retry_of=retry_of)))
    out(shlex.join(k1_argv(code, root, k1_unit)))
    for c in plan[:10]:
        out(shlex.join(run_argv(c, code, root, '<sha>', retry_of)))
    out(f'... (the other {len(plan)-10} differ only in --arm, --tier, --pair, --out, the unit and the log)')
    out(shlex.join(eval_argv(code, root, eval_unit)))
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv == ['--dry-run']:
        return dry_run()
    if argv[:2] == ['--dry-run', '--retry-of'] and len(argv) == 3:
        problem = mc.retry_problem(mc.RETRY_ROOT, argv[2])
        if problem:
            print(problem, file=sys.stderr)
            return 2
        return dry_run(root=mc.RETRY_ROOT, retry_of=Path(argv[2]))
    p = argparse.ArgumentParser(description='MC0P: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(DEFAULT_ROOT))
    q.add_argument('--commit', default=None)
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--retry-of', default=None)
    r = sub.add_parser('run-one')
    r.add_argument('--arm', required=True)
    r.add_argument('--tier', required=True)
    r.add_argument('--pair', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--device', default='cuda')
    r.add_argument('--code-sha', default='n/a')
    r.add_argument('--retry-of', default=None)
    sub.add_parser('evaluate').add_argument('--root', default=str(DEFAULT_ROOT))
    d = sub.add_parser('decide')
    d.add_argument('--root', default=str(DEFAULT_ROOT))
    d.add_argument('--merge', default=None)
    s = sub.add_parser('self-check-k1')
    s.add_argument('--device', default='cpu')
    s.add_argument('--out', default=None)
    o = p.parse_args(argv)
    if o.command in ('queue', 'evaluate', 'decide') and Path(o.root).name == SEED19_ROOT_NAME:
        print(f'{o.root}: the root of SEED19: experiments/mc0/run_mc0pc.py; nothing read, nothing moved', file=sys.stderr)
        return 2
    if o.command in ('evaluate', 'decide'):
        problem = decision.root_problem(o.root)
        if problem:
            print(problem, file=sys.stderr)
            return 2
    if o.command == 'decide':
        problem = mc.retry_problem(o.root, o.merge) if o.merge is not None else (f'{o.root}: the root of the retry: decide it with --merge {mc.RETRY_OF_ROOT}' if mc.under(o.root, mc.RETRY_ROOT) else None)
        if problem:
            print(problem, file=sys.stderr)
            return 2
    if o.command == 'queue':
        return queue(o)
    if o.command == 'run-one':
        problem = mc.v1_write_problem(o.out)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        return run_one(o)
    if o.command == 'evaluate':
        return evaluate_all(o)
    if o.command == 'decide':
        if o.merge is not None:
            write_merged(Path(o.root), Path(o.merge))
        else:
            decision.write_analysis(o.root, stops=run.ledger(o.root, mc.plan_runs())['arms_stopped'])
        return 0
    return run.self_check_k1(o)


if __name__ == '__main__':
    sys.exit(main())
