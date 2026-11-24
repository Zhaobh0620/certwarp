"""EF (docs/bayesreg/notes/提速与换预分割器规划_20261104.md §6; runs/20261104-ef/plan.md): the driver of the entry probe.  `oracle: false`, but not a method: entries of development pairs, no native level, in no result table.
Arms and guards: ef_core.py.  Faster channels: fast_channels.py, vector_colour.py.  Repair: repair.py.  Reading: decide_ef.py.  Optimizer, scale norms, evaluation, certificate record, queue skeleton: MC0's and MC0P's (experiments/mc0), unchanged behind one optional argument.

    python -B experiments/ef/run_ef.py --dry-run [--with-colouring]
        prints all runs, checks the inputs: selection.json, the npy headers of fixed, moving, the predictions, the source fields of E3 (sha256 against their run.json, tapered hash in their metrics.json), the start strings.  No optimization, no truth label, nothing written.
        --with-colouring also reads the first 5 pairs' predictions and prints the K = 8 colouring (no optimizer, no truth label).
    python -B experiments/ef/run_ef.py queue [--commit SHA] [--wait-minutes 30] [--gap-seconds 5] [--min-host-gib 15]
        the lead's start, in the unit the dry run prints.  Code: a detached worktree $ROOT/code of --commit (default HEAD; tracked files clean).  First the checks of the dry run: anything missing stops before batch.json and before any run (plan revision 1).
        Then two phases.  E1: every run in plan order, one unit tag-ef-<arm>-<tier>-<ffff>-<mmmm> at a time (MemoryMax 8G, nice 5), each behind the gate: no $ROOT/hold, no other tag-* or seq-vp-* unit, 1-minute load < 1.5, GPU free >= 15 GiB, MemAvailable >= 7 GiB, /home >= 100 GB, /mnt/c >= --min-host-gib; closed for --wait-minutes: STOP.  --gap-seconds between runs.
        A run directory that exists is never rerun, failures are kept.  An arm stops on the 9 GiB device limit, a device out-of-memory error or a K = 35 run over 15 minutes.  3 failures in a row, images that are not A5P's in Q2, a record that is not oracle false: STOP, no evaluation, no E3.  Then one CPU unit tag-ef-eval (numba 8) and the reading of E1.
        E3: the repairs, the same way; a STOP here stops only E3 (queue_stop_e3.json): what E1 read stands.  Then tag-ef-eval-e3 and the reading again.  A failed evaluation still gets the reading, and the queue ends with exit 8.
    python -B experiments/ef/run_ef.py smoke-repair --device cuda
        the lead's smoke test before the queue, on the GPU, in the unit the dry run prints: a small synthetic field through repair_field and the real certify_map, value and gradient against the CPU.  Writes nothing.  Exit 8 if a check fails.
    run_ef.py run-one --arm L150 --tier long --pair PAIR --out DIR [--device cuda] [--code-sha SHA]      one E1 run: field.npz, run.json
    run_ef.py repair-one --arm RF01 --tier fine --pair PAIR --out DIR [--device cuda] [--code-sha SHA]   one E3 run: repair.npz, repair.json.  No image reaches the solve and no label is read anywhere.
    run_ef.py evaluate                   every field and repair without metrics: taper, .9^k certificate, the evaluate.py metrics; the only place the truth labels are read
    run_ef.py decide                     analysis/decision_ef.json, long.csv, summary_ef.md

The only root is /home/rmcfz/seq_vp_tag_results/ef_v1: every entry that writes refuses any other.  run-one, repair-one and evaluate need PYTHONPATH=<frozen>:<code> (the unit sets it).  Exit codes: 0 done, 2 usage, another driver on the root, another root, 8 STOP, a failed check, or E3 stopped (E1's reading stands).
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
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent/'mc0'))

import numpy as np  # noqa: E402

import decide_ef as decision  # noqa: E402
import ef_core as ef  # noqa: E402
import fast_channels as fast  # noqa: E402
import mc0_core as core  # noqa: E402
import mc0p_core as mc  # noqa: E402
import repair  # noqa: E402
import run_mc0 as run  # noqa: E402
import run_mc0p as mc0p  # noqa: E402
import vector_colour as vc  # noqa: E402

SCRIPT = 'experiments/ef/run_ef.py'
QUEUE_UNIT, EVAL_UNIT, EVAL_E3_UNIT, SMOKE_UNIT = 'tag-ef-queue', 'tag-ef-eval', 'tag-ef-eval-e3', 'tag-ef-smoke'
MEMORY_MAX, NICE = '8G', 5                      # as MC0P's: its timings are the ones compared with
EVAL_THREADS, EVAL_QUOTA = 8, 800               # the machine is free here (the gate); a thread count cannot change a certificate
REPAIR_THREADS, REPAIR_QUOTA = 8, 800           # the certificate inside the repair, as the solver runs it
RUN_THREADS = 1                                 # E1's numba: no certificate in the unit
GAP_SECONDS = 5                                 # the lead, plan revision 1 (the brief §4 had 60): the seconds' clock is inside the process, the load gate stays, mc0p_v1b had no 60 s gap
DEFAULT_MIN_HOST_GIB = 15.                      # the brief §4
GATE = replace(mc0p.RETRY_GATE, own_units=(f'{QUEUE_UNIT}.service',), host_floor_gib=DEFAULT_MIN_HOST_GIB)     # load < 1.5, GPU free >= 15 GiB, MemAvailable >= 7 GiB, /home >= 100 GB
TRUTH_MEMBERS = core.PAIR_MEMBERS[2:]


# ---- start strings ------------------------------------------------------------------------------------------------------------------------
def unit_argv(unit, code, inner, *, log, gpu=True, numba=RUN_THREADS, quota=800, runtime=run.RUN_RUNTIME_MAX):
    return run.unit_argv(unit, code, inner, log=log, gpu=gpu, numba=numba, quota=quota, runtime=runtime, script=SCRIPT, memory_max=MEMORY_MAX, nice=NICE)


def run_argv(cell, code, root, sha='n/a'):
    d = core.run_dir(root, cell)
    command = 'repair-one' if cell.is_repair else 'run-one'
    return unit_argv(cell.unit, code, [command, '--arm', cell.arm, '--tier', cell.tier, '--pair', cell.pair, '--out', str(d), '--code-sha', sha], log=d/'log.txt', numba=REPAIR_THREADS if cell.is_repair else RUN_THREADS, quota=REPAIR_QUOTA)


def eval_argv(code, root, unit=EVAL_UNIT):
    return unit_argv(unit, code, ['evaluate', '--root', str(root)], log=f'{root}/eval.log', gpu=False, numba=EVAL_THREADS, quota=EVAL_QUOTA, runtime=run.EVAL_RUNTIME_MAX)


def smoke_launch(repo=core.MAIN_REPO):
    """The lead's start string of the smoke test: the output on the terminal, nothing written."""
    env = dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, NUMBA_NUM_THREADS=REPAIR_THREADS, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=f'{core.FROZEN}:{repo}')
    return run.check_launch(['systemd-run', '--user', f'--unit={SMOKE_UNIT}', '--wait', '--collect', '--pipe', f'--working-directory={repo}', f'--property=MemoryMax={MEMORY_MAX}', '--property=RuntimeMaxSec=600',
                             *[f'--setenv={k}={v}' for k, v in env.items()], run.PY, '-B', SCRIPT, 'smoke-repair', '--device', 'cuda'])


def queue_launch(root=None, repo=core.MAIN_REPO, gap=GAP_SECONDS, min_host_gib=DEFAULT_MIN_HOST_GIB):
    """The lead's start string of the queue."""
    root = ef.DEFAULT_ROOT if root is None else root
    return run.check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT}', '--collect', f'--working-directory={repo}', f'--property=MemoryMax={MEMORY_MAX}', f'--nice={NICE}', '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1',
                             run.PY, '-B', SCRIPT, 'queue', '--root', str(root), '--gap-seconds', str(gap), '--min-host-gib', f'{min_host_gib:g}'])


# ---- one E1 run ---------------------------------------------------------------------------------------------------------------------------------
def labels_digest(labels):
    return {side: hashlib.sha256(np.ascontiguousarray(labels[side]).tobytes()).hexdigest() for side in ('fixed', 'moving')}


def scale_check_of(record, scale):
    """The old path's second data_grad_norm, without a second upload: the norm of the native tensors the loop ran on (optimize_k's record).  A wall cap on a coarse level never reached them: passed None, the status is time_cap anyway."""
    norms = record['level_g_norm']
    if len(norms) < 4:
        return dict(g_norm_scaled=None, g_norm_Q0=scale['g_norm_Q0'], tol=core.SCALE_TOL, passed=None)
    scaled = norms[3]['raw']
    return dict(g_norm_scaled=scaled, g_norm_Q0=scale['g_norm_Q0'], tol=core.SCALE_TOL, passed=abs(scaled/scale['g_norm_Q0']-1) <= core.SCALE_TOL)


def run_baseline(o, cell):
    """Q2: MC0P's run_one, its old channel path, its first-5-pairs guard; nothing of EF between it and the optimizer."""
    return mc0p.run_one(o, cell=cell)


def run_fast(o, cell):
    """One arm on the faster channels.  The flow, the records and the stops of mc0p.run_one; no Q0 comparison with A5P's file (Q2 of the same pair does it; g_norm_Q0 and D_multiplier tie them, decide_ef.controls), no r_l of the levels."""
    t0 = perf_counter()
    spec = ef.ARMS[cell.arm]
    fz = core.frozen_api()
    import torch
    out = Path(o.out)
    if (out/'started.json').exists():
        raise SystemExit(f'{out}: this run has started before; a run is never rerun')
    out.mkdir(parents=True, exist_ok=True)
    run.write_json(out/'started.json', dict(run_id=cell.run_id, utc=run.utc(), argv=sys.argv, pid=os.getpid()))
    device_cap = mc0p.limit_device_memory(torch, o.device)
    pin = ef.read_images(cell)                          # the firewall view: no label member is decoded, here or below
    config = fz.PreoptConfig(**ef.arm_config(cell.arm))
    status, reason, u, record, scale, seconds_labels, seconds_channels, channel_sources, digest, scale_check = 'done', None, None, None, None, None, None, None, None, None
    try:
        predicted = fast.read_predicted(pin, workers=core.SETUP_WORKERS)
        seconds_labels = perf_counter()-t0
        channel_sources = predicted.record
        digest = labels_digest(predicted.labels)
        if channel_sources['oracle'] is not False:
            status, reason = 'not_predicted_channels', f"channel sources: oracle {channel_sources['oracle']}, members read {channel_sources['pair_keys_read']}; no optimization"
        else:
            stacks, scale = fast.prepare_stacks(spec, pin, predicted, core.load_table(), o.device, workers=core.SETUP_WORKERS)
            del predicted
            seconds_channels = perf_counter()-t0
            u, record = core.optimize_k(stacks, pin.grid, config, device=o.device, started=t0, observer=run.vram_guard(torch, o.device, mc.VRAM_LIMIT_BYTES), workers=core.SETUP_WORKERS, record_level_norms=True, downsample=fast.FastDownsample())
            del stacks
            record['role'] = ef.ROLE
            scale_check = scale_check_of(record, scale)
            if record['stopped'] is not None:
                status, reason = 'time_cap', f"the wall cap {core.RUN_CAP_SECONDS} s at {record['stopped']}"
            elif not scale_check['passed']:
                status, reason = 'scale_mismatch', f"||g_D|| of the native tensors {scale_check['g_norm_scaled']} differs from Q0 {scale['g_norm_Q0']}"
    except vc.ColouringFailed as error:
        status, reason = 'colouring_failed', str(error)
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
        schema='bayesreg.ef.run.v1', run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, k=spec['k'], native_iterations=spec['native'], channels=spec['channels'], oracle=None if channel_sources is None else channel_sources['oracle'], decides=spec['decides'], status=status, reason=reason, role=ef.ROLE,
        config=ef.arm_config(cell.arm), scale=scale, scale_check=scale_check, labels_sha256=digest, channel_sources=channel_sources, optimizer=record, field=field, taper=core.TAPER, code_sha=o.code_sha, frozen_commit=core.FROZEN_COMMIT, random_seed='none: no random step',
        data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], fixed_subject=pin.fixed_subject, moving_subject=pin.moving_subject, stage=str(core.STAGE)),
        env=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if cuda else 'cpu'), device_cap=device_cap,
        device_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None, device_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if cuda else None,
        host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, seconds_labels=seconds_labels, seconds_channels=seconds_channels, seconds_total=perf_counter()-t0, ended_utc=run.utc()))
    return 0


def run_one(o):
    cell = ef.require_cell(o.arm, o.tier, o.pair)
    if cell.is_repair:
        raise SystemExit(f'{cell.run_id}: a repair arm: repair-one')
    problem = ef.root_problem(Path(o.out).parents[2])     # <root>/runs/<arm>_<tier>/<ffff>-<mmmm>
    if problem:
        raise SystemExit(problem)
    return run_baseline(o, cell) if cell.arm == ef.BASELINE else run_fast(o, cell)


# ---- one E3 run ---------------------------------------------------------------------------------------------------------------------------------
def tiny_warmup(certify, config, locate):
    """The numba kernels of certify_map and of the locator, compiled before the clock starts (repair_推导.md §7): a small folded field through every path.  A compile is once per process, not a cost of a repair; it is recorded as seconds_warmup.
    Numba compiles one version per array type and a read-only array is another type: the first locate of a repair is on the read-only tapered field, later ones on the writable attempt, so both are compiled here (EF_tools_复核 B1: otherwise about 5 s of compile inside seconds_repair)."""
    from jd_curl.three_d.grid import make_grid_3d
    grid = make_grid_3d(nx=8, ny=8, nz=8, hx=1., hy=1., hz=1., unit='mm')
    u = np.zeros(grid.shape+(3,))
    certify(u, grid, config=config)
    u[3, 3, 3] = (1.6, 1.6, 1.6)
    certify(u, grid, config=config)
    locate(u, grid, config)
    u.setflags(write=False)
    certify(u, grid, config=config)
    locate(u, grid, config)


def device_warmup(device, certify, config):
    """The first use of the device's kernels and of cuBLAS, before the clock starts (EF_tools_复核 R1), the same reason as tiny_warmup: every repair-one is a new process, and that cost is once per process, not a cost of a repair.
    A very small repair_field on the device, weak enough not to pass in one round: the table branch of the objective and its flux, L-BFGS, lattice_points.  Then the per-row branch with its flux, directly: on this field the smallest J of a located cell may be on closed27, and then no round gets a per-row point.  Nothing of it is used; a synthetic field, no file."""
    import torch
    u, grid = repair.synthetic_folded_field(n=8)
    cells = repair.locate_cells(u, grid, config)
    settings = repair.RepairConfig(eps=.05, rho0=1e-3, max_rounds=2, max_inner=2)
    result = repair.repair_field(u, grid, cells, settings, certify=certify, certificate_config=config, device=device)
    some = cells[:4]
    objective = repair.Objective(torch, u, u, grid, repair.mask_nodes(cells, grid, 0), repair.union53(grid)[1], device=device, tile_cells=settings.tile_cells, extras=(some, np.full((len(some), 3), .25)))
    objective.value_and_grad(objective.start, 1., 1e3)  # eps above every J: every point active
    return dict(role='warm-up on a synthetic field, nothing of it is used', device=str(device), n_located=len(cells), rounds=len(result.rounds), n_per_row_points=len(some))


def source_checked(cell, fz, grid, results=None):
    """(tapered u, record of the source): the field a repair starts from, after every check.  The run's field.npz against its run.json, the taper of the evaluation, its hash against that metrics.json.  STOP otherwise."""
    d = ef.source_dir(cell, results)
    source_run, source_metrics = run.read_json(d/'run.json'), run.read_json(d/'metrics.json')
    if source_run is None or source_metrics is None:
        raise SystemExit(f'{d}: no run.json or metrics.json')
    if source_run.get('status') != 'done' or source_run.get('oracle') is not False or (source_run.get('arm'), source_run.get('tier')) != ('Q2', cell.tier):
        raise SystemExit(f"{d}: not a finished Q2 {cell.tier} with oracle false: {source_run.get('status')}, {source_run.get('oracle')}, {source_run.get('arm')} {source_run.get('tier')}")
    if core.sha_file(d/'field.npz') != source_run['field']['sha256']:
        raise SystemExit(f'{d}: field.npz differs from its run.json')
    with np.load(d/'field.npz') as z:
        raw = np.array(z['u'])
    u, _ = fz.boundary_taper(raw, grid, core.TAPER)
    u = np.ascontiguousarray(u)
    digest = hashlib.sha256(u.tobytes()).hexdigest()
    if digest != source_metrics['tapered_u_sha256']:
        raise SystemExit(f"{d}: the tapered field {digest} is not the one its metrics.json evaluated, {source_metrics['tapered_u_sha256']}")
    return u, dict(directory=str(d), run_id=source_run['run_id'], field_sha256=source_run['field']['sha256'], tapered_u_sha256=digest, code_sha=source_run.get('code_sha'), accepted_alpha=source_metrics['certificate']['accepted_alpha'], n_infeasible_at_1=source_metrics['certificate']['n_infeasible_at_1'])


def repair_config(cell):
    return repair.RepairConfig(eps=ef.REPAIR_ARMS[cell.arm]['eps'])


def repair_one(o):
    """No image, no label reaches the repair: read_images gives the grid and its firewall; the field, the grid and eps are all repair_field sees.  An existing repair.npz is never redone."""
    cell = ef.require_cell(o.arm, o.tier, o.pair)
    if not cell.is_repair:
        raise SystemExit(f'{cell.run_id}: not a repair arm: run-one')
    problem = ef.root_problem(Path(o.out).parents[2])
    if problem:
        raise SystemExit(problem)
    t0 = perf_counter()
    fz = core.frozen_api()
    import torch
    from jd_curl.three_d.feasibility import certify_map
    out = Path(o.out)
    if (out/'started.json').exists():
        raise SystemExit(f'{out}: this run has started before; a run is never rerun')
    out.mkdir(parents=True, exist_ok=True)
    run.write_json(out/'started.json', dict(run_id=cell.run_id, utc=run.utc(), argv=sys.argv, pid=os.getpid()))
    device_cap = mc0p.limit_device_memory(torch, o.device)
    pin = ef.read_images(cell)
    u, source = source_checked(cell, fz, pin.grid)
    config = run.certificate_config()                   # configuration C, explicitly: delta 0, depth 48, 255 boxes, the frozen driver's tile
    tick = perf_counter()
    tiny_warmup(certify_map, config, repair.locate_cells)
    seconds_warmup_numba = perf_counter()-tick
    tick = perf_counter()
    warmed = device_warmup(o.device, certify_map, config)
    seconds_warmup_device = perf_counter()-tick
    settings = repair_config(cell)
    tick = perf_counter()
    cells = repair.locate_cells(u, pin.grid, config)
    seconds_first_locate = perf_counter()-tick
    tick = perf_counter()
    result = repair.repair_field(u, pin.grid, cells, settings, certify=certify_map, certificate_config=config, device=o.device)
    seconds_repair = seconds_first_locate+perf_counter()-tick    # locate, solve and every round's certificate: what a repair costs on a compiled process
    np.savez(out/'repair.npz', nodes=result.attempt_nodes, values=result.attempt_values)
    cuda = torch.device(o.device).type == 'cuda'
    run.write_json(out/'repair.json', dict(
        schema='bayesreg.ef.repair.v1', run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, oracle=False, role=repair.ROLE, status='certified' if result.passed else 'failed_back_to_frozen_rule', passed=result.passed, reason=result.reason, eps=settings.eps, config=asdict(settings), source=source, n_located=len(cells),
        initial_certificate=result.initial_certificate, certificate=result.certificate, rounds=result.rounds, seconds_warmup=seconds_warmup_numba+seconds_warmup_device, seconds_warmup_numba=seconds_warmup_numba, seconds_warmup_device=seconds_warmup_device, device_warmup=warmed, seconds_first_locate=seconds_first_locate, seconds_repair=seconds_repair, attempt=dict(file='repair.npz', sha256=core.sha_file(out/'repair.npz'), n_free_nodes=len(result.attempt_nodes)),
        certificate_config=asdict(config), code_sha=o.code_sha, frozen_commit=core.FROZEN_COMMIT, data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], fixed_subject=pin.fixed_subject, moving_subject=pin.moving_subject, keys_read=list(pin.provenance['keys_read'])),
        env=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if cuda else 'cpu'), device_cap=device_cap, device_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None, device_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if cuda else None,
        host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, seconds_total=perf_counter()-t0, ended_utc=run.utc()))
    return 0


def smoke_repair(o):
    """A small synthetic field on o.device, before the queue (EF_tools_复核 I5): the objective against the CPU at one point (float64, relative difference <= 1e-12), then repair_field on the device, then certify_map again on what it returned.  No file is read or written; prints one JSON line."""
    import torch
    from jd_curl.three_d.feasibility import certify_map
    config = run.certificate_config()
    device_cap = mc0p.limit_device_memory(torch, o.device)
    tick = perf_counter()
    tiny_warmup(certify_map, config, repair.locate_cells)
    seconds_warmup = perf_counter()-tick
    tick = perf_counter()
    warmed = device_warmup(o.device, certify_map, config)
    seconds_warmup_device = perf_counter()-tick
    u, grid = repair.synthetic_folded_field()
    before = certify_map(u, grid, config=config)
    cells = repair.locate_cells(u, grid, config)
    settings = repair.RepairConfig(eps=.05)
    free = repair.mask_nodes(cells, grid, settings.radius0)
    _, table = repair.union53(grid)
    extra_cells, extra_points = repair.lattice_points(u, grid, cells, 1e9, device=o.device)     # with some per-row points too: the cells whose smallest J is off closed27 (a cell whose minimum is on closed27 gives none)
    extras = (extra_cells[:16], extra_points[:16])
    values = {}
    for device in ('cpu', o.device):
        objective = repair.Objective(torch, u, u, grid, free, table, device=device, tile_cells=settings.tile_cells, extras=extras)
        energy, gradient, _ = objective.value_and_grad(objective.start+.01, settings.rho0, settings.eps)
        values[device] = (energy, gradient.detach().cpu().numpy())
    energy_cpu, gradient_cpu = values['cpu']
    energy_device, gradient_device = values[o.device]
    energy_difference = abs(energy_device-energy_cpu)/max(abs(energy_cpu), 1e-300)
    gradient_difference = float(np.abs(gradient_device-gradient_cpu).max()/max(np.abs(gradient_cpu).max(), 1e-300))
    tick = perf_counter()
    result = repair.repair_field(u, grid, cells, settings, certify=certify_map, certificate_config=config, device=o.device)
    seconds_repair = perf_counter()-tick
    again = certify_map(result.u, grid, config=config)
    outside = np.setdiff1d(np.arange(u.size//3), result.attempt_nodes)
    checks = dict(input_not_certified=not repair.accepted(before), repair_passed=result.passed is True, certified_again=repair.accepted(again), state_hash_same=result.certificate['state_hash'] == again.state_hash,
                  outside_mask_bitwise=result.u.reshape(-1, 3)[outside].tobytes() == u.reshape(-1, 3)[outside].tobytes(), per_row_points_compared=len(extras[0]) > 0, energy_within_1e_12=energy_difference <= 1e-12, gradient_within_1e_12=gradient_difference <= 1e-12)
    cuda = torch.device(o.device).type == 'cuda'
    print(json.dumps(dict(passed=all(checks.values()), checks=checks, device=torch.cuda.get_device_name(0) if cuda else 'cpu', torch=torch.__version__, device_cap=device_cap, energy_relative_difference=energy_difference, gradient_relative_difference=gradient_difference, n_infeasible_before=before.n_infeasible,
                          n_located=len(cells), n_free=len(free), n_per_row_points=len(extras[0]), device_warmup=warmed, rounds=[dict(round=r['round'], rho=r['rho'], radius=r['radius'], n_free=r['n_free'], n_extra_points=r['n_extra_points'], iterations=r['inner']['iterations'], stop=r['inner']['stop'], seconds_solve=r['seconds_solve'], certificate=r['certificate']['status'], n_infeasible=r['certificate']['n_infeasible'], min_jd_lower_bound=r['certificate']['min_jd_lower_bound']) for r in result.rounds],
                          seconds_warmup=seconds_warmup, seconds_warmup_device=seconds_warmup_device, seconds_repair=seconds_repair, device_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if cuda else None, role='a smoke test on a synthetic field: its seconds are not the seconds of a repair of a real field')))
    return 0 if all(checks.values()) else 8


# ---- evaluation --------------------------------------------------------------------------------------------------------------------------------
def repaired_field(u, d, record):
    """u with the attempt's node values, after checking the file against the record."""
    if core.sha_file(d/'repair.npz') != record['attempt']['sha256']:
        raise ValueError('repair.npz differs from repair.json')
    with np.load(d/'repair.npz') as z:
        nodes, values = np.array(z['nodes']), np.array(z['values'])
    v = np.array(u, copy=True)
    v.reshape(-1, 3)[nodes] = values
    return v


def evaluate_repair(d, cell, labs, pin, fz, ev, certify, config):
    """After: the repaired field, certificate at alpha = 1 (the evaluation's own), metrics.  Before: the source's metrics.json, for the same tapered field.  Failed: the attempt's numbers are kept and the fallback is what the frozen rule gave on the input, from the same metrics.json.  Written from the fields, nothing from the run's opinion."""
    record = run.read_json(d/'repair.json')
    if record is None:
        raise ValueError('no repair.json')
    source_dir = ef.source_dir(cell)
    u, source = source_checked(cell, fz, pin.grid)
    if source['tapered_u_sha256'] != record['source']['tapered_u_sha256']:
        raise ValueError('the source field changed since the repair')
    source_metrics = run.read_json(source_dir/'metrics.json')
    v = repaired_field(u, d, record)
    tick = perf_counter()
    ladder = run.certify_ladder(v, pin.grid, (1.,), certify, config)
    seconds = perf_counter()-tick
    metrics = run.field_metrics(v, labs, tuple(pin.grid.spacing), ev)
    passed = ladder['accepted_alpha'] == 1.
    fallback = None if passed else dict(rule='the frozen .9^k rule on the input field (from the source metrics.json)', accepted_alpha=source_metrics['certificate']['accepted_alpha'], at_accepted_alpha=source_metrics.get('at_accepted_alpha'))
    run.write_json(d/'repair_metrics.json', dict(schema='bayesreg.ef.repair_metrics.v1', run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, eps=ef.REPAIR_ARMS[cell.arm]['eps'], oracle=False, evaluated_utc=run.utc(), field='the repaired field (no second taper)', repaired_u_sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(),
                                                   before=dict(source=source['directory'], tapered_u_sha256=source['tapered_u_sha256'], metrics=source_metrics['metrics'], certificate_accepted_alpha=source_metrics['certificate']['accepted_alpha'], n_infeasible_at_1=source_metrics['certificate']['n_infeasible_at_1'], at_accepted_alpha=source_metrics.get('at_accepted_alpha')),
                                                   after=dict(metrics=metrics, certificate=ladder, seconds_certificate=seconds), change=repair.change_record(u, v), fallback=fallback, evaluator=dict(script='experiments/d0/evaluate.py', sha256=core.sha_file(core.ROOT/'experiments/d0/evaluate.py'))))


def evaluate_all(o, *, plan=None):
    """As run_mc0p.evaluate_all.  The truth labels enter here, for the metrics only."""
    fz = core.frozen_api()
    ev = core.load_module('_mc0_evaluate', 'experiments/d0/evaluate.py')
    from jd_curl.three_d.feasibility import certify_map
    config = run.certificate_config()
    failures, loaded = 0, {}
    for cell in (ef.plan_runs() if plan is None else plan):
        d = core.run_dir(o.root, cell)
        done_names = ('repair_metrics.json',) if cell.is_repair else ('metrics.json',)
        ready = (d/'repair.json').is_file() if cell.is_repair else (d/'field.npz').is_file()
        if not ready or any((d/name).exists() for name in done_names+('metrics_error.json',)):
            continue
        try:
            if cell.pair not in loaded:
                loaded = {cell.pair: ef.read_truth(cell.pair, TRUTH_MEMBERS)}     # one pair in memory
            pin, labs = loaded[cell.pair]
            if cell.is_repair:
                evaluate_repair(d, cell, labs, pin, fz, ev, certify_map, config)
            else:
                spent = dict(seconds=0.)

                def timed(field, grid, *, config):
                    tick = perf_counter()
                    result = certify_map(field, grid, config=config)
                    spent['seconds'] += perf_counter()-tick
                    return result
                run.evaluate_one(d, cell, labs, pin, fz, ev, timed, config, oracle=mc.oracle_of_run(run.read_json(d/'run.json') or {}), schema='bayesreg.ef.metrics.v1')
                run.write_json(d/'ladder_seconds.json', dict(seconds=spent['seconds'], role='the certificates of the .9^k rule in the evaluation unit: numba 8, one process, compiled after the first field'))
        except Exception as error:                     # kept; the others go on; not retried
            run.write_json(d/'metrics_error.json', dict(run_id=cell.run_id, error=repr(error)[:2000], utc=run.utc()))
            failures += 1
    return 8 if failures else 0


# ---- the queue ------------------------------------------------------------------------------------------------------------------------------------
def run_phase(o, system, root, log, code, sha, cells, plan, gate):
    """One phase of the plan, in order.  Returns how it ended; run.Stop on 3 failures in a row, a closed gate, images that are not A5P's in Q2, a record that is not oracle false."""
    failures, end, first = 0, 'all runs handled', True
    for cell in cells:
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
        k = 1 if cell.is_repair else ef.ARMS[cell.arm]['k']
        stop = run.arm_stop_reason(cell, record, seconds, k=k, vram_limit=mc.VRAM_LIMIT_BYTES)
        state_name = 'failed' if failed else record['status']
        run.write_json(d/'queue_done.json', dict(run_id=cell.run_id, unit=cell.unit, rc=rc, unit_seconds=seconds, state=state_name, arm_stop=stop, ended_utc=run.utc()))
        log(f"ended {cell.run_id}: rc {rc}, {seconds:.0f} s, {state_name}" + (f'; arm stops: {stop}' if stop else ''))
        if cell.arm == ef.BASELINE and record is not None and (record.get('scale') or {}).get('images_equal_a5p_inputs') is not True:
            raise run.Stop(f"{cell.run_id}: Q0's images are not A5P's inputs bit for bit, or the check did not finish: {record.get('reason')}")
        if not cell.is_repair and record is not None and record.get('oracle') is not False:
            raise run.Stop(f"{cell.run_id}: run.json oracle {record.get('oracle')}: the channels are not what this probe is about: {record.get('reason')}")
        failures = failures+1 if failed else 0
        if failures >= run.MAX_CONSECUTIVE_FAILURES:
            raise run.Stop(f'{failures} failures in a row')
    return end


def queue_body(o, system, root, log):
    """Two phases (plan revision 1).  A STOP in E1 or at E1's evaluation gate ends the queue as it always did.  From E1's reading on, nothing that happens to E3 can take it away."""
    for name in ('queue_stop', 'queue_stop_e3'):        # evidence, kept under its time; an old stop of E3 is not this run's either
        if (root/f'{name}.json').exists():
            (root/f'{name}.json').rename(root/f"{name}_{run.utc().replace(':', '').replace('-', '')}.json")
    code, sha = system.code_checkout(root, o.commit)
    problems = safe_input_problems()
    if problems:
        raise run.Stop(f'the inputs, before any run: {problems}')
    plan = ef.plan_runs()
    e1, e3 = [c for c in plan if not c.is_repair], [c for c in plan if c.is_repair]
    gate = replace(GATE, host_floor_gib=o.min_host_gib)
    batch = run.read_json(root/'batch.json')
    if batch is None:
        run.write_json(root/'batch.json', dict(schema=ef.BATCH_SCHEMA, code_sha=sha, frozen_commit=core.FROZEN_COMMIT, started_utc=run.utc(), plan=[c.run_id for c in plan], gap_seconds=o.gap_seconds, phases=dict(E1=len(e1), E3=len(e3)),
                                               caps=dict(run_seconds=core.RUN_CAP_SECONDS, run_units_seconds=run.GPU_CAP_SECONDS, device_gib=mc.VRAM_LIMIT_BYTES/2**30, memory_max=MEMORY_MAX, nice=NICE),
                                               gate=dict(own_units=list(gate.own_units), load_limit=gate.load_limit, gpu_free_mib=gate.gpu_free_mib, mem_available_gib=gate.mem_available_gib, home_floor_gb=run.HOME_FLOOR_GB, host_floor_gib=gate.host_floor_gib),
                                               evaluation=dict(numba_threads=EVAL_THREADS, cpu_quota_percent=EVAL_QUOTA)))
    elif batch['code_sha'] != sha:
        raise run.Stop(f"batch.json is at {batch['code_sha']}, the code at {sha}")
    elif batch['plan'] != [c.run_id for c in plan]:
        raise run.Stop('batch.json has another plan')
    log(f'start: {len(e1)} E1 runs, then {len(e3)} E3 runs; the inputs checked; /mnt/c floor {o.min_host_gib:g} GiB; {o.gap_seconds} s between runs')
    end_e1 = run_phase(o, system, root, log, code, sha, e1, plan, gate)
    run.wait_for_gate(system, root, False, o.wait_minutes, log, gate)
    log(f'launch {EVAL_UNIT}')
    rc_e1 = system.run(eval_argv(code, root), root/'eval.systemd_run.log')
    state = run.ledger(root, plan)
    result = decision.write_analysis(root, stops=state['arms_stopped'])          # also after a failed evaluation: a field without metrics counts as not read
    run.write_json(root/'queue_e1_end.json', dict(end=end_e1, eval_rc=rc_e1, e1=result['e1']['reading'], e1_faster_only_with_e2=result['e1']['faster_only_with_e2'], ledger=state, utc=run.utc()))
    log(f"E1: {decision.e1_headline(result['e1'])}; now E3, whose failure leaves this reading as it is")
    stop_e3, rc_e3, end_e3 = None, None, None
    try:
        end_e3 = run_phase(o, system, root, log, code, sha, e3, plan, gate)
    except run.Stop as stop:
        stop_e3, end_e3 = str(stop), f'STOP in E3: {stop}'
        log(f'{end_e3}; the reading of E1 stands')
        run.write_json(root/'queue_stop_e3.json', dict(reason=str(stop), utc=run.utc()))
    try:
        run.wait_for_gate(system, root, False, o.wait_minutes, log, gate)
        log(f'launch {EVAL_E3_UNIT}')
        rc_e3 = system.run(eval_argv(code, root, EVAL_E3_UNIT), root/'eval_e3.systemd_run.log')
    except run.Stop as stop:                            # by hand later: evaluate, decide
        end_e3 = f'{end_e3}; E3 not evaluated: {stop}'
        log(end_e3)
        if stop_e3 is None:
            stop_e3 = str(stop)
    state = run.ledger(root, plan)
    result = decision.write_analysis(root, stops=state['arms_stopped'])
    run.write_json(root/'queue_end.json', dict(end=end_e1, end_e3=end_e3, stop_e3=stop_e3, eval_rc=rc_e1, eval_rc_e3=rc_e3, e1=result['e1']['reading'], e1_faster_only_with_e2=result['e1']['faster_only_with_e2'], e3=result['e3']['reading'], ledger=state, utc=run.utc()))
    log(f"E1: {decision.e1_headline(result['e1'])}; E3: {result['e3']['reading']}.  Handed to the lead: no next step starts here")
    if rc_e1 != 0 or rc_e3 not in (0, None):
        raise run.Stop(f'an evaluation unit failed: {EVAL_UNIT} exit {rc_e1}, {EVAL_E3_UNIT} exit {rc_e3} (metrics_error.json in the run directories, or the {run.EVAL_RUNTIME_MAX} s limit of the unit); the reading is written, a field without metrics counts as not read')
    if stop_e3 is not None:
        raise run.Stop(f'E3 stopped, the reading of E1 stands: {stop_e3}')
    return 0


def queue(o, system=None):
    problem = ef.root_problem(o.root)
    if problem:                                         # before the lock file: nothing is created or touched
        print(problem, file=sys.stderr)
        return 2
    return run.queue(o, system, body=queue_body, prefix='ef')


# ---- the inputs ------------------------------------------------------------------------------------------------------------------------------------
def input_problems(out=lambda text: None, *, stage=None, seg_root=None, a5p=None, allowlist=None, unet=None, results=None):
    """Why this batch cannot run, as a list; the dry run prints what it read, the queue stops on any (before batch.json, before any run).  Real files only up to the npy headers of `fixed`, `moving` and of the predictions' `p`, the manifest through the firewall's reader, the predictions' provenance.json, the source fields' records and bytes.  No truth label, no optimization, nothing written; no frozen PYTHONPATH needed."""
    stage = core.STAGE if stage is None else stage
    seg_root = mc.SEG_ROOT if seg_root is None else seg_root
    a5p = mc.A5P_INPUTS if a5p is None else a5p
    allowlist = core.ALLOWLIST if allowlist is None else allowlist
    problems = []
    if len(ef.plan_runs()) != 49:
        problems.append(f'{len(ef.plan_runs())} runs, not 49')
    selection = json.loads((Path(stage)/'selection.json').read_text())
    first = tuple(row['pair'] for row in selection['pairs'][:5])
    out(f"selection.json first 5 == the plan's pairs: {first == core.PAIRS}; {ef.EXCEPTION_PAIR} in selection.json: {ef.EXCEPTION_PAIR in [row['pair'] for row in selection['pairs']]} (only in RL01, RL05)")
    if first != core.PAIRS:
        problems.append('the first 5 pairs of selection.json differ')
    if ef.EXCEPTION_PAIR not in [row['pair'] for row in selection['pairs']]:
        problems.append('the exception pair is not in selection.json')
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
    for pair in core.PAIRS+(ef.EXCEPTION_PAIR,):
        d = Path(stage)/pair
        _, metadata, manifest = _read_manifest(d/'manifest.json')
        check_pair_subjects(manifest['fixed_subject'], manifest['moving_subject'], guard)
        heads = {m: run.npz_header(d/'pair.npz', m) for m in ('fixed', 'moving')}
        out(f"{core.short_name(pair)}: {d/'pair.npz'}; subjects on the allowlist ({manifest['fixed_subject']}, {manifest['moving_subject']}); spacing {metadata['spacing_xyz']}; " + ', '.join(f'{m} {h[0]} {h[1]}' for m, h in heads.items()))
        if len({tuple(h[0]) for h in heads.values()}) != 1:
            problems.append(f'{pair}: image shapes differ')
        if pair in core.PAIRS:                                  # the channels are read for the first 5 only
            subjects.update({manifest['fixed_subject']: None, manifest['moving_subject']: None})
            theirs = {m: run.npz_header(Path(a5p)/pair/'pair.npz', m) for m in ('fixed', 'moving')} if (Path(a5p)/pair/'pair.npz').is_file() else None
            if theirs != heads:
                problems.append(f"{pair}: A5P's input is missing or has other shapes")
    admission = mi.a4_admission(list(subjects), allowlist, unet)
    out(f"D24 admission of the {len(admission['subjects'])} subjects of E1 (ids only): folds sha256 {admission['folds_sha256']}")
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
    for (tier, pair), name in ef.SOURCES.items():
        cell = ef.Cell('RL01' if tier == 'long' else 'RF01', tier, pair)
        d = ef.source_dir(cell, results)
        source, metrics = run.read_json(d/'run.json') or {}, run.read_json(d/'metrics.json') or {}
        same = (d/'field.npz').is_file() and core.sha_file(d/'field.npz') == (source.get('field') or {}).get('sha256')
        certificate = metrics.get('certificate') or {}
        out(f"E3 source {tier} {cell.short}: {d}: status {source.get('status')}, oracle {source.get('oracle')}, field.npz equal to its run.json: {same}; alpha = 1: {certificate.get('n_infeasible_at_1')} infeasible cells, accepted alpha {certificate.get('accepted_alpha')}, seg35 / seg4 {(metrics.get('metrics') or {}).get('seg35')} / {(metrics.get('metrics') or {}).get('seg4')}")
        if source.get('status') != 'done' or source.get('oracle') is not False or not same or metrics.get('tapered_u_sha256') is None:
            problems.append(f'E3 source {tier} {cell.short} is not usable')
    out(f"colour table sha256 {core.COLOUR_TABLE_SHA256} pinned and read: {len(core.load_table())} labels")
    preopt_same = core.sha_file(core.FROZEN/'jd_curl/three_d/preopt.py') == core.FROZEN_PREOPT_SHA256
    out(f"frozen {core.FROZEN}: preopt.py sha256 equal to the pin: {preopt_same}; the units' PYTHONPATH puts it first")
    if not preopt_same:
        problems.append('the frozen preopt.py differs from the pin')
    return problems


def safe_input_problems(out=lambda text: None, **kwargs):
    """input_problems, with a missing file or a refused input as a problem and not a traceback: the queue says what it was, in queue_stop.json."""
    try:
        return input_problems(out, **kwargs)
    except (Exception, SystemExit) as error:            # noqa: BLE001 - any of them means the inputs are not there
        return [f'{type(error).__name__}: {error}'[:1000]]


# ---- dry run --------------------------------------------------------------------------------------------------------------------------------------
def dry_run(root=None, stage=None, out=print, unet=None, seg_root=None, a5p=None, allowlist=None, results=None, with_colouring=False):
    """The plan, the checks of the queue (input_problems), the start strings.  No truth label, no optimization, nothing written.  with_colouring: also the predictions' values."""
    root = ef.DEFAULT_ROOT if root is None else root
    plan = ef.plan_runs()
    gate = GATE
    out(f"EF dry run ({run.utc()}): no optimization, no truth label, nothing written; real files read up to the npy headers of fixed, moving and p, the source fields of E3 and their records (their seg35, seg4 below are the earlier evaluation's numbers){'; --with-colouring: also the predictions of the first 5 pairs' if with_colouring else ''}.")
    out(f'{len(plan)} runs ({sum(not c.is_repair for c in plan)} E1, then their evaluation; {sum(c.is_repair for c in plan)} E3, then theirs), root {root}; root exists: {Path(root).exists()}; cap per run {core.RUN_CAP_SECONDS} s, run units {run.GPU_CAP_SECONDS} s, device {mc.VRAM_LIMIT_BYTES/2**30} GiB; MemoryMax {MEMORY_MAX}, nice {NICE}; {GAP_SECONDS} s between runs (--gap-seconds)')
    out(f'gate: own units {gate.own_units}, load < {gate.load_limit}, GPU free >= {gate.gpu_free_mib} MiB, MemAvailable >= {gate.mem_available_gib} GiB, /home >= {run.HOME_FLOOR_GB} GB, /mnt/c >= {gate.host_floor_gib} GiB')
    out('  # | experiment | arm | K | tier | native steps | sigma_field | eps | channels | pair | unit | run_id')
    for i, c in enumerate(plan, 1):
        if c.is_repair:
            spec = ef.REPAIR_ARMS[c.arm]
            out(f"{i:3d} | E3 | {c.arm} | - | {c.tier} | - | - | {spec['eps']} | {ef.SOURCES[(c.tier, c.pair)]}/runs/Q2_{c.tier} | {c.short} | {c.unit} | {c.run_id}")
        else:
            spec = ef.ARMS[c.arm]
            out(f"{i:3d} | E1 | {c.arm} | {spec['k']} | {c.tier} | {spec['native']} | {spec['sigma_field']} | - | {spec['channels']} | {c.short} | {c.unit} | {c.run_id}")
    problems = safe_input_problems(out, stage=stage, seg_root=seg_root, a5p=a5p, allowlist=allowlist, unet=unet, results=results)
    if with_colouring:
        for pair in core.PAIRS:
            cell = ef.Cell('K8', 'long', pair)
            pin = ef.read_images(cell)
            predicted = fast.read_predicted(pin, unet=unet, seg_root=seg_root, allowlist=allowlist, workers=1)
            colours, record = vc.choose_colouring(predicted.labels['fixed'], predicted.labels['moving'], K=ef.ARMS['K8']['k'])
            out(f"K8 {cell.short}: {record['mode']}, {record['hard_edges']} face edges, same colour within {record['soft_reach']} voxels {record['same_colour_within_soft_reach']}, members {record['members']}; smallest code distance between labels with a common face {vc.check_separated(predicted.labels['fixed'], colours):.3f} (fixed), {vc.check_separated(predicted.labels['moving'], colours):.3f} (moving)")
    out(f'/home free {shutil.disk_usage("/home").free/2**30:.0f} GB; /mnt/c free {shutil.disk_usage("/mnt/c").free/2**30:.0f} GiB')
    code = Path(root)/'code'
    out('--- the smoke test on the GPU, by the lead, before the queue (no file is read or written; exit 8 if a check fails)')
    out(shlex.join(smoke_launch()))
    out('--- start strings (the queue: the lead; the others: the queue, one per run; the code at $ROOT/code)')
    out(shlex.join(queue_launch(root)))
    for c in plan:
        out(shlex.join(run_argv(c, code, root, '<sha>')))
    out(shlex.join(eval_argv(code, root)))
    out(shlex.join(eval_argv(code, root, EVAL_E3_UNIT)))
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['--dry-run']:
        if argv[1:] not in ([], ['--with-colouring']):
            print('usage: --dry-run [--with-colouring]', file=sys.stderr)
            return 2
        return dry_run(with_colouring=argv[1:] == ['--with-colouring'])
    p = argparse.ArgumentParser(description='EF: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(ef.DEFAULT_ROOT))
    q.add_argument('--commit', default=None)
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--gap-seconds', type=int, default=GAP_SECONDS)
    q.add_argument('--min-host-gib', type=float, default=DEFAULT_MIN_HOST_GIB)
    for name in ('run-one', 'repair-one'):
        r = sub.add_parser(name)
        r.add_argument('--arm', required=True)
        r.add_argument('--tier', required=True)
        r.add_argument('--pair', required=True)
        r.add_argument('--out', required=True)
        r.add_argument('--device', default='cuda')
        r.add_argument('--code-sha', default='n/a')
    for name in ('evaluate', 'decide'):
        sub.add_parser(name).add_argument('--root', default=str(ef.DEFAULT_ROOT))
    sub.add_parser('smoke-repair').add_argument('--device', default='cuda')
    o = p.parse_args(argv)
    if o.command == 'smoke-repair':
        return smoke_repair(o)
    if o.command == 'queue':
        return queue(o)
    if o.command == 'run-one':
        return run_one(o)
    if o.command == 'repair-one':
        return repair_one(o)
    problem = ef.root_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if o.command == 'evaluate':
        return evaluate_all(o)
    decision.write_analysis(o.root, stops=run.ledger(o.root, ef.plan_runs())['arms_stopped'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
