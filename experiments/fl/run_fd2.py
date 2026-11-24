"""FD2 (runs/20261105-fd2/任务书.md, plan.md): the driver of FD's extension, the D factor above 6.  `oracle: false`; not a method: no F_B, no certified descent, only the final field passes the certificate; development pairs only, in no result table.  Root fd2_v1; fd_v1 and fl_v1 are only read.
Arms and guards: fd2_core.py.  Reading and the code's choice: decide_fd2.py.  One run, one repair, one evaluation, the phases: run_fl.py's own functions, not copied: another instance of that module under another name, as run_fd.py does.  run_fl, run_fd, fl_core, fd_core, decide_fl, decide_fd are not touched.

    python -B experiments/fl/run_fd2.py --dry-run [--pairs rest14 --arm Hx8 --steps 600]
        prints the plan, the estimates of time and disk, checks the inputs: run_fl.input_problems (selection.json, the npy headers of fixed, moving, the predictions), fd_v1's records of Hx6 at 600 steps and of its repair.  No optimization, no truth label, torch not imported, nothing written.
    python -B experiments/fl/run_fd2.py queue [--commit SHA] [--wait-minutes 30] [--gap-seconds 5] [--min-host-gib 15] [--through fields|all]
        the lead's start, in the unit the dry run prints.  Code: a detached worktree $ROOT/code of --commit (default HEAD; tracked files clean).  First the checks of the dry run: anything missing stops before batch.json and before any run.
        Then the control: Hx6 on 0438-0439, one unit tag-fd2-hx6-first5-0438-0439; its u at 600 steps must be fd_v1's Hx6 at 600 steps bit for bit, or STOP.  Then the 20 runs and the 5 of Hx6L (1200 steps, only described), one unit at a time (MemoryMax 8G, nice 5), each behind FD's gate: no $ROOT/hold, no other tag-* or seq-vp-* unit, 1-minute load < 1.5, GPU free >= 15 GiB, MemAvailable >= 7 GiB, /home >= 100 GB, /mnt/c >= --min-host-gib; closed for --wait-minutes: STOP.
        Then one CPU unit tag-fd2-eval (numba 8), the code's choice, then the arm by arm repairs in the order of the D factor (eps .05, as RF05), each arm followed by its evaluation unit tag-fd2-eval-<arm>: an arm that does not reach 5/5 certificates by that evaluation stops the repairs, the arms after it are named.  The reading.  A run directory that exists is never rerun, failures are kept.  An arm stops on the 9 GiB device limit, a device out-of-memory error or a run over 15 minutes.  3 failures in a row, a record whose oracle mark is wrong: STOP.
        An evaluation unit that failed: exit 8 and nothing is chosen or repaired after it; a restart evaluates what is left and goes on.  A cell that failed in a way that says nothing about the D factor (decide_fd2.py): the reading is written, exit 8.  A choice that is not the file the code wrote, or a driver whose decide_fd2.py, fd2_core.py, decide_fd.py, fd_core.py, decide_fl.py or fl_core.py is not the batch's commit: STOP.
    python -B experiments/fl/run_fd2.py queue --pairs rest14 --arm Hx8 --steps 600 --commit SHA
        the lead's separate start for the other 14 pairs, after he read the first 5, not when they read 不可判: a field per pair, the evaluation, a repair per pair, the evaluation, the reading again.  Not part of the default queue.
    run_fd2.py run-one --arm Hx8 --tier first5 --pair PAIR --out DIR [--device cuda] [--code-sha SHA]     one field run: field_<step>.npz per checkpoint, run.json
    run_fd2.py repair-one --arm RHx8 --tier c600 --pair PAIR --out DIR [--device cuda] [--code-sha SHA]    one repair: repair.npz, repair.json.  No image reaches the solve and no label is read anywhere.
    run_fd2.py evaluate                  every checkpoint and repair without metrics; the only place the truth labels are read
    run_fd2.py decide                    analysis/decision_fd2.json, long.csv, summary_fd2.md

The only root is /home/rmcfz/seq_vp_tag_results/fd2_v1: every entry that writes refuses any other.  run-one, repair-one and evaluate need PYTHONPATH=<frozen>:<code> (the unit sets it).  Exit codes: 0 done, 2 usage, another driver on the root, another root, 8 STOP or a failed check.
"""
import argparse
from dataclasses import replace
import importlib.util
from pathlib import Path
import shlex
import shutil
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent/'ef'))
sys.path.insert(2, str(HERE.parent/'mc0'))

import decide_fd2 as decision  # noqa: E402
import fd2_core as fd2  # noqa: E402
import fd_core as fd  # noqa: E402
import fl_core as fl  # noqa: E402
import mc0_core as core  # noqa: E402

SCRIPT = 'experiments/fl/run_fd2.py'
QUEUE_UNIT, EVAL_UNIT, EVAL_REST_UNIT, EVAL_REST_R_UNIT = 'tag-fd2-queue', 'tag-fd2-eval', 'tag-fd2-eval-rest', 'tag-fd2-eval-rest-r'


def _batch_instance():
    """run_fl once more under another name, its module names pointed at this batch.  sys.modules['run_fl'] and run_fd's instance are separate module objects."""
    spec = importlib.util.spec_from_file_location('_fd2_run_fl', HERE/'run_fl.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.fl, module.decision, module.SCRIPT, module.QUEUE_UNIT = fd2, decision, SCRIPT, QUEUE_UNIT
    module.EVAL_UNIT, module.EVAL_REST_UNIT, module.EVAL_REST_R_UNIT = EVAL_UNIT, EVAL_REST_UNIT, EVAL_REST_R_UNIT
    module.EVAL_R1_UNIT, module.EVAL_X_UNIT, module.EVAL_R2_UNIT = 'tag-fd2-eval-r1', 'tag-fd2-eval-x', 'tag-fd2-eval-r2'     # no phase here uses them; no name of FL's stays in the instance
    module.eval_argv.__defaults__ = (EVAL_UNIT,)        # bound when run_fl was executed: the only name there that is
    module.GATE = replace(module.GATE, own_units=(f'{QUEUE_UNIT}.service',))
    return module


base = _batch_instance()
run = base.run
GATE = base.GATE
READING_FILES = ((decision, 'experiments/fl/decide_fd2.py'), (fd2, 'experiments/fl/fd2_core.py'), (decision.fd_reading, 'experiments/fl/decide_fd.py'), (fd, 'experiments/fl/fd_core.py'), (decision.fl_reading, 'experiments/fl/decide_fl.py'), (fl, 'experiments/fl/fl_core.py'))


def eval_unit_of(arm):
    return f'tag-fd2-eval-{arm.lower()}'


# ---- the checks that are this batch's ---------------------------------------------------------------------------------------------------------
def control_problem(root, results=None):
    """Why the other runs may not start, or None: the control has to have finished and its u at 600 steps has to be fd_v1's Hx6 at 600 steps, bit for bit."""
    cell = fd2.plan_control()[0]
    record = run.read_json(fd2.run_dir(root, cell)/'run.json')
    if record is None or record.get('status') != 'done':
        return f"{cell.run_id}: the control did not finish: {None if record is None else record.get('status')}"
    same, mine, theirs = decision.control_of_hx6(root, results)
    return None if same is True else f'{cell.run_id}: u at {fd2.NATIVE} steps {mine} is not fd_v1 Hx6 at {fd2.NATIVE} steps {theirs}'


def reading_code_problem(root):
    """Why this process may not write this root's reading, or None: the files that hold the rules must be the batch's commit, byte for byte (FL's rule, on this batch's files).  No batch.json: nothing has run."""
    batch = run.read_json(Path(root)/'batch.json')
    if batch is None:
        return None
    for module, rel in READING_FILES:
        theirs = base.committed_bytes(batch['code_sha'], rel)
        if theirs is None or theirs != Path(module.__file__).read_bytes():
            return f"{module.__file__} is not {rel} of the batch's commit {batch['code_sha']}: the reading's rules would not be the batch's; run the driver from $ROOT/code"
    return None


def input_problems(out=lambda text: None, *, pairs='first5', stage=None, seg_root=None, allowlist=None, unet=None, results=None):
    """run_fl.input_problems on this batch, then fd_v1's records.  fd_v1 is read up to its JSON records; nothing written."""
    problems = base.input_problems(out, pairs=pairs, stage=stage, seg_root=seg_root, allowlist=allowlist, unet=unet, results=results)
    root = fd2.fd_root(results)
    for pair in core.PAIRS:
        d = fd.run_dir(root, fd.Cell(fd2.CONTROL_ARM, fd.FIRST, pair))
        repair_d = fd.run_dir(root, fd.Cell('R'+fd2.CONTROL_ARM, f'c{fd.NATIVE}', pair))
        record, metrics, repaired = run.read_json(d/'run.json') or {}, run.read_json(d/f'metrics_{fd2.NATIVE:04d}.json') or {}, run.read_json(repair_d/'repair_metrics.json') or {}
        checkpoint = (record.get('checkpoints') or {}).get(str(fd2.NATIVE)) or {}
        accepted = ((repaired.get('after') or {}).get('certificate') or {}).get('accepted_alpha')
        out(f"fd_v1 Hx6@{fd2.NATIVE} {core.short_name(pair)}: {d}: status {record.get('status')}, oracle {record.get('oracle')}, D_factor {(record.get('scale') or {}).get('D_factor')}, u_sha256 {checkpoint.get('u_sha256')}, metrics there: {bool(metrics)}; RHx6 accepted alpha {accepted}")
        if record.get('status') != 'done' or record.get('oracle') is not False or checkpoint.get('u_sha256') is None or not metrics.get('metrics') or accepted != 1.:
            problems.append(f'fd_v1 Hx6@{fd2.NATIVE} {core.short_name(pair)} is not usable')
    return problems


def safe_input_problems(out=lambda text: None, **kwargs):
    """input_problems, with a missing file or a refused input as a problem and not a traceback: the queue says what it was, in queue_stop.json."""
    try:
        return input_problems(out, **kwargs)
    except (Exception, SystemExit) as error:            # noqa: BLE001 - any of them means the inputs are not there
        return [f'{type(error).__name__}: {error}'[:1000]]


# ---- the queue ---------------------------------------------------------------------------------------------------------------------------------------
def rest_body(o, system, root, log, code, sha, gate):
    """The lead's separate queue of the other 14 pairs, after the first reading."""
    reading = run.read_json(root/'analysis/decision_fd2.json')
    if reading is None or fd2.read_selection(root, 'repair_selection_1.json') is None:
        raise run.Stop('no reading of the first 5 pairs yet: the other 14 pairs start after it')
    if reading['reading'] == decision.READ_UNDECIDABLE:
        raise run.Stop(f"the first 5 pairs read {decision.READ_UNDECIDABLE}: back to the lead, the other 14 pairs do not start ({reading['reason']})"[:1000])
    fields, repairs = fd2.plan_rest(o.arm, o.steps)
    best = reading.get('best')
    if best is None or best['arm'] != o.arm:
        raise run.Stop(f"the other 14 pairs are only for the best arm of the reading ({None if best is None else best['arm']}), not {o.arm}: back to the lead")
    batch = run.read_json(root/'batch.json') or {}
    if batch.get('code_sha') != sha:
        raise run.Stop(f"the other 14 pairs run at the first batch's commit {batch.get('code_sha')}, not at {sha}")
    problem = reading_code_problem(root)
    if problem:
        raise run.Stop(problem)
    problems = safe_input_problems(pairs='rest14')
    if problems:
        raise run.Stop(f'the inputs, before any run: {problems}')
    base.write_batch(root, f'batch_rest14_{o.arm}_{o.steps}.json', dict(plan=[c.run_id for c in fields+repairs], arm=o.arm, steps=o.steps, gap_seconds=o.gap_seconds), sha)
    log(f'rest14 start: {o.arm} at {o.steps} steps, {len(fields)} fields, then {len(repairs)} repairs')
    rcs, stop = {}, None
    try:
        base.phase_and_evaluation(o, system, root, log, code, sha, fields, gate, EVAL_REST_UNIT, rcs)
        base.phase_and_evaluation(o, system, root, log, code, sha, repairs, gate, EVAL_REST_R_UNIT, rcs)
    except run.Stop as error:
        stop = str(error)
    state = base.ledger(root)
    result = decision.write_analysis(root, stops=state['arms_stopped'])
    run.write_json(root/f'queue_end_rest14_{o.arm}_{o.steps}.json', dict(stop=stop, eval_rc=rcs, rest14=[dict(arm=r['arm'], step=r['step'], seg35=r['repairs']['seg35'], seg4=r['repairs']['seg4'], n_certified_at_1=r['repairs']['n_certified_at_1']) for r in result['rest14']], ledger=state, utc=run.utc()))
    log('rest14 ended.  Handed to the lead: no next step starts here')
    if stop is not None or any(rc != 0 for rc in rcs.values()):
        raise run.Stop(f'the rest14 queue: stop {stop}, evaluation units {rcs}')
    return 0


def repair_in_order(o, system, root, log, code, sha, gate, chosen, rcs):
    """Arm by arm, by D factor: its 5 repairs, its evaluation, and on only if that evaluation gives it 5/5 certificates.  An evaluation that failed stops here: nothing is decided on a record that is not there."""
    for i, c in enumerate(chosen):
        unit = eval_unit_of(c['arm'])
        base.phase_and_evaluation(o, system, root, log, code, sha, fd2.repair_cells([c]), gate, unit, rcs)
        base.evaluated(rcs, unit, f"what comes after {c['arm']} is")
        if not decision.arm_reached_5_of_5(root, c['arm']):
            left = [other['arm'] for other in chosen[i+1:]]
            log(f"{c['arm']} did not reach 5/5 certificates: no higher factor is repaired; left without a repair: {left or 'none'}")
            return f"the repairs stopped at {c['arm']}; left: {left}"
    return 'all the chosen arms repaired'


def queue_body(o, system, root, log):
    """The control, the 20 runs and Hx6L, their evaluation, the choice, the repairs in order, the reading.  A STOP before the first evaluation ends the queue with no reading.  After it nothing can take the fields' records away."""
    base.keep_old_stops(root)
    code, sha = system.code_checkout(root, o.commit)
    gate = replace(GATE, host_floor_gib=o.min_host_gib)
    if o.pairs == 'rest14':
        return rest_body(o, system, root, log, code, sha, gate)
    problems = safe_input_problems()
    if problems:
        raise run.Stop(f'the inputs, before any run: {problems}')
    fields = fd2.plan_fields()
    base.write_batch(root, 'batch.json', dict(plan=[c.run_id for c in fields], later='repairs of the arms through the gate, by D factor, up to the first that does not reach 5/5: analysis/repair_selection_1.json', gap_seconds=o.gap_seconds, through=o.through, fd_root=str(fd2.fd_root()),
                                              caps=dict(run_seconds=fd2.RUN_CAP_SECONDS, run_units_seconds=base.RUN_UNITS_CAP_SECONDS, device_gib=base.mc.VRAM_LIMIT_BYTES/2**30, memory_max=base.MEMORY_MAX, nice=base.NICE, task_budget_seconds=fd2.TASK_BUDGET_SECONDS),
                                              gate=dict(own_units=list(gate.own_units), load_limit=gate.load_limit, gpu_free_mib=gate.gpu_free_mib, mem_available_gib=gate.mem_available_gib, home_floor_gb=run.HOME_FLOOR_GB, host_floor_gib=gate.host_floor_gib),
                                              evaluation=dict(numba_threads=base.EVAL_THREADS, cpu_quota_percent=base.EVAL_QUOTA)), sha)
    problem = reading_code_problem(root)
    if problem:
        raise run.Stop(problem)
    log(f'start: the control, then {len(fields)-1} field runs; the inputs checked; /mnt/c floor {o.min_host_gib:g} GiB; {o.gap_seconds} s between runs; through {o.through}')
    base.run_phase(o, system, root, log, code, sha, fd2.plan_control(), gate)
    problem = control_problem(root)
    if problem:
        raise run.Stop(problem)
    log(f'the control: u of {fd2.plan_control()[0].run_id} is fd_v1 Hx6 at {fd2.NATIVE} steps, bit for bit')
    end = base.run_phase(o, system, root, log, code, sha, fd2.plan_new()+fd2.plan_described(), gate)
    run.wait_for_gate(system, root, False, o.wait_minutes, log, gate)
    log(f'launch {EVAL_UNIT}')
    rcs = {EVAL_UNIT: system.run(base.eval_argv(code, root, EVAL_UNIT), root/f'{EVAL_UNIT}.systemd_run.log')}
    state = base.ledger(root)
    result = decision.write_analysis(root, stops=state['arms_stopped'])          # also after a failed evaluation: a checkpoint without metrics counts as not read
    run.write_json(root/'queue_first_end.json', dict(end=end, eval_rc=rcs[EVAL_UNIT], reading=result['reading'], reason=result['reason'], role='interim: the code has not chosen yet', ledger=state, utc=run.utc()))
    log(f"the fields are evaluated (interim record: {result['reason']})")
    stop_later, end_later = None, None
    if o.through != 'fields':
        try:
            base.evaluated(rcs, EVAL_UNIT, 'the repairs are')
            selection = decision.ensure_repair_selection(root)
            base.log_choice(log, root, 'repair_selection_1.json', f"the arms through the gate, by D factor: {selection['chosen'] or 'none'}")
            end_later = repair_in_order(o, system, root, log, code, sha, gate, selection['chosen'], rcs)
        except run.Stop as stop:
            stop_later, end_later = str(stop), f'STOP in a later phase: {stop}'
            log(end_later)
            run.write_json(root/'queue_stop_later.json', dict(reason=str(stop), utc=run.utc()))
    state = base.ledger(root)
    result = decision.write_analysis(root, stops=state['arms_stopped'])
    run.write_json(root/'queue_end.json', dict(end=end, end_later=end_later, stop_later=stop_later, eval_rc=rcs, reading=result['reading'], reason=result['reason'], not_read=result['not_read'], left_without_repair=result['left_without_repair'], first_failed_repair=result['first_failed_repair'], best=None if result['best'] is None else {k: v for k, v in result['best'].items() if k != 'delta'}, selections=fd2.selection_hashes(root), ledger=state, utc=run.utc()))
    log(f"reading: {result['reading']}.  Handed to the lead: no next step starts here")
    failed = {unit: rc for unit, rc in rcs.items() if rc != 0}
    if failed:
        raise run.Stop(f'an evaluation unit failed: {failed} (metrics errors in the run directories, or the {run.EVAL_RUNTIME_MAX} s limit of the unit); the reading is written, a checkpoint without metrics counts as not read')
    if stop_later is not None:
        raise run.Stop(f'a later phase stopped: {stop_later}')
    if result['not_read']:
        raise run.Stop(f"the reading {result['reading']} is written, and these cells failed: {result['not_read']}"[:1000])
    return 0


def guarded_body(o, system, root, log):
    """A choice that is not the file the code wrote is a STOP with its record, wherever it is met."""
    try:
        return queue_body(o, system, root, log)
    except fd2.SelectionChanged as changed:
        raise run.Stop(str(changed))


def queue(o, system=None):
    problem = fd2.root_problem(o.root)
    if problem:                                         # before the lock file: nothing is created or touched
        print(problem, file=sys.stderr)
        return 2
    return run.queue(o, system, body=guarded_body, prefix='fd2')


# ---- estimates --------------------------------------------------------------------------------------------------------------------------------------
GATE_WAIT_SECONDS = 90.                         # 〔估算〕about what fd_v1's queue.log shows after an evaluation unit, with other CPU work on the machine then


def estimates(results=None):
    """Seconds and bytes of the plan from fd_v1's own records of Hx6 and RHx6.  Estimates, 〔估算〕; None where there is no record.  Higher D factors mean more infeasible cells: nobody has run them; the upper end of the repairs is the wall of 120 s on each."""
    root = fd2.fd_root(results)
    tasks, ios, evals, sizes, repairs, starts = [], [], [], [], [], []
    for pair in core.PAIRS:
        d = fd.run_dir(root, fd.Cell(fd2.CONTROL_ARM, fd.FIRST, pair))
        record = run.read_json(d/'run.json') or {}
        checkpoint = (record.get('checkpoints') or {}).get(str(fd2.NATIVE)) or {}
        metrics = run.read_json(d/f'metrics_{fd2.NATIVE:04d}.json') or {}
        levels = (record.get('optimizer') or {}).get('levels') or []
        if checkpoint.get('seconds_task') is not None:
            tasks.append(checkpoint['seconds_task'])
            ios.append(checkpoint.get('seconds_io') or 0.)
            if len(levels) == 4:
                starts.append(checkpoint['seconds_task']-levels[3]['seconds'])
        if metrics.get('seconds_certificate') is not None and metrics.get('seconds_metrics') is not None:
            evals.append(metrics['seconds_certificate']+metrics['seconds_metrics'])
        if (d/f'field_{fd2.NATIVE:04d}.npz').is_file():
            sizes.append((d/f'field_{fd2.NATIVE:04d}.npz').stat().st_size)
        repaired = run.read_json(fd.run_dir(root, fd.Cell('R'+fd2.CONTROL_ARM, f'c{fd.NATIVE}', pair))/'repair.json') or {}
        if repaired.get('seconds_total') is not None and repaired.get('seconds_repair') is not None:
            repairs.append(repaired['seconds_total'])

    def average(values):
        return statistics.fmean(values) if values else None
    task, io, evaluation, size, repair, start = average(tasks), average(ios), average(evals), average(sizes), average(repairs), average(starts)
    n_short, n_long, n_repairs = 1+len(fd2.plan_new()), len(fd2.plan_described()), len(fd2.repair_cells([dict(arm=arm, step=fd2.NATIVE) for arm in fd2.NEW_ARMS]))
    known = None not in (task, io, evaluation, repair, start)
    long_task = None if not known else start+(task-start)*fd2.LONG/fd2.NATIVE
    gap = base.GAP_SECONDS
    n_files = n_short+2*n_long
    wall = fd2.REPAIR_WALL_SECONDS
    n_waits = len(fd2.NEW_ARMS)-1                   # after an arm's evaluation unit (numba 8 threads) the next arm waits for the 1-minute load < 1.5
    return dict(task_seconds=task, io_seconds=io, long_task_seconds=long_task, evaluation_seconds=evaluation, repair_process_seconds=repair, field_bytes=size, n_short=n_short, n_long=n_long, n_field_files=n_files, n_repairs_at_most=n_repairs, gap_seconds=gap,
                seconds_fields=None if not known else n_short*(task+io+gap)+n_long*(long_task+2*io+gap), seconds_fields_evaluation=None if not known else n_files*evaluation, seconds_repairs_low=None if not known else n_repairs*(repair+gap), seconds_repairs_high=None if not known else n_repairs*(wall+15.+gap), seconds_repairs_evaluation=None if not known else n_repairs*evaluation, n_gate_waits=n_waits, seconds_gate_waits=n_waits*GATE_WAIT_SECONDS,
                field_files_gib=None if size is None else n_files*size/2**30, rest14_field_files_gib=None if size is None else len(fd2.REST_PAIRS)*size/2**30, rest14_seconds=None if not known else len(fd2.REST_PAIRS)*(task+io+evaluation+repair+evaluation+2*gap))


def _minutes(seconds):
    return '—' if seconds is None else f'{seconds/60:.1f} min'


def _number(value, nd=1):
    return '—' if value is None else format(value, f'.{nd}f')


def print_cell(out, i, c, phase):
    if c.is_repair:
        out(f"{i:3d} | {phase} | {c.arm} | repair of {c.base} at {fd2.steps_of(c)} steps | eps {fd2.REPAIR_EPS} | {c.short} | {c.unit} | {c.run_id}")
    else:
        spec = fd2.ARMS[c.arm]
        out(f"{i:3d} | {phase} | {c.arm} | K {spec['k']} | sigma {spec['sigma_field']} | native {fd2.steps_of(c)} | checkpoints {'/'.join(map(str, fd2.checkpoints_of(c)))} | {spec['channels']}, weight {spec['weight']}, D factor {spec['D_factor']} | {c.short} | {c.unit} | {c.run_id}")


def dry_run(root=None, stage=None, out=print, unet=None, seg_root=None, allowlist=None, results=None, pairs='first5', arm=None, steps=None):
    """The plan, the estimates, the checks of the queue, the start strings.  No truth label, no optimization, nothing written."""
    root = fd2.DEFAULT_ROOT if root is None else root
    gate = GATE
    out(f"FD2 dry run ({run.utc()}): no optimization, no truth label, nothing written; real files read up to the npy headers of fixed, moving and p, and the JSON records of fd_v1.")
    out(f'root {root}; root exists: {Path(root).exists()}; fd_v1 read only at {fd2.fd_root(results)}; cap per run {fd2.RUN_CAP_SECONDS} s, run units {base.RUN_UNITS_CAP_SECONDS} s, device {base.mc.VRAM_LIMIT_BYTES/2**30} GiB; MemoryMax {base.MEMORY_MAX}, nice {base.NICE}; {base.GAP_SECONDS} s between runs (--gap-seconds); task budget {fd2.TASK_BUDGET_SECONDS} s (a premise of the reading, not a stop), BZ line {fd2.BZ_SECONDS} s (a column)')
    out(f'gate: own units {gate.own_units}, load < {gate.load_limit}, GPU free >= {gate.gpu_free_mib} MiB, MemAvailable >= {gate.mem_available_gib} GiB, /home >= {run.HOME_FLOOR_GB} GB, /mnt/c >= {gate.host_floor_gib} GiB')
    if pairs == 'first5':
        plan = [(c, 'control') for c in fd2.plan_control()]+[(c, 'fields') for c in fd2.plan_new()]+[(c, 'only described') for c in fd2.plan_described()]
        out(f'  # | phase | arm | ...  (then the repairs of the arms through the gate, by D factor, 5 pairs each, cells R<arm> c{fd2.NATIVE}, each followed by {eval_unit_of("<arm>")}, up to the first that does not reach 5/5)')
    else:
        fields, repairs = fd2.plan_rest(arm, steps)
        plan = [(c, 'rest14 fields') for c in fields]+[(c, 'rest14 repairs') for c in repairs]
    for i, (c, phase) in enumerate(plan, 1):
        print_cell(out, i, c, phase)
    e = estimates(results)
    out(f"estimates〔估算, from fd_v1's records of Hx6 and RHx6; nobody has run these arms〕: a field run {_number(e['task_seconds'])} s + {_number(e['io_seconds'])} s writing, Hx6L {_number(e['long_task_seconds'])} s; an evaluated field {_number(e['evaluation_seconds'])} s; a repair process {_number(e['repair_process_seconds'])} s at factor 6, its wall is 120 s")
    out(f"estimates: {e['n_short']} + {e['n_long']} field runs {_minutes(e['seconds_fields'])} with the gaps, their evaluation {_minutes(e['seconds_fields_evaluation'])}; at most {e['n_repairs_at_most']} repairs {_minutes(e['seconds_repairs_low'])} at factor 6's seconds to {_minutes(e['seconds_repairs_high'])} at the wall, their evaluation {_minutes(e['seconds_repairs_evaluation'])}, up to {e['n_gate_waits']} waits at the load gate after an arm's evaluation {_minutes(e['seconds_gate_waits'])}; one arm of the other 14 pairs {_minutes(e['rest14_seconds'])}")
    out(f"estimates: {e['n_field_files']} field files of {_number(None if e['field_bytes'] is None else e['field_bytes']/2**20, 0)} MiB = {_number(e['field_files_gib'])} GiB on /home; one arm of the other 14 pairs {_number(e['rest14_field_files_gib'])} GiB; repair.npz and records small; the JIT cache some MiB")
    problems = safe_input_problems(out, pairs=pairs, stage=stage, seg_root=seg_root, allowlist=allowlist, unet=unet, results=results)
    out(f'/home free {shutil.disk_usage("/home").free/2**30:.0f} GB; /mnt/c free {shutil.disk_usage("/mnt/c").free/2**30:.0f} GiB')
    code = Path(root)/'code'
    out('--- start strings (the queue: the lead; the others: the queue, one per run; the code at $ROOT/code)')
    if pairs == 'first5':
        out(shlex.join(queue_launch(root)))
        for c, _ in plan:
            out(shlex.join(base.run_argv(c, code, root, '<sha>')))
        out(shlex.join(base.eval_argv(code, root, EVAL_UNIT)))
        for name in fd2.NEW_ARMS:
            out(shlex.join(base.eval_argv(code, root, eval_unit_of(name))))
        out('--- the other 14 pairs: only the lead, after reading the first 5, at the same commit (example: arm Hx8)')
        out(shlex.join(rest_launch('Hx8', fd2.NATIVE, '<sha of batch.json>', root)))
    else:
        out(shlex.join(rest_launch(arm, steps, '<sha of batch.json>', root)))
        for c, _ in plan:
            out(shlex.join(base.run_argv(c, code, root, '<sha>')))
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


def queue_launch(root=None, repo=core.MAIN_REPO, gap=None, min_host_gib=None, extra=()):
    """The lead's start string of a queue: run_fl.queue_launch on this batch's unit and script."""
    return base.queue_launch(root, repo, base.GAP_SECONDS if gap is None else gap, base.DEFAULT_MIN_HOST_GIB if min_host_gib is None else min_host_gib, extra)


def rest_launch(arm, steps, commit, root=None):
    return queue_launch(root, extra=('--pairs', 'rest14', '--arm', arm, '--steps', str(steps), '--commit', commit))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['--dry-run']:
        d = argparse.ArgumentParser(description='FD2: the dry run')
        d.add_argument('--dry-run', action='store_true')
        d.add_argument('--pairs', choices=('first5', 'rest14'), default='first5')
        d.add_argument('--arm', default=None)
        d.add_argument('--steps', type=int, default=None)
        o = d.parse_args(argv)
        if o.pairs == 'rest14' and (o.arm is None or o.steps is None):
            print('usage: --dry-run --pairs rest14 --arm A --steps N', file=sys.stderr)
            return 2
        return dry_run(pairs=o.pairs, arm=o.arm, steps=o.steps)
    p = argparse.ArgumentParser(description='FD2: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(fd2.DEFAULT_ROOT))
    q.add_argument('--commit', default=None)
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--gap-seconds', type=int, default=base.GAP_SECONDS)
    q.add_argument('--min-host-gib', type=float, default=base.DEFAULT_MIN_HOST_GIB)
    q.add_argument('--through', choices=('fields', 'all'), default='all')
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
        sub.add_parser(name).add_argument('--root', default=str(fd2.DEFAULT_ROOT))
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
        return base.run_one(o)
    if o.command == 'repair-one':
        return base.repair_one(o)
    problem = fd2.root_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if o.command == 'evaluate':
        return base.evaluate_all(o)
    problem = reading_code_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 8
    decision.write_analysis(o.root, stops=base.ledger(o.root)['arms_stopped'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
