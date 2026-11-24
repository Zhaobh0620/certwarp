"""FD (runs/20261105-fd/任务书.md, plan.md): the driver of FL's working point probe, the D factor above 2.  `oracle: false`; not a method: no F_B, no certified descent, only the final field passes the certificate; development pairs only, in no result table.  Root fd_v1; fl_v1 is only read.
Arms and guards: fd_core.py.  Reading and the code's choice: decide_fd.py.  One run, one repair, one evaluation, the phases: run_fl.py's own functions, not copied: a second instance of that module under another name, whose `fl`, `decision`, script and unit names are this batch's.  The module run_fl, FL's entry, and fl_core, decide_fl are not touched: FL's rules and its reading of fl_v1 stay what they were.

    python -B experiments/fl/run_fd.py --dry-run [--pairs rest14 --arm Hx4 --steps 600]
        prints the plan, the estimates of time and disk, checks the inputs: run_fl.input_problems (selection.json, the npy headers of fixed, moving, the predictions), fl_v1's records of Hx, H, F at 600 steps.  No optimization, no truth label, torch not imported, nothing written.
    python -B experiments/fl/run_fd.py queue [--commit SHA] [--wait-minutes 30] [--gap-seconds 5] [--min-host-gib 15] [--through fields|all]
        the lead's start, in the unit the dry run prints.  Code: a detached worktree $ROOT/code of --commit (default HEAD; tracked files clean).  First the checks of the dry run: anything missing stops before batch.json and before any run.
        Then the control: Hx on 0438-0439, one unit tag-fd-hx-first5-0438-0439; its u at 600 steps must be fl_v1's Hx at 600 steps bit for bit, or STOP.  Then the 25 runs, one unit at a time (MemoryMax 8G, nice 5), each behind FL's gate: no $ROOT/hold, no other tag-* or seq-vp-* unit, 1-minute load < 1.5, GPU free >= 15 GiB, MemAvailable >= 7 GiB, /home >= 100 GB, /mnt/c >= --min-host-gib; closed for --wait-minutes: STOP.
        Then one CPU unit tag-fd-eval (numba 8), the code's choice of at most 2 arms, their repairs (eps .05, as RF05), then RHx600 (fl_v1's own Hx at 600 steps, the same settings, only described, its failure changes no reading), tag-fd-eval-r1, the reading.  A run directory that exists is never rerun, failures are kept.  An arm stops on the 9 GiB device limit, a device out-of-memory error or a run over 15 minutes.  3 failures in a row, a record whose oracle mark is wrong: STOP.
        An evaluation unit that failed: exit 8 and no choice is written after it (a choice is written once); a restart evaluates what is left and chooses then.  A cell that failed in a way that says nothing about the D factor (decide_fd.py): the reading is written, exit 8.  A choice that is not the file the code wrote, or a driver whose decide_fd.py, fd_core.py, decide_fl.py or fl_core.py is not the batch's commit: STOP.
    python -B experiments/fl/run_fd.py queue --pairs rest14 --arm Hx4 --steps 600 --commit SHA
        the lead's separate start for the other 14 pairs, after he read the first 5, not when they read 不可判: a field per pair, the evaluation, a repair per pair, the evaluation, the reading again.  Not part of the default queue.
    run_fd.py run-one --arm Hx3 --tier first5 --pair PAIR --out DIR [--device cuda] [--code-sha SHA]     one field run: field_0600.npz, run.json
    run_fd.py repair-one --arm RHx3 --tier c600 --pair PAIR --out DIR [--device cuda] [--code-sha SHA]    one repair: repair.npz, repair.json.  No image reaches the solve and no label is read anywhere.
    run_fd.py evaluate                   every checkpoint and repair without metrics; the only place the truth labels are read
    run_fd.py decide                     analysis/decision_fd.json, long.csv, summary_fd.md

The only root is /home/rmcfz/seq_vp_tag_results/fd_v1: every entry that writes refuses any other.  run-one, repair-one and evaluate need PYTHONPATH=<frozen>:<code> (the unit sets it).  Exit codes: 0 done, 2 usage, another driver on the root, another root, 8 STOP or a failed check.
"""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent/'ef'))
sys.path.insert(2, str(HERE.parent/'mc0'))

import decide_fd as decision  # noqa: E402
import fd_core as fd  # noqa: E402
import fl_core as fl  # noqa: E402
import mc0_core as core  # noqa: E402

SCRIPT = 'experiments/fl/run_fd.py'
QUEUE_UNIT, EVAL_UNIT, EVAL_R1_UNIT, EVAL_REST_UNIT, EVAL_REST_R_UNIT = 'tag-fd-queue', 'tag-fd-eval', 'tag-fd-eval-r1', 'tag-fd-eval-rest', 'tag-fd-eval-rest-r'
UNREACHED_UNITS = dict(EVAL_X_UNIT='tag-fd-eval-x', EVAL_R2_UNIT='tag-fd-eval-r2')       # no phase here uses them; no name of FL's stays in the instance


def source_of_repair(cell, root=None, results=None):
    """run_fl.source_of_repair, and RHx600: the field is fl_v1's Hx at 600 steps, read only, with its own record and evaluation; the checks are source_checked's, all of them."""
    if cell.arm not in fd.FL_REPAIRS:
        return base.batch_source_of_repair(cell, root, results)
    d = fl.run_dir(fd.fl_root(results), fl.Cell(cell.base, fl.FIRST, cell.pair))
    record = base.run.read_json(d/'run.json')
    checkpoint = ((record or {}).get('checkpoints') or {}).get(str(fd.NATIVE))
    if checkpoint is None:
        raise SystemExit(f'{d}: no checkpoint at {fd.NATIVE} steps')
    return d, checkpoint['file'], checkpoint['sha256'], f'metrics_{fd.NATIVE:04d}.json', record, checkpoint['seconds_task']


def repair_allowed(cell, root):
    """RHx600 is in the plan, whatever the code chose."""
    return cell.arm in fd.FL_REPAIRS or base.batch_repair_allowed(cell, root)


def _batch_instance():
    """run_fl once more under another name, its module names pointed at this batch.  sys.modules['run_fl'] is not touched, and neither is what FL's entry sees: it is a separate module object."""
    spec = importlib.util.spec_from_file_location('_fd_run_fl', HERE/'run_fl.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.fl, module.decision, module.SCRIPT, module.QUEUE_UNIT = fd, decision, SCRIPT, QUEUE_UNIT
    module.EVAL_UNIT, module.EVAL_R1_UNIT, module.EVAL_REST_UNIT, module.EVAL_REST_R_UNIT = EVAL_UNIT, EVAL_R1_UNIT, EVAL_REST_UNIT, EVAL_REST_R_UNIT
    module.EVAL_X_UNIT, module.EVAL_R2_UNIT = UNREACHED_UNITS['EVAL_X_UNIT'], UNREACHED_UNITS['EVAL_R2_UNIT']
    module.eval_argv.__kwdefaults__ = dict(module.eval_argv.__kwdefaults__ or {})
    module.eval_argv.__defaults__ = (EVAL_UNIT,)        # bound when run_fl was executed: the only name there that is
    module.batch_source_of_repair, module.source_of_repair = module.source_of_repair, source_of_repair
    module.batch_repair_allowed, module.repair_allowed = module.repair_allowed, repair_allowed
    module.GATE = replace(module.GATE, own_units=(f'{QUEUE_UNIT}.service',))
    return module


base = _batch_instance()
run = base.run
GATE = base.GATE
READING_FILES = ((decision, 'experiments/fl/decide_fd.py'), (fd, 'experiments/fl/fd_core.py'), (decision.fl_reading, 'experiments/fl/decide_fl.py'), (fl, 'experiments/fl/fl_core.py'))


# ---- the checks that are this batch's ---------------------------------------------------------------------------------------------------------
def control_problem(root, results=None):
    """Why the other runs may not start, or None: the control has to have finished and its u at 600 steps has to be fl_v1's Hx at 600 steps, bit for bit."""
    cell = fd.plan_control()[0]
    record = run.read_json(fd.run_dir(root, cell)/'run.json')
    if record is None or record.get('status') != 'done':
        return f"{cell.run_id}: the control did not finish: {None if record is None else record.get('status')}"
    same, mine, theirs = decision.control_of_hx(root, results)
    return None if same is True else f'{cell.run_id}: u at {fd.NATIVE} steps {mine} is not fl_v1 Hx at {fd.NATIVE} steps {theirs}'


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
    """run_fl.input_problems on this batch, then fl_v1's records.  fl_v1 is read up to its JSON records; nothing written."""
    problems = base.input_problems(out, pairs=pairs, stage=stage, seg_root=seg_root, allowlist=allowlist, unet=unet, results=results)
    root = fd.fl_root(results)
    for arm in decision.REFERENCE_ARMS:
        for pair in core.PAIRS:
            d = fl.run_dir(root, fl.Cell(arm, fl.FIRST, pair))
            record, metrics = run.read_json(d/'run.json') or {}, run.read_json(d/f'metrics_{fd.NATIVE:04d}.json') or {}
            checkpoint = (record.get('checkpoints') or {}).get(str(fd.NATIVE)) or {}
            out(f"fl_v1 {arm}@{fd.NATIVE} {core.short_name(pair)}: {d}: status {record.get('status')}, oracle {record.get('oracle')}, D_factor {(record.get('scale') or {}).get('D_factor')}, u_sha256 {checkpoint.get('u_sha256')}, metrics there: {bool(metrics)}")
            if record.get('status') != 'done' or record.get('oracle') is not False or checkpoint.get('u_sha256') is None or not metrics.get('metrics'):
                problems.append(f'fl_v1 {arm}@{fd.NATIVE} {core.short_name(pair)} is not usable')
            if arm == fd.CONTROL_ARM and pairs == 'first5':          # a line, not a problem: RHx600 is only described, and its failure is only recorded
                field = d/str(checkpoint.get('file'))
                out(f"RHx600 source {core.short_name(pair)}: {field}: equal to its run.json: {field.is_file() and core.sha_file(field) == checkpoint.get('sha256')}")
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
    reading = run.read_json(root/'analysis/decision_fd.json')
    if reading is None or fd.read_selection(root, 'repair_selection_1.json') is None:
        raise run.Stop('no reading of the first 5 pairs yet: the other 14 pairs start after it')
    if reading['reading'] == decision.READ_UNDECIDABLE:
        raise run.Stop(f"the first 5 pairs read {decision.READ_UNDECIDABLE}: back to the lead, the other 14 pairs do not start ({reading['reason']})"[:1000])
    fields, repairs = fd.plan_rest(o.arm, o.steps)
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


def queue_body(o, system, root, log):
    """The control, the 25 runs, their evaluation, the choice, the repairs, their evaluation, the reading.  A STOP before the first evaluation ends the queue with no reading.  After it nothing can take the fields' records away."""
    base.keep_old_stops(root)
    code, sha = system.code_checkout(root, o.commit)
    gate = replace(GATE, host_floor_gib=o.min_host_gib)
    if o.pairs == 'rest14':
        return rest_body(o, system, root, log, code, sha, gate)
    problems = safe_input_problems()
    if problems:
        raise run.Stop(f'the inputs, before any run: {problems}')
    fields = fd.plan_fields()
    base.write_batch(root, 'batch.json', dict(plan=[c.run_id for c in fields], later='repairs of at most 2 arms the code chooses: analysis/repair_selection_1.json; then RHx600, only described', gap_seconds=o.gap_seconds, through=o.through, fl_root=str(fd.fl_root()),
                                              caps=dict(run_seconds=fd.RUN_CAP_SECONDS, run_units_seconds=base.RUN_UNITS_CAP_SECONDS, device_gib=base.mc.VRAM_LIMIT_BYTES/2**30, memory_max=base.MEMORY_MAX, nice=base.NICE, task_budget_seconds=fd.TASK_BUDGET_SECONDS),
                                              gate=dict(own_units=list(gate.own_units), load_limit=gate.load_limit, gpu_free_mib=gate.gpu_free_mib, mem_available_gib=gate.mem_available_gib, home_floor_gb=run.HOME_FLOOR_GB, host_floor_gib=gate.host_floor_gib),
                                              evaluation=dict(numba_threads=base.EVAL_THREADS, cpu_quota_percent=base.EVAL_QUOTA)), sha)
    problem = reading_code_problem(root)
    if problem:
        raise run.Stop(problem)
    log(f'start: the control, then {len(fields)-1} field runs; the inputs checked; /mnt/c floor {o.min_host_gib:g} GiB; {o.gap_seconds} s between runs; through {o.through}')
    base.run_phase(o, system, root, log, code, sha, fd.plan_control(), gate)
    problem = control_problem(root)
    if problem:
        raise run.Stop(problem)
    log(f'the control: u of {fd.plan_control()[0].run_id} is fl_v1 Hx at {fd.NATIVE} steps, bit for bit')
    end = base.run_phase(o, system, root, log, code, sha, fd.plan_new(), gate)
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
            base.log_choice(log, root, 'repair_selection_1.json', f"the repairs the code chose: {selection['chosen'] or 'none'}; then RHx600, only described")
            base.phase_and_evaluation(o, system, root, log, code, sha, fd.repair_cells(selection['chosen'])+fd.plan_fl_repairs(), gate, EVAL_R1_UNIT, rcs)
            end_later = 'all phases handled'
        except run.Stop as stop:
            stop_later, end_later = str(stop), f'STOP in a later phase: {stop}'
            log(end_later)
            run.write_json(root/'queue_stop_later.json', dict(reason=str(stop), utc=run.utc()))
    state = base.ledger(root)
    result = decision.write_analysis(root, stops=state['arms_stopped'])
    run.write_json(root/'queue_end.json', dict(end=end, end_later=end_later, stop_later=stop_later, eval_rc=rcs, reading=result['reading'], reason=result['reason'], not_read=result['not_read'], untried=result['untried'], best=None if result['best'] is None else {k: v for k, v in result['best'].items() if k != 'delta'}, selections=fd.selection_hashes(root), ledger=state, utc=run.utc()))
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
    except fd.SelectionChanged as changed:
        raise run.Stop(str(changed))


def queue(o, system=None):
    problem = fd.root_problem(o.root)
    if problem:                                         # before the lock file: nothing is created or touched
        print(problem, file=sys.stderr)
        return 2
    return run.queue(o, system, body=guarded_body, prefix='fd')


# ---- estimates --------------------------------------------------------------------------------------------------------------------------------------
def estimates(results=None):
    """Seconds and bytes of the plan from fl_v1's own records of Hx (600 steps of the same channels, its repairs started from 2400 steps).  Estimates, 〔估算〕; None where there is no record.  Higher D factors mean more infeasible cells than these repairs saw: nobody has run them."""
    root = fd.fl_root(results)
    tasks, ios, evals, sizes, repairs = [], [], [], [], []
    for pair in core.PAIRS:
        d = fl.run_dir(root, fl.Cell('Hx', fl.FIRST, pair))
        checkpoint = ((run.read_json(d/'run.json') or {}).get('checkpoints') or {}).get(str(fd.NATIVE)) or {}
        metrics = run.read_json(d/f'metrics_{fd.NATIVE:04d}.json') or {}
        if checkpoint.get('seconds_task') is not None:
            tasks.append(checkpoint['seconds_task'])
            ios.append(checkpoint.get('seconds_io') or 0.)
        if metrics.get('seconds_certificate') is not None and metrics.get('seconds_metrics') is not None:
            evals.append(metrics['seconds_certificate']+metrics['seconds_metrics'])
        if (d/f'field_{fd.NATIVE:04d}.npz').is_file():
            sizes.append((d/f'field_{fd.NATIVE:04d}.npz').stat().st_size)
        record = run.read_json(fl.run_dir(root, fl.Cell('RHx', 'c2400', pair))/'repair.json') or {}
        if record.get('seconds_total') is not None:
            repairs.append(record['seconds_total'])

    def average(values):
        return statistics.fmean(values) if values else None
    task, io, evaluation, size, repair = average(tasks), average(ios), average(evals), average(sizes), average(repairs)
    fields, chosen, described = len(fd.plan_fields()), fd.REPAIR_TOP*len(core.PAIRS), len(fd.plan_fl_repairs())
    known = None not in (task, io, evaluation, repair)
    return dict(task_seconds=task, io_seconds=io, evaluation_seconds=evaluation, repair_process_seconds=repair, field_bytes=size, n_fields=fields, n_repairs_at_most=chosen, n_fl_repairs=described, gap_seconds=base.GAP_SECONDS,
                seconds_fields=None if not known else fields*(task+io+base.GAP_SECONDS), seconds_fields_evaluation=None if not known else fields*evaluation, seconds_repairs=None if not known else (chosen+described)*(repair+base.GAP_SECONDS), seconds_repairs_evaluation=None if not known else (chosen+described)*evaluation,
                field_files_gib=None if size is None else fields*size/2**30, rest14_field_files_gib=None if size is None else len(fd.REST_PAIRS)*size/2**30, rest14_seconds=None if not known else len(fd.REST_PAIRS)*(task+io+evaluation+repair+evaluation+2*base.GAP_SECONDS))


def _minutes(seconds):
    return '—' if seconds is None else f'{seconds/60:.1f} min'


def print_cell(out, i, c, phase):
    if c.is_repair:
        out(f"{i:3d} | {phase} | {c.arm} | repair of {'fl_v1 ' if c.arm in fd.FL_REPAIRS else ''}{c.base} at {fd.steps_of(c)} steps | eps {fd.REPAIR_EPS} | {c.short} | {c.unit} | {c.run_id}")
    else:
        spec = fd.ARMS[c.arm]
        out(f"{i:3d} | {phase} | {c.arm} | K {spec['k']} | sigma {spec['sigma_field']} | native {fd.steps_of(c)} | {spec['channels']}{'' if spec['weight'] is None else ', weight '+spec['weight']}, D factor {spec['D_factor']} | {c.short} | {c.unit} | {c.run_id}")


def dry_run(root=None, stage=None, out=print, unet=None, seg_root=None, allowlist=None, results=None, pairs='first5', arm=None, steps=None):
    """The plan, the estimates, the checks of the queue, the start strings.  No truth label, no optimization, nothing written."""
    root = fd.DEFAULT_ROOT if root is None else root
    gate = GATE
    out(f"FD dry run ({run.utc()}): no optimization, no truth label, nothing written; real files read up to the npy headers of fixed, moving and p, and fl_v1's JSON records.")
    out(f'root {root}; root exists: {Path(root).exists()}; fl_v1 read only at {fd.fl_root(results)}; cap per run {fd.RUN_CAP_SECONDS} s, run units {base.RUN_UNITS_CAP_SECONDS} s, device {base.mc.VRAM_LIMIT_BYTES/2**30} GiB; MemoryMax {base.MEMORY_MAX}, nice {base.NICE}; {base.GAP_SECONDS} s between runs (--gap-seconds); task budget {fd.TASK_BUDGET_SECONDS} s (a premise of the reading, not a stop)')
    out(f'gate: own units {gate.own_units}, load < {gate.load_limit}, GPU free >= {gate.gpu_free_mib} MiB, MemAvailable >= {gate.mem_available_gib} GiB, /home >= {run.HOME_FLOOR_GB} GB, /mnt/c >= {gate.host_floor_gib} GiB')
    if pairs == 'first5':
        plan = [(c, 'control') for c in fd.plan_control()]+[(c, 'fields') for c in fd.plan_new()]+[(c, 'only described') for c in fd.plan_fl_repairs()]
        out(f'  # | phase | arm | ...  (the repairs of at most {fd.REPAIR_TOP} arms the code chooses, 5 pairs each, cells R<arm> c{fd.NATIVE}, go before the ones only described)')
    else:
        fields, repairs = fd.plan_rest(arm, steps)
        plan = [(c, 'rest14 fields') for c in fields]+[(c, 'rest14 repairs') for c in repairs]
    for i, (c, phase) in enumerate(plan, 1):
        print_cell(out, i, c, phase)
    e = estimates(results)
    out(f"estimates〔估算, from fl_v1's records of Hx; nobody has run these arms〕: a field run {'—' if e['task_seconds'] is None else format(e['task_seconds'], '.1f')} s + {'—' if e['io_seconds'] is None else format(e['io_seconds'], '.1f')} s writing; an evaluated field {'—' if e['evaluation_seconds'] is None else format(e['evaluation_seconds'], '.1f')} s; a repair process {'—' if e['repair_process_seconds'] is None else format(e['repair_process_seconds'], '.1f')} s (Hx from 2400 steps; more infeasible cells may take longer, its wall is 120 s)")
    out(f"estimates: {e['n_fields']} field runs {_minutes(e['seconds_fields'])} with the gaps, their evaluation {_minutes(e['seconds_fields_evaluation'])}; at most {e['n_repairs_at_most']} repairs and {e['n_fl_repairs']} of RHx600 {_minutes(e['seconds_repairs'])}, their evaluation {_minutes(e['seconds_repairs_evaluation'])}; one arm of the other 14 pairs {_minutes(e['rest14_seconds'])}")
    out(f"estimates: {e['n_fields']} field files of {'—' if e['field_bytes'] is None else format(e['field_bytes']/2**20, '.0f')} MiB = {'—' if e['field_files_gib'] is None else format(e['field_files_gib'], '.1f')} GiB on /home; one arm of the other 14 pairs {'—' if e['rest14_field_files_gib'] is None else format(e['rest14_field_files_gib'], '.1f')} GiB; repair.npz and records small; the JIT cache some MiB")
    problems = safe_input_problems(out, pairs=pairs, stage=stage, seg_root=seg_root, allowlist=allowlist, unet=unet, results=results)
    out(f'/home free {shutil.disk_usage("/home").free/2**30:.0f} GB; /mnt/c free {shutil.disk_usage("/mnt/c").free/2**30:.0f} GiB')
    code = Path(root)/'code'
    out('--- start strings (the queue: the lead; the others: the queue, one per run; the code at $ROOT/code)')
    if pairs == 'first5':
        out(shlex.join(queue_launch(root)))
        for c, _ in plan:
            out(shlex.join(base.run_argv(c, code, root, '<sha>')))
        for unit in (EVAL_UNIT, EVAL_R1_UNIT):
            out(shlex.join(base.eval_argv(code, root, unit)))
        out('--- the other 14 pairs: only the lead, after reading the first 5, at the same commit (example: arm Hx4)')
        out(shlex.join(rest_launch('Hx4', fd.NATIVE, '<sha of batch.json>', root)))
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
        d = argparse.ArgumentParser(description='FD: the dry run')
        d.add_argument('--dry-run', action='store_true')
        d.add_argument('--pairs', choices=('first5', 'rest14'), default='first5')
        d.add_argument('--arm', default=None)
        d.add_argument('--steps', type=int, default=None)
        o = d.parse_args(argv)
        if o.pairs == 'rest14' and (o.arm is None or o.steps is None):
            print('usage: --dry-run --pairs rest14 --arm A --steps N', file=sys.stderr)
            return 2
        return dry_run(pairs=o.pairs, arm=o.arm, steps=o.steps)
    p = argparse.ArgumentParser(description='FD: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(fd.DEFAULT_ROOT))
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
        sub.add_parser(name).add_argument('--root', default=str(fd.DEFAULT_ROOT))
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
    problem = fd.root_problem(o.root)
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
