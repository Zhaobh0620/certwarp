"""RP (runs/20261105-rp/任务书.md, plan.md): the driver of the two repetitions of RW's repair, R2 and R3, and of the fallback of 2c(i) (two tiers: configuration C's field qualified by d0_v1's records, then the identity u = 0).  Not a method: no F_B, no certified descent, only the final field passes the certificate; development pairs only, in no result table.  The lead starts it; the gate is closed while any other tag-* or seq-vp-* unit is active.
Batches and the identity check: rp_core.py.  Reading: decide_rp.py.  The repair, its source, its fuse and its record are RW's own functions in run_fx.py (RepairWithFuse, rw_source_of_repair, repair_one), on another instance of run_fl.py per repetition, as run_fx.py does per batch; run_fx, fx_core, decide_fx, run_fl, decide_fl and every other module are not touched.

    python -B experiments/fl/run_rp.py --dry-run
        the plan (38 repairs), the identity of the code against RW's 85fc80d, the inputs (fd2_v1's fields against their pins, rw_v1's 19 repairs), the 19 fields of configuration C by d0_v1's records (hashed against their pins; a field that does not qualify is listed, it is not a problem: the identity takes its place), the estimates, the start strings.  No repair, no truth label, torch not imported, nothing written.
    run_rp.py queue [--commit SHA] [--wait-minutes 30] [--gap-seconds 5] [--min-host-gib 15]
        the lead's start.  Before any checkout: the identity of the commit, the inputs.  Then R2's 19 repairs in RW's order, R3's 19, one GPU unit at a time behind the gate; then tag-rp2-eval, tag-rp3-eval (CPU), and tag-rp-eval-id (CPU) only if a field of configuration C does not qualify; the reading.
    run_rp.py rp2|rp3 repair-one --arm RHx24 --tier T --pair P --out DIR [--device cuda] [--code-sha SHA]     one unit of the queue: RW's repair
    run_rp.py rp2|rp3 evaluate --root rp_v1/r2|r3       RW's evaluation of what has no metrics yet (truth labels, for the metrics only)
    run_rp.py id evaluate --root rp_v1/id               the identity u = 0 on the pairs whose field of configuration C does not qualify: its certificate at alpha = 1, the metrics (truth labels, for the metrics only)
    run_rp.py decide                                    the reading again
    run_rp.py write-pins                                rp_fallback_v1.json from d0_v1's JSON records, once (no field is read)

Writes only under rp_v1 (rp_v1/r2, rp_v1/r3, rp_v1/id); rw_v1, fd2_v1, d0_v1 are only read.  A run directory that exists is never rerun, failures are kept.  Exit codes: 0 done, 2 usage, another driver on the root, another root, 8 STOP or a failed check.
"""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_fx as fxrun  # noqa: E402  (puts experiments/fl, ef, mc0, nn on the path; its own instances of run_fl stay its own)
import decide_rp as decision  # noqa: E402
import rp_core as rp  # noqa: E402
from rp_core import ALL_PAIRS, BATCHES, REPEATS, RP  # noqa: E402

fx, core, fl, mc = fxrun.fx, fxrun.core, fxrun.fl, fxrun.mc
run = fxrun.run                                 # run_mc0
RW = fx.RW
SCRIPT = 'experiments/fl/run_rp.py'
UNDECIDABLE = decision.UNDECIDABLE


def _instance(batch):
    """run_fl once more under another name, its module names pointed at this repetition: run_fx._instance's lines, with this entry's script and queue unit; then RW's fuse, RW's source of a repair, every one of the 19 pairs in the plan."""
    p = batch.PREFIX
    spec = importlib.util.spec_from_file_location(f'_{p}_run_fl', HERE/'run_fl.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.fl, module.decision, module.SCRIPT, module.QUEUE_UNIT = batch, decision, SCRIPT, rp.QUEUE_UNIT
    module.EVAL_UNIT, module.EVAL_R1_UNIT, module.EVAL_X_UNIT, module.EVAL_R2_UNIT, module.EVAL_REST_UNIT, module.EVAL_REST_R_UNIT = f'tag-{p}-eval', f'tag-{p}-eval-r', f'tag-{p}-eval-x', f'tag-{p}-eval-r2', f'tag-{p}-eval-rest', f'tag-{p}-eval-rest-r'
    module.eval_argv.__defaults__ = (module.EVAL_UNIT,)
    module.GATE = replace(module.GATE, own_units=(f'{rp.QUEUE_UNIT}.service',))
    base = module.run

    def unit_argv(unit, code, inner, *, log, gpu=True, numba=module.RUN_THREADS, quota=800, runtime=base.RUN_RUNTIME_MAX):
        return base.unit_argv(unit, code, [p, *inner], log=log, gpu=gpu, numba=numba, quota=quota, runtime=runtime, script=SCRIPT, memory_max=module.MEMORY_MAX, nice=module.NICE)

    def run_argv(cell, code, root, sha='n/a'):
        d = batch.run_dir(root, cell)
        return unit_argv(cell.unit, code, ['repair-one', '--arm', cell.arm, '--tier', cell.tier, '--pair', cell.pair, '--out', str(d), '--code-sha', sha], log=d/'log.txt', numba=module.REPAIR_THREADS, quota=module.REPAIR_QUOTA, runtime=fx.REPAIR_RUNTIME_SECONDS)
    module.unit_argv, module.run_argv = unit_argv, run_argv
    module.repair = fxrun.RepairWithFuse(module.repair, fx.FUSE_SECONDS)
    module.source_of_repair = fxrun.rw_source_of_repair
    module.repair_allowed = lambda cell, root: True     # all 19 pairs are in every repetition's plan; require_cell refuses the rest
    return module


BASES = {rep: _instance(BATCHES[rep]) for rep in REPEATS}
ANY = BASES[REPEATS[0]]                         # the constants the two instances share: run_fl's
READING_FILES = tuple((HERE/Path(rel).name, rel) for rel in ('experiments/fl/decide_rp.py', 'experiments/fl/rp_core.py', 'experiments/fl/decide_fx.py', 'experiments/fl/fx_core.py', 'experiments/fl/decide_fl.py', 'experiments/fl/fl_core.py', 'experiments/fl/decide_fd2.py', 'experiments/fl/fd2_core.py', rp.FALLBACK_REL, 'experiments/fl/rw_sources_v1.json'))


# ---- the fallback of 2c(i): configuration C's fields -------------------------------------------------------------------------------------------
def fallback_pins_from_records(results=None):
    """The pins of rp_fallback_v1.json, from JSON records only (no field is read): d0_v1's final_native.json and metrics.json for each pair; the configuration and the version of the certificate RW's evaluation recorded (rw_v1's control, repair_metrics.json), which every A0 certificate must have."""
    control = RW.cell_of(RW.CONTROL_PAIR)
    attempt = run.read_json(RW.run_dir(RP.r1_root(results), control)/'repair_metrics.json')['after']['certificate']['attempts'][0]
    pairs = {}
    for pair in ALL_PAIRS:
        d = RP.a0_run_dir(pair, results)
        final, metrics = run.read_json(d/rp.A0_TASK/'final_native.json'), run.read_json(d/'metrics.json')
        cert = final['certificate']
        pairs[core.short_name(pair)] = dict(run_id=metrics['run_id'], field=f'runs/{rp.A0_ARM}/{pair}/{rp.A0_TASK}/{rp.A0_FILE}', file_sha256=final['sha256'], u_sha256=cert['state_hash'],
                                            certificate=dict(status=cert['status'], min_jd_lower_bound=cert['min_jd_lower_bound'], n_infeasible=cert['n_infeasible'], n_uncertain=cert['n_uncertain'], certificate_version=cert['certificate_version'], config=dict(cert['config'])),
                                            d0_evaluation=dict(passed=metrics['certificate']['passed'], state_hash_matches_u=metrics['certificate']['state_hash_matches_u'], final_npz_sha256=metrics['certificate']['final_npz_sha256'], final_npz_sha256_matches=metrics['certificate']['final_npz_sha256_matches'], classification=metrics['classification']['class'], evaluator_sha256=metrics['evaluator']['sha256']))
    return dict(schema='bayesreg.rp.fallback.v1', source='d0_v1 arm A0: configuration C (frozen eb808be, 90 s) on the 19 development pairs, final_native.npz, read only; the fallback of 2c(i) (BZ about 8:10 pm CST 11-05)', pairs=pairs,
                evaluation_certificate=dict(source=f'rw_v1/runs/{control.arm}_{control.tier}/{control.short}/repair_metrics.json after.certificate.attempts[0]', config=attempt['config'], certificate_version=attempt['certificate_version']))


def evaluate_identity(d, pair, labs, truth_pin, ev, certify, config):
    """The second tier: u = 0 on the pair's grid; its certificate at alpha = 1 (configuration C; J = 1 everywhere), the metrics with RW's function."""
    from time import perf_counter
    u = np.zeros(tuple(truth_pin.grid.shape)+(3,))
    tick = perf_counter()
    ladder = run.certify_ladder(u, truth_pin.grid, (1.,), certify, config)
    seconds_certificate = perf_counter()-tick
    metrics = ANY.field_metrics(u, labs, tuple(truth_pin.grid.spacing), ev)
    run.write_json(d/'identity_metrics.json', dict(schema='bayesreg.rp.identity_metrics.v1', pair=pair, short=core.short_name(pair), oracle=False, role="the second tier of 2c(i) (D75, for BZ): the identity u = 0, used only where a repair failed and configuration C's field does not qualify",
                                                  field='u = 0', certificate=ladder, certified_at_1=ladder['accepted_alpha'] == 1., seconds_certificate=seconds_certificate, metrics=metrics, certificate_config=dict(rp.CERTIFICATE_C),
                                                  evaluator=dict(script='experiments/d0/evaluate.py', sha256=core.sha_file(core.ROOT/'experiments/d0/evaluate.py')), evaluated_utc=run.utc()))


def identity_pairs(results=None, document=None, stage=None):
    """The pairs whose field of configuration C does not qualify by d0_v1's records: the identity is evaluated for these, whether a repair fails there or not."""
    return [pair for pair in ALL_PAIRS if not rp.fallback_status(pair, results=results, document=document, stage=stage)['qualified']]


def identity_evaluate(o, *, results=None, document=None, stage=None):
    """The unit tag-rp-eval-id.  First core.frozen_api(), as run_fl.evaluate_all: jd_curl from the frozen tree, preopt.py the pinned one.  A pair with a record (metrics or error) is not done again; an error is kept and the others go on."""
    root = Path(o.root)
    problem = RP.id_problem(root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    core.frozen_api()
    pairs = identity_pairs(results, document, stage)
    root.mkdir(parents=True, exist_ok=True)
    run.write_json(root/f"plan_{run.utc().replace(':', '').replace('-', '')}.json", dict(pairs=[core.short_name(p) for p in pairs], rule='the pairs whose field of configuration C does not qualify by d0_v1\'s records', utc=run.utc()))
    if not pairs:
        return 0
    ANY.enable_jit(root)
    ev = core.load_module('_mc0_evaluate', 'experiments/d0/evaluate.py')
    from jd_curl.three_d.feasibility import certify_map
    config = run.certificate_config()
    failures = 0
    for pair in pairs:
        d = RP.id_dir(root, pair)
        if (d/'identity_metrics.json').exists() or (d/'identity_error.json').exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        try:
            truth_pin, labs = fl.read_truth(pair, ANY.TRUTH_MEMBERS)
            evaluate_identity(d, pair, labs, truth_pin, ev, certify_map, config)
        except Exception as error:                      # kept; the others go on; not retried
            run.write_json(d/'identity_error.json', dict(pair=pair, error=repr(error)[:2000], utc=run.utc()))
            failures += 1
    return 8 if failures else 0


# ---- the checks of the inputs -------------------------------------------------------------------------------------------------------------------
def r1_input_problems(out=lambda text: None, *, results=None, sources=None):
    """rw_v1 as R1: RW's commit, its control judged not bit for bit (so all 19 repairs are its own), each repair of the pinned field with wall_seconds 1200, evaluated.  Records only."""
    root = RP.r1_root(results)
    batch = run.read_json(root/'batch.json') or {}
    problems = []
    out(f"R1 {root}: batch code {batch.get('code_sha')}; RW's commit {rp.BASE_COMMIT}: {batch.get('code_sha') == rp.BASE_COMMIT}")
    if batch.get('code_sha') != rp.BASE_COMMIT:
        problems.append(f"R1: rw_v1/batch.json is at {batch.get('code_sha')}, not {rp.BASE_COMMIT}")
    choice = RW.control_choice(root)
    out(f"R1 control: bit for bit fd2_v1's repair: {None if choice is None else choice['same']} (False: RW repaired all 19 itself)")
    if choice is None or choice['same'] is not False:
        problems.append("R1: rw_v1's control is not judged 'not bit for bit': rw_v1 does not hold 19 repairs of its own")
    pins = RW.sources(sources)
    for pair in ALL_PAIRS:
        short, d = core.short_name(pair), RW.run_dir(root, RW.cell_of(pair))
        repair, metrics = run.read_json(d/'repair.json') or {}, run.read_json(d/'repair_metrics.json') or {}
        certificate = (metrics.get('after') or {}).get('certificate') or {}
        source = (repair.get('source') or {}).get('tapered_u_sha256')
        out(f"R1 {short}: {repair.get('status')}, {len(repair.get('rounds') or [])} rounds, wall hit {fxrun.decision.hit_wall(repair)}, seconds_repair {repair.get('seconds_repair')}, wall_seconds {(repair.get('config') or {}).get('wall_seconds')}, code {str(repair.get('code_sha'))[:7]}; evaluated: {bool(metrics)}, accepted alpha {certificate.get('accepted_alpha')}; the source is the pin: {source == (pins.get(short) or {}).get('tapered_u_sha256')}")
        if repair.get('oracle') is not False or (repair.get('config') or {}).get('wall_seconds') != fx.FUSE_SECONDS or repair.get('code_sha') != rp.BASE_COMMIT or source != (pins.get(short) or {}).get('tapered_u_sha256') or not metrics:
            problems.append(f"R1 {short}: not RW's evaluated repair of the pinned field with wall_seconds {fx.FUSE_SECONDS:g} at {rp.BASE_COMMIT[:7]}")
    return problems


def fallback_lines(out=lambda text: None, *, results=None, fallback=None, stage=None):
    """The 19 fields of configuration C by d0_v1's records (rp_core.fallback_status): {short: status}.  A field that does not qualify is listed for the lead, not a problem: the identity takes its place (the lead, about 9 pm CST 11-05)."""
    document = RP.fallback_document(fallback)
    statuses = {}
    for pair in ALL_PAIRS:
        status = rp.fallback_status(pair, results=results, document=document, stage=stage)
        statuses[status['pair']] = status
        c = status['run_certificate']
        out(f"A0 {status['pair']}: {status['field']}; sha256 {status['file_sha256']}; u {status['u_header']}; the run's own certificate {c['status']}, lower bound {c['min_jd_lower_bound']}, infeasible {c['n_infeasible']}, uncertain {c['n_uncertain']}, {c['config']}; D0: class {status['d0_class']}, passed {status['d0_passed']}; qualified by the records: {status['qualified']}" + ('' if status['qualified'] else f"; why: {status['why']}"))
    bad = [k for k, v in statuses.items() if not v['qualified']]
    out(f"configuration C's fields that do not qualify (the identity u = 0 takes their place; not a problem, for the lead): {bad or 'none'}")
    return statuses


def rp_input_problems(out=lambda text: None, *, stage=None, allowlist=None, results=None, sources=None):
    """RW's own checks of its inputs (the stage, the frozen tree, fd2_v1's fields against their pins, hashed), then R1.  The fallback is not among them: fallback_lines."""
    return fxrun.rw_input_problems(out, stage=stage, allowlist=allowlist, results=results, sources=sources)+r1_input_problems(out, results=results, sources=sources)


# ---- the queue --------------------------------------------------------------------------------------------------------------------------------------
def hold_reason(root):
    """rp_v1/hold stops the next repair (the GPU gate) and the next evaluation (the gate at rp_v1)."""
    return lambda: 'hold file' if (Path(root)/'hold').exists() else None


def gate_of(o, root):
    return replace(ANY.GATE, own_units=(f'{rp.QUEUE_UNIT}.service',), host_floor_gib=o.min_host_gib, gpu_hold=hold_reason(root))


def batch_value(o, root, gate, identity_record, fallback=None):
    return dict(plan=[c.run_id for c in RP.plan()], repeats={f'R{rep}': str(RP.rep_root(root, rep)) for rep in REPEATS}, r1=dict(root=str(RP.r1_root()), code_sha=rp.BASE_COMMIT), identity_evaluation=str(RP.id_root(root)), gap_seconds=o.gap_seconds, identity=identity_record,
                fallback_at_start={k: dict(qualified=v['qualified'], why=v['why'], file_sha256=v['file_sha256']) for k, v in (fallback or {}).items()},
                pins=dict(rw_sources=dict(file=str(RW.SOURCES_FILE), sha256=core.sha_file(RW.SOURCES_FILE)), fallback=dict(file=str(rp.FALLBACK_FILE), sha256=core.sha_file(rp.FALLBACK_FILE))),
                caps=dict(run_seconds=fl.RUN_CAP_SECONDS, run_units_seconds=ANY.RUN_UNITS_CAP_SECONDS, device_gib=mc.VRAM_LIMIT_BYTES/2**30, memory_max=ANY.MEMORY_MAX, nice=ANY.NICE, repair_wall_seconds=fx.FUSE_SECONDS, repair_runtime_seconds=fx.REPAIR_RUNTIME_SECONDS),
                gate=dict(own_units=list(gate.own_units), load_limit=gate.load_limit, gpu_free_mib=gate.gpu_free_mib, mem_available_gib=gate.mem_available_gib, home_floor_gb=run.HOME_FLOOR_GB, host_floor_gib=gate.host_floor_gib, hold=str(Path(root)/'hold')), evaluation=dict(numba_threads=ANY.EVAL_THREADS, cpu_quota_percent=ANY.EVAL_QUOTA))


def write_batch(o, root, sha, gate, identity_record, fallback=None):
    """batch.json: written once; a restart must be at the same code and the same plan."""
    value = batch_value(o, root, gate, identity_record, fallback)
    batch = run.read_json(Path(root)/'batch.json')
    if batch is None:
        run.write_json(Path(root)/'batch.json', dict(value, schema='bayesreg.rp.batch.v1', code_sha=sha, frozen_commit=core.FROZEN_COMMIT, started_utc=run.utc()))
    elif batch['code_sha'] != sha:
        raise run.Stop(f"batch.json is at {batch['code_sha']}, the code at {sha}")
    elif batch['plan'] != value['plan']:
        raise run.Stop('batch.json has another plan')


def reading_code_problem(root):
    """Why this process may not write this root's reading, or None: the files that hold the rules must be the batch's commit, byte for byte.  No batch.json: nothing has run."""
    batch = run.read_json(Path(root)/'batch.json')
    if batch is None:
        return None
    for path, rel in READING_FILES:
        theirs = ANY.committed_bytes(batch['code_sha'], rel)
        if theirs is None or theirs != path.read_bytes():
            return f"{path} is not {rel} of the batch's commit {batch['code_sha']}: the reading's rules would not be the batch's; run the driver from $ROOT/code"
    return None


def id_eval_argv(code, root):
    where = RP.id_root(root)
    return run.unit_argv(rp.ID_EVAL_UNIT, code, ['id', 'evaluate', '--root', str(where)], log=where/'eval.log', gpu=False, numba=ANY.EVAL_THREADS, quota=ANY.EVAL_QUOTA, runtime=run.EVAL_RUNTIME_MAX, script=SCRIPT, memory_max=ANY.MEMORY_MAX, nice=ANY.NICE)


def evaluation_units(code, root):
    """(unit, argv, directory): R2's, R3's, then the identity's, all CPU."""
    units = [(BASES[rep].EVAL_UNIT, BASES[rep].eval_argv(code, RP.rep_root(root, rep), BASES[rep].EVAL_UNIT), RP.rep_root(root, rep)) for rep in REPEATS]
    return units+[(rp.ID_EVAL_UNIT, id_eval_argv(code, root), RP.id_root(root))]


def evaluations(o, system, root, log, code, gate, need_identity):
    """Every evaluation unit behind the CPU gate at rp_v1; a repetition that never started has nothing to evaluate; the identity only when a field of configuration C does not qualify, and only when a repair ran.  Returns ({unit: rc}, why one did not start or None)."""
    rcs = {}
    if not any((RP.rep_root(root, rep)/'runs').exists() for rep in REPEATS):
        log('no repair ran: nothing to evaluate')
        return rcs, None
    for unit, argv, where in evaluation_units(code, root):
        if unit == rp.ID_EVAL_UNIT and not need_identity:
            log(f"{unit}: every field of configuration C qualifies by d0_v1's records; the identity is not evaluated")
            continue
        if unit != rp.ID_EVAL_UNIT and not (Path(where)/'runs').exists():
            log(f'{unit}: {where} has no runs; nothing to evaluate')
            continue
        Path(where).mkdir(parents=True, exist_ok=True)
        try:
            run.wait_for_gate(system, root, False, o.wait_minutes, log, gate)
        except run.Stop as error:
            log(f'{unit} not started: {error}')
            return rcs, str(error)
        log(f'launch {unit}')
        rcs[unit] = system.run(argv, Path(root)/f'{unit}.systemd_run.log')
    return rcs, None


def ledgers(root):
    return {f'R{rep}': BASES[rep].ledger(RP.rep_root(root, rep)) for rep in REPEATS}


def finish(log, result, rcs, stop):
    """Exit 8 when something failed or a reading cannot be read: the reading is written all the same."""
    log(f"strict: reading 1 {result['reading']}, reading 2 {result['reading_spread']}; code: reading 1 {result['code_convention']['reading']}, reading 2 {result['code_convention']['reading_spread']}; the identity used {result['strict']['n_identity_used']} / {result['code_convention']['n_identity_used']} times.  Handed to the lead: no next step starts here")
    failed = {unit: rc for unit, rc in rcs.items() if rc != 0}
    if failed:
        raise run.Stop(f'an evaluation unit failed: {failed}; the reading is written, a repair without metrics counts as not read')
    if stop is not None:
        raise run.Stop(f'a phase stopped: {stop}')
    for how in ('strict', 'code_convention'):
        r = result[how]
        for reading, reason in ((r['reading'], r['reason']), (r['reading_spread'], r['reason_spread'])):
            if reading == UNDECIDABLE:
                raise run.Stop(f"a reading ({r['convention']}) is {UNDECIDABLE}: {reason}"[:1000])
    return 0


def body(o, system, root, log):
    """The identity of the commit and the inputs before anything is checked out; R2, R3; the evaluations; the reading."""
    ANY.keep_old_stops(root)
    gate = gate_of(o, root)
    sha = fxrun.nx_commit(o.commit)
    record, problems = rp.identity(sha)
    if problems:
        raise run.Stop(f"the code is not RW's: {problems}; nothing checked out")
    problems = fxrun.safe(rp_input_problems)
    if problems:
        raise run.Stop(f'the inputs, before any run: {problems}; nothing checked out')
    for rep in REPEATS:
        if RP.rep_root(root, rep).resolve() != Path(BATCHES[rep].DEFAULT_ROOT).resolve():
            raise run.Stop(f'{RP.rep_root(root, rep)} is not the root of R{rep}, {BATCHES[rep].DEFAULT_ROOT}')
    try:
        fallback = fallback_lines()
    except (Exception, SystemExit) as error:            # noqa: BLE001 - the pins or the records could not be read at all
        raise run.Stop(f"configuration C's fields could not be read, before any run: {type(error).__name__}: {error}; nothing checked out"[:1000])
    code, sha = system.code_checkout(root, sha)
    write_batch(o, root, sha, gate, record, fallback)
    problem = reading_code_problem(root)
    if problem:
        raise run.Stop(problem)
    log(f"start: R2 and R3, {len(RP.plan())} repairs in RW's order, wall_seconds {fx.FUSE_SECONDS:g}; the code {sha}, RW's files byte for byte those of {rp.BASE_COMMIT[:7]}; /mnt/c floor {o.min_host_gib:g} GiB")
    stop = None
    try:
        for rep in REPEATS:
            end = BASES[rep].run_phase(o, system, RP.rep_root(root, rep), log, code, sha, BATCHES[rep].plan(), gate)
            log(f'R{rep}: {end}')
    except run.Stop as error:
        stop = str(error)
        log(f'STOP: {stop}')
    bad = sorted(k for k, v in fallback.items() if not v['qualified'])
    log(f"configuration C's fields that do not qualify by d0_v1's records: {bad or 'none'}")
    rcs, eval_stop = evaluations(o, system, root, log, code, gate, bool(bad))
    stop = stop or eval_stop
    state = ledgers(root)
    result = decision.write_rp(root, stops={k: v['arms_stopped'] for k, v in state.items()})
    run.write_json(Path(root)/'queue_end.json', dict(stop=stop, eval_rc=rcs, reading=result['reading'], reason=result['reason'], reading_spread=result['reading_spread'], reason_spread=result['reason_spread'], method_seg35_range=result['method_seg35_range'],
                                                     code_convention={k: result['code_convention'][k] for k in ('reading', 'reason', 'reading_spread', 'reason_spread', 'method_seg35_range', 'n_identity_used')}, n_identity_used=result['strict']['n_identity_used'], fallback_not_qualified=bad, ledgers=state, utc=run.utc()))
    return finish(log, result, rcs, stop)


def queue(o, system=None):
    problem = RP.root_problem(o.root)
    if problem:                                         # before the lock file: nothing is created or touched
        print(problem, file=sys.stderr)
        return 2

    def guarded(o, system, root, log):
        try:
            return body(o, system, root, log)
        except SystemExit as refused:                   # SelectionChanged among them; a wrong mark met by the reading: a STOP with its record
            raise run.Stop(str(refused))
    return run.queue(o, system, body=guarded, prefix='rp')


def queue_launch(root=None):
    """The lead's start string."""
    return run.check_launch(['systemd-run', '--user', f'--unit={rp.QUEUE_UNIT}', '--collect', f'--working-directory={core.MAIN_REPO}', f'--property=MemoryMax={ANY.MEMORY_MAX}', f'--nice={ANY.NICE}', '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', run.PY, '-B', SCRIPT, 'queue', '--root', str(RP.DEFAULT_ROOT if root is None else root), '--gap-seconds', str(ANY.GAP_SECONDS), '--min-host-gib', f'{ANY.DEFAULT_MIN_HOST_GIB:g}'])


# ---- estimates ------------------------------------------------------------------------------------------------------------------------------------
def estimates(results=None):
    """〔估算〕from rw_v1's own records: one repetition = RW's unit seconds and gaps; an evaluation of a field = its certificate seconds plus run_fl's constants for the metrics; a gate wait after a CPU unit.  None where there is no record.  A repair that hits the fuse has no bound but the fuse."""
    root = RP.r1_root(results)
    units, certificates, sizes = [], [], []
    for pair in ALL_PAIRS:
        d = RW.run_dir(root, RW.cell_of(pair))
        done, metrics = run.read_json(d/'queue_done.json') or {}, run.read_json(d/'repair_metrics.json') or {}
        if done.get('unit_seconds') is not None:
            units.append(done['unit_seconds'])
        if (metrics.get('after') or {}).get('seconds_certificate') is not None:
            certificates.append(metrics['after']['seconds_certificate'])
        if (d/'repair.npz').is_file():
            sizes.append((d/'repair.npz').stat().st_size)
    n, gap = len(ALL_PAIRS), ANY.GAP_SECONDS
    out = dict(n_r1_records=len(units), gate_wait_seconds=fxrun.GATE_WAIT_SECONDS)
    if len(units) == n and len(certificates) == n:
        one_eval = sum(certificates)/n+ANY.HD95_SECONDS+ANY.EVAL_BASE_SECONDS
        repetition = sum(units)+(n-1)*gap
        out.update(repetition_seconds=repetition, evaluation_seconds_per_field=one_eval, evaluations_seconds=2*n*one_eval+2*fxrun.GATE_WAIT_SECONDS, total_seconds=2*(repetition+gap)+2*n*one_eval+2*fxrun.GATE_WAIT_SECONDS, identity_seconds_per_pair=ANY.HD95_SECONDS+ANY.EVAL_BASE_SECONDS, fuse_cell_seconds=fx.FUSE_SECONDS+15., disk_gib=None if len(sizes) != n else 2*sum(sizes)/2**30)
    return out


def _minutes(seconds):
    return '—' if seconds is None else f'{seconds/60:.0f} min'


# ---- dry run --------------------------------------------------------------------------------------------------------------------------------------
def dry_run(*, out=print, root=None, stage=None, allowlist=None, results=None, sources=None, fallback=None, commit=None):
    """The plan, the identity of the code, the checks of the queue, the estimates, the start strings.  No truth label, no repair, nothing written."""
    root = RP.DEFAULT_ROOT if root is None else Path(root)
    code = root/'code'
    out(f'RP dry run ({run.utc()}): no repair, no truth label, no certificate, nothing written; real files read: JSON records, the npy headers of fixed, moving and of configuration C\'s u, and for their sha256 the source fields of fd2_v1 and the 19 fields of configuration C')
    gate = ANY.GATE
    out(f'root {root}; root exists: {root.exists()}; R2 {RP.rep_root(root, 2)}, R3 {RP.rep_root(root, 3)}, the identity {RP.id_root(root)}; R1 {RP.r1_root(results)} (read only); repair wall_seconds {fx.FUSE_SECONDS:g} s (a fuse), RuntimeMaxSec {fx.REPAIR_RUNTIME_SECONDS}; device {mc.VRAM_LIMIT_BYTES/2**30} GiB; MemoryMax {ANY.MEMORY_MAX}, nice {ANY.NICE}; {ANY.GAP_SECONDS} s between runs')
    out(f"gate: own units ('{rp.QUEUE_UNIT}.service',), load < {gate.load_limit}, GPU free >= {gate.gpu_free_mib} MiB, MemAvailable >= {gate.mem_available_gib} GiB, /home >= {run.HOME_FLOOR_GB} GB, /mnt/c >= {gate.host_floor_gib} GiB, no {root/'hold'}; closed while any other tag-* or seq-vp-* unit is active")
    for i, cell in enumerate(RP.plan(), 1):
        out(f'{i:2d} | R{int(cell.PREFIX[2:])} | {cell.arm} | {cell.tier} | {cell.short} | {cell.unit} | {cell.run_id}')
    sha = fxrun.nx_commit(commit)
    rc, status, _ = rp.git('status', '--porcelain', '--untracked-files=no')
    record, problems = rp.identity(sha)
    out(f"the code: {sha}; tracked files clean: {rc == 0 and not status.strip()}; against RW's {rp.BASE_COMMIT}: {record.get('n_files_at_base')} files at RW's commit under {list(rp.IDENTITY_PATHS)} and the root's .py, changed {record.get('n_files_changed')}; added: this package's {record.get('added_own')}, others {record.get('added_other')}")
    f = record.get('frozen') or {}
    out(f"the frozen tree {f.get('frozen')}: {f.get('n_files')} tracked files of {f.get('commit')} under {f.get('paths')}, the same blob: {f.get('same')}; missing {f.get('missing')}, not the same {f.get('mismatched')}, untracked .py {f.get('untracked_py')}")
    out(f"pinned inputs: {RW.SOURCES_FILE.name} sha256 {core.sha_file(RW.SOURCES_FILE) if Path(RW.SOURCES_FILE).is_file() else None}; {rp.FALLBACK_FILE.name} sha256 {core.sha_file(rp.FALLBACK_FILE) if rp.FALLBACK_FILE.is_file() else None}")
    try:
        source_pins = RW.sources(sources)
    except (OSError, ValueError, KeyError) as error:
        source_pins = {}
        problems.append(f'{RW.SOURCES_FILE}: {type(error).__name__}: {error}')
    for short, pin in source_pins.items():
        out(f"pinned source {short}: field_sha256 {pin.get('field_sha256')}, tapered_u_sha256 {pin.get('tapered_u_sha256')}")
    problems += fxrun.safe(rp_input_problems, out, stage=stage, allowlist=allowlist, results=results, sources=sources)
    statuses = fxrun.safe(fallback_lines, out, results=results, fallback=fallback, stage=stage)
    if isinstance(statuses, list):
        problems += statuses
        statuses = {}
    e = estimates(results)
    out(f"estimates〔估算, from rw_v1's records; RP has not run〕: one repetition {_minutes(e.get('repetition_seconds'))} (RW's unit seconds and gaps), the evaluations of 2 x 19 repaired fields {_minutes(e.get('evaluations_seconds'))}, all {_minutes(e.get('total_seconds'))}; the identity on {sum(not v['qualified'] for v in statuses.values())} pairs about {e.get('identity_seconds_per_pair')} s each; a repair at the fuse {_minutes(e.get('fuse_cell_seconds'))}; disk {None if e.get('disk_gib') is None else round(e['disk_gib'], 2)} GiB of repair.npz and small records")
    out(f'/home free {shutil.disk_usage("/home").free/2**30:.0f} GB; /mnt/c free {shutil.disk_usage("/mnt/c").free/2**30:.0f} GiB')
    out('--- start strings (the queue: the lead; the others: the queue, one per run; the code at $ROOT/code)')
    out(shlex.join(queue_launch(root)))
    for rep in REPEATS:
        for cell in BATCHES[rep].plan():
            out(shlex.join(BASES[rep].run_argv(cell, code, RP.rep_root(root, rep), '<sha>')))
    for unit, argv, _ in evaluation_units(code, root):
        out(shlex.join(argv))
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


# ---- the entries -----------------------------------------------------------------------------------------------------------------------------------
def rep_main(rep, rest):
    base, batch = BASES[rep], BATCHES[rep]
    p = argparse.ArgumentParser(description=f'RP R{rep}: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('repair-one')
    for name in ('--arm', '--tier', '--pair', '--out'):
        r.add_argument(name, required=True)
    r.add_argument('--device', default='cuda')
    r.add_argument('--code-sha', default='n/a')
    sub.add_parser('evaluate').add_argument('--root', default=str(batch.DEFAULT_ROOT))
    o = p.parse_args(rest)
    if o.command == 'repair-one':
        return fxrun.repair_one(base, o)
    problem = batch.root_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    return base.evaluate_all(o)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv == ['--dry-run']:
        return dry_run()
    if argv[:1] in (['rp2'], ['rp3']):
        return rep_main(int(argv[0][2:]), argv[1:])
    if argv[:2] == ['id', 'evaluate']:
        a = argparse.ArgumentParser(description='RP: the evaluation of the identity, the second tier of the fallback')
        a.add_argument('--root', default=str(RP.id_root()))
        return identity_evaluate(a.parse_args(argv[2:]))
    p = argparse.ArgumentParser(description='RP: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(RP.DEFAULT_ROOT))
    q.add_argument('--commit', default=None)
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--gap-seconds', type=int, default=ANY.GAP_SECONDS)
    q.add_argument('--min-host-gib', type=float, default=ANY.DEFAULT_MIN_HOST_GIB)
    sub.add_parser('decide').add_argument('--root', default=str(RP.DEFAULT_ROOT))
    sub.add_parser('write-pins')
    o = p.parse_args(argv)
    if o.command == 'queue':
        return queue(o)
    if o.command == 'write-pins':
        if rp.FALLBACK_FILE.exists():
            raise SystemExit(f'{rp.FALLBACK_FILE} is there: a pin is written once')
        rp.FALLBACK_FILE.write_text(json.dumps(fallback_pins_from_records(), indent=1)+'\n')
        return 0
    problem = RP.root_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    problem = reading_code_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 8
    decision.write_rp(o.root, stops={k: v['arms_stopped'] for k, v in ledgers(o.root).items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
