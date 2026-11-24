"""P2 / P3 (runs/20261106-p2/任务书.md, plan.md): the drivers of the Bayesian parts on the line of Hx24@600.  Not a method: no F_B, no certified descent, only the final field passes the
certificate; development pairs only, in no result table.  The lead starts each queue, one at a time, after UF (and HS, if it runs); the gate of a queue is closed while any other tag-* or
seq-vp-* unit is active.  Numbers, optimizer, cells: p2_core.py.  Readings: decide_p2.py.  One repair, one evaluation, the phases: run_fl.py's own functions, not copied: another instance
of that module per batch, as run_fx.py does (fd2_core, fx_core, decide_fl, decide_fx, run_fl, run_fx are not touched).

    python -B experiments/p2/run_p2.py p2|p3 --dry-run [--segmenter ID]
        the plan, the unit strings, the sha256 of the inputs and of the reference, the estimates, the checks.  No optimization, no truth label, torch not imported, nothing written.
    run_p2.py write-reference                 once: experiments/p2/reference_unet35_v1_ens.json from fd2_v1's and rw_v1's JSON records (no field is read); the lead commits it
    run_p2.py p2 queue --segmenter ID [--commit SHA] [--wait-minutes 30] [--gap-seconds 5] [--min-host-gib 15]
        the 15 fields (A, C, AC on the first 5 pairs), tag-p2-eval, the repairs of the fields that are done and evaluated, tag-p2-eval-r, the reading (analysis/decision_p2.json).
    run_p2.py p3 queue --commit <P2's commit> [...]
        only when P2's reading is 有候选 (computed again from P2's records first): the candidate's 19 fields, tag-p3-eval, its repairs, tag-p3-eval-r; when a pair failed, tag-p3-fallback
        (CPU, no label: configuration C's field or the identity certified again, D75, C22.5-2) and tag-p3-eval-id (the identity's metrics); the reading.
    run_p2.py p2|p3 run-one|repair-one --arm A --tier T --pair P --out DIR [--device cuda] [--code-sha SHA]      one unit of a queue
    run_p2.py p2|p3 evaluate | decide

Each batch writes only under its own root (p2_v1, p3_v1); fd2_v1, rw_v1, d0_v1, p1_v1 are only read.  A run directory that exists is never rerun, failures are kept.  Exit codes: 0 done, 2 usage,
another driver on the root, another root, 8 STOP or a failed check.
"""
import argparse
from dataclasses import replace
import importlib.util
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
for _i, _rel in enumerate(('p2', 'fl', 'ef', 'mc0', 'p1')):
    sys.path.insert(_i, str(HERE.parent/_rel))

import numpy as np  # noqa: E402

import decide_p2 as decision  # noqa: E402
import decide_fx  # noqa: E402
import fd2_core as fd2  # noqa: E402
import fx_core as fx  # noqa: E402
import mc0_core as core  # noqa: E402
import p2_core as p2  # noqa: E402
import rp_core as rp  # noqa: E402

SCRIPT = 'experiments/p2/run_p2.py'
FALLBACK_UNIT, EVAL_ID_UNIT = 'tag-p3-fallback', 'tag-p3-eval-id'     # revision 2: D75's certificate again (no label), then the identity's metrics
GATE_WAIT_SECONDS = 90.                         # 〔估算〕after an evaluation unit, FD's queue.log
EM_OVERHEAD_SECONDS = 25.                       # 〔估算〕the Gram matrix (35 autograds at the native level) and 6 M steps on the host (35 trilinear samplings on Omega_B each)
TISSUE_SECONDS = 15.                            # 〔估算〕the tissue layer, the coarse levels of P~ (15 downsample_array calls), the moments (p1_core: 2-3 s a field)
HS_ADOPT = p2.HS_ADOPT


class RepairWithFuse:
    """run_fx.RepairWithFuse, copied: experiments/ef/repair.py with wall_seconds 1200 and nothing else changed; run_fl.repair_one asks this name for its RepairConfig."""

    def __init__(self, module, wall):
        self._module, self._wall = module, wall

    def __getattr__(self, name):
        return getattr(self._module, name)

    def RepairConfig(self, **kw):
        return self._module.RepairConfig(**dict(kw, wall_seconds=self._wall))


def _instance(name, batch):
    """run_fl once more under another name, its module names pointed at this batch (run_fx._instance's lines, with this script and the batch's name first in a unit's arguments)."""
    p = batch.PREFIX
    spec = importlib.util.spec_from_file_location(f'_{p}_run_fl', HERE.parent/'fl'/'run_fl.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.fl, module.decision, module.SCRIPT, module.QUEUE_UNIT = batch, decision, SCRIPT, f'tag-{p}-queue'
    module.EVAL_UNIT, module.EVAL_R1_UNIT, module.EVAL_X_UNIT, module.EVAL_R2_UNIT, module.EVAL_REST_UNIT, module.EVAL_REST_R_UNIT = f'tag-{p}-eval', f'tag-{p}-eval-r', f'tag-{p}-eval-x', f'tag-{p}-eval-r2', f'tag-{p}-eval-rest', f'tag-{p}-eval-rest-r'
    module.eval_argv.__defaults__ = (module.EVAL_UNIT,)
    module.GATE = replace(module.GATE, own_units=(f'{module.QUEUE_UNIT}.service',))
    run = module.run

    def unit_argv(unit, code, inner, *, log, gpu=True, numba=module.RUN_THREADS, quota=800, runtime=run.RUN_RUNTIME_MAX):
        return run.unit_argv(unit, code, [name, *inner], log=log, gpu=gpu, numba=numba, quota=quota, runtime=runtime, script=SCRIPT, memory_max=module.MEMORY_MAX, nice=module.NICE)

    def run_argv(cell, code, root, sha='n/a'):
        d = batch.run_dir(root, cell)
        command = 'repair-one' if cell.is_repair else 'run-one'
        return unit_argv(cell.unit, code, [command, '--arm', cell.arm, '--tier', cell.tier, '--pair', cell.pair, '--out', str(d), '--code-sha', sha], log=d/'log.txt', numba=module.REPAIR_THREADS if cell.is_repair else module.RUN_THREADS, quota=module.REPAIR_QUOTA, runtime=fx.REPAIR_RUNTIME_SECONDS if cell.is_repair else run.RUN_RUNTIME_MAX)
    module.unit_argv, module.run_argv = unit_argv, run_argv
    module.repair = RepairWithFuse(module.repair, fx.FUSE_SECONDS)
    module.repair_allowed = lambda cell, root: True             # every repair is planned; the queue starts one only for a field that is done and evaluated
    module.oracle_mark_checked = lambda cell, record: batch.channels.oracle_of_run(record)
    module.control_problem = lambda cell, record, results=None: c_control_problem(batch, cell, record)
    return module


NO_REFERENCE_FIELD, PASS0_TRUNCATED = '无参照场', '第 0 遍被墙钟截断'


def reference_row(batch, short):
    """The reference's row of this pair, None when the batch has no reference loaded or the row is not there."""
    stage = 'p2' if batch.PREFIX == 'p2' else 'p3'
    return (getattr(batch, 'reference', None) or {}).get(stage, {}).get('pairs', {}).get(short)


def reference_field_u(batch, short):
    return (reference_row(batch, short) or {}).get('field_u_sha256')


def pass0_control(batch, cell, row):
    """C's pass 0 against the reference's T88 field (revision 3): dict(checkable, same, why).  Not checkable, the queue goes on: the pass was cut by the wall (the arm's own K = 35 rule stops
    C then), or the reference row has no T88 field (an entry failure of HS, a fallback row)."""
    if row.get('stopped') is not None:
        return dict(checkable=False, same=None, why=PASS0_TRUNCATED)
    pin = reference_row(batch, cell.short)
    if pin is not None and pin.get('field_u_sha256') is None and p2.pin_line_status(pin) in p2.FALLBACK_STATUSES:
        return dict(checkable=False, same=None, why=NO_REFERENCE_FIELD)
    theirs = None if pin is None else pin.get('field_u_sha256')
    return dict(checkable=True, same=theirs is not None and row.get('u_sha256') == theirs, why=None, reference_field_u_sha256=theirs)


def c_control_problem(batch, cell, record):
    """After every run of a queue: C's pass 0 (a = 1, Q3h, the frozen Hx24) against the reference's T88 field of the pair, u bit for bit.  A reason when it does not hold or cannot be
    checked; None otherwise.  Revision 2 (the review's minor 9): asked whenever pass 0 is on record, whatever the run's status; a run done without pass 0 cannot be checked.  AC's pass 0
    has the EM inside, A has no frozen pass: no such control for them."""
    if cell.is_repair or cell.arm != 'C' or record is None:
        return None
    passes = record.get('passes') or []
    if not passes:
        return None if record.get('status') != 'done' else f"{cell.run_id}: done without pass 0 on record"
    control = pass0_control(batch, cell, passes[0])
    if not control['checkable']:                        # revision 3 (the follow-up's minors 1, 2): recorded, the queue goes on
        return None
    if not control['same'] or passes[0].get('u_sha256') is None:
        return f"{cell.run_id}: C's pass 0 (a = 1) {passes[0].get('u_sha256')} is not the reference's T88 field {control.get('reference_field_u_sha256')}" + ('; its field is kept: field_pass0.npz' if (passes[0].get('control') or {}).get('field_kept') else '')
    return None


BASES = dict(p2=_instance('p2', p2.P2), p3=_instance('p3', p2.P3))
run = BASES['p2'].run
mc = BASES['p2'].mc
mc0p = BASES['p2'].mc0p
channels = BASES['p2'].channels
fast = BASES['p2'].fast
efrun = BASES['p2'].efrun
READING_RELS = ('experiments/p2/decide_p2.py', 'experiments/p2/p2_core.py', 'experiments/p1/p1_core.py', 'experiments/fl/decide_fx.py', 'experiments/fl/fx_core.py', 'experiments/fl/decide_fl.py', 'experiments/fl/fl_core.py',
                'experiments/fl/decide_fd2.py', 'experiments/fl/fd2_core.py', 'experiments/fl/rp_core.py', 'experiments/fl/rp_fallback_v1.json', 'experiments/fl/rw_sources_v1.json')


# ---- one field run ------------------------------------------------------------------------------------------------------------------------------------
def read_tissue(pin, seg_root=None):
    """The fixed image's tissue layer, as P1 read it (run_p1.tissue_of): run_pair.load_tissue_layer (C03.1-2/3, provenance, hashes, grid), C03.2 simplex with background, C03.3 Omega_B,
    C03.5 smooth and floor.  (brain, ptilde, class names, wm_index, the channel index of p2_core.class_index, record)."""
    from bayesreg.cli.run_pair import load_tissue_layer
    from bayesreg.seg.channels import brain_mask, simplex_with_background, tissue_channels
    P, accepted, record = load_tissue_layer(pin, 'fixed', mc.SEG_ROOT if seg_root is None else seg_root)
    pbar, simplex = simplex_with_background(P)
    del P
    brain, mask = brain_mask(pbar)
    ptilde, smoothing = tissue_channels(pbar)
    del pbar
    names = list(accepted['class_names'])
    index = p2.class_index(names, accepted['wm_index'])
    return brain, ptilde, names, int(accepted['wm_index']), index, dict(record, accepted=dict(accepted), simplex=simplex, brain_mask=mask, smoothing=smoothing, n_brain=int(brain.sum()))


def run_field(o, cell, batch, base):
    """One field run of an arm: the channels as frozen Hx24 (fl_channels.prepare_stacks, once per pass), the tissue layer, one or three passes of T88 (optimize_p2), the checkpoint of the last
    pass at 600, run.json.  The flow, the records and the stops of run_fl.run_field."""
    t0 = perf_counter()
    spec = batch.ARMS[cell.arm]
    fz = core.frozen_api()
    import torch
    out = Path(o.out)
    if (out/'started.json').exists():
        raise SystemExit(f'{out}: this run has started before; a run is never rerun')
    out.mkdir(parents=True, exist_ok=True)
    run.write_json(out/'started.json', dict(run_id=cell.run_id, utc=run.utc(), argv=sys.argv, pid=os.getpid()))
    device_cap = mc0p.limit_device_memory(torch, o.device)
    config_dict = batch.arm_config(cell, spec['sigma_field'])
    config = fz.PreoptConfig(**config_dict)
    keeper = base.CheckpointKeeper(out, (p2.NATIVE,), t0)
    status, reason, u, record, scale, scale_check, channel_sources, digest, final_same = 'done', None, None, None, None, None, None, None, None
    seconds_labels = seconds_channels = seconds_tissue = None
    passes, tissue_record = [], None
    pin = batch.read_images(cell)                         # the firewall view: no label member is decoded, here or below
    try:
        predicted = batch.channels.read_predicted(pin, workers=core.SETUP_WORKERS)
        seconds_labels = perf_counter()-t0
        channel_sources = predicted.record
        digest = efrun.labels_digest(predicted.labels)
        if channel_sources['oracle'] is not False:
            status, reason = 'not_predicted_channels', f"channel sources: oracle {channel_sources['oracle']}, members read {channel_sources['pair_keys_read']}; no optimization"
        else:
            tick = perf_counter()
            brain, ptilde, names, wm_index, index, tissue_record = read_tissue(pin)
            levels = p2.tissue_levels(ptilde, brain, pin.grid, fz.four_level_grids(pin.grid), fz.downsample_array, index, workers=core.SETUP_WORKERS) if spec['tissue'] else None
            if not spec['tissue']:
                del ptilde
            seconds_tissue = perf_counter()-tick
            ratios = None
            n_passes = p2.C_PASSES if spec['tissue'] else 1
            for k in range(n_passes):
                tick = perf_counter()
                stacks, scale = channels.prepare_stacks(spec, pin, predicted, core.load_table(), o.device, workers=core.SETUP_WORKERS)
                seconds_prepare = perf_counter()-tick
                if seconds_channels is None:
                    seconds_channels = perf_counter()-t0
                em = p2.EMWeights(stacks, scale, brain, pin.grid.spacing, o.device, workers=core.SETUP_WORKERS) if spec['em'] else None
                tissue = p2.TissueWeights(levels, ratios) if spec['tissue'] else None
                hooks = None if em is None and ratios is None else p2.Hooks(em, tissue if ratios is not None else None)
                tick = perf_counter()
                u, record = p2.optimize_p2(stacks, pin.grid, config, device=o.device, started=t0, observer=run.vram_guard(torch, o.device, mc.VRAM_LIMIT_BYTES), workers=core.SETUP_WORKERS, record_level_norms=True,
                                           downsample=fast.FastDownsample(), checkpoint=keeper if k == n_passes-1 else None, hooks=hooks)
                del stacks
                scale_check = channels.scale_check_of(record, scale)
                row = dict(pass_index=k, u_sha256=p2.sha_bytes(u), seconds_prepare=seconds_prepare, seconds_optimize=perf_counter()-tick, scale=scale, scale_check=scale_check, stopped=record['stopped'],
                           levels=[dict(level=lv['level'], iterations=lv['iterations'], seconds=lv['seconds'], first=lv['first'], last=lv['last']) for lv in record['levels']], cap_hits=record['cap_hits'], hooks=hooks is not None)
                passes.append(row)
                if k == 0 and cell.arm == 'C':                      # revision 2 (the review's minor 9): the control of pass 0 in the unit too; the field stays when it is not the reference's
                    row['control'] = dict(pass0_control(batch, cell, row), field_kept=False)
                    if row['control']['checkable'] and not row['control']['same']:
                        np.savez(out/'field_pass0.npz', u=u)
                        row['control']['field_kept'] = True
                if em is not None:
                    if record['stopped'] is None:
                        try:                                                                    # sigma_post: a record only, it never fails the run
                            em.m_step(dict(level=3, step=p2.EM_POST_STEP, at='post'), u, used=False)
                        except p2.EstimateFailed as error:
                            row['em_post_error'] = str(error)[:500]
                    row['em'] = em.record()
                    em.host = None
                if tissue is not None:
                    row['tissue'] = tissue.record()
                del em, tissue, hooks
                if record['stopped'] is not None:
                    status, reason = 'time_cap', f"the wall cap {batch.RUN_CAP_SECONDS} s at {record['stopped']} (pass {k})"
                    np.savez(out/'field_stopped.npz', u=u)                                      # the evidence of a run cut short; never read
                    break
                if not scale_check['passed'] or ('em' in row and not (row['em']['gram_check']['passed'] and all(r['passed'] for r in row['em']['native_scale_checks']))):
                    status, reason = 'scale_mismatch', f"pass {k}: ||g_D|| of the native tensors {scale_check}, the EM checks {(row.get('em') or {}).get('gram_check')}"
                    break
                if spec['tissue'] and k < n_passes-1:
                    ratios, row['estimate'] = p2.moment_ratios(u, tuple(pin.grid.spacing), ptilde, brain, names, wm_index, config.j0)
                elif spec['tissue']:
                    try:                                                                        # the ratios of the last pass's field: a record only, never used, never fails the run
                        row['estimate'] = p2.moment_ratios(u, tuple(pin.grid.spacing), ptilde, brain, names, wm_index, config.j0)[1]
                    except p2.EstimateFailed as error:
                        row['estimate'] = dict(error=str(error)[:500], ratios=None)
            if status == 'done':
                last = keeper.records.get(str(p2.NATIVE))
                final_same = None if last is None else p2.sha_bytes(u) == last['u_sha256']
                if final_same is not True:
                    status, reason = 'checkpoint_mismatch', 'the last checkpoint is not the field the last pass returned'
    except run.VramLimit as error:
        status, reason = 'vram_stop', str(error)
    except torch.cuda.OutOfMemoryError as error:
        status, reason = 'cuda_oom', str(error)[:500]
    except FloatingPointError as error:
        status, reason = 'nonfinite', str(error)[:500]
    except p2.EstimateFailed as error:                  # an M step or a moment ratio without a finite value: the arm's estimate failed, an entry failure
        status, reason = 'estimate_failed', str(error)[:500]
    last = keeper.records.get(str(p2.NATIVE))
    cuda = torch.device(o.device).type == 'cuda'
    run.write_json(out/'run.json', dict(
        schema='bayesreg.p2.run.v1', run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, k=spec['k'], native_iterations=batch.steps_of(cell), checkpoint_steps=[p2.NATIVE], channels=spec['channels'], weight=spec['weight'], D_factor=spec['D_factor'],
        bayes=dict(em=spec['em'], tissue=spec['tissue'], passes=p2.C_PASSES if spec['tissue'] else 1, em_schedule=dict(heads=list(p2.EM_SCHEDULE['heads']), native=list(p2.EM_SCHEDULE['native']), post=p2.EM_POST_STEP)),
        segmenter_id=batch.channels.segmenter_id, oracle=None if channel_sources is None else channel_sources['oracle'], status=status, reason=reason, role=batch.ROLE, config=config_dict, scale=scale, scale_check=scale_check, labels_sha256=digest,
        channel_sources=channel_sources, tissue_layer=tissue_record, passes=passes, optimizer=record, checkpoints=keeper.records, seconds_checkpoint_io=sum(r['seconds_io'] for r in keeper.records.values()), final_equals_last_checkpoint=final_same,
        field=None if last is None else dict(file=last['file'], sha256=last['sha256'], u_sha256=last['u_sha256']), taper=core.TAPER, code_sha=o.code_sha, frozen_commit=core.FROZEN_COMMIT, random_seed='none: no random step',
        data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], fixed_subject=pin.fixed_subject, moving_subject=pin.moving_subject, stage=str(core.STAGE), keys_read=list(pin.provenance['keys_read'])),
        env=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if cuda else 'cpu'), device_cap=device_cap,
        device_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None, device_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if cuda else None,
        host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, seconds_labels=seconds_labels, seconds_channels=seconds_channels, seconds_tissue=seconds_tissue, seconds_total=perf_counter()-t0, ended_utc=run.utc()))
    return 0


def use_root_segmenter(batch):
    """A unit's segmenter and reference: the ones its queue wrote into batch.json."""
    return batch.use_segmenter(batch.segmenter_of_root(batch.DEFAULT_ROOT))


def run_one(name, o):
    batch, base = p2.BATCHES[name], BASES[name]
    cell = batch.require_cell(o.arm, o.tier, o.pair)
    if cell.is_repair:
        raise SystemExit(f'{cell.run_id}: a repair arm: repair-one')
    problem = batch.out_problem(o.out, cell)
    if problem:
        raise SystemExit(problem)
    if name == 'p3' and cell.arm != batch.arm_of(batch.DEFAULT_ROOT):
        raise SystemExit(f'{cell.run_id}: P3 runs only the arm P2 chose ({batch.ARM_FILE}: {batch.arm_of(batch.DEFAULT_ROOT)})')
    use_root_segmenter(batch)
    return run_field(o, cell, batch, base)


def repair_one(name, o):
    """run_fl.repair_one with the fuse, then the name of the status (run_fx.repair_one's lines): the .9^k rule did not run, and the record says so."""
    batch, base = p2.BATCHES[name], BASES[name]
    if name == 'p3':
        cell = batch.require_cell(o.arm, o.tier, o.pair)
        if cell.base != batch.arm_of(batch.DEFAULT_ROOT):
            raise SystemExit(f'{cell.run_id}: P3 repairs only the arm P2 chose')
    code = base.repair_one(o)
    path = Path(o.out)/'repair.json'
    record = run.read_json(path)
    run.write_json(path, dict(record, status=fx.CERTIFIED if record['passed'] else fx.NOT_CERTIFIED, status_in_run_fl=record['status'], fallback_rule_run=False, wall_hit=decide_fx.hit_wall(record)))
    return code


# ---- evaluation: run_fl's, and the identity where a failed pair of P3 has no qualified field of configuration C -----------------------------------------
def evaluate_identity(d, pair, labs, truth_pin, ev, certify, config, base):
    """run_rp.evaluate_identity's lines: u = 0 on the pair's grid, its certificate at alpha = 1 (configuration C; J = 1), the metrics with run_fl's function."""
    u = np.zeros(tuple(truth_pin.grid.shape)+(3,))
    tick = perf_counter()
    ladder = run.certify_ladder(u, truth_pin.grid, (1.,), certify, config)
    seconds_certificate = perf_counter()-tick
    metrics = base.field_metrics(u, labs, tuple(truth_pin.grid.spacing), ev)
    run.write_json(d/'identity_metrics.json', dict(schema='bayesreg.p2.identity_metrics.v1', pair=pair, short=core.short_name(pair), oracle=False, role='the second tier of 2c(i) (D75): the identity u = 0, only where a pair failed and its configuration C field does not qualify',
                                                  field='u = 0', certificate=ladder, certified_at_1=ladder['accepted_alpha'] == 1., seconds_certificate=seconds_certificate, metrics=metrics, evaluator=dict(script='experiments/d0/evaluate.py', sha256=core.sha_file(core.ROOT/'experiments/d0/evaluate.py')), evaluated_utc=run.utc()))


def fallback_pairs(root):
    """The pairs of P3's arm that failed (entry or repair): the fallback unit decides their fallback."""
    arm = p2.P3.arm_of(root)
    if arm is None:
        return []
    return [pair for pair in p2.ALL_PAIRS if decision.arm_row('p3', root, arm, pair)['kind'] in decision.FAILURES]


def identity_pairs(root):
    """The pairs whose fallback the fallback unit decided to be the identity (fallback.json): their metrics need the truth, so the evaluation unit writes them."""
    rows = decision.fallback_rows(root, p2.ALL_PAIRS)
    return [pair for pair, f in rows.items() if f is not None and f.get('kind') == decision.IDENTITY]


def fallback_one(root, arm, pair, certify, config, results=None):
    """D75 as C22.5-2 says it (revision 2, the lead on the review's minor 1): the fallback is certified again, with this line's certificate configuration, in this unit, which reads no label
    (load_pair gives the grid from fixed and moving; the field of configuration C; D0's records).  Configuration C's field of d0_v1 when its records qualify it (rp_core.fallback_status) and
    the certificate passes again here; else the identity u = 0, certified here too.  fallback.json, once; an error is kept and not retried."""
    d = Path(root)/decision.FALLBACK_DIR/core.short_name(pair)
    if (d/'fallback.json').exists() or (d/'fallback_error.json').exists():
        return None
    d.mkdir(parents=True, exist_ok=True)
    pin = p2.P3.read_images(p2.P3.field_cell(arm, pair))                 # the firewall view: fixed, moving
    status = rp.fallback_status(pair, results=results)
    record = dict(schema='bayesreg.p2.fallback.v1', pair=pair, short=core.short_name(pair), oracle=False, keys_read=list(pin.provenance['keys_read']), role='D75 / C22.5-2: the fallback certified again in a process that reads no label',
                  a0=dict(qualified=status['qualified'], why=status['why'], field=str(rp.RP.a0_field(pair, results))), certificate_config=dict(rp.CERTIFICATE_C))
    kind, why, a0_metrics = None, None, None
    if status['qualified']:
        with np.load(rp.RP.a0_field(pair, results), allow_pickle=False) as z:     # the field of configuration C (no label member is in that file)
            u = np.ascontiguousarray(z['u'])
        ladder = run.certify_ladder(u, pin.grid, (1.,), certify, config)
        pinned = rp.RP.fallback_pins().get(core.short_name(pair), {}).get('u_sha256')
        record['a0'].update(certificate=ladder, u_sha256=p2.sha_bytes(u), u_sha256_pinned=pinned)
        if ladder['accepted_alpha'] == 1. and p2.sha_bytes(u) == pinned:
            kind, a0_metrics = decision.A0, {m: status['metrics'].get(m) for m in decision.MEASURES}
        else:
            why = f"configuration C's field: the certificate here {ladder['status_at_1']}, its u the pinned one: {p2.sha_bytes(u) == pinned}"
    else:
        why = f"configuration C's field does not qualify: {status['why']}"
    if kind is None:
        ladder = run.certify_ladder(np.zeros(tuple(pin.grid.shape)+(3,)), pin.grid, (1.,), certify, config)
        record['identity'] = dict(certificate=ladder)
        kind = decision.IDENTITY if ladder['accepted_alpha'] == 1. else None
    run.write_json(d/'fallback.json', dict(record, kind=kind, why=why, a0_metrics=a0_metrics, utc=run.utc()))
    return kind


def fallback_unit(o):
    """The unit tag-p3-fallback (CPU): fallback_one for every failed pair of P3's arm.  First core.frozen_api(), as run_fl.evaluate_all."""
    root = Path(o.root)
    core.frozen_api()
    BASES['p3'].enable_jit(root)
    from jd_curl.three_d.feasibility import certify_map
    config = run.certificate_config()
    arm = p2.P3.arm_of(root)
    failures = 0
    for pair in fallback_pairs(root):
        try:
            fallback_one(root, arm, pair, certify_map, config)
        except Exception as error:                      # kept; the others go on; not retried
            d = root/decision.FALLBACK_DIR/core.short_name(pair)
            d.mkdir(parents=True, exist_ok=True)
            run.write_json(d/'fallback_error.json', dict(pair=pair, error=repr(error)[:2000], utc=run.utc()))
            failures += 1
    return 8 if failures else 0


def evaluate(name, o):
    batch, base = p2.BATCHES[name], BASES[name]
    use_root_segmenter(batch)
    code = base.evaluate_all(o)
    if name != 'p3':
        return code
    root = Path(o.root)
    pairs = identity_pairs(root)
    if not pairs:
        return code
    ev = core.load_module('_mc0_evaluate', 'experiments/d0/evaluate.py')
    from jd_curl.three_d.feasibility import certify_map
    config = run.certificate_config()
    failures = 0
    for pair in pairs:
        d = root/decision.ID_DIR/core.short_name(pair)
        if (d/'identity_metrics.json').exists() or (d/'identity_error.json').exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        try:
            truth_pin, labs = batch.read_truth(pair, base.TRUTH_MEMBERS)
            evaluate_identity(d, pair, labs, truth_pin, ev, certify_map, config, base)
        except Exception as error:                      # kept; the others go on; not retried
            run.write_json(d/'identity_error.json', dict(pair=pair, error=repr(error)[:2000], utc=run.utc()))
            failures += 1
    return 8 if failures or code else 0


# ---- the reference ------------------------------------------------------------------------------------------------------------------------------------
def v1_reference(results=None):
    """The reference of unet35_v1_ens from JSON records only (no field is read): P2 rw_v1's repairs of the first 5 pairs, P3 rw_v1's 19 (RHx24, wall_seconds 1200, the fuse of the three
    arms; the lead's ruling ~1:32 am CST 11-06 replaced the brief's fd2_v1 120 s repairs for P2); the T88 field of each pair (fd2_v1 Hx24@600) for C's control."""
    results = Path(fx.RESULTS if results is None else results)
    fd2_root, rw_root = results/fx.FD2_ROOT_NAME, results/fx.RW.ROOT_NAME

    def pins(rdir, fdir, tier):
        repair, metrics, record = run.read_json(rdir/'repair.json'), run.read_json(rdir/'repair_metrics.json'), run.read_json(fdir/'run.json')
        checkpoint = record['checkpoints'][str(p2.NATIVE)]
        return dict(tier=tier, repair_dir=str(rdir), run_id=repair['run_id'], attempt_sha256=repair['attempt']['sha256'], repaired_u_sha256=metrics['repaired_u_sha256'], certified_at_1=metrics['after']['certificate']['accepted_alpha'] == 1.,
                    wall_seconds=repair['config']['wall_seconds'], field_dir=str(fdir), field_sha256=checkpoint['sha256'], field_u_sha256=checkpoint['u_sha256'], field_D_factor=record['scale']['D_factor'])
    rw = {core.short_name(p): pins(fx.RW.run_dir(rw_root, fx.RW.cell_of(p)), fd2.run_dir(fd2_root, fx.RW.fd2_field_cell(p)), fx.RW.tier_of(p)) for p in p2.ALL_PAIRS}
    first = {core.short_name(p): rw[core.short_name(p)] for p in core.PAIRS}
    reading = (run.read_json(rw_root/'analysis/decision_rw.json') or {}).get('reading')
    return dict(schema=p2.REFERENCE_SCHEMA, segmenter_id=p2.DEFAULT_SEGMENTER, channels={}, written_from='JSON records of fd2_v1 and rw_v1 (run_p2.py write-reference); no field read',
                p2=dict(source=f'rw_v1 RHx24 c600: the repairs of frozen Hx24@600 (fd2_v1) on the first 5 pairs with wall_seconds 1200, the fuse of the three arms (RW, reading {reading}), read only; '
                        'the lead\'s ruling about 1:32 am CST 11-06 in place of the brief\'s fd2_v1 120 s repairs', rw_reading=reading, pairs=first),
                p3=dict(source=f'rw_v1 RHx24: the 19 repairs of the same fields with wall_seconds 1200 (RW, reading {reading}), read only', rw_reading=reading, pairs=rw))


# ---- the checks of the inputs -------------------------------------------------------------------------------------------------------------------
def stage_problems(out, pairs, *, stage=None, allowlist=None):
    """run_fx.stage_problems' lines: selection.json, the manifests through the firewall's reader, the subjects on the allowlist, the npy headers of fixed and moving; the pair.npz sha256 the manifest pins."""
    stage = core.STAGE if stage is None else stage
    allowlist = core.ALLOWLIST if allowlist is None else allowlist
    problems = []
    listed = [row['pair'] for row in json.loads((Path(stage)/'selection.json').read_text())['pairs']]
    same = tuple(listed[:len(p2.ALL_PAIRS)]) == p2.ALL_PAIRS
    out(f'selection.json pairs 1..19 == the plan: {same}; {len(listed)} pairs listed')
    if not same:
        problems.append('the pairs of selection.json differ from the plan')
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    from bayesreg.io.pair import _read_manifest
    guard = load_dev_allowlist(allowlist)
    for pair in pairs:
        d = Path(stage)/pair
        _, metadata, manifest = _read_manifest(d/'manifest.json')
        check_pair_subjects(manifest['fixed_subject'], manifest['moving_subject'], guard)
        heads = {m: run.npz_header(d/'pair.npz', m) for m in ('fixed', 'moving')}
        out(f"{core.short_name(pair)}: pair.npz sha256 (manifest) {manifest.get('pair_sha256') or manifest.get('sha256')}; subjects on the allowlist; spacing {metadata['spacing_xyz']}; " + ', '.join(f'{m} {h[0]} {h[1]}' for m, h in heads.items()))
        if len({tuple(h[0]) for h in heads.values()}) != 1:
            problems.append(f'{pair}: image shapes differ')
    return problems


def frozen_problems(out):
    problems = []
    same = core.sha_file(core.FROZEN/'jd_curl/three_d/preopt.py') == core.FROZEN_PREOPT_SHA256
    jit = (core.FROZEN/'jd_curl/three_d/jit_cache.py').is_file()
    out(f'frozen {core.FROZEN}: preopt.py sha256 equal to the pin: {same}; jit_cache.py there: {jit}; colour table sha256 {core.COLOUR_TABLE_SHA256} read: {len(core.load_table())} labels')
    if not same:
        problems.append('the frozen preopt.py differs from the pin')
    if not jit:
        problems.append('the frozen tree has no jit_cache.py')
    return problems


def prediction_problems(out, batch, pairs, *, seg_root=None, allowlist=None):
    """The segmenter's registry committed and clean, D24 admission (ids only), every subject's provenance.json (segmenter, policy, test_labels_seen), the probs36 header and the sha256 its
    provenance pins, the Atropos provenance and its sha256.  No probability, no label."""
    from bayesreg.seg.structure_layer import PROVENANCE_FILE, registry_git_state
    mi = core.make_inputs()
    source = batch.channels.source()
    seg_root = mc.SEG_ROOT if seg_root is None else seg_root
    allowlist = core.ALLOWLIST if allowlist is None else allowlist
    problems = []
    registry = mi._registry_of(source)
    git = registry_git_state(registry)
    out(f"registry {registry}: tracked {git.get('tracked')}, clean {git.get('clean')}")
    if not (git.get('tracked') is True and git.get('clean') is True):
        problems.append(f'the registry {registry} is not committed and clean')
    subjects = sorted({s for pair in pairs for s in pair.split('__')})
    admission = mi.a4_admission(subjects, allowlist, source)
    out(f"D24 admission of the {len(admission['subjects'])} subjects (ids only): folds sha256 {admission['folds_sha256']}")
    for subject in subjects:
        p = Path(source.pred_root)/subject
        missing = [name for name in (PROVENANCE_FILE, 'probs36.npz', 'structure_groups.npz') if not (p/name).is_file()]
        missing += [] if (Path(seg_root)/subject/'provenance.json').is_file() else [f'Atropos {Path(seg_root)/subject}']
        if missing:
            problems.append(f'{subject}: missing {missing}')
            continue
        prov = json.loads((p/PROVENANCE_FILE).read_text())
        head = run.npz_header(p/'probs36.npz', 'p')
        out(f"{subject}: {prov.get('segmenter_id')}, policy {prov.get('policy')}, test_labels_seen {prov.get('test_labels_seen')}, probs36 p {head[0]} {head[1]} sha256 (provenance) {(prov.get('output_sha256') or {}).get('probs36.npz')}; Atropos provenance sha256 {core.sha_file(Path(seg_root)/subject/'provenance.json')}")
        if prov.get('segmenter_id') != batch.channels.segmenter_id or prov.get('policy') != 'all_folds_mean' or prov.get('test_labels_seen') is not False:
            problems.append(f'{subject}: the prediction is not {batch.channels.segmenter_id} all_folds_mean with test_labels_seen false')
    return problems


def reference_problems(out, batch, stage, *, results=None, hs_decision=None):
    """The reference file against the records it names (revision 2, the review's I1 and I2 (b)), for every segmenter: each row through p2_core.reference_pair_problems (this pair, this
    segmenter, oracle false, the frozen Hx24 and RF05 with the fuse, the repair's source the pinned field, the pins); the T88 field's file sha256 hashed here; a row of hx_provenance_rejected
    or an unknown status stops the queue (no output of the reference there).  Fallback rows: HS's decision_hs.json, the one the file names, must hold the same line_status and values.
    unet35_v1_ens: the file is also what v1_reference makes of the records now, key for key."""
    problems = []
    reference = batch.reference
    if batch.channels.is_default:
        same = v1_reference(results) == reference
        out(f'{p2.reference_rel(batch.channels.segmenter_id)}: equal to the records now: {same}; sha256 {core.sha_file(p2.reference_path(batch.channels.segmenter_id))}')
        if not same:
            problems.append('the reference file is not what the records of fd2_v1 and rw_v1 say now')
    out(f"reference of {stage}: {reference[stage]['source']}")
    for pair in (core.PAIRS if stage == 'p2' else p2.ALL_PAIRS):
        pin = reference[stage]['pairs'][core.short_name(pair)]
        found, status = p2.reference_pair_problems(pin, pair, batch.channels.segmenter_id, str(batch.channels.source().pred_root))
        f = Path(pin.get('field_dir') or '/nonexistent')
        checkpoint = ((run.read_json(f/'run.json') or {}).get('checkpoints') or {}).get(str(p2.NATIVE)) or {}
        field_file = f/str(checkpoint.get('file'))
        field_sha = core.sha_file(field_file) if checkpoint and field_file.is_file() else None
        if status == p2.HX_CERTIFIED and field_sha != pin.get('field_sha256'):
            found.append('field_0600.npz: the file sha256 is not the pin')
        out(f"  {core.short_name(pair)}: {status}; {pin.get('repair_dir')}: repair.npz sha256 {pin.get('attempt_sha256')}, repaired u sha256 {pin.get('repaired_u_sha256')}; T88 field {field_file} sha256 {field_sha} (pin {pin.get('field_sha256')}), u sha256 {pin.get('field_u_sha256')}"
            + ('' if status == p2.HX_CERTIFIED else f"; method value {pin.get('method_metrics')}") + f"; problems: {found or 'none'}")
        problems += [f'{stage} reference {core.short_name(pair)}: {x}' for x in found]
    decision_path = p2.HS_DECISION if hs_decision is None else Path(hs_decision)
    decision = run.read_json(decision_path)
    problems += [f'{stage} reference: {x}' for x in p2.hs_reference_problems(reference, decision, core.sha_file(decision_path) if decision is not None else None, batch.channels.segmenter_id)]
    return problems


def hs_problem(segmenter, path=None):
    """Why this segmenter is not the reference configuration at the start (BZ's default 4), or None.  HS's reading from the one file hs_v1/analysis/decision_hs.json (revision 2, the lead
    on the review's I2: no other file of the directory is read): reading 采用新分割器 names the segmenter P2/P3 must use (its segmenter_id); no such file or another reading: unet35_v1_ens."""
    document = run.read_json(p2.HS_DECISION if path is None else Path(path))
    if not (isinstance(document, dict) and document.get('reading') == HS_ADOPT):
        return None if segmenter == p2.DEFAULT_SEGMENTER else f'HS does not read {HS_ADOPT} ({p2.HS_DECISION if path is None else path}): the reference configuration is {p2.DEFAULT_SEGMENTER}, not {segmenter}'
    if document.get('segmenter_id') != segmenter:
        return f"HS reads {HS_ADOPT} with {document.get('segmenter_id')}: the reference configuration is that one, not {segmenter}"
    return None


def input_problems(out, name, *, results=None):
    batch = p2.BATCHES[name]
    pairs = core.PAIRS if name == 'p2' else p2.ALL_PAIRS
    problems = stage_problems(out, pairs)+frozen_problems(out)+prediction_problems(out, batch, pairs)+reference_problems(out, batch, name, results=results)
    hs = hs_problem(batch.channels.segmenter_id)
    out(f"HS ({p2.HS_DECISION}): exists {p2.HS_DECISION.exists()}; the segmenter {batch.channels.segmenter_id} is the reference configuration: {hs is None}")
    if hs:
        problems.append(hs)
    if name == 'p3':
        qualified = {core.short_name(p): rp.fallback_status(p, results=results)['qualified'] for p in p2.ALL_PAIRS}
        out(f"the fallback of 2c(i): configuration C's field qualified by d0_v1's records on {sum(qualified.values())}/19 pairs {[k for k, v in qualified.items() if not v] or ''}")
    return problems


def safe(fn, out=lambda text: None, **kwargs):
    try:
        return fn(out, **kwargs)
    except (Exception, SystemExit) as error:            # noqa: BLE001 - any of them means the inputs are not there
        return [f'{type(error).__name__}: {error}'[:1000]]


# ---- the queues --------------------------------------------------------------------------------------------------------------------------------------
def git_commit(commit):
    """The sha a checkout of `commit` would take, without making one (run_fx.nx_commit)."""
    done = subprocess.run(['git', 'rev-parse', '--verify', f"{commit or 'HEAD'}^{{commit}}"], cwd=core.ROOT, capture_output=True, text=True)
    if done.returncode:
        raise run.Stop(f"git rev-parse {commit or 'HEAD'}: {done.stderr.strip()}")
    return done.stdout.strip()


def reading_code_problem(base, root):
    """Why this process may not write this root's reading, or None: the files that hold the rules (and the reference) must be the batch's commit, byte for byte."""
    batch = run.read_json(Path(root)/'batch.json')
    if batch is None:
        return None
    rels = READING_RELS+(p2.reference_rel(batch.get('segmenter_id', p2.DEFAULT_SEGMENTER)),)
    for rel in rels:
        theirs = base.committed_bytes(batch['code_sha'], rel)
        if theirs is None or theirs != (core.ROOT/rel).read_bytes():
            return f"{core.ROOT/rel} is not {rel} of the batch's commit {batch['code_sha']}: the reading's rules would not be the batch's; run the driver from the main checkout at that commit"
    return None


def write_batch(base, batch, o, root, sha, gate, plan, **extra):
    path = p2.reference_path(batch.channels.segmenter_id)
    existing = run.read_json(Path(root)/'batch.json')
    if existing is not None and (existing.get('segmenter_id') != batch.channels.segmenter_id or (existing.get('reference') or {}).get('sha256') != core.sha_file(path)):     # revision 2 (the review's minor 6)
        raise run.Stop(f"batch.json is of the segmenter {existing.get('segmenter_id')} and the reference {(existing.get('reference') or {}).get('sha256')}; this start has {batch.channels.segmenter_id} and {core.sha_file(path)}")
    base.write_batch(root, 'batch.json', dict(plan=plan, segmenter_id=batch.channels.segmenter_id, reference=dict(file=p2.reference_rel(batch.channels.segmenter_id), sha256=core.sha_file(path)), gap_seconds=o.gap_seconds,
                                              rules=dict(order=list(p2.ORDER), em_schedule=dict(heads=list(p2.EM_SCHEDULE['heads']), native=list(p2.EM_SCHEDULE['native']), post=p2.EM_POST_STEP), c_passes=p2.C_PASSES, D_factor=p2.D_FACTOR,
                                                         p2_lines=dict(n=p2.P2_N, sdlogj_max=p2.SDLOGJ_MAX, delta_min=p2.P2_DELTA_MIN), p3_lines=dict(lo=p2.P3_LO, mean=p2.P3_MEAN, sdlogj_max=p2.SDLOGJ_MAX, superior_lo=p2.SUPERIOR_LO), boot=dict(n=p2.BOOT_N, seed=p2.BOOT_SEED)),
                                              caps=dict(run_seconds=batch.RUN_CAP_SECONDS, run_units_seconds=base.RUN_UNITS_CAP_SECONDS, device_gib=mc.VRAM_LIMIT_BYTES/2**30, memory_max=base.MEMORY_MAX, nice=base.NICE, repair_wall_seconds=fx.FUSE_SECONDS, repair_runtime_seconds=fx.REPAIR_RUNTIME_SECONDS),
                                              gate=dict(own_units=list(gate.own_units), load_limit=gate.load_limit, gpu_free_mib=gate.gpu_free_mib, mem_available_gib=gate.mem_available_gib, home_floor_gb=run.HOME_FLOOR_GB, host_floor_gib=gate.host_floor_gib),
                                              evaluation=dict(numba_threads=base.EVAL_THREADS, cpu_quota_percent=base.EVAL_QUOTA), **extra), sha)
    problem = reading_code_problem(base, root)
    if problem:
        raise run.Stop(problem)


def repairs_of_done_fields(batch, root, repairs, log):
    """The repairs whose field is done and evaluated (run_fx.repairs_of_done_fields, with the evaluation: a repair reads the field's metrics).  The reading counts the others as failed or not read."""
    out = []
    for cell in repairs:
        field = batch.Cell(cell.base, batch.tier_of(cell.pair), cell.pair)
        d = batch.run_dir(root, field)
        status = (run.read_json(d/'run.json') or {}).get('status')
        if status == 'done' and (d/f'metrics_{p2.NATIVE:04d}.json').is_file():
            out.append(cell)
        else:
            log(f"skip {cell.run_id}: its field is {status}{'' if status != 'done' else ', not evaluated'}; nothing to repair")
    return out


def fallback_argv(code, root):
    """The unit tag-p3-fallback: CPU, numba 8, no label read (revision 2)."""
    base = BASES['p3']
    return base.unit_argv(FALLBACK_UNIT, code, ['fallback', '--root', str(root)], log=Path(root)/'fallback.log', gpu=False, numba=base.EVAL_THREADS, quota=base.EVAL_QUOTA, runtime=run.EVAL_RUNTIME_MAX)


def finish(root, log, result, rcs, stop):
    for x in result.get('needs_lead_rejudgment') or []:          # revision 2 (the review's minor 2): at the end of the queue's log too
        log(f"要总负责人改判: {x.get('arm', '')} {x['pair']}: {x['why']}"[:2000])
    log(f"reading: {result['reading']}.  Handed to the lead: no next step starts here")
    failed = {unit: rc for unit, rc in rcs.items() if rc != 0}
    if failed:
        raise run.Stop(f'an evaluation unit failed: {failed}; the reading is written, a pair without metrics counts as not read')
    if stop is not None:
        raise run.Stop(f'a phase stopped: {stop}')
    return 0


def p2_body(o, system, root, log):
    base, batch = BASES['p2'], p2.P2
    base.keep_old_stops(root)
    gate = replace(base.GATE, host_floor_gib=o.min_host_gib)
    batch.use_segmenter(o.segmenter)
    sha = git_commit(o.commit)
    problems = safe(input_problems, name='p2')
    if problems:
        raise run.Stop(f'the inputs, before any run: {problems}; nothing checked out')
    code, sha = system.code_checkout(root, sha)
    write_batch(base, batch, o, root, sha, gate, [c.run_id for c in batch.plan_fields()+batch.plan_repairs()])
    log(f'start: {len(batch.plan_fields())} fields of {list(p2.ORDER)} on the first 5 pairs, then their repairs; segmenter {batch.channels.segmenter_id}; /mnt/c floor {o.min_host_gib:g} GiB')
    rcs, stop = {}, None
    try:
        base.phase_and_evaluation(o, system, root, log, code, sha, batch.plan_fields(), gate, base.EVAL_UNIT, rcs)
        base.evaluated(rcs, base.EVAL_UNIT, 'the repairs are')
        base.phase_and_evaluation(o, system, root, log, code, sha, repairs_of_done_fields(batch, root, batch.plan_repairs(), log), gate, base.EVAL_R1_UNIT, rcs)
    except run.Stop as error:
        stop = str(error)
    state = base.ledger(root)
    result = decision.write_p2(root, stops=state['arms_stopped'])
    run.write_json(root/'queue_end.json', dict(stop=stop, eval_rc=rcs, reading=result['reading'], candidate=result.get('candidate'), reason=result.get('reason'), ledger=state, utc=run.utc()))
    return finish(root, log, result, rcs, stop)


def p3_body(o, system, root, log):
    base, batch = BASES['p3'], p2.P3
    base.keep_old_stops(root)
    gate = replace(base.GATE, host_floor_gib=o.min_host_gib)
    problem = reading_code_problem(BASES['p2'], p2.P2.DEFAULT_ROOT)      # P2's reading is computed again below with this process's files: they must be P2's commit
    if problem:
        raise run.Stop(f'{problem}; nothing checked out')
    choice, why = decision.p3_arm_choice(p2.P2.DEFAULT_ROOT)
    if choice is None:
        raise run.Stop(f'{why}; nothing checked out')
    sha = git_commit(o.commit)
    if sha != choice['p2_code_sha']:
        raise run.Stop(f"P3 runs at P2's commit {choice['p2_code_sha']} (the estimation rules fixed before P3), not at {sha}; nothing checked out")
    if o.segmenter is not None and o.segmenter != choice['segmenter_id']:
        raise run.Stop(f"P2 ran with {choice['segmenter_id']}, not {o.segmenter}")
    batch.use_segmenter(choice['segmenter_id'])
    problems = safe(input_problems, name='p3')
    if problems:
        raise run.Stop(f'the inputs, before any run: {problems}; nothing checked out')
    code, sha = system.code_checkout(root, sha)
    choice, why = decision.ensure_p3_arm(root, p2.P2.DEFAULT_ROOT)
    if choice is None:
        raise run.Stop(why)
    arm = choice['arm']
    write_batch(base, batch, o, root, sha, gate, [c.run_id for c in batch.plan_fields_of(arm)+batch.plan_repairs_of(arm)], p2_choice=choice)
    base.log_choice(log, root, batch.ARM_FILE, f'P3: the arm {arm}, from P2\'s reading')
    rcs, stop = {}, None
    try:
        base.phase_and_evaluation(o, system, root, log, code, sha, batch.plan_fields_of(arm), gate, base.EVAL_UNIT, rcs)
        base.evaluated(rcs, base.EVAL_UNIT, 'the repairs are')
        base.phase_and_evaluation(o, system, root, log, code, sha, repairs_of_done_fields(batch, root, batch.plan_repairs_of(arm), log), gate, base.EVAL_R1_UNIT, rcs)
        base.evaluated(rcs, base.EVAL_R1_UNIT, 'the fallbacks are')
        failed = fallback_pairs(root)
        if failed:                                      # revision 2 (D75, C22.5-2): the fallbacks certified again in a unit that reads no label, then the identity's metrics where it is one
            run.wait_for_gate(system, root, False, o.wait_minutes, log, gate)
            log(f"launch {FALLBACK_UNIT}: the failed pairs {[core.short_name(p) for p in failed]}")
            rcs[FALLBACK_UNIT] = system.run(fallback_argv(code, root), root/f'{FALLBACK_UNIT}.systemd_run.log')
            base.evaluated(rcs, FALLBACK_UNIT, "the identity's metrics are")
            if identity_pairs(root):
                run.wait_for_gate(system, root, False, o.wait_minutes, log, gate)
                log(f'launch {EVAL_ID_UNIT}')
                rcs[EVAL_ID_UNIT] = system.run(base.eval_argv(code, root, EVAL_ID_UNIT), root/f'{EVAL_ID_UNIT}.systemd_run.log')
    except run.Stop as error:
        stop = str(error)
    state = base.ledger(root)
    result = decision.write_p3(root, stops=state['arms_stopped'])
    run.write_json(root/'queue_end.json', dict(stop=stop, eval_rc=rcs, reading=result['reading'], superior=result.get('superior'), reason=result.get('reason'), ledger=state, utc=run.utc()))
    return finish(root, log, result, rcs, stop)


BODIES = dict(p2=p2_body, p3=p3_body)


def queue(name, o, system=None):
    batch = p2.BATCHES[name]
    problem = batch.root_problem(o.root)
    if problem:                                         # before the lock file: nothing is created or touched
        print(problem, file=sys.stderr)
        return 2

    def guarded(o, system, root, log):
        try:
            return BODIES[name](o, system, root, log)
        except SystemExit as refused:                   # SelectionChanged among them; a wrong mark met by the reading: a STOP with its record
            raise run.Stop(str(refused))
    return run.queue(o, system, body=guarded, prefix=name)


def queue_launch(name, root=None, *, extra=()):
    """The lead's start string of a queue."""
    batch, base = p2.BATCHES[name], BASES[name]
    return run.check_launch(['systemd-run', '--user', f'--unit={base.QUEUE_UNIT}', '--collect', f'--working-directory={core.MAIN_REPO}', f'--property=MemoryMax={base.MEMORY_MAX}', f'--nice={base.NICE}', '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1',
                             run.PY, '-B', SCRIPT, name, 'queue', '--root', str(batch.DEFAULT_ROOT if root is None else root), '--gap-seconds', str(base.GAP_SECONDS), '--min-host-gib', f'{base.DEFAULT_MIN_HOST_GIB:g}', *extra])


# ---- estimates ------------------------------------------------------------------------------------------------------------------------------------
def estimates(results=None):
    """〔估算〕from fd2_v1's records of Hx24@600 (the task clock, the labels, the channels, the T88 levels), rw_v1's repairs and the evaluations' seconds; the constants above for what nobody has run."""
    results = Path(fx.RESULTS if results is None else results)
    fd2_root, rw_root = results/fx.FD2_ROOT_NAME, results/fx.RW.ROOT_NAME
    labels, prepare, t88, evaluation, repairs = [], [], [], [], []
    for pair in p2.ALL_PAIRS:
        d = fd2.run_dir(fd2_root, fx.RW.fd2_field_cell(pair))
        record, metrics = run.read_json(d/'run.json') or {}, run.read_json(d/f'metrics_{p2.NATIVE:04d}.json') or {}
        if record.get('seconds_labels') is not None:
            labels.append(record['seconds_labels'])
            prepare.append(record['seconds_channels']-record['seconds_labels'])
            t88.append(sum(level['seconds'] for level in record['optimizer']['levels']))
        if metrics.get('seconds_certificate') is not None:
            evaluation.append(metrics['seconds_certificate']+metrics['seconds_metrics'])
        r = run.read_json(fx.RW.run_dir(rw_root, fx.RW.cell_of(pair))/'repair.json') or {}
        if r.get('seconds_total') is not None:
            repairs.append(r['seconds_total'])
    m = (lambda v: float(np.mean(v)) if v else None)
    L, P, T, E, R = m(labels), m(prepare), m(t88), m(evaluation), m(repairs)
    if None in (L, P, T, E, R):
        return dict(complete=False)
    one_pass = P+T
    arm = dict(A=L+one_pass+EM_OVERHEAD_SECONDS, C=L+TISSUE_SECONDS+p2.C_PASSES*one_pass, AC=L+TISSUE_SECONDS+p2.C_PASSES*(one_pass+EM_OVERHEAD_SECONDS))
    gap = BASES['p2'].GAP_SECONDS
    per_pair_rest = R+gap+2*E
    p2_seconds = sum(5*(arm[a]+gap+per_pair_rest) for a in p2.ORDER)+2*GATE_WAIT_SECONDS
    p3_seconds = {a: 19*(arm[a]+gap+per_pair_rest)+2*GATE_WAIT_SECONDS for a in p2.ORDER}
    return dict(complete=True, labels=L, prepare=P, t88=T, evaluation=E, repair=R, field_unit=arm, p2_seconds=p2_seconds, p3_seconds=p3_seconds, field_gib=166e6*(15+19)/2**30)


def _minutes(seconds):
    return '—' if seconds is None else f'{seconds/60:.0f} min'


# ---- dry run --------------------------------------------------------------------------------------------------------------------------------------
def dry_run(name, segmenter=None, *, out=print, root=None, results=None):
    """The plan, the unit strings, the sha256 of the inputs and of the reference, the estimates, the checks.  No truth label, no optimization, torch not imported, nothing written."""
    batch, base = p2.BATCHES[name], BASES[name]
    root = batch.DEFAULT_ROOT if root is None else root
    code = Path(root)/'code'
    out(f'{name.upper()} dry run ({run.utc()}): no optimization, no truth label, nothing written; real files read up to the npy headers of fixed, moving and p, JSON records, the reference T88 fields for their sha256')
    if name == 'p3':
        choice, why = decision.p3_arm_choice(p2.P2.DEFAULT_ROOT)
        out(f"P2's choice: {choice or why}（等 P2 不算问题）")
        segmenter = (choice or {}).get('segmenter_id') or segmenter
        arms = [choice['arm']] if choice else list(p2.ORDER)
    segmenter = segmenter or p2.DEFAULT_SEGMENTER
    try:
        batch.use_segmenter(segmenter)
    except SystemExit as missing:
        out(f'no reference for {segmenter}: {missing}')
        out(f'problems: {[str(missing)]}')
        return 8
    gate = base.GATE
    out(f'root {root}; exists: {Path(root).exists()}; segmenter {segmenter}; reference {p2.reference_rel(segmenter)}; cap per field run {batch.RUN_CAP_SECONDS} s (all passes), unit RuntimeMaxSec {run.RUN_RUNTIME_MAX}; repair wall_seconds {fx.FUSE_SECONDS:g} s, RuntimeMaxSec {fx.REPAIR_RUNTIME_SECONDS}; device {mc.VRAM_LIMIT_BYTES/2**30} GiB; MemoryMax {base.MEMORY_MAX}, nice {base.NICE}; {base.GAP_SECONDS} s between runs')
    out(f'gate: own units {gate.own_units}, load < {gate.load_limit}, GPU free >= {gate.gpu_free_mib} MiB, MemAvailable >= {gate.mem_available_gib} GiB, /home >= {run.HOME_FLOOR_GB} GB, /mnt/c >= {gate.host_floor_gib} GiB; closed while any other tag-* or seq-vp-* unit is active (UF, HS)')
    out(f"rules: order {list(p2.ORDER)}; EM M steps at the level heads {list(p2.EM_SCHEDULE['heads'])} and native steps {list(p2.EM_SCHEDULE['native'])} (post {p2.EM_POST_STEP}, not used); C passes {p2.C_PASSES}; D factor {p2.D_FACTOR}")
    if name == 'p2':
        cells = [(c, 'fields') for c in batch.plan_fields()]+[(c, 'repairs') for c in batch.plan_repairs()]
    else:
        cells = [(c, f'fields {a}') for a in arms for c in batch.plan_fields_of(a)]+[(c, f'repairs {a}') for a in arms for c in batch.plan_repairs_of(a)]
    for i, (c, phase) in enumerate(cells, 1):
        out(f'{i:3d} | {phase} | {c.arm} | {c.short} | {c.unit} | {c.run_id}')
    e = estimates(results)
    if e['complete']:
        out(f"estimates〔估算, from fd2_v1's Hx24 records and rw_v1's repairs; nobody has run these arms〕: a field unit A {e['field_unit']['A']:.0f} s, C {e['field_unit']['C']:.0f} s, AC {e['field_unit']['AC']:.0f} s (T88 {e['t88']:.0f} s a pass, channels {e['prepare']:.0f} s, labels {e['labels']:.0f} s); repair {e['repair']:.0f} s, evaluation {e['evaluation']:.0f} s a field")
        out(f"estimates: P2 {_minutes(e['p2_seconds'])}; P3 A {_minutes(e['p3_seconds']['A'])}, C {_minutes(e['p3_seconds']['C'])}, AC {_minutes(e['p3_seconds']['AC'])}; field files about {e['field_gib']:.1f} GiB for P2 and P3 together (only the last pass is stored)")
    problems = safe(input_problems, out, name=name, results=results)
    out(f'/home free {shutil.disk_usage("/home").free/2**30:.0f} GB; /mnt/c free {shutil.disk_usage("/mnt/c").free/2**30:.0f} GiB')
    out('--- start strings (the queue: the lead, after UF and HS; the others: the queue, one per run; the code at $ROOT/code)')
    out(shlex.join(queue_launch(name, root, extra=('--segmenter', segmenter) if name == 'p2' else ('--commit', "<P2's batch.json code_sha>"))))
    for c, _ in cells:
        out(shlex.join(base.run_argv(c, code, root, '<sha>')))
    out(shlex.join(base.eval_argv(code, root, base.EVAL_UNIT)))
    out(shlex.join(base.eval_argv(code, root, base.EVAL_R1_UNIT)))
    if name == 'p3':
        out('--- only when a pair failed: the fallback certified again (no label read), then the identity\'s metrics where the fallback is the identity')
        out(shlex.join(fallback_argv(code, root)))
        out(shlex.join(base.eval_argv(code, root, EVAL_ID_UNIT)))
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


# ---- the entries -----------------------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['write-reference']:
        path = p2.reference_path(p2.DEFAULT_SEGMENTER)
        if path.exists():
            raise SystemExit(f'{path} is there: a reference is written once')
        path.write_text(json.dumps(v1_reference(), indent=1, ensure_ascii=False)+'\n')
        return 0
    if len(argv) < 2 or argv[0] not in p2.BATCHES:
        print('usage: run_p2.py p2|p3 --dry-run | queue | run-one | repair-one | evaluate | decide;  run_p2.py write-reference', file=sys.stderr)
        return 2
    name, rest = argv[0], argv[1:]
    batch = p2.BATCHES[name]
    if rest[:1] == ['--dry-run']:
        d = argparse.ArgumentParser(description='P2/P3: the dry run')
        d.add_argument('--dry-run', action='store_true')
        d.add_argument('--segmenter', default=None)
        return dry_run(name, d.parse_args(rest).segmenter)
    p = argparse.ArgumentParser(description='P2/P3: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(batch.DEFAULT_ROOT))
    q.add_argument('--commit', default=None, required=(name == 'p3'))
    q.add_argument('--segmenter', default=None, required=(name == 'p2'))
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--gap-seconds', type=int, default=BASES[name].GAP_SECONDS)
    q.add_argument('--min-host-gib', type=float, default=BASES[name].DEFAULT_MIN_HOST_GIB)
    for command in ('run-one', 'repair-one'):
        r = sub.add_parser(command)
        r.add_argument('--arm', required=True)
        r.add_argument('--tier', required=True)
        r.add_argument('--pair', required=True)
        r.add_argument('--out', required=True)
        r.add_argument('--device', default='cuda')
        r.add_argument('--code-sha', default='n/a')
    for command in ('evaluate', 'decide')+(('fallback',) if name == 'p3' else ()):
        sub.add_parser(command).add_argument('--root', default=str(batch.DEFAULT_ROOT))
    o = p.parse_args(rest)
    if o.command == 'queue':
        return queue(name, o)
    if o.command == 'run-one':
        return run_one(name, o)
    if o.command == 'repair-one':
        return repair_one(name, o)
    problem = batch.root_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if o.command == 'evaluate':
        return evaluate(name, o)
    if o.command == 'fallback':
        return fallback_unit(o)
    problem = reading_code_problem(BASES[name], o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 8
    stops = BASES[name].ledger(o.root)['arms_stopped']
    (decision.write_p2 if name == 'p2' else decision.write_p3)(o.root, stops=stops)
    return 0


if __name__ == '__main__':
    sys.exit(main())
