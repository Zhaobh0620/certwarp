"""HS (runs/20261106-hs/任务书.md, plan.md): the driver.  The candidate of UF registered, then the frozen Hx24 (Hx24-RF05-F1200) on its channels, 19 development pairs, the fallbacks of D74/D75, the reading.  Not F_B: no certified descent, only the final field passes the certificate; development pairs only, in no result table.
It runs only after UF's reading says worth_hx24 (seg/unet35_v4_full/analysis/decision_uf.json, written once by UF); before that every queue stops with 等 UF, and the dry run prints the plan of both candidates.
Batch: hs_core.py.  The registration and the predictions: hs_seg.py.  The reading: decide_hs.py.  One field run, one repair, one evaluation: run_fl.py's own functions on another instance of it, with RW's fuse and NX's way of reading a registered segmenter (run_fx.py); fx_core, run_fx, run_fl, decide_fx, decide_fl, rp_core, run_rp are not touched.

    python -B experiments/hs/run_hs.py --dry-run
        both candidates' plans, what HS waits for, the checks that need nothing of UF, the estimates, the start strings.  No optimization, no truth label, torch not imported, nothing written; of UF's root only whether its files are there.
        Once the seg phase is done and the registry committed, also the fields queue's input check (fields_input_problems), its problems counted (HSF).
    run_hs.py drift-approval --review <path> [--root ROOT]
        HSF: prints the runs/20261106-hs/drift_approval.json the lead would commit for HEAD (the forbidden changes since batch_seg.json's commit that may be approved, each with the sha256 of its diff); writes nothing.
    run_hs.py queue --phase seg [--commit SHA] [--wait-minutes 30]
        the lead's start, after UF's reading.  Checkout $ROOT/seg/code; tag-hs-infer (GPU: stage the registry, the 20 predictions, the reader on each) then tag-hs-segcheck (CPU: the truth of the 20 subjects against eval_uf.json's rows of the candidate).  Then it stops: the independent review comes next.
    run_hs.py register                      by hand, after the review: configs/bayesreg/segmenters_hs_v1.json with the staged bytes; the lead commits it.
    run_hs.py predict                       the 〔模型〕expectation, printed (the fields queue writes it to analysis/prediction_hs.json before the first field)
    run_hs.py write-reference               by hand, after HS reads 采用新分割器: experiments/p2/reference_<segmenter_id>.json for P2/P3 (hs_reference.py; run_p2.hs_problem reads analysis/decision_hs.json itself); the lead commits it
    run_hs.py queue --phase fields [--commit SHA] [--wait-minutes 30] [--gap-seconds 5] [--min-host-gib 15]
        after the commit of the registration.  Checkout $ROOT/code; the 19 fields, tag-hs-eval, the repairs of the fields that are done, tag-hs-eval-r, tag-hs-fallback (CPU, no label: the third certificate where the two disagree, configuration C's field certified again, else the identity) and tag-hs-eval-id (the identity's metrics) when a pair needs them; the reading.
    run_hs.py run-one|repair-one --arm A --tier T --pair P --out DIR [--device cuda] [--code-sha SHA]      one unit of the fields queue
    run_hs.py infer-one|segcheck --code-sha SHA                                                             one unit of the seg queue
    run_hs.py evaluate|fallback|id-evaluate|decide [--root ROOT]

Writes only under hs_v1 and seg/<the candidate's segmenter id> (and, by `register`, the registry file); seg/unet35_v4_full, nnunet_v1, rw_v1, fd2_v1, d0_v1 are only read.  A run directory that exists is never rerun, failures are kept.  Exit codes: 0 done, 2 usage, another driver on the root, another root, 8 STOP or a failed check.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _p in (EXPERIMENTS/'unv2', EXPERIMENTS/'uf', EXPERIMENTS/'nn', EXPERIMENTS/'mc0', EXPERIMENTS/'ef', EXPERIMENTS/'fl', HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import run_fx as fxrun  # noqa: E402  (experiments/fl on the path; its instances of run_fl stay its own)
import decide_hs as decision  # noqa: E402
import hs_core as hc  # noqa: E402
import hs_seg  # noqa: E402
import rp_core as rp  # noqa: E402
from hs_core import ALL_PAIRS, HS  # noqa: E402

fx, core, fl, mc, nc = fxrun.fx, fxrun.core, fxrun.fl, fxrun.mc, fxrun.nc
run = fxrun.run                                 # run_mc0
SCRIPT = 'experiments/hs/run_hs.py'
QUEUE_UNIT = 'tag-hs-queue'
INFER_UNIT, SEGCHECK_UNIT, FALLBACK_UNIT, ID_EVAL_UNIT = 'tag-hs-infer', 'tag-hs-segcheck', 'tag-hs-fallback', 'tag-hs-eval-id'
SEG_MEMORY_MAX, INFER_RUNTIME, SEGCHECK_RUNTIME, FALLBACK_RUNTIME = '8G', 7200, 3600, 3*3600     # the fallback unit: up to 19 pairs x 3 certificates of about 74 s each (revision 1, review minor 14)
SEG_GPU_FREE_MIB = 21*1024                      # the seg phase's GPU gate: the inference caps itself at 20 GiB (hs_seg.GPU_CAP_GIB), UF's gate is 21 GiB (revision 1, review minor 14)
INFER_LAUNCHED = 'infer_launched.json'          # the seg queue writes it before the unit: a launch that left no infer_record.json is never launched again
SEGCHECK_LAUNCHED, SEGCHECK_REREADS, SEGCHECK_STARTED = 'segcheck_launched.json', 'segcheck_rereads.json', 'segcheck_started.json'     # the same for the control, which reads the truth: again only with the lead's words, recorded (revision 1, review minor 3)
CURRENT = {}                                    # the run directory of the field run in this process (HSFast writes the kind of a refusal there)


def _instance():
    """run_fl once more under another name, its module names pointed at HS: run_fx._instance's lines, with this entry's script and queue unit; then RW's fuse, NX's reader of a registered segmenter, every one of the 19 pairs in the plan."""
    spec = importlib.util.spec_from_file_location('_hs_run_fl', EXPERIMENTS/'fl'/'run_fl.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.fl, module.decision, module.SCRIPT, module.QUEUE_UNIT = HS, decision, SCRIPT, QUEUE_UNIT
    module.EVAL_UNIT, module.EVAL_R1_UNIT, module.EVAL_X_UNIT, module.EVAL_R2_UNIT, module.EVAL_REST_UNIT, module.EVAL_REST_R_UNIT = 'tag-hs-eval', 'tag-hs-eval-r', 'tag-hs-eval-x', 'tag-hs-eval-r2', 'tag-hs-eval-rest', 'tag-hs-eval-rest-r'
    module.eval_argv.__defaults__ = (module.EVAL_UNIT,)
    module.GATE = replace(module.GATE, own_units=(f'{QUEUE_UNIT}.service',))
    base = module.run

    def unit_argv(unit, code, inner, *, log, gpu=True, numba=module.RUN_THREADS, quota=800, runtime=base.RUN_RUNTIME_MAX):
        return base.unit_argv(unit, code, list(inner), log=log, gpu=gpu, numba=numba, quota=quota, runtime=runtime, script=SCRIPT, memory_max=module.MEMORY_MAX, nice=module.NICE)

    def run_argv(cell, code, root, sha='n/a'):
        d = HS.run_dir(root, cell)
        command = 'repair-one' if cell.is_repair else 'run-one'
        return unit_argv(cell.unit, code, [command, '--arm', cell.arm, '--tier', cell.tier, '--pair', cell.pair, '--out', str(d), '--code-sha', sha], log=d/'log.txt', numba=module.REPAIR_THREADS if cell.is_repair else module.RUN_THREADS, quota=module.REPAIR_QUOTA, runtime=fx.REPAIR_RUNTIME_SECONDS if cell.is_repair else base.RUN_RUNTIME_MAX)
    module.unit_argv, module.run_argv = unit_argv, run_argv
    module.repair = fxrun.RepairWithFuse(module.repair, fx.FUSE_SECONDS)
    module.repair_allowed = lambda cell, root: True     # all 19 pairs are in the plan; require_cell refuses the rest
    return module


BASE = _instance()
READING_FILES = tuple((EXPERIMENTS.parent/rel, rel) for rel in ('experiments/hs/decide_hs.py', 'experiments/hs/hs_core.py', 'experiments/fl/decide_fx.py', 'experiments/fl/fx_core.py', 'experiments/fl/decide_fl.py', 'experiments/fl/fl_core.py', 'experiments/fl/rp_core.py', rp.FALLBACK_REL))


# ---- the source of the channels: the registered candidate -----------------------------------------------------------------------------------------
def hs_source():
    """The predictions of the candidate under the structure layer's checks: a registered segmenter, its registry committed and clean (hard rule 7)."""
    mi = core.make_inputs()
    return mi.UNetSource(pred_root=HS.pred_root(), models_root=HS.UF_OUT, image_root=nc.DATA_ROOT, segmenter_id=HS.segmenter_id(), registry=HS.registry_path(), gate=dict(rule='runs/20261106-hs/任务书.md: UF reads worth_hx24; the candidate is its primary arm'))


def hs_oracle_mark(keys_read, sides, segmenter_id=None):
    """mc0p_core.oracle_mark with the candidate's segmenter id in the place of unet35_v1_ens (run_fx.nx_oracle_mark's rule): oracle is False only when pair.npz gave both images and no other member, and both predictions are the candidate's, all_folds_mean."""
    segmenter_id = HS.segmenter_id() if segmenter_id is None else segmenter_id
    other = sorted(set(keys_read)-set(mc.IMAGE_KEYS))
    images = {'fixed', 'moving'} <= set(keys_read)
    predictions = set(sides) == {'fixed', 'moving'} and all(s.get('segmenter_id') == segmenter_id and s.get('policy') == mc.POLICY for s in sides.values())
    return dict(oracle=not (images and predictions and not other), truth_labels_read=bool(other))


def hs_oracle_mark_checked(cell, record):
    """run_fl.oracle_mark_checked for HS: the mark again from the run's channel_sources; ValueError otherwise, and the field is not read."""
    sources = record.get('channel_sources') or {}
    mark = hs_oracle_mark(sources.get('pair_keys_read') or [], sources.get('sides') or {})
    if record.get('oracle') is not False or mark['oracle'] is not False:
        raise ValueError(f"run.json oracle {record.get('oracle')}; computed from its channel_sources: {mark}")
    return False


ALIGNMENT_QC_FAILURE = re.compile(r'^EF \S+ <- \S+:(fixed|moving): alignment QC (None|[-+0-9.eE]+) < [-+0-9.eE]+ \(protocol §12\.4 item 5\)$')
# exactly the message of experiments/ef/fast_channels.py read_side, the one refusal of a QC that was computed and failed (a test reads that line of the source)


def refusal_kind(message):
    """alignment_qc only for the failed QC itself (fast_channels.read_side: `EF <fixed> <- <moving>:<side>: alignment QC <dice> < <threshold> (protocol §12.4 item 5)`; D74-2: the fallback).  Every other refusal, also those whose
    text names the alignment QC (the Atropos posteriors missing or refused, no seg_root, arrays of other shapes, a grouping without cerebral_wm: make_inputs.py, fast_channels.py), is an input error: provenance_rejected (D74-4; revision 1, review important 1)."""
    return 'alignment_qc' if ALIGNMENT_QC_FAILURE.match(message) else 'provenance_rejected'


class HSFast:
    """What run_fl.run_field asks of fast_channels: read_predicted on the candidate's source, its mark computed for that segmenter; a refusal is recorded with its kind beside run.json, then raised as before."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def read_predicted(self, pin, *, workers=1):
        try:
            predicted = self._module.read_predicted(pin, unet=hs_source(), workers=workers)
        except SystemExit as refused:
            out = CURRENT.get('out')
            if out is not None:
                Path(out).mkdir(parents=True, exist_ok=True)
                run.write_json(Path(out)/hc.REFUSAL_FILE, dict(kind=refusal_kind(str(refused)), message=str(refused)[:2000], utc=run.utc(), rule='alignment_qc: an entry failure, the fallback (D74-2); provenance_rejected: an input error, no fallback (D74-4)'))
            raise
        predicted.record.update(hs_oracle_mark(predicted.record['pair_keys_read'], predicted.record['sides']))
        return predicted


BASE.fast = HSFast(BASE.fast)
BASE.oracle_mark_checked = hs_oracle_mark_checked


ENTRY_QC = 'entry_alignment_qc'                 # the status of a field run whose channels the alignment QC refused (D74-2: the fallback)


def run_one(o):
    """run_fl.run_one with the candidate's channels.  A refusal of the alignment QC is an entry failure (D74-2): run.json is written with that status (no channel read, no field), the unit ends with 0, so the queue goes on and the pair goes to the fallback;
    run_fl's guard 'a K = 35 run that failed without a record stops the arm' is for a crash that the next pair would repeat.  Any other refusal (an input error, D74-4) is raised as before: no record, the arm stops, the reading is 不可判.
    load_pair's refusals (the firewall, the admission, the hashes: HS.read_images, called by run_fl.run_field before the channels) are recorded the same way as provenance_rejected, on any pair, the last one included (revision 1, review minor 9)."""
    cell = HS.require_cell(o.arm, o.tier, o.pair)
    CURRENT['out'] = o.out
    restore = guard_read_images(cell, o.out)
    try:
        return BASE.run_one(o)
    except SystemExit as refused:
        out = Path(o.out)
        refusal = run.read_json(out/hc.REFUSAL_FILE)
        if refusal is None or refusal.get('kind') != 'alignment_qc' or (out/'run.json').exists():
            raise
        run.write_json(out/'run.json', dict(schema='bayesreg.hs.entry_refused.v1', run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, status=ENTRY_QC, reason=str(refused)[:2000], oracle=False, oracle_note='no channel was read: the reader refused at the alignment QC',
                                            channel_sources=None, checkpoints={}, field=None, role=HS.ROLE, code_sha=o.code_sha, frozen_commit=core.FROZEN_COMMIT, ended_utc=run.utc()))
        return 0
    finally:
        restore()


def guard_read_images(cell, out):
    """(restore): HS.read_images shadowed for this process so that a refusal of load_pair (the firewall, the admission, the hashes) on `cell` is written to <out>/entry_refusal.json as provenance_rejected (an input error, D74-4) before it is raised."""
    plain, shadowed = HS.read_images, HS.__dict__.get('read_images')

    def guarded(c):
        try:
            return plain(c)
        except (Exception, SystemExit) as error:        # noqa: BLE001 - every refusal of the reader of the images is an input error here
            if c == cell:
                Path(out).mkdir(parents=True, exist_ok=True)
                run.write_json(Path(out)/hc.REFUSAL_FILE, dict(kind='provenance_rejected', stage='load_pair', message=f'{type(error).__name__}: {error}'[:2000], utc=run.utc(), rule='load_pair refused the pair: an input error, no fallback (D74-4)'))
            raise
    HS.read_images = guarded                            # this process runs one unit; the instance attribute shadows the method for it only

    def restore():
        if shadowed is None:
            HS.__dict__.pop('read_images', None)
        else:
            HS.read_images = shadowed
    return restore


def repair_one(o):
    """fx's repair_one (run_fl.repair_one, the status renamed), with load_pair's refusals recorded as input errors, as run_one's (revision 1b, the follow-up's minor 4): classify reads such a repair as refused, not as a repair that crashed."""
    restore = guard_read_images(HS.require_cell(o.arm, o.tier, o.pair), o.out)
    try:
        return fxrun.repair_one(BASE, o)
    finally:
        restore()


# ---- the fallbacks (D74, D75): a unit that reads no label, then the identity's metrics --------------------------------------------------------------------
def u_sha256(u):
    return hashlib.sha256(np.ascontiguousarray(u).tobytes()).hexdigest()


def process_record():
    """pid and the cgroup line of this process (/proc/self/cgroup: under systemd the unit's name, e.g. .../tag-hs-fallback.service): a record beside the unit name the entry gives."""
    try:
        cgroup = Path('/proc/self/cgroup').read_text().strip()
    except OSError:
        cgroup = None
    return dict(pid=os.getpid(), cgroup=cgroup, invocation_id=os.environ.get('INVOCATION_ID'))


def decide_fallback(root, pair, *, fz, certify, config, results=None):
    """The record of one pair the fallback unit decides: the third certificate (C22.3-9 (d)) when the two certificates disagree; else, or when it does not pass, configuration C's field (d0_v1 A0, qualified by its records, certified again here), else the identity u = 0 (certified here).  No label is read: the firewall view gives the grid."""
    kind, why = decision.classify(root, pair)
    pin = HS.read_images(HS.field_cell(pair))
    record = dict(schema='bayesreg.hs.fallback.v1', pair=core.short_name(pair), kind=kind, why=why, reads_labels=False, certificate_config=dict(rp.CERTIFICATE_C), utc=run.utc(),
                  keys_read=list((getattr(pin, 'provenance', None) or {}).get('keys_read') or []),      # what load_pair gave this process: fixed, moving, initial_u at most (revision 1d, for P2 revision 4)
                  unit=FALLBACK_UNIT, process=process_record())                     # the unit that certifies (not an evaluation unit; P2 revision 4's key `unit`, revision 1e) and what the kernel says the process is in
    if kind == 'disagree':
        cell = HS.repair_cell(pair)
        d = HS.run_dir(root, cell)
        u, source = BASE.source_checked(cell, fz, pin.grid)
        v = BASE.efrun.repaired_field(u, d, run.read_json(d/'repair.json'))
        ladder = run.certify_ladder(v, pin.grid, (1.,), certify, config)
        record['third_certificate'] = dict(passed=ladder['accepted_alpha'] == 1., certificate=ladder, repaired_u_sha256=u_sha256(v), source=source)
        if record['third_certificate']['passed']:
            return dict(record, output='repair')
    status = rp.fallback_status(pair, results=results)
    record['a0_records'] = {k: status[k] for k in ('qualified', 'why', 'file_sha256', 'run_id', 'd0_class')}
    if status['qualified']:
        with np.load(rp.RP.a0_field(pair, results)) as z:
            u = np.array(z['u'])
        pin_u = (rp.RP.fallback_pins().get(core.short_name(pair)) or {}).get('u_sha256')
        if u_sha256(u) != pin_u:
            record.update(a0_why=f'u of final_native.npz is not the pinned state {pin_u}')
        else:
            ladder = run.certify_ladder(u, pin.grid, (1.,), certify, config)
            record['a0_certificate'] = ladder
            if ladder['accepted_alpha'] == 1.:
                return dict(record, output='A0', a0_why=None)
            record.update(a0_why=f"configuration C's field does not pass the certificate here: {ladder['status_at_1']}, infeasible {ladder['n_infeasible_at_1']}")
    else:
        record.update(a0_why=status['why'])
    zero = np.zeros(tuple(pin.grid.shape)+(3,))
    ladder = run.certify_ladder(zero, pin.grid, (1.,), certify, config)
    record['identity_certificate'] = ladder
    return dict(record, output='identity' if ladder['accepted_alpha'] == 1. else None)


def fallback_unit(o, results=None):
    """The unit tag-hs-fallback: first core.frozen_api() (jd_curl from the frozen tree, as run_fl.evaluate_all); every pair that needs a fallback or a third certificate (decide_hs.needs_fallback), once; an error is kept and the others go on.  No label."""
    root = Path(o.root)
    problem = HS.root_problem(root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    fz = core.frozen_api()
    BASE.enable_jit(root)
    from jd_curl.three_d.feasibility import certify_map
    config = run.certificate_config()
    failures = 0
    for pair in ALL_PAIRS:
        if not decision.needs_fallback(root, pair):
            continue
        d = HS.fallback_root(root)/core.short_name(pair)
        if (d/'fallback.json').exists() or (d/'fallback_error.json').exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        try:
            run.write_json(d/'fallback.json', decide_fallback(root, pair, fz=fz, certify=certify_map, config=config, results=results))
        except Exception as error:                      # kept; the others go on; not retried
            run.write_json(d/'fallback_error.json', dict(pair=pair, error=repr(error)[:2000], utc=run.utc()))
            failures += 1
    return 8 if failures else 0


def identity_pairs(root):
    return [p for p in ALL_PAIRS if (decision.fallback_decision(root, p) or {}).get('output') == 'identity']


def id_evaluate(o):
    """The unit tag-hs-eval-id: the identity u = 0 of the pairs whose fallback is the identity, its certificate at alpha = 1 and the metrics with RW's function; the truth labels for the metrics only (run_rp.evaluate_identity's lines)."""
    from time import perf_counter
    root = Path(o.root)
    problem = HS.root_problem(root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    core.frozen_api()
    pairs = identity_pairs(root)
    if not pairs:
        return 0
    BASE.enable_jit(root)
    ev = core.load_module('_mc0_evaluate', 'experiments/d0/evaluate.py')
    from jd_curl.three_d.feasibility import certify_map
    config = run.certificate_config()
    failures = 0
    for pair in pairs:
        d = HS.id_root(root)/core.short_name(pair)
        if (d/'identity_metrics.json').exists() or (d/'identity_error.json').exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        try:
            truth_pin, labs = fl.read_truth(pair, BASE.TRUTH_MEMBERS)
            u = np.zeros(tuple(truth_pin.grid.shape)+(3,))
            tick = perf_counter()
            ladder = run.certify_ladder(u, truth_pin.grid, (1.,), certify_map, config)
            seconds = perf_counter()-tick
            metrics = BASE.field_metrics(u, labs, tuple(truth_pin.grid.spacing), ev)
            run.write_json(d/'identity_metrics.json', dict(schema='bayesreg.hs.identity_metrics.v1', pair=pair, short=core.short_name(pair), oracle=False, role='the second tier of the fallback (D75): u = 0, only where the repair failed and configuration C\'s field did not qualify or pass', field='u = 0',
                                                          certificate=ladder, certified_at_1=ladder['accepted_alpha'] == 1., seconds_certificate=seconds, metrics=metrics, certificate_config=dict(rp.CERTIFICATE_C), evaluator=dict(script='experiments/d0/evaluate.py', sha256=core.sha_file(core.ROOT/'experiments/d0/evaluate.py')), evaluated_utc=run.utc()))
        except Exception as error:                      # kept; the others go on; not retried
            run.write_json(d/'identity_error.json', dict(pair=pair, error=repr(error)[:2000], utc=run.utc()))
            failures += 1
    return 8 if failures else 0


# ---- the seg phase's units ----------------------------------------------------------------------------------------------------------------------
def seg_unit_argv(unit, code, inner, *, log, gpu, runtime):
    """A unit of the seg phase: PYTHONPATH the checkout only (bayesreg and jd_curl from it, so the pinned allowlist's clean-code check holds: jd_curl/ is eb808be's byte for byte, hard rule 2), one thread, MemoryMax 8G."""
    env = dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, NUMBA_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=str(code))
    if not gpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return run.check_launch(['systemd-run', '--user', f'--unit={unit}', '--wait', '--collect', f'--working-directory={code}', f'--property=MemoryMax={SEG_MEMORY_MAX}', '--property=MemorySwapMax=4G', f'--nice={BASE.NICE}', '--property=CPUQuota=800%', f'--property=RuntimeMaxSec={runtime}',
                             f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}', *[f'--setenv={k}={v}' for k, v in env.items()], run.PY, '-B', f'{code}/{SCRIPT}', *inner])


def seg_inputs(source_root=None):
    """(subjects, split, allowlist, record): the 20 development subjects of the 19 pairs (eval_seg.admit_subjects: the allowlist source, not never_for_training, on no sealed list, no training subject; uf_nn.dev_subjects' rule) and the pinned folds_full_v2.json with the allowlist,
    as cli register reads them (load_pinned_full_inputs, the clean-code check included).  The allowlist's sources are verified against the main checkout (core.MAIN_REPO): data/ is its untracked link, a worktree has none."""
    import eval_seg
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.seg.unet.data import load_folds
    from bayesreg.seg.unet.pins import DEFAULT_ALLOWLIST, DEFAULT_FOLDS, DEFAULT_FULL_FOLDS, PINNED_FULL_COUNTS, TRAINING_SOURCE, load_pinned_full_inputs
    source_root = core.MAIN_REPO if source_root is None else source_root
    split, allowlist, record = load_pinned_full_inputs(DEFAULT_FULL_FOLDS, DEFAULT_ALLOWLIST, training_source=TRAINING_SOURCE, expect_counts=PINNED_FULL_COUNTS, source_root=source_root)
    subjects = eval_seg.admit_subjects(allowlist, 'stage_L2R_val_v1', load_sealed_lists(core.ROOT), load_folds(DEFAULT_FOLDS).training_subjects)
    if sorted(subjects) != sorted({s for p in ALL_PAIRS for s in p.split('__')}):
        raise hs_seg.HSSegFailed('the development subjects are not the 20 of the 19 pairs')
    return sorted(subjects), split, allowlist, record


def infer_one(o):
    """The unit tag-hs-infer: the candidate staged and its 20 predictions (hs_seg.infer_all).  Only images; the pinned inputs, a clean checkout."""
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.seg.unet.data import OasisFiles
    from bayesreg.seg.unet.pins import PINNED_FULL_FOLDS_SHA256, TRAINING_SOURCE
    from bayesreg.seg.unet.snapshot import code_record
    candidate = HS.candidate()
    subjects, split, allowlist, record = seg_inputs()
    evaluation = run.read_json(hc.uf_paths(HS.UF_OUT)['eval'])
    try:
        out = hs_seg.infer_all(candidate, subjects=subjects, seg_dir=HS.seg_dir(candidate), uf_out=HS.UF_OUT, nn_out=hc.NN_OUT, evaluation=evaluation, split=split, allowlist=allowlist, training_source=TRAINING_SOURCE, pinned_folds_sha256=PINNED_FULL_FOLDS_SHA256,
                               files=OasisFiles(nc.DATA_ROOT), device=o.device, code=dict(record.get('code') or code_record(), batch_sha=o.code_sha), command=sys.argv, sealed=load_sealed_lists(core.ROOT), evaluation_sha256=core.sha_file(hc.uf_paths(HS.UF_OUT)['eval']))
    except (hs_seg.HSSegFailed, FileExistsError) as failed:
        print(f'STOP: {failed}', file=sys.stderr)
        return 8
    return 0 if out['passed'] else 8


def segcheck_one(o):
    """The unit tag-hs-segcheck: hs_seg.segcheck, the truth of the 20 subjects only here."""
    import eval_seg
    candidate = HS.candidate()
    launch = run.read_json(HS.seg_dir(candidate)/SEGCHECK_LAUNCHED)
    if launch is None or launch.get('code_sha') != o.code_sha:
        print(f"STOP: no {SEGCHECK_LAUNCHED} of the seg queue for the code {o.code_sha} (there: {None if launch is None else launch.get('code_sha')}): the control reads the truth only when its queue launched it (revision 1b, the follow-up's minor 1)", file=sys.stderr)
        return 8
    try:                                                # written once, before the first truth label: a second start of the unit by hand reads nothing (revision 1c, minor 1); only the queue's approved reread moves it aside
        fd = os.open(HS.seg_dir(candidate)/SEGCHECK_STARTED, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        print(f"STOP: {HS.seg_dir(candidate)/SEGCHECK_STARTED} exists: the control started before; another read of the truth only by the lead's words (queue --phase seg --segcheck-reread)", file=sys.stderr)
        return 8
    with os.fdopen(fd, 'w') as stream:
        stream.write(json.dumps(dict(unit=SEGCHECK_UNIT, code_sha=o.code_sha, launch=launch, pid=os.getpid(), utc=run.utc()))+'\n')
    subjects, _, _, _ = seg_inputs()
    evaluation = run.read_json(hc.uf_paths(HS.UF_OUT)['eval'])
    try:
        document = hs_seg.segcheck(candidate, seg_dir=HS.seg_dir(candidate), evaluation=evaluation, subjects=subjects, data_root=eval_seg.DATA_ROOT, evaluation_sha256=core.sha_file(hc.uf_paths(HS.UF_OUT)['eval']))
    except (hs_seg.HSSegFailed, FileExistsError, ValueError) as failed:
        print(f'STOP: {failed}', file=sys.stderr)
        return 8
    return 0 if document['passed'] else 8


def write_reference(o, repo=None):
    """By hand, after HS reads 采用新分割器: experiments/p2/reference_<segmenter_id>.json for P2/P3 (hs_reference.py), from HS's JSON records only, once; the lead commits it.  The reading must be the batch's code's (as decide)."""
    import hs_reference as href
    problem = HS.root_problem(Path(o.root)) or reading_code_problem(Path(o.root))
    if problem:
        print(problem, file=sys.stderr)
        return 8
    root = Path(HS.DEFAULT_ROOT)                       # the spelling the records use (repair.json source.directory): another spelling of the same directory would not match P2's check (revision 1c, minor 3)
    try:
        on_disk = run.read_json(root/href.DECISION)
        again, _ = decision.decide(root, stops=BASE.ledger(root)['arms_stopped'])        # the reading again with the batch's code: a decision_hs.json changed by hand is not used (revision 1, review minor 12)
        if on_disk is None or json.dumps(again, sort_keys=True, ensure_ascii=False) != json.dumps(on_disk, sort_keys=True, ensure_ascii=False):
            raise href.ReferenceRefused(f'{root/href.DECISION} is not what decide_hs.py gives from the records now; run_hs.py decide first, or the lead looks')
        document = href.reference_document(root, on_disk)
        path = href.write(Path(core.ROOT if repo is None else repo)/href.reference_rel(document['segmenter_id']), document)
    except href.ReferenceRefused as refused:
        print(str(refused), file=sys.stderr)
        return 8
    fallback, third = href.fallback_rows(document), href.third_certificate_rows(document)
    print(f"{path}: sha256 {core.sha_file(path)}; commit it before P2 with --segmenter {document['segmenter_id']}" + (f"; the fallback rows (P2 takes their method_metrics, pinned by hs.decision_sha256): {fallback}" if fallback else '')
          + (f"; the rows hx_certified by the third certificate (the evaluation's review certificate did not pass, C22.3-9 (d); no method_metrics; P2 revision 3 accepts them by third_certificate_file and its sha256, the fallback unit's record): {third}" if third else ''))
    for short in third:
        problems = third_record_problems(run.read_json(document['p3']['pairs'][short].get('third_certificate_file') or '/nonexistent'))
        if problems:
            print(f'NOTE: the third certificate record of {short} does not hold what P2 asks: {problems}')
    return 0


EVALUATION_UNITS = ('tag-hs-eval', 'tag-hs-eval-r', ID_EVAL_UNIT, SEGCHECK_UNIT)


def third_record_problems(record):
    """Why a fallback unit's record does not show a third certificate from a process that read no label, as a list (what P2 revision 4 asks, revision 1d): load_pair's keys_read within fixed, moving,
    initial_u; reads_labels false; the certifying unit the fallback unit, not an evaluation unit; the third certificate passed at alpha = 1; the output the repaired field."""
    if record is None:
        return ['no record']
    third = record.get('third_certificate') or {}
    keys, unit = record.get('keys_read'), record.get('unit')
    checks = dict(keys_read=isinstance(keys, list) and {'fixed', 'moving'} <= set(keys) <= set(mc.IMAGE_KEYS), no_label=record.get('reads_labels') is False,      # P2 revision 4 (p2-tools 83af310): fixed and moving in, within fixed, moving, initial_u
                  unit=isinstance(unit, str) and unit == FALLBACK_UNIT and unit.startswith('tag-') and '-eval' not in unit and unit not in EVALUATION_UNITS,
                  passed=third.get('passed') is True and (third.get('certificate') or {}).get('accepted_alpha') == 1., output=record.get('output') == 'repair')
    return [k for k, ok in checks.items() if not ok]


def register(o):
    """By hand, after the review."""
    candidate = HS.candidate()
    subjects = sorted({s for p in ALL_PAIRS for s in p.split('__')})
    try:
        registry = hs_seg.register(candidate, seg_dir=HS.seg_dir(candidate), registry_path=HS.registry_path(), subjects=subjects)
    except (hs_seg.HSSegFailed, fxrun.nx_seg.RegistrationRefused) as refused:
        print(str(refused), file=sys.stderr)
        return 8
    print(f'{registry.path}: sha256 {registry.sha256}, models {sorted(registry.models)}; commit it before the fields queue')
    return 0


# ---- the checks of the inputs -------------------------------------------------------------------------------------------------------------------
def candidate_lines(out, candidate):
    """What a candidate's plan is: ids, files, members.  Names only."""
    c, ids = hc.CANDIDATES[candidate], hc.model_ids(candidate)
    out(f"candidate {candidate}: {c['words']}")
    out(f"  segmenter_id {c['segmenter_id']}; fold models {ids['A']} (fold 0: {hc.member_name('A', 0)}, {hc.member_name('A', 1)}), {ids['B']} (fold 1: {hc.member_name('B', 0)}, {hc.member_name('B', 1)})" + ('' if ids['nn'] is None else f", {ids['nn']} (fold {c['nn_fold']}: nnU-Net checkpoint_final.pth of {hc.NN_OUT})"))
    out(f'  registry {hc.REGISTRY_REL} (staged first at {hc.seg_dir(candidate, HS.RESULTS)/hc.STAGED_REGISTRY}); predictions {hc.seg_dir(candidate, HS.RESULTS)/"pred"}/<subject>/{{probs36.npz, structure_groups.npz, provenance.json}}')
    if ids['nn'] is not None:
        out(f"  the composer: 1/2 x UF4's float16 read back + 1/2 x {hc.uf_paths(HS.UF_OUT)['nn']}/probs/<subject>/probs36.npz read back (eval_uf.half_and_half), float16 with the argmax kept")


def uf_lines(out):
    """Whether UF's files are there; nothing of them is read before UF's reading."""
    paths = hc.uf_paths(HS.UF_OUT)
    members = [hc.member_name(f, s) for f, s in hc.UF_MEMBERS]
    out(f"UF {paths['out']}: decision_uf.json there {paths['decision'].is_file()}, eval_uf.json there {paths['eval'].is_file()}, nn/convert_record.json there {(paths['nn']/'convert_record.json').is_file()}; member.json there: " + ', '.join(f"{m} {(paths['out']/m/'member.json').is_file()}" for m in members) + ' (existence only: nothing of UF is read here)')


def rw_lines(out, results=None):
    """RW, the reference: 19/19 by both certificates (rw_v1's records)."""
    reference = decision.rw_reference(results)
    bad = [reference[p]['problem'] for p in ALL_PAIRS if reference[p]['problem']]
    seg35 = decision.mean(reference[p]['metrics']['seg35'] for p in ALL_PAIRS if reference[p]['metrics'])
    out(f"RW (rw_v1, the reference Hx24 on v1's channels): {sum(reference[p]['metrics'] is not None for p in ALL_PAIRS)}/19 by both certificates, seg35 mean {seg35}; problems {bad or 'none'}")
    return bad


def fallback_lines(out, results=None):
    """The 19 fields of configuration C by d0_v1's records (rp_core.fallback_status): listed; one that does not qualify is not a problem (the identity takes its place)."""
    statuses = {core.short_name(p): rp.fallback_status(p, results=results) for p in ALL_PAIRS}
    bad = [k for k, v in statuses.items() if not v['qualified']]
    out(f"configuration C's fields (d0_v1 A0) qualified by their records: {19-len(bad)}/19; not qualified (the identity would take their place): {bad or 'none'}")
    return statuses


def registry_lines(out, candidate, repo=None):
    """The registry file: tracked, clean, the staged bytes, the candidate's models."""
    problems = []
    path = HS.registry_path(repo)
    staged = hc.seg_dir(candidate, HS.RESULTS)/hc.STAGED_REGISTRY
    if not path.is_file():
        return ['等登记：没有 '+hc.REGISTRY_REL]
    from bayesreg.io.pair import load_segmenter_registry
    from bayesreg.seg.structure_layer import is_formal_registry, registry_git_state
    registry = load_segmenter_registry(path)
    git = registry_git_state(path)
    out(f"registry {path}: models {sorted(registry.models)}, tracked {git.get('tracked')}, clean {git.get('clean')}, formal {is_formal_registry(path)}; the staged bytes: {staged.is_file() and staged.read_bytes() == path.read_bytes()}")
    ids = hc.model_ids(candidate)
    if sorted(registry.models) != sorted(v for v in ids.values() if v is not None) or any(e['segmenter_id'] != hc.CANDIDATES[candidate]['segmenter_id'] for e in registry.models.values()):
        problems.append(f'{hc.REGISTRY_REL} is not the candidate {candidate}')
    if not (git.get('tracked') is True and git.get('clean') is True):
        problems.append(f'{hc.REGISTRY_REL} is not committed and clean')
    if not is_formal_registry(path):
        problems.append(f'{hc.REGISTRY_REL} is not one of structure_layer.FORMAL_REGISTRIES')
    if not staged.is_file() or staged.read_bytes() != path.read_bytes():
        problems.append(f'{hc.REGISTRY_REL} is not the staged bytes {staged}')
    return problems


def prediction_problems(out, candidate, *, seg_root=None, registry_sha256=None, stage=None):
    """The 20 predictions: provenance (segmenter, policy, test_labels_seen, the registry's sha256), the three files, Atropos beside; the seg phase's two records passed."""
    seg_root = mc.SEG_ROOT if seg_root is None else seg_root
    sd = hc.seg_dir(candidate, HS.RESULTS)
    subjects = sorted({s for p in ALL_PAIRS for s in p.split('__')})
    problems = list(hs_seg.infer_problems(run.read_json(sd/hc.INFER_RECORD), subjects, candidate))
    check = run.read_json(sd/hc.SEGCHECK_RECORD)
    out(f"seg phase: infer_record.json passed {(run.read_json(sd/hc.INFER_RECORD) or {}).get('passed')}; segcheck.json passed {None if check is None else check.get('passed')}, mean difference {None if check is None else check.get('mean_difference')}")
    if check is None or check.get('passed') is not True:
        problems.append('segcheck.json: the control against UF\'s rows did not pass or is not there')
    sources = {}                                    # the source image of each subject in the stage's manifests, through the firewall's reader
    from bayesreg.io.pair import _read_manifest
    for pair in ALL_PAIRS:
        _, _, manifest = _read_manifest(Path(core.STAGE if stage is None else stage)/pair/'manifest.json')
        for side in ('fixed', 'moving'):            # the reader's selected dict: pair_sha256, fixed_subject, moving_subject, source_sha256 = {side: sha256} (bayesreg/io/pair.py; HSF: not the file's `sources`)
            sources.setdefault(manifest[f'{side}_subject'], set()).add(manifest['source_sha256'][side])
    for subject in subjects:
        p = sd/'pred'/subject
        missing = [name for name in (hs_seg.PROVENANCE_FILE, hs_seg.PROBS_FILE, hs_seg.GROUPS_FILE) if not (p/name).is_file()]
        missing += [] if (Path(seg_root)/subject/'provenance.json').is_file() else [f'Atropos {Path(seg_root)/subject}']
        if missing:
            problems.append(f'{subject}: missing {missing}')
            continue
        prov = json.loads((p/hs_seg.PROVENANCE_FILE).read_text())
        if prov.get('segmenter_id') != hc.CANDIDATES[candidate]['segmenter_id'] or prov.get('policy') != 'all_folds_mean' or prov.get('test_labels_seen') is not False or (registry_sha256 is not None and prov.get('registry_sha256') != registry_sha256):
            problems.append(f'{subject}: the prediction is not the registered candidate')
        if {(prov.get('input_sha256') or {}).get(hs_seg.IMAGE_FILE)} != sources.get(subject):
            problems.append(f'{subject}: the prediction was made from another image than the manifests name')
    return problems


def fields_input_problems(out=lambda text: None, *, candidate=None, stage=None, allowlist=None, results=None, seg_root=None, repo=None):
    """The stage, the frozen tree, the registry, the 20 predictions, RW 19/19, configuration C's pins readable.  No label; nothing written."""
    candidate = HS.candidate() if candidate is None else candidate
    problems = fxrun.stage_problems(out, ALL_PAIRS, stage=stage, allowlist=allowlist)+fxrun.frozen_problems(out)
    registry = registry_lines(out, candidate, repo)
    problems += registry
    sha = None if registry else hashlib.sha256(HS.registry_path(repo).read_bytes()).hexdigest()
    problems += prediction_problems(out, candidate, seg_root=seg_root, registry_sha256=sha, stage=stage)
    problems += rw_lines(out, results)
    fallback_lines(out, results)
    return problems


def seg_input_problems(out=lambda text: None, *, candidate=None):
    """Before the seg phase: UF's members and, for (b), nnU-Net's records; the registry not yet written with other bytes.  Names and sha256 only."""
    candidate = HS.candidate() if candidate is None else candidate
    problems = []
    paths = hc.uf_paths(HS.UF_OUT)
    for f, s in hc.UF_MEMBERS:
        if not (paths['out']/hc.member_name(f, s)/'member.json').is_file():
            problems.append(f'UF member {hc.member_name(f, s)} has no member.json')
    if hc.CANDIDATES[candidate]['nn_model'] is not None:
        if not (paths['nn']/'convert_record.json').is_file():
            problems.append("UF's nn/convert_record.json is not there")
        if not (hc.NN_OUT/'train_record.json').is_file() or not (hc.NN_OUT/'prepare_manifest.json').is_file():
            problems.append('nnunet_v1/train_record.json or prepare_manifest.json is not there')
    path, staged = HS.registry_path(), hc.seg_dir(candidate, HS.RESULTS)/hc.STAGED_REGISTRY
    if path.is_file() and (not staged.is_file() or staged.read_bytes() != path.read_bytes()):
        problems.append(f'{hc.REGISTRY_REL} is there with other bytes than the staged registry')
    out(f'seg phase inputs of {candidate}: problems {problems or "none"}')
    return problems


# ---- the queues ------------------------------------------------------------------------------------------------------------------------------------
def reading_code_problem(root):
    """Why this process may not write this root's reading, or None: the files that hold the rules must be the batch's commit, byte for byte.  No batch.json: nothing has run."""
    batch = run.read_json(Path(root)/'batch.json')
    if batch is None:
        return None
    for path, rel in READING_FILES:
        theirs = BASE.committed_bytes(batch['code_sha'], rel)
        if theirs is None or theirs != Path(path).read_bytes():
            return f"{path} is not {rel} of the batch's commit {batch['code_sha']}: the reading's rules would not be the batch's; run the driver from $ROOT/code"
    return None


def registry_problem(sha, candidate):
    """Why the fields phase may not check out this commit, or None: the commit must hold the registration, byte for byte the working tree's file and the staged one."""
    theirs, mine = BASE.committed_bytes(sha, hc.REGISTRY_REL), HS.registry_path()
    staged = hc.seg_dir(candidate, HS.RESULTS)/hc.STAGED_REGISTRY
    if theirs is None:
        return f'{sha} has no {hc.REGISTRY_REL}: run_hs.py register after the review, commit it, then the fields; nothing checked out'
    if not mine.is_file() or theirs != mine.read_bytes() or not staged.is_file() or theirs != staged.read_bytes():
        return f'{hc.REGISTRY_REL} at {sha} is not the file {mine} or not the staged bytes; nothing checked out'
    return None


def batch_value(o, gate, plan, candidate, **extra):
    return dict(plan=plan, candidate=candidate, segmenter_id=hc.CANDIDATES[candidate]['segmenter_id'], gap_seconds=getattr(o, 'gap_seconds', None), uf_decision_sha256=core.sha_file(hc.uf_paths(HS.UF_OUT)['decision']), uf_eval_sha256=core.sha_file(hc.uf_paths(HS.UF_OUT)['eval']),
                caps=dict(run_seconds=fl.RUN_CAP_SECONDS, run_units_seconds=BASE.RUN_UNITS_CAP_SECONDS, device_gib=mc.VRAM_LIMIT_BYTES/2**30, memory_max=BASE.MEMORY_MAX, nice=BASE.NICE, repair_wall_seconds=fx.FUSE_SECONDS, repair_runtime_seconds=fx.REPAIR_RUNTIME_SECONDS),
                gate=dict(own_units=list(gate.own_units), load_limit=gate.load_limit, gpu_free_mib=gate.gpu_free_mib, mem_available_gib=gate.mem_available_gib, home_floor_gb=run.HOME_FLOOR_GB, host_floor_gib=gate.host_floor_gib), **extra)


def waiting_of(phase):
    """(waiting, stop, candidate): UF's reading, then what each phase waits for."""
    waiting, stop, candidate, _ = hc.uf_reading(HS.UF_OUT)
    if candidate is None or phase != 'fields':
        return waiting, stop, candidate
    sd = hc.seg_dir(candidate, HS.RESULTS)
    if (run.read_json(sd/hc.SEGCHECK_RECORD) or {}).get('passed') is not True:
        waiting.append('等分割：seg 阶段没有做完或控制没过（infer_record.json、segcheck.json）')
    if not HS.registry_path().is_file():
        waiting.append(f'等登记：没有 {hc.REGISTRY_REL}（复核之后 run_hs.py register，再提交）')
    rw = run.read_json(fx.RW.DEFAULT_ROOT/'analysis/decision_rw.json')
    if rw is None or rw.get('reading') != fxrun.decision.READ_RW_ALL:
        waiting.append(f"等 RW：RW 的读数不是 {fxrun.decision.READ_RW_ALL}（{None if rw is None else rw.get('reading')}）")
    return waiting, stop, candidate


def seg_body(o, system, root, log):
    """The seg phase: the checks that need no code first, then the checkout $ROOT/seg/code, tag-hs-infer, tag-hs-segcheck.  It ends by handing to the lead: the review, then `register`."""
    BASE.keep_old_stops(root)
    gate = replace(BASE.GATE, host_floor_gib=o.min_host_gib)
    sha = fxrun.nx_commit(o.commit)
    waiting, stop, candidate = waiting_of('seg')
    if waiting or stop:
        raise run.Stop('；'.join(waiting+([stop] if stop else []))+'; nothing checked out')
    problems = fxrun.safe(seg_input_problems)
    if problems:
        raise run.Stop(f'the inputs, before any run: {problems}; nothing checked out')
    (root/hc.SEG_CODE_DIR).mkdir(parents=True, exist_ok=True)
    code, sha = system.code_checkout(root/hc.SEG_CODE_DIR, sha)
    BASE.write_batch(root, 'batch_seg.json', batch_value(o, gate, [INFER_UNIT, SEGCHECK_UNIT], candidate), sha)
    sd = hc.seg_dir(candidate, HS.RESULTS)
    sd.mkdir(parents=True, exist_ok=True)
    log(f'start: the seg phase of {candidate} ({hc.CANDIDATES[candidate]["segmenter_id"]}); the code {sha}')
    if run.read_json(sd/hc.INFER_RECORD) is None:
        if (sd/INFER_LAUNCHED).exists() or (sd/'subjects').exists() or (sd/'pred').exists() or (sd/hc.STAGED_REGISTRY).exists():
            raise run.Stop(f'{sd}: the inference was launched before and left no infer_record.json; kept, not rerun; the lead decides')
        run.wait_for_gate(system, root, True, o.wait_minutes, log, replace(gate, gpu_free_mib=SEG_GPU_FREE_MIB))
        run.write_json(sd/INFER_LAUNCHED, dict(unit=INFER_UNIT, code_sha=sha, utc=run.utc()))
        log(f'launch {INFER_UNIT}')
        rc = system.run(seg_unit_argv(INFER_UNIT, code, ['infer-one', '--code-sha', sha], log=root/'infer_unit.log', gpu=True, runtime=INFER_RUNTIME), root/f'{INFER_UNIT}.systemd_run.log')
        if rc != 0:
            raise run.Stop(f'{INFER_UNIT} exit {rc}; kept, not rerun')
    subjects = sorted({s for p in ALL_PAIRS for s in p.split('__')})
    problems = hs_seg.infer_problems(run.read_json(sd/hc.INFER_RECORD), subjects, candidate)
    if problems:
        raise run.Stop(f'the inference: {problems}')
    reread = getattr(o, 'segcheck_reread', None)
    if reread is not None and not ((sd/SEGCHECK_LAUNCHED).exists() and run.read_json(sd/hc.SEGCHECK_RECORD) is None):
        raise run.Stop('--segcheck-reread given, but no launched control without segcheck.json waits for it; nothing started')
    if run.read_json(sd/hc.SEGCHECK_RECORD) is None:
        if (sd/SEGCHECK_LAUNCHED).exists() and reread is None:
            raise run.Stop(f'{sd}: the control was launched before and left no segcheck.json; another read of the development truth needs the lead\'s words (queue --phase seg --segcheck-reread "<原话与时间>"); nothing started')
        run.wait_for_gate(system, root, False, o.wait_minutes, log, gate)        # a gate closed to its limit stops here, and the approval is not used (revision 1b, the follow-up's minor 1)
        if (sd/SEGCHECK_LAUNCHED).exists():
            stamp = run.utc().replace(':', '').replace('-', '')
            previous = sd/f'segcheck_launched_{stamp}.json'
            (sd/SEGCHECK_LAUNCHED).rename(previous)
            started = None
            if (sd/SEGCHECK_STARTED).exists():
                started = sd/f'segcheck_started_{stamp}.json'
                (sd/SEGCHECK_STARTED).rename(started)
            rereads = run.read_json(sd/SEGCHECK_REREADS) or []
            run.write_json(sd/SEGCHECK_REREADS, rereads+[dict(words=reread, previous_launch=previous.name, previous_start=None if started is None else started.name, utc=run.utc())])
            log(f'the lead approved another read of the development truth by the control: {reread!r}; the previous launch kept as {previous.name}')
        run.write_json(sd/SEGCHECK_LAUNCHED, dict(unit=SEGCHECK_UNIT, code_sha=sha, approved_reread=reread, utc=run.utc()))
        log(f'launch {SEGCHECK_UNIT}')
        rc = system.run(seg_unit_argv(SEGCHECK_UNIT, code, ['segcheck', '--code-sha', sha], log=root/'segcheck_unit.log', gpu=False, runtime=SEGCHECK_RUNTIME), root/f'{SEGCHECK_UNIT}.systemd_run.log')
        check = run.read_json(sd/hc.SEGCHECK_RECORD)
        if check is None:
            raise run.Stop(f'{SEGCHECK_UNIT} exit {rc}, no segcheck.json; kept, not rerun')
    check = run.read_json(sd/hc.SEGCHECK_RECORD)
    run.write_json(root/'seg_end.json', dict(candidate=candidate, segcheck_passed=check['passed'], mean_difference=check['mean_difference'], n_rows_equal=check['n_rows_equal'], n_count_equal=check['n_count_equal'], next='the independent review, then run_hs.py register and its commit, then queue --phase fields', utc=run.utc()))
    if check['passed'] is not True:
        raise run.Stop(f"the control against UF's rows did not pass: {check['mean_difference']} (tolerance {check['tolerance']}); nothing is registered")
    log(f"the seg phase is done: the control passed ({check['mean_difference']}, rows equal {check['n_rows_equal']}/20).  Next by hand: the independent review, run_hs.py register, commit {hc.REGISTRY_REL}, then queue --phase fields")
    return 0


def repairs_of_done_fields(root, log):
    """The repairs whose field is done: a field that is not done has nothing to repair (it goes to the fallback)."""
    out = []
    for pair in ALL_PAIRS:
        status = (run.read_json(HS.run_dir(root, HS.field_cell(pair))/'run.json') or {}).get('status')
        if status == 'done':
            out.append(HS.repair_cell(pair))
        else:
            log(f'skip {HS.repair_cell(pair).run_id}: its field is {status}, nothing to repair (the fallback decides)')
    return out


def fallbacks(o, system, root, log, code, gate, rcs):
    """tag-hs-fallback when a pair needs it, then tag-hs-eval-id when the identity is an output.  Both CPU."""
    need = [core.short_name(p) for p in ALL_PAIRS if decision.needs_fallback(root, p)]
    if not need:
        log('no pair needs a fallback or a third certificate')
        return
    log(f'the fallback unit for {need}')
    run.wait_for_gate(system, root, False, o.wait_minutes, log, gate)
    log(f'launch {FALLBACK_UNIT}')
    rcs[FALLBACK_UNIT] = system.run(BASE.unit_argv(FALLBACK_UNIT, code, ['fallback', '--root', str(root)], log=root/'fallback.log', gpu=False, numba=BASE.EVAL_THREADS, quota=BASE.EVAL_QUOTA, runtime=FALLBACK_RUNTIME), root/f'{FALLBACK_UNIT}.systemd_run.log')
    if identity_pairs(root):
        run.wait_for_gate(system, root, False, o.wait_minutes, log, gate)
        log(f'launch {ID_EVAL_UNIT}')
        rcs[ID_EVAL_UNIT] = system.run(BASE.unit_argv(ID_EVAL_UNIT, code, ['id-evaluate', '--root', str(root)], log=root/'id_eval.log', gpu=False, numba=BASE.EVAL_THREADS, quota=BASE.EVAL_QUOTA, runtime=run.EVAL_RUNTIME_MAX), root/f'{ID_EVAL_UNIT}.systemd_run.log')


def finish(log, result, rcs, stop):
    log(f"reading: {result['reading']}.  Handed to the lead: no next step starts here")
    failed = {unit: rc for unit, rc in rcs.items() if rc != 0}
    if failed:
        raise run.Stop(f'a unit failed: {failed}; the reading is written, a pair without its record counts as not read')
    if stop is not None:
        raise run.Stop(f'a phase stopped: {stop}')
    if result['reading'] == hc.UNDECIDABLE:
        raise run.Stop(f"the reading is {hc.UNDECIDABLE}: {result['reason']}"[:1000])
    return 0


def fields_body(o, system, root, log):
    """The fields phase: every check that needs no code, then the checkout $ROOT/code at the commit that holds the registration; the expectation; the 19 fields, their evaluation, the repairs, their evaluation, the fallbacks; the reading."""
    BASE.keep_old_stops(root)
    gate = replace(BASE.GATE, host_floor_gib=o.min_host_gib)
    sha = fxrun.nx_commit(o.commit)
    waiting, stop, candidate = waiting_of('fields')
    if waiting or stop:
        raise run.Stop('；'.join(waiting+([stop] if stop else []))+'; nothing checked out')
    problem = registry_problem(sha, candidate)
    if problem:
        raise run.Stop(problem)
    problems = fxrun.safe(fields_input_problems)
    if problems:
        raise run.Stop(f'the inputs, before any run: {problems}; nothing checked out')
    seg_sha = (run.read_json(root/'batch_seg.json') or {}).get('code_sha')
    changed, problems, approval = code_drift(seg_sha, sha)
    if problems:
        raise run.Stop(f'the code of the fields is not the code of the seg phase but for the registration: {problems}; nothing checked out')
    code, sha = system.code_checkout(root, sha)
    cells = HS.all_cells(root)
    BASE.write_batch(root, 'batch.json', batch_value(o, gate, [c.run_id for c in cells], candidate, registry_sha256=hashlib.sha256(HS.registry_path().read_bytes()).hexdigest(), evaluation=dict(numba_threads=BASE.EVAL_THREADS, cpu_quota_percent=BASE.EVAL_QUOTA),
                                                     seg_code_sha=seg_sha, changed_since_seg=changed, drift_approval=approval, rw_pins=decision.rw_pins()), sha)     # drift_approval: None, or the approval that let a forbidden change through (HSF)
    problem = reading_code_problem(root)
    if problem:
        raise run.Stop(problem)
    reference = decision.rw_reference()
    prediction = decision.write_prediction(root, run.read_json(hc.uf_paths(HS.UF_OUT)['eval']), candidate, decision.mean(reference[p]['metrics']['seg35'] for p in ALL_PAIRS), core.sha_file(hc.uf_paths(HS.UF_OUT)['eval']))
    log(f"start: {candidate} on the 19 pairs; the expectation 〔模型〕 HS - RW {prediction['pairs19']['expected_delta']} (prediction_hs.json, before the first field); /mnt/c floor {o.min_host_gib:g} GiB")
    rcs, stop = {}, None
    try:
        BASE.phase_and_evaluation(o, system, root, log, code, sha, HS.plan_all_fields(), gate, BASE.EVAL_UNIT, rcs)
        BASE.evaluated(rcs, BASE.EVAL_UNIT, 'the repairs are')
        BASE.phase_and_evaluation(o, system, root, log, code, sha, repairs_of_done_fields(root, log), gate, BASE.EVAL_R1_UNIT, rcs)
        BASE.evaluated(rcs, BASE.EVAL_R1_UNIT, 'the fallbacks are')
        fallbacks(o, system, root, log, code, gate, rcs)
    except run.Stop as error:
        stop = str(error)
        log(f'STOP: {stop}')
    state = BASE.ledger(root)
    result = decision.write_hs(root, stops=state['arms_stopped'])
    run.write_json(root/'queue_end.json', dict(stop=stop, unit_rc=rcs, reading=result['reading'], reason=result['reason'], counts=result['counts'], delta_seg35=result['compare']['seg35']['mean'], lower=result['compare']['seg35'].get('lower'), ledger=state, utc=run.utc()))
    return finish(log, result, rcs, stop)


BODIES = dict(seg=seg_body, fields=fields_body)
DRIFT_FORBIDDEN = ('bayesreg', 'jd_curl', 'configs', 'splits', 'scripts', *(f'experiments/{d}' for d in ('fl', 'ef', 'mc0', 'hs', 'd0', 'nn', 'uf', 'unv2', 'ul', 'p1')))     # what the units of HS import or read (revision 1, review minor 8)
DRIFT_APPROVAL_REL = 'runs/20261106-hs/drift_approval.json'     # HSF: the lead writes and commits it after the fix's independent review; read only from the fields commit (git show), never from the working tree
DRIFT_APPROVAL_SCHEMA = 'bayesreg.hs.drift_approval.v1'
DRIFT_APPROVABLE = ('experiments/hs/run_hs.py',)                # HSF: the one path an approval may name; any other path in its entries voids the whole approval
DRIFT_APPROVAL_KEYS, DRIFT_ENTRY_KEYS = frozenset(('schema', 'seg_code_sha', 'entries')), frozenset(('path', 'diff_sha256', 'review'))


def git_bytes(*args, repo=None):
    """(rc, stdout as bytes, stderr as text) of git in the repository (core.ROOT, or `repo`): the bytes git writes, not decoded."""
    done = subprocess.run(['git', *args], cwd=core.ROOT if repo is None else repo, capture_output=True)
    return done.returncode, done.stdout, done.stderr.decode('utf-8', 'replace').strip()


def committed_file(sha, rel, repo=None):
    """The bytes of `rel` at the commit `sha`, None when git cannot say: run_fl.committed_bytes' command (BASE.committed_bytes, `git show <sha>:<rel>`) with the repository a parameter for the tests."""
    rc, out, _ = git_bytes('show', f'{sha}:{rel}', repo=repo)
    return out if rc == 0 else None


def is_drift_forbidden(path):
    """A change the fields may not have since the seg phase: under DRIFT_FORBIDDEN or a .py at the repository's root, the registry file excepted."""
    return path != hc.REGISTRY_REL and (('/' not in path and path.endswith('.py')) or any(path == d or path.startswith(d+'/') for d in DRIFT_FORBIDDEN))


def drift_changes(seg_sha, sha, repo=None):
    """(changed paths, the forbidden ones, error or None) of `git diff --no-renames --name-only -z <seg_sha> <sha>`: both paths of a move (revision 1b, the follow-up's minor 2); -z: the paths as they are, not quoted (HSF: a quoted path of a forbidden directory did not start with it)."""
    rc, out, err = rp.git('diff', '--no-renames', '--name-only', '-z', seg_sha, sha, cwd=repo)
    if rc:
        return [], [], f'git diff {seg_sha} {sha}: {err}'
    changed = [path for path in out.split('\0') if path]
    return changed, [path for path in changed if is_drift_forbidden(path)], None


DRIFT_DIFF_ARGS = ('diff', '--no-renames', '--full-index')     # --full-index: the index line's blob ids in full; abbreviated, their length grows with the repository's object count (7 -> 8 at 2**14 objects, about 12 thousand here on 11-06) and the bytes with it


def drift_diff_sha256(seg_sha, sha, path, repo=None):
    """(sha256 of the bytes of `git diff --no-renames --full-index <seg_sha> <sha> -- <path>`, None) or (None, error): what an entry of the approval pins."""
    rc, out, err = git_bytes(*DRIFT_DIFF_ARGS, seg_sha, sha, '--', path, repo=repo)
    return (hashlib.sha256(out).hexdigest(), None) if rc == 0 else (None, f'git diff {seg_sha} {sha} -- {path}: {err}')


def parse_drift_approval(data):
    """(document, None) or (None, why): the approval's bytes as the schema bayesreg.hs.drift_approval.v1, strictly (HSF): a JSON object of UTF-8 with exactly schema, seg_code_sha (40 hex), entries; entries a non-empty list of objects with exactly
    path, diff_sha256 (64 hex), review (a non-empty string), no path twice; a duplicate key anywhere is refused.  A path that is not in DRIFT_APPROVABLE voids the whole approval."""
    def no_duplicates(items):
        keys = [key for key, _ in items]
        if len(set(keys)) != len(keys):
            raise ValueError(f'duplicate keys {sorted(keys)}')
        return dict(items)
    try:
        document = json.loads(data.decode('utf-8'), object_pairs_hook=no_duplicates)
    except ValueError as error:                         # UnicodeDecodeError and json.JSONDecodeError are ValueErrors
        return None, f'not a JSON object of the schema {DRIFT_APPROVAL_SCHEMA}: {error}'[:500]
    if not isinstance(document, dict) or set(document) != DRIFT_APPROVAL_KEYS or document['schema'] != DRIFT_APPROVAL_SCHEMA:
        return None, f'not the schema {DRIFT_APPROVAL_SCHEMA} with exactly the keys {sorted(DRIFT_APPROVAL_KEYS)}'
    if not (isinstance(document['seg_code_sha'], str) and re.fullmatch('[0-9a-f]{40}', document['seg_code_sha'])):
        return None, 'seg_code_sha is not 40 lowercase hex'
    entries = document['entries']
    if not isinstance(entries, list) or not entries:
        return None, 'entries is not a non-empty list'
    for entry in entries:
        if not (isinstance(entry, dict) and set(entry) == DRIFT_ENTRY_KEYS and isinstance(entry['path'], str) and isinstance(entry['diff_sha256'], str) and re.fullmatch('[0-9a-f]{64}', entry['diff_sha256'])
                and isinstance(entry['review'], str) and entry['review'].strip()):
            return None, f'an entry is not exactly path, diff_sha256 (64 lowercase hex), review (a non-empty string): {str(entry)[:300]}'
    paths = [entry['path'] for entry in entries]
    if len(set(paths)) != len(paths):
        return None, f'a path is named twice: {paths}'
    other = [path for path in paths if path not in DRIFT_APPROVABLE]
    if other:
        return None, f'{other[:20]} may not be approved (only {list(DRIFT_APPROVABLE)}): the whole approval is void'
    return document, None


def drift_approval(seg_sha, sha, bad, repo=None):
    """(record, None) when the approval committed at `sha` approves every forbidden change in `bad`, else (None, why) and the drift STOPs as before (HSF).  It holds only when: the file is committed at `sha` (the working tree's does not count),
    it parses as the schema (parse_drift_approval: only DRIFT_APPROVABLE may be named), its seg_code_sha is batch_seg.json's, every path of `bad` has an entry, and each entry's diff_sha256 is the sha256 of `git diff --no-renames <seg_sha> <sha> -- <path>`.
    The record (batch.json's drift_approval): the file, its sha256, the entries as they are in it, the paths approved, and the sha256 of each review document at `sha` (None where it is not committed there: recorded, not a STOP)."""
    data = committed_file(sha, DRIFT_APPROVAL_REL, repo)
    if data is None:
        return None, f'no {DRIFT_APPROVAL_REL} committed at {sha[:7]} (a file only in the working tree does not count)'
    document, why = parse_drift_approval(data)
    if why:
        return None, why
    if document['seg_code_sha'] != seg_sha:
        return None, f"its seg_code_sha {document['seg_code_sha']} is not batch_seg.json's {seg_sha}"
    entries = {entry['path']: entry for entry in document['entries']}
    missing = [path for path in bad if path not in entries]
    if missing:
        return None, f'no entry for {missing[:20]}'
    for path, entry in entries.items():
        digest, error = drift_diff_sha256(seg_sha, sha, path, repo)
        if error:
            return None, error
        if digest != entry['diff_sha256']:
            return None, f"{path}: the sha256 of its diff since {seg_sha[:7]} is {digest}, the approval's {entry['diff_sha256']}"
    reviews = {}
    for entry in document['entries']:
        review = committed_file(sha, entry['review'], repo)
        reviews[entry['review']] = None if review is None else hashlib.sha256(review).hexdigest()
    return dict(file=DRIFT_APPROVAL_REL, sha256=hashlib.sha256(data).hexdigest(), commit=sha, seg_code_sha=seg_sha, entries=document['entries'], approved=list(bad), review_sha256=reviews), None


def code_drift(seg_sha, sha, repo=None):
    """(changed paths, problems, approval) between the seg phase's commit and this one: a change under DRIFT_FORBIDDEN or a .py at the repository's root is a problem, but the registry file; runs/, docs/, tests/, other packages' directories are listed only.
    HSF: forbidden changes are no problem when the drift approval committed at `sha` approves every one of them (drift_approval); approval is then its record for batch.json, else None.  The approval is read only when there is a forbidden change."""
    if seg_sha is None:
        return [], ['batch_seg.json has no code_sha: the seg phase has not run'], None
    changed, bad, error = drift_changes(seg_sha, sha, repo)
    if error:
        return [], [error], None
    if not bad:
        return changed, [], None
    approval, why = drift_approval(seg_sha, sha, bad, repo)
    if approval is not None:
        return changed, [], approval
    return changed, [f'changed since the seg phase ({seg_sha[:7]}): {bad[:20]}; not approved: {why}'], None


def drift_approval_document(seg_sha, sha, review, repo=None):
    """(document, notes): the approval file the lead would commit for the fields at `sha` (HSF, `run_hs.py drift-approval`): an entry for each forbidden change since seg_sha that may be approved, with the sha256 of its diff and `review`.
    notes: why the fields would STOP even with it (a forbidden change that may not be approved, no forbidden change at all, a git error); the review document not committed at `sha` is noted too.  Nothing written."""
    if seg_sha is None:
        return None, ['batch_seg.json has no code_sha: the seg phase has not run']
    _, bad, error = drift_changes(seg_sha, sha, repo)
    if error:
        return None, [error]
    notes = [f'{path} changed since the seg phase and may not be approved: the fields STOP with or without an approval' for path in bad if path not in DRIFT_APPROVABLE]
    entries = []
    for path in bad:
        if path in DRIFT_APPROVABLE:
            digest, error = drift_diff_sha256(seg_sha, sha, path, repo)
            if error:
                return None, [error]
            entries.append(dict(path=path, diff_sha256=digest, review=review))
    if not bad:
        notes.append(f'nothing under DRIFT_FORBIDDEN changed since {seg_sha[:7]}: no approval is needed (an approval file is not read then)')
    if committed_file(sha, review, repo) is None:
        notes.append(f'the review {review} is not committed at {sha[:7]} (batch.json would record its sha256 as None; not a STOP)')
    return dict(schema=DRIFT_APPROVAL_SCHEMA, seg_code_sha=seg_sha, entries=entries), notes


def print_drift_approval(o, repo=None):
    """`run_hs.py drift-approval --review <path>`: the approval file for the current HEAD, printed to stdout for the lead to copy (stderr: for which commit it holds, and the notes).  Reads batch_seg.json of the root and git; writes nothing."""
    problem = HS.root_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    seg_sha = (run.read_json(Path(o.root)/'batch_seg.json') or {}).get('code_sha')
    rc, out, err = git_bytes('rev-parse', '--verify', 'HEAD^{commit}', repo=repo)
    if rc:
        print(f'git rev-parse HEAD: {err}', file=sys.stderr)
        return 8
    sha = out.decode().strip()
    document, notes = drift_approval_document(seg_sha, sha, o.review, repo)
    if document is not None:
        print(json.dumps(document, indent=1, ensure_ascii=False))
    print(f"for the fields at HEAD {sha} against the seg phase's {seg_sha}: commit it as {DRIFT_APPROVAL_REL}; it holds at that commit as long as the changed paths and their diffs since the seg phase are these (runs/ is listed only, so the commit that adds it changes no forbidden path); "
          f"diff_sha256 = sha256 of the bytes of `git {' '.join(DRIFT_DIFF_ARGS)} <seg_code_sha> <commit> -- <path>`", file=sys.stderr)
    for note in notes:
        print(f'NOTE: {note}', file=sys.stderr)
    blocking = document is None or not document['entries'] or any('may not be approved' in note for note in notes)
    return 8 if blocking else 0


def queue(o, system=None):
    problem = HS.root_problem(o.root)
    if problem:                                         # before the lock file: nothing is created or touched
        print(problem, file=sys.stderr)
        return 2
    if getattr(o, 'segcheck_reread', None) is not None and o.phase != 'seg':
        print('--segcheck-reread goes with --phase seg', file=sys.stderr)
        return 2

    def guarded(o, system, root, log):
        try:
            return BODIES[o.phase](o, system, root, log)
        except SystemExit as refused:                   # a refusal met on the way: a STOP with its record
            raise run.Stop(str(refused))
    return run.queue(o, system, body=guarded, prefix='hs')


def queue_launch(phase, root=None):
    """The lead's start string of a phase."""
    return run.check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT}', '--collect', f'--working-directory={core.MAIN_REPO}', f'--property=MemoryMax={BASE.MEMORY_MAX}', f'--nice={BASE.NICE}', '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1',
                             run.PY, '-B', SCRIPT, 'queue', '--phase', phase, '--root', str(HS.DEFAULT_ROOT if root is None else root), '--gap-seconds', str(BASE.GAP_SECONDS), '--min-host-gib', f'{BASE.DEFAULT_MIN_HOST_GIB:g}'])


# ---- estimates and the dry run --------------------------------------------------------------------------------------------------------------------
def estimates(results=None):
    """〔估算〕from rw_v1's repairs and fd2_v1's Hx24 fields (the same configuration on v1's channels); None where there is no record.  Nobody has run HS."""
    rw_root = Path(fx.RESULTS if results is None else results)/fx.RW.ROOT_NAME
    fd2_root = fx.RW.fd2_root(results)
    units, fields, evals = [], [], []
    for pair in ALL_PAIRS:
        done = run.read_json(fx.RW.run_dir(rw_root, fx.RW.cell_of(pair))/'queue_done.json') or {}
        if done.get('unit_seconds') is not None:
            units.append(done['unit_seconds'])
        d = fxrun.fd2.run_dir(fd2_root, fx.RW.fd2_field_cell(pair))
        f = run.read_json(d/'queue_done.json') or {}
        if f.get('unit_seconds') is not None:
            fields.append(f['unit_seconds'])
        m = run.read_json(d/f'metrics_{hc.NATIVE:04d}.json') or {}
        if m.get('seconds_certificate') is not None and m.get('seconds_metrics') is not None:
            evals.append(m['seconds_certificate']+m['seconds_metrics'])
    gap = BASE.GAP_SECONDS
    out = dict(n_rw=len(units), n_fd2=len(fields), gate_wait_seconds=fxrun.GATE_WAIT_SECONDS, infer_seconds_per_subject='〔估算〕about 1 min a subject: unet35_v1 measured 55-64 s with 4 members and the compressed writing (04_分割器.md §14.8); UF\'s plan estimated its evaluation of 20 subjects (4 members, U1, the nnU-Net read-back) at about 15 min; v3_large has not been timed here')
    if len(units) == len(fields) == len(evals) == len(ALL_PAIRS):
        e = sum(evals)/len(evals)
        out.update(fields_seconds=sum(fields)+18*gap, repairs_seconds=sum(units)+18*gap, evaluations_seconds=2*19*e+2*fxrun.GATE_WAIT_SECONDS, total_seconds=sum(fields)+sum(units)+36*gap+2*19*e+2*fxrun.GATE_WAIT_SECONDS, fuse_cell_seconds=fx.FUSE_SECONDS+15.)
    return out


def _minutes(seconds):
    return '—' if seconds is None else f'{seconds/60:.0f} min'


def fields_check_lines(out, candidate, *, stage=None, allowlist=None, results=None):
    """The dry run's part of the fields queue (HSF: the dry run did not call fields_input_problems, so its KeyError was first met by the queue).  Once the seg phase is done (infer_record.json and segcheck.json passed) and the registry
    committed (tracked and clean, structure_layer.registry_git_state), fields_input_problems as the queue runs it before the checkout: each line and each problem printed, the problems returned.  Before that, what it waits for, not a problem.  Nothing written."""
    sd = hc.seg_dir(candidate, HS.RESULTS)
    infer, check = run.read_json(sd/hc.INFER_RECORD) or {}, run.read_json(sd/hc.SEGCHECK_RECORD) or {}
    waiting = []
    if infer.get('passed') is not True or check.get('passed') is not True:
        waiting.append(f"等分割：seg 阶段没有做完或控制没过（infer_record.json passed {infer.get('passed')}，segcheck.json passed {check.get('passed')}）")
    path = HS.registry_path()
    from bayesreg.seg.structure_layer import registry_git_state
    git = registry_git_state(path) if path.is_file() else {}
    if not (git.get('tracked') is True and git.get('clean') is True):
        waiting.append(f"等登记：{hc.REGISTRY_REL} 不在或没有提交（there {path.is_file()}, tracked {git.get('tracked')}, clean {git.get('clean')}）")
    if waiting:
        for line in waiting:
            out(f'{line}（不算问题：还没到场的输入核对）')
        return []
    out(f"--- the fields queue's input check (fields_input_problems of {candidate}, as the queue runs it before the checkout; through the firewall's reader, no label, nothing written):")
    problems = fxrun.safe(fields_input_problems, lambda text: out(f'[fields inputs] {text}'), candidate=candidate, stage=stage, allowlist=allowlist, results=results)
    for problem in problems:
        out(f'[fields inputs] problem: {problem}')
    out(f"--- fields_input_problems: {len(problems)} problem(s){': ' if problems else ' (none)'}{problems[:3] if problems else ''}")
    return problems


def dry_run(*, out=print, root=None, results=None, stage=None, allowlist=None):
    """Both candidates' plans, what HS waits for, the checks that need nothing of UF, the estimates, the start strings.  No truth label, no optimization, nothing written."""
    root = HS.DEFAULT_ROOT if root is None else Path(root)
    out(f'HS dry run ({run.utc()}): no optimization, no truth label, nothing written; of UF\'s root only whether its files are there; real files read: the npy headers of fixed and moving, JSON records, and for their sha256 the 19 fields of configuration C')
    gate = BASE.GATE
    out(f'root {root}; root exists: {root.exists()}; cap per run {fl.RUN_CAP_SECONDS} s; repair wall_seconds {fx.FUSE_SECONDS:g} s (a fuse), RuntimeMaxSec {fx.REPAIR_RUNTIME_SECONDS}; device {mc.VRAM_LIMIT_BYTES/2**30} GiB; MemoryMax {BASE.MEMORY_MAX}, nice {BASE.NICE}; {BASE.GAP_SECONDS} s between runs')
    out(f'gate: own units {gate.own_units}, load < {gate.load_limit}, GPU free >= {gate.gpu_free_mib} MiB, MemAvailable >= {gate.mem_available_gib} GiB, /home >= {run.HOME_FLOOR_GB} GB, /mnt/c >= {gate.host_floor_gib} GiB; closed while any other tag-* or seq-vp-* unit is active, so while tag-uf-chain runs')
    waiting, stop, candidate, _ = hc.uf_reading(HS.UF_OUT)
    for line in waiting:
        out(f'{line}（不算问题）')
    if stop:
        out(f'{stop}（不算问题：HS 不跑）')
    if candidate is not None:
        out(f'UF 的候选：{candidate}')
    uf_lines(out)
    for key in hc.CANDIDATES:
        candidate_lines(out, key)
    out(f'the arm: {hc.HS_ARM} = {json.dumps(dict(hc.SPEC))}, the frozen Hx24 (fd2_core.ARMS["Hx24"], fx_core.SPEC); the repair {HS.REPAIR_ARM}: RF05 with wall_seconds {fx.FUSE_SECONDS:g}; the evaluation: configuration C, alpha = 1')
    fxrun.print_cells(out, HS.plan_all_fields(), 'field')
    fxrun.print_cells(out, HS.plan_all_repairs(), 'repair')
    problems = fxrun.safe(lambda o, **kw: fxrun.stage_problems(o, ALL_PAIRS, **kw), out, stage=stage, allowlist=allowlist)
    problems += fxrun.safe(lambda o: fxrun.frozen_problems(o), out)
    problems += fxrun.safe(lambda o, **kw: rw_lines(o, **kw), out, results=results)
    statuses = fxrun.safe(lambda o, **kw: fallback_lines(o, **kw), out, results=results)
    if isinstance(statuses, list):
        problems += statuses
    from bayesreg.seg.unet.snapshot import CodeNotClean
    try:
        subjects, split, _, _ = seg_inputs()
        out(f"the seg phase's subjects (the pinned allowlist and folds_full_v2.json, ids only): {len(subjects)}, {subjects[0]} .. {subjects[-1]}; folds_full_v2.json sha256 {split.sha256}")
        if len(subjects) != hc.N_SUBJECTS:
            problems.append(f'{len(subjects)} development subjects, not {hc.N_SUBJECTS}')
    except CodeNotClean as error:
        out(f"the seg phase's pinned inputs are refused here because this tree is not clean ({str(error)[:160]}): the units run from a clean checkout; not a problem of a dry run")
    from bayesreg.seg.structure_layer import FORMAL_REGISTRIES, is_formal_registry
    formal = is_formal_registry(HS.registry_path())
    out(f'{hc.REGISTRY_REL} one of structure_layer.FORMAL_REGISTRIES: {formal}; there: {HS.registry_path().is_file()}')
    if not formal:
        problems.append(f'{hc.REGISTRY_REL} is not in FORMAL_REGISTRIES {[str(p) for p in FORMAL_REGISTRIES]}')
    nn_train = run.read_json(hc.NN_OUT/'train_record.json') or {}
    out(f"nnU-Net for (b): {hc.NN_OUT}/train_record.json checkpoint sha256 {nn_train.get('final_checkpoint_sha256')} (not re-hashed here); prepare_manifest.json there {(hc.NN_OUT/'prepare_manifest.json').is_file()}")
    if candidate is not None:                           # HSF: the fields queue's input check, once the seg phase is done and the registry committed
        problems += fields_check_lines(out, candidate, stage=stage, allowlist=allowlist, results=results)
    e = estimates(results)
    out(f"estimates〔估算, from fd2_v1's Hx24 fields and rw_v1's repairs on v1's channels; nobody has run HS〕: the seg phase: inference {e['infer_seconds_per_subject']}, 20 subjects; the control about 10 min〔估算: eval_seg reads the truth and the probabilities of 20 subjects〕; the fields {_minutes(e.get('fields_seconds'))}, the repairs {_minutes(e.get('repairs_seconds'))}, the evaluations {_minutes(e.get('evaluations_seconds'))}, all of the fields phase {_minutes(e.get('total_seconds'))}; a repair at the fuse {_minutes(e.get('fuse_cell_seconds'))}; the fallback units only if a pair needs them")
    out(f'/home free {shutil.disk_usage("/home").free/2**30:.0f} GB; /mnt/c free {shutil.disk_usage("/mnt/c").free/2**30:.0f} GiB')
    out('--- start strings (the queues: the lead, after UF\'s reading; the others: the queue, one per unit; seg at $ROOT/seg/code, fields at $ROOT/code)')
    out(shlex.join(queue_launch('seg', root)))
    seg_code = root/hc.SEG_CODE_DIR/'code'
    out(shlex.join(seg_unit_argv(INFER_UNIT, seg_code, ['infer-one', '--code-sha', '<sha>'], log=root/'infer_unit.log', gpu=True, runtime=INFER_RUNTIME)))
    out(shlex.join(seg_unit_argv(SEGCHECK_UNIT, seg_code, ['segcheck', '--code-sha', '<sha>'], log=root/'segcheck_unit.log', gpu=False, runtime=SEGCHECK_RUNTIME)))
    out(f'{run.PY} -B {SCRIPT} register        # by hand, after the independent review; commit {hc.REGISTRY_REL}')
    out(shlex.join(queue_launch('fields', root)))
    code = root/'code'
    for c in HS.all_cells(root):
        out(shlex.join(BASE.run_argv(c, code, root, '<sha>')))
    out(shlex.join(BASE.eval_argv(code, root, BASE.EVAL_UNIT)))
    out(shlex.join(BASE.eval_argv(code, root, BASE.EVAL_R1_UNIT)))
    out(shlex.join(BASE.unit_argv(FALLBACK_UNIT, code, ['fallback', '--root', str(root)], log=root/'fallback.log', gpu=False, numba=BASE.EVAL_THREADS, quota=BASE.EVAL_QUOTA, runtime=FALLBACK_RUNTIME)))
    out(shlex.join(BASE.unit_argv(ID_EVAL_UNIT, code, ['id-evaluate', '--root', str(root)], log=root/'id_eval.log', gpu=False, numba=BASE.EVAL_THREADS, quota=BASE.EVAL_QUOTA, runtime=run.EVAL_RUNTIME_MAX)))
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


# ---- the entries -----------------------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv == ['--dry-run']:
        return dry_run()
    p = argparse.ArgumentParser(description='HS: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--phase', choices=tuple(BODIES), required=True)
    q.add_argument('--root', default=str(HS.DEFAULT_ROOT))
    q.add_argument('--commit', default=None)
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--gap-seconds', type=int, default=BASE.GAP_SECONDS)
    q.add_argument('--min-host-gib', type=float, default=BASE.DEFAULT_MIN_HOST_GIB)
    q.add_argument('--segcheck-reread', default=None, help='phase seg only: the lead\'s words and the time, approving another read of the development truth by the control after a launch that left no segcheck.json (recorded in segcheck_rereads.json)')
    for command in ('run-one', 'repair-one'):
        r = sub.add_parser(command)
        for name in ('--arm', '--tier', '--pair', '--out'):
            r.add_argument(name, required=True)
        r.add_argument('--device', default='cuda')
        r.add_argument('--code-sha', default='n/a')
    for command in ('infer-one', 'segcheck'):
        r = sub.add_parser(command)
        r.add_argument('--code-sha', default='n/a')
        r.add_argument('--device', default='cuda')
    for command in ('evaluate', 'fallback', 'id-evaluate', 'decide'):
        sub.add_parser(command).add_argument('--root', default=str(HS.DEFAULT_ROOT))
    sub.add_parser('register')
    sub.add_parser('predict')
    sub.add_parser('write-reference').add_argument('--root', default=str(HS.DEFAULT_ROOT))
    a = sub.add_parser('drift-approval', help=f'print the {DRIFT_APPROVAL_REL} the lead would commit for HEAD (HSF); writes nothing')
    a.add_argument('--review', required=True, help='the repository-relative path of the independent review of the change')
    a.add_argument('--root', default=str(HS.DEFAULT_ROOT))
    o = p.parse_args(argv)
    if o.command == 'drift-approval':
        return print_drift_approval(o)
    if o.command == 'queue':
        return queue(o)
    if o.command == 'write-reference':
        return write_reference(o)
    if o.command == 'run-one':
        return run_one(o)
    if o.command == 'repair-one':
        return repair_one(o)
    if o.command == 'infer-one':
        return infer_one(o)
    if o.command == 'segcheck':
        return segcheck_one(o)
    if o.command == 'register':
        return register(o)
    if o.command == 'predict':
        candidate = HS.candidate()
        reference = decision.rw_reference()
        print(json.dumps(decision.expectation(run.read_json(hc.uf_paths(HS.UF_OUT)['eval']), candidate, decision.mean(reference[q]['metrics']['seg35'] for q in ALL_PAIRS if reference[q]['metrics'])), indent=1, ensure_ascii=False))
        return 0
    problem = HS.root_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if o.command == 'evaluate':
        return BASE.evaluate_all(o)
    if o.command == 'fallback':
        return fallback_unit(o)
    if o.command == 'id-evaluate':
        return id_evaluate(o)
    problem = reading_code_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 8
    decision.write_hs(o.root, stops=BASE.ledger(Path(o.root))['arms_stopped'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
