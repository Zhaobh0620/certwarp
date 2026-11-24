"""FX (runs/20261105-fx/任务书.md, plan.md): the drivers of RW, OX and NX on the line of Hx24@600.  Not a method: no F_B, no certified descent, only the final field passes the certificate; development pairs only, in no result table.  The lead starts each batch, one at a time, after nnU-Net's training and inference; the gate of every queue is closed while tag-nn-* (any other tag-* or seq-vp-* unit) is active.
Batches: fx_core.py.  Readings: decide_fx.py.  nnU-Net's probabilities and the registration: nx_seg.py.  One run, one repair, one evaluation, the phases: run_fl.py's own functions, not copied: another instance of that module per batch, as run_fd2.py does.  fd_core, decide_fd, fd2_core, decide_fd2, fl_core, decide_fl, run_fl, run_fd, run_fd2 are not touched.

    python -B experiments/fl/run_fx.py rw|ox|nx --dry-run
        the plan, the estimates, the checks of the inputs, the start strings.  No optimization, no truth label, torch not imported, nothing written.  NX without nnU-Net's weights says 等 NN, which is not a problem.
    run_fx.py rw queue [--commit SHA] [--wait-minutes 30] [--gap-seconds 5] [--min-host-gib 15]
        the control (0439-0440), its repaired field against fd2_v1's bit for bit, the code's judgment written once; the 5 pairs that hit the 120 s wall; the other 13 only when the control was not bit for bit (else fd2_v1's repairs stand); tag-rw-eval; the reading.
    run_fx.py ox queue [...]                5 runs of O_Hx24, tag-ox-eval, the reading.  Truth only through fl_oracle.prepare_hx; root ox_v1 only.
    run_fx.py nx queue --phase infer [...]  NN's reading must allow it.  tag-nx-infer (nnU-Net with --save-probabilities, the environment nnunet, 20 development subjects), tag-nx-convert (probs36 and the gate).
    run_fx.py nx register                   by hand, after the gate: configs/bayesreg/segmenters_nn935_v1.json; the lead commits it.
    run_fx.py nx queue --phase fields [...] NHx24 on the first 5 pairs, tag-nx-eval, repairs, tag-nx-eval-r, the code's rule of the other 14, their fields and repairs if it says so, the reading.
    run_fx.py <batch> run-one|repair-one --arm A --tier T --pair P --out DIR [--device cuda] [--code-sha SHA]      one unit of a queue
    run_fx.py <batch> evaluate              what has no metrics yet; the only place besides OX's channels where truth labels are read
    run_fx.py <batch> decide                the reading again

Each batch writes only under its own root (seg/nn935_v1 for NX's predictions); fd_v1, fd2_v1, fl_v1, nnunet_v1 are only read.  A run directory that exists is never rerun, failures are kept.  Exit codes: 0 done, 2 usage, another driver on the root, another root, 8 STOP or a failed check.
"""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent/'ef'))
sys.path.insert(2, str(HERE.parent/'mc0'))
sys.path.insert(3, str(HERE.parent/'nn'))

import decide_fx as decision  # noqa: E402
import fd2_core as fd2  # noqa: E402
import fl_core as fl  # noqa: E402
import fx_core as fx  # noqa: E402
from fx_core import ALL_PAIRS, NX, OX, RW  # noqa: E402
import mc0_core as core  # noqa: E402
import nn_common as nc  # noqa: E402
import nx_seg  # noqa: E402

SCRIPT = 'experiments/fl/run_fx.py'
GATE_WAIT_SECONDS = 90.                         # 〔估算〕after an evaluation unit (numba 8), FD's queue.log
SEGMENTER_MEMORY_MAX, INFER_RUNTIME, CONVERT_RUNTIME = '8G', 7200, 7200
INFER_DIR = 'infer'                             # NX step 1's checkout: $ROOT/infer/code; step 3's is $ROOT/code
INFER_PEAK_GIB = 9.


class RepairWithFuse:
    """experiments/ef/repair.py with wall_seconds 1200 and nothing else changed: run_fl.repair_one asks this name for its RepairConfig."""

    def __init__(self, module, wall):
        self._module, self._wall = module, wall

    def __getattr__(self, name):
        return getattr(self._module, name)

    def RepairConfig(self, **kw):
        return self._module.RepairConfig(**dict(kw, wall_seconds=self._wall))


def _instance(batch):
    """run_fl once more under another name, its module names pointed at this batch.  sys.modules['run_fl'] and the others' instances are separate module objects."""
    p = batch.PREFIX
    spec = importlib.util.spec_from_file_location(f'_{p}_run_fl', HERE/'run_fl.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.fl, module.decision, module.SCRIPT, module.QUEUE_UNIT = batch, decision, SCRIPT, f'tag-{p}-queue'
    module.EVAL_UNIT, module.EVAL_R1_UNIT, module.EVAL_X_UNIT, module.EVAL_R2_UNIT, module.EVAL_REST_UNIT, module.EVAL_REST_R_UNIT = f'tag-{p}-eval', f'tag-{p}-eval-r', f'tag-{p}-eval-x', f'tag-{p}-eval-r2', f'tag-{p}-eval-rest', f'tag-{p}-eval-rest-r'
    module.eval_argv.__defaults__ = (module.EVAL_UNIT,)
    module.GATE = replace(module.GATE, own_units=(f'{module.QUEUE_UNIT}.service',))
    run = module.run

    def unit_argv(unit, code, inner, *, log, gpu=True, numba=module.RUN_THREADS, quota=800, runtime=run.RUN_RUNTIME_MAX):
        return run.unit_argv(unit, code, [p, *inner], log=log, gpu=gpu, numba=numba, quota=quota, runtime=runtime, script=SCRIPT, memory_max=module.MEMORY_MAX, nice=module.NICE)

    def run_argv(cell, code, root, sha='n/a'):
        d = batch.run_dir(root, cell)
        command = 'repair-one' if cell.is_repair else 'run-one'
        return unit_argv(cell.unit, code, [command, '--arm', cell.arm, '--tier', cell.tier, '--pair', cell.pair, '--out', str(d), '--code-sha', sha], log=d/'log.txt', numba=module.REPAIR_THREADS if cell.is_repair else module.RUN_THREADS, quota=module.REPAIR_QUOTA, runtime=fx.REPAIR_RUNTIME_SECONDS if cell.is_repair else run.RUN_RUNTIME_MAX)
    module.unit_argv, module.run_argv = unit_argv, run_argv
    return module


RW_BASE, OX_BASE, NX_BASE = _instance(RW), _instance(OX), _instance(NX)
BASES = dict(rw=RW_BASE, ox=OX_BASE, nx=NX_BASE)
run = RW_BASE.run
mc = RW_BASE.mc
channels = RW_BASE.channels
READING_FILES = ((Path(decision.__file__), 'experiments/fl/decide_fx.py'), (Path(fx.__file__), 'experiments/fl/fx_core.py'), (Path(decision.fd2_reading.__file__), 'experiments/fl/decide_fd2.py'), (Path(fd2.__file__), 'experiments/fl/fd2_core.py'), (Path(decision.fl_reading.__file__), 'experiments/fl/decide_fl.py'), (Path(fl.__file__), 'experiments/fl/fl_core.py'),
                 (HERE/'nx_seg.py', 'experiments/fl/nx_seg.py'), (HERE/'fl_oracle.py', 'experiments/fl/fl_oracle.py'), (HERE/'rw_sources_v1.json', 'experiments/fl/rw_sources_v1.json'))     # the last three: FX review, minor 3


# ---- RW: its pins, its source, its rule ---------------------------------------------------------------------------------------------------------
def rw_pins_from_records(results=None):
    """The pins of rw_sources_v1.json, from fd2_v1's JSON records (no field is read)."""
    root = RW.fd2_root(results)
    pairs = {}
    for pair in ALL_PAIRS:
        d = fd2.run_dir(root, RW.fd2_field_cell(pair))
        r = fd2.run_dir(root, RW.fd2_repair_cell(pair))
        checkpoint = run.read_json(d/'run.json')['checkpoints'][str(fx.NATIVE)]
        metrics, repaired, repaired_metrics = run.read_json(d/f'metrics_{fx.NATIVE:04d}.json'), run.read_json(r/'repair.json'), run.read_json(r/'repair_metrics.json')
        pairs[core.short_name(pair)] = dict(field_file=checkpoint['file'], field_sha256=checkpoint['sha256'], u_sha256=checkpoint['u_sha256'], tapered_u_sha256=metrics['tapered_u_sha256'], seconds_task=checkpoint['seconds_task'], old_attempt_sha256=repaired['attempt']['sha256'], old_repaired_u_sha256=repaired_metrics['repaired_u_sha256'], old_reason=repaired['reason'], old_wall_hit=decision.hit_wall(repaired), old_rounds=len(repaired['rounds']), old_accepted_alpha=repaired_metrics['after']['certificate']['accepted_alpha'])
    return dict(schema='bayesreg.rw.sources.v1', source='fd2_v1, Hx24@600 and its repairs with the 120 s wall, read only', pairs=pairs)


def rw_source_of_repair(cell, root=None, results=None):
    """run_fl.source_of_repair for RW: the field of fd2_v1, read only, pinned.  run_fl.source_checked then checks the file, the taper and the marks."""
    results = RW.RESULTS if results is None else results
    d = fd2.run_dir(RW.fd2_root(results), RW.fd2_field_cell(cell.pair))
    record = run.read_json(d/'run.json')
    checkpoint = ((record or {}).get('checkpoints') or {}).get(str(fx.NATIVE))
    if checkpoint is None:
        raise SystemExit(f'{d}: no checkpoint at {fx.NATIVE} steps')
    pin = RW.sources()[cell.short]['field_sha256']
    if checkpoint['sha256'] != pin:
        raise SystemExit(f"{d}: the field's sha256 {checkpoint['sha256']} is not the pin {pin} of {RW.SOURCES_FILE}")
    return d, checkpoint['file'], checkpoint['sha256'], f'metrics_{fx.NATIVE:04d}.json', record, checkpoint['seconds_task']


def rw_repair_allowed(cell, root):
    """The control always; the 5 pairs after the code judged the control; the other 13 only when it was not bit for bit."""
    if cell.pair == RW.CONTROL_PAIR:
        return True
    choice = RW.control_choice(root)
    return choice is not None and (cell.pair in RW.WALL_PAIRS or choice['same'] is False)


RW_BASE.repair = RepairWithFuse(RW_BASE.repair, fx.FUSE_SECONDS)
RW_BASE.source_of_repair = rw_source_of_repair
RW_BASE.repair_allowed = rw_repair_allowed


def repair_one(base, o):
    """run_fl.repair_one, then the name of the status: the .9^k rule did not run, and the record says so."""
    code = base.repair_one(o)
    path = Path(o.out)/'repair.json'
    record = run.read_json(path)
    run.write_json(path, dict(record, status=fx.CERTIFIED if record['passed'] else fx.NOT_CERTIFIED, status_in_run_fl=record['status'], fallback_rule_run=False, wall_hit=decision.hit_wall(record)))
    return code


# ---- OX: the oracle's reader and its scale ----------------------------------------------------------------------------------------------------------
class OracleHx:
    """What run_fl.run_field asks of its oracle module: fl_oracle.prepare_hx on this arm's numbers."""

    def prepare(self, pair, device, workers=1):
        import fl_oracle
        return fl_oracle.prepare_hx(pair, device, dict(OX.ARMS[fx.OX_ARM], channels='onehot'), core.load_table(), workers=workers)


OX_BASE.oracle_arm = OracleHx()


def ox_run_one(o):
    """run_fl.run_one, then the check of the scale it leaves to core.prepare_stacks in FL's oracle arm: here the numbers are fl_channels' and the check is the same as Hx24's (||g_D|| of the native tensors = 24 x the truth's Q0)."""
    code = OX_BASE.run_one(o)
    path = Path(o.out)/'run.json'
    record = run.read_json(path)
    if record.get('optimizer') is not None:
        check = channels.scale_check_of(record['optimizer'], record['scale'])
        changes = dict(scale_check=check)
        if record['status'] == 'done' and check['passed'] is not True:
            changes.update(status='scale_mismatch', reason=f"||g_D|| of the native tensors {check['g_norm_scaled']} differs from {check['target']}")
        run.write_json(path, dict(record, **changes))
    return code


def ox_control_problem(cell, record, results=None):
    """After every run of the queue: the mark and the scale.  A mismatch stops the queue: every other pair would have the same rule."""
    if record is None or record.get('status') in ('cuda_oom', 'vram_stop', 'nonfinite', 'time_cap'):
        return None
    check = record.get('scale_check')
    if record.get('status') == 'scale_mismatch' or not isinstance(check, dict) or check.get('passed') is not True:
        return f"{cell.run_id}: the scale: {check}, status {record.get('status')}"
    return None


OX_BASE.control_problem = ox_control_problem


# ---- NX: its source of channels ---------------------------------------------------------------------------------------------------------------------
def nx_source():
    """The predictions of nn935_v1 under the structure layer's checks: a registered segmenter, its registry committed and clean (hard rule 7)."""
    mi = core.make_inputs()
    return mi.UNetSource(pred_root=NX.PRED_ROOT, models_root=NX.SEG_DIR, image_root=nc.DATA_ROOT, segmenter_id=NX.SEGMENTER_ID, registry=NX.registry_path(), gate=dict(rule='runs/20261105-fx/任务书.md: the reading of NN lets NX run'))


def nx_oracle_mark(keys_read, sides):
    """mc0p_core.oracle_mark, computed from what the channels were read from, with this batch's registered segmenter in the place of unet35_v1_ens: that function names one segmenter, and would mark these predictions oracle."""
    other = sorted(set(keys_read)-set(mc.IMAGE_KEYS))
    images = {'fixed', 'moving'} <= set(keys_read)
    predictions = set(sides) == {'fixed', 'moving'} and all(s.get('segmenter_id') == NX.SEGMENTER_ID and s.get('policy') == mc.POLICY for s in sides.values())
    return dict(oracle=not (images and predictions and not other), truth_labels_read=bool(other))


def nx_oracle_mark_checked(cell, record):
    """run_fl.oracle_mark_checked for NX: the mark again from the run's channel_sources; ValueError otherwise, and the field is not read."""
    sources = record.get('channel_sources') or {}
    mark = nx_oracle_mark(sources.get('pair_keys_read') or [], sources.get('sides') or {})
    if record.get('oracle') is not False or mark['oracle'] is not False:
        raise ValueError(f"run.json oracle {record.get('oracle')}; computed from its channel_sources: {mark}")
    return False


class NXFast:
    """What run_fl.run_field asks of fast_channels, read_predicted on nn935_v1's source, its mark computed for that segmenter."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def read_predicted(self, pin, *, workers=1):
        predicted = self._module.read_predicted(pin, unet=nx_source(), workers=workers)
        predicted.record.update(nx_oracle_mark(predicted.record['pair_keys_read'], predicted.record['sides']))
        return predicted


def nx_repair_allowed(cell, root):
    if cell.tier.startswith('c'):
        return True
    choice = NX.rest_choice(root)
    return choice is not None and choice['go'] is True


NX_BASE.repair = RepairWithFuse(NX_BASE.repair, fx.FUSE_SECONDS)
NX_BASE.fast = NXFast(NX_BASE.fast)
NX_BASE.repair_allowed = nx_repair_allowed
NX_BASE.oracle_mark_checked = nx_oracle_mark_checked


def nx_run_one(o):
    cell = NX.require_cell(o.arm, o.tier, o.pair)
    if cell.tier != NX.FIRST and not nx_repair_allowed(cell, NX.DEFAULT_ROOT):
        raise SystemExit(f'{cell.run_id}: the code has not opened the other 14 pairs (analysis/{NX.REST_FILE})')
    return NX_BASE.run_one(o)


def nn_root(results=None):
    return Path(fx.RESULTS if results is None else results)/nc.OUT.name


def nn_reading(results=None):
    return run.read_json(nn_root(results)/'analysis/decision_nn.json')


def dev_subjects(allowlist=None, results=None):
    """The 20 development subjects of the 19 pairs, each checked before anything of them is opened.  Ids only: no image, no label."""
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.io.guard import SubjectSealed, load_dev_allowlist
    from bayesreg.seg.unet.data import load_folds
    allow = load_dev_allowlist(core.ALLOWLIST if allowlist is None else allowlist)
    subjects = sorted({s for pair in ALL_PAIRS for s in pair.split('__')})
    source = {s['name']: s for s in allow.sources}.get('stage_L2R_val_v1') or {}
    if sorted(source.get('subjects') or []) != subjects:
        raise SubjectSealed('the subjects of the 19 pairs are not the allowlist source stage_L2R_val_v1')
    sealed, training = load_sealed_lists(core.ROOT), load_folds(core.make_inputs().FOLDS).training_subjects
    for subject in subjects:
        which = sealed.which(subject)
        if which is not None or subject in allow.never_for_training or subject in training or not allow.allows(subject):
            raise SubjectSealed(f'{subject}: on the sealed list {which}, or never_for_training, or a training subject, or off the allowlist')
    return subjects


def nx_infer_argv(train, *, out=None):
    """infer_nn.py in the environment nnunet, with --save-probabilities.  The argmax of this run is compared with pred_dev in nx_convert."""
    out = nc.OUT if out is None else out
    argv = nc.infer_argv(model_dir=nc.trainer_directory(train['trainer'], results=nc.results_root(out)), input_dir=NX.SEG_DIR/'dev_input', out_dir=NX.SEG_DIR/'raw')+['--save-probabilities']
    return argv[:3]+[str(HERE/'nx_infer_child.py')]+argv[3:]          # the child sets the 9 GiB device cap, then runs infer_nn.py unchanged


FALLBACK_LINE = 'Prediction on device was unsuccessful'      # nnunetv2 2.8.1, predict_from_raw_data.py: the first out-of-memory error moves the results to the CPU


def infer_fallbacks(text):
    """(lines, subjects): where nnU-Net fell back to the CPU for the results of a subject.  infer_nn.py prints `<subject>: <seconds> s` after each subject, so a fallback line belongs to the next such line."""
    lines, subjects, pending = 0, [], False
    for line in text.splitlines():
        if FALLBACK_LINE in line:
            lines, pending = lines+1, True
            continue
        head, _, tail = line.partition(': ')
        if pending and head.startswith('OASIS_') and tail.endswith(' s'):
            subjects.append(head)
            pending = False
    return dict(n_lines=lines, fallback_subjects=subjects)


def nx_infer_one(o):
    """The unit tag-nx-infer.  The links to the images of the 20 subjects, then nnU-Net as a child in its own environment.  nnunet_v1 and seg/nn_dev_read.lock are not written."""
    train = run.read_json(nc.OUT/'train_record.json')
    if train is None:
        raise SystemExit('等 NN: no train_record.json')
    if nx_seg.file_sha256(train['final_checkpoint']) != train['final_checkpoint_sha256']:
        raise SystemExit(f"{train['final_checkpoint']}: sha256 differs from train_record.json")
    subjects = dev_subjects()
    inputs = NX.SEG_DIR/'dev_input'
    if inputs.exists():
        raise SystemExit(f'{inputs} exists: the inference has started before; never rerun')
    inputs.mkdir(parents=True)
    for subject in subjects:
        (inputs/f'{subject}_0000.nii.gz').symlink_to(nc.DATA_ROOT/subject/nx_seg.IMAGE_FILE)
    with open(NX.SEG_DIR/'infer.log', 'ab') as log:
        code = subprocess.run(nx_infer_argv(train), env=nc.nn_env(nc.OUT, gpu=True), cwd=NX.SEG_DIR, stdout=log, stderr=subprocess.STDOUT).returncode
    record = run.read_json(NX.SEG_DIR/'raw/infer_record.json') or {}
    fallback = infer_fallbacks((NX.SEG_DIR/'infer.log').read_text(errors='replace'))
    run.write_json(NX.SEG_DIR/'infer_check.json', dict(schema='bayesreg.nx.infer_check.v1', returncode=code, device_fallback=bool(fallback['n_lines']), **fallback, gpu_peak_allocated_gib=record.get('gpu_peak_allocated_gib'), gpu_peak_reserved_gib=record.get('gpu_peak_reserved_gib'), device_cap_gib=INFER_PEAK_GIB, utc=run.utc()))
    if code != 0:
        print(f'infer_nn.py exit {code}; kept, not rerun', file=sys.stderr)
        return 8
    if record.get('checkpoint_sha256') != train['final_checkpoint_sha256'] or record.get('saved_probabilities') is not True or sorted(record.get('subjects') or []) != subjects or record.get('allowed_mirroring_axes') is not None or record.get('tta_views') != 1 or (record.get('gpu_peak_reserved_gib') or 0.) > INFER_PEAK_GIB:
        print(f"the record of this inference is not the plan's: {dict((k, record.get(k)) for k in ('checkpoint_sha256', 'saved_probabilities', 'allowed_mirroring_axes', 'tta_views', 'gpu_peak_reserved_gib'))}", file=sys.stderr)
        return 8
    return 0


def nx_convert(o):
    """The unit tag-nx-convert, one subject at a time.  A subject that is there is not done again; the gate stops the rest."""
    from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping, require_inference_grouping
    from jd_curl.three_d.nifti import load_scalar
    train = run.read_json(nc.OUT/'train_record.json')
    subjects = dev_subjects()
    grouping = load_grouping(DEFAULT_GROUPS_PATH, 'structure13_v1')
    check = require_inference_grouping(grouping)
    done = NX.SEG_DIR/'convert'
    done.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        if (done/f'{subject}.json').is_file():
            continue
        image = nc.DATA_ROOT/subject/nx_seg.IMAGE_FILE
        volume = load_scalar(image)
        grid = dict(shape_zyx=list(volume.grid.shape), spacing_xyz=list(volume.grid.spacing), origin_xyz=[float(a[0]) for a in volume.grid.axes])
        try:
            record = nx_seg.convert_subject(subject, raw_dir=NX.SEG_DIR/'raw', pred_dev=nc.OUT/'pred_dev', image_path=image, out_root=NX.PRED_ROOT, grouping=grouping, grouping_check=check, input_sha256=volume.metadata['sha256'], grid=grid, model_id=NX.MODEL_ID, segmenter_id=NX.SEGMENTER_ID, fold=NX.FOLD, checkpoint_sha256=train['final_checkpoint_sha256'], trainer=train['trainer'], code_sha=o.code_sha, registry_rel=NX.REGISTRY_REL, log=print)
        except nx_seg.GateFailed as failed:
            run.write_json(NX.SEG_DIR/'convert_stop.json', dict(subject=subject, reason=str(failed), utc=run.utc()))
            print(f'STOP: the gate: {failed}', file=sys.stderr)
            return 8
        run.write_json(done/f'{subject}.json', record)
    rows = {subject: run.read_json(done/f'{subject}.json') for subject in subjects}
    run.write_json(NX.SEG_DIR/'convert_record.json', dict(schema='bayesreg.nx.convert.v1', gate='passed', rule='the argmax of the stored probs36 against pred_dev, voxels that differ / all voxels <= the tolerance, per subject', tolerance=nx_seg.GATE_TOLERANCE, subjects=subjects, n_differing_by_subject={s: rows[s]['n_differing'] for s in subjects}, n_voxels_by_subject={s: rows[s]['n_voxels'] for s in subjects}, n_differing_total=sum(rows[s]['n_differing'] for s in subjects), n_voxels_total=sum(rows[s]['n_voxels'] for s in subjects), code_sha=o.code_sha, utc=run.utc()))
    return 0


def nx_register(o):
    """By hand, after the gate: the file, once; the lead commits it, the queue of the fields asks for it tracked and clean."""
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.io.guard import load_dev_allowlist
    from bayesreg.seg.unet.data import load_folds
    train, manifest = run.read_json(nc.OUT/'train_record.json'), run.read_json(nc.OUT/'prepare_manifest.json')
    if train is None or manifest is None:
        print('等 NN: no train_record.json or prepare_manifest.json', file=sys.stderr)
        return 8
    if run.read_json(NX.SEG_DIR/'convert_record.json') is None:
        print('等推断: no probs36 through the gate yet (step 1 comes first)', file=sys.stderr)
        return 8
    if nx_seg.file_sha256(train['final_checkpoint']) != train['final_checkpoint_sha256']:
        print(f"{train['final_checkpoint']}: sha256 differs from train_record.json", file=sys.stderr)
        return 8
    contact = manifest.get('label_contact_subjects') or []
    allow = load_dev_allowlist(core.ALLOWLIST)
    problems = nx_seg.contact_problems(contact, allowlist=allow, sealed=load_sealed_lists(core.ROOT), training_subjects=load_folds(core.make_inputs().FOLDS).training_subjects)
    if problems:
        print(f'not registered: {problems}', file=sys.stderr)
        return 8
    registry = nx_seg.register(NX.registry_path(), nx_seg.registry_document(segmenter_id=NX.SEGMENTER_ID, model_id=NX.MODEL_ID, fold=NX.FOLD, checkpoint_sha256=train['final_checkpoint_sha256'], contact=contact))
    print(f'{registry.path}: sha256 {registry.sha256}, {len(contact)} subjects of label contact; commit it before the queue of the fields')
    return 0


# ---- the checks of the inputs -------------------------------------------------------------------------------------------------------------------
def stage_problems(out, pairs, *, stage=None, allowlist=None):
    """selection.json, the manifests through the firewall's reader, the subjects on the allowlist, the npy headers of fixed and moving.  run_fl.input_problems' first half, which goes on to unet35_v1's predictions."""
    stage = core.STAGE if stage is None else stage
    allowlist = core.ALLOWLIST if allowlist is None else allowlist
    problems = []
    listed = [row['pair'] for row in json.loads((Path(stage)/'selection.json').read_text())['pairs']]
    same = tuple(listed[:len(ALL_PAIRS)]) == ALL_PAIRS
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
        out(f"{core.short_name(pair)}: subjects on the allowlist; spacing {metadata['spacing_xyz']}; " + ', '.join(f'{m} {h[0]} {h[1]}' for m, h in heads.items()))
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


def rw_input_problems(out=lambda text: None, *, stage=None, allowlist=None, results=None, sources=None):
    """Every source field against its record and its pin (the files are hashed), the 120 s repairs as the plan says they were.  fd2_v1 only read; no label; nothing written."""
    problems = stage_problems(out, ALL_PAIRS, stage=stage, allowlist=allowlist)+frozen_problems(out)
    pins = RW.sources(sources)
    root = RW.fd2_root(results)
    for pair in ALL_PAIRS:
        short = core.short_name(pair)
        d, r = fd2.run_dir(root, RW.fd2_field_cell(pair)), fd2.run_dir(root, RW.fd2_repair_cell(pair))
        record, metrics, old, old_metrics = run.read_json(d/'run.json') or {}, run.read_json(d/f'metrics_{fx.NATIVE:04d}.json') or {}, run.read_json(r/'repair.json') or {}, run.read_json(r/'repair_metrics.json') or {}
        checkpoint = (record.get('checkpoints') or {}).get(str(fx.NATIVE)) or {}
        pin = pins.get(short) or {}
        same = (d/str(checkpoint.get('file'))).is_file() and core.sha_file(d/checkpoint['file']) == checkpoint.get('sha256')
        accepted = ((old_metrics.get('after') or {}).get('certificate') or {}).get('accepted_alpha')
        rounds = len(old.get('rounds') or [])
        out(f"{short}: {d}: status {record.get('status')}, oracle {record.get('oracle')}, D_factor {(record.get('scale') or {}).get('D_factor')}, field equal to its record: {same}, to the pin: {checkpoint.get('sha256') == pin.get('field_sha256')}; the 120 s repair: reason {old.get('reason')}, wall hit {decision.hit_wall(old)}, {rounds} rounds, accepted alpha {accepted}")
        if record.get('status') != 'done' or record.get('oracle') is not False or (record.get('scale') or {}).get('D_factor') != fx.D_FACTOR or not same or metrics.get('tapered_u_sha256') != pin.get('tapered_u_sha256') or checkpoint.get('sha256') != pin.get('field_sha256'):
            problems.append(f'{short}: the source field is not usable or not the pinned one')
        if not (r/'repair.npz').is_file() or old.get('attempt', {}).get('sha256') != pin.get('old_attempt_sha256'):
            problems.append(f'{short}: the 120 s repair is not the pinned one')
        elif pair in RW.WALL_PAIRS and not decision.hit_wall(old):
            problems.append(f"{short}: in the list of the pairs that hit the wall, its repair did not: reason {old.get('reason')}, {rounds} rounds")
        elif pair not in RW.WALL_PAIRS and (decision.hit_wall(old) or rounds != 1 or accepted != 1.):
            problems.append(f'{short}: the plan reuses it as one round certified: reason {old.get("reason")}, {rounds} rounds, accepted alpha {accepted}')
    return problems


def ox_input_problems(out=lambda text: None, *, stage=None, allowlist=None, results=None):
    """The images' headers; the names of the truth members in the zip's directory (not one byte of them is read); the reference of the slope.  No label; nothing written."""
    problems = stage_problems(out, core.PAIRS, stage=stage, allowlist=allowlist)+frozen_problems(out)
    stage = core.STAGE if stage is None else stage
    for pair in core.PAIRS:
        with zipfile.ZipFile(Path(stage)/pair/'pair.npz') as z:
            names = set(z.namelist())
        there = {'fixed_seg35.npy', 'moving_seg35.npy'} <= names
        out(f'{core.short_name(pair)}: the members the oracle will read are in the file: {there} (names only)')
        if not there:
            problems.append(f'{core.short_name(pair)}: no fixed_seg35 or moving_seg35 in pair.npz')
    hx24 = decision.hx24_unrepaired(results)
    out(f"Hx24@600 not repaired, fd2_v1, first 5 pairs: read {hx24['n_read']}/5, seg35 / seg4 {hx24['seg35']} / {hx24['seg4']} (the slope's reference)")
    if not hx24['complete']:
        problems.append('Hx24@600 of fd2_v1 not read 5/5')
    return problems


def nx_waiting(results=None, *, phase='fields', repo=None):
    """(waiting, stop): what NX waits for, which is not a problem; why NN's reading does not let it run."""
    waiting, stop = [], None
    root = nn_root(results)
    if run.read_json(root/'train_record.json') is None:
        waiting.append('等 NN：训练没有结束（没有 train_record.json，没有权重）')
    document = run.read_json(root/'analysis/decision_nn.json')
    if document is None:
        waiting.append('等 NN：没有读数（没有 analysis/decision_nn.json）')
    elif decision.nn_action(document) in (None, 'none'):
        stop = f"NN 的读数 {document.get('reading')}：NX 不跑"
    if phase == 'fields':
        rw_root = Path(fx.RESULTS if results is None else results)/RW.ROOT_NAME
        rw_document = run.read_json(rw_root/'analysis/decision_rw.json')
        if rw_document is None:
            waiting.append('等 RW：Hx24 的对照还没有（运行顺序：RW 在 NX 之前）')
        elif rw_document.get('reading') == decision.UNDECIDABLE:
            waiting.append(f"等 RW：RW 的读数是{decision.UNDECIDABLE}，Hx24 的对照不全，不花 GPU")
        if not (NX.SEG_DIR/'convert_record.json').is_file():
            waiting.append('等推断：seg/nn935_v1 没有过闸门的 probs36')
        if not NX.registry_path(repo).is_file():
            waiting.append(f'等登记：没有 {NX.REGISTRY_REL}')
    return waiting, stop


def nx_input_problems(out=lambda text: None, *, stage=None, allowlist=None, results=None, seg_root=None, repo=None):
    """The stage and the frozen tree always; the registry and the predictions when they are there: the sha256 of the weights in all three places, the label contact, committed and clean, every subject's files.  No label; nothing written."""
    problems = stage_problems(out, ALL_PAIRS, stage=stage, allowlist=allowlist)+frozen_problems(out)
    train = run.read_json(nn_root(results)/'train_record.json')
    registry_file = NX.registry_path(repo)
    if registry_file.is_file():
        from bayesreg.io.pair import load_segmenter_registry
        from bayesreg.seg.structure_layer import registry_git_state
        entry = load_segmenter_registry(registry_file).models.get(NX.MODEL_ID)
        git = registry_git_state(registry_file)
        out(f"registry {registry_file}: {NX.MODEL_ID} there: {entry is not None}, tracked {git.get('tracked')}, clean {git.get('clean')}")
        if entry is None or (train is not None and list(entry['member_model_sha256']) != [train['final_checkpoint_sha256']]):
            problems.append(f"{NX.REGISTRY_REL}: {NX.MODEL_ID} is not the weights of train_record.json")
        if not (git.get('tracked') is True and git.get('clean') is True):
            problems.append(f'{NX.REGISTRY_REL} is not committed and clean')
    seg_root = mc.SEG_ROOT if seg_root is None else seg_root
    record = run.read_json(NX.SEG_DIR/'convert_record.json')
    if record is not None:
        out(f"convert_record.json: gate {record.get('gate')}, {len(record.get('subjects') or [])} subjects")
        subjects = sorted({s for pair in ALL_PAIRS for s in pair.split('__')})
        problems += nx_seg.convert_problems(record, subjects)
        for subject in subjects:
            p = NX.PRED_ROOT/subject
            missing = [name for name in (nx_seg.PROVENANCE_FILE, nx_seg.PROBS_FILE, nx_seg.GROUPS_FILE) if not (p/name).is_file()]
            missing += [] if (Path(seg_root)/subject/'provenance.json').is_file() else [f'Atropos {Path(seg_root)/subject}']
            if missing:
                problems.append(f'{subject}: missing {missing}')
                continue
            prov = json.loads((p/nx_seg.PROVENANCE_FILE).read_text())
            head = run.npz_header(p/nx_seg.PROBS_FILE, 'p')
            out(f"{subject}: {prov.get('segmenter_id')}, policy {prov.get('policy')}, test_labels_seen {prov.get('test_labels_seen')}, probs36 p {head[0]} {head[1]}; Atropos there")
            if prov.get('segmenter_id') != NX.SEGMENTER_ID or prov.get('policy') != 'all_folds_mean' or prov.get('test_labels_seen') is not False or (train is not None and prov.get('model_sha256') != {NX.MODEL_ID: [train['final_checkpoint_sha256']]}):
                problems.append(f'{subject}: the prediction is not nn935_v1 of these weights')
    return problems


def safe(fn, out=lambda text: None, **kwargs):
    """A missing file or a refused input is a problem and not a traceback: the queue says what it was, in queue_stop.json."""
    try:
        return fn(out, **kwargs)
    except (Exception, SystemExit) as error:            # noqa: BLE001 - any of them means the inputs are not there
        return [f'{type(error).__name__}: {error}'[:1000]]


# ---- the queues --------------------------------------------------------------------------------------------------------------------------------------
def reading_code_problem(base, root):
    """Why this process may not write this root's reading, or None: the files that hold the rules must be the batch's commit, byte for byte (FL's rule).  No batch.json: nothing has run."""
    batch = run.read_json(Path(root)/'batch.json')
    if batch is None:
        return None
    for path, rel in READING_FILES:
        theirs = base.committed_bytes(batch['code_sha'], rel)
        if theirs is None or theirs != path.read_bytes():
            return f"{path} is not {rel} of the batch's commit {batch['code_sha']}: the reading's rules would not be the batch's; run the driver from $ROOT/code"
    return None


def start(base, o, system, root, problems_of):
    base.keep_old_stops(root)
    code, sha = system.code_checkout(root, o.commit)
    gate = replace(base.GATE, host_floor_gib=o.min_host_gib)
    problems = problems_of()
    if problems:
        raise run.Stop(f'the inputs, before any run: {problems}')
    return code, sha, gate


def write_batch(base, o, root, sha, gate, plan, name='batch.json', **extra):
    base.write_batch(root, name, dict(plan=plan, gap_seconds=o.gap_seconds, caps=dict(run_seconds=fl.RUN_CAP_SECONDS, run_units_seconds=base.RUN_UNITS_CAP_SECONDS, device_gib=mc.VRAM_LIMIT_BYTES/2**30, memory_max=base.MEMORY_MAX, nice=base.NICE, repair_wall_seconds=fx.FUSE_SECONDS, repair_runtime_seconds=fx.REPAIR_RUNTIME_SECONDS), gate=dict(own_units=list(gate.own_units), load_limit=gate.load_limit, gpu_free_mib=gate.gpu_free_mib, mem_available_gib=gate.mem_available_gib, home_floor_gb=run.HOME_FLOOR_GB, host_floor_gib=gate.host_floor_gib), evaluation=dict(numba_threads=base.EVAL_THREADS, cpu_quota_percent=base.EVAL_QUOTA), **extra), sha)
    problem = reading_code_problem(base, root)
    if problem:
        raise run.Stop(problem)


def finish(root, log, result, rcs, stop):
    """Exit 8 when something failed: the reading is written all the same."""
    log(f"reading: {result['reading']}.  Handed to the lead: no next step starts here")
    failed = {unit: rc for unit, rc in rcs.items() if rc != 0}
    if failed:
        raise run.Stop(f'an evaluation unit failed: {failed}; the reading is written, a pair without metrics counts as not read')
    if stop is not None:
        raise run.Stop(f'a phase stopped: {stop}')
    if result['reading'] == decision.UNDECIDABLE:
        raise run.Stop(f"the reading is {decision.UNDECIDABLE}: {result['reason']}"[:1000])
    return 0


def rw_body(o, system, root, log):
    base = RW_BASE
    code, sha, gate = start(base, o, system, root, lambda: safe(rw_input_problems))
    write_batch(base, o, root, sha, gate, [c.run_id for c in RW.plan_control()+RW.plan_wall()], later='the other 13 pairs only when the control is not bit for bit: analysis/rw_control.json')
    log(f'start: the control, then the 5 pairs that hit the 120 s wall; wall_seconds {fx.FUSE_SECONDS:g}; /mnt/c floor {o.min_host_gib:g} GiB')
    base.run_phase(o, system, root, log, code, sha, RW.plan_control(), gate)
    choice = decision.ensure_rw_control(root)
    if choice is None:
        raise run.Stop(f'{RW.plan_control()[0].run_id}: the control left no record; kept, not rerun')
    base.log_choice(log, root, RW.CONTROL_FILE, f"the control: bit for bit fd2_v1's repair: {choice['same']}")
    cells = RW.plan_wall()+(RW.plan_reused() if choice['same'] is False else [])
    rcs, stop = {}, None
    try:
        base.phase_and_evaluation(o, system, root, log, code, sha, cells, gate, base.EVAL_UNIT, rcs)
    except run.Stop as error:
        stop = str(error)
    state = base.ledger(root)
    result = decision.write_rw(root, stops=state['arms_stopped'])
    run.write_json(root/'queue_end.json', dict(stop=stop, eval_rc=rcs, reading=result['reading'], reason=result['reason'], control=choice, fuse_hit=result['fuse_hit'], ledger=state, utc=run.utc()))
    return finish(root, log, result, rcs, stop)


def ox_body(o, system, root, log):
    base = OX_BASE
    code, sha, gate = start(base, o, system, root, lambda: safe(ox_input_problems))
    write_batch(base, o, root, sha, gate, [c.run_id for c in OX.plan_fields()])
    log(f'start: {len(OX.plan_fields())} runs of the oracle arm; /mnt/c floor {o.min_host_gib:g} GiB')
    rcs, stop = {}, None
    try:
        base.phase_and_evaluation(o, system, root, log, code, sha, OX.plan_fields(), gate, base.EVAL_UNIT, rcs)
    except run.Stop as error:
        stop = str(error)
    state = base.ledger(root)
    result = decision.write_ox(root, stops=state['arms_stopped'])
    run.write_json(root/'queue_end.json', dict(stop=stop, eval_rc=rcs, reading=result['reading'], reason=result['reason'], ledger=state, utc=run.utc()))
    return finish(root, log, result, rcs, stop)


def segmenter_unit(unit, code, inner, *, log, gpu, runtime):
    """The units of step 1: sci310 around the child; the child's environment is nn_env's."""
    return run.unit_argv(unit, code, ['nx', *inner], log=log, gpu=gpu, numba=1, quota=800, runtime=runtime, script=SCRIPT, memory_max=SEGMENTER_MEMORY_MAX, nice=NX_BASE.NICE)


def infer_body(o, system, root, log, code, sha, gate):
    """Step 1.  NN's reading must allow it; one GPU unit, one CPU unit; the gate of the labels stops in the second."""
    waiting, stop_reason = nx_waiting(phase='infer')
    if waiting or stop_reason:
        raise run.Stop('；'.join(waiting+([stop_reason] if stop_reason else [])))
    write_batch(NX_BASE, o, root, sha, gate, ['tag-nx-infer', 'tag-nx-convert'], name='batch_infer.json')
    NX.SEG_DIR.mkdir(parents=True, exist_ok=True)
    if not (NX.SEG_DIR/'raw/infer_record.json').is_file():
        if (NX.SEG_DIR/'dev_input').exists():
            raise run.Stop(f"{NX.SEG_DIR/'dev_input'} exists without an inference record: kept, not rerun; the lead decides")
        run.wait_for_gate(system, root, True, o.wait_minutes, log, gate)
        log('launch tag-nx-infer')
        rc = system.run(segmenter_unit('tag-nx-infer', code, ['infer-one', '--code-sha', sha], log=root/'infer_unit.log', gpu=True, runtime=INFER_RUNTIME), root/'tag-nx-infer.systemd_run.log')
        if rc != 0:
            raise run.Stop(f'tag-nx-infer exit {rc}; kept, not rerun')
    run.wait_for_gate(system, root, False, o.wait_minutes, log, gate)
    log('launch tag-nx-convert')
    rc = system.run(segmenter_unit('tag-nx-convert', code, ['convert', '--code-sha', sha], log=root/'convert_unit.log', gpu=False, runtime=CONVERT_RUNTIME), root/'tag-nx-convert.systemd_run.log')
    if rc != 0:
        raise run.Stop(f"tag-nx-convert exit {rc}: {(run.read_json(NX.SEG_DIR/'convert_stop.json') or {}).get('reason')}")
    record = run.read_json(NX.SEG_DIR/'convert_record.json') or {}
    log(f"the gate passed for the 20 subjects: {record.get('n_differing_total')} of {record.get('n_voxels_total')} voxels differ from pred_dev in all.  Next by hand: run_fx.py nx register, commit it")
    return 0


def repairs_of_done_fields(root, repairs, log):
    """The repairs whose field is done (FX review, minor 2): a nonfinite or failed field has nothing to repair; it would only crash in source_checked and count among the failures in a row.  The reading already counts such a pair as dropped or failed."""
    out = []
    for cell in repairs:
        field = NX.Cell(fx.NX_ARM, NX.FIRST if cell.tier.startswith('c') else cell.tier, cell.pair)
        status = (run.read_json(NX.run_dir(root, field)/'run.json') or {}).get('status')
        if status == 'done':
            out.append(cell)
        else:
            log(f'skip {cell.run_id}: its field is {status}, nothing to repair')
    return out


def nx_commit(commit):
    """The sha a checkout of `commit` would take, without making one: System.code_checkout's own rev-parse, in the repository."""
    done = subprocess.run(['git', 'rev-parse', '--verify', f"{commit or 'HEAD'}^{{commit}}"], cwd=core.ROOT, capture_output=True, text=True)
    if done.returncode:
        raise run.Stop(f"git rev-parse {commit or 'HEAD'}: {done.stderr.strip()}")
    return done.stdout.strip()


def registry_problem(base, sha):
    """Why step 3 may not check out this commit, or None: the commit must hold the registration, byte for byte the file the driver checks tracked and clean."""
    theirs, mine = base.committed_bytes(sha, NX.REGISTRY_REL), NX.registry_path()
    if theirs is None:
        return f'{sha} has no {NX.REGISTRY_REL}: commit the registration (run_fx.py nx register) before step 3; nothing checked out'
    if not mine.is_file() or theirs != mine.read_bytes():
        return f'{NX.REGISTRY_REL} at {sha} is not the file {mine}; nothing checked out'
    return None


def nx_body(o, system, root, log):
    """Step 1 and step 3 have their own checkouts (FX review, blocking 1): step 3 runs at the commit that holds the registration, which is made after step 1; one $ROOT/code at two commits would stop.
    Nothing is checked out before every check that needs no code (follow-up, important 1): a step started too early stops and leaves no $ROOT/code at a commit without the registration."""
    base = NX_BASE
    base.keep_old_stops(root)
    gate = replace(base.GATE, host_floor_gib=o.min_host_gib)
    sha = nx_commit(o.commit)
    waiting, stop_reason = nx_waiting(phase=o.phase)
    if waiting or stop_reason:
        raise run.Stop('；'.join(waiting+([stop_reason] if stop_reason else []))+'; nothing checked out')
    if o.phase == 'infer':
        (root/INFER_DIR).mkdir(parents=True, exist_ok=True)
        code, sha = system.code_checkout(root/INFER_DIR, sha)
        return infer_body(o, system, root, log, code, sha, gate)
    problem = registry_problem(base, sha)
    if problem:
        raise run.Stop(problem)
    problems = safe(nx_input_problems)
    if problems:
        raise run.Stop(f'the inputs, before any run: {problems}; nothing checked out')
    code, sha = system.code_checkout(root, sha)
    write_batch(base, o, root, sha, gate, [c.run_id for c in NX.plan_fields()+NX.plan_repairs()], later='the other 14 pairs by analysis/rest14_selection.json')
    document = nn_reading()
    reference = decision.nx_reference()
    log(f"start: NHx24 on the first 5 pairs; NN's reading {document.get('reading')}, Delta {document.get('delta_seg35')}; /mnt/c floor {o.min_host_gib:g} GiB")
    rcs, stop = {}, None
    try:
        base.phase_and_evaluation(o, system, root, log, code, sha, NX.plan_fields(), gate, base.EVAL_UNIT, rcs)
        base.evaluated(rcs, base.EVAL_UNIT, 'the repairs are')
        base.phase_and_evaluation(o, system, root, log, code, sha, repairs_of_done_fields(root, NX.plan_repairs(), log), gate, base.EVAL_R1_UNIT, rcs)
        base.evaluated(rcs, base.EVAL_R1_UNIT, 'the other 14 pairs are')
        selection = decision.ensure_rest_selection(root, document, reference)
        base.log_choice(log, root, NX.REST_FILE, f"the other 14 pairs: {selection['go']}: {selection['why']}")
        if selection['go'] is True:
            base.phase_and_evaluation(o, system, root, log, code, sha, NX.plan_rest_fields(), gate, base.EVAL_REST_UNIT, rcs)
            base.evaluated(rcs, base.EVAL_REST_UNIT, 'the repairs of the other 14 are')
            base.phase_and_evaluation(o, system, root, log, code, sha, repairs_of_done_fields(root, NX.plan_rest_repairs(), log), gate, base.EVAL_REST_R_UNIT, rcs)
    except run.Stop as error:
        stop = str(error)
    state = base.ledger(root)
    result = decision.write_nx(root, document, stops=state['arms_stopped'])
    run.write_json(root/'queue_end.json', dict(stop=stop, eval_rc=rcs, reading=result['reading'], reason=result['reason'], not_read=result['not_read'], rest14=result['rest14'], ledger=state, utc=run.utc()))
    if result['not_read'] and stop is None:
        stop = f"these cells failed: {result['not_read']}"[:1000]
    return finish(root, log, result, rcs, stop)


BODIES = dict(rw=rw_body, ox=ox_body, nx=nx_body)


def queue(name, o, system=None):
    batch = fx.BATCHES[name]
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
    batch, base = fx.BATCHES[name], BASES[name]
    return run.check_launch(['systemd-run', '--user', f'--unit={base.QUEUE_UNIT}', '--collect', f'--working-directory={core.MAIN_REPO}', f'--property=MemoryMax={base.MEMORY_MAX}', f'--nice={base.NICE}', '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', run.PY, '-B', SCRIPT, name, 'queue', '--root', str(batch.DEFAULT_ROOT if root is None else root), '--gap-seconds', str(base.GAP_SECONDS), '--min-host-gib', f'{base.DEFAULT_MIN_HOST_GIB:g}', *extra])


# ---- estimates --------------------------------------------------------------------------------------------------------------------------------------
def estimates(results=None):
    """Seconds and bytes from fd2_v1's records of Hx24 and fl_v1's oracle arm, 〔估算〕; None where there is no record.  Nobody has run these arms; a repair with the fuse has no upper bound but the fuse."""
    root = RW.fd2_root(results)
    tasks, evals, one_round, wall_first, sizes = [], [], [], [], []
    for pair in ALL_PAIRS:
        d, r = fd2.run_dir(root, RW.fd2_field_cell(pair)), fd2.run_dir(root, RW.fd2_repair_cell(pair))
        record, metrics, old = run.read_json(d/'run.json') or {}, run.read_json(d/f'metrics_{fx.NATIVE:04d}.json') or {}, run.read_json(r/'repair.json') or {}
        checkpoint = (record.get('checkpoints') or {}).get(str(fx.NATIVE)) or {}
        if checkpoint.get('seconds_task') is not None:
            tasks.append(checkpoint['seconds_task'])
        if metrics.get('seconds_certificate') is not None and metrics.get('seconds_metrics') is not None:
            evals.append(metrics['seconds_certificate']+metrics['seconds_metrics'])
        if old.get('seconds_total') is not None:
            (wall_first if pair in RW.WALL_PAIRS else one_round).append(old['seconds_total'])
        if (d/f'field_{fx.NATIVE:04d}.npz').is_file():
            sizes.append((d/f'field_{fx.NATIVE:04d}.npz').stat().st_size)
    average = (lambda v: sum(v)/len(v) if v else None)
    task, evaluation, once, wall, size = average(tasks), average(evals), average(one_round), average(wall_first), average(sizes)
    gap = RW_BASE.GAP_SECONDS
    out = dict(task_seconds=task, evaluation_seconds=evaluation, one_round_repair_seconds=once, repair_that_hit_the_wall_seconds=wall, field_gib=None if size is None else size/2**30, gate_wait_seconds=GATE_WAIT_SECONDS)
    if None not in (task, evaluation, once, wall, size):
        three_rounds = wall+90.                 # 〔估算〕a third round of about 90 s (the review of FD2: 690-780 thousand cells, 300 steps), more if it needs more
        out.update(rw_same_seconds=once+gap+5*(three_rounds+gap)+6*evaluation+GATE_WAIT_SECONDS, rw_other_seconds=once+gap+5*(three_rounds+gap)+13*(once+gap)+19*evaluation+GATE_WAIT_SECONDS, rw_fuse_cell_seconds=fx.FUSE_SECONDS+15.,
                   ox_seconds=5*(task*1.3+gap+evaluation)+GATE_WAIT_SECONDS, nx_first5_seconds=5*(task+gap+evaluation+once+gap+evaluation)+2*GATE_WAIT_SECONDS, nx_rest_seconds=14*(task+gap+evaluation+once+gap+evaluation)+2*GATE_WAIT_SECONDS, nx_field_gib=19*size/2**30, rw_disk='repair.npz and records only: MiB')
    return out


def _minutes(seconds):
    return '—' if seconds is None else f'{seconds/60:.0f} min'


# ---- dry run --------------------------------------------------------------------------------------------------------------------------------------
def print_gate(out, base, root):
    gate = base.GATE
    out(f'root {root}; root exists: {Path(root).exists()}; cap per run {fl.RUN_CAP_SECONDS} s; repair wall_seconds {fx.FUSE_SECONDS:g} s, RuntimeMaxSec {fx.REPAIR_RUNTIME_SECONDS}; device {mc.VRAM_LIMIT_BYTES/2**30} GiB; MemoryMax {base.MEMORY_MAX}, nice {base.NICE}; {base.GAP_SECONDS} s between runs')
    out(f'gate: own units {gate.own_units}, load < {gate.load_limit}, GPU free >= {gate.gpu_free_mib} MiB, MemAvailable >= {gate.mem_available_gib} GiB, /home >= {run.HOME_FLOOR_GB} GB, /mnt/c >= {gate.host_floor_gib} GiB; closed while any other tag-* or seq-vp-* unit is active, so while tag-nn-train runs')


def print_cells(out, cells, phase):
    for c in cells:
        out(f'{phase} | {c.arm} | {c.short} | {c.unit} | {c.run_id}')


def dry_run(name, *, out=print, root=None, stage=None, allowlist=None, results=None, seg_root=None, sources=None, repo=None):
    """The plan, the estimates, the checks of the queue, the start strings.  No truth label, no optimization, nothing written."""
    batch, base = fx.BATCHES[name], BASES[name]
    root = batch.DEFAULT_ROOT if root is None else root
    code = Path(root)/'code'
    out(f'{name.upper()} dry run ({run.utc()}): no optimization, no truth label, nothing written; real files read up to the npy headers of fixed and moving, JSON records' + (', and the source fields of fd2_v1 for their sha256' if name == 'rw' else ''))
    print_gate(out, base, root)
    e = estimates(results)
    if name == 'rw':
        print_cells(out, RW.plan_control(), 'control')
        print_cells(out, RW.plan_wall(), 'the 5 pairs that hit the 120 s wall')
        print_cells(out, RW.plan_reused(), 'only when the control is not bit for bit')
        out(f"estimates〔估算, from fd2_v1's records; nobody has run this configuration〕: control bit for bit {_minutes(e.get('rw_same_seconds'))}, not bit for bit {_minutes(e.get('rw_other_seconds'))}, with a third round of about 90 s on the 5 pairs; one repair at the fuse {_minutes(e.get('rw_fuse_cell_seconds'))}; disk {e.get('rw_disk')}")
        problems = safe(rw_input_problems, out, stage=stage, allowlist=allowlist, results=results, sources=sources)
        cells = RW.plan_control()+RW.plan_wall()
    elif name == 'ox':
        print_cells(out, OX.plan_fields(), 'oracle')
        out(f"estimates〔估算, Hx24's task clock x 1.3 for the slower pyramid of the oracle path; nobody has run this arm〕: {_minutes(e.get('ox_seconds'))}; 5 field files {None if e.get('field_gib') is None else round(5*e['field_gib'], 1)} GiB")
        problems = safe(ox_input_problems, out, stage=stage, allowlist=allowlist, results=results)
        cells = OX.plan_fields()
    else:
        print_cells(out, NX.plan_fields(), 'first 5 fields')
        print_cells(out, NX.plan_repairs(), 'first 5 repairs')
        print_cells(out, NX.plan_rest_fields()+NX.plan_rest_repairs(), 'the other 14, by the code')
        infer = run.read_json(nn_root(results)/'pred_dev/infer_record.json') or {}
        seconds_infer = '等 NN，没有记录' if not infer else f"pred_dev took {infer.get('seconds_median')} s a subject without probabilities"
        out(f"estimates〔估算, from fd2_v1's records and pred_dev's infer_record; nobody has run this arm〕: first 5 pairs {_minutes(e.get('nx_first5_seconds'))}, the other 14 {_minutes(e.get('nx_rest_seconds'))}; inference of 20 subjects: {seconds_infer}; disk: probs36 and groups about 6.7 GiB (unet35_v1 has 336 MiB a subject), nnU-Net raw npz some more, the lead cleans; fields {None if e.get('nx_field_gib') is None else round(e['nx_field_gib'], 1)} GiB")
        problems = safe(nx_input_problems, out, stage=stage, allowlist=allowlist, results=results, seg_root=seg_root, repo=repo)
        waiting, stop_reason = nx_waiting(results, repo=repo)
        for line in waiting:
            out(f'{line}（不算问题）')
        if stop_reason:
            out(f'{stop_reason}（不算问题）')
        cells = NX.plan_fields()+NX.plan_repairs()
    out(f'/home free {shutil.disk_usage("/home").free/2**30:.0f} GB; /mnt/c free {shutil.disk_usage("/mnt/c").free/2**30:.0f} GiB')
    out('--- start strings (the queue: the lead, after the training and the inference of nnU-Net; the others: the queue, one per run; the code at $ROOT/code)')
    if name == 'nx':
        out(shlex.join(queue_launch(name, root, extra=('--phase', 'infer'))))
        infer_code = Path(root)/INFER_DIR/'code'
        out(shlex.join(segmenter_unit('tag-nx-infer', infer_code, ['infer-one', '--code-sha', '<sha>'], log=Path(root)/'infer_unit.log', gpu=True, runtime=INFER_RUNTIME)))
        out(shlex.join(segmenter_unit('tag-nx-convert', infer_code, ['convert', '--code-sha', '<sha>'], log=Path(root)/'convert_unit.log', gpu=False, runtime=CONVERT_RUNTIME)))
        out(f'{run.PY} -B {SCRIPT} nx register        # by hand, after the gate; commit {NX.REGISTRY_REL}')
        out(shlex.join(queue_launch(name, root, extra=('--phase', 'fields'))))
    else:
        out(shlex.join(queue_launch(name, root)))
    for c in cells:
        out(shlex.join(base.run_argv(c, code, root, '<sha>')))
    out(shlex.join(base.eval_argv(code, root, base.EVAL_UNIT)))
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


# ---- the entries -----------------------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) < 2 or argv[0] not in fx.BATCHES:
        print('usage: run_fx.py rw|ox|nx --dry-run | queue | run-one | repair-one | evaluate | decide; nx also infer-one, convert, register; rw also write-pins', file=sys.stderr)
        return 2
    name, rest = argv[0], argv[1:]
    batch, base = fx.BATCHES[name], BASES[name]
    if rest == ['--dry-run']:
        return dry_run(name)
    p = argparse.ArgumentParser(description='FX: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(batch.DEFAULT_ROOT))
    q.add_argument('--commit', default=None)
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--gap-seconds', type=int, default=base.GAP_SECONDS)
    q.add_argument('--min-host-gib', type=float, default=base.DEFAULT_MIN_HOST_GIB)
    if name == 'nx':
        q.add_argument('--phase', choices=('infer', 'fields'), required=True)
    for command in ('run-one', 'repair-one'):
        r = sub.add_parser(command)
        r.add_argument('--arm', required=True)
        r.add_argument('--tier', required=True)
        r.add_argument('--pair', required=True)
        r.add_argument('--out', required=True)
        r.add_argument('--device', default='cuda')
        r.add_argument('--code-sha', default='n/a')
    for command in ('evaluate', 'decide'):
        sub.add_parser(command).add_argument('--root', default=str(batch.DEFAULT_ROOT))
    if name == 'rw':
        sub.add_parser('write-pins')
    if name == 'nx':
        for command in ('infer-one', 'convert'):
            sub.add_parser(command).add_argument('--code-sha', default='n/a')
        sub.add_parser('register')
    o = p.parse_args(rest)
    if o.command == 'queue':
        return queue(name, o)
    if o.command == 'run-one':
        if name == 'rw':
            raise SystemExit('RW has no field run: its fields are fd2_v1\'s')
        return ox_run_one(o) if name == 'ox' else nx_run_one(o)
    if o.command == 'repair-one':
        if name == 'ox':
            raise SystemExit('the oracle arm is never repaired')
        return repair_one(base, o)
    if o.command == 'write-pins':
        if RW.SOURCES_FILE.exists():
            raise SystemExit(f'{RW.SOURCES_FILE} is there: a pin is written once')
        RW.SOURCES_FILE.write_text(json.dumps(rw_pins_from_records(), indent=1)+'\n')
        return 0
    if o.command == 'infer-one':
        return nx_infer_one(o)
    if o.command == 'convert':
        return nx_convert(o)
    if o.command == 'register':
        return nx_register(o)
    problem = batch.root_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if o.command == 'evaluate':
        return base.evaluate_all(o)
    problem = reading_code_problem(base, o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 8
    stops = base.ledger(o.root)['arms_stopped']
    if name == 'rw':
        decision.write_rw(o.root, stops=stops)
    elif name == 'ox':
        decision.write_ox(o.root, stops=stops)
    else:
        decision.write_nx(o.root, nn_reading(), stops=stops)
    return 0


if __name__ == '__main__':
    sys.exit(main())
