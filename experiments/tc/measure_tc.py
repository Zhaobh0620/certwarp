"""TC's small measurement (runs/20261104-tc/plan.md; the brief §5, the time ledger): one GPU unit, tag-tc-measure, last in the queue.  Two parts, each the real code path with timers around its module attributes; no solver.

  segmenter   bayesreg.seg.unet.infer.segment_subject on OASIS_OAS1_0438_MR1 (a development subject), the 4 members of unet35_v1 in the repository's registry: reading the image, the checks, each member's file, its forward pass, the calibration (softmax(z/T) and the sum), the groups, the two compressed files.
              It writes to <out>/pred_tmp, never to seg/unet35_v1/pred; the two files are hashed against the recorded provenance and deleted.
  entry       experiments/mc0/run_mc0p.py run_one, Q2 long on 0438-0439 (K = 35, 600 native steps, predicted labels): each step of the channels, then the optimizer and its levels.  It writes field.npz and run.json to <out>/q2_tmp; u is hashed against the recorded run of mc0p_v1 and both go; nothing is written to mc0p_v1.

    PYTHONPATH=<frozen>:<code> python -B experiments/tc/measure_tc.py --out /home/rmcfz/seq_vp_tag_results/tc_v1/measure [--device cuda]

The timers replace module attributes for the duration of the call and put them back: they change no value and hold no state of the code they wrap.  They are nested: never added up.  One run on one subject: an observation, not a stable number.  Page cache: whatever the queue's last run left.
Writes only <out>/measure_0438.json (after each part) and the scratch directories it removes.  Refuses any --out but /home/rmcfz/seq_vp_tag_results/tc_v1/measure, and a second run (measure_0438.json exists).  Exit codes: 0 done, 2 usage or refused, 8 a part failed.
"""
import argparse
from contextlib import contextmanager
import gc
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))                          # jd_curl and bayesreg come from PYTHONPATH=<frozen>:<code>, frozen first (mc0_core.frozen_api refuses anything else)

import tc_curve as curve  # noqa: E402

SCHEMA = 'bayesreg.tc.measure.v1'
SUBJECT = 'OASIS_OAS1_0438_MR1'
PAIR = 'OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1'
RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
UNET_ROOT = RESULTS/'seg/unet35_v1'
RECORDED_PREDICTION = UNET_ROOT/'pred'/SUBJECT/'provenance.json'
RECORDED_ENTRY = RESULTS/'mc0p_v1/runs/Q2_long/0438-0439/run.json'
IMAGE_ROOT = Path('/mnt/d/data')                       # make_inputs.UNetSource.image_root; read only
REGISTRY = ROOT/'configs/bayesreg/segmenters.json'
ALLOWLIST = ROOT/'configs/bayesreg/dev_subjects.json'
GROUPS = ROOT/'configs/bayesreg/structure_groups_v1.json'
OUT_NAME, ROOT_NAME = 'measure', 'tc_v1'
RESULT_FILE = 'measure_0438.json'
# The module attributes each part calls: a test checks they exist on the real modules (a name that no longer exists would silently time nothing).
SEGMENTER_STEPS = ('check_subject_admitted', 'require_inference_grouping', 'select_fold_models', 'find_member_files', 'read_image', 'load_member', 'normalize_image', 'predict_logits', 'ensemble_probabilities', 'group_probabilities', '_save_npz', 'file_sha256', 'write_json')
MAKE_INPUTS_STEPS = ('a4_admission', 'pair_structure_layer', 'unet_posterior', 'support_dice_record', 'atropos_wm', 'alignment_qc')
MC_STEPS = ('read_images', 'read_predicted', 'predicted_labels', 'q0_channels', 'prepare_stacks')
CORE_STEPS = ('load_table', 'channels', 'equals_a5_input', 'data_grad_norm', 'p0_level_norms', 'coarse_pyramid', 'optimize_k')


def out_problem(out, results=None):
    """Why the measure may not write here, or None.  The resolved path, as run_tc.root_problem: not only the names."""
    out = Path(out)
    want = Path(RESULTS if results is None else results)/ROOT_NAME/OUT_NAME
    if out.resolve() != want.resolve():
        return f'{out}: the measure writes only to {want}; nothing written'
    if (out/RESULT_FILE).exists():
        return f'{out/RESULT_FILE} exists: a measurement is not run twice'
    return None


@contextmanager
def timed(module, names, sink, *, clock=time.perf_counter, after=None):
    """While inside, every call of module.<name> also appends its seconds to sink[name].  `after` (GPU: synchronize) is called before the clock stops, so that a queued computation is charged to the step that asked for it.  Exceptions pass through, the attributes always go back."""
    originals = {name: getattr(module, name) for name in names}

    def wrap(name, function):
        def wrapper(*args, **kwargs):
            tick = clock()
            try:
                return function(*args, **kwargs)
            finally:
                if after is not None:
                    after()
                sink.setdefault(name, []).append(clock()-tick)
        return wrapper
    try:
        for name, function in originals.items():
            setattr(module, name, wrap(name, function))
        yield sink
    finally:
        for name, function in originals.items():
            setattr(module, name, function)


def total(sink, name):
    return float(sum(sink.get(name, ())))


def segmenter_summary(sink, wall):
    """The steps of one segment_subject call in the words of the brief.  calibration: what ensemble_probabilities spends outside its generator, that is softmax(z/T), the sum and the division: its inclusive time minus the files, the normalization and the forward passes inside it."""
    inside = total(sink, 'load_member')+total(sink, 'normalize_image')+total(sink, 'predict_logits')
    calibration = total(sink, 'ensemble_probabilities')-inside
    write = total(sink, '_save_npz')
    top = (total(sink, 'check_subject_admitted')+total(sink, 'require_inference_grouping')+total(sink, 'select_fold_models')+total(sink, 'find_member_files')+total(sink, 'read_image')+total(sink, 'ensemble_probabilities')
           +total(sink, 'group_probabilities')+write+total(sink, 'file_sha256')+total(sink, 'write_json'))
    return dict(wall_seconds=wall, checks_seconds=total(sink, 'check_subject_admitted')+total(sink, 'require_inference_grouping')+total(sink, 'select_fold_models'), hash_all_member_files_seconds=total(sink, 'find_member_files'), read_image_seconds=total(sink, 'read_image'),
                member_load_seconds=list(sink.get('load_member', ())), normalize_seconds=total(sink, 'normalize_image'), forward_seconds=list(sink.get('predict_logits', ())), forward_seconds_total=total(sink, 'predict_logits'), calibration_seconds=calibration,
                groups_seconds=total(sink, 'group_probabilities'), write_seconds=list(sink.get('_save_npz', ())), write_seconds_total=write, hash_outputs_seconds=total(sink, 'file_sha256'), other_seconds=wall-top,
                other_means='what no timer covers: the float32 to float16 copies, the tensor to the host, provenance arguments')


def segmenter_part(work, device, *, subject, files, registry, allowlist, models_root, grouping, recorded=None, sync=None):
    """One segment_subject under the timers, into work/pred_tmp (which must not exist); the output files are hashed against `recorded` and removed."""
    from bayesreg.seg.unet import infer
    work = Path(work)
    pred_root = work/'pred_tmp'
    if pred_root.exists():
        raise SystemExit(f'{pred_root} exists')
    sink = {}
    with timed(infer, SEGMENTER_STEPS, sink, after=sync):
        tick = time.perf_counter()
        provenance = infer.segment_subject(subject, files=files, registry=registry, allowlist=allowlist, models_root=models_root, grouping=grouping, pred_root=pred_root, device=device, command=['experiments/tc/measure_tc.py'], log=print)
        wall = time.perf_counter()-tick
    out = pred_root/subject
    sizes = {p.name: p.stat().st_size for p in out.iterdir()}
    summary = segmenter_summary(sink, wall)
    want = None if recorded is None or not Path(recorded).is_file() else json.loads(Path(recorded).read_text())
    summary.update(subject=subject, n_members=len(provenance['members']), members=[dict(model_id=m['model_id'], seed=m['seed'], temperature=m['temperature']) for m in provenance['members']], policy=provenance['policy'], runtime_seconds_of_segment_subject=provenance['runtime_seconds'],
                   file_bytes=sizes, output_sha256=provenance['output_sha256'], recorded_runtime_seconds=None if want is None else want.get('runtime_seconds'), outputs_equal_recorded=None if want is None else provenance['output_sha256'] == want.get('output_sha256'),
                   recorded_file=None if recorded is None else str(recorded), device=str(device))
    curve.remove_tree(pred_root, work)
    return summary


def segmenter_defaults(out, device):
    """The real files of the part: the development subject only, through its admission check."""
    from bayesreg.io.guard import load_dev_allowlist
    from bayesreg.io.pair import load_segmenter_registry
    from bayesreg.seg.unet.data import OasisFiles
    from bayesreg.seg.unet.groups import load_grouping
    return segmenter_part(out, device, subject=SUBJECT, files=OasisFiles(IMAGE_ROOT), registry=load_segmenter_registry(REGISTRY), allowlist=load_dev_allowlist(ALLOWLIST), models_root=UNET_ROOT, grouping=load_grouping(GROUPS), recorded=RECORDED_PREDICTION, sync=gpu_sync(device))


def gpu_sync(device):
    import torch
    return torch.cuda.synchronize if torch.device(device).type == 'cuda' else None


def load_mc0p():
    """experiments/mc0/run_mc0p.py, read only, and its modules."""
    sys.path.insert(0, str(ROOT/'experiments/mc0'))
    name = '_tc_run_mc0p'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT/'experiments/mc0/run_mc0p.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def entry_part(work, device, *, pair=PAIR, recorded=RECORDED_ENTRY, code_sha='n/a'):
    """run_mc0p.run_one, Q2 long, under the timers, into work/q2_tmp (which must not exist); u is compared with the recorded run; the directory goes.  Q2 long is what S_P's entry is: the same function, the same cell, the same arguments as its unit."""
    run = load_mc0p()
    core, mc = run.core, run.mc
    work = Path(work)
    tmp = work/'q2_tmp'
    if tmp.exists():
        raise SystemExit(f'{tmp} exists')
    mi = core.make_inputs()
    sink = {}
    o = SimpleNamespace(arm='Q2', tier='long', pair=pair, out=str(tmp), device=device, code_sha=code_sha)
    with timed(mi, MAKE_INPUTS_STEPS, sink), timed(mc, MC_STEPS, sink), timed(core, CORE_STEPS, sink):
        run.run_one(o)
    record = json.loads((tmp/'run.json').read_text())
    optimizer = record.get('optimizer') or {}
    want = None if recorded is None or not Path(recorded).is_file() else json.loads(Path(recorded).read_text())
    summary = dict(pair=pair, arm='Q2', tier='long', status=record.get('status'), reason=record.get('reason'), k=record.get('k'), seconds_channels=record.get('seconds_channels'), seconds_total=record.get('seconds_total'), steps_seconds=sink, optimizer_seconds_from_its_start_clock=optimizer.get('seconds'),
                   optimizer_seconds_call=total(sink, 'optimize_k'), levels=[dict(level=row['level'], iterations=row['iterations'], seconds=row['seconds']) for row in optimizer.get('levels') or []], device_peak_allocated_gib=record.get('device_peak_allocated_gib'), device_peak_reserved_gib=record.get('device_peak_reserved_gib'),
                   host_peak_rss_gib=record.get('host_peak_rss_gib'), u_sha256=(record.get('field') or {}).get('u_sha256'), recorded_seconds_total=None if want is None else want.get('seconds_total'), recorded_seconds_channels=None if want is None else want.get('seconds_channels'),
                   u_equal_recorded=None if want is None or not record.get('field') else record['field']['u_sha256'] == (want.get('field') or {}).get('u_sha256'), recorded_file=None if recorded is None else str(recorded), note='seconds_channels is the clock at the start of the optimizer: images, predictions, QC, channels, scale; the steps are nested and are not added up')
    curve.remove_tree(tmp, work)
    return summary


def write_result(out, record):
    curve.write_json(Path(out)/RESULT_FILE, record)


class DeviceMemory:
    """What the parts do with the device's memory; None on the CPU."""

    def __init__(self, torch):
        self.torch = torch

    def reset(self):
        self.torch.cuda.reset_peak_memory_stats()

    def peak_gib(self):
        return self.torch.cuda.max_memory_allocated()/2**30

    def release(self):
        self.torch.cuda.empty_cache()


def run_parts(record, out, parts, *, memory, started):
    """Each part in turn, the record written after each.  A failure is recorded and the next part still runs.  The device's cache goes also after a failure (review m9): vram_guard of the entry part looks at the reserved peak, with its limit of 9 GiB."""
    for name, part in parts:
        try:
            if memory is not None:
                memory.reset()
            record[name] = part()
            if memory is not None:
                record[name]['device_peak_allocated_gib_of_part'] = memory.peak_gib()
        except (Exception, SystemExit):                  # the other part still runs; what happened stays in the record
            record['failures'].append(dict(part=name, traceback=traceback.format_exc()[-4000:]))
        finally:
            gc.collect()                                 # a caught exception's frames may still hold the tensors
            if memory is not None:
                try:
                    memory.release()
                except Exception:                        # the device itself: recorded like a part's failure, the record and the next part are not lost
                    record['failures'].append(dict(part=name, step='release', traceback=traceback.format_exc()[-4000:]))
        record['host_peak_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
        record['wall_seconds'] = time.perf_counter()-started
        write_result(out, record)
    return record


def main(argv=None):
    started = time.perf_counter()
    p = argparse.ArgumentParser(description='TC: see the module docstring')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--code-sha', default='n/a')
    o = p.parse_args(argv)
    problem = out_problem(o.out)
    if problem:                                          # before torch, before any file
        print(problem, file=sys.stderr)
        return 2
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    imported = time.perf_counter()-started
    tick = time.perf_counter()
    if torch.device(o.device).type == 'cuda':
        torch.zeros(1, device=o.device).sum().item()
    cuda_init = time.perf_counter()-tick
    record = dict(schema=SCHEMA, role='a measurement of time, one run on one subject and one pair; no Dice, no result table', subject=SUBJECT, pair=PAIR, code_sha=o.code_sha, utc_start=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(0) if torch.device(o.device).type == 'cuda' else 'cpu', image_root=str(IMAGE_ROOT), import_seconds=imported, device_init_seconds=cuda_init, segmenter=None, entry=None, failures=[])
    write_result(out, record)
    parts = (('segmenter', lambda: segmenter_defaults(out, o.device)), ('entry', lambda: entry_part(out, o.device, code_sha=o.code_sha)))
    run_parts(record, out, parts, memory=DeviceMemory(torch) if torch.device(o.device).type == 'cuda' else None, started=started)
    print(json.dumps({k: record[k] for k in ('wall_seconds', 'failures')}, ensure_ascii=False))
    return 8 if record['failures'] or (record['entry'] or {}).get('status') != 'done' else 0


if __name__ == '__main__':
    sys.exit(main())
