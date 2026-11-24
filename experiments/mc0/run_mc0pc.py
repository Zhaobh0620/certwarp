"""SEED19 (runs/20261104-seed19/任务书.md, plan.md; docs/bayesreg/05_D0协议.md §17): the K = 35 entries of S_P on the other 14 development pairs, in the new root mc0p_v1c.  Q2 long only, with MC0P's optimizer, scale rule, tier,
G_.75, 900 s and 9 GiB (run_mc0p.run_one, evaluate_all: the same code).  `oracle: false`; uncertified fields, not a method, in no result table.  mc0p_v1 and mc0p_v1b are read only.

    python -B experiments/mc0/run_mc0pc.py --dry-run
        prints the 15 runs and checks the inputs: selection.json, the npy headers of `fixed`, `moving` and of A5P's pair.npz (no label member), the manifest through the firewall's reader, each prediction, the `p` header of its probs36.npz.  No optimization, nothing written.
    python -B experiments/mc0/run_mc0pc.py queue [--commit SHA] [--wait-minutes 30] [--min-host-gib 15]
        the lead's start, in the unit printed by the dry run.  Code: a detached worktree $ROOT/code of --commit (default HEAD; tracked files clean).  First the bitwise K = 1 check on the GPU (unit tag-mc0pc-k1check; a mismatch stops the queue).
        Then the control Q2 long 0438-0439, then the 14 pairs, one unit tag-mc0pc-q2-long-<ffff>-<mmmm> at a time (MemoryMax 8G, nice 5), each behind the strict gate of mc0p_v1b (no $ROOT/hold, no other tag-* or seq-vp-* unit, 1-minute load < 1.5, GPU free >= 15 GiB,
        MemAvailable >= 7 GiB, /home >= 100 GB) plus /mnt/c free >= --min-host-gib; closed for --wait-minutes: queue_stop.json, what has finished is evaluated and summed up, exit 8.  A run directory that exists is never rerun.  The arm stops on the device limit of 9 GiB, a device
        out-of-memory error or a run over 15 minutes.  The control's failure stops nothing: it is only recorded (control.json: its u against mc0p_v1b's).  3 failures in a row, images that are not A5P's bit for bit, a record that is not oracle false: STOP, no evaluation.
        Finally one CPU unit tag-mc0pc-eval, analysis/{long.csv, summary_mc0pc.md}, queue_end.json.  No reading: S_P's reading is experiments/seed/decide_seed19.py.
    run_mc0pc.py run-one --arm Q2 --tier long --pair PAIR --out DIR [--device cuda] [--code-sha SHA]       one run: field.npz, run.json
    run_mc0pc.py evaluate             every field of this batch without metrics.json: as run_mc0p.py evaluate; the only place the truth labels are read
    run_mc0pc.py summarize            analysis/long.csv, summary_mc0pc.md

The only root is mc0p_v1c: every entry that writes refuses any other, mc0p_v1 and mc0p_v1b among them.  run-one and evaluate need PYTHONPATH=<frozen>:<code> (the unit sets it).  Exit codes: 0 done, 2 usage, another driver on the root or another root, 8 STOP or a failed check.

The scale rule asks nothing of Q0's optimized field: mc0p_core.prepare_stacks takes ||g_D|| of Q0's two images at u = 0 (mc0_core.data_grad_norm), and the record r_l of the levels likewise (mc0_core.p0_level_norms, level_grad_norms on a zero field).  So no Q0 cell runs here.
"""
import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
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
import run_mc0p as mc0p  # noqa: E402

SCRIPT = 'experiments/mc0/run_mc0pc.py'
DEFAULT_ROOT = Path('/home/rmcfz/seq_vp_tag_results/mc0p_v1c')
CONTROL_ROOT = Path('/home/rmcfz/seq_vp_tag_results/mc0p_v1b')      # read only: the control's u is compared with this batch's cell
QUEUE_UNIT, K1_UNIT, EVAL_UNIT = 'tag-mc0pc-queue', 'tag-mc0pc-k1check', 'tag-mc0pc-eval'
BATCH_SCHEMA = 'bayesreg.mc0pc.batch.v1'
DEFAULT_MIN_HOST_GIB = 15.                                          # the lead, the brief; BZ may change it.  F's own guard stays 10 GiB + pending (F/scripts/run_stage_3d_e19_mri.py:460, 546)
ARM, TIER = 'Q2', 'long'
REST_PAIRS = tuple(f'OASIS_OAS1_{n:04d}_MR1__OASIS_OAS1_{n+1:04d}_MR1' for n in range(443, 457))     # selection.json, pairs 6..19 (the dry run compares them)
GATE = replace(mc0p.RETRY_GATE, own_units=(f'{QUEUE_UNIT}.service',))       # the strict gate of mc0p_v1b; host_floor_gib comes from --min-host-gib


@dataclass(frozen=True)
class Cell(mc.Cell):
    """The same run_id and run directory layout as MC0P; only the unit is this batch's."""
    @property
    def unit(self):
        return f'tag-mc0pc-{self.arm.lower()}-{self.tier}-{self.short}'


CONTROL = Cell(ARM, TIER, core.PAIRS[0])                            # done in mc0p_v1b; run again to compare the field's u


def plan_runs():
    """The control first, then the 14 pairs in the order of selection.json."""
    return [CONTROL]+[Cell(ARM, TIER, pair) for pair in REST_PAIRS]


def require_cell(arm, tier, pair):
    """Only the planned cells: checked before any file of the pair is opened."""
    if pair not in REST_PAIRS+(CONTROL.pair,) or (arm, tier) != (ARM, TIER):
        raise SystemExit(f'this batch runs only {ARM} {TIER} on {CONTROL.short} (the control) and the other 14 development pairs; refused: {arm} {tier} {pair}')
    return Cell(arm, tier, pair)


def root_problem(root):
    """Why this batch may not write at this root, or None: only its own."""
    return None if Path(root).resolve() == Path(DEFAULT_ROOT).resolve() else f'{root}: this batch writes only to {DEFAULT_ROOT}; nothing read, nothing moved'


# ---- start strings ------------------------------------------------------------------------------------------------------------------------
def unit_argv(unit, code, inner, *, log, gpu=True, numba=1, quota=800, runtime=run.RUN_RUNTIME_MAX):
    return run.unit_argv(unit, code, inner, log=log, gpu=gpu, numba=numba, quota=quota, runtime=runtime, script=SCRIPT, memory_max=mc0p.MEMORY_MAX, nice=mc0p.NICE)


def run_argv(cell, code, root, sha='n/a'):
    d = core.run_dir(root, cell)
    return unit_argv(cell.unit, code, ['run-one', '--arm', cell.arm, '--tier', cell.tier, '--pair', cell.pair, '--out', str(d), '--code-sha', sha], log=d/'log.txt')


def eval_argv(code, root):
    return unit_argv(EVAL_UNIT, code, ['evaluate', '--root', str(root)], log=f'{root}/eval.log', gpu=False, numba=mc0p.EVAL_THREADS, quota=mc0p.EVAL_QUOTA, runtime=run.EVAL_RUNTIME_MAX)


def queue_launch(root=None, repo=core.MAIN_REPO, min_host_gib=DEFAULT_MIN_HOST_GIB):
    """The lead's start string of the queue."""
    root = DEFAULT_ROOT if root is None else root
    return run.check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT}', '--collect', f'--working-directory={repo}', f'--property=MemoryMax={mc0p.MEMORY_MAX}', f'--nice={mc0p.NICE}', '--setenv=PYTHONDONTWRITEBYTECODE=1',
                             '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', run.PY, '-B', SCRIPT, 'queue', '--root', str(root), '--min-host-gib', f'{min_host_gib:g}'])


# ---- one run, the evaluation ----------------------------------------------------------------------------------------------------------------------
def read_images(pair):
    """mc0p_core.read_images for a planned cell of this batch: load_pair's firewall view, `fixed` and `moving` (subject guard, sha256 against the manifest)."""
    require_cell(ARM, TIER, pair)
    from bayesreg.io.pair import load_pair
    return load_pair(core.STAGE/pair/'pair.npz', allowlist=core.ALLOWLIST)


def read_pair(pair, labels):
    """run_mc0.read_pair for a planned cell of this batch; only the evaluation calls it."""
    require_cell(ARM, TIER, pair)
    from bayesreg.io.pair import load_pair
    path = core.STAGE/pair/'pair.npz'
    pin = load_pair(path, allowlist=core.ALLOWLIST)
    with np.load(path) as z:
        return pin, {k: np.array(z[k]) for k in labels}


def run_one(o):
    cell = require_cell(o.arm, o.tier, o.pair)
    if Path(o.out).resolve() != core.run_dir(DEFAULT_ROOT, cell).resolve():
        raise SystemExit(f'{o.out}: this batch writes the run of {cell.run_id} only to {core.run_dir(DEFAULT_ROOT, cell)}')
    return mc0p.run_one(o, cell=cell, images=read_images)


def evaluate_all(o):
    return mc0p.evaluate_all(o, plan=plan_runs(), read_pair=read_pair)


# ---- the control, the summary ---------------------------------------------------------------------------------------------------------------------
def control_record(root, reference=None):
    """The u of the control's field.npz in mc0p_v1b and here.  Different is recorded as it is: no failure, nothing stops (the rule of mc0p_v1b's control)."""
    reference = CONTROL_ROOT if reference is None else reference
    a, b = (core.run_dir(r, CONTROL)/'field.npz' for r in (reference, root))
    done = run.read_json(core.run_dir(root, CONTROL)/'queue_done.json') or {}
    record = dict(cell=CONTROL.run_id, control_bitwise=None, u_sha256_v1b=None, u_sha256_v1c=None, control_stop=done.get('control_stop'), role='只记录：不算失败，它的失败也不停队列')
    if not a.is_file() or not b.is_file():
        return dict(record, reason=f'没有比较：缺 {[str(p) for p in (a, b) if not p.is_file()]}')
    va, vb = decision.field_u_sha256(a), decision.field_u_sha256(b)
    return dict(record, control_bitwise=va == vb, u_sha256_v1b=va, u_sha256_v1c=vb)


def f(value, digits=4):
    return '—' if value is None else (f'{value:.{digits}f}' if isinstance(value, float) else str(value))


def write_summary(root, note=None):
    """analysis/long.csv (the columns of MC0P's) and a table of the cells, no reading; an earlier version moved to analysis/previous_<UTC>/."""
    root = Path(root)
    rows = decision.base.collect(root, plan_runs(), decision.SCALE_KEYS, ('oracle', 'retry_of'))
    out = root/'analysis'
    decision.set_aside(out, ('long.csv', 'summary_mc0pc.md'))
    decision.write_long(out/'long.csv', rows)
    control = run.read_json(root/'control.json') or {}
    lines = ['# mc0p_v1c 汇总（由 run_mc0pc.py 写出，没有读数）', '', '开发集，单次观测。Q2 长档的场，通道来自 unet35_v1_ens 的预测（oracle: false），未认证，不是方法，不进结果表；它们是 S_P 其余 14 对的入口。接受 α 的数才是实际进原生层的入口。', '',
             f"控制格 {CONTROL.run_id} 对 mc0p_v1b：control_bitwise {control.get('control_bitwise')}（只记录）。", *([f'结束：{note}'] if note else []), '',
             '| 格 | 状态 | 秒 | 显存峰值 GiB | 接受 α | 接受 α seg35 / seg4 | α = 1 seg35 / seg4 | α = 1 官方 SDlogJ |', '|---|---|---|---|---|---|---|---|']
    for cell in plan_runs():
        row = rows.get((cell.arm, cell.tier, cell.pair))
        if row is None:
            lines.append(f'| {cell.run_id} | 没有运行 | — | — | — | — | — | — |')
            continue
        x, at = row.get('metrics') or {}, row.get('at_accepted_alpha') or {}
        lines.append(f"| {cell.run_id} | {row['status']} | {f(row['seconds_total'], 1)} | {f(row['device_peak_reserved_gib'], 2)} | {f(row['accepted_alpha'], 5)} | {f(at.get('seg35'))} / {f(at.get('seg4'))} | {f(x.get('seg35'))} / {f(x.get('seg4'))} | {f(x.get('sdlogj_official'))} |")
    (out/'summary_mc0pc.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return rows


# ---- the queue ------------------------------------------------------------------------------------------------------------------------------------
class GateClosed(run.Stop):
    """The gate stayed closed for its time: queue_stop.json, what has finished is summed up (the brief, the disk floor)."""


def wait(system, root, gpu, minutes, log, gate):
    try:
        run.wait_for_gate(system, root, gpu, minutes, log, gate)
    except run.Stop as stop:
        raise GateClosed(str(stop)) from None


def run_cells(o, system, root, log, code, sha, plan, gate):
    k1 = root/'k1_check_gpu.json'
    if not k1.is_file():
        wait(system, root, True, o.wait_minutes, log, gate)
        log(f'launch {K1_UNIT}')
        system.run(mc0p.k1_argv(code, root, K1_UNIT), root/'k1_check_gpu.systemd_run.log')
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
        wait(system, root, True, o.wait_minutes, log, gate)
        d.mkdir(parents=True)
        argv = run_argv(cell, code, root, sha)
        (d/'cmd.txt').write_text(shlex.join(argv)+'\n')
        log(f'launch {cell.unit}; load {system.load1()}; /mnt/c free {system.host_free_gib():.0f} GiB')
        tick = system.now()
        rc = system.run(argv, d/'systemd_run.log')
        seconds = system.now()-tick
        record = run.read_json(d/'run.json')
        failed = rc != 0 or record is None
        stop = run.arm_stop_reason(cell, record, seconds, k=mc.ARMS[cell.arm]['k'], vram_limit=mc.VRAM_LIMIT_BYTES)
        control_stop = None
        if cell == CONTROL:                             # a record only: its failure stops nothing (mc0p_v1b's rule); if K = 35 really cannot run, the next cell stops the arm
            stop, control_stop = None, stop
        run.write_json(d/'queue_done.json', dict(run_id=cell.run_id, unit=cell.unit, rc=rc, unit_seconds=seconds, state='failed' if failed else record['status'], arm_stop=stop, **({} if control_stop is None else dict(control_stop=control_stop)), ended_utc=run.utc()))
        log(f"ended {cell.run_id}: rc {rc}, {seconds:.0f} s, {'failed' if failed else record['status']}" + (f'; arm stops: {stop}' if stop else '') + (f'; the control, nothing stops: {control_stop}' if control_stop else ''))
        if record is not None and (record.get('scale') or {}).get('images_equal_a5p_inputs') is not True:
            raise run.Stop(f"{cell.run_id}: Q0's images are not A5P's inputs bit for bit, or the check did not finish: {record.get('reason')}")
        if record is not None and record.get('oracle') is not False:
            raise run.Stop(f"{cell.run_id}: run.json oracle {record.get('oracle')}: not what this batch is about: {record.get('reason')}")
        if cell == CONTROL:
            control = control_record(root)
            run.write_json(root/'control.json', control)
            log(f"control_bitwise {control['control_bitwise']}: recorded, no failure")
        failures = failures+1 if failed else 0
        if failures >= run.MAX_CONSECUTIVE_FAILURES:
            raise run.Stop(f'{failures} failures in a row')
    return end


def finish(system, root, code, plan, log, end):
    """The evaluation unit, the summary, queue_end.json.  Returns the unit's exit code."""
    log(f'launch {EVAL_UNIT}')
    rc = system.run(eval_argv(code, root), root/'eval.systemd_run.log')
    write_summary(root, end)
    run.write_json(root/'queue_end.json', dict(end=end, eval_rc=rc, ledger=run.ledger(root, plan), control=run.read_json(root/'control.json'), utc=run.utc()))
    return rc


def queue_body(o, system, root, log):
    if (root/'queue_stop.json').exists():               # evidence, kept under its time
        (root/'queue_stop.json').rename(root/f"queue_stop_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    code, sha = system.code_checkout(root, o.commit)
    plan = plan_runs()
    gate = replace(GATE, host_floor_gib=o.min_host_gib)
    batch = run.read_json(root/'batch.json')
    if batch is None:
        run.write_json(root/'batch.json', dict(schema=BATCH_SCHEMA, code_sha=sha, frozen_commit=core.FROZEN_COMMIT, started_utc=run.utc(), plan=[c.run_id for c in plan], control=CONTROL.run_id, control_against=str(CONTROL_ROOT),
                                               caps=dict(run_seconds=core.RUN_CAP_SECONDS, run_units_seconds=run.GPU_CAP_SECONDS, device_gib=mc.VRAM_LIMIT_BYTES/2**30, memory_max=mc0p.MEMORY_MAX, nice=mc0p.NICE),
                                               gate=dict(own_units=list(gate.own_units), load_limit=gate.load_limit, gpu_free_mib=gate.gpu_free_mib, mem_available_gib=gate.mem_available_gib, home_floor_gb=run.HOME_FLOOR_GB, host_floor_gib=gate.host_floor_gib),
                                               evaluation=dict(numba_threads=mc0p.EVAL_THREADS, cpu_quota_percent=mc0p.EVAL_QUOTA, fallback_workers=mc0p.EVAL_POOL)))
    elif batch['code_sha'] != sha:
        raise run.Stop(f"batch.json is at {batch['code_sha']}, the code at {sha}")
    elif batch['plan'] != [c.run_id for c in plan]:
        raise run.Stop('batch.json has another plan')
    log(f'start: {len(plan)} runs; /mnt/c floor {o.min_host_gib:g} GiB')
    try:
        end = run_cells(o, system, root, log, code, sha, plan, gate)
        stopped = run.ledger(root, plan)['arms_stopped']
        if stopped:                                     # exit 0 as in MC0P; `end`, the summary and the last log line say it (review minor 7): the entries after it are missing
            end = f'{end}; arm stopped, the later entries are missing: {json.dumps(stopped, ensure_ascii=False)}'
        wait(system, root, False, o.wait_minutes, log, gate)
    except GateClosed as stop:                          # what has finished is summed up; the disk floor does not hold back the evaluation, the rest of the gate does
        why = run.busy_reason(system, root, False, replace(gate, host_floor_gib=None))
        if why is None and any((core.run_dir(root, c)/'field.npz').is_file() for c in plan):
            finish(system, root, code, plan, log, f'STOP: {stop}')
        else:
            write_summary(root, f'STOP: {stop}（没有评测：{why or "没有场"}；手工 evaluate、summarize）')
        raise
    rc = finish(system, root, code, plan, log, end)
    log(f'{end}.  Handed to the lead: S_P starts only from the lead (experiments/seed/run_seed19.py)')
    if rc != 0:
        raise run.Stop(f'{EVAL_UNIT}: exit {rc} (metrics_error.json in the run directories, or the {run.EVAL_RUNTIME_MAX} s limit of the unit); the summary is written, a field without metrics.json is not an entry yet')
    return 0


def queue(o, system=None):
    problem = root_problem(o.root)
    if problem:                                         # before the lock file: nothing is created or touched
        print(problem, file=sys.stderr)
        return 2
    return run.queue(o, system, body=queue_body, prefix='mc0p')


# ---- dry run --------------------------------------------------------------------------------------------------------------------------------------
def dry_run(root=None, stage=None, out=print, unet=None, seg_root=None, a5p=None, allowlist=None, control_root=None):
    """Real files only up to the npy headers of `fixed`, `moving` and of the predictions' `p`, the manifest through the firewall's reader, the predictions' provenance.json, the control's run.json.  No truth label."""
    root = DEFAULT_ROOT if root is None else root
    stage = core.STAGE if stage is None else stage
    seg_root = mc.SEG_ROOT if seg_root is None else seg_root
    a5p = mc.A5P_INPUTS if a5p is None else a5p
    allowlist = core.ALLOWLIST if allowlist is None else allowlist
    control_root = CONTROL_ROOT if control_root is None else control_root
    plan = plan_runs()
    gate = replace(GATE, host_floor_gib=DEFAULT_MIN_HOST_GIB)
    out(f'mc0p_v1c dry run ({run.utc()}): no optimization, nothing written; real files read up to the npy headers of fixed, moving and p; no label member.')
    out(f'{len(plan)} runs, root {root}; root exists: {Path(root).exists()}; cap per run {core.RUN_CAP_SECONDS} s, run units {run.GPU_CAP_SECONDS} s, device {mc.VRAM_LIMIT_BYTES/2**30} GiB; MemoryMax {mc0p.MEMORY_MAX}, nice {mc0p.NICE}')
    out(f'gate: own units {gate.own_units}, load < {gate.load_limit}, GPU free >= {gate.gpu_free_mib} MiB, MemAvailable >= {gate.mem_available_gib} GiB, /home >= {run.HOME_FLOOR_GB} GB, /mnt/c >= {gate.host_floor_gib} GiB (--min-host-gib); no allow list, no hold for a chain')
    out("the scale rule: ||g_D|| of Q0's images at u = 0 (mc0_core.data_grad_norm, level_grad_norms): Q0's optimized field is not needed, no Q0 cell runs here")
    out('  # | arm | K | tier | iterations | sigma_field | pair | role | unit | run_id')
    config = core.tier_config(TIER)
    for i, c in enumerate(plan, 1):
        out(f"{i:3d} | {c.arm} | {mc.ARMS[c.arm]['k']} | {c.tier} | {'/'.join(map(str, config['iterations']))} | {config['sigma_field']} | {c.short} | {'control, record only' if c == CONTROL else 'entry of S_P'} | {c.unit} | {c.run_id}")
    problems = []
    if len(plan) != 15:
        problems.append(f'{len(plan)} runs, not 15')
    selection = json.loads((Path(stage)/'selection.json').read_text())
    rest = tuple(row['pair'] for row in selection['pairs'][5:19])
    out(f"selection.json pairs 6..19 == the plan's 14 pairs: {rest == REST_PAIRS}")
    if rest != REST_PAIRS:
        problems.append('pairs 6..19 of selection.json differ')
    control = run.read_json(core.run_dir(control_root, CONTROL)/'run.json') or {}
    out(f"control in {control_root} (read only): status {control.get('status')}, u sha256 {(control.get('field') or {}).get('u_sha256')}")
    if control.get('status') != 'done':
        problems.append(f'the control has no finished run in {control_root}')
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
    for pair in REST_PAIRS:
        d = Path(stage)/pair
        _, metadata, manifest = _read_manifest(d/'manifest.json')
        check_pair_subjects(manifest['fixed_subject'], manifest['moving_subject'], guard)
        subjects.update({manifest['fixed_subject']: None, manifest['moving_subject']: None})
        heads = {m: run.npz_header(d/'pair.npz', m) for m in ('fixed', 'moving')}
        theirs = {m: run.npz_header(Path(a5p)/pair/'pair.npz', m) for m in ('fixed', 'moving')} if (Path(a5p)/pair/'pair.npz').is_file() else None
        out(f"{core.short_name(pair)}: {d/'pair.npz'}; subjects on the allowlist ({manifest['fixed_subject']}, {manifest['moving_subject']}); spacing {metadata['spacing_xyz']}; " + ', '.join(f'{m} {h[0]} {h[1]}' for m, h in heads.items())+f"; A5P's images {theirs}")
        if len({tuple(h[0]) for h in heads.values()}) != 1:
            problems.append(f'{pair}: image shapes differ')
        if theirs != heads:
            problems.append(f"{pair}: A5P's input is missing or has other shapes")
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
    out(f"colour table sha256 {core.COLOUR_TABLE_SHA256} pinned and read: {len(core.load_table())} labels")
    out(f"frozen {core.FROZEN}: preopt.py sha256 equal to the pin: {core.sha_file(core.FROZEN/'jd_curl/three_d/preopt.py') == core.FROZEN_PREOPT_SHA256}; the units' PYTHONPATH puts it first")
    out(f'/home free {shutil.disk_usage("/home").free/2**30:.0f} GB; /mnt/c free {shutil.disk_usage("/mnt/c").free/2**30:.0f} GiB')
    code = Path(root)/'code'
    out('--- start strings (the queue: the lead; the others: the queue, one per run; the code at $ROOT/code)')
    out(shlex.join(queue_launch(root)))
    out(shlex.join(mc0p.k1_argv(code, root, K1_UNIT)))
    for c in plan:
        out(shlex.join(run_argv(c, code, root, '<sha>')))
    out(shlex.join(eval_argv(code, root)))
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv == ['--dry-run']:
        return dry_run()
    p = argparse.ArgumentParser(description='mc0p_v1c: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(DEFAULT_ROOT))
    q.add_argument('--commit', default=None)
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--min-host-gib', type=float, default=DEFAULT_MIN_HOST_GIB)
    r = sub.add_parser('run-one')
    r.add_argument('--arm', required=True)
    r.add_argument('--tier', required=True)
    r.add_argument('--pair', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--device', default='cuda')
    r.add_argument('--code-sha', default='n/a')
    for name in ('evaluate', 'summarize'):
        sub.add_parser(name).add_argument('--root', default=str(DEFAULT_ROOT))
    o = p.parse_args(argv)
    if o.command == 'queue':
        return queue(o)
    if o.command == 'run-one':
        return run_one(o)
    problem = root_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if o.command == 'evaluate':
        return evaluate_all(o)
    write_summary(o.root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
