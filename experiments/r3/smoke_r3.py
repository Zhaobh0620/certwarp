"""R3's GPU smoke before the chain (runs/20261111-r3/plan.md section 6; the brief: L 400 steps, N a few epochs, R2's smoke's rules): in another root, the training set only, no lock, no member
of the chain.

    python -B experiments/r3/smoke_r3.py --dry-run
    python -B experiments/r3/smoke_r3.py prepare --repo <a clean checkout with the sealed lists and a data link>
        foreground, after the nnU-Net root is prepared (run_nn.py prepare / preprocess --out <NN_ROOT>, run_r3.py nn-splits): refused when the smoke ran; the snapshot <SMOKE>/snapshot of --repo's
        HEAD (run_r3.SNAPSHOT_PATHS_R3; no sealed list: the guard reads them from SNAPSHOT.json's source_repo, contract C23.7-4 revision 6n), `cli check-inputs --profile v3_large --require-snapshot`.
    python -B <SMOKE>/snapshot/experiments/r3/smoke_r3.py run       in the unit tag-r3-smoke (GPU), behind R3's gate:
          (1) L: cli train --profile v3_large --fold A --seed 3, UF's config with L's schedule cut to SMOKE_STEPS steps (eval every SMOKE_EVAL, every step logged), out <SMOKE>/member;
          (2) N: nnUNetv2_train of nnA_s0 (the real trainer, the real preprocessed data and splits_final.json of <NN_ROOT>) in <SMOKE>/nnUNet_results_smoke, stopped when its 5th epoch starts
              (4 epochs and their checkpoints complete; nn_r3.smoke); then one real inference (smoke_infer, the review's I2): infer_r3nn.py on that run's checkpoint_best.pth, the first training
              subject of fold A (its image only), and nn_r3.check_outputs on it;
          (3) the synthetic chains of run_r3 with --device cuda in <SMOKE>/synthetic (in this process; an Exception or SystemExit is a synthetic problem with the traceback's last lines);
          then <SMOKE>/smoke_r3.json (the code record of require_snapshot, the sha256 of splits_final.json: prepare pins it) and STOP.
The rule (smoke_r3.json `reading`): 'stop' when
  L: the record's recipe is not v3_large or AMP was off; a logged loss is not finite; the member did not run its steps; the projected wall per step of the real schedule -- (data + 80000 x the
     median step after the first 10 + 160 x the median evaluation) / 80000 -- is above .30 s (the member's wall stop is 28800 s / 80000 = .36 s); the GPU allocated peak is above 18 GiB;
  N: nn_r3.smoke_reading (4 epochs, our split of 98 + 8, the seed line, finite losses, 250 x the median epoch from the 2nd on <= 8 h, host PSS <= 11 GiB, GPU used <= 22528 MiB);
  N inference: inference_reading (no refusal or failure; the check passed: grid, read-back, the model's labels against nnU-Net's own file, the checkpoint, reserved <= 9 GiB; host PSS <= 11 GiB);
  the synthetic chains on the GPU have a problem;
otherwise 'pass'.  The weight files (L about 200 MB, nnU-Net's checkpoints) are listed with their sha256 for the lead to delete after the reading; the tool deletes nothing.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(HERE), str(HERE.parent/'r2'), str(HERE.parent/'uf'), str(HERE.parent/'ul'), str(HERE.parent/'unv2'), str(HERE.parent/'nn')]

import r3_common as r3c  # noqa: E402
import run_r3  # noqa: E402

UNIT = 'tag-r3-smoke'
SMOKE_STEPS, SMOKE_EVAL = 400, 200
STEP_LIMIT = .30
GPU_PEAK_LIMIT_GIB = 18.
TIMEOUTS = dict(train=3600)
MEMORY_MAX, MEMORY_SWAP_MAX, THREADS = '12G', '2G', 4
RUNTIME_MAX = 3*3600
TRACEBACK_LINES = 20


def smoke_config():
    """L's --set values with the smoke's schedule: SMOKE_STEPS steps (poly over them), no early stop, the last step, every step logged, a wall stop of 20 minutes."""
    return dict(r3c.l_train_config(), max_iters=SMOKE_STEPS, min_iters=SMOKE_STEPS, patience=SMOKE_STEPS, eval_every=SMOKE_EVAL, log_every=1, max_wall_seconds=1200.)


def l_argv(out, *, python=r3c.PY, device='cuda', inputs=(), config=None):
    cli = [python, '-B', '-m', 'bayesreg.seg.unet.cli']
    return cli+['train', '--profile', r3c.L_PROFILE, '--fold', 'A', '--seed', str(r3c.L_SEED), *inputs, '--out', str(Path(out)/'member'), '--device', device, *run_r3.sets(smoke_config() if config is None else config)]


def measure(member_dir, *, real_steps=r3c.L_STEPS, eval_every=r3c.EVAL_EVERY):
    """What the L member of the smoke measured (train_log.jsonl, train_record.json; the training subjects' calib Dice only): finiteness of every numeric field of the train events, the medians,
    the projection, the peaks (smoke_r2.measure without the target terms)."""
    member_dir = Path(member_dir)
    rows = [json.loads(line) for line in (member_dir/'train_log.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    record = json.loads((member_dir/'train_record.json').read_text(encoding='utf-8')) if (member_dir/'train_record.json').is_file() else {}
    train = [r for r in rows if r.get('event') == 'train']
    evals = [r for r in rows if r.get('event') == 'eval']
    start = next((r for r in rows if r.get('event') == 'start'), {})
    values = [r.get('loss') for r in train]+[v for r in train for k, v in r.items() if k not in ('loss',) and isinstance(v, float)]
    finite = bool(train) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) and not any(r.get('event') == 'nonfinite_loss' for r in rows)
    late = [r['step_seconds'] for r in train if r.get('it', 0) >= 10]
    step = statistics.median(late) if late else None
    evaluation = statistics.median([r['eval_seconds'] for r in evals]) if evals else 0.
    data = start.get('data_ready_seconds') or 0.
    projected = None if step is None else (data+real_steps*step+(real_steps//eval_every)*evaluation)/real_steps
    memory = record.get('memory') or {}
    peaks = [r.get('gpu_peak_allocated_gib') for r in rows if r.get('gpu_peak_allocated_gib') is not None]+([memory['gpu_peak_allocated_gib']] if memory.get('gpu_peak_allocated_gib') is not None else [])
    return dict(finite=finite, n_logged_steps=len(train), stopped=record.get('stopped'), iterations_run=record.get('iterations_run'), recipe=record.get('recipe'), amp=(record.get('config') or {}).get('amp'),
                step_seconds_median=step, eval_seconds_median=evaluation, data_ready_seconds=data, projected_wall_per_step=projected, projected_member_hours=None if projected is None else projected*real_steps/3600,
                gpu_peak_allocated_gib=max(peaks) if peaks else None, gpu_peak_reserved_gib=memory.get('gpu_peak_reserved_gib'), host_peak_rss_gib=memory.get('host_peak_rss_gib'))


def l_reading(measured, *, steps_planned=SMOKE_STEPS):
    why = []
    if measured.get('recipe') != r3c.L_PROFILE:
        why.append(f"L: the member's recipe is {measured.get('recipe')!r}, not {r3c.L_PROFILE}")
    if measured.get('amp') is not True:
        why.append(f"L: AMP was {measured.get('amp')!r}, not on")
    if not measured.get('finite'):
        why.append('L: a logged value is not finite')
    if measured.get('stopped') != 'max_iters' or measured.get('iterations_run') != steps_planned:
        why.append(f"L: the member stopped {measured.get('stopped')!r} after {measured.get('iterations_run')} of {steps_planned} steps")
    projected = measured.get('projected_wall_per_step')
    if projected is None or projected > STEP_LIMIT:
        why.append(f'L: projected wall per step {projected} s > {STEP_LIMIT} s')
    peak = measured.get('gpu_peak_allocated_gib')
    if peak is None or peak > GPU_PEAK_LIMIT_GIB:
        why.append(f'L: GPU allocated peak {peak} GiB > {GPU_PEAK_LIMIT_GIB} GiB')
    return why


def smoke_subject(split, allowlist, *, training_source=None):
    """The one subject of the smoke's inference: the first training subject of fold A (sorted), admitted as run_r3.nn_subjects admits the calibration sets (LabelGuard.check, refuse_sealed);
    only its image is read, never its labels."""
    from bayesreg.io.guard import refuse_sealed
    from bayesreg.seg.unet.data import LabelGuard
    from bayesreg.seg.unet.pins import TRAINING_SOURCE
    subject = split.subjects('A', 'train')[0]
    refuse_sealed(subject, 'smoke_r3.smoke_subject')
    LabelGuard(split, allowlist, training_source=training_source or TRAINING_SOURCE).check(subject)
    return subject


def smoke_infer(out, *, root=r3c.NN_ROOT, name='nnA_s0', subject=None, data_root=Path('/mnt/d/data')):
    """One real nnU-Net inference before the chain (the review's I2): infer_r3nn.py (the chain's child) on the smoke's own checkpoint (checkpoint_best.pth of the 4 epochs, in the smoke's results
    root), one training subject (its image), then nn_r3.check_outputs -- the grid against the image's header, the reader's read-back against the mean's argmax, the model's labels against
    nnU-Net's own label file, the checkpoint, the device cap.  Returns what happened (a refusal or a failed check is recorded, not raised: the rule reads it)."""
    import nn_r3
    out = Path(out)
    results = out/'nnUNet_results_smoke'
    checkpoint = nn_r3.fold_dir(name, results)/nn_r3.SMOKE_CHECKPOINT
    if not checkpoint.is_file():
        return dict(problems=[f'{checkpoint}: no such file after the 4 epochs'])
    try:
        if subject is None:
            from bayesreg.seg.unet.pins import DEFAULT_ALLOWLIST, load_pinned_allowlist
            subject = smoke_subject(run_r3.real_split(), load_pinned_allowlist(DEFAULT_ALLOWLIST)[0])
        records = {name: dict(final_checkpoint_sha256=r3c.sha256_file(checkpoint))}
        argv = nn_r3.infer_argv([name], input_dir=out/'nn_infer'/'input', out_dir=out/'nn_infer', root=root, results=results, checkpoint=nn_r3.SMOKE_CHECKPOINT)
        check = nn_r3.infer_set(out/'nn_infer', [subject], [name], root=root, env=nn_r3.nn_env(root, results=results, gpu=True), data_root=data_root, argv=argv, records=records,
                                checkpoint=nn_r3.SMOKE_CHECKPOINT, log=run_r3.log)
    except Exception as exc:                                       # a refusal, a failed run or a failed check: the smoke's reading is stop, the evidence stays in <SMOKE>/nn_infer
        return dict(subject=subject, checkpoint=str(checkpoint), problems=[f'{type(exc).__name__}: {exc}'])
    return dict(subject=subject, checkpoint=str(checkpoint), checkpoint_sha256=records[name]['final_checkpoint_sha256'], problems=[], passed=check['passed'], rows=check['rows'], run=check['run'],
                infer_record=r3c.read_json(out/'nn_infer'/'infer_record.json'))


def inference_reading(result):
    """The rule of the smoke's inference: no problem; the check passed with the reader's argmax the mean's on every voxel and the model's labels nnU-Net's own (<= 1e-4); one subject; the
    host PSS of the child <= 11 GiB (the reserved GPU peak <= 9 GiB is in the check)."""
    import nn_r3
    if result is None:
        return ['not run']
    why = list(result.get('problems') or [])
    if why:
        return why
    rows = result.get('rows') or {}
    if result.get('passed') is not True or len(rows) != 1:
        why.append(f"the check: passed {result.get('passed')}, {len(rows)} subjects")
    for subject, row in rows.items():
        if row.get('n_reader_not_mean') != 0 or any(v/row['n_voxels'] > r3c.NN_GATE_TOLERANCE for v in (row.get('n_model_not_own') or {}).values()):
            why.append(f'{subject}: reader {row.get("n_reader_not_mean")}, model against its own file {row.get("n_model_not_own")}')
    pss = (result.get('run') or {}).get('host_peak_pss_gib')
    if pss is None or pss > nn_r3.HOST_LIMIT_GIB:
        why.append(f'the child\'s host PSS {pss} GiB above {nn_r3.HOST_LIMIT_GIB} GiB')
    return why


def launch_argv(*, out=r3c.SMOKE):
    snapshot = Path(out)/'snapshot'
    argv = ['systemd-run', '--user', f'--unit={UNIT}', '--collect', f'--working-directory={snapshot}', f'--property=MemoryMax={MEMORY_MAX}', f'--property=MemorySwapMax={MEMORY_SWAP_MAX}', f'--property=RuntimeMaxSec={RUNTIME_MAX}',
            '--setenv=PYTHONDONTWRITEBYTECODE=1', *[f'--setenv={name}={THREADS}' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')],
            r3c.PY, '-B', str(snapshot/'experiments/r3/smoke_r3.py'), 'run', '--out', str(out)]
    if 'pytest' in ' '.join(argv) or 'seq-vp-' in ' '.join(argv):
        raise ValueError('a start string must not name the test runner or the seq-vp- prefix')
    return argv


def nn_root_problems(root=r3c.NN_ROOT):
    """The nnU-Net root is prepared: the preprocess record, the plans (NN's), splits_final.json."""
    import nn_r3
    problems = [] if (Path(root)/'preprocess_record.json').is_file() else [f'{root}/preprocess_record.json: no such file (run_nn.py prepare, preprocess --out {root})']
    problems += nn_r3.plans_problems(root)
    if not nn_r3.splits_path(root).is_file():
        problems.append(f'{nn_r3.splits_path(root)}: no such file (run_r3.py nn-splits)')
    return problems


def prepare(out, repo):
    import subprocess
    from bayesreg.seg.unet.snapshot import make_snapshot, verify_snapshot
    out = Path(out)
    ran = [str(p) for p in (out/'smoke_r3.json', out/'member', out/'nnUNet_results_smoke') if p.exists()]
    if ran:
        raise run_r3.Stop(f'{ran} there: the smoke ran; its root is kept whole until the chain is over, never prepared again')
    problems = nn_root_problems()
    if problems:
        raise run_r3.Stop(f'the nnU-Net root is not prepared: {problems}')
    out.mkdir(parents=True, exist_ok=True)
    run_r3.set_run_log(out)
    snapshot = out/'snapshot'
    record = verify_snapshot(snapshot) if snapshot.exists() else make_snapshot(repo, snapshot, paths=run_r3.SNAPSHOT_PATHS_R3)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run([r3c.PY, '-B', '-m', 'bayesreg.seg.unet.cli', 'check-inputs', '--profile', r3c.L_PROFILE, '--require-snapshot'], cwd=str(snapshot), env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise run_r3.Stop(f'the pinned-input check failed: {done.stdout[-2000:]} {done.stderr[-2000:]}')
    run_r3.log(f"smoke snapshot {snapshot} at {record['commit']}; pinned inputs checked (no label or image read)")
    return record


def run(out, *, gate=True, wait_minutes=run_r3.WAIT_MINUTES, python=r3c.PY, device='cuda', inputs=(), config=None, nn_root=r3c.NN_ROOT, nn=True, synthetic_device='cuda', env=None, code=None):
    """The unit's work: the gate, L, N, the synthetic chains, smoke_r3.json.  Returns the document.  config, nn=False: the tests' tiny L and no nnU-Net."""
    import nn_r3
    out = Path(out)
    if (out/'smoke_r3.json').exists() or (out/'member').exists():
        raise run_r3.Stop(f'{out}: the smoke ran before; kept, not rerun')
    run_r3.set_run_log(out)
    problems = nn_root_problems(nn_root) if nn else []
    if problems:
        raise run_r3.Stop(f'the nnU-Net root is not prepared: {problems}')
    if nn:
        import run_nn
        from bayesreg.seg.unet.snapshot import code_record
        run_nn.require_preprocessed(nn_root, code=code_record())                 # as nn-train checks it in the chain, from this snapshot (links, cases, gt links; ids only)
    splits_sha256 = r3c.sha256_file(nn_r3.splits_path(nn_root)) if nn else None
    config = smoke_config() if config is None else config
    state, chain_log = out/'state', out/'smoke.log'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1') if env is None else env
    if gate:
        run_r3.ufrun.wait_for_gate(wait_minutes, problems=lambda: run_r3.gate_problems(UNIT))
    argv = l_argv(out, python=python, device=device, inputs=inputs, config=config)

    def action():
        code_, timed_out = run_r3.ufrun.run_command(argv, cwd=run_r3.CODE_ROOT, log_path=chain_log, timeout=TIMEOUTS['train'], env=env)
        return 124 if timed_out else code_
    run_r3.ufrun.run_step('train', action, state=state, log_path=chain_log, label=shlex.join(argv))
    measured = measure(out/'member'/r3c.member_name('A', r3c.L_SEED))
    why = l_reading(measured, steps_planned=config['max_iters'])
    nn_measured, nn_hours, nn_inference = None, None, None
    if nn:
        nn_measured = nn_r3.smoke(out, root=nn_root, device=device)
        nn_reading, nn_why, nn_hours = nn_r3.smoke_reading(nn_measured, root=nn_root)
        why += [f'N: {w}' for w in nn_why]
        nn_inference = smoke_infer(out, root=nn_root)
        why += [f'N inference: {w}' for w in inference_reading(nn_inference)]
    try:
        import r3_synthetic
        synthetic = list(r3_synthetic.synthetic_chain(out/'synthetic', env, synthetic_device))
    except (Exception, SystemExit) as exc:
        tail = traceback.format_exc().rstrip().splitlines()[-TRACEBACK_LINES:]
        synthetic = [f'the synthetic chain raised {type(exc).__name__}: {exc}', 'traceback (last lines): '+' | '.join(line.strip() for line in tail)]
        run_r3.log('the synthetic chain raised:\n'+'\n'.join(tail))
    finally:
        run_r3.set_run_log(out)
    why += [f'the synthetic chain on the GPU: {p}' for p in synthetic]
    member = out/'member'/r3c.member_name('A', r3c.L_SEED)
    weights = {f'member/{p.name}': dict(bytes=p.stat().st_size, sha256=r3c.sha256_file(p)) for p in sorted(member.glob('*.pt'))}
    if nn_measured:
        weights.update({f'nnUNet_results_smoke/{k}': dict(bytes=nn_measured['checkpoint_bytes'][k], sha256=v) for k, v in nn_measured['checkpoint_sha256'].items()})
    document = dict(schema='bayesreg.r3.smoke.v2', reading='stop' if why else 'pass', why=why, l=measured, nn=nn_measured, nn_hours_for_250=nn_hours, nn_inference=nn_inference, synthetic_problems=synthetic, config=config, code=code,
                    splits_sha256=splits_sha256, rule=dict(step_limit=STEP_LIMIT, gpu_peak_limit_gib=GPU_PEAK_LIMIT_GIB, steps=SMOKE_STEPS, nn_epochs=nn_r3.SMOKE_EPOCHS, nn_limit_hours=nn_r3.NN_LIMIT_HOURS),
                    weights_to_delete_after_the_reading=weights, nn_inference_files_to_delete_after_the_reading=str(out/'nn_infer'), plan=r3c.plan_record(run_r3.CODE_ROOT), reads_development_truth=False, takes_a_read_lock=False,
                    note='GPU 冒烟：只读训练集（LabelGuard、nnU-Net 的训练），不读开发集真值，不取读取锁，不在链条的根里；读数 stop 就在开训前停下交总负责人。', utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    r3c.write_json_once(out/'smoke_r3.json', document)
    run_r3.log(f"smoke: {document['reading']} {why}")
    return document


def dry_run(log=print):
    problems = []
    log(f'== R3 GPU smoke, dry run; nothing written; the root {r3c.SMOKE} ({"EXISTS" if r3c.SMOKE.exists() else "not there"})')
    log('prepare: '+shlex.join([r3c.PY, '-B', 'experiments/r3/smoke_r3.py', 'prepare', '--repo', '<a clean checkout with the sealed lists and a data link>']))
    log('launch:  '+shlex.join(launch_argv()))
    log(f"L [gate, limit {TIMEOUTS['train']} s]: {shlex.join(l_argv(r3c.SMOKE))}")
    import nn_r3
    log(f"N: {shlex.join(nn_r3.train_argv('nnA_s0'))}  (environment nnunet, nnUNet_results={r3c.SMOKE/'nnUNet_results_smoke'}, stopped when epoch {nn_r3.SMOKE_EPOCHS} starts)")
    smoke = r3c.SMOKE
    log(f"N inference: {shlex.join(nn_r3.infer_argv(['nnA_s0'], input_dir=smoke/'nn_infer'/'input', out_dir=smoke/'nn_infer', results=smoke/'nnUNet_results_smoke', checkpoint=nn_r3.SMOKE_CHECKPOINT))}"
        '  (the first training subject of fold A, its image only; then nn_r3.check_outputs)')
    log('synthetic: r3_synthetic.synthetic_chain(<SMOKE>/synthetic, device cuda) in this process')
    log(f'the nnU-Net root now: {nn_root_problems() or "prepared"} (the smoke needs it; not a dry-run problem)')
    problems += run_r3.snapshot_path_problems(log=log)
    log(f'the rule: L recipe {r3c.L_PROFILE}, AMP on, finite, {SMOKE_STEPS} steps, projected <= {STEP_LIMIT} s/step over {r3c.L_STEPS}, GPU <= {GPU_PEAK_LIMIT_GIB} GiB; N 4 epochs, split 98 + 8, the seed, finite losses, '
        f'{r3c.NN_EPOCHS} epochs <= {nn_r3.NN_LIMIT_HOURS} h, host <= {nn_r3.HOST_LIMIT_GIB} GiB, GPU <= {nn_r3.GPU_LIMIT_MIB} MiB; N inference: the check passed (grid, read-back, '
        f'labels against nnU-Net\'s own file <= {r3c.NN_GATE_TOLERANCE:g}, checkpoint, reserved <= {r3c.NN_DEVICE_CAP_GIB:g} GiB), host <= {nn_r3.HOST_LIMIT_GIB} GiB; the synthetic chains without a problem')
    log('〔估算〕the unit: the gate, L about 3 minutes (400 steps at about .22 s, 2 evaluations, data about 30 s), N about 6-8 minutes (4 epochs at about 82 s, the data loaders), the inference '
        'about 1-2 minutes, the synthetic chains a few minutes: about 15-20 minutes of GPU')
    if r3c.SMOKE.exists():
        problems.append(f'{r3c.SMOKE} exists')
    log(f"== dry run problems: {'; '.join(problems) if problems else 'none'}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true')
    sub = parser.add_subparsers(dest='command')
    p = sub.add_parser('prepare')
    p.add_argument('--repo', type=Path, default=r3c.MAIN_REPO)
    p.add_argument('--out', type=Path, default=r3c.SMOKE)
    sub.add_parser('print-launch').add_argument('--out', type=Path, default=r3c.SMOKE)
    p = sub.add_parser('run')
    p.add_argument('--out', type=Path, default=r3c.SMOKE)
    p.add_argument('--wait-minutes', type=int, default=run_r3.WAIT_MINUTES)
    args = parser.parse_args(argv)
    import run_nn
    try:
        if args.dry_run:
            return 0 if not dry_run() else 8
        if args.command == 'prepare':
            prepare(args.out, args.repo)
        elif args.command == 'print-launch':
            print(shlex.join(launch_argv(out=args.out)))
        elif args.command == 'run':
            from bayesreg.seg.unet.snapshot import require_snapshot
            require_snapshot()
            document = run(args.out, wait_minutes=args.wait_minutes, code=run_r3.run_r2.snapshot_identity(run_r3.CODE_ROOT))
            return 0 if document['reading'] == 'pass' else 8
        else:
            parser.print_help()
            return 2
    except (run_r3.Stop, run_nn.Stop, run_r3.CodeNotClean) as exc:
        run_r3.log(f'STOP: {exc}')
        return 8
    return 0


if __name__ == '__main__':
    sys.exit(main())
