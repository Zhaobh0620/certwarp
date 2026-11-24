"""R2's GPU smoke before the training (runs/20261110-r2/plan.md section 3.3; review I3, the lead's ruling of 11-10 ~10:59 PM CST): a few hundred steps of the recipe r2s on the real training set, and the
synthetic chain on the GPU, in another root; no development truth, no read lock, no member of the chain.

    python -B experiments/r2/smoke_r2.py --dry-run
        the start strings, the steps, the rule, targets.json there or not; nothing written, nothing read but file names.
    python -B experiments/r2/smoke_r2.py prepare --repo <a clean checkout with the sealed lists and a data link>
        foreground, after the split (targets.json): refused when <SMOKE>/smoke_r2.json or <SMOKE>/member is there (the smoke ran: the diff review 2's minor 1(a)); the snapshot <SMOKE>/snapshot of
        --repo's HEAD, `cli check-inputs --profile r2_small --require-snapshot` from it (no label, no image).  The snapshot carries no sealed list: the guard reads them from SNAPSHOT.json's
        source_repo, the --repo checkout (contract C23.7-4 revision 6n, D95 (b)).
    python -B experiments/r2/smoke_r2.py print-launch
    python -B <SMOKE>/snapshot/experiments/r2/smoke_r2.py run          in the unit tag-r2-smoke (GPU), behind the gate of the chain (run_r2.gate_problems with this unit's name):
          (1) cli train --profile r2_small --fold A --seed 2 on the training set (LabelGuard, folds_full_v2.json's fold A: 98 train + 8 calib), the recipe r2s with targets.json's T, AMP on, SMOKE_STEPS
              steps, evaluated every SMOKE_EVAL steps on the fold's calib subjects, out <SMOKE>/member (never the chain's root);
          (2) the synthetic chain of run_r2 (tiny members, synthetic_tree's subjects) with --device cuda in <SMOKE>/synthetic: eval_r2's accumulators and the R2S arm on the GPU; in this process,
              with no time limit of its own (its steps' own limits and the unit's RuntimeMaxSec); an Exception or SystemExit it raises is recorded as a synthetic problem with the last lines of
              its traceback (smoke_r2.json is still written, 'stop'); KeyboardInterrupt goes through (nothing written);
          then <SMOKE>/smoke_r2.json (with the code record of require_snapshot: smoke_gate checks it), and STOP.  Each step once; a failure of the training stops the unit, kept, not rerun.
The rule (smoke_r2.json `reading`), in this order: 'stop' when
  - the member is not the planned recipe: its record's recipe is not r2_small, its target_labels are not targets.json's T, or AMP was not on (the diff review's new minor 3);
  - the loss is not finite at any logged step, or the member did not run its SMOKE_STEPS steps (stopped is not max_iters);
  - the projected wall per step of the real schedule -- (the data loading + 40000 x the median step after the first 10 + 80 x the median evaluation) / 40000 -- is above STEP_LIMIT (.30 s; the
    member's wall stop is 14400 s / 40000 = .36 s, UF measured at most .222 s);
  - the GPU allocated peak is above GPU_PEAK_LIMIT_GIB (18 GiB: the process cap is 20 GiB);
  - the synthetic chain on the GPU has a problem;
otherwise 'pass'.  'stop' hands to the lead before the training; 'pass' lets the lead go on to prepare and the chain.  The member's three weight files (about 200 MB) are listed with their sha256 for the
lead to delete after the reading ("only small files"); the tool deletes nothing.
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
sys.path[:0] = [str(REPO), str(HERE), str(HERE.parent/'uf'), str(HERE.parent/'ul'), str(HERE.parent/'unv2')]

import r2_common as rc  # noqa: E402
import run_r2  # noqa: E402

UNIT = 'tag-r2-smoke'
SMOKE_STEPS, SMOKE_EVAL = 400, 200
STEP_LIMIT = .30                                 # seconds per step, projected to the real schedule
GPU_PEAK_LIMIT_GIB = 18.
REAL_STEPS, REAL_EVAL_EVERY = rc.RECIPES['r2s']['max_iters'], rc.EVAL_EVERY
TIMEOUTS = dict(train=3600)                      # the synthetic chain runs in this process: no limit of its own (the diff review's new minor 2)
MEMORY_MAX, MEMORY_SWAP_MAX, THREADS = '8G', '2G', 4
RUNTIME_MAX = 3*3600
TRACEBACK_LINES = 20                             # the last lines of a synthetic failure's traceback kept in smoke_r2.json (the diff review 2's minor 4)


def smoke_config(targets):
    """The recipe r2s's --set values with the smoke's schedule: SMOKE_STEPS steps (poly over them), no early stop, the last step, every step logged, a wall stop of 20 minutes."""
    config = rc.train_config('r2s', targets)
    config.update(max_iters=SMOKE_STEPS, min_iters=SMOKE_STEPS, patience=SMOKE_STEPS, eval_every=SMOKE_EVAL, log_every=1, max_wall_seconds=1200.)
    return config


def steps(out, targets, *, python=rc.PY, device='cuda', inputs=(), config=None):
    """(name, argv) of the real part: one training command (cli, the snapshot's code).  config: smoke_config(targets) (the tests give a tiny one)."""
    cli = [python, '-B', '-m', 'bayesreg.seg.unet.cli']
    config = smoke_config(targets) if config is None else config
    return [('train', cli+['train', '--profile', rc.PROFILE, '--fold', 'A', '--seed', str(rc.SEED), *inputs, '--out', str(Path(out)/'member'), '--device', device, *run_r2.sets(config)])]


def measure(member_dir, *, real_steps=REAL_STEPS, eval_every=REAL_EVAL_EVERY):
    """What the smoke's member measured (its train_log.jsonl and train_record.json; the training subjects' calib Dice only): finiteness, the step and evaluation medians, the projection, the peaks."""
    member_dir = Path(member_dir)
    rows = [json.loads(line) for line in (member_dir/'train_log.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    record = json.loads((member_dir/'train_record.json').read_text(encoding='utf-8')) if (member_dir/'train_record.json').is_file() else {}
    train = [r for r in rows if r.get('event') == 'train']
    evals = [r for r in rows if r.get('event') == 'eval']
    start = next((r for r in rows if r.get('event') == 'start'), {})
    values = [r.get(k) for r in train for k in ('loss', 'target_soft_dice', 'soft_dice', 'ce')]
    finite = bool(train) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) and not any(r.get('event') == 'nonfinite_loss' for r in rows)
    late = [r['step_seconds'] for r in train if r.get('it', 0) >= 10]
    step = statistics.median(late) if late else None
    evaluation = statistics.median([r['eval_seconds'] for r in evals]) if evals else 0.
    data = start.get('data_ready_seconds') or 0.
    projected = None if step is None else (data+real_steps*step+(real_steps//eval_every)*evaluation)/real_steps
    memory = record.get('memory') or {}
    peaks = [r.get('gpu_peak_allocated_gib') for r in rows if r.get('gpu_peak_allocated_gib') is not None]+([memory['gpu_peak_allocated_gib']] if memory.get('gpu_peak_allocated_gib') is not None else [])
    return dict(finite=finite, n_logged_steps=len(train), stopped=record.get('stopped'), iterations_run=record.get('iterations_run'), recipe=record.get('recipe'), targets=(record.get('config') or {}).get('target_labels'),
                amp=(record.get('config') or {}).get('amp'), step_seconds_median=step, eval_seconds_median=evaluation, data_ready_seconds=data, projected_wall_per_step=projected,
                projected_member_hours=None if projected is None else projected*real_steps/3600, gpu_peak_allocated_gib=max(peaks) if peaks else None,
                gpu_peak_reserved_gib=memory.get('gpu_peak_reserved_gib'), host_peak_rss_gib=memory.get('host_peak_rss_gib'))


def reading_of(measured, synthetic_problems, *, targets, steps_planned=SMOKE_STEPS, step_limit=STEP_LIMIT, peak_limit=GPU_PEAK_LIMIT_GIB):
    """('pass' or 'stop', reasons): the rule of the module docstring, in its order (the limits included as passes).  targets: targets.json's T (the member's record must hold it)."""
    why = []
    if measured.get('recipe') != rc.PROFILE:
        why.append(f"the member's recipe is {measured.get('recipe')!r}, not {rc.PROFILE}")
    if measured.get('targets') != [int(k) for k in targets]:
        why.append(f"the member's target_labels {measured.get('targets')} are not targets.json's T {list(targets)}")
    if measured.get('amp') is not True:
        why.append(f"AMP was {measured.get('amp')!r}, not on")
    if not measured.get('finite'):
        why.append('the loss is not finite at every logged step')
    if measured.get('stopped') != 'max_iters' or measured.get('iterations_run') != steps_planned:
        why.append(f"the member stopped {measured.get('stopped')!r} after {measured.get('iterations_run')} of {steps_planned} steps")
    projected = measured.get('projected_wall_per_step')
    if projected is None or projected > step_limit:
        why.append(f'projected wall per step {projected} s > {step_limit} s (the wall stop of a member is .36 s)')
    peak = measured.get('gpu_peak_allocated_gib')
    if peak is None or peak > peak_limit:
        why.append(f'GPU allocated peak {peak} GiB > {peak_limit} GiB')
    why += [f'the synthetic chain on the GPU: {p}' for p in synthetic_problems]
    return ('stop' if why else 'pass'), why


def launch_argv(*, out=rc.SMOKE):
    snapshot = Path(out)/'snapshot'
    argv = ['systemd-run', '--user', f'--unit={UNIT}', '--collect', f'--working-directory={snapshot}', f'--property=MemoryMax={MEMORY_MAX}', f'--property=MemorySwapMax={MEMORY_SWAP_MAX}', f'--property=RuntimeMaxSec={RUNTIME_MAX}',
            '--setenv=PYTHONDONTWRITEBYTECODE=1', *[f'--setenv={name}={THREADS}' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')],
            rc.PY, '-B', str(snapshot/'experiments/r2/smoke_r2.py'), 'run', '--out', str(out)]
    if 'pytest' in ' '.join(argv) or 'seq-vp-' in ' '.join(argv):
        raise ValueError('a start string must not name the test runner or the seq-vp- prefix')
    return argv


def targets_of(path=rc.PP_ROOT/'targets.json'):
    import eval_r2
    if not Path(path).is_file():
        raise run_r2.Stop(f'{path}: no such file; the pp diagnosis and its split come first')
    try:
        return eval_r2.read_targets(path), rc.sha256_file(path)
    except ValueError as exc:
        raise run_r2.Stop(str(exc)) from None


def prepare(out, repo):
    import subprocess
    from bayesreg.seg.unet.snapshot import make_snapshot, verify_snapshot
    out = Path(out)
    ran = [str(p) for p in (out/'smoke_r2.json', out/'member') if p.exists()]
    if ran:
        raise run_r2.Stop(f'{ran} there: the smoke ran; its root is kept whole until the training chain is over, never prepared again (the diff review 2\'s minor 1(a))')
    targets_of()
    out.mkdir(parents=True, exist_ok=True)
    run_r2.set_run_log(out)
    snapshot = out/'snapshot'
    record = verify_snapshot(snapshot) if snapshot.exists() else make_snapshot(repo, snapshot, paths=run_r2.SNAPSHOT_PATHS_R2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run([rc.PY, '-B', '-m', 'bayesreg.seg.unet.cli', 'check-inputs', '--profile', rc.PROFILE, '--require-snapshot'], cwd=str(snapshot), env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise run_r2.Stop(f'the pinned-input check failed: {done.stdout[-2000:]} {done.stderr[-2000:]}')
    run_r2.log(f"smoke snapshot {snapshot} at {record['commit']}; pinned inputs checked (no label or image read)")
    return record


def run(out, *, gate=True, wait_minutes=run_r2.WAIT_MINUTES, python=rc.PY, device='cuda', inputs=(), targets_path=rc.PP_ROOT/'targets.json', synthetic_device='cuda', env=None, config=None, code=None):
    """The unit's work: the gate, the training, the synthetic chain, smoke_r2.json.  Returns the document.  config: None for the smoke's (the tests give a tiny one, its target_labels set here).
    code: the identity of the snapshot the unit runs from (main: run_r2.snapshot_identity of the root require_snapshot verified), written into smoke_r2.json for smoke_gate."""
    out = Path(out)
    if (out/'smoke_r2.json').exists() or (out/'member').exists():
        raise run_r2.Stop(f'{out}: the smoke ran before; kept, not rerun')
    run_r2.set_run_log(out)
    targets, digest = targets_of(targets_path)
    config = smoke_config(targets) if config is None else dict(config, target_labels=list(targets))
    state, chain_log = out/'state', out/'smoke.log'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1') if env is None else env
    if gate:
        run_r2.ufrun.wait_for_gate(wait_minutes, problems=lambda: run_r2.gate_problems(UNIT))
    for name, argv in steps(out, targets, python=python, device=device, inputs=inputs, config=config):
        def action(argv=argv, name=name):
            code, timed_out = run_r2.ufrun.run_command(argv, cwd=run_r2.CODE_ROOT, log_path=chain_log, timeout=TIMEOUTS[name], env=env)
            return 124 if timed_out else code
        run_r2.ufrun.run_step(name, action, state=state, log_path=chain_log, label=shlex.join(argv))
    measured = measure(out/'member'/rc.member_name('A'))
    try:
        synthetic = list(run_r2.synthetic_chain(out/'synthetic', env, synthetic_device))
    except (Exception, SystemExit) as exc:                # Stop, CodeNotClean or anything else: a synthetic problem, smoke_r2.json still written (the diff review's new minor 2)
        tail = traceback.format_exc().rstrip().splitlines()[-TRACEBACK_LINES:]      # the diff review 2's minor 4
        synthetic = [f'the synthetic chain raised {type(exc).__name__}: {exc}', 'traceback (last lines): '+' | '.join(line.strip() for line in tail)]
        run_r2.log('the synthetic chain raised:\n'+'\n'.join(tail))
    finally:
        run_r2.set_run_log(out)                            # the synthetic chain points the log at its own world
    reading, why = reading_of(measured, synthetic, targets=targets, steps_planned=config['max_iters'])
    member = out/'member'/rc.member_name('A')
    weights = {p.name: dict(bytes=p.stat().st_size, sha256=rc.sha256_file(p)) for p in sorted(member.glob('*.pt'))}
    document = dict(schema='bayesreg.r2.smoke.v1', reading=reading, why=why, measured=measured, synthetic_problems=synthetic, targets=targets, targets_sha256=digest, config=config, code=code,
                    rule=dict(step_limit=STEP_LIMIT, gpu_peak_limit_gib=GPU_PEAK_LIMIT_GIB, steps=SMOKE_STEPS, projection=f'(data + {REAL_STEPS} x median step + {REAL_STEPS//REAL_EVAL_EVERY} x median evaluation) / {REAL_STEPS}'),
                    weights_to_delete_after_the_reading=weights, plan=rc.plan_record(run_r2.CODE_ROOT), reads_development_truth=False, takes_the_read_lock=False,
                    note='GPU 冒烟：只读训练集（LabelGuard），不读开发集真值，不取读取锁，不在链条的根里；读数 stop 就在开训前停下交总负责人。', utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    rc.write_json_once(out/'smoke_r2.json', document)
    run_r2.log(f'smoke: {reading} {why}')
    return document


def dry_run(log=print):
    problems = []
    log(f'== R2 GPU smoke, dry run; nothing written; the root {rc.SMOKE} ({"EXISTS" if rc.SMOKE.exists() else "not there"}), the chain root {rc.OUT} untouched')
    log('prepare: '+shlex.join([rc.PY, '-B', 'experiments/r2/smoke_r2.py', 'prepare', '--repo', '<a clean checkout with the sealed lists and a data link>']))
    log('launch:  '+shlex.join(launch_argv()))
    path = rc.PP_ROOT/'targets.json'
    targets = run_r2.SYNTHETIC_TARGETS if not path.is_file() else json.loads(path.read_text(encoding='utf-8'))['targets']
    log(f"targets.json {'there' if path.is_file() else 'not there yet (shown with a placeholder T)'}")
    for name, argv in steps(rc.SMOKE, targets):
        log(f'{name} [gate, limit {TIMEOUTS[name]} s]: {shlex.join(argv)}')
    log(f'synthetic: run_r2.synthetic_chain(<SMOKE>/synthetic, device cuda) in this process (no limit of its own: its steps\' limits and the unit\'s RuntimeMaxSec {RUNTIME_MAX} s); '
        'an Exception or SystemExit it raises is a synthetic problem (with the last traceback lines), smoke_r2.json still written; KeyboardInterrupt goes through')
    problems += run_r2.snapshot_path_problems(log=log)
    log(f'the rule: recipe {rc.PROFILE}, targets.json\'s T, AMP on, finite loss, {SMOKE_STEPS} steps, projected wall per step <= {STEP_LIMIT} s, GPU allocated peak <= {GPU_PEAK_LIMIT_GIB} GiB, '
        'the synthetic chain on the GPU without a problem')
    log('〔估算〕the unit: the gate, data loading about 30 s, 400 steps at about .21-.25 s, 2 evaluations, the synthetic chain about 1 minute: about 5-10 minutes of GPU')
    if rc.SMOKE.exists():
        problems.append(f'{rc.SMOKE} exists')
    log(f"== dry run problems: {'; '.join(problems) if problems else 'none'}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true')
    sub = parser.add_subparsers(dest='command')
    p = sub.add_parser('prepare')
    p.add_argument('--repo', type=Path, default=rc.MAIN_REPO)
    p.add_argument('--out', type=Path, default=rc.SMOKE)
    sub.add_parser('print-launch').add_argument('--out', type=Path, default=rc.SMOKE)
    p = sub.add_parser('run')
    p.add_argument('--out', type=Path, default=rc.SMOKE)
    p.add_argument('--wait-minutes', type=int, default=run_r2.WAIT_MINUTES)
    args = parser.parse_args(argv)
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
            document = run(args.out, wait_minutes=args.wait_minutes, code=run_r2.snapshot_identity(run_r2.CODE_ROOT))
            return 0 if document['reading'] == 'pass' else 8
        else:
            parser.print_help()
            return 2
    except (run_r2.Stop, run_r2.CodeNotClean) as exc:
        run_r2.log(f'STOP: {exc}')
        return 8
    return 0


if __name__ == '__main__':
    sys.exit(main())
