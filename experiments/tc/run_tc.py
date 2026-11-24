"""TC (runs/20261104-tc/plan.md; docs/bayesreg/notes/提速与换预分割器规划_20261104.md §5): the Dice of S_P, S_K1 and B1TM at each native time.  The three arms run again on the first 5 development pairs, with their inputs and command lines; only their intermediate commits wait for the curve.

    python -B experiments/tc/run_tc.py --dry-run [--scratch DIR]
        prints all 15 runs, their command lines against the originals' cmd.txt, the real frozen parser and prepare on each (a scratch copy of the frozen source), the protocol against the original's, the entries, which commit each time takes on the original records, one evaluation of a final map against its metrics.json (with the seconds it takes),
        a rehearsal of the reading, the time ledger of S_P's 19 runs, the files of the measurement, the disk estimate.  No solver, no GPU, no unit, nothing written outside the scratch directory.
    python -B experiments/tc/run_tc.py queue [--wait-minutes 30] [--min-host-gib 15]       (below 15: refused)
        the lead's start, in the unit printed by the dry run.  Start checks first (frozen worktree, inputs records, every entry, every original): a missing file stops before any run.  Then S_P, S_K1, B1TM on the 5 pairs, serial, one unit tag-tc-<arm>-<ffff>-<mmmm> per run (MemoryMax 12G, as the originals), each behind the gate: no $ROOT/hold,
        no other tag-* or seq-vp-* unit, 1-minute load < 1.5, no GPU compute process, GPU utilization <= 20%, GPU free >= 15 GiB, /home free >= 100 GiB, /mnt/c free >= --min-host-gib; closed for --wait-minutes: STOP.  60 s between runs.  A run directory is never rerun; failures are kept.
        After each run: classify, evaluate_run, the entry's trace (S_P, S_K1), curve.json and reproduction.json, then this run's own committed_*.npz and level_*.npz go (cleanup.json).  STOP keeps them: a prepare or check failure, a class `unsure`, an evaluation or curve failure, an entry not injected as recorded, 3 infrastructure failures in a row.
        At the end the reading (decide_tc.py) and the time ledger, then the measurement, one GPU unit tag-tc-measure (measure_tc.py); nothing else starts.  After a STOP what has finished is summed up.
    python -B experiments/tc/run_tc.py decide         analysis/decision_tc.json, summary_tc.md, long_curve.csv
    python -B experiments/tc/run_tc.py ledger         analysis/time_ledger_sp19.json, .md (S_P's 19 runs, only read)
    python -B experiments/tc/run_tc.py evaluate-one RUN --arm ARM --pair PAIR --run-id ID     the queue's step: experiments/d0/evaluate.py evaluate_run, plus the entry's record
    python -B experiments/tc/run_tc.py curve-one RUN --arm ARM --pair PAIR --run-id ID        the queue's step: curve.json, reproduction.json; deletes nothing

The only root is tc_v1: every writing command refuses anything else; seed_v1, seed_v1b, b1_v1, mc0p_v1, mc0p_v1b and d0_v1_a5 are only read.  Exit codes: 0 done, 2 usage, another driver on the root or another root, 8 STOP or a failed check.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT/'experiments/seed'))

import numpy as np  # noqa: E402

import decide_tc  # noqa: E402
import measure_tc  # noqa: E402
import run_seed as core  # noqa: E402
import run_seed19 as s19  # noqa: E402
import tc_curve as curve  # noqa: E402

ROOT_NAME = curve.ROOT_NAME
QUEUE_UNIT, MEASURE_UNIT = 'tag-tc-queue', 'tag-tc-measure'
PAIRS = core.PAIRS
DEFAULT_MIN_HOST_GIB = 15.                               # the brief §4; F's own guard stays 10 GiB + pending (F/scripts/run_stage_3d_e19_mri.py:460, 546)
GPU_FREE_MIB = 15*1024                                   # the brief §4 (SEED's 14 GiB stays in run_seed.py)
MAX_FIELDS = len(curve.TIMES)+1                          # fields a run's curve may evaluate: one per time, one final map
# stage: as the original, so that the prepared protocol is the original's, key by key.  original: the root of the run this one repeats.  injected: the entry field replaces the pre-optimizer (inject_run.py) and the 47 s hold is in the clock.
ARMS = dict(S_P=dict(stage='SEED-S_P', original='seed_v1', injected=True, method=core.method_name('S_P'), role='判定臂：K = 35 入口（mc0p_v1、mc0p_v1b 的 Q2 长档）'),
            S_K1=dict(stage='SEED-S_K1', original='seed_v1b', injected=True, method=s19.method_name('S_K1'), role='只描述：K = 1、600 步入口（mc0p_v1 的 Q0 长档）'),
            B1TM=dict(stage='B1-B1TM', original='b1_v1', injected=False, method='frozen_eb808be_b1_330s_gw1_1_0.1_0.01', role='只描述：T88 入口（60 步），没有持时'))
Stop = core.Stop
EntryError = core.EntryError


class GateClosed(Stop):
    """The gate stayed closed for its time."""


def short_name(pair):
    return core.short_name(pair)


def require_cell(arm, pair):
    """Only the arms of the plan on the first 5 development pairs: checked before any file of the pair is opened."""
    if arm not in ARMS:
        raise SystemExit(f'TC has no arm {arm}; only {tuple(ARMS)}')
    if pair not in PAIRS:
        raise SystemExit(f'TC runs only the first 5 development pairs {[short_name(p) for p in PAIRS]}; refused: {pair}')
    return arm, pair


def plan_runs():
    """S_P on the 5 pairs, then S_K1, then B1TM: the arm of the reading first, so that a stop costs description."""
    return [(arm, pair) for arm in ARMS for pair in PAIRS]


def default_root(results=None):
    return Path(core.RESULTS if results is None else results)/ROOT_NAME


def root_problem(root, results=None):
    """Why this batch may not write at this root, or None: only its own."""
    return None if Path(root).resolve() == default_root(results).resolve() else f'{root}: this batch writes only to {default_root(results)}; nothing read, nothing moved'


def run_dir(root, arm, pair):
    require_cell(arm, pair)
    return Path(root)/'runs'/arm/pair


def own_run_problem(run, arm, pair, results=None):
    """Why this directory is not this batch's run of this arm and pair, or None: evaluate-one and curve-one write only there."""
    want = run_dir(default_root(results), arm, pair)
    return None if Path(run).resolve() == want.resolve() else f'{run}: this step writes only to {want}; nothing read, nothing moved'


def original_dir(arm, pair, results=None):
    """The run this one repeats: only read."""
    require_cell(arm, pair)
    return Path(core.RESULTS if results is None else results)/ARMS[arm]['original']/'runs'/arm/pair


def unit_name(arm, pair):
    require_cell(arm, pair)
    return f'tag-tc-{arm}-{short_name(pair)}'


def new_run_id(arm, pair):
    return f"tc_{arm}_{short_name(pair)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def config_entry(results=None):
    """The configuration of all three arms, in b1_spec's form (the same as B1TM's and S_P's: the dry run compares the protocols)."""
    return core.b1_entry('S_P', results)


def prepare_argv(arm, pair, run, results=None):
    require_cell(arm, pair)
    return core.check_launch([core.PY, '-B', core.frozen_dir(results)/'scripts/run_stage_3d_e48_pg.py', 'prepare', run, *core.OPTS, '--stage', ARMS[arm]['stage'], '--pair', pair])


def unit_argv(arm, pair, run, root):
    """The original's unit (run_seed.unit_argv, the same resources and watchdog), under this batch's name; B1TM runs the snapshot's frozen entry itself, as run_queue.sh did."""
    require_cell(arm, pair)
    argv = core.unit_argv(arm, pair, run, root, unit=unit_name(arm, pair))
    if ARMS[arm]['injected']:
        return argv
    wrapper = f"'{run}/seed_inject.py' '{run}'"
    if argv[-1].count(wrapper) != 1:
        raise ValueError('run_seed.unit_argv no longer holds the wrapper call')
    return core.check_launch(argv[:-1]+[argv[-1].replace(wrapper, f"'{run}/snapshot/scripts/run_stage_3d_e48_pg.py' run '{run}'")])


def queue_launch(root=None, repo=core.MAIN_REPO, min_host_gib=DEFAULT_MIN_HOST_GIB):
    """The lead's start string of the queue."""
    root = default_root() if root is None else root
    return core.check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT}', '--collect', f'--working-directory={repo}', '--property=MemoryMax=9G', '--setenv=PYTHONDONTWRITEBYTECODE=1',
                              '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=NUMEXPR_NUM_THREADS=1',
                              core.PY, '-B', 'experiments/tc/run_tc.py', 'queue', '--root', str(root), '--min-host-gib', f'{min_host_gib:g}'])


def measure_argv(root, repo=core.MAIN_REPO, results=None, code_sha='n/a'):
    """The unit of the measurement: the environment of an MC0P run (one BLAS thread, CPUQuota 800, 7 setup threads inside), PYTHONPATH with the frozen tree first, MemoryMax 12G (Q2 long: host peak 5.6 GiB in mc0p_v1), 20 minutes."""
    out = Path(root)/measure_tc.OUT_NAME
    env = dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, NUMBA_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=f'{core.frozen_dir(results)}:{repo}')
    return core.check_launch(['systemd-run', '--user', f'--unit={MEASURE_UNIT}', '--wait', '--collect', f'--working-directory={repo}', '--property=MemoryMax=12G', '--property=MemorySwapMax=4G', '--property=CPUQuota=800%', '--property=RuntimeMaxSec=1200',
                              f'--property=StandardOutput=append:{out}/log.txt', f'--property=StandardError=append:{out}/log.txt', *[f'--setenv={k}={v}' for k, v in env.items()], core.PY, '-B', f'{repo}/experiments/tc/measure_tc.py', '--out', str(out), '--code-sha', code_sha])


# ---- the inputs, the entries, the originals --------------------------------------------------------------------------------------------------------------------------------------------------
def cell_files(arm, pair, results=None):
    """(input files of run_seed.input_files, the entry record or None).  Every arm has A5P's inputs; S_P's and S_K1's entries through their batches' own checks.  EntryError otherwise."""
    require_cell(arm, pair)
    if arm == 'S_K1':
        files = s19.input_files('S_K1', pair, results)
        return files, s19.check_entry('S_K1', pair, files[3], results)
    files = core.input_files('S_P', pair, results)
    return files, core.check_entry('S_P', pair, files[3], results) if arm == 'S_P' else None


def original_problems(arm, pair, files, entry, results=None):
    """The original against this run's inputs and entry: its records are all there, the same input files, the same entry field and hold, the same wrapper.  Read only."""
    d = original_dir(arm, pair, results)
    missing = [name for name in ('protocol.json', 'metrics.json', 'cmd.txt') if not (d/name).is_file()]
    if missing:
        return [f'{d}: no {missing}']
    protocol = core.read_json(d/'protocol.json')
    task = d/protocol['task']['name']
    problems = [f'{task}: no {name}' for name in ('commits.json', 'history.json', 'levels.json', 'final_native.json', 'final_native.npz') if not (task/name).is_file()]
    if protocol.get('input_sha256') != {'pair.npz': files[1], 'manifest.json': files[2]}:
        problems.append(f'{d}: its inputs differ from {files[0]}')
    if protocol.get('stage') != ARMS[arm]['stage']:
        problems.append(f"{d}: stage {protocol.get('stage')}, here {ARMS[arm]['stage']}")
    if entry is not None:
        recorded = core.read_json(d/'seed_entry.json') or {}
        for key in ('field_path', 'field_sha256', 'u_sha256', 'tapered_sha256', 'charge_seconds', 'source_run_id'):
            if recorded.get(key) != entry.get(key):
                problems.append(f'{d}: seed_entry.json {key} {recorded.get(key)}, here {entry.get(key)}')
        if core.sha_file(d/'seed_inject.py') != core.sha_file(core.HERE/'inject_run.py'):
            problems.append(f'{d}: its wrapper differs from experiments/seed/inject_run.py')
    return problems


# ---- the machine ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class System(core.System):
    """SEED's machine; the arms' inputs and the steps after a run are this batch's."""

    def inputs_problems(self, arm, repo, results=None):
        return core.b1_spec.check_one_input(config_entry(results), repo)

    def step(self, name, run, arm, pair, run_id):
        with open(Path(run)/'log.txt', 'ab') as f:
            return subprocess.run(core.check_launch([core.PY, '-B', HERE/'run_tc.py', name, run, '--arm', arm, '--pair', pair, '--run-id', run_id]), stdout=f, stderr=subprocess.STDOUT).returncode

    def evaluate(self, run, arm, pair, run_id):
        return self.step('evaluate-one', run, arm, pair, run_id)

    def curve(self, run, arm, pair, run_id):
        return self.step('curve-one', run, arm, pair, run_id)


def busy_reason(system, root, min_host_gib):
    """Why no run may start now, or None: SEED's gate with this queue's unit and the floor of the brief, and GPU free >= 15 GiB."""
    reason = core.busy_reason(system, root, queue_unit=QUEUE_UNIT, host_floor_gib=min_host_gib)
    if reason is None and system.gpu_free_mib() < GPU_FREE_MIB:
        return f'GPU free < {GPU_FREE_MIB} MiB'
    return reason


def wait_for_gate(system, root, minutes, log, min_host_gib):
    started = system.now()
    while True:
        reason = busy_reason(system, root, min_host_gib)
        if reason is None:
            return
        if system.now()-started >= minutes*60:
            raise GateClosed(f'gate closed for {minutes} minutes: {reason}')
        log(f'waiting: {reason}')
        system.sleep(core.POLL_SECONDS)


# ---- one run ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def write_records(run, arm, pair, unit, run_id, entry, files, env_text, repo, results=None):
    """The records of protocol §8 (git.txt, env.txt, data.txt, seed), tc_arm.json; for an injected arm the entry's record and the wrapper's copy, as the original."""
    run = Path(run)
    d, pair_sha, manifest_sha, manifest = files
    git = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', '-C', str(repo), 'diff', '--stat'], capture_output=True, text=True).stdout
    files_sha = '\n'.join(f'{p.relative_to(ROOT)} {core.sha_file(p)}' for pattern in ('experiments/tc/*.py', 'experiments/seed/*.py', 'experiments/d0/*.py', 'experiments/d0/*.json') for p in sorted(ROOT.glob(pattern)))
    frozen = core.frozen_dir(results)
    frozen_head = subprocess.run(['git', '-C', str(frozen), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    frozen_status = subprocess.run(['git', '-C', str(frozen), 'status', '--porcelain'], capture_output=True, text=True).stdout
    untracked = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--', 'experiments/tc', 'experiments/seed', 'experiments/d0', 'docs/bayesreg', 'configs/bayesreg', 'runs'], capture_output=True, text=True).stdout
    (run/'git.txt').write_text(f'repo: {repo}\nrepo HEAD: {git}\nrepo git diff --stat:\n{diff}\nrepo untracked or changed (experiments/tc, experiments/seed, experiments/d0, docs/bayesreg, configs/bayesreg, runs):\n{untracked}\n'
                               f'frozen worktree: {frozen}\nfrozen HEAD: {frozen_head}\nfrozen commit expected: {core.FROZEN_COMMIT}\nfrozen git status --porcelain (must be empty):\n{frozen_status}\nfile sha256:\n{files_sha}\n')
    (run/'env.txt').write_text(env_text)
    made = manifest['preprocessing_d0']
    lines = [f'run_id: {run_id}', f'arm: {arm}', f'pair: {pair}', f'unit: {unit}', f'repeats: {original_dir(arm, pair, results)}', f"fixed subject: {manifest['fixed_subject']}", f"moving subject: {manifest['moving_subject']}",
             f'arm stage directory (prepare source): {d.parent}', f'arm input pair.npz: {d}/pair.npz sha256 {pair_sha}', f'arm input manifest.json: {d}/manifest.json sha256 {manifest_sha}',
             f"original pair.npz sha256: {made['source_pair_sha256']}", f"original manifest sha256: {made['source_manifest_sha256']}", f"colouring: {made['recipe']}", 'oracle: False']
    if entry is not None:
        lines += [f"entry field: {entry['field_path']} sha256 {entry['field_sha256']} bytes sha256 {entry['u_sha256']}", f"entry source run: {entry['source_run_id']}; K {entry['source_k']}; tier {entry['source_tier']}; oracle {entry['oracle']}"]
    (run/'data.txt').write_text('\n'.join(lines+['truth labels for evaluation: the original stage pair.npz (evaluate.py checks its sha256)'])+'\n')
    (run/'seed').write_text(f'{arm} (runs/20261104-tc/plan.md): the inputs of A5P, no new colouring; '+('the entry field has no random step (run.json: none); ' if entry is not None else 'T88 has no random step; ')+'solver: configuration C has no random number generator or seed option; GPU reductions may still differ between runs\n')
    core.write_json(run/'tc_arm.json', dict(arm=arm, seconds=core.SECONDS, changes=core.CHANGES, inputs_root=str(core.inputs_root('S_P', results)), original=str(original_dir(arm, pair, results)), stage=ARMS[arm]['stage'], charge_seconds=None if entry is None else core.ENTRY_CHARGE_SECONDS,
                                            watchdog_seconds=core.WATCHDOG, method=ARMS[arm]['method'], input_oracle=False, role=ARMS[arm]['role'], times=list(curve.TIMES), rule=curve.RULE, clock=curve.CLOCK))
    if entry is not None:
        core.write_json(run/'seed_entry.json', entry)
        core.write_json(run/'seed_arm.json', dict(arm=arm, seconds=core.SECONDS, changes=core.CHANGES, inputs_root=str(core.inputs_root('S_P', results)), entry_root=entry.get('source_root'), source_arm=entry.get('source_arm'), charge_seconds=core.ENTRY_CHARGE_SECONDS,
                                                  watchdog_seconds=core.WATCHDOG, method=ARMS[arm]['method'], input_oracle=False, wrapper_sha256=core.sha_file(core.HERE/'inject_run.py')))
        shutil.copyfile(core.HERE/'inject_run.py', run/'seed_inject.py')


def finish_run(root, run, arm, pair, run_id, t0, system, log):
    """Classify, evaluate, check the entry, the curve and the comparison, then this run's own commits go.  Returns the state: ok, handover_only, method_failure, infrastructure.  Stop on anything else, with the commits still there."""
    run = Path(run)
    record = system.classify(run, t0)
    cls = (record or {}).get('class')
    with open(run/'log.txt', 'a') as f:
        f.write(f'{core.utc()} class {cls}\n')
    if cls == 'infrastructure':
        core.write_json(run/'queue_done.json', dict(run_id=run_id, state=cls, kind=record.get('kind'), reason=record.get('reason'), done_utc=core.utc()))
        log(f"run {arm} {pair}: infrastructure failure ({record.get('kind')}: {record.get('reason')}); kept whole, not rerun")
        return cls
    if cls not in ('ok', 'handover_only', 'method_failure'):
        raise Stop(f"{run} classified {cls} ({(record or {}).get('reason')}); kept whole, nothing rerun")
    if system.evaluate(run, arm, pair, run_id) != 0:
        raise Stop(f'evaluation failed for {run} (see log.txt); the commits stay')
    metrics = core.read_json(run/'metrics.json') or {}
    if ARMS[arm]['injected'] and (metrics.get('seed') or {}).get('passed') is not True:
        raise Stop(f"{run}: the entry was not injected as recorded: {(metrics.get('seed') or {}).get('problems')}; class {cls}; kept whole")
    if cls != 'method_failure' and not (run/'cleanup.json').is_file():      # cleanup.json exists: an earlier start got this far and only queue_done.json is missing; the commits are gone, the curve is on disk
        if system.curve(run, arm, pair, run_id) != 0:
            raise Stop(f'the curve failed for {run} (see log.txt); the commits stay')
        try:
            curve.cleanup_after_curve(run, Path(root)/'runs')
        except (curve.CurveError, OSError) as error:     # an OSError of the deletion too: queue_stop.json, not a driver that just dies (review m4)
            raise Stop(f'the cleanup of {run} stopped: {error}') from None
        unrecorded = (core.read_json(run/'cleanup.json') or {}).get('not_deleted_unrecorded') or []
        if unrecorded:                                   # the summary's anomalies say it too
            log(f"{arm} {pair}: {len(unrecorded)} intermediate files without a recorded sha256 stay: {[row['name'] for row in unrecorded]}")
    points =(core.read_json(run/'curve.json') or {}).get('n_fields_evaluated')
    x, c = metrics.get('metrics') or {}, metrics.get('certificate') or {}
    same = ((core.read_json(run/'reproduction.json') or {}).get('sequence') or {})
    summary = f"{cls} status {metrics.get('status')} seg35 {x.get('seg35')} seg4 {x.get('seg4')} sdlogj {x.get('sdlogj_official')} cert {c.get('status')}/{c.get('passed')} native {metrics.get('native_commits')} fields evaluated {points}; commits against the original: {same.get('common_prefix')} of {same.get('n_new')}/{same.get('n_old')}, identical {same.get('identical')}"
    core.write_json(run/'queue_done.json', dict(run_id=run_id, state=cls, summary=summary, done_utc=core.utc()))
    log(f'done {arm} {pair}: {summary}')
    return cls


def one_run(o, system, root, arm, pair, entry, files, log, env_text, results, repo, first):
    """Returns the state, or None if nothing ran because it had already ended."""
    run = run_dir(root, arm, pair)
    if (run/'queue_done.json').is_file():
        return None
    unit = unit_name(arm, pair)
    if run.exists():                                     # never rerun: an ended unit is post-processed, anything else stops
        if (run/'exit_code').is_file():
            log(f'resume: {run} has an exit_code; post-processing')
            return finish_run(root, run, arm, pair, (run/'run_id').read_text().strip(), (run/'uptime_before').read_text().strip(), system, log)
        raise Stop(f'{run} exists without exit_code or queue_done.json (an incomplete attempt, kept whole)')
    if not first:
        system.sleep(core.BETWEEN_RUNS_SECONDS)
    wait_for_gate(system, root, o.wait_minutes, log, o.min_host_gib)
    run.parent.mkdir(parents=True, exist_ok=True)
    argv = prepare_argv(arm, pair, run, results)
    target = files[0].parent
    prepare_log = Path(root)/'prepare.log'
    code = system.prepare(argv, target, prepare_log, results)
    if code != 0 or not run.is_dir():
        raise Stop(f'prepare exit {code} for {run} (see {prepare_log})')
    check = system.check_prepared(run, pair, ARMS[arm]['stage'], files[1], files[2], config_entry(results))
    core.write_json(run/'prepared_check.json', check)
    if not check.get('ok'):
        raise Stop(f"prepared run differs from the configuration or the arm's inputs: {check.get('problems')}")
    run_id = new_run_id(arm, pair)
    (run/'run_id').write_text(run_id+'\n')
    write_records(run, arm, pair, unit, run_id, entry, files, env_text, repo, results)
    sd = unit_argv(arm, pair, run, root)
    (run/'cmd.txt').write_text(f'# prepare (run from any directory; the frozen script resolves its root from its own path)\n{shlex.join(argv)}\n# worktree link during prepare: {core.frozen_dir(results)}/data/three_d/stage_L2R_val_v1 -> {target}\n# run\n{shlex.join(sd)}\n')
    (run/'service_unit.txt').write_text(unit+'\n')
    t0 = system.uptime()
    (run/'uptime_before').write_text(f'{t0}\n')
    log(f"run {arm} {pair} unit {unit}; load {system.load1()}; /home free {system.free_gib('/home'):.0f} GiB; /mnt/c free {system.free_gib('/mnt/c'):.0f} GiB")
    with open(run/'log.txt', 'a') as f:
        f.write(f'{core.utc()} launch {unit}; {system.state()}\n')
    code = system.launch(sd, run/'log.txt')
    with open(run/'log.txt', 'a') as f:
        f.write(f'{core.utc()} unit ended, systemd-run exit {code}; {system.state()}\n')
    return finish_run(root, run, arm, pair, run_id, t0, system, log)


def start_checks(system, results, repo, log):
    """Everything that can be known before the first run: STOP here rather than in the middle.  Returns {(arm, pair): (entry record or None, input files)}."""
    problems = list(system.frozen_problems(results))
    problems += [f'inputs: {p}' for p in system.inputs_problems('S_P', repo, results)]
    prepared = {}
    for arm, pair in plan_runs():
        try:
            files, entry = cell_files(arm, pair, results)
        except EntryError as error:
            problems.append(str(error))
            continue
        prepared[(arm, pair)] = (entry, files)
        problems += original_problems(arm, pair, files, entry, results)
    if problems:
        raise Stop('start checks: '+' | '.join(problems))
    log(f'start checks passed: frozen worktree, inputs record, all {len(prepared)} cells and their originals')
    return prepared


def trailing_infrastructure(root):
    n = 0
    for arm, pair in plan_runs():
        done = core.read_json(run_dir(root, arm, pair)/'queue_done.json')
        if done is None:
            continue
        n = n+1 if done.get('state') == 'infrastructure' else 0
    return n


def ledger_sources(results=None):
    """S_P's 19 runs: seed_v1 the first 5 pairs, seed_v1b the other 14; only read."""
    results = Path(core.RESULTS if results is None else results)
    return [('seed_v1', results/'seed_v1/runs/S_P'/pair) for pair in PAIRS]+[('seed_v1b', results/'seed_v1b/runs/S_P'/pair) for pair in s19.REST_PAIRS]


def load_evaluate():
    ev = core.load_evaluate()
    ev.STAGE = core.MAIN_REPO/'data/three_d/stage_L2R_val_v1'    # a worktree has no data link
    return ev


def write_analysis(root, *, ev=None):
    return decide_tc.write_analysis(root, plan_runs(), list(PAIRS), short_name, ev=load_evaluate() if ev is None else ev, utc=core.utc)


def run_measure(system, root, o, log, results, repo):
    """The GPU unit, after the reading.  A failure stops here and leaves the reading as it is."""
    out = Path(root)/measure_tc.OUT_NAME
    if (out/measure_tc.RESULT_FILE).exists():
        log(f'{out/measure_tc.RESULT_FILE} exists: the measurement is not run twice')
        return
    system.sleep(core.BETWEEN_RUNS_SECONDS)
    wait_for_gate(system, root, o.wait_minutes, log, o.min_host_gib)
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or 'n/a'
    argv = measure_argv(root, repo, results, head)
    (out/'cmd.txt').write_text(shlex.join(argv)+'\n')
    log(f'launch {MEASURE_UNIT}')
    code = system.launch(argv, out/'launch.log')
    log(f'{MEASURE_UNIT} ended, systemd-run exit {code}')
    if code != 0:
        raise Stop(f'{MEASURE_UNIT}: exit {code} (see {out}/log.txt); the reading stands; not rerun')


def batch_record(root, o, repo):
    """batch.json: what defines the batch is checked at every start; a difference is a STOP before any run."""
    fixed = dict(schema='bayesreg.tc.batch.v1', seconds=core.SECONDS, changes=core.CHANGES, charge_seconds=core.ENTRY_CHARGE_SECONDS, frozen_commit=core.FROZEN_COMMIT, times=list(curve.TIMES), plan=[f'{a} {short_name(p)}' for a, p in plan_runs()], originals={a: spec['original'] for a, spec in ARMS.items()})
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    this = dict(utc=core.utc(), min_host_gib=o.min_host_gib, repo_head=head)
    batch = core.read_json(root/'batch.json')
    if batch is not None:
        different = {k: (batch.get(k), v) for k, v in fixed.items() if batch.get(k) != v}
        if different:
            raise Stop(f'batch.json differs from this batch: {different}')
    core.write_json(root/'batch.json', dict(fixed, started_utc=this['utc'] if batch is None else batch.get('started_utc'), starts=([] if batch is None else batch.get('starts', []))+[this]))


def queue_body(o, system, root, log, results, repo, *, ev=None):
    prepared = start_checks(system, results, repo, log)
    env_text = core.environment_text(repo) if isinstance(system, System) else 'test\n'
    (root/'cache/numba_jit').mkdir(parents=True, exist_ok=True)
    (root/'cache/first_use').mkdir(parents=True, exist_ok=True)
    batch_record(root, o, repo)
    first = True
    try:
        for arm, pair in plan_runs():
            entry, files = prepared[(arm, pair)]
            state = one_run(o, system, root, arm, pair, entry, files, log, env_text, results, repo, first)
            if state is None:
                continue
            first = False
            if trailing_infrastructure(root) >= core.MAX_CONSECUTIVE_INFRA:
                raise Stop(f'{core.MAX_CONSECUTIVE_INFRA} infrastructure failures in a row')
    except Stop:                                         # what finished is summed up, then STOP
        if any((run_dir(root, a, p)/'metrics.json').is_file() for a, p in plan_runs()):
            try:
                result = write_analysis(root, ev=ev)
                log(f"the queue stops; reading of what finished: {result['reading'][:400]}")
            except (Exception, SystemExit) as error:
                log(f'the queue stops; the summary failed: {error!r}')
        raise
    try:
        result = write_analysis(root, ev=ev)
    except (Exception, SystemExit) as error:             # the runs stay; by hand: run_tc.py decide
        raise Stop(f'the reading failed: {error!r}') from None
    log(f"reading: {result['reading']}")
    try:
        curve.write_ledger(ledger_sources(results), root/'analysis')
    except (Exception, SystemExit) as error:             # only a table; by hand: run_tc.py ledger.  It stops nothing
        log(f'the time ledger failed: {error!r}')
    run_measure(system, root, o, log, results, repo)
    log('Handed to the lead: no next step starts here')
    return 0


def queue(o, system=None, results=None, repo=None, ev=None):
    results = core.RESULTS if results is None else results
    repo = core.MAIN_REPO if repo is None else repo
    problem = root_problem(o.root, results)
    if problem is None and not o.min_host_gib >= DEFAULT_MIN_HOST_GIB:     # the brief's floor can be raised here, not lowered (review m10); written so that nan is refused too: `free < nan` would open the gate
        problem = f'--min-host-gib {o.min_host_gib:g}: the floor is at least {DEFAULT_MIN_HOST_GIB:g} GiB; nothing read, nothing moved'
    if problem:                                          # before the machine and before the lock file: nothing is created or touched
        print(problem, file=sys.stderr)
        return 2
    system = System() if system is None else system
    root = Path(o.root)
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root/'queue.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f'another driver is on {root}', file=sys.stderr)
        return 2

    def log(text):
        with open(root/'queue.log', 'a') as f:
            f.write(f'{core.utc()} {text}\n')
    log(f'start: root {root}; repo {repo}; {len(plan_runs())} runs in order, then the measurement; /mnt/c floor {o.min_host_gib:g} GiB')
    if (root/'queue_stop.json').exists():                # evidence, kept under its time; the file of that name always means this start
        (root/'queue_stop.json').rename(root/f"queue_stop_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    try:
        return queue_body(o, system, root, log, results, repo, ev=ev)
    except Stop as stop:
        log(f'STOP: {stop}')
        core.write_json(root/'queue_stop.json', dict(reason=str(stop), utc=core.utc()))
        return 8


# ---- the steps after a run --------------------------------------------------------------------------------------------------------------------------------------------------------------------
def evaluate_one(o, *, ev=None):
    require_cell(o.arm, o.pair)
    ev = load_evaluate() if ev is None else ev
    rec = ev.evaluate_run(Path(o.run), o.arm, o.pair, o.run_id, method=ARMS[o.arm]['method'])
    if ARMS[o.arm]['injected']:
        trace, _ = core.injection_trace(o.run)
        rec['seed'] = dict(trace, arm=core.read_json(Path(o.run)/'seed_arm.json'))
    rec['tc'] = core.read_json(Path(o.run)/'tc_arm.json')
    core.write_json(Path(o.run)/'metrics.json', rec)
    x, c = rec.get('metrics') or {}, rec.get('certificate') or {}
    print(f"{o.arm} {short_name(o.pair)}: seg35 {x.get('seg35')} seg4 {x.get('seg4')} SDlogJ {x.get('sdlogj_official')} cert {c.get('status')} passed {c.get('passed')}")
    return 0


def curve_one(o, *, ev=None, results=None):
    """curve.json and reproduction.json; deletes nothing (finish_run cleans up afterwards)."""
    require_cell(o.arm, o.pair)
    ev = load_evaluate() if ev is None else ev
    try:
        record = curve.curve_of_run(o.run, o.arm, o.pair, o.run_id, ev, clock=time.perf_counter)
        comparison = curve.compare_runs(o.run, original_dir(o.arm, o.pair, results))
    except curve.CurveError as error:
        print(f'{o.arm} {short_name(o.pair)}: the curve stops: {error}', flush=True)
        return 8
    core.write_json(Path(o.run)/'reproduction.json', comparison)
    core.write_json(Path(o.run)/'curve.json', record)      # last: its presence means the whole step is done
    print(f"{o.arm} {short_name(o.pair)}: curve of {record['n_fields_evaluated']} fields, {sum(record['evaluation_seconds']):.1f} s; sequence against the original: {comparison['sequence']['common_prefix']} of {comparison['sequence']['n_new']}/{comparison['sequence']['n_old']}")
    return 0


# ---- the dry run ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def command_tokens(cmd_text):
    """The prepare argv and the unit argv of a cmd.txt, as tokens."""
    lines = [line for line in cmd_text.splitlines() if not line.startswith('#') and line.strip()]
    if len(lines) != 2:
        raise SystemExit(f'cmd.txt: {len(lines)} command lines')
    return [shlex.split(line) for line in lines]


def normalized(tokens, replacements):
    out = []
    for token in tokens:
        for old, new in replacements:
            token = token.replace(old, new)
        out.append(token)
    return out


def command_difference(arm, pair, results=None):
    """This batch's two command lines against the original's cmd.txt, with the run directory, the root and the unit put in place: the tokens that still differ."""
    root = default_root(results)
    run, old = run_dir(root, arm, pair), original_dir(arm, pair, results)
    old_root = old.parents[2]
    old_prepare, old_unit = command_tokens((old/'cmd.txt').read_text())
    old_unit_name = (old/'service_unit.txt').read_text().strip()
    swap = [(str(old), str(run)), (str(old_root), str(root)), (old_unit_name, unit_name(arm, pair))]
    # shlex re-quotes: the unit's inner command is one token; the original has it in the same form after shlex.split
    new_prepare, new_unit = [str(t) for t in prepare_argv(arm, pair, run, results)], [str(t) for t in unit_argv(arm, pair, run, root)]
    differ = {}
    for name, new, was in (('prepare', new_prepare, normalized(old_prepare, swap)), ('unit', new_unit, normalized(old_unit, swap))):
        if new != was:
            differ[name] = [(a, b) for a, b in zip(new, was) if a != b]+([f'lengths {len(new)} {len(was)}'] if len(new) != len(was) else [])
    return differ


def disk_estimate(results=None):
    """From the originals' cleanup.json: per run the intermediate files that existed at its end, and what stays."""
    rows = []
    for arm, pair in plan_runs():
        run = original_dir(arm, pair, results)
        record = core.read_json(run/'cleanup.json') or {}
        rows.append(dict(arm=arm, pair=short_name(pair), gib=record.get('deleted_bytes', 0)/2**30, n=len(record.get('deleted', []))))
    return rows


def rehearse_reading(scratch, out, results=None):
    """The reading on a made-up batch: every time of every arm is that original's final map (metrics.json), so every Delta is 0 and the code must say plateau.  Formats only; only codes are printed; deleted with the scratch directory."""
    results = core.RESULTS if results is None else results
    root = scratch/'tc_rehearsal'
    for arm, pair in plan_runs():
        doc = core.read_json(original_dir(arm, pair, results)/'metrics.json')
        x = doc['metrics']
        metrics = dict(x, small_six=.5, large_six=.8, other_23=.8, cortex=x['seg4_per_label'].get('1'), white_matter=x['seg4_per_label'].get('3'))
        points = [dict(t=t, commit=1, elapsed_seconds=float(t), visible_seconds=float(t)+50, certificate_passed=True, D=1., GF=1., min_jd_lower_bound=1e-4, metrics=metrics) for t in curve.TIMES]
        final = dict(t=curve.FINAL, commit=2, elapsed_seconds=300., visible_seconds=350., certificate_passed=True, is_last_commit=True, D=1., GF=1., min_jd_lower_bound=1e-4, metrics=metrics)
        target = run_dir(root, arm, pair)
        target.mkdir(parents=True)
        core.write_json(target/'metrics.json', dict(doc, seed=dict(passed=True, problems=[])))
        core.write_json(target/'curve.json', dict(schema=curve.SCHEMA_CURVE, run_id='rehearsal', complete=True, self_check=dict(passed=True), points=points, final=final, n_native_commits=2))
    result = write_analysis(root)
    out(f"reading on copies of the originals' final maps at every time: code {result['reading_code']}; S_P(t) − S_K1(t) table rows {len(result['differences']['S_P(t) − S_K1(t)'])}; files {sorted(p.name for p in (root/'analysis').iterdir())}")
    if result['reading_code'] != decide_tc.PLATEAU:
        raise SystemExit('the rehearsal of the reading did not give what it must')


def rehearse_evaluation(out, results=None):
    """curve's evaluate_field on one original's final map (read only): the same seg35, seg4, SDlogJ as its metrics.json; the seconds it takes.  This reads the truth labels of one development pair, for the evaluation only."""
    ev = load_evaluate()
    results = core.RESULTS if results is None else results
    pair = PAIRS[0]
    old = original_dir('S_P', pair, results)
    labs, manifest, _ = ev.truth_labels(pair)
    task = old/core.read_json(old/'protocol.json')['task']['name']
    tick = time.perf_counter()
    with np.load(task/'final_native.npz') as z:
        u = np.array(z['u'])
    loaded = time.perf_counter()-tick
    tick = time.perf_counter()
    metrics = curve.evaluate_field(ev, u, labs, tuple(manifest['spacing_xyz']))
    seconds = time.perf_counter()-tick
    recorded = core.read_json(old/'metrics.json')['metrics']
    differences = {key: metrics[key]-recorded[key] for key in curve.SELF_CHECK_KEYS}
    out(f"evaluate_field on {old.name} of S_P (seed_v1, read only): differences from its metrics.json {differences}; load {loaded:.1f} s, evaluation {seconds:.1f} s at nice 19 on a busy machine; at most {MAX_FIELDS} fields a run")
    if any(abs(v) > curve.SELF_CHECK_TOL for v in differences.values()):
        raise SystemExit('evaluate_field does not reproduce the metrics of the run')
    return seconds


def measure_files(out):
    """The real files the measurement will read: they exist; the member files of the registry are where find_member_files looks.  No torch, no GPU."""
    problems = []
    registry = json.loads(measure_tc.REGISTRY.read_text())
    wanted = {sha for entry in registry['models'].values() for sha in entry['member_model_sha256']}
    found = {core.sha_file(p): p for p in sorted(measure_tc.UNET_ROOT.glob('*/member.pt'))}
    missing = sorted(wanted-set(found))
    out(f'{measure_tc.REGISTRY.name}: {len(registry["models"])} fold models, {len(wanted)} members, all found under {measure_tc.UNET_ROOT}: {not missing}')
    problems += [f'member {sha[:12]} not found' for sha in missing]
    image = measure_tc.IMAGE_ROOT/measure_tc.SUBJECT/'aligned_norm.nii.gz'       # bayesreg.seg.unet.data OasisFiles, IMAGE_FILE (that module needs jd_curl; the dry run does not)
    out(f'{image}: exists {image.is_file()}; recorded prediction {measure_tc.RECORDED_PREDICTION.is_file()}; recorded Q2 long {measure_tc.RECORDED_ENTRY.is_file()}')
    if not image.is_file():
        problems.append(f'{image}: no such file')
    if not measure_tc.RECORDED_PREDICTION.is_file() or not measure_tc.RECORDED_ENTRY.is_file():
        problems.append('a recorded file of the measurement is missing')
    return problems


def dry_run(scratch=None, out=print, results=None, repo=None):
    """The scratch directory goes, however the dry run ends."""
    results = core.RESULTS if results is None else results
    repo = core.MAIN_REPO if repo is None else repo
    made = scratch is None
    scratch = Path(tempfile.mkdtemp(prefix='tc_dry_')) if made else Path(scratch)
    if not made and scratch.exists():
        raise SystemExit(f'{scratch} exists (nothing is overwritten)')
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        return dry_run_steps(scratch, out, results, repo)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def dry_run_steps(scratch, out, results, repo):
    root = default_root(results)
    link = core.frozen_dir(results)/'data/three_d/stage_L2R_val_v1'
    link_before = os.readlink(link)
    existed = root.exists()
    plan = plan_runs()
    out(f'TC dry run ({core.utc()}): no solver, no GPU (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}), no unit; everything written under a scratch directory, deleted at the end.')
    out(f"{len(plan)} runs: {', '.join(ARMS)} on the first 5 pairs; --seconds {core.SECONDS}; entry charge {core.ENTRY_CHARGE_SECONDS} s inside the clock for S_P and S_K1; watchdog {core.WATCHDOG} s; root {root}; /mnt/c floor {DEFAULT_MIN_HOST_GIB:g} GiB, free now {shutil.disk_usage('/mnt/c').free/2**30:.0f} GiB; /home free now {shutil.disk_usage('/home').free/2**30:.0f} GiB; times {curve.TIMES} and the final map")
    problems = []
    selection = json.loads((core.MAIN_REPO/'data/three_d/stage_L2R_val_v1/selection.json').read_text())
    first = tuple(row['pair'] for row in selection['pairs'][:5])
    out(f"selection.json first 5 == the plan's pairs: {first == PAIRS}")
    if first != PAIRS:
        problems.append('the first 5 pairs of selection.json differ')
    sys.path.insert(0, str(ROOT))
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')

    out('\n==== the entries and the originals (real files: run.json, npy header, file sha256, the field bytes; no optimization) ====')
    copy = scratch/'frozen_copy'
    n_files = core.frozen_copy(copy, results)
    out(f'frozen source copied: {n_files} files equal to the freeze hashes; the worktree link stays')
    cells = {}
    for arm, pair in plan:
        files, entry = cell_files(arm, pair, results)
        check_pair_subjects(files[3]['fixed_subject'], files[3]['moving_subject'], allowlist)
        bad = original_problems(arm, pair, files, entry, results)
        problems += bad
        cells[(arm, pair)] = (entry, files)
        where = 'T88 (no entry field)' if entry is None else f"{entry['field_path']} sha256 {entry['field_sha256'][:16]} K {entry['source_k']} oracle {entry['oracle']} source {entry['source_run_id']}"
        out(f"{arm} {short_name(pair)}: {where}; subjects on the allowlist; original {original_dir(arm, pair, results)}: {'as this cell' if not bad else bad}")

    out('\n==== the input record, colour table and registry checks (main checkout) ====')
    system = System()
    bad = system.inputs_problems('S_P', repo, results)
    out(f"{core.inputs_root('S_P', results)}: {'ok' if not bad else bad}")
    problems += bad

    out(f'\n==== all {len(plan)} runs: the real parser and prepare of the frozen source (scratch copy), check_prepared, the protocol against the original of the same arm and pair; the two command lines against its cmd.txt; then the unit that the queue would start ====')
    prepared = scratch/'prepared'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    scratch_link = copy/'data/three_d/stage_L2R_val_v1'
    scratch_link.parent.mkdir(parents=True, exist_ok=True)
    for k, (arm, pair) in enumerate(plan, 1):
        entry, files = cells[(arm, pair)]
        shown = run_dir(root, arm, pair)
        out(f"# item {k}: arm {arm}, pair {pair}, inputs {files[0]}, watchdog {core.WATCHDOG} s, unit {unit_name(arm, pair)}, run id tc_{arm}_{short_name(pair)}_<UTC>")
        out(shlex.join(prepare_argv(arm, pair, shown, results)))
        out(shlex.join(unit_argv(arm, pair, shown, root)))
        if scratch_link.is_symlink():
            scratch_link.unlink()
        scratch_link.symlink_to(files[0].parent)
        here = prepared/arm/pair
        here.parent.mkdir(parents=True, exist_ok=True)
        tokens = prepare_argv(arm, pair, here, results)
        tokens[2] = str(copy/'scripts/run_stage_3d_e48_pg.py')
        tick = time.perf_counter()
        done = subprocess.run(tokens, cwd=scratch, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise SystemExit(f'prepare refused {arm} {pair}: {done.stdout[-1500:]} {done.stderr[-1500:]}')
        check = system.check_prepared(here, pair, ARMS[arm]['stage'], files[1], files[2], config_entry(results))
        if not check['ok']:
            raise SystemExit(f"check_prepared refused {arm} {pair}: {check['problems']}")
        differs = core.protocol_differences(core.read_json(here/'protocol.json'), core.read_json(original_dir(arm, pair, results)/'protocol.json'))
        commands = command_difference(arm, pair, results)
        out(f'parser, prepare and check_prepared ok ({time.perf_counter()-tick:.1f} s); protocol against the original differs in: {json.dumps(differs, ensure_ascii=False)}; the command lines differ in: {json.dumps(commands, ensure_ascii=False)}')
        if differs:
            problems.append(f'{arm} {pair}: the protocol differs from the original in {sorted(differs)}')
        if commands:
            problems.append(f'{arm} {pair}: the command lines differ from the original')
        shutil.rmtree(here)

    out('\n==== which commit each time takes, on the original records (commits.json, history.json; no field is read; the clock: tc_curve.py) ====')
    out(f'clock: {curve.CLOCK}')
    out(f'rule: {curve.RULE}')
    clocks = {arm: dict(first=[], at_90=[], last=[]) for arm in ARMS}
    for arm, pair in plan:
        d = original_dir(arm, pair, results)
        commits = curve.native_commits(curve.task_dir(d))
        chosen = curve.choose(commits)
        offset = [c['visible_seconds']-c['elapsed_seconds'] for c in commits]
        out(f"{arm} {short_name(pair)}: {len(commits)} commits, first {commits[0]['elapsed_seconds']:.1f} s, last {commits[-1]['elapsed_seconds']:.1f} s; task clock - native clock {min(offset):.2f}-{max(offset):.2f} s; "+', '.join(f"{t}: c{None if p is None else commits[p]['index']}" for t, p in chosen.items()))
        if any(p is None for p in chosen.values()):
            problems.append(f'{arm} {pair}: a time without a commit on the original')
        else:
            clocks[arm]['first'].append(commits[0]['visible_seconds'])
            clocks[arm]['at_90'].append(commits[chosen[90]]['visible_seconds'])
            clocks[arm]['last'].append(commits[-1]['visible_seconds'])
    out('\n==== the three clocks on the original records (plan revision 1; the means of the commits taken; the real total time is an inference, decide_tc.REAL_SECONDS) ====')
    for line in decide_tc.CLOCKS.values():
        out(line)
    for arm, row in clocks.items():
        out(f'{arm}: '+'; '.join(f"{name}: task clock {np.mean(values):.1f} s, inferred real total {decide_tc.real_total_seconds(arm, float(np.mean(values))):.0f} s" for name, values in row.items() if values))

    out('\n==== evaluate_field against a metrics.json ====')
    seconds = rehearse_evaluation(out, results)

    out('\n==== the comparison of the sequences, on the original against itself (read only: every position must agree) ====')
    d = original_dir('S_K1', PAIRS[0], results)
    seq = curve.compare_sequences(curve.native_commits(curve.task_dir(d)), curve.native_commits(curve.task_dir(d)))
    out(f"S_K1 0438-0439 against itself: common prefix {seq['common_prefix']} of {seq['n_new']}, identical {seq['identical']}")
    if not seq['identical']:
        problems.append('the comparison of a run with itself')

    out('\n==== the reading (made-up curves from the originals\' final maps: formats only, not results) ====')
    rehearse_reading(scratch, out, results)

    out("\n==== the time ledger of S_P's 19 runs (real records, only read; a copy of the table the queue writes under analysis/) ====")
    ledger = curve.write_ledger(ledger_sources(results), scratch/'ledger')
    out((scratch/'ledger/time_ledger_sp19.md').read_text().rstrip())
    if ledger['n_runs'] != 19:
        problems.append(f"the ledger has {ledger['n_runs']} runs")

    out('\n==== the measurement: the unit, the files it reads ====')
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    out(shlex.join(measure_argv(root, repo, results, head)))
    problems += measure_files(out)

    out('\n==== the disk, from the originals\' cleanup.json (each run\'s intermediate files exist until its curve is done; before, until its evaluation) ====')
    rows = disk_estimate(results)
    for arm in ARMS:
        values = [r['gib'] for r in rows if r['arm'] == arm]
        out(f"{arm}: {min(values):.2f}-{max(values):.2f} GiB of committed_*.npz and level_*.npz per run ({[r['n'] for r in rows if r['arm'] == arm]} files)")
    out(f"the largest at one time: {max(r['gib'] for r in rows):.2f} GiB (serial, each deleted before the next starts); stays per run: as the originals, du 262-266 MiB of final_native.npz and small files")
    fields = curve.choose(curve.native_commits(curve.task_dir(original_dir('S_P', PAIRS[0], results))))
    distinct = len({v for v in fields.values() if v is not None})+1
    out(f'evaluation: {distinct} fields on the first original, {seconds:.1f} s each here (one field, nice 19, a busy machine): about {distinct*seconds:.0f} s a run on top of the original post-processing')

    out('\n==== the queue start string (the lead) ====')
    out(shlex.join(queue_launch(root, repo)))

    if os.readlink(link) != link_before:
        problems.append("the frozen worktree's link changed")
    if root.exists() != existed:
        problems.append(f'{root} was created')
    out(f"\nthe frozen worktree's link unchanged: {link_before}; {root} {'exists' if existed else 'not created'}; no solver run; evaluate_run and the curve on a real commit are not rehearsed (no commit file exists: the originals' are deleted); the scratch directory goes when this ends")
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['--dry-run']:
        d = argparse.ArgumentParser()
        d.add_argument('--dry-run', action='store_true')
        d.add_argument('--scratch', default=None)
        parsed = d.parse_args(argv)
        return dry_run(scratch=parsed.scratch)
    p = argparse.ArgumentParser(description='TC: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(default_root()))
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--min-host-gib', type=float, default=DEFAULT_MIN_HOST_GIB)
    sub.add_parser('decide').add_argument('--root', default=str(default_root()))
    sub.add_parser('ledger').add_argument('--root', default=str(default_root()))
    for name in ('evaluate-one', 'curve-one'):
        e = sub.add_parser(name)
        e.add_argument('run')
        e.add_argument('--arm', required=True)
        e.add_argument('--pair', required=True)
        e.add_argument('--run-id', required=True)
    o = p.parse_args(argv)
    if o.command == 'queue':
        return queue(o)
    if o.command in ('decide', 'ledger'):
        problem = root_problem(o.root)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        if o.command == 'decide':
            write_analysis(o.root)
        else:
            curve.write_ledger(ledger_sources(), Path(o.root)/'analysis')
        return 0
    require_cell(o.arm, o.pair)                          # before any file
    problem = own_run_problem(o.run, o.arm, o.pair)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    return evaluate_one(o) if o.command == 'evaluate-one' else curve_one(o)


if __name__ == '__main__':
    sys.exit(main())
