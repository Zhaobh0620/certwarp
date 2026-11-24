"""SEED19 (runs/20261104-seed19/任务书.md, plan.md; docs/bayesreg/05_D0协议.md §17): S_P on the other 14 development pairs, in the new root seed_v1b.  Configuration, segmenter and charge as S_P on the first 5 pairs (run_seed.py, not changed: inject_run.py, the options,
330 s, 47 s inside the clock); only the entries come from elsewhere: mc0p_v1c Q2 long, fixed pair by pair.  The control arm S_K1 (K = 1, 600 steps: mc0p_v1 Q0 long, the first 5 pairs; BZ authorized it, docs/decision_log.md 2026-11-04) runs only with --include-s-k1, and then first.  Not a method yet.

    python -B experiments/seed/run_seed19.py --dry-run [--scratch DIR] [--include-s-k1 | --only-s-k1]
        prints the 15 runs of mc0p_v1c (run_mc0pc.py --dry-run) and the 14 runs here, their command lines, the real frozen parser and prepare on each (a scratch copy of the frozen source), the protocol against B1TM's, the entry checks ("入口待 mc0p_v1c" until they exist),
        a rehearsal of the reading on B1TM's numbers.  No solver, no GPU, no unit, nothing written outside the scratch directory.
    python -B experiments/seed/run_seed19.py queue [--wait-minutes 30] [--min-host-gib 15] [--include-s-k1 | --only-s-k1]
        the lead's start, in the unit printed by the dry run.  Start checks first (frozen worktree, inputs records, every entry of the plan, the controls, S_P of seed_v1): a missing file stops before any run.  --only-s-k1 is the fallback (plan revision 1, off by default): S_K1 alone, the 14 entries of S_P are not looked at.
        Then, with a switch, S_K1 on the 5 pairs; then, without --only-s-k1, S_P on the 14 pairs; serial, each behind
        SEED's gate with its own queue unit and /mnt/c free >= --min-host-gib (no $ROOT/hold, no other tag-* or seq-vp-* unit, 1-minute load < 1.5, no GPU compute process, GPU utilization <= 20%, GPU free >= 14 GiB, /home free >= 100 GiB); closed for --wait-minutes: queue_stop.json,
        the runs that finished are summed up, exit 8.  60 s between runs.  One unit tag-seedb-<arm>-<ffff>-<mmmm> per run (MemoryMax 12G).  A run directory is never rerun.  The stops of SEED (protocol §16.5, §16.6).  At the end the reading (decide_seed19.py); nothing else starts.
    python -B experiments/seed/run_seed19.py decide       analysis/decision_seed19.json, summary_seed19.md, long.csv
    python -B experiments/seed/run_seed19.py evaluate-one RUN --arm ARM --pair PAIR --run-id ID     the queue's step: experiments/d0/evaluate.py evaluate_run, plus the record of the entry

The only root is seed_v1b: queue, decide and evaluate-one refuse anything else (evaluate-one: anything but this batch's run directory of that arm and pair); seed_v1, mc0p_v1, mc0p_v1b and mc0p_v1c are only read.  Exit codes: 0 done, 2 usage, another driver on the root or another root, 8 STOP or a failed check.
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
sys.path.insert(0, str(HERE))

import run_seed as core  # noqa: E402

ROOT = core.ROOT
ROOT_NAME = 'seed_v1b'
FIRST_ROOT = 'seed_v1'                                  # S_P of the first 5 pairs: only read
ENTRY_ROOT = 'mc0p_v1c'
QUEUE_UNIT = 'tag-seedb-queue'
DEFAULT_MIN_HOST_GIB = 15.                              # the lead, the brief; BZ may change it.  F's own guard stays 10 GiB + pending (F/scripts/run_stage_3d_e19_mri.py:460, 546)
REST_PAIRS = tuple(f'OASIS_OAS1_{n:04d}_MR1__OASIS_OAS1_{n+1:04d}_MR1' for n in range(443, 457))     # selection.json, pairs 6..19 (the dry run compares them; a test compares them with experiments/mc0/run_mc0pc.py)
FIRST_PAIRS = core.PAIRS
MC0P_SCHEMA = 'bayesreg.mc0p.run.v1'
INPUTS = core.ARMS['S_P']['inputs']
# The arm, the pairs and the root of the entry, written down: never "whichever exists".
ARMS = dict(S_P=dict(inputs=INPUTS, input_arm='A5P', pairs=REST_PAIRS, entry_root=ENTRY_ROOT, source_arm='Q2', source_prefix='mc0p_Q2_long_', k=35, control='B1TM'),
            S_K1=dict(inputs=INPUTS, input_arm='A5P', pairs=FIRST_PAIRS, entry_root=core.RETRIED_ROOT, source_arm='Q0', source_prefix='mc0p_Q0_long_', k=1, control='B1TM'))
ORACLE = False                                          # neither arm: predicted channels, predicted colouring
Stop, EntryError, EntryMissing = core.Stop, core.EntryError, core.EntryMissing


class GateClosed(core.Stop):
    """The gate stayed closed for its time: queue_stop.json, the runs that finished are summed up (the brief, the disk floor)."""


def require_cell(arm, pair):
    """Only the planned cells: S_P on the 14 pairs, S_K1 on the first 5.  Checked before any file of the pair is opened."""
    if arm not in ARMS:
        raise SystemExit(f'SEED19 has no arm {arm}; only {tuple(ARMS)}')
    if pair not in ARMS[arm]['pairs']:
        raise SystemExit(f"SEED19 runs {arm} only on {[core.short_name(p) for p in ARMS[arm]['pairs']]}; refused: {pair}")
    return arm, pair


def plan_runs(include_s_k1=False, only_s_k1=False):
    """S_P on the 14 pairs.  include_s_k1: S_K1 on the 5 pairs comes first (the brief, main 41e421a).  only_s_k1: S_K1 alone, the fallback when an entry of S_P is missing (plan revision 1).  Neither: S_K1 does not run."""
    if include_s_k1 and only_s_k1:
        raise SystemExit('--include-s-k1 and --only-s-k1 exclude each other')
    return ([('S_K1', pair) for pair in FIRST_PAIRS] if include_s_k1 or only_s_k1 else [])+([] if only_s_k1 else [('S_P', pair) for pair in REST_PAIRS])


def default_root(results=None):
    return Path(core.RESULTS if results is None else results)/ROOT_NAME


def root_problem(root, results=None):
    """Why this batch may not write at this root, or None: only its own."""
    return None if Path(root).resolve() == default_root(results).resolve() else f'{root}: this batch writes only to {default_root(results)}; nothing read, nothing moved'


def run_dir(root, arm, pair):
    require_cell(arm, pair)
    return Path(root)/'runs'/arm/pair


def unit_name(arm, pair):
    return f'tag-seedb-{arm}-{core.short_name(pair)}'


def new_run_id(arm, pair):
    return f"seedb_{arm}_{core.short_name(pair)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def inputs_root(arm, results=None):
    return Path(core.RESULTS if results is None else results)/ARMS[arm]['inputs']


def b1_entry(arm, results=None):
    return core.b1_spec.check_entry(dict(arm=arm, inputs_root=str(inputs_root(arm, results)), seconds=core.SECONDS, changes=core.CHANGES))


def method_name(arm):
    return f"frozen_eb808be_seed_{core.SECONDS}s_gw{'_'.join(f'{x:g}' for x in core.GEOMETRY)}_entry{ARMS[arm]['source_arm']}long"


def prepare_argv(arm, pair, run, results=None):
    require_cell(arm, pair)
    return core.check_launch([core.PY, '-B', core.frozen_dir(results)/'scripts/run_stage_3d_e48_pg.py', 'prepare', run, *core.OPTS, '--stage', f'SEED-{arm}', '--pair', pair])


def unit_argv(arm, pair, run, root):
    require_cell(arm, pair)
    return core.unit_argv(arm, pair, run, root, unit=unit_name(arm, pair))


def queue_launch(root=None, repo=core.MAIN_REPO, include_s_k1=False, min_host_gib=DEFAULT_MIN_HOST_GIB, only_s_k1=False):
    """The lead's start string of the queue."""
    root = default_root() if root is None else root
    return core.check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT}', '--collect', f'--working-directory={repo}', '--property=MemoryMax=9G', '--setenv=PYTHONDONTWRITEBYTECODE=1',
                              '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=NUMEXPR_NUM_THREADS=1',
                              core.PY, '-B', 'experiments/seed/run_seed19.py', 'queue', '--root', str(root), '--min-host-gib', f'{min_host_gib:g}', *(['--include-s-k1'] if include_s_k1 else []), *(['--only-s-k1'] if only_s_k1 else [])])


# ---- the inputs and the entry ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def input_files(arm, pair, results=None):
    """run_seed.input_files for this arm: (pair.npz directory, their sha256, the manifest), checked against the inputs record and the arm; EntryError otherwise."""
    require_cell(arm, pair)
    root = inputs_root(arm, results)
    record = core.read_json(root/'d0_inputs.json')
    if record is None:
        raise EntryError(f'{root}/d0_inputs.json: no such file')
    d = root/'stage_L2R_val_v1'/pair
    if not (d/'pair.npz').is_file() or not (d/'manifest.json').is_file():
        raise EntryError(f'{d}: no pair.npz or manifest.json')
    pair_sha, manifest_sha = core.sha_file(d/'pair.npz'), core.sha_file(d/'manifest.json')
    listed = record.get('files_sha256') or {}
    if listed.get(f'stage_L2R_val_v1/{pair}/pair.npz') != pair_sha or listed.get(f'stage_L2R_val_v1/{pair}/manifest.json') != manifest_sha:
        raise EntryError(f'{d}: the files differ from {root}/d0_inputs.json')
    manifest = json.loads((d/'manifest.json').read_text())
    made = manifest.get('preprocessing_d0') or {}
    if manifest.get('pair') != pair or manifest.get('pair_sha256') != pair_sha or made.get('arm') != ARMS[arm]['input_arm'] or made.get('oracle') is not ORACLE or record.get('oracle') is not ORACLE:
        raise EntryError(f"{d}: manifest pair {manifest.get('pair')}, arm {made.get('arm')}, oracle {made.get('oracle')}; expected {pair}, {ARMS[arm]['input_arm']}, {ORACLE}")
    return d, pair_sha, manifest_sha, manifest


def entry_dir(arm, pair, results=None):
    """The one place an entry may come from.  S_P's 14 pairs: mc0p_v1c/runs/Q2_long; S_K1's 5: mc0p_v1/runs/Q0_long."""
    require_cell(arm, pair)
    spec = ARMS[arm]
    return Path(core.RESULTS if results is None else results)/spec['entry_root']/'runs'/f"{spec['source_arm']}_{core.ENTRY_TIER}"/core.short_name(pair)


def check_entry(arm, pair, manifest, results=None):
    """run_seed.check_entry with this batch's map: the path, then the run record: the pair, the source arm and tier, K, status done, oracle false, retry_of empty, the channel sources, the run's images from the same original pair as the arm's inputs; then the file: sha256,
    one member u, float64, the grid's shape, bytes sha256, finite, zero faces, and the taper against the evaluation's record of the field.  The record for seed_entry.json, or EntryError (EntryMissing if the files are not there)."""
    spec = ARMS[arm]
    d = entry_dir(arm, pair, results)
    short = core.short_name(pair)
    field_path, run_path = d/'field.npz', d/'run.json'
    if not field_path.is_file() or not run_path.is_file():
        raise EntryMissing(f"{arm} {short}: 入口待 {spec['entry_root']}: {field_path} 或 {run_path.name} 不存在（{spec['source_arm']} 长档还没有跑，或这一对失败了）")
    run = json.loads(run_path.read_text())
    problems = []
    if run.get('pair') != pair:
        problems.append(f"run.json pair {run.get('pair')}")
    if run.get('arm') != spec['source_arm'] or run.get('tier') != core.ENTRY_TIER or run.get('k') != spec['k']:
        problems.append(f"run.json arm {run.get('arm')} tier {run.get('tier')} K {run.get('k')}, expected {spec['source_arm']} {core.ENTRY_TIER} {spec['k']}")
    if run.get('status') != 'done':
        problems.append(f"run.json status {run.get('status')}: only a finished run is an entry")
    if run.get('oracle') is not ORACLE:
        problems.append(f"run.json oracle {run.get('oracle')}, the arm {arm} must have {ORACLE}")
    if run.get('run_id') != spec['source_prefix']+short:
        problems.append(f"run.json run_id {run.get('run_id')}, expected {spec['source_prefix']+short}")
    if run.get('schema') != MC0P_SCHEMA:
        problems.append(f"run.json schema {run.get('schema')}, expected {MC0P_SCHEMA}")
    if run.get('retry_of'):
        problems.append(f"run.json retry_of {run.get('retry_of')!r}: an entry of this batch is no retry")
    config = run.get('config') or {}
    if {k: config.get(k) for k in core.LONG_TIER} != core.LONG_TIER:
        problems.append(f'run.json config {config}: not the long tier')
    problems += core.channel_sources_problems(run)
    made = manifest.get('preprocessing_d0') or {}
    data = run.get('data') or {}
    if data.get('pair_sha256') != made.get('source_pair_sha256') or data.get('manifest_sha256') != made.get('source_manifest_sha256'):
        problems.append(f"the field comes from original pair {data.get('pair_sha256')}, the inputs of {arm} from {made.get('source_pair_sha256')}")
    if data.get('fixed_subject') != manifest.get('fixed_subject') or data.get('moving_subject') != manifest.get('moving_subject'):
        problems.append(f"the field's subjects {data.get('fixed_subject')}, {data.get('moving_subject')}; the inputs' {manifest.get('fixed_subject')}, {manifest.get('moving_subject')}")
    field = run.get('field') or {}
    if field.get('file') != 'field.npz' or not field.get('sha256') or not field.get('u_sha256'):
        problems.append(f'run.json has no field record: {field}')
    if problems:
        raise EntryError(f'{d}: '+'; '.join(problems))
    got = core.sha_file(field_path)
    if got != field['sha256']:
        raise EntryError(f"{field_path}: sha256 {got} != run.json {field['sha256']}")
    shape = tuple(manifest['shape_zyx'])
    names, stored, dtype = core.npz_header(field_path)
    if names != ['u.npy'] or stored != shape+(3,) or dtype not in ('<f8', 'float64'):
        raise EntryError(f'{field_path}: members {names}, shape {stored}, dtype {dtype}; expected only u, {shape+(3,)}, float64')
    record = dict(arm=arm, pair=pair, field_path=str(field_path), field_sha256=got, u_sha256=field['u_sha256'], source_run_id=run['run_id'], oracle=ORACLE, charge_seconds=core.ENTRY_CHARGE_SECONDS, shape_zyx=list(shape))
    try:
        u = core.inject_run.load_field(record)           # the wrapper's own check, once more when it starts
    except core.inject_run.Refused as refused:
        raise EntryError(str(refused)) from None
    tapered = core.boundary_taper_sha256(u, shape)
    metrics = core.read_json(d/'metrics.json') or {}
    if metrics.get('tapered_u_sha256') not in (None, tapered):
        raise EntryError(f"{d}: the evaluation tapered this field to {metrics.get('tapered_u_sha256')}, here {tapered}")
    certificate = metrics.get('certificate') or {}
    x = metrics.get('metrics') or {}
    return dict(record, tapered_sha256=tapered, tapered_recorded_by_evaluation=metrics.get('tapered_u_sha256'), source_seconds_total=run.get('seconds_total'), source_code_sha=run.get('code_sha'), source_status=run.get('status'), source_k=run.get('k'),
                source_tier=run.get('tier'), source_arm=run.get('arm'), source_root=spec['entry_root'], retry_of=run.get('retry_of'), role=core.inject_run.ROLE,
                entry_metrics=dict(seg35=x.get('seg35'), seg4=x.get('seg4'), sdlogj_official=x.get('sdlogj_official'), accepted_alpha=certificate.get('accepted_alpha'), at_accepted_alpha=metrics.get('at_accepted_alpha')))


# ---- the machine -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class System(core.System):
    """SEED's machine; the arms' inputs and the evaluation step are this batch's."""

    def inputs_problems(self, arm, repo, results=None):
        return core.b1_spec.check_one_input(b1_entry(arm, results), repo)

    def evaluate(self, run, arm, pair, run_id):
        with open(Path(run)/'log.txt', 'ab') as f:
            return subprocess.run(core.check_launch([core.PY, '-B', HERE/'run_seed19.py', 'evaluate-one', run, '--arm', arm, '--pair', pair, '--run-id', run_id]), stdout=f, stderr=subprocess.STDOUT).returncode


def wait_for_gate(system, root, minutes, log, min_host_gib):
    try:
        core.wait_for_gate(system, root, minutes, log, queue_unit=QUEUE_UNIT, host_floor_gib=min_host_gib)
    except Stop as stop:
        raise GateClosed(str(stop)) from None


# ---- one run ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def write_records(run, arm, pair, unit, run_id, entry, files, env_text, repo, results=None):
    """run_seed.write_records for this arm: the records of protocol §8 (git.txt, env.txt, data.txt, seed), the entry's own record and the wrapper's copy."""
    run = Path(run)
    d, pair_sha, manifest_sha, manifest = files
    git = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', '-C', str(repo), 'diff', '--stat'], capture_output=True, text=True).stdout
    files_sha = '\n'.join(f'{p.relative_to(ROOT)} {core.sha_file(p)}' for pattern in ('experiments/seed/*.py', 'experiments/d0/*.py', 'experiments/d0/*.json') for p in sorted(ROOT.glob(pattern)))
    frozen = core.frozen_dir(results)
    frozen_head = subprocess.run(['git', '-C', str(frozen), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    frozen_status = subprocess.run(['git', '-C', str(frozen), 'status', '--porcelain'], capture_output=True, text=True).stdout
    untracked = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--', 'experiments/seed', 'experiments/d0', 'docs/bayesreg', 'configs/bayesreg', 'runs'], capture_output=True, text=True).stdout
    (run/'git.txt').write_text(f'repo: {repo}\nrepo HEAD: {git}\nrepo git diff --stat:\n{diff}\nrepo untracked or changed (experiments/seed, experiments/d0, docs/bayesreg, configs/bayesreg, runs):\n{untracked}\n'
                               f'frozen worktree: {frozen}\nfrozen HEAD: {frozen_head}\nfrozen commit expected: {core.FROZEN_COMMIT}\nfrozen git status --porcelain (must be empty):\n{frozen_status}\nfile sha256:\n{files_sha}\n')
    (run/'env.txt').write_text(env_text)
    made = manifest['preprocessing_d0']
    (run/'data.txt').write_text('\n'.join([f'run_id: {run_id}', f'arm: {arm}', f'pair: {pair}', f'unit: {unit}', f"fixed subject: {manifest['fixed_subject']}", f"moving subject: {manifest['moving_subject']}",
                                           f'arm stage directory (prepare source): {d.parent}', f'arm input pair.npz: {d}/pair.npz sha256 {pair_sha}', f'arm input manifest.json: {d}/manifest.json sha256 {manifest_sha}',
                                           f"original pair.npz sha256: {made['source_pair_sha256']}", f"original manifest sha256: {made['source_manifest_sha256']}", f"colouring: {made['recipe']}",
                                           f'oracle: {ORACLE}', f"entry field: {entry['field_path']} sha256 {entry['field_sha256']} bytes sha256 {entry['u_sha256']}",
                                           f"entry source run: {entry['source_run_id']}; K {entry['source_k']}; tier {entry['source_tier']}; oracle {entry['oracle']}", 'truth labels for evaluation: the original stage pair.npz (evaluate.py checks its sha256)'])+'\n')
    (run/'seed').write_text(f"{arm} (runs/20261104-seed19/plan.md): the inputs of {ARMS[arm]['input_arm']}, no new colouring; the entry field has no random step (run.json: none); solver: configuration C has no random number generator or seed option; GPU reductions may still differ between runs\n")
    core.write_json(run/'seed_entry.json', entry)
    core.write_json(run/'seed_arm.json', dict(arm=arm, seconds=core.SECONDS, changes=core.CHANGES, inputs_root=str(inputs_root(arm, results)), entry_root=ARMS[arm]['entry_root'], source_arm=ARMS[arm]['source_arm'], charge_seconds=core.ENTRY_CHARGE_SECONDS,
                                              watchdog_seconds=core.WATCHDOG, method=method_name(arm), input_oracle=ORACLE, control=ARMS[arm]['control'], wrapper_sha256=core.sha_file(HERE/'inject_run.py')))
    shutil.copyfile(HERE/'inject_run.py', run/'seed_inject.py')


def one_run(o, system, root, arm, pair, entry, files, log, env_text, results, repo, first):
    """run_seed.one_run for this batch.  Returns the state, or None if nothing ran because it had already ended."""
    run = run_dir(root, arm, pair)
    if (run/'queue_done.json').is_file():
        return None
    unit = unit_name(arm, pair)
    if run.exists():                                     # never rerun: an ended unit is post-processed, anything else stops
        if (run/'exit_code').is_file():
            log(f'resume: {run} has an exit_code; post-processing')
            return core.finish_run(run, arm, pair, (run/'run_id').read_text().strip(), (run/'uptime_before').read_text().strip(), system, log)
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
    check = system.check_prepared(run, pair, f'SEED-{arm}', files[1], files[2], b1_entry(arm, results))
    core.write_json(run/'prepared_check.json', check)
    if not check.get('ok'):
        raise Stop(f"prepared run differs from B1TM's configuration or the arm's inputs: {check.get('problems')}")
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
    return core.finish_run(run, arm, pair, run_id, t0, system, log)


def start_checks(system, results, repo, log, plan):
    """Everything that can be known before the first run: STOP here rather than in the middle.  Returns {(arm, pair): (entry record, input files)}."""
    problems = list(system.frozen_problems(results))
    prepared = {}
    for arm in sorted({a for a, _ in plan}):
        problems += [f'{arm}: {p}' for p in system.inputs_problems(arm, repo, results)]
    for arm, pair in plan:
        try:
            files = input_files(arm, pair, results)
            prepared[(arm, pair)] = (check_entry(arm, pair, files[3], results), files)
            if prepared[(arm, pair)][0]['tapered_recorded_by_evaluation'] is None:
                log(f"{arm} {core.short_name(pair)}: the source evaluation has no record of the taper yet; the field's taper is checked against the frozen formula only")
        except EntryError as error:
            problems.append(str(error))
    b1 = Path(results)/'b1_v1'
    if not (b1/'analysis/long.csv').is_file():
        problems.append(f'{b1}/analysis/long.csv: no such file (the control of the reading)')
    problems += [f"control B1TM {core.short_name(pair)}: no metrics.json" for pair in FIRST_PAIRS+REST_PAIRS if not (b1/'runs/B1TM'/pair/'metrics.json').is_file()]
    problems += [f'{FIRST_ROOT} S_P {core.short_name(pair)}: no metrics.json' for pair in FIRST_PAIRS if not (Path(results)/FIRST_ROOT/'runs/S_P'/pair/'metrics.json').is_file()]
    if problems:
        raise Stop('start checks: '+' | '.join(problems))
    log(f'start checks passed: frozen worktree, inputs records, the controls, {FIRST_ROOT}, all {len(prepared)} entries')
    return prepared


def trailing_infrastructure(root, plan):
    """Infrastructure failures in a row at the end of what has run, in plan order."""
    n = 0
    for arm, pair in plan:
        done = core.read_json(run_dir(root, arm, pair)/'queue_done.json')
        if done is None:
            continue
        n = n+1 if done.get('state') == 'infrastructure' else 0
    return n


def batch_record(root, plan, o, repo):
    """batch.json (review minor 8).  What defines the batch is checked at every start: a difference is a STOP before any run.  What may differ between starts (the plan under a switch, the floor) is the latest start's, and every start is listed."""
    fixed = dict(schema='bayesreg.seed19.batch.v1', seconds=core.SECONDS, changes=core.CHANGES, charge_seconds=core.ENTRY_CHARGE_SECONDS, frozen_commit=core.FROZEN_COMMIT, entry_roots={arm: spec['entry_root'] for arm, spec in ARMS.items()})
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    this = dict(utc=core.utc(), plan=[f'{a} {core.short_name(p)}' for a, p in plan], min_host_gib=o.min_host_gib, include_s_k1=o.include_s_k1, only_s_k1=o.only_s_k1, repo_head=head)
    batch = core.read_json(root/'batch.json')
    if batch is not None:
        different = {k: (batch.get(k), v) for k, v in fixed.items() if batch.get(k) != v}
        if different:
            raise Stop(f'batch.json differs from this batch: {different}')
    core.write_json(root/'batch.json', dict(fixed, started_utc=this['utc'] if batch is None else batch.get('started_utc'), **{k: v for k, v in this.items() if k != 'utc'}, starts=([] if batch is None else batch.get('starts', []))+[this]))


def queue_body(o, system, root, log, results, repo):
    import decide_seed19
    plan = plan_runs(o.include_s_k1, o.only_s_k1)
    prepared = start_checks(system, results, repo, log, plan)
    env_text = core.environment_text(repo) if isinstance(system, System) else 'test\n'
    (root/'cache/numba_jit').mkdir(parents=True, exist_ok=True)
    (root/'cache/first_use').mkdir(parents=True, exist_ok=True)
    batch_record(root, plan, o, repo)
    first = True
    try:
        for arm, pair in plan:
            state = one_run(o, system, root, arm, pair, *prepared[(arm, pair)], log, env_text, results, repo, first)
            if state is None:
                continue
            first = False
            if trailing_infrastructure(root, plan) >= core.MAX_CONSECUTIVE_INFRA:
                raise Stop(f'{core.MAX_CONSECUTIVE_INFRA} infrastructure failures in a row')
    except GateClosed as stop:                           # the runs that finished are summed up as usual, then STOP
        if any((run_dir(root, a, p)/'metrics.json').is_file() for a, p in plan):
            try:
                result = decide_seed19.write_analysis(root, results=results)
                log(f"the gate stopped the queue; reading of what finished: {result['reading_one']['text']}")
            except (Exception, SystemExit) as error:
                log(f'the gate stopped the queue; the summary failed: {error!r}')
        raise stop
    try:
        result = decide_seed19.write_analysis(root, results=results)
    except (Exception, SystemExit) as error:             # the runs stay; by hand: run_seed19.py decide
        raise Stop(f'the reading failed: {error!r}') from None
    log(f"reading: {result['reading_one']['text']}; target: {result['reading_two']['text']}; S_K1: {result['reading_s_k1']['text']}. Handed to the lead: no next step starts here")
    return 0


def queue(o, system=None, results=None, repo=None):
    plan_runs(o.include_s_k1, o.only_s_k1)               # the two switches exclude each other: before anything is touched
    system = System() if system is None else system
    results = core.RESULTS if results is None else results
    repo = core.MAIN_REPO if repo is None else repo
    problem = root_problem(o.root, results)
    if problem:                                          # before the lock file: nothing is created or touched
        print(problem, file=sys.stderr)
        return 2
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
    log(f'start: root {root}; repo {repo}; {len(plan_runs(o.include_s_k1, o.only_s_k1))} runs in order; S_K1 {o.include_s_k1 or o.only_s_k1}; only S_K1 {o.only_s_k1}; /mnt/c floor {o.min_host_gib:g} GiB')
    if (root/'queue_stop.json').exists():                # evidence, kept under its time; the file of that name always means this start
        (root/'queue_stop.json').rename(root/f"queue_stop_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    try:
        return queue_body(o, system, root, log, results, repo)
    except Stop as stop:
        log(f'STOP: {stop}')
        core.write_json(root/'queue_stop.json', dict(reason=str(stop), utc=core.utc()))
        return 8


# ---- evaluation ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def evaluate_one(o):
    require_cell(o.arm, o.pair)
    ev = core.load_evaluate()
    rec = ev.evaluate_run(Path(o.run), o.arm, o.pair, o.run_id, method=method_name(o.arm))
    trace, _ = core.injection_trace(o.run)
    rec['seed'] = dict(trace, arm=core.read_json(Path(o.run)/'seed_arm.json'))
    core.write_json(Path(o.run)/'metrics.json', rec)
    x, c = rec.get('metrics') or {}, rec.get('certificate') or {}
    print(f"{o.arm} {core.short_name(o.pair)}: seg35 {x.get('seg35')} seg4 {x.get('seg4')} SDlogJ {x.get('sdlogj_official')} cert {c.get('status')} passed {c.get('passed')} alpha {trace.get('accepted_alpha')} entry check {trace.get('passed')}")
    return 0


# ---- the dry run --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def rehearse_reading(scratch, out, results=None):
    """The reading on B1TM's own numbers (copies of its metrics.json as S_P of the 14 pairs): every delta is 0, so the code must say not replicated.  Formats and pins only; only codes are printed; deleted with the scratch directory."""
    import decide_seed19
    results = core.RESULTS if results is None else results
    root = scratch/'seed19_rehearsal'
    for pair in REST_PAIRS:
        doc = core.read_json(Path(results)/'b1_v1/runs/B1TM'/pair/'metrics.json')
        doc.update(arm='S_P', run_id=f'rehearsal_S_P_{core.short_name(pair)}', method=method_name('S_P'), seed=dict(passed=True, problems=[], accepted_alpha=None, arm=dict(seconds=core.SECONDS, changes=core.CHANGES, inputs_root=str(inputs_root('S_P', results)), charge_seconds=core.ENTRY_CHARGE_SECONDS)))
        target = root/'runs/S_P'/pair
        target.mkdir(parents=True)
        core.write_json(target/'metrics.json', doc)
    result = decide_seed19.write_analysis(root, results=results)
    out(f"reading on copies of B1TM: reading one {result['reading_one']['code']}, n {result['reading_one']['n']}; reading two code {result['reading_two']['code']}, n {result['reading_two']['n']} (first 5 from {FIRST_ROOT}; a rehearsal, its number is not printed); S_K1 {result['reading_s_k1']['code']}; reference problems {result['reference_problems']}; "
        f"first-5 problems {result['first5_problems']}; pins reproduced: {result['pins']}; files {sorted(p.name for p in (root/'analysis').iterdir())}")
    if result['reference_problems'] or result['first5_problems'] or result['reading_one']['code'] != decide_seed19.NOT_REPLICATED or result['reading_s_k1']['code'] != decide_seed19.K1_NOT_RUN:
        raise SystemExit('the rehearsal of the reading did not give what it must')


def dry_run(scratch=None, out=print, results=None, repo=None, include_s_k1=False, with_mc0pc=True, only_s_k1=False):
    """The scratch directory goes, however the dry run ends."""
    results = core.RESULTS if results is None else results
    repo = core.MAIN_REPO if repo is None else repo
    made = scratch is None
    scratch = Path(tempfile.mkdtemp(prefix='seed19_dry_')) if made else Path(scratch)
    if not made and scratch.exists():
        raise SystemExit(f'{scratch} exists (nothing is overwritten)')
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        return dry_run_steps(scratch, out, results, repo, include_s_k1, with_mc0pc and not only_s_k1, only_s_k1)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def dry_run_steps(scratch, out, results, repo, include_s_k1, with_mc0pc, only_s_k1=False):
    root = default_root(results)
    link = core.frozen_dir(results)/'data/three_d/stage_L2R_val_v1'
    link_before = os.readlink(link)
    existed = root.exists()
    plan = plan_runs(include_s_k1, only_s_k1)
    out(f'SEED19 dry run ({core.utc()}): no solver, no GPU (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}), no unit; everything written under a scratch directory, deleted at the end.')
    problems = []
    if with_mc0pc:
        out('\n==== the 15 runs of mc0p_v1c: experiments/mc0/run_mc0pc.py --dry-run ====')
        done = subprocess.run([core.PY, '-B', str(ROOT/'experiments/mc0/run_mc0pc.py'), '--dry-run'], capture_output=True, text=True)
        out(done.stdout.rstrip()+(f'\n{done.stderr.rstrip()}' if done.stderr.strip() else ''))
        if done.returncode:
            problems.append(f'run_mc0pc.py --dry-run: exit {done.returncode}')
    out(f'\n==== the {len(plan)} runs of {ROOT_NAME} ====')
    out(f"{len(plan)} runs; S_K1 in the plan: {include_s_k1 or only_s_k1} (only with --include-s-k1, and then first; BZ authorized it, docs/decision_log.md 2026-11-04); only S_K1: {only_s_k1} (the fallback, plan revision 1: the 14 entries of S_P are not looked at); --seconds {core.SECONDS}; --geometry-weight-levels {','.join(f'{x:g}' for x in core.GEOMETRY)}; entry charge {core.ENTRY_CHARGE_SECONDS} s inside the clock (no-competition measure 61 s, kept 47 s to compare with the first 5 pairs); watchdog {core.WATCHDOG} s; root {root}; /mnt/c floor {DEFAULT_MIN_HOST_GIB:g} GiB; /mnt/c free now {shutil.disk_usage('/mnt/c').free/2**30:.0f} GiB")
    selection = json.loads((core.MAIN_REPO/'data/three_d/stage_L2R_val_v1/selection.json').read_text())
    rest = tuple(row['pair'] for row in selection['pairs'][5:19])
    out(f"selection.json pairs 6..19 == the plan's 14 pairs: {rest == REST_PAIRS}; 19 pairs in all: {len(selection['pairs']) == 19}")
    if rest != REST_PAIRS or len(selection['pairs']) != 19:
        problems.append('selection.json: not the plan')
    sys.path.insert(0, str(ROOT))
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')

    out('\n==== the entries (real files when they exist: run.json, npy header, file sha256, the field bytes; no optimization) ====')
    copy = scratch/'frozen_copy'
    n_files = core.frozen_copy(copy, results)
    out(f'frozen source copied: {n_files} files equal to the freeze hashes; the worktree link stays')
    pending = {}
    for arm, pair in plan:
        files = input_files(arm, pair, results)
        check_pair_subjects(files[3]['fixed_subject'], files[3]['moving_subject'], allowlist)
        try:
            e = check_entry(arm, pair, files[3], results)
            out(f"{arm} {core.short_name(pair)}: {e['field_path']} sha256 {e['field_sha256'][:16]} bytes {e['u_sha256'][:16]} source {e['source_run_id']} K {e['source_k']} oracle {e['oracle']} seconds_total {e['source_seconds_total']:.2f}; subjects on the allowlist; the evaluation accepted alpha {e['entry_metrics']['accepted_alpha']}")
        except EntryMissing as missing:
            pending.setdefault(arm, []).append(pair)
            out(str(missing))
        except EntryError as error:
            problems.append(str(error))
            out(f'{arm} {core.short_name(pair)}: REFUSED: {error}')
    if pending:
        out(f"pending entries: { {a: len(p) for a, p in pending.items()} }: 入口待 {ENTRY_ROOT}; not an error of the dry run; the queue stops at its start checks, before any run, until they exist")
    out(f'wrong sources are refused (the wrong pair, an oracle record, the wrong root, a retry_of): tests/test_bayesreg/solve/test_seed19_tools.py')

    out('\n==== the colour table and registry checks on the real inputs record (main checkout) ====')
    system = System()
    for arm in sorted({a for a, _ in plan}):
        bad = system.inputs_problems(arm, repo, results)
        out(f'{arm}: {inputs_root(arm, results)}: {"ok" if not bad else bad}')
        problems += bad

    out(f"\n==== all {len(plan)} runs: the real parser and prepare of the frozen source (scratch copy), check_prepared with the arm, the protocol against B1TM of the same pair (only `stage` may differ: the injection's records are other files, seed_entry.json, seed_arm.json, seed_inject.py); then the unit that the queue would start ====")
    prepared = scratch/'prepared'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    scratch_link = copy/'data/three_d/stage_L2R_val_v1'
    scratch_link.parent.mkdir(parents=True, exist_ok=True)
    for k, (arm, pair) in enumerate(plan, 1):
        files = input_files(arm, pair, results)
        shown = run_dir(root, arm, pair)
        out(f"# item {k}: arm {arm}, pair {pair}, inputs {files[0]}, entry {entry_dir(arm, pair, results)}/field.npz, control {ARMS[arm]['control']}, watchdog {core.WATCHDOG} s, unit {unit_name(arm, pair)}, run id seedb_{arm}_{core.short_name(pair)}_<UTC>")
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
        check = system.check_prepared(here, pair, f'SEED-{arm}', files[1], files[2], b1_entry(arm, results))
        if not check['ok']:
            raise SystemExit(f"check_prepared refused {arm} {pair}: {check['problems']}")
        control = Path(results)/'b1_v1/runs/B1TM'/pair/'protocol.json'
        differs = core.protocol_differences(core.read_json(here/'protocol.json'), core.read_json(control))
        out(f'parser, prepare and check_prepared ok ({time.perf_counter()-tick:.1f} s); protocol against B1TM of this pair differs only in: {json.dumps(differs, ensure_ascii=False)}')
        if set(differs) != {'stage'}:
            problems.append(f'{arm} {pair}: the protocol differs from B1TM in {sorted(differs)}')
        shutil.rmtree(here)

    out('\n==== the queue start string (the lead) ====')
    out(shlex.join(queue_launch(root, repo, include_s_k1, only_s_k1=only_s_k1)))

    out('\n==== negative control on the protocol check ====')
    pair = REST_PAIRS[0]
    files = input_files('S_P', pair, results)
    tokens = prepare_argv('S_P', pair, prepared/'gw', results)
    tokens[2] = str(copy/'scripts/run_stage_3d_e48_pg.py')
    tokens[tokens.index('--geometry-weight-levels')+1] = '1,1,0.1,0.03'
    scratch_link.unlink()
    scratch_link.symlink_to(files[0].parent)
    subprocess.run(tokens, cwd=scratch, env=env, capture_output=True, text=True, check=True)
    check = system.check_prepared(prepared/'gw', pair, 'SEED-S_P', files[1], files[2], b1_entry('S_P', results))
    out(f"configuration C's geometry weight against the arm: ok {check['ok']}: {check['problems']}")
    if check['ok']:
        problems.append('check_prepared took the wrong geometry weight')

    out("\n==== the reading on the real baseline (copies of B1TM as S_P of the 14 pairs: formats and pins only, not results) ====")
    rehearse_reading(scratch, out, results)

    if os.readlink(link) != link_before:
        problems.append("the frozen worktree's link changed")
    if root.exists() != existed:
        problems.append(f'{root} was created')
    out(f"\nthe frozen worktree's link unchanged: {link_before}; {root} {'exists' if existed else 'not created'}; no solver run; evaluate_run is not rehearsed here (the same call ran 10 times in seed_v1); the scratch directory goes when this ends")
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['--dry-run']:
        d = argparse.ArgumentParser()
        d.add_argument('--dry-run', action='store_true')
        d.add_argument('--scratch', default=None)
        switch = d.add_mutually_exclusive_group()
        switch.add_argument('--include-s-k1', action='store_true')
        switch.add_argument('--only-s-k1', action='store_true')
        parsed = d.parse_args(argv)
        return dry_run(scratch=parsed.scratch, include_s_k1=parsed.include_s_k1, only_s_k1=parsed.only_s_k1)
    p = argparse.ArgumentParser(description='SEED19: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(default_root()))
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--min-host-gib', type=float, default=DEFAULT_MIN_HOST_GIB)
    switch = q.add_mutually_exclusive_group()
    switch.add_argument('--include-s-k1', action='store_true')
    switch.add_argument('--only-s-k1', action='store_true', help='the fallback of plan revision 1: S_K1 alone, without the entries of S_P')
    sub.add_parser('decide').add_argument('--root', default=str(default_root()))
    e = sub.add_parser('evaluate-one')
    e.add_argument('run')
    e.add_argument('--arm', required=True)
    e.add_argument('--pair', required=True)
    e.add_argument('--run-id', required=True)
    o = p.parse_args(argv)
    if o.command == 'queue':
        return queue(o)
    if o.command == 'decide':
        problem = root_problem(o.root)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        import decide_seed19
        decide_seed19.write_analysis(o.root)
        return 0
    if Path(o.run).resolve() != run_dir(default_root(), o.arm, o.pair).resolve():     # it writes <RUN>/metrics.json: only this batch's run of this arm and pair (review important 1)
        print(f'{o.run}: evaluate-one writes only to {run_dir(default_root(), o.arm, o.pair)}; nothing read, nothing moved', file=sys.stderr)
        return 2
    return evaluate_one(o)


if __name__ == '__main__':
    sys.exit(main())
