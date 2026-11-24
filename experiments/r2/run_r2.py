"""R2 step 2 and 3a (runs/20261110-r2/plan.md sections 3, 4): the recipe r2s (bayesreg/seg/unet/small.py, profile r2_small; one arm, the lead's ruling of 11-10 ~10:42 PM CST), two members (foldA_s2,
foldB_s2), trained on the training set only, then the one segmentation-level reading of R2S against UF4 (eval_r2.py, decide_r2.py) and STOP.  Everything real is started by the lead with the printed strings; this file starts no unit.
The steps of UF's driver (experiments/uf/run_uf.py: run_command, run_step, the hold file, the timeouts that kill the process group) are reused, not copied; the gate is UL's (experiments/ul/run_ul.py) plus
the brief's floor of 40 GiB free on C:.

    python -B experiments/r2/run_r2.py --dry-run [--scratch DIR]
        the start strings, the chain (printed), UF's measurement and the estimate, the gate now (read only), the lock and the targets (existence), and the whole chain on synthetic subjects on the CPU.
        No GPU, no unit; no real label, image or splits/ file is read; nothing written outside the scratch directory.
    python -B experiments/r2/run_r2.py prepare --repo <the clean checkout the smoke was made from>        the lead, foreground, after targets.json and the GPU smoke: first the smoke (smoke_gate:
        <SMOKE>/smoke_r2.json reads 'pass', its targets_sha256 is targets.json's now, its code record is <SMOKE>/snapshot's, and that snapshot is the commit and the files of this one; a refusal
        otherwise, before any snapshot: only <OUT> and its run_r2.log line are made), then the snapshot <OUT>/snapshot, cli check-inputs --profile r2_small from it, members_plan.json.
        The snapshot carries no sealed list: in it the guard reads them from SNAPSHOT.json's source_repo, the --repo checkout (contract C23.7-4 revision 6n, D95 (b)); --repo needs the four
        lists (a full checkout) and its data link (the allowlist sources, pins.source_root_for).
    python -B experiments/r2/run_r2.py print-launch
    python -B <snapshot>/experiments/r2/run_r2.py chain            in the unit tag-r2-chain:
          smoke_gate once more before the first step (the diff review 2's minor 6)
          control_check (eval_r2.py --control-only: UF4 recomputed against eval_uf.json, no lock)
          for r2s, foldA_s2 then foldB_s2: train (cli train --profile r2_small, the recipe's --set) -> check (in this process: the record against the plan) -> calibrate
          check_members_r2s (cli check-members) -> eval_r2 (the one read of the development truth) -> decide_r2 -> STOP
        Every step once; a step that started and did not finish stops the chain and is kept; <OUT>/hold stops it before the next step; nothing is rerun (hard rule 4).
    python -B experiments/r2/run_r2.py status
Exit codes: 0 done, 2 usage, 8 a stop.
"""
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(HERE), str(HERE.parent/'uf'), str(HERE.parent/'ul'), str(HERE.parent/'unv2')]

import r2_common as rc  # noqa: E402
import r2_lock  # noqa: E402
import run_uf as ufrun  # noqa: E402  (its steps: run_command, run_step, check_hold, Stop, the log)
from bayesreg.seg.unet.snapshot import CODE_ROOT, SNAPSHOT_FILE, SNAPSHOT_PATHS_V3, CodeNotClean, make_snapshot, require_snapshot, verify_snapshot  # noqa: E402

UNIT = 'tag-r2-chain'
SNAPSHOT_PATHS_R2 = SNAPSHOT_PATHS_V3+('experiments/uf', 'experiments/nn', 'experiments/r2', rc.PLAN_REL)     # the plan in the snapshot: members_plan.json and decide_r2 record its sha256 (review m8)
THREADS = 4
MEMORY_MAX, MEMORY_SWAP_MAX = '8G', '2G'
WAIT_MINUTES = 30
TIMEOUTS = dict(control_check=3600, **{f'train_{r}': int(rc.RECIPES[r]['max_wall_seconds'])+1800 for r in rc.RECIPE_ORDER}, calibrate=1800, check_members=1800, eval_r2=7200, decide_r2=600)
GPU_KINDS = ('control_check', *(f'train_{r}' for r in rc.RECIPE_ORDER), 'calibrate', 'eval_r2')
Stop = ufrun.Stop
log = ufrun.log


def snapshot_path_problems(*, paths=SNAPSHOT_PATHS_R2, log=lambda text: None):
    """The static check of the snapshot path table (follow-up 3, item 3, as contract C23.7-4 revision 6n rule 4 reads it): no path of SNAPSHOT_PATHS_R2 is, or holds, one of the
    sealed lists (bayesreg.io.sealed.SEALED_LISTS, their paths taken from the constant: no list is opened).  In the snapshot the guard reads them from SNAPSHOT.json's source_repo."""
    from bayesreg.io.sealed import SEALED_LISTS
    carried = sorted(spec['path'] for spec in SEALED_LISTS if any(spec['path'] == p or spec['path'].startswith(p.rstrip('/')+'/') for p in paths))
    log(f"the snapshot path table carries no sealed list (C23.7-4 revision 6n rule 4; the guard reads them from SNAPSHOT.json's source_repo): {'yes' if not carried else 'NO: '+', '.join(carried)}")
    return [f'the snapshot path table carries the sealed list {p}' for p in carried]


def set_run_log(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    ufrun.LOG_FILE = Path(out)/'run_r2.log'


def runtime_max_seconds():
    """Every step at its limit and every gate closed to its limit: a safety valve, not a budget."""
    steps = TIMEOUTS['control_check']+sum(TIMEOUTS[f'train_{r}']+TIMEOUTS['calibrate'] for r in rc.RECIPE_ORDER for _ in rc.MEMBERS)+len(rc.RECIPE_ORDER)*TIMEOUTS['check_members']+TIMEOUTS['eval_r2']+TIMEOUTS['decide_r2']
    gated = 1+2*len(rc.RECIPE_ORDER)*len(rc.MEMBERS)+1
    return steps+gated*WAIT_MINUTES*60+1800


# ---- the gate ------------------------------------------------------------------------------------------------------------------------------------------
def c_floor_problems(disk=shutil.disk_usage):
    try:
        free = disk('/mnt/c').free/2**30
    except OSError as exc:
        return [f'C: query failed: {exc}']
    return [] if free >= rc.C_FLOOR_GIB else [f'/mnt/c free {free:.1f} GiB < {rc.C_FLOOR_GIB:.0f} GiB (the brief)']


def gate_problems(own_unit=UNIT, **kwargs):
    import run_ul
    return run_ul.gate_problems(own_unit, **kwargs)+c_floor_problems()


def launch_argv(*, out=rc.OUT, wait_minutes=WAIT_MINUTES):
    snapshot = Path(out)/'snapshot'
    argv = ['systemd-run', '--user', f'--unit={UNIT}', '--collect', f'--working-directory={snapshot}', f'--property=MemoryMax={MEMORY_MAX}', f'--property=MemorySwapMax={MEMORY_SWAP_MAX}', f'--property=RuntimeMaxSec={runtime_max_seconds()}',
            '--setenv=PYTHONDONTWRITEBYTECODE=1', *[f'--setenv={name}={THREADS}' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')],
            rc.PY, '-B', str(snapshot/'experiments/r2/run_r2.py'), 'chain', '--out', str(out), '--wait-minutes', str(wait_minutes)]
    text = ' '.join(argv)
    if 'pytest' in text or 'seq-vp-' in text:
        raise ValueError(f'a start string must not name the test runner or the seq-vp- prefix: {text}')
    return argv


# ---- the steps -----------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class Spec:
    out: Path
    cwd: Path
    python: str = rc.PY
    device: str = 'cuda'
    inputs: list = field(default_factory=list)              # options of cli after --profile (real: none, the profile's pins)
    eval_extra: list = field(default_factory=list)          # options of eval_r2.py (real: none)
    overrides: dict = None                                  # {recipe: tiny config} (the dry run and the tests); None: the recipes of r2_common
    targets_path: Path = rc.PP_ROOT/'targets.json'
    uf_root: Path = rc.UF_OUT
    uf_pinned: dict = field(default_factory=lambda: dict(rc.UF_MEMBER_SHA256))
    gate: bool = True
    wait_minutes: int = WAIT_MINUTES
    env: dict = None

    def recipe_config(self, recipe, targets):
        if self.overrides is None:
            return rc.train_config(recipe, targets)
        return dict(self.overrides[recipe], target_labels=[int(t) for t in targets], p_target=rc.P_TARGET, target_dice_weight=rc.TARGET_DICE_WEIGHT)


def sets(config):
    out = []
    for key, value in config.items():
        out += ['--set', f'{key}={json.dumps(value)}']
    return out


def chain_steps(spec, targets, targets_sha256):
    """(name, kind, argv or None) in order; kind a key of TIMEOUTS, or 'check' (in this process)."""
    out, script = Path(spec.out), Path(spec.cwd)/'experiments'/'r2'
    cli = [spec.python, '-B', '-m', 'bayesreg.seg.unet.cli']
    steps = [('control_check', 'control_check', [spec.python, '-B', str(script/'eval_r2.py'), '--control-only', '--out-dir', str(out/'control_check'), '--device', spec.device, *spec.eval_extra])]
    for recipe in rc.RECIPE_ORDER:
        for fold, seed in rc.MEMBERS:
            name = rc.member_name(fold, seed)
            common = ['--profile', rc.PROFILE, '--fold', fold, '--seed', str(seed), *spec.inputs, '--out', str(out/recipe)]
            steps += [(f'train_{recipe}_{name}', f'train_{recipe}', cli+['train', *common, '--device', spec.device, *sets(spec.recipe_config(recipe, targets))]), (f'check_{recipe}_{name}', 'check', None),
                      (f'calibrate_{recipe}_{name}', 'calibrate', cli+['calibrate', *common, '--device', spec.device])]
    steps += [(f'check_members_{recipe}', 'check_members', cli+['check-members', '--profile', rc.PROFILE, *spec.inputs, '--out', str(out/recipe)]) for recipe in rc.RECIPE_ORDER]
    steps += [('eval_r2', 'eval_r2', [spec.python, '-B', str(script/'eval_r2.py'), '--out-dir', str(out/'eval_r2'), '--targets', str(spec.targets_path), '--targets-sha256', targets_sha256, '--device', spec.device, *spec.eval_extra]),
              ('decide_r2', 'decide_r2', [spec.python, '-B', str(script/'decide_r2.py'), '--eval', str(out/'eval_r2'/'eval_r2.json'), '--out-dir', str(out/'analysis')])]
    return steps


def member_check(spec, recipe, name, targets):
    """check_<recipe>_<name>: the train record against the plan (eval_r2.member_record_problems: the recipe, the config of the recipe and of targets.json, the schedule to its end, the last step, the fold's
    subjects, those of UF's same-fold members).  <out>/<recipe>/<name>/r2_check.json; a problem stops the chain."""
    import eval_r2
    record = json.loads((Path(spec.out)/recipe/name/'train_record.json').read_text(encoding='utf-8'))
    namesakes = eval_r2.uf_namesakes(spec.uf_root, spec.uf_pinned)
    expected = eval_r2.planned_config(recipe, targets, spec.overrides)
    problems = eval_r2.member_record_problems(record, recipe=recipe, name=name, expected_config=expected, namesake=namesakes[name[4]])
    rc.write_json_once(Path(spec.out)/recipe/name/'r2_check.json', dict(schema='bayesreg.r2.member_check.v1', recipe=recipe, member=name, problems=problems, stopped=record.get('stopped'), iterations_run=record.get('iterations_run'),
                                                                          checkpoint_iteration=record.get('checkpoint_iteration'), targets=targets, utc=ufrun.utc()))
    if problems:
        log(f'{recipe}/{name}: not the plan\'s member: {problems}')
        return 8
    return 0


def read_targets(spec, plan):
    """(targets, sha256): targets.json, the file members_plan.json pinned; Stop otherwise."""
    path = Path(spec.targets_path)
    if not path.is_file():
        raise Stop(f'{path}: no such file; the pp diagnosis and its split come first')
    digest = rc.sha256_file(path)
    if plan is not None and digest != plan.get('targets_sha256'):
        raise Stop(f'{path}: sha256 {digest} is not the one members_plan.json pinned ({plan.get("targets_sha256")})')
    import eval_r2
    try:
        return eval_r2.read_targets(path), digest
    except ValueError as exc:
        raise Stop(str(exc)) from None


def members_plan(spec, split):
    """<out>/members_plan.json, once (the same content again, or Stop): the targets and their sha256, the two recipes' configs, the members and their fold's subjects from the folds file against UF's
    same-fold members (ids only)."""
    import eval_r2
    targets, digest = read_targets(spec, None)
    namesakes = eval_r2.uf_namesakes(spec.uf_root, spec.uf_pinned)
    problems = []
    members = {}
    for fold, seed in rc.MEMBERS:
        name = rc.member_name(fold, seed)
        same = split.subjects(fold, 'train') == namesakes[fold]['train'] and split.subjects(fold, 'calib') == namesakes[fold]['calib'] and split.label_contact(fold) == namesakes[fold]['contact']
        if not same:
            problems.append(f'{name}: the folds file gives other subjects than UF\'s fold {fold}')
        members[name] = dict(fold=fold, seed=seed, n_train=len(split.subjects(fold, 'train')), n_calib=len(split.subjects(fold, 'calib')), calib=split.subjects(fold, 'calib'), same_as_uf=same)
    if problems:
        raise Stop(f'the members do not match UF\'s folds: {problems}')
    document = dict(schema='bayesreg.r2.members_plan.v1', plan=rc.plan_record(Path(spec.cwd)), folds_sha256=split.sha256, targets=targets, targets_sha256=digest, members=members, recipes={r: spec.recipe_config(r, targets) for r in rc.RECIPE_ORDER},
                    order=[f'{r}/{rc.member_name(f, s)}' for r in rc.RECIPE_ORDER for f, s in rc.MEMBERS])
    path = Path(spec.out)/'members_plan.json'
    text = json.dumps(document, indent=1)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise Stop(f'{path} exists with other content; never overwritten')
    else:
        rc.write_text_once(path, text)
    return document


def chain_end(out, stop):
    out = Path(out)
    decision = rc.read_json(out/'analysis'/'decision_r2.json') or {}
    document = dict(schema='bayesreg.r2.queue_end.v1', stop=stop, finished=stop is None and bool(decision), reading=decision.get('reading'), reading_text=decision.get('reading_text'), primary_arm=decision.get('primary_arm'),
                    delta_seg35=decision.get('delta_seg35'), done=sorted(p.name[:-5] for p in (out/'chain_state').glob('*.done')) if (out/'chain_state').is_dir() else [], utc=ufrun.utc())
    path, k = out/'queue_end.json', 0
    while path.exists():
        k += 1
        path = out/f"queue_end_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{k}.json"
    rc.write_json_once(path, document)
    return path


def run_chain(spec, plan):
    out = Path(spec.out)
    try:
        run_steps(spec, plan)
    except Stop as stop:
        chain_end(out, str(stop))
        raise
    chain_end(out, None)


def run_steps(spec, plan):
    out = Path(spec.out)
    targets, digest = read_targets(spec, plan)
    state, chain_log = out/'chain_state', out/'chain.log'
    for name, kind, argv in chain_steps(spec, targets, digest):
        if (state/f'{name}.done').exists():
            log(f'skip {name} (done)')
            continue
        ufrun.check_hold(out)
        if spec.gate and kind in GPU_KINDS:
            ufrun.wait_for_gate(spec.wait_minutes, problems=gate_problems)
            ufrun.check_hold(out)
        if kind == 'check':
            recipe, member = name.removeprefix('check_').split('_', 1)
            action, label = (lambda recipe=recipe, member=member: member_check(spec, recipe, member, targets)), 'in this process: the train record against the plan'
        else:
            def action(argv=argv, kind=kind, name=name):
                code, timed_out = ufrun.run_command(argv, cwd=spec.cwd, log_path=chain_log, timeout=TIMEOUTS[kind], env=spec.env)
                if timed_out:
                    log(f'{name}: over its time limit of {TIMEOUTS[kind]} s; the process group was killed')
                    return 124
                return code
            label = shlex.join(argv)
        ufrun.run_step(name, action, state=state, log_path=chain_log, label=label)


# ---- the estimate from UF's members ---------------------------------------------------------------------------------------------------------------------
def uf_measurement(uf_root=rc.UF_OUT):
    """UF's four members' train records (their training subjects' calib Dice only): wall seconds per step, GPU and host peaks."""
    rows = []
    for name in rc.UF_ORDER:
        record = json.loads((Path(uf_root)/name/'train_record.json').read_text(encoding='utf-8'))
        rows.append(dict(member=name, wall_seconds=record['wall_seconds'], iterations=record['iterations_run'], per_step=record['wall_seconds']/record['iterations_run'], gpu_peak_allocated_gib=(record.get('memory') or {}).get('gpu_peak_allocated_gib'),
                         host_peak_rss_gib=(record.get('memory') or {}).get('host_peak_rss_gib')))
    return rows


def estimate(measured, *, target_overhead=.05):
    """〔估算〕per member: UF's largest wall per step x the recipe's steps x (1 + the target term's overhead, 〔推断〕5 %)."""
    per_step = max(r['per_step'] for r in measured)*(1+target_overhead)
    members = {r: rc.RECIPES[r]['max_iters']*per_step/3600 for r in rc.RECIPE_ORDER}
    rest = dict(control_check=10/60, calibrations=len(rc.RECIPE_ORDER)*len(rc.MEMBERS)*3/60, check_members=len(rc.RECIPE_ORDER)/60, eval_r2=20/60, decide=1/60)     # 〔估算〕eval_r2: 6 members' forward and 4 arms' metrics a subject
    total = sum(len(rc.MEMBERS)*h for h in members.values())+sum(rest.values())
    return dict(per_step_seconds=per_step, per_member_hours=members, training_hours=sum(len(rc.MEMBERS)*h for h in members.values()), rest_hours=sum(rest.values()), total_hours=total,
                wall_stop_over_estimate={r: rc.RECIPES[r]['max_wall_seconds']/3600/members[r] for r in rc.RECIPE_ORDER})


# ---- prepare, chain, status ----------------------------------------------------------------------------------------------------------------------------
def real_split():
    from bayesreg.seg.unet.data import load_full_folds
    from bayesreg.seg.unet.pins import DEFAULT_FULL_FOLDS, PINNED_FULL_FOLDS_SHA256
    split = load_full_folds(DEFAULT_FULL_FOLDS)
    if split.sha256 != PINNED_FULL_FOLDS_SHA256:
        raise Stop(f'{DEFAULT_FULL_FOLDS}: not the pinned folds file')
    return split


def real_spec(out, wait_minutes=WAIT_MINUTES):
    return Spec(out=Path(out), cwd=CODE_ROOT, wait_minutes=wait_minutes)


IDENTITY_KEYS = ('root', 'commit', 'snapshot_json_sha256', 'files_sha256')


def snapshot_identity(root):
    """The identity of a verified snapshot (the diff review 2's minors 1(b), 3): its root (resolved), verify_snapshot's commit and snapshot_json_sha256, and the sha256 of SNAPSHOT.json's
    files dict (canonical JSON: sorted keys) with its length.  Whatever verify_snapshot or a broken SNAPSHOT.json raises is the caller's to turn into a problem."""
    root = Path(root)
    record = verify_snapshot(root)
    files = json.loads((root/SNAPSHOT_FILE).read_text(encoding='utf-8'))['files']
    if not isinstance(files, dict):
        raise ValueError(f'{root/SNAPSHOT_FILE}: files is not a dict')
    return dict(root=str(root.resolve()), commit=record['commit'], snapshot_json_sha256=record['snapshot_json_sha256'],
                files_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode('utf-8')).hexdigest(), n_files=len(files))


def smoke_gate(snapshot, repo, *, smoke=rc.SMOKE, targets_path=rc.PP_ROOT/'targets.json'):
    """(problems, the smoke snapshot's identity or None): the GPU smoke before the training (the diff review's new minor 1, the diff review 2's minors 1(b), 2, 3; the coordinator, 11-11).
    <smoke>/smoke_r2.json there, a JSON object, with reading 'pass'; its targets_sha256 the sha256 of targets.json now; its code (written by the smoke unit from require_snapshot) the identity of
    <smoke>/snapshot (verified file by file); the training snapshot at the smoke's commit -- the existing one's, with the same files dict, or else the commit `repo` is at now, which make_snapshot
    takes (prepare compares the files once it is made).  A broken input is a problem, not an exception.  Reads smoke_r2.json, targets.json, the two snapshots' SNAPSHOT.json and files, git rev-parse;
    no label, no image."""
    from bayesreg.seg.unet.snapshot import _git
    smoke = Path(smoke)
    try:
        document = rc.read_json(smoke/'smoke_r2.json')
    except (OSError, ValueError) as exc:
        return [f'{smoke}/smoke_r2.json cannot be read: {type(exc).__name__}: {exc}'], None
    if document is None:
        return [f'{smoke}/smoke_r2.json: no such file; the GPU smoke comes before the training'], None
    if not isinstance(document, dict):
        return [f'{smoke}/smoke_r2.json is not a JSON object ({type(document).__name__})'], None
    problems = []
    if document.get('reading') != 'pass':
        problems.append(f"the smoke's reading is {document.get('reading')!r}, not 'pass': {document.get('why')}")
    digest = rc.sha256_file(targets_path) if Path(targets_path).is_file() else None
    if digest is None or document.get('targets_sha256') != digest:
        problems.append(f"the smoke's targets_sha256 {document.get('targets_sha256')} is not that of {targets_path} now ({digest})")
    try:
        mine = snapshot_identity(smoke/'snapshot')
    except Exception as exc:                                    # does not verify, cannot be read, no commit: a problem (minor 2)
        return problems+[f"the smoke's snapshot {smoke/'snapshot'}: {type(exc).__name__}: {exc}"], None
    recorded = document.get('code')
    if not isinstance(recorded, dict) or any(recorded.get(k) != mine[k] for k in IDENTITY_KEYS):
        problems.append(f"smoke_r2.json's code {recorded if not isinstance(recorded, dict) else {k: recorded.get(k) for k in IDENTITY_KEYS}} is not the snapshot {smoke/'snapshot'} "
                        f"({ {k: mine[k] for k in IDENTITY_KEYS} }): the smoke did not run from it")
    try:
        if Path(snapshot).exists():
            theirs = snapshot_identity(snapshot)
            training = theirs['commit']
            if theirs['files_sha256'] != mine['files_sha256']:
                problems.append(f"the training snapshot {snapshot} holds other files than the smoke's (files_sha256 {theirs['files_sha256']} against {mine['files_sha256']})")
        else:
            training = _git(repo, 'rev-parse', '--verify', 'HEAD^{commit}')
    except Exception as exc:
        return problems+[f'the training snapshot or its commit: {type(exc).__name__}: {exc}'], mine
    if training != mine['commit']:
        problems.append(f"the smoke ran on commit {mine['commit']}, the training snapshot is (or would be made) at {training}: smoke the training's commit")
    return problems, mine


def prepare(out, repo, *, smoke=rc.SMOKE, targets_path=rc.PP_ROOT/'targets.json'):
    import subprocess
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    set_run_log(out)
    snapshot = out/'snapshot'
    problems, smoke_code = smoke_gate(snapshot, repo, smoke=smoke, targets_path=targets_path)
    if problems:
        raise Stop('prepare refused before the snapshot (the GPU smoke): '+'; '.join(problems))
    log(f"the smoke: {Path(smoke)/'smoke_r2.json'} reads pass, its targets_sha256 targets.json's, its code the snapshot {smoke_code['root']} at {smoke_code['commit']}")
    if snapshot.exists():
        record = verify_snapshot(snapshot)
        log(f"reusing snapshot {snapshot} (commit {record['commit']})")
    else:
        record = make_snapshot(repo, snapshot, paths=SNAPSHOT_PATHS_R2)
        log(f"snapshot {snapshot} made from commit {record['commit']}")
    made = snapshot_identity(snapshot)
    if (made['commit'], made['files_sha256']) != (smoke_code['commit'], smoke_code['files_sha256']):        # the checkout moved between the check and the snapshot
        raise Stop(f"the training snapshot {snapshot} is at {made['commit']} (files_sha256 {made['files_sha256']}), the smoke's at {smoke_code['commit']} ({smoke_code['files_sha256']}): "
                   'kept, not used; the lead decides')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    for argv in ([rc.PY, '-B', '-m', 'bayesreg.seg.unet.cli', 'check-inputs', '--profile', rc.PROFILE, '--require-snapshot'], [rc.PY, '-B', str(snapshot/'experiments/r2/run_r2.py'), 'members-plan', '--out', str(out)]):
        done = subprocess.run(argv, cwd=str(snapshot), env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise Stop(f'{shlex.join(argv)} failed: {done.stdout[-2000:]} {done.stderr[-2000:]}')
    log('pinned inputs checked, members_plan.json written: no label or image was read')
    return record


def cmd_chain(args):
    require_snapshot()
    set_run_log(args.out)
    problems, _ = smoke_gate(Path(args.out)/'snapshot', None, smoke=rc.SMOKE, targets_path=rc.PP_ROOT/'targets.json')      # the diff review 2's minor 6: the chain checks the smoke itself (its snapshot exists: no git)
    if problems:
        raise Stop('the chain refused before its first step (the GPU smoke): '+'; '.join(problems))
    spec = real_spec(args.out, args.wait_minutes)
    if not (Path(args.out)/'chain_state'/'eval_r2.done').exists():
        r2_lock.check()                         # not 15 hours of training for an evaluation the lock would refuse
    path = Path(args.out)/'members_plan.json'
    if not path.is_file():
        raise Stop(f'{path}: no such file; prepare comes first')
    plan = members_plan(spec, real_split())
    log(f"R2: targets {plan['targets']}; recipes {json.dumps(plan['recipes'])}")
    run_chain(spec, plan)
    log(f'finished: {args.out}/analysis/decision_r2.json. STOP (nothing after the reading)')


def cmd_status(args):
    out = Path(args.out)
    state = out/'chain_state'
    print('state:', ' '.join(sorted(p.name for p in state.iterdir())) if state.is_dir() else '(none)')
    print('hold:', 'YES' if (out/'hold').exists() else 'no')
    for path in sorted(out.glob('*/fold*/r2_check.json'))+[out/'control_check'/'eval_control.json', out/'analysis'/'decision_r2.json']:
        if path.is_file():
            document = json.loads(path.read_text(encoding='utf-8'))
            print(f'{path.relative_to(out)}:', json.dumps({k: document.get(k) for k in ('problems', 'stopped', 'iterations_run', 'control', 'reading_text', 'primary_arm', 'delta_seg35')}, ensure_ascii=False))
    print(f'read lock {r2_lock.DEV_READ_LOCK}:', json.dumps(r2_lock.describe(), ensure_ascii=False) if Path(r2_lock.DEV_READ_LOCK).exists() else '(none)')


# ---- the synthetic world (the dry run and the tests) ----------------------------------------------------------------------------------------------------
TINY_UF = dict(base_channels=4, depth=2, ds_levels=1, batch_size=2, patch_size=[16, 16, 16], max_iters=4, min_iters=4, patience=4, eval_every=2, warmup_iters=1, log_every=1, amp=False, flip_axis_check=False, checkpoint='last', max_wall_seconds=3600.)
TINY_R2 = {'r2s': dict(TINY_UF)}
SYNTHETIC_TARGETS = [4, 10, 18, 19, 23, 29, 34, 35]


def synthetic_uf_eval(path, *, uf_root, subjects, data_root, device='cpu'):
    """eval_uf.json's layout (what eval_r2 and r2_seg read) for the synthetic UF members: UF4 by ensemble_probabilities in UF's order, float16, argmax; each member alone (UF_A0 ...); summarized by eval_seg."""
    import numpy as np
    import torch
    import eval_nn
    import eval_seg
    import eval_u1
    from bayesreg.seg.unet.data import OasisFiles, normalize_image, read_image
    from bayesreg.seg.unet.infer import ensemble_probabilities
    from bayesreg.seg.unet.train import predict_logits
    lookup = eval_seg.load_seg4_lookup()
    members = {n: eval_u1.load_arm(Path(uf_root)/n, device) for n in rc.UF_ORDER}
    rows, singles, image_sha, truth_sha = {}, {a: {} for a in ('UF_A0', 'UF_A1', 'UF_B0', 'UF_B1')}, {}, {}
    for subject in subjects:
        values, meta = read_image(OasisFiles(Path(data_root)), subject)
        image_sha[subject] = meta['sha256']
        image, _ = normalize_image(values, percentile=members[rc.UF_ORDER[0]][1]['percentile'])
        logits = {}

        def gen():
            for n in rc.UF_ORDER:
                model, blob, _ = members[n]
                z = predict_logits(model, image, device=device, amp=blob['amp'], window=tuple(blob['window']), gaussian_sigma=blob['window_gaussian_sigma'])
                logits[n] = z.argmax(1)[0].to(torch.uint8).cpu().numpy()
                yield z, blob['temperature']
        top = ensemble_probabilities(gen()).cpu().numpy().astype(np.float16).argmax(axis=0).astype(np.uint8)
        true35, sha35 = eval_seg.read_truth(data_root, subject, 'seg35')
        true4, sha4 = eval_seg.read_truth(data_root, subject, 'seg4')
        truth_sha[subject] = dict(seg35=sha35, seg4=sha4)
        rows[subject] = eval_seg.no_nan(eval_nn.metrics_of(eval_seg.summarize_subject(top, true35, true4, lookup)))
        for n, arm in zip(rc.UF_ORDER, ('UF_A0', 'UF_A1', 'UF_B0', 'UF_B1')):
            singles[arm][subject] = eval_seg.no_nan(eval_nn.metrics_of(eval_seg.summarize_subject(logits[n], true35, true4, lookup)))
    document = dict(subjects=list(subjects), rows=dict(UF4=rows, **singles), members={n: dict(member_sha256=rc.sha256_file(Path(uf_root)/n/'member.pt')) for n in rc.UF_ORDER}, image_sha256=image_sha, truth_sha256=truth_sha, note='synthetic')
    Path(path).write_text(json.dumps(document, indent=1)+'\n', encoding='utf-8')
    return path


def build_synthetic(work, env, device='cpu'):
    """The synthetic world (CPU; ids OASIS_OAS1_09xx_MR1, a non-real data root): experiments/unv2/synthetic_tree.py's subjects and folds, four tiny v3_large UF members (seeds 0, 1; the last step),
    their eval_uf.json, a targets.json, and the Spec of the chain with every option pointing there.  Nothing of a real subject."""
    import types
    import synthetic_tree
    synthetic_tree.build(work)
    data, allow, source = work/'data', work/'dev_subjects.json', synthetic_tree.SOURCE
    inputs = ['--synthetic', '--folds', str(work/'folds_full_v2.json'), '--allowlist', str(allow), '--training-source', source, '--expect-counts', '5,1', '--data-root', str(data)]
    cli = [sys.executable, '-B', '-m', 'bayesreg.seg.unet.cli']
    setup_state, setup_log = work/'setup_state', work/'setup.log'

    def call(argv):
        return lambda: ufrun.run_command(argv, cwd=CODE_ROOT, log_path=setup_log, timeout=1800, env=env)[0]
    uf_root = work/'uf'
    for name in rc.UF_ORDER:
        fold, seed = name[4], name[-1]
        common = ['--profile', 'v3_large', '--fold', fold, '--seed', seed, *inputs, '--out', str(uf_root), '--device', device]
        ufrun.run_step(f'uf_train_{name}', call([*cli, 'train', *common, *sets(TINY_UF)]), state=setup_state, log_path=setup_log)
        ufrun.run_step(f'uf_calibrate_{name}', call([*cli, 'calibrate', *common]), state=setup_state, log_path=setup_log)
    import synthetic_tree as st
    synthetic_uf_eval(work/'eval_uf.json', uf_root=uf_root, subjects=st.DEV, data_root=data, device=device)
    targets = work/'targets.json'
    targets.write_text(json.dumps(dict(schema='bayesreg.r2.targets.v1', targets=SYNTHETIC_TARGETS, note='synthetic'))+'\n', encoding='utf-8')
    (work/'train_overrides.json').write_text(json.dumps(TINY_R2)+'\n', encoding='utf-8')
    eval_extra = ['--synthetic', '--uf-root', str(uf_root), '--member-root', str(work/'out'), '--uf-eval', str(work/'eval_uf.json'), '--train-overrides', str(work/'train_overrides.json'),
                  '--allowlist', str(allow), '--folds', str(work/'folds.json'), '--dev-source', 'synthetic_dev', '--data-root', str(data)]
    spec = Spec(out=work/'out', cwd=CODE_ROOT, python=sys.executable, device=device, inputs=inputs, eval_extra=eval_extra, overrides=TINY_R2, targets_path=targets, uf_root=uf_root, uf_pinned=None, gate=False, env=env)
    return types.SimpleNamespace(spec=spec, work=work, data=data, allowlist=allow, uf_root=uf_root, inputs=inputs, eval_extra=eval_extra, targets=targets)


def dry_run(scratch=None, *, synthetic=True):
    work = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix='r2_dryrun_'))
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    problems = []
    print("== the lead's start strings (printed, not run)")
    print('prepare:', shlex.join([rc.PY, '-B', 'experiments/r2/run_r2.py', 'prepare', '--repo', '<the clean checkout the smoke was made from>']))
    print(f"  prepare refuses before the snapshot unless {rc.SMOKE/'smoke_r2.json'} reads 'pass', its targets_sha256 is targets.json's, its code record is the smoke snapshot's and that snapshot "
          f"is the training snapshot's commit and files (smoke_gate; the chain checks it again); now: {'smoke_r2.json there' if (rc.SMOKE/'smoke_r2.json').is_file() else 'no smoke yet'}")
    problems += snapshot_path_problems(log=print)
    print('chain:', shlex.join(launch_argv()))
    print(f'RuntimeMaxSec {runtime_max_seconds()} s ({runtime_max_seconds()/3600:.1f} h): every step at its limit {json.dumps(TIMEOUTS)} and every gate closed to {WAIT_MINUTES} minutes; a safety valve, not a budget')
    print(f'\n== the chain of {UNIT} (printed, not run; the targets shown are a placeholder until targets.json exists); hold file {rc.OUT/"hold"}')
    placeholder = list(rc.SMALL_SIX)
    for name, kind, argv in chain_steps(Spec(out=rc.OUT, cwd=rc.OUT/'snapshot'), sorted(placeholder), '<the sha256 of targets.json>'):
        gated = 'gate, ' if kind in GPU_KINDS else ''
        print(f"{name} [{gated}hold, limit {TIMEOUTS.get(kind, 'in-process')} s]: {shlex.join(argv) if argv else 'in this process: eval_r2.member_record_problems'}")
    print('\n== UF\'s members (their train records; the training subjects\' calib Dice only) and the estimate')
    try:
        measured = uf_measurement()
        print(json.dumps(dict(measured=measured, estimate=estimate(measured)), indent=1))
        import eval_r2
        namesakes = eval_r2.uf_namesakes(rc.UF_OUT, rc.UF_MEMBER_SHA256)
        print('UF\'s fold subjects (ids from their train records, counted):', json.dumps({f: dict(n_train=len(v['train']), n_calib=len(v['calib']), n_contact=len(v['contact'])) for f, v in namesakes.items()}))
        if any(len(v['train']) != 98 or len(v['calib']) != 8 or len(v['contact']) != 106 for v in namesakes.values()):
            problems.append('UF\'s members are not 98 + 8 of 106')
    except (OSError, ValueError, KeyError) as exc:
        problems.append(f'UF\'s members: {exc}')
    print('\n== the gate on this machine now (read only; it decides nothing here)')
    print(json.dumps(gate_problems(), indent=1))
    print('\n== the real chain is refused here: it needs a verified snapshot')
    try:
        require_snapshot()
        problems.append('the chain was not refused outside a snapshot')
    except CodeNotClean as exc:
        print(f'refused: {str(exc)[:160]}')
    print(f"\n== the read lock {r2_lock.DEV_READ_LOCK}: {'EXISTS' if Path(r2_lock.DEV_READ_LOCK).exists() else 'none'}; targets.json {rc.PP_ROOT/'targets.json'}: {'there' if (rc.PP_ROOT/'targets.json').is_file() else 'not yet (the pp diagnosis and its split come first)'}")
    if Path(r2_lock.DEV_READ_LOCK).exists():
        problems.append('the read lock exists')
    print('\n(not done here, by the brief: the sha256 of the folds file and the pinned inputs -- prepare and the chain do them under the lead)')
    if synthetic:
        print('\n== the chain on synthetic subjects, CPU')
        problems += synthetic_chain(work, env)
    print(f"\n== dry run problems: {'; '.join(problems) if problems else 'none'}; work {work}")
    return 0 if not problems else 8


def synthetic_chain(work, env, device='cpu'):
    import decide_r2
    from bayesreg.seg.unet.data import load_full_folds
    problems = []
    lock = r2_lock.DEV_READ_LOCK
    r2_lock.DEV_READ_LOCK = work/'never_taken.lock'          # the synthetic evaluation does not take it; pointed away all the same
    try:
        world = build_synthetic(work, env, device)
        spec, out = world.spec, world.spec.out
        set_run_log(out)
        plan = members_plan(spec, load_full_folds(work/'folds_full_v2.json'))
        print('members plan (synthetic):', json.dumps({k: {kk: v[kk] for kk in ('fold', 'seed', 'n_train', 'n_calib', 'same_as_uf')} for k, v in plan['members'].items()}), 'targets', plan['targets'])
        (out/'hold').write_text('dry run: the hold file\n')
        try:
            run_chain(spec, plan)
            problems.append('the hold file did not stop the chain')
        except Stop as exc:
            print(f'hold: {exc}')
        (out/'hold').unlink()
        run_chain(spec, plan)
        run_chain(spec, plan)                                   # a second pass skips everything
        checks = {str(p.relative_to(out)): json.loads(p.read_text())['problems'] for p in sorted(out.glob('*/fold*/r2_check.json'))}
        evaluation = json.loads((out/'eval_r2'/'eval_r2.json').read_text(encoding='utf-8'))
        control = json.loads((out/'control_check'/'eval_control.json').read_text(encoding='utf-8'))
        result = json.loads((out/'analysis'/'decision_r2.json').read_text(encoding='utf-8'))
        order = [line.split()[2].rstrip(':') for line in (out/'chain.log').read_text(encoding='utf-8').splitlines() if len(line.split()) > 2 and line.split()[1] == 'start']
        print('member checks:', json.dumps(checks))
        print('control (chain\'s first step):', json.dumps(control['control']))
        print('control (evaluation):', json.dumps(evaluation['control']))
        print('order of the steps:', ' '.join(order))
        print('decision:', json.dumps({k: result.get(k) for k in ('reading', 'reading_text', 'why')}, ensure_ascii=False))
        if order[:2] != ['control_check', 'train_r2s_foldA_s2'] or order[-2:] != ['eval_r2', 'decide_r2'] or any(checks.values()) or len(checks) != len(rc.RECIPE_ORDER)*len(rc.MEMBERS):
            problems.append('the synthetic chain: order or member checks')
        if not control['control']['passed'] or not evaluation['control']['passed'] or result['reading'] != decide_r2.UNDECIDABLE or result['why'] != ['2 名受试者，不是 20']:
            problems.append('the synthetic chain: control or reading')
    finally:
        r2_lock.DEV_READ_LOCK = lock
    return problems


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--scratch', type=Path, default=None)
    parser.add_argument('--no-synthetic', action='store_true', help='--dry-run without the synthetic chain')
    sub = parser.add_subparsers(dest='command')
    p = sub.add_parser('prepare')
    p.add_argument('--repo', type=Path, default=rc.MAIN_REPO)
    p.add_argument('--out', type=Path, default=rc.OUT)
    p = sub.add_parser('members-plan', help='(prepare runs it from the snapshot) members_plan.json')
    p.add_argument('--out', type=Path, default=rc.OUT)
    p = sub.add_parser('print-launch')
    p.add_argument('--out', type=Path, default=rc.OUT)
    p.add_argument('--wait-minutes', type=int, default=WAIT_MINUTES)
    p = sub.add_parser('chain')
    p.add_argument('--out', type=Path, default=rc.OUT)
    p.add_argument('--wait-minutes', type=int, default=WAIT_MINUTES)
    sub.add_parser('status').add_argument('--out', type=Path, default=rc.OUT)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.dry_run:
            return dry_run(args.scratch, synthetic=not args.no_synthetic)
        if args.command == 'prepare':
            prepare(args.out, args.repo)
        elif args.command == 'members-plan':
            require_snapshot()
            print(json.dumps(members_plan(real_spec(args.out), real_split())['members'], indent=1))
        elif args.command == 'print-launch':
            print(shlex.join(launch_argv(out=args.out, wait_minutes=args.wait_minutes)))
        elif args.command == 'chain':
            cmd_chain(args)
        elif args.command == 'status':
            cmd_status(args)
        else:
            build_parser().print_help()
            return 2
    except (Stop, CodeNotClean, r2_lock.DevReadLocked) as exc:
        log(f'STOP: {exc}')
        return 8
    return 0


if __name__ == '__main__':
    sys.exit(main())
