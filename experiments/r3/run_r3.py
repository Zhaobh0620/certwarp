"""R3a's driver (runs/20261111-r3/plan.md): L and N trained on the training set only, the composition chosen on the training set's calibration subjects (out of fold), STOP; then, in another unit
released by the lead, the one segmentation-level reading on the development set, STOP.  Everything real is started by the lead with the printed strings; this file starts no unit.
UF's steps are reused (experiments/uf/run_uf.py: run_command, run_step, the hold file, the time limits that kill the process group), the gate is R2's (UL's plus the floor of 40 GiB free on C:),
R2's snapshot identity (run_r2.snapshot_identity), NN's nnU-Net tools (experiments/nn) through nn_r3.py.

The lead's order (plan.md section 7; the lead's ruling on the review's I1, 11-11 ~3:43 PM CST: R3a and R3b merged, then R3b's calibration diagnosis, then the switch, the smoke, prepare, the chain):
    run_nn.py prepare --out <NN_ROOT>; run_nn.py preprocess --out <NN_ROOT>       (NN's own commands, a clean checkout: raw links, the training labels checked, plan_and_preprocess, gt links)
    python -B experiments/r3/run_r3.py nn-splits                                  (splits_final.json from folds_full_v2.json, the plans against NN's; ids and plans only)
    python -B experiments/r3/run_r3.py decode-switch --off --words TEXT | --on --words TEXT     (after R3b's diagnosis, from the checkout of the smoke's commit; once: --on reads R3b's
                                                                                   decision through its reader, require_pass=True, and pins its sha256 and decode.py's)
    smoke_r3.py prepare --repo <clean checkout>; the unit tag-r3-smoke              (L 400 steps, N 4 epochs and one real nnU-Net inference, the synthetic chains on the GPU)
    python -B experiments/r3/run_r3.py prepare --repo <the smoke's checkout>     (the smoke's reading and code, the snapshot, check-inputs, members_plan.json)
    the unit tag-r3-chain (print-launch):
          control_check (eval_r3.py --control-only: UF4 against eval_uf.json, no lock)
          L: for foldA_s3, foldB_s3: train (cli --profile v3_large, 80000 steps) -> check (in this process) -> calibrate;  check_members_r3l (cli check-members)
          N: for nnA_s0, nnA_s1, nnB_s2, nnB_s3: nn_train (run_r3.py nn-train) -> nn_check (in this process)
          nn_infer_calibA, nn_infer_calibB (run_r3.py nn-infer: each fold's two models on its eight calibration subjects)
          compose (compose_r3.py: the training set's calibration truth, once) -> STOP
    the lead reads composition.json; then, released separately, the unit tag-r3-dev (print-launch --dev):
          dev_plan (in this process: composition.json pinned; no change -> STOP) -> nn_infer_dev (only with N) -> eval_r3 (the development truth, once) -> decide_r3 -> STOP
Every step once; a step that started and did not finish stops the chain and is kept; <OUT>/hold stops it before the next step; nothing is rerun (hard rule 4).
    python -B experiments/r3/run_r3.py --dry-run [--scratch DIR]       start strings, steps, estimates (read only), the gate now, and the whole chain on synthetic subjects on the CPU
    python -B experiments/r3/run_r3.py status
Exit codes: 0 done, 2 usage, 8 a stop.
"""
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(HERE), str(HERE.parent/'r2'), str(HERE.parent/'uf'), str(HERE.parent/'ul'), str(HERE.parent/'unv2'), str(HERE.parent/'nn'), str(HERE.parent/'fl'), str(HERE.parent/'ef'), str(HERE.parent/'mc0')]

import r3_common as r3c  # noqa: E402
import r3_lock  # noqa: E402
import run_r2  # noqa: E402
import run_uf as ufrun  # noqa: E402
from bayesreg.seg.unet.snapshot import CODE_ROOT, SNAPSHOT_PATHS_V3, CodeNotClean, code_record, make_snapshot, require_snapshot, verify_snapshot  # noqa: E402

UNIT, DEV_UNIT = 'tag-r3-chain', 'tag-r3-dev'
SNAPSHOT_PATHS_R3 = SNAPSHOT_PATHS_V3+('experiments/uf', 'experiments/nn', 'experiments/r2', 'experiments/r3', 'experiments/fl/nx_seg.py', 'experiments/ef/fast_channels.py', 'experiments/mc0/mc0_core.py',
                                      'experiments/mc0/mc0p_core.py', r3c.PLAN_REL)
THREADS = 4
MEMORY_MAX, MEMORY_SWAP_MAX = '12G', '2G'       # NN's units had 12G (six augmentation processes); the U-Net training measured about 5 GiB
WAIT_MINUTES = 30
TIMEOUTS = dict(control_check=3600, train_l=int(r3c.L_WALL_SECONDS)+1800, calibrate=1800, check_members=1800, nn_train=r3c.NN_WALL_SECONDS+1800, nn_infer=7200, compose=7200, eval_r3=7200, decide_r3=600)
GPU_KINDS = ('control_check', 'train_l', 'calibrate', 'nn_train', 'nn_infer', 'compose', 'eval_r3')
Stop = ufrun.Stop
log = ufrun.log


def set_run_log(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    ufrun.LOG_FILE = Path(out)/'run_r3.log'


def gate_problems(own_unit=UNIT, **kwargs):
    return run_r2.gate_problems(own_unit, **kwargs)


def snapshot_path_problems(*, log=lambda text: None):
    return run_r2.snapshot_path_problems(paths=SNAPSHOT_PATHS_R3, log=log)


# ---- the steps -----------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class Spec:
    out: Path
    cwd: Path
    python: str = r3c.PY
    device: str = 'cuda'
    inputs: list = field(default_factory=list)              # cli options after --profile (real: none, the profile's pins)
    l_overrides: dict = None                                # the synthetic world's --set values of L; None: r3c.l_train_config()
    run_extra: list = field(default_factory=list)           # options of run_r3.py nn-train / nn-infer (real: none)
    compose_extra: list = field(default_factory=list)
    eval_extra: list = field(default_factory=list)
    control_extra: list = field(default_factory=list)
    composition_path: Path = None                           # the development chain's composition.json (real: <out>/compose/composition.json)
    uf_root: Path = r3c.UF_OUT
    uf_pinned: dict = field(default_factory=lambda: dict(r3c.UF_MEMBER_SHA256))
    nn_root: Path = r3c.NN_ROOT
    gate: bool = True
    wait_minutes: int = WAIT_MINUTES
    env: dict = None
    synthetic: bool = False                                 # the synthetic world (the decoder's stub allowed there only)

    def l_config(self):
        return r3c.l_train_config() if self.l_overrides is None else dict(self.l_overrides)

    def composition(self):
        return Path(self.out)/'compose'/'composition.json' if self.composition_path is None else Path(self.composition_path)


def sets(config):
    return run_r2.sets(config)


def chain_steps(spec, splits_sha256):
    """(name, kind, argv or None) of the training chain, in order; kind a key of TIMEOUTS, or 'check_l' / 'check_nn' (in this process)."""
    out, script = Path(spec.out), Path(spec.cwd)/'experiments'/'r3'
    cli = [spec.python, '-B', '-m', 'bayesreg.seg.unet.cli']
    steps = [('control_check', 'control_check', [spec.python, '-B', str(script/'eval_r3.py'), '--control-only', '--out-dir', str(out/'control_check'), '--device', spec.device, *spec.control_extra])]
    for fold, seed in r3c.L_MEMBERS:
        name = r3c.member_name(fold, seed)
        common = ['--profile', r3c.L_PROFILE, '--fold', fold, '--seed', str(seed), *spec.inputs, '--out', str(out/r3c.L_FAMILY)]
        steps += [(f'train_r3l_{name}', 'train_l', cli+['train', *common, '--device', spec.device, *sets(spec.l_config())]), (f'check_r3l_{name}', 'check_l', None),
                  (f'calibrate_r3l_{name}', 'calibrate', cli+['calibrate', *common, '--device', spec.device])]
    steps.append(('check_members_r3l', 'check_members', cli+['check-members', '--profile', r3c.L_PROFILE, *spec.inputs, '--out', str(out/r3c.L_FAMILY)]))
    run = [spec.python, '-B', str(script/'run_r3.py')]
    for model in r3c.NN_NAMES:
        steps += [(f'nn_train_{model}', 'nn_train', run+['nn-train', '--model', model, '--out', str(out), '--splits-sha256', splits_sha256, *spec.run_extra]), (f'nn_check_{model}', 'check_nn', None)]
    for fold in r3c.FOLDS:
        steps.append((f'nn_infer_calib{fold}', 'nn_infer', run+['nn-infer', '--set', f'calib{fold}', '--out', str(out), '--splits-sha256', splits_sha256, *spec.run_extra]))
    steps.append(('compose', 'compose', [spec.python, '-B', str(script/'compose_r3.py'), '--out-dir', str(out/'compose'), '--splits-sha256', splits_sha256, '--device', spec.device, *spec.compose_extra]))
    return steps


def dev_steps(spec, composition_sha256, splits_sha256, with_nn):
    """The development chain after dev_plan: nn_infer_dev (only with N), eval_r3, decide_r3."""
    out, script = Path(spec.out), Path(spec.cwd)/'experiments'/'r3'
    steps = []
    if with_nn:
        steps.append(('nn_infer_dev', 'nn_infer', [spec.python, '-B', str(script/'run_r3.py'), 'nn-infer', '--set', 'dev', '--out', str(out), '--splits-sha256', splits_sha256, *spec.run_extra]))
    steps += [('eval_r3', 'eval_r3', [spec.python, '-B', str(script/'eval_r3.py'), '--out-dir', str(out/'eval_r3'), '--composition', str(spec.composition()), '--composition-sha256', composition_sha256,
                                      '--splits-sha256', splits_sha256, '--device', spec.device, *spec.eval_extra]),
              ('decide_r3', 'decide_r3', [spec.python, '-B', str(script/'decide_r3.py'), '--eval', str(out/'eval_r3'/'eval_r3.json'), '--out-dir', str(out/'analysis')])]
    return steps


def l_member_check(spec, name):
    """check_r3l_<name>: the train record against the plan (r3_core.l_record_problems against UF's same-fold member); <out>/r3l/<name>/r3_check.json; a problem stops the chain."""
    import eval_r2
    import r3_core
    record = json.loads((Path(spec.out)/r3c.L_FAMILY/name/'train_record.json').read_text(encoding='utf-8'))
    namesakes = eval_r2.uf_namesakes(spec.uf_root, spec.uf_pinned)
    problems = r3_core.l_record_problems(record, name, namesakes[name[4]], r3_core.l_expected_config(spec.uf_root, spec.l_overrides))
    r3c.write_json_once(Path(spec.out)/r3c.L_FAMILY/name/'r3_check.json', dict(schema='bayesreg.r3.l_check.v1', member=name, problems=problems, stopped=record.get('stopped'),
                                                                               iterations_run=record.get('iterations_run'), checkpoint_iteration=record.get('checkpoint_iteration'), utc=ufrun.utc()))
    if problems:
        log(f'r3l/{name}: not the plan\'s member: {problems}')
        return 8
    return 0


def nn_member_check(spec, model, splits_sha256):
    """nn_check_<model>: nn_r3.model_problems on <nn_root>/models/<model>/train_record.json; nn_check.json beside it; a problem stops the chain."""
    import nn_r3
    record = r3c.read_json(Path(spec.nn_root)/'models'/model/'train_record.json')
    problems = nn_r3.model_problems(record, model, splits_sha256=splits_sha256, root=spec.nn_root)
    r3c.write_json_once(Path(spec.nn_root)/'models'/model/'nn_check.json', dict(schema='bayesreg.r3.nn_check_model.v1', model=model, problems=problems, utc=ufrun.utc()))
    if problems:
        log(f'{model}: not the plan\'s model: {problems}')
        return 8
    return 0


def run_steps(spec, steps, *, splits_sha256, unit=UNIT):
    out = Path(spec.out)
    state, chain_log = out/'chain_state', out/'chain.log'
    for name, kind, argv in steps:
        if (state/f'{name}.done').exists():
            log(f'skip {name} (done)')
            continue
        ufrun.check_hold(out)
        if kind == 'compose':                                     # the switch before the step starts: a missing one stops the chain with nothing started, the lead sets it and starts again
            import r3_core
            try:
                r3_core.read_switch(out/r3c.DECODE_SWITCH, synthetic=spec.synthetic)
            except r3_core.Refused as exc:
                raise Stop(f'{exc}; set it (run_r3.py decode-switch), then start the chain again: the steps done are skipped') from None
        if spec.gate and kind in GPU_KINDS:
            ufrun.wait_for_gate(spec.wait_minutes, problems=lambda: gate_problems(unit))
            ufrun.check_hold(out)
        if kind == 'check_l':
            member = name.removeprefix('check_r3l_')
            action, label = (lambda member=member: l_member_check(spec, member)), 'in this process: the L record against the plan'
        elif kind == 'check_nn':
            model = name.removeprefix('nn_check_')
            action, label = (lambda model=model: nn_member_check(spec, model, splits_sha256)), 'in this process: the nnU-Net record against the plan'
        else:
            def action(argv=argv, kind=kind, name=name):
                code, timed_out = ufrun.run_command(argv, cwd=spec.cwd, log_path=chain_log, timeout=TIMEOUTS[kind], env=spec.env)
                if timed_out:
                    log(f'{name}: over its time limit of {TIMEOUTS[kind]} s; the process group was killed')
                    return 124
                return code
            label = shlex.join(argv)
        ufrun.run_step(name, action, state=state, log_path=chain_log, label=label)


def chain_end(out, stop, what):
    out = Path(out)
    document = dict(schema='bayesreg.r3.queue_end.v1', chain=what, stop=stop, done=sorted(p.name[:-5] for p in (out/'chain_state').glob('*.done')) if (out/'chain_state').is_dir() else [], utc=ufrun.utc())
    if what == 'train':
        composition = r3c.read_json(out/'compose'/'composition.json') or {}
        document.update(winner=composition.get('winner'), no_change=composition.get('no_change'))
    else:
        decision = r3c.read_json(out/'analysis'/'decision_r3.json') or {}
        document.update(reading=decision.get('reading'), reading_text=decision.get('reading_text'), delta_seg35=decision.get('delta_seg35'))
    path, k = out/f'queue_end_{what}.json', 0
    while path.exists():
        k += 1
        path = out/f"queue_end_{what}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{k}.json"
    r3c.write_json_once(path, document)
    return path


def run_chain(spec, plan):
    try:
        run_steps(spec, chain_steps(spec, plan['nn']['splits_sha256']), splits_sha256=plan['nn']['splits_sha256'])
    except Stop as stop:
        chain_end(spec.out, str(stop), 'train')
        raise
    chain_end(spec.out, None, 'train')


def dev_plan(spec, plan):
    """The development chain's pin: composition.json (sha256), its winner, whether it has N; <out>/dev_plan.json once (the same again, or Stop).  Stop when the winner is no change."""
    import eval_r3
    import r3_core
    path = spec.composition()
    try:
        document, digest = eval_r3.read_composition(path, real=not spec.synthetic)
    except (OSError, ValueError, r3_core.Refused) as exc:
        raise Stop(f'no development reading: {exc}') from None
    pin = dict(schema='bayesreg.r3.dev_plan.v1', composition=str(path), composition_sha256=digest, winner=document['winner'], delivered=document['delivered'], splits_sha256=plan['nn']['splits_sha256'])
    target = Path(spec.out)/'dev_plan.json'
    text = json.dumps(pin, indent=1, ensure_ascii=False)+'\n'
    if target.exists():
        if target.read_text(encoding='utf-8') != text:
            raise Stop(f'{target} exists with other content; never overwritten')
    else:
        r3c.write_text_once(target, text)
    return pin


def run_dev_chain(spec, plan):
    try:
        pin = dev_plan(spec, plan)
        run_steps(spec, dev_steps(spec, pin['composition_sha256'], pin['splits_sha256'], bool(pin['delivered']['nn'])), splits_sha256=pin['splits_sha256'], unit=DEV_UNIT)
    except Stop as stop:
        chain_end(spec.out, str(stop), 'dev')
        raise
    chain_end(spec.out, None, 'dev')


# ---- members_plan.json ----------------------------------------------------------------------------------------------------------------------------------
def members_plan(spec, split, *, splits_sha256):
    """<out>/members_plan.json, once (the same content again, or Stop): L's members against UF's same-fold members (ids only), L's planned and expected configs, N's models and trainers, the
    pinned splits_final.json (it must be what folds_full_v2.json gives) and the plans (NN's); before any training, L's config as TrainConfigLarge records it against the expected one (the
    review's m4), and decode.py of this tree (sha256, stub or not: whether D can be switched on with this snapshot)."""
    import eval_r2
    import nn_r3
    import r3_core
    from dataclasses import asdict
    from bayesreg.seg.decode import decoder_spec
    from bayesreg.seg.unet.large import TrainConfigLarge
    namesakes = eval_r2.uf_namesakes(spec.uf_root, spec.uf_pinned)
    problems, members = [], {}
    for fold, seed in r3c.L_MEMBERS:
        name = r3c.member_name(fold, seed)
        same = split.subjects(fold, 'train') == namesakes[fold]['train'] and split.subjects(fold, 'calib') == namesakes[fold]['calib'] and split.label_contact(fold) == namesakes[fold]['contact']
        if not same:
            problems.append(f"{name}: the folds file gives other subjects than UF's fold {fold}")
        members[name] = dict(fold=fold, seed=seed, n_train=len(split.subjects(fold, 'train')), n_calib=len(split.subjects(fold, 'calib')), calib=split.subjects(fold, 'calib'), same_as_uf=same)
    path = nn_r3.splits_path(spec.nn_root)
    if not path.is_file() or r3c.sha256_file(path) != splits_sha256:
        problems.append(f'{path}: missing or not the sha256 {splits_sha256}')
    elif json.loads(path.read_text(encoding='utf-8')) != nn_r3.splits_document(split):
        problems.append(f"{path}: not what the folds file gives")
    expected = r3_core.l_expected_config(spec.uf_root, spec.l_overrides)
    recorded = json.loads(json.dumps(asdict(TrainConfigLarge(**spec.l_config()))))
    if recorded != expected:
        problems.append(f"L's config as cli would record it differs from UF's with L's four keys on {sorted(k for k in set(recorded) | set(expected) if recorded.get(k) != expected.get(k))}")
    if problems:
        raise Stop(f'members_plan refused: {problems}')
    module = Path(spec.cwd)/r3c.DECODE_PY_REL
    document = dict(schema='bayesreg.r3.members_plan.v1', plan=r3c.plan_record(Path(spec.cwd)), folds_sha256=split.sha256, l=dict(members=members, set=spec.l_config(), expected_config=expected),
                    decode=dict(path=r3c.DECODE_PY_REL, sha256=r3c.sha256_file(module) if module.is_file() else None, stub=bool(decoder_spec()['stub'])),
                    nn=dict(models={m: dict(fold=f, fold_index=r3c.NN_FOLD_INDEX[f], seed=s, trainer=r3c.nn_trainer(s)) for (f, s), m in zip(r3c.NN_MODELS, r3c.NN_NAMES)}, epochs=r3c.NN_EPOCHS,
                    splits_sha256=splits_sha256, root=str(spec.nn_root)), candidates=dict(bases=list(r3c.BASE_ORDER), decode_suffix=r3c.DECODE_SUFFIX))
    target = Path(spec.out)/'members_plan.json'
    text = json.dumps(document, indent=1, ensure_ascii=False)+'\n'
    if target.exists():
        if target.read_text(encoding='utf-8') != text:
            raise Stop(f'{target} exists with other content; never overwritten')
    else:
        r3c.write_text_once(target, text)
    return document


def read_members_plan(out):
    document = r3c.read_json(Path(out)/'members_plan.json')
    if document is None:
        raise Stop(f'{Path(out)/"members_plan.json"}: no such file; prepare comes first')
    return document


# ---- nnU-Net subcommands (run by the chain from the snapshot) --------------------------------------------------------------------------------------------
def nn_subjects(which, *, split, code, nn_root, allowlist=None, dev_source=None, training_source=None):
    """The subjects of a set, each admitted before any of its files is linked: calibA / calibB the fold's calibration subjects, B-line training subjects through LabelGuard.check and
    bayesreg.io.guard.refuse_sealed (the review's m3; no label is read here); dev the development subjects (eval_seg.admit_subjects; outside the nnU-Net label contact)."""
    if which.startswith('calib'):
        from bayesreg.io.guard import refuse_sealed
        from bayesreg.seg.unet.data import LabelGuard
        from bayesreg.seg.unet.pins import TRAINING_SOURCE
        guard = LabelGuard(split, allowlist, training_source=training_source or TRAINING_SOURCE)
        subjects = split.subjects(which[-1], 'calib')
        for subject in subjects:
            refuse_sealed(subject, 'run_r3.nn_subjects')
            guard.check(subject)
        return subjects, r3c.nn_models_of_fold(which[-1])
    import eval_seg
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.io.guard import SubjectSealed
    from bayesreg.seg.unet.pins import source_root_for
    subjects = eval_seg.admit_subjects(allowlist, dev_source, load_sealed_lists(source_root_for(code)), split.training_subjects)
    contact = (r3c.read_json(Path(nn_root)/'prepare_manifest.json') or {}).get('label_contact_subjects') or []
    if not contact or set(subjects) & set(contact):
        raise SubjectSealed('the nnU-Net label contact is missing or holds development subjects')
    return subjects, list(r3c.NN_NAMES)


def set_directory(out, which):
    return Path(out)/('nn_dev' if which == 'dev' else f'nn_calib/fold{which[-1]}')


def cmd_nn_train(args):
    import nn_r3
    code = require_snapshot() if not args.synthetic else code_record()
    root = Path(args.nn_root)
    set_run_log(args.out)
    if not nn_r3.splits_path(root).is_file() or r3c.sha256_file(nn_r3.splits_path(root)) != args.splits_sha256:
        raise Stop(f'{nn_r3.splits_path(root)}: missing or not the pinned splits_final.json')
    if args.synthetic:
        import r3_synthetic
        record = r3_synthetic.nn_train(args.model, root=root)
    else:
        import run_nn
        run_nn.require_preprocessed(root, code=code)
        env = nn_r3.nn_env(root, gpu=True)
        run_nn.env_record(root, env=env)
        record = nn_r3.train_model(args.model, root=root, env=env, resume=args.resume, code_commit=code.get('commit'), log=log)
    log(f"{args.model}: {record['epochs_logged']} epochs, {record['wall_seconds']/3600:.2f} h, checkpoint {record['final_checkpoint_sha256'][:12]}")


def cmd_nn_infer(args):
    import nn_r3
    code = require_snapshot() if not args.synthetic else code_record()
    root = Path(args.nn_root)
    set_run_log(args.out)
    split, allowlist = cli_split(args)
    subjects, models = nn_subjects(args.set, split=split, code=code, nn_root=root, allowlist=allowlist, dev_source=args.dev_source, training_source=args.training_source)
    records = {m: r3c.read_json(root/'models'/m/'train_record.json') for m in models}
    problems = [p for m in models for p in nn_r3.model_problems(records[m], m, splits_sha256=args.splits_sha256, root=root)]
    if problems:
        raise Stop(f'nn-infer {args.set}: {problems}')
    directory = set_directory(args.out, args.set)
    if args.synthetic:
        import r3_synthetic
        check = r3_synthetic.nn_infer(directory, subjects, models, root=root, data_root=args.data_root, records=records)
    else:
        check = nn_r3.infer_set(directory, subjects, models, root=root, env=nn_r3.nn_env(root, gpu=True), log=log)
    log(f"nn-infer {args.set}: {len(check['subjects'])} subjects, passed {check['passed']}")


def cli_split(args):
    """(split, allowlist): real -- the pinned folds_full_v2.json and allowlist; synthetic -- the world's."""
    from bayesreg.io.guard import load_dev_allowlist
    from bayesreg.seg.unet.data import load_full_folds
    if args.synthetic:
        from bayesreg.seg.unet.pins import check_synthetic_data_root
        check_synthetic_data_root(args.data_root)
        return load_full_folds(args.folds), load_dev_allowlist(args.allowlist)
    from bayesreg.seg.unet.pins import DEFAULT_ALLOWLIST, load_pinned_allowlist
    allowlist, _ = load_pinned_allowlist(DEFAULT_ALLOWLIST)
    return real_split(), allowlist


def real_split():
    return run_r2.real_split()


def cmd_nn_splits(args):
    """splits_final.json of <NN_ROOT> from the pinned folds (from a clean checkout; ids and plans only), and the plans against NN's."""
    import nn_r3
    from bayesreg.seg.unet.snapshot import require_clean_code
    require_clean_code()
    root = Path(args.nn_root)
    set_run_log(root)
    if not (root/'preprocess_record.json').is_file():
        raise Stop(f'{root}/preprocess_record.json: no such file; run_nn.py prepare and preprocess --out {root} come first')
    problems = nn_r3.plans_problems(root)
    if problems:
        raise Stop(f'the recipe: {problems}')
    digest = nn_r3.write_splits(root, real_split())
    log(f'{nn_r3.splits_path(root)}: sha256 {digest}; the plans are NN\'s')
    print(digest)


def cmd_decode_switch(args):
    """<out>/decode_switch.json, once.  --off: D off (the lead's words; decode.py's sha256 kept for the record).  --on: R3b's decision record through R3b's reader with require_pass=True
    (r3_core.load_decision), its decoder_spec equal to decoder_spec of this code tree and not the stub, decode.py's sha256 here equal to the one the record holds; the switch pins both sha256
    (r3_core.switch_document).  Written from the clean checkout of the smoke's commit (or from the snapshot): compose checks the pins again in the snapshot (the review's I1)."""
    import r3_core
    target = Path(args.out)/r3c.DECODE_SWITCH
    if target.exists():
        raise Stop(f'{target} exists; the switch is set once')
    if not args.words or not args.words.strip():
        raise Stop("--words: the lead's words and the time")
    if args.on:
        decision = r3_core.load_decision(r3c.R3B_DECISION)
        document = dict(r3_core.switch_document(dict(decision, path=str(r3c.R3B_DECISION)), words=args.words), utc=ufrun.utc())
    else:
        module = r3_core.decode_module_path()
        document = dict(schema='bayesreg.r3.decode_switch.v2', decode=False, words=args.words, decode_py=dict(path=r3c.DECODE_PY_REL, sha256=r3c.sha256_file(module)), utc=ufrun.utc())
    r3c.write_json_once(target, document)
    print(json.dumps(document, ensure_ascii=False, indent=1))


# ---- prepare, chains, status ----------------------------------------------------------------------------------------------------------------------------
def smoke_gate(snapshot, repo, *, smoke=r3c.SMOKE):
    """(problems, the smoke snapshot's identity or None): <smoke>/smoke_r3.json reads 'pass', its code record is <smoke>/snapshot's (verified), and the training snapshot is at the same commit with
    the same files (the existing one's) or would be made from `repo`'s HEAD (run_r2.smoke_gate's rules without targets.json)."""
    from bayesreg.seg.unet.snapshot import _git
    smoke = Path(smoke)
    try:
        document = r3c.read_json(smoke/'smoke_r3.json')
    except (OSError, ValueError) as exc:
        return [f'{smoke}/smoke_r3.json cannot be read: {type(exc).__name__}: {exc}'], None
    if not isinstance(document, dict):
        return [f'{smoke}/smoke_r3.json: no such file or not a JSON object; the GPU smoke comes before the training'], None
    problems = [] if document.get('reading') == 'pass' else [f"the smoke's reading is {document.get('reading')!r}, not 'pass': {document.get('why')}"]
    try:
        mine = run_r2.snapshot_identity(smoke/'snapshot')
    except Exception as exc:
        return problems+[f"the smoke's snapshot {smoke/'snapshot'}: {type(exc).__name__}: {exc}"], None
    recorded = document.get('code')
    if not isinstance(recorded, dict) or any(recorded.get(k) != mine[k] for k in run_r2.IDENTITY_KEYS):
        problems.append(f"smoke_r3.json's code is not the snapshot {smoke/'snapshot'}: the smoke did not run from it")
    try:
        if Path(snapshot).exists():
            theirs = run_r2.snapshot_identity(snapshot)
            training = theirs['commit']
            if theirs['files_sha256'] != mine['files_sha256']:
                problems.append(f'the training snapshot {snapshot} holds other files than the smoke\'s')
        else:
            training = _git(repo, 'rev-parse', '--verify', 'HEAD^{commit}')
    except Exception as exc:
        return problems+[f'the training snapshot or its commit: {type(exc).__name__}: {exc}'], mine
    if training != mine['commit']:
        problems.append(f"the smoke ran on commit {mine['commit']}, the training snapshot is (or would be made) at {training}")
    return problems, mine


def prepare(out, repo, *, smoke=r3c.SMOKE):
    import subprocess
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    set_run_log(out)
    snapshot = out/'snapshot'
    problems, smoke_code = smoke_gate(snapshot, repo, smoke=smoke)
    if problems:
        raise Stop('prepare refused before the snapshot (the GPU smoke): '+'; '.join(problems))
    smoke_document = r3c.read_json(Path(smoke)/'smoke_r3.json')
    record = verify_snapshot(snapshot) if snapshot.exists() else make_snapshot(repo, snapshot, paths=SNAPSHOT_PATHS_R3)
    made = run_r2.snapshot_identity(snapshot)
    if (made['commit'], made['files_sha256']) != (smoke_code['commit'], smoke_code['files_sha256']):
        raise Stop(f"the training snapshot {snapshot} is not the smoke's code: kept, not used; the lead decides")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    for argv in ([r3c.PY, '-B', '-m', 'bayesreg.seg.unet.cli', 'check-inputs', '--profile', r3c.L_PROFILE, '--require-snapshot'],
                 [r3c.PY, '-B', str(snapshot/'experiments/r3/run_r3.py'), 'members-plan', '--out', str(out), '--splits-sha256', smoke_document['splits_sha256']]):
        done = subprocess.run(argv, cwd=str(snapshot), env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise Stop(f'{shlex.join(argv)} failed: {done.stdout[-2000:]} {done.stderr[-2000:]}')
    log('pinned inputs checked, members_plan.json written: no label or image was read')
    return record


def launch_argv(*, out=r3c.OUT, wait_minutes=WAIT_MINUTES, dev=False):
    snapshot = Path(out)/'snapshot'
    unit, runtime = (DEV_UNIT, dev_runtime_max_seconds()) if dev else (UNIT, runtime_max_seconds())
    argv = ['systemd-run', '--user', f'--unit={unit}', '--collect', f'--working-directory={snapshot}', f'--property=MemoryMax={MEMORY_MAX}', f'--property=MemorySwapMax={MEMORY_SWAP_MAX}', f'--property=RuntimeMaxSec={runtime}',
            '--setenv=PYTHONDONTWRITEBYTECODE=1', *[f'--setenv={name}={THREADS}' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')],
            r3c.PY, '-B', str(snapshot/'experiments/r3/run_r3.py'), 'dev-chain' if dev else 'chain', '--out', str(out), '--wait-minutes', str(wait_minutes)]
    text = ' '.join(argv)
    if 'pytest' in text or 'seq-vp-' in text:
        raise ValueError(f'a start string must not name the test runner or the seq-vp- prefix: {text}')
    return argv


def runtime_max_seconds():
    """Every step at its limit and every gate closed to its limit: a safety valve, not a budget."""
    steps = chain_steps(Spec(out=r3c.OUT, cwd=r3c.OUT/'snapshot'), '<sha>')
    seconds = sum(TIMEOUTS[k] for _, k, _ in steps if k in TIMEOUTS)
    gated = sum(1 for _, k, _ in steps if k in GPU_KINDS)
    return seconds+gated*WAIT_MINUTES*60+1800


def dev_runtime_max_seconds():
    steps = dev_steps(Spec(out=r3c.OUT, cwd=r3c.OUT/'snapshot'), '<sha>', '<sha>', True)
    return sum(TIMEOUTS[k] for _, k, _ in steps)+sum(1 for _, k, _ in steps if k in GPU_KINDS)*WAIT_MINUTES*60+1800


def real_spec(out, wait_minutes=WAIT_MINUTES):
    return Spec(out=Path(out), cwd=CODE_ROOT, wait_minutes=wait_minutes)


def cmd_chain(args, dev=False):
    require_snapshot()
    set_run_log(args.out)
    problems, _ = smoke_gate(Path(args.out)/'snapshot', None)
    if problems:
        raise Stop('the chain refused before its first step (the GPU smoke): '+'; '.join(problems))
    spec = real_spec(args.out, args.wait_minutes)
    plan = read_members_plan(args.out)
    again = members_plan(spec, real_split(), splits_sha256=plan['nn']['splits_sha256'])
    if again != plan:
        raise Stop('members_plan.json does not follow from the snapshot and the pinned inputs now')
    if dev:
        if not (Path(args.out)/'chain_state'/'eval_r3.done').exists():
            r3_lock.check('dev')
        run_dev_chain(spec, plan)
        log(f'finished: {args.out}/analysis/decision_r3.json. STOP (nothing after the reading)')
    else:
        if not (Path(args.out)/'chain_state'/'compose.done').exists():
            r3_lock.check('calib')                  # not 33 hours of training for a composition the lock would refuse
        switch = r3c.read_json(Path(args.out)/r3c.DECODE_SWITCH)
        log(f"the decoding switch: {'D ' + ('on' if switch.get('decode') else 'off') if isinstance(switch, dict) else 'not set yet (needed before compose; the chain stops there without it)'}")
        run_chain(spec, plan)
        log(f'finished: {args.out}/compose/composition.json. STOP (the lead reads it; the development reading is another unit)')


def cmd_status(args):
    out = Path(args.out)
    state = out/'chain_state'
    print('state:', ' '.join(sorted(p.name for p in state.iterdir())) if state.is_dir() else '(none)')
    print('hold:', 'YES' if (out/'hold').exists() else 'no')
    for path in sorted(out.glob('r3l/fold*/r3_check.json'))+sorted(Path(r3c.NN_ROOT).glob('models/*/nn_check.json'))+[out/'control_check'/'eval_control.json', out/'compose'/'composition.json', out/'analysis'/'decision_r3.json']:
        if path.is_file():
            document = json.loads(path.read_text(encoding='utf-8'))
            print(f'{path}:', json.dumps({k: document.get(k) for k in ('problems', 'stopped', 'iterations_run', 'control', 'winner', 'no_change', 'reading_text', 'delta_seg35')}, ensure_ascii=False))
    for kind in ('calib', 'dev'):
        print(f'read lock {r3_lock.path_of(kind)}:', json.dumps(r3_lock.describe(kind), ensure_ascii=False) if r3_lock.path_of(kind).exists() else '(none)')


# ---- the dry run --------------------------------------------------------------------------------------------------------------------------------------
def estimate(measured):
    """〔估算〕L from UF's largest wall per step x 80000; N from NN's measured median epoch (nnunet_v1/train_record.json, 250 epochs); the rest."""
    per_step = max(r['per_step'] for r in measured['uf'])
    l_hours = r3c.L_STEPS*per_step/3600
    nn_epoch = measured.get('nn_epoch_seconds_median')
    nn_hours = None if nn_epoch is None else r3c.NN_EPOCHS*nn_epoch/3600
    rest = dict(control_check=10/60, calibrations=2*3/60, nn_infer_calib=2*15/60, compose=20/60)
    return dict(l_per_step_seconds=per_step, l_member_hours=l_hours, l_hours=2*l_hours, nn_epoch_seconds=nn_epoch, nn_model_hours=nn_hours, nn_hours=None if nn_hours is None else 4*nn_hours,
                rest_hours=sum(rest.values()), train_chain_hours=None if nn_hours is None else 2*l_hours+4*nn_hours+sum(rest.values()), dev_chain_hours=dict(nn_infer_dev=40/60, eval_r3=20/60, decide=1/60),
                wall_stop_over_estimate=dict(l=r3c.L_WALL_SECONDS/3600/l_hours, nn=None if nn_hours is None else r3c.NN_WALL_SECONDS/3600/nn_hours))


DOCUMENTED = dict(uf_per_step=.2218, nn_epoch_seconds_median=82.)     # runs/20261110-r2/plan.md revision 1 section 8 (UF's largest measured wall per step); decision_log 11-05 (NN: a median 82 s per epoch)


def dry_run(scratch=None, *, synthetic=True, documented=False):
    work = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix='r3_dryrun_'))
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    problems = []
    print("== the lead's start strings (printed, not run)")
    for line in (['run_nn.py', 'prepare', '--out', str(r3c.NN_ROOT)], ['run_nn.py', 'preprocess', '--out', str(r3c.NN_ROOT)]):
        print('nn root:', shlex.join([r3c.PY, '-B', 'experiments/nn/'+line[0], *line[1:]]))
    print('splits:', shlex.join([r3c.PY, '-B', 'experiments/r3/run_r3.py', 'nn-splits']))
    print('switch (after R3a and R3b are merged and R3b\'s diagnosis ran, before the smoke):', shlex.join([r3c.PY, '-B', 'experiments/r3/run_r3.py', 'decode-switch', '--on', '--words', '<the lead>']),
          f"or --off; --on reads {r3c.R3B_DECISION} ({'there' if r3c.R3B_DECISION.is_file() else 'not there yet'}) through R3b's reader, require_pass=True, and pins it and decode.py")
    try:
        import r3_core
        from bayesreg.seg.decode import decoder_spec
        print(f"  decode.py here: sha256 {r3c.sha256_file(r3_core.decode_module_path())[:12]}..., stub {decoder_spec()['stub']} (the stub cannot be switched on)")
    except Exception as exc:
        problems.append(f'decode.py: {exc}')
    print('prepare:', shlex.join([r3c.PY, '-B', 'experiments/r3/run_r3.py', 'prepare', '--repo', '<the clean checkout the smoke was made from>']))
    problems += snapshot_path_problems(log=print)
    print('chain:', shlex.join(launch_argv()))
    print('dev chain:', shlex.join(launch_argv(dev=True)))
    print(f'RuntimeMaxSec {runtime_max_seconds()} s ({runtime_max_seconds()/3600:.1f} h) and {dev_runtime_max_seconds()} s: every step at its limit {json.dumps(TIMEOUTS)} and every gate closed; a safety valve, not a budget')
    print(f'\n== the chain of {UNIT} (printed, not run; hold file {r3c.OUT/"hold"})')
    for name, kind, argv in chain_steps(Spec(out=r3c.OUT, cwd=r3c.OUT/'snapshot'), '<splits_final.json sha256 of members_plan.json>'):
        print(f"{name} [{'gate, ' if kind in GPU_KINDS else ''}hold, limit {TIMEOUTS.get(kind, 'in-process')} s]: {shlex.join(argv) if argv else 'in this process'}")
    print(f'\n== the development chain of {DEV_UNIT} (released separately)')
    for name, kind, argv in dev_steps(Spec(out=r3c.OUT, cwd=r3c.OUT/'snapshot'), '<composition.json sha256>', '<splits sha256>', True):
        print(f"{name} [{'gate, ' if kind in GPU_KINDS else ''}limit {TIMEOUTS[kind]} s]: {shlex.join(argv)}")
    print('\n== the candidates (closed) and the rule')
    for c in r3c.candidate_order(True):
        print(f"  {c}: {json.dumps(r3c.delivered(c), ensure_ascii=False)}")
    print(f'  seg4 tolerance {r3c.SEG4_TOLERANCE} against {r3c.REFERENCE}; tie band {r3c.TIE_BAND}; simplest: fewer models, then no D, then UF, then seg35, then the order')
    print(f'the development gate (BZ\'s (b)): mean Delta seg35 >= +{r3c.SEG_BAR}, >= {r3c.SEG_POSITIVE_MIN}/20 positive, mean Delta seg4 >= {r3c.SEG4_FLOOR}')
    print('\n== measured (UF\'s members, NN\'s training record; the training subjects\' calib Dice only) and the estimate')
    try:
        if documented:                                      # the implementer's dry run: no file under <RESULTS>/seg is opened (the brief of R3a)
            print(f'(--documented: the numbers of the documents, {DOCUMENTED}; no train record is read)')
            measured, nn_train = dict(uf=[dict(member='UF (documented)', per_step=DOCUMENTED['uf_per_step'], gpu_peak_allocated_gib=9.77)], nn_epoch_seconds_median=DOCUMENTED['nn_epoch_seconds_median']), {}
        else:
            measured = dict(uf=run_r2.uf_measurement())
            nn_train = r3c.read_json(r3c.NN_REFERENCE/'train_record.json') or {}
            measured['nn_epoch_seconds_median'] = nn_train.get('epoch_seconds_median')
        print(json.dumps(dict(measured=dict(uf=[{k: r[k] for k in ('member', 'per_step', 'gpu_peak_allocated_gib')} for r in measured['uf']], nn_epoch_seconds_median=measured['nn_epoch_seconds_median'],
                                            nn_final_checkpoint_bytes=nn_train.get('final_checkpoint_bytes')), estimate=estimate(measured)), indent=1))
    except (OSError, ValueError, KeyError) as exc:
        problems.append(f'measured: {exc}')
    print('\n== the gate on this machine now (read only; it decides nothing here)')
    print(json.dumps(gate_problems(), indent=1))
    print('\n== the real chain is refused here: it needs a verified snapshot')
    try:
        require_snapshot()
        problems.append('the chain was not refused outside a snapshot')
    except CodeNotClean as exc:
        print(f'refused: {str(exc)[:160]}')
    for kind in ('calib', 'dev'):
        print(f"read lock {r3_lock.path_of(kind)}: {'EXISTS' if r3_lock.path_of(kind).exists() else 'none'}")
        if r3_lock.path_of(kind).exists():
            problems.append(f'the {kind} read lock exists')
    print(f"roots: {r3c.OUT} {'EXISTS' if r3c.OUT.exists() else 'not there'}; {r3c.NN_ROOT} {'EXISTS' if r3c.NN_ROOT.exists() else 'not there'}; {r3c.SMOKE} {'EXISTS' if r3c.SMOKE.exists() else 'not there'}")
    print(f"disk now: /home free {shutil.disk_usage('/home').free/2**30:.0f} GiB, C: free {shutil.disk_usage('/mnt/c').free/2**30:.0f} GiB (floor {r3c.C_FLOOR_GIB:.0f})")
    print('\n(not done here, by the brief: the sha256 of the folds file and splits/ lists, the nnU-Net root -- nn-splits, prepare and the chains do them under the lead)')
    if synthetic:
        print('\n== both chains on synthetic subjects, CPU (nnU-Net replaced by r3_synthetic: no nnunetv2 runs)')
        import r3_synthetic
        problems += r3_synthetic.synthetic_chain(work, env)
    print(f"\n== dry run problems: {'; '.join(problems) if problems else 'none'}; work {work}")
    return 0 if not problems else 8


def run_nn_stop():
    import run_nn
    return run_nn.Stop


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--scratch', type=Path, default=None)
    parser.add_argument('--no-synthetic', action='store_true')
    parser.add_argument('--documented', action='store_true', help='--dry-run: the documented numbers in place of the UF and NN train records')
    sub = parser.add_subparsers(dest='command')
    p = sub.add_parser('prepare')
    p.add_argument('--repo', type=Path, default=r3c.MAIN_REPO)
    p.add_argument('--out', type=Path, default=r3c.OUT)
    p = sub.add_parser('members-plan', help='(prepare runs it from the snapshot)')
    p.add_argument('--out', type=Path, default=r3c.OUT)
    p.add_argument('--splits-sha256', required=True)
    p = sub.add_parser('print-launch')
    p.add_argument('--out', type=Path, default=r3c.OUT)
    p.add_argument('--wait-minutes', type=int, default=WAIT_MINUTES)
    p.add_argument('--dev', action='store_true')
    for name in ('chain', 'dev-chain'):
        p = sub.add_parser(name)
        p.add_argument('--out', type=Path, default=r3c.OUT)
        p.add_argument('--wait-minutes', type=int, default=WAIT_MINUTES)
    sub.add_parser('status').add_argument('--out', type=Path, default=r3c.OUT)
    sub.add_parser('nn-splits').add_argument('--nn-root', type=Path, default=r3c.NN_ROOT)
    p = sub.add_parser('decode-switch')
    p.add_argument('--out', type=Path, default=r3c.OUT)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--on', action='store_true', help=f"R3b's {r3c.R3B_DECISION} must say pass (its reader, require_pass=True); its decoder and this tree's decode.py are pinned")
    group.add_argument('--off', action='store_true')
    p.add_argument('--words', default=None)
    for name in ('nn-train', 'nn-infer'):
        p = sub.add_parser(name)
        p.add_argument('--out', type=Path, default=r3c.OUT)
        p.add_argument('--nn-root', type=Path, default=r3c.NN_ROOT)
        p.add_argument('--splits-sha256', required=True)
        p.add_argument('--synthetic', action='store_true')
        p.add_argument('--data-root', type=Path, default=Path('/mnt/d/data'))
        p.add_argument('--folds', type=Path, default=None)
        p.add_argument('--allowlist', type=Path, default=None)
        p.add_argument('--dev-source', default='stage_L2R_val_v1')
        p.add_argument('--training-source', default='bayesreg_train_v1')
        if name == 'nn-train':
            p.add_argument('--model', required=True, choices=r3c.NN_NAMES)
            p.add_argument('--continue', dest='resume', action='store_true', help='nnUNetv2_train --c (the lead, after a failed run)')
        else:
            p.add_argument('--set', required=True, choices=('calibA', 'calibB', 'dev'))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from bayesreg.io.guard import SubjectSealed
    import r3_core
    try:
        if args.dry_run:
            return dry_run(args.scratch, synthetic=not args.no_synthetic, documented=args.documented)
        if args.command in ('nn-train', 'nn-infer') and not args.synthetic and (args.nn_root.resolve() != r3c.NN_ROOT.resolve() or args.out.resolve() != r3c.OUT.resolve() or args.folds or args.allowlist
                                                                              or args.data_root.resolve() != Path('/mnt/d/data') or args.dev_source != 'stage_L2R_val_v1' or args.training_source != 'bayesreg_train_v1'):
            raise Stop('real data: the roots, the folds, the allowlist and the data root are the plan\'s; other values only with --synthetic')
        if args.command == 'prepare':
            prepare(args.out, args.repo)
        elif args.command == 'members-plan':
            import run_nn
            run_nn.require_preprocessed(r3c.NN_ROOT, code=require_snapshot())      # the nnU-Net root as nn-train will check it, from the snapshot, before any training
            print(json.dumps(members_plan(real_spec(args.out), real_split(), splits_sha256=args.splits_sha256)['l']['members'], indent=1))
        elif args.command == 'print-launch':
            print(shlex.join(launch_argv(out=args.out, wait_minutes=args.wait_minutes, dev=args.dev)))
        elif args.command == 'chain':
            cmd_chain(args)
        elif args.command == 'dev-chain':
            cmd_chain(args, dev=True)
        elif args.command == 'status':
            cmd_status(args)
        elif args.command == 'nn-splits':
            cmd_nn_splits(args)
        elif args.command == 'decode-switch':
            cmd_decode_switch(args)
        elif args.command == 'nn-train':
            cmd_nn_train(args)
        elif args.command == 'nn-infer':
            cmd_nn_infer(args)
        else:
            build_parser().print_help()
            return 2
    except (Stop, run_nn_stop(), CodeNotClean, r3_lock.ReadLocked, r3_core.Refused, SubjectSealed) as exc:
        log(f'STOP: {exc}')
        return 8
    return 0


if __name__ == '__main__':
    sys.exit(main())
