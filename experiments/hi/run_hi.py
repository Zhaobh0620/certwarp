"""HI (runs/20261107-hi/任务书.md): the hold-out entry of the v4 structure-layer inference (unet35_v4_ens) and its development check.  The rules are in hi_core.py; the unseal record, the allowlist steps and the
estimates in runs/20261107-hi/入口说明.md; the reading of the check in runs/20261107-hi/devcheck_plan.md.

    python -B experiments/hi/run_hi.py infer --dry-run --record DIR [--unseal PATH] [--resume PATH] [--subjects-file PATH] [--out DIR]
        no image, no truth, no GPU, nothing written but DIR/dry_run.json (A5a review m8: passed, problems, expected, unexpected, p0_ok -- the checklist reads it, not the exit code): the code root, the pinned allowlist, the sealed lists, the registry against the anchor, the members re-hashed, the grouping, the decision of the check (None
        today: the entry would refuse), the lock, the unseal record (none today: the entry would refuse; with --unseal, every subject of it through checks 6 and 7 by id), the output directory, and the guard's
        census on the id lists (the B-line hold-out, H1-H4, T0, the training set, the development stages, ids off the allowlist).
    python -B experiments/hi/run_hi.py infer --unseal PATH [--out hi_core.HOLDOUT_OUT_SUGGESTED] [--subjects-file PATH] [--resume PATH]
        only after BZ's unseal record is committed (and the allowlist holds the unsealed subjects, on the unseal branch) and DEVCHECK_DECISION is filled: cuda, backend hs_a, the one output root, once (the lock);
        after a failure only with a committed resume record.  The output directory is hi_core.HOLDOUT_OUT_SUGGESTED, the one HE and XB read (A5a; prereg revision 2 list 5 (l)): --out may be omitted
        or name it; any other --out is refused (exit 8) before anything is read.  The code is the pinned commit's (rule one, hi_core.pinned_code_check; contract C24.3).
    python -B experiments/hi/run_hi.py devcheck --dry-run [--backend hs_a|segment_subject|both] [--out DIR]
        no image, no truth, no GPU, nothing written: the plan's sha256, the 20 development subjects, the guard on each, the registry, the members, the plans, the registered predictions' provenance, the output directory.
    python -B experiments/hi/run_hi.py devcheck [--backend hs_a|segment_subject|both] [--out DIR] [--device cuda]
        the lead's run, after the review and the merge (GPU): the 20 development subjects through the backend(s), bitwise against seg/unet35_v4_ens/pred, read by devcheck_plan.md; a new directory (default hi_devcheck_v1).

Both real runs need a clean code tree whose root is this entry's (hi_core.check_code_root, review I1) and run with PYTHONPATH=<code tree>; the allowlist's sources are re-hashed against the main checkout (data/ is its link).
Exit codes: 0 done, or a dry run without problems; 2 usage; 8 refused or a check failed (the message says which; nothing is rerun); 9 the dry run found problems.
"""
import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import hi_core as hi  # noqa: E402

PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
SCRIPT = 'experiments/hi/run_hi.py'
UNIT = dict(devcheck='tag-hi-devcheck', infer='tag-hi-infer')
RUNTIME = dict(devcheck=2*3600, infer=4*3600)
UNIT_LOG = dict(devcheck=hi.RESULTS/'seg'/'hi_devcheck_v1_unit.log', infer=hi.RESULTS/'seg'/'hi_holdout_infer_unit.log')     # review m5: outside the output directory (which must be new), appended
# 〔估算〕from HS's seg phase (seg/unet35_v4_ens/infer_record.json, 20 subjects): forward median 2.2 s (max 5.8 s), write median 43.3 s, total median 64.9 s (with a read-back of nnU-Net's probabilities that HI does not do)
SECONDS = dict(forward=2.2, write=43.3, image=2., reference=8., compare=4.)
MIB_PER_SUBJECT = 384                           # probs36.npz about 260 MB + structure_groups.npz about 124 MB (0438)


def read_subjects_file(path):
    lines = [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines()]
    return [line for line in lines if line and not line.startswith('#')]


def real_inputs():
    """The pinned inputs of a real run (CodeNotClean, PinnedInputRefused, SealedListError, ProvenanceRejected on any departure).  The code root first (review I1), then JSON and id lists, the stage lists the allowlist
    names; nothing of a subject."""
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.io.pair import load_segmenter_registry
    from bayesreg.seg.unet.data import OasisFiles, load_folds
    from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping
    from bayesreg.seg.unet.pins import DEFAULT_ALLOWLIST, DEFAULT_FOLDS, PINNED_COUNTS, PINNED_FOLDS_SHA256, PinnedInputRefused, load_pinned_allowlist
    from bayesreg.seg.unet.snapshot import code_record
    code = code_record()
    root_check = hi.check_code_root(code)
    allowlist, allowlist_record = load_pinned_allowlist(DEFAULT_ALLOWLIST, source_root=hi.MAIN_REPO, code=code)
    if hi.file_sha256(DEFAULT_FOLDS) != PINNED_FOLDS_SHA256:
        raise PinnedInputRefused(f'{DEFAULT_FOLDS}: not the pinned folds.json')
    training = load_folds(DEFAULT_FOLDS, expect_counts=PINNED_COUNTS).training_subjects
    sealed = load_sealed_lists(hi.REPO)
    registry = load_segmenter_registry(hi.REPO/hi.CONTRACT_SEGMENTER['registry'])
    grouping = load_grouping(DEFAULT_GROUPS_PATH, 'structure13_v1')
    allowlist_record = {k: v for k, v in allowlist_record.items() if k != 'code'}
    return SimpleNamespace(code=dict(code, root_check=root_check), allowlist=allowlist, allowlist_record=allowlist_record, training=sorted(training), sealed=sealed, registry=registry, grouping=grouping,
                           files=OasisFiles(hi.DATA_ROOT), anchor=hi.CONTRACT_SEGMENTER)


def dev_subjects(inputs):
    """The 20 development subjects as HS's seg phase admitted them (eval_seg.admit_subjects on stage_L2R_val_v1: SubjectSealed for a sealed list, T0, off the allowlist or a training subject)."""
    import eval_seg
    return sorted(eval_seg.admit_subjects(inputs.allowlist, hi.DEV_SOURCE, inputs.sealed, inputs.training))


def launch_line(command, code):
    """The start string the lead may use (printed by the dry run; nothing here starts a unit).  Review m5: the threads and the CPU quota of HS's seg units, the log appended to a fixed file outside the output."""
    env = dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, NUMBA_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=code)
    return ' '.join(['systemd-run', '--user', f'--unit={UNIT[command]}', '--wait', '--collect', f'--working-directory={code}', '--property=MemoryMax=8G', '--property=MemorySwapMax=4G', '--nice=10',
                     '--property=CPUQuota=800%', f'--property=RuntimeMaxSec={RUNTIME[command]}', f'--property=StandardOutput=append:{UNIT_LOG[command]}', f'--property=StandardError=append:{UNIT_LOG[command]}',
                     *[f'--setenv={k}={v}' for k, v in env.items()], PY, '-B', f'{code}/{SCRIPT}', command, '...'])


def free_gib(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free/2**30


def census_groups(inputs):
    """Id lists only: the sealed lists, T0, the training set, the development stages, and the OASIS ids 0001-0457 on no list and off the allowlist."""
    sealed, allowlist = inputs.sealed, inputs.allowlist
    sources = {s['name']: s for s in allowlist.sources}
    listed = {s for ids in sealed.subjects.values() for s in ids}
    off = [f'OASIS_OAS1_{n:04d}_MR1' for n in range(1, 458)]
    off = [s for s in off if s not in listed and not allowlist.allows(s)]
    groups = {f'sealed {name}': sorted(ids) for name, ids in sorted(sealed.subjects.items())}
    groups['H2 (LPBA40 S37-S40)'] = ['S37', 'S38', 'S39', 'S40']
    groups['T0 (stage_T0_tune_v1)'] = sorted(sources['stage_T0_tune_v1']['subjects'])
    groups['B-line training set'] = list(inputs.training)
    for name in (hi.DEV_SOURCE, 'stage_E11_native_v1', 'stage_LPBA40_v1_tune'):
        groups[f'development ({name})'] = sorted(sources[name]['subjects'])
    groups['OASIS 0001-0457 off the allowlist, on no list'] = off
    return groups


def record_dir_problem(path):
    """A5a review m8: --record DIR of a dry run: a new or empty directory outside the results root and the code trees (mktemp -d).  None, or why not."""
    target = Path(path).absolute()
    for root in (hi.RESULTS, hi.REPO, hi.MAIN_REPO):
        for form in (Path(root).absolute(), Path(root).resolve()):
            if target == form or form in target.parents or target.resolve() == form or form in target.resolve().parents:
                return f'--record {path} is inside {root}'
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        return f'--record {path} exists and is not an empty directory'
    return None


def write_dry_record(o, *, problems, rule_one_at=None, rule_one=None, out=print):
    """A5a review m8 (the lead's ruling ~12:29 PM CST 11-10): the dry run's record <--record>/dry_run.json in the shape every hold-out entry's dry run writes (holdout_common.dry_run_summary:
    passed, problems, expected, unexpected, p0_ok); the problem of rule one (index rule_one_at) under holdout_common.rule_one_kind, every other problem unexpected.  Nothing when the caller gave
    no --record (the functions' callers in the tests); run_hi's command line requires it."""
    record_dir = getattr(o, 'record', None)
    if record_dir is None:
        return None
    holdout = HERE.parent/'holdout'
    if str(holdout) not in sys.path:
        sys.path.insert(0, str(holdout))
    import holdout_common as hc
    rows = [(hc.rule_one_kind(rule_one or []) if i == rule_one_at else 'problem', p) for i, p in enumerate(problems)]
    path = Path(record_dir)/'dry_run.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    hc.write_json_once(path, dict(schema='bayesreg.hi.dry_run.v1', command='infer', unseal=getattr(o, 'unseal', None), resume=getattr(o, 'resume', None), out=getattr(o, 'out', None),
                                  rule_one=rule_one, utc=hi.utc(), **hc.dry_run_summary(rows)))
    out(f'dry-run record: {path} (the checklist reads it, not the exit code)')
    return path


def dry_run_infer(o, out=print):
    problems = []
    out(f'HI infer dry run ({hi.utc()}): no image, no truth label, no GPU; nothing written but the record --record DIR/dry_run.json; read: JSON records, id lists, the stage lists the allowlist '
        'names, the four member.pt files (their sha256)')
    try:
        inputs = real_inputs()
    except Exception as exc:
        out(f'the pinned inputs are refused: {type(exc).__name__}: {exc}')
        write_dry_record(o, problems=[f'the pinned inputs are refused: {type(exc).__name__}: {exc}'], out=out)
        return 9
    out(f"code: {inputs.code.get('kind')} {inputs.code.get('commit')} clean {inputs.code.get('clean')}; {inputs.code['root_check']['checked']}: {inputs.code['root_check']['root']}")
    out(f"allowlist {inputs.allowlist.sha256} (sources re-hashed against {hi.MAIN_REPO}: {sorted(inputs.allowlist_record['allowlist_sources_verified'])})")
    out(f'sealed lists: {[(r.get("name"), r.get("subjects")) for r in inputs.sealed.record]}; training set {len(inputs.training)} subjects')
    available = None
    try:
        registry = hi.check_registry(inputs.registry, inputs.anchor)
        out(f"registry {registry['path']}: sha256 {registry['sha256']} = the anchor; models {registry['models']}")
        available = hi.member_files(inputs.registry, hi.MODELS_ROOT)
        out(f'members: the 4 registered member.pt found under {hi.MODELS_ROOT} by sha256 ({len(available)} files there)')
    except hi.HIRefused as refused:
        problems.append(str(refused))
        out(f'REFUSED {refused}')
    from bayesreg.seg.unet.groups import require_inference_grouping
    out(f'grouping {inputs.grouping.name} {inputs.grouping.source_sha256}: {require_inference_grouping(inputs.grouping)}')
    try:
        decision = hi.check_decision(hi.DEVCHECK_DECISION)
        out(f"decision of the development check: filled and it holds ({decision['checked']}): {decision['reading']}, backend {decision['backend']}, devcheck.json {decision['devcheck_sha256']}, "
            f"decided by {decision['decided_by']!r}, review {decision['review']!r}")
        out(f"the environment check against the decision ({dict(decision['environment'])}) runs on cuda at the start of infer, after the machine gate and before the lock; this dry run uses no device, "
            "so it is not done here")
    except hi.HIRefused as refused:
        out(f'decision of the development check: `infer` refuses now ({refused})' + ('' if hi.DEVCHECK_DECISION is None else ' -- a decision is written but does not hold'))
        if hi.DEVCHECK_DECISION is not None:
            problems.append(str(refused))
    lock = Path(hi.HOLDOUT_LOCK)
    out(f"lock {lock}: {'exists: ' + str(hi.lock_lines(lock)[:1]) if lock.exists() else 'absent (the first run takes it with O_EXCL just before the first image)'}")
    unseal = None
    if o.unseal is None:
        out(f"unseal record: none given (none exists until BZ writes one, at {hi.UNSEAL_REL}): `infer` refuses at stage unseal before anything else; the format is in runs/20261107-hi/入口说明.md §3")
    else:
        try:
            unseal = hi.load_unseal(o.unseal, anchor=inputs.anchor)
            out(f'unseal record: {unseal.record()}')
            bound, _ = hi.bind(unseal, inputs.allowlist)          # C23.8-1: the guard of the record's subjects is the one bound to the record
            out(f'the pinned allowlist bound to the record (bayesreg.io.unseal.bind_unseal): sha256 {bound.sha256}')
            subjects = list(unseal.subjects)
            if o.resume is not None:
                resume = hi.load_resume(o.resume, unseal=unseal, lock=lock)
                out(f'resume record: {resume.record()}')
                subjects = list(resume.subjects)
            passed = 0
            for subject in subjects:                # review m1: every subject of the record through checks 6 and 7 with the bound guard, by id only
                hi.admit(subject, unseal=unseal, allowlist=bound, sealed=inputs.sealed, training=inputs.training)
                if available is not None:
                    hi.plan_of(inputs.registry, subject, bound, available)
                passed += 1
            out(f'{passed} of the {len(subjects)} subjects of the record pass checks 6 and 7 (admission through the bound guard, D24); no file of any of them opened')
        except Exception as refused:
            problems.append(f'{type(refused).__name__}: {refused}')
            out(f'REFUSED {type(refused).__name__}: {refused}')
    target = Path(o.out) if o.out else hi.HOLDOUT_OUT_SUGGESTED
    problem = out_problem(o.out)
    if problem:
        problems.append(problem)
        out(f'REFUSED {problem}')
    try:
        hi.check_out_location(target, resume=bool(o.resume))      # A5a review I2: a resume writes into the one root (it must be there)
        state = 'new'
    except (hi.HIRefused, FileExistsError) as refused:
        problems.append(str(refused))
        state = f'REFUSED {refused}'
    out(f"output directory {target}{'' if o.out else ' (the one output directory of infer, hi_core.HOLDOUT_OUT_SUGGESTED)'}: {state}; free {free_gib(target):.0f} GiB, about {MIB_PER_SUBJECT} MiB per subject")
    try:                                        # A5a: rule one; before BZ unseals there is no pinned record: the expected refusal, recorded (C24.3-2)
        pinned, pinned_problems = hi.pinned_code_check()
    except Exception as exc:                    # noqa: BLE001 - a dry run reports
        pinned, pinned_problems = None, [f'{type(exc).__name__}: {exc}']
    rule_one_at = None
    if pinned_problems:
        rule_one_at = len(problems)
        problems.append(f'rule one: {pinned_problems}')
        out(f'rule one (the pinned commit of runs/holdout_B_v1/pinned_code.json): REFUSED {pinned_problems}: `infer` refuses at stage code until the pinned record is committed')
    else:
        out(f"rule one: the code is the pinned commit {pinned['pinned']}'s (HEAD {pinned['head']})")
    if o.subjects_file:
        listed = read_subjects_file(o.subjects_file)
        out(f'subjects file {o.subjects_file}: {len(listed)} ids' + ('' if unseal is None else f", {sum(s in unseal.subjects for s in listed)} of them in the unseal record"))
    census = hi.guard_census(census_groups(inputs), allowlist=inputs.allowlist, sealed=inputs.sealed, training=inputs.training, registry=inputs.registry)
    out('the guard without an unseal record (the allowlist guard, the sealed lists, the sources, D24; no file of any subject opened):')
    for group, counts in census.items():
        out(f'  {group}: {counts}')
        admitted = sum(n for layer, n in counts.items() if layer.startswith('admitted'))
        if admitted:
            problems.append(f'{group}: {admitted} ids pass checks 6 and 7 without a record')
    n = len(unseal.subjects) if unseal is not None else 100
    out(f"〔估算〕{n} subjects with hs_a: forward about {SECONDS['forward']:.1f} s each on the GPU, the files about {SECONDS['write']:.0f} s each on the CPU: about {n*(SECONDS['forward']+SECONDS['write']+SECONDS['image'])/60:.0f} min, "
        f"{n*MIB_PER_SUBJECT/1024:.1f} GiB (HS's seg phase, seg/unet35_v4_ens/infer_record.json)")
    out(f"the machine gate (no other tag-* or seq-vp-* unit, no GPU compute process, utilization <= {hi.GPU_UTIL_MAX} %) runs at the start of infer, not here")
    out(f"start string (the lead's; not started here): {launch_line('infer', inputs.code.get('root'))}")
    out(f"torch imported by this dry run: {'torch' in sys.modules} (the guard's module imports it; no device used); problems: {problems or 'none'}")
    write_dry_record(o, problems=problems, rule_one_at=rule_one_at, rule_one=list(pinned_problems), out=out)
    return 9 if problems else 0


def dry_run_devcheck(o, out=print):
    problems = []
    backends = backends_of(o.backend)
    out(f'HI devcheck dry run ({hi.utc()}): no image, no truth label, no GPU, nothing written; read: JSON records, id lists, the stage lists the allowlist names, the four member.pt files (their sha256), the 20 registered provenance.json')
    try:
        inputs = real_inputs()
        subjects = dev_subjects(inputs)
    except Exception as exc:
        out(f'the pinned inputs are refused: {type(exc).__name__}: {exc}')
        return 9
    out(f"code: {inputs.code.get('kind')} {inputs.code.get('commit')} clean {inputs.code.get('clean')}; {inputs.code['root_check']['checked']}; allowlist {inputs.allowlist.sha256}")
    if subjects != sorted(hi.DEV_SUBJECTS):
        problems.append(f'the development subjects are not 0438-0457: {subjects}')
    try:
        plan = hi.check_plan()
        out(f"plan {plan['path']}: sha256 {plan['sha256']} = the pinned value; reading: {hi.READ_SELECT.format(n=20)} / {hi.READ_PENDING} / {hi.READ_UNDECIDED}")
        registry = hi.check_registry(inputs.registry, inputs.anchor)
        out(f"registry {registry['path']}: sha256 {registry['sha256']} = the anchor")
        for subject in subjects:
            hi.allowlist_guard(subject, inputs.allowlist)
        out(f'the guard (infer.check_subject_admitted, pinned allowlist): {len(subjects)} development subjects admitted')
        available = hi.member_files(inputs.registry, hi.MODELS_ROOT)
        plans = {s: hi.plan_of(inputs.registry, s, inputs.allowlist, available) for s in subjects}
        orders = {tuple(sha for _, sha, _ in plan) for plan, _, _ in plans.values()}
        policies = sorted({policy for _, _, policy in plans.values()})
        out(f'plans: {len(plans)} subjects, policies {policies}, one member order {len(orders) == 1}: {[sha[:8] for sha in next(iter(orders))]}')
        references = {s: hi.reference_record(hi.REGISTERED_PRED, s, registry=inputs.registry, plan=plans[s][0], anchor=inputs.anchor) for s in subjects}
        commits = sorted({r['code_commit'] for r in references.values()})
        out(f"registered predictions: {len(references)} provenance.json of this segmenter, registry and member order; made with code {commits}, environment {next(iter(references.values()))['environment']}")
    except Exception as refused:
        problems.append(f'{type(refused).__name__}: {refused}')
        out(f'REFUSED {type(refused).__name__}: {refused}')
    lock = Path(hi.DEVCHECK_LOCK)
    if lock.exists():                               # diff review R1: the check runs once
        problems.append(f'{lock} exists: the development check ran ({hi.lock_lines(lock)[:1]}); devcheck refuses')
    out(f"lock {lock}: {'EXISTS' if lock.exists() else 'absent (the check takes it with O_EXCL just before the first image)'}; order: hs_a on all 20 first, then segment_subject")
    out(f"the machine gate (no other tag-* or seq-vp-* unit than {list(hi.DEVCHECK_UNITS)} when this process runs in it, no GPU compute process, utilization <= {hi.GPU_UTIL_MAX} %) runs at the start of devcheck, "
        "before the lock, not here")
    target = Path(o.out) if o.out else hi.DEVCHECK_OUT
    try:
        hi.check_out_location(target)
        state = 'new'
    except (hi.HIRefused, FileExistsError) as refused:
        problems.append(str(refused))
        state = f'REFUSED {refused}'
    if target.resolve() != Path(hi.DEVCHECK_OUT).resolve():
        problems.append(f'--out {target} is not {hi.DEVCHECK_OUT}: the decision takes only {Path(hi.DEVCHECK_OUT)/hi.DEVCHECK_RECORD} (diff review R1); devcheck refuses another directory')
    per = sum(SECONDS[k] for k in ('forward', 'image', 'reference', 'compare'))
    minutes = {b: 20*(per+(SECONDS['write'] if b == 'segment_subject' else 0.))/60 for b in backends}
    disk = 20*MIB_PER_SUBJECT/1024 if 'segment_subject' in backends else 0.
    out(f"backends {backends}; output {target}: {state}; free {free_gib(target):.0f} GiB; segment_subject writes about {MIB_PER_SUBJECT} MiB per subject (hs_a compares in memory)")
    out(f"〔估算〕{', '.join(f'{b} about {m:.0f} min' for b, m in minutes.items())}; disk about {disk:.1f} GiB; GPU busy about {SECONDS['forward']:.1f} s per subject and backend")
    out(f"start string (the lead's; not started here): {launch_line('devcheck', inputs.code.get('root'))}")
    out(f"torch imported by this dry run: {'torch' in sys.modules} (no device used); problems: {problems or 'none'}")
    return 9 if problems else 0


def backends_of(text):
    return list(hi.BACKENDS) if text == 'both' else [text]


def out_problem(given):
    """A5a (prereg revision 2 list 5 (l); decision_log 2026-11-10 ~1:53 AM CST I4): the hold-out entry writes only into hi_core.HOLDOUT_OUT_SUGGESTED (read at call time), the pred root HE and XB read.
    None when --out is omitted or names it (compared resolved); the reason otherwise."""
    if given is None or Path(given).resolve() == Path(hi.HOLDOUT_OUT_SUGGESTED).resolve():
        return None
    return f'--out {given} is not {hi.HOLDOUT_OUT_SUGGESTED}: the hold-out entry writes only there (he_core.HOLDOUT_SOURCES and xb_core.HI_PRED_ROOTS read its pred/; A5a)'


def run_infer(o):
    if not o.unseal:
        print('infer needs --unseal (or --dry-run)', file=sys.stderr)
        return 2
    problem = out_problem(o.out)
    if problem:
        print(f'STOP: {problem}', file=sys.stderr)
        return 8
    try:
        inputs = real_inputs()
        subjects = read_subjects_file(o.subjects_file) if o.subjects_file else None
        record = hi.infer_holdout(subjects, unseal_path=o.unseal, out_root=hi.HOLDOUT_OUT_SUGGESTED, files=inputs.files, device='cuda', allowlist=inputs.allowlist, allowlist_record=inputs.allowlist_record, sealed=inputs.sealed,
                                  training=inputs.training, registry=inputs.registry, models_root=hi.MODELS_ROOT, grouping=inputs.grouping, code=inputs.code, command=sys.argv, anchor=inputs.anchor,
                                  resume_path=o.resume)
    except (Exception, SystemExit) as refused:      # review m13: fast_channels' SystemExit too, exit 8
        print(f'STOP: {type(refused).__name__}: {refused}', file=sys.stderr)
        return 8
    print(f"done: {len(record['rows'])} subjects, passed {record['passed']}")
    return 0 if record['passed'] else 8


def run_devcheck(o):
    if o.out and Path(o.out).resolve() != Path(hi.DEVCHECK_OUT).resolve():     # diff review R1: the one check the decision takes; the lock would be spent on another directory
        print(f'STOP: --out {o.out} is not {hi.DEVCHECK_OUT}: the decision of the check takes only {Path(hi.DEVCHECK_OUT)/hi.DEVCHECK_RECORD}', file=sys.stderr)
        return 8
    try:
        inputs = real_inputs()
        subjects = dev_subjects(inputs)
        record = hi.devcheck(subjects, backends=backends_of(o.backend), out_root=Path(o.out) if o.out else hi.DEVCHECK_OUT, files=inputs.files, device=o.device, allowlist=inputs.allowlist,
                             allowlist_record=inputs.allowlist_record, sealed=inputs.sealed, training=inputs.training, registry=inputs.registry, models_root=hi.MODELS_ROOT, grouping=inputs.grouping,
                             reference_root=hi.REGISTERED_PRED, code=inputs.code, command=sys.argv, anchor=inputs.anchor)
    except (Exception, SystemExit) as refused:
        print(f'STOP: {type(refused).__name__}: {refused}', file=sys.stderr)
        return 8
    print(f"done: reading {record['reading']['text']}; {record['summary']}")
    return 0


def parser():
    p = argparse.ArgumentParser(description='HI: the hold-out entry of the v4 structure-layer inference and its development check')
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('infer')
    a.add_argument('--dry-run', action='store_true')
    a.add_argument('--unseal')
    a.add_argument('--resume')
    a.add_argument('--subjects-file')
    a.add_argument('--out')
    a.add_argument('--record', help='with --dry-run (required then): a new or empty directory outside the results root and the code trees; the dry run writes its record DIR/dry_run.json (A5a review m8)')
    d = sub.add_parser('devcheck')
    d.add_argument('--dry-run', action='store_true')
    d.add_argument('--backend', choices=list(hi.BACKENDS)+['both'], default=hi.PRODUCTION_BACKEND)
    d.add_argument('--out')
    d.add_argument('--device', default='cuda')
    return p


def main(argv=None):
    o = parser().parse_args(argv)
    if o.command == 'infer':
        if o.dry_run:
            problem = 'infer --dry-run writes its record into --record DIR (mktemp -d)' if o.record is None else record_dir_problem(o.record)
            if problem:                         # A5a review m8: every dry run of a hold-out entry writes its record
                print(f'usage: {problem}', file=sys.stderr)
                return 2
            return dry_run_infer(o)
        if o.record is not None:
            print('usage: --record is for --dry-run only', file=sys.stderr)
            return 2
        return run_infer(o)
    return dry_run_devcheck(o) if o.dry_run else run_devcheck(o)


if __name__ == '__main__':
    sys.exit(main())
