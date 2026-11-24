"""R2's one segmentation-level evaluation (runs/20261110-r2/plan.md section 4): the 20 development subjects, argmax against the truth, per label, seg35 (nanmean over 1..35), seg4, the small six, the big six, the
other 23, the cortex and the white matter (experiments/unv2/eval_seg.py, experiments/nn/eval_nn.py, as eval_uf.py); OASIS_OAS1_0440_MR1 on its own in the reading.

Arms (one pass per subject; no prediction file is kept).  Every ensemble arm is bayesreg.seg.unet.infer.ensemble_probabilities' arithmetic (softmax(z_m / T_m) added in the order below, divided by n,
renormalized), float16 as infer.segment_subject stores probs36, argmax with ties to the smallest label:
  UF4       UF's four members foldA_s0 foldA_s1 foldB_s0 foldB_s1 (eval_uf.py's (a)); recomputed here, the control and the reference of the gate
  R2S       UF4's four and r2s's foldA_s2, foldB_s2: the order foldA_s0, foldA_s1, r2s foldA_s2, foldB_s0, foldB_s1, r2s foldB_s2 (the registry order of the candidate, r2_common.candidate_plan); the one arm of the gate
  r2s_2     described: r2s's two members alone
  r2s_A r2s_B   described: each new member alone (argmax of its logits)
One arm (the lead's ruling of 11-10 ~10:42 PM CST): r2sl is not trained, and no arm here reads it.
Control: UF4 against eval_uf.json's UF4 rows (pinned), |mean seg35| and |mean seg4| <= 2e-4 (a GPU forward pass, uf_common.CONTROL_TOL_FORWARD); the truth files and the images must be those eval_uf.json
records.  --control-only (the chain's first step, before the training): UF4 alone, the same comparison, no lock (numbers read long ago).

Order of the checks: the read lock first on real data (r2_lock); then, before any file of a subject: the subjects (the allowlist source, not never_for_training, no sealed list, no training subject, outside
every member's label contact); the members (UF's by their pinned sha256 and eval_uf.json's; R2's by the plan: recipe r2_small, the config of their recipe and of targets.json, the schedule run to its end,
the last step, their fold's subjects); then the lock, immediately before the first truth label.  A failure before it costs no read.
    python -B experiments/r2/eval_r2.py --out-dir DIR [--device cuda] [--bz-go TEXT]
    python -B experiments/r2/eval_r2.py --control-only --out-dir DIR [--device cuda]
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(HERE), str(REPO/'experiments/unv2'), str(REPO/'experiments/ul'), str(REPO/'experiments/nn')]
import eval_nn  # noqa: E402
import eval_seg  # noqa: E402
import eval_u1  # noqa: E402
import r2_common as rc  # noqa: E402
import r2_lock  # noqa: E402
from bayesreg.eval.truth import load_sealed_lists  # noqa: E402
from bayesreg.seg.unet.data import OasisFiles, load_folds, normalize_image, read_image  # noqa: E402
from bayesreg.seg.unet.large import GIB, cap_gpu_memory  # noqa: E402
from bayesreg.seg.unet.pins import check_synthetic_data_root, check_synthetic_subjects, load_pinned_allowlist, source_root_for  # noqa: E402
from bayesreg.seg.unet.snapshot import code_record, require_clean_code  # noqa: E402
from bayesreg.seg.unet.train import predict_logits  # noqa: E402

FORMAT = 'bayesreg.r2.eval.v1'
GPU_CAP_GIB = 20.
METRICS = eval_nn.METRICS                              # seg35, seg4, small_six, big_six, other_23, cortex, white_matter
CANDIDATE_ARMS = rc.CANDIDATE_ORDER                    # R2S: the arm of the gate
NEW = tuple((recipe, fold) for fold in ('A', 'B') for recipe in rc.RECIPE_ORDER)       # r2s A, r2s B


def forward_order():
    """The members in the order of their forward pass: (family, member name) with family 'uf' or the recipe; fold A first (UF's two, then r2s), then fold B."""
    out = []
    for fold in ('A', 'B'):
        out += [('uf', f'fold{fold}_s0'), ('uf', f'fold{fold}_s1')]+[(recipe, rc.member_name(fold)) for recipe in rc.RECIPE_ORDER]
    return out


def ensemble_arms():
    """{arm: [(family, member) in its order of addition]}: each a subsequence of forward_order(), so that one pass adds each member to every arm in the arm's own order."""
    order = forward_order()
    def pick(families):
        return [m for m in order if m[0] in families]
    arms = dict(UF4=pick({'uf'}), R2S=pick({'uf', 'r2s'}), r2s_2=pick({'r2s'}))
    for name in rc.CANDIDATE_ORDER:                    # the order of the candidate's registry (r2_common.candidate_plan)
        plan = [m for _, members in rc.candidate_plan(name) for m in members]
        if arms[name] != plan:
            raise AssertionError(f'{name}: {arms[name]} is not the registry order {plan}')
    return arms


SINGLE_ARMS = {f'{recipe}_{fold}': (recipe, rc.member_name(fold)) for recipe, fold in NEW}
ARMS = tuple(ensemble_arms())+tuple(SINGLE_ARMS)


class Accumulator:
    """ensemble_probabilities' arithmetic, member by member: total = p (the first), else total.add_(p); then total /= n and total / total.sum(dim=0)."""

    def __init__(self):
        self.total, self.n = None, 0

    def add(self, p):
        self.total = p.clone() if self.total is None else self.total.add_(p)
        self.n += 1

    def probabilities(self):
        total = self.total
        total /= self.n
        return total/total.sum(dim=0, keepdim=True)


def member_summary(directory, blob, record, train_record):
    return dict(directory=str(directory), member_sha256=rc.sha256_file(Path(directory)/'member.pt'), fold=record['fold'], seed=record['seed'], temperature=blob['temperature'], recipe=train_record.get('recipe'),
                checkpoint_choice=blob.get('checkpoint_choice'), checkpoint_iteration=blob.get('checkpoint_iteration'), window=list(blob['window']) if blob.get('window') else None, window_gaussian_sigma=blob.get('window_gaussian_sigma'),
                stopped=train_record.get('stopped'), iterations_run=train_record.get('iterations_run'), best_iteration=train_record.get('best_iteration'), best_calib_mean_dice35=train_record.get('best_calib_mean_dice35'),
                checkpoint_calib_mean_dice35=train_record.get('checkpoint_calib_mean_dice35'), wall_seconds=train_record.get('wall_seconds'), memory=train_record.get('memory'), n_label_contact=len(record['label_contact_subjects']))


def planned_config(recipe, targets, overrides=None):
    """The recorded config an R2 member must have: TrainConfigSmall's defaults with the recipe's --set values (r2_common.train_config), as JSON; overrides (the tests' tiny config) replace the recipe's."""
    from dataclasses import asdict
    from bayesreg.seg.unet.small import TrainConfigSmall
    config = rc.train_config(recipe, targets) if overrides is None else dict(overrides[recipe], target_labels=[int(t) for t in targets], p_target=rc.P_TARGET, target_dice_weight=rc.TARGET_DICE_WEIGHT)
    return json.loads(json.dumps(asdict(TrainConfigSmall(**config))))


def member_record_problems(train_record, *, recipe, name, expected_config, namesake):
    """Why an R2 member's train record is not the plan's (empty: it is): fold and seed; recipe r2_small; the config; early stop off; the schedule to its end; the last step; the subjects of its fold
    (those UF's same-fold member trained and was calibrated on, `namesake`: dict(train, calib, contact))."""
    fold, seed = name.removeprefix('fold').split('_s')
    problems = []
    if (train_record.get('fold'), train_record.get('seed')) != (fold, int(seed)):
        problems.append(f"{recipe}/{name}: the record is fold {train_record.get('fold')} seed {train_record.get('seed')}")
    if train_record.get('recipe') != rc.PROFILE:
        problems.append(f"{recipe}/{name}: recipe {train_record.get('recipe')!r}, not {rc.PROFILE}")
    config = train_record.get('config') or {}
    if config != expected_config:
        problems.append(f"{recipe}/{name}: the config differs from the plan's on {sorted(k for k in set(config) | set(expected_config) if config.get(k) != expected_config.get(k))}")
    max_iters = expected_config['max_iters']
    if not (config.get('min_iters', 0) >= max_iters and config.get('patience', 0) >= max_iters):
        problems.append(f'{recipe}/{name}: early stop not off')
    if train_record.get('stopped') != 'max_iters' or train_record.get('iterations_run') != max_iters:
        problems.append(f"{recipe}/{name}: stopped {train_record.get('stopped')!r} after {train_record.get('iterations_run')} steps, not the {max_iters} of the schedule")
    if train_record.get('checkpoint_choice') != 'last' or train_record.get('checkpoint_file') != 'model_last.pt' or train_record.get('checkpoint_iteration') != max_iters:
        problems.append(f'{recipe}/{name}: the member is not the last step')
    if sorted(train_record.get('train_subjects') or []) != namesake['train'] or sorted(train_record.get('calib_subjects') or []) != namesake['calib'] or list(train_record.get('label_contact_subjects') or []) != namesake['contact']:
        problems.append(f'{recipe}/{name}: the subjects are not those of the fold (UF\'s same-fold members)')
    return problems


def uf_namesakes(uf_root, pinned=None):
    """{fold: dict(train, calib, contact)} from UF's members' train records (re-hashed against member.json; member.pt against the pinned sha256 when given).  Ids only."""
    out = {}
    for name in rc.UF_ORDER:
        directory = Path(uf_root)/name
        member = json.loads((directory/'member.json').read_text(encoding='utf-8'))
        if rc.sha256_file(directory/'train_record.json') != member['train_record_sha256']:
            raise ValueError(f'{directory}/train_record.json: sha256 differs from member.json')
        if pinned is not None and member['member_sha256'] != pinned[name]:
            raise ValueError(f'{directory}: member {member["member_sha256"][:12]}... is not the pinned UF member')
        record = json.loads((directory/'train_record.json').read_text(encoding='utf-8'))
        entry = dict(train=sorted(record['train_subjects']), calib=sorted(record['calib_subjects']), contact=list(record['label_contact_subjects']))
        fold = record['fold']
        if fold in out and out[fold] != entry:
            raise ValueError(f'UF\'s two members of fold {fold} disagree on their subjects')
        out[fold] = entry
    return out


def load_members(*, uf_root, member_root, uf_evaluated, targets, device, overrides=None, pinned=None, uf_only=False):
    """{(family, name): (model, blob, summary)} in forward_order(); the problems.  UF's members: their member.pt the one eval_uf.json evaluated (and the pinned one on real data), the last step with a Gaussian
    window; R2's: the plan's (member_record_problems), member.pt re-hashed (eval_u1.load_arm), the last step, its train record the file member.json was written from."""
    members, problems = {}, []
    namesakes = uf_namesakes(uf_root, pinned)
    for family, name in forward_order():
        if uf_only and family != 'uf':
            continue
        directory = Path(uf_root)/name if family == 'uf' else Path(member_root)/family/name
        if not (directory/'member.json').is_file():
            problems.append(f'{family}/{name}: no member.json (member missing)')
            continue
        model, blob, record = eval_u1.load_arm(directory, device)
        train_record = json.loads((directory/'train_record.json').read_text(encoding='utf-8'))
        found = []
        if rc.sha256_file(directory/'train_record.json') != record.get('train_record_sha256'):
            found.append(f'{family}/{name}: train_record.json is not the file member.json was written from')
        if family == 'uf':
            if record['member_sha256'] != uf_evaluated.get(name):
                found.append(f'{family}/{name}: member.pt is not the one eval_uf.json evaluated')
        else:
            found += member_record_problems(train_record, recipe=family, name=name, expected_config=planned_config(family, targets, overrides), namesake=namesakes[name[4]])
            last = directory/str(train_record.get('checkpoint_file'))
            if not last.is_file() or rc.sha256_file(last) != train_record.get('checkpoint_sha256'):
                found.append(f'{family}/{name}: {last.name} missing or its sha256 differs from train_record.json')
        if blob.get('checkpoint_choice') != 'last' or blob.get('window_gaussian_sigma') is None or not blob.get('window'):
            found.append(f'{family}/{name}: member.pt is not a last-step member with a Gaussian sliding window')
        problems += found
        members[(family, name)] = (model, blob, dict(member_summary(directory, blob, record, train_record), problems=found))
    return members, problems


def read_uf_eval(path, *, pinned=None):
    """(document, sha256): eval_uf.json, against its pin on real data; its UF4 rows, members, images and truth sha256 are what the control compares with."""
    raw = Path(path).read_bytes()
    import hashlib
    digest = hashlib.sha256(raw).hexdigest()
    if pinned is not None and digest != pinned:
        raise ValueError(f'{path}: sha256 {digest} is not the pinned {pinned}')
    document = json.loads(raw.decode('utf-8'))
    if 'UF4' not in (document.get('rows') or {}) or sorted(document.get('members') or {}) != sorted(rc.UF_ORDER):
        raise ValueError(f'{path}: no UF4 rows or not the four UF members')
    return document, digest


def read_targets(path, *, pinned=None):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if pinned is not None and rc.sha256_file(path) != pinned:
        raise ValueError(f'{path}: not the targets.json the members were planned with')
    targets = [int(t) for t in document['targets']]
    if not targets or targets != sorted(set(targets)) or not set(rc.SMALL_SIX) <= set(targets):
        raise ValueError(f'{path}: targets {targets} are not the small six and more, sorted')
    return targets


def evaluate(*, out_dir, uf_root=rc.UF_OUT, member_root=rc.OUT, uf_eval=rc.UF_EVAL, targets_path=rc.PP_ROOT/'targets.json', targets_sha256=None, data_root=eval_seg.DATA_ROOT, allowlist_path=eval_seg.ALLOWLIST,
             folds_path=eval_seg.FOLDS, dev_source=eval_seg.DEV_SOURCE, sealed_repo=None, synthetic=False, overrides=None, device='cuda', argv=None, log=print, bz_go=None, control_only=False):
    """The evaluation (control_only: UF4 alone, no lock).  Returns the document written to <out_dir>/eval_r2.json (eval_control.json for control_only)."""
    if synthetic and bz_go is not None:
        raise ValueError('--bz-go is for the read lock of real data')
    if control_only and bz_go is not None:
        raise ValueError('--bz-go does not go with --control-only: the control takes no lock')
    if not synthetic and not control_only:
        r2_lock.check(bz_go)                                        # the first thing on real data
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'{out_dir} exists and is not empty; never overwritten')
    code = code_record()
    if synthetic:
        check_synthetic_data_root(data_root)
        from bayesreg.io.guard import load_dev_allowlist
        allowlist = load_dev_allowlist(allowlist_path)
        check_synthetic_subjects([s for ids in allowlist.subjects.values() for s in ids], 'the allowlist')
        allowlist_record = dict(mode='synthetic', allowlist_path=str(allowlist_path))
    else:
        require_clean_code(code)
        if overrides is not None or any(Path(a).resolve() != Path(b).resolve() for a, b in ((uf_root, rc.UF_OUT), (member_root, rc.OUT), (uf_eval, rc.UF_EVAL), (data_root, eval_seg.DATA_ROOT), (allowlist_path, eval_seg.ALLOWLIST), (folds_path, eval_seg.FOLDS))) or dev_source != eval_seg.DEV_SOURCE:
            raise ValueError('real data: the members, eval_uf.json, the data root, the allowlist, the folds and the development source are the plan\'s; other values only with --synthetic')
        allowlist, allowlist_record = load_pinned_allowlist(allowlist_path)
        allowlist_record = {k: v for k, v in allowlist_record.items() if k != 'code'}
    uf, uf_sha = read_uf_eval(uf_eval, pinned=None if synthetic else rc.UF_EVAL_SHA256)
    targets = None if control_only else read_targets(targets_path, pinned=targets_sha256)
    contacts = {}
    for family, name in forward_order():
        path = (Path(uf_root)/name if family == 'uf' else Path(member_root)/family/name)/'member.json'
        if path.is_file() and (family == 'uf' or not control_only):
            contacts[f'{family}/{name}'] = json.loads(path.read_text(encoding='utf-8'))['label_contact_subjects']
    sealed = load_sealed_lists(Path(sealed_repo) if sealed_repo is not None else source_root_for(code))
    folds = load_folds(folds_path)
    subjects = eval_u1.admit(allowlist, dev_source, sealed, folds.training_subjects, contacts)        # before any file of a subject
    if subjects != list(uf['subjects']):
        raise ValueError('the subjects differ from eval_uf.json\'s')
    device = torch.device(device)
    if device.type == 'cuda':
        cap_gpu_memory(device, GPU_CAP_GIB)
    uf_evaluated = {n: m['member_sha256'] for n, m in uf['members'].items()}
    members, problems = load_members(uf_root=uf_root, member_root=member_root, uf_evaluated=uf_evaluated, targets=targets, device=device, overrides=overrides, pinned=None if synthetic else rc.UF_MEMBER_SHA256, uf_only=control_only)
    if problems:
        raise ValueError(f'not evaluated, the truth is not read: {problems}')
    percentiles = {blob['percentile'] for _, blob, _ in members.values()}
    if len(percentiles) != 1:
        raise ValueError(f'the members disagree on the normalization percentile: {percentiles}')
    percentile = percentiles.pop()
    arms_of = {a: m for a, m in ensemble_arms().items() if not control_only or a == 'UF4'}
    singles = {} if control_only else dict(SINGLE_ARMS)
    arm_names = tuple(arms_of)+tuple(singles)
    lookup = eval_seg.load_seg4_lookup()
    files = OasisFiles(Path(data_root))
    rows, truth_sha, image_sha, seconds, dev_read = {a: [] for a in arm_names}, {}, {}, [], None
    for subject in subjects:
        started = time.perf_counter()
        values, meta = read_image(files, subject)
        image_sha[subject] = meta['sha256']
        if image_sha[subject] != (uf.get('image_sha256') or {}).get(subject):
            raise ValueError(f'{subject}: the image is not the one eval_uf.json evaluated')
        image, _ = normalize_image(values, percentile=percentile)
        del values
        accumulators = {a: Accumulator() for a in arms_of}
        tops = {}
        for key in forward_order():
            if key not in members:
                continue
            model, blob, _ = members[key]
            z = predict_logits(model, image, device=device, amp=blob['amp'], window=tuple(blob['window']), gaussian_sigma=blob['window_gaussian_sigma'])
            for arm, (recipe, name) in singles.items():
                if key == (recipe, name):
                    tops[arm] = z.argmax(1)[0].to(torch.uint8).cpu().numpy()
            p = torch.softmax(z.float()/float(blob['temperature']), dim=1)[0]
            del z
            for arm, plan in arms_of.items():
                if key in plan:
                    accumulators[arm].add(p)
            del p
        for arm, acc in accumulators.items():
            p16 = acc.probabilities().cpu().numpy().astype(np.float16)        # as infer.segment_subject stores probs36
            tops[arm] = p16.argmax(axis=0).astype(np.uint8)
            del p16
        del accumulators, image
        if not synthetic and not control_only and dev_read is None:          # the moment before the first truth label
            dev_read = r2_lock.take(dict(out_dir=str(out_dir), members={f'{f}/{n}': s['member_sha256'] for (f, n), (_, _, s) in members.items()}, targets=targets, command=argv, code_commit=code.get('commit')), bz_go=bz_go)
        true35, sha35 = eval_seg.read_truth(data_root, subject, 'seg35')
        true4, sha4 = eval_seg.read_truth(data_root, subject, 'seg4')
        truth_sha[subject] = dict(seg35=sha35, seg4=sha4)
        for arm in arm_names:
            rows[arm].append(eval_nn.metrics_of(eval_seg.summarize_subject(tops[arm], true35, true4, lookup)))
        seconds.append(time.perf_counter()-started)
        log(f'{subject}: '+', '.join(f"{a} {rows[a][-1]['seg35']:.4f}" for a in (('UF4',) if control_only else ('UF4',)+CANDIDATE_ARMS)))
    means = {a: {m: eval_nn.mean_of(r, m) for m in METRICS} for a, r in rows.items()}
    theirs = [eval_nn.metrics_of({k: uf['rows']['UF4'][s][k] for k in ('seg35', 'seg4', 'per_label', 'seg4_per_label')} | dict(groups={g: uf['rows']['UF4'][s][g] for g in eval_seg.GROUPS})) for s in subjects]
    control = dict(differences={m: means['UF4'][m]-eval_nn.mean_of(theirs, m) for m in ('seg35', 'seg4')}, tolerance=rc.CONTROL_TOL_FORWARD,
                   rows_equal=sum(1 for i, s in enumerate(subjects) if all(rows['UF4'][i][k] == uf['rows']['UF4'][s][k] for k in ('seg35', 'seg4'))),
                   truth_same=all((uf.get('truth_sha256') or {}).get(s) == truth_sha[s] for s in subjects), images_same=True)
    control['passed'] = all(abs(v) <= rc.CONTROL_TOL_FORWARD for v in control['differences'].values()) and control['truth_same'] and len(subjects) == len(uf['subjects'])
    document = dict(format=FORMAT+('.control' if control_only else ''), plan=rc.plan_record(REPO), created_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), command=argv, code=code, synthetic=synthetic, dev_source=dev_source, subjects=subjects,
                    arms=list(arm_names), ensemble_arms={a: [f'{f}/{n}' for f, n in plan] for a, plan in arms_of.items()}, candidate_arms=[] if control_only else list(CANDIDATE_ARMS), targets=targets,
                    targets_sha256=None if control_only else rc.sha256_file(targets_path), means=means, control=control, rows={a: {s: r[i] for i, s in enumerate(subjects)} for a, r in rows.items()},
                    per_label_mean={a: [eval_seg.nanmean([eval_nn.clean(row['per_label'][k]) for row in rows[a]]) for k in range(eval_seg.N_CLASSES-1)] for a in arm_names},
                    members={f'{f}/{n}': s for (f, n), (_, _, s) in members.items()}, member_problems=problems, uf_eval=dict(path=str(uf_eval), sha256=uf_sha), uf_rows={s: uf['rows']['UF4'][s] for s in subjects},
                    uf_singles={a: {s: uf['rows'][a][s] for s in subjects} for a in ('UF_A0', 'UF_A1', 'UF_B0', 'UF_B1') if a in (uf.get('rows') or {})},
                    image_sha256=image_sha, truth_sha256=truth_sha, dev_read=dev_read, allowlist=allowlist_record, allowlist_sha256=allowlist.sha256, folds_sha256=folds.sha256, sealed_lists=sealed.record,
                    seconds=eval_u1.seconds_summary(seconds), device=str(device), gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/GIB if device.type == 'cuda' else None,
                    note='开发集，单次观测；分割级读数，不是配准结果，不是留出集结果。' + ('只有 UF4 的控制，不取读取锁（这些数早就读过）。' if control_only else '真值只在本评测读取（r2_dev_read.lock）。'))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not control_only:
        with (out_dir/'per_subject.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['subject', 'arm', *METRICS])
            for arm in arm_names:
                for i, s in enumerate(subjects):
                    writer.writerow([s, arm, *[eval_nn.clean(rows[arm][i].get(m)) for m in METRICS]])
    rc.write_json_once(out_dir/('eval_control.json' if control_only else 'eval_r2.json'), document)      # last
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--uf-root', type=Path, default=rc.UF_OUT)
    parser.add_argument('--member-root', type=Path, default=rc.OUT)
    parser.add_argument('--uf-eval', type=Path, default=rc.UF_EVAL)
    parser.add_argument('--targets', type=Path, default=rc.PP_ROOT/'targets.json')
    parser.add_argument('--targets-sha256', default=None, help='the sha256 members_plan.json pinned (the chain passes it)')
    parser.add_argument('--allowlist', type=Path, default=eval_seg.ALLOWLIST)
    parser.add_argument('--folds', type=Path, default=eval_seg.FOLDS)
    parser.add_argument('--dev-source', default=eval_seg.DEV_SOURCE)
    parser.add_argument('--data-root', type=Path, default=eval_seg.DATA_ROOT)
    parser.add_argument('--sealed-repo', type=Path, default=None)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--train-overrides', type=Path, default=None, help='--synthetic only: {recipe: tiny config}')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--bz-go', default=None)
    parser.add_argument('--control-only', action='store_true')
    args = parser.parse_args(argv)
    if args.train_overrides is not None and not args.synthetic:
        parser.error('--train-overrides goes with --synthetic')
    overrides = json.loads(args.train_overrides.read_text(encoding='utf-8')) if args.train_overrides else None
    try:
        document = evaluate(out_dir=args.out_dir, uf_root=args.uf_root, member_root=args.member_root, uf_eval=args.uf_eval, targets_path=args.targets, targets_sha256=args.targets_sha256, data_root=args.data_root,
                            allowlist_path=args.allowlist, folds_path=args.folds, dev_source=args.dev_source, sealed_repo=args.sealed_repo, synthetic=args.synthetic, overrides=overrides, device=args.device,
                            argv=sys.argv if argv is None else ['eval_r2.py', *argv], bz_go=args.bz_go, control_only=args.control_only)
    except r2_lock.DevReadLocked as exc:
        print(f'STOP: {exc}', flush=True)
        return 8
    print(json.dumps(dict(means={a: {m: document['means'][a][m] for m in ('seg35', 'seg4')} for a in document['arms'] if a in ('UF4',)+CANDIDATE_ARMS}, control=document['control']), indent=1))
    return 0 if (document['control']['passed'] or not args.control_only) else 8


if __name__ == '__main__':
    sys.exit(main())
