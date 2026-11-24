"""R3's one segmentation-level evaluation on the development set (runs/20261111-r3/plan.md section 5; the second and last segmentation-level reading of round 2): the 20 development subjects,
argmax (or, with D, the decoded labels) against the truth, per label, seg35 (nanmean over 1..35), seg4, the small six, the big six, the other 23, the cortex, the white matter (the functions of
eval_seg and eval_nn, as eval_r2); OASIS_OAS1_0440_MR1 on its own in the reading.

Arms (one pass per subject; no prediction kept):
  UF4       UF's four members (eval_uf.py's (a), recomputed here): the reference of the gate and the control (|mean seg35|, |mean seg4| against eval_uf.json's rows <= 2e-4; the truth files those)
  COMBO     the composition delivered by compose_r3.py (composition.json, pinned): its U-Net members in their order (fold A then fold B), mixed with the mean of the four nnU-Net models
            (<OUT>/nn_dev, checked) at w = n_N / (n_U + n_N) when it has N, decoded with its spec when it has D -- r3_core's arithmetic; the one arm of the gate
  described: COMBO_noD (with D: the same mixture's argmax), L_2 (L's two members), L_A, L_B (each alone), N4 (the nnU-Net mean's argmax, when N is in), nnA_s0 ... (each model's labels)
--control-only (the chain's first step, before any training): UF4 alone, the control, no lock, no composition.
Order of the checks: the read lock first on real data; then, before any file of a subject: the subjects (the allowlist source, never_for_training, the sealed lists, no training subject, outside every
member's and nnU-Net's label contact), the composition (its sha256, its members), the members (UF's pinned; L's as planned), the nnU-Net check of the development set and the models' records;
then the lock, immediately before the first truth label.
    python -B experiments/r3/eval_r3.py --out-dir DIR --composition-sha256 SHA --splits-sha256 SHA [--device cuda] [--bz-go TEXT]
    python -B experiments/r3/eval_r3.py --control-only --out-dir DIR [--device cuda]
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
sys.path[:0] = [str(REPO), str(HERE), str(REPO/'experiments/unv2'), str(REPO/'experiments/ul'), str(REPO/'experiments/nn'), str(REPO/'experiments/uf'), str(REPO/'experiments/r2')]
import eval_nn  # noqa: E402
import eval_seg  # noqa: E402
import eval_u1  # noqa: E402
import nn_r3  # noqa: E402
import r3_common as r3c  # noqa: E402
import r3_core  # noqa: E402
import r3_lock  # noqa: E402
from bayesreg.eval.truth import load_sealed_lists  # noqa: E402
from bayesreg.seg.unet.data import OasisFiles, load_folds, normalize_image, read_image  # noqa: E402
from bayesreg.seg.unet.large import GIB, cap_gpu_memory  # noqa: E402
from bayesreg.seg.unet.pins import check_synthetic_data_root, check_synthetic_subjects, load_pinned_allowlist, source_root_for  # noqa: E402
from bayesreg.seg.unet.snapshot import code_record, require_clean_code  # noqa: E402

FORMAT = 'bayesreg.r3.eval.v1'
GPU_CAP_GIB = 20.
METRICS = eval_nn.METRICS
UF4_ORDER = [('uf', n) for n in r3c.UF_ORDER]
L_KEYS = [(r3c.L_FAMILY, n) for n in r3c.L_NAMES]


def forward_order(keys):
    """The members in the order of the forward pass: fold A (UF seed 0, seed 1, L), then fold B; only those in `keys`."""
    out = []
    for fold in r3c.FOLDS:
        out += [k for k in (('uf', f'fold{fold}_s0'), ('uf', f'fold{fold}_s1'), (r3c.L_FAMILY, r3c.member_name(fold, r3c.L_SEED))) if k in keys]
    return out


def combo_keys(delivered):
    return [tuple(k.split('/', 1)) for k in delivered['unet']]


def read_composition(path, *, pinned=None, real=True):
    """(document, sha256): composition.json, its sha256 the pinned one; refused when there is no winner or the winner is no change.  real (every reading but the synthetic world's): also
    refused when it is synthetic or forced, or when r3_common.choose on its own means does not give its winner (the review's m5)."""
    raw = Path(path).read_bytes()
    import hashlib
    digest = hashlib.sha256(raw).hexdigest()
    if pinned is not None and digest != pinned:
        raise r3_core.Refused(f'{path}: sha256 {digest} is not the pinned {pinned}')
    document = json.loads(raw.decode('utf-8'))
    if not document.get('winner') or document.get('no_change') or document.get('delivered') is None:
        raise r3_core.Refused(f"{path}: the composition is {document.get('winner')!r} (no change or none): there is no development reading")
    delivered = document['delivered']
    if delivered != r3c.delivered(document['winner']):
        raise r3_core.Refused(f'{path}: the delivered composition is not r3_common.delivered of its winner')
    keys = combo_keys(delivered)
    if forward_order(keys) != keys:
        raise r3_core.Refused(f'{path}: the U-Net members {keys} are not in the order of the forward pass')
    if real:
        if document.get('synthetic') or document.get('forced_for_the_synthetic_chain'):
            raise r3_core.Refused(f'{path}: a synthetic or forced composition')
        order = tuple(document.get('order') or ())
        means = document.get('means') or {}
        if order not in (r3c.candidate_order(False), r3c.candidate_order(True)) or r3c.choose({c: means.get(c) or {} for c in order}, order)['winner'] != document['winner']:
            raise r3_core.Refused(f"{path}: the rule on its own means does not give its winner {document['winner']!r}")
    return document, digest


def evaluate(*, out_dir, uf_root=r3c.UF_OUT, member_root=r3c.OUT, nn_dev=r3c.OUT/'nn_dev', nn_root=r3c.NN_ROOT, composition_path=r3c.OUT/'compose'/'composition.json', composition_sha256=None,
             switch_path=r3c.OUT/r3c.DECODE_SWITCH, splits_sha256=None, uf_eval=r3c.UF_EVAL, data_root=eval_seg.DATA_ROOT, allowlist_path=eval_seg.ALLOWLIST, folds_path=eval_seg.FOLDS,
             dev_source=eval_seg.DEV_SOURCE, sealed_repo=None, synthetic=False, l_overrides=None, device='cuda', argv=None, log=print, bz_go=None, control_only=False):
    if l_overrides is not None and not synthetic:
        raise ValueError('--l-overrides goes with --synthetic')
    if synthetic and bz_go is not None:
        raise ValueError('--bz-go is for the read lock of real data')
    if control_only and bz_go is not None:
        raise ValueError('--bz-go does not go with --control-only: the control takes no lock')
    if not synthetic and not control_only:
        r3_lock.check('dev', bz_go)                                     # the first thing on real data
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
        if any(Path(a).resolve() != Path(b).resolve() for a, b in ((uf_root, r3c.UF_OUT), (member_root, r3c.OUT), (nn_dev, r3c.OUT/'nn_dev'), (nn_root, r3c.NN_ROOT), (composition_path, r3c.OUT/'compose'/'composition.json'),
                                                                  (switch_path, r3c.OUT/r3c.DECODE_SWITCH), (uf_eval, r3c.UF_EVAL), (data_root, eval_seg.DATA_ROOT), (allowlist_path, eval_seg.ALLOWLIST), (folds_path, eval_seg.FOLDS))) \
                or dev_source != eval_seg.DEV_SOURCE:
            raise r3_core.Refused('real data: every input is the plan\'s; other values only with --synthetic')
        if not control_only and not (composition_sha256 and splits_sha256):
            raise r3_core.Refused('--composition-sha256 and --splits-sha256 (the development chain\'s pins) are required on real data')
        allowlist, allowlist_record = load_pinned_allowlist(allowlist_path)
        allowlist_record = {k: v for k, v in allowlist_record.items() if k != 'code'}
    import eval_r2
    uf, uf_sha = eval_r2.read_uf_eval(uf_eval, pinned=None if synthetic else r3c.UF_EVAL_SHA256)
    problems, composition, composition_digest, delivered, decoder, nn_records, check = [], None, None, None, None, {}, None
    if not control_only:
        composition, composition_digest = read_composition(composition_path, pinned=composition_sha256, real=not synthetic)
        delivered = composition['delivered']
        if delivered['decode']:
            on, spec, _ = r3_core.read_switch(switch_path, synthetic=synthetic)
            if not on or spec != composition['decode']['spec']:
                problems.append('the decoding switch now is not the one the composition was made with')
            decoder = spec
    keys = UF4_ORDER if control_only else forward_order(UF4_ORDER+L_KEYS)
    contacts = {}
    for family, name in keys:
        path = (Path(uf_root)/name if family == 'uf' else Path(member_root)/family/name)/'member.json'
        if path.is_file():
            contacts[f'{family}/{name}'] = json.loads(path.read_text(encoding='utf-8'))['label_contact_subjects']
    if delivered is not None and delivered['nn']:
        manifest = r3c.read_json(Path(nn_root)/'prepare_manifest.json') or {}
        contacts['nnunet'] = manifest.get('label_contact_subjects') or []
        if not contacts['nnunet']:
            problems.append(f'{nn_root}/prepare_manifest.json: no label contact')
    sealed = load_sealed_lists(Path(sealed_repo) if sealed_repo is not None else source_root_for(code))
    folds = load_folds(folds_path)
    subjects = eval_u1.admit(allowlist, dev_source, sealed, folds.training_subjects, contacts)        # before any file of a subject
    if subjects != list(uf['subjects']):
        raise r3_core.Refused('the subjects differ from eval_uf.json\'s')
    if delivered is not None and delivered['nn']:
        split_file = nn_r3.splits_path(nn_root)
        digest = r3c.sha256_file(split_file) if split_file.is_file() else None
        if splits_sha256 is not None and digest != splits_sha256:
            problems.append(f'{split_file}: not the pinned splits_final.json')
        nn_records = {m: r3c.read_json(Path(nn_root)/'models'/m/'train_record.json') for m in r3c.NN_NAMES}
        for m, record in nn_records.items():
            problems += nn_r3.model_problems(record, m, splits_sha256=digest, root=nn_root)
        try:
            check = nn_r3.read_check(nn_dev, subjects, list(r3c.NN_NAMES))
            if check['checkpoints'] != {m: (nn_records[m] or {}).get('final_checkpoint_sha256') for m in r3c.NN_NAMES}:
                problems.append('the nnU-Net check of the development set is not of the recorded checkpoints')
        except nn_r3.Stop as exc:
            problems.append(str(exc))
    device = torch.device(device)
    if device.type == 'cuda':
        cap_gpu_memory(device, GPU_CAP_GIB)
    members, member_problems = r3_core.load_unet(uf_root=uf_root, member_root=member_root, uf_evaluated={n: m['member_sha256'] for n, m in uf['members'].items()}, device=device, keys=keys,
                                                  pinned=None if synthetic else r3c.UF_MEMBER_SHA256, l_config=r3_core.l_expected_config(uf_root, l_overrides))
    problems += member_problems
    if problems:
        raise r3_core.Refused(f'not evaluated, the truth is not read: {problems}')
    percentiles = {blob['percentile'] for _, blob, _ in members.values()}
    if len(percentiles) != 1:
        raise r3_core.Refused(f'the members disagree on the normalization percentile: {percentiles}')
    percentile = percentiles.pop()
    sets = dict(UF4=UF4_ORDER) if control_only else dict(UF4=UF4_ORDER, COMBO=combo_keys(delivered), L_2=L_KEYS)
    singles = {} if control_only else dict(L_A=L_KEYS[0], L_B=L_KEYS[1])
    with_nn = delivered is not None and bool(delivered['nn'])
    arms = ['UF4'] if control_only else ['UF4', 'COMBO']+(['COMBO_noD'] if decoder else [])+['L_2', 'L_A', 'L_B']+(['N4', *r3c.NN_NAMES] if with_nn else [])
    lookup = eval_seg.load_seg4_lookup()
    files = OasisFiles(Path(data_root))
    rows, truth_sha, image_sha, seconds, dev_read, decode_records = {a: [] for a in arms}, {}, {}, [], None, {}
    for subject in subjects:
        started = time.perf_counter()
        values, meta = read_image(files, subject)
        image_sha[subject] = meta['sha256']
        if image_sha[subject] != (uf.get('image_sha256') or {}).get(subject):
            raise r3_core.Refused(f'{subject}: the image is not the one eval_uf.json evaluated')
        image, _ = normalize_image(values, percentile=percentile)
        del values
        p16, tops = r3_core.forward(members, forward_order(keys), image, device=device, arms=sets, singles=singles)
        del image
        tops['UF4'] = r3_core.mixed_labels(p16['UF4'])
        if not control_only:
            tops['L_2'] = r3_core.mixed_labels(p16['L_2'])
            p_nn = None
            if with_nn:
                p_nn, nn_labels = r3_core.read_nn_mean(nn_dev, subject, p16['UF4'].shape[1:], row=check['rows'][subject])
                tops['N4'] = nn_labels['mean']
                for m in r3c.NN_NAMES:
                    tops[m] = nn_labels[m]
            weight = delivered['nn_weight'] if with_nn else 0.
            tops['COMBO'], record = r3_core.candidate_labels(p16['COMBO'], p_nn, weight, decoder)
            if decoder:
                decode_records[subject] = record
                tops['COMBO_noD'] = r3_core.mixed_labels(p16['COMBO'], p_nn, weight)
            del p_nn
        del p16
        if not synthetic and not control_only and dev_read is None:          # the moment before the first truth label
            dev_read = r3_lock.take('dev', dict(out_dir=str(out_dir), members={f'{f}/{n}': s['member_sha256'] for (f, n), (_, _, s) in members.items()}, composition_sha256=composition_digest,
                                                nn_checkpoints={m: (r or {}).get('final_checkpoint_sha256') for m, r in nn_records.items()}, command=argv, code_commit=code.get('commit')), bz_go=bz_go)
        true35, sha35 = eval_seg.read_truth(data_root, subject, 'seg35')
        true4, sha4 = eval_seg.read_truth(data_root, subject, 'seg4')
        truth_sha[subject] = dict(seg35=sha35, seg4=sha4)
        for arm in arms:
            rows[arm].append(eval_nn.metrics_of(eval_seg.summarize_subject(tops[arm], true35, true4, lookup)))
        seconds.append(time.perf_counter()-started)
        log(f'{subject}: '+', '.join(f"{a} {rows[a][-1]['seg35']:.4f}" for a in arms[:2]))
    means = {a: {m: eval_nn.mean_of(r, m) for m in METRICS} for a, r in rows.items()}
    theirs = [eval_nn.metrics_of({k: uf['rows']['UF4'][s][k] for k in ('seg35', 'seg4', 'per_label', 'seg4_per_label')} | dict(groups={g: uf['rows']['UF4'][s][g] for g in eval_seg.GROUPS})) for s in subjects]
    control = dict(differences={m: means['UF4'][m]-eval_nn.mean_of(theirs, m) for m in ('seg35', 'seg4')}, tolerance=r3c.CONTROL_TOL_FORWARD,
                   rows_equal=sum(1 for i, s in enumerate(subjects) if all(rows['UF4'][i][k] == uf['rows']['UF4'][s][k] for k in ('seg35', 'seg4'))),
                   truth_same=all((uf.get('truth_sha256') or {}).get(s) == truth_sha[s] for s in subjects), images_same=True)
    control['passed'] = all(abs(v) <= r3c.CONTROL_TOL_FORWARD for v in control['differences'].values()) and control['truth_same'] and len(subjects) == len(uf['subjects'])
    document = dict(format=FORMAT+('.control' if control_only else ''), plan=r3c.plan_record(REPO), created_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), command=argv, code=code, synthetic=synthetic,
                    dev_source=dev_source, subjects=subjects, arms=arms, ensemble_arms={a: [f'{f}/{n}' for f, n in plan] for a, plan in sets.items()}, composition=None if control_only else dict(
                        sha256=composition_digest, winner=composition['winner'], delivered=delivered, decode=composition['decode']), means=means, control=control,
                    rows={a: {s: r[i] for i, s in enumerate(subjects)} for a, r in rows.items()}, per_label_mean={a: [eval_seg.nanmean([eval_nn.clean(row['per_label'][k]) for row in rows[a]]) for k in range(eval_seg.N_CLASSES-1)] for a in arms},
                    members={f'{f}/{n}': s for (f, n), (_, _, s) in members.items()}, member_problems=member_problems, nn=None if not with_nn else dict(check_sha256=r3c.sha256_file(Path(nn_dev)/'check.json'),
                    records={m: {k: (r or {}).get(k) for k in ('final_checkpoint_sha256', 'epochs_logged', 'wall_seconds')} for m, r in nn_records.items()}), decode_records=decode_records,
                    uf_eval=dict(path=str(uf_eval), sha256=uf_sha), uf_rows={s: uf['rows']['UF4'][s] for s in subjects}, image_sha256=image_sha, truth_sha256=truth_sha, dev_read=dev_read,
                    allowlist=allowlist_record, allowlist_sha256=allowlist.sha256, folds_sha256=folds.sha256, sealed_lists=sealed.record, seconds=eval_u1.seconds_summary(seconds), device=str(device),
                    gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/GIB if device.type == 'cuda' else None,
                    note='开发集，单次观测；分割级读数，不是配准结果，不是留出集结果。' + ('只有 UF4 的控制，不取读取锁（这些数早就读过）。' if control_only else '真值只在本评测读取（r3_dev_read.lock）。'))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not control_only:
        with (out_dir/'per_subject.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['subject', 'arm', *METRICS])
            for arm in arms:
                for i, s in enumerate(subjects):
                    writer.writerow([s, arm, *[eval_nn.clean(rows[arm][i].get(m)) for m in METRICS]])
    r3c.write_json_once(out_dir/('eval_control.json' if control_only else 'eval_r3.json'), document)      # last
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--uf-root', type=Path, default=r3c.UF_OUT)
    parser.add_argument('--member-root', type=Path, default=r3c.OUT)
    parser.add_argument('--nn-dev', type=Path, default=r3c.OUT/'nn_dev')
    parser.add_argument('--nn-root', type=Path, default=r3c.NN_ROOT)
    parser.add_argument('--composition', type=Path, default=r3c.OUT/'compose'/'composition.json')
    parser.add_argument('--composition-sha256', default=None)
    parser.add_argument('--switch', type=Path, default=r3c.OUT/r3c.DECODE_SWITCH)
    parser.add_argument('--splits-sha256', default=None)
    parser.add_argument('--uf-eval', type=Path, default=r3c.UF_EVAL)
    parser.add_argument('--allowlist', type=Path, default=eval_seg.ALLOWLIST)
    parser.add_argument('--folds', type=Path, default=eval_seg.FOLDS)
    parser.add_argument('--dev-source', default=eval_seg.DEV_SOURCE)
    parser.add_argument('--data-root', type=Path, default=eval_seg.DATA_ROOT)
    parser.add_argument('--sealed-repo', type=Path, default=None)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--l-overrides', type=Path, default=None, help='--synthetic only: the tiny --set values of L')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--bz-go', default=None)
    parser.add_argument('--control-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        document = evaluate(out_dir=args.out_dir, uf_root=args.uf_root, member_root=args.member_root, nn_dev=args.nn_dev, nn_root=args.nn_root, composition_path=args.composition, composition_sha256=args.composition_sha256,
                            switch_path=args.switch, splits_sha256=args.splits_sha256, uf_eval=args.uf_eval, data_root=args.data_root, allowlist_path=args.allowlist, folds_path=args.folds, dev_source=args.dev_source,
                            sealed_repo=args.sealed_repo, synthetic=args.synthetic, l_overrides=json.loads(args.l_overrides.read_text(encoding='utf-8')) if args.l_overrides else None, device=args.device, argv=sys.argv if argv is None else ['eval_r3.py', *argv], bz_go=args.bz_go, control_only=args.control_only)
    except (r3_lock.ReadLocked, r3_core.Refused) as exc:
        print(f'STOP: {exc}', flush=True)
        return 8
    print(json.dumps(dict(means={a: {m: document['means'][a][m] for m in ('seg35', 'seg4')} for a in document['arms'][:2]}, control=document['control']), indent=1))
    return 0 if (document['control']['passed'] or not args.control_only) else 8


if __name__ == '__main__':
    sys.exit(main())
