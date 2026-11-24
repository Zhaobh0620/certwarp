"""R3's composition on the training set (runs/20261111-r3/plan.md section 4; the brief's step 2; BZ 11-11 ~2:22 PM CST, hard rule 6's reading (b): the training set's calibration subjects may
be read at the segmentation level to decide the composition of the segmentation side; never for the registration method's hyperparameters, never in place of the development reading).

Out of fold only: fold A's 8 calibration subjects are read with fold A's members alone (UF foldA_s0, foldA_s1, L foldA_s3, nnU-Net nnA_s0, nnA_s1 -- none of them trained on these subjects), fold
B's likewise; 16 subjects.  The candidates (r3_common.candidate_order: five, and the same five with D when the switch is on) and the rule (r3_common.choose) are written in the plan before any reading.
Read once: the lock <SEG>/r3_calib_read.lock is taken immediately before the first truth label; every check comes before it (the members as planned, the nnU-Net check of the fold's set, the decoding
switch, the folds, the subjects out of every member's training set).  The truth through LabelGuard (bayesreg.seg.unet.data.read_labels: B-line training subjects only).

Arms (one pass per subject, no prediction kept): the candidates; described only: UF_s0, UF_s1, L (each member alone, argmax of its logits), N (the fold's nnU-Net mean), nn_s<k> (each model alone).
Output <out-dir>/composition.json (the rule's result, the composition delivered: the winner's type over both folds), per_subject.csv, summary_compose.md; then the chain STOPs (the lead).

    python -B experiments/r3/compose_r3.py --out-dir DIR --splits-sha256 SHA [--device cuda] [--bz-go TEXT]
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
import nn_r3  # noqa: E402
import r3_common as r3c  # noqa: E402
import r3_core  # noqa: E402
import r3_lock  # noqa: E402
from bayesreg.seg.unet.data import LabelGuard, OasisFiles, load_full_folds, normalize_image, read_image, read_labels  # noqa: E402
from bayesreg.seg.unet.large import GIB, cap_gpu_memory  # noqa: E402
from bayesreg.seg.unet.pins import DEFAULT_ALLOWLIST, DEFAULT_FULL_FOLDS, TRAINING_SOURCE, load_pinned_full_inputs, load_synthetic_inputs  # noqa: E402
from bayesreg.seg.unet.snapshot import code_record, require_clean_code  # noqa: E402

FORMAT = 'bayesreg.r3.compose.v1'
GPU_CAP_GIB = 20.
METRICS = eval_nn.METRICS
U_SETS = {'UF': 'U2', 'UF+L': 'U3', 'Lswap': 'Uswap', 'UF+N': 'U2', 'UF+L+N': 'U3'}       # the U-Net family of each base
DATA_ROOT = Path('/mnt/d/data')


def fold_keys(fold):
    """The fold's U-Net members in their order of addition, and the U-Net sets of the candidates (each a subsequence)."""
    s0, s1, l3 = ('uf', f'fold{fold}_s0'), ('uf', f'fold{fold}_s1'), (r3c.L_FAMILY, r3c.member_name(fold, r3c.L_SEED))
    sets = dict(U2=[s0, s1], U3=[s0, s1, l3], Uswap=[s0, l3])
    for base in r3c.BASE_ORDER:
        if sets[U_SETS[base]] != r3c.unet_members(base, fold):
            raise AssertionError(f'{base}: {sets[U_SETS[base]]} is not r3_common.unet_members {r3c.unet_members(base, fold)}')
    return [s0, s1, l3], sets, dict(UF_s0=s0, UF_s1=s1, L=l3)


def described_arms():
    return ('UF_s0', 'UF_s1', 'L', 'N')+tuple(f'nn_s{k}' for k in range(2))


def out_of_fold_problems(subject, fold, records, nn_splits):
    """The subject is a calibration subject of `fold`, in no training set of the fold's members (their train records) nor of the fold's nnU-Net split (fold index's train)."""
    problems = []
    for key, record in records.items():
        if subject in (record.get('train_subjects') or []) or subject not in (record.get('calib_subjects') or []):
            problems.append(f'{subject}: in the training set of {key} or not among its calibration subjects')
    index = r3c.NN_FOLD_INDEX[fold]
    if subject in nn_splits[index]['train'] or subject not in nn_splits[index]['val']:
        problems.append(f"{subject}: in the nnU-Net split's training set of fold {index} or not in its val")
    return problems


def compose(*, out_dir, uf_root=r3c.UF_OUT, member_root=r3c.OUT, nn_calib=r3c.OUT/'nn_calib', nn_root=r3c.NN_ROOT, switch_path=r3c.OUT/r3c.DECODE_SWITCH, folds_path=DEFAULT_FULL_FOLDS,
            allowlist_path=DEFAULT_ALLOWLIST, data_root=DATA_ROOT, uf_eval=r3c.UF_EVAL, splits_sha256=None, synthetic=False, l_overrides=None, device='cuda', argv=None, bz_go=None, log=print):
    if l_overrides is not None and not synthetic:
        raise ValueError('--l-overrides goes with --synthetic')
    if synthetic and bz_go is not None:
        raise ValueError('--bz-go is for the read lock of real data')
    if not synthetic:
        r3_lock.check('calib', bz_go)                                   # the first thing on real data
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'{out_dir} exists and is not empty; never overwritten')
    code = code_record()
    if synthetic:
        split, allowlist, inputs = load_synthetic_inputs(folds_path, allowlist_path, data_root=data_root, training_source='synthetic_bline_train', loader=load_full_folds)
        guard = LabelGuard(split, allowlist, training_source='synthetic_bline_train')
    else:
        require_clean_code(code)
        if any(Path(a).resolve() != Path(b).resolve() for a, b in ((uf_root, r3c.UF_OUT), (member_root, r3c.OUT), (nn_calib, r3c.OUT/'nn_calib'), (nn_root, r3c.NN_ROOT), (switch_path, r3c.OUT/r3c.DECODE_SWITCH),
                                                                  (folds_path, DEFAULT_FULL_FOLDS), (allowlist_path, DEFAULT_ALLOWLIST), (data_root, DATA_ROOT), (uf_eval, r3c.UF_EVAL))):
            raise r3_core.Refused('real data: every input is the plan\'s; other values only with --synthetic')
        if not splits_sha256:
            raise r3_core.Refused('--splits-sha256 (members_plan.json\'s pin) is required on real data')
        split, allowlist, inputs = load_pinned_full_inputs(folds_path, allowlist_path)
        guard = LabelGuard(split, allowlist, training_source=TRAINING_SOURCE)
        inputs = {k: v for k, v in inputs.items() if k != 'code'}
    import eval_r2
    uf, uf_sha = eval_r2.read_uf_eval(uf_eval, pinned=None if synthetic else r3c.UF_EVAL_SHA256)
    decode_on, spec, switch = r3_core.read_switch(switch_path, synthetic=synthetic)
    order = r3c.candidate_order(decode_on)
    subjects = {fold: split.subjects(fold, 'calib') for fold in r3c.FOLDS}
    if not synthetic and any(len(v) != r3c.N_CALIB_PER_FOLD for v in subjects.values()):
        raise r3_core.Refused(f'calibration subjects per fold: {[len(v) for v in subjects.values()]}, not {r3c.N_CALIB_PER_FOLD}')
    # the nnU-Net side: the pinned split file, each model's record, each fold's check
    split_file = nn_r3.splits_path(nn_root)
    nn_splits = json.loads(split_file.read_text(encoding='utf-8')) if split_file.is_file() else None
    problems = []
    if nn_splits is None or (splits_sha256 is not None and r3c.sha256_file(split_file) != splits_sha256):
        problems.append(f'{split_file}: missing or not the pinned splits_final.json')
    nn_records = {m: r3c.read_json(Path(nn_root)/'models'/m/'train_record.json') for m in r3c.NN_NAMES}
    for m, record in nn_records.items():
        problems += nn_r3.model_problems(record, m, splits_sha256=r3c.sha256_file(split_file) if split_file.is_file() else None, root=nn_root)
    checks = {}
    for fold in r3c.FOLDS:
        try:
            checks[fold] = nn_r3.read_check(Path(nn_calib)/f'fold{fold}', subjects[fold], r3c.nn_models_of_fold(fold))
        except nn_r3.Stop as exc:
            problems.append(str(exc))
        else:
            if checks[fold]['checkpoints'] != {m: (nn_records[m] or {}).get('final_checkpoint_sha256') for m in r3c.nn_models_of_fold(fold)}:
                problems.append(f'fold {fold}: the nnU-Net check is not of the recorded checkpoints')
    device = torch.device(device)
    if device.type == 'cuda':
        cap_gpu_memory(device, GPU_CAP_GIB)
    keys = [k for fold in r3c.FOLDS for k in fold_keys(fold)[0]]
    members, member_problems = r3_core.load_unet(uf_root=uf_root, member_root=member_root, uf_evaluated={n: m['member_sha256'] for n, m in uf['members'].items()}, device=device, keys=keys,
                                                  pinned=None if synthetic else r3c.UF_MEMBER_SHA256, l_config=r3_core.l_expected_config(uf_root, l_overrides))
    problems += member_problems
    records = {f'{f}/{n}': json.loads(((Path(uf_root)/n) if f == 'uf' else Path(member_root)/f/n).joinpath('train_record.json').read_text(encoding='utf-8')) for f, n in keys if (f, n) in members}
    for fold in r3c.FOLDS:
        for subject in subjects[fold]:
            problems += out_of_fold_problems(subject, fold, {k: v for k, v in records.items() if k.split('/')[1].startswith(f'fold{fold}')}, nn_splits or [dict(train=[], val=[])]*2)
    if problems:
        raise r3_core.Refused(f'not composed, no truth read: {problems}')
    percentiles = {blob['percentile'] for _, blob, _ in members.values()}
    if len(percentiles) != 1:
        raise r3_core.Refused(f'the members disagree on the normalization percentile: {percentiles}')
    percentile = percentiles.pop()
    lookup = eval_seg.load_seg4_lookup()
    files = OasisFiles(Path(data_root))
    arms = order+described_arms()
    rows = {a: {} for a in arms}
    decode_records, image_sha, truth_sha, seconds, calib_read = {}, {}, {}, [], None
    for fold in r3c.FOLDS:
        forward_order, sets, singles = fold_keys(fold)
        for subject in subjects[fold]:
            started = time.perf_counter()
            values, meta = read_image(files, subject)
            image_sha[subject] = meta['sha256']
            image, _ = normalize_image(values, percentile=percentile)
            del values
            p16, tops = r3_core.forward(members, forward_order, image, device=device, arms=sets, singles=singles)
            del image
            shape = p16['U2'].shape[1:]
            p_nn, nn_labels = r3_core.read_nn_mean(Path(nn_calib)/f'fold{fold}', subject, shape, row=checks[fold]['rows'][subject])
            tops['N'] = nn_labels['mean']
            for k, m in enumerate(r3c.nn_models_of_fold(fold)):
                tops[f'nn_s{k}'] = nn_labels[m]
            for candidate in order:
                base = r3c.base_of(candidate)
                with_nn = r3c.CANDIDATE_BASES[base]['nn']
                tops[candidate], record = r3_core.candidate_labels(p16[U_SETS[base]], p_nn if with_nn else None, r3c.nn_weight(candidate) if with_nn else 0., spec if r3c.decodes(candidate) else None)
                if record is not None:
                    decode_records.setdefault(candidate, {})[subject] = record
            del p_nn, p16
            if not synthetic and calib_read is None:                    # the moment before the first truth label
                calib_read = r3_lock.take('calib', dict(out_dir=str(out_dir), members={f'{f}/{n}': s['member_sha256'] for (f, n), (_, _, s) in members.items()},
                                                        nn_checkpoints={m: (r or {}).get('final_checkpoint_sha256') for m, r in nn_records.items()}, decode=decode_on, command=argv, code_commit=code.get('commit')), bz_go=bz_go)
            true35, meta35 = read_labels(files, subject, guard, kind='seg35')
            true4, meta4 = read_labels(files, subject, guard, kind='seg4')
            truth_sha[subject] = dict(seg35=meta35['sha256'], seg4=meta4['sha256'])
            for arm in arms:
                rows[arm][subject] = eval_nn.metrics_of(eval_seg.summarize_subject(tops[arm], true35, true4, lookup))
            seconds.append(time.perf_counter()-started)
            log(f"{fold} {subject}: "+', '.join(f"{c} {rows[c][subject]['seg35']:.4f}" for c in order))
    everyone = [s for fold in r3c.FOLDS for s in subjects[fold]]
    means = {a: {m: eval_nn.mean_of([rows[a][s] for s in everyone], m) for m in METRICS} for a in arms}
    fold_means = {fold: {a: {m: eval_nn.mean_of([rows[a][s] for s in subjects[fold]], m) for m in ('seg35', 'seg4')} for a in arms} for fold in r3c.FOLDS}
    result = r3c.choose({c: means[c] for c in order}, order)
    winner = result['winner']
    import decide_r2
    paired = {c: {m: decide_r2.describe(np.array([rows[c][s][m] for s in everyone], dtype=np.float64)-np.array([rows[r3c.REFERENCE][s][m] for s in everyone], dtype=np.float64), draws=r3c.SEG_BOOT_DRAWS, seed=r3c.SEG_BOOT_SEED)
                  for m in ('seg35', 'seg4', 'small_six')} for c in order if c != r3c.REFERENCE}
    document = dict(format=FORMAT, plan=r3c.plan_record(REPO), created_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), command=argv, code=code, synthetic=synthetic, inputs=inputs,
                    subjects=subjects, order=list(order), candidates={c: r3c.delivered(c) for c in order}, decode=dict(on=decode_on, spec=spec, switch=switch), rule=dict(
                        reference=r3c.REFERENCE, seg4_tolerance=r3c.SEG4_TOLERANCE, tie_band=r3c.TIE_BAND, text='16 人 seg35 均值最高者胜；seg4 比 UF 低超过 .001 的不参选；与最高者差不到 .0005 的取最简单的（每折模型更少优先，其次不带 D，再其次 seg35 高，再其次清单次序）'),
                    result=result, winner=winner, no_change=winner == r3c.REFERENCE, delivered=r3c.delivered(winner) if winner else None, means=means, fold_means=fold_means, paired_minus_reference=paired,
                    rows=rows, per_label_mean={a: [eval_seg.nanmean([eval_nn.clean(rows[a][s]['per_label'][k]) for s in everyone]) for k in range(eval_seg.N_CLASSES-1)] for a in arms},
                    decode_records=decode_records, members={f'{f}/{n}': s for (f, n), (_, _, s) in members.items()}, nn=dict(records={m: {k: (r or {}).get(k) for k in ('final_checkpoint_sha256', 'epochs_logged', 'last_train_loss', 'last_val_loss', 'wall_seconds')} for m, r in nn_records.items()},
                    checks={f: dict(sha256=r3c.sha256_file(Path(nn_calib)/f'fold{f}'/'check.json')) for f in r3c.FOLDS}, splits_sha256=r3c.sha256_file(split_file)),
                    uf_eval=dict(path=str(uf_eval), sha256=uf_sha), image_sha256=image_sha, truth_sha256=truth_sha, calib_read=calib_read, label_reads=guard.record()['label_reads'],
                    seconds=dict(per_subject=seconds, total=sum(seconds)), device=str(device), gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/GIB if device.type == 'cuda' else None,
                    limits='每折只有部分成员（U-Net 2–3 个、nnU-Net 2 个，开发集的组合是两折之并，成员数加倍），16 人；UF、L 成员的温度是在这 8 人上标定的；R3b 的标定诊断也读这 16 人。',
                    note='训练集标定受试者，折外，单次读数；只用于决定分割侧的组成（BZ 11-11 的口径），不是开发集结果，不用于配准方法的超参数。')
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir/'per_subject.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['fold', 'subject', 'arm', *METRICS])
        for fold in r3c.FOLDS:
            for s in subjects[fold]:
                for arm in arms:
                    writer.writerow([fold, s, arm, *[eval_nn.clean(rows[arm][s].get(m)) for m in METRICS]])
    r3c.write_text_once(out_dir/'summary_compose.md', summary(document))
    r3c.write_json_once(out_dir/'composition.json', document)          # last
    return document


def _f(v, d=4):
    return 'n/a' if not r3c.finite(v) else f'{v:.{d}f}'


def summary(document):
    L = ['# R3 定组成（compose_r3.py 写的；训练集标定受试者 16 人，折外，单次）', '', document['note'], '', document['limits'], '',
         f"**胜出：{document['winner']}**" + ('（= UF：组成不变，不做开发集读数，停，交总负责人）' if document['no_change'] else f"；交付的组合：{document['delivered']['words']}"), '',
         f"D：{'开' if document['decode']['on'] else '关'}。规则：{document['rule']['text']}。", '', f"可参选 {document['result']['eligible']}；不参选 {document['result']['excluded']}；带内 {document['result']['band']}。", '',
         '| 臂 | seg35 | seg4 | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, m in document['means'].items():
        L.append(f"| {arm} | {_f(m.get('seg35'))} | {_f(m.get('seg4'))} | {_f(m.get('small_six'))} | {_f(m.get('big_six'))} | {_f(m.get('other_23'))} | {_f(m.get('cortex'))} | {_f(m.get('white_matter'))} |")
    return '\n'.join(L)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--uf-root', type=Path, default=r3c.UF_OUT)
    parser.add_argument('--member-root', type=Path, default=r3c.OUT)
    parser.add_argument('--nn-calib', type=Path, default=r3c.OUT/'nn_calib')
    parser.add_argument('--nn-root', type=Path, default=r3c.NN_ROOT)
    parser.add_argument('--switch', type=Path, default=r3c.OUT/r3c.DECODE_SWITCH)
    parser.add_argument('--folds', type=Path, default=DEFAULT_FULL_FOLDS)
    parser.add_argument('--allowlist', type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--uf-eval', type=Path, default=r3c.UF_EVAL)
    parser.add_argument('--splits-sha256', default=None)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--l-overrides', type=Path, default=None, help='--synthetic only: the tiny --set values of L')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--bz-go', default=None)
    args = parser.parse_args(argv)
    try:
        document = compose(out_dir=args.out_dir, uf_root=args.uf_root, member_root=args.member_root, nn_calib=args.nn_calib, nn_root=args.nn_root, switch_path=args.switch, folds_path=args.folds,
                           allowlist_path=args.allowlist, data_root=args.data_root, uf_eval=args.uf_eval, splits_sha256=args.splits_sha256, synthetic=args.synthetic, l_overrides=json.loads(args.l_overrides.read_text(encoding='utf-8')) if args.l_overrides else None, device=args.device,
                           argv=sys.argv if argv is None else ['compose_r3.py', *argv], bz_go=args.bz_go)
    except (r3_lock.ReadLocked, r3_core.Refused) as exc:
        print(f'STOP: {exc}', flush=True)
        return 8
    print(json.dumps({k: document[k] for k in ('winner', 'no_change')} | dict(result={k: document['result'][k] for k in ('best', 'band', 'excluded')}), ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
