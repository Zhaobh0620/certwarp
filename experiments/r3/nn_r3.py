"""N of work package R3 (runs/20261111-r3/plan.md, section N): nnU-Net 3d_fullres with NN's recipe (runs/20261105-nn: nnUNetPlans, 250 epochs, no mirroring, no TTA, no compile, 6 augmentation
processes), trained on our two folds (splits_final.json: fold 0 = A, fold 1 = B; 98 train + 8 val each, folds_full_v2.json), two models per fold with four distinct seeds of the initial weights
(ext_trainers/nn_trainers_r3.py), and their inference with probabilities (infer_r3nn.py: the mean of a set of models per subject, float16).  This file runs in sci310 and imports no nnunetv2; every
nnU-Net command is a child in the environment nnunet (nn_common.nn_env with this package's trainers).  NN's driver experiments/nn/run_nn.py is reused, not copied: run_watched, training_progress,
check_resume, log_span_seconds, file_sha256 and, for the preparation of the root, its own commands (`run_nn.py prepare --out <NN_ROOT>`, `run_nn.py preprocess --out <NN_ROOT>`).

Only images and the training set's labels (through nnU-Net's own training, of the 106 training subjects) are read here; no development truth, no lock.
"""
import json
from pathlib import Path
import re
import shlex
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _p in (EXPERIMENTS/'unv2', EXPERIMENTS/'fl', EXPERIMENTS/'ef', EXPERIMENTS/'mc0', EXPERIMENTS/'nn', EXPERIMENTS/'r2', HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import nn_common as nc  # noqa: E402
import r3_common as r3c  # noqa: E402
import run_nn  # noqa: E402

EXT_TRAINERS = HERE/'ext_trainers'
INFER_SCRIPT = HERE/'infer_r3nn.py'
Stop = run_nn.Stop
SPLITS_FILE = 'splits_final.json'
SPLIT_LINE = re.compile(r'This split has (\d+) training and (\d+) validation cases')
SPLIT_FILE_LINE = re.compile(r'Using splits from existing split file:\s*(\S+)')
TRAIN_LOSS = re.compile(r': train_loss (\S+)')
VAL_LOSS = re.compile(r': val_loss (\S+)')
SEED_LINE = re.compile(r'R3: torch, numpy and random seeded with (\d+) before the network is built')
EPOCH_START = re.compile(r': Epoch (\d+)\s*$', re.MULTILINE)       # nnUNetTrainer.on_train_epoch_start, after the previous epoch's on_epoch_end has saved its checkpoints
SMOKE_EPOCHS = 4                                  # NN's: the rule on the median from the 2nd epoch on
NN_LIMIT_HOURS = 8.                              # 250 epochs at the smoke's median: more than this stops before the chain (the step's wall stop is 10 h)
HOST_LIMIT_GIB, GPU_LIMIT_MIB = run_nn.HOST_LIMIT_GIB, run_nn.GPU_LIMIT_MIB
SMOKE_CHECKPOINT = 'checkpoint_best.pth'          # after 4 epochs nnU-Net has written checkpoint_best.pth (epoch 0 sets the first best EMA), no checkpoint_final.pth


def nn_env(root=r3c.NN_ROOT, *, gpu, results=None, base=None):
    """nn_common.nn_env of the R3 root, with nnUNet_extTrainer at this package's trainers (the code tree this file runs from: the snapshot in a unit)."""
    env = nc.nn_env(root, results=results, gpu=gpu, base=base)
    env['nnUNet_extTrainer'] = str(EXT_TRAINERS)
    return env


def results_root(root=r3c.NN_ROOT):
    return nc.results_root(root)


def model_dir(name, results):
    fold, seed = r3c.nn_model_of(name)
    return nc.trainer_directory(r3c.nn_trainer(seed), results=results)


def fold_dir(name, results):
    fold, _ = r3c.nn_model_of(name)
    return model_dir(name, results)/f'fold_{r3c.NN_FOLD_INDEX[fold]}'


def splits_path(root=r3c.NN_ROOT):
    return nc.preprocessed_root(root)/nc.DATASET/SPLITS_FILE


# ---- the splits and the plans -------------------------------------------------------------------------------------------------------------------------------
def splits_document(split):
    """nnU-Net's splits_final.json from folds_full_v2.json: [fold 0 = A, fold 1 = B], each {'train': the fold's 98, 'val': its 8 calibration subjects}, sorted."""
    return [dict(train=split.subjects(fold, 'train'), val=split.subjects(fold, 'calib')) for fold in r3c.FOLDS]


def splits_problems(document, identifiers):
    """Each fold: train and val disjoint, together every preprocessed case; the two val sets disjoint."""
    problems = []
    every = sorted(identifiers)
    for index, fold in enumerate(document):
        train, val = fold.get('train') or [], fold.get('val') or []
        if set(train) & set(val):
            problems.append(f'fold {index}: train and val overlap')
        if sorted(train+val) != every:
            problems.append(f'fold {index}: train and val are not the {len(every)} preprocessed cases')
    if len(document) != len(r3c.FOLDS):
        problems.append(f'{len(document)} folds, not {len(r3c.FOLDS)}')
    elif set(document[0]['val']) & set(document[1]['val']):
        problems.append('the two val sets overlap')
    return problems


def write_splits(root, split, *, identifiers=None):
    """<preprocessed>/Dataset935_OASIS35/splits_final.json, once (a file that is there must have the same bytes).  Returns its sha256."""
    identifiers = run_nn.preprocessed_identifiers(root) if identifiers is None else identifiers
    document = splits_document(split)
    problems = splits_problems(document, identifiers)
    if problems:
        raise Stop(f'splits_final.json refused: {problems}')
    text = json.dumps(document, indent=1)+'\n'
    path = splits_path(root)
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise Stop(f'{path} exists with other content; never overwritten')
    else:
        r3c.write_text_once(path, text)
    return r3c.sha256_file(path)


def plans_problems(root=r3c.NN_ROOT, reference=r3c.NN_REFERENCE):
    """The recipe: R3's nnUNetPlans.json is NN's (the same planner on the same 106 subjects); the top-level keys that differ, else []."""
    mine, theirs = (nc.preprocessed_root(r)/nc.DATASET/f'{nc.PLANS}.json' for r in (root, reference))
    if not mine.is_file() or not theirs.is_file():
        return [f'no plans: {mine if not mine.is_file() else theirs}']
    a, b = (json.loads(p.read_text(encoding='utf-8')) for p in (mine, theirs))
    different = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return [f'nnUNetPlans.json differs from NN\'s on {different}'] if different else []


# ---- training ------------------------------------------------------------------------------------------------------------------------------------------------
def train_argv(name, *, resume=False, device='cuda'):
    fold, seed = r3c.nn_model_of(name)
    return [str(nc.NN_BIN/'nnUNetv2_train'), str(nc.DATASET_ID), nc.CONFIGURATION, str(r3c.NN_FOLD_INDEX[fold]), '-tr', r3c.nn_trainer(seed), '-p', nc.PLANS, '-device', device, *(['--c'] if resume else [])]


def _float(text):
    try:
        return float(text)
    except ValueError:
        return None


def log_facts(fold_directory, since=0.):
    """What the training logs written since `since` say: epoch seconds, the split nnU-Net used (counts, file), the seed line, the losses (None where not a number), the final validation begun."""
    texts = [p.read_text(encoding='utf-8', errors='replace') for p in sorted(Path(fold_directory).glob('training_log_*.txt')) if p.stat().st_mtime >= since]
    text = '\n'.join(texts)
    return dict(epoch_seconds=[float(v) for v in run_nn.EPOCH_TIME.findall(text)], validating=run_nn.VALIDATION_START.search(text) is not None,
                splits=[(int(a), int(b)) for a, b in SPLIT_LINE.findall(text)], split_files=sorted(set(SPLIT_FILE_LINE.findall(text))), seeds=sorted({int(v) for v in SEED_LINE.findall(text)}),
                train_losses=[_float(v) for v in TRAIN_LOSS.findall(text)], val_losses=[_float(v) for v in VAL_LOSS.findall(text)], n_logs=len(texts),
                wall_seconds=sum(run_nn.log_span_seconds(t) for t in texts))


def train_model(name, *, root=r3c.NN_ROOT, env, resume=False, code_commit=None, device='cuda', poll=run_nn.WATCH_SECONDS, wall_seconds=None, log=print):
    """One run of nnUNetv2_train for model `name` (run_nn.train_once's way): a started record, the run watched, stopped at nnU-Net's first `predicting` line (the final validation on the fold's 8
    calibration subjects is not run: the composition reads them once, by its own rule), or stopped past NN_WALL_SECONDS (recorded, Stop); its record; then the model's train_record.json if
    checkpoint_final.pth is there.  Stop when the run failed."""
    root = Path(root)
    fd = fold_dir(name, results_root(root))
    run_nn.check_resume(fd, resume)
    since = time.time()-run_nn.LOG_MARGIN_SECONDS
    stamp = run_nn.stamp()
    base = dict(model=name, trainer=r3c.nn_trainer(r3c.nn_model_of(name)[1]), fold=r3c.NN_FOLD_INDEX[r3c.nn_model_of(name)[0]], resumed=resume, code_commit=code_commit, started_utc=run_nn.utc())
    r3c.write_json_once(root/'train_runs'/name/f'{stamp}.started.json', base)
    argv = train_argv(name, resume=resume, device=device)
    log(f'nn train {name}: {shlex.join(argv)}')
    clock = time.monotonic()
    wall = r3c.NN_WALL_SECONDS if wall_seconds is None else wall_seconds

    def should_stop():                              # the final validation begins, or the wall of the model (nnUNetv2_train has its own session: the step's limit would not reach it; the review's m2)
        return run_nn.training_progress(fd, since)['validating'] or time.monotonic()-clock > wall
    outcome = run_nn.run_watched(argv, env=env, log_path=root/'logs'/f'train_{name}_{stamp}.log', cwd=root, should_stop=should_stop, sample_gpu=device == 'cuda', poll=poll)
    progress = run_nn.training_progress(fd, since)
    stopped = outcome['stopped'] == 'condition' and progress['validating']
    over_wall = outcome['stopped'] == 'condition' and not progress['validating']
    r3c.write_json_once(root/'train_runs'/name/f'{stamp}.json', dict(base, outcome=outcome, epoch_seconds=progress['epoch_seconds'], stopped_before_final_validation=stopped, stopped_at_the_wall=over_wall,
                                                                       wall_seconds_limit=wall))
    if over_wall:
        raise Stop(f'nnUNetv2_train {name} ran past its wall of {wall} s and was stopped; log {outcome["log"]}; kept, not rerun (the lead decides; --continue goes on from the last checkpoint)')
    if not stopped and outcome['returncode'] != 0:
        raise Stop(f"nnUNetv2_train {name} failed (exit {outcome['returncode']}); log {outcome['log']}; kept, not rerun (the lead may continue it: run_r3.py nn-train --model {name} --continue)")
    if not (fd/nc.CHECKPOINT).is_file():
        raise Stop(f'{fd/nc.CHECKPOINT}: no such file after the run; log {outcome["log"]}; kept')
    return model_record(name, root=root)


def model_record(name, *, root=r3c.NN_ROOT):
    """<root>/models/<name>/train_record.json, once: the trainer, fold and seed, the final checkpoint's sha256, what all its training logs say, the runs, splits_final.json's sha256 now."""
    root = Path(root)
    fold, seed = r3c.nn_model_of(name)
    fd = fold_dir(name, results_root(root))
    final = fd/nc.CHECKPOINT
    facts = log_facts(fd)
    runs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'train_runs'/name).glob('*.json')) if not p.name.endswith('.started.json')]
    started = sorted((root/'train_runs'/name).glob('*.started.json'))
    peaks = [r['outcome'] for r in runs]
    document = dict(schema='bayesreg.r3.nn_train_record.v1', model=name, trainer=r3c.nn_trainer(seed), fold=fold, fold_index=r3c.NN_FOLD_INDEX[fold], seed=seed, configuration=nc.CONFIGURATION, plans=nc.PLANS,
                    final_checkpoint=str(final), final_checkpoint_sha256=r3c.sha256_file(final), final_checkpoint_bytes=final.stat().st_size, checkpoints=run_nn.checkpoints_in(fd),
                    epochs_logged=len(facts['epoch_seconds']), epoch_seconds_median=statistics.median(facts['epoch_seconds']) if facts['epoch_seconds'] else None, wall_seconds=facts['wall_seconds'],
                    splits=facts['splits'], split_files=facts['split_files'], seeds=facts['seeds'], losses_finite=bool(facts['train_losses']) and all(v is not None and v == v and abs(v) != float('inf') for v in facts['train_losses']+facts['val_losses']),
                    n_train_losses=len(facts['train_losses']), last_train_loss=facts['train_losses'][-1] if facts['train_losses'] else None, last_val_loss=facts['val_losses'][-1] if facts['val_losses'] else None,
                    runs_started=len(started), runs_recorded=len(runs), stopped_before_final_validation=bool(runs) and runs[-1].get('stopped_before_final_validation') is True,
                    host_peak_pss_gib=max((p['host_peak_pss_gib'] for p in peaks), default=None), gpu_peak_used_mib=max(((p['gpu_peak_used_mib'] or 0) for p in peaks), default=None),
                    splits_sha256=r3c.sha256_file(splits_path(root)) if splits_path(root).is_file() else None,
                    validation_files=sorted(str(p.relative_to(fd)) for p in (fd/'validation').rglob('*')) if (fd/'validation').is_dir() else [], created_utc=run_nn.utc())
    r3c.write_json_once(root/'models'/name/'train_record.json', document)
    return document


def expected_split(root, fold):
    """(n train, n val) of the fold in splits_final.json (98, 8 on real data: members_plan.json checks the file against folds_full_v2.json); (None, None) without the file."""
    path = splits_path(root)
    if not path.is_file():
        return (None, None)
    document = json.loads(path.read_text(encoding='utf-8'))[r3c.NN_FOLD_INDEX[fold]]
    return (len(document['train']), len(document['val']))


def model_problems(record, name, *, splits_sha256, root=r3c.NN_ROOT, epochs=r3c.NN_EPOCHS):
    """Why a model's train_record.json is not the plan's (empty: it is): trainer, fold, seed; the split nnU-Net logged is our file with 98 + 8; the seed line; 250 epochs logged; finite losses;
    the final checkpoint the file now; stopped before the final validation, and no validation/summary.json (written or recorded); splits_final.json the pinned one."""
    fold, seed = r3c.nn_model_of(name)
    problems = []
    if not record:
        return [f'{name}: no train_record.json']
    expected = expected_split(root, fold)
    if (record.get('trainer'), record.get('fold'), record.get('seed'), record.get('fold_index')) != (r3c.nn_trainer(seed), fold, seed, r3c.NN_FOLD_INDEX[fold]):
        problems.append(f"{name}: trainer/fold/seed {record.get('trainer')} {record.get('fold')} {record.get('seed')}")
    if not record.get('splits') or any(tuple(s) != expected for s in record['splits']):
        problems.append(f"{name}: nnU-Net's split {record.get('splits')}, not {expected} training and validation cases (splits_final.json)")
    if record.get('split_files') != [str(splits_path(root))]:
        problems.append(f"{name}: split file {record.get('split_files')}, not {splits_path(root)}")
    if record.get('seeds') != [seed]:
        problems.append(f"{name}: the seed line says {record.get('seeds')}, not [{seed}]")
    if (record.get('epochs_logged') or 0) < epochs:
        problems.append(f"{name}: {record.get('epochs_logged')} epochs logged, not {epochs}")
    if record.get('losses_finite') is not True:
        problems.append(f'{name}: a logged loss is not finite')
    if record.get('stopped_before_final_validation') is not True:
        problems.append(f'{name}: not stopped at the final validation')
    if record.get('splits_sha256') != splits_sha256:
        problems.append(f"{name}: splits_final.json {record.get('splits_sha256')} is not the pinned {splits_sha256}")
    final = Path(record.get('final_checkpoint') or '/nonexistent')
    if not final.is_file() or r3c.sha256_file(final) != record.get('final_checkpoint_sha256'):
        problems.append(f'{name}: checkpoint_final.pth missing or not the recorded file')
    if 'summary.json' in (record.get('validation_files') or []) or (final.parent/'validation'/'summary.json').exists():
        problems.append(f'{name}: validation/summary.json is there: nnU-Net\'s final validation ran (a read of the calibration subjects outside the rule; the review\'s m1)')
    return problems


# ---- inference ----------------------------------------------------------------------------------------------------------------------------------------------
def infer_argv(models, *, input_dir, out_dir, root=r3c.NN_ROOT, device='cuda', results=None, checkpoint=nc.CHECKPOINT):
    """infer_r3nn.py in the environment nnunet; `results` another results root (the smoke's), `checkpoint` another file name (the smoke's checkpoint_best.pth)."""
    argv = [str(nc.NN_PY), '-I', '-B', str(INFER_SCRIPT)]
    for name in models:
        fold, _ = r3c.nn_model_of(name)
        argv += ['--model', f'{name}={model_dir(name, results_root(root) if results is None else results)}:{r3c.NN_FOLD_INDEX[fold]}']
    return argv+['--input-dir', str(input_dir), '--out-dir', str(out_dir), '--device', device]+([] if checkpoint == nc.CHECKPOINT else ['--checkpoint-name', checkpoint])


def link_images(directory, subjects, data_root):
    """<directory>/<subject>_0000.nii.gz -> the image, for the subjects of the set (run_nn.build_dev_input, whose words say 'development subjects'; here they may be calibration subjects)."""
    try:
        return run_nn.build_dev_input(directory, subjects, data_root)
    except run_nn.Stop as exc:
        raise Stop(f"the links to the images of the set's subjects: {exc}") from None


def infer_set(out_dir, subjects, models, *, root=r3c.NN_ROOT, env, data_root=nc.DATA_ROOT, argv=None, wall_seconds=None, records=None, checkpoint=nc.CHECKPOINT, poll=run_nn.WATCH_SECONDS, log=print):
    """One set: links to the images of `subjects` only (<out_dir>/input), the child once under run_nn.run_watched (its host PSS and GPU sampled; stopped past r3c.NN_INFER_WALL_SECONDS, which
    is shorter than the step's limit: the review's m9), never rerun (an out_dir with a record or an input is refused), then check_outputs.  Returns the check, with the run's outcome under 'run'."""
    out_dir = Path(out_dir)
    if (out_dir/'input').exists() or (out_dir/'infer_record.json').exists():
        raise Stop(f'{out_dir}: the inference started before; never rerun, the lead decides')
    link_images(out_dir/'input', subjects, data_root)
    argv = infer_argv(models, input_dir=out_dir/'input', out_dir=out_dir, root=root, checkpoint=checkpoint) if argv is None else argv
    log(f'nn infer {out_dir.name}: {shlex.join(str(a) for a in argv)}')
    wall = r3c.NN_INFER_WALL_SECONDS if wall_seconds is None else wall_seconds
    clock = time.monotonic()
    outcome = run_nn.run_watched([str(a) for a in argv], env=env, log_path=out_dir/'infer.log', cwd=out_dir, should_stop=lambda: time.monotonic()-clock > wall, sample_gpu=True, poll=poll)
    r3c.write_json_once(out_dir/'infer_run.json', dict(outcome=outcome, wall_seconds_limit=wall))
    if outcome['stopped'] is not None or outcome['returncode'] != 0:
        raise Stop(f"the nnU-Net inference {out_dir} exited {outcome['returncode']}, stopped {outcome['stopped']!r} ({outcome['stop_reason']}); kept, not rerun")
    check = check_outputs(out_dir, subjects, models, root=root, data_root=data_root, records=records, checkpoint=checkpoint, log=log)
    return dict(check, run=dict(wall_seconds=outcome['wall_seconds'], host_peak_pss_gib=outcome['host_peak_pss_gib'], gpu_peak_used_mib=outcome['gpu_peak_used_mib']))


def check_outputs(out_dir, subjects, models, *, root=r3c.NN_ROOT, data_root=nc.DATA_ROOT, records=None, checkpoint=nc.CHECKPOINT, log=print):
    """The set's outputs against the plan: infer_record.json names the models' recorded checkpoints (sha256, trainer, fold), no mirroring, one view, the device cap, the subjects; per subject the
    files are those subject.json hashed, the stored grid is the image's (its NIfTI header, in load_scalar's order), the reader's argmax of probs36.npz (read_back_probabilities) is
    labels.npz's `mean` on every voxel, and each model's labels are nnU-Net's own label file (nx_seg.read_labels_file, through the NIfTI header) on all but <= 1e-4 of the voxels (the orientation
    check).  `checkpoint`: the file name the record must name (the smoke's checkpoint_best.pth; records then hold its sha256).  <out_dir>/check.json, written once; Stop when anything fails."""
    import numpy as np
    import nx_seg
    from bayesreg.seg.channels import read_back_probabilities
    out_dir = Path(out_dir)
    problems, rows = [], {}
    record = r3c.read_json(out_dir/'infer_record.json') or {}
    records = {m: r3c.read_json(Path(root)/'models'/m/'train_record.json') for m in models} if records is None else records
    if sorted(record.get('models') or {}) != sorted(models) or sorted(record.get('subjects') or []) != sorted(subjects):
        problems.append(f"infer_record.json: models {sorted(record.get('models') or {})}, {len(record.get('subjects') or [])} subjects")
    for name in models:
        entry = (record.get('models') or {}).get(name) or {}
        fold, seed = r3c.nn_model_of(name)
        if entry.get('checkpoint_sha256') != (records.get(name) or {}).get('final_checkpoint_sha256'):
            problems.append(f'{name}: the checkpoint inferred is not the one trained')
        if (entry.get('trainer'), entry.get('checkpoint_fold')) != (r3c.nn_trainer(seed), r3c.NN_FOLD_INDEX[fold]):
            problems.append(f"{name}: trainer {entry.get('trainer')} fold {entry.get('checkpoint_fold')}")
        if entry.get('allowed_mirroring_axes') is not None or entry.get('tta_views') != 1:
            problems.append(f'{name}: mirroring axes or more than one view')
        if entry.get('checkpoint_name', nc.CHECKPOINT) != checkpoint:
            problems.append(f"{name}: checkpoint {entry.get('checkpoint_name')}, not {checkpoint}")
    if (record.get('gpu_peak_reserved_gib') or 0.) > r3c.NN_DEVICE_CAP_GIB:
        problems.append(f'reserved peak above {r3c.NN_DEVICE_CAP_GIB:g} GiB')
    for subject in subjects if not problems else []:
        target = out_dir/subject
        row = r3c.read_json(target/'subject.json') or {}
        files = dict(probs36_sha256=target/'probs36.npz', labels_sha256=target/'labels.npz')
        if any(not path.is_file() or r3c.sha256_file(path) != row.get(key) for key, path in files.items()):
            problems.append(f'{subject}: probs36.npz or labels.npz missing or not the file subject.json hashed')
            continue
        with np.load(target/'labels.npz', allow_pickle=False) as npz:
            labels = {k: npz[k] for k in npz.files}
        if sorted(labels) != sorted(['mean', *models]):
            problems.append(f'{subject}: labels.npz holds {sorted(labels)}')
            continue
        shape = labels['mean'].shape
        image = Path(data_root)/subject/nx_seg.IMAGE_FILE
        grid = image_grid(image)
        if tuple(shape) != grid or any(labels[m].shape != grid for m in models):
            problems.append(f'{subject}: the stored grid {tuple(shape)} is not the image\'s {grid} (load_scalar\'s order)')
            continue
        back, _ = read_back_probabilities(target/'probs36.npz', shape=(nx_seg.N_CLASSES,)+tuple(shape))
        n_mean = int((back.argmax(axis=0) != labels['mean']).sum())
        del back
        if n_mean:
            problems.append(f"{subject}: the reader's argmax of probs36.npz differs from the argmax of the mean on {n_mean} voxels")
        counts = {}
        for name in models:
            own, _ = nx_seg.read_labels_file(target/f'seg_{name}.nii.gz', image)
            counts[name] = int((own != labels[name]).sum())
            if counts[name]/own.size > r3c.NN_GATE_TOLERANCE:
                problems.append(f'{subject} {name}: the stored labels differ from nnU-Net\'s own label file on {counts[name]} voxels (> {r3c.NN_GATE_TOLERANCE:g}): orientation')
        rows[subject] = dict(n_reader_not_mean=n_mean, n_model_not_own=counts, n_voxels=int(np.prod(shape)), shape=list(shape), probs36_sha256=row['probs36_sha256'], labels_sha256=row['labels_sha256'],
                             image=str(image), seconds=row.get('seconds'))
        log(f'{subject}: reader = mean on every voxel: {n_mean == 0}; model labels against their own files {counts}')
    document = dict(schema='bayesreg.r3.nn_check.v1', passed=not problems, problems=problems, models=list(models), subjects=list(subjects), rows=rows, tolerance=r3c.NN_GATE_TOLERANCE,
                    infer_record_sha256=r3c.sha256_file(out_dir/'infer_record.json') if (out_dir/'infer_record.json').is_file() else None,
                    checkpoints={m: (records.get(m) or {}).get('final_checkpoint_sha256') for m in models}, utc=run_nn.utc())
    r3c.write_json_once(out_dir/'check.json', document)
    if problems:
        raise Stop(f'the nnU-Net outputs of {out_dir} are not the plan\'s: {problems[:5]}; kept')
    return document


def image_grid(path):
    """(nz, ny, nx) of an image in load_scalar's order, from its NIfTI header only: input axis i goes to RAS axis ornt[i, 0] (nibabel's apply_orientation, as nx_seg.orient_native), then reversed."""
    import nibabel as nib
    image = nib.load(str(path))
    ornt = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine), nib.orientations.axcodes2ornt(('R', 'A', 'S')))
    ras = [0, 0, 0]
    for i, (axis, _) in enumerate(ornt):
        ras[int(axis)] = image.shape[i]
    return tuple(ras[::-1])


def read_check(out_dir, subjects, models):
    """check.json of a set, passed, for exactly these subjects and models; Stop otherwise."""
    document = r3c.read_json(Path(out_dir)/'check.json')
    if not document or document.get('passed') is not True or sorted(document.get('subjects') or []) != sorted(subjects) or list(document.get('models') or []) != list(models):
        raise Stop(f'{out_dir}/check.json: not a passed check of {len(subjects)} subjects and the models {list(models)}')
    return document


# ---- the smoke (smoke_r3.py) --------------------------------------------------------------------------------------------------------------------------------
def smoke(smoke_root, *, root=r3c.NN_ROOT, name='nnA_s0', epochs=SMOKE_EPOCHS, env=None, device='cuda', poll=run_nn.WATCH_SECONDS):
    """The real trainer of model `name` on the real preprocessed data, in the smoke's own results root, stopped when epoch `epochs` starts (so `epochs` epochs and their checkpoints are
    complete).  Returns what it measured (nothing judged here)."""
    results = Path(smoke_root)/'nnUNet_results_smoke'
    if results.exists():
        raise Stop(f'{results} exists: the nnU-Net smoke ran; kept, not rerun')
    fd = fold_dir(name, results)
    env = nn_env(root, results=results, gpu=device == 'cuda') if env is None else env
    since = time.time()-run_nn.LOG_MARGIN_SECONDS
    def started_epoch(k):                           # the line of epoch k's start comes after epoch k-1's checkpoints were saved: stopping there leaves no half-written checkpoint_best.pth
        texts = [p.read_text(encoding='utf-8', errors='replace') for p in sorted(fd.glob('training_log_*.txt')) if p.stat().st_mtime >= since]
        return any(int(v) >= k for text in texts for v in EPOCH_START.findall(text))
    outcome = run_nn.run_watched(train_argv(name, device=device), env=env, log_path=Path(smoke_root)/'logs'/'nn_smoke.log', cwd=smoke_root,
                                 should_stop=lambda: started_epoch(epochs), sample_gpu=device == 'cuda', poll=poll)
    facts = log_facts(fd, since)
    return dict(model=name, trainer=r3c.nn_trainer(r3c.nn_model_of(name)[1]), epochs_asked=epochs, outcome=outcome, stopped=outcome['stopped'], epoch_seconds=facts['epoch_seconds'], splits=facts['splits'],
                split_files=facts['split_files'], seeds=facts['seeds'], train_losses=facts['train_losses'], val_losses=facts['val_losses'], host_peak_pss_gib=outcome['host_peak_pss_gib'],
                gpu_peak_used_mib=outcome['gpu_peak_used_mib'], gpu_mean_utilization_percent=outcome['gpu_mean_utilization_percent'], checkpoint_bytes={p: (fd/p).stat().st_size for p in run_nn.checkpoints_in(fd)},
                checkpoint_sha256={p: r3c.sha256_file(fd/p) for p in run_nn.checkpoints_in(fd)}, n_proc_da=int(env['nnUNet_n_proc_DA']), splits_file=str(splits_path(root)))


def smoke_reading(measured, *, root=r3c.NN_ROOT, epochs=SMOKE_EPOCHS, limit_hours=NN_LIMIT_HOURS):
    """('pass' or 'stop', reasons): `epochs` epochs and stopped by the condition; nnU-Net used our splits_final.json with 98 + 8 cases; the seed line; finite losses; 250 x the median epoch from the
    2nd on <= limit_hours; host PSS <= 11 GiB and GPU used <= 22528 MiB (NN's budgets)."""
    why = []
    seconds = measured.get('epoch_seconds') or []
    if len(seconds) < epochs or measured.get('stopped') != 'condition':
        why.append(f"{len(seconds)} epochs, stopped {measured.get('stopped')!r}: not the {epochs} epochs asked")
    expected = expected_split(root, r3c.nn_model_of(measured.get('model', 'nnA_s0'))[0])
    if not measured.get('splits') or any(tuple(s) != expected for s in measured['splits']) or measured.get('split_files') != [str(splits_path(root))]:
        why.append(f"nnU-Net's split {measured.get('splits')} from {measured.get('split_files')}, not {expected} from {splits_path(root)}")
    if measured.get('seeds') != [r3c.nn_model_of(measured.get('model', 'nnA_s0'))[1]]:
        why.append(f"the seed line says {measured.get('seeds')}")
    losses = (measured.get('train_losses') or [])+(measured.get('val_losses') or [])
    if not losses or any(v is None or v != v or abs(v) == float('inf') for v in losses):
        why.append('a logged loss is not finite (or none was logged)')
    hours = r3c.NN_EPOCHS*statistics.median(seconds[1:])/3600 if len(seconds) >= 2 else None
    if hours is None or hours > limit_hours:
        why.append(f'{r3c.NN_EPOCHS} epochs at the median of the epochs from the 2nd on: {hours} h > {limit_hours} h')
    if measured.get('host_peak_pss_gib') is None or measured['host_peak_pss_gib'] > HOST_LIMIT_GIB:
        why.append(f"host peak {measured.get('host_peak_pss_gib')} GiB above {HOST_LIMIT_GIB} GiB")
    if measured.get('gpu_peak_used_mib') is None or measured['gpu_peak_used_mib'] > GPU_LIMIT_MIB:
        why.append(f"GPU used peak {measured.get('gpu_peak_used_mib')} MiB above {GPU_LIMIT_MIB} MiB")
    return ('stop' if why else 'pass'), why, hours
