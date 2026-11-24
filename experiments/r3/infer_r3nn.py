"""The nnU-Net inference of work package R3 (runs/20261111-r3/plan.md, section N): several models of one set, one subject at a time, the mean of their probabilities kept in memory and only
what the readings need written.  Runs in the conda environment nnunet as `python -I -B infer_r3nn.py ...` (nn_r3.infer_argv; the environment of nn_r3.nn_env); imports nothing of this repository.
Written after experiments/nn/infer_nn.py (the predictor's settings: tile step .5, Gaussian window, use_mirroring=True with the checkpoint's axes, None for these trainers) and
experiments/fl/nx_infer_child.py (the 9 GiB device cap); the orientation and the float16 step are copies of experiments/fl/nx_seg.py's orient_native, orient_probabilities and
float16_keeping_argmax (a test compares them).

Why not infer_nn.py --save-probabilities: it writes every model's float32 probabilities of every subject (about .58 GB a file, UF's 11.65 GB for 20 subjects), which R3 would write about 112
times (16 calibration subjects x 2 models, 20 development subjects x 4) only to average them; here nothing but the mean is written (about .3 GB a subject, float16), and no file is deleted afterwards.

    python -I -B experiments/r3/infer_r3nn.py --model NAME=MODEL_DIR:FOLD [--model ...] --input-dir DIR --out-dir DIR [--device cuda] [--checkpoint-name checkpoint_best.pth]

--checkpoint-name (default checkpoint_final.pth, what the chain uses): the GPU smoke's checkpoint_best.pth, the only one 4 epochs leave (smoke_r3.py; the review's I2).

--input-dir holds <subject>_0000.nii.gz (links to the images, nothing else).  For each subject, in <out-dir>/<subject>/ (never overwritten):
  probs36.npz   key p: the mean over the models of nnU-Net's softmax (predict_single_npy_array, the geometry of the input), in the order of load_scalar (RAS storage, axes zyx), class c = seg35
                label c; float16 with the argmax of the float64 mean kept (one ULP where float16 makes a tie, as UF's conversion); read back by bayesreg.seg.channels.read_back_probabilities
  labels.npz    uint8, the order of load_scalar: `mean` the argmax of the float64 mean; one key per model, the argmax of its own probabilities
  seg_<NAME>.nii.gz   nnU-Net's own label file of each model (its writer, the geometry of the input): the reference of the orientation check in sci310 (nn_r3.check_outputs)
  subject.json  seconds per model, the probabilities' dtype, their largest |channel sum - 1|, nnU-Net's labels against the argmax of its probabilities, the float16 ties, the sha256 of the files
<out-dir>/infer_record.json: per model the checkpoint's sha256, trainer, fold, epoch and mirroring axes; the subjects; the GPU peaks.  Only images are read, never a label.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

N_CLASSES = 36
SUFFIX = '_0000.nii.gz'
CHECKPOINT = 'checkpoint_final.pth'
FORMAT = 'bayesreg.r3.nn_infer.v1'
LIMIT_BYTES = 9*2**30
PROBS_FILE, LABELS_FILE, SUBJECT_FILE, RECORD_FILE = 'probs36.npz', 'labels.npz', 'subject.json', 'infer_record.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ---- copies of experiments/fl/nx_seg.py (that module imports the FX line) -------------------------------------------------------------------------------
def orient_native(values, affine):
    """A native array (the NIfTI file's order) into load_scalar's: RAS storage, axes zyx (nx_seg.orient_native)."""
    import nibabel as nib
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(affine), nib.orientations.axcodes2ornt(('R', 'A', 'S')))
    return nib.orientations.apply_orientation(values, transform).transpose(2, 1, 0)


def orient_probabilities(probabilities, affine):
    """(classes, k, j, i) of SimpleITK's order into (classes, nz, ny, nx) of load_scalar, channel by channel, the dtype kept (nx_seg.orient_probabilities)."""
    if probabilities.ndim != 4 or probabilities.shape[0] != N_CLASSES:
        raise ValueError(f'probabilities of shape {probabilities.shape}: {N_CLASSES} classes on axis 0 expected')
    first = orient_native(probabilities[0].transpose(2, 1, 0), affine)
    out = np.empty((N_CLASSES,)+first.shape, dtype=probabilities.dtype)
    out[0] = first
    for c in range(1, N_CLASSES):
        out[c] = orient_native(probabilities[c].transpose(2, 1, 0), affine)
    return out


def float16_keeping_argmax(p, labels):
    """(float16 array, number of voxels touched) (nx_seg.float16_keeping_argmax)."""
    p16 = p.astype(np.float16)
    wrong = np.argwhere(p16.argmax(axis=0) != labels)
    for z, y, x in wrong:
        column = p16[:, z, y, x]
        label = int(labels[z, y, x])
        if float(column.astype(np.float64).sum()) > 1.:
            tied = np.flatnonzero(column[:label] >= column[label])
            p16[tied, z, y, x] = np.nextafter(column[label], np.float16(0.), dtype=np.float16)
        else:
            p16[label, z, y, x] = np.nextafter(column[label], np.float16(2.), dtype=np.float16)
    return p16, len(wrong)


def save_npz(path, **arrays):
    """Compressed, written then moved (nx_seg.save_npz with named keys)."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    tmp = Path(str(path)+'.tmp')
    tmp.write_bytes(buffer.getvalue())
    tmp.replace(path)


# ---- the work (no nnunetv2 here: the tests and the synthetic chain hand in their own predictors) -----------------------------------------------------------
def run(models, images, out_dir, *, read_image, write_seg, log=print):
    """models: [(name, predict)] with predict(image, properties) -> (labels, probabilities (36, k, j, i)) in SimpleITK's order; images: {subject: path}; read_image(path) -> (image, properties,
    affine of the file); write_seg(labels, path, properties).  Every subject once; returns {subject: record}.  The probabilities of each model are oriented and added in float64 as they come
    (one model's array alive at a time); the mean, its argmax, the float16 file; then the labels and the record."""
    out_dir = Path(out_dir)
    names = [name for name, _ in models]
    if not names or len(set(names)) != len(names) or any(not n.replace('_', '').isalnum() for n in names) or 'mean' in names:
        raise ValueError(f'model names {names}: distinct, alphanumeric (and _), not "mean"')
    rows = {}
    for subject, path in images.items():
        target = out_dir/subject
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f'{target} exists and is not empty; never overwritten')
        target.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        image, properties, affine = read_image(path)
        total, labels, per_model = None, {}, {}
        for name, predict in models:
            clock = time.perf_counter()
            own_labels, probabilities = predict(image, copy.deepcopy(properties))
            seconds = time.perf_counter()-clock
            probabilities = np.asarray(probabilities)
            if probabilities.ndim != 4 or probabilities.shape[0] != N_CLASSES or not np.issubdtype(probabilities.dtype, np.floating):
                raise ValueError(f'{subject} {name}: probabilities {probabilities.dtype} {probabilities.shape}, not {N_CLASSES} float channels')
            if not np.all(np.isfinite(probabilities)):
                raise ValueError(f'{subject} {name}: probabilities not finite')
            deviation = float(np.max(np.abs(probabilities.sum(axis=0, dtype=np.float64)-1.)))
            top = probabilities.argmax(axis=0)
            own_labels = np.asarray(own_labels)
            n_own = int((top != own_labels).sum()) if own_labels.shape == top.shape else None
            write_seg(own_labels, str(target/f'seg_{name}.nii.gz'), properties)
            labels[name] = np.ascontiguousarray(orient_native(top.transpose(2, 1, 0).astype(np.uint8), affine))
            del top, own_labels
            oriented = orient_probabilities(probabilities, affine)
            dtype = str(probabilities.dtype)
            del probabilities
            if total is None:
                total = oriented.astype(np.float64)
            else:
                total += oriented
            del oriented
            per_model[name] = dict(seconds=seconds, dtype=dtype, max_abs_sum_deviation=deviation, n_nnunet_labels_not_argmax=n_own)
        total /= len(models)
        mean_top = total.argmax(axis=0).astype(np.uint8)
        p16, ties = float16_keeping_argmax(total, mean_top)
        del total
        save_npz(target/PROBS_FILE, p=p16)
        shape = list(p16.shape)
        del p16
        save_npz(target/LABELS_FILE, mean=mean_top, **labels)
        row = dict(subject=subject, models=names, per_model=per_model, shape=shape, n_float16_ties_fixed=ties, probs36_sha256=sha256(target/PROBS_FILE), labels_sha256=sha256(target/LABELS_FILE),
                   seg_sha256={n: sha256(target/f'seg_{n}.nii.gz') for n in names}, image=str(path), seconds=time.perf_counter()-started, utc=utc())
        tmp = target/(SUBJECT_FILE+'.tmp')
        tmp.write_text(json.dumps(row, indent=1)+'\n', encoding='utf-8')
        tmp.replace(target/SUBJECT_FILE)
        rows[subject] = row
        timing = ', '.join('%s %.1f s' % (n, per_model[n]['seconds']) for n in names)
        log(f"{subject}: {timing}; {ties} float16 ties; {row['seconds']:.0f} s")
    return rows


def parse_model(text):
    """NAME=MODEL_DIR:FOLD -> (name, Path, fold as int or 'all')."""
    name, _, rest = text.partition('=')
    directory, _, fold = rest.rpartition(':')
    if not name or not directory or fold == '':
        raise ValueError(f'--model {text!r}: NAME=MODEL_DIR:FOLD')
    return name, Path(directory), (fold if fold == 'all' else int(fold))


def images_of(input_dir):
    images = sorted(p for p in Path(input_dir).iterdir() if p.name.endswith(SUFFIX))
    if not images or len(images) != len(list(Path(input_dir).iterdir())):
        raise ValueError(f'{input_dir} must hold <subject>{SUFFIX} files and nothing else')
    return {p.name[:-len(SUFFIX)]: p for p in images}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', action='append', required=True, help='NAME=MODEL_DIR:FOLD (MODEL_DIR is <trainer>__nnUNetPlans__3d_fullres)')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint-name', default=CHECKPOINT, choices=(CHECKPOINT, 'checkpoint_best.pth'))
    args = parser.parse_args(argv)
    try:
        specs = [parse_model(m) for m in args.model]
        images = images_of(args.input_dir)
    except ValueError as exc:
        print(f'STOP: {exc}', flush=True)
        return 8
    if (args.out_dir/RECORD_FILE).exists():
        print(f'STOP: {args.out_dir/RECORD_FILE} exists; never overwritten', flush=True)
        return 8
    for name, directory, fold in specs:
        if not (directory/f'fold_{fold}'/args.checkpoint_name).is_file():
            print(f'STOP: {directory}/fold_{fold}/{args.checkpoint_name}: no such file', flush=True)
            return 8

    from importlib.metadata import version
    import nibabel as nib
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device(args.device)
    if device.type == 'cuda':
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., LIMIT_BYTES/total), 0)          # nx_infer_child.py's cap
        print(f'device cap {LIMIT_BYTES/2**30:g} GiB of {total/2**30:.1f} GiB', flush=True)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    models, record_models, reader = [], {}, None
    for name, directory, fold in specs:
        predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=device.type == 'cuda', device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(str(directory), use_folds=(fold,), checkpoint_name=args.checkpoint_name)
        checkpoint = directory/f'fold_{fold}'/args.checkpoint_name
        blob = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
        axes = predictor.allowed_mirroring_axes
        record_models[name] = dict(model_dir=str(directory), fold=fold, checkpoint=str(checkpoint), checkpoint_name=args.checkpoint_name, checkpoint_sha256=sha256(checkpoint), trainer=predictor.trainer_name, checkpoint_fold=(blob.get('init_args') or {}).get('fold'),
                                   checkpoint_trainer=blob.get('trainer_name'), current_epoch=blob.get('current_epoch'), allowed_mirroring_axes=axes, use_mirroring=predictor.use_mirroring,
                                   tta_views=2**len(axes) if predictor.use_mirroring and axes else 1, tile_step_size=predictor.tile_step_size, use_gaussian=predictor.use_gaussian,
                                   patch_size=list(predictor.configuration_manager.patch_size))
        del blob
        if reader is None:
            reader = predictor.plans_manager.image_reader_writer_class()

        def predict(image, properties, predictor=predictor):
            return predictor.predict_single_npy_array(image, properties, None, None, True)
        models.append((name, predict))

    def read_image(path):
        image, properties = reader.read_images([str(path)])
        return image, properties, nib.load(str(path)).affine

    rows = run(models, images, args.out_dir, read_image=read_image, write_seg=reader.write_seg)
    document = dict(format=FORMAT, created_utc=utc(), command=sys.argv if argv is None else ['infer_r3nn.py', *argv], python=platform.python_version(), nnunetv2=version('nnunetv2'), torch=torch.__version__, cuda=torch.version.cuda,
                    device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None, device_cap_gib=LIMIT_BYTES/2**30 if device.type == 'cuda' else None, compile=os.environ.get('nnUNet_compile'),
                    ext_trainer=os.environ.get('nnUNet_extTrainer'), models=record_models, subjects=list(rows), seconds_per_subject={s: r['seconds'] for s, r in rows.items()},
                    gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None, gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type == 'cuda' else None,
                    test_labels_seen=False, note='只读图像，不读标签；只写各模型概率的均值（float16）与各模型的标签图。')
    tmp = args.out_dir/(RECORD_FILE+'.tmp')
    tmp.write_text(json.dumps(document, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    tmp.replace(args.out_dir/RECORD_FILE)
    print(json.dumps({k: document[k] for k in ('subjects', 'gpu_peak_allocated_gib', 'gpu_peak_reserved_gib')}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
