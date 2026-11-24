"""NX's steps 1 and 2 (runs/20261105-fx/任务书.md, plan.md): the probabilities of nnU-Net into the format and the orientation of /home/rmcfz/seq_vp_tag_results/seg/unet35_v1/pred, and the registration of nn935_v1.  Only images are read, never a truth label; seg/nn_dev_read.lock is not touched; nothing here imports torch or nnunetv2.

The layout of nnU-Net's file <subject>.npz, key probabilities (read in nnunetv2 2.8.1, export_prediction.py; a test asks SimpleITK of the environment nnunet): (classes, *the array of SimpleITK), that is voxel (c, k, j, i) is voxel (i, j, k) of the NIfTI file on disk.  So: reverse the three axes to the file's order, then what eval_nn.orient_like_load_scalar does (RAS storage, axes zyx).
float16: rounding is monotonic, so it can only make two channels equal; the reader takes the smallest label on a tie (fast_channels.raw_labels), which may not be nnU-Net's.  Such a voxel gets one ULP on the right side, chosen so that the channel sum does not move away from 1, and the count is on record.  The reader's own function reads the file back before anything counts.
The gate (the lead, 11-05, before any inference, changing the brief's "voxel by voxel"): the argmax of the probs36.npz finally stored (after the conversion and float16, as the reader's own function reads it) against the labels of NN's first inference (pred_dev), subject by subject: voxels that differ / all voxels <= 1e-4 passes; above it GateFailed, and nothing of that subject stays as an output.  It guards against the wrong orientation, subject or model, all far above 1e-4; the argmax on logits (pred_dev) and on the softmax (this run), and two inferences on the GPU, may differ by a few voxels.  Every subject's count is on record.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import time

import numpy as np

import fast_channels as fast
import mc0_core as core

N_CLASSES = 36
PROBS_FILE, GROUPS_FILE, PROVENANCE_FILE = 'probs36.npz', 'structure_groups.npz', 'provenance.json'
IMAGE_FILE = 'aligned_norm.nii.gz'
GEOMETRY_TOL = 1e-3                             # eval_nn's
GATE_TOLERANCE = 1e-4                           # the lead, 11-05: voxels that differ / all voxels, per subject; 688 voxels on the real grid (192 x 224 x 160)


class GateFailed(Exception):
    """A label map of this inference is not the one NN read."""


class RegistrationRefused(Exception):
    """The label contact of the model touches a list it must not touch, or the file differs from what is registered."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


# ---- orientation -----------------------------------------------------------------------------------------------------------------------------
def orient_native(values, affine):
    """A native array (the NIfTI file's order) into jd_curl.three_d.nifti.load_scalar's: RAS storage, axes zyx.  experiments/nn/eval_nn.orient_like_load_scalar, copied (that module imports the UNV2 evaluation); a test compares."""
    import nibabel as nib
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(affine), nib.orientations.axcodes2ornt(('R', 'A', 'S')))
    return nib.orientations.apply_orientation(values, transform).transpose(2, 1, 0)


def orient_probabilities(probabilities, affine):
    """(classes, k, j, i) of SimpleITK's order into (classes, nz, ny, nx) of load_scalar, channel by channel, the dtype kept."""
    if probabilities.ndim != 4 or probabilities.shape[0] != N_CLASSES:
        raise GateFailed(f'probabilities of shape {probabilities.shape}: {N_CLASSES} classes on axis 0 expected')
    first = orient_native(probabilities[0].transpose(2, 1, 0), affine)
    out = np.empty((N_CLASSES,)+first.shape, dtype=probabilities.dtype)
    out[0] = first
    for c in range(1, N_CLASSES):
        out[c] = orient_native(probabilities[c].transpose(2, 1, 0), affine)
    return out


# ---- float16 ---------------------------------------------------------------------------------------------------------------------------------
def float16_keeping_argmax(p, labels):
    """(float16 array, number of voxels touched): p.astype(float16), and where a tie would give the reader another label than `labels` (the argmax of p), one ULP: down on the channels tied with it when the raw sum is above 1, else up on the label's channel."""
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


def save_npz(path, array):
    """bayesreg.seg.unet.infer._save_npz, whose module imports torch: compressed, written then moved."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, p=array)
    tmp = Path(str(path)+'.tmp')
    tmp.write_bytes(buffer.getvalue())
    tmp.replace(path)


# ---- one subject -----------------------------------------------------------------------------------------------------------------------------
def read_labels_file(path, image_path):
    """A label file of nnU-Net in load_scalar's order, after its shape and affine against the image's (a header read of the image: the voxels of the image are not needed here).
    C23.6-2, D104 (a): the function takes no subject, so every canonical OASIS id in the two paths (and an IBSR name) goes through bayesreg.io.guard.refuse_sealed before either file is opened."""
    from bayesreg.io.guard import refuse_sealed
    for text in (str(path), str(image_path)):
        for subject in re.findall(r'OASIS_OAS1_[0-9]{4}_MR1', text) + (['IBSR'] if 'IBSR' in text.upper() else []):
            refuse_sealed(subject, 'nx_seg.read_labels_file')
    import nibabel as nib
    image, labels = nib.load(image_path), nib.load(path)
    if labels.shape != image.shape or not np.allclose(labels.affine, image.affine, rtol=0, atol=GEOMETRY_TOL):
        raise GateFailed(f'{path}: geometry differs from {image_path}')
    values = np.asarray(labels.dataobj)
    if not np.issubdtype(values.dtype, np.integer) or values.min() < 0 or values.max() >= N_CLASSES:
        raise GateFailed(f'{path}: {values.dtype}, values {values.min()}..{values.max()}: not labels 0..{N_CLASSES-1}')
    return orient_native(values.astype(np.uint8), image.affine), image.affine


def gate_passes(n_differing, n_voxels):
    """At the line passes."""
    return n_differing/n_voxels <= GATE_TOLERANCE


def check_gate(subject, what, n_differing, n_voxels):
    if not gate_passes(n_differing, n_voxels):
        raise GateFailed(f'{subject}: {what} differs from pred_dev on {n_differing} voxels of {n_voxels}, {n_differing/n_voxels:.2e} > {GATE_TOLERANCE:g}')


def convert_problems(record, subjects):
    """Why convert_record.json is not 20 subjects through the gate, as a list."""
    if record.get('gate') != 'passed' or sorted(record.get('subjects') or []) != sorted(subjects):
        return ['convert_record.json: not the 20 development subjects through the gate']
    counts, voxels = record.get('n_differing_by_subject') or {}, record.get('n_voxels_by_subject') or {}
    if record.get('tolerance') != GATE_TOLERANCE or sorted(counts) != sorted(subjects) or sorted(voxels) != sorted(subjects):
        return [f"convert_record.json: not the gate of this plan: tolerance {record.get('tolerance')}, {len(counts)} counts"]
    over = [s for s in subjects if not gate_passes(counts[s], voxels[s])]
    return [f'convert_record.json: over the tolerance: {over}'] if over else []


def provenance_of(subject, *, grouping, grouping_check, input_sha256, output_sha256, grid, shape, groups_shape, model_id, segmenter_id, fold, checkpoint_sha256, trainer, seconds, code_sha, registry_rel):
    """What StructureLayer and validate_segmentation_provenance ask of a prediction, with the same names as bayesreg.seg.unet.infer.segment_subject.  The policy is all_folds_mean: one model, every model the registry has (structure_layer.SINGLE_MODEL_SEGMENTERS).  Written before the registration, which it must then agree with: the sha256 is train_record's, the queue compares."""
    return dict(layer='structure', segmenter_id=segmenter_id, model_ids=[model_id], model_sha256={model_id: [checkpoint_sha256]}, folds={model_id: fold}, trained_with_labels=True, test_labels_seen=False, subject=subject, policy='all_folds_mean', members=[dict(model_id=model_id, fold='all', seed=None, member_sha256=checkpoint_sha256, trainer=trainer, checkpoint='checkpoint_final.pth')], **grouping.record(), grouping_check=grouping_check,
                input_sha256={IMAGE_FILE: input_sha256}, output_sha256=output_sha256, outputs=dict(probs36=dict(file=PROBS_FILE, key='p', dtype='float16', shape=list(shape), classes='seg35 label c at channel c (0 = background)'), groups=dict(file=GROUPS_FILE, key='p', dtype='float16', shape=list(groups_shape), channels='0 = background, k = class_names[k-1]'), layout='(channels, nz, ny, nx), jd_curl.three_d.nifti.load_scalar storage (RAS, zyx)', read_back='fast_channels.raw_labels and bayesreg.seg.channels.read_back_probabilities'),
                grid=grid, label_contact_check=dict(subject_in_any_used_model=False, image_level=True, note='pair-level check runs in validate_segmentation_provenance'), registry_path=registry_rel, registry_sha256=None, registry_note='written before the registration (step 2); the queue compares model_sha256 with the committed registry', version=dict(segmenter=segmenter_id, nnunet='see infer_record.json'), command='experiments/fl/run_fx.py nx convert', environment=dict(code=dict(kind='git_worktree', commit=code_sha)), runtime_seconds=seconds)


def convert_subject(subject, *, raw_dir, pred_dev, image_path, out_root, grouping, grouping_check, input_sha256, grid, model_id, segmenter_id, fold, checkpoint_sha256, trainer, code_sha, registry_rel, log=lambda text: None):
    """One subject: the gate, then probs36.npz, structure_groups.npz, read back by the reader, then provenance.json last.  Returns the subject's record.  Never overwritten; GateFailed leaves no output of the subject."""
    from bayesreg.seg.channels import read_back_probabilities
    from bayesreg.seg.unet.groups import group_probabilities
    started = time.perf_counter()
    out = Path(out_root)/subject
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} exists and is not empty; predictions are never overwritten')
    labels, affine = read_labels_file(Path(pred_dev)/f'{subject}.nii.gz', image_path)
    again, _ = read_labels_file(Path(raw_dir)/f'{subject}.nii.gz', image_path)
    n_label_files = int((labels != again).sum())    # a record; the gate is on what is stored
    with np.load(Path(raw_dir)/f'{subject}.npz', allow_pickle=False) as npz:
        probabilities = npz['probabilities']
    p = orient_probabilities(probabilities, affine)
    del probabilities
    if p.shape[1:] != labels.shape:
        raise GateFailed(f'{subject}: probabilities {p.shape} on labels {labels.shape}')
    own = p.argmax(axis=0)                          # this run's, what float16 must keep
    n_voxels = int(labels.size)
    n_argmax = int((own != labels).sum())
    check_gate(subject, 'the argmax of the raw probabilities', n_argmax, n_voxels)     # before the writing: the stored argmax is this one
    p16, n_ties = float16_keeping_argmax(p, own)
    groups = group_probabilities(p, grouping, dtype=np.float32)
    del p
    out.mkdir(parents=True, exist_ok=True)
    try:
        save_npz(out/PROBS_FILE, p16)
        save_npz(out/GROUPS_FILE, groups.astype(np.float16))
        shape, groups_shape = p16.shape, groups.shape
        del p16, groups
        array, _ = read_back_probabilities(out/GROUPS_FILE, shape=groups_shape)
        read = fast.raw_labels(out/PROBS_FILE, shape, array, (grouping.background_labels,)+tuple(grouping.group_labels))
        if read is None:
            raise GateFailed(f'{subject}: the reader refuses its fast path on {PROBS_FILE}')
        back, _, record, deviation = read
        tol = core.make_inputs().GROUP_TOL
        if not deviation <= tol:
            raise GateFailed(f'{subject}: probs36 summed by group differs from structure_groups.npz by {deviation:.3e} > {tol:g}')
        n_back = int((back != labels).sum())
        check_gate(subject, f'the argmax of the stored {PROBS_FILE}', n_back, n_voxels)
        n_not_own = int((back != own).sum())
        if n_not_own:
            raise GateFailed(f'{subject}: the reader reads {n_not_own} voxels other than the argmax of the probabilities: the conversion is wrong')
    except GateFailed:
        for name in (PROBS_FILE, GROUPS_FILE):
            (out/name).unlink(missing_ok=True)
        out.rmdir()
        raise
    seconds = time.perf_counter()-started
    output = {PROBS_FILE: file_sha256(out/PROBS_FILE), GROUPS_FILE: file_sha256(out/GROUPS_FILE)}
    provenance = provenance_of(subject, grouping=grouping, grouping_check=grouping_check, input_sha256=input_sha256, output_sha256=output, grid=grid, shape=shape, groups_shape=groups_shape, model_id=model_id, segmenter_id=segmenter_id, fold=fold, checkpoint_sha256=checkpoint_sha256, trainer=trainer, seconds=seconds, code_sha=code_sha, registry_rel=registry_rel)
    (out/PROVENANCE_FILE).write_text(json.dumps(provenance, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    log(f'{subject}: gate passed, {n_back} of {n_voxels} voxels differ from pred_dev, {n_ties} voxels with one ULP, raw sum deviation {record["raw_max_abs_sum_deviation"]:.2e}, group deviation {deviation:.2e}, {seconds:.0f} s')
    return dict(subject=subject, gate='passed', tolerance=GATE_TOLERANCE, n_voxels=n_voxels, n_differing=n_back, fraction_differing=n_back/n_voxels, n_differing_raw_argmax=n_argmax, n_differing_label_file=n_label_files, n_float16_ties_fixed=n_ties, raw_max_abs_sum_deviation=record['raw_max_abs_sum_deviation'], group_max_abs_deviation=deviation, output_sha256=output, seconds=seconds)


# ---- step 2 -----------------------------------------------------------------------------------------------------------------------------------
def contact_problems(contact, *, allowlist, sealed, training_subjects, dev_sources=('stage_L2R_val_v1', 'stage_E11_native_v1', 'stage_LPBA40_v1_tune')):
    """Why this label contact cannot be registered, as a list: it must be exactly the training set, and touch no development subject, no T0 subject (never_for_training) and no sealed list."""
    contact = list(contact)
    problems = []
    if len(set(contact)) != len(contact):
        problems.append('the label contact holds a subject twice')
    if set(contact) != set(training_subjects):
        problems.append(f'the label contact is not the training set: missing {sorted(set(training_subjects)-set(contact))[:5]}, extra {sorted(set(contact)-set(training_subjects))[:5]}')
    sources = {s['name']: s for s in allowlist.sources}
    for name in dev_sources:
        if name not in sources:
            problems.append(f'the allowlist has no source {name}')
            continue
        overlap = sorted(set(contact) & set(sources[name]['subjects']))
        if overlap:
            problems.append(f'development source {name}: {overlap[:5]}')
    t0 = sorted(set(contact) & set(allowlist.never_for_training))
    if t0:
        problems.append(f'never_for_training (T0): {t0[:5]}')
    outside = sorted(s for s in contact if not allowlist.allows(s))
    if outside:
        problems.append(f'off the allowlist: {outside[:5]}')
    for subject in contact:
        which = sealed.which(subject)
        if which is not None:
            problems.append(f'{subject} is on the sealed list {which}')
    return problems


def registry_document(*, segmenter_id, model_id, fold, checkpoint_sha256, contact):
    from bayesreg.io.pair import REGISTRY_SCHEMA, subject_list_sha256
    contact = sorted(contact)
    return dict(schema=REGISTRY_SCHEMA, comment='structure-layer model of NX: one nnU-Net fold-all model, trained on the whole B-line training set; append-only; written by experiments/fl/nx_seg.py (runs/20261105-fx/plan.md)', models={model_id: dict(segmenter_id=segmenter_id, fold=fold, member_model_sha256=[checkpoint_sha256], label_contact_subjects=contact, label_contact_subjects_sha256=subject_list_sha256(contact))})


def register(path, document):
    """Written once, validated by the schema's own reader before it counts: a file that is there must be identical (entries are append-only).  Returns the SegmenterRegistry."""
    from bayesreg.io.pair import load_segmenter_registry
    path = Path(path)
    text = json.dumps(document, indent=1, ensure_ascii=False)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise RegistrationRefused(f'{path} is there with other content; entries are append-only')
        return load_segmenter_registry(path)
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(text, encoding='utf-8')
    try:
        load_segmenter_registry(tmp)
    except Exception:
        tmp.unlink()
        raise
    tmp.replace(path)
    return load_segmenter_registry(path)
