"""Synthetic stand-ins for the dry run and the tests: NIfTI subjects in the neurite-oasis layout, a
folds file, a development allowlist and a pair.npz + manifest. Subject ids are canonical spellings in
the range OASIS_OAS1_09xx_MR1, which does not exist in OASIS (the real ids end at 0457); nothing under
/mnt/d/data or data/ is touched.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from jd_curl.three_d.nifti import file_sha256

from .data import IMAGE_FILE, LABEL_FILES


def oasis(n):
    return f"OASIS_OAS1_{n:04d}_MR1"


def synthetic_volume(shape_xyz, seed, n_classes=36):
    """(image float32, labels uint8) in native xyz order: an ellipsoidal 'brain' (label 1) holding
    blobs of labels 2..n-1, intensity = per-label mean + noise, background exactly 0."""
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape_xyz], indexing="ij"))
    centre = (np.asarray(shape_xyz)-1)/2
    radius = np.asarray(shape_xyz)*.42
    brain = np.sum(((grid-centre[:, None, None, None])/radius[:, None, None, None])**2, axis=0) <= 1
    labels = np.where(brain, 1, 0).astype(np.uint8)
    base = np.random.default_rng(12345).uniform(.2, 1., n_classes)       # same intensity table for all subjects
    for label in range(2, n_classes):
        c = centre+rng.uniform(-.25, .25, 3)*np.asarray(shape_xyz)
        r = rng.uniform(.06, .14)*min(shape_xyz)
        blob = np.sum((grid-c[:, None, None, None])**2, axis=0) <= r*r
        labels[blob & brain] = label
    image = np.where(labels > 0, base[labels]+rng.normal(0, .03, shape_xyz), 0.).astype(np.float32)
    image[labels > 0] = np.clip(image[labels > 0], .05, None)
    return image, labels


def _save(path, array, dtype):
    affine = np.diag([1., 1., 1., 1.])
    affine[:3, 3] = [-10., -12., -8.]
    img = nib.Nifti1Image(np.asarray(array, dtype=dtype), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))


def write_subjects(root, subjects, shape_xyz=(32, 32, 32), *, tissue4=None):
    """Write aligned_norm / aligned_seg35 (and aligned_seg4 from `tissue4`, a Grouping) per subject."""
    root = Path(root)
    for index, subject in enumerate(subjects):
        image, labels = synthetic_volume(shape_xyz, 1000+int(subject.split("_")[2]))
        directory = root/subject
        directory.mkdir(parents=True, exist_ok=True)
        _save(directory/IMAGE_FILE, image, np.float32)
        _save(directory/LABEL_FILES["seg35"], labels, np.int16)
        if tissue4 is not None:
            _save(directory/LABEL_FILES["seg4"], tissue4.class_to_channel()[labels], np.int16)
    return root


def write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=1)+"\n", encoding="utf-8")
    return Path(path)


def folds_document(fold_a_train, fold_a_calib, fold_b_train, fold_b_calib, holdout=()):
    subjects = {}
    for fold, role, ids in (("A", "train", fold_a_train), ("A", "calib", fold_a_calib), ("B", "train", fold_b_train),
                            ("B", "calib", fold_b_calib)):
        for s in ids:
            subjects[s] = dict(fold=fold, role=role)
    return dict(schema="synthetic.folds", seed=0, subjects=subjects, bline_holdout=list(holdout))


def allowlist_document(sources):
    """sources: list of (name, subjects, never_for_training)."""
    import hashlib
    items = [dict(name=name, dataset="OASIS", path=f"synthetic/{name}.json",
                  sha256=hashlib.sha256(name.encode()).hexdigest(), never_for_training=bool(never), subjects=list(ids))
             for name, ids, never in sources]
    union = []
    for item in items:
        union += [s for s in item["subjects"] if s not in union]
    return dict(schema="bayesreg.dev_subjects.v1", subjects={"OASIS": union}, sources=items)


def write_pair(directory, data_root, fixed_subject, moving_subject):
    """pair.npz (fixed, moving, initial_u) + manifest.json whose sources.<side>.sha256 are the image files.

    Synthetic only (C23.6-1, D104 (a)): the data root must not be a real one and both ids must be synthetic
    (pins.check_synthetic_data_root, pins.check_synthetic_subjects: PinnedInputRefused) before anything is read."""
    from jd_curl.three_d.nifti import load_scalar
    from .pins import check_synthetic_data_root, check_synthetic_subjects
    check_synthetic_data_root(data_root)
    check_synthetic_subjects([fixed_subject, moving_subject], "write_pair")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    volumes = {side: load_scalar(Path(data_root)/s/IMAGE_FILE)
               for side, s in (("fixed", fixed_subject), ("moving", moving_subject))}
    grid = volumes["fixed"].grid
    np.savez_compressed(directory/"pair.npz", fixed=volumes["fixed"].values, moving=volumes["moving"].values,
                        initial_u=np.zeros(grid.shape+(3,)))
    manifest = dict(fixed_subject=fixed_subject, moving_subject=moving_subject, shape_zyx=list(grid.shape),
                    spacing_xyz=list(grid.spacing), origin_xyz=[float(a[0]) for a in grid.axes], unit="mm",
                    pair_sha256=file_sha256(directory/"pair.npz"),
                    sources={side: dict(sha256=v.metadata["sha256"]) for side, v in volumes.items()})
    write_json(directory/"manifest.json", manifest)
    return directory/"pair.npz"


__all__ = ["allowlist_document", "folds_document", "oasis", "synthetic_volume", "write_json", "write_pair",
           "write_subjects"]
