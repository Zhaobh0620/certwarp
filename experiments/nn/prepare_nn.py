"""`prepare` of work package NN (runs/20261105-nn/任务书.md): the training subjects, checked against every list they must not touch; nnUNet_raw/Dataset935_OASIS35 made of symbolic links only (project rules, hard rule 8: no data is copied); dataset.json; the label check; the manifest. sci310, no nnunetv2.

Which subjects: splits/bayesreg_v1/train_subjects.txt (pinned by sha256), 106 people. Refused, before any file of a subject is opened, if the list is not exactly the training subjects of the folds file or if it meets any of: the sealed lists of bayesreg.eval.truth (holdout_B_v1, H1, H3, H4, each pinned there; H2 in code) or any source of configs/bayesreg/dev_subjects.json
other than the training set itself (that allowlist also lists the training set, as the source bayesreg_train_v1, for the label guard; intersecting with the whole file would always be non-empty): stage_L2R_val_v1, stage_T0_tune_v1 (never for training), stage_E11_native_v1, stage_LPBA40_v1_tune. A required source that is missing or of another size is refused too: a check against an empty list says nothing.

The labels are read here for the training subjects only, through the label guard of bayesreg.seg.unet.data (LabelGuard, read_labels; it also refuses a subject off the folds or in another allowlist source). The check: every value in 0..35. dataset.json asks for 0..35 consecutive; if some label never occurs in the 106 it is reported, nnU-Net accepts that.
No conversion is written: if a value outside 0..35 turned up, this stops (the brief: then a conversion is written, and its output goes to the results directory).

The manifest (prepare_manifest.json) records for every file its path, size, mtime, and the sha256 of its first HEAD_BYTES bytes, not of the whole file (the brief allows that; the labels are shorter than that, so theirs is of the whole file). No voxel of an image is read here; the images' headers are compared with the labels'. nnUNetv2_plan_and_preprocess --verify_dataset_integrity then reads every image.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

import nn_common  # noqa: E402
from bayesreg.eval.truth import SEALED_LPBA40, load_sealed_lists  # noqa: E402
from bayesreg.io.guard import CANONICAL_SUBJECT_PATTERNS, load_dev_allowlist  # noqa: E402
from bayesreg.seg.unet.data import IMAGE_FILE, LABEL_FILES, LabelGuard, OasisFiles, load_folds, read_labels  # noqa: E402
from bayesreg.seg.unet.pins import PINNED_COUNTS, PINNED_FOLDS_SHA256, TRAINING_SOURCE  # noqa: E402
from bayesreg.seg.unet.snapshot import code_record, require_clean_code  # noqa: E402

REPO = nn_common.REPO
TRAIN_LIST = REPO/"splits/bayesreg_v1/train_subjects.txt"
TRAIN_LIST_SHA256 = "81aefd1a52c1676e6450232cd76bae88dc977e63b533bfbc55b50f4aeda1280a"     # splits/bayesreg_v1/SHA256SUMS
FOLDS = REPO/"splits/bayesreg_v1/folds.json"
ALLOWLIST = REPO/"configs/bayesreg/dev_subjects.json"
LABEL_TABLE = REPO/"configs/bayesreg/structure_groups_v1.json"
N_SUBJECTS = 106
N_CLASSES = 36
REQUIRED_SOURCES = {"stage_L2R_val_v1": 20, "stage_T0_tune_v1": 24, "stage_E11_native_v1": 10, "stage_LPBA40_v1_tune": 24}
HEAD_BYTES = 1 << 20
FORMAT = "bayesreg.nn.prepare_manifest.v1"
IMAGE_SUFFIX, LABEL_SUFFIX = "_0000.nii.gz", ".nii.gz"
TOP_LEVEL = {"dataset.json", "imagesTr", "labelsTr"}


class SubjectsRefused(Exception):
    """The list of training subjects is not what this work package may touch; nothing of a subject has been opened. Status `subjects_refused`."""
    status = "subjects_refused"


class RawRefused(Exception):
    """nnUNet_raw/Dataset935_OASIS35 is not what this file would build; nothing is overwritten."""


def read_train_list(path=TRAIN_LIST, *, pinned_sha256=TRAIN_LIST_SHA256):
    """The ids of the list; SubjectsRefused if the file is not the pinned one (pinned_sha256=None: tests on a made list)."""
    raw = Path(path).read_bytes()
    if pinned_sha256 is not None and hashlib.sha256(raw).hexdigest() != pinned_sha256:
        raise SubjectsRefused(f"{path}: sha256 {hashlib.sha256(raw).hexdigest()} is not the pinned {pinned_sha256}")
    return [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]


def forbidden_sets(sealed, allowlist, *, training_source=TRAINING_SOURCE, required=REQUIRED_SOURCES):
    """{name: ids} of every list the training set must not touch. The source `training_source` is the training set itself and the only one left out; a required source missing or of another size is refused."""
    sources = {s["name"]: s for s in allowlist.sources}
    problems = [f"the allowlist has no training source {training_source!r}"] if training_source not in sources else []
    for name, count in required.items():
        if name not in sources:
            problems.append(f"the allowlist has no source {name!r}")
        elif len(sources[name]["subjects"]) != count:
            problems.append(f"the allowlist source {name!r} has {len(sources[name]['subjects'])} subjects, expected {count}")
    if problems:
        raise SubjectsRefused("; ".join(problems))
    out = {name: frozenset(ids) for name, ids in sealed.subjects.items()}
    out["H2"] = frozenset(SEALED_LPBA40)
    out["never_for_training"] = frozenset(allowlist.never_for_training)
    out.update({f"allowlist:{name}": frozenset(source["subjects"]) for name, source in sources.items() if name != training_source})
    return out


def check_training_subjects(subjects, forbidden, *, count=N_SUBJECTS, folds_subjects=None, allowlist_source=None):
    """The ids, sorted; SubjectsRefused on a wrong count, a repeated or non-canonical id, a difference from the folds or the allowlist's training source, or any overlap with a forbidden list."""
    subjects = list(subjects)
    problems = []
    if count is not None and len(subjects) != count:
        problems.append(f"{len(subjects)} subjects, expected {count}")
    if len(set(subjects)) != len(subjects):
        problems.append("an id is listed twice")
    wrong = [s for s in subjects if not (isinstance(s, str) and CANONICAL_SUBJECT_PATTERNS["OASIS"].fullmatch(s))]
    if wrong:
        problems.append(f"not canonical OASIS ids: {wrong[:5]}")
    for name, reference in (("folds", folds_subjects), ("allowlist training source", allowlist_source)):
        if reference is not None and set(reference) != set(subjects):
            problems.append(f"differs from the {name}: only there {sorted(set(reference)-set(subjects))[:5]}, only here {sorted(set(subjects)-set(reference))[:5]}")
    for name, ids in forbidden.items():
        overlap = sorted(set(subjects) & set(ids))
        if overlap:
            problems.append(f"in {name}: {overlap[:5]}{' ...' if len(overlap) > 5 else ''} ({len(overlap)})")
    if problems:
        raise SubjectsRefused("the training list is refused: "+"; ".join(problems))
    return sorted(subjects)


def admitted_training_subjects(*, sealed_repo=None, code=None, train_list=TRAIN_LIST, folds_path=FOLDS, allowlist_path=ALLOWLIST):
    """The real training list after every check, with what is needed to read its labels. Reads ids only."""
    code = code_record() if code is None else code
    sealed = load_sealed_lists(Path(sealed_repo) if sealed_repo is not None else Path(code.get("source_repo", code.get("root", REPO))))
    folds = load_folds(folds_path, expect_counts=PINNED_COUNTS)
    if folds.sha256 != PINNED_FOLDS_SHA256:
        raise SubjectsRefused(f"{folds_path}: sha256 {folds.sha256} is not the pinned {PINNED_FOLDS_SHA256}")
    allowlist = load_dev_allowlist(allowlist_path)
    forbidden = forbidden_sets(sealed, allowlist)
    source = {s["name"]: s for s in allowlist.sources}[TRAINING_SOURCE]
    subjects = check_training_subjects(read_train_list(train_list), forbidden, folds_subjects=folds.training_subjects, allowlist_source=source["subjects"])
    return dict(subjects=subjects, folds=folds, allowlist=allowlist, sealed=sealed, overlaps={name: len(set(subjects) & set(ids)) for name, ids in forbidden.items()}, list_sizes={name: len(ids) for name, ids in forbidden.items()})


def label_names(path=LABEL_TABLE):
    """{name: label} for dataset.json: the names of the seg35 label table in the repository; label 0 is called background, as nnU-Net requires."""
    names = json.loads(Path(path).read_text(encoding="utf-8"))["label_table"]["names"]
    table = {int(k): v for k, v in names.items()}
    if sorted(table) != list(range(N_CLASSES)) or len(set(table.values())) != N_CLASSES:
        raise ValueError(f"{path}: the table must name the labels 0..{N_CLASSES-1}, each once")
    return dict(background=0, **{table[k]: k for k in range(1, N_CLASSES)})


def dataset_json(n_training, names):
    return dict(name="OASIS35", description="neurite-oasis aligned_norm and aligned_seg35; B-line training subjects only (splits/bayesreg_v1/train_subjects.txt). The files are symbolic links; the data is not redistributed.", reference="runs/20261105-nn/plan.md", channel_names={"0": "T1"}, labels=dict(names), numTraining=int(n_training), file_ending=".nii.gz", overwrite_image_reader_writer="SimpleITKIO")


def expected_links(subjects, data_root):
    """{relative path in the dataset directory: target} of every link."""
    links = {}
    for subject in subjects:
        links[f"imagesTr/{subject}{IMAGE_SUFFIX}"] = Path(data_root)/subject/IMAGE_FILE
        links[f"labelsTr/{subject}{LABEL_SUFFIX}"] = Path(data_root)/subject/LABEL_FILES["seg35"]
    return links


def verify_raw(dataset_dir, subjects, data_root, names=None):
    """dataset_dir holds exactly dataset.json and the links to data_root. RawRefused otherwise: a regular file, a link elsewhere, a case too many or too few, another dataset.json."""
    dataset_dir = Path(dataset_dir)
    expected = expected_links(subjects, data_root)
    problems = []
    if not dataset_dir.is_dir():
        raise RawRefused(f"{dataset_dir}: no such directory")
    top = {p.name for p in dataset_dir.iterdir()}
    if top != TOP_LEVEL:
        problems.append(f"top level {sorted(top)}, expected {sorted(TOP_LEVEL)}")
    found = {p.relative_to(dataset_dir).as_posix(): p for folder in ("imagesTr", "labelsTr") for p in (dataset_dir/folder).iterdir()} if not problems else {}
    for relative, path in found.items():
        if not path.is_symlink():
            problems.append(f"{relative} is not a symbolic link")
        elif relative in expected and Path(os.readlink(path)) != expected[relative]:
            problems.append(f"{relative} points at {os.readlink(path)}, expected {expected[relative]}")
    if not problems and set(found) != set(expected):
        problems.append(f"cases: only present {sorted(set(found)-set(expected))[:5]}, missing {sorted(set(expected)-set(found))[:5]}")
    document = json.loads((dataset_dir/"dataset.json").read_text(encoding="utf-8")) if (dataset_dir/"dataset.json").is_file() else None
    if document != dataset_json(len(subjects), label_names() if names is None else names):
        problems.append("dataset.json differs from what this file writes")
    if problems:
        raise RawRefused(f"{dataset_dir}: "+"; ".join(problems))
    return dict(n_links=len(found), dataset_json_sha256=hashlib.sha256((dataset_dir/"dataset.json").read_bytes()).hexdigest())


def build_raw(raw_root, subjects, data_root, names=None):
    """The dataset directory: built whole and renamed, or verified if it is there. Every target must exist; nothing is copied."""
    dataset_dir = Path(raw_root)/nn_common.DATASET
    names = label_names() if names is None else names
    if dataset_dir.exists():
        return dict(verify_raw(dataset_dir, subjects, data_root, names), built=False)
    links = expected_links(subjects, data_root)
    missing = [str(target) for target in links.values() if not target.is_file()]
    if missing:
        raise RawRefused(f"{len(missing)} files are missing: {missing[:5]}")
    partial = Path(raw_root)/f".partial_{nn_common.DATASET}"           # not Dataset935*: nnU-Net finds its datasets by that prefix
    if partial.exists():
        raise RawRefused(f"{partial} exists (an interrupted prepare); remove it by hand")
    for folder in ("imagesTr", "labelsTr"):
        (partial/folder).mkdir(parents=True)
    for relative, target in links.items():
        os.symlink(target, partial/relative)
    (partial/"dataset.json").write_text(json.dumps(dataset_json(len(subjects), names), indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    partial.rename(dataset_dir)
    return dict(verify_raw(dataset_dir, subjects, data_root, names), built=True)


def gt_directory(preprocessed_root):
    return Path(preprocessed_root)/nn_common.DATASET/"gt_segmentations"


def expected_gt_links(subjects, data_root):
    return {f"{subject}{LABEL_SUFFIX}": Path(data_root)/subject/LABEL_FILES["seg35"] for subject in subjects}


def verify_gt_links(directory, subjects, data_root):
    """gt_segmentations holds exactly one symbolic link per training subject, to the label file; RawRefused otherwise (a regular file is a copy of a label file, hard rule 8)."""
    directory = Path(directory)
    expected = expected_gt_links(subjects, data_root)
    if not directory.is_dir():
        raise RawRefused(f"{directory}: no such directory")
    found = {p.name: p for p in directory.iterdir()}
    problems = [] if set(found) == set(expected) else [f"only present {sorted(set(found)-set(expected))[:5]}, missing {sorted(set(expected)-set(found))[:5]}"]
    copies = sorted(name for name, path in found.items() if not path.is_symlink())
    if copies:
        problems.append(f"{len(copies)} are not symbolic links: {copies[:3]}")
    wrong = sorted(name for name, path in found.items() if path.is_symlink() and name in expected and Path(os.readlink(path)) != expected[name])
    if wrong:
        problems.append(f"{len(wrong)} point elsewhere: {wrong[:3]}")
    if problems:
        raise RawRefused(f"{directory}: "+"; ".join(problems))
    return dict(n_links=len(found))


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_gt_segmentations(directory, subjects, data_root):
    """nnU-Net copies labelsTr into gt_segmentations (plan_and_preprocess_api.py, copy_file; only --final-validation reads them). Hard rule 8: each copy is compared with its source, whole file, and only when all of them are equal replaced by a symbolic link to the source. Links already there are accepted; run twice, the second run changes nothing. Nothing else is replaced: a differing copy refuses all of it."""
    directory = Path(directory)
    expected = expected_gt_links(subjects, data_root)
    found = {p.name: p for p in directory.iterdir()} if directory.is_dir() else {}
    if set(found) != set(expected):
        raise RawRefused(f"{directory}: only present {sorted(set(found)-set(expected))[:5]}, missing {sorted(set(expected)-set(found))[:5]}")
    copies = [name for name, path in sorted(found.items()) if not path.is_symlink()]
    differing = [name for name in copies if sha256_of(found[name]) != sha256_of(expected[name])]
    if differing:
        raise RawRefused(f"{directory}: {differing[:5]} differ from their source; nothing was replaced")
    freed = sum(found[name].stat().st_size for name in copies)
    for name in copies:
        temporary = directory/f".{name}.link"
        os.symlink(expected[name], temporary)
        os.replace(temporary, found[name])
    return dict(verify_gt_links(directory, subjects, data_root), replaced=len(copies), bytes_freed=freed, compared="sha256 of the whole file, against its source, before any was replaced")


def file_entry(path, *, head_bytes=HEAD_BYTES):
    """Path, size, mtime and the sha256 of the first head_bytes bytes (not of the whole file, unless it is shorter)."""
    path = Path(path)
    stat = path.stat()
    with path.open("rb") as stream:
        head = stream.read(head_bytes)
    return dict(path=str(path), bytes=stat.st_size, mtime=stat.st_mtime, head_sha256=hashlib.sha256(head).hexdigest(), head_bytes=len(head), whole_file=len(head) == stat.st_size)


def inspect_subject(subject, files, guard):
    """The entry of one subject: the two files, the headers compared, the label values through the guard. ValueError if the geometry differs."""
    image_path, label_path = files.image_path(subject), files.label_path(subject)
    image, label = nib.load(image_path), nib.load(label_path)       # headers only; the image's voxels are not read here
    if image.shape != label.shape or not np.allclose(image.affine, label.affine, rtol=0, atol=1e-5):
        raise ValueError(f"{subject}: image {image.shape} and label {label.shape} differ in shape or affine")
    values, _ = read_labels(files, subject, guard)
    present = np.unique(values)
    return dict(subject=subject, image=file_entry(image_path), label=file_entry(label_path), shape=list(image.shape), affine=image.affine.tolist(), image_dtype=str(image.header.get_data_dtype()), label_dtype=str(label.header.get_data_dtype()), spacing=[float(z) for z in image.header.get_zooms()[:3]], labels_present=[int(v) for v in present])


def label_summary(entries, n_classes=N_CLASSES):
    """How many subjects have each label; ok: every value in 0..n_classes-1. absent: labels of the table no subject has."""
    counts = {label: 0 for label in range(n_classes)}
    outside = []
    for entry in entries:
        for label in entry["labels_present"]:
            if label in counts:
                counts[label] += 1
            else:
                outside.append((entry["subject"], label))
    return dict(ok=not outside, outside=outside[:20], subjects_with_label={str(k): v for k, v in counts.items()}, absent_labels=[k for k, v in counts.items() if v == 0], labels_not_in_every_subject={str(k): v for k, v in counts.items() if 0 < v < len(entries)})


def prepare(out=nn_common.OUT, *, data_root=nn_common.DATA_ROOT, log=print):
    """The real step: from a clean tree. Refuses first, then builds, then reads the training labels once. The manifest is written once; a second run verifies the links and stops there."""
    out = Path(out)
    code = require_clean_code()
    admitted = admitted_training_subjects(code=code)
    subjects = admitted["subjects"]
    log(f"{len(subjects)} training subjects; no overlap with: {', '.join(f'{k} ({v})' for k, v in admitted['list_sizes'].items())}")
    manifest_path = out/"prepare_manifest.json"
    nn_common.raw_root(out).mkdir(parents=True, exist_ok=True)
    raw = build_raw(nn_common.raw_root(out), subjects, data_root)
    log(f"raw: {json.dumps(raw)}")
    if manifest_path.exists():
        log(f"{manifest_path} exists; the links were verified, nothing is read again")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    guard = LabelGuard(admitted["folds"], admitted["allowlist"], training_source=TRAINING_SOURCE)
    files = OasisFiles(Path(data_root))
    entries = []
    for index, subject in enumerate(subjects):
        entries.append(inspect_subject(subject, files, guard))
        if (index+1) % 20 == 0:
            log(f"{index+1}/{len(subjects)} subjects read")
    summary = label_summary(entries)
    if not summary["ok"]:
        raise ValueError(f"labels outside 0..{N_CLASSES-1}: {summary['outside']}; a conversion has to be written (the brief)")
    document = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), code=code, dataset=nn_common.DATASET, data_root=str(data_root), train_list=str(TRAIN_LIST), train_list_sha256=TRAIN_LIST_SHA256, folds_sha256=admitted["folds"].sha256, allowlist_sha256=admitted["allowlist"].sha256, sealed_lists=admitted["sealed"].record, checked_against=admitted["list_sizes"], overlaps=admitted["overlaps"],
                    n_subjects=len(subjects), label_contact_subjects=subjects, raw=raw, labels=summary, shapes=sorted({tuple(e["shape"]) for e in entries}), spacings=sorted({tuple(e["spacing"]) for e in entries}), hash_note=f"sha256 of the first {HEAD_BYTES} bytes of each file, not of the whole file (whole_file says when the file is shorter)", subjects=entries)
    manifest_path.write_text(json.dumps(document, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    log(f"{manifest_path}: labels ok, absent {summary['absent_labels']}, shapes {document['shapes']}, spacings {document['spacings']}")
    return document


def plan_summary(plans_path, fingerprint_path=None):
    """What nnU-Net planned for 3d_fullres, as a small dict for the record."""
    plans = json.loads(Path(plans_path).read_text(encoding="utf-8"))
    config = plans["configurations"][nn_common.CONFIGURATION]
    architecture = config["architecture"]
    out = dict(plans_name=plans.get("plans_name"), transpose_forward=plans.get("transpose_forward"), image_reader_writer=plans.get("image_reader_writer"), configurations=sorted(plans["configurations"]), patch_size=config["patch_size"], batch_size=config["batch_size"], spacing=config["spacing"], median_image_size_in_voxels=config["median_image_size_in_voxels"],
               normalization_schemes=config["normalization_schemes"], use_mask_for_norm=config["use_mask_for_norm"], network_class=architecture["network_class_name"], n_stages=architecture["arch_kwargs"].get("n_stages"), features_per_stage=architecture["arch_kwargs"].get("features_per_stage"), kernel_sizes=architecture["arch_kwargs"].get("kernel_sizes"), strides=architecture["arch_kwargs"].get("strides"),
               n_conv_per_stage=architecture["arch_kwargs"].get("n_conv_per_stage"), n_conv_per_stage_decoder=architecture["arch_kwargs"].get("n_conv_per_stage_decoder"), batch_dice=config.get("batch_dice"))
    if fingerprint_path is not None and Path(fingerprint_path).is_file():
        fingerprint = json.loads(Path(fingerprint_path).read_text(encoding="utf-8"))
        shapes = np.array(fingerprint["shapes_after_crop"])
        out.update(shapes_after_crop_median=[float(v) for v in np.median(shapes, axis=0)], shapes_after_crop_min=[int(v) for v in shapes.min(axis=0)], shapes_after_crop_max=[int(v) for v in shapes.max(axis=0)], median_relative_size_after_cropping=fingerprint.get("median_relative_size_after_cropping"))
    return out
