"""Folds, the label guard, image/label reading, normalization and augmentation (04_分割器.md §4, §7).

Label guard (project rules, hard rules 6 and 7; contract C03.1-2). The only way the training and
calibration code reaches a seg35 file is `read_labels(files, subject, guard)`, and the guard admits a
subject only if
  - it is a canonical OASIS id that the folds file places in the B-line training set (fold A or B,
    role train or calib) -- and, when a fold is named, in that fold;
and the guard itself refuses to be built (the whole split is refused) unless the allowlist source named
`training_source` (real runs: "bayesreg_train_v1", pins.py) exists, is not never_for_training and lists
exactly the training subjects of the folds file, and refuses it if any training subject is
  - off the development allowlist, or marked never_for_training there (stage_T0_tune_v1), or
  - listed by any other allowlist source (stage_L2R_val_v1, stage_E11_native_v1, ...), or
  - in the Learn2Reg OASIS validation range OASIS_OAS1_0438_MR1 .. OASIS_OAS1_0457_MR1 (built-in net;
    the 20 subject directories in that range are exactly the 20 stage_L2R_val_v1 subjects), or
  - in any list of the folds file whose key names a hold-out / test / sealed set, or in `denied`.
Everything else (development subjects, T0, H1-H4, the B-line hold-out set, LPBA40, IBSR18) is
refused with LabelAccessRefused before the label file is opened. Images (aligned_norm) carry no labels;
inference reads them only for allowlisted subjects (infer.py).

Folds file (splits/bayesreg_v1/folds.json, written by the draw). Accepted layouts, exactly one of:
  (a) {"subjects": {"<id>": {"fold": "A"|"B", "role": "train"|"calib", ...}, ...}, ...}
  (b) {"<any key>": [{"subject"|"id"|"subject_id": "<id>", "fold": ..., "role": ...}, ...], ...}
  (c) {"folds": {"A": {"train": [...], "calib": [...]}, "B": {...}}, ...}
optionally nested one level under "training". Fold may also be 0/1; role "calibration" = "calib".

Full-training-set folds of unet35_v2_full (splits/bayesreg_v1/folds_full_v2.json, 04_分割器.md §14): derived from
folds.json by `derive_full_folds`; the calib of fold X is the calib of fold X in folds.json, the train of fold X is every
other training subject, so a subject is in both folds. `load_full_folds` reads them into a FullFoldSplit; `load_folds`
refuses that file ("listed twice"). The guard's rules are the same; a fold no longer excludes a subject of the other.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from jd_curl.validation import InputValidationError
from jd_curl.three_d.nifti import load_scalar

from ...io.guard import CANONICAL_SUBJECT_PATTERNS, DevAllowlist, strict_json_object
from ...io.pair import subject_list_sha256

FOLD_NAMES = ("A", "B")
FOLD_INDEX = {"A": 0, "B": 1}
ROLES = ("train", "calib")
_ROLE_ALIASES = {"train": "train", "training": "train", "calib": "calib", "calibration": "calib"}
_ID_KEYS = ("subject", "id", "subject_id")
_SEALED_KEY = re.compile(r"holdout|hold[_-]out|sealed|(?:^|[^a-z])test(?:$|[^a-z])", re.IGNORECASE)
L2R_VAL_RANGE = (438, 457)          # OASIS_OAS1_0438_MR1 .. OASIS_OAS1_0457_MR1 (stage_L2R_val_v1)
IMAGE_FILE = "aligned_norm.nii.gz"
LABEL_FILES = {"seg35": "aligned_seg35.nii.gz", "seg4": "aligned_seg4.nii.gz"}
LABEL_RANGE = {"seg35": 36, "seg4": 5}
DEFAULT_DATA_ROOT = Path("/mnt/d/data")


class LabelAccessRefused(Exception):
    """The labels of this subject may not be read by the segmenter code; status `label_access_refused`."""
    status = "label_access_refused"


def _is_oasis(subject):
    return isinstance(subject, str) and CANONICAL_SUBJECT_PATTERNS["OASIS"].fullmatch(subject) is not None


def _oasis_number(subject):
    return int(subject.split("_")[2])


# ---- folds ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FoldSplit:
    path: str
    sha256: str
    layout: str
    assignment: Mapping                 # subject -> (fold, role)
    sealed_lists: tuple = field(default=())   # (key path, n ids) of hold-out-like lists checked disjoint

    @property
    def training_subjects(self):
        return frozenset(self.assignment)

    def subjects(self, fold, role=None):
        if fold not in FOLD_NAMES or role not in ROLES+(None,):
            raise InputValidationError(f"fold must be one of {FOLD_NAMES}, role one of {ROLES} or None")
        return sorted(s for s, (f, r) in self.assignment.items() if f == fold and (role is None or r == role))

    def fold_of(self, subject):
        return self.assignment[subject][0]

    def in_fold(self, subject, fold):
        return self.assignment[subject][0] == fold

    def label_contact(self, fold):
        """Every subject whose labels the fold-`fold` models touch: train + calib, sorted."""
        return self.subjects(fold)

    def counts(self):
        return {f: {r: len(self.subjects(f, r)) for r in ROLES} for f in FOLD_NAMES}

    def record(self):
        return dict(folds_path=self.path, folds_sha256=self.sha256, folds_layout=self.layout, counts=self.counts(),
                    sealed_lists_checked=[list(item) for item in self.sealed_lists])


def _fold_name(value):
    if isinstance(value, str) and value in FOLD_NAMES:
        return value
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(FOLD_NAMES):
        return FOLD_NAMES[value]
    raise InputValidationError(f"folds: fold {value!r} is not one of {FOLD_NAMES} (or 0/1)")


def _role_name(value):
    if isinstance(value, str) and value.lower() in _ROLE_ALIASES:
        return _ROLE_ALIASES[value.lower()]
    raise InputValidationError(f"folds: role {value!r} is not one of {sorted(_ROLE_ALIASES)}")


def _add(assignment, subject, fold, role):
    if not _is_oasis(subject):
        raise InputValidationError(f"folds: {subject!r} is not a canonical OASIS id")
    if subject in assignment:
        raise InputValidationError(f"folds: subject {subject} listed twice")
    assignment[subject] = (_fold_name(fold), _role_name(role))


def _layout_a(doc):
    subjects = doc.get("subjects")
    if not isinstance(subjects, dict) or not subjects or not all(
            isinstance(v, dict) and "fold" in v and "role" in v for v in subjects.values()):
        return None
    out = {}
    for subject, item in subjects.items():
        _add(out, subject, item["fold"], item["role"])
    return out


def _layout_b(doc):
    found = None
    for key, value in doc.items():
        if isinstance(value, list) and value and all(
                isinstance(v, dict) and "fold" in v and "role" in v and any(k in v for k in _ID_KEYS) for v in value):
            if found is not None:
                raise InputValidationError("folds: more than one list of {subject, fold, role} records")
            found = value
    if found is None:
        return None
    out = {}
    for item in found:
        ids = [item[k] for k in _ID_KEYS if k in item]
        if len(ids) != 1:
            raise InputValidationError(f"folds: record {item!r} must name the subject exactly once")
        _add(out, ids[0], item["fold"], item["role"])
    return out


def _layout_c(doc):
    folds = doc.get("folds")
    if not isinstance(folds, dict) or not folds or not all(isinstance(v, dict) for v in folds.values()):
        return None
    out = {}
    for fold, roles in folds.items():
        fold_name = _fold_name(fold if fold not in ("0", "1") else int(fold))
        listed = {key: value for key, value in roles.items() if key.lower() in _ROLE_ALIASES}
        if not listed:
            return None
        for role, ids in listed.items():
            if not isinstance(ids, list):
                raise InputValidationError(f"folds.{fold}.{role}: list of subject ids required")
            for subject in ids:
                _add(out, subject, fold_name, role)
    return out


def _sealed_lists(node, path=()):
    """Yield (key path, ids) for every list of canonical ids under a key naming a hold-out/test/sealed set."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            here = path+(str(key),)
            if _SEALED_KEY.search(str(key)):
                ids = sorted(_collect_ids(value))
                if ids:
                    yield "/".join(here), ids
            else:
                yield from _sealed_lists(value, here)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _sealed_lists(value, path+(str(index),))


def _collect_ids(node):
    found = set()
    if isinstance(node, str):
        if any(p.fullmatch(node) for p in CANONICAL_SUBJECT_PATTERNS.values()):
            found.add(node)
    elif isinstance(node, Mapping):
        for key, value in node.items():
            found |= _collect_ids(key) | _collect_ids(value)
    elif isinstance(node, list):
        for value in node:
            found |= _collect_ids(value)
    return found


def parse_folds(document, *, path="<memory>", sha256=None):
    if not isinstance(document, dict):
        raise InputValidationError("folds: JSON object required")
    candidates = [document]+([document["training"]] if isinstance(document.get("training"), dict) else [])
    matches = []
    for where, doc in enumerate(candidates):
        for name, parser in (("subjects_map", _layout_a), ("record_list", _layout_b), ("fold_lists", _layout_c)):
            result = parser(doc)
            if result is not None:
                matches.append((("training/" if where else "")+name, result))
    if len(matches) != 1:
        raise InputValidationError(f"folds: expected exactly one recognised layout, found {[m[0] for m in matches]}")
    layout, assignment = matches[0]
    for fold in FOLD_NAMES:
        for role in ROLES:
            if not any(f == fold and r == role for f, r in assignment.values()):
                raise InputValidationError(f"folds: fold {fold} has no {role} subjects")
    training = set(assignment)
    sealed = []
    for key, ids in _sealed_lists(document):
        overlap = sorted(training & set(ids))
        if overlap:
            raise InputValidationError(f"folds: training subjects {overlap} also listed under {key!r}")
        sealed.append((key, len(ids)))
    if sha256 is None:
        sha256 = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
    return FoldSplit(path=str(path), sha256=sha256, layout=layout, assignment=dict(assignment), sealed_lists=tuple(sealed))


def load_folds(path, *, expect_counts=None):
    """Read folds.json; `expect_counts` = (n_train, n_calib) per fold, checked if given."""
    raw = Path(path).read_bytes()
    document = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_json_object)
    split = parse_folds(document, path=str(path), sha256=hashlib.sha256(raw).hexdigest())
    if expect_counts is not None:
        want = dict(zip(ROLES, expect_counts))
        for fold, got in split.counts().items():
            if got != want:
                raise InputValidationError(f"folds: fold {fold} has {got}, expected {want}")
    return split


# ---- full-training-set folds (unet35_v2_full) --------------------------------------------------------

FULL_FOLDS_SCHEMA = "bayesreg.folds_full.v2"
FULL_FOLDS_RULE = ("derived from folds.json: the calib of fold X is the calib of fold X in folds.json; the train of "
                   "fold X is every other training subject of folds.json (sorted); nothing else is read")
FULL_FOLDS_LAYOUT = "full_v2"


@dataclass(frozen=True)
class FullFoldSplit(FoldSplit):
    """Folds whose train lists overlap: `assignment` maps a subject to its ((fold, role), (fold, role)), `folds` is
    {fold: {role: sorted subjects}}, `derived_from` the record of the file's source. `fold_of` is refused, because a
    subject has no single fold: code that needs one (the population EB of bayesreg.learn) cannot take this split."""
    folds: Mapping = field(default_factory=dict)
    derived_from: Mapping = field(default_factory=dict)

    def subjects(self, fold, role=None):
        if fold not in FOLD_NAMES or role not in ROLES+(None,):
            raise InputValidationError(f"fold must be one of {FOLD_NAMES}, role one of {ROLES} or None")
        roles = ROLES if role is None else (role,)
        return sorted(s for r in roles for s in self.folds[fold][r])

    def fold_of(self, subject):
        raise InputValidationError(f"{subject}: a subject of the full-training-set folds is in both folds")

    def in_fold(self, subject, fold):
        return any(f == fold for f, _ in self.assignment[subject])

    def record(self):
        return dict(super().record(), derived_from=dict(self.derived_from))


def derive_full_folds(split):
    """{fold: {"train": sorted, "calib": sorted}} of the full-training-set rule, from a FoldSplit of folds.json."""
    if type(split) is not FoldSplit:
        raise InputValidationError("derive_full_folds: a FoldSplit of folds.json is required")
    everyone = sorted(split.training_subjects)
    folds = {}
    for fold in FOLD_NAMES:
        calib = split.subjects(fold, "calib")
        held = set(calib)
        folds[fold] = dict(train=[s for s in everyone if s not in held], calib=calib)
    return folds


def full_folds_document(split, *, source="splits/bayesreg_v1/folds.json"):
    """The JSON object of folds_full_v2.json; `source` is recorded as written (a repository-relative path)."""
    return dict(schema=FULL_FOLDS_SCHEMA, rule=FULL_FOLDS_RULE, derived_from=dict(file=source, sha256=split.sha256),
                folds=derive_full_folds(split))


def render_full_folds(document):
    """The bytes of folds_full_v2.json (the file is exactly this rendering, so it can be re-derived and compared)."""
    return (json.dumps(document, indent=1)+"\n").encode("utf-8")


def parse_full_folds(document, *, path="<memory>", sha256=None):
    """FullFoldSplit from the JSON object; InputValidationError unless each fold's train and calib are disjoint sorted
    canonical OASIS ids, both folds cover the same subjects, and the two calib lists are disjoint."""
    if not isinstance(document, dict) or document.get("schema") != FULL_FOLDS_SCHEMA:
        raise InputValidationError(f"full folds: schema must be {FULL_FOLDS_SCHEMA!r}")
    if set(document) != {"schema", "rule", "derived_from", "folds"}:
        raise InputValidationError(f"full folds: unexpected keys {sorted(document)}")
    source, folds = document["derived_from"], document["folds"]
    if not isinstance(source, dict) or set(source) != {"file", "sha256"}:
        raise InputValidationError("full folds: derived_from must be {file, sha256}")
    if not isinstance(folds, dict) or set(folds) != set(FOLD_NAMES):
        raise InputValidationError(f"full folds: folds must be exactly {FOLD_NAMES}")
    assignment, covered = {}, {}
    for fold in FOLD_NAMES:
        roles = folds[fold]
        if not isinstance(roles, dict) or set(roles) != set(ROLES):
            raise InputValidationError(f"full folds: fold {fold} must have exactly {ROLES}")
        for role in ROLES:
            ids = roles[role]
            if not isinstance(ids, list) or not ids or any(not _is_oasis(s) for s in ids):
                raise InputValidationError(f"full folds: {fold}.{role} must be a non-empty list of canonical OASIS ids")
            if ids != sorted(set(ids)):
                raise InputValidationError(f"full folds: {fold}.{role} must be sorted and distinct")
            for subject in ids:
                assignment.setdefault(subject, []).append((fold, role))
        overlap = sorted(set(roles["train"]) & set(roles["calib"]))
        if overlap:
            raise InputValidationError(f"full folds: fold {fold} has subjects both in train and calib: {overlap[:5]}")
        covered[fold] = set(roles["train"]) | set(roles["calib"])
    if covered["A"] != covered["B"]:
        raise InputValidationError(f"full folds: the folds cover different subjects: "
                                   f"{sorted(covered['A'] ^ covered['B'])[:5]}")
    both_calib = sorted(set(folds["A"]["calib"]) & set(folds["B"]["calib"]))
    if both_calib:
        raise InputValidationError(f"full folds: subjects in the calib of both folds: {both_calib[:5]}")
    if sha256 is None:
        sha256 = hashlib.sha256(render_full_folds(document)).hexdigest()
    return FullFoldSplit(path=str(path), sha256=sha256, layout=FULL_FOLDS_LAYOUT,
                         assignment={s: tuple(a) for s, a in assignment.items()},
                         folds={f: {r: tuple(folds[f][r]) for r in ROLES} for f in FOLD_NAMES},
                         derived_from=dict(source))


def load_full_folds(path, *, expect_counts=None):
    """Read folds_full_v2.json; `expect_counts` = (n_train, n_calib) per fold, checked if given."""
    raw = Path(path).read_bytes()
    document = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_json_object)
    split = parse_full_folds(document, path=str(path), sha256=hashlib.sha256(raw).hexdigest())
    if expect_counts is not None:
        want = dict(zip(ROLES, expect_counts))
        for fold, got in split.counts().items():
            if got != want:
                raise InputValidationError(f"full folds: fold {fold} has {got}, expected {want}")
    return split


def check_full_folds_derivation(split, source):
    """`split` (FullFoldSplit) is what the rule gives from `source` (FoldSplit of folds.json), and names its sha256."""
    if split.derived_from.get("sha256") != source.sha256:
        raise InputValidationError(f"full folds: derived from sha256 {split.derived_from.get('sha256')}, the folds file "
                                   f"has {source.sha256}")
    want = derive_full_folds(source)
    if any(list(split.folds[f][r]) != want[f][r] for f in FOLD_NAMES for r in ROLES):
        raise InputValidationError("full folds: the file is not what the rule gives from the folds file")
    return dict(check="full folds re-derived from folds.json", source_path=source.path, source_sha256=source.sha256,
                rule=FULL_FOLDS_RULE)


# ---- label guard ----------------------------------------------------------------------------------

class LabelGuard:
    """Admits label reads only for B-line training subjects (see the module docstring)."""

    def __init__(self, split, allowlist, *, training_source, denied=()):
        if not isinstance(split, FoldSplit):
            raise InputValidationError("split: a FoldSplit from load_folds is required")
        if not isinstance(allowlist, DevAllowlist):
            raise InputValidationError("allowlist: a DevAllowlist from load_dev_allowlist is required")
        training = split.training_subjects
        problems = []
        sources = {source["name"]: source for source in allowlist.sources}
        if training_source not in sources:
            problems.append(f"the allowlist has no training source named {training_source!r}")
        else:
            listed = set(sources[training_source]["subjects"])
            if sources[training_source]["never_for_training"]:
                problems.append(f"training source {training_source!r} is never_for_training")
            if listed != set(training):
                problems.append(f"training source {training_source!r} differs from the folds: only in the source "
                                f"{sorted(listed-training)[:5]}, only in the folds {sorted(training-listed)[:5]}")
        off = sorted(s for s in training if not allowlist.allows(s))
        if off:
            problems.append(f"not on the development allowlist: {off}")
        never = sorted(training & allowlist.never_for_training)
        if never:
            problems.append(f"never_for_training (T0): {never}")
        for source in allowlist.sources:
            if source["name"] == training_source:
                continue
            clash = sorted(set(source["subjects"]) & training)
            if clash:
                problems.append(f"also in development source {source['name']!r}: {clash}")
        l2r = sorted(s for s in training if L2R_VAL_RANGE[0] <= _oasis_number(s) <= L2R_VAL_RANGE[1])
        if l2r:
            problems.append(f"in the Learn2Reg validation range: {l2r}")
        bad = sorted(training & set(denied))
        if bad:
            problems.append(f"explicitly denied: {bad}")
        if problems:
            raise LabelAccessRefused("folds refused, labels of these subjects may not be read: "+"; ".join(problems))
        self.split, self.allowlist, self.denied = split, allowlist, frozenset(denied)
        self.training_source = training_source
        self.reads = []

    def check(self, subject, fold=None):
        if not _is_oasis(subject) or subject not in self.split.assignment:
            raise LabelAccessRefused(f"{subject!r} is not a B-line training subject; its labels are not read")
        if fold is not None and not self.split.in_fold(subject, fold):
            raise LabelAccessRefused(f"{subject} belongs to fold {self.split.fold_of(subject)}, not {fold}")

    def record(self):
        return dict(guard="segmenter-label-guard-v2", allowlist_sha256=self.allowlist.sha256,
                    allowlist_path=self.allowlist.path, training_source=self.training_source,
                    l2r_val_range=list(L2R_VAL_RANGE),
                    denied_sha256=subject_list_sha256(sorted(self.denied)), label_reads=list(self.reads),
                    **self.split.record())


# ---- files ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class OasisFiles:
    """neurite-oasis layout: <root>/<subject>/aligned_norm.nii.gz, aligned_seg35.nii.gz (read-only).

    Sealed guard (C23.6-1, D104 (a)): for a subject on a sealed list (bayesreg.io.sealed), label_path always refuses
    (SubjectSealed) and image_path only answers on an OasisFiles of bayesreg.io.unseal.bind_files, whose bound
    allowlist then takes the subject through the whole guard (bayesreg.io.guard.check_subject).  `allowlist` is None
    except on that object; an OasisFiles carrying one that bind_files did not make is refused for every subject
    (D100).  Other subjects: as before."""
    root: Path = DEFAULT_DATA_ROOT
    allowlist: object = field(default=None, compare=False, repr=False)

    def image_path(self, subject):
        if not _is_oasis(subject):
            raise InputValidationError(f"{subject!r}: canonical OASIS id required")
        _sealed_files_check(self, subject, "image")
        return Path(self.root)/subject/IMAGE_FILE

    def label_path(self, subject, kind="seg35"):
        if not _is_oasis(subject):
            raise InputValidationError(f"{subject!r}: canonical OASIS id required")
        _sealed_files_check(self, subject, "label")
        return Path(self.root)/subject/LABEL_FILES[kind]


def _sealed_files_check(files, subject, what):
    """C23.6-1 for OasisFiles.image_path / label_path (admission only; no file is opened here)."""
    from ...io.guard import SubjectSealed, check_subject
    from ...io.sealed import guard_sealed_lists, lists_of
    from ...io.unseal import files_problem
    problem = files_problem(files)
    if problem:
        raise SubjectSealed(f"{what} path of {subject!r} refused: {problem} (C23.4-1, D100)")
    on = lists_of(guard_sealed_lists(), subject)
    if not on:
        return
    if what == "label":
        raise SubjectSealed(f"label path of {subject!r} refused: on the sealed list {list(on)}; the labels of a sealed "
                            "subject are never read through OasisFiles (C23.6-1; the hold-out evaluation reads its own, "
                            "C23.12)")
    if files.allowlist is None:
        raise SubjectSealed(f"image path of {subject!r} refused: on the sealed list {list(on)}; only an OasisFiles of "
                            "bayesreg.io.unseal.bind_files answers for a sealed subject (C23.4-1)")
    verdict = check_subject(subject, files.allowlist)
    if not verdict.admitted:
        raise SubjectSealed(f"image path refused: {verdict.message or f'{subject!r} is not on the development allowlist'}")


def read_image(files, subject):
    """aligned_norm as jd_curl load_scalar returns it: float64 (nz, ny, nx), RAS storage; plus metadata."""
    volume = load_scalar(files.image_path(subject))
    return volume.values, dict(volume.metadata, subject=subject, grid_shape_zyx=list(volume.grid.shape),
                               spacing_xyz=list(volume.grid.spacing),
                               origin_xyz=[float(a[0]) for a in volume.grid.axes])


def read_labels(files, subject, guard, *, fold=None, kind="seg35"):
    """Label map (uint8, zyx) of a training subject; the guard runs before the file is opened."""
    if not isinstance(guard, LabelGuard):
        raise InputValidationError("guard: a LabelGuard is required to read labels")
    guard.check(subject, fold)
    volume = load_scalar(files.label_path(subject, kind), labels=True)
    values = volume.values
    if values.max() >= LABEL_RANGE[kind]:
        raise InputValidationError(f"{subject} {kind}: label {values.max():g} outside 0..{LABEL_RANGE[kind]-1}")
    guard.reads.append(dict(subject=subject, kind=kind, sha256=volume.metadata["sha256"]))
    return values.astype(np.uint8), dict(subject=subject, kind=kind, sha256=volume.metadata["sha256"],
                                         grid_shape_zyx=list(volume.grid.shape), spacing_xyz=list(volume.grid.spacing),
                                         origin_xyz=[float(a[0]) for a in volume.grid.axes])


def normalize_image(values, *, percentile=99.5):
    """Divide by the `percentile` of the brain voxels (values > 0; the images are skull-stripped)."""
    values = np.asarray(values, dtype=np.float64)
    brain = values > 0
    if brain.sum() < 1:
        raise InputValidationError("normalize_image: no voxel > 0")
    scale = float(np.percentile(values[brain], percentile))
    if not np.isfinite(scale) or scale <= 0:
        raise InputValidationError("normalize_image: non-positive percentile")
    return (values/scale).astype(np.float32), dict(percentile=percentile, scale=scale, brain_voxels=int(brain.sum()),
                                                  brain_definition="aligned_norm > 0")


@dataclass
class Subject:
    subject: str
    image: np.ndarray                   # float32 (nz, ny, nx), normalized
    labels: np.ndarray = None           # uint8 (nz, ny, nx) or None
    image_sha256: str = None
    label_sha256: str = None
    normalization: dict = None


def load_training_subjects(files, guard, fold, role, *, percentile=99.5, log=None):
    """Read image + labels of every `role` subject of `fold` into memory (no disk cache)."""
    out = []
    for subject in guard.split.subjects(fold, role):
        labels, label_meta = read_labels(files, subject, guard, fold=fold)
        image, meta = read_image(files, subject)
        grid_keys = ("grid_shape_zyx", "spacing_xyz", "origin_xyz")
        if any(not np.allclose(meta[k], label_meta[k], rtol=0, atol=1e-6) for k in grid_keys):
            raise InputValidationError(f"{subject}: image and label grids differ")
        image, norm = normalize_image(image, percentile=percentile)
        out.append(Subject(subject, image, labels, meta["sha256"], label_meta["sha256"], norm))
        if log:
            log(f"loaded {subject} ({fold}/{role}) shape {image.shape}")
    return out


# ---- augmentation (torch) -------------------------------------------------------------------------

DEFAULT_AUGMENT = dict(p_spatial=.5, max_rotation_deg=10., max_translation_vox=8., p_gamma=.3, gamma_range=(.7, 1.5),
                       p_scale_shift=.5, scale_range=(.9, 1.1), shift_range=(-.05, .05))


def _rotation(angles):
    ax, ay, az = angles
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz@ry@rx


def spatial_theta(shape_zyx, angles_rad, translation_vox):
    """3x4 affine_grid theta (x = W, y = H, z = D order, align_corners=True) for a rotation about the
    volume centre and a translation, both in voxel (isotropic) units -- no shear from the aspect ratio."""
    half = np.array([(shape_zyx[2]-1)/2, (shape_zyx[1]-1)/2, (shape_zyx[0]-1)/2])
    half = np.where(half > 0, half, 1.)
    rotation = _rotation(angles_rad)
    linear = np.diag(1/half)@rotation@np.diag(half)
    return np.concatenate([linear, (np.asarray(translation_vox)/half)[:, None]], axis=1)


def augment(image, labels, rng, config=None):
    """Random intensity (gamma, scale/shift on brain voxels) and small rigid spatial transforms.

    `image` (1, 1, D, H, W) float tensor, `labels` (1, D, H, W) long tensor, same device. No left-right
    flip (the seg35 labels are lateralized). Returns (image, labels, record).
    """
    import torch
    import torch.nn.functional as F
    cfg = dict(DEFAULT_AUGMENT, **(config or {}))
    record = {}
    brain = image > 0
    if rng.random() < cfg["p_gamma"]:
        gamma = float(rng.uniform(*cfg["gamma_range"]))
        peak = image.max().clamp_min(1e-6)
        image = torch.where(brain, peak*(image.clamp_min(0)/peak)**gamma, image)
        record["gamma"] = gamma
    if rng.random() < cfg["p_scale_shift"]:
        scale, shift = float(rng.uniform(*cfg["scale_range"])), float(rng.uniform(*cfg["shift_range"]))
        image = torch.where(brain, image*scale+shift, image)
        record.update(scale=scale, shift=shift)
    if rng.random() < cfg["p_spatial"]:
        angles = np.deg2rad(rng.uniform(-cfg["max_rotation_deg"], cfg["max_rotation_deg"], 3))
        shift = rng.uniform(-cfg["max_translation_vox"], cfg["max_translation_vox"], 3)
        theta = torch.as_tensor(spatial_theta(image.shape[2:], angles, shift), dtype=image.dtype,
                                device=image.device)[None]
        grid = F.affine_grid(theta, list(image.shape), align_corners=True)
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        labels = F.grid_sample(labels[:, None].to(image.dtype), grid, mode="nearest", padding_mode="zeros",
                               align_corners=True)[:, 0].round().long()
        record.update(rotation_deg=np.rad2deg(angles).tolist(), translation_vox=shift.tolist())
    return image, labels, record


def random_patch(image, labels, patch, rng, *, p_foreground=1/3):
    """Crop a patch (D, H, W) from padded tensors; with probability p_foreground centred on a label > 0."""
    shape = image.shape[2:]
    if any(p > n for p, n in zip(patch, shape)):
        raise InputValidationError(f"patch {patch} larger than volume {tuple(shape)}")
    if rng.random() < p_foreground and bool((labels > 0).any()):
        idx = (labels[0] > 0).nonzero()
        centre = idx[int(rng.integers(len(idx)))].tolist()
        start = [min(max(c-p//2, 0), n-p) for c, p, n in zip(centre, patch, shape)]
    else:
        start = [int(rng.integers(0, n-p+1)) for p, n in zip(patch, shape)]
    sl = tuple(slice(s, s+p) for s, p in zip(start, patch))
    return image[(slice(None), slice(None))+sl], labels[(slice(None),)+sl]


__all__ = ["DEFAULT_AUGMENT", "DEFAULT_DATA_ROOT", "FOLD_INDEX", "FOLD_NAMES", "FULL_FOLDS_LAYOUT", "FULL_FOLDS_RULE",
           "FULL_FOLDS_SCHEMA", "FoldSplit", "FullFoldSplit", "LabelAccessRefused", "LabelGuard", "OasisFiles", "ROLES",
           "Subject", "augment", "check_full_folds_derivation", "derive_full_folds", "full_folds_document", "load_folds",
           "load_full_folds", "load_training_subjects", "normalize_image", "parse_folds", "parse_full_folds",
           "random_patch", "read_image", "read_labels", "render_full_folds", "spatial_theta"]
