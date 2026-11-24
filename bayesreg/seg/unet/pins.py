"""Pinned inputs of the real segmenter runs (review 2026-10-30, item 6) and the synthetic switch.

Real data (the default of every cli subcommand):
  - the code is a verified snapshot or a clean git worktree (snapshot.require_clean_code);
  - folds: the file's sha256 equals PINNED_FOLDS_SHA256 (splits/bayesreg_v1/folds.json, commit 044426f),
    per-fold counts PINNED_COUNTS = (45 train, 8 calib);
  - allowlist: exactly this code tree's configs/bayesreg/dev_subjects.json (resolved path; the tree is
    clean or a verified snapshot, so the file is the committed one), and
    bayesreg.io.verify_allowlist_sources passes against the source repository (the snapshot's
    source_repo, where the data/ link and the stage lists live);
  - the training source is named, TRAINING_SOURCE = "bayesreg_train_v1", and its subject set equals the
    folds' subject set (no inference from subset relations).
Synthetic (`--synthetic`, dry runs and tests): none of the pins, but every subject of the folds file and
of the allowlist must be a synthetic OASIS id (number >= SYNTHETIC_MIN; real OASIS ids end at 0457) and
the data root must not be a real data root, so the switch cannot reach real labels.

Full-training-set segmenter unet35_v2_full (cli --profile v2_full, 04_分割器.md §14), `load_pinned_full_inputs`: the same
pins with folds_full_v2.json in place of folds.json (its own sha256 and counts (98, 8)); in addition the file must be what
the rule gives from folds.json, which must have its pinned sha256. Nothing above changes for v1.
"""
import hashlib
from pathlib import Path

from jd_curl.validation import InputValidationError

from ...io.guard import load_dev_allowlist, verify_allowlist_sources
from ...io.pair import subject_list_sha256
from .data import check_full_folds_derivation, load_folds, load_full_folds
from .snapshot import CODE_ROOT, code_record, require_clean_code

PINNED_FOLDS_SHA256 = "98e9e676053d4e9386bc775c5d4cfc6be097afca9919c655ccb554d6236bf245"
TRAINING_SOURCE = "bayesreg_train_v1"
PINNED_COUNTS = (45, 8)
SYNTHETIC_MIN = 900
DEFAULT_FOLDS = CODE_ROOT/"splits"/"bayesreg_v1"/"folds.json"
DEFAULT_ALLOWLIST = CODE_ROOT/"configs"/"bayesreg"/"dev_subjects.json"
PINNED_FULL_FOLDS_SHA256 = "4573b7900b5ae4e7e3e02ff41c0a756f2ff48a8f84a0e7743965ec17b14558dc"
PINNED_FULL_COUNTS = (98, 8)
DEFAULT_FULL_FOLDS = CODE_ROOT/"splits"/"bayesreg_v1"/"folds_full_v2.json"
REAL_DATA_ROOTS = (Path("/mnt/d/data"), Path("/home/rmcfz/code/python_projects/seq_vp_3d/data"))


class PinnedInputRefused(Exception):
    """An input of a real run is not the pinned one; status `pinned_input_refused`."""
    status = "pinned_input_refused"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_root_for(code):
    """Where data/ and the allowlist source files live: the snapshot's source repository, else the tree."""
    return Path(code["source_repo"]) if code.get("kind") == "snapshot" else Path(code.get("root", CODE_ROOT))


def training_source_check(split, allowlist, training_source):
    """The allowlist source `training_source` exists, is trainable and lists exactly the folds' subjects."""
    sources = {s["name"]: s for s in allowlist.sources}
    if training_source not in sources:
        raise PinnedInputRefused(f"allowlist has no source named {training_source!r}")
    source = sources[training_source]
    if source["never_for_training"]:
        raise PinnedInputRefused(f"allowlist source {training_source!r} is never_for_training")
    listed, training = set(source["subjects"]), set(split.training_subjects)
    if listed != training:
        raise PinnedInputRefused(f"allowlist source {training_source!r} ({len(listed)} subjects) differs from the folds "
                                 f"({len(training)}): only in the source {sorted(listed-training)[:5]}, only in the "
                                 f"folds {sorted(training-listed)[:5]}")
    return dict(training_source=training_source, training_source_path=source["path"],
                training_source_file_sha256=source["sha256"],
                training_source_subjects_sha256=subject_list_sha256(sorted(listed)))


def load_pinned_allowlist(allowlist_path, *, source_root=None, code=None):
    """The repository allowlist with its sources re-hashed; PinnedInputRefused otherwise."""
    code = require_clean_code(code if code is not None else code_record())
    if Path(allowlist_path).resolve() != DEFAULT_ALLOWLIST.resolve():
        raise PinnedInputRefused(f"allowlist must be {DEFAULT_ALLOWLIST} for real data, got {allowlist_path}")
    allowlist = load_dev_allowlist(allowlist_path)
    root = Path(source_root) if source_root is not None else source_root_for(code)
    try:
        checked = verify_allowlist_sources(allowlist, root)
    except InputValidationError as exc:
        raise PinnedInputRefused(f"allowlist sources do not verify against {root}: {exc}") from None
    return allowlist, dict(allowlist_path=str(allowlist_path), allowlist_sha256=allowlist.sha256,
                           allowlist_sources_verified=checked, source_root=str(root), code=code)


def load_pinned_inputs(folds_path, allowlist_path, *, training_source=TRAINING_SOURCE, expect_counts=PINNED_COUNTS,
                       source_root=None, code=None):
    """(split, allowlist, record) of a real run; PinnedInputRefused on any departure from the pins."""
    if training_source != TRAINING_SOURCE:
        raise PinnedInputRefused(f"the training source of real runs is {TRAINING_SOURCE!r}, got {training_source!r}")
    if expect_counts is None or tuple(expect_counts) != PINNED_COUNTS:
        raise PinnedInputRefused(f"real runs require per-fold counts {PINNED_COUNTS}, got {expect_counts}")
    folds_sha = _sha256(folds_path)
    if folds_sha != PINNED_FOLDS_SHA256:
        raise PinnedInputRefused(f"{folds_path}: sha256 {folds_sha} is not the pinned {PINNED_FOLDS_SHA256}")
    allowlist, record = load_pinned_allowlist(allowlist_path, source_root=source_root, code=code)
    split = load_folds(folds_path, expect_counts=expect_counts)
    record.update(mode="pinned", folds_path=str(folds_path), folds_sha256=split.sha256,
                  pinned_folds_sha256=PINNED_FOLDS_SHA256, counts=split.counts(),
                  **training_source_check(split, allowlist, training_source))
    return split, allowlist, record


def load_pinned_full_inputs(folds_path, allowlist_path, *, training_source=TRAINING_SOURCE,
                            expect_counts=PINNED_FULL_COUNTS, source_root=None, code=None, v1_folds_path=None):
    """(split, allowlist, record) of a real run of unet35_v2_full; PinnedInputRefused on any departure from the pins
    or if the file is not what the rule gives from folds.json (`v1_folds_path`, default the file beside it)."""
    if training_source != TRAINING_SOURCE:
        raise PinnedInputRefused(f"the training source of real runs is {TRAINING_SOURCE!r}, got {training_source!r}")
    if expect_counts is None or tuple(expect_counts) != PINNED_FULL_COUNTS:
        raise PinnedInputRefused(f"real v2 runs require per-fold counts {PINNED_FULL_COUNTS}, got {expect_counts}")
    folds_sha = _sha256(folds_path)
    if folds_sha != PINNED_FULL_FOLDS_SHA256:
        raise PinnedInputRefused(f"{folds_path}: sha256 {folds_sha} is not the pinned {PINNED_FULL_FOLDS_SHA256}")
    v1_path = Path(folds_path).parent/"folds.json" if v1_folds_path is None else Path(v1_folds_path)
    v1_sha = _sha256(v1_path)
    if v1_sha != PINNED_FOLDS_SHA256:
        raise PinnedInputRefused(f"{v1_path}: sha256 {v1_sha} is not the pinned {PINNED_FOLDS_SHA256}")
    split = load_full_folds(folds_path, expect_counts=expect_counts)
    try:
        derivation = check_full_folds_derivation(split, load_folds(v1_path, expect_counts=PINNED_COUNTS))
    except InputValidationError as exc:
        raise PinnedInputRefused(str(exc)) from None
    allowlist, record = load_pinned_allowlist(allowlist_path, source_root=source_root, code=code)
    record.update(mode="pinned_full", folds_path=str(folds_path), folds_sha256=split.sha256,
                  pinned_folds_sha256=PINNED_FULL_FOLDS_SHA256, counts=split.counts(), derivation=derivation,
                  **training_source_check(split, allowlist, training_source))
    return split, allowlist, record


def _synthetic_number(subject):
    parts = subject.split("_") if isinstance(subject, str) else []
    return int(parts[2]) if len(parts) == 4 and subject.startswith("OASIS_OAS1_") and parts[2].isdigit() else -1


def check_synthetic_subjects(subjects, where):
    real = sorted(s for s in subjects if _synthetic_number(s) < SYNTHETIC_MIN)
    if real:
        raise PinnedInputRefused(f"--synthetic: {where} lists non-synthetic subjects {real[:5]} "
                                 f"(synthetic ids are OASIS_OAS1_{SYNTHETIC_MIN:04d}_MR1 and above)")


def check_synthetic_data_root(data_root):
    resolved = Path(data_root).resolve()
    for real in REAL_DATA_ROOTS+((CODE_ROOT/"data"),):
        real = real.resolve()
        if resolved == real or real in resolved.parents:
            raise PinnedInputRefused(f"--synthetic: data root {data_root} is a real data root ({real})")


def load_synthetic_inputs(folds_path, allowlist_path, *, data_root, training_source, expect_counts=None,
                          loader=load_folds):
    """(split, allowlist, record) of a synthetic run (no pins; synthetic ids and data root only). `loader`: load_folds
    or, for the full-training-set folds, load_full_folds."""
    check_synthetic_data_root(data_root)
    split = loader(folds_path, expect_counts=expect_counts)
    allowlist = load_dev_allowlist(allowlist_path)
    check_synthetic_subjects(split.training_subjects, "the folds file")
    check_synthetic_subjects([s for ids in allowlist.subjects.values() for s in ids], "the allowlist")
    record = dict(mode="synthetic", folds_path=str(folds_path), folds_sha256=split.sha256,
                  allowlist_path=str(allowlist_path), allowlist_sha256=allowlist.sha256, data_root=str(data_root),
                  counts=split.counts(), code=code_record(), **training_source_check(split, allowlist, training_source))
    return split, allowlist, record


__all__ = ["DEFAULT_ALLOWLIST", "DEFAULT_FOLDS", "DEFAULT_FULL_FOLDS", "PINNED_COUNTS", "PINNED_FOLDS_SHA256",
           "PINNED_FULL_COUNTS", "PINNED_FULL_FOLDS_SHA256", "PinnedInputRefused", "REAL_DATA_ROOTS", "SYNTHETIC_MIN",
           "TRAINING_SOURCE", "check_synthetic_data_root", "check_synthetic_subjects", "load_pinned_allowlist",
           "load_pinned_full_inputs", "load_pinned_inputs", "load_synthetic_inputs", "source_root_for",
           "training_source_check"]
