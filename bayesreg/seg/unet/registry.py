"""Registration of the fold models in the segmenter registry (04_分割器.md §7).

Schema bayesreg.segmenters.v1 (bayesreg/io/pair.py, configs/bayesreg/README.md) keys the registry by
fold model and allows one model per (segmenter_id, fold). So the ensemble is ONE segmenter_id
(`unet35_v1_ens`) with one registered model per fold (`unet35_v1_foldA` -> fold 0,
`unet35_v1_foldB` -> fold 1); each fold model lists the sha256 of its seed members (member.pt, which
carries the weights and the temperature) and its label contact = that fold's train + calib subjects.

Before anything is registered, `verify_members` checks every member against the folds file, the
allowlist and its own records (review 2026-10-30, item 3); any failure refuses the whole registration:
  - member.pt, train_record.json and calibration.json re-hashed against member.json;
  - label_contact_subjects == the folds' train + calib of that fold, in member.json, train_record and
    inside member.pt (with fold, seed, temperature and model_best_sha256; for a member of the last step,
    large.py checkpoint "last", also checkpoint_choice, checkpoint_file, checkpoint_sha256 and
    checkpoint_iteration, and that file re-hashed);
  - folds_sha256 of every member == the folds file given (== the pinned sha256 for real data);
  - a label-guard record exists in train_record.json and in calibration.json (a member trained or
    calibrated with `subjects=` bypassing the guard is refused), with the same folds, allowlist and
    training source; training reads == the contact (each read subject in the contact, every contact
    subject read); calibration reads == the fold's calib subjects, once each;
  - every contact subject is on the allowlist, not never_for_training, and in the training source.
A fold model is registered once all its expected seeds pass; entries are append-only (an existing
model id is accepted only if identical). The file is validated with bayesreg.io.load_segmenter_registry
before it replaces the old one. Then <models_root>/ensemble.json (index of the members, the member check and the
registry's sha256 -- the gate of the staging arm, bayesreg.seg.structure_layer.check_staging_ensemble) is written
by `write_ensemble`: a different existing ensemble.json is first kept as ensemble_before_<UTC>.json (review
PS1_PS2 item a).

Full-training-set segmenter unet35_v2_full (04_分割器.md §14), `register(..., full_training_set=True)`: one segmenter_id
`unet35_v2_full` with the fold models `unet35_v2_full_foldA`, `_foldB`. Every member saw the labels of all 106 training
subjects, so the two label contacts are equal instead of disjoint (no cross-fitting; a training subject has no admissible
model); a fold needs at least one calibrated seed instead of every seed. All the member checks are the same.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from jd_curl.validation import InputValidationError

from ...io.guard import DevAllowlist, strict_json_object
from ...io.pair import REGISTRY_SCHEMA, load_segmenter_registry, subject_list_sha256
from .data import FOLD_INDEX, FOLD_NAMES, FoldSplit
from .train import file_sha256, write_json

SEGMENTER_ID = "unet35_v1_ens"
MODEL_PREFIX = "unet35_v1"
SEGMENTER_ID_V2 = "unet35_v2_full"
MODEL_PREFIX_V2 = "unet35_v2_full"
DEFAULT_SEEDS = (0, 1)
REPO_REGISTRY = Path(__file__).resolve().parents[3]/"configs"/"bayesreg"/"segmenters.json"
CHECK_FORMAT = "bayesreg.unet35.member_check.v1"
ENSEMBLE_FILE = "ensemble.json"                 # = bayesreg.seg.structure_layer.ENSEMBLE_FILE


class MemberCheckFailed(InputValidationError):
    """A member does not match the folds file, the allowlist or its own records; nothing is registered."""
    status = "member_check_failed"


def model_id_for(fold, prefix=MODEL_PREFIX):
    return f"{prefix}_fold{fold}"


def read_members(models_root):
    """member.json records under <models_root>/*/, each checked against its member.pt sha256."""
    members = []
    for record_path in sorted(Path(models_root).glob("*/member.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        member_path = record_path.parent/record["member_file"]
        if file_sha256(member_path) != record["member_sha256"]:
            raise InputValidationError(f"{member_path}: sha256 differs from {record_path}")
        if subject_list_sha256(record["label_contact_subjects"]) != record["label_contact_subjects_sha256"]:
            raise InputValidationError(f"{record_path}: label_contact_subjects do not match their sha256")
        members.append(dict(record, path=str(member_path)))
    return members


def find_member_files(models_root):
    """{sha256: path} of every <models_root>/*/member.pt."""
    return {file_sha256(p): p for p in sorted(Path(models_root).glob("*/member.pt"))}


def _hashed_json(path, expected, where):
    if not path.is_file():
        raise MemberCheckFailed(f"{where}: {path.name} missing")
    if file_sha256(path) != expected:
        raise MemberCheckFailed(f"{where}: {path.name} sha256 differs from member.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _guard_reads(guard, where, *, split, allowlist, training_source):
    if not isinstance(guard, dict):
        raise MemberCheckFailed(f"{where}: no label-guard record (labels not read through LabelGuard); refused")
    reads = guard.get("label_reads")
    if not isinstance(reads, list) or not reads:
        raise MemberCheckFailed(f"{where}: the label-guard record lists no label reads; refused")
    if guard.get("folds_sha256") != split.sha256:
        raise MemberCheckFailed(f"{where}: label guard folds_sha256 {guard.get('folds_sha256')} != {split.sha256}")
    if guard.get("allowlist_sha256") != allowlist.sha256:
        raise MemberCheckFailed(f"{where}: label guard allowlist_sha256 differs from the allowlist given "
                                f"({allowlist.path}); register with the allowlist the member was trained with")
    if guard.get("training_source") != training_source:
        raise MemberCheckFailed(f"{where}: label guard training_source {guard.get('training_source')!r} "
                                f"!= {training_source!r}")
    if any(not isinstance(r, dict) or r.get("kind") != "seg35" for r in reads):
        raise MemberCheckFailed(f"{where}: label reads other than seg35")
    return [r["subject"] for r in reads]


def verify_members(models_root, *, split, allowlist, training_source, pinned_folds_sha256=None):
    """(members, check record) after the item-3 checks of the module docstring; MemberCheckFailed otherwise."""
    import torch
    from .calibrate import CHECKPOINT_KEYS, MEMBER_FORMAT, names_a_checkpoint
    if not isinstance(split, FoldSplit):
        raise InputValidationError("split: a FoldSplit from load_folds is required")
    if not isinstance(allowlist, DevAllowlist):
        raise InputValidationError("allowlist: a DevAllowlist from load_dev_allowlist is required")
    if pinned_folds_sha256 is not None and split.sha256 != pinned_folds_sha256:
        raise MemberCheckFailed(f"folds sha256 {split.sha256} is not the pinned {pinned_folds_sha256}")
    sources = {s["name"]: s for s in allowlist.sources}
    if training_source not in sources or set(sources[training_source]["subjects"]) != set(split.training_subjects):
        raise MemberCheckFailed(f"allowlist source {training_source!r} missing or not equal to the folds' subjects")
    source_subjects = set(sources[training_source]["subjects"])
    members = read_members(models_root)
    if not members:
        raise MemberCheckFailed(f"no member.json under {models_root}")
    checked = []
    for m in members:
        directory = Path(m["path"]).parent
        where = str(directory)
        fold, seed, contact = m.get("fold"), m.get("seed"), m["label_contact_subjects"]
        if fold not in FOLD_NAMES or not isinstance(seed, int) or isinstance(seed, bool):
            raise MemberCheckFailed(f"{where}: fold {fold!r} / seed {seed!r} malformed")
        want_train, want_calib = split.subjects(fold, "train"), split.subjects(fold, "calib")
        if contact != split.label_contact(fold):
            raise MemberCheckFailed(f"{where}: label_contact_subjects != folds train + calib of fold {fold}")
        if m.get("folds_sha256") != split.sha256:
            raise MemberCheckFailed(f"{where}: folds_sha256 {m.get('folds_sha256')} != the folds file {split.sha256}")
        train_record = _hashed_json(directory/"train_record.json", m["train_record_sha256"], where)
        calibration = _hashed_json(directory/"calibration.json", m["calibration_sha256"], where)
        if (train_record.get("fold"), train_record.get("seed")) != (fold, seed) \
                or (calibration.get("fold"), calibration.get("seed")) != (fold, seed):
            raise MemberCheckFailed(f"{where}: fold/seed of the records differ from member.json")
        if sorted(train_record["train_subjects"]) != want_train or sorted(train_record["calib_subjects"]) != want_calib \
                or train_record["label_contact_subjects"] != contact:
            raise MemberCheckFailed(f"{where}: train_record subjects differ from the folds file")
        read = _guard_reads(train_record.get("label_guard"), where+" (training)", split=split, allowlist=allowlist,
                            training_source=training_source)
        outside = sorted(set(read)-set(contact))
        if outside:
            raise MemberCheckFailed(f"{where}: training read labels outside the label contact: {outside}")
        unread = sorted(set(contact)-set(read))
        if unread:
            raise MemberCheckFailed(f"{where}: contact subjects without a recorded training read: {unread[:5]}")
        calib_read = _guard_reads(calibration.get("label_guard"), where+" (calibration)", split=split,
                                  allowlist=allowlist, training_source=training_source)
        if sorted(calib_read) != want_calib or max(Counter(calib_read).values()) != 1 \
                or sorted(calibration.get("calib_subjects", [])) != want_calib:
            raise MemberCheckFailed(f"{where}: calibration reads {sorted(calib_read)} != the fold's calib subjects once each")
        blob = torch.load(m["path"], map_location="cpu", weights_only=True)
        inside = dict(format=blob.get("format"), fold=blob.get("fold"), seed=blob.get("seed"),
                      contact=list(blob.get("label_contact_subjects") or []), temperature=blob.get("temperature"),
                      model_best_sha256=blob.get("model_best_sha256"))
        expected = dict(format=MEMBER_FORMAT, fold=fold, seed=seed, contact=contact, temperature=m["temperature"],
                        model_best_sha256=train_record["model_best_sha256"])
        named = names_a_checkpoint(train_record) or any(k in blob for k in CHECKPOINT_KEYS)     # a member of the last step (large.py, checkpoint "last"): the weights are not model_best.pt's
        if named:
            inside.update({k: blob.get(k) for k in CHECKPOINT_KEYS})
            expected.update({k: train_record.get(k) for k in CHECKPOINT_KEYS})
            ckpt = directory/str(train_record.get("checkpoint_file"))
            if not ckpt.is_file() or file_sha256(ckpt) != train_record.get("checkpoint_sha256"):
                raise MemberCheckFailed(f"{where}: {ckpt.name} missing or its sha256 differs from train_record.checkpoint_sha256")
        if inside != expected:
            bad = sorted(k for k in expected if inside[k] != expected[k])
            raise MemberCheckFailed(f"{where}: member.pt disagrees with its records on {bad}")
        off = sorted(s for s in contact if not allowlist.allows(s))
        never = sorted(set(contact) & allowlist.never_for_training)
        foreign = sorted(set(contact)-source_subjects)
        if off or never or foreign:
            raise MemberCheckFailed(f"{where}: contact off the allowlist {off}, never_for_training {never}, "
                                    f"outside {training_source!r} {foreign}")
        checked.append(dict(fold=fold, seed=seed, member_sha256=m["member_sha256"], temperature=m["temperature"],
                            n_contact=len(contact), label_contact_subjects_sha256=m["label_contact_subjects_sha256"],
                            training_reads=len(read), calibration_reads=len(calib_read),
                            **(dict(checkpoint_choice=blob.get("checkpoint_choice"), checkpoint_iteration=blob.get("checkpoint_iteration")) if named else {})))
    record = dict(format=CHECK_FORMAT, passed=True, models_root=str(models_root), folds_path=split.path,
                  folds_sha256=split.sha256, pinned_folds_sha256=pinned_folds_sha256, allowlist_path=allowlist.path,
                  allowlist_sha256=allowlist.sha256, training_source=training_source,
                  members=sorted(checked, key=lambda c: (c["fold"], c["seed"])))
    return members, record


def build_entries(members, *, segmenter_id=SEGMENTER_ID, prefix=MODEL_PREFIX, seeds=DEFAULT_SEEDS, folds=FOLD_NAMES,
                  full_training_set=False):
    """Registry entries {model id: entry} for the folds whose members are complete; error otherwise.
    full_training_set: a fold is complete with at least one seed of `seeds`, and all label contacts must be equal
    (default: every seed, and disjoint contacts)."""
    entries, contacts = {}, {}
    for fold in folds:
        group = sorted((m for m in members if m["fold"] == fold), key=lambda m: m["seed"])
        found = [m["seed"] for m in group]
        if full_training_set:
            if not found or not set(found) <= set(seeds):
                raise InputValidationError(f"fold {fold}: calibrated seeds {found}; a registration needs at least one "
                                           f"member per fold, seeds among {sorted(seeds)}")
        elif found != sorted(seeds):
            raise InputValidationError(f"fold {fold}: calibrated seeds {found}, expected {sorted(seeds)}")
        contact = group[0]["label_contact_subjects"]
        if any(m["label_contact_subjects"] != contact for m in group):
            raise InputValidationError(f"fold {fold}: members disagree on label_contact_subjects")
        contacts[fold] = set(contact)
        entries[model_id_for(fold, prefix)] = dict(
            segmenter_id=segmenter_id, fold=FOLD_INDEX[fold], member_model_sha256=[m["member_sha256"] for m in group],
            label_contact_subjects=list(contact), label_contact_subjects_sha256=subject_list_sha256(contact))
    for a in folds:
        for b in folds:
            if full_training_set:
                if contacts[a] != contacts[b]:
                    raise InputValidationError(f"folds {a} and {b}: the full-training-set label contacts differ")
            elif a < b and contacts[a] & contacts[b]:
                raise InputValidationError(f"folds {a} and {b} share label-contact subjects "
                                           f"{sorted(contacts[a] & contacts[b])}; cross-fitting requires disjoint folds")
    return entries


def register(models_root, registry_path=REPO_REGISTRY, *, split, allowlist, training_source, pinned_folds_sha256=None,
             segmenter_id=SEGMENTER_ID, prefix=MODEL_PREFIX, seeds=DEFAULT_SEEDS, folds=FOLD_NAMES, log=print,
             full_training_set=False):
    """verify_members, then append the fold models to the registry (validated, atomic). Returns the
    SegmenterRegistry."""
    members, check = verify_members(models_root, split=split, allowlist=allowlist, training_source=training_source,
                                    pinned_folds_sha256=pinned_folds_sha256)
    entries = build_entries(members, segmenter_id=segmenter_id, prefix=prefix, seeds=seeds, folds=folds,
                            full_training_set=full_training_set)
    registry_path = Path(registry_path)
    if registry_path.exists():
        document = json.loads(registry_path.read_text(encoding="utf-8"), object_pairs_hook=strict_json_object)
        if document.get("schema") != REGISTRY_SCHEMA or not isinstance(document.get("models"), dict):
            raise InputValidationError(f"{registry_path}: not a {REGISTRY_SCHEMA} registry")
    else:
        document = dict(schema=REGISTRY_SCHEMA, comment="structure-layer fold models; append-only; written by "
                        "bayesreg.seg.unet.registry (docs/bayesreg/04_分割器.md)", models={})
    added = []
    for model_id, entry in entries.items():
        if model_id in document["models"]:
            if document["models"][model_id] != entry:
                raise InputValidationError(f"{registry_path}: {model_id} is already registered with different content; "
                                           "entries are append-only")
            continue
        document["models"][model_id] = entry
        added.append(model_id)
    tmp = Path(str(registry_path)+".tmp")
    tmp.write_text(json.dumps(document, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    try:
        load_segmenter_registry(tmp)                    # ProvenanceRejected if the result is malformed
    except Exception:
        tmp.unlink()
        raise
    os.replace(tmp, registry_path)
    registry = load_segmenter_registry(registry_path)
    ensemble = dict(format="bayesreg.unet35.ensemble.v1", segmenter_id=segmenter_id, registry_path=str(registry_path),
                    registry_sha256=registry.sha256, fold_models={m: dict(e) for m, e in entries.items()},
                    members=[{k: m[k] for k in ("fold", "seed", "member_sha256", "temperature", "path")}
                             for m in members if m["fold"] in folds], member_check=check)
    backup = write_ensemble(Path(models_root)/ENSEMBLE_FILE, ensemble)
    log(f"registry {registry_path}: added {added or 'nothing (already registered)'}; sha256 {registry.sha256}"
        + (f"; previous {ENSEMBLE_FILE} kept as {backup.name}" if backup else ""))
    return registry


def write_ensemble(path, document):
    """Write ensemble.json (the staging arm's gate: bayesreg.seg.structure_layer.check_staging_ensemble reads its
    registry_sha256 and member_check). An existing ensemble.json whose content differs (ignoring replaced_ensemble)
    is first copied, byte for byte, to ensemble_before_<UTC>.json beside it (review PS1_PS2 item a: registering into
    the repository registry replaces the staging-period file; never overwritten, a suffix _<k> on a clash), and the
    new file records it under replaced_ensemble. An identical file is left untouched. Returns the backup path or
    None."""
    path = Path(path)
    new = Path(str(path)+".new")
    write_json(new, document)
    if path.exists():
        old_bytes = path.read_bytes()
        try:
            old = json.loads(old_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            old = None
        strip = (lambda d: {k: v for k, v in d.items() if k != "replaced_ensemble"} if isinstance(d, dict) else d)
        if strip(old) == strip(json.loads(new.read_text(encoding="utf-8"))):
            new.unlink()
            return None
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        backup, k = path.with_name(f"ensemble_before_{stamp}.json"), 1
        while backup.exists():
            backup, k = path.with_name(f"ensemble_before_{stamp}_{k}.json"), k+1
        with open(backup, "xb") as stream:
            stream.write(old_bytes)
        write_json(new, dict(document, replaced_ensemble=dict(
            backup=backup.name, sha256=hashlib.sha256(old_bytes).hexdigest(), replaced_utc=stamp,
            registry_path_before=old.get("registry_path") if isinstance(old, dict) else None)))
    else:
        backup = None
    os.replace(new, path)
    return backup


__all__ = ["CHECK_FORMAT", "DEFAULT_SEEDS", "ENSEMBLE_FILE", "MODEL_PREFIX", "MODEL_PREFIX_V2", "MemberCheckFailed",
           "REPO_REGISTRY", "SEGMENTER_ID", "SEGMENTER_ID_V2", "build_entries", "find_member_files", "model_id_for",
           "read_members", "register", "verify_members", "write_ensemble"]
