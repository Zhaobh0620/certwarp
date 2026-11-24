"""Truth labels for evaluation, behind the subject guard (project rules, hard constraints 6 and 7).

Evaluation may read the truth labels of development subjects only. Order for every pair, before any label is read:
  1. the names themselves (no file opened): a hold-out stage (stage_H1_v1, stage_H4_v1, stage_H3_IBSR18_v1,
     stage_IBSR18_v1), a name mentioning IBSR (H3, every subject sealed), or a canonical subject id in the pair name
     that is on a sealed list -> SubjectSealed;
  2. manifest.json through the loader's whitelisting decoder (bayesreg.io.pair._read_manifest: grid, pair_sha256,
     fixed_subject, moving_subject, source hashes only);
  3. both subjects against the sealed lists (below) and then against the development allowlist
     (bayesreg.io.guard.check_pair_subjects: canonical ids, on configs/bayesreg/dev_subjects.json; LPBA40 S37-S40
     never) -> SubjectSealed otherwise;
  4. sha256 of pair.npz equal to manifest.pair_sha256; only the label members of the scheme are read.
     load_truth does not trust the guard record it is handed (review EV item 4): it re-reads the manifest (its
     sha256, subjects and pair_sha256 must equal the record's), re-checks both subjects against the sealed lists
     and re-admits them on the allowlist the record names (its sha256 must match) before opening pair.npz.
Sealed lists (defence in depth: they also hold if an allowlist were wrong; each pinned by sha256, a missing or changed
list stops the evaluation):
  holdout_B_v1  splits/bayesreg_v1/holdout_B_subjects.txt (the B-line hold-out, 100 subjects; unsealed by BZ only)
  H1            the OASIS hold-out pairs of paper 1 (50 pairs, 100 subjects)
  H4            the 40 large-deformation hold-out subjects of paper 1
  H3            IBSR18, all 18 subjects (ibsr18_pairs_20261018.tsv, pinned; review EV item 6); any subject id
                containing "IBSR" is H3 whatever its form, and IBSR18 has no canonical id in the guard either
  H2            LPBA40 S37-S40 (fixed in code).
The lists and their one loader live in bayesreg.io.sealed (contract C23.2-2, SG); SEALED_LISTS, SEALED_LPBA40,
SealedListError, SealedLists and load_sealed_lists here are those very objects, re-exported.
unseal_token: deprecated, records only; unsealing is C23's unseal object (bayesreg.io.unseal).  A pair admitted
through an unseal object has unsealed = True and is refused by recheck_guard: the truth of a sealed subject is never
read here (C23.8-3; the hold-out evaluation is C23.12's own entry).
LPBA40 second test set (contract C26.3-6, D144): check_name refuses a pair whose name names an id of
bayesreg.io.guard.LPBA40_TEST_REJECT (S03-S36) through guard.check_lpba40_names, with no allowlist and so no exception;
recheck_guard (and so load_truth) applies the same name check to the pair directory first.  Revision 11c-1 (review I1):
the name check also takes the resolved directory's name (a link named otherwise), and after the manifest is decoded,
admit_pair and recheck_guard refuse a pair whose manifest names one of those subjects, whatever the allowlist
(refuse_lpba40_subjects; before check_pair_subjects, before pair.npz).  The truth of those subjects is read only by the
LPBA40 evaluation entry (C26.7), never here.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from jd_curl.three_d.nifti import file_sha256

from bayesreg.io.guard import (CANONICAL_SUBJECT_PATTERNS, LPBA40_TEST_REJECT, DevAllowlist, SubjectSealed, check_lpba40_names,
                               check_pair_subjects, load_dev_allowlist)
from bayesreg.io.pair import _read_manifest
from bayesreg.io.sealed import SEALED_LISTS, SEALED_LPBA40, SealedListError, SealedLists, load_sealed_lists

from .metrics import OASIS, lpba40_scheme

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO/"data"/"three_d"
DEFAULT_ALLOWLIST = REPO/"configs"/"bayesreg"/"dev_subjects.json"
SEALED_STAGES = frozenset({"stage_H1_v1", "stage_H4_v1", "stage_H3_IBSR18_v1", "stage_IBSR18_v1"})
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}", re.ASCII)
SAFE_NAME = _SAFE_NAME                       # also the form of a run id (cli --run-id, review EV item 5)
_OASIS_IN_NAME = re.compile(r"OASIS_OAS1_[0-9]{4}_MR1", re.ASCII)
_LPBA40_IN_NAME = re.compile(r"LPBA40_(S[0-9]{2})", re.ASCII)


def _token_record(unseal_token):
    if unseal_token is None:
        return None
    if not isinstance(unseal_token, str) or not unseal_token:
        raise ValueError("unseal_token: non-empty string or None")
    return hashlib.sha256(unseal_token.encode("utf-8")).hexdigest()


def refuse_lpba40_subjects(selected, where):
    """C26.3-6 revision 11c-1 (review I1): after the manifest's whitelisted decoding, a pair whose fixed or moving subject is in
    LPBA40_TEST_REJECT is refused, on any allowlist (no exception in truth); nothing else of the pair is opened."""
    named = [selected[f"{side}_subject"] for side in ("fixed", "moving") if selected[f"{side}_subject"] in LPBA40_TEST_REJECT]
    if named:
        raise SubjectSealed(f"{where}: the manifest names {named}, in the LPBA40 second test set (LPBA40_TEST_REJECT, C26.3-6): "
                            "eval.truth never reads their truth, on any allowlist")


def check_name(stage, pair, sealed, *, unseal_token=None):
    """Step 1: safe stage / pair names; no sealed subject named in the pair; no IBSR pair."""
    for label, name in (("stage", stage), ("pair", pair)):
        if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name) or ".." in name:
            raise ValueError(f"{label} {name!r}: a plain directory name is required")
    note = "; an unseal token was recorded but unsealing is not implemented (BZ)" if unseal_token else ""
    if stage in SEALED_STAGES:
        raise SubjectSealed(f"stage {stage!r} is a hold-out stage (H1, H3 or H4){note}")
    if "IBSR" in pair.upper() or "IBSR" in stage.upper():
        raise SubjectSealed(f"pair {pair!r}: IBSR18 subjects are all hold-out (H3){note}")
    check_lpba40_names([pair])          # C26.3-6 (D144): a pair named for the LPBA40 second test set, by name, no exception here
    named = _OASIS_IN_NAME.findall(pair)+_LPBA40_IN_NAME.findall(pair)
    for subject in named:
        which = sealed.which(subject)
        if which is not None:
            raise SubjectSealed(f"pair {pair!r} names {subject}, which is on the sealed list {which}{note}")
    return named


def admit_pair(data_root, stage, pair, allowlist, sealed, *, unseal_token=None):
    """Steps 1-3. Returns (pair_dir, grid, grid_metadata, guard record); SubjectSealed for any sealed subject."""
    token_sha = _token_record(unseal_token)
    named = check_name(stage, pair, sealed, unseal_token=unseal_token)
    if not isinstance(allowlist, DevAllowlist):
        allowlist = load_dev_allowlist(allowlist)
    pair_dir = Path(data_root)/stage/pair
    check_lpba40_names([pair_dir.resolve().name])       # revision 11c-1 (review I1): the resolved directory's name too (a link named otherwise)
    manifest_path = pair_dir/"manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: missing")
    grid, metadata, selected = _read_manifest(manifest_path)
    refuse_lpba40_subjects(selected, f"pair {pair!r}")     # revision 11c-1 (review I1): by the manifest's subjects, whatever the allowlist
    note = "; an unseal token was recorded but unsealing is not implemented (BZ)" if token_sha else ""
    for side in ("fixed", "moving"):
        subject = selected[f"{side}_subject"]
        which = sealed.which(subject)
        if which is not None:
            raise SubjectSealed(f"{side} subject {subject!r} is on the sealed list {which}{note}")
    if not set(named) <= {selected["fixed_subject"], selected["moving_subject"]}:
        raise ValueError(f"pair {pair!r}: subjects in the name differ from the manifest's")
    guard = check_pair_subjects(selected["fixed_subject"], selected["moving_subject"], allowlist,
                                unseal_token=unseal_token)
    sources = {s["name"]: s["subjects"] for s in allowlist.sources}
    guard = dict(guard, sealed_lists=sealed.record, pair_name_subjects=named,
                 allowlist_sources_of=dict(
                     fixed=sorted(n for n, ids in sources.items() if selected["fixed_subject"] in ids),
                     moving=sorted(n for n, ids in sources.items() if selected["moving_subject"] in ids)),
                 manifest_path=str(manifest_path), manifest_sha256=file_sha256(manifest_path),
                 pair_sha256_manifest=selected["pair_sha256"])
    return pair_dir, grid, metadata, guard


def scheme_for(guard, pair_dir):
    """The label scheme of the pair's dataset (OASIS or LPBA40; LPBA40 ids from the manifest's label_sets)."""
    datasets = set(guard["datasets"].values())
    if datasets == {"OASIS"}:
        return OASIS
    if datasets == {"LPBA40"}:
        manifest = json.loads((Path(pair_dir)/"manifest.json").read_text(encoding="utf-8"))
        return lpba40_scheme(manifest["label_sets"]["lpba56"])
    raise ValueError(f"no label scheme for datasets {sorted(datasets)}")


def recheck_guard(pair_dir, guard, *, repo=REPO):
    """Re-verify a guard record against the files (review EV item 4): manifest sha256, subjects and pair_sha256 equal
    to the record's, neither subject sealed, both admitted again on the allowlist the record names (same sha256).
    Raises ValueError (record does not describe this pair) or SubjectSealed."""
    check_lpba40_names([Path(pair_dir).name, Path(pair_dir).resolve().name])       # C26.3-6, T2 (revision 11c-1: the resolved name too): by name first
    if not isinstance(guard, dict) or guard.get("unsealed") is not False:
        raise ValueError("load_truth: the guard record of admit_pair is required")
    for key in ("manifest_sha256", "fixed_subject", "moving_subject", "pair_sha256_manifest", "allowlist_path",
                "allowlist_sha256"):
        if not isinstance(guard.get(key), str) or not guard[key]:
            raise ValueError(f"load_truth: guard record without {key}")
    manifest_path = Path(pair_dir)/"manifest.json"
    if not manifest_path.is_file() or file_sha256(manifest_path) != guard["manifest_sha256"]:
        raise ValueError(f"{manifest_path}: missing or differs from the guard record's manifest_sha256")
    _, _, selected = _read_manifest(manifest_path)
    refuse_lpba40_subjects(selected, f"{manifest_path}")      # revision 11c-1 (review I1): by the manifest's subjects, whatever the allowlist
    for key in ("fixed_subject", "moving_subject"):
        if selected[key] != guard[key]:
            raise ValueError(f"{manifest_path}: {key} differs from the guard record")
    if selected["pair_sha256"] != guard["pair_sha256_manifest"]:
        raise ValueError(f"{manifest_path}: pair_sha256 differs from the guard record")
    sealed = load_sealed_lists(repo)
    for side in ("fixed", "moving"):
        which = sealed.which(selected[f"{side}_subject"])
        if which is not None:
            raise SubjectSealed(f"{side} subject {selected[side+'_subject']!r} is on the sealed list {which}")
    allowlist = load_dev_allowlist(guard["allowlist_path"])
    if allowlist.sha256 != guard["allowlist_sha256"]:
        raise ValueError("load_truth: the allowlist differs from the one the guard record names")
    check_pair_subjects(selected["fixed_subject"], selected["moving_subject"], allowlist)


def load_truth(pair_dir, scheme, grid, guard, *, repo=REPO):
    """Step 4 (only after admit_pair; the guard record is re-verified first, recheck_guard): the scheme's label
    members of pair.npz and the brain node mask."""
    recheck_guard(pair_dir, guard, repo=repo)
    path = Path(pair_dir)/"pair.npz"
    digest = file_sha256(path)
    if digest != guard["pair_sha256_manifest"]:
        raise ValueError(f"{path}: sha256 differs from manifest.pair_sha256")
    label_stems = {scheme.primary} | {stem for _, stem, _ in scheme.macro_sets}
    keys = []
    for stem in scheme.stems():
        keys.append("fixed_"+stem)
        if stem in label_stems:
            keys.append("moving_"+stem)
    truth = {}
    with np.load(path, allow_pickle=False) as npz:
        for key in keys:
            if key not in npz:
                raise ValueError(f"{path.name}: label member {key!r} missing")
            value = np.array(npz[key])
            if value.shape != grid.shape:
                raise ValueError(f"{path.name}:{key}: shape {value.shape} differs from the grid {grid.shape}")
            if not np.all(np.isfinite(value)) or np.any(value < 0) or not np.array_equal(value, np.rint(value)):
                raise ValueError(f"{path.name}:{key}: labels must be finite non-negative integers")
            truth[key] = value
    truth["brain"] = truth["fixed_"+scheme.brain_stem] > 0
    record = dict(pair_npz=str(path), pair_sha256=digest, keys_read=keys, scheme=scheme.name,
                  primary=scheme.primary, primary_labels=list(scheme.primary_labels),
                  macro_sets={m: dict(stem=s, labels=list(ids)) for m, s, ids in scheme.macro_sets},
                  brain_mask=f"fixed_{scheme.brain_stem} > 0 (nodes); a cell is brain if its lower-corner node is")
    return truth, record


__all__ = ["CANONICAL_SUBJECT_PATTERNS", "DEFAULT_ALLOWLIST", "DEFAULT_DATA_ROOT", "SAFE_NAME", "SEALED_LISTS",
           "SEALED_STAGES", "SealedListError", "SealedLists", "SubjectSealed", "admit_pair", "check_name", "refuse_lpba40_subjects",
           "load_sealed_lists", "load_truth", "recheck_guard", "scheme_for"]
