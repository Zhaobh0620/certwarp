"""Subject guard (review F2): canonical subject ids and the development allowlist.

Rule (development phase). A pair is loaded only if both of its subjects are on the development
allowlist. Every other subject -- the hold-out sets H1-H4 and the H4 pool, anything not listed, and
any id not written in its dataset's canonical form -- is refused with run status `subject_sealed`.
The check needs only the subject ids of the pair's manifest; it runs before any image is decoded.

Canonical subject ids (exact, case-sensitive ASCII; no fuzzy matching, no normalization: a
non-canonical spelling of an allowed subject is refused, not repaired):
    OASIS   OASIS_OAS1_dddd_MR1  the subject directory name, as the mainline manifests write
                                 fixed_subject / moving_subject (e.g. OASIS_OAS1_0438_MR1)
    LPBA40  S01 ... S40          as the mainline LPBA40 manifests write them (S37-S40 = H2 test
                                 split: never admissible on an allowlist, see NEVER_ALLOWLISTED)
    IBSR18  TODO -- the directory-name convention of data/three_d/stage_IBSR18_v1 has not been
            verified (it may not be read here). All 18 IBSR18 subjects are hold-out (H3), so until a
            canonical form is fixed every IBSR18 id is refused as non-canonical.

Allowlist file (configs/bayesreg/dev_subjects.json; format also in configs/bayesreg/README.md). It is
generated from the development stage lists (stage_L2R_val_v1, stage_T0_tune_v1, stage_E11_native_v1,
LPBA40 S01-S36; later the B-line training list once its draw is committed) and committed to the
repository; this module only reads and checks it:
    {"schema": "bayesreg.dev_subjects.v1",
     "subjects": {"<dataset>": ["<canonical id>", ...], ...},
     "sources": [{"name": "<id>", "dataset": "<dataset>", "path": "<repo-relative POSIX path>",
                  "sha256": "<64 hex of that file>", "never_for_training": <bool>,
                  "subjects": ["<canonical id>", ...]}, ...]}
`subjects[dataset]` must equal the union of the sources' subject lists for that dataset. A subject
listed by any source with never_for_training = true (stage_T0_tune_v1) may be registered but never
appear in a model's label_contact_subjects (checked in pair.validate_segmentation_provenance).

Sealed guard (SG, contract C23, revision 6, BZ 2026-11-07; D95-D105 by default). Both guards
(check_pair_subjects here and bayesreg.seg.unet.infer.check_subject_admitted) apply one rule per subject,
`check_subject`, in a fixed order: (1) the canonical id; (2) on the allowlist (its raw membership,
DevAllowlist.lists); (3) the sealed lists of bayesreg.io.sealed (read by the guard itself from this code tree,
pinned by sha256; unreadable -> SealedListError, every subject refused). A subject on no sealed list passes as
before; a subject on one passes only through an unseal object of bayesreg.io.unseal.load_unseal on an allowlist
of bayesreg.io.unseal.bind_unseal, for exactly that one list (never H1-H4), if it is one of the record's subjects
and in the record's allowlist source only. Steps (1) and (2) refuse with the messages they always had (C23.7).
An allowlist that carries an unseal object that does not hold is refused for any subject (D100).
DevAllowlist.allows answers "may be used for training and label contact" and is False for every sealed subject,
bound or not (D96).

`unseal_token` is deprecated, records only (D101): its sha256 is recorded and it admits nothing; unsealing is
C23's unseal object. Its run-time words "unsealing is not implemented (BZ)" are kept as they were.

LPBA40 second test set (contract C26.3, revision 11/11a/11b; LPBA40 prereg §8.5 item 2; D142-D144): the frozen
constant LPBA40_TEST_REJECT (S03-S36, 34 ids) is refused by check_subject between step (1) (and the DevAllowlist type
check) and step (2): step "sealed", item "lpba40 test set", whatever allowlist is given -- with one exception only: the
allowlist is the LPBA entries' own file, i.e. (i) its path resolves to bayesreg.io.sealed.REPO/LPBA40_ALLOWLIST_REL,
(ii) the pinned file, read again now, has the sha256 LPBA40_ALLOWLIST_SHA256 (= the object's) and the object's subjects,
never_for_training and sources, and (iii) that file holds the source lpba40_test_v1 with this subject.  Then steps (2)
and (3) run as before and an admission is marked admitted_by_lpba40 (check_pair_subjects records it under the key
lpba40_guard; a pair with neither side in the constant keeps its record byte for byte).  check_lpba40_names is the one
name-only refusal of a directory name naming these ids (LPBA40_IN_NAME): bayesreg.eval.truth (no allowlist, no
exception) and bayesreg.io.pair.load_pair (its allowlist, the exception) call it before any file is opened.  Only
tighter: SG, configs/bayesreg/dev_subjects.json and DevAllowlist.allows are unchanged (C26.3-7, -8).
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
import weakref

from jd_curl.validation import InputValidationError
from jd_curl.three_d.nifti import file_sha256

ALLOWLIST_SCHEMA = "bayesreg.dev_subjects.v1"
GUARD_VERSION = "dev-allowlist-v1"

CANONICAL_SUBJECT_PATTERNS = {
    "OASIS": re.compile(r"OASIS_OAS1_[0-9]{4}_MR1", re.ASCII),
    "LPBA40": re.compile(r"S(?:0[1-9]|[1-3][0-9]|40)", re.ASCII),
    # "IBSR18": TODO (see module docstring); absent on purpose.
}
NEVER_ALLOWLISTED = {"LPBA40": frozenset({"S37", "S38", "S39", "S40"})}   # H2 test split

# ---- the LPBA40 second test set (contract C26.3; D142-D144) -------------------------------------------------------
LPBA40_TEST_REJECT = frozenset(f"S{n:02d}" for n in range(3, 37))         # S03-S36 (splits/lpba40_test_v1/test_subjects.txt)
# sha256 of the constant serialized as the sorted ids, one per line, UTF-8, each line ending in "\n" (C26.3-1); the same
# bytes as splits/lpba40_test_v1/test_subjects.txt (SHA256SUMS line 3)
LPBA40_TEST_REJECT_SHA256 = "161f379290b6d3b0e52118570d9bc23e14197e34e77e5263c87f22a84fd5deb9"
LPBA40_REJECT_ITEM = "lpba40 test set"
LPBA40_TEST_SOURCE = "lpba40_test_v1"                                    # the source of the LPBA allowlist that holds them
LPBA40_ALLOWLIST_REL = "configs/bayesreg/lpba40_test_v1_allowlist.json"  # repository-relative, resolved against sealed.REPO
LPBA40_ALLOWLIST_SHA256 = "0d92bf39be8c5bfbaf47d652265d674e12dde1d12016455131747d340ce98e75"   # experiments/lpba/make_lpba40_allowlist.py --write (LP-A)
LPBA40_IN_NAME = re.compile(r"LPBA40_(S[0-9]{2})", re.ASCII)             # = bayesreg.eval.truth._LPBA40_IN_NAME (a test compares)
LPBA40_GUARD_VERSION = "lpba40-guard-v1"

# Identifiers of segmenters, models, classes and allowlist sources: ASCII, no whitespace or newline.
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", re.ASCII)
_HEX64 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_REL_PATH = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*", re.ASCII)


class SubjectSealed(Exception):
    """The pair contains a subject that is not on the development allowlist; run status `subject_sealed`."""
    status = "subject_sealed"


def is_hex64(value):
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def is_safe_id(value):
    return isinstance(value, str) and SAFE_ID.fullmatch(value) is not None


def strict_json_object(pairs):
    """object_pairs_hook that refuses duplicate keys (json would otherwise keep the last silently)."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputValidationError(f"JSON: duplicate key {key!r}")
        result[key] = value
    return result


def subject_dataset(subject_id):
    """Dataset whose canonical pattern `subject_id` matches exactly; SubjectSealed otherwise."""
    if isinstance(subject_id, str):
        for dataset, pattern in CANONICAL_SUBJECT_PATTERNS.items():
            if pattern.fullmatch(subject_id):
                return dataset
    raise SubjectSealed(f"subject id {subject_id!r} is not in any canonical form; refused without matching")


def canonical_subject_list(values, name, dataset=None):
    """A list (possibly empty) of distinct canonical ids, of `dataset` if given; InputValidationError otherwise."""
    if not isinstance(values, list):
        raise InputValidationError(f"{name}: expected a list of canonical subject ids")
    for value in values:
        try:
            found = subject_dataset(value)
        except SubjectSealed as exc:
            raise InputValidationError(f"{name}: {exc}") from None
        if dataset is not None and found != dataset:
            raise InputValidationError(f"{name}: {value!r} is not a canonical {dataset} id")
    if len(set(values)) != len(values):
        raise InputValidationError(f"{name}: duplicate subject ids")
    return list(values)


@dataclass(frozen=True)
class DevAllowlist:
    """Parsed allowlist: subjects per dataset, the never-for-training subset, sources, file sha256.

    `unseal`: None, except on the allowlist bayesreg.io.unseal.bind_unseal returns (C23.4-1, D97); it is not part
    of the record, the comparison or the repr."""
    path: str
    sha256: str
    subjects: Mapping
    never_for_training: frozenset
    sources: tuple
    unseal: object = field(default=None, compare=False, repr=False)

    def lists(self, subject_id):
        """The raw membership: a canonical id of the allowlist (step (2) of check_subject; what allows was)."""
        try:
            dataset = subject_dataset(subject_id)
        except SubjectSealed:
            return False
        return subject_id in self.subjects.get(dataset, ())

    def allows(self, subject_id):
        """May be used for training and label contact (C23.1-1 (a1), D96): on the allowlist and on no sealed list,
        bound or not.  SealedListError when the sealed lists cannot be read (for a subject on the allowlist)."""
        if not self.lists(subject_id):
            return False
        from .sealed import guard_sealed_lists, lists_of
        return not lists_of(guard_sealed_lists(), subject_id)

    def record(self):
        return dict(allowlist_path=self.path, allowlist_sha256=self.sha256, allowlist_schema=ALLOWLIST_SCHEMA,
                    allowlist_sources=[dict(name=s["name"], path=s["path"], sha256=s["sha256"]) for s in self.sources])


def _parse_allowlist(document, path, sha256):
    if not isinstance(document, dict) or set(document) - {"comment"} != {"schema", "subjects", "sources"}:
        raise InputValidationError("allowlist: expected exactly the keys schema, subjects, sources (and optional comment)")
    if not isinstance(document.get("comment", ""), str):
        raise InputValidationError("allowlist.comment: string required")
    if document["schema"] != ALLOWLIST_SCHEMA:
        raise InputValidationError(f"allowlist: schema must be {ALLOWLIST_SCHEMA!r}")
    listed, sources = document["subjects"], document["sources"]
    if not isinstance(listed, dict) or not listed:
        raise InputValidationError("allowlist.subjects: non-empty object dataset -> ids required")
    if not isinstance(sources, list) or not sources:
        raise InputValidationError("allowlist.sources: non-empty list required")
    subjects = {}
    for dataset, ids in listed.items():
        if dataset not in CANONICAL_SUBJECT_PATTERNS:
            raise InputValidationError(f"allowlist.subjects: dataset {dataset!r} has no canonical id form")
        subjects[dataset] = frozenset(canonical_subject_list(ids, f"allowlist.subjects.{dataset}", dataset))
        sealed = subjects[dataset] & NEVER_ALLOWLISTED.get(dataset, frozenset())
        if sealed:
            raise InputValidationError(f"allowlist.subjects.{dataset}: hold-out subjects {sorted(sealed)} can never be allowlisted")
    union, never, names, parsed = {}, set(), set(), []
    for index, source in enumerate(sources):
        where = f"allowlist.sources[{index}]"
        keys = {"name", "dataset", "path", "sha256", "never_for_training", "subjects"}
        if not isinstance(source, dict) or set(source) != keys:
            raise InputValidationError(f"{where}: expected exactly the keys {sorted(keys)}")
        if not is_safe_id(source["name"]) or source["name"] in names:
            raise InputValidationError(f"{where}.name: distinct identifier required")
        if not isinstance(source["dataset"], str) or source["dataset"] not in subjects:
            raise InputValidationError(f"{where}.dataset: {source['dataset']!r} not in allowlist.subjects")
        path_text = source["path"]
        if not isinstance(path_text, str) or not _REL_PATH.fullmatch(path_text) \
                or any(part in (".", "..") for part in path_text.split("/")):
            raise InputValidationError(f"{where}.path: repository-relative POSIX path required")
        if not is_hex64(source["sha256"]):
            raise InputValidationError(f"{where}.sha256: 64 lowercase hex required")
        if not isinstance(source["never_for_training"], bool):
            raise InputValidationError(f"{where}.never_for_training: JSON boolean required")
        ids = canonical_subject_list(source["subjects"], f"{where}.subjects", source["dataset"])
        if not ids:
            raise InputValidationError(f"{where}.subjects: non-empty list required")
        names.add(source["name"])
        union.setdefault(source["dataset"], set()).update(ids)
        if source["never_for_training"]:
            never.update(ids)
        parsed.append(MappingProxyType(dict(source, subjects=tuple(ids))))
    if {dataset: frozenset(ids) for dataset, ids in union.items()} != subjects:
        raise InputValidationError("allowlist: subjects per dataset must equal the union of the sources' subjects")
    return DevAllowlist(path=path, sha256=sha256, subjects=MappingProxyType(subjects),
                        never_for_training=frozenset(never), sources=tuple(parsed))


def load_dev_allowlist(path):
    """Read and check the allowlist JSON; the returned DevAllowlist carries the file's sha256."""
    raw = Path(path).read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"allowlist: not UTF-8 JSON ({exc})") from None
    return _parse_allowlist(document, str(path), hashlib.sha256(raw).hexdigest())


def verify_allowlist_sources(allowlist, root):
    """Recompute the sha256 of every source file (paths relative to `root`) against the allowlist.

    This reads the stage lists under data/; call it only when that is permitted. Returns
    {source name: sha256}; InputValidationError on any mismatch or missing file.
    """
    checked = {}
    for source in allowlist.sources:
        target = Path(root)/source["path"]
        if not target.is_file():
            raise InputValidationError(f"allowlist source {source['name']!r}: {source['path']} missing")
        if file_sha256(target) != source["sha256"]:
            raise InputValidationError(f"allowlist source {source['name']!r}: sha256 differs from the allowlist")
        checked[source["name"]] = source["sha256"]
    return checked


# ---- the rule per subject (C23.4-2) ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class SubjectCheck:
    """What check_subject found.  step None: admitted; "canonical", "allowlist" or "sealed": refused at that step
    (`item` names the check of step 3).  `message`: the words of steps 1 and 3; the guards write step 2's themselves."""
    subject: object
    step: object = None
    item: object = None
    message: object = None
    dataset: object = None
    sealed_lists: tuple = ()
    admitted_by_unseal: bool = False
    admitted_by_lpba40: bool = False            # C26.3-2: admitted through the one exception of LPBA40_TEST_REJECT

    @property
    def admitted(self):
        return self.step is None


def lpba40_reject_text(ids=None):
    """The serialization of LPBA40_TEST_REJECT whose sha256 is pinned (C26.3-1): the sorted ids, one per line, UTF-8, each
    line ending in a newline."""
    return "".join(f"{s}\n" for s in sorted(LPBA40_TEST_REJECT if ids is None else ids))


def lpba40_reject_sha256(ids=None):
    return hashlib.sha256(lpba40_reject_text(ids).encode("utf-8")).hexdigest()


def lpba40_allowlist_path():
    """The pinned LPBA allowlist of this code tree: bayesreg.io.sealed.REPO (read at call time) / LPBA40_ALLOWLIST_REL."""
    from . import sealed
    return Path(sealed.REPO)/LPBA40_ALLOWLIST_REL


def lpba40_exception_problem(subject, allowlist):
    """C26.3-3: None when the one exception of LPBA40_TEST_REJECT holds for `subject` on `allowlist`, else which condition
    does not: (i) the object's path resolves to the pinned file REPO/LPBA40_ALLOWLIST_REL; (ii) that file, read again now,
    has the sha256 LPBA40_ALLOWLIST_SHA256 (read at call time), equal to the object's, and the object's subjects,
    never_for_training and sources; (iii) it holds the source lpba40_test_v1 and `subject` in it.  Reads only that file of
    the code tree, never a file of the subject; an unreadable or unparsable file is a problem, not an exception."""
    if not isinstance(allowlist, DevAllowlist):
        return "(type) not a DevAllowlist of load_dev_allowlist"
    pinned = lpba40_allowlist_path()
    try:
        if Path(allowlist.path).resolve() != pinned.resolve():
            return f"(i) path: the allowlist {allowlist.path} is not the pinned {LPBA40_ALLOWLIST_REL} of the code tree"
    except (OSError, TypeError, ValueError) as exc:
        return f"(i) path: the allowlist path cannot be resolved ({type(exc).__name__}: {exc})"
    try:
        fresh = load_dev_allowlist(pinned)
    except (InputValidationError, OSError, ValueError, KeyError, TypeError) as exc:
        return f"(ii) sha256 and content: {LPBA40_ALLOWLIST_REL} cannot be read again ({type(exc).__name__}: {exc})"
    if fresh.sha256 != LPBA40_ALLOWLIST_SHA256:
        return f"(ii) sha256: {LPBA40_ALLOWLIST_REL} has sha256 {fresh.sha256}, not the pinned {LPBA40_ALLOWLIST_SHA256}"
    if allowlist.sha256 != fresh.sha256:
        return f"(ii) sha256: the allowlist object carries {allowlist.sha256}, not the file's {fresh.sha256}"
    if (fresh.subjects, fresh.never_for_training, fresh.sources) != (allowlist.subjects, allowlist.never_for_training,
                                                                     allowlist.sources):
        return f"(ii) content: the allowlist object's subjects or sources are not those of {LPBA40_ALLOWLIST_REL} read now"
    holding = [s for s in fresh.sources if s["name"] == LPBA40_TEST_SOURCE]
    if len(holding) != 1:
        return f"(iii) source: {LPBA40_ALLOWLIST_REL} has no source {LPBA40_TEST_SOURCE!r}"
    if subject not in holding[0]["subjects"]:
        return f"(iii) source: {subject!r} is not in the source {LPBA40_TEST_SOURCE!r}"
    return None


def check_lpba40_names(names, allowlist=None):
    """C26.3-6 (revision 11b): the name-only refusal shared by bayesreg.eval.truth.check_name, bayesreg.io.pair.load_pair and
    the pre-P_L admission dry run.  Every id LPBA40_IN_NAME finds in the given directory names that is in
    LPBA40_TEST_REJECT is refused (SubjectSealed) unless `allowlist` is given and the exception (lpba40_exception_problem)
    holds for it.  Opens no file of any subject (with an allowlist, the pinned allowlist file of the code tree is read
    again).  Returns the ids found, in order."""
    found = []
    for name in names:
        named = LPBA40_IN_NAME.findall(str(name))
        found += named
        for subject in named:
            if subject not in LPBA40_TEST_REJECT:
                continue
            if allowlist is None:
                raise SubjectSealed(f"the directory name {str(name)!r} names {subject}, in the LPBA40 second test set "
                                    f"(LPBA40_TEST_REJECT, C26.3-6): refused by name before any file is opened, with no exception here")
            problem = lpba40_exception_problem(subject, allowlist)
            if problem:
                raise SubjectSealed(f"the directory name {str(name)!r} names {subject}, in the LPBA40 second test set "
                                    f"(LPBA40_TEST_REJECT, C26.3-6): refused by name before any file is opened; the exception "
                                    f"does not hold: {problem}")
    return found


def check_subject(subject, allowlist):
    """C23.4-2, the one rule of both guards, in this order: (1) the canonical id; (2) on `allowlist` (raw membership);
    (3) the sealed lists (bayesreg.io.sealed, read and cached by the guard; SealedListError if they cannot be read).
    C26.3-2 (D142): between (1) (and the type check) and (2), a subject of LPBA40_TEST_REJECT is refused (step "sealed",
    item "lpba40 test set") unless the one exception holds (lpba40_exception_problem); then (2) and (3) as before, and an
    admission is marked admitted_by_lpba40.
    Returns a SubjectCheck; reads no file of the subject.  InputValidationError if `allowlist` is not a DevAllowlist."""
    try:
        dataset = subject_dataset(subject)
    except SubjectSealed as exc:
        return SubjectCheck(subject, step="canonical", message=str(exc))
    if not isinstance(allowlist, DevAllowlist):
        raise InputValidationError("allowlist: a DevAllowlist from load_dev_allowlist is required")
    lpba40 = False
    if subject in LPBA40_TEST_REJECT:
        problem = lpba40_exception_problem(subject, allowlist)
        if problem:
            return SubjectCheck(subject, step="sealed", item=LPBA40_REJECT_ITEM, dataset=dataset,
                                message=f"{subject!r} is in the LPBA40 second test set (LPBA40_TEST_REJECT): refused by the "
                                        f"development admission; the one exception does not hold: {problem} (C26.3)")
        lpba40 = True
    if not allowlist.lists(subject):
        return SubjectCheck(subject, step="allowlist", dataset=dataset)
    from .sealed import PAPER1_LISTS, guard_sealed_lists, lists_of
    from .unseal import bound_problem
    on = lists_of(guard_sealed_lists(), subject)
    unseal = allowlist.unseal

    def refused(item, words, suffix=" (bayesreg sealed guard, C23)"):
        return SubjectCheck(subject, step="sealed", item=item, dataset=dataset, sealed_lists=on,
                            message=f"{subject!r} {words}{suffix}")
    if unseal is not None:
        problem = bound_problem(allowlist)
        if problem:
            where = f"is on the sealed list {list(on)}" if on else "is on no sealed list"
            return refused("unseal object", f"{where}; the allowlist carries an unseal object that does not hold: "
                                            f"{problem} (C23.4-3, D100)")
    if not on:
        return SubjectCheck(subject, dataset=dataset, admitted_by_lpba40=lpba40)
    if unseal is None:
        return refused("no unseal record", f"is on the sealed list {list(on)}: refused by the bayesreg sealed guard "
                                           "without an unseal record", suffix=" (C23)")      # C23.4-2 (3), word for word
    paper1 = sorted(set(on) & PAPER1_LISTS)
    if paper1:
        return refused("H1-H4", f"is on the sealed list {paper1} of paper 1: never unsealed by the B line")
    if set(on) != {unseal.sealed_list}:
        return refused("sealed list", f"is on the sealed lists {list(on)}, not exactly the unsealed "
                                      f"{unseal.sealed_list!r}")
    if subject not in unseal.subjects:
        return refused("not in the record", f"is not one of the {len(unseal.subjects)} subjects of the unseal record "
                                            f"{unseal.path}")
    holding = sorted(s["name"] for s in allowlist.sources if subject in s["subjects"])
    if holding != [unseal.allowlist_source]:
        return refused("allowlist source", f"is in the allowlist sources {holding}, not only the unsealed source "
                                           f"{unseal.allowlist_source!r}")
    return SubjectCheck(subject, dataset=dataset, sealed_lists=on, admitted_by_unseal=True)


def refuse_sealed(subject, where):
    """Step (3) only, for the readers that take no allowlist (C23.6-1/2, D104): SubjectSealed if `subject` is on a
    sealed list (H2 and H3 by their code rules too), whatever unseal there is; SealedListError if the lists cannot be
    read.  Reads no file of the subject."""
    from .sealed import guard_sealed_lists, lists_of
    on = lists_of(guard_sealed_lists(), subject)
    if on:
        raise SubjectSealed(f"{where}: {subject!r} is on the sealed list {list(on)}: refused by the bayesreg sealed "
                            "guard (C23.6)")


# ---- pair -> bound allowlist, by identity (C23.4-8, D103) -----------------------------------------------------------
_PAIR_BOUND = {}        # id(pair) -> (weak reference to the pair, the bound allowlist load_pair admitted it through)


def register_pair(pair, allowlist):
    """For bayesreg.io.pair.load_pair only: `pair` had a sealed subject admitted through the bound `allowlist`."""
    _PAIR_BOUND[id(pair)] = (weakref.ref(pair), allowlist)


def pair_allowlist(pair):
    """The bound allowlist load_pair registered for this very pair object (`ref() is pair`), or None."""
    entry = _PAIR_BOUND.get(id(pair))
    if entry is None or entry[0]() is not pair:
        return None
    return entry[1]


def _sealed_guard_record(checks, allowlist):
    from .sealed import guard_sealed_lists
    unseal = allowlist.unseal

    def side(check):
        return dict(sealed_list=(list(check.sealed_lists) or [None])[0], admitted_by_unseal=check.admitted_by_unseal)
    return dict(version="sealed-guard-v1", sealed_lists=[dict(r) for r in guard_sealed_lists().record], fixed=side(checks["fixed"]),       # a copy (review m4)
                moving=side(checks["moving"]), unseal=None if unseal is None else unseal.record())


def check_pair_subjects(fixed_subject, moving_subject, allowlist, *, unseal_token=None):
    """Admit a pair only if both subject ids pass check_subject on `allowlist` (C23.4-2); SubjectSealed otherwise.

    `unseal_token`: deprecated, records only (D101): a token is recorded (its sha256) and admits nobody; unsealing is
    C23's unseal object (bayesreg.io.unseal) on the allowlist.  Returns the guard record for provenance: byte for byte
    the record of before when no subject is sealed and the allowlist carries no unseal object (D102 (b)); otherwise it
    has one more key, sealed_guard, and `unsealed` is True when a subject was admitted through the unseal object."""
    if not isinstance(allowlist, DevAllowlist):
        raise InputValidationError("allowlist: a DevAllowlist from load_dev_allowlist is required")
    if unseal_token is not None and (not isinstance(unseal_token, str) or not unseal_token):
        raise InputValidationError("unseal_token: non-empty string or None")
    token_sha = None if unseal_token is None else hashlib.sha256(unseal_token.encode("utf-8")).hexdigest()
    note = "; an unseal_token was recorded but unsealing is not implemented (BZ)" if token_sha else ""
    datasets, checks = {}, {}
    for side, subject in (("fixed", fixed_subject), ("moving", moving_subject)):
        verdict = check_subject(subject, allowlist)
        if verdict.step == "canonical":
            raise SubjectSealed(f"{side}: {verdict.message}{note}")
        if verdict.step == "allowlist":
            raise SubjectSealed(f"{side} subject {subject!r} is not on the development allowlist{note}")
        if verdict.step == "sealed":
            raise SubjectSealed(f"{side}: {verdict.message}{note}")
        datasets[side], checks[side] = verdict.dataset, verdict
    unsealed = any(check.admitted_by_unseal for check in checks.values())
    record = dict(guard=GUARD_VERSION, fixed_subject=fixed_subject, moving_subject=moving_subject, datasets=datasets,
                  never_for_training=dict(fixed=fixed_subject in allowlist.never_for_training,
                                          moving=moving_subject in allowlist.never_for_training),
                  unseal_token_presented=token_sha is not None, unseal_token_sha256=token_sha, unsealed=unsealed,
                  **allowlist.record())
    if unsealed or allowlist.unseal is not None:
        record["sealed_guard"] = _sealed_guard_record(checks, allowlist)
    if any(check.admitted_by_lpba40 for check in checks.values()):      # C26.3-4: only then one more key
        record["lpba40_guard"] = lpba40_guard_record(checks, allowlist)
    return record


def lpba40_guard_record(checks, allowlist):
    """C26.3-4: the record of an admission through the exception of LPBA40_TEST_REJECT -- both sides' admitted_by_lpba40,
    the allowlist file's repository-relative path and sha256, the constant's sha256."""
    return dict(version=LPBA40_GUARD_VERSION, fixed=dict(admitted_by_lpba40=checks["fixed"].admitted_by_lpba40),
                moving=dict(admitted_by_lpba40=checks["moving"].admitted_by_lpba40),
                allowlist_rel=LPBA40_ALLOWLIST_REL, allowlist_sha256=allowlist.sha256,
                reject_constant_sha256=lpba40_reject_sha256(), reject_item=LPBA40_REJECT_ITEM, source=LPBA40_TEST_SOURCE)
