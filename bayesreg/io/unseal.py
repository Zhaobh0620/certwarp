"""C23.3 and C23.4-1 (SG, contract revision 6, BZ 2026-11-07; D97-D100): BZ's unseal record, the one function that makes
an unseal object of it, and the two named bindings that hand it to the guard.

The record (format of runs/20261107-hi/入口说明.md §3.1, schema bayesreg.hi.unseal.v1, not changed) lives at one path
only, UNSEAL_REL inside the code tree of this bayesreg package (bayesreg.io.sealed.REPO, read at call time).
`load_unseal` checks, in this order, and refuses (UnsealRefused, a SubjectSealed: status subject_sealed) at the first
departure, before any file of a subject is read:
  (i)    the path resolves to the pinned one (a path elsewhere, also outside the tree, is refused, not a ValueError);
  (ii)   git (literal pathspecs): tracked and unchanged against HEAD before a byte of it is read (untracked, ignored,
         only staged, modified: refused);
  (iii)  strict JSON (no duplicate key), exactly UNSEAL_KEYS, schema UNSEAL_SCHEMA;
  (iv)   written_by "BZ" or "lead on BZ's behalf" (D98); bz_words one line without "|" and, when the lead wrote it on
         BZ's behalf, containing DELEGATION_WORDS ("解封记录你代我写", BZ 2026-11-07); time_local, purpose non-empty;
         utc a real UTC time between the draw of the hold-out (2026-10-30) and now;
  (v)    sealed_list in UNSEALABLE = ("holdout_B_v1",); H1-H4 never (defined here only; hi_core imports it);
  (vi)   subjects_file and its sha256 the pinned ones (bayesreg.io.sealed.SEALED_LISTS) and the file's sha256 now;
         the list through the guard's loader; subjects_sha256 = subject_list_sha256(sorted list); n_subjects an int
         equal to the list's size and the pinned count;
  (vii)  the allowlist configs/bayesreg/dev_subjects.json of the tree, tracked and unchanged; allowlist_sha256 its
         sha256; allowlist_source a source holding exactly the list, never_for_training, its subjects in no other
         source;
  (viii) segmenter_id a safe id, registry_sha256 64 lowercase hex (the anchor is the entry's own check);
  (ix)   docs/decision_log.md tracked and unchanged, one line of it holding both the record's sha256 and bz_words.
Only then an Unseal is made and entered in a private table by identity.  Unseal is a frozen dataclass with eq=False;
the guard takes an object only if its class is exactly Unseal and `_ISSUED.get(id(obj)) is obj` (a copy, a replace, a
pickle round trip, an object made by hand, a namedtuple or a dict are other objects), and recomputes the sha256 of the
record file each time it is used (a record changed or deleted after loading is refused).

Binding (D97 (a)): `bind_unseal(allowlist, unseal)` takes a DevAllowlist of load_dev_allowlist (no unseal on it) and an
issued Unseal whose allowlist_sha256 is the allowlist's, and returns a DevAllowlist with the same content and sha256
carrying the unseal object, entered by identity; `bind_files(files, bound_allowlist)` returns the OasisFiles (its own
class kept: dataclasses.replace inside this function) carrying that bound allowlist, entered by identity.  Objects made
otherwise (replace, copy, pickle, set by hand) are not entered and are refused by the guard and by image_path.  Across
processes: an object pickled into another process (spawn, multiprocessing, subprocess) is not in that process's tables
and is refused; a forked child inherits the tables (and still recomputes the record's sha256); a second import of this
module under another name has its own class and tables (refused).  The places that call load_unseal, bind_unseal and
bind_files are pinned by tests/test_bayesreg/sg/test_sg_census.py (D99).

This guards against mistakes, not against code changed on purpose (C23.0-3).  Who wrote the record is a matter of the
process (BZ's words in the decision log, the independent review; C23.0-4).
"""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import MappingProxyType

from jd_curl.three_d.nifti import file_sha256
from jd_curl.validation import InputValidationError

from . import sealed as _sealed
from .guard import DevAllowlist, SubjectSealed, is_hex64, is_safe_id, load_dev_allowlist, strict_json_object

UNSEAL_REL = "splits/bayesreg_v1/unseal_holdout_B_v1.json"
DECISION_LOG_REL = "docs/decision_log.md"
ALLOWLIST_REL = "configs/bayesreg/dev_subjects.json"
UNSEAL_SCHEMA = "bayesreg.hi.unseal.v1"
UNSEAL_KEYS = ("schema", "written_by", "bz_words", "time_local", "utc", "sealed_list", "subjects_file",
               "subjects_file_sha256", "subjects_sha256", "n_subjects", "allowlist_source", "allowlist_sha256",
               "segmenter_id", "registry_sha256", "purpose")
UNSEALABLE = ("holdout_B_v1",)                  # C23.3-2 (v): H1-H4 never; a contract revision and its review first
WRITERS = ("BZ", "lead on BZ's behalf")         # D98 (a)
DELEGATION_WORDS = "解封记录你代我写"            # BZ 2026-11-07 (decision_log 11-07, about 11:48 AM CST)
EARLIEST_UTC = datetime(2026, 10, 30, tzinfo=timezone.utc)      # the draw of the B-line hold-out (044426f)
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
UTC_FORM = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", re.ASCII)


class UnsealRefused(SubjectSealed):
    """The unseal record, an unseal object or a binding does not hold; `item` names the check.  Nothing of a subject
    is read.  Status subject_sealed."""

    def __init__(self, item, message):
        super().__init__(f"unseal ({item}): {message}")
        self.item = item


@dataclass(frozen=True, eq=False)
class Unseal:
    """An unseal object; made only by load_unseal (C23.3-5)."""
    path: str
    sha256: str
    sealed_list: str
    subjects: tuple
    allowlist_source: str
    allowlist_sha256: str
    written_by: str
    document: MappingProxyType
    git: MappingProxyType

    def record(self):
        from .pair import subject_list_sha256
        return dict(path=self.path, sha256=self.sha256, sealed_list=self.sealed_list, n_subjects=len(self.subjects),
                    subjects_sha256=subject_list_sha256(list(self.subjects)), written_by=self.written_by)


_ISSUED = {}            # id(obj) -> obj: the unseal objects load_unseal made (strong references: no id is reused)
_BOUND = {}             # id(obj) -> obj: the allowlists bind_unseal made
_FILES = {}             # id(obj) -> obj: the OasisFiles bind_files made


# ---- git: committed and unchanged (as experiments/hi/hi_core.tracked_unchanged) ---------------------------------
def _git(repo, *args):
    try:
        return subprocess.run(["git", "--literal-pathspecs", "-C", str(repo), *args], capture_output=True, text=True,
                              timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(args, 127, "", f"{type(exc).__name__}: {exc}")


def tracked_unchanged(path, root, *, item):
    """The file is inside `root`, tracked by git and without any change against HEAD; nothing of it is read.
    UnsealRefused(item) otherwise."""
    root = Path(root).resolve()
    target = Path(path).resolve()
    if not target.is_relative_to(root):
        raise UnsealRefused(item, f"{path} is not inside the code tree {root}")
    rel = target.relative_to(root).as_posix()
    tracked = _git(root, "ls-files", "--error-unmatch", "--", rel)
    changed = _git(root, "status", "--porcelain", "--ignored", "--untracked-files=all", "--", rel)
    if tracked.returncode != 0 or changed.returncode != 0 or changed.stdout.strip() or tracked.stdout.strip() != rel:
        raise UnsealRefused(item, f"{rel}: not a committed, unchanged file of the code tree (ls-files exit "
                                  f"{tracked.returncode} {tracked.stdout.strip()!r}; status {changed.stdout.strip()!r})")
    last = _git(root, "log", "-1", "--format=%H", "--", rel).stdout.strip()
    return dict(path=rel, tracked=True, unchanged=True, last_commit=last or None)


def utc_problem(value, key):
    """None if `value` is a real UTC time YYYY-MM-DDTHH:MM:SSZ between the draw of the hold-out and now (10 minutes of
    slack; as hi_core.utc_problem); the problem otherwise."""
    if not (isinstance(value, str) and UTC_FORM.fullmatch(value)):
        return f"{key} {value!r}: YYYY-MM-DDTHH:MM:SSZ required"
    try:
        when = datetime.strptime(value, UTC_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return f"{key} {value!r}: not a real time"
    if not EARLIEST_UTC <= when <= datetime.now(timezone.utc)+timedelta(minutes=10):
        return f"{key} {value!r}: before the draw of the hold-out or in the future"
    return None


def _nonempty(value):
    return isinstance(value, str) and value.strip() != ""


# ---- the one loader (C23.3-2) -------------------------------------------------------------------------------------
def load_unseal(path):
    """BZ's unseal record at the pinned path, checked (i)-(ix) in order (module docstring); the issued Unseal, or
    UnsealRefused at the first departure.  No file of a subject is read."""
    from .pair import subject_list_sha256
    root = Path(_sealed.REPO)
    pinned = root/UNSEAL_REL
    if path is None:
        raise UnsealRefused("path", "no unseal record: the hold-out is sealed until BZ's record is committed")
    path = Path(path)
    # (i)
    if path.resolve() != pinned.resolve():
        raise UnsealRefused("path", f"{path} is not the one path of the record, {UNSEAL_REL} in the code tree {root}")
    if not path.is_file():
        raise UnsealRefused("path", f"{path}: no such file")
    # (ii)
    record_git = tracked_unchanged(path, root, item="git")
    # (iii)
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_json_object)
    except (UnicodeDecodeError, ValueError) as exc:
        raise UnsealRefused("json", f"{path}: not UTF-8 JSON without duplicate keys ({exc})") from None
    if not isinstance(document, dict) or sorted(document) != sorted(UNSEAL_KEYS):
        got = sorted(document) if isinstance(document, dict) else type(document).__name__
        raise UnsealRefused("keys", f"{path}: exactly the keys {list(UNSEAL_KEYS)} are required, got {got}")
    if document["schema"] != UNSEAL_SCHEMA:
        raise UnsealRefused("schema", f"schema {document['schema']!r} is not {UNSEAL_SCHEMA!r}")
    # (iv)
    writer, words = document["written_by"], document["bz_words"]
    if writer not in WRITERS:
        raise UnsealRefused("written_by", f"written_by {writer!r} is not one of {list(WRITERS)} (D98)")
    if not _nonempty(words) or "\n" in words or "\r" in words or "|" in words:
        raise UnsealRefused("bz_words", "bz_words: one non-empty line without '|' is required (D98: it is matched on "
                                        "one line of the decision log)")
    if writer == WRITERS[1] and DELEGATION_WORDS not in words:
        raise UnsealRefused("bz_words", f"written_by {writer!r}: bz_words must contain BZ's words {DELEGATION_WORDS!r} "
                                        "word for word (D98)")
    for key in ("time_local", "purpose"):
        if not _nonempty(document[key]):
            raise UnsealRefused(key, f"{key}: a non-empty string is required")
    problem = utc_problem(document["utc"], "utc")
    if problem:
        raise UnsealRefused("utc", problem)
    # (v)
    name = document["sealed_list"]
    specs = {spec["name"]: spec for spec in _sealed.SEALED_LISTS}
    if name not in UNSEALABLE or name not in specs:
        raise UnsealRefused("sealed_list", f"sealed_list {name!r} is not one that may be unsealed ({list(UNSEALABLE)}; "
                                           "H1-H4 never)")
    spec = specs[name]
    # (vi)
    if document["subjects_file"] != spec["path"]:
        raise UnsealRefused("subjects_file", f"subjects_file {document['subjects_file']!r} is not the sealed list's "
                                             f"file {spec['path']!r}")
    if document["subjects_file_sha256"] != spec["sha256"]:
        raise UnsealRefused("subjects_file_sha256", f"subjects_file_sha256 {document['subjects_file_sha256']!r} is not "
                                                    f"the pinned {spec['sha256']}")
    listed = root/spec["path"]
    # review m2: every failure of (vi) is an UnsealRefused (a SubjectSealed, C23.3-2); follow-up review n5: the item names what failed --
    # the record's list file ('subjects_file_sha256') or the guard's sealed lists, any of the four ('sealed_lists')
    try:
        same = listed.is_file() and file_sha256(listed) == spec["sha256"]
    except OSError as exc:
        raise UnsealRefused("subjects_file_sha256", f"{spec['path']} cannot be read ({type(exc).__name__}: {exc})") from None
    if not same:
        raise UnsealRefused("subjects_file_sha256", f"{spec['path']}: missing or not the pinned sha256")
    try:
        lists = _sealed.guard_sealed_lists()
    except (_sealed.SealedListError, OSError) as exc:
        raise UnsealRefused("sealed_lists", f"the sealed lists cannot be read ({type(exc).__name__}: {exc})") from None
    subjects = sorted(lists.subjects[name])
    if document["subjects_sha256"] != subject_list_sha256(subjects):
        raise UnsealRefused("subjects_sha256", f"subjects_sha256 {document['subjects_sha256']!r} is not "
                                               f"subject_list_sha256 of the {len(subjects)} sorted subjects of {name}")
    count = document["n_subjects"]
    if type(count) is not int or count != len(subjects) or count != spec["count"]:
        raise UnsealRefused("n_subjects", f"n_subjects {count!r}: an int equal to the list ({len(subjects)}) and the "
                                          f"pinned count ({spec['count']}) is required")
    # (vii)
    allowlist_path = root/ALLOWLIST_REL
    allowlist_git = tracked_unchanged(allowlist_path, root, item="allowlist")
    try:                                            # review m2: a malformed or unreadable allowlist is an UnsealRefused too
        allowlist = load_dev_allowlist(allowlist_path)
    except (InputValidationError, OSError, ValueError, KeyError, TypeError) as exc:
        raise UnsealRefused("allowlist", f"{ALLOWLIST_REL} cannot be loaded ({type(exc).__name__}: {exc})") from None
    problems = []                                   # (vii) is one step: every departure of it is named together
    if document["allowlist_sha256"] != allowlist.sha256:
        problems.append(f"allowlist_sha256 {document['allowlist_sha256']!r} is not the sha256 of {ALLOWLIST_REL} "
                        f"({allowlist.sha256})")
    sources = {s["name"]: s for s in allowlist.sources}
    source = sources.get(document["allowlist_source"])
    if source is None:
        problems.append(f"the allowlist has no source {document['allowlist_source']!r}")
    else:
        if sorted(source["subjects"]) != subjects:
            problems.append(f"allowlist source {source['name']!r} does not hold exactly the {len(subjects)} subjects "
                            f"of {name}")
        if source["never_for_training"] is not True:
            problems.append(f"allowlist source {source['name']!r} must be never_for_training")
        shared = sorted({s for other in allowlist.sources if other["name"] != source["name"]
                         for s in other["subjects"]} & set(subjects))
        if shared:
            problems.append(f"{len(shared)} subjects of {name} are also in other allowlist sources: {shared[:5]}")
    if problems:
        raise UnsealRefused("allowlist", f"{problems}")
    # (viii)
    if not is_safe_id(document["segmenter_id"]):
        raise UnsealRefused("segmenter_id", f"segmenter_id {document['segmenter_id']!r} is not a safe identifier")
    if not is_hex64(document["registry_sha256"]):
        raise UnsealRefused("registry_sha256", "registry_sha256: 64 lowercase hex required")
    # (ix)
    sha = hashlib.sha256(raw).hexdigest()
    log = root/DECISION_LOG_REL
    if not log.is_file():
        raise UnsealRefused("decision_log", f"{DECISION_LOG_REL}: no such file")
    log_git = tracked_unchanged(log, root, item="decision_log")
    lines = log.read_text(encoding="utf-8").splitlines()
    if not any(sha in line for line in lines):
        raise UnsealRefused("decision_log", f"{DECISION_LOG_REL} does not name the record's sha256 {sha}")
    if not any(sha in line and words in line for line in lines):
        raise UnsealRefused("decision_log", f"no line of {DECISION_LOG_REL} holds both the record's sha256 and "
                                            "bz_words word for word (D98)")
    unseal = Unseal(path=str(pinned.resolve()), sha256=sha, sealed_list=name, subjects=tuple(subjects),
                    allowlist_source=source["name"], allowlist_sha256=allowlist.sha256, written_by=writer,
                    document=MappingProxyType(dict(document)),
                    git=MappingProxyType(dict(record=record_git, allowlist=allowlist_git, decision_log=log_git)))
    _ISSUED[id(unseal)] = unseal
    return unseal


# ---- checks of the objects (C23.3-5, C23.4-1) ---------------------------------------------------------------------
def unseal_problem(unseal):
    """None if `unseal` is an object load_unseal made in this process and its record file is unchanged; else why not."""
    if type(unseal) is not Unseal:
        return f"{type(unseal).__name__} is not the unseal object of bayesreg.io.unseal.load_unseal"
    if _ISSUED.get(id(unseal)) is not unseal:
        return "this unseal object was not made by load_unseal in this process (a copy, replace, pickle or by hand)"
    path = Path(unseal.path)
    try:
        digest = file_sha256(path) if path.is_file() else None
    except OSError:
        digest = None
    if digest != unseal.sha256:
        return f"the record {unseal.path} changed or is gone since it was loaded (sha256 {digest})"
    return None


def bound_problem(allowlist):
    """None if `allowlist` carries no unseal object; for one that does, None only if it is the object bind_unseal made,
    its unseal object holds and its sha256 is the unseal's allowlist_sha256 (C23.4-1); else why not."""
    unseal = getattr(allowlist, "unseal", None)
    if unseal is None:
        return None
    if _BOUND.get(id(allowlist)) is not allowlist:
        return "the allowlist carries an unseal object but is not the one bind_unseal made (replace, copy or by hand)"
    problem = unseal_problem(unseal)
    if problem:
        return problem
    if allowlist.sha256 != unseal.allowlist_sha256:
        return f"the allowlist sha256 {allowlist.sha256} is not the unseal record's {unseal.allowlist_sha256}"
    return None


def bind_unseal(allowlist, unseal):
    """The allowlist (of load_dev_allowlist, without an unseal object) with `unseal` on it: same path, sha256 and
    content, type DevAllowlist, entered by identity.  UnsealRefused otherwise."""
    if type(allowlist) is not DevAllowlist or allowlist.unseal is not None:
        raise UnsealRefused("bind", "bind_unseal takes a DevAllowlist of load_dev_allowlist without an unseal object")
    problem = unseal_problem(unseal)
    if problem:
        raise UnsealRefused("bind", problem)
    if allowlist.sha256 != unseal.allowlist_sha256:
        raise UnsealRefused("bind", f"the allowlist sha256 {allowlist.sha256} is not the unseal record's "
                                    f"{unseal.allowlist_sha256}")
    # review m3 (C23.4-1, literally): the allowlist is the one load_dev_allowlist reads from the pinned path -- its path is the
    # pinned file of this code tree, and its content is the file's, read again now (an object made with dataclasses.replace
    # or by hand, with the right sha256 but other subjects or sources, is refused)
    pinned = Path(_sealed.REPO)/ALLOWLIST_REL
    if Path(allowlist.path).resolve() != pinned.resolve():
        raise UnsealRefused("bind", f"the allowlist {allowlist.path} is not {ALLOWLIST_REL} of the code tree {_sealed.REPO}")
    try:
        fresh = load_dev_allowlist(pinned)
    except (InputValidationError, OSError, ValueError, KeyError, TypeError) as exc:
        raise UnsealRefused("bind", f"{ALLOWLIST_REL} cannot be loaded again ({type(exc).__name__}: {exc})") from None
    if (fresh.sha256, fresh.subjects, fresh.never_for_training, fresh.sources) != (allowlist.sha256, allowlist.subjects, allowlist.never_for_training, allowlist.sources):
        raise UnsealRefused("bind", f"the allowlist's content is not that of {ALLOWLIST_REL} read now (C23.4-1: an allowlist "
                                    "of load_dev_allowlist)")
    bound = replace(allowlist, unseal=unseal)
    _BOUND[id(bound)] = bound
    return bound


def bind_files(files, bound_allowlist):
    """The OasisFiles `files` (its class kept) with the allowlist of bind_unseal on it, entered by identity; its
    image_path then runs the whole guard (C23.4-2) for a sealed subject.  UnsealRefused otherwise."""
    from ..seg.unet.data import OasisFiles
    if not isinstance(files, OasisFiles) or files.allowlist is not None:
        raise UnsealRefused("bind_files", "bind_files takes an OasisFiles without a bound allowlist")
    if getattr(bound_allowlist, "unseal", None) is None:
        raise UnsealRefused("bind_files", "bind_files takes the allowlist bind_unseal returned, not an unbound one")
    problem = bound_problem(bound_allowlist)
    if problem:
        raise UnsealRefused("bind_files", problem)
    bound = replace(files, allowlist=bound_allowlist)
    _FILES[id(bound)] = bound
    return bound


def files_problem(files):
    """None if `files` carries no allowlist, or the one bind_files made with a bound allowlist that holds; else why."""
    allowlist = getattr(files, "allowlist", None)
    if allowlist is None:
        return None
    if _FILES.get(id(files)) is not files:
        return "the OasisFiles carries an allowlist but is not the one bind_files made (replace, copy or by hand)"
    if getattr(allowlist, "unseal", None) is None:
        return "the OasisFiles carries an allowlist without an unseal object"
    return bound_problem(allowlist)


__all__ = ["ALLOWLIST_REL", "DECISION_LOG_REL", "DELEGATION_WORDS", "UNSEALABLE", "UNSEAL_KEYS", "UNSEAL_REL",
           "UNSEAL_SCHEMA", "Unseal", "UnsealRefused", "WRITERS", "bind_files", "bind_unseal", "bound_problem",
           "files_problem", "load_unseal", "tracked_unchanged", "unseal_problem", "utc_problem"]
