"""C23.2 (SG, contract revision 6, BZ 2026-11-07, D95 (a')): the sealed lists and their one loader.

The sealed lists are read by one function only, `load_sealed_lists` (moved here unchanged from bayesreg.eval.truth, which
re-exports these very objects: bayesreg.eval.truth imports bayesreg.io.guard, so the guard could not import truth).
Each list is pinned by its sha256; a missing list, another sha256 or another count is SealedListError.

  holdout_B_v1  splits/bayesreg_v1/holdout_B_subjects.txt (the B-line hold-out, 100 subjects; unsealed by BZ only)
  H1            the OASIS hold-out pairs of paper 1 (50 pairs, 100 subjects)
  H4            the 40 large-deformation hold-out subjects of paper 1
  H3            IBSR18, all 18 subjects (ibsr18_pairs_20261018.tsv); any id containing "IBSR" is H3 whatever its form
  H2            LPBA40 S37-S40 (fixed in code; the same set as bayesreg.io.guard.NEVER_ALLOWLISTED["LPBA40"])

The guard (bayesreg.io.guard.check_subject, DevAllowlist.allows, bayesreg.seg.unet.data.OasisFiles) takes no list from
its caller: `guard_sealed_lists` reads them from this code tree (REPO, the tree the bayesreg package is in) with the
pinned values SEALED_LISTS, once per process, and caches only the ids and the record, keyed by (root, pinned values).
Training snapshots do not carry the lists (snapshot.py unchanged).  Where the guard reads them (C23.7-4 revision 6n, D95 (b),
lead 2026-11-11, BZ can change; `guard_root`):
  1. the four lists all in the code tree: read there, exactly as before (lists, record and cache byte for byte);
  2. none of the four in the code tree, and the code tree is a training snapshot that bayesreg.seg.unet.snapshot.verify_snapshot
     verifies: read from SNAPSHOT.json's source_repo (absolute, a directory, not the code tree itself) with the same pins; the
     record then carries one more item, the root and how it was reached; the cache key carries that root;
  3. anything else fails closed (SealedListError): only some lists in the code tree; no SNAPSHOT.json (a partial git archive);
     a SNAPSHOT.json that does not verify, cannot be read or has no source_repo.
Only this module computes the lists' sha256 (load_sealed_lists); make_snapshot and verify_snapshot never touch them.  There is no
public way to write the cache; the tests change REPO / SEALED_LISTS with monkeypatch and call `_reset_cache` (C23.2-3).  `sealed_cache_is_empty` is the
public read-only query (C23.2-3, revision 6g): an entry calls it before it installs a spy (C23.6-5).
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from jd_curl.three_d.nifti import file_sha256

REPO = Path(__file__).resolve().parents[2]           # the code tree of this bayesreg package (= bayesreg.eval.truth.REPO)
_PLAN = "archive/seq_vp_3d/results/verification/3d_stage_HOLDOUT_v1/plan"
SEALED_LISTS = (
    dict(name="holdout_B_v1", path="splits/bayesreg_v1/holdout_B_subjects.txt",
         sha256="8ed5c06eb782db3a04afdb885c3e8047fc299c632d00cf74006a70b14efe0c50", columns=None, count=100),
    dict(name="H1", path=f"{_PLAN}/oasis_heldout_pairs_draft.tsv",
         sha256="734e12c1a888ab60c2bb451d89a0db77d857a756c3f723d5adf18e1d701edae5", columns=("fixed", "moving"), count=100),
    dict(name="H4", path=f"{_PLAN}/h4_subjects_20261020.txt",
         sha256="c2f8980d109afc4af88de419b4bf6a18af4e4ee5a24bb6de163ddbf533e2217f", columns=None, count=40),
    dict(name="H3", path=f"{_PLAN}/ibsr18_pairs_20261018.tsv",
         sha256="5658b33bbb5fda00a78ee76a48c43932b36bd1a5bfbfc03e5ea05bd07eae8a3d", columns=("fixed", "moving"), count=18),
)
SEALED_LPBA40 = frozenset({"S37", "S38", "S39", "S40"})            # H2
PAPER1_LISTS = frozenset({"H1", "H2", "H3", "H4"})                # never unsealed by the B line (C23.3-2 (v), C23.4-2 (3))


class SealedListError(RuntimeError):
    """A sealed list is missing or differs from its pinned sha256: evaluation stops (fail closed)."""


@dataclass(frozen=True)
class SealedLists:
    subjects: Mapping           # name -> frozenset of canonical ids: a dict from load_sealed_lists, a mapping proxy in the guard's cache (SG follow-up review n6)
    record: Sequence            # the lists' records: a list of dicts from load_sealed_lists, a tuple of mapping proxies in the cache; whoever records it copies it

    def which(self, subject):
        if subject in SEALED_LPBA40:
            return "H2"
        if "IBSR" in str(subject).upper():
            return "H3"
        for name, ids in self.subjects.items():
            if subject in ids:
                return name
        return None


def load_sealed_lists(repo=REPO):
    subjects, record = {}, []
    for spec in SEALED_LISTS:
        path = Path(repo)/spec["path"]
        if not path.is_file():
            raise SealedListError(f"sealed list {spec['name']}: {spec['path']} missing")
        digest = file_sha256(path)
        if digest != spec["sha256"]:
            raise SealedListError(f"sealed list {spec['name']}: sha256 {digest} differs from the pinned value")
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        lines = [line for line in lines if line and not line.startswith("#")]
        if spec["columns"] is None:
            ids = set(lines)
        else:
            header = lines[0].split("\t")
            cols = [header.index(c) for c in spec["columns"]]
            ids = {line.split("\t")[c] for line in lines[1:] for c in cols}
        if len(ids) != spec["count"]:
            raise SealedListError(f"sealed list {spec['name']}: {len(ids)} subjects, expected {spec['count']}")
        subjects[spec["name"]] = frozenset(ids)
        record.append(dict(name=spec["name"], path=spec["path"], sha256=digest, subjects=len(ids)))
    record.append(dict(name="H2", path="(code) LPBA40 S37-S40", sha256=None, subjects=len(SEALED_LPBA40)))
    return SealedLists(subjects, record)


# ---- the guard's copy (C23.2-3) ---------------------------------------------------------------------------------
_CACHE = {}
_ROOTS = {}                 # (code tree, pinned values) -> the key of the copy read through SNAPSHOT.json's source_repo (rule 2 below)
VIA_SNAPSHOT = "SNAPSHOT.json source_repo of the verified training snapshot (C23.7-4 revision 6n, D95 (b))"


def _cache_key():
    return str(REPO), tuple(tuple(sorted(spec.items())) for spec in SEALED_LISTS)


def guard_root(repo=None):
    """(root, record item or None): where the guard reads the sealed lists (C23.7-4 revision 6n, D95 (b); module docstring).
    Rule 1: the four lists all in the code tree `repo` (default REPO) -> (repo, None).  Rule 2: none of them there and `repo` a
    training snapshot that verify_snapshot verifies -> (SNAPSHOT.json's source_repo, the record item).  Rule 3: anything else ->
    SealedListError.  Only file names are looked at here (no list is opened); verify_snapshot hashes the snapshot's own files."""
    repo = Path(REPO if repo is None else repo)
    present = [(repo/spec["path"]).is_file() for spec in SEALED_LISTS]
    if all(present):
        return repo, None
    first = next(spec for spec, there in zip(SEALED_LISTS, present) if not there)
    missing = f"sealed list {first['name']}: {first['path']} missing"
    if any(present):
        raise SealedListError(f"{missing}: only some of the sealed lists are in the code tree {repo} (fail closed; C23.7-4 revision 6n rule 3)")
    from ..seg.unet.snapshot import SNAPSHOT_FILE, verify_snapshot           # lazy: bayesreg.seg imports bayesreg.io
    if not (repo/SNAPSHOT_FILE).is_file():
        raise SealedListError(f"{missing}: none of the sealed lists is in the code tree {repo} and it has no {SNAPSHOT_FILE}, not a training "
                              f"snapshot (a partial export?): fail closed (C23.7-4 revision 6n rule 3)")
    try:
        snapshot = verify_snapshot(repo)
        source = snapshot["source_repo"]
    except Exception as exc:                                                   # does not verify, cannot be read, no source_repo
        raise SealedListError(f"{missing}: the code tree {repo} is not a verified training snapshot ({type(exc).__name__}: {exc}): "
                              f"fail closed (C23.7-4 revision 6n rule 3)") from None
    if not isinstance(source, str) or not source or not Path(source).is_absolute():
        raise SealedListError(f"{missing}: {SNAPSHOT_FILE} of {repo} has source_repo {source!r}, not an absolute path: fail closed (C23.7-4 revision 6n rule 2)")
    source = Path(source)
    if not source.is_dir():
        raise SealedListError(f"{missing}: {SNAPSHOT_FILE} of {repo} has source_repo {source}, not a directory: fail closed (C23.7-4 revision 6n rule 2)")
    if source.resolve() == repo.resolve():
        raise SealedListError(f"{missing}: {SNAPSHOT_FILE} of {repo} has source_repo {source}, the code tree itself: fail closed (C23.7-4 revision 6n rule 2)")
    item = dict(name="root", path=str(source), sha256=None, subjects=None, read_via=VIA_SNAPSHOT, code_tree=str(repo),
                snapshot_commit=snapshot.get("commit"), snapshot_json_sha256=snapshot.get("snapshot_json_sha256"))
    return source, item


def guard_sealed_lists():
    """The sealed lists the guard uses: read with SEALED_LISTS (read at call time) by load_sealed_lists from the root guard_root
    gives (REPO itself, or SNAPSHOT.json's source_repo of a verified training snapshot without the lists), cached per (root, pinned
    values).  SealedListError when they cannot be read or differ (nothing is cached then).  On REPO itself the key, the lists and
    the record are those of before revision 6n; through source_repo the key carries that root and the record one more item.
    The cached copy is read-only (SG code review m4): `subjects` a mapping proxy of frozensets, `record` a tuple of mapping
    proxies; whoever puts the record into a document copies it (bayesreg.io.guard._sealed_guard_record)."""
    key = _cache_key()
    lists = _CACHE.get(key)
    if lists is None and key in _ROOTS:
        lists = _CACHE.get(_ROOTS[key])
    if lists is None:
        root, item = guard_root(REPO)
        read = load_sealed_lists(root)
        record = read.record if item is None else read.record+[item]
        lists = SealedLists(MappingProxyType(dict(read.subjects)), tuple(MappingProxyType(dict(r)) for r in record))
        if item is None:
            _CACHE[key] = lists
        else:
            via = key+((item["read_via"], str(root)),)
            _CACHE[via] = lists
            _ROOTS[key] = via
    return lists


def sealed_cache_is_empty():
    """C23.2-3 (revision 6g; C23.6-5): True when no copy of the sealed lists is cached in this process -- the read-only query an
    entry makes before it installs a spy, so that the guard's first read of the lists happens under the spy.  It reads no file and
    changes nothing (no pin is consulted, the cache is not touched)."""
    return not _CACHE


def _reset_cache():
    """For the tests only (C23.2-3): forget every cached copy."""
    _CACHE.clear()
    _ROOTS.clear()


def lists_of(sealed, subject):
    """Every sealed list `subject` is on (H2 and H3 by their code rules, the others by their ids), sorted."""
    names = set()
    if subject in SEALED_LPBA40:
        names.add("H2")
    if "IBSR" in str(subject).upper():
        names.add("H3")
    names.update(name for name, ids in sealed.subjects.items() if subject in ids)
    return tuple(sorted(names))


__all__ = ["PAPER1_LISTS", "REPO", "SEALED_LISTS", "SEALED_LPBA40", "SealedListError", "SealedLists", "VIA_SNAPSHOT", "guard_root",
           "guard_sealed_lists", "lists_of", "load_sealed_lists", "sealed_cache_is_empty"]
