"""Code snapshot of a real segmenter run and the code record of every output (review 2026-10-30, item 2).

A real training run executes from a `git archive` snapshot inside its results directory, never from
the live working tree (experiments/segnet/run_train.sh):

  make_snapshot(repo, dest)  refuses a repository whose `git status --porcelain` is not empty (tracked
                             changes or untracked files), extracts `git archive <HEAD> -- SNAPSHOT_PATHS`
                             into `dest` (which must not exist) and writes dest/SNAPSHOT.json: commit
                             SHA, source repository, UTC time, archived paths, sha256 of the tar stream
                             and of every extracted file.
  verify_snapshot(root)      recomputes every file hash; a changed, missing or extra file is refused.
  code_record(root)          the code that is running: the verified snapshot (kind "snapshot") or, outside
                             a snapshot, the git worktree with HEAD and a dirty flag (kind "git_worktree").
  require_clean_code(record) refuses a dirty or unknown tree; `require_snapshot` refuses anything but a
                             verified snapshot (real training and calibration, cli.py).

    python -m bayesreg.seg.unet.snapshot make --repo REPO --dest DIR [--profile v2_full]
    python -m bayesreg.seg.unet.snapshot verify --root DIR

--profile v2_full (the chain of unet35_v2_full, experiments/unv2/chain.sh) archives SNAPSHOT_PATHS_V2, --profile v3_large
(experiments/ul/run_ul.py) SNAPSHOT_PATHS_V3; the default is SNAPSHOT_PATHS.
"""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

SNAPSHOT_FILE = "SNAPSHOT.json"
SNAPSHOT_FORMAT = "bayesreg.unet35.snapshot.v1"
# Only what the segmenter chain imports or reads; splits/ is limited to the two training-set files
# (the hold-out lists of the same directory are not copied).
SNAPSHOT_PATHS = ("bayesreg", "jd_curl", "configs", "experiments/segnet", "splits/bayesreg_v1/folds.json",
                  "splits/bayesreg_v1/train_subjects.txt")
SNAPSHOT_PATHS_V2 = SNAPSHOT_PATHS+("experiments/unv2", "splits/bayesreg_v1/folds_full_v2.json")
# v3_large (experiments/ul/run_ul.py) also reads the reference of its reading, the single members of v2, from the repository
SNAPSHOT_PATHS_V3 = SNAPSHOT_PATHS_V2+("experiments/ul", "runs/20261103-unv2/v2_member_subsets.json")
SNAPSHOT_PROFILES = {"v1": SNAPSHOT_PATHS, "v2_full": SNAPSHOT_PATHS_V2, "v3_large": SNAPSHOT_PATHS_V3}
CODE_ROOT = Path(__file__).resolve().parents[3]


class CodeNotClean(Exception):
    """The running code is not a verified snapshot or a clean git worktree; status `code_not_clean`."""
    status = "code_not_clean"


def _git(root, *args, binary=False):
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodeNotClean(f"git {' '.join(args)} could not run in {root}: {exc}") from None
    if result.returncode != 0:
        raise CodeNotClean(f"git {' '.join(args)} failed in {root}: "
                           f"{result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def _sha256_file(path):
    path = Path(path)
    if path.is_symlink():
        return hashlib.sha256(("symlink:"+os.readlink(path)).encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_files(root):
    """Relative POSIX paths of every file (or symlink) under `root`, except SNAPSHOT.json and __pycache__."""
    root = Path(root)
    out = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts or rel.as_posix() == SNAPSHOT_FILE:
            continue
        if path.is_symlink() or path.is_file():
            out.append(rel.as_posix())
    return out


def make_snapshot(repo, dest, *, paths=SNAPSHOT_PATHS, commit="HEAD"):
    """git archive of `commit` (paths only) extracted to `dest` (must not exist) + SNAPSHOT.json."""
    repo, dest = Path(repo).resolve(), Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} exists; a snapshot is never overwritten")
    status = _git(repo, "status", "--porcelain")
    if status:
        raise CodeNotClean(f"{repo}: working tree is not clean (git status --porcelain):\n{status[:4000]}")
    sha = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    archive = _git(repo, "archive", "--format=tar", sha, "--", *paths, binary=True)
    partial = dest.with_name(dest.name+".partial")
    if partial.exists():
        raise FileExistsError(f"{partial} exists (an interrupted snapshot); remove it by hand")
    partial.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(partial, filter="data")
    files = {rel: _sha256_file(partial/rel) for rel in tree_files(partial)}
    record = dict(format=SNAPSHOT_FORMAT, commit=sha, source_repo=str(repo),
                  created_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  paths=list(paths), archive_sha256=hashlib.sha256(archive).hexdigest(), n_files=len(files),
                  files=files)
    (partial/SNAPSHOT_FILE).write_text(json.dumps(record, indent=1)+"\n", encoding="utf-8")
    partial.rename(dest)
    return record


def verify_snapshot(root):
    """Record of the verified snapshot at `root`; CodeNotClean on any changed, missing or extra file."""
    root = Path(root)
    raw = (root/SNAPSHOT_FILE).read_bytes()
    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict) or document.get("format") != SNAPSHOT_FORMAT \
            or not isinstance(document.get("files"), dict):
        raise CodeNotClean(f"{root/SNAPSHOT_FILE}: not a {SNAPSHOT_FORMAT} record")
    present = tree_files(root)
    missing = sorted(set(document["files"])-set(present))
    extra = sorted(set(present)-set(document["files"]))
    changed = sorted(rel for rel in present if rel in document["files"] and _sha256_file(root/rel) != document["files"][rel])
    if missing or extra or changed:
        raise CodeNotClean(f"snapshot {root} differs from its SNAPSHOT.json: missing {missing[:10]}, extra {extra[:10]}, "
                           f"changed {changed[:10]}")
    return dict(kind="snapshot", clean=True, commit=document["commit"], root=str(root),
                source_repo=document["source_repo"], created_utc=document["created_utc"],
                archive_sha256=document["archive_sha256"], n_files=len(present),
                snapshot_json_sha256=hashlib.sha256(raw).hexdigest())


def _imported_from():
    """Directories the bayesreg and jd_curl packages were imported from (both must lie in the code root)."""
    import bayesreg
    import jd_curl
    return dict(bayesreg=str(Path(bayesreg.__file__).resolve().parent), jd_curl=str(Path(jd_curl.__file__).resolve().parent))


def code_record(root=CODE_ROOT):
    """Which code is running: a verified snapshot, a git worktree (HEAD + dirty flag), or unknown; plus
    where bayesreg and jd_curl were imported from (`imports_inside_root` False if either lies elsewhere)."""
    root = Path(root)
    imported = _imported_from()
    inside = all(Path(p).is_relative_to(root.resolve()) for p in imported.values())
    if (root/SNAPSHOT_FILE).exists():
        return dict(verify_snapshot(root), imported_from=imported, imports_inside_root=inside)
    try:
        commit = _git(root, "rev-parse", "HEAD")
        status = _git(root, "status", "--porcelain")
    except CodeNotClean as exc:
        return dict(kind="unknown", clean=False, root=str(root), error=str(exc), imported_from=imported,
                    imports_inside_root=inside)
    lines = status.splitlines()
    return dict(kind="git_worktree", clean=not lines, commit=commit, root=str(root), n_dirty=len(lines),
                dirty_paths=lines[:50], imported_from=imported, imports_inside_root=inside)


def require_clean_code(record=None):
    record = code_record() if record is None else record
    if not record.get("imports_inside_root", False):
        raise CodeNotClean(f"bayesreg / jd_curl imported from {record.get('imported_from')}, not from the code root "
                           f"{record.get('root')}; refused")
    if record.get("kind") not in ("snapshot", "git_worktree") or not record.get("clean"):
        raise CodeNotClean(f"code at {record.get('root')} is not a verified snapshot or a clean git worktree "
                           f"(kind {record.get('kind')}, {record.get('n_dirty', '?')} dirty paths "
                           f"{record.get('dirty_paths', [])[:5]}); refused")
    return record


def require_snapshot(record=None):
    record = require_clean_code(record)
    if record["kind"] != "snapshot":
        raise CodeNotClean(f"real training and calibration run only from a git-archive snapshot "
                           f"(experiments/segnet/run_train.sh); this code runs from {record['root']} ({record['kind']})")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m bayesreg.seg.unet.snapshot")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("make")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--profile", choices=sorted(SNAPSHOT_PROFILES), default="v1")
    p = sub.add_parser("verify")
    p.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "make":
        record = make_snapshot(args.repo, args.dest, paths=SNAPSHOT_PROFILES[args.profile])
        print(json.dumps({k: v for k, v in record.items() if k != "files"}, indent=1))
    else:
        print(json.dumps(verify_snapshot(args.root), indent=1))
    return 0


__all__ = ["CODE_ROOT", "CodeNotClean", "SNAPSHOT_FILE", "SNAPSHOT_FORMAT", "SNAPSHOT_PATHS", "SNAPSHOT_PATHS_V2", "SNAPSHOT_PATHS_V3",
           "SNAPSHOT_PROFILES", "code_record", "make_snapshot", "require_clean_code", "require_snapshot", "tree_files",
           "verify_snapshot"]

if __name__ == "__main__":
    sys.exit(main())
