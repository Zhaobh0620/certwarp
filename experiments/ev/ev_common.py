"""Shared helpers of the two pre-unsealing tools of work package EV (runs/20261107-ev): the truth-reader gate (truth_gate.py, C23.12-10) and the
evaluation-entry control (entry_control.py, preregistration 5.1 item 8).  Both read development data only; neither touches a hold-out subject.

  same          a recursive exact comparison of two JSON-like values: floats bit for bit (NaN equals NaN; JSON null equals NaN only when asked), the
                differing paths listed (the values themselves are not reported: the development numbers of these tools are no result, C23.12-10)
  normalized    a record as JSON gives it back (numpy types, tuples and mapping proxies made plain; NaN kept)
  sha256_list   a sha256sum file read strictly ("<64 hex>  <absolute path>" per line)
  verify_list   every file of such a list at its sha256 (before anything of it is used)
  code_state    HEAD and whether the entry's CLEAN_DIRS (the five code directories and splits/) are clean (the tools run from a clean tree; B4 review m1)
  new_out       an output directory under RESULTS, outside the code tree, not there yet
"""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
_LINE = re.compile(r"([0-9a-f]{64})  (/.+)", re.ASCII)


class ToolRefused(Exception):
    def __init__(self, stage, message):
        super().__init__(f"{stage}: {message}")
        self.stage = stage


def normalized(value):
    from bayesreg.eval.holdout import json_default
    return json.loads(json.dumps(value, default=json_default, allow_nan=True))


def _float_same(a, b, null_is_nan):
    if a is None or b is None:
        if null_is_nan:
            a_nan = a is None or (isinstance(a, float) and math.isnan(a))
            b_nan = b is None or (isinstance(b, float) and math.isnan(b))
            return a_nan and b_nan
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    return fa == fb and math.copysign(1., fa) == math.copysign(1., fb)


def same(a, b, path="", *, null_is_nan=False, skip=()):
    """[differing paths] of two JSON-like values (empty: the same); keys in `skip` (at any depth) are left out (e.g. timing)."""
    numbers = (int, float)
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key in skip:
                continue
            if key not in a or key not in b:
                out.append(f"{path}/{key} (only one side)")
                continue
            out += same(a[key], b[key], f"{path}/{key}", null_is_nan=null_is_nan, skip=skip)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} (length {len(a)} / {len(b)})"]
        out = []
        for k, (x, y) in enumerate(zip(a, b)):
            out += same(x, y, f"{path}[{k}]", null_is_nan=null_is_nan, skip=skip)
        return out
    scalar = (numbers+(type(None),))
    if isinstance(a, scalar) and isinstance(b, scalar) and not isinstance(a, str) and not isinstance(b, str):
        return [] if _float_same(a, b, null_is_nan) else [path]
    return [] if a == b else [path]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_list(path):
    """{absolute path: sha256} of a sha256sum file; ToolRefused on any other line or a duplicate."""
    out = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        match = _LINE.fullmatch(line)
        if not match or match.group(2) in out:
            raise ToolRefused("list", f"{path}:{number}: '<64 hex>  <absolute path>' without duplicates required")
        out[match.group(2)] = match.group(1)
    return out


def verify_list(listed):
    bad = [p for p, sha in listed.items() if not Path(p).is_file() or file_sha256(p) != sha]
    if bad:
        raise ToolRefused("list", f"{len(bad)} files are missing or not at their pinned sha256: {bad[:5]}")
    return dict(files=len(listed), verified=True)


def pinned_file(rel, sha256, stage):
    """A file of the code tree at its pinned sha256 (None pinned: refused)."""
    if sha256 is None:
        raise ToolRefused(stage, f"{rel}: its sha256 is not pinned yet (the lead pins it after the review of the plan); nothing is read")
    path = ROOT/rel
    if not path.is_file() or file_sha256(path) != sha256:
        raise ToolRefused(stage, f"{rel}: missing or not the pinned sha256 {sha256}")
    return dict(path=rel, sha256=sha256)


def code_state(root=ROOT):
    def git(*args):
        return subprocess.run(["git", "--literal-pathspecs", "-C", str(root), *args], capture_output=True, text=True, timeout=60)
    from bayesreg.eval.holdout import CLEAN_DIRS            # B4 review, 3rd plan m1 (A5b): the entry's directories -- the five code directories and splits (folds.json, review m7)
    status = git("status", "--porcelain", "--untracked-files=all", "--", *CLEAN_DIRS)
    return dict(head=git("rev-parse", "HEAD").stdout.strip() or None, clean=status.returncode == 0 and not status.stdout.strip(), status=status.stdout, dirs=list(CLEAN_DIRS))


def new_out(path, results):
    target = Path(path).resolve()
    if not target.is_relative_to(Path(results).resolve()) or target.is_relative_to(ROOT.resolve()):
        raise ToolRefused("output", f"{path}: under {results} and outside the code tree {ROOT} required")
    if target.exists():
        raise ToolRefused("output", f"{path} exists: nothing is overwritten")
    return target
