"""Byte comparison of the run_pair record files under a measured mask S (work package SCHI, 2026-10-31; contract C13
SCH gate 1, second item, with the lead's round-2 minor 1).

Gate 1 (C13 gate of C13-4a / 5a, item 1): without the two new switches (--max-accepted-per-level, --level-time-shares;
--max-accepted N alone included) the seven record files config.json, provenance.json, hyper.json, qc.json, status.json,
timing.json and log.jsonl are byte-identical to those of the code before SCH, once the leaves of a mask S are replaced
by a fixed placeholder.  S is MEASURED on the code before the implementation (commit 11409eb): the default arm run twice,
under two different temporary roots (code extraction and inputs), the leaves whose values differ; each differing leaf
must fall in one of the five contract classes, otherwise it is NOT put in S and the gate fails (reported to the lead):

  timing        a key on the leaf's path is in _swli_digest.TIMING_KEYS or contains "seconds"            (计时键)
  run_root      the value is a string that starts with the run's temporary root, in both runs   (以运行根目录为前缀的路径)
  code          provenance.json /code/...                                                              (provenance.code)
  timing_value  a number in timing.json, except the counts certificate_calls, witness_records and the level number
                                                                                      (timing.json 的数值叶子，层号与两个计数除外)
  npz_sha256    the sha256 of a .npz container (zip members carry their write time), named by its *_path sibling
                                                                                        (带写入时间的容器级 npz 的 sha256)

Fixed items added to S whatever the measurement (contract): the path values --out, --data-root, --tissue-root,
--allowlist (config.json args) and the file paths of the pair provenance (provenance.json /pair/..._path); every field
of provenance.json /code; every number of timing.json except certificate_calls and witness_records.  The level number
solve_levels/levels[]/level is NOT masked here (stricter than the contract's fixed item; review SCH round 2, section 3:
"被屏蔽无害，但没有必要").

Paths: "/key/key" for dict keys, "[]" for any list item (index-free), so that a pattern masks the same field in every
round, level or log row; log.jsonl is the list of its rows ("[]/key").  Concrete measured paths carry the indices.
Comparison (contract): parse keeping key order, replace every leaf of S by PLACEHOLDER, re-serialize with the
arguments of bayesreg.solve.export.write_json (dumps(document, indent=1) + "\\n"; log.jsonl: dumps(row) + "\\n" per row),
sha256 of the bytes.
"""
import hashlib
import json
from pathlib import Path
import re

from . import _swli_digest as SD

FILES = ("config.json", "provenance.json", "hyper.json", "qc.json", "status.json", "timing.json", "log.jsonl")
JSON_FILES = FILES[:-1]
PLACEHOLDER = "<SCH-masked>"
CLASSES = ("timing", "run_root", "code", "timing_value", "npz_sha256")
TIMING_COUNTS = ("certificate_calls", "witness_records")
LEVEL_NUMBER = "/solve_levels/levels[]/level"
FIXED_ARGS = ("out", "data_root", "tissue_root", "allowlist")
# solve quantities that S must never contain (the reviewer's check of gate 1, automated): a key on the path
SOLVE_KEYS = ("energy", "F_B", "GF_B", "gradient", "sigma", "G0", "T_B", "state_hash", "min_jd_lower_bound", "status",
              "reason", "accepted", "commits", "ared", "certified", "tau")
_INDEX = re.compile(r"\[\d+\]")


def read_texts(run):
    """{file: text} of the seven record files of a finished run directory."""
    run = Path(run)
    return {name: (run/name).read_text(encoding="utf-8") for name in FILES}


def parse(name, text):
    if name == "log.jsonl":
        return [json.loads(line) for line in text.splitlines()]
    return json.loads(text)


def serialize(name, document):
    """The bytes write_json / JsonlLog would write for `document` (bayesreg.solve.export.dumps, same arguments)."""
    from bayesreg.solve.export import dumps
    if name == "log.jsonl":
        return "".join(dumps(row)+"\n" for row in document).encode("utf-8")
    return (dumps(document, indent=1)+"\n").encode("utf-8")


def pattern(path):
    """The index-free pattern of a concrete path."""
    return _INDEX.sub("[]", path)


def _keys(path):
    return [part for part in re.split(r"/|\[\]|\[\d+\]", path) if part]


def scalar_leaves(document, path=""):
    """[(concrete path, value)] of every scalar leaf (dict keys in order, list items by index)."""
    out = []
    if isinstance(document, dict):
        for key, value in document.items():
            out += scalar_leaves(value, f"{path}/{key}")
    elif isinstance(document, list):
        for k, value in enumerate(document):
            out += scalar_leaves(value, f"{path}[{k}]")
    else:
        out.append((path, document))
    return out


def _same(a, b):
    return type(a) is type(b) and json.dumps(a) == json.dumps(b)


def differences(a, b, path=""):
    """[(concrete path, value a, value b, kind)] where two documents differ; kind "value" for a scalar leaf, "structure"
    for different key lists (order included) or list lengths or a container against a scalar."""
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return [(path, list(a), list(b), "structure")]
        out = []
        for key in a:
            out += differences(a[key], b[key], f"{path}/{key}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [(path, len(a), len(b), "structure")]
        out = []
        for k, (x, y) in enumerate(zip(a, b)):
            out += differences(x, y, f"{path}[{k}]")
        return out
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return [(path, type(a).__name__, type(b).__name__, "structure")]
    return [] if _same(a, b) else [(path, a, b, "value")]


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sibling(document, path, key):
    """The value of `key` next to the leaf at the concrete `path` (None when absent)."""
    node = document
    parts = re.findall(r"/([^/\[]+)|\[(\d+)\]", path)[:-1]
    for name, index in parts:
        node = node[int(index)] if index else node.get(name) if isinstance(node, dict) else None
        if node is None:
            return None
    return node.get(key) if isinstance(node, dict) else None


def classify(name, path, a, b, *, roots, documents=None):
    """The contract class of a measured differing leaf (None: none of the five, the gate fails).

    roots: (root a, root b), the temporary roots of the two runs; documents: (document a, document b) of `name`."""
    keys = _keys(path)
    leaf = keys[-1] if keys else ""
    if any(k in SD.TIMING_KEYS or "seconds" in k for k in keys):
        return "timing"
    if isinstance(a, str) and isinstance(b, str) and a.startswith(str(roots[0])) and b.startswith(str(roots[1])):
        return "run_root"
    if name == "provenance.json" and path.startswith("/code/"):
        return "code"
    if (name == "timing.json" and _number(a) and _number(b) and leaf not in TIMING_COUNTS
            and pattern(path) != LEVEL_NUMBER):
        return "timing_value"
    if leaf.endswith("sha256") and isinstance(a, str) and isinstance(b, str) and documents is not None:
        stem = leaf[:-len("sha256")]
        containers = [_sibling(doc, path, f"{stem}path") for doc in documents]
        if all(isinstance(c, str) and c.endswith(".npz") for c in containers):
            return "npz_sha256"
    return None


def fixed_patterns(documents):
    """{file: set of patterns} of the contract's fixed items, from one run's parsed documents."""
    out = {name: set() for name in FILES}
    out["config.json"] |= {f"/args/{key}" for key in FIXED_ARGS}
    for path, value in scalar_leaves(documents["provenance.json"]):
        if path.startswith("/code/"):
            out["provenance.json"].add(pattern(path))
        if path.startswith("/pair/") and _keys(path)[-1].endswith("_path") and isinstance(value, str):
            out["provenance.json"].add(pattern(path))
    for path, value in scalar_leaves(documents["timing.json"]):
        if _number(value) and _keys(path)[-1] not in TIMING_COUNTS:
            if pattern(path) != LEVEL_NUMBER:
                out["timing.json"].add(pattern(path))
    return out


def mask(document, patterns, path=""):
    """A copy of `document` with every scalar leaf whose pattern is in `patterns` replaced by PLACEHOLDER."""
    if isinstance(document, dict):
        return {key: mask(value, patterns, f"{path}/{key}") for key, value in document.items()}
    if isinstance(document, list):
        return [mask(value, patterns, f"{path}[]") for value in document]
    return PLACEHOLDER if path in patterns else document


def masked_digests(texts, patterns):
    """{file: sha256 of the re-serialized masked document}; patterns: {file: iterable of patterns}."""
    out = {}
    for name in FILES:
        document = mask(parse(name, texts[name]), set(patterns.get(name, ())))
        out[name] = hashlib.sha256(serialize(name, document)).hexdigest()
    return out


def roundtrip_exact(texts):
    """{file: True when parse -> serialize gives the file's bytes back} (the comparison's premise)."""
    return {name: serialize(name, parse(name, texts[name])) == texts[name].encode("utf-8") for name in FILES}


def key_sets(texts):
    """SWLI-style key sets: {json file: sorted key paths}; log.jsonl: {row kind: sorted keys}."""
    from ._swli_keys import key_paths
    out = {name: sorted(key_paths(parse(name, texts[name]))) for name in JSON_FILES}
    kinds = {}
    for row in parse("log.jsonl", texts["log.jsonl"]):
        kinds.setdefault(row.get("kind"), set()).update(row)
    out["log.jsonl"] = {str(kind): sorted(keys) for kind, keys in sorted(kinds.items(), key=lambda kv: str(kv[0]))}
    return out


def measure(texts_a, texts_b, roots):
    """Gate 1 measurement on two runs of one arm: (S {file: sorted patterns}, measured [records], unclassified [records],
    digests a, digests b).  S = the classified measured patterns plus the fixed items of run a."""
    docs_a = {name: parse(name, texts_a[name]) for name in FILES}
    docs_b = {name: parse(name, texts_b[name]) for name in FILES}
    measured, unclassified = [], []
    S = {name: set(p) for name, p in fixed_patterns(docs_a).items()}
    for name in FILES:
        for path, a, b, kind in differences(docs_a[name], docs_b[name]):
            cls = None if kind != "value" else classify(name, path, a, b, roots=roots,
                                                        documents=(docs_a[name], docs_b[name]))
            record = dict(file=name, path=path, pattern=pattern(path), kind=kind, cls=cls, a=a, b=b)
            if cls is None:
                unclassified.append(record)
            else:
                measured.append(record)
                S[name].add(pattern(path))
    S = {name: sorted(p) for name, p in S.items()}
    return S, measured, unclassified, masked_digests(texts_a, S), masked_digests(texts_b, S)


def solve_quantities_in(S):
    """[(file, pattern)] of S entries whose path names a solve quantity (must be empty)."""
    out = []
    for name, patterns in S.items():
        for p in patterns:
            keys = _keys(p)
            if name == "provenance.json" and p.startswith("/code/"):
                continue
            if any(k in SOLVE_KEYS for k in keys) and not any(k in SD.TIMING_KEYS or "seconds" in k for k in keys):
                out.append((name, p))
    return out
