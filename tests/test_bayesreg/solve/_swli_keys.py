"""Key sets of the run_pair records for the gate-1 measure of work package SWLI (review SWLI minor 4, 2026-10-31).

Gate 1 says the configuration records have only the three keys sigma_domain, support_threshold, support_rule more.
The digests of _swli_digest.py cover the numeric fields of hyper.json and qc.json, the log rows and the arrays; this
module records the KEY PATHS of every JSON file of a run (config.json, provenance.json, hyper.json, qc.json, status.json,
timing.json): dict keys as /a/b, list items as [] (index-free), so that a run of the code before SWLI (ddcbfce, by
tests/test_bayesreg/_golden/make_golden_swli.py --keys) and a run of this code can be compared key by key.
Run: the default arm of _swli_digest (run_pair, numpy, the 26^3 synthetic pair, channel source tissue, rounds 1,1,1,1,
2 accepted steps).
"""
from pathlib import Path
import json

from . import _run_pair_digest as RD

FILES = ("config.json", "provenance.json", "hyper.json", "qc.json", "status.json", "timing.json")


def key_paths(value, path=""):
    out = set()
    if isinstance(value, dict):
        for key, item in value.items():
            out.add(f"{path}/{key}")
            out |= key_paths(item, f"{path}/{key}")
    elif isinstance(value, list):
        for item in value:
            out |= key_paths(item, f"{path}[]")
    return out


def run_key_sets(out):
    """{file: sorted key paths} of a finished run directory."""
    out = Path(out)
    return {name: sorted(key_paths(json.loads((out/name).read_text(encoding="utf-8")))) for name in FILES}


def all_key_sets(root, options=()):
    """Key sets of the default run (or run_pair with `options`) under `root` (a new directory)."""
    root = Path(root)
    root.mkdir(parents=True)
    base = RD.write_inputs(root)
    code, run = RD.run_arm(base, root/"run_keys", options)
    if code != 0:
        raise RuntimeError(f"exit {code}")
    return run_key_sets(run)
