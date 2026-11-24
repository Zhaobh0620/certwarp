"""Digests of the brain default for the gate of work package SWLI (contract C04-6 gate 1, gate 10; 2026-10-31).

Gate 1: without the switch and with an explicit `--sigma-domain brain`, run_pair gives the arrays, the log.jsonl rows,
and the numeric fields of hyper.json and qc.json (timing fields left out) of the code before SWLI; the configuration
records have only the three keys sigma_domain ("brain"), support_threshold (null), support_rule (null) more.  The
existing goldens (_run_pair_digest, MEM, EB, DEF) cover the arrays and the log rows; this module adds ("补摘要") the
numeric fields of hyper.json and qc.json and the brain Laplace samples (gate 10), computed ONCE on the code before SWLI
(commit ddcbfce) by tests/test_bayesreg/_golden/make_golden_swli.py (git archive, this module copied in).

Run (run_pair, numpy backend, the 26^3 synthetic pair of solve/test_run_pair.py, channel source tissue, rounds 1,1,1,1,
2 accepted steps per (level, round), the code defaults): digest = sha256 of a canonical JSON of
  run    _run_pair_digest.run_digest (log rows, status, exported state, cert.json);
  hyper  every leaf of hyper.json (floats by float.hex) except: keys naming time (TIMING_KEYS, and any key containing
         "seconds"), the backend records (backend_after_solve: thread / device descriptions), and in "algorithm" the
         three SWLI keys (absent before SWLI; checked separately by the test);
  qc     every leaf of qc.json, same rules;
  disp   sha256 of disp.npz's array d and final_native.npz's u.
Laplace (gate 10): sample_perturbations on infer/_cases.cube6 (seed 1) at u_hat_of(grid) with rng default_rng(7), S = 4:
sha256 of the stacked deltas, of the PCG iteration counts and of the converged flags.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import _run_pair_digest as RD

TIMING_KEYS = ("started_unix", "total_seconds", "seconds", "elapsed", "cell_bounds_cached", "peak_rss_mb",
               "wall_budget", "backend_after_solve", "device_info", "resources")
SWLI_KEYS = ("sigma_domain", "support_threshold", "support_rule")
ARMS = dict(default=(), brain_explicit=("--sigma-domain", "brain"))
GOLDEN_ARMS = ("default",)             # the code before SWLI has no --sigma-domain


def _skip(key):
    return key in TIMING_KEYS or "seconds" in key


def leaves(value, path=""):
    """[(path, canonical leaf)] of a JSON value (floats by float.hex), timing keys left out."""
    out = []
    if isinstance(value, dict):
        for key in sorted(value):
            if _skip(key):
                continue
            item = value[key]
            if path == "/algorithm" and key in SWLI_KEYS:
                continue
            out += leaves(item, f"{path}/{key}")
    elif isinstance(value, list):
        for k, item in enumerate(value):
            out += leaves(item, f"{path}[{k}]")
    elif isinstance(value, float):
        out.append((path, value.hex()))
    else:
        out.append((path, value))
    return out


def json_digest(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(leaves(document), sort_keys=True).encode()).hexdigest()


def array_digest(path, key):
    with np.load(path) as data:
        return hashlib.sha256(np.ascontiguousarray(data[key]).tobytes()).hexdigest()


def run_digests(out):
    out = Path(out)
    return dict(run=RD.run_digest(out), hyper=json_digest(out/"hyper.json"), qc=json_digest(out/"qc.json"),
                disp=array_digest(out/"disp.npz", "d"), final_native=array_digest(out/"final_native.npz", "u"))


def laplace_digest():
    from bayesreg.infer.laplace import sample_perturbations
    from ..infer._cases import cube6, u_hat_of
    grid, state = cube6(seed=1)
    u_hat = u_hat_of(grid)
    samples = sample_perturbations(state, u_hat, np.random.default_rng(7), S=4)
    deltas = np.stack([np.asarray(d) for d in samples.deltas])
    return dict(deltas=hashlib.sha256(deltas.tobytes()).hexdigest(),
                iterations=list(int(i) for i in samples.iterations), converged=list(bool(c) for c in samples.converged))


def all_digests(root, arms=GOLDEN_ARMS):
    """{arm: run digests} of the arms run under `root` (a new directory) and the Laplace digest."""
    root = Path(root)
    root.mkdir(parents=True)
    base = RD.write_inputs(root)
    out = {}
    for arm in arms:
        code, run = RD.run_arm(base, root/f"run_{arm}", ARMS[arm])
        if code != 0:
            raise RuntimeError(f"{arm}: exit {code}")
        out[arm] = run_digests(run)
    return dict(runs=out, laplace=laplace_digest())
