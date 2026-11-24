"""Bitwise digests of the solver paths the MEM golden did not reach (work package FUP1; review MEM_复核_20261031,
important 1).

The MEM golden (tests/test_bayesreg/_golden/golden_9747553_mem.json, digests of _mem_digest.py) runs run_pair with no
CPU fallback, no device anomaly, no C08-8(f) recheck, option (a) and gauss27+closed27 only (review: gpu_comparison of
those runs has cpu_fallback 0, cpu_evaluations 0, cpu_cache_hits 0, no GF recheck).  The MEM changes that only those
paths exercise -- StableComparator.forget_other_states (the C08-8(d) CPU cache), Torch64Backend.evaluate sharing the
caller's u on the C08-8(e) fallback, run_pg adopting theta / u read-only under option (b) -- are digested here on the
same synthetic 26^3 pair (14 structure channels, run_pair end to end, --device cpu):

  case                      backend  run_pair options (besides the pair inputs)                 module patches
  torch64_fallback          torch64  BASE                                                       band
  torch64_faults            torch64  BASE                                                       faults
  torch64_fallback_faults   torch64  BASE                                                       band, faults
  {numpy,torch64}_outer_b   both     BASE with --outer b
  {numpy,torch64}_corners   both     --logvol-points gauss27+corners --jd-cell-estimate gauss27+corners, rest as BASE
  {numpy,torch64}_gauss27_off_r3  both  --logvol-points gauss27 --metric-jd-scaling off --rounds 1,1,1,3
                                        --max-accepted 4 (outer a)
  ---- regression only (current code; the GF recheck raised TypeError before GFR 34ddaef, so no 9747553 golden) ----
  torch64_gf_a              torch64  BASE                                                       gf
  torch64_gf_b              torch64  BASE with --outer b                                        gf
  torch64_gf_fallback_faults torch64 BASE                                                       band, faults, gf

BASE = _mem_digest.RUN_OPTIONS (gauss27+closed27, share 1, simpson, affine_v1, certificate cell estimate, outer a,
rounds 1,1,1,2, 3 accepted steps per round).  Patches (module attributes set for the run and restored; the source is
not changed; the attributes exist on 9747553 and on the current code):
  band    bayesreg.backends.torch64.stable.BAND_FACTOR = 1e300: every C08-8(d) comparison falls back to the CPU
          reference (cache, base slot, reduction memo);
  faults  bayesreg.backends.torch64.evaluate raises RuntimeError("injected device fault (test, call k)") on every 3rd
          call k = 3, 6, ...: Torch64Backend.evaluate recomputes on the CPU (C08-8(e)) and the comparisons with a CPU
          point take the reference path;
  gf      bayesreg.backends.torch64.stable.GF_BAND = 1e30: every GF_B is rechecked on the CPU (C08-8(f)).

Digest of a case: _mem_digest.rows_digest of log.jsonl (timing keys removed), exit code, status.json status, exported
hash and device_anomalies, the hyper.json part of _mem_digest (rounds, transfers, sigma_used, sigma_post,
solver_status, status, GF_B_cpu_final, sigma_monitor) PLUS gpu_comparison (C08-8(h): fallbacks, reasons, device
anomalies, cpu_evaluations, cpu_cache_hits, cpu_base_slot_hits, cpu_reductions, memo hits, GF rechecks; timing keys
removed), cert.json state_hash and min_jd_lower_bound, final_native.npz state_hash, sha256 of disp.npz d.  `paths`
repeats the gpu_comparison counters in clear so that a test can check that the case reaches its path.
hyper.json backend_after_solve (device_states changed on purpose by MEM) is not part of the digest.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile

import numpy as np

from . import _mem_digest as D

BASE = tuple(D.RUN_OPTIONS)


def _replace(options, flag, value):
    options = list(options)
    options[options.index(flag)+1] = value
    return tuple(options)


OUTER_B = _replace(BASE, "--outer", "b")
CORNERS = ("--logvol-points", "gauss27+corners", "--metric-jd-scaling", "affine_v1", "--jd-cell-estimate",
           "gauss27+corners", "--outer", "a", "--rounds", "1,1,1,2", "--max-accepted", "3")
GAUSS27_OFF_R3 = ("--logvol-points", "gauss27", "--metric-jd-scaling", "off", "--outer", "a", "--rounds", "1,1,1,3",
                  "--max-accepted", "4")

# name -> (backend, options, patches)
GOLDEN_CASES = {
    "torch64_fallback": ("torch64", BASE, ("band",)),
    "torch64_faults": ("torch64", BASE, ("faults",)),
    "torch64_fallback_faults": ("torch64", BASE, ("band", "faults")),
    "numpy_outer_b": ("numpy", OUTER_B, ()),
    "torch64_outer_b": ("torch64", OUTER_B, ()),
    "numpy_corners": ("numpy", CORNERS, ()),
    "torch64_corners": ("torch64", CORNERS, ()),
    "numpy_gauss27_off_r3": ("numpy", GAUSS27_OFF_R3, ()),
    "torch64_gauss27_off_r3": ("torch64", GAUSS27_OFF_R3, ()),
}
GF_CASES = {
    "torch64_gf_a": ("torch64", BASE, ("gf",)),
    "torch64_gf_b": ("torch64", OUTER_B, ("gf",)),
    "torch64_gf_fallback_faults": ("torch64", BASE, ("band", "faults", "gf")),
}
CASE_SETS = dict(golden=GOLDEN_CASES, gf=GF_CASES)
FAULT_EVERY = 3
PATH_KEYS = ("comparisons", "cpu_fallback", "cpu_evaluations", "cpu_cache_hits", "cpu_base_slot_hits",
             "cpu_reductions", "cpu_reduction_memo_hits")


@contextmanager
def _setattr(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextmanager
def patched(patches):
    """The module patches of a case (restored on exit)."""
    unknown = set(patches)-{"band", "faults", "gf"}
    if unknown:
        raise ValueError(f"unknown patches {sorted(unknown)}")
    if not patches:
        yield
        return
    import bayesreg.backends.torch64 as T
    from bayesreg.backends.torch64 import stable
    with _setattr(stable, "BAND_FACTOR", 1e300 if "band" in patches else stable.BAND_FACTOR), \
            _setattr(stable, "GF_BAND", 1e30 if "gf" in patches else stable.GF_BAND):
        if "faults" not in patches:
            yield
            return
        evaluate = T.evaluate
        calls = [0]

        def faulty(*args, **kwargs):
            calls[0] += 1
            if calls[0] % FAULT_EVERY == 0:
                raise RuntimeError(f"injected device fault (test, call {calls[0]})")
            return evaluate(*args, **kwargs)

        with _setattr(T, "evaluate", faulty):
            yield


def run_pair_digest(backend, options, patches=()):
    """run_pair.main with structure channels on the synthetic pair (tmp directory) under `patches`, digested."""
    from bayesreg.cli import run_pair
    from ..io_seg._helpers import DEV, allowlist_document, write_json
    from .test_run_pair_structure import cli_args, make_pair, registry_models, write_registry
    with tempfile.TemporaryDirectory(prefix="mem_paths_digest_") as tmp:
        root = Path(tmp)
        allowlist = write_json(root/"dev_subjects.json", allowlist_document())
        registry = write_registry(root/"segmenters.json", registry_models())
        _, pair = make_pair(root, DEV[0], DEV[1], registry_models())
        out = root/f"run_{backend}"
        extra = list(options)+["--backend", backend]+(["--device", "cpu"] if backend == "torch64" else [])
        with patched(tuple(patches)):
            code = run_pair.main(cli_args(root, allowlist, registry, pair, out, *extra))
        rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
        hyper = json.loads((out/"hyper.json").read_text())
        status = json.loads((out/"status.json").read_text())
        cert = json.loads((out/"cert.json").read_text())
        with np.load(out/"disp.npz") as z:
            disp = D.sha(np.ascontiguousarray(z["d"]).tobytes())
        with np.load(out/"final_native.npz") as z:
            final = str(z["state_hash"])
    keys = ("rounds", "transfers", "sigma_used", "sigma_post", "solver_status", "status", "GF_B_cpu_final",
            "sigma_monitor", "gpu_comparison")
    hyper_part = {k: hyper.get(k) for k in keys}
    comparison = hyper.get("gpu_comparison")
    paths = None
    if isinstance(comparison, dict) and "comparisons" in comparison:          # torch64 only
        paths = {k: comparison.get(k) for k in PATH_KEYS}
        paths.update(device_anomalies=comparison.get("device_anomalies"),
                     gf_rechecks=(comparison.get("gf_rechecks") or {}).get("count"))
    return dict(D.rows_digest(rows), exit_code=code, status=status.get("status"), exported=status.get("exported"),
                device_anomalies=status.get("device_anomalies"), hyper_sha256=D.sha(D.canonical(hyper_part)),
                gpu_comparison_sha256=D.sha(D.canonical(comparison)), paths=paths,
                cert_state_hash=cert.get("state_hash"), cert_min_jd_lower_bound=cert.get("min_jd_lower_bound"),
                final_state_hash=final, disp_d_sha256=disp)


def case_digest(name):
    backend, options, patches = {**GOLDEN_CASES, **GF_CASES}[name]
    return run_pair_digest(backend, options, patches)


def compute(case_set):
    return {name: run_pair_digest(*spec) for name, spec in CASE_SETS[case_set].items()}
