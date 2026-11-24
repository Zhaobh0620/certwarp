"""Digests for the gate 1 of work package JDTI (contract C09.1-3a certificate_bernstein, gate 1): the existing options
("certificate", the default; "closed27"; "gauss27+corners") run bitwise as before the new option.

The digests were computed ONCE on the code of main 96c8d35 (before JDTI) with this module and the run helpers of
solve/test_jd_cell_estimate.py, by tests/test_bayesreg/_golden/make_golden_jdti.py (`git archive 96c8d35` in a
temporary directory; this file loaded from the JDTI tree by path, it only uses json and hashlib); record
tests/test_bayesreg/_golden/golden_96c8d35_jdti.json; constants GOLDEN_96C8D35 in solve/test_jd_cell_estimate_bernstein.py.

Digest: sha256 of a canonical JSON of every row of kind candidate / accept / jd_scaling / commit / commit_check /
warning (every key; floats by float.hex; keys ending in "seconds" left out: wall clock), the status, the accepted count
and the energies; for the outer solvers also every S~ computation of the round record (same rule).
"""
import hashlib
import json

KINDS = ("candidate", "accept", "jd_scaling", "commit", "commit_check", "warning")


def _canon(value):
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, dict):
        return {str(k): _canon(v) for k, v in sorted(value.items()) if not str(k).endswith("seconds")}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return _canon(value.item())
    return value


def rows_digest(rows, status, accepted, energies, computations=None):
    keep = [_canon(r) for r in rows if r.get("kind") in KINDS]
    doc = dict(rows=keep, status=status, accepted=accepted, energies=[float(e).hex() for e in energies],
               computations=None if computations is None else _canon(list(computations)))
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def golden_cases(T, jds, PGConfig, JDScalingConfig):
    """{case: digest} with the helpers of test_jd_cell_estimate (module T) and the jd_scaling module `jds`."""
    out = {}
    for estimate in (jds.CERTIFICATE, jds.CLOSED27, jds.GAUSS_CORNERS):
        for kind in ("mid", "off"):
            jds.clear_cell_bounds_cache()
            result, rows, metric = T.run_quadratic(estimate, kind=kind)
            out[f"quadratic_{kind}_{estimate}"] = rows_digest(rows, result.status, result.accepted, result.energies,
                                                              metric.computations)
    for option in ("a", "b"):
        jds.clear_cell_bounds_cache()
        result, rows = T.run_outer(option, PGConfig(jd=JDScalingConfig(cell_estimate=jds.CERTIFICATE)))
        out[f"outer_{option}_{jds.CERTIFICATE}"] = rows_digest(rows, result.status, result.accepted, result.energies,
                                                               result.metric["jd_scaling"]["computations"])
    return out
