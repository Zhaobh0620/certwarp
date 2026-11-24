"""Bitwise digests of whole solver trajectories for the MEM gate (work package MEM; memory-only changes).

The MEM changes (bayesreg.readonly, ModelState sharing, LevelInput release, early drops in solve_multilevel /
solve_direct / run_pg, zero-copy hashes, device states of other grids evicted) must leave every value unchanged.  These
functions run the same synthetic problems on the code found on sys.path (the extracted commit before MEM, or this tree)
and reduce the trajectory to digests:

  run_pair_digest(backend)  run_pair.main end to end on the synthetic 26^3 pair of test_run_pair /
                            test_run_pair_structure: structure channels (K_S = 13, i.e. 14 model channels as the native
                            runs), --logvol-points gauss27+closed27 --logvol-closed27-share 1 --closed27-weights simpson
                            --metric-jd-scaling affine_v1 --jd-cell-estimate certificate --outer a --rounds 1,1,1,2
                            --max-accepted 3, --backend numpy | torch64 (--device cpu); every option explicit.
  solve_digest(backend, release)  solve_multilevel on four synthetic LevelInputs with 3 channels (intensity + two
                            derived channels), the same options through SolverConfig, GaussianMStep, rounds (1, 1, 1, 2),
                            3 accepted steps per round; release=True (only where solve_multilevel has release_inputs)
                            hands the levels over as a list.

Digest of a run: the log.jsonl rows (run_pair) or the log rows (solve) with every timing field removed (keys that
contain "seconds", "elapsed", "_unix" or "timing"), each as canonical JSON (sorted keys) -> sha256, the list of those
row digests, their count, the accepted state_hash sequence in order, the round rows' state_hash, the final state's
hash; run_pair also the hyper.json rounds / transfers / sigma_used / sigma_post (timing removed), status.json status and
exported hash, cert.json state_hash and min_jd_lower_bound, and the sha256 of disp.npz's d.  Records that MEM may change
on purpose (hyper.json backend_after_solve.device_states: the device states of other grids are dropped) are not part
of the digest.
"""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

TIME_KEYS = ("seconds", "elapsed", "_unix", "timing")
RUN_OPTIONS = ("--logvol-points", "gauss27+closed27", "--logvol-closed27-share", "1", "--closed27-weights", "simpson",
               "--metric-jd-scaling", "affine_v1", "--jd-cell-estimate", "certificate", "--outer", "a",
               "--rounds", "1,1,1,2", "--max-accepted", "3")


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if not any(t in str(k) for t in TIME_KEYS)}
    if isinstance(value, (list, tuple)):
        return [strip(v) for v in value]
    if isinstance(value, np.ndarray):
        return strip(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def canonical(value):
    return json.dumps(strip(value), sort_keys=True, default=repr, separators=(",", ":"))


def sha(text):
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def rows_digest(rows):
    digests = [sha(canonical(r))[:16] for r in rows]
    accepted = [r["state_hash"] for r in rows if r.get("kind") == "accept"]
    rounds = [r.get("state_hash") for r in rows if r.get("kind") == "round"]
    return dict(n_rows=len(rows), rows_sha256=sha("".join(digests)), row_digests=digests, accepted_state_hashes=accepted,
                round_state_hashes=rounds)


# ---- run_pair end to end ----------------------------------------------------------------------------------

def run_pair_digest(backend):
    """run_pair.main with structure channels on the synthetic pair (tmp directory), digested."""
    from bayesreg.cli import run_pair
    from ..io_seg._helpers import DEV, allowlist_document, write_json
    from .test_run_pair_structure import cli_args, make_pair, registry_models, write_registry
    with tempfile.TemporaryDirectory(prefix="mem_digest_") as tmp:
        root = Path(tmp)
        allowlist = write_json(root/"dev_subjects.json", allowlist_document())
        registry = write_registry(root/"segmenters.json", registry_models())
        _, pair = make_pair(root, DEV[0], DEV[1], registry_models())
        out = root/f"run_{backend}"
        extra = list(RUN_OPTIONS)+["--backend", backend]+(["--device", "cpu"] if backend == "torch64" else [])
        code = run_pair.main(cli_args(root, allowlist, registry, pair, out, *extra))
        rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
        hyper = json.loads((out/"hyper.json").read_text())
        status = json.loads((out/"status.json").read_text())
        cert = json.loads((out/"cert.json").read_text())
        with np.load(out/"disp.npz") as z:
            disp = sha(np.ascontiguousarray(z["d"]).tobytes())
        with np.load(out/"final_native.npz") as z:
            final = str(z["state_hash"])
    hyper_part = {k: hyper.get(k) for k in ("rounds", "transfers", "sigma_used", "sigma_post", "solver_status", "status",
                                             "GF_B_cpu_final", "sigma_monitor")}
    return dict(rows_digest(rows), exit_code=code, status=status.get("status"), exported=status.get("exported"),
                hyper_sha256=sha(canonical(hyper_part)), cert_state_hash=cert.get("state_hash"),
                cert_min_jd_lower_bound=cert.get("min_jd_lower_bound"), final_state_hash=final, disp_d_sha256=disp)


# ---- solve_multilevel ---------------------------------------------------------------------------------------

def _levels(n=26):
    """Four LevelInputs of tests/test_bayesreg/solve/_synthetic.four_levels with two more channels (smooth functions
    of the intensity image) on both sides, so that every channel array has C = 3."""
    from bayesreg.solve import LevelInput
    from . import _synthetic as syn
    _, levels, _ = syn.four_levels(n)
    out = []
    for lv in levels:
        def stack(image):
            return np.stack([image[0], np.sqrt(1.+image[0]**2), np.cos(image[0])])
        tissue = syn.uniform_tissue(lv.grid)
        out.append(LevelInput(lv.level, lv.grid, _ro(stack(lv.fixed)), _ro(stack(lv.moving)), lv.mask, _ro(tissue)))
    return out


def _ro(array):
    array = np.array(array, dtype=np.float64, copy=True, order="C")
    array.flags.writeable = False
    return array


def solve_digest(backend_name, release=False):
    import inspect
    from bayesreg.solve import (Caps, Eta, GaussianMStep, JDScalingConfig, PGConfig, SolverConfig, make_backend,
                                solve_multilevel)
    levels = _levels()
    backend = make_backend(backend_name, device="cpu") if backend_name == "torch64" else make_backend("numpy")
    config = SolverConfig(outer="a", rounds=(1, 1, 1, 2), caps=Caps(max_accepted=3),
                          pg=PGConfig(metric_jd_scaling="affine_v1", jd=JDScalingConfig(cell_estimate="certificate")),
                          logvol_points="gauss27+closed27", logvol_closed27_share=1., closed27_weights="simpson")
    rows = []
    kwargs = {}
    if release:
        if "release_inputs" not in inspect.signature(solve_multilevel).parameters:
            raise RuntimeError("solve_multilevel has no release_inputs in this tree")
        kwargs["release_inputs"] = True
    else:
        levels = tuple(levels)
    result = solve_multilevel(backend, levels, eta=Eta(1., 1., (1., .25, .25, .25)), sigma=GaussianMStep(),
                              config=config, log=rows.append, **kwargs)
    return dict(rows_digest(rows), status=result.status, native=bool(result.native),
                final_state_hash=None if result.commit is None else result.commit.cert.state_hash,
                sigma_post=strip(result.sigma_post), rounds_sha256=sha(canonical(result.rounds)),
                transfers_sha256=sha(canonical(result.transfers)))


def compute(include_release):
    digests = dict(run_pair={}, solve={})
    for backend in ("numpy", "torch64"):
        digests["run_pair"][backend] = run_pair_digest(backend)
        digests["solve"][backend] = solve_digest(backend)
        if include_release:
            digests["solve"][backend+"_release"] = solve_digest(backend, release=True)
    return digests
