"""Digests of whole bayesreg.cli.run_pair runs for the gate of work package DEF (2026-10-31).

DEF (contract 8a4d060: C05-2b first paragraph, C09.1-3a last item, D43, D47, D48; decision_log 10-31) changed two
defaults: logvol_points "gauss27" -> "gauss27+closed27" (beta_S 1, Simpson) and jd_cell_estimate "gauss27+corners" ->
"certificate"; with --metric-jd-scaling off an unset cell estimate is recorded as "none" and an explicit one refused.
The gate (the work package's acceptance (d) and the new default's golden):
  * the old options given explicitly run bitwise as the code before DEF (commit 8a4d060) ran them;
  * the new default (no option) runs bitwise as the code before DEF ran the same options given explicitly.
Both are checked against digests computed ONCE on the code of 8a4d060 (tests/test_bayesreg/_golden/
make_golden_run_pair.py: `git archive 8a4d060` in a temporary directory, this module copied in, the helpers it imports
byte-identical there), record golden_8a4d060_run_pair.json; constants GOLDEN_8A4D060_RUN_PAIR in
solve/test_def_defaults.py.

Arms (run_pair on the 26^3 synthetic pair of solve/test_run_pair.py, channel source tissue, numpy backend, option (a),
rounds 1,1,1,1, 2 accepted steps per (level, round), all four levels; ARM_ARGS):
  pre_def      --logvol-points gauss27 --jd-cell-estimate gauss27+corners       (the defaults before DEF, explicit)
  pre_def_off  --logvol-points gauss27 --metric-jd-scaling off                  (no cell estimate: before DEF it was
                                                                                 the default gauss27+corners, since
                                                                                 DEF "none"; the path is the same)
  corners      --logvol-points gauss27+corners --jd-cell-estimate gauss27+corners           (C05-2a, the G2/G3 arms)
  def_explicit --logvol-points gauss27+closed27 --logvol-closed27-share 1 --closed27-weights simpson
               --jd-cell-estimate certificate                                  (the DEF defaults, explicit)
On the code after DEF the test also runs `default` (no model / path option) and compares it with def_explicit.

The digest (sha256 of a canonical JSON; floats by float.hex, wall-clock and cache fields left out): every log.jsonl
row of kind candidate (level, round, outer, iteration, attempt, tau, rule, decision, certificate status / reason /
bound, energy, ared, certified, required), accept (state hash, energy, tau, bound, gradient norm), round (status,
reason, counts, F_B / GF_B / T_B / G0, sigma^2 used, state hash, bound), transfer (per back-off row: k, scale,
certificate status, state hash, bound) and jd_scaling (where, min J_a, factor counts, min r, the estimate's own fields
-- cell_bounds_cached and seconds are left out: the per-state cache is process-global), the status, solver status
and exported state hash of status.json, and cert.json (status, state hash, bound).
"""
import hashlib
import json
from pathlib import Path

from ..io_seg._helpers import DEV, allowlist_document, write_json
from .test_run_pair import write_synthetic_pair, write_tissue

COMMON = ("--rounds", "1,1,1,1", "--max-accepted", "2")
ARM_ARGS = dict(
    pre_def=("--logvol-points", "gauss27", "--jd-cell-estimate", "gauss27+corners"),
    pre_def_off=("--logvol-points", "gauss27", "--metric-jd-scaling", "off"),
    corners=("--logvol-points", "gauss27+corners", "--jd-cell-estimate", "gauss27+corners"),
    def_explicit=("--logvol-points", "gauss27+closed27", "--logvol-closed27-share", "1", "--closed27-weights",
                  "simpson", "--jd-cell-estimate", "certificate"),
)
GOLDEN_ARMS = tuple(ARM_ARGS)                  # computed on 8a4d060
DEFAULT_ARM = ()                               # the code's defaults (after DEF: compared with def_explicit)


def write_inputs(root):
    """The synthetic pair, tissue posteriors and allowlist of solve/test_run_pair.py under `root`; the run arguments
    without --out and the arm's options."""
    root = Path(root)
    allowlist = write_json(root/"dev_subjects.json", allowlist_document())
    grid, pair = write_synthetic_pair(root, DEV[0], DEV[1])
    write_tissue(root, grid, DEV[0])
    write_tissue(root, grid, DEV[1], shift=(.7, -.5, .4))
    return ["--pair", pair, "--channel-source", "tissue", "--data-root", str(root/"data"), "--allowlist",
            str(allowlist), "--tissue-root", str(root/"seg")]


def run_arm(base, out, arm_args):
    """run_pair.main on the inputs `base` with the arm's options; (exit code, run directory)."""
    from bayesreg.cli import run_pair
    from bayesreg.solve import jd_scaling as jds
    if hasattr(jds, "clear_cell_bounds_cache"):
        jds.clear_cell_bounds_cache()         # the per-state bounds cache is process-global (records only)
    code = run_pair.main([*base, "--out", str(out), *COMMON, *arm_args])
    return code, Path(out)


def _hex(value):
    return None if value is None else float(value).hex()


def _cert(record):
    record = record or {}
    return [record.get("status"), record.get("state_hash"), _hex(record.get("min_jd_lower_bound"))]


def _row(r):
    kind = r["kind"]
    if kind == "candidate":
        return ["c", r.get("level"), r.get("round"), r.get("outer"), r["iteration"], r["attempt"], _hex(r["tau"]),
                r.get("rule"), r["accepted"], r["reason"], r["cert_status"], r.get("cert_reason"), _hex(r["energy"]),
                _hex(r["min_jd_lower_bound"]), _hex(r.get("ared")), _hex(r.get("certified")), _hex(r.get("required"))]
    if kind == "accept":
        return ["a", r.get("level"), r.get("round"), r.get("outer"), r["iteration"], _hex(r["tau"]), _hex(r["energy"]),
                r["state_hash"], _hex(r["min_jd_lower_bound"]), _hex(r.get("gradient_norm"))]
    if kind == "round":
        return ["r", r["level"], r["round"], r["status"], r["reason"], r["inner_status"], r["accepted"], r["commits"],
                _hex(r["F_B_start"]), _hex(r["F_B_end"]), _hex(r["GF_B"]), _hex(r["T_B"]), _hex(r["G0"]),
                [_hex(s) for s in r["sigma2_used"]], r["state_hash"], _hex(r["min_jd_lower_bound"])]
    if kind == "transfer":
        return ["t", r["from_level"], r["to_level"], [[x["k"], _hex(x["scale"])]+_cert(x["cert"]) for x in r["rows"]]]
    if kind == "jd_scaling":
        keep = {k: (_hex(v) if isinstance(v, float) else v) for k, v in sorted(r.items())
                if "seconds" not in k and k != "cell_bounds_cached"}
        return ["j", keep]
    if kind in ("commit", "commit_check"):
        return [kind, r.get("level"), r.get("round"), r.get("outer"), r.get("state_hash")]
    return None


def run_digest(out):
    """The digest of a finished run directory (see the module docstring)."""
    out = Path(out)
    rows = []
    for line in (out/"log.jsonl").read_text(encoding="utf-8").splitlines():
        row = _row(json.loads(line))
        if row is not None:
            rows.append(row)
    status = json.loads((out/"status.json").read_text(encoding="utf-8"))
    final = [status.get("status"), status.get("solver_status"), status.get("exported")]
    cert = json.loads((out/"cert.json").read_text(encoding="utf-8")) if (out/"cert.json").is_file() else None
    text = json.dumps(dict(rows=rows, final=final, cert=_cert(cert)), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def all_digests(root):
    """{arm: digest} of the golden arms, run under `root` (a new directory, created here)."""
    root = Path(root)
    root.mkdir(parents=True)
    base = write_inputs(root)
    out = {}
    for arm in GOLDEN_ARMS:
        code, run = run_arm(base, root/f"run_{arm}", ARM_ARGS[arm])
        if code != 0:
            raise RuntimeError(f"{arm}: exit {code}")
        out[arm] = run_digest(run)
    return out
