"""Unified evaluation entry of the B line (work package EV; project rules "评测约定", hard constraints 4, 6, 7).

    python -m bayesreg.eval.cli RUN_DIR... --pairs-stage STAGE --out metrics.json --long long.csv --out-dir DIR
        [--summary summary.json] [--pair PAIR] [--run-id ID]
        [--set NAME] [--split dev] [--method NAME] [--arm NAME]
        [--cert auto|full|jacobian_only|none] [--cert-tile-size 8192] [--no-cell-split]
        [--data-root DIR] [--allowlist PATH] [--unseal-token TOKEN]

Every method's output is scored by the same code from its displacement field (fields.py: disp.npz:d, or a run's
final_native u converted to d); metrics.py and cert_eval.py hold the definitions, which are copied verbatim into each
metrics.json. Truth labels come from <data-root>/<STAGE>/<pair>/pair.npz (never from a run directory).

Order. Phase 1, for every run directory and before any label is read: run kind, completion and pair (fields.py:
a run that has not ended -- no status.json exit_code, no exit_code file, or exit 0 without its completion records --
is refused; bayesreg: config.json args.pair, whose stage must be STAGE; configuration C: protocol.json; disp.npz
directory: --pair), then the subject guard of truth.py (names, sealed lists, manifest subjects on the development
allowlist), then the run's own input record against the admitted pair (configuration C protocol.json input_sha256
and pair_dataset; bayesreg data.txt pair_provenance; checked when recorded). One refused pair stops the whole call
with exit code 3 and nothing is read or written (an unseal token is deprecated, records only; unsealing is C23's unseal
object, and the truth of a sealed subject is never read here: the hold-out evaluation is C23.12's own entry). Output paths are checked: --out-dir is required (nothing is ever written into an evaluated run
directory, hard constraint 5; review EV item 1), it may not lie inside an evaluated run directory or a read-only
seq_vp_3d tree, run ids are plain names, nothing is overwritten, and the parent directories of --long and --summary
must exist. Phase 2, per run: field, truth labels, metrics, certificate (auto: full for bayesreg and configuration C
maps, jacobian_only for disp.npz directories), and <out-dir>/<run_id>/<out> is written. An evaluation error is
written into that run's metrics.json (error, traceback) and the call ends with exit code 1 after the others; nothing
is retried.
Field role (fields.py; review EV item 2): every record and long row carries field_role (native_formal, handover_only,
no_native_result, unknown, disp_only) and run_status; the summary groups by (set, method, arm, field_role, and the
structure_registry label below), so a
configuration C handover_only state is never averaged with native formal states, and bayesreg.eval.paired compares
native_formal and disp_only rows only unless asked otherwise.
Structure-layer registry (review PS item 2; a label, the metrics are unchanged): every record carries
structure_registry (fields.py; None unless a bayesreg run with channel_source structure) and
development_diagnostic_only (true when that registry is staging_unregistered, 未正式登记); the printed line shows
[staging_unregistered]; the long CSV carries the label in its column structure_registry (n/a, registered,
staging_unregistered, not_recorded), and the summary groups by it too (review PS1_PS2 item d: staging and registered
runs are never averaged together; each group still lists structure_registry_status and staging_unregistered_runs);
bayesreg.eval.paired leaves staging rows out unless --include-staging.
Then --long (columns set,split,method,arm,pair,measure,value,run_id,field_role,run_status,structure_registry;
non-finite values written as "inf", "-inf", "nan", review EV item 3) and, with --summary, the per (set, method, arm,
field_role, structure_registry) aggregate
(scripts/evaluate_l2r_style.aggregate plus the means of every scalar measure over its finite values, with the count
of non-finite values next to each mean). Certificate measures carry the mode: cert_full_* or cert_jacobian_only_*.
Exit codes: 0 all evaluated, 1 an evaluation error, 2 usage or input error, 3 subject sealed.
Environment (EVF): run as the program (python -m bayesreg.eval.cli), the evaluation defaults to the CPU and numba 2
threads (CUDA_VISIBLE_DEVICES="" and NUMBA_NUM_THREADS=2 unless already set). Importing this module or calling main()
in another process changes no environment variable.
"""
import os

if __name__ == "__main__":
    # EVF: command-line defaults only, set before any import below. Not at module level (that hid the GPU from every
    # importer, e.g. a pytest session, and from its child processes) and not in main() (tests call main() in-process;
    # main() could also come after numba was loaded). The packages bayesreg and bayesreg.eval, which runpy imports
    # first, import neither numba nor torch, and this module loads them lazily (certificate), so the defaults hold.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import argparse                                                    # noqa: E402
from dataclasses import dataclass                                  # noqa: E402
from datetime import datetime, timezone                            # noqa: E402
import hashlib                                                     # noqa: E402
import importlib.metadata                                          # noqa: E402
from pathlib import Path                                           # noqa: E402
import subprocess                                                  # noqa: E402
import sys                                                         # noqa: E402
import traceback                                                   # noqa: E402

import numpy as np                                                 # noqa: E402

from bayesreg.io.guard import SubjectSealed, load_dev_allowlist   # noqa: E402
from bayesreg.solve.backend import CERT_TILE_SIZE                  # noqa: E402
from bayesreg.solve.export import dumps                            # noqa: E402

from . import cert_eval, metrics                                   # noqa: E402
from .fields import (COMPARABLE_ROLES, ROLE_HANDOVER, FieldError, d_to_u, describe_run,  # noqa: E402
                     field_present, load_field)
from .paired import LONG_COLUMNS, format_value, write_long         # noqa: E402,F401 (LONG_COLUMNS re-exported)
from .truth import (DEFAULT_ALLOWLIST, DEFAULT_DATA_ROOT, REPO, SAFE_NAME, SealedListError, admit_pair,  # noqa: E402
                    load_sealed_lists, load_truth, scheme_for)

SCHEMA = "bayesreg.eval.metrics.v2"           # v2: field_role, completion, input_check, units
READ_ONLY_ROOTS = (Path("/home/rmcfz/code/python_projects/seq_vp_3d"), Path("/mnt/d/seq_vp_3d_results"))
DEFAULT_METHOD = dict(bayesreg="bayesreg", configC="configC")
AUTO_CERT = dict(bayesreg="full", configC="full", disp="jacobian_only")
PACKAGES = ("numpy", "scipy", "numba", "surface-distance")
SKIP_LONG = ("n_cells", "n_brain_cells")
CODE_FILES = ("bayesreg/io/guard.py", "bayesreg/io/pair.py", "bayesreg/solve/backend.py", "bayesreg/solve/export.py",
              "jd_curl/three_d/feasibility.py", "jd_curl/three_d/nifti.py")


class UsageError(ValueError):
    """A command-line or output-path problem (exit code 2)."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m bayesreg.eval.cli", description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("--pairs-stage", required=True, help="stage directory under --data-root holding the pairs")
    parser.add_argument("--out", required=True, help="metrics file name written per run (plain name)")
    parser.add_argument("--long", required=True, type=Path, help="long CSV of every run (must not exist)")
    parser.add_argument("--out-dir", type=Path, help="REQUIRED: <out-dir>/<run_id>/<out> is written (never into "
                        "an evaluated run directory)")
    parser.add_argument("--summary", type=Path, help="aggregate JSON per (set, method, arm, field_role) (must not exist)")
    parser.add_argument("--pair", help="pair name of a disp.npz run directory (one run directory only)")
    parser.add_argument("--run-id", help="run id (one run directory only; default: the run directory's run_id file, "
                        "else the directory name)")
    parser.add_argument("--set", dest="set_name", help="set name in long.csv (default: the stage name)")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--method", help="method name (default bayesreg / configC by run kind; required for disp.npz)")
    parser.add_argument("--arm", help="arm name (default: a bayesreg run's channel_source, otherwise empty)")
    parser.add_argument("--cert", choices=("auto", "full", "jacobian_only", "none"), default="auto")
    parser.add_argument("--cert-tile-size", type=int, default=CERT_TILE_SIZE)
    parser.add_argument("--no-cell-split", action="store_true", help="skip the per-cell brain/background pass")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--unseal-token", help="deprecated, records only (sha256), admits nothing; unsealing is C23's unseal object (D101)")
    return parser.parse_args(argv)


def _inside(path, root):
    resolved, root = Path(path).resolve(), Path(root).resolve()
    return resolved == root or root in resolved.parents


def _under_read_only(path):
    return any(_inside(path, root) for root in READ_ONLY_ROOTS)


def check_inputs(info, guard, stage):
    """Item 8 of review EV: the run's own record of its input against the admitted pair (when recorded).
    Returns the record; UsageError on a mismatch."""
    own = info.extra.get("inputs") or {}
    checks = {}
    if own.get("pair_sha256") is not None:
        checks["pair_sha256"] = own["pair_sha256"] == guard["pair_sha256_manifest"]
    if info.kind == "configC" and own.get("stage") is not None:
        checks["stage"] = own["stage"] == stage
    for side in ("fixed", "moving"):
        if own.get(f"{side}_subject") is not None:
            checks[f"{side}_subject"] = own[f"{side}_subject"] == guard[f"{side}_subject"]
    bad = sorted(k for k, ok in checks.items() if not ok)
    if bad:
        raise UsageError(f"{info.run_dir}: the run's input record ({own.get('source')}) differs from the admitted pair "
                         f"in {bad}; nothing was read or written")
    return dict(source=own.get("source"), checked=checks, recorded=bool(checks),
                note=None if checks else "the run records no input identity; not checked")


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout
    except OSError as exc:
        return f"(git unavailable: {exc})"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_record():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    status = _git("status", "--porcelain", "--", "bayesreg", "jd_curl")
    files = {str(p.relative_to(REPO)): _sha(p) for p in sorted((REPO/"bayesreg"/"eval").glob("*.py"))}
    files.update({name: _sha(REPO/name) for name in CODE_FILES if (REPO/name).is_file()})
    return dict(repo=str(REPO), git_head=_git("rev-parse", "HEAD").strip(), git_dirty_bayesreg_jd_curl=bool(status.strip()),
                git_status=status, file_sha256=files, python=sys.version.split()[0], packages=versions)


@dataclass
class Planned:
    info: object
    run_id: str
    method: str
    arm: str
    set_name: str
    pair_dir: Path
    grid: object
    guard: dict
    out_path: Path
    cert_mode: str | None
    input_check: dict | None = None


def plan(args, allowlist, sealed):
    """Phase 1 (no label read): describe every run (completion, role), admit its pair, check the run's input record,
    fix its output path."""
    if (args.pair or args.run_id) and len(args.runs) != 1:
        raise UsageError("--pair and --run-id need exactly one run directory")
    if Path(args.out).name != args.out or args.out in ("", ".", ".."):
        raise UsageError("--out: a plain file name (written as <out-dir>/<run_id>/<out>)")
    if args.cert_tile_size < 1:
        raise UsageError("--cert-tile-size: positive integer")
    if args.out_dir is None:
        raise UsageError("--out-dir is required: nothing is written into an evaluated run directory (hard constraint 5)")
    if _under_read_only(args.out_dir):
        raise UsageError("--out-dir lies under a read-only seq_vp_3d tree")
    if args.run_id is not None and (not SAFE_NAME.fullmatch(args.run_id) or ".." in args.run_id):
        raise UsageError(f"--run-id {args.run_id!r}: a plain name ([A-Za-z0-9._-], no '..') is required")
    for path, name in ((args.long, "--long"), (args.summary, "--summary")):
        if path is None:
            continue
        if path.exists() or _under_read_only(path):
            raise UsageError(f"{name} {path}: exists or lies under a read-only tree; nothing is overwritten")
        if not path.parent.is_dir():
            raise UsageError(f"{name} {path}: the parent directory does not exist")
    planned, outs = [], set()
    for run in args.runs:
        info = describe_run(run, pair=args.pair)
        if info.stage is not None and info.stage != args.pairs_stage:
            raise UsageError(f"{run}: the run's stage {info.stage!r} differs from --pairs-stage {args.pairs_stage!r}")
        pair_dir, grid, _, guard = admit_pair(args.data_root, args.pairs_stage, info.pair, allowlist, sealed,
                                              unseal_token=args.unseal_token)
        input_check = check_inputs(info, guard, args.pairs_stage)
        method = args.method or DEFAULT_METHOD.get(info.kind)
        if not method:
            raise UsageError(f"{run}: --method is required for a disp.npz directory")
        run_id = args.run_id or info.run_id_hint or Path(run).resolve().name
        if not SAFE_NAME.fullmatch(run_id) or ".." in run_id:
            raise UsageError(f"{run}: run id {run_id!r} is not a plain name; give --run-id")
        out_path = args.out_dir/run_id/args.out
        if _under_read_only(out_path):
            raise UsageError(f"{out_path}: lies under a read-only seq_vp_3d tree")
        if out_path.exists() or out_path.resolve() in outs:
            raise UsageError(f"{out_path}: exists or is written twice; nothing is overwritten")
        outs.add(out_path.resolve())
        cert_mode = None if args.cert == "none" else (AUTO_CERT[info.kind] if args.cert == "auto" else args.cert)
        arm = (info.arm_hint or "") if args.arm is None else args.arm
        planned.append(Planned(info, run_id, method, arm, args.set_name or args.pairs_stage, pair_dir, grid, guard,
                               out_path, cert_mode, input_check))
    run_dirs = [item.info.run_dir for item in planned]
    for path, name in [(item.out_path, "output") for item in planned]+[(args.long, "--long"), (args.summary, "--summary")]:
        if path is not None and any(_inside(path, run_dir) for run_dir in run_dirs):
            raise UsageError(f"{name} {path}: lies inside an evaluated run directory (hard constraint 5); use another "
                             "--out-dir")
    return planned


def _bitwise(a, b):
    a, b = np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def _run_fields(item, args):
    """The identification and run-state fields every record carries (also an error record)."""
    info = item.info
    registry = info.extra.get("structure_registry")
    seed = info.extra.get("init_field")               # --init-field (C12-4a): the key only on such a run
    return dict(schema=SCHEMA, run_id=item.run_id, set=item.set_name, split=args.split, method=item.method,
                arm=item.arm, pair=info.pair, stage=args.pairs_stage, kind=info.kind, run_dir=str(info.run_dir),
                run_status=info.run_status, field_role=info.field_role, metrics_role=info.metrics_role,
                comparable_by_default=info.field_role in COMPARABLE_ROLES, completion=info.completion,
                input_check=item.input_check, guard=item.guard, structure_registry=registry,
                development_diagnostic_only=bool(registry and registry.get("staging_unregistered")),
                **({} if seed is None else dict(init_field=seed)))


def nonfinite_metrics(scalars):
    """{measure: "inf" | "-inf" | "nan"} of the non-finite scalar metrics (metrics.json writes them as null)."""
    return {k: format_value(v) for k, v in (scalars or {}).items()
            if isinstance(v, (float, np.floating)) and not np.isfinite(v)}


def evaluate(item, args, code):
    """Phase 2 for one run: (record, official row or None)."""
    info = item.info
    record = dict(_run_fields(item, args), run_certificate=info.run_certificate,
                  evaluated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  definitions=dict(metrics=metrics.DEFINITIONS, certificate=cert_eval.DEFINITIONS), code=code,
                  field_present=field_present(info), field=None, truth=None, metrics=None, per_label=None,
                  certificate=None)
    if not record["field_present"]:
        record["note"] = (f"no field evaluated (field_role {info.field_role}, run_status {info.run_status}: run failure, "
                          "failed final certificate, stopped before native or no native result): no metrics")
        return record, None
    if info.field_role == ROLE_HANDOVER:
        record["note"] = ("handover_only: the certified native initializer, not a native formal state; reported "
                          "separately and not compared by default (review EV item 2)")
    d, u_run, field_rec = load_field(info, item.grid.shape, item.grid.spacing)
    record["field"] = field_rec
    scheme = scheme_for(item.guard, item.pair_dir)
    truth, truth_rec = load_truth(item.pair_dir, scheme, item.grid, item.guard)
    record["truth"] = truth_rec
    scalars, per_label, official = metrics.evaluate_field(d, truth, scheme, item.grid.spacing)
    record.update(metrics=scalars, per_label=per_label, nonfinite_metrics=nonfinite_metrics(scalars),
                  units=dict(spacing_xyz=list(item.grid.spacing),
                             hd95="voxel units (surface_distance spacing ones(3)); equal to mm only when every spacing "
                                  "is 1 mm", hd95_equals_mm=all(float(h) == 1. for h in item.grid.spacing)))
    if item.cert_mode is not None:
        u_cert = d_to_u(d, item.grid.spacing)
        cert = cert_eval.certify_field(u_cert, item.grid, mode=item.cert_mode, tile_size=args.cert_tile_size,
                                       brain_nodes=truth["brain"], split_cells=not args.no_cell_split)
        cert["certified_field"] = "u = h d (voxels_to_displacement of the evaluated d)"
        cert["u_equals_run_u_bitwise"] = None if u_run is None else _bitwise(u_cert, u_run)
        own = info.run_certificate
        if own is not None:
            cert["run_certificate_state_hash_matches"] = own.get("state_hash") == cert["state_hash"]
            cert["run_certificate_status_matches"] = own.get("status") == cert["status"]
        record["certificate"] = cert
    return record, official


def long_rows(record):
    """The long-CSV rows of one metrics record (per-label Dice once per label stem). Non-finite values are kept
    (write_long writes them as "inf", "-inf", "nan"); None (not computed) is left out."""
    scheme_sets = [(measure, spec["stem"]) for measure, spec in ((record.get("truth") or {}).get("macro_sets") or {}).items()]
    base = {k: record[k] for k in ("set", "split", "method", "arm", "pair", "run_id")}
    base.update(field_role=record.get("field_role") or "", run_status=record.get("run_status") or "",
                structure_registry=registry_label(record))
    rows = [dict(base, measure="field_present", value=int(bool(record["field_present"])))]
    if record.get("error"):
        rows.append(dict(base, measure="evaluation_error", value=1))
    m = record.get("metrics") or {}
    for key, value in m.items():
        if key in SKIP_LONG or value is None or isinstance(value, (list, dict)):
            continue
        rows.append(dict(base, measure=key, value=value))
    stems_done = set()
    for measure, stem in scheme_sets:
        if stem in stems_done:
            continue
        stems_done.add(stem)
        for label, value in ((record.get("per_label") or {}).get(measure+"_per_label") or {}).items():
            if value is not None:
                rows.append(dict(base, measure=f"{measure}_label_{int(label):02d}", value=value))
    c = record.get("certificate")
    if c:
        mode = c["mode"]                  # full or jacobian_only (review EV item 9: the mode is in the name)
        vals = {f"cert_{mode}_passed": int(bool(c["passed"])), f"cert_{mode}_min_jd_lower_bound": c.get("min_jd_lower_bound"),
                f"cert_{mode}_n_infeasible": c.get("n_infeasible"), f"cert_{mode}_n_uncertain": c.get("n_uncertain"),
                "cert_domain_ok": int(bool(c["domain"]["ok"])), "cert_faces_max_abs_u_mm": c.get("faces_max_abs_u_mm")}
        regions = (c.get("cells") or {}).get("regions") or {}
        for region in ("brain", "background"):
            if region in regions:
                r = regions[region]
                vals.update({f"cert_n_infeasible_{region}": r["infeasible"], f"cert_n_uncertain_{region}": r["uncertain"],
                             f"cert_min_lower_bound_{region}": r["min_lower_bound"]})
        rows += [dict(base, measure=k, value=v) for k, v in vals.items() if v is not None]
    for row in rows:
        if isinstance(row["value"], (bool, np.bool_)):
            row["value"] = int(row["value"])
    return rows


def registry_label(record):
    """The structure-registry label of a record (long CSV column structure_registry, summary group key): n/a (not a
    structure-layer run), registered, staging_unregistered or not_recorded."""
    registry = record.get("structure_registry")
    return "n/a" if registry is None else registry.get("status") or "not_recorded"


def summarize(records, officials):
    """Per (set, method, arm, field_role, structure_registry): counts, the official aggregate
    (scripts/evaluate_l2r_style.aggregate, as defined: a non-finite case value propagates), and per scalar measure the
    mean over its finite values with the number of finite (n) and non-finite (n_nonfinite, by kind) values next to it
    -- the same rows paired compares. The registration state is part of the key (review PS1_PS2 item d), so runs on
    the staging registry (未正式登记) are never averaged with registered runs."""
    groups = {}
    for record, official in zip(records, officials):
        key = (record["set"], record["method"], record["arm"], record.get("field_role") or "", registry_label(record))
        groups.setdefault(key, []).append((record, official))
    out = []
    for (set_name, method, arm, role, registry_key), items in sorted(groups.items()):
        rows = [o for _, o in items if o is not None]
        means, counts = {}, {}
        keys = sorted({k for r, _ in items for k, v in (r.get("metrics") or {}).items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in SKIP_LONG})
        for key in keys:
            vals = [float(r["metrics"][key]) for r, _ in items if r.get("metrics") and r["metrics"].get(key) is not None]
            finite = [v for v in vals if np.isfinite(v)]
            bad = [format_value(v) for v in vals if not np.isfinite(v)]
            means[key] = float(np.mean(finite)) if finite else None
            counts[key] = dict(n=len(finite), n_nonfinite=len(bad),
                               nonfinite={t: bad.count(t) for t in sorted(set(bad))})
        cert = [r["certificate"] for r, _ in items if r.get("certificate")]
        by_mode = {}
        for c in cert:
            entry = by_mode.setdefault(c["mode"], dict(passed=0, evaluated=0))
            entry["passed"] += bool(c["passed"])
            entry["evaluated"] += 1
        official_nonfinite = sum(not all(np.isfinite(o[k]) for k in ("dice", "hd95", "sdlogj")) for o in rows)
        registry_status = {}
        for r, _ in items:
            status = registry_label(r)
            registry_status[status] = registry_status.get(status, 0)+1
        out.append(dict(set=set_name, method=method, arm=arm, field_role=role, structure_registry=registry_key,
                        structure_registry_status=dict(sorted(registry_status.items())),
                        staging_unregistered_runs=sum(bool(r.get("development_diagnostic_only")) for r, _ in items),
                        comparable_by_default=role in COMPARABLE_ROLES, runs=len(items), with_field=len(rows),
                        errors=sum(bool(r.get("error")) for r, _ in items),
                        no_field=sum(not r.get("field_present") for r, _ in items),
                        certificate_by_mode=by_mode, official_aggregate=metrics.aggregate(rows) if rows else None,
                        official_cases_nonfinite=official_nonfinite, means_over_finite=means, value_counts=counts,
                        run_ids=[r["run_id"] for r, _ in items]))
    return out


def _fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}" if np.isfinite(v) else format_value(v)


def main(argv=None):
    args = parse_args(argv)
    try:
        sealed = load_sealed_lists()
        allowlist = load_dev_allowlist(args.allowlist)
        planned = plan(args, allowlist, sealed)
    except SubjectSealed as exc:
        print(f"subject_sealed: {exc}; nothing was read or written", file=sys.stderr)
        return 3
    except SealedListError as exc:
        print(f"sealed list: {exc}; nothing was read or written", file=sys.stderr)
        return 2
    except (UsageError, FieldError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}; nothing was read or written", file=sys.stderr)
        return 2
    code = code_record()
    records, officials, failed = [], [], False
    for item in planned:
        try:
            record, official = evaluate(item, args, code)
        except Exception as exc:                                  # evidence kept, never retried (constraint 4)
            failed = True
            record = dict(_run_fields(item, args), code=code, field_present=field_present(item.info), metrics=None,
                          certificate=None, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            official = None
        item.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(item.out_path, "x", encoding="utf-8") as stream:
            stream.write(dumps(record, indent=1)+"\n")
        records.append(record)
        officials.append(official)
        m, c = record.get("metrics") or {}, record.get("certificate") or {}
        staging = " [staging_unregistered]" if record.get("development_diagnostic_only") else ""
        print(f"{item.run_id[:48]:48s} [{record.get('field_role')}]{staging} seg35 {_fmt(m.get('seg35'))} seg4 {_fmt(m.get('seg4'))} "
              f"Dice35 {_fmt(m.get('dice_official'))} HD95 {_fmt(m.get('hd95'), 3)} SDlogJ {_fmt(m.get('sdlogj_official'))} "
              f"brain {_fmt(m.get('sdlogj_brain'), 3)} cert {c.get('mode')} {c.get('status')} passed {c.get('passed')}"
              + (f"  ERROR {record['error']}" if record.get("error") else ""), flush=True)
    rows = [row for record in records for row in long_rows(record)]
    write_long(args.long, rows)
    if args.summary is not None:
        with open(args.summary, "x", encoding="utf-8") as stream:
            stream.write(dumps(dict(schema="bayesreg.eval.summary.v2", groups=summarize(records, officials),
                                    definitions=metrics.DEFINITIONS, code=code), indent=1)+"\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
