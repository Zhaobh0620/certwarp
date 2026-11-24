"""Recertify the exported state of one bayesreg run without the prescreen (D53's audit; contract C05-7 item 7b
"审核停止"; ETA4 review round 3 R3-2; RECERT review). Shared by the path pilot's D53 audit and the eta sweep v4, hence
not under experiments/eta/.

    nice -n 19 env NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -B experiments/audit/recertify.py RUN_DIR --out OUT_JSON [--workers 8]

Reads, read only, RUN_DIR/status.json and the state it records as exported: its "export" record names the file --
final_level{L}.npz when the run stopped before the native level (export.level = L), else final_native.npz -- read
through bayesreg.eval.fields.read_final_native (u and its stored state_hash, grid record and status; nothing else of
the run, never pair.npz, so no label is ever read). The grid is rebuilt from that file's grid record, whose grid_id is
required and must agree. u is certified with configuration C of the main line minus the prescreen:
jd_curl.three_d.feasibility.CompiledFallbackCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=8192,
fallback_workers=N) -- the parent class of PrescreenCertificateConfig, passed explicitly to certify_map (whose own
default configuration is not this one) -- on the CPU in FP64 with outward rounding. --workers N (default 8, the
configuration's default) is the number of threads of the Bernstein fallback (recorded; it changes no decision, E50);
numba's parallel kernels take NUMBA_NUM_THREADS from the environment (use 1, and few workers, while a timed run is
going on). When the certificate does not pass, the per-cell pass of bayesreg.eval.cert_eval.cell_pass (the same
decisions, sequential, the non-prescreen branch) lists the uncertain and the infeasible ("zero") cells.

Writes one JSON (a new file outside RUN_DIR, written to a temporary file in the same directory and linked into place:
an existing file is never overwritten, a failure leaves nothing behind):
  tool, schema, script_path, script_sha256 (this file), code_sha and code_dirty (git of the repository holding this
  file), run_dir (resolved), export_file (path, file sha256, status, level: "native" or L), status_exported
  (status.json "exported"), state_hash (sha256 of the float64 C-order bytes of u: bayesreg.eval.fields.array_sha256,
  the convention of status.json "exported" and of the evaluator certificate's state_hash), stored_state_hash,
  state_hash_matches_stored, state_hash_matches_exported, certificate_state_hash (certify_map's own digest; equal to
  state_hash, else exit 8), grid (record, rebuilt grid_id), certificate_config ({"class": ..., every dataclass field}),
  certificate_version, status, reason, passed (status certified_feasible, min_jd_lower_bound > 0, no infeasible and
  no uncertain cell: "zero" and "uncertain" do not pass, C07-3), min_jd_lower_bound, min_margin_lower_bound, n_cells,
  n_cells_checked, n_infeasible ("zero"), n_uncertain ("未决"), n_fast_cells, n_prescreen_cells (0: no prescreen),
  n_fallback_cells, n_boxes, max_depth, witness_cell, witness_reference_point, cells (when not passed: the first 1000
  uncertain and infeasible cell indices, flat and (z, y, x), with their totals and the comparison with the map-level
  counts), seconds, started_utc, ended_utc (UTC with a zone), threads (fallback_workers and the environment
  variables NUMBA_NUM_THREADS and the BLAS ones).
Exit codes: 0 written (whatever the certificate's outcome), 2 usage (the output exists, lies inside RUN_DIR or its
directory does not exist -- checked before any certificate), 8 the exported state cannot be read, bound or certified
(no status.json or no exported state, a missing, unreadable or inconsistent file, a grid record without grid_id or
not agreeing, any other failure: one message, no traceback).
"""
import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from jd_curl.three_d import feasibility as F  # noqa: E402
from jd_curl.three_d.grid import make_grid_3d  # noqa: E402

from bayesreg.eval import cert_eval  # noqa: E402
from bayesreg.eval.fields import array_sha256, read_final_native  # noqa: E402

TOOL = "experiments/audit/recertify.py"
SCHEMA = "bayesreg.audit.recertify.v1"
CONFIG_CLASS = "CompiledFallbackCertificateConfig"          # configuration C without the prescreen
CONFIG_FIELDS = dict(delta=0., max_depth=48, max_boxes=255)
TILE_SIZE = 8192                                            # the evaluator's and the solver's T (C07-2); fixed
WORKERS = 8                                                 # the configuration's default fallback_workers
CELL_LIST = 1000                                            # at most this many uncertain / infeasible cells listed
RESULT_FIELDS = ("status", "reason", "min_jd_lower_bound", "min_margin_lower_bound", "n_cells_checked", "n_uncertain",
                 "n_infeasible", "n_boxes", "max_depth", "witness_cell", "witness_reference_point", "n_fast_cells",
                 "n_fallback_cells", "n_prescreen_cells", "certificate_version")
THREAD_VARIABLES = ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


class AuditError(Exception):
    """The exported state cannot be read, bound or certified (exit 8)."""


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    """numpy scalars and tuples -> JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    return value


def config_record(config):
    """{"class": name, every dataclass field}."""
    return dict({"class": type(config).__name__}, **{f.name: plain(getattr(config, f.name))
                                                      for f in dataclasses.fields(config)})


def recertification_config(workers=WORKERS):
    """CompiledFallbackCertificateConfig(delta=0, max_depth=48, max_boxes=255, tile_size=8192, fallback_workers):
    configuration C without the prescreen, checked to be the parent class of the prescreen configuration."""
    cls = getattr(F, CONFIG_CLASS)
    if F.PrescreenCertificateConfig.__mro__[1] is not cls:
        raise AuditError(f"{CONFIG_CLASS} is not the parent class of PrescreenCertificateConfig in this jd_curl")
    config = cls(tile_size=TILE_SIZE, fallback_workers=workers, **CONFIG_FIELDS)
    if isinstance(config, F.PrescreenCertificateConfig):
        raise AuditError("the recertification configuration must not carry the prescreen")
    return config


def code_record():
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=False)
    return (out.stdout.strip() or None), (bool(dirty.stdout.strip()) if dirty.returncode == 0 else None)


def exported_file(run_dir, status):
    """The file of the run's exported state (run_pair: export.level = L for a run stopped before the native level ->
    final_level{L}.npz; else final_native.npz) and its level ("native" or L)."""
    export = status.get("export") or {}
    level = export.get("level") if isinstance(export, dict) else None
    if isinstance(level, int) and not isinstance(level, bool):
        return Path(run_dir)/f"final_level{level}.npz", level
    return Path(run_dir)/"final_native.npz", "native"


def cell_lists(u, grid, config, result):
    """The uncertain and infeasible cells of a state that did not pass (cert_eval.cell_pass: the decisions of
    _certify_map_parallel, non-prescreen branch), at most CELL_LIST of each, with totals and the map comparison."""
    tick = time.perf_counter()
    code, _ = cert_eval.cell_pass(u, grid, config)
    shape = tuple(n-1 for n in grid.shape)
    out = {}
    for name, mask in (("uncertain", code >= cert_eval.UNCERTAIN), ("infeasible", code == cert_eval.INFEASIBLE)):
        ids = np.flatnonzero(mask)
        out[name] = dict(total=int(ids.size), listed=int(min(ids.size, CELL_LIST)),
                         cells=[dict(flat=int(i), zyx=[int(v) for v in np.unravel_index(int(i), shape)])
                                for i in ids[:CELL_LIST]])
    out["matches_map"] = (out["uncertain"]["total"] == result.n_uncertain
                          and out["infeasible"]["total"] == result.n_infeasible)
    out["seconds"] = time.perf_counter()-tick
    out["source"] = "bayesreg.eval.cert_eval.cell_pass (sequential, the same decisions; non-prescreen branch)"
    return out


def recertify(run_dir, *, workers=WORKERS):
    """The record of the module docstring for run_dir (nothing is written). Any failure to read, bind or certify is
    an AuditError."""
    started = utc()
    run_dir = Path(run_dir)
    try:
        status = json.loads((run_dir/"status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuditError(f"{run_dir}/status.json: {exc}") from None
    if not isinstance(status, dict) or not status.get("exported"):
        raise AuditError(f"{run_dir}/status.json records no exported state: nothing to recertify (a no-export "
                         "confirmation is the only way on)")
    path, level = exported_file(run_dir, status)
    if not path.is_file():
        raise AuditError(f"{path} missing (the exported state of status.json)")
    try:
        u, record = read_final_native(path)
        grid_doc = record.get("grid")
        if not isinstance(grid_doc, dict) or not grid_doc.get("grid_id"):
            raise AuditError(f"{path}: no grid record with a grid_id (the grid cannot be bound)")
        (nz, ny, nx), (hx, hy, hz), (x0, y0, z0) = (grid_doc["shape_zyx"], grid_doc["spacing_xyz"],
                                                    grid_doc["origin_xyz"])
        grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=hx, hy=hy, hz=hz, x0=x0, y0=y0, z0=z0, unit=grid_doc["unit"])
        if grid.grid_id != grid_doc["grid_id"]:
            raise AuditError(f"{path}: rebuilt grid_id {grid.grid_id} differs from the file's {grid_doc['grid_id']}")
        state_hash = array_sha256(u)
        config = recertification_config(workers)
        tick = time.perf_counter()
        result = F.certify_map(u, grid, config=config)
        seconds = time.perf_counter()-tick
    except AuditError:
        raise
    except Exception as exc:          # noqa: BLE001 -- every read / validation failure is one exit-8 message
        raise AuditError(f"{path}: {type(exc).__name__}: {exc}") from None
    if result.state_hash != state_hash:
        raise AuditError(f"certify_map's digest {result.state_hash} differs from sha256(u) {state_hash}")
    out = {key: plain(getattr(result, key, None)) for key in RESULT_FIELDS}
    if out["n_prescreen_cells"] is None:     # no prescreen in this configuration: no cell proven by it
        out["n_prescreen_cells"] = 0
    lowest = out["min_jd_lower_bound"]
    passed = bool(result.status == "certified_feasible" and lowest is not None and lowest > 0
                  and result.n_infeasible == 0 and result.n_uncertain == 0)
    cells = None
    if not passed and result.reason in ("JACOBIAN_VIOLATION", "subdivision_budget", "arithmetic_uncertain"):
        try:
            cells = cell_lists(u, grid, config, result)
        except Exception as exc:      # noqa: BLE001 -- the list is a record; its failure is recorded, not fatal
            cells = dict(error=f"{type(exc).__name__}: {exc}")
    code_sha, code_dirty = code_record()
    script = Path(__file__).resolve()
    return dict(
        tool=TOOL, schema=SCHEMA, script_path=str(script), script_sha256=file_sha256(script), code_sha=code_sha,
        code_dirty=code_dirty, run_dir=str(run_dir.resolve()),
        export_file=dict(path=str(path.resolve()), file_sha256=record["file_sha256"], status=record.get("status"),
                         level=level),
        status_exported=status.get("exported"), state_hash=state_hash, stored_state_hash=record["stored_state_hash"],
        state_hash_matches_stored=record["state_hash_matches"],
        state_hash_matches_exported=status.get("exported") == state_hash, certificate_state_hash=result.state_hash,
        grid=dict(record=grid_doc, rebuilt_grid_id=grid.grid_id), certificate_config=config_record(config),
        passed=passed, n_cells=int(grid.n_cells), **out, cells=cells, seconds=seconds, started_utc=started,
        ended_utc=utc(), threads=dict({k: os.environ.get(k) for k in THREAD_VARIABLES}, fallback_workers=workers),
        note=("D53 审核：配置 C 去掉预筛重新认证导出状态（CPU FP64，外向舍入）；只读运行目录的 status.json 与导出文件，"
              "不读 pair.npz"))


def write_new(out, document, run_dir):
    """The JSON to a new file outside run_dir: serialised first, written to a temporary file in the same directory,
    then linked into place (os.link fails when the target exists); the temporary file is always removed."""
    out, run_dir = Path(out).resolve(), Path(run_dir).resolve()
    if out == run_dir or run_dir in out.parents:
        raise ValueError(f"--out {out} lies inside the run directory {run_dir} (read only)")
    text = json.dumps(document, indent=1, ensure_ascii=False, allow_nan=False)+"\n"
    handle, temporary = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, out)
    finally:
        os.unlink(temporary)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog=f"python -B {TOOL}", description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path, help="a new JSON file outside RUN_DIR")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"threads of the Bernstein fallback (default {WORKERS}; recorded, no decision changes; "
                        "fewer while a timed run is going on)")
    args = parser.parse_args(argv)
    out, run_dir = args.out.resolve(), args.run_dir.resolve()
    if out == run_dir or run_dir in out.parents:
        parser.error(f"--out {out} lies inside the run directory {run_dir} (read only)")
    if out.exists():
        parser.error(f"--out {out} exists (never overwritten)")
    if not out.parent.is_dir():
        parser.error(f"--out {out}: the directory {out.parent} does not exist")
    if args.workers < 1:
        parser.error("--workers: at least 1")
    try:
        document = recertify(args.run_dir, workers=args.workers)
        write_new(out, document, run_dir)
    except AuditError as exc:
        print(f"recertify: {exc}", file=sys.stderr)
        return 8
    except (OSError, ValueError, TypeError) as exc:
        print(f"recertify: writing {out}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 8
    print(f"recertify: {out}: passed {document['passed']}, status {document['status']} ({document['reason']}), min J "
          f"lower bound {document['min_jd_lower_bound']}, uncertain {document['n_uncertain']}, infeasible "
          f"{document['n_infeasible']}, {document['seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
