"""Displacement fields for evaluation (contract C16; project rules "评测约定": metrics only from the field).

The evaluated object is always d: (3, nz, ny, nx) float64, voxel units, pull-back
warped(p) = moving(p + d(p)) with p the (z, y, x) voxel index (Learn2Reg evaluation.py semantics).
The mm field of the certificate and of the cell-centre Jacobian is u = h d: u[..., 2 - k] = d[k] h_(z,y,x)[k]
((nz, ny, nx, 3), components xyz).  Conversions (bayesreg.solve.export, C16):
    u_to_d: d[k] = u[..., 2 - k] / h_(z,y,x)[k]   (= scripts/evaluate_l2r_style.disp_voxels bit for bit)
    d_to_u: u[..., 2 - k] = d[k] * h_(z,y,x)[k]
At h = 1 mm both are exact (u -> d -> u bit for bit); otherwise |u - h d| <= 1 ulp of u (checked, recorded).

Run kinds (detect_kind):
  bayesreg  run directory of bayesreg.cli.run_pair: config.json (args.pair = "<stage>/<pair>"), status.json,
            final_native.npz (u, state_hash, grid, status), disp.npz (d, meta), cert.json. Evaluated field:
            disp.npz:d; it must equal u_to_d(final_native u) bit for bit (C16 gate, recorded). A run without
            disp.npz (failed final certificate, run failure, stopped before native) has no field: no metrics.
  configC   seq_vp_3d / configuration C run directory (VAL10, D0): protocol.json (task.name, task.pair),
            <task>/final_native.npz:u (mm, (z, y, x, 3), xyz), <task>/final_native.json (its certificate),
            inputs/<pair>/manifest.json (its grid). d = u_to_d(u).
  disp      any other directory holding disp.npz:d (baselines, float32 or float64; stored as given, read as
            float64 exactly); the pair is named on the command line.
Only the members u / d (and the small metadata members state_hash, grid, status, meta) are read from the npz
files; nothing here reads labels.

Completion (review EV item 1; a run that has not ended is never evaluated: describe_run raises FieldError, which
the cli turns into "nothing read or written"):
  bayesreg  status.json with an integer exit_code (run_pair writes it last).
  configC   <run>/exit_code (the supervisor writes it when the unit ends).  exit 0 additionally needs
            <task>/summary.json, <run>/completion.json and <task>/worker_completion.json with complete = true
            (the D0 classifier's record check); missing -> refused.  A nonzero exit_code is a finished failed run:
            described (run_status "exit_<code>") but its field is not evaluated (field_present false).
Field role (review EV item 2; the source of the evaluated field, recorded with every metric):
  native_formal     a formal state of the native level: configuration C metrics_role formal_native_outer_commit;
                    bayesreg with at least one formal commit in the native level's rounds (hyper.json rounds)
  handover_only     the certified native initializer, no native commit: configuration C metrics_role
                    certified_native_initializer_NOT_native_outer_commit; bayesreg with zero native commits
  no_native_result  no native field (configuration C no_native_result or a failed run; bayesreg without disp.npz)
  unknown           the run does not say (no / unrecognized metrics_role, no hyper.json rounds)
  disp_only         a disp.npz directory (a baseline's exported field; no formal-state notion)
  oracle_seeded     bayesreg started by --init-field from a seed that used truth labels (status.json init_field.oracle
                    true; contract C12-4a item 3): never a method result, whatever the commits
COMPARABLE_ROLES (native_formal, disp_only) enter paired comparisons by default; handover_only and oracle_seeded are
reported separately and enter only on request (bayesreg.eval.paired --roles).  A run with --init-field carries the seed's
summary in RunInfo.extra["init_field"] (and metrics.json init_field; absent otherwise); so does a disp.npz directory whose meta carries
it (the entry and the checkpoints of such a run, contract C12-4a item 8), with field_role oracle_seeded when the seed is an oracle one.
Structure-layer registry (review PS item 2; label only, the metrics do not change): a bayesreg run with
channel_source structure carries the registration state of its segmenter registry in RunInfo.extra
["structure_registry"] -- status.json structure_registry (run_pair since PS1), else provenance.json
channels.structure.registry, else status None (not recorded); staging_unregistered marks a development
diagnostic (未正式登记); git_tracked / git_clean: the registry file's git state as the run recorded it (review
PS1_PS2 item b; None when not recorded).
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from jd_curl.three_d.nifti import file_sha256

KINDS = ("bayesreg", "configC", "disp")
STAGING_UNREGISTERED = "staging_unregistered"     # = bayesreg.seg.structure_layer.STAGING
ROLE_NATIVE = "native_formal"
ROLE_HANDOVER = "handover_only"
ROLE_NONE = "no_native_result"
ROLE_UNKNOWN = "unknown"
ROLE_DISP = "disp_only"
ROLE_ORACLE_SEEDED = "oracle_seeded"              # --init-field from a truth-label run (C12-4a item 3)
FIELD_ROLES = (ROLE_NATIVE, ROLE_HANDOVER, ROLE_NONE, ROLE_UNKNOWN, ROLE_DISP, ROLE_ORACLE_SEEDED)
COMPARABLE_ROLES = (ROLE_NATIVE, ROLE_DISP)
CONFIGC_METRICS_ROLES = {"formal_native_outer_commit": ROLE_NATIVE,
                         "certified_native_initializer_NOT_native_outer_commit": ROLE_HANDOVER,
                         "no_native_result": ROLE_NONE}


class FieldError(ValueError):
    """A run directory or field file does not satisfy the evaluation conventions."""


def array_sha256(array):
    """sha256 of the float64 C-order bytes (the certificate's state_hash convention, C07-7)."""
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes(order="C")).hexdigest()


def _spacing(spacing_xyz):
    spacing = tuple(float(h) for h in spacing_xyz)
    if len(spacing) != 3 or not all(np.isfinite(h) and h > 0 for h in spacing):
        raise FieldError(f"spacing_xyz: three positive finite numbers required, got {spacing_xyz!r}")
    return spacing


def u_to_d(u, spacing_xyz):
    """(nz, ny, nx, 3) mm, components xyz -> (3, nz, ny, nx) voxels, components zyx."""
    hx, hy, hz = _spacing(spacing_xyz)
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise FieldError(f"u: (nz, ny, nx, 3) required, got {u.shape}")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return np.stack([u[..., 2]/hz, u[..., 1]/hy, u[..., 0]/hx])


def d_to_u(d, spacing_xyz):
    """Inverse of u_to_d: u[..., 2 - k] = d[k] h_(z,y,x)[k]."""
    h_zyx = _spacing(spacing_xyz)[::-1]
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 4 or d.shape[0] != 3:
        raise FieldError(f"d: (3, nz, ny, nx) required, got {d.shape}")
    u = np.empty(d.shape[1:]+(3,))
    with np.errstate(over="raise", invalid="raise"):
        for k in range(3):
            u[..., 2-k] = d[k]*h_zyx[k]
    return u


def _bitwise_equal(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(np.ascontiguousarray(a).view(np.uint64),
                                                 np.ascontiguousarray(b).view(np.uint64))


def roundtrip_check(u, d, spacing_xyz):
    """C16 gate between a stored u and a d: d == u_to_d(u) bit for bit; at h = 1, d_to_u(d) == u bit for bit,
    otherwise |u - d_to_u(d)| <= 1 ulp of u. Returns the record (passed, and each part)."""
    spacing = _spacing(spacing_xyz)
    unit = all(h == 1. for h in spacing)
    d_equal = _bitwise_equal(u_to_d(u, spacing), d)
    back = d_to_u(d, spacing)
    u_equal = _bitwise_equal(back, u)
    with np.errstate(over="ignore", invalid="ignore"):
        excess = np.abs(np.asarray(u, dtype=np.float64)-back)-np.spacing(np.abs(np.asarray(u, dtype=np.float64)))
    within_ulp = bool(np.all(excess <= 0))
    passed = d_equal and (u_equal if unit else within_ulp)
    return dict(unit_spacing=unit, d_equals_u_to_d_bitwise=d_equal, u_equals_d_to_u_bitwise=u_equal,
                u_within_1ulp=within_ulp, passed=bool(passed))


def _read_npz_members(path, required, optional=()):
    """Only the named members of an npz archive (no pickle, no iteration over the archive)."""
    path = Path(path)
    if not path.is_file():
        raise FieldError(f"{path}: missing")
    out = {}
    with np.load(path, allow_pickle=False) as npz:
        for key in required:
            if key not in npz:
                raise FieldError(f"{path.name}: required member {key!r} missing")
            out[key] = np.array(npz[key])
        for key in optional:
            if key in npz:
                out[key] = np.array(npz[key])
    return out


def _text(value):
    return str(value.item()) if isinstance(value, np.ndarray) and value.shape == () else None


def read_disp(path, shape_zyx=None):
    """disp.npz -> (d float64 (3, nz, ny, nx), record). Stored dtype float32 or float64 (converted exactly);
    every value must be finite."""
    members = _read_npz_members(path, ("d",), ("meta",))
    raw = members["d"]
    if raw.dtype.kind != "f" or raw.dtype.itemsize not in (4, 8):
        raise FieldError(f"disp.npz:d: float32 or float64 required, got {raw.dtype}")
    if raw.ndim != 4 or raw.shape[0] != 3:
        raise FieldError(f"disp.npz:d: shape (3, nz, ny, nx) required, got {raw.shape}")
    if shape_zyx is not None and tuple(raw.shape[1:]) != tuple(shape_zyx):
        raise FieldError(f"disp.npz:d: grid {raw.shape[1:]} differs from the pair's {tuple(shape_zyx)}")
    d = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(d)):
        raise FieldError("disp.npz:d: non-finite values")
    meta = None
    if "meta" in members:
        text = _text(members["meta"])
        try:
            meta = json.loads(text) if text is not None else None
        except json.JSONDecodeError:
            meta = dict(unparsed=text)
    return d, dict(path=str(path), file_sha256=file_sha256(path), stored_dtype=str(raw.dtype), d_sha256=array_sha256(d),
                   meta=meta)


def read_final_native(path, shape_zyx=None):
    """final_native.npz -> (u float64 (nz, ny, nx, 3) mm, record). bayesreg files also carry state_hash (checked
    against the array), grid (JSON) and status; seq_vp_3d files carry u only (their certificate is in the JSON
    next to them)."""
    members = _read_npz_members(path, ("u",), ("state_hash", "grid", "status"))
    raw = members["u"]
    if raw.dtype != np.float64 or raw.ndim != 4 or raw.shape[-1] != 3:
        raise FieldError(f"final_native.npz:u: float64 (nz, ny, nx, 3) required, got {raw.dtype} {raw.shape}")
    if shape_zyx is not None and tuple(raw.shape[:3]) != tuple(shape_zyx):
        raise FieldError(f"final_native.npz:u: grid {raw.shape[:3]} differs from the pair's {tuple(shape_zyx)}")
    u = np.ascontiguousarray(raw)
    if not np.all(np.isfinite(u)):
        raise FieldError("final_native.npz:u: non-finite values")
    digest = array_sha256(u)
    stored = _text(members.get("state_hash"))
    grid = _text(members.get("grid"))
    record = dict(path=str(path), file_sha256=file_sha256(path), u_sha256=digest, stored_state_hash=stored,
                  state_hash_matches=None if stored is None else stored == digest,
                  status=_text(members.get("status")), grid=json.loads(grid) if grid else None)
    if stored is not None and stored != digest:
        raise FieldError(f"{path}: stored state_hash differs from sha256 of u (C07-7)")
    return u, record


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class RunInfo:
    """What a run directory says about itself (small files only; no arrays)."""
    kind: str
    run_dir: Path
    pair: str | None
    stage: str | None
    files: dict
    run_status: str | None = None
    run_certificate: dict | None = None
    arm_hint: str | None = None
    extra: dict = field(default_factory=dict)
    field_role: str = ROLE_UNKNOWN
    metrics_role: str | None = None          # configuration C summary.json:metrics_role as written
    completion: dict = field(default_factory=dict)
    run_id_hint: str | None = None           # <run>/run_id when the run directory has one (D0)


def _exit_code(value, where):
    if isinstance(value, bool) or not isinstance(value, int):
        raise FieldError(f"{where}: integer exit_code required (run not finished or record broken); not evaluated")
    return value


def _bayesreg_native_commits(run_dir, status):
    """Formal commits in the native level's rounds (hyper.json), or None when the run does not record them."""
    hyper = _read_json(run_dir/"hyper.json")
    rounds = (hyper or {}).get("rounds")
    if not status.get("native") or not isinstance(rounds, list) or not rounds:
        return None
    levels = [r.get("level") for r in rounds if isinstance(r, dict) and isinstance(r.get("level"), int)]
    if not levels:
        return None
    native = max(levels)
    return sum(int(r.get("commits") or 0) for r in rounds if isinstance(r, dict) and r.get("level") == native)


def _bayesreg_structure_registry(run_dir, status, channel_source):
    """Registration state of the structure layer's segmenter registry (review PS item 2), None unless channel_source
    structure: status.json structure_registry, else provenance.json channels.structure.registry."""
    if channel_source != "structure":
        return None
    record, source = status.get("structure_registry"), "status.json:structure_registry"
    if not isinstance(record, dict):
        provenance = _read_json(run_dir/"provenance.json") or {}
        record = ((provenance.get("channels") or {}).get("structure") or {}).get("registry")
        source = "provenance.json:channels.structure.registry"
    if not isinstance(record, dict) or not isinstance(record.get("status"), str):
        return dict(status=None, staging_unregistered=None, source=None, note="registration state not recorded")
    git = record.get("git") if isinstance(record.get("git"), dict) else {}
    return dict(status=record["status"], staging_unregistered=record["status"] == STAGING_UNREGISTERED, source=source,
                path=record.get("path"), git_tracked=git.get("tracked"), git_clean=git.get("clean"))


def _describe_bayesreg(run_dir):
    config = _read_json(run_dir/"config.json") or {}
    args = config.get("args") or {}
    if "pair" not in args:
        raise FieldError(f"{run_dir}/config.json: args.pair missing")
    stage, own = _split_stage_pair(args["pair"])
    status = _read_json(run_dir/"status.json")
    if status is None:
        raise FieldError(f"{run_dir}: status.json missing (run not finished); not evaluated")
    exit_code = _exit_code(status.get("exit_code"), f"{run_dir}/status.json")
    cert = _read_json(run_dir/"cert.json")
    files = {name: run_dir/name for name in ("final_native.npz", "disp.npz", "cert.json", "status.json", "config.json")}
    commits = _bayesreg_native_commits(run_dir, status)
    if not (files["disp.npz"].is_file() and files["final_native.npz"].is_file()):
        role = ROLE_NONE
    elif commits is None:
        role = ROLE_UNKNOWN
    else:
        role = ROLE_NATIVE if commits > 0 else ROLE_HANDOVER
    seed = status.get("init_field")                   # --init-field (C12-4a); absent on every other run
    if isinstance(seed, dict) and seed.get("oracle") is True and role != ROLE_NONE:
        role = ROLE_ORACLE_SEEDED
    data = _read_json(run_dir/"data.txt") if (run_dir/"data.txt").is_file() else None
    pair_prov = (data or {}).get("pair_provenance") or {}
    info = RunInfo("bayesreg", run_dir, own, stage, files, run_status=status.get("status"),
                   run_certificate=None if cert is None else {k: cert.get(k) for k in (
                       "status", "reason", "passed", "min_jd_lower_bound", "n_infeasible", "n_uncertain",
                       "state_hash", "certificate_version")},
                   arm_hint=args.get("channel_source"), field_role=role,
                   completion=dict(marker="status.json", exit_code=exit_code, native=status.get("native"),
                                   native_commits=commits),
                   run_id_hint=_run_id_file(run_dir))
    info.extra["inputs"] = dict(pair_sha256=pair_prov.get("pair_sha256"), fixed_subject=pair_prov.get("fixed_subject"),
                                moving_subject=pair_prov.get("moving_subject"), stage=stage,
                                source="data.txt:pair_provenance" if pair_prov else None)
    info.extra["structure_registry"] = _bayesreg_structure_registry(run_dir, status, args.get("channel_source"))
    if seed is not None:
        info.extra["init_field"] = seed
    return info


def _run_id_file(run_dir):
    path = Path(run_dir)/"run_id"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def _describe_configc(run_dir):
    protocol = _read_json(run_dir/"protocol.json") or {}
    task = protocol.get("task") or {}
    if "name" not in task or "pair" not in task:
        raise FieldError(f"{run_dir}/protocol.json: task.name and task.pair required")
    task_dir = run_dir/task["name"]
    exit_path = run_dir/"exit_code"
    if not exit_path.is_file():
        raise FieldError(f"{run_dir}: no exit_code (run not finished); not evaluated")
    text = exit_path.read_text(encoding="utf-8").strip()
    try:
        exit_code = int(text)
    except ValueError:
        raise FieldError(f"{run_dir}/exit_code: {text!r} is not an integer; not evaluated") from None
    summary = _read_json(task_dir/"summary.json")
    worker = _read_json(task_dir/"worker_completion.json")
    completion = dict(marker="exit_code", exit_code=exit_code, summary_json=summary is not None,
                      completion_json=(run_dir/"completion.json").is_file(),
                      worker_complete=None if worker is None else worker.get("complete"))
    if exit_code == 0 and not (summary is not None and completion["completion_json"] and worker is not None
                               and worker.get("complete") is True):
        raise FieldError(f"{run_dir}: exit 0 without summary.json / completion.json / worker_completion.json "
                         "(complete = true): records incomplete; not evaluated")
    metrics_role = None if summary is None else summary.get("metrics_role")
    if exit_code != 0:
        role, run_status = ROLE_NONE, f"exit_{exit_code}"
    else:
        role, run_status = CONFIGC_METRICS_ROLES.get(metrics_role, ROLE_UNKNOWN), summary.get("status")
    final = _read_json(task_dir/"final_native.json") or {}
    cert = final.get("certificate") or None
    files = {"final_native.npz": task_dir/"final_native.npz", "final_native.json": task_dir/"final_native.json",
             "manifest.json": run_dir/"inputs"/task["pair"]/"manifest.json"}
    info = RunInfo("configC", run_dir, task["pair"], None, files, run_status=run_status,
                   run_certificate=None if cert is None else {k: cert.get(k) for k in (
                       "status", "min_jd_lower_bound", "n_infeasible", "n_uncertain", "state_hash",
                       "certificate_version")},
                   field_role=role, metrics_role=metrics_role, completion=completion,
                   run_id_hint=_run_id_file(run_dir))
    inputs = protocol.get("input_sha256") or {}
    info.extra["inputs"] = dict(pair_sha256=inputs.get("pair.npz"), stage=protocol.get("pair_dataset"),
                                protocol_stage=protocol.get("stage"),
                                source="protocol.json:input_sha256" if inputs else None)
    return info


def detect_kind(run_dir):
    run_dir = Path(run_dir)
    if (run_dir/"protocol.json").is_file():
        return "configC"
    if (run_dir/"config.json").is_file() and (run_dir/"cmd.txt").is_file():
        return "bayesreg"
    if (run_dir/"disp.npz").is_file():
        return "disp"
    raise FieldError(f"{run_dir}: not a bayesreg, configuration C or disp.npz run directory")


def _split_stage_pair(text):
    parts = Path(str(text)).parts
    if len(parts) != 2:
        raise FieldError(f"pair {text!r}: '<stage>/<pair>' required")
    return parts[0], parts[1]


def _disp_seed(path):
    """The seed tag in a disp.npz meta, or None (no file, no meta, no tag; read_disp reports a bad file when the field is loaded)."""
    try:
        with np.load(path, allow_pickle=False) as npz:
            if "meta" not in npz.files:
                return None
            meta = json.loads(_text(npz["meta"]) or "{}")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    return meta.get("init_field") if isinstance(meta, dict) else None


def describe_run(run_dir, *, pair=None, kind=None):
    """RunInfo of one run directory; `pair` names the pair of a disp-kind run (and must agree with a run's own)."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FieldError(f"{run_dir}: not a directory")
    kind = detect_kind(run_dir) if kind is None else kind
    if kind not in KINDS:
        raise FieldError(f"kind: one of {KINDS}")
    if kind == "bayesreg":
        info = _describe_bayesreg(run_dir)
    elif kind == "configC":
        info = _describe_configc(run_dir)
    else:
        files = {"disp.npz": run_dir/"disp.npz"}
        seed = _disp_seed(run_dir/"disp.npz")          # --init-field (C12-4a item 8); absent on every other disp.npz
        oracle = isinstance(seed, dict) and seed.get("oracle") is True
        info = RunInfo(kind, run_dir, None, None, files, field_role=ROLE_ORACLE_SEEDED if oracle else ROLE_DISP,
                       completion=dict(marker="disp.npz (a baseline's exported field; no run record)"))
        if seed is not None:
            info.extra["init_field"] = seed
    if pair is not None:
        if info.pair is not None and info.pair != pair:
            raise FieldError(f"{run_dir}: --pair {pair!r} differs from the run's own pair {info.pair!r}")
        info.pair = pair
    if info.pair is None:
        raise FieldError(f"{run_dir}: the pair of a disp.npz directory must be given (--pair)")
    return info


def load_field(info, shape_zyx, spacing_xyz):
    """(d, u_run or None, record) of a described run on the pair's grid. d is the evaluated field; u_run the
    stored mm field when the run has one (bayesreg, configC). Raises FieldError on any convention violation."""
    spacing = _spacing(spacing_xyz)
    record = dict(kind=info.kind)
    if info.kind == "disp":
        d, rec = read_disp(info.files["disp.npz"], shape_zyx)
        record.update(source="disp.npz", disp=rec)
        return d, None, record
    u, rec_u = read_final_native(info.files["final_native.npz"], shape_zyx)
    record["final_native"] = rec_u
    if info.kind == "configC":
        manifest = _read_json(info.files["manifest.json"])
        if manifest is not None:
            own = tuple(float(h) for h in manifest["spacing_xyz"])
            if own != spacing or tuple(manifest["shape_zyx"]) != tuple(shape_zyx):
                raise FieldError(f"{info.run_dir}: run grid {manifest['shape_zyx']} / {own} differs from the pair's")
        d = u_to_d(u, spacing)
        record.update(source="final_native.npz:u (mm) -> d", d_sha256=array_sha256(d),
                      roundtrip=roundtrip_check(u, d, spacing))
        return d, u, record
    grid = rec_u.get("grid")
    if grid is not None and (tuple(float(h) for h in grid["spacing_xyz"]) != spacing
                             or tuple(grid["shape_zyx"]) != tuple(shape_zyx)):
        raise FieldError(f"{info.run_dir}: final_native grid differs from the pair's")
    d, rec_d = read_disp(info.files["disp.npz"], shape_zyx)
    record.update(source="disp.npz", disp=rec_d, roundtrip=roundtrip_check(u, d, spacing))
    if not record["roundtrip"]["passed"]:
        raise FieldError(f"{info.run_dir}: disp.npz:d is not the exported final_native u (C16 round trip failed)")
    return d, u, record


def field_present(info):
    """Whether the run exported a field to evaluate (a bayesreg run without disp.npz did not; a configuration C run
    that ended with a nonzero exit code or metrics_role no_native_result is not evaluated)."""
    if info.kind == "bayesreg":
        return info.files["disp.npz"].is_file() and info.files["final_native.npz"].is_file()
    if info.kind == "configC":
        return info.field_role != ROLE_NONE and info.files["final_native.npz"].is_file()
    return info.files["disp.npz"].is_file()


__all__ = ["COMPARABLE_ROLES", "CONFIGC_METRICS_ROLES", "FIELD_ROLES", "FieldError", "KINDS", "ROLE_DISP",
           "ROLE_HANDOVER", "ROLE_NATIVE", "ROLE_NONE", "ROLE_ORACLE_SEEDED", "ROLE_UNKNOWN", "RunInfo", "STAGING_UNREGISTERED",
           "array_sha256", "d_to_u",
           "describe_run", "detect_kind", "field_present", "load_field", "read_disp", "read_final_native",
           "roundtrip_check", "u_to_d"]
