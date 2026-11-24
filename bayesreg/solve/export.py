"""C16 output files of one run directory.

- final_native.npz: `u` the native level's last formal commit, (nz, ny, nx, 3) float64 mm, components
  xyz, six faces zero, exactly the certified array; `state_hash` = sha256(u.tobytes(order="C")) (the
  digest of certify_map, C07-7); grid metadata; `status`.
- disp.npz: `d` (3, nz, ny, nx) float64 in voxel units, pull-back warped(p) = moving(p + d(p)) with p the
  (z, y, x) index, d[k] = u[..., 2 - k] / h_{(z, y, x)[k]} (C16 bench convention); `meta` a JSON string
  with both SHA, grid and convention.  h = 1 mm: u -> d -> u bit for bit; otherwise |u - h d| <= 1 ulp.
- cert.json: a fresh CPU FP64 certificate of the exported u (status, reason, bounds, counts, witness,
  certificate_version, config: delta hex, max_depth, max_boxes, tile_size), state_hash, and whether it
  equals the solver's last certificate.
- hyper.json, provenance.json, timing.json: dictionaries assembled by the caller; log.jsonl is written
  row by row during the run (`JsonlLog`).
Nothing is overwritten: every writer refuses an existing file (hard constraint 4).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from jd_curl.validation import InputValidationError, as_float64_array
from jd_curl.three_d.validation import zero_boundary

from .backend import certificate_record

DISP_CONVENTION = ("pull-back: warped(p) = moving(p + d(p)), p = (z, y, x) voxel index; d[k] = u[..., 2 - k] / "
                   "h_(z,y,x)[k]; same semantics as Learn2Reg evaluation.py")


def _sha(array):
    from .stopping import c_order_bytes
    return hashlib.sha256(c_order_bytes(array)).hexdigest()           # = sha256(tobytes(order="C")), no copy (MEM)


def displacement_to_voxels(u, grid):
    """(nz, ny, nx, 3) mm, components xyz -> d (3, nz, ny, nx) voxels, components zyx."""
    u = as_float64_array(u, "u", grid.shape+(3,))
    h_zyx = grid.spacing[::-1]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return np.stack([u[..., 2-k]/h_zyx[k] for k in range(3)])


def voxels_to_displacement(d, grid):
    """Inverse of displacement_to_voxels: u[..., 2 - k] = d[k] h_(z,y,x)[k]."""
    d = as_float64_array(d, "d", (3,)+grid.shape)
    h_zyx = grid.spacing[::-1]
    u = np.empty(grid.shape+(3,))
    with np.errstate(over="raise", invalid="raise"):
        for k in range(3):
            u[..., 2-k] = d[k]*h_zyx[k]
    return u


def grid_record(grid):
    return dict(shape_zyx=list(grid.shape), spacing_xyz=list(grid.spacing),
                origin_xyz=[float(axis[0]) for axis in grid.axes], unit=grid.unit, grid_id=grid.grid_id)


def _json_default(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _clean(value):
    """Replace non-finite floats by None (strict JSON)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.longdouble):
        return float(value)
    return value


def dumps(document, **kwargs):
    return json.dumps(_clean(document), default=_json_default, allow_nan=False, ensure_ascii=False, **kwargs)


def write_json(path, document):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(dumps(document, indent=1)+"\n", encoding="utf-8")
    return path


class JsonlLog:
    """log.jsonl: one JSON object per line, appended and flushed as the run goes (C16)."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self._stream = self.path.open("w", encoding="utf-8")
        self.rows = 0

    def __call__(self, row):
        self._stream.write(dumps(row)+"\n")
        self._stream.flush()
        self.rows += 1

    def close(self):
        self._stream.close()


def write_final_state(path, u, grid, *, status, init_field=None):
    """final_native.npz: the certified u as is (float64), its state_hash, grid metadata and status; `init_field` (C16 WST
    item; None: no such member) the seed tag of --init-field as a JSON string."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    u = np.asarray(u)
    if u.dtype != np.float64 or u.shape != grid.shape+(3,) or not zero_boundary(u):
        raise InputValidationError("final state: float64 (nz, ny, nx, 3) with six zero faces required")
    digest = _sha(u)
    extra = {} if init_field is None else dict(init_field=np.array(json.dumps(init_field)))
    np.savez(path, u=u, state_hash=np.array(digest), grid=np.array(json.dumps(grid_record(grid))),
             status=np.array(status), **extra)
    return digest


def write_disp(path, u, grid, *, state_hash, init_field=None):
    """disp.npz: key d (3, nz, ny, nx) voxels plus meta (JSON string); `init_field` (C16 WST item; None: no such key) the seed
    tag of --init-field in meta."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    d = displacement_to_voxels(u, grid)
    meta = dict(convention=DISP_CONVENTION, units="voxels", dtype="float64", layout="(3, nz, ny, nx)",
                grid=grid_record(grid), u_state_hash=state_hash, d_sha256=_sha(d),
                exact_inverse=bool(all(h == 1. for h in grid.spacing)))
    if init_field is not None:
        meta["init_field"] = init_field
    np.savez(path, d=d, meta=np.array(json.dumps(meta)))
    return meta


def cert_document(outcome, config, *, solver_cert=None):
    document = dict(outcome.record(), config=certificate_record(config))
    if solver_cert is not None:
        document["matches_solver_certificate"] = bool(solver_cert.state_hash == outcome.state_hash
                                                      and solver_cert.status == outcome.status)
    return document


def export_native(out_dir, *, u, grid, status, backend, solver_cert, init_field=None):
    """final_native.npz, cert.json and -- only when the fresh CPU certificate of u passes -- disp.npz
    (C16 gate, hard constraint 1).  `init_field` (C16 WST item; None: nothing added): the seed tag of --init-field, written
    into the three files.

    The state is certified first (CPU FP64, the C07-2 configuration of `backend`).  PASS: the three files with
    the run status.  Not PASS: final_native.npz with status FINAL_CERTIFICATE_REJECTED and cert.json with
    rejected = true are written as failure evidence, disp.npz is NOT written, and the returned dict has
    certificate_passed = False and status FINAL_CERTIFICATE_REJECTED (the caller ends the run as a failure)."""
    from .stopping import FINAL_CERTIFICATE_REJECTED
    from .. import memtrace
    out_dir = Path(out_dir)
    u = np.asarray(u)
    memtrace.mark("export.start")
    outcome = backend.certify(u, grid)
    memtrace.mark("export.certified")
    passed = bool(outcome.passed)
    final_status = status if passed else FINAL_CERTIFICATE_REJECTED
    digest = write_final_state(out_dir/"final_native.npz", u, grid, status=final_status, init_field=init_field)
    if outcome.state_hash != digest:
        raise RuntimeError("certificate digest differs from the stored state_hash (C07-7)")
    document = cert_document(outcome, backend.cert_config, solver_cert=solver_cert)
    document.update(rejected=not passed, disp_written=passed, solver_status=status, exported_status=final_status)
    if init_field is not None:
        document["init_field"] = init_field
    write_json(out_dir/"cert.json", document)
    meta = write_disp(out_dir/"disp.npz", u, grid, state_hash=digest, init_field=init_field) if passed else None
    return dict(state_hash=digest, cert=outcome.record(), disp_meta=meta, certificate_passed=passed,
                status=final_status, solver_status=status)
