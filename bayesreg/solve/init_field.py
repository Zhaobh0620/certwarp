"""The seed of the native-level warm start `run_pair --init-field` (contract C12-4a, C16 WST item, C20 D59; WST 2026-11-02, WARM
2026-11-03: default off, path only; BZ 11-03: a seed may enter the method, whether it becomes the default is open).

`load_init_field(path, pair=, pair_name=, oracle_ok=)` reads the `final_native.npz` of a finished run, binds it to this run's
pair and native grid, reads its provenance and returns an `InitSeed` (the float64 array and the record), or raises
`InitFieldRejected` (run_pair: status init_field_rejected, exit 5, before any channel is read).  Nothing here certifies:
the entry certificate is the .9^k back-off of solve_multilevel (C12-4a item 4).

Two source formats, detected from the files next to the npz; anything else is refused (provenance unknown):
  bayesreg_run_pair  <run>/final_native.npz (u, state_hash, grid, status), <run>/cert.json, <run>/config.json (args.pair),
                     <run>/provenance.json (pair, channels, init_field)
  frozen_configC     <run>/<task>/final_native.npz (u; `warped` is never read), <run>/<task>/final_native.json (sha256,
                     certificate.state_hash, certificate.grid_id), <run>/protocol.json (stage, task, input_sha256),
                     <run>/inputs/<pair>/manifest.json (subjects, pair_sha256, preprocessing_d0)
Bindings (C12-4a item 2; each failure names itself): u float64, shape grid.shape + (3,), finite, six faces zero; sha256 of the
C-order bytes = the state hash the source recorded; the source's grid_id = this native grid's; the source's pair name, its
subjects and its pair sha256 (the recoloured inputs of D0: preprocessing_d0.source_pair_sha256) = this pair's.
Provenance (C12-4a item 3): oracle = truth labels shaped the field.  frozen_configC: preprocessing_d0.oracle (arms A1, A3, A5 must
say true; no preprocessing_d0 = the original images, arm taken from the stage name); bayesreg_run_pair: test_labels_seen /
oracle of every accepted layer provenance, and the source's own init_field.oracle (transitive).  An oracle seed needs
oracle_ok; the flag then marks every record.
Label firewall (hard constraint 7): the source's manifest is read through a JSON hook that drops every key whose name
contains seg, label or colour (baseline_label_metrics, labels_policy, preprocessing_d0.colour), and its pair.npz is not opened.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from jd_curl.three_d.nifti import file_sha256
from jd_curl.three_d.validation import zero_boundary

from .stopping import c_order_bytes

CONTRACT = "C12-4a (WST 2026-11-02, WARM 2026-11-03; default off; path only)"
# The run status of a refused seed (exit 5).  Not in stopping.RUN_STATUSES: it exists only under the switch, the solver never
# starts, and the run vocabulary of the default code stays as it is (test_sch_solver.test_level_budget_status_vocabulary).
INIT_FIELD_REJECTED = "init_field_rejected"
INIT_FIELD_REJECTED_WORDING = "运行被拒（--init-field 的种子不合格）"
FORMAT_BAYESREG = "bayesreg_run_pair"
FORMAT_FROZEN = "frozen_configC"
SEED_NAME = "final_native.npz"
ORACLE_ARMS = ("A1", "A3", "A5")          # D0: truth-label colourings (docs/bayesreg/05_D0协议.md §3; A5: runs/20261103-a5)
DROP_MARKERS = ("seg", "label", "colour")
SKIP_REASON = "--init-field: the seed enters at the native level; levels 0..2 are not run (C12-4a item 5)"


class InitFieldRejected(ValueError):
    """The seed is refused: format, binding to this pair and grid, or provenance (C12-4a items 2, 3)."""


@dataclass
class InitSeed:
    """The admitted seed: `u` the float64 array as stored, `record` its JSON record (with `oracle`)."""
    u: np.ndarray
    record: dict

    @property
    def oracle(self):
        return self.record["oracle"]

    def release(self):
        """solve_multilevel drops the array once the entry state exists (release_inputs; memory only)."""
        self.u = None


def _reject(reason):
    raise InitFieldRejected(f"--init-field: {reason}")


def _json_object(document, what, path):
    if not isinstance(document, dict):
        _reject(f"{what} {path}: a JSON object is required")
    return document


def _read_json(path, what):
    try:
        return _json_object(json.loads(Path(path).read_text(encoding="utf-8")), what, path)
    except (OSError, ValueError) as exc:
        _reject(f"{what} {path} unreadable ({type(exc).__name__}: {exc})")


def _no_label_keys(pairs):
    return {key: value for key, value in pairs if not any(marker in key.lower() for marker in DROP_MARKERS)}


def _read_manifest(path):
    """The source's manifest without any key naming labels (hard constraint 7)."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return _json_object(json.load(stream, object_pairs_hook=_no_label_keys), "manifest", path)
    except (OSError, ValueError) as exc:
        _reject(f"manifest {path} unreadable ({type(exc).__name__}: {exc})")


def _read_members(path, names, required):
    """{name: array} of the npz members `names` (others are never decompressed); `required` must exist."""
    try:
        with np.load(path, allow_pickle=False) as npz:
            missing = [k for k in required if k not in npz.files]
            if missing:
                _reject(f"{path} lacks {missing}")
            return {k: npz[k] for k in names if k in npz.files}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        _reject(f"{path} unreadable ({type(exc).__name__}: {exc})")


def _sha(array):
    return hashlib.sha256(c_order_bytes(array)).hexdigest()


def _check_array(u, grid):
    if not isinstance(u, np.ndarray) or u.dtype != np.float64 or u.shape != grid.shape+(3,):
        _reject(f"u must be float64 of shape {grid.shape+(3,)} (the native grid), got {getattr(u, 'dtype', None)} "
                f"{getattr(u, 'shape', None)}")
    if not np.all(np.isfinite(u)):
        _reject("u has non-finite values")
    if not zero_boundary(u):
        _reject("u is not zero on the six faces (C01)")


def _check_pair_identity(what, name, fixed, moving, pair_sha, *, pair, own):
    if name != own:
        _reject(f"the source run's pair {name!r} is not this pair {own!r} ({what})")
    if (fixed, moving) != (pair.fixed_subject, pair.moving_subject):
        _reject(f"the source's subjects {(fixed, moving)} are not this pair's "
                f"{(pair.fixed_subject, pair.moving_subject)} ({what})")
    if pair_sha != pair.provenance["pair_sha256"]:
        _reject(f"the source's pair sha256 {pair_sha} is not this pair's {pair.provenance['pair_sha256']} ({what})")


def _certificate_summary(cert):
    keys = ("status", "reason", "min_jd_lower_bound", "n_infeasible", "n_uncertain", "state_hash", "certificate_version")
    return {k: cert.get(k) for k in keys} if isinstance(cert, dict) else None


def _load_frozen(path, grid, pair, own):
    task_dir = path.parent
    run_dir = task_dir.parent
    final = _read_json(task_dir/"final_native.json", "final_native.json")
    protocol = _read_json(run_dir/"protocol.json", "protocol.json")
    task = protocol.get("task") if isinstance(protocol.get("task"), dict) else {}
    if task.get("name") != task_dir.name:
        _reject(f"protocol.json task.name {task.get('name')!r} is not the directory {task_dir.name!r}")
    file_sha = file_sha256(path)
    if final.get("sha256") != file_sha:
        _reject(f"final_native.json sha256 {final.get('sha256')} is not the file's {file_sha}")
    u = _read_members(path, ("u",), ("u",))["u"]        # `warped` is never decompressed
    _check_array(u, grid)
    cert = final.get("certificate") if isinstance(final.get("certificate"), dict) else {}
    digest = _sha(u)
    if cert.get("state_hash") != digest:
        _reject(f"the recorded state_hash {cert.get('state_hash')} is not sha256(u) = {digest}")
    if cert.get("grid_id") != grid.grid_id:
        _reject(f"the source's grid_id {cert.get('grid_id')} is not this native grid's {grid.grid_id}")
    name = task.get("pair")
    manifest_path = run_dir/"inputs"/str(name)/"manifest.json"
    inputs = protocol.get("input_sha256") if isinstance(protocol.get("input_sha256"), dict) else {}
    if not manifest_path.is_file() or inputs.get("manifest.json") != file_sha256(manifest_path):
        _reject(f"{manifest_path}: missing or not the manifest of protocol.json input_sha256")
    manifest = _read_manifest(manifest_path)
    if manifest.get("pair") != name or inputs.get("pair.npz") != manifest.get("pair_sha256"):
        _reject(f"{manifest_path}: pair or pair_sha256 disagree with protocol.json")
    stage = protocol.get("stage")
    pre = manifest.get("preprocessing_d0")
    if pre is None:
        arm, oracle, source_pair = None, False, manifest.get("pair_sha256")
        if isinstance(stage, str) and stage.startswith("D0-"):
            arm = stage[3:]
            if arm != "A0":
                _reject(f"stage {stage} without the manifest's preprocessing_d0 (provenance unknown)")
    else:
        if not isinstance(pre, dict) or not isinstance(pre.get("arm"), str) or not isinstance(pre.get("oracle"), bool):
            _reject("manifest preprocessing_d0: arm and a boolean oracle are required (provenance unknown)")
        arm, oracle, source_pair = pre["arm"], pre["oracle"], pre.get("source_pair_sha256")
        if arm in ORACLE_ARMS and not oracle:
            _reject(f"arm {arm} is a truth-label colouring but preprocessing_d0.oracle is false (inconsistent)")
        if isinstance(stage, str) and stage.startswith("D0-") and stage != f"D0-{arm}":
            _reject(f"stage {stage} and preprocessing_d0.arm {arm} disagree")
    _check_pair_identity("frozen run", name, manifest.get("fixed_subject"), manifest.get("moving_subject"), source_pair,
                         pair=pair, own=own)
    record = dict(source_format=FORMAT_FROZEN, source_run_dir=str(run_dir), file_sha256=file_sha, state_hash=digest,
                  grid_id=cert.get("grid_id"), source_pair=name, source_pair_sha256=source_pair, arm=arm, oracle=oracle,
                  source_stage=stage, source_role=final.get("role"), source_certificate=_certificate_summary(cert),
                  source_protocol_sha256=file_sha256(run_dir/"protocol.json"),
                  source_pre_optimizer=protocol.get("pre_optimizer"), source_seconds=task.get("wall_seconds"))
    return u, record


def _layer_oracle(entry, where):
    """True when an accepted layer provenance says truth labels were seen; a missing record is refused."""
    if not isinstance(entry, dict) or not isinstance(entry.get("accepted"), dict):
        _reject(f"provenance.json {where}: no accepted provenance record (provenance unknown)")
    accepted = entry["accepted"]
    return bool(accepted.get("oracle")) or accepted.get("test_labels_seen") is not False


def _bayesreg_oracle(provenance):
    channels = provenance.get("channels") if isinstance(provenance.get("channels"), dict) else None
    if channels is None or not isinstance(channels.get("tissue"), dict):
        _reject("provenance.json channels.tissue missing (provenance unknown)")
    oracle = any(_layer_oracle(channels["tissue"].get(side), f"channels.tissue.{side}") for side in ("fixed", "moving"))
    structure = channels.get("structure")
    if isinstance(structure, dict):
        oracle |= any(_layer_oracle(structure.get(side), f"channels.structure.{side}") for side in ("fixed", "moving"))
    upstream = provenance.get("init_field")
    if upstream is not None and (not isinstance(upstream, dict) or not isinstance(upstream.get("oracle"), bool)):
        _reject("provenance.json init_field: a boolean oracle is required (provenance unknown)")
    return oracle or bool(upstream and upstream["oracle"])


def _load_bayesreg(path, grid, pair, own):
    run_dir = path.parent
    config = _read_json(run_dir/"config.json", "config.json")
    provenance = _read_json(run_dir/"provenance.json", "provenance.json")
    cert = _read_json(run_dir/"cert.json", "cert.json")
    names = ("u", "state_hash", "grid", "status")
    members = _read_members(path, names, names)
    u, stored_hash, grid_text, status = members["u"], str(members["state_hash"]), str(members["grid"]), str(members["status"])
    _check_array(u, grid)
    digest = _sha(u)
    if stored_hash != digest or cert.get("state_hash") != digest:
        _reject(f"the recorded state_hash (npz {stored_hash}, cert.json {cert.get('state_hash')}) is not sha256(u) = {digest}")
    try:
        source_grid = _json_object(json.loads(grid_text), "grid record of", path)
    except ValueError as exc:
        _reject(f"{path} grid record unreadable ({exc})")
    if source_grid.get("grid_id") != grid.grid_id:
        _reject(f"the source's grid_id {source_grid.get('grid_id')} is not this native grid's {grid.grid_id}")
    args = config.get("args") if isinstance(config.get("args"), dict) else {}
    name = Path(str(args.get("pair"))).name
    source_pair = provenance.get("pair") if isinstance(provenance.get("pair"), dict) else {}
    _check_pair_identity("bayesreg run", name, source_pair.get("fixed_subject"), source_pair.get("moving_subject"),
                         source_pair.get("pair_sha256"), pair=pair, own=own)
    oracle = _bayesreg_oracle(provenance)
    upstream = provenance.get("init_field")
    code = provenance.get("code") if isinstance(provenance.get("code"), dict) else {}
    record = dict(source_format=FORMAT_BAYESREG, source_run_dir=str(run_dir), file_sha256=file_sha256(path),
                  state_hash=digest, grid_id=source_grid.get("grid_id"), source_pair=name,
                  source_pair_sha256=source_pair.get("pair_sha256"), arm=provenance.get("channel_source"), oracle=oracle,
                  source_status=status, source_certificate=_certificate_summary(cert),
                  source_provenance_sha256=file_sha256(run_dir/"provenance.json"), source_code_head=code.get("head"),
                  source_channel_source=provenance.get("channel_source"),
                  source_init_field=None if upstream is None else {k: upstream.get(k) for k in (
                      "source_format", "arm", "oracle", "state_hash")})
    return u, record


def load_init_field(path, *, pair, pair_name, oracle_ok=False):
    """The admitted seed of this pair (see the module docstring), or InitFieldRejected.  `pair`: the PairInput of load_pair;
    `pair_name`: run_pair's --pair, "<stage>/<pair>"."""
    path = Path(path).absolute()
    if path.name != SEED_NAME or not path.is_file():
        _reject(f"{path}: a run's {SEED_NAME} is required")
    own = Path(pair_name).name
    grid = pair.grid
    is_bayesreg = all((path.parent/name).is_file() for name in ("config.json", "provenance.json", "cert.json"))
    is_frozen = (path.parent/"final_native.json").is_file() and (path.parent.parent/"protocol.json").is_file()
    if is_bayesreg == is_frozen:
        _reject(f"{path}: not recognized as a run_pair or a frozen configuration C run directory (provenance unknown)")
    u, record = (_load_bayesreg if is_bayesreg else _load_frozen)(path, grid, pair, own)
    if record["oracle"] and not oracle_ok:
        _reject(f"{path}: an oracle seed (truth labels shaped this field, arm {record['arm']}); refused without "
                "--init-field-oracle-ok (C12-4a item 3)")
    return InitSeed(u, dict(contract=CONTRACT, path=str(path), oracle_ok=bool(oracle_ok), **record))


def status_summary(record):
    """The part of the seed record that status.json carries (and the eval labels read)."""
    return {k: record.get(k) for k in ("path", "source_format", "arm", "oracle", "state_hash", "accepted_k", "scale",
                                       "entry_state_hash")}


__all__ = ["CONTRACT", "FORMAT_BAYESREG", "FORMAT_FROZEN", "INIT_FIELD_REJECTED", "INIT_FIELD_REJECTED_WORDING",
           "InitFieldRejected", "InitSeed", "SKIP_REASON", "load_init_field", "status_summary"]
