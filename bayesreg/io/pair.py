"""C03.1 pair input behind the label firewall and the subject guard; C03.1-2/3 segmentation provenance.

pair.npz as written by the mainline data preparation (eb808be scripts/prepare_stage_3d_*.py, now
archive/seq_vp_3d/scripts/) holds fixed, moving, the label maps (fixed_seg4, ..., or LPBA40's
fixed_lpba56, ...) and an all-zero initial_u; the sibling manifest.json carries the grid (shape_zyx,
spacing_xyz, origin_xyz, unit), pair_sha256, fixed_subject / moving_subject and sources.<key>.sha256,
next to label-derived records (baseline_label_metrics, sources.fixed_seg4, ...).

Firewall (C03.1-2), whitelist only:
  - pair.npz: the loader asks the archive only for the members `fixed`, `moving` (and `initial_u`),
    one `npz[key]` at a time; it never iterates the archive, never lists its members, never calls
    dict(npz). A container-level sha256 of the whole file is allowed; decoding any other member is not.
  - manifest.json: only MANIFEST_FIELDS at the top level, `sources.fixed.sha256` and
    `sources.moving.sha256` are kept. While the JSON is decoded, every object keeps only keys from
    that whitelist (and none containing "seg" or "label"), so label-derived records are discarded by
    the decoder hook before any object holding them is built and never reach this module's values.
  - Subject guard (guard.py): fixed_subject / moving_subject come from the manifest (they are not
    label-derived), must be canonical ids and on the development allowlist, and a subject of a sealed
    list only through an unseal object bound to the allowlist (C23.4); this runs before the npz is
    opened or hashed. A pair admitted that way is registered by identity with that bound allowlist,
    and validate_segmentation_provenance (both layers), the StructureLayer and run_pair's tissue
    layer require the very same object again (C23.4-8, D103).

Segmentation inputs come in two layers, each with its own provenance per image (contract C03.1-2):
the tissue layer P_T (Atropos EM, no labels; tau^2, brain mask, white-matter normalization) and the
structure layer P_S (registered self-trained fold models; likelihood channels s^k only).
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np

from jd_curl.validation import InputValidationError, as_float64_array, readonly_copy
from jd_curl.three_d.grid import Grid3D, make_grid_3d
from jd_curl.three_d.nifti import file_sha256

from .guard import (DevAllowlist, SubjectSealed, canonical_subject_list, check_lpba40_names, check_pair_subjects, check_subject,
                    is_hex64, is_safe_id, load_dev_allowlist, pair_allowlist, register_pair, strict_json_object)

IMAGE_KEYS = ("fixed", "moving")
OPTIONAL_ZERO_KEYS = ("initial_u",)
SIDES = ("fixed", "moving")
MANIFEST_GRID_KEYS = ("shape_zyx", "spacing_xyz", "origin_xyz", "unit")
MANIFEST_FIELDS = MANIFEST_GRID_KEYS+("pair_sha256", "fixed_subject", "moving_subject", "sources")
MANIFEST_SOURCE_FIELDS = ("sha256",)
_MANIFEST_KEY_NAMES = frozenset(MANIFEST_FIELDS+SIDES+MANIFEST_SOURCE_FIELDS)
MANIFEST_DROP_MARKERS = ("seg", "label")

LAYERS = ("tissue", "structure")
TISSUE_SEGMENTERS = ("atropos",)
PENDING_TISSUE_SEGMENTERS = ("fast",)              # refused until BZ decides (C20 D7)
FORBIDDEN_ID = re.compile(r"oracle|truth|seg4|seg35", re.IGNORECASE)
WM_CLASS_NAMES = ("wm", "white_matter")            # compared case-insensitively; canonical spelling "WM"
SHA256_FIELDS = ("input_sha256", "output_sha256")
REGISTRY_SCHEMA = "bayesreg.segmenters.v1"
REGISTRY_ENTRY_KEYS = ("segmenter_id", "fold", "member_model_sha256", "label_contact_subjects",
                       "label_contact_subjects_sha256")
CHECK = "C03.1-2/review-F2"


@dataclass(frozen=True)
class PairInput:
    """Fixed/moving T1 on one node grid (zyx storage, float64, read-only) plus the manifest's canonical
    subject ids and source-image sha256 per side; nothing label-derived."""
    fixed: np.ndarray
    moving: np.ndarray
    grid: Grid3D
    grid_metadata: dict
    fixed_subject: str
    moving_subject: str
    source_sha256: Mapping
    provenance: dict


class ProvenanceRejected(Exception):
    """C03.1-2: the posteriors may not be used; run status `provenance_rejected`."""
    status = "provenance_rejected"


class AcceptedProvenance(dict):
    """A provenance record returned by `validate_segmentation_provenance` (JSON-serializable, no item
    assignment). `normalize_intensity` accepts only this type for the tissue layer."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("AcceptedProvenance is read-only")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _immutable

    def __reduce__(self):                          # copy/deepcopy/pickle without item assignment
        return (type(self), (dict(self),))


def _open_npz(path):
    # Lazy archive: members are decompressed only when indexed. Tests replace this hook with a
    # guard that raises on any non-whitelisted access.
    return np.load(path, allow_pickle=False)


def _keep_whitelisted_keys(pairs):
    """json object_pairs_hook: keep only whitelisted key names (never one containing seg/label)."""
    kept = {}
    for key, value in pairs:
        if key in _MANIFEST_KEY_NAMES and not any(marker in key.lower() for marker in MANIFEST_DROP_MARKERS):
            if key in kept:
                raise InputValidationError(f"manifest: duplicate key {key!r}")
            kept[key] = value
    return kept


def _read_manifest(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        decoded = json.load(stream, object_pairs_hook=_keep_whitelisted_keys)
    if not isinstance(decoded, dict):
        raise InputValidationError("manifest: expected a JSON object")
    missing = [key for key in MANIFEST_FIELDS if key not in decoded]
    if missing:
        raise InputValidationError(f"manifest: missing fields {missing}")
    sources = decoded["sources"]
    if not isinstance(sources, dict) or not all(
            isinstance(sources.get(side), dict) and is_hex64(sources[side].get("sha256")) for side in SIDES):
        raise InputValidationError("manifest: sources.fixed.sha256 and sources.moving.sha256 (64 hex) required")
    if not is_hex64(decoded["pair_sha256"]):
        raise InputValidationError("manifest.pair_sha256: 64 lowercase hex required")
    shape, spacing, origin, unit = (decoded[key] for key in MANIFEST_GRID_KEYS)
    for name, value in (("shape_zyx", shape), ("spacing_xyz", spacing), ("origin_xyz", origin)):
        if not isinstance(value, list) or len(value) != 3:
            raise InputValidationError(f"manifest.{name}: expected a list of 3 numbers")
    if unit != "mm":
        raise InputValidationError("manifest.unit: C01-1 requires mm")
    grid = make_grid_3d(nx=shape[2], ny=shape[1], nz=shape[0], hx=spacing[0], hy=spacing[1], hz=spacing[2],
                        x0=origin[0], y0=origin[1], z0=origin[2], unit=unit)
    metadata = dict(shape_zyx=list(grid.shape), spacing_xyz=list(grid.spacing),
                    origin_xyz=[float(a[0]) for a in grid.axes], unit=grid.unit)
    selected = dict(pair_sha256=decoded["pair_sha256"], fixed_subject=decoded["fixed_subject"],
                    moving_subject=decoded["moving_subject"],
                    source_sha256={side: sources[side]["sha256"] for side in SIDES})
    return grid, metadata, selected


def load_pair(path, *, allowlist, manifest_path=None, unseal_token=None):
    """C03.1: read `fixed`, `moving` (and an all-zero `initial_u` if present) from pair.npz.

    Order: the directory names (C26.3-6: a name naming an id of guard.LPBA40_TEST_REJECT is refused unless `allowlist` is
    the LPBA allowlist and its exception holds -- guard.check_lpba40_names; given a manifest_path, its directory's name
    too) -> manifest (whitelisted fields only) -> subject guard (`allowlist`: a DevAllowlist or the
    path of the allowlist JSON; SubjectSealed, status `subject_sealed`, for any subject not on it or
    not canonical) -> container sha256 of pair.npz against the required manifest.pair_sha256 ->
    whitelisted members. The grid is built by `make_grid_3d` from the manifest (default:
    manifest.json next to the npz). `unseal_token`: deprecated, records only -- the token's sha256 is
    recorded and admits nothing; unsealing is C23's unseal object, on the allowlist of
    bayesreg.io.unseal.bind_unseal (a pair with a subject admitted through it is registered by identity
    with that allowlist, C23.4-8). Returns `PairInput`.
    """
    path = Path(path)
    named_dirs = [path.parent.name, path.resolve().parent.name]      # review minor 1: the resolved name too ('pair.npz' given from inside the pair directory, a link)
    if manifest_path is not None:
        named_dirs += [Path(manifest_path).parent.name, Path(manifest_path).resolve().parent.name]
    manifest_path = path.parent/"manifest.json" if manifest_path is None else Path(manifest_path)
    if not isinstance(allowlist, DevAllowlist):
        if not isinstance(allowlist, (str, Path)):
            raise InputValidationError("allowlist: DevAllowlist or path to the allowlist JSON required")
        allowlist = load_dev_allowlist(allowlist)
    check_lpba40_names(named_dirs, allowlist)       # C26.3-6 (D144): by the directory names first, before the manifest; the exception holds here
    grid, metadata, manifest = _read_manifest(manifest_path)
    guard = check_pair_subjects(manifest["fixed_subject"], manifest["moving_subject"], allowlist,
                                unseal_token=unseal_token)
    pair_sha = file_sha256(path)
    if pair_sha != manifest["pair_sha256"]:
        raise InputValidationError("pair.npz sha256 differs from manifest.pair_sha256")
    keys_read = []
    with _open_npz(path) as npz:
        images = {}
        for key in IMAGE_KEYS:
            if key not in npz:
                raise InputValidationError(f"pair.npz: required member {key!r} missing")
            images[key] = readonly_copy(as_float64_array(npz[key], key, grid.shape))
            keys_read.append(key)
        initial_u_present = OPTIONAL_ZERO_KEYS[0] in npz
        if initial_u_present:
            initial_u = as_float64_array(npz[OPTIONAL_ZERO_KEYS[0]], "initial_u", grid.shape+(3,), check_finite=False)
            keys_read.append(OPTIONAL_ZERO_KEYS[0])
            if not np.all(initial_u == 0):
                raise InputValidationError("initial_u: C03.1-2 admits only an all-zero initial_u")
    provenance = dict(pair_path=str(path), pair_sha256=pair_sha, manifest_path=str(manifest_path),
                      manifest_sha256=file_sha256(manifest_path), keys_read=keys_read,
                      initial_u_present=initial_u_present, label_firewall="whitelist-v2",
                      fixed_subject=manifest["fixed_subject"], moving_subject=manifest["moving_subject"],
                      source_sha256=dict(manifest["source_sha256"]), subject_guard=guard, omega0=grid.omega)
    pair = PairInput(images["fixed"], images["moving"], grid, metadata, manifest["fixed_subject"],
                     manifest["moving_subject"], MappingProxyType(dict(manifest["source_sha256"])), provenance)
    if guard["unsealed"]:
        register_pair(pair, allowlist)
    return pair


def recheck_sealed_pair(pair, allowlist, *, where):
    """C23.4-8 (D103 (a)), before anything of the pair's subjects is read: None for a pair with no sealed subject.  For
    a pair with one, `allowlist` must let both ids through the guard (check_subject: ids only), be the very object
    load_pair registered for this pair object (`is`), and carry the unseal object whose sha256 the pair's guard record
    names.  ProvenanceRejected or SubjectSealed otherwise (no allowlist, another one, a PairInput not of load_pair)."""
    from .sealed import guard_sealed_lists, lists_of
    lists = guard_sealed_lists()
    sealed = [s for s in (pair.fixed_subject, pair.moving_subject) if lists_of(lists, s)]
    if not sealed:
        return None
    if not isinstance(allowlist, DevAllowlist):
        raise ProvenanceRejected(f"{where}: the pair has sealed subjects {sealed}: the bound allowlist load_pair admitted "
                                 "it through is required (C23.4-8)")
    for side, subject in (("fixed", pair.fixed_subject), ("moving", pair.moving_subject)):
        verdict = check_subject(subject, allowlist)
        if not verdict.admitted:
            raise SubjectSealed(f"{where}: {side}: {verdict.message or f'{subject!r} is not on the development allowlist'}")
    if pair_allowlist(pair) is not allowlist:
        raise ProvenanceRejected(f"{where}: this pair was not admitted by load_pair through this very allowlist object "
                                 "(another bound allowlist, or a PairInput not made by load_pair; C23.4-8)")
    record = (pair.provenance.get("subject_guard") or {}).get("sealed_guard") or {}
    if (record.get("unseal") or {}).get("sha256") != allowlist.unseal.sha256:
        raise ProvenanceRejected(f"{where}: the pair's guard record does not name the unseal record of this allowlist")
    return dict(check="C23.4-8: the pair and its bound allowlist by identity", sealed_subjects=sealed,
                unseal_sha256=allowlist.unseal.sha256)


def subject_list_sha256(subjects):
    """Canonical hash of a subject list: UTF-8 text, one id per line in list order, each line ending in \\n."""
    return hashlib.sha256("".join(f"{s}\n" for s in subjects).encode("utf-8")).hexdigest()


def _reject(reason):
    raise ProvenanceRejected(reason)


def _admissible_id(value):
    return is_safe_id(value) and FORBIDDEN_ID.search(value) is None


# ---- segmenter registry (per fold model) ------------------------------------------------------

@dataclass(frozen=True)
class SegmenterRegistry:
    """Parsed configs/bayesreg/segmenters.json (location pending the architecture document)."""
    path: str
    sha256: str
    models: Mapping


def _parse_registry(document, path, sha256):
    if not isinstance(document, dict) or set(document) - {"comment"} != {"schema", "models"} \
            or document["schema"] != REGISTRY_SCHEMA or not isinstance(document.get("comment", ""), str):
        _reject(f"registry malformed: expected {{schema: {REGISTRY_SCHEMA!r}, models: {{...}}}}")
    models = document["models"]
    if not isinstance(models, dict) or not models:
        _reject("registry malformed: models must be a non-empty object")
    parsed, folds = {}, set()
    for model_id, entry in models.items():
        if not _admissible_id(model_id):
            _reject(f"registry malformed: model id {model_id!r} is not an admissible identifier")
        if not isinstance(entry, dict) or set(entry) != set(REGISTRY_ENTRY_KEYS):
            _reject(f"registry entry {model_id!r} malformed: expected exactly the keys {list(REGISTRY_ENTRY_KEYS)}")
        sid, fold, members = entry["segmenter_id"], entry["fold"], entry["member_model_sha256"]
        if not _admissible_id(sid) or sid in TISSUE_SEGMENTERS+PENDING_TISSUE_SEGMENTERS:
            _reject(f"registry entry {model_id!r} malformed: segmenter_id {sid!r}")
        if not isinstance(fold, int) or isinstance(fold, bool) or fold < 0:
            _reject(f"registry entry {model_id!r} malformed: fold must be a non-negative integer")
        if not isinstance(members, list) or not members or not all(is_hex64(m) for m in members) \
                or len(set(members)) != len(members):
            _reject(f"registry entry {model_id!r} malformed: member_model_sha256 must list distinct 64-hex digests")
        try:
            contact = canonical_subject_list(entry["label_contact_subjects"], f"{model_id}.label_contact_subjects")
        except InputValidationError as exc:
            _reject(f"registry entry {model_id!r} malformed: {exc}")
        if subject_list_sha256(contact) != entry["label_contact_subjects_sha256"]:
            _reject(f"registry entry {model_id!r}: label_contact_subjects do not match their sha256")
        if (sid, fold) in folds:
            _reject(f"registry malformed: two models of {sid!r} for fold {fold}")
        folds.add((sid, fold))
        parsed[model_id] = MappingProxyType(dict(segmenter_id=sid, fold=fold, member_model_sha256=tuple(members),
                                                 label_contact_subjects=tuple(contact),
                                                 label_contact_subjects_sha256=entry["label_contact_subjects_sha256"]))
    return SegmenterRegistry(path=path, sha256=sha256, models=MappingProxyType(parsed))


def load_segmenter_registry(path):
    """Read the per-fold-model registry; the returned SegmenterRegistry carries the file's sha256.

        {"schema": "bayesreg.segmenters.v1",
         "models": {"<model id>": {"segmenter_id": "<ensemble id>", "fold": <int >= 0>,
                                   "member_model_sha256": ["<64 hex of each ensemble member's weights>", ...],
                                   "label_contact_subjects": ["<canonical id>", ...],
                                   "label_contact_subjects_sha256": "<64 hex>"}}}   # subject_list_sha256(list)
    label_contact_subjects = every subject whose labels the model saw: training, validation, early
    stopping and temperature calibration. Ids are canonical (guard.py), distinct. Ids of models and
    segmenters: ASCII identifiers without whitespace, none containing oracle/truth/seg4/seg35 (any case).
    Malformed entries raise ProvenanceRejected.
    """
    raw = Path(path).read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, InputValidationError) as exc:
        _reject(f"registry malformed: not UTF-8 JSON without duplicate keys ({exc})")
    return _parse_registry(document, str(path), hashlib.sha256(raw).hexdigest())


# ---- per-image provenance --------------------------------------------------------------------

def _id_list(value, field):
    if not isinstance(value, list) or not value or not all(_admissible_id(v) for v in value) \
            or len(set(value)) != len(value):
        _reject(f"{field}: non-empty list of distinct admissible identifiers required")
    return list(value)


def _sha_map(value, field):
    if not isinstance(value, Mapping) or not value or not all(
            is_safe_id(name) and is_hex64(digest) for name, digest in value.items()):
        _reject(f"{field}: non-empty {{file name: sha256}} object required (plain file names, 64 lowercase hex)")
    return dict(value)


def _class_names_problem(class_names):
    if not isinstance(class_names, (list, tuple)) or not class_names or not all(is_safe_id(c) for c in class_names):
        return "class_names: non-empty list of identifiers required"
    if len({c.lower() for c in class_names}) != len(class_names):
        return "class_names: duplicate names (case-insensitive)"
    return None


def wm_index_problem(class_names, wm_index):
    """None if `wm_index` (0-based into class_names, or None) is consistent; otherwise the reason."""
    problem = _class_names_problem(class_names)
    if problem:
        return problem
    wm_named = [k for k, c in enumerate(class_names) if c.lower() in WM_CLASS_NAMES]
    if len(wm_named) > 1:
        return f"class_names: more than one white-matter class {[class_names[k] for k in wm_named]}"
    if wm_index is None:
        return None
    if isinstance(wm_index, bool) or not isinstance(wm_index, (int, np.integer)) \
            or not 0 <= wm_index < len(class_names):
        return f"wm_index: integer in [0, {len(class_names)}) or null required"
    if class_names[wm_index].lower() not in WM_CLASS_NAMES:
        return f"wm_index {wm_index} names class {class_names[wm_index]!r}, not white matter ('WM')"
    return None


def validate_segmentation_provenance(prov, pair, *, side, layer, registry=None, allowlist=None):
    """C03.1-2/3: accept one image's posteriors for one layer, or raise ProvenanceRejected.

    `pair`: the PairInput of this run (subjects and source hashes come from its manifest); `side`:
    "fixed" or "moving"; `layer`: "tissue" (P_T) or "structure" (P_S); `prov["layer"]` must equal it.
    Both layers require (JSON types exact; booleans are JSON booleans):
      - `segmenter_id`, `model_ids` (non-empty list): identifiers; any id containing oracle, truth,
        seg4 or seg35 (any case) is a label source and refused, and a truthy `oracle` marker is refused.
        The oracle ablation arm (methodology 8.6-h) will get its own entry point; not implemented here.
      - `test_labels_seen`: exactly false; `trained_with_labels`: false (tissue) / true (structure);
      - `class_names`: the K_X class names, distinct (case-insensitive);
      - `input_sha256`: exactly one {file name: sha256}, equal to manifest `sources.<side>.sha256`
        (hash binding: the posteriors are a function of this image only);
        `output_sha256`: non-empty {file name: sha256}; check the files with `verify_segmentation_files`.
    Tissue layer: `segmenter_id` = "atropos" ("fast" is refused, pending BZ); `model_ids` = the
      Atropos version string(s); `wm_index` = 0-based index into class_names of the white-matter class
      (name "WM", or "white_matter", any case); null is refused for Atropos (C03.1-3). Atropos numbers
      its posterior files and labels from 1 (posteriors01 = class 1); class_names / wm_index here are
      0-based, and the conversion is done when the posterior files are read (file 0k+1 -> index k).
    Structure layer: `registry` (load_segmenter_registry) and `allowlist` (the DevAllowlist that
      load_pair used) are required. Every id in `model_ids` must be a registered fold model of this
      `segmenter_id`; `model_sha256` = {model id: [member sha256, ...]} equal to the registry. Each
      model's label_contact_subjects must all be on the allowlist and none marked never_for_training
      (stage_T0_tune_v1), else that model is refused; their union must not contain either subject of
      the pair (both, for each image's P_S).
    A pair with a subject of a sealed list (C23.4-8, D103): either layer requires `allowlist`, the very bound
    allowlist load_pair admitted this pair object through (recheck_sealed_pair); ProvenanceRejected or SubjectSealed
    otherwise.  A pair without one: as before (the tissue layer needs no allowlist).
    Returns an AcceptedProvenance for provenance.json.
    """
    if layer not in LAYERS:
        raise InputValidationError(f"layer: expected one of {LAYERS}")
    if side not in SIDES:
        raise InputValidationError(f"side: expected one of {SIDES}")
    if not isinstance(pair, PairInput):
        raise InputValidationError("pair: the PairInput returned by load_pair is required")
    recheck_sealed_pair(pair, allowlist, where=f"{layer} layer")
    if layer == "structure":
        if not isinstance(registry, SegmenterRegistry):
            raise InputValidationError("registry: a SegmenterRegistry from load_segmenter_registry is required")
        if not isinstance(allowlist, DevAllowlist) or \
                allowlist.sha256 != pair.provenance["subject_guard"]["allowlist_sha256"]:
            raise InputValidationError("allowlist: the DevAllowlist that load_pair used is required")
    if not isinstance(prov, Mapping):
        _reject("provenance is not a JSON object")
    if prov.get("layer") != layer:
        _reject(f"provenance layer {prov.get('layer')!r} given for the {layer!r} slot")
    sid = prov.get("segmenter_id")
    if not is_safe_id(sid):
        _reject("segmenter_id missing or not an identifier")
    if FORBIDDEN_ID.search(sid):
        _reject(f"segmenter_id {sid!r} is a label source")
    if prov.get("oracle"):
        _reject("input carries the oracle marker")
    if prov.get("test_labels_seen") is not False:
        _reject("test_labels_seen must be present and false")
    if prov.get("trained_with_labels") is not (layer == "structure"):
        _reject(f"trained_with_labels must be {layer == 'structure'} for the {layer} layer")
    model_ids = _id_list(prov.get("model_ids"), "model_ids")
    class_names = prov.get("class_names")
    problem = _class_names_problem(class_names)
    if problem:
        _reject(problem)
    inputs, outputs = (_sha_map(prov.get(field), field) for field in SHA256_FIELDS)
    if len(inputs) != 1 or next(iter(inputs.values())) != pair.source_sha256[side]:
        _reject(f"input_sha256 must be exactly the {side} source image (manifest sources.{side}.sha256)")
    subject = pair.fixed_subject if side == "fixed" else pair.moving_subject
    record = dict(check=CHECK, layer=layer, side=side, subject=subject, segmenter_id=sid, model_ids=model_ids,
                  trained_with_labels=layer == "structure", test_labels_seen=False, class_names=list(class_names),
                  input_sha256=inputs, output_sha256=outputs)
    if layer == "tissue":
        if sid in PENDING_TISSUE_SEGMENTERS:
            _reject(f"tissue layer: {sid!r} is pending BZ; only {TISSUE_SEGMENTERS} is admitted")
        if sid not in TISSUE_SEGMENTERS:
            _reject(f"tissue layer requires one of {TISSUE_SEGMENTERS}, got {sid!r}")
        if "wm_index" not in prov:
            _reject("wm_index: required for the tissue layer")
        wm_index = prov["wm_index"]
        problem = wm_index_problem(class_names, wm_index)
        if problem:
            _reject(problem)
        if wm_index is None:
            _reject("wm_index: Atropos records must name the white-matter class (C03.1-3)")
        return AcceptedProvenance(record, wm_index=int(wm_index), wm_class_name=class_names[wm_index])
    if sid in TISSUE_SEGMENTERS+PENDING_TISSUE_SEGMENTERS:
        _reject(f"structure layer requires a registered fold model, got tissue segmenter {sid!r}")
    entries = {}
    for model_id in model_ids:
        entry = registry.models.get(model_id)
        if entry is None:
            _reject(f"model {model_id!r} is not registered")
        if entry["segmenter_id"] != sid:
            _reject(f"model {model_id!r} is registered for {entry['segmenter_id']!r}, not {sid!r}")
        entries[model_id] = entry
    hashes = prov.get("model_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(model_ids) or any(
            not isinstance(hashes[m], list) or tuple(hashes[m]) != entries[m]["member_model_sha256"]
            for m in model_ids):
        _reject("model_sha256 must map every model id to its registered member sha256 list")
    pair_subjects = {pair.fixed_subject, pair.moving_subject}
    for model_id, entry in entries.items():
        contact = set(entry["label_contact_subjects"])
        sealed = sorted(s for s in contact if not allowlist.allows(s))
        if sealed:
            _reject(f"model {model_id!r} had label contact with subjects off the allowlist {sealed}")
        tuning = sorted(contact & allowlist.never_for_training)
        if tuning:
            _reject(f"model {model_id!r} had label contact with never_for_training subjects {tuning}")
        overlap = sorted(contact & pair_subjects)
        if overlap:
            _reject(f"pair subjects {overlap} are in the label contact set of {model_id!r}")
    return AcceptedProvenance(
        record, model_sha256={m: list(entries[m]["member_model_sha256"]) for m in model_ids},
        folds={m: entries[m]["fold"] for m in model_ids},
        label_contact_subjects_sha256={m: entries[m]["label_contact_subjects_sha256"] for m in model_ids},
        registry_path=registry.path, registry_sha256=registry.sha256, allowlist_sha256=allowlist.sha256,
        pair_subjects_checked=[pair.fixed_subject, pair.moving_subject])


def verify_segmentation_files(prov, path):
    """C03.1-2 hash binding of the outputs: recompute the sha256 of the posterior files.

    `path` is the directory holding every file named in `prov["output_sha256"]`, or the single file
    when exactly one is named (its name must match). Returns {file name: sha256}; raises
    ProvenanceRejected on a missing file or a differing digest.
    """
    outputs = _sha_map(prov.get("output_sha256") if isinstance(prov, Mapping) else None, "output_sha256")
    path = Path(path)
    if path.is_dir():
        targets = {name: path/name for name in outputs}
    elif path.is_file():
        if set(outputs) != {path.name}:
            _reject(f"output_sha256 names {sorted(outputs)}, not the single file {path.name!r}")
        targets = {path.name: path}
    else:
        _reject(f"posterior path {str(path)!r} does not exist")
    verified = {}
    for name, target in targets.items():
        if not target.is_file():
            _reject(f"posterior file {name!r} missing")
        digest = file_sha256(target)
        if digest != outputs[name]:
            _reject(f"posterior file {name!r}: sha256 differs from output_sha256")
        verified[name] = digest
    return verified


__all__ = ["AcceptedProvenance", "PairInput", "ProvenanceRejected", "SegmenterRegistry", "SubjectSealed",
           "load_pair", "load_segmenter_registry", "recheck_sealed_pair", "subject_list_sha256",
           "validate_segmentation_provenance",
           "verify_segmentation_files", "wm_index_problem"]
