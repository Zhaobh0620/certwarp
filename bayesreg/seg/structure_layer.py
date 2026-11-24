"""Structure layer P_S of one pair for run_pair (contract C03.1-2/3, C03.2-4a, C20 D24, D37, D39).

Each image of the pair reads its prediction directory <root>/<canonical subject id>/ as written by
bayesreg.seg.unet.infer.segment_subject: structure_groups.npz (key "p", float16 (K_S+1, nz, ny, nx), channel 0 =
the background group), probs36.npz and provenance.json (docs/bayesreg/04_分割器.md §6, §8). When provenance.json
is missing and `infer` is "missing" (default "never": refuse), the prediction is generated here by segment_subject
(torch, imported only then) with the fold models that never saw that subject's labels; the outputs go to the same
directory, never overwritten. Under the per-subject lock (review PS item 6): a provenance.json written meanwhile
by a concurrent run is read back (generated_in_this_run false); an existing non-empty directory without
provenance.json is kept and the run is refused.

Checks, in this order; every refusal raises ProvenanceRejected (run status provenance_rejected):
  0. grouping (default structure13_v1): its status must admit inference (groups.INFERENCE_STATUSES); an
     unverified grouping (e.g. tissue4_ref, "reference_unverified") is refused, with no override here;
  1. registry (schema bayesreg.segmenters.v1, load_segmenter_registry): the repository registry
     configs/bayesreg/segmenters.json ("registered"), or -- only with `allow_staging` (run_pair
     --structure-registry-staging; review PS item 2) -- any other file, e.g. the staging copy written by the
     training chain ($OUT/segmenters.json), recorded as "staging_unregistered" (未正式登记; development
     diagnostics only). A staging registry needs ensemble.json beside it (written by
     bayesreg.seg.unet.registry.register) with registry_sha256 equal to the registry file's and
     member_check.passed true. The record also carries the registry file's git state (`registry_git_state`:
     in a work tree, tracked, clean = no staged, unstaged or untracked change; review PS1_PS2 item b) -- recorded
     here, required by experiments/d0/make_inputs.py for A4p (A4 review item 3). `open_registry` is this check,
     shared with bayesreg.learn.cli (review PS1_PS2 item c);
  2. preflight, before any prediction is read or generated (D24): per image the expected fold models are the
     registered models of `segmenter_id` without label contact with that subject -- a B-line training subject
     gets the other fold only (cross_fit_other_folds), everyone else every fold (all_folds_mean, which needs at
     least 2 registered folds; review PS item 5); the union of the label contact of the models of both images
     must not contain either subject of the pair (pair level);
  3. per image, on provenance.json: validate_segmentation_provenance(layer "structure": test_labels_seen false,
     trained_with_labels true, registered model ids with their member sha256, contact on the allowlist and not
     never_for_training, pair-level contact, input sha256 = the manifest's source image); segmenter_id and
     subject; model_ids and policy equal to the preflight expectation; members = every registered seed member of
     those models (model id, sha256) and folds equal to the registry; grouping equal to the requested one (name,
     class_names, background and group labels), its recorded status admitted and no unverified override; grid
     equal to the pair's; output files re-hashed (verify_segmentation_files);
  4. read-back (`read_structure_groups` -> bayesreg.seg.channels.read_back_probabilities, the read-back shared with
     infer.read_probabilities and learn.cli.load_structure_layer): float64, finite, >= 0, no
     zero-sum voxel, and BEFORE the division by the channel sum the raw sum on every voxel within RAW_SUM_TOL of 1
     (1e-3 for float16 storage, unit roundoff 2^-11 ~ 4.9e-4; 1e-6 for float32 / float64; any other storage type
     refused; C03.2-4a, D39, review PS item 1); then divided by the channel sum. The array then goes through
     simplex_from_background_class (C03.2-4a, D39) in run_pair.prepare_channels.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

from jd_curl.validation import InputValidationError

from ..io.guard import DevAllowlist, SubjectSealed, load_dev_allowlist
from ..io.pair import (SIDES, PairInput, ProvenanceRejected, load_segmenter_registry, recheck_sealed_pair,
                       validate_segmentation_provenance, verify_segmentation_files)
from .channels import RAW_SUM_TOL, read_back_probabilities
from .unet.groups import DEFAULT_GROUPS_PATH, INFERENCE_STATUSES, GroupingUnverified, load_grouping, \
    require_inference_grouping

REPO = Path(__file__).resolve().parents[2]
REPO_REGISTRY = REPO/"configs"/"bayesreg"/"segmenters.json"
# The only files StructureSource.formal_registry may name (UNV2 review, important 2): "registered" keeps a fixed meaning.
FORMAL_REGISTRIES = (REPO_REGISTRY, REPO/"configs"/"bayesreg"/"segmenters_v2.json", REPO/"configs"/"bayesreg"/"segmenters_nn935_v1.json",
                     REPO/"configs"/"bayesreg"/"segmenters_hs_v1.json",     # HS: the UF candidate, written once after its review (runs/20261106-hs/plan.md)
                     REPO/"configs"/"bayesreg"/"segmenters_r2_v1.json")     # R2: the candidate of development round 2, written once after its gate and review (runs/20261110-r2/plan.md)
SINGLE_MODEL_SEGMENTERS = ("nn935_v1",)        # one nnU-Net fold-all model, no folds to average (runs/20261105-fx/plan.md); every other segmenter still needs 2 folds
SEGMENTER_ID = "unet35_v1_ens"                 # = bayesreg.seg.unet.SEGMENTER_ID
DEFAULT_GROUPING = "structure13_v1"
GROUPS_FILE = "structure_groups.npz"           # = bayesreg.seg.unet.infer.GROUPS_FILE (that module imports torch)
PROVENANCE_FILE = "provenance.json"            # = bayesreg.seg.unet.infer.PROVENANCE_FILE
INFER_MODES = ("missing", "never")
REGISTERED, STAGING = "registered", "staging_unregistered"
ENSEMBLE_FILE = "ensemble.json"                # = bayesreg.seg.unet.registry.register, beside the staging registry
CONTRACT = "C03.1-2/3, C03.2-4a, D24, D37, D39"


def _reject(reason):
    raise ProvenanceRejected(reason)


@dataclass(frozen=True)
class StructureSource:
    """Where the structure layer comes from and how a missing prediction is made.

    root: predictions <root>/<subject>/; registry_path: segmenters.json (staging copy or the repository one);
    allowlist_path: the development allowlist load_pair used; models_root: member.pt files (<models_root>/*/);
    image_root: <image_root>/<subject>/aligned_norm.nii.gz for inference; groups_path / grouping: the grouping
    table and name; infer: "never" (default: refuse when provenance.json is absent) or "missing" (generate it);
    device: torch device of the inference (None: cuda when available, else cpu -- it occupies the GPU, not
    during timed runs); synthetic: inference admits only synthetic ids and a non-real image root (tests, dry
    runs) instead of the pinned repository allowlist; command: recorded by the inference; allow_staging: admit a
    registry other than the repository one (staging_unregistered, development diagnostics only); formal_registry:
    the file that counts as the repository registry here, when it is not configs/bayesreg/segmenters.json (A5Pv2:
    configs/bayesreg/segmenters_v2.json; experiments/d0/make_inputs.py), one of FORMAL_REGISTRIES, anything else is
    refused; None for everything else."""
    root: Path
    registry_path: Path
    allowlist_path: Path
    models_root: Path
    image_root: Path
    groups_path: Path = DEFAULT_GROUPS_PATH
    grouping: str = DEFAULT_GROUPING
    segmenter_id: str = SEGMENTER_ID
    infer: str = "never"
    device: str | None = None
    synthetic: bool = False
    command: tuple = ()
    allow_staging: bool = False
    formal_registry: Path | None = None

    def __post_init__(self):
        if self.infer not in INFER_MODES:
            raise InputValidationError(f"infer: one of {INFER_MODES}")

    def record(self):
        record = dict(root=str(self.root), registry_path=str(self.registry_path), allowlist_path=str(self.allowlist_path),
                      models_root=str(self.models_root), image_root=str(self.image_root),
                      groups_path=str(self.groups_path), grouping=self.grouping, segmenter_id=self.segmenter_id,
                      infer=self.infer, device=self.device, synthetic=self.synthetic, allow_staging=self.allow_staging)
        if self.formal_registry is not None:
            record["formal_registry"] = str(self.formal_registry)
        return record


def registry_git_state(path):
    """Git state of a registry file (review PS1_PS2 item b, A4 review item 3), never raising: in_work_tree, tracked
    (`git ls-files --error-unmatch`), clean (tracked and `git status --porcelain -- <file>` empty: no staged,
    unstaged or untracked change), the work tree, HEAD and the last commit that touched the file. Outside a git work
    tree (e.g. the training chain's results directory or a snapshot) tracked and clean are False."""
    path = Path(path).resolve()

    def git(*args):
        return subprocess.run(["git", "-C", str(path.parent), *args], capture_output=True, text=True, timeout=60)

    try:
        top = git("rev-parse", "--show-toplevel")
        if top.returncode != 0:
            return dict(in_work_tree=False, tracked=False, clean=False, note="not inside a git work tree")
        tracked = git("ls-files", "--error-unmatch", "--", path.name).returncode == 0
        status = git("status", "--porcelain", "--", path.name)
        head = git("rev-parse", "HEAD")
        last = git("log", "-1", "--format=%H", "--", path.name) if tracked else None
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(in_work_tree=None, tracked=False, clean=False, error=f"{type(exc).__name__}: {exc}")
    porcelain = status.stdout.rstrip() if status.returncode == 0 else None          # leading XY columns kept
    return dict(in_work_tree=True, tracked=tracked, clean=bool(tracked and porcelain == ""),
                porcelain=porcelain, work_tree=top.stdout.strip(),
                head=head.stdout.strip() if head.returncode == 0 else None,
                last_commit=(last.stdout.strip() or None) if last is not None and last.returncode == 0 else None)


def registry_status(path, repo_registry=None):
    """"registered" for the repository registry configs/bayesreg/segmenters.json (`repo_registry`, default
    REPO_REGISTRY), otherwise "staging_unregistered" (未正式登记: the staging copy of the training chain awaiting
    review); `git` = registry_git_state(path) (recorded; the status itself depends on the path only)."""
    named = repo_registry is not None
    repo_registry = REPO_REGISTRY if repo_registry is None else repo_registry
    formal = Path(path).resolve() == Path(repo_registry).resolve()
    return dict(status=REGISTERED if formal else STAGING, repository_registry=str(repo_registry),
                note=(f"正式登记（{repo_registry}）" if named else "正式登记（configs/bayesreg/segmenters.json）") if formal else
                     "未正式登记：暂存登记表（训练链条写出，待总负责人复核后登记进 configs/bayesreg/segmenters.json）",
                git=registry_git_state(path))


def is_formal_registry(path):
    """`path` is one of FORMAL_REGISTRIES (compared as resolved paths)."""
    return Path(path).resolve() in {Path(p).resolve() for p in FORMAL_REGISTRIES}


def open_registry(registry_path, *, allow_staging=False, formal_registry=None):
    """Check 1 of the module docstring (shared by StructureLayer -- run_pair, experiments/d0/make_inputs.py -- and
    bayesreg.learn.cli, review PS1_PS2 item c): the file exists; a registry other than the repository one is
    refused unless `allow_staging`, and then needs the ensemble.json check; the schema (load_segmenter_registry).
    formal_registry: the repository registry of this call when it is not REPO_REGISTRY (StructureSource); refused
    unless it is one of FORMAL_REGISTRIES.
    Returns (SegmenterRegistry, record) with the record of provenance.json channels.structure.registry; refusals
    are ProvenanceRejected."""
    if formal_registry is not None and not is_formal_registry(formal_registry):
        _reject(f"formal_registry {formal_registry} is not one of the repository registries "
                f"{[str(p) for p in FORMAL_REGISTRIES]}")
    if not Path(registry_path).is_file():
        _reject(f"no segmenter registry at {registry_path}")
    status = registry_status(registry_path, formal_registry)
    if status["status"] == STAGING and not allow_staging:
        _reject(f"segmenter registry {registry_path} is not the repository registry "
                f"{status['repository_registry']} (staging_unregistered, 未正式登记): refused unless "
                "--structure-registry-staging is given (development diagnostics only)")
    registry = load_segmenter_registry(registry_path)                          # ProvenanceRejected if malformed
    ensemble = check_staging_ensemble(registry_path, registry) if status["status"] == STAGING else None
    return registry, dict(path=registry.path, sha256=registry.sha256, **status, staging_opt_in=bool(allow_staging),
                          staging_ensemble_check=ensemble,
                          use="registered" if ensemble is None else "development diagnostics only (未正式登记)")


def registry_label(record, *, path=None):
    """The short registration label of status.json structure_registry (run_pair, learn.cli): status, path,
    staging_opt_in, development_diagnostic_only and the git tracked / clean flags, from an open_registry record (or
    a registry_status record plus `path`)."""
    git = record.get("git") or {}
    return dict(status=record["status"], path=str(path if path is not None else record["path"]),
                staging_opt_in=bool(record.get("staging_opt_in")),
                development_diagnostic_only=record["status"] == STAGING,
                git=dict(tracked=git.get("tracked"), clean=git.get("clean")))


def check_staging_ensemble(registry_path, registry):
    """A staging registry (review PS item 2): ensemble.json beside it, as bayesreg.seg.unet.registry.register writes
    it, must record registry_sha256 equal to the registry file's sha256 and member_check.passed true; else
    ProvenanceRejected. Returns the record."""
    path = Path(registry_path).parent/ENSEMBLE_FILE
    if not path.is_file():
        _reject(f"staging registry {registry_path}: no {ENSEMBLE_FILE} beside it (written by the registration "
                "with its member check)")
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _reject(f"staging registry: {path} is not UTF-8 JSON ({exc})")
    if not isinstance(document, Mapping):
        _reject(f"staging registry: {path} is not a JSON object")
    if document.get("registry_sha256") != registry.sha256:
        _reject(f"staging registry: {path} records registry_sha256 {document.get('registry_sha256')!r}, the registry "
                f"file {registry_path} has {registry.sha256}")
    check = document.get("member_check")
    if not isinstance(check, Mapping) or check.get("passed") is not True:
        _reject(f"staging registry: {path} member_check.passed is not true")
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), registry_sha256_matches=True,
                member_check_passed=True, member_check_format=check.get("format"),
                ensemble_registry_path=document.get("registry_path"))


def expected_models(registry, subject, segmenter_id=SEGMENTER_ID):
    """(sorted model ids, policy) that may segment `subject`: the registered fold models of `segmenter_id`
    whose label contact does not contain it; policy all_folds_mean when that is every fold model, else
    cross_fit_other_folds (the selection of bayesreg.seg.unet.infer.select_fold_models, D24). all_folds_mean
    needs at least 2 registered folds (review PS item 5: a hand-written one-fold registry is refused)."""
    models = {m: e for m, e in registry.models.items() if e["segmenter_id"] == segmenter_id}
    if not models:
        _reject(f"no registered fold model of {segmenter_id!r} in {registry.path}")
    usable = sorted(m for m, e in models.items() if subject not in e["label_contact_subjects"])
    if not usable:
        _reject(f"every fold model of {segmenter_id!r} saw the labels of {subject}")
    policy = "all_folds_mean" if len(usable) == len(models) else "cross_fit_other_folds"
    folds = sorted({e["fold"] for e in models.values()})
    if policy == "all_folds_mean" and len(folds) < 2 and segmenter_id not in SINGLE_MODEL_SEGMENTERS:
        _reject(f"all_folds_mean for {subject} needs at least 2 registered folds of {segmenter_id!r}; {registry.path} "
                f"has fold(s) {folds}")
    return usable, policy


def read_structure_groups(path, n_channels, shape):
    """structure_groups.npz key "p" -> (float64 (n_channels,) + shape divided by the channel sum, record) through
    bayesreg.seg.channels.read_back_probabilities, the read-back shared with bayesreg.seg.unet.infer.read_probabilities
    and bayesreg.learn.cli.load_structure_layer (without torch). Its refusals -- missing key, a storage type outside
    RAW_SUM_TOL, wrong shape, non-finite or negative values, a voxel with zero total, and -- before the division -- a
    voxel whose raw channel sum differs from 1 by more than RAW_SUM_TOL[storage type] (C03.2-4a, D39: 1e-3 for
    float16, 1e-6 for float32 / float64) -- are ProvenanceRejected here."""
    try:
        return read_back_probabilities(path, shape=(n_channels,)+tuple(shape))
    except InputValidationError as exc:
        _reject(str(exc))


class StructureLayer:
    """The structure layer of one pair: checks 0-2 at construction, `load(side)` checks 3-4 (and the inference
    when needed), `record()` the pair-level record for provenance.json."""

    def __init__(self, pair, source, *, log=None, allowlist=None):
        """`allowlist` (C23.4-8, D103): required for a pair with a sealed subject -- the very bound allowlist load_pair
        admitted this pair object through (bayesreg.io.pair.recheck_sealed_pair, before anything else); otherwise None
        (the allowlist is read from source.allowlist_path, as before) or a DevAllowlist with load_pair's sha256."""
        if not isinstance(pair, PairInput):
            raise InputValidationError("pair: the PairInput returned by load_pair is required")
        if not isinstance(source, StructureSource):
            raise InputValidationError("source: a StructureSource is required")
        sealed_pair = recheck_sealed_pair(pair, allowlist, where="structure layer")
        self.pair, self.source, self.log = pair, source, log or (lambda *_: None)
        try:
            self.grouping = load_grouping(source.groups_path, source.grouping)
            self.grouping_check = require_inference_grouping(self.grouping)       # no override in run_pair
        except GroupingUnverified as exc:
            _reject(f"grouping_unverified: {exc}")
        except (ValueError, OSError) as exc:                               # InputValidationError, bad JSON, no file
            _reject(f"grouping {source.grouping!r} of {source.groups_path}: {exc}")
        self.registry, self.registry_record = open_registry(source.registry_path, allow_staging=source.allow_staging,
                                                            formal_registry=source.formal_registry)
        if allowlist is None and sealed_pair is None:
            self.allowlist = load_dev_allowlist(source.allowlist_path)
        elif isinstance(allowlist, DevAllowlist):
            self.allowlist = allowlist
        else:
            raise InputValidationError("structure layer: allowlist must be a DevAllowlist or None")
        if self.allowlist.sha256 != pair.provenance["subject_guard"]["allowlist_sha256"]:
            raise InputValidationError("structure layer: the allowlist differs from the one load_pair used")
        self.subjects = dict(fixed=pair.fixed_subject, moving=pair.moving_subject)
        self.expected = {side: expected_models(self.registry, self.subjects[side], source.segmenter_id)
                         for side in SIDES}
        self.preflight = self._pair_contact_check()

    def _pair_contact_check(self):
        """D24 pair level before any prediction is read or made: no model used for either image saw either subject."""
        pair_subjects = set(self.subjects.values())
        used = sorted({m for ids, _ in self.expected.values() for m in ids})
        for model_id in used:
            overlap = sorted(set(self.registry.models[model_id]["label_contact_subjects"]) & pair_subjects)
            if overlap:
                _reject(f"D24 pair-level label contact: model {model_id!r} (needed for "
                        f"{[s for s, (ids, _) in self.expected.items() if model_id in ids]}) saw the labels of "
                        f"{overlap}; a cross-fold training pair cannot be registered with the structure layer")
        return dict(check="D24 pair-level label contact (preflight)", pair_subjects=[self.pair.fixed_subject,
                    self.pair.moving_subject], models_used=used,
                    expected={side: dict(model_ids=ids, policy=policy)
                              for side, (ids, policy) in self.expected.items()},
                    contact_union_contains_pair_subject=False)

    # ---- one image -------------------------------------------------------------------------------------

    def load(self, side):
        """(P_S (K_S + 1, nz, ny, nx) float64 divided by the channel sum, AcceptedProvenance, record) of one image."""
        if side not in SIDES:
            raise InputValidationError(f"side: one of {SIDES}")
        subject = self.subjects[side]
        directory = Path(self.source.root)/subject
        path = directory/PROVENANCE_FILE
        generated = None
        if not path.is_file():
            generated = self._generate(side, subject, directory)
        raw = path.read_bytes()
        try:
            prov = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _reject(f"{side}: {path} is not UTF-8 JSON ({exc})")
        accepted = validate_segmentation_provenance(prov, self.pair, side=side, layer="structure",
                                                    registry=self.registry, allowlist=self.allowlist)
        models = self._check_models(side, subject, prov, accepted)
        grouping = self._check_grouping(side, prov)
        self._check_grid(side, prov)
        verified = verify_segmentation_files(prov, directory)
        if GROUPS_FILE not in verified:
            _reject(f"{side}: {GROUPS_FILE} is not among the hash-verified outputs")
        array, read_back = read_structure_groups(directory/GROUPS_FILE, self.grouping.k+1, self.pair.grid.shape)
        environment = prov.get("environment") if isinstance(prov.get("environment"), Mapping) else {}
        record = dict(accepted=dict(accepted), directory=str(directory),
                      provenance_sha256=hashlib.sha256(raw).hexdigest(), verified_outputs=verified,
                      generated_in_this_run=generated is not None and generated["generated_here"], inference=generated,
                      models=models, grouping=grouping, read_back=read_back,
                      prediction_registry=dict(path=prov.get("registry_path"), sha256=prov.get("registry_sha256")),
                      prediction_code=environment.get("code"), prediction_runtime_seconds=prov.get("runtime_seconds"))
        return array, accepted, record

    def _check_models(self, side, subject, prov, accepted):
        ids, policy = self.expected[side]
        if prov.get("segmenter_id") != self.source.segmenter_id:
            _reject(f"{side}: segmenter_id {prov.get('segmenter_id')!r} != {self.source.segmenter_id!r}")
        if prov.get("subject") != subject:
            _reject(f"{side}: the prediction is of {prov.get('subject')!r}, not {subject!r}")
        if sorted(accepted["model_ids"]) != ids:
            _reject(f"{side}: D24 model selection: {subject} must be segmented by {ids} ({policy}), the prediction "
                    f"used {sorted(accepted['model_ids'])}")
        if prov.get("policy") != policy:
            _reject(f"{side}: policy {prov.get('policy')!r} != {policy!r} (D24)")
        members = prov.get("members")
        if not isinstance(members, list) or not all(isinstance(m, Mapping) for m in members):
            _reject(f"{side}: members: a list of member records required")
        got = sorted((m.get("model_id"), m.get("member_sha256")) for m in members)
        want = sorted((m, sha) for m in ids for sha in self.registry.models[m]["member_model_sha256"])
        if got != want:
            _reject(f"{side}: members must be every registered seed member of {ids} (model id, sha256); the "
                    f"prediction lists {len(got)}, the registry {len(want)}")
        folds = {m: self.registry.models[m]["fold"] for m in ids}
        if prov.get("folds") != folds:
            _reject(f"{side}: folds {prov.get('folds')} != the registry {folds}")
        return dict(check="D24 model selection and member sha256 against the registry", model_ids=ids, policy=policy,
                    folds=folds, members=[dict(model_id=m, member_sha256=sha) for m, sha in want])

    def _check_grouping(self, side, prov):
        g = self.grouping
        want = dict(grouping=g.name, class_names=list(g.group_names), background_labels=list(g.background_labels),
                    group_labels={n: list(labels) for n, labels in zip(g.group_names, g.group_labels)})
        for key, value in want.items():
            if prov.get(key) != value:
                _reject(f"{side}: grouping field {key!r} of the prediction differs from {g.name!r}")
        status, check = prov.get("grouping_status"), prov.get("grouping_check")
        if status not in INFERENCE_STATUSES or not isinstance(check, Mapping) \
                or check.get("grouping_status_admitted") is not True \
                or check.get("allow_unverified_grouping") is not False:
            _reject(f"grouping_unverified: {side}: the prediction was made with grouping status {status!r} "
                    f"(check {check}); only {INFERENCE_STATUSES} without override are admitted")
        recorded = prov.get("groups_sha256")
        return dict(grouping=g.name, grouping_status=status, K_S=g.k, prediction_groups_sha256=recorded,
                    current_groups_sha256=g.source_sha256, same_groups_file=recorded == g.source_sha256)

    def _check_grid(self, side, prov):
        grid, meta = prov.get("grid"), self.pair.grid_metadata
        keys = ("shape_zyx", "spacing_xyz", "origin_xyz")
        if not isinstance(grid, Mapping) or any(not isinstance(grid.get(k), list) or list(grid[k]) != list(meta[k])
                                                for k in keys):
            _reject(f"{side}: the structure layer is not on the pair's grid ({ {k: meta[k] for k in keys} })")

    # ---- inference of a missing prediction --------------------------------------------------------------

    def _inference_allowlist(self, side):
        from .unet.pins import (PinnedInputRefused, check_synthetic_data_root, check_synthetic_subjects,
                                load_pinned_allowlist)
        from .unet.snapshot import CodeNotClean
        try:
            if self.source.synthetic:
                check_synthetic_data_root(self.source.image_root)
                check_synthetic_subjects([s for ids in self.allowlist.subjects.values() for s in ids], "the allowlist")
                return self.allowlist, dict(mode="synthetic", allowlist_sha256=self.allowlist.sha256)
            allowlist, record = load_pinned_allowlist(self.source.allowlist_path)
        except (PinnedInputRefused, CodeNotClean) as exc:
            _reject(f"{side}: structure-layer inference refused before reading any image ({type(exc).__name__}: {exc})")
        if allowlist.sha256 != self.allowlist.sha256:
            _reject(f"{side}: the pinned allowlist differs from the one load_pair used")
        return allowlist, dict(mode="pinned", allowlist_sha256=allowlist.sha256, code=record.get("code"),
                               allowlist_sources_verified=sorted(record.get("allowlist_sources_verified", {})))

    def _generate(self, side, subject, directory):
        if self.source.infer != "missing":
            _reject(f"{side}: no structure-layer prediction at {directory} (infer = {self.source.infer!r})")
        allowlist, admission = self._inference_allowlist(side)
        root = Path(self.source.root)
        root.mkdir(parents=True, exist_ok=True)
        with open(root/f".{subject}.lock", "a") as lock:          # one inference per subject at a time
            fcntl.flock(lock, fcntl.LOCK_EX)
            # both checks under the lock (review PS item 6): a concurrent run may still be writing the directory
            if (directory/PROVENANCE_FILE).is_file():               # made by a concurrent run meanwhile
                return dict(made_by="a concurrent run (found after the lock)", generated_here=False,
                            admission=admission)
            if directory.exists() and any(directory.iterdir()):
                _reject(f"{side}: {directory} exists without {PROVENANCE_FILE} (an incomplete prediction is kept, "
                        "never overwritten; the lead decides)")
            import torch
            from .unet.data import OasisFiles
            from .unet.infer import ModelSelectionRefused, segment_subject
            device = self.source.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.log(f"structure layer: inferring {subject} ({side}) on {device} with {self.expected[side][0]}")
            tick = time.perf_counter()
            try:
                prov = segment_subject(subject, files=OasisFiles(Path(self.source.image_root)), registry=self.registry,
                                       allowlist=allowlist, models_root=self.source.models_root,
                                       grouping=self.grouping, pred_root=root, device=device,
                                       segmenter_id=self.source.segmenter_id, command=list(self.source.command),
                                       log=self.log)
            except (ModelSelectionRefused, GroupingUnverified, SubjectSealed, FileExistsError) as exc:
                _reject(f"{side}: structure-layer inference refused ({type(exc).__name__}: {exc})")
        return dict(made_by="this run (bayesreg.seg.unet.infer.segment_subject)", generated_here=True, device=device,
                    seconds=time.perf_counter()-tick, policy=prov["policy"], model_ids=prov["model_ids"],
                    admission=admission)

    def record(self):
        """Pair-level record; run_pair.prepare_channels adds background_channel from the pyramid's provenance."""
        return dict(contract=CONTRACT, source=self.source.record(), registry=self.registry_record,
                    grouping=dict(self.grouping.record(), grouping_check=self.grouping_check), preflight=self.preflight,
                    allowlist_sha256=self.allowlist.sha256)


__all__ = ["DEFAULT_GROUPING", "ENSEMBLE_FILE", "FORMAL_REGISTRIES", "GROUPS_FILE", "INFER_MODES", "PROVENANCE_FILE",
           "RAW_SUM_TOL", "REGISTERED", "REPO_REGISTRY", "SEGMENTER_ID", "STAGING", "StructureLayer", "StructureSource",
           "check_staging_ensemble", "expected_models", "is_formal_registry", "open_registry", "read_structure_groups",
           "registry_git_state", "registry_label", "registry_status"]
