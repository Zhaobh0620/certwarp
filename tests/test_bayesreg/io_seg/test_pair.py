"""C03.1: label firewall and subject guard of load_pair; per-image segmentation provenance with the
per-fold-model registry, hash binding and the posterior-file check (synthetic files only)."""
import copy
import json

import numpy as np
import pytest

from bayesreg.io import (AcceptedProvenance, ProvenanceRejected, SubjectSealed, load_dev_allowlist, load_pair,
                         load_segmenter_registry, validate_segmentation_provenance, verify_segmentation_files)
from bayesreg.io import guard as guard_module
from bayesreg.io import pair as pair_module
from bayesreg.seg import normalize_intensity
from jd_curl.three_d.nifti import file_sha256
from jd_curl.validation import InputValidationError

from ._helpers import (DEV, FOLD0, FOLD1, HEX, LPBA, SEALED, SHAPE, SYN_FOLD0, SYN_FOLD1, T0, allowlist_document, is_synthetic,
                       model_entry, registry_document, source_sha, synthetic_fold_sources, synthetic_folds,
                       synthetic_registry_models, write_json, write_pair)

WHITELIST = ("fixed", "moving", "initial_u")


@pytest.fixture
def allowlist(tmp_path):
    return load_dev_allowlist(write_json(tmp_path/"dev_subjects.json", allowlist_document()))


class GuardedNpz:
    """NpzFile stand-in: any access outside the whitelist (key, iteration, listing) raises."""

    def __init__(self, npz):
        self._npz, self.items_read, self.queried = npz, [], []

    def _check(self, key):
        if key not in WHITELIST or "seg" in key:
            raise AssertionError(f"label firewall breach: {key!r}")

    def __getitem__(self, key):
        self._check(key)
        self.items_read.append(key)
        return self._npz[key]

    def __contains__(self, key):
        self._check(key)
        self.queried.append(key)
        return key in self._npz

    def __iter__(self):
        raise AssertionError("label firewall breach: iteration")

    def __len__(self):
        raise AssertionError("label firewall breach: len")

    def __getattr__(self, name):                     # files, keys, items, values, get, f, zip, ...
        raise AssertionError(f"label firewall breach: attribute {name!r}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._npz.close()
        return False


@pytest.fixture
def guarded(monkeypatch):
    opened = []

    def open_guarded(path):
        opened.append(GuardedNpz(np.load(path, allow_pickle=False)))
        return opened[-1]
    monkeypatch.setattr(pair_module, "_open_npz", open_guarded)
    return opened


def test_guard_is_not_vacuous(tmp_path):
    path, _ = write_pair(tmp_path)
    with GuardedNpz(np.load(path)) as guard:
        for breach in (lambda: guard["fixed_seg4"], lambda: guard["moving_seg35"], lambda: guard["fixed_lpba56"],
                       lambda: list(guard), lambda: dict(guard), lambda: guard.files, lambda: guard.keys(),
                       lambda: "fixed_seg4" in guard):
            with pytest.raises(AssertionError, match="firewall"):
                breach()


def test_load_pair_reads_only_whitelisted_members(tmp_path, guarded, allowlist):
    path, arrays = write_pair(tmp_path/"pair")
    loaded = load_pair(path, allowlist=allowlist)
    assert guarded[0].items_read == ["fixed", "moving", "initial_u"]
    assert set(guarded[0].queried) <= set(WHITELIST)
    assert np.array_equal(loaded.fixed, arrays["fixed"]) and np.array_equal(loaded.moving, arrays["moving"])
    assert loaded.fixed.dtype == np.float64 and not loaded.fixed.flags.writeable and not loaded.moving.flags.writeable
    assert loaded.grid.shape == SHAPE and loaded.grid.spacing == (1., 1.2, .8) and loaded.grid.unit == "mm"
    assert [float(a[0]) for a in loaded.grid.axes] == [-3., 2., .5]
    assert loaded.grid_metadata == dict(shape_zyx=list(SHAPE), spacing_xyz=[1., 1.2, .8], origin_xyz=[-3., 2., .5], unit="mm")
    assert (loaded.fixed_subject, loaded.moving_subject) == (DEV[0], DEV[1])
    assert dict(loaded.source_sha256) == dict(fixed=source_sha(DEV[0]), moving=source_sha(DEV[1]))
    with pytest.raises(TypeError):
        loaded.source_sha256["fixed"] = HEX("other")
    provenance = loaded.provenance
    assert provenance["keys_read"] == ["fixed", "moving", "initial_u"]
    assert provenance["pair_sha256"] == file_sha256(path) and provenance["omega0"] == loaded.grid.omega
    guard = provenance["subject_guard"]
    assert guard["allowlist_sha256"] == allowlist.sha256 and guard["datasets"] == dict(fixed="OASIS", moving="OASIS")
    assert guard["unseal_token_presented"] is False and guard["unsealed"] is False
    fields = set(vars(loaded))
    assert fields == {"fixed", "moving", "grid", "grid_metadata", "fixed_subject", "moving_subject", "source_sha256",
                      "provenance"}
    text = json.dumps({k: v for k, v in provenance.items() if k not in ("pair_path", "manifest_path")})
    assert not any(word in text for word in ("seg4", "seg35", "lpba56", "macro_dice", "labels_policy", "baseline"))


def test_allowlist_given_as_path(tmp_path):
    path, _ = write_pair(tmp_path/"pair")
    listing = write_json(tmp_path/"dev_subjects.json", allowlist_document())
    loaded = load_pair(path, allowlist=str(listing))
    assert loaded.provenance["subject_guard"]["allowlist_sha256"] == file_sha256(listing)
    with pytest.raises(InputValidationError):
        load_pair(path, allowlist=None)


def test_real_npzfile_sees_only_whitelisted_keys(tmp_path, monkeypatch, allowlist):
    """Without the stand-in: spy on the real NpzFile.__getitem__ (dict(npz) or iteration would hit it too)."""
    path, _ = write_pair(tmp_path/"pair")
    seen, real = [], np.lib.npyio.NpzFile.__getitem__

    def spy(self, key):
        seen.append(key)
        return real(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", spy)
    load_pair(path, allowlist=allowlist)
    assert seen == ["fixed", "moving", "initial_u"]


def test_manifest_is_decoded_through_a_whitelist(tmp_path):
    write_pair(tmp_path)
    decoded = json.loads((tmp_path/"manifest.json").read_text(), object_pairs_hook=pair_module._keep_whitelisted_keys)
    assert set(decoded) == {"fixed_subject", "moving_subject", "shape_zyx", "spacing_xyz", "origin_xyz", "unit",
                            "sources", "pair_sha256"}           # dataset, pair, weights, baseline_*, labels_* gone
    assert set(decoded["sources"]) == {"fixed", "moving"}                # fixed_seg4, fixed_lpba56 are gone
    assert decoded["sources"]["fixed"] == dict(sha256=source_sha(DEV[0]))
    _, _, selected = pair_module._read_manifest(tmp_path/"manifest.json")
    assert selected == dict(pair_sha256=file_sha256(tmp_path/"pair.npz"), fixed_subject=DEV[0], moving_subject=DEV[1],
                            source_sha256=dict(fixed=source_sha(DEV[0]), moving=source_sha(DEV[1])))
    with pytest.raises(InputValidationError, match="duplicate"):
        json.loads('{"unit": "mm", "unit": "mm"}', object_pairs_hook=pair_module._keep_whitelisted_keys)


def test_initial_u_optional_but_must_be_zero(tmp_path, guarded, allowlist):
    path, _ = write_pair(tmp_path/"a", initial_u=False)
    loaded = load_pair(path, allowlist=allowlist)
    assert loaded.provenance["initial_u_present"] is False and guarded[0].items_read == ["fixed", "moving"]
    path, _ = write_pair(tmp_path/"b", initial_value=1e-12)
    with pytest.raises(InputValidationError, match="initial_u"):
        load_pair(path, allowlist=allowlist)


@pytest.mark.parametrize("case", ["unit", "sha", "sha_missing", "sha_upper", "source_missing", "source_sha_bad",
                                  "subject_missing", "missing_moving", "missing_manifest"])
def test_load_pair_rejections(tmp_path, guarded, allowlist, case):
    changes = dict(sha=dict(pair_sha256=HEX("other")), sha_missing=dict(pair_sha256=None),
                   source_missing=dict(sources=dict(fixed=dict(sha256=source_sha(DEV[0])))),
                   source_sha_bad=dict(sources=dict(fixed=dict(sha256=source_sha(DEV[0])), moving=dict(sha256="abc"))),
                   subject_missing=dict(moving_subject=None)).get(case)
    if case == "unit":
        path, _ = write_pair(tmp_path, unit="voxel")
    elif case == "missing_moving":
        path, _ = write_pair(tmp_path, drop=("moving",))
    else:
        path, _ = write_pair(tmp_path, manifest_changes=changes)
    if case == "sha_upper":
        manifest = json.loads((tmp_path/"manifest.json").read_text())
        write_json(tmp_path/"manifest.json", dict(manifest, pair_sha256=manifest["pair_sha256"].upper()))
    if case == "missing_manifest":
        (tmp_path/"manifest.json").unlink()
        with pytest.raises(FileNotFoundError):
            load_pair(path, allowlist=allowlist)
        return
    with pytest.raises(InputValidationError):
        load_pair(path, allowlist=allowlist)


# ---- subject guard inside load_pair -----------------------------------------------------------

@pytest.mark.parametrize("fixed, moving", [
    (SEALED, DEV[1]), (DEV[0], SEALED),                                   # canonical, not allowlisted
    ("S37", "S01"), ("S01", "S40"),                                       # LPBA40 H2 test split
    ("OAS1_0438_MR1", DEV[1]), ("oasis_oas1_0438_mr1", DEV[1]),           # non-canonical spellings of DEV[0]
    (DEV[0]+" ", DEV[1]), (DEV[0]+"\n", DEV[1]), ("S1", "S02"), ("s01", "S02"),
    ("IBSR_01", "IBSR_02"), ("", DEV[1])])
def test_sealed_or_non_canonical_subjects_are_refused_before_the_npz(tmp_path, guarded, allowlist, fixed, moving):
    path, _ = write_pair(tmp_path, fixed_subject=fixed, moving_subject=moving)
    with pytest.raises(SubjectSealed) as info:
        load_pair(path, allowlist=allowlist)
    assert info.value.status == "subject_sealed" and guarded == []


def test_lpba40_dev_pair_and_t0_pair_load(tmp_path, allowlist):
    path, _ = write_pair(tmp_path/"lpba", fixed_subject="S01", moving_subject="S02")
    loaded = load_pair(path, allowlist=allowlist)
    assert loaded.provenance["subject_guard"]["datasets"] == dict(fixed="LPBA40", moving="LPBA40")
    path, _ = write_pair(tmp_path/"lpba_test", fixed_subject="S01", moving_subject="S36")     # C26.10: the expected tightening (C26.3)
    with pytest.raises(SubjectSealed):
        load_pair(path, allowlist=allowlist)
    verdict = guard_module.check_subject("S36", allowlist)
    assert verdict.step == "sealed" and verdict.item == "lpba40 test set"
    path, _ = write_pair(tmp_path/"t0", fixed_subject=T0[0], moving_subject=T0[1])
    loaded = load_pair(path, allowlist=allowlist)                         # tuning pairs may be registered
    assert loaded.provenance["subject_guard"]["never_for_training"] == dict(fixed=True, moving=True)


def test_unseal_token_is_only_recorded(tmp_path, guarded, allowlist):
    path, _ = write_pair(tmp_path/"sealed", fixed_subject=SEALED)
    with pytest.raises(SubjectSealed, match="unsealing is not implemented"):
        load_pair(path, allowlist=allowlist, unseal_token="BZ-2026-10-24")
    assert guarded == []
    path, _ = write_pair(tmp_path/"dev")
    loaded = load_pair(path, allowlist=allowlist, unseal_token="BZ-2026-10-24")
    guard = loaded.provenance["subject_guard"]
    assert guard["unseal_token_presented"] is True and guard["unsealed"] is False
    assert guard["unseal_token_sha256"] == HEX("BZ-2026-10-24") and "BZ-2026" not in json.dumps(guard)
    with pytest.raises(InputValidationError):
        load_pair(path, allowlist=allowlist, unseal_token=123)


# ---- C03.1-2 provenance: fixtures ------------------------------------------------------------

@pytest.fixture
def setup(tmp_path, request):
    """allowlist, registry, and a pair factory (fixed, moving) -> PairInput.  With synthetic_folds("setup"): the synthetic
    training draw (C23.9-13 (8), revision 6e)."""
    synthetic = is_synthetic(request)
    listing = load_dev_allowlist(write_json(tmp_path/"dev_subjects.json", allowlist_document(synthetic_fold_sources() if synthetic else None)))
    registry_path = write_json(tmp_path/"segmenters.json", registry_document(synthetic_registry_models() if synthetic else None))
    registry = load_segmenter_registry(registry_path)
    assert registry.sha256 == file_sha256(registry_path)

    def make(fixed=DEV[0], moving=DEV[1]):
        path, _ = write_pair(tmp_path/f"{fixed}__{moving}", fixed_subject=fixed, moving_subject=moving)
        return load_pair(path, allowlist=listing)
    return listing, registry, make


def prov(layer, segmenter_id, pair, side="fixed", /, **changes):
    subject = pair.fixed_subject if side == "fixed" else pair.moving_subject
    record = dict(layer=layer, segmenter_id=segmenter_id, trained_with_labels=layer == "structure",
                  test_labels_seen=False, input_sha256={"aligned_norm.nii.gz": source_sha(subject)},
                  output_sha256={"posteriors01.nii.gz": HEX("p1"), "posteriors02.nii.gz": HEX("p2")})
    if layer == "tissue":
        record.update(model_ids=["ants-2.5.0-atropos"], class_names=["CSF", "GM", "WM"], wm_index=2)
    else:
        record.update(class_names=["cortex", "wm_struct", "ventricles", "deep_gm"])
    record.update(changes)
    return record


def structure_prov(pair, model_ids, registry, side="fixed", /, **changes):
    sid = registry.models[model_ids[0]]["segmenter_id"]
    fields = dict(model_ids=list(model_ids),
                  model_sha256={m: list(registry.models[m]["member_model_sha256"]) for m in model_ids})
    fields.update(changes)
    return prov("structure", sid, pair, side, **fields)


def check(record, pair, layer, setup, side="fixed", **kwargs):
    listing, registry, _ = setup
    return validate_segmentation_provenance(record, pair, side=side, layer=layer,
                                            registry=kwargs.get("registry", registry),
                                            allowlist=kwargs.get("allowlist", listing))


# ---- tissue layer ---------------------------------------------------------------------------

@pytest.mark.parametrize("side", ["fixed", "moving"])
def test_tissue_layer_accepts_atropos(setup, side):
    pair = setup[2]()
    record = check(prov("tissue", "atropos", pair, side), pair, "tissue", setup, side)
    assert isinstance(record, AcceptedProvenance) and record["layer"] == "tissue" and record["side"] == side
    assert record["trained_with_labels"] is False and record["test_labels_seen"] is False
    assert record["wm_index"] == 2 and record["wm_class_name"] == "WM"
    assert record["subject"] == (pair.fixed_subject if side == "fixed" else pair.moving_subject)
    assert "subjects_checked" not in record and "pair_subjects_checked" not in record
    assert json.loads(json.dumps(record)) == dict(record)
    with pytest.raises(TypeError):
        record["wm_index"] = 0
    with pytest.raises(TypeError):
        record.update(wm_index=0)
    assert copy.deepcopy(record) == record and isinstance(copy.deepcopy(record), AcceptedProvenance)


@pytest.mark.parametrize("names, index", [(["csf", "gm", "wm"], 2), (["CSF", "White_Matter", "GM"], 1),
                                          (["white_matter", "csf"], 0)])
def test_white_matter_names(setup, names, index):
    pair = setup[2]()
    record = check(prov("tissue", "atropos", pair, class_names=names, wm_index=index), pair, "tissue", setup)
    assert record["wm_index"] == index


@pytest.mark.parametrize("changes, reason", [
    (dict(segmenter_id="fast"), "pending BZ"), (dict(segmenter_id="spm"), "atropos"),
    (dict(wm_index=None), "Atropos"), (dict(wm_index=1), "not white matter"), (dict(wm_index=3), "wm_index"),
    (dict(wm_index=-1), "wm_index"), (dict(wm_index=True), "wm_index"), (dict(wm_index=2.), "wm_index"),
    (dict(class_names=None), "class_names"), (dict(class_names=["CSF", "csf", "WM"]), "duplicate"),
    (dict(class_names=["WM", "white_matter", "GM"], wm_index=0), "more than one"),
    (dict(class_names=["CSF", "G M", "WM"]), "class_names"), (dict(model_ids=None), "model_ids"),
    (dict(model_ids=[]), "model_ids"), (dict(model_ids=["atropos 2.5"]), "model_ids")])
def test_tissue_provenance_rejections(setup, changes, reason):
    pair = setup[2]()
    with pytest.raises(ProvenanceRejected, match=reason):
        check(prov("tissue", "atropos", pair, **changes), pair, "tissue", setup)
    record = prov("tissue", "atropos", pair)
    del record["wm_index"]
    with pytest.raises(ProvenanceRejected, match="wm_index"):
        check(record, pair, "tissue", setup)


@pytest.mark.parametrize("layer", ["tissue", "structure"])
@synthetic_folds("setup")
def test_input_hash_bound_to_manifest_source(setup, layer):
    listing, registry, make = setup
    pair = make()

    def record(**changes):
        if layer == "tissue":
            return prov("tissue", "atropos", pair, **changes)
        return structure_prov(pair, ["unet_s_v1_f0", "unet_s_v1_f1"], registry, **changes)
    check(record(), pair, layer, setup)
    for inputs in ({"aligned_norm.nii.gz": HEX("other")},                                 # another file
                   {"aligned_norm.nii.gz": source_sha(pair.moving_subject)},               # the other side
                   {"aligned_norm.nii.gz": source_sha(pair.fixed_subject), "mask.nii.gz": HEX("m")}):
        with pytest.raises(ProvenanceRejected, match="input_sha256"):
            check(record(input_sha256=inputs), pair, layer, setup)


def test_normalize_intensity_takes_the_accepted_record(setup):
    pair = setup[2]()
    record = check(prov("tissue", "atropos", pair), pair, "tissue", setup)
    shape = (12, 12, 12)
    pbar = np.zeros((4,)+shape)
    pbar[3] = .95                                                    # class index 2 = "WM" -> channel 3
    pbar[2] = .05
    image = np.random.default_rng(3).uniform(50, 150, shape)
    mask = np.ones(shape, dtype=bool)
    normalized, provenance = normalize_intensity(image, pbar, mask, record)
    assert provenance["normalization"] == "wm_median" and provenance["wm_class_name"] == "WM"
    assert abs(np.median(normalized)-1) <= 1e-15
    with pytest.raises(InputValidationError, match="accepted"):
        normalize_intensity(image, pbar, mask, dict(record))                               # plain dict
    with pytest.raises(InputValidationError, match="classes"):
        normalize_intensity(image, pbar[:3], mask, record)                                 # K_T mismatch


# ---- structure layer: per-fold registry, label contact ------------------------------------------

@synthetic_folds("setup")
def test_structure_layer_outside_training_uses_both_folds(setup):
    listing, registry, make = setup
    pair = make()
    record = check(structure_prov(pair, ["unet_s_v1_f0", "unet_s_v1_f1"], registry), pair, "structure", setup)
    assert record["layer"] == "structure" and record["trained_with_labels"] is True
    assert record["folds"] == {"unet_s_v1_f0": 0, "unet_s_v1_f1": 1}
    assert record["model_sha256"]["unet_s_v1_f1"] == list(registry.models["unet_s_v1_f1"]["member_model_sha256"])
    assert record["registry_sha256"] == registry.sha256 and record["allowlist_sha256"] == listing.sha256
    assert record["pair_subjects_checked"] == [pair.fixed_subject, pair.moving_subject]


@synthetic_folds("setup")
def test_cross_fitting_same_fold_passes_cross_fold_rejected(setup):
    _, registry, make = setup
    same_fold = make(SYN_FOLD0[0], SYN_FOLD0[1])             # both fold 0: segmented by the fold-1 model
    for side in ("fixed", "moving"):
        check(structure_prov(same_fold, ["unet_s_v1_f1"], registry, side), same_fold, "structure", setup, side)
    with pytest.raises(ProvenanceRejected, match="label contact"):
        check(structure_prov(same_fold, ["unet_s_v1_f0"], registry), same_fold, "structure", setup)
    cross = make(SYN_FOLD0[0], SYN_FOLD1[0])                 # the other fold's model saw one of the two subjects
    for model in ("unet_s_v1_f0", "unet_s_v1_f1"):
        for side in ("fixed", "moving"):
            with pytest.raises(ProvenanceRejected, match="label contact"):
                check(structure_prov(cross, [model], registry, side), cross, "structure", setup, side)


@pytest.mark.parametrize("model, reason", [("unet_t0_f0", "never_for_training"), ("unet_sealed_f0", "off the allowlist"),
                                           ("unet_saw_dev_f0", "label contact set")])
@synthetic_folds("setup")
def test_label_contact_rules(setup, model, reason):
    _, registry, make = setup
    pair = make()
    with pytest.raises(ProvenanceRejected, match=reason) as info:
        check(structure_prov(pair, [model], registry), pair, "structure", setup)
    assert info.value.status == "provenance_rejected"


def test_model_hashes_and_registration(setup):
    _, registry, make = setup
    pair = make()
    both = ["unet_s_v1_f0", "unet_s_v1_f1"]
    good = structure_prov(pair, both, registry)["model_sha256"]
    for hashes in ({"unet_s_v1_f0": good["unet_s_v1_f0"]},                                  # one model missing
                   dict(good, unet_s_v1_f0=good["unet_s_v1_f0"][::-1]),                     # member order
                   dict(good, unet_s_v1_f0=good["unet_s_v1_f0"][:1]),                       # a member missing
                   dict(good, unet_s_v1_f0=[HEX("x"), good["unet_s_v1_f0"][1]]), None):
        with pytest.raises(ProvenanceRejected, match="model_sha256"):
            check(structure_prov(pair, both, registry, model_sha256=hashes), pair, "structure", setup)
    with pytest.raises(ProvenanceRejected, match="not registered"):
        check(structure_prov(pair, both, registry, model_ids=both+["unet_s_v1_f2"]), pair, "structure", setup)
    with pytest.raises(ProvenanceRejected, match="registered for"):
        check(structure_prov(pair, ["unet_s_v1_f0"], registry, segmenter_id="unet_t0"), pair, "structure", setup)
    with pytest.raises(ProvenanceRejected, match="model_ids"):
        check(structure_prov(pair, both, registry, model_ids=both+both[:1]), pair, "structure", setup)


@pytest.mark.parametrize("layer, sid, slot", [
    ("tissue", "atropos", "structure"), ("structure", "unet_s_v1", "tissue"),          # wrong slot
    ("structure", "atropos", "structure"), ("structure", "fast", "structure")])          # structure needs a fold model
def test_layer_segmenter_combinations_rejected(setup, layer, sid, slot):
    _, registry, make = setup
    pair = make()
    record = structure_prov(pair, ["unet_s_v1_f0"], registry, segmenter_id=sid, layer=layer)
    if layer == "tissue":
        record = prov("tissue", sid, pair)
    with pytest.raises(ProvenanceRejected):
        check(record, pair, slot, setup)


@pytest.mark.parametrize("layer", ["tissue", "structure"])
@pytest.mark.parametrize("sid", ["truth", "seg4", "seg35", "oracle", "Truth", "SEG35", "my_Oracle_net", "unet_seg4_v2",
                                 "GroundTruth"])
def test_label_sources_rejected_by_substring(setup, layer, sid):
    """Forbidden as segmenter_id or model id (any substring, any case); the registry cannot hold them
    either (test_registry_malformed: model_id_oracle, segmenter_truth)."""
    _, registry, make = setup
    pair = make()
    record = structure_prov(pair, ["unet_s_v1_f0"], registry, segmenter_id=sid) if layer == "structure" \
        else prov("tissue", sid, pair)
    with pytest.raises(ProvenanceRejected, match="label source"):
        check(record, pair, layer, setup)
    bad = structure_prov(pair, ["unet_s_v1_f0"], registry) if layer == "structure" else prov("tissue", "atropos", pair)
    with pytest.raises(ProvenanceRejected, match="model_ids"):
        check(dict(bad, model_ids=[sid]), pair, layer, setup)


@pytest.mark.parametrize("layer", ["tissue", "structure"])
@pytest.mark.parametrize("changes", [
    dict(layer=None), dict(segmenter_id=""), dict(segmenter_id=None), dict(segmenter_id="unet s v1"),
    dict(segmenter_id="unet_s_v1\n"), dict(segmenter_id="ünet"), dict(oracle=True), dict(oracle="yes"),
    dict(test_labels_seen=True), dict(test_labels_seen="false"), dict(test_labels_seen=0), dict(test_labels_seen=None),
    dict(input_sha256={}), dict(output_sha256=None), dict(output_sha256={"a.nii.gz": "abc"}),
    dict(output_sha256={"b.nii.gz": HEX("x").upper()}), dict(output_sha256={"../b.nii.gz": HEX("x")}),
    dict(output_sha256={"dir/b.nii.gz": HEX("x")})])
def test_malformed_provenance_rejected(setup, layer, changes):
    _, registry, make = setup
    pair = make()
    record = prov("tissue", "atropos", pair) if layer == "tissue" else \
        structure_prov(pair, ["unet_s_v1_f0", "unet_s_v1_f1"], registry)
    record.update(changes)
    with pytest.raises(ProvenanceRejected):
        check(record, pair, layer, setup)


@pytest.mark.parametrize("layer, value", [("tissue", True), ("tissue", 0), ("tissue", None),
                                          ("structure", False), ("structure", 1), ("structure", "true")])
def test_trained_with_labels_must_match_layer(setup, layer, value):
    _, registry, make = setup
    pair = make()
    record = prov("tissue", "atropos", pair) if layer == "tissue" else \
        structure_prov(pair, ["unet_s_v1_f0", "unet_s_v1_f1"], registry)
    with pytest.raises(ProvenanceRejected, match="trained_with_labels"):
        check(dict(record, trained_with_labels=value), pair, layer, setup)


@pytest.mark.parametrize("field", ["test_labels_seen", "trained_with_labels", "layer", "model_ids", "class_names"])
def test_required_fields_missing_rejected(setup, field):
    _, registry, make = setup
    pair = make()
    record = structure_prov(pair, ["unet_s_v1_f0", "unet_s_v1_f1"], registry)
    del record[field]
    with pytest.raises(ProvenanceRejected):
        check(record, pair, "structure", setup)


# ---- registry file -----------------------------------------------------------------------------

def broken_registries():
    good = model_entry("unet_s_v1", 0, FOLD0)
    cases = {}
    for name, fold in (("fold_bool", True), ("fold_negative", -1), ("fold_none", None), ("fold_text", "0"), ("fold_float", 0.)):
        cases[name] = {"m": dict(good, fold=fold)}
    cases["extra_key"] = {"m": dict(good, training_subjects=FOLD0)}
    cases["missing_key"] = {"m": {k: v for k, v in good.items() if k != "label_contact_subjects_sha256"}}
    cases["contact_sha"] = {"m": dict(good, label_contact_subjects=FOLD0[:3])}
    cases["contact_non_canonical"] = {"m": model_entry("unet_s_v1", 0, ["OAS1_0101_MR1"])}
    cases["contact_space"] = {"m": model_entry("unet_s_v1", 0, [FOLD0[0]+" "])}
    cases["contact_duplicate"] = {"m": model_entry("unet_s_v1", 0, FOLD0+FOLD0[:1])}
    cases["members_empty"] = {"m": dict(good, member_model_sha256=[])}
    cases["members_duplicate"] = {"m": dict(good, member_model_sha256=[HEX("a"), HEX("a")])}
    cases["members_upper"] = {"m": dict(good, member_model_sha256=[HEX("a").upper()])}
    cases["same_fold_twice"] = {"m": good, "n": model_entry("unet_s_v1", 0, FOLD1)}
    cases["model_id_space"] = {"m 1": good}
    cases["model_id_oracle"] = {"oracle_f0": good}
    cases["segmenter_truth"] = {"m": dict(good, segmenter_id="unet_truth")}
    cases["segmenter_atropos"] = {"m": dict(good, segmenter_id="atropos")}
    cases["no_models"] = {}
    return cases


@pytest.mark.parametrize("case", sorted(broken_registries()))
def test_registry_malformed(tmp_path, case):
    path = write_json(tmp_path/"segmenters.json", registry_document(broken_registries()[case]))
    with pytest.raises(ProvenanceRejected, match="registry|label_contact"):
        load_segmenter_registry(path)


def test_registry_document_level_rejections(tmp_path):
    for text in ('{"schema": "bayesreg.segmenters.v0", "models": {}}', '[]',
                 '{"schema": "bayesreg.segmenters.v1", "models": {"m": {}}, "models": {}}', 'not json'):
        (tmp_path/"r.json").write_text(text)
        with pytest.raises(ProvenanceRejected, match="registry"):
            load_segmenter_registry(tmp_path/"r.json")


# ---- posterior files ---------------------------------------------------------------------------

def test_verify_segmentation_files(tmp_path):
    folder = tmp_path/"posteriors"
    folder.mkdir()
    for k in (1, 2):
        (folder/f"posteriors0{k}.nii.gz").write_bytes(bytes([k])*100)
    outputs = {f"posteriors0{k}.nii.gz": file_sha256(folder/f"posteriors0{k}.nii.gz") for k in (1, 2)}
    record = dict(output_sha256=outputs)
    assert verify_segmentation_files(record, folder) == outputs
    single = dict(output_sha256={"posteriors01.nii.gz": outputs["posteriors01.nii.gz"]})
    assert verify_segmentation_files(single, folder/"posteriors01.nii.gz") == single["output_sha256"]
    with pytest.raises(ProvenanceRejected, match="single file"):
        verify_segmentation_files(record, folder/"posteriors01.nii.gz")
    with pytest.raises(ProvenanceRejected, match="does not exist"):
        verify_segmentation_files(record, tmp_path/"nowhere")
    (folder/"posteriors02.nii.gz").write_bytes(bytes([9])*100)
    with pytest.raises(ProvenanceRejected, match="differs"):
        verify_segmentation_files(record, folder)
    (folder/"posteriors02.nii.gz").unlink()
    with pytest.raises(ProvenanceRejected, match="missing"):
        verify_segmentation_files(record, folder)
    with pytest.raises(ProvenanceRejected, match="output_sha256"):
        verify_segmentation_files(dict(output_sha256={}), folder)


def test_arguments_validated(setup, tmp_path):
    listing, registry, make = setup
    pair = make()
    record = prov("tissue", "atropos", pair)
    with pytest.raises(InputValidationError):
        validate_segmentation_provenance(record, pair, side="fixed", layer="label")
    with pytest.raises(InputValidationError):
        validate_segmentation_provenance(record, pair, side="left", layer="tissue")
    with pytest.raises(InputValidationError):
        validate_segmentation_provenance(record, dict(fixed_subject=DEV[0]), side="fixed", layer="tissue")
    structure = structure_prov(pair, ["unet_s_v1_f0"], registry)
    with pytest.raises(InputValidationError, match="registry"):
        validate_segmentation_provenance(structure, pair, side="fixed", layer="structure", allowlist=listing)
    other = load_dev_allowlist(write_json(tmp_path/"other.json", dict(allowlist_document(), comment="x")))
    with pytest.raises(InputValidationError, match="allowlist"):
        validate_segmentation_provenance(structure, pair, side="fixed", layer="structure", registry=registry,
                                         allowlist=other)
    assert LPBA[0] == "S01" and T0[0] in listing.never_for_training
