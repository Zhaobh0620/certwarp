"""run_pair --channel-source structure (bayesreg.seg.structure_layer; contract C03.1-2/3, C03.2-4a, D24, D37, D39)
on synthetic files only (tmp_path; no data/, no real segmentation, no real subject's image).

- end to end on a 26^3 pair with synthetic segmenter outputs (structure_groups.npz float16 + provenance.json as
  bayesreg.seg.unet.infer writes them): the likelihood channels are sqrt of smooth -> floor of
  simplex_from_background_class(groups / sum) without the background channel, F_B evaluates at u = 0 with
  1 + K_S channels, provenance fields (test_labels_seen false, registry staging_unregistered, D24 preflight);
  run_pair.main up to level 0 writes them to provenance.json;
- leakage: a cross-fold training pair and a model that saw a pair subject are refused before any prediction is
  read; a same-fold training pair is accepted (other fold only); a prediction from the wrong models is refused;
- provenance: each required field and the bindings (model / member sha256, folds, grid, input and output hashes);
- grouping: an unverified grouping is refused, also when recorded in a prediction; a different grouping too;
- C03.2-4a: |sum - 1| > 1e-6 on one voxel and a zero-sum voxel are refused; the raw stored sum before the
  read-back normalization (review PS item 1): a float16 file with sum 0.7, or one voxel off by 2e-3, is refused, a
  float16 deviation within rounding (2^-11) is accepted; float32 / float64 files use 1e-6;
- registry (review PS items 2, 5): a staging registry needs the explicit opt-in and an ensemble.json beside it with
  the registry's sha256 and member_check.passed; the registration state reaches status.json and bayesreg.eval
  (tests/test_bayesreg/eval); a one-fold registry cannot give all_folds_mean;
- the generation of a missing prediction through bayesreg.seg.unet.infer.segment_subject with tiny CPU members
  (synthetic ids >= 0900), `--structure-infer never` (the default), the pinned-allowlist refusal of real-mode
  inference, and the checks under the per-subject lock (review PS item 6: a prediction finished by a concurrent run
  while waiting is read back with generated_in_this_run false; an incomplete directory is kept and refused).
"""
import fcntl
import types

import json
import logging

import numpy as np
import pytest
import torch

from jd_curl.three_d.nifti import file_sha256, save_scalar

from bayesreg.cli import run_pair
from bayesreg.io import ProvenanceRejected, load_pair, load_segmenter_registry
from bayesreg.model import ModelState, build_samplers, evaluate
from bayesreg.seg import simplex_from_background_class, tissue_channels
from bayesreg.seg import structure_layer
from bayesreg.seg.structure_layer import (ENSEMBLE_FILE, GROUPS_FILE, PROVENANCE_FILE, STAGING, StructureLayer,
                                          StructureSource, expected_models, read_structure_groups, registry_status)
from bayesreg.seg.unet.groups import load_grouping
from bayesreg.solve import eta_temporary

from ..io_seg._helpers import (DEV, FOLD0, FOLD1, SYN_FOLD0, SYN_FOLD1, T0, allowlist_document, is_synthetic, model_entry, source_sha,
                               synthetic_fold_sources, synthetic_folds, write_json)
from .test_run_pair import N, STAGE, write_synthetic_pair, write_tissue

LOG = logging.getLogger("test_run_pair_structure")
SEG_ID = "unet35_v1_ens"
GROUPING = load_grouping()                                  # structure13_v1, K_S = 13
K_S = GROUPING.k


def registry_models(fold0=FOLD0, fold1=FOLD1):
    return {"unet35_v1_foldA": model_entry(SEG_ID, 0, fold0), "unet35_v1_foldB": model_entry(SEG_ID, 1, fold1)}


_registry_models = registry_models                    # the function itself, for use_synthetic_folds


def structure_groups(grid, shift=(0., 0., 0.)):
    """(K_S + 1, nz, ny, nx): background = 1 - soft brain ellipsoid, the brain split among K_S smooth bumps."""
    X = grid.identity()
    c = (N-1)/2
    centre = c+np.asarray(shift)
    brain = 1/(1+np.exp((np.linalg.norm((X-centre)/c, axis=-1)-.8)/.05))
    rng = np.random.default_rng(7)
    weights = np.stack([np.exp(-np.sum((X-centre-d)**2, axis=-1)/(2*(.3*c)**2))+1e-3
                        for d in rng.uniform(-.5, .5, (K_S, 3))*c])
    return np.concatenate([(1-brain)[None], brain*weights/weights.sum(axis=0)])


def write_structure(root, grid, subject, models, *, shift=(0., 0., 0.), model_ids=None, grouping=GROUPING,
                    array=None, changes=None):
    """A prediction directory as bayesreg.seg.unet.infer.segment_subject writes it (structure_groups.npz only);
    `changes`: provenance fields replaced after the fact (None removes the field)."""
    directory = root/"structure"/subject
    directory.mkdir(parents=True)
    groups = structure_groups(grid, shift) if array is None else array
    np.savez_compressed(directory/GROUPS_FILE, p=groups.astype(np.float16))
    usable = sorted(m for m, e in models.items() if subject not in e["label_contact_subjects"])
    model_ids = usable if model_ids is None else model_ids
    members = [dict(model_id=m, fold=models[m]["fold"], seed=k, member_sha256=sha, temperature=1.2, best_iteration=3)
               for m in model_ids for k, sha in enumerate(models[m]["member_model_sha256"])]
    prov = dict(layer="structure", segmenter_id=SEG_ID, model_ids=model_ids,
                model_sha256={m: list(models[m]["member_model_sha256"]) for m in model_ids},
                folds={m: models[m]["fold"] for m in model_ids}, trained_with_labels=True, test_labels_seen=False,
                subject=subject, policy="all_folds_mean" if len(model_ids) == len(models) else "cross_fit_other_folds",
                members=members, **grouping.record(),
                grouping_check=dict(grouping_status=grouping.status, grouping_status_admitted=True,
                                    allow_unverified_grouping=False),
                input_sha256={"aligned_norm.nii.gz": source_sha(subject)},
                output_sha256={GROUPS_FILE: file_sha256(directory/GROUPS_FILE)},
                grid=dict(shape_zyx=list(grid.shape), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.]),
                registry_path="synthetic", registry_sha256="0"*64, environment=dict(code=dict(kind="synthetic")),
                runtime_seconds=0.)
    for key, value in (changes or {}).items():
        if value is None:
            prov.pop(key, None)
        else:
            prov[key] = value
    write_json(directory/PROVENANCE_FILE, prov)
    return directory


def write_registry(path, models, *, ensemble=True, **ensemble_changes):
    """A staging registry and, unless ensemble=False, the ensemble.json beside it as
    bayesreg.seg.unet.registry.register writes it (registry_sha256 of the file, member_check passed);
    `ensemble_changes` replace its fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(schema="bayesreg.segmenters.v1", models=models))
    if ensemble:
        document = dict(format="bayesreg.unet35.ensemble.v1", segmenter_id=SEG_ID, registry_path=str(path),
                        registry_sha256=load_segmenter_registry(path).sha256, fold_models=models, members=[],
                        member_check=dict(format="bayesreg.unet35.member_check.v1", passed=True))
        document.update(ensemble_changes)
        write_json(path.parent/ENSEMBLE_FILE, document)
    return path


@pytest.fixture
def env(tmp_path, request):
    """(root, allowlist, registry).  With synthetic_folds("env"): the synthetic training draw (C23.9-13 (8), revision 6e)."""
    synthetic = is_synthetic(request)
    allowlist = write_json(tmp_path/"dev_subjects.json", allowlist_document(synthetic_fold_sources() if synthetic else None))
    registry = write_registry(tmp_path/"segmenters.json", registry_models(SYN_FOLD0, SYN_FOLD1) if synthetic else registry_models())
    return tmp_path, allowlist, registry


def use_synthetic_folds(monkeypatch):
    """C23.9-13 (8), revision 6f (MEM): the digest modules (_mem_digest.py, _mem_paths_digest.py, unchanged) import allowlist_document
    and registry_models in their function bodies; these return the synthetic training draw's content for the rest of the test."""
    import sys
    from ..io_seg import _helpers
    plain = _helpers.allowlist_document
    monkeypatch.setattr(_helpers, "allowlist_document", lambda sources=None: plain(synthetic_fold_sources() if sources is None else sources))
    monkeypatch.setattr(sys.modules[__name__], "registry_models", lambda fold0=SYN_FOLD0, fold1=SYN_FOLD1: _registry_models(fold0, fold1))


def make_pair(root, fixed, moving, models, *, structure=True, model_ids=None, changes=None):
    """Pair, tissue layer and (unless structure=False) structure predictions; `model_ids` / `changes` apply to the
    moving image's prediction."""
    grid, pair = write_synthetic_pair(root, fixed, moving)
    write_tissue(root, grid, fixed)
    write_tissue(root, grid, moving, shift=(.7, -.5, .4))
    if structure:
        write_structure(root, grid, fixed, models)
        write_structure(root, grid, moving, models, shift=(.7, -.5, .4), model_ids=model_ids, changes=changes)
    return grid, pair


def source(root, allowlist, registry, **changes):
    fields = dict(root=root/"structure", registry_path=registry, allowlist_path=allowlist, models_root=root/"models",
                  image_root=root/"images", allow_staging=True)
    fields.update(changes)
    return StructureSource(**fields)


def prepare(root, pair_name, allowlist, registry, **changes):
    pair = load_pair(run_pair.resolve_pair(root/"data", pair_name), allowlist=allowlist)
    inputs, pt_f, mask_f, prov = run_pair.prepare_channels(pair, "structure", root/"seg", LOG,
                                                           structure=source(root, allowlist, registry, **changes))
    return pair, inputs, pt_f, mask_f, prov


def cli_args(root, allowlist, registry, pair, out, *extra, staging=True):
    return ["--pair", pair, "--channel-source", "structure", "--out", str(out), "--data-root", str(root/"data"),
            "--allowlist", str(allowlist), "--tissue-root", str(root/"seg"), "--structure-root", str(root/"structure"),
            "--segmenter-registry", str(registry), "--segmenter-models-root", str(root/"models"),
            "--structure-image-root", str(root/"images"), *(["--structure-registry-staging"] if staging else []), *extra]


def read(path):
    return json.loads(path.read_text())


# ---- end to end -------------------------------------------------------------------------------------------

@synthetic_folds("env")
def test_structure_channels_reach_energy_evaluation(env):
    root, allowlist, registry = env
    _, pair_name = make_pair(root, DEV[0], DEV[1], registry_models(SYN_FOLD0, SYN_FOLD1))
    pair, inputs, pt_f, mask_f, prov = prepare(root, pair_name, allowlist, registry)
    native = inputs[-1]
    assert [lv.fixed.shape[0] for lv in inputs] == [1+K_S]*4                  # intensity + K_S (no background, D2)
    for side, channels in (("fixed", native.fixed[1:]), ("moving", native.moving[1:])):
        with np.load(root/"structure"/getattr(pair, f"{side}_subject")/GROUPS_FILE) as z:
            p = z["p"].astype(np.float64)
        pbar, _ = simplex_from_background_class(p/p.sum(axis=0))
        ptilde, _ = tissue_channels(pbar)
        assert np.array_equal(channels, np.sqrt(ptilde[1:]))                  # the C03.2-4a -> C03.5 wiring
    assert prov["K_S"] == K_S and prov["K_T"] == 3
    s = prov["structure"]
    for side in ("fixed", "moving"):
        acc = s[side]["accepted"]
        assert acc["layer"] == "structure" and acc["test_labels_seen"] is False and acc["trained_with_labels"] is True
        assert acc["model_ids"] == ["unet35_v1_foldA", "unet35_v1_foldB"]
        assert acc["class_names"] == list(GROUPING.group_names)
        assert acc["pair_subjects_checked"] == [DEV[0], DEV[1]]
        assert s[side]["models"]["policy"] == "all_folds_mean" and len(s[side]["models"]["members"]) == 4
        assert s[side]["generated_in_this_run"] is False and s[side]["inference"] is None
        assert s[side]["simplex"]["contract"].startswith("C03.2-4a")
        assert s[side]["simplex"]["max_abs_input_sum_deviation"] <= 1e-12
        assert s[side]["smoothing"]["floor_denominator"] == pytest.approx(1+(K_S+1)*1e-3)
        read_back = s[side]["read_back"]
        assert 0 < read_back["raw_max_abs_sum_deviation"] <= 1e-3 and read_back["stored_dtype"] == "float16"
        assert read_back["raw_sum_tol"] == 1e-3
        assert s[side]["grouping"]["grouping"] == "structure13_v1" and s[side]["grouping"]["same_groups_file"]
    assert s["registry"]["status"] == STAGING and "未正式登记" in s["registry"]["note"]
    assert s["registry"]["sha256"] == load_segmenter_registry(registry).sha256
    assert s["registry"]["staging_opt_in"] is True and s["registry"]["use"].startswith("development diagnostics")
    ensemble = s["registry"]["staging_ensemble_check"]
    assert ensemble["registry_sha256_matches"] and ensemble["member_check_passed"]
    assert ensemble["path"] == str(registry.parent/ENSEMBLE_FILE)
    assert s["preflight"]["contact_union_contains_pair_subject"] is False and s["background_channel"] is False
    assert prov["pyramid"]["likelihood_channels"] == list(range(1, K_S+1))
    eta = eta_temporary(pt_f, mask_f, pair.grid)
    state = ModelState(grid=native.grid, omega0=pair.grid.omega, fixed=native.fixed,
                       samplers=build_samplers(native.moving, native.grid), mask=native.mask, sigma2=np.ones(1+K_S),
                       alpha=eta.alpha, nu=eta.nu, tau2=np.asarray(eta.tau2), tissue=native.tissue)
    ev = evaluate(state, np.zeros(native.grid.shape+(3,)))
    assert np.isfinite(ev.energy) and len(ev.energies["channel"]) == 1+K_S
    assert all(e > 0 for e in ev.energies["channel"])                          # moving channels shifted


@synthetic_folds("env")
def test_run_pair_main_structure_records_provenance(env):
    root, allowlist, registry = env
    _, pair_name = make_pair(root, DEV[0], DEV[1], registry_models(SYN_FOLD0, SYN_FOLD1))
    out = root/"run_structure"
    assert run_pair.main(cli_args(root, allowlist, registry, pair_name, out, "--max-levels", "1", "--rounds", "1,1,1,1",
                                  "--max-accepted", "1")) == 2
    status = read(out/"status.json")
    assert status["status"] == "stopped_before_native" and not (out/"disp.npz").exists()
    assert status["structure_registry"] == dict(status=STAGING, path=str(registry), staging_opt_in=True,
                                                development_diagnostic_only=True,
                                                git=dict(tracked=False, clean=False))       # tmp_path: no work tree
    assert status["resources"]["peak_rss_mb"] > 0                             # review PS item 7
    prov = read(out/"provenance.json")
    assert prov["channel_source"] == "structure" and prov["channels"]["K_S"] == K_S
    s = prov["channels"]["structure"]
    assert s["registry"]["status"] == "staging_unregistered" and s["source"]["grouping"] == "structure13_v1"
    assert s["fixed"]["accepted"]["test_labels_seen"] is False and s["moving"]["accepted"]["segmenter_id"] == SEG_ID
    assert s["preflight"]["models_used"] == ["unet35_v1_foldA", "unet35_v1_foldB"]
    hyper = read(out/"hyper.json")
    assert len(hyper["sigma_used"]) == 1+K_S
    data = json.loads((out/"data.txt").read_text())
    assert data["structure"]["root"] == str(root/"structure") and data["structure"]["infer"] == "never"
    assert data["structure"]["allow_staging"] is True
    assert read(out/"config.json")["args"]["segmenter_registry"] == str(registry)


def test_cli_defaults_refuse_staging_and_inference(env):
    """Defaults (review PS items 2, 4): the repository registry, no staging, no inference; the staging copy only with
    --structure-registry-staging; a staging registry without the switch ends provenance_rejected, labeled in
    status.json."""
    root, allowlist, registry = env
    base = ["--pair", "s/p", "--channel-source", "structure", "--out", "o"]
    args = run_pair.parse_args(base)
    assert args.segmenter_registry == str(structure_layer.REPO_REGISTRY) and not args.structure_registry_staging
    assert args.structure_infer == "never" and args.structure_device is None
    args = run_pair.parse_args(base+["--structure-registry-staging"])
    assert args.segmenter_registry == str(run_pair.DEFAULT_SEGMENTER_REGISTRY)
    source = run_pair.structure_source(args)
    assert source.allow_staging and source.infer == "never"
    assert StructureSource(root=root, registry_path=registry, allowlist_path=allowlist, models_root=root,
                           image_root=root).infer == "never"
    _, pair_name = make_pair(root, DEV[0], DEV[1], registry_models())
    out = root/"run_no_staging"
    assert run_pair.main(cli_args(root, allowlist, registry, pair_name, out, staging=False)) == 4
    status = read(out/"status.json")
    assert status["status"] == "provenance_rejected" and "--structure-registry-staging" in status["reason"]
    assert status["structure_registry"]["status"] == STAGING and status["structure_registry"]["staging_opt_in"] is False
    tissue = run_pair.parse_args(["--pair", "s/p", "--channel-source", "tissue", "--out", "o"])
    assert run_pair.structure_registry_label(tissue) is None


# ---- D24 leakage ------------------------------------------------------------------------------------------

def _no_reads(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("no prediction may be read or generated after a pair-level leak")
    monkeypatch.setattr(structure_layer, "read_structure_groups", refuse)
    monkeypatch.setattr(structure_layer.StructureLayer, "_generate", refuse)


@synthetic_folds("env")
def test_cross_fold_training_pair_is_refused_before_reading(env, monkeypatch):
    root, allowlist, registry = env
    _, pair_name = make_pair(root, SYN_FOLD0[0], SYN_FOLD1[0], registry_models(SYN_FOLD0, SYN_FOLD1))
    _no_reads(monkeypatch)
    out = root/"run_cross_fold"
    assert run_pair.main(cli_args(root, allowlist, registry, pair_name, out)) == 4
    status = read(out/"status.json")
    assert status["status"] == "provenance_rejected" and "D24 pair-level" in status["reason"]
    assert not (out/"provenance.json").exists()


def test_model_that_saw_a_pair_subject_is_refused(env, monkeypatch):
    root, allowlist, _ = env
    models = registry_models(fold1=FOLD1+[DEV[1]])                            # fold B saw the moving subject
    registry = write_registry(root/"leak"/"segmenters.json", models)
    _, pair_name = make_pair(root, DEV[0], DEV[1], models)
    _no_reads(monkeypatch)
    with pytest.raises(ProvenanceRejected, match="D24 pair-level.*unet35_v1_foldB"):
        prepare(root, pair_name, allowlist, registry)


@synthetic_folds("env")
def test_same_fold_training_pair_uses_the_other_fold(env):
    root, allowlist, registry = env
    _, pair_name = make_pair(root, SYN_FOLD0[0], SYN_FOLD0[1], registry_models(SYN_FOLD0, SYN_FOLD1))
    _, inputs, _, _, prov = prepare(root, pair_name, allowlist, registry)
    for side in ("fixed", "moving"):
        assert prov["structure"][side]["accepted"]["model_ids"] == ["unet35_v1_foldB"]
        assert prov["structure"][side]["models"]["policy"] == "cross_fit_other_folds"
    assert inputs[-1].fixed.shape[0] == 1+K_S


@synthetic_folds("env")
def test_prediction_from_the_wrong_models_is_refused(env):
    root, allowlist, registry = env
    _, pair_name = make_pair(root, DEV[0], DEV[1], registry_models(SYN_FOLD0, SYN_FOLD1), model_ids=["unet35_v1_foldA"])
    with pytest.raises(ProvenanceRejected, match="D24 model selection"):     # dev subject: both folds (D24)
        prepare(root, pair_name, allowlist, registry)
    root2 = root/"own_fold"
    _, pair_name = make_pair(root2, SYN_FOLD0[0], SYN_FOLD0[1], registry_models(SYN_FOLD0, SYN_FOLD1), model_ids=["unet35_v1_foldA"])
    with pytest.raises(ProvenanceRejected, match="label contact"):           # fold A saw the moving subject
        prepare(root2, pair_name, allowlist, registry)


def test_one_fold_registry_cannot_give_all_folds_mean(env, monkeypatch):
    """Review PS item 5: a hand-written registry with one fold model would let a development subject be segmented by
    that fold alone under the name all_folds_mean; refused before any prediction is read."""
    root, allowlist, _ = env
    models = {"unet35_v1_foldA": registry_models()["unet35_v1_foldA"]}
    registry = write_registry(root/"one_fold"/"segmenters.json", models)
    _, pair_name = make_pair(root, DEV[0], DEV[1], registry_models())
    _no_reads(monkeypatch)
    with pytest.raises(ProvenanceRejected, match="at least 2 registered folds"):
        prepare(root, pair_name, allowlist, registry)
    with pytest.raises(ProvenanceRejected, match="at least 2 registered folds"):
        expected_models(load_segmenter_registry(registry), FOLD1[0], SEG_ID)     # the other fold's subject too
    with pytest.raises(ProvenanceRejected, match="every fold model"):
        expected_models(load_segmenter_registry(registry), FOLD0[0], SEG_ID)


@synthetic_folds("env")
def test_expected_models_match_the_inference_selection(env):
    from bayesreg.io import load_dev_allowlist
    from bayesreg.seg.unet.infer import select_fold_models
    _, allowlist, registry_path = env
    registry, allow = load_segmenter_registry(registry_path), load_dev_allowlist(allowlist)
    for subject in (SYN_FOLD0[0], SYN_FOLD1[2], DEV[3], T0[0]):
        models, policy = select_fold_models(registry, subject, allow, segmenter_id=SEG_ID)
        assert expected_models(registry, subject, SEG_ID) == (sorted(models), policy)


# ---- provenance bindings ----------------------------------------------------------------------------------

def _hex(tag):
    import hashlib
    return hashlib.sha256(tag.encode()).hexdigest()


MODELS = registry_models()
SYN_MODELS = registry_models(SYN_FOLD0, SYN_FOLD1)      # C23.9-13 (8), revision 6e: the synthetic training draw's models
BAD_PROVENANCE = {                                  # case: (fields of the moving image's provenance, reason)
    "test_labels_seen true": (dict(test_labels_seen=True), "test_labels_seen"),
    "test_labels_seen missing": (dict(test_labels_seen=None), "test_labels_seen"),
    "trained_with_labels false": (dict(trained_with_labels=False), "trained_with_labels"),
    "oracle segmenter id": (dict(segmenter_id="seg35_oracle"), "label source"),
    "other segmenter id": (dict(segmenter_id="unet_other_ens"), "is registered for"),
    "model sha256 differs": (dict(model_sha256={m: [_hex("x"), _hex("y")] for m in MODELS}), "model_sha256"),
    "member missing": (dict(members=[dict(model_id="unet35_v1_foldA",
                                          member_sha256=MODELS["unet35_v1_foldA"]["member_model_sha256"][0])]),
                       "every registered seed member"),
    "member sha256 differs": (dict(members=[dict(model_id=m, member_sha256=_hex(f"{m}{k}")) for m in sorted(MODELS)
                                            for k in range(2)]), "every registered seed member"),
    "folds differ": (dict(folds={"unet35_v1_foldA": 1, "unet35_v1_foldB": 0}), "folds"),
    "policy differs": (dict(policy="cross_fit_other_folds"), "policy"),
    "other subject": (dict(subject=DEV[2]), "the prediction is of"),
    "input sha256 differs": (dict(input_sha256={"aligned_norm.nii.gz": _hex("other image")}), "input_sha256"),
    "grid differs": (dict(grid=dict(shape_zyx=[N, N, N], spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., .5])),
                     "not on the pair's grid"),
    "output sha256 differs": (dict(output_sha256={GROUPS_FILE: _hex("other output")}), "differs from output_sha256"),
}


@pytest.mark.parametrize("case", sorted(BAD_PROVENANCE))
@synthetic_folds("env")
def test_provenance_bindings_are_enforced(env, case):
    root, allowlist, registry = env
    changes, reason = BAD_PROVENANCE[case]
    _, pair_name = make_pair(root, DEV[0], DEV[1], SYN_MODELS, changes=changes)
    with pytest.raises(ProvenanceRejected, match=reason):
        prepare(root, pair_name, allowlist, registry)


@synthetic_folds("env")
def test_registry_status_records_formal_registration(env, monkeypatch):
    root, allowlist, registry = env
    assert registry_status(root/"a.json", repo_registry=root/"b.json")["status"] == STAGING
    assert registry_status(root/"a.json", repo_registry=root/"a.json")["status"] == "registered"
    monkeypatch.setattr(structure_layer, "REPO_REGISTRY", registry)
    _, pair_name = make_pair(root, DEV[0], DEV[1], SYN_MODELS)
    (registry.parent/ENSEMBLE_FILE).unlink()                                 # not needed for the repository registry
    record = prepare(root, pair_name, allowlist, registry, allow_staging=False)[4]["structure"]["registry"]
    assert record["status"] == "registered" and record["note"].startswith("正式登记")
    assert record["staging_ensemble_check"] is None and record["use"] == "registered"
    with pytest.raises(ProvenanceRejected, match="no segmenter registry"):
        prepare(root, pair_name, allowlist, root/"missing.json")


def git_repo(path):
    """A fresh git work tree at `path` (local identity, no hooks); returns a function running git there."""
    import subprocess
    path.mkdir(parents=True, exist_ok=True)

    def git(*args):
        return subprocess.run(["git", "-C", str(path), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                               "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
                              check=True, capture_output=True, text=True)
    git("init", "-q")
    return git


def test_registry_git_state_tracked_and_clean(tmp_path):
    """Review PS1_PS2 item b / A4 review item 3: the registry file's git state -- outside a work tree, untracked,
    committed and clean, modified, staged -- is recorded with every registration state."""
    outside = write_registry(tmp_path/"plain"/"segmenters.json", MODELS)
    state = structure_layer.registry_git_state(outside)
    assert state["in_work_tree"] is False and state["tracked"] is False and state["clean"] is False
    git = git_repo(tmp_path/"repo")
    registry = write_registry(tmp_path/"repo"/"configs"/"segmenters.json", MODELS, ensemble=False)
    state = structure_layer.registry_git_state(registry)
    assert state["in_work_tree"] is True and state["tracked"] is False and state["clean"] is False
    assert state["porcelain"].startswith("??")
    git("add", "configs/segmenters.json")
    state = structure_layer.registry_git_state(registry)
    assert state["tracked"] is True and state["clean"] is False                    # staged, not committed
    git("commit", "-q", "-m", "registry")
    state = structure_layer.registry_git_state(registry)
    assert state["tracked"] is True and state["clean"] is True and state["porcelain"] == ""
    assert state["last_commit"] == state["head"] and len(state["head"]) == 40
    (tmp_path/"repo"/"other.txt").write_text("x")                                  # other files do not matter
    assert structure_layer.registry_git_state(registry)["clean"] is True
    registry.write_text(registry.read_text()+"\n")
    state = structure_layer.registry_git_state(registry)
    assert state["tracked"] is True and state["clean"] is False and state["porcelain"].startswith(" M")
    status = registry_status(registry, repo_registry=registry)
    assert status["status"] == "registered" and status["git"]["clean"] is False   # recorded; the status is by path


@synthetic_folds("env")
def test_registry_record_carries_the_git_state(env, monkeypatch):
    root, allowlist, _ = env
    git = git_repo(root/"repo")
    registry = write_registry(root/"repo"/"configs"/"segmenters.json", SYN_MODELS, ensemble=False)
    git("add", "configs/segmenters.json")
    git("commit", "-q", "-m", "registry")
    monkeypatch.setattr(structure_layer, "REPO_REGISTRY", registry)
    _, pair_name = make_pair(root, DEV[0], DEV[1], SYN_MODELS)
    record = prepare(root, pair_name, allowlist, registry, allow_staging=False)[4]["structure"]["registry"]
    assert record["status"] == "registered" and record["git"]["tracked"] is True and record["git"]["clean"] is True
    args = run_pair.parse_args(["--pair", "s/p", "--channel-source", "structure", "--out", "o",
                                "--segmenter-registry", str(registry)])
    assert run_pair.structure_registry_label(args)["git"] == dict(tracked=True, clean=True)


def test_staging_registry_needs_the_opt_in(env, monkeypatch):
    root, allowlist, registry = env
    _, pair_name = make_pair(root, DEV[0], DEV[1], MODELS)
    _no_reads(monkeypatch)
    with pytest.raises(ProvenanceRejected, match="staging_unregistered.*--structure-registry-staging"):
        prepare(root, pair_name, allowlist, registry, allow_staging=False)


STAGING_ENSEMBLE = {                               # case: (write_registry keywords, reason)
    "ensemble.json missing": (dict(ensemble=False), "no ensemble.json"),
    "registry sha256 differs": (dict(registry_sha256="0"*64), "registry_sha256"),
    "member check failed": (dict(member_check=dict(passed=False)), "member_check.passed"),
    "member check missing": (dict(member_check=None), "member_check.passed"),
    "member check passed not a bool": (dict(member_check=dict(passed="true")), "member_check.passed"),
}


@pytest.mark.parametrize("case", sorted(STAGING_ENSEMBLE))
def test_staging_registry_needs_a_passed_member_check(env, monkeypatch, case):
    """Review PS item 2: with the opt-in, ensemble.json beside the staging registry must carry the registry file's
    sha256 and member_check.passed true; checked before any prediction is read."""
    root, allowlist, _ = env
    changes, reason = STAGING_ENSEMBLE[case]
    registry = write_registry(root/"staging"/"segmenters.json", MODELS, **changes)
    _, pair_name = make_pair(root, DEV[0], DEV[1], MODELS)
    _no_reads(monkeypatch)
    with pytest.raises(ProvenanceRejected, match=reason):
        prepare(root, pair_name, allowlist, registry)


# ---- grouping ---------------------------------------------------------------------------------------------

def test_unverified_grouping_is_refused(env, monkeypatch):
    root, allowlist, registry = env
    _, pair_name = make_pair(root, DEV[0], DEV[1], MODELS)
    _no_reads(monkeypatch)
    out = root/"run_tissue4"
    argv = cli_args(root, allowlist, registry, pair_name, out, "--structure-grouping", "tissue4_ref")
    assert run_pair.main(argv) == 4
    status = read(out/"status.json")
    assert status["status"] == "provenance_rejected" and status["reason"].startswith("grouping_unverified")


@synthetic_folds("env")
def test_prediction_with_an_unverified_or_other_grouping_is_refused(env):
    root, allowlist, registry = env
    override = dict(grouping_status="reference_unverified",
                    grouping_check=dict(grouping_status="reference_unverified", grouping_status_admitted=False,
                                        allow_unverified_grouping=True))
    _, pair_name = make_pair(root, DEV[0], DEV[1], SYN_MODELS, changes=override)
    with pytest.raises(ProvenanceRejected, match="grouping_unverified"):
        prepare(root, pair_name, allowlist, registry)
    root2 = root/"identity35"
    identity = load_grouping(name="identity35")
    grid, pair_name = make_pair(root2, DEV[0], DEV[1], SYN_MODELS, structure=False)
    write_structure(root2, grid, DEV[0], SYN_MODELS)
    write_structure(root2, grid, DEV[1], SYN_MODELS, grouping=identity)
    with pytest.raises(ProvenanceRejected, match="grouping field"):
        prepare(root2, pair_name, allowlist, registry)


# ---- C03.2-4a ---------------------------------------------------------------------------------------------

@synthetic_folds("env")
def test_background_class_entry_checks_the_sum(env, monkeypatch):
    root, allowlist, registry = env
    _, pair_name = make_pair(root, DEV[0], DEV[1], SYN_MODELS)
    real = structure_layer.read_structure_groups

    def off_by_more_than_tolerance(*args, **kwargs):
        array, record = real(*args, **kwargs)
        array = array.copy()
        array[:, 3, 4, 5] *= 1+5e-6                                           # one voxel, |sum - 1| = 5e-6 > 1e-6
        return array, record
    monkeypatch.setattr(structure_layer, "read_structure_groups", off_by_more_than_tolerance)
    with pytest.raises(ProvenanceRejected, match="C03.2-4a"):
        prepare(root, pair_name, allowlist, registry)
    monkeypatch.setattr(structure_layer, "read_structure_groups", real)
    root2 = root/"zero"
    grid, pair_name = make_pair(root2, DEV[0], DEV[1], SYN_MODELS, structure=False)
    groups = structure_groups(grid)
    groups[:, 0, 0, 0] = 0.
    write_structure(root2, grid, DEV[0], SYN_MODELS)
    write_structure(root2, grid, DEV[1], SYN_MODELS, array=groups)
    with pytest.raises(ProvenanceRejected, match="zero total"):
        prepare(root2, pair_name, allowlist, registry)


@synthetic_folds("env")
def test_raw_float16_sum_off_by_far_more_than_rounding_is_refused(env):
    """Review PS item 1: a stored file whose channels sum to 0.7 used to be divided by its sum and accepted; the raw
    sum is now checked before the read-back normalization."""
    root, allowlist, registry = env
    grid, pair_name = make_pair(root, DEV[0], DEV[1], SYN_MODELS, structure=False)
    write_structure(root, grid, DEV[0], SYN_MODELS)
    write_structure(root, grid, DEV[1], SYN_MODELS, array=.7*structure_groups(grid))
    with pytest.raises(ProvenanceRejected, match=r"raw channel sum .* > 0\.001 \(float16 storage\)"):
        prepare(root, pair_name, allowlist, registry)


def _write_groups(path, array):
    np.savez_compressed(path, p=array)
    return path


def test_read_back_raw_sum_tolerances(tmp_path):
    """read_structure_groups: float16 storage within 1e-3 (unit roundoff 2^-11), float32 / float64 within 1e-6, on
    every voxel and before the division; other storage types refused."""
    shape = (3, 4, 5)
    base = np.zeros((3,)+shape)
    base[0], base[1], base[2] = .25, .5, .25                                  # exact in float16
    _, record = read_structure_groups(_write_groups(tmp_path/"exact.npz", base.astype(np.float16)), 3, shape)
    assert record["raw_max_abs_sum_deviation"] == 0 and record["raw_sum_tol"] == 1e-3
    within = base.copy()
    within[:, 1, 2, 3] = (.5+2**-11, .5, 0.)                                 # |sum - 1| = 2^-11 ~ 4.9e-4
    array, record = read_structure_groups(_write_groups(tmp_path/"within.npz", within.astype(np.float16)), 3, shape)
    assert record["raw_max_abs_sum_deviation"] == 2**-11 and record["stored_dtype"] == "float16"
    assert np.max(np.abs(array.sum(axis=0)-1)) <= 1e-15                      # divided by the channel sum
    off = base.copy()
    off[:, 2, 0, 1] = (.5, .5, 2**-9)                                         # one voxel, |sum - 1| ~ 2e-3
    with pytest.raises(ProvenanceRejected, match="raw channel sum"):
        read_structure_groups(_write_groups(tmp_path/"off.npz", off.astype(np.float16)), 3, shape)
    with pytest.raises(ProvenanceRejected, match="raw channel sum"):
        read_structure_groups(_write_groups(tmp_path/"scaled.npz", (.7*base).astype(np.float16)), 3, shape)
    for dtype in (np.float32, np.float64):
        name = np.dtype(dtype).name
        _, record = read_structure_groups(_write_groups(tmp_path/f"{name}.npz", base.astype(dtype)), 3, shape)
        assert record["raw_sum_tol"] == 1e-6 and record["stored_dtype"] == name
        small = base.copy()
        small[0, 0, 0, 0] += 5e-6                                             # within float16 rounding, not float32/64
        with pytest.raises(ProvenanceRejected, match=rf"raw channel sum .* \({name} storage\)"):
            read_structure_groups(_write_groups(tmp_path/f"{name}_off.npz", small.astype(dtype)), 3, shape)
    with pytest.raises(ProvenanceRejected, match="stored as one of"):
        read_structure_groups(_write_groups(tmp_path/"int.npz", (base*4).astype(np.int8)), 3, shape)
    if np.dtype(np.longdouble).itemsize > 8:
        with pytest.raises(ProvenanceRejected, match="stored as one of"):
            read_structure_groups(_write_groups(tmp_path/"long.npz", base.astype(np.longdouble)), 3, shape)


def test_missing_prediction_with_infer_never_is_refused(env):
    root, allowlist, registry = env
    _, pair_name = make_pair(root, DEV[0], DEV[1], MODELS, structure=False)
    with pytest.raises(ProvenanceRejected, match="infer = 'never'"):
        prepare(root, pair_name, allowlist, registry)


def test_real_mode_inference_needs_the_pinned_allowlist(env):
    """Without --structure-synthetic a missing prediction needs the pinned repository allowlist (and clean code):
    a tmp allowlist is refused before any image is read, and nothing is written."""
    root, allowlist, registry = env
    _, pair_name = make_pair(root, DEV[0], DEV[1], MODELS, structure=False)
    with pytest.raises(ProvenanceRejected, match="inference refused before reading any image"):
        prepare(root, pair_name, allowlist, registry, infer="missing")
    assert not (root/"structure"/DEV[0]).exists()


def _admit_inference(monkeypatch):
    """Skip the inference admission (pinned allowlist / synthetic ids) so that the checks under the lock are reached;
    segment_subject itself must not be called."""
    monkeypatch.setattr(StructureLayer, "_inference_allowlist", lambda self, side: (self.allowlist, dict(mode="test")))
    import bayesreg.seg.unet.infer as infer

    def refuse(*args, **kwargs):
        raise AssertionError("segment_subject must not run")
    monkeypatch.setattr(infer, "segment_subject", refuse)


def test_incomplete_prediction_directory_is_kept(env, monkeypatch):
    root, allowlist, registry = env
    _, pair_name = make_pair(root, DEV[0], DEV[1], MODELS)
    (root/"structure"/DEV[0]/PROVENANCE_FILE).unlink()
    _admit_inference(monkeypatch)
    with pytest.raises(ProvenanceRejected, match="incomplete prediction is kept"):
        prepare(root, pair_name, allowlist, registry, infer="missing")
    assert (root/"structure"/DEV[0]/GROUPS_FILE).is_file()
    assert not (root/"structure"/DEV[0]/PROVENANCE_FILE).exists()


@synthetic_folds("env")
def test_prediction_finished_by_a_concurrent_run_is_read_back(env, monkeypatch):
    """Review PS item 6: while this run waits for the per-subject lock, a concurrent run holds it and is still writing
    the directory (structure_groups.npz present, provenance.json not yet); it finishes before releasing the lock.
    The checks under the lock read the finished prediction back instead of refusing it, and the record says it was
    not generated in this run."""
    root, allowlist, registry = env
    _, pair_name = make_pair(root, DEV[0], DEV[1], SYN_MODELS)
    directory = root/"structure"/DEV[0]
    held = root/"held_provenance.json"
    (directory/PROVENANCE_FILE).rename(held)                                  # the concurrent run is mid-write
    _admit_inference(monkeypatch)
    calls = []

    def flock(handle, operation):                                           # the concurrent run finishes, then we lock
        calls.append(operation)
        held.rename(directory/PROVENANCE_FILE)
        fcntl.flock(handle, operation)
    monkeypatch.setattr(structure_layer, "fcntl", types.SimpleNamespace(flock=flock, LOCK_EX=fcntl.LOCK_EX))
    _, _, _, _, prov = prepare(root, pair_name, allowlist, registry, infer="missing")
    record = prov["structure"]["fixed"]
    assert calls == [fcntl.LOCK_EX]
    assert record["generated_in_this_run"] is False and record["inference"]["generated_here"] is False
    assert record["inference"]["made_by"].startswith("a concurrent run")
    assert prov["structure"]["moving"]["generated_in_this_run"] is False and prov["structure"]["moving"]["inference"] is None


# ---- inference of missing predictions (tiny CPU members, synthetic ids) -----------------------------------

def test_missing_predictions_are_generated_by_the_segmenter(tmp_path):
    from bayesreg.seg.unet.calibrate import MEMBER_FORMAT
    from bayesreg.seg.unet.infer import GROUPS_FILE as INFER_GROUPS, PROBS_FILE, PROVENANCE_FILE as INFER_PROV
    from bayesreg.seg.unet.model import build_model
    from bayesreg.seg.unet.synthetic import write_pair, write_subjects
    from bayesreg.io import subject_list_sha256
    from ..segnet._helpers import A_CALIB, A_TRAIN, B_CALIB, B_TRAIN, DEV as SDEV, make_allowlist

    assert (INFER_GROUPS, INFER_PROV) == (GROUPS_FILE, PROVENANCE_FILE)
    torch.set_num_threads(1)
    allowlist = make_allowlist(tmp_path)
    write_subjects(tmp_path/"images", SDEV, (N, N, N))
    name = f"{SDEV[0]}__{SDEV[1]}"
    write_pair(tmp_path/"data"/STAGE/name, tmp_path/"images", SDEV[0], SDEV[1])
    pair = load_pair(tmp_path/"data"/STAGE/name/"pair.npz", allowlist=allowlist)
    for subject, side, shift in ((SDEV[0], "fixed", (0, 0, 0)), (SDEV[1], "moving", (1, -1, 0))):
        directory = tmp_path/"seg"/subject                                    # Atropos-like, index coordinates
        directory.mkdir(parents=True)
        idx = np.stack(np.indices(pair.grid.shape), axis=-1)
        c = (N-1)/2
        brain = (np.linalg.norm((idx-c-np.asarray(shift))/c, axis=-1) < .8).astype(float)
        files = [f"posteriors{k+1:02d}.nii.gz" for k in range(3)]
        for file, share in zip(files, (.2, .3, .5)):
            save_scalar(directory/file, share*brain, pair.grid)
        write_json(directory/"provenance.json", dict(
            layer="tissue", segmenter_id="atropos", model_ids=["ANTs-synthetic"], trained_with_labels=False,
            test_labels_seen=False, class_names=["CSF", "GM", "WM"], wm_index=2, posterior_files=files,
            posterior_file_index_base=1, input_sha256={"aligned_norm.nii.gz": pair.source_sha256[side]},
            output_sha256={f: file_sha256(directory/f) for f in files}))
    models = {}
    for fold, index, contact in (("A", 0, sorted(A_TRAIN+A_CALIB)), ("B", 1, sorted(B_TRAIN+B_CALIB))):
        shas = []
        for seed in (0, 1):
            torch.manual_seed(10*index+seed)
            member = tmp_path/"models"/f"fold{fold}_s{seed}"
            member.mkdir(parents=True)
            torch.save(dict(format=MEMBER_FORMAT, state_dict=build_model(dict(base_channels=4, depth=2)).state_dict(),
                            model_config=dict(base_channels=4, depth=2), temperature=1.3, fold=fold, seed=seed,
                            best_iteration=1, label_contact_subjects=contact,
                            label_contact_subjects_sha256=subject_list_sha256(contact), model_best_sha256="0"*64,
                            percentile=99.5, window=None, amp=False), member/"member.pt")
            shas.append(file_sha256(member/"member.pt"))
        models[f"unet35_v1_fold{fold}"] = dict(segmenter_id=SEG_ID, fold=index, member_model_sha256=shas,
                                               label_contact_subjects=contact,
                                               label_contact_subjects_sha256=subject_list_sha256(contact))
    registry = write_registry(tmp_path/"segmenters.json", models)
    src = StructureSource(root=tmp_path/"pred", registry_path=registry, allowlist_path=allowlist.path,
                          models_root=tmp_path/"models", image_root=tmp_path/"images", infer="missing", device="cpu",
                          synthetic=True, command=("test",), allow_staging=True)
    inputs, _, _, prov = run_pair.prepare_channels(pair, "structure", tmp_path/"seg", LOG, structure=src)
    for subject, side in ((SDEV[0], "fixed"), (SDEV[1], "moving")):
        directory = tmp_path/"pred"/subject
        assert {p.name for p in directory.iterdir()} == {PROBS_FILE, GROUPS_FILE, PROVENANCE_FILE}
        record = prov["structure"][side]
        assert record["generated_in_this_run"] is True and record["inference"]["device"] == "cpu"
        assert record["inference"]["generated_here"] is True
        assert record["inference"]["admission"]["mode"] == "synthetic"
        assert record["models"]["policy"] == "all_folds_mean" and len(record["models"]["members"]) == 4
        assert read(directory/PROVENANCE_FILE)["command"] == ["test"]
        read_back = read(directory/PROVENANCE_FILE)["outputs"]["read_back"]        # review PS1_PS2 item e
        assert "raw channel sum" in read_back and "BEFORE the division" in read_back and "1e-3" in read_back
    assert inputs[-1].fixed.shape[0] == 1+K_S
    again, _, _, prov2 = run_pair.prepare_channels(pair, "structure", tmp_path/"seg", LOG, structure=src)
    assert not prov2["structure"]["fixed"]["generated_in_this_run"]                  # read back, not regenerated
    assert all(np.array_equal(a.fixed, b.fixed) and np.array_equal(a.moving, b.moving) for a, b in zip(inputs, again))
