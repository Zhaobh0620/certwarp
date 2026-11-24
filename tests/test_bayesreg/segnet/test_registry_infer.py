"""Grouping table and its inference status gate, registry entries accepted by bayesreg.io, the member
cross-checks before registration (folds, label guard, records, member.pt), append-only registration,
load_member against the registry entry, model selection (cross-fitting, refusal of models that saw the
subject or T0), ensemble averaging and renormalization, float16 read-back with hash binding."""
import hashlib
import json

import numpy as np
import pytest
import torch

from bayesreg.io.guard import SubjectSealed
from bayesreg.io.pair import ProvenanceRejected, load_segmenter_registry, subject_list_sha256
from bayesreg.seg.unet.calibrate import MEMBER_FORMAT
from bayesreg.seg.unet.data import FOLD_INDEX
from bayesreg.seg.unet.groups import (DEFAULT_GROUPS_PATH, GroupingUnverified, group_labels, group_probabilities,
                                      load_grouping, require_inference_grouping)
from bayesreg.seg.unet.infer import (GROUPS_FILE, PROBS_FILE, ModelSelectionRefused, check_subject_admitted,
                                     ensemble_probabilities, load_member, read_probabilities, select_fold_models)
from bayesreg.seg.unet.model import build_model
from bayesreg.seg.unet.registry import MemberCheckFailed, build_entries, register, verify_members
from bayesreg.seg.unet.synthetic import write_json
from bayesreg.seg.unet.train import file_sha256
from jd_curl.validation import InputValidationError

from ._helpers import (A_CALIB, A_TRAIN, B_CALIB, B_TRAIN, DEV, SEALED, T0, TRAIN_SOURCE, allowlist_sources,
                       make_allowlist, make_split)


def H(tag):                                                    # noqa: N802
    return hashlib.sha256(tag.encode()).hexdigest()


# ---- groups ---------------------------------------------------------------------------------------

def test_repository_grouping_table():
    default = load_grouping()
    assert default.name == "structure13_v1" and default.k == 13 and default.status == "default"
    table = default.class_to_channel()
    assert table[0] == 0 and sorted(set(table[1:].tolist())) == list(range(1, 14))
    assert table[10] == table[29] == 1+default.group_names.index("pallidum")
    assert table[16] != table[8] and table[4] != table[3]      # accumbens / caudate; inf-lat / lateral ventricle
    assert load_grouping(name="identity35").k == 35
    reference = load_grouping(name="tissue4_ref")
    assert reference.k == 4 and reference.status == "reference_unverified"


def test_group_probabilities_sum_and_group_labels():
    grouping = load_grouping()
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(36), size=(3, 4, 5)).transpose(3, 0, 1, 2)
    groups = group_probabilities(probs, grouping)
    assert groups.shape == (14, 3, 4, 5) and np.max(np.abs(groups.sum(0)-1)) < 1e-12
    hard = rng.integers(0, 36, (3, 4, 5))
    assert np.array_equal(group_labels(hard, grouping), grouping.class_to_channel()[hard])


def test_unverified_grouping_is_refused_for_inference_unless_allowed():
    assert require_inference_grouping(load_grouping())["grouping_status_admitted"]
    assert require_inference_grouping(load_grouping(name="identity35"))["grouping_status"] == "ablation"
    reference = load_grouping(name="tissue4_ref")
    with pytest.raises(GroupingUnverified):
        require_inference_grouping(reference)
    record = require_inference_grouping(reference, allow_unverified=True)
    assert record == dict(grouping_status="reference_unverified", grouping_status_admitted=False,
                          allow_unverified_grouping=True)


def test_grouping_rule_lists_its_exceptions():
    document = json.loads(DEFAULT_GROUPS_PATH.read_text(encoding="utf-8"))
    exceptions = " ".join(document["groupings"]["structure13_v1"]["rule_exceptions"])
    for word in ("vermis", "brain stem", "corpus callosum"):
        assert word in exceptions


def test_malformed_grouping_is_refused(tmp_path):
    document = json.loads(DEFAULT_GROUPS_PATH.read_text(encoding="utf-8"))
    document["groupings"]["structure13_v1"]["groups"][0]["labels"] = [1]          # label 20 now unassigned
    path = write_json(tmp_path/"groups.json", document)
    with pytest.raises(InputValidationError):
        load_grouping(path)


# ---- registry entries -----------------------------------------------------------------------------

def _members(contact_a=A_TRAIN+A_CALIB, contact_b=B_TRAIN+B_CALIB, seeds=(0, 1)):
    out = []
    for fold, contact in (("A", sorted(contact_a)), ("B", sorted(contact_b))):
        for seed in seeds:
            out.append(dict(fold=fold, seed=seed, member_sha256=H(f"{fold}{seed}"), temperature=1.2,
                            label_contact_subjects=contact, label_contact_subjects_sha256=subject_list_sha256(contact),
                            path=f"/x/fold{fold}_s{seed}/member.pt"))
    return out


def test_entries_are_accepted_by_the_io_registry_loader(tmp_path):
    entries = build_entries(_members())
    assert sorted(entries) == ["unet35_v1_foldA", "unet35_v1_foldB"]
    assert entries["unet35_v1_foldB"]["fold"] == 1 and len(entries["unet35_v1_foldA"]["member_model_sha256"]) == 2
    path = write_json(tmp_path/"segmenters.json", dict(schema="bayesreg.segmenters.v1", models=entries))
    registry = load_segmenter_registry(path)
    assert registry.models["unet35_v1_foldA"]["segmenter_id"] == "unet35_v1_ens"


def test_entries_refuse_incomplete_or_overlapping_folds():
    with pytest.raises(InputValidationError):
        build_entries(_members()[:3])                             # fold B seed 1 missing
    with pytest.raises(InputValidationError):
        build_entries(_members(contact_b=B_TRAIN+B_CALIB+A_TRAIN[:1]))


# ---- member checks and registration ----------------------------------------------------------------

def _consistent_members(root, tmp_path, *, pre=None, post=None, seeds=(0, 1)):
    """Member directories whose records agree with make_split() and make_allowlist(tmp_path), as
    train_member + calibrate_member write them; `pre(fold, seed, docs)` may change the documents before
    they are written, `post(fold, seed, member)` the member.json record after its hashes are filled in."""
    split, allowlist = make_split(), make_allowlist(tmp_path)
    guard = dict(guard="segmenter-label-guard-v2", folds_sha256=split.sha256, allowlist_sha256=allowlist.sha256,
                 training_source=TRAIN_SOURCE)
    state = build_model(dict(base_channels=4, depth=2)).state_dict()
    for fold in ("A", "B"):
        contact = split.label_contact(fold)
        for seed in seeds:
            directory = root/f"fold{fold}_s{seed}"
            directory.mkdir(parents=True)
            best = H(f"best{fold}{seed}")
            reads = [dict(subject=x, kind="seg35", sha256=H(x)) for x in contact]
            calib_reads = [dict(subject=x, kind="seg35", sha256=H(x)) for x in split.subjects(fold, "calib")]
            docs = dict(
                blob=dict(format=MEMBER_FORMAT, state_dict=state, model_config=dict(base_channels=4, depth=2),
                          temperature=1.25, fold=fold, seed=seed, best_iteration=2, label_contact_subjects=list(contact),
                          label_contact_subjects_sha256=subject_list_sha256(contact), model_best_sha256=best,
                          percentile=99.5, window=None, amp=False),
                train_record=dict(fold=fold, seed=seed, train_subjects=split.subjects(fold, "train"),
                                  calib_subjects=split.subjects(fold, "calib"), label_contact_subjects=list(contact),
                                  model_best_sha256=best, label_guard=dict(guard, label_reads=reads)),
                calibration=dict(fold=fold, seed=seed, calib_subjects=split.subjects(fold, "calib"),
                                 label_guard=dict(guard, label_reads=calib_reads)))
            if pre:
                pre(fold, seed, docs)
            torch.save(docs["blob"], directory/"member.pt")
            write_json(directory/"train_record.json", docs["train_record"])
            write_json(directory/"calibration.json", docs["calibration"])
            member = dict(format="bayesreg.unet35.member_record.v1", member_file="member.pt",
                          member_sha256=file_sha256(directory/"member.pt"), fold=fold, fold_index=FOLD_INDEX[fold],
                          seed=seed, temperature=1.25, label_contact_subjects=list(contact),
                          label_contact_subjects_sha256=subject_list_sha256(contact),
                          train_record_sha256=file_sha256(directory/"train_record.json"),
                          calibration_sha256=file_sha256(directory/"calibration.json"), folds_sha256=split.sha256)
            if post:
                post(fold, seed, member)
            write_json(directory/"member.json", member)
    return split, allowlist


def _register(tmp_path, split, allowlist, **kwargs):
    return register(tmp_path/"models", kwargs.pop("registry_path", tmp_path/"segmenters.json"), split=split,
                    allowlist=allowlist, training_source=kwargs.pop("training_source", TRAIN_SOURCE),
                    log=lambda *_: None, **kwargs)


def test_register_is_append_only_and_validated(tmp_path):
    split, allowlist = _consistent_members(tmp_path/"models", tmp_path)
    registry_path = tmp_path/"segmenters.json"
    first = _register(tmp_path, split, allowlist)
    again = _register(tmp_path, split, allowlist)                    # identical: no change
    ensemble = json.loads((tmp_path/"models"/"ensemble.json").read_text())
    assert first.sha256 == again.sha256 and ensemble["members"] and ensemble["member_check"]["passed"]
    assert first.models["unet35_v1_foldA"]["label_contact_subjects"] == tuple(sorted(A_TRAIN+A_CALIB))
    document = json.loads(registry_path.read_text())
    document["models"]["unet35_v1_foldA"]["member_model_sha256"] = [H("other"), H("other2")]
    registry_path.write_text(json.dumps(document))
    with pytest.raises(InputValidationError):
        _register(tmp_path, split, allowlist)


def test_register_into_another_registry_keeps_the_previous_ensemble(tmp_path):
    """Review PS1_PS2 item a: registering into the repository registry after the staging one replaces ensemble.json
    (registry_path changes); the staging-period file is first kept byte for byte as ensemble_before_<UTC>.json,
    recorded in the new file; an identical re-registration leaves ensemble.json and the directory untouched."""
    import re
    split, allowlist = _consistent_members(tmp_path/"models", tmp_path)
    ensemble_path = tmp_path/"models"/"ensemble.json"
    (tmp_path/"staging").mkdir()
    _register(tmp_path, split, allowlist, registry_path=tmp_path/"staging"/"segmenters.json")
    staging_bytes = ensemble_path.read_bytes()
    _register(tmp_path, split, allowlist, registry_path=tmp_path/"staging"/"segmenters.json")       # identical
    assert ensemble_path.read_bytes() == staging_bytes and not list(ensemble_path.parent.glob("ensemble_before_*"))
    (tmp_path/"repo").mkdir()
    lines = []
    register(tmp_path/"models", tmp_path/"repo"/"segmenters.json", split=split, allowlist=allowlist,
             training_source=TRAIN_SOURCE, log=lines.append)
    backups = sorted(ensemble_path.parent.glob("ensemble_before_*.json"))
    assert len(backups) == 1 and re.fullmatch(r"ensemble_before_\d{8}T\d{6}Z\.json", backups[0].name)
    assert backups[0].read_bytes() == staging_bytes and backups[0].name in lines[-1]
    new = json.loads(ensemble_path.read_text())
    assert new["registry_path"] == str(tmp_path/"repo"/"segmenters.json")
    assert new["replaced_ensemble"]["backup"] == backups[0].name
    assert new["replaced_ensemble"]["sha256"] == hashlib.sha256(staging_bytes).hexdigest()
    assert new["replaced_ensemble"]["registry_path_before"] == str(tmp_path/"staging"/"segmenters.json")
    # the two registries hold the same bytes, so the staging arm's gate still accepts the staging copy
    assert (tmp_path/"repo"/"segmenters.json").read_bytes() == (tmp_path/"staging"/"segmenters.json").read_bytes()
    assert new["registry_sha256"] == json.loads(staging_bytes)["registry_sha256"]
    repo_bytes = ensemble_path.read_bytes()
    register(tmp_path/"models", tmp_path/"repo"/"segmenters.json", split=split, allowlist=allowlist,
             training_source=TRAIN_SOURCE, log=lambda *_: None)                                    # identical again
    assert ensemble_path.read_bytes() == repo_bytes and len(list(ensemble_path.parent.glob("ensemble_before_*"))) == 1
    _register(tmp_path, split, allowlist, registry_path=tmp_path/"staging"/"segmenters.json")       # back: 2nd copy
    backups = sorted(ensemble_path.parent.glob("ensemble_before_*.json"))
    assert len(backups) == 2 and repo_bytes in [b.read_bytes() for b in backups]
    assert not list(ensemble_path.parent.glob("*.new"))


def _with_extra_contact(member):
    contact = sorted(member["label_contact_subjects"]+[DEV[0]])
    member.update(label_contact_subjects=contact, label_contact_subjects_sha256=subject_list_sha256(contact))


def _only(target, change):
    """Apply `change(document)` only to member (fold, seed) == target (None = every member)."""
    def apply(fold, seed, document):
        if target is None or (fold, seed) == target:
            change(document)
    return apply


TAMPER = {
    "contact_not_folds": dict(post=_only(("A", 0), _with_extra_contact)),
    "folds_sha256_differs": dict(post=_only(("B", 1), lambda m: m.update(folds_sha256=H("other folds")))),
    "no_label_guard": dict(pre=_only(("A", 1), lambda d: d["train_record"].update(label_guard=None))),
    "empty_label_reads": dict(pre=_only(("B", 0), lambda d: d["train_record"]["label_guard"].update(label_reads=[]))),
    "read_outside_contact": dict(pre=_only(("B", 1), lambda d: d["train_record"]["label_guard"]["label_reads"].append(
        dict(subject=DEV[0], kind="seg35", sha256=H("x"))))),
    "contact_not_read": dict(pre=_only(("A", 1), lambda d: d["train_record"]["label_guard"]["label_reads"].pop())),
    "read_seg4": dict(pre=_only(("A", 0), lambda d: d["train_record"]["label_guard"]["label_reads"][0].update(
        kind="seg4"))),
    "guard_other_folds": dict(pre=_only(("A", 0), lambda d: d["train_record"]["label_guard"].update(
        folds_sha256=H("f")))),
    "guard_other_source": dict(pre=_only(("B", 0), lambda d: d["train_record"]["label_guard"].update(
        training_source="synthetic_dev"))),
    "no_calibration_guard": dict(pre=_only(("A", 0), lambda d: d["calibration"].update(label_guard=None))),
    "calibration_read_train": dict(pre=_only(("A", 0), lambda d: d["calibration"]["label_guard"].update(
        label_reads=[dict(subject=A_TRAIN[0], kind="seg35", sha256=H("x"))]))),
    "calibration_read_twice": dict(pre=_only(("B", 0), lambda d: d["calibration"]["label_guard"]["label_reads"].append(
        dict(d["calibration"]["label_guard"]["label_reads"][0])))),
    "train_subjects_differ": dict(pre=_only(("B", 1), lambda d: d["train_record"].update(
        train_subjects=B_TRAIN[:1]))),
    "member_pt_fold": dict(pre=_only(("A", 1), lambda d: d["blob"].update(fold="B"))),
    "member_pt_contact": dict(pre=_only(("A", 1), lambda d: d["blob"].update(label_contact_subjects=list(B_TRAIN)))),
    "member_pt_temperature": dict(pre=_only(("B", 1), lambda d: d["blob"].update(temperature=2.))),
    "member_pt_best": dict(pre=_only(("B", 0), lambda d: d["blob"].update(model_best_sha256=H("other")))),
}


@pytest.mark.parametrize("case", sorted(TAMPER))
def test_register_cross_checks_members_against_folds_guard_and_records(tmp_path, case):
    split, allowlist = _consistent_members(tmp_path/"models", tmp_path, **TAMPER[case])
    with pytest.raises(MemberCheckFailed):
        _register(tmp_path, split, allowlist)
    assert not (tmp_path/"segmenters.json").exists()


def test_register_refuses_other_pins_allowlist_or_training_source(tmp_path):
    split, allowlist = _consistent_members(tmp_path/"models", tmp_path)
    with pytest.raises(MemberCheckFailed):
        _register(tmp_path, split, allowlist, pinned_folds_sha256=H("pinned"))
    other = make_allowlist(tmp_path, allowlist_sources(extra=[("synthetic_more", [SEALED], False)]), name="other.json")
    with pytest.raises(MemberCheckFailed):                                   # trained with another allowlist
        _register(tmp_path, split, other)
    with pytest.raises(MemberCheckFailed):
        _register(tmp_path, split, allowlist, training_source="synthetic_dev")
    assert not (tmp_path/"segmenters.json").exists()
    _, record = verify_members(tmp_path/"models", split=split, allowlist=allowlist, training_source=TRAIN_SOURCE,
                               pinned_folds_sha256=split.sha256)
    assert record["passed"] and len(record["members"]) == 4 and record["pinned_folds_sha256"] == split.sha256


def test_register_refuses_a_tampered_member_file(tmp_path):
    split, allowlist = _consistent_members(tmp_path/"models", tmp_path)
    (tmp_path/"models"/"foldA_s0"/"member.pt").write_bytes(b"changed")
    with pytest.raises(InputValidationError):
        _register(tmp_path, split, allowlist)


def test_register_refuses_forbidden_segmenter_ids(tmp_path):
    split, allowlist = _consistent_members(tmp_path/"models", tmp_path)
    with pytest.raises(ProvenanceRejected):
        _register(tmp_path, split, allowlist, segmenter_id="seg35_oracle")
    assert not (tmp_path/"segmenters.json").exists()


def test_load_member_compares_member_pt_with_the_registry_entry(tmp_path):
    split, allowlist = _consistent_members(tmp_path/"models", tmp_path)
    registry = _register(tmp_path, split, allowlist)
    entry = registry.models["unet35_v1_foldA"]
    path = tmp_path/"models"/"foldA_s0"/"member.pt"
    sha = file_sha256(path)
    _, blob = load_member(path, sha, "cpu", entry=entry)
    assert blob["fold"] == "A" and blob["temperature"] == 1.25
    for change in (dict(fold=1), dict(label_contact_subjects=tuple(sorted(B_TRAIN+B_CALIB))),
                   dict(label_contact_subjects_sha256=H("x")), dict(member_model_sha256=(H("y"),))):
        with pytest.raises(ModelSelectionRefused):
            load_member(path, sha, "cpu", entry=dict(entry, **change))
    with pytest.raises(ModelSelectionRefused):                         # a fold-A member listed under fold B's entry
        load_member(path, sha, "cpu", entry=dict(registry.models["unet35_v1_foldB"], member_model_sha256=(sha,)))


# ---- model selection ------------------------------------------------------------------------------

def _registry(tmp_path, extra=None):
    models = build_entries(_members())
    models.update(extra or {})
    return load_segmenter_registry(write_json(tmp_path/"segmenters.json",
                                              dict(schema="bayesreg.segmenters.v1", models=models)))


def test_selection_cross_fits_training_subjects_and_averages_the_rest(tmp_path):
    registry, allowlist = _registry(tmp_path), make_allowlist(tmp_path)
    models, policy = select_fold_models(registry, A_TRAIN[0], allowlist)
    assert list(models) == ["unet35_v1_foldB"] and policy == "cross_fit_other_folds"
    models, policy = select_fold_models(registry, B_CALIB[0], allowlist)
    assert list(models) == ["unet35_v1_foldA"]
    for subject in (DEV[0], T0[0]):
        models, policy = select_fold_models(registry, subject, allowlist)
        assert sorted(models) == ["unet35_v1_foldA", "unet35_v1_foldB"] and policy == "all_folds_mean"


def test_selection_refuses_models_that_saw_the_subject_or_t0(tmp_path):
    allowlist = make_allowlist(tmp_path)
    contact = sorted(A_TRAIN+B_TRAIN)
    both = dict(segmenter_id="unet_both_ens", fold=0, member_model_sha256=[H("both")], label_contact_subjects=contact,
                label_contact_subjects_sha256=subject_list_sha256(contact))
    t0 = dict(segmenter_id="unet_t0_ens", fold=0, member_model_sha256=[H("t0")], label_contact_subjects=T0,
              label_contact_subjects_sha256=subject_list_sha256(T0))
    registry = _registry(tmp_path, {"unet_both_f0": both, "unet_t0_f0": t0})
    with pytest.raises(ModelSelectionRefused):
        select_fold_models(registry, A_TRAIN[0], allowlist, segmenter_id="unet_both_ens")
    with pytest.raises(ModelSelectionRefused):
        select_fold_models(registry, DEV[0], allowlist, segmenter_id="unet_t0_ens")
    with pytest.raises(ModelSelectionRefused):
        select_fold_models(registry, DEV[0], allowlist, segmenter_id="not_registered")


def test_inference_refuses_subjects_off_the_allowlist(tmp_path):
    allowlist = make_allowlist(tmp_path)
    check_subject_admitted(DEV[0], allowlist)
    for subject in (SEALED, "S37", "OASIS_OAS1_901_MR1"):
        with pytest.raises(SubjectSealed):
            check_subject_admitted(subject, allowlist)


# ---- ensemble and read-back -----------------------------------------------------------------------

def test_ensemble_is_mean_of_tempered_softmax_and_renormalized():
    g = torch.Generator().manual_seed(0)
    z1, z2 = torch.randn(1, 5, 3, 4, 2, generator=g), torch.randn(1, 5, 3, 4, 2, generator=g)
    p = ensemble_probabilities(iter([(z1, 2.), (z2, .5)]))
    expected = (torch.softmax(z1/2., 1)+torch.softmax(z2/.5, 1))[0]/2
    assert torch.allclose(p, expected, atol=1e-6) and torch.allclose(p.sum(0), torch.ones(3, 4, 2), atol=1e-6)
    with pytest.raises(InputValidationError):
        ensemble_probabilities(iter([(z1, 0.)]))
    with pytest.raises(InputValidationError):
        ensemble_probabilities(iter([]))


def test_read_probabilities_renormalizes_float16_and_checks_hashes(tmp_path):
    rng = np.random.default_rng(3)
    groups = rng.dirichlet(np.ones(14), size=(4, 5, 6)).transpose(3, 0, 1, 2).astype(np.float16)
    np.savez_compressed(tmp_path/GROUPS_FILE, p=groups)
    np.savez_compressed(tmp_path/PROBS_FILE, p=groups[:2])
    digest = {name: hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in (GROUPS_FILE, PROBS_FILE)}
    write_json(tmp_path/"provenance.json", dict(output_sha256=digest))
    array, _ = read_probabilities(tmp_path)
    assert array.dtype == np.float64 and np.max(np.abs(array.sum(0)-1)) < 1e-12
    np.savez_compressed(tmp_path/GROUPS_FILE, p=groups[::-1])
    with pytest.raises(ProvenanceRejected):
        read_probabilities(tmp_path)


def test_read_probabilities_checks_the_raw_sum_before_renormalizing(tmp_path):
    """C03.2-4a (D39; PS2): the same raw-sum read-back as structure_layer.read_structure_groups -- a float16 file
    whose channels sum to 0.7 is refused (groups and probs36), a deviation within float16 rounding is accepted."""
    base = np.zeros((3, 4, 5, 6))
    base[0], base[1], base[2] = .25, .5, .25                                  # exact in float16
    within = base.copy()
    within[:, 1, 2, 3] = (.5+2**-11, .5, 0.)                                 # |sum - 1| = 2^-11 ~ 4.9e-4

    def write(name, groups, probs):
        directory = tmp_path/name
        directory.mkdir()
        np.savez_compressed(directory/GROUPS_FILE, p=groups.astype(np.float16))
        np.savez_compressed(directory/PROBS_FILE, p=probs.astype(np.float16))
        write_json(directory/"provenance.json", dict(output_sha256={f: file_sha256(directory/f)
                                                                     for f in (GROUPS_FILE, PROBS_FILE)}))
        return directory
    accepted = write("within", within, within)
    for which in ("groups", "probs36"):
        array, _ = read_probabilities(accepted, which)
        assert array.dtype == np.float64 and np.max(np.abs(array.sum(0)-1)) <= 1e-15
    with pytest.raises(InputValidationError, match=r"raw channel sum .* > 0\.001 \(float16 storage\)"):
        read_probabilities(write("scaled_groups", .7*base, base))
    with pytest.raises(InputValidationError, match=r"raw channel sum .* > 0\.001 \(float16 storage\)"):
        read_probabilities(write("scaled_probs36", base, .7*base), "probs36")


# ---- infer machine gate (A4 review item 10) ---------------------------------------------------------

def fake_run(units="", pids="", util="3", unit_rc=0, calls=None):
    """subprocess.run stand-in for machine_gate: systemctl prints `units`, nvidia-smi prints `pids` / `util`."""
    import subprocess

    def run(cmd, **kwargs):
        if calls is not None:
            calls.append(list(cmd))
        if cmd[0] == "systemctl":
            return subprocess.CompletedProcess(cmd, unit_rc, units, "no bus" if unit_rc else "")
        if "--query-compute-apps=pid" in cmd:
            return subprocess.CompletedProcess(cmd, 0, pids, "")
        return subprocess.CompletedProcess(cmd, 0, util, "")
    return run


def test_machine_gate_passes_on_a_quiet_machine():
    from bayesreg.seg.unet import cli
    calls = []
    record = cli.machine_gate("cpu", run=fake_run(calls=calls))
    assert record["passed"] is True and record["active_units"] == [] and len(calls) == 1      # cpu: GPU not queried
    assert calls[0][:3] == ["systemctl", "--user", "list-units"] and set(cli.TIMED_UNIT_PATTERNS) <= set(calls[0])
    assert {"seq-vp-*", "tag-d0-*", "tag-d0a4-*", "tag-unet35-train*"} == set(cli.TIMED_UNIT_PATTERNS)
    record = cli.machine_gate("cuda", run=fake_run(util="20\n"))
    assert record["passed"] is True and record["gpu_utilization_percent"] == 20 and record["gpu_compute_pids"] == []


MACHINE_BUSY = {                           # case: (device, fake_run keywords, message fragment)
    "D0 queue unit": ("cpu", dict(units="tag-d0a4-queue.service loaded active running x\n"), "tag-d0a4-queue.service"),
    "training unit": ("cuda", dict(units="tag-unet35-train.service loaded active running x\n"), "tag-unet35-train"),
    "paper-1 chain": ("cpu", dict(units="seq-vp-chain.service loaded activating start x\n"), "seq-vp-chain.service"),
    "systemctl fails": ("cpu", dict(unit_rc=1), "systemctl --user list-units failed"),
    "GPU compute process": ("cuda", dict(pids="4242\n"), "GPU busy"),
    "GPU utilization": ("cuda", dict(util="21"), "GPU busy"),
    "no GPU reading": ("cuda", dict(util="[N/A]"), "nvidia-smi gave no reading"),
}


@pytest.mark.parametrize("case", sorted(MACHINE_BUSY))
def test_machine_gate_refuses(case):
    from bayesreg.seg.unet import cli
    device, kwargs, fragment = MACHINE_BUSY[case]
    with pytest.raises(cli.MachineBusy, match=fragment.replace(".", r"\.")):
        cli.machine_gate(device, run=fake_run(**kwargs))


def test_infer_runs_the_gate_before_reading_anything(tmp_path, monkeypatch, capsys):
    """A real `infer` (no --synthetic) stops at the gate before the allowlist, registry or any image is read;
    --skip-machine-gate skips it (and says so)."""
    from bayesreg.seg.unet import cli, pins
    real = cli.machine_gate
    monkeypatch.setattr(cli, "machine_gate", lambda device: real(device, run=fake_run(
        units="tag-d0a4-A4p-0438-0439.service loaded active running x\n")))

    def no_reads(*args, **kwargs):
        raise AssertionError("nothing may be read before the machine gate")
    monkeypatch.setattr(pins, "load_pinned_allowlist", no_reads)
    with pytest.raises(cli.MachineBusy, match="timed runs or training in progress"):
        cli.main(["infer", "--subjects", "OASIS_OAS1_0913_MR1", "--device", "cpu", "--out", str(tmp_path)])
    with pytest.raises(AssertionError, match="before the machine gate"):     # skipped: goes on to the allowlist
        cli.main(["infer", "--subjects", "OASIS_OAS1_0913_MR1", "--device", "cpu", "--out", str(tmp_path),
                  "--skip-machine-gate"])
    assert "machine gate skipped (--skip-machine-gate)" in capsys.readouterr().out
