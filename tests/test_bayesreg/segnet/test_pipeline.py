"""End to end on synthetic NIfTI subjects (CPU, tiny network): train -> calibrate -> register -> infer,
then the P_S provenance through bayesreg.io.validate_segmentation_provenance for a development pair
(accepted), a same-fold training pair (accepted) and a cross-fold training pair (rejected); and the
no-overwrite rules."""
import json

import pytest

from bayesreg.seg.unet.calibrate import calibrate_member
from bayesreg.seg.unet.cli import dry_run, main
from bayesreg.seg.unet.data import OasisFiles
from bayesreg.seg.unet.train import train_member

from ._helpers import TINY


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return dry_run(tmp_path_factory.mktemp("dry"), device="cpu", config=TINY, log=lambda *_: None)


def test_dry_run_passes_all_checks(run):
    assert run["ok"], json.dumps(run, indent=1, default=str)
    assert run["pair_checks"]["dev_t0"] == "accepted" and run["pair_checks"]["same_fold_A"] == "accepted"
    assert run["pair_checks"]["cross_fold"].startswith("provenance_rejected")
    assert all(v.startswith("refused") for v in run["refusals"].values())


def test_dry_run_outputs_and_provenance_fields(run):
    from pathlib import Path
    work = Path(run["work"])
    prov = json.loads((work/"models"/"pred"/"OASIS_OAS1_0907_MR1"/"provenance.json").read_text())
    assert prov["layer"] == "structure" and prov["trained_with_labels"] is True and prov["test_labels_seen"] is False
    assert prov["segmenter_id"] == "unet35_v1_ens" and len(prov["members"]) == 4
    assert prov["class_names"][0] == "cerebral_wm" and len(prov["class_names"]) == 13
    assert set(prov["output_sha256"]) == {"probs36.npz", "structure_groups.npz"}
    assert prov["grouping_check"]["grouping_status"] == "default"
    for fold, seed in (("A", 0), ("A", 1), ("B", 0), ("B", 1)):
        member = work/"models"/f"fold{fold}_s{seed}"
        record = json.loads((member/"train_record.json").read_text())
        assert record["label_contact_subjects"] == sorted(record["train_subjects"]+record["calib_subjects"])
        # this member's reads only (the dry run shares one guard across the four members)
        reads = [r["subject"] for r in record["label_guard"]["label_reads"]]
        assert sorted(reads) == record["label_contact_subjects"]
        assert record["label_guard"]["training_source"] == "synthetic_bline_train"
        environment = record["environment"]
        assert {"scipy", "cudnn", "code"} <= set(environment) and environment["code"]["kind"] in ("git_worktree",
                                                                                                   "snapshot")
        calibration = json.loads((member/"calibration.json").read_text())
        assert calibration["fit"]["temperature"] > 0
        assert calibration["fit"]["nll_after"] <= calibration["fit"]["nll_before"]
        calib_reads = [r["subject"] for r in calibration["label_guard"]["label_reads"]]
        assert calib_reads == record["calib_subjects"] == calibration["calib_subjects"]   # exactly this fold's calib


def test_dry_run_cross_fold_calibration(run):
    from pathlib import Path
    from bayesreg.seg.unet.crosscal import evaluate_cross_fold_calibration
    work = Path(run["work"])
    document = json.loads((work/"models"/"calibration_crossfit.json").read_text())
    assert document["ensemble_temperature_applied"] is False
    assert {f: r["models_fold"] for f, r in document["results"].items()} == {"A": "B", "B": "A"}
    for fold, result in document["results"].items():
        assert result["target_subjects"] == {"A": ["OASIS_OAS1_0903_MR1"], "B": ["OASIS_OAS1_0906_MR1"]}[fold]
        assert len(result["members"]) == 2 and result["ensemble"]["n_voxels"] > 0
        assert result["ensemble"]["nll"] > 0 and 0 <= result["ensemble"]["ece"] <= 1
    assert sorted(r["subject"] for r in document["label_guard"]["label_reads"]) == ["OASIS_OAS1_0903_MR1",
                                                                                     "OASIS_OAS1_0906_MR1"]
    with pytest.raises(FileExistsError):
        evaluate_cross_fold_calibration(files=None, guard=None, models_root=work/"models", split=None, allowlist=None,
                                        training_source="synthetic_bline_train", log=lambda *_: None)


def test_cli_check_members_and_register_on_the_dry_run(run, tmp_path):
    from pathlib import Path
    work = Path(run["work"])
    common = ["--synthetic", "--training-source", "synthetic_bline_train", "--folds", str(work/"folds.json"),
              "--allowlist", str(work/"dev_subjects.json"), "--expect-counts", "2,1", "--data-root", str(work/"data"),
              "--out", str(work/"models")]
    assert main(["check-members", *common, "--report", str(tmp_path/"check.json")]) == 0
    check = json.loads((tmp_path/"check.json").read_text())
    assert check["passed"] and len(check["members"]) == 4 and check["inputs"]["mode"] == "synthetic"
    assert main(["register", *common, "--registry", str(tmp_path/"segmenters.json")]) == 0
    assert json.loads((tmp_path/"segmenters.json").read_text())["models"] == \
        json.loads((work/"segmenters.json").read_text())["models"]
    with pytest.raises(Exception):                          # real-data mode: the synthetic folds are not pinned
        main(["check-members", "--folds", str(work/"folds.json"), "--allowlist", str(work/"dev_subjects.json"),
              "--out", str(work/"models"), "--report", str(tmp_path/"check2.json")])
    assert not (tmp_path/"check2.json").exists()


def test_members_are_never_overwritten(run, tmp_path):
    from pathlib import Path
    from ._helpers import make_guard
    work = Path(run["work"])
    guard = make_guard(tmp_path)
    with pytest.raises(FileExistsError):
        train_member(files=OasisFiles(work/"data"), guard=guard, fold="A", seed=0, models_root=work/"models",
                     config=TINY, device="cpu", log=lambda *_: None)
    with pytest.raises(FileExistsError):
        calibrate_member(files=OasisFiles(work/"data"), guard=guard, fold="A", seed=0, models_root=work/"models",
                         device="cpu", log=lambda *_: None)


def test_cli_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = capsys.readouterr().out
    for command in ("train", "calibrate", "check-members", "register", "evaluate-calibration", "infer",
                    "check-tissue4", "--dry-run"):
        assert command in text


def test_check_tissue4_reads_training_subjects_only_and_reports(run, tmp_path, capsys):
    from pathlib import Path
    work = Path(run["work"])
    report = tmp_path/"check.json"
    assert main(["check-tissue4", "--synthetic", "--training-source", "synthetic_bline_train",
                 "--folds", str(work/"folds.json"), "--allowlist", str(work/"dev_subjects.json"),
                 "--data-root", str(work/"data"), "--expect-counts", "2,1", "--n-subjects", "2",
                 "--report", str(report)]) == 0
    result = json.loads(report.read_text())
    assert result["mismatched_labels"] == []            # synthetic seg4 was written from tissue4_ref
    assert result["subjects"] == ["OASIS_OAS1_0901_MR1", "OASIS_OAS1_0902_MR1"]
    reads = result["guard"]["label_reads"]
    assert sorted((r["subject"], r["kind"]) for r in reads) == sorted(
        (s, k) for s in result["subjects"] for k in ("seg35", "seg4"))
    assert set(result["adjacency"]["claims"]) == {"accumbens_vs_ventral_dc", "vessel_vs_choroid_plexus",
                                                  "inf_lat_ventricle_vs_3rd_4th_ventricle"}
    assert result["inputs"]["mode"] == "synthetic"
    with pytest.raises(FileExistsError):                # reports are never overwritten
        main(["check-tissue4", "--synthetic", "--training-source", "synthetic_bline_train",
              "--folds", str(work/"folds.json"), "--allowlist", str(work/"dev_subjects.json"),
              "--data-root", str(work/"data"), "--expect-counts", "2,1", "--n-subjects", "1", "--report", str(report)])


def test_check_tissue4_reads_at_most_three_subjects(run, tmp_path):
    from pathlib import Path
    work = Path(run["work"])
    with pytest.raises(SystemExit):
        main(["check-tissue4", "--synthetic", "--training-source", "synthetic_bline_train",
              "--folds", str(work/"folds.json"), "--allowlist", str(work/"dev_subjects.json"),
              "--data-root", str(work/"data"), "--expect-counts", "2,1", "--n-subjects", "4",
              "--report", str(tmp_path/"r.json")])
    assert not (tmp_path/"r.json").exists()
