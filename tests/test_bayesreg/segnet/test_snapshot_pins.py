"""Code snapshot (git archive into a fresh directory, verification, code record), the pinned inputs of
real runs and the --synthetic switch, and the launcher script's syntax and refusals. Temporary git
repositories and files only; nothing under data/ or /mnt/d is read."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from bayesreg.seg.unet import pins
from bayesreg.seg.unet.cli import main
from bayesreg.seg.unet.data import LabelGuard
from bayesreg.seg.unet.pins import (PINNED_FOLDS_SHA256, PinnedInputRefused, load_pinned_inputs,
                                    load_synthetic_inputs)
from bayesreg.seg.unet.snapshot import (CODE_ROOT, CodeNotClean, code_record, make_snapshot, require_clean_code,
                                        require_snapshot, verify_snapshot)
from bayesreg.seg.unet.synthetic import allowlist_document, folds_document, oasis, write_json

from ._helpers import A_CALIB, A_TRAIN, B_CALIB, B_TRAIN, DEV, HOLDOUT, T0, TRAIN_SOURCE, TRAINING

SCRIPT = CODE_ROOT/"experiments"/"segnet"/"run_train.sh"


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path/"repo"
    (root/"pkg"/"sub").mkdir(parents=True)
    (root/"pkg"/"__init__.py").write_text("x = 1\n")
    (root/"pkg"/"sub"/"data.json").write_text("{}\n")
    (root/"other").mkdir()
    (root/"other"/"secret.txt").write_text("not archived\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def test_snapshot_archives_only_the_paths_and_verifies(repo, tmp_path):
    record = make_snapshot(repo, tmp_path/"snap", paths=("pkg",))
    assert record["commit"] == _git(repo, "rev-parse", "HEAD") and sorted(record["files"]) == [
        "pkg/__init__.py", "pkg/sub/data.json"]
    assert not (tmp_path/"snap"/"other").exists() and not (tmp_path/"snap.partial").exists()
    verified = verify_snapshot(tmp_path/"snap")
    assert verified["kind"] == "snapshot" and verified["commit"] == record["commit"]
    with pytest.raises(FileExistsError):                                    # never overwritten
        make_snapshot(repo, tmp_path/"snap", paths=("pkg",))


@pytest.mark.parametrize("change", ["edit", "extra", "missing"])
def test_snapshot_verification_refuses_changes(repo, tmp_path, change):
    make_snapshot(repo, tmp_path/"snap", paths=("pkg",))
    if change == "edit":
        (tmp_path/"snap"/"pkg"/"__init__.py").write_text("x = 2\n")
    elif change == "extra":
        (tmp_path/"snap"/"pkg"/"new.py").write_text("")
    else:
        (tmp_path/"snap"/"pkg"/"sub"/"data.json").unlink()
    with pytest.raises(CodeNotClean):
        verify_snapshot(tmp_path/"snap")


@pytest.mark.parametrize("dirt", ["tracked", "untracked"])
def test_snapshot_refuses_a_dirty_repository(repo, tmp_path, dirt):
    if dirt == "tracked":
        (repo/"pkg"/"__init__.py").write_text("x = 3\n")
    else:
        (repo/"new.txt").write_text("")
    with pytest.raises(CodeNotClean):
        make_snapshot(repo, tmp_path/"snap", paths=("pkg",))
    assert not (tmp_path/"snap").exists()


def test_code_record_and_requirements():
    record = code_record()
    assert record["kind"] in ("git_worktree", "snapshot") and record["imports_inside_root"]
    assert "commit" in record and Path(record["imported_from"]["bayesreg"]).parent == CODE_ROOT
    with pytest.raises(CodeNotClean):
        require_clean_code(dict(record, kind="git_worktree", clean=False, n_dirty=1, dirty_paths=[" M x.py"]))
    with pytest.raises(CodeNotClean):
        require_clean_code(dict(record, clean=True, imports_inside_root=False))
    with pytest.raises(CodeNotClean):
        require_snapshot(dict(record, kind="git_worktree", clean=True))
    assert require_snapshot(dict(record, kind="snapshot", clean=True))["kind"] == "snapshot"


# ---- pinned inputs --------------------------------------------------------------------------------

CLEAN = dict(kind="git_worktree", clean=True, commit="0"*40, root=str(CODE_ROOT), imports_inside_root=True)


def _sources_tree(tmp_path, training=TRAINING):
    """Allowlist whose sources are real files under tmp_path/src (so verify_allowlist_sources can pass)."""
    root = tmp_path/"src"
    sources = [("synthetic_dev", DEV, False), ("synthetic_t0", T0, True), (TRAIN_SOURCE, training, False)]
    items = []
    for name, ids, never in sources:
        path = root/"lists"/f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{s}\n" for s in ids))
        items.append(dict(name=name, dataset="OASIS", path=f"lists/{name}.txt",
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(), never_for_training=never,
                          subjects=list(ids)))
    union = []
    for item in items:
        union += [s for s in item["subjects"] if s not in union]
    allow = write_json(tmp_path/"dev_subjects.json", dict(schema="bayesreg.dev_subjects.v1", subjects={"OASIS": union},
                                                          sources=items))
    folds = write_json(tmp_path/"folds.json", folds_document(A_TRAIN, A_CALIB, B_TRAIN, B_CALIB, HOLDOUT))
    return root, allow, folds


@pytest.fixture()
def pinned(tmp_path, monkeypatch):
    """The pins redirected to a synthetic folds file / allowlist (the real files are never opened here)."""
    root, allow, folds = _sources_tree(tmp_path)
    monkeypatch.setattr(pins, "PINNED_FOLDS_SHA256", hashlib.sha256(folds.read_bytes()).hexdigest())
    monkeypatch.setattr(pins, "DEFAULT_ALLOWLIST", allow)
    monkeypatch.setattr(pins, "TRAINING_SOURCE", TRAIN_SOURCE)
    monkeypatch.setattr(pins, "PINNED_COUNTS", (2, 1))
    return root, allow, folds


def test_pinned_inputs_pass_with_the_pinned_files(pinned):
    root, allow, folds = pinned
    split, allowlist, record = pins.load_pinned_inputs(folds, allow, training_source=TRAIN_SOURCE, expect_counts=(2, 1),
                                                       source_root=root, code=CLEAN)
    assert record["mode"] == "pinned" and record["folds_sha256"] == pins.PINNED_FOLDS_SHA256
    assert sorted(record["allowlist_sources_verified"]) == sorted(["synthetic_dev", "synthetic_t0", TRAIN_SOURCE])
    assert LabelGuard(split, allowlist, training_source=TRAIN_SOURCE).record()["training_source"] == TRAIN_SOURCE


@pytest.mark.parametrize("case", ["folds_sha", "allowlist_path", "source_hash", "source_name", "counts", "dirty"])
def test_pinned_inputs_refuse_departures(pinned, tmp_path, case):
    root, allow, folds = pinned
    kwargs = dict(training_source=TRAIN_SOURCE, expect_counts=(2, 1), source_root=root, code=CLEAN)
    if case == "folds_sha":
        folds.write_text(folds.read_text().replace("synthetic.folds", "synthetic.folds2"))
    elif case == "allowlist_path":
        allow = write_json(tmp_path/"copy.json", json.loads(allow.read_text()))
    elif case == "source_hash":
        (root/"lists"/f"{TRAIN_SOURCE}.txt").write_text("changed\n")
    elif case == "source_name":
        kwargs["training_source"] = "synthetic_dev"
    elif case == "counts":
        kwargs["expect_counts"] = (45, 8)
    else:
        kwargs["code"] = dict(CLEAN, clean=False, n_dirty=1, dirty_paths=["?? x"])
    with pytest.raises((PinnedInputRefused, CodeNotClean)):
        pins.load_pinned_inputs(folds, allow, **kwargs)


def test_real_pins_are_the_committed_split():
    assert PINNED_FOLDS_SHA256 == "98e9e676053d4e9386bc775c5d4cfc6be097afca9919c655ccb554d6236bf245"
    assert pins.TRAINING_SOURCE == "bayesreg_train_v1" and pins.PINNED_COUNTS == (45, 8)
    with pytest.raises(PinnedInputRefused):                    # a non-pinned folds file is refused before parsing
        load_pinned_inputs(Path(__file__), pins.DEFAULT_ALLOWLIST, code=CLEAN)
    with pytest.raises(PinnedInputRefused):
        load_pinned_inputs(pins.DEFAULT_FOLDS, pins.DEFAULT_ALLOWLIST, training_source="synthetic_bline_train",
                           code=CLEAN)


def test_synthetic_switch_admits_only_synthetic_subjects_and_data(tmp_path):
    _, allow, folds = _sources_tree(tmp_path)
    split, _, record = load_synthetic_inputs(folds, allow, data_root=tmp_path/"data", training_source=TRAIN_SOURCE)
    assert record["mode"] == "synthetic" and split.training_subjects == frozenset(TRAINING)
    for root in ("/mnt/d/data", "/mnt/d/data/OASIS_OAS1_0001_MR1", str(CODE_ROOT/"data")):
        with pytest.raises(PinnedInputRefused):
            load_synthetic_inputs(folds, allow, data_root=Path(root), training_source=TRAIN_SOURCE)
    real = write_json(tmp_path/"real_folds.json", folds_document([oasis(101)]+A_TRAIN, A_CALIB, B_TRAIN, B_CALIB))
    with pytest.raises(PinnedInputRefused):
        load_synthetic_inputs(real, allow, data_root=tmp_path/"data", training_source=TRAIN_SOURCE)
    real_allow = write_json(tmp_path/"real_allow.json", allowlist_document(
        [("dev", [oasis(438)], False), (TRAIN_SOURCE, TRAINING, False)]))
    with pytest.raises(PinnedInputRefused):
        load_synthetic_inputs(folds, real_allow, data_root=tmp_path/"data", training_source=TRAIN_SOURCE)


def test_cli_check_inputs_without_synthetic_refuses_synthetic_files(tmp_path):
    _, allow, folds = _sources_tree(tmp_path)
    with pytest.raises((PinnedInputRefused, CodeNotClean)):
        main(["check-inputs", "--folds", str(folds), "--allowlist", str(allow)])
    assert main(["check-inputs", "--synthetic", "--training-source", TRAIN_SOURCE, "--folds", str(folds),
                 "--allowlist", str(allow), "--expect-counts", "2,1", "--data-root", str(tmp_path/"data")]) == 0


# ---- launcher script ------------------------------------------------------------------------------

def test_run_train_script_syntax_and_chain_refuses_outside_a_snapshot(tmp_path):
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0
    result = subprocess.run(["bash", str(SCRIPT), "chain", "--out", str(tmp_path)], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 1
    assert "chain must run from the snapshot" in (tmp_path/"chain.log").read_text()
    usage = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, timeout=60)
    assert usage.returncode == 2 and "launch" in usage.stdout
    text = SCRIPT.read_text()
    for needle in ("tag-unet35-train", "is-active --quiet tag-d0-queue", "MIN_FREE_MIB:-14336", "status --porcelain",
                   "snapshot make", "check-inputs --require-snapshot", "check-members", "evaluate-calibration"):
        assert needle in text
