"""python -m bayesreg.learn.cli on synthetic split files: dry run (guard + file presence, exp-log procedure files), refusals and
exit codes, eta0 rules.  No image is read and nothing is solved, except the structure-layer read-back
(load_structure_layer, C03.2-4a raw channel sum) on a synthetic 26^3 pair in tmp_path."""
import argparse
import json

import numpy as np
import pytest

from jd_curl.validation import InputValidationError

from bayesreg.cli import run_pair
from bayesreg.io import ProvenanceRejected, load_dev_allowlist, load_pair, load_segmenter_registry
from bayesreg.learn import EBConfig, EBPair, alpha_nugget
from bayesreg.learn import cli
from bayesreg.seg import structure_layer
from bayesreg.seg.unet.pins import PINNED_FOLDS_SHA256
from bayesreg.seg.unet.synthetic import oasis

from ..io_seg._helpers import DEV, SYN_FOLD0, SYN_FOLD1, synthetic_folds
from ..solve.test_run_pair_structure import env, make_pair, registry_models, structure_groups, write_structure  # noqa: F401
from . import _guard_fixtures as gf


def argv(paths, out, *extra, eta0=None, pairs=f"{oasis(901)}__{oasis(902)},{oasis(906)}__{oasis(907)}"):
    return ["--out", str(out), "--pairs-root", str(out.parent/"pairs"), "--eta0-json", str(eta0),
            "--pairs", pairs, "--eb-pairs", str(paths["eb_pairs"]), "--folds", str(paths["folds"]),
            "--allowlist", str(paths["allowlist"]), "--registry", str(paths["registry"]),
            "--tissue-root", str(out.parent/"tissue"), "--structure-root", str(out.parent/"structure"), "--synthetic",
            *extra]


def eta_file(tmp_path, document=None):
    path = tmp_path/"eta0.json"
    path.write_text(json.dumps(document or {"nu": 2., "tau": [1., .5, .5, .5]}), encoding="utf-8")
    return path


def test_dry_run_writes_the_record_and_plan(tmp_path, monkeypatch):
    paths = gf.write_files(tmp_path/"split")
    monkeypatch.setattr(structure_layer, "REPO_REGISTRY", paths["registry"])     # the repository registry of this test
    out = tmp_path/"eb_v1"/"run1"
    code = cli.main(argv(paths, out, "--dry-run", eta0=eta_file(tmp_path)))
    assert code == 0
    status = json.loads((out/"status.json").read_text())
    assert status["status"] == "dry_run" and status["pairs"] == 2 and status["missing"] == 2
    assert status["structure_registry"] == dict(status="registered", path=str(paths["registry"]), staging_opt_in=False,
                                                development_diagnostic_only=False, git=dict(tracked=False, clean=False))
    for name in ("cmd.txt", "git.txt", "env.txt", "data.txt", "config.json", "plan.json", "timing.json", "log.txt"):
        assert (out/name).is_file(), name
    data = json.loads((out/"data.txt").read_text())
    assert data["admission"]["admitted"] == [f"{oasis(901)}__{oasis(902)}", f"{oasis(906)}__{oasis(907)}"]
    assert len(data["registry_sha256"]) == 64
    assert data["structure_registry"]["status"] == "registered" and data["structure_registry"]["use"] == "registered"
    assert data["structure_registry"]["sha256"] == data["registry_sha256"]
    config = json.loads((out/"config.json").read_text())
    assert config["eb"]["parametrization"] == "P2" and config["eta0"]["alpha"] == pytest.approx(1e-2)
    plan = json.loads((out/"plan.json").read_text())
    assert all(len(files) == 8 for files in plan["missing"].values())
    with pytest.raises(FileExistsError):
        cli.main(argv(paths, out, "--dry-run", eta0=eta_file(tmp_path)))       # nothing is overwritten


def test_refused_pair_and_missing_registry_exit_codes(tmp_path):
    paths = gf.write_files(tmp_path/"split")
    out = tmp_path/"r1"
    assert cli.main(argv(paths, out, "--dry-run", eta0=eta_file(tmp_path), pairs=f"{oasis(903)}__{oasis(908)}")) == 5
    assert json.loads((out/"status.json").read_text())["status"] == "eb_pair_refused"
    paths["registry"].unlink()
    out = tmp_path/"r2"
    assert cli.main(argv(paths, out, eta0=eta_file(tmp_path))) == 4               # structure layer unregistered
    assert json.loads((out/"status.json").read_text())["status"] == "provenance_rejected"
    out = tmp_path/"r3"
    assert cli.main(argv(paths, out, "--dry-run", eta0=eta_file(tmp_path))) == 0  # a dry run reports it
    assert json.loads((out/"status.json").read_text())["registry_present"] is False


def write_ensemble(registry_path, **changes):
    """ensemble.json beside a staging registry, as bayesreg.seg.unet.registry.register writes it."""
    document = dict(format="bayesreg.unet35.ensemble.v1", segmenter_id="unet35_v1_ens", registry_path=str(registry_path),
                    registry_sha256=load_segmenter_registry(registry_path).sha256, fold_models={}, members=[],
                    member_check=dict(format="bayesreg.unet35.member_check.v1", passed=True))
    document.update(changes)
    (registry_path.parent/"ensemble.json").write_text(json.dumps(document), encoding="utf-8")


def test_registry_rules_match_run_pair(tmp_path, monkeypatch):
    """Review PS1_PS2 item c: the default --registry is the repository one; another registry (the training chain's
    staging copy) is refused without --structure-registry-staging (also in a dry run, before any image), needs the
    ensemble.json check with it, and is then labelled staging_unregistered in status.json and data.txt."""
    paths = gf.write_files(tmp_path/"split")
    monkeypatch.setattr(structure_layer, "REPO_REGISTRY", tmp_path/"repo"/"segmenters.json")
    args = cli.parse_args(["--out", "o", "--pairs-root", "p", "--eta0-json", "e"])
    assert args.registry == str(cli.DEFAULT_REGISTRY) and args.structure_registry_staging is False
    args = cli.parse_args(["--out", "o", "--pairs-root", "p", "--eta0-json", "e", "--structure-registry-staging"])
    assert args.registry == str(cli.DEFAULT_STAGING_REGISTRY) == str(run_pair.DEFAULT_SEGMENTER_REGISTRY)
    out = tmp_path/"no_opt_in"
    assert cli.main(argv(paths, out, "--dry-run", eta0=eta_file(tmp_path))) == 4
    status = json.loads((out/"status.json").read_text())
    assert status["status"] == "provenance_rejected" and "--structure-registry-staging" in status["reason"]
    assert status["structure_registry"]["status"] == "staging_unregistered"
    assert status["structure_registry"]["staging_opt_in"] is False and not (out/"plan.json").exists()
    out = tmp_path/"no_ensemble"
    assert cli.main(argv(paths, out, "--dry-run", "--structure-registry-staging", eta0=eta_file(tmp_path))) == 4
    assert "no ensemble.json" in json.loads((out/"status.json").read_text())["reason"]
    write_ensemble(paths["registry"], registry_sha256="0"*64)
    out = tmp_path/"stale_ensemble"
    assert cli.main(argv(paths, out, "--dry-run", "--structure-registry-staging", eta0=eta_file(tmp_path))) == 4
    assert "registry_sha256" in json.loads((out/"status.json").read_text())["reason"]
    write_ensemble(paths["registry"])
    out = tmp_path/"staging"
    assert cli.main(argv(paths, out, "--dry-run", "--structure-registry-staging", eta0=eta_file(tmp_path))) == 0
    status = json.loads((out/"status.json").read_text())
    assert status["status"] == "dry_run" and status["structure_registry"]["development_diagnostic_only"] is True
    assert status["structure_registry"]["staging_opt_in"] is True
    data = json.loads((out/"data.txt").read_text())
    record = data["structure_registry"]
    assert record["status"] == "staging_unregistered" and record["use"].startswith("development diagnostics")
    assert record["staging_ensemble_check"]["member_check_passed"] is True and "git" in record


def test_eta0_rules(tmp_path):
    p2 = EBConfig()
    eta = cli.read_eta0(eta_file(tmp_path), p2)
    assert eta.alpha == alpha_nugget(eta.tau2) and eta.tau2 == (1., .25, .25, .25)
    with pytest.raises(InputValidationError, match="alpha_nug"):
        cli.read_eta0(eta_file(tmp_path, {"nu": 2., "tau": [1., .5], "alpha": .5}), p2)
    with pytest.raises(InputValidationError, match="needs alpha"):
        cli.read_eta0(eta_file(tmp_path, {"nu": 2., "tau": [1., .5]}), EBConfig(parametrization="P3", eb_count="dim"))
    assert cli.read_eta0(eta_file(tmp_path, {"nu": 2., "tau": [1., .5], "alpha": .5}),
                         EBConfig(parametrization="P3", eb_count="dim")).alpha == .5
    with pytest.raises(InputValidationError):
        cli.read_eta0(eta_file(tmp_path, {"nu": 2., "tau": [1.]}), p2)
    args = cli.parse_args(["--out", "o", "--pairs-root", "p", "--eta0-json", "e", "--learn-tau0"])
    assert cli.eb_config(args).learn_tau0 is True and cli.eb_config(args).parametrization == "P2"


def test_real_runs_require_the_pinned_split_files(tmp_path):
    paths = gf.write_files(tmp_path/"split")
    out = tmp_path/"pinned"
    args = [a for a in argv(paths, out, "--dry-run", eta0=eta_file(tmp_path)) if a != "--synthetic"]
    assert cli.main(args) == 6
    status = json.loads((out/"status.json").read_text())
    assert status["status"] == "pinned_input_refused" and "eb_pairs.tsv" in status["reason"]
    assert cli.PINNED_FOLDS_SHA256 == PINNED_FOLDS_SHA256


def test_pin_rules():
    real = EBPair("OASIS_OAS1_0291_MR1", "OASIS_OAS1_0107_MR1", "A", {"check": "test"})
    fake = EBPair(oasis(901), oasis(902), "A", {"check": "test"})
    synthetic = argparse.Namespace(synthetic=True)
    assert cli.pin_problem(synthetic, {}, [fake]) is None
    assert "synthetic" in cli.pin_problem(synthetic, {}, [real])
    pinned = dict(eb_pairs_sha256=cli.PINNED_EB_PAIRS_SHA256, folds_sha256=cli.PINNED_FOLDS_SHA256)
    real_run = argparse.Namespace(synthetic=False)
    assert cli.pin_problem(real_run, pinned, [real]) is None
    assert "folds.json" in cli.pin_problem(real_run, dict(pinned, folds_sha256="0"*64), [real])


@synthetic_folds("env")
def test_load_structure_layer_checks_the_raw_sum_before_renormalizing(env):                  # noqa: F811
    """C03.2-4a (D39; PS2): load_structure_layer reads back through the same raw-sum check as
    structure_layer.read_structure_groups -- a float16 file whose channels sum to 0.7 is refused, the float16 cast of
    a simplex (deviation within rounding) is accepted and divided by its sum."""
    root, allowlist, registry = env
    models = registry_models(SYN_FOLD0, SYN_FOLD1)
    grid, pair_name = make_pair(root, DEV[0], DEV[1], models, structure=False)
    write_structure(root, grid, DEV[0], models)
    write_structure(root, grid, DEV[1], models, array=.7*structure_groups(grid))
    pair = load_pair(run_pair.resolve_pair(root/"data", pair_name), allowlist=allowlist)
    args = (root/"structure", load_segmenter_registry(registry), load_dev_allowlist(allowlist))
    array, _, record = cli.load_structure_layer(pair, "fixed", *args)
    read_back = record["read_back"]
    assert read_back["stored_dtype"] == "float16" and read_back["raw_sum_tol"] == 1e-3
    assert 0 < read_back["raw_max_abs_sum_deviation"] <= 1e-3
    assert array.dtype == np.float64 and np.max(np.abs(array.sum(axis=0)-1)) <= 1e-12
    with pytest.raises(ProvenanceRejected, match=r"moving: .*raw channel sum .* > 0\.001 \(float16 storage\)"):
        cli.load_structure_layer(pair, "moving", *args)


def test_eb_count_option(tmp_path, monkeypatch, capsys):
    """C15-9 / D45: --eb-count, default gull_mackay with P2 (BZ 2026-10-30, EBD); dim explicit; both go to EBConfig,
    config.json (args and eb) and the dry-run plan.  gull_mackay with P1 (C15-9 gate (iv)) or P3 (EBC3, review R2
    important 1), given or by default, is refused when the arguments are parsed (exit 2, the message naming
    --eb-count dim, no run directory); P1 / P3 run with --eb-count dim."""
    base = ["--out", "o", "--pairs-root", "p", "--eta0-json", "e"]
    args = cli.parse_args(base)
    assert args.eb_count == "gull_mackay" and args.parametrization == "P2"
    assert cli.eb_config(args).gull_mackay and cli.eb_config(args) == EBConfig()
    assert cli.eb_config(cli.parse_args(base+["--eb-count", "gull_mackay"])).gull_mackay
    assert cli.eb_config(cli.parse_args(base+["--eb-count", "dim"])).eb_count == "dim"
    for extra in (["--parametrization", "P1"], ["--parametrization", "P3"],
                  ["--eb-count", "gull_mackay", "--parametrization", "P1"],
                  ["--eb-count", "gull_mackay", "--parametrization", "P3"]):
        with pytest.raises(SystemExit) as exc:
            cli.parse_args(base+extra)
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert f"--parametrization {extra[-1]} needs --eb-count dim" in err and "C15-9" in err, extra
    for name in ("P1", "P3"):
        config = cli.eb_config(cli.parse_args(base+["--parametrization", name, "--eb-count", "dim"]))
        assert config.parametrization == name and config.eb_count == "dim"
    with pytest.raises(SystemExit):
        cli.parse_args(base+["--eb-count", "mackay"])
    paths = gf.write_files(tmp_path/"split")
    monkeypatch.setattr(structure_layer, "REPO_REGISTRY", paths["registry"])     # the repository registry of this test
    out = tmp_path/"eb_default"
    assert cli.main(argv(paths, out, "--dry-run", eta0=eta_file(tmp_path))) == 0
    config = json.loads((out/"config.json").read_text())
    assert config["eb"]["eb_count"] == "gull_mackay" and config["args"]["eb_count"] == "gull_mackay"
    assert "prior PCG" in json.loads((out/"plan.json").read_text())["work"]
    out = tmp_path/"eb_gm"
    assert cli.main(argv(paths, out, "--dry-run", "--eb-count", "gull_mackay", eta0=eta_file(tmp_path))) == 0
    assert json.loads((out/"config.json").read_text())["eb"] == config["eb"]
    out = tmp_path/"eb_dim"
    assert cli.main(argv(paths, out, "--dry-run", "--eb-count", "dim", eta0=eta_file(tmp_path))) == 0
    config = json.loads((out/"config.json").read_text())
    assert config["eb"]["eb_count"] == "dim" and config["args"]["eb_count"] == "dim"
    assert "prior" not in json.loads((out/"plan.json").read_text())["work"]
    out = tmp_path/"eb_p3_default"
    with pytest.raises(SystemExit):
        cli.main(argv(paths, out, "--dry-run", "--parametrization", "P3", eta0=eta_file(tmp_path)))
    assert not out.exists()
