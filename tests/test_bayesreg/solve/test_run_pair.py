"""bayesreg.cli.run_pair on synthetic files only (tmp_path; no data/, no real segmentation).

- the subject guard: a pair with a subject off the (synthetic) allowlist ends with subject_sealed (exit 3)
  before pair.npz is opened; no state, no disp;
- the run directory is never reused (nothing overwritten);
- a provenance record with test_labels_seen = true ends with provenance_rejected (exit 4);
- end to end on a 26^3 synthetic pair with Atropos-like tissue posteriors (NIfTI + provenance bound to
  the manifest's source hashes), channel_source tissue, two accepted steps per (level, round):
  cap_reached ("上限收尾"), every C16 / exp-log procedure file present, cert.json a fresh CPU PASS.
"""
import json

import numpy as np
import pytest

from jd_curl.three_d.nifti import file_sha256, save_scalar

from bayesreg.cli import run_pair
from bayesreg.io import pair as pair_module

from . import _synthetic as syn
from ..io_seg._helpers import DEV, SEALED, allowlist_document, source_sha, write_json

N = 26
STAGE = "stage_synthetic_v1"


def write_synthetic_pair(root, fixed_subject, moving_subject):
    grid = syn.cube_grid(N)
    moving = syn.moving_image(grid.identity())+.5
    fixed = syn.warp_with_spline(moving, grid, syn.solenoidal_displacement(grid, .6))
    name = f"{fixed_subject}__{moving_subject}"
    directory = root/"data"/STAGE/name
    directory.mkdir(parents=True)
    labels = np.zeros(grid.shape)
    np.savez_compressed(directory/"pair.npz", fixed=fixed, moving=moving, initial_u=np.zeros(grid.shape+(3,)),
                        fixed_seg35=labels, moving_seg35=labels)
    manifest = dict(fixed_subject=fixed_subject, moving_subject=moving_subject, shape_zyx=list(grid.shape),
                    spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit="mm",
                    sources=dict(fixed=dict(sha256=source_sha(fixed_subject)),
                                 moving=dict(sha256=source_sha(moving_subject))),
                    pair_sha256=file_sha256(directory/"pair.npz"))
    write_json(directory/"manifest.json", manifest)
    return grid, f"{STAGE}/{name}"


def write_tissue(root, grid, subject, shift=(0., 0., 0.), **changes):
    """Atropos-like output: three posteriors (CSF, GM, WM) = (.2, .3, .5) inside an ellipsoid (centre shifted
    by `shift` mm, so the moving image's channels differ from the fixed one's), 0 outside."""
    directory = root/"seg"/subject
    directory.mkdir(parents=True)
    X = grid.identity()
    c = (N-1)/2
    brain = (np.linalg.norm((X-c-np.asarray(shift))/c, axis=-1) < .8).astype(float)
    files = [f"posteriors{k+1:02d}.nii.gz" for k in range(3)]
    for name, share in zip(files, (.2, .3, .5)):
        save_scalar(directory/name, share*brain, grid)
    prov = dict(layer="tissue", segmenter_id="atropos", model_ids=["ANTs-2.6.5-synthetic-Atropos"],
                trained_with_labels=False, test_labels_seen=False, oracle=False, class_names=["CSF", "GM", "WM"],
                wm_index=2, posterior_files=files, posterior_file_index_base=1,
                input_sha256={"aligned_norm.nii.gz": source_sha(subject)},
                output_sha256={name: file_sha256(directory/name) for name in files})
    prov.update(changes)
    write_json(directory/"provenance.json", prov)


@pytest.fixture
def setup(tmp_path):
    allowlist = write_json(tmp_path/"dev_subjects.json", allowlist_document())

    def args(pair, out, *extra):
        return ["--pair", pair, "--channel-source", "tissue", "--out", str(out), "--data-root", str(tmp_path/"data"),
                "--allowlist", str(allowlist), "--tissue-root", str(tmp_path/"seg"), *extra]
    return tmp_path, args


def read(path):
    return json.loads(path.read_text())


def test_sealed_subject_is_refused_before_the_npz(setup, monkeypatch):
    root, args = setup
    _, pair = write_synthetic_pair(root, SEALED, DEV[1])
    opened = []
    monkeypatch.setattr(pair_module, "_open_npz", lambda path: opened.append(path))
    out = root/"run_sealed"
    assert run_pair.main(args(pair, out)) == 3
    status = read(out/"status.json")
    assert status["status"] == "subject_sealed" and status["exit_code"] == 3
    assert not opened
    for name in ("cmd.txt", "git.txt", "env.txt", "log.txt", "timing.json"):
        assert (out/name).is_file()
    for name in ("final_native.npz", "disp.npz", "cert.json", "data.txt"):
        assert not (out/name).exists()
    with pytest.raises(FileExistsError):
        run_pair.main(args(pair, out))                    # never reuse a run directory


def test_provenance_rejected(setup):
    root, args = setup
    grid, pair = write_synthetic_pair(root, DEV[0], DEV[1])
    write_tissue(root, grid, DEV[0])
    write_tissue(root, grid, DEV[1], test_labels_seen=True)
    out = root/"run_rejected"
    assert run_pair.main(args(pair, out)) == 4
    status = read(out/"status.json")
    assert status["status"] == "provenance_rejected" and "test_labels_seen" in status["reason"]
    assert not (out/"disp.npz").exists()


def test_end_to_end_tissue_channels(setup):
    root, args = setup
    grid, pair = write_synthetic_pair(root, DEV[0], DEV[1])
    write_tissue(root, grid, DEV[0])
    write_tissue(root, grid, DEV[1], shift=(.7, -.5, .4))
    out = root/"run_ok"
    assert run_pair.main(args(pair, out, "--rounds", "1,1,1,1", "--max-accepted", "2")) == 0
    status = read(out/"status.json")
    assert status["status"] == "cap_reached" and status["wording"] == "上限收尾" and status["native"]
    for name in ("cmd.txt", "git.txt", "env.txt", "data.txt", "config.json", "log.txt", "log.jsonl", "hyper.json",
                 "provenance.json", "timing.json", "final_native.npz", "disp.npz", "cert.json"):
        assert (out/name).is_file(), name
    cert = read(out/"cert.json")
    assert cert["status"] == "certified_feasible" and cert["matches_solver_certificate"]
    assert cert["config"]["delta_hex"] == "0x0.0p+0" and cert["config"]["max_depth"] == 48
    with np.load(out/"final_native.npz") as z:
        assert str(z["state_hash"]) == cert["state_hash"] == status["exported"]
    with np.load(out/"disp.npz") as z:
        assert z["d"].shape == (3, N, N, N)
    hyper = read(out/"hyper.json")
    assert [r["level"] for r in hyper["rounds"]] == [0, 1, 2, 3]
    assert all(r["status"] == "cap_reached" and r["accepted"] == 2 for r in hyper["rounds"])
    assert hyper["sigma_post"] is not None and len(hyper["sigma_used"]) == 4         # intensity + 3 tissue channels
    assert all(s > 1e-8 for s in hyper["sigma_used"])
    prov = read(out/"provenance.json")
    assert prov["channel_source"] == "tissue" and prov["channels"]["K_T"] == 3 and prov["omega0"] == 1.
    assert prov["channels"]["tissue"]["fixed"]["accepted"]["segmenter_id"] == "atropos"
    assert prov["pair"]["keys_read"] == ["fixed", "moving", "initial_u"]
    assert prov["backend"]["certificate"]["certificate_version"] == cert["config"]["certificate_version"]
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in rows} >= {"candidate", "accept", "round", "transfer"}
    assert all(r["energy"] is None for r in rows if r["kind"] == "candidate" and r["cert_status"] != "certified_feasible")
    qc = read(out/"qc.json")
    assert qc["sigma_collapse_suspected"] in (True, False) and qc["sigma_trajectory"]
    assert hyper["sigma_monitor"]["criterion"].startswith("an adjacent-round M step")
    assert status["export"]["certificate_passed"] is True
    resources = status["resources"]                          # FB1: thread count and peak RSS of the run
    assert resources["cpu_threads"] == 4 and resources["peak_rss_mb"] > 0 and resources["user_seconds"] > 0
    assert read(out/"timing.json")["resources"]["cpu_threads"] == 4 and prov["resources"]["cpu_threads"] == 4
    assert read(out/"config.json")["solver"]["cpu_threads"] == 4
    assert hyper["backend_after_solve"]["backend"] == "numpy"
    assert status["device_anomalies"] is None and prov["switches"]["opt1_in_effect"] is False     # OPT1b


def test_end_to_end_torch64_backend(setup):
    """--backend torch64 --device cpu: the same run with the device backend; hyper.json carries the C08-8(h)
    comparison statistics (count, fallback ratio, per-context and per-GF_B/T_B-bin counts) and the CPU GF_B of the
    final native state; provenance.json the reduction structure and depth; cert.json a fresh CPU PASS."""
    root, args = setup
    grid, pair = write_synthetic_pair(root, DEV[0], DEV[1])
    write_tissue(root, grid, DEV[0])
    write_tissue(root, grid, DEV[1], shift=(.7, -.5, .4))
    out = root/"run_torch64"
    assert run_pair.main(args(pair, out, "--rounds", "1,1,1,1", "--max-accepted", "2", "--backend", "torch64",
                              "--device", "cpu", "--cpu-threads", "2")) == 0
    status = read(out/"status.json")
    assert status["resources"]["cpu_threads"] == 2 and status["resources"]["peak_rss_mb"] > 0
    assert status["status"] == "cap_reached" and status["export"]["certificate_passed"]
    hyper = read(out/"hyper.json")
    comparison = hyper["gpu_comparison"]
    assert comparison["comparisons"] >= 8 and comparison["contract"] == "C08-8(h)"
    assert sum(v["comparisons"] for v in comparison["by_gf_over_tb"].values()) == comparison["comparisons"]
    assert set(comparison["by_context"]) == {"pg"}
    assert hyper["GF_B_cpu_final"] is not None and hyper["sigma_monitor"]["trace"]
    prov = read(out/"provenance.json")
    assert prov["backend"]["backend"] == "torch64" and prov["backend"]["comparison"]["depth_bound"] == 8225
    assert prov["resources"]["cpu_threads"] == 2 and hyper["backend_after_solve"]["backend"] == "torch64"
    assert prov["env"]["device"] == "cpu" and prov["env"]["gpu_used"] is False
    assert read(out/"cert.json")["status"] == "certified_feasible" and (out/"disp.npz").is_file()
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    evaluated = [r for r in rows if r["kind"] == "candidate" and r["energy"] is not None]
    assert evaluated and all(r["comparison"]["context"] == "pg" for r in evaluated)
    # OPT1b (review OPT1_复核_20261030 items 2 and 4): the three switches and their effective values are recorded,
    # the device states' provenance after the solve, and the device anomaly counts in status.json
    config = read(out/"config.json")
    assert {k: config["args"][k] for k in run_pair.OPT1_DEFAULTS} == run_pair.OPT1_DEFAULTS
    assert config["opt1"]["opt1_in_effect"] is True and config["opt1"]["opt1_note"] == run_pair.OPT1_NOTE
    assert {k: prov["switches"][k] for k in run_pair.OPT1_DEFAULTS} == run_pair.OPT1_DEFAULTS
    effective = dict(jacobian_cache_slots=2, jacobian_cache_log=False, speculative_gradient=True)
    for record in (config["backend"], prov["backend"], hyper["backend_after_solve"]):
        assert {k: record["state_options_effective"][k] for k in effective} == effective
    states = hyper["backend_after_solve"]["device_states"]
    assert states and all({k: s[k] for k in effective} == effective for s in states)
    assert config["backend"]["device_states"] == []                    # written before the solve
    assert status["device_anomalies"] == dict(evaluate=0, compare=0)


def test_opt1_switches(capsys):
    """OPT1b (review OPT1_复核_20261030 item 2): --jacobian-cache, --jacobian-cache-log, --speculative-gradient
    (defaults on, off, on) -> TorchModelState options; a non-default value with the numpy backend and log J without
    the cache are refused; the help says the switches are bitwise neutral."""
    base = ["--pair", "s/p", "--channel-source", "none", "--out", "x"]
    for extra in ([], ["--backend", "torch64"]):
        args = run_pair.parse_args(base+extra)
        assert (args.jacobian_cache, args.jacobian_cache_log, args.speculative_gradient) == ("on", "off", "on")
    args = run_pair.parse_args(base+["--backend", "torch64"])
    assert run_pair.device_state_options(args) == dict(jacobian_cache_log=False, speculative_gradient=True)
    args = run_pair.parse_args(base+["--backend", "torch64", "--device", "cpu", "--jacobian-cache", "off",
                                     "--speculative-gradient", "off"])
    options = run_pair.device_state_options(args)
    assert options == dict(jacobian_cache_log=False, speculative_gradient=False, jacobian_cache_slots=0)
    backend = run_pair.backend_of(args)
    assert backend.state_options == options and backend.record()["state_options_effective"]["jacobian_cache_slots"] == 0
    args = run_pair.parse_args(base+["--backend", "torch64", "--device", "cpu", "--jacobian-cache-log", "on"])
    assert run_pair.backend_of(args).state_options_effective["jacobian_cache_log"] is True
    assert run_pair.opt1_switches(args)["jacobian_cache_log"] == "on"
    assert run_pair.backend_of(run_pair.parse_args(base)).name == "numpy"
    for bad in (["--jacobian-cache", "off"], ["--speculative-gradient", "off"], ["--jacobian-cache-log", "on"],
                ["--backend", "torch64", "--jacobian-cache", "off", "--jacobian-cache-log", "on"],
                ["--backend", "torch64", "--jacobian-cache", "maybe"]):
        with pytest.raises(SystemExit):
            run_pair.parse_args(base+bad)
    capsys.readouterr()
    with pytest.raises(SystemExit):
        run_pair.parse_args(["--help"])
    assert run_pair.OPT1_NOTE in capsys.readouterr().out
