"""The C05-7 eta sweep tools (experiments/eta; work packages ETA, ETA2), on synthetic files only (tmp_path; no data/, no
unit, no real segmentation):

- probe.py: u_1 is bit for bit the first accepted state of an uncapped level-0 solve (state_hash of its first "accept"
  row), nu* / alpha* are the contract's ratios at u_1 with the round's state, the run directory records, no disp.npz;
  bayesreg.cli.run_pair is not touched by the tool (the default path stays as it was). ETA2: the same with the final
  configuration (structure layer of experiments/eta/dry_inputs.py, certificate cell estimate, gauss27+closed27 with
  beta_S 3): 1 + K_S model channels, the configuration recorded; the probe's options mirror run_pair's (defaults --
  since DEF, 2026-10-31, the point set too: gauss27+closed27, and the cell estimate certificate --, value checks, the
  structure group; no inference); the first test gives the first plan's configuration (gauss27+corners,
  gauss27+corners) explicitly, as the probe's defaults were before DEF;
- experiments/d0/atropos_seg.py: the defaults are the D0 behaviour (stage_L2R_val_v1, 20 subjects, the provenance stage
  string); --stage stage_T0_tune_v1 --data-root reads that stage and names it;
- sweep.py, pure parts: the pair rule on the T0 strata, the candidates' eta (alpha_nug, tau_0 fixed), nu* combination, the
  selection rule with its tie order and eligibility; the option checks (beta_S only with closed27, cell estimate with
  the metric off, --extra repeating a managed option, --segmenter-registry outside a dry run), the probe configuration
  comparison;
- sweep.py end to end with a fake machine (units emulated; probe and run_pair arguments parsed by their real parsers):
  atropos skip, three probes, s1 -> s2 -> s3 with the reused centre runs not rerun (21 runs, 21 evaluations), the
  expected path, summary and eta_selected written once, the configuration in every record; a restart launches nothing;
  a restart under another configuration STOPs; a run without status.json STOPs and is never relaunched; a busy gate
  STOPs before any unit; dry-run input checks;
- the structure precheck: complete predictions pass (--until precheck launches nothing); a missing prediction writes the
  subject file and the exact inference command and STOPs; a wrong policy or an incomplete directory STOPs; real mode: the
  code tree's registry must be committed and clean;
- slow (-m slow): the dry run of the final configuration end to end in process (real probe, run_pair and bayesreg.eval
  on the synthetic structure layer).
- ETA4 (design v4, runs/20261101-eta-sweep-v4/plan.md): the v3 command line gives the v3 run_pair arguments, systemd
  command and configuration keys (and a v3 root none of the v4 records); explicit per-level caps; the v4 option checks;
  pilot_path on constructed pilot summaries (P3 / P1 / P2 resolved, the current path for the lead, missing, 待审核 /
  待核对 / check-(b) violations, a recommendation not reproduced by the pilot's recommend, another eta, other arm
  switches, schema, root, location, not the newest summary); v4 end to end on the fake machine (no probe, the centre
  record, 24 runs with the path's switches parsed by run_pair's parser, reuse, edges, "仍在下端", readouts, restart, a
  restart under another path STOPs); a held path starts nothing until --path is given; a check-(b) violation STOPs
  until the audit file; the SDlogJ eligibility condition; the centre checks; the driver-process gate, --extra with caps,
  the run_pair preflight, the deadline; driver_processes on a fake /proc.
"""
import contextlib
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
STAGE = "stage_T0_tune_v1"
PAIR1 = "OASIS_OAS1_0901_MR1__OASIS_OAS1_0902_MR1"
SUBJECTS = [f"OASIS_OAS1_{n:04d}_MR1" for n in range(901, 907)]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep():
    return load("eta_sweep", "experiments/eta/sweep.py")


@pytest.fixture(scope="module")
def probe_module():
    return load("eta_probe", "experiments/eta/probe.py")


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    """Synthetic inputs of experiments/eta/dry_inputs.py (four 26^3 pairs, strata, allowlist, tissue, structure layer,
    staging registry)."""
    out = tmp_path_factory.mktemp("eta")/"dry"
    load("eta_dry_inputs", "experiments/eta/dry_inputs.py").main(["--out", str(out)])
    return out


def dry_common(dry):
    return ["--data-root", str(dry/"data"), "--allowlist", str(dry/"dev_subjects.json"), "--tissue-root",
            str(dry/"seg"), "--cpu-threads", "1"]


def final_model_args(dry, share="3"):
    """The final configuration's model options as the sweep passes them in a dry run (beta_S 3: not the default)."""
    return ["--channel-source", "structure", "--logvol-points", "gauss27+closed27", "--logvol-closed27-share", share,
            "--closed27-weights", "simpson", "--metric-jd-scaling", "affine_v1", "--jd-cell-estimate", "certificate",
            "--structure-root", str(dry/"structure"), "--structure-grouping", "structure13_v1", "--segmenter-id",
            "unet35_v1_ens", "--segmenter-registry", str(dry/"registry"/"segmenters.json"),
            "--structure-registry-staging"]


class _Quiet:
    def info(self, *args):
        pass


# ---- probe ---------------------------------------------------------------------------------------------------------

def test_probe_u1_is_the_first_step_and_nu_star_is_the_ratio(dry, tmp_path, probe_module):
    probe = probe_module
    from bayesreg.cli import run_pair as rp
    from bayesreg.io import load_pair
    from bayesreg.model import evaluate, gradient
    from bayesreg.solve import (Caps, GaussianMStep, PGConfig, SolverConfig, eta_temporary, make_backend,
                                solve_multilevel)
    from bayesreg.solve.jd_scaling import JDScalingConfig
    out = tmp_path/"probe"
    first_plan = ["--logvol-points", "gauss27+corners", "--jd-cell-estimate", "gauss27+corners"]   # DEF: explicit
    assert probe.main(["--pair", f"{STAGE}/{PAIR1}", "--channel-source", "none", "--out", str(out), *first_plan,
                       *dry_common(dry)]) == 0
    record = json.loads((out/"probe.json").read_text())
    status = json.loads((out/"status.json").read_text())
    assert status["status"] == "probe_done" and status["exit_code"] == 0
    assert record["solver_status"] == "stopped_before_native" and record["native"] is False
    assert record["level0_round"]["accepted"] == 1 and record["logvol_points"] == "gauss27+corners"
    assert record["configuration"] == dict(channel_source="none", logvol_points="gauss27+corners",
                                           logvol_closed27_share=None, closed27_weights=None,
                                           metric_jd_scaling="affine_v1", jd_cell_estimate="gauss27+corners",
                                           structure=None)
    assert record["model_channels"] == 1 and status["structure_registry"] is None
    for name in ("cmd.txt", "git.txt", "env.txt", "data.txt", "config.json", "provenance.json", "log.txt", "log.jsonl",
                 "final_level0.npz", "timing.json"):
        assert (out/name).is_file(), name
    assert not (out/"disp.npz").exists() and not (out/"final_native.npz").exists()
    # the same inputs through an uncapped (3 steps) level-0 solve: its first accepted state is u_1, bit for bit
    pair = load_pair(rp.resolve_pair(dry/"data", f"{STAGE}/{PAIR1}"), allowlist=str(dry/"dev_subjects.json"))
    inputs, pt_f, mask_f, _ = rp.prepare_channels(pair, "none", str(dry/"seg"), _Quiet())
    eta = eta_temporary(pt_f, mask_f, pair.grid)
    assert record["eta_tmp"]["nu"] == eta.nu and record["eta_tmp"]["tau"] == [1., .5, .5, .5]
    rows = []
    config = SolverConfig(outer="a", rounds=(1, 1, 1, 1), caps=Caps(3, 60),
                          pg=PGConfig(jd=JDScalingConfig(cell_estimate="gauss27+corners")), max_levels=1, cpu_threads=1,
                          logvol_points="gauss27+corners")
    result = solve_multilevel(make_backend("numpy"), inputs, eta=eta, sigma=GaussianMStep(), config=config,
                              log=rows.append)
    accepts = [r for r in rows if r.get("kind") == "accept"]
    assert len(accepts) >= 2
    assert accepts[0]["state_hash"] == record["u1_state_hash"]
    assert result.rounds[0]["F_B_trace"][1] == record["level0_round"]["F_B_end"]
    u1 = np.load(out/"final_level0.npz")["u"]
    state = result.state                                   # the round's state: the same sigma used in round 0
    assert np.asarray(state.sigma2).tolist() == record["sigma2_used"]
    point = evaluate(state, u1)
    data = np.linalg.norm(gradient(state, point, blocks=("channel",)))
    unit = state.replace(nu=1.0)
    curl = np.linalg.norm(gradient(unit, evaluate(unit, u1), blocks=("curl",)))
    assert record["nu_star"] == pytest.approx(data/curl, rel=1e-12)
    assert record["checks"]["curl_linearity_relative"] < 1e-12


def test_probe_final_configuration_structure_certificate_closed27(dry, tmp_path, probe_module):
    """ETA2: the probe under the final configuration (structure layer, certificate cell estimate, closed27 with
    beta_S 3): u_1 is the first accepted state of the uncapped solve of the same configuration, the data-term gradient
    has the 1 + K_S structure channels, nu* is the ratio, the configuration and the registry state are recorded."""
    probe = probe_module
    from bayesreg.cli import run_pair as rp
    from bayesreg.io import load_pair
    from bayesreg.model import evaluate, gradient
    from bayesreg.seg.structure_layer import StructureSource
    from bayesreg.solve import (Caps, GaussianMStep, PGConfig, SolverConfig, eta_temporary, make_backend,
                                solve_multilevel)
    from bayesreg.solve.jd_scaling import JDScalingConfig
    out = tmp_path/"probe"
    assert probe.main(["--pair", f"{STAGE}/{PAIR1}", "--out", str(out), *final_model_args(dry), *dry_common(dry)]) == 0
    record = json.loads((out/"probe.json").read_text())
    status = json.loads((out/"status.json").read_text())
    assert status["status"] == "probe_done" and record["level0_round"]["accepted"] == 1
    assert record["model_channels"] == 14 and record["K_S"] == 13 and status["model_channels"] == 14
    conf = record["configuration"]
    assert conf["channel_source"] == "structure" and conf["logvol_points"] == "gauss27+closed27"
    assert conf["logvol_closed27_share"] == 3. and conf["closed27_weights"] == "simpson"
    assert conf["jd_cell_estimate"] == "certificate" and record["jd_cell_estimate"] == "certificate"
    assert conf["structure"]["registry_status"] == "staging_unregistered" and conf["structure"]["infer"] == "never"
    assert conf["structure"]["root"] == str(dry/"structure") and status["configuration"] == conf
    assert status["structure_registry"]["status"] == "staging_unregistered"
    prov = json.loads((out/"provenance.json").read_text())
    assert prov["switches"]["jd_cell_estimate"] == "certificate" and prov["switches"]["logvol_closed27_share"] == 3.
    assert prov["channels"]["structure"] is not None and prov["channels"]["K_S"] == 13
    solver = json.loads((out/"config.json").read_text())["solver"]
    assert json.dumps(solver).count("certificate") >= 1 and json.dumps(solver).count("gauss27+closed27") >= 1
    # the uncapped level-0 solve of the same configuration
    pair = load_pair(rp.resolve_pair(dry/"data", f"{STAGE}/{PAIR1}"), allowlist=str(dry/"dev_subjects.json"))
    source = StructureSource(root=dry/"structure", registry_path=dry/"registry"/"segmenters.json",
                             allowlist_path=dry/"dev_subjects.json", models_root=tmp_path/"none",
                             image_root=tmp_path/"none", allow_staging=True)
    inputs, pt_f, mask_f, _ = rp.prepare_channels(pair, "structure", str(dry/"seg"), _Quiet(), structure=source)
    assert inputs[0].fixed.shape[0] == 14
    eta = eta_temporary(pt_f, mask_f, pair.grid)
    rows = []
    config = SolverConfig(outer="a", rounds=(1, 1, 1, 1), caps=Caps(3, 60),
                          pg=PGConfig(metric_jd_scaling="affine_v1", jd=JDScalingConfig(cell_estimate="certificate")),
                          max_levels=1, cpu_threads=1, logvol_points="gauss27+closed27", logvol_closed27_share=3.,
                          closed27_weights="simpson")
    result = solve_multilevel(make_backend("numpy"), inputs, eta=eta, sigma=GaussianMStep(), config=config,
                              log=rows.append)
    accepts = [r for r in rows if r.get("kind") == "accept"]
    assert len(accepts) >= 2 and accepts[0]["state_hash"] == record["u1_state_hash"]
    u1 = np.load(out/"final_level0.npz")["u"]
    state = result.state
    point = evaluate(state, u1)
    data = np.linalg.norm(gradient(state, point, blocks=("channel",)))
    unit = state.replace(nu=1.0)
    curl = np.linalg.norm(gradient(unit, evaluate(unit, u1), blocks=("curl",)))
    assert record["nu_star"] == pytest.approx(data/curl, rel=1e-12)


def test_probe_options_mirror_run_pair(probe_module):
    probe = probe_module
    from bayesreg.cli import run_pair as rp
    shared = ["--pair", "a/b", "--channel-source", "structure", "--out", "x", "--logvol-points", "gauss27+closed27",
              "--logvol-closed27-share", "2.5", "--jd-cell-estimate", "certificate", "--structure-root", "/s",
              "--structure-grouping", "structure13_v1", "--segmenter-id", "unet35_v1_ens"]
    p, r = vars(probe.parse_args(shared)), vars(rp.parse_args(shared))
    for key in p:
        if key in r:
            assert p[key] == r[key], key
    assert p["structure_infer"] == "never" and p["segmenter_registry"] == str(rp.REPO_REGISTRY)
    # the defaults of the shared options are run_pair's (DEF, 2026-10-31: the point set too; before DEF the probe's
    # point set defaulted to gauss27+corners, the first plan's)
    p, r = vars(probe.parse_args(["--pair", "a/b", "--channel-source", "none", "--out", "x"])), \
        vars(rp.parse_args(["--pair", "a/b", "--channel-source", "none", "--out", "x"]))
    for key in ("logvol_points", "logvol_closed27_share", "closed27_weights", "metric_jd_scaling", "jd_cell_estimate",
                "data_root", "allowlist", "tissue_root", "cpu_threads", "structure_root", "segmenter_registry",
                "structure_groups", "structure_grouping", "segmenter_id", "segmenter_models_root",
                "structure_image_root", "structure_registry_staging", "structure_infer"):
        assert p[key] == r[key], key
    assert (p["logvol_points"], p["logvol_closed27_share"], p["closed27_weights"], p["jd_cell_estimate"]) == \
        ("gauss27+closed27", 1., "simpson", "certificate")
    # --metric-jd-scaling off: the cell estimate unset is "none" for both, any explicit value refused by both (DEF)
    off = ["--pair", "a/b", "--channel-source", "none", "--out", "x", "--metric-jd-scaling", "off"]
    assert probe.parse_args(off).jd_cell_estimate == rp.parse_args(off).jd_cell_estimate == "none"
    for estimate in ("gauss27+corners", "closed27", "certificate"):
        for parse in (probe.parse_args, rp.parse_args):
            with pytest.raises(SystemExit):
                parse(off+["--jd-cell-estimate", estimate])
    staged = probe.parse_args(["--pair", "a/b", "--channel-source", "structure", "--out", "x",
                               "--structure-registry-staging"])
    assert staged.segmenter_registry == str(rp.DEFAULT_SEGMENTER_REGISTRY)
    for bad in (["--jd-cell-estimate", "certificate", "--metric-jd-scaling", "off"],
                ["--logvol-points", "gauss27", "--logvol-closed27-share", "3"],
                ["--logvol-points", "gauss27+corners", "--closed27-weights", "uniform"],
                ["--logvol-points", "gauss27+closed27", "--logvol-closed27-share", "0"],
                ["--structure-infer", "missing"]):
        with pytest.raises(SystemExit):
            probe.parse_args(["--pair", "a/b", "--channel-source", "structure", "--out", "x", *bad])
    text = (ROOT/"experiments"/"eta"/"probe.py").read_text()
    assert "max_levels=1" in text and "Caps(max_accepted=1" in text


# ---- Atropos script --------------------------------------------------------------------------------------------------

def test_atropos_defaults_are_d0_and_stage_option(tmp_path, monkeypatch):
    atropos = load("atropos_seg", "experiments/d0/atropos_seg.py")
    assert atropos.DEFAULT_STAGE == "stage_L2R_val_v1"
    assert atropos.STAGE == atropos.ROOT/"data/three_d/stage_L2R_val_v1"
    calls = []

    def images(allowlist, stage):
        calls.append(("images", Path(allowlist), Path(stage)))
        count = 24 if Path(stage).name == STAGE else 20
        return {f"OASIS_OAS1_{k:04d}_MR1": [] for k in range(count)}
    monkeypatch.setattr(atropos, "subject_images", images)
    monkeypatch.setattr(atropos, "segment", lambda *a: calls.append(("segment",)+a[2:]))
    atropos.main(["--subjects", "OASIS_OAS1_0001_MR1", "--out-root", str(tmp_path/"a")])
    assert calls[0] == ("images", atropos.ALLOWLIST, atropos.ROOT/"data/three_d/stage_L2R_val_v1")
    assert calls[1] == ("segment", tmp_path/"a", 8, "stage_L2R_val_v1")
    calls.clear()
    atropos.main(["--stage", STAGE, "--data-root", str(tmp_path/"d"), "--allowlist", str(tmp_path/"al.json"),
                  "--subjects", "OASIS_OAS1_0002_MR1", "--out-root", str(tmp_path/"b")])
    assert calls[0] == ("images", tmp_path/"al.json", tmp_path/"d"/STAGE)
    assert calls[1] == ("segment", tmp_path/"b", 8, STAGE)
    source = (ROOT/"experiments"/"d0"/"atropos_seg.py").read_text()
    assert "image_source=dict(stage=stage_name," in source


# ---- pure parts of the driver ---------------------------------------------------------------------------------------

T0_STRATA = [("0012", "0013", "moving larger"), ("0014", "0015", "moving larger"), ("0016", "0017", "moving smaller"),
             ("0018", "0019", "moving larger"), ("0020", "0021", "moving larger"), ("0023", "0025", "moving smaller"),
             ("0026", "0027", "balanced"), ("0028", "0029", "moving smaller"), ("0031", "0032", "balanced"),
             ("0033", "0034", "moving smaller"), ("0037", "0038", "balanced"), ("0041", "0042", "balanced")]


def test_pair_rule_on_the_t0_strata(sweep):
    rows = [dict(pair=f"OASIS_OAS1_{a}_MR1__OASIS_OAS1_{b}_MR1", stratum=s) for a, b, s in reversed(T0_STRATA)]
    pairs = sweep.select_pairs(dict(pairs=rows))
    assert [sweep.short_name(p) for p in pairs] == ["0012-0013", "0016-0017", "0026-0027"]
    with pytest.raises(sweep.Stop):
        sweep.select_pairs(dict(pairs=rows[:4]))


def test_candidate_eta_and_constants(sweep):
    from bayesreg.eval.fields import COMPARABLE_ROLES
    from bayesreg.solve import multilevel
    assert sweep.ALPHA_NUG_FACTOR == multilevel.ALPHA_NUG_FACTOR
    assert tuple(sweep.COMPARABLE_ROLES) == tuple(COMPARABLE_ROLES)
    assert set(sweep.CHANNEL_SOURCES) == set(multilevel.CHANNEL_SOURCES)
    tau0 = [1., .5, .5, .5]
    e = sweep.candidate_eta("s1", "x0p1", .1, nu_center=3., tau0=tau0)
    assert e["nu"] == 3./10 and e["tau"] == tau0 and e["alpha"] == .01 and e["s_tau"] == 1.
    assert sweep.candidate_eta("s1", "x10", 10., nu_center=3., tau0=tau0)["nu"] == 30.
    assert sweep.candidate_eta("s2", "x0p25", .25, nu_center=3., tau0=tau0)["nu"] == 3./4
    e = sweep.candidate_eta("s3", "st2", 2., nu_center=3., tau0=tau0)
    assert e["nu"] == 3. and e["tau"] == [1., 1., 1., 1.] and e["alpha"] == .01
    e = sweep.candidate_eta("s3", "st0p5", .5, nu_center=3., tau0=tau0)
    assert e["tau"] == [1., .25, .25, .25] and e["alpha"] == .01          # tau_0 = 1 stays the largest: alpha_nug = .01
    assert sweep.alpha_nug([1., 1.5]) == pytest.approx(.01/2.25)
    assert sweep.geometric_mean([2., 8., 4.]) == pytest.approx(4.)
    assert [label for label, _ in sweep.LABELS["s2"]] == ["x0p25", "x1", "x4"]


def cand(label, factor, seg35, seg4=.8, eligible=True):
    return dict(label=label, factor=factor, mean_seg35=seg35, mean_seg4=seg4, eligible=eligible)


def test_selection_rule(sweep):
    chosen, tied, _ = sweep.choose([cand("x0p1", .1, .70), cand("x1", 1., .72), cand("x10", 10., .75)])
    assert chosen["label"] == "x10" and tied == ["x10"]
    # within 1e-3 of the best: the centre wins
    chosen, tied, reason = sweep.choose([cand("x0p1", .1, .7495), cand("x1", 1., .7491), cand("x10", 10., .75)])
    assert chosen["label"] == "x1" and set(tied) == {"x0p1", "x1", "x10"} and "tie" in reason
    # equidistant from the centre: the higher seg4, then the smaller factor
    chosen, _, _ = sweep.choose([cand("x0p1", .1, .75, .80), cand("x1", 1., .70), cand("x10", 10., .75, .81)])
    assert chosen["label"] == "x10"
    chosen, _, _ = sweep.choose([cand("x0p1", .1, .75, .80), cand("x1", 1., .70), cand("x10", 10., .75, .80)])
    assert chosen["label"] == "x0p1"
    # an ineligible candidate is never chosen; none eligible -> None
    chosen, _, _ = sweep.choose([cand("x0p1", .1, .90, eligible=False), cand("x1", 1., .70)])
    assert chosen["label"] == "x1"
    assert sweep.choose([cand("x1", 1., .9, eligible=False)])[0] is None


def test_eligibility(sweep):
    good = dict(pair="a", field_role="native_formal", certificate=dict(passed=True), metrics=dict(seg35=.8))
    assert sweep.eligibility([good, dict(good, pair="b", field_role="disp_only")]) == (True, [])
    ok, reasons = sweep.eligibility([good, dict(good, pair="b", field_role="handover_only"),
                                     dict(good, pair="c", certificate=dict(passed=False)),
                                     dict(good, pair="d", metrics=dict(seg35=None)), dict(good, pair="e", eval_error="x")])
    assert not ok and len(reasons) == 4


def test_option_checks_and_defaults(sweep):
    args = sweep.parse_args([])
    assert (args.channel_source, args.logvol_points, args.logvol_closed27_share, args.jd_cell_estimate) == \
        ("structure", "gauss27+closed27", 1., "certificate")
    assert (args.wall_seconds, args.backend, args.device, args.rounds, args.outer) == (900., "torch64", "cuda", "1,1,1,1",
                                                                                      "a")
    assert args.structure_root == "/home/rmcfz/seq_vp_tag_results/seg/unet35_v1/pred"
    assert (args.memory_max, args.probe_memory_max, args.min_gpu_free_mib) == ("9G", "9G", 22528)
    assert args.root == "/home/rmcfz/seq_vp_tag_results/eta_sweep_v2"
    assert sweep.parse_args(["--wall-seconds", "1800"]).wall_seconds == 1800.
    for bad in (["--logvol-points", "gauss27+corners", "--logvol-closed27-share", "3"],
                ["--logvol-closed27-share", "0"], ["--logvol-closed27-share", "nan"],
                ["--metric-jd-scaling", "off", "--jd-cell-estimate", "gauss27+corners"],   # DEF: any explicit value
                ["--metric-jd-scaling", "off", "--jd-cell-estimate", "certificate"],
                ["--extra", "--logvol-closed27-share 3"], ["--extra", "--structure-infer=missing"],
                ["--extra", "--jd-cell est"], ["--extra", "--channel structure"], ["--wall-seconds", "0"],
                ["--memory-max", "7.5G"]):
        with pytest.raises(SystemExit):
            sweep.parse_args(bad)
    assert sweep.parse_args(["--extra", "--max-accepted 3"]).extra == "--max-accepted 3"
    off = sweep.parse_args(["--metric-jd-scaling", "off"])               # DEF: the unset cell estimate is "none"
    assert (off.metric_jd_scaling, off.jd_cell_estimate) == ("off", "none")
    run = sweep.Sweep(off, FakeSystem())
    assert "--jd-cell-estimate" not in run.model_args() and run.expected_run_args()["jd_cell_estimate"] == "none"
    assert run.probe_configuration()["jd_cell_estimate"] == "none"
    assert [sweep.number(v) for v in (900., 1., .5, 3., 1234567.)] == ["900", "1", "0.5", "3", "1234567.0"]


def test_probe_configuration_comparison(sweep):
    want = dict(channel_source="structure", logvol_points="gauss27+closed27", logvol_closed27_share=1.,
                closed27_weights="simpson", metric_jd_scaling="affine_v1", jd_cell_estimate="certificate",
                structure=dict(root="/s", grouping="structure13_v1", segmenter_id="unet35_v1_ens",
                               registry="/code/configs/bayesreg/../bayesreg/segmenters.json",
                               registry_status="registered", staging_opt_in=False, infer="never"))
    got = json.loads(json.dumps(want))
    got["structure"].update(registry="/code/configs/bayesreg/segmenters.json", groups="/code/x.json")
    assert sweep.same_probe_configuration(got, want)
    for key, value in (("logvol_closed27_share", 3.), ("jd_cell_estimate", "gauss27+corners"),
                       ("channel_source", "tissue")):
        assert not sweep.same_probe_configuration(dict(got, **{key: value}), want)
    assert not sweep.same_probe_configuration(dict(got, structure=dict(got["structure"], registry_status="staging_"
                                                                                                         "unregistered")), want)
    assert not sweep.same_probe_configuration(dict(got, structure=None), want)
    assert not sweep.same_probe_configuration(None, want)
    assert sweep.same_probe_configuration(dict(want, structure=None), dict(want, structure=None))


# ---- the driver end to end on a fake machine ------------------------------------------------------------------------

NU_PROBE = {"0901-0902": 2., "0903-0904": 8., "0905-0906": 4.}     # geometric mean 4
OFFSET = {"0901-0902": 0., "0903-0904": -.01, "0905-0906": .01}


def seg35_of(nu, tau):
    """A made-up response: best near nu = 16 and s_tau = 2 (tau_k = 1)."""
    return .8-.02*math.log10(nu/16)**2-.01*math.log2(tau[1]/1.)**2


class FakeSystem:
    """Emulates the units: run_pair, bayesreg.eval.cli, probe.py, atropos_seg.py write what they would. The probe's and
    run_pair's arguments go through their real parsers (the wiring of the configuration is the real one)."""

    def __init__(self, *, busy=(), drop=(), roles=None, violations=None, sdlogj=None, drivers=(), clock=None,
                 tick=None, cert_fail=(), pilot=((), ()), eta_shift=None, no_export=(), sdlogj_nan=()):
        self.self_unit, self.units, self.busy, self.drop, self.roles = "", [], list(busy), set(drop), roles or {}
        # ETA4: certificate_bernstein check-(b) totals per run unit; official SDlogJ as a function of (nu, tau); driver
        # processes; the clock (tick: the clock after each launched run unit); run ids whose evaluator certificate does
        # not pass (hard constraint 1); the path pilot's (units, drivers); run units whose config.json eta is shifted
        self.violations, self.sdlogj, self.drivers = violations or {}, sdlogj, list(drivers)
        self.clock, self.tick, self.preflights = clock, tick, []
        self.cert_fail, self.pilot, self.eta_shift = set(cert_fail), pilot, set(eta_shift or ())
        # ETA4 round 2: run units without an exported state; run ids whose official SDlogJ is NaN
        self.no_export, self.sdlogj_nan = set(no_export), set(sdlogj_nan)

    def now(self):
        assert self.clock is not None, "the clock is read only with --no-start-after"
        return self.clock

    def drivers_running(self):
        return list(self.drivers)

    def pilot_state(self):
        return list(self.pilot[0]), list(self.pilot[1])

    def check_run_args(self, code, args):
        from bayesreg.cli import run_pair as rp
        self.preflights.append([str(a) for a in args])
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                record = json.loads(json.dumps(rp.args_record(rp.parse_args([str(a) for a in args]))))
        except SystemExit as exc:
            return False, f"run_pair parse_args exit {exc.code}: {err.getvalue().strip()[-300:]}", None
        return True, "run_pair parse_args: ok", record

    def unit_active(self, unit):
        return False

    def other_units(self):
        return list(self.busy)

    def active_units(self):
        return []

    def gpu_free_mib(self):
        return 24000

    def gpu_state(self):
        return "fake gpu"

    def mem_available_mib(self):
        return 12000

    def home_free_gb(self):
        return 500

    def train_state(self):
        return "tag-unet35-train: fake"

    def uptime(self):
        return 1.

    def loadavg(self):
        return "0 0 0"

    def dmesg_since(self, t0, pattern):
        return 0

    def sleep(self, seconds):
        raise AssertionError("no waiting in the tests")

    @staticmethod
    def split(cmd):
        """(unit, log, exit file, python argv) of a systemd-run command."""
        unit = next(c for c in cmd if c.startswith("--unit=")).split("=", 1)[1]
        if "/bin/bash" in cmd:
            k = cmd.index("/bin/bash")
            return unit, Path(cmd[k+4]), Path(cmd[k+5]), cmd[k+6:]
        return unit, None, None, cmd[cmd.index(next(c for c in cmd if c.endswith("python"))):]

    def run(self, cmd, record):
        unit, log, exit_file, argv = self.split(cmd)
        self.units.append(unit)
        arg = (lambda flag: argv[argv.index(flag)+1])
        if "-probe-" in unit:
            probe = load("eta_probe_fake", "experiments/eta/probe.py")
            assert argv[2] == "experiments/eta/probe.py"
            args = probe.parse_args([str(a) for a in argv[3:]])
            out = Path(arg("--out"))
            out.mkdir()
            short = re.search(r"probe-(.*)$", unit).group(1)
            doc = dict(nu_star=NU_PROBE[short], alpha_star=1., u1_state_hash="0"*64, channel_source=args.channel_source,
                       logvol_points=args.logvol_points, metric_jd_scaling=args.metric_jd_scaling,
                       configuration=probe.configuration(args), model_channels=14,
                       eta_tmp=dict(alpha=.01, nu=3.5, tau=[1., .5, .5, .5]))
            (out/"probe.json").write_text(json.dumps(doc))
            (out/"status.json").write_text(json.dumps(dict(status="probe_done", exit_code=0)))
        elif "-eval-" in unit:
            run, ev, rid = Path(argv[argv.index("bayesreg.eval.cli")+1]), Path(arg("--out-dir")), arg("--run-id")
            config = json.loads((run/"config.json").read_text())
            short = run.name
            role = self.roles.get(rid, "native_formal")
            seg35 = seg35_of(config["eta"]["nu"], config["eta"]["tau"])+OFFSET[short]
            sd = .05 if self.sdlogj is None else self.sdlogj(config["eta"]["nu"], config["eta"]["tau"])
            sd = float("nan") if rid in self.sdlogj_nan else sd
            exported = (json.loads((run/"status.json").read_text()).get("exported"))
            failed = rid in self.cert_fail
            (ev/rid).mkdir(parents=True)
            (ev/rid/"metrics.json").write_text(json.dumps(dict(
                run_id=rid, field_role=role if exported else "handover_only", field_present=bool(exported),
                run_status="budget_exhausted",
                metrics=dict(seg35=seg35, seg4=seg35+.05, hd95=2., sdlogj_official=sd, sdlogj_brain=.2,
                             logj_q01_brain=-.4, logj_q99_brain=.5, pct_J_lt_half_brain=1.5),
                certificate=dict(mode="full", status="certified_uncertain" if failed else "certified_feasible",
                                 passed=not failed, min_jd_lower_bound=None if failed else .1, n_prescreen_cells=7,
                                 n_fallback_cells=0, state_hash=exported, grid_id="g"*64))))
            Path(arg("--long")).write_text("x\n")
        elif "-atropos-" in unit:
            raise AssertionError("the synthetic tissue is complete; no Atropos unit expected")
        else:
            from bayesreg.cli import run_pair as rp
            assert argv[2:4] == ["-m", "bayesreg.cli.run_pair"]
            args = rp.parse_args([str(a) for a in argv[4:]])
            run = Path(args.out)
            if unit in self.drop:
                return 137                                     # killed: no status.json
            eta = json.loads(Path(args.eta_json).read_text())
            run.mkdir()
            record = rp.args_record(args)
            shift = 2. if unit in self.eta_shift else 1.
            (run/"config.json").write_text(json.dumps(dict(args=record, eta=dict(
                alpha=eta["alpha"], nu=eta["nu"]*shift, tau=eta["tau"]))))
            status = dict(status="budget_exhausted", wording="预算收尾", exit_code=0,
                          wall_budget=dict(seconds=args.wall_seconds),
                          structure_registry=dict(status="staging_unregistered"),
                          exported=None if unit in self.no_export else hashlib.sha256(unit.encode()).hexdigest())
            if args.jd_cell_estimate == "certificate_bernstein":        # run_pair writes the run total (JDTI)
                status["bernstein_j53_violations_total"] = self.violations.get(unit, 0)
            (run/"status.json").write_text(json.dumps(status))
            if status["exported"]:                                      # the exported file (bytes only: hashed)
                (run/"final_native.npz").write_bytes(f"synthetic final state of {unit}".encode())
            caps = getattr(args, "max_accepted_per_level", None) or [args.max_accepted]*4
            (run/"hyper.json").write_text(json.dumps(dict(rounds=[
                dict(level=0, round=0, status="cap_reached", accepted=caps[0], cap=caps[0], seconds=1.),
                dict(level=3, round=0, status="budget_exhausted", accepted=5, cap=caps[3], seconds=2.)])))
            if self.tick is not None:
                self.clock = self.clock+self.tick
        if log is not None:
            log.write_text(f"fake {unit}\n")
            exit_file.write_text("0\n")
        return 0


@pytest.fixture
def driver(sweep, dry, monkeypatch):
    def prepare(self):
        self.code, self.code_sha = ROOT, "fake"
        self.allowlist = self.allowlist or str(dry/"dev_subjects.json")
    monkeypatch.setattr(sweep.Sweep, "prepare_code", prepare)
    monkeypatch.setattr(sweep.Sweep, "write_env", lambda self: None)

    def go(root, system, *extra, structure_root=None):
        return sweep.main(["--root", str(root), "--dry-run", "--data-root", str(dry/"data"), "--allowlist",
                           str(dry/"dev_subjects.json"), "--tissue-root", str(dry/"seg"), "--structure-root",
                           str(structure_root or dry/"structure"), "--segmenter-registry",
                           str(dry/"registry"/"segmenters.json"), "--wall-seconds", "60", *extra], system=system)
    return go


def test_three_stages_end_to_end_and_restart(driver, tmp_path):
    system = FakeSystem()
    root = tmp_path/"sweep"
    assert driver(root, system) == 0
    probes = [u for u in system.units if "-probe-" in u]
    runs = [u for u in system.units if re.fullmatch(r"tag-etadry-s[123]-.*", u)]
    evals = [u for u in system.units if "-eval-" in u]
    assert probes == ["tag-etadry-probe-0901-0902", "tag-etadry-probe-0903-0904", "tag-etadry-probe-0905-0906"]
    assert len(runs) == 21 and len(evals) == 21
    assert not any(u.startswith(("tag-etadry-s2-x1-", "tag-etadry-s3-st1-")) for u in runs)      # reused, not rerun
    assert runs[0] == "tag-etadry-s1-x0p1-0901-0902"
    centre = json.loads((root/"probe"/"nu_star.json").read_text())
    assert centre["nu_star"] == pytest.approx(4.)
    assert centre["configuration"]["jd_cell_estimate"] == "certificate"
    assert centre["model_channels"] == {"0901-0902": 14, "0903-0904": 14, "0905-0906": 14}
    s1, s2, s3 = (json.loads((root/f"selection_{s}.json").read_text()) for s in ("s1", "s2", "s3"))
    assert (s1["chosen"], s2["chosen"], s3["chosen"]) == ("x10", "x0p25", "st2")
    assert s2["candidates"][1]["reused_from"] == "s1-x10" and s3["candidates"][1]["reused_from"] == "s2-x0p25"
    # ETA4: a v3 root has none of the v4 records or keys
    assert not {"design", "edge", "excluded_by_sdlogj"} & set(s1) and "mean_sdlogj_official" not in s1["candidates"][0]
    assert not list(root.glob("readouts_*")) and not (root/"path.json").exists() and not (root/"centre").exists()
    assert "bernstein_violations" not in s1["candidates"][0]["rows"][0]
    for s in (s1, s2, s3):
        c = s["configuration"]
        assert (c["channel_source"], c["logvol_points"], c["logvol_closed27_share"], c["closed27_weights"],
                c["jd_cell_estimate"], c["wall_seconds"]) == ("structure", "gauss27+closed27", 1., "simpson",
                                                              "certificate", 60.)
        assert c["structure"]["registry_status"] == "staging_unregistered" and c["registry"]["folds"] == [0, 1]
        assert all(r["configuration_check"].startswith("config.json args =") for cand_ in s["candidates"]
                   for r in cand_["rows"])
    selected = json.loads((root/"eta_selected.json").read_text())
    assert selected["nu"] == pytest.approx(10.) and selected["tau"] == [1., 1., 1., 1.] and selected["alpha"] == .01
    assert [p["chosen"] for p in selected["selection_path"]] == ["x10", "x0p25", "st2"]
    assert selected["configuration"]["logvol_points"] == "gauss27+closed27"
    summary = json.loads((root/"summary.json").read_text())
    assert summary["configuration"] == s3["configuration"]
    md = (root/"summary.md").read_text()
    assert "seg35" in md and "seg4" in md and "经验贝叶斯" in md and "st2" in md
    assert "jd_cell_estimate certificate" in md and "β_S 1（simpson）" in md
    eta = json.loads((root/"eta"/"s3_st2.json").read_text())
    assert eta["nu"] == pytest.approx(10.) and eta["tau"] == [1., 1., 1., 1.]
    text = next((root/"launch").glob("run_s1-x0p1-0901-0902_*.txt")).read_text()
    for piece in ("--channel-source structure", "--logvol-points gauss27+closed27", "--logvol-closed27-share 1",
                  "--closed27-weights simpson", "--jd-cell-estimate certificate", "--structure-infer never",
                  "--structure-registry-staging", "--segmenter-registry", "--wall-seconds 60", "--device cpu",
                  "--nice=19", "--eta-json", "configuration: {"):
        assert piece in text, piece
    text = next((root/"launch").glob("probe_0901-0902_*.txt")).read_text()
    for piece in ("--channel-source structure", "--logvol-closed27-share 1", "--jd-cell-estimate certificate",
                  "--structure-root", "--structure-registry-staging"):
        assert piece in text, piece
    assert "--structure-infer" not in text                # the probe never infers (no such option)
    # restart: nothing is launched again, the selections are recomputed and agree
    again = FakeSystem()
    assert driver(root, again) == 0
    assert again.units == []
    # restart under another configuration: the probes' records differ -> STOP, nothing launched
    other = FakeSystem()
    assert driver(root, other, "--logvol-closed27-share", "3") == 8
    assert other.units == [] and "differs from this sweep's" in (root/"sweep.log").read_text()


def test_missing_status_stops_and_is_never_relaunched(driver, tmp_path):
    root = tmp_path/"sweep"
    system = FakeSystem(drop={"tag-etadry-s1-x1-0903-0904"})
    assert driver(root, system) == 8
    assert system.units[-1] == "tag-etadry-s1-x1-0903-0904"
    assert "STOP" in (root/"sweep.log").read_text()
    again = FakeSystem()
    assert driver(root, again) == 8
    assert "tag-etadry-s1-x1-0903-0904" not in again.units and again.units == []


def test_ineligible_candidate_is_not_chosen(driver, tmp_path):
    root = tmp_path/"sweep"
    system = FakeSystem(roles={"eta_s1_x10_0905-0906": "handover_only"})
    assert driver(root, system, "--until", "s1") == 0
    s1 = json.loads((root/"selection_s1.json").read_text())
    assert s1["chosen"] == "x1"
    assert s1["candidates"][2]["eligible"] is False and "handover_only" in s1["candidates"][2]["ineligible"][0]


def test_busy_gate_stops_before_any_unit(driver, tmp_path):
    system = FakeSystem(busy=["tag-m2-H2.service"])
    root = tmp_path/"sweep"
    assert driver(root, system) == 8
    assert system.units == []
    assert "tag-m2-H2.service" in (root/"sweep.log").read_text()


def test_dry_run_refuses_real_paths(sweep, dry, tmp_path):
    assert sweep.main(["--root", str(sweep.REAL_ROOT/"x"), "--dry-run"], system=FakeSystem()) == 2
    assert sweep.main(["--root", "/home/rmcfz/seq_vp_tag_results/eta_sweep_v1/x", "--dry-run"], system=FakeSystem()) == 2
    base = ["--dry-run", "--allowlist", str(dry/"dev_subjects.json"), "--tissue-root", str(dry/"seg")]
    code = sweep.main(["--root", str(tmp_path/"r"), *base, "--data-root", str(ROOT/"data"/"three_d"),
                       "--structure-root", str(dry/"structure"), "--segmenter-registry",
                       str(dry/"registry"/"segmenters.json")], system=FakeSystem())
    assert code == 8 and "real data" in (tmp_path/"r"/"sweep.log").read_text()
    # structure in a dry run: --structure-root and --segmenter-registry required, never under the real roots
    code = sweep.main(["--root", str(tmp_path/"s"), *base, "--data-root", str(dry/"data")], system=FakeSystem())
    assert code == 8 and "--segmenter-registry are required" in (tmp_path/"s"/"sweep.log").read_text()
    code = sweep.main(["--root", str(tmp_path/"t"), *base, "--data-root", str(dry/"data"), "--structure-root",
                       "/home/rmcfz/seq_vp_tag_results/seg/unet35_v1/predx", "--segmenter-registry",
                       str(dry/"registry"/"segmenters.json")], system=FakeSystem())
    assert code == 8 and "real data" in (tmp_path/"t"/"sweep.log").read_text()
    # --segmenter-registry outside a dry run
    code = sweep.main(["--root", str(tmp_path/"u"), "--segmenter-registry", str(dry/"registry"/"segmenters.json")],
                      system=FakeSystem())
    assert code == 8 and "dry runs only" in (tmp_path/"u"/"sweep.log").read_text()


# ---- structure precheck ---------------------------------------------------------------------------------------------

def copy_structure(dry, tmp_path):
    target = tmp_path/"structure"
    shutil.copytree(dry/"structure", target)
    return target


def test_precheck_passes_and_until_precheck_launches_nothing(driver, tmp_path):
    system = FakeSystem()
    root = tmp_path/"sweep"
    assert driver(root, system, "--until", "precheck") == 0
    assert system.units == []
    log = (root/"sweep.log").read_text()
    assert "predictions of 6 subjects present" in log and "until precheck: end" in log
    assert not list(root.glob("structure_infer_subjects_*.txt"))


def test_precheck_missing_prediction_gives_the_inference_command(driver, dry, tmp_path, sweep):
    structure = copy_structure(dry, tmp_path)
    shutil.rmtree(structure/"OASIS_OAS1_0904_MR1")
    shutil.rmtree(structure/"OASIS_OAS1_0905_MR1")
    (structure/"OASIS_OAS1_0905_MR1").mkdir()                              # an empty directory counts as missing
    system = FakeSystem()
    root = tmp_path/"sweep"
    assert driver(root, system, structure_root=structure) == 8
    assert system.units == []
    files = list(root.glob("structure_infer_subjects_*.txt"))
    assert len(files) == 1 and files[0].read_text() == "OASIS_OAS1_0904_MR1\nOASIS_OAS1_0905_MR1\n"
    log = (root/"sweep.log").read_text()
    command = (f"cd {sweep.REPO} && {sweep.INFER_ENV} {sweep.PY} -B -m bayesreg.seg.unet.cli infer --subjects-file "
               f"{files[0]} --pred-out {structure}")
    assert command in log and "STOP" in log and "all_folds_mean" in log
    # a second start writes a new subject file (never overwrites) and still launches nothing
    again = FakeSystem()
    assert driver(root, again, structure_root=structure) == 8 and again.units == []
    files = sorted(root.glob("structure_infer_subjects_*.txt"))
    assert len(files) == 2 and {f.read_text() for f in files} == {"OASIS_OAS1_0904_MR1\nOASIS_OAS1_0905_MR1\n"}


def test_precheck_refuses_unfit_or_incomplete_predictions(driver, dry, tmp_path):
    structure = copy_structure(dry, tmp_path)
    prov = structure/"OASIS_OAS1_0902_MR1"/"provenance.json"
    document = json.loads(prov.read_text())
    prov.write_text(json.dumps(dict(document, policy="cross_fit_other_folds")))
    root = tmp_path/"a"
    system = FakeSystem()
    assert driver(root, system, structure_root=structure) == 8 and system.units == []
    assert "policy 'cross_fit_other_folds'" in (root/"sweep.log").read_text()
    prov.write_text(json.dumps(document))
    (structure/"OASIS_OAS1_0906_MR1"/"provenance.json").unlink()            # structure_groups.npz alone: incomplete
    root = tmp_path/"b"
    assert driver(root, system, structure_root=structure) == 8 and system.units == []
    text = (root/"sweep.log").read_text()
    assert "incomplete prediction" in text and not list(root.glob("structure_infer_subjects_*.txt"))


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          capture_output=True, text=True, check=True)


def test_real_mode_registry_must_be_committed_and_clean(sweep, dry, tmp_path):
    code = tmp_path/"code"
    (code/"configs"/"bayesreg").mkdir(parents=True)
    git(code, "init", "-q")
    registry = code/"configs"/"bayesreg"/"segmenters.json"
    shutil.copy(dry/"registry"/"segmenters.json", registry)
    root = tmp_path/"root"
    root.mkdir()
    opts = sweep.parse_args(["--root", str(root), "--data-root", str(dry/"data"), "--allowlist",
                             str(dry/"dev_subjects.json"), "--tissue-root", str(dry/"seg"), "--structure-root",
                             str(tmp_path/"pred")])
    run = sweep.Sweep(opts, FakeSystem())
    run.code = code
    run.pairs = sweep.select_pairs(json.loads((dry/"data"/STAGE/"selection.json").read_text()))
    with pytest.raises(sweep.Stop, match="committed and clean"):
        run.structure_precheck()                                          # untracked
    git(code, "add", "configs/bayesreg/segmenters.json")
    git(code, "commit", "-q", "-m", "registry")
    registry.write_text(registry.read_text()+"\n")
    with pytest.raises(sweep.Stop, match="committed and clean"):
        run.structure_precheck()                                          # tracked, modified
    git(code, "checkout", "--", "configs/bayesreg/segmenters.json")
    with pytest.raises(sweep.Stop, match="run the inference first") as stop:
        run.structure_precheck()                                          # registry fine; no prediction at all
    assert "--pred-out" in str(stop.value) and "infer --subjects-file" in str(stop.value)
    files = list(root.glob("structure_infer_subjects_*.txt"))
    assert len(files) == 1 and files[0].read_text().split() == SUBJECTS
    assert run.registry_record["status"] == "registered" and run.registry_record["git"]["clean"] is True
    assert run.probe_configuration()["structure"]["registry_status"] == "registered"
    assert "--segmenter-registry" not in run.model_args() and "--structure-registry-staging" not in run.model_args()
    # the default prediction root is the inference's own default: no --pred-out
    default = sweep.Sweep(sweep.parse_args(["--root", str(root)]), FakeSystem())
    command = default.infer_command(Path("/x/t0.txt"))
    assert command.endswith("bayesreg.seg.unet.cli infer --subjects-file /x/t0.txt") and "--pred-out" not in command


# ---- the dry run in process (slow) ----------------------------------------------------------------------------------

class InlineSystem(FakeSystem):
    """Runs the units' commands in this process instead of systemd: experiments/eta/probe.py, bayesreg.cli.run_pair and
    bayesreg.eval.cli with the exact argument lists the sweep built (the dry run of the final configuration)."""

    def run(self, cmd, record):
        unit, log, exit_file, argv = self.split(cmd)
        self.units.append(unit)
        argv = [str(a) for a in argv]
        if argv[2] == "experiments/eta/probe.py":
            entry, rest = load("eta_probe_inline", "experiments/eta/probe.py").main, argv[3:]
        elif argv[2:4] == ["-m", "bayesreg.cli.run_pair"]:
            from bayesreg.cli import run_pair
            entry, rest = run_pair.main, argv[4:]
        elif argv[2:4] == ["-m", "bayesreg.eval.cli"]:
            from bayesreg.eval import cli
            entry, rest = cli.main, argv[4:]
        else:
            raise AssertionError(f"unexpected unit {unit}: {argv}")
        if log is None:
            return entry(rest)
        with open(log, "a", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), \
                contextlib.redirect_stderr(stream):
            rc = entry(rest)
        exit_file.write_text(f"{rc}\n")
        return rc


@pytest.mark.slow
def test_dry_run_final_configuration_in_process(sweep, dry, tmp_path, monkeypatch):
    def prepare(self):
        self.code, self.code_sha = ROOT, "inline"
        self.allowlist = self.allowlist or str(dry/"dev_subjects.json")
    monkeypatch.setattr(sweep.Sweep, "prepare_code", prepare)
    monkeypatch.setattr(sweep.Sweep, "write_env", lambda self: None)
    root = tmp_path/"sweep"
    args = ["--root", str(root), "--dry-run", "--data-root", str(dry/"data"), "--allowlist", str(dry/"dev_subjects.json"),
            "--tissue-root", str(dry/"seg"), "--structure-root", str(dry/"structure"), "--segmenter-registry",
            str(dry/"registry"/"segmenters.json"), "--logvol-closed27-share", "3", "--wall-seconds", "120",
            "--extra", "--max-accepted 2"]
    system = InlineSystem()
    assert sweep.main(args, system=system) == 0, (root/"sweep.log").read_text()[-3000:]
    assert len([u for u in system.units if "-probe-" in u]) == 3
    assert len([u for u in system.units if re.fullmatch(r"tag-etadry-s[123]-.*", u)]) == 21
    assert len([u for u in system.units if "-eval-" in u]) == 21
    for probe in (root/"probe").iterdir():
        if probe.is_dir():
            record = json.loads((probe/"probe.json").read_text())
            assert record["model_channels"] == 14 and record["configuration"]["logvol_closed27_share"] == 3.
    for config in (root/"runs").glob("*/*/*/config.json"):
        a = json.loads(config.read_text())["args"]
        assert (a["channel_source"], a["logvol_points"], a["logvol_closed27_share"], a["jd_cell_estimate"],
                a["structure_infer"], a["structure_registry_staging"]) == ("structure", "gauss27+closed27", 3.,
                                                                          "certificate", "never", True)
        prov = json.loads((config.parent/"provenance.json").read_text())
        assert prov["channels"]["K_S"] == 13 and prov["switches"]["jd_cell_estimate"] == "certificate"
    for s in ("s1", "s2", "s3"):
        selection = json.loads((root/f"selection_{s}.json").read_text())
        rows = [r for c in selection["candidates"] for r in c["rows"]]
        assert all(r["field_role"] == "native_formal" and r["certificate"]["passed"] is True for r in rows)
    assert (root/"eta_selected.json").is_file() and (root/"summary.md").is_file()
    again = InlineSystem()
    assert sweep.main(args, system=again) == 0 and again.units == []


# ---- ETA4: design v4, the solver path, the caps, the deadline -------------------------------------------------------

V3_RUN_ARGS = ["--pair", "stage_T0_tune_v1/OASIS_OAS1_0012_MR1__OASIS_OAS1_0013_MR1", "--channel-source", "structure",
               "--logvol-points", "gauss27+closed27", "--logvol-closed27-share", "1", "--closed27-weights", "simpson",
               "--metric-jd-scaling", "affine_v1", "--jd-cell-estimate", "certificate", "--structure-root",
               "/home/rmcfz/seq_vp_tag_results/seg/unet35_v1/pred", "--structure-grouping", "structure13_v1",
               "--segmenter-id", "unet35_v1_ens", "--structure-infer", "never", "--backend", "torch64", "--device",
               "cuda", "--outer", "a", "--rounds", "1,1,1,1", "--wall-seconds", "900", "--cpu-threads", "4"]
V3_CONFIG_KEYS = ["channel_source", "logvol_points", "logvol_closed27_share", "closed27_weights", "metric_jd_scaling",
                  "jd_cell_estimate", "structure", "backend", "device", "outer", "rounds", "wall_seconds", "cpu_threads",
                  "opt1", "extra", "registry"]
LEAN = "倾向列入候选（交总负责人）"
UNDECIDED = "无法区分（只描述）"


def test_v3_command_and_configuration_unchanged(sweep, tmp_path):
    """The v3 command line (eta_sweep_v3: only --root and --logvol-closed27-share 1) gives the run_pair arguments, the
    systemd command and the configuration keys of the v3 launch records (the same 42 records were also rebuilt byte for
    byte from the real eta_sweep_v3/launch/ outside the tests, read only; ETA4 report)."""
    root = tmp_path/"v3"
    opts = sweep.parse_args(["--root", str(root), "--logvol-closed27-share", "1"])
    assert (opts.design, opts.max_accepted_per_level, opts.no_start_after_utc) == ("v3", None, None)
    s = sweep.Sweep(opts, FakeSystem())
    s.code, s.code_sha, s.allowlist = Path("/c"), "x", "/c/configs/bayesreg/dev_subjects.json"
    s.registry_path = Path("/c/configs/bayesreg/segmenters.json")
    s.registry_record = dict(path=str(s.registry_path), sha256="0"*64, folds=[0, 1], status="registered", git={})
    pair = "OASIS_OAS1_0012_MR1__OASIS_OAS1_0013_MR1"
    eta_file = root/"eta"/"s1_x0p1.json"
    want = V3_RUN_ARGS+["--data-root", str(sweep.REPO/"data"/"three_d"), "--allowlist", s.allowlist, "--tissue-root",
                        "/home/rmcfz/seq_vp_tag_results/seg/atropos_v1", "--eta-json", str(eta_file)]
    assert s.run_args(pair, eta_file) == want
    command = [sweep.PY, "-B", "-m", "bayesreg.cli.run_pair", "--out", root/"runs"/"s1"/"x0p1"/"0012-0013", *want]
    sd = s.systemd_command(kind="run", unit="tag-eta-s1-x0p1-0012-0013", command=command, memmax_text="9G", gpu=True,
                           nice=None, runtime=2700, quota="600%")
    assert sd[:11] == ["systemd-run", "--user", "--unit=tag-eta-s1-x0p1-0012-0013", "--wait", "--collect",
                       "--expand-environment=no", "--working-directory=/c", "--property=MemoryMax=9G",
                       "--property=MemorySwapMax=4G", "--property=CPUQuota=600%", "--property=RuntimeMaxSec=2700"]
    assert sd[11:18] == [f"--setenv={k}=4" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                       "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")] + \
        ["--setenv=PYTHONDONTWRITEBYTECODE=1", f"--setenv=NUMBA_CACHE_DIR={root/'cache'/'numba'}"]
    assert sd[18:] == [str(c) for c in command]
    assert list(s.configuration()) == V3_CONFIG_KEYS
    assert set(s.expected_run_args()) == {"channel_source", "logvol_points", "logvol_closed27_share",
                                          "closed27_weights", "metric_jd_scaling", "jd_cell_estimate", "backend",
                                          "outer", "rounds", "wall_seconds", "device", "structure_root",
                                          "structure_grouping", "segmenter_id", "structure_infer",
                                          "structure_registry_staging"}
    assert s.labels is sweep.LABELS and not s.v4
    # explicit caps in v3: passed to run_pair (last, before --extra), recorded, checked (C16 SCH)
    capped = sweep.Sweep(sweep.parse_args(["--root", str(root), "--max-accepted-per-level", "500,500,100,500"]),
                         FakeSystem())
    capped.code, capped.allowlist = Path("/c"), s.allowlist
    assert capped.run_args(pair, eta_file)[-2:] == ["--max-accepted-per-level", "500,500,100,500"]
    assert capped.configuration()["max_accepted_per_level"] == [500, 500, 100, 500]
    e = capped.expected_run_args()
    assert (e["max_accepted"], e["max_accepted_per_level"], e["level_time_shares"]) == (None, [500, 500, 100, 500], None)
    assert "--max-accepted-per-level" not in capped.model_args()                 # never to the probe


DEADLINE = ["--no-start-after", "2026-11-02T06:00-06:00"]


def test_v4_option_checks(sweep):
    base = ["--design", "v4", "--path", "p3", "--path-note", "lead 11-01"]
    args = sweep.parse_args(base+DEADLINE)
    assert args.root == str(sweep.REAL_ROOT_V4) and args.centre_eta == str(sweep.V3_ETA)
    assert args.jd_cell_estimate == "certificate"         # set by the path at the start (resolve_path)
    # contract 7b (review I1): the SDlogJ condition is on by default under v4; off / another value only with a note
    assert (args.sdlogj_official_max, args.sdlogj_note) == (.1, None)
    off = sweep.parse_args(base+DEADLINE+["--sdlogj-official-max", "none", "--sdlogj-note", "BZ 11-02: off"])
    assert (off.sdlogj_official_max, off.sdlogj_note) == (None, "BZ 11-02: off")
    assert sweep.parse_args(base+DEADLINE+["--sdlogj-official-max", ".08", "--sdlogj-note", "n"]).sdlogj_official_max \
        == .08
    assert sweep.parse_args(base+DEADLINE+["--sdlogj-official-max", "0.10"]).sdlogj_official_max == .1
    # review M8: a deadline only where a real v4 call starts runs
    assert sweep.parse_args(base+["--until", "centre"]).until == "centre"
    assert sweep.parse_args(base+["--until", "precheck"]).no_start_after_utc is None
    for bad in (["--design", "v4"]+DEADLINE,                                     # no path record
                ["--design", "v4", "--path", "p3"]+DEADLINE,                     # no note
                base, base+["--until", "s1"],                                    # real, runs, no deadline
                base+DEADLINE+["--until", "probe"], base+DEADLINE+["--jd-cell-estimate", "certificate_bernstein"],
                base+DEADLINE+["--max-accepted-per-level", "500,500,100,500"],
                base+DEADLINE+["--metric-jd-scaling", "off"],
                base+DEADLINE+["--sdlogj-official-max", "none"],                # off without a note
                base+DEADLINE+["--sdlogj-official-max", ".2"],                  # another value without a note
                base+DEADLINE+["--sdlogj-official-max", "0", "--sdlogj-note", "n"],
                base+DEADLINE+["--sdlogj-official-max", "x", "--sdlogj-note", "n"],
                base+DEADLINE+["--sdlogj-note", "n"],                           # a note for the contract's value
                base+DEADLINE+["--sigma-domain", "support_shared"],              # 7b scope: brain only
                base+DEADLINE+["--dry-run"],                                     # a dry run names its synthetic centre
                ["--centre-eta", "/x.json"], ["--path", "p1", "--path-note", "n"], ["--path-from-pilot", "/s.json"],
                ["--sdlogj-official-max", ".1"], ["--sdlogj-note", "n"], ["--accept-code-change"],
                ["--until", "centre"], ["--path-note", "n"],
                ["--max-accepted-per-level", "500,500,100"], ["--max-accepted-per-level", "+1,2,3,4"],
                ["--max-accepted-per-level", "1,1,1,1", "--extra", "--max-accepted 2"],
                ["--extra", "--max-accepted-per-level 1,1,1,1"], ["--extra", "--max-accepted-p 1,1,1,1"],
                ["--extra", "--level-time-shares .1,.1,.1"], ["--extra", "--level .1,.1,.1"],
                ["--no-start-after", "08:30"], ["--no-start-after", "2026-11-02T08:30"]):
        with pytest.raises(SystemExit):
            sweep.parse_args(bad)
    assert sweep.parse_args(["--extra", "--max-accepted 3"]).extra == "--max-accepted 3"       # v3 dry runs
    assert sweep.parse_args([]).sdlogj_official_max is None                                   # v3: v3's rule
    deadline = sweep.parse_args(DEADLINE).no_start_after_utc
    assert deadline == datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
    assert sweep.extra_conflicts("--max-accepted 2", None) == []
    assert sweep.extra_conflicts("--max-accepted 2", [1, 1, 1, 1]) == ["--max-accepted"]
    assert sweep.extra_conflicts("--max-outer 3 --max-levels 2", [1, 1, 1, 1]) == []
    assert [label for label, _ in sweep.LABELS_V4["s1"]] == ["x0p0625", "x0p25", "x1", "x4"]
    assert [sweep.scaled(21.855, f) for _, f in sweep.LABELS_V4["s1"]] == [21.855/16, 21.855/4, 21.855, 21.855*4]
    assert [sweep.scaled(8., f) for _, f in sweep.LABELS_V4["s2"]] == [4., 8., 16.]
    assert sweep.LABELS_V4["s3"] == sweep.LABELS["s3"] and sweep.REUSED == dict(s2="x1", s3="st1")
    cands = [dict(label="a", nu=1.), dict(label="b", nu=4.), dict(label="c", nu=16.)]
    assert [sweep.edge_of(cands, x, "nu") for x in "abc"] == ["lower", "interior", "upper"]
    # review M9 and M6: v4 units are tag-eta4 / tag-eta4dry; expected args of the current path
    s = sweep.Sweep(args, FakeSystem())
    assert s.prefix == "tag-eta4" and sweep.Sweep(sweep.parse_args(DEADLINE), FakeSystem()).prefix == "tag-eta"
    s.o.jd_cell_estimate, s.level_caps, s.path_name = "certificate", None, "current"
    e = s.expected_run_args()
    assert (e["max_accepted"], e["max_accepted_per_level"], e["level_time_shares"], e["sigma_domain"],
            e["support_threshold"]) == (500, None, None, "brain", None)


@pytest.fixture(scope="module")
def pilot_sum(sweep):
    """The path pilot's own summarize.py (arm_flags, recommend), as sweep.resolve_path loads it."""
    return sweep.load_module("_t_pilot_summarize", "experiments/pilot/summarize.py")


def pilot_arm(kind, *, bernstein, n=5):
    """(rows, comparison against P0n) of one pilot arm that the pilot's arm_flags reads as `kind`: lean (native steps
    x2.5 on every pair, Dice unchanged), undecided (steps x1), review (lean, one check-(b) violation), check (lean, one
    count not recorded)."""
    rows = []
    for i in range(n):
        violations = 0 if bernstein else None
        if kind == "review" and i == 0:
            violations = 1
        if kind == "check" and i == 0:
            violations = None
        rows.append(dict(state="ran", comparable=True, exported="x", field_role="native_formal", short=f"p{i}",
                         pair=f"pair{i}", n_blocked=0,
                         bernstein=dict(violations=violations) if (bernstein or kind == "check") else None))
    ratio, positive = (1., 0) if kind == "undecided" else (2.5, n)
    comparison = dict(seg35=dict(n=n, median=0.), seg4=dict(n=n, median=0.),
                      native_accepted=dict(n=n, median_ratio=ratio, positive=positive),
                      native_gf_over_tb=dict(n=n, per_pair=[dict(ratio=1.)]*n, negative=0))
    return rows, comparison


def write_pilot(directory, kinds, pilot_sum, *, eta=None, changes=None, schema=None, name="summary.json", root=None,
                rec=None, tamper=None, log="ended", log_names=None, counts=None):
    """A synthetic path-pilot root: summary.json in the format of experiments/pilot/summarize.py summary() (rows,
    comparisons, the arms' switches, flags from the pilot's own arm_flags, the recommendation from its recommend,
    counts.bernstein_violation_runs) and queue.log ("ended": a summary line naming the file, then "end"; "running":
    no end). `rec` / `tamper` overwrite the recorded recommendation / verdicts after the fact."""
    directory.mkdir(parents=True, exist_ok=True)
    arms = dict(p0n={}, p1=dict(max_accepted_per_level=[500, 500, 100, 500]),
                p2=dict(jd_cell_estimate="certificate_bernstein"),
                p3=dict(jd_cell_estimate="certificate_bernstein", max_accepted_per_level=[500, 500, 100, 500]),
                p4=dict(jd_cell_estimate="certificate_bernstein", max_accepted_per_level=[500, 500, 100, 500],
                        sigma_domain="support_shared", support_threshold=.05))
    arms.update(changes or {})
    base_rows = [dict(state="ran", comparable=True, exported="x", field_role="native_formal", short=f"p{i}",
                      pair=f"pair{i}", n_blocked=0, bernstein=None) for i in range(5)]
    rows, comparisons, flags = dict(p0n=base_rows), {}, {}
    for a in ("p1", "p2", "p3", "p4"):
        bern = a != "p1"
        rows[a], comparisons[a] = pilot_arm(kinds.get(a, "undecided"), bernstein=bern)
        if a != "p4":
            flags[a] = pilot_sum.arm_flags(a, rows[a], rows["p0n"], comparisons[a],
                                           "certificate_bernstein" if bern else "certificate",
                                           [500, 500, 100, 500] if a in ("p1", "p3") else None)
    flags["p4"] = dict(verdict="只描述（探索臂 P4：σ 估计域 support_shared；不贴候选标签）")
    extra = dict(p3_vs_p1=dict(native_accepted=dict(median_ratio=1.), seg35=dict(median=0.), seg4=dict(median=0.)))
    path, why = pilot_sum.recommend({a: f["verdict"] for a, f in flags.items()}, extra["p3_vs_p1"])
    for a, verdict in (tamper or {}).items():
        flags[a] = dict(flags[a], verdict=verdict)
    bad = [dict(arm=a, short=r["short"], violations=r["bernstein"]["violations"]) for a in ("p1", "p2", "p3", "p4")
           for r in rows[a] if isinstance((r.get("bernstein") or {}).get("violations"), int)
           and r["bernstein"]["violations"] > 0]
    doc = dict(schema=schema or "bayesreg.path_pilot.summary.v1", note="synthetic (tests; ETA4)",
               root=str(root or directory), baseline="p0n", arm_order=["p0n", "p1", "p2", "p3", "p4"],
               arms={a: dict(label=a.upper(), changes=c) for a, c in arms.items()},
               eta=eta or dict(alpha=.01, nu=1024., tau=[1., .5, .5, .5]), rows=rows, comparisons=comparisons,
               extra_comparisons=extra, flags=flags,
               counts=dict(dict(ran=25, not_run=0, failed_no_status=0), **(counts or {}), bernstein_violation_runs=bad),
               recommendation=dict(path=rec or path, reason=why))
    target = directory/name
    target.write_text(json.dumps(doc))
    if log is not None:
        named = log_names or target
        lines = ["2026-11-01T14:30:00Z start: synthetic", f"2026-11-01T14:31:00Z summary: {named}, "
                 f"{Path(named).with_suffix('.md')} (new)"]
        if log == "ended":
            lines.append("2026-11-01T14:31:01Z end")
        (directory/"queue.log").write_text("\n".join(lines)+"\n")
    return target


def test_pilot_path_taken_mechanically(sweep, pilot_sum, tmp_path):
    eta = dict(alpha=.01, nu=1024., tau=[1., .5, .5, .5])
    take = (lambda p, **kw: sweep.pilot_path(p, eta=eta, dry=True, summarize=pilot_sum, **kw))
    for case in ("p3", "p1", "p2"):
        r = take(write_pilot(tmp_path/case, {case: "lean"}, pilot_sum))
        assert (r["status"], r["path"], r["pilot_recommendation"]) == ("resolved", case, case) and not r["reasons"]
        assert r["verdicts_recomputed"] == r["verdicts"] and r["verdicts"][case].startswith("倾向")
        assert sweep.PATHS[r["path"]]["cell"] == ("certificate_bernstein" if case in ("p2", "p3") else "certificate")
    # round 2 R2-6: a pilot stopped by its deadline still resolves; its counts are carried along
    r = take(write_pilot(tmp_path/"cut", {"p3": "lean"}, pilot_sum, counts=dict(ran=20, not_run=4, failed_no_status=1)))
    assert r["status"] == "resolved" and r["pilot_counts"] == dict(ran=20, not_run=4, failed_no_status=1)
    # the rule's branch 4: no arm leans -> the current path, the lead decides (no automatic start)
    r = take(write_pilot(tmp_path/"cur", {}, pilot_sum))
    assert (r["status"], r["path"]) == ("lead", "current") and "交总负责人" in r["reasons"][0]
    # missing summary; the summarize tool not loadable (contract 7b; review M1)
    r = take(tmp_path/"none"/"summary.json")
    assert (r["status"], r["path"]) == ("hold", "current") and "汇总缺失" in r["reasons"][0]
    r = sweep.pilot_path(write_pilot(tmp_path/"nosum", {"p3": "lean"}, pilot_sum), eta=eta, dry=True, summarize=None)
    assert r["status"] == "hold" and "载入失败" in r["reasons"][0]
    # 待审核: a check-(b) violation in the pilot (P3 待审核, P1 leans: the rule falls to P1; held here)
    r = take(write_pilot(tmp_path/"rev", {"p1": "lean", "p3": "review"}, pilot_sum))
    assert r["verdicts"]["p3"].startswith("待审核") and r["pilot_recommendation"] == "p1"
    assert r["status"] == "hold" and "D53" in r["reasons"][0] and r["bernstein_violation_runs"][0]["violations"] == 1
    r = take(write_pilot(tmp_path/"chk", {"p1": "lean", "p2": "check"}, pilot_sum))
    assert r["verdicts"]["p2"].startswith("待核对") and r["status"] == "hold" and "待核对" in r["reasons"][0]
    # the recorded verdicts and recommendation must follow from the summary's own data (review M2)
    r = take(write_pilot(tmp_path/"tamper", {}, pilot_sum, tamper=dict(p3=LEAN), rec="p3"))
    assert r["status"] == "hold" and "重算" in r["reasons"][0] and "读数标签" in r["reasons"][0]
    r = take(write_pilot(tmp_path/"rec", {}, pilot_sum, rec="p3"))
    assert r["status"] == "hold" and "建议" in r["reasons"][0]
    # the pilot has not ended (queue.log), or its last summary line names another file (contract 7b; review M3)
    r = take(write_pilot(tmp_path/"run", {"p3": "lean"}, pilot_sum, log="running"))
    assert r["status"] == "hold" and "尚未结束" in r["reasons"][0]
    r = take(write_pilot(tmp_path/"other", {"p3": "lean"}, pilot_sum, log_names=tmp_path/"other"/"summary_x.json"))
    assert r["status"] == "hold" and "summary_x.json" in r["reasons"][0]
    assert take(write_pilot(tmp_path/"nolog", {"p3": "lean"}, pilot_sum, log=None))["status"] == "hold"
    # another eta, other arm switches, another schema, another root, a temp summary in a real run
    assert "中心 η" in take(write_pilot(tmp_path/"eta", {"p3": "lean"}, pilot_sum, eta=dict(eta, nu=1.)))["reasons"][0]
    r = take(write_pilot(tmp_path/"arms", {"p3": "lean"}, pilot_sum,
                         changes=dict(p1=dict(max_accepted_per_level=[1]*4))))
    assert r["status"] == "hold" and "p1" in r["reasons"][0]
    assert take(write_pilot(tmp_path/"sch", {"p3": "lean"}, pilot_sum, schema="x"))["status"] == "hold"
    assert take(write_pilot(tmp_path/"rt", {"p3": "lean"}, pilot_sum, root=tmp_path/"else"))["status"] == "hold"
    r = sweep.pilot_path(write_pilot(tmp_path/"real", {"p3": "lean"}, pilot_sum), eta=eta, dry=False,
                         summarize=pilot_sum)
    assert r["status"] == "hold" and "真实运行" in r["reasons"][0]
    # not the newest summary of its directory (a differing recomputation went to summary_<UTC>.json)
    first = write_pilot(tmp_path/"two", {}, pilot_sum, log=None)
    newer = write_pilot(tmp_path/"two", {"p3": "lean"}, pilot_sum, name="summary_20261101T150000Z.json")
    r = take(first)
    assert r["status"] == "hold" and "最新" in r["reasons"][0]
    r = take(newer)
    assert (r["status"], r["path"]) == ("resolved", "p3") and r["other_summaries"] == [str(first)]


def write_centre(sweep, dry, directory, nu=1024., *, alpha=None, pairs=None, args_change=None, **conf_changes):
    """A synthetic v3-like centre: eta_selected.json and summary.json (the configuration a v3 dry sweep on the same dry
    inputs records, built by the sweep's own code; eta alpha_nug(tau), nu, tau^(0)), and the config.json of a v3 run of
    the first pair (run_pair's own parse_args / args_record on the v3 design's run arguments; `args_change` edits it)."""
    from bayesreg.cli import run_pair as rp
    directory.mkdir(parents=True, exist_ok=True)
    opts = sweep.parse_args(["--root", str(directory), "--dry-run", "--data-root", str(dry/"data"), "--allowlist",
                             str(dry/"dev_subjects.json"), "--tissue-root", str(dry/"seg"), "--structure-root",
                             str(dry/"structure"), "--segmenter-registry", str(dry/"registry"/"segmenters.json"),
                             "--wall-seconds", "60"])
    s = sweep.Sweep(opts, FakeSystem())
    s.code = ROOT
    s.check_registry()
    configuration = dict(s.configuration(), **conf_changes)
    tau = [1., .5, .5, .5]
    eta = dict(alpha=.01 if alpha is None else alpha, nu=nu, tau=tau)
    real_pairs = sweep.select_pairs(json.loads((dry/"data"/STAGE/"selection.json").read_text()))
    (directory/"eta_selected.json").write_text(json.dumps(dict(eta, configuration=configuration, code_sha="fake",
                                                               note="synthetic (tests; ETA4)")))
    (directory/"summary.json").write_text(json.dumps(dict(eta_selected=eta, stage_set=STAGE,
                                                          pairs=real_pairs if pairs is None else pairs)))
    run = directory/"runs"/"s1"/"x1"/sweep.short_name(real_pairs[0])
    run.mkdir(parents=True)
    args = rp.args_record(rp.parse_args(["--out", str(run), *s.run_args(real_pairs[0], directory/"eta"/"s1_x1.json")]))
    for key, value in (args_change or {}).items():          # ... deletes a key (a v3 record without it)
        if value is ...:
            args.pop(key, None)
        else:
            args[key] = value
    (run/"config.json").write_text(json.dumps(dict(args=args, eta=eta)))
    (run/"cmd.txt").write_text(f"{ROOT/'bayesreg'/'cli'/'run_pair.py'} --out {run}\n")       # as run_pair writes it
    return directory/"eta_selected.json"


@pytest.fixture
def driver4(sweep, dry, driver):
    """The dry v4 driver on the fake machine (the `driver` fixture's code-tree patches)."""
    def go(root, system, *extra, centre, pilot=None, path=None):
        args = ["--root", str(root), "--dry-run", "--design", "v4", "--data-root", str(dry/"data"), "--allowlist",
                str(dry/"dev_subjects.json"), "--tissue-root", str(dry/"seg"), "--structure-root", str(dry/"structure"),
                "--segmenter-registry", str(dry/"registry"/"segmenters.json"), "--wall-seconds", "60",
                "--centre-eta", str(centre)]
        if pilot is not None:
            args += ["--path-from-pilot", str(pilot)]
        if path is not None:
            args += ["--path", path, "--path-note", "lead (test)"]
        return sweep.main(args+list(extra), system=system)
    return go


def test_v4_end_to_end_p3_from_the_pilot_and_restart(sweep, dry, driver4, pilot_sum, tmp_path):
    centre = write_centre(sweep, dry, tmp_path/"v3")
    pilot = write_pilot(tmp_path/"pilot", {"p3": "lean"}, pilot_sum)
    root, system = tmp_path/"v4", FakeSystem()
    assert driver4(root, system, centre=centre, pilot=pilot) == 0, (root/"sweep.log").read_text()[-2000:]
    runs = [u for u in system.units if re.fullmatch(r"tag-eta4dry-s[123]-.*", u)]
    assert len(runs) == 24 and len([u for u in system.units if "-eval-" in u]) == 24
    assert all(u.startswith("tag-eta4dry-") for u in system.units)               # review M9
    assert not [u for u in system.units if "-probe-" in u]                       # v4: no probe
    assert runs[0] == "tag-eta4dry-s1-x0p0625-0901-0902"
    assert not any(u.startswith(("tag-eta4dry-s2-x1-", "tag-eta4dry-s3-st1-")) for u in runs)
    assert len(system.preflights) == 1 and system.preflights[0][-2:] == ["--max-accepted-per-level", "500,500,100,500"]
    path = json.loads((root/"path.json").read_text())
    assert (path["path"], path["source"], path["pilot"]["status"]) == ("p3", "pilot (--path-from-pilot)", "resolved")
    assert path["run_pair_switches"] == ["--jd-cell-estimate", "certificate_bernstein", "--max-accepted-per-level",
                                         "500,500,100,500"]
    assert path["pilot"]["summarize_tool"]["sha256"] == sweep.file_sha256(sweep.REPO/sweep.PILOT_SUMMARIZE)
    # round 2 R2-6: the pilot's completeness in path.json and on the "path:" line
    assert path["pilot_counts"] == path["pilot"]["pilot_counts"] == dict(ran=25, not_run=0, failed_no_status=0)
    assert "pilot counts {'ran': 25, 'not_run': 0, 'failed_no_status': 0}" in (root/"sweep.log").read_text()
    c = json.loads((root/"centre"/"centre.json").read_text())
    assert (c["nu_star"], c["tau0"], c["design"]) == (1024., [1., .5, .5, .5], "v4")
    assert c["grid"]["s1"] == [["x0p0625", .0625], ["x0p25", .25], ["x1", 1.], ["x4", 4.]]
    assert c["selection"]["sdlogj_official_max"] == .1 and c["selection"]["sdlogj_note"] is None
    check = c["run_pair_args_check"]                                             # review M7
    assert check["unexpected"] == {} and set(check["allowed_differences"]) == {
        "jd_cell_estimate", "max_accepted", "max_accepted_per_level"}
    s1, s2, s3 = (json.loads((root/f"selection_{s}.json").read_text()) for s in ("s1", "s2", "s3"))
    assert [x["nu"] for x in s1["candidates"]] == [64., 256., 1024., 4096.]
    assert (s1["chosen"], s1["edge"], s2["chosen"], s2["edge"], s3["chosen"], s3["edge"]) == \
        ("x0p0625", "lower", "x0p5", "lower", "st2", "upper")
    assert [x["nu"] for x in s2["candidates"]] == [32., 64., 128.] and s2["candidates"][1]["reused_from"] == "s1-x0p0625"
    for s in (s1, s2, s3):
        conf = s["configuration"]
        assert (conf["jd_cell_estimate"], conf["max_accepted_per_level"], conf["design"], conf["path"]) == \
            ("certificate_bernstein", [500, 500, 100, 500], "v4", "p3")
        assert all(r["bernstein_violations"] == 0 and r["bernstein_audit"] is None
                   for x in s["candidates"] for r in x["rows"])
        assert s["sdlogj_official_max"] == .1 and s["excluded_by_sdlogj"] == [] and "SDlogJ" in s["rule"]
        assert all("eta = the candidate's" in r["configuration_check"] for x in s["candidates"] for r in x["rows"])
    text = next((root/"launch").glob("run_s1-x0p0625-0901-0902_*.txt")).read_text()
    for piece in ("--jd-cell-estimate certificate_bernstein", "--max-accepted-per-level 500,500,100,500",
                  '"design": "v4"', '"path": "p3"', "--unit=tag-eta4dry-s1-x0p0625-0901-0902"):
        assert piece in text, piece
    config = json.loads(next((root/"runs").glob("s1/x0p0625/0901-0902/config.json")).read_text())["args"]
    assert (config["max_accepted"], config["max_accepted_per_level"], config["level_time_shares"]) == \
        (None, [500, 500, 100, 500], None)
    summary = json.loads((root/"summary.json").read_text())
    e = summary["edges"]
    assert e["still_at_lower_end"] is True and e["still_at_upper_end"] is False and e["nu_lowest_run"] == 32.
    assert e["nu_lowest_eligible"] == 32. and e["excluded_by_sdlogj_below_chosen"] == []
    assert (summary["design"], summary["path"]) == ("v4", "p3") and "未验证" in summary["unverified"]
    selected = json.loads((root/"eta_selected.json").read_text())
    assert (selected["nu"], selected["tau"], selected["alpha"]) == (32., [1., 1., 1., 1.], .01)
    md = (root/"summary.md").read_text()
    assert "仍在下端" in md and "seg4" in md and "seg35" in md and "路径：p3" in md
    readouts = json.loads((root/"readouts_s1.json").read_text())
    assert len(readouts["rows"]) == 12 and readouts["rows"][0]["metrics"]["logj_q01_brain"] == -.4
    assert readouts["rows"][0]["levels"][0]["cap"] == 500 and readouts["rows"][0]["native_accepted"] == 5
    assert "各层读数" in (root/"readouts_s1.md").read_text() and (root/"readouts_s3.json").is_file()
    # restart: nothing launched; the records are recomputed and agree
    again = FakeSystem()
    assert driver4(root, again, centre=centre, pilot=pilot) == 0 and again.units == []
    # restart under another path: path.json differs -> STOP, nothing launched
    other = FakeSystem()
    assert driver4(root, other, centre=centre, pilot=pilot, path="p1") == 8 and other.units == []
    assert "path.json differs" in (root/"sweep.log").read_text()


def test_v4_held_path_starts_nothing_until_the_lead_gives_it(sweep, dry, driver4, pilot_sum, tmp_path, monkeypatch):
    centre = write_centre(sweep, dry, tmp_path/"v3")
    root = tmp_path/"v4"
    good = write_pilot(tmp_path/"good", {"p3": "lean"}, pilot_sum)
    cases = [(write_pilot(tmp_path/"rev", {"p1": "lean", "p3": "review"}, pilot_sum), FakeSystem()),
             (write_pilot(tmp_path/"cur", {}, pilot_sum), FakeSystem()),
             (tmp_path/"missing"/"summary.json", FakeSystem()),
             (good, FakeSystem(pilot=(["tag-pilot-p3-0442-0443.service"], []))),      # the pilot still running
             (good, FakeSystem(pilot=([], [4711])))]
    for pilot, system in cases:
        assert driver4(root, system, centre=centre, pilot=pilot) == 8 and system.units == []
    # the pilot's summarize.py not loadable (contract 7b; review M1)
    real_load = sweep.load_module

    def broken(name, rel):
        if rel == sweep.PILOT_SUMMARIZE:
            raise ImportError("synthetic")
        return real_load(name, rel)
    monkeypatch.setattr(sweep, "load_module", broken)
    assert driver4(root, FakeSystem(), centre=centre, pilot=good) == 8
    monkeypatch.setattr(sweep, "load_module", real_load)
    holds = sorted(root.glob("path_hold_*.json"))
    assert len(holds) == 6 and not (root/"path.json").exists() and not (root/"centre").exists()
    docs = [json.loads(h.read_text())["pilot"] for h in holds]
    assert sorted(d["status"] for d in docs) == ["hold"]*5+["lead"]
    reasons = " ".join(r for d in docs for r in d["reasons"])
    assert "PILOT 仍在运行" in reasons and "tag-pilot-p3-0442-0443.service" in reasons and "4711" in reasons
    assert "载入失败" in reasons
    # the lead keeps the current path explicitly: recorded as an override beside the pilot record
    system = FakeSystem()
    assert driver4(root, system, "--until", "centre", centre=centre, pilot=tmp_path/"cur"/"summary.json",
                   path="current") == 0 and system.units == []
    path = json.loads((root/"path.json").read_text())
    assert (path["path"], path["source"], path["pilot"]["status"], path["run_pair_switches"]) == \
        ("current", "override (--path over --path-from-pilot)", "lead", [])
    centre_doc = json.loads((root/"centre"/"centre.json").read_text())
    assert centre_doc["configuration"]["path"] == "current"
    assert set(centre_doc["run_pair_args_check"]["allowed_differences"]) == set()      # the v3 path itself


RECERT_CONFIG = {"class": "CompiledFallbackCertificateConfig", "delta": 0., "max_depth": 48, "max_boxes": 255,
                 "tile_size": 8192, "fallback_workers": 8}
RECERT_VERSION = "q1-3d-parallel-hoisted-edge-hull-compiled-bernstein-v1"


def after_end(root, stage, label, short, minutes=1):
    """A UTC time with a zone, `minutes` after the run's evaluation ended (done.json; the audit's utc must be later)."""
    done = json.loads((root/"eval"/stage/label/short/"done.json").read_text())["done_utc"]
    moment = datetime.fromisoformat(done.replace("Z", "+00:00"))+timedelta(minutes=minutes)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def recert_doc(run, **change):
    """The output of experiments/audit/recertify.py for `run` (its fields as that tool writes them; the times 20 and 30
    s after the run's evaluation ended)."""
    stage, label, short = run.parts[-3:]
    started, ended = (after_end(run.parents[3], stage, label, short, minutes=m) for m in (1/3, 1/2))
    doc = dict(tool="experiments/audit/recertify.py", schema="bayesreg.audit.recertify.v1",
               script_sha256=hashlib.sha256((ROOT/"experiments"/"audit"/"recertify.py").read_bytes()).hexdigest(),
               code_sha="c"*40, state_hash=json.loads((run/"status.json").read_text())["exported"],
               certificate_config=dict(RECERT_CONFIG), passed=True, status="certified_feasible", min_jd_lower_bound=.02,
               n_uncertain=0, n_infeasible=0, started_utc=started, ended_utc=ended, code_dirty=False,
               run_dir=str(run.resolve()), export_file=dict(file_sha256=hashlib.sha256(
                   (run/"final_native.npz").read_bytes()).hexdigest(), level="native"),
               grid=dict(rebuilt_grid_id="g"*64), certificate_version=RECERT_VERSION, n_prescreen_cells=0)
    doc["certificate_state_hash"] = doc["state_hash"]
    doc.update(change)
    return {k: v for k, v in doc.items() if v is not ...}


def audit_doc(run, count, utc_, record="recert.json", **change):
    status = json.loads((run/"status.json").read_text())
    doc = dict(state_hash=status["exported"], violations=count, certificate_config="CompiledFallbackCertificateConfig",
               recertified_passed=True, min_jd_lower_bound=.02, auditor="lead (test)", utc=utc_,
               recertification_record=record)
    doc.update(change)
    return {k: v for k, v in doc.items() if v is not ...}


def test_v4_bernstein_violation_stops_until_a_bound_audit(sweep, dry, driver4, tmp_path):
    centre = write_centre(sweep, dry, tmp_path/"v3")
    root = tmp_path/"v4"
    name = "s1-x0p25-0903-0904"
    system = FakeSystem(violations={f"tag-eta4dry-{name}": 2})
    assert driver4(root, system, "--until", "s1", centre=centre, path="p2") == 8
    assert system.units[-2:] == [f"tag-eta4dry-{name}", f"tag-eta4dry-eval-{name}"]
    assert "待审核（违反 2）" in (root/"sweep.log").read_text()
    run, audit, recert = root/"runs"/"s1"/"x0p25"/"0903-0904", root/"audit"/f"{name}.json", root/"audit"/"recert.json"
    (root/"audit").mkdir()
    later = after_end(root, "s1", "x0p25", "0903-0904")
    earlier = after_end(root, "s1", "x0p25", "0903-0904", minutes=-10)
    cfg = (lambda **kw: dict(RECERT_CONFIG, **kw))
    # contract 7b (review I3, round 2 R2-1 to R2-3): every item, bound to this run and to the recertification record;
    # anything missing or off does not release
    bad = dict(only_flag=(dict(recertified_passed=True), recert_doc(run)),
               no_auditor=(audit_doc(run, 2, later, auditor=...), recert_doc(run)),
               empty_auditor=(audit_doc(run, 2, later, auditor=" "), recert_doc(run)),
               no_utc=(audit_doc(run, 2, later, utc=...), recert_doc(run)),
               bad_utc=(audit_doc(run, 2, "yesterday"), recert_doc(run)),
               date_only=(audit_doc(run, 2, later[:10]), recert_doc(run)),
               no_zone=(audit_doc(run, 2, later[:-1]), recert_doc(run)),
               before_the_run_ended=(audit_doc(run, 2, earlier), recert_doc(run)),
               other_state=(audit_doc(run, 2, later, state_hash="0"*64), recert_doc(run)),
               other_count=(audit_doc(run, 1, later), recert_doc(run)),
               bool_count=(audit_doc(run, True, later), recert_doc(run)),
               prescreen=(audit_doc(run, 2, later, certificate_config="PrescreenCertificateConfig"), recert_doc(run)),
               other_class=(audit_doc(run, 2, later, certificate_config="CertificateConfig"), recert_doc(run)),
               not_passed=(audit_doc(run, 2, later, recertified_passed=False), recert_doc(run)),
               zero_bound=(audit_doc(run, 2, later, min_jd_lower_bound=0.), recert_doc(run)),
               no_record_key=(audit_doc(run, 2, later, recertification_record=...), recert_doc(run)),
               record_missing=(audit_doc(run, 2, later, record="none.json"), recert_doc(run)),
               record_other_state=(audit_doc(run, 2, later), recert_doc(run, state_hash="0"*64)),
               record_prescreen=(audit_doc(run, 2, later),
                                 recert_doc(run, certificate_config=cfg(**{"class": "PrescreenCertificateConfig"}))),
               record_delta=(audit_doc(run, 2, later), recert_doc(run, certificate_config=cfg(delta=.01))),
               record_depth=(audit_doc(run, 2, later), recert_doc(run, certificate_config=cfg(max_depth=12))),
               record_boxes=(audit_doc(run, 2, later), recert_doc(run, certificate_config={
                   k: v for k, v in RECERT_CONFIG.items() if k != "max_boxes"})),
               record_unknown_field=(audit_doc(run, 2, later), recert_doc(run, certificate_config=cfg(foo=1))),
               record_not_passed=(audit_doc(run, 2, later), recert_doc(run, passed=False)),
               record_other_bound=(audit_doc(run, 2, later), recert_doc(run, min_jd_lower_bound=.03)),
               # round 3 R3-2: written by the repository's experiments/audit/recertify.py, its counts and times
               record_other_tool=(audit_doc(run, 2, later), recert_doc(run, tool="experiments/eta/recertify.py")),
               record_other_script=(audit_doc(run, 2, later), recert_doc(run, script_sha256="0"*64)),
               record_no_code_sha=(audit_doc(run, 2, later), recert_doc(run, code_sha=...)),
               record_uncertain=(audit_doc(run, 2, later), recert_doc(run, n_uncertain=1)),
               record_infeasible_bool=(audit_doc(run, 2, later), recert_doc(run, n_infeasible=False)),
               record_started_before_end=(audit_doc(run, 2, later), recert_doc(run, started_utc=earlier)),
               record_ended_after_audit=(audit_doc(run, 2, later),
                                         recert_doc(run, ended_utc=after_end(root, "s1", "x0p25", "0903-0904", 5))),
               record_no_zone=(audit_doc(run, 2, later), recert_doc(run, ended_utc=later[:-1])),
               # RECERT review I1: bound to this run's directory, exported file and the evaluator's grid
               record_other_run_dir=(audit_doc(run, 2, later), recert_doc(run, run_dir=str(root/"elsewhere"))),
               record_other_file=(audit_doc(run, 2, later), recert_doc(run, export_file=dict(file_sha256="0"*64))),
               record_other_grid=(audit_doc(run, 2, later), recert_doc(run, grid=dict(rebuilt_grid_id="h"*64))),
               # RECERT review S1-S3: a clean commit, internal consistency, the recorded tile size
               record_dirty_without_note=(audit_doc(run, 2, later), recert_doc(run, code_dirty=True)),
               record_dirty_empty_note=(audit_doc(run, 2, later, code_dirty_note=" "), recert_doc(run, code_dirty=True)),
               record_bad_code_sha=(audit_doc(run, 2, later), recert_doc(run, code_sha="not-a-commit")),
               record_schema=(audit_doc(run, 2, later), recert_doc(run, schema="other")),
               record_status=(audit_doc(run, 2, later), recert_doc(run, status="uncertain")),
               record_certificate_hash=(audit_doc(run, 2, later), recert_doc(run, certificate_state_hash="0"*64)),
               record_version=(audit_doc(run, 2, later), recert_doc(run, certificate_version="q1-3d-other")),
               record_prescreen_count=(audit_doc(run, 2, later), recert_doc(run, n_prescreen_cells=None)),
               record_tile=(audit_doc(run, 2, later), recert_doc(run, certificate_config=cfg(tile_size=4096))),
               record_workers=(audit_doc(run, 2, later), recert_doc(run, certificate_config=cfg(fallback_workers=0))),
               # round 3 R3-3: not later than the reading time plus 10 minutes
               future_utc=(audit_doc(run, 2, after_end(root, "s1", "x0p25", "0903-0904", 24*60)), recert_doc(run)),
               no_export_for_an_exported_run=(dict(run=name, no_export=True, state_hash=None, auditor="lead",
                                                   utc=later), recert_doc(run)))
    for case, (doc, record) in bad.items():
        audit.write_text(json.dumps(doc))
        recert.write_text(json.dumps(record))
        again = FakeSystem()
        assert driver4(root, again, "--until", "s1", centre=centre, path="p2") == 8 and again.units == [], case
        assert "not complete" in (root/"sweep.log").read_text().splitlines()[-1], case
    # released: a recertification on a dirty code tree is accepted with the audit's note, which is recorded
    audit.write_text(json.dumps(audit_doc(run, 2, later, code_dirty_note="lead (test): uncommitted docs only")))
    recert.write_text(json.dumps(recert_doc(run, code_dirty=True, certificate_config=cfg(fallback_workers=1))))
    after = FakeSystem()
    assert driver4(root, after, "--until", "s1", centre=centre, path="p2") == 0
    assert len([u for u in after.units if re.fullmatch(r"tag-eta4dry-s1-.*", u)]) == 12-5
    s1 = json.loads((root/"selection_s1.json").read_text())
    rows = {(c["label"], r["pair"]): r for c in s1["candidates"] for r in c["rows"]}
    row = rows[("x0p25", "0903-0904")]
    assert (row["bernstein_violations"], row["bernstein_audit_sha256"], row["bernstein_recertification_sha256"],
            row["bernstein_no_export"]) == (2, sweep.file_sha256(audit), sweep.file_sha256(recert), False)
    assert row["bernstein_code_dirty_note"] == "lead (test): uncommitted docs only"
    assert sum(r["bernstein_violations"] for r in rows.values()) == 2
    readouts = json.loads((root/"readouts_s1.json").read_text())
    assert [(r["bernstein_audit_sha256"], r["bernstein_recertification_sha256"]) for r in readouts["rows"]
            if r["bernstein_violations"]] == [(sweep.file_sha256(audit), sweep.file_sha256(recert))]
    # a run without an exported state is never released by an audit; the evaluator's hash must agree
    exported = json.loads((run/"status.json").read_text())["exported"]
    good = recert_doc(run)
    end = datetime.fromisoformat(json.loads((root/"eval"/"s1"/"x0p25"/"0903-0904"/"done.json").read_text())[
        "done_utc"].replace("Z", "+00:00"))
    assert not sweep.audit_problems(audit_doc(run, 2, later), state_hash=exported, eval_state_hash=None, count=2,
                                    record=good, run_end=end)
    assert sweep.audit_problems(audit_doc(run, 2, later), state_hash=None, eval_state_hash=None, count=2, record=good,
                                run_end=end)
    assert sweep.audit_problems(audit_doc(run, 2, later), state_hash=exported, eval_state_hash="f"*64, count=2,
                                record=good, run_end=end)
    # round 3 R3-1: a run end that cannot be determined never releases
    problems = sweep.audit_problems(audit_doc(run, 2, later), state_hash=exported, eval_state_hash=None, count=2,
                                    record=good, run_end=None)
    assert problems and all("cannot be determined" in p for p in problems)
    assert any("cannot be determined" in p for p in sweep.time_problems(later, None))
    assert sweep.no_export_problems(dict(run=name, no_export=True, state_hash=None, auditor="a", utc=later), name=name,
                                    state_hash=None, run_end=None)
    # round 3 R3-3: the reading time plus 10 minutes is the latest accepted
    now = datetime.fromisoformat(later.replace("Z", "+00:00"))
    assert not sweep.time_problems((now+timedelta(minutes=9)).isoformat(), end, now)
    assert sweep.time_problems((now+timedelta(minutes=11)).isoformat(), end, now)
    # violations: an integer equal to the count, never a bool (True == 1 in Python); null only when not recorded
    assert any("never a bool" in p for p in sweep.audit_problems(
        audit_doc(run, True, later), state_hash=exported, eval_state_hash=None, count=1, record=good, run_end=end))
    assert not sweep.audit_problems(audit_doc(run, None, later), state_hash=exported, eval_state_hash=None, count=None,
                                    record=good, run_end=end)
    # configuration C without the prescreen is the prescreen configuration's parent class
    assert sweep.jd_feasibility.PrescreenCertificateConfig.__mro__[1].__name__ == sweep.RECERT_CLASS


def test_v4_no_export_run_confirmed_and_counted_ineligible(sweep, dry, driver4, tmp_path):
    """Round 2 R2-5: a run that needs the audit but has no exported state is released only by a no-export
    confirmation (run, no_export, state_hash null, auditor, utc with a zone after the run's end); the candidate is then
    ineligible and the sweep goes on."""
    centre = write_centre(sweep, dry, tmp_path/"v3")
    root = tmp_path/"v4"
    name = "s1-x4-0905-0906"
    system = FakeSystem(violations={f"tag-eta4dry-{name}": 1}, no_export={f"tag-eta4dry-{name}"})
    assert driver4(root, system, "--until", "s1", centre=centre, path="p2") == 8
    assert system.units[-1] == f"tag-eta4dry-eval-{name}" and "待审核（违反 1）" in (root/"sweep.log").read_text()
    run, confirm = root/"runs"/"s1"/"x4"/"0905-0906", root/"audit"/f"{name}.json"
    (root/"audit").mkdir()
    later = after_end(root, "s1", "x4", "0905-0906")
    early = after_end(root, "s1", "x4", "0905-0906", minutes=-5)
    ok = dict(run=name, no_export=True, state_hash=None, auditor="lead (test)", utc=later)
    audit = dict(state_hash=None, violations=1, certificate_config="CompiledFallbackCertificateConfig",
                 recertified_passed=True, min_jd_lower_bound=.02, auditor="lead", utc=later,
                 recertification_record="recert.json")
    for case, doc in dict(audit=audit, other_run=dict(ok, run="s1-x4-0901-0902"), not_true=dict(ok, no_export="yes"),
                          with_state=dict(ok, state_hash="0"*64),
                          no_auditor={k: v for k, v in ok.items() if k != "auditor"},
                          no_zone=dict(ok, utc=later[:-1]), early=dict(ok, utc=early)).items():
        confirm.write_text(json.dumps(doc))
        again = FakeSystem()
        assert driver4(root, again, "--until", "s1", centre=centre, path="p2") == 8 and again.units == [], case
    confirm.write_text(json.dumps(ok))
    assert driver4(root, FakeSystem(), "--until", "s1", centre=centre, path="p2") == 0
    s1 = json.loads((root/"selection_s1.json").read_text())
    x4 = next(c for c in s1["candidates"] if c["label"] == "x4")
    row = next(r for r in x4["rows"] if r["pair"] == "0905-0906")
    assert row["bernstein_no_export"] is True and row["bernstein_audit_sha256"] == sweep.file_sha256(confirm)
    assert x4["eligible"] is False and any("不合格（无导出）" in r for r in x4["ineligible"])
    assert s1["chosen"] != "x4"


def test_v4_sdlogj_must_be_finite_on_every_pair(sweep, dry, driver4, tmp_path):
    """Contract 7b (round 2 R2-4): the condition needs the three official SDlogJ finite, not only a finite mean."""
    centre = write_centre(sweep, dry, tmp_path/"v3", nu=16.)
    root = tmp_path/"v4"
    assert driver4(root, FakeSystem(sdlogj_nan={"eta_s1_x1_0903-0904"}), "--until", "s1", centre=centre,
                   path="current") == 0
    s1 = json.loads((root/"selection_s1.json").read_text())
    x1 = next(c for c in s1["candidates"] if c["label"] == "x1")
    assert x1["eligible"] is False and x1["excluded_by_sdlogj"] is True
    assert any("not finite on ['0903-0904']" in r for r in x1["ineligible"]) and s1["chosen"] != "x1"


def test_v4_model_code_change_listed_before_it_is_accepted(sweep, tmp_path):
    """Round 2 R2-7: without --accept-code-change a non-empty list of changed model files goes to
    centre/model_code_changed_pending_<UTC>.json and the sweep stops before centre.json; with it the change is
    accepted; no change needs nothing."""
    base = ["--root", str(tmp_path/"v4"), "--design", "v4", "--path", "p3", "--path-note", "n", "--until", "centre"]
    s = sweep.Sweep(sweep.parse_args(base), FakeSystem())
    s.root.mkdir()
    s.code_sha = "c"*40
    with pytest.raises(sweep.Stop, match="reruns with --accept-code-change"):
        s.model_code_gate(["bayesreg/model/x.py", "jd_curl/y.py"], "68721d1", dict(unexpected={}))
    pending = list((s.root/"centre").glob("model_code_changed_pending_*.json"))
    assert len(pending) == 1
    doc = json.loads(pending[0].read_text())
    assert doc["model_code_changed"] == ["bayesreg/model/x.py", "jd_curl/y.py"] and doc["centre_code_sha"] == "68721d1"
    assert not (s.root/"centre"/"centre.json").exists()
    assert s.model_code_gate([], "68721d1") is False
    accepted = sweep.Sweep(sweep.parse_args(base+["--accept-code-change"]), FakeSystem())
    assert accepted.model_code_gate(["bayesreg/model/x.py"], "68721d1") is True


def test_v4_hard_constraint_1_stops_and_v3_does_not(sweep, dry, driver4, driver, tmp_path):
    centre = write_centre(sweep, dry, tmp_path/"v3")
    root = tmp_path/"v4"
    system = FakeSystem(cert_fail={"eta_s1_x0p25_0901-0902"})
    assert driver4(root, system, "--until", "s1", centre=centre, path="current") == 8
    assert system.units[-1] == "tag-eta4dry-eval-s1-x0p25-0901-0902"
    assert "hard constraint 1" in (root/"sweep.log").read_text().splitlines()[-1]
    again = FakeSystem()                                                         # a restart stops again, starts nothing
    assert driver4(root, again, "--until", "s1", centre=centre, path="current") == 8 and again.units == []
    assert "hard constraint 1" in (root/"sweep.log").read_text().splitlines()[-1]
    # v3: unchanged -- the candidate is only ineligible
    v3, system = tmp_path/"v3run", FakeSystem(cert_fail={"eta_s1_x1_0901-0902"})
    assert driver(v3, system, "--until", "s1") == 0
    s1 = json.loads((v3/"selection_s1.json").read_text())
    assert s1["candidates"][1]["eligible"] is False and "certificate not passed" in s1["candidates"][1]["ineligible"][0]


def test_v4_sdlogj_condition_and_edges(sweep, dry, driver4, tmp_path):
    centre = write_centre(sweep, dry, tmp_path/"v3", nu=16.)
    root = tmp_path/"v4"
    # s1: nu 1, 4, 16, 64; the response is best near 16 but the two lowest exceed .10
    system = FakeSystem(sdlogj=lambda nu, tau: .12 if nu < 8 else .05)
    assert driver4(root, system, centre=centre, path="current") == 0
    s1 = json.loads((root/"selection_s1.json").read_text())
    assert s1["excluded_by_sdlogj"] == ["x0p0625", "x0p25"] and s1["chosen"] == "x1" and "SDlogJ" in s1["rule"]
    first = s1["candidates"][0]
    assert first["eligible"] is False and first["mean_sdlogj_official"] == pytest.approx(.12)
    assert "官方 SDlogJ 平均" in (root/"selection_s1.md").read_text()
    e = json.loads((root/"summary.json").read_text())["edges"]
    assert [x["label"] for x in e["excluded_by_sdlogj_below_chosen"]] == ["x0p0625", "x0p25"]
    assert e["nu_lowest_run"] == 1. and e["nu_lowest_eligible"] == 8. and e["still_at_lower_end"] is False
    assert "选中 ν 下侧 x0p0625" in (root/"summary.md").read_text()
    centre_doc = json.loads((root/"centre"/"centre.json").read_text())
    assert centre_doc["selection"]["sdlogj_official_max"] == .1
    # the condition belongs to the root: a restart that switches it off STOPs at the centre record
    other = FakeSystem()
    assert driver4(root, other, "--sdlogj-official-max", "none", "--sdlogj-note", "lead (test)", centre=centre,
                   path="current") == 8 and other.units == []
    # switched off explicitly, with the note recorded
    off = tmp_path/"off"
    assert driver4(off, FakeSystem(sdlogj=lambda nu, tau: .12), "--until", "s1", "--sdlogj-official-max", "none",
                   "--sdlogj-note", "BZ (test): off", centre=centre, path="current") == 0
    s1 = json.loads((off/"selection_s1.json").read_text())
    assert s1["sdlogj_official_max"] is None and s1["excluded_by_sdlogj"] == [] and "SDlogJ" not in s1["rule"]
    assert json.loads((off/"centre"/"centre.json").read_text())["selection"]["sdlogj_note"] == "BZ (test): off"


def test_v4_after_run_checks(sweep, dry, driver4, tmp_path):
    centre = write_centre(sweep, dry, tmp_path/"v3")
    # config.json eta not the candidate's (review M6)
    root = tmp_path/"a"
    system = FakeSystem(eta_shift={"tag-eta4dry-s1-x4-0905-0906"})
    assert driver4(root, system, "--until", "s1", centre=centre, path="current") == 8
    assert "is not the candidate's" in (root/"sweep.log").read_text().splitlines()[-1]
    # structure_repr other than sqrt_prob, sigma_domain other than brain, max_accepted other than 500
    for key, value, word in (("structure_repr", "sdf", "structure_repr"), ("sigma_domain", "support", "sigma_domain"),
                             ("max_accepted", 300, "max_accepted")):
        root = tmp_path/key
        assert driver4(root, FakeSystem(), "--until", "s1", centre=centre, path="current") == 0
        config = root/"runs"/"s1"/"x1"/"0903-0904"/"config.json"
        doc = json.loads(config.read_text())
        doc["args"][key] = value
        config.write_text(json.dumps(doc))
        again = FakeSystem()
        assert driver4(root, again, "--until", "s1", centre=centre, path="current") == 8 and again.units == []
        assert word in (root/"sweep.log").read_text().splitlines()[-1], key


def test_v4_centre_must_fit(sweep, dry, driver4, tmp_path):
    cases = dict(beta=dict(logvol_closed27_share=3.), wall=dict(wall_seconds=900.))
    for name, change in cases.items():
        centre = write_centre(sweep, dry, tmp_path/name, **change)
        system = FakeSystem()
        assert driver4(tmp_path/f"r_{name}", system, centre=centre, path="p3") == 8 and system.units == []
        assert "beyond the path keys" in (tmp_path/f"r_{name}"/"sweep.log").read_text()
    # the path keys may differ (the v3 centre ran the current path)
    centre = write_centre(sweep, dry, tmp_path/"cell", jd_cell_estimate="gauss27+corners")
    assert driver4(tmp_path/"r_cell", FakeSystem(), "--until", "centre", centre=centre, path="p3") == 0
    centre = write_centre(sweep, dry, tmp_path/"alpha", alpha=.02)
    assert driver4(tmp_path/"r_alpha", FakeSystem(), centre=centre, path="p3") == 8
    assert "alpha_nug" in (tmp_path/"r_alpha"/"sweep.log").read_text()
    centre = write_centre(sweep, dry, tmp_path/"pairs", pairs=["OASIS_OAS1_0901_MR1__OASIS_OAS1_0902_MR1"])
    assert driver4(tmp_path/"r_pairs", FakeSystem(), centre=centre, path="p3") == 8
    assert "does not fit the centre" in (tmp_path/"r_pairs"/"sweep.log").read_text()
    # the run_pair args against the v3 run's (review M7): a key that changed, or a new key, STOPs; nothing written
    for name, change in (("threads", dict(cpu_threads=4)), ("newkey", dict(inner="pg")),
                         ("tissue", dict(tissue_root="/elsewhere"))):
        centre = write_centre(sweep, dry, tmp_path/f"a_{name}", args_change=change)
        system, root = FakeSystem(), tmp_path/f"r_a_{name}"
        assert driver4(root, system, "--until", "centre", centre=centre, path="p3") == 8, name
        assert "beyond the path, the code trees" in (root/"sweep.log").read_text() and not (root/"centre").exists()
    # SWLI's keys absent in the v3 record (as in eta_sweep_v3): allowed as brain / null, recorded
    centre = write_centre(sweep, dry, tmp_path/"a_swli", args_change=dict(sigma_domain=..., support_threshold=...))
    assert driver4(tmp_path/"r_a_swli", FakeSystem(), "--until", "centre", centre=centre, path="p3") == 0
    allowed = json.loads((tmp_path/"r_a_swli"/"centre"/"centre.json").read_text())["run_pair_args_check"][
        "allowed_differences"]
    assert allowed["sigma_domain"] == ["<absent>", "brain"] and allowed["support_threshold"] == ["<absent>", None]
    # a dry run never reads a real result as its centre; a real v4 root never lies under v1..v3
    assert driver4(tmp_path/"r_real", FakeSystem(), centre=sweep.V3_ETA, path="p3") == 8
    assert "a real result" in (tmp_path/"r_real"/"sweep.log").read_text()
    assert sweep.main(["--root", str(sweep.V3_ROOT/"v4"), "--design", "v4", "--path", "p3", "--path-note", "n"]
                      + DEADLINE, system=FakeSystem()) == 2


def test_args_differences(sweep):
    v3 = dict(pair="a", out="/r3/runs/x", eta_json="/r3/e.json", jd_cell_estimate="certificate", max_accepted=500,
              allowlist="/r3/code/configs/bayesreg/dev_subjects.json", wall_seconds=900.)
    ours = dict(pair="b", out="/r4/runs/y", eta_json="/r4/e.json", jd_cell_estimate="certificate_bernstein",
                max_accepted=None, max_accepted_per_level=[500, 500, 100, 500], level_time_shares=None,
                allowlist="/r4/code/configs/bayesreg/dev_subjects.json", wall_seconds=900., sigma_domain="brain",
                support_threshold=None)
    path = dict(jd_cell_estimate="certificate_bernstein", max_accepted=None, max_accepted_per_level=[500, 500, 100, 500],
                level_time_shares=None)
    unexpected, allowed = sweep.args_differences(v3, ours, v3_code=Path("/r3/code"), our_code=Path("/r4/code"),
                                                 expected_path=path)
    assert unexpected == {} and set(allowed) == {"jd_cell_estimate", "max_accepted", "max_accepted_per_level",
                                                 "allowlist", "sigma_domain", "support_threshold"}
    assert allowed["allowlist"] == ["<code>/configs/bayesreg/dev_subjects.json"]*2
    for key, value in (("wall_seconds", 60.), ("allowlist", "/r4/elsewhere.json"), ("sigma_domain", "support"),
                       ("max_accepted_per_level", [500]*4), ("new_key", 1)):
        unexpected, _ = sweep.args_differences(v3, dict(ours, **{key: value}), v3_code=Path("/r3/code"),
                                               our_code=Path("/r4/code"), expected_path=path)
        assert list(unexpected) == [key], key


def test_v4_gate_preflight_extra_and_deadline(sweep, dry, driver4, tmp_path):
    centre = write_centre(sweep, dry, tmp_path/"v3")
    # another driver process (M2V1, the pilot, another sweep): STOP before any unit
    system = FakeSystem(drivers=[4242])
    assert driver4(tmp_path/"a", system, centre=centre, path="current") == 8 and system.units == []
    assert "driver processes running: [4242]" in (tmp_path/"a"/"sweep.log").read_text()
    # --extra --max-accepted with a capped path: refused at the path (C13-4a), nothing started
    system = FakeSystem()
    assert driver4(tmp_path/"b", system, "--extra", "--max-accepted 2", centre=centre, path="p1") == 8
    assert system.units == [] and "--extra names ['--max-accepted']" in (tmp_path/"b"/"sweep.log").read_text()

    class Refusing(FakeSystem):
        def check_run_args(self, code, args):
            return False, "usage: refused", None
    system = Refusing()
    assert driver4(tmp_path/"c", system, centre=centre, path="p1") == 8 and system.units == []
    assert "run_pair parse_args refused" in (tmp_path/"c"/"sweep.log").read_text()
    # the deadline: already past -> nothing; reached between runs -> STOP before the next run unit (its evaluation done);
    # a restart with a later deadline resumes
    t0 = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
    system = FakeSystem(clock=t0+timedelta(minutes=1))
    assert driver4(tmp_path/"d", system, *DEADLINE, centre=centre, path="current") == 8 and system.units == []
    assert "already passed" in (tmp_path/"d"/"sweep.log").read_text()
    system = FakeSystem(clock=t0-timedelta(minutes=30), tick=timedelta(minutes=20))
    assert driver4(tmp_path/"e", system, "--until", "s1", *DEADLINE, centre=centre, path="current") == 8
    assert system.units == ["tag-eta4dry-s1-x0p0625-0901-0902", "tag-eta4dry-eval-s1-x0p0625-0901-0902",
                            "tag-eta4dry-s1-x0p0625-0903-0904", "tag-eta4dry-eval-s1-x0p0625-0903-0904"]
    assert "--no-start-after 2026-11-02T12:00:00+00:00 reached" in (tmp_path/"e"/"sweep.log").read_text()
    later = FakeSystem(clock=t0)
    assert driver4(tmp_path/"e", later, "--until", "s1", "--no-start-after", "2026-11-02T08:00-06:00", centre=centre,
                   path="current") == 0
    assert later.units[0] == "tag-eta4dry-s1-x0p0625-0905-0906" and len(later.units) == 2*(12-2)


def test_check_v3_records_on_a_dry_v3_root(sweep, dry, driver, tmp_path):
    """experiments/eta/check_v3_records.py (the v3-unchanged evidence, run on the real eta_sweep_v3 outside the tests):
    on a dry v3 root made by this sweep it rebuilds every probe, run and evaluation record through the real
    probe_phase / one_run / evaluate / launch byte for byte; with per-level caps added none is identical."""
    check = load("check_v3_records", "experiments/eta/check_v3_records.py")
    root = tmp_path/"v3"
    assert driver(root, FakeSystem()) == 0
    args = ["--dry-run", "--data-root", str(dry/"data"), "--allowlist", str(dry/"dev_subjects.json"), "--tissue-root",
            str(dry/"seg"), "--structure-root", str(dry/"structure"), "--segmenter-registry",
            str(dry/"registry"/"segmenters.json"), "--wall-seconds", "60"]
    report = check.check(root, tmp_path/"scratch", args)
    assert (report["records"], report["rebuilt"], report["identical"], report["different"]) == (45, 45, 45, [])
    mutated = check.check(root, tmp_path/"scratch2", args+["--max-accepted-per-level", "500,500,100,500"])
    assert mutated["identical"] == 0 and len(mutated["different"]) == 45


def test_driver_processes_on_a_fake_proc(sweep, tmp_path):
    proc = tmp_path/"proc"

    def make(pid, ppid, argv):
        (proc/str(pid)).mkdir(parents=True)
        (proc/str(pid)/"cmdline").write_bytes(b"\0".join(a.encode() for a in argv)+b"\0")
        (proc/str(pid)/"stat").write_text(f"{pid} (x) S {ppid} 0 0")
    py = "/home/rmcfz/miniconda3/envs/sci310/bin/python"
    make(1, 0, ["/sbin/init"])
    make(10, 1, [py, "-B", "experiments/eta/sweep.py", "--design", "v4"])               # this process
    make(11, 1, [py, "-B", "experiments/pilot/run_pilot.py"])                           # a pilot driver
    make(12, 1, ["python3.10", "/abs/experiments/m2v1/run_batch.py"])                  # an M2V1 driver
    make(13, 1, ["/bin/bash", "-c", "nohup python experiments/eta/sweep.py"])           # a shell, not python
    make(14, 1, [py, "-B", "experiments/eta/sweep.py.bak"])                            # not ending in a script
    assert sweep.driver_processes(proc, 10) == [11, 12]
