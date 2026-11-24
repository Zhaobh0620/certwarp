"""The eta probe and sweep with --sigma-domain (work package SWLI step 2; contract C04-6 gate 11 (probe part), C05-7a
appended note; review SWLI important 2).  Synthetic files only (experiments/eta/dry_inputs.py), no unit, no data/.

- probe.py: under support_shared nu* differs from brain's and equals an independent computation (v by the threshold
  rule written out at the level-0 seed, the level-start sigma by the C11-8 literal sums, the data gradient
  sum_c (q v_c / (omega0 sigma_c^2)) rho_c grad G_c at u_1); the configuration and the records name the domain;
- brain records are the code before SWLI's, key for key: probe.configuration and the sweep's probe_configuration,
  configuration, expected_run_args and model_args equal those of experiments/eta/{probe,sweep}.py at fd3bbcf (git show);
- sweep.py: option checks; the configuration, the probe / run_pair arguments and the run checks carry sigma_domain and
  theta; a root started under support_shared STOPs on a restart under brain or another theta (fake machine);
- slow (-m slow): the dry run in process (InlineSystem, real probe / run_pair / eval) under support_shared to s1.
"""
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from .test_eta_sweep_tools import (PAIR1, ROOT, STAGE, FakeSystem, InlineSystem, _Quiet, driver, dry,  # noqa: F401
                                   dry_common, final_model_args, load, probe_module, sweep)
from .test_sigma_domain import _literal_start

REF = "fd3bbcf"                     # main before SWLI's step 2 (the probe and sweep without --sigma-domain)
SHARED = ["--sigma-domain", "support_shared"]


def old_module(name, rel, tmp_path):
    """experiments/eta/<rel> as it was at REF (git show), loaded under `name`; skipped without git."""
    try:
        text = subprocess.run(["git", "show", f"{REF}:{rel}"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git show {REF}:{rel} unavailable ({exc})")
    path = tmp_path/f"{name}.py"
    path.write_text(text, encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- probe ---------------------------------------------------------------------------------------------------------

def test_probe_nu_star_under_support_shared(dry, tmp_path, probe_module):
    probe = probe_module
    from bayesreg.cli import run_pair as rp
    from bayesreg.io import load_pair
    from bayesreg.model import channel_point, evaluate, gradient
    from bayesreg.model.support import node_weights_masked
    from bayesreg.seg.structure_layer import StructureSource
    from bayesreg.solve import eta_temporary
    from bayesreg.solve.multilevel import _template
    records = {}
    for name, extra in (("brain", []), ("shared", SHARED)):
        out = tmp_path/name
        assert probe.main(["--pair", f"{STAGE}/{PAIR1}", "--out", str(out), *final_model_args(dry), *extra,
                           *dry_common(dry)]) == 0
        records[name] = json.loads((out/"probe.json").read_text())
    brain, shared = records["brain"], records["shared"]
    assert "sigma_domain" not in brain["configuration"]
    assert shared["configuration"] == dict(brain["configuration"], sigma_domain="support_shared", support_threshold=.05)
    assert shared["nu_star"] != brain["nu_star"]
    status = json.loads((tmp_path/"shared"/"status.json").read_text())
    assert status["configuration"] == shared["configuration"]
    config = json.loads((tmp_path/"shared"/"config.json").read_text())
    assert config["solver"]["sigma_domain"] == "support_shared" and config["sigma"]["sigma_domain"] == "support_shared"
    prov = json.loads((tmp_path/"shared"/"provenance.json").read_text())
    assert prov["switches"]["sigma_domain"] == "support_shared" and prov["switches"]["support_threshold"] == .05
    assert "sigma_domain" not in json.loads((tmp_path/"brain"/"provenance.json").read_text())["switches"]
    # independent nu*: the level-0 template at u = 0 (the seed), v by the rule, sigma by the literal sums, then the data
    # gradient at u_1 written out
    pair = load_pair(rp.resolve_pair(dry/"data", f"{STAGE}/{PAIR1}"), allowlist=str(dry/"dev_subjects.json"))
    source = StructureSource(root=dry/"structure", registry_path=dry/"registry"/"segmenters.json",
                             allowlist_path=dry/"dev_subjects.json", models_root=tmp_path/"none",
                             image_root=tmp_path/"none", allow_staging=True)
    inputs, pt_f, mask_f, _ = rp.prepare_channels(pair, "structure", str(dry/"seg"), _Quiet(), structure=source)
    eta = eta_temporary(pt_f, mask_f, pair.grid)
    level0 = inputs[0]
    host = _template(level0, eta, pair.grid.omega, 8192, level0.fixed.shape[0], 1, "gauss27+closed27",
                     logvol_closed27_share=3., closed27_weights="simpson")
    zero = np.zeros(level0.grid.shape+(3,))
    rho0 = channel_point(host, zero).rho
    t = np.sqrt(np.float64(.05))
    v = np.stack([((host.fixed_masked[c] > t) | (host.fixed_masked[c]+rho0[c] > t)).astype(np.uint8)
                  for c in range(1, host.n_channels)])
    sigma = _literal_start(host, zero, v, "support_shared")
    assert sigma == shared["sigma2_used"]
    internal = [s if s is not None else next(x for x in sigma[1:] if x is not None) for s in sigma]
    u1 = np.load(tmp_path/"shared"/"final_level0.npz")["u"]
    state = host.replace(sigma2=np.array(internal), sigma_domain="support_shared", support_threshold=.05, support_v=v)
    point = channel_point(state, u1)
    q = node_weights_masked(state)
    total = np.zeros((state.node_index.size, 3))
    for c in range(state.n_channels):
        weight = q*(1. if c == 0 else v[c-1])/(state.omega0*internal[c])
        total += (weight*point.rho[c])[:, None]*point.grad[c]
    full = np.zeros((int(np.prod(level0.grid.shape)), 3))
    full[state.node_index] = total
    data = np.linalg.norm(full.reshape(level0.grid.shape+(3,))[1:-1, 1:-1, 1:-1])
    unit = state.replace(nu=1.0)
    curl = np.linalg.norm(gradient(unit, evaluate(unit, u1), blocks=("curl",)))
    assert shared["nu_star"] == pytest.approx(data/curl, rel=1e-10)


def test_probe_options(probe_module):
    probe = probe_module
    base = ["--pair", "a/b", "--out", "x"]
    args = probe.parse_args([*base, "--channel-source", "tissue", *SHARED])
    assert (args.sigma_domain, args.support_threshold) == ("support_shared", .05)
    assert probe.configuration(args)["sigma_domain"] == "support_shared"
    brain = probe.parse_args([*base, "--channel-source", "tissue"])
    assert (brain.sigma_domain, brain.support_threshold) == ("brain", None)
    assert "sigma_domain" not in probe.configuration(brain)
    for bad in (["--channel-source", "none", *SHARED], ["--channel-source", "tissue", "--support-threshold", ".05"],
                ["--channel-source", "tissue", *SHARED, "--support-threshold", "1"]):
        with pytest.raises(SystemExit):
            probe.parse_args([*base, *bad])


# ---- brain records are those before SWLI ---------------------------------------------------------------------------

BRAIN_ARGS = ([], ["--channel-source", "tissue"], ["--channel-source", "none", "--logvol-points", "gauss27+corners",
                                                   "--jd-cell-estimate", "gauss27+corners"],
              ["--logvol-closed27-share", "3", "--metric-jd-scaling", "off"])


def test_brain_records_equal_the_code_before(sweep, probe_module, tmp_path):
    old_sweep = old_module("eta_sweep_old", "experiments/eta/sweep.py", tmp_path)
    old_probe = old_module("eta_probe_old", "experiments/eta/probe.py", tmp_path)
    for extra in BRAIN_ARGS:
        for dry_run in ([], ["--dry-run", "--segmenter-registry", "/r/segmenters.json"]):
            argv = [*extra, *dry_run]
            new, old = sweep.Sweep(sweep.parse_args(argv), FakeSystem()), \
                old_sweep.Sweep(old_sweep.parse_args(argv), FakeSystem())
            for name in ("probe_configuration", "configuration", "expected_run_args", "model_args"):
                got, want = getattr(new, name)(), getattr(old, name)()
                assert json.dumps(got, sort_keys=True, default=str) == json.dumps(want, sort_keys=True, default=str), \
                    (name, extra)
        probe_argv = ["--pair", "a/b", "--out", "x", "--channel-source", "structure", *extra]
        a, b = probe_module.parse_args(probe_argv), old_probe.parse_args(probe_argv)
        assert json.dumps(probe_module.configuration(a)) == json.dumps(old_probe.configuration(b)), extra


# ---- sweep ---------------------------------------------------------------------------------------------------------

def test_sweep_options_and_records(sweep):
    from bayesreg.model.support import theta_lower_bound
    from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping
    args = sweep.parse_args(SHARED)
    assert (args.sigma_domain, args.support_threshold) == ("support_shared", .05)
    brain = sweep.parse_args([])
    assert (brain.sigma_domain, brain.support_threshold) == ("brain", None)
    k_s = load_grouping(DEFAULT_GROUPS_PATH, None).k
    for bad in (["--support-threshold", ".05"], ["--channel-source", "none", *SHARED], [*SHARED, "--support-threshold",
                                                                                         "0"],
                [*SHARED, "--support-threshold", "1.5"], [*SHARED, "--support-threshold", "nan"],
                [*SHARED, "--support-threshold", repr(theta_lower_bound(k_s))],
                ["--channel-source", "tissue", *SHARED, "--support-threshold", repr(theta_lower_bound(1))],
                ["--extra", "--sigma-domain support"], ["--extra", "--support-thr .1"]):
        with pytest.raises(SystemExit):
            sweep.parse_args(bad)
    run = sweep.Sweep(sweep.parse_args([*SHARED, "--support-threshold", ".1"]), FakeSystem())
    conf = run.probe_configuration()
    assert (conf["sigma_domain"], conf["support_threshold"]) == ("support_shared", .1)
    assert run.configuration()["sigma_domain"] == "support_shared"
    model = run.model_args()
    assert model[model.index("--sigma-domain")+1] == "support_shared" and \
        model[model.index("--support-threshold")+1] == "0.1"
    expected = run.expected_run_args()
    assert (expected["sigma_domain"], expected["support_threshold"]) == ("support_shared", .1)
    assert "sigma_domain" not in sweep.Sweep(brain, FakeSystem()).expected_run_args()
    assert "sigma_domain support_shared（θ 0.1）" in sweep.configuration_text(run.configuration())
    # the code tree check asks the run_pair / probe of the tree for the switches (this tree has them)
    run.code = ROOT
    run.check_code_tree()


def test_sweep_end_to_end_support_shared_and_restarts(driver, tmp_path):
    root = tmp_path/"sweep"
    system = FakeSystem()
    assert driver(root, system, *SHARED, "--until", "s1") == 0
    centre = json.loads((root/"probe"/"nu_star.json").read_text())
    assert centre["configuration"]["sigma_domain"] == "support_shared"
    assert centre["configuration"]["support_threshold"] == .05
    s1 = json.loads((root/"selection_s1.json").read_text())
    assert s1["configuration"]["sigma_domain"] == "support_shared"
    assert all(r["configuration_check"].startswith("config.json args =") for c in s1["candidates"] for r in c["rows"])
    for kind in ("probe_0901-0902_*.txt", "run_s1-x1-0901-0902_*.txt"):
        text = next((root/"launch").glob(kind)).read_text()
        assert "--sigma-domain support_shared" in text and "--support-threshold 0.05" in text, kind
    assert "sigma_domain support_shared" in (root/"selection_s1.md").read_text()
    again = FakeSystem()
    assert driver(root, again, *SHARED, "--until", "s1") == 0 and again.units == []
    for other in ([], [*SHARED, "--support-threshold", ".1"], ["--sigma-domain", "support"]):
        system = FakeSystem()
        assert driver(root, system, *other, "--until", "s1") == 8, other
        assert system.units == []
    assert (root/"sweep.log").read_text().count("differs from this sweep's") >= 3


@pytest.mark.slow
def test_dry_run_support_shared_in_process(sweep, dry, tmp_path, monkeypatch):
    """The dry run in process (real probe, run_pair and bayesreg.eval on the synthetic structure layer) under
    support_shared, to the end of s1."""
    def prepare(self):
        self.code, self.code_sha = ROOT, "inline"
        self.allowlist = self.allowlist or str(dry/"dev_subjects.json")
    monkeypatch.setattr(sweep.Sweep, "prepare_code", prepare)
    monkeypatch.setattr(sweep.Sweep, "write_env", lambda self: None)
    root = tmp_path/"sweep"
    args = ["--root", str(root), "--dry-run", "--data-root", str(dry/"data"), "--allowlist", str(dry/"dev_subjects.json"),
            "--tissue-root", str(dry/"seg"), "--structure-root", str(dry/"structure"), "--segmenter-registry",
            str(dry/"registry"/"segmenters.json"), "--wall-seconds", "120", "--extra", "--max-accepted 2", *SHARED,
            "--until", "s1"]
    system = InlineSystem()
    assert sweep.main(args, system=system) == 0, (root/"sweep.log").read_text()[-3000:]
    assert len([u for u in system.units if "-probe-" in u]) == 3
    assert len([u for u in system.units if re.fullmatch(r"tag-etadry-s1-.*", u)]) == 9
    for probe in (root/"probe").iterdir():
        if probe.is_dir():
            record = json.loads((probe/"probe.json").read_text())
            assert record["configuration"]["sigma_domain"] == "support_shared"
            assert json.loads((probe/"config.json").read_text())["solver"]["sigma_domain"] == "support_shared"
    for config in (root/"runs").glob("*/*/*/config.json"):
        a = json.loads(config.read_text())["args"]
        assert (a["sigma_domain"], a["support_threshold"]) == ("support_shared", .05)
        hyper = json.loads((config.parent/"hyper.json").read_text())
        assert hyper["algorithm"]["sigma_domain"] == "support_shared" and len(hyper["support_levels"]) >= 1
    selection = json.loads((root/"selection_s1.json").read_text())
    assert selection["configuration"]["sigma_domain"] == "support_shared"
    again = InlineSystem()
    assert sweep.main(args, system=again) == 0 and again.units == []
