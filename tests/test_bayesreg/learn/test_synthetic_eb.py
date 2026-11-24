"""One EB round on the real path (solver MAP -> C14 perturbations with real certificates -> C15 statistics ->
P2 update) for a synthetic population with known eta (see _synthetic.py), compared with the linearized oracle.

Gate (written before the run): with n_T pairs, the pooled denominators are sums of Gaussian quadratic forms, so
their relative standard deviation is rse = sqrt(n_T Var(u^T A u)) / (n_T tr(A Sigma)) (oracle).  The one-step
estimate from eta_true must lie within 4 rse + 5 % of the oracle EB point (nu_inf, tau_k,inf^2) -- the 5 %
covering the linearization (log J vs div u), the MAP tolerance and the S = 16 trace noise.  The direction of the
deviation from the truth is recorded: B12 predicts nu_EB > nu_true (the block-wise (N) counts 3 N_int curl
degrees of freedom where the joint prior has fewer); for tau the oracle itself says which way.

`multilevel_map` is checked against a stubbed solve_multilevel (fast).  Two slow tests run only with BAYESREG_SLOW=1
(skipped by default to keep tests/test_bayesreg < 3 min and the machine free for timed runs):
  test_multilevel_map_real_four_levels      four-level solve of the solver tests' 26^3 problem + S = 16 samples;
                                            ~1.5-2 min with the code defaults (one run, see the test);
  test_three_rounds_from_an_off_truth_start 3 rounds x 3 pairs on the 8^3 population; ~1 min (one probe run,
                                            2026-10-30: 56 s).
"""
import json
import os

import numpy as np
import pytest

from bayesreg.learn import EB_ROUND_CAP, EBConfig, EBPair, LAPLACE_ENDINGS, multilevel_map, pair_statistics
from bayesreg.learn import population as pop
from bayesreg.learn import run_population_eb
from bayesreg.solve import Eta, FixedSigma, NumpyBackend, SolverConfig
from bayesreg.solve.multilevel import RunResult
from bayesreg.solve.outer import Commit

from . import _synthetic as syn

N_PAIRS = 3


def test_one_round_matches_the_linearized_oracle(tmp_path, record_property):
    oracle = syn.oracle()
    data = {f"OASIS_OAS1_09{k:02d}_MR1__OASIS_OAS1_09{k+50:02d}_MR1": p
            for k, p in enumerate(syn.make_pairs(N_PAIRS))}
    pairs = tuple(EBPair(*pid.split("__"), "A", {"check": "synthetic"}) for pid in data)
    backend = NumpyBackend()
    eta0 = syn.eta_true()
    result = run_population_eb(pairs, lambda pair, eta, *, round_index: syn.native_map(data[pair.pair_id], eta,
                                                                                       backend=backend),
                               eta0=eta0, config=EBConfig(max_rounds=1, eb_count="dim"), run_id="synthetic_eb",
                               out_dir=tmp_path, backend=backend)
    assert result.status == EB_ROUND_CAP and len(result.rounds) == 1     # one round requested (not a C15 run)
    rnd = result.rounds[0]
    for row in rnd["pairs"]:
        assert row["map_status"] in LAPLACE_ENDINGS and row["excluded"] is None
        st = row["statistics"]
        assert st["S_conv"] == 16 and st["S_pass"] == 16 and st["pass_rate"] == 1.
        assert st["n_curl"] == st["n_free"] == oracle["n_free"]
    eta = result.eta
    # P2: alpha is the nugget of the new tau; tau_0 (the largest tau) fixed, so alpha does not move (bitwise)
    assert eta.alpha == eta0.alpha and eta.tau2[0] == eta0.tau2[0]
    assert eta.alpha == pop.alpha_nugget(eta.tau2)                               # the C05-6 rule itself
    assert rnd["update"]["learned"] == ["nu", "tau_1", "tau_2"]

    n = N_PAIRS
    rse_nu = np.sqrt(n*oracle["curl_var"])/(n*oracle["curl_mean"])
    gate_nu = 4*rse_nu+.05
    dev_nu = eta.nu/oracle["nu_inf"]-1
    record_property("nu_true", syn.NU_TRUE)
    record_property("nu_inf_oracle", oracle["nu_inf"])
    record_property("nu_eb", eta.nu)
    record_property("nu_eb_over_true", eta.nu/syn.NU_TRUE)
    record_property("nu_rel_dev_from_oracle", dev_nu)
    record_property("nu_gate", gate_nu)
    assert abs(dev_nu) <= gate_nu
    assert eta.nu > syn.NU_TRUE and oracle["nu_inf"] > syn.NU_TRUE              # B12 direction (upward)

    candidates = rnd["update"]["candidates"]
    for k in range(3):
        rse = np.sqrt(n*oracle["tau_var"][k])/(n*oracle["tau_mean"][k])
        tau2_eb = candidates[f"tau_{k}"]["tau2"]
        dev = tau2_eb/oracle["tau_inf"][k]**2-1
        record_property(f"tau{k}_true", syn.TAU_TRUE[k])
        record_property(f"tau{k}_inf_oracle", oracle["tau_inf"][k])
        record_property(f"tau{k}_eb", float(np.sqrt(tau2_eb)))
        record_property(f"tau{k}_rel_dev_tau2_from_oracle", dev)
        record_property(f"tau{k}_gate", 4*rse+.05)
        assert abs(dev) <= 4*rse+.05, k
        same_side = np.sign(np.sqrt(tau2_eb)-syn.TAU_TRUE[k]) == np.sign(oracle["tau_inf"][k]-syn.TAU_TRUE[k])
        assert same_side, k                                                     # the oracle's direction
    for k in (1, 2):
        assert eta.tau2[k] == candidates[f"tau_{k}"]["tau2"]

    prior = result.sigma_prior                                                  # C15-5 from sigma_post
    assert prior["n_T"] == N_PAIRS and len(prior["channels"]) == syn.N_CHANNELS
    for row in prior["channels"]:
        assert row["a"] > 2 and row["share_n0_over_Nq"] > 0
    stored = json.loads((tmp_path/"eb_result.json").read_text())
    assert stored["status"] == EB_ROUND_CAP and stored["trajectory"][1]["nu"] == pytest.approx(eta.nu)
    assert len(list((tmp_path/"pairs"/"round_0").iterdir())) == N_PAIRS


def test_multilevel_map_maps_the_run_result(monkeypatch):
    """multilevel_map passes eta / sigma / config to solve_multilevel and exposes the native output only for a
    native run (usable only for a C13-3 ending)."""
    grid = syn.grid8()
    u = np.zeros(grid.shape+(3,))
    commit = Commit(u, np.zeros(3*grid.n_free_scalar), type("C", (), dict(state_hash="h", min_jd_lower_bound=1.))())
    state = syn.make_state(grid, syn.moving_channels(grid), syn.moving_channels(grid), syn.eta_true())
    calls = []

    def fake(backend, levels, *, eta, sigma, config, log):
        calls.append((levels, eta, sigma, config))
        status = calls_status.pop(0)
        native = status != "transfer_rejected"
        return RunResult(status, "r", "w", 3 if native else 1, native, commit, state, [1e-4]*3,
                         [dict(level=3, F_B_trace=[1.], GF_B_trace=[1.], x=1)], [], {}, {})
    calls_status = ["converged_regularized_gradient_B", "level_failed", "transfer_rejected"]
    monkeypatch.setattr(pop, "solve_multilevel", fake)
    eta = syn.eta_true()
    out = multilevel_map(("levels",), eta, solver_config=SolverConfig(rounds=(1, 1, 1, 2)))
    assert out.usable and out.u_hat is u and out.state is state and out.sigma_post == [1e-4]*3
    assert calls[0][1] is eta and calls[0][3].rounds == (1, 1, 1, 2)
    assert "F_B_trace" not in out.record["rounds"][0] and out.record["state_hash"] == "h"
    assert not multilevel_map(("levels",), eta).usable                              # level_failed: no Laplace
    rejected = multilevel_map(("levels",), eta)
    assert not rejected.usable and rejected.u_hat is None


@pytest.mark.skipif(os.environ.get("BAYESREG_SLOW") != "1", reason="slow (four-level solve); BAYESREG_SLOW=1")
def test_multilevel_map_real_four_levels():
    """The real four-level MAP (code defaults) and the C14 samples with real certificates on the 26^3 problem.

    eta and sigma changed in EBX (2026-10-31; lead's decision after the EBX report; the assertions are unchanged).
    The original setting, the solver test's eta = (alpha, nu, tau) = (1, 1, 1) with sigma^2 = 1, never passed, from
    the commit that introduced this test (6bd6f7b) on: the data block is about 1 % of H (delta^T Pi delta /
    delta^T H delta ~ .99), so the Laplace posterior is about the wide prior -- rms delta ~ .45 mm per component,
    max |delta| ~ 2 mm, about 40 nodes per sample out of the domain (certificate INVALID_DOMAIN) and ~6 % of the Gauss
    points with J <= 0 -- and no sample can pass; the sampler's scale is right (delta^T H delta / d = .991-1.010,
    expected 1 +- .007).  sigma^2 = 1e-4 alone does not help (the data constrain grad M only; 3.5 % of the Gauss points
    still J <= 0, and the MAP hit the 500-step cap).  Failure logs (BAYESREG_SLOW=1, this test only):
        /home/rmcfz/seq_vp_tag_results/tests/EBX_20261031T104508Z_bisect_6bd6f7b.log
        /home/rmcfz/seq_vp_tag_results/tests/EBX_20261031T110323Z_bisect_58552a1.log
        /home/rmcfz/seq_vp_tag_results/tests/EBX_20261031T110354Z_bisect_3df971a.log
    (and the DEF review's DEFR_20261031T103849Z_synthetic_eb.log, DEFR_20261031T104114Z_synthetic_eb_8a4d060.log).
    Now: tau = .1 for all four classes, nu = 100, alpha = 1, sigma^2 = 1e-3 -- the data are about 8 % of H, PCG 8
    iterations, u_hat within ~.2 mm of u*.  One run each (single observations, the machine not idle), the body of this
    test in a scratch script before the change: code defaults (gauss27+closed27, certificate) MAP converged
    (C13), 16 / 16 samples PCG-converged and certified, smallest certified lower bound .21, ~103 s; logvol_points
    gauss27 the same 16 / 16, .29, ~63 s."""
    from ..solve import _synthetic as solver_syn
    native, levels, _ = solver_syn.four_levels(n=26)
    eta = Eta(1., 100., (.01,)*4)               # tau = .1 (EBX; was (1, 1, 1) with sigma^2 = 1, see the docstring)
    out = multilevel_map(levels, eta, solver_config=SolverConfig(rounds=(1, 1, 1, 1)), sigma_policy=FixedSigma([1e-3]))
    assert out.usable, out.record["status"]
    stats, record = pair_statistics(out.state, out.u_hat, pair_id="four_levels", config=EBConfig(eb_count="dim"),
                                    seed=3, sigma_post=out.sigma_post)
    assert stats.n_passed >= 2 and np.isfinite(stats.curl_trace) and stats.n_curl == 3*native.n_free_scalar


@pytest.mark.skipif(os.environ.get("BAYESREG_SLOW") != "1", reason="slow (3 rounds x 3 pairs); BAYESREG_SLOW=1")
def test_three_rounds_from_an_off_truth_start(record_property):
    """C15-6 on the real path: eta0 with nu = nu_true/5 and tau_k = 2 tau_true (tau_0 fixed at the truth).  At most
    3 rounds; the result is reported as it is (criterion met or round cap).

    Assertions (corrected in FB1 after review M5_EB item A1):
      * P2 (C05-6, C15-4): alpha is never learned; every eta of the trajectory has alpha = alpha_nug(its tau)
        = 1e-2 min_k 1/tau_k^2, recomputed from the new tau each round (bitwise).  alpha is NOT constant here: the
        start has tau_2 = .20 > tau_0 = .15 (alpha = .25), and once tau_2 falls below tau_0, alpha = 1e-2/.15^2 = .444.
        The old assertion "alpha constant" would fail by this rule;
      * tau_0 fixed bitwise (D28);
      * nu rises strictly in every round from its start nu_true/5 and ends above nu_true (the direction B12 predicts
        for the block-wise normalization (N)).  [inferred] The nu update N_C / E[||C u||^2_W + tr(C^T W C H^-1)] grows
        with nu (a stronger prior shrinks the posterior), so iterates from below increase; a probe run gave
        20 -> 93 -> 156 -> 188.  The oracle point nu_inf of _synthetic.py is the ONE-STEP value from eta_true, not a
        fixed point of the iteration (the iterates pass it), so no assertion is made about approaching it."""
    oracle = syn.oracle()
    data = {f"OASIS_OAS1_09{k:02d}_MR1__OASIS_OAS1_09{k+50:02d}_MR1": p
            for k, p in enumerate(syn.make_pairs(N_PAIRS))}
    pairs = tuple(EBPair(*pid.split("__"), "A", {"check": "synthetic"}) for pid in data)
    tau2 = (syn.TAU_TRUE[0]**2,)+tuple((2*t)**2 for t in syn.TAU_TRUE[1:])
    eta0 = Eta(pop.alpha_nugget(tau2), syn.NU_TRUE/5, tau2)
    backend = NumpyBackend()
    result = run_population_eb(pairs, lambda pair, eta, *, round_index: syn.native_map(data[pair.pair_id], eta,
                                                                                       backend=backend),
                               eta0=eta0, config=EBConfig(eb_count="dim"), run_id="synthetic_eb_3", backend=backend)
    record_property("status", result.status)
    record_property("nu_trajectory", [round(t["nu"], 2) for t in result.trajectory])
    record_property("tau1_trajectory", [round(t["tau"][1], 4) for t in result.trajectory])
    record_property("tau2_trajectory", [round(t["tau"][2], 4) for t in result.trajectory])
    record_property("max_relative_change", result.record["max_relative_change"])
    record_property("oracle", dict(nu_inf=oracle["nu_inf"], tau_inf=oracle["tau_inf"]))
    record_property("alpha_trajectory", [t["alpha"] for t in result.trajectory])
    assert result.status in (pop.EB_CRITERION_MET, EB_ROUND_CAP) and len(result.rounds) <= 3
    assert len(result.trajectory) == len(result.rounds)+1
    for t in result.trajectory:                                                  # P2: alpha = alpha_nug(tau), bitwise
        assert t["alpha"] == pop.alpha_nugget(t["tau2"])
        assert t["tau2"][0] == eta0.tau2[0]                                      # tau_0 fixed (D28)
    for rnd in result.rounds:
        assert rnd["update"]["alpha_source"] == "alpha_nug(tau_new)" and "alpha" not in rnd["update"]["learned"]
        assert rnd["eta_out"]["alpha"] == rnd["update"]["alpha_nug_of_new_tau"]
    nus = [t["nu"] for t in result.trajectory]
    assert all(b > a for a, b in zip(nus, nus[1:])), nus                         # rises every round from nu_true/5
    assert nus[-1] > syn.NU_TRUE                                                  # B12 direction (upward)
