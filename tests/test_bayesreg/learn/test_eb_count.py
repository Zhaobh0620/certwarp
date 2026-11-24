"""Contract C15-9 (eb_count) apart from the LG gate (test_eb_gm_linear_gaussian.py) and the dense prior-side check
(test_eb_gm_dense.py):
  * EBConfig: default "gull_mackay" with P2 (BZ 2026-10-30, EBD; C15-9 / D45); "dim" explicit; P1 + gull_mackay
    refused (gate (iv)); P3 + gull_mackay refused until a P3 gate exists (C15-9, review R2 important 1); P1 / P3 with
    the default eb_count refused the same way, the message naming eb_count='dim'; the record gains only the key
    eb_count under dim;
  * dim bitwise (C15-9 "dim 逐位不变的范围"), with eb_count = "dim" passed explicitly: the digests of _eb_digest.py
    equal those of the code before C15-9 (commit 7579645; tests/test_bayesreg/_golden/golden_7579645_eb_def.json, made
    by make_golden_eb.py after the explicit GAUSS27 of DEF (logvol_points "gauss27", the model default before DEF);
    golden_7579645_eb_ebd.json, made after the explicit DIM, and golden_7579645_eb.json, made before it, have the same
    digests);
  * prior samples use a stream separate from the posterior samples of the same run id (gate (v));
  * the C15-9 update by hand: Gull-MacKay values, the ratio fallback (gm_fallback), the recorded dim values, z values,
    pair exclusions (S0_conv = 0), statistics of another eta refused, the alpha_nug coupling flag (B17 4.3);
  * the driver under gull_mackay: prior_seed passed, step ratios, the "步长判据" wording, coupled rounds; and nothing of
    this under dim.
No data/, no certificate.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jd_curl.validation import InputValidationError

from bayesreg.infer.laplace import seed_from_run_id
from bayesreg.learn import (DEFAULT_EB_COUNT, EB_CRITERION_MET, EB_ROUND_CAP, EB_WORDING, EB_WORDING_GM, GM_FALLBACK,
                            GM_FORM, EBConfig, GMStatistics, PairStatistics, alpha_nugget, eb_update, prior_round_seed,
                            round_seed, run_population_eb)
from bayesreg.learn import population as pop
from bayesreg.solve import Eta

from . import _eb_digest as D
from .test_population import contraction_statistics, fake_pairs, solve_ready

TAU2 = (1., .25, .36, .16)
GOLDEN_7579645_EB = dict(
    config="b1ee1827d53371ce20403df86f29c22d401508b4700e0a82f381ab61518ed3ab",
    driver="41fc5d0658a74750ef4331bdbe5ad8415e2dd20876d5eaf5dad46ba04c5999db",
    gauss="9b08f79fd9713adf8538a868b1e3fbbaeb8bc982e5bd58dcd3b59393718bfbd6",
    pair="dd4cbc36705f0564b67351d7db3172ac8d7ce09cb73ece24c1b77fc51e88674a",
    update="3371c7243ed5d50766f47e308f596f830e174954f11654efd5ae7871fcc01ca5")
GOLDEN_DIR = Path(__file__).resolve().parents[1]/"_golden"
GOLDEN_RECORD = GOLDEN_DIR/"golden_7579645_eb_def.json"        # _eb_digest.py with the explicit GAUSS27 (DEF)
GOLDEN_RECORD_EBD = GOLDEN_DIR/"golden_7579645_eb_ebd.json"     # the module with the explicit DIM (kept; same digests)
GOLDEN_RECORD_EBC2 = GOLDEN_DIR/"golden_7579645_eb.json"       # the module before EBD (kept; same digests)
GM = EBConfig(eb_count="gull_mackay")
DIM = EBConfig(eb_count="dim")


# ----------------------------------------------------------------------------------------------
# Configuration and the dim gate
# ----------------------------------------------------------------------------------------------

def p3_gm_bypassing_the_refusal():
    """P3 + gull_mackay is refused by EBConfig (C15-9, R2 important 1); the alpha arithmetic of gull_mackay_update is
    kept for when a P3 gate exists, so this test-only config sets parametrization after construction."""
    config = replace(GM)
    object.__setattr__(config, "parametrization", "P3")
    return config


def test_config_eb_count_and_p1_refused():
    """EBD (BZ 2026-10-30, C15-9 / D45): the default is gull_mackay with P2; dim is the explicit option."""
    assert DEFAULT_EB_COUNT == "gull_mackay"
    assert EBConfig().eb_count == "gull_mackay" and EBConfig().gull_mackay and EBConfig().parametrization == "P2"
    assert EBConfig() == GM and EBConfig().record() == GM.record()
    assert DIM.eb_count == "dim" and not DIM.gull_mackay and DIM.parametrization == "P2"
    assert GM.gull_mackay and GM.parametrization == "P2"
    with pytest.raises(InputValidationError, match="P2 and P3"):
        EBConfig(parametrization="P1", eb_count="gull_mackay")             # gate (iv)
    with pytest.raises(InputValidationError, match=r"P2 and P3 only.*pass eb_count='dim'"):
        EBConfig(parametrization="P1")                                     # P1 + the default: refused, dim named
    assert EBConfig(parametrization="P1", eb_count="dim").learned(3) == ["alpha", "nu", "tau_0", "tau_1", "tau_2"]
    with pytest.raises(InputValidationError, match="eb_count"):
        EBConfig(eb_count="gull-mackay")
    dim, gm = DIM.record(), GM.record()
    assert dim["eb_count"] == "dim" and "gull_mackay" not in dim
    assert gm["eb_count"] == "gull_mackay" and "C15-9" in gm["gull_mackay"]["contract"]
    assert "BZ 10-30" in gm["gull_mackay"]["contract"]


def test_p3_gull_mackay_refused_until_a_p3_gate():
    """C15-9 (EBC3, review R2 important 1): P3 + gull_mackay is refused (no P3 gate; on a toy the evidence supremum
    lies at alpha -> 0); P3 under dim and P2 under gull_mackay are unchanged.  P3 with the default eb_count
    (gull_mackay since EBD) is refused the same way, the message naming eb_count='dim'."""
    with pytest.raises(InputValidationError, match=r"P3 is not allowed yet \(C15-9; review R2 important 1\)"):
        EBConfig(parametrization="P3", eb_count="gull_mackay")
    with pytest.raises(InputValidationError, match=r"P3 is not allowed yet.*pass eb_count='dim'"):
        EBConfig(parametrization="P3")                                     # P3 + the default
    with pytest.raises(InputValidationError, match="P3 is not allowed yet"):
        replace(EBConfig(parametrization="P3", eb_count="dim"), eb_count="gull_mackay")
    assert EBConfig(parametrization="P3", eb_count="dim").eb_count == "dim"
    assert EBConfig(eb_count="gull_mackay").parametrization == "P2"
    assert EBConfig(learn_tau0=True, eb_count="gull_mackay").gull_mackay


@pytest.mark.skipif(np.__version__ != "2.0.2", reason="bitwise golden digests recorded with numpy 2.0.2")
def test_dim_is_bitwise_the_code_before_c15_9():
    """C15-9: under eb_count = "dim" (passed explicitly, _eb_digest.DIM) the per-pair statistics, Gauss sums,
    posterior qc, update records, eta trajectory and result record are bitwise those of 7579645; the config record
    differs only by the key eb_count (= "dim", the one key the digest drops)."""
    assert D.DIM == dict(eb_count="dim") and D.GAUSS27 == dict(logvol_points="gauss27")     # DEF: explicit gauss27
    assert D.all_digests() == GOLDEN_7579645_EB


def test_golden_constants_match_the_generation_record():
    record = json.loads(GOLDEN_RECORD.read_text(encoding="utf-8"))
    assert record["ref_commit"].startswith("7579645") and record["digests"] == GOLDEN_7579645_EB
    module = Path(D.__file__).read_bytes()
    assert hashlib.sha256(module).hexdigest() == record["digest_module_sha256"]     # regenerate after edits
    for path in (GOLDEN_RECORD_EBD, GOLDEN_RECORD_EBC2):       # the records before the explicit GAUSS27 / DIM
        before = json.loads(path.read_text(encoding="utf-8"))
        assert before["ref_commit"] == record["ref_commit"] and before["digests"] == GOLDEN_7579645_EB


def test_prior_seed_stream_is_separate_from_the_posterior_stream():
    """Gate (v): prior_round_seed(run, r, pair) = seed_from_run_id(run/eb_round<r>/<pair>/prior) differs from the
    posterior round_seed and gives different normals for every sample child stream."""
    pair = "OASIS_OAS1_0901_MR1__OASIS_OAS1_0951_MR1"
    prior, post = prior_round_seed("run", 0, pair), round_seed("run", 0, pair)
    assert prior == seed_from_run_id(f"run/eb_round0/{pair}/prior") and prior != post
    assert prior != prior_round_seed("run", 1, pair) and prior != prior_round_seed("run", 0, pair+"x")
    for a, b in zip(np.random.default_rng(prior).spawn(16), np.random.default_rng(post).spawn(16)):
        assert not np.array_equal(a.standard_normal(8), b.standard_normal(8))


# ----------------------------------------------------------------------------------------------
# The C15-9 update by hand
# ----------------------------------------------------------------------------------------------

def gm_stats(eta, pair_id="a", *, g_nu=1.2, g_alpha=.4, t_nu=.25, q_nu=1.5, m_prior=(1., 2., 3., 4.),
             m_hat=(.5, 1., 1.5, 2.), m_var=(.2, .4, .6, .8), conv0=16, n=300):
    gm = GMStatistics(n_prior_samples=16, n_prior_converged=conv0, g_alpha=g_alpha, g_alpha_stderr=.02, g_nu=g_nu,
                      g_nu_stderr=.05, m_prior=m_prior, m_prior_stderr=(.1,)*4, m_hat=m_hat, m_var=m_var,
                      m_var_stderr_bound=(.05,)*4, alpha=eta.alpha, nu=eta.nu, tau2=tuple(eta.tau2), count_check=1.,
                      identity_max_rel=1e-16, seed=7)
    return PairStatistics(pair_id=pair_id, n_free=n, n_curl=n, n_q=1e3, kq_quadratic=2., curl_quadratic=q_nu,
                          kq_trace=.5, kq_trace_stderr=.01, curl_trace=t_nu, curl_trace_stderr=.01,
                          tau_hat=(1., 2., 3., 4.), tau_var=(.5, .5, .5, .5), tau_den=(10., 20., 30., 40.),
                          sigma_used=(.01,), sigma_post=(.01,), n_samples=16, n_converged=16, n_passed=16,
                          pass_rate=1., pass_rate_low=False, gm=gm)


def p2(nu=2.):
    return Eta(alpha_nugget(TAU2), nu, TAU2)


def test_gull_mackay_update_formulas_by_hand():
    eta = p2()
    a, b = gm_stats(eta, "a"), gm_stats(eta, "b", g_nu=2., t_nu=.5, q_nu=2.5, m_hat=(1., 2., 3., 4.))
    new, record = eb_update(eta, [a, b], GM)
    nu_gm = (1.2+2.-2.*(.25+.5))/(1.5+2.5)
    assert new.nu == pytest.approx(nu_gm, rel=1e-15) and record["candidates"]["nu"]["form"] == GM_FORM
    assert record["candidates"]["nu"]["ratio"] == pytest.approx(3.2/(4.+.75), rel=1e-15)
    assert record["candidates"]["nu"]["dim"] == pytest.approx(600/(4.+.75), rel=1e-15)         # C15-1, same pairs
    z = (3.2-2.*.75)/np.sqrt(2*.05**2+2*(2.*.01)**2)
    assert record["candidates"]["nu"]["z_numerator"] == pytest.approx(z, rel=1e-12)
    for k in (1, 2, 3):
        hat, prior, var = a.gm.m_hat[k]+b.gm.m_hat[k], 2*a.gm.m_prior[k], 2*a.gm.m_var[k]
        assert new.tau2[k] == pytest.approx(TAU2[k]*hat/(prior-var), rel=1e-15)
        c = record["candidates"][f"tau_{k}"]
        assert c["ratio"] == pytest.approx(TAU2[k]*(hat+var)/prior, rel=1e-15)
        assert c["dim"] == pytest.approx((2*a.tau_hat[k]+2*a.tau_var[k])/(2*a.tau_den[k]), rel=1e-15)
        assert c["z_denominator"] == pytest.approx((prior-var)/np.sqrt(2*.1**2+2*.05**2), rel=1e-12)
    assert new.tau2[0] == eta.tau2[0] and "tau_0" in record["candidates"]          # tau_0 fixed, recorded (D28)
    assert new.alpha == alpha_nugget(new.tau2) == eta.alpha                         # tau_0 = 1 still the largest
    assert record["alpha_nug_coupling"]["flag"] is False and record["gm_fallback"] == []
    assert record["eb_count"] == "gull_mackay" and record["count_checks"] == {"a": 1., "b": 1.}
    p3, rec3 = eb_update(Eta(.5, 2., TAU2), [gm_stats(Eta(.5, 2., TAU2))], p3_gm_bypassing_the_refusal())
    assert p3.alpha == pytest.approx((.4-.5*.5)/2., rel=1e-15) and rec3["alpha_nug_coupling"]["flag"] is None


def test_ratio_fallback_and_failure():
    eta = p2()
    low = gm_stats(eta, g_nu=.1, m_var=(.2, 2.5, .6, .8))   # nu numerator .1 - 2 x .25 < 0; tau_1 m_var > m_prior
    new, record = eb_update(eta, [low], GM)
    assert record["candidates"]["nu"]["form"] == GM_FALLBACK and new.nu == pytest.approx(.1/(1.5+.25), rel=1e-15)
    assert record["candidates"]["tau_1"]["form"] == GM_FALLBACK
    assert new.tau2[1] == pytest.approx(TAU2[1]*(1.+2.5)/2., rel=1e-15)
    assert record["gm_fallback"] == ["nu", "tau_1"] and record["forms"]["tau_2"] == GM_FORM
    with pytest.raises(pop.EBUpdateFailed):
        eb_update(eta, [gm_stats(eta, m_prior=(1., 0., 3., 4.), m_var=(.2, 0., .6, .8))], GM)   # learned tau_1
    dead_background = gm_stats(eta, m_prior=(0., 2., 3., 4.), m_var=(0., .4, .6, .8), m_hat=(0., 1., 1.5, 2.))
    _, record = eb_update(eta, [dead_background], GM)
    assert "tau_0" in record["candidates_unavailable"]                             # not learned: diagnostic only


def test_pair_exclusions_and_refusals():
    eta = p2()
    good, no_prior = gm_stats(eta, "good"), gm_stats(eta, "no_prior", conv0=0)
    no_gm = replace(gm_stats(eta, "no_gm"), gm=None)
    new, record = eb_update(eta, [good, no_prior, no_gm, None], GM)
    assert record["pairs_used_alpha_nu"] == ["good"] and record["pairs_used_tau"] == ["good"]
    excluded = {r["pair_id"]: r for r in record["pairs_excluded"]}
    assert excluded["no_prior"]["S0_conv"] == 0 and excluded["no_gm"]["S0_conv"] is None
    with pytest.raises(pop.EBUpdateFailed, match="S0_conv"):
        eb_update(eta, [no_prior, no_gm], GM)
    with pytest.raises(InputValidationError, match="not this round's eta"):
        eb_update(p2(nu=3.), [good], GM)                                            # statistics of another eta


def test_alpha_nug_coupling_flag():
    """B17 corollary 4.3: alpha_nug = 1e-2/max_k tau_k^2 moves with a learned tau_k only when it is the largest."""
    eta = p2()
    big = gm_stats(eta, m_hat=(.5, 1., 1.5, 400.))                  # tau_3^2 -> .16 x 400/3.2 = 20 > tau_0^2 = 1
    new, record = eb_update(eta, [big], GM)
    assert new.tau2[3] > 1 and new.alpha == pytest.approx(1e-2/new.tau2[3])
    assert record["alpha_nug_coupling"] == dict(flag=True, components=[3], rule=record["alpha_nug_coupling"]["rule"])
    _, record = eb_update(eta, [gm_stats(eta)], EBConfig(learn_tau0=True, eb_count="gull_mackay"))
    assert record["alpha_nug_coupling"]["flag"] is True                             # learned tau_0: always coupled
    assert pop.alpha_nug_coupling((1., .25), ["nu", "tau_1"], "P2")["flag"] is False
    assert pop.alpha_nug_coupling((1., 1.), ["nu", "tau_1"], "P2")["components"] == [1]    # tie counts


# ----------------------------------------------------------------------------------------------
# The driver
# ----------------------------------------------------------------------------------------------

def gm_driver_statistics(target, calls):
    """nu_new = target(nu) (q = 1, t = 0, g_nu = target(nu)); tau unchanged (m_hat = m_prior - m_var)."""
    def statistics(outcome, *, pair_id, config, seed, backend, prior_seed):
        eta = outcome.record["eta"]
        calls.append((pair_id, seed, prior_seed))
        n = len(eta.tau2)
        gm = GMStatistics(n_prior_samples=16, n_prior_converged=16, g_alpha=.1, g_alpha_stderr=.01,
                          g_nu=target(eta.nu), g_nu_stderr=.01, m_prior=(2.,)*n, m_prior_stderr=(.01,)*n,
                          m_hat=(1.,)*n, m_var=(1.,)*n, m_var_stderr_bound=(.01,)*n, alpha=eta.alpha, nu=eta.nu,
                          tau2=tuple(eta.tau2), count_check=1., identity_max_rel=0., seed=prior_seed)
        s = PairStatistics(pair_id=pair_id, n_free=300, n_curl=300, n_q=1e3, kq_quadratic=1., curl_quadratic=1.,
                           kq_trace=0., kq_trace_stderr=0., curl_trace=0., curl_trace_stderr=0., tau_hat=(1.,)*n,
                           tau_var=(0.,)*n, tau_den=(10.,)*n, sigma_used=(.01,), sigma_post=(.01,), n_samples=16,
                           n_converged=16, n_passed=16, pass_rate=1., pass_rate_low=False, gm=gm)
        return s, dict(seed=seed, prior_seed=prior_seed)
    return statistics


def test_driver_gull_mackay_records(tmp_path):
    calls = []
    target = lambda nu: 100+.5*(nu-100)                         # noqa: E731
    result = run_population_eb(fake_pairs(), solve_ready, eta0=p2(nu=300.), config=GM, run_id="t", out_dir=tmp_path,
                               statistics=gm_driver_statistics(target, calls))
    assert result.status == EB_ROUND_CAP and [t["nu"] for t in result.trajectory] == pytest.approx([300, 200, 150, 125])
    assert result.wording == EB_WORDING_GM[EB_ROUND_CAP] and "步长判据" in result.wording
    assert result.record["eb_count"] == "gull_mackay"
    assert result.record["step_ratios"] == [None, pytest.approx(.75), pytest.approx(2/3)]   # 1/3, 1/4, 1/6
    assert result.record["alpha_nug_coupled_rounds"] == [] and result.record["gm_fallback_rounds"] == {}
    assert len(calls) == 6 and all(prior_seed != seed for _, seed, prior_seed in calls)
    assert calls[0][2] == prior_round_seed("t", 0, calls[0][0]) and calls[0][1] == round_seed("t", 0, calls[0][0])
    assert calls[-1][2] == prior_round_seed("t", 2, calls[-1][0])
    stored = json.loads((tmp_path/"eb_result.json").read_text())
    assert stored["eb_count"] == "gull_mackay" and stored["config"]["eb_count"] == "gull_mackay"
    met = run_population_eb(fake_pairs(), solve_ready, eta0=p2(nu=104.), config=GM, run_id="t2",
                            statistics=gm_driver_statistics(lambda nu: 100+.5*(nu-100), []))
    assert met.status == EB_CRITERION_MET and met.wording == EB_WORDING_GM[EB_CRITERION_MET]
    assert "步长判据满足" in met.wording and "收敛" in met.wording and "不是收敛" in met.wording


def test_driver_under_dim_adds_nothing():
    result = run_population_eb(fake_pairs(), solve_ready, eta0=p2(nu=300.), config=DIM, run_id="t",
                               statistics=contraction_statistics(lambda nu: 100+.5*(nu-100), []))
    assert result.wording == EB_WORDING[result.status]
    assert not {"eb_count", "step_ratios", "alpha_nug_coupled_rounds"} & set(result.record)
    assert all("step_ratio" not in r and "gull_mackay" not in r["update"] for r in result.rounds)
    assert result.record["config"]["eb_count"] == "dim"


def test_gull_mackay_statistics_step_gets_the_prior_seed():
    def no_prior(outcome, *, pair_id, config, seed, backend):
        raise AssertionError("not reached")
    with pytest.raises(TypeError):                                  # the driver passes prior_seed under gull_mackay
        run_population_eb(fake_pairs(), solve_ready, eta0=p2(), config=GM, run_id="t", statistics=no_prior)
