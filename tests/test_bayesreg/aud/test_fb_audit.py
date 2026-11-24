"""AUD tools (experiments/aud; runs/20261102-fb-audit/plan.md) on toy grids and made-up numbers: no data/, no GPU, no
certificate.

- the blocks of fb_audit.field_report add up to F_B; the two log-volume parts to F_vol; brain + non-brain cells to each
  prior block; the tile pass agrees with evaluate; u^T K_Q u against stiffness_apply; the F_vol table against evaluate on a
  state whose tau_k are scaled (tau_0 fixed); the M step and L against a re-evaluation at sigma~; linear rescaling of the
  channel energies; a wrong tile pass stops;
- analyze: the crossings make the margin zero; the sides; J1 / J2 / J3; the main term; flip_boundaries against a direct
  computation of F at the crossing.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bayesreg.geom import stiffness_apply
from bayesreg.model import evaluate
from jd_curl.three_d.boundary import restrict_free

from ..model._helpers import aniso, feasible_field, make_state

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(f"aud_{name}", ROOT/"experiments"/"aud"/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUD, ANALYZE = load("fb_audit"), load("analyze")


@pytest.fixture(scope="module")
def case():
    grid = aniso()
    state = make_state(grid, seed=3)
    assert state.logvol_closed27
    u = feasible_field(grid, np.random.default_rng(4))
    return state, u, AUD.field_report(state, u)


def test_blocks_add_up_to_f_b(case):
    state, u, report = case
    assert report["F_B"] == pytest.approx(report["F_intensity"]+report["F_structure"]+report["F_diff"]+report["F_vol"]
                                          + report["F_curl"], rel=1e-14)
    assert report["F_data"] == pytest.approx(sum(report["channel"]), rel=1e-15)
    assert report["F_vol"] == pytest.approx(report["F_vol_gauss"]+report["F_vol_closed27"], rel=1e-14)
    assert report["F_B"] == evaluate(state, u).energy


def test_tile_pass_against_evaluate(case):
    state, u, report = case
    passes = report["prior_pass"]
    assert .5*state.alpha*(passes["grad_sq"]["brain"]+passes["grad_sq"]["other"]) == pytest.approx(report["F_diff"], rel=1e-13)
    assert .5*state.nu*(passes["curl_sq"]["brain"]+passes["curl_sq"]["other"]) == pytest.approx(report["F_curl"], rel=1e-13)
    assert (.5*(passes["logvol_gauss_sq"]["brain"]+passes["logvol_gauss_sq"]["other"])
            == pytest.approx(report["F_vol_gauss"], rel=1e-13))
    assert (.5*(passes["logvol_closed27_sq"]["brain"]+passes["logvol_closed27_sq"]["other"])
            == pytest.approx(report["F_vol_closed27"], rel=1e-13))
    assert all(v < 1e-12 for v in report["checks"].values())


def test_brain_split_uses_the_lower_corner_node():
    grid = aniso()
    state = make_state(grid, seed=3, mask=np.ones(grid.shape, dtype=bool))
    u = feasible_field(grid, np.random.default_rng(4))
    passes = AUD.prior_pass(state, u)
    assert passes["curl_sq"]["other"] == 0. and passes["logvol_gauss_sq"]["other"] == 0.
    mask = np.zeros(grid.shape, dtype=bool)
    mask[1, 1, 1] = True
    state = make_state(grid, seed=3, mask=mask)
    passes_one = AUD.prior_pass(state, u)
    geo_sum = passes_one["curl_sq"]["brain"]
    assert 0. < geo_sum < passes["curl_sq"]["brain"]


def test_uKu_against_stiffness(case):
    state, u, report = case
    free = restrict_free(u, state.grid)
    assert report["uKu"] == pytest.approx(float(free@stiffness_apply(free, state.grid)), rel=1e-12)
    assert report["F_diff"] == pytest.approx(.5*state.alpha/state.omega0*report["uKu"], rel=1e-12)


@pytest.mark.parametrize("s", [.25, 1/np.sqrt(2), 2., 4.])
def test_vol_table_against_scaled_state(case, s):
    state, u, report = case
    passes = report["prior_pass"]
    j = int(np.argmin(np.abs(np.asarray(passes["s_grid"])-s)))
    scaled = np.array(state.tau2)
    scaled[1:] = (passes["s_grid"][j]**2)*scaled[1:]                 # tau_0 fixed, tau_k -> s tau_k
    energies = evaluate(state.replace(tau2=scaled), u).energies
    assert passes["F_vol_gauss_of_s"][j]+passes["F_vol_closed27_of_s"][j] == pytest.approx(energies["logvol"], rel=1e-13)
    assert passes["F_vol_gauss_of_s"][j] == pytest.approx(energies["logvol_parts"]["gauss"], rel=1e-13)


def test_m_step_and_joint_objective(case):
    state, u, report = case
    from bayesreg.solve.multilevel import n_q
    m = report["m_step"]
    again = evaluate(state.replace(sigma2=np.asarray(m["sigma2"])), u)
    assert again.energy == pytest.approx(m["F_B"], rel=1e-13)
    nq = n_q(state)
    assert sum(again.energies["channel"]) == pytest.approx(state.n_channels*(nq+2.)/2., rel=1e-13)
    # C11-1 at sigma~, a = b = 0, against its definition term by term
    L = 0.
    for c in range(state.n_channels):
        L += .5*report["rho2_B"][c]/m["sigma2"][c]+(nq/2.+1.)*np.log(m["sigma2"][c])
    L += report["F_diff"]+report["F_vol"]+report["F_curl"]
    assert m["L"] == pytest.approx(L, rel=1e-13)
    # sigma~ minimizes L over sigma for fixed u
    for factor in (.5, 2.):
        other = 0.
        for c in range(state.n_channels):
            value = factor*m["sigma2"][c]
            other += .5*report["rho2_B"][c]/value+(nq/2.+1.)*np.log(value)
        assert other+report["F_diff"]+report["F_vol"]+report["F_curl"] > m["L"]


def test_channel_energies_rescale_linearly(case):
    state, u, report = case
    other = np.asarray(state.sigma2)*np.array([2., .5, 3.])
    direct = evaluate(state.replace(sigma2=other), u).energies["channel"]
    rescaled = AUD.data_at_sigma(dict(channel=report["channel"]), state.sigma2, other)
    assert rescaled == pytest.approx(list(direct), rel=1e-13)
    assert ANALYZE.data_at_sigma(report["rho2_B"], other) == pytest.approx(list(direct), rel=1e-13)


def test_u_zero_has_data_only():
    grid = aniso()
    state = make_state(grid, seed=3)
    report = AUD.field_report(state, np.zeros(grid.shape+(3,)))
    assert report["F_diff"] == report["F_vol"] == report["F_curl"] == 0.
    assert report["F_B"] == pytest.approx(report["F_data"], rel=1e-15)
    assert report["log_j"]["values"] == [0.]*len(report["log_j"]["values"])


def test_tile_check_stops_on_a_mismatch(case):
    state, _, report = case
    bad = dict(report, F_curl=report["F_curl"]*(1.+1e-6))
    with pytest.raises(AUD.AuditStop):
        AUD.tile_checks(state, bad)


def test_write_new_refuses_an_existing_file(tmp_path):
    path = tmp_path/"a.json"
    AUD.write_new(path, dict(x=1))
    with pytest.raises(FileExistsError):
        AUD.write_new(path, dict(x=2))
    assert path.read_text().strip().replace(" ", "").replace("\n", "") == '{"x":1}'


# ---- analyze ---------------------------------------------------------------------------------------------------------

def test_linear_crossing():
    c = ANALYZE.linear_crossing(3., -2., 1.)                # M = 3 - 2p: X preferred above 1.5
    assert c == dict(star=1.5, side="above", always=None)
    c = ANALYZE.linear_crossing(-3., 2., 10.)               # M = -3 + .2 p: X preferred below 15
    assert c == dict(star=15., side="below", always=None)
    assert ANALYZE.linear_crossing(3., 2.)["always"] is False and ANALYZE.linear_crossing(3., 2.)["star"] is None
    assert ANALYZE.linear_crossing(-3., -2.)["always"] is True
    assert ANALYZE.linear_crossing(-3., 0.)["always"] is True and ANALYZE.linear_crossing(3., 0.)["always"] is False


def test_grid_crossings():
    grid = [.5, 1., 2., 4.]
    c = ANALYZE.grid_crossings(grid, [3., 1., -1., -3.])
    assert c["stars"] == [pytest.approx(np.sqrt(2.))] and not c["preferred_at_low_end"] and c["preferred_at_high_end"]
    assert ANALYZE.grid_crossings(grid, [1., 1., 1., 1.])["stars"] == []


def test_judge_and_main_term():
    assert [ANALYZE.judge(n) for n in (0, 1, 2, 3, 4, 5)] == ["J2", "J2", "J3", "J3", "J1", "J1"]
    m = ANALYZE.main_term(dict(F_intensity=-5., F_structure=3., F_diff=0., F_vol=1., F_curl=0.))
    assert m == dict(term="F_structure", share=.75)
    assert ANALYZE.main_term(dict(F_intensity=-5.))["term"] is None


def fake_field(F_I, F_S, F_diff, F_curl, gauss, closed):
    grid = [.5, 1., 2.]
    vol = gauss[1]+closed[1]
    return dict(F_intensity=F_I, F_structure=F_S, F_diff=F_diff, F_vol=vol, F_curl=F_curl,
                prior_pass=dict(s_grid=grid, F_vol_gauss_of_s=gauss, F_vol_closed27_of_s=closed))


def margin(x, r, nu_factor=1., w_I=1., w_S=1., j=1):
    def F(f):
        return (w_I*f["F_intensity"]+w_S*f["F_structure"]+f["F_diff"]+nu_factor*f["F_curl"]
                + f["prior_pass"]["F_vol_gauss_of_s"][j]+f["prior_pass"]["F_vol_closed27_of_s"][j])
    return F(x)-F(r)


def test_flip_boundaries_against_a_direct_computation(monkeypatch):
    monkeypatch.setattr(ANALYZE, "S_LIST", (1., 2.))
    monkeypatch.setattr(ANALYZE, "NU_LIST", (10., 5.))
    x = fake_field(50., 400., 2., 105., [8., 4., 2.], [8., 4., 2.])
    r = fake_field(90., 100., 1., 100., [6., 3., 1.], [6., 3., 1.])
    nu0 = 10.
    out = ANALYZE.flip_boundaries(x, r, nu0)
    assert out["M"] == pytest.approx(margin(x, r))
    assert out["terms"]["F_structure"] == 300.
    # w_S: M = 0 at the crossing
    assert margin(x, r, w_S=out["w_S"]["star"]) == pytest.approx(0., abs=1e-9)
    assert out["w_S"]["side"] == "below"
    # w_I: F_I lower in X, a larger w_I helps; the crossing is M = 0
    assert margin(x, r, w_I=out["w_I"]["star"]) == pytest.approx(0., abs=1e-9)
    assert out["w_I"]["side"] == "above"
    # nu: data and vol keep X worse, the curl of X is larger: no positive nu flips it
    assert out["nu"]["star"] is None and out["nu"]["always"] is False
    # the plane: at s_tau = 1 the same as the nu row
    assert out["nu_of_s"][1]["star"] == out["nu"]["star"]
    # w_S on (nu, s) at nu0, s = 1 equals the w_S row
    row = [t for t in out["w_S_on_nu_s"] if t["nu"] == 10. and t["s_tau"] == 1.][0]
    assert row["star"] == pytest.approx(out["w_S"]["star"])


def test_energy_at_sigma_and_shared_sigma(case):
    state, u, report = case
    other = np.asarray(state.sigma2)*np.array([2., .5, 3.])
    assert ANALYZE.energy_at_sigma(report, other) == pytest.approx(evaluate(state.replace(sigma2=other), u).energy, rel=1e-13)
    nq = 100.
    sig = ANALYZE.shared_structure_sigma(report, nq)
    assert sig[1] == sig[2] and sig[0] == pytest.approx(report["rho2_B"][0]/102.)
    assert sum(ANALYZE.data_at_sigma(report["rho2_B"], sig)[1:]) == pytest.approx(102., rel=1e-13)


# ---- revision 1: the profile objective, subsets, flips, high-frequency displacement ------------------------------------

def profile_field(rho2, F_diff, F_curl, vol):
    grid = [.5, 1., 2.]
    half = [v/2. for v in vol]
    return dict(rho2_B=rho2, F_diff=F_diff, F_curl=F_curl, F_vol=vol[1],
                prior_pass=dict(s_grid=grid, F_vol_gauss_of_s=half, F_vol_closed27_of_s=half))


def test_profile_terms_definition(case):
    state, _, report = case
    nq = 1000.
    terms = ANALYZE.profile_terms(report, nq)
    assert terms == pytest.approx([.5*nq*np.log(v/(nq+2.)) for v in report["rho2_B"]], rel=1e-14)
    # the sigma~ of the report's M step at the state's own N_q
    from bayesreg.solve.multilevel import n_q
    assert np.exp(2.*ANALYZE.profile_terms(report, n_q(state))/n_q(state)) == pytest.approx(report["m_step"]["sigma2"], rel=1e-12)


def test_profile_grid_against_a_direct_computation(monkeypatch):
    names = ["intensity", "cerebral_wm", "cerebral_cortex", "small"]
    x = profile_field([.5, .4, .4, 9.], 3., 500., [40., 20., 10.])
    r = profile_field([2., 1., 1., 1.], 1., 100., [8., 4., 2.])
    nq, nu0 = 100., 21.85496512638118
    out = ANALYZE.profile_grid(x, r, nq, nu0, names)
    assert list(out["subsets"]) == [label for label, _ in ANALYZE.SUBSETS]
    tau_s = dict(ANALYZE.TAU_GRID)
    subsets = {label: list(range(4)) if chosen is None else [names.index(n) for n in chosen] for label, chosen in ANALYZE.SUBSETS}
    for label, indices in subsets.items():
        entry = out["subsets"][label]
        assert entry["channels"] == [names[i] for i in indices]
        cells = iter(entry["cells"])
        for nu in ANALYZE.NU_GRID:
            for tau, s in ANALYZE.TAU_GRID:
                cell = next(cells)
                assert (cell["nu"], cell["tau"]) == (nu, tau)
                direct = (ANALYZE.profile_value(x, nq, indices, nu/nu0, s)-ANALYZE.profile_value(r, nq, indices, nu/nu0, s))
                assert cell["M"] == pytest.approx(direct, rel=1e-12, abs=1e-9)
        # the critical nu makes the difference zero
        for row in entry["nu_of_tau"]:
            if row["star"] is not None:
                s = tau_s[row["tau"]]
                assert (ANALYZE.profile_value(x, nq, indices, row["star"]/nu0, s) - ANALYZE.profile_value(r, nq, indices, row["star"]/nu0, s)
                        == pytest.approx(0., abs=1e-8))
        # the data weight makes the difference zero at (nu0, tau .5)
        w = entry["data_weight"]
        if w["star"] is not None:
            assert w["star"]*entry["D"]+out["d_diff"]+out["d_curl"]+out["d_vol"]["0.5"] == pytest.approx(0., abs=1e-8)
    # tau = inf removes F_vol, nu = 0 removes F_curl: only the data and the diffusion remain
    cell = out["subsets"]["只有强度"]["cells"][-1]
    assert (cell["nu"], cell["tau"]) == (0., float("inf"))
    assert cell["M"] == pytest.approx(.5*nq*np.log(.5/2.)+2., rel=1e-12)


def test_region_dice_reads_the_right_labels():
    per35 = {str(i): i/100. for i in range(1, 36)}
    per4 = {"1": .1, "2": .2, "3": .3, "4": .4}
    d = ANALYZE.region_dice(per35, per4)
    assert d == dict(cortex_seg35=pytest.approx((.02+.21)/2.), wm_seg35=pytest.approx((.01+.20)/2.), cortex_seg4=.1, wm_seg4=.3)


def test_subset_names_are_the_groups_of_the_labels():
    import json
    groups = json.loads((ROOT/"configs"/"bayesreg"/"structure_groups_v1.json").read_text())["groupings"]["structure13_v1"]["groups"]
    by_name = {g["name"]: g["labels"] for g in groups}
    assert by_name["cerebral_wm"] == [1, 20] and by_name["cerebral_cortex"] == [2, 21]


def test_hf_stats():
    hf = load("hf_disp")
    n = 64
    z = np.arange(n, dtype=np.float64)
    k = 2.*np.pi/8.
    u = np.zeros((n, 32, 32, 3))
    u[..., 0] = np.sin(k*z)[:, None, None]
    region = np.zeros(u.shape[:3], dtype=bool)
    region[24:40, 12:20, 12:20] = True
    out = hf.hf_stats(u, dict(r=region), sigma=2.)["r"]
    # along z the Gaussian damps the sine by exp(-sigma^2 k^2/2); in y, x the field is constant, so the region must lie more
    # than 8 voxels (truncate 4 sigma) from the zero extension
    damp = np.exp(-2.*k**2)
    amp = float(np.sqrt(np.mean(np.sin(k*z[24:40])**2)))
    assert out["rms_total"] == pytest.approx(amp, rel=1e-12)
    assert out["rms_low"] == pytest.approx(damp*amp, rel=2e-2)
    assert out["rms_high"] == pytest.approx((1.-damp)*amp, rel=2e-2)
    constant = np.zeros_like(u)
    constant[..., 1] = 1.
    still = hf.hf_stats(constant, dict(r=region), sigma=2.)["r"]
    assert still["rms_high"] < 1e-3 and still["rms_low"] == pytest.approx(1., abs=1e-3)


def test_crossing_texts():
    assert ANALYZE.crossing_text(dict(star=1.5, side="above", always=None), "w_I") == "> 1.5"
    assert ANALYZE.crossing_text(dict(star=142., side="above", always=None), "w_I") == "> 142（超出）"
    assert ANALYZE.crossing_text(dict(star=None, side=None, always=False)) == "无解"
    assert ANALYZE.s_tau_text(ANALYZE.grid_crossings([.5, 1., 2.], [1., 1., -1.])) == "> 1.41"
    assert ANALYZE.s_tau_text(ANALYZE.grid_crossings([.5, 1., 2.], [-1., -1., 1.])) == "< 1.41"
    assert ANALYZE.s_tau_text(ANALYZE.grid_crossings([.5, 1., 2.], [1., 1., 1.])) == "网格内无解"
    assert ANALYZE.s_tau_text(ANALYZE.grid_crossings([.5, 1., 2.], [-1., -1., -1.])) == "网格内都偏爱 X"


def test_shared_sigma_is_the_minimum_of_the_joint_objective():
    """The post hoc L_shared: sigma~_S minimizes the joint objective with one sigma_S over the kept channels, and the factor K
    of the log term is the one of that objective (up to the constants of the a = b = 0 count N_q + 2)."""
    posthoc = load("posthoc_shared")
    nq = 1000.
    rho2 = np.array([2., .4, .3, .001])
    field = dict(rho2_B=rho2.tolist())
    indices = [1, 2, 3]
    _, term = posthoc.shared_terms(field, nq, indices)
    sigma2 = rho2[indices].sum()/(3*(nq+2.))
    assert term == pytest.approx(3*.5*nq*np.log(sigma2), rel=1e-14)

    def joint(value):
        return rho2[indices].sum()/(2.*value)+(3*nq/2.+1.)*np.log(value)
    best = rho2[indices].sum()/(3*nq+2.)
    assert joint(best) < min(joint(best*1.01), joint(best*.99))
    assert sigma2 == pytest.approx(best, rel=2e-3)


def test_shared_difference_against_a_direct_computation():
    posthoc = load("posthoc_shared")
    x = profile_field([.5, .4, .4, 9.], 3., 500., [40., 20., 10.])
    r = profile_field([2., 1., 1., 1.], 1., 100., [8., 4., 2.])
    nq, nu0 = 100., 21.85496512638118
    out = posthoc.shared_difference(x, r, nq, [1, 2, 3], nu0)

    def direct(f, nu_factor, s):
        i, t = posthoc.shared_terms(f, nq, [1, 2, 3])
        vol = 0. if s is None else float(ANALYZE.vol_table(f)[f["prior_pass"]["s_grid"].index(s)])
        return i+t+f["F_diff"]+nu_factor*f["F_curl"]+vol

    cells = iter(out["cells"])
    for nu in ANALYZE.NU_GRID:
        for _, s in ANALYZE.TAU_GRID:
            assert next(cells)["M"] == pytest.approx(direct(x, nu/nu0, s)-direct(r, nu/nu0, s), rel=1e-12, abs=1e-9)
    for row in out["nu_of_tau"]:
        if row["star"] is not None:
            s = dict(ANALYZE.TAU_GRID)[row["tau"]]
            assert direct(x, row["star"]/nu0, s)-direct(r, row["star"]/nu0, s) == pytest.approx(0., abs=1e-8)
