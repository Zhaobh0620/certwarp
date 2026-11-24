"""sigma_domain in the solver (work package SWLI; contract C04-6 / C11-8 gates 3, 5, 7 (solver part), 13, 14, 15;
draft SWL 2026-10-31).  Synthetic data only (solve/_swl_synthetic.py), numpy backend.

Gate 13 recomputes L^v of C11-8 in the test, independently of the solver's records:
    L^v(u, sigma) = sum_p [S_p(u)/(2 sigma_p^2) + (N_p/2 + a + 1) log sigma_p^2 + b/sigma_p^2] + Pi_D + Pi_J + Pi_C,
S_p from channel_point's rho (which does not depend on sigma or v), the state's v and the node weights in longdouble;
the prior blocks from the reference evaluation.  Within a level, after the level start: after every formal commit (u
step) and every M step (sigma step) L^v must not increase beyond 16 eps |L| + 1e-12 |L| (the C11 gate).
"""
import hashlib

import numpy as np
import pytest

from bayesreg import model as M
from bayesreg.learn import population as pop
from bayesreg.model import channel_point, support_level
from bayesreg.model.support import node_weights_masked
from bayesreg.solve import (Caps, GaussianMStep, NumpyBackend, SigmaMonitor, SolverConfig, SupportMStep,
                            eta_temporary, solve_multilevel)
from bayesreg.solve import multilevel as ML

from . import _swl_synthetic as S
from ..model import _swl_cases as cases

LD = np.longdouble


# ---- independent L^v (C11-8) --------------------------------------------------------------------------------------

def independent_sums(state, u, v=None):
    """{c: (S_c, N_c)} in longdouble (c = 0 the intensity with v = 1) from channel_point's rho."""
    point = channel_point(state, u)
    q = node_weights_masked(state)
    omega0 = LD(state.omega0)
    v = state.support_v if v is None else v
    out = {0: (np.sum(q*(point.rho[0]*point.rho[0]), dtype=LD)/omega0, np.sum(q, dtype=LD)/omega0)}
    for c in range(1, state.n_channels):
        qv = q*v[c-1]
        out[c] = (np.sum(qv*(point.rho[c]*point.rho[c]), dtype=LD)/omega0, np.sum(qv, dtype=LD)/omega0)
    return out


def parameters(state, sums):
    """[(S_p, N_p, sigma_p^2)] of the variance parameters of C11-8 (empty channels and an empty S left out)."""
    out = [(sums[0][0], sums[0][1], state.sigma2[0])]
    nonempty = [c for c in range(1, state.n_channels) if c not in state.support_empty]
    if state.sigma_domain == "support":
        out += [(sums[c][0], sums[c][1], state.sigma2[c]) for c in nonempty]
    elif nonempty:
        s = n = LD(0)
        for c in nonempty:
            s += sums[c][0]
            n += sums[c][1]
        out.append((s, n, state.sigma2[nonempty[0]]))
    return out


def l_v(state, u, a=0., b=0.):
    ev = M.evaluate(state, u)
    total = LD(ev.energies["diffusion"])+LD(ev.energies["logvol"])+LD(ev.energies["curl"])
    for s, n, sigma2 in parameters(state, independent_sums(state, u)):
        sigma2 = LD(sigma2)
        total += s/(2*sigma2)+(n/2+LD(a)+1)*np.log(sigma2)+LD(b)/sigma2
    return float(total)


# ---- a recorded solve -------------------------------------------------------------------------------------------

class Recorder:
    """Wraps multilevel.solve_round and multilevel.support_level: per round the state and the u of the start and of
    every formal commit; per level the support pass's (state, seed, SupportLevel)."""

    def __init__(self, monkeypatch):
        self.rounds, self.levels = [], []
        solve_round, support = ML.solve_round, ML.support_level

        def round_wrapper(backend, state, commit, **kwargs):
            tracker = kwargs["tracker"]
            seq = [commit.u]
            update = tracker.update

            def record(new):
                seq.append(new.u)
                update(new)
            tracker.update = record
            result = solve_round(backend, state, commit, **kwargs)
            self.rounds.append(dict(level=kwargs["context"]["level"], round=kwargs["context"]["round"], state=state,
                                    seq=seq))
            return result

        def support_wrapper(state, u, theta, **kwargs):
            level = support(state, u, theta, **kwargs)
            self.levels.append(dict(state=state, u=np.array(u), theta=theta, kwargs=kwargs, level=level))
            return level
        monkeypatch.setattr(ML, "solve_round", round_wrapper)
        monkeypatch.setattr(ML, "support_level", support_wrapper)


def solve(domain, levels, *, rounds=(1, 1, 1, 3), caps=Caps(3, 2), max_levels=4, monkeypatch=None, theta=None):
    native = levels[-1].grid
    eta = eta_temporary(levels[-1].tissue, levels[-1].mask, native)
    recorder = Recorder(monkeypatch) if monkeypatch is not None else None
    config = SolverConfig(rounds=rounds, caps=caps, sigma_domain=domain, max_levels=max_levels,
                          support_threshold=theta)
    result = solve_multilevel(NumpyBackend(), list(levels), eta=eta, sigma=GaussianMStep(), config=config)
    return result, recorder


@pytest.fixture(scope="module", params=["support", "support_shared"])
def recorded(request):
    mp = pytest.MonkeyPatch()
    try:
        _, levels, _ = S.four_levels_structure()
        result, recorder = solve(request.param, levels, monkeypatch=mp)
    finally:
        mp.undo()
    return request.param, levels, result, recorder


# ---- gate 13: solver-level monotonicity of L^v within a level -----------------------------------------------------

def test_l_v_does_not_increase_within_a_level(recorded):
    domain, _, result, rec = recorded
    assert result.native and result.support is not None and len(result.support) == 4
    checked = 0
    for ell in range(4):
        rows = [r for r in rec.rounds if r["level"] == ell]
        values = []
        for r in rows:
            values += [l_v(r["state"], u) for u in r["seq"]]          # first entry of round r > 0: the sigma step
        for before, after in zip(values, values[1:]):
            assert after <= before+(16*np.finfo(float).eps+1e-12)*abs(before), (domain, ell, values)
            checked += 1
    native = [r for r in rec.rounds if r["level"] == 3]
    assert len(native) == 3 and checked >= 6
    # the native last round's sigma_post (review SWLI minor 11): L^v(u_hat, sigma_post) <= L^v(u_hat, sigma_used)
    state, u_end = native[-1]["state"], native[-1]["seq"][-1]
    assert np.array_equal(u_end, result.commit.u)
    post = SupportMStep(domain).m_step(state, M.evaluate(state, u_end), rec.levels[3]["level"])
    assert post.record == result.sigma_post
    before, after = l_v(state, u_end), l_v(state.replace(sigma2=post.internal), u_end)
    assert after <= before+(16*np.finfo(float).eps+1e-12)*abs(before)


def test_live_model_states_per_round_match_brain(monkeypatch):
    """Review SWLI important 1: under support_shared every round holds as many host ModelStates (and beta arrays) of
    the level grid as under brain -- no second resident level state."""
    import gc
    from bayesreg.model import ModelState
    original = ML.solve_round
    counts = {}
    for domain in ("brain", "support_shared"):
        seen = []

        def wrapper(backend, state, commit, **kwargs):
            gc.collect()
            states = [o for o in gc.get_objects() if isinstance(o, ModelState) and o.grid is state.grid]
            seen.append((len(states), len({id(s.beta) for s in states}), sum(s.beta.nbytes for s in states)))
            return original(backend, state, commit, **kwargs)
        monkeypatch.setattr(ML, "solve_round", wrapper)
        _, levels, _ = S.four_levels_structure()
        solve(domain, levels, rounds=(1, 1, 1, 2), caps=Caps(1, 1))
        counts[domain] = seen
    assert len(counts["brain"]) == len(counts["support_shared"]) == 5
    assert counts["brain"] == counts["support_shared"]


# ---- gate 5: v frozen within a level, recomputed after a transfer --------------------------------------------------

def test_v_is_frozen_per_level_and_recomputed_from_the_seed(recorded):
    domain, _, result, rec = recorded
    assert [lv["level"].level for lv in rec.levels] == [0, 1, 2, 3]
    for ell, lv in enumerate(rec.levels):
        record = result.support[ell]
        level = lv["level"]
        assert record["v_sha256"] == level.v_sha256
        # every round of the level ran with this very v
        assert {r["state"].support_v_sha256 for r in rec.rounds if r["level"] == ell} == {level.v_sha256}
        # the same seed bytes and level arrays give the same bytes again
        again = support_level(lv["state"], lv["u"], lv["theta"], **lv["kwargs"])
        assert again.v.tobytes() == level.v.tobytes() and again.s_ld == level.s_ld
        assert hashlib.sha256(np.ascontiguousarray(lv["u"]).tobytes()).hexdigest() == record["seed_state_hash"]
        if ell > 0:
            transfer = result.transfers[ell-1]
            accepted = transfer["rows"][transfer["accepted_k"]]
            assert accepted["cert"]["state_hash"] == record["seed_state_hash"]   # the certified transfer seed
    assert len({lv["level"].v_sha256 for lv in rec.levels}) == 4                 # one v per level
    # the native sigma_post and the output state use the native v
    native = rec.levels[3]["level"]
    assert result.state.support_v_sha256 == native.v_sha256
    policy = SupportMStep(domain)
    post = policy.m_step(result.state, M.evaluate(result.state, result.commit.u), native)
    assert post.record == result.sigma_post


# ---- gate 7 (solver part): empty support records -------------------------------------------------------------------

def test_empty_support_records(recorded):
    domain, _, result, _ = recorded
    for record in result.support:
        assert record["empty_support_channels"] == [3] and record["N_c"][2] == 0.
        assert record["sigma2_start"][3] is None
        if domain == "support_shared":
            assert record["sigma2_S_start"] is not None and record["N_S"] == record["N_c"][0]+record["N_c"][1]
            assert record["sigma2_start"][1] == record["sigma2_start"][2] == record["sigma2_S_start"]
    for row in result.rounds:
        assert row["sigma2_used"][3] is None and row["sigma_tilde2"][2] is None
        if row.get("sigma2_post") is not None:
            assert row["sigma2_post"][3] is None
        if domain == "support_shared":
            assert row["sigma2_used"][1] == row["sigma2_S_used"]
            assert "sigma2_S_post" in row
        else:
            assert "sigma2_S_used" not in row
    assert result.sigma_post[3] is None
    trace = result.sigma["trace"]
    assert all(t["sigma2"][3] is None and t["ratio_to_intensity"][2] is None for t in trace)
    assert result.timing["levels"][0]["support_cpu_seconds"] >= 0


def _zero_rows(monkeypatch, channels):
    original = ML.support_level

    def wrapper(state, u, theta, **kwargs):
        point = channel_point(state, u)

        def rows(c):
            rho = np.array(point.rho[c])
            if c in channels:
                rho[:] = 0.
            return rho
        return original(state, u, theta, rows=rows, **kwargs)
    monkeypatch.setattr(ML, "support_level", wrapper)


@pytest.mark.parametrize("domain,channels,label", [("support", (2,), "channel 2"),
                                                   ("support_shared", (1, 2), "channel S")])
def test_zero_residual_on_the_support_fails_the_level_start(monkeypatch, domain, channels, label):
    _, levels, _ = S.four_levels_structure()
    _zero_rows(monkeypatch, channels)
    result, _ = solve(domain, levels, rounds=(1, 1, 1, 1), max_levels=1)
    assert result.status == "level_failed" and "level 0 start" in result.reason
    assert domain in result.reason and label in result.reason


@pytest.mark.parametrize("final", [False, True])
def test_zero_residual_in_the_m_step(monkeypatch, final):
    _, levels, _ = S.four_levels_structure()
    original = ML.channel_mean_square

    def zero_channel_2(state, point):
        out = original(state, point)
        out[2] = 0.
        return out
    monkeypatch.setattr(ML, "channel_mean_square", zero_channel_2)
    result, _ = solve("support", levels, rounds=(1 if final else 2, 1, 1, 1), max_levels=1)
    if final:                       # the last round of the run: sigma_post only recorded (C11-6), ending unchanged
        assert result.status == "stopped_before_native"
        row = result.rounds[-1]
        assert row["sigma2_post"] is None and "support" in row["sigma2_post_error"]
        assert "channel 2" in row["sigma2_post_error"]
    else:
        assert result.status == "level_failed" and "M step" in result.reason and "channel 2" in result.reason


# ---- gate 14: SigmaMonitor with null sigma --------------------------------------------------------------------------

def test_monitor_null_ratios_unit():
    mon = SigmaMonitor("support")
    mon.update([1., None, .5], level=0, round_=0, kind="level_start", used=True)
    mon.update([1., .01, .5], level=1, round_=0, kind="level_start", used=True)       # empty at 0, non-empty at 1
    mon.update([1., .001, None], level=1, round_=0, kind="m_step", used=True)          # and back (channel 2)
    mon.update([1., .00001, .3], level=1, round_=1, kind="m_step", used=True)
    trace = mon.record()["trace"]
    assert trace[1]["ratio_to_previous"][1] is None and trace[2]["ratio_to_previous"][2] is None
    assert trace[0]["ratio_to_intensity"] == [None, .5]
    assert not trace[2]["collapse"] and trace[2]["ratio_to_previous"][1] == pytest.approx(.1)
    assert trace[3]["collapse"] and mon.record()["events"][0]["channels"] == [1]
    assert trace[3]["ratio_to_previous"][2] is None
    shared = SigmaMonitor("support_shared")
    shared.update([1., None, None], level=0, round_=0, kind="level_start", used=True, sigma2_S=None)
    shared.update([1., .2, .2], level=1, round_=0, kind="level_start", used=True, sigma2_S=.2)
    shared.update([1., .01, .01], level=1, round_=0, kind="m_step", used=True, sigma2_S=.01)
    rows = shared.record()["trace"]
    assert rows[0]["sigma2_S"] is None and rows[0]["sigma2_S_ratio_to_intensity"] is None
    assert rows[1]["sigma2_S_ratio_to_previous"] is None and rows[2]["collapse"]
    assert shared.record()["sigma_collapse_suspected"] is True


NARROW = (((8., 9., 10.), 4., .6), ((16., 14., 12.), 3.5, .5), ((9.4, 9.4, 15.6), 1.2, .6))
ALL_NARROW = (((9.4, 9.4, 15.6), 1.2, .6), ((15.6, 9.4, 9.4), 1.2, .6), ((9.4, 15.6, 9.4), 1.2, .6))


def test_monitor_in_the_solver_across_empty_and_nonempty_levels():
    _, levels, _ = S.four_levels_structure(blobs=NARROW)
    result, _ = solve("support", levels, rounds=(1, 1, 1, 1), max_levels=2)
    empty = [r["empty_support_channels"] for r in result.support]
    assert 3 in empty[0] and 3 not in empty[1]
    trace = result.sigma["trace"]
    starts = [t for t in trace if t["kind"] == "level_start"]
    assert starts[0]["sigma2"][3] is None and starts[1]["sigma2"][3] is not None
    assert starts[1]["ratio_to_previous"][3] is None
    assert result.status == "stopped_before_native"


def test_all_structure_channels_empty_shared():
    _, levels, _ = S.four_levels_structure(blobs=ALL_NARROW)
    result, _ = solve("support_shared", levels, rounds=(1, 1, 1, 1), max_levels=1)
    record = result.support[0]
    assert record["empty_support_channels"] == [1, 2, 3] and record["sigma2_S_start"] is None
    assert record["N_S"] == 0.
    assert result.sigma["trace"][0]["sigma2_S"] is None
    row = result.rounds[0]
    assert row["sigma2_used"][1:] == [None, None, None] and row["sigma2_S_used"] is None
    assert row["sigma2_S_post"] is None


# ---- gate 3: M step, level start ------------------------------------------------------------------------------------

def _field(grid, seed=4, scale=.08):
    from jd_curl.three_d.boundary import inject_free
    rng = np.random.default_rng(seed)
    return inject_free(scale*min(grid.spacing)*rng.normal(size=3*grid.n_free_scalar), grid)


@pytest.mark.parametrize("domain", ["support", "support_shared"])
@pytest.mark.parametrize("ab", [(0., 0.), (3., 2e-4)])
def test_m_step_is_the_minimizer(domain, ab):
    a, b = ab
    grid, state, _, _ = cases.template()
    level, st = cases.with_support(state, np.zeros(grid.shape+(3,)), sigma_domain=domain)
    u = _field(grid)
    policy = SupportMStep(domain, a=a, b=b)
    post = policy.m_step(st, M.evaluate(st, u), level)
    sums = independent_sums(st, u)
    checked = st.replace(sigma2=post.internal)
    for s, n, sigma2 in parameters(checked, sums):
        s, n = float(s), float(n)
        expected = (s+2*b)/(n+2*a+2)
        assert abs(sigma2-expected) <= 1e-12*expected
        terms = (-s/(2*sigma2**2), (n/2+a+1)/sigma2, -b/sigma2**2)
        assert abs(sum(terms)) <= 1e-12*sum(abs(t) for t in terms)                # analytic dL/dsigma^2 = 0

        def L(x):
            return s/(2*x)+(n/2+a+1)*np.log(x)+b/x
        h = 1e-4*sigma2
        assert abs((L(sigma2+h)-L(sigma2-h))/(2*h)) <= 1e-7*sum(abs(t) for t in terms)
    assert post.record[3] is None


def _literal_start(state, u, v, domain, a=0., b=0.):
    """C11-8 "层首的算式" written out: rho_c filled (here from channel_point), (q v_c)(rho_c rho_c), one np.sum in
    longdouble over the whole array, / omega0; support_shared adds the longdouble S_c and N_c in ascending c."""
    point = channel_point(state, u)
    q = node_weights_masked(state)
    omega0 = LD(state.omega0)
    rho = [np.array(point.rho[c]) for c in range(state.n_channels)]
    s0 = np.sum(q*(rho[0]*rho[0]), dtype=LD)/omega0
    n_q = np.sum(q, dtype=LD)/omega0
    out = [(float(s0)+2*b)/(float(n_q)+2*a+2)]
    s_c, n_c = [], []
    for c in range(1, state.n_channels):
        qv = q*v[c-1]
        s_c.append(np.sum(qv*(rho[c]*rho[c]), dtype=LD)/omega0)
        n_c.append(np.sum(qv, dtype=LD)/omega0)
    nonempty = [c for c in range(1, state.n_channels) if v[c-1].any()]
    if domain == "support":
        out += [(float(s_c[c-1])+2*b)/(float(n_c[c-1])+2*a+2) if c in nonempty else None
                for c in range(1, state.n_channels)]
    else:
        s = n = LD(0)
        for c in nonempty:
            s += s_c[c-1]
            n += n_c[c-1]
        shared = (float(s)+2*b)/(float(n)+2*a+2) if nonempty else None
        out += [shared if c in nonempty else None for c in range(1, state.n_channels)]
    return out


@pytest.mark.parametrize("domain", ["support", "support_shared"])
def test_level_start_sigma_is_bitwise_the_literal_sum(domain):
    grid, state, _, _ = cases.template()
    u = _field(grid, seed=11, scale=.05)
    level = support_level(state, u, .05, sigma_domain=domain)
    start = SupportMStep(domain, a=1.5, b=1e-4).level_start(level)
    assert start.record == _literal_start(state, u, level.v, domain, a=1.5, b=1e-4)
    # N_q, N_c, N_S as C11-8 writes them
    q = node_weights_masked(state)
    assert level.n_q_ld == np.sum(q, dtype=LD)/LD(state.omega0)
    assert level.n_q == ML.n_q(state)
    # flipping one bit of v changes sigma_start (of channel 1, or of S)
    sums = independent_sums(state, u, level.v)
    point = channel_point(state, u)
    k = int(np.flatnonzero(level.v[0] & (np.abs(point.rho[1]) > 1e-6))[0])
    flipped = np.array(level.v)
    flipped[0, k] = 0
    assert _literal_start(state, u, flipped, domain)[1] != SupportMStep(domain).level_start(level).record[1]
    assert sums[1][1] > 0


@pytest.mark.parametrize("domain", ["support", "support_shared"])
def test_pseudo_count_round_trip_with_support_counts(domain):
    grid, state, _, _ = cases.template()
    level, st = cases.with_support(state, np.zeros(grid.shape+(3,)), sigma_domain=domain)
    a, b = 4.5, 3e-4
    n0, s0 = pop.pseudo_count_form(a, b)
    a2, b2 = pop.inverse_gamma_from_pseudo_counts(n0, s0)
    assert abs(a2-a) <= 1e-12*a and abs(b2-b) <= 1e-12*b
    u = _field(grid)
    post = SupportMStep(domain, a=a, b=b).m_step(st, M.evaluate(st, u), level)
    ms = ML.channel_mean_square(st, M.evaluate(st, u))
    if domain == "support":
        numerator, count = ms[1], level.n_c[0]
    else:
        total = LD(0)
        for c in level.nonempty:
            total += LD(ms[c])
        numerator, count = float(total), level.n_s()
    pseudo = (numerator+n0*s0)/(count+n0)
    assert abs(post.record[1]-pseudo) <= 2*np.spacing(pseudo)


# ---- gate 15: tissue counts ------------------------------------------------------------------------------------------

def test_counts_match_an_independent_count():
    grid, state, _, _ = cases.template()
    u = _field(grid, seed=5, scale=.1)
    level = support_level(state, u, .05, sigma_domain="support")
    point = channel_point(state, u)
    t = np.sqrt(np.float64(.05))
    for c in range(1, state.n_channels):
        g = state.fixed_masked[c]
        ghat = g+point.rho[c]
        f, m = g > t, ghat > t
        assert level.support_nodes[c-1] == int(np.count_nonzero(f | m))
        assert level.fixed_only[c-1] == int(np.count_nonzero(f & ~m))
        assert level.moving_only[c-1] == int(np.count_nonzero(m & ~f))
        assert level.near_threshold[c-1] == int(np.count_nonzero(np.abs(ghat-t) <= 1e-12))
