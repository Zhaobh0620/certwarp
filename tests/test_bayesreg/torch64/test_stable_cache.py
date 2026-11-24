"""C08-8(d) cache of the CPU fallback (work package CPUPAR): the base slot and the reduction memo.

Every comparison here takes the CPU path (the points carry no device evaluation: a device anomaly, C08-8(e)), so
the comparator's CPU reference is exercised directly; a spy module counts the CPU evaluations and reductions.
- the base of consecutive fallbacks is evaluated once, whatever the trials in between and even with no LRU
  (cache_size 0 or 1): a fallback re-evaluates only the candidate;
- each key component (state hash, ModelState identity: sigma, w, eta, tile, the ModelState object itself; energy
  type; reference v) forces a re-evaluation of the base;
- a second fallback on the same pair under the same key reuses the CPU Reduction; another key or trial recomputes;
- the GF recheck point is kept as the base;
- the decisions equal those of the uncached reference (bayesreg.model.evaluate + stable_reduction).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from bayesreg import model as M
from bayesreg.backends.torch64 import stable as S
from bayesreg.solve.proximal import PROX_BLOCKS

from ._cases import case, smooth_field

LD = np.longdouble


class Spy(SimpleNamespace):
    """The comparator's CPU model: bayesreg.model with counted evaluations and reductions."""

    def __init__(self):
        super().__init__(evaluated=[], reductions=0)

    def evaluate(self, state, u):
        self.evaluated.append(S.state_hash(u))
        return M.evaluate(state, u)

    def stable_reduction(self, state, base, trial):
        self.reductions += 1
        return M.stable_reduction(state, base, trial)

    def gradient(self, state, ev):
        return M.gradient(state, ev)


def host(state, u):
    """A point without a device evaluation (C08-8(e): the comparison goes to the CPU reference)."""
    u = np.array(u, dtype=np.float64, order="C")
    u.flags.writeable = False
    return S.HostEvaluation(state, u, LD(0), 0., {}, anomaly="no device (test)")


@pytest.fixture(scope="module")
def setup():
    state, u, _ = case("small")
    g = state.grid
    base = u
    trials = [u+smooth_field(g, a, s) for a, s in ((.02, 3), (-.015, 4), (.01, 5), (.025, 6))]
    return state, base, trials


def no_extra():
    return S.ExtraBlocks({name: LD(0) for name in PROX_BLOCKS}, LD(0), LD(0))


@pytest.mark.parametrize("cache_size", [0, 1, 2])
def test_fallback_evaluates_the_base_once(setup, cache_size):
    state, u, trials = setup
    spy = Spy()
    comparator = S.StableComparator(cache_size=cache_size, cpu_model=spy)
    mid = S.model_identity(state)
    base = host(state, u)
    for k, w in enumerate(trials):
        trial = host(state, w)
        red = comparator.compare(state, None, base, trial, model_id=mid, context="pg", required=0.)
        assert red.path == "cpu" and red.info["fallback"]
        ref = M.stable_reduction(state, M.evaluate(state, u), M.evaluate(state, w))
        assert red.ared == ref.ared and red.guard == ref.guard and red.delta == ref.delta
    assert spy.evaluated == [S.state_hash(u)]+[S.state_hash(w) for w in trials]     # the base once
    assert comparator.stats.base_hits == len(trials)-1 and spy.reductions == len(trials)
    summary = comparator.stats.summary()
    assert summary["cpu_evaluations"] == 1+len(trials) and summary["cpu_base_slot_hits"] == len(trials)-1


def test_every_key_component_reevaluates_the_base(setup):
    state, u, trials = setup
    spy = Spy()
    comparator = S.StableComparator(cache_size=0, cpu_model=spy)

    def base_evaluations(st, base_u, trial_u, **kw):
        before = list(spy.evaluated)
        comparator.compare(st, None, host(st, base_u), host(st, trial_u), model_id=S.model_identity(st),
                           context="outer_F" if "energy" not in kw else "outer_E", **kw)
        new = spy.evaluated[len(before):]
        return new.count(S.state_hash(base_u))

    w = trials[0]
    assert base_evaluations(state, u, w) == 1
    assert base_evaluations(state, u, trials[1]) == 0                                 # same key: the slot
    assert base_evaluations(state, trials[2], w) == 1                                 # state hash
    assert base_evaluations(state, u, w) == 1                                         # the slot moved on
    assert base_evaluations(state, u, w, energy="E_v", reference="v1", extra=no_extra()) == 1    # energy type
    assert base_evaluations(state, u, w, energy="E_v", reference="v1", extra=no_extra()) == 0
    assert base_evaluations(state, u, w, energy="E_v", reference="v2", extra=no_extra()) == 1    # reference v
    changed = dict(sigma=state.replace(sigma2=state.sigma2*1.5), w=state.replace(weights=state.weights*.9),
                   alpha=state.replace(alpha=state.alpha*2), nu=state.replace(nu=state.nu*2),
                   tau=state.replace(tau2=state.tau2*1.1), tile=state.replace(tile_size=5))
    for name, st in changed.items():
        assert base_evaluations(st, u, w) == 1, name
        assert base_evaluations(st, u, trials[1]) == 0, name
    twin = state.replace()                                  # same identity, another ModelState object
    assert S.model_identity(twin) == S.model_identity(state)
    assert base_evaluations(state, u, w) == 1
    assert base_evaluations(twin, u, w) == 1
    threads = state.replace(cpu_threads=3)                  # the thread count is not in the identity ...
    assert S.model_identity(threads) == S.model_identity(state)
    assert base_evaluations(threads, u, w) == 1             # ... but a hit needs the very same ModelState object


def test_reduction_memo_same_pair_same_key(setup):
    state, u, trials = setup
    spy = Spy()
    comparator = S.StableComparator(cache_size=2, cpu_model=spy)
    mid = S.model_identity(state)
    base, trial = host(state, u), host(state, trials[0])
    first = comparator.compare(state, None, base, trial, model_id=mid, context="pg", required=1e-9)
    second = comparator.compare(state, None, base, trial, model_id=mid, context="outer_F")
    assert spy.reductions == 1 and comparator.stats.reduction_hits == 1
    assert first.ared == second.ared and first.delta == second.delta and first.blocks == second.blocks
    comparator.compare(state, None, base, trial, model_id=mid, context="outer_E", energy="E_v", reference="v",
                       extra=no_extra())
    assert spy.reductions == 2                                                       # another key: recomputed
    comparator.compare(state, None, base, host(state, trials[1]), model_id=mid, context="pg")
    assert spy.reductions == 3                                                       # another trial
    comparator.compare(state, None, base, trial, model_id=mid, context="pg")
    assert spy.reductions == 4                                                       # only the latest pair is kept
    assert comparator.stats.summary()["cpu_reduction_memo_hits"] == 1
    comparator.clear()
    assert comparator.base is None and comparator._reduction is None and not comparator.cache


def test_gf_recheck_point_becomes_the_base(setup):
    state, u, trials = setup
    spy = Spy()
    comparator = S.StableComparator(cache_size=0, cpu_model=spy)
    mid = S.model_identity(state)
    base = host(state, u)
    g = M.gradient(state, M.evaluate(state, u))
    from bayesreg.solve.stopping import gf_b
    G = gf_b(g, state.grid.omega, state.omega0)
    used, row = comparator.confirm_gradient_norm(state, base, G, G, model_id=mid)
    assert row is not None and used == G
    comparator.compare(state, None, base, host(state, trials[0]), model_id=mid, context="pg")
    assert spy.evaluated == [S.state_hash(u), S.state_hash(trials[0])]
