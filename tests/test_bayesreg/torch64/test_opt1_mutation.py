"""OPT1 mutation tests (review docs/bayesreg/reviews/OPT1_复核_20261030.md item 3): the bitwise checks of
test_opt1_bitwise.py catch a change of rounding order in the OPT1 paths.

Each mutation is injected with monkeypatch (no implementation file is changed) and changes only the order of
floating-point operations, so the mutated values stay within rounding of the reference while some bits differ:
(a) C3 speculative gradient: `evaluate` adds the channel term to its gradient field last instead of first (`gradient`
    adds it first).  The unit pipeline of test_opt1_bitwise differs from the pre-OPT1 reference in the gradients that
    use the field, and only there; the run_pg sequence (gradient norms and log rows) differs.
(b) C2a Jacobian cache: the cached J is the row-0 cofactor expansion summed in reverse order (F_02 A_02 + F_01 A_01 +
    F_00 A_00 instead of F_00 A_00 + F_01 A_01 + F_02 A_02; log J, when kept, is torch.log of it).  The device tile sums
    of the comparison, Delta F_B and the comparator results differ, and only those; in the run_pg sequence the
    recorded device Delta F_B differ (the logged float64 A_hat does not resolve the change on that problem, so the
    run_pg test compares Delta F_B in longdouble too, OPT1b).
Without a mutation the same checks pass (test_opt1_bitwise.py); here they must fail.  cpu device only.
"""
import numpy as np
import pytest
import torch

import bayesreg.backends.torch64 as T
from bayesreg.backends.torch64 import model as TM
from bayesreg.backends.torch64 import stable as S

from . import _opt1_reference as R
from . import test_opt1_bitwise as B

CORNERS = B.CORNERS


def channel_term_last(mp):
    """(a): evaluate's speculative field gets the channel term after every chunk's flux."""
    original_evaluate, original_add = TM.evaluate, TM._channel_add

    def evaluate(dstate, u):
        deferred = []
        mp.setattr(TM, "_channel_add", lambda *args: deferred.append(args))     # only evaluate's call (C3 field)
        try:
            ev = original_evaluate(dstate, u)
        finally:
            mp.setattr(TM, "_channel_add", original_add)
        for args in deferred:
            original_add(*args)                                 # in place into ev._speculative["out"]
        return ev
    mp.setattr(TM, "evaluate", evaluate)
    mp.setattr(T, "evaluate", evaluate)


def reversed_cached_J(mp):
    """(b): every cached J (Gauss points and corners) replaced by the reverse-order row-0 expansion."""
    original_evaluate = TM.evaluate

    def evaluate(dstate, u):
        ev = original_evaluate(dstate, u)
        if isinstance(ev, TM.TorchEvaluation) and ev._jacobians:
            entries = ev._jacobians["entries"]
            for key, (J, logJ) in list(entries.items()):
                corners = isinstance(key, tuple)
                chunk = dstate.q1.chunks[key[1] if corners else key]
                _, F, A, J0 = TM._geometry(dstate, ev.u_cm, chunk, dstate.q1_corners if corners else None)
                assert torch.equal(J0, J)
                Jr = F[0, :, 2]*A[0, :, 2]+F[0, :, 1]*A[0, :, 1]+F[0, :, 0]*A[0, :, 0]
                entries[key] = (Jr, None if logJ is None else torch.log(Jr))
        return ev
    mp.setattr(TM, "evaluate", evaluate)
    mp.setattr(T, "evaluate", evaluate)


MUTATIONS = dict(speculative=channel_term_last, jacobian=reversed_cached_J)
TOUCHED = dict(speculative={"gradient_base", "gradient_trial", "gradient_permuted"},
               jacobian={"tiles", "delta", "compare"})


@pytest.mark.parametrize("option", ["default", "log"])
@pytest.mark.parametrize("points", ["gauss27", CORNERS])
@pytest.mark.parametrize("mutation", list(MUTATIONS))
def test_unit_checks_catch_a_rounding_order_mutation(mutation, points, option, monkeypatch):
    if mutation == "speculative" and option == "log":
        pytest.skip("log J does not enter the speculative gradient")
    state, u, u_trial, chunk = B.unit_case("box_c3", points)
    options = B.OPTIONS[option]
    with pytest.MonkeyPatch.context() as mp:
        MUTATIONS[mutation](mp)
        new, ds, tb, tt = B.pipeline(state, u, u_trial, "cpu", chunk, options)
        mutated_gradient = T.to_numpy(T.gradient(ds, tt))
        mutated_delta, _ = S.gpu_fb_change(ds, tb, tt)
    old, *_ = B.pipeline(state, u, u_trial, "cpu", chunk, options, monkeypatch=monkeypatch)
    differ = {key for key in new if new[key] != old[key]}
    assert differ, "the mutation went undetected"
    assert differ <= TOUCHED[mutation], differ                   # nothing the mutation does not touch moved
    # rounding only: the mutated values are within rounding of the reference
    ref_ds = T.TorchModelState(state, "cpu", chunk_cells=chunk, jacobian_cache_slots=0, speculative_gradient=False)
    rb, rt = R.evaluate(ref_ds, u), R.evaluate(ref_ds, u_trial)
    if mutation == "speculative":
        reference = T.to_numpy(R.gradient(ref_ds, rt))
        assert not np.array_equal(mutated_gradient, reference)
        assert np.max(np.abs(mutated_gradient-reference)) <= 1e-12*np.max(np.abs(reference))
    else:
        reference, _ = R.gpu_fb_change(ref_ds, rb, rt)
        assert mutated_delta != reference
        assert abs(float(mutated_delta-reference)) <= S.gpu_guard(rb.energy, rt.energy)


@pytest.mark.parametrize("mutation", list(MUTATIONS))
def test_run_pg_checks_catch_a_rounding_order_mutation(mutation, monkeypatch):
    """The checks of test_run_pg_sequence_bitwise (option (a), gauss27): (a) changes the gradient norms and the log
    rows; (b) changes the recorded device Delta F_B.  On this problem (b) does not reach the logged rows: its effect
    on the tile sums is below the resolution of the logged float64 A_hat, which is why the run_pg test also compares
    every Delta F_B in longdouble (OPT1b)."""
    with pytest.MonkeyPatch.context() as mp:
        MUTATIONS[mutation](mp)
        backend, result, rows, _, deltas = B.solve_recording("a", "cpu", "gauss27", mp)
    R.patch(monkeypatch)
    ref_backend, ref_result, ref_rows, _, ref_deltas = B.solve_recording("a", "cpu", "gauss27", monkeypatch)
    monkeypatch.undo()
    if mutation == "speculative":
        assert [B.canonical(r) for r in rows] != [B.canonical(r) for r in ref_rows], "the mutation went undetected"
        assert B.canonical(list(result.gradient_norms)) != B.canonical(list(ref_result.gradient_norms))
    else:
        assert len(deltas) == len(ref_deltas) and deltas != ref_deltas, "the mutation went undetected"
