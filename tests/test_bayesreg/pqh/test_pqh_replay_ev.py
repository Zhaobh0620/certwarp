"""The replay gate with EV's merged per-pair and class-B functions (bayesreg.eval.holdout.pq_pair_acceptance, pq_b_class; main 6a44598) on the PQ package's synthetic world: the
development reading is replayed bit for bit.  The truth still comes from the test's reader (EV's pq_dev_truth reads the real development truth through the entry's reader; the gate itself
runs it).  The interface check of alignment item 6, on arrays only (CPU, tmp_path)."""
from . import test_pqh_replay as T
from .test_pqh_replay import world  # noqa: F401  (the fixture)

hp, rp = T.hp, T.rp


def test_evs_functions_replay_the_development_reading_bit_for_bit(world):  # noqa: F811
    from bayesreg.eval import holdout
    functions = rp.ev_functions(holdout)
    assert sorted(functions) == ['b_class', 'pair', 'truth']
    result = T.run(world, ev=dict(truth=T.truth_of, pair=functions['pair'], b_class=functions['b_class']))
    assert result['error'] is None, result['error']
    assert result['reading'] == hp.REPLAY_READING_PASS.replace('19', str(len(world.w.ctx.shorts))), (result['pairs'], result['b_class']['differences'])
