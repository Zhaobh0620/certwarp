"""PQ (contract C22.Q7 "评测与判定代码"; plan.md sections 5, 6): experiments/pq/pq_stats.py on synthetic numbers (CPU, no file).

- AUSE: e itself as the score gives 0; a strictly increasing transform of the score does not change it; a hand example with tie groups equals the mean over 20000 random orders inside the ties (seed fixed,
  within 4 Monte Carlo standard errors); a binary score equals the hand linear interpolation inside its two groups; X_lex is ordered by its two keys, not by a weighted float;
- AUROC: the Mann-Whitney value with average ranks;
- B class: p1_core.block_interval called as it is, bit for bit; R and the intersection-union rule;
- A class: pair i repeated w_i times gives the weighted average-rank Spearman of the hand formula; X constant gives 0; the void draws (fewer than 5 pairs with weight, no structure) are counted;
- the reading of plan.md section 6: one synthetic case for every row, and the intersection-union rule both ways;
- the reliability bins and ECE, the constant prediction K (pairs with |j - i| >= 2).
"""
import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

from . import _world as W  # noqa: F401  (puts experiments/pq on sys.path)

import pq_stats as stats  # noqa: E402
import p1_core  # noqa: E402


# ---- AUSE ------------------------------------------------------------------------------------------------------------------------------------------
def test_ause_of_the_error_itself_is_zero():
    rng = np.random.default_rng(0)
    for n, rate in ((1000, .1), (777, .5), (101, .99), (50, .02)):
        e = rng.uniform(size=n) < rate
        e[0] = True
        assert stats.ause([e.astype(np.float32)], e) == 0.
        assert stats.ause([e.astype(np.uint8)], e) == 0.


def test_ause_is_invariant_under_a_strictly_increasing_transform():
    rng = np.random.default_rng(1)
    e = rng.uniform(size=3000) < .2
    s = np.round(rng.uniform(size=3000)+.5*e, 2)                                     # ties on purpose
    a = stats.ause([s], e)
    assert a == stats.ause([np.exp(3*s)], e) == stats.ause([7.*s-2.], e) == stats.ause([np.arctan(s)], e)
    assert a != stats.ause([-s], e)                                                  # a decreasing one does change it
    t = (s.astype(np.float32), rng.integers(0, 3, 3000).astype(np.uint8))
    assert stats.ause([t[1], t[0]], e) == stats.ause([t[1].astype(np.float64)*10, np.exp(t[0].astype(np.float64))], e)    # each key of a lexicographic score


def mc_ause(score, e, n_perm, seed):
    """The tie rule's meaning: remove in score order, a random order inside each tie group; mean AUSE over n_perm orders, and its standard error."""
    rng = np.random.default_rng(seed)
    N, n1 = e.size, int(e.sum())
    m = (np.arange(100)*N)//100
    r_o = np.maximum(n1-m, 0)/(N-m)
    values = []
    for _ in range(n_perm):
        order = np.lexsort((rng.uniform(size=N), -score))
        removed = np.concatenate([[0], np.cumsum(e[order])])[m]
        values.append(np.mean((n1-removed)/(N-m)-r_o))
    values = np.array(values)
    return values.mean(), values.std(ddof=1)/np.sqrt(n_perm)


def test_ause_tie_rule_equals_the_monte_carlo_mean_of_random_orders_inside_ties():
    score = np.array([3]*7+[2]*11+[1]*5+[0]*9, dtype=np.float64)                    # four tie groups, N = 32
    e = np.zeros(score.size, dtype=bool)
    e[[0, 2, 5, 7, 8, 9, 12, 17, 20, 25, 31]] = True
    mean, se = mc_ause(score, e, 20000, 20261107)
    value = stats.ause([score], e)
    assert abs(value-mean) <= 4*se and se > 0


def test_ause_of_a_binary_score_is_the_hand_linear_interpolation():
    rng = np.random.default_rng(2)
    N = 257
    h = (rng.uniform(size=N) < .3).astype(np.uint8)
    e = np.where(h == 1, rng.uniform(size=N) < .6, rng.uniform(size=N) < .1)
    n_a, e_a = int(h.sum()), int(e[h == 1].sum())
    n_b, e_b = N-n_a, int(e[h == 0].sum())
    n1 = e_a+e_b
    total = 0.
    for j in range(100):
        m = (j*N)//100
        E = m*e_a/n_a if m <= n_a else e_a+(m-n_a)*e_b/n_b                         # inside the first group, then inside the second
        total += (n1-E)/(N-m)-max(n1-m, 0)/(N-m)
    assert stats.ause([h], e) == pytest.approx(total/100, rel=1e-13, abs=1e-15)


def test_x_lex_is_ordered_by_its_two_keys_not_by_a_float_composite():
    h = np.array([1, 1, 0, 0, 1, 0], dtype=np.uint8)
    H2 = np.array([.2, .9, .95, .1, .2, .3], dtype=np.float32)
    codes = stats.group_codes([h, H2])
    removal = np.argsort(-codes, kind='stable')
    assert [(int(h[i]), float(H2[i])) for i in removal] == sorted([(int(a), float(b)) for a, b in zip(h, H2)], reverse=True)
    assert codes[0] == codes[4] and len(set(codes.tolist())) == 5                     # the tie (1, .2) is one group
    big = np.array([1, 1], dtype=np.uint8), np.array([1.0, 1.0+2**-23], dtype=np.float32)
    composite = (big[0].astype(np.float32)*np.float32(1e8)+big[1]).astype(np.float32)
    assert composite[0] == composite[1] and len(set(stats.group_codes([big[0], big[1]]).tolist())) == 2     # a weighted float in float32 would merge what the keys tell apart


def test_auroc_is_the_mann_whitney_value_with_average_ranks():
    rng = np.random.default_rng(3)
    e = rng.uniform(size=500) < .3
    s = np.round(rng.normal(size=500)+e, 1)
    ranks = rankdata(s)
    n1, n0 = int(e.sum()), int((~e).sum())
    expected = (ranks[e].sum()-n1*(n1+1)/2)/(n1*n0)
    assert stats.auroc_from_table(*stats.tie_table([s], e)) == pytest.approx(expected, rel=1e-13)
    assert stats.auroc_from_table(*stats.tie_table([e.astype(float)], e)) == 1.


def test_score_record_none_without_an_error():
    rec = stats.score_record([np.arange(5.)], np.zeros(5, dtype=bool))
    assert rec['ause'] is None and rec['auroc'] is None and rec['n_errors'] == 0


# ---- B class ---------------------------------------------------------------------------------------------------------------------------------------
def b_table(factor, *, q=None, override=None):
    rng = np.random.default_rng(4)
    q = rng.uniform(.01, .03, 19) if q is None else q
    table = {k: list(q*factor+rng.normal(0, 1e-4, 19)) for k in stats.JUDGED_BASELINES}
    table['q'] = list(q)
    table.update(override or {})
    return table


def test_b_class_calls_block_interval_as_it_is():
    table = b_table(1.3)
    index = np.arange(19)
    b = stats.b_class(table, index)
    for k in stats.JUDGED_BASELINES:
        delta = np.asarray(table[k])-np.asarray(table['q'])
        assert b['baselines'][k]['interval'] == p1_core.block_interval(delta, index)
        assert b['baselines'][k]['R'] == float(delta.sum()/np.sum(table[k]))
    assert b['judged'] and b['passed'] and b['failed'] == []


def test_b_class_needs_every_baseline_lower_bound_above_zero_and_five_percent():
    q = np.random.default_rng(5).uniform(.01, .03, 19)
    close = list(q*1.03)                                                             # LB > 0 but R = .03 < .05
    b = stats.b_class(b_table(1.3, q=q, override=dict(B_d=close)))
    assert b['baselines']['B_d']['LB'] > 0 and b['baselines']['B_d']['R'] < .05 and not b['baselines']['B_d']['passed'] and b['passed'] is False and b['failed'] == ['B_d']
    noisy = list(q*(1+np.random.default_rng(6).normal(0, .5, 19)))                    # R may be large but the interval holds 0
    b = stats.b_class(b_table(1.3, q=q, override=dict(X_lex=noisy)))
    assert b['baselines']['X_lex']['LB'] <= 0 and b['passed'] is False and 'X_lex' in b['failed']
    b = stats.b_class(b_table(1.3, q=q, override=dict(h=list(q*1.3), V2=[None]+list(q[1:]))))
    assert b['judged'] is False and b['passed'] is None                               # a pair without an AUSE: not judged


# ---- A class ---------------------------------------------------------------------------------------------------------------------------------------
def test_replication_is_the_weighted_average_rank_spearman():
    rng = np.random.default_rng(7)
    for _ in range(20):
        x = np.round(rng.normal(size=19), 1)
        y = np.round(x+rng.normal(size=19), 1)
        w = rng.integers(0, 4, 19)
        w[:6] = np.maximum(w[:6], 1)
        assert stats.replicated_spearman(x, y, w) == pytest.approx(stats.weighted_rank_spearman(x, y, w), rel=1e-12, abs=1e-14)
    x, y = rng.normal(size=12), rng.normal(size=12)
    assert stats.spearman(x, y) == pytest.approx(spearmanr(x, y)[0], rel=1e-12)
    assert stats.spearman(np.ones(12), y) == 0. and stats.replicated_spearman(np.ones(12), y, np.full(12, 2)) == 0.


def a_inputs(seed=8, n=19, constant_structure=None, few=None):
    rng = np.random.default_rng(seed)
    D = rng.uniform(.5, .95, (n, 35))
    X_pq = D+rng.normal(0, .03, D.shape)
    X_h = D+rng.normal(0, .05, D.shape)
    if constant_structure is not None:
        D[:, constant_structure-1] = .8
    if few is not None:
        D[5:, few-1] = np.nan                                                         # defined on 5 pairs only: not in C_A (>= 10)
    S = dict(seg35=np.nanmean(D, axis=1), seg4=np.nanmean(D[:, :4], axis=1))
    S_pq = dict(seg35=X_pq.mean(axis=1), seg4=X_pq[:, :4].mean(axis=1))
    S_h = dict(seg35=X_h.mean(axis=1), seg4=X_h[:, :4].mean(axis=1))
    return D, X_pq, X_h, S, S_pq, S_h


def test_a_class_point_estimates_and_the_structure_set():
    D, X_pq, X_h, S, S_pq, S_h = a_inputs(constant_structure=7, few=9)
    point, _ = stats.a_statistics(D, X_pq, X_h, S, S_pq, S_h)
    used = point['structure']['structures']
    assert 7 not in used and 9 not in used and len(used) == 33
    rho = [stats.spearman(X_pq[:, c-1], D[:, c-1]) for c in used]
    assert point['structure']['M_pq'] == pytest.approx(float(np.median(rho)))
    assert point['structure']['T_H'] == pytest.approx(point['structure']['M_pq']-point['structure']['M_h'])
    assert point['pair']['seg35']['T_0'] == pytest.approx(stats.spearman(S_pq['seg35'], S['seg35']))


def test_a_class_void_draws_are_counted():
    D, X_pq, X_h, S, S_pq, S_h = a_inputs()
    w = np.zeros(19, dtype=np.int64)
    w[:4] = 1
    assert stats.a_statistics(D, X_pq, X_h, S, S_pq, S_h, w=w) == (None, 'positive_pairs')
    D2 = D.copy()
    D2[:, :] = np.nan
    D2[:3, :] = .7                                                                     # no structure has 5 pairs defined
    w = np.ones(19, dtype=np.int64)
    S2 = dict(S)
    assert stats.a_statistics(D2, X_pq, X_h, S2, S_pq, S_h, w=w)[1] == 'no_structure'
    S3 = dict(seg35=np.full(19, .8), seg4=S['seg4'])
    assert stats.a_statistics(D, X_pq, X_h, S3, S_pq, S_h, w=w)[1] == 'constant_actual'
    result = stats.a_class(D, X_pq, X_h, S, S_pq, S_h, n=300)
    assert sum(result['void_draws'].values()) >= 0 and all(v['n_valid_draws'] == 300 and v['n_valid'] <= 300 for v in result['intervals'].values())
    assert set(result['intervals']) == {'structure.T_H', 'structure.T_0', 'pair.seg35.T_H', 'pair.seg35.T_0', 'pair.seg4.T_H', 'pair.seg4.T_0'}


def test_a_class_n_valid_counts_the_draws_a_statistic_took_part_in():
    """Review minor 6: a pair statistic that is not defined in a valid draw (fewer than 2 of its pairs drawn) is left out of its interval and of its n_valid."""
    D, X_pq, X_h, S, S_pq, S_h = a_inputs(seed=13)
    S_pq = dict(S_pq, seg4=np.where(np.arange(19) < 10, S_pq['seg4'], np.nan))         # defined on the first 10 pairs only
    result = stats.a_class(D, X_pq, X_h, S, S_pq, S_h, n=400)
    seg4, structure = result['intervals']['pair.seg4.T_0'], result['intervals']['structure.T_0']
    assert structure['n_valid'] == structure['n_valid_draws'] == 400
    assert seg4['n_valid_draws'] == 400 and seg4['n_valid'] < 400


def test_a_class_resampling_uses_block_intervals_generator():
    rng = np.random.default_rng(stats.BOOT_SEED)
    counts = np.bincount(rng.integers(0, 20, 20), minlength=20)
    w = counts[np.arange(19)]*counts[np.arange(19)+1]
    D, X_pq, X_h, S, S_pq, S_h = a_inputs()
    first, _ = stats.a_statistics(D, X_pq, X_h, S, S_pq, S_h, w=w, structures=list(range(1, 36)))
    result = stats.a_class(D, X_pq, X_h, S, S_pq, S_h, n=1)
    if (w > 0).sum() >= 5 and first is not None:                                       # the first draw of the generator is valid: it is the only draw of n = 1
        assert result['intervals']['pair.seg35.T_0']['lo'] == pytest.approx(first['pair']['seg35']['T_0'])


# ---- the reading -----------------------------------------------------------------------------------------------------------------------------------
GOOD = dict(m1=True, m2=True, m3=True, m4=True, m5=True, m6=True)
EVAL_OK = dict(e1='bitwise', e2=True, complete=True, integrity=True)


def judged(passed, failed=()):
    return dict(judged=True, passed=passed, failed=list(failed))


def case(truth_read=True, method=GOOD, released=True, b=None, **eval_changes):
    return dict(method_controls=dict(method), released=released, eval_controls=dict(EVAL_OK, **eval_changes), b=judged(True) if b is None else b, truth_read=truth_read)


@pytest.mark.parametrize('kwargs, expected', [
    # the truth not read (no lock): the first row, e2 before the truth, or no reading yet
    (case(truth_read=False, method=dict(GOOD, m3=False)), 'method_controls_or_release'),
    (case(truth_read=False, released=False), 'method_controls_or_release'),
    (case(truth_read=False, method={}), 'method_controls_or_release'),
    (case(truth_read=False, method={k: v for k, v in GOOD.items() if k != 'm6'}), 'method_controls_or_release'),      # review minor 2: a missing key is not passed
    (case(truth_read=False, method=dict(GOOD, m7=True)), 'method_controls_or_release'),                                 # an extra key too
    (case(truth_read=False, e2=False, e1=None, complete=False), 'eval_controls'),
    (case(truth_read=False, e1=None, e2=None, complete=False), 'not_evaluated'),
    # the truth read (the lock): never the first row
    (case(method=dict(GOOD, m3=False)), 'eval_controls'),
    (case(method=dict(m1=True)), 'eval_controls'),
    (case(released=False), 'eval_controls'),
    (case(e1=False), 'eval_controls'),
    (case(e1=None), 'eval_controls'),
    (case(e2=False), 'eval_controls'),
    (case(complete=False), 'eval_controls'),
    (case(integrity=False), 'eval_controls'),
    (case(b=dict(judged=False, reason='a pair without an error')), 'b_not_judged'),
    (case(), 'b_pass'),
    (case(e1='within_1e-12'), 'b_pass'),
    (case(b=judged(False, ['X_lex'])), 'b_fail'),
])
def test_every_row_of_the_decision_table(kwargs, expected):
    r = stats.reading(**kwargs)
    assert r['row'] == expected
    assert r['counts_as_reading'] is (expected not in ('method_controls_or_release', 'not_evaluated'))
    if expected == 'b_pass':
        assert r['reading'] == stats.READ_PASS and r['passed'] is True and r['e1_listed_separately'] is (kwargs['eval_controls']['e1'] == 'within_1e-12')
    if expected == 'b_fail':
        assert r['reading'] == stats.READ_FAIL and r['failed'] == ['X_lex']
    if expected == 'eval_controls':
        assert r['reading'] == stats.READ_UNDECIDED_EVAL


def test_intersection_union_and_class_a_does_not_enter():
    q = np.random.default_rng(9).uniform(.01, .03, 19)
    all_pass = stats.b_class(b_table(1.4, q=q))
    one_fails = stats.b_class(b_table(1.4, q=q, override=dict(H_F=list(q*.9))))
    good = dict(method_controls=dict(GOOD), released=True, eval_controls=dict(EVAL_OK), truth_read=True)
    assert stats.reading(b=all_pass, **good)['row'] == 'b_pass' and stats.reading(b=one_fails, **good)['row'] == 'b_fail'
    import inspect
    assert 'a_class' not in inspect.signature(stats.reading).parameters                 # D87 = (c): class A has no way into the reading


# ---- descriptions ----------------------------------------------------------------------------------------------------------------------------------
def test_reliability_bins_and_ece():
    q = np.array([0., .05, .1, .5, .51, .99, 1.])
    e = np.array([0, 0, 1, 1, 0, 1, 1], dtype=bool)
    r = stats.reliability(q, e, bins='width')
    assert sum(r['count']) == 7 and r['count'][14] == 2 and r['count'][7] == 2 and r['count'][0] == 2 and r['count'][1] == 1
    expected = (abs(.05-0)+abs(.1-1)+abs(1.01-1)+abs(1.99-2))/7
    assert r['ece'] == pytest.approx(expected)
    d = stats.reliability(np.linspace(0, 1, 100), np.zeros(100, dtype=bool), bins='decile')
    assert d['count'] == [10]*10


def test_constant_prediction_uses_pairs_without_a_shared_subject():
    values = np.arange(19, dtype=np.float64)
    K = stats.constant_prediction(values)
    assert K[0] == pytest.approx(np.mean(np.arange(2, 19))) and K[5] == pytest.approx(np.mean([v for v in range(19) if abs(v-5) >= 2]))
    values[3] = np.nan
    assert stats.constant_prediction(values)[0] == pytest.approx(np.mean([v for v in range(2, 19) if v != 3]))
