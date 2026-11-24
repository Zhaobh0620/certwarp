"""PQ, the reading on 19 pairs (plan.md sections 5-7): experiments/pq/pq_decide.py on made-up eval.json and pq.json documents with the JSON shapes the units write (string keys), so that the B class,
the A-class descriptions (a small number of draws) and every description run at the plan's size.  No file of a run is read."""
import json

import numpy as np

from . import _world as W

import pq_decide as decide  # noqa: E402
import pq_eval as pe  # noqa: E402
import pq_stats as stats  # noqa: E402

io_, pins = W.io_, W.pins


def documents(seed, factor, *, override=None):
    rng = np.random.default_rng(seed)
    evals, pqs = {}, {}
    for i, short in enumerate(pins.SHORTS):
        D = rng.uniform(.5, .95, 35)
        per35 = {str(c): (None if (c == 33 and i % 4 == 0) else float(D[c-1])) for c in range(1, 36)}
        per4 = {str(g): float(rng.uniform(.8, .95)) for g in range(1, 5)}
        q = float(rng.uniform(.01, .03))
        scores = {}
        for mask in pe.MASKS:
            scores[mask] = {name: dict(ause=q if name == 'q' else q*(override or {}).get(name, factor)*(1+.02*rng.standard_normal()), auroc=float(rng.uniform(.6, .9)), largest_tie_share=float(rng.uniform(0, .5)))
                            for name in pe.SCORES}
        reliability = {b: dict(count=[100]*n, sum_q=list(rng.uniform(0, 100, n)), errors=list(rng.uniform(0, 100, n)), ece=float(rng.uniform(0, .1))) for b, n in (('width', 15), ('decile', 10))}
        evals[short] = dict(actual=dict(seg35=float(np.mean([v for v in per35.values() if v is not None])), seg4=float(np.mean(list(per4.values()))), dice_official=float(D.mean()),
                                        seg35_per_label=per35, seg4_per_label=per4),
                            scores=scores, seg4_scale=dict(q4=dict(ause=q*.5, auroc=.8)), reliability=reliability, e1='bitwise')
        X = D+.03+rng.normal(0, .02, 35)
        defined = [c for c in range(1, 36) if per35[str(c)] is not None]
        pqs[short] = dict(numbers=dict(D_hat={str(c): float(X[c-1]) for c in range(1, 36)}, H_c={str(c): float(X[c-1]+rng.normal(0, .02)) for c in range(1, 36)}, defined=defined,
                                       S35_hat=float(np.mean(X[[c-1 for c in defined]])), S4_hat=float(np.mean(list(per4.values()))+.02), H35=float(np.mean(X[[c-1 for c in defined]])+.01),
                                       H4=float(np.mean(list(per4.values()))+.03), D4_hat={str(g): per4[str(g)]+.02 for g in range(1, 5)}, official_dice35_hat=float(X.mean())),
                          eps_em=dict(eps=float(rng.uniform(.02, .1)), at_floor=False, at_ceiling=False), tail_mass=dict(fixed=[0.]*36, moving_pulled_back=[0.]*36))
    return json.loads(json.dumps(evals)), json.loads(json.dumps(pqs))


def run(evals, pqs, **changes):
    ctx = io_.Context(repo=W.ROOT, source_root=W.ROOT, results=W.ROOT/'nonexistent')
    kwargs = dict(evals=evals, pqs=pqs, out_record=dict(controls=dict(m1=True, m2=True, m3=True, m4=True, m5=True, m6=True)), members=dict(m5_passed=True, subjects={}),
                  eval_record=dict(e2=dict(passed=True), release=dict(sha256='a'*64), truth_read=True, complete=True, aborted=False, status='complete'), released=True, truth_read=True,
                  integrity=dict(passed=True), a_draws=60)
    kwargs.update(changes)
    return decide.decide(ctx, **kwargs)


def test_nineteen_pairs_that_pass(tmp_path):
    evals, pqs = documents(0, 1.3)
    doc = run(evals, pqs)
    assert doc['reading']['row'] == 'b_pass' and doc['b_class']['judged'] and all(r['passed'] for r in doc['b_class']['baselines'].values())
    n = doc['described']
    assert set(n['pair_seg']) == {'seg35', 'seg4'} and n['a_class']['point']['structure'] is not None and n['a_class']['intervals']
    assert n['pair_seg']['seg35']['pq']['interval']['n'] == stats.BOOT_N and len(n['ause_table']['q']) == 19
    io_.write_once(tmp_path/'decision.json', doc)                                     # JSON as the unit writes it
    report = decide.report_markdown(doc)
    assert f"**读数**：{stats.READ_PASS}" in report and stats.READ_FAIL not in report        # review minor 8: the report tells pass from fail


def test_nineteen_pairs_where_one_baseline_holds():
    evals, pqs = documents(1, 1.3, override=dict(B_d=1.01))
    doc = run(evals, pqs)
    assert doc['reading']['row'] == 'b_fail' and doc['reading']['failed'] == ['B_d']
    report = decide.report_markdown(doc)
    assert f"**读数**：{stats.READ_FAIL}" in report and f"**读数**：{stats.READ_PASS}" not in report


def test_a_pair_without_its_evaluation_is_not_judged():
    evals, pqs = documents(2, 1.3)
    evals.pop(pins.SHORTS[7])
    doc = run(evals, pqs)
    assert doc['reading']['row'] == 'eval_controls' and doc['eval_controls']['e1'] is False and doc['b_class'] is None


def test_an_incomplete_or_missing_evaluation_record_is_counted_as_used():
    evals, pqs = documents(4, 1.3)
    aborted = dict(e2=dict(passed=True), release=dict(sha256='a'*64), truth_read=True, complete=False, aborted=True, status='真值已读、中止')
    for record in (aborted, None):
        doc = run(evals, pqs, eval_record=record)
        assert doc['reading']['row'] == 'eval_controls' and doc['reading']['counts_as_reading'] is True and doc['b_class'] is None


def test_without_the_release_or_the_controls():
    evals, pqs = documents(3, 1.3)
    assert run(evals, pqs, released=False, truth_read=False)['reading']['row'] == 'method_controls_or_release'
    assert run(evals, pqs, members=dict(m5_passed=False), truth_read=False)['reading']['row'] == 'method_controls_or_release'
    assert run({}, {}, eval_record=None, released=False, truth_read=False)['reading']['row'] == 'method_controls_or_release'
    assert run(evals, pqs, released=False)['reading']['row'] == 'eval_controls'          # the truth read: never the first row
