"""PQ, the reading (plan.md sections 5, 6, 7; contract C22.Q4, D87 = (c), D88 = (a)): from the evaluation unit's eval.json, the PQ unit's pq.json and the records of the controls; the reading by code.

Judged: class B only, the AUSE of q on Omega_e against the 8 baselines (pq_stats.b_class, block_interval as it is, LB_k > 0 and R_k >= .05 for all).  Described: class A (the two groups of plan.md 5.1 with their
block intervals and void draws), the pair's seg35 / seg4 against H and K, per structure group bias, AUROC, the other masks, -||grad F||, the seg4 scale, the reliability of q, the largest tie shares, eps,
the tail masses, the members' control numbers.  The predicted Dice are the method's own numbers: they enter no Dice table (C22.Q0-2); seg35 and seg4 are always both reported.
"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pq_io as io_  # noqa: E402
import pq_pins as pins  # noqa: E402
import pq_stats as stats  # noqa: E402

DECISION_SCHEMA = 'bayesreg.pq.decision.v1'


def _num(value):
    return np.nan if value is None else float(value)


def tables(ctx, evals, pqs):
    """Per pair, in pair order: the AUSE table on Omega_e, the per-structure arrays of class A, the pair-level values."""
    shorts = list(ctx.shorts)
    ause = {name: [((evals[s]['scores']['omega_e'].get(name) or {}).get('ause')) for s in shorts] for name in ('q',)+stats.JUDGED_BASELINES+stats.DESCRIBED_SCORES}
    D = np.array([[_num(evals[s]['actual']['seg35_per_label'].get(c, evals[s]['actual']['seg35_per_label'].get(str(c)))) for c in range(1, 36)] for s in shorts])
    defined = [set(pqs[s]['numbers']['defined']) for s in shorts]
    X_pq = np.array([[(_num(pqs[s]['numbers']['D_hat'].get(str(c))) if c in defined[i] else np.nan) for c in range(1, 36)] for i, s in enumerate(shorts)])
    X_h = np.array([[(_num(pqs[s]['numbers']['H_c'].get(str(c))) if c in defined[i] else np.nan) for c in range(1, 36)] for i, s in enumerate(shorts)])
    S = dict(seg35=np.array([_num(evals[s]['actual']['seg35']) for s in shorts]), seg4=np.array([_num(evals[s]['actual']['seg4']) for s in shorts]))
    S_pq = dict(seg35=np.array([_num(pqs[s]['numbers']['S35_hat']) for s in shorts]), seg4=np.array([_num(pqs[s]['numbers']['S4_hat']) for s in shorts]))
    S_h = dict(seg35=np.array([_num(pqs[s]['numbers']['H35']) for s in shorts]), seg4=np.array([_num(pqs[s]['numbers']['H4']) for s in shorts]))
    return ause, D, X_pq, X_h, S, S_pq, S_h


def bias_rows(S, S_x, index):
    """Mean of (prediction - actual) with its block interval, mean absolute deviation, number of positive pairs."""
    diff = np.asarray(S_x, dtype=np.float64)-np.asarray(S, dtype=np.float64)
    ok = np.isfinite(diff)
    if not ok.all():
        return dict(mean=None, note='a pair without a defined value', n=int(ok.sum()))
    interval = stats.block_interval(diff, index)
    return dict(mean=float(diff.mean()), interval=interval, mad=float(np.abs(diff).mean()), n_positive=int((diff > 0).sum()), n=int(diff.size))


def correlations(S, S_x):
    from scipy.stats import pearsonr
    S, S_x = np.asarray(S, dtype=np.float64), np.asarray(S_x, dtype=np.float64)
    ok = np.isfinite(S) & np.isfinite(S_x)
    if ok.sum() < 3 or np.all(S_x[ok] == S_x[ok][0]) or np.all(S[ok] == S[ok][0]):
        return dict(spearman=stats.spearman(S_x[ok], S[ok]) if ok.sum() >= 2 else None, pearson=None, n=int(ok.sum()))
    return dict(spearman=stats.spearman(S_x[ok], S[ok]), pearson=float(pearsonr(S_x[ok], S[ok])[0]), n=int(ok.sum()))


def described(ctx, evals, pqs, members, D, X_pq, X_h, S, S_pq, S_h, ause, a_draws=stats.BOOT_N):
    index = np.arange(len(ctx.shorts))
    out = {}
    pair_level = {}
    for key in ('seg35', 'seg4'):
        K = stats.constant_prediction(S[key])
        corrected = np.asarray(S_pq[key])-stats.constant_prediction(np.asarray(S_pq[key])-np.asarray(S[key]))
        pair_level[key] = dict(pq=bias_rows(S[key], S_pq[key], index), hard=bias_rows(S[key], S_h[key], index), constant=bias_rows(S[key], K, index),
                               corrected_pq_mad=stats.nanmean(np.abs(corrected-np.asarray(S[key]))), constant_mad=stats.nanmean(np.abs(K-np.asarray(S[key]))),
                               correlation_pq=correlations(S[key], S_pq[key]), correlation_hard=correlations(S[key], S_h[key]),
                               hard_more_optimistic_pairs=int(np.sum((np.asarray(S_h[key])-np.asarray(S[key])) > (np.asarray(S_pq[key])-np.asarray(S[key])))))
    out['pair_seg'] = pair_level
    groups = dict(small_six=stats.SMALL_SIX, large_six=stats.LARGE_SIX, other_23=stats.OTHER_23)
    bias = X_pq-D
    out['structure_bias'] = {name: dict(pq=stats.nanmean(bias[:, [c-1 for c in labels]]), hard=stats.nanmean((X_h-D)[:, [c-1 for c in labels]])) for name, labels in groups.items()}
    D4 = np.array([[_num(evals[s]['actual']['seg4_per_label'].get(g, evals[s]['actual']['seg4_per_label'].get(str(g)))) for g in range(1, 5)] for s in ctx.shorts])
    X4 = np.array([[_num(pqs[s]['numbers']['D4_hat'].get(str(g))) for g in range(1, 5)] for s in ctx.shorts])
    out['seg4_structures_spearman'] = {g: stats.spearman(X4[np.isfinite(X4[:, g-1]) & np.isfinite(D4[:, g-1]), g-1], D4[np.isfinite(X4[:, g-1]) & np.isfinite(D4[:, g-1]), g-1]) for g in range(1, 5)}
    out['official_dice35'] = dict(predicted=[pqs[s]['numbers']['official_dice35_hat'] for s in ctx.shorts], actual=[evals[s]['actual']['dice_official'] for s in ctx.shorts])
    out["a_class"] = stats.a_class(D, X_pq, X_h, S, S_pq, S_h, n=a_draws)
    masks = {}
    for mask in ('omega_e', 'band', 'whole'):
        masks[mask] = {name: dict(ause=[evals[s]['scores'][mask][name]['ause'] for s in ctx.shorts], auroc=[evals[s]['scores'][mask][name]['auroc'] for s in ctx.shorts],
                                  largest_tie_share=[evals[s]['scores'][mask][name]['largest_tie_share'] for s in ctx.shorts]) for name in evals[ctx.shorts[0]]['scores'][mask]}
    out['scores'] = masks
    out['seg4_scale'] = dict(ause=[evals[s]['seg4_scale']['q4']['ause'] for s in ctx.shorts], auroc=[evals[s]['seg4_scale']['q4']['auroc'] for s in ctx.shorts])
    for bins in ('width', 'decile'):
        count = sum(np.asarray(evals[s]['reliability'][bins]['count'], dtype=np.float64) for s in ctx.shorts)
        sums = sum(np.asarray(evals[s]['reliability'][bins]['sum_q'], dtype=np.float64) for s in ctx.shorts)
        errs = sum(np.asarray(evals[s]['reliability'][bins]['errors'], dtype=np.float64) for s in ctx.shorts)
        per = [evals[s]['reliability'][bins]['ece'] for s in ctx.shorts]
        out[f'reliability_{bins}'] = dict(count=count.tolist(), mean_q=(sums/np.maximum(count, 1)).tolist(), error_rate=(errs/np.maximum(count, 1)).tolist(), ece_pooled=stats.ece(count, sums, errs),
                                          ece_per_pair_range=[min(x for x in per if x is not None), max(x for x in per if x is not None)] if any(x is not None for x in per) else None)
    eps = [pqs[s]['eps_em']['eps'] for s in ctx.shorts]
    out['eps'] = dict(values=eps, range=[min(eps), max(eps)] if all(e is not None for e in eps) else None, spearman_with_seg35=stats.spearman(np.asarray(eps, dtype=np.float64), S['seg35']) if all(e is not None for e in eps) else None,
                      at_floor=[pqs[s]['eps_em']['at_floor'] for s in ctx.shorts], at_ceiling=[pqs[s]['eps_em']['at_ceiling'] for s in ctx.shorts])
    out['tail_mass'] = {s: pqs[s]['tail_mass'] for s in ctx.shorts}
    out["members"] = {s: dict(agreement=r.get("agreement"), m5=r.get("m5"), n_differ=r.get("n_differ"), max_abs_difference_to_registered=r.get("max_abs_difference_to_registered")) for s, r in (members.get("subjects") or {}).items()}
    out['ause_table'] = ause
    return out


def decide(ctx, *, evals, pqs, out_record, members, eval_record, released, truth_read, integrity=None, a_draws=stats.BOOT_N):
    """The decision document of plan.md section 6 (and section 7's descriptions when the evaluation is complete).  truth_read: the evaluation unit's lock exists (revision 1, review important 1);
    integrity: {'passed': bool, ...} of the files read here against the list and the evaluation record (run_pq.decide_unit; revision 1, review minor 9)."""
    controls = (out_record or {}).get('controls') or {}
    method = {k: controls.get(k) for k in stats.METHOD_KEYS} if controls else {}
    if method:
        method['m5'] = method.get('m5') is True and (members or {}).get('m5_passed') is True
    record_complete = bool(eval_record) and eval_record.get('truth_read') is True and eval_record.get('complete') is True and eval_record.get('aborted') is False
    have_all = bool(evals) and all(s in evals for s in ctx.shorts)
    complete = record_complete and have_all
    e1_values = [evals[s].get('e1') for s in ctx.shorts] if have_all else []
    if not have_all:
        e1 = False if (evals or truth_read) else None          # a pair without its evaluation: e1 has a missing item
    elif any(v not in stats.E1_PASSING for v in e1_values):
        e1 = False
    else:
        e1 = 'within_1e-12' if any(v == 'within_1e-12' for v in e1_values) else 'bitwise'
    e2_record = (eval_record or {}).get('e2')
    e2 = None if e2_record is None else e2_record.get('passed') is True
    integrity = integrity or dict(passed=True, note='not checked')
    eval_controls = dict(e1=e1, e2=e2, complete=complete, integrity=integrity.get('passed') is True)
    b = None
    numbers = None
    if complete and integrity.get('passed') is True:
        ause, D, X_pq, X_h, S, S_pq, S_h = tables(ctx, evals, pqs)
        b = stats.b_class(ause, np.arange(len(ctx.shorts)))
        numbers = described(ctx, evals, pqs, members or {}, D, X_pq, X_h, S, S_pq, S_h, ause, a_draws=a_draws)
    read = stats.reading(method_controls=method, released=released, eval_controls=eval_controls, b=b, truth_read=truth_read)
    return dict(schema=DECISION_SCHEMA, plan_sha256=pins.PLAN_SHA256, contract=pins.CONTRACT, d87='(c): only class B is judged; class A is described', reading=read, b_class=b,
                method_controls=method, released=released, truth_read=truth_read, eval_controls=eval_controls, integrity=integrity,
                eval_record_state=None if eval_record is None else {k: eval_record.get(k) for k in ('truth_read', 'complete', 'aborted', 'status', 'error', 'pairs')},
                e1_listed_apart=[s for s in ctx.shorts if evals and evals.get(s, {}).get('e1') == 'within_1e-12'],
                described=numbers, wording='C22.Q3-5: passed -> only "开发集 19 对单次读数上，q 对标签传播误差的 AUSE 低于预登记所列的每一个非贝叶斯基线" with the 8 baselines named; never "预测 Dice"')


def report_markdown(document):
    """A short Chinese summary of the decision document (seg35 and seg4 both; the predicted Dice are the method's own numbers, in no Dice table)."""
    r = document['reading']
    lines = ['# PQ 读数（开发集 19 对，单次；选择读数第 3 次）', '', f"**读数**：{r['reading']}", '']
    b = document.get('b_class')
    if b and b.get('judged'):
        lines += ['| 基线 | LB（2.5 百分位） | R | 过 |', '|---|---|---|---|']
        for k, row in b['baselines'].items():
            lines.append(f"| {k} | {row['LB']:.6f} | {row['R'] if row['R'] is None else round(row['R'], 4)} | {'是' if row['passed'] else '否'} |")
        lines.append('')
    n = document.get('described')
    if n:
        for key in ('seg35', 'seg4'):
            p = n['pair_seg'][key]
            lines.append(f"- 这一对的 {key}（只描述，方法自报的数，不进 Dice 表）：PQ 偏差均值 {p['pq'].get('mean')}，硬标签 {p['hard'].get('mean')}，常数 {p['constant'].get('mean')}")
    lines += ['', 'A 类只描述（D87 = (c)）；不能说"预测 Dice"。']
    return '\n'.join(lines)+'\n'
