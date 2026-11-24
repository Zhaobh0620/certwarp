"""PQ (contract C22.Q4, C22.Q7 "评测与判定代码"; runs/20261107-pq/plan.md sections 5-7): the statistics of the evaluation and of the reading, pure numpy / scipy; no file is read here.

AUSE (plan.md 5.2, methodology 8.4, Ilg et al. ECCV 2018)
  On the mask (N voxels, n_1 errors) the voxels are removed in the order of the score, the largest first; equal scores are one tie group; a lexicographic score (X_lex = h before H2) is ordered by its
  keys, primary first, never by a weighted float.  E_s(m), the errors among the first m removed: whole groups count their errors; inside a group the cut takes the group's error share (the expectation over
  a random order inside the tie).  r_s(m) = (n_1 - E_s(m)) / (N - m), the oracle r_o(m) = max(n_1 - m, 0) / (N - m), m_j = floor(j N / 100) in integers (j = 0..99; the exact floor of f_j N, f_j = j/100),
  AUSE_s = (1/100) sum_j [r_s(m_j) - r_o(m_j)]; not normalized by r_s(0).  No error in the mask: None (B is then not judged).
AUROC (description): Mann-Whitney with average ranks, from the same tie groups: sum_k e_k (C0_below(k) + c0_k / 2) / (n_1 n_0).
B class (plan.md 5.2, D88 = (a)): for every one of the 8 judged baselines k, Delta_k(p) = AUSE_k(p) - AUSE_q(p); experiments/p1/p1_core.block_interval called as it is (10000 draws, seed 20261105, the
  2.5 and 97.5 percentiles of the weighted mean; reseeded at every call, so all baselines share the draws); LB_k = its lo; R_k = sum_p Delta_k / sum_p AUSE_k.  Passed: LB_k > 0 and R_k >= .05 for all 8
  (intersection-union test, no multiplicity correction within the class).
A class (plan.md 5.1, D87 = (c), D94 = (a): described only, never judged): Spearman by average ranks; pair i repeated w_i times in a draw; the draws of block_interval's generator
  (default_rng(20261105), counts of 20 subjects, w_i = counts_i counts_{i+1}); a draw is void when fewer than 5 pairs have positive weight, no structure takes part, or the distinct pairs' actual seg35 or seg4
  are all equal; void draws are counted and the draws go on to 10000 valid ones.  X constant on the pairs used: rho = 0.  K, the constant prediction, is rho = 0 by definition (not computed).
  The structures of a draw are those of C_A (the original data: >= 10 pairs with both defined and actual not all equal) that have, in the draw, >= 5 distinct pairs with both defined and actual not all equal
  (the plan's sentence on the draw names no set; the tool reads it inside C_A -- a reading of the implementer, only for the described interval).
"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for _rel in ('p1', 'ef', 'mc0'):                    # p1_core imports fast_channels (experiments/ef), which imports mc0_core and mc0p_core (experiments/mc0)
    if str(HERE.parent/_rel) not in sys.path:
        sys.path.append(str(HERE.parent/_rel))

N_SUBJECTS = 20
N_PAIRS = 19
N_F = 100                                           # f_j = j / 100, j = 0..99
BOOT_N, BOOT_SEED = 10000, 20261105
R_MIN = .05                                         # D88 = (a)
JUDGED_BASELINES = ('h', 'H_F', 'H2', 'V_F', 'V2', 'G_plus', 'X_lex', 'B_d')         # plan.md 3.2, the 8 of the decision (D91 = (a))
DESCRIBED_SCORES = ('G_minus',)                     # -||grad F||, methodology 8.4's direction: described only
A_MIN_PAIRS_ORIGINAL, A_MIN_PAIRS_DRAW, A_MIN_POSITIVE = 10, 5, 5
SMALL_SIX = (4, 23, 18, 34, 19, 35)
LARGE_SIX = (1, 2, 5, 20, 21, 24)
OTHER_23 = tuple(c for c in range(1, 36) if c not in SMALL_SIX+LARGE_SIX)


def block_interval(values, index):
    """experiments/p1/p1_core.block_interval, called as it is (C22.Q7: "原样调用")."""
    import p1_core
    return p1_core.block_interval(values, index)


# ---- tie groups, AUSE, AUROC ------------------------------------------------------------------------------------------------------------------------
def group_codes(keys):
    """Dense codes of the tie groups of one or more keys (primary first): code order = lexicographic order of the key tuples (ascending).  Every key is compared exactly (no float composite)."""
    keys = [np.asarray(k).ravel() for k in (keys if isinstance(keys, (list, tuple)) else [keys])]
    if not keys or any(k.shape != keys[0].shape for k in keys):
        raise ValueError('the keys of a score must be arrays of one shape')
    codes = None
    for key in keys:
        if key.dtype.kind == 'f' and not np.all(np.isfinite(key)):
            raise ValueError('a score that is not finite')
        uniq, inverse = np.unique(key, return_inverse=True)
        inverse = inverse.astype(np.int64)
        if codes is None:
            codes = inverse
        else:
            if codes.max(initial=0)+1 > (2**62)//max(len(uniq), 1):
                raise ValueError('too many tie groups to combine exactly')
            codes = codes*len(uniq)+inverse
            codes = np.unique(codes, return_inverse=True)[1].astype(np.int64)
    return codes


def tie_table(keys, errors):
    """(counts, errors) of the tie groups in removal order: the largest score first."""
    codes = group_codes(keys)
    e = np.asarray(errors).ravel().astype(bool)
    if e.shape != codes.shape:
        raise ValueError('the score and the errors differ in size')
    n = int(codes.max())+1 if codes.size else 0
    counts = np.bincount(codes, minlength=n).astype(np.int64)
    errs = np.bincount(codes, weights=e.astype(np.float64), minlength=n).astype(np.int64)
    return counts[::-1], errs[::-1]


def ause_from_table(counts, errs):
    """AUSE of plan.md 5.2 from the tie table in removal order; None without an error.  Also returns the curve (r_s, r_o at m_j)."""
    counts, errs = np.asarray(counts, dtype=np.int64), np.asarray(errs, dtype=np.int64)
    N, n1 = int(counts.sum()), int(errs.sum())
    if N == 0 or n1 == 0:
        return None, None
    before_c = np.concatenate([[0], np.cumsum(counts)])
    before_e = np.concatenate([[0], np.cumsum(errs)]).astype(np.float64)
    m = (np.arange(N_F, dtype=np.int64)*N)//N_F
    k = np.searchsorted(before_c, m, side='right')-1
    E = before_e[k]+(m-before_c[k]).astype(np.float64)*errs[k].astype(np.float64)/counts[k].astype(np.float64)
    rest = (N-m).astype(np.float64)
    r_s = (n1-E)/rest
    r_o = np.maximum(n1-m, 0).astype(np.float64)/rest
    return float(np.sum(r_s-r_o)/N_F), dict(m=m.tolist(), r_s=r_s.tolist(), r_o=r_o.tolist())


def ause(keys, errors):
    value, _ = ause_from_table(*tie_table(keys, errors))
    return value


def auroc_from_table(counts, errs):
    """Mann-Whitney AUROC with average ranks (errors positive, larger score more likely an error); None when one class is empty."""
    counts, errs = np.asarray(counts, dtype=np.int64)[::-1], np.asarray(errs, dtype=np.int64)[::-1]     # ascending
    n1 = int(errs.sum())
    n0 = int(counts.sum())-n1
    if n1 == 0 or n0 == 0:
        return None
    c0 = (counts-errs).astype(np.float64)
    below = np.concatenate([[0.], np.cumsum(c0)[:-1]])
    return float(np.sum(errs.astype(np.float64)*(below+.5*c0))/(float(n1)*float(n0)))


def largest_tie_share(counts):
    counts = np.asarray(counts)
    return float(counts.max()/counts.sum()) if counts.size and counts.sum() else None


def score_record(keys, errors):
    """AUSE, AUROC, the largest tie group's share, N and n_1 of one score on one mask."""
    counts, errs = tie_table(keys, errors)
    value, curve = ause_from_table(counts, errs)
    return dict(ause=value, auroc=auroc_from_table(counts, errs), largest_tie_share=largest_tie_share(counts), n_groups=int(counts.size), N=int(counts.sum()), n_errors=int(errs.sum()), curve=curve)


# ---- reliability of q (description, plan.md 7) -------------------------------------------------------------------------------------------------------
def reliability(q, errors, *, bins='width'):
    """Per bin: count, sum of q, errors.  bins 'width': 15 equal-width bins on [0, 1] (N0 (d)); 'decile': the ten deciles of q on this mask (np.quantile edges; a voxel on an edge goes to the upper bin)."""
    q = np.asarray(q, dtype=np.float64).ravel()
    e = np.asarray(errors).ravel().astype(bool)
    if bins == 'width':
        index = np.minimum((np.clip(q, 0., 1.)*15).astype(np.int64), 14)
        n = 15
    elif bins == 'decile':
        edges = np.quantile(q, np.linspace(0., 1., 11)) if q.size else np.zeros(11)
        index = np.searchsorted(edges[1:-1], q, side='right')
        n = 10
    else:
        raise ValueError(bins)
    counts = np.bincount(index, minlength=n)
    sums = np.bincount(index, weights=q, minlength=n)
    errs = np.bincount(index, weights=e.astype(np.float64), minlength=n)
    return dict(bins=bins, count=counts.tolist(), sum_q=sums.tolist(), errors=errs.tolist(), ece=ece(counts, sums, errs))


def ece(counts, sums, errs):
    counts, sums, errs = (np.asarray(a, dtype=np.float64) for a in (counts, sums, errs))
    total = counts.sum()
    if total == 0:
        return None
    used = counts > 0
    return float(np.sum(np.abs(sums[used]-errs[used]))/total)        # sum_b n_b |mean q_b - error rate_b| / N


# ---- Spearman ---------------------------------------------------------------------------------------------------------------------------------------
def spearman(x, y):
    """Spearman by average ranks (scipy.stats.rankdata, Pearson of the ranks); x constant -> 0 (plan.md 5.1); y constant -> None."""
    from scipy.stats import rankdata
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return None
    if np.all(x == x[0]):
        return 0.
    if np.all(y == y[0]):
        return None
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx-rx.mean(), ry-ry.mean()
    return float(np.sum(rx*ry)/np.sqrt(np.sum(rx*rx)*np.sum(ry*ry)))


def replicated_spearman(x, y, w):
    """Pair i repeated w_i times (integer weights), then spearman (plan.md 5.1)."""
    w = np.asarray(w, dtype=np.int64)
    return spearman(np.repeat(np.asarray(x, dtype=np.float64), w), np.repeat(np.asarray(y, dtype=np.float64), w))


def weighted_rank_spearman(x, y, w):
    """The hand formula the replication must equal: weighted average ranks rank(v) = W(< v) + (W(= v) + 1) / 2, then the weighted Pearson of the ranks (C22.Q7 A 类重抽, first item)."""
    x, y, w = (np.asarray(a, dtype=np.float64) for a in (x, y, w))

    def ranks(v):
        return np.array([w[v < t].sum()+(w[v == t].sum()+1.)/2. for t in v])
    keep = w > 0
    x, y, w = x[keep], y[keep], w[keep]
    if np.all(x == x[0]):
        return 0.
    rx, ry = ranks(x), ranks(y)
    W = w.sum()
    mx, my = (w*rx).sum()/W, (w*ry).sum()/W
    return float((w*(rx-mx)*(ry-my)).sum()/np.sqrt((w*(rx-mx)**2).sum()*(w*(ry-my)**2).sum()))


# ---- A class (described) ----------------------------------------------------------------------------------------------------------------------------
def _structure_ok(defined, actual, w=None, min_pairs=A_MIN_PAIRS_ORIGINAL):
    use = defined if w is None else defined & (w > 0)
    if use.sum() < min_pairs:
        return False
    values = actual[use]
    return not np.all(values == values[0])


def a_statistics(D, X_pq, X_h, S, S_pq, S_h, w=None, structures=None):
    """The A-class numbers of one data set (w None: the original 19 pairs, the thresholds of 10) or one draw (integer weights w, the thresholds of 5).
    D, X_pq, X_h: (pairs, 35) with NaN where undefined (actual / PQ / hard, label c in column c - 1); S, S_pq, S_h: {'seg35': (pairs,), 'seg4': (pairs,)} with NaN where undefined.
    Returns (stats or None, reason): reasons for a void draw 'positive_pairs', 'no_structure', 'constant_actual'."""
    D, X_pq, X_h = (np.asarray(a, dtype=np.float64) for a in (D, X_pq, X_h))
    n = D.shape[0]
    weights = np.ones(n, dtype=np.int64) if w is None else np.asarray(w, dtype=np.int64)
    original = w is None
    if not original and int((weights > 0).sum()) < A_MIN_POSITIVE:
        return None, 'positive_pairs'
    if not original:
        for key in ('seg35', 'seg4'):
            use = (weights > 0) & np.isfinite(S[key])
            values = np.asarray(S[key])[use]
            if values.size == 0 or np.all(values == values[0]):
                return None, 'constant_actual'
    defined = np.isfinite(D) & np.isfinite(X_pq) & np.isfinite(X_h)
    candidates = range(D.shape[1]) if structures is None else [c-1 for c in structures]
    used, rho_pq, rho_h = [], [], []
    for j in candidates:
        if not _structure_ok(defined[:, j], D[:, j], None if original else weights, A_MIN_PAIRS_ORIGINAL if original else A_MIN_PAIRS_DRAW):
            continue
        use = defined[:, j] & (weights > 0)
        used.append(j+1)
        rho_pq.append(replicated_spearman(X_pq[use, j], D[use, j], weights[use]))
        rho_h.append(replicated_spearman(X_h[use, j], D[use, j], weights[use]))
    if not used:
        if not original:
            return None, 'no_structure'
        structure_stats = None
    else:
        M_pq, M_h = float(np.median(rho_pq)), float(np.median(rho_h))
        structure_stats = dict(structures=used, M_pq=M_pq, M_h=M_h, T_H=M_pq-M_h, T_0=M_pq, rho_pq=dict(zip(used, rho_pq)), rho_h=dict(zip(used, rho_h)))
    pair_stats = {}
    for key in ('seg35', 'seg4'):
        use = (weights > 0) & np.isfinite(S[key]) & np.isfinite(S_pq[key]) & np.isfinite(S_h[key])
        if original and use.sum() < A_MIN_PAIRS_ORIGINAL:
            pair_stats[key] = None
            continue
        r_pq = replicated_spearman(np.asarray(S_pq[key])[use], np.asarray(S[key])[use], weights[use])
        r_h = replicated_spearman(np.asarray(S_h[key])[use], np.asarray(S[key])[use], weights[use])
        pair_stats[key] = None if r_pq is None or r_h is None else dict(rho_pq=r_pq, rho_h=r_h, T_H=r_pq-r_h, T_0=r_pq, n_pairs=int(use.sum()))
    return dict(structure=structure_stats, pair=pair_stats), None


def a_class(D, X_pq, X_h, S, S_pq, S_h, *, n=BOOT_N, seed=BOOT_SEED, n_subjects=N_SUBJECTS):
    """Point estimates on the original data and the 2.5 / 97.5 percentiles of n valid draws (plan.md 5.1); the void draws counted by reason.  Described only (D87 = (c))."""
    point, _ = a_statistics(D, X_pq, X_h, S, S_pq, S_h)
    structures = None if point['structure'] is None else point['structure']['structures']
    names = (['structure.T_H', 'structure.T_0'] if structures else [])+[f'pair.{k}.{t}' for k in ('seg35', 'seg4') if point['pair'][k] is not None for t in ('T_H', 'T_0')]
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in names}
    void = dict(positive_pairs=0, no_structure=0, constant_actual=0)
    index = np.arange(np.asarray(D).shape[0])
    valid = 0
    while valid < n and names:
        counts = np.bincount(rng.integers(0, n_subjects, n_subjects), minlength=n_subjects)
        w = counts[index]*counts[index+1]
        stats, reason = a_statistics(D, X_pq, X_h, S, S_pq, S_h, w=w, structures=structures)
        if stats is None:
            void[reason] += 1
            continue
        valid += 1
        for name in names:
            part, *rest = name.split('.')
            node = stats[part] if part == 'structure' else (stats['pair'][rest[0]] or {})
            value = (node or {}).get(rest[-1])
            draws[name].append(np.nan if value is None else value)
    intervals = {name: dict(lo=float(np.nanpercentile(v, 2.5)), hi=float(np.nanpercentile(v, 97.5)), n_valid=int(np.isfinite(v).sum()), n_valid_draws=valid)     # revision 1 (review minor 6):
                 for name, v in draws.items() if np.isfinite(v).any()}                                                                                           # the draws this statistic took part in
    return dict(point=point, intervals=intervals, void_draws=void, n=n, seed=seed, role='described only (D87 = (c), D94 = (a)): no reading; an interval is conditional on the valid draws',
                structures_rule='C_A from the original data (>= 10 pairs); in a draw those of C_A with >= 5 distinct pairs, both defined, actual not all equal (an implementer reading of plan.md 5.1)')


def nanmean(values):
    """The mean of the finite values; NaN, without a warning, when there is none."""
    values = np.asarray(values, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float('nan')


def constant_prediction(values):
    """K of plan.md 3.1: for pair i (a 1-D array over the pairs in chain order) the mean of the finite values of the pairs j with |j - i| >= 2 (no shared subject); NaN when there is none."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError('constant_prediction takes one value per pair')
    return np.array([nanmean([values[j] for j in range(values.size) if abs(j-i) >= 2]) for i in range(values.size)])


# ---- B class and the reading ------------------------------------------------------------------------------------------------------------------------
def b_class(ause_table, index=None, *, r_min=R_MIN, baselines=JUDGED_BASELINES):
    """ause_table: {score: [AUSE per pair in pair order]} with 'q' and the 8 judged baselines.  Returns the per-baseline Delta, interval, R and pass, and the class's pass (all 8).  Not judged (None) when
    a value is missing (a pair without an error in its mask)."""
    index = np.arange(N_PAIRS) if index is None else np.asarray(index)
    missing = sorted(k for k in ('q',)+tuple(baselines) if k not in ause_table or len(ause_table[k]) != len(index) or any(v is None for v in ause_table[k]))
    if missing:
        return dict(judged=False, reason=f'AUSE missing for {missing}', passed=None, baselines={})
    q = np.asarray(ause_table['q'], dtype=np.float64)
    rows = {}
    for k in baselines:
        base = np.asarray(ause_table[k], dtype=np.float64)
        delta = base-q
        interval = block_interval(delta, index)
        total = float(base.sum())
        ratio = float(delta.sum()/total) if total > 0 else None
        ok = bool(interval['lo'] > 0 and ratio is not None and ratio >= r_min)
        rows[k] = dict(delta=delta.tolist(), interval=interval, LB=interval['lo'], R=ratio, passed=ok, n_positive=int((delta > 0).sum()))
    return dict(judged=True, passed=all(r['passed'] for r in rows.values()), failed=[k for k, r in rows.items() if not r['passed']], baselines=rows, r_min=r_min,
                rule='LB_k > 0 (2.5 percentile of p1_core.block_interval of Delta_k = AUSE_k - AUSE_q) and R_k = sum Delta_k / sum AUSE_k >= .05, for every one of the 8 (intersection-union)')


READ_STOPPED = '停在读真值之前，不算读数，交总负责人'
READ_NOT_EVALUATED = '真值未读，还没有读数'
READ_UNDECIDED_EVAL = '不判（评测侧控制不过）；最后 1 次选择读数算用掉，不重跑'
READ_UNDECIDED_B = '不判（B 类不判）；交总负责人'
READ_PASS = 'PQ 承重（误差定位）'
READ_FAIL = 'PQ 不承重：这层贝叶斯只是标签，"贝叶斯"只靠 C（C22.B2-3）'
METHOD_KEYS = ('m1', 'm2', 'm3', 'm4', 'm5', 'm6')
E1_PASSING = ('bitwise', 'within_1e-12')


def reading(*, method_controls, released, eval_controls, b, truth_read):
    """plan.md section 6, D87 = (c): only class B is judged; class A never enters.
    method_controls: exactly {m1..m6: bool} (revision 1, review minor 2: a missing or an extra key counts as not passed); released: the lead's release found committed; eval_controls: {'e1': 'bitwise' |
    'within_1e-12' | False | None, 'e2': bool, 'complete': the evaluation record says every pair was evaluated and nothing stopped it, 'integrity': the files read here are the listed and recorded ones};
    truth_read: the lock of the evaluation unit exists (revision 1, review important 1: the lock, not a record, says whether the truth was read).
    Rows: the truth not read -> the first row (stopped before the truth, not a reading) when a method control or the release does not hold, the second row when e2 failed (counted as used), else no
    reading yet; the truth read -> never the first row: any control that does not hold, a missing or incomplete evaluation record, a changed file -> the second row (不判, counted as used); then class B."""
    mc = method_controls or {}
    failed_method = sorted(set(k for k in METHOD_KEYS if mc.get(k) is not True) | set(k for k in mc if k not in METHOD_KEYS))
    ec = eval_controls or {}
    e1, e2, complete, integrity = ec.get('e1'), ec.get('e2'), ec.get('complete'), ec.get('integrity')
    if not truth_read:
        if failed_method or released is not True:
            return dict(row='method_controls_or_release', reading=READ_STOPPED, passed=None, counts_as_reading=False, failed=failed_method, released=released, truth_read=False)
        if e2 is False:
            return dict(row='eval_controls', reading=READ_UNDECIDED_EVAL, passed=None, counts_as_reading=True, e1=e1, e2=e2, truth_read=False, reason='e2 not passed before the truth')
        return dict(row='not_evaluated', reading=READ_NOT_EVALUATED, passed=None, counts_as_reading=False, truth_read=False)
    if failed_method or released is not True:
        return dict(row='eval_controls', reading=READ_UNDECIDED_EVAL, passed=None, counts_as_reading=True, failed=failed_method, released=released, truth_read=True,
                    reason='the truth was read although a method control or the release does not hold')
    if e1 not in E1_PASSING or e2 is not True or complete is not True or integrity is not True:
        return dict(row='eval_controls', reading=READ_UNDECIDED_EVAL, passed=None, counts_as_reading=True, e1=e1, e2=e2, complete=complete, integrity=integrity, truth_read=True)
    if b is None or not b.get('judged'):
        return dict(row='b_not_judged', reading=READ_UNDECIDED_B, passed=None, counts_as_reading=True, truth_read=True, reason=None if b is None else b.get('reason'))
    common = dict(counts_as_reading=True, truth_read=True, e1=e1, e1_listed_separately=e1 == 'within_1e-12', class_a='described only (D87 = (c)); "预测 Dice" may not be said')
    if b['passed']:
        return dict(row='b_pass', reading=READ_PASS, passed=True, **common)
    return dict(row='b_fail', reading=READ_FAIL, passed=False, failed=b['failed'], **common)
