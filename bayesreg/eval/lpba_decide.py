"""bayesreg.eval.lpba_decide -- the decisions L1 (lpba54) and L2 (lpba56) of the LPBA40 second test set from the evaluation entry's records (prereg
docs/bayesreg/notes/LPBA40第二测试集预登记_20261119.md §6, §12 item 13; contract C26.5-8; D146).

A new module (D146): bayesreg/eval/holdout_decide.py, its constants and its tests are not changed (the B-line decision code stays frozen on main).  What is reused is imported
from it, never copied: draw_idx (n = 17, seed 20261119), boot_means, pvalues, interval, holm (ties in the main family's order), wilcoxon_p, collect, matrices, official_rows,
dsc30_draws, dsc30_point, described_only, certificate_summary, has_identity, has_arm, complete_mask, method_name.  Rewritten with their source named (C26.5-8):
  judge_family       from holdout_decide.judge_family (holdout_decide.py:403-465): the suspicious threshold SUSPICIOUS_STOP = 2 (was 3 at :426, :437, :446; S5 likewise, so S5
                     exists when exactly 1 pair is suspicious); the paired t gate (prereg §6.4): each class's p^c = max(p^B, p^t), Holm on p^c, ties in the main family's order;
                     a k not judged (stopped, suspicious >= 2, lacking records) has p^B and p^t both 1; S1 for the ITT guard with both p's on the complete pairs (df n_c - 1,
                     n_c < 2: both p = 1) at k's Holm threshold.  With t_gate False and suspicious_stop 3 it returns holdout_decide.judge_family's document item for item (T12).
  sdlogj_guard_row   from holdout_decide.sdlogj_guard_row (holdout_decide.py:389-400): the upper end max(U^B, U^t) (prereg §6.5): U^B the 97.5 percentile of the valid draws on the
                     complete pairs (a draw with none void, not redrawn), U^t = D̄_c + t_{.975, n_c-1} sd_c / sqrt(n_c) (n_c < 2: +inf); holds iff max < .02; no complete pair:
                     it does not hold.  With t_gate False it is holdout_decide's row.
  t p values         p_t (prereg §6.4): F the t distribution's cdf with df = n - 1 (scipy.stats.t.cdf), SE = sd(ddof 1)/sqrt(n): sup 1 - F((Δ̄ - .005)/SE), inf F((Δ̄ + .005)/SE),
                     eq max(1 - F((Δ̄ + .01)/SE), F((Δ̄ - .01)/SE)); SE = 0: 0 when the numerator lies strictly on the alternative's side, else 1; n < 2: all 1.
Rules (prereg §6): pairs i = 0..16 in test_pairs.tsv row order; ITT (a cell without output takes the pair's identity row); one idx for all (draw_idx seed SEED, B 10000);
families L1 = lpba54, L2 = lpba56 over the main family (configc, ants_s, fireants_greedy, fireants_syn, convexadam_v4seg), alpha .025 one-sided, ROPE ±.005, 相当 ±.01;
verdict order 优于 → 劣于 → 相当 → 无法下结论; the ITT guard; the SDlogJ guard on 优于 only; pairs still without a complete evaluation record after the reevaluation leave every
arm (乙), idx drawn on the n' positions, df n' - 1, n' < MISSING_MIN_PAIRS (16) -> both families only described; L3 (described: official Dice, DSC30, HD95, SDlogJ, brain log J,
the identity reference): unadjusted bootstrap and t intervals only.  No seg4 = G rule, no F4, no PQ, no repair ablation (prereg §6, §4.5).
The code identity (C26.9-4): main() checks rule one with the LPBA40 constants (bayesreg.eval.lpba_eval.rule_one) before it reads anything else.
"""
from pathlib import Path
import json
import sys

import numpy as np

from . import holdout_decide as HD

SEED = 20261119
B = 10000
N_PAIRS = 17
MISSING_MIN_PAIRS = 16
ROPE = HD.ROPE                    # .005 (prereg §6.2)
EQ_BOUND = HD.EQ_BOUND            # .01 (prereg §6.4)
ALPHA = HD.ALPHA                  # .025
M_MAIN = 5
SDLOGJ_GUARD = .02                # prereg §6.5 ("界 .02 照抄")
SUSPICIOUS_STOP = 2               # prereg §6.6, 裁决 (8); holdout_decide's 3 at :426 (judge_family) and S5
PREREG_SHA256 = "06121278d8b229401afea7cb86c841f80ba02f23bc40f8c5f5594c7da8d95643"     # C26.2-4; the evaluation entry pins the same (a test compares)
FAMILIES = (("L1", "lpba54"), ("L2", "lpba56"))
MAIN_ORDER = ("configc", "ants_s", "fireants_greedy", "fireants_syn", "convexadam_v4seg")     # prereg §6.4: ties in this order (the arm table's main_family; a test compares)
S4_ARM, S4_STATUS, S2_STATUS = HD.S4_ARM, HD.S4_STATUS, HD.S2_STATUS
LOWER_BETTER = HD.LOWER_BETTER
DESCRIBED = HD.DESCRIBED
SUP, EQ, INF, NONE = HD.SUP, HD.EQ, HD.INF, HD.NONE
SUP_ITT, INF_ITT, SUP_GUARD = HD.SUP_ITT, HD.INF_ITT, HD.SUP_GUARD
NOT_JUDGED, NOT_JUDGED_MISSING = HD.NOT_JUDGED, HD.NOT_JUDGED_MISSING
DESCRIBED_FEW = "只描述（n′ < 16）"


class DecisionRefused(HD.DecisionRefused):
    """The decision code does not run (pins missing or the records do not fit)."""


# ---- the paired t p values (prereg §6.4) -------------------------------------------------------------------------------------------------------------------
def tvalues(delta, rope=ROPE, eq_bound=EQ_BOUND):
    """The three one-sided paired t p values of prereg §6.4 on the differences `delta` (finite), df = n - 1; SE = 0: 0 on the alternative's side strictly, else 1; n < 2: 1."""
    from scipy import stats
    d = np.asarray(delta, dtype=np.float64)
    n = int(d.size)
    if n < 2:
        return dict(sup=1., eq=1., eq_lo=1., eq_hi=1., inf=1., n=n, df=None, mean=float(d.mean()) if n else None, se=None, rule="n < 2: every p = 1")
    mean = float(d.mean())
    se = float(d.std(ddof=1)/np.sqrt(n))
    df = n-1
    equal = bool(np.all(d == d[0]))           # review minor 1: "所有差相等" judged on the differences themselves (a float std of equal values need not be exactly 0)

    def upper(x):                 # alternative "> bound": p = 1 - F((mean - bound)/SE)
        if equal:
            return 0. if mean-x > 0 else 1.
        return float(1.-stats.t.cdf((mean-x)/se, df))

    def lower(x):                 # alternative "< bound": p = F((mean - bound)/SE)
        if equal:
            return 0. if mean-x < 0 else 1.
        return float(stats.t.cdf((mean-x)/se, df))
    lo, hi = upper(-eq_bound), lower(eq_bound)
    return dict(sup=upper(rope), inf=lower(-rope), eq=max(lo, hi), eq_lo=lo, eq_hi=hi, n=n, df=df, mean=mean, se=se, all_equal=equal, rope=rope, eq_bound=eq_bound)


def t_interval(delta, level=.975):
    """Δ̄ ± t_{.975, n-1} SE (prereg §6.3); None for n < 2."""
    from scipy import stats
    d = np.asarray(delta, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return None
    half = float(stats.t.ppf(level, d.size-1)*d.std(ddof=1)/np.sqrt(d.size))
    return [float(d.mean())-half, float(d.mean())+half]


def combine(p_boot, p_t):
    """p^c = max(p^B, p^t) per class (prereg §6.4 合成)."""
    return {cls: max(float(p_boot[cls]), float(p_t[cls])) for cls in ("sup", "eq", "eq_lo", "eq_hi", "inf")}


# ---- the SDlogJ guard (from holdout_decide.sdlogj_guard_row, holdout_decide.py:389-400) --------------------------------------------------------------------------
def sdlogj_guard_row(sdlogj, imputed, missing, method, k, pos, idx, bound, *, t_gate=True):
    """The SDlogJ guard of k (prereg §6.5): the official SDlogJ difference (method - k) on the complete pairs (both arms with their own output), the same idx restricted to
    them (void draws not redrawn); U^B the 97.5 percentile; with t_gate also U^t = D̄_c + t_{.975, n_c-1} sd_c/sqrt(n_c) (n_c < 2: +inf); holds iff max(U^B, U^t) < bound.
    No complete pair: it does not hold.  t_gate False: holdout_decide.sdlogj_guard_row's row exactly (copied lines 392-400)."""
    own = ~imputed[method][pos] & ~imputed[k][pos] & ~missing[method][pos] & ~missing[k][pos]          # holdout_decide.py:392
    diff = sdlogj[method][pos]-sdlogj[k][pos]
    use = own & np.isfinite(diff)
    means = HD.boot_means(np.where(use, diff, 0.), idx, use)
    iv = HD.interval(means)
    upper_b = None if iv is None else iv[1]
    if not t_gate:                                                                                         # holdout_decide.py:396-400 as they are
        return dict(rule="official SDlogJ, method - arm, complete pairs only; holds iff the 95% interval's upper end < bound", bound=bound, n=int(use.sum()),
                    mean=float(diff[use].mean()) if use.any() else None, interval=iv, upper=upper_b, void_draws=int(np.isnan(means).sum()),
                    holds=bool(upper_b is not None and upper_b < bound))
    from scipy import stats
    n_c = int(use.sum())
    if n_c >= 2:
        upper_t = float(diff[use].mean()+stats.t.ppf(.975, n_c-1)*diff[use].std(ddof=1)/np.sqrt(n_c))
    else:
        upper_t = float("inf")
    upper = None if upper_b is None else max(upper_b, upper_t)
    return dict(rule="official SDlogJ, method - arm, complete pairs only; holds iff max(U^B, U^t) < bound (prereg §6.5)", bound=bound, n=n_c,
                mean=float(diff[use].mean()) if use.any() else None, interval=iv, upper=upper, upper_boot=upper_b, upper_t=upper_t, t_interval=t_interval(diff[use]),
                void_draws=int(np.isnan(means).sum()), holds=bool(upper is not None and upper < bound))


# ---- one family (from holdout_decide.judge_family, holdout_decide.py:403-465) --------------------------------------------------------------------------------------
def judge_family(endpoint, values, imputed, suspicious, stopped, table, idx, *, sdlogj, guard, positions=None, alpha=ALPHA, missing=None, suspicious_counts=None,
                 t_gate=True, suspicious_stop=SUSPICIOUS_STOP):
    """Verdicts of the main family on `endpoint` (module docstring).  Arguments as holdout_decide.judge_family; t_gate adds the paired t gate (prereg §6.4), suspicious_stop the
    P19 threshold (prereg §6.6: 2)."""
    if guard is None:
        raise DecisionRefused("the bound of the SDlogJ guard is None")
    method = HD.method_name(table)
    order = list(table["main_family"])
    pos = np.arange(len(values[method])) if positions is None else np.asarray(positions)
    missing = {a: np.zeros(len(values[method]), dtype=bool) for a in values} if missing is None else missing
    rows = {}
    for k in order:
        lacking = bool(missing[k][pos].any() or missing[method][pos].any())
        delta = values[method][pos]-values[k][pos]
        if lacking:
            delta = np.where(np.isfinite(delta), delta, 0.)
        elif not np.all(np.isfinite(delta)):
            raise DecisionRefused(f"{endpoint} {k}: {int((~np.isfinite(delta)).sum())} non-finite differences in the main family: hand over to BZ")
        means = HD.boot_means(delta, idx)
        n_suspicious = int(suspicious[k][pos].sum()) if suspicious_counts is None else int(suspicious_counts[k])
        p_boot = HD.pvalues(means)
        not_judged = stopped[k] or n_suspicious >= suspicious_stop or lacking                       # holdout_decide.py:426 (3 -> suspicious_stop)
        complete = ~imputed[method][pos] & ~imputed[k][pos]
        s1_means = HD.boot_means(delta, idx, complete)
        s1_boot = HD.pvalues(s1_means)
        if t_gate:
            p_t = tvalues(delta)
            s1_t = tvalues(delta[complete]) if int(complete.sum()) >= 2 else dict(tvalues(delta[complete]), sup=1., eq=1., eq_lo=1., eq_hi=1., inf=1.)
            if int(complete.sum()) < 2:                                                             # prereg §6.6: n_c < 2 -> both p of S1 are 1
                s1_boot = dict(s1_boot, sup=1., eq=1., eq_lo=1., eq_hi=1., inf=1.)
            p = dict(p_boot, **combine(p_boot, p_t))
            s1_p = dict(s1_boot, **combine(s1_boot, s1_t))
        else:
            p, s1_p = p_boot, s1_boot
        if not_judged:                                                                              # prereg §6.4 置 1: both p's of k are 1
            p = dict(p, sup=1., eq=1., eq_lo=1., eq_hi=1., inf=1.)
            if t_gate:
                p_boot = dict(p_boot, sup=1., eq=1., eq_lo=1., eq_hi=1., inf=1.)
                p_t = dict(p_t, sup=1., eq=1., eq_lo=1., eq_hi=1., inf=1.)
        rows[k] = dict(mean=float(delta.mean()), median=float(np.median(delta)), wins=int((delta > 0).sum()), n=int(delta.size), wilcoxon_p=HD.wilcoxon_p(delta),
                       interval=HD.interval(means), p=p, not_judged=not_judged, n_suspicious=n_suspicious, stopped=stopped[k], lacking_records=lacking,
                       imputed_method=int(imputed[method][pos].sum()), imputed_arm=int(imputed[k][pos].sum()),
                       s1=dict(n=int(complete.sum()), mean=float(delta[complete].mean()) if complete.any() else None, p=s1_p, interval=HD.interval(s1_means)),
                       s5=None, sdlogj_guard=sdlogj_guard_row(sdlogj, imputed, missing, method, k, pos, idx, guard, t_gate=t_gate))
        if t_gate:
            rows[k].update(p_boot=p_boot, p_t=p_t, t_interval=t_interval(delta))
            rows[k]["s1"].update(p_boot=s1_boot, p_t=s1_t, t_interval=t_interval(delta[complete]))
        keep = ~suspicious[k][pos]
        if 0 < n_suspicious < suspicious_stop and not keep.all():                                   # holdout_decide.py:437 (3 -> suspicious_stop): S5
            s5 = HD.boot_means(delta, idx, keep)
            rows[k]["s5"] = dict(n=int(keep.sum()), mean=float(delta[keep].mean()), interval=HD.interval(s5))
    steps = {cls: HD.holm({k: rows[k]["p"][cls] for k in order}, order, alpha) for cls in ("sup", "eq", "inf")}
    for k in order:
        row = rows[k]
        row["holm"] = {cls: steps[cls][k] for cls in steps}
        row["rejected_classes"] = [cls for cls in ("sup", "inf", "eq") if steps[cls][k]["rejected"]]
        if row["not_judged"]:
            row["verdict"] = NOT_JUDGED_MISSING if row["lacking_records"] and not (row["stopped"] or row["n_suspicious"] >= suspicious_stop) else NOT_JUDGED   # :446
            continue
        rejected = row["rejected_classes"]
        verdict = {"sup": SUP, "inf": INF, "eq": EQ}[rejected[0]] if rejected else NONE
        if verdict == SUP and row["imputed_arm"] > 0:
            threshold = steps["sup"][k]["threshold"]
            row["itt_guard"] = dict(rule="S1 at k's Holm step", threshold=threshold, s1_p=row["s1"]["p"]["sup"], holds=row["s1"]["p"]["sup"] <= threshold)
            if not row["itt_guard"]["holds"]:
                verdict = SUP_ITT
        if verdict == INF and row["imputed_method"] > 0:
            threshold = steps["inf"][k]["threshold"]
            row["itt_guard"] = dict(rule="S1 at k's Holm step", threshold=threshold, s1_p=row["s1"]["p"]["inf"], holds=row["s1"]["p"]["inf"] <= threshold)
            if not row["itt_guard"]["holds"]:
                verdict = INF_ITT
        if verdict in (SUP, SUP_ITT) and not row["sdlogj_guard"]["holds"]:
            row["verdict_before_guard"] = verdict
            verdict = SUP_GUARD
        row["verdict"] = verdict
    out = dict(endpoint=endpoint, m=len(order), alpha=alpha, rope=ROPE, eq_bound=EQ_BOUND, sdlogj_guard=guard, arms=rows, order=order,
               holm_order={cls: sorted(order, key=lambda k: steps[cls][k]["step"]) for cls in steps})
    if t_gate:
        out.update(t_gate="p^c = max(p^B, p^t); Holm on p^c (prereg §6.4)", suspicious_stop=suspicious_stop)
    return out


def describe(values, imputed, table, idx, key, sign):
    """L3 (prereg §6.4): mean difference, the unadjusted bootstrap and t intervals for every main arm (holdout_decide.describe plus the t interval)."""
    method = HD.method_name(table)
    out = HD.describe(values, imputed, table, idx, key, sign)
    for k, row in out.items():
        delta = sign*(values[key][method]-values[key][k])
        row["t_interval"] = t_interval(delta)
    return out


def identity_reference(records, rows, idx, keys):
    """L3: the identity reference (d = 0), described: its mean per key over the pairs; in no family, not tested."""
    out = {}
    for key in keys:
        raw = [(((records.get(r[3]) or {}).get("identity") or {}).get("metrics") or {}).get(key) for r in rows]
        vals = np.array([np.nan if v is None else float(v) for v in raw], dtype=np.float64)
        out[key] = dict(mean=float(np.nanmean(vals)) if np.isfinite(vals).any() else None, n=int(np.isfinite(vals).sum()))
    return dict(role="reference only (prereg §4.4): through the same entry, in no family, not tested", endpoints=out)


def tet24_summary(records, table, rows):
    """Described only (prereg §5.3 折叠 (2)): per arm over its cells with output, the pairs with a cell ratio <= 0 in each scheme and the minimum ratio."""
    out = {}
    for arm in table["arms"]:
        per = {}
        for r in rows:
            entry = (((records.get(r[3]) or {}).get("arms") or {}).get(arm["name"]) or {})
            summary = (entry.get("tet24") or {}).get("tet_ratio_summary") if entry.get("outcome") == "output" else None
            for scheme, value in (summary or {}).items():
                row = per.setdefault(scheme, dict(n=0, pairs_with_nonpositive_cells=0, min_ratio=None))
                row["n"] += 1
                row["pairs_with_nonpositive_cells"] += int((value or {}).get("cells_nonpositive") or 0) > 0
                low = (value or {}).get("min_ratio")
                if isinstance(low, (int, float)):
                    row["min_ratio"] = low if row["min_ratio"] is None else min(row["min_ratio"], low)
        out[arm["name"]] = per
    return dict(role="described only (prereg §5.3; C24.12-1): neither it nor BL1 implies the other", arms=out)


# ---- the whole decision ----------------------------------------------------------------------------------------------------------------------------------------
def decide(records, table, rows, *, sdlogj_guard=SDLOGJ_GUARD, idx=None):
    """The decision document.  records: {pair: merged record} (holdout_decide.collect); rows: the 17 rows (index, fixed, moving, name); idx: the resampling of the 17 pairs
    (default draw_idx(seed SEED, n 17))."""
    if sdlogj_guard is None:
        raise DecisionRefused("the bound of the SDlogJ guard is None")
    if list(table["main_family"]) != list(MAIN_ORDER):
        raise DecisionRefused(f"main_family {table['main_family']} is not {list(MAIN_ORDER)} (prereg §6.4: the tie order)")
    full_idx = HD.draw_idx(seed=SEED, n=len(rows)) if idx is None else idx
    complete = HD.complete_mask(records, table, rows)
    missing_pairs = [rows[i][3] for i in np.flatnonzero(~complete)]
    conflicts = {name: r["_conflicts"] for name, r in records.items() if r.get("_conflicts")}
    head = dict(seed=SEED, B=int(full_idx.shape[0]), missing_rule="yi", missing_pairs=missing_pairs, sdlogj_guard=sdlogj_guard, suspicious_stop=SUSPICIOUS_STOP,
                merge_conflicts=conflicts, n_pairs=len(rows))
    use, idx, few = list(range(len(rows))), full_idx, False
    if missing_pairs:                                         # prereg §6.6 (乙): the pair leaves every arm; idx on the n' positions; n' < 16 -> described
        use = [i for i in range(len(rows)) if complete[i]]
        few = len(use) < MISSING_MIN_PAIRS
        idx = HD.draw_idx(seed=SEED, n=len(use), b=full_idx.shape[0])
    sub_rows = [rows[i] for i in use]
    keys = tuple(dict.fromkeys(("lpba54", "lpba56")+DESCRIBED))
    values, imputed, suspicious, stopped, status, missing = HD.matrices(records, table, sub_rows, keys)
    _, _, suspicious_all, _, _, _ = HD.matrices(records, table, rows, ())
    counts = {a: int(v.sum()) for a, v in suspicious_all.items()}
    method = HD.method_name(table)
    common = dict(sdlogj=values["sdlogj_official"], guard=sdlogj_guard, missing=missing, suspicious_counts=counts)
    families = {}
    for name, endpoint in FAMILIES:
        family = judge_family(endpoint, values[endpoint], imputed, suspicious, stopped, table, idx, **common)
        families[name] = HD.described_only(family, DESCRIBED_FEW) if few else family
    l3 = dict(rule="unadjusted 95% intervals (bootstrap percentile and t), not judged (prereg §6.4 L3)",
              endpoints={key: describe(values, imputed, table, idx, key, -1 if key in LOWER_BETTER else 1) for key in DESCRIBED},
              identity=identity_reference(records, sub_rows, idx, ("lpba54", "lpba56", "sdlogj_official")))
    rows_off = HD.official_rows(records, table, sub_rows)
    l3["dsc30"] = {}
    if rows_off[method] is not None:
        method_draws = HD.dsc30_draws(rows_off[method], idx)
        for a in table["arms"]:
            if a["name"] != method and rows_off[a["name"]] is not None:
                l3["dsc30"][a["name"]] = dict(family=a["family"], method=HD.dsc30_point(rows_off[method]), arm=HD.dsc30_point(rows_off[a["name"]]),
                                              interval=HD.interval(method_draws-HD.dsc30_draws(rows_off[a["name"]], idx)), written="method - arm")
    sd = values["sdlogj_official"]
    l3["sdlogj_official_mean_le_010"] = dict(
        itt={a["name"]: bool(np.nanmean(sd[a["name"]]) <= .10) for a in table["arms"] if np.isfinite(sd[a["name"]]).any()},
        outputs_only={a["name"]: bool(np.nanmean(sd[a["name"]][~imputed[a["name"]]]) <= .10) for a in table["arms"] if np.isfinite(sd[a["name"]][~imputed[a["name"]]]).any()})
    certified = np.array([s == S2_STATUS for s in status[method]])
    sensitivity = {}
    for name, endpoint in FAMILIES:
        part = dict(S2={k: dict(n=int(certified.sum()), interval=HD.interval(HD.boot_means(np.nan_to_num(values[endpoint][method]-values[endpoint][k]), idx, certified)))
                        for k in table["main_family"]})
        fallback = np.array([s == S4_STATUS for s in status[method]])
        if S4_ARM in values[endpoint] and fallback.any():
            delta = np.nan_to_num(values[endpoint][method]-values[endpoint][S4_ARM])
            part["S4"] = dict(n=int(fallback.sum()), mean=float(delta[fallback].mean()), interval=HD.interval(HD.boot_means(delta, idx, fallback)))
        else:
            part["S4"] = None
        sensitivity[name] = part
    no_output = {}
    for a in table["arms"]:
        classes = set()
        for r in sub_rows:
            entry = ((records.get(r[3]) or {}).get("arms") or {}).get(a["name"]) or {}
            if entry and entry.get("outcome") != "output":
                classes.add(entry.get("no_output_class"))
        no_output[a["name"]] = dict(n=int(imputed[a["name"]].sum()), stopped=stopped[a["name"]], classes=sorted(classes-{None}),
                                    missing_records=int(missing[a["name"]].sum()), suspicious_pairs_all=counts[a["name"]])
    fallbacks = sum(s in ("hx_fallback_configC", "hx_fallback_identity") for s in status[method])
    return dict(head, reading="decided" if not few else DESCRIBED_FEW, n_used=len(use), positions=[dict(position=j, pair=r[3], index=r[0]) for j, r in enumerate(sub_rows)],
                idx_sha256=HD.idx_sha256(idx), L1=families["L1"], L2=families["L2"], L3=l3, sensitivity=sensitivity, no_output=no_output,
                certificates=HD.certificate_summary(records, table, sub_rows), tet24=tet24_summary(records, table, sub_rows),
                method_line_status={s: status[method].count(s) for s in sorted(set(s for s in status[method] if s))},
                fallback_note=dict(n_fallback=fallbacks, many=fallbacks >= 6, rule="prereg §6.9: >= 6 fallback pairs -> the method vs configc reading is the S4 noise reading"))


def main(argv=None):
    import argparse
    from bayesreg.eval import lpba_eval as E
    from bayesreg.eval import holdout as H
    parser = argparse.ArgumentParser(prog="python -m bayesreg.eval.lpba_decide", description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", type=Path, help="optional: must equal the output directories of the master and reevaluation locks, in order")
    parser.add_argument("--out", required=True, type=Path, help="decision JSON (must not exist)")
    args = parser.parse_args(argv)
    if PREREG_SHA256 != E.PREREG_SHA256:
        print("refused: PREREG_SHA256 differs from the evaluation entry's", file=sys.stderr)
        return 2
    try:
        identity = E.rule_one()                                    # C26.9-4: before anything else is read
        prereg = E.check_prereg()
        table, table_record = E.load_arm_table("lpba40_test")
        lc = E._lpba_module("lpba_common")
        rows = E.pair_rows("lpba40_test", lc.read_lists())
        if len(rows) != N_PAIRS:
            raise DecisionRefused(f"{len(rows)} pairs, {N_PAIRS} expected")
        lines = E.run_lines("lpba40_test")
        if not lines:
            raise DecisionRefused(f"{E.master_lock('lpba40_test')}: no master lock; nothing was evaluated")
        outs = [line["out_dir"] for _, line in lines]
        for path, line in lines:
            if line.get("arm_table_sha256") != table_record["sha256"] or line.get("prereg_sha256") != prereg["sha256"] or \
                    line.get("manifest_sha256") != lines[0][1].get("manifest_sha256") or line.get("pair_set") != "lpba40_test":
                raise DecisionRefused(f"{path}: another arm table, preregistration, manifest or pair set than the master lock's")
        if args.runs is not None and [str(Path(o).resolve()) for o in args.runs] != [str(Path(o).resolve()) for o in outs]:
            raise DecisionRefused(f"--runs {[str(o) for o in args.runs]} are not the locks' output directories {outs} (in lock order)")
        records = HD.collect(outs, rows)
        document = dict(decide(records, table, rows), prereg=prereg, arm_table=table_record, runs=outs, locks=[str(p) for p, _ in lines], rule_one=identity)
    except (H.HoldoutRefused, HD.DecisionRefused) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    with open(args.out, "x", encoding="utf-8") as stream:
        stream.write(H.dumps(document)+"\n")
    return 0


__all__ = ["B", "FAMILIES", "MISSING_MIN_PAIRS", "N_PAIRS", "PREREG_SHA256", "SDLOGJ_GUARD", "SEED", "SUSPICIOUS_STOP", "combine", "decide", "judge_family",
           "sdlogj_guard_row", "t_interval", "tvalues"]


if __name__ == "__main__":
    sys.exit(main())
