"""bayesreg.eval.holdout_decide -- the decisions F1-F4 of the B-line hold-out from the evaluation entry's records (preregistration 留出集预登记_20261107.md §4;
contract C22.H7 of revision 7, adopted by BZ 2026-11-09).

It reads only the per-pair records that bayesreg.eval.holdout wrote (<out>/pairs/<pair>.json) in the output directories of the master lock and of the
reevaluation locks, in lock order (holdout.run_lines; review I8: never a directory given on the command line); no truth, no field.  Why here (the brief let
the implementer choose): it is one step of the same evaluation package (bayesreg.eval.paired is its development-set counterpart), it pins the same constants
as the entry (the preregistration, the arm table) and is imported by the tests with the package; experiments/ holds the one-off tools.

Rules (preregistration 4.1-4.7, the lead's changes 4, 6, 7 of 11-07, revision 1b items 2-4, BZ 2026-11-07 ~6:54 PM items (3) and (4), the lead's rulings of
2026-11-09 ~5:59 PM (1)-(3), (8), (9)):
  pairs       i = 0..49 in holdout_B_pairs.tsv row order.  collect merges the runs' records in lock order (a partial record is never overwritten; per arm
              the latest record that holds it; an arm held by two records with different numbers is listed in merge_conflicts, review m11).  Pairs still
              without a complete record (N3), by MISSING_PAIR_RULE: 甲 "jia" no confirmatory F1-F4 ("交 BZ"); 乙 "yi" (default) the pair leaves every arm,
              F1-F4 on the n' others with idx drawn on n' positions (seed and B unchanged), n' < 45 -> "只描述（n′ < 45）"; 丙 "bing" only the arms lacking
              a record are "不判（缺评测记录）" (p = 1; the method lacking one leaves every arm so).  The n' positions (𝒩) are in tsv row order and written
              into the decision (positions).
  ITT         a cell without output (any written no-output class) takes the pair's identity row (d = 0, computed by the entry with the same functions).
  resampling  rng = numpy.random.default_rng(20261107); idx_b = rng.integers(0, n', n') drawn B = 10000 times in sequence (one call per b, as written);
              Δ̄*_{k,b} = mean(Δ_{k, idx_b}); the same idx for every comparison, seg35, seg4, the described endpoints, the SDlogJ guard and F4.
  p values    p_sup = (1 + #{Δ̄* <= +.005})/(B + 1); p_inf = (1 + #{Δ̄* >= -.005})/(B + 1) (ROPE, ±.005);
              p_eq = max(p_lo, p_hi), p_lo = (1 + #{Δ̄* <= -.01})/(B + 1), p_hi = (1 + #{Δ̄* >= +.01})/(B + 1) (EQ_BOUND, ±.01; BZ 11-07 item (4)).
              Equality counts (boundary).  A non-finite Δ of a judged arm refuses the decision (review m5: the main family has no void draws).
  Holm        per class (sup, eq, inf) over the m = 5 arms of the main family, alpha .025: p~_(j) = max_{i <= j} min(1, (m - i + 1) p_(i)), rejected while
              p~_(j) <= alpha, stopping at the first that is not; ties in p ordered by the main family's order.  Verdict: the first rejected class in the
              order sup -> inf -> eq (优于 → 劣于 → 相当), none -> 无法下结论.  With ±.01 for 相当, sup and eq (or inf and eq) can both be rejected: the order
              decides and both are recorded (rejected_classes); no refusal.
  ITT guard   k judged 优于 while k has imputed pairs: S1 (the pairs where neither the method nor k is imputed; the same idx restricted to them) must have
              p_sup <= alpha / (m - j + 1), j = k's step in the sup ranking; otherwise "优于（仅意向处理；完整对不支持）".  Likewise 劣于 while the method has
              imputed pairs.  Only k's own verdict changes; the step-down and the arms after k are untouched.  相当 is not affected; a downgraded 优于 is
              not looked at again for 相当 (open: revision 2 writes it).
  SDlogJ      (BZ 11-07 item (3); the lead 11-09 ruling (2)) a 优于 of k (also the ITT-only one) also needs the upper end of the 95% percentile interval of
  guard       the official SDlogJ difference (method - k), over the complete pairs (both arms with their own output; an imputed identity has SDlogJ 0 and
              would make an arm with many failures look smooth), same idx restricted to them, < SDLOGJ_GUARD; otherwise "无法下结论（护栏）".  Only k's
              verdict changes.  SDLOGJ_GUARD is None until preregistration revision 2 writes the bound: the decision refuses to run.
  P19         an arm with >= 3 export-suspicious pairs over the whole list (review m4: counted on all pairs, not on the positions of one family), or a
              committed stop record (cells of class arm_stopped): "不判（导出约定可疑）", p = 1 in all three classes, kept in the family (m unchanged).  1-2
              suspicious pairs stay in the analysis; S5 leaves them out (descriptive).
  seg4 = G    a subject whose truth seg4 is not G(seg35) (the entry's seg4_equals_G): SEG4_G_RULE "a" (default) -> F2 whole family described, not judged;
              "b" -> the pairs holding such a subject leave F2, F2 is judged on the n' others with idx' drawn as above on n' positions (seed and B unchanged),
              n' < 45 -> described.  F1 is not affected.  The lead pins the rule after BZ (both have synthetic cases).
  described   a family that is only described keeps its numbers (means, intervals, p values) but no verdict, Holm step or guard outcome (review m13).
  F3          official Dice35, DSC30 (recomputed on each resample's pairs), the descriptive arms and the 90 s tier (seg35 and seg4, method - arm; review
              I9), every other endpoint: unadjusted 95% percentile intervals (numpy.percentile, linear) only, never judged.  HD95 and SDlogJ are written
              "arm - method" (smaller is better).  The official SDlogJ mean <= .10 is given with ITT and on the cells with output only (review m12).
  F4          C22.H7-2, -3 as adopted (revision 7d): 𝒫 = the positions of 𝒩 where the method has output, PQ has its outputs and the PQ evaluation record is
              complete; pairs whose PQ evaluation record is still incomplete after the reevaluation leave 𝒫 first (reasons listed), then Ev1-Ev3 are
              checked on 𝒫.  Δ_k(i) = AUSE_k(i) - AUSE_q(i) on Ω_e for the 8 baselines; R_k = ΣΔ_k / ΣAUSE_k over 𝒫 (ΣAUSE_k = 0: R None, fails);
              Δ̄*_{k,b} the mean over the positions of idx_b in 𝒫 (repeats counted; a draw with none is void, not redrawn); LB_k = percentile(valid Δ̄*_k,
              2.5) (D116 (a)); passes when LB_k > 0 and R_k >= .05 for every baseline.  Table rows: the method side's BZ decision (ii) -> "F4 只描述，不判
              （方法侧）"; Ev1-Ev3 -> "F4 不判（评测侧）"; 𝒫 empty or an Ω_e without errors -> "F4 不判"; n' < 45 or |𝒫| < 45 (D125 (a)) -> only described.
              There is no row for an unpinned replay gate: the entry refuses at (0) (C22.H7-3 revision 7a).
  ablation    (prereg revision 2 item 4; contract revision 9 draft C24.2; A5b) the repair ablation, described only (ablation_section): the entry's per-pair
              numbers (before = the pre-repair field, after = the method's output) in two tables by line status, mean, median, per pair and the unadjusted
              interval over the same idx restricted to the table's pairs; it changes none of F1-F4, the guard, Holm, ITT, S1-S5 (a test compares them).
The preregistration's sha256 is pinned (PREREG_SHA256, filled at P0 on 11-14; None refuses to run), and so is the SDlogJ bound (SDLOGJ_GUARD, P0).
The code identity (contract revision 9 draft C24.3-7; A5b): main() checks rule one (bayesreg.eval.holdout.rule_one: P1 from runs/holdout_B_v1/pinned_code.json) before it
reads anything else and records it.
"""
from pathlib import Path
import json
import sys

import numpy as np

SEED = 20261107
B = 10000
N_PAIRS = 50
ROPE = .005                       # 优于 / 劣于 (preregistration 4.3)
EQ_BOUND = .01                    # 相当: TOST on both sides at ±.01 (BZ 2026-11-07 ~6:54 PM item (4); the lead's ruling (1) of 11-09)
ALPHA = .025
M_MAIN = 5
SDLOGJ_GUARD = .02                # P0 (11-14): the bound of the SDlogJ guard (BZ 11-07 item (3); preregistration revision 2, adopted 11-10); None: the decision does not run
SEG4_G_RULE = "a"                 # (a) F2 described when a subject fails seg4 = G(seg35); (b) leave its pairs out, n' >= 45; the lead pins after BZ
SEG4_G_MIN_PAIRS = 45
MISSING_PAIR_RULE = "yi"          # N3, pairs still without an evaluation record after the reevaluation: "jia" (甲) no confirmatory F1-F4; "yi" (乙, default)
                                  # the pair leaves every arm, F1-F4 on n' pairs with idx drawn on n' positions, n' < 45 described; "bing" (丙) only the arms
                                  # that lack a record are not judged
MISSING_MIN_PAIRS = 45
PREREG_SHA256 = "df7b497887dc7b8290493c040635edf7334e9ae3be584e31a89177080836f493"     # P0 (11-14): the BZ-adopted preregistration after revision 2; None: the decision code does not run
S4_ARM = "configc"                # S4: on the pairs the method fell back to configuration C, the method minus this arm (noise reference)
S4_STATUS = "hx_fallback_configC"
S2_STATUS = "hx_certified"
PQ_BASELINES = ("h", "H_F", "H2", "V_F", "V2", "G_plus", "X_lex", "B_d")      # experiments/pq/pq_stats.JUDGED_BASELINES (plan.md 3.2)
PQ_R_MIN = .05                    # D88 (a)
PQ_MIN_PAIRS = 45                 # D125 (a): |𝒫| < 45 -> F4 only described
DESCRIBED_FAMILIES = ("descriptive", "tier_90s")

SUP, EQ, INF, NONE = "优于", "相当", "劣于", "无法下结论"
SUP_ITT, INF_ITT = "优于（仅意向处理；完整对不支持）", "劣于（仅意向处理；完整对不支持）"
SUP_GUARD = "无法下结论（护栏）"
NOT_JUDGED = "不判（导出约定可疑）"
NOT_JUDGED_MISSING = "不判（缺评测记录）"
HAND_OVER = "交 BZ"
DESCRIBED_FEW = "只描述（n′ < 45）"
F2_DESCRIBED = "只描述（seg4 ≠ G(seg35)）"
F4_METHOD, F4_EVAL, F4_NOT = "F4 只描述，不判（方法侧）", "F4 不判（评测侧）", "F4 不判"
F4_DESCRIBED = "F4 只描述（n′ 或 𝒫 的对数 < 45）"
F4_PASS, F4_FAIL = "留出集上确认：PQ 承重（误差定位）", "留出集上没有确认"
LOWER_BETTER = ("hd95", "sdlogj_official", "sdlogj_brain", "pct_J_lt_half_brain", "frac_abs_logj_gt_log2_brain")
DESCRIBED = ("dice_official",)+LOWER_BETTER
TIMING_KEYS = ("seconds", "started_utc", "ended_utc", "utc", "checked_utc")


class DecisionRefused(Exception):
    """The decision code does not run (pins missing or the records do not fit)."""


# ---- resampling and p values --------------------------------------------------------------------------------------------------------------------------
def draw_idx(seed=SEED, b=B, n=N_PAIRS):
    """idx_b = rng.integers(0, n, n), b = 1..B, one call each in sequence (preregistration 4.3)."""
    rng = np.random.default_rng(seed)
    idx = np.empty((b, n), dtype=np.int64)
    for k in range(b):
        idx[k] = rng.integers(0, n, n)
    return idx


def idx_sha256(idx):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()


def boot_means(delta, idx, mask=None):
    """Δ̄*_b = mean(delta[idx_b]); with a mask only the positions where it is true count (repeats counted); a draw with none is NaN (void)."""
    delta = np.asarray(delta, dtype=np.float64)
    if mask is None:
        return delta[idx].mean(axis=1)
    keep = np.asarray(mask, dtype=bool)[idx]
    counts = keep.sum(axis=1)
    sums = np.where(keep, delta[idx], 0.).sum(axis=1)
    out = np.full(idx.shape[0], np.nan)
    valid = counts > 0
    out[valid] = sums[valid]/counts[valid]
    return out


def pvalues(means, rope=ROPE, eq_bound=EQ_BOUND):
    """The three one-sided p values: sup and inf against ±rope, eq (TOST) against ±eq_bound; non-finite draws are left out and counted (void)."""
    v = np.asarray(means, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    lo, hi = (1+np.count_nonzero(v <= -eq_bound))/(n+1), (1+np.count_nonzero(v >= eq_bound))/(n+1)
    return dict(sup=(1+np.count_nonzero(v <= rope))/(n+1), eq=max(lo, hi), eq_lo=lo, eq_hi=hi, inf=(1+np.count_nonzero(v >= -rope))/(n+1), draws=int(n),
                void=int(np.asarray(means).size-n), rope=rope, eq_bound=eq_bound)


def interval(means):
    v = np.asarray(means, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [float(lo), float(hi)]


def holm(p, order, alpha=ALPHA):
    """{arm: dict(step, raw, adjusted, threshold, rejected)}; ties ordered by `order`."""
    m = len(order)
    ranked = sorted(order, key=lambda k: (p[k], order.index(k)))
    running, stopped, out = 0., False, {}
    for j, k in enumerate(ranked, 1):
        running = max(running, min(1., (m-j+1)*p[k]))
        rejected = not stopped and running <= alpha
        stopped = stopped or not rejected
        out[k] = dict(step=j, raw=p[k], adjusted=running, threshold=alpha/(m-j+1), rejected=rejected)
    return out


def wilcoxon_p(delta):
    from scipy import stats
    d = np.asarray(delta, dtype=np.float64)
    if not np.any(d != 0):
        return None
    try:
        return float(stats.wilcoxon(d).pvalue)
    except ValueError:
        return None


# ---- the records ------------------------------------------------------------------------------------------------------------------------------------
def _without_timing(value):
    if isinstance(value, dict):
        return {k: _without_timing(v) for k, v in value.items() if k not in TIMING_KEYS}
    if isinstance(value, list):
        return [_without_timing(v) for v in value]
    return value


def _same(a, b):
    return json.dumps(_without_timing(a), sort_keys=True) == json.dumps(_without_timing(b), sort_keys=True)


def collect(out_dirs, rows=None):
    """{pair: merged record} over the runs' output directories in lock order (the first run, then the reevaluations).  Per arm (and the identity row, the
    PQ evaluation) the latest record that holds it; a partial record is never overwritten (revision 1b item 2).  The same arm (or identity row) in two
    records with different numbers (timings left out) is kept in `_conflicts` (review m11: the evaluation is a deterministic CPU computation).  The PQ
    evaluation: the latest complete one, else the latest.  Two complete records of one pair refuse (a pair is reevaluated at most once, and only when not
    complete).  With `rows`, every record's pair must be a row of the list at its index (review I8)."""
    names = None if rows is None else {r[3]: r[0] for r in rows}
    found = {}
    for out in out_dirs:
        for path in sorted((Path(out)/"pairs").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            pair = record.get("pair")
            if names is not None and (pair not in names or record.get("index") != names[pair]):
                raise DecisionRefused(f"{path}: pair {pair!r} at index {record.get('index')!r} is not a row of the list at that index")
            merged = found.setdefault(pair, dict(pair=pair, index=record.get("index"), arms={}, complete=False, _sources=[], _conflicts=[]))
            if record.get("complete") is True and merged["complete"]:
                raise DecisionRefused(f"{pair}: two complete records ({merged['_sources']}, {path})")
            merged["_sources"].append(str(path))
            merged["complete"] = merged["complete"] or record.get("complete") is True
            for key in ("seg4_equals_G", "truth", "grid"):
                if record.get(key) is not None:
                    merged[key] = record[key]
            if record.get("identity") is not None:
                if merged.get("identity") is not None and not _same(merged["identity"], record["identity"]):
                    merged["_conflicts"].append(dict(item="identity", sources=list(merged["_sources"])))
                merged["identity"] = record["identity"]
            for arm, entry in (record.get("arms") or {}).items():
                if arm in merged["arms"] and not _same(merged["arms"][arm], entry):
                    merged["_conflicts"].append(dict(item=f"arm {arm}", sources=list(merged["_sources"])))
                merged["arms"][arm] = entry
            pq = record.get("pq")
            if pq is not None and not ((merged.get("pq") or {}).get("complete") is True and pq.get("complete") is not True):
                merged["pq"] = pq
            ablation = record.get("ablation")            # the repair ablation (described only): the latest complete one, else the latest
            if ablation is not None and not ((merged.get("ablation") or {}).get("complete") is True and ablation.get("complete") is not True):
                merged["ablation"] = ablation
    return found


# ---- the repair ablation (prereg revision 2 item 4; contract revision 9 draft C24.2; A5b item 3): described only ------------------------------------
ABLATION_KEYS = ("seg35", "seg4", "sdlogj_official")       # bayesreg.eval.holdout.ABLATION_KEYS (a test compares)
ABLATION_TABLES = (("main", ("hx_certified",)), ("fallback", ("hx_fallback_configC", "hx_fallback_identity")))     # main = hx_certified only (BZ adopted the default, 11-10)


def ablation_section(records, rows, idx):
    """The repair ablation (ablation_table), described only.  A5b review I1 (the lead's ruling, 11-10 ~9:22 PM CST): any error in it is written into this
    section only (error, no tables); F1-F4 and the rest of the decision file are computed apart and are not touched."""
    try:
        return ablation_table(records, rows, idx)
    except Exception as exc:                                          # noqa: BLE001 - described only: never in the decision's way
        return dict(role="described only (prereg revision 2 item 4; C24.2)", error=f"{type(exc).__name__}: {exc}", tables=None,
                    note="the ablation section could not be computed from the evaluation records; no decision depends on it (A5b review I1)")


def ablation_table(records, rows, idx):
    """Prereg revision 2 item 4 "汇总在哪里算" (review m16): the repair ablation from the evaluation entry's per-pair numbers (record['ablation']: before = the
    prepared pre-repair displacement, after = the method arm's output, difference = after - before, for ABLATION_KEYS).  Two tables by the method's line status
    (main: hx_certified, BZ adopted the default 11-10; fallback: the fallbacks, where "after" is configuration C or the identity, not the repair); the pairs
    without the ablation listed with the entry's reason.  Per table and key: n, the two sides' means, the difference's mean, median, per-pair values and the
    unadjusted 95% interval (numpy.percentile linear 2.5, 97.5) of the means over the decision's idx restricted to the table's positions (repeats counted; a draw
    without any is void, not redrawn, counted).  Described only: it is computed beside F1-F4 and changes none of them (C24.2-4)."""
    per_pair, excluded = [], {}
    for i, row in enumerate(rows):
        section = (records.get(row[3]) or {}).get("ablation")
        if not isinstance(section, dict):
            excluded[row[3]] = "no ablation section in the evaluation record"
        elif section.get("in_ablation") is not True:
            excluded[row[3]] = section.get("reason")
        elif section.get("complete") is not True:
            excluded[row[3]] = "the ablation of the pair did not complete in the evaluation"
        else:
            per_pair.append(dict(position=i, pair=row[3], line_status=section.get("line_status"), before=section.get("before") or {}, after=section.get("after") or {},
                                 difference=section.get("difference") or {}))
    tables, listed = {}, set()
    for name, statuses in ABLATION_TABLES:
        chosen = [r for r in per_pair if r["line_status"] in statuses]
        listed |= {r["pair"] for r in chosen}
        endpoints = {}
        for key in ABLATION_KEYS:
            diff = np.full(len(rows), np.nan)
            for r in chosen:
                value = r["difference"].get(key)
                diff[r["position"]] = np.nan if value is None else float(value)
            mask = np.isfinite(diff)
            means = boot_means(np.where(mask, diff, 0.), idx, mask)
            side = {s: [r[s].get(key) for r in chosen if r[s].get(key) is not None] for s in ("before", "after")}
            endpoints[key] = dict(n=int(mask.sum()), mean_before=float(np.mean(side["before"])) if side["before"] else None,
                                  mean_after=float(np.mean(side["after"])) if side["after"] else None,
                                  mean=float(diff[mask].mean()) if mask.any() else None, median=float(np.median(diff[mask])) if mask.any() else None,
                                  interval=interval(means), void_draws=int(np.count_nonzero(~np.isfinite(means))),
                                  per_pair=[dict(pair=r["pair"], before=r["before"].get(key), after=r["after"].get(key), difference=r["difference"].get(key)) for r in chosen])
        tables[name] = dict(line_statuses=list(statuses), n_pairs=len(chosen), pairs=[r["pair"] for r in chosen], endpoints=endpoints)
    return dict(role="described only (prereg revision 2 item 4; C24.2): not certified, not an arm, in no family; no Holm, no ITT, no S1-S5, no guard; no decision changed",
                written="after - before (after: the method arm's output; before: the pre-repair field, boundary_taper(field_0600))", tables=tables,
                other_statuses={r["pair"]: r["line_status"] for r in per_pair if r["pair"] not in listed}, excluded=excluded,
                scope="main: hx_certified only (BZ adopted the default, decision_log 2026-11-10); the fallbacks apart", idx="the decision's idx restricted to each table's positions")


def has_identity(record):
    return record is not None and isinstance(record.get("identity"), dict) and isinstance(record["identity"].get("metrics"), dict)


def has_arm(record, arm):
    entry = None if record is None else (record.get("arms") or {}).get(arm)
    if entry is None:
        return False
    return entry.get("outcome") != "output" or isinstance(entry.get("metrics"), dict)


def method_name(table):
    return next(a["name"] for a in table["arms"] if a["family"] == "method")


def matrices(records, table, rows, keys):
    """{key: {arm: values per pair}} with ITT imputation (a cell without output takes the pair's identity row), and per arm the masks imputed, export-
    suspicious and missing (no evaluation record of the cell: NaN), whether it was stopped (P19), and its terminal statuses."""
    names = [a["name"] for a in table["arms"]]
    values = {key: {a: np.full(len(rows), np.nan) for a in names} for key in keys}
    imputed = {a: np.zeros(len(rows), dtype=bool) for a in names}
    suspicious = {a: np.zeros(len(rows), dtype=bool) for a in names}
    missing = {a: np.zeros(len(rows), dtype=bool) for a in names}
    stopped = {a: False for a in names}
    status = {a: [None]*len(rows) for a in names}
    for i, row in enumerate(rows):
        record = records.get(row[3])
        for a in names:
            if not (has_identity(record) and has_arm(record, a)):
                missing[a][i] = True
                continue
            entry = record["arms"][a]
            status[a][i] = entry.get("terminal_status")
            if entry.get("outcome") == "output":
                source = entry["metrics"]
            else:
                source, imputed[a][i] = record["identity"]["metrics"], True
                stopped[a] = stopped[a] or entry.get("no_output_class") == "arm_stopped"
            suspicious[a][i] = entry.get("export_suspicious") is True
            for key in keys:
                value = source.get(key)
                values[key][a][i] = np.nan if value is None else float(value)
    return values, imputed, suspicious, stopped, status, missing


def official_rows(records, table, rows):
    """{arm: [official row per pair] (ITT: the identity row), or None when a cell of the arm has no record}."""
    names = [a["name"] for a in table["arms"]]
    out = {a: [] for a in names}
    for row in rows:
        record = records.get(row[3])
        for a in names:
            if out[a] is None:
                continue
            if not (has_identity(record) and has_arm(record, a)):
                out[a] = None
                continue
            entry = record["arms"][a]
            out[a].append(_nulls_as_nan(entry["official"] if entry.get("outcome") == "output" else record["identity"]["official"]))
    return out


def _nulls_as_nan(official):
    """The entry writes a non-finite number as JSON null (bayesreg.solve.export.dumps); metrics.aggregate takes NaN (A5b's chain test: a label absent in both
    images gives a NaN Dice, read back as None, and np.isnan(None) raised).  Lists and scalars of the official row with None made NaN; nothing else changes."""
    if not isinstance(official, dict):
        return official
    out = {}
    for key, value in official.items():
        if isinstance(value, list):
            out[key] = [np.nan if v is None else v for v in value]
        else:
            out[key] = np.nan if value is None else value
    return out


def dsc30_draws(rows_of_arm, idx):
    """DSC30 (scripts/evaluate_l2r_style.aggregate's dsc30_ranking) on each resample's pairs: entries (initial, final) of the drawn pairs in draw order, NaN
    entries left out, a stable sort by the initial Dice, the mean final Dice of the lowest round(.3 n) (at least 1)."""
    per_pair = []
    for r in rows_of_arm:
        a = np.array([[x0, x1] for x0, x1 in zip(r["initial_dice_labels"], r["dice_labels"])], dtype=np.float64).reshape(-1, 2)
        per_pair.append(a[~(np.isnan(a[:, 0]) | np.isnan(a[:, 1]))])
    out = np.empty(idx.shape[0])
    for b in range(idx.shape[0]):
        entries = np.concatenate([per_pair[i] for i in idx[b]])
        order = np.argsort(entries[:, 0], kind="stable")
        k = max(1, int(round(.3*len(entries))))
        out[b] = float(np.mean(entries[order[:k], 1]))
    return out


def dsc30_point(rows_of_arm):
    from bayesreg.eval.metrics import aggregate
    return aggregate(rows_of_arm)["dsc30_ranking"]


# ---- one family (F1 or F2) --------------------------------------------------------------------------------------------------------------------------
def sdlogj_guard_row(sdlogj, imputed, missing, method, k, pos, idx, bound):
    """The SDlogJ guard of k: the official SDlogJ difference (method - k) on the complete pairs (both arms with their own output), the same idx restricted
    to them; holds when the upper end of the 95% percentile interval is < bound.  No complete pair (every draw void): it does not hold."""
    own = ~imputed[method][pos] & ~imputed[k][pos] & ~missing[method][pos] & ~missing[k][pos]
    diff = sdlogj[method][pos]-sdlogj[k][pos]
    use = own & np.isfinite(diff)
    means = boot_means(np.where(use, diff, 0.), idx, use)
    iv = interval(means)
    upper = None if iv is None else iv[1]
    return dict(rule="official SDlogJ, method - arm, complete pairs only; holds iff the 95% interval's upper end < bound", bound=bound, n=int(use.sum()),
                mean=float(diff[use].mean()) if use.any() else None, interval=iv, upper=upper, void_draws=int(np.isnan(means).sum()),
                holds=bool(upper is not None and upper < bound))


def judge_family(endpoint, values, imputed, suspicious, stopped, table, idx, *, sdlogj, guard, positions=None, alpha=ALPHA, missing=None,
                 suspicious_counts=None):
    """Verdicts of the main family on `endpoint`; `positions` restricts the pairs (idx is then drawn on them); `missing`: per arm, cells without an
    evaluation record (missing-pair rule (丙): such an arm is not judged, p = 1; the method missing a pair leaves every arm not judged); `sdlogj`: per arm
    the official SDlogJ of the same pairs (the guard); `suspicious_counts`: per arm the export-suspicious pairs over the whole list (P19, review m4;
    None: counted on these positions)."""
    if guard is None:
        raise DecisionRefused("SDLOGJ_GUARD is None: the bound of the SDlogJ guard is not pinned (preregistration revision 2)")
    method = method_name(table)
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
            raise DecisionRefused(f"{endpoint} {k}: {int((~np.isfinite(delta)).sum())} non-finite differences in the main family (review m5): hand over to BZ")
        means = boot_means(delta, idx)
        n_suspicious = int(suspicious[k][pos].sum()) if suspicious_counts is None else int(suspicious_counts[k])
        p = pvalues(means)
        not_judged = stopped[k] or n_suspicious >= 3 or lacking
        if not_judged:
            p = dict(p, sup=1., eq=1., eq_lo=1., eq_hi=1., inf=1.)
        complete = ~imputed[method][pos] & ~imputed[k][pos]
        s1_means = boot_means(delta, idx, complete)
        rows[k] = dict(mean=float(delta.mean()), median=float(np.median(delta)), wins=int((delta > 0).sum()), n=int(delta.size), wilcoxon_p=wilcoxon_p(delta),
                       interval=interval(means), p=p, not_judged=not_judged, n_suspicious=n_suspicious, stopped=stopped[k], lacking_records=lacking,
                       imputed_method=int(imputed[method][pos].sum()), imputed_arm=int(imputed[k][pos].sum()),
                       s1=dict(n=int(complete.sum()), mean=float(delta[complete].mean()) if complete.any() else None, p=pvalues(s1_means), interval=interval(s1_means)),
                       s5=None, sdlogj_guard=sdlogj_guard_row(sdlogj, imputed, missing, method, k, pos, idx, guard))
        keep = ~suspicious[k][pos]
        if 0 < n_suspicious < 3 and not keep.all():
            s5 = boot_means(delta, idx, keep)
            rows[k]["s5"] = dict(n=int(keep.sum()), mean=float(delta[keep].mean()), interval=interval(s5))
    steps = {cls: holm({k: rows[k]["p"][cls] for k in order}, order, alpha) for cls in ("sup", "eq", "inf")}
    for k in order:
        row = rows[k]
        row["holm"] = {cls: steps[cls][k] for cls in steps}
        row["rejected_classes"] = [cls for cls in ("sup", "inf", "eq") if steps[cls][k]["rejected"]]
        if row["not_judged"]:
            row["verdict"] = NOT_JUDGED_MISSING if row["lacking_records"] and not (row["stopped"] or row["n_suspicious"] >= 3) else NOT_JUDGED
            continue
        rejected = row["rejected_classes"]
        verdict = {"sup": SUP, "inf": INF, "eq": EQ}[rejected[0]] if rejected else NONE        # 优于 → 劣于 → 相当 → 无法下结论
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
    return dict(endpoint=endpoint, m=len(order), alpha=alpha, rope=ROPE, eq_bound=EQ_BOUND, sdlogj_guard=guard, arms=rows, order=order,
                holm_order={cls: sorted(order, key=lambda k: steps[cls][k]["step"]) for cls in steps})


def describe(values, imputed, table, idx, key, sign, families=None):
    """F3 / 4.6: mean difference and unadjusted interval for every non-method arm (of `families`, all when None); sign +1: method - arm; -1: arm - method."""
    method = method_name(table)
    out = {}
    for arm in table["arms"]:
        k = arm["name"]
        if k == method or (families is not None and arm["family"] not in families):
            continue
        delta = sign*(values[key][method]-values[key][k])
        finite = np.isfinite(delta)
        means = boot_means(np.where(finite, delta, 0.), idx, finite)
        out[k] = dict(family=arm["family"], mean=float(delta[finite].mean()) if finite.any() else None, interval=interval(means),
                      written="method - arm" if sign > 0 else "arm - method (smaller is better)", n_nonfinite=int((~finite).sum()),
                      imputed_arm=int(imputed[k].sum()), imputed_method=int(imputed[method].sum()))
    return out


def described_only(family, reading):
    """A family that is only described: its numbers stay, its verdicts, Holm steps and guard outcomes go (review m13)."""
    family = dict(family, reading=reading)
    family.pop("holm_order", None)
    for row in family.get("arms", {}).values():
        row["verdict"] = None
        for key in ("holm", "rejected_classes", "itt_guard", "verdict_before_guard"):
            row.pop(key, None)
        if isinstance(row.get("sdlogj_guard"), dict):
            row["sdlogj_guard"] = {k: v for k, v in row["sdlogj_guard"].items() if k != "holds"}
    return family


# ---- F4 (C22.H7, contract revision 7, adopted 2026-11-09) ------------------------------------------------------------------------------------------
def pq_rows(records, table, rows):
    """One row per position of 𝒩 (rows, in tsv order): the pair, whether the method arm has an evaluation record and an output, its terminal status and the
    merged PQ evaluation."""
    method = method_name(table)
    out = []
    for i, row in enumerate(rows):
        record = records.get(row[3]) or {}
        entry = (record.get("arms") or {}).get(method)
        out.append(dict(position=i, pair=row[3], method_record=entry is not None, method_output=(entry or {}).get("outcome") == "output",
                        status=(entry or {}).get("terminal_status"), pq=record.get("pq")))
    return out


def _ause(pq, score):
    value = (((pq.get("acceptance") or {}).get("scores") or {}).get("omega_e") or {}).get(score, {}).get("ause")
    return None if value is None or not np.isfinite(float(value)) else float(value)


def f4_class_b(rows_pq, P, idx, baselines, r_min):
    """Class B on 𝒫 (C22.H7-2) through bayesreg.eval.holdout.pq_b_class (pq_stats.b_class's arithmetic, the one the replay gate checks): the AUSE table over
    𝒫's positions, the interval the hold-out resampling -- idx (drawn on 𝒩's n' positions) restricted to 𝒫, repeats counted, a draw with none void and not
    redrawn, LB the 2.5 percentile (numpy linear) of the valid draws (D116 (a))."""
    from bayesreg.eval.holdout import pq_b_class
    n = len(rows_pq)
    mask = np.zeros(n, dtype=bool)
    mask[P] = True

    def holdout_interval(delta):
        full = np.zeros(n)
        full[P] = np.asarray(delta, dtype=np.float64)
        means = boot_means(full, idx, mask)
        valid = means[np.isfinite(means)]
        return dict(lo=float(np.percentile(valid, 2.5)) if valid.size else None, hi=float(np.percentile(valid, 97.5)) if valid.size else None,
                    valid_draws=int(valid.size), void_draws=int(means.size-valid.size), rule="idx on 𝒩 restricted to 𝒫; void draws not redrawn (C22.H7-2)")
    table = {k: [_ause(rows_pq[i]["pq"], k) for i in P] for k in ("q",)+tuple(baselines)}
    return pq_b_class(table, interval=holdout_interval, r_min=r_min, baselines=baselines)


def decide_f4(rows_pq, idx, *, n_prime, baselines=PQ_BASELINES, r_min=PQ_R_MIN, min_pairs=PQ_MIN_PAIRS):
    """The table of C22.H7-3 on 𝒩 = rows_pq (one row per position, tsv order; idx drawn on these n' positions, the F1-F3 idx).  A problem inside F4 (records
    of different method sides, an AUSE missing on 𝒫) reads 不判（评测侧） with its reason and never stops F1-F3 (review2 m5)."""
    base = dict(contract="C22.H7 (contract revision 7d, adopted by BZ 2026-11-09)", idx_sha256=idx_sha256(idx), n_prime=n_prime,
                rule="LB_k = percentile(valid Δ̄*_k, 2.5) > 0 and R_k >= .05 for every baseline (D116 (a), D88 (a)); |𝒫| and n' >= 45 (D125 (a))")
    sides = sorted({(r["pq"] or {}).get("method_side") for r in rows_pq if r["pq"] is not None} - {None})
    if len(sides) > 1:
        return dict(base, reading=F4_EVAL, note=f"the PQ records name different method sides {sides}")
    if sides == ["bz_ii"]:
        return dict(base, reading=F4_METHOD, note="a method-side control of PQ failed and BZ chose (ii) (C22.H4-2): F4 is only described; F1-F3 as usual")
    excluded, P = [], []
    for r in rows_pq:                                # revision 7d: first the pairs that leave 𝒫, each with its reason, then Ev1-Ev3 on 𝒫 (review2 m10)
        pq = r["pq"]
        if not r["method_record"]:
            reason = "方法臂缺评测记录（缺对规则丙）"
        elif not r["method_output"]:
            reason = f"方法没有输出（{r['status']}）"
        elif pq is None:
            reason = "缺 PQ 评测记录（这一对的评测没有走到 PQ）"
        elif pq.get("state") != "present":
            reason = f"缺 PQ 输出：{pq.get('state')}（{pq.get('reason')}）"
        elif pq.get("complete") is not True:
            reason = "PQ 评测记录补评之后仍不完整：先从 𝒫 去掉，再在 𝒫 上核 Ev3（C22.H7-1 修订 7d）"
        else:
            P.append(r["position"])
            continue
        excluded.append(dict(pair=r["pair"], position=r["position"], reason=reason))
    out = dict(base, P=[dict(position=i, pair=rows_pq[i]["pair"]) for i in P], n_P=len(P), excluded=excluded)
    ev = dict(Ev1=[], Ev2=[], input_lock=[], Ev3=[])
    for i in P:
        pq = rows_pq[i]["pq"]
        if (pq.get("ev1") or {}).get("passed") is not True:
            ev["Ev1"].append(rows_pq[i]["pair"])
        if (pq.get("ev2") or {}).get("passed") is not True:
            ev["Ev2"].append(rows_pq[i]["pair"])
        elif (pq.get("input_lock") or {}).get("passed") is not True:          # C22.H4-2 Hm2 after the step-8 input lock (the lead's ruling (10), 11-09)
            ev["input_lock"].append(rows_pq[i]["pair"])
        elif not isinstance(pq.get("acceptance"), dict):
            ev["Ev3"].append(rows_pq[i]["pair"])
    out["eval_side"] = ev
    if any(ev.values()):
        return dict(out, reading=F4_EVAL)
    if not P:
        return dict(out, reading=F4_NOT, note="𝒫 is empty")
    zero = [rows_pq[i]["pair"] for i in P if ((rows_pq[i]["pq"]["acceptance"].get("n_errors") or {}).get("omega_e")) == 0]
    if zero:
        return dict(out, reading=F4_NOT, note=f"Ω_e without error voxels: {zero}")
    stats = f4_class_b(rows_pq, P, idx, baselines, r_min)
    if not stats["judged"]:
        return dict(out, reading=F4_EVAL, note=f"class B not computable on 𝒫: {stats['reason']}")
    certified = [i for i in P if rows_pq[i]["status"] == S2_STATUS]
    subset = None
    if certified:                    # described only (C22.H7-5)
        sub = f4_class_b(rows_pq, certified, idx, baselines, r_min)
        subset = {k: dict(n=len(certified), R=v["R"], LB=v["LB"], mean_delta=float(np.mean(v["delta"]))) for k, v in sub.get("baselines", {}).items()}
    rows = {}
    for k, v in stats["baselines"].items():
        rows[k] = dict(mean_delta=float(np.mean(v["delta"])), LB=v["LB"], R=v["R"], LB_gap=v["LB"], R_gap=None if v["R"] is None else v["R"]-r_min,
                       void_draws=v["interval"]["void_draws"], valid_draws=v["interval"]["valid_draws"], interval=[v["interval"]["lo"], v["interval"]["hi"]],
                       per_pair=[dict(position=int(i), pair=rows_pq[i]["pair"], delta=float(dlt)) for i, dlt in zip(P, v["delta"])])
    out.update(baselines=rows, hx_certified_subset=subset, class_b_rule=stats["rule"])
    if n_prime < min_pairs or len(P) < min_pairs:
        return dict(out, reading=F4_DESCRIBED)
    for k in baselines:
        rows[k]["passed"] = stats["baselines"][k]["passed"]
    failed = list(stats["failed"])
    note = None
    x = stats["baselines"].get("X_lex")
    if failed == ["X_lex"] and x["LB"] is not None and x["LB"] > 0 and not (x["R"] is not None and x["R"] >= r_min):
        note = "相对 X_lex 的 5% 最小效应没有确认（不写成 PQ 不如 X_lex）"
    return dict(out, reading=F4_PASS if not failed else F4_FAIL, failed=failed, note=note)


# ---- the whole decision ------------------------------------------------------------------------------------------------------------------------------
def seg4_failures(records, rows):
    bad = []
    for i, row in enumerate(rows):
        check = (records.get(row[3]) or {}).get("seg4_equals_G") or {}
        for side, subject in (("fixed", row[1]), ("moving", row[2])):
            if (check.get(side) or {}).get("equal") is not True:
                bad.append(dict(pair=row[3], index=i, subject=subject))
    return bad


def complete_mask(records, table, rows):
    names = [a["name"] for a in table["arms"]]
    return np.array([has_identity(records.get(r[3])) and all(has_arm(records.get(r[3]), a) for a in names) for r in rows], dtype=bool)


def _certificate_counts(certificates):
    lows = [c.get("min_jd_lower_bound") for c in certificates if isinstance(c.get("min_jd_lower_bound"), (int, float))]
    return dict(n=len(certificates), passed=sum(c.get("passed") is True for c in certificates), min_jd_lower_bound=min(lows) if lows else None,
                n_uncertain=sum(int(c.get("n_uncertain") or 0) for c in certificates), n_infeasible=sum(int(c.get("n_infeasible") or 0) for c in certificates))


def certificate_summary(records, table, rows):
    """Described only (preregistration §3 "折叠" and "方法的证书统计"; D112 (a), review I3): per arm the BL1 Jacobian-only certificate every arm shares, over
    its cells with output; the method's certificate statistics from the entry's decision-configuration certificate of output/final.npz only (never the
    cell's own decision certificate).  The two tiles' undecided counts are not set side by side (C07-2)."""
    out = {}
    for arm in table["arms"]:
        entries = [((records.get(r[3]) or {}).get("arms") or {}).get(arm["name"]) or {} for r in rows]
        entries = [e for e in entries if e.get("outcome") == "output"]
        out[arm["name"]] = dict(jacobian_only_bl1=_certificate_counts([e.get("certificate") or {} for e in entries]))
        if arm["family"] == "method":
            out[arm["name"]]["decision_configuration"] = dict(
                _certificate_counts([(e.get("decision_certificate") or {}).get("certificate") or {} for e in entries]),
                role="the method's certificate statistics (D112 (a): the entry's certificate of output/final.npz, configuration C, tile 65536)")
    return out


def decide(records, table, rows, *, seg4_rule=SEG4_G_RULE, missing_rule=MISSING_PAIR_RULE, sdlogj_guard=None, idx=None):
    """The decision document.  records: {pair: merged record} (collect); rows: holdout.load_pairs()'s rows; idx: the resampling of the 50 pairs (default
    draw_idx()); sdlogj_guard: the bound (default SDLOGJ_GUARD; None refuses)."""
    guard = SDLOGJ_GUARD if sdlogj_guard is None else sdlogj_guard
    if guard is None:
        raise DecisionRefused("SDLOGJ_GUARD is None: the bound of the SDlogJ guard is not pinned (preregistration revision 2); the decision does not run")
    if seg4_rule not in ("a", "b"):
        raise DecisionRefused("seg4_rule: 'a' or 'b'")
    if missing_rule not in ("jia", "yi", "bing"):
        raise DecisionRefused("missing_rule: 'jia' (甲), 'yi' (乙) or 'bing' (丙)")
    full_idx = draw_idx(n=len(rows)) if idx is None else idx
    complete = complete_mask(records, table, rows)
    missing_pairs = [rows[i][3] for i in np.flatnonzero(~complete)]
    conflicts = {name: r["_conflicts"] for name, r in records.items() if r.get("_conflicts")}
    head = dict(seed=SEED, B=int(full_idx.shape[0]), seg4_rule=seg4_rule, missing_rule=missing_rule, missing_pairs=missing_pairs, sdlogj_guard=guard,
                merge_conflicts=conflicts)
    use, idx, few = list(range(len(rows))), full_idx, False
    if missing_pairs:
        if missing_rule == "jia":
            return dict(head, reading=HAND_OVER, reason=f"{len(missing_pairs)} pairs have no complete evaluation record after the reevaluation (D109, rule 甲): "
                                                        "no confirmatory F1-F4", F1=None, F2=None, F3=None, F4=None)
        if missing_rule == "yi":
            use = [i for i in range(len(rows)) if complete[i]]
            few = len(use) < MISSING_MIN_PAIRS
            idx = draw_idx(n=len(use), b=full_idx.shape[0])
    sub_rows = [rows[i] for i in use]
    keys = ("seg35", "seg4")+DESCRIBED
    values, imputed, suspicious, stopped, status, missing = matrices(records, table, sub_rows, keys)
    _, _, suspicious_all, _, _, _ = matrices(records, table, rows, ())
    counts = {a: int(v.sum()) for a, v in suspicious_all.items()}
    method = method_name(table)
    common = dict(sdlogj=values["sdlogj_official"], guard=guard, missing=missing, suspicious_counts=counts)
    f1 = judge_family("seg35", values["seg35"], imputed, suspicious, stopped, table, idx, **common)
    bad = seg4_failures(records, sub_rows)
    if not bad:
        f2 = judge_family("seg4", values["seg4"], imputed, suspicious, stopped, table, idx, **common)
    elif seg4_rule == "a":
        f2 = dict(described_only(judge_family("seg4", values["seg4"], imputed, suspicious, stopped, table, idx, **common), F2_DESCRIBED),
                  seg4_failures=bad, rule="a")
    else:
        drop = {b["index"] for b in bad}
        keep = [i for i in range(len(sub_rows)) if i not in drop]
        if len(keep) < SEG4_G_MIN_PAIRS:
            f2 = dict(reading=F2_DESCRIBED, seg4_failures=bad, rule="b", n_prime=len(keep), note=f"n' < {SEG4_G_MIN_PAIRS}: F2 described as a whole")
        else:
            idx_b = draw_idx(n=len(keep), b=full_idx.shape[0])
            f2 = dict(judge_family("seg4", values["seg4"], imputed, suspicious, stopped, table, idx_b, positions=keep, **common), seg4_failures=bad,
                      rule="b", n_prime=len(keep), idx_sha256=idx_sha256(idx_b))
    if few:
        f1 = described_only(f1, DESCRIBED_FEW)
        f2 = described_only(f2, DESCRIBED_FEW)
    f3 = dict(rule="unadjusted 95% percentile intervals, not judged",
              endpoints={key: describe(values, imputed, table, idx, key, -1 if key in LOWER_BETTER else 1) for key in DESCRIBED},
              described_arms={key: describe(values, imputed, table, idx, key, 1, families=DESCRIBED_FAMILIES) for key in ("seg35", "seg4")})
    rows_off = official_rows(records, table, sub_rows)
    f3["dsc30"] = {}
    if rows_off[method] is not None:
        method_draws = dsc30_draws(rows_off[method], idx)
        for a in table["arms"]:
            if a["name"] != method and rows_off[a["name"]] is not None:
                f3["dsc30"][a["name"]] = dict(family=a["family"], method=dsc30_point(rows_off[method]), arm=dsc30_point(rows_off[a["name"]]),
                                              interval=interval(method_draws-dsc30_draws(rows_off[a["name"]], idx)), written="method - arm")
    sd = values["sdlogj_official"]
    f3["sdlogj_official_mean_le_010"] = dict(
        itt={a["name"]: bool(np.nanmean(sd[a["name"]]) <= .10) for a in table["arms"] if np.isfinite(sd[a["name"]]).any()},
        outputs_only={a["name"]: bool(np.nanmean(sd[a["name"]][~imputed[a["name"]]]) <= .10) for a in table["arms"]
                      if np.isfinite(sd[a["name"]][~imputed[a["name"]]]).any()})
    certified = np.array([s == S2_STATUS for s in status[method]])
    sensitivity = dict(S2={k: dict(n=int(certified.sum()), interval=interval(boot_means(np.nan_to_num(values["seg35"][method]-values["seg35"][k]), idx, certified)))
                           for k in table["main_family"]})
    fallback = np.array([s == S4_STATUS for s in status[method]])
    if S4_ARM in values["seg35"] and fallback.any():
        delta = np.nan_to_num(values["seg35"][method]-values["seg35"][S4_ARM])
        sensitivity["S4"] = dict(n=int(fallback.sum()), mean=float(delta[fallback].mean()), interval=interval(boot_means(delta, idx, fallback)))
    else:
        sensitivity["S4"] = None
    f4 = decide_f4(pq_rows(records, table, sub_rows), idx, n_prime=len(sub_rows))         # n' < 45 is one of its rows (only described)
    no_output = {}
    for a in table["arms"]:
        classes = set()
        for r in sub_rows:
            entry = ((records.get(r[3]) or {}).get("arms") or {}).get(a["name"]) or {}
            if entry and entry.get("outcome") != "output":
                classes.add(entry.get("no_output_class"))
        no_output[a["name"]] = dict(n=int(imputed[a["name"]].sum()), stopped=stopped[a["name"]], classes=sorted(classes-{None}),
                                    missing_records=int(missing[a["name"]].sum()), suspicious_pairs_all=counts[a["name"]])
    return dict(head, reading="decided" if not few else DESCRIBED_FEW, n_used=len(use), positions=[dict(position=j, pair=r[3], index=r[0]) for j, r in enumerate(sub_rows)],
                idx_sha256=idx_sha256(idx), F1=f1, F2=f2, F3=f3, F4=f4, sensitivity=sensitivity, no_output=no_output,
                certificates=certificate_summary(records, table, sub_rows),
                method_line_status={s: status[method].count(s) for s in sorted(set(s for s in status[method] if s))},
                ablation=ablation_section(records, sub_rows, idx))


def main(argv=None):
    import argparse
    from bayesreg.eval import holdout
    parser = argparse.ArgumentParser(prog="python -m bayesreg.eval.holdout_decide", description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", type=Path, help="optional: must equal the output directories of the master and reevaluation locks, in order")
    parser.add_argument("--out", required=True, type=Path, help="decision JSON (must not exist)")
    args = parser.parse_args(argv)
    if PREREG_SHA256 is None or holdout.PREREG_SHA256 != PREREG_SHA256:
        print("refused: PREREG_SHA256 is not pinned (the BZ-adopted preregistration after revision 2)", file=sys.stderr)
        return 2
    if SDLOGJ_GUARD is None:
        print("refused: SDLOGJ_GUARD is not pinned (the bound of the SDlogJ guard, preregistration revision 2)", file=sys.stderr)
        return 2
    try:
        identity = holdout.rule_one()                     # rule one before anything else is read (contract revision 9 draft C24.3-7; decision_log 2026-11-10 ~2:59 AM CST (7); A5b)
        prereg_raw, prereg = holdout.check_prereg()
        table, table_record = holdout.load_arm_table(prereg_raw)
        rows, _ = holdout.load_pairs()
        lines = holdout.run_lines()
        if not lines:
            raise DecisionRefused(f"{holdout.MASTER_LOCK}: no master lock; nothing was evaluated")
        outs = [line["out_dir"] for _, line in lines]
        for path, line in lines:
            if line.get("arm_table_sha256") != table_record["sha256"] or line.get("prereg_sha256") != prereg["sha256"] or \
                    line.get("manifest_sha256") != lines[0][1].get("manifest_sha256"):
                raise DecisionRefused(f"{path}: another arm table, preregistration or manifest than the master lock's")
        if args.runs is not None and [str(Path(o).resolve()) for o in args.runs] != [str(Path(o).resolve()) for o in outs]:
            raise DecisionRefused(f"--runs {[str(o) for o in args.runs]} are not the locks' output directories {outs} (in lock order)")
        records = collect(outs, rows)
        document = dict(decide(records, table, rows), prereg=prereg, arm_table=table_record, runs=outs, locks=[str(p) for p, _ in lines], rule_one=identity)
    except (holdout.HoldoutRefused, DecisionRefused) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    with open(args.out, "x", encoding="utf-8") as stream:
        stream.write(holdout.dumps(document)+"\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
