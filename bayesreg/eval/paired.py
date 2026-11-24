"""Paired comparison of two methods on common pairs (M2 against VAL10 and the like).

    python -m bayesreg.eval.paired LONG.csv... --a METHOD[:ARM] --b METHOD[:ARM] --out paired.json
                                   [--set SET] [--measures seg35,seg4,...] [--block subjects|pairs]
                                   [--resamples 10000] [--seed 20261017] [--roles native_formal,disp_only|any]
                                   [--include-staging]

Definitions (after the paper-1 freeze's paired_stats.py, seq_vp_3d results/verification/3d_stage_FREEZE_v1/
manifest_20261022_FREEZE/paired_stats.py.copy; pairs named '<fixed>__<moving>'):
  long CSV       columns set, split, method, arm, pair, measure, value, run_id, field_role, run_status,
                 structure_registry; value is a decimal (repr of the float, exact round trip) or one of the strings
                 "inf", "-inf", "nan", which are kept as such (a label warped away gives HD95 = inf; review EV item 3);
                 structure_registry is the registration state of a structure-layer run's segmenter registry
                 (registered, staging_unregistered, not_recorded; n/a for every other run; empty in a file written
                 before the column existed)
  roles          only rows whose field_role is in --roles (default native_formal, disp_only: the native formal
                 state of a run, a baseline's disp.npz) are compared; handover_only (the certified native
                 initializer without a native commit), unknown and rows without a role are excluded and listed
                 (excluded_by_role_a / _b); --roles any compares every row
  staging        rows with structure_registry staging_unregistered (未正式登记: a run on the training chain's staging
                 registry, development_diagnostic_only; review PS1_PS2 item d) are excluded by default and listed
                 (excluded_staging_a / _b); --include-staging admits them (recorded as include_staging)
  non-finite     a common pair with a non-finite value on either side is not in d: it is listed (nonfinite_pairs)
                 and counted (n_nonfinite, n_nonfinite_a, n_nonfinite_b); n counts the finite pairs used
  difference     d_p = a_p - b_p on the pairs present for both (pairs missing on either side are listed)
  mean, median   of d_p (numpy; median of an even count = mean of the two middle values)
  positive       number of pairs with d_p > 0 (also negative, zero); better = pairs where a is better than b
                 when the measure's direction is known (MEASURE_DIRECTION: +1 larger is better, -1 smaller is better)
  wilcoxon       scipy.stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided'): statistic and p (p = 1 when
                 every difference is zero); with subject blocks it treats pairs as independent (flagged).
                 p_holm: Holm-Bonferroni adjustment over the measures of one call (the family = the measures
                 compared together; a measure without p is left out of the family)
  bootstrap      percentile 95% intervals (2.5, 97.5) of the mean and of the median difference, R resamples, generator
                 numpy default_rng([seed, sub]) with sub = first 4 bytes (little endian) of sha256('<set>|<measure>|
                 <a>|<b>'), so adding a comparison never changes another's interval.
                 block = subjects (default): each resample draws the S subjects of the common pairs with replacement
                 (multinomial counts) and weights pair (i, j) by count_i * count_j; resample statistic = weighted mean
                 and weighted median (the average of the lower and upper weighted medians, = numpy's median at unit
                 weights); a resample in which every weight is 0 is drawn again (counted).
                 block = pairs: pairs drawn with replacement.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

DEFAULT_SEED = 20261017
DEFAULT_RESAMPLES = 10_000
REQUIRED_COLUMNS = ("set", "split", "method", "arm", "pair", "measure", "value", "run_id")
LONG_COLUMNS = REQUIRED_COLUMNS+("field_role", "run_status", "structure_registry")
DEFAULT_ROLES = ("native_formal", "disp_only")        # bayesreg.eval.fields.COMPARABLE_ROLES
STAGING_UNREGISTERED = "staging_unregistered"         # bayesreg.eval.fields.STAGING_UNREGISTERED
NONFINITE_TEXT = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}
MEASURE_DIRECTION = {
    "seg35": +1, "seg4": +1, "dice_official": +1, "lpba56": +1, "lpba54": +1, "hd95": -1, "sdlogj_official": -1,
    "sdlogj_brain": -1, "central_det_nonpositive": -1, "central_det_nonpositive_fraction": -1,
    "frac_abs_logj_gt_log2_brain": -1, "pct_J_lt_half_brain": -1, "cell_centre_J_nonpositive": -1,
    **{f"cert_{mode}_{key}": sign for mode in ("full", "jacobian_only")
       for key, sign in (("n_infeasible", -1), ("n_uncertain", -1), ("passed", +1))},
}


def format_value(value):
    """The long-CSV text of a value: integers as integers, floats as their repr (exact round trip), non-finite
    floats as "inf", "-inf" or "nan" (kept, never dropped)."""
    if isinstance(value, (bool, np.bool_, int, np.integer)):
        return str(int(value))
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(value)


def parse_value(text):
    """Inverse of format_value (float)."""
    text = str(text).strip()
    if text in NONFINITE_TEXT:
        return NONFINITE_TEXT[text]
    value = float(text)
    if not np.isfinite(value):
        raise ValueError(f"long CSV value {text!r}: non-finite values are written as 'inf', '-inf' or 'nan'")
    return value


def read_long(paths):
    """Rows of one or more long CSV files (columns LONG_COLUMNS; field_role, run_status and structure_registry may
    be absent in a file written before they existed and are then empty -- an empty field_role is admitted by no
    default role, an empty structure_registry is not staging), value as float."""
    rows = []
    for path in ([paths] if isinstance(paths, (str, Path)) else paths):
        with open(path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = set(REQUIRED_COLUMNS)-set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path}: long CSV without columns {sorted(missing)}")
            for row in reader:
                row = {k: ("" if row.get(k) is None else row[k]) for k in LONG_COLUMNS}
                rows.append(dict(row, value=parse_value(row["value"])))
    return rows


def write_long(path, rows):
    """Exclusive create of a long CSV (columns LONG_COLUMNS, values through format_value)."""
    with open(path, "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(LONG_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict({k: row.get(k, "") for k in LONG_COLUMNS}, value=format_value(row["value"])))


def parse_selector(text):
    method, _, arm = str(text).partition(":")
    if not method:
        raise ValueError(f"selector {text!r}: METHOD[:ARM] required")
    return method, arm


def is_staging(row):
    """True for a row of a run on the training chain's staging registry (structure_registry staging_unregistered)."""
    return (row.get("structure_registry") or "") == STAGING_UNREGISTERED


def select_values(rows, *, measure, method, arm="", set_name=None, roles=DEFAULT_ROLES, include_staging=False):
    """({pair: value}, {pair: field_role of a row excluded by role}, {pair: run_id of an excluded staging row}) of
    one (method, arm, measure[, set]); rows whose field_role is not in `roles` (None: every role) and, unless
    `include_staging`, rows with structure_registry staging_unregistered (review PS1_PS2 item d) are excluded and
    returned separately; a duplicate pair (among the admitted rows) is an error."""
    out, excluded, staging = {}, {}, {}
    for row in rows:
        if row["measure"] != measure or row["method"] != method or row["arm"] != arm:
            continue
        if set_name is not None and row["set"] != set_name:
            continue
        if not include_staging and is_staging(row):
            staging[row["pair"]] = row.get("run_id") or ""
            continue
        role = row.get("field_role") or ""
        if roles is not None and role not in roles:
            excluded[row["pair"]] = role
            continue
        if row["pair"] in out:
            raise ValueError(f"duplicate value for {method}:{arm} {measure} {row['pair']}")
        out[row["pair"]] = float(row["value"])
    return out, excluded, staging


def values_by_pair(rows, *, measure, method, arm="", set_name=None, roles=None, include_staging=True):
    """{pair: value} of one (method, arm, measure[, set]) over the rows with field_role in `roles` (None: all),
    staging rows included unless include_staging is False."""
    return select_values(rows, measure=measure, method=method, arm=arm, set_name=set_name, roles=roles,
                         include_staging=include_staging)[0]


def weighted_mean_median(d, w):
    """Per row of the weight matrix w (R, n) of the values d (n,): weighted mean and weighted median (average of the
    lower and upper weighted medians). Every row must have a positive total weight."""
    d = np.asarray(d, dtype=np.float64)
    w = np.asarray(w)
    total = w.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("weighted statistics: every row needs a positive total weight")
    mean = (w*d).sum(axis=1)/total
    order = np.argsort(d, kind="stable")
    ds = d[order]
    cs = np.cumsum(w[:, order], axis=1)
    lo = np.argmax(2*cs >= total[:, None], axis=1)
    hi = np.argmax(2*cs > total[:, None], axis=1)
    return mean, (ds[lo]+ds[hi])/2


def _subject_index(pairs):
    ends = [p.split("__") for p in pairs]
    if any(len(e) != 2 or not all(e) for e in ends):
        raise ValueError(f"pair names must be '<fixed>__<moving>': {pairs}")
    subjects = sorted({s for e in ends for s in e})
    index = {s: k for k, s in enumerate(subjects)}
    return np.array([index[e[0]] for e in ends]), np.array([index[e[1]] for e in ends]), len(subjects)


def bootstrap(d, pairs, *, block, resamples, rng):
    """Percentile 95% intervals of the mean and median difference (see the module docstring)."""
    d = np.asarray(d, dtype=np.float64)
    if block == "pairs":
        idx = rng.integers(0, len(d), size=(resamples, len(d)))
        means, medians = d[idx].mean(axis=1), np.median(d[idx], axis=1)
        subjects, redraws = None, 0
    elif block == "subjects":
        fi, mi, subjects = _subject_index(pairs)
        means, medians = np.empty(resamples), np.empty(resamples)
        todo, redraws = np.arange(resamples), 0
        while todo.size:
            counts = rng.multinomial(subjects, np.full(subjects, 1./subjects), size=todo.size)
            w = counts[:, fi]*counts[:, mi]
            ok = w.sum(axis=1) > 0
            if ok.any():
                means[todo[ok]], medians[todo[ok]] = weighted_mean_median(d, w[ok])
            redraws += int(np.count_nonzero(~ok))
            todo = todo[~ok]
    else:
        raise ValueError("block: 'subjects' or 'pairs'")
    ci = (lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))])
    return dict(block=block, resamples=int(resamples), ci95_mean=ci(means), ci95_median=ci(medians),
                n_subjects=subjects, redraws=redraws)


def wilcoxon(d):
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        return dict(statistic=None, p=None, zero_method="wilcox", alternative="two-sided")
    if np.all(d == 0):
        return dict(statistic=None, p=1.0, zero_method="wilcox", alternative="two-sided", note="all differences zero")
    result = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return dict(statistic=float(result.statistic), p=float(result.pvalue), zero_method="wilcox",
                alternative="two-sided")


def comparison_rng(seed, tag):
    sub = int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:4], "little")
    return np.random.default_rng([int(seed), sub])


def holm(pvalues):
    """Holm-Bonferroni adjusted p values (None stays None and is not part of the family)."""
    index = [k for k, p in enumerate(pvalues) if p is not None]
    out = [None]*len(pvalues)
    m, running = len(index), 0.
    for rank, k in enumerate(sorted(index, key=lambda k: pvalues[k])):
        running = max(running, min(1., (m-rank)*pvalues[k]))
        out[k] = running
    return out


def paired_compare(a, b, *, measure=None, tag=None, block="subjects", resamples=DEFAULT_RESAMPLES, seed=DEFAULT_SEED):
    """Paired comparison of {pair: value} dicts a and b (difference a - b) over the common pairs whose two values are
    finite; the others are listed and counted (module docstring)."""
    both = sorted(set(a) & set(b))
    nonfinite = {p: dict(a=format_value(a[p]), b=format_value(b[p])) for p in both
                 if not (np.isfinite(a[p]) and np.isfinite(b[p]))}
    common = [p for p in both if p not in nonfinite]
    d = np.array([a[p]-b[p] for p in common], dtype=np.float64)
    out = dict(measure=measure, n=len(common), n_common=len(both), pairs=common, missing_in_a=sorted(set(b)-set(a)),
               missing_in_b=sorted(set(a)-set(b)), nonfinite_pairs=nonfinite, n_nonfinite=len(nonfinite),
               n_nonfinite_a=sum(not np.isfinite(a[p]) for p in nonfinite),
               n_nonfinite_b=sum(not np.isfinite(b[p]) for p in nonfinite), difference="a - b")
    if not common:
        return dict(out, mean_diff=None, median_diff=None, n_positive=0, n_negative=0, n_zero=0,
                    wilcoxon=wilcoxon(d), bootstrap=None)
    out.update(mean_diff=float(d.mean()), median_diff=float(np.median(d)), n_positive=int(np.sum(d > 0)),
               n_negative=int(np.sum(d < 0)), n_zero=int(np.sum(d == 0)), wilcoxon=wilcoxon(d))
    if block == "subjects":
        out["wilcoxon"]["blocks_note"] = "pairs treated as independent (not adjusted for shared subjects)"
    direction = MEASURE_DIRECTION.get(measure)
    out["direction"] = direction
    out["n_a_better"] = None if direction is None else int(np.sum(direction*d > 0))
    rng = comparison_rng(seed, tag if tag is not None else f"{measure}")
    out["bootstrap"] = dict(bootstrap(d, common, block=block, resamples=resamples, rng=rng), seed=int(seed),
                            tag=tag if tag is not None else f"{measure}")
    return out


def compare_long(rows, a, b, measures, *, set_name=None, block="subjects", resamples=DEFAULT_RESAMPLES,
                 seed=DEFAULT_SEED, roles=DEFAULT_ROLES, include_staging=False):
    (ma, aa), (mb, ab) = parse_selector(a), parse_selector(b)
    results = []
    for measure in measures:
        va, xa, sa = select_values(rows, measure=measure, method=ma, arm=aa, set_name=set_name, roles=roles,
                                   include_staging=include_staging)
        vb, xb, sb = select_values(rows, measure=measure, method=mb, arm=ab, set_name=set_name, roles=roles,
                                   include_staging=include_staging)
        tag = f"{set_name or ''}|{measure}|{a}|{b}"
        results.append(dict(paired_compare(va, vb, measure=measure, tag=tag, block=block, resamples=resamples,
                                           seed=seed), a=a, b=b, set=set_name,
                            roles=None if roles is None else list(roles), excluded_by_role_a=xa,
                            excluded_by_role_b=xb, include_staging=bool(include_staging), excluded_staging_a=sa,
                            excluded_staging_b=sb))
    for result, p in zip(results, holm([r["wilcoxon"]["p"] for r in results])):
        result["wilcoxon"]["p_holm"] = p
        result["wilcoxon"]["holm_family"] = list(measures)
    return results


def parse_roles(text):
    """--roles: a comma list of field roles, or 'any' (no filter)."""
    if text.strip() == "any":
        return None
    roles = tuple(r for r in (s.strip() for s in text.split(",")) if r)
    if not roles:
        raise ValueError("--roles: a comma list of field roles or 'any'")
    return roles


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m bayesreg.eval.paired", description=__doc__.split("\n\n")[0])
    parser.add_argument("long", nargs="+", help="long CSV file(s) written by bayesreg.eval.cli")
    parser.add_argument("--a", required=True, help="METHOD[:ARM] (difference a - b)")
    parser.add_argument("--b", required=True, help="METHOD[:ARM]")
    parser.add_argument("--out", required=True, help="output JSON (must not exist)")
    parser.add_argument("--set", dest="set_name", default=None)
    parser.add_argument("--measures", default="seg35,seg4,dice_official,hd95,sdlogj_official,sdlogj_brain")
    parser.add_argument("--block", choices=("subjects", "pairs"), default="subjects")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--roles", default=",".join(DEFAULT_ROLES),
                        help="field roles compared (comma list; default native_formal,disp_only) or 'any'")
    parser.add_argument("--include-staging", action="store_true",
                        help="admit rows of runs on the staging segmenter registry (structure_registry "
                        "staging_unregistered, 未正式登记; excluded by default)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"{out}: exists; nothing is overwritten", file=sys.stderr)
        return 2
    try:
        roles = parse_roles(args.roles)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rows = read_long(args.long)
    results = compare_long(rows, args.a, args.b, [m for m in args.measures.split(",") if m], set_name=args.set_name,
                           block=args.block, resamples=args.resamples, seed=args.seed, roles=roles,
                           include_staging=args.include_staging)
    document = dict(schema="bayesreg.eval.paired.v3", inputs=[dict(path=str(p), sha256=hashlib.sha256(
        Path(p).read_bytes()).hexdigest()) for p in args.long], a=args.a, b=args.b, set=args.set_name,
        block=args.block, resamples=args.resamples, seed=args.seed, roles=None if roles is None else list(roles),
        include_staging=args.include_staging, definitions=__doc__, results=results)
    out.write_text(json.dumps(document, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    for r in results:
        boot = r["bootstrap"] or {}
        print(f"{r['measure']}: n {r['n']} (common {r['n_common']}; missing in a {len(r['missing_in_a'])}, in b "
              f"{len(r['missing_in_b'])}; non-finite {r['n_nonfinite']} (a {r['n_nonfinite_a']}, b {r['n_nonfinite_b']}); "
              f"excluded by role a {len(r['excluded_by_role_a'])}, b {len(r['excluded_by_role_b'])}; "
              f"staging excluded a {len(r['excluded_staging_a'])}, b {len(r['excluded_staging_b'])}) "
              f"mean {r['mean_diff']} median {r['median_diff']} positive {r['n_positive']} "
              f"Wilcoxon p {r['wilcoxon']['p']} Holm {r['wilcoxon'].get('p_holm')} "
              f"CI95 mean {boot.get('ci95_mean')} median {boot.get('ci95_median')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
