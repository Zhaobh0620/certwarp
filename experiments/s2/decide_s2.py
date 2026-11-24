"""S2's reading (runs/20261105-s2/plan.md, by code; a label for the lead and BZ, not a conclusion): analysis/decision_s2.json, summary_s2.md, long_curve.csv.

Only S_F decides.  Delta = S_F's final map - S_P's final map (seed_v1, only read), pair by pair, the first 5 development pairs, seg35, unrounded:
  premises not met                            不可判, the reasons written
  Delta >= +.002 and >= 4/5 positive          细档的好处过了原生层还在，扩到其余 14 对要交 BZ 或总负责人
  Delta <= 0 or <= 2/5 positive               过了原生层看不到
  otherwise                                   中间带
Premises: all 5 pairs of S_F have class ok, a certified final map bound to the exported field, the entry injected as recorded, the transfer accepted alpha = 1 at its first attempt and its state hash is the repaired field, which is the one repair_metrics.json of the repair records; the mean official SDlogJ <= .10; S_P's 5 final maps are there, certified, and give the pinned means.
Everything else is described: S_Pp, S_X, the curves (S_P's from tc_v1, another run: TC showed its sequence equal to seed_v1's within the common prefix), the retention from entry to final map, the rejected trials, the commits.  The line .88 is no gate.
"""
import csv
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]/'tc'))

import numpy as np  # noqa: E402

import decide_tc as dt  # noqa: E402
import tc_curve as curve  # noqa: E402

SCHEMA = 'bayesreg.s2.decision.v1'
N = 5
GAIN_BAR, MIN_POSITIVE, NULL_POSITIVE = .002, 4, 2
SDLOGJ_MAX = .10
S_P_PIN = (.8430, .8984)                                 # seed_v1, 5 pairs, seg35, seg4, to 4 decimals (runs/20261104-seed/result.md)
READING_ARM, ENTRY_ARM, EXTRA_ARM, REFERENCE = 'S_F', 'S_Pp', 'S_X', 'S_P'
KEPT, NOT_VISIBLE, MIDDLE, NOT_DECIDABLE = 'kept', 'not_visible', 'middle', 'not_decidable'
READINGS = {KEPT: '细档的好处过了原生层还在，扩到其余 14 对要交 BZ 或总负责人', NOT_VISIBLE: '过了原生层看不到', MIDDLE: '中间带', NOT_DECIDABLE: '不可判'}
MEAN_KEYS = dt.MEAN_KEYS
KEY_NAMES = dt.KEY_NAMES
GROUP_KEYS = ('seg35', 'seg4', 'cortex', 'white_matter', 'small_six', 'large_six', 'other_23')
DIFFERENCE_KEYS = ('seg35', 'seg4', 'cortex', 'white_matter', 'small_six', 'large_six', 'other_23', 'sdlogj_official', 'sdlogj_brain')
TASK_CLOCKS = (60, 65, 80, 95, 110, 140, 170, 200, 260, 320, 330)          # about 50 + t for S_P; the last evaluated state not later than each
ALPHA_CONFOUNDED = 'OASIS_OAS1_0439_MR1__OASIS_OAS1_0440_MR1'               # S_P is at alpha .9 here, S_F and S_Pp at 1
NOTE = ('开发集单次观测，读数不是结论，下一步交 BZ 或总负责人；n = 5，前 5 对之前选配用过（0455-0456 只在 S_Pp）。终态都是预算收尾的状态，不是 B 线收敛。只换入口：路径改变不改目标函数，原生层仍是标量 D；这不是 F_B，进方法先改合同。.88 的线不作门。'
        'S_P 不重跑：终态来自 seed_v1（0455-0456 来自 seed_v1b），曲线来自 tc_v1，都是另一时段；TC 已证 S_P 的序列在共同前缀内逐项相同，但 tc_v1 的终态与 seed_v1 不是同一个场。'
        '持时：S_Pp、S_F 都记 47 s，同 S_P；真实入口 S_P 约 58–67 s，S_F 约 57–70 s（runs/20261104-ef/result.md），S_X 的持时写在表里。同一个 t 不是同样的总时间。'
        '被拒比例的分母是 costs.levels[3] 的试探（S_P 前 5 对 .36），不能和 TC 结果的 217/673 = .32 比：那里的分母是 inner_trials，多算了每次内层调用的起点评估。')
HOLD_DIRECTION = 'S_X 记满，S_F、S_P 少记 10–23 s，同任务时钟的差对 S_X 偏不利。S_X 的持时 = seconds_task + seconds_repair 的中位，不含 warm-up，与 S_F 的 57–70 s 同口径。'
CLOCK_NOTE = '两种读法。**同 t**：原生层时钟，从原生层 solve_sequential 开始算，不含入口、持时、转移证书。**同任务时钟**：visible_seconds，预算 330 s 看的就是它；每一格取不晚于这个任务时钟、已经评过的最后一个状态（只有 t = 0、15、…、270 和终态评过），所以差里含最多一个评测点间隔的"陈旧"。'


def load_runs(root, plan):
    """{(arm, pair): {metrics, curve, cleanup, seed_entry, repair_metrics, rejection}} of what exists; the batch's own records only."""
    out = {}
    for arm, pair in plan:
        run = Path(root)/'runs'/arm/pair
        rec = {name: curve.read_json(run/f'{name}.json') for name in ('curve', 'metrics', 'cleanup', 'seed_entry') if (run/f'{name}.json').is_file()}
        if rec.get('metrics') is not None:
            rec['rejection'] = rejection_row(run)
        directory = (rec.get('seed_entry') or {}).get('repair_dir')
        if directory and (Path(directory)/'repair_metrics.json').is_file():
            rec['repair_metrics'] = curve.read_json(Path(directory)/'repair_metrics.json')
        if rec:
            out[(arm, pair)] = rec
    return out


def rejection_row(run):
    """What the native level did, from the run's own records (they stay after the cleanup): trials, Jacobian rejections and their share (costs.levels[3]: JACOBIAN_VIOLATION / attempts), commits, accepted inner steps, seconds per accepted step.  None without a summary."""
    task = curve.task_dir(run)
    if not (task/'summary.json').is_file() or not (task/'history.json').is_file():
        return None
    summary = curve.read_json(task/'summary.json')
    level = next((c for c in (summary.get('costs') or {}).get('levels') or [] if c.get('level') == curve.NATIVE_LEVEL), {})
    history = [r for r in curve.read_json(task/'history.json') if r.get('level') == curve.NATIVE_LEVEL]
    attempts, rejected = level.get('attempts'), (level.get('rejected_reasons') or {}).get('JACOBIAN_VIOLATION', 0)
    others = {k: v for k, v in (level.get('rejected_reasons') or {}).items() if k != 'JACOBIAN_VIOLATION'}
    steps = sum(r.get('inner_accepted_steps') or 0 for r in history)
    last = history[-1]['elapsed_seconds'] if history else None
    return dict(attempts=attempts, jacobian_rejected=rejected, rejected_share=None if not attempts else rejected/attempts, other_rejected=others, other_rejected_total=sum(others.values()), native_commits=summary.get('native_commits'), accepted_steps=steps, native_elapsed_last=last,
                seconds_per_accepted_step=None if not steps or last is None else last/steps, inner_accepted=level.get('inner_accepted'), status=summary.get('status'), e2e_seconds=summary.get('e2e_seconds'))


def reference_paths(pair, results, first_pairs):
    results = Path(results)
    final = results/('seed_v1' if pair in first_pairs else 'seed_v1b')/'runs/S_P'/pair
    return final, results/'tc_v1/runs/S_P'/pair


def load_references(results, pairs, first_pairs):
    """{pair: {metrics (seed_v1 or seed_v1b), rejection, curve and rejection_tc (tc_v1, the first 5), entry (the metrics.json of S_P's entry field)}}: only read."""
    out = {}
    for pair in pairs:
        final, again = reference_paths(pair, results, first_pairs)
        rec = {}
        if (final/'metrics.json').is_file():
            rec['metrics'] = curve.read_json(final/'metrics.json')
            rec['rejection'] = rejection_row(final)
        if (again/'curve.json').is_file():
            rec['curve'] = curve.read_json(again/'curve.json')
            rec['rejection_tc'] = rejection_row(again)
        if (final/'seed_entry.json').is_file():
            field = Path(curve.read_json(final/'seed_entry.json')['field_path'])
            if (field.parent/'metrics.json').is_file():
                rec['entry'] = curve.read_json(field.parent/'metrics.json')
        out[pair] = rec
    return out


def group_values(metrics, groups):
    """The reported values of one metrics dict, per pair: seg35, seg4, cortex and white matter (seg4 labels 1, 3), the groups' means over the labels present (ev.A5_GROUPS, as tc_curve.evaluate_field), the two SDlogJ.  A value not there: None."""
    metrics = metrics or {}
    per35 = {int(k): v for k, v in (metrics.get('seg35_per_label') or {}).items()}
    per4 = {int(k): v for k, v in (metrics.get('seg4_per_label') or {}).items()}
    out = dict(seg35=metrics.get('seg35'), seg4=metrics.get('seg4'), cortex=per4.get(1), white_matter=per4.get(3), sdlogj_official=metrics.get('sdlogj_official'), sdlogj_brain=metrics.get('sdlogj_brain'))
    for name, labels in groups.items():
        values = [per35[k] for k in labels if per35.get(k) is not None]
        out[name] = float(np.mean(values)) if values else None
    return out


def paired(a, b, pairs, key, groups):
    """Pair by pair a - b of one key; a, b: {pair: metrics dict}.  The mean only over all the pairs asked for, or None."""
    per_pair = {}
    for pair in pairs:
        x, y = group_values(a.get(pair), groups).get(key), group_values(b.get(pair), groups).get(key)
        if x is not None and y is not None:
            per_pair[pair] = x-y
    values = list(per_pair.values())
    complete = len(values) == len(pairs) and values
    return dict(per_pair=per_pair, mean=float(np.mean(values)) if complete else None, median=float(statistics.median(values)) if complete else None, positive=sum(v > 0 for v in values), n=len(values))


def label_of(delta, positive):
    """The plan's words; the bars included, the mean unrounded."""
    if delta >= GAIN_BAR and positive >= MIN_POSITIVE:
        return KEPT
    if delta <= 0 or positive <= NULL_POSITIVE:
        return NOT_VISIBLE
    return MIDDLE


def run_premises(pair, rec, short):
    """Why this run of S_F cannot enter the reading (empty: it can)."""
    name = f'{READING_ARM} {short}'
    metrics = (rec or {}).get('metrics')
    if metrics is None:
        return [f'{name}: 没有 metrics.json']
    bad = []
    cls = (metrics.get('classification') or {}).get('class')
    if cls != 'ok':
        bad.append(f'{name}: 分类 {cls}')
    certificate = metrics.get('certificate') or {}
    if not (certificate.get('passed') and certificate.get('state_hash_matches_u') is True and certificate.get('final_npz_sha256_matches') is True):
        bad.append(f'{name}: 终态证书未通过，或没有与导出场绑定')
    seed = metrics.get('seed') or {}
    if seed.get('passed') is not True:
        bad.append(f"{name}: 入口核对未通过 {seed.get('problems')}")
    if seed.get('accepted_alpha') != 1.:
        bad.append(f"{name}: 转移 α = {seed.get('accepted_alpha')}，不是 1")
    attempts = seed.get('transfer_attempts') or []
    repaired = seed.get('repaired_u_sha256')
    if not attempts or repaired is None or attempts[0].get('state_hash') != repaired:
        bad.append(f"{name}: 转移的 state hash {attempts[0].get('state_hash') if attempts else None} 不等于修复场 {repaired}")
    recorded = ((rec or {}).get('repair_metrics') or {}).get('repaired_u_sha256')
    if recorded is None or recorded != repaired:         # review minor 6: this run's record against the evaluation's own, in ef_v1
        bad.append(f'{name}: 修复记录（repair_metrics.json）的修复场 {recorded} 不等于这次运行的 {repaired}')
    return bad


def reference_premises(references, pairs, short, pin=S_P_PIN):
    bad = []
    finals = []
    for pair in pairs:
        doc = (references.get(pair) or {}).get('metrics')
        if doc is None:
            bad.append(f'S_P {short(pair)}: 没有 metrics.json（seed_v1）')
            continue
        if not (doc.get('certificate') or {}).get('passed'):
            bad.append(f'S_P {short(pair)}: 终态证书未通过')
        finals.append(doc.get('metrics') or {})
    if not bad and pin is not None:
        got = (round(float(np.mean([x.get('seg35') for x in finals])), 4), round(float(np.mean([x.get('seg4') for x in finals])), 4))
        if got != pin:
            bad.append(f'S_P 的终态 5 对均值 {got}，不是钉住的 {pin}')
    return bad


def decide(runs, references, pairs, short, groups, *, pin=S_P_PIN):
    reasons = [r for pair in pairs for r in run_premises(pair, runs.get((READING_ARM, pair)), short(pair))]
    reasons += reference_premises(references, pairs, short, pin)
    ours = {p: ((runs.get((READING_ARM, p)) or {}).get('metrics') or {}).get('metrics') for p in pairs}
    ref = {p: ((references.get(p) or {}).get('metrics') or {}).get('metrics') for p in pairs}
    delta = paired(ours, ref, pairs, 'seg35', groups)
    delta4 = paired(ours, ref, pairs, 'seg4', groups)
    values = [v.get('sdlogj_official') for v in ours.values() if v]
    sdlogj = float(np.mean(values)) if len(values) == len(pairs) and all(v is not None for v in values) else None
    if sdlogj is not None and not sdlogj <= SDLOGJ_MAX:
        reasons.append(f'官方 SDlogJ 均值 {sdlogj:.4f} > {SDLOGJ_MAX}')
    if not reasons and (delta['mean'] is None or sdlogj is None):
        reasons.append(f"前提都满足，但只有 {delta['n']} 对的 seg35 或缺官方 SDlogJ，Δ 不可算")
    code = NOT_DECIDABLE if reasons else label_of(delta['mean'], delta['positive'])
    if reasons:
        text = READINGS[code]+'：'+'；'.join(reasons)
    else:
        text = f"{READINGS[code]}：Δ(seg35) = S_F 终态 − S_P 终态 = {delta['mean']:+.4f}，{delta['positive']}/{N} 为正；Δ(seg4) {delta4['mean']:+.4f}（{delta4['positive']}/{N}），官方 SDlogJ 均值 {sdlogj:.4f}，都只描述，不进判定"
    return dict(reading_code=code, reading=text, reasons=reasons, delta_seg35=delta, delta_seg4=delta4, mean_sdlogj_official=sdlogj)


# ---- the descriptions ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def final_differences(a, b, pairs, groups):
    """{key: paired} over the reported keys."""
    return {key: paired(a, b, pairs, key, groups) for key in DIFFERENCE_KEYS}


def curve_records(runs, references, arm, pairs):
    """{pair: rec with a curve} of an arm; S_P: tc_v1's."""
    if arm == REFERENCE:
        return {p: references[p] for p in pairs if (references.get(p) or {}).get('curve')}
    return {p: runs[(arm, p)] for p in pairs if ((runs.get((arm, p)) or {}).get('curve'))}


def arm_table(records, pairs):
    """One row per time: the means over the pairs it has, and how many; both clocks."""
    rows = []
    for t in curve.TIMES+(curve.FINAL,):
        points = [dt.point_at(records.get(pair), t) for pair in pairs]
        points = [p for p in points if p and p.get('metrics')]
        row = dict(t=t, n=len(points))
        for key in MEAN_KEYS:
            row[key] = mean_or_none(p['metrics'].get(key) for p in points)
        for key in ('elapsed_seconds', 'visible_seconds', 'D', 'GF', 'min_jd_lower_bound'):
            row[key] = mean_or_none(p.get(key) for p in points)
        row['commit'] = mean_or_none(p.get('commit') for p in points)
        rows.append(row)
    return rows


def same_t_table(a, b, pairs):
    """a(t) - b(t), pair by pair, at each native time."""
    rows = []
    for t in curve.TIMES+(curve.FINAL,):
        row = dict(t=t, task_clock_a=mean_or_none((dt.point_at(a.get(p), t) or {}).get('visible_seconds') for p in pairs), task_clock_b=mean_or_none((dt.point_at(b.get(p), t) or {}).get('visible_seconds') for p in pairs))
        for key in MEAN_KEYS:
            values = []
            for pair in pairs:
                x, y = dt.value_at(a.get(pair), t, key), dt.value_at(b.get(pair), t, key)
                if x is not None and y is not None:
                    values.append(x-y)
            row[key] = dict(mean=float(np.mean(values)) if values else None, positive=sum(v > 0 for v in values), n=len(values))
        rows.append(row)
    return rows


def at_task_clock(rec, tau):
    """The evaluated state of a run latest not after the task clock tau (among its t points and the final map), or None."""
    c = (rec or {}).get('curve') or {}
    points = [p for p in list(c.get('points') or [])+[c.get('final') or {}] if p.get('metrics') and p.get('visible_seconds') is not None and p['visible_seconds'] <= tau]
    return max(points, key=lambda p: p['visible_seconds']) if points else None


def same_clock_table(a, b, pairs, clocks=TASK_CLOCKS):
    """a - b at the same task clock, pair by pair, only where both already have a state; n says how many pairs; age: how old the states taken are, on average."""
    rows = []
    for tau in clocks:
        row = dict(task_clock=tau)
        chosen = [(at_task_clock(a.get(p), tau), at_task_clock(b.get(p), tau)) for p in pairs]
        row['n_a'], row['n_b'] = sum(x is not None for x, _ in chosen), sum(y is not None for _, y in chosen)
        row['age_a'], row['age_b'] = mean_or_none(tau-x['visible_seconds'] for x, _ in chosen if x), mean_or_none(tau-y['visible_seconds'] for _, y in chosen if y)
        both = [(x, y) for x, y in chosen if x and y]
        for key in MEAN_KEYS:
            values = [x['metrics'][key]-y['metrics'][key] for x, y in both if x['metrics'].get(key) is not None and y['metrics'].get(key) is not None]
            row[key] = dict(mean=float(np.mean(values)) if values else None, positive=sum(v > 0 for v in values), n=len(values))
        rows.append(row)
    return rows


def retention_table(runs, references, arm, pairs, groups):
    """From entry to final map, by group: Delta at the entry (the repaired field, alpha = 1, against S_P's entry field at alpha = 1: both before any native step) and at the final map (against S_P's final map), the share kept.  Means over the pairs that have both.
    seg35, seg4 also against S_P's entry at the alpha it accepted, what the native level really started from (the question of the brief: +.0184 / +.0268; at_accepted_alpha holds only these two Dice)."""
    rows = []
    entry_a = {p: ((runs.get((arm, p)) or {}).get('repair_metrics') or {}).get('after', {}).get('metrics') for p in pairs}
    entry_b = {p: ((references.get(p) or {}).get('entry') or {}).get('metrics') for p in pairs}
    final_a = {p: ((runs.get((arm, p)) or {}).get('metrics') or {}).get('metrics') for p in pairs}
    final_b = {p: ((references.get(p) or {}).get('metrics') or {}).get('metrics') for p in pairs}
    entry_accepted = {p: ((references.get(p) or {}).get('entry') or {}).get('at_accepted_alpha') for p in pairs}

    def share(final, entry):
        return None if entry['mean'] is None or final['mean'] is None or entry['mean'] == 0 else final['mean']/entry['mean']
    for key in GROUP_KEYS:
        d_entry, d_final = paired(entry_a, entry_b, pairs, key, groups), paired(final_a, final_b, pairs, key, groups)
        kept = share(d_final, d_entry)
        d_accepted = paired(entry_a, entry_accepted, pairs, key, groups) if key in ('seg35', 'seg4') else None
        rows.append(dict(key=key, delta_entry_accepted=d_accepted, kept_accepted=None if d_accepted is None else share(d_final, d_accepted), entry_arm=mean_or_none(group_values(entry_a.get(p), groups).get(key) for p in pairs), entry_s_p=mean_or_none(group_values(entry_b.get(p), groups).get(key) for p in pairs), delta_entry=d_entry, final_arm=mean_or_none(group_values(final_a.get(p), groups).get(key) for p in pairs),
                         final_s_p=mean_or_none(group_values(final_b.get(p), groups).get(key) for p in pairs), delta_final=d_final, kept=kept))
    return rows


def rejection_table(runs, references, arm, pairs):
    """Pair by pair: the arm's native level against S_P's in seed_v1 (seed_v1b for 0455-0456) and in tc_v1."""
    rows = []
    for pair in pairs:
        rec = runs.get((arm, pair)) or {}
        ref = references.get(pair) or {}
        rows.append(dict(pair=pair, arm=rec.get('rejection'), s_p=ref.get('rejection'), s_p_tc=ref.get('rejection_tc')))
    return rows


def rejection_means(rows, name):
    keep = [r[name] for r in rows if r.get(name)]
    return {key: mean_or_none(row.get(key) for row in keep) for key in ('attempts', 'jacobian_rejected', 'rejected_share', 'other_rejected_total', 'native_commits', 'accepted_steps', 'seconds_per_accepted_step')} | {'n': len(keep)}


def entry_gain_table(runs, references, pairs, groups, short):
    """S_Pp against S_P pair by pair: the gain at the entry (repaired field at alpha = 1, S_P's entry at the alpha it accepted) and at the final map, the share kept.  This is the only difference of the two arms."""
    rows = []
    for pair in pairs:
        rec, ref = runs.get((ENTRY_ARM, pair)) or {}, references.get(pair) or {}
        entry = ((rec.get('repair_metrics') or {}).get('after') or {}).get('metrics') or {}
        at_alpha = (ref.get('entry') or {}).get('at_accepted_alpha') or {}
        final, reference = ((rec.get('metrics') or {}).get('metrics')), ((ref.get('metrics') or {}).get('metrics'))
        row = dict(pair=short(pair), s_p_accepted_alpha=(((ref.get('metrics') or {}).get('seed') or {}).get('accepted_alpha')), s_pp_accepted_alpha=(((rec.get('metrics') or {}).get('seed') or {}).get('accepted_alpha')))
        for key in ('seg35', 'seg4'):
            gain = None if entry.get(key) is None or at_alpha.get(key) is None else entry[key]-at_alpha[key]
            x, y = group_values(final, groups).get(key), group_values(reference, groups).get(key)
            at_end = None if x is None or y is None else x-y
            row[f'{key}_entry_gain'], row[f'{key}_final_gain'], row[f'{key}_kept'] = gain, at_end, None if gain in (None, 0) or at_end is None else at_end/gain
        for key in DIFFERENCE_KEYS:
            x, y = group_values(final, groups).get(key), group_values(reference, groups).get(key)
            row[f'final_{key}'] = None if x is None or y is None else x-y
        rows.append(row)
    return rows


def anomalies(runs, plan, short):
    out = []
    for arm, pair in plan:
        rec = runs.get((arm, pair))
        if rec is None:
            continue
        name = dict(arm=arm, pair=short(pair))
        metrics = rec.get('metrics')
        if metrics is None:
            out.append(dict(name, what='没有 metrics.json'))
            continue
        status = metrics.get('status')
        if status != 'native_budget_exhausted':
            out.append(dict(name, what=f'状态 {status}（其余都是预算收尾）'))
        value = (metrics.get('metrics') or {}).get('sdlogj_official')
        if value is not None and value > SDLOGJ_MAX:
            out.append(dict(name, what=f'终态官方 SDlogJ {value:.4f} > {SDLOGJ_MAX}'))
        c = rec.get('curve')
        if c is None:
            out.append(dict(name, what='没有 curve.json'))
            continue
        for p in c.get('points') or []:
            if p.get('commit') is None:
                out.append(dict(name, what=p.get('reason')))
        if (c.get('final') or {}).get('is_last_commit') is False:
            out.append(dict(name, what='终态不是最后一次提交'))
        unrecorded = (rec.get('cleanup') or {}).get('not_deleted_unrecorded') or []
        if unrecorded:
            out.append(dict(name, what=f"{len(unrecorded)} 个中间文件没有 sha256 记录，没有删：{', '.join(row['name'] for row in unrecorded)}"))
    return out


def describe(runs, references, pairs, s_pp_pairs, plan, short, groups):
    """Everything the plan lists as description."""
    ours = {p: ((runs.get((READING_ARM, p)) or {}).get('metrics') or {}).get('metrics') for p in pairs}
    ref = {p: ((references.get(p) or {}).get('metrics') or {}).get('metrics') for p in pairs}
    others = tuple(p for p in pairs if p != ALPHA_CONFOUNDED)
    s_f, s_p = curve_records(runs, references, READING_ARM, pairs), curve_records(runs, references, REFERENCE, pairs)
    out = dict(final_differences=final_differences(ours, ref, pairs, groups), without_alpha_pair=final_differences(ours, ref, others, groups), alpha_pair=final_differences(ours, ref, (ALPHA_CONFOUNDED,), groups))
    s_pp_final = {ALPHA_CONFOUNDED: ((runs.get((ENTRY_ARM, ALPHA_CONFOUNDED)) or {}).get('metrics') or {}).get('metrics')}
    out['alpha_pair_split'] = dict(s_f_minus_s_pp=final_differences(ours, s_pp_final, (ALPHA_CONFOUNDED,), groups), s_pp_minus_s_p=final_differences(s_pp_final, ref, (ALPHA_CONFOUNDED,), groups))
    out['tables'] = {arm: arm_table(curve_records(runs, references, arm, pairs if arm != ENTRY_ARM else s_pp_pairs), pairs if arm != ENTRY_ARM else s_pp_pairs) for arm in (READING_ARM, REFERENCE, ENTRY_ARM, EXTRA_ARM)}
    out['same_t'] = same_t_table(s_f, s_p, pairs)
    out['same_clock'] = same_clock_table(s_f, s_p, pairs)
    out['retention'] = retention_table(runs, references, READING_ARM, pairs, groups)
    rejection = rejection_table(runs, references, READING_ARM, pairs)
    out['rejection'] = dict(rows=[dict(r, pair=short(r['pair'])) for r in rejection], arm=rejection_means(rejection, 'arm'), s_p=rejection_means(rejection, 's_p'), s_p_tc=rejection_means(rejection, 's_p_tc'))
    rejection_pp = rejection_table(runs, references, ENTRY_ARM, s_pp_pairs)
    out['s_pp'] = dict(rows=entry_gain_table(runs, references, s_pp_pairs, groups, short), rejection=[dict(r, pair=short(r['pair'])) for r in rejection_pp], same_t=same_t_table({p: r for p, r in curve_records(runs, references, ENTRY_ARM, s_pp_pairs).items()}, curve_records(runs, references, REFERENCE, tuple(p for p in s_pp_pairs if p in pairs)), tuple(p for p in s_pp_pairs if p in pairs)))
    s_x = {p: ((runs.get((EXTRA_ARM, p)) or {}).get('metrics') or {}).get('metrics') for p in pairs}
    x_curves = curve_records(runs, references, EXTRA_ARM, pairs)
    holds = [((runs.get((EXTRA_ARM, p)) or {}).get('seed_entry') or {}).get('charge_seconds') for p in pairs]
    rejection_x = rejection_table(runs, references, EXTRA_ARM, pairs)
    out['s_x'] = dict(ran=any((runs.get((EXTRA_ARM, p)) or {}).get('metrics') for p in pairs), hold_seconds=sorted({h for h in holds if h is not None}), hold=next((((runs.get((EXTRA_ARM, p)) or {}).get('seed_entry') or {}).get('hold') for p in pairs if (runs.get((EXTRA_ARM, p)) or {}).get('seed_entry')), None), against_s_f=final_differences(s_x, ours, pairs, groups), against_s_p=final_differences(s_x, ref, pairs, groups), same_t_against_s_f=same_t_table(x_curves, s_f, pairs), same_clock_against_s_f=same_clock_table(x_curves, s_f, pairs), same_clock_against_s_p=same_clock_table(x_curves, s_p, pairs), rejection=rejection_means(rejection_x, 'arm'),
                      source=sorted({str(((runs.get((EXTRA_ARM, p)) or {}).get('seed_entry') or {}).get('source_tier')) for p in pairs if (runs.get((EXTRA_ARM, p)) or {}).get('seed_entry')}))
    out['anomalies'] = anomalies(runs, plan, short)
    return out


# ---- the text ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
f, signed = dt.f, dt.signed


def signed_paired(row, digits=4):
    return '—' if row.get('mean') is None else f"{row['mean']:+.{digits}f}（{row['positive']}/{row['n']}）"


def difference_lines(title, differences):
    lines = ['', title, '', '| 项 | 差（均值，为正的对数/对数） |', '|---|---|']
    lines += [f"| {KEY_NAMES[key]} | {signed_paired(differences[key])} |" for key in DIFFERENCE_KEYS]
    return lines


def curve_lines(title, table):
    lines = ['', title, '', '| t（原生层 s） | 任务时钟（s） | n | 平均提交序号 | '+' | '.join(KEY_NAMES[k] for k in MEAN_KEYS)+' | D | GF |', '|---|---|---|---|'+'---|'*(len(MEAN_KEYS)+2)]
    for row in table:
        lines.append(f"| {row['t']} | {f(row['visible_seconds'], 1)} | {row['n']} | {f(row['commit'], 1)} | "+' | '.join(f(row[k]) for k in MEAN_KEYS)+f" | {f(row['D'], 1)} | {f(row['GF'], 3)} |")
    return lines


def same_t_lines(title, table):
    lines = ['', title, '', '每行是同一个原生层时刻 t，不是同一个总时间；时钟写的是"前者 / 后者"。', '', '| t（原生层 s） | 任务时钟（s） | '+' | '.join(KEY_NAMES[k] for k in MEAN_KEYS)+' |', '|---|---|'+'---|'*len(MEAN_KEYS)]
    for row in table:
        lines.append(f"| {row['t']} | {f(row['task_clock_a'], 0)} / {f(row['task_clock_b'], 0)} | "+' | '.join(signed_paired(row[k]) for k in MEAN_KEYS)+' |')
    return lines


def same_clock_lines(title, table):
    lines = ['', title, '', '| 任务时钟（s） | 有状态的对数（前者 / 后者） | 所取状态的陈旧秒数（前者 / 后者） | '+' | '.join(KEY_NAMES[k] for k in MEAN_KEYS)+' |', '|---|---|---|'+'---|'*len(MEAN_KEYS)]
    for row in table:
        lines.append(f"| {row['task_clock']} | {row['n_a']} / {row['n_b']} | {f(row['age_a'], 1)} / {f(row['age_b'], 1)} | "+' | '.join(signed_paired(row[k]) for k in MEAN_KEYS)+' |')
    return lines


def rejection_line(name, row):
    return f"| {name} | {f(row['n'], 0)} | {f(row['attempts'], 1)} | {f(row['jacobian_rejected'], 1)} | {f(row['rejected_share'], 3)} | {f(row['other_rejected_total'], 1)} | {f(row['native_commits'], 1)} | {f(row['accepted_steps'], 1)} | {f(row['seconds_per_accepted_step'], 3)} |"


def summary_text(result):
    d = result['description']
    d35, d4 = result['delta_seg35'], result['delta_seg4']
    code = result['decide_code']
    warning = ['', f"**判定代码与队列第一次启动时不同**：batch.json {code['at_first_start']}，现在 {code['now']}。线是预登记的，下面的读数要对照 plan.md 的修订读。"] if code['same'] is False else []
    lines = ['# S2 汇总（代码写出，读数不是结论）', '', result['note'], *warning, '', f"**读数（只看 S_F）**：{result['reading']}", '', f"结果根目录 {result['root']}；写出 {result['written_utc']}。", '', '## 判定：Δ = S_F 终态 − S_P 终态（seed_v1）', '', '| 对 | Δ seg35 | Δ seg4 | S_P 的转移 α | S_F 的转移 α |', '|---|---|---|---|---|']
    for pair in result['pairs']:
        s = short_of(result, pair)
        alpha_ref = result['alpha'].get('S_P', {}).get(pair)
        alpha_arm = result['alpha'].get('S_F', {}).get(pair)
        a, b = d35['per_pair'].get(pair), d4['per_pair'].get(pair)
        lines.append(f"| {s} | {'—' if a is None else format(a, '+.4f')} | {'—' if b is None else format(b, '+.4f')} | {alpha_ref} | {alpha_arm} |")
    lines += ['', f"均值 Δ seg35 {f(d35['mean'])}（{d35['positive']}/{d35['n']} 为正），Δ seg4 {f(d4['mean'])}（{d4['positive']}/{d4['n']} 为正），官方 SDlogJ 均值 {f(result['mean_sdlogj_official'])}。线：Δ ≥ +.002 且 ≥ 4/5 为正，好处还在；Δ ≤ 0 或 ≤ 2/5 为正，看不到；其余中间带。"]
    lines += difference_lines('## 只描述：S_F 终态 − S_P 终态，各项', d['final_differences'])
    lines += difference_lines('## 只描述：去掉 0439-0440（S_P 在这一对是 α = .9，有 α 混淆），n = 4', d['without_alpha_pair'])
    lines += ['', '## 只描述：0439-0440 单独（n = 1），拆成 (S_F − S_Pp) + (S_Pp − S_P)', '', 'S_F、S_Pp 在这一对都是 α = 1，S_F − S_Pp 只差入口档，不含 α；S_Pp − S_P 只差 α。', '', '| 项 | S_F − S_P | S_F − S_Pp | S_Pp − S_P |', '|---|---|---|---|']
    split = d['alpha_pair_split']
    lines += [f"| {KEY_NAMES[key]} | {sign4(d['alpha_pair'][key]['mean'])} | {sign4(split['s_f_minus_s_pp'][key]['mean'])} | {sign4(split['s_pp_minus_s_p'][key]['mean'])} |" for key in DIFFERENCE_KEYS]
    lines += ['', '## 只描述：入口 → 终态的留存，逐组（S_F 对 S_P）', '', '入口：修复场（α = 1）对 S_P 的入口场（也按 α = 1，都没有过原生层）；终态：对 seed_v1。**两个分母**：S_P 的 α = 1 场在 0439-0440 上从来没有进过原生层（那一对实际从 α = .9 出发），"接受 α"那两列才是任务书问的 +.0184 / +.0268，只有 seg35、seg4 有。留存 = Δ 终态 / Δ 入口，只是均值之比，分母小时不稳。', '', '| 项 | S_P 入口（α = 1） | S_F 入口 | Δ 入口（α = 1） | Δ 入口（S_P 接受 α） | S_P 终态 | S_F 终态 | Δ 终态 | 留存（α = 1） | 留存（接受 α） |', '|---|---|---|---|---|---|---|---|---|---|']
    for row in d['retention']:
        lines.append(f"| {KEY_NAMES[row['key']]} | {f(row['entry_s_p'])} | {f(row['entry_arm'])} | {signed_paired(row['delta_entry'])} | {'—' if row['delta_entry_accepted'] is None else signed_paired(row['delta_entry_accepted'])} | {f(row['final_s_p'])} | {f(row['final_arm'])} | {signed_paired(row['delta_final'])} | {f(row['kept'], 2)} | {f(row['kept_accepted'], 2)} |")
    lines += ['', '## 只描述：被拒比例、提交数、每步秒数', '', '被拒比例 = JACOBIAN_VIOLATION / 试探（summary.json 的 costs.levels[3]）。这里的 .36 不能和 TC 结果的 217/673 = .32 比，分母不同（那里是 inner_trials，含每次内层调用的起点评估）。其他拒绝单列（例如 FEASIBILITY_UNCERTAIN），只报 JACOBIAN_VIOLATION 会低估。', '', '| 臂 | n | 试探 | 被拒（Jacobian） | 被拒比例 | 其他拒绝 | 提交数 | 接受步 | 每接受步秒数 |', '|---|---|---|---|---|---|---|---|---|', rejection_line('S_F', d['rejection']['arm']), rejection_line('S_P（seed_v1）', d['rejection']['s_p']), rejection_line('S_P（tc_v1）', d['rejection']['s_p_tc'])]
    lines += ['', '逐对：', '', '| 对 | S_F 被拒 / 试探 | S_P（seed_v1）被拒 / 试探 | S_F 其他拒绝 | S_F 提交 | S_P 提交 | S_F 每步秒数 | S_P 每步秒数 |', '|---|---|---|---|---|---|---|---|']
    for row in d['rejection']['rows']:
        a, b = row.get('arm') or {}, row.get('s_p') or {}
        lines.append(f"| {row['pair']} | {a.get('jacobian_rejected')} / {a.get('attempts')} | {b.get('jacobian_rejected')} / {b.get('attempts')} | {a.get('other_rejected') or 0} | {a.get('native_commits')} | {b.get('native_commits')} | {f(a.get('seconds_per_accepted_step'), 3)} | {f(b.get('seconds_per_accepted_step'), 3)} |")
    lines += ['', '## 只描述：两种时钟', '', CLOCK_NOTE]
    lines += curve_lines('### S_F 各时刻的均值', d['tables'][READING_ARM])
    lines += curve_lines('### S_P 各时刻的均值（tc_v1，另一时段）', d['tables'][REFERENCE])
    lines += same_t_lines('### S_F(t) − S_P(t)，同 t（S_P 是 tc_v1，另一时段；最后一行是 tc_v1 的终态，不是判定的 Δ，判定用 seed_v1）', d['same_t'])
    lines += same_clock_lines('### S_F − S_P，同任务时钟（S_P 是 tc_v1，另一时段）', d['same_clock'])
    lines += ['', '## 只描述：S_Pp（只差 α）', '', '入口差：修复场 α = 1 对 S_P 实际接受的 α。终态差：对 S_P 终态（0439-0440 是 seed_v1，0455-0456 是 seed_v1b）。0455-0456 没有 S_P 的曲线。S_Pp 走通的是注入路径，不是 S_F 的 J 区间：它只动了几十个节点，转移下界也高得多。', '', '| 对 | S_P / S_Pp 的 α | 入口 seg35 / seg4 | 终态 seg35 / seg4 | 留存 seg35 / seg4 | 终态皮层 | 终态小六个 | 官方 SDlogJ 差 |', '|---|---|---|---|---|---|---|---|']
    for row in d['s_pp']['rows']:
        lines.append(f"| {row['pair']} | {row['s_p_accepted_alpha']} / {row['s_pp_accepted_alpha']} | {sign4(row['seg35_entry_gain'])} / {sign4(row['seg4_entry_gain'])} | {sign4(row['seg35_final_gain'])} / {sign4(row['seg4_final_gain'])} | {f(row['seg35_kept'], 2)} / {f(row['seg4_kept'], 2)} | {sign4(row['final_cortex'])} | {sign4(row['final_small_six'])} | {sign4(row['final_sdlogj_official'])} |")
    lines += ['', '| 对 | S_Pp 被拒 / 试探 | S_P 被拒 / 试探 | S_Pp 提交 | S_P 提交 |', '|---|---|---|---|---|']
    for row in d['s_pp']['rejection']:
        a, b = row.get('arm') or {}, row.get('s_p') or {}
        lines.append(f"| {row['pair']} | {a.get('jacobian_rejected')} / {a.get('attempts')} | {b.get('jacobian_rejected')} / {b.get('attempts')} | {a.get('native_commits')} | {b.get('native_commits')} |")
    lines += curve_lines('### S_Pp 各时刻的均值（n 见表）', d['tables'][ENTRY_ARM])
    lines += same_t_lines('### S_Pp(t) − S_P(t)，同 t，只有 0439-0440（S_P 是 tc_v1，另一时段）', d['s_pp']['same_t'])
    lines += ['', '## 只描述：S_X（FL 最好的臂）', '']
    x = d['s_x']
    if not x['ran']:
        lines.append('没有跑（要显式给参数才有）。')
    else:
        hold = x['hold'] or {}
        lines.append(f"来源 {x['source']}；持时 {x['hold_seconds']} s（seconds_task + seconds_repair 的中位 {f(hold.get('median_without_warmup'), 1)} s，含 warm-up {f(hold.get('median_with_warmup'), 1)} s）；预算总数不变，所以原生层相应变短，同 t 与同任务时钟要分开读。{HOLD_DIRECTION}")
        lines += difference_lines('### S_X 终态 − S_F 终态', x['against_s_f'])
        lines += difference_lines('### S_X 终态 − S_P 终态', x['against_s_p'])
        lines += ['', '| 臂 | n | 试探 | 被拒（Jacobian） | 被拒比例 | 其他拒绝 | 提交数 | 接受步 | 每接受步秒数 |', '|---|---|---|---|---|---|---|---|---|', rejection_line('S_X', x['rejection'])]
        lines += curve_lines('### S_X 各时刻的均值', d['tables'][EXTRA_ARM])
        lines += same_t_lines('### S_X(t) − S_F(t)，同 t', x['same_t_against_s_f'])
        lines += same_clock_lines('### S_X − S_F，同任务时钟', x['same_clock_against_s_f'])
        lines += same_clock_lines('### S_X − S_P，同任务时钟（S_P 是 tc_v1，另一时段）', x['same_clock_against_s_p'])
    lines += ['', '## 异常', '']
    lines += [f"- {a['arm']} {a['pair']}: {a['what']}" for a in d['anomalies']] or ['没有。']
    lines += ['', '## 它证伪不了什么', '', '路径改变不改目标函数，原生层仍是标量 D；开发集单次，n = 5，前 5 对选配用过；这不是 F_B，进方法先改合同。']
    return '\n'.join(lines)+'\n'


def sign4(value):
    return '—' if value is None or (isinstance(value, float) and math.isnan(value)) else f'{value:+.4f}'


def short_of(result, pair):
    return result['short'][pair]


def long_rows(runs, references, plan, pairs, short):
    """One row per arm, pair, time and measure; t final is the final map.  S_P is tc_v1's, the arm named so."""
    records = [(arm, pair, runs.get((arm, pair)) or {}) for arm, pair in plan]+[('S_P_tc_v1', pair, references.get(pair) or {}) for pair in pairs]
    for arm, pair, rec in records:
        c = rec.get('curve')
        if c is None:
            continue
        for t in curve.TIMES+(curve.FINAL,):
            point = dt.point_at(rec, t)
            if not point or not point.get('metrics'):
                continue
            base = dict(arm=arm, pair=pair, short=short(pair), t=t, commit=point.get('commit'), elapsed_seconds=point.get('elapsed_seconds'), visible_seconds=point.get('visible_seconds'), run_id=c.get('run_id'))
            for key in MEAN_KEYS:
                if point['metrics'].get(key) is not None:
                    yield dict(base, measure=key, value=point['metrics'][key])
            for key in ('D', 'F', 'GF', 'min_jd_lower_bound'):
                if point.get(key) is not None:
                    yield dict(base, measure=key, value=point[key])


def write_analysis(root, plan, pairs, s_pp_pairs, short, *, ev, utc, results, pin=S_P_PIN):
    """Reads the runs of this batch and S_P's records, writes the three files under root/analysis; an earlier summary goes to analysis/previous_<UTC>/.  Returns the decision."""
    root = Path(root)
    runs = load_runs(root, plan)
    every = tuple(dict.fromkeys(tuple(pairs)+tuple(s_pp_pairs)))
    references = load_references(results, every, pairs)
    groups = ev.A5_GROUPS
    result = dict(schema=SCHEMA, root=str(root), written_utc=utc(), note=NOTE, clock_note=CLOCK_NOTE, rule='runs/20261105-s2/plan.md', reading_arm=READING_ARM, pairs=list(pairs), short={p: short(p) for p in every}, **decide(runs, references, pairs, short, groups, pin=pin))
    result['alpha'] = dict(S_P={p: (((references.get(p) or {}).get('metrics') or {}).get('seed') or {}).get('accepted_alpha') for p in pairs}, S_F={p: (((runs.get((READING_ARM, p)) or {}).get('metrics') or {}).get('seed') or {}).get('accepted_alpha') for p in pairs})
    result['description'] = describe(runs, references, pairs, s_pp_pairs, plan, short, groups)
    now = curve.sha_file(__file__)
    pinned = (curve.read_json(root/'batch.json') if (root/'batch.json').is_file() else {}).get('decide_sha256')
    result['decide_code'] = dict(at_first_start=pinned, now=now, same=None if pinned is None else pinned == now)
    result['sources'] = dict(evaluator_sha256=curve.sha_file(ev.__file__), code_sha256=curve.sha_file(__file__), curve_code_sha256=curve.sha_file(curve.__file__), references={short(p): dict(zip(('final', 'curve'), map(str, reference_paths(p, results, pairs)))) for p in every})
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    moved = ev._set_aside(out, ('long_curve.csv', 'summary_s2.md', 'decision_s2.json'))
    result['previous_summary_moved_to'] = None if moved is None else str(moved)
    with open(out/'long_curve.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['arm', 'pair', 'short', 't', 'measure', 'value', 'commit', 'elapsed_seconds', 'visible_seconds', 'run_id'])
        writer.writeheader()
        writer.writerows(long_rows(runs, references, plan, pairs, short))
    curve.write_json(out/'decision_s2.json', result)
    (out/'summary_s2.md').write_text(summary_text(result))
    return result
