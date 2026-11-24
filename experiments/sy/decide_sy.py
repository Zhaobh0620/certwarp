"""S_Y's reading (runs/20261105-sy/任务书.md, plan.md, by code; a label for the lead and BZ, not a conclusion): analysis/decision_sy.json, summary_sy.md, long_curve.csv.

Delta = S_Y's final map - the same repaired field of Hx24 at 600 steps, not through the native level (fd2_v1's repair_metrics.json), pair by pair, the first 5 development pairs, seg35, unrounded:
  a run failed or a premise not met            不可判, the reasons written
  Delta >= +.005 and >= 4/5 positive           原生层还能加 seg35
  Delta <= +.002                               原生层加不了 seg35
  otherwise                                    中间带
Premises: all 5 pairs have class ok and ended ok in the queue (queue_done.json: the steps after the run all passed); the final map passes the certificate computed by the evaluation unit from the field (field_certificate.json), bound to the exported file and to the run's own record; the entry injected as recorded, the transfer at alpha = 1 at its first attempt on the repaired field, which is the one fd2_v1's record has; the hold is 47 s; both sides of Delta come from the same evaluate.py; the mean official SDlogJ <= .10; Hx24's 5 records are there, finite, certified at alpha = 1 by the evaluation, and give the pinned means.
The line is about seg35.  Everything else is described: seg4, the groups, HD95, the two SDlogJ, the spread of Delta, the certificates' lower bounds, S_Y - S_P and S_Y - S_X (other runs), S_X - Hx@2400 as the reference of the same question, the Dice along the native clock, the rejected trials, the time.
No number of the summary is written before the runs: each comes from the records (review of the tools, minor 10).
"""
import csv
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[0]/'s2'))
sys.path.insert(1, str(HERE.parents[0]/'tc'))

import numpy as np  # noqa: E402

import decide_s2 as ds2  # noqa: E402
import decide_tc as dt  # noqa: E402
import tc_curve as curve  # noqa: E402

SCHEMA = 'bayesreg.sy.decision.v1'
N = 5
ADD_BAR, NO_BAR, MIN_POSITIVE = .005, .002, 4
SDLOGJ_MAX = .10
HOLD = 47.
HX24_PIN = (.8617, .9043)                        # 5 pairs, seg35, seg4, to 4 decimals: runs/20261105-fd2/result.md
ARM, REFERENCE, S_X, HX24 = 'S_Y', ds2.REFERENCE, 'S_X', 'Hx24'
ADDS, NONE, MIDDLE, NOT_DECIDABLE = 'adds', 'none', 'middle', 'not_decidable'
READINGS = {ADDS: '原生层还能加 seg35', NONE: '原生层加不了 seg35', MIDDLE: '中间带', NOT_DECIDABLE: '不可判'}
KEY_NAMES = dict(ds2.KEY_NAMES, hd95='HD95')
MEAN_KEYS = ds2.MEAN_KEYS
DIFFERENCE_KEYS = ('seg35', 'seg4', 'cortex', 'white_matter', 'large_six', 'small_six', 'other_23', 'hd95', 'sdlogj_official', 'sdlogj_brain')
WINDOWS = ((0, 15), (15, 30), (30, 60), (60, 90), (90, 120), (120, 175), (175, None), (0, 160))       # seconds after the first trial, as the review of S_X; the last is a window every arm of these batches has
# FL result, reading 9: the same way of counting for every arm, an estimate; not run this way, no Atropos, a GPU in every case.
UNET_SECONDS, FL_START_SECONDS, FROZEN_START_SECONDS = 91., 1.5, 16.5
S_P_ENTRY_SECONDS = 60.                          # FL's reading 9 and decide_tc.REAL_SECONDS: S_P's seconds_total was measured beside a training (run_seed.py, ENTRY_CHARGE_SECONDS), so it is not the cost
REPAIR_WALL_SECONDS = 120.                       # FD2's repair (RepairConfig wall_seconds, repair.json config): a repair at it is cut, not done
HOLD_TOLERANCE = 1.                              # the measured hold (the wrapper's seconds) against 47 s: an anomaly beyond it
T_95, MDE_PER_SD = 2.7764451051977987, 1.66      # Student t, 4 degrees of freedom (n = 5); MDE about 1.66 sigma_d, S2's ratio (its result, .0047 / .0028)
# The code a reading depends on beyond decide_sy.py (review of the tools, important 2): batch.json pins it at the first start, a later start stops on a change, decide_code says it.
CODE_FILES = ('experiments/sy/run_sy.py', 'experiments/sy/sy_core.py', 'experiments/s2/run_s2.py', 'experiments/s2/s2_core.py', 'experiments/s2/decide_s2.py', 'experiments/tc/decide_tc.py', 'experiments/tc/tc_curve.py',
              'experiments/seed/run_seed.py', 'experiments/seed/inject_run.py', 'experiments/d0/evaluate.py', 'experiments/d0/classify_run.py', 'experiments/d0/check_prepared.py')
NOTE = ('开发集单次观测，读数不是结论，下一步不预先写，交 BZ。n = 5，前 5 对之前选配用过；Hx24 是从 4 个新臂里取的（同样 5 对上累计还试过 FL 的 19 个检查点），所以这是选择之后的量，有选择效应，5 对只有 6 个受试者，相邻对共用人，5/5 不是 5 次独立。'
        '预算收尾、上限收尾都不是 B 线收敛，这批每对的状态见"原生层跑了多久"一节。'
        '只换入口：路径改变不改目标函数，原生层仍是标量 D；这不是 F_B，进方法先改合同，由 BZ 定。.88 的线不作门。Δ 的两边是同一个修复场过不过原生层。'
        '入口是 FD2 的 120 s 墙版修复（撞墙的对数见时间一节）；RW 去掉墙之后的场可能不同，这个读数不是 RW 版入口的读数；它也只说 unet35_v1_ens 上色与通道下的 Hx24。'
        'S_P、S_X 都是另一时段，不重跑：S_P 终态 seed_v1，曲线 tc_v1；S_X 终态和曲线 s2_v1。')


def code_hashes():
    """{path: sha256} of CODE_FILES, now."""
    return {name: curve.sha_file(ROOT/name) for name in CODE_FILES}


def code_changed(pinned, now):
    """The paths whose sha256 differ between two code_hashes() records; None without a pinned one."""
    if pinned is None:
        return None
    return sorted(name for name in set(pinned) | set(now) if pinned.get(name) != now.get(name))


def load_runs(root, plan):
    """{(arm, pair): {metrics, curve, cleanup, seed_entry, field_certificate, queue_done, rejection, windows}} of what exists; the batch's own records only."""
    out = {}
    for arm, pair in plan:
        rec = load_run(Path(root)/'runs'/arm/pair)
        if rec:
            out[(arm, pair)] = rec
    return out


def load_run(run):
    rec = {name: curve.read_json(run/f'{name}.json') for name in ('curve', 'metrics', 'cleanup', 'seed_entry', 'field_certificate', 'queue_done') if (run/f'{name}.json').is_file()}
    if rec.get('metrics') is not None:
        rec['rejection'] = ds2.rejection_row(run)
        rec['windows'] = rejection_windows(run)
    return rec


def load_hx24(results, pairs, short):
    """{pair: {metrics (the evaluation's, of the repaired field), certificate, repaired_u_sha256, evaluator_sha256}}: fd2_v1, only read."""
    out = {}
    for pair in pairs:
        d = Path(results)/'fd2_v1/runs/RHx24_c600'/short(pair)
        if (d/'repair_metrics.json').is_file():
            doc = curve.read_json(d/'repair_metrics.json')
            out[pair] = dict(metrics=(doc.get('after') or {}).get('metrics'), certificate=(doc.get('after') or {}).get('certificate'), repaired_u_sha256=doc.get('repaired_u_sha256'), evaluator_sha256=(doc.get('evaluator') or {}).get('sha256'))
    return out


def load_hx2400(results, pairs, short):
    """{pair: the metrics of Hx at 2400 steps, repaired}: S_X's entry, fl_v1, only read: S_X - Hx is the brief's reference for S_Y - Hx24.  Only described: nothing is required of it."""
    out = {}
    for pair in pairs:
        d = Path(results)/'fl_v1/runs/RHx_c2400'/short(pair)
        if (d/'repair_metrics.json').is_file():
            out[pair] = (curve.read_json(d/'repair_metrics.json').get('after') or {}).get('metrics')
    return out


def load_s_x(results, pairs):
    """{pair: S_X's records}: s2_v1, only read."""
    return {pair: load_run(Path(results)/'s2_v1/runs/S_X'/pair) for pair in pairs}


def load_s_p(results, pairs):
    """{pair: S_P's records}: decide_s2's reader (seed_v1, tc_v1) plus the windows on seed_v1 and its entry's record.  Only read."""
    references = ds2.load_references(results, pairs, pairs)
    for pair in pairs:
        final, _ = ds2.reference_paths(pair, results, pairs)
        if (final/'protocol.json').is_file():
            references[pair]['windows'] = rejection_windows(final)
        if (final/'seed_entry.json').is_file():
            references[pair]['seed_entry'] = curve.read_json(final/'seed_entry.json')
    return references


def value_of(metrics, key, groups):
    """A reported value of one metrics dict: decide_s2's groups, and HD95."""
    if key == 'hd95':
        return (metrics or {}).get('hd95')
    return ds2.group_values(metrics, groups).get(key)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def paired(a, b, pairs, key, groups):
    """Pair by pair a - b of one key.  The mean only over all the pairs asked for, and only if every value is finite; otherwise None."""
    per_pair = {}
    for pair in pairs:
        x, y = value_of(a.get(pair), key, groups), value_of(b.get(pair), key, groups)
        if finite(x) and finite(y):
            per_pair[pair] = x-y
    values = list(per_pair.values())
    complete = bool(values) and len(values) == len(pairs)
    return dict(per_pair=per_pair, mean=float(np.mean(values)) if complete else None, positive=sum(v > 0 for v in values), n=len(values))


def spread(row):
    """sigma_d, the t interval of the mean and the MDE of a paired row of 5 pairs; None otherwise.  A description: neighbouring pairs share a subject, so the 5 differences are not independent."""
    values = list(row['per_pair'].values())
    if row.get('mean') is None or len(values) != N:
        return None
    sd = float(np.std(values, ddof=1))
    half = T_95*sd/math.sqrt(N)
    return dict(sd=sd, low=row['mean']-half, high=row['mean']+half, mde=MDE_PER_SD*sd)


def label_of(delta, positive):
    """The brief's table; the bars included, the mean unrounded."""
    if delta >= ADD_BAR and positive >= MIN_POSITIVE:
        return ADDS
    if delta <= NO_BAR:
        return NONE
    return MIDDLE


def run_premises(pair, rec, hx24, short):
    """Why this run cannot enter the reading (empty: it can)."""
    name = f'{ARM} {short}'
    metrics = (rec or {}).get('metrics')
    if metrics is None:
        return [f'{name}: 运行失败，没有 metrics.json']
    bad = []
    cls = (metrics.get('classification') or {}).get('class')
    if cls != 'ok':
        bad.append(f'{name}: 运行失败，分类 {cls}')
    done = (rec or {}).get('queue_done') or {}
    if done.get('state') != 'ok':                        # review of the tools, important 1: a STOP after the run (the curve, the cleanup) leaves no queue_done.json
        bad.append(f"{name}: 运行没有以 ok 结束（queue_done {done.get('state')}）")
    solver = metrics.get('certificate') or {}
    if not (solver.get('passed') and solver.get('state_hash_matches_u') is True and solver.get('final_npz_sha256_matches') is True):
        bad.append(f'{name}: 终态证书的记录未通过，或没有与导出场绑定')
    field = (rec or {}).get('field_certificate')
    if field is None:
        bad.append(f'{name}: 没有评测单元从场算的证书（field_certificate.json）')
    elif field.get('passed') is not True:
        bad.append(f"{name}: 评测单元从场算的证书未通过：{field.get('status')}，下界 {field.get('min_jd_lower_bound')}，不可行 {field.get('n_infeasible')}，未决 {field.get('n_uncertain')}")
    elif field.get('state_hash') != solver.get('state_hash') or field.get('file_sha256') != solver.get('final_npz_sha256'):
        bad.append(f'{name}: 评测单元算证书的场不是这次运行导出的场')
    seed = metrics.get('seed') or {}
    if seed.get('passed') is not True:
        bad.append(f"{name}: 入口核对未通过 {seed.get('problems')}")
    if seed.get('accepted_alpha') != 1.:
        bad.append(f"{name}: 转移 α = {seed.get('accepted_alpha')}，不是 1")
    attempts = seed.get('transfer_attempts') or []
    repaired = seed.get('repaired_u_sha256')
    if not attempts or repaired is None or attempts[0].get('state_hash') != repaired:
        bad.append(f"{name}: 转移的 state hash {attempts[0].get('state_hash') if attempts else None} 不等于修复场 {repaired}")
    recorded = (hx24 or {}).get('repaired_u_sha256')
    if recorded is None or recorded != repaired:
        bad.append(f'{name}: fd2_v1 记录的修复场 {recorded} 不等于这次运行的 {repaired}')
    if seed.get('charge_seconds') != HOLD:
        bad.append(f"{name}: 持时 {seed.get('charge_seconds')} s，不是 {HOLD:g} s")
    evaluator, theirs = (metrics.get('evaluator') or {}).get('sha256'), (hx24 or {}).get('evaluator_sha256')
    if evaluator is None or evaluator != theirs:            # minor 7: both sides of Delta from the same evaluate.py
        bad.append(f'{name}: 评测代码 {evaluator} 与 Hx24 的 {theirs} 不同，Δ 的两边不是同一套指标')
    return bad


def reference_premises(hx24, pairs, short, pin=HX24_PIN):
    bad, finals = [], []
    for pair in pairs:
        rec = hx24.get(pair)
        if rec is None or rec.get('metrics') is None:
            bad.append(f'Hx24 {short(pair)}: 没有 repair_metrics.json')
            continue
        ladder = rec.get('certificate') or {}
        if ladder.get('accepted_alpha') != 1. or ladder.get('status_at_1') != 'certified_feasible' or ladder.get('n_infeasible_at_1') != 0 or ladder.get('n_uncertain_at_1') != 0:
            bad.append(f'Hx24 {short(pair)}: 评测单元在 α = 1 的证书未通过')
        values = [rec['metrics'].get(key) for key in ('seg35', 'seg4')]
        if not all(finite(v) for v in values):               # minor 5: not a traceback
            bad.append(f'Hx24 {short(pair)}: seg35、seg4 不是有限数 {values}')
            continue
        finals.append(rec['metrics'])
    if not bad and pin is not None:
        got = (round(float(np.mean([x['seg35'] for x in finals])), 4), round(float(np.mean([x['seg4'] for x in finals])), 4))
        if got != pin:
            bad.append(f'Hx24 修复后的 5 对均值 {got}，不是钉住的 {pin}')
    return bad


def decide(runs, hx24, pairs, short, groups, *, pin=HX24_PIN):
    reasons = [r for pair in pairs for r in run_premises(pair, runs.get((ARM, pair)), hx24.get(pair), short(pair))]
    reasons += reference_premises(hx24, pairs, short, pin)
    ours = {p: ((runs.get((ARM, p)) or {}).get('metrics') or {}).get('metrics') for p in pairs}
    base = {p: (hx24.get(p) or {}).get('metrics') for p in pairs}
    delta, delta4 = paired(ours, base, pairs, 'seg35', groups), paired(ours, base, pairs, 'seg4', groups)
    values = [v.get('sdlogj_official') for v in ours.values() if v]
    sdlogj = float(np.mean(values)) if len(values) == len(pairs) and all(v is not None for v in values) else None
    if sdlogj is not None and not sdlogj <= SDLOGJ_MAX:              # written so that nan does not pass
        reasons.append(f'官方 SDlogJ 均值 {sdlogj:.4f} > {SDLOGJ_MAX}')
    if not reasons and (delta['mean'] is None or sdlogj is None):
        reasons.append(f"前提都满足，但只有 {delta['n']} 对有 seg35，或缺官方 SDlogJ，Δ 不可算")
    code = NOT_DECIDABLE if reasons else label_of(delta['mean'], delta['positive'])
    if reasons:
        text = READINGS[code]+'：'+'；'.join(reasons)
    else:
        text = f"{READINGS[code]}：Δ(seg35) = S_Y 终态 − Hx24 修复后 = {delta['mean']:+.4f}，{delta['positive']}/{N} 为正；Δ(seg4) {delta4['mean']:+.4f}（{delta4['positive']}/{N}），官方 SDlogJ 均值 {sdlogj:.4f}，都只描述，不进判定"
    return dict(reading_code=code, reading=text, reasons=reasons, delta_seg35=delta, delta_seg4=delta4, mean_sdlogj_official=sdlogj)


# ---- the descriptions ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
mean_or_none = ds2.mean_or_none


def rejection_windows(run):
    """Trials and Jacobian rejections of the native level per window of seconds after its first trial, from inner_attempts.json (it stays after the cleanup): {window: [trials, rejected]}, None without the file.  The whole-run share depends on how long the native level ran: this is the comparison the review of S_X asked for."""
    task = curve.task_dir(run)
    if not (task/'inner_attempts.json').is_file():
        return None
    rows = [r for r in curve.read_json(task/'inner_attempts.json') if r.get('level') == curve.NATIVE_LEVEL and r.get('visible_seconds') is not None]     # visible_seconds, the task clock: this file's elapsed_seconds starts again at each outer step
    start = min((r['visible_seconds'] for r in rows), default=0.)
    out = {}
    for a, b in WINDOWS:
        inside = [r for r in rows if r['visible_seconds']-start >= a and (b is None or r['visible_seconds']-start < b)]
        out[window_name(a, b)] = [len(inside), sum(r.get('reason') == 'JACOBIAN_VIOLATION' for r in inside)]
    return out


def window_name(a, b):
    return f'{a}–{b} s' if b is not None else f'{a} s 之后'


def pooled_windows(records, pairs):
    """{window: rejected / trials, over the pairs together}, or None if a pair has no record."""
    rows = [(records.get(p) or {}).get('windows') for p in pairs]
    if not all(rows):
        return None
    return {window_name(a, b): (lambda t, r: None if not t else r/t)(sum(row[window_name(a, b)][0] for row in rows), sum(row[window_name(a, b)][1] for row in rows)) for a, b in WINDOWS}


def entry_clock(entry):
    """What the entry really took on this pair: seconds of the field, the repair, its warm-up (FD2's task clock; a repair that hit its wall counts what it used).  None without a record."""
    seconds = [(entry or {}).get(key) for key in ('source_seconds_total', 'seconds_repair', 'seconds_warmup')]
    if seconds[0] is None:
        return None
    return sum(s or 0. for s in seconds)


def time_row(name, rec, pairs, *, startup, entry_seconds=None):
    """The estimate of FL's reading 9 for one arm through the native level, pair by pair: the real entry (entry_seconds if given, else its record) + the task clock with the hold taken out + the U-Net + start and exit."""
    rows = {}
    for pair in pairs:
        r = rec(pair) or {}
        entry, metrics = r.get('seed_entry'), r.get('metrics') or {}
        clock, e2e, hold = entry_clock(entry) if entry_seconds is None else entry_seconds, metrics.get('e2e_seconds'), (entry or {}).get('charge_seconds')
        if clock is not None and e2e is not None and hold is not None:
            rows[pair] = dict(entry=clock, native=e2e-hold, total=clock+e2e-hold+UNET_SECONDS+startup, hold=hold)
    return dict(name=name, n=len(rows), entry=mean_or_none(r['entry'] for r in rows.values()), entry_max=max((r['entry'] for r in rows.values()), default=None), native=mean_or_none(r['native'] for r in rows.values()), total=mean_or_none(r['total'] for r in rows.values()), hold=mean_or_none(r['hold'] for r in rows.values()))


def time_account(runs, s_x, s_p, pairs):
    """〔估算〕 FL's reading 9, one way of counting for all.  Hx24 alone: the same entry, no native level.  wall: how many of S_Y's entries were repairs cut at FD2's wall."""
    entries = [(runs.get((ARM, p)) or {}).get('seed_entry') for p in pairs]
    clocks = [entry_clock(entry) for entry in entries]
    clocks = [clock for clock in clocks if clock is not None]
    repairs = [(entry or {}).get('seconds_repair') for entry in entries]
    return dict(s_y=time_row('S_Y', lambda p: runs.get((ARM, p)), pairs, startup=FL_START_SECONDS+FROZEN_START_SECONDS), s_x=time_row('S_X', lambda p: s_x.get(p), pairs, startup=FL_START_SECONDS+FROZEN_START_SECONDS), s_p=time_row('S_P', lambda p: s_p.get(p), pairs, startup=FROZEN_START_SECONDS, entry_seconds=S_P_ENTRY_SECONDS),
                hx24=dict(name='Hx24 修复后，不过原生层', n=len(clocks), entry=mean_or_none(clocks), entry_max=max(clocks, default=None), native=None, total=None if not clocks else mean_or_none(clocks)+UNET_SECONDS+FL_START_SECONDS, hold=None), wall=sum(finite(s) and s >= REPAIR_WALL_SECONDS for s in repairs), n_entries=sum(entry is not None for entry in entries),
                unet_seconds=UNET_SECONDS, fl_start_seconds=FL_START_SECONDS, frozen_start_seconds=FROZEN_START_SECONDS, s_p_entry_seconds=S_P_ENTRY_SECONDS, wall_seconds=REPAIR_WALL_SECONDS)


def final_elapsed(rec):
    return (((rec or {}).get('curve') or {}).get('final') or {}).get('elapsed_seconds')


def first_bound(rec):
    points = ((rec or {}).get('curve') or {}).get('points') or []
    return points[0].get('min_jd_lower_bound') if points and points[0].get('t') == 0 else None


def native_table(runs, s_x, s_p, pairs):
    """How long the native level ran, pair by pair: the last commit's seconds on its own clock, the commits, the status, GF against its threshold; the certified lower bounds of the first commit and of the final map (S_Y's final: the evaluation's, from the field)."""
    rows = []
    for pair in pairs:
        rec = runs.get((ARM, pair)) or {}
        metrics = rec.get('metrics') or {}
        rows.append(dict(pair=pair, elapsed=final_elapsed(rec), commits=metrics.get('native_commits'), status=metrics.get('status'), GF=metrics.get('GF'), GF_threshold=metrics.get('GF_threshold'), s_x_elapsed=final_elapsed(s_x.get(pair)), s_p_elapsed=final_elapsed(s_p.get(pair)),
                         first_bound=first_bound(rec), final_bound=(rec.get('field_certificate') or {}).get('min_jd_lower_bound'), s_x_first_bound=first_bound(s_x.get(pair)), s_x_final_bound=(((s_x.get(pair) or {}).get('curve') or {}).get('final') or {}).get('min_jd_lower_bound'), s_p_first_bound=first_bound(s_p.get(pair))))
    return rows


def geometric_mean(values):
    values = [v for v in values if finite(v) and v > 0]
    return float(math.exp(np.mean(np.log(values)))) if values else None


def against_entry_table(records, base, pairs, groups):
    """S_Y(t) - Hx24 repaired, pair by pair, at each native time: what the native level has done by t."""
    rows = []
    for t in curve.TIMES+(curve.FINAL,):
        row = dict(t=t)
        for key in MEAN_KEYS:
            values = []
            for pair in pairs:
                x, y = dt.value_at(records.get(pair), t, key), value_of(base.get(pair), key, groups)
                if x is not None and y is not None:
                    values.append(x-y)
            row[key] = dict(mean=float(np.mean(values)) if values else None, positive=sum(v > 0 for v in values), n=len(values))
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
            out.append(dict(name, what=f"没有 metrics.json（{(rec.get('queue_done') or {}).get('state')}）"))
            continue
        if metrics.get('status') != 'native_budget_exhausted':
            out.append(dict(name, what=f"状态 {metrics.get('status')}（不是预算收尾）"))
        value = (metrics.get('metrics') or {}).get('sdlogj_official')
        if value is not None and value > SDLOGJ_MAX:
            out.append(dict(name, what=f'终态官方 SDlogJ {value:.4f} > {SDLOGJ_MAX}'))
        if (metrics.get('self_check') or {}).get('passed') is not True:         # minor 4
            out.append(dict(name, what=f"evaluate.py 的自检（对冻结路径的 labels.json、smoothness.json）没有通过：{(metrics.get('self_check') or {}).get('diffs')}"))
        if metrics.get('run_labels_equal_truth') is not True:
            out.append(dict(name, what='运行自己的标签副本不等于真值（run_labels_equal_truth）'))
        if metrics.get('arm') != arm or metrics.get('pair') != pair:
            out.append(dict(name, what=f"metrics.json 记的是 {metrics.get('arm')} {metrics.get('pair')}"))
        injected = (metrics.get('seed') or {}).get('injected_seconds')
        if finite(injected) and abs(injected-HOLD) > HOLD_TOLERANCE:         # minor 3: the hold the wrapper measured
            out.append(dict(name, what=f'实测持时 {injected:.2f} s，与 {HOLD:g} s 差超过 {HOLD_TOLERANCE:g} s'))
        field = rec.get('field_certificate') or {}
        if field and field.get('lower_bound_as_the_solver') is False:
            out.append(dict(name, what=f"评测单元的下界 {field.get('min_jd_lower_bound')} 与求解器记录的 {field.get('solver_lower_bound')} 不逐位相同"))
        if field and (field.get('same_grid_as_the_solver') is False or field.get('same_version_as_the_solver') is False):         # minor 8
            out.append(dict(name, what=f"评测单元证书的网格或版本与求解器记录不同：grid {field.get('grid_id')} / {field.get('solver_grid_id')}，版本 {field.get('certificate_version')} / {field.get('solver_certificate_version')}"))
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


def reference_certificates(root):
    """The certificates of S_P's and S_X's final maps, computed from the fields before the first run (run_sy.py taper-check, its latest record by the UTC in its name): {arm: passed of n}.  The run's own record is not read here."""
    found = sorted(Path(root).glob('taper_check_*.json'))
    rows = (curve.read_json(found[-1]).get('references') or []) if found else []
    return {arm: dict(n=sum(r['arm'] == arm for r in rows), passed=sum(r['arm'] == arm and r.get('passed') is True for r in rows), file=found[-1].name if found else None) for arm in (REFERENCE, S_X)}


def describe(runs, hx24, hx2400, s_x, s_p, pairs, plan, short, groups, root):
    ours = {p: ((runs.get((ARM, p)) or {}).get('metrics') or {}).get('metrics') for p in pairs}
    base = {p: (hx24.get(p) or {}).get('metrics') for p in pairs}
    ref_p = {p: ((s_p.get(p) or {}).get('metrics') or {}).get('metrics') for p in pairs}
    ref_x = {p: ((s_x.get(p) or {}).get('metrics') or {}).get('metrics') for p in pairs}
    s_p_alpha = {p: (((s_p.get(p) or {}).get('metrics') or {}).get('seed') or {}).get('accepted_alpha') for p in pairs}
    alpha_one = tuple(p for p in pairs if s_p_alpha[p] == 1.)

    def differences(a, b, keep=pairs):
        return {key: paired(a, b, keep, key, groups) for key in DIFFERENCE_KEYS}
    records = {p: runs[(ARM, p)] for p in pairs if (runs.get((ARM, p)) or {}).get('curve')}
    x_records = {p: s_x[p] for p in pairs if (s_x.get(p) or {}).get('curve')}
    p_records = {p: s_p[p] for p in pairs if (s_p.get(p) or {}).get('curve')}
    rejection = [(runs.get((ARM, p)) or {}).get('rejection') for p in pairs]
    x_rejection = [(s_x.get(p) or {}).get('rejection') for p in pairs]
    p_rejection = [(s_p.get(p) or {}).get('rejection') for p in pairs]
    certified = sum(((runs.get((ARM, p)) or {}).get('field_certificate') or {}).get('passed') is True for p in pairs)
    hx24_certified = sum(((hx24.get(p) or {}).get('certificate') or {}).get('status_at_1') == 'certified_feasible' for p in pairs)
    means = {name: {key: mean_or_none(value_of(values.get(p), key, groups) for p in pairs) for key in DIFFERENCE_KEYS} for name, values in ((ARM, ours), (HX24, base), (REFERENCE, ref_p), (S_X, ref_x))}
    against_hx24 = differences(ours, base)
    native = native_table(runs, s_x, s_p, pairs)
    return dict(means=means, against_hx24=against_hx24, spread=dict(seg35=spread(against_hx24['seg35']), seg4=spread(against_hx24['seg4'])), against_s_p=differences(ours, ref_p), s_p_alpha={short(p): a for p, a in s_p_alpha.items()}, against_s_p_alpha_one=differences(ours, ref_p, alpha_one), against_s_x=differences(ours, ref_x), s_x_against_its_entry=differences(ref_x, hx2400),
                certificates=dict(s_y=certified, hx24=hx24_certified, **reference_certificates(root)),
                tables={ARM: ds2.arm_table(records, pairs), S_X: ds2.arm_table(x_records, pairs), REFERENCE: ds2.arm_table(p_records, pairs)}, against_entry=against_entry_table(records, base, pairs, groups), s_x_against_its_entry_by_t=against_entry_table(x_records, hx2400, pairs, groups), same_t_s_x=ds2.same_t_table(records, x_records, pairs), same_t_s_p=ds2.same_t_table(records, p_records, pairs),
                rejection=dict(s_y=rejection_means(rejection), s_x=rejection_means(x_rejection), s_p=rejection_means(p_rejection), windows=dict(s_y=pooled_windows({p: runs.get((ARM, p)) for p in pairs}, pairs), s_x=pooled_windows(s_x, pairs), s_p=pooled_windows(s_p, pairs))),
                native=native, bounds=dict(s_y_first=geometric_mean(r['first_bound'] for r in native), s_y_final=geometric_mean(r['final_bound'] for r in native), s_x_first=geometric_mean(r['s_x_first_bound'] for r in native), s_x_final=geometric_mean(r['s_x_final_bound'] for r in native), s_p_first=geometric_mean(r['s_p_first_bound'] for r in native)),
                time=time_account(runs, s_x, s_p, pairs), anomalies=anomalies(runs, plan, short))


def rejection_means(rows):
    keep = [r for r in rows if r]
    return {key: mean_or_none(row.get(key) for row in keep) for key in ('attempts', 'jacobian_rejected', 'rejected_share', 'other_rejected_total', 'native_commits', 'accepted_steps', 'seconds_per_accepted_step', 'native_elapsed_last')} | {'n': len(keep)}


# ---- the text ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
f, signed_paired, sign4 = dt.f, ds2.signed_paired, ds2.sign4


def e(value):
    return '—' if value is None else f'{value:.2g}'


def hold_note(d):
    """What the hold is not, with this batch's numbers (from the entries' records and the S_X runs), none written before the runs."""
    t = d['time']
    entry, entry_max, x = t['hx24']['entry'], t['hx24']['entry_max'], t['s_x']
    numbers = '' if entry is None else f"（这批 {t['hx24']['n']} 对平均 {entry:.1f} s，最长 {entry_max:.1f} s；其中 {t['wall']} 对的修复撞了 {t['wall_seconds']:g} s 的墙，是截断值）。所以这里的任务时钟（visible_seconds）比真实时间平均少记约 {entry-HOLD:.0f} s"
    s_x = '' if x['hold'] is None else f"S_X 的持时 {x['hold']:g} s，原生部分平均 {x['native']:.0f} s（S_Y 的见时间一节）。"
    return f"**持时 {HOLD:g} s，与 S_P 相同，这不是入口的成本**：入口实际任务时钟见时间一节{numbers}；同一个 t 不是同样的总时间。{s_x}比的是原生层能加多少，不是谁快。"


def clock_note(d):
    """The clock of the curves; which rows of S_X are its final map, from its own records (S_X review, minor 8)."""
    x = mean_or_none(row['s_x_elapsed'] for row in d['native'])
    past = [t for t in curve.TIMES if x is not None and t > x]
    s_x = '' if not past else f"S_X 的原生层最后一次提交平均在 {x:.0f} s，它在 t = {'、'.join(str(t) for t in past)} 这几行取的是自己的终态（S_X 复核次要 8）。"
    return f'**t 是原生层时钟**：从原生层 solve_sequential 开始算，不含入口、持时、转移证书。{s_x}有的对早期还没有提交，那一行 n 小于 5，均值不能直接和别的行比。'


def difference_lines(title, differences):
    lines = ['', title, '', '| 项 | 差（均值，为正的对数/对数） |', '|---|---|']
    lines += [f"| {KEY_NAMES[key]} | {signed_paired(differences[key])} |" for key in DIFFERENCE_KEYS]
    return lines


def spread_line(name, row):
    return f"{name}：σ_d —，t 区间 —（不到 5 对）" if row is None else f"{name}：σ_d {row['sd']:.4f}，95% t 区间 [{row['low']:+.4f}, {row['high']:+.4f}]，MDE ≈ {MDE_PER_SD:g} σ_d = {row['mde']:.4f}"


def time_line(row):
    return f"| {row['name']} | {row['n']} | {f(row['entry'], 1)} / {f(row['entry_max'], 1)} | {f(row['hold'], 0)} | {f(row['native'], 1)} | {f(row['total'], 0)} |"


def summary_text(result):
    d = result['description']
    d35, d4 = result['delta_seg35'], result['delta_seg4']
    code = result['decide_code']
    warning = ['', f"**判定代码或它依赖的代码与队列第一次启动时不同**：decide_sy.py batch.json {code['at_first_start']}，现在 {code['now']}；其余变了的文件 {code['code_changed']}。线是预登记的，下面的读数要对照 plan.md 的修订读。"] if code['same'] is False else []
    c = d['certificates']
    alpha = [f'{p}（{a}）' for p, a in d['s_p_alpha'].items() if a != 1.]
    lines = ['# S_Y 汇总（代码写出，读数不是结论）', '', result['note'], '', result['hold_note'], *warning, '', f"**读数**：{result['reading']}", '', f"结果根目录 {result['root']}；写出 {result['written_utc']}。", '',
             '线（含等号，均值不舍入）：Δ seg35 ≥ +.005 且 ≥ 4/5 为正，原生层还能加 seg35；Δ seg35 ≤ +.002，原生层加不了 seg35；其余中间带；运行失败或前提不满足，不可判。', '',
             '## 判定：Δ = S_Y 终态 − Hx24 修复后（同一个修复场，不过原生层；fd2_v1）', '', '| 对 | Δ seg35 | Δ seg4 |', '|---|---|---|']
    for pair in result['pairs']:
        a, b = d35['per_pair'].get(pair), d4['per_pair'].get(pair)
        lines.append(f"| {result['short'][pair]} | {sign4(a)} | {sign4(b)} |")
    lines += ['', f"均值 Δ seg35 {sign4(d35['mean'])}（{d35['positive']}/{d35['n']} 为正），Δ seg4 {sign4(d4['mean'])}（{d4['positive']}/{d4['n']} 为正），官方 SDlogJ 均值 {f(result['mean_sdlogj_official'])}。", '',
              '离散度（只描述，不是检验；相邻对共用受试者）：'+spread_line('seg35', d['spread']['seg35'])+'；'+spread_line('seg4', d['spread']['seg4'])+'。MDE 大于 +.005 时，"还能加"那一行只是筛查。']
    lines += ['', '## 只描述：主表', '', '| 臂（前 5 对） | '+' | '.join(KEY_NAMES[k] for k in DIFFERENCE_KEYS)+' |', '|---|'+'---|'*len(DIFFERENCE_KEYS)]
    for name, title in ((ARM, 'S_Y 终态'), (HX24, 'Hx24 修复后（不过原生层）'), (S_X, 'S_X 终态（s2_v1，另一时段）'), (REFERENCE, 'S_P 终态（seed_v1，另一时段）')):
        lines.append(f'| {title} | '+' | '.join(f(d['means'][name][k], 3 if k == 'hd95' else 4) for k in DIFFERENCE_KEYS)+' |')
    lines += ['', f"过证书（都是评测单元从场算的，α = 1，配置 C）：S_Y {c['s_y']}/{N}；Hx24 修复后 {c['hx24']}/{N}（repair_metrics.json）；S_P 终态 {c[REFERENCE]['passed']}/{c[REFERENCE]['n']}，S_X 终态 {c[S_X]['passed']}/{c[S_X]['n']}（队列开始时重算，{c[REFERENCE]['file']}）。"]
    lines += difference_lines('## 只描述：S_Y 终态 − Hx24 修复后，各项', d['against_hx24'])
    lines += difference_lines('## 只描述（参照）：S_X 终态 − Hx@2400 修复后（s2_v1、fl_v1，另一时段；因子 2；原生层长度见"原生层跑了多久"一节）', d['s_x_against_its_entry'])
    lines += difference_lines('## 只描述：S_Y 终态 − S_P 终态（seed_v1，另一时段）', d['against_s_p'])
    lines += difference_lines(f"## 只描述：同上，只留 S_P 转移 α = 1 的对（去掉 {'、'.join(alpha) or '没有'}），n = {d['against_s_p_alpha_one']['seg35']['n']}", d['against_s_p_alpha_one'])
    lines += difference_lines('## 只描述：S_Y 终态 − S_X 终态（s2_v1，另一时段；S_X 的入口是因子 2 的 Hx@2400，持时与原生层长度见下面两节）', d['against_s_x'])
    lines += ['', '## 只描述：原生层跑了多久，证书下界', '', '下界到正就停，不是 J 的真实最小值。S_Y 终态的下界是评测单元从场算的；其余来自曲线（求解器的记录）。', '', '| 对 | 最后一次提交的秒数（原生层时钟） | 提交数 | 状态 | GF / 阈值 | 第一次提交下界 | 终态下界 | S_X 同一项（秒数、第一次、终态） | S_P（tc_v1）秒数、第一次 |', '|---|---|---|---|---|---|---|---|---|']
    for row in d['native']:
        lines.append(f"| {result['short'][row['pair']]} | {f(row['elapsed'], 1)} | {row['commits']} | {row['status']} | {f(row['GF'], 3)} / {e(row['GF_threshold'])} | {e(row['first_bound'])} | {e(row['final_bound'])} | {f(row['s_x_elapsed'], 1)}、{e(row['s_x_first_bound'])}、{e(row['s_x_final_bound'])} | {f(row['s_p_elapsed'], 1)}、{e(row['s_p_first_bound'])} |")
    b = d['bounds']
    lines += ['', f"几何均值：S_Y 第一次提交 {e(b['s_y_first'])}、终态 {e(b['s_y_final'])}；S_X {e(b['s_x_first'])}、{e(b['s_x_final'])}；S_P（tc_v1）第一次提交 {e(b['s_p_first'])}。"]
    lines += ['', '## 只描述：Dice 随原生层时刻', '', result['clock_note']]
    lines += ds2.curve_lines('### S_Y 各时刻的均值', d['tables'][ARM])
    lines += ds2.same_t_lines('### S_Y(t) − Hx24 修复后（入口不变，只看原生层到 t 做了什么）', [dict(row, task_clock_a=None, task_clock_b=None) for row in d['against_entry']])
    lines += ds2.same_t_lines('### 参照：S_X(t) − Hx@2400 修复后（s2_v1、fl_v1，另一时段）', [dict(row, task_clock_a=None, task_clock_b=None) for row in d['s_x_against_its_entry_by_t']])
    lines += ds2.curve_lines('### S_X 各时刻的均值（s2_v1，另一时段）', d['tables'][S_X])
    lines += ds2.same_t_lines('### S_Y(t) − S_X(t)，同 t（另一时段）', d['same_t_s_x'])
    lines += ds2.curve_lines('### S_P 各时刻的均值（tc_v1，另一时段）', d['tables'][REFERENCE])
    lines += ds2.same_t_lines('### S_Y(t) − S_P(t)，同 t（S_P 是 tc_v1，另一时段；最后一行是 tc_v1 的终态）', d['same_t_s_p'])
    r = d['rejection']
    lines += ['', '## 只描述：接受与被拒', '', '被拒比例 = JACOBIAN_VIOLATION / 试探（summary.json 的 costs.levels[3]）。**全程比例依赖原生层跑了多久**（被拒比例随时间下降；各臂原生层长度见上表），比入口要看下面按窗口的表（S_X 复核阻断 1）。其他拒绝单列。', '', '| 臂 | n | 试探 | 被拒（Jacobian） | 被拒比例 | 其他拒绝 | 提交数 | 接受步 | 每接受步秒数 |', '|---|---|---|---|---|---|---|---|---|', ds2.rejection_line('S_Y', r['s_y']), ds2.rejection_line('S_X', r['s_x']), ds2.rejection_line('S_P（seed_v1）', r['s_p'])]
    lines += ['', '按首次试探后的秒数（inner_attempts.json，5 对合计）：', '', '| 窗口 | S_Y | S_X | S_P（seed_v1） |', '|---|---|---|---|']
    for a, b_ in WINDOWS:
        name = window_name(a, b_)
        lines.append(f"| {name} | {f((r['windows']['s_y'] or {}).get(name), 3)} | {f((r['windows']['s_x'] or {}).get(name), 3)} | {f((r['windows']['s_p'] or {}).get(name), 3)} |")
    t = d['time']
    lines += ['', '## 只描述：时间〔估算，没有这样跑过〕', '', f"FL 读法 9 的口径：入口的真实任务时钟（场 + 修复 + 预热）+ 原生部分（e2e 去掉持时）+ U-Net 约 {t['unet_seconds']:g} s + 启动退出（FL 约 {t['fl_start_seconds']:g} s，冻结求解器约 {t['frozen_start_seconds']:g} s）。都不含 Atropos，都要 GPU。S_Y 的 {t['n_entries']} 个入口里 {t['wall']} 个修复撞了 {t['wall_seconds']:g} s 的墙，是截断值，不代表修好要多久。S_P 的入口按约 {t['s_p_entry_seconds']:g} s 算（它自己的 seconds_total 是与训练同跑时量的）。", '', '| 臂 | n | 入口真实任务时钟 平均 / 最长 s | 持时 s | 原生部分 s | 总计约 s |', '|---|---|---|---|---|---|', time_line(t['s_y']), time_line(t['hx24']), time_line(t['s_x']), time_line(t['s_p'])]
    lines += ['', '## 异常', '']
    lines += [f"- {a['arm']} {a['pair']}: {a['what']}" for a in d['anomalies']] or ['没有。']
    lines += ['', '## 它证伪不了什么', '', '路径改变不改目标函数，原生层仍是标量 D；开发集单次，n = 5，前 5 对选配用过，Hx24 是选出来的；这不是 F_B，进方法先改合同。S_Y 只说这个入口（FD2 的 120 s 墙版修复，unet35_v1_ens）、47 s 持时下的原生层（跑了多久见上表）能加多少，不说入口要多久，不说 RW 版的入口，也不说其余 14 对（那 14 对要新的预登记，交 BZ）。']
    return '\n'.join(lines)+'\n'


def long_rows(runs, s_x, s_p, plan, pairs, short):
    """One row per arm, pair, time and measure; t final is the final map.  S_X is s2_v1's, S_P tc_v1's, the arm named so."""
    records = [(arm, pair, runs.get((arm, pair)) or {}) for arm, pair in plan]+[('S_X_s2_v1', pair, s_x.get(pair) or {}) for pair in pairs]+[('S_P_tc_v1', pair, s_p.get(pair) or {}) for pair in pairs]
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


def write_analysis(root, plan, pairs, short, *, ev, utc, results, pin=HX24_PIN):
    """Reads the runs of this batch and the references, writes the three files under root/analysis; an earlier summary goes to analysis/previous_<UTC>/.  Returns the decision."""
    root = Path(root)
    runs = load_runs(root, plan)
    hx24, hx2400, s_x, s_p = load_hx24(results, pairs, short), load_hx2400(results, pairs, short), load_s_x(results, pairs), load_s_p(results, pairs)
    groups = ev.A5_GROUPS
    result = dict(schema=SCHEMA, root=str(root), written_utc=utc(), note=NOTE, rule='runs/20261105-sy/plan.md', reading_arm=ARM, pairs=list(pairs), short={p: short(p) for p in pairs}, **decide(runs, hx24, pairs, short, groups, pin=pin))
    result['description'] = describe(runs, hx24, hx2400, s_x, s_p, pairs, plan, short, groups, root)
    result['hold_note'], result['clock_note'] = hold_note(result['description']), clock_note(result['description'])
    now, code_now = curve.sha_file(__file__), code_hashes()
    batch = curve.read_json(root/'batch.json') if (root/'batch.json').is_file() else {}
    pinned, changed = batch.get('decide_sha256'), code_changed(batch.get('code_sha256'), code_now)
    result['decide_code'] = dict(at_first_start=pinned, now=now, code_changed=changed, same=None if pinned is None else pinned == now and not changed)
    result['sources'] = dict(evaluator_sha256=curve.sha_file(ev.__file__), code_sha256=now, code=code_now, curve_code_sha256=curve.sha_file(curve.__file__), hx24=str(Path(results)/'fd2_v1/runs/RHx24_c600'), s_x=str(Path(results)/'s2_v1/runs/S_X'), s_p_final=str(Path(results)/'seed_v1/runs/S_P'), s_p_curve=str(Path(results)/'tc_v1/runs/S_P'))
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    moved = ev._set_aside(out, ('long_curve.csv', 'summary_sy.md', 'decision_sy.json'))
    result['previous_summary_moved_to'] = None if moved is None else str(moved)
    with open(out/'long_curve.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['arm', 'pair', 'short', 't', 'measure', 'value', 'commit', 'elapsed_seconds', 'visible_seconds', 'run_id'])
        writer.writeheader()
        writer.writerows(long_rows(runs, s_x, s_p, plan, pairs, short))
    curve.write_json(out/'decision_sy.json', result)
    (out/'summary_sy.md').write_text(summary_text(result))
    return result
