"""HS's reading (runs/20261106-hs/任务书.md, plan.md; by code; a label for the lead and BZ, not a conclusion).  The first of the three selection readings (D71, D72).

What the method outputs for a pair (C22.4, C22.5, D74 and D75 as BZ settled them on 11-05, "都按建议"):
    hx_certified            the field is done, its repair passed the deciding certificate (repair.json, the repair's own process, no label), the evaluation unit's review certificate agrees; a fuse hit then a pass is a pass (D74-3, (c-ii)), counted
    hx_fallback_configC     the field not done (an entry failure: a run stop, nonfinite, the alignment QC; D74-2), or the repair crashed (no repair.json) or did not pass (D74-1): configuration C's field of d0_v1 (A0), when d0_v1's records qualify it (rp_core.fallback_status) and the fallback unit
                            certified it again in a process that reads no label (C22.5-2)
    hx_fallback_identity    the same, when that field does not qualify or its certificate does not pass: u = 0 (D75)
    hx_provenance_rejected  the channels refused as an input error (the structure layer, the firewall, the admission; D74-4): no output, the run refused
    a review certificate that disagrees with the deciding one: the fallback unit's third certificate decides, mechanically (C22.3-9 (d)); no metric is looked at
Not read (says nothing about the segmenter; the reading is 不可判): a cell not run (a STOP before it, an arm stopped), a field done but not evaluated, a repair not run or not evaluated, a fallback not decided or the identity not evaluated, a third certificate missing.

The reading, on the 19 pairs, method level (the fallback pairs in):  Delta = HS - RW (rw_v1, the reference Hx24 with v1's channels), seg35, per pair.
    采用新分割器   mean Delta >= 0, the subject-block 95 % bootstrap lower bound > -.003, all 19 pairs with an output (the certificate or a fallback), the mean official SDlogJ of HS <= .10
    不采用         otherwise
    不可判         a pair not read; a pair refused (hx_provenance_rejected); RW not 19/19 certified by both of its certificates; a metric undefined
The subject-block bootstrap: runs/20261105-fd2/recompute.py block (10000 draws, seed 20261105): the 20 subjects resampled, pair i = (subject i, i + 1) weighs the product of their counts.
Reported with it, not read: seg4, the groups, the cortex, the white matter, HD95, the SDlogJ (official, in the brain), the task clock, the repair rounds, the fuse hits, the fallbacks (each kind, which pairs, why), the distance to .88, and the 〔模型〕 expectation written before the run (prediction_hs.json).
"""
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from decide_ef import read_json, set_aside
import decide_fx as fx_reading
import fl_core as fl
import fx_core as fx
import hs_core as hc
import mc0_core as core
import rp_core as rp
from hs_core import ALL_PAIRS, HS

records_hs = fx_reading._records(HS)
MEASURES = ('seg35', 'seg4', 'hd95', 'sdlogj_official', 'sdlogj_brain', 'small_six', 'large_six', 'other_23', 'cortex', 'white_matter')
STEP = hc.NATIVE
NOT_RUN, NOT_READ = 'not_run', 'not_read'
mean = fx_reading.mean
_f, _signed = fx_reading._f, fx_reading._signed
NOTE = ('开发集单次观测，19 对；通道来自已登记的候选分割器的预测（oracle: false），真值标签只进评测。冻结的 Hx24（Hx24-RF05-F1200）只换了分割器 id。**不是 F_B，没有经认证的下降，只有终态过证书；不进结果表，进方法先改合同（D72），由 BZ 定。** '
        '这是 3 次选择读数的第 1 次（D71）。候选是在同样 20 名开发受试者的分割级读数上选的（UF 的主臂取 (a)、(b) 较高者），这 19 对之前也选过这条线的配置，选择效应在内。方法级均值含退路（失败的对按退路计入）；只算本线输出的对的均值有生存者偏差，单列。')


def block_interval(values, index, *, n=hc.BOOT_N, seed=hc.BOOT_SEED, n_subjects=hc.N_SUBJECTS):
    """(lo, hi): the 95 % subject-block bootstrap interval of the mean over pairs (runs/20261105-fd2/recompute.py block; experiments/p1/p1_core.block_interval): the 20 subjects resampled, pair i = (subject i, i + 1) weighs the product of their counts.  index: each value's pair number 0..18."""
    x, index = np.asarray(values, dtype=np.float64), np.asarray(index)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        counts = np.bincount(rng.integers(0, n_subjects, n_subjects), minlength=n_subjects)
        w = counts[index]*counts[index+1]
        if w.sum():
            out.append(float((w*x).sum()/w.sum()))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


# ---- what a pair is ------------------------------------------------------------------------------------------------------------------------------
def field_row(root, pair):
    return records_hs.field_rows(root, HS.field_cell(pair)).get(STEP)


def repair_record(root, pair):
    return read_json(HS.run_dir(root, HS.repair_cell(pair))/'repair.json')


def repair_row(root, pair):
    cell = HS.repair_cell(pair)
    return fx_reading.with_wall(records_hs.repair_row(root, cell), HS.run_dir(root, cell))


def refusal(root, pair):
    return read_json(HS.run_dir(root, HS.field_cell(pair))/hc.REFUSAL_FILE)


def classify(root, pair):
    """(kind, why) of a pair before any fallback: certified, fallback (an entry failure or a repair failure), disagree (the two certificates), rejected, not_run, not_read.  The fallback unit and the reading call this same function."""
    short = core.short_name(pair)
    d = HS.run_dir(root, HS.field_cell(pair))
    if not (d/'queue_done.json').is_file() and not (d/'run.json').is_file():
        return NOT_RUN, f'{short} 场没有运行'
    refused = refusal(root, pair)
    if refused is not None and refused.get('kind') == 'provenance_rejected':
        return 'rejected', f"{short} 通道被拒（输入错误，D74-4）：{refused.get('message', '')[:300]}"
    field = field_row(root, pair)
    status = None if field is None else field['status']
    if status != 'done':
        what = '对齐 QC 不过' if refused is not None and refused.get('kind') == 'alignment_qc' else f'状态 {status}'
        return 'fallback', f'入口失败（{what}，D74-2）'
    if field['metrics'] is None:
        return NOT_READ, f'{short} 场没有评测'
    rd = HS.run_dir(root, HS.repair_cell(pair))
    record, done = read_json(rd/'repair.json'), read_json(rd/'queue_done.json')
    refused_repair = read_json(rd/hc.REFUSAL_FILE)
    if record is None and refused_repair is not None and refused_repair.get('kind') == 'provenance_rejected':
        return 'rejected', f"{short} 修复单元的读图被拒（输入错误，D74-4）：{refused_repair.get('message', '')[:300]}"     # revision 1b, the follow-up's minor 4
    if record is None:
        if done is None:
            return NOT_RUN, f'{short} 修复没有运行'
        return 'fallback', f"修复运行出错，没有 repair.json（{done.get('state')}，rc {done.get('rc')}，D74-1）"
    row = repair_row(root, pair)
    if row is None or not row.get('metrics'):
        return NOT_READ, f'{short} 修复没有评测（复核证书没有）'
    if record.get('passed') is not True:
        if row.get('certified_at_1') is True:
            return 'disagree', '修复进程的证书不过，评测单元的复核证书通过（C22.3-9 (d)）'
        return 'fallback', f"修复没有通过证书（{record.get('reason')}，D74-1）"
    if row.get('certified_at_1') is not True:
        return 'disagree', '修复进程的证书通过，评测单元的复核证书不过（C22.3-9 (d)）'
    return 'certified', None


def fallback_decision(root, pair):
    return read_json(HS.fallback_root(root)/core.short_name(pair)/'fallback.json')


def identity_metrics(root, pair):
    m = read_json(HS.id_root(root)/core.short_name(pair)/'identity_metrics.json')
    if m is None or m.get('certified_at_1') is not True or m.get('oracle') is not False or not m.get('metrics'):
        return None
    return m['metrics']


def method_row(root, pair, results=None):
    """What the method outputs for this pair: dict(line_status, metrics, why, ...) or a pair not read (line_status None, kind NOT_RUN or NOT_READ)."""
    kind, why = classify(root, pair)
    short = core.short_name(pair)
    out = dict(pair=short, kind=kind, why=why, line_status=None, metrics=None, wall_hit=None, rounds=None, task_clock=None, seconds_repair=None)
    row = repair_row(root, pair)
    if row is not None:
        out.update(wall_hit=row.get('wall_hit'), rounds=row.get('rounds'), task_clock=row.get('task_clock'), seconds_repair=row.get('seconds_repair'), deciding_passed=row.get('run_says_passed'), review_certified=row.get('certified_at_1'), min_lower_bound_after=row.get('min_lower_bound_after'))
    if kind in (NOT_RUN, NOT_READ):
        return out
    if kind == 'rejected':
        return dict(out, line_status=hc.HX_REJECTED)
    if kind == 'certified':
        return dict(out, line_status=hc.HX_CERTIFIED, metrics={m: row['after'].get(m) for m in MEASURES}, source='repaired field')
    decision = fallback_decision(root, pair)
    if decision is None:
        return dict(out, kind=NOT_READ, why=f'{short} 的退路（或第三次认证）还没有判（{why}）')
    if kind == 'disagree':
        third = decision.get('third_certificate') or {}
        if third.get('passed') is True:
            return dict(out, line_status=hc.HX_CERTIFIED, metrics={m: row['after'].get(m) for m in MEASURES}, source='repaired field (the third certificate passed)', third_certificate=True)
        if third.get('passed') is not False:
            return dict(out, kind=NOT_READ, why=f'{short} 的第三次认证没有结果')
        why = f'{why}；第三次认证不过'
    if decision.get('output') == 'A0':
        status = rp.fallback_status(pair, results=results)
        if not status['qualified']:
            return dict(out, kind=NOT_READ, why=f"{short} 退路记成 A0，但 d0_v1 的记录这次不合格：{status['why']}")
        return dict(out, line_status=hc.HX_A0, metrics={m: status['metrics'].get(m) for m in MEASURES}, source='configuration C, d0_v1 A0 (D0 records; certified again by the fallback unit)', why=why)
    if decision.get('output') == 'identity':
        metrics = identity_metrics(root, pair)
        if metrics is None:
            return dict(out, kind=NOT_READ, why=f'{short} 的恒等场没有评测或没过证书')
        return dict(out, line_status=hc.HX_IDENTITY, metrics={m: metrics.get(m) for m in MEASURES}, source='identity u = 0', why=f"{why}；配置 C 的场：{decision.get('a0_why')}")
    return dict(out, kind=NOT_READ, why=f"{short} 的退路记录不全：{decision.get('output')}")


def needs_fallback(root, pair):
    """The pairs the fallback unit decides: a fallback or a third certificate (classify)."""
    return classify(root, pair)[0] in ('fallback', 'disagree')


# ---- RW, the reference ----------------------------------------------------------------------------------------------------------------------
def rw_pins(results=None):
    """{short: sha256 of rw_v1's repair.json and repair_metrics.json}: written into batch.json at the start of the fields, checked by every reading (revision 1, review minor 11)."""
    root = Path(fx.RW.RESULTS if results is None else results)/fx.RW.ROOT_NAME
    out = {}
    for pair in ALL_PAIRS:
        d = fx.RW.run_dir(root, fx.RW.cell_of(pair))
        out[core.short_name(pair)] = {name: core.sha_file(d/name) if (d/name).is_file() else None for name in ('repair.json', 'repair_metrics.json')}
    return out


def rw_reference(results=None, pins=None):
    """{pair: dict(metrics, problem)} of rw_v1: the repaired field when its repair passed both certificates (repair.json passed, the evaluation's alpha = 1), else a problem.  RW was 19/19 (runs/20261105-fx/result.md); a pair that is not makes the reading 不可判.
    pins: batch.json's rw_pins; a record that is not the pinned one is a problem too."""
    root = Path(fx.RW.RESULTS if results is None else results)/fx.RW.ROOT_NAME
    rows, sources = fx_reading.rw_rows(root, results)
    now = rw_pins(results) if pins is not None else None
    out = {}
    for pair in ALL_PAIRS:
        row, record = rows.get(pair), read_json(fx.RW.run_dir(root, fx.RW.cell_of(pair))/'repair.json') or {}
        if pins is not None and now[core.short_name(pair)] != pins.get(core.short_name(pair)):
            out[pair] = dict(metrics=None, problem=f'{core.short_name(pair)} RW 的记录不是 batch.json 钉住的那份')
        elif row is None or not row.get('metrics') or sources.get(pair) != fx.RW.ROOT_NAME:
            out[pair] = dict(metrics=None, problem=f'{core.short_name(pair)} RW 没有读到（{sources.get(pair)}）')
        elif row.get('certified_at_1') is not True or record.get('passed') is not True:
            out[pair] = dict(metrics=None, problem=f'{core.short_name(pair)} RW 没有通过证书')
        else:
            out[pair] = dict(metrics={m: row['after'].get(m) for m in MEASURES}, problem=None, task_clock=row.get('task_clock'), wall_hit=row.get('wall_hit'))
    return out


# ---- the 〔模型〕expectation ---------------------------------------------------------------------------------------------------------------------
def pair_level(rows, pair, metric='seg35'):
    a, b = pair.split('__')
    if a not in rows or b not in rows or rows[a].get(metric) is None or rows[b].get(metric) is None:
        return None
    return (rows[a][metric]+rows[b][metric])/2


def expectation(evaluation, candidate, rw_mean=None, slope=hc.OX_SLOPE):
    """〔模型〕The segmentation-level Delta of the candidate against v1 (eval_uf.json, the 20 subjects), a pair = the mean of its two subjects (OX's .9172), over the 19 pairs and the first 5; times OX's slope .890: HS - RW expected.  Not a measurement."""
    arm = hc.CANDIDATES[candidate]['uf_arm']
    cand = (evaluation.get('rows') or {}).get(arm) or {}
    v1 = (evaluation.get('reference_rows') or {}).get('reference_v1') or {}
    out = dict(candidate=candidate, uf_arm=arm, slope=slope, slope_source='runs/20261105-fx/result.md OX 读法 2: (.9362 - .8625) / (1 - .9172), two points, linear', rw_seg35_mean=rw_mean)
    for name, pairs in (('pairs19', ALL_PAIRS), ('first5', core.PAIRS)):
        d = [None if pair_level(cand, p) is None or pair_level(v1, p) is None else pair_level(cand, p)-pair_level(v1, p) for p in pairs]
        ok = [x for x in d if x is not None]
        seg = mean(ok) if len(ok) == len(pairs) else None
        out[name] = dict(seg_level_delta=seg, expected_delta=None if seg is None else slope*seg, candidate_seg_level=mean(pair_level(cand, p) for p in pairs), v1_seg_level=mean(pair_level(v1, p) for p in pairs))
    e19 = out['pairs19']['expected_delta']
    out['expected_hs_seg35'] = None if e19 is None or rw_mean is None else rw_mean+e19
    out['note'] = '〔模型〕不是测量：OX 的斜率只有两个点、假定线性，只在这条线上量过；分割级 Δ 来自 UF 的同一批 20 人（候选在这上面选出，偏乐观）。写在运行之前。'
    return out


def write_prediction(root, evaluation, candidate, rw_mean, evaluation_sha256=None):
    """analysis/prediction_hs.json, once, before the first field: the expectation and the time it was written."""
    path = Path(root)/'analysis'/'prediction_hs.json'
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(expectation(evaluation, candidate, rw_mean), eval_uf_sha256=evaluation_sha256, written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False)+'\n')
    return document


# ---- the reading --------------------------------------------------------------------------------------------------------------------------------
def read_hs(rows, reference):
    """(reading, reason, comparison).  rows: {pair: method_row}; reference: rw_reference."""
    problems = []
    for pair in ALL_PAIRS:
        r = rows[pair]
        if r['kind'] in (NOT_RUN, NOT_READ):
            problems.append(r['why'])
        elif r['line_status'] == hc.HX_REJECTED:
            problems.append(f"{r['why']}（没有输出：输入错误，不是分割器的读数）")
    problems += [reference[p]['problem'] for p in ALL_PAIRS if reference[p]['problem']]
    compare = compare_rows(rows, reference)
    if problems:
        return hc.UNDECIDABLE, '；'.join(problems), compare
    d, sd = compare['seg35'], compare['hs_sdlogj_official']
    if d['n'] != len(ALL_PAIRS) or d['mean'] is None or d['lower'] is None or sd is None or not all(math.isfinite(v) for v in (d['mean'], d['lower'], sd)):
        return hc.UNDECIDABLE, f"Δ seg35 或官方 SDlogJ 没有定义（{d['n']}/{len(ALL_PAIRS)} 对，均值 {d['mean']}，下界 {d['lower']}，SDlogJ {sd}）", compare
    n_output = sum(rows[p]['metrics'] is not None for p in ALL_PAIRS)
    why = [w for w, ok in ((f"Δ 均值 {d['mean']:+.5f} < 0", d['mean'] >= hc.ADOPT_MEAN_MIN), (f"受试者块 95% 下界 {d['lower']:+.5f} ≤ {hc.ADOPT_LOWER}", d['lower'] > hc.ADOPT_LOWER),
                                 (f'有输出的对 {n_output}/19', n_output == len(ALL_PAIRS)), (f'官方 SDlogJ {sd:.4f} > {hc.SDLOGJ_MAX}', sd <= hc.SDLOGJ_MAX)) if not ok]
    if why:
        return hc.READ_NOT, '；'.join(why), compare
    return hc.READ_ADOPT, None, compare


def compare_rows(rows, reference):
    """HS - RW per pair, method level, on the pairs both have; the block interval on seg35 and seg4."""
    out = {}
    both = [p for p in ALL_PAIRS if rows[p]['metrics'] is not None and reference[p]['metrics'] is not None]
    for m in MEASURES:
        pairs = [p for p in both if rows[p]['metrics'].get(m) is not None and reference[p]['metrics'].get(m) is not None]
        d = [rows[p]['metrics'][m]-reference[p]['metrics'][m] for p in pairs]
        entry = dict(mean=mean(d), n=len(d), positive=sum(x > 0 for x in d), hs=mean(rows[p]['metrics'][m] for p in pairs), rw=mean(reference[p]['metrics'][m] for p in pairs), per_pair={core.short_name(p): x for p, x in zip(pairs, d)})
        if m in ('seg35', 'seg4') and len(d) >= 2:
            lo, hi = block_interval(d, [ALL_PAIRS.index(p) for p in pairs])
            entry.update(lower=lo, upper=hi, t95=t_interval(d))
        else:
            entry.update(lower=None, upper=None)
        out[m] = entry
    values = [None if rows[p]['metrics'] is None else rows[p]['metrics'].get('sdlogj_official') for p in ALL_PAIRS]
    out['hs_sdlogj_official'] = mean(values) if all(v is not None for v in values) else None      # all 19 or undefined (revision 1, review minor 10)
    return out


def t_interval(d):
    from scipy import stats
    d = np.asarray(d, dtype=np.float64)
    half = float(stats.t.ppf(.975, len(d)-1)*d.std(ddof=1)/math.sqrt(len(d)))
    return [float(d.mean())-half, float(d.mean())+half]


def counts(rows):
    by = {s: [r['pair'] for r in rows.values() if r['line_status'] == s] for s in (hc.HX_CERTIFIED, hc.HX_A0, hc.HX_IDENTITY, hc.HX_REJECTED)}
    return dict(by_status=by, n_fallback=len(by[hc.HX_A0])+len(by[hc.HX_IDENTITY]), n_entry_failure=sum(1 for r in rows.values() if r['line_status'] in (hc.HX_A0, hc.HX_IDENTITY) and (r['why'] or '').startswith('入口失败')),
                n_wall_hit=sum(1 for r in rows.values() if r.get('wall_hit')), wall_hit_pairs=[r['pair'] for r in rows.values() if r.get('wall_hit')], n_third_certificate=sum(1 for r in rows.values() if r.get('third_certificate')))


def certified_only(rows):
    """The means of the pairs the line itself outputs (hx_certified): biased, conditioned on passing (C22.5-4), reported apart."""
    chosen = [r for r in rows.values() if r['line_status'] == hc.HX_CERTIFIED]
    clocks = [r['task_clock'] for r in chosen if r.get('task_clock') is not None]
    out = dict(n=len(chosen), mean_task_clock=mean(clocks), max_task_clock=max(clocks, default=None), clock_90=dict(n_within=sum(c <= fx_reading.BZ_SECONDS for c in clocks), n=len(clocks)), mean_rounds=mean(r.get('rounds') for r in chosen), max_rounds=max((r['rounds'] for r in chosen if r.get('rounds') is not None), default=None),
               mean_seconds_repair=mean(r.get('seconds_repair') for r in chosen))
    out.update({m: mean(r['metrics'].get(m) for r in chosen) for m in MEASURES})
    return out


def method_means(rows):
    have = [r for r in rows.values() if r['metrics'] is not None]
    return dict(n=len(have), **{m: mean(r['metrics'].get(m) for r in have) for m in MEASURES}, note='含退路；退路 A0 的分组均值只在 D0 记录有的对上算' if any(r['line_status'] == hc.HX_A0 for r in have) else '含退路（这次没有退路）')


def code_record(root):
    files = dict(decide_hs=__file__, hs_core=hc.__file__, decide_fx=fx_reading.__file__, fx_core=fx.__file__, fl_core=fl.__file__, rp_core=rp.__file__)
    return dict({f'{k}_sha256': core.sha_file(Path(v)) for k, v in files.items()}, batch_code_sha=(read_json(Path(root)/'batch.json') or {}).get('code_sha'))


def decide(root, *, results=None, stops=None, uf_out=None):
    root = Path(root)
    rows = {p: method_row(root, p, results) for p in ALL_PAIRS}
    bad = sorted(r['run_id'] for p in ALL_PAIRS for r in [field_row(root, p)] if r is not None and r['metrics'] is not None and r['oracle'] is not False)
    if bad:
        raise SystemExit(f'run.json is not oracle false for {bad}; nothing written')
    reference = rw_reference(results, (read_json(root/'batch.json') or {}).get('rw_pins'))
    reading, reason, compare = read_hs(rows, reference)
    waiting, stop, candidate, uf = hc.uf_reading(HS.UF_OUT if uf_out is None else uf_out)
    hs_method = method_means(rows)
    prediction = read_json(root/'analysis'/'prediction_hs.json')
    return dict(schema='bayesreg.hs.decision.v1', code=code_record(root), is_conclusion=False, next_step='交总负责人与 BZ；采用之后它是 P2/P3 的参照配置（BZ 默认第 4 条）', oracle=False, note=NOTE, reading=reading, reason=reason, candidate=candidate, segmenter_id=None if candidate is None else hc.CANDIDATES[candidate]['segmenter_id'],
                uf_reading=None if uf is None else {k: uf.get(k) for k in ('reading', 'primary_arm', 'candidate', 'delta_seg35', 'delta_seg4')}, arms_stopped=stops or {},
                rules=dict(delta='HS - RW, seg35, per pair, method level (the fallbacks in), 19 pairs', adopt=f'mean >= {hc.ADOPT_MEAN_MIN} and subject-block 95 % lower > {hc.ADOPT_LOWER} and 19/19 with an output and the mean official SDlogJ <= {hc.SDLOGJ_MAX}', bootstrap=dict(n=hc.BOOT_N, seed=hc.BOOT_SEED), fuse='(c-ii): a fuse hit then a pass is a pass (D74-3), counted', wall_seconds=fx.FUSE_SECONDS),
                compare=compare, counts=counts(rows), method=hs_method, rw=dict(n=sum(reference[p]['metrics'] is not None for p in ALL_PAIRS), **{m: mean(reference[p]['metrics'].get(m) for p in ALL_PAIRS if reference[p]['metrics'] is not None) for m in MEASURES}),
                certified_only=certified_only(rows), gap_to_goal=None if hs_method['seg35'] is None else hc.GOAL-hs_method['seg35'], prediction=prediction, per_pair=[rows[p] for p in ALL_PAIRS]), rows


def summary(d):
    c = d['compare']
    L = ['# HS 读数（由 decide_hs.py 写出）', '', f"**读数：{d['reading']}**（不是结论，下一步交总负责人与 BZ）", '', d['note'], '']
    if d['reason']:
        L += [f"原因：{d['reason']}", '']
    u = d.get('uf_reading') or {}
    L += [f"候选：{d['candidate']}（{d['segmenter_id']}）；UF 的读数 {u.get('reading')}，主臂 {u.get('primary_arm')}，分割级 Δ（对 v1，20 人）seg35 {_signed(u.get('delta_seg35'), 5)} / seg4 {_signed(u.get('delta_seg4'), 5)}。", '']
    s35, s4 = c['seg35'], c['seg4']
    L += [f"Δ = HS − RW（方法级，含退路，配对）：seg35 {_signed(s35['mean'], 5)}（{s35['positive']}/{s35['n']} 为正；受试者块 95% [{_signed(s35.get('lower'), 4)}, {_signed(s35.get('upper'), 4)}]）；seg4 {_signed(s4['mean'], 5)}（{s4['positive']}/{s4['n']}；块 [{_signed(s4.get('lower'), 4)}, {_signed(s4.get('upper'), 4)}]）。",
          f"线：Δ 均值 ≥ 0，块下界 > {hc.ADOPT_LOWER}，19/19 有输出，官方 SDlogJ ≤ {hc.SDLOGJ_MAX} 读“采用新分割器”；否则“不采用”。", '']
    m, r = d['method'], d['rw']
    L += ['| | seg35 | seg4 | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | HD95 | 官方 / 脑内 SDlogJ |', '|---|---|---|---|---|---|---|---|---|---|']
    for name, x in (('HS（方法级，含退路）', m), ('RW（v1 通道）', r), ('HS 只算本线输出的对（有偏）', d['certified_only'])):
        L.append(f"| {name}（{x['n']} 对） | {_f(x['seg35'])} | {_f(x['seg4'])} | {_f(x['small_six'])} | {_f(x['large_six'])} | {_f(x['other_23'])} | {_f(x['cortex'])} | {_f(x['white_matter'])} | {_f(x['hd95'], 3)} | {_f(x['sdlogj_official'])} / {_f(x['sdlogj_brain'])} |")
    L += ['', 'HS − RW 其余：' + '，'.join(f"{name} {_signed(c[k]['mean'])}（{c[k]['positive']}/{c[k]['n']}）" for k, name in (('small_six', '小六个'), ('large_six', '大六个'), ('other_23', '其余 23 个'), ('cortex', '皮层'), ('white_matter', '白质'), ('sdlogj_official', '官方 SDlogJ'))) + f"，HD95 {_signed(c['hd95']['mean'], 3)}。", '']
    L += [f"离 .88 差 {_f(d['gap_to_goal'])}（HS 方法级 seg35）。", '']
    k, co = d['counts'], d['certified_only']
    L += [f"本线输出 {len(k['by_status'][hc.HX_CERTIFIED])}/19；退路 A0（配置 C）{k['by_status'][hc.HX_A0] or '无'}；退路恒等 {k['by_status'][hc.HX_IDENTITY] or '无'}；其中入口失败 {k['n_entry_failure']}；通道被拒 {k['by_status'][hc.HX_REJECTED] or '无'}；撞 1200 s 保险丝 {k['n_wall_hit']}（{'、'.join(k['wall_hit_pairs']) or '无'}，撞了之后证书过算过，D74-3）；第三次认证 {k['n_third_certificate']}。",
          f"任务时钟（本线输出的对，不含 U-Net）平均 / 最长 {_f(co['mean_task_clock'], 1)} / {_f(co['max_task_clock'], 1)} s，{fx_reading.CLOCK_COLUMN}：{co['clock_90']['n_within']}/{co['clock_90']['n']}；修复轮数 平均 / 最多 {_f(co['mean_rounds'], 2)} / {co['max_rounds']}，修复秒数平均 {_f(co['mean_seconds_repair'], 1)}。", '']
    p = d.get('prediction')
    if p:
        L += [f"〔模型〕运行之前写下的预期（prediction_hs.json，{p.get('written_utc')}）：分割级 Δ（19 对，一对取两人平均）{_signed(p['pairs19']['seg_level_delta'], 5)} × 斜率 {p['slope']} = HS − RW {_signed(p['pairs19']['expected_delta'], 5)}，即 HS 约 {_f(p.get('expected_hs_seg35'))}；前 5 对 {_signed(p['first5']['expected_delta'], 5)}。{p['note']}", '']
    L += ['## 每对', '', '| 对 | 输出 | 原因 | 轮数 | 撞保险丝 | 任务时钟 s | seg35 / seg4 | 官方 SDlogJ | Δ seg35 |', '|---|---|---|---|---|---|---|---|---|']
    for row in d['per_pair']:
        mt = row['metrics'] or {}
        L.append(f"| {row['pair']} | {row['line_status'] or row['kind']} | {row['why'] or ''} | {row.get('rounds')} | {row.get('wall_hit')} | {_f(row.get('task_clock'), 1)} | {_f(mt.get('seg35'))} / {_f(mt.get('seg4'))} | {_f(mt.get('sdlogj_official'))} | {_signed(c['seg35']['per_pair'].get(row['pair']))} |")
    L += ['', f"读数的规则：{d['code']}", '', '## 它证伪不了什么', '', '不是 F_B，没有经认证的下降，只有终态过证书；开发集 19 对单次；候选在同样 20 人的分割级读数上选出，19 对之前选过配；修复在 GPU 上不逐位可复现（D76），轮数与秒数是一次抽样。“采用”不说明留出集上会更好；“不采用”也不说明别的分割器、别的权重不行。进方法先改合同（D72），由 BZ 定。']
    return '\n'.join(L)+'\n'


def write_hs(root, *, results=None, stops=None, uf_out=None):
    document, rows = decide(root, results=results, stops=stops, uf_out=uf_out)
    out = Path(root)/'analysis'
    names = ('decision_hs.json', 'summary_hs.md', 'per_pair_hs.csv')
    set_aside(out, names)
    (out/names[0]).write_text(json.dumps(document, indent=1, ensure_ascii=False)+'\n')
    (out/names[1]).write_text(summary(document), encoding='utf-8')
    with (out/names[2]).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'line_status', 'kind', *MEASURES, 'rw_seg35', 'delta_seg35', 'rounds', 'wall_hit', 'task_clock'])
        reference = rw_reference(results, (read_json(Path(root)/'batch.json') or {}).get('rw_pins'))
        for pair in ALL_PAIRS:
            r, ref = rows[pair], reference[pair]['metrics'] or {}
            mt = r['metrics'] or {}
            writer.writerow([r['pair'], r['line_status'], r['kind'], *[mt.get(m) for m in MEASURES], ref.get('seg35'), None if mt.get('seg35') is None or ref.get('seg35') is None else mt['seg35']-ref['seg35'], r.get('rounds'), r.get('wall_hit'), r.get('task_clock')])
    return document
