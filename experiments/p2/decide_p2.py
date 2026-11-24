"""P2's and P3's readings (runs/20261106-p2/任务书.md, plan.md; by code; labels for the lead and BZ, not conclusions).  decide_fl, decide_fx and their batches are not edited: their functions on
the records are called, through another instance of decide_fl for each batch here (decide_fx._records), and one more for the reference's directories.

What a pair of an arm is (the evaluation's certificate, from the field, configuration C, alpha = 1; D74 as BZ set it on 11-05: a fuse hit with the certificate then passing counts as passed):
    certified        the repaired field passes the certificate (a fuse hit is reported beside it)
    entry_failure    the T88 run has no status done (time_cap, vram_stop, cuda_oom, nonfinite, scale_mismatch, checkpoint_mismatch, not_predicted_channels, estimate_failed, or no run.json)
    repair_failure   the repaired field does not pass, or the repair process left no repair.json
    not_read         not run, or not evaluated: it says nothing about the arm
A failure goes to the fallback of 2c(i) (D74, D75): configuration C's field of d0_v1 (A0) when d0_v1's records qualify it (rp_core.fallback_status) and its certificate passes again in the
fallback unit, which reads no label (C22.5-2; <root>/fallback/<pair>/fallback.json), else the identity u = 0 (its metrics evaluated into <root>/id).  The reference by the same rule
(revision 2, I2 (b)): a row HS sent to a fallback enters with the value HS recorded.

P2 (the first 5 pairs, by arm; not one of the 3 selection readings):
    a failure on any pair                                                     没过门 (the arm's failures named)
    a pair not read, or the reference not complete                            不可判
    5/5 certified, mean official SDlogJ <= .10, mean seg35 difference >= -.004 against the reference     过门
    otherwise                                                                 没过门
  The order AC > A > C: the first arm through the gate is the candidate (有候选); an arm before it that is 不可判 makes the reading 不可判; none through: 没有候选.
P3 (the 19 pairs, the candidate; selection reading 2), the method's value of each pair (the repaired field, or its fallback) against the reference's:
    a pair not read, a failed pair without a fallback, the reference not complete     不可判
    the subject-block 95 % interval of the seg35 difference has its lower bound > -.003, the mean difference >= -.001, the mean official SDlogJ <= .10      有收益 (and 优效也成立 when the lower bound > 0)
    otherwise                                                                 没有收益
  "The Bayesian part is not a no-op" is P1's judgment (A 不是空操作, C 弱改动); P3 reports the estimates' distance from the reference values again, in no line.
"""
import json
import math
from pathlib import Path

import numpy as np

from decide_ef import read_json, set_aside
import decide_fl as fl_reading
import decide_fx as fx_reading
import mc0_core as core
import p1_core as p1
import p2_core as p2
from p2_core import ALL_PAIRS, NATIVE, ORDER
import rp_core as rp

MEASURES = fl_reading.MEASURES
mean = fl_reading.mean
_f, _signed = fl_reading._f, fl_reading._signed
UNDECIDABLE = '不可判'
THROUGH, NOT_THROUGH = '过门', '没过门'
READ_CANDIDATE, READ_NONE = '有候选', '没有候选'
READ_GAIN, READ_NO_GAIN, SUPERIOR = '有收益', '没有收益', '优效也成立'
CERTIFIED, ENTRY_FAILURE, REPAIR_FAILURE, NOT_READ, FALLBACK = 'certified', 'entry_failure', 'repair_failure', 'not_read', 'fallback'
FAILURES = (ENTRY_FAILURE, REPAIR_FAILURE)
REPAIR, A0, IDENTITY = 'repair', 'A0', 'identity'
ID_DIR = 'id'
FALLBACK_DIR = 'fallback'                       # revision 2 (D75, C22.5-2): <root>/fallback/<pair>/fallback.json, the fallback recertified in a process that reads no label
LOG_TAIL = 5                                    # the last lines of a unit's log copied into a failure without run.json (revision 2, the review's minor 2)
REJUDGE_KINDS = ('没有 run.json 的入口失败，可能是防火墙或 provenance 的拒绝，D74-4 不走退路，记录分不开，C22.11 第 5 条；'
                 '或修复进程的证书与评测单元的复核证书不一致，C22.3-9 (d)，候选不做第三次认证')
P1_JUDGMENT = dict(A='A 不是空操作', C='C 是弱改动', source='runs/20261105-p1/result.md (p1_v1/summary.json)')
NOTE = ('开发集单次观测。通道来自已登记分割器的预测（oracle: false），真值标签只进评测。**不是 F_B，没有经认证的下降，只有终态过证书；不进结果表，进方法先改合同，由 BZ 定。** '
        'D 因子 24 是显式的手选常数，不是贝叶斯估计；EM 只估相对权，矩匹配只估两个比值（设计说明 2.2、3）。T88 不是 EM 的下降法，B4 的单调性不适用。'
        '前 5 对在选配时已多次用过；P3 的 19 对含这 5 对。撞 1200 s 保险丝之后证书通过算通过（D74，BZ 11-05），撞的次数单列。')

records = {name: fx_reading._records(batch) for name, batch in p2.BATCHES.items()}
REF = p2.DirectoryBatch()
ref_records = fx_reading._records(REF)


# ---- the rows ---------------------------------------------------------------------------------------------------------------------------------
def classify(field, repair, has_repair_json):
    """(kind, why) of one pair of one arm from its field row (decide_fl.field_rows at 600) and its repair row (decide_fl.repair_row, with the wall)."""
    if field is None:
        return NOT_READ, '没有运行'
    if field['status'] != 'done':
        return ENTRY_FAILURE, f"入口失败（{field['status']}）"
    if field['metrics'] is None:
        return NOT_READ, '场没有评测'
    return classify_repair(repair, has_repair_json)


def classify_repair(repair, has_repair_json):
    """(kind, why) of a repair row: certified (a fuse hit beside it), repair_failure, not_read."""
    if repair is None:
        return NOT_READ, '没有修复'
    if not has_repair_json:
        return REPAIR_FAILURE, f"修复运行出错，没有 repair.json（{repair.get('status')}）"
    if not repair.get('metrics'):
        return NOT_READ, '修复没有评测'
    if repair.get('certified_at_1'):
        return CERTIFIED, ('撞 1200 s 保险丝之后证书通过' if repair.get('wall_hit') else None)
    return REPAIR_FAILURE, '修复后没有通过证书'


def bayes_record(run):
    """What the run estimated: A's weights (the final used ones, their distance to Q3h, the scale, the checks), C's ratios per pass, the control of pass 0."""
    passes = (run or {}).get('passes') or []
    out = dict(n_passes=len(passes), pass0_u_sha256=passes[0].get('u_sha256') if passes else None)
    em = [p.get('em') for p in passes if p.get('em')]
    if em:
        last = em[-1]
        out['em'] = dict(d_to_q3h_final=last['used_final']['d_to_q3h'], s_final=last['used_final']['s'], w_final=last['used_final']['w'], s0=last['s0'],
                         d_by_step=[[row['where'], row['d_to_q3h'], row['used']] for row in last['steps']], collapse_flagged=sorted({c for e in em for row in e['steps'] for c in row['collapse_flagged']}),
                         gram_checks=all(e['gram_check']['passed'] for e in em), native_scale_checks=all(r['passed'] for e in em for r in e['native_scale_checks']), left_out=sorted({c for e in em for row in e['steps'] for c in row['left_out']}))
    tissue = [p.get('tissue') for p in passes]
    if any(t for t in tissue):
        out['tissue'] = dict(ratios_used=[(t or {}).get('ratios_used') for t in tissue], ratios_estimated=[(p.get('estimate') or {}).get('ratios') for p in passes])
    return out


def log_tail(directory, n=LOG_TAIL):
    """The last lines of a unit's log (log.txt, else systemd_run.log): the reason of a SystemExit that left no run.json."""
    for name in ('log.txt', 'systemd_run.log'):
        path = Path(directory)/name
        if path.is_file():
            lines = [line for line in path.read_text(errors='replace').splitlines() if line.strip()]
            if lines:
                return lines[-n:]
    return []


def arm_row(name, root, arm, pair):
    batch, rec = p2.BATCHES[name], records[name]
    fcell, rcell = batch.field_cell(arm, pair), batch.repair_cell(arm, pair)
    fdir = batch.run_dir(root, fcell)
    field = rec.field_rows(root, fcell).get(NATIVE)
    rdir = batch.run_dir(root, rcell)
    repair = fx_reading.with_wall(rec.repair_row(root, rcell), rdir)
    kind, why = classify(field, repair, (rdir/'repair.json').is_file())
    run = read_json(fdir/'run.json')
    # revision 2 (the review's minor 2): an entry failure without run.json may be a firewall or provenance refusal, which D74-4 does not send to the fallback; the records cannot tell (C22.11 item 5)
    rejudge = kind == ENTRY_FAILURE and run is None
    if rejudge:
        why = f"{why}；没有 run.json，可能是防火墙或 provenance 的拒绝（D74-4 不走退路），要总负责人改判；日志末行：{' | '.join(log_tail(fdir))}"
    # revision 4 (the review of revision 3, minor 2): the two certificates of a candidate disagree (C22.3-9 (d): an implementation error, for the lead).  The candidates get no third
    # certificate: the reading counts the evaluation's review certificate (a repair failure, or certified the other way round) and goes on; the pair is marked
    if repair is not None and repair.get('metrics') and repair.get('run_says_passed') is not None and bool(repair.get('run_says_passed')) != bool(repair.get('certified_at_1')):
        rejudge = True
        why = (f"{why}；" if why else '') + f"修复进程的证书{'过' if repair.get('run_says_passed') else '不过'}、评测单元的复核证书{'过' if repair.get('certified_at_1') else '不过'}（C22.3-9 (d) 的不一致，候选不做第三次认证，读数按复核证书），要总负责人改判"
    return dict(pair=pair, short=core.short_name(pair), arm=arm, kind=kind, why=why, field=field, repair=repair, bayes=bayes_record(run), task_clock=None if repair is None else repair.get('task_clock'), needs_lead_rejudgment=rejudge)


def reference_rows(reference, stage, segmenter_id, hs_decision=None):
    """{pair: row}: the reference by the rule of the candidates (revision 2, the lead on the review's I2 (b)): hx_certified, the repair through decide_fl.repair_row on its pinned directory,
    certified at alpha = 1, its value the repaired field's metrics; a fallback of HS, its value the method_metrics HS recorded (and decision_hs.json says the same).  Every row through
    p2_core.reference_pair_problems first (I1: the pair, the segmenter, oracle false, the frozen Hx24, the fuse, the source, the pins); a row with a problem is not read."""
    path = p2.HS_DECISION if hs_decision is None else Path(hs_decision)
    decision = read_json(path)
    hs = p2.hs_reference_problems(reference, decision, core.sha_file(path) if decision is not None else None, segmenter_id)
    pred_root = p2.expected_pred_root(segmenter_id, reference)
    out = {}
    for pair in (core.PAIRS if stage == 'p2' else ALL_PAIRS):
        short = core.short_name(pair)
        pin = reference[stage]['pairs'][short]
        problems, status = p2.reference_pair_problems(pin, pair, segmenter_id, pred_root)
        problems += [x for x in hs if x.startswith('HS:') or x.startswith(f'{stage} {short}:')]
        d = Path(pin['repair_dir']) if pin.get('repair_dir') else None
        row = None if d is None else fx_reading.with_wall(ref_records.repair_row(d, p2.RefCell('RHx24', pin['tier'], pair)), d)
        value, source = None, None
        if problems:
            kind, why = NOT_READ, '参照这一行不对：' + '；'.join(problems)
        elif status == p2.HX_CERTIFIED:
            kind, why = classify_repair(row, d is not None and (d/'repair.json').is_file())
            if kind == REPAIR_FAILURE and row is not None and row.get('metrics') and not p2.third_certificate_problems(pin, pair):
                kind, why = CERTIFIED, '评测单元的复核证书不过，第三次认证（不读标签的进程）通过：按 C22.3-9 (d) 的机械规则算过证书'     # revision 3 (HS 1b's review, new minor 2)
            if kind == CERTIFIED:
                value, source = row['after'], REPAIR
            else:
                kind, why = NOT_READ, f'参照记为 hx_certified，但{why or kind}'
        else:
            kind, why, value, source = FALLBACK, f'参照走退路（{status}），代入 HS 记录的方法值', dict(pin['method_metrics']), status
        out[pair] = dict(pair=pair, short=short, kind=kind, why=why, repair=row, value=value, source=source, line_status=status, field_u_sha256=pin.get('field_u_sha256'), directory=None if d is None else str(d), pinned=not problems)
    return out


def fallback_rows(root, pairs):
    """{pair: <root>/fallback/<pair>/fallback.json or None}: the fallback the fallback unit decided after certifying it again in a process that reads no label (revision 2, D75, C22.5-2)."""
    return {pair: read_json(Path(root)/FALLBACK_DIR/core.short_name(pair)/'fallback.json') for pair in pairs}


def identity_rows(root, pairs):
    """{pair: what <root>/id holds of the identity u = 0}: usable when certified at alpha = 1, oracle false, its metrics there."""
    out = {}
    for pair in pairs:
        m = read_json(Path(root)/ID_DIR/core.short_name(pair)/'identity_metrics.json')
        usable = m is not None and m.get('certified_at_1') is True and m.get('oracle') is False and bool(m.get('metrics'))
        out[pair] = dict(usable=usable, metrics=(m or {}).get('metrics'), why=None if usable else '没有可用的恒等场评测')
    return out


def method_value(row, fallback, identity):
    """(source, metrics, why) of the method on one pair: the repaired field, or the fallback of 2c(i) as the fallback unit decided it (fallback.json; D75: configuration C's field of
    d0_v1 when its records qualify it and the certificate passes again here, else the identity, whose metrics the evaluation unit writes)."""
    if row['kind'] == CERTIFIED:
        return REPAIR, row['repair']['after'], row['why']
    if row['kind'] in FAILURES:
        if fallback is None:
            return None, None, f"{row['why']}；退路还没有决定（没有 fallback.json）"
        if fallback.get('kind') == A0:
            return A0, fallback.get('a0_metrics'), f"{row['why']}；配置 C 的场重新认证通过"
        if fallback.get('kind') == IDENTITY:
            if identity['usable']:
                return IDENTITY, identity['metrics'], f"{row['why']}；配置 C 的场不能用（{fallback.get('why')}），代入恒等"
            return None, None, f"{row['why']}；退路是恒等，但{identity['why']}"
        return None, None, f"{row['why']}；退路单元没有给出可用的场（{fallback.get('why')}）"
    return None, None, row['why']


# ---- P2 -------------------------------------------------------------------------------------------------------------------------------------
P2_MIN_DELTA_PAIRS = 4                          # revision 3 (the lead on the follow-up's R1, (a)): fewer pairs left for Delta than this: 不可判


def arm_gate(rows, ref):
    """(status, why, summary) of one arm on the first 5 pairs against the reference's rows.  Revision 3 (the lead on R1, (a)): a pair whose reference row is a fallback of the frozen
    line does not enter Delta (it would compare the frozen line's own failure, not the Bayesian part); with at least 4 pairs left the gate reads as usual and names the pairs left out,
    with fewer it is 不可判.  The arm's own 5/5 and its SDlogJ are over all 5 pairs."""
    failed = [r for r in rows if r['kind'] in FAILURES]
    unread = [r for r in rows if r['kind'] == NOT_READ]
    ref_bad = [r for r in ref.values() if r['kind'] != FALLBACK and (r.get('value') is None or any(r['value'].get(m) is None for m in p2.METHOD_KEYS))]
    excluded = [r['short'] for r in ref.values() if r['kind'] == FALLBACK]
    certified = [r for r in rows if r['kind'] == CERTIFIED]
    paired = [r for r in certified if ref[r['pair']]['kind'] == CERTIFIED and ref[r['pair']].get('value') is not None]
    deltas = {m: [r['repair']['after'][m]-ref[r['pair']]['value'][m] for r in paired if r['repair']['after'].get(m) is not None and ref[r['pair']]['value'].get(m) is not None] for m in MEASURES}
    summary = dict(n=len(rows), n_certified=len(certified), failures=[dict(pair=r['short'], kind=r['kind'], why=r['why'], needs_lead_rejudgment=r.get('needs_lead_rejudgment', False)) for r in failed], not_read=[dict(pair=r['short'], why=r['why']) for r in unread],
                   fuse_hits=[r['short'] for r in certified if r['repair'].get('wall_hit')], mean_task_clock=mean(r['task_clock'] for r in certified), max_task_clock=max((r['task_clock'] for r in certified if r['task_clock'] is not None), default=None),
                   after={m: mean(r['repair']['after'].get(m) for r in certified) for m in MEASURES},
                   # revision 4 (the review of revision 3, minor 3): the reference and the arm on the pairs that enter Delta only, so that Delta is the difference of these two means
                   reference={m: mean(ref[r['pair']]['value'].get(m) for r in paired) for m in MEASURES}, after_delta_pairs={m: mean(r['repair']['after'].get(m) for r in paired) for m in MEASURES},
                   delta_pairs=[r['short'] for r in paired],
                   reference_fallbacks=[dict(pair=r['short'], line_status=r['line_status']) for r in ref.values() if r['kind'] == FALLBACK], delta_excluded=excluded, n_delta=len(paired),
                   delta={m: dict(mean=mean(v), positive=sum(x > 0 for x in v), n=len(v)) for m, v in deltas.items()})
    if failed:
        return NOT_THROUGH, '失败：' + '；'.join(f"{r['short']} {r['why']}" for r in failed), summary
    if unread or ref_bad:
        return UNDECIDABLE, '没有读到：' + '；'.join([f"{r['short']} {r['why']}" for r in unread]+[f"参照 {r['short']} {r['why'] or r['kind']}" for r in ref_bad]), summary
    left = len(ref)-len(excluded)
    if left < P2_MIN_DELTA_PAIRS:
        return UNDECIDABLE, f"参照在 {'、'.join(excluded)} 走了退路，这些对不进 Δ，只剩 {left} 对（< {P2_MIN_DELTA_PAIRS}）：交总负责人", summary
    sdlogj, delta = summary['after']['sdlogj_official'], summary['delta']['seg35']['mean']
    lines = dict(certified=len(certified) == p2.P2_N, sdlogj=sdlogj is not None and sdlogj <= p2.SDLOGJ_MAX, delta=delta is not None and delta >= p2.P2_DELTA_MIN)
    summary['lines'] = dict(lines, sdlogj_value=sdlogj, delta_value=delta, sdlogj_max=p2.SDLOGJ_MAX, delta_min=p2.P2_DELTA_MIN, n_needed=p2.P2_N, n_delta=len(paired), delta_excluded=excluded)
    note = f"（参照在 {'、'.join(excluded)} 走了退路，这些对不进 Δ，Δ 按 {len(paired)} 对算）" if excluded else None
    if all(lines.values()):
        return THROUGH, note, summary
    return NOT_THROUGH, '线：' + '；'.join(k for k, ok in lines.items() if not ok) + (note or ''), summary


def read_p2(arms):
    """(reading, candidate, reason) from {arm: (status, why, summary)} in the order AC > A > C."""
    for arm in ORDER:
        status, why, _ = arms[arm]
        if status == THROUGH:
            return READ_CANDIDATE, arm, None
        if status == UNDECIDABLE:
            return UNDECIDABLE, None, f'{arm} 不可判（{why}），顺序在它之后的臂不能代替它'
    return READ_NONE, None, '三个臂都没过门：停下，回到 1′(i)'


def code_record(root):
    files = dict(decide_p2=__file__, p2_core=p2.__file__, decide_fx=fx_reading.__file__, decide_fl=fl_reading.__file__, rp_core=rp.__file__)
    return dict({f'{k}_sha256': core.sha_file(Path(v)) for k, v in files.items()}, batch_code_sha=(read_json(Path(root)/'batch.json') or {}).get('code_sha'))


def batch_reference(root, batch):
    """The segmenter and the reference of a batch, from its batch.json (the queue wrote both at the start); the file itself, its sha256 the pinned one."""
    record = read_json(Path(root)/'batch.json')
    if record is None:
        return None, None
    segmenter = record.get('segmenter_id')
    path = p2.reference_path(segmenter)
    if not path.is_file() or core.sha_file(path) != (record.get('reference') or {}).get('sha256'):
        raise SystemExit(f"{path}: not the reference batch.json pinned ({(record.get('reference') or {}).get('sha256')}); nothing written")
    return segmenter, p2.load_reference(segmenter, path)


def decide_p2(root, stops=None):
    root = Path(root)
    segmenter, reference = batch_reference(root, p2.P2)
    if reference is None:
        return dict(schema='bayesreg.p2.decision.v1', reading=UNDECIDABLE, reason='没有 batch.json：什么都还没跑', candidate=None), None
    ref = reference_rows(reference, 'p2', segmenter)
    rows = {arm: [arm_row('p2', root, arm, pair) for pair in core.PAIRS] for arm in ORDER}
    bad = sorted(r['repair']['run_id'] for rs in rows.values() for r in rs if r['repair'] is not None and r['repair'].get('metrics') and r['repair'].get('oracle') is not False)
    if bad:
        raise SystemExit(f'repair.json is not oracle false for {bad}; nothing written')
    arms = {arm: arm_gate(rows[arm], ref) for arm in ORDER}
    reading, candidate, reason = read_p2(arms)
    control = [dict(pair=r['short'], pass0=r['bayes'].get('pass0_u_sha256'), reference=ref[r['pair']]['field_u_sha256'], same=None if r['bayes'].get('pass0_u_sha256') is None or ref[r['pair']]['field_u_sha256'] is None else r['bayes']['pass0_u_sha256'] == ref[r['pair']]['field_u_sha256'],
                    note='无参照场（参照这一对走了退路，没有 T88 场）' if ref[r['pair']]['field_u_sha256'] is None else None) for r in rows['C']]
    rejudge = [dict(arm=r['arm'], pair=r['short'], why=r['why']) for rs in rows.values() for r in rs if r.get('needs_lead_rejudgment')]
    return dict(schema='bayesreg.p2.decision.v1', code=code_record(root), is_conclusion=False, next_step='有候选：P3（BZ 默认第 3 条）；没有候选或不可判：交总负责人', oracle=False, note=NOTE, segmenter_id=segmenter, reference=reference['p2'].get('source'),
                reading=reading, candidate=candidate, reason=reason, needs_lead_rejudgment=rejudge, order=list(ORDER), lines=dict(n=p2.P2_N, sdlogj_max=p2.SDLOGJ_MAX, delta_min=p2.P2_DELTA_MIN), arms_stopped=stops or {},
                arms={arm: dict(status=s, why=w, summary=summ) for arm, (s, w, summ) in arms.items()}, control_c_pass0=control,
                per_pair={arm: [pair_line_record(r, ref[r['pair']], exclude_fallback=True) for r in rows[arm]] for arm in ORDER}, reference_pairs=[dict(pair=r['short'], kind=r['kind'], line_status=r['line_status'], why=r['why'], directory=r['directory'], pinned=r['pinned']) for r in ref.values()]), rows


def pair_line_record(r, ref, *, exclude_fallback=False):
    """One pair's line.  exclude_fallback (P2, revision 4): a pair whose reference is a fallback gets no Delta, it does not enter Delta (R1, (a))."""
    after = (r['repair'] or {}).get('after') if r['kind'] == CERTIFIED else None
    excluded = exclude_fallback and ref.get('kind') == FALLBACK
    ref_after = None if excluded else ref.get('value')
    return dict(pair=r['short'], kind=r['kind'], why=r['why'], excluded_from_delta=excluded, reference_line_status=ref.get('line_status'), rounds=(r['repair'] or {}).get('rounds'), wall_hit=(r['repair'] or {}).get('wall_hit'), task_clock=r['task_clock'],
                seg35=None if after is None else after['seg35'], seg4=None if after is None else after['seg4'], sdlogj_official=None if after is None else after['sdlogj_official'],
                delta_seg35=None if after is None or ref_after is None else after['seg35']-ref_after['seg35'], delta_seg4=None if after is None or ref_after is None else after['seg4']-ref_after['seg4'], bayes=r['bayes'])


def summary_p2(d):
    L = ['# P2 读数（由 decide_p2.py 写出）', '', f"**读数：{d['reading']}**" + (f"，候选 {d['candidate']}" if d.get('candidate') else '') + '（不是结论）', '', d.get('note', ''), '']
    if d.get('reason'):
        L += [f"原因：{d['reason']}", '']
    if d.get('needs_lead_rejudgment'):
        L += [f"**要总负责人改判**（{REJUDGE_KINDS}；读数照常给出，不自动重跑）：" + '；'.join(f"{x['arm']} {x['pair']}：{x['why']}" for x in d['needs_lead_rejudgment']), '']
    L += [f"分割器 {d.get('segmenter_id')}；参照：{d.get('reference')}", '', '| 臂 | 门 | 过证书 | seg35 / seg4（5 对） | 进 Δ 的对：臂 / 参照 seg35 | 对参照 Δseg35 / Δseg4（正的对数） | 官方 / 脑内 SDlogJ | HD95 | 小六个 | 大六个 | 皮层 | 白质 | 任务时钟 平均 / 最长 s | 撞保险丝 | 原因 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in d.get('order', []):
        a = d['arms'][arm]
        s = a['summary']
        L.append(f"| {arm} | {a['status']} | {s['n_certified']}/{s['n']} | {_f(s['after']['seg35'])} / {_f(s['after']['seg4'])} | {len(s['delta_pairs'])} 对：{_f(s['after_delta_pairs']['seg35'])} / {_f(s['reference']['seg35'])} | {_signed(s['delta']['seg35']['mean'])}（{s['delta']['seg35']['positive']}/{s['delta']['seg35']['n']}）/ {_signed(s['delta']['seg4']['mean'])}（{s['delta']['seg4']['positive']}/{s['delta']['seg4']['n']}） | {_f(s['after']['sdlogj_official'])} / {_f(s['after']['sdlogj_brain'])} | {_f(s['after']['hd95'], 3)} | {_f(s['after']['small_six'])} | {_f(s['after']['large_six'])} | {_f(s['after']['cortex'])} | {_f(s['after']['white_matter'])} | {_f(s['mean_task_clock'], 1)} / {_f(s['max_task_clock'], 1)} | {'、'.join(s['fuse_hits']) or '无'} | {a['why'] or ''} |")
    if d.get('arms'):
        s0 = next(iter(d['arms'].values()))['summary']
        r = s0['reference']
        L += ['', f"参照（进 Δ 的对，{len(s0['delta_pairs'])} 对 {'、'.join(s0['delta_pairs'])}；修订 4 起参照均值只用这些对）：seg35 / seg4 {_f(r['seg35'])} / {_f(r['seg4'])}，官方 / 脑内 SDlogJ {_f(r['sdlogj_official'])} / {_f(r['sdlogj_brain'])}，HD95 {_f(r['hd95'], 3)}，大六个 {_f(r['large_six'])}，皮层 {_f(r['cortex'])}，白质 {_f(r['white_matter'])}。"]
        if s0['reference_fallbacks']:
            L += ['', '**不进 Δ 的对**（参照在这一对走了退路，R1 (a)；只列臂的值，不给 Δ）：', '', '| 臂 | 对 | 参照的 line_status | 臂的种类 | 臂 seg35 / seg4 |', '|---|---|---|---|---|']
            L += [f"| {arm} | {p['pair']} | {p['reference_line_status']} | {p['kind']} | {_f(p['seg35'])} / {_f(p['seg4'])} |" for arm in d['order'] for p in d['per_pair'][arm] if p['excluded_from_delta']]
        L += ['', '## 估出的值（每对）', '', '| 臂 | 对 | 种类 | d(w_EM, Q3h) 末次 | s 末次 / s0 | CSF/WM、GM/WM（各遍用的） | 末遍场上再估（不用） | Δseg35 |', '|---|---|---|---|---|---|---|---|']
        for arm in d['order']:
            for p in d['per_pair'][arm]:
                b = p['bayes']
                em, ti = b.get('em') or {}, b.get('tissue') or {}
                used = '；'.join('a ≡ 1' if u is None else f"{u['CSF']:.3f}、{u['GM']:.3f}" for u in ti.get('ratios_used', [])) if ti else '—'
                est = (ti.get('ratios_estimated') or [None])[-1] if ti else None
                est_text = '—' if not est else f"{est['CSF']:.3f}、{est['GM']:.3f}"
                L.append(f"| {arm} | {p['pair']} | {p['kind']} | {_f(em.get('d_to_q3h_final'), 3)} | {_f(em.get('s_final'), 3)} / {_f(em.get('s0'), 3)} | {used} | {est_text} | {'不进 Δ' if p['excluded_from_delta'] else _signed(p['delta_seg35'])} |")
        L += ['', f"C 第 0 遍（a ≡ 1）对参照的 T88 场逐位：{[(c['pair'], c['same']) for c in d['control_c_pass0']]}"]
    L += ['', '## 它证伪不了什么', '', '前 5 对、单次，只是一道无效性门（MDE 约 .004）；不说明 19 对上会怎样。不是 F_B，没有经认证的下降；EM 与矩匹配是在 T88 的终态上做的，不是后验众数上的估计。进方法先改合同，由 BZ 定。']
    return '\n'.join(L)+'\n'


def write_p2(root, stops=None):
    decision, rows = decide_p2(root, stops)
    out = Path(root)/'analysis'
    set_aside(out, ('decision_p2.json', 'summary_p2.md', 'long.csv'))
    (out/'decision_p2.json').write_text(json.dumps(decision, indent=1, ensure_ascii=False)+'\n')
    (out/'summary_p2.md').write_text(summary_p2(decision), encoding='utf-8')
    if rows:
        write_long(out/'long.csv', [r for rs in rows.values() for r in rs], records['p2'])
    return decision


def write_long(path, rows, rec):
    repairs = {(r['repair']['arm'], r['repair']['tier'], r['repair']['step'], r['pair']): r['repair'] for r in rows if r['repair'] is not None}
    fields = {(r['field']['arm'], r['field']['tier'], r['field']['step'], r['pair']): r['field'] for r in rows if r['field'] is not None}
    rec.write_long(path, fields, repairs)


# ---- P3 -------------------------------------------------------------------------------------------------------------------------------------
def p3_arm_choice(p2_root, *, recompute=True):
    """What P3 may run, from P2's reading: dict(arm, ...) or a reason why not.  The arm P2 chose, its decision file's sha256, P2's commit and segmenter.  recompute (revision 2, the review's
    minor 7; the queue and the dry run): the file's code must be P2's batch commit, and decide_p2 on P2's records now gives the same reading and the same candidate."""
    path = Path(p2_root)/'analysis'/'decision_p2.json'
    document, batch = read_json(path), read_json(Path(p2_root)/'batch.json')
    if document is None or batch is None:
        return None, f'等 P2：没有 {path} 或 batch.json'
    if document.get('reading') != READ_CANDIDATE or document.get('candidate') not in ORDER:
        return None, f"P2 的读数是 {document.get('reading')}：P3 不跑"
    if recompute:
        if (document.get('code') or {}).get('batch_code_sha') != batch.get('code_sha'):
            return None, f"{path} was written for the commit {(document.get('code') or {}).get('batch_code_sha')}, P2 ran at {batch.get('code_sha')}: P3 does not start"
        again, _ = decide_p2(p2_root)
        if (again.get('reading'), again.get('candidate')) != (document['reading'], document['candidate']):
            return None, f"P2's records now read {again.get('reading')} {again.get('candidate')}, the file {document['reading']} {document['candidate']}: P3 does not start"
    return dict(arm=document['candidate'], p2_decision_sha256=core.sha_file(path), p2_code_sha=batch.get('code_sha'), segmenter_id=batch.get('segmenter_id'), p2_reading=document['reading'], recomputed=recompute), None


def ensure_p3_arm(root, p2_root, *, recompute=True):
    """The arm, written once into analysis/p3_arm.json; a second call reads it back (a file that changed stops whoever asks)."""
    existing = p2.P3.read_selection(root, p2.P3.ARM_FILE)
    if existing is not None:
        return existing, None
    choice, why = p3_arm_choice(p2_root, recompute=recompute)
    if choice is not None:
        p2.P3.write_selection(root, p2.P3.ARM_FILE, choice)
    return choice, why


def block(values, pairs):
    return p1.block_interval(values, [ALL_PAIRS.index(p) for p in pairs])


def read_p3(cand, ref, problems=()):
    """(reading, reason, numbers) from {pair: (source, metrics, why)} of the candidate and of the reference."""
    def incomplete(value):                      # revision 2 (the review's minor 8): seg35, seg4 and the official SDlogJ on every pair, or no reading
        return value is None or any(value.get(m) is None or not math.isfinite(value[m]) for m in p2.METHOD_KEYS)
    missing = [f'{core.short_name(p)}（{cand[p][2]}）' if cand[p][1] is None else core.short_name(p) for p in ALL_PAIRS if incomplete(cand[p][1])]+[f'参照 {core.short_name(p)}' for p in ALL_PAIRS if incomplete(ref[p][1])]
    if problems or missing:
        return UNDECIDABLE, '；'.join(list(problems)+([f"没有完整的方法级值（{'、'.join(p2.METHOD_KEYS)}）：{'、'.join(missing)}"] if missing else [])), None
    d = {m: [cand[p][1][m]-ref[p][1][m] for p in ALL_PAIRS] for m in ('seg35', 'seg4')}
    seg35 = np.asarray(d['seg35'], dtype=np.float64)
    interval = block(seg35, ALL_PAIRS)
    sdlogj = mean(cand[p][1].get('sdlogj_official') for p in ALL_PAIRS)
    numbers = dict(n=len(ALL_PAIRS), mean_delta_seg35=float(seg35.mean()), positive_seg35=int((seg35 > 0).sum()), block_seg35=interval, mean_sdlogj_official=sdlogj, method_seg35=mean(cand[p][1]['seg35'] for p in ALL_PAIRS), method_seg4=mean(cand[p][1].get('seg4') for p in ALL_PAIRS),
                   reference_seg35=mean(ref[p][1]['seg35'] for p in ALL_PAIRS), reference_seg4=mean(ref[p][1].get('seg4') for p in ALL_PAIRS))
    seg4 = np.asarray(d['seg4'], dtype=np.float64)
    numbers.update(mean_delta_seg4=float(seg4.mean()), positive_seg4=int((seg4 > 0).sum()), block_seg4=block(seg4, ALL_PAIRS))
    lines = dict(lower=interval['lo'] > p2.P3_LO, mean=numbers['mean_delta_seg35'] >= p2.P3_MEAN, sdlogj=sdlogj is not None and sdlogj <= p2.SDLOGJ_MAX)
    numbers['lines'] = dict(lines, lo_line=p2.P3_LO, mean_line=p2.P3_MEAN, sdlogj_max=p2.SDLOGJ_MAX)
    numbers['superior'] = interval['lo'] > p2.SUPERIOR_LO
    if all(lines.values()):
        return READ_GAIN, (SUPERIOR if numbers['superior'] else None), numbers
    return READ_NO_GAIN, '线：' + '；'.join(k for k, ok in lines.items() if not ok) + '；回到 1′(i)', numbers


def decide_p3(root, stops=None, results=None):
    root = Path(root)
    segmenter, reference = batch_reference(root, p2.P3)
    choice = p2.P3.read_selection(root, p2.P3.ARM_FILE)
    if reference is None or choice is None:
        return dict(schema='bayesreg.p3.decision.v1', reading=UNDECIDABLE, reason='没有 batch.json 或 p3_arm.json：什么都还没跑'), None
    arm = choice['arm']
    rows = [arm_row('p3', root, arm, pair) for pair in ALL_PAIRS]
    bad = sorted(r['repair']['run_id'] for r in rows if r['repair'] is not None and r['repair'].get('metrics') and r['repair'].get('oracle') is not False)
    if bad:
        raise SystemExit(f'repair.json is not oracle false for {bad}; nothing written')
    ref = reference_rows(reference, 'p3', segmenter)
    fallback = fallback_rows(root, ALL_PAIRS)
    identity = identity_rows(root, ALL_PAIRS)
    cand = {r['pair']: method_value(r, fallback[r['pair']], identity[r['pair']]) for r in rows}
    refv = {p: (ref[p]['source'], ref[p]['value'], ref[p]['why']) for p in ALL_PAIRS}       # revision 2 (I2 (b)): the reference by the same rule, its fallbacks HS's recorded values
    problems = [f"{r['short']} {r['why']}" for r in rows if r['kind'] == NOT_READ]+[f"参照 {r['short']} {r['why']}" for r in ref.values() if r['kind'] == NOT_READ]
    rejudge = [dict(pair=r['short'], why=r['why']) for r in rows if r.get('needs_lead_rejudgment')]
    reading, reason, numbers = read_p3(cand, refv, problems)
    sources = {core.short_name(p): cand[p][0] for p in ALL_PAIRS}
    em = [r['bayes'].get('em') for r in rows if r['bayes'].get('em')]
    tissue = [r['bayes'].get('tissue') for r in rows if r['bayes'].get('tissue')]
    distance = dict(p1=P1_JUDGMENT, em_d_to_q3h_final_median=None if not em else float(np.median([e['d_to_q3h_final'] for e in em])), em_d_line_of_p1=p1.A_NULL,
                    tissue_ratios_last_used_median=None if not tissue else {k: float(np.median([t['ratios_used'][-1][k] for t in tissue if t['ratios_used'][-1]])) for k in p2.C_CLASSES},
                    tissue_null_band_of_p1=list(p1.NULL_BAND), note='只报告，不进读数：P1 已判非空操作')
    certified = [r for r in rows if r['kind'] == CERTIFIED]
    return dict(schema='bayesreg.p3.decision.v1', code=code_record(root), is_conclusion=False, next_step='交总负责人与 BZ；没有收益：回到 1′(i)', oracle=False, note=NOTE, arm=arm, choice=choice, segmenter_id=segmenter, reference=reference['p3'].get('source'),
                reading=reading, superior=None if numbers is None else numbers['superior'], reason=reason, needs_lead_rejudgment=rejudge, numbers=numbers, arms_stopped=stops or {}, sources=sources,
                reference_fallbacks=[dict(pair=r['short'], line_status=r['line_status']) for r in ref.values() if r['kind'] == FALLBACK],
                n_certified=len(certified), fallbacks=[dict(pair=core.short_name(p), source=cand[p][0], why=cand[p][2]) for p in ALL_PAIRS if cand[p][0] != REPAIR], fuse_hits=[r['short'] for r in certified if r['repair'].get('wall_hit')],
                repaired_only=dict(n=len(certified), **{m: mean(r['repair']['after'].get(m) for r in certified) for m in MEASURES}), mean_task_clock=mean(r['task_clock'] for r in certified), distance=distance,
                per_pair=[dict(pair_line_record(r, ref[r['pair']]), source=cand[r['pair']][0], method_seg35=None if cand[r['pair']][1] is None else cand[r['pair']][1]['seg35'], reference_seg35=None if refv[r['pair']][1] is None else refv[r['pair']][1]['seg35']) for r in rows],
                reference_pairs=[dict(pair=r['short'], kind=r['kind'], line_status=r['line_status'], why=r['why'], directory=r['directory'], pinned=r['pinned']) for r in ref.values()]), rows


def summary_p3(d):
    L = ['# P3 读数（由 decide_p2.py 写出）', '', f"**读数：{d['reading']}**" + (f"（{SUPERIOR}）" if d.get('superior') and d['reading'] == READ_GAIN else '') + '（不是结论）', '', d.get('note', ''), '']
    if d.get('reason'):
        L += [f"原因 / 附注：{d['reason']}", '']
    if d.get('needs_lead_rejudgment'):
        L += [f"**要总负责人改判**（{REJUDGE_KINDS}；读数照常给出，不自动重跑）：" + '；'.join(f"{x['pair']}：{x['why']}" for x in d['needs_lead_rejudgment']), '']
    n = d.get('numbers')
    if n:
        b = n['block_seg35']
        L += [f"臂 {d['arm']}，分割器 {d['segmenter_id']}；参照：{d['reference']}", '',
              f"方法级（含退路）19 对：seg35 / seg4 {_f(n['method_seg35'])} / {_f(n['method_seg4'])}；参照 {_f(n['reference_seg35'])} / {_f(n['reference_seg4'])}。",
              f"Δseg35 均值 {_signed(n['mean_delta_seg35'])}（{n['positive_seg35']}/19 为正），受试者块 95% 区间 [{_signed(b['lo'])}, {_signed(b['hi'])}]；线：下界 > {p2.P3_LO}、均值 ≥ {p2.P3_MEAN}。"
              + ('' if n.get('block_seg4') is None else f" Δseg4 均值 {_signed(n['mean_delta_seg4'])}（{n['positive_seg4']}/19），区间 [{_signed(n['block_seg4']['lo'])}, {_signed(n['block_seg4']['hi'])}]（只报告）。"),
              f"官方 SDlogJ 均值（方法级）{_f(n['mean_sdlogj_official'])}，线 ≤ {p2.SDLOGJ_MAX}。", '']
    r = d.get('repaired_only') or {}
    L += [f"只算过证书的 {r.get('n')} 对：seg35 / seg4 {_f(r.get('seg35'))} / {_f(r.get('seg4'))}，官方 / 脑内 SDlogJ {_f(r.get('sdlogj_official'))} / {_f(r.get('sdlogj_brain'))}，HD95 {_f(r.get('hd95'), 3)}，小六个 {_f(r.get('small_six'))}，大六个 {_f(r.get('large_six'))}，皮层 {_f(r.get('cortex'))}，白质 {_f(r.get('white_matter'))}；任务时钟平均 {_f(d.get('mean_task_clock'), 1)} s。",
          f"退路：{d.get('fallbacks') or '无'}；参照走退路的对：{d.get('reference_fallbacks') or '无'}；撞保险丝：{'、'.join(d.get('fuse_hits') or []) or '无'}。", '', f"估出的值离参照值（只报告）：{d.get('distance')}", '',
          '## 它证伪不了什么', '', '开发集 19 对单次；前 5 对已多次用于选择。"不劣"只说明这一处贝叶斯化在 −.003 的界内不伤 seg35，不说明它提高 Dice，也不说明留出集上的表现。不是 F_B，没有经认证的下降。进方法先改合同，由 BZ 定。']
    return '\n'.join(L)+'\n'


def write_p3(root, stops=None, results=None):
    decision, rows = decide_p3(root, stops, results)
    out = Path(root)/'analysis'
    set_aside(out, ('decision_p3.json', 'summary_p3.md', 'long.csv'))
    (out/'decision_p3.json').write_text(json.dumps(decision, indent=1, ensure_ascii=False, default=_plain)+'\n')
    (out/'summary_p3.md').write_text(summary_p3(decision), encoding='utf-8')
    if rows:
        write_long(out/'long.csv', rows, records['p3'])
    return decision


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(type(value))
