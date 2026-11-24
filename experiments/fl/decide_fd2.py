"""FD2's reading (runs/20261105-fd2/任务书.md, plan.md, by code; a label for the lead and BZ, not a conclusion): analysis/decision_fd2.json, summary_fd2.md, long.csv; the code's choice analysis/repair_selection_1.json.

The choice.  Among Hx8, Hx12, Hx16, Hx24 at 600 steps (5/5 read, alpha = 1, uncertified): seg35 strictly above fd_v1's Hx6 at 600 steps, not repaired, and official SDlogJ <= .10.  All of them, in the order of the D factor.  The queue repairs them in that order and stops at the first arm that does not reach 5/5 certificates; the arms after it are named, not repaired.
The premise of a repaired arm, every number from the evaluation unit: 5/5 pass the certificate at alpha = 1, mean official SDlogJ <= .10, on every pair the task clock (channels + T88 + repair with its warm-up) <= 330 s.  Whether the clock is <= 90 s (BZ's line) is a column, not a premise.
Delta = the best arm that satisfies the premise (repaired) - fd_v1's Hx6 at 600 steps (repaired), seg35, paired, the mean.  The brief's four lines, in this order:
    the code has not chosen, the control not bit for bit, a cell not run or not evaluated, the reference not read                                     不可判 (not settled; a rule of the tool, as FD's)
    no arm through the gate                                                                                                                             到顶了; 不可判 if a run failed
    a repair did not reach 5/5, and no arm below it has a gain (satisfies the premise with Delta > +.002; none below counts as none)                    到顶了（修复撑不住）, written out; 不可判 if a run failed or a repair process crashed
    a repair did not reach 5/5, some arm below it has a gain, and arms through the gate were left without a repair                                     不可判, those arms named
    otherwise, on the best arm that satisfies the premise: Delta >= +.005 and at least 4 of 5 positive  还在涨;  Delta <= +.002  到顶了, 不可判 if a run failed or a repair process crashed;  else  中间带
    every arm repaired 5/5 and none satisfies the premise                                                                                                不可判 (a rule of the tool: the brief's lines have no Delta here)
What is a failed run: a field of the four arms with a status other than done or nonfinite, a stopped arm's cells, a repair process that crashed.  A repair that ran, was evaluated and did not pass is a result.  还在涨 and 中间带 stand whatever failed; the failed cells are written.
The best arm that satisfies the premise, its Delta, better_than_hx6 and the seg4 note are on every return when there is such an arm, whatever the reading: the lead decides the other 14 pairs on them.
Hx6L is only described: not in the choice, not in the reading, its failures change nothing.
A cell is read only with status done.  Everything else is only described.

The functions on the records are decide_fl's, not copied: another instance of that module whose `fl` is fd2_core.  decide_fl and decide_fd themselves are not touched, and decide_fd reads the reference out of fd_v1.
"""
import importlib.util
import json
from pathlib import Path
import statistics
import sys

from decide_ef import read_json, set_aside
import decide_fd as fd_reading
import decide_fl as fl_reading
import fd2_core as fd2
import fd_core as fd
import fl_core as fl
import mc0_core as core

SCHEMA = 'bayesreg.fd2.decision.v1'
N = len(core.PAIRS)
BAR_RISING, BAR_TOP, N_POSITIVE, SDLOGJ_MAX = .005, .002, 4, .10
STEP = fd2.NATIVE
REFERENCE_ARM = 'Hx6'
MEASURES = fl_reading.MEASURES
RESULT_STATUSES = ('done', 'nonfinite')
READ_RISING, READ_TOP, READ_MIDDLE, READ_UNDECIDABLE, READ_TOP_REPAIR = '还在涨', '到顶了', '中间带', '不可判', '到顶了（修复撑不住）'
NOTE = ('开发集单次观测，通道来自 unet35_v1_ens 的预测（oracle: false），真值标签只进评测。**不是 F_B，没有经认证的下降，只有终态过证书；不是方法，不进结果表，进方法先改合同，由 BZ 定。** n = 5，前 5 对之前选配用过，从 4 个新臂里取最好，要确认得看其余 14 对。')


def _batch_instance():
    """decide_fl once more under another name, with `fl` = fd2_core: its functions on the records read this batch's plan.  sys.modules['decide_fl'] stays FL's, and FD's instance stays FD's."""
    spec = importlib.util.spec_from_file_location('_fd2_decide_fl', Path(fl_reading.__file__))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.fl = fd2
    return module


records = _batch_instance()


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


# ---- fd_v1, read only ------------------------------------------------------------------------------------------------------------------------------
def reference_fields(results=None):
    """{(Hx6, first5, 600, pair): row} of fd_v1, by FD's own instance."""
    root = fd2.fd_root(results)
    fields = {}
    for pair in core.PAIRS:
        for step, row in fd_reading.records.field_rows(root, fd.Cell(REFERENCE_ARM, fd.FIRST, pair)).items():
            fields[(REFERENCE_ARM, fd.FIRST, step, pair)] = row
    return fields


def reference_repairs(results=None):
    """{(Hx6, first5, 600, pair): row} of fd_v1's RHx6, by FD's own instance."""
    root = fd2.fd_root(results)
    out = {}
    for pair in core.PAIRS:
        row = fd_reading.records.repair_row(root, fd.Cell('R'+REFERENCE_ARM, f'c{fd.NATIVE}', pair))
        if row is not None:
            out[(REFERENCE_ARM, fd.FIRST, fd.NATIVE, pair)] = row
    return out


def reference_row(references):
    return fd_reading.records.arm_step(references, REFERENCE_ARM, STEP)


# ---- the choice ----------------------------------------------------------------------------------------------------------------------------------------
def select_repairs(fields, references):
    """The rule of the module docstring.  The bar is the mean of fd_v1's Hx6 at 600 steps, not repaired, not rounded; no bar when it is not read 5/5, and then nothing is chosen."""
    bar = reference_row(references)
    rows = []
    for arm in fd2.NEW_ARMS:
        r = records.arm_step(fields, arm, STEP)
        rows.append(dict(arm=arm, step=STEP, D_factor=fd2.ARMS[arm]['D_factor'], n_read=r['n_read'], seg35=r['seg35'], seg4=r['seg4'], sdlogj_official=r['sdlogj_official'], passes=bool(bar['complete'] and r['complete'] and r['seg35'] > bar['seg35'] and r['sdlogj_official'] <= SDLOGJ_MAX)))
    chosen = [dict(arm=r['arm'], step=r['step']) for r in rows if r['passes']]
    return dict(schema=SCHEMA+'.repair_selection', rule=f'Hx8、Hx12、Hx16、Hx24 在 {STEP} 步，5/5 读到，α = 1 seg35 严格高于 fd_v1 的 Hx6@{STEP} 未修复均值且官方 SDlogJ ≤ {SDLOGJ_MAX}；都修复，按 D 因子从小到大，一个臂没做到 5/5 证书就不再修更高因子的臂', bar_hx6_seg35=bar['seg35'], bar_n_read=bar['n_read'], candidates=rows, chosen=chosen, eps=fd2.REPAIR_EPS)


def ensure_repair_selection(root, references=None):
    """The code's choice, written once."""
    references = reference_fields() if references is None else references
    return records.ensure_selection(root, 'repair_selection_1.json', lambda fields, repairs: select_repairs(fields, references))


# ---- how far the repairs went ---------------------------------------------------------------------------------------------------------------------------------
def repair_progress(repairs, chosen, stops):
    """The repairs in the order of the choice, up to the first arm that did not reach 5/5.  dict(passed: arms at 5/5, failed: that arm or None, left: the arms after it, missing: what has not run or not been evaluated, which is not settled, as the queue goes on from there)."""
    passed, failed, left, missing = [], None, [], []
    arms = [c['arm'] for c in chosen]
    for i, arm in enumerate(arms):
        rows = {pair: repairs.get((arm, fd2.FIRST, STEP, pair)) for pair in core.PAIRS}
        not_run = [core.short_name(p) for p, r in rows.items() if r is None]
        not_evaluated = [core.short_name(p) for p, r in rows.items() if r is not None and not r['metrics'] and r['status'] != 'failed']
        if (not_run and 'R'+arm not in stops) or not_evaluated:
            missing += [f'修复 {arm} {short} 没有运行' for short in not_run]+[f'修复 {arm} {short} 没有评测' for short in not_evaluated]
            break
        if sum(bool(r is not None and r['certified_at_1']) for r in rows.values()) == N:
            passed.append(arm)
        else:
            failed, left = arm, arms[i+1:]
            break
    return dict(passed=passed, failed=failed, left=left, missing=missing)


def arm_reached_5_of_5(root, arm):
    """What the queue asks after an arm's evaluation: the evaluation unit's certificates, from its records."""
    _, repairs = records.collect(root)
    return records.repair_summary(repairs, arm, STEP)['n_certified_at_1'] == N


def not_read(fields, repairs, progress, stops):
    """The failures that say nothing about the D factor (the module docstring), each with its state.  Hx6L is not here: it is only described."""
    out = []
    for arm in fd2.NEW_ARMS:
        for pair in core.PAIRS:
            row = fields.get((arm, fd2.FIRST, STEP, pair))
            if row is None:
                if arm in stops:
                    out.append(f'{arm} {core.short_name(pair)}：臂被停，没有运行')
            elif row['status'] not in RESULT_STATUSES:
                out.append(f"{arm} {core.short_name(pair)}：{row['status']}")
    for arm in progress['passed']+([progress['failed']] if progress['failed'] else []):
        for pair in core.PAIRS:
            row = repairs.get((arm, fd2.FIRST, STEP, pair))
            if row is None:
                if 'R'+arm in stops:
                    out.append(f'修复 {arm} {core.short_name(pair)}：臂被停，没有运行')
            elif not row['metrics']:
                out.append(f"修复 {arm} {core.short_name(pair)}：{row['status']}")
    return out


def not_settled(fields, references, reference_rep, progress, selected, stops, control_same=True):
    """What is missing before the reading can be read; [] when it is settled."""
    if not selected:
        return ['中间读数：代码还没有选臂']
    missing = []
    if control_same is not True:
        missing.append(f'Hx6 控制没有过（u 对 fd_v1 的 Hx6@{STEP}：{control_same}）')
    if not reference_row(references)['complete']:
        missing.append(f"fd_v1 的 Hx6@{STEP} 只读到 {reference_row(references)['n_read']}/{N}")
    certified = sum(bool(r['metrics'] and r['certified_at_1']) for r in reference_rep.values())
    if certified < N:
        missing.append(f'fd_v1 的 RHx6 修复后证书只有 {certified}/{N}')
    for arm in fd2.NEW_ARMS:
        for pair in core.PAIRS:
            row = fields.get((arm, fd2.FIRST, STEP, pair))
            if row is None:
                if arm not in stops:
                    missing.append(f'{arm} {core.short_name(pair)} 没有运行')
            elif row['status'] == 'done' and row['metrics'] is None:
                missing.append(f'{arm} {core.short_name(pair)} 没有评测')
    return missing+progress['missing']


def paired_with_reference(repairs, reference_rep, arm, measures=('seg35', 'seg4', 'cortex', 'small_six', 'large_six', 'other_23', 'white_matter', 'hd95', 'sdlogj_official', 'sdlogj_brain')):
    """The repaired arm - fd_v1's repaired Hx6, pair by pair.  Positive is strictly > 0."""
    out = {}
    for m in measures:
        diffs = {}
        for pair in core.PAIRS:
            a, b = repairs.get((arm, fd2.FIRST, STEP, pair)), reference_rep.get((REFERENCE_ARM, fd.FIRST, fd.NATIVE, pair))
            if a and b and a['metrics'] and b['metrics'] and a['after'].get(m) is not None and b['after'].get(m) is not None:
                diffs[core.short_name(pair)] = a['after'][m]-b['after'][m]
        out[m] = dict(mean=mean(diffs.values()), positive=sum(d > 0 for d in diffs.values()), n=len(diffs), per_pair=diffs)
    return out


def bz_column(rows):
    """Task clock <= 90 s, BZ's line: how many pairs.  Only described."""
    clocks = [r['task_clock'] for r in rows if r is not None and r.get('task_clock') is not None]
    return dict(n_within=sum(c <= fd2.BZ_SECONDS for c in clocks), n=len(clocks), max=max(clocks, default=None), mean=mean(clocks))


def how_it_failed(repairs, arm):
    """The words of an arm short of 5/5: how many pairs passed, how many hit the wall (reason wall: time, not a repair that cannot be solved)."""
    rows = [repairs.get((arm, fd2.FIRST, STEP, pair)) for pair in core.PAIRS]
    rows = [r for r in rows if r is not None]
    passed = sum(bool(r['certified_at_1']) for r in rows)
    wall = [r for r in rows if r.get('reason') == 'wall']
    return f"证书 {passed}/{N}；其中 {len(wall)} 对撞 {fd2.REPAIR_WALL_SECONDS:g} s 墙（reason 是 wall），证书过的 {sum(bool(r['certified_at_1']) for r in wall)} 对" if wall else f'证书 {passed}/{N}；没有一对撞 {fd2.REPAIR_WALL_SECONDS:g} s 墙'


def read_all(fields, repairs, references, reference_rep, chosen, selected=True, stops=None, control_same=True):
    """The rule of the module docstring."""
    stops = stops or {}
    progress = repair_progress(repairs, chosen, stops)
    tried = progress['passed']+([progress['failed']] if progress['failed'] else [])
    rows = [records.repair_summary(repairs, arm, STEP) for arm in tried]
    good = [r for r in rows if r['satisfies_premise']]
    n_read = sum(records.arm_step(fields, arm, STEP)['complete'] for arm in fd2.NEW_ARMS)
    effect = dict(arms_read=n_read, through_the_gate=len(chosen), repaired=len(rows), satisfying_the_premise=len(good), text=f'最好的臂是从 {len(fd2.NEW_ARMS)} 个新臂里取的（{n_read} 个 5/5 读到）：{len(chosen)} 个过修复门，{len(rows)} 个做了修复，{len(good)} 个满足前提。n = {N}，前 5 对之前选配用过。')

    # The best arm that satisfies the premise, its Delta and the seg4 note: on every return, whatever the reading is (the lead decides the other 14 pairs on them)
    deltas = {r['arm']: paired_with_reference(repairs, reference_rep, r['arm']) for r in good}
    best_row, note = None, None
    if good:
        best = max(good, key=lambda r: r['seg35'])
        delta = deltas[best['arm']]
        d35, d4 = delta['seg35']['mean'], delta['seg4']['mean']
        note = None if d4 is None or d4 >= 0 else f"seg4 下降：{best['arm']} 对 Hx6@{STEP} {d4:+.4f}（{delta['seg4']['positive']}/{delta['seg4']['n']} 为正）。判定只用 seg35。"
        best_row = dict(arm=best['arm'], step=best['step'], D_factor=fd2.ARMS[best['arm']]['D_factor'], seg35=best['seg35'], seg4=best['seg4'], sdlogj_official=best['sdlogj_official'], max_task_clock=best['max_task_clock'], delta=delta, better_than_hx6=None if d35 is None else d35 > 0)

    def out(reading, reason=None, failed=()):
        return dict(reading=reading, reason=reason, best=best_row, premise=rows, selection_effect=effect, seg4_note=note, not_read=list(failed), left_without_repair=progress['left'], first_failed_repair=progress['failed'], repaired_5_of_5=progress['passed'])
    missing = not_settled(fields, references, reference_rep, progress, selected, stops, control_same)
    if missing:
        return out(READ_UNDECIDABLE, '；'.join(missing))
    failed = not_read(fields, repairs, progress, stops)
    failed_reason = f"这些格的失败和 D 因子无关，算没读到，不能读成到顶了：{'、'.join(failed)}"
    if not chosen:
        if failed:
            return out(READ_UNDECIDABLE, failed_reason, failed=failed)
        return out(READ_TOP, f'没有臂过修复门（α = 1 seg35 严格高于 fd_v1 的 Hx6@{STEP} 未修复均值且官方 SDlogJ ≤ {SDLOGJ_MAX}）')
    if progress['failed'] is not None:
        below = [r['arm'] for r in good if r['arm'] in progress['passed']]
        gains = [arm for arm in below if deltas[arm]['seg35']['mean'] is not None and deltas[arm]['seg35']['mean'] > BAR_TOP]
        left = f"，没有修复的臂：{'、'.join(progress['left'])}" if progress['left'] else ''
        how = how_it_failed(repairs, progress['failed'])
        if not gains:
            if failed:
                return out(READ_UNDECIDABLE, failed_reason, failed=failed)
            return out(READ_TOP_REPAIR, f"修复撑不住：{progress['failed']} 没有做到 5/5 证书（{how}）{left}；它以下的臂没有增益（Δ seg35 都不超过 +{BAR_TOP}，或没有）", failed=failed)
        if progress['left']:
            return out(READ_UNDECIDABLE, f"{progress['failed']} 修复没有做到 5/5 证书（{how}）{left}；它以下有增益的臂：{'、'.join(gains)}", failed=failed)
    if not good:
        return out(READ_UNDECIDABLE, '所有臂都修到 5/5，但没有臂满足前提：'+'；'.join(f"{r['arm']}@{r['step']}：{'、'.join(r['missed'])}" for r in rows), failed=failed)
    positive = delta['seg35']['positive']
    reason = None
    if d35 >= BAR_RISING and positive >= N_POSITIVE:
        reading = READ_RISING
    elif d35 <= BAR_TOP:
        reading = READ_UNDECIDABLE if failed else READ_TOP
        reason = failed_reason if failed else None
    else:
        reading = READ_MIDDLE
    return out(reading, reason, failed)


# ---- controls (records) ---------------------------------------------------------------------------------------------------------------------------------------
def fd_hash(pair, results=None):
    return ((read_json(fd.run_dir(fd2.fd_root(results), fd.Cell(fd2.CONTROL_ARM, fd.FIRST, pair))/'run.json') or {}).get('checkpoints') or {}).get(str(STEP), {}).get('u_sha256')


def my_hash(root, arm, pair):
    return ((read_json(fd2.run_dir(root, fd2.Cell(arm, fd2.FIRST, pair))/'run.json') or {}).get('checkpoints') or {}).get(str(STEP), {}).get('u_sha256')


def control_of_hx6(root, results=None):
    """(same, mine, theirs): the control's u at 600 steps against fd_v1's Hx6.  same None when a side has no record."""
    mine, theirs = my_hash(root, fd2.CONTROL_ARM, fd2.CONTROL_PAIR), fd_hash(fd2.CONTROL_PAIR, results)
    return (None if mine is None or theirs is None else mine == theirs), mine, theirs


def controls(root, results=None):
    """The control; Hx6L's 600 steps against fd_v1, pair by pair; the scale: an arm's D multiplier over fd_v1's Hx6 of the same pair is the ratio of the factors.  True, False, or None when a side has no record.  Records only: Hx6L's stop nothing."""
    same, mine, theirs = control_of_hx6(root, results)
    out = dict(hx6_control=dict(same=same, mine=mine, theirs=theirs), hx6l_600={}, scale={})
    for pair in core.PAIRS:
        a, b = my_hash(root, fd2.DESCRIBED_ARM, pair), fd_hash(pair, results)
        out['hx6l_600'][core.short_name(pair)] = None if a is None or b is None else a == b
        reference = (read_json(fd.run_dir(fd2.fd_root(results), fd.Cell(fd2.CONTROL_ARM, fd.FIRST, pair))/'run.json') or {}).get('scale') or {}
        row = {}
        for arm in fd2.NEW_ARMS:
            scale = (read_json(fd2.run_dir(root, fd2.Cell(arm, fd2.FIRST, pair))/'run.json') or {}).get('scale') or {}
            ratio = fd2.ARMS[arm]['D_factor']/fd2.ARMS[fd2.CONTROL_ARM]['D_factor']
            row[arm] = dict(D_multiplier_is_ratio_times_Hx6=None if scale.get('D_multiplier') is None or reference.get('D_multiplier') is None else abs(scale['D_multiplier']/(ratio*reference['D_multiplier'])-1) <= 1e-12, g_norm_Q0_equal_Hx6=None if scale.get('g_norm_Q0') is None or reference.get('g_norm_Q0') is None else scale['g_norm_Q0'] == reference['g_norm_Q0'])
        out['scale'][core.short_name(pair)] = row
    return out


# ---- the description ---------------------------------------------------------------------------------------------------------------------------------------------
def versus(fields, references, arm, measures=('seg35', 'seg4', 'cortex', 'small_six', 'sdlogj_official', 'sdlogj_brain')):
    """arm at 600 steps (alpha = 1, not repaired) - fd_v1's Hx6 at 600 steps, paired; only described."""
    out = dict(arm=arm)
    for m in measures:
        diffs = []
        for pair in core.PAIRS:
            a, b = fields.get((arm, fd2.FIRST, STEP, pair)), references.get((REFERENCE_ARM, fd.FIRST, STEP, pair))
            if a and b and a['metrics'] and b['metrics'] and a['status'] == 'done' and b['status'] == 'done' and a['metrics'].get(m) is not None and b['metrics'].get(m) is not None:
                diffs.append(a['metrics'][m]-b['metrics'][m])
        out[m] = dict(mean=mean(diffs), positive=sum(d > 0 for d in diffs), n=len(diffs))
    return out


def describe(fields, repairs, references, reference_rep):
    table = [dict(records.arm_step(fields, arm, STEP), D_factor=fd2.ARMS[arm]['D_factor']) for arm in fd2.NEW_ARMS]
    reference = dict(reference_row(references), D_factor=fd2.ARMS[fd2.CONTROL_ARM]['D_factor'])
    long_rows = [dict(records.arm_step(fields, fd2.DESCRIBED_ARM, step), D_factor=fd2.ARMS[fd2.DESCRIBED_ARM]['D_factor']) for step in fd2.ARMS[fd2.DESCRIBED_ARM]['checkpoints']]
    long_delta = records.paired(fields, fd2.DESCRIBED_ARM, fd2.DESCRIBED_ARM, fd2.LONG, reference_step=STEP)
    reference_repaired = fd_reading.records.repair_summary(reference_rep, REFERENCE_ARM, fd.NATIVE)
    return dict(table=table, reference=reference, reference_repaired=dict(reference_repaired, bz_90=bz_column(reference_rep.values())), against_hx6=[versus(fields, references, arm) for arm in fd2.NEW_ARMS], hx6l=long_rows, hx6l_1200_minus_600=long_delta)


def code_record(root):
    """What wrote this reading: the sha256 of the files that hold its rules, and the batch's commit."""
    return dict(decide_fd2_sha256=core.sha_file(Path(__file__)), fd2_core_sha256=core.sha_file(Path(fd2.__file__)), decide_fd_sha256=core.sha_file(Path(fd_reading.__file__)), fd_core_sha256=core.sha_file(Path(fd.__file__)), decide_fl_sha256=core.sha_file(Path(fl_reading.__file__)), fl_core_sha256=core.sha_file(Path(fl.__file__)), batch_code_sha=(read_json(Path(root)/'batch.json') or {}).get('code_sha'))


def decide(root, stops=None, results=None):
    root = Path(root)
    fields, repairs = records.collect(root)
    bad = sorted(r['run_id'] for r in fields.values() if r['metrics'] is not None and r['oracle'] is not False)
    if bad:
        raise SystemExit(f'run.json is not oracle false for {bad}; nothing written')
    references, reference_rep = reference_fields(results), reference_repairs(results)
    chosen = fd2.chosen_repairs(root)
    selection = fd2.read_selection(root, 'repair_selection_1.json')
    result = read_all(fields, repairs, references, reference_rep, chosen, selected=selection is not None, stops=stops, control_same=control_of_hx6(root, results)[0])
    for row in result['premise']:
        row['bz_90'] = bz_column(repairs.get((row['arm'], fd2.FIRST, STEP, pair)) for pair in core.PAIRS)
    decision = dict(schema=SCHEMA, code=code_record(root), selections=fd2.selection_hashes(root), is_conclusion=False, next_step='交总负责人与 BZ', oracle=False, note=NOTE, arms_stopped=stops or {}, rules=dict(bar_rising=BAR_RISING, bar_top=BAR_TOP, n_positive=N_POSITIVE, sdlogj_max=SDLOGJ_MAX, task_budget_seconds=fd2.TASK_BUDGET_SECONDS, bz_seconds=fd2.BZ_SECONDS, n=N, repair_eps=fd2.REPAIR_EPS),
                    reading=result['reading'], reason=result['reason'], best=result['best'], premise=result['premise'], selection_effect=result['selection_effect'], seg4_note=result['seg4_note'], not_read=result['not_read'], left_without_repair=result['left_without_repair'], first_failed_repair=result['first_failed_repair'], repaired_5_of_5=result['repaired_5_of_5'], repair_selection=selection, description=describe(fields, repairs, references, reference_rep), rest14=records.describe_rest(fields, repairs, root), controls=controls(root, results))
    return decision, fields, repairs


# ---- the words --------------------------------------------------------------------------------------------------------------------------------------------------------
_f, _signed = fl_reading._f, fl_reading._signed


def _cell(d):
    return f"{_signed(d['mean'])}（{d['positive']}/{d['n']}）"


def _bz(b):
    return f"{b['n_within']}/{b['n']}"


def summary_text(decision):
    d = decision['description']
    best = decision['best']
    L = ['# FD2 读数（由 decide_fd2.py 写出）', '', f"**读数：{decision['reading']}**（不是结论，下一步交总负责人与 BZ）", '', NOTE, '']
    if best is not None:
        delta = best['delta']
        L += [f"最好的满足前提的臂：{best['arm']}（因子 {best['D_factor']:g}，{best['step']} 步），修复后 seg35 / seg4 {_f(best['seg35'])} / {_f(best['seg4'])}，官方 SDlogJ {_f(best['sdlogj_official'])}，任务时钟最长 {_f(best['max_task_clock'], 1)} s。Δ（对 fd_v1 的 Hx6@{STEP}，都修复后，配对）seg35 {_cell(delta['seg35'])}，seg4 {_cell(delta['seg4'])}，皮层 {_cell(delta['cortex'])}，小六个 {_cell(delta['small_six'])}，HD95 {_cell(delta['hd95'])}，官方 SDlogJ {_cell(delta['sdlogj_official'])}，脑内 SDlogJ {_cell(delta['sdlogj_brain'])}。比 Hx6 好：{best['better_than_hx6']}。", '']
    if decision['seg4_note']:
        L += [f"**{decision['seg4_note']}**", '']
    if decision['reason']:
        L += [f"原因：{decision['reason']}", '']
    if decision['not_read']:
        L += [f"**失败的格（算没读到，不算结果）：{'、'.join(decision['not_read'])}**", '']
    if decision['first_failed_repair']:
        L += [f"修复停在 {decision['first_failed_repair']}（没有做到 5/5 证书）；没有修复的过门臂：{'、'.join(decision['left_without_repair']) or '无'}", '']
    L += [decision['selection_effect']['text'], '']
    if decision['arms_stopped']:
        L += [f"停掉的臂：{json.dumps(decision['arms_stopped'], ensure_ascii=False)}", '']
    r = d['reference_repaired']
    L += ['## 前提（修复后，证书都由评测单元从场算；≤ 90 s 是 BZ 的口径，单独一列，不进前提）', '', '| 臂 | 满足前提 | α = 1 证书 | seg35 / seg4 修复前 → 修复后 | 官方 / 脑内 SDlogJ | HD95 | 小六个 | 皮层 | 任务时钟 平均 / 最长 s | 任务时钟 ≤ 90 s 的对 | 修复 平均 / 最长 s | 预热平均 s | 运行与评测不一致 | 差在哪 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|',
          f"| Hx6（fd_v1，对照） | {r['satisfies_premise']} | {r['n_certified_at_1']}/{N} | {_f(r['seg35_before'])} / {_f(r['seg4_before'])} → {_f(r['seg35'])} / {_f(r['seg4'])} | {_f(r['sdlogj_official'])} / {_f(r['sdlogj_brain'])} | {_f(r['hd95'], 3)} | {_f(r['small_six'])} | {_f(r['cortex'])} | {_f(r['mean_task_clock'], 1)} / {_f(r['max_task_clock'], 1)} | {_bz(r['bz_90'])} | {_f(r['mean_seconds_repair'], 1)} / {_f(r['max_seconds_repair'], 1)} | {_f(r['mean_seconds_warmup'], 1)} | {r['run_and_evaluation_disagree'] or '无'} | {'；'.join(r['missed']) or '—'} |"]
    for r in decision['premise']:
        L.append(f"| {r['arm']} | {r['satisfies_premise']} | {r['n_certified_at_1']}/{N} | {_f(r['seg35_before'])} / {_f(r['seg4_before'])} → {_f(r['seg35'])} / {_f(r['seg4'])} | {_f(r['sdlogj_official'])} / {_f(r['sdlogj_brain'])} | {_f(r['hd95'], 3)} | {_f(r['small_six'])} | {_f(r['cortex'])} | {_f(r['mean_task_clock'], 1)} / {_f(r['max_task_clock'], 1)} | {_bz(r['bz_90'])} | {_f(r['mean_seconds_repair'], 1)} / {_f(r['max_seconds_repair'], 1)} | {_f(r['mean_seconds_warmup'], 1)} | {r['run_and_evaluation_disagree'] or '无'} | {'；'.join(r['missed']) or '—'} |")
    L += ['', '## 只描述：D 因子曲线（α = 1，taper 后，未修复；不是通过证书的状态）', '', '| 臂 | 来源 | D 因子 | 读到 | seg35 / seg4 | HD95 | 官方 / 脑内 SDlogJ | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | 不可行单元 | 任务时钟 平均 / 最长 s | D | 几何 | floor |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for source, rows in ((fd2.FD_ROOT_NAME, [d['reference']]), ('本批', d['table'])):
        for r in rows:
            L.append(f"| {r['arm']} | {source} | {r['D_factor']:g} | {r['n_read']}/{N} | {_f(r['seg35'])} / {_f(r['seg4'])} | {_f(r['hd95'], 3)} | {_f(r['sdlogj_official'])} / {_f(r['sdlogj_brain'])} | {_f(r['small_six'])} | {_f(r['large_six'])} | {_f(r['other_23'])} | {_f(r['cortex'])} | {_f(r['white_matter'])} | {'、'.join(str(x) for x in r['n_infeasible'])} | {_f(r['seconds_task'], 1)} / {_f(r['max_seconds_task'], 1)} | {_f(r['data'], 1)} | {_f(r['geometry'], 1)} | {_f(r['floor'], 2)} |")
    L += ['', 'D、几何、floor 是该步之前最后一次求值，D 因子不同的臂 D 的尺度不同，不直接比。', '', f'## 只描述：对 fd_v1 的 Hx6@{STEP}（配对，未修复）', '', '| 臂 | seg35 | seg4 | 皮层 | 小六个 | 官方 SDlogJ | 脑内 SDlogJ |', '|---|---|---|---|---|---|---|']
    for r in d['against_hx6']:
        L.append(f"| {r['arm']} | {_cell(r['seg35'])} | {_cell(r['seg4'])} | {_cell(r['cortex'])} | {_cell(r['small_six'])} | {_cell(r['sdlogj_official'])} | {_cell(r['sdlogj_brain'])} |")
    L += ['', '## 只描述：修复（ε = .05）：秒数对不可行单元数', '', '| 臂 | 对 | 状态 | 证书 α = 1 | 不可行单元 前 → 后 | 修复秒 | 预热秒 | 任务时钟 | 轮数 | W 点 | 改动节点 | 最大位移变化 mm | 最小认证下界 | seg35 前 → 后 | seg4 前 → 后 | 结束原因 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in decision['premise']:
        for row in r['per_pair']:
            L.append(f"| {r['arm']} | {row['pair']} | {row['status']} | {row.get('certified_at_1')} | {row.get('n_infeasible_before')} → {row.get('n_infeasible_after')} | {_f(row.get('seconds_repair'), 1)} | {_f(row.get('seconds_warmup'), 1)} | {_f(row.get('task_clock'), 1)} | {row.get('rounds')} | {row.get('n_extra_points')} | {row.get('n_changed_nodes')} | {_f(row.get('max_change_mm'), 3)} | {_f(row.get('min_lower_bound_after'), 6)} | {_f(row.get('seg35_before'))} → {_f(row.get('seg35_after'))} | {_f(row.get('seg4_before'))} → {_f(row.get('seg4_after'))} | {row.get('reason')} |")
    delta = d['hx6l_1200_minus_600']
    L += ['', '## 只描述：Hx6L（因子 6，原生 1200 步；不进选择，不进读数）', '', '| 步 | 读到 | seg35 / seg4 | 官方 SDlogJ | 不可行单元 | 任务时钟 平均 / 最长 s |', '|---|---|---|---|---|---|']
    for r in d['hx6l']:
        L.append(f"| {r['step']} | {r['n_read']}/{N} | {_f(r['seg35'])} / {_f(r['seg4'])} | {_f(r['sdlogj_official'])} | {'、'.join(str(x) for x in r['n_infeasible'])} | {_f(r['seconds_task'], 1)} / {_f(r['max_seconds_task'], 1)} |")
    L += ['', f"1200 步 − 600 步（配对）：seg35 {_cell(delta['seg35'])}，seg4 {_cell(delta['seg4'])}，皮层 {_cell(delta['cortex'])}，小六个 {_cell(delta['small_six'])}，官方 SDlogJ {_cell(delta['sdlogj_official'])}。"]
    different = [short for short, same in decision['controls']['hx6l_600'].items() if same is False]
    if different:
        L += ['', f"**Hx6L 的 {STEP} 步对 fd_v1 的 Hx6@{STEP} 不逐位相同的：{'、'.join(different)}。这些对上 1200 − 600 的描述不再是 Hx6@{STEP} 的延伸。**"]
    if decision['rest14']:
        L += ['', '## 只描述：其余 14 对（总负责人单独启动；不改上面的读数）', '', '| 臂 | 步 | 场 读到 | α = 1 seg35 / seg4 | 修复 读到 | α = 1 证书 | 修复后 seg35 / seg4 | 官方 SDlogJ | 任务时钟 平均 / 最长 s |', '|---|---|---|---|---|---|---|---|---|']
        for r in decision['rest14']:
            a, b = r['fields'], r['repairs']
            L.append(f"| {r['arm']} | {r['step']} | {a['n_read']}/14 | {_f(a['seg35'])} / {_f(a['seg4'])} | {b['n_read']}/14 | {b['n_certified_at_1']}/14 | {_f(b['seg35'])} / {_f(b['seg4'])} | {_f(b['sdlogj_official'])} | {_f(b['mean_task_clock'], 1)} / {_f(b['max_task_clock'], 1)} |")
    c = decision['controls']
    code = decision['code']
    bad = [f'{short} {arm} {key}' for short, row in c['scale'].items() for arm, checks in row.items() for key, value in checks.items() if value is False]
    L += ['', '## 控制（只记录；Hx6 的控制不过时队列已经停了）', '', f"Hx6 控制（{core.short_name(fd2.CONTROL_PAIR)}，u 对 fd_v1 的 Hx6@{STEP}）逐位相同：{c['hx6_control']['same']}", '', f"Hx6L 的 {STEP} 步对 fd_v1 的 Hx6@{STEP}：{c['hx6l_600']}", '', f"尺度数字不符（D_multiplier = 因子比 × fd_v1 同对 Hx6 的，g_norm_Q0 相同）：{bad or '无'}（没有读到的不算不符）", '', f"读数的规则：{code}；选择文件 {decision['selections']}", '',
          '## 它证伪不了什么', '', '不是 F_B，没有经认证的下降，只有终态过证书；开发集单次；n = 5。工作点的差过不过标量原生层这里不量。进方法先改合同，由 BZ 定。']
    return '\n'.join(L)+'\n'


def write_analysis(root, stops=None, results=None):
    """analysis/: an earlier version moved to analysis/previous_<UTC>/, never overwritten."""
    root = Path(root)
    decision, fields, repairs = decide(root, stops, results)
    out = root/'analysis'
    set_aside(out, ('decision_fd2.json', 'long.csv', 'summary_fd2.md'))
    (out/'decision_fd2.json').write_text(json.dumps(decision, indent=1, ensure_ascii=False)+'\n')
    (out/'summary_fd2.md').write_text(summary_text(decision), encoding='utf-8')
    records.write_long(out/'long.csv', fields, repairs)
    return decision
