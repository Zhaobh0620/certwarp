"""FD's reading (runs/20261105-fd/任务书.md, plan.md, by code; a label for the lead and BZ, not a conclusion): analysis/decision_fd.json, summary_fd.md, long.csv; the code's choice analysis/repair_selection_1.json.

The choice.  Among Hx3, Hx4, Hx6, Fx2, Fx4 at 600 steps (5/5 read, alpha = 1, uncertified): seg35 strictly above fl_v1's Hx at 600 steps and official SDlogJ <= .10, the 2 highest; ties go to the plan's order.  Hx itself is not repaired.
The premise of a repaired arm, every number from the evaluation unit: 5/5 pass the certificate at alpha = 1, mean official SDlogJ <= .10, on every pair the task clock (channels + T88 + repair with its warm-up) <= 330 s.
The reading.  Delta = the best arm that satisfies the premise (repaired) - fl_v1's Hx at 600 steps (alpha = 1), seg35, paired, the mean:
    Delta >= +.005 and at least 4 of 5 positive    工作点是杠杆
    Delta <= +.002, or no arm satisfies the premise  工作点不是杠杆
    otherwise                                      中间带
Not settled (rule added by the tool, plan.md): the code has not chosen yet; the control not bit for bit; a cell of the plan that has not run, unless its arm was stopped; a finished run or a chosen repair without its evaluation; the reference not read 5/5: 不可判, with what is missing.  Not read is not "no arm".
What is a result (the lead on FD_tools_复核 blocking 1): a field with status done (evaluated) or nonfinite, a repair with its evaluation.  Any other failure (cuda_oom, vram_stop, time_cap, scale_mismatch, checkpoint_mismatch, a stopped arm's cells, a repair process that crashed) says nothing about the D factor: where the reading would be 工作点不是杠杆 it reads 不可判 with those cells named.  工作点是杠杆 and 中间带 stand: an arm satisfied the premise whatever another did; the cells are still written.
No arm satisfies the premise while a candidate through the gate was not repaired (the lead on important 1, option a): 不可判, those arms named; no arm takes another's place.
RHx600, the repair of fl_v1's own Hx at 600 steps (the lead, plan revision 2), is only described: not a candidate, not in the premise, not in the reading, not among the failed cells.  It is in `repairs` under the arm Hx, which no choice can name.
A cell is read only with status done.  Everything else is only described.

The functions on the records are decide_fl's, not copied: a second instance of that module whose `fl` is fd_core.  The module decide_fl itself, FL's reading of fl_v1, is not touched, and it is the one that reads the references out of fl_v1.
"""
import importlib.util
import json
from pathlib import Path
import statistics
import sys

from decide_ef import read_json, set_aside
import decide_fl as fl_reading
import fd_core as fd
import fl_core as fl
import mc0_core as core

SCHEMA = 'bayesreg.fd.decision.v1'
N = len(core.PAIRS)
BAR_LEVER, BAR_NOT_LEVER, N_POSITIVE, SDLOGJ_MAX = .005, .002, 4, .10
STEP = fd.NATIVE
REFERENCE_ARMS = ('Hx', 'H', 'F')
MEASURES = fl_reading.MEASURES
READ_LEVER, READ_NOT_LEVER, READ_MIDDLE, READ_UNDECIDABLE = '工作点是杠杆', '工作点不是杠杆', '中间带', '不可判'
RESULT_STATUSES = ('done', 'nonfinite')                 # of a field run: what the D factor can have done; every other failure comes from the machine or the tools
NOTE = ('开发集单次观测，通道来自 unet35_v1_ens 的预测（oracle: false），真值标签只进评测。**不是 F_B，没有经认证的下降，只有终态过证书；不是方法，不进结果表，进方法先改合同，由 BZ 定。** n = 5，前 5 对之前选配用过，从 5 个新臂里取最好，要确认得看其余 14 对。')


def _batch_instance():
    """decide_fl once more under another name, with `fl` = fd_core: its functions on the records read this batch's plan.  sys.modules['decide_fl'] stays FL's."""
    spec = importlib.util.spec_from_file_location('_fd_decide_fl', Path(fl_reading.__file__))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.fl = fd
    return module


records = _batch_instance()


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


# ---- fl_v1, read only ------------------------------------------------------------------------------------------------------------------------------
def reference_fields(results=None):
    """{(arm, first5, step, pair): row} of Hx, H, F in fl_v1, by FL's own decide_fl."""
    root = fd.fl_root(results)
    fields = {}
    for arm in REFERENCE_ARMS:
        for pair in core.PAIRS:
            for step, row in fl_reading.field_rows(root, fl.Cell(arm, fl.FIRST, pair)).items():
                fields[(arm, fl.FIRST, step, pair)] = row
    return fields


def reference_row(references, arm):
    return fl_reading.arm_step(references, arm, STEP)


# ---- the choice ----------------------------------------------------------------------------------------------------------------------------------------
def select_repairs(fields, references):
    """The rule of the module docstring.  The bar is the mean of fl_v1's Hx at 600 steps, not rounded (.8501 in the brief); no bar when it is not read 5/5, and then nothing is chosen."""
    bar = reference_row(references, 'Hx')
    rows = []
    for arm in fd.NEW_ARMS:
        r = records.arm_step(fields, arm, STEP)
        rows.append(dict(arm=arm, step=STEP, D_factor=fd.ARMS[arm]['D_factor'], weight=fd.ARMS[arm]['weight'], n_read=r['n_read'], seg35=r['seg35'], seg4=r['seg4'], sdlogj_official=r['sdlogj_official'], passes=bool(bar['complete'] and r['complete'] and r['seg35'] > bar['seg35'] and r['sdlogj_official'] <= SDLOGJ_MAX)))
    ranked = sorted((r for r in rows if r['passes']), key=lambda r: -r['seg35'])
    chosen = [dict(arm=r['arm'], step=r['step']) for r in ranked[:fd.REPAIR_TOP]]
    return dict(schema=SCHEMA+'.repair_selection', rule=f'Hx3、Hx4、Hx6、Fx2、Fx4 在 {STEP} 步，5/5 读到，α = 1 seg35 严格高于 fl_v1 的 Hx@{STEP} 且官方 SDlogJ ≤ {SDLOGJ_MAX}，取 seg35 最高的 {fd.REPAIR_TOP} 个', bar_hx_seg35=bar['seg35'], bar_n_read=bar['n_read'], candidates=rows, chosen=chosen, eps=fd.REPAIR_EPS)


def ensure_repair_selection(root, references=None):
    """The code's choice, written once."""
    references = reference_fields() if references is None else references
    return records.ensure_selection(root, 'repair_selection_1.json', lambda fields, repairs: select_repairs(fields, references))


# ---- the reading ------------------------------------------------------------------------------------------------------------------------------------------
def untried_arms(selection):
    """The candidates through the gate that the code did not choose: nobody has repaired them."""
    if selection is None:
        return []
    chosen = {c['arm'] for c in selection['chosen']}
    return [r['arm'] for r in selection.get('candidates', []) if r['passes'] and r['arm'] not in chosen]


def not_read(fields, repairs, chosen, stops):
    """The cells that failed in a way that says nothing about the D factor (the module docstring), each with its state."""
    out = []
    for arm in fd.NEW_ARMS:
        for pair in core.PAIRS:
            row = fields.get((arm, fd.FIRST, STEP, pair))
            if row is None:
                if arm in stops:
                    out.append(f'{arm} {core.short_name(pair)}：臂被停，没有运行')
            elif row['status'] not in RESULT_STATUSES:
                out.append(f"{arm} {core.short_name(pair)}：{row['status']}")
    for c in chosen:
        for pair in core.PAIRS:
            row = repairs.get((c['arm'], fd.FIRST, c['step'], pair))
            if row is None:
                if 'R'+c['arm'] in stops:
                    out.append(f"修复 {c['arm']} {core.short_name(pair)}：臂被停，没有运行")
            elif not row['metrics']:
                out.append(f"修复 {c['arm']} {core.short_name(pair)}：{row['status']}")
    return out


def not_settled(fields, repairs, references, chosen, selected, stops, control_same=True):
    """What is missing before the reading can be read; [] when it is settled."""
    if not selected:
        return ['中间读数：代码还没有选臂']
    missing = []
    if control_same is not True:
        missing.append(f'Hx 控制没有过（u 对 fl_v1 的 Hx@{STEP}：{control_same}）')
    if not reference_row(references, 'Hx')['complete']:
        missing.append(f"fl_v1 的 Hx@{STEP} 只读到 {reference_row(references, 'Hx')['n_read']}/{N}")
    for arm in fd.NEW_ARMS:
        for pair in core.PAIRS:
            row = fields.get((arm, fd.FIRST, STEP, pair))
            if row is None:
                if arm not in stops:
                    missing.append(f'{arm} {core.short_name(pair)} 没有运行')
            elif row['status'] == 'done' and row['metrics'] is None:
                missing.append(f'{arm} {core.short_name(pair)} 没有评测')
    for c in chosen:
        for pair in core.PAIRS:
            row = repairs.get((c['arm'], fd.FIRST, c['step'], pair))
            if row is None:
                if 'R'+c['arm'] not in stops:
                    missing.append(f"修复 {c['arm']} {core.short_name(pair)} 没有运行")
            elif not row['metrics'] and row['status'] != 'failed':
                missing.append(f"修复 {c['arm']} {core.short_name(pair)} 没有评测")
    return missing


def paired_with_reference(repairs, references, arm, measures=('seg35', 'seg4', 'cortex', 'small_six', 'large_six', 'other_23', 'white_matter', 'hd95', 'sdlogj_official', 'sdlogj_brain')):
    """The repaired arm - fl_v1's Hx at 600 steps (alpha = 1), pair by pair.  Positive is strictly > 0."""
    out = {}
    for m in measures:
        diffs = {}
        for pair in core.PAIRS:
            a, b = repairs.get((arm, fd.FIRST, STEP, pair)), references.get(('Hx', fd.FIRST, STEP, pair))
            if a and b and a['metrics'] and b['metrics'] and a['after'].get(m) is not None and b['metrics'].get(m) is not None:
                diffs[core.short_name(pair)] = a['after'][m]-b['metrics'][m]
        out[m] = dict(mean=mean(diffs.values()), positive=sum(d > 0 for d in diffs.values()), n=len(diffs), per_pair=diffs)
    return out


def read_all(fields, repairs, references, chosen, selected=True, stops=None, untried=(), control_same=True):
    """The rule of the module docstring.  untried: untried_arms of the choice.  control_same: control_of_hx's first value."""
    stops = stops or {}
    untried = list(untried)
    rows = [records.repair_summary(repairs, c['arm'], c['step']) for c in chosen]
    good = [r for r in rows if r['satisfies_premise']]
    n_read = sum(records.arm_step(fields, arm, STEP)['complete'] for arm in fd.NEW_ARMS)
    effect = dict(arms_read=n_read, repaired=len(rows), satisfying_the_premise=len(good), text=f'最好的臂是从 {len(fd.NEW_ARMS)} 个新臂里取的（{n_read} 个 5/5 读到）：{len(rows)} 个做了修复，{len(good)} 个满足前提。n = {N}，前 5 对之前选配用过。')
    missing = not_settled(fields, repairs, references, chosen, selected, stops, control_same)
    if missing:
        return dict(reading=READ_UNDECIDABLE, reason='；'.join(missing), best=None, premise=rows, selection_effect=effect, seg4_note=None, not_read=[], untried=untried)
    failed = not_read(fields, repairs, chosen, stops)
    failed_reason = f"这些格的失败和 D 因子无关，算没读到，不能读成工作点不是杠杆：{'、'.join(failed)}"
    if not good:
        premise = '；'.join(f"{r['arm']}@{r['step']}：{'、'.join(r['missed'])}" for r in rows) if rows else f'没有臂过修复门（α = 1 seg35 严格高于 fl_v1 的 Hx@{STEP} 且官方 SDlogJ ≤ {SDLOGJ_MAX}）'
        reasons = ([failed_reason] if failed else [])+([f"选中的臂都不满足前提，还有过门的臂没有修复：{'、'.join(untried)}"] if untried else [])
        reading = READ_UNDECIDABLE if reasons else READ_NOT_LEVER
        return dict(reading=reading, reason='；'.join(reasons+[f'没有臂满足前提：{premise}']), best=None, premise=rows, selection_effect=effect, seg4_note=None, not_read=failed, untried=untried)
    best = max(good, key=lambda r: r['seg35'])
    delta = paired_with_reference(repairs, references, best['arm'])
    d35, positive = delta['seg35']['mean'], delta['seg35']['positive']
    if d35 >= BAR_LEVER and positive >= N_POSITIVE:
        reading = READ_LEVER
    elif d35 <= BAR_NOT_LEVER:
        reading = READ_UNDECIDABLE if failed else READ_NOT_LEVER
    else:
        reading = READ_MIDDLE
    d4 = delta['seg4']['mean']
    note = None if d4 is None or d4 >= 0 else f"seg4 下降：{best['arm']} 对 Hx@{STEP} {d4:+.4f}（{delta['seg4']['positive']}/{delta['seg4']['n']} 为正）。判定只用 seg35。"
    return dict(reading=reading, reason=failed_reason if reading == READ_UNDECIDABLE else None, best=dict(arm=best['arm'], step=best['step'], seg35=best['seg35'], seg4=best['seg4'], sdlogj_official=best['sdlogj_official'], max_task_clock=best['max_task_clock'], delta=delta), premise=rows, selection_effect=effect, seg4_note=note, not_read=failed, untried=untried)


# ---- controls (records) ---------------------------------------------------------------------------------------------------------------------------------------
def control_of_hx(root, results=None):
    """(same, mine, theirs): the control's u at 600 steps against fl_v1's Hx.  same None when a side has no record."""
    mine = ((read_json(fd.run_dir(root, fd.plan_control()[0])/'run.json') or {}).get('checkpoints') or {}).get(str(STEP), {}).get('u_sha256')
    theirs = ((read_json(fl.run_dir(fd.fl_root(results), fl.Cell(fd.CONTROL_ARM, fl.FIRST, fd.CONTROL_PAIR))/'run.json') or {}).get('checkpoints') or {}).get(str(STEP), {}).get('u_sha256')
    return (None if mine is None or theirs is None else mine == theirs), mine, theirs


def controls(root, results=None):
    """The control, and the scale: an arm's D multiplier over the one of the same pair's unweighted-or-weighted arm of fl_v1 (H, F) is its D factor; Q0's norm is the same.  True, False, or None when a side has no record.  Records only."""
    same, mine, theirs = control_of_hx(root, results)
    out = dict(hx_control=dict(same=same, mine=mine, theirs=theirs), scale={})
    for pair in core.PAIRS:
        row = {}
        for arm in fd.NEW_ARMS:
            scale = (read_json(fd.run_dir(root, fd.Cell(arm, fd.FIRST, pair))/'run.json') or {}).get('scale') or {}
            source = 'H' if fd.ARMS[arm]['weight'] else 'F'
            reference = (read_json(fl.run_dir(fd.fl_root(results), fl.Cell(source, fl.FIRST, pair))/'run.json') or {}).get('scale') or {}
            row[arm] = dict(D_multiplier_is_factor_times_FL=None if scale.get('D_multiplier') is None or reference.get('D_multiplier') is None else scale['D_multiplier'] == fd.ARMS[arm]['D_factor']*reference['D_multiplier'], g_norm_Q0_equal_FL=None if scale.get('g_norm_Q0') is None or reference.get('g_norm_Q0') is None else scale['g_norm_Q0'] == reference['g_norm_Q0'])
        out['scale'][core.short_name(pair)] = row
    return out


# ---- the description ---------------------------------------------------------------------------------------------------------------------------------------------
def versus(fields, references, arm, reference, measures=('seg35', 'seg4', 'cortex', 'small_six', 'sdlogj_official', 'sdlogj_brain')):
    """arm at 600 steps (alpha = 1, not repaired) - fl_v1's reference arm at 600 steps, paired; only described."""
    out = dict(arm=arm, reference=reference)
    for m in measures:
        diffs = []
        for pair in core.PAIRS:
            a, b = fields.get((arm, fd.FIRST, STEP, pair)), references.get((reference, fd.FIRST, STEP, pair))
            if a and b and a['metrics'] and b['metrics'] and a['status'] == 'done' and b['status'] == 'done' and a['metrics'].get(m) is not None and b['metrics'].get(m) is not None:
                diffs.append(a['metrics'][m]-b['metrics'][m])
        out[m] = dict(mean=mean(diffs), positive=sum(d > 0 for d in diffs), n=len(diffs))
    return out


def describe(fields, repairs, references):
    table = [dict(records.arm_step(fields, arm, STEP), D_factor=fd.ARMS[arm]['D_factor'], weight=fd.ARMS[arm]['weight']) for arm in fd.NEW_ARMS]
    reference_rows = [dict(fl_reading.arm_step(references, arm, STEP), D_factor=dict(Hx=2., H=1., F=1.)[arm], weight=None if arm == 'F' else 'Q3h') for arm in REFERENCE_ARMS]
    against_hx = [versus(fields, references, arm, 'Hx') for arm in fd.NEW_ARMS]
    against_source = [versus(fields, references, arm, 'H' if fd.ARMS[arm]['weight'] else 'F') for arm in fd.NEW_ARMS]
    return dict(table=table, references=reference_rows, against_hx=against_hx, against_source=against_source, fl_repair=dict(records.repair_summary(repairs, fd.CONTROL_ARM, STEP), repair_arm='RHx600', role='只描述：修复 fl_v1 的 Hx@600 场；不进选择，不进读数，不算修复候选'))


def code_record(root):
    """What wrote this reading: the sha256 of the files that hold its rules, and the batch's commit."""
    return dict(decide_fd_sha256=core.sha_file(Path(__file__)), fd_core_sha256=core.sha_file(Path(fd.__file__)), decide_fl_sha256=core.sha_file(Path(fl_reading.__file__)), fl_core_sha256=core.sha_file(Path(fl.__file__)), batch_code_sha=(read_json(Path(root)/'batch.json') or {}).get('code_sha'))


def decide(root, stops=None, results=None):
    root = Path(root)
    fields, repairs = records.collect(root)
    bad = sorted(r['run_id'] for r in fields.values() if r['metrics'] is not None and r['oracle'] is not False)
    if bad:
        raise SystemExit(f'run.json is not oracle false for {bad}; nothing written')
    references = reference_fields(results)
    chosen = fd.chosen_repairs(root)
    selection = fd.read_selection(root, 'repair_selection_1.json')
    result = read_all(fields, repairs, references, chosen, selected=selection is not None, stops=stops, untried=untried_arms(selection), control_same=control_of_hx(root, results)[0])
    decision = dict(schema=SCHEMA, code=code_record(root), selections=fd.selection_hashes(root), is_conclusion=False, next_step='交总负责人与 BZ', oracle=False, note=NOTE, arms_stopped=stops or {}, rules=dict(bar_lever=BAR_LEVER, bar_not_lever=BAR_NOT_LEVER, n_positive=N_POSITIVE, sdlogj_max=SDLOGJ_MAX, task_budget_seconds=fd.TASK_BUDGET_SECONDS, n=N, repair_eps=fd.REPAIR_EPS, repair_top=fd.REPAIR_TOP),
                    reading=result['reading'], reason=result['reason'], best=result['best'], premise=result['premise'], selection_effect=result['selection_effect'], seg4_note=result['seg4_note'], not_read=result['not_read'], untried=result['untried'], repair_selection=selection, description=describe(fields, repairs, references), rest14=records.describe_rest(fields, repairs, root), controls=controls(root, results))
    return decision, fields, repairs


# ---- the words --------------------------------------------------------------------------------------------------------------------------------------------------------
_f, _signed = fl_reading._f, fl_reading._signed


def _cell(d):
    return f"{_signed(d['mean'])}（{d['positive']}/{d['n']}）"


def summary_text(decision):
    d = decision['description']
    best = decision['best']
    L = ['# FD 读数（由 decide_fd.py 写出）', '', f"**读数：{decision['reading']}**（不是结论，下一步交总负责人与 BZ）", '', NOTE, '']
    if best is not None:
        delta = best['delta']
        L += [f"最好的满足前提的臂：{best['arm']}（{best['step']} 步），修复后 seg35 / seg4 {_f(best['seg35'])} / {_f(best['seg4'])}，官方 SDlogJ {_f(best['sdlogj_official'])}，任务时钟最长 {_f(best['max_task_clock'], 1)} s。Δ（对 fl_v1 的 Hx@{STEP}，α = 1，配对）seg35 {_cell(delta['seg35'])}，seg4 {_cell(delta['seg4'])}，皮层 {_cell(delta['cortex'])}，小六个 {_cell(delta['small_six'])}，官方 SDlogJ {_cell(delta['sdlogj_official'])}，脑内 SDlogJ {_cell(delta['sdlogj_brain'])}。", '']
    if decision['seg4_note']:
        L += [f"**{decision['seg4_note']}**", '']
    if decision['reason']:
        L += [f"原因：{decision['reason']}", '']
    if decision['not_read']:
        L += [f"**失败的格（算没读到，不算结果）：{'、'.join(decision['not_read'])}**", '']
    if decision['untried']:
        L += [f"过修复门但没有修复的臂：{'、'.join(decision['untried'])}", '']
    L += [decision['selection_effect']['text'], '']
    if decision['arms_stopped']:
        L += [f"停掉的臂：{json.dumps(decision['arms_stopped'], ensure_ascii=False)}", '']
    L += ['## 前提（修复后，证书都由评测单元从场算）', '', '| 臂 | 满足前提 | α = 1 证书 | seg35 / seg4 修复前 → 修复后 | 官方 / 脑内 SDlogJ | HD95 | 任务时钟 平均 / 最长 s | 修复 平均 / 最长 s | 预热平均 s | 运行与评测不一致 | 差在哪 |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in decision['premise']:
        L.append(f"| {r['arm']}@{r['step']} | {r['satisfies_premise']} | {r['n_certified_at_1']}/{N} | {_f(r['seg35_before'])} / {_f(r['seg4_before'])} → {_f(r['seg35'])} / {_f(r['seg4'])} | {_f(r['sdlogj_official'])} / {_f(r['sdlogj_brain'])} | {_f(r['hd95'], 3)} | {_f(r['mean_task_clock'], 1)} / {_f(r['max_task_clock'], 1)} | {_f(r['mean_seconds_repair'], 1)} / {_f(r['max_seconds_repair'], 1)} | {_f(r['mean_seconds_warmup'], 1)} | {r['run_and_evaluation_disagree'] or '无'} | {'；'.join(r['missed']) or '—'} |")
    L += ['', '## 只描述：D 因子曲线（α = 1，taper 后，未修复；不是通过证书的状态）', '', '| 臂 | 来源 | 权重 | D 因子 | 读到 | seg35 / seg4 | HD95 | 官方 / 脑内 SDlogJ | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | 不可行单元 | 任务时钟 平均 / 最长 s | D | 几何 | floor |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for source, rows in ((fd.FL_ROOT_NAME, d['references']), ('本批', d['table'])):
        for r in rows:
            L.append(f"| {r['arm']} | {source} | {r['weight'] or '无'} | {r['D_factor']:g} | {r['n_read']}/{N} | {_f(r['seg35'])} / {_f(r['seg4'])} | {_f(r['hd95'], 3)} | {_f(r['sdlogj_official'])} / {_f(r['sdlogj_brain'])} | {_f(r['small_six'])} | {_f(r['large_six'])} | {_f(r['other_23'])} | {_f(r['cortex'])} | {_f(r['white_matter'])} | {'、'.join(str(x) for x in r['n_infeasible'])} | {_f(r['seconds_task'], 1)} / {_f(r['max_seconds_task'], 1)} | {_f(r['data'], 1)} | {_f(r['geometry'], 1)} | {_f(r['floor'], 2)} |")
    L += ['', 'D、几何、floor 是该步之前最后一次求值，D 因子不同的臂 D 的尺度不同，不直接比。', '']
    for title, rows in ((f'对 fl_v1 的 Hx@{STEP}（配对，未修复）', d['against_hx']), (f'对同样权重、D 因子 1 的臂（H 或 F，fl_v1，配对，未修复）', d['against_source'])):
        L += [f'## 只描述：{title}', '', '| 臂 | 对照 | seg35 | seg4 | 皮层 | 小六个 | 官方 SDlogJ | 脑内 SDlogJ |', '|---|---|---|---|---|---|---|---|']
        for r in rows:
            L.append(f"| {r['arm']} | {r['reference']} | {_cell(r['seg35'])} | {_cell(r['seg4'])} | {_cell(r['cortex'])} | {_cell(r['small_six'])} | {_cell(r['sdlogj_official'])} | {_cell(r['sdlogj_brain'])} |")
        L.append('')
    L += ['## 只描述：修复（ε = .05）：秒数对不可行单元数', '', '| 臂 | 对 | 状态 | 证书 α = 1 | 不可行单元 前 → 后 | 修复秒 | 预热秒 | 任务时钟 | 轮数 | W 点 | 改动节点 | 最大位移变化 mm | 最小认证下界 | seg35 前 → 后 | seg4 前 → 后 | 结束原因 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in decision['premise']:
        for row in r['per_pair']:
            L.append(f"| {r['arm']} | {row['pair']} | {row['status']} | {row.get('certified_at_1')} | {row.get('n_infeasible_before')} → {row.get('n_infeasible_after')} | {_f(row.get('seconds_repair'), 1)} | {_f(row.get('seconds_warmup'), 1)} | {_f(row.get('task_clock'), 1)} | {row.get('rounds')} | {row.get('n_extra_points')} | {row.get('n_changed_nodes')} | {_f(row.get('max_change_mm'), 3)} | {_f(row.get('min_lower_bound_after'), 6)} | {_f(row.get('seg35_before'))} → {_f(row.get('seg35_after'))} | {_f(row.get('seg4_before'))} → {_f(row.get('seg4_after'))} | {row.get('reason')} |")
    r = d['fl_repair']
    L += ['', '## 只描述：RHx600（修复 fl_v1 的 Hx@600 场；不进选择，不进读数，不算修复候选，失败只记录）', '', '| α = 1 证书 | seg35 / seg4 修复前 → 修复后 | 官方 / 脑内 SDlogJ | HD95 | 任务时钟 平均 / 最长 s | 修复 平均 / 最长 s | 预热平均 s | 运行与评测不一致 | 差在哪 |', '|---|---|---|---|---|---|---|---|---|',
          f"| {r['n_certified_at_1']}/{N} | {_f(r['seg35_before'])} / {_f(r['seg4_before'])} → {_f(r['seg35'])} / {_f(r['seg4'])} | {_f(r['sdlogj_official'])} / {_f(r['sdlogj_brain'])} | {_f(r['hd95'], 3)} | {_f(r['mean_task_clock'], 1)} / {_f(r['max_task_clock'], 1)} | {_f(r['mean_seconds_repair'], 1)} / {_f(r['max_seconds_repair'], 1)} | {_f(r['mean_seconds_warmup'], 1)} | {r['run_and_evaluation_disagree'] or '无'} | {'；'.join(r['missed']) or '—'} |", '', '| 对 | 状态 | 证书 α = 1 | 不可行单元 前 → 后 | 修复秒 | 预热秒 | 任务时钟 | 轮数 | W 点 | 改动节点 | 最大位移变化 mm | 最小认证下界 | seg35 前 → 后 | seg4 前 → 后 | 结束原因 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for row in r['per_pair']:
        L.append(f"| {row['pair']} | {row['status']} | {row.get('certified_at_1')} | {row.get('n_infeasible_before')} → {row.get('n_infeasible_after')} | {_f(row.get('seconds_repair'), 1)} | {_f(row.get('seconds_warmup'), 1)} | {_f(row.get('task_clock'), 1)} | {row.get('rounds')} | {row.get('n_extra_points')} | {row.get('n_changed_nodes')} | {_f(row.get('max_change_mm'), 3)} | {_f(row.get('min_lower_bound_after'), 6)} | {_f(row.get('seg35_before'))} → {_f(row.get('seg35_after'))} | {_f(row.get('seg4_before'))} → {_f(row.get('seg4_after'))} | {row.get('reason')} |")
    if decision['rest14']:
        L += ['', '## 只描述：其余 14 对（总负责人单独启动；不改上面的读数）', '', '| 臂 | 步 | 场 读到 | α = 1 seg35 / seg4 | 修复 读到 | α = 1 证书 | 修复后 seg35 / seg4 | 官方 SDlogJ | 任务时钟 平均 / 最长 s |', '|---|---|---|---|---|---|---|---|---|']
        for r in decision['rest14']:
            a, b = r['fields'], r['repairs']
            L.append(f"| {r['arm']} | {r['step']} | {a['n_read']}/14 | {_f(a['seg35'])} / {_f(a['seg4'])} | {b['n_read']}/14 | {b['n_certified_at_1']}/14 | {_f(b['seg35'])} / {_f(b['seg4'])} | {_f(b['sdlogj_official'])} | {_f(b['mean_task_clock'], 1)} / {_f(b['max_task_clock'], 1)} |")
    c = decision['controls']
    code = decision['code']
    bad = [f'{short} {arm} {key}' for short, row in c['scale'].items() for arm, checks in row.items() for key, value in checks.items() if value is False]
    L += ['', '## 控制（只记录；Hx 的控制不过时队列已经停了）', '', f"Hx 控制（{core.short_name(fd.CONTROL_PAIR)}，u 对 fl_v1 的 Hx@{STEP}）逐位相同：{c['hx_control']['same']}", '', f"尺度数字不符（D_multiplier = D 因子 × fl_v1 同对 H 或 F 的，g_norm_Q0 相同）：{bad or '无'}（没有读到的不算不符）", '', f"读数的规则：decide_fd.py {code['decide_fd_sha256']}，fd_core.py {code['fd_core_sha256']}，decide_fl.py {code['decide_fl_sha256']}，fl_core.py {code['fl_core_sha256']}，批次提交 {code['batch_code_sha']}；选择文件 {decision['selections']}", '',
          '## 它证伪不了什么', '', '不是 F_B，没有经认证的下降，只有终态过证书；开发集单次；n = 5。工作点的差过不过标量原生层这里不量。进方法先改合同，由 BZ 定。']
    return '\n'.join(L)+'\n'


def write_analysis(root, stops=None, results=None):
    """analysis/: an earlier version moved to analysis/previous_<UTC>/, never overwritten."""
    root = Path(root)
    decision, fields, repairs = decide(root, stops, results)
    out = root/'analysis'
    set_aside(out, ('decision_fd.json', 'long.csv', 'summary_fd.md'))
    (out/'decision_fd.json').write_text(json.dumps(decision, indent=1, ensure_ascii=False)+'\n')
    (out/'summary_fd.md').write_text(summary_text(decision), encoding='utf-8')
    records.write_long(out/'long.csv', fields, repairs)
    return decision
