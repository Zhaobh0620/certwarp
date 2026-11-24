"""FX's three readings (runs/20261105-fx/任务书.md, plan.md, by code; labels for the lead and BZ, not conclusions).  decide_fl, decide_fd, decide_fd2 and their batches are not edited: their functions on the records are called, through another instance of decide_fl for each batch here, and FD2's own instance reads fd2_v1.

In every summary here "修复后" counts only the pairs whose repaired field passes the certificate at alpha = 1 by the evaluation unit; an uncertified attempt is listed apart and enters no mean (FD_FD2_结果_复核, tool problem of decide_fl.repair_summary, not repeated).

RW.  Only the certificates read: 19/19  Hx24 19/19;  some pair not  修复撑不住 n/19, those pairs named.  A pair is either this batch's (the control, the 5 that hit the 120 s wall, and the other 13 when the control was not bit for bit) or fd2_v1's repair (the other 13 when it was).  A pair not run or not evaluated says nothing about the wall: 不可判 (a rule of the tool, as FD2's).
OX.  seg35, the mean of the first 5 pairs at alpha = 1, not repaired:  >= .8946  上限够 .90;  < .8868  上限够不到 .90;  else 中间带;  a run that failed  不可判.
NX.  Repaired, paired against Hx24 with RW's configuration, on the pairs where both pass the certificate (the pairs dropped are named):
    19 pairs certified, mean seg35 >= .88, mean official SDlogJ <= .10   过 .88 (first of all)
    fewer than 4/5 of the planned pairs certified on both sides           不可判, which side and which pairs (the lead's ruling, FX review important 1)
    Delta seg35 >= +.005 and at least 3/4 of the pairs positive          分割器是杠杆
    Delta seg35 <= +.002                                                 不是杠杆, 不可判 if a run failed
    otherwise                                                            中间带
    the plan not settled (the rule of the other 14 not taken, a pair not run or not evaluated, no reference)  不可判 (a rule of the tool)
What is a failed run: a field with a status other than done or nonfinite, a stopped arm's cells, a repair process that crashed.  A repair that ran, was evaluated and did not pass is a result (a pair dropped).
"""
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

from decide_ef import read_json, set_aside
import decide_fd2 as fd2_reading
import decide_fl as fl_reading
import fd2_core as fd2
import fl_core as fl
import fx_core as fx
import mc0_core as core
from fx_core import ALL_PAIRS, NATIVE, NX, OX, RW

STEP = NATIVE
MEASURES = fl_reading.MEASURES
mean = fl_reading.mean
_f, _signed = fl_reading._f, fl_reading._signed
SDLOGJ_MAX = .10
RESULT_STATUSES = ('done', 'nonfinite')
UNDECIDABLE = '不可判'
CLOCK_COLUMN = '任务时钟 ≤ 90 s（不含 U-Net，BZ 11-05 定）'
BZ_SECONDS = 90.
READ_RW_ALL = 'Hx24 19/19'
OX_ENOUGH, OX_SHORT = .8946, .8868              # .90 minus the lowest and the highest offset .0054, .0132 (SV's S_T layers, the constant)
READ_OX_ENOUGH, READ_OX_SHORT, READ_OX_MIDDLE = '上限够 .90', '上限够不到 .90', '中间带'
OX_OFFSETS = (.0054, .0132)
OX_SEG_LEVEL = .9172                            # the brief: the segmentation-level seg35 of these 5 pairs
NX_PASS, NX_LEVER, NX_NO_LEVER, NX_MIDDLE = '过 .88', '分割器是杠杆', '不是杠杆', '中间带'
NX_BAR, NX_LEVER_BAR, NX_NO_BAR, NX_POSITIVE_SHARE = .88, .005, .002, .75
NX_REST_BAR, NX_REST_POSITIVE = .005, 4
RW_NOTE = ('开发集单次观测，通道来自 unet35_v1_ens 的预测（oracle: false），真值标签只进评测。**不是 F_B，没有经认证的下降，只有终态过证书；不进结果表，进方法先改合同，由 BZ 定。** '
           '**120 s 墙下 1/19 失败（0451-0452）是已有结果，fd2_v1 的记录不动；这是看到那次失败之后才去掉墙的。** 不再重试。"修复后"只算过证书的对，未认证的尝试单列。')
OX_NOTE = ('开发集前 5 对单次观测。**这是 oracle（oracle: true）：两幅图的通道都是真值，不是方法，场不进任何结果表，不当入口。上限够不等于预测标签够得到。** α = 1 未修复读数，不修复。')
NX_NOTE = ('开发集单次观测，通道来自已登记的 nn935_v1 的预测（oracle: false），真值标签只进评测。**不是 F_B，没有经认证的下降，只有终态过证书；不进结果表，进方法先改合同，由 BZ 定。** '
           '这条线已经用开发集前 5 对选过配（因子、步数都在这些对上定），NHx24 没有独立的留出集读数，选择效应在内。"修复后"只算过证书的对，未认证的尝试单列。')


TEMPERATURE_NOTE = ('温度标定（读代码核对，不是测量）：Hx24 和 NHx24 的通道只用 argmax 的 one-hot，概率值不进通道。v1 集成是各成员 softmax(z/T_m) 的均值（T_m 约 1.28–1.41），T_m 只通过成员的相对权重影响 argmax；nnU-Net 是单个模型、没有标定，单个模型的温度不改变 argmax。所以 NHx24 − Hx24 里含"有 T_m 的集成 argmax 对单个模型的 argmax"这个差，不含校准后概率值的差。NHx24 只用了 argmax；存下的概率只为读取器的格式与 QC，raw 的 npz 由总负责人在闸门之后清理。')


def _records(batch):
    """decide_fl once more under another name, with `fl` = this batch: its functions on the records read this batch's plan.  sys.modules['decide_fl'] stays FL's, FD's and FD2's instances stay theirs."""
    spec = importlib.util.spec_from_file_location(f'_{batch.PREFIX}_decide_fl', Path(fl_reading.__file__))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.fl = batch
    return module


records_rw, records_ox, records_nx = _records(RW), _records(OX), _records(NX)


# ---- what every summary of repairs shares -------------------------------------------------------------------------------------------------------
def status_words(status):
    """A status says whether the .9^k rule ran: it never does in these batches or in fd2_v1."""
    return {fx.CERTIFIED: '过证书', fx.NOT_CERTIFIED: '没过证书（退回规则没有跑）', 'failed_back_to_frozen_rule': '没过证书（退回规则没有跑；旧名 failed_back_to_frozen_rule）'}.get(status, status)


def hit_wall(record):
    """A repair.json: did it hit its wall?  reason says wall only when the certificate then failed; a repair that hit the wall and passed says certified, and only its rounds show it."""
    return record.get('reason') == 'wall' or any((r.get('inner') or {}).get('stop') == 'wall' for r in record.get('rounds') or [])


def with_wall(row, directory):
    return None if row is None else dict(row, wall_hit=hit_wall(read_json(Path(directory)/'repair.json') or {}))


def certified_row(row):
    return row is not None and bool(row.get('metrics')) and bool(row.get('certified_at_1'))


def attempt_row(pair, row):
    return dict(pair=core.short_name(pair), status=row['status'], status_words=status_words(row['status']), reason=row.get('reason'), n_infeasible_after=row.get('n_infeasible_after'), min_lower_bound_after=row.get('min_lower_bound_after'), attempt_seg35=row['after']['seg35'], attempt_seg4=row['after']['seg4'])


def certified_summary(rows):
    """rows {pair: row or None}.  The means of "修复后" over the certified pairs only; an uncertified attempt is listed, with what it is, and enters no mean."""
    read = {p: r for p, r in rows.items() if r is not None and r.get('metrics')}
    certified = {p: r for p, r in read.items() if r['certified_at_1']}
    clocks = [r['task_clock'] for r in certified.values() if r.get('task_clock') is not None]
    out = dict(n=len(rows), n_read=len(read), n_certified=len(certified), certified_pairs=[core.short_name(p) for p in certified], uncertified_attempts=[attempt_row(p, r) for p, r in read.items() if not r['certified_at_1']], mean_task_clock=mean(clocks), max_task_clock=max(clocks, default=None), clock_90=dict(n_within=sum(c <= BZ_SECONDS for c in clocks), n=len(clocks)), mean_seconds_repair=mean(r.get('seconds_repair') for r in certified.values()), max_seconds_repair=max((r['seconds_repair'] for r in certified.values() if r.get('seconds_repair') is not None), default=None))
    out.update({m: mean(r['after'].get(m) for r in certified.values()) for m in MEASURES})
    out.update({f'{m}_before': mean(r['before'].get(m) for r in certified.values()) for m in MEASURES})
    return out


def per_pair_row(pair, row, source=None):
    """A line of the table: rounds, seconds, the task clock, the certified bound, seg35 and seg4 before and after."""
    if row is None:
        return dict(pair=core.short_name(pair), source=source, status='没有运行', status_words='没有运行')
    out = dict(pair=core.short_name(pair), source=source, status=row['status'], status_words=status_words(row['status']), reason=row.get('reason'), wall_hit=row.get('wall_hit'), rounds=row.get('rounds'), seconds_repair=row.get('seconds_repair'), task_clock=row.get('task_clock'), certified_at_1=row.get('certified_at_1'), metrics=bool(row.get('metrics')))
    if row.get('metrics'):
        out.update(n_infeasible_before=row['n_infeasible_before'], n_infeasible_after=row['n_infeasible_after'], min_lower_bound_after=row['min_lower_bound_after'], seg35_before=row['before']['seg35'], seg4_before=row['before']['seg4'], seg35_after=row['after']['seg35'], seg4_after=row['after']['seg4'], sdlogj_official_after=row['after']['sdlogj_official'], is_attempt=not row['certified_at_1'])
    return out


def pair_line(r, extra=''):
    after = '' if r.get('seg35_after') is None else f"{'（未认证的尝试）' if r.get('is_attempt') else ''}{_f(r['seg35_before'])} → {_f(r['seg35_after'])} / {_f(r['seg4_before'])} → {_f(r['seg4_after'])}"
    return f"| {r['pair']} | {r.get('source') or ''} | {r['status_words']} | {r.get('reason') or ''}{'，撞墙' if r.get('wall_hit') else ''} | {r.get('rounds')} | {_f(r.get('seconds_repair'), 1)} | {_f(r.get('task_clock'), 1)} | {_f(r.get('min_lower_bound_after'), 8)} | {after} |{extra}"


PAIR_HEAD = '| 对 | 来源 | 状态 | 结束原因 | 轮数 | 修复秒 | 任务时钟 s | 认证下界 | seg35 前 → 后 / seg4 前 → 后 |'
PAIR_RULE = '|---|---|---|---|---|---|---|---|---|'


def summary_block(s, name):
    return f"{name}（只算过证书的 {s['n_certified']}/{s['n']} 对）：seg35 / seg4 {_f(s['seg35'])} / {_f(s['seg4'])}，官方 / 脑内 SDlogJ {_f(s['sdlogj_official'])} / {_f(s['sdlogj_brain'])}，HD95 {_f(s['hd95'], 3)}，小六个 {_f(s['small_six'])}，大六个 {_f(s['large_six'])}，皮层 {_f(s['cortex'])}，白质 {_f(s['white_matter'])}；任务时钟 平均 / 最长 {_f(s['mean_task_clock'], 1)} / {_f(s['max_task_clock'], 1)} s；{CLOCK_COLUMN}：{s['clock_90']['n_within']}/{s['clock_90']['n']}。"


def code_record(root, extra=()):
    """What wrote this reading: the sha256 of the files that hold its rules, and the batch's commit."""
    files = dict(decide_fx=__file__, fx_core=fx.__file__, decide_fl=fl_reading.__file__, fl_core=fl.__file__, decide_fd2=fd2_reading.__file__, fd2_core=fd2.__file__, **dict(extra))
    return dict({f'{k}_sha256': core.sha_file(Path(v)) for k, v in files.items()}, batch_code_sha=(read_json(Path(root)/'batch.json') or {}).get('code_sha'))


def write_files(root, names, decision, text):
    out = Path(root)/'analysis'
    set_aside(out, names)
    (out/names[0]).write_text(json.dumps(decision, indent=1, ensure_ascii=False)+'\n')
    (out/names[1]).write_text(text, encoding='utf-8')
    return out


def long_of_repairs(path, rows, records):
    """records.write_long on repairs: 'repaired' or 'attempt_uncertified' row by row."""
    repairs = {(r['arm'], r['tier'], r['step'], p): r for p, r in rows.items() if r is not None}
    records.write_long(path, {}, repairs)


# ---- RW ---------------------------------------------------------------------------------------------------------------------------------------
def old_rows(results=None):
    """fd2_v1's repairs with the 120 s wall, by FD2's own instance: {pair: row or None}.  The record stays as it is."""
    root = RW.fd2_root(results)
    return {pair: with_wall(fd2_reading.records.repair_row(root, RW.fd2_repair_cell(pair)), fd2.run_dir(root, RW.fd2_repair_cell(pair))) for pair in ALL_PAIRS}


def rw_control_compare(root, results=None):
    """The control against fd2_v1's repair of that pair: its free nodes and their values, bit for bit.  None when the control left no record: the caller stops."""
    cell = RW.plan_control()[0]
    mine, theirs = RW.run_dir(root, cell), fd2.run_dir(RW.fd2_root(results), RW.fd2_repair_cell(cell.pair))
    record, old = read_json(mine/'repair.json'), read_json(theirs/'repair.json')
    if record is None or old is None or not (mine/'repair.npz').is_file() or not (theirs/'repair.npz').is_file():
        return None
    with np.load(mine/'repair.npz', allow_pickle=False) as a, np.load(theirs/'repair.npz', allow_pickle=False) as b:
        same = all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes() for k in ('nodes', 'values'))
    return dict(schema='bayesreg.rw.control.v1', pair=core.short_name(cell.pair), same=same, rule='控制（旧配置一轮通过）的修复场逐位相同：其余 13 对沿用 fd2_v1；不相同：其余 13 对也用新配置跑，19 对都出自同一配置（工具的规则，不是失败）', attempt_sha256=record['attempt']['sha256'], old_attempt_sha256=old['attempt']['sha256'], rounds=len(record['rounds']), old_rounds=len(old['rounds']), passed=record['passed'], old_passed=old['passed'], wall_seconds=record['config']['wall_seconds'], old_wall_seconds=old['config']['wall_seconds'])


def ensure_rw_control(root, results=None):
    """The code's judgment of the control, once; None when the control left no record."""
    existing = RW.control_choice(root)
    if existing is not None:
        return existing
    value = rw_control_compare(root, results)
    if value is not None:
        RW.write_selection(root, RW.CONTROL_FILE, value)
    return value


def rw_rows(root, results=None):
    """({pair: row or None}, {pair: where it comes from}).  With the control bit for bit the other 13 pairs are fd2_v1's repairs (the wall did not act on them: one round); without a judgment nothing is taken from there."""
    choice = RW.control_choice(root)
    planned = {cell.pair: cell for cell in RW.all_cells(root)}
    rows, sources = {}, {}
    for pair in ALL_PAIRS:
        if pair in planned:
            rows[pair], sources[pair] = with_wall(records_rw.repair_row(root, planned[pair]), RW.run_dir(root, planned[pair])), RW.ROOT_NAME
        elif choice is not None and choice['same'] is True:
            rows[pair], sources[pair] = with_wall(fd2_reading.records.repair_row(RW.fd2_root(results), RW.fd2_repair_cell(pair)), fd2.run_dir(RW.fd2_root(results), RW.fd2_repair_cell(pair))), fx.FD2_ROOT_NAME
        else:
            rows[pair], sources[pair] = None, None
    return rows, sources


def read_rw(rows, selection, stops=None):
    """(reading, reason): only the certificates."""
    stops = stops or {}
    if selection is None:
        return UNDECIDABLE, '控制还没有判（中间读数或控制没有跑完）'
    missing = []
    for pair in ALL_PAIRS:
        r, short = rows.get(pair), core.short_name(pair)
        if r is None:
            missing.append(f"{short} {'臂被停，没有运行' if RW.REPAIR_ARM in stops else '没有运行'}")
        elif not r.get('metrics'):
            missing.append(f"{short} 没有评测（状态 {r['status']}）")
    if missing:
        return UNDECIDABLE, f"这些对没有读到，和墙无关，不能读成修复撑不住：{'、'.join(missing)}"
    n = sum(certified_row(r) for r in rows.values())
    if n == len(ALL_PAIRS):
        return READ_RW_ALL, None
    return f'修复撑不住 {n}/{len(ALL_PAIRS)}', f"没有过证书的对：{'、'.join(core.short_name(p) for p, r in rows.items() if not certified_row(r))}；只报过证书的对。不再重试。"


def source_scales(results=None):
    """The scale numbers of every source field, all 19 pairs: D factor 24, the check of the run."""
    root = RW.fd2_root(results)
    out, bad = {}, []
    for pair in ALL_PAIRS:
        record = read_json(fd2.run_dir(root, RW.fd2_field_cell(pair))/'run.json') or {}
        row = dict(D_factor=(record.get('scale') or {}).get('D_factor'), scale_check_passed=(record.get('scale_check') or {}).get('passed'), status=record.get('status'), oracle=record.get('oracle'))
        out[core.short_name(pair)] = row
        if row['D_factor'] != fx.D_FACTOR or row['scale_check_passed'] is not True or row['status'] != 'done' or row['oracle'] is not False:
            bad.append(core.short_name(pair))
    return dict(per_pair=out, bad=bad)


def decide_rw(root, stops=None, results=None):
    root = Path(root)
    rows, sources = rw_rows(root, results)
    old = old_rows(results)
    bad = sorted(r['run_id'] for r in rows.values() if r is not None and r.get('metrics') and r.get('oracle') is not False)
    if bad:
        raise SystemExit(f'repair.json is not oracle false for {bad}; nothing written')
    selection = RW.control_choice(root)
    reading, reason = read_rw(rows, selection, stops)
    fuse = [core.short_name(p) for p, s in sources.items() if s == RW.ROOT_NAME and rows[p] is not None and rows[p].get('wall_hit')]
    return dict(schema='bayesreg.rw.decision.v1', code=code_record(root), is_conclusion=False, next_step='交总负责人与 BZ', oracle=False, note=RW_NOTE, reading=reading, reason=reason, arms_stopped=stops or {}, control=selection, control_repaired_field=control_repaired(root, results), rules=dict(wall_seconds=fx.FUSE_SECONDS, old_wall_seconds=fx.OLD_WALL_SECONDS, repair_eps=RW.REPAIR_EPS, max_rounds=5, max_inner=300),
                summary=certified_summary(rows), old_120s=certified_summary(old), fuse_hit=fuse, sources={core.short_name(p): s for p, s in sources.items()}, per_pair=[per_pair_row(p, rows[p], sources[p]) for p in ALL_PAIRS], per_pair_old_120s=[per_pair_row(p, old[p], fx.FD2_ROOT_NAME) for p in ALL_PAIRS], scale=source_scales(results)), rows


def control_repaired(root, results=None):
    """After the evaluation: the repaired field of the control against fd2_v1's, the same comparison by the evaluation's own hash.  Only a record."""
    cell = RW.plan_control()[0]
    mine = (read_json(RW.run_dir(root, cell)/'repair_metrics.json') or {}).get('repaired_u_sha256')
    theirs = (read_json(fd2.run_dir(RW.fd2_root(results), RW.fd2_repair_cell(cell.pair))/'repair_metrics.json') or {}).get('repaired_u_sha256')
    return dict(mine=mine, theirs=theirs, same=None if mine is None or theirs is None else mine == theirs)


def summary_rw(d):
    L = ['# RW 读数（由 decide_fx.py 写出）', '', f"**读数：{d['reading']}**（不是结论，下一步交总负责人与 BZ）", '', d['note'], '']
    if d['reason']:
        L += [f"原因：{d['reason']}", '']
    c = d['control']
    if c is not None:
        L += [f"控制（{c['pair']}，修复场对 fd2_v1 逐位）：{c['same']}。" + ('其余 13 对沿用 fd2_v1 的修复场（第 1 轮就过了，墙没有起作用）。' if c['same'] else '**不逐位相同：其余 13 对也用新配置跑，19 对都出自同一配置（工具的规则，不是失败）。**'), '']
    L += [summary_block(d['summary'], '新结果，wall_seconds 1200'), '', summary_block(d['old_120s'], '已有结果，120 s 墙（fd2_v1，记录不动）'), '']
    if d['fuse_hit']:
        L += [f"**撞到 1200 s 保险丝的对：{'、'.join(d['fuse_hit'])}。**", '']
    attempts = d['summary']['uncertified_attempts']
    if attempts:
        L += ['**未认证的尝试（没有通过证书的状态，不进任何均值，退回规则没有跑）**：'+'；'.join(f"{a['pair']} {_f(a['attempt_seg35'])} / {_f(a['attempt_seg4'])}（结束原因 {a['reason']}，不可行单元 {a['n_infeasible_after']}）" for a in attempts), '']
    L += ['## 每对（任务时钟这次不是截断值；撞保险丝的除外）', '', PAIR_HEAD, PAIR_RULE]
    L += [pair_line(r) for r in d['per_pair']]
    L += ['', '## 每对：120 s 墙的已有结果（fd2_v1；撞墙的对任务时钟是截断值）', '', PAIR_HEAD, PAIR_RULE]
    L += [pair_line(r) for r in d['per_pair_old_120s']]
    bad = d['scale']['bad']
    L += ['', '## 控制（尺度数字，19 对）', '', f"D 因子 24、run.json 的尺度核对通过、done、oracle false，不符的对：{bad or '无'}", '', f"读数的规则：{d['code']}", '', '## 它证伪不了什么', '', '不是 F_B，没有经认证的下降，只有终态过证书；开发集单次。墙是看到失败之后才去掉的。进方法先改合同，由 BZ 定。']
    return '\n'.join(L)+'\n'


def write_rw(root, stops=None, results=None):
    decision, rows = decide_rw(root, stops, results)
    out = write_files(root, ('decision_rw.json', 'summary_rw.md', 'long.csv'), decision, summary_rw(decision))
    long_of_repairs(out/'long.csv', {p: r for p, r in rows.items() if r is not None}, records_rw)
    return decision


# ---- OX ---------------------------------------------------------------------------------------------------------------------------------------
def ox_fields(root):
    fields = {}
    for cell in OX.plan_fields():
        for step, row in records_ox.field_rows(root, cell).items():
            fields[(cell.arm, cell.tier, step, cell.pair)] = row
    return fields


def hx24_unrepaired(results=None):
    """fd2_v1's Hx24@600 at alpha = 1, not repaired, first 5 pairs: by FD2's own instance."""
    root = RW.fd2_root(results)
    fields = {}
    for pair in core.PAIRS:
        for step, row in fd2_reading.records.field_rows(root, fd2.Cell(fx.SOURCE_ARM, fd2.FIRST, pair)).items():
            fields[(fx.SOURCE_ARM, fd2.FIRST, step, pair)] = row
    return fd2_reading.records.arm_step(fields, fx.SOURCE_ARM, STEP)


def read_ox(fields):
    """(reading, reason, the arm's row)."""
    row = records_ox.arm_step(fields, fx.OX_ARM, STEP)
    if not row['complete']:
        return UNDECIDABLE, f"运行失败或没有评测：读到 {row['n_read']}/{len(core.PAIRS)}，状态 {row['statuses']}", row
    seg35 = row['seg35']
    if seg35 is None or not math.isfinite(seg35):
        return UNDECIDABLE, f'seg35 不是有限值（{seg35}）', row
    return (READ_OX_ENOUGH if seg35 >= OX_ENOUGH else READ_OX_SHORT if seg35 < OX_SHORT else READ_OX_MIDDLE), None, row


def slope_model(oracle, hx24, seg_level=OX_SEG_LEVEL, offsets=OX_OFFSETS):
    """〔模型〕The slope of this line from segmentation level to final seg35: (oracle - Hx24) / (1 - segmentation level); for a 19-pair goal and a 19-pair offset, the segmentation level on these 5 pairs that reaches it.  Not a measurement of a segmenter; a slope of 0 or below has no answer."""
    if oracle is None or hx24 is None:
        return None
    slope = (oracle-hx24)/(1-seg_level)
    need = {f'{target:.2f}': {f'{offset:.4f}': None if slope <= 0 else seg_level+(target-offset-hx24)/slope for offset in offsets} for target in (.88, .90)}
    return dict(slope=slope, seg_level=seg_level, offsets=list(offsets), oracle=oracle, hx24_unrepaired=hx24, need=need, note='〔模型〕斜率 = (O_Hx24 − Hx24 未修复) ÷ (1 − 这 5 对的分割级)；只有一条线，两个点；分割级超过 1 的数表示这条直线上够不到')


def ox_scales(root):
    """The scale numbers of all 5 pairs: the check of the native tensors against D factor x Q0's, written by the run; the truth colouring's Q0."""
    out, bad = {}, []
    for cell in OX.plan_fields():
        record = read_json(OX.run_dir(root, cell)/'run.json') or {}
        row = dict(D_factor=(record.get('scale') or {}).get('D_factor'), g_norm_Q0=(record.get('scale') or {}).get('g_norm_Q0'), D_multiplier=(record.get('scale') or {}).get('D_multiplier'), scale_check=record.get('scale_check'), oracle=record.get('oracle'))
        out[cell.short] = row
        if row['D_factor'] != fx.D_FACTOR or not isinstance(row['scale_check'], dict) or row['scale_check'].get('passed') is not True:
            bad.append(cell.short)
    return dict(per_pair=out, bad=bad)


def decide_ox(root, stops=None, results=None):
    root = Path(root)
    fields = ox_fields(root)
    wrong = sorted(r['run_id'] for r in fields.values() if r['metrics'] is not None and r['oracle'] is not True)
    if wrong:
        raise SystemExit(f'run.json is not oracle true for {wrong}; nothing written')
    reading, reason, row = read_ox(fields)
    hx24 = hx24_unrepaired(results)
    return dict(schema='bayesreg.ox.decision.v1', code=code_record(root, (('fl_oracle', Path(fx.__file__).parent/'fl_oracle.py'),)), is_conclusion=False, next_step='交总负责人与 BZ', oracle=True, note=OX_NOTE, reading=reading, reason=reason, arms_stopped=stops or {}, rules=dict(enough=OX_ENOUGH, short=OX_SHORT, offsets=list(OX_OFFSETS), seg_level=OX_SEG_LEVEL), row=row, hx24_unrepaired=hx24, model=slope_model(row['seg35'] if row['complete'] else None, hx24['seg35'] if hx24['complete'] else None), scale=ox_scales(root))


def summary_ox(d):
    r, h = d['row'], d['hx24_unrepaired']
    L = ['# OX 读数（由 decide_fx.py 写出）', '', f"**读数：{d['reading']}**（不是结论，下一步交总负责人与 BZ）", '', d['note'], '']
    if d['reason']:
        L += [f"原因：{d['reason']}", '']
    L += ['| 臂 | 读到 | seg35 / seg4 | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | HD95 | 官方 / 脑内 SDlogJ | 不可行单元 |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, row in ((fx.OX_ARM, r), ('Hx24 未修复（fd2_v1）', h)):
        L.append(f"| {name} | {row['n_read']}/{len(core.PAIRS)} | {_f(row['seg35'])} / {_f(row['seg4'])} | {_f(row['small_six'])} | {_f(row['large_six'])} | {_f(row['other_23'])} | {_f(row['cortex'])} | {_f(row['white_matter'])} | {_f(row['hd95'], 3)} | {_f(row['sdlogj_official'])} / {_f(row['sdlogj_brain'])} | {'、'.join(str(x) for x in row['n_infeasible'])} |")
    m = d['model']
    if m is not None:
        L += ['', f"{m['note']}。斜率 {_f(m['slope'], 3)}。"]
        for target, by_offset in m['need'].items():
            L.append(f"19 对 seg35 {target} 要这 5 对的分割级：" + '，'.join(f"偏移 {offset} 时 {_f(value)}" for offset, value in by_offset.items()) + '。')
    L += ['', '**上限够不等于预测标签够得到。**', '', '## 控制（尺度数字，5 对）', '', f"D 因子 24、原生张量的 ‖g_D‖ 等于 24 × 真值 Q0 的（run.json 的 scale_check）不符的对：{d['scale']['bad'] or '无'}", '', f"读数的规则：{d['code']}", '', '## 它证伪不了什么', '', '两幅图都给真值，不说明任何分割器能到这里。开发集前 5 对单次，n = 5；19 对只靠偏移折算。']
    return '\n'.join(L)+'\n'


def write_ox(root, stops=None, results=None):
    decision = decide_ox(root, stops, results)
    write_files(root, ('decision_ox.json', 'summary_ox.md'), decision, summary_ox(decision))
    return decision


# ---- NX ---------------------------------------------------------------------------------------------------------------------------------------
def nn_action(document):
    """What the NN reading (runs/20261105-nn/plan.md, decision_nn.json) lets NX do: all19, first5, none; None when there is no reading or it is not decidable."""
    if document is None:
        return None
    return dict(clearly_better='all19', middle='first5', not_better='none').get(document.get('reading'))


def nx_tier_key(pair):
    return fd2.FIRST if pair in core.PAIRS else f'rest{NATIVE}'


def nx_rows(root):
    """(field rows {pair: row}, repair rows {pair: row}) of the pairs this root has."""
    fields, repairs = records_nx.collect(root)
    def repair_of(p):
        return with_wall(repairs.get((fx.NX_ARM, nx_tier_key(p), STEP, p)), NX.run_dir(root, NX.Cell(NX.REPAIR_ARM, f'c{NATIVE}' if p in core.PAIRS else f'rest{NATIVE}', p)))
    return {p: fields.get((fx.NX_ARM, nx_tier_key(p), STEP, p)) for p in ALL_PAIRS}, {p: repair_of(p) for p in ALL_PAIRS}


def nx_compare(rows, reference, pairs):
    """NHx24 - Hx24, both repaired, paired, on the pairs both certify.  The pairs dropped are named, with why."""
    both, dropped = [], {}
    for pair in pairs:
        a, b = rows.get(pair), reference.get(pair)
        why = [w for w, ok in (('NHx24 没有过证书', certified_row(a)), ('Hx24 没有过证书', certified_row(b))) if not ok]
        if why:
            dropped[core.short_name(pair)] = '，'.join(why)
        else:
            both.append(pair)
    out = dict(n=len(both), pairs=[core.short_name(p) for p in both], dropped=dropped)
    for m in ('seg35', 'seg4', 'small_six', 'large_six', 'other_23', 'cortex', 'white_matter', 'hd95', 'sdlogj_official', 'sdlogj_brain'):
        diffs = [rows[p]['after'][m]-reference[p]['after'][m] for p in both if rows[p]['after'].get(m) is not None and reference[p]['after'].get(m) is not None]
        out[m] = dict(mean=mean(diffs), positive=sum(d > 0 for d in diffs), n=len(diffs))
    return out


def select_rest(document, rows, reference):
    """The rule of the other 14 pairs, by the NN reading.  Delta >= +.005 at the segmentation level: they run.  Middle band: NHx24 - Hx24 on the first 5 pairs >= +.005 and at least 4 of them positive (a pair dropped is not positive)."""
    action = nn_action(document)
    compare = nx_compare(rows, reference, core.PAIRS)
    if action == 'all19':
        go, why = True, 'NN 的读数 Δ ≥ +.005：19 对都跑'
    elif action == 'first5':
        go = compare['seg35']['mean'] is not None and compare['seg35']['mean'] >= NX_REST_BAR and compare['seg35']['positive'] >= NX_REST_POSITIVE
        why = f"NN 的读数是中间带：前 5 对 NHx24 − Hx24 要 ≥ +{NX_REST_BAR} 且 ≥ {NX_REST_POSITIVE}/5 为正；这里 {_signed(compare['seg35']['mean'])}，{compare['seg35']['positive']}/5"
    else:
        go, why = False, f'NN 的读数不放行：{(document or {}).get("reading")}'
    return dict(schema='bayesreg.nx.rest14.v1', go=go, why=why, nn_reading=(document or {}).get('reading'), nn_delta_seg35=(document or {}).get('delta_seg35'), first5=compare)


def ensure_rest_selection(root, document, reference):
    """The code's choice, once."""
    existing = NX.rest_choice(root)
    if existing is not None:
        return existing
    _, repairs = nx_rows(root)
    value = select_rest(document, repairs, reference)
    NX.write_selection(root, NX.REST_FILE, value)
    return value


def nx_failed_and_missing(root, fields, repairs, stops):
    """(failed, missing): the failures that say nothing about the segmenter; what is not there yet."""
    failed, missing = [], []
    for pair in NX.planned_pairs(root):
        short = core.short_name(pair)
        field, repair = fields.get(pair), repairs.get(pair)
        if field is None:
            (failed if fx.NX_ARM in stops else missing).append(f"{short} 场{'：臂被停，没有运行' if fx.NX_ARM in stops else ' 没有运行'}")
            continue
        if field['status'] not in RESULT_STATUSES:
            failed.append(f"{short} 场：{field['status']}")
            continue
        if field['status'] == 'nonfinite':
            continue
        if field['metrics'] is None:
            missing.append(f'{short} 场没有评测')
        elif repair is None:
            (failed if NX.REPAIR_ARM in stops else missing).append(f"{short} 修复{'：臂被停，没有运行' if NX.REPAIR_ARM in stops else ' 没有运行'}")
        elif not repair['metrics']:
            (missing if repair['status'] in (fx.CERTIFIED, fx.NOT_CERTIFIED) else failed).append(f"{short} 修复{' 没有评测' if repair['status'] in (fx.CERTIFIED, fx.NOT_CERTIFIED) else '：'+repair['status']}")
    return failed, missing


def read_nx(rows, reference, planned, *, selected=True, failed=(), missing=(), document=None):
    """(reading, reason, comparison).  A rule of the module docstring."""
    compare = nx_compare(rows, reference, planned)
    problems = []
    if nn_action(document) is None:
        problems.append('NN 的读数没有放行（没有读数或不可判）')
    if not selected:
        problems.append('中间读数：其余 14 对的规则还没有判')
    if any(reference.get(p) is None or not reference[p].get('metrics') for p in planned):
        problems.append('Hx24 的 RW 对照不全（RW 没有跑完）')
    problems += list(missing)
    if problems:
        return UNDECIDABLE, '；'.join(problems), compare
    certified = [rows[p] for p in planned if certified_row(rows.get(p))]
    if len(planned) == len(ALL_PAIRS) and len(certified) == len(ALL_PAIRS):
        seg35, sdlogj = statistics.fmean(r['after']['seg35'] for r in certified), statistics.fmean(r['after']['sdlogj_official'] for r in certified)
        if math.isfinite(seg35) and math.isfinite(sdlogj) and seg35 >= NX_BAR and sdlogj <= SDLOGJ_MAX:
            return NX_PASS, None, compare
    need = -(-4*len(planned)//5)                    # the lead's ruling (FX review, important 1): at least 4/5 of the planned pairs certified on both sides, 16 of 19, 4 of 5
    if compare['n'] < need:
        sides = {side: [k for k, why in compare['dropped'].items() if f'{side} 没有过证书' in why.split('，')] for side in ('NHx24', 'Hx24')}
        return UNDECIDABLE, f"两边都过证书的对 {compare['n']}/{len(planned)}，少于 4/5（要 ≥ {need}）。没有通过证书的状态：NHx24 {'、'.join(sides['NHx24']) or '无'}；Hx24 {'、'.join(sides['Hx24']) or '无'}", compare
    delta, positive, n = compare['seg35']['mean'], compare['seg35']['positive'], compare['seg35']['n']
    reason_failed = f"这些格的失败和分割器无关，算没读到，不能读成不是杠杆：{'、'.join(failed)}"
    if delta is None or not math.isfinite(delta):
        return UNDECIDABLE, (reason_failed if failed else f'Δ seg35 不是有限值（{delta}）'), compare
    if delta >= NX_LEVER_BAR and positive >= NX_POSITIVE_SHARE*n:
        return NX_LEVER, None, compare
    if delta <= NX_NO_BAR:
        return (UNDECIDABLE, reason_failed, compare) if failed else (NX_NO_LEVER, None, compare)
    return NX_MIDDLE, None, compare


def nx_reference(results=None):
    return rw_rows(Path(RW.RESULTS if results is None else results)/RW.ROOT_NAME, results)[0]


def nx_segmentation(document):
    """nnU-Net's segmentation level against v1 (Hx24's channels) and v2: only described, from NN's own reading."""
    if not document or nn_action(document) is None:
        return None
    return dict(nn=(document.get('means') or {}).get('nn'), reference_v1=(document.get('means') or {}).get('reference_v1'), reference_v2=(document.get('means') or {}).get('reference_v2'), against_v1=document.get('against_v1'), delta_seg35_against_v2=document.get('delta_seg35'), delta_seg4_against_v2=document.get('delta_seg4'))


def nx_scales(root, fields):
    """The scale numbers of every field read: D factor 24 and the check of the native tensors, all pairs."""
    out, bad = {}, []
    for pair, row in fields.items():
        if row is None:
            continue
        record = read_json(NX.run_dir(root, NX.Cell(fx.NX_ARM, nx_tier_key(pair), pair))/'run.json') or {}
        one = dict(D_factor=(record.get('scale') or {}).get('D_factor'), scale_check_passed=(record.get('scale_check') or {}).get('passed'), segmenter=((record.get('channel_sources') or {}).get('segmenter') or {}).get('segmenter_id'))
        out[core.short_name(pair)] = one
        if row['status'] == 'done' and (one['D_factor'] != fx.D_FACTOR or one['scale_check_passed'] is not True or one['segmenter'] != NX.SEGMENTER_ID):
            bad.append(core.short_name(pair))
    return dict(per_pair=out, bad=bad)


def nx_gate():
    """The gate of step 1 as it was: every subject's count of voxels and the total.  Only a record here: the queue does not start without it."""
    record, infer = read_json(NX.SEG_DIR/'convert_record.json') or {}, read_json(NX.SEG_DIR/'infer_check.json') or {}
    return dict(tolerance=record.get('tolerance'), n_differing_by_subject=record.get('n_differing_by_subject'), n_differing_total=record.get('n_differing_total'), n_voxels_total=record.get('n_voxels_total'), device_fallback=infer.get('device_fallback'), fallback_subjects=infer.get('fallback_subjects'), gpu_peak_reserved_gib=infer.get('gpu_peak_reserved_gib'))


def decide_nx(root, document, stops=None, results=None):
    root = Path(root)
    stops = stops or {}
    fields, repairs = nx_rows(root)
    bad = sorted(r['run_id'] for r in fields.values() if r is not None and r['metrics'] is not None and r['oracle'] is not False)
    if bad:
        raise SystemExit(f'run.json is not oracle false for {bad}; nothing written')
    reference = nx_reference(results)
    planned = NX.planned_pairs(root)
    failed, missing = nx_failed_and_missing(root, fields, repairs, stops)
    selection = NX.rest_choice(root)
    reading, reason, compare = read_nx(repairs, reference, planned, selected=selection is not None, failed=failed, missing=missing, document=document)
    rows_planned = {p: repairs.get(p) for p in planned}
    d4 = compare['seg4']['mean']
    note = None if d4 is None or d4 >= 0 else f"seg4 下降：NHx24 对 Hx24 {d4:+.4f}（{compare['seg4']['positive']}/{compare['seg4']['n']} 为正）。判定只用 seg35。"
    return dict(schema='bayesreg.nx.decision.v1', code=code_record(root), is_conclusion=False, next_step='交总负责人与 BZ', oracle=False, note=NX_NOTE, reading=reading, reason=reason, not_read=list(failed), arms_stopped=stops, rest14=selection, n_planned=len(planned), compare=compare, seg4_note=note, summary=certified_summary(rows_planned), reference=certified_summary({p: reference.get(p) for p in planned}), segmentation=nx_segmentation(document), gate=nx_gate(), scale=nx_scales(root, {p: fields.get(p) for p in planned}), per_pair=[per_pair_row(p, repairs.get(p), NX.ROOT_NAME) for p in planned],
                rules=dict(bar=NX_BAR, sdlogj_max=SDLOGJ_MAX, lever=NX_LEVER_BAR, no_lever=NX_NO_BAR, positive_share=NX_POSITIVE_SHARE, wall_seconds=fx.FUSE_SECONDS)), fields, repairs


def summary_nx(d):
    c = d['compare']
    L = ['# NX 读数（由 decide_fx.py 写出）', '', f"**读数：{d['reading']}**（不是结论，下一步交总负责人与 BZ）", '', d['note'], '']
    if d['seg4_note']:
        L += [f"**{d['seg4_note']}**", '']
    if d['reason']:
        L += [f"原因：{d['reason']}", '']
    if d['not_read']:
        L += [f"**失败的格（算没读到，不算结果）：{'、'.join(d['not_read'])}**", '']
    L += [f"Δ（NHx24 − Hx24，都修复，RW 配置，配对，两边都过证书的 {c['n']}/{d['n_planned']} 对）seg35 {_signed(c['seg35']['mean'])}（{c['seg35']['positive']}/{c['seg35']['n']} 为正），seg4 {_signed(c['seg4']['mean'])}（{c['seg4']['positive']}/{c['seg4']['n']}），皮层 {_signed(c['cortex']['mean'])}，小六个 {_signed(c['small_six']['mean'])}，HD95 {_signed(c['hd95']['mean'], 3)}，官方 SDlogJ {_signed(c['sdlogj_official']['mean'])}。"]
    if c['dropped']:
        L += ['', '**对集不同：只比两边都过证书的对。掉的对：**'+'；'.join(f'{k}（{v}）' for k, v in c['dropped'].items())]
    L += ['', summary_block(d['summary'], 'NHx24'), '', summary_block(d['reference'], 'Hx24（RW 配置）'), '']
    attempts = d['summary']['uncertified_attempts']
    if attempts:
        L += ['**未认证的尝试（没有通过证书的状态，不进任何均值，退回规则没有跑）**：'+'；'.join(f"{a['pair']} {_f(a['attempt_seg35'])} / {_f(a['attempt_seg4'])}（结束原因 {a['reason']}）" for a in attempts), '']
    s = d['segmentation']
    if s is not None:
        v1 = s['against_v1'] or {}
        L += [f"分割级（20 名开发受试者，NN 自己的读数，只描述）：nnU-Net 对 v1 集成（Hx24 的通道，.9260 / .9584）seg35 {_signed(v1.get('seg35'))}，seg4 {_signed(v1.get('seg4'))}；对 v2 集成 {_signed(s['delta_seg35_against_v2'])} / {_signed(s['delta_seg4_against_v2'])}。", '']
    g = d['gate']
    if g['n_differing_by_subject'] is not None:
        L += [f"闸门（存下的 probs36 的 argmax 对 pred_dev，每人不同体素 ÷ 全部体素 ≤ {g['tolerance']:g}）：20 人合计 {g['n_differing_total']} / {g['n_voxels_total']} 个体素不同；每人：" + '、'.join(f"{s.split('_')[2]} {n}" for s, n in g['n_differing_by_subject'].items()) + '。', '']
    if g.get('device_fallback') is not None:
        L += [f"推断：nnU-Net 在设备上显存不足、把结果数组移到 CPU 重算：{'有，' + '、'.join(g['fallback_subjects'] or []) if g['device_fallback'] else '没有'}；设备峰值（reserved）{_f(g['gpu_peak_reserved_gib'], 2)} GiB，上限 9 GiB。", '']
    L += [TEMPERATURE_NOTE, '']
    if d['rest14'] is not None:
        L += [f"其余 14 对：{'跑' if d['rest14']['go'] else '不跑'}（{d['rest14']['why']}）", '']
    L += [PAIR_HEAD, PAIR_RULE]
    L += [pair_line(r) for r in d['per_pair']]
    L += ['', '## 控制（尺度数字，全部读到的对）', '', f"D 因子 24、原生张量的尺度核对、分割器 {NX.SEGMENTER_ID}，不符的对：{d['scale']['bad'] or '无'}", '', f"读数的规则：{d['code']}", '', '## 它证伪不了什么', '', '不是 F_B，没有经认证的下降，只有终态过证书；开发集单次；这条线已经用开发集前 5 对选过配。一个 fold all 模型，不是集成。进方法先改合同，由 BZ 定。']
    return '\n'.join(L)+'\n'


def write_nx(root, document, stops=None, results=None):
    decision, fields, repairs = decide_nx(root, document, stops, results)
    out = write_files(root, ('decision_nx.json', 'summary_nx.md', 'long.csv'), decision, summary_nx(decision))
    long_of_repairs(out/'long.csv', {p: r for p, r in repairs.items() if r is not None}, records_nx)
    return decision
