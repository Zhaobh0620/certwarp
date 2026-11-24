"""R2 step 3b, the 19-pair registration reading (runs/20261110-r2/plan.md section 5): HS's driver (experiments/hs/run_hs.py) on R2's candidate.  Development round 2's first registration reading (of 2).
It runs only after R2's segmentation-level reading says 'pass' (seg/unet35_v4_full's place is taken by seg/unet35_r2_full/analysis/decision_r2.json); before that every queue stops with 等 R2.

The modules of HS loaded once more (r2f_core.load_copy) with r2f_core in the place of hs_core, r2_seg's copy of hs_seg in the place of hs_seg, and the copy of decide_hs below in the place of decide_hs;
hs_core, hs_seg, decide_hs, run_hs, run_fl, fx_core, decide_fx, rp_core are not edited.  So the field run, the repair with RW's fuse (1200 s), the evaluation units, the third certificate, the fallbacks of
D74/D75 (configuration C's field certified again, else the identity), the classification of each pair and the queue's checks are HS's code; what this file changes:
  - the names: the root r2_v1, the units tag-r2-*, this script; 10 s between units (A5a's rule for the queues), the floor of 40 GiB free on C: (the brief);
  - the reference of the reading is HS (hs_v1, 19/19 hx_certified), not RW: decide_hs.rw_pins / rw_reference are replaced by HS's records, pinned by the committed experiments/p2/reference_unet35_v4_ens.json
    and decision_hs.json's sha256 (the names stay HS's code path's; in R2 they carry HS);
  - the reading (read_r2, decide_r2f): Delta = R2 - HS, seg35, per pair, method level: 采用新分割器 when mean >= +.003, the subject-block 95 % lower bound > -.003, 19/19 with an output and the mean
    official SDlogJ <= .10; else 不采用; 不可判 as HS's.  "更好" only when the lower bound > 0 (about Delta >= +.0045 at HS's lower half width .0045); reported, not a third reading;
  - the 〔模型〕 expectation: slope .890 x the pair-level segmentation Delta of the candidate against UF4 (eval_r2.json), written before the first field (analysis/prediction_r2f.json);
  - the seg phase: R2's candidate staged and predicted by r2_seg (six members), the control against eval_r2.json; write-reference (HS's for P2) is refused.

    python -B experiments/r2/run_r2f.py --dry-run
    run_r2f.py queue --phase seg|fields [--commit SHA] [--wait-minutes 30] [--gap-seconds 10] [--min-host-gib 40] [--segcheck-reread TEXT]
    run_r2f.py register | predict | drift-approval --review PATH | run-one ... | repair-one ... | infer-one | segcheck | evaluate | fallback | id-evaluate | decide
Writes only under r2_v1 and seg/<the candidate's segmenter id> (and, by register, the registry file).  Exit codes as run_hs.py's.
"""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _p in (EXPERIMENTS/'unv2', EXPERIMENTS/'uf', EXPERIMENTS/'nn', EXPERIMENTS/'mc0', EXPERIMENTS/'ef', EXPERIMENTS/'fl', EXPERIMENTS/'hs', HERE):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

import r2_common as rc  # noqa: E402
import r2f_core as hc  # noqa: E402
import r2_seg  # noqa: E402

SCRIPT = 'experiments/r2/run_r2f.py'
QUEUE_UNIT = 'tag-r2-queue'
INFER_UNIT, SEGCHECK_UNIT, FALLBACK_UNIT, ID_EVAL_UNIT = 'tag-r2-infer', 'tag-r2-segcheck', 'tag-r2-fallback', 'tag-r2-eval-id'
EVAL_UNITS = dict(EVAL_UNIT='tag-r2-eval', EVAL_R1_UNIT='tag-r2-eval-r', EVAL_X_UNIT='tag-r2-eval-x', EVAL_R2_UNIT='tag-r2-eval-r2', EVAL_REST_UNIT='tag-r2-eval-rest', EVAL_REST_R_UNIT='tag-r2-eval-rest-r')
GAP_SECONDS = 10
MIN_HOST_GIB = rc.C_FLOOR_GIB
SEG_MIN_HOST_GIB = 51.         # review m6: the seg phase's one inference unit writes about 7.3 GB of predictions, the fields then about 3.2 GB: 40 + 10.5 + a margin, at its start
DECISION, SUMMARY, PER_PAIR, PREDICTION = 'decision_r2f.json', 'summary_r2f.md', 'per_pair_r2f.csv', 'prediction_r2f.json'


# ---- the reading's module: decide_hs once more, its reference and its reading replaced -------------------------------------------------------------------
D = hc.load_copy('_r2_decide_hs', EXPERIMENTS/'hs'/'decide_hs.py', {'hs_core': hc})
ALL_PAIRS, core = hc.ALL_PAIRS, hc.core


def committed_reference(repo=None):
    """{short: dict(repair_json_sha256, repair_metrics_sha256)} of the committed HS reference (its sha256 pinned); None when it is not the pinned file."""
    path = Path(core.ROOT if repo is None else repo)/rc.HS_REFERENCE_REL
    if not path.is_file() or rc.sha256_file(path) != rc.HS_REFERENCE_SHA256:
        return None
    pairs = json.loads(path.read_text(encoding='utf-8'))['p3']['pairs']
    return {s: dict(repair_json_sha256=p.get('repair_json_sha256'), repair_metrics_sha256=p.get('repair_metrics_sha256')) for s, p in pairs.items()}


def hs_pins(results=None):
    """{short: sha256 of hs_v1's repair.json and repair_metrics.json}: written into batch.json (under HS's key rw_pins) at the start of the fields, checked by every reading."""
    root = Path(hc.RESULTS if results is None else results)/'hs_v1'
    out = {}
    for pair in ALL_PAIRS:
        short = core.short_name(pair)
        hits = sorted(root.glob(f'runs/RSHx24_*/{short}'))
        d = hits[0] if len(hits) == 1 else None
        out[short] = {name: core.sha_file(d/name) if d is not None and (d/name).is_file() else None for name in ('repair.json', 'repair_metrics.json')}
    return out


def hs_reference(results=None, pins=None):
    """{pair: dict(metrics, problem)}: HS's method-level metrics of each pair (hs_v1/analysis/decision_hs.json, its sha256 pinned), every pair hx_certified; the pair's repair.json and repair_metrics.json
    the committed reference's and, given batch.json's pins, those pinned at the start.  A problem on any pair makes the reading 不可判 (as RW's in HS)."""
    root = Path(hc.RESULTS if results is None else results)/'hs_v1'
    path = root/'analysis'/'decision_hs.json'
    whole = None
    if not path.is_file() or core.sha_file(path) != rc.HS_DECISION_SHA256:
        whole = 'HS 的 decision_hs.json 不是钉住的那份'
    document = D.read_json(path) or {}
    if whole is None and document.get('reading') != hc.READ_ADOPT:
        whole = f"HS 的读数是 {document.get('reading')}"
    rows = {r.get('pair'): r for r in document.get('per_pair') or []}
    committed, now = committed_reference(), hs_pins(results)
    if committed is None:
        whole = whole or f'{rc.HS_REFERENCE_REL} 不是钉住的那份'
    out = {}
    for pair in ALL_PAIRS:
        short = core.short_name(pair)
        r = rows.get(short) or {}
        mine = now[short]
        if whole:
            out[pair] = dict(metrics=None, problem=f'{short} {whole}')
        elif pins is not None and mine != pins.get(short):
            out[pair] = dict(metrics=None, problem=f'{short} HS 的记录不是 batch.json 钉住的那份')
        elif (committed.get(short) or {}).get('repair_json_sha256') != mine['repair.json'] or (committed.get(short) or {}).get('repair_metrics_sha256') != mine['repair_metrics.json']:
            out[pair] = dict(metrics=None, problem=f'{short} HS 的记录不是已提交参照钉住的那份')
        elif r.get('line_status') != hc.HX_CERTIFIED or not r.get('metrics'):
            out[pair] = dict(metrics=None, problem=f"{short} HS 这一对不是 hx_certified（{r.get('line_status')}）")
        else:
            out[pair] = dict(metrics={m: r['metrics'].get(m) for m in D.MEASURES}, problem=None, task_clock=r.get('task_clock'), wall_hit=r.get('wall_hit'))
    return out


def read_r2(rows, reference):
    """(reading, reason, comparison): decide_hs.read_hs with R2's lines (mean >= +.003 and the block lower bound > -.003) and their words."""
    problems = []
    for pair in ALL_PAIRS:
        r = rows[pair]
        if r['kind'] in (D.NOT_RUN, D.NOT_READ):
            problems.append(r['why'])
        elif r['line_status'] == hc.HX_REJECTED:
            problems.append(f"{r['why']}（没有输出：输入错误，不是分割器的读数）")
    problems += [reference[p]['problem'] for p in ALL_PAIRS if reference[p]['problem']]
    compare = D.compare_rows(rows, reference)
    if problems:
        return hc.UNDECIDABLE, '；'.join(problems), compare
    d, sd = compare['seg35'], compare['hs_sdlogj_official']
    if d['n'] != len(ALL_PAIRS) or d['mean'] is None or d['lower'] is None or sd is None or not all(D.math.isfinite(v) for v in (d['mean'], d['lower'], sd)):
        return hc.UNDECIDABLE, f"Δ seg35 或官方 SDlogJ 没有定义（{d['n']}/{len(ALL_PAIRS)} 对，均值 {d['mean']}，下界 {d['lower']}，SDlogJ {sd}）", compare
    n_output = sum(rows[p]['metrics'] is not None for p in ALL_PAIRS)
    why = [w for w, ok in ((f"Δ 均值 {d['mean']:+.5f} < +{hc.ADOPT_MEAN_MIN}", d['mean'] >= hc.ADOPT_MEAN_MIN), (f"受试者块 95% 下界 {d['lower']:+.5f} ≤ {hc.ADOPT_LOWER}", d['lower'] > hc.ADOPT_LOWER),
                                 (f'有输出的对 {n_output}/19', n_output == len(ALL_PAIRS)), (f'官方 SDlogJ {sd:.4f} > {hc.SDLOGJ_MAX}', sd <= hc.SDLOGJ_MAX)) if not ok]
    if why:
        return hc.READ_NOT, '；'.join(why), compare
    return hc.READ_ADOPT, None, compare


def better_of(reading, compare):
    """"更好" (reported with 采用新分割器, not a reading of its own): the subject-block 95 % lower bound of Delta seg35 > 0."""
    lower = (compare.get('seg35') or {}).get('lower')
    return reading == hc.READ_ADOPT and lower is not None and lower > hc.BETTER_LOWER


def expectation(evaluation, candidate, rw_mean=None, slope=hc.OX_SLOPE):
    """〔模型〕the pair-level segmentation Delta of the candidate against UF4 (eval_r2.json rows, a pair the mean of its two subjects), 19 pairs and the first 5, times the slope: R2 - HS expected;
    rw_mean (HS's code path's name): HS's mean seg35, so R2 expected = HS + that.  Not a measurement."""
    rows = (evaluation or {}).get('rows') or {}
    cand, uf4 = rows.get(hc.CANDIDATES[candidate]['uf_arm']) or {}, rows.get('UF4') or {}
    out = dict(candidate=candidate, arm=hc.CANDIDATES[candidate]['uf_arm'], slope=slope, slope_range=[.84, .92], slope_source='OX 读法 2 (.890); the gap analysis review: .84 (stratified) to .92 (v4)', hs_seg35_mean=rw_mean)
    for name, pairs in (('pairs19', ALL_PAIRS), ('first5', core.PAIRS)):
        d = [None if D.pair_level(cand, p) is None or D.pair_level(uf4, p) is None else D.pair_level(cand, p)-D.pair_level(uf4, p) for p in pairs]
        ok = [x for x in d if x is not None]
        seg = D.mean(ok) if len(ok) == len(pairs) else None
        out[name] = dict(seg_level_delta=seg, expected_delta=None if seg is None else slope*seg, expected_range=None if seg is None else [.84*seg, .92*seg])
    e19 = out['pairs19']['expected_delta']
    out['expected_r2_seg35'] = None if e19 is None or rw_mean is None else rw_mean+e19
    out['note'] = '〔模型〕不是测量：斜率只在这条线上、v1 → v4 一小段量过，假定线性；分割级 Δ 与目标标签 T 都来自同一批 20 人 / 19 对，偏乐观。写在运行之前。'
    return out


def write_prediction(root, evaluation, candidate, rw_mean, evaluation_sha256=None):
    path = Path(root)/'analysis'/PREDICTION
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(expectation(evaluation, candidate, rw_mean), eval_r2_sha256=evaluation_sha256, written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False)+'\n')
    return document


NOTE = ('开发集单次观测，19 对；通道来自已登记的候选分割器（R2S，单臂）的预测（oracle: false），真值标签只进评测。冻结的 Hx24（Hx24-RF05-F1200）只换了分割器 id。**不是 F_B，没有经认证的下降，只有终态过证书；不进结果表；进方法先改合同 C22（照 D80 的写法），由 BZ 定。** '
        '这是开发第二轮配准读数的第 1 次（共 2 次）。候选在同样 20 名开发受试者的分割级读数上过门，目标标签 T 也在这 19 对、20 人上选出，这 19 对之前也选过这条线的配置，选择效应在内。参照是 HS（hs_v1，19/19 本线输出）。')


def code_record(root):
    files = dict(decide_hs=D.__file__, run_r2f=__file__, r2f_core=hc.__file__, r2_common=rc.__file__, decide_fx=D.fx_reading.__file__, fx_core=D.fx.__file__, fl_core=D.fl.__file__, rp_core=D.rp.__file__)
    return dict({f'{k}_sha256': core.sha_file(Path(v)) for k, v in files.items()}, batch_code_sha=(D.read_json(Path(root)/'batch.json') or {}).get('code_sha'))


def decide(root, *, results=None, stops=None, uf_out=None):
    root = Path(root)
    rows = {p: D.method_row(root, p, results) for p in ALL_PAIRS}
    bad = sorted(r['run_id'] for p in ALL_PAIRS for r in [D.field_row(root, p)] if r is not None and r['metrics'] is not None and r['oracle'] is not False)
    if bad:
        raise SystemExit(f'run.json is not oracle false for {bad}; nothing written')
    reference = hs_reference(results, (D.read_json(root/'batch.json') or {}).get('rw_pins'))
    reading, reason, compare = read_r2(rows, reference)
    _, _, candidate, upstream = hc.uf_reading(hc.HS.UF_OUT if uf_out is None else uf_out)
    method = D.method_means(rows)
    return dict(schema='bayesreg.r2f.decision.v1', code=code_record(root), is_conclusion=False, next_step='交总负责人与 BZ；采用之后要进方法，先改合同 C22（分割器 id 与登记，照 D80），由 BZ 定', oracle=False, note=NOTE, reading=reading, reason=reason, better=better_of(reading, compare),
                candidate=candidate, segmenter_id=None if candidate is None else hc.CANDIDATES[candidate]['segmenter_id'],
                r2_seg_reading=None if upstream is None else {k: upstream.get(k) for k in ('reading', 'primary_arm', 'candidate', 'delta_seg35', 'delta_seg4', 'n_positive')}, arms_stopped=stops or {},
                rules=dict(delta='R2 - HS, seg35, per pair, method level (the fallbacks in), 19 pairs', adopt=f'mean >= +{hc.ADOPT_MEAN_MIN} and subject-block 95 % lower > {hc.ADOPT_LOWER} and 19/19 with an output and the mean official SDlogJ <= {hc.SDLOGJ_MAX}',
                           better=f'采用 and the block lower bound > {hc.BETTER_LOWER}', bootstrap=dict(n=hc.BOOT_N, seed=hc.BOOT_SEED), fuse='(c-ii): a fuse hit then a pass is a pass (D74-3), counted', wall_seconds=hc.fx.FUSE_SECONDS),
                compare=compare, counts=D.counts(rows), method=method, hs=dict(n=sum(reference[p]['metrics'] is not None for p in ALL_PAIRS), **{m: D.mean(reference[p]['metrics'].get(m) for p in ALL_PAIRS if reference[p]['metrics'] is not None) for m in D.MEASURES}),
                certified_only=D.certified_only(rows), gap_to_goal=None if method['seg35'] is None else hc.GOAL-method['seg35'], gap_to_89=None if method['seg35'] is None else .89-method['seg35'],
                prediction=D.read_json(root/'analysis'/PREDICTION), per_pair=[rows[p] for p in ALL_PAIRS]), rows


def summary(d):
    c = d['compare']
    s35, s4 = c['seg35'], c['seg4']
    L = ['# R2 的 19 对配准读数（run_r2f.py 写出）', '', f"**读数：{d['reading']}**" + ('，块下界 > 0：更好' if d.get('better') else '') + '（不是结论，交总负责人与 BZ）', '', d['note'], '']
    if d['reason']:
        L += [f"原因：{d['reason']}", '']
    L += [f"候选：{d['candidate']}（{d['segmenter_id']}）；R2 分割级读数 {d.get('r2_seg_reading')}", '',
          f"Δ = R2 − HS（方法级，含退路，配对）：seg35 {D._signed(s35['mean'], 5)}（{s35['positive']}/{s35['n']} 为正；受试者块 95% [{D._signed(s35.get('lower'), 4)}, {D._signed(s35.get('upper'), 4)}]）；seg4 {D._signed(s4['mean'], 5)}（{s4['positive']}/{s4['n']}；块 [{D._signed(s4.get('lower'), 4)}, {D._signed(s4.get('upper'), 4)}]）。",
          f"线：Δ 均值 ≥ +{hc.ADOPT_MEAN_MIN}，块下界 > {hc.ADOPT_LOWER}，19/19 有输出，官方 SDlogJ ≤ {hc.SDLOGJ_MAX} 读“采用新分割器”；块下界 > 0 才说“更好”。", '']
    m, h = d['method'], d['hs']
    L += ['| | seg35 | seg4 | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | HD95 | 官方 / 脑内 SDlogJ |', '|---|---|---|---|---|---|---|---|---|---|']
    for name, x in (('R2（方法级，含退路）', m), ('HS（v4 通道）', h), ('R2 只算本线输出的对（有偏）', d['certified_only'])):
        L.append(f"| {name}（{x['n']} 对） | {D._f(x['seg35'])} | {D._f(x['seg4'])} | {D._f(x['small_six'])} | {D._f(x['large_six'])} | {D._f(x['other_23'])} | {D._f(x['cortex'])} | {D._f(x['white_matter'])} | {D._f(x['hd95'], 3)} | {D._f(x['sdlogj_official'])} / {D._f(x['sdlogj_brain'])} |")
    L += ['', f"离 .88 差 {D._f(d['gap_to_goal'])}，离 .89 差 {D._f(d['gap_to_89'])}（R2 方法级 seg35）。", '']
    p = d.get('prediction')
    if p:
        L += [f"〔模型〕运行之前写下的预期（{PREDICTION}，{p.get('written_utc')}）：分割级 Δ（19 对，一对取两人平均）{D._signed(p['pairs19']['seg_level_delta'], 5)} × 斜率 {p['slope']} = R2 − HS {D._signed(p['pairs19']['expected_delta'], 5)}，即 R2 约 {D._f(p.get('expected_r2_seg35'))}。{p['note']}", '']
    k = d['counts']
    L += [f"本线输出 {len(k['by_status'][hc.HX_CERTIFIED])}/19；退路 A0（配置 C）{k['by_status'][hc.HX_A0] or '无'}；退路恒等 {k['by_status'][hc.HX_IDENTITY] or '无'}；通道被拒 {k['by_status'][hc.HX_REJECTED] or '无'}；撞 1200 s 保险丝 {k['n_wall_hit']}；第三次认证 {k['n_third_certificate']}。", '',
          '## 它证伪不了什么', '', '不是 F_B，没有经认证的下降，只有终态过证书；开发集 19 对单次；候选在同样 20 人的分割级读数上选出；修复在 GPU 上不逐位可复现（D76）。“采用”不说明留出集上会更好；“不采用”不说明别的配方不行。进方法先改合同 C22，由 BZ 定。']
    return '\n'.join(L)+'\n'


def write_hs(root, *, results=None, stops=None, uf_out=None):
    import csv
    document, rows = decide(root, results=results, stops=stops, uf_out=uf_out)
    out = Path(root)/'analysis'
    names = (DECISION, SUMMARY, PER_PAIR)
    D.set_aside(out, names)
    (out/DECISION).write_text(json.dumps(document, indent=1, ensure_ascii=False)+'\n')
    (out/SUMMARY).write_text(summary(document), encoding='utf-8')
    reference = hs_reference(results, (D.read_json(Path(root)/'batch.json') or {}).get('rw_pins'))
    with (out/PER_PAIR).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'line_status', 'kind', *D.MEASURES, 'hs_seg35', 'delta_seg35', 'rounds', 'wall_hit', 'task_clock'])
        for pair in ALL_PAIRS:
            r, ref = rows[pair], reference[pair]['metrics'] or {}
            mt = r['metrics'] or {}
            writer.writerow([r['pair'], r['line_status'], r['kind'], *[mt.get(m) for m in D.MEASURES], ref.get('seg35'), None if mt.get('seg35') is None or ref.get('seg35') is None else mt['seg35']-ref['seg35'], r.get('rounds'), r.get('wall_hit'), r.get('task_clock')])
    return document


hc.replace_names(D, rw_pins=hs_pins, rw_reference=hs_reference, read_hs=read_r2, expectation=expectation, write_prediction=write_prediction, decide=decide, summary=summary, write_hs=write_hs,
                 code_record=code_record)     # each name checked to exist first (review m1)


# ---- the driver: run_hs once more, its names and its R2 parts replaced -----------------------------------------------------------------------------------
R = hc.load_copy('_r2_run_hs', EXPERIMENTS/'hs'/'run_hs.py', {'hs_core': hc, 'decide_hs': D, 'hs_seg': r2_seg.seg})


def instance():
    """run_fl once more under another name (run_hs._instance's lines with R2's names): the units tag-r2-*, this script, RW's fuse, the reader of the registered candidate, every pair in the plan;
    10 s between units and the floor of 40 GiB on C:."""
    spec = importlib.util.spec_from_file_location('_r2_run_fl', EXPERIMENTS/'fl'/'run_fl.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    hc.replace_names(module, fl=hc.HS, decision=D, SCRIPT=SCRIPT, QUEUE_UNIT=QUEUE_UNIT, **EVAL_UNITS)
    module.eval_argv.__defaults__ = (module.EVAL_UNIT,)
    hc.replace_names(module, GAP_SECONDS=GAP_SECONDS, DEFAULT_MIN_HOST_GIB=MIN_HOST_GIB, GATE=replace(module.GATE, own_units=(f'{QUEUE_UNIT}.service',), host_floor_gib=MIN_HOST_GIB))
    base = module.run

    def unit_argv(unit, code, inner, *, log, gpu=True, numba=module.RUN_THREADS, quota=800, runtime=base.RUN_RUNTIME_MAX):
        return base.unit_argv(unit, code, list(inner), log=log, gpu=gpu, numba=numba, quota=quota, runtime=runtime, script=SCRIPT, memory_max=module.MEMORY_MAX, nice=module.NICE)

    def run_argv(cell, code, root, sha='n/a'):
        d = hc.HS.run_dir(root, cell)
        command = 'repair-one' if cell.is_repair else 'run-one'
        return unit_argv(cell.unit, code, [command, '--arm', cell.arm, '--tier', cell.tier, '--pair', cell.pair, '--out', str(d), '--code-sha', sha], log=d/'log.txt', numba=module.REPAIR_THREADS if cell.is_repair else module.RUN_THREADS,
                         quota=module.REPAIR_QUOTA, runtime=hc.fx.REPAIR_RUNTIME_SECONDS if cell.is_repair else base.RUN_RUNTIME_MAX)
    hc.replace_names(module, unit_argv=unit_argv, run_argv=run_argv, repair=R.fxrun.RepairWithFuse(module.repair, hc.fx.FUSE_SECONDS), repair_allowed=lambda cell, root: True,
                     fast=R.HSFast(module.fast), oracle_mark_checked=R.hs_oracle_mark_checked)
    return module


hc.replace_names(R, SCRIPT=SCRIPT, QUEUE_UNIT=QUEUE_UNIT, INFER_UNIT=INFER_UNIT, SEGCHECK_UNIT=SEGCHECK_UNIT, FALLBACK_UNIT=FALLBACK_UNIT, ID_EVAL_UNIT=ID_EVAL_UNIT)
hc.replace_names(R, BASE=instance(), EVALUATION_UNITS=(EVAL_UNITS['EVAL_UNIT'], EVAL_UNITS['EVAL_R1_UNIT'], ID_EVAL_UNIT, SEGCHECK_UNIT))
READING_FILES = tuple((EXPERIMENTS.parent/rel, rel) for rel in ('experiments/hs/decide_hs.py', 'experiments/hs/hs_core.py', 'experiments/r2/run_r2f.py', 'experiments/r2/r2f_core.py', 'experiments/r2/r2_common.py',
                                                                   'experiments/fl/decide_fx.py', 'experiments/fl/fx_core.py', 'experiments/fl/decide_fl.py', 'experiments/fl/fl_core.py', 'experiments/fl/rp_core.py', R.rp.FALLBACK_REL))
hc.replace_names(R, READING_FILES=READING_FILES, DRIFT_FORBIDDEN=('bayesreg', 'jd_curl', 'configs', 'splits', 'scripts', *(f'experiments/{d}' for d in ('fl', 'ef', 'mc0', 'hs', 'd0', 'nn', 'uf', 'unv2', 'ul', 'p1', 'r2'))),
                 DRIFT_APPROVAL_REL='runs/20261110-r2/drift_approval.json', DRIFT_APPROVAL_SCHEMA='bayesreg.r2f.drift_approval.v1', DRIFT_APPROVABLE=('experiments/r2/run_r2f.py',))


def hs_lines(out, results=None):
    """HS, the reference (in run_hs's place of rw_lines): 19/19 hx_certified by its pinned records."""
    reference = hs_reference(results)
    bad = [reference[p]['problem'] for p in ALL_PAIRS if reference[p]['problem']]
    seg35 = D.mean(reference[p]['metrics']['seg35'] for p in ALL_PAIRS if reference[p]['metrics'])
    out(f"HS (hs_v1, the reference Hx24 on v4's channels): {sum(reference[p]['metrics'] is not None for p in ALL_PAIRS)}/19 hx_certified by its pinned records, seg35 mean {seg35}; problems {bad or 'none'}")
    return bad


def waiting_of(phase):
    """(waiting, stop, candidate): R2's segmentation-level reading, then what each phase waits for (the fields: the seg phase, the registration, HS's pinned reading)."""
    waiting, stop, candidate, _ = hc.uf_reading(hc.HS.UF_OUT)
    if candidate is None or phase != 'fields':
        return waiting, stop, candidate
    sd = hc.seg_dir(candidate, hc.HS.RESULTS)
    if (R.run.read_json(sd/hc.SEGCHECK_RECORD) or {}).get('passed') is not True:
        waiting.append('等分割：seg 阶段没有做完或控制没过（infer_record.json、segcheck.json）')
    if not hc.HS.registry_path().is_file():
        waiting.append(f'等登记：没有 {hc.REGISTRY_REL}（复核之后 run_r2f.py register，再提交）')
    path = Path(hc.HS.RESULTS)/'hs_v1'/'analysis'/'decision_hs.json'
    if not path.is_file() or core.sha_file(path) != rc.HS_DECISION_SHA256:
        waiting.append('等 HS：hs_v1 的 decision_hs.json 不是钉住的那份')
    return waiting, stop, candidate


def seg_input_problems(out=lambda text: None, *, candidate=None):
    """Before the seg phase: the six members' member.json there (UF's four, the recipe's two); the registry not yet written with other bytes.  Names only."""
    candidate = hc.HS.candidate() if candidate is None else candidate
    problems = []
    for name in rc.UF_ORDER:
        if not (rc.UF_OUT/name/'member.json').is_file():
            problems.append(f'UF member {name} has no member.json')
    for name in rc.MEMBER_NAMES:
        if not (r2_seg.recipe_root(candidate)/name/'member.json').is_file():
            problems.append(f"{hc.CANDIDATES[candidate]['recipe']} member {name} has no member.json")
    path, staged = hc.HS.registry_path(), hc.seg_dir(candidate, hc.HS.RESULTS)/hc.STAGED_REGISTRY
    if path.is_file() and (not staged.is_file() or staged.read_bytes() != path.read_bytes()):
        problems.append(f'{hc.REGISTRY_REL} is there with other bytes than the staged registry')
    out(f'seg phase inputs of {candidate}: problems {problems or "none"}')
    return problems


def infer_one(o):
    """The unit tag-r2-infer: the candidate staged and its 20 predictions (r2_seg.infer_all).  Only images; the pinned inputs, a clean checkout."""
    from bayesreg.seg.unet.data import OasisFiles
    from bayesreg.seg.unet.pins import PINNED_FULL_FOLDS_SHA256, TRAINING_SOURCE
    from bayesreg.seg.unet.snapshot import code_record as snapshot_code
    candidate = hc.HS.candidate()
    subjects, split, allowlist, record = R.seg_inputs()
    path = hc.uf_paths(hc.HS.UF_OUT)['eval']
    try:
        out = r2_seg.infer_all(candidate, subjects=subjects, seg_dir=hc.HS.seg_dir(candidate), uf_root=rc.UF_OUT, r2_out=hc.HS.UF_OUT, evaluation=R.run.read_json(path), split=split, allowlist=allowlist, training_source=TRAINING_SOURCE,
                               pinned_folds_sha256=PINNED_FULL_FOLDS_SHA256, files=OasisFiles(R.nc.DATA_ROOT), device=o.device, code=dict(record.get('code') or snapshot_code(), batch_sha=o.code_sha), command=sys.argv, evaluation_sha256=core.sha_file(path))
    except (r2_seg.HSSegFailed, FileExistsError) as failed:
        print(f'STOP: {failed}', file=sys.stderr)
        return 8
    return 0 if out['passed'] else 8


def write_reference(o, repo=None):
    print('write-reference is HS\'s, for P2; R2 has no P2 reference', file=sys.stderr)
    return 2


def candidate_lines(out, candidate):
    c, ids = hc.CANDIDATES[candidate], hc.model_ids(candidate)
    out(f"candidate {candidate}: {c['words']}")
    out(f"  segmenter_id {c['segmenter_id']}; fold models {ids['A']} (fold 0: UF foldA_s0, foldA_s1, {c['recipe']} {rc.member_name('A')}), {ids['B']} (fold 1: UF foldB_s0, foldB_s1, {c['recipe']} {rc.member_name('B')})")
    out(f'  registry {hc.REGISTRY_REL} (staged first at {hc.seg_dir(candidate, hc.HS.RESULTS)/hc.STAGED_REGISTRY}); predictions {hc.seg_dir(candidate, hc.HS.RESULTS)/"pred"}/<subject>/{{probs36.npz, structure_groups.npz, provenance.json}}')


def uf_lines(out):
    paths = hc.uf_paths(hc.HS.UF_OUT)
    out(f"R2's segmentation level {paths['out']}: decision_r2.json there {paths['decision'].is_file()}, eval_r2.json there {paths['eval'].is_file()} (existence only)")


hc.replace_names(R, rw_lines=hs_lines, waiting_of=waiting_of, seg_input_problems=seg_input_problems, infer_one=infer_one, write_reference=write_reference, candidate_lines=candidate_lines, uf_lines=uf_lines)


def queue(o, system=None):
    """run_hs.queue with this root's prefix (r2); the seg phase refuses a /mnt/c floor below SEG_MIN_HOST_GIB (review m6)."""
    problem = hc.HS.root_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if o.phase == 'seg' and float(getattr(o, 'min_host_gib', 0.) or 0.) < SEG_MIN_HOST_GIB:
        print(f'--phase seg needs --min-host-gib {SEG_MIN_HOST_GIB:g} or more (the predictions and the fields that follow, about 10.5 GB, above the floor of {MIN_HOST_GIB:g} GiB)', file=sys.stderr)
        return 2
    if getattr(o, 'segcheck_reread', None) is not None and o.phase != 'seg':
        print('--segcheck-reread goes with --phase seg', file=sys.stderr)
        return 2

    def guarded(o, system, root, log):
        try:
            return R.BODIES[o.phase](o, system, root, log)
        except SystemExit as refused:
            raise R.run.Stop(str(refused))
    return R.run.queue(o, system, body=guarded, prefix='r2')


def queue_launch(phase, root=None):
    """run_hs.queue_launch with the seg phase's floor of SEG_MIN_HOST_GIB (review m6)."""
    argv = list(PLAIN_QUEUE_LAUNCH(phase, root))
    if phase == 'seg':
        argv[argv.index('--min-host-gib')+1] = f'{SEG_MIN_HOST_GIB:g}'
    return argv


PLAIN_QUEUE_LAUNCH = R.queue_launch
hc.replace_names(R, queue=queue, queue_launch=queue_launch)


def dry_run(*, out=print, root=None, results=None, stage=None, allowlist=None):
    """run_hs.dry_run's lines with R2's parts: what R2 waits for, both candidates' plans, the cells, the stage and the frozen tree, HS (the reference), configuration C's fields, the registry, the fields
    queue's input check once the seg phase is done, the estimates, the start strings.  No truth label, nothing written; the seg phase's pinned inputs (the folds file's sha256) are not computed here (the brief)."""
    import shlex
    import shutil
    root = hc.HS.DEFAULT_ROOT if root is None else Path(root)
    run, fxrun, BASE = R.run, R.fxrun, R.BASE
    out(f'R2 registration reading, dry run ({run.utc()}): no optimization, no truth label, nothing written')
    gate = BASE.GATE
    out(f'root {root}; root exists: {root.exists()}; repair wall_seconds {hc.fx.FUSE_SECONDS:g} s; MemoryMax {BASE.MEMORY_MAX}, nice {BASE.NICE}; {BASE.GAP_SECONDS} s between units; gate: own units {gate.own_units}, load < {gate.load_limit}, GPU free >= {gate.gpu_free_mib} MiB, '
        f'MemAvailable >= {gate.mem_available_gib} GiB, /home >= {run.HOME_FLOOR_GB} GB, /mnt/c >= {gate.host_floor_gib} GiB')
    waiting, stop, candidate, _ = hc.uf_reading(hc.HS.UF_OUT)
    for line in waiting:
        out(f'{line}（不算问题）')
    if stop:
        out(f'{stop}（不算问题：配准读数不跑）')
    uf_lines(out)
    for key in hc.CANDIDATES:
        candidate_lines(out, key)
    out(f'the arm: {hc.HS_ARM} = {json.dumps(dict(hc.SPEC))}, the frozen Hx24 (fx_core.SPEC); the repair {hc.HS.REPAIR_ARM}: RF05 with wall_seconds {hc.fx.FUSE_SECONDS:g}; the evaluation: configuration C, alpha = 1')
    fxrun.print_cells(out, hc.HS.plan_all_fields(), 'field')
    fxrun.print_cells(out, hc.HS.plan_all_repairs(), 'repair')
    problems = fxrun.safe(lambda o, **kw: fxrun.stage_problems(o, ALL_PAIRS, **kw), out, stage=stage, allowlist=allowlist)
    problems += fxrun.safe(lambda o: fxrun.frozen_problems(o), out)
    problems += fxrun.safe(lambda o, **kw: hs_lines(o, **kw), out, results=results)
    statuses = fxrun.safe(lambda o, **kw: R.fallback_lines(o, **kw), out, results=results)
    if isinstance(statuses, list):
        problems += statuses
    from bayesreg.seg.structure_layer import is_formal_registry
    formal = is_formal_registry(hc.HS.registry_path())
    out(f'{hc.REGISTRY_REL} one of structure_layer.FORMAL_REGISTRIES: {formal}; there: {hc.HS.registry_path().is_file()}')
    if not formal:
        problems.append(f'{hc.REGISTRY_REL} is not in FORMAL_REGISTRIES')
    if candidate is not None:
        problems += R.fields_check_lines(out, candidate, stage=stage, allowlist=allowlist, results=results)
    e = R.estimates(results)
    out(f"estimates〔估算, from fd2_v1's Hx24 fields and rw_v1's repairs; HS took about the same〕: the seg phase about 6 members x 20 subjects (〔估算〕about 25 s a subject, 10 min) and the control about 10 min; "
        f"the fields {R._minutes(e.get('fields_seconds'))}, the repairs {R._minutes(e.get('repairs_seconds'))}, the evaluations {R._minutes(e.get('evaluations_seconds'))}, the fields phase {R._minutes(e.get('total_seconds'))}")
    out(f'disk〔估算〕: predictions about 7.3 GB (unet35_v4_ens/pred measured), fields about 3.2 GB (hs_v1/runs measured); /home free {shutil.disk_usage("/home").free/2**30:.0f} GiB; /mnt/c free {shutil.disk_usage("/mnt/c").free/2**30:.0f} GiB (floor {MIN_HOST_GIB:g})')
    out('--- start strings (the queues: the lead; the others: the queue, one per unit)')
    out(shlex.join(R.queue_launch('seg', root)))
    seg_code = root/hc.SEG_CODE_DIR/'code'
    out(shlex.join(R.seg_unit_argv(INFER_UNIT, seg_code, ['infer-one', '--code-sha', '<sha>'], log=root/'infer_unit.log', gpu=True, runtime=R.INFER_RUNTIME)))
    out(shlex.join(R.seg_unit_argv(SEGCHECK_UNIT, seg_code, ['segcheck', '--code-sha', '<sha>'], log=root/'segcheck_unit.log', gpu=False, runtime=R.SEGCHECK_RUNTIME)))
    out(f'{run.PY} -B {SCRIPT} register        # by hand, after the independent review; commit {hc.REGISTRY_REL}')
    out(shlex.join(R.queue_launch('fields', root)))
    code = root/'code'
    for c in hc.HS.all_cells(root):
        out(shlex.join(BASE.run_argv(c, code, root, '<sha>')))
    out(shlex.join(BASE.eval_argv(code, root, BASE.EVAL_UNIT)))
    out(shlex.join(BASE.eval_argv(code, root, BASE.EVAL_R1_UNIT)))
    out(shlex.join(BASE.unit_argv(FALLBACK_UNIT, code, ['fallback', '--root', str(root)], log=root/'fallback.log', gpu=False, numba=BASE.EVAL_THREADS, quota=BASE.EVAL_QUOTA, runtime=R.FALLBACK_RUNTIME)))
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


hc.replace_names(R, dry_run=dry_run)


def main(argv=None):
    """run_hs.main on R2's names; --dry-run is dry_run above (no truth label, nothing written)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv == ['--dry-run']:
        return R.dry_run()
    return R.main(argv)


if __name__ == '__main__':
    sys.exit(main())
