"""R4 的单步命令（run_r4.py 分派到这里）：台的准备（stage、Atropos、推断）、GPU 冒烟、0438-0439 的逐位对照、评测与判定、开发集的下一步、HS 输出的预核。
每个读受试者文件的步骤都在 spy 之下（r4_spy.UnitSpy），打开记录写成 opened_files.json；真实运行只写在 <R4> 之下。
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import r4_common as rc  # noqa: E402

DEV_REFERENCE_SUBJECT = 'OASIS_OAS1_0438_MR1'                 # C25.6-4：头文件的参照（复核员核过 16/16 与它相同）
SMOKE_ARMS = rc.ARM_ORDER
SMOKE_CELLS = ('CTL', 'A1', 'A2')                             # C25.10-7：locate_cells 与 certify_map
VRAM_STOP_GIB = 9.


def _identity(root, role):
    import run_r4
    b = run_r4.batch(root, role)
    problems, code = run_r4.batch_identity(b)                                              # 修订 6：HEAD 是钉的提交，代码根是 batch.json 记的运行检出
    if problems:
        raise rc.Refused(f'代码身份：{problems}')
    return b, code


def dev_grid_meta():
    """开发集 pair.npz 的网格元数据（0438-0439 的 manifest，经 load_pair 的白名单解码器；不读体素）。"""
    from bayesreg.io.pair import _read_manifest
    _, metadata, _ = _read_manifest(rc.ROOT/rc.DEV_STAGE_REL/rc.SMOKE_DEV_PAIR/'manifest.json')
    return dict(metadata)


def prepare_bench(o, root=rc.R4_ROOT):
    import r4_bench as rb
    import r4_spy as spy
    from bayesreg.seg.unet.data import OasisFiles
    _identity(root, 'bench')
    bench = rb.load_bench()
    files = OasisFiles(rc.DATA_ROOT)
    if o.dry_run:
        print(f'stage → {root/"bench_stage"}；{len(bench.pairs)} 对；参照头文件 {DEV_REFERENCE_SUBJECT}；开发集网格 {dev_grid_meta()}')
        return 0
    rules = spy.bench_rules(root, unit='prepare-bench', stage_dir=root/'bench_stage', pred_root=root/'bench_stage', atropos_root=root/'bench_stage', extra=(root/'bench_stage_record',))
    dev_grid = dev_grid_meta()                                                             # 复核 I2：开发集的 manifest 在 spy 之前读（不在台的白名单里）
    from bayesreg.io.guard import check_subject
    verdict = check_subject(DEV_REFERENCE_SUBJECT, bench.allowlist)                      # C23.4-2 before the reference header is opened
    if not verdict.admitted or DEV_REFERENCE_SUBJECT not in {s for src in bench.allowlist.sources if src['name'] == rc.DEV_SOURCE for s in src['subjects']}:
        raise rc.Refused(f'{DEV_REFERENCE_SUBJECT}: not a development subject of the allowlist ({verdict.step})')
    with spy.UnitSpy(rules, expect_reads=False) as s:
        reference = rb.header_record(files.image_path(DEV_REFERENCE_SUBJECT))
        summary = rb.write_bench_stage(root/'bench_stage', bench, files, reference_header=reference, dev_grid=dev_grid)
    record = s.record()
    rc.write_json(root/'bench_stage_record'/'opened_files.json', record)
    rc.write_json(root/'bench_stage_record'/'prepare.json', dict(summary, reference_header=reference, spy_passed=record['passed']))
    return 0 if record['passed'] else 8


def atropos_bench(o, root=rc.R4_ROOT):
    import r4_bench as rb
    import r4_spy as spy
    _identity(root, 'bench')
    bench = rb.load_bench()
    if o.dry_run:
        print(f'Atropos → {root/"atropos"}；16 人：{bench.ordered_subjects()[:2]}…')
        return 0
    rules = spy.bench_rules(root, unit='atropos-bench', stage_dir=root/'bench_stage', pred_root=root/'atropos', atropos_root=root/'atropos', extra=(root/'atropos_record',), allow_images=False)
    with spy.UnitSpy(rules, expect_reads=False) as s:
        rows, argv = rb.bench_atropos(bench, root/'bench_stage', root/'atropos')
    record = s.record()
    rc.write_json(root/'atropos_record'/'opened_files.json', record)
    rc.write_json(root/'atropos_record'/'atropos.json', dict(subjects=sorted(rows), spy_passed=record['passed'], argv=argv,
                                                             subprocess_rule='C25.10-5: only ImageMath and Atropos of ANTs, in <R4>/atropos/<subject>, on t1.nii.gz and mask.nii.gz that atropos_seg.segment wrote from the stage PairInput; no other path'))
    return 0 if record['passed'] else 8


def seg_bench(o, root=rc.R4_ROOT):
    import run_r4
    import r4_bench as rb
    import r4_spy as spy
    from bayesreg.seg.unet.data import OasisFiles
    b, code = _identity(root, 'bench')
    bench = rb.load_bench()
    registry, contact, record = rb.load_bench_registry(bench, run_r4.pins_of(root))
    models_root = rc.RESULTS/'seg'/'unet35_v4_full'
    if o.dry_run:
        print(f'推断 → {root/"seg"/rc.BENCH_SEGMENTER_ID/"pred"}；成员在 {models_root}；登记 {record["sha256"]}')
        return 0
    rules = spy.bench_rules(root, unit='seg-bench', stage_dir=root/'seg'/rc.BENCH_SEGMENTER_ID, pred_root=root/'seg'/rc.BENCH_SEGMENTER_ID/'pred', atropos_root=root/'seg'/rc.BENCH_SEGMENTER_ID,
                            models_root=models_root)
    with spy.UnitSpy(rules, expect_reads=False) as s:
        rows = rb.bench_infer(bench, registry, record, contact, pred_root=root/'seg'/rc.BENCH_SEGMENTER_ID/'pred', files=OasisFiles(rc.DATA_ROOT), models_root=models_root, device=o.device,
                              code=code, command=sys.argv)
    rec = s.record()
    rc.write_json(root/'seg'/rc.BENCH_SEGMENTER_ID/'opened_files.json', rec)
    rc.write_json(root/'seg'/rc.BENCH_SEGMENTER_ID/'infer_record.json', dict(rows=rows, spy_passed=rec['passed'], registry=record))
    return 0 if rec['passed'] else 8


# ---- GPU 冒烟（C25.10-7） ------------------------------------------------------------------------------------------------------------------------
def smoke_arm(ctx, *, cells, child=None):
    """一臂的冒烟：T88 在它自己的子进程里（run_r4.py t88，带子进程的 spy；设备与主机峰值不继承，复核次要 6），然后在本单元的 spy 之下做 locate_cells 与 certify_map。
    返回 (run.json, 两个计数或 None, spy 的问题或 None)。"""
    import r4_chain as ch
    import r4_spy as spy
    Path(ctx['out']).mkdir(parents=True, exist_ok=False)
    rc.write_json(ch.context_path(ctx), ctx)
    code = (ch.SubprocessChild() if child is None else child)('t88', ctx)
    problem = ch.child_spy_problem(ctx['out'], 't88', code)
    row = rc.read_json(Path(ctx['out'])/'t88'/'run.json')
    counts = None
    if cells and problem is None and (row or {}).get('status') == 'done':
        with spy.UnitSpy(spy.pair_rules(ctx, 'smoke-cells'), Path(ctx['stage_dir'])/ctx['pair']/'pair.npz') as s:
            counts = smoke_cells(ctx, row)
        record = s.record()
        rc.write_json(Path(ctx['out'])/'opened_files_cells.json', record)
        if not record['passed']:
            problem = 'the spy of the counts did not pass'
    return row, counts, problem


def smoke_verdict(rows, spy_problems=None):
    """冒烟"过"：每个 T88 的 status 为 done；尺度核对过；设备保留峰值不超过 9 GiB；spy 过。主机峰值只记录。返回 {arm: (passed, 原因)}。"""
    out = {}
    for arm, row in rows.items():
        why = [f'spy：{(spy_problems or {}).get(arm)}'] if (spy_problems or {}).get(arm) else []
        if (row or {}).get('status') != 'done':
            why.append(f"status {(row or {}).get('status')}")
        if ((row or {}).get('scale_check') or {}).get('passed') is not True:
            why.append('尺度核对没过')
        peak = ((row or {}).get('env') or {}).get('device_peak_reserved_gib')
        if peak is not None and peak > VRAM_STOP_GIB:
            why.append(f'设备保留峰值 {peak:.2f} GiB > {VRAM_STOP_GIB}')
        out[arm] = (not why, why)
    return out


def smoke_attribution(verdicts):
    """冒烟失败的归属（C25.7-3）：CTL 的失败归整台，候选的归该候选。"""
    whole = not verdicts.get(rc.CONTROL, (False,))[0]
    return dict(whole_bench=whole, candidates=sorted(a for a, (ok, _) in verdicts.items() if a != rc.CONTROL and not ok))


def smoke(o, root=rc.R4_ROOT):
    import run_r4
    import r4_chain as ch
    import r4_eval as re_
    _identity(root, 'bench')
    pair = run_r4.bench_pairs(root)[0]
    if o.dry_run:
        print(f'冒烟：{pair}；臂 {SMOKE_ARMS}；locate_cells 与 certify_map 在 {SMOKE_CELLS}')
        return 0
    rows, cells, spies_ = {}, {}, {}
    for arm in SMOKE_ARMS:
        ctx = ch.make_context('bench', arm, pair, root=root, stage_dir=root/'bench_stage', out=root/'smoke'/arm/pair, device=o.device, bench_pins=run_r4.pins_of(root))
        rows[arm], cells_row, spies_[arm] = smoke_arm(ctx, cells=arm in SMOKE_CELLS)
        if cells_row is not None:
            cells[arm] = cells_row
    verdicts = smoke_verdict(rows, spies_)
    attribution = smoke_attribution(verdicts)
    rc.write_json(root/'smoke'/'smoke_result.json', dict(pair=pair, rows={a: dict(status=(r or {}).get('status'), scale_check=(r or {}).get('scale_check'), seconds_task=(r or {}).get('seconds_task'),
                                                                                   env=(r or {}).get('env'), soft_channels=(r or {}).get('soft_channels'), field_mask_record=(r or {}).get('field_mask_record'))
                                                                    for a, r in rows.items()},
                                                         cells=cells, verdicts={a: dict(passed=ok, why=why) for a, (ok, why) in verdicts.items()}, attribution=attribution,
                                                         note='the smoke fields enter no reading (C25.10-7); the two counts are descriptive'))
    if attribution['whole_bench']:
        re_.write_stop(root, dict(role='bench', step='smoke', scope='bench', arm=None, reason=f'冒烟不过（CTL：整台）：{attribution}'))
        return 9
    if attribution['candidates']:
        for arm in attribution['candidates']:                                              # 复核次要 3：每个失败的候选各写一份 stop
            re_.write_stop(root, dict(role='bench', step='smoke', scope='candidate', arm=arm, reason=f'{arm} 冒烟不过：{verdicts[arm][1]}'))
        return 9
    return 0


def smoke_cells(ctx, row):
    """渐变后的冒烟场：locate_cells 的个数（留给区间回退的单元，不是不可行单元）与 certify_map（配置 C，α = 1，CPU FP64）的 n_infeasible。只描述。"""
    import r4_chain as ch
    from jd_curl.three_d.feasibility import certify_map
    import repair as rp
    fz = ch.core.frozen_api()
    pin, _ = ch.admit(ctx)
    with np.load(Path(ctx['out'])/'t88'/row['field']['file'], allow_pickle=False) as z:
        raw = np.array(z['u'])
    u, _ = fz.boundary_taper(raw, pin.grid, ctx['settings']['t88']['taper'])
    u = np.ascontiguousarray(u)
    config = ch.he.certificate_config(ctx['settings'])
    cells = rp.locate_cells(u, pin.grid, config)
    result = certify_map(u, pin.grid, config=config)
    return dict(n_located_for_interval_fallback=len(cells), n_infeasible=result.n_infeasible, n_uncertain=result.n_uncertain, status=result.status)


# ---- 0438-0439 的逐位对照（C25.10-2） ----------------------------------------------------------------------------------------------------------
def devcheck(o, root=rc.R4_ROOT):
    import r4_chain as ch
    import r4_eval as re_
    _identity(root, 'dev')
    pair = rc.SMOKE_DEV_PAIR
    field, _ = re_.hs_dirs(rc.HS_ROOT, pair)
    if o.dry_run:
        print(f'对照：{pair}，HS 的场 {field}')
        return 0
    ctx = ch.make_context('dev', rc.CONTROL, pair, root=root, stage_dir=rc.ROOT/rc.DEV_STAGE_REL, out=root/'devcheck'/pair, device=o.device)
    row, _, problem = smoke_arm(ctx, cells=False)                                          # 复核 I3：T88 在子进程里，带它的 spy
    row = row or {}
    result = dict(devcheck_compare(row, Path(ctx['out'])/'t88', field), spy_problem=problem)
    if problem:
        result['same_array'] = False
    rc.write_json(root/'devcheck'/'devcheck_result.json', result)
    if not result['same_array']:
        why = (f"T88 子进程的 spy 不过（{problem}）" if problem else f"T88 没有出场（status {row.get('status')}）" if row.get('status') != 'done' else
               f"选项全关的 T88 场与 HS 的不逐位相同（u 的数组 sha256 {result['mine_u_sha256']} 对 {result['hs_u_sha256']}）")
        re_.write_stop(root, dict(role='dev', step='devcheck', scope='bench', reason=f'0438-0439：{why}（C25.10-2：读真值之前，归整台）'))
        return 9
    return 0


def devcheck_compare(row, t88_dir, hs_field):
    """u 的数组 sha256（C 序字节）与 HS 的 field_0600.npz 的 u 比；文件 sha256 另记（np.savez 的 zip 成员带写入时间，文件字节不可比）。"""
    from_hs = None
    with np.load(hs_field, allow_pickle=False) as z:
        from_hs = rc.array_sha256(np.array(z['u']))
    mine = (row.get('field') or {}).get('u_sha256')
    return dict(status=row.get('status'), mine_u_sha256=mine, hs_u_sha256=from_hs, same_array=mine is not None and mine == from_hs,
                mine_file_sha256=(row.get('field') or {}).get('sha256'), hs_file_sha256=rc.sha256_file(hs_field),
                note='the arrays are compared (C25.10-2 names the file; np.savez stamps each zip member with the time it was written, so the file bytes differ even for the same array)')


# ---- 评测、判定、开发集的下一步 ------------------------------------------------------------------------------------------------------------------
def eval_bench(o, root=rc.R4_ROOT):
    import r4_bench as rb
    import r4_eval as re_
    import r4_spy as spy
    import run_r4
    b, code = _identity(root, 'bench')
    bench = rb.load_bench()
    problems = eval_bench_checks(root, bench, b)                                           # 复核次要 10：建锁之前重核 plan.md、配对表、登记
    if problems:
        raise re_.PreLockStop('评测单元的启动核对：'+'；'.join(problems))
    resume_repo = re_.resume_repo_of(root)                                                 # 修订 6：续跑记录在续跑仓库（main）里核
    if o.dry_run:
        pre = re_.bench_prelock(root, list(rc.ARM_ORDER) if o.read == 'first' else [o.combo], bench, stage_dir=root/'bench_stage', repo=resume_repo)
        print(f'锁前核对：过；排除 {pre["excluded"]}；(甲) 不可判 {pre["undecidable_before_truth"]}')
        return 0
    rules = spy.bench_rules(root, unit=f'eval-bench-{o.read}', stage_dir=root/'bench_stage', pred_root=root/'runs', atropos_root=root/'final',
                            extra=(root/'eval', root/'locks', root/'analysis', root/'configC', resume_repo/re_.RESUME_DIR_REL, root/'stops'),
                            record_dirs=(root/'eval', root/'analysis'), allow_images=False, resume_repo=resume_repo)
    rules.allow_truth = True
    del run_r4
    with spy.UnitSpy(rules, expect_reads=False, truth_readers=TRUTH_READERS_BENCH) as s:
        doc = re_.bench_eval(root, bench, read=o.read, combo=o.combo, stage_dir=root/'bench_stage', repo=resume_repo, code_commit=code['commit'], plan_sha256=rc.PLAN_SHA256)
    record = s.record()
    record['truth_read_problems'] = spy.truth_reads(record, expected_keys=[(subject, kind) for subject in bench.ordered_subjects() for kind in ('seg35', 'seg4')]) if 'truth_sha256' in doc else \
        spy.truth_reads(record)
    record['passed'] = record['passed'] and not record['truth_read_problems']
    rc.write_json(root/'eval'/f'opened_files_{o.read}.json', record)
    return 0 if not doc.get('failed') and record['passed'] else 8


def label_key(args, kwargs):
    """read_labels(files, subject, guard, *, fold=None, kind='seg35') 的键：(受试者, 种类)。"""
    return (args[1] if len(args) > 1 else kwargs.get('subject'), kwargs.get('kind', 'seg35'))


TRUTH_READERS_BENCH = (('bayesreg.seg.unet.data', 'read_labels', label_key),)
TRUTH_READERS_DEV = (('_mc0_evaluate', 'truth_labels', lambda args, kwargs: (args[0] if args else kwargs.get('pair'),)),)


def eval_bench_checks(root, bench, batch):
    """评测单元建锁之前（复核次要 10）：plan.md 的 sha256 是钉住的；配对表的 sha256 与 batch.json 钉的相同；台的登记与附属文件照 batch.json 的钉再读一遍。"""
    import r4_bench as rb
    problems = []
    plan = rc.plan_problem()
    if plan:
        problems.append(plan)
    if rb.pairs_table_sha256(bench) != batch.get('pairs_table_sha256'):
        problems.append('配对表的 sha256 与 batch.json 不同')
    try:
        rb.load_bench_registry(bench, dict(registry_sha256=batch.get('registry_sha256'), contact_sha256=batch.get('contact_sha256')))
    except rc.Refused as exc:
        problems.append(str(exc))
    return problems


def decide_bench(o, root=rc.R4_ROOT):
    import r4_bench as rb
    import r4_eval as re_
    decision = re_.bench_decide(root, rb.load_bench())
    print(json.dumps(dict(reading=decision.get('reading'), order=decision.get('order'), combo=decision.get('combo')), ensure_ascii=False))
    return 0


def dev_state(root=rc.R4_ROOT, repo=None):
    import r4_eval as re_
    return re_.dev_state(root, repo)


def next_dev(o, root=rc.R4_ROOT):
    import r4_decide as rd
    order, feasibility, readings = dev_state(root)
    step = rd.next_dev_step(order, feasibility, readings)
    print(json.dumps(dict(order=order, feasibility=feasibility, readings=readings, next=step), ensure_ascii=False))
    return 0


def precheck_hs(o, root=rc.R4_ROOT):
    """HS 的 19 个输出重建，sha256 对 repaired_u_sha256（不读真值；起草员风险 3）。只打印，不写文件。"""
    import r4_eval as re_
    rows, bad = {}, []
    for pair in re_.dev_pairs():
        try:
            pin, _ = re_.run_hx_admit(rc.ROOT/rc.DEV_STAGE_REL, pair)
            _, record = re_.hs_rebuild(pair, pin.grid)
            rows[pair] = dict(ok=True, repaired_u_sha256=record['repaired_u_sha256'])
        except Exception as exc:                                                        # noqa: BLE001 - reported
            rows[pair] = dict(ok=False, error=f'{type(exc).__name__}: {exc}')
            bad.append(pair)
    print(json.dumps(dict(n=len(rows), n_ok=len(rows)-len(bad), bad=bad, rows=rows), ensure_ascii=False, indent=1))
    return 0 if not bad else 8


def eval_dev(o, root=rc.R4_ROOT):
    import r4_eval as re_
    import run_r4
    b = run_r4.batch(root, 'dev')
    problems, code = run_r4.batch_identity(b)                                                # 修订 6
    plan = rc.plan_problem()
    resume_repo = re_.resume_repo_of(root)
    import r4_spy as spy
    if o.dry_run:
        re_.dev_prelock(root, o.arm, pairs=re_.dev_pairs(), hs_root=rc.HS_ROOT, code_problems=problems, plan_problem=plan, read_number=o.read, resume_repo=resume_repo)
        print('建锁之前的清单：过')
        return 0
    rules = spy.dev_rules(root, unit=f'eval-dev-{o.read}', extra=(rc.ROOT/rc.DEV_STAGE_REL, rc.HS_ROOT/'runs', rc.HS_ROOT/'analysis', root/'dev_final', root/'dev_runs', root/'dev_eval', root/'locks',
                                                                  root/'analysis', resume_repo/re_.RESUME_DIR_REL, root/'stops'),
                          record_dirs=(rc.HS_ROOT/'runs', rc.HS_ROOT/'analysis', root/'dev_eval', root/'analysis'), resume_repo=resume_repo)
    re_.evaluator()                                                                       # the module whose truth_labels the spy counts (修订 3 次要 3)
    with spy.UnitSpy(rules, expect_reads=False, truth_readers=TRUTH_READERS_DEV) as s:
        doc = re_.dev_eval(root, o.arm, o.read, hs_root=rc.HS_ROOT, code_commit=code['commit'], plan_sha256=rc.PLAN_SHA256, code_problems=problems, plan_problem=plan,
                           resume_repo=resume_repo)                                                # 修订 8：hs_root 在调用时取（与白名单用的 rc.HS_ROOT 是同一个）
    record = s.record()
    record['truth_read_problems'] = spy.truth_reads(record, expected_keys=[(p,) for p in doc.get('pairs') or []]) if doc.get('lock_line') else spy.truth_reads(record)
    record['passed'] = record['passed'] and not record['truth_read_problems']
    rc.write_json(root/'dev_eval'/f'opened_files_{o.read}.json', record)
    return 0 if not doc.get('failed') and record['passed'] else 8


def decide_dev(o, root=rc.R4_ROOT):
    import r4_eval as re_
    decision = re_.dev_decide(root, o.read)
    print(json.dumps(dict(reading=decision.get('reading'), better=decision.get('better')), ensure_ascii=False))
    return 0


def dispatch(o):
    table = {'prepare-bench': prepare_bench, 'atropos-bench': atropos_bench, 'seg-bench': seg_bench, 'smoke': smoke, 'devcheck': devcheck, 'eval-bench': eval_bench,
             'decide-bench': decide_bench, 'next-dev': next_dev, 'precheck-hs': precheck_hs, 'eval-dev': eval_dev, 'decide-dev': decide_dev}
    return table[o.command](o)
