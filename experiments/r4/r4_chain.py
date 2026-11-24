"""R4 不读真值的整条链（合同 C25.6-10、C25.6-12、C25.8-2），以 HE（experiments/he/run_hx.py，未改）为参照实现。

每臂每对：T88（加上臂的选项）→ boundary_taper → 修复 RF05（1200 s 保险丝）→ 判定证书（修复进程自己的，CPU FP64，不读标签）→ 失败时退路（配置 C → 恒等）→ line_status。
修复之前只核 T88 的 run.json（场文件的 sha256、u_sha256、status done、oracle false；C22.3-1）；不读任何评测记录。
退路一：每对至多跑一次配置 C（<R4>/configC/<pair>/ 或 dev_configC/<pair>/），各臂共用；configc_fallback 是 run_hx.configc_fallback 的逐语句复制，只改结果路径（测试做 AST 对照）。
定稿单元：一臂的输出都定了之后，对每对的输出数组再认证一次；与判定证书不一致时照 C22.3-9 (d) 的机械规则（第三次认证 → 退路）。都在读真值之前做完。
不 import experiments/hs/decide_hs；不调 run_fl.evaluate_*、repair_one、repair_allowed、fl_core.chosen_repairs、run_hs 与 rp_core 的退路函数（测试做传递闭包的静态核对）。

进程：
  supervisor（run_r4.py pair）  准入 → T88 子进程 → 修复子进程 → 判定 → 退路 → 输出 <out>/output/；line.json。
  t88 子进程                    准入再做一次（台：台守卫 + load_pair；开发集：HE 的两道锁）→ 通道（台：台的读取器；开发集：fast_channels.read_predicted + HS 的 UNetSource）→ 掩膜 / 软通道 → T88。
  repair 子进程                 准入再做一次 → 只读 t88/run.json 与场文件 → 渐变 → 修复 → 判定证书。
退出码：0 有输出；9 读真值之前的失败（hx_provenance_rejected 或 hx_implementation_error：队列停下，交总负责人与 BZ）；8 代码身份不符；2 用法或拒绝启动。
"""
import ast  # noqa: F401  (tests use the module's source)
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import traceback

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _d in ('d0', 'fl', 'ef', 'mc0', 'he'):
    _p = str(EXPERIMENTS/_d)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import firewall_c as fc  # noqa: E402
import he_core as he  # noqa: E402
import run_hx  # noqa: E402
import r4_common as rc  # noqa: E402
import r4_options as r4o  # noqa: E402

core = he.core
SCRIPT = HERE/'run_r4.py'
PY = fc.PY
STOP_EXIT = he.STOP_QUEUE_EXIT
T88_IMPLEMENTATION = 'implementation_error'
ROLES = ('bench', 'dev')


# ---- 上下文 --------------------------------------------------------------------------------------------------------------------------------------
def settings_for(arm, role, *, bench_registry_sha256=None):
    """HE 的冻结设定（配置 C、证书、修复、子进程超时、设备上限），T88 一段按臂的设定改（C25.1-1）；台上的分割器换成台的登记（D134）。"""
    spec = rc.arm_spec(arm)
    s = he.frozen_settings()
    pre = dict(s['t88']['pre'])
    pre.update(iterations=[50, 50, 100, int(spec['native'])], sigma_field=float(spec['sigma_field']))
    s['t88']['pre'] = pre
    s['channels'].update(channels=spec['channels'], D_factor=float(spec['D_factor']))
    if role == 'bench':
        s['segmenter'] = dict(segmenter_id=rc.BENCH_SEGMENTER_ID, policy=rc.BENCH_POLICY, registry=rc.BENCH_REGISTRY_REL, registry_sha256=bench_registry_sha256)
    what = 'Hx24-RF05-F1200-v4' if arm == rc.CONTROL else "Hx24-RF05-F1200-v4 with the arm's options"
    s['name'] = f'R4 {arm} on {role} ({what}; contract C25)'
    s['r4'] = dict(arm=arm, role=role, spec={k: list(v) if isinstance(v, tuple) else v for k, v in spec.items()}, spec_sha256=rc.spec_sha256(spec), options=rc.options_of(spec))
    return s


def default_paths(role, root):
    root = Path(root)
    paths = dict(allowlist=str(rc.ROOT/rc.ALLOWLIST_REL), frozen=str(core.FROZEN), cache=str(root/'cache'))
    if role == 'dev':
        paths.update(pred_root=str(rc.HS_PRED_ROOT), models_root=str(he.V4_SOURCES['models_root']), image_root=str(rc.DATA_ROOT), registry=str(rc.ROOT/rc.HS_REGISTRY_REL),
                     folds=str(core.make_inputs().UNetSource().folds), folds_sha256=core.make_inputs().UNetSource().folds_sha256, atropos_root=str(rc.DEV_ATROPOS_ROOT))
    else:
        paths.update(pred_root=str(root/'seg'/rc.BENCH_SEGMENTER_ID/'pred'), atropos_root=str(root/'atropos'), models_root=str(rc.RESULTS/'seg'/'unet35_v4_full'))
    return paths


def make_context(role, arm, pair, *, root, stage_dir, out=None, paths=None, device='cuda', launcher='direct', check_identity=True, workers=None, bench_pins=None, test=False):
    """一对一臂的上下文：HE 的键（HE 的函数照用）加 R4 的键。check_identity False 只在测试里（test=True）：标记为测试上下文，写在结果根之外。
    launcher（修订 3 次要 6）：写死 direct——supervisor 已在它自己的 tag-r4-* 单元里，配置 C 的冻结运行在同一单元里起（firewall_c.DirectLauncher），不再嵌套 tag-he-c-* 单元。"""
    if launcher != 'direct':
        raise ValueError('R4 runs configuration C with the DirectLauncher (the supervisor is in its own tag-r4- unit)')
    if role not in ROLES:
        raise ValueError(f'role {role!r}')
    root = Path(root)
    rc.check_results_place(root, real=not test)
    settings = settings_for(arm, role, bench_registry_sha256=(bench_pins or {}).get('registry_sha256'))
    out = Path(out) if out is not None else root/('runs' if role == 'bench' else 'dev_runs')/arm/pair
    paths = default_paths(role, root) if paths is None else paths
    ctx = dict(schema=rc.SCHEMA.format('context'), role=role, arm=arm, pair=pair, stage_dir=str(stage_dir), out=str(out), root=str(root), device=device, workers=core.SETUP_WORKERS if workers is None else workers,
               settings=settings, settings_sha256=he.settings_sha256(settings), settings_frozen=he.is_frozen(settings), paths=paths, launcher=dict(kind=launcher),
               check_identity=check_identity, test_context=run_hx.TEST_CONTEXT if (test or check_identity is not True) else None, code_head=run_hx.code_head(), pair_set='dev',
               code_root=str(rc.ROOT.resolve()),                                                                  # 修订 6a M1：子进程核 core.ROOT 就是它
               configc_root=str(root/('configC' if role == 'bench' else 'dev_configC')), bench_pins=bench_pins, created_utc=rc.utc())
    if check_identity is not True and not test:
        raise ValueError('只有测试上下文可以不核代码身份')
    return ctx


def context_path(ctx):
    return Path(ctx['out'])/'context.json'


def read_context(path):
    ctx = rc.read_json(path)
    if ctx is None or ctx.get('schema') != rc.SCHEMA.format('context'):
        raise SystemExit(f'{path}: not an R4 context')
    if he.settings_sha256(ctx['settings']) != ctx['settings_sha256']:
        raise SystemExit(f'{path}: the settings differ from their recorded sha256')
    return ctx


def spec_of(ctx):
    return {k: tuple(v) if isinstance(v, list) else v for k, v in ctx['settings']['r4']['spec'].items()}


# ---- 准入 --------------------------------------------------------------------------------------------------------------------------------------
_BENCH = {}


def bench_objects(ctx):
    """台的对象（每个进程各自从钉住的文件建：load_bench、load_bench_registry）；测试可以预先放进 _BENCH。"""
    import r4_bench as rb
    key = ctx['root']
    if key not in _BENCH:
        bench = rb.load_bench()
        registry, contact, record = rb.load_bench_registry(bench, ctx['bench_pins'] or {})
        _BENCH[key] = (bench, registry, contact, record)
    return _BENCH[key]


def refusals():
    import r4_bench as rb
    return run_hx.admission_refusals()+(rb.BenchRefused,)


def admit(ctx):
    """(PairInput, 准入记录)：台上先台守卫（打开任何受试者文件之前），再 load_pair（开发白名单）；开发集照 HE 的两道锁（封存名单、load_pair）。"""
    if ctx['role'] == 'bench':
        import r4_bench as rb
        bench = bench_objects(ctx)[0]
        return rb.load_bench_pair(bench, ctx['stage_dir'], ctx['pair'])
    if ctx.get('pair_set', 'dev') != 'dev':
        raise ValueError('R4 never runs a hold-out pair')
    record = run_hx.sealed_admission(ctx)                                       # HE's second lock (the sealed lists on the names and the manifest), then the allowlist (load_pair)
    return run_hx.load_images(ctx), record


# ---- 通道的读取（HE 的 SideOutcomes，侧读取函数换成台的） ------------------------------------------------------------------------------------------
class SideOutcomes(run_hx.SideOutcomes):
    """run_hx.SideOutcomes，被观察的侧读取函数可换：开发集 fast_channels.read_side（HE 原样），台上 r4_bench.bench_read_side。"""

    def __init__(self, module_name='fast_channels', attr='read_side'):
        self.module_name, self.attr = module_name, attr

    def __enter__(self):
        import importlib
        import threading
        self.fast, self.mi, self.local = importlib.import_module(self.module_name), core.make_inputs(), threading.local()
        self.real_qc, self.real_side = self.mi.alignment_qc, getattr(self.fast, self.attr)
        self.rows, self.sides = [], {}
        recorder = self

        def observed(*args, **kwargs):
            row = recorder.real_qc(*args, **kwargs)
            recorder.rows.append(row)
            recorder.local.qc = row
            return row

        def side(layer, pin, name, *args, **kwargs):
            recorder.local.qc = None
            try:
                return recorder.real_side(layer, pin, name, *args, **kwargs)
            except SystemExit as exc:
                qc = recorder.local.qc
                own_qc = qc is not None and qc.get('passed') is False
                recorder.sides[name] = dict(outcome='qc' if own_qc else 'refused', reason=str(exc)[:1500], alignment_qc=qc)
                if own_qc:
                    return None, dict(qc_refused=True)
                raise
        self.mi.alignment_qc = observed
        setattr(self.fast, self.attr, side)
        return self

    def __exit__(self, *exc):
        self.mi.alignment_qc = self.real_qc
        setattr(self.fast, self.attr, self.real_side)
        return False


def read_channels(ctx, pin, admission):
    """(predicted, sources 的锚核对问题, qc 行, 两侧的结局)。台：台的读取器（BenchLayer），oracle 按台的 id 与 policy 重算；开发集：HE 的写法（HS 的 UNetSource，oracle 按 unet35_v4_ens 重算）。"""
    import fast_channels as fast
    if ctx['role'] == 'bench':
        import r4_bench as rb
        from bayesreg.io.pair import ProvenanceRejected
        from jd_curl.validation import InputValidationError
        bench, registry, contact, record = bench_objects(ctx)
        try:
            layer = rb.BenchLayer(pin, bench, registry, contact, record, pred_root=ctx['paths']['pred_root'], models_root=ctx['paths'].get('models_root'))
        except (ProvenanceRejected, InputValidationError) as exc:                     # 复核 I1：读取器在两侧之前的拒绝是 provenance_rejected，不是崩溃
            raise SystemExit(f'R4 bench structure layer refused ({type(exc).__name__}: {exc})') from None
        with SideOutcomes('r4_bench', 'bench_read_side') as outcomes:
            try:
                predicted = rb.bench_read_predicted(pin, bench, layer, seg_root=Path(ctx['paths']['atropos_root']), admission=admission, workers=ctx['workers'])
            finally:
                qc_rows, sides = list(outcomes.rows), dict(outcomes.sides)
        if sides:
            raise SystemExit(outcomes.reason())
        problems = bench_source_problems(predicted.record, record)
        return predicted, problems, qc_rows, sides
    import he_c
    with run_hx.SideOutcomes() as outcomes:
        try:
            predicted = fast.read_predicted(pin, unet=run_hx.unet_source(ctx), seg_root=Path(ctx['paths']['atropos_root']), allowlist=ctx['paths']['allowlist'], workers=ctx['workers'])
        finally:
            qc_rows, sides = list(outcomes.rows), dict(outcomes.sides)
    if sides:
        raise SystemExit(outcomes.reason())
    sources = predicted.record
    reader_mark = {k: sources.get(k) for k in ('oracle', 'truth_labels_read')}
    sources.update(he_c.mark(he.CONTRACT_SEGMENTER['segmenter_id'], sources['pair_keys_read'], sources['sides']), reader_mark=reader_mark, mark_rule=run_hx.MARK_RULE)
    return predicted, he.channel_source_problems(sources), qc_rows, sides


def bench_source_problems(sources, registry_record):
    problems = []
    sides = (sources or {}).get('sides') or {}
    if sorted(sides) != ['fixed', 'moving']:
        problems.append('channel_sources.sides: fixed and moving required')
    for side, s in sorted(sides.items()):
        for key, want in (('segmenter_id', rc.BENCH_SEGMENTER_ID), ('policy', rc.BENCH_POLICY), ('registry_sha256', registry_record['sha256'])):
            if (s or {}).get(key) != want:
                problems.append(f'channel_sources.sides.{side}.{key} {(s or {}).get(key)!r} is not {want!r}')
    return problems


# ---- T88 子进程 ----------------------------------------------------------------------------------------------------------------------------------
def start_checks(ctx):
    """T88 子进程开始时（复核次要 12）：通道的 σ 与颜色表是读取器的（fast_channels.SIGMA、mc0_core.COLOUR_TABLE_SHA256）；开发集再核分割器的锚（he_core.segmenter_start_problems：
    设定、此刻读到的登记与它的模型）。返回问题列表；不空就 provenance_rejected。"""
    import fast_channels as fast
    s = ctx['settings']
    ch = s['channels']
    problems = []
    if ch.get('sigma') != fast.SIGMA or ch.get('colour_table_sha256') != core.COLOUR_TABLE_SHA256:
        problems.append(f"the settings name channel sigma {ch.get('sigma')}, colour table {str(ch.get('colour_table_sha256'))[:12]}: not the reader's")
    if ctx['role'] == 'dev':
        anchor_problems, _ = he.segmenter_start_problems(s, registry=Path(ctx['paths'].get('registry') or he.anchor_registry()))
        problems += anchor_problems
    return problems


def started(directory, ctx, stage):
    return run_hx.started(directory, ctx, stage)


def t88_main(ctx):
    """run_hx.t88_main 的流程（tissue_prior off）加臂的选项：A 的掩膜（C25.2-4：读完两幅图的标签、建通道的同一步里建好四层），B 的软通道（C25.3），C1–C3 的数。
    掩膜抛异常、软通道的 argmax 不一致：status implementation_error（原因串 mask_error、soft_argmax_mismatch）：hx_implementation_error，队列停下。"""
    t0 = perf_counter()
    fz = core.frozen_api()
    import torch
    import fast_channels as fast
    import fl_channels as channels
    import run_ef as efrun
    import run_mc0 as run
    import run_mc0p as mc0p
    s = ctx['settings']
    spec = spec_of(ctx)
    out = started(Path(ctx['out'])/'t88', ctx, 't88')
    base = dict(schema=rc.SCHEMA.format('t88'), pair=ctx['pair'], arm=ctx['arm'], role=ctx['role'], stage_dir=ctx['stage_dir'], r4_spec=s['r4']['spec'], r4_spec_sha256=s['r4']['spec_sha256'],
                r4_options=s['r4']['options'], settings_sha256=ctx['settings_sha256'])
    try:
        pin, admission = admit(ctx)
    except refusals() as exc:
        rc.write_json(out/'run.json', dict(base, status=he.T88_REJECTED, reason=f'admission: {type(exc).__name__}: {exc}'[:2000], oracle=None, field=None, ended_utc=rc.utc()))
        return 0
    device = ctx['device']
    device_cap = mc0p.limit_device_memory(torch, device, limit=s['device_limit_gib']*2**30)
    config, pre = r4o.preopt_config(fz, s['t88']['pre'], spec)
    status, reason, u, record, scale, scale_check, sources, digest, qc_rows, sides = 'done', None, None, None, None, None, None, None, [], {}
    rejection_stage = source_problems = mask_record = soft_record = None
    seconds_labels = seconds_channels = seconds_task = None
    start_problems = start_checks(ctx)
    try:
        if start_problems:
            raise SystemExit(f'the start checks of the T88 child: {start_problems}')
        predicted, source_problems, qc_rows, sides = read_channels(ctx, pin, admission)
        seconds_labels = perf_counter()-t0
        sources = predicted.record
        if ctx['role'] == 'bench':
            sources.update(bench=dict(admission=admission, pairs_row=bench_objects(ctx)[0].pair_of(ctx['pair']), registry=bench_objects(ctx)[3], d24_exception='C25.6-7'))
        digest = efrun.labels_digest(predicted.labels)
        if source_problems:
            status, reason, rejection_stage = he.T88_REJECTED, f'the segmenter after the channels: {source_problems}'[:2000], 'segmenter_anchor'
        elif sources['oracle'] is not False:
            status, reason = 'not_predicted_channels', f"channel sources: oracle {sources['oracle']}, members read {sources['pair_keys_read']}; no optimization"
        else:
            masks = None
            if spec.get('field_mask') is not None:
                try:
                    m3 = r4o.native_mask(predicted.labels['fixed'], predicted.labels['moving'])
                    _, masks, mask_record = r4o.level_masks(m3, pin.grid, fz.four_level_grids(pin.grid))
                    mask_record = dict(mask_record, sigma_small=spec['sigma_small'])
                    del m3
                except Exception as exc:                                         # noqa: BLE001 - C25.2-4: the mask's computation raising is an implementation error
                    raise r4o.MaskError(f'{type(exc).__name__}: {exc}') from exc
            soft = None
            if spec['channels'] == 'soft':
                soft, rows = {}, {}
                try:
                    for side, (path, sha) in r4o.soft_sources(predicted).items():
                        soft[side], rows[side] = r4o.soft_side(path, sha, pin.grid.shape, predicted.labels[side], workers=ctx['workers'])
                except r4o.IMPLEMENTATION_ERRORS:
                    raise
                except Exception as exc:                                         # noqa: BLE001 - 复核 I1: the soft read-back raising is an implementation error of option B
                    raise r4o.OptionError('soft_error', f'{type(exc).__name__}: {exc}') from exc
                soft_record = dict(sides=rows, host_peak_rss_gib=run_hx.env_record()['host_peak_rss_gib'])
            try:
                stacks, scale = r4o.prepare(spec, pin, predicted, core.load_table(), device, soft=soft, workers=ctx['workers'])
            except (SystemExit, r4o.IMPLEMENTATION_ERRORS):
                raise
            except Exception as exc:                                             # noqa: BLE001 - 复核 I1
                if spec['channels'] != 'soft':
                    raise
                raise r4o.OptionError('soft_prepare_error', f'{type(exc).__name__}: {exc}') from exc
            del predicted, soft
            seconds_channels = perf_counter()-t0
            engine = r4o.core_for(spec, masks)
            try:
                u, record = engine.optimize_k(stacks, pin.grid, config, device=device, started=t0, observer=run.vram_guard(torch, device, s['device_limit_gib']*2**30), workers=ctx['workers'],
                                              record_level_norms=True, downsample=fast.FastDownsample())
            except (run.VramLimit, torch.cuda.OutOfMemoryError, FloatingPointError):
                raise
            except Exception as exc:                                             # noqa: BLE001 - 复核 I1: run_level_k_masked raising is an implementation error of option A
                if spec.get('field_mask') is None:
                    raise
                raise r4o.OptionError('mask_runner_error', f'{type(exc).__name__}: {exc}') from exc
            seconds_task = perf_counter()-t0
            del stacks
            scale_check = channels.scale_check_of(record, scale)
            if record['stopped'] is not None:
                status, reason = 'time_cap', f"the wall cap {pre['seconds']} s at {record['stopped']}"
            elif not scale_check['passed']:
                status, reason = 'scale_mismatch', f"||g_D|| of the native tensors {scale_check['g_norm_scaled']} differs from {scale_check['target']}"
        if status == 'time_cap':
            np.savez(out/'field_stopped.npz', u=u)
    except SystemExit as exc:
        only_qc = bool(sides) and all(x['outcome'] == 'qc' for x in sides.values())
        status, reason = (he.T88_QC if only_qc else he.T88_REJECTED), str(exc)[:2000]
    except r4o.IMPLEMENTATION_ERRORS as exc:
        status, reason = T88_IMPLEMENTATION, f'{exc.reason}: {exc}'[:2000]
    except run.VramLimit as error:
        status, reason = 'vram_stop', str(error)
    except torch.cuda.OutOfMemoryError as error:
        status, reason = 'cuda_oom', str(error)[:500]
    except FloatingPointError as error:
        status, reason = 'nonfinite', str(error)[:500]
    field = None
    if status == he.T88_DONE:
        name = rc.field_file(spec)
        np.savez(out/name, u=u)
        field = dict(file=name, sha256=core.sha_file(out/name), u_sha256=he.array_sha256(u))
    rc.write_json(out/'run.json', dict(
        base, status=status, reason=reason, rejection_stage=rejection_stage, oracle=None if sources is None else sources.get('oracle'), method=he.METHOD, role_of_line='R4 bench or development feasibility (C25); not F_B',
        spec=dict(k=spec['k'], channels=spec['channels'], weight=spec['weight'], D_factor=spec['D_factor']), config=pre, start_checks=start_problems, scale=scale, scale_check=scale_check, labels_sha256=digest, channel_sources=sources,
        alignment_qc=qc_rows, side_refusals=sides, admission=admission, optimizer=record, field=field, taper=s['t88']['taper'], field_mask_record=mask_record, soft_channels=soft_record,
        engine_replaced=spec.get('field_mask') is not None, seconds_task=seconds_task, seconds_labels=seconds_labels, seconds_channels=seconds_channels, seconds_total=perf_counter()-t0,
        frozen_commit=core.FROZEN_COMMIT, data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], fixed_subject=pin.fixed_subject, moving_subject=pin.moving_subject,
                                                    keys_read=list(pin.provenance['keys_read']), label_firewall=pin.provenance['label_firewall'], subject_guard=pin.provenance['subject_guard']),
        env=run_hx.env_record(torch, device), device_cap=device_cap, ended_utc=rc.utc()))
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    return 0


# ---- 修复子进程 --------------------------------------------------------------------------------------------------------------------------------
def source_checked(t88_dir, fz, pin, ctx):
    """(渐变后的 u, 来源记录)。C22.3-1：t88/run.json 说 done、oracle false，场文件的 sha256 是记录的，u 的散列是记录的 u_sha256；渐变在这里做。不读评测写的任何东西。ValueError 否则。"""
    t88_dir = Path(t88_dir)
    record = rc.read_json(t88_dir/'run.json')
    if record is None or record.get('schema') != rc.SCHEMA.format('t88'):
        raise ValueError(f'{t88_dir}: no T88 record')
    if record.get('status') != he.T88_DONE or record.get('oracle') is not False or record.get('pair') != ctx['pair'] or record.get('arm') != ctx['arm']:
        raise ValueError(f"{t88_dir}: not a done field of this pair and arm with oracle false: {record.get('status')}, {record.get('oracle')}, {record.get('pair')}, {record.get('arm')}")
    if record.get('data', {}).get('pair_sha256') != pin.provenance['pair_sha256']:
        raise ValueError(f'{t88_dir}: the field is of another pair.npz')
    field = record['field']
    if field['file'] != rc.field_file(spec_of(ctx)) or core.sha_file(t88_dir/field['file']) != field['sha256']:
        raise ValueError(f"{t88_dir}/{field['file']} differs from run.json")
    with np.load(t88_dir/field['file'], allow_pickle=False) as z:
        raw = np.array(z['u'])
    if he.array_sha256(raw) != field['u_sha256']:
        raise ValueError(f"{t88_dir}/{field['file']}: u differs from run.json u_sha256")
    tick = perf_counter()
    u, _ = fz.boundary_taper(raw, pin.grid, ctx['settings']['t88']['taper'])
    u = np.ascontiguousarray(u)
    return u, dict(directory=str(t88_dir), file=field['file'], field_sha256=field['sha256'], u_sha256=field['u_sha256'], tapered_u_sha256=he.array_sha256(u), seconds_taper=perf_counter()-tick,
                   seconds_task=record.get('seconds_task'), checked='t88/run.json and the field file by their own sha256; the taper in this process (C22.3-1); no evaluation record read')


def repair_main(ctx):
    """run_hx.repair_main 的流程：只读 T88 的 run.json 与场文件，渐变，局部修复（保险丝 1200 s），修复进程自己的证书是判定证书（C22.4-1）。"""
    t0 = perf_counter()
    fz = core.frozen_api()
    run_hx.enable_jit(ctx)
    import torch
    from jd_curl.three_d.feasibility import certify_map
    import repair as rp
    import run_ef as efrun
    import run_mc0 as run
    import run_mc0p as mc0p
    s = ctx['settings']
    out = started(Path(ctx['out'])/'repair', ctx, 'repair')
    device = ctx['device']
    record = dict(schema=rc.SCHEMA.format('repair'), pair=ctx['pair'], arm=ctx['arm'], role=rp.ROLE+'; R4: its certificate of the final v is the decision (C22.4-1)', settings_sha256=ctx['settings_sha256'])
    try:
        pin, _ = admit(ctx)
    except refusals() as exc:
        rc.write_json(out/'repair.json', dict(record, status=he.T88_REJECTED, passed=False, reason=f'admission: {type(exc).__name__}: {exc}'[:2000], env=run_hx.env_record(), ended_utc=rc.utc()))
        return 0
    try:
        device_cap = mc0p.limit_device_memory(torch, device, limit=s['device_limit_gib']*2**30)
        u, source = source_checked(Path(ctx['out'])/'t88', fz, pin, ctx)
        config = he.certificate_config(s)
        if asdict(config) != asdict(run.certificate_config()):
            raise ValueError(f'the settings certificate {asdict(config)} is not configuration C {asdict(run.certificate_config())}')
        tick = perf_counter()
        efrun.tiny_warmup(certify_map, config, rp.locate_cells)
        warm_numba = perf_counter()-tick
        tick = perf_counter()
        warmed = efrun.device_warmup(device, certify_map, config)
        warm_device = perf_counter()-tick
        settings = rp.RepairConfig(**s['repair'])
        tick = perf_counter()
        cells = rp.locate_cells(u, pin.grid, config)
        first_locate = perf_counter()-tick
        recorder = he.RecordingCertify(certify_map)
        tick = perf_counter()
        result = rp.repair_field(u, pin.grid, cells, settings, certify=recorder, certificate_config=config, device=device)
        seconds_repair = first_locate+perf_counter()-tick
        np.savez(out/'repair.npz', nodes=result.attempt_nodes, values=result.attempt_values)
        decision = None
        if result.passed:
            v = np.ascontiguousarray(result.u)
            digest = he.array_sha256(v)
            final = recorder.of(digest)
            if final is None or not he.passes(final):
                raise ValueError('no passing certificate of the returned field among the calls of certify')
            doc = he.certificate_document(final, config, process=f'the repair process (run_r4.py repair, pid {os.getpid()}; reads no label)', array_sha256_of_u=digest)
            np.savez(out/'certified.npz', u=v)
            he.write_json(out/'cert.json', doc)
            decision = dict(file='certified.npz', sha256=core.sha_file(out/'certified.npz'), state_hash=digest, cert='cert.json', cert_sha256=core.sha_file(out/'cert.json'))
        record.update(status='certified' if result.passed else 'not_certified', passed=result.passed, reason=result.reason, eps=settings.eps, config=asdict(settings), source=source, n_located=len(cells),
                      initial_certificate=result.initial_certificate, certificate=result.certificate, rounds=result.rounds, decision=decision, n_certify_calls=len(recorder.results),
                      seconds_warmup=warm_numba+warm_device, device_warmup=warmed, seconds_first_locate=first_locate, seconds_repair=seconds_repair,
                      attempt=dict(file='repair.npz', sha256=core.sha_file(out/'repair.npz'), n_free_nodes=len(result.attempt_nodes)), certificate_config=asdict(config), frozen_commit=core.FROZEN_COMMIT,
                      data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], keys_read=list(pin.provenance['keys_read'])), device_cap=device_cap)
        record['wall_hit'] = he.hit_wall(record)
    except Exception as exc:                            # C22.3-9 (b): the supervisor goes to the fallback; the evidence stays
        record.update(status='error', passed=False, error=f'{type(exc).__name__}: {exc}'[:2000], traceback=traceback.format_exc()[-4000:])
    record.update(env=run_hx.env_record(torch, device), seconds_total=perf_counter()-t0, ended_utc=rc.utc())
    rc.write_json(out/'repair.json', record)
    return 0


# ---- 子进程 --------------------------------------------------------------------------------------------------------------------------------------
class SubprocessChild:
    kind = 'subprocess'

    def __call__(self, stage, ctx):
        timeout = {'t88': ctx['settings']['children']['t88_timeout_seconds'], 'repair': ctx['settings']['children']['repair_timeout_seconds']}.get(stage)
        argv = fc.check_launch([PY, '-B', str(SCRIPT), stage, '--context', str(context_path(ctx))])
        with (Path(ctx['out'])/f'{stage}.log').open('a') as f:
            try:
                return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                f.write(f'\n{rc.utc()} killed at the timeout {timeout} s\n')
                return 'timeout'


class InProcessChild:
    """同一进程里跑（测试），子进程那一层的 spy 照样装（run_child_spied），与 SubprocessChild 走的是同一个函数。"""
    kind = 'in_process'

    def __call__(self, stage, ctx):
        try:
            return run_child_spied(read_context(context_path(ctx)), stage)
        except Exception:                                                                    # noqa: BLE001 - as a crashed child process: exit non-zero, the rules of D74 and I1 judge
            return 1


SPY_FILE = 'opened_files.json'
KILLED_NO_RECORD = 'killed, no spy record'


def run_child_spied(ctx, stage, *, rules=None):
    """T88 或修复子进程在 spy 之下（C25.10-5；复核 B2）：本对的白名单（r4_spy.pair_rules）、原 pair.npz 的成员守卫与正条件。spies.Breach 被捕获：记录照写，退出 8。
    记录写在 <out>/<stage>/opened_files.json；spy 不过：退出 8。supervisor 读这份记录（child_spy_problem）。"""
    import r4_spy as spy
    import spies
    rules = spy.pair_rules(ctx, stage) if rules is None else rules
    breach, code, crashed = None, None, None
    s = spy.UnitSpy(rules, Path(ctx['stage_dir'])/ctx['pair']/'pair.npz')
    try:
        with s:
            try:
                code = t88_main(ctx) if stage == 't88' else repair_main(ctx)
            except spies.Breach as exc:
                code, breach = 8, f'{type(exc).__name__}: {exc}'
            except BaseException as exc:                                                     # 修订 3 次要 1：记录照写，再把异常抛出，让 D74 与 I1 的规则去判
                crashed = f'{type(exc).__name__}: {exc}'[:1000]
                raise
    finally:
        refused = admission_refused(ctx, stage)                                              # 修订 3 次要 2：子进程自己的准入被拒（什么都没读）：不套"至少读到一次"的正条件
        record = dict(s.record(expect_reads=False if refused else None), breach=breach, crashed=crashed, admission_refused=refused)
        if breach:
            record['passed'] = False
        d = Path(ctx['out'])/stage
        d.mkdir(parents=True, exist_ok=True)
        rc.write_json(d/SPY_FILE, record)
    return code if record['passed'] else 8


def admission_refused(ctx, stage):
    """子进程自己的准入是否被拒：t88/run.json 或 repair/repair.json 记 provenance_rejected，原因以 admission 开头。"""
    doc = rc.read_json(Path(ctx['out'])/stage/('run.json' if stage == 't88' else 'repair.json')) or {}
    return doc.get('status') == he.T88_REJECTED and str(doc.get('reason') or '').startswith('admission')


def killed(returncode):
    return returncode == 'timeout' or (isinstance(returncode, int) and returncode < 0)


def spy_state(out, stage, returncode):
    """line.json 记的子进程 spy 状态（修订 3，N-I2）：passed / failed / 'killed, no spy record' / None（没有跑）。"""
    record = rc.read_json(Path(out)/stage/SPY_FILE)
    if record is None:
        return KILLED_NO_RECORD if killed(returncode) else (None if returncode is None else 'missing')
    return 'passed' if record.get('passed') is True and returncode != 8 else 'failed'


def child_spy_problem(out, stage, returncode):
    """None，或子进程的 spy 不成立的原因（复核 B2）：退出 8、记录不在、passed 不是 true。被信号杀掉或超时的子进程写不出记录：这一种按 D74 走退路，记下（实现员的读法，交总负责人）。"""
    path = Path(out)/stage/SPY_FILE
    record = rc.read_json(path)
    if record is None:
        return None if killed(returncode) else f'{stage}: no {SPY_FILE} (exit {returncode})'
    if returncode == 8 or record.get('passed') is not True:
        return f"{stage}: the spy did not pass (exit {returncode}; violations {record.get('violations')}; breach {record.get('breach')}; forbidden {(record.get('he_spy') or {}).get('forbidden_access')})"
    return None


# ---- 退路（C25.6-12） ------------------------------------------------------------------------------------------------------------------------------
def configc_fallback(ctx, pin, *, launcher, certify, config, suffix='', shared_dir):
    """Fallback one (C22.5-2, D75): (u, cert, record) or (None, None, record)."""
    s = ctx['settings']['configc']
    d = Path(shared_dir)
    d.mkdir(parents=False, exist_ok=False)
    rec = dict(kind='configC', run_dir=str(d/'run'), file=None, final_native_sha256=None, state_hash=None, certificate=None, configC_classification=None, firewall=None, prepare=None, launch=None, candidates=[], error=None,
               rule='D75: classes ok and handover_only; final_native.npz, else the certified native seed; certified again here, in a process that reads no label')
    try:
        rec['firewall'] = fc.derive_pair(pin, d/'derived', pair=ctx['pair'])
        prepared = fc.prepare(d/'derived', d/'run', pair=ctx['pair'], scratch=d, frozen=ctx['paths']['frozen'], opts=tuple(s['opts']), stage_label=s['stage_label'])
        rec['prepare'] = {k: prepared[k] for k in ('slot', 'frozen_files', 'seconds')}
        admission = Path(ctx['out'])/'admission.json'
        he.write_json(d/'run'/'slot.json', dict(stage_dir=ctx['stage_dir'], pair=ctx['pair'], slot=prepared['slot'], slot_is_not_the_stage=True, admission_sha256=core.sha_file(admission) if admission.is_file() else None,
                                                note='protocol.json pair_dataset and pair_source, and prepared_check.json, name the slot of a private copy of the frozen tree (firewall_c.SLOT_STAGE), not this stage; '
                                                     'the admission is load_pair\'s and the sealed lists\' (revision 1, review minor 10)'))
        gate, problems = fc.firewall_check(pin, d/'derived', d/'run', pair=ctx['pair'])
        rec['firewall']['gate'] = gate
        if problems:
            raise RuntimeError(f'the firewall gate: {problems}')
        t0 = fc.uptime()
        tick = perf_counter()
        rec['launch'] = dict(launcher(d/'run', he.short_name(ctx['pair'])), seconds=None)
        rec['launch']['seconds'] = perf_counter()-tick
        dxg, oom = fc.dmesg_counts(t0)
        rec['launch'].update(dxg_lines=dxg, oom_lines=oom, dmesg_read=dxg is not None)
        rec['configC_classification'] = fc.classify(d/'run', dxg=dxg or 0, oom=oom or 0, watchdog=s['watchdog_seconds'])
        for name, path, crec in fc.candidates(d/'run', rec['configC_classification']):
            row = dict(file=name, path=str(path), **{k: crec.get(k) for k in ('recorded_sha256', 'file_sha256', 'role')})
            try:
                u = fc.load_candidate(path, crec, pin.grid)
                problem = he.field_problem(u, pin.grid)         # revision 1, review minor 4: a candidate that cannot be an output is passed over
                if problem:
                    raise ValueError(problem)
                digest = he.array_sha256(u)
                doc = he.certificate_document(certify(u, pin.grid), config, process=f'the HE supervisor (pid {os.getpid()}; reads no label): configuration C field certified again', array_sha256_of_u=digest)
                row.update(state_hash=digest, passed=he.passes(doc) and doc['state_hash'] == digest, status=doc['status'], min_jd_lower_bound=doc['min_jd_lower_bound'])
            except (OSError, ValueError, KeyError) as exc:
                row.update(passed=False, error=f'{type(exc).__name__}: {exc}')
            rec['candidates'].append(row)
            if row['passed']:
                rec.update(file=name, final_native_sha256=crec['file_sha256'], state_hash=digest, certificate=doc)
                he.write_json(d/'fallback.json', rec)
                return u, doc, rec
        rec['error'] = 'no certified state of configuration C' if rec['configC_classification'].get('class') in fc.CANDIDATE_CLASSES else f"class {rec['configC_classification'].get('class')}: no candidate (D75)"
    except Exception as exc:                            # the fallback is unavailable; the evidence stays; the identity follows
        rec['error'] = f'{type(exc).__name__}: {exc}'[:2000]
        rec['traceback'] = traceback.format_exc()[-3000:]
    he.write_json(d/'fallback.json', rec)
    return None, None, rec


class SharedConfigCUnavailable(RuntimeError):
    """共用的配置 C 目录在，却没有 fallback.json（别处在跑或中途死了）：共用步骤的失败，交 BZ。"""


def configc_once(ctx, pin, *, launcher, certify, config, suffix=''):
    """退路一，每对至多跑一次（D135）：目录 <configc_root>/<pair> 不在 → 在这里以 mkdir（O_EXCL 语义）占位并跑（ran_here）；在 → 只读（reused），用之前在本进程（不读标签）再认证一次。"""
    d = Path(ctx['configc_root'])/ctx['pair']
    d.parent.mkdir(parents=True, exist_ok=True)
    try:
        u, doc, rec = configc_fallback(ctx, pin, launcher=launcher, certify=certify, config=config, suffix=suffix, shared_dir=d)
        source = 'ran_here'
    except FileExistsError:
        shared = rc.read_json(d/'fallback.json')
        if shared is None:
            raise SharedConfigCUnavailable(f'{d} is there without fallback.json') from None
        rec = dict(kind='configC', reused_from=str(d), file=shared.get('file'), state_hash=shared.get('state_hash'), error=shared.get('error'), configC_classification=shared.get('configC_classification'))
        u = doc = None
        if shared.get('file') is not None:
            row = next((r for r in shared.get('candidates') or [] if r.get('file') == shared['file'] and r.get('passed')), None)
            try:
                u = fc.load_candidate(row['path'], dict(recorded_sha256=row.get('recorded_sha256'), file_sha256=core.sha_file(row['path'])), pin.grid)
                if core.sha_file(row['path']) != shared.get('final_native_sha256') or he.field_problem(u, pin.grid) or he.array_sha256(u) != shared['state_hash']:
                    raise ValueError('the shared configuration C field is not the one its record names')
                doc = he.certificate_document(certify(u, pin.grid), config, process=f'the R4 supervisor (pid {os.getpid()}; reads no label): the shared configuration C field certified again',
                                              array_sha256_of_u=he.array_sha256(u))
                if not (he.passes(doc) and doc['state_hash'] == shared['state_hash']):
                    rec['error'], u, doc = 'the shared configuration C field did not pass the certificate again here', None, None
                else:
                    rec['certificate'] = doc
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rec['error'], u, doc = f'{type(exc).__name__}: {exc}', None, None
        source = 'reused'
    rec.update(configC_source=source, configC_dir=str(d), configC_result_sha256=core.sha_file(d/'fallback.json') if (d/'fallback.json').is_file() else None)
    return u, doc, rec


def fallback_chain(ctx, pin, *, start, launcher, certify, config, suffix=''):
    """(u, cert, 退路记录, line status)：配置 C（共用，每对至多一次），再恒等（run_hx.identity_fallback，未改）。"""
    before = []
    if start == 'configC':
        u, doc, rec = configc_once(ctx, pin, launcher=launcher, certify=certify, config=config, suffix=suffix)
        if u is not None:
            return u, doc, rec, he.FALLBACK_C
        before.append(rec)
    u, doc, rec = run_hx.identity_fallback(ctx, pin, certify=certify, config=config, suffix=suffix)
    rec['before'] = before
    return u, doc, rec, he.FALLBACK_ID


# ---- supervisor ---------------------------------------------------------------------------------------------------------------------------------
def line_base(ctx, pin=None):
    return dict(run_hx.line_base(ctx, pin), schema=rc.SCHEMA.format('line'), arm=ctx['arm'], role=ctx['role'], r4_spec=ctx['settings']['r4']['spec'], r4_spec_sha256=ctx['settings']['r4']['spec_sha256'])


def t88_status_decision(t88, rc_t88, options=()):
    """he_core.t88_decision，另加 R4 的 implementation_error（掩膜、软通道）：line_status hx_implementation_error，没有输出，队列停下（C25.2-4、C25.3-5）。
    复核 I1：选项开着的臂，T88 子进程没写 run.json（被杀之外）也按实现错误停队列，不静默退路；现行臂（没有选项）照 D74 走退路。"""
    if t88 is not None and t88.get('status') == T88_IMPLEMENTATION:
        return 'implementation', dict(stage='t88', reason=t88.get('reason'), kind=(t88.get('reason') or '').split(':')[0])
    killed = rc_t88 == 'timeout' or (isinstance(rc_t88, int) and rc_t88 < 0)
    if t88 is None and options and not killed:
        return 'implementation', dict(stage='t88', reason=f'the T88 child of an arm with options {list(options)} wrote no run.json (exit {rc_t88})', kind='no_run_json')
    return he.t88_decision(t88, rc_t88)


def run_pair(ctx, *, child=None, launcher=None, certification=None):
    """一对一臂。返回 0（有输出）、STOP_EXIT（hx_provenance_rejected 或 hx_implementation_error：读真值之前的失败，队列停下）、8（代码身份）、2（拒绝启动）。"""
    t0 = perf_counter()
    out = Path(ctx['out'])
    if ctx.get('check_identity') is not True and ctx.get('test_context') != run_hx.TEST_CONTEXT:
        return 2
    out.mkdir(parents=True, exist_ok=False)
    he.write_json(context_path(ctx), ctx)
    he.write_json(out/'started.json', dict(utc=rc.utc(), argv=sys.argv, pid=os.getpid(), child=getattr(child, 'kind', None)))
    _, problems = run_hx.check_identity(ctx, out, 'identity.json')
    if problems:
        return 8
    core.frozen_api()
    child = SubprocessChild() if child is None else child
    try:
        pin, admission = admit(ctx)
    except refusals() as exc:
        he.write_json(out/'line.json', dict(line_base(ctx), line_status=he.REJECTED, rejection=dict(stage='admission', reason=f'{type(exc).__name__}: {exc}'[:2000]), output=None, stop_queue=True,
                                            note='no output; a refusal before any truth (C25.7-3: the shared step), the queue stops', seconds_total=perf_counter()-t0, ended_utc=rc.utc()))
        return STOP_EXIT
    he.write_json(out/'admission.json', dict(keys_read=list(pin.provenance['keys_read']), label_firewall=pin.provenance['label_firewall'], subject_guard=pin.provenance['subject_guard'], admission=admission,
                                            pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], grid=pin.grid_metadata))
    rc_t88 = child('t88', ctx)
    t88 = rc.read_json(out/'t88/run.json')
    spy_problem = child_spy_problem(out, 't88', rc_t88)
    kind, info = ('implementation', dict(stage='t88', kind='spy', reason=spy_problem)) if spy_problem else t88_status_decision(t88, rc_t88, ctx['settings']['r4']['options'])
    repair_record = rc_repair = None
    if kind == 'go':
        rc_repair = child('repair', ctx)
        repair_record = rc.read_json(out/'repair/repair.json')
        spy_problem = child_spy_problem(out, 'repair', rc_repair)
        if spy_problem:
            kind, info = 'implementation', dict(stage='repair', kind='spy', reason=spy_problem)
        else:
            kind, info = he.repair_decision(repair_record, run_hx.certified_consistency(out/'repair', repair_record, pin.grid), rc_repair)
    if kind in ('rejected', 'implementation'):
        status = he.REJECTED if kind == 'rejected' else he.IMPLEMENTATION
        he.write_json(out/'line.json', dict(line_base(ctx, pin), line_status=status, rejection=info if kind == 'rejected' else None, implementation_error=info if kind == 'implementation' else None,
                                            output=None, t88=dict(status=None if t88 is None else t88.get('status'), reason=None if t88 is None else t88.get('reason'), returncode=rc_t88), stop_queue=True,
                                            repair=None if rc_repair is None else dict(status=None if repair_record is None else repair_record.get('status'), returncode=rc_repair),
                                            spy=dict(t88=spy_state(out, 't88', rc_t88), repair=spy_state(out, 'repair', rc_repair)),
                                            note='no output; a failure before any truth (C25.7-3): the queue stops, to the lead and BZ', seconds_total=perf_counter()-t0, ended_utc=rc.utc()))
        return STOP_EXIT
    try:
        certify, config = run_hx.certifier(ctx) if certification is None else certification
        launcher = run_hx.make_launcher(ctx) if launcher is None else launcher
        if kind == 'hx':
            with np.load(out/'repair/certified.npz', allow_pickle=False) as z:
                u = np.ascontiguousarray(np.array(z['u']))
            cert, fallback, status, failure_record = rc.read_json(out/'repair/cert.json'), None, he.HX, None
        else:
            failure_record = info
            u, cert, fallback, status = fallback_chain(ctx, pin, start='configC', launcher=launcher, certify=certify, config=config)
        line = dict(line_base(ctx, pin), line_status=status, failure=failure_record, fuse=he.fuse_record(repair_record, status), fallback=fallback, rejection=None,
                    spy=dict(t88=spy_state(out, 't88', rc_t88), repair=spy_state(out, 'repair', rc_repair), rule='修订 3 N-I2: a child killed by a signal or the timeout writes no spy record; D74 decides'),
                    configC_source=(fallback or {}).get('configC_source') or next((b.get('configC_source') for b in (fallback or {}).get('before') or []), None),
                    t88=dict(status=None if t88 is None else t88.get('status'), reason=None if t88 is None else t88.get('reason'), returncode=rc_t88, seconds_task=None if t88 is None else t88.get('seconds_task')),
                    repair=(None if rc_repair is None else dict(returncode=rc_repair, note='no repair.json')) if repair_record is None else
                    dict(status=repair_record.get('status'), reason=repair_record.get('reason'), passed=repair_record.get('passed'), wall_hit=repair_record.get('wall_hit'),
                         rounds=len(repair_record.get('rounds') or []), seconds_repair=repair_record.get('seconds_repair'), returncode=rc_repair),     # 修订 4 N3-I1
                    decision_certificate=dict(process=cert.get('process'), state_hash=cert.get('state_hash'), passed=he.passes(cert)), seconds_total=perf_counter()-t0)
        _, problems = run_hx.check_identity(ctx, out, 'identity_end.json')
        if problems:
            he.write_json(out/'line_not_written.json', dict(line, reason=f'the frozen tree or the code changed during the run: {problems}'))
            return 8
        he.write_output(out/he.OUTPUT_DIR, u, pin.grid, cert, line)
    except Exception as exc:                                            # noqa: BLE001 - C22.5-6: no output; line.json says why; the queue stops
        he.write_json(out/'line.json', dict(run_hx.implementation_error_line(ctx, pin, exc, unit=None, t88=t88, rc_t88=rc_t88, repair_record=repair_record, rc_repair=rc_repair,
                                                                            failure_record=info if kind != 'hx' else None, c_line=None, seconds_total=perf_counter()-t0), schema=rc.SCHEMA.format('line'), arm=ctx['arm'], role=ctx['role']))
        return STOP_EXIT
    return 0


# ---- 线记录 ---------------------------------------------------------------------------------------------------------------------------------------
def line_path(out):
    """一对的 line.json：有输出时在 <out>/output/line.json（he_core.write_output 写），没有输出时在 <out>/line.json（拒绝、实现错误）。两处都在：读 output 的（不会发生）。"""
    out = Path(out)
    return out/he.OUTPUT_DIR/'line.json' if (out/he.OUTPUT_DIR/'line.json').is_file() else out/'line.json'


def line_record(out):
    return rc.read_json(line_path(out))


# ---- 定稿单元（C25.6-10、C25.9-3） ---------------------------------------------------------------------------------------------------------------
def final_dir(root, role, arm, pair):
    return Path(root)/('final' if role == 'bench' else 'dev_final')/arm/pair


def finalize_pair(ctx, *, certification=None, launcher=None):
    """一对：输出数组再认证一次（不读标签的进程）；与判定证书不一致 → 第三次认证；第三次不过 → 下一个退路（输出改名保留）；没有下一个退路 → 输出撤回（没有输出）。写 final.json。"""
    out = Path(ctx['out'])
    line0 = line_record(out)
    target = final_dir(ctx['root'], ctx['role'], ctx['arm'], ctx['pair'])/'final.json'
    base = dict(schema=rc.SCHEMA.format('final'), role=ctx['role'], arm=ctx['arm'], pair=ctx['pair'], run_dir=str(out), line_json_sha256=core.sha_file(line_path(out)) if line_path(out).is_file() else None,
                r4_spec_sha256=ctx['settings']['r4']['spec_sha256'], utc=rc.utc())
    if line0 is None or line0.get('line_status') not in rc.OUTPUT_STATUSES:
        return rc.write_json(target, dict(base, line_status=(line0 or {}).get('line_status'), output=None, note='no output (no line.json, or hx_provenance_rejected / hx_implementation_error)'))
    pin, _ = admit(ctx)
    certify, config = run_hx.certifier(ctx) if certification is None else certification
    o = out/he.OUTPUT_DIR
    u, line, cert = he.read_output(o, pin.grid)
    thirds, same = [], None
    for round_ in range(3):                                                                  # 复核次要 5：退路之后的新输出照样和它的判定证书比
        digest = he.array_sha256(u)
        review = he.certificate_document(certify(u, pin.grid), config, process=f'the R4 finalize unit (pid {os.getpid()}), round {round_}, before any label was read', array_sha256_of_u=digest)
        same, diffs = he.same_certificate(review, cert)
        if same:
            break
        third_doc = he.certificate_document(certify(u, pin.grid), config, process=f'the third certification (R4 finalize unit, pid {os.getpid()}; reads no label)', array_sha256_of_u=digest)
        third = dict(certificate=third_doc, passed=he.passes(third_doc) and third_doc['state_hash'] == digest, rule='C22.3-9 (d)', line_status=line['line_status'])
        thirds.append(third)
        rc.write_json(out/f'finalize_cert_mismatch_{len(list(out.glob("finalize_cert_mismatch_*.json")))}.json', dict(review=review, decision=cert, differences=diffs, third=third))
        if third['passed']:
            break
        start = he.next_fallback(line['line_status'])
        k = len(list(out.glob('output_superseded_*')))
        o.rename(out/f'output_superseded_{k}')
        if start is None:                                                                    # C22.5-6 (a)：恒等的第三次认证也不过：实现错误，没有输出，停队列
            return rc.write_json(target, dict(base, line_status=rc.IMPLEMENTATION, output=None, withdrawn=str(out/f'output_superseded_{k}'), third_certificates=thirds,
                                              implementation_error=dict(kind='a', rule='C22.5-6 (a): the third certificate of the identity output failed'),
                                              note='no output; the finalize unit stops the queue (exit 9), to the lead and BZ'))
        launcher = run_hx.make_launcher(ctx) if launcher is None else launcher
        try:
            u, cert, fallback, status = fallback_chain(ctx, pin, start=start, launcher=launcher, certify=certify, config=config, suffix=f'_after_third_{k}')
        except run_hx.IdentityNotCertified as exc:
            return rc.write_json(target, dict(base, line_status=rc.IMPLEMENTATION, output=None, withdrawn=str(out/f'output_superseded_{k}'), third_certificates=thirds,
                                              implementation_error=dict(kind='a', rule='C22.5-6 (a)', exception=str(exc)), note='no output; the queue stops'))
        line = dict({k_: v for k_, v in line.items() if k_ not in ('output', 'ended_utc')}, line_status=status, fallback=fallback, third_certificate=third,
                    superseded=list(line.get('superseded') or [])+[str(out/f'output_superseded_{k}')], configC_source=(fallback or {}).get('configC_source'))
        he.write_output(o, u, pin.grid, cert, line)
        u, line, cert = he.read_output(o, pin.grid)
    else:
        return rc.write_json(target, dict(base, line_status=rc.IMPLEMENTATION, output=None, third_certificates=thirds, note='the review certificate still differs after three rounds: the queue stops'))
    third = thirds[-1] if thirds else None
    fallback = line.get('fallback') or {}
    repair = rc.read_json(out/'repair/repair.json')
    return rc.write_json(target, dict(base, line_status=line['line_status'], output=dict(directory=str(o), final_npz_sha256=core.sha_file(o/'final.npz'), disp_npz_sha256=core.sha_file(o/'disp.npz'),
                                                                                         state_hash=digest, line_json_sha256=core.sha_file(o/'line.json')),
                                      decision_certificate=dict(state_hash=cert.get('state_hash'), passed=he.passes(cert), process=cert.get('process')), review_certificate=review, review_consistent=same,
                                      third_certificate=third, fallback=dict(kind=fallback.get('kind'), configC_source=line.get('configC_source'), configC_result_sha256=fallback.get('configC_result_sha256')),
                                      repair_json_sha256=core.sha_file(out/'repair/repair.json') if (out/'repair/repair.json').is_file() else None,
                                      repair_rounds=None if repair is None else len(repair.get('rounds') or []), task_clock=(line.get('t88') or {}).get('seconds_task'),
                                      seconds_repair=(line.get('repair') or {}).get('seconds_repair')))


def read_final(root, role, arm, pair):
    return rc.read_json(final_dir(root, role, arm, pair)/'final.json')


def output_array(final):
    """(u, 问题)：定稿记录指向的输出数组，sha256 核过（final.npz 的文件 sha256 与 state_hash）。"""
    o = Path((final.get('output') or {}).get('directory') or '')
    try:
        if core.sha_file(o/'final.npz') != final['output']['final_npz_sha256']:
            return None, 'final.npz 的 sha256 与定稿记录不符'
        with np.load(o/'final.npz', allow_pickle=False) as z:
            u = np.ascontiguousarray(np.array(z['u']))
        if he.array_sha256(u) != final['output']['state_hash']:
            return None, '输出数组的散列与定稿记录不符'
        return u, None
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return None, f'{type(exc).__name__}: {exc}'


def sdlogj_of(u, spacing):
    """官方 SDlogJ（scripts/evaluate_l2r_style.py:64 sdlogj_l2r，:30 disp_voxels）：只用位移场。"""
    l2r = core.load_module('_r4_l2r_style', 'scripts/evaluate_l2r_style.py')
    return l2r.sdlogj_l2r(l2r.disp_voxels(u, spacing))[0]
