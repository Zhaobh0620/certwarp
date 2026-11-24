"""R4 修订 7（prepare-bench 在真实运行里撞上错的导入，总负责人 11-14 约 1:50 AM CST）的测试：
1. 不打桩地真跑台的数据准备：合成 NIfTI（台的 16 人与开发集参照的 id，只有 id 是真的；图像在 tmp 里），真实的 header_record、load_scalar、require_common_grid、
   write_bench_stage / write_bench_pair、load_bench_pair 与 prepare_bench 的 spy；
2. experiments/r4 的每一条导入、模块别名上的属性与调用签名，在 PYTHONPATH=<冻结树>:<代码> 下逐个解析（_import_check.py）；
3. 其余真实路径上能用合成数据真跑的（见实现说明第 14 节的逐条结论）。不读真值、不用 GPU、不起单元、不碰 /home/rmcfz/seq_vp_tag_results（只读冻结树）。"""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in ('r4', 'he', 'mc0', 'ef', 'fl', 'd0', 'unv2'):
    p = str(ROOT/'experiments'/_p)
    if p not in sys.path:
        sys.path.insert(0, p)

import r4_bench as rb  # noqa: E402
import r4_common as rc  # noqa: E402
import r4_steps  # noqa: E402

PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
SHAPE_XYZ = (12, 14, 10)


@pytest.fixture(scope='module')
def bench():
    return rb.load_bench()


def tissue_image(shape_xyz, seed):
    """三层同心壳（WM .8、GM .5、CSF .25，背景 0，加噪声）：三类体积相当，Atropos 的 QC（每类 ≥ 5%）能过。"""
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(*[np.arange(n, dtype=float) for n in shape_xyz], indexing='ij'))
    c = (np.asarray(shape_xyz)-1)/2
    r = np.sqrt(np.sum(((g-c[:, None, None, None])/(np.asarray(shape_xyz)[:, None, None, None]*.45))**2, axis=0))
    image = np.where(r <= .55, .8, np.where(r <= .8, .5, np.where(r <= 1., .25, 0.)))
    return np.where(image > 0, image+rng.normal(0, .02, shape_xyz), 0.).astype(np.float32)


@pytest.fixture()
def data(tmp_path, bench, monkeypatch):
    """tmp/data/<id>/{aligned_norm,aligned_seg35,aligned_seg4}.nii.gz（bayesreg.seg.unet.synthetic 的头文件：mm、qform 与 sform 都编码；aligned_norm 换成三层壳）；
    rc.DATA_ROOT 指过来，spy 照常判它。"""
    import nibabel as nib
    from bayesreg.seg.unet.synthetic import write_subjects
    root = tmp_path/'data'
    subjects = list(bench.ordered_subjects())+[r4_steps.DEV_REFERENCE_SUBJECT]
    write_subjects(root, subjects, SHAPE_XYZ)
    for subject in subjects:
        path = root/subject/'aligned_norm.nii.gz'
        old = nib.load(str(path))
        nib.save(nib.Nifti1Image(tissue_image(SHAPE_XYZ, int(subject.split('_')[2])), old.affine, old.header), str(path))
    monkeypatch.setattr(rc, 'DATA_ROOT', root)
    return root


@pytest.fixture()
def prepared(tmp_path, bench, data, monkeypatch):
    """真跑 prepare-bench 的单元主体（代码身份与开发集网格之外不打桩）。"""
    monkeypatch.setattr(r4_steps, '_identity', lambda root, role: ({}, {}))                            # 代码身份另有测试（batch_identity）
    reference = data/r4_steps.DEV_REFERENCE_SUBJECT/'aligned_norm.nii.gz'
    monkeypatch.setattr(r4_steps, 'dev_grid_meta', lambda: grid_of(reference))                         # 开发集 manifest 的网格（worktree 里没有 data 链接）
    root = tmp_path/'r4'
    assert r4_steps.prepare_bench(SimpleNamespace(dry_run=False), root=root) == 0
    return root


def grid_of(path):
    """开发集网格元数据的独立算法：load_scalar 之后照 manifest 的字段写（不经 write_bench_pair）。"""
    from jd_curl.three_d.nifti import load_scalar
    grid = load_scalar(path).grid
    return dict(shape_zyx=list(grid.shape), spacing_xyz=list(grid.spacing), origin_xyz=[float(a[0]) for a in grid.axes])


# ---- 第 1、2 条：prepare-bench 真跑 -------------------------------------------------------------------------------------------------------------------
def test_prepare_bench_writes_the_stage_with_the_real_jd_curl_functions(prepared, bench, data):
    from bayesreg.io.pair import load_pair
    from jd_curl.three_d.nifti import load_scalar
    root = prepared
    reference = data/r4_steps.DEV_REFERENCE_SUBJECT/'aligned_norm.nii.gz'
    stage = root/'bench_stage'
    record = rc.read_json(root/'bench_stage_record'/'opened_files.json')
    assert record['passed'] is True and record['truth_opened'] == [] and record['violations'] == []
    assert any(r['path'].endswith('aligned_norm.nii.gz') and r['judged'] for r in record['opened'])   # 图像的读取是判过的（rc.DATA_ROOT 是判定的根）
    sums = dict(line.split('  ')[::-1] for line in (stage/'SHA256SUMS').read_text(encoding='utf-8').splitlines())
    assert len(sums) == 1+2*len(bench.pairs) and all(rc.sha256_file(stage/n) == h for n, h in sums.items())
    want_grid = grid_of(reference)
    for name, fold, index, fixed, moving in bench.pairs:
        with np.load(stage/name/'pair.npz', allow_pickle=False) as z:
            assert sorted(z.files) == ['fixed', 'initial_u', 'moving']
            arrays = {k: np.array(z[k]) for k in z.files}
        assert np.array_equal(arrays['fixed'], load_scalar(data/fixed/'aligned_norm.nii.gz').values)
        assert np.array_equal(arrays['moving'], load_scalar(data/moving/'aligned_norm.nii.gz').values)
        assert arrays['initial_u'].shape == tuple(want_grid['shape_zyx'])+(3,) and not arrays['initial_u'].any()
        manifest = rc.read_json(stage/name/'manifest.json')
        assert {k: manifest[k] for k in want_grid} == want_grid and manifest['bench'] == dict(fold=fold, index=index)
        assert (manifest['fixed_subject'], manifest['moving_subject']) == (fixed, moving) and manifest['pair_sha256'] == rc.sha256_file(stage/name/'pair.npz')
        assert not any('seg' in k for k in manifest)
        pin, admission = rb.load_bench_pair(bench, stage, name)                                       # 配准一侧读台的 pair.npz 的唯一方式（真实的 load_pair）
        assert sorted(pin.provenance['keys_read']) == ['fixed', 'initial_u', 'moving'] and pin.grid.shape == tuple(want_grid['shape_zyx']) and admission['pair'] == name
    assert load_pair(stage/bench.pairs[0][0]/'pair.npz', allowlist=bench.world.allowlist_path).fixed_subject == bench.pairs[0][3]


def test_write_bench_pair_refuses_another_grid_or_header(tmp_path, bench, data):
    from bayesreg.io.unseal import bind_files  # noqa: F401  (the sealed guard is the real one: bench subjects are not sealed)
    from bayesreg.seg.unet.data import OasisFiles
    files = OasisFiles(data)
    name, _, _, fixed, moving = bench.pairs[0]
    header = rb.header_record(data/fixed/'aligned_norm.nii.gz')
    grid = grid_of(data/fixed/'aligned_norm.nii.gz')
    with pytest.raises(rb.BenchRefused, match='网格'):
        rb.write_bench_pair(tmp_path/'a', fixed, moving, files, bench=bench, reference_header=header, dev_grid=dict(grid, shape_zyx=[1, 2, 3]))
    with pytest.raises(rb.BenchRefused, match='头文件'):
        rb.write_bench_pair(tmp_path/'b', fixed, moving, files, bench=bench, reference_header=dict(header, zooms=[1., 1., 2.]), dev_grid=grid)
    manifest = rb.write_bench_pair(tmp_path/'c', fixed, moving, files, bench=bench, reference_header=header, dev_grid=grid)
    assert manifest['pair'] == name


# ---- 第 3 条：每一条导入、模块别名上的属性、调用签名 -------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not FROZEN.is_dir(), reason='the frozen tree is not on this machine')
def test_every_import_attribute_and_call_signature_of_experiments_r4_resolves():
    env = dict(os.environ, PYTHONPATH=f'{FROZEN}:{ROOT}', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run([PY, '-B', str(Path(__file__).with_name('_import_check.py')), str(ROOT)], capture_output=True, text=True, env=env, timeout=600, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-3000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result['problems'] == [] and result['checked'] > 1000, result['problems']


# ---- 第 4 条：atropos-bench 真跑（真实的 ANTs 2.6.5 ImageMath 与 Atropos，合成的三层壳） -----------------------------------------------------------------
@pytest.mark.skipif(not Path('/home/rmcfz/opt/baselines/ants/ants-2.6.5/bin/Atropos').is_file(), reason='ANTs is not on this machine')
def test_atropos_bench_runs_the_real_ants_on_the_prepared_stage(prepared, bench):
    import atropos_seg
    root = prepared
    assert r4_steps.atropos_bench(SimpleNamespace(dry_run=False), root=root) == 0
    record = rc.read_json(root/'atropos_record'/'atropos.json')
    assert record['spy_passed'] is True and sorted(record['subjects']) == sorted(bench.ordered_subjects())
    for subject, row in record['argv'].items():
        assert [Path(c['argv'][0]).name for c in row['calls']] == ['ImageMath', 'Atropos'] and all(c['problems'] == [] for c in row['calls'])
        prov = rc.read_json(root/'atropos'/subject/'provenance.json')
        assert prov['checks']['qc_min_class_fraction_ge_0.05'] is True and prov['class_names'] == ['CSF', 'GM', 'WM'] and prov['segmenter_id'] == 'atropos'
    assert atropos_seg.subprocess is subprocess                                                        # AtroposArgv 已拆下
    assert rc.read_json(root/'atropos_record'/'opened_files.json')['passed'] is True


# ---- 第 4 条：seg-bench 的包装真跑（只把网络前向换成合成概率），写出的预测由 T88 一侧的读取器（BenchLayer）读回 -------------------------------------------
def test_bench_infer_writes_what_the_t88_reader_accepts(prepared, bench, data, tmp_path, monkeypatch):
    if str(ROOT/'experiments'/'hs') not in sys.path:
        sys.path.insert(0, str(ROOT/'experiments'/'hs'))
    import hs_seg
    import bayesreg.seg.unet.registry as registry_module
    from bayesreg.seg.unet.data import OasisFiles, read_image
    pins = dict(registry_sha256=rc.sha256_file(ROOT/rc.BENCH_REGISTRY_REL), contact_sha256=rc.sha256_file(ROOT/rc.BENCH_CONTACT_REL))
    registry, contact, record = rb.load_bench_registry(bench, pins)
    models_root = tmp_path/'models'
    available = {}
    for f in rc.FOLDS:
        for k, sha in enumerate(registry.models[rc.BENCH_MODELS[f]]['member_model_sha256']):
            d = models_root/f'fold{f}_s{k}'
            d.mkdir(parents=True)
            (d/'member.json').write_text(json.dumps(dict(member_sha256=sha, temperature=1.1)))
            available[sha] = d/'member.pt'
    monkeypatch.setattr(registry_module, 'find_member_files', lambda root: available)                 # 真的成员文件不读（2 GB；网络前向另由 HS 在真实数据上走过）
    forward = []

    def uf4(subject, *, files, registry, models, plan, device):
        values, meta = read_image(files, subject)                                                      # 真实的读图与元数据
        rng = np.random.default_rng(int(subject.split('_')[2]))
        probs = rng.uniform(0, .1, (36,)+values.shape)
        probs[1] += values > .6
        probs[2] += (values > 0) & (values <= .6)
        probs /= probs.sum(axis=0)
        forward.append((subject, [p[0] for p in plan], device))
        fold = rc.FOLDS.index(bench.fold_of(subject))
        members = [dict(model_id=m, fold=fold, seed=k, member_sha256=sha, temperature=1.1) for k, (m, sha, path) in enumerate(plan)]
        return probs.astype(np.float32), meta, dict(percentile=99.5), members
    monkeypatch.setattr(hs_seg, 'uf4_probabilities', uf4)
    pred = tmp_path/'pred'
    rows = rb.bench_infer(bench, registry, record, contact, pred_root=pred, files=OasisFiles(data), models_root=models_root, device='cpu', code={}, command=['test'], log=lambda m: None)
    assert sorted(rows) == sorted(bench.ordered_subjects())
    assert all(model == rc.BENCH_MODELS[bench.fold_of(s)] for s, models, _ in forward for model in models) and len(forward) == 16
    for name, _, _, fixed, moving in bench.pairs:
        pin, _ = rb.load_bench_pair(bench, prepared/'bench_stage', name)
        layer = rb.BenchLayer(pin, bench, registry, contact, record, pred_root=pred, models_root=models_root)
        for side in ('fixed', 'moving'):
            array, accepted, rec = layer.load(side)                                                    # 台的读取器：来源、登记、本折模型、温度、网格、文件 sha256
            assert array.shape[1:] == pin.grid.shape and accepted['test_labels_seen'] == rc.BENCH_TEST_LABELS_SEEN and rec['models']['policy'] == rc.BENCH_POLICY


# ---- 第 4 条：devcheck 的数组比较 -------------------------------------------------------------------------------------------------------------------------
def test_devcheck_compares_the_arrays_not_the_file_bytes(tmp_path):
    u = np.random.default_rng(0).normal(size=(4, 5, 6, 3))
    np.savez(tmp_path/'hs.npz', u=u)
    np.savez(tmp_path/'mine.npz', u=u)
    row = dict(status='done', field=dict(u_sha256=rc.array_sha256(u), sha256=rc.sha256_file(tmp_path/'mine.npz')))
    import he_core
    assert rc.array_sha256(u) == he_core.array_sha256(u)                                              # 修订 8 次要 3：R4 与 HE 的数组散列是同一个算法
    result = r4_steps.devcheck_compare(row, tmp_path, tmp_path/'hs.npz')
    assert result['same_array'] is True and result['mine_u_sha256'] == result['hs_u_sha256']
    assert r4_steps.devcheck_compare(dict(row, field=dict(u_sha256='0'*64)), tmp_path, tmp_path/'hs.npz')['same_array'] is False


# ---- 第 4 条：一对一臂真跑（子进程的命令行 run_r4.py t88/repair --context、真实的修复与认证、定稿、评测一侧的指标、冒烟的计数） --------------------------------
REAL_QUEUE = r'''
import json, sys
from pathlib import Path
import numpy as np
ROOT, TMP = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(ROOT/'experiments'/d) for d in ('r4', 'he', 'mc0', 'ef', 'fl', 'd0')]
import mc0_core as core
core.frozen_api()
exec(compile(Path(sys.argv[3]).read_text(), 'world', 'exec'))
import shutil
import r4_eval as re_
import r4_spy as spy
import r4_steps
real_roots = spy.judged_roots
spy.judged_roots = lambda repo=rc.ROOT, resume_repo=None: real_roots(repo, resume_repo)+[spy.resolved(TMP)]   # 修订 8 重要 2：tmp 由真实的白名单判（本进程的单元；子进程照真实的根）
child = ch.SubprocessChild()                                                                   # 真实的命令行：python -B run_r4.py t88|repair --context <file>
code = run_r4.supervise(ctx, run=lambda c: ch.run_pair(c, child=child))
out = Path(ctx['out'])
line = ch.line_record(out)
finalize_code = run_r4.finalize_arm(root, 'bench', ctx['arm'], [pair])                         # 定稿单元的主体：每对在自己的 spy 之下，真实的复核认证（CPU FP64，配置 C）
final = ch.read_final(root, 'bench', ctx['arm'], pair)
ctx_b = ch.make_context('bench', 'B', pair, root=root, stage_dir=TMP/'stage', paths=paths, device='cpu', check_identity=False, test=True, bench_pins=pins)
ctx_b['settings']['r4']['spec']['native'] = 3
ctx_b['settings_sha256'] = ch.he.settings_sha256(ctx_b['settings'])
code_b = run_r4.supervise(ctx_b, run=lambda c: ch.run_pair(c, child=ch.InProcessChild()))     # B（软通道）：子进程那一层在本进程里，它们的 spy 也判 tmp
finalize_b = run_r4.finalize_arm(root, 'bench', 'B', [pair])
dev_out = root/'dev_runs'/ctx['arm']/pair
shutil.copytree(Path(final['output']['directory']), dev_out/'output')                         # 开发集形状的定稿：sdlogj_arm 的白名单是 dev_final 与 dev_runs
rc.write_json(root/'dev_final'/ctx['arm']/pair/'final.json', dict(final, role='dev', run_dir=str(dev_out), output=dict(final['output'], directory=str(dev_out/'output'))))
sdlogj_code = run_r4.sdlogj_arm(root, ctx['arm'], [pair])

def records(*paths):
    rows = {}
    for path in paths:
        doc = rc.read_json(path)
        docs = [doc] if 'pairs' not in doc else list(doc['pairs'].values())+[doc]
        rows[str(path.relative_to(root))] = dict(passed=doc.get('passed'), violations=sum((d.get('violations') or [] for d in docs), []),
                                                 judged_tmp=sum(1 for d in docs for r in d.get('opened') or [] if r['judged'] and r['resolved'].startswith(str(TMP.resolve()))))
    return rows
unit_records = records(out/'opened_files_supervisor.json', root/'final'/ctx['arm']/'opened_files_finalize.json', Path(ctx_b['out'])/'opened_files_supervisor.json',
                       Path(ctx_b['out'])/'t88'/'opened_files.json', Path(ctx_b['out'])/'repair'/'opened_files.json', root/'final'/'B'/'opened_files_finalize.json',
                       root/'dev_final'/ctx['arm']/'opened_files_sdlogj.json')
pin, _ = ch.admit(ctx)
spacing = (1., 1., 1.)
truth = {}
for subject, m in maps.items():
    truth[(subject, 'seg35')] = (m.astype(np.uint8), {})
    truth[(subject, 'seg4')] = (np.where(m > 0, 1+m % 4, 0).astype(np.uint8), {})
row = re_.evaluate_row(final, re_.labs_for(truth, fixed, moving), spacing, pin.grid)
control = re_.independent_control(final, truth, fixed, moving, spacing)
u_out, output_problem = ch.output_array(final)
sdlogj = ch.sdlogj_of(u_out, spacing)                                                          # 开发集可行性一步用的官方 SDlogJ（不读标签）
rebuild = re_.ctl_rebuild(root, pair, final, stage_dir=TMP/'stage', bench=bench) if final.get('line_status') == rc.HX else 'not hx'
smoke_ctx = ch.make_context('bench', 'CTL', pair, root=root, stage_dir=TMP/'stage', out=TMP/'smoke'/'CTL'/pair, paths=paths, device='cpu', check_identity=False, test=True, bench_pins=pins)
smoke_ctx['settings']['r4']['spec']['native'] = 3
smoke_ctx['settings_sha256'] = ch.he.settings_sha256(smoke_ctx['settings'])
smoke_row, counts, smoke_problem = r4_steps.smoke_arm(smoke_ctx, cells=True)
print(json.dumps(dict(code=code, finalize_code=finalize_code, code_b=code_b, finalize_b=finalize_b, line_b=ch.line_record(Path(ctx_b['out']))['line_status'], sdlogj_code=sdlogj_code,
                      unit_records=unit_records, line_status=line['line_status'], t88=line.get('t88'), repair=line.get('repair'), spy=line.get('spy'),
                      final_status=final.get('line_status'), spy_problems=re_.spy_record_problems(out, final), reader_problems=re_.reader_record_problems(out, 'bench', record),
                      metrics=row.get('metrics'), pre_repair=row.get('pre_repair'), control=control, rebuild=rebuild, output_problem=output_problem, sdlogj=sdlogj,
                      smoke_status=(smoke_row or {}).get('status'), counts=counts, smoke_problem=smoke_problem,
                      logs=sorted(p.name for p in out.glob('*.log'))), default=str))
'''


@pytest.mark.skipif(not FROZEN.is_dir(), reason='the frozen tree is not on this machine')
def test_one_pair_one_arm_runs_for_real_through_the_child_command_line():
    import shutil
    import tempfile
    from .test_r4_review import E2E
    tmp_path = Path(tempfile.mkdtemp(prefix='r4_rev7_queue_'))                                          # 不用 pytest 的 tmp：SubprocessChild 的启动串里不得有测试运行器的名字（firewall_c.check_launch）
    assert 'pytest' not in str(tmp_path)
    try:
        run_real_queue(tmp_path, E2E)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def run_real_queue(tmp_path, E2E):
    split, header = "ch.repair_main = fake_repair", "ROOT, TMP, MODE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]"
    assert E2E.count(split) == 1 and E2E.count(header) == 1                                           # 修订 8 次要 1：被切开、被替换的两段都在
    world = E2E.split(split)[0].replace(header, "MODE = 'world'")
    (tmp_path/'world.py').write_text(world)
    env = dict(os.environ, PYTHONPATH=f'{FROZEN}:{ROOT}', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run([PY, '-B', '-c', REAL_QUEUE, str(ROOT), str(tmp_path), str(tmp_path/'world.py')], capture_output=True, text=True, env=env, timeout=1800, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-4000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out['code'] == 0 and out['line_status'] == rc.HX and out['final_status'] == rc.HX, out
    assert out['finalize_code'] == 0 and out['code_b'] == 0 and out['line_b'] in rc.OUTPUT_STATUSES and out['finalize_b'] == 0 and out['sdlogj_code'] in (0, 9)   # 9：一对不是 19/19，可行性不过
    for name, rec in out['unit_records'].items():                                                     # 修订 8 重要 2：supervisor、定稿、B 的两个子进程层、SDlogJ，在真实的白名单下判 tmp
        assert rec['passed'] is True and rec['violations'] == [] and rec['judged_tmp'] > 0, (name, rec)
    assert out['t88']['status'] == 'done' and out['repair']['status'] == 'certified' and out['repair']['returncode'] == 0 and out['t88']['returncode'] == 0
    assert out['spy']['t88'] == 'passed' and out['spy']['repair'] == 'passed' and out['spy_problems'] == [] and out['reader_problems'] == []
    assert out['logs'] == ['repair.log', 't88.log']                                                   # 两个子进程都经 SubprocessChild 起
    assert out['rebuild'] is None                                                                      # CTL 的输出能照 HS 的写法重建（C25.6-11）
    assert out['output_problem'] is None and 0 <= out['sdlogj'] < 1
    for k in ('seg35', 'seg4'):
        assert 0 < out['metrics'][k] <= 1 and abs(out['metrics'][k]-out['control'][k]) < 2e-4 and 0 < out['pre_repair'][k] <= 1
    counts = out['counts']                                                                             # locate_cells 与 certify_map（配置 C）在本单元的 spy 之下真跑；只描述
    assert out['smoke_status'] == 'done' and out['smoke_problem'] is None and counts['status'] in ('certified_feasible', 'infeasible', 'uncertain')     # jd_curl/three_d/feasibility.py:490
    assert all(isinstance(counts[k], int) and counts[k] >= 0 for k in ('n_located_for_interval_fallback', 'n_infeasible', 'n_uncertain'))


# ---- 第 4 条：decide-bench 与 next-dev 的单元主体（真实的 16 对；第一读的行是写好的） --------------------------------------------------------------------------
def test_decide_bench_and_next_dev_run_on_the_real_bench(tmp_path, bench, capsys):
    root = tmp_path/'r4'
    ctl = {p: dict(line_status=rc.HX, metrics=dict(seg35=.8, seg4=.9, sdlogj_official=.07)) for p in bench.pair_names}
    rows = {rc.CONTROL: ctl, 'B': {p: dict(r, metrics=dict(r['metrics'], seg35=.804)) for p, r in ctl.items()}}
    rc.write_json(root/'eval'/'bench_eval_1.json', dict(rows=rows, excluded={}, undecidable_before_truth={}))
    assert r4_steps.decide_bench(SimpleNamespace(), root=root) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    decision = rc.read_json(root/'analysis'/'bench_decision.json')
    assert printed['order'] == decision['order'] == ['B'] and printed['combo'] is None
    assert r4_steps.next_dev(SimpleNamespace(), root=root) == 0
    step = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert step['order'] == ['B'] and step['next'] == dict(action='feasibility', item='B', read_number=1)
