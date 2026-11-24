"""R4 修订 8（修订 7 的复核：阻断 1、重要 2、次要 3；总负责人 11-14 约 2:30 AM CST）的测试。
- 阻断：评测单元的底层 spy 用 R4 的规则重算（读过的评测记录要落在 record_dirs 之内），不再直接用 HE 的 passed；eval-dev 的单元主体在合成的开发集与 HS 树（含 repair_metrics.json）上真跑。
- 重要 2：把 tmp 列进判定根，用真实的 spy 规则判 eval-bench、eval-dev 的单元主体（supervisor、finalize_arm、sdlogj_arm 与子进程在 test_r4_rev7 的一对一臂真跑里）。
- 重要 1：导入核对的变异测试（把 r4_eval 里的 core.frozen_api 改名，核对必须报出来）。
合成数据；只有受试者 id 是真的。不读真值、不用 GPU、不起单元、不碰 /home/rmcfz/seq_vp_tag_results（只读冻结树）。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from .test_r4_rev7 import FROZEN, PY, ROOT, SHAPE_XYZ, grid_of, tissue_image  # noqa: F401
from .test_r4_rev7 import bench  # noqa: F401  (the module-scoped fixture)

for _p in ('r4', 'he', 'mc0', 'ef', 'fl', 'd0'):
    p = str(ROOT/'experiments'/_p)
    if p not in sys.path:
        sys.path.insert(0, p)

import r4_bench as rb  # noqa: E402
import r4_chain as ch  # noqa: E402
import r4_common as rc  # noqa: E402
import r4_decide as rd  # noqa: E402
import r4_eval as re_  # noqa: E402
import r4_spy as spy  # noqa: E402
import r4_steps  # noqa: E402
import run_r4  # noqa: E402

from tests.test_bayesreg.io_seg._helpers import source_sha  # noqa: E402  (absolute: the subprocess imports this module as tests.test_bayesreg.r4.test_r4_rev8)

DEV_PAIRS = [f'OASIS_OAS1_{k:04d}_MR1__OASIS_OAS1_{k+1:04d}_MR1' for k in range(438, 457)]          # 19 对的 id（开发集 0438–0457）；图像与标签都是合成的
DEV_SHAPE = (12, 14, 16)


def judge_tmp_too(monkeypatch, tmp):
    """真实的判定根之外再把 tmp 列进去：tmp 下的读写都由真实的白名单判（重要 2）。"""
    real = spy.judged_roots
    monkeypatch.setattr(spy, 'judged_roots', lambda repo=rc.ROOT, resume_repo=None: real(repo, resume_repo)+[spy.resolved(tmp)])


def identity_final(root, role, arm, pair, shape):
    """一对一臂的恒等退路输出（有输出），带 supervisor、T88 子进程的 spy 记录与定稿记录。"""
    run_dir = Path(root)/('runs' if role == 'bench' else 'dev_runs')/arm/pair
    o = run_dir/'output'
    o.mkdir(parents=True)
    u = np.zeros(tuple(shape)+(3,))
    np.savez(o/'final.npz', u=u)
    rc.write_json(run_dir/'line.json', dict(line_status=rc.FALLBACK_ID))
    rc.write_json(run_dir/'opened_files_supervisor.json', dict(passed=True))
    rc.write_json(run_dir/'t88'/'opened_files.json', dict(passed=True))
    rc.write_json(ch.final_dir(root, role, arm, pair)/'final.json', dict(line_status=rc.FALLBACK_ID, run_dir=str(run_dir), fallback=dict(kind='identity'), role=role, arm=arm, pair=pair,
                                                                          output=dict(directory=str(o), final_npz_sha256=rc.sha256_file(o/'final.npz'), state_hash=ch.he.array_sha256(u))))


# ---- 阻断：评测单元的底层 spy --------------------------------------------------------------------------------------------------------------------------
def test_an_evaluation_unit_reads_evaluation_records_only_inside_its_record_dirs(tmp_path):
    inside, outside = tmp_path/'records'/'repair_metrics.json', tmp_path/'elsewhere'/'repair_metrics.json'
    for path in (inside, outside):
        path.parent.mkdir(parents=True)
        path.write_text('{}', encoding='utf-8')
    rules = spy.dev_rules(tmp_path/'r4', unit='eval-dev-1', extra=(tmp_path/'records',), record_dirs=(tmp_path/'records',))
    rules.roots = [tmp_path.resolve()]
    with spy.UnitSpy(rules, expect_reads=False) as s:
        rc.read_json(inside)
    record = s.record()
    assert record['he_spy']['passed'] is False and record['base_passed'] is True and record['passed'] is True               # HE 的 passed 不看位置；R4 的规则看
    with spy.UnitSpy(rules, expect_reads=False) as s:
        rc.read_json(outside)
    record = s.record()
    assert record['base_passed'] is False and record['passed'] is False and 'outside' in record['base_rule']
    other = spy.dev_rules(tmp_path/'r4', unit='sdlogj', extra=(tmp_path/'records',))                                         # 不是评测单元：照旧用 HE 的 passed
    other.roots = [tmp_path.resolve()]
    with spy.UnitSpy(other, expect_reads=False) as s:
        rc.read_json(inside)
    assert s.record()['passed'] is False


# ---- 次要 2：台一侧的读取记录核对必须给登记记录 ----------------------------------------------------------------------------------------------------------------
def test_the_bench_reader_record_check_needs_the_registry_record(tmp_path):
    rc.write_json(tmp_path/'t88'/'run.json', dict(status='done', oracle=False, data=dict(keys_read=['fixed', 'moving', 'initial_u']), channel_sources=dict(sides={})))
    with pytest.raises(ValueError, match='registry record'):
        re_.reader_record_problems(tmp_path, 'bench')
    assert re_.reader_record_problems(tmp_path, 'bench', dict(sha256='x')) == ['channel_sources.sides: fixed and moving required']
    assert isinstance(re_.reader_record_problems(tmp_path, 'dev'), list)                                  # 开发集一侧不要登记记录


# ---- 阻断与重要 2：eval-dev 的单元主体真跑（子进程：渐变要冻结树的 jd_curl，PYTHONPATH=<冻结树>:<代码>） --------------------------------------------------
def build_dev_world(tmp, assign):
    """合成的开发集 stage（19 对，pair.npz 带 *_seg* 成员）与 HS 树（SHx24 的场、RSHx24 的 repair.json / repair.npz / repair_metrics.json、analysis/decision_hs.json）。
    assign(对象, 名字, 值)：rc.DEV_STAGE_REL、rc.HS_ROOT、rc.HS_DECISION_SHA256、_mc0_evaluate.STAGE 指过来；已提交的 HS 参照与 dev_pairs 换成这棵树的；代码身份打桩；tmp 列进判定根。"""
    stage, hs = tmp/'stage_r4_dev_synthetic', tmp/'hs'
    rng = np.random.default_rng(7)
    labels = np.zeros(DEV_SHAPE, dtype=np.uint8)
    labels[2:-2, 2:-2, 2:8], labels[2:-2, 2:-2, 8:-2] = 2, 3
    for pair in DEV_PAIRS:
        fixed, moving = pair.split('__')
        d = stage/pair
        d.mkdir(parents=True)
        np.savez_compressed(d/'pair.npz', fixed=rng.uniform(0, 1, DEV_SHAPE), moving=rng.uniform(0, 1, DEV_SHAPE), initial_u=np.zeros(DEV_SHAPE+(3,)),
                            fixed_seg35=labels, moving_seg35=np.roll(labels, 1, axis=2), fixed_seg4=(labels > 0).astype(np.uint8), moving_seg4=(np.roll(labels, 1, axis=2) > 0).astype(np.uint8))
        rc.write_json(d/'manifest.json', dict(pair=pair, fixed_subject=fixed, moving_subject=moving, shape_zyx=list(DEV_SHAPE), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm',
                                             sources=dict(fixed=dict(sha256=source_sha(fixed)), moving=dict(sha256=source_sha(moving))), pair_sha256=rc.sha256_file(d/'pair.npz')))
    rc.write_json(stage/'selection.json', dict(pairs=[dict(pair=p) for p in DEV_PAIRS]))
    assign(rc, 'DEV_STAGE_REL', str(stage))                                                             # 绝对路径：rc.ROOT/它 就是它
    assign(rc, 'HS_ROOT', hs)
    assign(re_, 'dev_pairs', lambda stage=None: list(DEV_PAIRS))
    ev = re_.evaluator()
    assign(ev, 'STAGE', stage)
    reference, per_pair = {}, []
    for pair in DEV_PAIRS:
        short = ch.core.short_name(pair)
        pin, _ = re_.run_hx_admit(stage, pair)
        raw = rng.normal(0, .3, DEV_SHAPE+(3,))
        f = hs/'runs'/'SHx24_synthetic'/short
        f.mkdir(parents=True)
        np.savez(f/'field_0600.npz', u=raw)
        tapered, _ = ch.core.frozen_api().boundary_taper(raw, pin.grid, ch.core.TAPER)
        tapered = np.ascontiguousarray(tapered)
        r = hs/'runs'/'RSHx24_synthetic'/short
        r.mkdir(parents=True)
        np.savez(r/'repair.npz', nodes=np.zeros(0, dtype=np.int64), values=np.zeros((0, 3)))
        rc.write_json(r/'repair.json', dict(source=dict(tapered_u_sha256=ch.he.array_sha256(tapered)), attempt=dict(sha256=ch.core.sha_file(r/'repair.npz'))))
        rc.write_json(r/'repair_metrics.json', dict(repaired_u_sha256=ch.he.array_sha256(tapered)))
        reference[short] = dict(repair_json_sha256=rc.sha256_file(r/'repair.json'), repair_metrics_sha256=rc.sha256_file(r/'repair_metrics.json'))
        labs, manifest, _ = ev.truth_labels(pair)
        m = re_.metrics_of(tapered, {k: np.asarray(v).astype(np.int16) for k, v in labs.items()}, tuple(manifest['spacing_xyz']))
        per_pair.append(dict(pair=short, metrics={k: m[k] for k in ('seg35', 'seg4', 'sdlogj_official')}))
    rc.write_json(hs/'analysis'/'decision_hs.json', dict(per_pair=per_pair))
    assign(rc, 'HS_DECISION_SHA256', rc.sha256_file(hs/'analysis'/'decision_hs.json'))
    assign(re_, 'committed_hs_reference', lambda repo=rc.ROOT: reference)                              # 已提交的参照（experiments/p2 下，代码目录）换成这棵树的散列
    assign(run_r4, 'batch_identity', lambda b, repo=rc.ROOT: ([], dict(commit='c', clean=True)))
    real = spy.judged_roots
    assign(spy, 'judged_roots', lambda repo=rc.ROOT, resume_repo=None: real(repo, resume_repo)+[spy.resolved(tmp)])
    return SimpleNamespace(stage=stage, hs=hs, tmp=tmp)


def dev_root(world, name, arm='B'):
    root = world.tmp/name
    rc.write_json(root/'dev_batch.json', dict(role='dev', checkout=str(rc.ROOT.resolve()), code_commit='c', resume_repo=str(world.tmp/'resume')))
    for pair in DEV_PAIRS:
        identity_final(root, 'dev', arm, pair, DEV_SHAPE)
    rc.write_json(root/'dev_final'/arm/'feasibility.json', dict(verdict=dict(verdict=rd.FEASIBLE), spy_passed=True))
    rc.write_json(root/'dev_final'/arm/'opened_files_finalize.json', dict(passed=True))
    rc.write_json(root/'analysis'/'bench_decision.json', dict(order=[arm]))
    return root


DEV_UNIT = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
ROOT, TMP = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(ROOT/'experiments'/d) for d in ('r4', 'he', 'mc0', 'ef', 'fl', 'd0')]
import mc0_core as core
core.frozen_api()
from tests.test_bayesreg.r4 import test_r4_rev8 as t
import r4_common as rc, r4_eval as re_, r4_steps
world = t.build_dev_world(TMP, setattr)
out = {}
root = t.dev_root(world, 'r4')
out['clean'] = dict(code=r4_steps.eval_dev(SimpleNamespace(read=1, arm='B', dry_run=False), root=root), record=rc.read_json(root/'dev_eval'/'opened_files_1.json'),
                    doc=rc.read_json(root/'dev_eval'/'dev_eval_1.json'))
stray = TMP/'elsewhere'/'stray_metrics.json'
stray.parent.mkdir()
stray.write_text('{}', encoding='utf-8')
real = re_.hs_rebuild
def rebuild(pair, grid, *, hs_root):
    rc.read_json(stray)                                                                             # 一份落在 record_dirs 之外的评测记录
    return real(pair, grid, hs_root=hs_root)
re_.hs_rebuild = rebuild
root = t.dev_root(world, 'r4b')
out['stray'] = dict(code=r4_steps.eval_dev(SimpleNamespace(read=1, arm='B', dry_run=False), root=root), record=rc.read_json(root/'dev_eval'/'opened_files_1.json'), stray=str(stray))
out['paths'] = dict(stage=str(world.stage), hs=str(world.hs))
print(json.dumps(out, default=str))
'''


@pytest.mark.skipif(not FROZEN.is_dir(), reason='the frozen tree is not on this machine')
def test_the_development_reading_unit_passes_its_spy_with_the_hs_records_and_fails_on_a_stray_one(tmp_path):
    env = dict(os.environ, PYTHONPATH=f'{FROZEN}:{ROOT}', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run([PY, '-B', '-c', DEV_UNIT, str(ROOT), str(tmp_path)], capture_output=True, text=True, env=env, timeout=1200, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-4000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    clean, stray, paths = out['clean'], out['stray'], out['paths']
    record = clean['record']
    assert clean['code'] == 0
    assert record['he_spy']['passed'] is False and any('repair_metrics.json' in p for p in record['he_spy']['evaluation_records_opened'])   # 复核的阻断：HE 的 passed 必然假
    assert record['base_passed'] is True and record['passed'] is True and record['violations'] == [] and record['truth_read_problems'] == []
    assert sorted(tuple(row['key']) for row in record['truth_logical']) == sorted((p,) for p in DEV_PAIRS)                 # 真值的逻辑读恰好 19 对
    assert record['truth_members'] and all(m['inside'] and Path(m['file']).parent.name in DEV_PAIRS for m in record['truth_members'])
    judged = [r for r in record['opened'] if r['judged']]
    assert any(paths['hs'] in r['resolved'] for r in judged) and any(paths['stage'] in r['resolved'] for r in judged)     # tmp 下的读都由真实的白名单判过
    assert clean['doc'].get('failed') is None and clean['doc']['control']['mismatches'] == [] and sorted(clean['doc']['rows']) == sorted(DEV_PAIRS)
    assert stray['code'] == 8 and stray['record']['passed'] is False and stray['record']['base_passed'] is False
    assert any(v['path'] == stray['stray'] for v in stray['record']['violations'])


# ---- 重要 2：eval-bench 的单元主体真跑（七臂 × 16 对的恒等退路输出；真实的 LabelGuard 读合成 NIfTI） --------------------------------------------------------
@pytest.fixture()
def bench_world(tmp_path, bench, monkeypatch):
    import nibabel as nib
    from bayesreg.seg.unet.groups import load_grouping
    from bayesreg.seg.unet.synthetic import write_subjects
    data = tmp_path/'data'
    subjects = list(bench.ordered_subjects())+[r4_steps.DEV_REFERENCE_SUBJECT]
    write_subjects(data, subjects, SHAPE_XYZ, tissue4=load_grouping(name='tissue4_ref'))               # aligned_seg35 与 aligned_seg4
    for subject in subjects:
        path = data/subject/'aligned_norm.nii.gz'
        old = nib.load(str(path))
        nib.save(nib.Nifti1Image(tissue_image(SHAPE_XYZ, int(subject.split('_')[2])), old.affine, old.header), str(path))
    monkeypatch.setattr(rc, 'DATA_ROOT', data)
    root = tmp_path/'r4'
    pins = dict(registry_sha256=rc.sha256_file(ROOT/rc.BENCH_REGISTRY_REL), contact_sha256=rc.sha256_file(ROOT/rc.BENCH_CONTACT_REL))
    rc.write_json(root/'batch.json', dict(role='bench', checkout=str(rc.ROOT.resolve()), code_commit='c', resume_repo=str(tmp_path/'resume'), pairs_table_sha256=rb.pairs_table_sha256(bench), **pins))
    monkeypatch.setattr(r4_steps, '_identity', lambda root_, role: (rc.read_json(Path(root_)/('batch.json' if role == 'bench' else 'dev_batch.json')), dict(commit='c')))
    monkeypatch.setattr(r4_steps, 'dev_grid_meta', lambda: grid_of(data/r4_steps.DEV_REFERENCE_SUBJECT/'aligned_norm.nii.gz'))
    assert r4_steps.prepare_bench(SimpleNamespace(dry_run=False), root=root) == 0
    shape = rc.read_json(root/'bench_stage'/bench.pairs[0][0]/'manifest.json')['shape_zyx']
    for arm in rc.ARM_ORDER:
        for pair in bench.pair_names:
            identity_final(root, 'bench', arm, pair, shape)
        rc.write_json(root/'final'/arm/'opened_files_finalize.json', dict(passed=True))
    judge_tmp_too(monkeypatch, tmp_path)
    return root


def test_the_first_bench_read_unit_passes_its_spy_under_the_real_whitelist(bench_world, bench):
    root = bench_world
    assert r4_steps.eval_bench(SimpleNamespace(read='first', combo=None, dry_run=False), root=root) == 0
    record = rc.read_json(root/'eval'/'opened_files_first.json')
    assert record['passed'] is True and record['violations'] == [] and record['truth_read_problems'] == []
    assert len(record['truth_logical']) == 2*len(bench.ordered_subjects())
    judged = [r for r in record['opened'] if r['judged'] and str(root.parent) in r['resolved']]
    assert any('bench_stage' in r['resolved'] for r in judged) and any('/final/' in r['resolved'] for r in judged) and any('aligned_seg35' in r['resolved'] for r in judged)
    doc = rc.read_json(root/'eval'/'bench_eval_1.json')
    assert doc.get('failed') is None and sorted(doc['rows']) == sorted(rc.ARM_ORDER) and doc['control']['mismatches'] == []


# ---- 重要 1：导入核对的变异测试 -----------------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not FROZEN.is_dir(), reason='the frozen tree is not on this machine')
def test_the_import_check_reports_a_renamed_attribute_of_a_module_assigned_alias(tmp_path):
    """r4_eval.py 的 core（模块级 `core, he = ch.core, ch.he` 得到的别名）上的 frozen_api 改名：核对必须报出来。代码树是 tmp 里的一份：experiments/r4 复制并改动，其余链接到真的。"""
    work = tmp_path/'tree'
    (work/'experiments').mkdir(parents=True)
    for entry in ROOT.iterdir():
        if entry.name not in ('experiments', '.git') and not entry.name.startswith('.'):
            (work/entry.name).symlink_to(entry)
    for entry in (ROOT/'experiments').iterdir():
        if entry.name != 'r4':
            (work/'experiments'/entry.name).symlink_to(entry)
    shutil.copytree(ROOT/'experiments'/'r4', work/'experiments'/'r4', ignore=shutil.ignore_patterns('__pycache__'))
    target = work/'experiments'/'r4'/'r4_eval.py'
    text = target.read_text(encoding='utf-8')
    assert 'core.frozen_api()' in text
    target.write_text(text.replace('core.frozen_api()', 'core.frozen_api_renamed()'), encoding='utf-8')
    env = dict(os.environ, PYTHONPATH=f'{FROZEN}:{work}', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run([PY, '-B', str(Path(__file__).with_name('_import_check.py')), str(work)], capture_output=True, text=True, env=env, timeout=600, cwd=work)
    assert done.returncode == 0, done.stderr[-3000:]
    problems = json.loads(done.stdout.strip().splitlines()[-1])['problems']
    assert problems and all('frozen_api_renamed' in p and p.startswith('r4_eval.py:') for p in problems), problems
