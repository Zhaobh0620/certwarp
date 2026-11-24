"""R4 的代码复核（0c9d1c68：阻断 2、重要 5、次要 12；总负责人 2026-11-13 照复核建议裁决）落下的测试：
B1 被排除的候选不评、其余照常；B2 spy 的断言被拦截（端到端，InProcessChild，子进程里 PYTHONPATH = <frozen>:<repo>）；I1 读取器的拒绝与选项的异常；I4 读数只给计划里的那一项；
I5 拒绝归整台；次要 7 白名单收窄；次要 8 现有入口的反例、评测成功路径、A + C3、函数内导入；次要 12 描述项、原因串、开始核对。合成数据，不读真值，不用 GPU，不起单元。"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in ('r4', 'he', 'mc0', 'ef', 'fl', 'd0'):
    p = str(ROOT/'experiments'/_p)
    if p not in sys.path:
        sys.path.insert(0, p)

import r4_bench as rb  # noqa: E402
import r4_chain as ch  # noqa: E402
import r4_common as rc  # noqa: E402
import r4_decide as rd  # noqa: E402
import r4_eval as re_  # noqa: E402
import r4_lock as lk  # noqa: E402
import r4_options as r4o  # noqa: E402
import run_r4  # noqa: E402
from ._batch import batch_here  # noqa: E402

from .test_r4_bench import World  # noqa: E402
from .test_r4_chain import PAIRS, FakeBench, full_bench, prelock, resume, write_final  # noqa: E402

PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')


@pytest.fixture(autouse=True)
def no_guard(monkeypatch):
    monkeypatch.setattr(rb, 'bench_admission', lambda bench, **kw: dict(ok=True))


# ---- B1：锁后，被排除的候选不评，其余臂照常出行 ------------------------------------------------------------------------------------------------------
def fake_eval(monkeypatch, tmp_path):
    for pair in PAIRS:
        rc.write_json(tmp_path/pair/'manifest.json', dict(shape_zyx=[4, 4, 4], spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.]))
    truth = {(s, k): (np.zeros((4, 4, 4), dtype=np.uint8), dict(subject=s, kind=k, sha256='t', grid_shape_zyx=[4, 4, 4], spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.]))
             for p in PAIRS for s in p.split('__') for k in ('seg35', 'seg4')}
    monkeypatch.setattr(re_, 'bench_truth', lambda bench, files=None: (truth, SimpleNamespace(record=lambda: {})))
    monkeypatch.setattr(rb, 'load_bench_pair', lambda bench, stage, name: (SimpleNamespace(grid=None), None))
    monkeypatch.setattr(re_, 'evaluate_row', lambda final, labs, spacing, grid: dict(line_status=final['line_status'], metrics=dict(seg35=.8, seg4=.9, sdlogj_official=.07)))
    monkeypatch.setattr(re_, 'independent_control', lambda final, truth, f, m, spacing: dict(seg35=.8, seg4=.9, sdlogj_official=.07))


def test_an_excluded_candidate_is_not_evaluated_and_the_rest_read_normally(tmp_path, monkeypatch):
    full_bench(tmp_path, corrupt={('A2', 4)})
    fake_eval(monkeypatch, tmp_path)
    doc = re_.bench_eval(tmp_path, FakeBench(), read='first', stage_dir=tmp_path, repo=tmp_path, code_commit='c', plan_sha256='p', require_committed=False, rebuild=lambda *a, **k: None)
    assert 'failed' not in doc and doc.get('whole_undecidable') is None and 'A2' not in doc['rows'] and sorted(doc['rows']) == sorted(set(rc.ARM_ORDER)-{'A2'})
    assert 'A2' in doc['excluded'] and doc['control']['mismatches'] == []


# ---- B2：spy 的断言被拦截 ----------------------------------------------------------------------------------------------------------------------------
def test_the_supervisor_reads_the_childs_spy_record(tmp_path):
    (tmp_path/'t88').mkdir()
    assert ch.child_spy_problem(tmp_path, 't88', 0)                                                  # 记录不在
    assert ch.child_spy_problem(tmp_path, 't88', 'timeout') is None and ch.child_spy_problem(tmp_path, 't88', -9) is None     # 被杀：写不出记录，照 D74
    rc.write_json(tmp_path/'t88'/'opened_files.json', dict(passed=True))
    assert ch.child_spy_problem(tmp_path, 't88', 0) is None and ch.child_spy_problem(tmp_path, 't88', 8)
    (tmp_path/'repair').mkdir()
    rc.write_json(tmp_path/'repair'/'opened_files.json', dict(passed=False, violations=[dict(path='x')]))
    assert ch.child_spy_problem(tmp_path, 'repair', 0)


def test_a_failed_or_missing_spy_record_stops_before_the_lock(tmp_path):
    full_bench(tmp_path)
    (tmp_path/'runs'/'C1'/PAIRS[3]/'t88'/'opened_files.json').unlink()
    with pytest.raises(re_.PreLockStop, match='spy'):
        prelock(tmp_path)
    full_bench(tmp_path/'b')
    rc.write_json(tmp_path/'b'/'runs'/'CTL'/PAIRS[0]/'repair'/'opened_files.json', dict(passed=False))
    with pytest.raises(re_.PreLockStop, match='spy'):
        prelock(tmp_path/'b')


E2E = r'''
import json, sys
from pathlib import Path
import numpy as np
ROOT, TMP, MODE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path[:0] = [str(ROOT/'experiments'/d) for d in ('r4', 'he', 'mc0', 'ef', 'fl', 'd0')]          # the repository root comes from PYTHONPATH, after the frozen tree
import mc0_core as core
core.frozen_api()
from scipy.ndimage import gaussian_filter
from tests.test_bayesreg.he import _world as hw
from tests.test_bayesreg.solve import test_d0_a4_tools as a4
from tests.test_bayesreg.io_seg._helpers import source_sha
a4.SHAPE = hw.GRID
import r4_bench as rb, r4_chain as ch, r4_common as rc, run_r4, spies
from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, group_probabilities, load_grouping, require_inference_grouping
bench = rb.load_bench()
pins = dict(registry_sha256=rc.sha256_file(ROOT/rc.BENCH_REGISTRY_REL), contact_sha256=rc.sha256_file(ROOT/rc.BENCH_CONTACT_REL))
registry, contact, record = rb.load_bench_registry(bench, pins)
pair = bench.pairs[0][0]
fixed, moving = pair.split('__')
d = TMP/'stage'/pair
d.mkdir(parents=True)
L = hw.label_map()
maps = {fixed: L, moving: np.roll(L, 1, axis=0)}
rng = np.random.default_rng(3)
images = {s: gaussian_filter(rng.uniform(.1, .7, hw.GRID), 1.) for s in (fixed, moving)}
np.savez_compressed(d/'pair.npz', fixed=images[fixed], moving=images[moving], initial_u=np.zeros(hw.GRID+(3,)))
(d/'manifest.json').write_text(json.dumps(dict(pair=pair, fixed_subject=fixed, moving_subject=moving, shape_zyx=list(hw.GRID), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm',
                                               sources=dict(fixed=dict(sha256=source_sha(fixed)), moving=dict(sha256=source_sha(moving))), pair_sha256=rc.sha256_file(d/'pair.npz'))))
grouping = load_grouping(DEFAULT_GROUPS_PATH, 'structure13_v1')
check = require_inference_grouping(grouping)
model = rc.BENCH_MODELS['A']
for subject, m in maps.items():
    out = TMP/'pred'/subject
    out.mkdir(parents=True)
    noise = np.random.default_rng(len(subject)).uniform(0, 1, (36,)+hw.GRID)
    probs = .6*(np.arange(36)[:, None, None, None] == m[None])+.4*noise/noise.sum(axis=0)
    probs /= probs.sum(axis=0)
    np.savez_compressed(out/'probs36.npz', p=probs.astype(np.float16))
    np.savez_compressed(out/'structure_groups.npz', p=group_probabilities(probs, grouping, dtype=np.float32).astype(np.float16))
    members = [dict(model_id=model, fold=0, seed=k, member_sha256=sha, temperature=1.1) for k, sha in enumerate(registry.models[model]['member_model_sha256'])]
    prov = rb.bench_provenance(subject, bench=bench, registry=registry, registry_record=record, contact=contact, models={model: registry.models[model]}, members=members,
                               meta=dict(sha256=source_sha(subject), grid_shape_zyx=list(hw.GRID), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.]), norm={}, grouping=grouping,
                               grouping_check=check, allowlist=bench.allowlist, code={}, environment={}, command=[], seconds=0.)
    prov['output_sha256'] = {n: rc.sha256_file(out/n) for n in ('probs36.npz', 'structure_groups.npz')}
    (out/'provenance.json').write_text(json.dumps(prov))
    wm = np.where(np.isin(m, a4.MI.CEREBRAL_WM), .9, .05)
    csf = .3*(1-wm)
    hw.as_atropos_seg_writes(a4.write_atropos(TMP/'atropos', subject, posteriors=np.stack([csf, 1-wm-csf, wm])))
for k, sha in enumerate(registry.models[model]['member_model_sha256']):
    (TMP/'models'/f'foldA_s{k}').mkdir(parents=True)
    (TMP/'models'/f'foldA_s{k}'/'member.json').write_text(json.dumps(dict(member_sha256=sha, temperature=1.1)))
root = TMP/'r4'
paths = dict(ch.default_paths('bench', root), pred_root=str(TMP/'pred'), atropos_root=str(TMP/'atropos'), models_root=str(TMP/'models'), cache=str(hw.jit_cache()))
arm = 'B' if MODE == 'good_b' else 'CTL'
ctx = ch.make_context('bench', arm, pair, root=root, stage_dir=TMP/'stage', paths=paths, device='cpu', check_identity=False, test=True, bench_pins=pins)
ctx['settings']['r4']['spec']['native'] = 3                                       # a short native level (the coarse levels are the frozen 50 / 50 / 100)
ctx['settings_sha256'] = ch.he.settings_sha256(ctx['settings'])


def fake_repair(c):
    """The repair child's interface: admit, the source check (t88/run.json only), a certified output written as repair_main writes it (no numba)."""
    fz = core.frozen_api()
    out = ch.started(Path(c['out'])/'repair', c, 'repair')
    pin, _ = ch.admit(c)
    u, source = ch.source_checked(Path(c['out'])/'t88', fz, pin, c)
    digest = ch.he.array_sha256(u)
    np.savez(out/'repair.npz', nodes=np.zeros(0, dtype=np.int64), values=np.zeros((0, 3)))
    np.savez(out/'certified.npz', u=u)
    cert = dict(status='certified_feasible', min_jd_lower_bound=.5, n_infeasible=0, n_uncertain=0, state_hash=digest, process='test')
    ch.he.write_json(out/'cert.json', cert)
    rc.write_json(out/'repair.json', dict(schema=rc.SCHEMA.format('repair'), pair=c['pair'], status='certified', passed=True, reason='certified', source=source, certificate=cert, rounds=[],
                                          decision=dict(file='certified.npz', sha256=core.sha_file(out/'certified.npz'), state_hash=digest, cert='cert.json', cert_sha256=core.sha_file(out/'cert.json')),
                                          attempt=dict(file='repair.npz', sha256=core.sha_file(out/'repair.npz'))))
    return 0


ch.repair_main = fake_repair
if MODE == 'breach':
    real = ch.t88_main

    def breaching(c):
        pin, _ = ch.admit(c)
        np.load(Path(c['stage_dir'])/c['pair']/'pair.npz')['fixed_seg35']           # a label member of the original: the spy raises Breach
        return real(c)
    ch.t88_main = breaching
code = run_r4.supervise(ctx, run=lambda c: ch.run_pair(c, child=ch.InProcessChild()))
out = Path(ctx['out'])
line = ch.line_record(out)
t88 = json.loads((out/'t88'/'run.json').read_text()) if (out/'t88'/'run.json').is_file() else None
spy = json.loads((out/'t88'/'opened_files.json').read_text())
queue = None
if MODE == 'breach':
    rc.write_json(TMP/'q'/'batch.json', dict(role='bench', checkout=str(rc.ROOT.resolve()), resume_repo=str(TMP), note='a test batch (修订 6a M2)'))
    queue = run_r4.queue_arm('bench', 'A1', root=TMP/'q', pairs=[pair], gate=lambda own: (True, {}), gap=0, repo=TMP, require_committed=False,
                             launch_unit=lambda name, args: run_r4.supervise(dict(ctx, arm='A1', out=str(TMP/'q'/'runs'/'A1'/pair)), run=lambda c: ch.run_pair(c, child=ch.InProcessChild())))
print(json.dumps(dict(code=code, line_status=line['line_status'], output=(out/'output'/'final.npz').is_file(), t88_status=None if t88 is None else t88['status'], spy_passed=spy['passed'],
                      breach=spy.get('breach'), loader_calls=spy['he_spy']['loader_calls'], queue=queue, stops=len(list((TMP/'q'/'stops').glob('stop_*.json'))) if MODE == 'breach' else None,
                      soft=None if t88 is None else t88.get('soft_channels') is not None, supervisor=json.loads((out/'opened_files_supervisor.json').read_text())['passed'])))
'''


def run_e2e(tmp_path, mode):
    env = dict(os.environ, PYTHONPATH=f'{FROZEN}:{ROOT}', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    done = subprocess.run([PY, '-B', '-c', E2E, str(ROOT), str(tmp_path), mode], capture_output=True, text=True, env=env, timeout=1500, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not FROZEN.is_dir(), reason='the frozen tree is not on this machine')
def test_end_to_end_a_spy_violation_stops_the_queue_and_leaves_no_output(tmp_path):
    out = run_e2e(tmp_path, 'breach')
    assert out['code'] == ch.STOP_EXIT and out['line_status'] == rc.IMPLEMENTATION and out['output'] is False and out['spy_passed'] is False and out['breach']
    assert out['queue'] == 9 and out['stops'] == 1


@pytest.mark.skipif(not FROZEN.is_dir(), reason='the frozen tree is not on this machine')
@pytest.mark.parametrize('mode', ['good', 'good_b'])
def test_end_to_end_run_pair_and_t88_main_on_a_synthetic_bench_pair(tmp_path, mode):
    out = run_e2e(tmp_path, mode)
    assert out['code'] == 0 and out['line_status'] == rc.HX and out['output'] is True and out['t88_status'] == 'done' and out['spy_passed'] is True and out['supervisor'] is True
    assert out['loader_calls'] and out['soft'] is (mode == 'good_b')


# ---- I1：读取器的拒绝；选项开着时没写 run.json ----------------------------------------------------------------------------------------------------------
def test_a_refusal_of_the_bench_reader_before_the_sides_is_provenance_rejected(monkeypatch):
    from bayesreg.io.pair import ProvenanceRejected
    ctx = dict(role='bench', root='/nonexistent/r4i1', paths=dict(pred_root='/x', atropos_root='/y'), workers=1)
    ch._BENCH['/nonexistent/r4i1'] = (None, None, None, None)

    def refuse(*a, **k):
        raise ProvenanceRejected('cross-fold pair')
    monkeypatch.setattr(rb, 'BenchLayer', refuse)
    with pytest.raises(SystemExit, match='refused'):
        ch.read_channels(ctx, None, None)
    del ch._BENCH['/nonexistent/r4i1']


def test_an_arm_with_options_whose_t88_wrote_nothing_stops():
    assert ch.t88_status_decision(None, 1, ['A1'])[0] == 'implementation'
    assert ch.t88_status_decision(None, 1, [])[0] == 'fallback'                                    # 现行臂：D74
    assert ch.t88_status_decision(None, -9, ['B'])[0] == 'fallback' and ch.t88_status_decision(None, 'timeout', ['B'])[0] == 'fallback'
    assert issubclass(r4o.SoftSourceMismatch, r4o.SoftArgmaxMismatch) and r4o.SoftSourceMismatch.reason == 'soft_probs_sha_mismatch' != r4o.SoftArgmaxMismatch.reason
    assert r4o.OptionError('soft_error', 'x').reason == 'soft_error' and r4o.OptionError in r4o.IMPLEMENTATION_ERRORS


def test_the_t88_start_checks():
    import he_core
    s = ch.settings_for('CTL', 'dev')
    ctx = dict(role='dev', settings=s, paths=dict(registry=str(ROOT/rc.HS_REGISTRY_REL)))
    assert ch.start_checks(ctx) == []
    bad = dict(s, channels=dict(s['channels'], sigma=1.))
    assert ch.start_checks(dict(ctx, settings=bad))
    bad = dict(s, segmenter=dict(he_core.CONTRACT_SEGMENTER, segmenter_id='unet35_v1_ens'))
    assert ch.start_checks(dict(ctx, settings=bad))
    assert ch.start_checks(dict(role='bench', settings=ch.settings_for('B', 'bench', bench_registry_sha256='0'*64), paths={})) == []


# ---- I4：读数只给计划里的那一项 ---------------------------------------------------------------------------------------------------------------------
def test_the_queue_takes_only_the_planned_arms(tmp_path, monkeypatch):
    assert run_r4.queue_arm_problem(tmp_path, 'bench', 'C3') is None and run_r4.queue_arm_problem(tmp_path, 'bench', 'C4')
    assert run_r4.queue_arm_problem(tmp_path, 'bench', 'COMBO-B-C1')
    rc.write_json(tmp_path/'analysis'/'bench_decision_first.json', dict(combo='COMBO-B-C1', combo_needs_second_read=True))
    assert run_r4.queue_arm_problem(tmp_path, 'bench', 'COMBO-B-C1') is None and run_r4.queue_arm_problem(tmp_path, 'bench', 'COMBO-B-C3')
    monkeypatch.setattr(re_, 'dev_state', lambda root, repo=None: (['B', 'C1'], {}, []))
    assert run_r4.queue_arm_problem(tmp_path, 'dev', 'B') is None and run_r4.queue_arm_problem(tmp_path, 'dev', 'C1')


def test_a_development_reading_needs_the_feasibility_and_the_order(tmp_path):
    with pytest.raises(re_.PreLockStop, match=r'\(0\)'):
        re_.dev_prelock(tmp_path, 'C1', pairs=['D0'], hs_root=tmp_path, repo=tmp_path, read_number=1, state=(['B', 'C1'], {'B': rd.FEASIBLE, 'C1': rd.FEASIBLE}, []))
    with pytest.raises(re_.PreLockStop) as info:
        re_.dev_prelock(tmp_path, 'B', pairs=['D0'], hs_root=tmp_path, repo=tmp_path, read_number=1, state=(['B', 'C1'], {'B': rd.FEASIBLE}, []))
    assert 'next_dev_step' not in str(info.value) and '这次读数不是它' not in str(info.value)       # the item and the number are right: only the other items fail


# ---- I5：拒绝归整台 ---------------------------------------------------------------------------------------------------------------------------------
def test_a_rejection_in_a_candidate_stops_with_the_bench_scope(tmp_path):
    def launch(name, args):
        d = tmp_path/'runs'/args[4]/args[6]
        d.mkdir(parents=True)
        rc.write_json(d/'line.json', dict(line_status=rc.REJECTED))
        return 9
    batch_here(tmp_path)
    assert run_r4.queue_arm('bench', 'C1', root=tmp_path, pairs=PAIRS[:1], launch_unit=launch, gate=lambda own: (True, {}), gap=0, repo=tmp_path, require_committed=False) == 9
    assert rc.read_json(re_.stop_files(tmp_path)[0])['scope'] == 'bench'


def test_a_rejection_in_a_jia_arm_is_seen_before_the_lock(tmp_path):
    full_bench(tmp_path)
    for pair in PAIRS:
        (ch.final_dir(tmp_path, 'bench', 'B', pair)/'final.json').unlink()
    rc.write_json(tmp_path/'runs'/'B'/PAIRS[0]/'line_rejected.json', {})
    (tmp_path/'runs'/'B'/PAIRS[0]/'line.json').unlink()
    rc.write_json(tmp_path/'runs'/'B'/PAIRS[0]/'line.json', dict(line_status=rc.REJECTED))
    resume(tmp_path, re_.write_stop(tmp_path, dict(arm='B', scope='candidate', reason='x')))
    with pytest.raises(re_.PreLockStop, match='hx_provenance_rejected'):
        prelock(tmp_path)


# ---- 次要 8：现有入口的反例 --------------------------------------------------------------------------------------------------------------------------
def test_more_existing_entries_refuse_the_bench(tmp_path):
    world = World(tmp_path, rb.load_bench.__wrapped__() if hasattr(rb.load_bench, '__wrapped__') else rb.load_bench())
    subject = world.pin.fixed_subject
    import mc0_core
    mi = mc0_core.make_inputs()
    with pytest.raises(SystemExit):
        mi.a4_admission([subject], ROOT/rc.ALLOWLIST_REL, mi.UNetSource())                       # 训练集受试者
    from bayesreg.seg.structure_layer import StructureLayer, StructureSource
    from bayesreg.io.pair import ProvenanceRejected
    for staging in (False, True):
        with pytest.raises(ProvenanceRejected):
            StructureLayer(world.pin, StructureSource(root=tmp_path/'pred', registry_path=ROOT/rc.BENCH_REGISTRY_REL, allowlist_path=ROOT/rc.ALLOWLIST_REL, models_root=tmp_path,
                                                      image_root=tmp_path, segmenter_id=rc.BENCH_SEGMENTER_ID, allow_staging=staging))
    sys.path.insert(0, str(ROOT/'experiments'/'hs'))
    import hs_seg
    from bayesreg.io.guard import load_dev_allowlist
    from bayesreg.io.pair import load_segmenter_registry
    hs = load_segmenter_registry(ROOT/rc.HS_REGISTRY_REL)
    with pytest.raises(Exception):
        hs_seg.uf_plan(hs, next(iter(hs_seg.hc.CANDIDATES)), subject, load_dev_allowlist(ROOT/rc.ALLOWLIST_REL), {})
    sys.path.insert(0, str(ROOT/'experiments'/'hi'))
    import hi_core
    with pytest.raises(hi_core.HIRefused):
        hi_core.check_registry(load_segmenter_registry(ROOT/rc.BENCH_REGISTRY_REL))
    with pytest.raises(hi_core.HIRefused):
        hi_core.admit(subject, unseal=None, allowlist=load_dev_allowlist(ROOT/rc.ALLOWLIST_REL), sealed=None, training=None)


# ---- 次要 8：开发集读数的成功路径 --------------------------------------------------------------------------------------------------------------------
def test_the_development_reading_success_path_uses_one_evaluation_path(tmp_path, monkeypatch):
    pairs = [f'D{k}' for k in range(19)]
    u = np.zeros((4, 4, 4, 3))
    finals = {p: dict(line_status=rc.HX, run_dir=str(tmp_path/'none'), output=None) for p in pairs}
    monkeypatch.setattr(re_, 'dev_prelock', lambda *a, **k: (finals, {p: u for p in pairs}, {}))
    monkeypatch.setattr(re_, 'run_hx_admit', lambda stage, pair: (SimpleNamespace(grid=None), None))
    seen = []

    def evaluate_row(final, labs, spacing, grid):
        seen.append(final.get('note', 'arm')[:2])
        return dict(line_status=rc.HX, metrics=dict(seg35=.88, seg4=.92, sdlogj_official=.07))
    monkeypatch.setattr(re_, 'evaluate_row', evaluate_row)
    decision = {p: dict(pair=__import__('mc0_core').short_name(p) if False else p, metrics=dict(seg35=.88, seg4=.92, sdlogj_official=.07)) for p in pairs}
    (tmp_path/'hs'/'analysis').mkdir(parents=True)
    (tmp_path/'hs'/'analysis'/'decision_hs.json').write_text(json.dumps(dict(per_pair=list(decision.values()))))
    monkeypatch.setattr(ch.core, 'short_name', lambda p: p)
    doc = re_.dev_eval(tmp_path, 'B', 1, pairs=pairs, hs_root=tmp_path/'hs', repo=tmp_path, code_commit='c', plan_sha256='p', truth=lambda pair: ({}, dict(spacing_xyz=[1., 1., 1.]), 'sha'))
    assert 'failed' not in doc and doc['undecidable'] is None and doc['control']['mismatches'] == [] and seen.count('HS') == 19 and seen.count('ar') == 19
    assert (tmp_path/'dev_eval'/'hs_outputs_1_0'/'D0'/'output'/'final.npz').is_file()


# ---- 次要 8：A + C3 外侧 .65；函数内导入 ------------------------------------------------------------------------------------------------------------------
def test_a_with_c3_uses_065_outside_the_mask():
    import torch
    import mc0_core
    spec = rc.arm_spec('COMBO-A1-C3')
    _, pre = r4o.preopt_config(SimpleNamespace(PreoptConfig=lambda **k: SimpleNamespace(**k)), ch.settings_for('CTL', 'bench')['t88']['pre'], spec)
    assert pre['sigma_field'] == .65 and spec['sigma_small'] == .42
    rng = np.random.default_rng(0)
    F = torch.tensor(rng.uniform(0, 1, (2, 8, 9, 10)), dtype=torch.float32)
    M = torch.tensor(rng.uniform(0, 1, (2, 8, 9, 10)), dtype=torch.float32)

    def run(runner, sigma_field):
        u = torch.zeros((3, 6, 7, 8), dtype=torch.float32, requires_grad=True)
        cfg = SimpleNamespace(iterations=(2,), seconds=1e9, mu=(.01,), alpha=(0.,), kappa_pre=10., j0=.2, sigma_grad=1., beta1=.9, beta2=.999, lr=.5, eps=1e-8, cap=.5, sigma_field=sigma_field)
        m, v = torch.zeros_like(u), torch.zeros_like(u)
        runner(u, m, v, F, M, (1., 1., 1.), 0, cfg, __import__('time').perf_counter(), None, dict(cap_hits=0))
        return u.detach()
    zero = r4o.masked_level_runner(mc0_core, {0: np.zeros((6, 7, 8), dtype=np.float32)}, spec['sigma_small'])
    one = r4o.masked_level_runner(mc0_core, {0: np.ones((6, 7, 8), dtype=np.float32)}, spec['sigma_small'])
    assert torch.equal(run(zero, pre['sigma_field']), run(mc0_core.run_level_k, .65))              # 掩膜外：S_.65
    assert torch.equal(run(one, pre['sigma_field']), run(mc0_core.run_level_k, .42))               # 掩膜里：S_.42


IMPORTS = r'''
import ast, json, sys
from pathlib import Path
ROOT = Path(sys.argv[1])
sys.path[:0] = [str(ROOT/'experiments'/d) for d in ('r4', 'he', 'mc0', 'ef', 'fl', 'd0', 'hs')]
names = set()
for path in sorted((ROOT/'experiments'/'r4').glob('*.py')):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
failed = {}
for name in sorted(names):
    try:
        __import__(name)
    except Exception as exc:
        failed[name] = type(exc).__name__
print(json.dumps(dict(n=len(names), failed=failed, loaded=sorted(m for m in sys.modules if 'decide_hs' in m or m in ('run_hs', 'rp_core')))))
'''


def test_every_module_r4_imports_anywhere_keeps_decide_hs_out(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    done = subprocess.run([sys.executable, '-B', '-c', IMPORTS, str(ROOT)], capture_output=True, text=True, env=env, cwd=tmp_path, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out['n'] > 30 and out['loaded'] == [], out
    assert set(out['failed']) <= {'jd_curl.three_d.feasibility', 'jd_curl.three_d.multiresolution', 'jd_curl.three_d.preopt'}, out['failed']     # only the frozen tree's modules


# ---- 次要 12：描述项 ----------------------------------------------------------------------------------------------------------------------------------
def test_the_bench_decision_carries_the_descriptive_items():
    pairs = [f'p{k:02d}' for k in range(16)]
    subjects = [f'A{k}' for k in range(8)]+[f'B{k}' for k in range(8)]
    ps = {p: (subjects[k], subjects[(k+1) % 8+(8 if k >= 8 else 0)]) for k, p in enumerate(pairs)}
    m = dict(seg35=.85, seg4=.9, sdlogj_official=.07, small_six=.6, large_six=.9, other_23=.85, cortex=.8, white_matter=.9, hd95=2., sdlogj_brain=.05)
    ctl = {p: dict(line_status=rc.HX, metrics=m, pre_repair=dict(seg35=.84, seg4=.9), task_clock=100., repair_rounds=1) for p in pairs}
    arm = {p: dict(line_status=rc.FALLBACK_C if p == 'p00' else rc.HX, metrics=dict(m, seg35=.86, small_six=.65), pre_repair=dict(seg35=.85, seg4=.9), task_clock=120., repair_rounds=2) for p in pairs}
    d = rd.screen(arm, ctl, pairs, ps, subjects)['describe']
    assert abs(d['delta_small_six']['mean']-.05) < 1e-12 and d['fallbacks_arm'][rc.FALLBACK_C] == 1 and d['task_clock_mean_arm'] == 120. and abs(d['pre_repair_seg35_arm']-.85) < 1e-12


# ---- 次要 1、2、3、4、5、11、12 (5) ---------------------------------------------------------------------------------------------------------------------
def test_a_feasibility_stop_under_jia_is_not_feasible_even_without_its_record(tmp_path):
    rc.write_json(tmp_path/'analysis'/'bench_decision.json', dict(order=['B', 'C1']))
    resume(tmp_path, re_.write_stop(tmp_path, dict(arm='B', scope='feasibility', reason='a pair failed before feasibility.json')), scope='feasibility', arm='B')
    order, feasibility, readings = re_.dev_state(tmp_path, tmp_path, require_committed=False)
    assert feasibility == {'B': rd.NOT_FEASIBLE} and rd.next_dev_step(order, feasibility, readings) == dict(action='feasibility', item='C1', read_number=1)


class Bench16:
    pair_names = tuple(PAIRS)

    def pair_subjects(self):
        return {p: tuple(p.split('__')) for p in PAIRS}

    def ordered_subjects(self):
        return sorted({s for p in PAIRS for s in p.split('__')})


def test_a_combination_judged_before_its_read_still_gives_the_order(tmp_path):
    ctl = {p: dict(line_status=rc.HX, metrics=dict(seg35=.8, seg4=.9, sdlogj_official=.07)) for p in PAIRS}
    rows = {rc.CONTROL: ctl, 'B': {p: dict(r, metrics=dict(r['metrics'], seg35=.804)) for p, r in ctl.items()}, 'C1': {p: dict(r, metrics=dict(r['metrics'], seg35=.803)) for p, r in ctl.items()}}
    rc.write_json(tmp_path/'eval'/'bench_eval_1.json', dict(rows=rows, excluded={}, undecidable_before_truth={}))
    first = re_.bench_decide(tmp_path, Bench16(), repo=tmp_path, require_committed=False)
    assert first['combo'] == 'COMBO-B-C1' and first['combo_needs_second_read'] and (tmp_path/'analysis'/'bench_decision_first.json').is_file()
    resume(tmp_path, re_.write_stop(tmp_path, dict(arm='COMBO-B-C1', scope='combo', reason='the second read stopped before the lock')), scope='combo', arm='COMBO-B-C1')
    d = re_.bench_decide(tmp_path, Bench16(), repo=tmp_path, require_committed=False)
    assert d['combo_adopted'] is False and d['order'] == ['B', 'C1'] and (tmp_path/'analysis'/'bench_decision.json').is_file()


def test_every_failed_candidate_of_the_smoke_gets_its_own_stop(tmp_path, monkeypatch):
    import r4_steps
    monkeypatch.setattr(r4_steps, '_identity', lambda root, role: ({}, {}))
    monkeypatch.setattr(run_r4, 'bench_pairs', lambda root: PAIRS)
    monkeypatch.setattr(run_r4, 'pins_of', lambda root: {})
    monkeypatch.setattr(ch, 'make_context', lambda role, arm, pair, **kw: dict(arm=arm))
    good = dict(status='done', scale_check=dict(passed=True), env={})
    monkeypatch.setattr(r4_steps, 'smoke_arm', lambda ctx, cells: (good if ctx['arm'] not in ('A2', 'C2') else dict(good, status='vram_stop'), None, None))
    assert r4_steps.smoke(SimpleNamespace(dry_run=False, device='cpu'), root=tmp_path) == 9
    stops = [rc.read_json(p) for p in re_.stop_files(tmp_path)]
    assert sorted((s['scope'], s['arm']) for s in stops) == [('candidate', 'A2'), ('candidate', 'C2')]


def test_an_output_without_a_computable_sdlogj_stops_the_feasibility():
    rows = {f'd{k}': dict(line_status=rc.HX, sdlogj_official=.07) for k in range(19)}
    rows['d5']['sdlogj_official'] = None
    assert rd.feasibility_verdict(rows)['verdict'] == rd.FEAS_STOP


def test_the_identitys_third_certificate_failing_is_an_implementation_error(tmp_path, monkeypatch):
    from dataclasses import dataclass
    import he_core as he
    from jd_curl.three_d.feasibility import PrescreenCertificateConfig
    from jd_curl.three_d.grid import make_grid_3d
    grid = make_grid_3d(nx=5, ny=5, nz=5, hx=1., hy=1., hz=1., unit='mm')
    u = np.zeros(grid.shape+(3,))
    out = tmp_path/'r4'/'runs'/'B'/'P__Q'
    out.mkdir(parents=True)
    cert = dict(status='certified_feasible', min_jd_lower_bound=1., n_infeasible=0, n_uncertain=0, state_hash=he.array_sha256(u), process='decision')
    he.write_output(out/'output', u, grid, cert, dict(line_status=rc.FALLBACK_ID))
    monkeypatch.setattr(ch, 'admit', lambda ctx: (SimpleNamespace(grid=grid), None))

    @dataclass
    class Result:
        status: str
        min_jd_lower_bound: float
        n_infeasible: int
        n_uncertain: int
        state_hash: str
        n_cells_checked: int = 1
        witness_cell: object = None
        witness_reference_point: object = None

    def certify(v, g):
        return Result('certified_infeasible', -1., 2, 0, he.array_sha256(v))
    ctx = dict(out=str(out), root=str(tmp_path/'r4'), role='bench', arm='B', pair='P__Q', settings=dict(r4=dict(spec_sha256='s')))
    ch.finalize_pair(ctx, certification=(certify, PrescreenCertificateConfig()))
    final = ch.read_final(tmp_path/'r4', 'bench', 'B', 'P__Q')
    assert final['line_status'] == rc.IMPLEMENTATION and final['output'] is None and (out/'output_superseded_0').is_dir()


def test_the_finalize_record_carries_the_code(tmp_path):
    out = tmp_path/'r4'/'runs'/'B'/'P__Q'
    out.mkdir(parents=True)
    rc.write_json(out/'context.json', dict(role='bench', root=str(tmp_path/'r4'), arm='B', pair='P__Q', stage_dir=str(tmp_path/'stage'), out=str(out),
                                           paths=dict(pred_root=str(tmp_path), atropos_root=str(tmp_path))))
    rc.write_json(out/'line.json', dict(line_status=rc.REJECTED))
    assert run_r4.finalize_arm(tmp_path/'r4', 'bench', 'B', ['P__Q'], finalize=lambda c: None, code=dict(commit='abc', clean=True)) == 0
    assert rc.read_json(tmp_path/'r4'/'final'/'B'/'opened_files_finalize.json')['code'] == dict(commit='abc', clean=True)


def test_the_feasibility_reader_records_are_checked_again(tmp_path):
    run = tmp_path/'run'
    (run/'t88').mkdir(parents=True)
    rc.write_json(run/'t88'/'run.json', dict(status='done', oracle=True, data=dict(keys_read=['fixed', 'moving', 'fixed_seg35']), channel_sources={}))
    problems = re_.reader_record_problems(run, 'dev')
    assert any('oracle' in p for p in problems) and any('fixed_seg35' in p for p in problems) and any('channel_sources' in p for p in problems)
