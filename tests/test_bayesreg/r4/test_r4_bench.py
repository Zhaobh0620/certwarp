"""R4 的筛选台（合同 C25.6，C25.10-5）：16 人与 16 对、台守卫、台的登记与附属文件、结构层读取器的反例、现有入口照旧拒绝。
真实的只有 id（钉住的 folds_full_v2.json、白名单、封存名单经守卫读）；图像、预测都是 tmp 里的合成数组，不打开任何受试者文件（spy 核）。"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import MappingProxyType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in ('r4', 'he', 'mc0', 'ef', 'fl'):
    p = str(ROOT/'experiments'/_p)
    if p not in sys.path:
        sys.path.insert(0, p)

import r4_bench as rb  # noqa: E402
import r4_common as rc  # noqa: E402
from bayesreg.io.guard import load_dev_allowlist  # noqa: E402
from bayesreg.io.pair import ProvenanceRejected, SegmenterRegistry, load_pair  # noqa: E402
from jd_curl.validation import InputValidationError  # noqa: E402

from ..io_seg._helpers import source_sha  # noqa: E402

SHAPE = (8, 9, 10)


@pytest.fixture(scope='module')
def bench():
    return rb.load_bench()


def pins():
    return dict(registry_sha256=rc.sha256_file(ROOT/rc.BENCH_REGISTRY_REL), contact_sha256=rc.sha256_file(ROOT/rc.BENCH_CONTACT_REL))


# ---- 受试者与配对（C25.6-2、-3） -----------------------------------------------------------------------------------------------------------------
def test_sixteen_subjects_and_two_within_fold_rings(bench):
    assert [len(bench.subjects[f]) for f in rc.FOLDS] == [8, 8] and len(bench.pairs) == 16
    for f in rc.FOLDS:
        s = sorted(bench.subjects[f])
        rows = [p for p in bench.pairs if p[1] == f]
        assert [(p[3], p[4]) for p in rows] == [(s[i], s[(i+1) % 8]) for i in range(8)]
    assert [p[1] for p in bench.pairs] == ['A']*8+['B']*8
    assert len({p[3] for p in bench.pairs}) == 16 and len({p[4] for p in bench.pairs}) == 16          # 每人当一次固定图、一次移动图
    assert set(bench.subjects['A']) <= set(bench.split.subjects('B', 'train')) and set(bench.subjects['B']) <= set(bench.split.subjects('A', 'train'))


def test_a_changed_or_unpinned_folds_file_is_refused(tmp_path):
    changed = tmp_path/'folds_full_v2.json'
    changed.write_bytes((ROOT/rc.FOLDS_REL).read_bytes()+b'\n')
    with pytest.raises(rb.BenchRefused):
        rb.load_bench(rb.BenchWorld(folds_path=changed))
    with pytest.raises(rb.BenchRefused):
        rb.load_bench(rb.BenchWorld(folds_sha256='0'*64))


# ---- 台守卫（C25.6-6 (a)；C25.10-5） -----------------------------------------------------------------------------------------------------------
def outsiders(bench):
    """（开发集的一人，T0 的一人，B 线留出集的一人，H1 的一人，训练集其余 90 人中的一人）：只取 id（封存名单经 bayesreg.eval.truth 的加载器读；不打开任何受试者文件）。"""
    from bayesreg.eval.truth import load_sealed_lists
    sources = {s['name']: s for s in bench.allowlist.sources}
    dev = sorted(sources[rc.DEV_SOURCE]['subjects'])[0]
    t0 = sorted(bench.allowlist.never_for_training)[0]
    sealed = load_sealed_lists()
    holdout, h1 = sorted(sealed.subjects['holdout_B_v1'])[0], sorted(sealed.subjects['H1'])[0]
    other = sorted(set(bench.split.training_subjects)-bench.whitelist)[0]
    return dict(dev=dev, t0=t0, holdout=holdout, h1=h1, other_train=other)


def test_the_bench_guard_refuses_everyone_else_and_opens_no_subject_file(bench):
    import spies
    pair = bench.pairs[0][0]
    with spies.Spies(Path('/nonexistent/none.npz')) as s:
        rb.bench_admission(bench, pair=pair, model_id=rc.BENCH_MODELS['A'])
        for kind, subject in outsiders(bench).items():
            with pytest.raises(rb.BenchRefused):
                rb.bench_admission(bench, subjects=[subject])
        f, m = pair.split('__')
        a_last, b_first = bench.pairs[7][3], bench.pairs[8][3]
        for bad in (f'{m}__{f}', f'{a_last}__{b_first}', f'{f}__{bench.pairs[2][3]}'):                    # 颠倒、跨折、不在表里
            with pytest.raises(rb.BenchRefused):
                rb.bench_admission(bench, pair=bad)
        with pytest.raises(rb.BenchRefused):
            rb.bench_admission(bench, subjects=[f], model_id=rc.BENCH_MODELS['B'])                         # 另一折的模型
        with pytest.raises(rb.BenchRefused):
            rb.bench_admission(bench, subjects=[f], model_id=rc.HS_FOLD_MODELS['A'])
    data = [p for p in s.opened if str(p).startswith(str(rc.DATA_ROOT)) or 'seq_vp_3d/data' in str(p)]
    assert data == []


# ---- 台的登记（D134；C25.6-6 (b) 第四处） --------------------------------------------------------------------------------------------------------
def test_the_committed_bench_registry_and_contact_file(bench):
    registry, contact, record = rb.load_bench_registry(bench, pins())
    assert sorted(registry.models) == sorted(rc.BENCH_MODELS.values()) and record['staging'] is False
    for f in rc.FOLDS:
        c = contact['models'][rc.BENCH_MODELS[f]]
        assert len(c['label_contact_training']) == 98 and len(c['label_contact_calibration']) == 8
        assert sorted(c['label_contact_calibration']) == bench.subjects[f]
    from bayesreg.seg.structure_layer import FORMAL_REGISTRIES
    assert ROOT/rc.BENCH_REGISTRY_REL not in [Path(p) for p in FORMAL_REGISTRIES]


def git_repo(tmp_path):
    repo = tmp_path/'repo'
    for rel in (rc.BENCH_REGISTRY_REL, rc.BENCH_CONTACT_REL):
        (repo/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, repo/rel)
    for args in (['init', '-q'], ['add', '.'], ['-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'x']):
        subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)
    return repo


def test_another_changed_or_uncommitted_registry_or_contact_file_is_refused(bench, tmp_path):
    from dataclasses import replace
    repo = git_repo(tmp_path)
    world = replace(bench.world, repo=repo)
    b = replace(bench, world=world)
    rb.load_bench_registry(b, pins())
    with pytest.raises(rb.BenchRefused):
        rb.load_bench_registry(b, dict(pins(), registry_sha256='0'*64))                    # 别的登记（batch.json 钉的不是它）
    with pytest.raises(rb.BenchRefused):
        rb.load_bench_registry(b, dict(pins(), contact_sha256='0'*64))
    (repo/rc.BENCH_CONTACT_REL).write_text((repo/rc.BENCH_CONTACT_REL).read_text()+' ')    # 改过的附属文件（没提交）
    with pytest.raises(rb.BenchRefused):
        rb.load_bench_registry(b, pins())
    repo2 = git_repo(tmp_path/'two')
    doc = json.loads((repo2/rc.BENCH_REGISTRY_REL).read_text())
    doc['comment'] = 'changed'
    (repo2/rc.BENCH_REGISTRY_REL).write_text(json.dumps(doc))                              # 改过的登记
    with pytest.raises(rb.BenchRefused):
        rb.load_bench_registry(replace(bench, world=replace(bench.world, repo=repo2)), pins())
    import inspect
    assert list(inspect.signature(rb.load_bench_registry).parameters) == ['bench', 'pins']     # 没有路径参数，也没有 allow_staging（暂存登记无从进入）


# ---- 结构层读取器（C25.6-6 (b)；C25.10-5 的反例） --------------------------------------------------------------------------------------------------
class World:
    def __init__(self, tmp_path, bench):
        from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, group_probabilities, load_grouping, require_inference_grouping
        self.bench, self.tmp = bench, Path(tmp_path)
        self.registry, self.contact, self.record = rb.load_bench_registry(bench, pins())
        self.pair = bench.pairs[0][0]
        fixed, moving = self.pair.split('__')
        d = self.tmp/'stage'/self.pair
        d.mkdir(parents=True)
        rng = np.random.default_rng(0)
        np.savez_compressed(d/'pair.npz', fixed=rng.uniform(0, 1, SHAPE), moving=rng.uniform(0, 1, SHAPE), initial_u=np.zeros(SHAPE+(3,)))
        manifest = dict(pair=self.pair, fixed_subject=fixed, moving_subject=moving, shape_zyx=list(SHAPE), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm',
                        sources=dict(fixed=dict(sha256=source_sha(fixed)), moving=dict(sha256=source_sha(moving))), pair_sha256=rc.sha256_file(d/'pair.npz'))
        (d/'manifest.json').write_text(json.dumps(manifest))
        self.pin = load_pair(d/'pair.npz', allowlist=bench.world.allowlist_path)
        self.grouping = load_grouping(DEFAULT_GROUPS_PATH, 'structure13_v1')
        self.check = require_inference_grouping(self.grouping)
        self.provs = {}
        self.models_root = self.tmp/'models'
        for f in rc.FOLDS:
            for k, sha in enumerate(self.registry.models[rc.BENCH_MODELS[f]]['member_model_sha256']):
                (self.models_root/f'fold{f}_s{k}').mkdir(parents=True)
                (self.models_root/f'fold{f}_s{k}'/'member.json').write_text(json.dumps(dict(member_sha256=sha, temperature=1.1)))
        for side, subject in (('fixed', fixed), ('moving', moving)):
            out = self.tmp/'pred'/subject
            out.mkdir(parents=True)
            p = rng.uniform(0, .1, (36,)+SHAPE)
            p[1] += 1.
            p /= p.sum(axis=0)
            np.savez_compressed(out/'probs36.npz', p=p.astype(np.float16))
            np.savez_compressed(out/'structure_groups.npz', p=group_probabilities(p, self.grouping, dtype=np.float32).astype(np.float16))
            model = rc.BENCH_MODELS['A']
            members = [dict(model_id=model, fold=0, seed=k, member_sha256=sha, temperature=1.1) for k, sha in enumerate(self.registry.models[model]['member_model_sha256'])]
            prov = rb.bench_provenance(subject, bench=bench, registry=self.registry, registry_record=self.record, contact=self.contact, models={model: self.registry.models[model]}, members=members,
                                       meta=dict(sha256=source_sha(subject), grid_shape_zyx=list(SHAPE), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.]), norm={}, grouping=self.grouping,
                                       grouping_check=self.check, allowlist=bench.allowlist, code={}, environment={}, command=[], seconds=0.)
            prov['output_sha256'] = {n: rc.sha256_file(out/n) for n in ('probs36.npz', 'structure_groups.npz')}
            self.provs[side] = prov
            (out/'provenance.json').write_text(json.dumps(prov))

    def layer(self):
        return rb.BenchLayer(self.pin, self.bench, self.registry, self.contact, self.record, pred_root=self.tmp/'pred', models_root=self.models_root)

    def write(self, side='fixed', **changes):
        prov = dict(self.provs[side], **changes)
        subject = self.pin.fixed_subject if side == 'fixed' else self.pin.moving_subject
        (self.tmp/'pred'/subject/'provenance.json').write_text(json.dumps(prov))
        return prov


@pytest.fixture()
def world(tmp_path, bench):
    return World(tmp_path, bench)


def test_the_reader_accepts_the_bench_prediction(world):
    array, accepted, record = world.layer().load('fixed')
    assert array.shape[1:] == SHAPE and accepted['test_labels_seen'] == rc.BENCH_TEST_LABELS_SEEN and record['models']['policy'] == rc.BENCH_POLICY
    mark = rb.bench_oracle_mark(['fixed', 'moving', 'initial_u'], dict(fixed=dict(segmenter_id=rc.BENCH_SEGMENTER_ID, policy=rc.BENCH_POLICY), moving=dict(segmenter_id=rc.BENCH_SEGMENTER_ID, policy=rc.BENCH_POLICY)))
    assert mark == dict(oracle=False, truth_labels_read=False)
    assert rb.bench_oracle_mark(['fixed', 'moving'], dict(fixed=dict(segmenter_id=rc.BENCH_SEGMENTER_ID, policy='all_folds_mean'), moving=dict(segmenter_id=rc.BENCH_SEGMENTER_ID, policy=rc.BENCH_POLICY)))['oracle'] is True
    assert rb.bench_oracle_mark(['fixed', 'moving', 'fixed_seg35'], dict(fixed=dict(segmenter_id=rc.BENCH_SEGMENTER_ID, policy=rc.BENCH_POLICY), moving=dict(segmenter_id=rc.BENCH_SEGMENTER_ID, policy=rc.BENCH_POLICY)))['oracle'] is True


COUNTEREXAMPLES = dict(
    other_image=lambda w: dict(input_sha256={'aligned_norm.nii.gz': source_sha(w.pin.moving_subject)}),
    test_labels_false=lambda w: dict(test_labels_seen=False),
    test_labels_true=lambda w: dict(test_labels_seen=True),
    policy=lambda w: dict(policy='all_folds_mean'),
    other_fold_model=lambda w: dict(model_ids=[rc.BENCH_MODELS['B']], model_sha256={rc.BENCH_MODELS['B']: list(w.registry.models[rc.BENCH_MODELS['B']]['member_model_sha256'])},
                                    folds={rc.BENCH_MODELS['B']: 1}),
    hs_models=lambda w: dict(model_ids=[rc.HS_FOLD_MODELS['A']]),
    output_sha=lambda w: dict(output_sha256=dict(w.provs['fixed']['output_sha256'], **{'probs36.npz': '0'*64})),
    grouping=lambda w: dict(grouping='structure12_x'),
    grid=lambda w: dict(grid=dict(shape_zyx=[8, 9, 11], spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.])),
    oracle=lambda w: dict(oracle=True),
    forbidden_id=lambda w: dict(segmenter_id='unet_oracle_bench'),
    layer=lambda w: dict(layer='tissue'),
    class_names=lambda w: dict(class_names=list(w.provs['fixed']['class_names'])[::-1]),
    subject=lambda w: dict(subject=w.pin.moving_subject),
    folds=lambda w: dict(folds={rc.BENCH_MODELS['A']: 1}),
    not_bench_id=lambda w: dict(segmenter_id='unet35_v4_ens'),
    contact_record=lambda w: dict(label_contact=dict(w.provs['fixed']['label_contact'], side_file_sha256='0'*64)),
    members=lambda w: dict(members=w.provs['fixed']['members'][:1]),
    temperature=lambda w: dict(members=[dict(m, temperature=1.2) for m in w.provs['fixed']['members']]))


@pytest.mark.parametrize('case', sorted(COUNTEREXAMPLES))
def test_the_reader_refuses_each_counterexample(world, case):
    world.write(**COUNTEREXAMPLES[case](world))
    with pytest.raises((ProvenanceRejected, InputValidationError)):
        world.layer().load('fixed')


def test_contact_off_the_allowlist_or_never_for_training_and_registry_ids_are_refused(world, tmp_path):
    model = rc.BENCH_MODELS['A']
    entry = dict(world.registry.models[model])

    def registry_with(**changes):
        return SegmenterRegistry(path='mem', sha256='0'*64, models=MappingProxyType({**{m: e for m, e in world.registry.models.items()}, model: MappingProxyType(dict(entry, **changes))}))
    t0 = sorted(world.bench.allowlist.never_for_training)[0]
    for bad in (registry_with(label_contact_subjects=tuple(entry['label_contact_subjects'])+(t0,)), registry_with(label_contact_subjects=tuple(entry['label_contact_subjects'])+('OASIS_OAS1_0999_MR1',)),
                registry_with(segmenter_id='unet35_v4_ens')):
        with pytest.raises(ProvenanceRejected):
            rb.bench_validate_provenance(world.provs['fixed'], world.pin, side='fixed', registry=bad, allowlist=world.bench.allowlist, bench=world.bench)
    doc = json.loads((ROOT/rc.ALLOWLIST_REL).read_text())
    doc['comment'] = 'another allowlist'
    other = tmp_path/'other_allowlist.json'
    other.write_text(json.dumps(doc))
    with pytest.raises(InputValidationError):                                                     # 允许名单与 load_pair 用的不是同一份
        rb.bench_validate_provenance(world.provs['fixed'], world.pin, side='fixed', registry=world.registry, allowlist=load_dev_allowlist(other), bench=world.bench)


def test_a_cross_fold_pair_is_refused_before_any_prediction(world, bench):
    a, b = bench.pairs[0][3], bench.pairs[8][3]
    d = world.tmp/'stage'/f'{a}__{b}'
    d.mkdir(parents=True)
    shutil.copyfile(world.tmp/'stage'/world.pair/'pair.npz', d/'pair.npz')
    m = json.loads((world.tmp/'stage'/world.pair/'manifest.json').read_text())
    m.update(pair=f'{a}__{b}', fixed_subject=a, moving_subject=b, sources=dict(fixed=dict(sha256=source_sha(a)), moving=dict(sha256=source_sha(b))))
    (d/'manifest.json').write_text(json.dumps(m))
    pin = load_pair(d/'pair.npz', allowlist=bench.world.allowlist_path)
    with pytest.raises(ProvenanceRejected):
        rb.BenchLayer(pin, bench, world.registry, world.contact, world.record, pred_root=world.tmp/'pred', models_root=world.models_root)


def test_bench_read_side_differs_from_read_side_only_in_the_policy_statement():
    import ast
    import fast_channels
    from .test_r4_options import body_dump, function_def
    is_policy = lambda s: isinstance(s, ast.If) and 'policy' in ast.unparse(s.test)          # noqa: E731
    assert body_dump(function_def(rb, 'bench_read_side'), is_policy) == body_dump(function_def(fast_channels, 'read_side'), is_policy)


# ---- 现有入口照旧拒绝（C25.6-6 (d)；C25.10-5 末项） --------------------------------------------------------------------------------------------------
def test_existing_entries_refuse_the_bench_registry_and_subjects(world, bench):
    from bayesreg.io.pair import validate_segmentation_provenance
    from bayesreg.seg.structure_layer import expected_models, open_registry
    from bayesreg.seg.unet.infer import select_fold_models
    with pytest.raises(ProvenanceRejected):
        open_registry(ROOT/rc.BENCH_REGISTRY_REL)                                                     # 不是正式登记，也没有暂存的许可
    with pytest.raises(ProvenanceRejected):
        validate_segmentation_provenance(world.provs['fixed'], world.pin, side='fixed', layer='structure', registry=world.registry, allowlist=bench.allowlist)
    subject = world.pin.fixed_subject
    with pytest.raises(ProvenanceRejected):
        expected_models(world.registry, subject, segmenter_id=rc.BENCH_SEGMENTER_ID)
    from bayesreg.io.pair import load_segmenter_registry
    hs = load_segmenter_registry(ROOT/rc.HS_REGISTRY_REL)
    with pytest.raises(Exception):
        select_fold_models(hs, subject, bench.allowlist, segmenter_id='unet35_v4_ens')
    import he_core
    settings = dict(he_core.frozen_settings(), segmenter=dict(segmenter_id=rc.BENCH_SEGMENTER_ID, policy=rc.BENCH_POLICY, registry=rc.BENCH_REGISTRY_REL, registry_sha256=pins()['registry_sha256']))
    assert he_core.segmenter_start_problems(settings, registry=ROOT/rc.BENCH_REGISTRY_REL)[0]
    sides = {s: dict(segmenter_id=rc.BENCH_SEGMENTER_ID, policy=rc.BENCH_POLICY, registry_sha256=pins()['registry_sha256'], registry_path=rc.BENCH_REGISTRY_REL) for s in ('fixed', 'moving')}
    assert he_core.channel_source_problems(dict(sides=sides, segmenter=dict(segmenter_id=rc.BENCH_SEGMENTER_ID, registry=rc.BENCH_REGISTRY_REL)))


def test_the_header_check(tmp_path):
    import nibabel as nib
    a = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
    a.header.set_qform(np.eye(4), code=1)
    a.header.set_sform(np.eye(4), code=1)
    nib.save(a, str(tmp_path/'a.nii.gz'))
    b = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.diag([1., 1., 1.5, 1.]))
    nib.save(b, str(tmp_path/'b.nii.gz'))
    ra, rb_ = rb.header_record(tmp_path/'a.nii.gz'), rb.header_record(tmp_path/'b.nii.gz')
    assert rb.header_problems(ra, ra) == [] and rb.header_problems(rb_, ra)
