"""R2's pp diagnosis (experiments/r2/pp_r2.py) on PQ's synthetic world (tests/test_bayesreg/pq/_world.py: a stage whose pair.npz holds fixed_seg35 and the other label keys, the registered v4
predictions, the certified T88 and repair records with the evaluation records metrics_0600.json and repair_metrics.json beside them, a label file under the images).  CPU, tmp_path only.

- the world's run.json in the key shape of HS's real one (no top-level segmenter_id; the segmenter under channel_sources), not P3's: m1's segmenter check on those keys (pp's second start
  stopped on P3's key, 11-11 02:13:33 AM CST); every departure refused; bind_hs_field is pq_io.bind_field statement for statement but the segmenter one; the dry run's one real run.json;
- the unit end to end: pp per label equal to an independent recomputation (the evaluation's warp and Dice on the predicted labels); opened_files.json with no breach; from pair.npz only fixed, moving,
  initial_u; no seg key, no label file, no evaluation record opened (the coordinator's ruling of 11-10 ~8:25 PM: the pp diagnosis reads no truth);
- the firewall stops the unit: the member fixed_seg35 of a pair.npz, repair_metrics.json, a label file; the record says so and pp_r2.json does not pass;
- once: a second run is refused before anything is opened;
- the committed reference's rows: every pair hx_certified, hs_v1's directories; refusals;
- the split and the target rule (plan.md section 2): R3b's quantities, the left-right pairing, the eligibility, the top three, the ties; split() end to end on synthetic records.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from bayesreg.seg import structure_layer

from ..pq import _world as W
from ..pq._world import reset_sealed_cache

ROOT = Path(__file__).resolve().parents[3]
import sys  # noqa: E402
for _p in (ROOT/'experiments'/'r2', ROOT/'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import pp_r2  # noqa: E402
import r2_common as rc  # noqa: E402
import pq_firewall as fw  # noqa: E402

GATE = dict(passed=True, note='the tests query no machine')
CODE = dict(commit='0'*40, clean=True, note='the tests record no code tree')


def quiet(*args, **kwargs):
    pass


def hs_shaped_run(run, short, *, pred_root, registry, registry_sha256, v4, image_root='/images'):
    """The run.json of an HS field in the key shape of the real one (hs_v1/runs/SHx24_first5/0438-0439/run.json, schema bayesreg.fl.run.v1, read by the coordinator 11-11): no top-level
    segmenter_id; the segmenter under channel_sources -- segmenter {pred_root, folds, models_root, image_root, segmenter_id, registry, a5pv2_gate}, structure_layer {contract, source, registry
    {path, sha256, status, ...}, grouping, preflight, allowlist_sha256}, sides {fixed, moving} {unet_provenance_sha256, unet_output_sha256, unet_directory, segmenter_id, model_ids, policy, members,
    registry_path, registry_sha256, registry_status, registry_git, ..., subject, read_path}.  `run` gives pair, status, oracle, field, labels_sha256 (the world's; its other keys are dropped, so it can be applied again)."""
    f, m = W.pins.subjects_of(short)
    ids = ['unet35_v4_ens_foldA', 'unet35_v4_ens_foldB']

    def side(subject):
        pin = v4[subject]
        return dict(unet_provenance_sha256=pin.provenance, unet_output_sha256={'probs36.npz': pin.probs36, 'structure_groups.npz': pin.structure_groups}, unet_directory=str(Path(pred_root)/subject),
                    segmenter_id='unet35_v4_ens', model_ids=ids, policy='all_folds_mean', members=[dict(model_id=i, member_sha256='0'*64) for i in ids], registry_path=str(registry),
                    registry_sha256=registry_sha256, registry_status='registered', registry_git=dict(tracked=True, clean=True, last_commit='0'*40), group_sum_tol=.002, subject=subject, read_path='fast')
    registry_record = dict(path=str(registry), sha256=registry_sha256, status='registered', repository_registry=str(registry), note='synthetic', staging_opt_in=False, use='registered')
    out = dict(schema='bayesreg.fl.run.v1', run_id=f'hs_SHx24_first5_{short}', arm='SHx24', tier='first5', pair=run['pair'], k=35, oracle=run['oracle'], status=run['status'], labels_sha256=run['labels_sha256'],
               field=run['field'], taper=4)
    out['channel_sources'] = dict(oracle=False, truth_labels_read=False, pair_keys_read=['fixed', 'moving', 'initial_u'], label_firewall='whitelist-v2',
                                  segmenter=dict(pred_root=str(pred_root), folds='/code/splits/bayesreg_v1/folds.json', models_root='/seg/unet35_v4_full', image_root=str(image_root), segmenter_id='unet35_v4_ens',
                                                 registry=str(registry), a5pv2_gate=dict(rule='synthetic')),
                                  atropos_root='/seg/atropos_v1', subject_admission=dict(subjects=[f, m], n_training_subjects=106),
                                  structure_layer=dict(contract='C03.1-2/3', source=dict(root=str(pred_root), registry_path=str(registry), segmenter_id='unet35_v4_ens', infer='never'), registry=registry_record,
                                                       grouping=dict(grouping='structure13_v1'), preflight=dict(pair_subjects=[f, m], models_used=ids), allowlist_sha256='0'*64),
                                  sides=dict(fixed=side(f), moving=side(m)))
    return out


def to_hs_shape(w, change=None):
    """Every run.json of the world rewritten in HS's shape (then `change(short, document)`), its pin in w.ctx.run_json with it."""
    for short, row in w.ctx.p3.items():
        path = w.results/'p3_v1'/'runs'/row.t88_dir/short/'run.json'
        document = hs_shaped_run(json.loads(path.read_text()), short, pred_root=w.results/W.pins.PRED_ROOT_REL, registry=w.registry, registry_sha256=w.ctx.anchor['registry_sha256'], v4=w.ctx.v4,
                                 image_root=w.images)
        if change is not None:
            change(short, document)
        path.write_text(json.dumps(document))
        w.ctx.run_json[short] = hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = W.build(tmp_path, synthetic=True)
    to_hs_shape(w)
    monkeypatch.setattr(structure_layer, 'FORMAL_REGISTRIES', tuple(structure_layer.FORMAL_REGISTRIES)+(w.registry,))
    ctx = pp_r2._context_class()(repo=w.repo, source_root=w.repo, results=w.results, images=w.images, shorts=W.SHORTS, p3=w.ctx.p3, run_json=w.ctx.run_json, v4=w.ctx.v4, anchor=w.ctx.anchor,
                                  taper=W.frozen_formula_taper, pp_dir=w.results/'r2_pp_v1', runs_root=w.results/'p3_v1'/'runs')
    w.pp = ctx
    return w


def run_pp(world):
    reset_sealed_cache()
    return pp_r2.pp_unit(world.pp, gate=GATE, code=CODE, log=quiet)


def expected_pp(world, short):
    from evaluate_l2r_style import dice_l2r, disp_voxels, warp_labels_l2r
    f, m = W.pins.subjects_of(short)
    d = disp_voxels(world.fields[short], (1., 1., 1.))
    _, per = dice_l2r(world.labels[f], world.labels[m], warp_labels_l2r(world.labels[m], d))
    return per


def test_the_unit_end_to_end_reads_no_truth(world):
    assert run_pp(world) == 0
    root = world.pp.pq_root
    document = json.loads((root/pp_r2.PP_FILE).read_text())
    opened = json.loads((root/pp_r2.OPENED_FILE).read_text())
    assert document['passed'] is True and document['reads_truth'] is False and document['pairs'] == list(W.SHORTS)
    for short in W.SHORTS:
        row = json.loads((root/pp_r2.PP_DIR/f'{short}.json').read_text())
        np.testing.assert_array_equal(np.array(row['pp_per_label'], dtype=float), np.array(expected_pp(world, short), dtype=float))
        assert set(row['keys_read']) <= {'fixed', 'moving', 'initial_u'}
    assert opened['m6'] is True and not opened['breaches']
    assert all(set(m) <= {'fixed', 'moving', 'initial_u'} for m in opened['pair_members'].values()) and opened['pair_members']
    names = [Path(e['path']).name for e in opened['files']]
    assert not [n for n in names if n in ('repair_metrics.json', 'aligned_seg35.nii.gz', 'aligned_seg4.nii.gz') or n.startswith('metrics_') or n.startswith('decision')]
    assert {'run.json', 'field_0600.npz', 'repair.json', 'repair.npz', 'pair.npz', 'probs36.npz'} <= set(names)


def test_the_member_fixed_seg35_stops_the_unit(world, monkeypatch):
    plain = pp_r2.pp_of_labels
    stage = world.stage

    def peeking(L_F, L_M, d):
        with np.load(stage/W.pins.pair_name(W.SHORTS[0])/'pair.npz') as z:
            z['fixed_seg35']
        return plain(L_F, L_M, d)
    monkeypatch.setattr(pp_r2, 'pp_of_labels', peeking)
    assert run_pp(world) == 8
    document = json.loads((world.pp.pq_root/pp_r2.PP_FILE).read_text())
    opened = json.loads((world.pp.pq_root/pp_r2.OPENED_FILE).read_text())
    assert document['passed'] is False and document['stage'] == 'firewall' and opened['m6'] is False
    assert any(b.get('member') == 'fixed_seg35' for b in opened['breaches'])


@pytest.mark.parametrize('target', ['repair_metrics', 'metrics', 'label'])
def test_an_evaluation_record_or_a_label_file_stops_the_unit(world, monkeypatch, target):
    path = dict(repair_metrics=world.results/'p3_v1'/'runs'/'RC_c600'/W.SHORTS[0]/'repair_metrics.json', metrics=world.results/'p3_v1'/'runs'/'C_first5'/W.SHORTS[0]/'metrics_0600.json',
                label=world.images/W.SUBJECTS[0]/'aligned_seg35.nii.gz')[target]
    plain = pp_r2.pp_of_labels

    def peeking(L_F, L_M, d):
        path.read_bytes()
        return plain(L_F, L_M, d)
    monkeypatch.setattr(pp_r2, 'pp_of_labels', peeking)
    assert run_pp(world) == 8
    opened = json.loads((world.pp.pq_root/pp_r2.OPENED_FILE).read_text())
    assert opened['breaches'] and json.loads((world.pp.pq_root/pp_r2.PP_FILE).read_text())['stage'] == 'firewall'


@pytest.fixture
def untrained_world(tmp_path, monkeypatch):
    w = W.build(tmp_path, prediction_changes={W.SUBJECTS[1]: dict(trained_with_labels=False)}, synthetic=True)
    to_hs_shape(w)
    monkeypatch.setattr(structure_layer, 'FORMAL_REGISTRIES', tuple(structure_layer.FORMAL_REGISTRIES)+(w.registry,))
    w.pp = pp_r2._context_class()(repo=w.repo, source_root=w.repo, results=w.results, images=w.images, shorts=W.SHORTS, p3=w.ctx.p3, run_json=w.ctx.run_json, v4=w.ctx.v4, anchor=w.ctx.anchor,
                                   taper=W.frozen_formula_taper, pp_dir=w.results/'r2_pp_v1', runs_root=w.results/'p3_v1'/'runs')
    return w


def test_a_prediction_not_trained_with_labels_is_refused(untrained_world):
    """Review m3: PQ's check trained_with_labels true (the structure layer or this reader refuses it before any pp)."""
    assert run_pp(untrained_world) == 8
    document = json.loads((untrained_world.pp.pq_root/pp_r2.PP_FILE).read_text())
    assert document['passed'] is False and document['stage'] == 'm2' and json.loads((untrained_world.pp.pq_root/pp_r2.OPENED_FILE).read_text())['m6'] is True


def test_the_reader_checks_the_read_back_posterior():
    import inspect
    source = inspect.getsource(pp_r2.predicted_labels)
    assert 'read_back_probabilities' in source and 'argmax_smallest' in source and "trained_with_labels" in source


def test_the_unit_runs_once(world):
    assert run_pp(world) == 0
    before = sorted(p.name for p in world.pp.pq_root.iterdir())
    with pytest.raises(pp_r2.PPRefused, match='once'):
        run_pp(world)
    assert sorted(p.name for p in world.pp.pq_root.iterdir()) == before


def test_the_rules_allow_the_records_and_forbid_the_truth(world):
    rules = pp_r2.rules_for(world.pp)
    runs = world.results/'p3_v1'/'runs'
    for allowed in (runs/'C_first5'/W.SHORTS[0]/'run.json', runs/'RC_c600'/W.SHORTS[0]/'repair.npz', world.pp.pq_root/'pp'/'x.json'):
        assert rules.judge(allowed) is None
    for forbidden in (runs/'RC_c600'/W.SHORTS[0]/'repair_metrics.json', runs/'C_first5'/W.SHORTS[0]/'metrics_0600.json', world.images/W.SUBJECTS[0]/'aligned_seg35.nii.gz',
                      world.results/'hs_v1'/'analysis'/'decision_hs.json', world.results/'seg'/'uf_dev_read.lock'):
        assert rules.judge(forbidden) is not None


# ---- m1 on HS's run.json keys (pp's second start, 11-11 02:13:33 AM CST: PQRefused m1 "run.json segmenter None") ----------------------------------------
def test_the_world_has_hs_run_json_keys_not_p3s(world):
    for short, row in world.pp.p3.items():
        document = json.loads((world.pp.t88_dir(short)/'run.json').read_text())
        assert 'segmenter_id' not in document and document['schema'] == 'bayesreg.fl.run.v1'
        assert document['channel_sources']['segmenter']['segmenter_id'] == 'unet35_v4_ens' and set(document['channel_sources']['sides']) == {'fixed', 'moving'}
        v, record = pp_r2.bind_hs_field(world.pp, short, world.grid)
        assert record['state_hash'] == row.state_hash and record['labels_sha256'] == document['labels_sha256']


def _p3_shape(short, d):
    d.pop('channel_sources')
    d.update(segmenter_id='unet35_v4_ens', schema='bayesreg.p2.run.v1')


M1_CHANGES = {
    'p3_shape': (_p3_shape, 'channel_sources.segmenter.segmenter_id None'),
    'segmenter_id': (lambda s, d: d['channel_sources']['segmenter'].update(segmenter_id='unet35_v1_ens'), "segmenter.segmenter_id 'unet35_v1_ens'"),
    'registry_sha256': (lambda s, d: d['channel_sources']['structure_layer']['registry'].update(sha256='f'*64), 'structure_layer.registry: sha256'),
    'registry_status': (lambda s, d: d['channel_sources']['structure_layer']['registry'].update(status='staged'), "status 'staged'"),
    'registry_name': (lambda s, d: d['channel_sources']['segmenter'].update(registry='/code/configs/bayesreg/segmenters_v2.json'), 'segmenter.registry'),
    'pred_root': (lambda s, d: d['channel_sources']['segmenter'].update(pred_root='/elsewhere/unet35_v4_ens/pred'), 'segmenter.pred_root'),
    'side_probs36': (lambda s, d: d['channel_sources']['sides']['moving']['unet_output_sha256'].update({'probs36.npz': 'e'*64}), 'sides.moving: the sha256'),
    'side_policy': (lambda s, d: d['channel_sources']['sides']['fixed'].update(policy='fold_own'), "policy 'fold_own'"),
    'side_subject': (lambda s, d: d['channel_sources']['sides']['fixed'].update(subject='OASIS_OAS1_0450_MR1'), 'sides.fixed: subject'),
    'side_directory': (lambda s, d: d['channel_sources']['sides']['fixed'].update(unet_directory='/elsewhere'), 'sides.fixed.unet_directory'),
    'oracle': (lambda s, d: d['channel_sources'].update(oracle=True), 'oracle True'),
    'truth_read': (lambda s, d: d['channel_sources'].update(truth_labels_read=True), 'truth_labels_read True'),
    'schema': (lambda s, d: d.update(schema='bayesreg.p2.run.v1'), 'schema'),
}


@pytest.mark.parametrize('name', sorted(M1_CHANGES))
def test_m1_refuses_a_run_json_not_of_an_hs_v4_run(world, name):
    change, words = M1_CHANGES[name]
    to_hs_shape(world, change)
    import pq_core
    with pytest.raises(pq_core.PQRefused, match='m1') as caught:
        pp_r2.bind_hs_field(world.pp, W.SHORTS[0], world.grid)
    assert words in str(caught.value)


def test_m1_on_a_p3_run_json_stops_the_unit_before_any_label(world):
    to_hs_shape(world, M1_CHANGES['p3_shape'][0])
    assert run_pp(world) == 8
    document = json.loads((world.pp.pq_root/pp_r2.PP_FILE).read_text())
    opened = json.loads((world.pp.pq_root/pp_r2.OPENED_FILE).read_text())
    assert document['passed'] is False and 'PQRefused: m1:' in document['error'] and document['pairs'] == [] and opened['m6'] is True
    assert not [e for e in opened['files'] if Path(e['path']).name in ('probs36.npz', 'repair_metrics.json')]


def test_bind_hs_field_is_pq_io_bind_field_but_the_segmenter_statement():
    import ast
    import inspect
    import textwrap
    import pq_io as io_
    ours = ast.parse(textwrap.dedent(inspect.getsource(pp_r2.bind_hs_field))).body[0].body
    theirs = ast.parse(textwrap.dedent(inspect.getsource(io_.bind_field))).body[0].body
    assert isinstance(ours[0], ast.Expr) and isinstance(theirs[0], ast.Expr)                       # the docstrings
    prelude, ours, theirs = ours[1:4], ours[4:], theirs[1:]
    assert [type(s) for s in prelude] == [ast.Import, ast.Import, ast.Assign]
    assert ast.unparse(prelude[0]) == 'import hashlib' and ast.unparse(prelude[1]) == 'import pq_io as io_'
    names = [t.id for t in prelude[2].targets[0].elts]
    assert [ast.unparse(v) for v in prelude[2].value.elts] == [f'io_.{n}' for n in names]          # each name pq_io's own object
    assert all(hasattr(io_, n) for n in names) and pp_r2.np is io_.np and pp_r2.Path is io_.Path
    assert len(ours) == len(theirs)
    differ = [i for i, (a, b) in enumerate(zip(ours, theirs)) if ast.dump(a) != ast.dump(b)]
    assert len(differ) == 1
    assert ast.unparse(theirs[differ[0]]).startswith("need(run.get('segmenter_id') == ctx.anchor['segmenter_id']")
    assert ast.unparse(ours[differ[0]]) == 'problems.extend(hs_run_problems(run, ctx, short))'


def _dry_world(tmp_path, change=None):
    """One HS-shaped run.json under <tmp>/hs_v1/runs/SHx24_first5/0438-0439 with the real pins (pq_pins.ANCHOR, pq_pins.V4: constants, no data file) and the prediction root under tmp."""
    import pq_pins as pins
    short = rc.SHORTS[0]
    f, m = pins.subjects_of(short)
    base = dict(pair=pins.pair_name(short), status='done', oracle=False, field=dict(file='field_0600.npz', sha256='a'*64), labels_sha256=dict(fixed='b'*64, moving='c'*64))
    document = hs_shaped_run(base, short, pred_root=tmp_path/pins.PRED_ROOT_REL, registry='/results/hs_v1/code/'+pins.ANCHOR['registry'], registry_sha256=pins.ANCHOR['registry_sha256'], v4=pins.V4)
    if change is not None:
        change(short, document)
    path = tmp_path/'hs_v1'/'runs'/'SHx24_first5'/short/'run.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document))
    rows = {short: pins.P3Row('SHx24_first5', 'RSHx24_c600', repair_npz='0'*64, state_hash='0'*64, field_npz='a'*64, repair_json='0'*64)}
    return rows, {short: hashlib.sha256(path.read_bytes()).hexdigest()}


def test_the_dry_run_opens_one_run_json_and_checks_its_shape(tmp_path):
    rows, pins_ = _dry_world(tmp_path)
    lines = []
    assert pp_r2.run_json_shape(rows, pins_, tmp_path, log=lines.append) == [] and 'passes' in lines[0]


@pytest.mark.parametrize('name', ['p3_shape', 'segmenter_id', 'registry_sha256', 'side_probs36'])
def test_the_dry_run_shows_a_run_json_of_the_wrong_shape(tmp_path, name):
    rows, pins_ = _dry_world(tmp_path, M1_CHANGES[name][0])
    problems = pp_r2.run_json_shape(rows, pins_, tmp_path, log=quiet)
    assert problems and all(p.startswith('0438-0439 run.json: ') for p in problems)


def test_the_dry_run_refuses_a_run_json_that_is_not_the_reference_s(tmp_path):
    rows, pins_ = _dry_world(tmp_path)
    assert 'not the reference' in pp_r2.run_json_shape(rows, {k: '0'*64 for k in pins_}, tmp_path, log=quiet)[0]


# ---- the committed reference ---------------------------------------------------------------------------------------------------------------------------
def write_reference(repo, hs_root, *, change=None):
    pairs = {}
    for i, short in enumerate(rc.SHORTS):
        t88, rep = ('SHx24_first5', 'RSHx24_c600') if i < 5 else ('SHx24_rest600', 'RSHx24_rest600')
        pairs[short] = dict(field_dir=str(Path(hs_root)/'runs'/t88/short), repair_dir=str(Path(hs_root)/'runs'/rep/short), hs_line_status='hx_certified', certified_at_1=True,
                            **{k: hashlib.sha256(f'{i} {k}'.encode()).hexdigest() for k in ('attempt_sha256', 'repaired_u_sha256', 'field_sha256', 'repair_json_sha256', 'run_json_sha256', 'repair_metrics_sha256')})
    document = dict(schema='bayesreg.p2.reference.v1', segmenter_id='unet35_v4_ens', hs=dict(reading='采用新分割器', decision_sha256=rc.HS_DECISION_SHA256), p3=dict(pairs=pairs))
    if change:
        change(document)
    path = Path(repo)/rc.HS_REFERENCE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


def test_the_reference_rows(tmp_path):
    write_reference(tmp_path/'repo', tmp_path/'hs_v1')
    rows, run_json, metrics, record = pp_r2.reference_rows(tmp_path/'repo', pinned=None, hs_root=tmp_path/'hs_v1')
    assert sorted(rows) == sorted(rc.SHORTS) and rows['0438-0439'].t88_dir == 'SHx24_first5' and rows['0456-0457'].repair_dir == 'RSHx24_rest600'
    assert len(run_json) == len(metrics) == 19


@pytest.mark.parametrize('change', [
    lambda d: d['p3']['pairs']['0440-0441'].update(hs_line_status='hx_fallback_configC'),
    lambda d: d['p3']['pairs']['0440-0441'].update(field_dir='/elsewhere/SHx24_first5/0440-0441'),
    lambda d: d['p3']['pairs'].pop('0456-0457'),
    lambda d: d['hs'].update(reading='不采用'),
    lambda d: d.update(segmenter_id='unet35_v1_ens'),
    lambda d: d['p3']['pairs']['0441-0442'].update(attempt_sha256='abc'),
])
def test_the_reference_is_refused(tmp_path, change):
    write_reference(tmp_path/'repo', tmp_path/'hs_v1', change=change)
    with pytest.raises(pp_r2.PPRefused):
        pp_r2.reference_rows(tmp_path/'repo', pinned=None, hs_root=tmp_path/'hs_v1')


def test_the_reference_must_be_the_pinned_file(tmp_path):
    write_reference(tmp_path/'repo', tmp_path/'hs_v1')
    with pytest.raises(pp_r2.PPRefused, match='pinned'):
        pp_r2.reference_rows(tmp_path/'repo', hs_root=tmp_path/'hs_v1')


# ---- the split and the targets -----------------------------------------------------------------------------------------------------------------------
NAMES = pp_r2.label_names(ROOT)


def rows_for(share_of, contribution_of=None):
    """Synthetic decompose() rows with a given share and contribution_seg per label."""
    out = []
    for k in rc.LABELS:
        share = share_of(k)
        c = (contribution_of or (lambda k: .001))(k)
        out.append(dict(label=k, undefined=False, share_seg=share, contribution=c, contribution_seg=share*c))
    return out


def test_decompose_is_r3b():
    pp = [.9]*35
    hs = [.8]*35
    seg = [.95]*35
    rows = pp_r2.decompose(pp, hs, seg)
    r = rows[0]
    a, b = .1, np.sqrt(2)*.05
    assert r['a_not_aligned'] == pytest.approx(a) and r['b_wrong'] == pytest.approx(b) and r['share_seg'] == pytest.approx(b*b/(a*a+b*b)) and r['r3b_model_error'] == pytest.approx(np.hypot(a, b))
    assert r['contribution'] == pytest.approx(.2/35) and r['contribution_seg'] == pytest.approx(r['share_seg']*.2/35) and r['beyond_pp'] == pytest.approx(.1)
    assert pp_r2.decompose([None]+pp[1:], hs, seg)[0]['undefined'] is True


def test_the_structures_pair_left_and_right():
    structures = dict(pp_r2.structures(rc.OTHER_23, NAMES))
    assert structures['Hippocampus'] == (14, 30) and structures['3rd-Ventricle'] == (11,) and structures['Brain-Stem'] == (13,) and len(structures) == 13
    assert sum(len(v) for v in structures.values()) == 23


def test_the_rule_takes_the_small_six_and_the_top_three_eligible():
    weights = {14: .009, 30: .009, 16: .02, 32: .0, 11: .015, 7: .03, 26: .03}             # thalamus highest contribution but ineligible below
    rows = rows_for(lambda k: .2 if k in (7, 26) else .7, lambda k: weights.get(k, .001))
    targets, ranking = pp_r2.choose_targets(rows, NAMES)
    chosen = [r['structure'] for r in ranking if r['chosen']]
    assert chosen == ['Accumbens', 'Hippocampus', '3rd-Ventricle'] and targets == sorted(set(rc.SMALL_SIX) | {16, 32, 14, 30, 11})


def test_the_rule_with_nothing_eligible_is_the_small_six():
    targets, ranking = pp_r2.choose_targets(rows_for(lambda k: .3), NAMES)
    assert targets == sorted(rc.SMALL_SIX) and not any(r['chosen'] for r in ranking)


def test_the_share_bar_is_included_and_ties_go_to_the_smaller_label():
    rows = rows_for(lambda k: .5)
    targets, ranking = pp_r2.choose_targets(rows, NAMES)
    singles = [r for r in ranking if len(r['labels']) == 1]
    pairs = [r for r in ranking if r['chosen']]
    assert all(r['eligible'] for r in ranking) and [r['labels'][0] for r in pairs] == [3, 6, 7]       # pairs weigh twice the singles; among equal pairs the smaller label first
    assert singles and len(pairs) == 3


def test_an_undefined_label_makes_its_structure_ineligible():
    rows = rows_for(lambda k: .9, lambda k: .05 if k in (14, 30) else .001)
    rows[13] = dict(label=14, undefined=True)
    targets, ranking = pp_r2.choose_targets(rows, NAMES)
    assert 14 not in targets and 30 not in targets and next(r for r in ranking if r['structure'] == 'Hippocampus')['eligible'] is False


def split_world(tmp_path, *, passed=True):
    repo, hs_root, pp_root = tmp_path/'repo', tmp_path/'hs_v1', tmp_path/'pp'
    (repo/'configs'/'bayesreg').mkdir(parents=True)
    shutil.copy(ROOT/pp_r2.GROUPS_FILE_REL, repo/pp_r2.GROUPS_FILE_REL)
    rng = np.random.default_rng(0)
    metrics_pins = {}
    for i, short in enumerate(rc.SHORTS):
        rep = 'RSHx24_c600' if i < 5 else 'RSHx24_rest600'
        d = hs_root/'runs'/rep/short
        d.mkdir(parents=True)
        (d/'repair_metrics.json').write_text(json.dumps(dict(after=dict(metrics=dict(seg35_per_label={str(k): float(rng.uniform(.6, .95)) for k in rc.LABELS})))))
        metrics_pins[short] = rc.sha256_file(d/'repair_metrics.json')
    write_reference(repo, hs_root, change=lambda d: [d['p3']['pairs'][s].update(repair_metrics_sha256=metrics_pins[s]) for s in rc.SHORTS])
    uf = dict(rows=dict(UF4={s: dict(per_label=[float(v) for v in rng.uniform(.8, .97, 35)]) for s in rc.SUBJECTS}))
    (tmp_path/'eval_uf.json').write_text(json.dumps(uf))
    (pp_root/'pp').mkdir(parents=True)
    for short in rc.SHORTS:
        (pp_root/'pp'/f'{short}.json').write_text(json.dumps(dict(pp_per_label=[float(v) for v in rng.uniform(.85, .99, 35)])))
    (pp_root/pp_r2.PP_FILE).write_text(json.dumps(dict(passed=passed, reads_truth=False)))
    (pp_root/pp_r2.OPENED_FILE).write_text('{}')
    return dict(pp_root=pp_root, repo=repo, hs_root=hs_root, uf_eval=tmp_path/'eval_uf.json', uf_pinned=None, ref_pinned=None, log=quiet)


def test_split_end_to_end(tmp_path):
    kwargs = split_world(tmp_path)
    targets = pp_r2.split(**kwargs)
    document = json.loads((kwargs['pp_root']/pp_r2.SPLIT_FILE).read_text())
    written = json.loads((kwargs['pp_root']/pp_r2.TARGETS_FILE).read_text())
    assert written['targets'] == targets and set(rc.SMALL_SIX) <= set(targets) and len(targets) <= 12 and targets == sorted(targets)
    assert written['split_sha256'] == rc.sha256_file(kwargs['pp_root']/pp_r2.SPLIT_FILE) and len(document['rows']) == 35
    assert (kwargs['pp_root']/pp_r2.REPORT_FILE).read_text().startswith('# R2 的 pp 诊断')
    with pytest.raises(pp_r2.PPRefused, match='once'):
        pp_r2.split(**kwargs)


def test_split_refuses_a_pp_run_that_did_not_pass(tmp_path):
    with pytest.raises(pp_r2.PPRefused):
        pp_r2.split(**split_world(tmp_path, passed=False))


def test_split_refuses_an_hs_record_that_is_not_the_pinned_one(tmp_path):
    kwargs = split_world(tmp_path)
    path = kwargs['hs_root']/'runs'/'RSHx24_c600'/'0438-0439'/'repair_metrics.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(pp_r2.PPRefused, match='repair_metrics'):
        pp_r2.split(**kwargs)


def test_the_start_string_and_the_unit_check():
    argv = pp_r2.start_string()
    assert f'--unit={pp_r2.UNIT}' in argv and 'pytest' not in ' '.join(argv) and argv[-1] == 'pp' and any(a.startswith('--setenv=PYTHONPATH=/home/rmcfz/seq_vp_tag_results/frozen_eb808be:') for a in argv)
    assert pp_r2.in_own_unit('/nonexistent') is False


def test_the_machine_gate(tmp_path):
    from types import SimpleNamespace
    meminfo = tmp_path/'meminfo'
    meminfo.write_text('MemAvailable: 10485760 kB\n')

    def run(argv, **kwargs):
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    big = lambda path: SimpleNamespace(free=500e9)       # noqa: E731
    assert pp_r2.machine_gate(run=run, meminfo=meminfo, disk=big)['passed'] is True
    with pytest.raises(pp_r2.PPRefused, match='C:'):
        pp_r2.machine_gate(run=run, meminfo=meminfo, disk=lambda path: SimpleNamespace(free=500e9 if path == '/home' else 30*2**30))
    with pytest.raises(pp_r2.PPRefused, match='other units'):
        pp_r2.machine_gate(run=lambda argv, **k: SimpleNamespace(returncode=0, stdout='tag-uf-chain.service loaded active running x\n', stderr=''), meminfo=meminfo, disk=big)
