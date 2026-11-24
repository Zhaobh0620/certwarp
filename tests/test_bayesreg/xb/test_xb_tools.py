"""XB's pieces: the method scripts are the archive's bytes, the frozen configuration and its check, the environment check, the export check, ConvexAdam-v4seg's maps and coordinate convention
(prereg §2.3, §2.4; C22.1-4).  Synthetic data; the real files read are the repository's archive and configs only (and, read-only, convexAdam's apply_convex source when it imports)."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from . import _world as W

hc, xc = W.hc, W.xc
sys.path.insert(0, str(W.ROOT/'experiments'/'xb'))
import build_frozen_config  # noqa: E402
import v4seg_convention as conv  # noqa: E402


# ---- the method scripts and the frozen configuration --------------------------------------------------------------------------------------------
def test_the_method_scripts_are_the_archives_bytes():
    for name in ('m_fireants.py', 'm_convexadam.py', 'm_unigradicon.py', 'm_voxelmorph.py'):
        copy, archived = W.ROOT/'experiments/xb/harness'/name, W.ROOT/xc.ARCHIVE_HARNESS/name
        assert copy.read_bytes() == archived.read_bytes()
        assert xc.sha256(copy) == build_frozen_config.HARNESS_COPIES[name]
    assert xc.sha256(W.ROOT/xc.ARCHIVE_HARNESS/'drive.py') == xc.ARCHIVE_DRIVE_SHA256
    assert xc.sha256(W.ROOT/xc.BL2_PROTOCOL) == xc.BL2_PROTOCOL_SHA256


def test_the_frozen_configuration_is_pinned_and_rebuilds_to_the_same_bytes(tmp_path):
    config, pin = xc.frozen_config()
    assert pin['sha256'] == xc.FROZEN_CONFIG_SHA256
    out = tmp_path/'again.json'
    assert build_frozen_config.main(['--out', str(out)]) == 0
    assert out.read_bytes() == (W.ROOT/xc.FROZEN_CONFIG_REL).read_bytes()
    assert build_frozen_config.main(['--out', str(out)]) == 2                        # only grows
    for arm, cfg in xc.ARMS.items():
        assert {k: config['arms'][arm][k] for k in xc.EXECUTABLE_KEYS} == {k: cfg.get(k) for k in xc.EXECUTABLE_KEYS}
    assert {a: config['arms'][a]['factor']['value'] for a in ('fireants_greedy_eq', 'fireants_syn_eq', 'unigradicon_io_eq')} == dict(fireants_greedy_eq=22, fireants_syn_eq=27, unigradicon_io_eq=123)
    assert set(config['env_records']['sha256']) == {str(p.relative_to(W.ROOT/xc.ENV_RECORDS_REL)) for p in (W.ROOT/xc.ENV_RECORDS_REL).glob('*/*') if p.is_file()}
    assert {a['role'] for a in config['arms'].values()} == {'main', 'descriptive', 'tier90'}
    assert sorted(a for a, c in config['arms'].items() if c['role'] == 'main') == ['ants_s', 'convexadam_v4seg', 'fireants_greedy', 'fireants_syn']     # + configuration C = the 5 of the main family
    assert config['baseline_files']['convexadam']['convexadam/src/src/convexAdam/convex_adam_nnUNet.py'] == xc.V4SEG_NNUNET_SHA256


def test_check_frozen_on_synthetic_baseline_files(tmp_path):
    config, _ = xc.frozen_config()
    base = tmp_path/'baselines'
    (base/'fireants').mkdir(parents=True)
    (base/'fireants/a.py').write_text('a')
    config = json.loads(json.dumps(config))
    config['baseline_files']['fireants'] = {'fireants/a.py': xc.sha256(base/'fireants/a.py')}
    assert xc.check_frozen('fireants_greedy', baselines=base, config=config)['passed']
    (base/'fireants/a.py').write_text('b')
    with pytest.raises(hc.Refused) as e:
        xc.check_frozen('fireants_greedy', baselines=base, config=config)
    assert e.value.stage == 'frozen'
    (base/'fireants/a.py').write_text('a')
    config['arms']['fireants_greedy']['post_args'] = ['2']
    with pytest.raises(hc.Refused):
        xc.check_frozen('fireants_greedy', baselines=base, config=config)


@pytest.mark.parametrize('arm, record', [('ants_s', 'ants'), ('fireants_syn', 'bl_fireants'), ('convexadam_v4seg', 'convexadam'), ('niftyreg', 'niftyreg'), ('deeds', 'deedsBCV'),
                                         ('unigradicon_io', 'bl_unigradicon'), ('voxelmorph', 'voxelmorph')])
def test_check_frozen_checks_the_archives_smoke_cmd(tmp_path, monkeypatch, arm, record):
    """m11 (prereg §5.1 item 9): the archive's smoke_cmd.md of each environment record the arm uses is pinned (SMOKE_CMD_SHA256) and checked; another sha256 refuses."""
    config, _ = xc.frozen_config()
    config = json.loads(json.dumps(config))
    base = tmp_path/'baselines'
    for group in xc.ARMS[arm].get('baselines', []):
        (base/group).mkdir(parents=True)
        (base/group/'a.py').write_text(group)
        config['baseline_files'][group] = {f'{group}/a.py': xc.sha256(base/group/'a.py')}
    rec = xc.check_frozen(arm, baselines=base, config=config)
    smoke = [r for r in rec['files_checked'] if r['what'] == 'archive smoke_cmd.md']
    assert [Path(r['file']).parent.name for r in smoke] == [record] and smoke[0]['sha256'] == xc.SMOKE_CMD_SHA256[record]
    monkeypatch.setitem(xc.SMOKE_CMD_SHA256, record, '0'*64)
    with pytest.raises(hc.Refused) as e:
        xc.check_frozen(arm, baselines=base, config=config)
    assert 'smoke_cmd.md' in str(e.value)


def env_runner(conda_text, pip_text, head='f8a9cc259ab9113fc43bd99797000fa8fc34eb59', status=''):
    def run(argv, **kwargs):
        argv = [str(a) for a in argv]
        if argv[1:3] == ['list', '-p']:
            return subprocess.CompletedProcess(argv, 0, conda_text, '')
        if argv[-2:] == ['pip', 'freeze']:
            return subprocess.CompletedProcess(argv, 0, pip_text, '')
        if 'rev-parse' in argv:
            return subprocess.CompletedProcess(argv, 0, head+'\n', '')
        if 'status' in argv:
            return subprocess.CompletedProcess(argv, 0, status, '')
        raise AssertionError(argv)
    return run


def test_the_environment_check_compares_the_records_line_for_line(tmp_path):
    config, _ = xc.frozen_config()
    config = json.loads(json.dumps(config))
    rec = W.ROOT/xc.ENV_RECORDS_REL/'bl_fireants'
    conda_text, pip_text = (rec/'conda_list.txt').read_text(), (rec/'pip_freeze.txt').read_text()
    envs = tmp_path/'envs'
    so = envs/'bl_fireants'/'lib/python3.10/site-packages/fireants_fused_ops.cpython-310-x86_64-linux-gnu.so'
    so.parent.mkdir(parents=True)
    so.write_bytes(b'so')
    config['environments']['bl_fireants']['files'] = {str(so.relative_to(envs/'bl_fireants')): xc.sha256(so)}
    header_changed = conda_text.replace('# packages in environment at /home/rmcfz/miniconda3/envs/bl_fireants:', '# packages in environment at /elsewhere:')
    record = xc.check_environment('fireants_greedy', config=config, run=env_runner(header_changed, pip_text), envs=envs, baselines=tmp_path)
    assert record['passed'] and record['conda_list']['same'] and record['pip_freeze']['same']
    with pytest.raises(hc.Refused) as e:
        xc.check_environment('fireants_greedy', config=config, run=env_runner(conda_text, pip_text.replace('torch==2.5.1+cu124', 'torch==2.6.0')), envs=envs, baselines=tmp_path)
    assert e.value.stage == 'environment'
    with pytest.raises(hc.Refused):
        xc.check_environment('fireants_greedy', config=config, run=env_runner(conda_text, pip_text, status=' M x.py'), envs=envs, baselines=tmp_path)
    so.write_bytes(b'changed')
    with pytest.raises(hc.Refused):
        xc.check_environment('fireants_greedy', config=config, run=env_runner(conda_text, pip_text), envs=envs, baselines=tmp_path)


# ---- the export check ---------------------------------------------------------------------------------------------------------------------------
def test_the_export_check_and_a_field_that_is_not_complete(tmp_path):
    fixed, moving = W.images()
    d = np.stack([np.zeros(W.SHAPE), np.zeros(W.SHAPE), np.full(W.SHAPE, 1.)])
    rec = xc.export_check(fixed, moving, d, W.warp(moving, d), {})
    assert rec['check']['corr'] > .9999 and not rec['suspicious'] and rec['threshold'] == .999
    rec = xc.export_check(fixed, moving, d, np.random.default_rng(0).standard_normal(W.SHAPE), {})
    assert rec['suspicious']
    with pytest.raises(xc.MethodFailure):
        xc.finish_field(np.full((3,)+W.SHAPE, np.nan), W.SHAPE, tmp_path)
    with pytest.raises(xc.MethodFailure):
        xc.finish_field(np.zeros((3, 2, 2, 2)), W.SHAPE, tmp_path)


# ---- ConvexAdam-v4seg -----------------------------------------------------------------------------------------------------------------------------
def test_the_argmax_rule_and_the_raw_sum():
    p = np.zeros((36, 2, 2, 2), np.float16)
    p[3] = .5
    p[7] = .5                                       # a tie: the smaller label
    labels, worst = xc.argmax_labels(p)
    assert (labels == 3).all() and labels.dtype == np.int16 and worst == 0.
    p[3] = .6
    with pytest.raises(hc.InfrastructureFailure):
        xc.argmax_labels(p)                         # raw sum 1.1 > 1 + 1e-3
    from bayesreg.seg.channels import RAW_SUM_TOL
    assert xc.RAW_SUM_TOL_FLOAT16 == RAW_SUM_TOL['float16']


def test_the_argmax_rule_is_the_method_channels_one():
    """mc0p_core.predicted_labels on the read-back P (C22.1-4) and fast_channels' bit-pattern reader give the same map as argmax_labels on a random float16 stack."""
    sys.path.insert(0, str(W.ROOT/'experiments'/'mc0'))
    import mc0p_core
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(36), size=(6, 5, 4)).transpose(3, 0, 1, 2).astype(np.float16)
    P = p.astype(np.float64)
    labels, _ = xc.argmax_labels(p)
    assert np.array_equal(labels, mc0p_core.predicted_labels(P/P.sum(0)))


def test_a_prediction_is_checked_against_the_anchor(tmp_path):
    fixed = 'OASIS_OAS1_0438_MR1'
    W.write_prediction(tmp_path, fixed, np.zeros(W.SHAPE, int))
    assert xc.check_prediction(tmp_path, fixed, shape=W.SHAPE)['probs36_sha256']
    for key, value in (('segmenter_id', 'unet35_v1_ens'), ('policy', 'fold'), ('registry_sha256', '0'*64), ('test_labels_seen', True), ('subject', 'OASIS_OAS1_0439_MR1')):
        d = tmp_path/key
        W.write_prediction(d, fixed, np.zeros(W.SHAPE, int), override={key: value})
        with pytest.raises(hc.Refused):
            xc.check_prediction(d, fixed, shape=W.SHAPE)
    (tmp_path/fixed/'probs36.npz').write_bytes(b'changed')
    with pytest.raises(hc.Refused):
        xc.check_prediction(tmp_path, fixed, shape=W.SHAPE)


def test_a_hold_out_prediction_counts_only_with_his_subject_record(tmp_path):
    """A5a review I2 point 3: HI's first run and its resume write into one root; a subject is written when <root>/subjects/<s>.json names it with the probs36.npz (and provenance.json) that
    check_prediction hashed.  A prediction without a record (a killed run's residue) is refused, as is a record of another subject or of other bytes."""
    fixed = 'OASIS_OAS1_0438_MR1'
    pred = tmp_path/'root'/'pred'
    W.write_prediction(pred, fixed, np.zeros(W.SHAPE, int))
    checked = xc.check_prediction(pred, fixed, shape=W.SHAPE)
    with pytest.raises(hc.Refused, match='HI has not written this subject') as refused:
        xc.check_hi_record(pred, fixed, checked['probs36_sha256'], checked['provenance_sha256'])
    assert refused.value.stage == 'provenance'
    record = xc.hi_record_path(pred, fixed)
    assert record == tmp_path/'root'/'subjects'/f'{fixed}.json'
    record.parent.mkdir()
    row = dict(subject=fixed, output_sha256={'probs36.npz': checked['probs36_sha256'], 'structure_groups.npz': '0'*64}, provenance_sha256=checked['provenance_sha256'], run=1)
    record.write_text(json.dumps(row))
    assert xc.check_hi_record(pred, fixed, checked['probs36_sha256'], checked['provenance_sha256']) == dict(record=str(record), record_sha256=xc.sha256(record), run=1)
    for change in (dict(subject='OASIS_OAS1_0439_MR1'), dict(output_sha256={'probs36.npz': '0'*64}), dict(provenance_sha256='0'*64)):
        record.write_text(json.dumps(dict(row, **change)))
        with pytest.raises(hc.Refused) as refused:
            xc.check_hi_record(pred, fixed, checked['probs36_sha256'], checked['provenance_sha256'])
        assert refused.value.stage == 'provenance'
    record.write_text('{')
    with pytest.raises(hc.Refused, match='HI has not written this subject'):
        xc.check_hi_record(pred, fixed, checked['probs36_sha256'])


def test_every_arm_stores_on_d_and_the_hold_out_predictions_are_his_one_root():
    """A5a review m7: decision_log 2026-11-10 ~1:18 AM CST (every arm's kept big files on D:); xb_core.HI_PRED_ROOTS['holdout'] is HI's one root's pred/ (hi_core.HOLDOUT_OUT_SUGGESTED, the
    first run and its resume), as imported."""
    assert {arm: cfg['store'] for arm, cfg in xc.ARMS.items()} == dict.fromkeys(xc.ARMS, 'd')
    assert W.REAL_HI_PRED_ROOTS['holdout'] == hc.hi.HOLDOUT_OUT_SUGGESTED/'pred'
    assert W.REAL_HI_PRED_ROOTS['dev'] == hc.hi.REGISTERED_PRED


def test_the_hold_out_driver_checks_his_subject_records():
    """A5a review I2 point 3: the hold-out ConvexAdam-v4seg cell checks HI's subject record of both subjects (xb_core.check_hi_record, after check_prediction) and names the two records one by one
    to the spy; the development cells (HS's registered predictions) do not."""
    import inspect
    source = inspect.getsource(W.xb_drive)
    assert "extra_exact += [xc.hi_record_path(pred_root, s) for s in (fixed, moving)]" in source
    assert "checks['hi_records'] = {side: xc.check_hi_record(pred_root, c['subject'], c['probs36_sha256'], c['provenance_sha256']) for side, c in checks['predictions'].items()}" in source
    assert source.count("pair_set == 'holdout'") >= 2


def known_fields(shape):
    z, y, x = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing='ij')
    translation = np.stack([np.full(shape, .7), np.full(shape, -1.3), np.full(shape, 2.1)])
    A = np.array([[.03, -.02, .01], [.02, .04, -.03], [-.01, .02, .05]])
    p = np.stack([z, y, x])
    affine = np.tensordot(A, p-np.array([5., 6., 4.]).reshape(3, 1, 1, 1), axes=1)+np.array([.4, -.6, .9]).reshape(3, 1, 1, 1)
    return dict(translation=translation, affine=affine)


def test_the_layout_is_m_convexadams_transpose():
    """m12: the layout is ours.transpose(2, 1, 0), as m_convexadam.py:22-23 (pinned on an array whose three axes differ), and from_layout undoes it."""
    a = np.arange(2*3*4).reshape(2, 3, 4)
    assert np.array_equal(conv.to_layout(a), a.transpose(2, 1, 0)) and conv.to_layout(a).shape == (4, 3, 2)
    assert np.array_equal(conv.from_layout(conv.to_layout(a)), a)
    text = (W.ROOT/'experiments/xb/harness/m_convexadam.py').read_text()
    assert '.transpose(2, 1, 0)' in text


@pytest.mark.parametrize('kind', ['translation', 'affine'])
def test_the_coordinate_convention_on_known_fields(kind):
    """The layout's disp of a known d (built from apply_convex's semantics) comes back as d; the export check passes on it and catches a wrong axis order."""
    fixed, moving = W.images()
    d = known_fields(W.SHAPE)[kind]
    disp = conv.disp_from_d(d)
    assert disp.shape == conv.to_layout(fixed).shape+(3,)
    assert np.array_equal(conv.d_from_disp(disp), d)
    own = conv.from_layout(conv.warp_layout(conv.to_layout(moving), disp))
    assert np.allclose(own, W.warp(moving, d))
    assert xc.export_check(fixed, moving, d, own, {})['check']['corr'] > .9999
    wrong = np.ascontiguousarray(disp.transpose(3, 2, 1, 0))                         # a deliberately wrong axis order: the channels not reversed (d[0] would hold the x component)
    assert xc.export_check(fixed, moving, wrong, own, {})['suspicious']
    assert not np.allclose(wrong, d)


def test_the_convention_matches_convexadams_own_apply_convex():
    """apply_convex from the pinned source (read-only import; skipped when it cannot import here): our warp of the layout is its warp."""
    src = Path('/home/rmcfz/opt/baselines/convexadam/src/src')
    if not (src/'convexAdam'/'apply_convex.py').is_file():
        pytest.skip('convexAdam source not on this machine')
    sys.path.insert(0, str(src))
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True                  # m12: never a __pycache__ in the baselines tree, with or without -B
    try:
        from convexAdam.apply_convex import apply_convex
    except Exception as exc:                        # noqa: BLE001
        pytest.skip(f'apply_convex does not import in sci310: {exc}')
    finally:
        sys.path.remove(str(src))
        sys.dont_write_bytecode = bytecode
    assert not (src/'convexAdam'/'__pycache__').exists()
    assert xc.sha256(src/'convexAdam'/'apply_convex.py') == '5b119a0b18690a2d3befc7eb74955c23ac1402ab4923d5b6e1d189e37175eaf2'
    fixed, moving = W.images()
    d = known_fields(W.SHAPE)['affine']
    disp = conv.disp_from_d(d)
    theirs = conv.from_layout(apply_convex(disp=disp.astype(np.float32), moving=conv.to_layout(moving).astype(np.float32)))
    assert xc.export_check(fixed, moving, d, theirs, {})['check']['corr'] > .9999
    assert not xc.export_check(fixed, moving, np.ascontiguousarray(disp.transpose(3, 2, 1, 0)), theirs, {})['check']['corr'] >= .999


def test_the_maps_file_and_the_methods_labels(tmp_path):
    rng = np.random.default_rng(2)
    subjects = ('OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0439_MR1')
    labels = {s: rng.integers(0, 36, W.SHAPE) for s in subjects}
    checked = {}
    for side, s in zip(('fixed', 'moving'), subjects):
        W.write_prediction(tmp_path, s, labels[s])
        checked[side] = xc.check_prediction(tmp_path, s, shape=W.SHAPE)
    run = tmp_path/'run.json'
    digest = xc.labels_digest({'fixed': labels[subjects[0]].astype(np.int16), 'moving': labels[subjects[1]].astype(np.int16)})
    run.write_text(json.dumps(dict(labels_sha256=digest)))
    rec = xc.write_maps(tmp_path, *subjects, tmp_path/'maps.npz', shape=W.SHAPE, checked=checked, method_run=run)
    assert rec['method_labels_check']['equal'] and rec['labels_sha256'] == digest
    run.write_text(json.dumps(dict(labels_sha256=dict(fixed='0'*64, moving='0'*64))))
    with pytest.raises(hc.InfrastructureFailure):
        xc.write_maps(tmp_path, *subjects, tmp_path/'maps2.npz', shape=W.SHAPE, checked=checked, method_run=run)
