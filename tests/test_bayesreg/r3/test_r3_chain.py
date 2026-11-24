"""R3a's chains (experiments/r3/run_r3.py, compose_r3.py, eval_r3.py, decide_r3.py, r3_lock.py, smoke_r3.py).  CPU; the synthetic world of r3_synthetic (synthetic_tree's subjects, R2's tiny UF
members, tiny L members trained by the chain, the stand-in for nnU-Net) in tmp_path; no real label, image, prediction or splits/ file is read, no nnunetv2 runs.

- the training chain: its steps and their order (control, L A and B with their checks and calibrations, check-members, four nnU-Net models with their checks, the two calibration sets, the
  composition), L's --set values, the splits pin in every nnU-Net step; the hold file; a second pass skips everything; the composition's candidates are the closed list;
- the development chain: dev_plan pins composition.json, stops on no change; nn_infer_dev only with N; the evaluation's arms, the control 0, the reading 不可判 for 2 subjects;
- the start strings (units tag-r3-*, MemoryMax, RuntimeMaxSec, the snapshot), the locks (fixed paths, O_EXCL, BZ's words), the composition refused without the switch before any truth;
- the smoke's rule for L;
- the revision round: BZ's gate (b) and its bars; a real reading refuses a synthetic or forced composition and one whose winner the rule does not give; prepare stops on a drift of L's config.
"""
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'r3', ROOT/'experiments'/'r2', ROOT/'experiments'/'nn', ROOT/'experiments'/'fl', ROOT/'experiments'/'ef', ROOT/'experiments'/'mc0', ROOT/'experiments'/'unv2', ROOT/'experiments'/'uf', ROOT/'experiments'/'ul'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import r3_common as r3c  # noqa: E402
import r3_lock  # noqa: E402
import r3_synthetic  # noqa: E402
import run_r3  # noqa: E402
import smoke_r3  # noqa: E402


def quiet(*args, **kwargs):
    pass


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    """Both synthetic chains once for the module (about half a minute on the CPU)."""
    work = tmp_path_factory.mktemp('r3world')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    problems = r3_synthetic.synthetic_chain(work, env, log=quiet)
    return dict(work=work, problems=problems, out=work/'out')


def test_both_synthetic_chains_run_as_planned(world):
    assert world['problems'] == []
    out = world['out']
    composition = json.loads((out/'compose'/'composition.json').read_text())
    assert composition['order'] == list(r3c.candidate_order(True)) and composition['synthetic'] and composition['calib_read'] is None
    assert sorted(composition['subjects']) == ['A', 'B'] and all(len(v) == 1 for v in composition['subjects'].values())
    assert set(composition['rows']) == set(r3c.candidate_order(True)) | {'UF_s0', 'UF_s1', 'L', 'N', 'nn_s0', 'nn_s1'}
    assert composition['winner'] == r3c.choose({c: composition['means'][c] for c in composition['order']}, composition['order'])['winner']
    assert all(c.endswith('+D') for c in composition['decode_records'])
    plan = json.loads((out/'dev_plan.json').read_text())
    assert plan['winner'] == r3_synthetic.FORCED and plan['delivered'] == r3c.delivered(r3_synthetic.FORCED)
    evaluation = json.loads((out/'eval_r3'/'eval_r3.json').read_text())
    assert evaluation['ensemble_arms']['COMBO'] == r3c.delivered(r3_synthetic.FORCED)['unet'] and evaluation['ensemble_arms']['UF4'] == [f'uf/{n}' for n in r3c.UF_ORDER]
    assert evaluation['control']['differences'] == {'seg35': 0., 'seg4': 0.} and evaluation['dev_read'] is None
    ends = sorted(p.name for p in out.glob('queue_end_*.json'))
    assert 'queue_end_train.json' in ends and 'queue_end_dev.json' in ends


def test_the_l_members_are_checked_against_uf(world):
    out = world['out']
    for name in r3c.L_NAMES:
        check = json.loads((out/'r3l'/name/'r3_check.json').read_text())
        record = json.loads((out/'r3l'/name/'train_record.json').read_text())
        uf = json.loads((world['work']/'uf'/f'fold{name[4]}_s0'/'train_record.json').read_text())
        assert check['problems'] == [] and record['recipe'] == 'v3_large'
        assert {k for k in record['config'] if record['config'][k] != uf['config'][k]} == set(r3c.UF_CHANGES)
        assert record['calib_subjects'] == uf['calib_subjects'] and record['train_subjects'] == uf['train_subjects']


def test_the_steps_of_the_training_chain():
    steps = run_r3.chain_steps(run_r3.Spec(out=Path('/o'), cwd=Path('/s')), 'S'*64)
    names = [n for n, _, _ in steps]
    assert names == ['control_check', 'train_r3l_foldA_s3', 'check_r3l_foldA_s3', 'calibrate_r3l_foldA_s3', 'train_r3l_foldB_s3', 'check_r3l_foldB_s3', 'calibrate_r3l_foldB_s3', 'check_members_r3l',
                     'nn_train_nnA_s0', 'nn_check_nnA_s0', 'nn_train_nnA_s1', 'nn_check_nnA_s1', 'nn_train_nnB_s2', 'nn_check_nnB_s2', 'nn_train_nnB_s3', 'nn_check_nnB_s3',
                     'nn_infer_calibA', 'nn_infer_calibB', 'compose']
    by = {n: a for n, _, a in steps}
    train = by['train_r3l_foldA_s3']
    assert train[train.index('--profile')+1] == 'v3_large' and train[train.index('--seed')+1] == '3' and train[train.index('--out')+1] == '/o/r3l'
    sets = dict(a.split('=', 1) for a in train[train.index('--set'):] if a != '--set')
    assert {k: json.loads(v) for k, v in sets.items()} == r3c.l_train_config()
    assert all(by[n][by[n].index('--splits-sha256')+1] == 'S'*64 for n in by if n.startswith(('nn_train', 'nn_infer', 'compose')))
    assert by['control_check'][by['control_check'].index('--out-dir')-1] == '--control-only'
    assert not any('r2s' in str(a) or 'r2_small' in str(a) for _, _, argv in steps if argv for a in argv)


def test_the_development_chain_steps():
    spec = run_r3.Spec(out=Path('/o'), cwd=Path('/s'))
    assert [n for n, _, _ in run_r3.dev_steps(spec, 'C', 'S', True)] == ['nn_infer_dev', 'eval_r3', 'decide_r3']
    steps = run_r3.dev_steps(spec, 'C', 'S', False)
    assert [n for n, _, _ in steps] == ['eval_r3', 'decide_r3']
    argv = steps[0][2]
    assert argv[argv.index('--composition')+1] == '/o/compose/composition.json' and argv[argv.index('--composition-sha256')+1] == 'C'


def test_dev_plan_stops_on_no_change(world, tmp_path):
    source = json.loads((world['out']/'compose'/'composition.json').read_text())
    path = tmp_path/'composition.json'
    path.write_text(json.dumps(dict(source, winner='UF', delivered=r3c.delivered('UF'), no_change=True)))
    spec = run_r3.Spec(out=tmp_path/'o', cwd=Path('/s'), composition_path=path)
    with pytest.raises(run_r3.Stop, match='no development reading'):
        run_r3.dev_plan(spec, dict(nn=dict(splits_sha256='S')))
    path.write_text(json.dumps(dict(source, winner='UF+N', delivered=r3c.delivered('UF+L'), no_change=False)))
    with pytest.raises(run_r3.Stop, match='not r3_common.delivered'):
        run_r3.dev_plan(spec, dict(nn=dict(splits_sha256='S')))
    assert not (tmp_path/'o'/'dev_plan.json').exists()


def test_the_start_strings():
    for dev in (False, True):
        argv = run_r3.launch_argv(dev=dev)
        text = ' '.join(argv)
        assert f"--unit={'tag-r3-dev' if dev else 'tag-r3-chain'}" in argv and '--property=MemoryMax=12G' in argv and 'pytest' not in text and 'seq-vp-' not in text
        assert argv[-6:-4] == [str(r3c.OUT/'snapshot'/'experiments/r3/run_r3.py'), 'dev-chain' if dev else 'chain']
    assert run_r3.runtime_max_seconds() > 2*int(r3c.L_WALL_SECONDS)+4*r3c.NN_WALL_SECONDS
    assert '--unit=tag-r3-smoke' in smoke_r3.launch_argv()
    assert run_r3.snapshot_path_problems() == [] and 'experiments/r3' in run_r3.SNAPSHOT_PATHS_R3 and r3c.PLAN_REL in run_r3.SNAPSHOT_PATHS_R3


def test_the_locks(tmp_path, monkeypatch):
    monkeypatch.setitem(r3_lock.PATHS, 'calib', tmp_path/'calib.lock')
    r3_lock.check('calib')
    line = r3_lock.take('calib', dict(out_dir='x'))
    assert line['event'] == 'first_read'
    with pytest.raises(r3_lock.ReadLocked):
        r3_lock.check('calib')
    with pytest.raises(r3_lock.ReadLocked):
        r3_lock.take('calib', dict(out_dir='y'))
    with pytest.raises(ValueError):
        r3_lock.check('calib', '  ')
    r3_lock.check('calib', 'BZ: ok, 11-12 9:00 AM')
    r3_lock.take('calib', dict(out_dir='z'), bz_go='BZ: ok, 11-12 9:00 AM')
    assert [d['event'] for d in r3_lock.describe('calib')] == ['first_read', 'bz_approved_reread']
    assert r3_lock.path_of('dev') == r3c.DEV_READ_LOCK


def test_the_composition_needs_the_switch_before_any_truth(world, tmp_path):
    import compose_r3
    import r3_core
    work = world['work']
    with pytest.raises(r3_core.Refused, match='switch'):
        compose_r3.compose(out_dir=tmp_path/'c', uf_root=work/'uf', member_root=work/'out', nn_calib=work/'out'/'nn_calib', nn_root=work/'nnroot', switch_path=tmp_path/'none.json',
                           folds_path=work/'folds_full_v2.json', allowlist_path=work/'dev_subjects.json', data_root=work/'data', uf_eval=work/'eval_uf.json', synthetic=True,
                           l_overrides=r3_synthetic.tiny_l(), device='cpu', log=quiet)
    assert not (tmp_path/'c').exists()


def test_the_composition_refuses_real_paths_without_pins(tmp_path, monkeypatch):
    import compose_r3
    import r3_core
    monkeypatch.setitem(r3_lock.PATHS, 'calib', tmp_path/'calib.lock')
    monkeypatch.setattr(compose_r3, 'require_clean_code', lambda code: code)
    with pytest.raises(r3_core.Refused, match='plan'):
        compose_r3.compose(out_dir=tmp_path/'c', uf_root=tmp_path, device='cpu')
    with pytest.raises(r3_core.Refused, match='splits-sha256'):
        compose_r3.compose(out_dir=tmp_path/'c', device='cpu')
    assert not (tmp_path/'calib.lock').exists()


def test_the_l_smoke_rule():
    good = dict(recipe='v3_large', amp=True, finite=True, stopped='max_iters', iterations_run=400, projected_wall_per_step=.25, gpu_peak_allocated_gib=9.8)
    assert smoke_r3.l_reading(good) == []
    for change in (dict(recipe='r2_small'), dict(amp=False), dict(finite=False), dict(iterations_run=399), dict(projected_wall_per_step=.31), dict(gpu_peak_allocated_gib=None)):
        assert smoke_r3.l_reading(dict(good, **change)), change
    config = smoke_r3.smoke_config()
    assert config['max_iters'] == config['min_iters'] == config['patience'] == 400 and config['checkpoint'] == 'last' and config['log_every'] == 1


# ---- the revision round (the review's I4, m4, m5) ------------------------------------------------------------------------------------------------------------
def test_the_gate_is_bzs_b():
    """BZ's (b), 11-11 ~3:38 PM CST: mean Delta seg35 >= +.002, >= 13/20 positive, mean Delta seg4 >= -.001, the bars included; R2's +.005 / 15 is not used."""
    import decide_r3
    d35 = dict(mean=.002, n_positive=13, n=20)
    d4 = dict(mean=-.001)
    assert decide_r3.reading_of(d35, d4) == (decide_r3.PASS, [])
    assert decide_r3.reading_of(dict(d35, mean=.0045, n_positive=14), d4)[0] == decide_r3.PASS          # would not pass R2's gate
    for a, b in ((dict(d35, mean=.00199), d4), (dict(d35, n_positive=12), d4), (d35, dict(mean=-.0011))):
        reading, why = decide_r3.reading_of(a, b)
        assert reading == decide_r3.NOT_PASS and len(why) == 1
    assert decide_r3.reading_of(dict(d35, mean=None), d4)[0] == decide_r3.UNDECIDABLE
    assert 'seg35 >= +0.002' in decide_r3.decide(None)['rule']['pass_'] and '13/20' in decide_r3.decide(None)['rule']['pass_']


def test_a_real_reading_refuses_a_synthetic_or_forced_composition(world, tmp_path):
    import eval_r3
    import r3_core
    path = world['out']/'compose'/'composition.json'
    with pytest.raises(r3_core.Refused, match='synthetic or forced'):
        eval_r3.read_composition(path, real=True)
    forced = world['work']/'forced'/'composition.json'
    document = json.loads(forced.read_text())
    clean = tmp_path/'composition.json'
    clean.write_text(json.dumps({k: v for k, v in document.items() if k not in ('synthetic', 'forced_for_the_synthetic_chain')}))
    with pytest.raises(r3_core.Refused, match='does not give its winner'):                              # the forced winner is not the rule's
        eval_r3.read_composition(clean, real=True)
    original = json.loads(path.read_text())
    if not original['no_change']:
        clean.write_text(json.dumps(dict(original, synthetic=False)))
        assert eval_r3.read_composition(clean, real=True)[0]['winner'] == original['winner']
    assert eval_r3.read_composition(forced, real=False)[0]['winner'] == r3_synthetic.FORCED


def test_members_plan_stops_on_a_drift_of_ls_config(world, monkeypatch):
    """The review's m4: a default of TrainConfigLarge that moved since UF trained (here base_channels) stops prepare before any training, not check_r3l after 4.9 h."""
    import dataclasses
    import bayesreg.seg.unet.large as large
    from bayesreg.seg.unet.data import load_full_folds
    drifted = dataclasses.make_dataclass('Drifted', [('base_channels', int, dataclasses.field(default=99))], bases=(large.TrainConfigLarge,))
    plan = json.loads((world['out']/'members_plan.json').read_text())
    assert plan['decode']['path'] == r3c.DECODE_PY_REL and plan['decode']['sha256'] and isinstance(plan['decode']['stub'], bool)
    monkeypatch.setattr(large, 'TrainConfigLarge', drifted)
    tiny = r3_synthetic.tiny_l()
    tiny.pop('base_channels')
    spec = run_r3.Spec(out=world['out'], cwd=ROOT, l_overrides=tiny, uf_root=world['work']/'uf', uf_pinned=None, nn_root=world['work']/'nnroot', gate=False, synthetic=True)
    with pytest.raises(run_r3.Stop, match="L's config as cli would record it differs"):
        run_r3.members_plan(spec, load_full_folds(world['work']/'folds_full_v2.json'), splits_sha256=plan['nn']['splits_sha256'])
