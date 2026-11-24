"""R2's segmentation level (experiments/r2/run_r2.py, eval_r2.py, decide_r2.py, r2_lock.py) and the seg phase of its registration reading (r2_seg.py).  CPU; the synthetic world of run_r2.build_synthetic
(experiments/unv2/synthetic_tree.py's subjects, tiny UF members, ids OASIS_OAS1_09xx_MR1, a non-real data root) in tmp_path; no real label, image, prediction or splits/ file is read.

- one arm (the lead's ruling of 11-10 ~10:42 PM CST, review B1): no step, arm, member or candidate of r2sl anywhere;
- the chain on synthetic subjects: the order (control, r2s A B, check-members, the evaluation, the reading), the hold file, a second pass skips everything, every member check empty, the
  control 0, the reading 不可判 for 2 subjects only;
- the evaluation's arithmetic: the running sum of ensemble_probabilities bit for bit; the arms' orders are the candidate's registry order;
- the members' checks (each departure), the targets, the lock (fixed path, O_EXCL, BZ's words), the gate's C: floor, the start string;
- the reading's lines on synthetic evaluations: the bars included, the tie, the 15 of 20, the seg4 floor, each undecidable case;
- the seg phase of the registration reading on the chain's outputs: the staged registry (two fold models of three members), the 20-subject inference here 2, the control against eval_r2.json's rows
  (equal row for row on the CPU), the registration once.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'r2', ROOT/'experiments'/'unv2'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import decide_r2  # noqa: E402
import eval_r2  # noqa: E402
import r2_common as rc  # noqa: E402
import r2_lock  # noqa: E402
import run_r2  # noqa: E402

from bayesreg.seg.unet.infer import ensemble_probabilities  # noqa: E402


def quiet(*args, **kwargs):
    pass


@pytest.fixture(scope='module')
def chain(tmp_path_factory):
    """The synthetic chain once for the module (about half a minute on the CPU)."""
    import os
    work = tmp_path_factory.mktemp('r2world')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    lock = r2_lock.DEV_READ_LOCK
    r2_lock.DEV_READ_LOCK = work/'never.lock'
    try:
        problems = run_r2.synthetic_chain(work, env)
    finally:
        r2_lock.DEV_READ_LOCK = lock
    return dict(work=work, problems=problems, out=work/'out')


def test_the_synthetic_chain(chain):
    out = chain['out']
    assert chain['problems'] == []
    order = [line.split()[2].rstrip(':') for line in (out/'chain.log').read_text().splitlines() if len(line.split()) > 2 and line.split()[1] == 'start']
    assert order == ['control_check']+[f'{s}_{r}_{m}' for r in rc.RECIPE_ORDER for m in rc.MEMBER_NAMES for s in ('train', 'check', 'calibrate')]+['check_members_r2s', 'eval_r2', 'decide_r2']
    assert not [n for n in order if 'r2sl' in n] and not (out/'r2sl').exists()
    evaluation = json.loads((out/'eval_r2'/'eval_r2.json').read_text())
    assert evaluation['arms'] == list(eval_r2.ARMS) and evaluation['control']['passed'] and evaluation['control']['differences'] == dict(seg35=0., seg4=0.)
    assert sorted(evaluation['members']) == sorted([f'uf/{n}' for n in rc.UF_ORDER]+[f'{r}/{n}' for r in rc.RECIPE_ORDER for n in rc.MEMBER_NAMES])
    for r in rc.RECIPE_ORDER:
        record = json.loads((out/r/'foldA_s2'/'train_record.json').read_text())
        assert record['recipe'] == 'r2_small' and record['config']['target_labels'] == run_r2.SYNTHETIC_TARGETS and record['checkpoint_choice'] == 'last'
    decision = json.loads((out/'analysis'/'decision_r2.json').read_text())
    assert decision['reading'] == decide_r2.UNDECIDABLE and decision['why'] == ['2 名受试者，不是 20']
    ends = [json.loads(p.read_text()) for p in [out/'queue_end.json']+sorted(out.glob('queue_end_*.json'))]
    assert len(ends) == 3 and 'hold' in ends[0]['stop'] and ends[0]['finished'] is False and ends[-1]['finished'] is True and ends[-1]['reading'] == decide_r2.UNDECIDABLE


def test_the_running_sum_is_ensemble_probabilities_bit_for_bit():
    g = torch.Generator().manual_seed(0)
    members = [(torch.randn(1, 36, 4, 5, 6, generator=g), t) for t in (1.3, .9, 1.1)]
    want = ensemble_probabilities(iter(members))
    acc = eval_r2.Accumulator()
    for z, t in members:
        acc.add(torch.softmax(z.float()/float(t), dim=1)[0])
    assert torch.equal(acc.probabilities(), want)


def test_the_arms_are_the_registry_order():
    arms = eval_r2.ensemble_arms()
    assert arms['UF4'] == [('uf', n) for n in rc.UF_ORDER]
    assert arms['R2S'] == [('uf', 'foldA_s0'), ('uf', 'foldA_s1'), ('r2s', 'foldA_s2'), ('uf', 'foldB_s0'), ('uf', 'foldB_s1'), ('r2s', 'foldB_s2')]
    assert arms['R2S'] == [m for _, ms in rc.candidate_plan('R2S') for m in ms] and list(arms) == ['UF4', 'R2S', 'r2s_2']
    assert set(eval_r2.SINGLE_ARMS) == {'r2s_A', 'r2s_B'} and eval_r2.CANDIDATE_ARMS == ('R2S',) and eval_r2.ARMS == ('UF4', 'R2S', 'r2s_2', 'r2s_A', 'r2s_B')
    assert [f for f, _ in eval_r2.forward_order()] == ['uf', 'uf', 'r2s', 'uf', 'uf', 'r2s']


def test_one_arm_only():
    assert rc.RECIPE_ORDER == ('r2s',) and set(rc.RECIPES) == {'r2s'} and rc.CANDIDATE_ORDER == ('R2S',) and set(rc.CANDIDATES) == {'R2S'}
    assert set(run_r2.TIMEOUTS) == {'control_check', 'train_r2s', 'calibrate', 'check_members', 'eval_r2', 'decide_r2'} and 'train_r2sl' not in run_r2.GPU_KINDS and set(run_r2.TINY_R2) == {'r2s'}
    assert run_r2.runtime_max_seconds() == 3600+2*(16200+1800)+1800+7200+600+(1+4+1)*30*60+1800
    with pytest.raises(KeyError):
        rc.train_config('r2sl', [4])
    assert rc.UF_PAIRED_SE == .00076


def namesake():
    return dict(train=['a', 'b'], calib=['c'], contact=['a', 'b', 'c'])


def good_record(recipe='r2s', targets=(4, 18)):
    config = eval_r2.planned_config(recipe, list(targets))
    m = config['max_iters']
    return dict(fold='A', seed=2, recipe='r2_small', config=config, stopped='max_iters', iterations_run=m, checkpoint_choice='last', checkpoint_file='model_last.pt', checkpoint_iteration=m,
                train_subjects=['b', 'a'], calib_subjects=['c'], label_contact_subjects=['a', 'b', 'c'])


@pytest.mark.parametrize('change, word', [
    (lambda r: r.update(seed=0), 'fold'), (lambda r: r.update(recipe='v3_large'), 'recipe'), (lambda r: r['config'].update(target_labels=[4]), 'config'), (lambda r: r['config'].update(p_target=.3), 'config'),
    (lambda r: r.update(stopped='wall_budget'), 'stopped'), (lambda r: r.update(iterations_run=10), 'stopped'), (lambda r: r.update(checkpoint_choice='best_calib'), 'last step'),
    (lambda r: r.update(train_subjects=['a']), 'subjects'), (lambda r: r.update(label_contact_subjects=['a', 'b']), 'subjects')])
def test_a_member_that_is_not_the_plans_is_named(change, word):
    expected = eval_r2.planned_config('r2s', [4, 18])
    assert eval_r2.member_record_problems(good_record(), recipe='r2s', name='foldA_s2', expected_config=expected, namesake=namesake()) == []
    record = good_record()
    change(record)
    problems = eval_r2.member_record_problems(record, recipe='r2s', name='foldA_s2', expected_config=expected, namesake=namesake())
    assert problems and any(word in p for p in problems)


def test_the_planned_configs():
    s = eval_r2.planned_config('r2s', [4, 18])
    assert s['max_iters'] == s['min_iters'] == s['patience'] == 40000 and s['checkpoint'] == 'last' and s['target_labels'] == [4, 18]
    assert s['p_target'] == .5 and s['target_dice_weight'] == 1. and s['base_channels'] == 32 and list(s['patch_size']) == [128, 160, 112] and s['max_wall_seconds'] == 14400.


@pytest.mark.parametrize('targets', [[4, 18], [18, 4, 19, 23, 34, 35], []])
def test_the_targets_file_must_hold_the_small_six_sorted(tmp_path, targets):
    path = tmp_path/'targets.json'
    path.write_text(json.dumps(dict(targets=targets)))
    with pytest.raises(ValueError):
        eval_r2.read_targets(path)
    path.write_text(json.dumps(dict(targets=[4, 10, 18, 19, 23, 29, 34, 35])))
    assert eval_r2.read_targets(path) == [4, 10, 18, 19, 23, 29, 34, 35]
    with pytest.raises(ValueError):
        eval_r2.read_targets(path, pinned='0'*64)


def test_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(r2_lock, 'DEV_READ_LOCK', tmp_path/'seg'/'r2_dev_read.lock')
    r2_lock.check()
    with pytest.raises(ValueError):
        r2_lock.check('BZ ok')
    line = r2_lock.take(dict(out_dir='x'))
    assert line['event'] == r2_lock.FIRST_READ and r2_lock.describe()[0]['out_dir'] == 'x'
    with pytest.raises(r2_lock.DevReadLocked):
        r2_lock.check()
    with pytest.raises(r2_lock.DevReadLocked):
        r2_lock.take(dict(out_dir='y'))
    with pytest.raises(ValueError):
        r2_lock.check('  ')
    r2_lock.check('BZ: 再读一次, 11-11 9:00 AM')
    r2_lock.take(dict(out_dir='z'), bz_go='BZ: 再读一次')
    assert [l['event'] for l in r2_lock.describe()] == [r2_lock.FIRST_READ, r2_lock.APPROVED_REREAD]
    assert rc.DEV_READ_LOCK == Path('/home/rmcfz/seq_vp_tag_results/seg/r2_dev_read.lock')


def test_the_real_evaluation_refuses_at_the_lock_first(tmp_path, monkeypatch):
    lock = tmp_path/'lock'
    lock.write_text(json.dumps(dict(utc='then', out_dir='x'))+'\n')
    monkeypatch.setattr(r2_lock, 'DEV_READ_LOCK', lock)
    with pytest.raises(r2_lock.DevReadLocked):
        eval_r2.evaluate(out_dir=tmp_path/'out', device='cpu')
    assert not (tmp_path/'out').exists()


def test_the_real_evaluation_refuses_other_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(r2_lock, 'DEV_READ_LOCK', tmp_path/'lock')
    monkeypatch.setattr(eval_r2, 'require_clean_code', lambda code: None)
    with pytest.raises(ValueError, match='plan'):
        eval_r2.evaluate(out_dir=tmp_path/'out', uf_root=tmp_path, device='cpu')
    assert not (tmp_path/'lock').exists()


def test_the_start_string_and_the_gate(monkeypatch):
    argv = run_r2.launch_argv()
    text = ' '.join(argv)
    assert f'--unit={run_r2.UNIT}' in argv and 'pytest' not in text and '--property=MemoryMax=8G' in argv and any(a.startswith('--property=RuntimeMaxSec=') for a in argv)
    assert run_r2.runtime_max_seconds() > 2*14400
    from types import SimpleNamespace
    assert run_r2.c_floor_problems(lambda p: SimpleNamespace(free=41*2**30)) == []
    assert run_r2.c_floor_problems(lambda p: SimpleNamespace(free=39*2**30))


def test_the_chain_refuses_targets_other_than_the_plans(tmp_path):
    path = tmp_path/'targets.json'
    path.write_text(json.dumps(dict(targets=run_r2.SYNTHETIC_TARGETS)))
    spec = run_r2.Spec(out=tmp_path/'out', cwd=tmp_path, targets_path=path)
    with pytest.raises(run_r2.Stop, match='members_plan'):
        run_r2.read_targets(spec, dict(targets_sha256='0'*64))
    assert run_r2.read_targets(spec, dict(targets_sha256=rc.sha256_file(path)))[0] == run_r2.SYNTHETIC_TARGETS
    with pytest.raises(run_r2.Stop, match='no such file'):
        run_r2.read_targets(run_r2.Spec(out=tmp_path, cwd=tmp_path, targets_path=tmp_path/'none.json'), None)


def test_the_chain_steps_of_the_real_run():
    steps = run_r2.chain_steps(run_r2.Spec(out=rc.OUT, cwd=rc.OUT/'snapshot'), [4, 18], 'f'*64)
    names = [n for n, _, _ in steps]
    assert names[0] == 'control_check' and names[-2:] == ['eval_r2', 'decide_r2'] and len(names) == 1+3*2+1+2
    assert not [n for n in names if 'r2sl' in n] and not [a for _, _, argv in steps if argv for a in argv if 'r2sl' in a]          # review B1 (a): no step of r2sl
    train = dict((n, a) for n, _, a in steps)['train_r2s_foldB_s2']
    assert train[train.index('--profile')+1] == 'r2_small' and '--seed' in train and train[train.index('--seed')+1] == '2'
    assert 'max_iters=40000' in train and 'target_labels=[4, 18]' in train and 'p_target=0.5' in train and 'target_dice_weight=1.0' in train
    assert dict((n, k) for n, k, _ in steps)['train_r2s_foldA_s2'] == 'train_r2s' and set(run_r2.GPU_KINDS) == {'train_r2s', 'eval_r2', 'control_check', 'calibrate'}
    evaluation = dict((n, a) for n, _, a in steps)['eval_r2']
    assert evaluation[evaluation.index('--targets-sha256')+1] == 'f'*64 and '--bz-go' not in evaluation


# ---- the reading ----------------------------------------------------------------------------------------------------------------------------------
SUBJECTS = list(rc.SUBJECTS)


def row(seg35, seg4=.95):
    return dict(seg35=seg35, seg4=seg4, small_six=.8, big_six=.95, other_23=.93, cortex=.94, white_matter=.97, per_label=[.9]*35, seg4_per_label=[.94, .96, .97, .96])


def evaluation_of(delta35, delta4=0., *, n_positive=20, control=0., truth_same=True, subjects=SUBJECTS, members=True, extra_members=()):
    rows, means = {}, {}
    base = {s: .93+.001*i for i, s in enumerate(subjects)}
    n = len(subjects)
    if n_positive in (0, n):                                    # all the same sign: every subject delta35
        per = {s: delta35 for s in subjects}
    else:                                                       # n_positive at x > 0, the others at -.001, the mean delta35
        x = (n*delta35+(n-n_positive)*.001)/n_positive
        per = {s: (x if i < n_positive else -.001) for i, s in enumerate(subjects)}
    rows['UF4'] = {s: row(base[s]) for s in subjects}
    rows['R2S'] = {s: row(base[s]+per[s], .95+delta4) for s in subjects}
    for arm in ('r2s_2', 'r2s_A', 'r2s_B'):
        rows[arm] = {s: row(base[s]) for s in subjects}
    for arm, r in rows.items():
        means[arm] = {m: float(np.mean([r[s][m] for s in subjects])) for m in ('seg35', 'seg4', 'small_six', 'big_six', 'other_23', 'cortex', 'white_matter')}
    names = [f'uf/{n}' for n in rc.UF_ORDER]+[f'{r}/{n}' for r in rc.RECIPE_ORDER for n in rc.MEMBER_NAMES]+list(extra_members)
    return dict(subjects=subjects, rows=rows, means=means, arms=list(rows), members={n: {} for n in names} if members else {}, member_problems=[], targets=[4, 18],
                control=dict(differences=dict(seg35=control, seg4=0.), tolerance=rc.CONTROL_TOL_FORWARD, truth_same=truth_same, rows_equal=20), per_label_mean={a: [.9]*35 for a in rows},
                uf_singles={})


@pytest.mark.parametrize('delta, positive, delta4, want', [
    (.005, 20, 0., decide_r2.PASS), (.0049999, 20, 0., decide_r2.NOT_PASS), (.006, 15, 0., decide_r2.PASS), (.006, 14, 0., decide_r2.NOT_PASS),
    (.006, 20, -.0009, decide_r2.PASS), (.006, 20, -.0011, decide_r2.NOT_PASS), (-.002, 0, 0., decide_r2.NOT_PASS)])
def test_the_lines(delta, positive, delta4, want):
    document = decide_r2.decide(evaluation_of(delta, delta4, n_positive=positive))
    assert document['reading'] == want and document['primary_arm'] == 'R2S'
    assert (document['candidate'] == 'R2S') == (want == decide_r2.PASS)
    assert document['n_positive'] == positive


def test_the_reading_reads_no_r2sl():
    """Review B1 (b): the evaluation has no R2SL arm and no r2sl member, and the reading is made; the members it wants are UF's four and r2s's two, and a member more is not the plan's."""
    evaluation = evaluation_of(.006)
    assert not [a for a in evaluation['rows'] if 'r2sl' in a.lower()]
    document = decide_r2.decide(evaluation)
    assert document['reading'] == decide_r2.PASS and document['primary_arm'] == 'R2S' and set(document['means']) == {'UF4', 'R2S', 'r2s_2', 'r2s_A', 'r2s_B'}
    assert not [k for k in document['comparisons'] if 'r2sl' in k.lower()] and set(document['singles']) == {'r2s_A', 'r2s_B'}
    assert decide_r2.decide(evaluation_of(.006, extra_members=('r2sl/foldA_s2',)))['reading'] == decide_r2.UNDECIDABLE
    assert document['plan']['path'] == rc.PLAN_REL and document['plan']['sha256'] == rc.sha256_file(ROOT/rc.PLAN_REL)


@pytest.mark.parametrize('kwargs, word', [(dict(control=3e-4), '控制不符'), (dict(truth_same=False), '真值'), (dict(subjects=SUBJECTS[:19]), '19 名'), (dict(members=False), '成员缺失')])
def test_undecidable(kwargs, word):
    document = decide_r2.decide(evaluation_of(.006, **kwargs))
    assert document['reading'] == decide_r2.UNDECIDABLE and any(word in w for w in document['why']) and document['candidate'] is None
    assert decide_r2.decide(None)['reading'] == decide_r2.UNDECIDABLE


def test_the_expectation_and_the_summary(tmp_path):
    document = decide_r2.decide(evaluation_of(.006))
    e = document['expectation']
    assert e['pairs19']['seg_level_delta'] == pytest.approx(.006) and e['pairs19']['expected_delta'] == pytest.approx(.890*.006) and e['pairs19']['expected_range'] == pytest.approx([.84*.006, .92*.006])
    decide_r2.write(tmp_path, document)
    assert (tmp_path/'summary_r2.md').read_text().startswith('# R2 分割级读数') and json.loads((tmp_path/'decision_r2.json').read_text())['reading'] == decide_r2.PASS
    with pytest.raises(FileExistsError):
        decide_r2.write(tmp_path, document)


# ---- the seg phase of the registration reading, on the chain's outputs ---------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def seg_phase(chain, tmp_path_factory):
    import r2_seg
    import synthetic_tree
    from bayesreg.io.guard import load_dev_allowlist
    from bayesreg.seg.unet.data import OasisFiles, load_full_folds
    work = chain['work']
    root = tmp_path_factory.mktemp('r2seg')
    evaluation = json.loads((chain['out']/'eval_r2'/'eval_r2.json').read_text())
    common = dict(uf_root=work/'uf', r2_out=chain['out'], evaluation=evaluation, split=load_full_folds(work/'folds_full_v2.json'), allowlist=load_dev_allowlist(work/'dev_subjects.json'),
                  training_source=synthetic_tree.SOURCE, pinned_folds_sha256=None)
    seg_dir = root/'unet35_r2s_ens'
    record = r2_seg.infer_all('R2S', subjects=synthetic_tree.DEV, seg_dir=seg_dir, files=OasisFiles(work/'data'), device='cpu', code=dict(commit='test'), command=['test'], log=quiet, **common)
    check = r2_seg.segcheck('R2S', seg_dir=seg_dir, evaluation=evaluation, subjects=synthetic_tree.DEV, data_root=work/'data', n_subjects=len(synthetic_tree.DEV), log=quiet)
    return dict(record=record, check=check, seg_dir=seg_dir, root=root, common=common, dev=synthetic_tree.DEV, evaluation=evaluation)


def test_the_staged_registry_has_two_fold_models_of_three_members(seg_phase):
    from bayesreg.io.pair import load_segmenter_registry
    registry = load_segmenter_registry(seg_phase['seg_dir']/'registry_staged.json')
    assert sorted(registry.models) == ['unet35_r2s_ens_foldA', 'unet35_r2s_ens_foldB']
    members = seg_phase['evaluation']['members']
    for fold in ('A', 'B'):
        entry = registry.models[f'unet35_r2s_ens_fold{fold}']
        assert entry['segmenter_id'] == 'unet35_r2s_ens' and list(entry['member_model_sha256']) == [members[f'uf/fold{fold}_s0']['member_sha256'], members[f'uf/fold{fold}_s1']['member_sha256'], members[f'r2s/fold{fold}_s2']['member_sha256']]


def test_the_registered_predictions_are_the_evaluations_rows(seg_phase):
    record, check = seg_phase['record'], seg_phase['check']
    assert record['passed'] is True and record['candidate'] == 'R2S' and record['n_reader_not_own_total'] == 0
    assert check['passed'] is True and check['n_rows_equal'] == len(seg_phase['dev']) and check['mean_difference'] == dict(seg35=0., seg4=0.) and check['uf_arm'] == 'R2S'
    for subject in seg_phase['dev']:
        provenance = json.loads((seg_phase['seg_dir']/'pred'/subject/'provenance.json').read_text())
        assert provenance['segmenter_id'] == 'unet35_r2s_ens' and provenance['test_labels_seen'] is False and len(provenance['members']) == 6 and 'segmenters_r2_v1.json' in provenance['registry_note']


def test_the_registration_is_the_staged_bytes_once(seg_phase):
    import r2_seg
    path = seg_phase['root']/'segmenters_r2_v1.json'
    registry = r2_seg.register('R2S', seg_dir=seg_phase['seg_dir'], registry_path=path, subjects=seg_phase['dev'])
    assert path.read_bytes() == (seg_phase['seg_dir']/'registry_staged.json').read_bytes() and sorted(registry.models) == ['unet35_r2s_ens_foldA', 'unet35_r2s_ens_foldB']
    r2_seg.register('R2S', seg_dir=seg_phase['seg_dir'], registry_path=path, subjects=seg_phase['dev'])        # the same bytes again: accepted
    path.write_text(path.read_text()+' ')
    with pytest.raises(Exception, match='other content'):
        r2_seg.register('R2S', seg_dir=seg_phase['seg_dir'], registry_path=path, subjects=seg_phase['dev'])


def test_the_inference_is_not_rerun_and_other_members_are_refused(seg_phase):
    import r2_seg
    from bayesreg.seg.unet.data import OasisFiles
    with pytest.raises(FileExistsError):
        r2_seg.infer_all('R2S', subjects=seg_phase['dev'], seg_dir=seg_phase['seg_dir'], files=OasisFiles(Path('/nonexistent')), device='cpu', code={}, command=[], log=quiet, **seg_phase['common'])
    other = json.loads(json.dumps(seg_phase['evaluation']))
    other['members']['r2s/foldA_s2']['member_sha256'] = '0'*64
    common = dict(seg_phase['common'], evaluation=other)
    with pytest.raises(r2_seg.HSSegFailed, match='eval_r2.json evaluated'):
        r2_seg.stage('R2S', seg_dir=seg_phase['root']/'other', **{k: v for k, v in common.items()})
