"""PQH's firewall (contract C22.H4-3 with revisions 7a-7d; C22.H8 实现门槛 防火墙): the watched scope (code out of it only inside the code trees, review m8), the forbidden list (it wins), default
deny, the additions recorded, the four sealed lists by path and every read of them recorded (no count judged), the development spy (pq_firewall.Spy) on these rules: pair.npz members, links; the
counterexamples the contract names (review I1: another arm's disp.npz, pq_v1, the other files of the lists' directory, a pair.npz seg key, an evaluation record), and the output bit for bit the same
when the seg keys of pair.npz are changed and the evaluation records deleted (CPU, tmp_path)."""
import os
from pathlib import Path

import numpy as np
import pytest

from . import _world as W

hfw = __import__('pqh_firewall')
hp, hu, io_ = W.hp, W.hu, W.io_


@pytest.fixture
def rules(tmp_path):
    watched = tmp_path/'home'
    (watched/'cache').mkdir(parents=True)
    (watched/'results'/'pq').mkdir(parents=True)
    code = watched/'code'
    (code/'runs'/'20261107-pq').mkdir(parents=True)
    (code/'runs'/'x').mkdir(parents=True)
    r = hfw.HoldoutRules(unit='out', mode='holdout', watched=[watched, tmp_path/'d'], unwatched=[watched/'cache'], files=[watched/'results'/'allowed.json', code/'runs'/'20261107-pq'/'plan.md'],
                         dirs=[watched/'results'/'pq'], patterns=[(watched/'auth', r'holdout_B_v1_pqh_rerun_[0-9]\.json')], forbidden_dirs=[watched/'results'/'holdout_B_v1'/'eval'],
                         result_roots=[code/'runs', code/'docs'], pinned_documents=[code/'runs'/'20261107-pq'/'plan.md'], code_trees=[code])
    return W.namespace(r=r, home=watched, code=code, tmp=tmp_path)


def judge(rules, path):
    p = Path(os.path.abspath(path))
    return rules.r.judge(p, p.resolve())


def test_the_watched_scope(rules):
    r, home = rules.r, rules.home
    assert r.in_scope(home/'results'/'x.json') and r.in_scope(rules.tmp/'d'/'data'/'a.nii.gz')
    assert not r.in_scope(home/'code'/'x.py') and not r.in_scope(home/'cache'/'k.bin') and not r.in_scope(rules.tmp/'elsewhere.json')
    assert not r.in_scope(home/'code'/'__pycache__'/'x.cpython-310.pyc')
    link = home/'cache'/'link.json'
    link.symlink_to(home/'results'/'allowed.json')
    assert r.in_scope(link)                                                       # a link in an unwatched cache that leads into the scope is watched (its resolved form)


def test_m8_code_is_out_of_scope_only_inside_the_code_trees(rules):
    """C22.H4-3: "分离检出里不在监视范围内的只是代码": a .py or a __pycache__ outside the code trees (the results root, D:) is watched."""
    r, home = rules.r, rules.home
    assert not r.in_scope(home/'code'/'pkg'/'x.py') and not r.in_scope(home/'code'/'pkg'/'__pycache__'/'x.cpython-310.pyc')
    for path in (home/'results'/'x.py', rules.tmp/'d'/'data'/'x.py', home/'results'/'__pycache__'/'x.cpython-310.pyc'):
        assert r.in_scope(path) and 'default deny' in judge(rules, path)
    (home/'results'/'payload.py').write_text('x = 1\n')
    link = home/'code'/'link.py'
    link.symlink_to(home/'results'/'payload.py')                                  # a .py in the code tree that leads out of it is watched
    assert r.in_scope(link)


@pytest.mark.parametrize('rel, reason', [('results/pq/aligned_seg35.nii.gz', 'label'), ('results/pq/x_seg4.nii.gz', 'label'), ('results/pq/evaluation.json', 'evaluation'),
                                         ('results/pq/evaluation_started.json', 'evaluation'), ('results/pq/metrics_0600.json', 'evaluation'), ('results/pq/repair_metrics.json', 'evaluation'),
                                         ('results/pq/identity_eval_1.json', 'evaluation'), ('results/pq/cert_eval/x.json', 'evaluation'), ('results/pq/output_superseded_0/final.npz', 'withdrawn'),
                                         ('results/pq/output_failed_1/cert.json', 'withdrawn'), ('results/pq/tta_dev_read.lock', 'read lock'), ('results/pq/analysis/a.json', 'analysis'),
                                         ('results/pq/decision_pq.json', 'analysis'), ('results/holdout_B_v1/eval/x/y.json', 'never reads'), ('code/runs/x/result.md', 'result'),
                                         ('code/runs/x/reviews/r.md', 'result')])
def test_the_forbidden_list_wins_over_the_allowed_directory(rules, rel, reason):
    verdict = judge(rules, rules.home/rel)
    assert verdict is not None and verdict.startswith('forbidden') and reason in verdict


def test_default_deny_and_the_allowed_items(rules):
    home = rules.home
    assert judge(rules, home/'results'/'allowed.json') is None and judge(rules, home/'results'/'pq'/'x'/'pq.json') is None
    assert judge(rules, home/'code'/'runs'/'20261107-pq'/'plan.md') is None                         # the pinned plan is not a result document
    assert judge(rules, home/'auth'/'holdout_B_v1_pqh_rerun_1.json') is None
    assert 'default deny' in judge(rules, home/'auth'/'holdout_B_v1_xb_rerun_1.json')
    assert 'default deny' in judge(rules, home/'results'/'other.json')
    assert judge(rules, rules.tmp/'elsewhere.json') is None                                           # outside the watched scope


def test_additions_are_recorded(rules):
    path = rules.home/'results'/'method'/'final.npz'
    assert 'default deny' in judge(rules, path)
    rules.r.allow_files([path], 'the finalization record names it')
    assert judge(rules, path) is None and rules.r.describe()['added'][0]['reason'] == 'the finalization record names it'


def test_the_sealed_lists_by_path():
    from bayesreg.io import sealed
    files = hfw.sealed_list_files(W.ROOT)
    assert len(files) == len({s['path'] for s in sealed.SEALED_LISTS}) and all(Path(f).name in {Path(s['path']).name for s in sealed.SEALED_LISTS} for f in files)


def test_the_development_spy_on_these_rules(rules):
    home = rules.home
    (home/'results'/'pq'/'ok.json').write_text('{}')
    (home/'results'/'pq'/'metrics_0600.json').write_text('{}')
    stage = home/'results'/'pq'/'stage'/'pair.npz'
    stage.parent.mkdir()
    np.savez(stage, fixed=np.zeros(2), moving=np.zeros(2), initial_u=np.zeros(2), fixed_seg35=np.zeros(2))
    spy = hfw.spy(rules.r)
    with spy:
        (home/'results'/'pq'/'ok.json').read_text()
        (home/'results'/'pq'/'ok.json').read_text()
        with np.load(stage) as z:
            z['fixed']
            with pytest.raises(hfw.Breach):
                z['fixed_seg35']
        with pytest.raises(hfw.Breach):
            (home/'results'/'pq'/'metrics_0600.json').read_text()
    record = spy.record()
    assert record['m6'] is False and len(record['breaches']) == 2
    row = next(f for f in record['files'] if f['path'].endswith('ok.json'))
    assert row['n_opens'] == 2                                                    # every read recorded, any number (revision 7d)


def test_the_units_rules_name_the_sealed_lists_and_allow_no_hold_out_image_to_the_out_unit(tmp_path, monkeypatch):
    row = W.hc.holdout_pairs()[0]
    layout = W.hin.holdout_layout(W.ROOT, pairs=[row])
    layout.extra.update(watched=(tmp_path, W.ROOT), unwatched=())                 # the sandbox's data root is in tmp_path (production: /mnt/d and /home/rmcfz)
    rules_out, rules_members = hu.rules_for(layout, 'out'), hu.rules_for(layout, 'members')
    for path in hfw.sealed_list_files(W.ROOT):
        assert rules_out.judge(Path(path)) is None and rules_members.judge(Path(path)) is None
    image = Path(hp.DATA_IMAGES)/row[0]/'aligned_norm.nii.gz'
    assert rules_members.allowed(image.resolve()) and not rules_out.allowed(image.resolve())
    assert rules_members.judge(Path(hp.DATA_IMAGES)/row[0]/'aligned_seg35.nii.gz').startswith('forbidden')


# ---- the counterexamples of C22.H8 实现门槛 防火墙 (review I1) ------------------------------------------------------------------------------------------
@pytest.fixture
def unit_rules(tmp_path):
    row = W.hc.holdout_pairs()[0]
    layout = W.hin.holdout_layout(W.ROOT, pairs=[row])
    layout.extra.update(watched=(tmp_path, W.ROOT), unwatched=())
    return W.namespace(layout=layout, pair=layout.pairs[0], rules=dict(out=hu.rules_for(layout, 'out'), members=hu.rules_for(layout, 'members')))


def counterexamples(layout, pair):
    from bayesreg.io import sealed
    root, big = Path(W.hc.HOLDOUT_ROOT), Path(W.hc.BIG_ROOT)
    return {
        "another arm's disp.npz (work)": root/'work'/'configC'/pair/'output'/'disp.npz',
        "another arm's disp.npz (D:)": big/'xb_convexadam'/pair/'disp.npz',
        "another arm's disp.npz (method arm, not named by the record)": big/'method'/'other_pair'/'output'/'disp.npz',
        'pq_v1 (the development reading)': Path(hp.PQ_V1)/'runs'/'0438-0439'/'pq.npz',
        'pq_v1 members': Path(hp.PQ_V1)/'members'/'OASIS_OAS1_0438_MR1'/'var.npz',
        "the lists' directory: SHA256SUMS": W.ROOT/'splits'/'bayesreg_v1'/'SHA256SUMS',
        "the lists' directory: eb_pairs.tsv": W.ROOT/'splits'/'bayesreg_v1'/'eb_pairs.tsv',
        "the lists' directory: holdout_B_pairs draft": Path(sealed.REPO)/'splits'/'bayesreg_v1'/'holdout_B_pairs_draft.tsv',
        "a draft under the sealed lists' plan/": W.ROOT/'archive'/'seq_vp_3d'/'results'/'verification'/'3d_stage_HOLDOUT_v1'/'plan'/'draft.tsv',
        'an evaluation record (the entry)': root/hp.EV_EVAL_DIR_NAME/pair/'evaluation.json',
        'an evaluation record (in a cell)': root/'work'/'method'/pair/'evaluation.json',
        'an evaluation record (cert_eval)': root/'work'/'method'/pair/'cert_eval'/'cert.json',
        'an evaluation record (metrics)': root/'work'/'method'/pair/'t88'/'metrics_0600.json',
        'a label file': Path(hp.DATA_IMAGES)/pair.split('__')[0]/'aligned_seg35.nii.gz',
    }


@pytest.mark.parametrize('unit', ['out', 'members'])
def test_the_counterexamples_are_refused_by_both_units(unit_rules, unit):
    rules = unit_rules.rules[unit]
    for name, path in counterexamples(unit_rules.layout, unit_rules.pair).items():
        verdict = rules.judge(Path(path))
        assert verdict is not None, (unit, name, path)


def test_a_seg_key_of_pair_npz_is_a_breach_under_the_units_rules(unit_rules, tmp_path):
    stage = Path(unit_rules.layout.stage)/unit_rules.pair/'pair.npz'
    stage.parent.mkdir(parents=True)
    np.savez(stage, fixed=np.zeros(2), moving=np.zeros(2), initial_u=np.zeros(2), fixed_seg35=np.zeros(2), moving_seg4=np.zeros(2))
    spy = hfw.spy(unit_rules.rules['out'])
    with spy:
        with np.load(stage) as z:
            z['fixed'], z['moving']
            for key in ('fixed_seg35', 'moving_seg4'):
                with pytest.raises(hfw.Breach):
                    z[key]
    assert spy.record()['m6'] is False


def test_changed_seg_keys_and_deleted_evaluation_records_leave_the_output_bit_for_bit(tmp_path, monkeypatch):
    """C22.H8 防火墙 反例: the development unit's arrays first; then the seg keys of pair.npz are given other contents (its manifest's pair_sha256 follows) and the evaluation records of the world
    deleted; the hold-out computation (the PQ unit, dev mode) gives the same pq.npz and baselines.npz, bit for bit."""
    import json
    import run_pq
    w, layout, _ = W.dev_world(tmp_path, monkeypatch)
    W.XW.sealed_module._reset_cache()
    assert run_pq.out_unit(w.ctx, gate=W.GATE, code=W.CODE, log=W.quiet) == 0
    reference = {}
    for name in (io_.PQ_NPZ, io_.BASELINES_FILE):
        with np.load(w.ctx.pq_root/'runs'/W.SHORT/name) as z:
            reference.update({k: np.array(z[k]) for k in z.files})
    d = Path(layout.stage)/W.PAIR
    with np.load(d/'pair.npz') as z:
        data = {k: np.array(z[k]) for k in z.files}
    rng = np.random.default_rng(99)
    for key in [k for k in data if 'seg' in k]:
        data[key] = rng.integers(0, 36, data[key].shape).astype(data[key].dtype)
    (d/'pair.npz').unlink()
    np.savez_compressed(d/'pair.npz', **data)
    manifest = json.loads((d/'manifest.json').read_text())
    manifest['pair_sha256'] = io_.sha_file(d/'pair.npz')
    (d/'manifest.json').write_text(json.dumps(manifest))
    removed = [p for p in w.results.rglob('*') if p.is_file() and (p.name in ('metrics_0600.json', 'repair_metrics.json') or p.name.startswith(('evaluation', 'identity_eval')))]
    assert removed
    for p in removed:
        p.unlink()
    assert W.run_out(layout, deps=W.deps()) == 0
    mine = {}
    for name in (hu.PQ_NPZ, hu.BASELINES):
        with np.load(Path(layout.pq_big)/W.PAIR/name) as z:
            mine.update({k: np.array(z[k]) for k in z.files})
    assert sorted(mine) == sorted(reference) and all(mine[k].dtype == reference[k].dtype and mine[k].tobytes() == reference[k].tobytes() for k in mine)
