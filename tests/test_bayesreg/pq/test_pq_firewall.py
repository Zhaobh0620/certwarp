"""PQ's firewall (contract C22.Q1-3 "禁读清单 + 允许的目录", C22.Q7 防火墙): experiments/pq/pq_firewall.py on files in tmp_path.

- the rules: an allowed file passes; the forbidden list wins over an allowed directory; a watched file outside the allowed items is stopped; a file outside the watched scope and a code file pass, recorded;
- the spy: builtins.open, io.open, Path.open / read_text, os.open, np.load's members of a pair.npz and zipfile reads are all seen; a forbidden one raises inside the open (the unit stops), recorded.
"""
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

from . import _world as W  # noqa: F401

import pq_firewall as fw  # noqa: E402


@pytest.fixture
def tree(tmp_path):
    repo, results, data, images = tmp_path/'repo', tmp_path/'results', tmp_path/'repo'/'data', tmp_path/'images'
    pair = 'OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1'
    files = {
        'allowed_config': repo/'configs'/'bayesreg'/'dev_subjects.json',
        'forbidden_in_config': repo/'configs'/'bayesreg'/'metrics_0600.json',
        'plan': repo/'runs'/'20261107-pq'/'plan.md',
        'result_doc': repo/'runs'/'20261107-pq'/'result.md',
        'review': repo/'docs'/'bayesreg'/'reviews'/'x.md',
        'folds': repo/'splits'/'bayesreg_v1'/'folds.json',
        'other_split': repo/'splits'/'bayesreg_v1'/'holdout_B_subjects.txt',
        'manifest': data/'three_d'/'stage_L2R_val_v1'/pair/'manifest.json',
        'label': images/'OASIS_OAS1_0438_MR1'/'aligned_seg35.nii.gz',
        'label_data': data/'three_d'/'stage_L2R_val_v1'/pair/'aligned_seg4.nii.gz',
        'image': images/'OASIS_OAS1_0438_MR1'/'aligned_norm.nii.gz',
        'run_json': results/'p3_v1'/'runs'/'C_first5'/'0438-0439'/'run.json',
        'metrics': results/'p3_v1'/'runs'/'C_first5'/'0438-0439'/'metrics_0600.json',
        'repair_metrics': results/'p3_v1'/'runs'/'RC_c600'/'0438-0439'/'repair_metrics.json',
        'batch': results/'p3_v1'/'batch.json',
        'own': results/'pq_v1'/'runs'/'0438-0439'/'pq.json',
        'eval': results/'pq_v1'/'eval'/'0438-0439'/'eval.json',
        'analysis': results/'pq_v1'/'analysis'/'decision_pq.json',
        'lock': results/'seg'/'uf_dev_read.lock',
        'protected': results/'seg'/'unet35_v4_full'/'eval_uf'/'eval_uf.json',
        'probs': results/'seg'/'unet35_v4_ens'/'pred'/'OASIS_OAS1_0438_MR1'/'probs36.npz',
        'outside': tmp_path/'elsewhere'/'notes.txt',
        'code': results/'frozen_eb808be'/'jd_curl'/'three_d'/'preopt.py',
    }
    for p in files.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('x')
    rules = fw.Rules(repo=repo, results=results, data=data, images=images, pairs=[pair], subjects=['OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0439_MR1'], p3_dirs={'0438-0439': ('C_first5', 'RC_c600')})
    members_rules = fw.Rules(repo=repo, results=results, data=data, images=images, pairs=[pair], subjects=['OASIS_OAS1_0438_MR1'], unit='members')
    return dict(files=files, rules=rules, members=members_rules, pair_dir=data/'three_d'/'stage_L2R_val_v1'/pair)


@pytest.mark.parametrize('name, verdict', [
    ('allowed_config', None), ('plan', None), ('folds', None), ('manifest', None), ('run_json', None), ('own', None), ('probs', None), ('outside', None), ('code', None),
    ('forbidden_in_config', 'forbidden'), ('result_doc', 'forbidden'), ('review', 'forbidden'), ('label', 'forbidden'), ('label_data', 'forbidden'), ('metrics', 'forbidden'),
    ('repair_metrics', 'forbidden'), ('eval', 'forbidden'), ('analysis', 'forbidden'), ('lock', 'forbidden'), ('protected', 'forbidden'),
    ('other_split', 'outside'), ('batch', 'outside'), ('image', 'outside'),
])
def test_the_rules(tree, name, verdict):
    reason = tree['rules'].judge(tree['files'][name].resolve())
    if verdict is None:
        assert reason is None
    elif verdict == 'forbidden':
        assert reason is not None and reason.startswith('forbidden')
    else:
        assert reason == 'outside the allowed items of C22.Q1-3'


def test_the_members_unit_may_read_the_image_only_there(tree):
    assert tree['members'].judge(tree['files']['image'].resolve()) is None
    assert tree['rules'].judge(tree['files']['image'].resolve()) is not None
    assert tree['members'].judge(tree['files']['label'].resolve()).startswith('forbidden')


def test_the_spy_sees_every_way_of_opening(tree):
    f = tree['files']
    with fw.Spy(tree['rules']) as spy:
        open(f['allowed_config']).close()
        io.open(f['folds']).close()
        Path(f['manifest']).read_text()
        os.close(os.open(f['run_json'], os.O_RDONLY))
        open(f['outside']).read()
    record = spy.record()
    seen = {Path(e['path']) for e in record['files']}
    assert {f[k].resolve() for k in ('allowed_config', 'folds', 'manifest', 'run_json', 'outside')} <= seen
    assert record['m6'] is True and not record['breaches']
    assert [e['in_scope'] for e in record['files'] if Path(e['path']) == f['outside'].resolve()] == [False]


@pytest.mark.parametrize('name', ['metrics', 'repair_metrics', 'label', 'batch', 'eval', 'lock', 'review'])
def test_a_forbidden_or_unlisted_open_stops_inside_the_open(tree, name):
    with fw.Spy(tree['rules']) as spy:
        with pytest.raises(fw.Breach):
            open(tree['files'][name]).read()
        with pytest.raises(fw.Breach):
            Path(tree['files'][name]).read_bytes()
    record = spy.record()
    assert record['m6'] is False and len(record['breaches']) == 2 and all(Path(b['path']) == tree['files'][name].resolve() for b in record['breaches'])


def test_pair_npz_members_beyond_the_images_stop(tree):
    path = tree['pair_dir']/'pair.npz'
    np.savez(path, fixed=np.zeros(3), moving=np.ones(3), initial_u=np.zeros(3), fixed_seg35=np.arange(3.))
    with fw.Spy(tree['rules']) as spy:
        with np.load(path) as z:
            assert np.array_equal(z['fixed'], np.zeros(3)) and 'fixed_seg35' in z.files      # listing the members reads none
            z['moving']
            with pytest.raises(fw.Breach):
                z['fixed_seg35']
        with zipfile.ZipFile(path) as archive:
            archive.read('initial_u.npy')
            with pytest.raises(fw.Breach):
                archive.read('fixed_seg35.npy')
    record = spy.record()
    assert record['pair_members'][str(path.resolve())] == ['fixed', 'fixed_seg35', 'initial_u', 'moving'] and record['m6'] is False
    assert sorted(b['member'] for b in record['breaches']) == ['fixed_seg35', 'fixed_seg35']


def test_the_spy_is_removed(tree):
    real = open
    spy = fw.Spy(tree['rules']).install()
    spy.remove()
    assert open is real and np.lib.npyio.NpzFile.__getitem__ is spy.real_getitem and zipfile.ZipFile.open is spy.real_zip_open
    open(tree['files']['metrics']).read()                                            # nothing watches any more
    assert spy.record()['files'] == []


def test_symbolic_links_are_judged_in_both_forms(tmp_path):
    """Review minor 4: data/ a link (as in production); a link inside the scope that leads outside; a link with an innocent name that leads to a forbidden file."""
    repo, results, images = tmp_path/'repo', tmp_path/'results', tmp_path/'images'
    target = tmp_path/'elsewhere'/'seq_vp_3d_data'
    pair = 'OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1'
    stage = target/'three_d'/'stage_L2R_val_v1'/pair
    stage.mkdir(parents=True)
    (stage/'manifest.json').write_text('{}')
    (stage/'aligned_seg35.nii.gz').write_text('x')
    repo.mkdir()
    (repo/'data').symlink_to(target)
    outside = tmp_path/'outside.txt'
    outside.write_text('x')
    (results/'pq_v1').mkdir(parents=True)
    (results/'pq_v1'/'notes.json').symlink_to(outside)                                 # inside an allowed directory, leading out of the scope
    metrics = results/'p3_v1'/'runs'/'C_first5'/'0438-0439'/'metrics_0600.json'
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{}')
    (results/'pq_v1'/'innocent.json').symlink_to(metrics)                              # an innocent name for a forbidden file
    (repo/'configs'/'bayesreg').mkdir(parents=True)
    (repo/'configs'/'bayesreg'/'helper.py').symlink_to(metrics)                        # a .py name for a data file
    images.mkdir()
    rules = fw.Rules(repo=repo, results=results, data=repo/'data', images=images, pairs=[pair], subjects=['OASIS_OAS1_0438_MR1'])
    with fw.Spy(rules) as spy:
        open(repo/'data'/'three_d'/'stage_L2R_val_v1'/pair/'manifest.json').read()      # through the link: judged as its target, allowed
        for path in (repo/'data'/'three_d'/'stage_L2R_val_v1'/pair/'aligned_seg35.nii.gz', results/'pq_v1'/'notes.json', results/'pq_v1'/'innocent.json', repo/'configs'/'bayesreg'/'helper.py'):
            with pytest.raises(fw.Breach):
                open(path).read()
    record = spy.record()
    reasons = [b['reason'] for b in record['breaches']]
    assert len(reasons) == 4 and reasons[0].startswith('forbidden') and reasons[1] == 'outside the allowed items of C22.Q1-3' and reasons[2].startswith('forbidden') and reasons[3].startswith('forbidden')
    manifest = [f for f in record['files'] if f['path'].endswith('manifest.json')][0]
    assert manifest['in_scope'] and manifest['path'] == str((stage/'manifest.json').resolve()) and manifest['as_opened'] == str(repo/'data'/'three_d'/'stage_L2R_val_v1'/pair/'manifest.json')


def test_rules_record_is_json(tree):
    json.dumps(tree['rules'].describe())
