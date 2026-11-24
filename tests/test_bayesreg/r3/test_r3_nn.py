"""N of R3 (experiments/r3/nn_r3.py, infer_r3nn.py, ext_trainers/, r3_synthetic.py's stand-in).  CPU; no nnunetv2 is imported or run here, no real file of a subject is read.

- the environment: NN's, with nnUNet_extTrainer at R3's trainers; the commands (fold index, the seeded trainer, --c);
- splits_final.json from the folds: fold 0 = A, fold 1 = B, train and val the fold's 98 + 8 (here the synthetic 5 + 1), written once, refused when it is not a partition;
- the training logs read (split, split file, seed, epochs, losses, the final validation begun) and the model's checks, one departure at a time; the smoke's rule;
- the inference child: it imports nothing of this repository; its copies of nx_seg's functions give nx_seg's numbers; run() on made-up predictors writes the mean (float16, the argmax of the
  float64 mean kept), the labels and nnU-Net's own label files; the check passes, and fails on a changed file, on labels that are not nnU-Net's (orientation), on another checkpoint.
"""
import ast
import os
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'r3', ROOT/'experiments'/'r2', ROOT/'experiments'/'nn', ROOT/'experiments'/'fl', ROOT/'experiments'/'ef', ROOT/'experiments'/'mc0', ROOT/'experiments'/'unv2', ROOT/'experiments'/'uf'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import infer_r3nn  # noqa: E402
import nn_common as nc  # noqa: E402
import nn_r3  # noqa: E402
import r3_common as r3c  # noqa: E402
import r3_synthetic  # noqa: E402


def build_tree(tmp_path):
    import synthetic_tree
    synthetic_tree.build(tmp_path/'tree')
    return tmp_path/'tree'


def test_the_environment_and_the_commands():
    env = nn_r3.nn_env(Path('/r'), gpu=False, base=dict(PATH='/usr/bin:/bin', HOME='/h'))
    assert env['nnUNet_extTrainer'] == str(ROOT/'experiments/r3/ext_trainers') and env['nnUNet_preprocessed'] == '/r/nnUNet_preprocessed' and env['CUDA_VISIBLE_DEVICES'] == '' and env['nnUNet_compile'] == 'f'
    assert nn_r3.train_argv('nnB_s3') == [str(nc.NN_BIN/'nnUNetv2_train'), '935', '3d_fullres', '1', '-tr', 'nnUNetTrainer_250epochs_NoMirroring_r3s3', '-p', 'nnUNetPlans', '-device', 'cuda']
    assert nn_r3.train_argv('nnA_s0', resume=True)[-1] == '--c'
    argv = nn_r3.infer_argv(['nnA_s0', 'nnA_s1'], input_dir=Path('/i'), out_dir=Path('/o'), root=Path('/r'))
    assert argv[:4] == [str(nc.NN_PY), '-I', '-B', str(ROOT/'experiments/r3/infer_r3nn.py')]
    assert argv[argv.index('--model')+1] == 'nnA_s0=/r/nnUNet_results/Dataset935_OASIS35/nnUNetTrainer_250epochs_NoMirroring_r3s0__nnUNetPlans__3d_fullres:0'
    assert nn_r3.fold_dir('nnB_s2', Path('/x')) == Path('/x/Dataset935_OASIS35/nnUNetTrainer_250epochs_NoMirroring_r3s2__nnUNetPlans__3d_fullres/fold_1')


def test_the_inference_child_imports_nothing_of_this_repository():
    tree = ast.parse((ROOT/'experiments/r3/infer_r3nn.py').read_text(encoding='utf-8'))
    modules = {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert modules <= {'argparse', 'copy', 'datetime', 'hashlib', 'io', 'json', 'os', 'pathlib', 'platform', 'sys', 'time', 'numpy', 'importlib', 'nibabel', 'torch', 'nnunetv2'}
    assert 'nnunetv2' not in sys.modules
    assert infer_r3nn.LIMIT_BYTES == 9*2**30 == int(r3c.NN_DEVICE_CAP_GIB*2**30)


def test_the_copies_are_nx_segs():
    import nx_seg
    rng = np.random.default_rng(3)
    affine = np.array([[-1, 0, 0, 80], [0, 0, 1, -112], [0, -1, 0, 96], [0, 0, 0, 1]], dtype=float)      # the OASIS affine (runs/20261105-nn/prepare_record.md)
    p = rng.dirichlet(np.ones(36)*.3, size=(5, 6, 7)).transpose(3, 0, 1, 2).astype(np.float32)
    assert np.array_equal(infer_r3nn.orient_probabilities(p, affine), nx_seg.orient_probabilities(p, affine))
    v = rng.integers(0, 36, size=(7, 6, 5)).astype(np.uint8)
    assert np.array_equal(infer_r3nn.orient_native(v, affine), nx_seg.orient_native(v, affine))
    q = p.astype(np.float64)
    q[:, 0, 0, 0] = 0
    q[3, 0, 0, 0] = q[5, 0, 0, 0] = .5
    labels = q.argmax(axis=0)
    a, na = infer_r3nn.float16_keeping_argmax(q, labels)
    b, nb = nx_seg.float16_keeping_argmax(q, labels)
    assert np.array_equal(a, b) and na == nb


def test_splits_from_the_folds(tmp_path):
    from bayesreg.seg.unet.data import load_full_folds
    work = build_tree(tmp_path)
    split = load_full_folds(work/'folds_full_v2.json')
    document = nn_r3.splits_document(split)
    assert [len(f['train']) for f in document] == [5, 5] and [len(f['val']) for f in document] == [1, 1]
    assert document[0]['val'] == split.subjects('A', 'calib') and document[1]['train'] == split.subjects('B', 'train')
    ids = sorted(split.training_subjects)
    assert nn_r3.splits_problems(document, ids) == []
    assert nn_r3.splits_problems(document, ids+['OASIS_OAS1_0999_MR1'])
    bad = json.loads(json.dumps(document))
    bad[1]['val'] = bad[0]['val']
    assert nn_r3.splits_problems(bad, ids)
    root = tmp_path/'nnroot'
    digest = nn_r3.write_splits(root, split, identifiers=ids)
    assert digest == r3c.sha256_file(nn_r3.splits_path(root)) and nn_r3.write_splits(root, split, identifiers=ids) == digest
    nn_r3.splits_path(root).write_text('[]\n')
    with pytest.raises(nn_r3.Stop):
        nn_r3.write_splits(root, split, identifiers=ids)
    with pytest.raises(nn_r3.Stop):
        nn_r3.write_splits(tmp_path/'other', split, identifiers=ids[1:])


@pytest.fixture
def trained(tmp_path):
    """A synthetic nnU-Net root with every model 'trained' by r3_synthetic.nn_train."""
    from bayesreg.seg.unet.data import load_full_folds
    work = build_tree(tmp_path)
    split = load_full_folds(work/'folds_full_v2.json')
    root = tmp_path/'nnroot'
    digest = nn_r3.write_splits(root, split, identifiers=sorted(split.training_subjects))
    records = {m: r3_synthetic.nn_train(m, root=root) for m in r3c.NN_NAMES}
    return dict(work=work, split=split, root=root, digest=digest, records=records)


def test_the_models_as_planned(trained):
    root, digest = trained['root'], trained['digest']
    for m, record in trained['records'].items():
        assert nn_r3.model_problems(record, m, splits_sha256=digest, root=root) == []
        assert record['splits'] == [(5, 1)] or record['splits'] == [[5, 1]]
    record = trained['records']['nnA_s0']
    for change, word in ((dict(seed=1), 'trainer/fold/seed'), (dict(splits=[[6, 1]]), 'split'), (dict(split_files=['/x']), 'split file'), (dict(seeds=[1]), 'seed line'),
                         (dict(epochs_logged=249), 'epochs'), (dict(losses_finite=False), 'loss'), (dict(stopped_before_final_validation=False), 'final validation'),
                         (dict(splits_sha256='0'*64), 'splits_final'), (dict(final_checkpoint_sha256='0'*64), 'checkpoint_final')):
        problems = nn_r3.model_problems(dict(record, **change), 'nnA_s0', splits_sha256=digest, root=root)
        assert problems and any(word in p for p in problems), (change, problems)
    assert nn_r3.model_problems(None, 'nnA_s0', splits_sha256=digest, root=root) == ['nnA_s0: no train_record.json']
    with pytest.raises(FileExistsError):
        r3_synthetic.nn_train('nnA_s0', root=root)


def test_the_logs_read(tmp_path):
    fd = tmp_path/'fold_0'
    fd.mkdir()
    text = ('2026-11-11 00:00:00.000000: R3: torch, numpy and random seeded with 2 before the network is built (x)\n2026-11-11 00:00:01.000000: Using splits from existing split file: /a/splits_final.json\n'
            '2026-11-11 00:00:02.000000: This split has 98 training and 8 validation cases.\n2026-11-11 00:01:00.000000: train_loss -0.61\n2026-11-11 00:01:00.000000: val_loss nan\n'
            '2026-11-11 00:01:00.000000: Epoch time: 81.5 s\n2026-11-11 00:02:00.000000: predicting OASIS_OAS1_0001_MR1\n')
    (fd/'training_log_x.txt').write_text(text)
    facts = nn_r3.log_facts(fd)
    assert facts['splits'] == [(98, 8)] and facts['split_files'] == ['/a/splits_final.json'] and facts['seeds'] == [2] and facts['epoch_seconds'] == [81.5] and facts['validating']
    assert facts['train_losses'] == [-.61] and np.isnan(facts['val_losses'][0]) and facts['wall_seconds'] == 120.


def test_the_smokes_rule(trained):
    root = trained['root']
    good = dict(model='nnA_s0', stopped='condition', epoch_seconds=[120., 82., 83., 84.], splits=[(5, 1)], split_files=[str(nn_r3.splits_path(root))], seeds=[0], train_losses=[-.1]*4, val_losses=[-.1]*4,
                host_peak_pss_gib=6., gpu_peak_used_mib=12000)
    reading, why, hours = nn_r3.smoke_reading(good, root=root)
    assert reading == 'pass' and why == [] and abs(hours-250*83/3600) < 1e-9
    for change in (dict(epoch_seconds=[120., 82., 83.]), dict(stopped='memory'), dict(splits=[(98, 8)]), dict(seeds=[1]), dict(val_losses=[float('nan')]), dict(epoch_seconds=[1, 116., 116., 116.]),
                   dict(host_peak_pss_gib=11.5), dict(gpu_peak_used_mib=None)):
        assert nn_r3.smoke_reading(dict(good, **change), root=root)[0] == 'stop', change


def test_the_inference_and_its_check(trained, tmp_path):
    work, root, records = trained['work'], trained['root'], trained['records']
    subjects = trained['split'].subjects('A', 'calib')
    out = tmp_path/'nn_calib'/'foldA'
    check = r3_synthetic.nn_infer(out, subjects, r3c.nn_models_of_fold('A'), root=root, data_root=work/'data', records=records)
    assert check['passed'] and all(r['n_reader_not_mean'] == 0 and set(r['n_model_not_own'].values()) == {0} for r in check['rows'].values())
    subject = subjects[0]
    with np.load(out/subject/'probs36.npz') as npz:
        p16 = npz['p']
    with np.load(out/subject/'labels.npz') as npz:
        labels = {k: npz[k] for k in npz.files}
    assert p16.dtype == np.float16 and sorted(labels) == ['mean', 'nnA_s0', 'nnA_s1'] and np.array_equal(p16.argmax(axis=0), labels['mean'])
    # the mean of the two models, recomputed from the made-up predictors and nx_seg's orientation
    import nibabel as nib
    import nx_seg
    image = nib.load(str(work/'data'/subject/'aligned_norm.nii.gz'))
    x = np.asarray(image.dataobj, dtype=np.float32).transpose(2, 1, 0)[None]
    ps = [r3_synthetic.fake_predictor(m)(x, {})[1] for m in ('nnA_s0', 'nnA_s1')]
    mean = (nx_seg.orient_probabilities(ps[0], image.affine).astype(np.float64)+nx_seg.orient_probabilities(ps[1], image.affine))/2
    assert np.array_equal(mean.argmax(axis=0), labels['mean']) and np.max(np.abs(p16.astype(np.float64)-mean)) < 1e-3
    assert nn_r3.read_check(out, subjects, r3c.nn_models_of_fold('A'))['passed']
    with pytest.raises(nn_r3.Stop):
        nn_r3.read_check(out, subjects, ['nnA_s1', 'nnA_s0'])
    with pytest.raises(FileExistsError):
        r3_synthetic.nn_infer(out, subjects, r3c.nn_models_of_fold('A'), root=root, data_root=work/'data', records=records)


@pytest.mark.parametrize('damage', ['probs', 'labels', 'checkpoint'])
def test_the_check_fails(trained, tmp_path, damage):
    work, root, records = trained['work'], trained['root'], dict(trained['records'])
    subjects = trained['split'].subjects('B', 'calib')
    out = tmp_path/'nn_calib'/'foldB'
    r3_synthetic.nn_infer(out, subjects, r3c.nn_models_of_fold('B'), root=root, data_root=work/'data', records=records)
    (out/'check.json').rename(out/'check_first.json')
    subject = subjects[0]
    if damage == 'probs':
        with np.load(out/subject/'probs36.npz') as npz:
            p = npz['p'].copy()
        np.savez_compressed(out/subject/'probs36.npz', p=p[:, ::-1])
    elif damage == 'labels':                      # not nnU-Net's own labels (on real data: a wrong orientation; the synthetic volumes may be symmetric, so every label is moved)
        with np.load(out/subject/'labels.npz') as npz:
            labels = {k: npz[k] for k in npz.files}
        labels['nnB_s2'] = ((labels['nnB_s2'].astype(np.int64)+1) % 36).astype(np.uint8)
        np.savez_compressed(out/subject/'labels.npz', **labels)
        row = json.loads((out/subject/'subject.json').read_text())
        row['labels_sha256'] = r3c.sha256_file(out/subject/'labels.npz')
        (out/subject/'subject.json').write_text(json.dumps(row))
    else:
        records['nnB_s3'] = dict(records['nnB_s3'], final_checkpoint_sha256='f'*64)
    with pytest.raises(nn_r3.Stop):
        nn_r3.check_outputs(out, subjects, r3c.nn_models_of_fold('B'), root=root, data_root=work/'data', records=records, log=lambda text: None)
    assert json.loads((out/'check.json').read_text())['passed'] is False


def test_the_child_refuses_before_nnunetv2(tmp_path):
    (tmp_path/'in').mkdir()
    (tmp_path/'in'/'x.nii.gz').write_text('')
    assert infer_r3nn.main(['--model', 'a=/m:0', '--input-dir', str(tmp_path/'in'), '--out-dir', str(tmp_path/'o')]) == 8
    (tmp_path/'in'/'x.nii.gz').rename(tmp_path/'in'/'OASIS_OAS1_0901_MR1_0000.nii.gz')
    assert infer_r3nn.main(['--model', 'a=/m:0', '--input-dir', str(tmp_path/'in'), '--out-dir', str(tmp_path/'o')]) == 8      # no checkpoint
    assert infer_r3nn.parse_model('nnA_s0=/a/b:c:1') == ('nnA_s0', Path('/a/b:c'), 1) and infer_r3nn.parse_model('x=/a:all')[2] == 'all'
    with pytest.raises(ValueError):
        infer_r3nn.parse_model('x=/a')
    assert 'nnunetv2' not in sys.modules


# ---- the revision round (the review's I2, m1, m2, m3, m7 a b, m9) ------------------------------------------------------------------------------------------
def asymmetric_world(tmp_path, subject='OASIS_OAS1_0951_MR1'):
    """One synthetic subject whose image has no symmetry (a ramp along every axis plus noise), with the OASIS affine: a flipped or transposed label file cannot match it."""
    import nibabel as nib
    data = tmp_path/'data'
    (data/subject).mkdir(parents=True)
    rng = np.random.default_rng(7)
    i, j, k = np.meshgrid(np.arange(12), np.arange(10), np.arange(8), indexing='ij')
    values = (3*i+2*j+k+rng.normal(size=i.shape)).astype(np.float32)
    affine = np.array([[-1, 0, 0, 80], [0, 0, 1, -112], [0, -1, 0, 96], [0, 0, 0, 1]], dtype=float)
    nib.save(nib.Nifti1Image(values, affine), str(data/subject/'aligned_norm.nii.gz'))
    return data, subject


def test_a_wrong_orientation_fails_the_check(tmp_path):
    """The review's m7 (a): nnU-Net's own label file written flipped along x (as a wrong axis order would leave it) on an image with no symmetry: the orientation check stops."""
    data, subject = asymmetric_world(tmp_path)
    records = {'nnA_s0': dict(final_checkpoint_sha256='a'*64)}

    def flipped(labels, path, properties):
        r3_synthetic.write_seg(np.asarray(labels)[:, :, ::-1], path, properties)
    out = tmp_path/'set'
    with pytest.raises(nn_r3.Stop, match='orientation'):
        r3_synthetic.nn_infer(out, [subject], ['nnA_s0'], root=tmp_path/'nnroot', data_root=data, records=records, write=flipped)
    check = json.loads((out/'check.json').read_text())
    assert check['passed'] is False and any('orientation' in p for p in check['problems'])
    out = tmp_path/'set_ok'
    assert r3_synthetic.nn_infer(out, [subject], ['nnA_s0'], root=tmp_path/'nnroot', data_root=data, records=records)['passed']
    assert nn_r3.image_grid(data/subject/'aligned_norm.nii.gz') == np.load(out/subject/'labels.npz')['mean'].shape


def test_the_read_back_check_fails_on_its_own(tmp_path):
    """The review's m7 (b): probs36.npz changed and subject.json re-hashed, so that only the reader's argmax against the mean's argmax can stop it."""
    data, subject = asymmetric_world(tmp_path)
    records = {'nnA_s0': dict(final_checkpoint_sha256='a'*64)}
    out = tmp_path/'set'
    r3_synthetic.nn_infer(out, [subject], ['nnA_s0'], root=tmp_path/'nnroot', data_root=data, records=records)
    (out/'check.json').rename(out/'check_first.json')
    with np.load(out/subject/'probs36.npz') as npz:
        p = npz['p'].copy()
    p = np.roll(p, 1, axis=0)                                                                         # every voxel's argmax moves by one class; the sums stay
    np.savez_compressed(out/subject/'probs36.npz', p=p)
    row = json.loads((out/subject/'subject.json').read_text())
    row['probs36_sha256'] = r3c.sha256_file(out/subject/'probs36.npz')
    (out/subject/'subject.json').write_text(json.dumps(row))
    with pytest.raises(nn_r3.Stop):
        nn_r3.check_outputs(out, [subject], ['nnA_s0'], root=tmp_path/'nnroot', data_root=data, records=records, log=lambda text: None)
    check = json.loads((out/'check.json').read_text())
    assert any("reader's argmax" in p for p in check['problems'])


def test_the_final_validation_must_not_have_run(trained):
    root, digest = trained['root'], trained['digest']
    record = trained['records']['nnA_s1']
    assert record['validation_files'] == [] and nn_r3.model_problems(record, 'nnA_s1', splits_sha256=digest, root=root) == []
    summary = Path(record['final_checkpoint']).parent/'validation'/'summary.json'
    summary.parent.mkdir()
    summary.write_text('{}')
    assert any('summary.json' in p for p in nn_r3.model_problems(record, 'nnA_s1', splits_sha256=digest, root=root))
    summary.unlink()
    assert any('summary.json' in p for p in nn_r3.model_problems(dict(record, validation_files=['summary.json']), 'nnA_s1', splits_sha256=digest, root=root))


def test_the_training_is_stopped_at_its_wall(tmp_path, monkeypatch):
    """The review's m2: nnUNetv2_train runs in its own session; train_model stops it itself past the wall, records it, and stops the chain."""
    monkeypatch.setattr(nn_r3, 'train_argv', lambda name, resume=False, device='cuda': [sys.executable, '-c', 'import time; time.sleep(60)'])
    with pytest.raises(nn_r3.Stop, match='past its wall'):
        nn_r3.train_model('nnA_s0', root=tmp_path, env=dict(os.environ), device='cpu', poll=.1, wall_seconds=.5, log=lambda text: None)
    run = [json.loads(p.read_text()) for p in (tmp_path/'train_runs'/'nnA_s0').glob('*.json') if not p.name.endswith('.started.json')]
    assert len(run) == 1 and run[0]['stopped_at_the_wall'] is True and run[0]['outcome']['stopped'] == 'condition' and run[0]['outcome']['wall_seconds'] < 40


def test_the_inference_is_stopped_at_its_inner_wall(tmp_path):
    """The review's m9: the inner wall of a set is shorter than the step's limit and stops the child itself (run_watched); the link refusal names the set's subjects."""
    import run_r3
    assert r3c.NN_INFER_WALL_SECONDS < run_r3.TIMEOUTS['nn_infer'] and r3c.NN_WALL_SECONDS < run_r3.TIMEOUTS['nn_train']
    data, subject = asymmetric_world(tmp_path)
    with pytest.raises(nn_r3.Stop, match='stopped'):
        nn_r3.infer_set(tmp_path/'set', [subject], ['nnA_s0'], root=tmp_path, env=dict(os.environ), data_root=data, argv=[sys.executable, '-c', 'import time; time.sleep(60)'],
                        wall_seconds=.5, poll=.1, log=lambda text: None)
    assert json.loads((tmp_path/'set'/'infer_run.json').read_text())['outcome']['stopped'] == 'condition'
    with pytest.raises(nn_r3.Stop, match="the set's subjects"):
        nn_r3.link_images(tmp_path/'links', ['OASIS_OAS1_0952_MR1'], data)


def test_the_smokes_inference_command_and_rule(tmp_path):
    import smoke_r3
    argv = nn_r3.infer_argv(['nnA_s0'], input_dir=tmp_path/'i', out_dir=tmp_path/'o', results=tmp_path/'smoke_results', checkpoint=nn_r3.SMOKE_CHECKPOINT)
    assert argv[argv.index('--checkpoint-name')+1] == 'checkpoint_best.pth' and argv[argv.index('--model')+1].startswith(f"nnA_s0={tmp_path/'smoke_results'}/")
    assert '--checkpoint-name' not in nn_r3.infer_argv(['nnA_s0'], input_dir=tmp_path/'i', out_dir=tmp_path/'o')
    good = dict(problems=[], passed=True, rows={'s': dict(n_reader_not_mean=0, n_model_not_own={'nnA_s0': 0}, n_voxels=1000)}, run=dict(host_peak_pss_gib=3.))
    assert smoke_r3.inference_reading(good) == []
    for change in (dict(problems=['Stop: x']), dict(passed=False), dict(rows={}), dict(rows={'s': dict(n_reader_not_mean=1, n_model_not_own={'nnA_s0': 0}, n_voxels=1000)}),
                   dict(rows={'s': dict(n_reader_not_mean=0, n_model_not_own={'nnA_s0': 1}, n_voxels=1000)}), dict(run=dict(host_peak_pss_gib=None)), dict(run=dict(host_peak_pss_gib=12.))):
        assert smoke_r3.inference_reading(dict(good, **change)), change
    assert smoke_r3.inference_reading(None) == ['not run']
    assert smoke_r3.smoke_infer(tmp_path)['problems'][0].endswith('no such file after the 4 epochs')


def test_the_calibration_sets_pass_the_guard(trained, monkeypatch):
    """The review's m3: the calibration subjects are admitted (LabelGuard.check, refuse_sealed) before any link; a subject outside the training set among them is refused."""
    import run_r3
    import smoke_r3
    from bayesreg.io.guard import load_dev_allowlist
    from bayesreg.seg.unet.data import LabelAccessRefused
    allowlist = load_dev_allowlist(trained['work']/'dev_subjects.json')
    split = trained['split']
    subjects, models = run_r3.nn_subjects('calibB', split=split, code={}, nn_root=trained['root'], allowlist=allowlist, training_source='synthetic_bline_train')
    assert subjects == split.subjects('B', 'calib') and models == ['nnB_s2', 'nnB_s3']
    assert smoke_r3.smoke_subject(split, allowlist, training_source='synthetic_bline_train') == split.subjects('A', 'train')[0]
    original = type(split).subjects

    def subjects_with_a_stranger(self, fold, role=None):
        out = original(self, fold, role)
        return out+['OASIS_OAS1_0907_MR1'] if role == 'calib' else out                                    # a development subject of the synthetic world
    monkeypatch.setattr(type(split), 'subjects', subjects_with_a_stranger)
    with pytest.raises(LabelAccessRefused):
        run_r3.nn_subjects('calibA', split=split, code={}, nn_root=trained['root'], allowlist=allowlist, training_source='synthetic_bline_train')
