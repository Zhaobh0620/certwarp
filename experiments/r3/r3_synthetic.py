"""The synthetic world of R3 (the dry run, the smoke's GPU pass and the tests): experiments/unv2/synthetic_tree.py's subjects (ids OASIS_OAS1_09xx_MR1, a non-real data root), R2's four tiny UF
members and their eval_uf.json (run_r2.build_synthetic), two tiny L members trained by the chain itself (cli --profile v3_large), and a stand-in for nnU-Net: no nnunetv2 runs here.
  nn_train   writes what a finished nnUNetv2_train leaves for R3's checks (a checkpoint file of random bytes, a training log with the split, the seed line, 250 epochs, finite losses and the
             `predicting` line, a run record), then nn_r3.model_record -- the real record and the real checks follow;
  nn_infer   infer_r3nn.run itself with deterministic made-up predictors (a softmax of the image, per model), nibabel for the image and the label files, an infer_record.json of the real shape;
             then nn_r3.check_outputs, the real check (the orientation check through nx_seg.read_labels_file included).
synthetic_chain runs both chains of run_r3 on it (the training chain to the composition; the development chain on a composition with every part forced, so that each path is taken).
"""
import json
import os
from pathlib import Path
import sys
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent/'r2', HERE.parent/'nn', HERE.parent/'uf', HERE.parent/'unv2'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import r3_common as r3c  # noqa: E402

TINY_L = None                                   # set from run_r2.TINY_UF at first use (that module imports the R2 chain)
DEV_SOURCE = 'synthetic_dev'
FORCED = 'UF+L+N+D'                              # the development chain's composition in the synthetic world: every part taken


def tiny_l():
    import run_r2
    return dict(run_r2.TINY_UF, max_iters=6, min_iters=6, patience=6, max_wall_seconds=1800.)


# ---- the stand-in for nnU-Net ------------------------------------------------------------------------------------------------------------------------------
def nn_train(name, *, root):
    """The files a finished run leaves (fold directory, checkpoint, log, run record), then nn_r3.model_record."""
    import nn_r3
    root = Path(root)
    fold, seed = r3c.nn_model_of(name)
    fd = nn_r3.fold_dir(name, nn_r3.results_root(root))
    if (fd/'checkpoint_final.pth').exists():
        raise FileExistsError(f'{fd}: trained before')
    fd.mkdir(parents=True)
    (fd/'checkpoint_final.pth').write_bytes(os.urandom(4096))
    n_train, n_val = nn_r3.expected_split(root, fold)
    stamp = '2026-11-11 00:00:00.000000'
    lines = [f'{stamp}: R3: torch, numpy and random seeded with {seed} before the network is built (synthetic)', f'{stamp}: Using splits from existing split file: {nn_r3.splits_path(root)}',
             f'{stamp}: This split has {n_train} training and {n_val} validation cases.']
    for epoch in range(r3c.NN_EPOCHS):
        lines += [f'{stamp}: Epoch {epoch}', f'{stamp}: train_loss -0.5', f'{stamp}: val_loss -0.4', f'{stamp}: Epoch time: 1.0 s']
    lines.append('2026-11-11 00:05:00.000000: predicting OASIS_OAS1_0903_MR1')
    (fd/'training_log_2026_11_11_00_00_00.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    base = dict(model=name, trainer=r3c.nn_trainer(seed), fold=r3c.NN_FOLD_INDEX[fold], resumed=False, code_commit=None, started_utc='2026-11-11T06:00:00Z', synthetic=True)
    r3c.write_json_once(root/'train_runs'/name/'20261111T060000Z.started.json', base)
    r3c.write_json_once(root/'train_runs'/name/'20261111T060000Z.json', dict(base, outcome=dict(returncode=-15, stopped='condition', host_peak_pss_gib=.1, gpu_peak_used_mib=None), epoch_seconds=[1.]*r3c.NN_EPOCHS,
                                                                             stopped_before_final_validation=True))
    return nn_r3.model_record(name, root=root)


def fake_predictor(name):
    """predict(image (1, k, j, i), properties) -> (labels, probabilities (36, k, j, i) float32): a deterministic softmax of the image per model, not a segmentation."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    w, b = rng.normal(size=36), rng.normal(size=36)

    def predict(image, properties):
        x = np.asarray(image[0], dtype=np.float64)
        x = x/(np.abs(x).max() or 1.)
        z = 4*w[:, None, None, None]*x[None]+b[:, None, None, None]
        z -= z.max(axis=0)
        p = np.exp(z)
        p = (p/p.sum(axis=0)).astype(np.float32)
        return p.argmax(axis=0).astype(np.uint8), p
    return predict


def read_image(path):
    import nibabel as nib
    image = nib.load(str(path))
    return np.asarray(image.dataobj, dtype=np.float32).transpose(2, 1, 0)[None], dict(affine=image.affine), image.affine


def write_seg(labels, path, properties):
    import nibabel as nib
    nib.save(nib.Nifti1Image(np.asarray(labels).transpose(2, 1, 0).astype(np.uint8), properties['affine']), str(path))


def nn_infer(directory, subjects, models, *, root, data_root, records, write=None):
    """infer_r3nn.run on the made-up predictors, an infer_record.json of the real shape, nn_r3.check_outputs."""
    import infer_r3nn
    import nn_r3
    import run_nn
    directory = Path(directory)
    if (directory/'input').exists():
        raise FileExistsError(f'{directory}: inferred before')
    run_nn.build_dev_input(directory/'input', subjects, data_root)
    images = infer_r3nn.images_of(directory/'input')
    rows = infer_r3nn.run([(m, fake_predictor(m)) for m in models], images, directory, read_image=read_image, write_seg=write_seg if write is None else write, log=lambda text: None)
    document = dict(format=infer_r3nn.FORMAT, synthetic=True, models={m: dict(checkpoint_sha256=records[m]['final_checkpoint_sha256'], trainer=r3c.nn_trainer(r3c.nn_model_of(m)[1]),
                                                                            checkpoint_fold=r3c.NN_FOLD_INDEX[r3c.nn_model_of(m)[0]], allowed_mirroring_axes=None, tta_views=1,
                                                                            checkpoint_name='checkpoint_final.pth') for m in models},
                    subjects=list(rows), gpu_peak_reserved_gib=None)
    (directory/infer_r3nn.RECORD_FILE).write_text(json.dumps(document, indent=1)+'\n', encoding='utf-8')
    return nn_r3.check_outputs(directory, subjects, models, root=root, data_root=data_root, records=records, log=lambda text: None)


# ---- the world -------------------------------------------------------------------------------------------------------------------------------------------
def synthetic_switch(path, *, verdict='pass'):
    """The stand-in for R3b's decision record (r3_core.SYNTHETIC_DECISION_FORMAT: the verdict, this tree's decoder_spec and decode.py's sha256) written at `path`, and the switch with D on
    that r3_core.switch_document makes of it (the code of the real switch; the stub is allowed in the synthetic world only)."""
    import r3_core
    from bayesreg.seg.decode import decoder_spec
    record = dict(format=r3_core.SYNTHETIC_DECISION_FORMAT, verdict=verdict, decoder_spec=decoder_spec(), synthetic=True, **{r3c.DECISION_DECODE_KEY: r3c.sha256_file(r3_core.decode_module_path())})
    Path(path).write_text(json.dumps(record, indent=1)+'\n', encoding='utf-8')
    decision = r3_core.load_decision(path, synthetic=True)
    return r3_core.switch_document(dict(decision, path=str(path)), words='synthetic', synthetic=True)



def build(work, env, device='cpu', *, decode=True):
    """The world and the Spec of run_r3's chains with every option pointing into it."""
    import types
    import nn_r3
    import run_nn
    import run_r2
    import run_r3
    from bayesreg.seg.unet.data import load_full_folds
    from bayesreg.seg.unet.snapshot import CODE_ROOT
    work = Path(work)
    world = run_r2.build_synthetic(work, env, device)                  # UF's four tiny members, eval_uf.json, the subjects
    split = load_full_folds(work/'folds_full_v2.json')
    nnroot = work/'nnroot'
    cases = run_nn.preprocessed_data_directory(nnroot)
    cases.mkdir(parents=True)
    for subject in sorted(split.training_subjects):
        (cases/f'{subject}.pkl').write_bytes(b'')
    (nnroot/'prepare_manifest.json').write_text(json.dumps(dict(label_contact_subjects=sorted(split.training_subjects), note='synthetic'))+'\n', encoding='utf-8')
    splits_sha256 = nn_r3.write_splits(nnroot, split, identifiers=sorted(split.training_subjects))
    out = work/'out'
    out.mkdir(parents=True, exist_ok=True)
    if decode:
        switch = synthetic_switch(work/'decision_synthetic.json')
    else:
        switch = dict(schema='bayesreg.r3.decode_switch.v2', decode=False, words='synthetic')
    (out/r3c.DECODE_SWITCH).write_text(json.dumps(switch, indent=1)+'\n', encoding='utf-8')
    l_set = tiny_l()
    (work/'l_overrides.json').write_text(json.dumps(l_set)+'\n', encoding='utf-8')
    data, allow = work/'data', work/'dev_subjects.json'
    common = ['--synthetic', '--uf-root', str(world.uf_root), '--member-root', str(out), '--uf-eval', str(work/'eval_uf.json'), '--l-overrides', str(work/'l_overrides.json'), '--data-root', str(data), '--allowlist', str(allow)]
    control_extra = common+['--folds', str(work/'folds.json'), '--dev-source', DEV_SOURCE]
    eval_extra = control_extra+['--nn-dev', str(out/'nn_dev'), '--nn-root', str(nnroot), '--switch', str(out/r3c.DECODE_SWITCH)]
    compose_extra = common+['--nn-calib', str(out/'nn_calib'), '--nn-root', str(nnroot), '--switch', str(out/r3c.DECODE_SWITCH), '--folds', str(work/'folds_full_v2.json')]
    run_extra = ['--synthetic', '--nn-root', str(nnroot), '--data-root', str(data), '--folds', str(work/'folds_full_v2.json'), '--allowlist', str(allow), '--dev-source', DEV_SOURCE,
                 '--training-source', 'synthetic_bline_train']
    spec = run_r3.Spec(out=out, cwd=CODE_ROOT, python=sys.executable, device=device, inputs=world.inputs, l_overrides=l_set, run_extra=run_extra, compose_extra=compose_extra, eval_extra=eval_extra,
                       control_extra=control_extra, uf_root=world.uf_root, uf_pinned=None, nn_root=nnroot, gate=False, env=env, synthetic=True)
    return types.SimpleNamespace(spec=spec, work=work, data=data, allowlist=allow, uf_root=world.uf_root, split=split, nnroot=nnroot, splits_sha256=splits_sha256)


def force_composition(path, target, winner=FORCED):
    """A copy of composition.json with `winner` in place of the rule's (marked; the synthetic development chain only)."""
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    document.update(winner=winner, delivered=r3c.delivered(winner), no_change=False, forced_for_the_synthetic_chain=True)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text(json.dumps(document, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    return target


def synthetic_chain(work, env, device='cpu', log=print):
    """Both chains on the world; returns the problems found (empty: as planned)."""
    import r3_lock
    import run_r3
    problems = []
    saved = dict(r3_lock.PATHS)
    r3_lock.PATHS.update(calib=Path(work)/'never_taken_calib.lock', dev=Path(work)/'never_taken_dev.lock')      # the synthetic readers do not take them; pointed away all the same
    try:
        world = build(work, env, device)
        spec, out = world.spec, world.spec.out
        run_r3.set_run_log(out)
        plan = run_r3.members_plan(spec, world.split, splits_sha256=world.splits_sha256)
        (out/'hold').write_text('dry run: the hold file\n')
        try:
            run_r3.run_chain(spec, plan)
            problems.append('the hold file did not stop the chain')
        except run_r3.Stop as exc:
            log(f'hold: {exc}')
        (out/'hold').unlink()
        run_r3.run_chain(spec, plan)
        run_r3.run_chain(spec, plan)                                   # a second pass skips everything
        order = [line.split()[2].rstrip(':') for line in (out/'chain.log').read_text(encoding='utf-8').splitlines() if len(line.split()) > 2 and line.split()[1] == 'start']
        composition = json.loads((out/'compose'/'composition.json').read_text(encoding='utf-8'))
        control = json.loads((out/'control_check'/'eval_control.json').read_text(encoding='utf-8'))
        log('order of the training chain: '+' '.join(order))
        log('control (chain\'s first step): '+json.dumps(control['control']))
        log('composition (synthetic): '+json.dumps(dict(winner=composition['winner'], band=composition['result']['band'], excluded=composition['result']['excluded'], order=composition['order']), ensure_ascii=False))
        want = [n for n, _, _ in run_r3.chain_steps(spec, world.splits_sha256)]
        if order != want:
            problems.append(f'the synthetic training chain: order {order} is not {want}')
        if not control['control']['passed'] or composition['order'] != list(r3c.candidate_order(True)) or composition['winner'] not in composition['order']:
            problems.append('the synthetic training chain: control or composition')
        spec.composition_path = force_composition(out/'compose'/'composition.json', Path(work)/'forced'/'composition.json')
        run_r3.run_dev_chain(spec, plan)
        evaluation = json.loads((out/'eval_r3'/'eval_r3.json').read_text(encoding='utf-8'))
        decision = json.loads((out/'analysis'/'decision_r3.json').read_text(encoding='utf-8'))
        log('development chain (forced composition): arms '+' '.join(evaluation['arms'])+'; control '+json.dumps(evaluation['control'])+'; decision '+json.dumps({k: decision.get(k) for k in ('reading', 'why')}, ensure_ascii=False))
        if evaluation['arms'][:3] != ['UF4', 'COMBO', 'COMBO_noD'] or 'N4' not in evaluation['arms'] or not evaluation['control']['passed'] or decision['reading'] != 'undecidable' or decision['why'] != ['2 名受试者，不是 20']:
            problems.append('the synthetic development chain: arms, control or reading')
    finally:
        r3_lock.PATHS.clear()
        r3_lock.PATHS.update(saved)
    return problems
