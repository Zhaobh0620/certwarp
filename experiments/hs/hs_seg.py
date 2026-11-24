"""HS, the seg phase (runs/20261106-hs/任务书.md, plan.md): the candidate of UF registered as a structure-layer segmenter and its predictions of the 20 development subjects, in the layout of /home/rmcfz/seq_vp_tag_results/seg/unet35_v1/pred.
Only images are read for the predictions; the truth only in `segcheck`, the control against UF's own evaluation.  Nothing here writes under UF's root (seg/unet35_v4_full) or nnunet_v1: they are only read.

stage     the registry document of the candidate, from UF's four members (bayesreg.seg.unet.registry.verify_members, the member check of every registration, on the pinned folds_full_v2.json; then each member.pt against the sha256 eval_uf.json evaluated) and, for (b), nnU-Net's
          checkpoint_final.pth (re-hashed against nnunet_v1/train_record.json and UF's convert_record.json; its label contact checked as NX's, experiments/fl/nx_seg.contact_problems).  The bytes go to <seg dir>/registry_staged.json, once; `register` later writes the same bytes to
          configs/bayesreg/segmenters_hs_v1.json, after the independent review (the brief: the registration counts only after the review).
infer     every subject, UF4's probabilities with the code of eval_uf.py (infer.load_member, predict_logits with each member's sliding and Gaussian window, ensemble_probabilities, the members in eval_uf's order A0 A1 B0 B1), as float16 the way v2 stores probs36;
          (b) the composer: 1/2 x UF4's float16 read back (float64 divided by the channel sum) + 1/2 x nnU-Net's stored probabilities read back (UF's files, pinned by its convert_record.json), the arithmetic of eval_uf.half_and_half; then float16 with the argmax kept (nx_seg.float16_keeping_argmax,
          one ULP where float16 would make a tie).  The reader (fast_channels.raw_labels) reads every stored file back: its argmax must be the composer's on every voxel, the group sums within make_inputs.GROUP_TOL.  provenance.json last.  A subject that fails keeps its files without provenance.json
          (a structure layer refuses such a directory: evidence kept, nothing readable) and the unit stops.
segcheck  the truth of the 20 subjects, after every check of eval_uf (admission, the predictions' provenance and hashes): the registered labels' rows (eval_seg.summarize_subject, eval_nn.metrics_of, eval_uf's functions) against eval_uf.json's rows of the candidate's arm.
          Passes: the mean seg35 and seg4 of the 20 subjects within SEGCHECK_TOL (2e-4, UF's U1_control: a GPU forward pass) of eval_uf's.  Whether every row is equal and the count of voxels where (b) differs from (a) against eval_uf's per_subject_extra are recorded.
          eval_uf.py kept no label map, so "the same labels as UF's evaluation, voxel for voxel" (the brief) can only be seen through these numbers: a deviation of this package, for the lead (plan.md).
register  by hand, after the review: the staged bytes into configs/bayesreg/segmenters_hs_v1.json, once.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _p in (HERE, EXPERIMENTS/'fl', EXPERIMENTS/'ef', EXPERIMENTS/'mc0', EXPERIMENTS/'nn', EXPERIMENTS/'uf', EXPERIMENTS/'unv2'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fast_channels as fast  # noqa: E402
import hs_core as hc  # noqa: E402
import mc0_core as core  # noqa: E402
import nx_seg  # noqa: E402

PROBS_FILE, GROUPS_FILE, PROVENANCE_FILE = nx_seg.PROBS_FILE, nx_seg.GROUPS_FILE, nx_seg.PROVENANCE_FILE
IMAGE_FILE = nx_seg.IMAGE_FILE
N_CLASSES = 36
SLAB = 8                                        # eval_uf.SLAB
GPU_CAP_GIB = 20.                               # eval_uf.GPU_CAP_GIB
CHECK_FAILED = 'check_failed.json'
READ_BACK = ('fast_channels.raw_labels (the reader of the Hx line) and bayesreg.seg.channels.read_back_probabilities: the raw channel sum on every voxel before the division (|sum - 1| <= 1e-3 for float16), then float64 divided by the channel sum')


class HSSegFailed(Exception):
    """The registration or a prediction is not what the plan says; kept, not rerun."""


def sha256_file(path):
    return nx_seg.file_sha256(path)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write_json_once(path, document):
    """Written to <path>.tmp, then moved; a file that is there is never overwritten (FileExistsError)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists; never overwritten')
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(document, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    tmp.replace(path)
    return path


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


# ---- the registry document ----------------------------------------------------------------------------------------------------------------------
def uf_evaluated_members(evaluation):
    """{member name: member.pt sha256} that eval_uf.py evaluated (eval_uf.json 'members'); HSSegFailed if it is not the four of UF."""
    members = (evaluation or {}).get('members') or {}
    want = sorted(hc.member_name(f, s) for f, s in hc.UF_MEMBERS)
    if sorted(members) != want or any(not (members[n] or {}).get('member_sha256') for n in want):
        raise HSSegFailed(f'eval_uf.json does not name the four UF members {want}: {sorted(members)}')
    return {n: members[n]['member_sha256'] for n in want}


def nn_checkpoint(nn_out, convert_record):
    """(sha256, train record): nnU-Net's checkpoint_final.pth re-hashed against nnunet_v1/train_record.json; it must be the one UF's conversion recorded."""
    train = read_json(Path(nn_out)/'train_record.json')
    if train is None:
        raise HSSegFailed(f'{nn_out}/train_record.json: no such file')
    if sha256_file(train['final_checkpoint']) != train['final_checkpoint_sha256']:
        raise HSSegFailed(f"{train['final_checkpoint']}: sha256 differs from train_record.json")
    if (convert_record or {}).get('checkpoint_sha256') != train['final_checkpoint_sha256']:
        raise HSSegFailed("UF's convert_record.json names other nnU-Net weights than nnunet_v1/train_record.json")
    return train['final_checkpoint_sha256'], train


def registry_document(candidate, *, members, nn_sha256=None, nn_contact=None):
    """The schema bayesreg.segmenters.v1 for the candidate: build_entries of the UF members (full training set: each fold's seeds sorted, the two contacts equal) and, for (b), nnU-Net as the fold-2 model.  members: verify_members' list."""
    from bayesreg.io.pair import REGISTRY_SCHEMA, subject_list_sha256
    from bayesreg.seg.unet.registry import build_entries
    c = hc.CANDIDATES[candidate]
    entries = build_entries(members, segmenter_id=c['segmenter_id'], prefix=c['prefix'], seeds=(0, 1), full_training_set=True)
    if sorted(entries) != sorted([hc.model_ids(candidate)['A'], hc.model_ids(candidate)['B']]) or any(len(e['member_model_sha256']) != 2 for e in entries.values()):
        raise HSSegFailed(f'the UF members do not make two fold models of two seeds each: {sorted(entries)}')
    if c['nn_model'] is not None:
        if nn_sha256 is None or nn_contact is None:
            raise HSSegFailed('(b) needs the nnU-Net checkpoint and its label contact')
        contact = sorted(nn_contact)
        if set(contact) != set(entries[hc.model_ids(candidate)['A']]['label_contact_subjects']):
            raise HSSegFailed("nnU-Net's label contact is not the UF members' (the training set, 106 subjects)")
        entries[c['nn_model']] = dict(segmenter_id=c['segmenter_id'], fold=c['nn_fold'], member_model_sha256=[nn_sha256], label_contact_subjects=contact, label_contact_subjects_sha256=subject_list_sha256(contact))
    comment = (f"structure-layer models of HS (runs/20261106-hs/plan.md): {c['words']}; the UF members of seg/unet35_v4_full (v3_large trained to its last step, runs/20261105-uf)"
               + ('' if c['nn_model'] is None else ", and nnU-Net's checkpoint_final.pth of nnunet_v1 as the fold-2 model (fold all), weighted 1/2 against the four members' mean") + '; append-only; written by experiments/hs/hs_seg.py')
    return dict(schema=REGISTRY_SCHEMA, comment=comment, models={m: entries[m] for m in sorted(entries)})


def document_text(document):
    """The bytes of the file, as nx_seg.register writes them."""
    return json.dumps(document, indent=1, ensure_ascii=False)+'\n'


def validated(text, where):
    """The SegmenterRegistry of these bytes, through the schema's own reader (on a temporary file next to `where`)."""
    from bayesreg.io.pair import load_segmenter_registry
    tmp = Path(str(where)+'.check.tmp')
    tmp.write_text(text, encoding='utf-8')
    try:
        return load_segmenter_registry(tmp)
    finally:
        tmp.unlink()


def stage(candidate, *, seg_dir, uf_out, evaluation, split, allowlist, training_source, pinned_folds_sha256, nn_out, sealed=None, contact_check=True):
    """The staged registry: verify_members on UF's root (every check of a registration), the member.pt files those eval_uf.json evaluated, for (b) nnU-Net's checkpoint and contact.  Written once; a file that is there must have the same bytes.  Returns (registry, record)."""
    from bayesreg.seg.unet.registry import verify_members
    members, check = verify_members(uf_out, split=split, allowlist=allowlist, training_source=training_source, pinned_folds_sha256=pinned_folds_sha256)
    found = {hc.member_name(m['fold'], m['seed']): m['member_sha256'] for m in members}
    evaluated = uf_evaluated_members(evaluation)
    if found != evaluated:
        raise HSSegFailed(f'the members under {uf_out} are not those eval_uf.json evaluated: {sorted(found)} against {sorted(evaluated)}')
    nn_sha = nn_contact = nn_record = None
    if hc.CANDIDATES[candidate]['nn_model'] is not None:
        convert = read_json(Path(uf_out)/hc.UF_NN_REL/'convert_record.json')
        nn_sha, train = nn_checkpoint(nn_out, convert)
        manifest = read_json(Path(nn_out)/'prepare_manifest.json') or {}
        nn_contact = manifest.get('label_contact_subjects') or []
        if contact_check:
            problems = nx_seg.contact_problems(nn_contact, allowlist=allowlist, sealed=sealed, training_subjects=sorted(set(split.training_subjects)))
            if problems:
                raise HSSegFailed(f"nnU-Net's label contact: {problems}")
        nn_record = dict(checkpoint_sha256=nn_sha, trainer=train.get('trainer'), checkpoint=train.get('final_checkpoint'), n_label_contact=len(nn_contact), convert_record_sha256=sha256_file(Path(uf_out)/hc.UF_NN_REL/'convert_record.json'))
    document = registry_document(candidate, members=members, nn_sha256=nn_sha, nn_contact=nn_contact)
    text = document_text(document)
    seg_dir = Path(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    path = seg_dir/hc.STAGED_REGISTRY
    validated(text, path)
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise HSSegFailed(f'{path} is there with other content: a staged registry is written once')
    else:
        tmp = Path(str(path)+'.tmp')
        tmp.write_text(text, encoding='utf-8')
        tmp.replace(path)
    from bayesreg.io.pair import load_segmenter_registry
    registry = load_segmenter_registry(path)
    return registry, dict(path=str(path), sha256=registry.sha256, candidate=candidate, segmenter_id=hc.CANDIDATES[candidate]['segmenter_id'], models=sorted(registry.models), member_check=check, evaluated_members=evaluated, nn=nn_record)


# ---- the probabilities ---------------------------------------------------------------------------------------------------------------------------
def uf_plan(registry, candidate, subject, allowlist, available):
    """[(model id, member sha256, path)] of the UF members, in eval_uf's order (fold models sorted, seeds in order: A0 A1 B0 B1), after infer.select_fold_models (D24: all_folds_mean) and the admission of the subject."""
    from bayesreg.seg.unet.infer import check_subject_admitted, select_fold_models
    check_subject_admitted(subject, allowlist)
    c = hc.CANDIDATES[candidate]
    models, policy = select_fold_models(registry, subject, allowlist, segmenter_id=c['segmenter_id'])
    if policy != 'all_folds_mean' or sorted(models) != sorted(registry.models):
        raise HSSegFailed(f'{subject}: policy {policy} with {sorted(models)}: a development subject is segmented by every registered model (D24, all_folds_mean)')
    plan = []
    for model_id in sorted(m for m in models if m != c['nn_model']):
        for sha in models[model_id]['member_model_sha256']:
            if sha not in available:
                raise HSSegFailed(f'member {sha[:12]}... of {model_id} not found')
            plan.append((model_id, sha, available[sha]))
    return plan, models, policy


def uf4_probabilities(subject, *, files, registry, models, plan, device):
    """(float32 numpy (36, nz, ny, nx), image metadata, normalization, member records): eval_uf.py's (a), member after member (one member's logits alive at a time)."""
    import torch
    from bayesreg.seg.unet.data import normalize_image, read_image
    from bayesreg.seg.unet.infer import ensemble_probabilities, load_member
    from bayesreg.seg.unet.train import predict_logits
    values, meta = read_image(files, subject)
    state, members = {}, []

    def member_logits():
        for model_id, sha, path in plan:
            model, blob = load_member(path, sha, device, entry=models[model_id])
            if 'norm' not in state:
                state['image'], state['norm'] = normalize_image(values, percentile=blob['percentile'])
            elif blob['percentile'] != state['norm']['percentile']:
                raise HSSegFailed(f'{path}: the members disagree on the normalization percentile')
            if blob.get('window_gaussian_sigma') is None or not blob.get('window'):
                raise HSSegFailed(f'{path}: member.pt has no sliding window with a Gaussian window (eval_uf evaluated them with it)')
            z = predict_logits(model, state['image'], device=device, amp=blob['amp'], window=tuple(blob['window']), gaussian_sigma=blob['window_gaussian_sigma'])
            members.append(dict(model_id=model_id, fold=blob['fold'], seed=blob['seed'], member_sha256=sha, temperature=blob['temperature'], checkpoint_choice=blob.get('checkpoint_choice'), checkpoint_iteration=blob.get('checkpoint_iteration'),
                                window=list(blob['window']), window_gaussian_sigma=blob['window_gaussian_sigma'], percentile=blob['percentile'], amp=blob['amp']))
            del model
            yield z, blob['temperature']
    probs = ensemble_probabilities(member_logits()).cpu().numpy()
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(torch.device(device))
    return probs, meta, state['norm'], members


def read_back(path, shape):
    """bayesreg.seg.channels.read_back_probabilities: float64 divided by the channel sum, after the raw-sum check."""
    from bayesreg.seg.channels import read_back_probabilities
    array, record = read_back_probabilities(path, shape=shape)
    return array, record


def mix_in_place(p16, p_nn, *, weight=hc.NN_WEIGHT, slab=SLAB):
    """(b): p_nn becomes (1 - weight) * P_UF + weight * P_nn, with P_UF = the float16 of UF4 read back (float64 divided by the channel sum), by slabs of z: the arithmetic of eval_uf.half_and_half, element by element, so its argmax is that function's.  weight is the plan's 1/2 and nothing else."""
    if weight != hc.NN_WEIGHT:
        raise ValueError(f'the weight of (b) is fixed at {hc.NN_WEIGHT}')
    if p16.dtype != np.float16 or p16.shape != p_nn.shape or p_nn.dtype != np.float64:
        raise ValueError(f'UF4 float16 and nnU-Net float64 of one shape required: {p16.dtype} {p16.shape}, {p_nn.dtype} {p_nn.shape}')
    for z0 in range(0, p16.shape[1], slab):
        part = p16[:, z0:z0+slab].astype(np.float64)
        part /= part.sum(axis=0)
        p_nn[:, z0:z0+slab] = (1-weight)*part+weight*p_nn[:, z0:z0+slab]
    return p_nn


def b_differs_from_a(p16, p_nn_read_back, *, slab=SLAB):
    """The count eval_uf.json keeps per subject (per_subject_extra.voxels_b_differs_from_a): voxels where (b)'s argmax is not (a)'s.  p_nn_read_back is not changed."""
    a = p16.argmax(axis=0)
    n = 0
    for z0 in range(0, p16.shape[1], slab):
        part = p16[:, z0:z0+slab].astype(np.float64)
        part /= part.sum(axis=0)
        b = ((1-hc.NN_WEIGHT)*part+hc.NN_WEIGHT*p_nn_read_back[:, z0:z0+slab]).argmax(axis=0)
        n += int((b != a[z0:z0+slab]).sum())
    return n


# ---- one prediction on disk ---------------------------------------------------------------------------------------------------------------------
def save_npz(path, array):
    nx_seg.save_npz(path, array)


def write_prediction(subject, probs, labels, *, out_root, grouping, provenance, group_tol=None):
    """probs36.npz (float16 with the argmax `labels` kept), structure_groups.npz, read back by the Hx line's reader, then provenance.json last.  Returns the subject's record.  Never overwritten.
    A check that fails keeps the two files without provenance.json (no structure layer reads such a directory), writes check_failed.json and raises HSSegFailed."""
    from bayesreg.seg.unet.groups import group_probabilities
    group_tol = core.make_inputs().GROUP_TOL if group_tol is None else group_tol
    started = time.perf_counter()
    out = Path(out_root)/subject
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} exists and is not empty; predictions are never overwritten')
    p16, n_ties = nx_seg.float16_keeping_argmax(probs, labels)
    groups = group_probabilities(probs, grouping, dtype=np.float32)
    out.mkdir(parents=True, exist_ok=True)
    save_npz(out/PROBS_FILE, p16)
    save_npz(out/GROUPS_FILE, groups.astype(np.float16))
    shape, groups_shape = p16.shape, groups.shape
    del p16, groups
    array, _ = read_back(out/GROUPS_FILE, groups_shape)
    read = fast.raw_labels(out/PROBS_FILE, shape, array, (grouping.background_labels,)+tuple(grouping.group_labels))
    problems = []
    if read is None:
        problems.append(f'the reader refuses its fast path on {PROBS_FILE}')
        n_not_own, deviation, record = None, None, {}
    else:
        back, _, record, deviation = read
        n_not_own = int((back != labels).sum())
        if n_not_own:
            problems.append(f'the reader reads {n_not_own} voxels other than the argmax of the probabilities')
        if not deviation <= group_tol:
            problems.append(f'probs36 summed by group differs from structure_groups.npz by {deviation:.3e} > {group_tol:g}')
    output = {PROBS_FILE: sha256_file(out/PROBS_FILE), GROUPS_FILE: sha256_file(out/GROUPS_FILE)}
    row = dict(subject=subject, n_voxels=int(labels.size), n_float16_ties_fixed=n_ties, n_reader_not_own=n_not_own, raw_max_abs_sum_deviation=record.get('raw_max_abs_sum_deviation'), group_max_abs_deviation=deviation, group_tol=group_tol, output_sha256=output)
    if problems:
        write_json_once(out/CHECK_FAILED, dict(row, problems=problems, utc=utc(), note='kept without provenance.json: no structure layer reads this directory; not rerun'))
        raise HSSegFailed(f'{subject}: {problems}')
    seconds = time.perf_counter()-started
    document = dict(provenance, input_sha256=dict(provenance['input_sha256']), output_sha256=output, outputs=dict(probs36=dict(file=PROBS_FILE, key='p', dtype='float16', shape=list(shape), classes='seg35 label c at channel c (0 = background)'),
                    groups=dict(file=GROUPS_FILE, key='p', dtype='float16', shape=list(groups_shape), channels='0 = background, k = class_names[k-1]'), layout='(channels, nz, ny, nx), jd_curl.three_d.nifti.load_scalar storage (RAS, zyx)', read_back=READ_BACK), seconds_write=seconds)
    (out/PROVENANCE_FILE).write_text(json.dumps(document, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    return dict(row, seconds_write=seconds)


def _bayesreg_version():
    from bayesreg import __version__
    return __version__


def provenance_of(subject, *, candidate, registry, models, policy, members, meta, norm, grouping, grouping_check, registry_rel, staged_sha256, code, environment, command, composition, seconds, allowlist=None):
    """What StructureLayer and validate_segmentation_provenance ask of a prediction, with the names of bayesreg.seg.unet.infer.segment_subject.  members: every registered member (UF's four, and nnU-Net for (b)), with what each is.
    The member records carry checkpoint_choice and checkpoint_iteration (the weights of the last step) instead of best_iteration (04_分割器.md §16 未决 2: in a member of the last step that key names the step the weights are from, not the best calib evaluation)."""
    ids = sorted(models)
    return dict(layer='structure', segmenter_id=hc.CANDIDATES[candidate]['segmenter_id'], model_ids=ids, model_sha256={m: list(models[m]['member_model_sha256']) for m in ids}, folds={m: models[m]['fold'] for m in ids}, trained_with_labels=True, test_labels_seen=False,
                subject=subject, policy=policy, members=members, **grouping.record(), grouping_check=grouping_check, input_sha256={IMAGE_FILE: meta['sha256']},
                grid=dict(shape_zyx=meta['grid_shape_zyx'], spacing_xyz=meta['spacing_xyz'], origin_xyz=meta['origin_xyz']), normalization=norm,
                label_contact_check=dict(subject_in_any_used_model=False, image_level=True, note='pair-level check runs in validate_segmentation_provenance'), registry_path=registry_rel, registry_sha256=staged_sha256,
                registry_note=f'the staged registry ({hc.STAGED_REGISTRY}); configs/bayesreg/segmenters_hs_v1.json gets the same bytes after the review (run_hs.py register), the queue of the fields compares them', composition=composition,
                allowlist_path=None if allowlist is None else allowlist.path, allowlist_sha256=None if allowlist is None else allowlist.sha256,
                version=dict(bayesreg=_bayesreg_version(), segmenter=hc.CANDIDATES[candidate]['segmenter_id'], candidate=candidate), command=command, environment=dict(environment, code=code), runtime_seconds=seconds)


def predict_subject(subject, *, candidate, files, registry, staged_sha256, allowlist, available, grouping, grouping_check, pred_root, nn_probs_root, nn_sha256_by_subject, device, code, command, environment, registry_rel=hc.REGISTRY_REL):
    """One subject: UF4 with eval_uf's code, (b) the composer when it is the candidate; the file; the record.  The count of voxels where (b) differs from (a) is computed for both candidates (eval_uf.json keeps it): it needs nnU-Net's stored probabilities, read back."""
    started = time.perf_counter()
    c = hc.CANDIDATES[candidate]
    plan, models, policy = uf_plan(registry, candidate, subject, allowlist, available)
    tick = time.perf_counter()
    probs, meta, norm, members = uf4_probabilities(subject, files=files, registry=registry, models=models, plan=plan, device=device)
    seconds_forward = time.perf_counter()-tick
    p16 = probs.astype(np.float16)                  # eval_uf: ensemble_probabilities(...).cpu().numpy().astype(np.float16); infer.segment_subject stores probs.astype(np.float16)
    nn_path = Path(nn_probs_root)/subject/PROBS_FILE
    nn_sha = sha256_file(nn_path)
    if nn_sha != (nn_sha256_by_subject or {}).get(subject):
        raise HSSegFailed(f"{nn_path}: sha256 {nn_sha} is not the file UF's gate passed")
    p_nn, _ = read_back(nn_path, (N_CLASSES,)+p16.shape[1:])
    n_b_not_a = b_differs_from_a(p16, p_nn)
    if c['nn_model'] is None:
        del p_nn
        labels = p16.argmax(axis=0)                 # eval_uf's tops['UF4']: ties to the smallest label
        stored, composition = probs, dict(rule='(a) UF4: the mean over the four members of softmax(z_m / T_m) (ensemble_probabilities), float16 as v2 stores probs36', weights=dict(UF4=1.), sources=dict(uf4=dict(members=[m['member_sha256'] for m in members])))
    else:
        del probs                                   # (b) stores the mixture; UF4's float32 is not needed any more (host memory)
        mixed = mix_in_place(p16, p_nn)
        labels = mixed.argmax(axis=0)               # eval_uf.half_and_half: ties to the smallest label
        stored = mixed
        composition = dict(rule='(b) 1/2 x UF4 (its float16 read back: float64 divided by the channel sum) + 1/2 x nnU-Net (its stored probs36 read back), eval_uf.half_and_half; float16 with this argmax kept (one ULP where float16 makes a tie)', weights=dict(UF4=1-hc.NN_WEIGHT, nnunet=hc.NN_WEIGHT),
                           sources=dict(uf4=dict(members=[m['member_sha256'] for m in members]), nnunet=dict(file=str(nn_path), sha256=nn_sha, model_id=c['nn_model'], checkpoint_sha256=list(models[c['nn_model']]['member_model_sha256'])[0])))
        members = members+[dict(model_id=c['nn_model'], fold='all', seed=None, member_sha256=list(models[c['nn_model']]['member_model_sha256'])[0], checkpoint='checkpoint_final.pth', probabilities_file=str(nn_path), probabilities_sha256=nn_sha, weight=hc.NN_WEIGHT)]
    del p16
    provenance = provenance_of(subject, candidate=candidate, registry=registry, models=models, policy=policy, members=members, meta=meta, norm=norm, grouping=grouping, grouping_check=grouping_check, registry_rel=registry_rel, staged_sha256=staged_sha256, code=code, environment=environment, command=command, allowlist=allowlist,
                               composition=composition, seconds=time.perf_counter()-started)
    row = write_prediction(subject, stored, labels, out_root=pred_root, grouping=grouping, provenance=provenance)
    return dict(row, seconds_forward=seconds_forward, seconds_total=time.perf_counter()-started, voxels_b_differs_from_a=n_b_not_a, image_sha256=meta['sha256'], members=[m['member_sha256'] for m in members])


# ---- the units --------------------------------------------------------------------------------------------------------------------------------
def infer_all(candidate, *, subjects, seg_dir, uf_out, nn_out, evaluation, split, allowlist, training_source, pinned_folds_sha256, files, device, code, command, sealed=None, evaluation_sha256=None, log=print):
    """The unit tag-hs-infer: stage, then every subject once (a subject done is not done again), then infer_record.json.  HSSegFailed stops the rest and leaves its evidence."""
    import torch
    from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping, require_inference_grouping
    from bayesreg.seg.unet.registry import find_member_files
    from bayesreg.seg.unet.train import environment_record
    seg_dir = Path(seg_dir)
    if (seg_dir/hc.INFER_RECORD).exists():
        raise FileExistsError(f'{seg_dir/hc.INFER_RECORD} exists: the inference is done; never rerun')
    registry, staged = stage(candidate, seg_dir=seg_dir, uf_out=uf_out, evaluation=evaluation, split=split, allowlist=allowlist, training_source=training_source, pinned_folds_sha256=pinned_folds_sha256, nn_out=nn_out, sealed=sealed)
    convert = read_json(Path(uf_out)/hc.UF_NN_REL/'convert_record.json') or {}
    if sorted((convert.get('probs36_sha256') or {})) != sorted(subjects) or convert.get('gate') != 'passed':
        raise HSSegFailed("UF's convert_record.json is not the 20 subjects through the gate")
    grouping = load_grouping(DEFAULT_GROUPS_PATH, 'structure13_v1')
    grouping_check = require_inference_grouping(grouping)
    available = find_member_files(uf_out)
    device = torch.device(device)
    if device.type == 'cuda':
        from bayesreg.seg.unet.large import cap_gpu_memory
        cap_gpu_memory(device, GPU_CAP_GIB)
    environment = {k: v for k, v in environment_record(device).items() if k != 'code'}
    environment.update(cudnn_benchmark=bool(torch.backends.cudnn.benchmark), cudnn_deterministic=bool(torch.backends.cudnn.deterministic))
    done = seg_dir/'subjects'
    done.mkdir(parents=True, exist_ok=True)
    rows = {}
    for subject in subjects:
        if (done/f'{subject}.json').is_file():
            rows[subject] = read_json(done/f'{subject}.json')
            continue
        row = predict_subject(subject, candidate=candidate, files=files, registry=registry, staged_sha256=staged['sha256'], allowlist=allowlist, available=available, grouping=grouping, grouping_check=grouping_check, pred_root=seg_dir/'pred', nn_probs_root=Path(uf_out)/hc.UF_NN_REL/'probs',
                              nn_sha256_by_subject=convert.get('probs36_sha256'), device=device, code=code, command=command, environment=environment)
        write_json_once(done/f'{subject}.json', row)
        rows[subject] = row
        log(f"{subject}: {row['seconds_total']:.1f} s, {row['n_float16_ties_fixed']} voxels with one ULP, the reader's argmax the composer's on every voxel, group deviation {row['group_max_abs_deviation']:.2e}, (b) differs from (a) on {row['voxels_b_differs_from_a']} voxels")
    composition_code = {rel: sha256_file(EXPERIMENTS.parent/rel) for rel in ('experiments/hs/hs_seg.py', 'experiments/hs/hs_core.py', 'experiments/uf/uf_nn.py', 'experiments/uf/eval_uf.py', 'experiments/fl/nx_seg.py')}     # where the rule of the channels lives (review important 2: (b) is not reproducible from the registry)
    record = dict(schema='bayesreg.hs.infer.v1', candidate=candidate, segmenter_id=hc.CANDIDATES[candidate]['segmenter_id'], subjects=list(subjects), staged=staged, eval_uf_sha256=evaluation_sha256, launch=read_json(seg_dir/'infer_launched.json'), composition_code=composition_code, rows=rows, n_reader_not_own_total=sum(r['n_reader_not_own'] for r in rows.values()),
                  gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None, environment=environment, code=code, command=command, utc=utc(), passed=all(r['n_reader_not_own'] == 0 for r in rows.values()) and len(rows) == len(subjects))
    write_json_once(seg_dir/hc.INFER_RECORD, record)
    return record


def infer_problems(record, subjects, candidate):
    """Why infer_record.json is not the 20 subjects done, as a list."""
    if record is None:
        return ['no infer_record.json']
    problems = []
    if record.get('candidate') != candidate or sorted(record.get('subjects') or []) != sorted(subjects) or record.get('passed') is not True or sorted(record.get('rows') or {}) != sorted(subjects):
        problems.append(f"infer_record.json: candidate {record.get('candidate')}, {len(record.get('rows') or {})} subjects, passed {record.get('passed')}")
    return problems


def segcheck(candidate, *, seg_dir, evaluation, subjects, data_root, lookup=None, tol=hc.SEGCHECK_TOL, n_subjects=hc.N_SUBJECTS, evaluation_sha256=None, log=print):
    """The unit tag-hs-segcheck: the registered predictions against eval_uf.json's rows of the candidate's arm, through eval_uf's functions; the truth of the 20 subjects is read here and only here.  Every check of a prediction comes first (provenance, hashes; eval_seg.read_prediction).  segcheck.json, once."""
    import eval_nn
    import eval_seg
    seg_dir = Path(seg_dir)
    if (seg_dir/hc.SEGCHECK_RECORD).exists():
        raise FileExistsError(f'{seg_dir/hc.SEGCHECK_RECORD} exists; never rerun')
    arm = hc.CANDIDATES[candidate]['uf_arm']
    sid = hc.CANDIDATES[candidate]['segmenter_id']
    theirs = ((evaluation or {}).get('rows') or {}).get(arm) or {}
    extra = (evaluation or {}).get('per_subject_extra') or {}
    if sorted(theirs) != sorted(subjects) or sorted((evaluation or {}).get('subjects') or []) != sorted(subjects):
        raise HSSegFailed(f'eval_uf.json has no rows of {arm} for these {len(subjects)} subjects')
    lookup = eval_seg.load_seg4_lookup() if lookup is None else lookup
    tops = {}
    for subject in subjects:                        # every prediction's checks before the first truth label
        tops[subject] = eval_seg.read_prediction(seg_dir/'pred', subject, expected_segmenter=sid)
    infer = read_json(seg_dir/hc.INFER_RECORD) or {}
    rows, differences, truth_sha = {}, {}, {}
    for subject in subjects:
        top, _ = tops[subject]
        true35, sha35 = eval_seg.read_truth(data_root, subject, 'seg35')
        true4, sha4 = eval_seg.read_truth(data_root, subject, 'seg4')
        truth_sha[subject] = dict(seg35=sha35, seg4=sha4)
        mine = eval_seg.no_nan(eval_nn.metrics_of(eval_seg.summarize_subject(top, true35, true4, lookup)))     # an undefined Dice is null, as eval_uf.json stores it
        rows[subject] = mine
        other = theirs[subject]
        per_label = [abs(eval_nn.clean(a)-eval_nn.clean(b)) for a, b in zip(mine['per_label'], other.get('per_label') or [])]
        finite = [d for d in per_label if np.isfinite(d)]
        differences[subject] = dict(seg35=mine['seg35']-other['seg35'], seg4=mine['seg4']-other['seg4'], max_abs_per_label=max(finite, default=0.), rows_equal=all(mine.get(k) == other.get(k) for k in ('seg35', 'seg4', 'per_label', 'seg4_per_label')) and len(per_label) == len(mine['per_label']),
                                    voxels_b_differs_from_a=((infer.get('rows') or {}).get(subject) or {}).get('voxels_b_differs_from_a'), uf_voxels_b_differs_from_a=(extra.get(subject) or {}).get('voxels_b_differs_from_a'),
                                    truth_sha256=truth_sha[subject], truth_same_as_uf=(evaluation.get('truth_sha256') or {}).get(subject) == truth_sha[subject])     # missing in eval_uf.json: not passed (revision 1b, the follow-up's minor 3)
        log(f"{subject}: seg35 {mine['seg35']:.6f} against UF's {other['seg35']:.6f}; rows equal {differences[subject]['rows_equal']}")
    mean = {m: float(np.mean([rows[s][m] for s in subjects])-np.mean([theirs[s][m] for s in subjects])) for m in ('seg35', 'seg4')}
    passed = len(subjects) == n_subjects and all(abs(mean[m]) <= tol for m in mean) and all(d['truth_same_as_uf'] for d in differences.values())     # the truth files must be those UF read (eval_uf.json truth_sha256; revision 1, review minor 5)
    document = dict(schema='bayesreg.hs.segcheck.v1', candidate=candidate, segmenter_id=sid, uf_arm=arm, subjects=list(subjects), tolerance=tol, rule=f'|mean seg35| and |mean seg4| of the registered labels minus eval_uf.json rows of {arm}, 20 subjects, <= {tol:g} (UF U1_control: a GPU forward pass)',
                    mean_difference=mean, passed=passed, eval_uf_sha256=evaluation_sha256, launch=read_json(seg_dir/'segcheck_launched.json'), rereads=read_json(seg_dir/'segcheck_rereads.json'), n_rows_equal=sum(d['rows_equal'] for d in differences.values()), n_count_equal=sum(d['voxels_b_differs_from_a'] == d['uf_voxels_b_differs_from_a'] for d in differences.values()), per_subject=differences,
                    note='eval_uf.py kept no label map: the labels of UF\'s evaluation are seen here only through its rows and its counts; the truth of the 20 development subjects is read here, numbers UF already read', utc=utc())
    write_json_once(seg_dir/hc.SEGCHECK_RECORD, eval_seg.no_nan(document))
    return document


def register(candidate, *, seg_dir, registry_path, subjects):
    """By hand, after the review: the staged bytes into the repository registry, once.  The inference and the control must have passed."""
    seg_dir = Path(seg_dir)
    problems = infer_problems(read_json(seg_dir/hc.INFER_RECORD), subjects, candidate)
    check = read_json(seg_dir/hc.SEGCHECK_RECORD)
    if check is None or check.get('passed') is not True or check.get('candidate') != candidate:
        problems.append(f"segcheck.json: {None if check is None else check.get('passed')}")
    staged = seg_dir/hc.STAGED_REGISTRY
    if not staged.is_file():
        problems.append(f'no {staged}')
    if problems:
        raise HSSegFailed(f'not registered: {problems}')
    text = staged.read_text(encoding='utf-8')
    registry_path = Path(registry_path)
    if registry_path.exists():
        if registry_path.read_text(encoding='utf-8') != text:
            raise nx_seg.RegistrationRefused(f'{registry_path} is there with other content; entries are append-only')
    else:
        validated(text, registry_path)
        tmp = Path(str(registry_path)+'.tmp')
        tmp.write_text(text, encoding='utf-8')
        tmp.replace(registry_path)
    from bayesreg.io.pair import load_segmenter_registry
    return load_segmenter_registry(registry_path)
