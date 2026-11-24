"""R2 step 3b, the seg phase of the registration reading (runs/20261110-r2/plan.md section 5): the candidate (R2S; one arm) registered as a structure-layer segmenter, its 20 development predictions, the control.

HS's seg phase (experiments/hs/hs_seg.py) loaded once more with r2f_core in the place of hs_core (r2f_core.load_copy), its general parts unchanged:
  uf_plan, uf4_probabilities  the members of the registered fold models in the registry's order, each on its sliding and Gaussian window, ensemble_probabilities, float16 (here six members: UF4's four
                              and the recipe's two, fold A then fold B; the order of eval_r2's arm, so that the running sum is the same);
  write_prediction            probs36.npz (float16, the argmax kept), structure_groups.npz, the reader's read-back, provenance.json last;
  segcheck                    the registered labels against eval_r2.json's rows of the candidate's arm, |mean seg35|, |mean seg4| <= 2e-4, the truth files those eval_r2 read; the truth only here;
  register, infer_problems    as HS's.
Replaced (R2 has no nnU-Net and two member roots): uf_evaluated_members (eval_r2.json's members), registry_document (three members per fold), stage (verify_members on UF's root and the recipe's root),
predict_subject (no nnU-Net file, no (b)), infer_all (the member files of both roots).  hs_seg.py is not edited.
"""
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _p in (HERE, EXPERIMENTS/'hs', EXPERIMENTS/'fl', EXPERIMENTS/'ef', EXPERIMENTS/'mc0', EXPERIMENTS/'nn', EXPERIMENTS/'uf', EXPERIMENTS/'unv2'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import r2_common as rc  # noqa: E402
import r2f_core as hc  # noqa: E402

seg = hc.load_copy('_r2_hs_seg', EXPERIMENTS/'hs'/'hs_seg.py', {'hs_core': hc})
HSSegFailed = seg.HSSegFailed
COMPOSITION_FILES = ('experiments/hs/hs_seg.py', 'experiments/r2/r2_seg.py', 'experiments/r2/r2f_core.py', 'experiments/r2/r2_common.py', 'experiments/r2/eval_r2.py')


def recipe_root(candidate, uf_out=None):
    return Path(hc.UF_OUT if uf_out is None else uf_out)/hc.CANDIDATES[candidate]['recipe']


def r2_evaluated_members(evaluation, candidate):
    """{(family, member name): member.pt sha256} that eval_r2.py evaluated for this candidate (eval_r2.json 'members'); HSSegFailed if any of the six is missing."""
    members = (evaluation or {}).get('members') or {}
    recipe = hc.CANDIDATES[candidate]['recipe']
    want = [('uf', n) for n in rc.UF_ORDER]+[(recipe, n) for n in rc.MEMBER_NAMES]
    out = {}
    for family, name in want:
        entry = members.get(f'{family}/{name}') or {}
        if not entry.get('member_sha256'):
            raise HSSegFailed(f'eval_r2.json does not name {family}/{name}')
        out[(family, name)] = entry['member_sha256']
    return out


def registry_document(candidate, *, members):
    """The schema bayesreg.segmenters.v1 for the candidate: build_entries of UF's four and the recipe's two members (full training set: each fold's seeds 0, 1, 2 in order, the contacts equal)."""
    from bayesreg.io.pair import REGISTRY_SCHEMA
    from bayesreg.seg.unet.registry import build_entries
    c = hc.CANDIDATES[candidate]
    entries = build_entries(members, segmenter_id=c['segmenter_id'], prefix=c['prefix'], seeds=(0, 1, rc.SEED), full_training_set=True)
    ids = hc.model_ids(candidate)
    if sorted(entries) != sorted([ids['A'], ids['B']]) or any(len(e['member_model_sha256']) != 3 for e in entries.values()):
        raise HSSegFailed(f'the members do not make two fold models of three seeds each: {sorted(entries)}')
    comment = (f"structure-layer models of R2 (runs/20261110-r2/plan.md): {c['words']}; UF's members of seg/unet35_v4_full (unet35_v4_ens) and the recipe {c['recipe']}'s of seg/unet35_r2_full/{c['recipe']}; "
               'an equal-weight mean of the six members; append-only; written by experiments/r2/r2_seg.py')
    return dict(schema=REGISTRY_SCHEMA, comment=comment, models={m: entries[m] for m in sorted(entries)})


def stage(candidate, *, seg_dir, uf_root, r2_out, evaluation, split, allowlist, training_source, pinned_folds_sha256):
    """The staged registry: verify_members on UF's root and on the recipe's root (every check of a registration), each member.pt the one eval_r2.json evaluated.  Written once; a file that is there must
    have the same bytes.  Returns (registry, record)."""
    from bayesreg.io.pair import load_segmenter_registry
    from bayesreg.seg.unet.registry import verify_members
    members, checks = [], {}
    for family, root in (('uf', uf_root), (hc.CANDIDATES[candidate]['recipe'], recipe_root(candidate, r2_out))):
        found, check = verify_members(root, split=split, allowlist=allowlist, training_source=training_source, pinned_folds_sha256=pinned_folds_sha256)
        members += [dict(m, family=family) for m in found]
        checks[family] = check
    found = {(m['family'], hc.member_name(m['fold'], m['seed'])): m['member_sha256'] for m in members}
    evaluated = r2_evaluated_members(evaluation, candidate)
    if found != evaluated:
        raise HSSegFailed(f'the members are not those eval_r2.json evaluated: {sorted(found)} against {sorted(evaluated)}')
    text = seg.document_text(registry_document(candidate, members=members))
    seg_dir = Path(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    path = seg_dir/hc.STAGED_REGISTRY
    seg.validated(text, path)
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise HSSegFailed(f'{path} is there with other content: a staged registry is written once')
    else:
        tmp = Path(str(path)+'.tmp')
        tmp.write_text(text, encoding='utf-8')
        tmp.replace(path)
    registry = load_segmenter_registry(path)
    return registry, dict(path=str(path), sha256=registry.sha256, candidate=candidate, segmenter_id=hc.CANDIDATES[candidate]['segmenter_id'], models=sorted(registry.models), member_check=checks,
                          evaluated_members={f'{f}/{n}': s for (f, n), s in evaluated.items()})


def predict_subject(subject, *, candidate, files, registry, staged_sha256, allowlist, available, grouping, grouping_check, pred_root, device, code, command, environment):
    """One subject: the six members' mean (hs_seg.uf_plan and uf4_probabilities, the registry's order), float16 as infer.segment_subject stores it, the argmax with ties to the smallest label; the file; the record."""
    import numpy as np
    started = time.perf_counter()
    plan, models, policy = seg.uf_plan(registry, candidate, subject, allowlist, available)
    tick = time.perf_counter()
    probs, meta, norm, members = seg.uf4_probabilities(subject, files=files, registry=registry, models=models, plan=plan, device=device)
    seconds_forward = time.perf_counter()-tick
    labels = probs.astype(np.float16).argmax(axis=0)            # eval_r2: the float16 of the mean, ties to the smallest label
    composition = dict(rule=f"{candidate}: the mean over the six members of softmax(z_m / T_m) (ensemble_probabilities), UF4's four and the recipe {hc.CANDIDATES[candidate]['recipe']}'s two, fold A then fold B, each fold's seeds 0, 1, 2; float16 as v2 stores probs36",
                       weights={m['member_sha256']: 1/len(members) for m in members})
    provenance = seg.provenance_of(subject, candidate=candidate, registry=registry, models=models, policy=policy, members=members, meta=meta, norm=norm, grouping=grouping, grouping_check=grouping_check, registry_rel=hc.REGISTRY_REL,
                                   staged_sha256=staged_sha256, code=code, environment=environment, command=command, allowlist=allowlist, composition=composition, seconds=time.perf_counter()-started)
    provenance['registry_note'] = f'the staged registry ({hc.STAGED_REGISTRY}); {hc.REGISTRY_REL} gets the same bytes after the review (run_r2f.py register), the queue of the fields compares them'
    row = seg.write_prediction(subject, probs, labels, out_root=pred_root, grouping=grouping, provenance=provenance)
    return dict(row, seconds_forward=seconds_forward, seconds_total=time.perf_counter()-started, image_sha256=meta['sha256'], members=[m['member_sha256'] for m in members])


def infer_all(candidate, *, subjects, seg_dir, uf_root, r2_out, evaluation, split, allowlist, training_source, pinned_folds_sha256, files, device, code, command, evaluation_sha256=None, log=print):
    """The unit tag-r2-infer: stage, then every subject once (a subject done is not done again), then infer_record.json.  HSSegFailed stops the rest and keeps its evidence."""
    import torch
    from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping, require_inference_grouping
    from bayesreg.seg.unet.registry import find_member_files
    from bayesreg.seg.unet.train import environment_record
    seg_dir = Path(seg_dir)
    if (seg_dir/hc.INFER_RECORD).exists():
        raise FileExistsError(f'{seg_dir/hc.INFER_RECORD} exists: the inference is done; never rerun')
    registry, staged = stage(candidate, seg_dir=seg_dir, uf_root=uf_root, r2_out=r2_out, evaluation=evaluation, split=split, allowlist=allowlist, training_source=training_source, pinned_folds_sha256=pinned_folds_sha256)
    grouping = load_grouping(DEFAULT_GROUPS_PATH, 'structure13_v1')
    grouping_check = require_inference_grouping(grouping)
    available = dict(find_member_files(uf_root))
    available.update(find_member_files(recipe_root(candidate, r2_out)))
    device = torch.device(device)
    if device.type == 'cuda':
        from bayesreg.seg.unet.large import cap_gpu_memory
        cap_gpu_memory(device, seg.GPU_CAP_GIB)
    environment = {k: v for k, v in environment_record(device).items() if k != 'code'}
    environment.update(cudnn_benchmark=bool(torch.backends.cudnn.benchmark), cudnn_deterministic=bool(torch.backends.cudnn.deterministic))
    done = seg_dir/'subjects'
    done.mkdir(parents=True, exist_ok=True)
    rows = {}
    for subject in subjects:
        if (done/f'{subject}.json').is_file():
            rows[subject] = seg.read_json(done/f'{subject}.json')
            continue
        row = predict_subject(subject, candidate=candidate, files=files, registry=registry, staged_sha256=staged['sha256'], allowlist=allowlist, available=available, grouping=grouping, grouping_check=grouping_check,
                              pred_root=seg_dir/'pred', device=device, code=code, command=command, environment=environment)
        seg.write_json_once(done/f'{subject}.json', row)
        rows[subject] = row
        log(f"{subject}: {row['seconds_total']:.1f} s, the reader's argmax the composer's on every voxel, group deviation {row['group_max_abs_deviation']:.2e}")
    composition_code = {rel: seg.sha256_file(EXPERIMENTS.parent/rel) for rel in COMPOSITION_FILES}
    record = dict(schema='bayesreg.r2.infer.v1', candidate=candidate, segmenter_id=hc.CANDIDATES[candidate]['segmenter_id'], subjects=list(subjects), staged=staged, eval_r2_sha256=evaluation_sha256, launch=seg.read_json(seg_dir/'infer_launched.json'),
                  composition_code=composition_code, rows=rows, n_reader_not_own_total=sum(r['n_reader_not_own'] for r in rows.values()),
                  gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None, environment=environment, code=code, command=command, utc=seg.utc(),
                  passed=all(r['n_reader_not_own'] == 0 for r in rows.values()) and len(rows) == len(subjects))
    seg.write_json_once(seg_dir/hc.INFER_RECORD, record)
    return record


hc.replace_names(seg, uf_evaluated_members=None,        # HS's: four UF members from eval_uf.json; not R2's (r2_evaluated_members)
                 nn_checkpoint=None,                     # no nnU-Net in R2
                 registry_document=registry_document, stage=stage, predict_subject=predict_subject, infer_all=infer_all)     # each name checked to exist first (review m1)
segcheck, register, infer_problems, read_json = seg.segcheck, seg.register, seg.infer_problems, seg.read_json
