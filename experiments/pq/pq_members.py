"""PQ, the members unit (D91 (a); contract C22.Q1-3; plan.md sections 1.3, 3.2, 8 m5): the variance of the four registered members of unet35_v4_ens, per voxel, for the baselines V_F and V2.

Per subject (the 20 development subjects): the image aligned_norm.nii.gz only (bayesreg.seg.unet.data.read_image; its sha256 against the registered provenance's input_sha256), the four members by path and sha256
(the pinned four files, not their directory), HS's plan (hs_seg.uf_plan: infer.check_subject_admitted on the allowlist, then D24 select_fold_models, all_folds_mean, A0 A1 B0 B1), and the registered
inference of UF4 (hs_seg.uf4_probabilities, the unflipped arm of the TTA tool: infer.load_member, normalize_image with the member's percentile, predict_logits with each member's own amp, window and Gaussian
window, softmax(z / T_m)) copied here with one addition: every member's p_m = softmax(z_m / T_m) is also added, in float64, to sum p and sum p^2.  Then
    V(x) = sum_{c=0}^{35} [ (1/4) sum_m p_m^c(x)^2 - ((1/4) sum_m p_m^c(x))^2 ],  clamped at 0, stored float32 (N0 (e)'s var_total, here accumulated in float64);
the ensemble mean is infer.ensemble_probabilities of the same logits, as registered (a test: bit for bit hs_seg.uf4_probabilities on tiny members).  Only V is stored, not the member probabilities.
m5: the argmax of the mean's float16 cast (ties to the smallest label, as hs_seg and the TTA tool take UF4's) agrees with the registered probs36's argmax on >= .9999 of the voxels (the TTA control line).
"""
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
for _rel in ('hs', 'unv2'):
    if str(HERE.parent/_rel) not in sys.path:
        sys.path.append(str(HERE.parent/_rel))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pq_pins as pins  # noqa: E402
from pq_core import PQRefused  # noqa: E402
from pq_io import read_json, save_npz_once, sha_file, write_once, MEMBERS_RECORD, VAR_FILE  # noqa: E402

AGREEMENT_MIN = .9999
MEMBER_SCHEMA = 'bayesreg.pq.members.v1'


def member_files(models_root, members=None):
    """{sha256: path} of the four pinned member files (pq_pins.MEMBERS, read at call time), each hashed (PQRefused 'members' if one differs); no directory is listed (C22.Q1-3: the files, not their
    directory)."""
    out = {}
    for _, name, sha in (pins.MEMBERS if members is None else members):
        path = Path(models_root)/name/'member.pt'
        if not path.is_file() or sha_file(path) != sha:
            raise PQRefused('members', f'{path}: missing or not the registered sha256 {sha[:12]}...')
        out[sha] = path
    return out


def plan_for(registry, subject, allowlist, available):
    """hs_seg.uf_plan: the admission (infer.check_subject_admitted), D24 (all_folds_mean over every registered model), the members by sha256 in HS's order."""
    import hs_seg
    plan, models, policy = hs_seg.uf_plan(registry, pins.SEGMENTER_SCHEMA_CANDIDATE, subject, allowlist, available)
    if [sha for _, sha, _ in plan] != [sha for _, _, sha in pins.MEMBERS]:
        raise PQRefused('members', f'{subject}: the plan\'s members are not the four pinned ones in HS\'s order')
    return plan, models, policy


def mean_and_variance(subject, *, files, models, plan, device):
    """(mean float32 numpy (36, ...) = hs_seg.uf4_probabilities' array, V float32 (nz, ny, nx), image metadata, normalization, member records).  hs_seg.uf4_probabilities copied (experiments/hs/hs_seg.py:184,
    its checks and records), plus the float64 sums of p_m and p_m^2 on the device."""
    import torch
    from bayesreg.seg.unet.data import normalize_image, read_image
    from bayesreg.seg.unet.infer import ensemble_probabilities, load_member
    from bayesreg.seg.unet.train import predict_logits
    values, meta = read_image(files, subject)
    state, members, sums = {}, [], {}

    def member_logits():
        for model_id, sha, path in plan:
            model, blob = load_member(path, sha, device, entry=models[model_id])
            if 'norm' not in state:
                state['image'], state['norm'] = normalize_image(values, percentile=blob['percentile'])
            elif blob['percentile'] != state['norm']['percentile']:
                raise PQRefused('members', f'{path}: the members disagree on the normalization percentile')
            if blob.get('window_gaussian_sigma') is None or not blob.get('window'):
                raise PQRefused('members', f'{path}: member.pt has no sliding window with a Gaussian window')
            z = predict_logits(model, state['image'], device=device, amp=blob['amp'], window=tuple(blob['window']), gaussian_sigma=blob['window_gaussian_sigma'])
            members.append(dict(model_id=model_id, fold=blob['fold'], seed=blob['seed'], member_sha256=sha, temperature=blob['temperature'], checkpoint_choice=blob.get('checkpoint_choice'),
                                checkpoint_iteration=blob.get('checkpoint_iteration'), window=list(blob['window']), window_gaussian_sigma=blob['window_gaussian_sigma'], percentile=blob['percentile'], amp=blob['amp']))
            del model
            with torch.no_grad():
                p = torch.softmax(z.float()/float(blob['temperature']), dim=1)[0].double()
                if 'p2' not in sums:
                    sums['p2'] = p*p
                else:
                    sums['p2'].addcmul_(p, p)                         # in place: no second float64 stack on the device
                sums['p'] = p if 'p' not in sums else sums['p'].add_(p)
                del p
            yield z, blob['temperature']
    with torch.no_grad():
        mean = ensemble_probabilities(member_logits()).cpu().numpy()
        n = float(len(members))
        first = sums.pop('p').div_(n)                             # in place (revision 2): two float64 stacks alive here, not five
        variance = sums.pop('p2').div_(n).addcmul_(first, first, value=-1.)
        del first
        V = variance.sum(dim=0).clamp_min(0).float().cpu().numpy()
        del variance
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(torch.device(device))
    return mean, V, meta, state['norm'], members


def top16(probabilities):
    """The argmax as hs_seg and the TTA tool take UF4's: of the float16 cast, numpy, ties to the smallest label; uint8."""
    return np.asarray(probabilities).astype(np.float16).argmax(axis=0).astype(np.uint8)


def registered(pred_root, subject, want):
    """The registered prediction: provenance.json and probs36.npz with the plan's sha256 (PQRefused otherwise); (stored float16 p, provenance)."""
    d = Path(pred_root)/subject
    if sha_file(d/'provenance.json') != want.provenance or sha_file(d/'probs36.npz') != want.probs36:
        raise PQRefused('members', f'{d}: provenance.json or probs36.npz is not the plan\'s file')
    prov = read_json(d/'provenance.json')
    with np.load(d/'probs36.npz', allow_pickle=False) as z:
        stored = np.array(z['p'])
    return stored, prov


def subject_record(subject, *, mean, V, meta, norm, members, stored, prov, out_dir, seconds):
    """m5 and the description of one subject; var.npz written once."""
    mine, theirs = top16(mean), stored.argmax(axis=0).astype(np.uint8)
    agreement = float(np.mean(mine == theirs))
    image_ok = (prov.get('input_sha256') or {}).get('aligned_norm.nii.gz') == meta.get('sha256')
    out_dir = Path(out_dir)
    save_npz_once(out_dir/VAR_FILE, V=V.astype(np.float32))
    record = dict(subject=subject, m5=bool(agreement >= AGREEMENT_MIN and image_ok), agreement=agreement, agreement_min=AGREEMENT_MIN, n_differ=int((mine != theirs).sum()),
                  max_abs_difference_to_registered=float(np.max(np.abs(mean.astype(np.float32)-stored.astype(np.float32)))), image_sha256=meta.get('sha256'), image_matches_provenance=image_ok,
                  normalization=norm, members=members, var_sha256=sha_file(out_dir/VAR_FILE), V_range=[float(V.min()), float(V.max())], var_rule='sum_c Var_m softmax(z_m/T_m)^c, population variance, float64 sums, float32 stored',
                  seconds=seconds)
    write_once(out_dir/'member.json', record)
    return record


def run_members(ctx, *, allowlist, registry, files, device, log=print):
    """Every subject of the plan, one at a time (no truth anywhere): the plan, the inference, m5, var.npz and member.json under pq_v1/members/<subject>/.  The caller installs the spy and writes the record."""
    available = member_files(Path(ctx.results)/pins.MODELS_ROOT_REL)
    if registry.sha256 != ctx.anchor['registry_sha256']:
        raise PQRefused('members', f"the registry sha256 {registry.sha256} is not the anchor's")
    rows = {}
    for subject in ctx.subjects():
        tick = perf_counter()
        plan, models, policy = plan_for(registry, subject, allowlist, available)
        stored, prov = registered(ctx.pred_root, subject, ctx.v4[subject])
        mean, V, meta, norm, members = mean_and_variance(subject, files=files, models=models, plan=plan, device=device)
        rows[subject] = subject_record(subject, mean=mean, V=V, meta=meta, norm=norm, members=members, stored=stored, prov=prov, out_dir=ctx.pq_root/'members'/subject, seconds=perf_counter()-tick)
        rows[subject]['policy'] = policy
        log(f"{subject}: agreement {rows[subject]['agreement']:.6f}, m5 {rows[subject]['m5']}, {rows[subject]['seconds']:.1f} s")
        del mean, V, stored
    return rows


def members_document(rows, *, code, environment, spy_record, device, seconds):
    return dict(schema=MEMBER_SCHEMA, contract=pins.CONTRACT, plan_sha256=pins.PLAN_SHA256, d91='(a): the four members inferred again, only V stored', device=str(device),
                subjects={s: dict(m5=r['m5'], agreement=r['agreement'], n_differ=r['n_differ'], max_abs_difference_to_registered=r['max_abs_difference_to_registered'], var_sha256=r['var_sha256'],
                                  image_sha256=r['image_sha256']) for s, r in rows.items()},
                m5_passed=bool(rows) and all(r['m5'] for r in rows.values()), m5_rule=f'per subject: argmax agreement >= {AGREEMENT_MIN} and the image is the provenance\'s',
                code=code, environment=environment, spy=dict(m6=spy_record['m6'], n_files=spy_record['n_files'], breaches=spy_record['breaches']), seconds=seconds)


def write_members_record(ctx, document):
    return write_once(ctx.pq_root/'members'/MEMBERS_RECORD, document)
