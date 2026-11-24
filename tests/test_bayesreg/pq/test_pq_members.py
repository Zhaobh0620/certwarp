"""PQ, the members unit (D91 (a); plan.md sections 3.2, 8 m5): experiments/pq/pq_members.py and run_pq.members_unit on tiny made-up members (CPU, tmp_path; the synthetic UF4 of
tests/test_bayesreg/segprobe/test_flip_tta_v4.py: two subjects' images, four tiny members of unet35_v4_ens in two fold models, their registry, the registered predictions as HS's seg phase writes them).

- the mean is hs_seg.uf4_probabilities bit for bit (the registered inference), V the population variance of the four members' softmax(z/T) summed over the 36 classes (an independent computation);
- m5: the argmax agreement with the registered prediction, and the image against the provenance's input_sha256;
- the members unit end to end under the spy: only the images, the four member files and the registered predictions are opened (no label file of the image root), var.npz and the record written once;
- the plan's members must be the four pinned ones in HS's order.
"""
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from . import _world as W
from .conftest import CODE, GATE
from ..segprobe.test_flip_tta_v4 import SUBJECTS, flip_tta, world as tta_world

io_, pins = W.io_, W.pins

import pq_members as members  # noqa: E402
import pq_core as core  # noqa: E402
import run_pq  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture
def tw(tmp_path, monkeypatch):
    """The TTA world, with the PQ pins pointed at it: its four members (names fold{A,B}_s{0,1}, the plan's order), its registry as the anchor, its predictions as the plan's table."""
    w = tta_world(tmp_path)
    shas = w.shas
    monkeypatch.setattr(pins, 'MEMBERS', tuple((f'unet35_v4_ens_fold{n[4]}', n, shas[n]) for n in ('foldA_s0', 'foldA_s1', 'foldB_s0', 'foldB_s1')))
    monkeypatch.setattr(pins, 'MODELS_ROOT_REL', 'members')
    monkeypatch.setattr(pins, 'PRED_ROOT_REL', 'pred')
    repo = tmp_path/'repo'
    (repo/'runs'/'20261107-pq').mkdir(parents=True)
    shutil.copy(W.ROOT/pins.PLAN_REL, repo/pins.PLAN_REL)
    (repo/'configs'/'bayesreg').mkdir(parents=True)
    shutil.copy(w.spec['registry'], repo/pins.ANCHOR['registry'])
    v4 = {s: pins.V4Row(probs36=io_.sha_file(w.pred_root/s/'probs36.npz'), provenance=io_.sha_file(w.pred_root/s/'provenance.json'), structure_groups=None) for s in SUBJECTS}
    short = f'{int(SUBJECTS[0].split("_")[2]):04d}-{int(SUBJECTS[1].split("_")[2]):04d}'
    ctx = io_.Context(repo=repo, source_root=repo, results=tmp_path, images=w.data, shorts=(short,), p3={short: pins.P3Row('C_first5', 'RC_c600', '', '', '', '')}, run_json={short: ''},
                      v4=v4, anchor=dict(pins.ANCHOR, registry_sha256=w.registry.sha256))
    w.ctx = ctx
    return w


def independent_variance(w, subject):
    """sum_c Var_m softmax(z_m / T_m)^c from the members run one by one here (the same loader, normalization and predict_logits as the registered inference)."""
    from bayesreg.seg.unet.data import normalize_image, read_image
    from bayesreg.seg.unet.infer import load_member
    from bayesreg.seg.unet.train import predict_logits
    plan, models, _ = w.plans[subject]
    values, _ = read_image(w.files, subject)
    ps = []
    for model_id, sha, path in plan:
        model, blob = load_member(path, sha, 'cpu', entry=models[model_id])
        image, _ = normalize_image(values, percentile=blob['percentile'])
        with torch.no_grad():
            z = predict_logits(model, image, device='cpu', amp=blob['amp'], window=tuple(blob['window']), gaussian_sigma=blob['window_gaussian_sigma'])
            ps.append(torch.softmax(z.float()/float(blob['temperature']), dim=1)[0].double().numpy())
    P = np.stack(ps)
    return np.clip(P.var(axis=0).sum(axis=0), 0, None)


def test_the_mean_is_the_registered_inference_and_v_the_members_variance(tw):
    hs = flip_tta.hs_seg_module()
    for subject in SUBJECTS:
        plan, models, _ = tw.plans[subject]
        mean, V, meta, norm, records = members.mean_and_variance(subject, files=tw.files, models=models, plan=plan, device='cpu')
        theirs, meta2, norm2, records2 = hs.uf4_probabilities(subject, files=tw.files, registry=tw.registry, models=models, plan=plan, device='cpu')
        assert mean.dtype == np.float32 and mean.tobytes() == theirs.tobytes() and meta == meta2 and norm == norm2 and records == records2
        assert V.dtype == np.float32 and V.shape == mean.shape[1:]
        assert np.max(np.abs(V.astype(np.float64)-independent_variance(tw, subject))) <= 1e-6 and V.max() > 0


def test_m5_and_the_files_of_run_members(tw):
    rows = members.run_members(tw.ctx, allowlist=tw.allowlist, registry=tw.registry, files=tw.files, device=torch.device('cpu'), log=lambda *a: None)
    for subject in SUBJECTS:
        r = rows[subject]
        assert r['m5'] is True and r['agreement'] == 1. and r['image_matches_provenance'] is True
        path = tw.ctx.pq_root/'members'/subject/io_.VAR_FILE
        assert io_.sha_file(path) == r['var_sha256']
    with pytest.raises(FileExistsError):
        members.run_members(tw.ctx, allowlist=tw.allowlist, registry=tw.registry, files=tw.files, device=torch.device('cpu'), log=lambda *a: None)


def test_m5_fails_on_a_disagreeing_prediction(tw):
    plan, models, _ = tw.plans[SUBJECTS[0]]
    mean, V, meta, norm, records = members.mean_and_variance(SUBJECTS[0], files=tw.files, models=models, plan=plan, device='cpu')
    stored = mean.astype(np.float16)
    stored = stored[::-1].copy()                                                     # another prediction: the classes reversed
    prov = io_.read_json(tw.pred_root/SUBJECTS[0]/'provenance.json')
    r = members.subject_record(SUBJECTS[0], mean=mean, V=V, meta=meta, norm=norm, members=records, stored=stored, prov=prov, out_dir=tw.ctx.pq_root/'x', seconds=0.)
    assert r['m5'] is False and r['agreement'] < members.AGREEMENT_MIN
    prov['input_sha256'] = {'aligned_norm.nii.gz': '9'*64}
    r = members.subject_record(SUBJECTS[0], mean=mean, V=V, meta=meta, norm=norm, members=records, stored=mean.astype(np.float16), prov=prov, out_dir=tw.ctx.pq_root/'y', seconds=0.)
    assert r['m5'] is False and r['agreement'] == 1. and r['image_matches_provenance'] is False


def test_the_members_unit_end_to_end_under_the_spy(tw):
    W.reset_sealed_cache()                                                           # exception (13), C23.9-13 revision 6f
    rc = run_pq.members_unit(tw.ctx, device='cpu', allowlist=tw.allowlist, files=tw.files, gate=GATE, code=CODE, log=lambda *a: None)
    assert rc == 0
    record = io_.read_json(tw.ctx.pq_root/'members'/io_.MEMBERS_RECORD)
    assert record['m5_passed'] is True and sorted(record['subjects']) == sorted(SUBJECTS) and record['error'] is None
    opened = io_.read_json(tw.ctx.pq_root/'members'/'opened_files_members.json')
    assert opened['m6'] is True and not opened['breaches']
    names = sorted({Path(f['path']).name for f in opened['files'] if f['in_scope']})
    assert 'aligned_seg35.nii.gz' not in names and 'aligned_norm.nii.gz' in names and 'member.pt' in names
    images = sorted(Path(f['path']).parent.name for f in opened['files'] if Path(f['path']).name == 'aligned_norm.nii.gz')
    assert images == sorted(SUBJECTS)
    W.reset_sealed_cache()                                                           # exception (13)
    with pytest.raises(core.PQRefused, match='once'):
        run_pq.members_unit(tw.ctx, device='cpu', allowlist=tw.allowlist, files=tw.files, gate=GATE, code=CODE, log=lambda *a: None)


def test_a_breach_in_the_members_unit_is_kept_and_fails(tw, monkeypatch):
    real = members.registered

    def peeking(pred_root, subject, want):
        open(Path(tw.data)/subject/'aligned_seg35.nii.gz', 'rb').read()             # a label file of the image root
        return real(pred_root, subject, want)
    monkeypatch.setattr(members, 'registered', peeking)
    W.reset_sealed_cache()                                                           # exception (13)
    assert run_pq.members_unit(tw.ctx, device='cpu', allowlist=tw.allowlist, files=tw.files, gate=GATE, code=CODE, log=lambda *a: None) == 8
    record = io_.read_json(tw.ctx.pq_root/'members'/io_.MEMBERS_RECORD)
    assert record['m5_passed'] is False and 'Breach' in record['error'] and record['spy']['m6'] is False


def test_the_plans_members_are_the_pinned_four_in_order(tw, monkeypatch):
    available = members.member_files(Path(tw.ctx.results)/pins.MODELS_ROOT_REL)
    assert members.plan_for(tw.registry, SUBJECTS[0], tw.allowlist, available)[2] == 'all_folds_mean'
    monkeypatch.setattr(pins, 'MEMBERS', tuple(reversed(pins.MEMBERS)))
    with pytest.raises(core.PQRefused, match='order'):
        members.plan_for(tw.registry, SUBJECTS[0], tw.allowlist, available)


def test_a_breach_in_the_window_before_the_inner_try_is_recorded(tw, monkeypatch):
    """Revision 2 (diff review minor 5): a breach after the spy and before the inner try (here inside the plan check) writes opened_files_members.json and members_record.json marked 越界、尚无输出."""
    real = io_.check_plan

    def peeking(repo, *args, **kwargs):
        open(Path(tw.data)/SUBJECTS[0]/'aligned_seg35.nii.gz', 'rb').read()
        return real(repo, *args, **kwargs)
    monkeypatch.setattr(io_, 'check_plan', peeking)
    W.reset_sealed_cache()                                                           # exception (13)
    assert run_pq.members_unit(tw.ctx, device='cpu', allowlist=tw.allowlist, files=tw.files, gate=GATE, code=CODE, log=lambda *a: None) == 8
    record = io_.read_json(tw.ctx.pq_root/'members'/io_.MEMBERS_RECORD)
    opened = io_.read_json(tw.ctx.pq_root/'members'/'opened_files_members.json')
    assert record['status'] == run_pq.WINDOW_BREACH and record['m5_passed'] is False and 'Breach' in record['error'] and opened['status'] == run_pq.WINDOW_BREACH and len(opened['breaches']) == 1
    assert not any((tw.ctx.pq_root/'members'/s/io_.VAR_FILE).exists() for s in SUBJECTS)


def test_a_member_file_that_is_not_the_registered_one_is_refused(tw):
    target = Path(tw.ctx.results)/pins.MODELS_ROOT_REL/'foldB_s1'/'member.pt'
    target.write_bytes(target.read_bytes()+b'\0')
    with pytest.raises(core.PQRefused, match='members'):
        members.member_files(Path(tw.ctx.results)/pins.MODELS_ROOT_REL)
