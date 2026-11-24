"""PQ end to end on the synthetic world (tests/test_bayesreg/pq/_world.py; CPU, tmp_path only): the PQ unit, the release, the evaluation unit, the reading.

- the PQ unit: m1-m4 on every pair before anything is computed, the outputs against an independent recomputation (each channel pulled back by the evaluation's own warp), opened_files.json and the list;
- the binding (C22.Q7 绑定): a byte of repair.npz, another field_0600.npz, a state_hash that differs, a labels_sha256 that differs, the provenance's segmenter_id, registry_sha256 and test_labels_seen,
  each refused before anything of the pair is computed;
- the firewall (C22.Q7 防火墙): a normal run passes with only fixed, moving, initial_u read from pair.npz; opening metrics_0600.json, aligned_seg35.nii.gz, the member fixed_seg35, or a watched file outside the
  allowed items stops the unit (no list, no evaluation); the counter-examples: other content under pair.npz's seg keys, the evaluation records deleted -> PQ's outputs and baselines bit for bit the same;
- the evaluation unit: refused before any truth without the release, with the list and the release in two commits, without 放行, with a list that is not the committed one, when the lock is there; e2
  failing stops before the truth; the truth only after the lock; e1 bitwise, within 1e-12, or failed;
- the reading on the outputs, written once.
"""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from bayesreg.seg import structure_layer

from . import _world as W
from .conftest import CODE, GATE
from ..io_seg._helpers import synthetic_folds
from ._world import reset_sealed_cache

core, io_, pins = W.core, W.io_, W.pins

import pq_eval as pe  # noqa: E402
import pq_firewall as fw  # noqa: E402
import run_pq  # noqa: E402


def quiet(*args, **kwargs):
    pass


def run_out(world, **kwargs):
    reset_sealed_cache()                                                          # exception (13) of C23.9-13 (revision 6f): each in-process call of the unit
    return run_pq.out_unit(world.ctx, gate=GATE, code=CODE, log=quiet, **kwargs)


def out_record(world):
    return io_.read_json(world.ctx.pq_root/'out_record.json')


def released(world):
    assert run_out(world) == 0
    sha = io_.sha_file(world.ctx.pq_root/io_.MANIFEST_FILE)
    W.release(world, sha)
    return sha


def run_eval(world, *, read_truth=None):
    return run_pq.eval_unit(world.ctx, gate=GATE, code=CODE, ev=W.EV, read_truth=read_truth or world.truth, guard=W.guard_of(world), log=quiet)


# ---- the PQ unit -----------------------------------------------------------------------------------------------------------------------------------
@synthetic_folds('world')
def test_the_pq_unit_end_to_end(world):
    assert run_out(world) == 0
    record = out_record(world)
    assert record['passed'] is True and all(record['controls'].values()) and record['pairs'] == list(W.SHORTS)
    opened = io_.read_json(world.ctx.pq_root/'opened_files.json')['out']
    assert opened['m6'] is True and not opened['breaches']
    assert all(set(m) == {'fixed', 'moving', 'initial_u'} for m in opened['pair_members'].values()) and len(opened['pair_members']) == 2
    names = {Path(f['path']).name for f in opened['files']}
    assert not names & {'metrics_0600.json', 'repair_metrics.json', 'aligned_seg35.nii.gz'}
    listed = io_.parse_manifest((world.ctx.pq_root/io_.MANIFEST_FILE).read_text())
    for short in W.SHORTS:
        for name in (io_.PQ_NPZ, io_.BASELINES_FILE, io_.PQ_JSON):
            assert listed[f'runs/{short}/{name}'] == io_.sha_file(world.ctx.pq_root/'runs'/short/name)
    assert 'opened_files.json' in listed and 'out_record.json' in listed and not any(k.startswith('eval/') for k in listed)
    with pytest.raises(core.PQRefused, match='once'):
        run_out(world)                                                               # once: pq_v1/runs exists


def independent_maps(world, short):
    """q, D^, the baselines from the files, through other code: each channel of P_M pulled back by warp_labels_l2r itself (outside: a pulled-back ones map is 0)."""
    from bayesreg.seg.channels import read_back_probabilities
    f, m = pins.subjects_of(short)
    P_F, _ = read_back_probabilities(world.ctx.pred_root/f/'probs36.npz')
    P_M, _ = read_back_probabilities(world.ctx.pred_root/m/'probs36.npz')
    v = world.fields[short]
    d = core.disp_voxels(v, (1., 1., 1.))
    inside = core.warp_labels_l2r(np.ones(W.SHAPE), d) == 1.
    P_t = np.stack([np.where(inside, core.warp_labels_l2r(P_M[c], d), 1. if c == 0 else 0.) for c in range(36)])
    q = 1.-(P_F*P_t).sum(axis=0)
    D = {c: 2*(P_F[c]*P_t[c]).sum()/(P_F[c].sum()+P_t[c].sum()) for c in range(1, 36)}
    L_F, L_M = core.argmax_smallest(P_F), core.argmax_smallest(P_M)
    L_t = core.warp_labels_l2r(L_M, d)
    defined = [c for c in range(1, 36) if (L_F == c).any() or (L_t == c).any()]
    H_F = -(np.where(P_F > 0, P_F*np.log(np.where(P_F > 0, P_F, 1.)), 0.)).sum(axis=0)
    return dict(q=q, D=D, defined=defined, S35=np.mean([D[c] for c in defined]), h=(L_F != L_t).astype(np.uint8), H_F=H_F, L_F=L_F, L_t=L_t)


@synthetic_folds('world')
def test_outputs_against_an_independent_recomputation(world):
    assert run_out(world) == 0
    for short in W.SHORTS:
        mine = independent_maps(world, short)
        with np.load(world.ctx.pq_root/'runs'/short/io_.PQ_NPZ) as z:
            q = np.array(z['q'])
        with np.load(world.ctx.pq_root/'runs'/short/io_.BASELINES_FILE) as z:
            base = {k: np.array(z[k]) for k in z.files}
        doc = io_.read_json(world.ctx.pq_root/'runs'/short/io_.PQ_JSON)
        assert np.max(np.abs(q.astype(np.float64)-mine['q'])) <= 1e-6 and q.dtype == np.float32
        assert np.array_equal(base['h'], mine['h']) and np.max(np.abs(base['H_F']-mine['H_F'])) <= 1e-6
        assert doc['numbers']['defined'] == mine['defined']
        for c in mine['defined']:
            assert doc['numbers']['D_hat'][str(c)] == pytest.approx(mine['D'][c], rel=1e-12)
        assert doc['numbers']['S35_hat'] == pytest.approx(mine['S35'], rel=1e-12)
        delta = np.minimum(core.boundary_distance(mine['L_F'], (1., 1., 1.)), core.boundary_distance(mine['L_t'], (1., 1., 1.)))
        assert base['delta'].tobytes() == delta.astype(np.float32).tobytes()
        assert doc['eps_em']['eps'] is not None and doc['inputs']['field']['state_hash'] == world.ctx.p3[short].state_hash


@synthetic_folds('world')
def test_m1_to_m4_of_every_pair_come_before_any_output(world):
    rep = world.ctx.repair_dir(W.SHORTS[1])/'repair.npz'
    raw = bytearray(rep.read_bytes())
    raw[-10] ^= 1
    rep.write_bytes(bytes(raw))
    assert run_out(world) == 8
    record = out_record(world)
    assert record['passed'] is False and record['stage'] == 'm1' and record['controls']['m1'] is False
    assert not (world.ctx.pq_root/'runs'/W.SHORTS[0]/io_.PQ_NPZ).exists() and not (world.ctx.pq_root/io_.MANIFEST_FILE).exists()


# ---- the binding -----------------------------------------------------------------------------------------------------------------------------------
def bound(world, short=W.SHORTS[0]):
    from bayesreg.io.guard import load_dev_allowlist
    allowlist = load_dev_allowlist(world.repo/pins.ALLOWLIST_REL)
    lookup = io_.load_lookup(world.repo, world.ctx.seg4_sha256)
    return io_.check_pair(world.ctx, short, allowlist, lookup)


def repin(world, short, **changes):
    world.ctx.p3 = dict(world.ctx.p3)
    world.ctx.p3[short] = world.ctx.p3[short]._replace(**changes)


@synthetic_folds('world')
def test_binding_refuses_a_changed_repair_npz_another_field_and_another_state(world):
    short, other = W.SHORTS
    assert bound(world)['m'] == dict(m1=True, m2=True, m3=True, m4=True)
    rep = world.ctx.repair_dir(short)/'repair.npz'
    keep = rep.read_bytes()
    raw = bytearray(keep)
    raw[-10] ^= 1
    rep.write_bytes(bytes(raw))
    with pytest.raises(core.PQRefused, match='m1.*repair.npz'):
        bound(world)
    rep.write_bytes(keep)
    field = world.ctx.t88_dir(short)/'field_0600.npz'
    keep_field = field.read_bytes()
    shutil.copy(world.ctx.t88_dir(other)/'field_0600.npz', field)
    with pytest.raises(core.PQRefused, match='m1.*field_0600'):
        bound(world)
    field.write_bytes(keep_field)
    # the chain itself, the pins updated with the change: the attempt's values changed, repair.json's attempt sha256 and the pins follow, the state_hash does not
    with np.load(rep) as z:
        nodes, values = np.array(z['nodes']), np.array(z['values'])
    np.savez(rep, nodes=nodes, values=values*1.0001)
    repair = io_.read_json(world.ctx.repair_dir(short)/'repair.json')
    repair['attempt']['sha256'] = io_.sha_file(rep)
    W.write_json(world.ctx.repair_dir(short)/'repair.json', repair)
    repin(world, short, repair_npz=io_.sha_file(rep), repair_json=io_.sha_file(world.ctx.repair_dir(short)/'repair.json'))
    with pytest.raises(core.PQRefused, match='m1.*sha256 of v'):
        bound(world)


def test_binding_refuses_a_state_hash_other_than_the_plans(world):
    repin(world, W.SHORTS[0], state_hash='0'*64)
    with pytest.raises(core.PQRefused, match='m1.*state_hash'):
        bound(world)


@synthetic_folds('world')
def test_labels_sha256_that_differs_is_refused(world):
    short = W.SHORTS[0]
    t88 = world.ctx.t88_dir(short)
    run = io_.read_json(t88/'run.json')
    run['labels_sha256']['moving'] = '1'*64
    W.write_json(t88/'run.json', run)
    world.ctx.run_json = dict(world.ctx.run_json, **{short: io_.sha_file(t88/'run.json')})
    with pytest.raises(core.PQRefused, match='m3.*labels_sha256'):
        bound(world)


@pytest.mark.parametrize('change, message', [(dict(segmenter_id='unet35_v1_ens'), 'm2'), (dict(registry_sha256='2'*64), 'm2.*registry_sha256'), (dict(test_labels_seen=True), 'm2')])
def test_provenance_that_is_not_the_anchors_is_refused(tmp_path, monkeypatch, change, message):
    w = W.build(tmp_path, prediction_changes={W.SUBJECTS[1]: change}, synthetic=True)          # C23.9-13 (8), revision 6e: the synthetic training draw
    monkeypatch.setattr(structure_layer, 'FORMAL_REGISTRIES', tuple(structure_layer.FORMAL_REGISTRIES)+(w.registry,))
    with pytest.raises(core.PQRefused, match=message):
        bound(w)


@synthetic_folds('world')
def test_a_provenance_or_probs_file_other_than_the_plans_is_refused(world):
    world.ctx.v4 = dict(world.ctx.v4)
    s = W.SUBJECTS[0]
    world.ctx.v4[s] = world.ctx.v4[s]._replace(probs36='3'*64)
    with pytest.raises(core.PQRefused, match="m2.*probs36"):
        bound(world)


def test_the_plan_is_pinned(world):
    plan = world.repo/pins.PLAN_REL
    plan.write_text(plan.read_text()+'\n')
    with pytest.raises(core.PQRefused, match='plan'):
        run_out(world)


# ---- the firewall in the unit ----------------------------------------------------------------------------------------------------------------------
def stopping_bind(target):
    real = io_.bind_field

    def bind(ctx, short, grid):
        target(ctx, short)
        return real(ctx, short, grid)
    return bind


@pytest.mark.parametrize('what', ['metrics_0600', 'aligned_seg35', 'fixed_seg35', 'unlisted'])
def test_a_breach_stops_the_unit_and_no_list_is_written(world, monkeypatch, what):
    def target(ctx, short):
        if what == 'metrics_0600':
            open(ctx.t88_dir(short)/'metrics_0600.json').read()
        elif what == 'aligned_seg35':
            open(Path(ctx.images)/W.SUBJECTS[0]/'aligned_seg35.nii.gz', 'rb').read()
        elif what == 'fixed_seg35':
            with np.load(ctx.stage/pins.pair_name(short)/'pair.npz') as z:
                z['fixed_seg35']
        else:
            open(Path(ctx.results)/'p3_v1'/'runs'/'C_first5'/short/'queue_done.json', 'w').write('{}')
    monkeypatch.setattr(io_, 'bind_field', stopping_bind(target))
    assert run_out(world) == 8
    record = out_record(world)
    assert record['stage'] == 'm6' and record['controls']['m6'] is False and record['passed'] is False
    opened = io_.read_json(world.ctx.pq_root/'opened_files.json')['out']
    assert opened['m6'] is False and len(opened['breaches']) == 1
    assert not (world.ctx.pq_root/io_.MANIFEST_FILE).exists()
    assert not (world.ctx.pq_root/'runs'/W.SHORTS[0]/io_.PQ_NPZ).exists()


def outputs(world):
    out = {}
    for short in W.SHORTS:
        for name in (io_.PQ_NPZ, io_.BASELINES_FILE):
            out[(short, name)] = (world.ctx.pq_root/'runs'/short/name).read_bytes()
        out[(short, 'numbers')] = io_.read_json(world.ctx.pq_root/'runs'/short/io_.PQ_JSON)['numbers']
    return out


def test_counter_examples_seg_keys_and_evaluation_records_change_nothing(tmp_path, monkeypatch):
    a = W.build(tmp_path/'a', synthetic=True)                                       # C23.9-13 (8), revision 6e: the synthetic training draw
    b = W.build(tmp_path/'b', synthetic=True)
    monkeypatch.setattr(structure_layer, 'FORMAL_REGISTRIES', tuple(structure_layer.FORMAL_REGISTRIES)+(a.registry, b.registry))
    for f, m in b.pairs:                                                             # other content under every seg key of pair.npz
        d = b.stage/f'{f}__{m}'
        with np.load(d/'pair.npz') as z:
            arrays = {k: np.array(z[k]) for k in z.files}
        rng = np.random.default_rng(99)
        for k in ('fixed_seg35', 'moving_seg35', 'fixed_seg4', 'moving_seg4'):
            arrays[k] = rng.integers(0, 36, arrays[k].shape).astype(float)
        np.savez_compressed(d/'pair.npz', **arrays)
        manifest = json.loads((d/'manifest.json').read_text())
        manifest['pair_sha256'] = io_.sha_file(d/'pair.npz')
        W.write_json(d/'manifest.json', manifest)
    for short in W.SHORTS:                                                           # the evaluation records deleted
        (b.ctx.t88_dir(short)/'metrics_0600.json').unlink()
        (b.ctx.repair_dir(short)/'repair_metrics.json').unlink()
    W.commit_all(b.repo, 'other seg keys')
    assert run_out(a) == 0 and run_out(b) == 0
    assert outputs(a) == outputs(b)


# ---- the evaluation unit ---------------------------------------------------------------------------------------------------------------------------
@synthetic_folds('world')
def test_evaluation_end_to_end_and_the_reading(world):
    released(world)
    reads = []

    def read_truth(pair):
        assert (world.ctx.pq_root/pe.EVAL_LOCK_NAME).is_file()                         # the truth only after the lock
        reads.append(pair)
        return world.truth(pair)
    assert run_eval(world, read_truth=read_truth) == 0
    assert reads == [pins.pair_name(s) for s in W.SHORTS]
    record = io_.read_json(world.ctx.pq_root/'eval'/'eval_record.json')
    assert record['e2']['passed'] and record['e1'] == {s: 'bitwise' for s in W.SHORTS} and record['error'] is None
    doc = io_.read_json(world.ctx.pq_root/'eval'/W.SHORTS[0]/'eval.json')
    assert set(doc['scores']) == set(pe.MASKS) and set(doc['scores']['omega_e']) == set(pe.SCORES)
    assert doc['scores']['omega_e']['q']['ause'] is not None and doc['masks']['whole'] == int(np.prod(W.SHAPE))
    labs, _, _ = world.truth(pins.pair_name(W.SHORTS[0]))
    w35 = core.warp_labels_l2r(labs['moving_seg35'], core.disp_voxels(world.fields[W.SHORTS[0]], (1., 1., 1.)))
    e = labs['fixed_seg35'] != w35
    mask = (labs['fixed_seg35'] > 0) | (w35 > 0)
    with np.load(world.ctx.pq_root/'runs'/W.SHORTS[0]/io_.PQ_NPZ) as z:
        q = np.array(z['q'])
    import pq_stats
    assert doc['scores']['omega_e']['q']['ause'] == pq_stats.ause([q[mask]], e[mask]) and doc['n_errors']['omega_e'] == int(e[mask].sum())
    assert record['status'] == 'complete' and record['complete'] is True and record['aborted'] is False and sorted(record['eval_sha256']) == sorted(W.SHORTS)
    assert run_pq.decide_unit(world.ctx, log=quiet) == 0
    decision = io_.read_json(world.ctx.pq_root/'analysis'/'decision_pq.json')
    assert decision['reading']['row'] in ('b_pass', 'b_fail') and decision['b_class']['judged'] and decision['truth_read'] is True
    assert decision['eval_controls'] == dict(e1='bitwise', e2=True, complete=True, integrity=True) and decision['integrity']['passed'] is True
    with pytest.raises(core.PQRefused, match='once'):
        run_pq.decide_unit(world.ctx, log=quiet)
    with pytest.raises(core.PQRefused, match='once'):
        run_eval(world)                                                              # the lock stays: never a second read


def never(pair):
    raise AssertionError('the truth was read')


@synthetic_folds('world')
def test_no_release_no_truth(world):
    assert run_out(world) == 0
    with pytest.raises(core.PQRefused, match='release'):
        run_eval(world, read_truth=never)
    assert not (world.ctx.pq_root/pe.EVAL_LOCK_NAME).exists() and not (world.ctx.pq_root/'eval').exists()


@pytest.mark.parametrize('how', ['two_commits', 'no_line', 'refused', 'not_yet', 'english_no', 'extra_text', 'two_lines', 'uncommitted', 'other_sha'])
@synthetic_folds('world')
def test_a_release_that_is_not_the_rule_is_refused_before_the_truth(world, how):
    """Revision 1 (review important 2): release.md is exactly the line 放行：<sha>; 不放行, 未放行, a 'no', more text, another sha: refused before the truth, nothing written."""
    assert run_out(world) == 0
    sha = io_.sha_file(world.ctx.pq_root/io_.MANIFEST_FILE)
    notes = dict(no_line=f'看过：{sha}\n', refused=f'不放行：{sha}\n', not_yet=f'未放行：{sha}\n', english_no=f'放行：{sha} no\n', extra_text=f'opened_files.json 看过，没有越界\n放行：{sha}\n',
                 two_lines=f'放行：{sha}\n放行：{sha}\n', other_sha=f'放行：{"e"*64}\n')
    if how == 'two_commits':
        W.release(world, sha, separate=True)
    elif how == 'uncommitted':
        W.release(world, sha)
        (world.repo/pe.RELEASE_NOTE_REL).write_text((world.repo/pe.RELEASE_NOTE_REL).read_text()+'\n')
    else:
        W.release(world, sha, note=notes[how])
    with pytest.raises(core.PQRefused, match='release'):
        run_eval(world, read_truth=never)
    assert not (world.ctx.pq_root/pe.EVAL_LOCK_NAME).exists() and not (world.ctx.pq_root/'eval').exists()


def test_the_release_line_rule():
    sha = 'a'*64
    assert pe.release_note_problem(f'放行：{sha}\n', sha) is None and pe.release_note_problem(f'放行：{sha}', sha) is None
    for note in (f'不放行：{sha}\n', f'放行：{sha}（否）\n', f'放行: {sha}\n', f'放行：{sha}\nnot yet\n', f'拒绝放行：{sha}\n', f' 放行：{"b"*64}\n', '放行\n', ''):
        assert pe.release_note_problem(note, sha) is not None, note


@synthetic_folds('world')
def test_a_list_that_is_not_the_committed_one_is_e2_and_counted_as_used(world):
    """Revision 1 (review important 3; plan.md section 8 e2, section 6 second row; the lead's ruling of 11-07 on e2): the committed value is not the list's sha256 -> e2 not passed, before the truth."""
    assert run_out(world) == 0
    W.release(world, 'f'*64)
    assert run_eval(world, read_truth=never) == 8
    record = io_.read_json(world.ctx.pq_root/'eval'/'eval_record.json')
    assert record['e2']['passed'] is False and record['e2']['list_is_committed'] is False and record['truth_read'] is False and record['status'] == 'e2_failed'
    assert not (world.ctx.pq_root/pe.EVAL_LOCK_NAME).exists()
    with pytest.raises(core.PQRefused, match='e2'):
        run_pq.decide_unit(world.ctx, log=quiet)                                       # no lock: no reading is written; the record says 不判, counted as used
    assert not (world.ctx.pq_root/'analysis').exists()
    with pytest.raises(core.PQRefused, match='once'):
        run_eval(world, read_truth=never)                                           # eval/ is there: not rerun
    import pq_stats
    r = pq_stats.reading(method_controls={k: True for k in pq_stats.METHOD_KEYS}, released=True, eval_controls=dict(e1=None, e2=False, complete=False, integrity=True), b=None, truth_read=False)
    assert r['row'] == 'eval_controls' and r['counts_as_reading'] is True


def stopping_reader(world, stop):
    calls = []

    def read_truth(pair):
        calls.append(pair)
        if len(calls) == 2:
            if stop == 'sigterm':
                import os
                import signal
                os.kill(os.getpid(), signal.SIGTERM)
                import time
                time.sleep(5)                                                        # the handler raises before this ends
            raise stop
        return world.truth(pair)
    return read_truth, calls


@pytest.mark.parametrize('stop', ['system_exit', 'keyboard', 'sigterm', 'error'])
@synthetic_folds('world')
def test_a_stop_after_the_lock_is_recorded_and_read_as_counted(world, stop):
    """Revision 1 (review important 1): after the lock, SystemExit (truth_labels' own refusal), KeyboardInterrupt, SIGTERM (systemd), an exception: eval_record.json says 真值已读、中止, the reading is the
    second row (不判, counted as used), never the first."""
    import signal
    released(world)
    exc = dict(system_exit=SystemExit('pair.npz sha256 differs'), keyboard=KeyboardInterrupt(), sigterm='sigterm', error=RuntimeError('broken'))[stop]
    read_truth, calls = stopping_reader(world, exc)
    before = signal.getsignal(signal.SIGTERM)
    if stop == 'error':
        assert run_eval(world, read_truth=read_truth) == 8
    else:
        kind = dict(system_exit=SystemExit, keyboard=KeyboardInterrupt, sigterm=run_pq.EvalStopped)[stop]
        with pytest.raises(kind):
            run_eval(world, read_truth=read_truth)
    assert signal.getsignal(signal.SIGTERM) == before                                   # the handler is put back
    record = io_.read_json(world.ctx.pq_root/'eval'/'eval_record.json')
    assert record['truth_read'] is True and record['aborted'] is True and record['complete'] is False and record['status'] == '真值已读、中止' and record['pairs'] == [W.SHORTS[0]]
    assert (world.ctx.pq_root/pe.EVAL_LOCK_NAME).exists()
    assert run_pq.decide_unit(world.ctx, log=quiet) == 0
    decision = io_.read_json(world.ctx.pq_root/'analysis'/'decision_pq.json')
    assert decision['reading']['row'] == 'eval_controls' and decision['reading']['counts_as_reading'] is True and decision['b_class'] is None


@synthetic_folds('world')
def test_a_lock_without_an_evaluation_record_is_counted_as_used(world):
    """A kill that cannot be caught (SIGKILL, the OOM killer): the lock is there, the record is not -> the second row."""
    released(world)
    (world.ctx.pq_root/pe.EVAL_LOCK_NAME).write_text('{"utc": "2026-11-07T20:00:00Z"}\n')
    assert run_pq.decide_unit(world.ctx, log=quiet) == 0
    decision = io_.read_json(world.ctx.pq_root/'analysis'/'decision_pq.json')
    assert decision['reading']['row'] == 'eval_controls' and decision['reading']['counts_as_reading'] is True and decision['truth_read'] is True


@synthetic_folds('world')
def test_the_reading_is_refused_before_the_evaluation(world):
    released(world)
    with pytest.raises(core.PQRefused, match='has not read the truth'):
        run_pq.decide_unit(world.ctx, log=quiet)
    assert not (world.ctx.pq_root/'analysis').exists()
    assert run_eval(world) == 0 and run_pq.decide_unit(world.ctx, log=quiet) == 0          # the reading is still the code's, after the evaluation


@synthetic_folds('world')
def test_a_pq_json_changed_after_the_list_is_not_judged(world):
    """Revision 1 (review minor 9): the reading re-hashes pq.json against the list and eval.json against the evaluation record."""
    released(world)
    assert run_eval(world) == 0
    target = world.ctx.pq_root/'runs'/W.SHORTS[0]/io_.PQ_JSON
    target.write_text(target.read_text()+' ')
    assert run_pq.decide_unit(world.ctx, log=quiet) == 0
    decision = io_.read_json(world.ctx.pq_root/'analysis'/'decision_pq.json')
    assert decision['integrity']['passed'] is False and decision['reading']['row'] == 'eval_controls' and decision['b_class'] is None


@synthetic_folds('world')
def test_opened_files_holds_the_files_before_the_inputs(world):
    """Revision 1 (review minor 1): the spy from the unit's first file: the plan and the members' record are in opened_files.json."""
    assert run_out(world) == 0
    files = {Path(f['path']) for f in io_.read_json(world.ctx.pq_root/'opened_files.json')['out']['files']}
    assert (world.repo/pins.PLAN_REL).resolve() in files and (world.ctx.pq_root/'members'/io_.MEMBERS_RECORD).resolve() in files


@synthetic_folds('world')
def test_a_lock_that_is_there_refuses_before_anything(world):
    released(world)
    (world.ctx.pq_root/pe.EVAL_LOCK_NAME).write_text('{}')
    with pytest.raises(core.PQRefused, match='once'):
        run_eval(world, read_truth=never)


@synthetic_folds('world')
def test_e2_failing_stops_before_the_truth(world):
    released(world)
    target = world.ctx.pq_root/'runs'/W.SHORTS[1]/io_.BASELINES_FILE
    raw = bytearray(target.read_bytes())
    raw[-5] ^= 1
    target.write_bytes(bytes(raw))
    assert run_eval(world, read_truth=never) == 8
    record = io_.read_json(world.ctx.pq_root/'eval'/'eval_record.json')
    assert record['e2']['passed'] is False and record['truth_read'] is False and record['e2']['differing'] == [f'runs/{W.SHORTS[1]}/{io_.BASELINES_FILE}']
    assert not (world.ctx.pq_root/pe.EVAL_LOCK_NAME).exists()


@synthetic_folds('world')
def test_a_sealed_subject_is_refused_before_the_truth(world):
    released(world)

    class Sealed(W.NoSealed):
        def which(self, subject):
            return 'holdout_B_v1' if subject == W.SUBJECTS[2] else None
    guard = dict(W.guard_of(world), sealed=Sealed())
    with pytest.raises(Exception, match='sealed'):
        run_pq.eval_unit(world.ctx, gate=GATE, code=CODE, ev=W.EV, read_truth=never, guard=guard, log=quiet)
    assert not (world.ctx.pq_root/pe.EVAL_LOCK_NAME).exists()
    assert not (world.ctx.pq_root/'eval').exists()                                    # revision 1 (review minor 5): a refusal of the guard leaves nothing; the evaluation may start later
    assert run_eval(world) == 0


def test_e1_bitwise_within_and_beyond_the_tolerance():
    actual = dict(seg35=.81, seg4=.9, seg35_per_label={c: .5+c/100 for c in range(1, 36)}, seg4_per_label={g: .8 for g in range(1, 5)})
    recorded = dict(seg35=.81, seg4=.9, seg35_per_label={str(c): .5+c/100 for c in range(1, 36)}, seg4_per_label={str(g): .8 for g in range(1, 5)})
    assert pe.e1_compare(actual, recorded)[0] == 'bitwise'
    near = json.loads(json.dumps(recorded))
    near['seg4_per_label']['2'] = .8+5e-13
    assert pe.e1_compare(actual, near)[0] == 'within_1e-12'
    far = json.loads(json.dumps(recorded))
    far['seg35'] = .81+1e-9
    assert pe.e1_compare(actual, far)[0] is False
    missing = json.loads(json.dumps(recorded))
    del missing['seg35_per_label']['7']
    assert pe.e1_compare(actual, missing)[0] is False
    a_none, r_none = dict(actual, seg35_per_label=dict(actual['seg35_per_label'])), json.loads(json.dumps(recorded))
    a_none['seg35_per_label'][3], r_none['seg35_per_label']['3'] = None, None          # a label in neither image: None on both sides is equal
    assert pe.e1_compare(a_none, r_none)[0] == 'bitwise'
    r_none['seg35_per_label']['3'] = .53
    assert pe.e1_compare(a_none, r_none)[0] is False


def test_the_frozen_taper_is_the_formula_of_the_world():
    """The frozen tree's boundary_taper against the replica the synthetic world uses (skipped where the frozen tree is not on the machine); a subprocess, the frozen tree first on its path."""
    import subprocess
    import sys
    frozen = pins.FROZEN
    if not (frozen/'jd_curl'/'three_d'/'preopt.py').is_file():
        pytest.skip('the frozen tree is not on this machine')
    code = ('import sys, numpy as np; sys.path.insert(0, sys.argv[1]); from jd_curl.three_d import preopt; from jd_curl.three_d.grid import make_grid_3d; '
            'g = make_grid_3d(nx=9, ny=12, nz=10, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit="mm"); u = np.random.default_rng(1).normal(size=(10, 12, 9, 3)); '
            'np.save(sys.argv[2], np.ascontiguousarray(preopt.boundary_taper(u, g, 4)[0]))')
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)/'t.npy'
        result = subprocess.run([sys.executable, '-B', '-c', code, str(frozen), str(out)], capture_output=True, text=True, timeout=300, env=dict(__import__('os').environ, PYTHONPATH=str(frozen)))
        if result.returncode != 0:
            pytest.skip(f'the frozen tree could not be imported here: {result.stderr[-300:]}')
        theirs = np.load(out)
    u = np.random.default_rng(1).normal(size=(10, 12, 9, 3))
    mine, _ = W.frozen_formula_taper(u, W.SimpleNamespace(shape=(10, 12, 9)))
    assert theirs.tobytes() == np.ascontiguousarray(mine).tobytes()


# ---- revision 2 (diff review minors 1-5) -------------------------------------------------------------------------------------------------------------
def evaluated(world):
    released(world)
    assert run_eval(world) == 0


@pytest.mark.parametrize('which', ['record', 'eval_json', 'record_empty'])
@synthetic_folds('world')
def test_a_truncated_record_or_eval_json_is_read_as_missing_and_counted_as_used(world, which):
    """Minor 1: a record or an eval.json cut short (a kill during its write; write_once now writes a temporary file and links it, so this needs a damaged file) is taken as missing: the second row."""
    evaluated(world)
    target = world.ctx.pq_root/'eval'/('eval_record.json' if which != 'eval_json' else f'{W.SHORTS[1]}/eval.json')
    text = target.read_text()
    target.write_text('' if which == 'record_empty' else text[:len(text)//2])
    assert run_pq.decide_unit(world.ctx, log=quiet) == 0
    decision = io_.read_json(world.ctx.pq_root/'analysis'/'decision_pq.json')
    assert decision['reading']['row'] == 'eval_controls' and decision['reading']['counts_as_reading'] is True and decision['b_class'] is None
    assert any(str(target) in u for u in decision['unreadable'])


def test_write_once_is_atomic(tmp_path, monkeypatch):
    import os
    target = tmp_path/'r'/'eval_record.json'
    io_.write_once(target, dict(a=1))
    assert io_.read_json(target) == dict(a=1) and sorted(p.name for p in target.parent.iterdir()) == ['eval_record.json']
    with pytest.raises(FileExistsError):
        io_.write_once(target, dict(a=2))
    other = tmp_path/'r'/'eval.json'

    def killed(src, dst):
        raise KeyboardInterrupt('killed between the write and the link')
    monkeypatch.setattr(os, 'link', killed)
    with pytest.raises(KeyboardInterrupt):
        io_.write_once(other, dict(b=1))
    assert not other.exists() and sorted(p.name for p in target.parent.iterdir()) == ['eval_record.json']     # no partial file under the name, the temporary one removed


@pytest.mark.parametrize('how', ['lock_removed', 'lock_removed_no_record', 'record_says_not_read'])
@synthetic_folds('world')
def test_contradicting_marks_of_the_truth_are_refused(world, how):
    """Minor 2: the lock, eval_record.json's truth_read and the eval files are the three marks; when they contradict each other the reading is refused and the contradiction named (never the first row)."""
    evaluated(world)
    lock = world.ctx.pq_root/pe.EVAL_LOCK_NAME
    record = world.ctx.pq_root/'eval'/'eval_record.json'
    if how == 'lock_removed':
        lock.unlink()
    elif how == 'lock_removed_no_record':
        lock.unlink()
        record.unlink()
    else:
        document = io_.read_json(record)
        record.unlink()
        W.write_json(record, dict(document, truth_read=False))
    with pytest.raises(core.PQRefused, match='contradict'):
        run_pq.decide_unit(world.ctx, log=quiet)
    assert not (world.ctx.pq_root/'analysis').exists()


@pytest.mark.parametrize('how', ['bom', 'upper'])
@synthetic_folds('world')
def test_a_release_with_a_bom_passes_and_upper_case_hex_is_refused_with_the_rule(world, how):
    """Minor 3: release.md is read as utf-8-sig; the refusal names the full-width colon and the lowercase hex."""
    assert run_out(world) == 0
    sha = io_.sha_file(world.ctx.pq_root/io_.MANIFEST_FILE)
    if how == 'upper':
        W.release(world, sha, note=f'放行：{sha.upper()}\n')
        with pytest.raises(core.PQRefused, match='全角冒号.*小写十六进制'):
            run_eval(world, read_truth=never)
        assert not (world.ctx.pq_root/'eval').exists()
    else:
        W.release(world, sha, note=f'﻿放行：{sha}\r\n')
        assert run_eval(world) == 0


@pytest.mark.parametrize('committed', ['the_original', 'the_broken'])
@synthetic_folds('world')
def test_a_list_that_cannot_be_parsed_is_e2(world, committed):
    """Minor 4: the list's sha256 is compared first; a list that cannot be parsed is e2 not passed (清单格式坏), counted as used, before the truth."""
    assert run_out(world) == 0
    listed = world.ctx.pq_root/io_.MANIFEST_FILE
    sha = io_.sha_file(listed)
    lines = listed.read_text().splitlines(keepends=True)
    listed.chmod(0o644)
    listed.write_text(''.join(lines+[lines[0]]))                                       # a repeated line
    W.release(world, sha if committed == 'the_original' else io_.sha_file(listed))
    assert run_eval(world, read_truth=never) == 8
    record = io_.read_json(world.ctx.pq_root/'eval'/'eval_record.json')
    assert record['e2']['passed'] is False and record['e2']['problem'].startswith('清单格式坏') and record['truth_read'] is False and record['status'] == 'e2_failed'
    assert record['e2']['list_is_committed'] is (committed == 'the_broken')
    assert not (world.ctx.pq_root/pe.EVAL_LOCK_NAME).exists()


def test_a_breach_in_the_window_before_the_inner_try_is_recorded(world, monkeypatch):
    """Minor 5: a breach after the spy is installed and before the unit's inner try (here inside the members' record check) writes opened_files.json and out_record.json marked 越界、尚无输出."""
    real = io_.members_record

    def peeking(ctx):
        open(ctx.t88_dir(W.SHORTS[0])/'metrics_0600.json').read()
        return real(ctx)
    monkeypatch.setattr(io_, 'members_record', peeking)
    assert run_out(world) == 8
    record = io_.read_json(world.ctx.pq_root/'out_record.json')
    opened = io_.read_json(world.ctx.pq_root/'opened_files.json')
    assert record['status'] == run_pq.WINDOW_BREACH and record['passed'] is False and record['stage'] == 'm6' and 'Breach' in record['error']
    assert opened['status'] == run_pq.WINDOW_BREACH and len(opened['breaches']) == 1 and opened['breaches'][0]['path'].endswith('metrics_0600.json')
    assert not (world.ctx.pq_root/'runs').exists() and not (world.ctx.pq_root/io_.MANIFEST_FILE).exists()
    with pytest.raises(core.PQRefused, match='once'):
        run_out(world)                                                               # kept, not rerun
