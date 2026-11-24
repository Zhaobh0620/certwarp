"""PQH, the PQ unit (contract C22.H3, H4, H6; C22.H8 实现门槛 算术不变, 绑定, 次序, 防火墙) in its dev mode on the PQ package's synthetic world (CPU, tmp_path):

- 算术不变: the hold-out unit's pq.npz and baselines.npz are bit for bit the development unit's (run_pq.out_unit on the same pair, the same v, posteriors and V);
- the outputs: pq/<pair>/pq.json on the record root, pq.npz and baselines.npz on the big-file root, the sha256 list naming every file that counts by its absolute path, line.json unchanged;
- the order: no finalization record -> refused; the run once (the lock); a pair directory there -> refused;
- the controls: a changed byte of any method file, a d that differs, a rebuilt v that differs -> the unit stops with pq_refused.json and writes no PQ output; the dev mode never reruns;
- the firewall: a watched file outside the allowed items opened during the run -> Hm6, the unit stops; before the lock a breach, a filled sealed-list cache and a failed members record stop the
  unit with pq_refused.json too (review m3).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from . import _world as W

hp, hin, hu, io_, core = W.hp, W.hin, W.hu, W.io_, W.core


@pytest.fixture
def dev(tmp_path, monkeypatch):
    w, layout, entry = W.dev_world(tmp_path, monkeypatch)
    return W.namespace(w=w, layout=layout, entry=entry, tmp=tmp_path)


def development_arrays(dev):
    """run_pq.out_unit on the same world (the development unit): its pq.npz and baselines.npz of 0438-0439."""
    import run_pq
    W.XW.sealed_module._reset_cache()
    assert run_pq.out_unit(dev.w.ctx, gate=W.GATE, code=W.CODE, log=W.quiet) == 0
    out = {}
    for name in (io_.PQ_NPZ, io_.BASELINES_FILE):
        with np.load(dev.w.ctx.pq_root/'runs'/W.SHORT/name) as z:
            out.update({k: np.array(z[k]) for k in z.files})
    return out


def holdout_arrays(dev):
    small, big = dev.layout.pq_small, dev.layout.pq_big
    out = {}
    for name in (hu.PQ_NPZ, hu.BASELINES):
        with np.load(big/W.PAIR/name) as z:
            out.update({k: np.array(z[k]) for k in z.files})
    return out


def test_the_unit_end_to_end_and_its_arithmetic_is_the_developments(dev):
    line_before = io_.sha_file(dev.layout.dev_cell/'output'/'line.json')
    assert W.run_out(dev.layout, deps=W.deps()) == 0
    mine, theirs = holdout_arrays(dev), development_arrays(dev)
    assert sorted(mine) == sorted(theirs)
    for k in mine:
        assert mine[k].dtype == theirs[k].dtype and mine[k].tobytes() == theirs[k].tobytes(), k
    document = io_.read_json(dev.layout.pq_small/W.PAIR/hu.PQ_JSON)
    assert document['settings_sha256'] == hp.settings_sha256() and document['hm']['rebuilt_equals_final'] is True and document['hm']['labels_checked_against_run_json'] is True
    assert document['line_status'] == 'hx_certified' and document['sampling']['d_sha256'] == io_.read_json(dev.layout.dev_cell/'output'/'line.json')['output']['disp']['d_sha256']
    assert document['files'][hu.PQ_NPZ]['sha256'] == io_.sha_file(dev.layout.pq_big/W.PAIR/hu.PQ_NPZ)
    assert io_.sha_file(dev.layout.dev_cell/'output'/'line.json') == line_before                  # D114 (a)
    record = io_.read_json(dev.layout.pq_small/hu.OUT_RECORD)
    assert record['complete'] is True and record['controls'] == dict(hm1=True, hm2=True, hm3=True, hm4=True, hm5=True, hm6=True)
    listed = (dev.layout.pq_small/hu.MANIFEST).read_text().splitlines()
    paths = {line.split('  ', 1)[1] for line in listed}
    for f in (dev.layout.pq_small/W.PAIR/hu.PQ_JSON, dev.layout.pq_big/W.PAIR/hu.PQ_NPZ, dev.layout.pq_big/W.PAIR/hu.BASELINES, dev.layout.pq_small/hu.OUT_RECORD,
              dev.layout.pq_small/'members'/hin.MEMBERS_RECORD):
        assert str(f) in paths
    for line in listed:
        digest, path = line.split('  ', 1)
        assert io_.sha_file(path) == digest
    opened = io_.read_json(dev.layout.pq_small/'opened_files_out.json')
    assert opened['m6'] is True and opened['breaches'] == []
    assert all(set(m) <= {'fixed', 'moving', 'initial_u'} for m in opened['pair_members'].values()) and opened['pair_members']
    seen = {Path(f['path']).name for f in opened['files']}
    assert not seen & {'metrics_0600.json', 'repair_metrics.json', 'aligned_seg35.nii.gz'}


def test_the_unit_runs_once(dev):
    assert W.run_out(dev.layout, deps=W.deps()) == 0
    assert W.run_out(dev.layout, deps=W.deps()) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(dev.layout)[-1])
    assert refusal['stage'] == 'once' and refusal['pair'] == 'out' and refusal['tool'] == hp.TOOL


def test_no_finalization_record_refuses_before_any_subject(dev):
    dev.layout.finalized.unlink()
    assert W.run_out(dev.layout, deps=W.deps()) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(dev.layout)[-1])
    assert refusal['stage'] == 'order'
    assert not [f for f in refusal['spy']['files'] if f['path'].endswith(('pair.npz', 'probs36.npz', 'final.npz'))]
    assert not (dev.layout.pq_small/W.PAIR).exists() and not Path(dev.layout.out_lock).exists()


@pytest.mark.parametrize('rel', ['output/final.npz', 'output/cert.json', 'output/line.json', 'output/disp.npz', 't88/run.json', 't88/field_0600.npz', 'repair/repair.json', 'repair/repair.npz'])
def test_a_changed_byte_of_any_method_file_stops_the_unit_without_output(dev, rel):
    path = dev.layout.dev_cell/rel
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    assert W.run_out(dev.layout, deps=W.deps()) == hu.EXIT['stopped']
    stop = io_.read_json(dev.layout.pq_small/hu.STOPPED)
    assert stop['stage'] == 'hm1' and 'finalization record' in stop['error']
    assert not (dev.layout.pq_small/W.PAIR).exists() and not (dev.layout.pq_big/W.PAIR).exists() and not (dev.layout.pq_small/hu.MANIFEST).exists()
    assert W.run_out(dev.layout, deps=W.deps()) == hu.EXIT['refused']                     # the dev mode never reruns (C22.H8)
    assert io_.read_json(W.refusals(dev.layout)[-1])['stage'] == 'rerun'


def rewrite_output(dev, *, d=None, u=None):
    """The output rewritten consistently (final.npz / disp.npz, cert.json, line.json and the finalization record agree), so that only the property under test differs."""
    out = dev.layout.dev_cell/'output'
    with np.load(out/'final.npz') as z:
        final = {k: np.array(z[k]) for k in z.files}
    with np.load(out/'disp.npz') as z:
        disp = {k: np.array(z[k]) for k in z.files}
    if u is not None:
        final['u'] = u
        final['state_hash'] = np.array(io_.sha_array(u))
    if d is not None:
        disp['d'] = d
    (out/'final.npz').unlink()
    np.savez(out/'final.npz', **final)
    (out/'disp.npz').unlink()
    np.savez(out/'disp.npz', **disp)
    state = str(final['state_hash'].item())
    cert = json.loads((out/'cert.json').read_text())
    cert['state_hash'] = state
    (out/'cert.json').write_text(json.dumps(cert))
    line = json.loads((out/'line.json').read_text())
    line['output']['final'].update(sha256=io_.sha_file(out/'final.npz'), state_hash=state)
    line['output']['disp']['sha256'] = io_.sha_file(out/'disp.npz')
    line['output']['cert']['sha256'] = io_.sha_file(out/'cert.json')
    (out/'line.json').write_text(json.dumps(line))
    W.write_records(dev.layout.dev_cell, W.PAIR, status='hx_certified', state_hash=state)          # HB's review record and terminal.json of the rewritten output


def test_a_d_one_bit_off_stops_the_unit(dev):
    with np.load(dev.layout.dev_cell/'output'/'disp.npz') as z:
        d = np.array(z['d'])
    d.reshape(-1)[123] = np.nextafter(d.reshape(-1)[123], np.inf)
    rewrite_output(dev, d=d)
    assert W.run_out(dev.layout, deps=W.deps()) == hu.EXIT['stopped']
    assert io_.read_json(dev.layout.pq_small/hu.STOPPED)['stage'] == 'hm4'


def test_a_d_not_float64_is_refused(dev):
    with np.load(dev.layout.dev_cell/'output'/'disp.npz') as z:
        d = np.array(z['d'])
    rewrite_output(dev, d=d.astype(np.float32))
    assert W.run_out(dev.layout, deps=W.deps()) == hu.EXIT['stopped']
    stop = io_.read_json(dev.layout.pq_small/hu.STOPPED)
    assert stop['stage'] == 'hm4' and 'float64' in stop['error']


def test_a_final_state_other_than_the_rebuilt_v_stops_the_unit(dev):
    with np.load(dev.layout.dev_cell/'output'/'final.npz') as z:
        u = np.array(z['u'])
    u.reshape(-1)[3000] = np.nextafter(u.reshape(-1)[3000], np.inf)
    rewrite_output(dev, u=u)
    assert W.run_out(dev.layout, deps=W.deps()) == hu.EXIT['stopped']
    stop = io_.read_json(dev.layout.pq_small/hu.STOPPED)
    assert stop['stage'] == 'hm1' and ('differs from final.npz' in stop['error'] or 'state_hash' in stop['error'])


def test_a_watched_file_outside_the_allowed_items_stops_the_unit(dev, monkeypatch):
    decoy = dev.tmp/'results'/'p3_v1'/'runs'/'C_first5'/W.SHORT/'metrics_0600.json'
    original = hin.check_disp

    def opening(*args, **kwargs):
        decoy.read_text()
        return original(*args, **kwargs)
    monkeypatch.setattr(hin, 'check_disp', opening)
    assert W.run_out(dev.layout, deps=W.deps()) == hu.EXIT['stopped']
    stop = io_.read_json(dev.layout.pq_small/hu.STOPPED)
    assert stop['stage'] == 'hm6'
    opened = io_.read_json(dev.layout.pq_small/'opened_files_out.json')
    assert opened['m6'] is False and opened['breaches'][0]['path'].endswith('metrics_0600.json')
    assert not (dev.layout.pq_small/hu.MANIFEST).exists()


def test_a_pair_npz_label_member_is_a_breach(dev, monkeypatch):
    original = hin.load_pair_checked

    def reading_labels(layout, pair, allowlist):
        pin = original(layout, pair, allowlist)
        with np.load(Path(layout.stage)/pair/'pair.npz') as z:
            z['fixed_seg35']
        return pin
    monkeypatch.setattr(hin, 'load_pair_checked', reading_labels)
    assert W.run_out(dev.layout, deps=W.deps()) == hu.EXIT['stopped']
    assert io_.read_json(dev.layout.pq_small/hu.STOPPED)['stage'] == 'hm6'


def test_a_populated_sealed_list_cache_stops_the_unit_before_the_spy(dev):
    """C22.H4-3 item 2 (review m3): the cache filled before the spy -> pq_refused.json ("名单缓存在装 spy 之前已经有值"), Hm6, BZ decides; no lock, nothing of a subject read."""
    from bayesreg.io.sealed import guard_sealed_lists
    W.XW.sealed_module._reset_cache()
    guard_sealed_lists()
    assert hu.out_unit(dev.layout, deps=W.deps()) == hu.EXIT['stopped']
    stop = io_.read_json(dev.layout.pq_small/hu.STOPPED)
    assert stop['stage'] == 'hm6' and stop['before_lock'] is True and stop['spy'] is None and 'sealed list cache' in stop['error'] and not Path(dev.layout.out_lock).exists()
    assert W.refusals(dev.layout) == [] and hu.previous_stop_records(dev.layout, 'out') == [dev.layout.pq_small/hu.STOPPED]


def test_the_sealed_lists_are_read_under_the_spy_and_recorded(dev):
    assert W.run_out(dev.layout, deps=W.deps(), ) == 0
    from bayesreg.io import sealed
    lists = {str((Path(sealed.REPO)/s['path']).resolve()) for s in sealed.SEALED_LISTS}
    opened = {f['path'] for f in io_.read_json(dev.layout.pq_small/'opened_files_out.json')['files']}
    assert lists <= opened


def test_a_machine_refusal_is_an_infrastructure_refusal_record(dev):
    def busy(unit):
        raise W.hc.Refused('machine', 'another unit is active')
    assert W.run_out(dev.layout, deps=W.deps(gate=busy)) == hu.EXIT['infra']
    refusal = io_.read_json(W.refusals(dev.layout)[-1])
    assert refusal['refusal_class'] == 'infrastructure' and refusal['stage'] == 'machine' and not Path(dev.layout.out_lock).exists()


def test_outside_its_unit_the_unit_refuses(dev):
    assert W.run_out(dev.layout, deps=W.deps(in_unit=lambda name: False)) == hu.EXIT['refused']
    assert io_.read_json(W.refusals(dev.layout)[-1])['stage'] == 'unit'


def test_without_the_members_record_the_unit_refuses_on_the_order(dev):
    (dev.layout.pq_small/'members'/hin.MEMBERS_RECORD).unlink()
    assert W.run_out(dev.layout, deps=W.deps()) == hu.EXIT['refused']
    assert io_.read_json(W.refusals(dev.layout)[-1])['stage'] == 'order' and not (dev.layout.pq_small/hu.STOPPED).exists()


def test_a_members_spy_breach_stops_the_pq_unit(dev):
    """Review m3: a members record whose spy failed is a control (Hm5/Hm6) found before the lock: pq_refused.json, BZ decides (C22.H4-2 last item)."""
    path = dev.layout.pq_small/'members'/hin.MEMBERS_RECORD
    record = json.loads(path.read_text())
    record['spy']['m6'] = False
    path.write_text(json.dumps(record))
    assert W.run_out(dev.layout, deps=W.deps()) == hu.EXIT['stopped']
    stop = io_.read_json(dev.layout.pq_small/hu.STOPPED)
    assert stop['stage'] == 'hm5' and 'Hm6' in stop['error'] and stop['before_lock'] is True and not Path(dev.layout.out_lock).exists()


def test_a_breach_before_the_lock_stops_the_unit(dev, monkeypatch):
    """Review m3: a breach of the spy before the lock is Hm6 (pq_refused.json with the spy's record), not an infrastructure refusal."""
    decoy = dev.tmp/'results'/'p3_v1'/'runs'/'C_first5'/W.SHORT/'metrics_0600.json'
    original = hu.admit_pairs

    def opening(*args, **kwargs):
        decoy.read_text()
        return original(*args, **kwargs)
    monkeypatch.setattr(hu, 'admit_pairs', opening)
    assert W.run_out(dev.layout, deps=W.deps()) == hu.EXIT['stopped']
    stop = io_.read_json(dev.layout.pq_small/hu.STOPPED)
    assert stop['stage'] == 'hm6' and stop['spy']['m6'] is False and stop['spy']['breaches'][0]['path'].endswith('metrics_0600.json') and not Path(dev.layout.out_lock).exists()


def test_a_pair_without_the_methods_output_gets_pq_absent(dev):
    terminal = dev.layout.dev_cell/'terminal.json'
    document = json.loads(terminal.read_text())
    document.update(line_status='hx_implementation_error', output_dir=None)
    terminal.write_text(json.dumps(document))
    assert W.run_out(dev.layout, deps=W.deps()) == 0
    absent = io_.read_json(dev.layout.pq_small/W.PAIR/hu.ABSENT)
    assert absent['line_status'] == 'hx_implementation_error' and not (dev.layout.pq_big/W.PAIR).exists()
    assert str(dev.layout.pq_small/W.PAIR/hu.ABSENT) in (dev.layout.pq_small/hu.MANIFEST).read_text()


def test_pq_json_records_the_structure_layers_d24_preflight(dev):
    assert W.run_out(dev.layout, deps=W.deps()) == 0
    document = io_.read_json(dev.layout.pq_small/W.PAIR/hu.PQ_JSON)
    assert document['inputs']['structure_layer']['preflight']['contact_union_contains_pair_subject'] is False
    assert document['method_files']['review/review.json']['sha256'] == io_.sha_file(dev.layout.dev_cell/'review'/'review.json') and document['code']['commit'] == W.CODE['commit']
    assert document['method_files']['terminal.json']['sha256'] == io_.sha_file(dev.layout.dev_cell/'terminal.json') and document['method']['review']['consistent'] is True
