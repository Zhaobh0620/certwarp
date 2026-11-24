"""SM3 (runs/20261114-p0/任务书.md; runs/20261109-pqh/smoke_plan_pq_segments.md §4 item 1, the lead's default "加"): PQ's dev-mode out unit on the fallback pairs that the smoke's (iv-b) and (iv-c)
leave, each on its own copy, with its own lock, records and unit; the members run of (i) shared.  And the gate of the dev-mode units: this process's own unit (the P3 replay's was another's), the
smoke's v3 queue as a parent unit in the dev mode only.  CPU, tmp_path, the PQ package's synthetic world."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from . import _world as W

hp, hin, hu, io_ = W.hp, W.hin, W.hu, W.io_


def test_the_dev_layout_of_each_out_segment():
    smoke = Path(hp.SMOKE_ROOT)
    ii, ivb, ivc = (hin.dev_layout(W.ROOT, segment=s) for s in ('ii', 'iv-b', 'iv-c'))
    assert ii.dev_cell == smoke/hp.DEV_CELL_REL and ii.out_lock == Path(hp.DEV_OUT_LOCK) and ii.out_subdir is None and ii.out_unit is None
    assert hin.run_dirs(ii, 'out', 'first_run') == (Path(hp.DEV_PQ_ROOT), Path(hp.DEV_PQ_ROOT)/'big')
    for layout, seg, unit in ((ivb, 'iv-b', 'tag-hb-pq-out-dev-ivb'), (ivc, 'iv-c', 'tag-hb-pq-out-dev-ivc')):
        assert layout.dev_cell == smoke/seg/'a'/'0438-0439' and layout.out_lock == Path(hp.DEV_PQ_ROOT)/f'pq_out_dev_{seg}.lock' and layout.out_unit == unit
        assert hin.run_dirs(layout, 'out', 'first_run') == (Path(hp.DEV_PQ_ROOT)/seg, Path(hp.DEV_PQ_ROOT)/'big'/seg)
        assert hin.run_dirs(layout, 'members', 'first_run') == hin.run_dirs(ii, 'members', 'first_run')                  # (i)'s members run, shared
        assert (layout.pq_small, layout.pq_big, layout.members_lock, layout.batch_root) == (ii.pq_small, ii.pq_big, ii.members_lock, ii.batch_root)
        assert hu.unit_name_of(layout, 'out') == unit and hu.unit_name_of(layout, 'members') == hp.DEV_UNITS['members']
    assert hu.unit_name_of(ii, 'out') == hp.DEV_UNITS['out'] and hu.unit_name_of(hin.dev_layout(W.ROOT), 'out') == 'tag-hb-pq-out-dev'
    with pytest.raises(ValueError):
        hin.dev_layout(W.ROOT, segment='iv-a')                                          # (iv-a) ends hx_certified: PQ is (ii)'s already
    assert len({ii.out_lock, ivb.out_lock, ivc.out_lock}) == 3


def test_run_pqhs_commands_of_the_segments():
    import run_pqh
    assert run_pqh.COMMANDS['out-dev-ivb'] == run_pqh.COMMANDS['out-dev-ivc'] == ('dev', 'out')
    assert {c: run_pqh.unit_name(c) for c in ('out-dev', 'out-dev-ivb', 'out-dev-ivc', 'members-dev', 'p3-replay')} == {
        'out-dev': 'tag-hb-pq-out-dev', 'out-dev-ivb': 'tag-hb-pq-out-dev-ivb', 'out-dev-ivc': 'tag-hb-pq-out-dev-ivc', 'members-dev': 'tag-hb-pq-members-dev', 'p3-replay': 'tag-hb-pq-p3-replay'}
    assert run_pqh.layout_for('dev', W.ROOT, 'out-dev-ivc').dev_cell == Path(hp.SMOKE_ROOT)/'iv-c'/'a'/'0438-0439'
    assert run_pqh.layout_for('dev', W.ROOT, 'p3-replay').dev_cell == run_pqh.layout_for('dev', W.ROOT, 'out-dev').dev_cell == Path(hp.SMOKE_ROOT)/hp.DEV_CELL_REL
    assert run_pqh.units_dir('out-dev-ivb', W.ROOT) == run_pqh.units_dir('out-dev', W.ROOT) == Path(hp.DEV_PQ_ROOT)/'units'
    argv = run_pqh.start_argv('out-dev-ivb', code=Path('/c'), record_dir=Path('/r'))
    assert '--unit=tag-hb-pq-out-dev-ivb' in argv and argv[-1] == 'out-dev-ivb' and argv[argv.index('--unit')+1] == 'tag-hb-pq-out-dev-ivb'
    assert f"--property=MemoryMax={hp.UNIT_SPECS['out']['memory_max']}" in argv and '--setenv=PYTHONPATH=/home/rmcfz/seq_vp_tag_results/frozen_eb808be:/c' in argv


@pytest.mark.parametrize('mode', ['dev', 'holdout'])
def test_the_gate_takes_this_processs_own_unit_and_the_smoke_queue_as_a_parent_in_the_dev_mode_only(monkeypatch, mode):
    seen = []
    monkeypatch.setattr(W.hc, 'machine_gate', lambda **kw: seen.append(kw) or dict(passed=True))
    monkeypatch.setattr(W.hc, 'wait_for', lambda check, **kw: check())
    deps = W.deps(gate=None)
    hu.gate_of('out', mode, deps)
    hu.gate_of('out', mode, deps, unit_name='tag-hb-pq-p3-replay')
    assert seen[0]['own_units'] == (hu.unit_names(mode)['out'],) and seen[1]['own_units'] == ('tag-hb-pq-p3-replay',)
    assert all(kw['parent_units'] == (tuple(hp.DEV_PARENT_UNITS) if mode == 'dev' else ()) for kw in seen)
    assert hp.DEV_PARENT_UNITS == ('tag-he-smoke-v3-queue',)
    layout = hin.dev_layout(W.ROOT, segment='iv-b')
    checks = {}
    hu.disk_and_machine(layout, 'out', deps, checks)
    assert seen[-1]['own_units'] == ('tag-hb-pq-out-dev-ivb',)


def test_the_p3_replay_gates_on_its_own_unit(tmp_path, monkeypatch):
    """The replay runs in tag-hb-pq-p3-replay; its gate's own unit was tag-hb-pq-out-dev (an active unit the process is not in counts as another unit: the gate never opened).  Revision 2
    (review minor 8): the replay run for real on the PQ package's world, its gate holdout_common.machine_gate's stand-in recording what it is asked."""
    import pqh_smoke
    import run_pq
    w, layout, _ = W.dev_world(tmp_path, monkeypatch)
    W.XW.sealed_module._reset_cache()
    assert run_pq.out_unit(w.ctx, gate=W.GATE, code=W.CODE, log=W.quiet) == 0
    release = io_.sha_file(w.ctx.pq_root/io_.MANIFEST_FILE)
    ctx_pair = io_.Context(repo=w.repo, source_root=w.repo, results=w.results, images=w.images, shorts=(W.SHORT,), p3=w.ctx.p3, run_json=w.ctx.run_json, v4=w.ctx.v4, anchor=w.ctx.anchor,
                           taper=W.PW.frozen_formula_taper)
    seen = []
    monkeypatch.setattr(W.hc, 'machine_gate', lambda **kw: seen.append(kw) or dict(passed=True, stand_in=True))
    monkeypatch.setattr(W.hc, 'wait_for', lambda check, **kw: check())
    W.XW.sealed_module._reset_cache()
    assert pqh_smoke.p3_replay(layout, deps=W.deps(gate=None), release_sha256=release, contexts_of=lambda layout, taper: (ctx_pair, w.ctx)) == 0
    assert [kw['own_units'] for kw in seen] == [('tag-hb-pq-p3-replay',)] and seen[0]['parent_units'] == ('tag-he-smoke-v3-queue',)


def test_the_rules_of_a_segments_out_unit_are_its_own_directories_and_the_members_run(tmp_path, monkeypatch):
    """Revision 2 (review minor 10): a (iv-b) out unit may read and write its own pq/iv-b/ and pq/big/iv-b/ and read the members run; (ii)'s records under pq/ and the other segment's are
    outside its allowed items.  (ii)'s rules are as before (the whole pq/ root)."""
    ii, ivb = hin.dev_layout(W.ROOT, segment='ii'), hin.dev_layout(W.ROOT, segment='iv-b')
    rules, before = hu.rules_for(ivb, 'out'), hu.rules_for(ii, 'out')
    pq = Path(hp.DEV_PQ_ROOT)
    for p in (pq/'iv-b'/W.PAIR/'pq.json', pq/'big'/'iv-b'/W.PAIR/'pq.npz', pq/'iv-b'/'rerun'/'x.json', pq/'members'/'members_record.json', pq/'big'/'members'/'s'/'var.npz', pq/'batches'/'x'/'y.json'):
        assert rules.allowed(p.resolve()), p
    for p in (pq/W.PAIR/'pq.json', pq/'big'/W.PAIR/'pq.npz', pq/'pq_record.json', pq/'iv-c'/W.PAIR/'pq.json', pq/'big'/'iv-c'/W.PAIR/'pq.npz', pq/'units'/'x'/'unit_end.json'):
        assert not rules.allowed(p.resolve()), p
    assert before.allowed((pq/W.PAIR/'pq.json').resolve()) and before.allowed((pq/'iv-c'/'x.json').resolve())


def test_the_out_unit_refuses_outside_its_own_segments_unit(tmp_path, monkeypatch):
    w, layout, _ = W.dev_world(tmp_path, monkeypatch)
    ivb = replace(layout, out_subdir='iv-b', out_unit='tag-hb-pq-out-dev-ivb', out_lock=Path(hp.DEV_PQ_ROOT)/'pq_out_dev_iv-b.lock')
    assert W.run_out(ivb, deps=W.deps(in_unit=lambda name: name == 'tag-hb-pq-out-dev')) == hu.EXIT['refused']           # (ii)'s unit is not (iv-b)'s
    assert not (Path(hp.DEV_PQ_ROOT)/'pq_out_dev_iv-b.lock').exists()


@pytest.mark.parametrize('kind', ['configC', 'identity'])
def test_the_out_unit_on_a_fallback_copy_writes_under_its_own_segment_and_ii_is_untouched(tmp_path, monkeypatch, kind):
    w, layout, _ = W.dev_world(tmp_path, monkeypatch)
    segment = 'iv-b' if kind == 'configC' else 'iv-c'
    cell = Path(hp.SMOKE_ROOT)/segment/'a'/'0438-0439'
    u = np.zeros(w.grid.shape+(3,)) if kind == 'identity' else np.ascontiguousarray(w.fields[W.SHORT])*.5
    with np.load(w.ctx.t88_dir(W.SHORT)/'field_0600.npz') as z:
        raw = np.array(z['u'])
    run = json.loads((w.ctx.t88_dir(W.SHORT)/'run.json').read_text())
    W.he_pair(cell, W.PAIR, raw=raw, nodes=np.array([1]), values=np.zeros((1, 3)), grid=w.grid, taper=W.PW.frozen_formula_taper, labels_sha256=run['labels_sha256'],
              status='hx_fallback_configC' if kind == 'configC' else 'hx_fallback_identity', fallback_u=u, fallback_kind=kind)
    seg = replace(layout, dev_cell=cell, out_subdir=segment, out_unit=hp.DEV_OUT_UNITS[segment], out_lock=Path(hp.DEV_PQ_ROOT)/f'pq_out_dev_{segment}.lock')
    line_before = io_.sha_file(cell/'output'/'line.json')
    assert W.run_out(seg, deps=W.deps(in_unit=lambda name: name == hp.DEV_OUT_UNITS[segment])) == 0
    small, big = Path(hp.DEV_PQ_ROOT)/segment, Path(hp.DEV_PQ_ROOT)/'big'/segment
    document = io_.read_json(small/W.PAIR/hu.PQ_JSON)
    assert document['line_status'] == ('hx_fallback_configC' if kind == 'configC' else 'hx_fallback_identity') and document['hm']['rebuilt_equals_final'] is None
    assert document['method']['fallback']['kind'] == kind and document['method']['fallback']['state_hash'] == io_.sha_array(u)
    assert (big/W.PAIR/hu.PQ_NPZ).is_file() and (small/hu.OUT_RECORD).is_file() and io_.read_json(small/hu.OUT_RECORD)['complete'] is True
    listed = {line.split('  ', 1)[1] for line in (small/hu.MANIFEST).read_text().splitlines()}
    assert str(small/W.PAIR/hu.PQ_JSON) in listed and str(Path(hp.DEV_PQ_ROOT)/'members'/hin.MEMBERS_RECORD) in listed
    assert not (Path(hp.DEV_PQ_ROOT)/hu.OUT_RECORD).exists() and not (Path(hp.DEV_PQ_ROOT)/W.PAIR).exists()                  # (ii)'s places untouched
    assert io_.sha_file(cell/'output'/'line.json') == line_before
    assert W.run_out(layout, deps=W.deps()) == 0                                                                          # then (ii), its own lock and places
    assert (Path(hp.DEV_PQ_ROOT)/hu.OUT_RECORD).is_file() and (Path(hp.DEV_PQ_ROOT)/'big'/W.PAIR/hu.PQ_NPZ).is_file()
    assert W.run_out(seg, deps=W.deps(in_unit=lambda name: True)) == hu.EXIT['refused']                                  # each once
