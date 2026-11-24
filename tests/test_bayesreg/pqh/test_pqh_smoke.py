"""PQH, the smoke's segment (iii) (contract C22.H8 (iii)): the hold-out computation on P3's 0438-0439 against the development reading's arrays, bit for bit, after their files are checked against
the released list -- on the PQ package's synthetic world, whose development unit (run_pq.out_unit) writes the reference (CPU, tmp_path)."""
import numpy as np
import pytest

from . import _world as W

hp, hin, hu, io_ = W.hp, W.hin, W.hu, W.io_
import pqh_smoke  # noqa: E402


@pytest.fixture
def dev(tmp_path, monkeypatch):
    import run_pq
    w, layout, entry = W.dev_world(tmp_path, monkeypatch)
    W.XW.sealed_module._reset_cache()
    assert run_pq.out_unit(w.ctx, gate=W.GATE, code=W.CODE, log=W.quiet) == 0
    release = io_.sha_file(w.ctx.pq_root/io_.MANIFEST_FILE)
    ctx_pair = io_.Context(repo=w.repo, source_root=w.repo, results=w.results, images=w.images, shorts=(W.SHORT,), p3=w.ctx.p3, run_json=w.ctx.run_json, v4=w.ctx.v4, anchor=w.ctx.anchor,
                           taper=W.PW.frozen_formula_taper)
    return W.namespace(w=w, layout=layout, release=release, contexts=lambda layout, taper: (ctx_pair, w.ctx))


def replay(dev, **kwargs):
    W.XW.sealed_module._reset_cache()
    return pqh_smoke.p3_replay(dev.layout, deps=W.deps(), release_sha256=dev.release, contexts_of=dev.contexts, **kwargs)


def test_the_p3_replay_is_bit_for_bit(dev):
    assert replay(dev) == 0
    record = io_.read_json(dev.layout.pq_small/pqh_smoke.P3_REPLAY_DIR/'p3_replay.json')
    assert record['reading'] == pqh_smoke.READ_PASS and len(record['rows']) == 2+len(W.core.BASELINE_KEYS) and all(r['equal'] and r['file_listed'] for r in record['rows'].values())
    assert record['spy_m6'] is True
    assert replay(dev) == hu.EXIT['refused']                                         # once


def test_a_computation_that_differs_reads_not_passed(dev, monkeypatch):
    original = hin.pq_arrays

    def off(**kwargs):
        pq, base, extra = original(**kwargs)
        pq['q'] = np.nextafter(pq['q'], np.float32(2))
        return pq, base, extra
    monkeypatch.setattr(hin, 'pq_arrays', off)
    assert replay(dev) == hu.EXIT['stopped']
    record = io_.read_json(dev.layout.pq_small/pqh_smoke.P3_REPLAY_DIR/'p3_replay.json')
    assert record['reading'] == pqh_smoke.READ_FAIL and record['rows']['pq.npz:q']['equal'] is False and record['rows']['pq.npz:q4']['equal'] is True


def test_a_reference_list_that_is_not_the_released_one_refuses(dev):
    W.XW.sealed_module._reset_cache()
    assert pqh_smoke.p3_replay(dev.layout, deps=W.deps(), release_sha256='0'*64, contexts_of=dev.contexts) == hu.EXIT['refused']
    assert io_.read_json(W.refusals(dev.layout)[-1])['stage'] == 'reference'


def test_the_p3_replay_is_dev_only(dev):
    layout = dev.layout
    layout.mode = 'holdout'
    with pytest.raises(ValueError, match='dev mode'):
        pqh_smoke.p3_replay(layout, deps=W.deps())
