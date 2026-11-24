"""C23.6-5 in run_pq (SG, revisions 6e-6g; C22.Q1-3): the sealed lists of C23.2 are allowed items of PQ's spy, file by file, and every read of them is recorded; before
each of the four entries (members_unit, out_unit, dry_members, dry_out) installs its spy, the guard's cache of the lists must be empty (bayesreg.io.sealed.
sealed_cache_is_empty, C23.2-3), else the entry stops with 'sealed list cache already populated' and installs no spy.  Ids only, files in tmp_path."""
import hashlib
from pathlib import Path

import pytest

import bayesreg.io.sealed as sealed

from . import _world as W
from .conftest import CODE, GATE
from ..io_seg._helpers import synthetic_folds
from .test_pq_pipeline import out_record, run_out

io_, pins = W.io_, W.pins

import pq_core as core  # noqa: E402
import pq_firewall as fw  # noqa: E402
import run_pq  # noqa: E402


def synthetic_sealed_lists(repo, monkeypatch):
    """bayesreg's sealed lists pinned (monkeypatch) to synthetic lists at the pinned paths under `repo`, inside PQ's watched scope (ids 0960-0967, on no list of the
    world); the cache reset.  The resolved paths."""
    specs = []
    for k, spec in enumerate(sealed.SEALED_LISTS):
        path = Path(repo)/spec['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = [f'OASIS_OAS1_{960+2*k:04d}_MR1', f'OASIS_OAS1_{961+2*k:04d}_MR1']
        path.write_text('fixed\tmoving\n'+'\t'.join(ids)+'\n' if spec['columns'] else ''.join(f'{s}\n' for s in ids), encoding='utf-8')
        specs.append(dict(spec, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), count=2))
    monkeypatch.setattr(sealed, 'REPO', Path(repo))
    monkeypatch.setattr(sealed, 'SEALED_LISTS', tuple(specs))
    sealed._reset_cache()
    return [(Path(repo)/spec['path']).resolve() for spec in specs]


def populate_cache():
    """The guard's cache filled the way an earlier read in the process would (ids only)."""
    sealed._reset_cache()
    sealed.guard_sealed_lists()
    assert not sealed.sealed_cache_is_empty()


@synthetic_folds('world')
def test_the_out_unit_records_every_read_of_the_sealed_lists_inside_its_spy(world, monkeypatch):
    lists = synthetic_sealed_lists(world.repo, monkeypatch)
    assert {str(p) for p in lists} <= {str(p) for p in fw.Rules(repo=world.repo, results=world.results, data=world.repo/'data', images=world.images, pairs=[], subjects=[]).sealed_lists}
    assert run_out(world) == 0 and out_record(world)['passed'] is True
    opened = io_.read_json(world.ctx.pq_root/'opened_files.json')['out']
    assert opened['m6'] is True and not opened['breaches']
    seen = [f for f in opened['files'] if Path(f['path']) in lists]
    assert {Path(f['path']) for f in seen} == set(lists)                                               # every list read is recorded, none is a breach
    holdout_b = (world.repo/'splits'/'bayesreg_v1'/'holdout_B_subjects.txt').resolve()
    assert all(f['in_scope'] for f in seen if Path(f['path']) == holdout_b)                             # splits/ is watched: judged, allowed
    assert all(not f['in_scope'] for f in seen if Path(f['path']) != holdout_b)                         # archive/ is outside the watched scope (C22.Q1-3): recorded only


@pytest.mark.parametrize('unit', ['out', 'members'])
def test_a_unit_stops_before_its_spy_when_the_cache_is_populated(world, monkeypatch, unit):
    populate_cache()
    monkeypatch.setattr(fw, 'Spy', lambda *a, **k: (_ for _ in ()).throw(AssertionError('a spy was installed')))
    before = sorted(str(p) for p in world.ctx.pq_root.rglob('*'))
    with pytest.raises(core.PQRefused, match='sealed list cache already populated') as info:
        if unit == 'out':
            run_pq.out_unit(world.ctx, gate=GATE, code=CODE, log=lambda *a: None)
        else:
            run_pq.members_unit(world.ctx, device='cpu', allowlist=None, files=None, gate=GATE, code=CODE, log=lambda *a: None)
    assert info.value.stage == 'spy'
    assert sorted(str(p) for p in world.ctx.pq_root.rglob('*')) == before                    # nothing written


@pytest.mark.parametrize('unit', ['out', 'members'])
def test_a_dry_run_reports_a_populated_cache_and_installs_no_spy(world, monkeypatch, unit):
    populate_cache()
    monkeypatch.setattr(fw, 'Spy', lambda *a, **k: (_ for _ in ()).throw(AssertionError('a spy was installed')))
    lines = []
    problems = (run_pq.dry_out if unit == 'out' else run_pq.dry_members)(world.ctx, lines.append)
    assert any(run_pq.SEALED_CACHE_POPULATED in p for p in problems) and any(run_pq.SEALED_CACHE_POPULATED in line for line in lines)


def test_the_check_is_the_read_only_query():
    """run_pq asks bayesreg.io.sealed.sealed_cache_is_empty (C23.2-3) and nothing else: an empty cache is no problem, a populated one is, and asking changes nothing."""
    sealed._reset_cache()
    assert run_pq.sealed_cache_problem('out') is None and sealed.sealed_cache_is_empty()
    populate_cache()
    before = dict(sealed._CACHE)
    assert run_pq.SEALED_CACHE_POPULATED in run_pq.sealed_cache_problem('out') and sealed._CACHE == before
    sealed._reset_cache()
