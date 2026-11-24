"""experiments/hb/hb_prerepair.py (A5b item 3 (a); prereg revision 2 item 4; contract revision 9 draft C24.2-3): the repair ablation's pre-repair displacement from the finalized and
moved method cell, its three checks against repair/repair.json's source, the record the evaluation entry reads, the spy, written once.  HB's sandbox (every root in a temporary directory,
systemd-run a shim exiting 99); the method cells are HB's (he_core.write_output on a small grid), finalized by the review queue played by test_hb_queues' FakeSystem and moved by
hb_store.move_pair; the frozen tree's boundary_taper is the replica of tests/test_bayesreg/pq/_world.py (the frozen one is compared with it there)."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesreg.eval import holdout as H

from ..pq import _world as PW
from . import _world as W
from .test_hb_queues import FakeSystem

R, he, hc, hb, review, hm = W.R, W.he, W.hc, W.hb, W.hb_review, W.hb_method
import hb_prerepair as P  # noqa: E402
import hb_store  # noqa: E402


@pytest.fixture
def unsealed(tmp_path, monkeypatch):
    monkeypatch.setattr(R.core, 'frozen_api', lambda: SimpleNamespace(boundary_taper=PW.frozen_formula_taper))     # the tests' jd_curl is the code tree's
    monkeypatch.setattr(hm, 'PREREG_SHA256', hb.sha(hb.REPO/hm.PREREG_REL))
    monkeypatch.setattr(hb, 'code_record', lambda commit=None, run=None: dict(commit='1'*40, head='1'*40, clean=True, pinned_code=dict(stub=True)))
    sim = W.XW.simulated_unseal(tmp_path, monkeypatch)
    W.XW.sealed_module._reset_cache()
    return sim


def pairs(n=1):
    return [hc.pair_name(*r) for r in W.XW.holdout_rows()[:n]]


def overwrite(path, document):
    """In place of the stand-in records W.write_cell leaves (he_core.write_json writes once)."""
    Path(path).write_text(json.dumps(document, indent=1), encoding='utf-8')


def repair_inputs(pair, *, source_changes=None, source=True):
    """t88/run.json and repair/repair.json as HE writes them (run_hx: the field's sha256 and u_sha256, data.manifest_sha256, settings_sha256; repair_main: source from source_checked)."""
    cell = hb.cell(pair)
    field = cell/'t88'/he.FIELD_FILE
    with np.load(field, allow_pickle=False) as z:
        u = np.array(z['u'])
    manifest = Path(hc.STAGE_ROOT)/pair/'manifest.json'
    from bayesreg.io.pair import _read_manifest
    grid, _, _ = _read_manifest(manifest)
    tapered = np.ascontiguousarray(PW.frozen_formula_taper(u, grid, 4)[0])
    overwrite(cell/'t88'/'run.json', dict(schema=he.SCHEMA.format('t88'), pair=pair, status=he.T88_DONE, oracle=False, field=dict(file=he.FIELD_FILE, sha256=hb.sha(field), u_sha256=he.array_sha256(u)),
                                              data=dict(manifest_sha256=hb.sha(manifest)), settings_sha256=hm.SETTINGS_SHA256))
    doc = dict(schema=he.SCHEMA.format('repair'), pair=pair, status='certified', passed=True)
    if source:
        doc['source'] = dict(dict(directory=str(cell/'t88'), file=he.FIELD_FILE, field_sha256=hb.sha(field), u_sha256=he.array_sha256(u), tapered_u_sha256=he.array_sha256(tapered)),
                             **(source_changes or {}))
    overwrite(cell/'repair'/'repair.json', doc)
    return u, tapered, grid


def finalized_and_moved(ps, monkeypatch, *, source_changes=None, source=True, move=True):
    """Step 4 (HB's cells), the repair records, step 5 finalized by the review queue (FakeSystem's 'finalize'), the move to D: (hb_store.move_pair), the input lock in a committed code tree."""
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    out = {}
    for p in ps:
        W.XW.write_stage_pair(hc.STAGE_ROOT, p, seg=False)
        W.write_cell(R.make_holdout_context(p, check_identity=False), status=he.HX)
        out[p] = repair_inputs(p, source_changes=source_changes, source=source)
    assert review.queue(system=FakeSystem([dict(review='finalize') for _ in ps]), echo=lambda m: None) == 0
    if move:
        for p in ps:
            assert hb_store.move_pair(p, disk=lambda: None)['passed']
    return out


def lock_repo(tmp_path, ps, *, wrong=False):
    repo = Path(tmp_path)/'code'
    (repo/'runs'/'holdout_B_v1').mkdir(parents=True)
    W.git(repo, 'init', '-q')
    rows = [dict(kind='method_field', pair=p, path=str(Path(hb_store.where(p))/'t88'/he.FIELD_FILE), sha256='0'*64 if wrong else hb.sha(Path(hb_store.where(p))/'t88'/he.FIELD_FILE))
            for p in ps]
    (repo/P.LOCK_REL).write_text(json.dumps(dict(schema=hb.SCHEMA.format('input_lock'), rows=rows)))
    W.git(repo, 'add', '-A')
    W.git(repo, 'commit', '-q', '-m', 'the input lock')
    return repo


def run(repo, **kwargs):
    W.XW.sealed_module._reset_cache()                                  # a fresh process: the guard's cache empty (C23.6-5)
    return P.run(repo=repo, gate=lambda: dict(passed=True), echo=lambda m: None, **kwargs)


def test_the_constants_are_the_entrys_and_hb_store_s():
    assert (P.RECORD, P.DISP, P.CHECKS, P.SCHEMA) == (H.PREREPAIR_RECORD, H.PREREPAIR_DISP, H.PREREPAIR_CHECKS, H.PREREPAIR_SCHEMA)
    assert P.LOCK_REL == hb_store.DEFAULT_LOCK_REL and P.INPUTS == ('t88/field_0600.npz', 't88/run.json', 'repair/repair.json')
    assert Path(hc.REAL_RESULTS)/'holdout_B_v1'/'prerepair' == Path(H.PREREPAIR_ROOT) and Path(hc.REAL_BIG)/'prerepair' == Path(H.PREREPAIR_BIG)


def test_the_pre_repair_displacement_is_the_repairs_input_through_u_to_d(tmp_path, monkeypatch, unsealed):
    ps = pairs(1)
    (u, tapered, grid), = finalized_and_moved(ps, monkeypatch).values()
    repo = lock_repo(tmp_path, ps)
    assert run(repo) == 0
    record_path, disp_path = P.paths(ps[0])
    record = json.loads(record_path.read_text())
    assert record['passed'] and record['checks'] == {k: True for k in P.CHECKS} and record['schema'] == P.SCHEMA and record['code']['commit'] == '1'*40
    with np.load(disp_path, allow_pickle=False) as z:
        d = np.array(z['d'])
    assert d.shape == (3,)+grid.shape and np.array_equal(d.view(np.uint64), he.u_to_d(tapered, grid).view(np.uint64))       # bit for bit he_core.u_to_d of the taper
    final = json.loads((hb.review_root()/'finalized_outputs.json').read_text())
    pinned = final['pairs'][ps[0]]['part']['t88_repair']
    assert {k: record['inputs'][k]['sha256'] for k in P.INPUTS} == {k: pinned[k] for k in P.INPUTS}
    assert all(Path(record['inputs'][k]['path']).is_relative_to(Path(hc.big_dir('method', ps[0]))) for k in P.INPUTS)     # read on D: through the move record
    assert record['output']['sha256'] == hb.sha(disp_path)
    # the evaluation entry's own check of the record (bayesreg.eval.holdout._ablation_problem) holds on it
    assert H._ablation_problem(record, (0, *hc.split_pair(ps[0]), ps[0]), final['pairs'][ps[0]], disp_path) is None
    assert run(repo) == 0                                              # written once: a second run skips the pair
    runs = sorted((P.record_root()/'runs').glob('*.json'))
    assert json.loads(runs[-1].read_text())['pairs'][ps[0]]['state'] == 'skipped' and json.loads(runs[0].read_text())['spy']['passed']


@pytest.mark.parametrize('check', P.CHECKS)
def test_a_check_that_does_not_hold_writes_a_record_and_no_displacement(tmp_path, monkeypatch, unsealed, check):
    ps = pairs(1)
    finalized_and_moved(ps, monkeypatch, source_changes={check: '0'*64})
    assert run(lock_repo(tmp_path, ps)) == 0
    record_path, disp_path = P.paths(ps[0])
    record = json.loads(record_path.read_text())
    assert record['passed'] is False and record['checks'][check] is False and not disp_path.exists()
    assert H._ablation_problem(record, (0, *hc.split_pair(ps[0]), ps[0]), {}, disp_path) is not None


@pytest.mark.parametrize('how', ['no_source', 'input_lock', 'not_moved'])
def test_no_repair_input_or_another_record_leaves_the_pair_out(tmp_path, monkeypatch, unsealed, how):
    ps = pairs(1)
    finalized_and_moved(ps, monkeypatch, source=how != 'no_source', move=how != 'not_moved')
    if how == 'not_moved':
        repo = Path(tmp_path)/'code'
        (repo/'runs'/'holdout_B_v1').mkdir(parents=True)
        W.git(repo, 'init', '-q')
        (repo/P.LOCK_REL).write_text(json.dumps(dict(schema=hb.SCHEMA.format('input_lock'), rows=[])))
        W.git(repo, 'add', '-A')
        W.git(repo, 'commit', '-q', '-m', 'lock')
    else:
        repo = lock_repo(tmp_path, ps, wrong=how == 'input_lock')
    assert run(repo) == 0
    record_path, disp_path = P.paths(ps[0])
    record = json.loads(record_path.read_text())
    word = dict(no_source='source', input_lock='input lock', not_moved='move record')[how]
    assert record['passed'] is False and any(word in p for p in record['problems']) and not disp_path.exists(), record['problems']


def test_a_displacement_without_its_record_refuses_that_pair(tmp_path, monkeypatch, unsealed):
    ps = pairs(1)
    finalized_and_moved(ps, monkeypatch)
    _, disp_path = P.paths(ps[0])
    disp_path.parent.mkdir(parents=True)
    disp_path.write_bytes(b'half')
    assert run(lock_repo(tmp_path, ps)) == P.EXIT['pair_refused']
    assert not P.paths(ps[0])[0].exists() and disp_path.read_bytes() == b'half'


def test_the_start_refuses_without_rule_one_and_reads_no_pair(tmp_path, monkeypatch, unsealed):
    ps = pairs(1)
    finalized_and_moved(ps, monkeypatch)

    def no(commit=None, run=None):
        raise hb.Stop(hb.EXIT['refused'], 'no runs/holdout_B_v1/pinned_code.json (synthetic)')
    monkeypatch.setattr(hb, 'code_record', no)
    assert run(lock_repo(tmp_path, ps)) == P.EXIT['refused'] and not P.paths(ps[0])[0].exists()


def test_the_spy_refuses_what_the_tool_never_reads(unsealed):
    spy = P.Spy()
    home, big = Path(hc.HOLDOUT_ROOT), Path(hc.BIG_ROOT)
    for bad in (Path('/mnt/d/data')/'OASIS_OAS1_0001_MR1'/'aligned_seg35.nii.gz', home/'stage'/'p'/'pair.npz', home/'eval'/'x.json', Path(hc.RESULTS)/'ev'/'truth_gate.json',
                home/'configc'/'p'/'result.json', home/'work'/'configc'/'p'/'x', big/'ants_s'/'p'/'disp.npz', big/'pq'/'p'/'pq.npz', home/'pq'/'p'/'pq.json'):
        assert spy.why(bad), bad
    for good in (home/'method'/'p'/'move.json', big/'method'/'p'/'t88'/'field_0600.npz', home/'review'/'finalized_outputs.json', home/'stage'/'p'/'manifest.json',
                 home/'prerepair'/'p'/'prerepair.json', big/'prerepair'/'p'/'prerepair_disp.npz', home/'locks'/'hb__method__p.lock'):
        assert spy.why(good) is None, good
    W.XW.sealed_module.guard_sealed_lists()                            # the cache filled: the spy cannot be installed (C23.6-5)
    with pytest.raises(hc.Refused, match='cached'):
        P.install_spy()
    W.XW.sealed_module._reset_cache()
    spy = P.install_spy()
    try:
        with pytest.raises(P.Breach):
            open(Path(hc.HOLDOUT_ROOT)/'eval'/'anything.json', 'rb')
        assert spy.breaches and spy.record()['passed'] is False
    finally:
        P.remove_spy()


# ---- A5b review minors 3-5 ------------------------------------------------------------------------------------------------------------------------------------------
def last_run_record():
    return json.loads(sorted((P.record_root()/'runs').glob('*.json'))[-1].read_text())


def test_an_input_missing_on_d_writes_no_record_and_stops_as_infrastructure(tmp_path, monkeypatch, unsealed):
    """Minor 5: a file a move row names on D: that is not there (D: not mounted, a file gone) is not the pair's: no record, the run stops with exit 20; with the file back the pair is prepared."""
    ps = pairs(1)
    finalized_and_moved(ps, monkeypatch)
    repo = lock_repo(tmp_path, ps)
    field = Path(hc.big_dir('method', ps[0]))/'t88'/he.FIELD_FILE
    kept = field.with_name('kept.npz')
    field.rename(kept)
    assert run(repo) == P.EXIT['infra'] and not P.paths(ps[0])[0].exists()
    assert 'not there' in last_run_record()['infrastructure'] and last_run_record()['exit'] == P.EXIT['infra']
    kept.rename(field)
    assert run(repo) == 0 and json.loads(P.paths(ps[0])[0].read_text())['passed']


def test_an_unexpected_exception_is_recorded_and_never_exit_0(tmp_path, monkeypatch, unsealed):
    """Minor 3: an exception other than the refusals (here np.load of a field that is not an npz) is recorded in the run record with its traceback and exit 23."""
    ps = pairs(1)
    finalized_and_moved(ps, monkeypatch)
    repo = lock_repo(tmp_path, ps)
    real = np.load
    monkeypatch.setattr(P.np, 'load', lambda *a, **k: (_ for _ in ()).throw(OSError('synthetic: not an npz')) if str(a[0]).endswith(he.FIELD_FILE) else real(*a, **k))
    assert run(repo) == P.EXIT['error']
    out = last_run_record()
    assert out['exit'] == P.EXIT['error'] and out['error']['type'] == 'OSError' and 'synthetic' in out['error']['traceback']


def test_the_frozen_api_is_checked_at_the_start_before_any_pair(tmp_path, monkeypatch, unsealed):
    """Minor 3: mc0_core.frozen_api exits when jd_curl is not the frozen tree's; the start records it as a problem (refused, 22) and no pair is read."""
    ps = pairs(1)
    finalized_and_moved(ps, monkeypatch)

    def no_frozen():
        raise SystemExit('jd_curl is not imported from the frozen tree (synthetic)')
    monkeypatch.setattr(R.core, 'frozen_api', no_frozen)
    assert run(lock_repo(tmp_path, ps)) == P.EXIT['refused'] and not P.paths(ps[0])[0].exists()
    assert 'frozen API' in last_run_record()['refused'] and 'SystemExit' in last_run_record()['start']['frozen_api']


def test_pairs_not_on_the_list_are_refused_before_any_record(tmp_path, monkeypatch, unsealed):
    """Minor 4: --pairs must be rows of the pair list; a misspelled name leaves no record under prerepair/<name>/."""
    ps = pairs(1)
    finalized_and_moved(ps, monkeypatch)
    wrong = ps[0].replace('__', '__X')
    assert run(lock_repo(tmp_path, ps), pairs=[wrong]) == P.EXIT['refused']
    assert not (P.record_root()/wrong).exists() and 'not rows of the pair list' in last_run_record()['refused']


def test_the_launch_string_names_the_unit_the_frozen_tree_and_the_cpu():
    argv = P.launch_string(['--commit', '1'*40], repo=Path('/somewhere/holdout_B_v1/code'))
    text = ' '.join(argv)
    assert argv[:3] == ['systemd-run', '--user', f'--unit={P.UNIT}'] and f'PYTHONPATH={he.core.FROZEN}:/somewhere/holdout_B_v1/code' in text
    assert '--setenv=CUDA_VISIBLE_DEVICES=' in argv and argv[-4:] == ['experiments/hb/hb_prerepair.py', 'run', '--commit', '1'*40]
    with pytest.raises(ValueError):
        P.launch_string([], repo=Path('/x/pytest-of-y'))
