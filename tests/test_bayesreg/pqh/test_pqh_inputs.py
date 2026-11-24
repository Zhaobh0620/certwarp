"""PQH's inputs (contract C22.H4-1, -2; C22.H8 实现门槛 绑定): the finalization record, the move record, Hm1 for the three line statuses, Hm2/Hm3's reader against the development reader,
HI's runs taken together, the members' records taken together, and the computation against the development unit's order (CPU, tmp_path, synthetic arrays)."""
import json
from pathlib import Path

import numpy as np
import pytest

from . import _world as W

hp, hin, hu, io_, core, PW = W.hp, W.hin, W.hu, W.io_, W.core, W.PW
SHAPE = (8, 9, 7)


@pytest.fixture
def grid():
    from jd_curl.three_d.grid import make_grid_3d
    return make_grid_3d(nx=SHAPE[2], ny=SHAPE[1], nz=SHAPE[0], hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')


def raw_field(seed=0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.uniform(-1, 1, SHAPE+(3,)), dtype=np.float64)


def layout_of(tmp_path, pairs=(W.PAIR,)):
    """A hold-out-shaped layout in tmp_path (the method's records under method/, HB's finalization record under review/)."""
    return hin.Layout(mode='holdout', code=W.ROOT, pairs=pairs, stage=tmp_path/'stage', pq_small=tmp_path/'pq', pq_big=tmp_path/'pq'/'big', method_records=tmp_path/'method',
                      finalized_path=tmp_path/'review'/'finalized_outputs.json', images=tmp_path/'img', models_root=tmp_path/'models', members_lock=tmp_path/'m.lock', out_lock=tmp_path/'o.lock',
                      batch_root=tmp_path/'pq'/'batches')


def entry_of(name, row):
    problems = []
    entry = hin.normalized_entry(name, row, problems)
    assert not problems, problems
    return entry


def certified_pair(tmp_path, grid, *, nodes=None, **kwargs):
    kwargs.setdefault('labels_sha256', dict(fixed='1'*64, moving='2'*64))          # I3: an hx_certified pair's run.json has them (run_hx); a test that leaves them out says so
    raw = raw_field()
    nodes = np.array([10, 200, 333]) if nodes is None else nodes
    tapered, _ = PW.frozen_formula_taper(raw, grid)
    values = np.asarray(tapered).reshape(-1, 3)[nodes]*.5
    layout = layout_of(tmp_path)
    d, v, row = W.he_pair(tmp_path/'work'/'method'/W.PAIR, W.PAIR, raw=raw, nodes=nodes, values=values, grid=grid, taper=PW.frozen_formula_taper, **kwargs)
    return layout, d, v, entry_of(W.PAIR, row)


def bind(layout, entry, grid):
    paths, _ = hin.method_paths(layout, W.PAIR, entry)
    return hin.bind_output(layout, W.PAIR, entry, paths, grid, PW.frozen_formula_taper)


def test_hx_certified_rebuilds_v_bit_for_bit(tmp_path, grid):
    layout, _, v, entry = certified_pair(tmp_path, grid, labels_sha256=dict(fixed='1'*64, moving='2'*64))
    assert set(entry['files']) == set(hp.HE_FILES.values())
    got, record = bind(layout, entry, grid)
    assert got.tobytes() == v.tobytes() and record['rebuilt_equals_final'] is True and record['state_hash'] == entry['state_hash']
    assert record['labels_sha256'] == dict(fixed='1'*64, moving='2'*64) and record['rebuilt']['already_certified'] is False and record['review']['consistent'] is True


def test_already_certified_has_no_nodes_and_rebuilds(tmp_path, grid):
    layout, _, v, entry = certified_pair(tmp_path, grid, nodes=np.array([], dtype=np.int64), labels_sha256=dict(fixed='1'*64, moving='2'*64))
    got, record = bind(layout, entry, grid)
    assert got.tobytes() == v.tobytes() and record['rebuilt']['already_certified'] is True and record['labels_sha256'] == dict(fixed='1'*64, moving='2'*64)


@pytest.mark.parametrize('labels', [None, dict(fixed='1'*64), dict(fixed='1'*64, moving='short'), 'not a dict'])
def test_hx_certified_without_the_labels_sha256_fails_hm3(tmp_path, grid, labels):
    """Review I3: run_hx writes labels_sha256 of both sides whenever T88 is done, so an hx_certified pair whose t88/run.json lacks them fails Hm3 (C22.H4-2: "照实记" is for a T88 that did not read its
    channels, i.e. a fallback pair; test_a_fallback_pair_without_labels_records_why below)."""
    layout, _, _, entry = certified_pair(tmp_path, grid, labels_sha256=labels)
    with pytest.raises(core.PQRefused, match='hm3.*labels_sha256'):
        bind(layout, entry, grid)


def fallback_pair(tmp_path, grid, kind, u=None):
    status = 'hx_fallback_configC' if kind == 'configC' else 'hx_fallback_identity'
    tapered, _ = PW.frozen_formula_taper(raw_field(5), grid)
    u = (np.zeros(SHAPE+(3,)) if kind == 'identity' else np.ascontiguousarray(tapered)*.3) if u is None else u
    layout = layout_of(tmp_path)
    d, _, row = W.he_pair(tmp_path/'work'/'method'/W.PAIR, W.PAIR, raw=raw_field(), nodes=np.array([1]), values=np.zeros((1, 3)), grid=grid, taper=PW.frozen_formula_taper, status=status,
                          fallback_u=u, fallback_kind=kind)
    return layout, d, u, row


@pytest.mark.parametrize('kind', ['configC', 'identity'])
def test_a_fallback_pair_is_bound_to_its_fallback_records_recertification(tmp_path, grid, kind):
    layout, d, u, row = fallback_pair(tmp_path, grid, kind)
    entry = entry_of(W.PAIR, row)
    got, record = bind(layout, entry, grid)
    assert got.tobytes() == u.tobytes() and record['fallback']['kind'] == kind and record['rebuilt_equals_final'] is None
    line = json.loads((d/'output'/'line.json').read_text())                     # the fallback's certificate in line.json of another array, the records rewritten to agree
    line['fallback']['certificate']['state_hash'] = '0'*64
    (d/'output'/'line.json').write_text(json.dumps(line))
    W.write_records(d, W.PAIR, status=row['line_status'], state_hash=entry['state_hash'], line_fallback=line['fallback'])
    with pytest.raises(core.PQRefused, match='hm1.*re-certification'):
        bind(layout, entry_of(W.PAIR, W.finalized_row(d)), grid)


def test_a_fallback_row_must_name_the_kind_and_a_passing_recertification(tmp_path, grid):
    _, d, _, row = fallback_pair(tmp_path, grid, 'configC')
    for change in (dict(kind='identity'), dict(certificate_passed=False), dict(state_hash='0'*64)):
        problems = []
        assert hin.normalized_entry(W.PAIR, dict(row, part=dict(row['part'], fallback=dict(row['part']['fallback'], **change))), problems) is None and problems


def test_an_identity_fallback_whose_u_is_not_zero_is_refused(tmp_path, grid):
    tapered, _ = PW.frozen_formula_taper(raw_field(5), grid)
    layout, _, _, row = fallback_pair(tmp_path, grid, 'identity', u=np.ascontiguousarray(tapered))
    with pytest.raises(core.PQRefused, match='identity fallback whose u is not 0'):
        bind(layout, entry_of(W.PAIR, row), grid)


def test_the_review_must_be_consistent_or_a_passing_third_certificate_of_this_array(tmp_path, grid):
    layout, d, _, entry = certified_pair(tmp_path, grid)
    W.write_records(d, W.PAIR, status='hx_certified', state_hash=entry['state_hash'], review=dict(consistent=False, third_certificate=dict(passed=True, output_state_hash=entry['state_hash'])))
    bind(layout, entry_of(W.PAIR, W.finalized_row(d)), grid)
    W.write_records(d, W.PAIR, status='hx_certified', state_hash=entry['state_hash'], review=dict(consistent=False, third_certificate=dict(passed=True, output_state_hash='0'*64)))
    with pytest.raises(core.PQRefused, match='review is not consistent'):
        bind(layout, entry_of(W.PAIR, W.finalized_row(d)), grid)
    with pytest.raises(core.PQRefused, match='review/review.json: sha256'):
        bind(layout, entry, grid)                                                  # the record's sha256 of review.json is the first one's


def test_terminal_json_must_be_this_output_and_this_review(tmp_path, grid):
    layout, d, _, entry = certified_pair(tmp_path, grid)
    terminal = json.loads((d/'terminal.json').read_text())
    terminal['line_status'] = 'hx_fallback_identity'
    (d/'terminal.json').write_text(json.dumps(terminal))
    entry['files']['terminal.json'] = io_.sha_file(d/'terminal.json')
    with pytest.raises(core.PQRefused, match='terminal.json line_status'):
        bind(layout, entry, grid)


def test_source_directory_is_compared_by_its_last_part(tmp_path, grid):
    """C22.H1-3 revision 7b n10: a pair copied or moved keeps its records' absolute paths; the binding is by sha256, the directory only by its name t88."""
    layout, d, _, entry = certified_pair(tmp_path, grid)
    moved = tmp_path/'elsewhere'/W.PAIR
    moved.parent.mkdir()
    d.rename(moved)
    entry['directory'] = str(moved)
    bind(layout, entry, grid)


def test_a_cell_moved_whole_is_found_through_hb_stores_move_record(tmp_path, grid):
    """hb_store.move_pair: the whole cell to D: with holdout_common.move_files, its record holdout_B_v1/method/<pair>/move.json (the alignment, item 1)."""
    layout, d, v, entry = certified_pair(tmp_path, grid)
    dst = tmp_path/'D'/'method'/W.PAIR
    record = layout.method_records/W.PAIR/hp.MOVE_RECORD_NAME
    W.hc.move_files(d, dst, record)
    assert not (d/'output'/'final.npz').exists() and hp.MOVE_RECORD_NAME == 'move.json'
    paths, moved = hin.method_paths(layout, W.PAIR, entry)
    assert paths['t88/field_0600.npz'] == dst/'t88'/'field_0600.npz' and sorted(moved['moved']) == sorted(entry['files'])
    got, _ = hin.bind_output(layout, W.PAIR, entry, paths, grid, PW.frozen_formula_taper)
    assert got.tobytes() == v.tobytes()
    document = json.loads(record.read_text())
    record.unlink()
    W.write_json(record, dict(document, passed=False))
    with pytest.raises(core.PQRefused, match='move did not pass'):
        hin.method_paths(layout, W.PAIR, entry)


def half_move(layout, d, dst, monkeypatch, *, fail_at=4):
    """holdout_common.move_files stopped half way (an OSError at the fail_at-th copy, as D: full): move.json with passed false, some rows ok, the rest still in the cell."""
    import shutil
    original, calls = shutil.copyfile, []

    def failing(src, target, *args, **kwargs):
        calls.append(src)
        if len(calls) == fail_at:
            raise OSError('D: full (test)')
        return original(src, target, *args, **kwargs)
    monkeypatch.setattr(W.hc.shutil, 'copyfile', failing)
    with pytest.raises(W.hc.InfrastructureFailure):
        W.hc.move_files(d, dst, layout.method_records/W.PAIR/hp.MOVE_RECORD_NAME)
    monkeypatch.setattr(W.hc.shutil, 'copyfile', original)
    first = json.loads((layout.method_records/W.PAIR/hp.MOVE_RECORD_NAME).read_text())
    assert first['passed'] is False and 0 < sum(r['ok'] for r in first['files']) < len(first['files'])
    return sorted(str(q.relative_to(d)) for q in d.rglob('*') if q.is_file())


def test_a_half_move_moved_on_is_found_through_move_resume(tmp_path, grid, monkeypatch):
    """Review NI1 (HB revision 2, I3): move.json passed false and move_resume.json passed (hb_store.move_resume) -> the files are found where the two records put them, each in exactly one ok row,
    bound by the finalization record's sha256; both records are in the moves record.  Without a passed move_resume.json the pair stops at hm1 (hb_store.where refuses it too)."""
    layout, d, v, entry = certified_pair(tmp_path, grid)
    dst = tmp_path/'D'/'method'/W.PAIR
    remaining = half_move(layout, d, dst, monkeypatch)
    with pytest.raises(core.PQRefused, match='hm1.*no move_resume.json: half moved'):
        hin.method_paths(layout, W.PAIR, entry)
    resume = layout.method_records/W.PAIR/hp.MOVE_RESUME_NAME
    W.hc.move_files(d, dst, resume, names=remaining)
    paths, moved = hin.method_paths(layout, W.PAIR, entry)
    assert all(q.parent == dst or dst in q.parents for q in paths.values()) and sorted(moved['moved']) == sorted(entry['files'])
    assert moved['move_record']['passed'] is False and moved['move_record']['move_resume'] == dict(path=str(resume), sha256=io_.sha_file(resume), passed=True)
    got, _ = hin.bind_output(layout, W.PAIR, entry, paths, grid, PW.frozen_formula_taper)
    assert got.tobytes() == v.tobytes()
    document = json.loads(resume.read_text())
    resume.unlink()
    W.write_json(resume, dict(document, passed=False))
    with pytest.raises(core.PQRefused, match='did not pass either'):
        hin.method_paths(layout, W.PAIR, entry)


def test_a_file_in_two_ok_rows_of_the_move_records_is_refused(tmp_path, grid, monkeypatch):
    layout, d, v, entry = certified_pair(tmp_path, grid)
    dst = tmp_path/'D'/'method'/W.PAIR
    remaining = half_move(layout, d, dst, monkeypatch)
    W.hc.move_files(d, dst, layout.method_records/W.PAIR/hp.MOVE_RESUME_NAME, names=remaining)
    first = json.loads((layout.method_records/W.PAIR/hp.MOVE_RECORD_NAME).read_text())
    resume = layout.method_records/W.PAIR/hp.MOVE_RESUME_NAME
    document = json.loads(resume.read_text())
    document['files'].append(next(r for r in first['files'] if r['ok']))                   # the same file claimed by both records
    resume.unlink()
    W.write_json(resume, document)
    with pytest.raises(core.PQRefused, match='exactly one passed row'):
        hin.method_paths(layout, W.PAIR, entry)


def test_the_anchor_fixture_has_run_hxs_shape():
    """Review NB1: he_core.segmenter_start_problems (what run_hx.py:305 records as segmenter_anchor.start) on the real anchor registry gives the fixture's keys and nesting -- segmenter_id and
    policy inside 'anchor' -- and passes PQH's check; the check reads that shape (the registry file is configuration, read here; no run directory is read)."""
    import he_core
    problems, record = he_core.segmenter_start_problems(dict(segmenter=dict(he_core.CONTRACT_SEGMENTER)), registry=W.ROOT/he_core.CONTRACT_SEGMENTER['registry'], code_root=W.ROOT)
    assert problems == []
    fixture = W.anchor_start()
    assert sorted(record) == sorted(fixture) == sorted(hin.ANCHOR_START_KEYS) and 'segmenter_id' not in record and 'policy' not in record
    assert record['anchor'] == fixture['anchor'] == dict(hp.ANCHOR) and record['settings_segmenter'] == dict(hp.ANCHOR) and record['registry_rel'] == fixture['registry_rel']
    assert set(record['model_segmenter_ids'].values()) == {hp.ANCHOR['segmenter_id']} and record['rule'] == fixture['rule']
    run = dict(segmenter_anchor=json.loads(json.dumps(dict(start=record, start_problems=problems, after_channels_problems=[]))))
    assert hin.anchor_problems(run) == [] and hin.anchor_problems(dict(segmenter_anchor=dict(start=fixture, start_problems=[], after_channels_problems=[]))) == []


def test_run_hx_writes_the_anchor_record_pqh_reads():
    """Review 2 n2: the outer layer too -- run_hx.py's segmenter_anchor=dict(...) has exactly the keywords start, start_problems, after_channels_problems, start and start_problems the two names
    of one `<problems>, <record> = he.segmenter_start_problems(...)` (read with ast; nothing is run)."""
    import ast
    tree = ast.parse((W.ROOT/'experiments'/'he'/'run_hx.py').read_text(encoding='utf-8'))
    values = [kw.value for node in ast.walk(tree) if isinstance(node, ast.Call) for kw in node.keywords if kw.arg == 'segmenter_anchor']
    assert values
    binds = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func).endswith('segmenter_start_problems'):
            names = [e.id for e in node.targets[0].elts if isinstance(e, ast.Name)]
            binds[names[1]] = names[0]
    for value in values:
        assert isinstance(value, ast.Call) and ast.unparse(value.func) == 'dict' and sorted(k.arg for k in value.keywords) == ['after_channels_problems', 'start', 'start_problems']
        kw = {k.arg: k.value for k in value.keywords}
        assert isinstance(kw['start'], ast.Name) and isinstance(kw['start_problems'], ast.Name) and binds.get(kw['start'].id) == kw['start_problems'].id


@pytest.mark.parametrize('how', ['the anchor at the top (the old fixture)', 'no anchor', 'another policy', 'start problems', 'after-channels problems', 'another registry sha256',
                                 'a model of another segmenter', 'no models', 'no record'])
def test_the_anchor_check_refuses(how):
    start = W.anchor_start()
    record = dict(start=start, start_problems=[], after_channels_problems=[])
    if how == 'the anchor at the top (the old fixture)':
        record['start'] = dict(hp.ANCHOR)
    elif how == 'no anchor':
        del start['anchor']
    elif how == 'another policy':
        start['anchor'] = dict(start['anchor'], policy='fold_a_only')
    elif how == 'start problems':
        record['start_problems'] = ['x']
    elif how == 'after-channels problems':
        record['after_channels_problems'] = ['x']
    elif how == 'another registry sha256':
        start['registry_sha256'] = '0'*64
    elif how == 'a model of another segmenter':
        start['model_segmenter_ids'] = dict(start['model_segmenter_ids'], other='unet35_v1')
    elif how == 'no models':
        start['model_segmenter_ids'] = {}
    run = dict(segmenter_anchor=None if how == 'no record' else record)
    assert hin.anchor_problems(run), how


def test_an_hx_certified_pair_whose_run_json_has_the_old_anchor_shape_fails_hm1(tmp_path, grid):
    layout, d, _, entry = certified_pair(tmp_path, grid)
    run_path = d/'t88'/'run.json'
    run = json.loads(run_path.read_text())
    run['segmenter_anchor']['start'] = dict(hp.ANCHOR)
    run_path.write_text(json.dumps(run))
    W.write_records(d, W.PAIR, status='hx_certified', state_hash=entry['state_hash'])            # HB's records over the changed run.json, so that only the anchor differs
    row = W.finalized_row(d)
    with pytest.raises(core.PQRefused, match="hm1.*segmenter_anchor"):
        bind(layout, entry_of(W.PAIR, row), grid)


def test_the_finalization_record_must_cover_the_pairs_and_name_the_files(tmp_path, grid):
    layout, d, _, _ = certified_pair(tmp_path, grid)
    row = W.finalized_row(d)
    with pytest.raises(core.PQRefused, match='order.*no finalization record'):
        hin.read_finalized(layout)
    W.finalized(layout, {W.PAIR: row, 'OASIS_OAS1_0439_MR1__OASIS_OAS1_0440_MR1': row})
    with pytest.raises(core.PQRefused, match='does not cover exactly'):
        hin.read_finalized(layout)
    W.finalized(layout, {W.PAIR: dict(row, part=dict(row['part'], t88_repair=dict(row['part']['t88_repair'], **{'repair/repair.npz': None})))})
    with pytest.raises(core.PQRefused, match='D113'):
        hin.read_finalized(layout)
    W.finalized(layout, {W.PAIR: dict(row, line_status='hx_somewhere')})
    with pytest.raises(core.PQRefused, match='line_status'):
        hin.read_finalized(layout)
    W.finalized(layout, {W.PAIR: dict(row, review=None)})
    with pytest.raises(core.PQRefused, match='without the review record'):
        hin.read_finalized(layout)
    W.finalized(layout, {W.PAIR: dict(row, terminal=None)})
    with pytest.raises(core.PQRefused, match='not written'):
        hin.read_finalized(layout)
    W.finalized(layout, {W.PAIR: W.no_output_row('hx_implementation_error')})
    document, sha = hin.read_finalized(layout)
    assert document['pairs'][W.PAIR]['line_status'] == 'hx_implementation_error' and sha == io_.sha_file(layout.finalized) and document['code']['commit'] == 'c'*40
    W.finalized(layout, {W.PAIR: row})
    document, _ = hin.read_finalized(layout)
    assert document['pairs'][W.PAIR]['files'] == entry_of(W.PAIR, row)['files'] and layout.finalized == tmp_path/'review'/'finalized_outputs.json'


def test_the_dev_mode_reads_the_copys_own_records(tmp_path, grid):
    """HB's dev review writes review/review.json and terminal.json in the copy and no finalized_outputs.json: they stand for the one pair's row."""
    d, _, row = W.he_pair(tmp_path/'he_smoke_v3'/'ii'/'a'/'0438-0439', W.PAIR, raw=raw_field(), nodes=np.array([3]), values=np.zeros((1, 3)), grid=grid, taper=PW.frozen_formula_taper)
    layout = hin.Layout(mode='dev', code=W.ROOT, pairs=(W.PAIR,), stage=tmp_path, pq_small=tmp_path/'pq', pq_big=tmp_path/'pq'/'big', method_records=None, dev_cell=d, images=tmp_path,
                        models_root=tmp_path, members_lock=tmp_path/'m.lock', out_lock=tmp_path/'o.lock')
    document, sha = hin.read_finalized(layout)
    assert sha is None and document['pairs'][W.PAIR]['files'] == entry_of(W.PAIR, row)['files'] and layout.finalized == d/'review'/'review.json'
    (d/'review'/'review.json').unlink()
    with pytest.raises(core.PQRefused, match='order'):
        hin.read_finalized(layout)


def test_the_reader_is_the_development_readers_step_for_step(tmp_path, monkeypatch):
    """Hm2, Hm3: pqh_inputs.read_side gives pq_io.read_side's P, L and record (with labels); without labels the second half of Hm3 is not checked and recorded so."""
    w, layout, entry = W.dev_world(tmp_path, monkeypatch)
    from bayesreg.io.guard import load_dev_allowlist
    allowlist = load_dev_allowlist(w.repo/pins_allowlist())
    pin = hin.load_pair_checked(layout, W.PAIR, allowlist)
    run = json.loads((w.ctx.t88_dir(W.SHORT)/'run.json').read_text())
    layer_dev = io_.structure_layer(w.ctx, pin)
    layer = hin.structure_layer(layout.code, pin, layout.dev_pred_root, allowlist)
    for side in ('fixed', 'moving'):
        P0, L0, r0 = io_.read_side(w.ctx, layer_dev, pin, side, run['labels_sha256'])
        P1, L1, r1 = hin.read_side(layer, pin, side, want=layout.v4[pin.fixed_subject if side == 'fixed' else pin.moving_subject]._asdict(), labels_sha256=run['labels_sha256'])
        assert P0.tobytes() == P1.tobytes() and L0.tobytes() == L1.tobytes()
        assert {k: r0[k] for k in r0} == {k: r1[k] for k in r0}
        _, _, r2 = hin.read_side(layer, pin, side, want={}, labels_sha256=None)
        assert r2['labels_checked_against_run_json'] is False
        with pytest.raises(core.PQRefused, match='hm3'):
            hin.read_side(layer, pin, side, want={}, labels_sha256=dict(fixed='0'*64, moving='0'*64))
        with pytest.raises(core.PQRefused, match='hm2'):
            hin.read_side(layer, pin, side, want=dict(probs36='0'*64), labels_sha256=None)


def pins_allowlist():
    return W.pins.ALLOWLIST_REL


def test_the_computation_is_the_development_units(tmp_path, monkeypatch):
    """算术不变: pq_arrays with the development readers gives pq_io.compute_pair's arrays bit for bit (the same functions in the same order)."""
    w, layout, entry = W.dev_world(tmp_path, monkeypatch)
    from bayesreg.io.guard import load_dev_allowlist
    allowlist = load_dev_allowlist(w.repo/pins_allowlist())
    lookup = io_.load_lookup(w.repo)
    out = tmp_path/'dev_out'
    record = io_.members_record(w.ctx)
    io_.compute_pair(w.ctx, W.SHORT, allowlist, lookup, record, out)
    pin = io_.load_pair_checked(w.ctx, W.SHORT, allowlist)
    v, field_record = io_.bind_field(w.ctx, W.SHORT, pin.grid)
    layer = io_.structure_layer(w.ctx, pin)
    V_F, V_M = (io_.read_variance(w.ctx, record, s, tuple(pin.grid.shape)) for s in (pin.fixed_subject, pin.moving_subject))
    labels = field_record['labels_sha256']
    pq, base, extra = hin.pq_arrays(v=v, pin=pin, read_fixed=lambda: io_.read_side(w.ctx, layer, pin, 'fixed', labels), read_moving=lambda: io_.read_side(w.ctx, layer, pin, 'moving', labels),
                                    V_F=V_F, V_M=V_M, lookup=lookup)
    with np.load(out/io_.PQ_NPZ) as z:
        assert all(np.array(z[k]).tobytes() == pq[k].tobytes() for k in ('q', 'q4'))
    with np.load(out/io_.BASELINES_FILE) as z:
        assert all(np.array(z[k]).tobytes() == base[k].tobytes() and np.array(z[k]).dtype == base[k].dtype for k in core.BASELINE_KEYS)
    document = io_.read_json(out/io_.PQ_JSON)
    assert same_numbers(extra['numbers'], document['numbers'])
    assert extra['d_sha256'] == document['sampling']['d_sha256']


def same_numbers(a, b):
    """m1: the numbers of pq.json equal item by item, NaN equal to NaN (JSON writes NaN as null: pq_io.no_nan)."""
    a, b = json.loads(json.dumps(a, default=float)), json.loads(json.dumps(b, default=float))
    if isinstance(a, dict) and isinstance(b, dict):
        return sorted(a) == sorted(b) and all(same_numbers(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_numbers(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    if (a is None and isinstance(b, float) and np.isnan(b)) or (b is None and isinstance(a, float) and np.isnan(a)):
        return True
    return a == b


def test_same_numbers_is_strict_but_takes_nan_as_equal():
    assert same_numbers(dict(S35_hat=float('nan'), x=[1., 2.]), dict(S35_hat=None, x=[1., 2.]))
    assert not same_numbers(dict(S35_hat=.5, x=[1., 2.]), dict(S35_hat=.5, x=[1., 3.]))          # the old assertion passed this on S35_hat alone
    assert not same_numbers(dict(S35_hat=.5), dict(S35_hat=.5, extra=1.))


# ---- HI's runs (Hm2) ---------------------------------------------------------------------------------------------------------------------------------
hi_run = W.hi_run


def test_hi_runs_are_taken_together(tmp_path):
    lock = tmp_path/'seg'/'hi_holdout_infer.lock'
    a, b = ['OASIS_OAS1_0001_MR1', 'OASIS_OAS1_0002_MR1'], ['OASIS_OAS1_0003_MR1']
    first, second = tmp_path/'hi1', tmp_path/'hi2'
    hi_run(lock, first, a, record=False)
    W.write_json(first/'hi_stopped.json', dict(done=a))
    hi_run(lock, second, b, resume_of=first)
    rows = hin.hi_predictions(lock, a+b)
    assert rows[a[0]]['pred_dir'] == first/'pred'/a[0] and rows[b[0]]['pred_dir'] == second/'pred'/b[0] and rows[b[0]]['output_sha256']['probs36.npz'] == 'a'*64
    files = {str(p) for p in hin.hi_files(lock, a+b)}
    assert str(second/'pred'/b[0]/'probs36.npz') in files and str(first/'hi_stopped.json') in files
    with pytest.raises(core.PQRefused, match='order.*not complete'):
        hin.hi_predictions(lock, a+b+['OASIS_OAS1_0004_MR1'])
    (first/'subjects'/f'{a[1]}.json').unlink()                                      # a deleted record shows (hi_core.run_consistency_problems)
    with pytest.raises(core.PQRefused, match='hm2'):
        hin.hi_predictions(lock, a+b)
    with pytest.raises(core.PQRefused, match='has not run'):
        hin.hi_predictions(tmp_path/'none.lock', a)


def hi_resumed_root(lock, root, first, rest, residue):
    """A5a (review I2): HI's first run killed in `root` after `first` (hi_started.json, hi_stopped.json; `residue` predicted without a record and moved to attempt1/pred/<s>/ before the resume), then
    the one resume into the same root writing `rest` (hi_started_resume_1.json, hi_record_resume_1.json, its lock line in the resume lock)."""
    import hi_core
    hi_run(lock, root, first, record=False)
    W.write_json(root/'hi_started.json', dict(out=str(root)))
    W.write_json(root/'hi_stopped.json', dict(done=first))
    for s in residue:
        d = root/'attempt1'/'pred'/s
        d.mkdir(parents=True)
        (d/'probs36.npz').write_bytes(b'half')
    W.write_json(root/'attempt1'/'residue.json', dict(subjects=residue))
    names = hi_core.run_names(1)
    line = dict(out=str(root), utc='2026-11-09T01:00:00Z', pid=2, event='authorized_resume', resume=dict(previous_out=str(root)))
    hi_core.resume_lock_path(lock, root).write_text(json.dumps(line)+'\n')
    for s in rest:
        d = root/'pred'/s
        d.mkdir(parents=True)
        (d/'provenance.json').write_text('{}')
        W.write_json(root/'subjects'/f'{s}.json', dict(subject=s, output_sha256={'probs36.npz': 'c'*64, 'structure_groups.npz': 'd'*64}, run=1))
    W.write_json(root/names['started'], dict(out=str(root)))
    W.write_json(root/names['record'], dict(rows={s: {} for s in rest}))
    return names


def test_hi_resumed_into_its_one_root_is_read_as_the_union_and_the_spy_lets_the_resume_records_through(tmp_path):
    """A5b item 13 (A5a implementation notes 6.2, PQH's row): HI's first run and its resume share one root; hi_files names the resume's records (hi_core.run_files) besides the first run's, the
    subjects' records and predictions of both runs, and nothing under attempt<k>/.  Under the unit's spy with these files allowed, hi_predictions (which opens hi_record_resume_1.json through
    hi_core.run_consistency_problems) reads the union; a residue file is outside the allowed items."""
    import hi_core
    lock = tmp_path/'seg'/'hi_holdout_infer.lock'
    root = tmp_path/'seg'/'unet35_v4_ens_holdout_B_v1'
    first, rest, residue = ['OASIS_OAS1_0001_MR1'], ['OASIS_OAS1_0002_MR1', 'OASIS_OAS1_0003_MR1'], ['OASIS_OAS1_0002_MR1']
    names = hi_resumed_root(lock, root, first, rest, residue)
    assert hi_core.run_outs(lock) == [str(root)] and hi_core.complete_subjects(lock, first+rest) == (True, [])
    files = {Path(p) for p in hin.hi_files(lock, first+rest)}
    assert {root/names['started'], root/names['record'], root/'hi_started.json', root/'hi_stopped.json'} <= files
    assert not any('attempt1' in p.parts for p in files) and {root/'subjects'/f'{s}.json' for s in first+rest} <= files
    from pqh_firewall import HoldoutRules, spy
    rules = HoldoutRules(unit='out', mode='holdout', watched=(tmp_path,), files=files, patterns=[(lock.parent, hin.HI_LOCK_PATTERN)])     # as pqh_units.rules_for in the hold-out mode
    with spy(rules) as watch:
        rows = hin.hi_predictions(lock, first+rest)
        with pytest.raises(AssertionError, match='default deny'):
            (root/'attempt1'/'pred'/residue[0]/'probs36.npz').read_bytes()
    opened = {Path(r['path']) for r in watch.opened}
    assert root/names['record'] in opened and root/'hi_stopped.json' in opened and len(watch.breaches) == 1
    assert sorted(rows) == sorted(first+rest) and {Path(r['run']) for r in rows.values()} == {root} and rows[rest[0]]['output_sha256']['probs36.npz'] == 'c'*64
    old = {root/n for n in hin.HI_RUN_FILES}                                       # the first run's three names only (before A5b): the resume's final record would breach
    assert root/names['record'] not in old


# ---- the members' records (Hm5) -------------------------------------------------------------------------------------------------------------------
def members_record(layout, kind, rows, m6=True):
    small, big = hin.run_dirs(layout, 'members', kind)
    for s in rows:
        (big/s).mkdir(parents=True, exist_ok=True)
        np.savez(big/s/'var.npz', V=np.zeros(SHAPE, dtype=np.float32))
        rows[s] = dict(rows[s], var_sha256=io_.sha_file(big/s/'var.npz'))
    W.write_json(small/(hin.MEMBERS_RECORD if kind != 'resume' else hin.MEMBERS_RESUME_RECORD), dict(subjects=rows, spy=dict(m6=m6)))


def test_the_members_records_taken_together(tmp_path):
    layout = layout_of(tmp_path)
    s1, s2 = 'OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0439_MR1'
    with pytest.raises(core.PQRefused, match='order.*has not run'):
        hin.members_view(layout, [s1])
    members_record(layout, 'first_run', {s1: dict(m5=True)})
    with pytest.raises(core.PQRefused, match='order.*without a record'):
        hin.members_view(layout, [s1, s2])
    members_record(layout, 'resume', {s2: dict(m5=True)})
    rows, records = hin.members_view(layout, [s1, s2])
    assert len(records) == 2 and hin.read_variance(rows, s2, SHAPE).dtype == np.float32
    members_record(layout, 'control_rerun', {s1: dict(m5=True), s2: dict(m5=False)})            # a rerun's record counts once it is there
    with pytest.raises(core.PQRefused, match='Hm5 false'):
        hin.members_view(layout, [s1, s2])
    assert hin.members_dirs(layout)[0].name == 'members_rerun'


def test_a_members_record_whose_spy_failed_refuses(tmp_path):
    layout = layout_of(tmp_path)
    members_record(layout, 'first_run', {'OASIS_OAS1_0438_MR1': dict(m5=True)}, m6=False)
    with pytest.raises(core.PQRefused, match='Hm6'):
        hin.members_view(layout, ['OASIS_OAS1_0438_MR1'])


def test_a_variance_file_that_differs_is_refused(tmp_path):
    layout = layout_of(tmp_path)
    members_record(layout, 'first_run', {'OASIS_OAS1_0438_MR1': dict(m5=True)})
    rows, _ = hin.members_view(layout, ['OASIS_OAS1_0438_MR1'])
    np.savez(Path(rows['OASIS_OAS1_0438_MR1']['var_path']), V=np.ones(SHAPE, dtype=np.float32))
    with pytest.raises(core.PQRefused, match='hm5'):
        hin.read_variance(rows, 'OASIS_OAS1_0438_MR1', SHAPE)


def test_the_hash_rule_is_float64_c_order():
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    assert hin.float64_sha(a) == io_.sha_array(a)
    with pytest.raises(core.PQRefused, match='float64'):
        hin.float64_sha(a.astype(np.float32))
