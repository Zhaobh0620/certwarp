"""Shared by the PQH tests: the fixed paths moved into tmp_path, the method's output of a pair as HE writes it (he_core.write_output itself: output/{final.npz, disp.npz, cert.json, line.json}, with
t88/{run.json, field_0600.npz}, repair/{repair.json, repair.npz} in HE's format, and a fallback record), the finalization record of step 5 in the form PQH reads it (实现说明 §5), the dev layout on
the PQ package's synthetic world (tests/test_bayesreg/pq/_world.py: P3's 0438-0439 rebuilt as an HE output; its registered predictions; its V), and the fakes of the machine.  The arrays are synthetic;
the ids are real development ids (lists only)."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'pq', ROOT/'experiments'/'he', ROOT/'experiments'/'holdout', ROOT/'experiments'/'hi'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import he_core  # noqa: E402
import pq_core as core  # noqa: E402
import pq_io as io_  # noqa: E402
import pq_pins as pins  # noqa: E402
import pqh_inputs as hin  # noqa: E402
import pqh_pins as hp  # noqa: E402
import pqh_units as hu  # noqa: E402

from ..pq import _world as PW  # noqa: E402
from ..xb import _world as XW  # noqa: E402

hc = XW.hc
PAIR = hp.DEV_SMOKE_PAIR
SHORT = '0438-0439'
GATE = dict(passed=True, note='the tests query no machine')
CODE = dict(commit='c'*40, clean=True, note='the tests record no code tree')


def quiet(*args, **kwargs):
    pass


def move_pins(tmp_path, monkeypatch):
    """Every fixed path of pqh_pins (and HI's hold-out lock) into tmp_path/results (the XB sandbox's results root)."""
    import hi_core
    results = Path(tmp_path)/'results'
    smoke = results/'he_smoke_v3'
    values = dict(RESULTS=results, MEMBERS_LOCK=results/'seg'/'pq_holdout_members.lock', OUT_LOCK=results/'holdout_B_v1'/'locks'/'pq_out.lock', SMOKE_ROOT=smoke, DEV_PQ_ROOT=smoke/'pq',
                  DEV_MEMBERS_LOCK=smoke/'pq'/'pq_members_dev.lock', DEV_OUT_LOCK=smoke/'pq'/'pq_out_dev.lock', DEV_P3_REPLAY_LOCK=smoke/'pq'/'pq_p3_replay.lock', PQ_V1=results/'pq_v1',
                  REPLAY_ROOT=results/'pq_replay_v1', REPLAY_LOCK=results/'pq_replay_v1'/'pq_replay_gate.lock', MODELS_ROOT=results/'seg'/'unet35_v4_full', DATA_IMAGES=Path(tmp_path)/'d'/'data')
    for name, value in values.items():
        monkeypatch.setattr(hp, name, value)
    monkeypatch.setattr(hi_core, 'HOLDOUT_LOCK', results/'seg'/'hi_holdout_infer.lock')
    monkeypatch.setattr(hi_core, 'DEVCHECK_OUT', results/'seg'/'hi_devcheck_v1')                 # B1: the development check's record and lock, read by check_decision at call time
    monkeypatch.setattr(hi_core, 'DEVCHECK_LOCK', results/'seg'/'hi_devcheck.lock')


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1), encoding='utf-8')
    return path


def certificate(state_hash, n_cells=10):
    return dict(status='certified_feasible', reason='synthetic', min_jd_lower_bound=.25, min_margin_lower_bound=.1, n_cells_checked=n_cells, n_infeasible=0, n_uncertain=0, state_hash=state_hash,
                certificate_version='synthetic', passed=True)


def he_pair(cell, pair, *, raw, nodes, values, grid, taper, labels_sha256=None, status='hx_certified', fallback_u=None, fallback_kind=None, review=None):
    """The method cell of one pair as HE and HB write it: t88/, repair/, output/ (he_core.write_output; a fallback's whole record, with its certificate, in line.json's fallback as run_hx writes it),
    review/review.json (hb_review: C22.H1-5's keys at its top and finalized_part) and terminal.json (hb_core.write_terminal).  hx_certified: v = taper(raw) with the nodes' values; a fallback:
    v = fallback_u.  `review`: fields that replace the review record's (consistent, third_certificate, ...).  Returns (cell, v, the row of HB's finalized_outputs.json)."""
    d = Path(cell)
    t88, rep = d/'t88', d/'repair'
    t88.mkdir(parents=True)
    rep.mkdir(parents=True)
    np.savez(t88/'field_0600.npz', u=np.asarray(raw, dtype=np.float64))
    run = dict(schema=he_core.SCHEMA.format('t88'), pair=pair, status='done', oracle=False, field=dict(file='field_0600.npz', sha256=io_.sha_file(t88/'field_0600.npz'), u_sha256=io_.sha_array(raw)),
               labels_sha256=labels_sha256, segmenter_anchor=dict(start=anchor_start(), start_problems=[], after_channels_problems=[]))     # run_hx.py:379's anchor record (m4, NB1)
    write_json(t88/'run.json', run)
    tapered, _ = taper(np.asarray(raw, dtype=np.float64), grid, hp.TAPER_L)
    tapered = np.ascontiguousarray(tapered, dtype=np.float64)
    np.savez(rep/'repair.npz', nodes=np.asarray(nodes, dtype=np.int64), values=np.asarray(values, dtype=np.float64).reshape(-1, 3))
    v = tapered.copy()
    v.reshape(-1, 3)[np.asarray(nodes, dtype=np.int64)] = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    state = io_.sha_array(v)
    repair = dict(schema=he_core.SCHEMA.format('repair'), pair=pair, status='certified', passed=True, reason='synthetic',
                  source=dict(directory=str(t88), file='field_0600.npz', field_sha256=run['field']['sha256'], u_sha256=run['field']['u_sha256'], tapered_u_sha256=io_.sha_array(tapered)),
                  attempt=dict(file='repair.npz', sha256=io_.sha_file(rep/'repair.npz'), n_free_nodes=int(np.asarray(nodes).size)), decision=dict(file='certified.npz', state_hash=state),
                  certificate=certificate(state))
    write_json(rep/'repair.json', repair)
    line_fallback = None
    if status != 'hx_certified':
        v = np.ascontiguousarray(fallback_u, dtype=np.float64)
        digest = io_.sha_array(v)
        line_fallback = dict(kind=fallback_kind, run_dir=None, file=None, state_hash=digest, certificate=certificate(digest))
    digest = io_.sha_array(v)
    he_core.write_output(d/'output', v, grid, certificate(digest), dict(line_status=status, fallback=line_fallback, pair=pair))
    write_records(d, pair, status=status, state_hash=digest, line_fallback=line_fallback, review=review)
    return d, v, finalized_row(d, status)


def anchor_start(anchor=None):
    """The `start` of run_hx's segmenter_anchor (run_hx.py:305, :379): he_core.segmenter_start_problems's record (he_core.py:209), its keys and nesting, for the anchor (default pqh_pins.ANCHOR as it
    is now: the dev world moves its registry_sha256).  The segmenter_id and policy are inside 'anchor', not at the top (review NB1).  test_pqh_inputs.test_the_anchor_fixture_has_run_hxs_shape
    calls he_core.segmenter_start_problems on the real registry and checks this shape against it."""
    import he_core
    a = dict(hp.ANCHOR if anchor is None else anchor)
    return dict(anchor=dict(a), settings_segmenter=dict(a), registry=f"/code/{a['registry']}", registry_rel=a['registry'], registry_sha256=a['registry_sha256'],
                model_segmenter_ids={f"{a['segmenter_id']}_foldA": a['segmenter_id'], f"{a['segmenter_id']}_foldB": a['segmenter_id']}, rule=he_core.ANCHOR_RULE)


def write_records(cell, pair, *, status, state_hash, line_fallback=None, review=None):
    """review/review.json and terminal.json as hb_review writes them for a finalized pair (both written anew)."""
    d = Path(cell)
    part = dict(line_status=status, output=hb_rows(d, OUTPUT_FILES), state_hash=state_hash, t88_repair=hb_rows(d, T88_FILES),
                fallback=None if not line_fallback else dict(kind=line_fallback['kind'], state_hash=line_fallback['state_hash'], certificate_passed=True, run_dir=None), superseded=[], evidence={})
    doc = dict(schema='bayesreg.hb.review.v1', mode='test', pair=pair, finalized=True, **part, consistent=True, differences=[], review=certificate(state_hash), decision=dict(state_hash=state_hash),
               third_certificate=None, rounds=[dict(state_hash=state_hash, consistent=True)])
    doc.update(review or {})
    rev = d/'review'/'review.json'
    if rev.exists():
        rev.unlink()
    write_json(rev, doc)
    if (d/'terminal.json').exists():
        (d/'terminal.json').unlink()
    write_json(d/'terminal.json', dict(schema='bayesreg.hb.terminal.v1', arm=hp.METHOD_ARM, pair=pair, line_status=status, output_dir='output',
                                       review_record=dict(path='review/review.json', sha256=io_.sha_file(rev)), written_by='test'))


OUTPUT_FILES = ('output/final.npz', 'output/disp.npz', 'output/cert.json', 'output/line.json')     # hb_core.OUTPUT_FILES
T88_FILES = ('t88/run.json', 't88/field_0600.npz', 'repair/repair.json', 'repair/repair.npz')


def hb_rows(d, names):
    return {n: (io_.sha_file(Path(d)/n) if (Path(d)/n).is_file() else None) for n in names}


def finalized_row(cell, status=None):
    """The row of one pair in HB's finalized_outputs.json (hb_review.finalize) from the cell as it is now."""
    d = Path(cell)
    term, rev = d/'terminal.json', d/'review'/'review.json'
    doc = json.loads(rev.read_text()) if rev.is_file() else None
    return dict(cell=str(d), terminal=dict(path=str(term), sha256=io_.sha_file(term)) if term.is_file() else None, review=dict(path=str(rev), sha256=io_.sha_file(rev)) if rev.is_file() else None,
                line_status=json.loads(term.read_text()).get('line_status') if term.is_file() else status,
                part=None if doc is None else {k: doc.get(k) for k in ('output', 'state_hash', 't88_repair', 'fallback', 'superseded', 'evidence')})


def no_output_row(status='hx_provenance_rejected'):
    """The row of a pair recorded at step 4 without an output (terminal.json only)."""
    return dict(cell='/nowhere', terminal=dict(path='/nowhere/terminal.json', sha256='0'*64), review=None, line_status=status, part=None)


def finalized(layout, rows, *, commit='c'*40):
    """HB's finalized_outputs.json (hb_review.finalize's fields) at the layout's place."""
    path = Path(layout.finalized_path)
    if path.exists():
        path.unlink()
    write_json(path, dict(schema='bayesreg.hb.finalized_outputs.v1', batch='test', code=dict(root='test', commit=commit, clean=True, status='', untracked=[]), frozen_tree=None, environment=None,
                          utc_start=None, pairs=rows, utc='2026-11-09T12:00:00Z'))
    return path


def dev_world(tmp_path, monkeypatch):
    """The PQ package's synthetic world (synthetic training draw), its registry formal, the anchor's sha256 its registry's, holdout_common's code tree its tree (admit_dev_pair's allowlist);
    P3's 0438-0439 rebuilt as an HE output in the smoke's copy of segment (ii) with HB's review records, the development V copied as the dev members run.  Returns (world, layout, row)."""
    from bayesreg.seg import structure_layer
    w = PW.build(tmp_path, synthetic=True)
    monkeypatch.setattr(structure_layer, 'FORMAL_REGISTRIES', tuple(structure_layer.FORMAL_REGISTRIES)+(w.registry,))
    monkeypatch.setattr(hp, 'ANCHOR', dict(pins.ANCHOR, registry_sha256=io_.sha_file(w.registry)))
    monkeypatch.setattr(hc, 'REPO', w.repo)
    layout = dev_layout_of(w, tmp_path)
    t88, rep = w.ctx.t88_dir(SHORT), w.ctx.repair_dir(SHORT)
    with np.load(t88/'field_0600.npz') as z:
        raw = np.array(z['u'])
    with np.load(rep/'repair.npz') as z:
        nodes, values = np.array(z['nodes']), np.array(z['values'])
    run = json.loads((t88/'run.json').read_text())
    _, v, row = he_pair(layout.dev_cell, PAIR, raw=raw, nodes=nodes, values=values, grid=w.grid, taper=PW.frozen_formula_taper, labels_sha256=run['labels_sha256'])
    assert v.tobytes() == w.fields[SHORT].tobytes()
    copy_members(w, layout)
    return w, layout, row


def dev_layout_of(w, tmp_path):
    """The dev layout: its code tree is the smoke root's checkout he_smoke_v3/code (I5), here a link to the synthetic code tree."""
    smoke = Path(hp.SMOKE_ROOT)
    smoke.mkdir(parents=True, exist_ok=True)
    if not (smoke/'code').exists():
        (smoke/'code').symlink_to(w.repo)
    return hin.Layout(mode='dev', code=smoke/'code', pairs=(PAIR,), stage=w.stage, pq_small=smoke/'pq', pq_big=smoke/'pq'/'big', method_records=None, dev_cell=smoke/hp.DEV_CELL_REL, images=w.images,
                      models_root=Path(hp.MODELS_ROOT), members_lock=Path(hp.DEV_MEMBERS_LOCK), out_lock=Path(hp.DEV_OUT_LOCK), dev_pred_root=w.ctx.pred_root,
                      forbidden_dirs=(Path(hp.PQ_V1)/'eval', Path(hp.PQ_V1)/'analysis'), batch_root=smoke/'pq'/'batches', v4=w.ctx.v4, extra=dict(watched=(Path(tmp_path),), unwatched=()))


def copy_members(w, layout, subjects=None, **record):
    """The development V of the pair's subjects as a dev members run (var.npz copied; members_record.json with Hm5 passed and the spy passed)."""
    small, big = hin.run_dirs(layout, 'members', 'first_run')
    rows = {}
    for s in subjects or PAIR.split('__'):
        src = w.ctx.pq_root/'members'/s/io_.VAR_FILE
        (big/s).mkdir(parents=True)
        shutil.copy(src, big/s/io_.VAR_FILE)
        rows[s] = dict(m5=True, agreement=1., var_sha256=io_.sha_file(big/s/io_.VAR_FILE))
    write_json(small/hin.MEMBERS_RECORD, dict(dict(schema='test', subjects=rows, m5_passed=True, spy=dict(m6=True, n_files=0, breaches=[]), stage=None), **record))
    return rows


def identity_stand_in(layout):
    """The code identity of the tests' trees (none is a pinned checkout with runs/holdout_B_v1/pinned_code.json): pqh_units.code_check's record, as unit_code_identity maps HB's."""
    return dict(commit=CODE['commit'], head=CODE['commit'], clean=True, identity=None, rule='test: no code tree of a test is the pinned checkout'), []


def deps(**kwargs):
    base = dict(gate=lambda unit: GATE, disk_usage=XW.fake_usage(), identity=identity_stand_in, imports=lambda layout: [], in_unit=lambda name: True, probe=lambda when: dict(when=when, note='test'),
                taper=PW.frozen_formula_taper, log=quiet)
    base.update(kwargs)
    return hu.Deps(**base)


def run_out(layout, **kwargs):
    XW.sealed_module._reset_cache()             # C23.9-13 exception (13): each in-process call of a unit starts with the guard's cache empty
    return hu.out_unit(layout, **kwargs)


def refusals(layout):
    return sorted(Path(layout.batch_root).glob('*/refusals/*.json'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def namespace(**kwargs):
    return SimpleNamespace(**kwargs)


def commit_authorization(kind, unit, *, reason='test\nrecompute: none', **fields):
    """A committed authorization of this tool (holdout_common.load_authorization's form) in the sandbox's authorization repository.  A resume names what it computes again on one line of its
    reason (pqh_units.resume_items, I2)."""
    repo = XW.auth_repo()
    directory = hc.authorization_dir('holdout')
    k = 1
    while (directory/f'{hc.AUTH_PREFIX["holdout"]}{hp.TOOL}_{kind}_{k}.json').exists():
        k += 1
    document = dict(schema=f'bayesreg.{hp.TOOL}.{kind}.v1', written_by="lead on BZ's behalf", words='（测试）BZ：可以', time_local='2026-11-09 9:00 PM CST', utc='2026-11-09T12:00:00Z',
                    reason=reason, tool=hp.TOOL, arm=hp.PQ_ARM, pair=unit, pair_set='holdout', **fields)
    path = XW.write_json(directory/f'{hc.AUTH_PREFIX["holdout"]}{hp.TOOL}_{kind}_{k}.json', document)
    XW.git(repo, 'add', '-A')
    XW.git(repo, 'commit', '-q', '-m', f'{kind} {k}')
    return path


# ---- HI's runs, HI's development check, a hold-out-mode world on the synthetic one (review B1, I1, I2) ------------------------------------------------
def hi_run(lock, out, subjects, *, resume_of=None, record=True, source=None):
    """One run of HI's hold-out inference as hi_core records it: the line of its lock (or of the resume lock), subjects/<s>.json with output_sha256, pred/<s>/ with provenance.json, hi_record.json.
    source: a predictions root whose <s>/{probs36.npz, structure_groups.npz, provenance.json} are copied (the synthetic world's registered predictions), with their real sha256."""
    import hi_core
    line = dict(out=str(out), utc='2026-11-09T00:00:00Z', pid=1, event='first_run' if resume_of is None else 'authorized_resume', resume=None if resume_of is None else dict(previous_out=str(resume_of)))
    path = Path(lock) if resume_of is None else hi_core.resume_lock_path(lock, resume_of)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(line)+'\n')
    (Path(out)/'subjects').mkdir(parents=True)
    for s in subjects:
        d = Path(out)/'pred'/s
        if source is None:
            d.mkdir(parents=True)
            (d/'provenance.json').write_text('{}')
            output = {'probs36.npz': 'a'*64, 'structure_groups.npz': 'b'*64}
        else:
            shutil.copytree(Path(source)/s, d)
            output = {n: io_.sha_file(d/n) for n in ('probs36.npz', 'structure_groups.npz')}
        write_json(Path(out)/'subjects'/f'{s}.json', dict(subject=s, output_sha256=output))
    if record:
        write_json(Path(out)/'hi_record.json', dict(rows={s: {} for s in subjects}))


TEST_ENVIRONMENT = dict(gpu='NVIDIA GeForce RTX 4090', torch='2.5.1', cuda='12.1', cudnn=90100, cudnn_benchmark=False, cudnn_deterministic=False)


def decided(monkeypatch):
    """HI's development check as it ran (its lock line at DEVCHECK_LOCK, its devcheck.json at DEVCHECK_OUT, both moved into tmp_path by move_pins) and a decision constant that passes
    hi_core.check_decision (tests/test_bayesreg/hi/_world.devcheck_document's keys).  Returns the decision."""
    import hi_core as hi
    hi.take_lock(hi.DEVCHECK_LOCK, dict(out=str(hi.DEVCHECK_OUT), plan_sha256=hi.DEVCHECK_PLAN_SHA256, backends=['hs_a'], event=hi.FIRST_RUN, utc='2026-11-07T13:00:00Z', pid=1))
    path = write_json(Path(hi.DEVCHECK_OUT)/hi.DEVCHECK_RECORD, dict(schema=hi.DEVCHECK_SCHEMA, backends=['hs_a'], environment=dict(TEST_ENVIRONMENT, device='cuda'),
                                                                    reading=dict(text=hi.READ_SELECT.format(n=20)), plan=dict(path=hi.DEVCHECK_PLAN_REL, sha256=hi.DEVCHECK_PLAN_SHA256),
                                                                    subjects=list(hi.DEV_SUBJECTS), lock=dict(path=str(hi.DEVCHECK_LOCK), line=hi.lock_lines(hi.DEVCHECK_LOCK)[0])))
    decision = dict(devcheck_json=str(path), devcheck_sha256=io_.sha_file(path), reading=hi.READ_SELECT.format(n=20), backend='hs_a', environment=dict(TEST_ENVIRONMENT), decided_by='test',
                    review='test')
    monkeypatch.setattr(hi, 'DEVCHECK_DECISION', decision)
    return decision


def holdout_world(tmp_path, monkeypatch):
    """The synthetic world in the hold-out mode, for the units' own steps after the admission (the resume of review I2): the code tree the detached checkout holdout_B_v1/code (a link to the
    synthetic code tree), the preregistration pinned to a file there, HI's run of the two subjects (the registered predictions copied), the finalization record of step 5 naming segment (ii)'s cell,
    the members' run of the two subjects; SG's hold-out admission stood in by the development one (holdout_common.admit_holdout_pair -> admit_dev_pair: the admission has its own tests in
    test_pqh_admission.py with the real loaders).  Returns (world, layout)."""
    import hi_core
    w, dev_layout, row = dev_world(tmp_path, monkeypatch)
    checkout = Path(hc.HOLDOUT_ROOT)/'code'
    checkout.parent.mkdir(parents=True, exist_ok=True)
    checkout.symlink_to(w.repo)
    prereg = w.repo/hp.PREREG_REL
    prereg.parent.mkdir(parents=True, exist_ok=True)
    prereg.write_text('synthetic preregistration (test)\n', encoding='utf-8')
    monkeypatch.setattr(hp, 'PREREG_SHA256', io_.sha_file(prereg))
    hi_run(hi_core.HOLDOUT_LOCK, Path(hp.RESULTS)/'seg'/'hi_run_1', PAIR.split('__'), source=w.ctx.pred_root)
    root = Path(hc.HOLDOUT_ROOT)
    layout = hin.Layout(mode='holdout', code=checkout, pairs=(PAIR,), stage=w.stage, pq_small=root/hp.PQ_ARM, pq_big=Path(hc.BIG_ROOT)/hp.PQ_ARM, method_records=root/hp.METHOD_ARM,
                        finalized_path=root/hp.FINALIZED_DIR/hp.FINALIZED_NAME, images=w.images, models_root=Path(hp.MODELS_ROOT), members_lock=Path(hp.MEMBERS_LOCK), out_lock=Path(hp.OUT_LOCK),
                        hi_lock=Path(hi_core.HOLDOUT_LOCK), forbidden_dirs=(root/hp.EV_EVAL_DIR_NAME, Path(hp.PQ_V1)), ev_master_lock=root/hp.EV_EVAL_DIR_NAME/hp.EV_MASTER_LOCK_NAME,
                        batch_root=root/hp.PQ_ARM/'batches', extra=dict(watched=(Path(tmp_path),), unwatched=()))
    finalized(layout, {PAIR: finalized_row(dev_layout.dev_cell)})
    copy_members(w, layout)
    monkeypatch.setattr(hc, 'admit_holdout_pair', lambda fixed, moving: hc.admit_dev_pair(fixed, moving))
    return w, layout
