"""SM3 (runs/20261114-p0/任务书.md): experiments/he/smoke_v3.py -- the he_smoke_v3 tool after the A segment -- on synthetic files in HB's sandbox (CPU; no unit is started, no GPU, no truth, no real
subject's file; the certificate a stand-in of the decision configuration, configuration C's frozen run a stand-in).

- the injections: only the array of the target state hash is changed (a counterexample on every other array), only the fixed values, only for the forced segments; the launcher of (iv-c) only
  after this process forced the target's third certificate and only on this copy's fallback;
- the forced contexts: exactly injection_settings on their copy, refused otherwise; the production command lines refuse them; the forced entries refuse before anything is written;
- the development loop end to end in this process (hb_review.review_pair in its development mode, the third certification's child through forced_third_main): (ii) one round; (iv-a) two rounds,
  the third certificate passing; (iv-b) the third certificate forced, configuration C's stand-in; (iv-c) no candidate, the identity; and the check's items on what they leave;
- the copies (method-side files only, sha256 before and after, the contexts), the queue (order, gates, PQ's dry run before its launch, a failure stops, a root runs once), the check's PQ items and
  the evaluation entry's PQ functions on the development outputs (the lead's ruling 5), /mnt/d's three spellings;
- no production module imports smoke_v3.
"""
import ast
import json
import shutil
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from . import _world as W

R, he, hc, hb, review = W.R, W.he, W.hc, W.hb, W.hb_review
for _p in (W.ROOT/'experiments'/'he', W.ROOT/'experiments'/'pq'):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
import smoke_v3 as V  # noqa: E402

S = V.S
PAIR = S.PAIR


def put(path, document):
    """A record written by the test (its directory made, an earlier one replaced)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, default=str), encoding='utf-8')
    return path


# ---- the world --------------------------------------------------------------------------------------------------------------------------------------
def results():
    return Path(hc.RESULTS)


def smoke_root():
    return results()/'he_smoke_v3'


def dev_stage():
    """A development stage named as the real one (hb_review's development mode checks the name) with the pair 0438-0439 (synthetic arrays, the real ids: the real allowlist admits them)."""
    stage = Path(hc.RESULTS).parent/'dev'/'stage_L2R_val_v1'
    if not (stage/PAIR/'pair.npz').is_file():
        W.XW.write_stage_pair(stage, PAIR, seg=False)
    return stage


def a_arm(root=None, *, extra=None):
    """The A segment's arm a as HE leaves it (HB's writer): context.json, output/, t88/, repair/ (with certified.npz), and beside them the records the copy must not take (the evaluation's)."""
    root = smoke_root() if root is None else root
    arm = S.run_dir(root, 'a')
    ctx = R.make_context(dev_stage(), PAIR, arm, device='cpu', settings=he.frozen_settings(), check_identity=True)
    u, _ = W.write_cell(ctx)
    for name, doc in (('admission.json', dict(rule='test')), ('evaluation.json', dict(metrics=dict(seg35=.9))), ('evaluation_started.json', {}), ('identity_eval_0.json', {}),
                      ('started.json', {}), ('identity.json', {})):
        put(arm/name, doc)
    for rel, doc in (extra or {}).items():
        put(arm/rel, doc)
    return u


def a_segment_records(root, sha, *, verdicts=None, tissue='off'):
    """smoke_he's records of an ended A segment: batch.json, queue_end.json, smoke_report.json (every judged item 过 unless `verdicts`)."""
    put(root/'batch.json', dict(code_sha=sha))
    put(root/'queue_end.json', dict(rc={}))
    items = [dict(item=n, name=f'item{n}', verdict=(verdicts or {}).get(n, S.PASS if n not in (10, 11) else S.DESCRIBE)) for n in range(1, 14)]
    put(root/'smoke_report.json', dict(items=items, tissue_prior=tissue, code_sha=sha, counts={}, summary='test'))


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(R.core, 'frozen_api', lambda: None)                              # third_cert_main's check of the frozen tree (no frozen run here)
    monkeypatch.setattr(R, 'check_identity', lambda ctx, out, name: (dict(repo=dict(head=ctx.get('code_head'))), []))     # the worktree of a test is not the smoke's checkout
    root = smoke_root()
    u = a_arm(root)
    V.copy_all(root, results=results(), stage=dev_stage(), log=lambda text: None)
    sha = R.code_head()
    put(V.v3_dir(root)/'batch.json', dict(code_sha=sha))                                 # the queue's batch record (the check reads its code_sha)
    return SimpleNamespace(root=root, u=u, target=he.array_sha256(u), stage=dev_stage(), sha=sha)


# ---- the injections -----------------------------------------------------------------------------------------------------------------------------
def test_the_injected_certificate_changes_only_the_target_array_and_only_by_its_fixed_values():
    g = W.grid()
    u1, u2 = W.field(g, seed=1), W.field(g, seed=2)
    target = he.array_sha256(u1)
    real = W.Certify()
    perturb = V.InjectedCertify(real, target=target, kind='perturb', role='review')
    a, b = perturb(u1, g), perturb(u2, g)
    assert a.reason == V.PERTURB_VALUE and V.dataclasses.replace(a, reason=real(u1, g).reason) == real(u1, g) and he.passes(a)
    assert b == real(u2, g)                                                              # another array: the real certificate, unchanged
    assert [(c['on_target'], c['applied']) for c in perturb.calls] == [(True, True), (False, False)]
    same, diffs = he.same_certificate(he.certificate_document(a, W.config(), process='t'), he.certificate_document(real(u1, g), W.config(), process='t'))
    assert not same and set(diffs) == {V.PERTURB_FIELD}                                    # a compared field, nothing else
    fail = V.InjectedCertify(real, target=target, kind='fail', role='third')
    a, b = fail(u1, g), fail(u2, g)
    assert a.status == V.FORCED_STATUS and a.reason == V.FORCED_REASON and not he.passes(a) and a.min_jd_lower_bound == real(u1, g).min_jd_lower_bound
    assert b == real(u2, g) and he.passes(b) and fail.forced_failure()
    counted = V.InjectedCertify(real, target=target, kind=None, role='third')
    assert counted(u1, g) == real(u1, g) and counted.record()['n_applied'] == 0 and counted.record()['n_on_target'] == 1 and not counted.forced_failure()
    rec = fail.record()
    assert rec['n_applied'] == 1 and rec['other_states'] == [he.array_sha256(u2)] and rec['target_state_hash'] == target


def test_the_state_hash_is_the_certificates_whatever_the_memory_layout():
    """The target is the C-order sha256 (C07-7, he_core.array_sha256): the same values in another layout are the same state; one value off is another."""
    g = W.grid()
    u = W.field(g, seed=3)
    inj = V.InjectedCertify(W.Certify(), target=he.array_sha256(u), kind='fail', role='third')
    assert inj(np.asfortranarray(u), g).status == V.FORCED_STATUS
    v = u.copy()
    v[2, 2, 2, 0] = np.nextafter(v[2, 2, 2, 0], 1.)
    assert inj(v, g).status == 'certified_feasible'


@pytest.mark.parametrize('kind, target', [('perturb_all', 'a'*64), ('perturb', 'A'*64), ('perturb', 'a'*63), ('fail', None)])
def test_the_injected_certificate_takes_only_its_kinds_and_a_sha256(kind, target):
    with pytest.raises(ValueError):
        V.InjectedCertify(W.Certify(), target=target, kind=kind, role='x')


def test_the_launcher_of_iv_c_starts_nothing_only_after_the_targets_forced_third_certificate_on_this_copy(tmp_path):
    g = W.grid()
    u = W.field(g, seed=1)
    out = tmp_path/'iv-c'/'a'/'0438-0439'
    calls = []
    real = lambda run_dir, short: calls.append(str(run_dir)) or dict(kind='unit', started=True)      # noqa: E731
    run_dir = out/'fallback_configc_after_third_0'/'run'
    run_dir.mkdir(parents=True)
    counted = V.InjectedCertify(W.Certify(), target=he.array_sha256(u), kind='fail', role='third')
    launcher = V.NoCandidateLauncher(real, counted, out)
    assert launcher(run_dir, 'x')['started'] is True and calls == [str(run_dir)]                 # nothing forced yet: the real launcher
    counted(W.field(g, seed=2), g)
    assert launcher(run_dir, 'x')['started'] is True and len(calls) == 2                         # another array's certificate: still the real one
    counted(u, g)
    for other in (out/'fallback_configc'/'run', out/'fallback_configc_after_third_0'/'other', tmp_path/'iv-b'/'a'/'0438-0439'/'fallback_configc_after_third_0'/'run'):
        other.mkdir(parents=True, exist_ok=True)
        assert launcher(other, 'x')['started'] is True                                          # step 4's fallback, another directory, another copy
    assert len(calls) == 5 and not (run_dir/V.NO_CANDIDATE_RECORD).exists()
    got = launcher(run_dir, 'x')
    assert got['started'] is False and got['kind'] == V.NO_CANDIDATE_KIND and len(calls) == 5 and he.read_json(run_dir/V.NO_CANDIDATE_RECORD)['target_state_hash'] == he.array_sha256(u)
    assert launcher.record()['n_applied'] == 1 and launcher.record()['n_calls'] == 6
    never = V.NoCandidateLauncher(real, V.InjectedCertify(W.Certify(), target=he.array_sha256(u), kind=None, role='third'), out)
    never.certify(u, g)
    assert never(run_dir, 'x')['started'] is True                                                # a counting certificate never forces: never the injection


# ---- the forced contexts --------------------------------------------------------------------------------------------------------------------------
def test_the_injection_values_are_the_frozen_ones_with_the_segments_injection_only():
    frozen = he.frozen_settings()
    for segment, kinds in V.INJECTIONS.items():
        s = V.injection_settings(segment, 'b'*64)
        assert sorted(k for k in set(s) | set(frozen) if he.canonical(s.get(k)) != he.canonical(frozen.get(k))) == ['name', 'smoke_injection']
        assert not he.is_frozen(s) and s['smoke_injection']['target_state_hash'] == 'b'*64 and tuple(s['smoke_injection']['injections']) == kinds
        assert he.certificate_config(s) == he.certificate_config(frozen) and s['configc'] == frozen['configc'] and s['units'] == frozen['units']
    assert V.INJECTIONS == {'iv-a': ('review',), 'iv-b': ('review', 'third'), 'iv-c': ('review', 'third', 'configc')}
    with pytest.raises(ValueError):
        V.injection_settings('ii', 'b'*64)


def test_the_copies_and_their_contexts(world):
    root = world.root
    for segment in V.SEGMENTS:
        doc = he.read_json(V.copy_record_path(root, segment))
        ctx = he.read_json(V.cell_dir(root, segment)/'context.json')
        assert doc['all_same'] is True and doc['target_state_hash'] == world.target == doc['line_state_hash'] and doc['line_status'] == he.HX
        names = sorted(r['file'] for r in doc['files'])
        assert 'admission.json' in names and 'repair/certified.npz' in names and sorted(n for n in names if n.startswith('output/')) == [f'output/{n}' for n in sorted(V.OUTPUT_FILES)]
        assert not any(w in n for n in names for w in ('evaluation', 'identity')) and 'evaluation.json' in doc['not_copied'] and 'context.json' in doc['not_copied']
        assert ctx['out'] == str(V.cell_dir(root, segment)) and ctx['code_head'] == world.sha and ctx['check_identity'] is True
        if segment == 'ii':
            assert ctx['settings_frozen'] is True and ctx['test_context'] is None and V.injection_context_problem(ctx, root, segment, results=results(), stage=world.stage)
            assert R.cli_context_problem(ctx, results=results()) is None                       # (ii) runs through hb_review.py dev-unit, the production command line
        else:
            assert ctx['settings_frozen'] is False and ctx['test_context'] == R.TEST_CONTEXT and V.injection_context_problem(ctx, root, segment, results=results(), stage=world.stage) is None
            assert R.cli_context_problem(ctx, results=results())                               # every production command line refuses a forced context
    base = he.read_json(V.v3_dir(root)/'baseline.json')
    assert base['a_arm']['files'] and 'evaluation.json' in base['a_arm']['files']
    item = V.v1_copies(root)
    assert item['verdict'] == S.PASS, item
    with pytest.raises(V.run.Stop):
        V.copy_segment(root, 'ii', stage=world.stage)                                        # once


@pytest.mark.parametrize('how', ['frozen_values', 'another_segment', 'another_target', 'another_copy', 'not_checked', 'not_marked', 'another_pair', 'another_root', 'hold_out', 'direct_launcher'])
def test_a_forced_context_is_refused_unless_exactly_its_segments_injection_on_its_copy(world, how):
    root = world.root
    ctx = he.read_json(V.cell_dir(root, 'iv-a')/'context.json')
    assert V.injection_context_problem(ctx, root, 'iv-a', results=results(), stage=world.stage) is None
    segment = 'iv-a'
    if how == 'frozen_values':
        ctx = dict(ctx, settings=he.frozen_settings())
    elif how == 'another_segment':
        ctx = dict(ctx, settings=V.injection_settings('iv-b', world.target))
    elif how == 'another_target':
        ctx = dict(ctx, settings=V.injection_settings('iv-a', 'c'*64))
    elif how == 'another_copy':
        segment = 'iv-b'                                                                   # iv-a's context named as iv-b's
    elif how == 'not_checked':
        ctx = dict(ctx, check_identity=False)
    elif how == 'not_marked':
        ctx = dict(ctx, test_context=None)
    elif how == 'another_pair':
        ctx = dict(ctx, pair=hc.DEV_PAIRS[1])
    elif how == 'another_root':
        root = results()/'hs_v1'
    elif how == 'hold_out':
        ctx = dict(ctx, pair_set='holdout')                                                # a hold-out context never takes the injection
    elif how == 'direct_launcher':
        ctx = dict(ctx, launcher=dict(kind='direct'))
    assert V.injection_context_problem(ctx, root, segment, results=results(), stage=world.stage)


def test_the_forced_entries_refuse_a_context_that_is_not_forced_before_anything_is_written(world):
    root = world.root
    ii = V.cell_dir(root, 'ii')/'context.json'
    assert V.forced_third_main(ii, root, 'iv-a', results=results(), stage=world.stage, certification=W.Certify().pair()) == review.THIRD_REFUSED_EXIT
    shutil.rmtree(V.cell_dir(root, 'iv-a'))
    V.cell_dir(root, 'ii').rename(V.cell_dir(root, 'iv-a'))                               # the frozen context of (ii) where (iv-a)'s copy is
    assert V.forced_review_main(root, 'iv-a', world.sha, results=results(), stage=world.stage, certification=W.Certify().pair()) == 2
    assert not (V.cell_dir(root, 'iv-a')/'review').exists() and not (V.segment_root(root, 'iv-a')/'review').exists() and not (V.segment_root(root, 'iv-a')/'injection').exists()


def test_the_third_certifications_child_of_a_forced_review_is_this_files_own_entry(world, monkeypatch):
    seen = []
    monkeypatch.setattr(V.subprocess, 'run', lambda argv, **kw: seen.append((argv, kw)) or subprocess.CompletedProcess(argv, 0))
    ctx = he.read_json(V.cell_dir(world.root, 'iv-b')/'context.json')
    assert V.SmokeThirdChild(world.root, 'iv-b')(ctx) == 0
    argv, kw = seen[0]
    assert argv[:3] == [V.PY, '-B', str(V.SMOKE_V3)] and argv[3] == 'forced-third' and argv[argv.index('--segment')+1] == 'iv-b' and argv[argv.index('--root')+1] == str(world.root)
    assert argv[argv.index('--context')+1] == str(R.context_path(ctx)) and 'timeout' not in kw


# ---- the development loop in this process ---------------------------------------------------------------------------------------------------------
def configc_stand_in(*, start):
    """run_hx.configc_fallback's stand-in (no frozen run): it calls the launcher it is given as configuration C's would; when the launcher started a run, a field of the grid is the candidate,
    certified by the chain's own certify (the injected one: a pass-through for this array)."""
    def configc(ctx, pin, *, launcher, certify, config, suffix=''):
        d = Path(ctx['out'])/f'fallback_configc{suffix}'
        (d/'run').mkdir(parents=True)
        launch = dict(launcher(d/'run', S.SHORT), seconds=1.)
        rec = dict(kind='configC', run_dir=str(d/'run'), file=None, state_hash=None, certificate=None, launch=launch, candidates=[], error=None)
        if launch.get('started') is False:
            rec['error'] = 'class unsure: no candidate (D75)'
            put(d/'fallback.json', rec)
            return None, None, rec
        u2 = W.field(pin.grid, scale=.1, seed=5)
        digest = he.array_sha256(u2)
        doc = he.certificate_document(certify(u2, pin.grid), config, process='test configuration C', array_sha256_of_u=digest)
        rec.update(state_hash=digest, certificate=doc, candidates=[dict(file='final_native.npz', state_hash=digest, passed=he.passes(doc))])
        put(d/'fallback.json', rec)
        return u2, doc, rec
    return configc


def tag_he_c3(run_dir, short):
    """firewall_c.UnitLauncher's record for tag-he-c3 (the frozen run not started here)."""
    argv = ['systemd-run', '--user', f'--unit={R.THIRD_UNIT_PREFIX}-{short}', '--wait', '--collect', *V.fc.UNIT_RESOURCES, V.fc.UNIT_OOM_POLICY]
    return dict(kind='unit', unit=f'{R.THIRD_UNIT_PREFIX}-{short}', argv=argv, returncode=0, probes={n: dict(unit=f'{R.THIRD_UNIT_PREFIX}-{short}.service') for n in V.fc.PROBE_FILES},
                started=True)


def in_process_third(root, segment, stage, launcher):
    """The forced third certification's child in this process (its own process in production: the guard's cache is empty there; reset here, as HB's tests)."""
    def third(ctx):
        W.XW.sealed_module._reset_cache()
        return V.forced_third_main(R.context_path(ctx), root, segment, results=results(), stage=stage, certification=W.Certify().pair(), launcher=launcher)
    return third


def run_forced(world, segment, monkeypatch, *, probe=W.probe, configc=True, batch_sha=None, launcher=tag_he_c3):
    if configc:
        monkeypatch.setattr(R, 'configc_fallback', configc_stand_in(start=True))
    W.XW.sealed_module._reset_cache()
    return V.forced_review_main(world.root, segment, world.sha if batch_sha is None else batch_sha, results=results(), stage=world.stage, certification=W.Certify().pair(),
                                third=in_process_third(world.root, segment, world.stage, launcher), probe=probe)


def records(world, segment):
    cell = V.cell_dir(world.root, segment)
    stops = sorted((cell/'review').glob('review_not_finalized*.json'))
    for p in stops:
        print(p.name, he.read_json(p)['kind'], he.read_json(p)['reason'][:500])
    return he.read_json(cell/'review'/'review.json'), he.read_json(cell/'terminal.json'), stops


def test_ii_the_development_review_of_the_copy_is_consistent_in_one_round(world):
    root = world.root
    seg = V.segment_root(root, 'ii')
    ctx = R.read_context(V.cell_dir(root, 'ii')/'context.json')
    mode = review.Mode('dev', seg/'review', world.sha, review.dev_policy(seg, world.stage/PAIR), segment=seg)
    W.manifest(mode, [])
    mode.manifest().write_text(json.dumps(dict(pairs={PAIR: dict(step4='output', cell=str(V.cell_dir(root, 'ii')), files=hb.file_rows(V.cell_dir(root, 'ii'), hb.INPUT_FILES))})))
    W.XW.sealed_module._reset_cache()
    assert review.review_pair(ctx, mode, batch=str(mode.root), certification=W.Certify().pair(), probe=W.probe) == 0
    doc, term, stops = records(world, 'ii')
    assert doc['consistent'] is True and len(doc['rounds']) == 1 and term['line_status'] == he.HX and stops == []
    item = V.v4_review_ii(root, world.sha)
    assert item['verdict'] == S.PASS, item


def test_iv_a_two_rounds_and_the_third_certificate_passes(world, monkeypatch):
    assert run_forced(world, 'iv-a', monkeypatch) == 0
    doc, term, stops = records(world, 'iv-a')
    cell = V.cell_dir(world.root, 'iv-a')
    assert stops == [] and term['line_status'] == he.HX and len(doc['rounds']) == 2 and doc['rounds'][0]['differences'].keys() == {V.PERTURB_FIELD}
    assert (cell/'output'/'review_mismatch_0.json').is_file() and he.read_json(cell/'output'/'third_certificate.json')['passed'] is True
    review_inj = he.read_json(V.segment_root(world.root, 'iv-a')/'injection'/'review.json')
    third_inj = he.read_json(V.segment_root(world.root, 'iv-a')/'injection'/'third.json')
    assert review_inj['n_applied'] == 2 and review_inj['other_states'] == [] and third_inj['n_applied'] == 0 and third_inj['kind'] is None and third_inj['exit'] == 0
    item = V.forced_review_item('V8', world.root, 'iv-a', world.sha)
    assert item['verdict'] == S.PASS, item


def test_iv_b_the_forced_third_certificate_goes_to_configuration_c(world, monkeypatch):
    assert run_forced(world, 'iv-b', monkeypatch) == 0
    doc, term, stops = records(world, 'iv-b')
    cell = V.cell_dir(world.root, 'iv-b')
    assert stops == [] and term['line_status'] == he.FALLBACK_C and doc['rounds'][1]['consistent'] is True and doc['state_hash'] != world.target
    assert he.read_json(cell/'output_superseded_0'/'third_certificate.json')['certificate']['status'] == V.FORCED_STATUS
    third_inj = he.read_json(V.segment_root(world.root, 'iv-b')/'injection'/'third.json')
    assert third_inj['n_applied'] == 1 and third_inj['other_states'] == [doc['state_hash']]                 # the candidate: the real certificate
    item = V.forced_review_item('V9', world.root, 'iv-b', world.sha)
    assert item['verdict'] == S.PASS, item


def test_iv_b_with_the_end_probe_of_tag_he_c3_missing_reads_not_passed_on_that_item_only(world, monkeypatch):
    """Revision 3 (diff review 1): tag-he-c3's unit_probe_end.json not read (None) -> V9 does not pass, and tag_he_c3_probes is its only check that does not hold."""
    def launcher(run_dir, short):
        launch = tag_he_c3(run_dir, short)
        launch['probes']['unit_probe_end.json'] = None
        return launch
    assert run_forced(world, 'iv-b', monkeypatch, launcher=launcher) == 0
    item = V.forced_review_item('V9', world.root, 'iv-b', world.sha)
    assert item['verdict'] == S.OPEN and [k for k, v in item['values']['checks'].items() if v is not True] == ['tag_he_c3_probes']


def test_iv_c_no_candidate_and_the_identity(world, monkeypatch):
    assert run_forced(world, 'iv-c', monkeypatch) == 0
    doc, term, stops = records(world, 'iv-c')
    cell = V.cell_dir(world.root, 'iv-c')
    assert stops == [] and term['line_status'] == he.FALLBACK_ID and (cell/'fallback_configc_after_third_0'/'run'/V.NO_CANDIDATE_RECORD).is_file()
    third_inj = he.read_json(V.segment_root(world.root, 'iv-c')/'injection'/'third.json')
    assert third_inj['launcher']['n_applied'] == 1 and third_inj['n_applied'] == 1
    item = V.forced_review_item('V11', world.root, 'iv-c', world.sha)
    assert item['verdict'] == S.PASS, item
    (cell/'fallback_configc_after_third_0'/'run'/V.NO_CANDIDATE_RECORD).unlink()          # without the launcher's record, (iv-c) does not read as passed
    item = V.forced_review_item('V11', world.root, 'iv-c', world.sha)
    assert item['verdict'] == S.OPEN and item['values']['checks']['no_candidate'] is False


def test_a_forced_review_whose_batch_commit_is_not_the_copys_is_refused_before_anything_is_written(world, monkeypatch):
    """forced_review_main's --batch-sha against the copy's code_head (review I3): refused (2), no review record, no lock, no injection record."""
    assert run_forced(world, 'iv-a', monkeypatch, batch_sha='0'*40) == 2
    assert not (V.cell_dir(world.root, 'iv-a')/'review').exists() and not (V.segment_root(world.root, 'iv-a')/'review').exists()
    assert not (V.segment_root(world.root, 'iv-a')/'injection').exists()


def test_a_forced_review_whose_unit_probe_is_missing_does_not_read_as_passed(world, monkeypatch):
    """Revision 2 (review minor 3): the review unit's probes (review.json unit.start, unit.end) are there."""
    assert run_forced(world, 'iv-a', monkeypatch, probe=lambda when: W.probe(when) if when == 'start' else {}) == 0
    item = V.forced_review_item('V8', world.root, 'iv-a', world.sha)
    assert item['verdict'] == S.OPEN and item['values']['checks']['unit_probes'] is False


def test_iv_c_through_the_real_configuration_c_fallback(world, monkeypatch):
    """Revision 2 (review minor 9): (iv-c) through run_hx.configc_fallback itself -- the derived pair, prepare, the firewall gate and the classifier stood in (no frozen tree copied, no
    classify_run.py), the launcher the injection's: no candidate by fc.candidates' own rule, then the identity."""
    fc = R.fc
    seen = {}
    monkeypatch.setattr(fc, 'derive_pair', lambda pin, d, pair: Path(d).mkdir() or dict(lam_hex=(1.).hex(), stand_in='derive_pair'))
    monkeypatch.setattr(fc, 'prepare', lambda derived, run_dir, **kw: Path(run_dir).mkdir() or dict(slot='slot', frozen_files=[], seconds=0.))
    monkeypatch.setattr(fc, 'firewall_check', lambda pin, derived, run_dir, pair: (dict(passed=True, stand_in='firewall_check'), []))
    monkeypatch.setattr(fc, 'dmesg_counts', lambda since: (0, 0))
    monkeypatch.setattr(fc, 'classify', lambda run_dir, **kw: seen.setdefault('classify', dict({'class': 'unsure'}, kind='stand-in', reason='a run that never ran', script_sha256=None)))
    assert run_forced(world, 'iv-c', monkeypatch, configc=False) == 0
    doc, term, stops = records(world, 'iv-c')
    cell = V.cell_dir(world.root, 'iv-c')
    fb = he.read_json(cell/'fallback_configc_after_third_0'/'fallback.json')
    assert stops == [] and term['line_status'] == he.FALLBACK_ID and seen['classify']['class'] == 'unsure'
    assert fb['launch']['kind'] == V.NO_CANDIDATE_KIND and fb['candidates'] == [] and 'no candidate' in fb['error'] and fb['state_hash'] is None
    assert he.read_json(cell/'fallback_configc_after_third_0'/'run'/'slot.json')['slot'] == 'slot'
    item = V.forced_review_item('V11', world.root, 'iv-c', world.sha)
    assert item['verdict'] == S.PASS, item


def test_a_forced_review_on_a_copy_whose_injection_did_not_act_does_not_read_as_passed(world, monkeypatch):
    """The counterexample of the check: the injection records say the review's injection acted once (not twice) -- the item is not passed."""
    assert run_forced(world, 'iv-a', monkeypatch) == 0
    path = V.segment_root(world.root, 'iv-a')/'injection'/'review.json'
    doc = he.read_json(path)
    path.unlink()
    put(path, dict(doc, n_applied=1))
    item = V.forced_review_item('V8', world.root, 'iv-a', world.sha)
    assert item['verdict'] == S.OPEN and item['values']['checks']['injections'] is False


def test_v2_sees_a_changed_file_of_the_a_arm_or_of_he_smoke_v2(world):
    v2 = results()/'he_smoke_v2'
    assert V.v2_untouched(world.root, results())['verdict'] == S.PASS
    (S.run_dir(world.root, 'a')/'admission.json').write_text('{"changed": 1}')
    item = V.v2_untouched(world.root, results())
    assert item['verdict'] == S.OPEN and item['values']['a_arm_differs'] == ['admission.json'] and not v2.exists()


def test_v2_sees_a_changed_file_of_he_smoke_v2(monkeypatch):
    """V2's other branch (review I3): he_smoke_v2 hashed before the copies and again in the check."""
    root = smoke_root()
    a_arm(root)
    v2 = results()/'he_smoke_v2'
    put(v2/'a'/S.SHORT/'evaluation.json', dict(metrics=dict(seg35=.8)))
    put(v2/'batch.json', dict(code_sha='a'*40))
    V.copy_all(root, results=results(), stage=dev_stage(), log=lambda text: None)
    assert V.v2_untouched(root, results())['verdict'] == S.PASS
    assert sorted(he.read_json(V.v3_dir(root)/'baseline.json')['he_smoke_v2']['files']) == ['a/0438-0439/evaluation.json', 'batch.json']
    put(v2/'batch.json', dict(code_sha='b'*40))
    item = V.v2_untouched(root, results())
    assert item['verdict'] == S.OPEN and item['values']['he_smoke_v2_differs'] == ['batch.json'] and item['values']['checks']['a_arm'] is True
    (v2/'batch.json').unlink()
    put(v2/'extra.json', {})
    assert V.v2_untouched(root, results())['values']['he_smoke_v2_differs'] == ['batch.json', 'extra.json']                 # removed and added count too


def test_a_copy_whose_bytes_differ_from_the_source_is_refused(monkeypatch):
    """The copy's three sha256 (before, after, the copy) compared (review I3): a copy that wrote one byte wrong stops the copy step, no copy.json written."""
    root = smoke_root()
    a_arm(root)
    real = V.shutil.copyfile

    def bad(src, dst):
        real(src, dst)
        if Path(dst).name == 'run.json':
            data = bytearray(Path(dst).read_bytes())
            data[-2] ^= 1
            Path(dst).write_bytes(bytes(data))
    monkeypatch.setattr(V.shutil, 'copyfile', bad)
    with pytest.raises(V.run.Stop, match='the copy does not hold'):
        V.copy_all(root, results=results(), stage=dev_stage(), log=lambda text: None)
    assert not V.copy_record_path(root, 'ii').exists() and not (V.cell_dir(root, 'ii')/'context.json').exists()


def test_a_copy_of_an_arm_whose_output_holds_an_evaluation_record_is_refused(monkeypatch):
    root = smoke_root()
    a_arm(root, extra={'output/review_mismatch_0.json': {}})
    with pytest.raises(V.run.Stop, match='output/ holds'):
        V.copy_all(root, results=results(), stage=dev_stage(), log=lambda text: None)
    assert not V.cell_dir(root, 'ii').exists()


# ---- the queue ---------------------------------------------------------------------------------------------------------------------------------
class FakeSystem:
    """The machine as the v3 queue sees it; run() leaves what a step would leave (or not, or exits non-zero)."""

    def __init__(self, root, sha, *, rc=None, skip=(), runs_from_it=True, head=None, clean=True, load=.2, mem=12., checkout_raises=None):
        self.root, self.sha, self.rc, self.skip, self.runs_from_it = Path(root), sha, dict(rc or {}), set(skip), runs_from_it
        self.head, self.clean, self.load, self.mem, self.checkout_raises = sha if head is None else head, clean, load, mem, checkout_raises
        self.argvs, self.clock, self.up, self.gpu_asked = [], 0., 100., []

    def load1(self):
        return self.load

    def units(self):
        return []

    def gpu_free_mib(self):
        self.gpu_asked.append(self.clock)
        return 23000.

    def home_free_gb(self):
        return 500.

    def mem_available_gib(self):
        return self.mem

    def host_free_gib(self):
        return 70.

    def now(self):
        return self.clock

    def sleep(self, seconds):
        self.clock += seconds

    def uptime(self):
        self.up += 1.
        return self.up

    def code_identity(self, code):
        return {}, []

    def checkout(self, code):
        if self.checkout_raises is not None:
            raise self.checkout_raises
        return dict(code=str(code), head=self.head, clean=self.clean, status='' if self.clean else ' M experiments/he/run_hx.py', this_process_tree='x', runs_from_it=self.runs_from_it)

    def run(self, argv, log):
        assert 'py'+'test' not in ' '.join(map(str, argv))
        self.argvs.append(list(argv))
        self.clock += 60.
        if 'run_pqh.py' in str(argv[2]):
            what, command = argv[3], argv[4]
            step = next(s for s in V.STEPS if s.command == command)
            if what == 'dry-run':
                return self.rc.get(f'{step.name}:dry-run', 0)
        else:
            unit = next(a for a in argv if a.startswith('--unit=')).split('=', 1)[1]
            step = next(s for s in V.STEPS if s.unit == unit)
        if step.name not in self.skip:
            for path in V.expected_files(step, self.root):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(summary='test') if path.name == f'{V.REPORT}.json' else {}))
        return self.rc.get(step.name, 0)


def queue_world(monkeypatch, *, verdicts=None):
    root = smoke_root()
    a_arm(root)
    sha = R.code_head()
    a_segment_records(root, sha, verdicts=verdicts)
    monkeypatch.setattr(V._pins(), 'SMOKE_ROOT', root)
    return root, sha


def options(root, **kw):
    return SimpleNamespace(**dict(dict(root=str(root), wait_minutes=1, gap_seconds=15, min_host_gib=15.), **kw))


def test_the_queue_runs_the_steps_in_order_and_ends_with_the_check(monkeypatch):
    root, sha = queue_world(monkeypatch)
    system = FakeSystem(root, sha)
    assert V.queue(options(root), system, results=results(), stage=dev_stage()) == 0
    names = [s.name for s in V.STEPS]
    assert names == ['copy', 'i-members', 'ii-review', 'ii-pq', 'iii-replay', 'iva-review', 'ivb-review', 'ivb-pq', 'ivc-review', 'ivc-pq', 'v3-check']
    launched = [he.read_json(V.v3_dir(root)/'steps'/f'{n}.launched.json') for n in names]
    assert [x['unit'] for x in launched] == [None, 'tag-hb-pq-members-dev', 'tag-he-smoke-ii-review', 'tag-hb-pq-out-dev', 'tag-hb-pq-p3-replay', 'tag-he-smoke-iva-review',
                                             'tag-he-smoke-ivb-review', 'tag-hb-pq-out-dev-ivb', 'tag-he-smoke-ivc-review', 'tag-hb-pq-out-dev-ivc', 'tag-he-smoke-v3-check']
    pq = [a for a in system.argvs if 'run_pqh.py' in str(a[2])]
    assert [a[3:5] for a in pq] == [['dry-run', 'members-dev'], ['launch', 'members-dev'], ['dry-run', 'out-dev'], ['launch', 'out-dev'], ['dry-run', 'p3-replay'], ['launch', 'p3-replay'],
                                    ['dry-run', 'out-dev-ivb'], ['launch', 'out-dev-ivb'], ['dry-run', 'out-dev-ivc'], ['launch', 'out-dev-ivc']]
    assert all(a[2] == str(root/'code'/'experiments/pq/run_pqh.py') for a in pq)
    units = [a for a in system.argvs if a[0] == 'systemd-run']
    assert [next(x for x in a if x.startswith('--unit=')) for a in units] == ['--unit=tag-he-smoke-ii-review', '--unit=tag-he-smoke-iva-review', '--unit=tag-he-smoke-ivb-review',
                                                                              '--unit=tag-he-smoke-ivc-review', '--unit=tag-he-smoke-v3-check']
    ii = units[0]
    assert ii[ii.index('-B')+1] == f'{root}/code/experiments/hb/hb_review.py' and ii[ii.index('-B')+2:] == ['dev-unit', '--cell', str(V.cell_dir(root, 'ii')), '--root', str(root/'ii'), '--batch-sha', sha]
    assert units[2][-7:] == ['forced-review', '--root', str(root), '--segment', 'iv-b', '--batch-sha', sha]
    for a in units:
        assert he.unit_argv_problems(a, V.EVALUATION) == [] and '--property=OOMPolicy=continue' in a and '--setenv=CUDA_VISIBLE_DEVICES=' in a and f'--working-directory={root}/code' in a
    gates = {x['step']: x['gate_mem_available_gib'] for x in launched}
    assert gates['ii-review'] == gates['iva-review'] == gates['ivb-review'] == 10. and gates['i-members'] == 8. and gates['ii-pq'] == 9.
    assert he.read_json(V.v3_dir(root)/'steps'/'ii-pq.launched.json')['before']['cell']['output/line.json']                    # D114 (a): the copy before PQ
    batch = he.read_json(V.v3_dir(root)/'batch.json')
    assert batch['code_sha'] == sha and batch['a_segment']['problems'] == [] and batch['plan']['path'] == V.PLAN_REL
    assert he.read_json(V.v3_dir(root)/'queue_end.json')['summary'] == 'test'
    assert V.queue(options(root), FakeSystem(root, sha), results=results(), stage=dev_stage()) == 8                           # a root runs once
    assert 'were launched in this root before' in he.read_json(V.v3_dir(root)/'queue_stop.json')['reason']


@pytest.mark.parametrize('how', ['dry_run', 'exit', 'no_record'])
def test_a_failed_step_stops_the_queue_and_nothing_after_it_starts(monkeypatch, how):
    root, sha = queue_world(monkeypatch)
    system = FakeSystem(root, sha, rc={'ii-pq:dry-run': 9} if how == 'dry_run' else {'ii-pq': 30} if how == 'exit' else None, skip={'ii-pq'} if how == 'no_record' else ())
    assert V.queue(options(root), system, results=results(), stage=dev_stage()) == 8
    pq = [a[3:5] for a in system.argvs if 'run_pqh.py' in str(a[2])]
    assert pq[-1] == (['dry-run', 'out-dev'] if how == 'dry_run' else ['launch', 'out-dev'])                                  # a dry run with problems: no launch
    done = he.read_json(V.v3_dir(root)/'steps'/'ii-pq.done.json')
    assert not (V.v3_dir(root)/'steps'/'iii-replay.launched.json').exists()
    assert (done['rc'] == 0 and done['missing']) if how == 'no_record' else done['rc'] != 0
    assert 'the queue stops here' in he.read_json(V.v3_dir(root)/'queue_stop.json')['reason']


@pytest.mark.parametrize('how', ['a_not_passed', 'a_with_c', 'checkout', 'root', 'head', 'unclean', 'a_arm_fallback', 'report_sha', 'no_queue_end', 'no_report', 'exception'])
def test_the_queue_refuses_before_any_step(monkeypatch, how):
    """The start's checks, each alone (revision 2, review I3): the A segment (its items, C off, its report's commit, queue_end.json, arm a hx_certified), the checkout (this process's tree,
    its HEAD the A segment's commit, no tracked file changed), the root's pin; an exception of the queue is recorded too (review I2).  Nothing launched, nothing copied."""
    root, sha = queue_world(monkeypatch, verdicts={4: S.OPEN} if how == 'a_not_passed' else None)
    if how == 'a_with_c':
        a_segment_records(root, sha, tissue='moment3')
    elif how == 'report_sha':
        report = he.read_json(root/'smoke_report.json')
        put(root/'smoke_report.json', dict(report, code_sha='f'*40))
    elif how == 'no_queue_end':
        (root/'queue_end.json').unlink()
    elif how == 'no_report':
        (root/'smoke_report.json').unlink()
    elif how == 'a_arm_fallback':
        line = S.run_dir(root, 'a')/'output'/'line.json'
        put(line, dict(he.read_json(line), line_status=he.FALLBACK_C))
    if how == 'root':
        monkeypatch.setattr(V._pins(), 'SMOKE_ROOT', results()/'he_smoke_v4')
        assert V.queue(options(root), FakeSystem(root, sha), results=results(), stage=dev_stage()) == 2 and not V.v3_dir(root).exists()
        return
    system = FakeSystem(root, sha, runs_from_it=how != 'checkout', head='e'*40 if how == 'head' else None, clean=how != 'unclean',
                        checkout_raises=RuntimeError('git is gone') if how == 'exception' else None)
    assert V.queue(options(root), system, results=results(), stage=dev_stage()) == 8 and system.argvs == []
    assert not (V.v3_dir(root)/'steps').exists() and not (V.cell_dir(root, 'ii')).exists() and not (V.v3_dir(root)/'batch.json').exists()
    stop = he.read_json(V.v3_dir(root)/'queue_stop.json')
    assert stop['launched'] == []
    want = {'a_not_passed': 'items not passed', 'a_with_c': 'tissue_prior', 'checkout': "not the smoke's checkout", 'head': 'is not the A segment', 'unclean': 'tracked files changed',
            'a_arm_fallback': "arm a's line status", 'report_sha': 'is not the batch', 'no_queue_end': 'did not end', 'no_report': 'was not read', 'exception': 'RuntimeError: git is gone'}[how]
    assert want in stop['reason'], stop['reason']
    if how == 'exception':
        assert 'Traceback' in stop['traceback']


def test_a_gate_closed_before_the_first_step_lets_the_same_start_string_start_again(monkeypatch):
    """Revision 2 (review I2), revision 3 (the lead's ruling B2): the machine gate stays closed before the first step -> STOP of kind gate, nothing launched; the same start later goes on
    (batch.json taken again: the same code and plan; a restart record with the queue's options); once a step is launched the root is used."""
    root, sha = queue_world(monkeypatch)
    for k in range(2):                                                                  # closed twice: still only the gate's stops
        busy = FakeSystem(root, sha, load=3.)
        assert V.queue(options(root), busy, results=results(), stage=dev_stage()) == 8 and busy.argvs == []
    stops = [he.read_json(V.v3_dir(root)/n) for n in ('queue_stop.json', 'queue_stop_0.json')]
    assert all(x['kind'] == 'gate' and 'gate closed' in x['reason'] and x['launched'] == [] for x in stops) and not V.cell_dir(root, 'ii').exists()
    again = FakeSystem(root, sha)
    assert V.queue(options(root), again, results=results(), stage=dev_stage()) == 0 and he.read_json(V.v3_dir(root)/'queue_end.json')
    restart = he.read_json(V.v3_dir(root)/'restart.json')
    assert restart['batch_sha256'] == V.sha_file(V.v3_dir(root)/'batch.json') and restart['options'] == dict(wait_minutes=1, gap_seconds=15, min_host_gib=15.)     # revision 3
    assert V.queue(options(root), FakeSystem(root, sha), results=results(), stage=dev_stage()) == 8                 # used: a new root
    assert 'were launched in this root before' in he.read_json(V.v3_dir(root)/'queue_stop_1.json')['reason']


@pytest.mark.parametrize('first', ['batch_of_another_code', 'a_not_passed', 'exception'])
def test_a_stop_that_is_not_the_machine_gates_goes_to_bz_and_the_queue_never_starts_again(monkeypatch, first):
    """Revision 3 (the lead's ruling B2): a refusal at the start (here an A segment not passed, or a batch.json of another code after a gate stop) or an exception of the queue is not
    started again with the same start string, even once the cause is gone: the next start is refused before anything (to BZ)."""
    root, sha = queue_world(monkeypatch, verdicts={4: S.OPEN} if first == 'a_not_passed' else None)
    if first == 'batch_of_another_code':
        assert V.queue(options(root), FakeSystem(root, sha, load=3.), results=results(), stage=dev_stage()) == 8                # a gate stop first: batch.json written
        batch = he.read_json(V.v3_dir(root)/'batch.json')
        put(V.v3_dir(root)/'batch.json', dict(batch, code_sha='d'*40))
        assert V.queue(options(root), FakeSystem(root, sha), results=results(), stage=dev_stage()) == 8
        assert 'v3/batch.json is at' in he.read_json(V.v3_dir(root)/'queue_stop_0.json')['reason']
        put(V.v3_dir(root)/'batch.json', batch)                                       # the cause gone
    else:
        system = FakeSystem(root, sha, checkout_raises=RuntimeError('git is gone') if first == 'exception' else None)
        assert V.queue(options(root), system, results=results(), stage=dev_stage()) == 8
        assert he.read_json(V.v3_dir(root)/'queue_stop.json')['kind'] == ('exception' if first == 'exception' else 'stop')
        a_segment_records(root, sha)                                                    # the cause gone
    later = FakeSystem(root, sha)
    assert V.queue(options(root), later, results=results(), stage=dev_stage()) == 8 and later.argvs == []
    last = max(V.v3_dir(root).glob('queue_stop*.json'), key=lambda q: -1 if q.stem == 'queue_stop' else int(q.stem.rsplit('_', 1)[1]))      # numbered: the newest the largest
    assert 'other than the machine gate' in he.read_json(last)['reason'] and not V.cell_dir(root, 'ii').exists() and not (V.v3_dir(root)/'steps').exists()


def test_the_queues_start_string_runs_from_the_smokes_checkout():
    root = Path('/home/rmcfz/seq_vp_tag_results/he_smoke_v3')                            # only strings
    a = V.queue_launch(root)
    assert a[:2] == ['systemd-run', '--user'] and f'--unit={V.QUEUE_UNIT}' in a and '--wait' not in a and f'--working-directory={root}/code' in a
    assert a[-6:] == [V.PY, '-B', f'{root}/code/{V.SCRIPT}', 'queue', '--root', str(root)]
    assert f'--setenv=PYTHONPATH={V.core.FROZEN}:{root}/code' in a and '--setenv=CUDA_VISIBLE_DEVICES=' in a
    assert V.QUEUE_UNIT == V._pins().DEV_PARENT_UNITS[0]


# ---- the PQ items and the evaluation entry's PQ functions ------------------------------------------------------------------------------------------
def pq_outputs(world, segment):
    """PQ's out unit's records on one segment as pqh_units writes them (synthetic arrays on the grid)."""
    import pq_core
    small, big = V.pq_dir(world.root, segment), V.pq_big(world.root, segment)
    cell = V.cell_dir(world.root, segment)
    line = he.read_json(cell/'output'/'line.json')
    shape = W.SHAPE
    (big/PAIR).mkdir(parents=True)
    np.savez(big/PAIR/V.PQ_NPZ, q=np.zeros(shape, np.float32), q4=np.zeros(shape, np.float32))
    np.savez(big/PAIR/V.BASELINES, **{k: np.zeros(shape, np.float32) for k in pq_core.BASELINE_KEYS})
    put(small/PAIR/V.PQ_JSON, dict(line_status=line['line_status'], hm=dict(rebuilt_equals_final=True, labels_checked_against_run_json=True),
                                   sampling=dict(d_sha256=line['output']['disp']['d_sha256']), numbers=dict(S35_hat=.9),
                                   inputs={side: {k: '7'*64 for k in ('probs36_sha256', 'structure_groups_sha256', 'provenance_sha256')} for side in ('fixed', 'moving')},
                                   method_files={'t88/field_0600.npz': dict(path=str(cell/'t88'/'field_0600.npz'), sha256=V.sha_file(cell/'t88'/'field_0600.npz'))}))
    put(small/V.OUT_RECORD, dict(complete=True, controls={k: True for k in ('hm1', 'hm2', 'hm3', 'hm4', 'hm5', 'hm6')},
                                          checks=dict(code=dict(clean=True, checkout=str(world.root/'code'), commit=world.sha))))
    put(small/V.OPENED_OUT, dict(m6=True, breaches=[], pair_members={str(world.stage/PAIR/'pair.npz'): ['fixed', 'initial_u', 'moving']}, files=[dict(path=p) for p in sorted(V.sealed_paths())]))
    files = [small/PAIR/V.PQ_JSON, big/PAIR/V.PQ_NPZ, big/PAIR/V.BASELINES, small/V.OUT_RECORD, small/V.OPENED_OUT]
    (small/V.PQ_MANIFEST).write_text(''.join(f'{V.sha_file(f)}  {f}\n' for f in sorted(files)))
    unit = V._pins().DEV_OUT_UNITS[segment]
    put(world.root/'pq'/'units'/f'{unit}_20261114T200000Z'/'unit_reading.json', dict(status=None, returncode=0))
    launched = V.v3_dir(world.root)/'steps'/f"{ {'ii': 'ii-pq', 'iv-b': 'ivb-pq', 'iv-c': 'ivc-pq'}[segment] }.launched.json"
    put(launched, dict(before=dict(cell=V.tree_rows(cell))))


def review_records(world, segment):
    """The review's records of a finalized consistent copy (enough for PQ's items: line status and state hash)."""
    cell = V.cell_dir(world.root, segment)
    put(cell/'review'/'review.json', dict(line_status=he.HX, state_hash=world.target, finalized=True))


def test_v5_the_pq_unit_on_ii_and_its_counterexamples(world):
    review_records(world, 'ii')
    pq_outputs(world, 'ii')
    item = V.v5_pq_ii(world.root, world.sha)
    assert item['verdict'] == S.PASS, item
    small = V.pq_dir(world.root, 'ii')
    opened = he.read_json(small/V.OPENED_OUT)
    for name, bad in (('m6', dict(opened, pair_members={str(world.stage/PAIR/'pair.npz'): ['fixed', 'fixed_seg35', 'moving']})),       # a label member of pair.npz read
                      ('sealed_all_four', dict(opened, files=opened['files'][1:])),                                                  # C22.H8 (ii) 修订 7d (review I1): a list missing
                      ('sealed_no_other_file_of_their_directories', dict(opened, files=opened['files']+[dict(path=str(Path(opened['files'][0]['path']).parent/'train_subjects.txt'))]))):
        put(small/V.OPENED_OUT, bad)
        item = V.v5_pq_ii(world.root, world.sha)
        assert item['verdict'] == S.OPEN and item['values']['checks'][name] is False, name
    put(small/V.OPENED_OUT, opened)
    manifest = (small/V.PQ_MANIFEST).read_text()
    (small/V.PQ_MANIFEST).write_text('0'*64+manifest[64:])                                                                       # a line whose sha256 is not its file's
    item = V.v5_pq_ii(world.root, world.sha)
    assert item['verdict'] == S.OPEN and item['values']['checks']['manifest'] is False
    (small/V.PQ_MANIFEST).write_text(manifest)
    assert V.v5_pq_ii(world.root, world.sha)['verdict'] == S.PASS
    cell = V.cell_dir(world.root, 'ii')
    (cell/'review'/'review.json').write_text(json.dumps(dict(line_status=he.HX, state_hash=world.target, finalized=True, extra=1)))      # D114 (a): a copy's record changed after PQ
    item = V.v5_pq_ii(world.root, world.sha)
    assert item['verdict'] == S.OPEN and item['values']['checks']['d114_cell_unchanged'] is False


def test_v6_the_evaluation_entrys_pq_functions_read_the_development_outputs(world):
    review_records(world, 'ii')
    pq_outputs(world, 'ii')
    out, problems = V.entry_reading(world.root, 'ii')
    assert problems == [] and out['state'] == 'present' and out['ev1']['passed'] is True and len(set(out['ev2'].values())) == 1
    assert out['arrays']['q'] == list(W.SHAPE)
    assert set(out['input_lock_keys']) == {f'inputs.{side}.{k}' for side in ('fixed', 'moving') for k in ('probs36_sha256', 'structure_groups_sha256', 'provenance_sha256')} | {
        "method_files['t88/field_0600.npz'].sha256"}
    small = V.pq_dir(world.root, 'ii')
    doc = he.read_json(small/PAIR/V.PQ_JSON)

    def rewrite(document):
        """pq.json changed and PQ's list rewritten to name it (Ev1 holds): only what is read past Ev1 can differ."""
        put(small/PAIR/V.PQ_JSON, document)
        lines = [ln.split('  ', 1) for ln in (small/V.PQ_MANIFEST).read_text().splitlines()]
        (small/V.PQ_MANIFEST).write_text(''.join(f'{V.sha_file(p) if p == str(small/PAIR/V.PQ_JSON) else h}  {p}\n' for h, p in lines))
    rewrite(dict(doc, sampling=dict(d_sha256='0'*64)))                                                     # Ev2 alone (review I3)
    out, problems = V.entry_reading(world.root, 'ii')
    assert out['ev1']['passed'] is True and len(problems) == 1 and problems[0].startswith('Ev2'), problems
    rewrite(dict(doc, inputs=dict(doc['inputs'], moving=dict(doc['inputs']['moving'], provenance_sha256=None))))     # a key of the input-lock check not a sha256 (review minor 2)
    out, problems = V.entry_reading(world.root, 'ii')
    assert len(problems) == 1 and 'input-lock' in problems[0], problems
    rewrite(doc)
    assert V.entry_reading(world.root, 'ii')[1] == []
    (small/PAIR/V.PQ_JSON).write_text(json.dumps(dict(doc, sampling=dict(d_sha256='0'*64))))                 # the list not rewritten: Ev1 (its bytes) and Ev2
    out, problems = V.entry_reading(world.root, 'ii')
    assert problems


def test_the_three_spellings_of_pqs_roots_on_the_hold_out(monkeypatch):
    monkeypatch.setattr(hc, 'HOLDOUT_ROOT', Path('/home/rmcfz/seq_vp_tag_results/holdout_B_v1'))                  # the real constants (the sandbox moved them); nothing is read or written there
    monkeypatch.setattr(hc, 'BIG_ROOT', Path('/mnt/d/seq_vp_tag_holdout_B_v1'))
    record, problems = V.mnt_d_spellings()
    assert problems == [] and record['arm_table']['big'] == record['pqh_layout']['big'] == record['holdout_common']['big'] == '/mnt/d/seq_vp_tag_holdout_B_v1/pq'.replace(
        '/mnt/d/seq_vp_tag_holdout_B_v1', str(hc.REAL_BIG))
    monkeypatch.setattr(hc, 'BIG_ROOT', Path('/mnt/d/seq_vp_tag_holdout_B_v2'))                                     # PQH's layout spelled from another root
    _, problems = V.mnt_d_spellings()
    assert problems and 'big root' in problems[0]


def test_v3_and_v7_read_the_pq_units_records(world):
    pins = V._pins()
    pq = world.root/'pq'
    subjects = list(pins.DEV_SMOKE_SUBJECTS)
    sealed = sorted(V.sealed_paths())
    put(pq/'members'/V.MEMBERS_RECORD, dict(m5_passed=True, subjects={s: dict(m5=True) for s in subjects}, dev_reference=dict(equal=True),
                                                      checks=dict(code=dict(clean=True, checkout=str(world.root/'code'), commit=world.sha), imports=dict(problems=[]))))
    put(pq/'members'/V.OPENED_MEMBERS, dict(m6=True, breaches=[], files=[dict(path=p) for p in sealed]))
    (pq/'pq_members_dev.lock').write_text('{}\n')
    put(pq/'units'/f"{pins.DEV_UNITS['members']}_20261114T200000Z"/'unit_reading.json', dict(status=None, returncode=0))
    item = V.v3_members(world.root, world.sha)
    assert item['verdict'] == S.PASS, item
    put(pq/'units'/f"{pins.DEV_UNITS['members']}_20261114T210000Z"/'unit_reading.json', dict(status='unit_limit', returncode=30))     # the newest reading counts
    assert V.v3_members(world.root, world.sha)['values']['checks']['unit'] is False
    rows = {f'{n}:{k}': dict(equal=True, file_listed=True) for n, k in [('pq.npz', 'q'), ('pq.npz', 'q4')]+[('baselines.npz', k) for k in range(8)]}
    put(pq/'p3_replay'/'p3_replay.json', dict(reading='逐位相同', rows=rows, spy_m6=True, checks=dict(code=dict(clean=True, checkout=str(world.root/'code'), commit=world.sha))))
    put(pq/'units'/f"{pins.DEV_UNITS['p3_replay']}_20261114T200000Z"/'unit_reading.json', dict(status=None, returncode=0))
    assert V.v7_replay(world.root, world.sha)['verdict'] == S.PASS
    rows['pq.npz:q'] = dict(equal=False, file_listed=True)
    (pq/'p3_replay'/'p3_replay.json').unlink()
    put(pq/'p3_replay'/'p3_replay.json', dict(reading='不过（待查）', rows=rows, spy_m6=True, checks=dict(code=dict(clean=True, checkout=str(world.root/'code'), commit=world.sha))))
    assert V.v7_replay(world.root, world.sha)['verdict'] == S.OPEN


def test_the_record_names_are_pqs():
    import pqh_inputs as hin
    import pqh_units as hu
    assert (V.OUT_RECORD, V.PQ_MANIFEST, V.PQ_JSON, V.PQ_NPZ, V.BASELINES) == (hu.OUT_RECORD, hu.MANIFEST, hu.PQ_JSON, hu.PQ_NPZ, hu.BASELINES) and V.MEMBERS_RECORD == hin.MEMBERS_RECORD
    layout = hin.dev_layout(W.ROOT, smoke_root=Path('/x/he_smoke_v3'), segment='iv-b')
    assert hin.run_dirs(layout, 'out', 'first_run') == (V.pq_dir(Path('/x/he_smoke_v3'), 'iv-b'), V.pq_big(Path('/x/he_smoke_v3'), 'iv-b'))
    assert layout.dev_cell == V.cell_dir(Path('/x/he_smoke_v3'), 'iv-b') and V._pins().dev_cell_rel('ii') == 'ii/a/0438-0439'


# ---- hb_review's development policy keeps the smoke's checkout ---------------------------------------------------------------------------------------
def test_the_development_review_keeps_its_own_code_tree_under_the_results_root(monkeypatch):
    """SM3: the smoke's checkout <results>/he_smoke_v3/code lies under the results root.  Before, the development products rule forbade it -- the review's admission reads the code tree's
    sealed lists and allowlist under the spy, so (ii) would have stopped at its first read (the tests' code tree is the worktree, outside the results root, and never saw it).  The reviews under
    its docs/, a result document, the A arm and the other segments stay forbidden."""
    root = smoke_root()
    seg, code = root/'ii', root/'code'
    monkeypatch.setattr(hb, 'REPO', W.ROOT)
    before = review.dev_policy(seg, dev_stage()/PAIR)
    assert before.why(code/'splits'/'bayesreg_v1'/'holdout_B_subjects.txt') and before.why(code/'configs'/'bayesreg'/'dev_subjects.json')       # the code tree a development product
    monkeypatch.setattr(hb, 'REPO', code)                                                  # the review runs from the smoke's checkout
    pol = review.dev_policy(seg, dev_stage()/PAIR)
    for p in (code/'splits'/'bayesreg_v1'/'holdout_B_subjects.txt', code/'configs'/'bayesreg'/'dev_subjects.json', code/'experiments'/'he'/'run_hx.py', seg/'a'/S.SHORT/'output'/'final.npz'):
        assert pol.why(p) is None, p
    for p in (code/'docs'/'bayesreg'/'reviews'/'x.md', code/'runs'/'x'/'result.md', root/'a'/S.SHORT/'output'/'final.npz', root/'iv-a'/'a'/S.SHORT/'output'/'final.npz',
              results()/'he_smoke_v2'/'a'/'x.json', results()/'pq_v1'/'runs'/'x.npz', code/'evaluation.json'):
        assert pol.why(p), p


# ---- the injection stays in the smoke tool -----------------------------------------------------------------------------------------------------------
def test_no_production_module_imports_the_smoke_tool():
    """C22.H8 (iv) 修订 7b: the injection is the smoke tool's own code; nothing of bayesreg/ or experiments/ imports smoke_v3 (its own file aside), so no hold-out process can reach it."""
    found = []
    for path in sorted(list((W.ROOT/'bayesreg').rglob('*.py'))+list((W.ROOT/'experiments').rglob('*.py'))):
        if path.name == 'smoke_v3.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            names += [a.name for a in node.names] if isinstance(node, ast.ImportFrom) else []
            if any(n == 'smoke_v3' or n.endswith('.smoke_v3') for n in names):
                found.append(str(path))
        if 'smoke_v3' in path.read_text(encoding='utf-8') and 'import_module' in path.read_text(encoding='utf-8'):
            found.append(f'{path} (import_module)')
    assert found == []


def test_the_dry_run_writes_nothing_and_imports_no_torch():
    """In its own process, systemd-run, systemctl, nvidia-smi and journalctl a PATH stand-in that exits 99 (revision 2, review minor 6): the dry run on a root that was never written (not the
    real he_smoke_v3) reports the root's pin and the A segment that has not run, writes nothing, imports no torch."""
    import os
    import tempfile
    root = Path(f'/home/rmcfz/seq_vp_tag_results/he_smoke_v3_never_written_{os.getpid()}')
    assert not root.exists()
    shim = Path(tempfile.mkdtemp(prefix='sm3shim_'))
    try:
        for name in ('systemd-run', 'systemctl', 'nvidia-smi', 'journalctl'):
            (shim/name).write_text('#!/bin/sh\nexit 99\n')
            (shim/name).chmod(0o755)
        env = dict(os.environ, PATH=f"{shim}:{os.environ.get('PATH', '')}")
        done = subprocess.run([V.PY, '-B', str(V.SMOKE_V3), '--dry-run', '--root', str(root)], capture_output=True, text=True, cwd=str(W.ROOT), timeout=600, env=env)
    finally:
        shutil.rmtree(shim, ignore_errors=True)
    assert 'torch imported by this dry run: False' in done.stdout, done.stdout[-2000:]+done.stderr[-2000:]
    assert not root.exists() and done.returncode == 8 and 'the A segment has not run' in done.stdout and "PQ's dev mode is pinned" in done.stdout


class AlternatingMemory(FakeSystem):
    """MemAvailable 12 GiB when the gate asks, 3 GiB when the queue reads it again at launch (one gate check, then one read at launch, per step)."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.calls = 0

    def mem_available_gib(self):
        self.calls += 1
        return 12. if self.calls % 2 else 3.


def test_v13_judges_the_memory_the_gate_opened_on(monkeypatch):
    """Revision 2 (review minor 4, I3), revision 3 (diff review 2): V13 judges the MemAvailable each step's gate opened on (launched.json mem_available_gib_at_gate) against the step's gate;
    the value read again at launch is described only."""
    root, sha = queue_world(monkeypatch)
    assert V.queue(options(root), AlternatingMemory(root, sha), results=results(), stage=dev_stage()) == 0
    launched = he.read_json(V.v3_dir(root)/'steps'/'ivb-review.launched.json')
    assert launched['mem_available_gib_at_gate'] == 12. and launched['mem_available_gib_at_launch'] == 3. and launched['gate_mem_available_gib'] == 10.
    item = V.v13_units(root)
    assert item['verdict'] == S.PASS and item['values']['steps']['ivb-review']['mem_available_gib_at_launch'] == 3.
    path = V.v3_dir(root)/'steps'/'ivb-review.launched.json'
    put(path, dict(launched, mem_available_gib_at_gate=9.5))
    item = V.v13_units(root)
    assert item['verdict'] == S.OPEN and item['values']['checks']['ivb-review_gate'] is False and item['values']['checks']['ii-review_gate'] is True
