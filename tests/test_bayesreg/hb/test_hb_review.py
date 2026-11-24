"""HB item 4: the step-5 review certificate unit (contract C22.H1-2..8; D111, D123; C22.H1-7's test list).  A synthetic hold-out cell written by HE's own writer, the hold-out admission on a simulated
unseal branch (SG's real functions), a stand-in certificate (he_core's comparisons); the third certification in this process through the unit's own child entry (third_child_main: the spy, the cache
check, run_hx.third_cert_main unchanged); configuration C's fallback replaced (run_hx.configc_fallback: no frozen run here).  No unit, no GPU, no label."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from . import _world as W

R, he, hc, hb, review = W.R, W.he, W.hc, W.hb, W.hb_review


def setup(tmp_path, monkeypatch, *, status=None, u=None, tamper=None):
    monkeypatch.setattr(R.core, 'frozen_api', lambda: None)                 # third_cert_main's check that jd_curl is the frozen tree's: the tests' jd_curl is the code tree's (no frozen run here)
    pair, ctx, sim = W.holdout_ctx(tmp_path, monkeypatch)
    u, line = W.write_cell(ctx, status=status or he.HX, u=u)
    if tamper:
        tamper(Path(ctx['out']))
    mode = review.holdout_mode(ctx['code_head'])
    W.manifest(mode, [pair])
    return pair, ctx, u, mode


def third_in_process(mode, certification):
    """The third certification's child, in this process (its own process in production: the guard's cache is empty there; reset here, as PQ's exception (13))."""
    calls = []

    def third(ctx):
        W.XW.sealed_module._reset_cache()
        rc = review.third_child_main(ctx, mode, launcher=object(), certification=certification)
        calls.append(rc)
        return rc
    third.calls = calls
    return third


def quiet_child(rc, calls):
    """A third certification's child that changes nothing and returns rc, with the record a child leaves: opened_files_third_<k>.json, a passing report (HB review I2: the parent reads it)."""
    def third(ctx):
        calls.append(rc)
        rev = Path(ctx['out'])/'review'
        hb.write_once(hb.numbered(rev, 'opened_files_third'), dict(passed=True, breaches=[], rc=rc))
        return rc
    return third


def fake_configc(result):
    """run_hx.configc_fallback's stand-in: None -> no candidate (fallback.json with the error); 'pass' -> a field of the grid certified by the chain's own certify."""
    def configc(ctx, pin, *, launcher, certify, config, suffix=''):
        d = Path(ctx['out'])/f'fallback_configc{suffix}'
        d.mkdir(parents=False, exist_ok=False)
        if result is None:
            rec = dict(kind='configC', error='test: no certified state of configuration C', state_hash=None, certificate=None)
            he.write_json(d/'fallback.json', rec)
            return None, None, rec
        u2 = W.field(pin.grid, scale=.1, seed=5)
        doc = he.certificate_document(certify(u2, pin.grid), config, process='test configuration C', array_sha256_of_u=he.array_sha256(u2))
        rec = dict(kind='configC', state_hash=he.array_sha256(u2), certificate=doc, run_dir=str(d/'run'))
        he.write_json(d/'fallback.json', rec)
        return u2, doc, rec
    return configc


def run(ctx, mode, certify, third=None, **kw):
    rc = review.review_pair(ctx, mode, batch='test-batch', certification=certify.pair(), third=third, probe=W.probe, **kw)
    for p in sorted((Path(ctx['out'])/'review').glob('review_not_finalized*.json')):
        doc = he.read_json(p)
        print(p.name, doc['kind'], doc['reason'][:500])                     # shown when a test fails
    for p in sorted((Path(ctx['out'])/'review').glob('opened_files*.json')):
        doc = he.read_json(p)
        print(p.name, {k: doc.get(k) for k in ('loader_calls', 'reads_seen', 'reads_at_the_class', 'original_access', 'forbidden_access', 'evaluation_records_opened', 'breaches')})
    return rc


def records(ctx):
    rev = Path(ctx['out'])/'review'
    return he.read_json(rev/review.REVIEW_RECORD), he.read_json(hb.terminal_path(ctx['out'])), sorted(p.name for p in rev.glob('review_not_finalized*.json'))


def test_consistent_review_finalizes_the_pair(tmp_path, monkeypatch):
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    certify = W.Certify()
    assert run(ctx, mode, certify) == 0
    doc, term, stops = records(ctx)
    assert doc['line_status'] == term['line_status'] == he.HX and doc['consistent'] is True and doc['differences'] == {} and len(doc['rounds']) == 1 and stops == []
    assert doc['decision']['state_hash'] == he.array_sha256(u) and doc['state_hash'] == he.array_sha256(u) and term['output_dir'] == 'output' and term['review_record']['path'] == 'review/review.json'
    assert doc['t88_repair']['t88/field_0600.npz'] and doc['output']['output/final.npz'] == hb.sha(Path(ctx['out'])/'output/final.npz') and doc['context_settings_are_pinned'] is True
    assert mode.lock(pair).is_file() and certify.calls == [he.array_sha256(u)]
    spy = he.read_json(Path(ctx['out'])/'review'/review.OPENED)
    assert spy['passed'] is True and spy['breaches'] == [] and spy['n_sealed_list_opens'] >= 4 and len(spy['loader_calls']) == 1          # C23.6-5: the four lists read under the spy, recorded
    assert run(ctx, mode, certify) != 0 and he.read_json(Path(ctx['out'])/'review'/review.REVIEW_RECORD) == doc                          # once


def test_a_mismatch_then_a_passing_third_certificate_takes_two_rounds(tmp_path, monkeypatch):
    """C22.H8 (iv-a)'s expected loop (run_hx.py:868-885): round 1 differs -> review_mismatch -> the third certification passes -> round 2 differs again but is resolved -> finalized, 2 rounds."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    h = he.array_sha256(u)
    third = third_in_process(mode, W.Certify().pair())
    assert run(ctx, mode, W.Certify(perturb={h}), third) == 0
    doc, term, _ = records(ctx)
    assert third.calls == [0] and len(doc['rounds']) == 2 and doc['rounds'][0]['consistent'] is False and doc['rounds'][1]['third_certificate']['passed'] is True
    assert doc['line_status'] == term['line_status'] == he.HX and (Path(ctx['out'])/'output'/'review_mismatch_0.json').is_file() and doc['third_certificate']['output_state_hash'] == h
    assert he.read_json(Path(ctx['out'])/'review'/'opened_files_third.json')['passed'] is True


@pytest.mark.parametrize('configc, want', [('pass', 'hx_fallback_configC'), (None, 'hx_fallback_identity')])
def test_a_failed_third_certificate_goes_to_the_next_fallback(tmp_path, monkeypatch, configc, want):
    """(iv-b), (iv-c): the third certification fails -> output_superseded_0 -> configuration C (here its stand-in) or the identity -> the new output reviewed again (consistent) -> finalized."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    h = he.array_sha256(u)
    monkeypatch.setattr(R, 'configc_fallback', fake_configc(configc))
    third = third_in_process(mode, W.Certify(fail={h}).pair())
    assert run(ctx, mode, W.Certify(perturb={h}), third) == 0
    doc, term, _ = records(ctx)
    out = Path(ctx['out'])
    assert doc['line_status'] == term['line_status'] == want and len(doc['rounds']) == 2 and doc['rounds'][1]['consistent'] is True and doc['rounds'][0]['state_hash'] == h != doc['state_hash']
    assert (out/'output_superseded_0'/'final.npz').is_file() and (out/'output_superseded_0'/'third_certificate.json').is_file() and 'output_superseded_0' in doc['evidence']
    line = he.read_json(out/'output'/'line.json')
    assert line['third_certificate']['passed'] is False and line['superseded'] == [str(out/'output_superseded_0')] and doc['fallback']['kind'] == ('configC' if configc else 'identity')


def test_the_identity_not_passing_in_the_chain_is_an_implementation_error(tmp_path, monkeypatch):
    """C22.5-6 (a) in the (d) chain, the second form: the identity's certificate fails after the old output was renamed: no output, no output_failed -> review_failed_0.json, hx_implementation_error, STOP."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    h = he.array_sha256(u)
    monkeypatch.setattr(R, 'configc_fallback', fake_configc(None))
    third = third_in_process(mode, W.Certify(fail={h}, fail_zero=True).pair())
    assert run(ctx, mode, W.Certify(perturb={h}), third) == review.STOP_EXIT['implementation']
    doc, term, _ = records(ctx)
    rev = Path(ctx['out'])/'review'
    assert third.calls == [review.THIRD_ERROR_EXIT] and doc['line_status'] == term['line_status'] == he.IMPLEMENTATION and doc['finalized'] is False and term['output_dir'] is None
    assert (rev/'review_failed_0.json').is_file() and 'IdentityNotCertified' in he.read_json(rev/'third_cert_error.json')['exception']
    assert 'output_superseded_0' in doc['evidence'] and any(k.startswith('fallback_identity_after_third_0') for k in doc['evidence'])


def test_the_identity_outputs_failed_third_certificate_withdraws_it(tmp_path, monkeypatch):
    """C22.5-6 (a), the first form: the output is the identity (hx_fallback_identity); its third certificate fails; no fallback after it: output_failed_0 (+ .json) -> hx_implementation_error, STOP."""
    g = W.grid()
    pair, ctx, u, mode = setup(tmp_path, monkeypatch, status=he.FALLBACK_ID, u=np.zeros(g.shape+(3,)))
    third = third_in_process(mode, W.Certify(fail_zero=True).pair())
    assert run(ctx, mode, W.Certify(perturb={he.array_sha256(u)}), third) == review.STOP_EXIT['implementation']
    doc, term, _ = records(ctx)
    out = Path(ctx['out'])
    assert third.calls == [8] and (out/'output_failed_0').is_dir() and (out/'output_failed_0.json').is_file() and term['line_status'] == he.IMPLEMENTATION and 'output_failed_0' in doc['evidence']


def test_three_rounds_without_a_resolution_stop_without_finalizing(tmp_path, monkeypatch):
    """D111 (a): three rounds; the line status unchanged, no terminal record; review_not_finalized (rounds).  HB review m5: no third certification after the last round (its output would not be
    reviewed): two children for three rounds."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    calls = []
    assert run(ctx, mode, W.Certify(perturb={he.array_sha256(u)}), quiet_child(1, calls)) == review.STOP_EXIT['not_finalized']
    doc, term, stops = records(ctx)
    assert doc is None and term is None and len(stops) == 1 and len(calls) == 2
    stop = he.read_json(Path(ctx['out'])/'review'/stops[0])
    assert stop['kind'] == 'rounds' and len(stop['rounds']) == 3 and [r.get('third_returncode') for r in stop['rounds']] == [1, 1, None]
    assert len(list((Path(ctx['out'])/'output').glob('review_mismatch_*.json'))) == 3
    assert he.read_json(Path(ctx['out'])/'output'/'line.json')['line_status'] == he.HX


def change_cert_json(out):
    path = out/'output'/'cert.json'
    path.write_text(path.read_text().replace('"process": "test: the decision certificate"', '"process": "test: the decision certificate, changed"'))


def change_disp(out):
    """disp.npz with d changed by one ulp, line.json's output.disp written for it (so that only the round trip can see it)."""
    with np.load(out/'output'/'disp.npz') as z:
        d, meta = np.array(z['d']), np.array(z['meta'])
    d[0, 2, 2, 2] = np.nextafter(d[0, 2, 2, 2], 1.)
    np.savez(out/'output'/'disp.npz', d=d, meta=meta)
    line = json.loads((out/'output'/'line.json').read_text())
    line['output']['disp'] = dict(file='disp.npz', sha256=hb.sha(out/'output'/'disp.npz'), d_sha256=he.array_sha256(d))
    (out/'output'/'line.json').write_text(json.dumps(line))


@pytest.mark.parametrize('tamper, kind', [(change_cert_json, 'line_json'), (change_disp, 'roundtrip')])
def test_the_units_own_checks_stop_without_changing_the_line_status(tmp_path, monkeypatch, tamper, kind):
    """C22.H1-2: line.json's output.* against the files; the C16 round trip.  Not finalized, line_status unchanged, STOP (C22.5-6: not an implementation error until BZ withdraws the output)."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch, tamper=tamper)
    assert run(ctx, mode, W.Certify()) == review.STOP_EXIT['not_finalized']
    doc, term, stops = records(ctx)
    assert doc is None and term is None and he.read_json(Path(ctx['out'])/'review'/stops[0])['kind'] == kind


def test_a_file_changed_after_the_input_manifest_is_refused_before_the_pairs_lock(tmp_path, monkeypatch):
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    he.write_json(Path(ctx['out'])/'t88'/'run.json.x', {})
    p = Path(ctx['out'])/'t88'/'run.json'
    p.write_text(p.read_text()+' ')
    certify = W.Certify()
    assert run(ctx, mode, certify) == review.STOP_EXIT['not_finalized']
    _, term, stops = records(ctx)
    assert term is None and he.read_json(Path(ctx['out'])/'review'/stops[0])['kind'] == 'input_manifest' and not mode.lock(pair).exists() and certify.calls == []


def test_the_contexts_code_head_must_be_the_batchs(tmp_path, monkeypatch):
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    mode.batch_sha = 'f'*40
    assert run(ctx, mode, W.Certify()) == review.STOP_EXIT['not_finalized']
    assert he.read_json(Path(ctx['out'])/'review'/records(ctx)[2][0])['kind'] == 'code_head' and not mode.lock(pair).exists()


def test_a_sealed_list_cache_filled_before_the_spy_stops_the_unit(tmp_path, monkeypatch):
    """C23.6-5, Hm6: the cache is checked empty before the spy is installed; not empty -> STOP to BZ, nothing of the pair read."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    hc.sealed_lists()
    certify = W.Certify()
    assert run(ctx, mode, certify) == review.STOP_EXIT['sealed_cache'] and certify.calls == []
    W.XW.sealed_module._reset_cache()
    hc.sealed_lists()                                   # HB review m14 (ii): the cache filled again before the child's own check
    assert review.third_child_main(ctx, mode, launcher=object(), certification=certify.pair()) == review.STOP_EXIT['sealed_cache']


@pytest.mark.parametrize('evidence, code', [(True, 'killed'), (False, 'rejudge')])
def test_a_killed_third_child_is_d123_and_no_round(tmp_path, monkeypatch, evidence, code):
    """C22.H1-4 (修订 7b, 7c): killed with OOM evidence of this unit -> D123 (b), STOP; without -> needs_lead_rejudgment, STOP; neither counts as a round of D111."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    ev = dict(items=[dict(source='memory.events', oom_kill=1)] if evidence else [], events=dict(oom_kill=1 if evidence else 0))
    assert run(ctx, mode, W.Certify(perturb={he.array_sha256(u)}), lambda c: -9, kill_evidence=lambda start: ev) == review.STOP_EXIT[code]
    stop = he.read_json(Path(ctx['out'])/'review'/records(ctx)[2][0])
    assert stop['kind'] == ('third_child_oom' if evidence else 'third_child_signal') and len(stop['rounds']) == 1 and stop['extra']['evidence'] == ev
    assert he.read_json(hb.terminal_path(ctx['out'])) is None


def test_the_policy_names_and_roots(tmp_path, monkeypatch):
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    pol = review.holdout_policy()
    out = Path(ctx['out'])
    # HB review m3 (C22.H1-8): the forbidden list only -- labels, evaluation outputs and locks, other arms' outputs, records, locks and batches, development products, H1-H4/T0/training files
    forbidden = [out/'evaluation.json', out/'evaluation_started.json', out/'metrics.json', tmp_path/'data'/'OASIS_OAS1_0423_MR1'/'aligned_seg35.nii.gz', tmp_path/'x'/'labels_seg4.nii.gz',
                 Path(hc.HOLDOUT_ROOT)/'fireants_greedy'/pair/'result.json', Path(hc.HOLDOUT_ROOT)/'work'/'configc'/pair/'output'/'final.npz', Path(hc.BIG_ROOT)/'convexadam_v4seg'/pair/'disp.npz',
                 Path(hc.HOLDOUT_ROOT)/'locks'/f'xb__ants_s__{pair}.lock', Path(hc.HOLDOUT_ROOT)/'locks'/f'hb__configc__{pair}.lock', Path(hc.HOLDOUT_ROOT)/'batches'/'xb_20261110T000000Z'/'batch.json',
                 Path(hc.HOLDOUT_ROOT)/'batches'/'hb_configc_x'/'queue.log', Path(hc.RESULTS)/'pq_v1'/'runs'/'0438-0439'/'pq.npz', Path(he.RESULTS)/'hs_v1'/'runs'/'x'/'field_0600.npz',
                 Path(he.RESULTS)/'seg'/'unet35_v4_ens'/'pred'/'OASIS_OAS1_0438_MR1'/'probs36.npz', Path(hc.HOLDOUT_ROOT)/'eval'/'pairs'/f'{pair}.json', Path(he.RESULTS)/'ev'/'truth_gate.lock',
                 hc.MAIN_REPO/'data'/'three_d'/'stage_H1_v1'/'x'/'pair.npz', hc.MAIN_REPO/'data'/'three_d'/'stage_T0_tune_v1'/'x'/'pair.npz', Path('/mnt/d/data')/'OASIS_OAS1_0001_MR1'/'aligned_norm.nii.gz',
                 out/'x_dev_read.lock', W.ROOT/'runs'/'20261107-pq'/'result.md', W.ROOT/'docs'/'bayesreg'/'reviews'/'x.md',
                 hc.MAIN_REPO/'data'/'three_d'/'stage_L2R_val_v1'/'x'/'pair.npz', hc.MAIN_REPO/'data'/'OASIS_OAS1_0001_MR1'/'x.npz', W.ROOT/'data'/'x']     # n5: the whole data/ in the hold-out mode
    allowed = [out/'output'/'review_mismatch_0.json', out/'output'/'third_certificate.json', out/'review'/'review.json', Path(hc.STAGE_ROOT)/pair/'pair.npz', *map(Path, hc.sealed_list_paths()),
               W.ROOT/'docs/decision_log.md', Path(he.core.FROZEN)/'jd_curl'/'x.py', Path(hc.BIG_ROOT)/'method'/pair/'output'/'final.npz', Path(hc.HOLDOUT_ROOT)/'work'/'method'/'cache'/'jit'/'x',
               Path(hc.HOLDOUT_ROOT)/'locks'/f'hb__method__{pair}.lock', Path(hc.HOLDOUT_ROOT)/'batches'/'hb_review_x'/'batch.json', Path(hc.HOLDOUT_ROOT)/'atropos'/'x'/'provenance.json',
               Path(he.RESULTS)/'seg'/'unet35_v4_ens_holdout_B_v1'/'pred'/'x'/'provenance.json', Path.home()/'.triton'/'cache'/'x', W.ROOT/'experiments'/'he'/'run_hx.py']
    assert [p for p in forbidden if pol.why(p) is None] == []
    assert [(p, pol.why(p)) for p in allowed if pol.why(p) is not None] == []
    dev = review.dev_policy(Path(he.RESULTS)/'he_smoke_v3'/'ii', tmp_path)
    assert dev.why(Path(he.RESULTS)/'he_smoke_v3'/'ii'/'a'/'x'/'output'/'final.npz') is None and dev.why(Path(he.RESULTS)/'he_smoke_v2'/'a'/'x'/'output'/'final.npz')
    assert dev.why(hc.MAIN_REPO/'data'/'three_d'/'stage_L2R_val_v1'/'x'/'pair.npz') is None and dev.why(Path(hc.HOLDOUT_ROOT)/'stage'/pair/'pair.npz')
    third = lambda c: open(Path(c['out'])/'evaluation.json', 'w') and 0       # noqa: E731
    assert run(ctx, mode, W.Certify(perturb={he.array_sha256(u)}), third) == review.STOP_EXIT['spy']
    assert he.read_json(Path(ctx['out'])/'review'/records(ctx)[2][0])['kind'] == 'spy' and not (out/'evaluation.json').exists()


# ---- D123 (b): on along the chain from the failed third certificate's record -----------------------------------------------------------------------
def killed_after_rename(tmp_path, monkeypatch, *, passed=False):
    """A cell as a unit killed after third_cert_main renamed the output leaves it: output_superseded_0 with its third_certificate.json (passed False), a half output/ and a half fallback directory."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    out = Path(ctx['out'])
    g = W.grid()
    doc = he.certificate_document(W.result_of(u, g, passed=passed), W.config(), process='test third', array_sha256_of_u=he.array_sha256(u))
    he.write_json(out/'output'/'third_certificate.json', dict(output_state_hash=he.array_sha256(u), line_status=he.HX, certificate=doc, passed=passed))
    (out/'output').rename(out/'output_superseded_0')
    (out/'output').mkdir()
    (out/'output'/'final.npz').write_bytes(b'half')
    (out/'fallback_configc_after_third_0').mkdir()
    (out/'fallback_configc_after_third_0'/'half.txt').write_text('a killed attempt')
    return pair, ctx, u, mode, out


def test_continue_fallback_goes_on_from_the_record_with_a_new_suffix(tmp_path, monkeypatch):
    pair, ctx, u, mode, out = killed_after_rename(tmp_path, monkeypatch)
    monkeypatch.setattr(R, 'configc_fallback', fake_configc('pass'))
    rec = out/'output_superseded_0'/'third_certificate.json'
    certify = W.Certify()
    got = review.continue_fallback(ctx, k=0, redo=1, record_sha256=hb.sha(rec), launcher=object(), certification=certify.pair())
    assert got['status'] == he.FALLBACK_C and got['suffix'] == '_after_third_0_redo1' and (out/'output_killed_0'/'final.npz').read_bytes() == b'half'
    assert (out/'fallback_configc_after_third_0'/'half.txt').is_file() and (out/'fallback_configc_after_third_0_redo1'/'fallback.json').is_file()
    line = he.read_json(out/'output'/'line.json')
    assert line['line_status'] == he.FALLBACK_C and line['continued']['record_sha256'] == hb.sha(rec) and he.array_sha256(u) not in certify.calls     # the old array never certified again


@pytest.mark.parametrize('case', ['passed', 'sha256', 'array'])
def test_continue_fallback_refuses_a_record_that_does_not_hold(tmp_path, monkeypatch, case):
    pair, ctx, u, mode, out = killed_after_rename(tmp_path, monkeypatch, passed=case == 'passed')
    rec = out/'output_superseded_0'/'third_certificate.json'
    sha = hb.sha(rec)
    if case == 'sha256':
        sha = '0'*64
    if case == 'array':
        doc = json.loads(rec.read_text())
        doc['output_state_hash'] = '1'*64
        rec.write_text(json.dumps(doc))
        sha = hb.sha(rec)
    monkeypatch.setattr(R, 'configc_fallback', fake_configc('pass'))
    with pytest.raises(ValueError):
        review.continue_fallback(ctx, k=0, redo=1, record_sha256=sha, launcher=object(), certification=W.Certify().pair())
    assert (out/'output'/'final.npz').read_bytes() == b'half' and not (out/'fallback_configc_after_third_0_redo1').exists()


def test_a_redo_continues_the_chain_and_reviews_the_new_output(tmp_path, monkeypatch):
    """The unit with a redo document (the queue's supplement): the chain continued from the record, then the review of the new output, finalized; the redo's own lock."""
    pair, ctx, u, mode, out = killed_after_rename(tmp_path, monkeypatch)
    monkeypatch.setattr(R, 'configc_fallback', fake_configc(None))
    rec = out/'output_superseded_0'/'third_certificate.json'
    sup = mode.root/'supplements'/f'{pair}_1.json'
    sup.parent.mkdir(parents=True)
    sup.write_text(json.dumps(dict(pairs={pair: dict(files=hb.file_rows(out, ['t88/run.json', 't88/field_0600.npz', 'repair/repair.json', 'repair/repair.npz']))})))
    assert run(ctx, mode, W.Certify(), redo=1, redo_doc={'continue': dict(k=0, record_sha256=hb.sha(rec))}, launcher=object()) == 0
    doc, term, _ = records(ctx)
    assert doc['line_status'] == term['line_status'] == he.FALLBACK_ID and doc['continued']['suffix'] == '_after_third_0_redo1' and mode.lock(pair, True).is_file() and not mode.lock(pair).exists()


def test_the_development_mode_writes_only_under_a_smoke_segment(tmp_path):
    """C22.H1-7 last item, C22.H8 "只在新根里写": a development review runs on a copy under <results>/he_smoke*/<segment>, never elsewhere (and never on he_smoke_v2's own arms: the plan's copies)."""
    results = tmp_path/'results'
    seg = results/'he_smoke_v3'/'ii'
    assert review.dev_root_problem(seg, seg/'a'/'0438-0439', results=results) is None
    assert review.dev_root_problem(results/'he_smoke_v3', results/'he_smoke_v3'/'a'/'0438-0439', results=results)          # the root itself is not a segment
    assert review.dev_root_problem(results/'hs_v1'/'x', results/'hs_v1'/'x'/'c', results=results)
    assert review.dev_root_problem(seg, results/'he_smoke_v3'/'a'/'0438-0439', results=results)                           # the cell must be under the segment
    pol = review.dev_policy(seg, tmp_path/'stage'/'0438-0439')
    assert pol.why(seg/'a'/'0438-0439'/'output'/'final.npz') is None and pol.why(tmp_path/'stage'/'0438-0439'/'pair.npz') is None
    assert pol.why(W.Path('/home/rmcfz/seq_vp_tag_results/he_smoke_v2/a/0438-0439/evaluation.json')) and pol.why(W.Path('/home/rmcfz/seq_vp_tag_results/pq_v1/runs/x.npz'))


def test_the_third_certifications_child_is_this_units_own_entry(tmp_path, monkeypatch):
    """C22.H1-7 (甲): the child is `hb_review.py third-cert` (the spy and the cache check first), never `run_hx.py third-cert`; no timeout of its own; a development review tells it its segment."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr(review.subprocess, 'run', lambda argv, **kw: seen.append((argv, kw)) or W.subprocess.CompletedProcess(argv, 0))
    (Path(ctx['out'])/'review').mkdir(exist_ok=True)
    assert review.ThirdChild(mode, 'b')(ctx) == 0
    argv, kw = seen[0]
    assert argv[2].endswith('experiments/hb/hb_review.py') and argv[3] == 'third-cert' and argv[argv.index('--mode')+1] == 'holdout' and 'timeout' not in kw
    segment = Path(hc.RESULTS)/'he_smoke_v3'/'ii'                       # under the sandbox's roots, not pytest's tmp_path: the child's start string refuses the word pytest (fc.check_launch)
    dev = review.Mode('dev', segment/'review', None, review.dev_policy(segment, tmp_path), segment=segment)
    review.ThirdChild(dev, 'b')(ctx)
    assert seen[1][0][seen[1][0].index('--root')+1] == str(segment)


@pytest.mark.parametrize('which', ['holdout_pair', 'other_stage', 'not_a_dev_pair', 'a_dev_pair'])
def test_the_development_mode_reviews_only_one_of_stage_l2r_val_v1s_19_pairs(tmp_path, monkeypatch, which):
    """runs/20261109-align item 8: a development review checks first that the pair is one of the 19 development pairs (holdout_common.DEV_PAIRS) of stage_L2R_val_v1; nothing is read, nothing locked."""
    pair, ctx, u, _ = setup(tmp_path, monkeypatch)
    seg = tmp_path/'results'/'he_smoke_v3'/'ii'
    mode = review.Mode('dev', seg/'review', None, review.dev_policy(seg, tmp_path/'stage'), segment=seg)
    if which == 'other_stage':                          # HB review m14 (i): ctx['stage_dir'] is the stage directory itself, its name is what is checked
        ctx = dict(ctx, pair=hc.DEV_PAIRS[0], stage_dir=str(tmp_path/'stage_T0_tune_v1'))
    elif which == 'not_a_dev_pair':
        ctx = dict(ctx, pair='OASIS_OAS1_0457_MR1__OASIS_OAS1_0458_MR1', stage_dir=str(tmp_path/review.DEV_STAGE_NAME))
    elif which == 'a_dev_pair':
        ctx = dict(ctx, pair=hc.DEV_PAIRS[18], stage_dir=str(tmp_path/review.DEV_STAGE_NAME))
    assert len(hc.DEV_PAIRS) == 19
    assert run(ctx, mode, W.Certify()) == review.STOP_EXIT['not_finalized']
    rev = Path(ctx['out'])/'review'
    doc = he.read_json(rev/'review_not_finalized.json')
    if which == 'a_dev_pair':                           # past the pair check (it stops later: this cell's records are the hold-out's)
        assert doc['kind'] != 'dev_pair'
        return
    assert doc['kind'] == 'dev_pair' and doc['mode'] == 'dev' and not (rev/review.REVIEW_RECORD).exists() and not list(rev.glob('identity_review*')) and not list(mode.root.glob('*.lock'))


# ---- HB review I2: the third certification's child's firewall records, read every round -----------------------------------------------------------
@pytest.mark.parametrize('case', ['breach_before', 'breach_swallowed', 'cache_not_empty', 'no_report'])
def test_a_firewall_event_of_the_third_child_stops_the_pair_not_a_round(tmp_path, monkeypatch, case):
    """C22.H1-8: the child's spy stopped a read (exit 7), a read the chain swallowed (run_hx.configc_fallback's except Exception: exit 0, the report's passed false), the sealed-list cache not empty in
    the child (exit 27), or no report at all: the pair is not finalized, no further round or child, the batch stops (27 or 28), to BZ."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    h = he.array_sha256(u)
    forbidden = Path(ctx['out'])/'evaluation.json'
    forbidden.write_text('{}')
    calls = []
    if case == 'breach_before':
        real = R.third_cert_main

        def third_cert_main(ctx_, **kw):
            open(forbidden).close()                     # stopped by the child's spy before the third certificate
            return real(ctx_, **kw)
        monkeypatch.setattr(R, 'third_cert_main', third_cert_main)
        third = third_in_process(mode, W.Certify(fail={h}).pair())
    elif case == 'breach_swallowed':
        fake = fake_configc(None)

        def configc(ctx_, pin, **kw):
            try:                                        # as run_hx.configc_fallback: every exception of fallback one is taken as "not usable"
                open(forbidden).close()
            except Exception:                           # noqa: BLE001
                pass
            return fake(ctx_, pin, **kw)
        monkeypatch.setattr(R, 'configc_fallback', configc)
        third = third_in_process(mode, W.Certify(fail={h}).pair())
    elif case == 'cache_not_empty':
        def third(ctx_):
            W.XW.sealed_module._reset_cache()
            hc.sealed_lists()
            rc = review.third_child_main(ctx_, mode, launcher=object(), certification=W.Certify().pair())
            calls.append(rc)
            return rc
    else:
        def third(ctx_):
            calls.append(0)
            return 0
    rc = run(ctx, mode, W.Certify(perturb={h}), third)
    want = 'third_sealed_cache' if case == 'cache_not_empty' else 'third_spy'
    assert rc == review.STOP_EXIT['sealed_cache' if case == 'cache_not_empty' else 'spy']
    doc, term, stops = records(ctx)
    stop = he.read_json(Path(ctx['out'])/'review'/stops[-1])
    assert doc is None and term is None and stop['kind'] == want and len(stop['rounds']) == 1
    if case == 'breach_swallowed':
        report = he.read_json(Path(ctx['out'])/'review'/'opened_files_third.json')
        assert report['rc'] == 0 and report['passed'] is False and report['breaches']          # the child "succeeded": the record caught it
    if case == 'breach_before':
        assert he.read_json(Path(ctx['out'])/'review'/'opened_files_third.json')['rc'] == review.SPY_BREACH_EXIT


@pytest.mark.parametrize('case', ['identity', 'context'])
def test_a_third_child_refused_before_it_read_the_pair_is_not_a_firewall_event(tmp_path, monkeypatch, case):
    """HB diff review n3: the child refused by its identity (exit 8) or its context (exit 2) before it read pair.npz: its spy report is clean and shows no access to the pair; the parent records
    third_child_refused (29, with the exit, the cause and the identity records), not third_spy (28); no round counted, the pair not finalized."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    calls = []

    def third(ctx_):
        with monkeypatch.context() as m:                # only in the child
            if case == 'identity':
                m.setattr(R, 'check_identity', lambda c, out, name: (he.write_json(Path(out)/name, dict(problems=['test'])), ['test: the code changed']))
            else:
                m.setattr(R, 'function_context_problem', lambda c, results=None: 'test: not this context')
            W.XW.sealed_module._reset_cache()
            rc = review.third_child_main(ctx_, mode, launcher=object(), certification=W.Certify().pair())
        calls.append(rc)
        return rc
    rc = run(ctx, mode, W.Certify(perturb={he.array_sha256(u)}), third)
    assert calls == [{'identity': 8, 'context': 2}[case]] and rc == review.STOP_EXIT['third_refused']
    doc, term, stops = records(ctx)
    stop = he.read_json(Path(ctx['out'])/'review'/stops[-1])
    assert doc is None and term is None and stop['kind'] == 'third_child_refused' and stop['code'] == 29 and len(stop['rounds']) == 1
    assert stop['extra']['cause'] == case and stop['extra']['child_exit'] == calls[0] and (case == 'context' or stop['extra']['identity_records'])
    report = he.read_json(Path(ctx['out'])/'review'/'opened_files_third.json')
    assert report['passed'] is False and report['loader_calls'] == [] and report['breaches'] == [] and report['forbidden_access'] == []


def test_refused_before_reading_on_its_own(tmp_path):
    """n3's rule: only a whole, clean report without any access to the pair and a non-zero exit, or THIRD_REFUSED_EXIT without a report; anything else stays the spy's (28).  HB diff review 2 r2:
    the report's `opened` must not hold the original pair.npz (compared resolved; `opened` and `original` are keys of a whole report), and THIRD_ERROR_EXIT's cause is the exception's family."""
    pair = tmp_path/'stage'/'p'/'pair.npz'
    pair.parent.mkdir(parents=True)
    pair.write_bytes(b'x')
    (tmp_path/'link').symlink_to(pair.parent)
    clean = dict(passed=False, loader_calls=[], original_access=[], npz_access=[], n_zip_opens=0, breaches=[], forbidden_access=[], evaluation_records_opened=[], original=str(pair),
                 opened=[str(tmp_path/'context.json')])
    f = review.refused_before_reading
    assert f(review.THIRD_REFUSED_EXIT, None) == 'before_spy' and f(1, None) is None and f(2, None) is None
    assert f(2, clean) == 'context' and f(8, clean) == 'identity' and f(review.THIRD_ERROR_EXIT, clean) == 'exception' and f(5, clean) == 'exit 5'
    assert f(0, clean) is None and f(review.SPY_BREACH_EXIT, clean) is None and f(8, dict(clean, passed=True)) is None
    for key, value in (('breaches', ['x']), ('forbidden_access', ['member seg35']), ('evaluation_records_opened', ['metrics.json']), ('loader_calls', ['load_pair']),
                       ('original_access', ['read fixed']), ('npz_access', ['read fixed']), ('n_zip_opens', 1)):
        assert f(8, dict(clean, **{key: value})) is None, key
    for key in ('npz_access', 'opened', 'original'):
        assert f(8, {k: v for k, v in clean.items() if k != key}) is None, key
    for opened in (str(pair), str(tmp_path/'link'/'pair.npz'), os.path.relpath(pair)):           # r2: the pair's bytes opened (as written, through a link, relative): the spy's
        assert f(review.THIRD_ERROR_EXIT, dict(clean, opened=[str(tmp_path/'x.json'), opened])) is None, opened
    assert f(8, dict(clean, original=None)) is None                                             # not known: fail closed
    # A5a review m10: the child resolves the paths itself (report() writes opened_realpath, original_realpath); the parent compares those, not its own realpath of a relative path
    resolved = dict(clean, opened_realpath=[str(tmp_path/'context.json')], original_realpath=os.path.realpath(pair))
    assert f(8, resolved) == 'identity' and f(8, dict(resolved, opened_realpath=[os.path.realpath(pair)])) is None
    assert f(8, dict(resolved, opened=['pair.npz'], opened_realpath=[str(tmp_path/'elsewhere'/'pair.npz')])) == 'identity'     # a relative path as the child resolved it, not the parent
    error = dict(exception_class='SubjectSealed', exception_mro=['SubjectSealed', 'RuntimeError', 'Exception', 'BaseException', 'object'])
    assert f(review.THIRD_ERROR_EXIT, clean, error) == 'admission'
    assert f(review.THIRD_ERROR_EXIT, clean, dict(exception_class='InputValidationError', exception_mro=['InputValidationError', 'ValueError', 'Exception'])) == 'input'
    assert f(review.THIRD_ERROR_EXIT, clean, dict(exception_class='FileNotFoundError')) is not None
    assert f(review.THIRD_ERROR_EXIT, clean, dict(exception_class='FileNotFoundError', exception_mro=['FileNotFoundError', 'OSError', 'Exception'])) == 'admission'
    assert f(review.THIRD_ERROR_EXIT, clean, dict(exception_class='RuntimeError', exception_mro=['RuntimeError', 'Exception'])) == 'exception'
    assert f(review.THIRD_ERROR_EXIT, clean, dict(refusal_family='input', exception_class='X')) == 'input'
    assert f(8, clean, dict(refusal_family='input')) == 'identity'                              # the family names only THIRD_ERROR_EXIT's cause
    from bayesreg.io.guard import SubjectSealed
    from jd_curl.validation import InputValidationError
    assert review.refusal_family(SubjectSealed('test')) == 'admission' and review.refusal_family(InputValidationError('test')) == 'input'
    assert review.refusal_family(RuntimeError('test')) is None and review.refusal_family(R.IdentityNotCertified('test')) is None


@pytest.mark.parametrize('case', ['admission', 'input', 'bytes_read'])
def test_a_third_child_that_raised_before_the_pair_is_named_by_its_class_and_one_that_read_its_bytes_is_the_spys(tmp_path, monkeypatch, case):
    """HB diff review 2 r2: the child raised in its admission (a subject guard: SubjectSealed) or on a malformed input (InputValidationError) before anything of pair.npz was opened -> STOP 29 with
    the cause admission / input (third_cert_error_<k>.json carries refusal_family and the class's MRO); the child opened pair.npz's bytes (builtins.open, as load_pair's container sha256 does)
    and then raised -> not "before it read pair.npz": STOP 28 (third_spy)."""
    from bayesreg.io.guard import SubjectSealed
    from jd_curl.validation import InputValidationError
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    calls = []

    def admitted(c):
        if case == 'bytes_read':
            with open(R.pair_path(c), 'rb') as stream:
                stream.read(16)
            raise InputValidationError('test: the container sha256 differs from the manifest')
        raise SubjectSealed('test: the subject is sealed') if case == 'admission' else InputValidationError('test: the manifest is malformed')

    def third(ctx_):
        with monkeypatch.context() as m:                # only in the child
            m.setattr(R, 'admitted_images', admitted)
            W.XW.sealed_module._reset_cache()
            rc = review.third_child_main(ctx_, mode, launcher=object(), certification=W.Certify().pair())
        calls.append(rc)
        return rc
    rc = run(ctx, mode, W.Certify(perturb={he.array_sha256(u)}), third)
    assert calls == [review.THIRD_ERROR_EXIT]
    doc, term, stops = records(ctx)
    stop = he.read_json(Path(ctx['out'])/'review'/stops[-1])
    error = he.read_json(Path(ctx['out'])/'review'/'third_cert_error.json')
    assert doc is None and term is None and len(stop['rounds']) == 1
    if case == 'bytes_read':
        assert rc == review.STOP_EXIT['spy'] and stop['kind'] == 'third_spy'
        report = he.read_json(Path(ctx['out'])/'review'/'opened_files_third.json')
        assert any(os.path.realpath(p) == os.path.realpath(report['original']) for p in report['opened'])
        assert report['original_realpath'] == os.path.realpath(report['original']) and report['original_realpath'] in report['opened_realpath']      # A5a review m10
    else:
        assert rc == review.STOP_EXIT['third_refused'] and stop['kind'] == 'third_child_refused' and stop['extra']['cause'] == case
        assert error['refusal_family'] == case and error['exception_mro'][0] == error['exception_class']


@pytest.mark.parametrize('exc, infra', [(RuntimeError('test: a bug'), False), (OSError(28, 'No space left on device (test)'), True)])
def test_another_exception_of_the_child_is_c22_5_6_b_with_its_class(tmp_path, monkeypatch, exc, infra):
    """HB review m4: the round's third_cert_error_<k>.json decides: an exception that is not IdentityNotCertified is C22.5-6 (b); the class recorded, OSError and MemoryError marked as possibly
    infrastructure (the class is not changed)."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    h = he.array_sha256(u)

    def chain(*a, **k):
        raise exc
    monkeypatch.setattr(R, 'fallback_chain', chain)
    third = third_in_process(mode, W.Certify(fail={h}).pair())
    assert run(ctx, mode, W.Certify(perturb={h}), third) == review.STOP_EXIT['implementation']
    doc, term, _ = records(ctx)
    err = doc['implementation_error']
    assert err['kind'] == 'b' and err['exception_class'] == type(exc).__name__ and bool(err['possibly_infrastructure']) is infra and term['line_status'] == he.IMPLEMENTATION
    assert he.read_json(Path(ctx['out'])/'review'/'review_failed_0.json')['kind'] == 'b' and (review.home_records(pair, mode.root)/'review.json').is_file()


def test_a_fallback_line_is_bound_to_its_fallback_json(tmp_path, monkeypatch):
    """HB review m6 (i), C22.H1-8 修订 7b: a line a third certification's chain wrote is bound to its fallback_*_after_third_<k>/fallback.json (kind, state hash); one that is not stops the pair."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    h = he.array_sha256(u)
    monkeypatch.setattr(R, 'configc_fallback', fake_configc('pass'))
    inner = third_in_process(mode, W.Certify(fail={h}).pair())

    def third(ctx_):
        rc = inner(ctx_)
        path = Path(ctx_['out'])/'fallback_configc_after_third_0'/'fallback.json'
        doc = he.read_json(path)
        path.write_text(json.dumps(dict(doc, state_hash='0'*64)))
        return rc
    assert run(ctx, mode, W.Certify(perturb={h}), third) == review.STOP_EXIT['not_finalized']
    doc, term, stops = records(ctx)
    assert doc is None and he.read_json(Path(ctx['out'])/'review'/stops[-1])['kind'] == 'fallback_json'


def test_the_pairs_lock_refuses_a_second_review_without_a_record(tmp_path, monkeypatch):
    """C22.H1-7's list, HB review m14 (v): the pair's lock is there, review.json is not (the first review stopped): a second review is refused at the lock, nothing written but its record."""
    pair, ctx, u, mode = setup(tmp_path, monkeypatch)
    calls = []
    assert run(ctx, mode, W.Certify(perturb={he.array_sha256(u)}), quiet_child(1, calls)) == review.STOP_EXIT['not_finalized']
    assert mode.lock(pair).is_file() and not (Path(ctx['out'])/'review'/review.REVIEW_RECORD).exists()
    W.XW.sealed_module._reset_cache()
    assert run(ctx, mode, W.Certify(), quiet_child(1, calls)) == review.STOP_EXIT['not_finalized'] and len(calls) == 2
    doc, term, stops = records(ctx)
    assert doc is None and term is None and he.read_json(Path(ctx['out'])/'review'/stops[-1])['kind'] == 'refused:once'
