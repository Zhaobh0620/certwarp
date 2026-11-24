"""HB items 1, 2 (contract C23.8-2, C22.H0-3, C22.H1-7): HE's frozen settings for the method, the hold-out context, the admission of every method process (load_unseal, bind_unseal in that process),
the bound allowlist handed on (read_predicted's a4_admission, StructureLayer, atropos_wm), the refusals before anything of a subject is opened.  Synthetic arrays, real ids; no frozen run."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from . import _world as W

R, he, hc, hold = W.R, W.he, W.hc, W.hold


def test_the_methods_settings_and_their_pin():
    s = he.frozen_settings()
    assert s['name'] == 'Hx24-RF05-F1200-v4' and s['tissue_prior'] == 'off' and s['review_certificate_in'] == he.REVIEW_CERTIFICATE_IN
    assert he.settings_sha256(s) == W.hb_method.SETTINGS_SHA256 == '414eccd37fe264d081e87bb63235790dc3a1ec40655c57ac64337bf84992f35d' and W.hb_method.settings_problem() is None
    assert he.LINE_STATUSES[-1] == he.IMPLEMENTATION == 'hx_implementation_error' and set(he.OUTPUT_STATUSES) == {he.HX, he.FALLBACK_C, he.FALLBACK_ID}
    assert he.STOP_QUEUE_EXIT == 9
    assert s['units']['supervisor'] == he.SUPERVISOR_UNIT and s['units']['evaluation'] == he.EVALUATION_UNIT          # C22.H0-3: the units not changed


def test_the_hold_out_sources_are_his_and_the_atropos_entrys():
    import xb_core
    hi_core = hc.hi
    source = (W.ROOT/'experiments/he/he_core.py').read_text()           # the sandbox moved HOLDOUT_SOURCES into tmp_path: the module's own text is compared
    assert "HOLDOUT_SOURCES = dict(pred_root=RESULTS/'seg'/'unet35_v4_ens_holdout_B_v1'/'pred', atropos_root=RESULTS/'holdout_B_v1'/'atropos')" in source
    assert hi_core.HOLDOUT_OUT_SUGGESTED/'pred' == Path('/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_ens_holdout_B_v1/pred') == Path(he.RESULTS)/'seg'/'unet35_v4_ens_holdout_B_v1'/'pred'
    assert Path('/home/rmcfz/seq_vp_tag_results/holdout_B_v1/atropos') == Path(he.RESULTS)/'holdout_B_v1'/'atropos' and xb_core.HB_METHOD_ARM == hold.ARM == W.hb.ARM == 'method'


def test_the_hold_out_context_is_fixed(tmp_path, monkeypatch):
    pair, ctx, _ = W.holdout_ctx(tmp_path, monkeypatch, unseal=False)
    assert ctx['pair_set'] == 'holdout' and Path(ctx['stage_dir']) == Path(hc.STAGE_ROOT) and Path(ctx['out']) == Path(hc.work_dir('method', pair))
    assert ctx['paths']['pred_root'] == str(he.HOLDOUT_SOURCES['pred_root']) and ctx['paths']['atropos_root'] == str(he.HOLDOUT_SOURCES['atropos_root']) and ctx['launcher'] == dict(kind='unit')
    assert R.holdout_context_problem(ctx) is None and ctx['settings_sha256'] == W.hb_method.SETTINGS_SHA256
    for change in (dict(paths=dict(ctx['paths'], pred_root='/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_ens/pred')), dict(out=str(Path(ctx['out']).with_name('other'))),
                   dict(stage_dir='/home/rmcfz/code/python_projects/seq_vp_tag/data/three_d/stage_L2R_val_v1'), dict(launcher=dict(kind='direct')), dict(pair_set='neither')):
        bad = dict(ctx, **change)
        assert R.holdout_context_problem(bad) and R.cli_context_problem(bad), change
        assert R.run_pair(bad) == 2 and not Path(bad['out']).exists()
    assert R.evaluate_main(ctx) == 2                                        # C22.H1-6: never evaluated by run_hx on the hold-out
    assert R.holdout_roots_problem(hc.STAGE_ROOT, '/home/rmcfz/seq_vp_tag_results/x') and R.holdout_roots_problem('/a', hc.work_dir('method', pair))


def test_without_an_unseal_record_every_process_is_refused_before_anything_is_opened(tmp_path, monkeypatch):
    import spies as spying
    pair, ctx, _ = W.holdout_ctx(tmp_path, monkeypatch, unseal=False)
    spy = spying.Spies(R.pair_path(ctx)).install()
    try:
        with pytest.raises(hold.HoldoutAdmissionRefused, match='unseal'):
            R.admitted_images(ctx)
    finally:
        spy.remove()
    assert spy.loader_calls == [] and not [p for p in spy.opened if p.startswith(str(hc.STAGE_ROOT))], spy.opened     # the sealed lists only (ids), nothing of the stage
    assert isinstance(hold.HoldoutAdmissionRefused('x', 'y'), R.admission_refusals())


def test_with_the_unseal_object_the_pair_is_admitted_and_the_bound_allowlist_handed_on(tmp_path, monkeypatch):
    pair, ctx, sim = W.holdout_ctx(tmp_path, monkeypatch)
    pin, record = R.admitted_images(ctx)
    bound = hold.allowlist_of(ctx, pin)
    from bayesreg.io.unseal import bound_problem
    assert pin.fixed_subject == pair.split('__')[0] and pin.provenance['keys_read'] == ['fixed', 'moving', 'initial_u'] and bound.unseal is not None and bound_problem(bound) is None
    assert record['admission']['pairs_tsv']['row'] == 1 and record['allowlist_sha256'] == bound.sha256
    with pytest.raises(hold.HoldoutAdmissionRefused, match='allowlist'):
        hold.allowlist_of(ctx, SimpleNamespace())                        # a PairInput not admitted in this process
    assert hold.allowlist_of(dict(ctx, pair_set='dev'), pin) == ctx['paths']['allowlist']
    # the reverse direction and a pair not on the list: refused (C23.12-2 (a))
    fixed, moving = pair.split('__')
    rev = dict(ctx, pair=f'{moving}__{fixed}')
    with pytest.raises(hold.HoldoutAdmissionRefused, match='other way round'):
        hold.admitted_images(rev)


def test_a4_admission_of_a_hold_out_pair_through_the_unseal_object(tmp_path, monkeypatch):
    """make_inputs.a4_admission_unsealed: check_subject through the unseal object; D24 as hi_core.other_layers (never_for_training only in the unsealed source; not a training subject)."""
    pair, ctx, sim = W.holdout_ctx(tmp_path, monkeypatch)
    pin, _ = R.admitted_images(ctx)
    bound = hold.allowlist_of(ctx, pin)
    mi = he.core.make_inputs()
    subjects = [pin.fixed_subject, pin.moving_subject]
    split = SimpleNamespace(training_subjects=frozenset(), sha256='f'*64)
    rec = mi.a4_admission_unsealed(subjects, bound, split)
    assert rec['unsealed_source'] == 'holdout_B_v1_unsealed' and all(r['admitted_by_unseal'] for r in rec['guard'].values()) and mi.unsealed_allowlist(bound) and not mi.unsealed_allowlist(ctx['paths']['allowlist'])
    with pytest.raises(SystemExit, match='training subject'):
        mi.a4_admission_unsealed(subjects, bound, SimpleNamespace(training_subjects=frozenset([pin.moving_subject]), sha256='f'*64))
    from bayesreg.io.guard import load_dev_allowlist
    with pytest.raises(SystemExit):
        mi.a4_admission_unsealed(subjects, load_dev_allowlist(sim.repo/'configs/bayesreg/dev_subjects.json'), split)     # not bound: refused


def test_the_structure_layer_and_the_atropos_qc_get_the_bound_allowlist(tmp_path, monkeypatch):
    pair, ctx, _ = W.holdout_ctx(tmp_path, monkeypatch)
    pin, _ = R.admitted_images(ctx)
    bound = hold.allowlist_of(ctx, pin)
    mi = he.core.make_inputs()
    seen = {}
    import bayesreg.seg.structure_layer as sl

    class Layer:
        def __init__(self, pair, source, **kwargs):
            seen.update(pair=pair, source=source, kwargs=kwargs)
            raise sl.ProvenanceRejected('test: stop after the call')
    monkeypatch.setattr(sl, 'StructureLayer', Layer)
    unet = SimpleNamespace(segmenter_id='unet35_v4_ens', registry=str(W.ROOT/'configs/bayesreg/segmenters_hs_v1.json'), pred_root=tmp_path/'pred', models_root=tmp_path/'m', image_root=tmp_path/'i',
                           folds=None, folds_sha256=None)
    with pytest.raises(SystemExit, match='structure layer refused'):
        mi.pair_structure_layer(pin, bound, unet)
    assert seen['kwargs'] == dict(allowlist=bound) and seen['kwargs']['allowlist'] is bound and seen['source'].allowlist_path == Path(bound.path)
    calls = {}

    def posteriors(seg_root, subject, pair, side, allowlist=None):
        calls.update(allowlist=allowlist)
        raise ValueError('test: stop after the call')
    monkeypatch.setattr(mi, 'posteriors', posteriors)
    d = tmp_path/'atropos'/pin.fixed_subject
    d.mkdir(parents=True)
    (d/'provenance.json').write_text('{}')
    with pytest.raises(SystemExit, match='Atropos posteriors refused'):
        mi.atropos_wm(tmp_path/'atropos', pin, 'fixed', allowlist=bound)
    assert calls['allowlist'] is bound


def test_the_holdout_dry_run_refuses_at_the_unseal_step_and_writes_nothing(tmp_path, monkeypatch, capsys):
    """Without the unseal record the dry run is refused at the admission; the code identity (rule one, P1 from runs/holdout_B_v1/pinned_code.json) is stubbed as passing here (the code tree has no
    pinned record before the unsealing: then the dry run is refused at the identity already, the second case)."""
    pair, ctx, _ = W.holdout_ctx(tmp_path, monkeypatch, unseal=False)
    monkeypatch.setattr(R.core, 'frozen_api', lambda: None)                 # the tests' jd_curl is the code tree's (the frozen tree is checked by the real dry run)
    real = hold.hc.pinned_code_problems
    monkeypatch.setattr(hold.hc, 'pinned_code_problems', lambda pinned=None, *, repo=None, run=None: (dict(pinned=R.code_head(), passed=True), []))
    monkeypatch.setattr(R.he, 'identity', lambda root, frozen=None: (dict(repo=dict(head=R.code_head())), []))
    before = sorted(str(p) for p in tmp_path.rglob('*'))
    assert R.holdout_dry_run(pair) == 8
    out = capsys.readouterr().out
    assert 'admission refused' in out and 'unseal' in out and 'pair_set holdout' in out and sorted(str(p) for p in tmp_path.rglob('*')) == before
    monkeypatch.setattr(hold.hc, 'pinned_code_problems', real)              # the real rule one on this code tree: no pinned record -> refused, nothing written
    assert R.holdout_dry_run(pair) != 0
    out = capsys.readouterr().out
    assert 'pinned_code.json' in out and sorted(str(p) for p in tmp_path.rglob('*')) == before


def test_run_pairs_implementation_error_line(tmp_path, monkeypatch):
    """C22.5-6: implementation_error_line's record (the full path is the frozen pipeline test of tests/test_bayesreg/he)."""
    pair, ctx, _ = W.holdout_ctx(tmp_path, monkeypatch, unseal=False)
    out = Path(ctx['out'])
    (out/'fallback_identity').mkdir(parents=True)
    he.write_json(out/'fallback_identity'/'fallback.json', dict(kind='identity', state_hash='0'*64, certificate=dict(passed=False)))
    try:
        raise R.IdentityNotCertified('the identity did not pass the certificate: an implementation error, no output')
    except R.IdentityNotCertified as exc:
        line = R.implementation_error_line(ctx, None, exc, unit=None, t88=None, rc_t88=None, repair_record=None, rc_repair=None, failure_record=None, c_line=None, seconds_total=0.)
    assert line['line_status'] == he.IMPLEMENTATION and line['stop_queue'] is True and line['output'] is None and line['implementation_error']['kind'] == 'a'
    assert line['implementation_error']['certificates'] == [dict(kind='identity', state_hash='0'*64, certificate=dict(passed=False))]
    assert any(r['path'].endswith('fallback_identity/fallback.json') for r in line['implementation_error']['evidence'])
    try:
        raise OSError('disk')
    except OSError as exc:
        assert R.implementation_error_line(ctx, None, exc, unit=None, t88=None, rc_t88=None, repair_record=None, rc_repair=None, failure_record=None, c_line=None, seconds_total=0.)['implementation_error']['kind'] == 'b'


def test_a_hold_out_process_checks_the_code_against_the_pinned_commit_not_heads_equality(tmp_path, monkeypatch):
    """runs/20261109-align item 7 (decision_log 2026-11-09, df8541d): make_holdout_context(pinned=) records the pinned commit as code_head after holdout_common.pinned_code_problems passes; every
    process of a hold-out pair (check_identity) runs that check, so a later commit that only adds an authorization (HEAD moved) passes and a code change does not."""
    calls = []
    verdict = dict(problems=[])

    def pinned_code_problems(pinned, *, repo=None, run=None):
        calls.append((pinned, Path(repo)))
        return dict(pinned='c'*40, passed=not verdict['problems']), list(verdict['problems'])
    monkeypatch.setattr(hold.hc, 'pinned_code_problems', pinned_code_problems)
    pair, ctx, _ = W.holdout_ctx(tmp_path, monkeypatch, unseal=False)
    ctx = R.make_holdout_context(pair, device='cpu', check_identity=False, pinned='c'*12)
    assert ctx['code_head'] == 'c'*40 and calls[-1] == ('c'*12, Path(R.core.ROOT))
    monkeypatch.setattr(R.he, 'identity', lambda root, frozen: (dict(repo=dict(head='d'*40)), []))          # HEAD moved by a committed authorization
    record, problems = R.check_identity(dict(ctx, check_identity=True), tmp_path, 'identity.json')
    assert problems == [] and record['pinned_head'] == 'c'*40 and calls[-1][0] == 'c'*40
    verdict['problems'] = ['M experiments/he/run_hx.py: a code directory changed since the pinned commit']
    record, problems = R.check_identity(dict(ctx, check_identity=True), tmp_path, 'identity_1.json')
    assert problems == verdict['problems'] and W.he.read_json(tmp_path/'identity_1.json')['problems'] == problems
    with pytest.raises(ValueError, match='pinned'):
        R.make_holdout_context(pair, device='cpu', check_identity=False, pinned='c'*40)
