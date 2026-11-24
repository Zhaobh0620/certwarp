"""HP Atropos entry (experiments/holdout/atropos_holdout.py; runs/20261107-hp/任务书.md item 3, 4): experiments/d0/atropos_seg.py's own segment() with the same
commands, threads and checks on the hold-out stage that prep_stage.py wrote, admitted only through SG's unseal object; the tissue-layer check with the bound allowlist
(C23.4-8); a subject in two pairs; once; the development check's readings.  ANTs is replaced by fake binaries (no ANTs runs in the tests)."""
import json
from pathlib import Path

import numpy as np
import pytest

from .. import _p0_pins as P0
from . import _world as W

hc, core, prep, ah, atropos_seg, S = W.hc, W.core, W.prep, W.ah, W.atropos_seg, W.S


def prepared(tmp_path, monkeypatch, **kwargs):
    w = W.make_world(tmp_path, monkeypatch, **kwargs)
    S._reset_cache()
    prep.run_holdout(unseal_path=w.record_path, code=W.CODE, log=W.quiet)
    w.ants_log = W.fake_ants(tmp_path, monkeypatch)
    return w


def resync_prep_record(stage):
    """The synthetic stage changed by a test: prep_record.json made to name the new SHA256SUMS (so that only the change under test is seen)."""
    path = Path(stage)/prep.RECORD
    document = json.loads(path.read_text())
    document['stage_summary']['sha256sums_sha256'] = W.file_sha256(Path(stage)/prep.SUMS)
    path.write_text(json.dumps(document))


def run(w, **kwargs):
    S._reset_cache()
    return ah.run_holdout(unseal_path=kwargs.pop('unseal_path', w.record_path), code=W.CODE, log=W.quiet, **kwargs)


def test_the_hold_out_tissue_layer_is_atropos_seg_s_own(tmp_path, monkeypatch):
    w = prepared(tmp_path, monkeypatch)
    record = run(w)
    out = Path(hc.ATROPOS_ROOT)
    order = [s for p in W.PAIRS for s in p]
    assert list(record['subjects']) == order and record['passed'] and not record['spy']['breaches']
    calls = W.ants_calls(w.ants_log)
    assert [c['tool'] for c in calls] == ['ImageMath', 'Atropos']*len(order)
    imagemath = ['3', 't1_trunc.nii.gz', 'TruncateImageIntensity', 't1.nii.gz', '0.01', '0.99', '256', 'mask.nii.gz']
    atropos = ['-d', '3', '-a', 't1_trunc.nii.gz', '-x', 'mask.nii.gz', '-i', 'KMeans[3]', '-k', 'Gaussian', '-c', '[5,0.001]', '-m', '[0.3,1x1x1]', '-r', '0', '-o',
               '[labels.nii.gz,posteriors%02d.nii.gz]', '-v', '1']
    for i, subject in enumerate(order):
        assert calls[2*i]['argv'] == imagemath and calls[2*i+1]['argv'] == atropos
        assert Path(calls[2*i]['cwd']) == out/subject and calls[2*i]['threads'] == '8' and calls[2*i+1]['threads'] == '8'
        prov = json.loads((out/subject/'provenance.json').read_text())
        assert prov['class_names'] == ['CSF', 'GM', 'WM'] and prov['wm_index'] == 2 and prov['segmenter_id'] == 'atropos'
        assert prov['image_source']['stage'] == ah.STAGE_NAME and prov['image_source']['keys_read'] == ['fixed', 'moving', 'initial_u']
        assert prov['environment'] == dict(ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8) and prov['parameters'] == atropos_seg.PARAMS
        assert prov['command'][1] == [str(atropos_seg.ANTS/'Atropos')]+atropos
        assert prov['checks']['qc_min_class_fraction_ge_0.05'] is True
    assert atropos_seg.validate_segmentation_provenance is ah.validate_segmentation_provenance          # the wrapper restored the module's name
    assert record['shared_subjects'] == []
    assert {m['member'] for m in record['spy']['members']} <= set(core.PAIR_MEMBERS)
    assert json.loads((out/ah.RECORD).read_text())['admission']['unseal']['sha256'] == W.file_sha256(w.record_path)


def test_without_the_wrapper_the_tissue_check_refuses_a_sealed_pair(tmp_path, monkeypatch):
    """Why tissue_layer_check exists: segment() validates without an allowlist; for a pair with a sealed subject C23.4-8 refuses that (ProvenanceRejected)."""
    from bayesreg.io.pair import ProvenanceRejected
    from bayesreg.io.unseal import bind_unseal, load_unseal
    w = prepared(tmp_path, monkeypatch)
    S._reset_cache()
    bound = bind_unseal(w.allowlist, load_unseal(w.record_path))
    entries = ah.StageEntries(hc.STAGE_ROOT, bound)
    subject = W.PAIRS[0][0]
    with pytest.raises(ProvenanceRejected, match='C23.4-8'):
        atropos_seg.segment(subject, entries[subject], tmp_path/'bare', 8, 'x')
    with ah.tissue_layer_check(bound):
        prov = atropos_seg.segment(W.PAIRS[0][1], entries[W.PAIRS[0][1]], tmp_path/'wrapped', 8, 'x')
    assert prov['class_names'] == ['CSF', 'GM', 'WM']
    assert atropos_seg.validate_segmentation_provenance is ah.validate_segmentation_provenance


def test_the_wrapper_restores_on_an_exception_and_refuses_a_foreign_name(monkeypatch):
    with pytest.raises(ZeroDivisionError):
        with ah.tissue_layer_check(object()):
            assert atropos_seg.validate_segmentation_provenance is not ah.validate_segmentation_provenance
            1/0
    assert atropos_seg.validate_segmentation_provenance is ah.validate_segmentation_provenance
    monkeypatch.setattr(atropos_seg, 'validate_segmentation_provenance', lambda *a, **k: None)
    with pytest.raises(RuntimeError, match='not wrapped'):
        with ah.tissue_layer_check(object()):
            pass


def test_no_unseal_object_is_refused_before_the_stage_is_read(tmp_path, monkeypatch):
    w = prepared(tmp_path, monkeypatch)
    for path in (None, tmp_path/'elsewhere.json'):
        with pytest.raises(hc.Refused) as info:
            run(w, unseal_path=path)
        assert info.value.stage == 'unseal'
        stage = str(Path(hc.STAGE_ROOT).resolve())
        assert not [e for e in info.value.hp_spy['events'] if e['resolved'].startswith(stage)]
    assert not Path(hc.ATROPOS_ROOT).exists() and not (Path(hc.LOCK_ROOT)/ah.LOCK_NAME).exists()
    assert W.ants_calls(w.ants_log) == []


@pytest.mark.parametrize('name', ['H1', 'H2', 'H3', 'H4'])
def test_h1_to_h4_are_refused(tmp_path, monkeypatch, name):
    subject = dict(H1=W.H1[0], H4=W.H4[0], H2='S38', H3=W.H3[1])[name]
    w = W.make_world(tmp_path, monkeypatch, rows=W.PAIRS+[(subject, W.HOLDOUT[0])])
    with pytest.raises(hc.Refused) as info:
        run(w)
    assert info.value.stage == 'never'
    assert not Path(hc.ATROPOS_ROOT).exists()


def test_a_stage_that_is_not_the_admitted_rows_is_refused(tmp_path, monkeypatch):
    w = prepared(tmp_path, monkeypatch)
    sums = Path(hc.STAGE_ROOT)/ah.SUMS
    text = sums.read_text()
    sums.write_text(text.replace(text.split('  ')[0], '0'*64, 1))
    with pytest.raises(hc.Refused, match='differ from SHA256SUMS'):
        run(w)
    sums.write_text(text)
    selection = Path(hc.STAGE_ROOT)/'selection.json'
    document = json.loads(selection.read_text())
    document['pairs'] = document['pairs'][::-1]
    selection.write_text(json.dumps(document))
    with pytest.raises(hc.Refused, match='not the admitted rows in their order'):
        run(w)
    assert not Path(hc.ATROPOS_ROOT).exists() and W.ants_calls(w.ants_log) == []


def test_a_subject_in_two_pairs_is_checked_as_atropos_seg_does(tmp_path, monkeypatch):
    a, b, c, d = W.HOLDOUT
    w = prepared(tmp_path, monkeypatch, rows=[(a, b), (b, c)], data_subjects=W.HOLDOUT)
    record = run(w)
    assert record['shared_subjects'] == [b] and list(record['subjects']) == [a, b, c]
    prov = json.loads((Path(hc.ATROPOS_ROOT)/b/'provenance.json').read_text())
    assert prov['image_source']['pairs'] == [f'{a}__{b}:moving', f'{b}__{c}:fixed']


def test_a_subject_whose_image_differs_between_its_pairs_stops(tmp_path, monkeypatch):
    a, b, c, d = W.HOLDOUT
    w = prepared(tmp_path, monkeypatch, rows=[(a, b), (b, c)], data_subjects=W.HOLDOUT)
    stage = Path(hc.STAGE_ROOT)
    name = f'{b}__{c}'
    with np.load(stage/name/'pair.npz') as npz:                       # the synthetic stage: change b's array in its second pair, keep the stage self-consistent
        arrays = {k: np.array(npz[k]) for k in ('fixed', 'moving', 'initial_u')}
    arrays['fixed'].flat[0] += 1.
    np.savez_compressed(stage/name/'pair.npz', **arrays)
    manifest = json.loads((stage/name/'manifest.json').read_text())
    manifest['pair_sha256'] = W.file_sha256(stage/name/'pair.npz')
    (stage/name/'manifest.json').write_text(json.dumps(manifest))
    (stage/ah.SUMS).write_text(''.join(f'{W.file_sha256(stage/n)}  {n}\n' for n in sorted(prep.stage_files([(a, b), (b, c)]))))
    resync_prep_record(stage)
    with pytest.raises(SystemExit, match='differs between pairs'):
        run(w)
    stopped = json.loads((Path(hc.ATROPOS_ROOT)/ah.STOPPED).read_text())
    assert 'differs between pairs' in stopped['stopped']


def test_once_and_never_overwritten(tmp_path, monkeypatch):
    w = prepared(tmp_path, monkeypatch)
    run(w)
    out = Path(hc.ATROPOS_ROOT)
    before = {p: W.file_sha256(p) for p in out.rglob('*') if p.is_file()}
    with pytest.raises(hc.Refused) as info:
        run(w)
    assert info.value.stage == 'once'                                        # the lock first (holdout_common.check_once)
    lock = Path(hc.LOCK_ROOT)/ah.LOCK_NAME
    lock.rename(lock.with_name('moved.lock'))                                # synthetic: without the lock the output root still refuses
    with pytest.raises(hc.Refused) as info:
        run(w)
    assert info.value.stage == 'output'
    assert {p: W.file_sha256(p) for p in out.rglob('*') if p.is_file()} == before


def test_the_dry_run_on_the_real_tree_is_refused_at_the_admission(tmp_path, monkeypatch):
    W.real_pins(monkeypatch)
    out = tmp_path/'dry'
    assert ah.main(['run', '--dry-run', '--out', str(out)], code=dict(W.CODE, root=str(W.ROOT)), out=W.quiet) == 8
    record = json.loads((out/'dry_run.json').read_text())
    assert record['refused']['stage'] == 'unseal'
    assert 'no_unseal_record' in [e['kind'] for e in record['expected']] and record['passed'] is False and len(record['problems']) == len(record['expected'])+len(record['unexpected'])   # A5a review m8
    assert not [e for e in record['spy']['events'] if '/mnt/d' in e['resolved'] or 'stage' in Path(e['resolved']).parts]
    assert not Path(hc.ATROPOS_ROOT).exists() and not Path(hc.LOCK_ROOT).exists()


def test_the_machine_gate_of_holdout_common_is_hi_core_s_plus_memory_and_load(tmp_path, monkeypatch):
    import subprocess
    gate = W.REAL_HC['machine_gate']

    def fake(argv, **kwargs):
        if argv[0] == 'systemctl':
            return subprocess.CompletedProcess(argv, 0, 'tag-hb-other.service loaded active running x\n', '')
        return subprocess.CompletedProcess(argv, 0, '0\n' if 'utilization' in ' '.join(argv) else '', '')
    with pytest.raises(hc.Refused, match='other units are active'):
        gate(own_units=('tag-hb-atropos',), run=fake)

    def idle(argv, **kwargs):
        if argv[0] == 'systemctl':
            return subprocess.CompletedProcess(argv, 0, '', '')
        if argv[0] == 'journalctl':                         # A5a review I1: the user journal readable (its newest entry), no unit of this project in the window
            entry = json.dumps({'__MONOTONIC_TIMESTAMP': '1000000', '_BOOT_ID': 'b'*32})+'\n'
            return subprocess.CompletedProcess(argv, 0, entry if '-n' in argv else '', '')
        return subprocess.CompletedProcess(argv, 0, '0\n' if 'utilization' in ' '.join(argv) else '', '')
    meminfo = W.write_text(tmp_path/'meminfo', 'MemAvailable:   1048576 kB\n')
    loadavg = W.write_text(tmp_path/'loadavg', '0.10 0.1 0.1 1/1 1\n')
    with pytest.raises(hc.Refused, match='MemAvailable'):
        gate(own_units=('tag-hb-atropos',), run=idle, meminfo=meminfo, loadavg=loadavg)
    W.write_text(meminfo, 'MemAvailable:   10485760 kB\n')
    W.write_text(loadavg, '2.5 0.1 0.1 1/1 1\n')
    with pytest.raises(hc.Refused, match='load'):
        gate(own_units=('tag-hb-atropos',), run=idle, meminfo=meminfo, loadavg=loadavg)
    W.write_text(loadavg, '0.2 0.1 0.1 1/1 1\n')
    assert gate(own_units=('tag-hb-atropos',), run=idle, meminfo=meminfo, loadavg=loadavg)['passed'] is True


def test_the_launch_line_is_a_tag_unit():
    line = core.launch_line(unit=ah.UNIT, script=ah.SCRIPT, args=['run'], runtime_s=60, cpu_quota='800%')
    assert '--unit=tag-hb-atropos' in line and 'MemorySwapMax=0' in line and 'CPUQuota=800%' in line
    assert f"StandardOutput=append:{hc.RESULTS/'tag-hb-atropos_unit.log'}" in line and 'systemd-run --user' in line
    with pytest.raises(ValueError):
        core.launch_line(unit='seq-vp-x', script=ah.SCRIPT, args=[], runtime_s=60)


# ---- the development check ---------------------------------------------------------------------------------------------------------------------------
def devcheck_world(tmp_path, monkeypatch, *, change=None):
    """The synthetic development stage in the old format and a reference written by atropos_seg itself (subject_images, segment: D0's path), with an evaluation file
    beside it that nobody may read.  change: (file name, how) applied to the reference of the first subject."""
    w = W.make_world(tmp_path, monkeypatch, decided=False)
    monkeypatch.setattr(hc, 'DEV_PAIRS', W.DEV_PAIRS)
    monkeypatch.setattr(ah, 'DEV_SUBJECTS', (W.DEV[0], W.DEV[1]))
    W.write_old_stage(hc.DEV_STAGE, w.data, W.DEV_PAIRS)
    monkeypatch.setattr(ah, 'ATROPOS_DEVCHECK_PLAN_SHA256', W.plan_file(w, ah.DEVCHECK_PLAN_REL))
    w.ants_log = W.fake_ants(tmp_path, monkeypatch)
    ref = ah.reference_root()
    ref.mkdir(parents=True)
    S._reset_cache()
    found = atropos_seg.subject_images(w.allowlist, Path(hc.DEV_STAGE))
    for subject in (W.DEV[0], W.DEV[1]):
        atropos_seg.segment(subject, found[subject], ref, 8)
    (ref/'dice_vs_truth_seg4.json').write_text('{}')
    if change:
        name, how = change
        path = ref/W.DEV[0]/name
        import nibabel as nib
        img = nib.load(str(path))
        a = np.asarray(img.dataobj).copy()
        a.flat[np.flatnonzero(a)[0]] = how(a.flat[np.flatnonzero(a)[0]])
        nib.save(nib.Nifti1Image(a, img.affine, img.header), str(path))
    return w


def test_the_development_check_reads_bitwise(tmp_path, monkeypatch):
    devcheck_world(tmp_path, monkeypatch)
    S._reset_cache()
    record = ah.run_devcheck(code=W.CODE, log=W.quiet)
    assert record['reading'] == '逐位相同 2/2', record['rows']
    assert record['spy']['passed'] and not [e for e in record['spy']['events'] if 'dice' in e['path']]
    prov = json.loads((ah.devcheck_root()/W.DEV[1]/'provenance.json').read_text())
    assert prov['image_source']['pairs'] == [f'{W.DEV[0]}__{W.DEV[1]}:moving', f'{W.DEV[1]}__{W.DEV[2]}:fixed']
    assert {m['member'] for m in record['spy']['members']} <= set(core.PAIR_MEMBERS)       # the old stage's label members were never asked


def test_the_development_check_reads_a_difference_of_the_outputs(tmp_path, monkeypatch):
    devcheck_world(tmp_path, monkeypatch, change=('posteriors02.nii.gz', lambda v: v+np.float32(.25)))
    S._reset_cache()
    record = ah.run_devcheck(code=W.CODE, log=W.quiet)
    assert record['reading'] == ah.READ_NONDETERMINISTIC
    row = record['rows'][W.DEV[0]]
    assert row['inputs_bitwise'] and not row['outputs_bitwise']
    assert row['outputs']['posteriors02.nii.gz']['max_abs_diff'] == pytest.approx(.25)
    assert 0 < row['posterior_argmax_agreement_in_mask'] <= 1


def test_the_development_check_reads_a_difference_of_the_inputs(tmp_path, monkeypatch):
    devcheck_world(tmp_path, monkeypatch, change=('t1.nii.gz', lambda v: v*2))
    S._reset_cache()
    record = ah.run_devcheck(code=W.CODE, log=W.quiet)
    assert record['reading'] == ah.READ_INPUTS_DIFFER


def test_the_development_checks_plan_pin_is_the_committed_plan():
    """P0: ATROPOS_DEVCHECK_PLAN_SHA256 is the sha256 of the committed atropos_devcheck_plan.md; check_plan accepts it on this tree (only the plan is read).  None and a
    wrong value are refused in the test below."""
    P0.assert_pinned(ah.ATROPOS_DEVCHECK_PLAN_SHA256, ah.DEVCHECK_PLAN_REL)
    assert ah.check_plan(P0.REPO) == dict(path=ah.DEVCHECK_PLAN_REL, sha256=ah.ATROPOS_DEVCHECK_PLAN_SHA256)


def test_the_development_check_refuses_a_draft_plan_and_missing_references(tmp_path, monkeypatch):
    devcheck_world(tmp_path, monkeypatch)
    monkeypatch.setattr(ah, 'ATROPOS_DEVCHECK_PLAN_SHA256', None)
    S._reset_cache()
    with pytest.raises(hc.Refused, match='draft'):
        ah.run_devcheck(code=W.CODE, log=W.quiet)
    monkeypatch.setattr(ah, 'ATROPOS_DEVCHECK_PLAN_SHA256', '0'*64)
    S._reset_cache()
    with pytest.raises(hc.Refused, match='is not the pinned'):
        ah.run_devcheck(code=W.CODE, log=W.quiet)
    assert not ah.devcheck_root().exists()
