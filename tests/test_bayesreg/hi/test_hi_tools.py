"""HI (runs/20261107-hi/任务书.md): experiments/hi/{hi_core,run_hi}.py, the backends, the provenance, the guard, the unseal record's content, the development check and the dry runs.  CPU only; synthetic small volumes and
tiny made-up members in tmp_path; the record and the decision log in a temporary git repository.  The real repository files read are JSON and id lists only: the allowlist (configs/bayesreg/dev_subjects.json, parsed,
its sources not re-hashed), the sealed lists of bayesreg.eval.truth (splits/bayesreg_v1/holdout_B_subjects.txt and the paper-1 plan lists under archive/), the B-line folds, the v4 registry, the structure grouping and
runs/20261107-hi/*.md.  No image, label, member or prediction of any real subject or results directory is opened; the guard's counter-examples use the ids only, and a spy shows that no image was opened.
The run-once lock, the resume record, the decision of the check, the machine gate, the code root and the git checks: test_hi_once.py.  The tripwire of the allowlist: test_holdout_tripwire.py.
"""
import dataclasses
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from bayesreg.eval.truth import SealedLists, load_sealed_lists
from bayesreg.io.guard import LPBA40_TEST_REJECT, SubjectSealed, check_subject, load_dev_allowlist
from bayesreg.io.pair import load_segmenter_registry, subject_list_sha256, verify_segmentation_files
from bayesreg.seg.unet import synthetic
from bayesreg.seg.unet.data import load_folds
from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping
from bayesreg.seg.unet.infer import ModelSelectionRefused, segment_subject
from bayesreg.seg.unet.train import file_sha256

from bayesreg.io import unseal as bayes_unseal

from ._world import (CONTACT, DEV, H1, MEMBERS, REAL, RECORD, ROOT, SUBJECTS, T0, Spy, commit_all, holdout_kwargs, hi, load_p, log_record, quiet, real_pins, run_hi, simulated_unseal_branch, world,
                     write_json, write_unseal)

REGISTERED_KEYS = {'layer', 'segmenter_id', 'model_ids', 'model_sha256', 'folds', 'trained_with_labels', 'test_labels_seen', 'subject', 'policy', 'members', 'grouping', 'grouping_status', 'groups_path', 'groups_sha256',
                   'class_names', 'background_class', 'background_labels', 'group_labels', 'grouping_check', 'input_sha256', 'grid', 'normalization', 'label_contact_check', 'registry_path', 'registry_sha256',
                   'registry_note', 'composition', 'allowlist_path', 'allowlist_sha256', 'version', 'command', 'environment', 'runtime_seconds', 'output_sha256', 'outputs', 'seconds_write'}
# the keys of the registered /home/rmcfz/seq_vp_tag_results/seg/unet35_v4_ens/pred/OASIS_OAS1_0438_MR1/provenance.json, sha256 a92523d233d4bb8edb722d57fb35427bb48e0a3f45b1228ffdc35681056e3c3b, read once by the
# implementer on 2026-11-07 (review m11: the tests do not read a results directory)
MEMBER_KEYS = {'model_id', 'fold', 'seed', 'member_sha256', 'temperature', 'checkpoint_choice', 'checkpoint_iteration', 'window', 'window_gaussian_sigma', 'percentile', 'amp'}


@pytest.fixture
def w(tmp_path, decided, monkeypatch):
    return world(tmp_path, monkeypatch=monkeypatch)


# ---- the anchor and the registry -----------------------------------------------------------------------------------------------------------------
def test_the_anchor_is_the_contracts_four_values():
    """C22.1-2 修订 4 (登记) and C22.9-4 (xv) (锚点), written here again: the id, the policy, the registry file, its sha256."""
    assert dict(hi.CONTRACT_SEGMENTER) == dict(segmenter_id='unet35_v4_ens', policy='all_folds_mean', registry='configs/bayesreg/segmenters_hs_v1.json',
                                               registry_sha256='b7682421bf6e8d07c606ad55cd47db7936a293c5e79d00839c82e3f8c2edefa1')
    text = (ROOT/'docs/bayesreg/01_合同.md').read_text(encoding='utf-8')
    assert 'b7682421bf6e8d07c606ad55cd47db7936a293c5e79d00839c82e3f8c2edefa1' in text and 'configs/bayesreg/segmenters_hs_v1.json' in text
    assert hi.PRODUCTION_BACKEND == 'hs_a' and hi.BACKENDS == ('hs_a', 'segment_subject') and hi.UNSEALABLE == ('holdout_B_v1',)
    assert hi.hc.CANDIDATES[hi.CANDIDATE]['segmenter_id'] == 'unet35_v4_ens' and hi.hc.CANDIDATES[hi.CANDIDATE]['nn_model'] is None
    assert REAL['devcheck_out'] == Path('/home/rmcfz/seq_vp_tag_results/seg/hi_devcheck_v1') and hi.REGISTERED_PRED == Path('/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_ens/pred')
    assert hi.MODELS_ROOT == Path('/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_full') and hi.DATA_ROOT == Path('/mnt/d/data')
    assert hi.UNSEAL_REL == 'splits/bayesreg_v1/unseal_holdout_B_v1.json' and hi.DECISION_LOG_REL == 'docs/decision_log.md'


def test_the_real_registry_is_the_anchors_and_a_changed_one_is_refused(tmp_path):
    real = load_segmenter_registry(ROOT/hi.CONTRACT_SEGMENTER['registry'])
    record = hi.check_registry(real)
    assert record['sha256'] == hi.CONTRACT_SEGMENTER['registry_sha256'] and record['models'] == ['unet35_v4_ens_foldA', 'unet35_v4_ens_foldB']
    changed = tmp_path/'changed.json'
    changed.write_bytes((ROOT/hi.CONTRACT_SEGMENTER['registry']).read_bytes()+b'\n')
    with pytest.raises(hi.HIRefused, match='registry: .*is not the anchor'):
        hi.check_registry(load_segmenter_registry(changed))
    document = json.loads((ROOT/hi.CONTRACT_SEGMENTER['registry']).read_text(encoding='utf-8'))
    for entry in document['models'].values():
        entry['segmenter_id'] = 'unet35_v1_ens'
    other = write_json(tmp_path/'other.json', document)
    with pytest.raises(hi.HIRefused, match='models are of'):
        hi.check_registry(load_segmenter_registry(other), dict(hi.CONTRACT_SEGMENTER, registry_sha256=load_segmenter_registry(other).sha256))


# ---- the two backends and the provenance ---------------------------------------------------------------------------------------------------------
def test_the_two_backends_store_the_same_bits_and_hs_a_in_memory_is_its_file(w, monkeypatch):
    spy = Spy(monkeypatch)
    record = hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out'))
    assert record['passed'] is True and sorted(record['rows']) == SUBJECTS and record['backend'] == 'hs_a'
    other = w.tmp/'segment_subject'
    for subject in SUBJECTS:
        segment_subject(subject, files=w.files, registry=w.registry, allowlist=w.allowlist, models_root=w.models_root, grouping=w.grouping, pred_root=other, device='cpu', segmenter_id='unet35_v4_ens', command=['test'], log=quiet)
        for name in ('probs36.npz', 'structure_groups.npz'):
            a, b = load_p(w.tmp/'out'/'pred'/subject/name), load_p(other/subject/name)
            assert a.dtype == b.dtype == np.float16 and a.shape == b.shape and np.array_equal(a.view(np.uint16), b.view(np.uint16)), (subject, name)
        plan, models, _ = hi.plan_of(w.registry, subject, w.allowlist, hi.member_files(w.registry, w.models_root))
        probs, *_ = hi.hs_a_probabilities(subject, files=w.files, registry=w.registry, models=models, plan=plan, device=torch.device('cpu'))
        p16, g16, ties = hi.stored_arrays(probs, w.grouping)
        assert ties == 0 and np.array_equal(p16.view(np.uint16), load_p(w.tmp/'out'/'pred'/subject/'probs36.npz').view(np.uint16))
        assert np.array_equal(g16.view(np.uint16), load_p(w.tmp/'out'/'pred'/subject/'structure_groups.npz').view(np.uint16))
        assert np.array_equal(p16.view(np.uint16), probs.astype(np.float16).view(np.uint16))         # (a)'s float16 with the argmax kept is the plain cast
    assert spy.labels() == [] and not [a for a in spy.asked if a[0] == 'label']


def test_the_provenance_has_every_registered_field_and_the_member_records_of_the_contract(w):
    hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out'))
    for subject in SUBJECTS:
        directory = w.tmp/'out'/'pred'/subject
        prov = json.loads((directory/'provenance.json').read_text(encoding='utf-8'))
        assert set(prov) == REGISTERED_KEYS | {'hi'}
        assert prov['segmenter_id'] == 'unet35_v4_ens' and prov['policy'] == 'all_folds_mean' and prov['test_labels_seen'] is False and prov['trained_with_labels'] is True and prov['subject'] == subject
        assert prov['registry_path'] == 'configs/bayesreg/segmenters_hs_v1.json' and prov['registry_sha256'] == w.registry.sha256 and 'staged' not in prov['registry_note']
        assert [(m['model_id'], m['member_sha256']) for m in prov['members']] == [('unet35_v4_ens_foldA', w.shas['foldA_s0']), ('unet35_v4_ens_foldA', w.shas['foldA_s1']),
                                                                              ('unet35_v4_ens_foldB', w.shas['foldB_s0']), ('unet35_v4_ens_foldB', w.shas['foldB_s1'])]
        for member, (fold, seed, temperature, window, sigma, amp) in zip(prov['members'], MEMBERS):
            assert set(member) == MEMBER_KEYS and 'best_iteration' not in member
            assert (member['fold'], member['seed'], member['temperature'], member['window'], member['window_gaussian_sigma']) == (fold, seed, temperature, window, sigma)
            assert member['amp'] is False                       # review I3: on the CPU predict_logits uses no AMP, whatever member.pt says (two of the four say true)
            assert member['checkpoint_choice'] == 'last' and member['checkpoint_iteration'] == 40 and member['percentile'] == 99.5
        assert prov['normalization']['percentile'] == 99.5 and prov['normalization']['brain_definition'] == 'aligned_norm > 0'
        assert prov['environment']['code']['commit'] == 'c'*40 and {'torch', 'cudnn_benchmark', 'cudnn_deterministic', 'device', 'numpy'} <= set(prov['environment'])
        assert prov['input_sha256'] == {'aligned_norm.nii.gz': file_sha256(w.data/subject/'aligned_norm.nii.gz')}
        block = prov['hi']
        assert block['schema'] == hi.HI_PREDICTION_SCHEMA and block['backend'] == 'hs_a' and block['purpose'] == 'hold-out' and block['amp_rule'] == hi.AMP_RULE
        assert block['unseal']['sha256'] == hashlib.sha256(w.unseal_path.read_bytes()).hexdigest() and block['unseal']['sealed_list'] == 'holdout_B_v1' and block['unseal']['bz_words'] == w.unseal['bz_words']
        assert block['unseal']['git']['record']['path'] == hi.UNSEAL_REL and block['devcheck']['sha256'] == hi.DEVCHECK_DECISION['devcheck_sha256'] and block['resume'] is None
        assert block['admission']['guard'].startswith('bayesreg.seg.unet.infer.check_subject_admitted') and block['admission']['allowlist_sources'] == ['synthetic_unsealed']
        assert verify_segmentation_files(prov, directory) == {k: file_sha256(directory/k) for k in ('probs36.npz', 'structure_groups.npz')}
    record = json.loads((w.tmp/'out'/hi.INFER_RECORD).read_text(encoding='utf-8'))
    assert record['schema'] == hi.HI_INFER_SCHEMA and record['unseal']['subjects_sha256'] == subject_list_sha256(sorted(SUBJECTS)) and record['n_reader_not_own_total'] == 0
    assert record['decision']['devcheck_sha256'] == hi.DEVCHECK_DECISION['devcheck_sha256'] and record['machine_gate']['passed'] is True and record['lock']['line']['event'] == hi.FIRST_RUN
    assert sorted(p.name for p in (w.tmp/'out'/'subjects').iterdir()) == [f'{s}.json' for s in SUBJECTS] and (w.tmp/'out'/hi.STARTED_RECORD).is_file()


def test_amp_is_recorded_as_used():
    assert hi.effective_amp(True, torch.device('cuda')) is True and hi.effective_amp(True, 'cpu') is False and hi.effective_amp(False, torch.device('cuda')) is False


def test_only_the_image_is_read(w, monkeypatch):
    """The firewall: aligned_norm.nii.gz once per subject; no label file (they exist in the synthetic data root), no pair.npz, no evaluation record."""
    spy = Spy(monkeypatch)
    hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out', files=spy.files(w.data)))
    assert spy.loads == [str(w.data/s/'aligned_norm.nii.gz') for s in SUBJECTS] and spy.asked == [('image', s) for s in SUBJECTS]
    assert spy.labels() == [] and all(Path(p).name == 'aligned_norm.nii.gz' for p in spy.images())


# ---- the guard ----------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def real():
    """The repository's allowlist (parsed only: the stage lists of data/ are not read here), the sealed lists, the training set, the v4 registry: JSON and id lists."""
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')
    sealed = load_sealed_lists(ROOT)
    training = sorted(load_folds(ROOT/'splits/bayesreg_v1/folds.json').training_subjects)
    sources = {s['name']: s for s in allowlist.sources}
    listed = {s for ids in sealed.subjects.values() for s in ids}
    off = [s for s in (f'OASIS_OAS1_{n:04d}_MR1' for n in range(1, 458)) if s not in listed and not allowlist.allows(s)]
    groups = dict(holdout_B=sorted(sealed.subjects['holdout_B_v1']), H1=sorted(sealed.subjects['H1']), H4=sorted(sealed.subjects['H4']), H3=sorted(sealed.subjects['H3']), H2=['S37', 'S38', 'S39', 'S40'],
                  T0=sorted(sources['stage_T0_tune_v1']['subjects']), training=training, dev=sorted(sources['stage_L2R_val_v1']['subjects']), E11=sorted(sources['stage_E11_native_v1']['subjects']),
                  LPBA40=sorted(sources['stage_LPBA40_v1_tune']['subjects']), off=off[:12], noncanonical=['oasis_0438', 'OASIS_OAS1_438_MR1', 'S41', ''])
    registry = load_segmenter_registry(ROOT/hi.CONTRACT_SEGMENTER['registry'])
    return types.SimpleNamespace(allowlist=allowlist, sealed=sealed, training=training, groups=groups, registry=registry)


def test_the_real_id_lists_are_what_the_census_assumes(real):
    g = real.groups
    assert len(g['holdout_B']) == 100 and len(g['H1']) == 100 and len(g['H4']) == 40 and len(g['H3']) == 18 and len(g['T0']) == 24 and len(g['training']) == 106 and len(g['dev']) == 20 and len(g['off']) == 12
    assert len(g['E11']) == 10 and len(g['LPBA40']) == 24 and all(s.startswith('S') and int(s[1:]) <= 36 for s in g['LPBA40'])
    assert g['holdout_B'] == sorted((ROOT/'splits/bayesreg_v1/holdout_B_subjects.txt').read_text().split())


EXPECTED = dict(holdout_B=(SubjectSealed, 'not on the development allowlist'), H1=(SubjectSealed, 'not on the development allowlist'), H4=(SubjectSealed, 'not on the development allowlist'),
                H3=(SubjectSealed, 'canonical form'), H2=(SubjectSealed, 'not on the development allowlist'), off=(SubjectSealed, 'not on the development allowlist'), noncanonical=(SubjectSealed, 'canonical form'),
                T0=(hi.HIRefused, 'never_for_training'), training=(hi.HIRefused, 'training subject'), dev=(hi.HIRefused, 'allowlist sources'), E11=(hi.HIRefused, 'allowlist sources'),
                LPBA40=(hi.HIRefused, 'allowlist sources'))
# C23.9-13 (2): the counter-examples now run on the simulated unseal branch (simulated_unseal_branch: the repository's allowlist with the 100 in a never_for_training source, bayesreg's real loader, bind_unseal), so the
# 100 pass the sealed guard and are refused by HI's own layers when the record object names another source -- what the unseal branch asserted (diff review n4); the refusal on main's allowlist (step 2 of the guard)
# stays in test_the_hold_out_is_admitted_only_through_a_valid_record.
EXPECTED['holdout_B'] = (hi.HIRefused, 'never_for_training|allowlist sources')
FAKE_SOURCE = 'not_the_unsealed_source'


@pytest.fixture
def sim(tmp_path, monkeypatch):
    return simulated_unseal_branch(tmp_path, monkeypatch)


def unseal_naming(sim, ids, *, sealed_list='holdout_B_v1', source=FAKE_SOURCE):
    """C23.9-13 (2): a record object that names `ids`, made from a real unseal object of bayesreg's loader (sim.unseal: a committed record in a temporary git repository, bound to sim.allowlist by bind_unseal)
    with its subjects, list and source replaced in memory: the guard takes the base object on the bound allowlist, so the refusal must come from the guard and HI's own layers, not from the record."""
    return dataclasses.replace(sim.unseal, subjects=tuple(ids), sealed_list=sealed_list, allowlist_source=source)


@pytest.mark.parametrize('group', sorted(EXPECTED))
def test_every_counter_example_is_refused_by_the_guard_even_if_a_record_named_it(real, group, sim):
    kind, words = EXPECTED[group]
    unseal = unseal_naming(sim, real.groups[group])
    for subject in real.groups[group]:
        if subject in LPBA40_TEST_REJECT:       # contract C26 revision 11c (decision_log line 558): the expected tightening -- S03-S24 refused by check_subject itself (step "sealed")
            with pytest.raises(SubjectSealed, match='LPBA40_TEST_REJECT'):
                hi.admit(subject, unseal=unseal, allowlist=sim.allowlist, sealed=real.sealed, training=real.training)
            assert (check_subject(subject, sim.allowlist).step, check_subject(subject, sim.allowlist).item) == ('sealed', 'lpba40 test set')
            continue
        with pytest.raises(kind, match=words):
            hi.admit(subject, unseal=unseal, allowlist=sim.allowlist, sealed=real.sealed, training=real.training)


def test_the_hold_out_is_admitted_only_through_a_valid_record(real):
    """Diff review n4: without a record every B-line hold-out subject is refused; with BZ's committed record (the unseal branch) exactly its subjects are admitted through it, and through nothing else.  C23.9-13 (4):
    on the unseal branch the record's allowlist must be bound to it (bayesreg.io.unseal.bind_unseal); unbound, the sealed guard refuses every one of them."""
    ids = real.groups['holdout_B']
    for subject in ids:
        with pytest.raises(hi.HIRefused, match='no unseal record'):
            hi.admit(subject, unseal=None, allowlist=real.allowlist, sealed=real.sealed, training=real.training)
    if RECORD:
        unseal = hi.load_unseal(ROOT/hi.UNSEAL_REL)
        bound, _ = hi.bind(unseal, real.allowlist)
        assert sorted(unseal.subjects) == ids and all(hi.admit(s, unseal=unseal, allowlist=bound, sealed=real.sealed, training=real.training)['sealed_list'] == 'holdout_B_v1' for s in ids)
        for subject in ids:
            with pytest.raises(SubjectSealed, match='without an unseal record'):
                hi.admit(subject, unseal=unseal, allowlist=real.allowlist, sealed=real.sealed, training=real.training)
    else:
        for subject in ids:
            with pytest.raises(SubjectSealed, match='not on the development allowlist'):
                hi.allowlist_guard(subject, real.allowlist)


def test_admit_refuses_an_allowlist_without_this_records_unseal_object(real, sim):
    """SG code review m9 (hi_core.admit, `allowlist.unseal is unseal.base`): for a subject the guard itself lets through (a development subject), the unbound
    allowlist and an allowlist bound to another unseal object of the same record are refused with HIRefused, before HI's own layers."""
    other = hi.load_unseal(sim.repo/hi.UNSEAL_REL)                                   # the same committed record, another object
    other_bound, _ = hi.bind(other, sim.allowlist_unbound)
    assert other.base is not sim.unseal.base
    for allowlist in (sim.allowlist_unbound, other_bound):
        with pytest.raises(hi.HIRefused, match="does not carry this record's unseal object"):
            hi.admit('OASIS_OAS1_0438_MR1', unseal=sim.unseal, allowlist=allowlist, sealed=real.sealed, training=real.training)      # a development subject of the real allowlist


@pytest.mark.parametrize('group', sorted(EXPECTED))
def test_the_entry_refuses_each_counter_example_before_any_image(real, group, tmp_path, monkeypatch, decided, sim):
    """The whole entry on the simulated unseal branch's allowlist and the real sealed lists, the record check replaced by one naming the subjects (built on a real unseal object, C23.9-13 (2); load_unseal itself
    is tested below): refused at the admission, no image path asked, no file of a subject opened, no output directory made, no lock taken."""
    spy = Spy(monkeypatch)
    ids = [s for s in real.groups[group] if s][:6]
    monkeypatch.setattr(hi, 'load_unseal', lambda *args, **kwargs: unseal_naming(sim, ids))
    out = tmp_path/'out'
    with pytest.raises((SubjectSealed, hi.HIRefused)) as refused:
        hi.infer_holdout(ids, unseal_path=tmp_path/'none.json', out_root=out, files=spy.files(tmp_path/'no_data'), device='cpu', allowlist=sim.allowlist_unbound, allowlist_record={}, sealed=real.sealed,
                         training=real.training, registry=real.registry, models_root=tmp_path/'no_members', grouping=load_grouping(DEFAULT_GROUPS_PATH, 'structure13_v1'), code={}, command=[], repo=tmp_path, log=quiet)
    assert isinstance(refused.value, SubjectSealed) or refused.value.stage == 'admission'
    assert spy.nothing_read() and not out.exists() and not Path(hi.HOLDOUT_LOCK).exists()
    assert not [p for p in spy.opened if any(s in p for s in ids)]


def test_d24_refuses_every_training_subject_for_v4(real):
    """C22.1-2 修订 4 (成员): every v4 member saw the 106 training subjects; select_fold_models (through hs_seg.uf_plan) refuses them before any member file is needed."""
    for subject in real.training:
        with pytest.raises(ModelSelectionRefused, match='saw the labels'):
            hi.plan_of(real.registry, subject, real.allowlist, {})


def test_the_census_of_the_dry_run_refuses_every_group(real):
    census = hi.guard_census(real.groups, allowlist=real.allowlist, sealed=real.sealed, training=real.training, registry=real.registry)
    for group, counts in census.items():
        assert sum(counts.values()) == len(real.groups[group]) and not [k for k in counts if k.startswith('admitted')], (group, counts)
    assert census['training'] == {'training set + D24 (ModelSelectionRefused)': 106} and census['T0'] == {'never_for_training (T0)': 24}
    # diff review n4: the census uses no record.  C23.9-13 (1): on the unseal branch too the hold-out is now refused by the first guard (the sealed guard needs the bound record), not by HI's sealed lists
    assert census['holdout_B'] == {'allowlist guard (SubjectSealed)': 100}
    # contract C26 revision 11c (decision_log line 558): the expected tightening -- of the 24 LPBA40 ids S03-S24 are refused by the first guard (check_subject step "sealed"), S01-S02 as before
    assert census['E11'] == {'allowlist sources': 10} and census['LPBA40'] == {'allowlist sources': 2, 'allowlist guard (SubjectSealed)': 22}
    assert all(check_subject(s, real.allowlist).step == 'sealed' for s in real.groups['LPBA40'] if s in LPBA40_TEST_REJECT)       # revision 11c: the 22, by step "sealed"


def test_a_wrongly_allowlisted_h1_subject_is_stopped_by_the_sealed_lists(w):
    """The second lock: an allowlist that wrongly holds an H1 subject in the unsealed source.  C23.9-13 (3): since SG the first guard (bayesreg's sealed guard) stops it itself -- refused earlier than HI's sealed
    lists, unbound or bound to the record (H1-H4 never); the record cannot be bound to the mistaken allowlist; the record's own subject passes on the allowlist bound to it."""
    document = synthetic.allowlist_document([('synthetic_unsealed', SUBJECTS+H1, True), ('synthetic_train', CONTACT, False)])
    allowlist = load_dev_allowlist(write_json(w.tmp/'mistaken.json', document))
    with pytest.raises(SubjectSealed, match=r"sealed list \['H1'\]"):
        hi.allowlist_guard(H1[0], allowlist)                    # the first guard refuses it now
    with pytest.raises(SubjectSealed, match=r"sealed list \['H1'\]"):
        hi.admit(H1[0], unseal=w.hi_unseal, allowlist=allowlist, sealed=w.sealed, training=w.training)
    with pytest.raises(hi.HIRefused, match='cannot be bound'):
        hi.bind(w.hi_unseal, allowlist)
    assert hi.admit(SUBJECTS[0], unseal=w.hi_unseal, allowlist=w.allowlist, sealed=w.sealed, training=w.training)['sealed_list'] == 'holdout_B_v1'


def test_a_subject_not_in_the_record_is_refused(w, monkeypatch):
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match='not one of the 2 subjects of the unseal record'):
        hi.infer_holdout([SUBJECTS[0], DEV[0]], **holdout_kwargs(w, out_root=w.tmp/'out', files=spy.files(w.data)))
    assert spy.nothing_read() and not (w.tmp/'out').exists()


@pytest.mark.parametrize('path', [None, 'missing'])
def test_no_unseal_record_refuses_before_anything(w, monkeypatch, path):
    if path == 'missing':
        w.unseal_path.unlink()
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused) as refused:
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out', files=spy.files(w.data), unseal_path=None if path is None else w.unseal_path))
    assert refused.value.stage == 'unseal' and spy.nothing_read() and not (w.tmp/'out').exists()


def test_the_backend_of_the_hold_out_is_hs_a_only(w, monkeypatch):
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match="backend: 'segment_subject'"):
        hi.infer_holdout(None, backend='segment_subject', **holdout_kwargs(w, out_root=w.tmp/'out', files=spy.files(w.data)))
    assert spy.opened == [] and spy.loads == []


# ---- the unseal record's content (the git checks: test_hi_once.py) -------------------------------------------------------------------------------
# C23.9-13 (6): the refusals come from bayesreg's loader now (wrapped in HIRefused('unseal')); the words of `match=` follow it and still name the item that departs.  D98: written_by takes "BZ" and
# "lead on BZ's behalf" (then with BZ's words); bz_words is one line without '|'.
DEPARTURES = dict(
    schema=(lambda d: d.update(schema='bayesreg.hi.unseal.v0'), 'schema'),
    written_by=(lambda d: d.update(written_by='lead'), r"written_by 'lead' is not one of"),
    written_by_lead_without_bz_words=(lambda d: d.update(written_by="lead on BZ's behalf"), 'bz_words must contain BZ'),
    bz_words=(lambda d: d.update(bz_words='  '), 'bz_words'),
    bz_words_two_lines=(lambda d: d.update(bz_words='（合成测试）第一行\n第二行'), 'bz_words: one non-empty line'),
    bz_words_bar=(lambda d: d.update(bz_words='（合成测试）a | b'), 'bz_words: one non-empty line'),
    time_local=(lambda d: d.update(time_local=''), 'time_local'),
    purpose=(lambda d: d.update(purpose=None), 'purpose'),
    utc_form=(lambda d: d.update(utc='2026-11-07 11:30'), 'YYYY-MM-DDTHH:MM:SSZ'),
    utc_not_real=(lambda d: d.update(utc='2026-02-30T00:00:00Z'), 'not a real time'),
    utc_future=(lambda d: d.update(utc='2030-01-01T00:00:00Z'), 'in the future'),
    utc_before_the_draw=(lambda d: d.update(utc='2026-10-01T00:00:00Z'), 'before the draw'),
    sealed_list_H1=(lambda d: d.update(sealed_list='H1'), "sealed_list 'H1' is not one that may be unsealed"),
    sealed_list_unknown=(lambda d: d.update(sealed_list='holdout_C'), "sealed_list 'holdout_C' is not one that may be unsealed"),
    subjects_file=(lambda d: d.update(subjects_file='lists/other.txt'), 'subjects_file'),
    subjects_file_sha256=(lambda d: d.update(subjects_file_sha256='1'*64), 'subjects_file_sha256'),
    subjects_sha256=(lambda d: d.update(subjects_sha256=subject_list_sha256(SUBJECTS[:1])), 'subjects_sha256'),
    n_subjects=(lambda d: d.update(n_subjects=3), 'n_subjects'),
    n_subjects_float=(lambda d: d.update(n_subjects=2.0), 'n_subjects'),
    n_subjects_bool=(lambda d: d.update(n_subjects=True), 'n_subjects'),
    allowlist_sha256=(lambda d: d.update(allowlist_sha256='2'*64), 'allowlist_sha256'),
    allowlist_source_missing=(lambda d: d.update(allowlist_source='nowhere'), 'no source'),
    allowlist_source_not_the_list=(lambda d: d.update(allowlist_source='synthetic_train'), 'does not hold exactly'),
    allowlist_source_trainable=(lambda d: d.update(allowlist_source='synthetic_dev'), 'must be never_for_training'),
    segmenter_id=(lambda d: d.update(segmenter_id='unet35_v1_ens'), "anchor's"),
    registry_sha256=(lambda d: d.update(registry_sha256='3'*64), "anchor's"),
    extra_key=(lambda d: d.update(note='x'), 'exactly the keys'),
    missing_key=(lambda d: d.pop('purpose'), 'exactly the keys'),
)


@pytest.mark.parametrize('name', sorted(DEPARTURES))
def test_every_departure_of_the_unseal_record_is_refused(w, monkeypatch, name):
    """Each record committed and logged, so the refusal is the content's, not the git check's."""
    change, words = DEPARTURES[name]
    document = dict(w.unseal)
    change(document)
    write_unseal(w, document)
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match=words) as refused:
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out', files=spy.files(w.data)))
    assert refused.value.stage == 'unseal' and spy.nothing_read() and not (w.tmp/'out').exists()


def test_the_record_must_be_named_in_the_decision_log(w):
    write_unseal(w, dict(w.unseal, bz_words='（合成测试）另一份原话'), log=False)
    with pytest.raises(hi.HIRefused, match='does not name the record'):
        hi.load_unseal(w.unseal_path, anchor=w.anchor)
    log_record(w.repo, w.unseal_path, words='（合成测试）别的原话')          # D98: the sha256 is there, but not on one line with bz_words
    commit_all(w.repo)
    with pytest.raises(hi.HIRefused, match='holds both the record'):
        hi.load_unseal(w.unseal_path, anchor=w.anchor)


@pytest.mark.parametrize('writer, words, passes', [('BZ', '（合成测试）解封合成留出集', True), ("lead on BZ's behalf", '（合成测试）BZ 原话：解封记录你代我写；BZ 过目：同意', True),
                                                   ("lead on BZ's behalf", '（合成测试）BZ 原话：解封记录你帮我写', False), ('lead', '（合成测试）解封记录你代我写', False)])
def test_written_by_takes_bz_and_the_lead_on_bzs_behalf_with_bzs_words(w, writer, words, passes):
    """D98 (C23.3-4, C23.8-1): "BZ", and "lead on BZ's behalf" whose bz_words hold "解封记录你代我写" word for word, pass; another wording or another writer is refused, nothing of a subject read."""
    write_unseal(w, dict(w.unseal, written_by=writer, bz_words=words))
    if passes:
        unseal = hi.load_unseal(w.unseal_path, anchor=w.anchor)
        assert unseal.document['written_by'] == writer and unseal.record()['written_by'] == writer and unseal.base.written_by == writer
    else:
        with pytest.raises(hi.HIRefused, match='written_by|bz_words') as refused:
            hi.load_unseal(w.unseal_path, anchor=w.anchor)
        assert refused.value.stage == 'unseal'


def test_the_unseal_record_refuses_duplicate_keys_another_path_a_changed_list_and_a_shared_subject(w):
    text = w.unseal_path.read_text(encoding='utf-8').replace('"purpose"', '"written_by": "BZ", "purpose"', 1)
    w.unseal_path.write_text(text, encoding='utf-8')
    log_record(w.repo, w.unseal_path, words=w.unseal['bz_words'])
    commit_all(w.repo)
    with pytest.raises(hi.HIRefused, match='duplicate'):
        hi.load_unseal(w.unseal_path, anchor=w.anchor)
    write_unseal(w, w.unseal)
    assert hi.load_unseal(w.unseal_path, anchor=w.anchor).subjects == tuple(sorted(SUBJECTS))
    other = write_json(w.repo/'splits/bayesreg_v1/unseal_holdout_B_v2.json', w.unseal)
    commit_all(w.repo)
    for path in (other, write_json(w.tmp/'outside.json', w.unseal)):
        with pytest.raises(hi.HIRefused, match='not the one path of the record'):
            hi.load_unseal(path, anchor=w.anchor)
    # C23.9-13 (5): the shared allowlist goes where bayesreg's loader reads the allowlist (the pinned path of the tree), with the record naming its sha256
    shared = load_dev_allowlist(write_json(w.repo/'configs'/'bayesreg'/'dev_subjects.json', synthetic.allowlist_document([('synthetic_unsealed', SUBJECTS, True), ('synthetic_dev', SUBJECTS[:1]+DEV, False)])))
    write_unseal(w, dict(w.unseal, allowlist_sha256=shared.sha256))
    with pytest.raises(hi.HIRefused, match='also in other allowlist sources'):
        hi.load_unseal(w.unseal_path, anchor=w.anchor)
    (w.repo/'lists'/'holdout.txt').write_text(f'{SUBJECTS[0]}\n', encoding='utf-8')
    with pytest.raises(hi.HIRefused, match='not the pinned sha256'):
        hi.load_unseal(w.unseal_path, anchor=w.anchor)


def test_the_real_unsealable_list_is_the_pinned_b_line_hold_out():
    spec = {s['name']: s for s in hi.SEALED_LISTS}['holdout_B_v1']
    assert spec['path'] == 'splits/bayesreg_v1/holdout_B_subjects.txt' and spec['count'] == 100 and file_sha256(ROOT/spec['path']) == spec['sha256']


# ---- outputs ------------------------------------------------------------------------------------------------------------------------------------
def test_an_output_directory_that_exists_is_refused_before_any_image(w, monkeypatch):
    out = w.tmp/'out'
    out.mkdir()
    spy = Spy(monkeypatch)
    with pytest.raises(FileExistsError, match='the first run writes a new root'):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=out, files=spy.files(w.data)))
    assert spy.nothing_read() and list(out.iterdir()) == [] and not Path(hi.HOLDOUT_LOCK).exists()


def test_an_output_directory_outside_the_results_or_inside_the_code_tree_is_refused(w, monkeypatch):
    """Review m6."""
    spy = Spy(monkeypatch)
    monkeypatch.setattr(hi, 'RESULTS', w.tmp/'results_only')
    with pytest.raises(hi.HIRefused, match='output: .*must be under'):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out', files=spy.files(w.data)))
    monkeypatch.setattr(hi, 'RESULTS', ROOT.parent)
    monkeypatch.setattr(hi, 'HOLDOUT_OUT_SUGGESTED', ROOT/'hi_out_never_made')       # A5a: the one root, put in the code tree to reach the location check
    with pytest.raises(hi.HIRefused, match='outside the code tree'):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=ROOT/'hi_out_never_made', files=spy.files(w.data)))
    assert spy.nothing_read() and not (w.tmp/'out').exists() and not (ROOT/'hi_out_never_made').exists() and not Path(hi.HOLDOUT_LOCK).exists()


def test_infer_holdout_takes_only_the_one_root(w, monkeypatch):
    """A5a review I2 point 1: the --out check is in infer_holdout too (stage output, first of all): another root is refused before the code identity, the record or anything of a subject."""
    spy = Spy(monkeypatch)
    monkeypatch.setattr(hi, 'pinned_code_check', lambda: (_ for _ in ()).throw(AssertionError('rule one ran')))
    with pytest.raises(hi.HIRefused, match='is not') as refused:
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'elsewhere', files=spy.files(w.data)))
    assert refused.value.stage == 'output' and spy.nothing_read() and not (w.tmp/'elsewhere').exists() and not Path(hi.HOLDOUT_LOCK).exists()


def test_a_failure_stops_the_run_keeps_its_evidence_and_writes_hi_stopped(w, monkeypatch):
    real_write = hi.hs_seg.write_prediction
    calls = []

    def failing(subject, *args, **kwargs):
        calls.append(subject)
        if len(calls) == 2:
            raise hi.hs_seg.HSSegFailed(f'{subject}: the reader (made up by the test)')
        return real_write(subject, *args, **kwargs)
    monkeypatch.setattr(hi.hs_seg, 'write_prediction', failing)
    with pytest.raises(hi.hs_seg.HSSegFailed):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out'))
    stopped = json.loads((w.tmp/'out'/hi.STOPPED_RECORD).read_text(encoding='utf-8'))
    assert stopped['subject'] == SUBJECTS[1] and stopped['done'] == [SUBJECTS[0]] and not (w.tmp/'out'/hi.INFER_RECORD).exists() and (w.tmp/'out'/'pred'/SUBJECTS[0]/'provenance.json').is_file()
    assert Path(hi.HOLDOUT_LOCK).is_file()


# ---- the development check -------------------------------------------------------------------------------------------------------------------------
def dev_world(tmp_path, *, perturb=None):
    """SUBJECTS as development subjects, their 'registered' predictions written by hs_a's writer (hi.predict_hs_a), optionally perturbed (the outputs' sha256 in the provenance follow)."""
    v = world(tmp_path, role='dev')
    reference = tmp_path/'registered'
    available = hi.member_files(v.registry, v.models_root)
    for subject in SUBJECTS:
        plan, models, policy = hi.plan_of(v.registry, subject, v.allowlist, available)
        hi.predict_hs_a(subject, files=v.files, registry=v.registry, models=models, plan=plan, policy=policy, device=torch.device('cpu'), grouping=v.grouping,
                        grouping_check=dict(grouping_status='default', grouping_status_admitted=True, allow_unverified_grouping=False), pred_root=reference, code=dict(commit='r'*40), environment=dict(torch='x'),
                        command=['test'], allowlist=v.allowlist, hi=dict(backend='hs_a', purpose='test reference'))
        if perturb is not None and subject == SUBJECTS[0]:
            directory = reference/subject
            p = load_p(directory/'probs36.npz')
            perturb(p)
            np.savez_compressed(directory/'probs36.npz', p=p)
            prov = json.loads((directory/'provenance.json').read_text(encoding='utf-8'))
            prov['output_sha256']['probs36.npz'] = file_sha256(directory/'probs36.npz')
            write_json(directory/'provenance.json', prov)
    v.reference = reference
    return v


def devcheck_of(v, backends, out, *, subjects=SUBJECTS, sealed=None, log=quiet):
    return hi.devcheck(list(subjects), backends=backends, out_root=out, files=v.files, device='cpu', allowlist=v.allowlist, allowlist_record={}, sealed=v.sealed if sealed is None else sealed, training=v.training,
                       registry=v.registry, models_root=v.models_root, grouping=v.grouping, reference_root=v.reference, code=dict(commit='d'*40), command=['test'], anchor=v.anchor,
                       expected_subjects=list(subjects), log=log)


def test_the_development_check_finds_both_backends_bitwise_equal_and_reads_select(tmp_path, monkeypatch):
    v = dev_world(tmp_path)
    spy = Spy(monkeypatch)
    record = devcheck_of(v, ['hs_a', 'segment_subject'], hi.DEVCHECK_OUT)
    for backend in ('hs_a', 'segment_subject'):
        s = record['summary'][backend]
        assert s['all_bitwise_equal'] is True and s['n_probs36_bitwise_equal'] == s['n_groups_bitwise_equal'] == s['n_probs36_argmax_equal'] == s['n_image_same_as_registered'] == 2 and s['max_elements_differ'] == 0
        assert s['min_argmax_agreement'] == 1.
    reading = record['reading']
    assert reading['text'] == '逐位相同 2/2：选 hs_a' and reading['n_bitwise_equal'] == 2 and reading['min_probs36_argmax_agreement'] == 1. and reading['plan']['sha256'] == hi.DEVCHECK_PLAN_SHA256
    assert sorted(p.name for p in (hi.DEVCHECK_OUT/'subjects').iterdir()) == sorted(f'{b}__{s}.json' for b in ('hs_a', 'segment_subject') for s in SUBJECTS)
    assert record['decision']['made'] is False and (hi.DEVCHECK_OUT/hi.DEVCHECK_RECORD).is_file() and (hi.DEVCHECK_OUT/'segment_subject'/'pred'/SUBJECTS[0]/'probs36.npz').is_file()
    assert not (hi.DEVCHECK_OUT/'hs_a').exists()                    # hs_a compares in memory
    assert spy.labels() == [] and sorted(set(spy.loads)) == sorted(str(v.data/s/'aligned_norm.nii.gz') for s in SUBJECTS)


def test_the_development_check_runs_once(tmp_path, monkeypatch):
    """Diff review R1: the fixed lock DEVCHECK_LOCK is taken with O_EXCL just before the first image; a second check, into any directory, is refused before anything is read; the lock is never changed."""
    v = dev_world(tmp_path)
    devcheck_of(v, ['hs_a'], hi.DEVCHECK_OUT)
    lock = Path(hi.DEVCHECK_LOCK)
    line = hi.lock_lines(lock)[0]
    assert line['out'] == str(hi.DEVCHECK_OUT) and line['plan_sha256'] == hi.DEVCHECK_PLAN_SHA256 and line['backends'] == ['hs_a'] and line['event'] == hi.FIRST_RUN and line['pid'] > 0
    before = (lock.read_bytes(), lock.stat().st_mtime_ns)
    spy = Spy(monkeypatch)
    for out in (tmp_path/'check2', hi.DEVCHECK_OUT):
        with pytest.raises(hi.HIRefused, match='runs once') as refused:
            devcheck_of(v, ['hs_a', 'segment_subject'], out)
        assert refused.value.stage == 'once'
    assert spy.nothing_read() and not [p for p in spy.opened if not p.endswith('hi_devcheck.lock')] and not (tmp_path/'check2').exists() and (lock.read_bytes(), lock.stat().st_mtime_ns) == before
    record = json.loads((hi.DEVCHECK_OUT/hi.DEVCHECK_RECORD).read_text(encoding='utf-8'))
    assert record['lock'] == dict(path=str(lock), line=line) and record['machine_gate']['own_units'] == list(hi.DEVCHECK_UNITS)      # diff review 2, s2


def test_the_development_check_refuses_another_directory_at_the_api(tmp_path, monkeypatch):
    """Diff review 2 (optional): hi_core.devcheck itself takes only DEVCHECK_OUT; another directory is refused before anything is read and the lock is not spent."""
    v = dev_world(tmp_path)
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match='is not') as refused:
        devcheck_of(v, ['hs_a'], tmp_path/'elsewhere')
    assert refused.value.stage == 'output' and spy.nothing_read() and not (tmp_path/'elsewhere').exists() and not Path(hi.DEVCHECK_LOCK).exists()


def test_a_busy_machine_stops_the_check_before_its_lock(tmp_path, monkeypatch):
    """Diff review 2, s2: the machine gate runs before the GPU and before the one lock; its own unit is tag-hi-devcheck (only when this process runs in it)."""
    v = dev_world(tmp_path)
    calls = []

    def busy(**kwargs):
        calls.append(kwargs)
        raise hi.HIRefused('machine', 'other units are active (made up by the test)')
    monkeypatch.setattr(hi, 'machine_gate', busy)
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match='other units are active') as refused:
        devcheck_of(v, ['hs_a'], hi.DEVCHECK_OUT)
    assert refused.value.stage == 'machine' and calls == [dict(own_units=hi.DEVCHECK_UNITS)] and spy.nothing_read()
    assert not Path(hi.DEVCHECK_LOCK).exists() and not hi.DEVCHECK_OUT.exists()


def test_the_spy_sees_every_way_of_opening(tmp_path, monkeypatch):
    """Diff review 2, s3: pathlib's reads (bound to io.open at import in Python 3.10) are seen as well as builtins.open, io.open, gzip and np.load."""
    import gzip
    for name in ('a.txt', 'b.bin', 'c.txt', 'd.txt', 'e.gz', 'f.npy', 'g.txt'):
        (tmp_path/name).write_bytes(b'x')
    with gzip.open(tmp_path/'e.gz', 'wb') as stream:
        stream.write(b'x')
    np.save(tmp_path/'f.npy', np.zeros(2))
    spy = Spy(monkeypatch)
    (tmp_path/'a.txt').read_text()
    (tmp_path/'b.bin').read_bytes()
    with (tmp_path/'c.txt').open() as stream:
        stream.read()
    with open(tmp_path/'d.txt') as stream:
        stream.read()
    with gzip.open(tmp_path/'e.gz') as stream:
        stream.read()
    np.load(tmp_path/'f.npy')
    import io as io_module
    with io_module.open(tmp_path/'g.txt') as stream:
        stream.read()
    seen = {Path(p).name for p in spy.opened}
    assert {'a.txt', 'b.bin', 'c.txt', 'd.txt', 'e.gz', 'f.npy', 'g.txt'} <= seen, seen
    assert REAL['devcheck_lock'] == Path('/home/rmcfz/seq_vp_tag_results/seg/hi_devcheck.lock') and REAL['devcheck_out'] == Path('/home/rmcfz/seq_vp_tag_results/seg/hi_devcheck_v1')


def test_both_runs_hs_a_on_every_subject_first_and_a_failure_of_segment_subject_leaves_the_reading(tmp_path, monkeypatch):
    """Diff review n3."""
    v = dev_world(tmp_path)
    real = hi.devcheck_subject
    calls = []

    def subject_check(subject, backend, **kwargs):
        calls.append((backend, subject))
        if backend == 'segment_subject' and subject == SUBJECTS[1]:
            raise OSError('the disk is full (made up by the test)')
        return real(subject, backend, **kwargs)
    monkeypatch.setattr(hi, 'devcheck_subject', subject_check)
    record = devcheck_of(v, ['segment_subject', 'hs_a'], hi.DEVCHECK_OUT)
    assert calls == [('hs_a', s) for s in SUBJECTS]+[('segment_subject', s) for s in SUBJECTS] and record['backends'] == ['hs_a', 'segment_subject']
    assert record['reading']['text'] == '逐位相同 2/2：选 hs_a' and sorted(record['rows']['segment_subject']) == [SUBJECTS[0]]
    assert record['described_failures']['segment_subject']['subject'] == SUBJECTS[1] and (hi.DEVCHECK_OUT/'devcheck_segment_subject_stopped.json').is_file()
    assert not (hi.DEVCHECK_OUT/'devcheck_stopped.json').exists() and (hi.DEVCHECK_OUT/hi.DEVCHECK_RECORD).is_file()


def test_a_failure_of_hs_a_stops_the_check_and_reads_undecided(tmp_path, monkeypatch):
    v = dev_world(tmp_path)
    real = hi.devcheck_subject

    def subject_check(subject, backend, **kwargs):
        if backend == 'hs_a' and subject == SUBJECTS[1]:
            raise OSError('made up by the test')
        return real(subject, backend, **kwargs)
    monkeypatch.setattr(hi, 'devcheck_subject', subject_check)
    with pytest.raises(OSError):
        devcheck_of(v, ['hs_a', 'segment_subject'], hi.DEVCHECK_OUT)
    stopped = json.loads((hi.DEVCHECK_OUT/'devcheck_stopped.json').read_text(encoding='utf-8'))
    assert stopped['reading'] == hi.READ_UNDECIDED and stopped['backend'] == 'hs_a' and stopped['done'] == [SUBJECTS[0]] and not (hi.DEVCHECK_OUT/hi.DEVCHECK_RECORD).exists()
    assert not (hi.DEVCHECK_OUT/'segment_subject').exists()          # the described backend never started


def ulp_off_argmax(p):
    c = int(p[:, 0, 0, 0].astype(np.float64).argmin())           # the smallest channel: one ULP up cannot reach the argmax
    p[c, 0, 0, 0] = np.nextafter(p[c, 0, 0, 0], np.float16(1.), dtype=np.float16)


def argmax_flip(p):
    column = p[:, 1, 1, 1].copy()
    top, second = np.argsort(column.astype(np.float64))[-1], np.argsort(column.astype(np.float64))[-2]
    assert column[top] != column[second]
    p[top, 1, 1, 1], p[second, 1, 1, 1] = column[second], column[top]


@pytest.mark.parametrize('perturb, n_differ, argmax_equal', [(ulp_off_argmax, 1, True), (argmax_flip, 2, False)])
def test_the_development_check_counts_what_differs_and_reads_pending(tmp_path, perturb, n_differ, argmax_equal):
    v = dev_world(tmp_path, perturb=perturb)
    record = devcheck_of(v, ['hs_a'], hi.DEVCHECK_OUT)
    row = record['rows']['hs_a'][SUBJECTS[0]]['probs36']
    assert row['bitwise_equal'] is False and row['n_elements_differ'] == n_differ and row['argmax_equal'] is argmax_equal and row['max_abs_difference'] > 0
    assert row['n_voxels_argmax_differ'] == (0 if argmax_equal else 1) and row['argmax_agreement'] == 1.-row['n_voxels_argmax_differ']/row['n_voxels']
    reading = record['reading']
    assert reading['text'] == hi.READ_PENDING and reading['per_subject'][SUBJECTS[0]]['probs36_elements_differ'] == n_differ and reading['per_subject'][SUBJECTS[1]]['probs36_bitwise_equal'] is True
    assert reading['min_probs36_argmax_agreement'] == row['argmax_agreement']       # argmax equal with other bits is pending too (devcheck_plan.md §3)


def test_without_hs_a_the_reading_is_undecided(tmp_path):
    v = dev_world(tmp_path)
    assert devcheck_of(v, ['segment_subject'], hi.DEVCHECK_OUT)['reading']['text'] == hi.READ_UNDECIDED


def test_the_plan_is_pinned_and_a_changed_plan_stops_the_check_before_anything(tmp_path, monkeypatch):
    assert file_sha256(ROOT/hi.DEVCHECK_PLAN_REL) == hi.DEVCHECK_PLAN_SHA256
    text = (ROOT/hi.DEVCHECK_PLAN_REL).read_text(encoding='utf-8')
    assert all(words in text for words in ('逐位相同 20/20：选 hs_a', '不过（待查）', '不判', '在真实运行之前写定'))
    v = dev_world(tmp_path)
    monkeypatch.setattr(hi, 'DEVCHECK_PLAN_SHA256', '0'*64)
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused) as refused:
        devcheck_of(v, ['hs_a'], hi.DEVCHECK_OUT)
    assert refused.value.stage == 'plan' and spy.loads == [] and not (hi.DEVCHECK_OUT).exists()


def test_the_development_check_refuses_another_registration_before_any_image(tmp_path, monkeypatch):
    v = dev_world(tmp_path)
    prov_path = v.reference/SUBJECTS[1]/'provenance.json'
    prov = json.loads(prov_path.read_text(encoding='utf-8'))
    prov['members'] = prov['members'][::-1]
    write_json(prov_path, prov)
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match="reference: .*not HS's plan"):
        devcheck_of(v, ['hs_a'], hi.DEVCHECK_OUT)
    assert spy.loads == [] and not (hi.DEVCHECK_OUT).exists()


@pytest.mark.parametrize('case', ['T0', 'training', 'sealed'])
def test_the_development_check_refuses_t0_training_and_sealed_subjects_before_any_image(tmp_path, monkeypatch, case):
    """Review m9: through expected_subjects (the tests' door), a T0 subject, a training subject and a subject on a sealed list are each refused by the check's own admission; nothing is read."""
    v = dev_world(tmp_path)
    subjects = dict(T0=T0, training=CONTACT[:1], sealed=SUBJECTS[:1])[case]
    sealed = SealedLists({'H1': frozenset(SUBJECTS[:1])}, []) if case == 'sealed' else None
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match='on a sealed list, never_for_training or a training subject') as refused:
        devcheck_of(v, ['hs_a'], hi.DEVCHECK_OUT, subjects=subjects, sealed=sealed)
    assert refused.value.stage == 'admission' and spy.nothing_read() and not (hi.DEVCHECK_OUT).exists()


def test_the_development_check_takes_only_the_twenty_development_subjects(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch=monkeypatch)
    with pytest.raises(hi.HIRefused, match='20 development subjects'):
        hi.devcheck(SUBJECTS, backends=['hs_a'], out_root=hi.DEVCHECK_OUT, files=w.files, device='cpu', allowlist=w.allowlist, allowlist_record={}, sealed=w.sealed, training=w.training, registry=w.registry,
                    models_root=w.models_root, grouping=w.grouping, reference_root=w.tmp, code={}, command=[], anchor=w.anchor, log=quiet)
    assert list(hi.DEV_SUBJECTS) == [f'OASIS_OAS1_{n:04d}_MR1' for n in range(438, 458)]


def test_compare_on_arrays():
    rng = np.random.default_rng(0)
    a = rng.random((4, 3, 2, 2)).astype(np.float16)
    assert hi.compare(a, a.copy())['bitwise_equal'] is True
    b = a.copy()
    b[0, 2, 1, 1] = -b[0, 2, 1, 1]
    r = hi.compare(a, b)
    assert r['n_elements_differ'] == 1 and r['bitwise_equal'] is False and r['max_abs_difference'] == pytest.approx(2*float(a[0, 2, 1, 1]))
    assert hi.compare(a, b.astype(np.float32))['comparable'] is False


# ---- the dry runs on synthetic inputs ------------------------------------------------------------------------------------------------------------
def test_the_dry_runs_open_no_image_and_pass_on_synthetic_inputs(tmp_path, monkeypatch, decided):
    v = dev_world(tmp_path)
    u = types.SimpleNamespace(**vars(world(tmp_path/'holdout', monkeypatch=monkeypatch)))      # pins bayesreg into u's repository (C23.9-13 (5))
    inputs = types.SimpleNamespace(code=dict(kind='git_worktree', commit='e'*40, clean=True, root=str(ROOT), root_check=dict(checked='(test)', root=str(ROOT))), allowlist=v.allowlist,
                                   allowlist_record=dict(allowlist_sources_verified={'x': 'y'}), training=v.training, sealed=v.sealed, registry=v.registry, grouping=v.grouping, files=v.files, anchor=v.anchor)
    monkeypatch.setattr(run_hi, 'real_inputs', lambda: inputs)
    monkeypatch.setattr(run_hi, 'dev_subjects', lambda i: sorted(hi.DEV_SUBJECTS))
    monkeypatch.setattr(hi, 'MODELS_ROOT', v.models_root)
    monkeypatch.setattr(run_hi, 'census_groups', lambda i: dict(synthetic_training=list(CONTACT), synthetic_T0=list(T0), synthetic_H1=list(H1)))
    spy = Spy(monkeypatch)
    lines = []
    o = types.SimpleNamespace(unseal=None, resume=None, out=str(hi.HOLDOUT_OUT_SUGGESTED), subjects_file=None)
    assert run_hi.dry_run_infer(o, out=lines.append) == 0, lines
    assert any('refuses at stage unseal' in line for line in lines) and any('synthetic_training' in line for line in lines) and any('lock' in line and 'absent' in line for line in lines)
    assert any(line.startswith('decision of the development check: filled and it holds') and hi.READ_SELECT.format(n=20) in line for line in lines)
    assert any(line.startswith('the environment check against the decision') and 'runs on cuda' in line for line in lines)
    # A5a review m8: with --record, the record in the shape of every hold-out entry's dry run; rule one without the pinned record is the refusal expected on P0, anything else is unexpected
    passing = hi.pinned_code_check
    monkeypatch.setattr(hi, 'pinned_code_check', lambda: (dict(passed=False, pinned=None), ['runs/holdout_B_v1/pinned_code.json is not committed at HEAD (test): no pinned commit']))
    assert run_hi.dry_run_infer(types.SimpleNamespace(**dict(vars(o), record=str(tmp_path/'rec1'))), out=lines.append) == 9
    doc = json.loads((tmp_path/'rec1'/'dry_run.json').read_text())
    assert doc['passed'] is False and doc['p0_ok'] is True and [e['kind'] for e in doc['expected']] == ['rule_one_unpinned'] and doc['unexpected'] == [] and len(doc['problems']) == 1
    monkeypatch.setattr(hi, 'pinned_code_check', lambda: (dict(passed=False, pinned='p'*40), ['bayesreg/x.py differs from the pinned commit (test)']))
    assert run_hi.dry_run_infer(types.SimpleNamespace(**dict(vars(o), record=str(tmp_path/'rec2'))), out=lines.append) == 9
    doc = json.loads((tmp_path/'rec2'/'dry_run.json').read_text())
    assert doc['p0_ok'] is False and [e['kind'] for e in doc['unexpected']] == ['rule_one'] and doc['expected'] == []
    monkeypatch.setattr(hi, 'pinned_code_check', passing)
    assert run_hi.dry_run_infer(types.SimpleNamespace(**dict(vars(o), record=str(tmp_path/'rec3'))), out=lines.append) == 0
    doc = json.loads((tmp_path/'rec3'/'dry_run.json').read_text())
    assert doc['passed'] is True and doc['problems'] == [] and doc['p0_ok'] is True and doc['summary_schema'] == 'bayesreg.holdout.dry_run_summary.v1'
    # review m1: with a record, every subject of it through the admission and D24, by id
    # C23.9-13 (5): the record is read by bayesreg's loader from the pinned tree (u's repository); the dry run is handed the pinned allowlist unbound and binds it to the record
    monkeypatch.setattr(run_hi, 'real_inputs', lambda: types.SimpleNamespace(**dict(vars(inputs), allowlist=u.allowlist_unbound, sealed=u.sealed, registry=u.registry, anchor=u.anchor)))
    monkeypatch.setattr(hi, 'MODELS_ROOT', u.models_root)
    lines = []
    assert run_hi.dry_run_infer(types.SimpleNamespace(unseal=str(u.unseal_path), resume=None, out=str(hi.HOLDOUT_OUT_SUGGESTED), subjects_file=None), out=lines.append) == 0, lines
    assert any('2 of the 2 subjects of the record pass checks 6 and 7' in line for line in lines) and any('bound to the record' in line for line in lines)
    real_pins(monkeypatch)                                  # the rest of the test is on the development world: bayesreg's pinned tree again
    monkeypatch.setattr(hi, 'DEVCHECK_DECISION', None)
    lines = []
    assert run_hi.dry_run_infer(o, out=lines.append) == 0 and any('DEVCHECK_DECISION is None' in line for line in lines)
    assert spy.nothing_read() and not Path(hi.HOLDOUT_OUT_SUGGESTED).exists()
    # the development check's dry run needs the registered provenance of the 20 subjects: the guard and plans for SUBJECTS stand in (their admission is synthetic)
    monkeypatch.setattr(run_hi, 'real_inputs', lambda: inputs)
    monkeypatch.setattr(hi, 'MODELS_ROOT', v.models_root)
    monkeypatch.setattr(run_hi, 'dev_subjects', lambda i: list(SUBJECTS))
    monkeypatch.setattr(hi, 'REGISTERED_PRED', v.reference)
    lines = []
    assert run_hi.dry_run_devcheck(types.SimpleNamespace(backend='both', out=str(tmp_path/'check')), out=lines.append) == 9      # 9: these are not the 20 development subjects
    assert any('not 0438-0457' in line for line in lines) and any('registered predictions: 2 provenance.json' in line for line in lines) and any(hi.DEVCHECK_PLAN_SHA256 in line for line in lines)
    assert spy.nothing_read() and not (tmp_path/'check').exists()


def test_the_start_strings_carry_the_threads_the_quota_and_a_log_outside_the_output():
    """Review m5."""
    for command in ('infer', 'devcheck'):
        line = run_hi.launch_line(command, '/code')
        assert '--setenv=NUMBA_NUM_THREADS=1' in line and '--property=CPUQuota=800%' in line and f'--unit={run_hi.UNIT[command]}' in line
        assert f'--property=StandardOutput=append:{run_hi.UNIT_LOG[command]}' in line and f'--property=StandardError=append:{run_hi.UNIT_LOG[command]}' in line
    assert not run_hi.UNIT_LOG['infer'].is_relative_to(hi.HOLDOUT_OUT_SUGGESTED) and not run_hi.UNIT_LOG['devcheck'].is_relative_to(hi.DEVCHECK_OUT)


def test_the_infer_dry_run_needs_its_record_directory(monkeypatch, tmp_path):
    """A5a review m8: infer --dry-run writes its record, so the command line requires --record DIR (new or empty, outside the results root and the code trees); --record without --dry-run is a
    usage error.  All refused before anything is read."""
    monkeypatch.setattr(run_hi, 'real_inputs', lambda: (_ for _ in ()).throw(AssertionError('the inputs were read')))
    assert run_hi.main(['infer', '--dry-run']) == 2
    assert run_hi.main(['infer', '--dry-run', '--record', str(hi.RESULTS/'rec')]) == 2 and not (hi.RESULTS/'rec').exists()
    assert run_hi.main(['infer', '--dry-run', '--record', str(ROOT/'runs')]) == 2
    full = tmp_path/'full'
    full.mkdir()
    (full/'x').write_text('x')
    assert run_hi.record_dir_problem(full) is not None and run_hi.record_dir_problem(ROOT/'experiments') is not None
    assert run_hi.main(['infer', '--record', '/somewhere']) == 2


def test_devcheck_takes_only_its_one_directory(monkeypatch, tmp_path):
    """Diff review R1: the decision takes only DEVCHECK_OUT/devcheck.json, so the command refuses another --out before anything (the lock would be spent on a check no decision can take)."""
    monkeypatch.setattr(run_hi, 'real_inputs', lambda: (_ for _ in ()).throw(AssertionError('the inputs were read')))
    assert run_hi.main(['devcheck', '--out', str(tmp_path/'elsewhere')]) == 8 and not (tmp_path/'elsewhere').exists() and not Path(hi.DEVCHECK_LOCK).exists()


def test_run_infer_exits_8_on_a_systemexit(monkeypatch):
    """Review m13."""
    def refuse():
        raise SystemExit('fast_channels refused (made up by the test)')
    monkeypatch.setattr(run_hi, 'real_inputs', refuse)
    assert run_hi.main(['infer', '--unseal', 'x', '--out', str(hi.HOLDOUT_OUT_SUGGESTED)]) == 8 and run_hi.main(['infer', '--unseal', 'x']) == 8
    assert run_hi.main(['infer', '--out', 'y']) == 2


def test_the_entry_documents_the_unseal_format_the_allowlist_steps_and_the_limits():
    text = (ROOT/'runs/20261107-hi/入口说明.md').read_text(encoding='utf-8')
    for key in hi.UNSEAL_KEYS+hi.RESUME_KEYS:
        assert f'`{key}`' in text, key
    for words in (hi.UNSEAL_SCHEMA, hi.RESUME_SCHEMA, '白名单收录', 'never_for_training', '不进 main', '绊线', '代码分不出', hi.UNSEAL_REL, str(REAL['lock']), 'DEVCHECK_DECISION',
                  str(REAL['devcheck_lock']), 'test_nothing_is_decided_in_the_code', '有意的失败即关闭', '至多补跑一次', '/proc/self/cgroup'):
        assert words in text, words


# ---- A5a: rule one and the one output directory of the hold-out entry ----------------------------------------------------------------------------
def test_the_holdout_entry_checks_rule_one_before_the_record_and_its_lock_names_p1(w, monkeypatch):
    """A5a (contract C24.3 rule one; decision_log 2026-11-10 ~2:28 AM CST n8): infer_holdout checks the code against the pinned commit before BZ's record is read; a problem refuses (stage code)
    with nothing read or written; when it passes, the lock line's code_commit is P1 (HEAD beside it in the run's code record)."""
    spy = Spy(monkeypatch)
    monkeypatch.setattr(hi, 'pinned_code_check', lambda: (dict(passed=False, pinned=None), ['runs/holdout_B_v1/pinned_code.json is not committed at HEAD (test)']))
    with pytest.raises(hi.HIRefused) as info:
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out', files=spy.files(w.data)))
    assert info.value.stage == 'code' and 'rule one' in str(info.value)
    assert spy.nothing_read() and not Path(hi.HOLDOUT_LOCK).exists() and not (w.tmp/'out').exists()
    assert not any(str(p).endswith(hi.UNSEAL_REL) for p in spy.opened)                    # the record was not read
    monkeypatch.setattr(hi, 'pinned_code_check', lambda: (dict(passed=True, pinned='a'*40, head='b'*40), []))
    record = hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out'))
    assert record['lock']['line']['code_commit'] == 'a'*40 and record['code']['head'] == 'c'*40 and record['code']['pinned_code']['pinned'] == 'a'*40


def test_the_holdout_entry_takes_only_its_one_output_directory(monkeypatch, tmp_path):
    """A5a (prereg revision 2 list 5 (l); decision_log 2026-11-10 ~1:53 AM CST I4): `infer` writes only into hi_core.HOLDOUT_OUT_SUGGESTED (HE's and XB's pred root): --out omitted or naming it is
    taken, any other refused (exit 8) before anything is read; the dry run reports it as a problem."""
    read = []
    monkeypatch.setattr(run_hi, 'real_inputs', lambda: (read.append(1), (_ for _ in ()).throw(SystemExit('stop here (test)')))[1])
    assert run_hi.main(['infer', '--unseal', 'x', '--out', str(tmp_path/'elsewhere')]) == 8 and read == [] and not (tmp_path/'elsewhere').exists()
    assert run_hi.main(['infer', '--unseal', 'x', '--out', str(hi.HOLDOUT_OUT_SUGGESTED)+'/']) == 8 and read == [1]       # the same directory, written otherwise: taken, then stopped by the test
    assert run_hi.main(['infer', '--unseal', 'x']) == 8 and read == [1, 1]
    assert run_hi.out_problem(None) is None and run_hi.out_problem(str(hi.HOLDOUT_OUT_SUGGESTED)) is None and 'is not' in run_hi.out_problem(str(tmp_path/'elsewhere'))
    assert REAL['holdout_out'].name == 'unet35_v4_ens_holdout_B_v1'
    assert REAL['holdout_out'] == Path('/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_ens_holdout_B_v1')


def test_the_real_rule_one_refuses_this_tree_without_a_pinned_record():
    """main has no runs/holdout_B_v1/pinned_code.json: hi_core's real check (git of this code tree, read only) refuses, never HEAD by default."""
    if (ROOT/'runs/holdout_B_v1/pinned_code.json').exists():
        pytest.skip('a tree with a pinned record (the dedicated branch)')
    record, problems = REAL['pinned_code_check']()
    assert problems and record['pinned'] is None and any('never HEAD' in p for p in problems)
