"""C23.6-2 (SG; revisions 6e, 6f): the guards of scripts/evaluate_l2r_style.evaluate_run (check_subjects), experiments/d0/check_seg4_map.run (sealed_guard) and
experiments/seed/e2e_control.field_process (check_run_subjects).

- evaluate_l2r_style and check_seg4_map: no synthetic admission in the code (revision 6e withdrew it; D104 (a)); every id goes through C23.4-2 on the allowlist.
  The tests pin the allowlist with monkeypatch (C23.9 总则) so that synthetic ids are admitted the normal way, and show that without the pin they are refused.
- e2e_control: the narrow S-class rule of revisions 6f, 6g (a relaxation of the C23.6-2 row, flagged for BZ) -- only this control's own scratch pair: the ids equal
  the module constants SYN_FIXED, SYN_MOVING, the pair PAIR, the scratch marker holds them and this control's nonce, no directory down to the pair is a link or in a
  real data root; any other ids through C23.4-2; a manifest without the ids is refused.
Ids only and files in tmp_path; the real sealed and development ids are read at run time from the pinned lists and the allowlist (ids only)."""
import importlib.util
import json
import pathlib
from pathlib import Path
import sys

import pytest

from bayesreg.io.guard import SubjectSealed, load_dev_allowlist
from bayesreg.io.sealed import lists_of, load_sealed_lists
from bayesreg.seg.unet import synthetic

from . import _sg_world as W

SYN = (W.oasis(985), W.oasis(986))
DEV = ('OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0439_MR1')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def real_ids():
    """(a sealed holdout_B_v1 id, an OASIS id on no list and off the allowlist): ids only, read at run time."""
    lists = load_sealed_lists()
    allowlist = load_dev_allowlist(W.ROOT/'configs/bayesreg/dev_subjects.json')
    sealed = min(lists.subjects['holdout_B_v1'])
    off = next(s for s in (f'OASIS_OAS1_{n:04d}_MR1' for n in range(1, 458)) if not lists_of(lists, s) and not allowlist.lists(s))
    return sealed, off


def synthetic_allowlist(tmp_path, ids=SYN):
    """A synthetic allowlist holding `ids` (the tests' pinned value, C23.9 总则)."""
    return synthetic.write_json(tmp_path/'pinned_allowlist.json', synthetic.allowlist_document([('dev', list(ids), False)]))


def manifest(directory, ids):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    document = dict(shape_zyx=[5, 6, 7], spacing_xyz=[1., 1., 1.])
    if ids is not None:
        document.update(fixed_subject=ids[0], moving_subject=ids[1])
    W.write_json(directory/'manifest.json', document)
    return directory/'manifest.json'


# ---- scripts/evaluate_l2r_style.check_subjects --------------------------------------------------------------------------------------------------------
@pytest.fixture
def l2r():
    return load('_sg_test_evaluate_l2r_style', W.ROOT/'scripts'/'evaluate_l2r_style.py')


def test_l2r_has_no_synthetic_admission(l2r, tmp_path):
    """Revision 6e: synthetic ids under a synthetic root are refused on the repository's allowlist (they are on no allowlist)."""
    assert l2r.ALLOWLIST == W.ROOT/'configs'/'bayesreg'/'dev_subjects.json'
    with pytest.raises(SubjectSealed, match='not on the development allowlist'):
        l2r.check_subjects(manifest(tmp_path/'run'/'inputs'/'p', SYN))


def test_l2r_synthetic_ids_pass_through_the_guard_on_a_pinned_allowlist(l2r, tmp_path, monkeypatch):
    monkeypatch.setattr(l2r, 'ALLOWLIST', synthetic_allowlist(tmp_path))
    record = l2r.check_subjects(manifest(tmp_path/'run'/'inputs'/'p', SYN))
    assert record['guard'] == 'dev-allowlist-v1' and record['fixed_subject'] == SYN[0] and record['moving_subject'] == SYN[1] and 'sealed_guard' not in record


def test_l2r_development_ids_pass_through_the_guard(l2r, tmp_path):
    record = l2r.check_subjects(manifest(tmp_path/'run'/'inputs'/'p', DEV))
    assert record['guard'] == 'dev-allowlist-v1' and record['fixed_subject'] == DEV[0] and 'sealed_guard' not in record


@pytest.mark.parametrize('case', ['sealed', 'off_the_allowlist', 'one_synthetic_one_sealed', 'synthetic_and_development', 'no_ids', 'one_id'])
def test_l2r_refusals(l2r, tmp_path, monkeypatch, case):
    """With the synthetic ids pinned on the allowlist, a sealed id is still refused (C23.4-2 step 3), and an id off the pinned allowlist too."""
    sealed, off = real_ids()
    monkeypatch.setattr(l2r, 'ALLOWLIST', synthetic_allowlist(tmp_path, SYN+(sealed,)))
    ids = dict(sealed=(SYN[0], sealed), off_the_allowlist=(off, SYN[0]), one_synthetic_one_sealed=(SYN[1], sealed), synthetic_and_development=(SYN[0], DEV[0]),
               no_ids=None, one_id=None)[case]
    path = manifest(tmp_path/'run'/'inputs'/'p', ids)
    if case == 'one_id':
        W.write_json(path, dict(shape_zyx=[5, 6, 7], fixed_subject=SYN[0]))
    with pytest.raises(SubjectSealed):
        l2r.check_subjects(path)


def test_l2r_evaluate_run_refuses_before_pair_npz_is_opened(l2r, tmp_path, monkeypatch):
    sealed, _ = real_ids()
    run = tmp_path/'run'
    (run/'protocol.json').parent.mkdir(parents=True)
    (run/'protocol.json').write_text(json.dumps(dict(task=dict(name='task', pair='p'))))
    manifest(run/'inputs'/'p', (DEV[0], sealed))
    opened = []
    monkeypatch.setattr(l2r.np, 'load', lambda *a, **k: opened.append(a) or (_ for _ in ()).throw(AssertionError('pair.npz was opened')))
    with pytest.raises(SubjectSealed):
        l2r.evaluate_run(run)
    assert opened == []


# ---- experiments/d0/check_seg4_map.sealed_guard ------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def seg4():
    return load('_sg_test_check_seg4_map', W.ROOT/'experiments'/'d0'/'check_seg4_map.py')


def test_seg4_has_no_synthetic_admission(seg4, tmp_path):
    """Revision 6e: synthetic ids are refused on the repository's allowlist; on a pinned allowlist they go through the guard."""
    with pytest.raises(SystemExit, match='refused by the bayesreg guard'):
        seg4.sealed_guard(list(SYN), W.ROOT/'configs/bayesreg/dev_subjects.json')
    assert seg4.sealed_guard(list(SYN), synthetic_allowlist(tmp_path))['check'].startswith('bayesreg.io.guard.check_subject')


def test_seg4_development_subjects_go_through_the_guard_and_sealed_ones_are_refused(seg4, tmp_path):
    allowlist = W.ROOT/'configs/bayesreg/dev_subjects.json'
    assert seg4.sealed_guard(list(DEV), allowlist)['check'].startswith('bayesreg.io.guard.check_subject')
    sealed, off = real_ids()
    for subjects in ([DEV[0], sealed], [off], [SYN[0], DEV[0]], [SYN[0], sealed]):
        with pytest.raises(SystemExit, match='refused by the bayesreg guard'):
            seg4.sealed_guard(subjects, allowlist)
    pinned = synthetic_allowlist(tmp_path, SYN+(sealed,))
    with pytest.raises(SystemExit, match='refused by the bayesreg guard'):
        seg4.sealed_guard([SYN[0], sealed], pinned)


# ---- experiments/seed/e2e_control.check_run_subjects (C23.6-3 S, revisions 6f, 6g) -----------------------------------------------------------------
NONCE = '0123456789abcdef0123456789abcdef'


@pytest.fixture(scope='module')
def e2e():
    sys.path.insert(0, str(W.ROOT/'experiments'/'seed'))
    try:
        return load('_sg_test_e2e_control', W.ROOT/'experiments'/'seed'/'e2e_control.py')
    finally:
        sys.path.remove(str(W.ROOT/'experiments'/'seed'))


def scratch(e2e, base, ids, *, pair=None, written=None, manifest_pair=None, marker='default'):
    """A scratch as e2e_control.setup leaves it: the marker at its root, the pair's manifest and pair.npz in the frozen copy's stage (as pair_files writes them)
    and their copies in plain/inputs (as prepare copies them).  `written`: other bytes of the frozen copy's manifest; `manifest_pair`: the manifest's pair field;
    `marker`: the marker's document (None: no marker).  pair.npz holds synthetic bytes; the manifest names their sha256."""
    import hashlib
    base = Path(base)
    pair = e2e.PAIR if pair is None else pair
    archive = b'synthetic pair.npz bytes'
    document = dict(pair=e2e.PAIR if manifest_pair is None else manifest_pair, shape_zyx=[34, 36, 38], preprocessing='synthetic', pair_sha256=hashlib.sha256(archive).hexdigest())
    if ids is not None:
        document.update(fixed_subject=ids[0], moving_subject=ids[1])
    text = json.dumps(document)
    stage = base/'frozen_copy'/e2e.STAGE_REL/e2e.PAIR
    stage.mkdir(parents=True)
    (stage/'manifest.json').write_text(text if written is None else written)
    (stage/'pair.npz').write_bytes(archive)
    if marker == 'default':
        marker = dict(pair=e2e.PAIR, fixed_subject=e2e.SYN_FIXED, moving_subject=e2e.SYN_MOVING, nonce=NONCE)
    if marker is not None:
        (base/e2e.MARKER).write_text(json.dumps(marker))
    run = base/'plain'
    (run/'inputs'/pair).mkdir(parents=True)
    (run/'protocol.json').write_text(json.dumps(dict(task=dict(name='task', pair=pair))))
    (run/'inputs'/pair/'manifest.json').write_text(text)
    (run/'inputs'/pair/'pair.npz').write_bytes(archive)
    return base, run


def test_e2e_its_own_scratch_pair_is_admitted(e2e, tmp_path):
    assert (e2e.SYN_FIXED, e2e.SYN_MOVING) == ('SYN1', 'SYN2') and e2e.PAIR == 'OASIS_OAS1_9001_MR1__OASIS_OAS1_9002_MR1'
    base, run = scratch(e2e, tmp_path/'s', (e2e.SYN_FIXED, e2e.SYN_MOVING))
    record = e2e.check_run_subjects(base, run, NONCE)
    assert record['subjects'] == ['SYN1', 'SYN2'] and record['rule'].startswith('C23.6-3 S')


def test_e2e_setup_writes_the_marker_and_the_field_process_gets_the_nonce():
    """The control's own code: setup writes the marker with the constants and the nonce; control hands the same nonce to the --field process; pair_files writes the
    constants (read as text, nothing run)."""
    text = (W.ROOT/'experiments'/'seed'/'e2e_control.py').read_text(encoding='utf-8')
    assert "(base/MARKER).write_text(json.dumps(dict(pair=PAIR, fixed_subject=SYN_FIXED, moving_subject=SYN_MOVING, nonce=nonce)))" in text
    assert "'--field', str(base), nonce]" in text and 'setup=setup(base, cache, nonce)' in text and 'fixed_subject=SYN_FIXED, moving_subject=SYN_MOVING,' in text
    assert text.index('    check_run_subjects(base, run, nonce)\n') < text.index("    with np.load(run/'inputs'/pair/'pair.npz') as z:")


def test_e2e_development_pairs_go_through_the_guard(e2e, tmp_path):
    record = e2e.check_run_subjects(*scratch(e2e, tmp_path/'s', DEV, pair='OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1', marker=None))
    assert record['rule'].startswith('C23.4-2')


@pytest.mark.parametrize('case', ['no_marker', 'another_nonce', 'no_nonce', 'marker_of_other_ids', 'other_scratch_manifest', 'other_pair', 'manifest_pair_differs', 'not_plain',
                                  'ids_swapped', 'stage_is_a_link', 'inputs_is_a_link', 'in_a_real_data_root',
                                  'npz_is_a_link', 'npz_differs', 'npz_missing'])
def test_e2e_the_synthetic_ids_off_the_rule_are_refused_before_pair_npz(e2e, tmp_path, monkeypatch, case):
    """Revision 6g: every condition of the rule, one at a time; the refusal comes before pair.npz is read (np.load is not reached)."""
    ids = (e2e.SYN_MOVING, e2e.SYN_FIXED) if case == 'ids_swapped' else (e2e.SYN_FIXED, e2e.SYN_MOVING)
    kwargs = dict(no_marker=dict(marker=None), marker_of_other_ids=dict(marker=dict(pair=e2e.PAIR, fixed_subject='SYN3', moving_subject='SYN2', nonce=NONCE)),
                  other_scratch_manifest=dict(written='{"pair": "x"}'), other_pair=dict(pair='OASIS_OAS1_9003_MR1__OASIS_OAS1_9004_MR1'),
                  manifest_pair_differs=dict(manifest_pair='OASIS_OAS1_9003_MR1__OASIS_OAS1_9004_MR1')).get(case, {})
    base, run = scratch(e2e, tmp_path/'s', ids, **kwargs)
    nonce = dict(another_nonce='f'*32, no_nonce=None).get(case, NONCE)
    if case == 'not_plain':
        run = run.rename(base/'other')
    if case in ('stage_is_a_link', 'inputs_is_a_link'):
        target = base/'frozen_copy'/e2e.STAGE_REL if case == 'stage_is_a_link' else run/'inputs'
        moved = target.rename(tmp_path/'elsewhere')
        target.symlink_to(moved, target_is_directory=True)
    if case == 'in_a_real_data_root':
        monkeypatch.setattr(e2e, 'REAL_DATA_ROOTS', (tmp_path,))                  # the scratch lies in a "real data root" (the pinned table, changed by monkeypatch)
    npz = run/'inputs'/e2e.PAIR/'pair.npz'
    if case == 'npz_is_a_link':                                                     # review m11: the run's pair.npz a link to a file elsewhere
        (tmp_path/'elsewhere.npz').write_bytes(npz.read_bytes())
        npz.unlink()
        npz.symlink_to(tmp_path/'elsewhere.npz')
    if case == 'npz_differs':                                                       # review m11: not the file the manifest's pair_sha256 names
        npz.write_bytes(b'other bytes')
    if case == 'npz_missing':
        npz.unlink()
    if case != 'ids_swapped':
        monkeypatch.setattr(e2e, 'subprocess', None)                                # the synthetic branch never reaches the guard's process
    with pytest.raises(SystemExit, match='refused'):
        e2e.check_run_subjects(base, run, nonce)


def test_e2e_field_process_reads_nothing_after_a_refusal(e2e, tmp_path, monkeypatch):
    """--field on a scratch whose marker is not this control's: SystemExit before the snapshot is put on the path or pair.npz is opened."""
    base, run = scratch(e2e, tmp_path/'s', (e2e.SYN_FIXED, e2e.SYN_MOVING))
    import numpy as np
    monkeypatch.setattr(np, 'load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('pair.npz was opened')))
    path = list(sys.path)
    with pytest.raises(SystemExit, match='refused'):
        e2e.field_process(base, 'f'*32)
    assert sys.path == path


@pytest.mark.parametrize('case', ['sealed', 'no_ids', 'synthetic_and_sealed', 'synthetic_names_off_the_rule', 'one_constant_id'])
def test_e2e_refusals(e2e, tmp_path, case):
    sealed, _ = real_ids()
    ids = dict(sealed=(DEV[0], sealed), no_ids=None, synthetic_and_sealed=(SYN[0], sealed), synthetic_names_off_the_rule=(SYN[0], SYN[1]), one_constant_id=(e2e.SYN_FIXED, DEV[0]))[case]
    with pytest.raises(SystemExit, match='refused'):
        e2e.check_run_subjects(*scratch(e2e, tmp_path/'s', ids), NONCE)


def test_the_synthetic_admission_is_gone_from_the_code():
    """Revision 6e: no production entry calls a synthetic check and pins.check_synthetic (6d) is gone; e2e_control's rule is its own (C23.6-3 S), pinned to its
    constants."""
    import bayesreg.seg.unet.pins as pins
    assert not hasattr(pins, 'check_synthetic')
    for rel in ('scripts/evaluate_l2r_style.py', 'experiments/d0/check_seg4_map.py', 'experiments/seed/e2e_control.py'):
        text = (W.ROOT/rel).read_text(encoding='utf-8')
        assert 'check_synthetic' not in text, rel


# ---- experiments/d0/evaluate.py, the F rows (C23.6-2 revision 6h, SG code review I2) ------------------------------------------------------------------------
@pytest.fixture
def d0ev(tmp_path, monkeypatch):
    """d0/evaluate.py (a module of its own) with STAGE pointed at a synthetic stage: selection.json naming one pair (dict rows, as the real stage), the pair's
    manifest and pair.npz (synthetic arrays), and beside it another stage holding a pair of the same name that is not registered there."""
    import hashlib
    import types
    import numpy as np
    ev = load('_sg_test_d0_evaluate', W.ROOT/'experiments'/'d0'/'evaluate.py')
    stage, other = tmp_path/'three_d'/'stage_dev', tmp_path/'three_d'/'stage_other'
    pair = 'OASIS_OAS1_0985_MR1__OASIS_OAS1_0986_MR1'
    for root in (stage, other):
        d = root/pair
        d.mkdir(parents=True)
        labels = (np.arange(60).reshape(3, 4, 5) % 4).astype(float)
        np.savez(d/'pair.npz', fixed=np.zeros((3, 4, 5)), moving=np.zeros((3, 4, 5)), fixed_seg4=labels, moving_seg4=labels, fixed_seg35=labels, moving_seg35=labels)
        (d/'manifest.json').write_text(json.dumps(dict(pair=pair, pair_sha256=hashlib.sha256((d/'pair.npz').read_bytes()).hexdigest(), spacing_xyz=[1., 1., 1.])))
    (stage/'selection.json').write_text(json.dumps(dict(pairs=[dict(pair=pair)])))
    monkeypatch.setattr(ev, 'STAGE', stage)
    return types.SimpleNamespace(ev=ev, stage=stage, other=other, pair=pair)


def test_d0_evaluate_reads_a_registered_plain_pair(d0ev):
    assert d0ev.ev.stage_pair(d0ev.pair) == d0ev.pair
    labs, man, sha = d0ev.ev.truth_labels(d0ev.pair)
    assert set(labs) == {'fixed_seg4', 'fixed_seg35', 'moving_seg4', 'moving_seg35'} and man['pair'] == d0ev.pair and sha == man['pair_sha256']


@pytest.mark.parametrize('case', ['up_and_over', 'absolute', 'slash', 'backslash', 'dotdot', 'dot', 'empty', 'not_a_string', 'not_registered', 'a_link', 'no_selection'])
def test_d0_evaluate_refuses_any_other_pair_before_reading_anything(d0ev, monkeypatch, case):
    """`--pair ../stage_other/<pair>` and the like: SystemExit before any file but selection.json is opened (no sha256, no np.load), in truth_labels and in
    evaluate_run (before the run directory is read)."""
    ev, pair = d0ev.ev, d0ev.pair
    if case == 'a_link':
        target = d0ev.stage/pair
        target.rename(d0ev.stage/'moved')
        target.symlink_to(d0ev.other/pair, target_is_directory=True)
    if case == 'no_selection':
        (d0ev.stage/'selection.json').unlink()
    name = dict(up_and_over=f'../stage_other/{pair}', absolute=str(d0ev.other/pair), slash=f'x/{pair}', backslash=f'x\\{pair}', dotdot='..', dot='.', empty='',
                not_a_string=None, not_registered='OASIS_OAS1_0987_MR1__OASIS_OAS1_0988_MR1').get(case, pair)
    monkeypatch.setattr(ev, 'sha_file', lambda *a, **k: (_ for _ in ()).throw(AssertionError('a file was hashed')))
    monkeypatch.setattr(ev.np, 'load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('pair.npz was opened')))
    monkeypatch.setattr(ev, 'rd', lambda *a, **k: (_ for _ in ()).throw(AssertionError('the run directory was read')))
    real_read_text = pathlib.Path.read_text

    def only_selection(self, *a, **k):                                   # SG follow-up diff review n4 (A5b): a manifest (or any file but selection.json) read before the refusal fails
        if self.name != 'selection.json':
            raise AssertionError(f'{self} was read before the refusal')
        return real_read_text(self, *a, **k)
    monkeypatch.setattr(pathlib.Path, 'read_text', only_selection)
    with pytest.raises(SystemExit, match='C23.6-2 revision 6h'):
        ev.truth_labels(name)
    with pytest.raises(SystemExit, match='C23.6-2 revision 6h'):
        ev.evaluate_run(d0ev.stage.parent/'run', 'A0', name, 'run_id')


@pytest.mark.parametrize('case', ['plain', 'inputs_link', 'pair_dir_link', 'copy_link', 'copy_outside'])
def test_d0_evaluate_reads_the_run_directorys_copy_only_when_it_is_no_link(d0ev, tmp_path, case):
    """Contract C23 revision 6i (SG follow-up review n2): run_copy -- the run's inputs/<pair>/pair.npz is read only when neither it nor inputs/ nor inputs/<pair> is a
    link and it lies in the run directory; evaluate_run calls it right before that read."""
    ev, pair = d0ev.ev, d0ev.pair
    run = tmp_path/'run'
    elsewhere = tmp_path/'elsewhere'/'inputs'/pair
    elsewhere.mkdir(parents=True)
    (elsewhere/'pair.npz').write_bytes(b'not read')
    if case == 'inputs_link':
        run.mkdir()
        (run/'inputs').symlink_to(elsewhere.parent, target_is_directory=True)
    elif case == 'pair_dir_link':
        (run/'inputs').mkdir(parents=True)
        (run/'inputs'/pair).symlink_to(elsewhere, target_is_directory=True)
    else:
        (run/'inputs'/pair).mkdir(parents=True)
        if case == 'plain':
            (run/'inputs'/pair/'pair.npz').write_bytes(b'the copy')
        else:
            (run/'inputs'/pair/'pair.npz').symlink_to(elsewhere/'pair.npz')
    if case == 'copy_outside':                                                     # a link is refused whatever it points to
        assert (run/'inputs'/pair/'pair.npz').is_symlink()
    if case == 'plain':
        assert ev.run_copy(run, pair) == run/'inputs'/pair/'pair.npz'
    else:
        with pytest.raises(SystemExit, match='review n2'):
            ev.run_copy(run, pair)
    text = (W.ROOT/'experiments'/'d0'/'evaluate.py').read_text(encoding='utf-8')
    call = text.index('    run_copy(run, pair)  ')
    assert text.index("    with np.load(run/'inputs'/pair/'pair.npz') as z:", call) == text.index('\n', call)+1          # the very next line reads the copy
