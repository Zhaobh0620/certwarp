"""LP-C2 integration review I1 and the lead's ruling (docs/bayesreg/reviews/LP-C2整合_复核_20261120.md; contract C26.4 L51, revision 11e): experiments/lpba/make_eval_manifest_lpba.py
on the synthetic world of _eval_world only (a temporary code tree and git repository, tmp_path roots, tiny NIfTI files).  The manifest it derives from the evaluation entry's own
input_roles is accepted by the entry's manifest check (and by its whole admission); a line more, a line less, a sha256 that does not match are refused (by the entry and by the
script's own check); no label file is opened (the guard refuses one, and an open under a guarded root outside the original images); --write writes once, a failed read-back
leaves the code tree as it was."""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

import bayesreg.io.sealed as S
from bayesreg.eval import holdout as H
from bayesreg.eval import lpba_eval as E

from . import _eval_world as EW

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments'/'lpba'))
import make_eval_manifest_lpba as MK  # noqa: E402

FORMAL = 'lpba40_test'


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = EW.make(tmp_path, monkeypatch)
    monkeypatch.setattr(MK, 'REPO', w.repo)
    monkeypatch.setattr(MK, 'TMP_ROOTS', ())                     # the tests' tmp_path is under /tmp
    rows = EW.build_stage(w, FORMAL)
    EW.write_cells(w, FORMAL, rows)
    w.rows = rows
    S._reset_cache()
    yield w
    E.stop_spy()
    MK.guard_off()
    S._reset_cache()


def no_truth(monkeypatch):
    """Any reader of a label fails the test (besides the guard)."""
    def refuse(*args, **kwargs):
        raise AssertionError('a label reader was called')
    monkeypatch.setattr(E, 'read_label', refuse)
    monkeypatch.setattr(E, '_label_volume', refuse)


def run(world, *args, name='out'):
    S._reset_cache()
    out = world.repo.parent/name
    code = MK.run([*args, '--out', str(out)])
    report = json.loads((out/'report.json').read_text()) if (out/'report.json').is_file() else None
    return code, out, report


def commit_manifest(world, text):
    EW.W.write_text(world.repo/E.MANIFEST_REL, text)
    EW.W.write_text(world.repo/E.RECORDS_REL, f'# record-only lines (test) of {hashlib.sha256(text.encode()).hexdigest()}\n')
    EW.W.commit_all(world.repo, 'the step-10 manifest (test)')
    S._reset_cache()


def entry_check(world):
    """lpba_eval.check_manifest as the entry calls it (the admission's cells and originals first)."""
    table, table_record = E.load_arm_table(FORMAL, world.repo)
    cells = E.check_cells(table, world.rows, repo=world.repo)
    originals = E.image_originals(table, world.rows)
    return E.check_manifest(table, world.rows, cells, originals, prereg_sha256=E.PREREG_SHA256, arm_table_sha256=table_record['sha256'], repo=world.repo)


def test_the_derived_manifest_is_the_entrys_and_accepted_by_its_admission(world, monkeypatch):
    no_truth(monkeypatch)
    code, out, report = run(world, '--check-only')
    assert code == 0 and report['verdict'] == 'written and checked' and report['guard_hits'] == [] and report['spy']['refused'] == []
    text = (out/MK.OUT_NAME).read_text(encoding='utf-8')
    assert not (world.repo/E.MANIFEST_REL).exists()                                            # --check-only leaves the code tree alone
    roles = EW.write_manifest(world, FORMAL, world.rows, commit=False)                           # the tests' own writer, from the same input_roles
    assert text == (world.repo/E.MANIFEST_REL).read_text(encoding='utf-8')
    assert {line['path']: line['role'] for line in report['lines']} == roles and report['n_inputs'] == len(roles)
    assert set(report['roles']) == {'cell', 'stage', 'image_header'} and report['roles']['stage'] == 2*17 and report['roles']['image_header'] == 34
    assert report['header'] == dict(prereg_sha256=E.PREREG_SHA256, arm_table_sha256=E.ARM_TABLE_SHA256) and report['pairs'] == 17
    assert report['manifest_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert any(line['path'] == str(E.finalized_path(FORMAL)) and line['role'] == 'cell' for line in report['lines'])      # review I1: the finalized record is a line
    commit_manifest(world, text)
    record, listed = entry_check(world)
    assert record['n_inputs'] == len(roles) and listed == {line['path']: line['sha256'] for line in report['lines']}
    S._reset_cache()
    admission, context = E.open_lpba(E.eval_dir(FORMAL)/'run1', FORMAL, run=EW.no_units, repo=world.repo)      # the whole admission of the entry, up to its master lock
    assert context['record']['manifest']['sha256'] == hashlib.sha256(text.encode()).hexdigest()
    E.stop_spy()


@pytest.mark.parametrize('change', ['extra', 'missing', 'sha256'])
def test_a_line_more_a_line_less_or_another_sha256_is_refused(world, change):
    code, out, report = run(world, '--check-only')
    assert code == 0
    text = (out/MK.OUT_NAME).read_text(encoding='utf-8')
    lines = text.splitlines()
    terminal = next(line['path'] for line in report['lines'] if line['path'].endswith('/method/LPBA40_S03__LPBA40_S04/terminal.json'))
    if change == 'extra':
        lines.append(f"{'0'*64}  {world.repo}/README_extra.txt")
        match = 'outside the files this entry reads'
    elif change == 'missing':
        lines = [line for line in lines if not line.endswith(f'  {terminal}')]
        match = 'not in the manifest'
    else:
        lines = [f"{'f'*64}  {terminal}" if line.endswith(f'  {terminal}') else line for line in lines]
        match = 'read in the admission at sha256'
    tampered = '\n'.join(lines)+'\n'
    table, table_record = E.load_arm_table(FORMAL, world.repo)
    cells = E.check_cells(table, world.rows, repo=world.repo)
    originals = E.image_originals(table, world.rows)
    roles = E.input_roles(table, world.rows, cells, originals)
    header = dict(prereg_sha256=E.PREREG_SHA256, arm_table_sha256=table_record['sha256'])
    with pytest.raises((MK.Refusal, H.HoldoutRefused)):                                          # the script's own check before writing
        MK.verify_text(E, tampered.encode(), roles=roles, header=header, cells=cells, originals=originals)
    commit_manifest(world, tampered)
    with pytest.raises(H.HoldoutRefused, match=match):                                           # the entry's check of the committed file
        entry_check(world)


def test_write_writes_once_into_the_code_tree(world, monkeypatch):
    no_truth(monkeypatch)
    code, out, report = run(world, '--write', name='w1')
    target = world.repo/E.MANIFEST_REL
    assert code == 0 and target.is_file() and report['written'] == str(target) and not (out/MK.OUT_NAME).exists()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == report['manifest_sha256']
    before = target.read_bytes()
    code, out, report = run(world, '--write', name='w2')
    assert code == MK.EXIT['refused'] and report['stage'] == 'once' and target.read_bytes() == before
    S._reset_cache()
    assert MK.run(['--write', '--out', str(out)]) == MK.EXIT['refused'] and target.read_bytes() == before        # --out exists: refused before anything


def test_a_failed_read_back_moves_the_written_file_out_of_the_code_tree(world, monkeypatch):
    original = MK.verify_text

    def failing(*args, rehash=False, **kwargs):
        if rehash:
            raise MK.Refusal('verify', 'an input changed (test)')
        return original(*args, rehash=rehash, **kwargs)
    monkeypatch.setattr(MK, 'verify_text', failing)
    code, out, report = run(world, '--write')
    assert code == MK.EXIT['check_failed'] and not (world.repo/E.MANIFEST_REL).exists()
    moved = report['moved_to_failed']
    assert len(moved) == 1 and Path(moved[0]['new']) == out/MK.FAILED_DIR/'eval_manifest.txt' and Path(moved[0]['new']).is_file()


def test_the_guard_refuses_a_label_file_and_writes_nothing(world, monkeypatch):
    """A label among the input roles (as if the entry's roles had one): the guard refuses its open before it happens; exit 4, nothing written."""
    roles = E.input_roles
    label = world.raw/'S03'/'delineation'/'label.nii.gz'
    monkeypatch.setattr(E, 'input_roles', lambda *a: dict(roles(*a), **{str(label): 'cell'}))
    code, out, report = run(world, '--write')
    assert code == MK.EXIT['guard'] and report['verdict'] == 'guard' and [h['path'] for h in report['guard_hits']] == [str(label)]
    assert not (world.repo/E.MANIFEST_REL).exists() and 'moved_to_failed' not in report


def test_the_guard_refuses_an_open_under_a_guarded_root_before_the_originals_are_known(world, monkeypatch):
    image = world.raw/'S03'/'delineation'/'image.nii.gz'
    cells = E.check_cells

    def peeking(*args, **kwargs):
        image.read_bytes()                                                                      # an image read before the stage manifests name it
        return cells(*args, **kwargs)
    monkeypatch.setattr(E, 'check_cells', peeking)
    code, out, report = run(world, '--check-only')
    assert code == MK.EXIT['guard'] and report['guard_hits'][0]['path'] == str(image) and not (out/MK.OUT_NAME).exists()


def test_a_swallowed_guard_violation_still_fails(world, monkeypatch):
    label = world.raw/'S04'/'delineation'/'label.nii.gz'
    originals = E.image_originals

    def swallowing(*args, **kwargs):
        try:
            label.read_bytes()
        except Exception:                                                                       # noqa: BLE001 - the test's caller hides it
            pass
        return originals(*args, **kwargs)
    monkeypatch.setattr(E, 'image_originals', swallowing)
    code, out, report = run(world, '--check-only')
    assert code == MK.EXIT['guard'] and report['guard_hits'][0]['path'] == str(label)


def test_refusals_before_anything_is_read(world, monkeypatch):
    code, out, report = run(world, '--check-only')
    assert code == 0
    assert MK.run(['--check-only', '--out', str(out)]) == MK.EXIT['refused']                     # --out exists
    S._reset_cache()
    assert MK.run(['--check-only', '--out', str(E.root(FORMAL)/'mk')]) == MK.EXIT['refused']     # under the formal root
    E.master_lock(FORMAL).parent.mkdir(parents=True, exist_ok=True)
    E.master_lock(FORMAL).write_text('{}\n')
    code, out, report = run(world, '--check-only', name='locked')
    assert code == MK.EXIT['refused'] and report['stage'] == 'once'                              # after the entry's master lock the manifest never changes


def test_a_cell_not_written_refuses(world):
    table, _ = E.load_arm_table(FORMAL, world.repo)
    arm = next(a for a in table['arms'] if a['name'] == 'ants_s')
    (Path(arm['record_dir'].format(pair='LPBA40_S05__LPBA40_S06', arm='ants_s'))/'result.json').unlink()
    code, out, report = run(world, '--check-only')
    assert code == MK.EXIT['refused'] and report['stage'] == 'arms' and not (out/MK.OUT_NAME).exists()


@pytest.mark.parametrize('swallow', [True, False])
def test_a_refusal_of_the_entrys_spy_decides_the_exit_as_the_guards_do(world, monkeypatch, swallow):
    """LP-C2 final diff review minor 1: an open the evaluation entry's spy refuses after its allowance (a file under a root it watches that is no input) exits 4 whether a caller
    swallows the SpyRefused or not; it is in the report and nothing stays in the code tree."""
    table, _ = E.load_arm_table(FORMAL, world.repo)
    stray = Path(table['stage_root'])/'stray.txt'
    stray.write_text('x\n')
    check_stage = E.check_stage

    def peeking(*args, **kwargs):
        result = check_stage(*args, **kwargs)
        try:
            stray.read_text()
        except PermissionError:                                                                 # SpyRefused
            if not swallow:
                raise
        return result
    monkeypatch.setattr(E, 'check_stage', peeking)
    code, out, report = run(world, '--write')
    assert code == MK.EXIT['guard'] and report['verdict'] == 'spy' and report['guard_hits'] == []
    assert report['spy']['refused'] == [os.path.realpath(stray)] and not (world.repo/E.MANIFEST_REL).exists()
    if swallow:                                                                                 # written, then moved out of the code tree
        assert [Path(m['new']).name for m in report['moved_to_failed']] == ['eval_manifest.txt']
    else:
        assert 'moved_to_failed' not in report and report['error_after'].startswith('SpyRefused')
