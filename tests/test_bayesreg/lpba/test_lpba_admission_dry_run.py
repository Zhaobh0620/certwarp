"""LP-C2 item 4 (contract C26.9-3 第 10a 项; C26.10 T16's id-only half): experiments/lpba/admission_dry_run.py on the synthetic world (_eval_world): the 34 ids and 17 pairs
through every entry's admission function, the contrasts, a spy that allows code-tree files only; LP-B's lpba40_pre_manifest missing -> the dry run does not pass (fail closed);
present -> it is called for the 17 test pairs and the two feasibility pairs."""
import json
from pathlib import Path
import sys

import pytest

import bayesreg.io.sealed as S
from bayesreg.io.pair import REGISTRY_SCHEMA, subject_list_sha256

from . import _eval_world as EW

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments'/'lpba'))
import admission_dry_run as dry  # noqa: E402

lc, hc = EW.lc, EW.hc


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = EW.make(tmp_path, monkeypatch)
    registry = EW.write_json(tmp_path/'registry.json', dict(schema=REGISTRY_SCHEMA, comment='synthetic', models={
        f'unet35_v4_ens_fold{f}': dict(segmenter_id='unet35_v4_ens', fold=i, member_model_sha256=['ab'*32, 'cd'*32], label_contact_subjects=list(EW.CONTACT),
                                       label_contact_subjects_sha256=subject_list_sha256(EW.CONTACT)) for i, f in enumerate('AB')}))
    monkeypatch.setattr(dry, 'registry_path', lambda: registry)
    yield w
    S._reset_cache()


def test_10a_without_lp_bs_function_everything_else_passes_and_it_fails_closed(world, tmp_path, monkeypatch):
    import he_holdout
    monkeypatch.delattr(he_holdout, 'lpba40_pre_manifest')          # integration (LP-B merged): the function is in the code tree now; taken away here to keep the fail-closed case
    out = tmp_path/'dry.json'
    S._reset_cache()
    assert dry.main(['--out', str(out)], out=lambda text: None) == 8
    record = json.loads(out.read_text())
    assert record['passed'] is False and len(record['problems']) == 1 and 'lpba40_pre_manifest' in record['problems'][0]
    assert len(record['subjects']) == 34 and all(r['check_subject']['admitted_by_lpba40'] and r['development_allowlist']['step'] == 'sealed' for r in record['subjects'].values())
    assert all(r['select_fold_models']['policy'] == 'all_folds_mean' for r in record['subjects'].values())
    assert list(record['pairs']) == [f'LPBA40_S{n:02d}__LPBA40_S{n+1:02d}' for n in range(3, 37, 2)]
    assert all(r['names_without_allowlist'] == 'refused (SubjectSealed)' and r['check_pair_subjects']['lpba40_guard']['fixed']['admitted_by_lpba40'] for r in record['pairs'].values())
    assert record['contrasts'] == {'lpba40_test:S01': 'refused (admission)', 'lpba40_test:S02': 'refused (admission)', 'lpba40_test:S37': 'refused (admission)',
                                   'lpba40_feas:S03': 'refused (admission)', 'lpba40_feas:S37': 'refused (admission)'}
    assert record['a4_admission']['subjects'][0] == 'S03' and record['admit']['lpba40_feas']['pairs'] == ['LPBA40_S01__LPBA40_S02', 'LPBA40_S02__LPBA40_S01']
    assert record['spy']['passed'] is True
    opened = [e for e in record['spy'].get('events', record['spy'].get('opened', []))] if isinstance(record['spy'], dict) else []
    assert not any(str(world.raw) in json.dumps(e) or str(world.results) in json.dumps(e) for e in opened)


def test_10a_with_the_function_it_is_called_for_both_pair_sets_and_passes(world, tmp_path, monkeypatch):
    calls = []

    def stub(pair_set, pair):
        calls.append((pair_set, pair))
        return dict(pair_set=pair_set, pair=pair, steps='1-3 (stub)')
    import he_holdout
    monkeypatch.setattr(he_holdout, 'lpba40_pre_manifest', stub, raising=False)
    out = tmp_path/'dry.json'
    S._reset_cache()
    assert dry.main(['--out', str(out)], out=lambda text: None) == 0
    record = json.loads(out.read_text())
    assert record['passed'] is True and record['pre_manifest']['function'] == 'he_holdout.lpba40_pre_manifest'
    assert calls == [('lpba40_test', f'LPBA40_S{n:02d}__LPBA40_S{n+1:02d}') for n in range(3, 37, 2)]+[('lpba40_feas', 'LPBA40_S01__LPBA40_S02'), ('lpba40_feas', 'LPBA40_S02__LPBA40_S01')]
    assert dry.main(['--out', str(out)], out=lambda text: None) == 2              # a new file for every dry run


def test_10a_with_lp_bs_own_function_it_passes(world, tmp_path):
    """Integration (LP-B merged): LP-B's he_holdout.lpba40_pre_manifest, unchanged, on the synthetic world: the 17 test pairs and the two feasibility pairs, the dry run passes."""
    import he_holdout
    out = tmp_path/'dry.json'
    S._reset_cache()
    assert dry.pre_manifest_function()[1] == 'he_holdout.lpba40_pre_manifest' and callable(he_holdout.lpba40_pre_manifest)
    assert dry.main(['--out', str(out)], out=lambda text: None) == 0
    record = json.loads(out.read_text())
    assert record['passed'] is True and record['problems'] == [] and record['pre_manifest']['function'] == 'he_holdout.lpba40_pre_manifest' and record['spy']['passed'] is True


def test_10a_a_development_allowlist_in_place_of_the_lpba_one_is_refused(world, tmp_path, monkeypatch):
    import bayesreg.io.guard as G
    monkeypatch.setattr(G, 'LPBA40_ALLOWLIST_SHA256', 'e'*64)
    out = tmp_path/'dry.json'
    S._reset_cache()
    assert dry.main(['--out', str(out)], out=lambda text: None) == 8
    record = json.loads(out.read_text())
    assert record['passed'] is False and (record.get('refused') or {}).get('stage') == 'allowlist' or any('allowlist' in p for p in record.get('problems', []))


def test_minor2_any_other_exception_is_written_into_the_record(world, tmp_path, monkeypatch):
    def broken(**kwargs):
        raise RuntimeError('a stand-in failure (test)')
    monkeypatch.setattr(dry, 'check', broken)
    out = tmp_path/'dry.json'
    S._reset_cache()
    assert dry.main(['--out', str(out)], out=lambda text: None) == 8
    record = json.loads(out.read_text())
    assert record['passed'] is False and record['refused']['stage'] == 'exception' and 'RuntimeError' in record['refused']['message']
