"""LP-H (decision_log 2026-11-21 ~2:58 PM CST; the scan after the feasibility run3 stop): bayesreg.eval.lpba_eval end to end in the production layout -- the code tree at <formal root>/code
with no data/ and as the working directory (tests/test_bayesreg/hb/_lpba_world.formal_checkout), the sealed lists at their production paths (production_lists) -- on _eval_world's synthetic
world (as test_lpba_eval.py's T11): both modes evaluate with no refusal of the entry's spy, no truth copy left, the labels read only in the formal mode.  No real file of a subject."""
import json
import os
from pathlib import Path

import pytest

import bayesreg.io.sealed as S
from bayesreg.eval import lpba_eval as E

from ..hb import _lpba_world as L
from . import _eval_world as EW

hc = EW.hc


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = EW.make(tmp_path, monkeypatch)
    L.production_lists(w, monkeypatch)
    code = L.formal_checkout(w, monkeypatch)
    monkeypatch.setattr(hc, 'AUTH_ROOT', code)
    assert Path(os.getcwd()) == code == w.repo and not (code/'data').exists()
    yield w
    E.stop_spy()
    S._reset_cache()


def summary_of(context):
    return json.loads((context['out']/'lpba_eval.json').read_text())


def test_lph_the_no_truth_mode_evaluates_from_the_checkout(world):
    rows = EW.build_stage(world, 'lpba40_feas')
    EW.write_cells(world, 'lpba40_feas', rows)
    S._reset_cache()
    admission, context = E.open_lpba(E.eval_dir('lpba40_feas')/'feas1', 'lpba40_feas', run=EW.no_units, repo=world.repo)
    assert E.evaluate_lpba(admission, context) == 0
    summary = summary_of(context)
    assert summary['opened_files']['refused'] == [] and summary['label_files_opened'] == [] and summary['truth_mode'] is False
    sealed = {str(p) for p in hc.sealed_list_paths()}
    assert {e['path'] for e in summary['opened_files']['events'] if e['kind'] == 'sealed_list'} == sealed


def test_lph_the_formal_mode_evaluates_from_the_checkout(world):
    rows = EW.build_stage(world, 'lpba40_test')
    EW.write_cells(world, 'lpba40_test', rows)
    EW.write_manifest(world, 'lpba40_test', rows)
    S._reset_cache()
    admission, context = E.open_lpba(E.eval_dir('lpba40_test')/'run1', 'lpba40_test', run=EW.no_units, repo=world.repo)
    code = E.evaluate_lpba(admission, context)
    assert code == 0, (context['out']/'stopped.json').read_text() if (context['out']/'stopped.json').exists() else code
    summary = summary_of(context)
    assert summary['opened_files']['refused'] == [] and len(list((context['out']/'pairs').glob('*.json'))) == 17
    assert list(world.tmp.iterdir()) == [], 'the private temporary directories are removed'
