"""LP-H (decision_log 2026-11-21 ~2:58 PM CST; the scan after the feasibility run3 stop): an LPBA40 cell of the external arms ants_s, fireants_greedy and fireants_syn through
xb_drive.run_cell in the production layout -- the code tree at <formal root>/code, no data/, the working directory (_lpba_world.formal_checkout); the sealed lists and BL2's frozen files at
their production relative paths (test_xb_lpg's bl2 fixture: synthetic contents, the pins moved to them) -- with the REAL check_frozen and check_environment (conda, pip, git stood in) under the
driver's spy; the method step is XB's FakeMethod (a real method process needs its conda environment and the GPU: not run here).  No real file of a subject, no real list, no unit."""
import json
import os
from pathlib import Path

import pytest

from ..hb import _lpba_world as L
from . import _world as W
from .test_xb_lpg import bl2  # noqa: F401  (the fixture: BL2's frozen files in world.repo)

hc, xc, xb_drive = W.hc, W.xc, W.xb_drive
FEAS, FEAS_PAIR = L.FEAS, L.FEAS_PAIR
REVERSE = 'LPBA40_S02__LPBA40_S01'
ARMS = ('ants_s', 'fireants_greedy', 'fireants_syn')


def step_of(arm):
    """The method step stood in: ANTs' files from a zero field (test_xb_ants_field.FakeAnts) for ants_s, XB's FakeMethod for the Python arms."""
    if xc.ARMS[arm]['kind'] == 'ants':
        import numpy as np
        from .test_xb_ants_field import FakeAnts
        return FakeAnts(np.zeros(tuple(L.LE.STORED)+(3,)))
    return W.FakeMethod()


def deps(arm, **kwargs):
    d = W.deps(step=step_of(arm), **kwargs)
    d.code_run = L.detached_git()
    return d


@pytest.fixture
def world(tmp_path, monkeypatch):
    """LP-B's LPBA40 world, the feasibility stage, the lists at their production paths; one cell of each arm run first in the tests' own layout with check_frozen and check_environment
    stood in (the reverse pair), so that the driver's lazy imports of this checkout's modules are done before the code tree moves (in production the code tree is the running code); then the
    code tree at <formal root>/code and the working directory there."""
    w = L.make(tmp_path, monkeypatch)
    w.prod = L.production_lists(w, monkeypatch)
    L.prepare(w, monkeypatch, FEAS, hi=False, atropos=False)
    with monkeypatch.context() as m:
        W.fake_checks(m)
        for arm in ARMS:
            assert W.run_cell(arm, FEAS, REVERSE, deps=deps(arm)) == 0, (arm, (lambda d: (d.get('status'), d.get('failure')))(result(arm, REVERSE)))
    w.code = L.formal_checkout(w, monkeypatch)
    return w


@pytest.fixture
def production(world, bl2, monkeypatch):
    """bl2 wrote a synthetic frozen configuration into the code tree: its sha256 is the pin of lpba_common's frozen files too (lpba_common.check_pins), as the real one is in production."""
    monkeypatch.setitem(L.lc.FROZEN_FILES, xc.FROZEN_CONFIG_REL, xc.FROZEN_CONFIG_SHA256)
    # In production the running code and the checkout are one tree (<formal root>/code); here the running code is this checkout's, whose own files the driver hashes (its harness
    # record, the settings' configC_reference.json): this checkout's experiments/, bayesreg/ and jd_curl/ are read roots of the test (code, as the code tree is in production).
    monkeypatch.setattr(W.xb_spy, 'SYSTEM_READ', (*W.xb_spy.SYSTEM_READ, *(str(W.ROOT/d) for d in ('experiments', 'bayesreg', 'jd_curl'))))
    assert Path(xc.REPO) == world.code == Path(os.getcwd()) and not (world.code/'data').exists()
    return world


def result(arm, pair):
    return json.loads(xc.cell_paths(arm, FEAS, pair)['result'].read_text())


@pytest.mark.parametrize('arm', ARMS)
def test_lph_an_external_arms_lpba_cell_runs_in_the_production_layout_with_the_real_checks(production, bl2, arm):
    assert W.run_cell(arm, FEAS, FEAS_PAIR, deps=deps(arm, env_run=bl2.run)) == 0
    doc = result(arm, FEAS_PAIR)
    assert doc['status'] == 'ok' and doc['spy_passed'] is True, (doc.get('failure'), doc.get('spy'))
    assert doc['checks']['frozen']['passed'] is True and doc['checks']['environment']['passed'] is True and doc['checks']['frozen']['config']['sha256'] == xc.FROZEN_CONFIG_SHA256
    spy = doc['spies']['driver']
    assert spy['breaches'] == [] and spy['passed'] is True and spy['n_sealed_list_reads'] >= 8
    files = {f['file'] for f in doc['checks']['frozen']['files_checked']}
    assert {str(p) for p in xc.archive_reads(arm)} <= files                          # every archive/ file the policy names was checked (the conda_list and pip_freeze among the records)


@pytest.mark.parametrize('arm', ARMS)
def test_lph_without_the_archive_reads_the_cell_is_refused_at_bl2s_drive_py(production, bl2, monkeypatch, arm):
    """LP-G revision 1 taken out in this layout: the refusal (before the lock) names BL2's drive.py under the checkout's archive/."""
    monkeypatch.setattr(xc, 'archive_reads', lambda arm, **kwargs: [])
    assert W.run_cell(arm, FEAS, FEAS_PAIR, deps=deps(arm, env_run=bl2.run)) == xc.EXIT['infra_failure']
    batch = hc.batch_root(xc.roots(FEAS)[0])
    refusals = [json.loads(p.read_text()) for p in sorted(batch.glob('*/refusals/*.json'))]
    mine = [r for r in refusals if r['arm'] == arm and r['pair'] == FEAS_PAIR]
    assert [r['stage'] for r in mine] == ['spy_breach'] and str(production.code/xc.ARCHIVE_HARNESS/'drive.py') in mine[0]['reason']
    assert not xc.cell_paths(arm, FEAS, FEAS_PAIR)['result'].exists()
