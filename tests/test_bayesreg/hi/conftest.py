"""The sandbox of every HI test: the results root, the two locks and the development check's directory point into tmp_path, the decision is None (as in the code); on the CPU the environment check and the
machine gate are replaced (their own tests use the real ones, _world.REAL).  `decided`: a development check that ran (its lock and its devcheck.json at DEVCHECK_OUT) and a decision that holds, for the tests that
run the hold-out entry; the tests that run the check itself do not take it (the check's lock would refuse them).  Nothing of the real results directory is read or written by a test through these names."""
import types

import pytest

from . import _world as W


@pytest.fixture(autouse=True)
def hi_sandbox(tmp_path, monkeypatch):
    hi = W.hi
    monkeypatch.setattr(hi, 'RESULTS', tmp_path)
    monkeypatch.setattr(hi, 'HOLDOUT_LOCK', tmp_path/'locks'/'hi_holdout_infer.lock')
    monkeypatch.setattr(hi, 'DEVCHECK_LOCK', tmp_path/'locks'/'hi_devcheck.lock')
    monkeypatch.setattr(hi, 'DEVCHECK_OUT', tmp_path/'seg'/'hi_devcheck_v1')
    monkeypatch.setattr(hi, 'DEVCHECK_DECISION', None)
    monkeypatch.setattr(hi, 'check_environment', lambda device, decision: hi.environment_of(device))
    monkeypatch.setattr(hi, 'machine_gate', lambda **kwargs: dict(passed=True, note='the tests query no machine', **kwargs))
    monkeypatch.setattr(hi, 'HOLDOUT_OUT_SUGGESTED', tmp_path/'out')        # A5a: the one root of the hold-out entry (infer_holdout and run_hi refuse any other), under the sandbox's results root
    monkeypatch.setattr(hi, 'pinned_code_check', lambda: (dict(passed=True, pinned='c'*40, head='c'*40, note='test: rule one replaced'), []))   # A5a: P1 = the tests' code commit; its own test runs the real one
    return types.SimpleNamespace(results=tmp_path, lock=tmp_path/'locks'/'hi_holdout_infer.lock')


@pytest.fixture
def decided(monkeypatch):
    hi = W.hi
    hi.take_lock(hi.DEVCHECK_LOCK, dict(out=str(hi.DEVCHECK_OUT), plan_sha256=hi.DEVCHECK_PLAN_SHA256, backends=['hs_a', 'segment_subject'], event=hi.FIRST_RUN, utc='2026-11-07T13:00:00Z', pid=1))
    devcheck_json = W.devcheck_document()
    decision = W.decision_for(devcheck_json)
    monkeypatch.setattr(hi, 'DEVCHECK_DECISION', decision)
    return types.SimpleNamespace(decision=decision, devcheck_json=devcheck_json)
