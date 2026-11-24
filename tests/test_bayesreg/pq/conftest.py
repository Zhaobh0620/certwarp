"""The sandbox of the PQ tests: a synthetic world in tmp_path whose registry counts as a formal one (bayesreg.seg.structure_layer.FORMAL_REGISTRIES gets its path), the units' machine gate and code record
replaced by records (their own tests use the real functions on fakes).  Nothing of the real results directory is read or written by a test."""
import pytest

from bayesreg.seg import structure_layer

from . import _world as W


@pytest.fixture
def world(tmp_path, monkeypatch, request):
    w = W.build(tmp_path, synthetic=getattr(request, 'param', None) == 'synthetic_folds')        # synthetic_folds('world'): C23.9-13 (8), revision 6e
    monkeypatch.setattr(structure_layer, 'FORMAL_REGISTRIES', tuple(structure_layer.FORMAL_REGISTRIES)+(w.registry,))
    return w


GATE = dict(passed=True, note='the tests query no machine')
CODE = dict(commit='0'*40, clean=True, note='the tests record no code tree')
