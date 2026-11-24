"""_cases with the C05-2 model (logvol_points "gauss27", the ModelState default before DEF; work package DEF).

DEF (2026-10-31; contract 8a4d060 C05-2b, D48) made gauss27+closed27 the ModelState default.  The Laplace / prior-side
tests whose dense oracle (`dense_model`) is that of C05-2 take their states from here: everything of _cases, with
`scaled_state` and `cube6` giving "gauss27" explicitly (model/_gauss27.gauss27: the state _cases built before DEF, bit
for bit).  _cases.py itself is unchanged (tests/test_bayesreg/_golden/make_golden_eb.py requires it byte-identical to
the commit it extracts).
"""
from . import _cases as _base
from ._cases import (CertOutcome, NumpyBackend, RecordingDraw, ScriptedCertificate, dense_blocks,  # noqa: F401
                     dense_operator, u_hat_of)
from ..model._gauss27 import gauss27


def scaled_state(grid, factor=1., **kwargs):
    return gauss27(_base.scaled_state(grid, factor, **kwargs))


def cube6(seed=1, factor=1., **kwargs):
    grid, state = _base.cube6(seed, factor, **kwargs)
    return grid, gauss27(state)
