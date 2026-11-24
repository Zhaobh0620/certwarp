"""The C05-2 model (log-volume prior on the Gauss27 points only) for the tests written for it (work package DEF).

DEF (2026-10-31; contract 8a4d060: C05-2b first paragraph, D43, D48) made logvol_points = "gauss27+closed27" (beta_S 1,
Simpson) the ModelState default.  The tests whose oracles, serial references or golden digests are those of the C05-2
model -- the ModelState default before DEF -- give "gauss27" explicitly through these helpers, so that they check
exactly what they checked before DEF.  _helpers.py and the other helper modules are left unchanged:
tests/test_bayesreg/_golden/make_golden.py and make_golden_eb.py require them byte-identical to the commits they
extract.

    make_state(grid, **kwargs)   _helpers.make_state(grid, **kwargs) with logvol_points "gauss27"
    gauss27(state)               state.replace(logvol_points="gauss27")

`replace` rebuilds the state from the same init fields (the same arrays, samplers and draws), so the result is the
state that _helpers.make_state built before DEF, bit for bit.
"""
from . import _helpers

GAUSS27 = dict(logvol_points="gauss27")


def gauss27(state):
    """The state with logvol_points "gauss27" (the C05-2 model; the ModelState default before DEF)."""
    return state.replace(**GAUSS27)


def make_state(grid, **kwargs):
    """_helpers.make_state with logvol_points "gauss27" (the C05-2 model; the default before DEF)."""
    return gauss27(_helpers.make_state(grid, **kwargs))
