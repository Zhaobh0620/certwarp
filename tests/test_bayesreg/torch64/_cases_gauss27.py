"""_cases with the C05-2 model (logvol_points "gauss27", the ModelState default before DEF; work package DEF).

DEF (2026-10-31; contract 8a4d060 C05-2b, D48) made gauss27+closed27 the ModelState default.  The torch64 tests whose
reference is the C05-2 model (the numpy reference of the Gauss27 blocks, the pre-OPT1 reference _opt1_reference, a
"gauss27" parametrization) take their states from here: `case` and `medium_state` of _cases with "gauss27" explicitly
(model/_gauss27.gauss27: the state _cases built before DEF, bit for bit).  _cases.py itself is unchanged
(tests/test_bayesreg/_golden/make_golden.py requires it byte-identical to the commit it extracts).
"""
from . import _cases as _base
from ._cases import smooth_field  # noqa: F401
from ..model._gauss27 import gauss27


def case(name):
    """_cases.case(name) with logvol_points "gauss27": (state, u, torch-state options)."""
    state, u, options = _base.case(name)
    return gauss27(state), u, options


def medium_state():
    return gauss27(_base.medium_state())
