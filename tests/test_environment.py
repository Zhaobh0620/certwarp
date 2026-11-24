"""Stage 0: import and library API checks, not solver verification."""

import inspect
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import LinearOperator, cg

import jd_curl


def test_package_imports_from_this_project():
    root = Path(__file__).resolve().parents[1]
    assert Path(jd_curl.__file__).resolve() == root / "jd_curl" / "__init__.py"
    assert jd_curl.__version__ == "0.1.0"


def test_required_library_interfaces():
    assert {"rtol", "atol", "M", "callback"} <= set(inspect.signature(cg).parameters)
    for transform in (dstn, idstn):
        assert {"type", "norm", "axes"} <= set(inspect.signature(transform).parameters)
    assert {"dx", "dy"} <= set(inspect.signature(RectBivariateSpline.ev).parameters)
    assert callable(LinearOperator)
    assert np.dtype(np.float64).itemsize == 8
