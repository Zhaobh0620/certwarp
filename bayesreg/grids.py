"""C03.6 / C12 four-level grids and Q1 level transfer, copied from the frozen mainline.

Source: seq_vp_3d tag freeze-20261022, commit eb808be, file jd_curl/three_d/multiresolution.py
(sha256 4a9b6b07241d841beda8155e5e572e0fe596e0fac23a48750477e3c14f0ed501 of `git show
eb808be:jd_curl/three_d/multiresolution.py`; identical to archive/seq_vp_3d/jd_curl/three_d/multiresolution.py):
  - FACTORS                 line 25
  - four_level_grids        lines 45-52
  - prolong_displacement    lines 55-86
Function bodies are verbatim. Only the imports changed: the original module also imported the
archived solver modules; here only the helpers these two functions use are imported, all from
modules that remain in jd_curl/ unchanged. tests/test_bayesreg/io_seg/test_grids.py compares both
functions value by value against the eb808be source, checks that the archive copy equals the eb808be
blob, and checks that every jd_curl module the helpers come from (with its jd_curl imports,
transitively) is byte-identical to its eb808be blob.
"""
import numpy as np

from jd_curl.validation import InputValidationError, as_float64_array, readonly_copy
from jd_curl.three_d.grid import Grid3D
from jd_curl.three_d.nifti import coarse_grid
from jd_curl.three_d.validation import zero_boundary

SOURCE_COMMIT = "eb808be"
SOURCE_PATH = "jd_curl/three_d/multiresolution.py"
SOURCE_SHA256 = "4a9b6b07241d841beda8155e5e572e0fe596e0fac23a48750477e3c14f0ed501"

FACTORS = (8, 4, 2, 1)


def four_level_grids(native):
    """Preserve physical node-center FOV, including the original native grid."""
    if not isinstance(native, Grid3D) or native.unit != "mm":
        raise InputValidationError("four-level registration requires a physical mm Grid3D")
    shapes = [tuple((n-2+s)//s+1 for n in native.shape) for s in FACTORS]
    if any(n < 5 for shape in shapes for n in shape):
        raise InputValidationError("all four levels need >=5 nodes on every axis")
    return tuple(coarse_grid(native, shape) for shape in shapes[:-1])+(native,)


def prolong_displacement(u, source, target):
    """Physical Q1 interpolation only; caller MUST certify the new Q1 map.

    Separable interpolation avoids allocating a native dense query-coordinate
    volume. No displacement amplitude scaling or boundary clipping is used.
    """
    if source.unit != target.unit or source.unit != "mm":
        raise InputValidationError("transfer requires matching physical mm units")
    u = as_float64_array(u, "transfer displacement", source.shape+(3,))
    if not zero_boundary(u):
        raise InputValidationError("transfer requires zero displacement on all six faces")
    result = u
    for axis, (old, new, h) in enumerate(zip(source.axes[::-1], target.axes[::-1], source.spacing[::-1])):
        tolerance = 32*np.finfo(float).eps*max(1., np.max(np.abs(old)), np.max(np.abs(new)))
        if len(new) < len(old) or not np.allclose(old[[0, -1]], new[[0, -1]], rtol=0., atol=tolerance):
            raise InputValidationError("transfer must refine the same physical FOV")
        t = (new-old[0])/h
        # Identical analytical FOV endpoints can differ by construction roundoff.
        # Pin just their interpolation coordinates, not any physical displacement.
        t[0], t[-1] = 0., len(old)-1
        if np.any(t[1:-1] <= 0) or np.any(t[1:-1] >= len(old)-1):
            raise InputValidationError("target interior is not inside the source FOV")
        left = np.floor(t).astype(np.intp)
        left[-1] = len(old)-2
        weight = t-left
        shape = [1]*4
        shape[axis] = len(new)
        weight = weight.reshape(shape)
        result = (1-weight)*np.take(result, left, axis=axis)+weight*np.take(result, left+1, axis=axis)
    if not zero_boundary(result):
        raise InputValidationError("Q1 transfer did not preserve the six zero faces")
    return readonly_copy(result)
