"""Read-only arrays without redundant copies (work package MEM; memory only, every value unchanged).

jd_curl.validation.readonly_copy always copies.  Two cases need no copy, and the helpers below skip it:
  adopt_readonly(value)  `value` is already an ndarray of the dtype that owns its C-contiguous buffer and is read-only
                         (a readonly_copy made earlier, the array of another ModelState or LevelInput): it is returned
                         itself -- the same values in the same kind of array;
  freeze_new(array)      `array` was just computed by the caller (a fresh ufunc / np.stack / np.concatenate result that
                         owns its C-contiguous float64 buffer and that nothing else references): it is made read-only
                         in place.
Anything else falls back to the copy, exactly as before.  Note: a read-only flag is not a hard guarantee in NumPy (an
owner can set writeable back to True); the arrays shared here are only ever read, as their copies were.
"""
import numpy as np

from jd_curl.validation import readonly_copy


def _owned(array, dtype):
    return (type(array) is np.ndarray and array.dtype == dtype and array.base is None and array.flags.c_contiguous)


def adopt_readonly(value, dtype=np.float64):
    """`value` itself when it is an owned, C-contiguous, read-only ndarray of `dtype`; else a read-only C-order copy
    (readonly_copy for float64)."""
    if _owned(value, dtype) and not value.flags.writeable:
        return value
    if dtype == np.float64:
        return readonly_copy(value)
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.flags.writeable = False
    return array


def freeze_new(array):
    """A float64 array the caller has just allocated, made read-only in place; readonly_copy when it is not an owned
    C-contiguous float64 ndarray."""
    if _owned(array, np.float64):
        array.flags.writeable = False
        return array
    return readonly_copy(array)
