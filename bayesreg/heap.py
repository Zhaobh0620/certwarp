"""Hand freed heap memory back to the operating system (work package MEM; allocator only, no value read or changed).

glibc serves allocations below its mmap threshold from per-thread arenas and keeps what is freed there for reuse; the
threshold grows dynamically up to 32 MiB once a larger mmapped block is freed, so the per-tile arrays of the threaded
CPU reference (bayesreg.geom.parallel workers, the certificate's fallback workers) land in the arenas.  In the
measurement M02 (runs/20261031-mem/result.md) about 0.8 GiB of such freed memory stayed resident from the coarse levels
and the native G0 through the whole native PG solve (mallinfo2 fordblks).  `release_free_heap()` calls glibc's
malloc_trim(0), which returns the free pages of every arena to the kernel (the arenas stay usable; a later allocation
simply faults the pages in again).  On a C library without malloc_trim it does nothing.
The library is opened as "libc.so.6" (glibc's soname), else as the symbols already loaded in the process
(ctypes.CDLL(None)); ctypes.util.find_library is not used: on Linux it starts an `ldconfig -p` child process, and the
first call comes from a process holding several GiB (FUP1, review MEM minor 1; bayesreg.memtrace opens "libc.so.6"
the same way).  Any failure (no such library, no malloc_trim symbol: musl, macOS, Windows) leaves release_free_heap a
no-op returning None.
"""
import ctypes

_TRIM = dict(fn=False)
LIBRARIES = ("libc.so.6", None)          # glibc's soname; None: the symbols of the running process


def _trim():
    if _TRIM["fn"] is False:
        _TRIM["fn"] = None
        for name in LIBRARIES:
            try:
                function = ctypes.CDLL(name).malloc_trim
            except (OSError, AttributeError, TypeError):
                continue
            function.argtypes = [ctypes.c_size_t]
            function.restype = ctypes.c_int
            _TRIM["fn"] = function
            break
    return _TRIM["fn"]


def release_free_heap():
    """malloc_trim(0): 1 when memory was returned, 0 when there was none, None without glibc's malloc_trim."""
    function = _trim()
    return None if function is None else int(function(0))
