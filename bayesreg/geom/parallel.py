"""Ordered thread parallelism of the CPU FP64 reference (work package CPUPAR; contract C06-7, C08-1, C17-4).

`ordered_map(function, items, threads)` yields function(item) for the items in their given order.

- threads == 1, a single item, or a call from inside one of its own workers: the plain serial loop, inline in
  the caller's thread and lazy (an item is computed only when the consumer asks for it).
- otherwise: one persistent ThreadPoolExecutor per thread count computes at most `window` (default 2 threads)
  items ahead of the consumer; results and exceptions are handed back strictly in item order.

Why the parallel path is bitwise the serial one (the reason the thread count is not part of any cache key):
- every item is computed by the same code on the same inputs.  Workers only read shared arrays (the node
  fields, the ModelState) and return newly allocated arrays or scalars; they never write an accumulator;
- every reduction across items stays with the consumer, in ascending item order: the longdouble tile sums of
  C06-7 / C08-1 and the node scatters of the adjoints (each node receives its additions in the serial order);
- NumPy 2 keeps the floating-point error state and the ufunc buffer size per context, and a new thread does not
  inherit them (the default buffer size 8192 sets the grouping of every longdouble np.sum).  Each job runs under
  the caller's np.geterr(), np.geterrcall() and np.getbufsize(), so a job raises exactly where the serial loop
  raises and groups its sums the same way;
- BLAS: the (cells, 8) @ (8, 27) products split no inner dimension, so their values do not depend on the
  calling thread or on OpenBLAS's own thread count.
Context variables other than NumPy's (e.g. a jd_curl work session) are not propagated: the consumer, not the
workers, observes them (jd_curl.three_d.work_control).

Early exit: closing the generator (the consumer returns, or raises) cancels the queued items and waits for the
running ones, so no job outlives the call.  The first exception in item order is re-raised by the consumer
exactly where the serial loop would raise it; exceptions of later items are discarded with them.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
import threading

import numpy as np

from jd_curl.validation import InputValidationError

# Default thread count of the CPU reference (ModelState, SolverConfig, run_pair --cpu-threads): 4, not 8 (FB1, review
# LVC_CPUPAR important item 3).  The machine (WSL, 15 GB RAM) runs at most one native-size run at a time, possibly next
# to a segmenter training or the other project's chain; 4 workers leave cores and memory for them.  Memory grows about
# linearly with the thread count (one tile's working set per worker in flight; [inferred, not measured] at native size
# about 0.3 GB serial, 2-2.5 GB with 8 threads in stable_reduction, ~30 % more with the C05-2a corners), so run_pair
# records the thread count and the peak RSS (status.json resources).  Values are bitwise the same for every count
# (CPUPAR), so the default changes no result.
DEFAULT_CPU_THREADS = 4
_PREFIX = "bayesreg-cpu"
_LOCK = threading.Lock()
_POOLS = {}          # thread count -> ThreadPoolExecutor; never shut down, so concurrent callers never race


def cpu_threads(value, name="cpu_threads"):
    """A thread count: integer >= 1 (bool refused)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise InputValidationError(f"{name}: integer >= 1 required")
    return int(value)


def _pool(threads):
    with _LOCK:
        pool = _POOLS.get(threads)
        if pool is None:
            pool = _POOLS[threads] = ThreadPoolExecutor(max_workers=threads, thread_name_prefix=f"{_PREFIX}{threads}")
        return pool


def in_worker():
    """True inside a worker of this module (a nested ordered_map then runs serially: no pool waits on itself)."""
    return threading.current_thread().name.startswith(_PREFIX)


def _job(function, item, errors, call, bufsize):
    previous = np.getbufsize()
    if previous != bufsize:
        np.setbufsize(bufsize)
    try:
        with np.errstate(call=call, **errors):
            return function(item)
    finally:
        if previous != bufsize:
            np.setbufsize(previous)


def ordered_map(function, items, threads, *, window=None):
    """Generator of function(item) in item order; see the module docstring.  Use it under
    contextlib.closing (or exhaust it) so that an early exit cancels the work still queued."""
    threads = cpu_threads(threads, "threads")
    items = list(items)
    if threads == 1 or len(items) <= 1 or in_worker():
        for item in items:
            yield function(item)
        return
    window = 2*threads if window is None else cpu_threads(window, "window")
    errors, call, bufsize = np.geterr(), np.geterrcall(), np.getbufsize()
    pool = _pool(threads)
    pending = deque()
    upcoming = iter(items)
    try:
        for item in upcoming:
            pending.append(pool.submit(_job, function, item, errors, call, bufsize))
            if len(pending) >= window:
                break
        while pending:
            result = pending.popleft().result()
            item = next(upcoming, _END)
            if item is not _END:
                pending.append(pool.submit(_job, function, item, errors, call, bufsize))
            yield result
    finally:
        for future in pending:
            future.cancel()
        wait(pending)


_END = object()


def provenance(threads):
    return dict(cpu_threads=int(threads), cpu_parallel=(
        "bayesreg.geom.parallel.ordered_map: Gauss-point tiles (geometry, block values, adjoint local products) "
        "and spline chunks computed by a thread pool; longdouble tile sums and node scatters in ascending tile "
        "order by the consumer; bitwise identical for every thread count (CPUPAR)"))
