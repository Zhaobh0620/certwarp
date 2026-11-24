"""Memory marks of a run (work package MEM; a diagnostic, default off; no value of the solver is read or changed).

    BAYESREG_MEMTRACE=<path>            mark(label, **fields) appends one JSON line to <path>
    BAYESREG_MEMTRACE_RESET_HWM=1       also reset the peak RSS after every mark (`echo 5 > /proc/self/clear_refs`), so
                                        that hwm_mb is the peak since the previous mark (default off: hwm_mb is the
                                        lifetime peak)

Without BAYESREG_MEMTRACE `mark` returns at once (one module-global test): the hooks in run_pair, solve_multilevel,
solve_direct and run_pg cost nothing measurable and touch no array.

One line: t (time.time), mono (time.perf_counter), seq, label, the caller's fields, and
  host   rss_mb, hwm_mb, swap_mb, rss_anon_mb, rss_file_mb (/proc/self/status VmRSS, VmHWM, VmSwap, RssAnon, RssFile;
         MiB), hwm_lifetime_mb (the largest hwm_mb seen so far: the lifetime peak also under the HWM reset);
  cgroup cg_mb, cg_peak_mb, cg_swap_mb, cg_swap_peak_mb (memory.current, memory.peak, memory.swap.current,
         memory.swap.peak of the process's cgroup v2 directory; MiB; null where absent), cg_anon_mb, cg_file_mb,
         cg_anon_thp_mb (memory.stat anon, file, anon_thp);
  malloc malloc_arena_mb, malloc_mmap_mb, malloc_inuse_mb, malloc_free_mb (glibc mallinfo2 over all arenas: bytes of
         the arena heaps, of mmapped chunks, in use, free inside the arenas; MiB);
  device (only when torch is imported and CUDA is initialized) cuda_mb, cuda_peak_mb, cuda_reserved_mb
         (torch.cuda.memory_allocated, max_memory_allocated, memory_reserved; MiB); the device peak is reset after every
         mark, so cuda_peak_mb is the peak since the previous mark.
Note: with the HWM reset, resource.getrusage ru_maxrss (run_pair's peak_rss_mb) no longer covers the whole process
lifetime; use hwm_lifetime_mb and the cgroup's memory.peak instead.
"""
import json
import os
import sys
import time

ENV = "BAYESREG_MEMTRACE"
ENV_RESET = "BAYESREG_MEMTRACE_RESET_HWM"
MIB = 1024.*1024.
_STATUS_KEYS = dict(VmRSS="rss_mb", VmHWM="hwm_mb", VmSwap="swap_mb", RssAnon="rss_anon_mb", RssFile="rss_file_mb")
_STAT_KEYS = dict(anon="cg_anon_mb", file="cg_file_mb", anon_thp="cg_anon_thp_mb")

_path = os.environ.get(ENV) or None
_reset = os.environ.get(ENV_RESET, "0") == "1"
_state = dict(stream=None, cgroup=False, seq=0, hwm=0.)


def enabled():
    return _path is not None


def configure(path, *, reset_hwm=False):
    """Switch the marks on (path) or off (None) inside a process (tests); the environment does it for runs."""
    global _path, _reset
    if _state["stream"] is not None:
        _state["stream"].close()
    _path, _reset = (None if path is None else str(path)), bool(reset_hwm)
    _state.update(stream=None, cgroup=False, seq=0, hwm=0.)


def _cgroup_dir():
    if _state["cgroup"] is False:
        directory = None
        try:
            with open("/proc/self/cgroup", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("0::"):
                        candidate = "/sys/fs/cgroup"+line[3:].strip()
                        if os.path.isfile(os.path.join(candidate, "memory.current")):
                            directory = candidate
        except OSError:
            directory = None
        _state["cgroup"] = directory
    return _state["cgroup"]


def _read_mib(path):
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read().strip()
    except OSError:
        return None
    return None if text == "max" else int(text)/MIB


def _host():
    out = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in _STATUS_KEYS:
                    out[_STATUS_KEYS[key]] = int(rest.split()[0])/1024.
    except OSError:
        pass
    return out


def _cgroup():
    directory = _cgroup_dir()
    if directory is None:
        return {}
    out = {name: _read_mib(os.path.join(directory, f)) for name, f in (
        ("cg_mb", "memory.current"), ("cg_peak_mb", "memory.peak"), ("cg_swap_mb", "memory.swap.current"),
        ("cg_swap_peak_mb", "memory.swap.peak"))}
    try:
        with open(os.path.join(directory, "memory.stat"), encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(" ")
                if key in _STAT_KEYS:
                    out[_STAT_KEYS[key]] = int(value)/MIB
    except OSError:
        pass
    return out


_MALLINFO = dict(fn=False)


def _malloc():
    """glibc mallinfo2 over all arenas: bytes obtained by the arenas (sbrk / arena heaps), mmapped chunks, in use
    and free inside the arenas (MiB); {} without glibc >= 2.33."""
    if _MALLINFO["fn"] is False:
        try:
            import ctypes

            class Info(ctypes.Structure):
                _fields_ = [(name, ctypes.c_size_t) for name in ("arena", "ordblks", "smblks", "hblks", "hblkhd",
                                                                 "usmblks", "fsmblks", "uordblks", "fordblks",
                                                                 "keepcost")]
            function = ctypes.CDLL("libc.so.6").mallinfo2
            function.restype = Info
            _MALLINFO["fn"] = function
        except (OSError, AttributeError):
            _MALLINFO["fn"] = None
    if _MALLINFO["fn"] is None:
        return {}
    info = _MALLINFO["fn"]()
    return dict(malloc_arena_mb=info.arena/MIB, malloc_mmap_mb=info.hblkhd/MIB, malloc_inuse_mb=info.uordblks/MIB,
                malloc_free_mb=info.fordblks/MIB)


def _device():
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    try:
        if not torch.cuda.is_initialized():
            return {}
        out = dict(cuda_mb=torch.cuda.memory_allocated()/MIB, cuda_peak_mb=torch.cuda.max_memory_allocated()/MIB,
                   cuda_reserved_mb=torch.cuda.memory_reserved()/MIB)
        torch.cuda.reset_peak_memory_stats()
        return out
    except Exception:                                             # noqa: BLE001 - a diagnostic never raises
        return {}


def mark(label, context=None, **fields):
    """Append one memory record (module docstring); a no-op unless BAYESREG_MEMTRACE is set.  `context`: a dict of
    the caller (e.g. level, round) merged first, so that its keys never collide with keyword fields at the call."""
    if _path is None:
        return
    record = dict(t=time.time(), mono=time.perf_counter(), seq=_state["seq"], label=label)
    if context:
        record.update({str(k): v for k, v in dict(context).items()})
    record.update(fields)
    record.update(_host())
    if "hwm_mb" in record:
        _state["hwm"] = max(_state["hwm"], record["hwm_mb"])
        record["hwm_lifetime_mb"] = _state["hwm"]
    record.update(_cgroup())
    record.update(_malloc())
    record.update(_device())
    if _reset:
        try:
            with open("/proc/self/clear_refs", "w", encoding="utf-8") as handle:
                handle.write("5")
        except OSError:
            record["hwm_reset"] = False
    try:
        if _state["stream"] is None:
            _state["stream"] = open(_path, "a", encoding="utf-8")
        _state["stream"].write(json.dumps(record, default=str)+"\n")
        _state["stream"].flush()
    except OSError:
        pass
    _state["seq"] += 1
