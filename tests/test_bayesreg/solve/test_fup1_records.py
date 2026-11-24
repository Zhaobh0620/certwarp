"""Records and helpers of work package FUP1 (review MEM_复核_20261031 minor 1, 4, 5; no value changes):
- bayesreg.heap opens "libc.so.6" directly (no ctypes.util.find_library, which starts `ldconfig -p`), falls back to
  CDLL(None) when that fails (FUP1 review minor 4) and degrades to a no-op returning None when no library with
  malloc_trim can be opened;
- run_pair's env record (env.txt, provenance.json env) carries BAYESREG_MEMTRACE and BAYESREG_MEMTRACE_RESET_HWM,
  "未设" when unset;
- experiments/mem/analyze.py names the prefix verdict prefix_identical (it was "identical") and says what it compared.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from bayesreg import heap
from bayesreg.cli import run_pair

REPO = Path(__file__).resolve().parents[3]


def test_heap_opens_libc_without_find_library(monkeypatch):
    import ctypes.util

    def forbidden(name):
        raise AssertionError("ctypes.util.find_library starts an ldconfig child process")

    monkeypatch.setattr(ctypes.util, "find_library", forbidden)
    monkeypatch.setitem(heap._TRIM, "fn", False)                    # restored after the test
    assert heap.LIBRARIES[0] == "libc.so.6"
    assert heap.release_free_heap() in (0, 1)                        # glibc on this machine
    monkeypatch.setitem(heap._TRIM, "fn", False)
    monkeypatch.setattr(heap, "LIBRARIES", ("libfup1_no_such_library.so.9",))
    assert heap.release_free_heap() is None                          # safe no-op
    assert heap._TRIM["fn"] is None


def test_heap_falls_back_to_the_process_symbols(monkeypatch):
    """FUP1 review minor 4: when CDLL("libc.so.6") fails, the second entry CDLL(None) (the symbols already loaded in
    the process; glibc's malloc_trim on this machine) is used."""
    import ctypes
    real = ctypes.CDLL
    tried = []

    def cdll(name, *args, **kwargs):
        tried.append(name)
        if name == "libc.so.6":
            raise OSError("libc.so.6: refused by the test")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(ctypes, "CDLL", cdll)
    monkeypatch.setitem(heap._TRIM, "fn", False)                    # restored after the test
    assert heap.release_free_heap() in (0, 1)
    assert tried == ["libc.so.6", None] and heap._TRIM["fn"] is not None


def test_env_record_names_the_memtrace_switches(monkeypatch):
    monkeypatch.delenv("BAYESREG_MEMTRACE", raising=False)
    monkeypatch.delenv("BAYESREG_MEMTRACE_RESET_HWM", raising=False)
    record = run_pair.env_record()
    assert record["memtrace_env"] == dict(BAYESREG_MEMTRACE="未设", BAYESREG_MEMTRACE_RESET_HWM="未设")
    monkeypatch.setenv("BAYESREG_MEMTRACE", "/tmp/marks.jsonl")
    monkeypatch.setenv("BAYESREG_MEMTRACE_RESET_HWM", "1")
    assert run_pair.env_record()["memtrace_env"] == dict(BAYESREG_MEMTRACE="/tmp/marks.jsonl",
                                                         BAYESREG_MEMTRACE_RESET_HWM="1")
    json.dumps(run_pair.env_record(), ensure_ascii=False)


def _analyze():
    spec = importlib.util.spec_from_file_location("fup1_mem_analyze", REPO/"experiments"/"mem"/"analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_dir(root, rows, final, status):
    root.mkdir(parents=True)
    (root/"log.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (root/"cert.json").write_text(json.dumps(dict(state_hash=final)))
    (root/"status.json").write_text(json.dumps(dict(status=status)))
    return root


def test_analyze_names_the_prefix_verdict(tmp_path):
    analyze = _analyze()
    assert analyze.TIME_KEYS == ("seconds", "elapsed", "_unix", "time", "budget", "started")
    common = [dict(kind="candidate", tau=.5, seconds_gpu=1.), dict(kind="accept", state_hash="h1", seconds=2.)]
    a = _run_dir(tmp_path/"a", common+[dict(kind="budget", at="x")], "fa", "budget_exhausted")
    b = _run_dir(tmp_path/"b", [dict(common[0], seconds_gpu=9.), dict(common[1], seconds=7.),
                                dict(kind="accept", state_hash="h2"), dict(kind="budget", at="y")],
                 "fb", "budget_exhausted")
    prefix = analyze.compare(a, b, prefix=True)
    assert prefix["prefix_identical"] is True and "identical" not in prefix
    assert prefix["final_identical"] is False and "prefix" in prefix["compared"]
    assert prefix["removed_keys_containing"] == list(analyze.TIME_KEYS)
    whole = analyze.compare(a, a)
    assert whole["identical"] is True and "prefix_identical" not in whole and whole["compared"].startswith("whole")
    assert analyze.compare(a, b)["identical"] is False
