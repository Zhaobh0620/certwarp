"""EVF: bayesreg.eval.cli changes no environment variable when imported or when main() is called in-process; run as
the program it still evaluates on the CPU with numba 2 threads (CUDA_VISIBLE_DEVICES="", NUMBA_NUM_THREADS=2 by
default). Before EVF the module set both at import, which hid the GPU from a whole pytest session and its children."""
import json
import os
import subprocess
import sys

from bayesreg.eval import cli

from . import _synthetic as syn

KEYS = ("CUDA_VISIBLE_DEVICES", "NUMBA_NUM_THREADS")
MODULE = cli.__name__
P01 = f"{syn.DEV[0]}__{syn.DEV[1]}"

IMPORT_PROBE = """
import json, os, sys
keys = ("CUDA_VISIBLE_DEVICES", "NUMBA_NUM_THREADS")
before = {k: os.environ.get(k) for k in keys}
import importlib
module = importlib.import_module(sys.argv[1])
print(json.dumps(dict(before=before, after={k: os.environ.get(k) for k in keys}, file=module.__file__,
                      numba="numba" in sys.modules, torch="torch" in sys.modules)))
"""

# Runs the module as __main__ the way python -m does (runpy), then records the process's environment and, when they
# were loaded by the evaluation, numba's thread count and whether torch initialized CUDA.
RUN_PROBE = """
import importlib.util, json, os, runpy, sys
out, module = sys.argv[1], sys.argv[2]
origin = importlib.util.find_spec(module).origin
sys.argv = [module, *sys.argv[3:]]
code = None
try:
    runpy.run_module(module, run_name="__main__", alter_sys=True)
except SystemExit as exc:
    code = exc.code
rec = dict(exit_code=code, env={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "NUMBA_NUM_THREADS")},
           numba_loaded="numba" in sys.modules, torch_loaded="torch" in sys.modules, file=origin)
if rec["numba_loaded"]:
    import numba
    rec.update(numba_config_threads=numba.config.NUMBA_NUM_THREADS, numba_threads=numba.get_num_threads())
if rec["torch_loaded"]:
    import torch
    rec["torch_cuda_initialized"] = torch.cuda.is_initialized()
with open(out, "w") as stream:
    json.dump(rec, stream)
"""


def child_env(**values):
    """The test process's environment without KEYS (plus `values`), single-thread BLAS, the repository on the path."""
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(syn.REPO), **values)
    return env


def synthetic_run(tmp_path):
    data = tmp_path/"data"
    syn.write_pair(data, P01, syn.DEV[0], syn.DEV[1], syn.pair_arrays(seed=1))
    allowlist = syn.write_allowlist(tmp_path/"dev_subjects.json")
    prov = dict(pair_sha256=syn.file_sha256(data/syn.STAGE/P01/"pair.npz"), fixed_subject=syn.DEV[0],
                moving_subject=syn.DEV[1])
    run, _ = syn.write_bayesreg_run(tmp_path/"runs"/"b1", P01, syn.d_to_u(syn.smooth_d()), pair_provenance=prov)
    (tmp_path/"out").mkdir()
    return run, ["--pairs-stage", syn.STAGE, "--data-root", str(data), "--allowlist", str(allowlist),
                 "--out", "metrics.json", "--long", str(tmp_path/"long.csv"), "--out-dir", str(tmp_path/"out")]


def test_import_changes_no_environment_variable(tmp_path):
    """A fresh interpreter: importing the module leaves both variables as they were (absent, or set by the caller),
    and loads neither numba nor torch (so the __main__ defaults are in place before either loads)."""
    for preset in ({}, dict(CUDA_VISIBLE_DEVICES="0", NUMBA_NUM_THREADS="3")):
        done = subprocess.run([sys.executable, "-B", "-c", IMPORT_PROBE, MODULE], capture_output=True, text=True,
                              env=child_env(**preset), cwd=str(tmp_path), timeout=300)
        assert done.returncode == 0, done.stdout+done.stderr
        rec = json.loads(done.stdout.strip().splitlines()[-1])
        assert rec["before"] == {k: preset.get(k) for k in KEYS}
        assert rec["after"] == rec["before"], rec
        assert os.path.realpath(rec["file"]) == os.path.realpath(cli.__file__)
        assert not rec["numba"] and not rec["torch"], rec


def test_main_in_process_changes_no_environment_variable(tmp_path, monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    run, args = synthetic_run(tmp_path)
    assert cli.main([str(run), *args, "--cert", "none"]) == 0
    assert {k: os.environ.get(k) for k in KEYS} == dict.fromkeys(KEYS)


def test_cli_subprocess_evaluates_on_the_cpu_with_two_numba_threads(tmp_path):
    """As the program (runpy, run_name __main__, as python -m does) with neither variable set: the process hides the
    GPU and numba, loaded by the full certificate, runs 2 threads; the evaluation itself is unchanged (exit 0, the
    certificate passes)."""
    run, args = synthetic_run(tmp_path)
    out = tmp_path/"probe.json"
    done = subprocess.run([sys.executable, "-B", "-c", RUN_PROBE, str(out), MODULE, str(run), *args],
                          capture_output=True, text=True, env=child_env(), cwd=str(tmp_path), timeout=600)
    assert done.returncode == 0, done.stdout+done.stderr
    rec = json.loads(out.read_text())
    assert rec["exit_code"] == 0, done.stdout+done.stderr
    assert os.path.realpath(rec["file"]) == os.path.realpath(cli.__file__)
    assert rec["env"] == dict(CUDA_VISIBLE_DEVICES="", NUMBA_NUM_THREADS="2")
    assert rec["numba_loaded"] and rec["numba_config_threads"] == 2 and rec["numba_threads"] == 2, rec
    assert not rec["torch_loaded"] or not rec["torch_cuda_initialized"], rec
    metrics = json.loads((tmp_path/"out"/"b1"/"metrics.json").read_text())
    assert metrics["certificate"]["mode"] == "full" and metrics["certificate"]["passed"]
