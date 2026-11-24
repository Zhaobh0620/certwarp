"""run_pair --init-field on the torch64 backend (work packages WST, WARM; contract C12-4a): the same interface and records as numpy, the
same entry state, the same step sequence (C09.1-3a gate's measure: every candidate's level, round, iteration, attempt, decision and reason;
F_B to rounding); the records leave the path of the same backend bit for bit (C12-4a item 7); the probe's combination with the driver's arguments (review WARM_tools I2:
structure channels, support_shared, the seed, checkpoints).  cpu device always; the cuda cases run only with WST_TEST_CUDA=1, GPU idle (the lead runs them).  Synthetic data only: the 26^3
pairs of solve/test_run_pair.py and experiments/m2v1/dry_inputs.py.
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")

from bayesreg.cli import run_pair                                    # noqa: E402
from bayesreg.solve import jd_scaling as jds                         # noqa: E402

from ..solve import _run_pair_digest as RD                           # noqa: E402
from ..solve.test_pilot_tools import load                            # noqa: E402
from ..solve.test_run_pair import read                               # noqa: E402
from ..solve.test_warm_tools import check_probe_combination          # noqa: E402

COMMON = ("--rounds", "1,1,1,1", "--max-accepted", "2")
CUDA = pytest.mark.skipif(os.environ.get("WST_TEST_CUDA") != "1",
                          reason="cuda only on request (WST_TEST_CUDA=1, GPU idle; the lead runs it)")
RECORD_KEYS = ("source_format", "arm", "oracle", "state_hash", "accepted_k", "scale", "entry_state_hash", "levels_skipped")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("wst_torch64")
    base = RD.write_inputs(root)
    jds.clear_cell_bounds_cache()
    source = root/"run_source"
    assert run_pair.main([*base, "--out", str(source), *COMMON]) == 0
    return root, base, source/"final_native.npz"


def seeded(world, name, *options):
    root, base, seed = world
    jds.clear_cell_bounds_cache()
    out = root/f"run_{name}"
    code = run_pair.main([*base, "--out", str(out), *COMMON, "--init-field", str(seed), "--start-level", "3",
                          "--checkpoint-accepted", "1", *options])
    assert code == 0, name
    return out


def candidates(out):
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    return [(r["level"], r["round"], r["iteration"], r["attempt"], r["accepted"], r["reason"], r["energy"]) for r in rows
            if r["kind"] == "candidate"]


def check_same(numpy_run, device_run):
    a, b = read(numpy_run/"hyper.json")["init_field"], read(device_run/"hyper.json")["init_field"]
    assert {k: a[k] for k in RECORD_KEYS} == {k: b[k] for k in RECORD_KEYS}
    assert a["entry_rows"][0]["cert"]["state_hash"] == b["entry_rows"][0]["cert"]["state_hash"]
    ca, cb = candidates(numpy_run), candidates(device_run)
    assert [c[:6] for c in ca] == [c[:6] for c in cb] and ca
    for x, y in zip(ca, cb):
        assert (x[6] is None) == (y[6] is None)
        if x[6] is not None:
            assert y[6] == pytest.approx(x[6], rel=1e-10)
    for name in ("status.json", "config.json", "provenance.json", "qc.json"):
        document = read(device_run/name)
        assert "init_field" in document or "init_field_oracle" in document, name
    status = read(device_run/"status.json")
    assert status["status"] == read(numpy_run/"status.json")["status"] and status["export"]["certificate_passed"] is True
    assert [r["level"] for r in read(device_run/"hyper.json")["rounds"]] == [3]
    assert read(device_run/"provenance.json")["switches"]["backend"] == "torch64"
    pa, pb = read(numpy_run/"hyper.json")["profile"], read(device_run/"hyper.json")["profile"]     # C12-4a item 8
    assert pa["n_checkpoints"] == pb["n_checkpoints"] == 2 and pa["accepted"] == pb["accepted"]
    for where in ("entry", "end"):
        assert pb[where]["L"] == pytest.approx(pa[where]["L"], rel=1e-10)
    # the entry state is the same bytes on both backends; after a step only to rounding (the backends are not bitwise, C08)
    assert pb["entry"]["state_hash"] == pa["entry"]["state_hash"]
    with np.load(device_run/"checkpoints"/"ckpt_001"/"disp.npz") as z, np.load(numpy_run/"checkpoints"/"ckpt_001"/"disp.npz") as y:
        assert np.max(np.abs(z["d"]-y["d"])) <= 1e-9
        assert json.loads(str(z["meta"]))["init_field"]["state_hash"] == json.loads(str(y["meta"]))["init_field"]["state_hash"]


def test_torch64_cpu_matches_numpy_with_a_seed(world):
    check_same(seeded(world, "numpy"), seeded(world, "torch64_cpu", "--backend", "torch64", "--device", "cpu"))


@CUDA
def test_torch64_cuda_matches_numpy_with_a_seed(world):
    check_same(seeded(world, "numpy_c"), seeded(world, "torch64_cuda", "--backend", "torch64", "--device", "cuda"))
    with np.load(world[2]) as z:
        assert z["u"].dtype == np.float64


def path_rows(out):
    return [(r["level"], r["round"], r["F_B_trace"], r["GF_B_trace"], r["state_hash"], r["status"]) for r in read(out/"hyper.json")["rounds"]]


def check_records_leave_the_path(world, device):
    """C12-4a item 7: --checkpoint-accepted (one more evaluation at the native entry, the files) against the same run without it."""
    root, base, _ = world
    runs = {}
    for name, extra in (("plain", ()), ("records", ("--checkpoint-accepted", "1"))):
        jds.clear_cell_bounds_cache()
        runs[name] = root/f"run_path_{device}_{name}"
        code = run_pair.main([*base, "--out", str(runs[name]), *COMMON, "--backend", "torch64", "--device", device, *extra])
        assert code == 0, name
    assert path_rows(runs["plain"]) == path_rows(runs["records"]) and path_rows(runs["plain"])
    assert read(runs["plain"]/"status.json")["exported"] == read(runs["records"]/"status.json")["exported"]
    with np.load(runs["plain"]/"disp.npz") as a, np.load(runs["records"]/"disp.npz") as b:
        assert a["d"].tobytes() == b["d"].tobytes()
    profile = read(runs["records"]/"hyper.json")["profile"]
    assert profile["errors"] == [] and profile["n_checkpoints"] == 2 and profile["entry"]["directory"]
    assert (runs["records"]/"checkpoints"/"entry"/"disp.npz").is_file()


def test_torch64_cpu_records_leave_the_path(world):
    check_records_leave_the_path(world, "cpu")


@CUDA
def test_torch64_cuda_records_leave_the_path(world):
    check_records_leave_the_path(world, "cuda")


def probe_combination(tmp_path, monkeypatch, device):
    warm = load("warm_run_warm_torch64", "experiments/gate_a/run_warm.py")
    dry_mod = load("warm_m2v1_dry_inputs_torch64", "experiments/m2v1/dry_inputs.py")
    check_probe_combination(warm, dry_mod, tmp_path, monkeypatch, backend="torch64", device=device)


def test_torch64_cpu_the_probes_combination(tmp_path, monkeypatch):
    probe_combination(tmp_path, monkeypatch, "cpu")


@CUDA
def test_torch64_cuda_the_probes_combination(tmp_path, monkeypatch):
    """Review WARM_tools I2: the probe is that pair's only run, so its combination runs here first (about 1-2 minutes of GPU, estimated)."""
    probe_combination(tmp_path, monkeypatch, "cuda")
