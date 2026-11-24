"""Self-test of the C08-9 L1 gate tool experiments/sg/l1_gate.py on synthetic problems only (CPU torch device):
the gate's checks pass on synthetic base points; base points are rebuilt from a (synthetic) run_pair directory with
the run's own F_B; output directories are never reused.  FB1: premise (i) and the |A_hat - A| statistics cover every
residual block (channels, diffusion, log-volume incl. the C05-2a corners, curl, the proximal blocks of E_v); a
device error in the channel or the corner residuals fails the premise gate."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bayesreg import model as M
from bayesreg.cli import run_pair

from ..io_seg._helpers import DEV
from ..solve.test_run_pair import read, setup, write_synthetic_pair, write_tissue  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[3]


def load_tool():
    spec = importlib.util.spec_from_file_location("l1_gate", ROOT/"experiments"/"sg"/"l1_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_on_synthetic_bases(tmp_path, record_property):
    """Two synthetic base points (PG after 2 and 6 steps), both directions, 3 near + 1 far + 1 inside candidates:
    no out-of-band decision mismatch, the comparator equals the CPU decision, max |A_hat - A|/s_hat <= 1, premise
    (i) <= 4e-12, GF_B relative difference <= 1e-3, near out-of-band candidates in every class; the verdict is FAIL
    only for the base count (2 < 3) and names LM as the missing class."""
    tool = load_tool()
    state = tool.synthetic_state(13, tile_size=512)
    named = tool.selftest_bases(state, steps=(2, 6))
    assert [n for n, _, _ in named] == ["synthetic_k2", "synthetic_k6"]
    out = tool.prepare_out(tmp_path/"gate")
    result = tool.run_gate(named, out, device="cpu", near=3, far=1, inside=1, n_tiles=4, check_prox_shortcut=True)
    gates = result["gates"]
    record_property("gates", json.dumps({k: v for k, v in gates.items() if k != "near_per_class"}))
    for key in ("out_of_band_mismatches", "comparator_mismatches", "max_err_over_s_hat", "premise", "gf_relative",
                "blocks_cover_total"):
        assert gates[key]["passed"], key
    assert result["premise_blocks"]["gauss_on_sampled_tiles"] == ["curl", "diffusion", "logvol"]
    for base in result["bases"]:
        assert base["premise_max"] == max(base["premise_tiles_max"], base["premise_channel"])
        assert set(base["premise_gauss_block_max"]) == {"diffusion", "logvol", "curl"}
    assert not gates["bases"]["passed"] and result["verdict"].startswith("FAIL") and "lm" in result["verdict"]
    assert all(v["near"] >= 1 for v in result["per_class"].values())
    rows = [json.loads(line) for line in (out/"candidates.jsonl").read_text().splitlines()]
    assert {r["cls"] for r in rows} == {"pg", "outer_F", "outer_E"}
    assert {r["band"] for r in rows} >= {"near", "inside"}
    assert all(r["comparator_path"] == "cpu" for r in rows if r["band"] == "inside")
    blocks = {"channel", "diffusion", "logvol", "curl"}
    for r in rows:
        if r["cls"] is None:
            continue
        expected = blocks | ({"prox_lambda", "prox_mu", "prox_nu2"} if r["cls"] == "outer_E" else set())
        assert set(r["block_err_over_s_hat"]) == expected and r["blocks_cover_total"]
        assert r["err_over_s_hat"] <= r["block_err_sum_over_s_hat"]*(1+1e-12)+1e-300
    with pytest.raises(FileExistsError):
        tool.prepare_out(out)


def test_gate_with_the_corner_term(tmp_path):
    """C05-2a (logvol_points = gauss27+corners): the premise compares the corner residuals too (logvol_corners on the
    sampled tiles) and stays <= 4e-12; |A_hat - A| (log-volume block incl. the corners) <= s_hat; every row's totals
    are the sums of its blocks."""
    tool = load_tool()
    state = tool.synthetic_state(11, tile_size=256, logvol_points="gauss27+corners")
    assert state.logvol_corners
    named = tool.selftest_bases(state, steps=(3,))
    out = tool.prepare_out(tmp_path/"gate_corners")
    result = tool.run_gate(named, out, device="cpu", directions=("+",), near=2, far=0, inside=1, n_tiles=3)
    gates = result["gates"]
    for key in ("max_err_over_s_hat", "premise", "blocks_cover_total", "out_of_band_mismatches",
                "comparator_mismatches"):
        assert gates[key]["passed"], (key, gates[key])
    assert result["premise_blocks"]["gauss_on_sampled_tiles"] == ["curl", "diffusion", "logvol", "logvol_corners"]
    base = result["bases"][0]
    assert set(base["premise_gauss_block_max"]) == {"diffusion", "logvol", "curl", "logvol_corners"}
    assert base["candidates"] >= 1


def test_gate_with_the_closed27_part(tmp_path):
    """C05-2b (logvol_points = gauss27+closed27, IMPL27): the premise compares the closed27 residuals too
    (logvol_closed27 on the sampled tiles) and stays <= 4e-12; |A_hat - A| (log-volume block incl. the closed27 part)
    <= s_hat; every row's totals are the sums of its blocks; an error of 1e-9 in the closed27 residuals alone fails
    the premise."""
    from dataclasses import replace

    from bayesreg.backends.torch64 import stable as S
    tool = load_tool()
    state = tool.synthetic_state(11, tile_size=256, logvol_points="gauss27+closed27")
    assert state.logvol_closed27
    named = tool.selftest_bases(state, steps=(3,))
    out = tool.prepare_out(tmp_path/"gate_closed27")
    result = tool.run_gate(named, out, device="cpu", directions=("+",), near=2, far=0, inside=1, n_tiles=3)
    gates = result["gates"]
    for key in ("max_err_over_s_hat", "premise", "blocks_cover_total", "out_of_band_mismatches",
                "comparator_mismatches"):
        assert gates[key]["passed"], (key, gates[key])
    assert result["premise_blocks"]["gauss_on_sampled_tiles"] == ["curl", "diffusion", "logvol", "logvol_closed27"]
    backend = tool.make_backend("torch64", device="cpu")
    prepared = backend.prepare(state)
    base = backend.evaluate(prepared, named[0][2])
    base = replace(base, cpu=M.evaluate(prepared, base.u))

    def closed27_error(dstate, tev, lo, hi):
        out = S.device_residuals(dstate, tev, lo, hi)
        out["logvol_closed27"] = out["logvol_closed27"]*(1+1e-9)+1e-9
        return out
    noisy = tool.premise(backend, prepared, [base], 3, device_residuals=closed27_error)
    assert noisy["gauss_block_max"]["logvol_closed27"] > tool.GATES["max_premise"] > noisy["channel_block"]


def test_premise_gates_channel_and_corner_residuals(tmp_path):
    """A device error of 1e-9 (relative) in the channel residuals or in the corner residuals alone makes the premise
    exceed 4e-12 and the premise gate fail; the exact device residuals pass."""
    from dataclasses import replace

    from bayesreg.backends.torch64 import stable as S
    from bayesreg.solve import make_backend
    tool = load_tool()
    state = tool.synthetic_state(9, tile_size=128, logvol_points="gauss27+corners")
    backend = make_backend("torch64", device="cpu")
    state = backend.prepare(state)
    u = np.zeros(state.grid.shape+(3,))
    u[2:-2, 2:-2, 2:-2, 0] = .05
    base = backend.evaluate(state, u)
    base = replace(base, cpu=M.evaluate(state, base.u))
    exact = tool.premise(backend, state, [base], 4)
    assert exact["max"] <= tool.GATES["max_premise"] and exact["gauss_blocks"][-1] == "logvol_corners"
    assert exact["proximal"]["value"] == 0. and exact["proximal"]["blocks"] == ["prox_lambda", "prox_mu", "prox_nu2"]

    def corner_error(dstate, tev, lo, hi):
        out = S.device_residuals(dstate, tev, lo, hi)
        out["logvol_corners"] = out["logvol_corners"]*(1+1e-9)+1e-9
        return out
    corners = tool.premise(backend, state, [base], 4, device_residuals=corner_error)
    assert corners["gauss_block_max"]["logvol_corners"] > tool.GATES["max_premise"] > corners["channel_block"]
    assert corners["max"] > tool.GATES["max_premise"]
    noisy = replace(base, device=replace(base.device, r_host=base.device.r_host*(1+1e-9)))
    channel = tool.premise(backend, state, [noisy], 4)
    assert channel["channel_block"] > tool.GATES["max_premise"] >= channel["tiles_max"]
    assert channel["max"] == channel["channel_block"]
    (tmp_path/"candidates.jsonl").write_text("")
    summary = tool.summarize([dict(premise=channel, gf_rel=0.)], tmp_path/"candidates.jsonl")
    assert not summary["gates"]["premise"]["passed"] and summary["gates"]["premise"]["value"] == channel["max"]


def test_bases_from_a_run_pair_directory(setup, tmp_path):
    """`bases` rebuilds the native ModelState of a run_pair run (same F_B at its final_native u as the run's last
    native F_B) and saves the state after k = 0 and 1 further PG steps."""
    tool = load_tool()
    root, args = setup
    grid, pair = write_synthetic_pair(root, DEV[0], DEV[1])
    write_tissue(root, grid, DEV[0])
    write_tissue(root, grid, DEV[1], shift=(.7, -.5, .4))
    run = root/"run_for_bases"
    assert run_pair.main(args(pair, run, "--rounds", "1,1,1,1", "--max-accepted", "2")) == 0
    state, u, name = tool.native_state_of_run(run)
    assert name == pair
    native = [r for r in read(run/"hyper.json")["rounds"] if r["level"] == 3][-1]
    assert M.evaluate(state, u).energy == native["F_B_end"]
    out = tmp_path/"bases"
    assert tool.main(["bases", "--run-dir", str(run), "--steps", "0,1", "--out", str(out)]) == 0
    files = sorted(out.glob("base_*.npz"))
    assert [f.name.split("_k")[-1] for f in files] == ["0.npz", "1.npz"]
    with np.load(files[0]) as z:
        assert np.array_equal(z["u"], u) and str(z["run_dir"]) == str(run.resolve())
    with np.load(files[1]) as z:
        assert not np.array_equal(z["u"], u)
    assert json.loads((out/"bases.json").read_text())["missing_steps"] == []
