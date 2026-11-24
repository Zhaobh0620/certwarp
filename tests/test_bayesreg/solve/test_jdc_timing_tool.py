"""The JDC1 timing tool experiments/jdc/time_certificate_estimate.py on synthetic states only (no data/ access, no GPU):
the synthetic dry run, the final_native.npz path (written here by bayesreg.solve.export.write_final_state, with a small
log.jsonl), and its refusals.  The tool's real run (the GPU smoke G3 state) is scheduled by the lead, not run here."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bayesreg.solve import jd_scaling as jds
from bayesreg.solve.export import write_final_state


TOOL = Path(__file__).resolve().parents[3]/"experiments"/"jdc"/"time_certificate_estimate.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("jdc_time_certificate_estimate", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _empty_cache():
    jds.clear_cell_bounds_cache()
    yield
    jds.clear_cell_bounds_cache()


def test_synthetic_dry_run(tool, tmp_path):
    out = tmp_path/"run"
    record = tool.main(["--synthetic", "8", "--repeats", "2", "--warmup", "0", "--out", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["cmd.txt", "env.txt", "git.txt", "timing.json"]
    assert json.loads((out/"timing.json").read_text()) == json.loads(json.dumps(record))
    assert record["state"]["source"] == "synthetic" and record["certificate"]["outcome"]["passed"]
    timing = record["timing"]
    assert list(timing) == list(tool.VARIANTS)
    for variant, entry in timing.items():
        s = entry["summary"]
        assert s["n"] == 2 and 0 < s["min"] <= s["q1"] <= s["median"] <= s["q3"] <= s["max"] and s["iqr"] >= 0
    assert "cell_estimate" not in timing[jds.GAUSS_CORNERS]["record"]
    assert timing[jds.CLOSED27]["record"]["cell_estimate"] == jds.CLOSED27
    assert timing[jds.CERTIFICATE]["record"]["cell_bounds_cached"] is False
    assert timing["certificate_cached"]["record"]["cell_bounds_cached"] is True
    for variant in (jds.CERTIFICATE, "certificate_cached"):
        assert timing[variant]["record"]["cert_min_matches"] and timing[variant]["cell_bounds_summary"]["n"] == 2
        assert timing[variant]["record"]["n_nodes_r_lowered"] >= 8            # the edge-dip cell's vertices
    # JDTI review minor 4 (schema v2): the certificate_bernstein variants, uncached and cached
    assert record["schema"] == "jdc.time_certificate_estimate.v2" and record["memtrace"] is False
    bern, bern_cached = timing[jds.CERTIFICATE_BERNSTEIN]["record"], timing[tool.BERN_CACHED]["record"]
    assert bern["cell_estimate"] == bern_cached["cell_estimate"] == jds.CERTIFICATE_BERNSTEIN
    assert not bern["cell_bounds_cached"] and not bern["bernstein_cached"]
    assert bern_cached["cell_bounds_cached"] and bern_cached["bernstein_cached"]
    assert bern["n_cells_bernstein"] == bern_cached["n_cells_bernstein"] and bern["n_bernstein_j53_violations"] == 0
    assert bern["n_r_lt_0p01"] >= 8                                           # the edge-dip cell's vertices
    for variant in (jds.CERTIFICATE_BERNSTEIN, tool.BERN_CACHED):
        assert timing[variant]["bernstein_summary"]["n"] == 2 and timing[variant]["cell_bounds_summary"]["n"] == 2
    assert timing[jds.CERTIFICATE]["bernstein_summary"] is None
    assert set(record["median_ratio_to_default"]) == {jds.CLOSED27, jds.CERTIFICATE, "certificate_cached",
                                                      jds.CERTIFICATE_BERNSTEIN, tool.BERN_CACHED}
    assert record["native_refreshes"] is None and record["run_cost_estimate"] is None
    with pytest.raises(SystemExit):                                              # nothing is overwritten
        tool.main(["--synthetic", "8", "--repeats", "1", "--out", str(out)])


def test_final_native_input_and_refresh_count(tool, tmp_path):
    u, grid, _ = tool.synthetic_state(8)
    u = np.array(u)
    run = tmp_path/"G"
    run.mkdir()
    digest = write_final_state(run/"final_native.npz", u, grid, status="line_search_failed")
    rows = [dict(kind="jd_scaling", level=3, at="round_start"), dict(kind="jd_scaling", level=3, at="refresh"),
            dict(kind="jd_scaling", level=3, at="refresh"), dict(kind="jd_scaling", level=2, at="refresh"),
            dict(kind="candidate", level=3)]
    (run/"log.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    record = tool.main(["--state", str(run/"final_native.npz"), "--repeats", "1", "--warmup", "0",
                        "--variants", "gauss27+corners,certificate", "--count-refreshes", "--out", str(tmp_path/"o")])
    assert record["state"]["source"] == "final_native" and record["state"]["state_hash"] == digest
    assert record["state"]["state_hash_matches"] and record["state"]["grid"]["grid_id"] == grid.grid_id
    assert record["certificate"]["outcome"]["state_hash"] == digest
    assert list(record["timing"]) == [jds.GAUSS_CORNERS, jds.CERTIFICATE]
    assert record["native_refreshes"]["n_computations"] == 3
    assert record["native_refreshes"]["by_at"] == {"round_start": 1, "refresh": 2}
    assert record["run_cost_estimate"]["extra_seconds_total"] == pytest.approx(
        3*record["run_cost_estimate"]["extra_seconds_per_computation"])
    # a tampered state hash is refused (the reader's check)
    bad = tmp_path/"bad.npz"
    with np.load(run/"final_native.npz") as data:
        members = {k: data[k] for k in data.files}
    members["state_hash"] = np.array("0"*64)
    np.savez(bad, **members)
    with pytest.raises(Exception, match="state_hash"):
        tool.main(["--state", str(bad), "--repeats", "1", "--out", str(tmp_path/"b")])


def test_bad_arguments(tool, tmp_path):
    with pytest.raises(SystemExit):
        tool.main(["--synthetic", "8", "--variants", "closed64", "--out", str(tmp_path/"x")])
    with pytest.raises(SystemExit):
        tool.main(["--synthetic", "8", "--repeats", "0", "--out", str(tmp_path/"y")])
    assert not (tmp_path/"x").exists() and not (tmp_path/"y").exists()
