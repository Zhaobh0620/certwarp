"""experiments/audit/recertify.py (D53's audit; contract C05-7 item 7b; ETA4 review round 3 R3-2; RECERT review) on
synthetic exported states of a non-cubic grid (7 x 6 x 5 nodes, spacing (1, 1.25, .75), origin (-3, 2.5, 10); 120
cells; nothing real, no label). One interior node is moved along +x by `shift` mm (hx = 1, so the smallest corner J of
the compressed cells is 1 - shift):

- .9 mm (J min .1): passed, a positive bound not above .1, no uncertain or infeasible cell; the record binds the state
  (state_hash = sha256 of u = stored hash = status.json "exported" = certify_map's digest), the run (run_dir, the
  exported file's sha256, the rebuilt grid_id) and the configuration (CompiledFallbackCertificateConfig, delta 0,
  max_depth 48, max_boxes 255, tile 8192, the requested fallback_workers; the prescreen configuration's parent);
  n_prescreen_cells 0; the eta sweep's check accepts it with the run's binding (RECERT review I1, S1-S3);
- 1.0 mm (J = 0 on an edge): not passed, uncertain cells (subdivision budget), listed (RECERT review I2, S9);
- 1.3 mm (folded): not passed, infeasible cells, listed;
- the reviewer's counterexample (I1): the folded u in a copy whose grid record says hx = 2 passes there, and the sweep
  refuses it for the original run (another file, another run directory, another grid_id);
- a run stopped before the native level (export.level 2): final_level2.npz is recertified and its level recorded (S8);
- refusals (S5, S6): an existing output, an output inside the run directory or in a missing directory (before any
  certificate), a grid record without grid_id, an unreadable status.json, a corrupt npz, no exported state -> exit 8 with
  one message; a document that cannot be serialised leaves no file behind.
The certificate compiles numba kernels on first use (single thread here: NUMBA_NUM_THREADS from the environment).
"""
from datetime import datetime
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from jd_curl.three_d.grid import make_grid_3d

from bayesreg.solve.export import write_final_state

ROOT = Path(__file__).resolve().parents[3]
NODE = (2, 2, 3)                                         # (z, y, x) of the moved interior node


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recertify():
    return load("audit_recertify", "experiments/audit/recertify.py")


@pytest.fixture(scope="module")
def sweep():
    return load("audit_sweep", "experiments/eta/sweep.py")


def grid_of(hx=1.):
    return make_grid_3d(nx=7, ny=6, nz=5, hx=hx, hy=1.25, hz=.75, x0=-3., y0=2.5, z0=10., unit="mm")


def make_run(directory, shift, *, exported=None, hx=1., level=None):
    """A run directory with the exported file (final_native.npz, or final_level{level}.npz with export.level) and
    status.json."""
    grid = grid_of(hx)
    u = np.zeros(grid.shape+(3,))
    u[NODE+(0,)] = shift
    directory.mkdir(parents=True)
    name = "final_native.npz" if level is None else f"final_level{level}.npz"
    digest = write_final_state(directory/name, u, grid, status="budget_exhausted")
    export = dict(state_hash=digest) if level is None else dict(state_hash=digest, level=level)
    (directory/"status.json").write_text(json.dumps(dict(status="budget_exhausted", exported=exported or digest,
                                                         export=export)))
    return directory, digest, grid


def binding(recertify, run, grid):
    return dict(run_dir=run, export_sha256=recertify.file_sha256(run/"final_native.npz"), eval_grid_id=grid.grid_id)


def test_legal_state_passes_and_the_record_binds_state_run_and_configuration(recertify, sweep, tmp_path):
    run, digest, grid = make_run(tmp_path/"run", .9)
    out = tmp_path/"audit"/"recert.json"
    out.parent.mkdir()
    assert recertify.main([str(run), "--out", str(out), "--workers", "2"]) == 0
    assert not list(out.parent.glob(".*.tmp"))                                   # the temporary file is gone
    doc = json.loads(out.read_text())
    assert (doc["tool"], doc["schema"]) == ("experiments/audit/recertify.py", "bayesreg.audit.recertify.v1")
    assert doc["script_sha256"] == recertify.file_sha256(ROOT/"experiments"/"audit"/"recertify.py")
    assert isinstance(doc["code_sha"], str) and len(doc["code_sha"]) == 40 and isinstance(doc["code_dirty"], bool)
    assert doc["state_hash"] == digest == doc["status_exported"] == doc["stored_state_hash"] == \
        doc["certificate_state_hash"]
    assert doc["state_hash_matches_stored"] is True and doc["state_hash_matches_exported"] is True
    assert doc["run_dir"] == str(run.resolve())
    assert doc["export_file"]["file_sha256"] == recertify.file_sha256(run/"final_native.npz")
    assert doc["export_file"]["level"] == "native" and doc["grid"]["rebuilt_grid_id"] == grid.grid_id
    cfg = doc["certificate_config"]
    assert (cfg["class"], cfg["delta"], cfg["max_depth"], cfg["max_boxes"], cfg["tile_size"], cfg["fallback_workers"]) \
        == ("CompiledFallbackCertificateConfig", 0., 48, 255, 8192, 2)
    assert "prescreen_kernel" not in cfg and set(cfg) >= {"policy", "prepass", "kernel", "fallback_kernel"}
    assert doc["threads"]["fallback_workers"] == 2
    assert doc["certificate_version"] == "q1-3d-parallel-hoisted-edge-hull-compiled-bernstein-v1"
    assert doc["passed"] is True and doc["status"] == "certified_feasible" and doc["cells"] is None
    assert (doc["n_uncertain"], doc["n_infeasible"], doc["n_prescreen_cells"], doc["n_cells"]) == (0, 0, 0, 120)
    assert doc["n_fast_cells"]+doc["n_fallback_cells"] == doc["n_cells_checked"] == 120
    # the smallest corner J of the compressed cells is 1 - .9 = .1: the bound is positive and not above it
    assert 0 < doc["min_jd_lower_bound"] <= .1 + 1e-12
    for key in ("started_utc", "ended_utc"):
        assert datetime.fromisoformat(doc[key].replace("Z", "+00:00")).tzinfo is not None
    # the eta sweep accepts the record for this run (a dirty test tree needs the audit's note)
    note = "test tree" if doc["code_dirty"] else None
    assert sweep.recertification_problems(doc, state_hash=digest, dirty_note=note,
                                          **binding(recertify, run, grid)) == []
    assert sweep.recertification_problems(doc, state_hash="0"*64, dirty_note=note)       # another run's state
    assert sweep.recertification_problems(dict(doc, tool="experiments/other.py"), state_hash=digest, dirty_note=note)
    assert sweep.recertification_problems(dict(doc, script_sha256="0"*64), state_hash=digest, dirty_note=note)
    # never overwritten; never inside the run directory; the output directory must exist before any certificate
    with pytest.raises(SystemExit):
        recertify.main([str(run), "--out", str(out)])
    with pytest.raises(SystemExit):
        recertify.main([str(run), "--out", str(run/"recert.json")])
    with pytest.raises(SystemExit):
        recertify.main([str(run), "--out", str(tmp_path/"missing"/"recert.json")])
    assert not (run/"recert.json").exists() and not (tmp_path/"missing").exists()


def test_zero_and_folded_states_do_not_pass_and_list_their_cells(recertify, sweep, tmp_path):
    # RECERT review I2: J = 0 on an edge -> "zero" lands on uncertain (outward rounding finds no J <= 0 witness)
    run, digest, grid = make_run(tmp_path/"zero", 1.)
    doc = recertify.recertify(run)
    assert doc["passed"] is False and doc["n_uncertain"] > 0 and doc["state_hash"] == digest
    assert not (doc["min_jd_lower_bound"] is not None and doc["min_jd_lower_bound"] > 0)
    assert doc["cells"]["uncertain"]["total"] == doc["n_uncertain"] and doc["cells"]["matches_map"] is True
    assert len(doc["cells"]["uncertain"]["cells"]) == doc["n_uncertain"]
    problems = sweep.recertification_problems(doc, state_hash=digest, dirty_note="test")
    assert any("n_uncertain" in p for p in problems) and any("passed is not true" in p for p in problems)
    # folded
    run, digest, grid = make_run(tmp_path/"fold", 1.3)
    doc = recertify.recertify(run)
    assert doc["passed"] is False and doc["status"] == "infeasible" and doc["n_infeasible"] > 0
    assert doc["state_hash"] == digest and doc["cells"]["infeasible"]["total"] == doc["n_infeasible"]
    zyx = {tuple(c["zyx"]) for c in doc["cells"]["infeasible"]["cells"]}
    assert zyx and all(z in (1, 2) and y in (1, 2) and x == 3 for z, y, x in zyx)  # the cells on the node's +x side
    assert any("passed" in p for p in sweep.recertification_problems(doc, state_hash=digest, dirty_note="test"))


def test_a_copy_with_another_grid_is_not_the_run(recertify, sweep, tmp_path):
    """RECERT review I1: the folded u written with a grid record hx = 2 passes in the copy, and the sweep refuses it
    for the original run (the exported file, the run directory and the evaluator's grid_id differ)."""
    run, digest, grid = make_run(tmp_path/"run", 1.3)
    copy, copy_digest, copy_grid = make_run(tmp_path/"copy", 1.3, hx=2.)
    assert copy_digest == digest and copy_grid.grid_id != grid.grid_id               # the same u bytes
    doc = recertify.recertify(copy)
    assert doc["passed"] is True and doc["state_hash"] == digest                      # J min 1 - 1.3/2 = .35 there
    problems = sweep.recertification_problems(doc, state_hash=digest, dirty_note="test",
                                              **binding(recertify, run, grid))
    assert any("run_dir" in p for p in problems) and any("export_file sha256" in p for p in problems)
    assert any("grid_id" in p for p in problems)
    # each binding alone already refuses it
    for key in ("run_dir", "export_sha256", "eval_grid_id"):
        one = {k: (v if k == key else None) for k, v in binding(recertify, run, grid).items()}
        assert sweep.recertification_problems(doc, state_hash=digest, dirty_note="test", **one), key


def test_a_run_stopped_before_the_native_level(recertify, tmp_path):
    """RECERT review S8: status.json export.level = 2 -> final_level2.npz on its own grid, the level recorded."""
    run, digest, grid = make_run(tmp_path/"run", .5, level=2)
    doc = recertify.recertify(run)
    assert doc["export_file"]["level"] == 2 and doc["export_file"]["path"].endswith("final_level2.npz")
    assert doc["passed"] is True and doc["state_hash"] == digest and doc["grid"]["rebuilt_grid_id"] == grid.grid_id


def test_refusals_are_one_exit_8_message(recertify, tmp_path, capsys):
    out = tmp_path/"x.json"
    # no exported state
    empty = tmp_path/"empty"
    empty.mkdir()
    (empty/"status.json").write_text(json.dumps(dict(status="stopped_before_native", exported=None)))
    assert recertify.main([str(empty), "--out", str(out)]) == 8
    # an unreadable status.json
    bad = tmp_path/"bad"
    bad.mkdir()
    (bad/"status.json").write_text("{not json")
    assert recertify.main([str(bad), "--out", str(out)]) == 8
    # a grid record without grid_id
    run, digest, grid = make_run(tmp_path/"nogrid", .5)
    with np.load(run/"final_native.npz") as data:
        members = {k: data[k] for k in data.files}
    record = json.loads(str(members["grid"]))
    record.pop("grid_id")
    members["grid"] = np.array(json.dumps(record))
    (run/"final_native.npz").unlink()
    np.savez(run/"final_native.npz", **members)
    assert recertify.main([str(run), "--out", str(out)]) == 8
    # a corrupt npz
    run, digest, grid = make_run(tmp_path/"corrupt", .5)
    (run/"final_native.npz").write_bytes(b"not a zip")
    assert recertify.main([str(run), "--out", str(out)]) == 8
    err = capsys.readouterr().err
    assert err.count("recertify:") == 4 and "Traceback" not in err and "grid_id" in err
    assert not out.exists()
    # a document that cannot be serialised leaves nothing behind (RECERT review S6)
    target = tmp_path/"nan.json"
    with pytest.raises(ValueError):
        recertify.write_new(target, dict(value=float("nan")), tmp_path/"run_dir_elsewhere")
    assert not target.exists() and not list(tmp_path.glob(".nan.json.*"))
