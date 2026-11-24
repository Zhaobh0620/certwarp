"""bayesreg.eval.holdout_tet24 (contract C24.12, revision 9d; BZ 2026-11-19 (乙)): the 24-tetrahedron proxy of the hold-out after the unsealing.  Synthetic only: the
convention of fields.d_to_u against tet_ratio_summary and jd_curl's cell geometry on anisotropic fields, the cell computation bit for bit against a direct call, the
plan's refusals (a path outside the manifest, a sha256 that differs, a truth path, a field elsewhere), the stops (a changed field, d or u not step 11's, an open the
spy refuses), the once-only lock and the rerun authorization, the summary without ids, and the code identity's one exception on temporary git repositories.  The
end-to-end tests run the evaluation entry on the synthetic hold-out of _holdout_world.py first (step 11), then this module on its records; no real file is opened."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest

from jd_curl.three_d.bernstein_barrier import to_bernstein
from jd_curl.three_d.cell_q1 import CellQ1Operators3D, cell_tiles
from jd_curl.three_d.geometry import _compute_cell_geometry
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.nifti import file_sha256

import bayesreg.io.sealed as S
from bayesreg.io.guard import SubjectSealed
from bayesreg.eval import holdout as H
from bayesreg.eval import holdout_tet24 as T
from bayesreg.eval.fields import array_sha256, d_to_u
from bayesreg.io import code_identity as CI

from . import _synthetic as syn
from ._holdout_world import World, commit, git, head, no_units

FM = T.folding_module()                                     # scripts/folding_metrics.py of this code tree
SPACING = (.9, 1.1, 1.3)                                    # anisotropic, xyz


def direct(path, spacing):
    """The independent path: np.load the npz, d as float64, u by hand ((nz, ny, nx, 3), components x, y, z in mm), tet_ratio_summary called directly."""
    with np.load(path) as z:
        d = np.asarray(z["d"], dtype=np.float64)
    hx, hy, hz = spacing
    u = np.stack([d[2]*hx, d[1]*hy, d[0]*hz], axis=-1)
    return u, FM.tet_ratio_summary(u, spacing)


def bits(summary):
    """Every number of a tet_ratio_summary result, floats by their bits."""
    return {s: {k: (float(v).hex() if isinstance(v, float) else v) for k, v in r.items()} for s, r in summary.items()}


def folded_d(shape=(6, 7, 8)):
    """A small field (voxels, zyx) with one inverted cell: an interior node pushed past its +x neighbour."""
    d = np.zeros((3,)+shape)
    d[2, 3, 3, 3] = 1.6
    return d


# ---- the convention (C24.12-4) ---------------------------------------------------------------------------------------------------------------------------
def test_identity_has_no_nonpositive_and_a_fold_is_counted():
    shape = (6, 7, 8)
    for d, folded in ((np.zeros((3,)+shape), False), (folded_d(shape), True)):
        u = d_to_u(d, SPACING)
        out = FM.tet_ratio_summary(u, SPACING)
        assert sorted(out) == sorted(f"tet{s}" for s in T.SCHEMES)
        for s, r in out.items():
            assert r["cells"] == 5*6*7 and r["tets"] == r["cells"]*int(s[3:])
            if folded:
                assert r["cells_nonpositive"] > 0 and r["tets_nonpositive"] > 0 and r["min_ratio"] < 0, s
            else:
                assert r["cells_nonpositive"] == 0 and r["tets_nonpositive"] == 0 and abs(r["min_ratio"]-1.) < 1e-14, s


def test_d_to_u_is_the_convention_of_tet_ratio_summary_and_of_the_certificate_geometry():
    """d (voxels, zyx) -> fields.d_to_u -> u: the mean of the 24 ratios of a cell equals jd_curl's cell average of J (the Bernstein mean of the certificate's Q1
    geometry on the same u and grid) on an anisotropic grid; an affine physical map gives det(I + A) on every tetrahedron.  The negative controls (components
    reversed, spacing reversed) do not match: the check has teeth."""
    grid = make_grid_3d(nx=8, ny=7, nz=6, hx=SPACING[0], hy=SPACING[1], hz=SPACING[2], unit="mm")
    rng = np.random.default_rng(3)
    d = np.zeros((3,)+grid.shape)
    d[:, 1:-1, 1:-1, 1:-1] = .2*rng.standard_normal((3,)+tuple(n-2 for n in grid.shape))
    u = d_to_u(d, grid.spacing)
    assert grid.spacing == SPACING and u.shape == grid.shape+(3,)

    def bernstein_mean(field):
        ops = CellQ1Operators3D(grid, location="closed27_uniform")
        J = np.concatenate([_compute_cell_geometry(field, ops, tile=t).J for t in cell_tiles(grid, 64)])
        return to_bernstein(J).mean(axis=1)
    mean24 = FM.cell_mean_ratio24(u, grid.spacing).ravel()
    assert np.allclose(mean24, bernstein_mean(u), rtol=1e-12, atol=1e-13)
    assert not np.allclose(FM.cell_mean_ratio24(u[..., ::-1].copy(), grid.spacing).ravel(), bernstein_mean(u), rtol=1e-6)
    assert not np.allclose(FM.cell_mean_ratio24(u, grid.spacing[::-1]).ravel(), bernstein_mean(u), rtol=1e-6)
    A = np.array([[.15, .05, -.02], [.03, -.1, .04], [-.06, .02, .08]])             # u(x) = A x in mm, x = (x, y, z)
    X = grid.identity()
    ua = X@A.T
    da = np.stack([ua[..., 2]/SPACING[2], ua[..., 1]/SPACING[1], ua[..., 0]/SPACING[0]])     # the same map as a voxel displacement (zyx)
    for name, r in FM.tet_ratio_summary(d_to_u(da, SPACING), SPACING).items():
        assert np.isclose(r["min_ratio"], np.linalg.det(np.eye(3)+A), rtol=1e-12), name


def write_field(path, d, dtype=np.float32):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, d=d.astype(dtype))
    return path


def cell_for(path, d, spacing, **changes):
    u = d_to_u(np.asarray(np.load(path)["d"], dtype=np.float64), spacing)
    base = dict(arm="a", family="main", kind="disp", index=3, pair="P", outcome="output", path=str(path), file_sha256=file_sha256(path),
                d_sha256=array_sha256(np.asarray(np.load(path)["d"], dtype=np.float64)), state_hash=array_sha256(u), bl1_passed=False)
    base.update(changes)
    return T.Cell24(**base)


def test_compute_cell_bit_for_bit_equal_to_the_direct_call(tmp_path):
    shape = syn.SHAPE
    for k, (d, dtype) in enumerate(((syn.smooth_d(shape), np.float32), (folded_d(shape)+.3*syn.smooth_d(shape), np.float64))):
        path = write_field(tmp_path/f"f{k}"/"disp.npz", d, dtype)
        cell = cell_for(path, d, SPACING)
        listed = {str(path): file_sha256(path)}
        document = T.compute_cell(cell, dict(shape_zyx=list(shape), spacing_xyz=list(SPACING)), listed, FM)
        u, theirs = direct(path, SPACING)
        assert bits(document["tet_ratio_summary"]) == bits(theirs)
        assert array_sha256(u) == cell.state_hash
        assert document["position"] == 3 and document["bl1_passed"] is False and all(document["checks"].values())
        assert all(isinstance(v, (int, float, bool)) for v in leaves(document))


def leaves(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from leaves(v)
    elif isinstance(value, list):
        for v in value:
            yield from leaves(v)
    else:
        yield value


def test_compute_cell_stops_on_a_changed_field_or_a_d_or_u_that_is_not_step_11s(tmp_path):
    shape, meta = syn.SHAPE, dict(shape_zyx=list(syn.SHAPE), spacing_xyz=list(SPACING))
    path = write_field(tmp_path/"f"/"disp.npz", syn.smooth_d(shape))
    good = cell_for(path, None, SPACING)
    with pytest.raises(H.ManifestChanged):
        T.compute_cell(good, meta, {str(path): "0"*64}, FM)                       # the manifest's sha256 differs: nothing loaded
    with pytest.raises(H.ManifestChanged):
        T.compute_cell(good, meta, {}, FM)                                       # not a line of the manifest
    listed = {str(path): file_sha256(path)}
    with pytest.raises(T.FieldMismatch):
        T.compute_cell(cell_for(path, None, SPACING, d_sha256="1"*64), meta, listed, FM)
    with pytest.raises(T.FieldMismatch):
        T.compute_cell(cell_for(path, None, SPACING, state_hash="2"*64), meta, listed, FM)
    with pytest.raises(T.FieldMismatch):                                         # another spacing: u is not the certificate's
        T.compute_cell(good, dict(meta, spacing_xyz=[1., 1., 1.]), listed, FM)


def test_redact_and_the_schemes_pin():
    assert T.redact("x OASIS_OAS1_0123_MR1__OASIS_OAS1_0456_MR2/disp.npz y") == "x <id>__<id>/disp.npz y"
    assert T.SCHEMES == (24, 6, 5)


# ---- the plan (C24.12-3): only the manifest's field lines at step 11's sha256 ---------------------------------------------------------------------------
def plan_world(tmp_path):
    """A minimal table, rows and step-11 records (the plan reads nothing but them and the manifest's lines)."""
    data = tmp_path/"data"
    method = dict(name="method", family="method", kind="disp", field=dict(dir=str(tmp_path/"d"/"method"/"{pair}"/"output")))
    ext = dict(name="ext", family="main", kind="disp", field=dict(file_key="disp.file", sha256_key="disp.sha256", root=str(tmp_path/"d"/"{arm}")))
    table = dict(arms=[method, ext])
    rows = ((0, "F0", "M0", "F0__M0"), (1, "F1", "M1", "F1__M1"))
    listed, records = {}, {}
    for row in rows:
        arms = {}
        for arm, path in ((method, tmp_path/"d"/"method"/row[3]/"output"/"disp.npz"), (ext, tmp_path/"d"/"ext"/row[3]/"disp.npz")):
            sha = hashlib.sha256(str(path).encode()).hexdigest()
            listed[str(path)] = sha
            arms[arm["name"]] = dict(outcome="output", field=dict(kind="disp", source="disp.npz", disp=dict(path=str(path), file_sha256=sha, d_sha256="d"*64)),
                                     certificate=dict(mode="jacobian_only", configuration="BL1", state_hash="5"*64, passed=row[0] == 0))
        records[row[3]] = dict(arms=arms)
    return table, rows, records, listed, data


def test_the_plan_takes_the_step_11_fields_listed_at_their_sha256(tmp_path, monkeypatch):
    table, rows, records, listed, data = plan_world(tmp_path)
    monkeypatch.setattr(H, "DATA_ROOT", data)
    records["F1__M1"]["arms"]["ext"] = dict(outcome="no_output")
    cells = T.plan_cells(table, rows, records, listed)
    assert [(c.arm, c.index, c.outcome) for c in cells] == [("method", 0, "output"), ("ext", 0, "output"), ("method", 1, "output"), ("ext", 1, "no_output")]
    assert [c.bl1_passed for c in cells if c.outcome == "output"] == [True, True, False]
    del records["F0__M0"]
    assert [c.outcome for c in T.plan_cells(table, rows, records, listed)][:2] == ["not_evaluated", "not_evaluated"]


@pytest.mark.parametrize("how", ["not_listed", "sha256_differs", "truth_path", "elsewhere", "not_bl1", "no_state_hash", "kind", "same_file_twice"])
def test_the_plan_refuses(tmp_path, monkeypatch, how):
    table, rows, records, listed, data = plan_world(tmp_path)
    monkeypatch.setattr(H, "DATA_ROOT", data)
    entry = records["F0__M0"]["arms"]["ext"]
    disp = entry["field"]["disp"]
    if how == "not_listed":
        del listed[disp["path"]]
    elif how == "sha256_differs":
        listed[disp["path"]] = "0"*64
    elif how == "truth_path":                                 # a label file under the data root given as a field: refused before anything is opened
        truth = str(data/"F0"/H.LABEL_FILES["seg35"])
        disp["path"], listed[truth] = truth, disp["file_sha256"]
    elif how == "elsewhere":                                  # a disp.npz outside the arm's field root
        other = str(tmp_path/"other"/"disp.npz")
        disp["path"], listed[other] = other, disp["file_sha256"]
    elif how == "not_bl1":
        entry["certificate"]["configuration"] = "C07-2"
    elif how == "no_state_hash":
        entry["certificate"]["state_hash"] = None
    elif how == "kind":
        entry["field"]["source"] = "final_native.npz:u (mm) -> d"
    else:
        records["F1__M1"]["arms"]["ext"]["field"]["disp"] = dict(disp)
    with pytest.raises(H.HoldoutRefused) as info:
        T.plan_cells(table, rows, records, listed)
    assert info.value.stage == "cells"


# ---- end to end on the synthetic hold-out (step 11 first) -------------------------------------------------------------------------------------------------
@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "_LAST_SPY", None)
    w = World(tmp_path)
    w.patch(monkeypatch)
    admission, context = w.open()
    assert H.evaluate_holdout(admission, context) == 0                       # step 11 on the synthetic hold-out
    w.eval_out = context["out"]
    w.truth_reads = len(w.reads)
    monkeypatch.setattr(T, "T24_ROOT", w.results/"holdout_B_v1"/"tet24_v1")
    monkeypatch.setattr(T, "identity", stub_identity)
    monkeypatch.setattr(T, "origins", lambda repo=None: dict(stub="origins of the world's code tree: tested apart"))
    monkeypatch.setattr(T, "folding_module", lambda repo=None: FM)
    yield w
    H.stop_spy()
    S._reset_cache()


def stub_identity(repo=None, *, reviewed_commit=None, run=None):
    """The world's master lock names P1 = None (its rule one is stubbed); the identity's exception is tested apart on temporary git repositories."""
    return dict(stub=True, pinned=None, head=None, module=dict(path=T.MODULE_REL, commit=None, blob="b"*40, reviewed_commit=reviewed_commit, reviewed_blob=None)), []


def admit(world, **kwargs):
    S._reset_cache()                                                         # a fresh process: the guard's cache empty (C23.6-5)
    return T.admit(repo=world.repo, run=no_units, **kwargs)


def outputs(world):
    return [(arm["name"], row) for row in world.rows for arm in world.table["arms"]]


def field_path(world, name, row):
    return (world.cell(name, row)/"output"/"disp.npz") if name == "method" else (world.tmp/"big"/name/row[3]/"disp.npz")


def test_end_to_end_every_cell_once_bit_for_bit_no_truth(world):
    context = admit(world)
    assert world.reads[world.truth_reads:] == [] and not T.root().exists()
    assert T.compute(context, echo=lambda *_: None) == 0
    assert len(world.reads) == world.truth_reads                               # no truth through the reader
    for name, row in outputs(world):
        document = json.loads(T.cell_path(T.root(), name, row[0]).read_text())
        _, theirs = direct(field_path(world, name, row), (1., 1., 1.))
        assert bits(document["tet_ratio_summary"]) == bits(theirs), (name, row[0])
        assert all(isinstance(v, (int, float, bool)) for v in leaves(document)) and all(document["checks"].values())
    run = json.loads((T.root()/"run_0.json").read_text())
    counts = run["spy_counts"]
    assert counts["truth"] == 0 and counts["outside"] == 0 and counts["truth_kind_events"] == 0 and counts["refused"] == 0
    assert counts["distinct_fields"] == len(outputs(world)) and counts["stage_manifest"] >= len(world.rows) and counts["eval_lock"] >= 1
    assert run["done"] == [[n, r[0]] for n, r in outputs(world)] and run["failed"] == [] and run["stopped"] is None
    data = str(world.data.resolve())
    assert not [e for e in run["spy"]["events"] if e["path"].startswith(data)]
    # the summary and its CSV: no subject id, no pair name, no position
    for name in ("summary_0.json", "summary_0.csv"):
        text = (T.root()/name).read_text()
        assert not re.search(r"OASIS_OAS1_[0-9]{4}", text) and all(r[3] not in text for r in world.rows) and "position" not in text
    summary = json.loads((T.root()/"summary_0.json").read_text())
    ants = next(a for a in summary["arms"] if a["arm"] == "ants")
    assert ants["computed"] == ants["output"] == len(world.rows) and ants["missing"] == 0
    assert ants["side_by_side"]["bl1_not_passed_pairs"] == sum(
        json.loads((world.eval_out/"pairs"/f"{r[3]}.json").read_text())["arms"]["ants"]["certificate"]["passed"] is not True for r in world.rows)
    rows = (T.root()/"summary_0.csv").read_text().splitlines()
    assert rows[0].split(",") == list(T.CSV_COLUMNS) and len(rows) == 1+len(world.table["arms"])*len(T.SCHEMES)


def test_the_lock_refuses_a_second_run(world):
    assert T.compute(admit(world), echo=lambda *_: None) == 0
    context = admit(world)
    with pytest.raises(H.HoldoutRefused) as info:
        T.compute(context, echo=lambda *_: None)
    assert info.value.stage == "once" and H._SPY is None
    assert sorted(p.name for p in T.root().iterdir()) == sorted(["cells", "run_0.json", "run_0.public.json", "run_0.started.json", "summary_0.csv", "summary_0.json",
                                                                 T.LOCK_NAME])


def test_a_manifest_other_than_the_master_locks_is_refused(world):
    world.write_manifest(extra_input=world.repo/H.ARM_TABLE_REL)              # a new committed manifest: not the one the master lock names
    with pytest.raises(H.HoldoutRefused) as info:
        admit(world)
    assert info.value.stage in ("eval", "manifest") and H._SPY is None and not T.root().exists()


def test_a_field_changed_after_the_manifest_stops_the_run(world):
    context = admit(world)
    path = field_path(world, "ants", world.rows[0])
    np.savez(path, d=np.zeros((3,)+syn.SHAPE, dtype=np.float32))
    assert T.compute(context, echo=lambda *_: None) == 4
    stopped = json.loads((T.root()/"stopped_0.json").read_text())
    assert stopped["at"] == ["ants", 0] and stopped["kind"] == "ManifestChanged" and "ManifestChanged" in stopped["reason"]
    assert not T.cell_path(T.root(), "ants", 0).exists() and T.cell_path(T.root(), "method", 0).exists()


def test_the_spy_refuses_a_truth_read_and_the_run_stops(world, monkeypatch):
    context = admit(world)
    subject = world.rows[0][1]

    class Reader:
        @staticmethod
        def tet_ratio_summary(u, spacing):
            (world.data/subject/H.LABEL_FILES["seg35"]).read_bytes()            # a truth file: refused at the open
            return FM.tet_ratio_summary(u, spacing)
    monkeypatch.setattr(T, "folding_module", lambda repo=None: Reader)
    assert T.compute(context, echo=lambda *_: None) == 4
    run = json.loads((T.root()/"run_0.json").read_text())
    assert "SpyRefused" in run["stopped"] and run["spy"]["refused"] and run["spy_counts"]["truth"] == 1 and run["done"] == []
    assert len(world.reads) == world.truth_reads


def test_a_failed_cell_is_computed_again_only_under_a_committed_authorization(world, monkeypatch):
    original = T.compute_cell

    def flaky(cell, metadata, listed, fm):
        if (cell.arm, cell.index) in (("fga", 1), ("ants", 0)):
            raise OSError("synthetic I/O error")
        return original(cell, metadata, listed, fm)
    monkeypatch.setattr(T, "compute_cell", flaky)
    assert T.compute(admit(world), echo=lambda *_: None) == 1
    assert T.cell_path(T.root(), "fga", 1, ".failed_0").is_file() and not T.cell_path(T.root(), "fga", 1).exists()
    monkeypatch.setattr(T, "compute_cell", original)
    folder = world.repo/T.RERUN_DIR_REL

    def authorize(k, cells, **changes):
        document = dict(schema=T.SCHEMA_RERUN, written_by="lead on BZ's behalf", words="（合成测试）BZ：补算", time_local="2026-11-19 1:00 PM CST", utc=H.utc(),
                        reason="synthetic I/O error", reason_class="io_error", lock_sha256=file_sha256(T.root()/T.LOCK_NAME), cells=cells)  # k: str or int
        document.update(changes)
        path = folder/f"holdout_B_v1_tet24_rerun_{k}.json"
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        commit(world.repo, f"rerun {k}")
        return path

    def refused(path):
        with pytest.raises(H.HoldoutRefused) as info:
            T.compute(admit(world), rerun_path=path, echo=lambda *_: None)
        assert info.value.stage == "rerun", info.value
        assert H._SPY is None
    refused(authorize("0", [["fga", 1]]))                                       # review important 4: no number 0 or 00 (it would lock the cell for good)
    refused(authorize("00", [["fga", 1]]))
    refused(authorize(1, [["fga", 0]]))                                         # complete already
    refused(authorize(2, [["nope", 1]]))                                        # not a cell of the plan
    refused(authorize(3, [["ants", 0]], reason_class="method_failure"))         # not an infrastructure reason
    refused(authorize(4, [["fga", 1]], lock_sha256="0"*64))                     # not the first run's lock
    path = authorize(5, [["fga", 1]])
    refused(path)                                                               # named by authorization 4 (refused, but committed): at most once
    assert not (T.root()/"rerun_5.lock").exists() and not T.cell_path(T.root(), "fga", 1).exists()


def test_a_rerun_authorization_computes_its_cells_once(world, monkeypatch):
    original = T.compute_cell

    def flaky(cell, metadata, listed, fm):
        if (cell.arm, cell.index) == ("fga", 1):
            raise OSError("synthetic I/O error")
        return original(cell, metadata, listed, fm)
    monkeypatch.setattr(T, "compute_cell", flaky)
    assert T.compute(admit(world), echo=lambda *_: None) == 1
    monkeypatch.setattr(T, "compute_cell", original)
    document = dict(schema=T.SCHEMA_RERUN, written_by="BZ", words="（合成测试）补算", time_local="2026-11-19 1:00 PM CST", utc=H.utc(), reason="synthetic I/O error",
                    reason_class="io_error", lock_sha256=file_sha256(T.root()/T.LOCK_NAME), cells=[["fga", 1]])
    path = world.repo/T.RERUN_DIR_REL/"holdout_B_v1_tet24_rerun_1.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    commit(world.repo, "rerun 1")
    assert T.compute(admit(world), rerun_path=path, echo=lambda *_: None) == 0
    assert T.cell_path(T.root(), "fga", 1).is_file() and json.loads((T.root()/"rerun_1.lock").read_text())["cells"] == [["fga", 1]]
    run = json.loads((T.root()/"run_1.json").read_text())
    assert run["done"] == [["fga", 1]] and run["spy_counts"]["truth"] == 0 and run["spy_counts"]["outside"] == 0
    summary = json.loads((T.root()/"summary_1.json").read_text())
    assert all(a["missing"] == 0 and a["computed"] == a["output"] for a in summary["arms"])
    with pytest.raises(H.HoldoutRefused):                                      # the same authorization again: its lock is there
        T.compute(admit(world), rerun_path=path, echo=lambda *_: None)
    _, theirs = direct(field_path(world, "fga", world.rows[1]), (1., 1., 1.))
    assert bits(json.loads(T.cell_path(T.root(), "fga", 1).read_text())["tet_ratio_summary"]) == bits(theirs)


def test_the_dry_run_loads_no_field_and_writes_no_id(world, tmp_path, monkeypatch):
    monkeypatch.setattr(H, "REPO", world.repo)
    S._reset_cache()
    lines = []
    assert T.dry_run(tmp_path/"dry", run=no_units, echo=lines.append) == 0
    report = json.loads((tmp_path/"dry"/"dry_run.json").read_text())
    assert report["first_refusal"] is None and report["spy_counts"]["field"] == 0 and report["spy_counts"]["truth"] == 0 and report["spy_counts"]["outside"] == 0
    assert report["n_output"] == len(outputs(world)) and all(r["exists"] == r["output"] == r["listed_sha_equals_step11"] for r in report["arms"].values())
    assert all(r["sha256_now_equals_listed"] is None for r in report["arms"].values()) and report["field_loaded"] is False
    text = (tmp_path/"dry"/"dry_run.json").read_text()+"\n".join(lines)
    assert not re.search(r"OASIS_OAS1_[0-9]{4}", text) and all(r[3] not in text for r in world.rows) and not T.root().exists()
    # --hash-fields: every planned field's bytes hashed under the spy (nothing decoded); the CLI's dry-run the same
    S._reset_cache()
    assert T.dry_run(tmp_path/"dry2", run=no_units, hash_fields=True, echo=lines.append) == 0
    report = json.loads((tmp_path/"dry2"/"dry_run.json").read_text())
    assert all(r["sha256_now_equals_listed"] == r["output"] for r in report["arms"].values()) and report["spy_counts"]["outside"] == 0
    assert report["spy_counts"]["distinct_fields"] == len(outputs(world)) and report["spy_counts"]["truth"] == 0
    monkeypatch.setattr(H, "machine_gate", lambda **kwargs: dict(stub="machine gate (tested in the entry's tests)", passed=True))
    S._reset_cache()
    assert T.main(["dry-run", "--out-dir", str(tmp_path/"dry3"), "--reviewed-commit", "f"*40]) == 0


def test_the_dry_run_stops_before_the_results_root_when_the_identity_is_refused(world, tmp_path, monkeypatch):
    """Review important 1: a refused identity is recorded, the code-tree part of the admission and the machine gate follow, and nothing under the results root or D:
    is opened (no lock line, no step-11 record, no stage manifest, no field)."""
    monkeypatch.setattr(H, "REPO", world.repo)
    monkeypatch.setattr(T, "identity", lambda repo=None, *, reviewed_commit=None, run=None: (stub_identity()[0], ["no pinned record (synthetic)"]))
    S._reset_cache()
    assert T.dry_run(tmp_path/"dry", run=no_units, echo=lambda *_: None) == 2
    report = json.loads((tmp_path/"dry"/"dry_run.json").read_text())
    assert report["first_refusal"] == "code identity (C24.12-2)" and report["conditions"]["admission (C24.12-3)"].startswith("code-tree part passed")
    assert "arms" not in report and all(report["spy_counts"][k] == 0 for k in ("field", "stage_manifest", "eval_lock", "eval_record", "truth", "outside", "refused"))
    assert H._LAST_SPY is not None and all(e["kind"] == "sealed_list" for e in H._LAST_SPY.events)        # only the guard's reads of the sealed lists
    S._reset_cache()
    with pytest.raises(H.HoldoutRefused) as info:                               # the real run refuses at once
        admit(world)
    assert info.value.stage == "code_identity" and H._SPY is None


def test_the_dry_runs_out_dir_is_checked_like_t24_root(world, tmp_path, monkeypatch):
    monkeypatch.setattr(H, "REPO", world.repo)
    table = json.loads((world.repo/H.ARM_TABLE_REL).read_text())
    for bad in (T.root()/"dry", Path(H.EVAL_DIR)/"dry", world.tmp/"stage"/"dry", world.data/"dry", world.repo/"dry"):
        with pytest.raises(H.HoldoutRefused) as info:
            T.check_dry_out(bad, table)
        assert info.value.stage == "output"
    with pytest.raises(H.HoldoutRefused):
        T.check_dry_out(tmp_path, table)                                         # exists
    assert T.check_dry_out(tmp_path/"fresh", table) == tmp_path/"fresh"
    # diff review minor 2: a refused --out-dir is exit 2 (the table of C24.12-5), nothing written, no path printed
    lines = []
    assert T.dry_run(T.root()/"dry", run=no_units, echo=lines.append) == T.EXIT["refused"] == 2
    assert not T.root().exists() and lines and all(str(world.tmp) not in line for line in lines)
    monkeypatch.setattr(H, "machine_gate", lambda **kwargs: dict(stub="machine gate (tested in the entry's tests)", passed=True))
    assert T.main(["dry-run", "--out-dir", str(tmp_path), "--reviewed-commit", "f"*40]) == 2


def test_a_refused_plan_in_the_dry_run_reports_kinds_and_counts_only(world, tmp_path, monkeypatch, capsys):
    """Diff review minor 3: the plan's refusal names cells and paths; the dry run's report and output keep only its kind and the counts by kind."""
    monkeypatch.setattr(H, "REPO", world.repo)
    for row in world.rows:                                                      # step 11's records now name another sha256 for one arm's fields
        path = world.eval_out/"pairs"/f"{row[3]}.json"
        record = json.loads(path.read_text())
        record["arms"]["ants"]["field"]["disp"]["file_sha256"] = "0"*64
        path.write_text(json.dumps(record))
    S._reset_cache()
    lines = []
    assert T.dry_run(tmp_path/"dry", run=no_units, echo=lines.append) == 2
    report = json.loads((tmp_path/"dry"/"dry_run.json").read_text())
    text = report["conditions"]["admission (C24.12-3)"]
    assert text == "refused at cells: PlanRefused: 2 cells cannot be planned; by kind {'not_listed_at_sha256': 2}"
    everything = (tmp_path/"dry"/"dry_run.json").read_text()+"\n".join(lines)+capsys.readouterr().out
    assert "position" not in everything and str(world.tmp) not in everything and "disp.npz" not in everything and not re.search(r"OASIS_OAS1_[0-9]{4}", everything)
    S._reset_cache()
    with pytest.raises(T.PlanRefused) as info:                                   # the real run refuses with the full message (for the run record, not the output)
        admit(world)
    assert info.value.kinds == {"not_listed_at_sha256": 2} and "position" in str(info.value)
    assert T.scrub(str(info.value)).count("position <n>") == 2 and "disp.npz" not in T.scrub(str(info.value))


# ---- review important 2: a swallowed refusal, an outside open; review minors 5, 8, 9: stops, places, exit codes --------------------------------------
def reader(world, how):
    """A stand-in of folding_metrics whose tet_ratio_summary opens a file it may not (the first cell only)."""
    target = world.cell("method", world.rows[0])/"terminal.json"                 # a record under a watched root, not allowed after (i): an "outside" open

    class Reader:
        calls = 0

        @staticmethod
        def tet_ratio_summary(u, spacing):
            Reader.calls += 1
            if Reader.calls == 1:
                try:
                    target.read_bytes()
                except PermissionError:                                         # SpyRefused is a PermissionError
                    if how == "outside":
                        raise
                    if how == "swallowed_then_error":
                        raise RuntimeError("the cell's code swallowed the refusal and failed otherwise")
            return FM.tet_ratio_summary(u, spacing)
    return Reader


@pytest.mark.parametrize("how", ["outside", "swallowed_then_error", "swallowed_then_success"])
def test_a_refused_open_stops_the_run_even_when_the_cells_code_swallows_it(world, monkeypatch, how):
    context = admit(world)
    monkeypatch.setattr(T, "folding_module", lambda repo=None: reader(world, how))
    assert T.compute(context, echo=lambda *_: None) == 4
    run = json.loads((T.root()/"run_0.json").read_text())
    assert run["stopped"] and run["stop_at"] == ["method", 0] and run["exit_code"] == 4 and run["done"] == [] and run["failed"] == []
    assert run["spy_counts"]["outside"] == 1 and run["spy_counts"]["refused"] == 1 and run["spy_counts"]["truth"] == 0
    assert not list((T.root()/"cells").rglob("*.failed_*")) if (T.root()/"cells").exists() else True       # no failure record: a stop, not a failure
    assert len(run["missing_after_stop"]) == len(outputs(world))                 # the stopped cell and every cell after it
    stopped = json.loads((T.root()/"stopped_0.json").read_text())
    assert stopped["kind"] == ("SpyRefused" if how == "outside" else "SpyRefused_swallowed") and len(stopped["missing"]) == len(outputs(world))
    public = json.loads((T.root()/"run_0.public.json").read_text())
    assert public["exit_code"] == 4 and public["stopped"] and public["missing_after_stop"] == len(outputs(world)) and public["stop_kind"] == stopped["kind"]


def test_after_a_stop_no_rerun_authorization_is_accepted(world):
    context = admit(world)
    np.savez(field_path(world, "ants", world.rows[0]), d=np.zeros((3,)+syn.SHAPE, dtype=np.float32))
    assert T.compute(context, echo=lambda *_: None) == 4
    document = dict(schema=T.SCHEMA_RERUN, written_by="BZ", words="（合成测试）补算", time_local="t", utc=H.utc(), reason="x", reason_class="io_error",
                    lock_sha256=file_sha256(T.root()/T.LOCK_NAME), cells=[["fga", 1]])
    path = world.repo/T.RERUN_DIR_REL/"holdout_B_v1_tet24_rerun_1.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    commit(world.repo, "rerun 1")
    with pytest.raises(H.HoldoutRefused) as info:
        T.compute(admit(world), rerun_path=path, echo=lambda *_: None)
    assert info.value.stage == "rerun" and "BZ decides" in str(info.value)
    summary = json.loads((T.root()/"summary_0.json").read_text())
    assert summary["stopped"] is True and summary["missing_rule"]


def test_the_master_locks_p1_and_the_records_grid_must_agree(world, monkeypatch):
    monkeypatch.setattr(T, "identity", lambda repo=None, *, reviewed_commit=None, run=None: (dict(stub_identity()[0], pinned="a"*40), []))
    with pytest.raises(H.HoldoutRefused) as info:
        admit(world)
    assert info.value.stage == "eval" and "pinned commit" in str(info.value) and H._SPY is None
    monkeypatch.setattr(T, "identity", stub_identity)
    path = world.eval_out/"pairs"/f"{world.rows[1][3]}.json"
    record = json.loads(path.read_text())
    record["grid"]["spacing_xyz"] = [1., 1., 2.]
    path.write_text(json.dumps(record))
    with pytest.raises(H.HoldoutRefused) as info:
        admit(world)
    assert info.value.stage == "pairs" and "position 1" in str(info.value) and not T.root().exists()


def test_only_outcome_field_certificate_and_grid_are_kept_of_the_step_11_records(world):
    context = admit(world)
    assert all(set(r) == {"grid"} for r in context["records"].values())
    full = {"P": dict(grid=dict(x=1), truth=dict(a=1), identity=dict(metrics=1), pq=dict(q=1),
                      arms=dict(m=dict(outcome="output", field=dict(f=1), metrics=dict(seg35=.9), per_label=[1], official=dict(o=1),
                                       certificate=dict(mode="jacobian_only", configuration="BL1", state_hash="5"*64, passed=True, status="certified_feasible",
                                                        cells=dict(brain=dict(n=7)), min_jd_lower_bound=.1))))}
    # diff review minor 1: of the certificate only mode, configuration, state_hash, passed (its cells are counted on the truth brain mask)
    assert T.slim_records(full) == {"P": dict(grid=dict(x=1), arms=dict(m=dict(outcome="output", field=dict(f=1),
                                                                             certificate=dict(mode="jacobian_only", configuration="BL1", state_hash="5"*64, passed=True))))}
    assert T.CERTIFICATE_KEYS == ("mode", "configuration", "state_hash", "passed")
    H.stop_spy()


@pytest.mark.parametrize("where", ["watched", "arm_name", "eval_dir"])
def test_t24_root_must_lie_under_no_watched_root(world, monkeypatch, where):
    place = dict(watched=world.tmp/"stage"/"tet24_v1", arm_name=world.results/"holdout_B_v1"/"ants", eval_dir=Path(H.EVAL_DIR)/"tet24_v1")[where]
    monkeypatch.setattr(T, "T24_ROOT", place)
    with pytest.raises(H.HoldoutRefused) as info:
        admit(world)
    assert info.value.stage == "output" and H._SPY is None


def test_the_code_tree_may_not_lie_under_the_step_11_checkout_or_a_watched_root(world, monkeypatch):
    monkeypatch.setattr(T, "STEP11_CHECKOUT", world.repo.parent)
    with pytest.raises(H.HoldoutRefused) as info:
        admit(world)
    assert info.value.stage == "code" and "C24.12-13" in str(info.value) and H._SPY is None


def test_folding_metrics_is_checked_before_the_lock(world, monkeypatch):
    context = admit(world)

    def refused(repo=None):
        raise H.HoldoutRefused("code", "folding_metrics imported from elsewhere (synthetic)")
    monkeypatch.setattr(T, "folding_module", refused)
    with pytest.raises(H.HoldoutRefused) as info:
        T.compute(context, echo=lambda *_: None)
    assert info.value.stage == "code" and not (T.root()/T.LOCK_NAME).exists() and H._SPY is None


@pytest.mark.parametrize("case", ["done_then_refused", "failed", "stopped", "error", "sealed", "identity"])
def test_the_command_lines_exit_codes(world, monkeypatch, case, capsys):
    monkeypatch.setattr(H, "REPO", world.repo)
    monkeypatch.setattr(H, "machine_gate", lambda **kwargs: dict(stub="machine gate (tested in the entry's tests)", passed=True))
    argv = ["compute", "--reviewed-commit", "f"*40]
    S._reset_cache()
    if case == "done_then_refused":
        assert T.main(argv) == T.EXIT["done"] == 0
        S._reset_cache()
        assert T.main(argv) == T.EXIT["refused"] == 2                           # the lock is there
    elif case == "failed":
        original = T.compute_cell
        monkeypatch.setattr(T, "compute_cell", lambda cell, *a: (_ for _ in ()).throw(OSError("synthetic")) if cell.arm == "fga" else original(cell, *a))
        assert T.main(argv) == T.EXIT["failed"] == 1
    elif case == "stopped":
        np.savez(field_path(world, "cadam", world.rows[1]), d=np.zeros((3,)+syn.SHAPE, dtype=np.float32))
        assert T.main(argv) == T.EXIT["stopped"] == 4
    elif case == "error":                                                       # an exception outside the cell loop
        monkeypatch.setattr(T, "summarize", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic")))
        assert T.main(argv) == T.EXIT["error"] == 5
        assert (T.root()/T.LOCK_NAME).exists() and len(list((T.root()/"cells").rglob("*.json"))) == len(outputs(world))     # what was written is kept
    elif case == "sealed":
        monkeypatch.setattr(T, "admit", lambda **kwargs: (_ for _ in ()).throw(SubjectSealed("synthetic OASIS_OAS1_0901_MR1")))
        assert T.main(argv) == T.EXIT["sealed"] == 3
    else:
        monkeypatch.setattr(T, "identity", lambda repo=None, *, reviewed_commit=None, run=None: (stub_identity()[0], ["synthetic OASIS_OAS1_0901_MR1"]))
        assert T.main(argv) == T.EXIT["refused"] == 2
    captured = capsys.readouterr()
    assert not re.search(r"OASIS_OAS1_[0-9]{4}", captured.out+captured.err) and all(r[3] not in captured.out+captured.err for r in world.rows)


def test_write_atomic_writes_once_and_leaves_no_temporary_file(tmp_path):
    path = T.write_json(tmp_path/"a"/"x.json", dict(a=1))
    with pytest.raises(FileExistsError):
        T.write_json(path, dict(a=2))
    assert json.loads(path.read_text()) == dict(a=1) and [p.name for p in path.parent.iterdir()] == ["x.json"]


def test_the_rerun_numbers_start_at_1_without_a_leading_zero():
    good = [f"holdout_B_v1_tet24_rerun_{k}.json" for k in ("1", "9", "10", "999")]
    bad = [f"holdout_B_v1_tet24_rerun_{k}.json" for k in ("0", "00", "01", "007", "1000", "")]
    assert all(T.RERUN_NAME.fullmatch(n) for n in good) and not any(T.RERUN_NAME.fullmatch(n) for n in bad)
    assert all(CI.LATER_AUTHORIZATION.fullmatch(f"{T.RERUN_DIR_REL}/{n}") for n in good)       # rule one admits the authorization


# ---- the code identity's one exception (C24.12-2), on temporary git repositories --------------------------------------------------------------------------
ALLOW, UNSEAL, RECORD = CI.ALLOWLIST_REL, CI.UNSEAL_RECORD_REL, CI.PINNED_RECORD_REL
CODE = ("bayesreg/a.py", "bayesreg/eval/fields.py", "scripts/folding_metrics.py", ALLOW, "jd_curl/b.py")
OTHER = ("docs/decision_log.md", "splits/bayesreg_v1/holdout_B_pairs.tsv", "runs/holdout_B_v1/README.md", "tests/test_bayesreg/__init__.py")


def put(r, rel, text):
    (r/rel).parent.mkdir(parents=True, exist_ok=True)
    (r/rel).write_text(text, encoding="utf-8")


def code_repo(tmp_path):
    """main: P1 (the code), then the reviewed module (the stand-in of the merge that brings it), then a later commit that does not touch it.  holdout: P1, then the
    inclusion commit of C23.3-6 (the allowlist modified, the unseal record and pinned_code.json added, the decision log's line appended).  Returns (repo on holdout,
    P1, the reviewed commit, main's tip)."""
    r = tmp_path/"code"
    for rel in CODE+OTHER:
        put(r, rel, rel+"\n")
    git(r.parent, "init", "-q", str(r))
    commit(r, "P1")
    git(r, "branch", "-M", "main")                                             # T.MAIN_REF is its full ref refs/heads/main
    pin = head(r)
    git(r, "checkout", "-q", "-b", "holdout")
    text = '{"included": ["OASIS_OAS1_0001_MR1"]}\n'
    put(r, ALLOW, text)
    put(r, UNSEAL, json.dumps(dict(schema="bayesreg.unseal.v1", allowlist_sha256=hashlib.sha256(text.encode()).hexdigest())))
    line = f"| 2026-11-16 | P1 = {pin} (test) |"
    put(r, RECORD, json.dumps(dict(schema=CI.PINNED_RECORD_SCHEMA, commit=pin, decision_log_line=line, written_by="lead", time_local="t", utc="2026-11-16T00:00:00Z",
                                   reason="test")))
    with open(r/"docs/decision_log.md", "a") as stream:
        stream.write(line+"\n")
    commit(r, "inclusion")
    git(r, "checkout", "-q", "main")
    put(r, T.MODULE_REL, "# the module\n")
    commit(r, "main: the reviewed T24 module")
    reviewed = head(r)
    put(r, "docs/later.md", "later\n")
    commit(r, "main: a later commit")
    tip = head(r)
    git(r, "checkout", "-q", "holdout")
    return r, pin, reviewed, tip


def add_module(r, reviewed, extra=()):
    """The T24 branch's one commit: the module exactly as on the reviewed commit of main."""
    git(r, "checkout", reviewed, "--", T.MODULE_REL)
    for rel in extra:
        put(r, rel, "x\n")
    commit(r, "the T24 module (= main's reviewed commit)")
    return head(r)


def test_the_identity_allows_one_commit_that_only_adds_the_reviewed_module(tmp_path):
    r, pin, reviewed, _ = code_repo(tmp_path)
    record, problems = T.identity(r, reviewed_commit=reviewed)
    assert problems and any("exactly one commit" in p for p in problems)            # no module on the hold-out line yet
    module_commit = add_module(r, reviewed)
    with open(r/"docs/decision_log.md", "a") as stream:                         # later records are still allowed (rule one)
        stream.write("| 2026-11-19 | appended |\n")
    commit(r, "log")
    rule_one, rule_one_problems = CI.unit_identity_problems(repo=r)
    assert rule_one_problems                                                    # rule one alone refuses the module
    record, problems = T.identity(r, reviewed_commit=reviewed[:12])
    assert problems == [] and record["passed"] and record["pinned"] == pin and record["module"]["commit"] == module_commit
    assert record["module"]["reviewed_commit"] == reviewed and len(record["excused"]) == 2 and record["module"]["blob"] == record["module"]["reviewed_blob"]


@pytest.mark.parametrize("how", ["with_a_test", "changed_after", "other_code", "no_reviewed", "not_on_main", "main_tip", "main_changed_it", "reviewed_differs",
                                 "uncommitted", "dirty", "pyc"])
def test_the_identity_refuses(tmp_path, how):
    r, _, reviewed, tip = code_repo(tmp_path)
    given = reviewed
    if how == "with_a_test":
        add_module(r, reviewed, extra=("tests/test_x.py",))
    elif how == "changed_after":
        add_module(r, reviewed)
        put(r, T.MODULE_REL, "# changed\n")
        commit(r, "change")
    elif how == "other_code":
        add_module(r, reviewed)
        put(r, "bayesreg/eval/fields.py", "changed\n")
        commit(r, "other")
    elif how == "no_reviewed":
        add_module(r, reviewed)
        given = None
    elif how == "not_on_main":                                                  # the hold-out line's own commit: not an ancestor of main
        given = add_module(r, reviewed)
    elif how == "main_tip":                                                     # a later commit of main that does not bring the module
        add_module(r, reviewed)
        given = tip
    elif how == "main_changed_it":
        add_module(r, reviewed)
        git(r, "checkout", "-q", "main")
        put(r, T.MODULE_REL, "# main changed it since\n")
        commit(r, "main: change")
        git(r, "checkout", "-q", "holdout")
    elif how == "reviewed_differs":                                             # another version on a side branch
        add_module(r, reviewed)
        git(r, "checkout", "-q", "-b", "side", reviewed)
        put(r, T.MODULE_REL, "# another version\n")
        commit(r, "side")
        given = head(r)
        git(r, "checkout", "-q", "holdout")
    elif how == "uncommitted":
        git(r, "checkout", reviewed, "--", T.MODULE_REL)
        git(r, "reset", "-q")
    elif how == "dirty":
        add_module(r, reviewed)
        put(r, T.MODULE_REL, "# edited in the working tree\n")
    else:                                                                       # an untracked .pyc: why the units run python -B with PYTHONDONTWRITEBYTECODE=1
        add_module(r, reviewed)
        put(r, "bayesreg/eval/__pycache__/holdout_tet24.cpython-310.pyc", "x")
    record, problems = T.identity(r, reviewed_commit=given)
    assert problems and not record["passed"], how
