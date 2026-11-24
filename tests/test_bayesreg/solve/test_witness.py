"""Certificate witness record (contract C05-2b "记录与 CLI", C16 log.jsonl; work packages IMPL27, IMPL27b;
bayesreg.solve.witness).

- classify_point: the exact class of a reference point, one example per type (corner, edge_mid, face_center,
  body_center, edge_interior, face_interior, cell_interior) and floating-point equality (a coordinate 1e-12 off a
  lattice value is not on it); point_type_53 of the 53 sample points (Gauss centre = body_center, other Gauss points
  "gauss27");
- W1(17/20) (02_理论 B3'-2; edge-midpoint dip): JACOBIAN_VIOLATION, witness_cell K0, witness_xi (0, 1, 1/2),
  witness_type edge_mid, witness_cell_min_j53 J = 1 - b - b^2/4 < 0 at a closed27 edge midpoint (K* / W1 type);
- W2(19/20) (53 sample points J >= 1/20, the edge interior negative): witness_type is not a closed27 lattice class and
  witness_cell_min_j53 J > 0 (>= 1/20; W2 type) -- the contract gate;
- the optional estimate (witness_estimate on): W2's -41/800 at (0, 1, 29/38), off the sample points; W1's equals the
  53-point minimum; no 〔估计〕 in any field value;
- every outcome without a witness point (passed; an uncertain result) has the four fields null; the record reaches
  CertOutcome.record() (cert.json, transfer rows) at the top level; witness_estimate off by default; WitnessSummary
  counts by type and by the sign of the 53-point minimum;
- run_pg: every candidate row carries the four fields (values for the rejection with a witness, null otherwise).
Small grids; the CPU certificate (numba compilation on first use), no GPU.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from jd_curl.three_d.boundary import restrict_free

from bayesreg.solve import NumpyBackend, make_backend, run_pg
from bayesreg.solve import witness as W
from bayesreg.solve.backend import certificate_config, certify

from ..model.test_logvol_closed27 import w1_field, w2_field, w_grid
from .test_jd_scaling import Holder, Point, Quadratic, fresh_metric, outcome, wavy_field

K0 = (2, 2, 2)


def _no_estimate_marker(value):
    """No 〔估计〕 anywhere in a (nested) record value."""
    if isinstance(value, dict):
        return all(_no_estimate_marker(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_no_estimate_marker(v) for v in value)
    return "〔估计〕" not in str(value)


def test_classify_point_exact():
    cases = {(0., 0., 0.): "corner", (1., 0., 1.): "corner", (0., 1., .5): "edge_mid", (.5, 0., 1.): "edge_mid",
             (.5, .5, 0.): "face_center", (.5, .5, .5): "body_center", (0., 1., .7631): "edge_interior",
             (0., .25, 1.): "edge_interior", (.3, 1., .2): "face_interior", (.5, .5, 1e-3): "cell_interior",
             (.3, .6, .2): "cell_interior", (.75, .5, 0.): "face_interior"}
    for point, kind in cases.items():
        assert W.classify_point(point) == kind, point
    assert set(W.WITNESS_TYPES) == set(cases.values()) and len(W.WITNESS_TYPES) == 7
    assert set(W.SAMPLE_TYPES) == {"corner", "edge_mid", "face_center", "body_center"}
    # exact comparison: no tolerance (the IMPL27 version used 1e-6)
    assert W.classify_point((0., 1., .5+1e-12)) == "edge_interior"
    assert W.classify_point((1e-12, 1., .5)) == "face_interior"
    g1, g2 = .5-np.sqrt(15.)/10, .5+np.sqrt(15.)/10
    assert W.point_type_53((g1, .5, g2)) == "gauss27" and W.point_type_53((.5, .5, .5)) == "body_center"
    assert W.point_type_53((0., 1., .5)) == "edge_mid"
    assert W.null_fields() == dict(witness_cell=None, witness_xi=None, witness_type=None, witness_cell_min_j53=None)


@pytest.mark.parametrize("spacing", [(1., 1., 1.), (.9, 1.1, 1.3)], ids=["isotropic", "anisotropic"])
def test_w1_witness_at_the_edge_midpoint(spacing):
    grid = w_grid(spacing)
    out = certify(w1_field(grid, 17/20), grid, certificate_config())
    assert not out.passed and out.rejection == "JACOBIAN_VIOLATION"
    w = out.witness
    k0 = int(np.ravel_multi_index(K0, grid.cell_shape))
    assert set(w) == set(W.WITNESS_FIELDS)
    assert w["witness_cell"] == out.witness_cell == k0 and w["witness_xi"] == [0., 1., .5]
    assert w["witness_type"] == out.witness_type == "edge_mid"
    m = w["witness_cell_min_j53"]
    assert m["point_set"] == "closed27" and m["point_type"] == "edge_mid" and m["xi"] == [0., 1., .5]
    assert abs(m["J"]-(1-.85-.85**2/4)) <= 1e-14 and m["J"] < 0                    # K* / W1 type
    assert out.witness_estimate is None                                            # default: no estimate
    record = out.record()
    for key in W.WITNESS_FIELDS:
        assert record[key] == w[key]
    assert record["witness_estimate"] is None and "witness" not in record and _no_estimate_marker(record)


def test_w2_witness_off_the_lattice_with_positive_53_points():
    """Contract gate: the W2(19/20) candidate's witness_type is not a closed27 lattice class and
    witness_cell_min_j53 > 0 (the 18 cells W2 touches have J >= 1/20 at the 54 sample values, B3'-2)."""
    grid = w_grid((1., 1., 1.))
    out = certify(w2_field(grid, 19/20), grid, certificate_config())
    assert not out.passed and out.rejection == "JACOBIAN_VIOLATION"
    w = out.witness
    assert w["witness_type"] in W.OFF_SAMPLE_TYPES and w["witness_type"] not in W.SAMPLE_TYPES
    xi = w["witness_xi"]
    assert xi == [float(t) for t in out.witness_reference_point] and not all(t in (0., .5, 1.) for t in xi)
    m = w["witness_cell_min_j53"]
    assert m["J"] > 0 and m["J"] >= 1/20-1e-14                                    # W2 type
    assert _no_estimate_marker(out.record())


def test_witness_estimate_option():
    """witness_estimate on: W2's estimated minimum -41/800 at (0, 1, 29/38) (off the sample points, an edge interior);
    W1's estimate equals the 53-point minimum (at a sample point); values are plain numbers and names."""
    grid = w_grid((1., 1., 1.))
    w2 = certify(w2_field(grid, 19/20), grid, certificate_config(), witness_estimate=True)
    k0 = int(np.ravel_multi_index(K0, grid.cell_shape))
    assert w2.witness_cell == k0
    e = w2.witness_estimate
    assert abs(e["J"]+41/800) <= 1e-12 and e["location_type"] == "edge_interior" and not e["at_sample_point"]
    x, y, t = e["xi"]
    assert (x, y) == (0., 1.) and abs(t-29/38) <= 1e-6 and e["nearest_sample_distance"] > .1
    direct = W.witness_estimate(w2_field(grid, 19/20), grid, k0)
    assert abs(direct["J"]+41/800) <= 1e-12 and direct["location_type"] == "edge_interior"
    w1 = certify(w1_field(grid, 17/20), grid, certificate_config(), witness_estimate=True)
    e1 = w1.witness_estimate
    assert e1["J"] == w1.witness["witness_cell_min_j53"]["J"] and e1["at_sample_point"]
    assert e1["location_type"] == "edge_mid"
    assert _no_estimate_marker(w1.record()) and _no_estimate_marker(w2.record())
    assert w1.record()["witness_estimate"] == e1


def test_outcomes_without_a_witness_point_have_null_fields():
    grid = w_grid((1., 1., 1.))
    backend = NumpyBackend()
    assert backend.witness_estimate is False and backend.record()["witness_estimate"] is False
    passed = backend.certify(np.zeros(grid.shape+(3,)), grid)
    assert passed.passed and passed.witness is None and passed.witness_fields() == W.null_fields()
    assert all(passed.record()[k] is None for k in W.WITNESS_FIELDS)
    uncertain = SimpleNamespace(status="uncertain", reason="subdivision_budget", witness_cell=None,
                                witness_reference_point=None)
    assert W.witness_fields(np.zeros(grid.shape+(3,)), grid, uncertain) == W.null_fields()
    infeasible = SimpleNamespace(status="infeasible", reason="INVALID_DOMAIN", witness_cell=3,
                                 witness_reference_point=(0., 0., 0.))
    assert W.witness_fields(np.zeros(grid.shape+(3,)), grid, infeasible) == W.null_fields()
    with pytest.raises(Exception):
        make_backend("numpy", witness_estimate="on")
    assert make_backend("torch64", device="cpu", witness_estimate=True).witness_estimate is True


def test_backend_summary_counts_types():
    grid = w_grid((1., 1., 1.))
    backend = NumpyBackend()
    for u in (w1_field(grid, 17/20), w2_field(grid, 19/20), w1_field(grid, 17/20), np.zeros(grid.shape+(3,))):
        backend.certify(u, grid)
    w2_type = backend.certify(w2_field(grid, 19/20), grid).witness_type
    summary = backend.witness_summary.record()
    assert summary["certificates"] == 5 and summary["jacobian_violations"] == 4
    assert summary["witness_type_counts"]["edge_mid"] == 2 and summary["witness_type_counts"][w2_type] == 2
    assert summary["min_j53_positive"] == 2 and summary["min_j53_nonpositive_point_type_counts"] == {"edge_mid": 2}
    assert backend.witness_records == 4 and backend.witness_seconds > 0
    again = NumpyBackend().certify(w1_field(grid, 17/20), grid)
    assert again.witness == backend.certify(w1_field(grid, 17/20), grid).witness      # deterministic record


def test_run_pg_rows_carry_the_witness_fields():
    """Every candidate row carries witness_cell, witness_xi, witness_type, witness_cell_min_j53: values for the
    certificate rejection with a witness record, null for the accepted candidate (C16: no witness point -> null)."""
    metric, grid = fresh_metric()
    objective = Quadratic(Holder(grid), .3*restrict_free(wavy_field(grid, amplitude=1., seed=4), grid))
    calls = []
    fields = dict(witness_cell=3, witness_xi=[0., 1., .25], witness_type="edge_interior",
                  witness_cell_min_j53=dict(J=.05, point_set="closed27", point_index=0, point_type="corner",
                                            xi=[0., 0., 0.]))

    def certify_(u, grid_):
        calls.append(1)
        out = outcome(u, grid_, len(calls) > 1)
        return out if out.passed else replace(out, witness=fields)
    zero = np.zeros(grid.shape+(3,))
    rows = []
    run_pg(objective, Point(zero, 1.), theta=restrict_free(zero, grid), cert=outcome(zero, grid, True),
           metric=metric, certify=certify_, target=0., max_steps=1, log=rows.append)
    candidates = [r for r in rows if r["kind"] == "candidate"]
    assert len(candidates) >= 2
    assert {k: candidates[0][k] for k in W.WITNESS_FIELDS} == fields
    assert all(all(r[k] is None for k in W.WITNESS_FIELDS) for r in candidates[1:])
    assert all("witness" not in r and "witness_estimate" not in r for r in candidates)
