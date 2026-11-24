"""The certificate witness record (contract C05-2b "记录与 CLI", C16 log.jsonl; work packages IMPL27, IMPL27b after
the review IMPL27_复核_20261030 item 2).  A record only: no decision reads it, the certificate (C07) and its outcome
are never changed here, and the step sequence and every other log field keep their values.

Contract fields (`WITNESS_FIELDS`; recorded for every logvol_points value, on every candidate row; a candidate without a
certificate witness point -- passed, FEASIBILITY_UNCERTAIN, INVALID_DOMAIN, NONFINITE_STATE, no certificate call --
carries them as null):
  witness_cell          certify_map's witness_cell (the first infeasible cell in tile order; not where J is smallest
                        over the field);
  witness_xi            its witness_reference_point, reference coordinates in that cell, order x, y, z;
  witness_type          the exact class of witness_xi (`classify_point`, floating-point equality, no tolerance):
                        every coordinate in {0, 1/2, 1}: one of the 53 sample points (a closed27 lattice point),
                        "corner" / "edge_mid" / "face_center" / "body_center" by the number (3, 2, 1, 0) of coordinates
                        in {0, 1}; otherwise by that number: 2 "edge_interior", 1 "face_interior", 0 "cell_interior".
                        jd_curl's witnesses are lattice points (sample_witness, POINTS = {0, 1/2, 1}^3) or corners of a
                        subdivision box (subdivision_witness, dyadic coordinates), never a Gauss point;
  witness_cell_min_j53  the floating-point minimum of J of the witness cell's own Q1 map over its 53 sample points
                        (Gauss27 U closed27, the body centre in both sets), by geom.tile_geometry with the model's
                        operators (bitwise the model's J there): dict J, point_set ("gauss27" | "closed27"), point_index
                        (index into that set's ops.points), point_type (`point_type_53`: the lattice class of a closed27
                        point, "body_center" for the Gauss centre point, else "gauss27") and xi.  Ties go to the Gauss
                        set (C06-7 order: Gauss block first).
Reading: J > 0 while the certificate found a violation is W2 type (53 points positive, negative between them);
J <= 0 with point_type "edge_mid" is K* / W1 type.

Optional diagnostic (`witness_estimate`, off by default; run_pair --witness-estimate on): an estimate of the minimum J
over the closed witness cell -- dense sampling (33^3 points, the closed27 lattice included) plus the Gauss points, then
bounded L-BFGS-B from the 4 best samples (the method of experiments/diag/g3_direction.py).  J is a polynomial of degree
<= 2 in each variable and the search is local: the value is an estimate (an upper bound of the true minimum up to
rounding), not a certified quantity; its location class uses a tolerance (POINT_TOL).  It costs a few dense J
evaluations per rejection, so it is off unless asked for (timing comparisons between arms stay clean).

`WitnessSummary` counts, per run (one arm), the JACOBIAN_VIOLATION outcomes by witness_type and by the sign and
point type of witness_cell_min_j53 (contract: "运行汇总按臂统计各类型的计数").
"""
import numpy as np

from jd_curl.three_d.cell_q1 import VERTICES

from ..geom import closed27, gauss27, node_field, tile_geometry

LATTICE = (0., .5, 1.)
SAMPLE_TYPES = ("corner", "edge_mid", "face_center", "body_center")       # the closed27 lattice (53 sample points)
OFF_SAMPLE_TYPES = ("edge_interior", "face_interior", "cell_interior")
WITNESS_TYPES = SAMPLE_TYPES+OFF_SAMPLE_TYPES
GAUSS_POINT = "gauss27"            # point_type of a Gauss point off the lattice (witness_cell_min_j53 only)
WITNESS_FIELDS = ("witness_cell", "witness_xi", "witness_type", "witness_cell_min_j53")
_LATTICE_KIND = {3: "corner", 2: "edge_mid", 1: "face_center", 0: "body_center"}
_OFF_KIND = {2: "edge_interior", 1: "face_interior", 0: "cell_interior"}

POINT_TOL = 1e-6                   # estimate only: location class with a tolerance (L-BFGS-B ends on active bounds)
VALUE_TOL = 1e-12                  # estimate only: "at a sample point" when min53 - estimate <= VALUE_TOL max(1, |J|)
GRID_N = 33
STARTS = 4
ESTIMATE_METHOD = ("estimate (not a bound): min J over the witness cell from 33^3 samples (closed27 lattice included) "
                   "+ the 27 Gauss points, then bounded L-BFGS-B from the 4 best samples; local search")


def _coordinates(point):
    p = [float(t) for t in np.asarray(point, dtype=np.float64).ravel()]
    if len(p) != 3:
        raise ValueError("a reference point (x, y, z) required")
    return p


def classify_point(point):
    """The exact class of a reference point xyz of a cell (contract C05-2b; floating-point equality): a lattice point
    {0, 1/2, 1}^3 is "corner" / "edge_mid" / "face_center" / "body_center" (3, 2, 1, 0 coordinates in {0, 1});
    otherwise "edge_interior" / "face_interior" / "cell_interior" (2, 1, 0 coordinates in {0, 1})."""
    p = _coordinates(point)
    boundary = sum(1 for t in p if t == 0. or t == 1.)
    if all(t in LATTICE for t in p):
        return _LATTICE_KIND[boundary]
    return _OFF_KIND[boundary]


def point_type_53(point):
    """The type of one of the 53 sample points: its lattice class when every coordinate is in {0, 1/2, 1} (a closed27
    point or the Gauss centre), else "gauss27"."""
    p = _coordinates(point)
    return classify_point(p) if all(t in LATTICE for t in p) else GAUSS_POINT


def null_fields():
    """The contract fields of a candidate without a certificate witness point (all null)."""
    return {name: None for name in WITNESS_FIELDS}


def sample_values(u, grid, cell):
    """J of the model's operators at the 27 Gauss and the 27 closed27 points of `cell`: {"gauss27": (J (27,), points
    (27, 3)), "closed27": (J, points)} (geom.tile_geometry; bitwise the model's J there)."""
    tile = slice(int(cell), int(cell)+1)
    field = node_field(u, grid, allow_nonzero_faces=True)
    out = {}
    for name, ops in (("gauss27", gauss27(grid)), ("closed27", closed27(grid))):
        out[name] = (np.asarray(tile_geometry(field, ops, tile=tile).J[0]), np.asarray(ops.points))
    return out


def min_j53(u, grid, cell):
    """witness_cell_min_j53 of `cell` (module docstring)."""
    samples = sample_values(u, grid, cell)
    (jg, pg), (js, ps) = samples["gauss27"], samples["closed27"]
    kg, ks = int(np.argmin(jg)), int(np.argmin(js))
    if js[ks] < jg[kg]:
        value, point_set, index, point = float(js[ks]), "closed27", ks, ps[ks]
    else:
        value, point_set, index, point = float(jg[kg]), "gauss27", kg, pg[kg]
    return dict(J=value, point_set=point_set, point_index=index, point_type=point_type_53(point),
                xi=[float(t) for t in point])


def witness_fields(u, grid, result):
    """The contract fields of a certify_map result: values for an infeasible JACOBIAN_VIOLATION outcome with a witness
    cell, else `null_fields()`."""
    if not (result.status == "infeasible" and result.reason == "JACOBIAN_VIOLATION"
            and result.witness_cell is not None):
        return null_fields()
    cell = int(result.witness_cell)
    point = result.witness_reference_point
    xi = None if point is None else _coordinates(point)
    return dict(witness_cell=cell, witness_xi=xi, witness_type=None if xi is None else classify_point(xi),
                witness_cell_min_j53=min_j53(u, grid, cell))


# ---- optional diagnostic: an estimate of the cell minimum ------------------------------------------------------

def _near_boundary(t, tol):
    return t <= tol or t >= 1-tol


def classify_estimate_location(point, tol=POINT_TOL):
    """Location class of an estimated minimum (tolerance tol): a lattice class when every coordinate is within tol of
    0, 1/2 or 1; "gauss27" when every coordinate is within tol of a Gauss node; else edge / face / cell interior by the
    number of coordinates within tol of 0 or 1."""
    p = _coordinates(point)
    if all(min(abs(t-c) for c in LATTICE) <= tol for t in p):
        return _LATTICE_KIND[sum(1 for t in p if _near_boundary(t, tol))]
    nodes = (.5-np.sqrt(15.)/10, .5, .5+np.sqrt(15.)/10)
    if all(min(abs(t-g) for g in nodes) <= tol for t in p):
        return GAUSS_POINT
    return _OFF_KIND[sum(1 for t in p if _near_boundary(t, tol))]


def _dense_points(n=GRID_N):
    t = np.linspace(0., 1., n)
    z, y, x = np.meshgrid(t, t, t, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)


def cell_vertices(u, grid, cell):
    """(8, 3) displacements (mm, xyz) of the vertices of `cell` (C order of grid.cell_shape) in VERTICES order."""
    cz, cy, cx = (int(v) for v in np.unravel_index(int(cell), grid.cell_shape))
    return np.array([u[cz+z, cy+y, cx+x] for x, y, z in VERTICES], dtype=np.float64)


def cell_J(vertices, spacing, points):
    """J = det(I + grad u_h) of one cell's Q1 map at reference points (P, 3) xyz (independent formula: shape
    derivatives of the scalar Q1 basis, cofactor expansion along the first row)."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    dN = np.empty((points.shape[0], 8, 3))
    for a, bits in enumerate(VERTICES):
        f = [points[:, j] if bits[j] else 1-points[:, j] for j in range(3)]
        d = [1. if bits[j] else -1. for j in range(3)]
        dN[:, a, 0] = d[0]*f[1]*f[2]
        dN[:, a, 1] = f[0]*d[1]*f[2]
        dN[:, a, 2] = f[0]*f[1]*d[2]
    dN /= np.asarray(spacing, dtype=np.float64)[None, None, :]
    F = np.eye(3)+np.einsum("ar,paj->prj", vertices, dN)
    return (F[:, 0, 0]*(F[:, 1, 1]*F[:, 2, 2]-F[:, 1, 2]*F[:, 2, 1])
            - F[:, 0, 1]*(F[:, 1, 0]*F[:, 2, 2]-F[:, 1, 2]*F[:, 2, 0])
            + F[:, 0, 2]*(F[:, 1, 0]*F[:, 2, 1]-F[:, 1, 1]*F[:, 2, 0]))


def cell_minimum(vertices, spacing, extra_points=None):
    """Estimate (min J, reference point xyz) of one cell: dense samples (+ extra_points), then bounded L-BFGS-B from
    the STARTS best samples."""
    from scipy.optimize import minimize
    points = _dense_points()
    if extra_points is not None:
        points = np.concatenate((points, np.asarray(extra_points, dtype=np.float64)))
    values = cell_J(vertices, spacing, points)
    k = int(np.argmin(values))
    best = (float(values[k]), points[k].copy())
    for start in np.argsort(values, kind="stable")[:STARTS]:
        res = minimize(lambda x: float(cell_J(vertices, spacing, x[None])[0]), points[start], method="L-BFGS-B",
                       bounds=[(0., 1.)]*3, options=dict(ftol=1e-15, gtol=1e-13, maxiter=500))
        x = np.clip(res.x, 0., 1.)
        value = float(cell_J(vertices, spacing, x[None])[0])
        if value < best[0]:
            best = (value, x)
    return best


def witness_estimate(u, grid, cell, min53=None):
    """The optional diagnostic of the witness cell (module docstring): dict J, xi, location_type (tolerance class),
    at_sample_point (the 53-point minimum is within VALUE_TOL of the estimate), nearest_sample_distance, method."""
    u = np.ascontiguousarray(np.asarray(u, dtype=np.float64))
    min53 = min_j53(u, grid, cell) if min53 is None else min53
    samples = sample_values(u, grid, cell)
    gauss_points = samples["gauss27"][1]
    value, location = cell_minimum(cell_vertices(u, grid, cell), grid.spacing, extra_points=gauss_points)
    sample_points = np.concatenate((gauss_points, samples["closed27"][1]))
    distance = float(np.min(np.max(np.abs(sample_points-location[None, :]), axis=1)))
    return dict(J=float(value), xi=[float(t) for t in location], location_type=classify_estimate_location(location),
                at_sample_point=bool(min53["J"]-value <= VALUE_TOL*max(1., abs(value))),
                nearest_sample_distance=distance, method=ESTIMATE_METHOD)


# ---- per-run summary ------------------------------------------------------------------------------------------------

class WitnessSummary:
    """Counts over the certificate outcomes of one run (one arm): JACOBIAN_VIOLATION outcomes by witness_type (None:
    no witness point reported), by the sign of witness_cell_min_j53 (J > 0: W2 type) and, for J <= 0, by its
    point_type (edge_mid: K* / W1 type)."""

    def __init__(self):
        self.certificates = 0
        self.jacobian_violations = 0
        self.by_type = {}
        self.min53_positive = 0
        self.min53_nonpositive = {}
        self.errors = 0

    def observe(self, outcome):
        self.certificates += 1
        if getattr(outcome, "rejection", None) != "JACOBIAN_VIOLATION":
            return
        self.jacobian_violations += 1
        fields = outcome.witness_fields() if callable(getattr(outcome, "witness_fields", None)) else null_fields()
        if fields.get("witness_error"):
            self.errors += 1
        kind = str(fields.get("witness_type"))
        self.by_type[kind] = self.by_type.get(kind, 0)+1
        min53 = fields.get("witness_cell_min_j53")
        if min53 is None:
            return
        if min53["J"] > 0:
            self.min53_positive += 1
        else:
            point = min53["point_type"]
            self.min53_nonpositive[point] = self.min53_nonpositive.get(point, 0)+1

    def record(self):
        return dict(certificates=self.certificates, jacobian_violations=self.jacobian_violations,
                    witness_type_counts={k: self.by_type[k] for k in sorted(self.by_type)},
                    min_j53_positive=self.min53_positive,
                    min_j53_nonpositive_point_type_counts={k: self.min53_nonpositive[k]
                                                           for k in sorted(self.min53_nonpositive)},
                    witness_errors=self.errors,
                    note="every certificate call of the backend (line-search and transfer candidates, level starts, "
                         "the export); witness_type 'None': no witness point; min_j53_positive: the witness cell's 53 "
                         "sample points all J > 0 (W2 type); min_j53 <= 0 at edge_mid: K* / W1 type")


__all__ = ["ESTIMATE_METHOD", "GAUSS_POINT", "LATTICE", "OFF_SAMPLE_TYPES", "POINT_TOL", "SAMPLE_TYPES", "VALUE_TOL",
           "WITNESS_FIELDS", "WITNESS_TYPES", "WitnessSummary", "cell_J", "cell_minimum", "cell_vertices",
           "classify_estimate_location", "classify_point", "min_j53", "null_fields", "point_type_53", "sample_values",
           "witness_estimate", "witness_fields"]
