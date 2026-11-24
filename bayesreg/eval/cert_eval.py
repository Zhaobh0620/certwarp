"""Certificate of an evaluated field (contract C07; P2, P3, P7 cited, never changed).

Definitions:
  configuration  the C07-2 configuration only: PrescreenCertificateConfig(delta=0., tile_size=T) with max_depth 48,
                 max_boxes 255 (bayesreg.solve.backend.certificate_config; T = 8192 unless given, recorded); CPU FP64
                 with outward rounding. Never the CertificateConfig() default.
  certified map  u = h d (mm, (nz, ny, nx, 3), xyz) on the pair's node grid; its trilinear (Q1) interpolant is the map
                 the certificate covers and the map Learn2Reg-style trilinear warping applies.
  full           jd_curl certify_map(u, grid, config): non-finite state, six faces zero (NONZERO_BOUNDARY), sampling
                 domain (INVALID_DOMAIN), then every cell. For bayesreg outputs (and configuration C maps).
  jacobian_only  the same per-cell certificate without the NONZERO_BOUNDARY and INVALID_DOMAIN early returns (neither
                 is a Jacobian property): the same owned C-order copy, digest, result record and per-cell core
                 feasibility._certify_map_parallel (copy with provenance of BL1 item 6, bl1_cert.certify_jd_only,
                 archive/seq_vp_3d/results/verification/3d_stage_BL1_v1/analysis/bl1_cert.py). For baselines, whose
                 faces are generally not zero. The face and domain facts are reported next to it (faces_max_abs_u_mm,
                 domain), never folded into the status.
  passed         status certified_feasible, min_jd_lower_bound > 0, n_infeasible = n_uncertain = 0. "Zero" and
                 "uncertain" do not pass (C07-3).
  cells          a per-cell pass with the decisions of _certify_map_parallel (same edge-hull margins with the
                 whole-tile flag rule, same float Bernstein prescreen, same compiled interval Bernstein fallback per
                 tile job, same _classify), recording each cell's outcome and lower bound:
                 proven (edge hull, prescreen, or fallback), infeasible (a point with J <= 0 certified), uncertain
                 (budget or arithmetic); counted over brain cells (lower-corner node in the brain mask) and background
                 cells; min lower bound per region. Its counts are compared with the map-level result whenever that
                 result reached its Jacobian part (cells.matches_map); a mismatch is reported, never hidden.
"""
import time

import numpy as np

from jd_curl.validation import as_float64_array
from jd_curl.three_d import feasibility as F
from jd_curl.three_d import interval as iv
from jd_curl.three_d.cell_q1 import cell_tiles
from jd_curl.three_d.interpolation import check_map_domain

from bayesreg.solve.backend import CERT_TILE_SIZE, certificate_config, certificate_record, check_certificate_config

MODES = ("full", "jacobian_only")
DEFINITIONS = "Definitions" + __doc__.split("Definitions", 1)[1].rstrip()
PROVEN_HULL, PROVEN_PRESCREEN, PROVEN_FALLBACK, INFEASIBLE, UNCERTAIN, UNCERTAIN_ARITHMETIC = range(6)
EARLY_REASONS = ("NONFINITE_STATE", "NONZERO_BOUNDARY", "INVALID_DOMAIN", "unsupported_arithmetic")
RESULT_FIELDS = ("status", "reason", "min_jd_lower_bound", "min_margin_lower_bound", "sampled_min_jd", "n_cells_checked",
                 "n_uncertain", "n_infeasible", "n_boxes", "max_depth", "witness_cell", "witness_reference_point",
                 "n_fast_cells", "n_fallback_cells", "n_prescreen_cells", "state_hash", "grid_id", "delta_hex",
                 "certificate_version")


def certify_jacobian_only(u, grid, config):
    """certify_map(u, grid, config=config) without its NONZERO_BOUNDARY and INVALID_DOMAIN early returns."""
    import hashlib
    from dataclasses import replace
    u = np.array(as_float64_array(u, "u", grid.shape+(3,), check_finite=False), copy=True, order="C")
    digest = hashlib.sha256(memoryview(u).cast("B")).hexdigest()
    base = F.PrescreenFeasibilityResult("uncertain", None, None, None, 0, grid.n_cells, 0, 0, 0, None, None,
                                        "not_checked", digest, grid.grid_id, config.delta.hex(), config)
    base = replace(base, certificate_version=F.certificate_version(config))
    if not np.all(np.isfinite(u)):
        return replace(base, reason="NONFINITE_STATE")
    if not iv.arithmetic_supported():
        return replace(base, reason="unsupported_arithmetic")
    return F._certify_map_parallel(u, grid, config, base)


def cell_pass(u, grid, config):
    """Per-cell outcomes (int8 codes PROVEN_HULL ... UNCERTAIN_ARITHMETIC) and lower bounds (NaN where none) with
    the decisions of feasibility._certify_map_parallel (sequential; see the module docstring)."""
    from jd_curl.three_d.cell_q1 import CellQ1Operators3D
    from jd_curl.three_d.certificate_prescreen import prescreen_lower_bounds
    from jd_curl.three_d.compiled_enclosure import parallel_margins
    u = np.ascontiguousarray(u, dtype=np.float64)
    n = grid.n_cells
    code = np.full(n, PROVEN_HULL, dtype=np.int8)
    lower = np.full(n, np.nan)
    try:
        margin, flag = parallel_margins(u, grid, config.delta)
        all_unproven = False
    except FloatingPointError:
        margin = flag = None
        all_unproven = True
    jobs = []
    for tile in cell_tiles(grid, config.tile_size):
        count = tile.stop-tile.start
        if all_unproven or bool(flag[tile].any()):
            proven = np.zeros(count, dtype=bool)
        else:
            proven = margin[tile] > 0
            lower[tile][proven] = margin[tile][proven]
        local = np.flatnonzero(~proven)
        if len(local):
            jobs.append((tile, local))
    if isinstance(config, F.PrescreenCertificateConfig) and jobs:
        cells = np.concatenate([tile.start+local for tile, local in jobs])
        try:
            table = CellQ1Operators3D(grid, location="closed27_uniform").coefficients
        except FloatingPointError:
            table = None
        bound = prescreen_lower_bounds(u, grid, table, cells) if table is not None else np.full(len(cells), np.nan)
        with np.errstate(invalid="ignore"):
            margin_pre = bound if config.delta == 0 else np.nextafter(bound-config.delta, -np.inf)
            ok = margin_pre > 0
        code[cells[ok]] = PROVEN_PRESCREEN
        lower[cells[ok]] = margin_pre[ok]
        kept, start = [], 0
        for tile, local in jobs:
            stop = start+len(local)
            rest = local[~ok[start:stop]]
            if len(rest):
                kept.append((tile, rest))
            start = stop
        jobs = kept
    if isinstance(config, F.CompiledFallbackCertificateConfig):
        from jd_curl.three_d.compiled_fallback import fallback_arrays as arrays_fn
    else:
        arrays_fn = F._fallback_arrays
    errors = np.geterr()
    flat = u.reshape(-1, 3)
    for tile, local in jobs:
        ids = tile.start+local
        arrays = arrays_fn(flat[F._node_indices_for(grid, ids)], grid.spacing, config.delta, errors)
        if arrays is None:
            code[ids] = UNCERTAIN_ARITHMETIC
            continue
        p, coefficients, samples = arrays
        minima = coefficients.lo.min(axis=(-3, -2, -1))
        safe = minima > 0
        code[ids[safe]] = PROVEN_FALLBACK
        lower[ids[safe]] = minima[safe]
        for k in np.flatnonzero(~safe):
            cell = ids[k]
            try:
                result = F._classify(coefficients[k], p[k], config.delta, float(samples[k].min()), config)
            except FloatingPointError:
                code[cell] = UNCERTAIN_ARITHMETIC
                continue
            if result.min_margin_lower_bound is not None:
                lower[cell] = result.min_margin_lower_bound
            if result.reason == "arithmetic_uncertain":
                code[cell] = UNCERTAIN_ARITHMETIC
            elif result.status == "certified_feasible":
                code[cell] = PROVEN_FALLBACK
            elif result.status == "infeasible":
                code[cell] = INFEASIBLE
            else:
                code[cell] = UNCERTAIN
    return code, lower


def _region_counts(code, lower, mask):
    sub, low = code[mask], lower[mask]
    finite = low[np.isfinite(low)]
    return dict(cells=int(sub.size), proven=int(np.count_nonzero(sub <= PROVEN_FALLBACK)),
                infeasible=int(np.count_nonzero(sub == INFEASIBLE)),
                uncertain=int(np.count_nonzero(sub >= UNCERTAIN)),
                uncertain_arithmetic=int(np.count_nonzero(sub == UNCERTAIN_ARITHMETIC)),
                min_lower_bound=float(finite.min()) if finite.size else None)


def cell_summary(code, lower, result, brain_nodes=None):
    """Counts per region and the comparison with the map-level result."""
    regions = dict(all=_region_counts(code, lower, np.ones(code.size, dtype=bool)))
    if brain_nodes is not None:
        brain = np.asarray(brain_nodes, dtype=bool)[:-1, :-1, :-1].reshape(-1)
        regions.update(brain=_region_counts(code, lower, brain), background=_region_counts(code, lower, ~brain))
    reached = result.reason not in EARLY_REASONS
    mine = dict(n_infeasible=int(np.count_nonzero(code == INFEASIBLE)), n_uncertain=int(np.count_nonzero(code >= UNCERTAIN)),
                n_fast_cells=int(np.count_nonzero(code == PROVEN_HULL)),
                n_prescreen_cells=int(np.count_nonzero(code == PROVEN_PRESCREEN)),
                n_fallback_cells=int(np.count_nonzero(code >= PROVEN_FALLBACK)))
    finite = lower[np.isfinite(lower)]
    mine["min_margin_lower_bound"] = float(finite.min()) if finite.size else None
    mismatch = {}
    if reached:
        for key, value in mine.items():
            theirs = getattr(result, key, None)
            if key == "min_margin_lower_bound" and theirs is None:
                continue          # the map withholds its bound after an arithmetic failure
            if value != theirs:
                mismatch[key] = dict(cell_pass=value, map=theirs)
    return dict(regions=regions, cell_pass=mine, map_reached_cells=reached,
                matches_map=(not mismatch) if reached else None, mismatch=mismatch)


def _faces_max(u):
    return float(max(np.abs(u[[0, -1]]).max(), np.abs(u[:, [0, -1]]).max(), np.abs(u[:, :, [0, -1]]).max()))


def _value(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, tuple):
        return [_value(x) for x in v]
    return v


def certify_field(u, grid, *, mode, tile_size=CERT_TILE_SIZE, brain_nodes=None, split_cells=True):
    """Certificate record of the mm field u on `grid` in `mode` ("full" or "jacobian_only")."""
    if mode not in MODES:
        raise ValueError(f"mode: one of {MODES}")
    config = certificate_config(tile_size)
    check_certificate_config(config, tile_size)
    u = np.ascontiguousarray(as_float64_array(u, "u", grid.shape+(3,), check_finite=False))
    tick = time.perf_counter()
    result = F.certify_map(u, grid, config=config) if mode == "full" else certify_jacobian_only(u, grid, config)
    seconds = time.perf_counter()-tick
    record = {key: _value(getattr(result, key, None)) for key in RESULT_FIELDS}
    lowest = record["min_jd_lower_bound"]
    record.update(mode=mode, n_cells=int(grid.n_cells), seconds=seconds, config=certificate_record(config),
                  passed=bool(result.status == "certified_feasible" and lowest is not None and lowest > 0
                              and result.n_infeasible == 0 and result.n_uncertain == 0))
    finite = bool(np.all(np.isfinite(u)))
    record["faces_max_abs_u_mm"] = _faces_max(u) if finite else None
    if finite:
        domain = check_map_domain(u, grid)
        record["domain"] = dict(ok=domain is None, reason=None if domain is None else domain.reason,
                                count=None if domain is None else int(domain.invalid_count))
    else:
        record["domain"] = dict(ok=False, reason="NONFINITE_STATE", count=None)
    if split_cells and finite and iv.arithmetic_supported():
        tick = time.perf_counter()
        code, lower = cell_pass(u, grid, config)
        record["cells"] = dict(cell_summary(code, lower, result, brain_nodes), seconds=time.perf_counter()-tick)
    else:
        record["cells"] = None
    return record


__all__ = ["DEFINITIONS", "MODES", "cell_pass", "cell_summary", "certify_field", "certify_jacobian_only"]
