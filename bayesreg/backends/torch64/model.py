"""F_B, gradient, GN HVP, GN diagonal, c_dat and the C09.1 metric on the device in FP64 (work package G1).

Same names and semantics as bayesreg.model (contract C06, C09.1); the numpy functions stay the reference
(C17-1; reference commit db98ac2) and the L0 tests compare against them (gpu-port procedure L0: FP64, relative error
<= 1e-12; tests/test_bayesreg/torch64).  Scope of G1: FP64 only; no FP32 mode, no fused kernels; the
certificate is not touched (CPU FP64, C07).

Usage:
    dstate = TorchModelState(state)               # device constants of one numpy ModelState (cuda if available)
    ev = evaluate(dstate, u)                      # TorchEvaluation, or the numpy InvalidTrial
    g = gradient(dstate, ev); Hv = hvp(dstate, ev, v); d = gn_diagonal(dstate, ev); c = c_dat(dstate, ev)
    M = build_metric(dstate, c_dat=c, b_P=metric_bP(state)); M.solve(g)
u and v: free vector (3 N_int,) or (nz, ny, nx, 3) node field with zero faces, numpy or a float64 torch
tensor (moved to the device).  gradient, hvp, gn_diagonal, Metric.solve / matvec return device float64
tensors in the free-vector layout (`to_numpy` copies one to the host).

Failure semantics (as bayesreg.model.evaluate): nonfinite u, wrong shape or nonzero faces raise
InputValidationError; id + u out of the sampling domain returns InvalidTrial INVALID_DOMAIN (or
NONFINITE_STATE) with the node count of `check_map_domain` (same per-node rule, evaluated on the device);
a nonfinite spline output returns InvalidTrial NONFINITE_STATE with the mask-node count; a floating
J_q <= 0 returns InvalidTrial NONPOSITIVE_GAUSS_JD with the count of the first tile (state.tile_size
cells, ascending) that has one -- the count the reference reports; nonfinite geometry (F, A, J) in an
earlier tile raises FloatingPointError, as the reference does.

Where each step runs, and in what precision:
  device FP64   domain check, spline values and gradients, channel residuals, Q1 geometry (grad u_h, F,
                A = cof F, J, curl, tau_q^2), the three Gauss-point residual blocks and their squares;
                per-cell sums of r_i^2, then per-tile sums over the tiles of state.tile_size cells
                (padding + reshape + torch.sum: fixed order, no atomics); gradient, HVP, GN diagonal,
                K_Q, DST-I (torch.fft, complex128).
  host longdouble  (C06-7, C08-6) the channel blocks: r is downloaded and summed exactly as the reference
                (`bayesreg.model.energy._node_sum`, np.sum(dtype=longdouble) over the C-order (nz, ny, nx)
                array); the Gauss-point blocks: the downloaded per-tile FP64 sums accumulated in longdouble
                in ascending tile order, as the reference accumulates its per-tile longdouble sums; F_B =
                float(longdouble total of the parts in block order); c_dat: the downloaded (C, n, 3) terms
                summed by np.sum(dtype=longdouble).  The only difference from the reference is the
                per-tile partial sum (FP64 on the device instead of longdouble on the host).
  acceptance    (C08) not here: `TorchEvaluation.as_evaluation()` gives a numpy Evaluation (channel
                residuals downloaded) for the CPU `bayesreg.model.stable_reduction`, which recomputes the
                Gauss-point geometry of both states on the host.

Geometry cache.  HVP at a fixed ev reuses W_q = (s_J,q / J_q) A_q (the log-volume Jacobian row,
(3, c, 3, 27) per chunk), built at the first `hvp` call, chunk by chunk in ascending order while the
cache stays within `geometry_cache_bytes`; chunks beyond the budget recompute W from u.  Cached and
recomputed W are the same expression, so the HVP is bitwise the same either way (tested).  `evaluate`
and `gradient` never use or build the cache; `ev.release_geometry_cache()` frees it.

Cell corners (C05-2a, ModelState.logvol_points = "gauss27+corners"; work package LVC).  A second DeviceQ1 on the
corner operators (`q1_corners`, same chunks) gives the corner geometry; the corner part of the log-volume block,
r = sqrt(beta_c omega/(8 omega0 tau_{K,v}^2)) log J_{K,v} with tau_{K,v}^2 the gathered node value, is computed per
chunk after the Gauss points: evaluate sums it per cell (8 terms) into a fourth row of the per-cell sums, the tile
sums get a fourth column and the host adds the tile's corner sum to the log-volume accumulator right after the
tile's Gauss sum (the order of the reference); a J <= 0 at a corner counts with the Gauss points of the cell (the
reference's count and reason); gradient and HVP scatter the corner flux after the chunk's Gauss-point flux; the HVP
cache keeps the corner rows W_c = (s_c / J_c) A_c under the keys ("corners", chunk) within the same byte budget.
With "gauss27" (the default before DEF, an explicit option since) none of this runs and every value is bitwise that
of the code before C05-2a (481e33b).
`estimate_peak_bytes(..., corners=True)` counts the corner terms (FB1): the corner rows of the HVP cache (72 B x 8
per cell, within the same byte budget), the corner chunk temporaries (8/27 of the Gauss chunk's per cell, alive
together with them) and the fourth row of the per-cell sums with its corner J <= 0 counter.

Closed27 points (C05-2b, ModelState.logvol_points = "gauss27+closed27"; work package IMPL27).  The corner machinery
with the closed27 operators in their place (`dstate.logvol_extra` = "corners" | "closed27" | None; one code path,
`q1_extra`, `_extra_geometry`, `_extra_scale`): a DeviceQ1 on `geom.closed27` (`q1_closed27`, same chunks, 27 points);
tau_{K,p}^2 the Q1 interpolation of tau2_nodes at the closed27 points (`_closed27_tau2`, as at the Gauss points);
r = sqrt(omega s_p theta_S/(omega0 tau_{K,p}^2)) log J_{K,p} (weights `closed27_weights` = the ModelState's
closed27_point_weights); the Gauss part uses `logvol_gauss_weights` (omega w_g theta_G; under the other options the
tensor q1.weights itself, so their values are unchanged).  evaluate: per-cell sums of the 27 closed27 terms in the fourth
row, the host adds the tile's closed27 sum right after its Gauss sum; a J <= 0 at a closed27 point fails like the
reference (count and message); energies["logvol_parts"] as the reference (Gauss and closed27 parts from the host
longdouble sums of the device tile sums; the minimum J over the closed27 points: per chunk torch.argmin, the first
minimum, then the first chunk with the smallest value).  gradient / HVP scatter the closed27 flux after the chunk's
Gauss-point flux; the HVP cache keeps the closed27 rows under ("closed27", chunk); OPT1 C2a keeps the closed27 J (c, 27)
under ("closed27", chunk).  `estimate_peak_bytes(..., closed27=True)` counts 27 extra points per cell (cache rows,
chunk temporaries like a Gauss chunk, the fourth per-cell row, the Jacobian cache).

Jacobian cache and speculative gradient (work package OPT1; bitwise neutral, no contract change).
  C2a  `evaluate` keeps, per chunk, the J (c, 27) and log J it computed (and, under C05-2a, the corner J (c, 8) and
       log J) in the evaluation (`ev._jacobians`; log J only with jacobian_cache_log).  The C08-8 device comparison
       (stable.device_gauss_changes) takes J and log J of both states from there and recomputes only G = grad u_h
       (gather + DGEMM); the same tensors the old comparison recomputed with the same code, so every compared value is
       bitwise the same.  A chunk without a cache entry (cache released, evicted or off) is recomputed in full, as
       before.  Lifetime: the cache lives as long as its evaluation (freed when the evaluation is discarded) unless
       released (`ev.release_jacobian_cache()`) or evicted: at most `jacobian_cache_slots` evaluations per physical
       device (torch.device "cpu", "cuda:0", ...; shared by every TorchModelState and backend of the process, OPT1b)
       hold one; a new evaluation evicts the least recently used holder first (use = creation, `gradient`, comparison,
       the trial touched before the base), so the solver's base and newest trial keep theirs.  Memory per cached
       evaluation: 8 B x cells x (27 + 8 with corners): 1.46 GB at the native size (1.90 GB with corners), twice that
       with jacobian_cache_log (log J kept too; it saves only two torch.log per compared chunk; default off since OPT1b,
       review docs/bayesreg/reviews/OPT1_复核_20261030.md item 1).
  C3   `evaluate` also scatters the gradient flux of every chunk (channel term first, then per chunk the Gauss-point
       flux and the corner flux, the operations and order of `gradient`) into a node field kept in the evaluation
       (`ev._speculative`, 8 B x 3 N: 165 MB at the native size).  `gradient` with all blocks finishes that field (the
       face check, interior, + K_Q u, finiteness) instead of recomputing the geometry; any other block selection, or an
       evaluation without the field, computes as before.  The field goes with its evaluation; the solver backend
       (bayesreg.solve.Torch64Backend.stable_compare) also releases it as soon as a PG comparison rejects the candidate
       (OPT1b, review item 6; a released field is recomputed bitwise the same if ever needed).  The nonfinite checks of
       the gradient stay in `gradient` (evaluate raises nothing new).
  Values are unchanged; memory is not: the extra device memory can turn an evaluation that fitted before into an OOM,
  which C08-8(e) recomputes on the CPU (then the trajectory is no longer bitwise the device path's, and slower).
  `estimate_peak_bytes` at the native size, C = 4, mask 2.08e6 nodes, corners: 8.86 GB before OPT1, 12.98 GB with the
  defaults (two slots with J: +3.79 GB; two gradient fields: +0.33 GB), 16.78 GB with jacobian_cache_log=True; a PG
  solve holds no HVP cache, so its peak is lower by up to 4 GiB.  Options: TorchModelState(jacobian_cache_slots=0)
  switches C2a off, speculative_gradient=False C3 (run_pair --jacobian-cache, --jacobian-cache-log,
  --speculative-gradient).

Memory at the native size (160 x 192 x 224 nodes, N = 6.88e6, 6.77e6 cells, 1.83e8 Gauss points;
C = 4 channels; n = mask nodes, bounded here by N): see `estimate_peak_bytes`.  The full W cache would be
72 B x 1.83e8 = 13.2 GB, more than the 12 GB budget alone, so at native size only part of it is cached
(default 4 GiB, ~32 % of the chunks); one cached evaluation plus a second live evaluation and the chunk
temporaries stay below ~10.5 GB (worst case n = N).
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import inspect
import weakref

import numpy as np
import torch

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError, as_float64_array

from ...model import BLOCK_NAMES, ModelState
from ...model import build_metric as _numpy_build_metric
from ...model.blocks import ChannelPoint, channel_cdat_weight, nonpositive_extra_jd, nonpositive_gauss_jd
from ...model.energy import Evaluation, _node_sum, logvol_parts_record
from .device import Device
from .q1 import (CHUNK_CELLS, DeviceQ1, cofactor, curl_of, determinant_row0, flux_of, poisson_operator,
                 poisson_solve, stiffness_apply)
from .spline import SPLINE_CHUNK, TorchSpline, sample_channels, shared_knots

LD = np.longdouble
GEOMETRY_CACHE_BYTES = 4*2**30
JACOBIAN_CACHE_SLOTS = 2          # OPT1 C2a: evaluations per device holding a Jacobian cache (base + newest trial)
JACOBIAN_CACHE_LOG = False        # OPT1b (review item 1): keep J only; log J = torch.log(J) in the comparison
SPECULATIVE_GRADIENT = True       # OPT1 C3
GAUSS_NAMES = ("diffusion", "logvol", "curl")


def to_numpy(tensor):
    """A host float64 copy of a device tensor."""
    return tensor.detach().cpu().numpy().copy()


class JacobianSlots:
    """OPT1 C2a: the live evaluations of one physical device that hold a Jacobian cache, least recently used first."""

    def __init__(self):
        self._order = OrderedDict()            # id(ev) -> weakref.ref(ev)

    def _prune(self):
        for key in [k for k, ref in self._order.items() if (ev := ref()) is None or not ev._jacobians]:
            del self._order[key]

    def holders(self):
        self._prune()
        return [ref() for ref in self._order.values()]

    def touch(self, ev):
        key = id(ev)
        if key in self._order and self._order[key]() is ev:
            self._order.move_to_end(key)

    def acquire(self, slots):
        """Room for one new cache: evict least recently used holders until fewer than `slots` remain.  False when
        slots == 0 (no cache)."""
        self._prune()
        if slots <= 0:
            return False
        while len(self._order) >= slots:
            _, ref = self._order.popitem(last=False)
            ev = ref()
            if ev is not None:
                ev._jacobians.clear()
        return True

    def register(self, ev):
        if ev._jacobians:
            self._order.pop(id(ev), None)
            self._order[id(ev)] = weakref.ref(ev)

    def release_all(self):
        """Drop every Jacobian cache on this device (later comparisons recompute, bitwise the same)."""
        self._prune()
        while self._order:
            _, ref = self._order.popitem(last=False)
            ev = ref()
            if ev is not None:
                ev._jacobians.clear()


_JACOBIAN_SLOTS = {}


def jacobian_slots(device):
    """The JacobianSlots of one physical device: keyed by the torch.device of `device` ("cpu", "cuda:0", ...; Device
    resolves "cuda" to its index), so every TorchModelState and every backend of the process on that device share one
    bound (OPT1b, review item 7; before, each Device object had its own and two backends doubled the bound).  The slots
    hold weak references only."""
    return _JACOBIAN_SLOTS.setdefault(str(device.device), JacobianSlots())


# ----------------------------------------------------------------------------------------------
# Device state
# ----------------------------------------------------------------------------------------------

class TorchModelState:
    """Device constants of exactly one numpy ModelState (sigma, w, eta frozen): mask-node arrays, the
    moving-side splines, tau^2 at the nodes, the Gauss27 operators and the block weights.
    OPT1 options (bitwise neutral; only device memory and speed): jacobian_cache_slots (evaluations per physical device
    holding a Jacobian cache, 0 = off), jacobian_cache_log (also keep log J; default off since OPT1b),
    speculative_gradient (evaluate scatters the gradient flux)."""

    def __init__(self, state, device=None, *, chunk_cells=CHUNK_CELLS, spline_chunk=SPLINE_CHUNK,
                 geometry_cache_bytes=GEOMETRY_CACHE_BYTES, jacobian_cache_slots=JACOBIAN_CACHE_SLOTS,
                 jacobian_cache_log=JACOBIAN_CACHE_LOG, speculative_gradient=SPECULATIVE_GRADIENT, splines=None):
        if not isinstance(state, ModelState):
            raise InputValidationError("state: ModelState required")
        for name, value, low in (("chunk_cells", chunk_cells, 1), ("spline_chunk", spline_chunk, 1),
                                 ("geometry_cache_bytes", geometry_cache_bytes, 0),
                                 ("jacobian_cache_slots", jacobian_cache_slots, 0)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
                raise InputValidationError(f"{name}: integer >= {low} required")
        for name, value in (("jacobian_cache_log", jacobian_cache_log), ("speculative_gradient", speculative_gradient)):
            if not isinstance(value, bool):
                raise InputValidationError(f"{name}: bool required")
        self.state = state
        self.device = device if isinstance(device, Device) else Device(device)
        self.chunk_cells = int(chunk_cells)
        self.spline_chunk = int(spline_chunk)
        self.geometry_cache_bytes = int(geometry_cache_bytes)
        self.jacobian_cache_slots = int(jacobian_cache_slots)
        self.jacobian_cache_log = jacobian_cache_log
        self.speculative_gradient = speculative_gradient
        self.jacobian_slots = jacobian_slots(self.device)
        grid = state.grid
        up = self.device.upload
        self.q1 = DeviceQ1(grid, state.ops, self.device, self.chunk_cells)
        self.logvol_corners = state.logvol_corners
        self.logvol_closed27 = state.logvol_closed27
        self.logvol_extra = state.logvol_extra                           # None | "corners" | "closed27"
        if self.logvol_corners:                                          # C05-2a
            self.q1_corners = DeviceQ1(grid, state.corner_ops, self.device, self.chunk_cells)
            self.corner_weights = up(state.corner_weights)               # (8,) beta_c omega/8
        else:
            self.q1_corners = self.corner_weights = None
        if self.logvol_closed27:                                         # C05-2b
            self.q1_closed27 = DeviceQ1(grid, state.closed27_ops, self.device, self.chunk_cells)
            self.closed27_weights = up(state.closed27_point_weights)     # (27,) omega s_p theta_S
            self.logvol_gauss_weights = up(state.logvol_gauss_weights)   # (27,) omega w_g theta_G
        else:
            self.q1_closed27 = self.closed27_weights = None
            self.logvol_gauss_weights = self.q1.weights                  # the same tensor: values unchanged
        self.q1_extra = self.q1_corners if self.logvol_corners else self.q1_closed27
        self.node_index = self.device.upload_index(state.node_index)
        self.strict = self.device.upload_bool(state.strict)
        self.points0 = up(state.points0)
        self.fixed_masked = up(state.fixed_masked)
        self.beta = up(state.beta)
        cdat_weight = channel_cdat_weight(state)      # the reference's expression (times v under sigma_domain support)
        self.cdat_weight = up(cdat_weight)
        self.tau2_nodes = up(state.tau2_nodes)[None]
        if splines is None:
            splines = tuple(TorchSpline(s, up) for s in state.samplers)
        else:
            splines = tuple(splines)
            if (len(splines) != len(state.samplers) or any(
                    not isinstance(t, TorchSpline) or t.image_id != s.image_id or t.grid_id != s.grid.grid_id
                    or t._c.device != self.device.device for t, s in zip(splines, state.samplers))):
                raise InputValidationError("splines: the device splines of these samplers on this device required")
        self.splines = splines
        self.shared_knots = shared_knots(splines)
        self.axes = tuple(up(a) for a in grid.axes)
        self.bounds = tuple((float(a[0]), float(a[-1])) for a in grid.axes)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            w = state.ops.weights
            self.scale_diffusion = up(np.sqrt(state.alpha*w/state.omega0))      # Diffusion.scale
            self.scale_curl = up(np.sqrt(state.nu*w/state.omega0))              # Curl.scale
        self.n_tiles = -(-grid.n_cells//state.tile_size)

    @property
    def grid(self):
        return self.state.grid

    @property
    def n_channels(self):
        return self.state.n_channels

    def with_state(self, state, **options):
        """The next (level, EM round) on the same grid and samplers: reuse the uploaded splines."""
        if (not isinstance(state, ModelState) or len(state.samplers) != len(self.state.samplers)
                or any(a is not b for a, b in zip(state.samplers, self.state.samplers))):
            raise InputValidationError("with_state: a ModelState with the same samplers required")
        kwargs = dict(chunk_cells=self.chunk_cells, spline_chunk=self.spline_chunk,
                      geometry_cache_bytes=self.geometry_cache_bytes, jacobian_cache_slots=self.jacobian_cache_slots,
                      jacobian_cache_log=self.jacobian_cache_log, speculative_gradient=self.speculative_gradient)
        kwargs.update(options)
        return TorchModelState(state, self.device, splines=self.splines, **kwargs)

    def provenance(self):
        return dict(backend="bayesreg.backends.torch64", precision="float64", **self.device.describe(),
                    chunk_cells=self.chunk_cells, slabs_per_chunk=self.q1.slabs_per_chunk,
                    n_chunks=len(self.q1.chunks), spline_chunk=self.spline_chunk,
                    geometry_cache_bytes=self.geometry_cache_bytes, jacobian_cache_slots=self.jacobian_cache_slots,
                    jacobian_cache_log=self.jacobian_cache_log, speculative_gradient=self.speculative_gradient,
                    tile_size=self.state.tile_size,
                    reductions=("channel: host np.sum(longdouble) over the C-order node array; Gauss blocks: "
                                "device FP64 per-tile sums, host longdouble over tiles in ascending order"),
                    **self.state.logvol_record(), **self.state.sigma_domain_record())


def resolved_state_options(options=None):
    """Every keyword option of TorchModelState (chunk_cells, spline_chunk, geometry_cache_bytes, jacobian_cache_slots,
    jacobian_cache_log, speculative_gradient) with its default, overridden by `options`: the values a TorchModelState
    built with these options uses (OPT1b, review item 2: recorded by the solver backend before any state exists).
    Unknown names are refused; the values are validated by TorchModelState."""
    parameters = inspect.signature(TorchModelState.__init__).parameters
    out = {name: p.default for name, p in parameters.items() if p.kind is p.KEYWORD_ONLY and name != "splines"}
    options = dict(options or {})
    unknown = sorted(set(options)-set(out))
    if unknown:
        raise InputValidationError(f"state options: unknown {unknown} (known: {sorted(out)})")
    out.update(options)
    return out


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------

def _as_tensor(value, name, dstate):
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float64 or value.requires_grad:
            raise InputValidationError(f"{name}: a float64 tensor without autograd required")
        return value.detach().to(dstate.device.device)
    array = np.asarray(value)
    if array.ndim not in (1, 4):
        raise InputValidationError(f"{name}: a free vector (3 N_int,) or a (nz, ny, nx, 3) node field required")
    shape = (3*dstate.grid.n_free_scalar,) if array.ndim == 1 else dstate.grid.shape+(3,)
    return torch.from_numpy(np.ascontiguousarray(as_float64_array(array, name, shape))).to(dstate.device.device)


def node_field_cm(value, dstate, *, name="u"):
    """Layer-1 validation of `geom.node_field` on the device: a free vector is injected with zero faces; a
    node field must be finite with exactly zero faces.  Returns a new (3, nz, ny, nx) float64 tensor."""
    grid = dstate.grid
    x = _as_tensor(value, name, dstate)
    if x.ndim == 1:
        if tuple(x.shape) != (3*grid.n_free_scalar,):
            raise InputValidationError(f"{name}: expected shape {(3*grid.n_free_scalar,)}, got {tuple(x.shape)}")
        if not bool(torch.isfinite(x).all()):
            raise InputValidationError(f"{name}: all values must be finite float64")
        out = torch.zeros((3,)+grid.shape, dtype=torch.float64, device=x.device)
        out[:, 1:-1, 1:-1, 1:-1] = x.reshape((3,)+grid.interior_shape)
        return out
    if tuple(x.shape) != grid.shape+(3,):
        raise InputValidationError(f"{name}: expected shape {grid.shape+(3,)}, got {tuple(x.shape)}")
    if not bool(torch.isfinite(x).all()):
        raise InputValidationError(f"{name}: all values must be finite float64")
    out = x.permute(3, 0, 1, 2).contiguous()
    faces = torch.cat([out[:, [0, -1]].reshape(-1), out[:, :, [0, -1]].reshape(-1), out[..., [0, -1]].reshape(-1)])
    if bool((faces != 0).any()):
        raise InputValidationError(f"{name}: all three components on all six faces must be exactly zero (U0); "
                                   "pass a free vector or an inject_free field")
    return out


def domain_check(dstate, u_cm):
    """`interpolation.check_map_domain` on the device (same per-node rule and counts) -> InvalidTrial or None."""
    nonfinite = ~torch.isfinite(u_cm).all(dim=0)
    count = int(nonfinite.sum())
    if count:
        return InvalidTrial("NONFINITE_STATE", "u contains nonfinite coordinates", count)
    bad = torch.zeros(dstate.grid.shape, dtype=torch.bool, device=u_cm.device)
    for j, (coordinates, (lo, hi)) in enumerate(zip(dstate.axes, dstate.bounds)):
        shape = [1, 1, 1]
        shape[2-j] = coordinates.numel()
        value = u_cm[j]+coordinates.reshape(shape)
        bad |= ~torch.isfinite(value) | (value < lo) | (value > hi)
        free = value[1:-1, 1:-1, 1:-1]
        bad[1:-1, 1:-1, 1:-1] |= (free <= lo) | (free >= hi)
    count = int(bad.sum())
    if count:
        return InvalidTrial("INVALID_DOMAIN", "id+u violates closed boundary / open free-node support", count)
    return None


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TorchChannelPoint:
    """The channel block at one u on the mask nodes, device tensors: rho, r (C, n), grad (C, n, 3)."""
    rho: torch.Tensor
    grad: torch.Tensor
    r: torch.Tensor


@dataclass(frozen=True, eq=False)
class TorchEvaluation:
    """F_B at one u of one TorchModelState.  u_cm: (3, nz, ny, nx) device node field in U0."""
    state: TorchModelState
    u_cm: torch.Tensor
    channel: TorchChannelPoint
    total: np.longdouble
    energy: float
    energies: dict
    r_host: np.ndarray = field(repr=False)            # (C, n) downloaded channel residuals (read-only)
    tile_sums: np.ndarray = field(repr=False)         # (n_tiles, 3) device FP64 per-tile sums of r_i^2 (C05-2a:
                                                      # a 4th column, the log-volume corner part)
    _cache: dict = field(default_factory=dict, repr=False)
    _jacobians: dict = field(default_factory=dict, repr=False)     # OPT1 C2a: owner stamp, entries chunk -> (J,
                                                                   # log J or None), corners under ("corners", chunk)
    _speculative: dict = field(default_factory=dict, repr=False)   # OPT1 C3: owner stamp, "out" the gradient node field

    @property
    def u(self):
        """(nz, ny, nx, 3) view of u_cm (the reference layout)."""
        return self.u_cm.permute(1, 2, 3, 0)

    def u_host(self):
        out = np.ascontiguousarray(to_numpy(self.u))
        out.flags.writeable = False
        return out

    def as_evaluation(self):
        """A numpy bayesreg.model Evaluation with this evaluation's channel point (downloaded), total and
        energies, for the CPU stable comparison (C08).  The CPU recomputes the Gauss-point geometry there."""
        def ro(array):
            array = np.ascontiguousarray(array, dtype=np.float64)
            array.flags.writeable = False
            return array
        point = ChannelPoint(ro(to_numpy(self.channel.rho)), ro(to_numpy(self.channel.grad)), ro(self.r_host))
        return Evaluation(self.state.state, self.u_host(), point, self.total, self.energy, dict(self.energies))

    def release_geometry_cache(self):
        self._cache.clear()

    @property
    def geometry_cache_nbytes(self):
        return sum(t.numel()*t.element_size() for t in self._cache.values())

    def release_jacobian_cache(self):
        """OPT1 C2a: drop the cached J / log J (a later comparison recomputes them, bitwise the same)."""
        self._jacobians.clear()

    @property
    def jacobian_cache_nbytes(self):
        return sum(t.numel()*t.element_size() for pair in self._jacobians.get("entries", {}).values()
                   for t in pair if t is not None)

    def jacobian_entry(self, key):
        """OPT1 C2a: the cached (J, log J or None) of chunk `key` (("corners", chunk) for the corners), or None when
        there is none or the cache was not computed from this evaluation's dstate and u_cm (e.g. an evaluation made
        by dataclasses.replace, or u_cm changed in place since)."""
        cache = self._jacobians
        if not cache or not _stamp_matches(cache["owner"], self):
            return None
        return cache["entries"].get(key)

    def release_speculative_gradient(self):
        """OPT1 C3: drop the scattered gradient field (a later `gradient` recomputes it, bitwise the same)."""
        self._speculative.clear()

    def speculative_field(self):
        """OPT1 C3: the gradient node field scattered by `evaluate`, or None (none, or not computed from this
        evaluation's dstate, u_cm and channel residuals and gradients)."""
        spec = self._speculative
        if not spec or not _stamp_matches(spec["owner"], self, channel=True):
            return None
        return spec["out"]


def _stamp(dstate, u_cm, point=None):
    """What a cached quantity was computed from: dstate, u_cm (identity and in-place version counter) and, for the
    gradient, the channel point with the version counters of r and grad."""
    stamp = (dstate, u_cm, u_cm._version)
    if point is not None:
        stamp += (point, point.r._version, point.grad._version)
    return stamp


def _stamp_matches(stamp, ev, channel=False):
    dstate, u_cm, version = stamp[:3]
    if ev.state is not dstate or ev.u_cm is not u_cm or u_cm._version != version:
        return False
    if channel:
        point, r_version, grad_version = stamp[3:]
        return ev.channel is point and point.r._version == r_version and point.grad._version == grad_version
    return True


def _channel_point(dstate, u_cm):
    invalid = domain_check(dstate, u_cm)
    if invalid is not None:
        return invalid
    displacement = u_cm.reshape(3, -1)[:, dstate.node_index].T
    points = dstate.points0+displacement                      # = state.points0 + u.reshape(-1, 3)[node_index]
    values, grads = sample_channels(dstate.splines, points, chunk_size=dstate.spline_chunk, strict=dstate.strict,
                                    shared=dstate.shared_knots)
    finite = torch.isfinite(values).all(dim=1) & torch.isfinite(grads).reshape(grads.shape[0], -1).all(dim=1)
    if not bool(finite.all()):
        return InvalidTrial("NONFINITE_STATE", "nonfinite spline output", int(points.shape[0]))
    rho = values-dstate.fixed_masked
    r = dstate.beta*rho
    if not bool(torch.isfinite(r).all()):
        raise FloatingPointError("channel residual: nonfinite numerical result")
    return TorchChannelPoint(rho, grads, r)


def _geometry(dstate, u_cm, chunk, q1=None):
    """grad u_h (3, c, 3, 27), F, A = cof F, J (c, 27) on one chunk (C02-2, as geom.tile_geometry); with
    q1 = dstate.q1_corners the same at the eight cell corners, (..., 8) (C05-2a)."""
    G = (dstate.q1 if q1 is None else q1).gradient(u_cm, chunk)
    F = G.clone()
    for r in range(3):
        F[r, :, r] += 1.
    A = cofactor(F)
    J = determinant_row0(F, A)
    return G, F, A, J


def _corner_geometry(dstate, u_cm, chunk):
    """C05-2a: _geometry at the cell corners (each cell's own Q1 map), (3, c, 3, 8) and J (c, 8)."""
    return _geometry(dstate, u_cm, chunk, dstate.q1_corners)


def _logvol_scale(dstate, chunk):
    """LogVolume.scale: sqrt(omega w_g / (omega0 tau_q^2)) on the chunk, (c, 27) (C05-2b: omega w_g theta_G;
    dstate.logvol_gauss_weights is the tensor q1.weights itself under the other options)."""
    tau2 = dstate.q1.interpolate(dstate.tau2_nodes, chunk)
    return torch.sqrt(dstate.logvol_gauss_weights/(dstate.state.omega0*tau2))


def _corner_tau2(dstate, chunk):
    """C05-2a: tau_{K,v}^2, the node values of tau2_nodes at the eight corners of every cell of the chunk, (c, 8)."""
    return dstate.q1_corners.gather(dstate.tau2_nodes, chunk).reshape(-1, 8)


def _corner_scale(dstate, chunk):
    """LogVolumeCorners.scale (C05-2a): sqrt(beta_c omega/8 / (omega0 tau_{K,v}^2)) on the chunk, (c, 8)."""
    return torch.sqrt(dstate.corner_weights/(dstate.state.omega0*_corner_tau2(dstate, chunk)))


def _closed27_tau2(dstate, chunk):
    """C05-2b: tau_{K,p}^2, the Q1 interpolation of tau2_nodes at the closed27 points of every cell of the chunk,
    (c, 27) (LogVolumeClosed27.tau2)."""
    return dstate.q1_closed27.interpolate(dstate.tau2_nodes, chunk)


def _closed27_scale(dstate, chunk):
    """LogVolumeClosed27.scale (C05-2b): sqrt(omega s_p theta_S / (omega0 tau_{K,p}^2)) on the chunk, (c, 27)."""
    return torch.sqrt(dstate.closed27_weights/(dstate.state.omega0*_closed27_tau2(dstate, chunk)))


def _extra_geometry(dstate, u_cm, chunk):
    """_geometry at the points of the second log-volume part (C05-2a corners / C05-2b closed27 points)."""
    return _geometry(dstate, u_cm, chunk, dstate.q1_extra)


def _extra_tau2(dstate, chunk):
    return _corner_tau2(dstate, chunk) if dstate.logvol_corners else _closed27_tau2(dstate, chunk)


def _extra_scale(dstate, chunk):
    return _corner_scale(dstate, chunk) if dstate.logvol_corners else _closed27_scale(dstate, chunk)


def _extra_weights(dstate):
    """The omega-scaled point weights of the second part: beta_c omega/8 (corners) or omega s_p theta_S (closed27)."""
    return dstate.corner_weights if dstate.logvol_corners else dstate.closed27_weights


def _tile_reduce(per_cell, tile_size, n_tiles, n_cells):
    """(..., n_cells) -> (..., n_tiles): sums over consecutive tiles of tile_size cells (zero padding)."""
    pad = n_tiles*tile_size-n_cells
    if pad:
        per_cell = torch.nn.functional.pad(per_cell, (0, pad))
    return per_cell.reshape(per_cell.shape[:-1]+(n_tiles, tile_size)).sum(dim=-1)


def evaluate(dstate, u):
    """C06-1/7 on the device: TorchEvaluation of F_B at u, or an InvalidTrial (see the module docstring)."""
    if not isinstance(dstate, TorchModelState):
        raise InputValidationError("dstate: TorchModelState required")
    state = dstate.state
    u_cm = node_field_cm(u, dstate)
    point = _channel_point(dstate, u_cm)
    if isinstance(point, InvalidTrial):
        return point
    q1 = dstate.q1
    n_cells = state.grid.n_cells
    dev = u_cm.device
    extra = dstate.logvol_extra                                     # C05-2a "corners" / C05-2b "closed27" / None
    closed27 = dstate.logvol_closed27
    cell_sums = torch.empty((4 if extra else 3, n_cells), dtype=torch.float64, device=dev)
    nonpositive = torch.empty(n_cells, dtype=torch.int64, device=dev)
    nonfinite = torch.empty(n_cells, dtype=torch.bool, device=dev)
    if extra:
        nonpositive_extra = torch.empty(n_cells, dtype=torch.int64, device=dev)
    chunk_minima, chunk_argmins = [], []                            # C05-2b record (first minimum per chunk)
    sD, sC = dstate.scale_diffusion[None, None, None, :], dstate.scale_curl
    slots = dstate.jacobian_slots
    jacobians = {} if slots.acquire(dstate.jacobian_cache_slots) else None     # OPT1 C2a
    keep_log = dstate.jacobian_cache_log
    out = None
    if dstate.speculative_gradient:                                 # OPT1 C3: gradient's order, channel term first
        out = torch.zeros((3,)+state.grid.shape, dtype=torch.float64, device=dev)
        _channel_add(dstate, out, dstate.beta*point.r, point.grad)
    for number, chunk in enumerate(q1.chunks):
        lo, hi = q1.cell_range(chunk)
        G, F, A, J = _geometry(dstate, u_cm, chunk)
        del F
        bad = ~torch.isfinite(J)
        for value in (G, A):
            bad |= ~torch.isfinite(value).all(dim=(0, 2))
        nonfinite[lo:hi] = bad.any(dim=1)
        nonpositive[lo:hi] = (J <= 0).sum(dim=1)
        rD = sD*G                                                   # Diffusion.residual
        cell_sums[0, lo:hi] = (rD*rD).sum(dim=(0, 2)).sum(dim=1)
        del rD
        sJ = _logvol_scale(dstate, chunk)
        logJ = torch.log(J)
        rJ = sJ*logJ                                                # LogVolume.residual (J > 0 where used)
        cell_sums[1, lo:hi] = (rJ*rJ).sum(dim=1)
        curl, rC = 0., []
        for component in curl_of(G):                                # Curl.residual, sum over (point, i)
            rC.append(sC*component)
            curl = curl+rC[-1]*rC[-1]
        cell_sums[2, lo:hi] = curl.sum(dim=1)
        if out is not None:                                         # OPT1 C3: gradient's Gauss-point flux
            gJ = sJ*rJ/J                                            # LogVolume.cotangent(residual)
            gcurl = tuple(sC*r for r in rC)                         # Curl.cotangent(residual)
            q1.scatter_add(out, flux_of(A, gJ, gcurl, G), chunk)
            del gJ, gcurl
        if jacobians is not None:
            jacobians[number] = (J, logJ if keep_log else None)
        del G, A, J, logJ, rJ, rC, sJ
        if extra:                                                   # C05-2a / C05-2b: the second log-volume part
            Gc, Fc, Ac, Jc = _extra_geometry(dstate, u_cm, chunk)
            del Fc
            bad = ~torch.isfinite(Jc)
            for value in (Gc, Ac):
                bad |= ~torch.isfinite(value).all(dim=(0, 2))
            nonfinite[lo:hi] |= bad.any(dim=1)
            nonpositive_extra[lo:hi] = (Jc <= 0).sum(dim=1)
            sJc = _extra_scale(dstate, chunk)
            logJc = torch.log(Jc)
            rJc = sJc*logJc
            cell_sums[3, lo:hi] = (rJc*rJc).sum(dim=1)
            if out is not None:                                     # OPT1 C3: after the chunk's Gauss points
                gJc = sJc*rJc/Jc                                    # LogVolumeCorners / Closed27 .cotangent(residual)
                dstate.q1_extra.scatter_add(out, flux_of(Ac, gJc, None, Gc), chunk)
                del gJc
            if jacobians is not None:
                jacobians[(extra, number)] = (Jc, logJc if keep_log else None)
            if closed27:                                            # C05-2b record: the chunk's first minimum
                flat = Jc.reshape(-1)
                k = torch.argmin(flat)
                chunk_minima.append(flat[k])
                chunk_argmins.append(k+lo*Jc.shape[1])
            del Gc, Ac, Jc, logJc, rJc, sJc
    tiles, n_tiles = state.tile_size, dstate.n_tiles
    tile_sums = dstate.device.download(_tile_reduce(cell_sums, tiles, n_tiles, n_cells)).T   # (n_tiles, 3 or 4)
    tile_nonpositive = dstate.device.download(_tile_reduce(nonpositive, tiles, n_tiles, n_cells))
    tile_nonfinite = dstate.device.download(_tile_reduce(nonfinite.to(torch.int64), tiles, n_tiles, n_cells))
    if extra:
        tile_nonpositive_extra = dstate.device.download(_tile_reduce(nonpositive_extra, tiles, n_tiles, n_cells))
    for t in range(n_tiles):
        if tile_nonfinite[t]:
            raise FloatingPointError("Gauss-point geometry: nonfinite numerical result (tile %d)" % t)
        if not extra:
            if tile_nonpositive[t]:
                return nonpositive_gauss_jd(int(tile_nonpositive[t]))
        elif tile_nonpositive[t] or tile_nonpositive_extra[t]:
            return nonpositive_extra_jd(state, int(tile_nonpositive[t]), int(tile_nonpositive_extra[t]))
    r_host = dstate.device.download(point.r)
    r_host.flags.writeable = False
    with np.errstate(over="raise", invalid="raise"):
        channel = [_node_sum(state, r_host[c]*r_host[c]) for c in range(state.n_channels)]
        gauss = [LD(0) for _ in GAUSS_NAMES]
        part_gauss, part_closed27 = LD(0), LD(0)
        for t in range(n_tiles):                                     # C06-7: ascending tiles
            for k in range(3):
                gauss[k] += LD(tile_sums[t, k])
            if extra:
                gauss[1] += LD(tile_sums[t, 3])                      # C05-2a / C05-2b: right after the Gauss sum
            if closed27:
                part_gauss += LD(tile_sums[t, 1])
                part_closed27 += LD(tile_sums[t, 3])
        half = LD(.5)
        parts = [half*s for s in channel]+[half*s for s in gauss]
        total = LD(0)
        for part in parts:
            total += part
    energy = float(total)
    if not np.isfinite(energy) or not np.all(np.isfinite(tile_sums)):
        raise FloatingPointError("F_B: nonfinite numerical result")
    energies = dict(channel=tuple(float(p) for p in parts[:state.n_channels]),
                    **{name: float(p) for name, p in zip(GAUSS_NAMES, parts[state.n_channels:])})
    if closed27:
        minima = dstate.device.download(torch.stack(chunk_minima))
        argmins = dstate.device.download(torch.stack(chunk_argmins))
        best = 0
        for k in range(1, len(minima)):                             # the first chunk with the smallest value
            if minima[k] < minima[best]:
                best = k
        flat, n_points = int(argmins[best]), len(state.closed27_ops.points)
        energies["logvol_parts"] = logvol_parts_record(state, float(half*part_gauss), float(half*part_closed27),
                                                       float(minima[best]), flat//n_points, flat % n_points)
    tile_sums = np.ascontiguousarray(tile_sums)
    tile_sums.flags.writeable = False
    ev = TorchEvaluation(dstate, u_cm, point, total, energy, energies, r_host, tile_sums,
                         _jacobians={} if jacobians is None else dict(owner=_stamp(dstate, u_cm), entries=jacobians),
                         _speculative={} if out is None else dict(owner=_stamp(dstate, u_cm, point), out=out))
    slots.register(ev)
    return ev


# ----------------------------------------------------------------------------------------------
# Gradient, HVP, GN diagonal, c_dat
# ----------------------------------------------------------------------------------------------

def _check(dstate, ev):
    if not isinstance(dstate, TorchModelState):
        raise InputValidationError("dstate: TorchModelState required")
    if not isinstance(ev, TorchEvaluation) or ev.state is not dstate:
        raise InputValidationError("evaluation: a TorchEvaluation of this TorchModelState required "
                                   "(sigma, w, eta frozen)")


def _names(blocks):
    names = tuple(blocks)
    if not names or any(name not in BLOCK_NAMES for name in names) or len(set(names)) != len(names):
        raise InputValidationError(f"blocks: a nonempty subset of {BLOCK_NAMES} required")
    return names


def _channel_add(dstate, out, coefficient, grad):
    """out[:, a] += sum_c coefficient[c, a] grad[c, a, :] at the mask nodes (channels ascending)."""
    total = coefficient[0][:, None]*grad[0]
    for c in range(1, coefficient.shape[0]):
        total = total+coefficient[c][:, None]*grad[c]
    out.reshape(3, -1).index_add_(1, dstate.node_index, total.T)      # unique indices: one add per node


def _finish(out, name):
    if not bool(torch.isfinite(out).all()):
        raise FloatingPointError(f"{name}: nonfinite numerical result")
    return out[:, 1:-1, 1:-1, 1:-1].reshape(-1)


def _require_finite(x, name):
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(f"{name}: nonfinite numerical result")
    return x


def gradient(dstate, ev, *, blocks=BLOCK_NAMES):
    """C06-4: grad F_B = sum_i K_i^T r_i at ev.u as the device free vector; `blocks` selects terms.  With all blocks
    and the field scattered by `evaluate` (OPT1 C3) only the finish runs (face check, interior, + K_Q u)."""
    _check(dstate, ev)
    names = _names(blocks)
    state = dstate.state
    dstate.jacobian_slots.touch(ev)
    out = ev.speculative_field() if set(names) == set(BLOCK_NAMES) else None
    if out is None:
        out = torch.zeros((3,)+state.grid.shape, dtype=torch.float64, device=ev.u_cm.device)
        if "channel" in names:
            _channel_add(dstate, out, dstate.beta*ev.channel.r, ev.channel.grad)
        logvol, curl = "logvol" in names, "curl" in names
        if logvol or curl:
            sC = dstate.scale_curl
            for chunk in dstate.q1.chunks:
                G, F, A, J = _geometry(dstate, ev.u_cm, chunk)
                gJ = gcurl = None
                if logvol:
                    sJ = _logvol_scale(dstate, chunk)
                    gJ = sJ*(sJ*torch.log(J))/J                     # LogVolume.cotangent(residual)
                if curl:
                    gcurl = tuple(sC*(sC*component) for component in curl_of(G))     # Curl.cotangent(residual)
                dstate.q1.scatter_add(out, flux_of(A, gJ, gcurl, G), chunk)
                if logvol and dstate.logvol_extra:                  # C05-2a / C05-2b, after the chunk's Gauss points
                    Gc, _, Ac, Jc = _extra_geometry(dstate, ev.u_cm, chunk)
                    sJc = _extra_scale(dstate, chunk)
                    gJc = sJc*(sJc*torch.log(Jc))/Jc                 # LogVolumeCorners / Closed27 .cotangent(residual)
                    dstate.q1_extra.scatter_add(out, flux_of(Ac, gJc, None, Gc), chunk)
    g = _finish(out, "gradient of F_B")
    if "diffusion" in names:
        g = g+(state.alpha/state.omega0)*stiffness_apply(ev.u_cm, state.grid)
    return _require_finite(g, "gradient of F_B")


def _logvol_row(dstate, ev, chunk, number):
    """W = (s_J / J) A on one chunk: from the geometry cache, else computed (and cached within budget)."""
    W = ev._cache.get(number)
    if W is not None:
        return W
    _, _, A, J = _geometry(dstate, ev.u_cm, chunk)
    W = A*(_logvol_scale(dstate, chunk)/J)[None, :, None, :]
    if ev.geometry_cache_nbytes+W.numel()*W.element_size() <= dstate.geometry_cache_bytes:
        ev._cache[number] = W
    return W


def _extra_row(dstate, ev, chunk, number):
    """C05-2a / C05-2b: W_x = (s_x / J_x) A_x at the points of the second log-volume part of one chunk, (3, c, 3, 8)
    for the corners, (3, c, 3, 27) for the closed27 points; cached under ("corners" | "closed27", number) within the
    same byte budget as the Gauss rows."""
    key = (dstate.logvol_extra, number)
    W = ev._cache.get(key)
    if W is not None:
        return W
    _, _, A, J = _extra_geometry(dstate, ev.u_cm, chunk)
    W = A*(_extra_scale(dstate, chunk)/J)[None, :, None, :]
    if ev.geometry_cache_nbytes+W.numel()*W.element_size() <= dstate.geometry_cache_bytes:
        ev._cache[key] = W
    return W


def hvp(dstate, ev, v, *, blocks=BLOCK_NAMES):
    """C06-5: H v = sum_i K_i^T (K_i v) at ev.u (GN), v a free vector or U0 node field; device free vector."""
    _check(dstate, ev)
    names = _names(blocks)
    state = dstate.state
    v_cm = node_field_cm(v, dstate, name="v")
    out = torch.zeros_like(v_cm)
    if "channel" in names:
        g = ev.channel.grad
        vm = v_cm.reshape(3, -1)[:, dstate.node_index]
        kv = dstate.beta*(g[..., 0]*vm[0]+g[..., 1]*vm[1]+g[..., 2]*vm[2])          # channel_jvp
        _channel_add(dstate, out, dstate.beta*kv, g)
    logvol, curl = "logvol" in names, "curl" in names
    if logvol or curl:
        sC = dstate.scale_curl
        for number, chunk in enumerate(dstate.q1.chunks):
            V = dstate.q1.gradient(v_cm, chunk)
            W = gJ = gcurl = None
            if logvol:
                W = _logvol_row(dstate, ev, chunk, number)
                gJ = (W*V).sum(dim=(0, 2))                          # K_J v = (s_J/J) A : V
            if curl:
                gcurl = tuple(sC*(sC*component) for component in curl_of(V))
            dstate.q1.scatter_add(out, flux_of(W, gJ, gcurl, V), chunk)
            if logvol and dstate.logvol_extra:                      # C05-2a / C05-2b, after the chunk's Gauss points
                Vc = dstate.q1_extra.gradient(v_cm, chunk)
                Wc = _extra_row(dstate, ev, chunk, number)
                gJc = (Wc*Vc).sum(dim=(0, 2))                       # K_Jx v = (s_x/J_x) A_x : V_x
                dstate.q1_extra.scatter_add(out, flux_of(Wc, gJc, None, Vc), chunk)
    h = _finish(out, "GN Hessian action")
    if "diffusion" in names:
        h = h+(state.alpha/state.omega0)*stiffness_apply(v_cm, state.grid)
    return _require_finite(h, "GN Hessian action")


def gn_diagonal(dstate, ev):
    """C06-6: the channel-block GN diagonal at ev.u (device free vector)."""
    _check(dstate, ev)
    beta, grad = dstate.beta, ev.channel.grad
    total = (beta[0]**2)[:, None]*grad[0]**2
    for c in range(1, dstate.n_channels):
        total = total+(beta[c]**2)[:, None]*grad[c]**2
    out = torch.zeros((3,)+dstate.grid.shape, dtype=torch.float64, device=grad.device)
    out.reshape(3, -1)[:, dstate.node_index] = total.T
    return _finish(out, "GN diagonal")


def c_dat(dstate, ev):
    """C09.1: (1/(3 N_B)) sum_mask sum_d sum_c (w_c/sigma_c^2)(d_d G_c)^2; terms on the device, the
    longdouble sum on the host exactly as the reference (np.sum(dtype=longdouble) of the (C, n, 3) array)."""
    _check(dstate, ev)
    terms = dstate.device.download(dstate.cdat_weight[..., None]*ev.channel.grad**2)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        total = np.sum(terms, dtype=LD)/LD(3*dstate.state.node_index.size)
    result = float(total)
    if not np.isfinite(result):
        raise FloatingPointError("c_dat: nonfinite numerical result")
    return result


# ----------------------------------------------------------------------------------------------
# C09.1 metric
# ----------------------------------------------------------------------------------------------

class TorchMetric:
    """The numpy Metric (same a, b, denominator) applied on the device: M^-1 g by DST-I, M v by the stencil."""

    def __init__(self, metric, device):
        self.metric = metric
        self.omega0, self.a, self.b = metric.omega0, metric.a, metric.b
        self.grid = metric.inverse.grid
        self.device = device
        self.denominator = device.upload(metric.inverse.denominator)

    @property
    def shift(self):
        return self.metric.shift

    def _vector(self, x, name):
        n = 3*self.grid.n_free_scalar
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.ascontiguousarray(as_float64_array(x, name, (n,)))).to(self.device.device)
        self.device.check(x, (n,))
        return x

    def solve(self, g):
        """M^-1 g = omega0 (omega_ell a I + b K_Q)^-1 g, free vector in and out."""
        g = self._vector(g, "g")
        return _require_finite(self.omega0*poisson_solve(g, self.denominator, self.grid.interior_shape), "metric solve")

    def matvec(self, v):
        """M v = omega_ell (a v + b apply_poisson_operator(v)) / omega0 per component (stencil)."""
        v = self._vector(v, "v")
        fields = v.reshape((3,)+self.grid.interior_shape)
        out = self.grid.omega*(self.a*fields+self.b*poisson_operator(fields, self.grid.spacing))
        return _require_finite(out.reshape(-1)/self.omega0, "metric action")


def build_metric(dstate, *, c_dat, b_P, lam_b=0., gamma=0.):
    """C09.1: bayesreg.model.build_metric (validation, a, b, DST denominator on the host) on the device."""
    if not isinstance(dstate, TorchModelState):
        raise InputValidationError("dstate: TorchModelState required")
    metric = _numpy_build_metric(dstate.state, c_dat=c_dat, b_P=b_P, lam_b=lam_b, gamma=gamma)
    return TorchMetric(metric, dstate.device)


# ----------------------------------------------------------------------------------------------
# Memory estimate
# ----------------------------------------------------------------------------------------------

CHUNK_BYTES_PER_CELL = 12*1024     # device temporaries of one chunk per cell (upper estimate; measured in tests)
CORNER_CHUNK_BYTES_PER_CELL = -(-CHUNK_BYTES_PER_CELL*8//27)   # C05-2a: the same per point, 8 corners per cell


def estimate_peak_bytes(shape, n_mask, n_channels, *, chunk_cells=CHUNK_CELLS, spline_chunk=SPLINE_CHUNK,
                        geometry_cache_bytes=GEOMETRY_CACHE_BYTES, live_evaluations=2, corners=False,
                        jacobian_cache_slots=JACOBIAN_CACHE_SLOTS, jacobian_cache_log=JACOBIAN_CACHE_LOG,
                        speculative_gradient=SPECULATIVE_GRADIENT, closed27=False):
    """Upper estimate of the device peak (bytes) of a solver holding `live_evaluations` evaluations, one with
    a full geometry cache, while computing an HVP: persistent state + evaluations + cache + transients.
    corners: the C05-2a corner term is on (logvol_points = "gauss27+corners"): the cache may also hold the corner
    rows (72 B x 8 per cell) within the same budget; per chunk the corner geometry (CORNER_CHUNK_BYTES_PER_CELL)
    is alive together with the Gauss-point temporaries; evaluate keeps a fourth row of per-cell sums and a second
    J <= 0 counter.  With corners=False every value is that of the estimate before C05-2a.
    OPT1: min(jacobian_cache_slots, live_evaluations) Jacobian caches of 8 B x cells x (27 + 8 corners) x (2 with log
    J), and with speculative_gradient a gradient node field (8 B x 3 N) per live evaluation; with
    jacobian_cache_slots=0 and speculative_gradient=False every value is that of the estimate before OPT1.
    live_evaluations (review OPT1_复核_20261030 item 6): a PG solve holds up to 4 device evaluations at once (the
    caller's start point, the base, the previous candidate while the next one is evaluated, the new candidate); the
    Jacobian caches stay bounded by the slots, the speculative fields of rejected PG candidates are released by the
    backend, so up to 3 fields (start, base, newest candidate) remain; pass live_evaluations=4 for an upper bound of
    such a solve (the numbers in the module docstring use the default 2).
    closed27: the C05-2b closed27 part is on (logvol_points = "gauss27+closed27"): as corners with 27 points per cell
    (cache rows 72 B x 27, chunk temporaries CHUNK_BYTES_PER_CELL, Jacobian cache 8 B x 27 per cell), the fourth
    per-cell row and counter; exclusive with corners.  With closed27=False every value is the estimate before C05-2b.
    The defaults (corners=False, closed27=False) estimate a gauss27 run: since DEF (2026-10-31) the default run is
    gauss27+closed27, so pass closed27=True for it (corners=True for gauss27+corners); the defaults underestimate the
    default run (FUP1, review MIN1 minor 6; run_pair's module docstring says the same)."""
    if corners and closed27:
        raise InputValidationError("estimate_peak_bytes: corners (C05-2a) and closed27 (C05-2b) are exclusive")
    extra_points = 8 if corners else 27 if closed27 else 0
    nz, ny, nx = shape
    N = nz*ny*nx
    cells = (nz-1)*(ny-1)*(nx-1)
    slab = (ny-1)*(nx-1)
    chunk = max(1, chunk_cells//slab)*slab
    gauss_W = 72*27*cells
    corner_W = 72*extra_points*cells
    persistent = dict(splines=8*n_channels*N, tau2=8*N, mask_arrays=8*(3*n_channels*n_mask+3*n_mask)+9*n_mask)
    evaluation = 8*3*N+8*n_channels*n_mask*5                         # u_cm, rho, r, grad
    chunk_bytes = CHUNK_BYTES_PER_CELL+(CORNER_CHUNK_BYTES_PER_CELL if corners else 0)+(
        CHUNK_BYTES_PER_CELL if closed27 else 0)
    cell_sums = (4*8+8+8+1) if extra_points else (3*8+8+1)            # rows fp64, J <= 0 counters int64, nonfinite
    transient = dict(chunk=chunk_bytes*chunk, node_fields=8*3*N*6,
                     spline=8*64*spline_chunk*6, evaluate_cell_sums=cell_sums*cells)
    full = gauss_W+corner_W
    cache = min(geometry_cache_bytes, full)
    jacobian_cache = (min(jacobian_cache_slots, live_evaluations)*8*cells*(27+extra_points)
                      * (2 if jacobian_cache_log else 1))
    speculative = live_evaluations*8*3*N if speculative_gradient else 0
    total = sum(persistent.values())+live_evaluations*evaluation+cache+jacobian_cache+speculative+max(
        transient["chunk"]+transient["node_fields"], transient["spline"]+transient["evaluate_cell_sums"]
        + transient["chunk"])
    out = dict(total=total, persistent=persistent, evaluation=evaluation, cache=cache, full_cache=full,
               cached_fraction=cache/full, jacobian_cache=jacobian_cache, speculative_gradient=speculative,
               transient=transient, chunk_cells=chunk, corners=bool(corners))
    if closed27:
        out["closed27"] = True
    return out
