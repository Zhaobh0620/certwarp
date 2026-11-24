"""Compute backend of the B-line solver and the one certificate call (contract C07, C17).

Backend protocol.  The solver touches the model only through an object with the functions of
`bayesreg.model`, with the same names and signatures:

    prepare(state)                                   -> the backend's frozen state (numpy: the ModelState)
    evaluate(state, u)                               -> Evaluation | InvalidTrial          (C06-1/7)
    gradient(state, ev, *, blocks=BLOCK_NAMES)       -> free vector (3 N_int,)             (C06-4)
    hvp(state, ev, v, *, blocks=BLOCK_NAMES)         -> free vector                         (C06-5)
    gn_diagonal(state, ev)                           -> free vector                         (C06-6)
    c_dat(state, ev)                                 -> float                               (C09.1)
    metric_bP(state, *, mode, mu=None, nu2=None)     -> float                               (C09.1)
    build_metric(state, *, c_dat, b_P, lam_b, gamma) -> object with solve(g), matvec(v)     (C09.1)
    stable_reduction(state, base, trial)             -> object with ared, guard, certified,
                                                        reason, blocks                       (C08)

Host-side requirements of the solver (a GPU backend must meet them; the numpy backend does):
  - `u` passed to evaluate is a host C-order float64 (nz, ny, nx, 3) U0 field; an Evaluation exposes
    `.energy` (Python float, the float64 value of the longdouble F_B) and `.energies` (dict of floats:
    "channel" a tuple of C values, "diffusion", "logvol", "curl"); the solver never reads anything else.
  - gradient / hvp / gn_diagonal / Metric.solve / Metric.matvec return host numpy float64 free vectors.
  - stable_reduction follows C08 bit for bit in longdouble on the CPU (C08-6: a GPU state ships the
    residual blocks or J_q pairs back to the CPU; the decision is taken on the host).
  - the certificate is never a backend function: `certify` below is the only call, CPU FP64 (C07-2,
    C17-1), for every backend.
The proximal blocks of outer option (b) are evaluated by `bayesreg.solve.proximal` (numpy geometry on
the host u); a backend may carry an attribute `proximal` with that module's evaluate / gradient / change
signatures to take them over.  Option (b) additionally reads `Evaluation.total` (the longdouble F_B) and
`Reduction.delta` (the longdouble Delta F_B) to form E_v and its stable difference.

torch64 (`Torch64Backend`, `--backend torch64`; work package SG).  prepare(state) uploads the device constants
of the host ModelState (bayesreg.backends.torch64.TorchModelState, splines reused across the rounds of a level) and
returns the host ModelState itself, so the solver keeps reading grid, omega0, sigma2 from it.  evaluate returns a
`HostEvaluation` (host u, energy, energies, total; the TorchEvaluation inside); a device evaluation that fails is
recomputed on the CPU and the CPU result decides (C08-8(e)).  gradient / hvp / gn_diagonal / metric solves run on
the device and come back as host float64 vectors; c_dat on the device with the host longdouble sum; metric_bP on
the host.  Comparisons go through `stable_compare` (C08-8: GPU path with the guard 4096 eps max(1, E), the unified
fallback rule and the CPU reference; `bayesreg.backends.torch64.stable`); `stable_reduction` (no required decrease
known) is always the CPU reference.  `confirm_gradient_norm` is the C08-8(f) recheck, `reference` the CPU backend
used for G0, `cpu_gradient_norm` the CPU GF_B of the native level's final state.  The certificate is the same CPU
FP64 call as the numpy backend.

Certificate (C07-2/3).  `certificate_config(T)` is PrescreenCertificateConfig(delta=0., tile_size=T):
delta 0, max_depth 48, max_boxes 255, edge-hull preproof + float Bernstein prescreen + interval
Bernstein fallback, CPU FP64 with outward rounding; checked field by field (never the CertificateConfig()
default).  `certify` maps the map-level reason to the solver's rejection reason:
  certified_feasible (and min J lower bound > 0)  -> passed, rejection None
  infeasible JACOBIAN_VIOLATION / INVALID_DOMAIN  -> that reason
  infeasible NONZERO_BOUNDARY                     -> CertificateContractError (injection guarantees zero faces)
  uncertain NONFINITE_STATE                       -> NONFINITE_STATE
  uncertain (any other map reason)                -> FEASIBILITY_UNCERTAIN, the map reason kept in cert_reason
"Zero" (a lower bound <= 0) and "uncertain" never pass.
Witness record (contract C05-2b "记录与 CLI", C16 log.jsonl; work packages IMPL27, IMPL27b; a record only, no
decision reads it).  For an infeasible JACOBIAN_VIOLATION outcome with a witness cell, `certify` adds the contract fields
of `bayesreg.solve.witness` (witness_cell, witness_xi, witness_type -- the exact class of witness_xi -- and
witness_cell_min_j53) as CertOutcome.witness; every other outcome has witness None and `witness_fields()` gives the
fields as null.  They cost J at the 54 sample points of one cell and are always computed (every logvol_points value).
Backend option witness_estimate (default False; run_pair --witness-estimate on|off) adds the optional estimate of the
witness cell's minimum J (CertOutcome.witness_estimate; dense sampling + L-BFGS-B, an estimate, not a bound).  No
FEASIBILITY_UNCERTAIN outcome gets a record (jd_curl reports no witness point for it; no extra certificate pass).  A
failure inside the record is caught (witness_error) -- the certificate outcome itself is never changed.  Every
backend counts its outcomes by witness type (`witness_summary`, bayesreg.solve.witness.WitnessSummary).
"""
from collections import OrderedDict
from dataclasses import dataclass
import time

import numpy as np

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError
from jd_curl.three_d.feasibility import PrescreenCertificateConfig, certificate_version, certify_map
from jd_curl.three_d.validation import integer_at_least

from .. import model as _model
from ..readonly import adopt_readonly

CERT_TILE_SIZE = 8192             # C07-2 default T (PrescreenCertificateConfig default), recorded
CERT_DELTA = 0.
CERT_MAX_DEPTH = 48
CERT_MAX_BOXES = 255
CERT_PASS = "certified_feasible"
NONFINITE_STATE = "NONFINITE_STATE"
FEASIBILITY_UNCERTAIN = "FEASIBILITY_UNCERTAIN"
CERT_INFEASIBLE_REASONS = ("JACOBIAN_VIOLATION", "INVALID_DOMAIN")
BACKENDS = ("numpy", "torch64")
BACKEND_FUNCTIONS = ("evaluate", "gradient", "hvp", "gn_diagonal", "c_dat", "metric_bP", "build_metric",
                     "stable_reduction")


class CertificateContractError(RuntimeError):
    """The certificate returned something the contract says cannot happen (implementation error)."""


def certificate_config(tile_size=CERT_TILE_SIZE):
    """C07-2: the only certificate configuration of the B line, checked field by field."""
    tile_size = integer_at_least(tile_size, "certificate tile_size")
    config = PrescreenCertificateConfig(delta=0., tile_size=tile_size)
    check_certificate_config(config, tile_size)
    return config


def check_certificate_config(config, tile_size):
    if not isinstance(config, PrescreenCertificateConfig):
        raise InputValidationError("certificate: PrescreenCertificateConfig required (C07-2)")
    if (config.delta != CERT_DELTA or config.delta.hex() != (0.).hex() or config.max_depth != CERT_MAX_DEPTH
            or config.max_boxes != CERT_MAX_BOXES or config.tile_size != tile_size):
        raise InputValidationError("certificate: delta 0, max_depth 48, max_boxes 255 and the recorded tile size "
                                   "are required (C07-2)")


def certificate_record(config):
    """What provenance.json and cert.json record of the configuration (C07 gate, C16)."""
    return dict(config_class=type(config).__name__, certificate_version=certificate_version(config),
                delta_hex=config.delta.hex(), max_depth=config.max_depth, max_boxes=config.max_boxes,
                tile_size=config.tile_size, kernel=config.kernel, fallback_kernel=config.fallback_kernel,
                prescreen_kernel=config.prescreen_kernel, fallback_workers=config.fallback_workers,
                precision="CPU FP64, outward rounding")


@dataclass(frozen=True)
class CertOutcome:
    """One certify_map call, mapped to the solver's vocabulary (C07-3)."""
    passed: bool
    status: str                   # map-level status
    cert_reason: str              # map-level reason
    rejection: str | None         # None when passed; else the C09.2-7 rejection reason
    min_jd_lower_bound: float | None
    min_margin_lower_bound: float | None
    n_uncertain: int
    n_infeasible: int
    n_cells: int
    witness_cell: int | None
    witness_reference_point: tuple | None
    state_hash: str
    certificate_version: str
    seconds: float
    witness: dict | None = None   # C05-2b / C16 witness fields (bayesreg.solve.witness.WITNESS_FIELDS) of a
    #                               JACOBIAN_VIOLATION outcome with a witness cell; None otherwise
    witness_estimate: dict | None = None   # optional estimate of the witness cell's minimum J (witness_estimate on)
    witness_seconds: float = 0.   # wall time of both (kept out of the records themselves, which are deterministic)

    @property
    def witness_type(self):
        """The exact class of witness_xi (contract C05-2b), or None."""
        return None if self.witness is None else self.witness.get("witness_type")

    def witness_fields(self):
        """The contract fields of the log row (C16): this outcome's witness record, or all null."""
        from .witness import null_fields
        return null_fields() if self.witness is None else dict(self.witness)

    def record(self):
        fields = self.witness_fields()
        return dict(status=self.status, reason=self.cert_reason, passed=self.passed, rejection=self.rejection,
                    min_jd_lower_bound=self.min_jd_lower_bound, min_margin_lower_bound=self.min_margin_lower_bound,
                    n_uncertain=self.n_uncertain, n_infeasible=self.n_infeasible, n_cells=self.n_cells,
                    witness_cell=self.witness_cell,
                    witness_reference_point=None if self.witness_reference_point is None
                    else [float(t) for t in self.witness_reference_point],
                    state_hash=self.state_hash, certificate_version=self.certificate_version,
                    seconds=self.seconds, **{k: v for k, v in fields.items() if k != "witness_cell"},
                    witness_estimate=self.witness_estimate, witness_seconds=self.witness_seconds)


def rejection_reason(status, reason, min_jd_lower_bound):
    """C07-3 map-level (status, reason) -> the solver's rejection reason (None = passed)."""
    if status == CERT_PASS:
        if min_jd_lower_bound is None or not min_jd_lower_bound > 0:
            return FEASIBILITY_UNCERTAIN          # "zero" never passes
        return None
    if reason == "NONZERO_BOUNDARY":
        raise CertificateContractError("certificate: NONZERO_BOUNDARY although every candidate is injected "
                                       "from the free vector (C07-3: implementation error)")
    if reason == NONFINITE_STATE:
        return NONFINITE_STATE
    if status == "infeasible":
        if reason in CERT_INFEASIBLE_REASONS:
            return reason
        raise CertificateContractError(f"certificate: unknown infeasible reason {reason!r}")
    if status == "uncertain":
        return FEASIBILITY_UNCERTAIN
    raise CertificateContractError(f"certificate: unknown status {status!r}")


def certify(u, grid, config, *, witness_estimate=False):
    """The only certificate call (C07-2): certify_map(u, grid, config) on the CPU in FP64.  An infeasible
    JACOBIAN_VIOLATION outcome with a witness cell gets the contract witness fields (bayesreg.solve.witness; always)
    and, with witness_estimate, the optional estimate of the cell minimum (module docstring)."""
    tick = time.perf_counter()
    result = certify_map(u, grid, config=config)
    seconds = time.perf_counter()-tick
    rejection = rejection_reason(result.status, result.reason, result.min_jd_lower_bound)
    record, estimate, witness_seconds = None, None, 0.
    if rejection == "JACOBIAN_VIOLATION" and result.witness_cell is not None:
        from .witness import null_fields, witness_estimate as estimate_minimum, witness_fields
        tick = time.perf_counter()
        try:
            record = witness_fields(u, grid, result)
            if witness_estimate:
                estimate = estimate_minimum(u, grid, result.witness_cell, record["witness_cell_min_j53"])
        except Exception as exc:                                  # a record only: never changes the outcome
            record = dict(null_fields(), witness_cell=int(result.witness_cell),
                          witness_error=f"{type(exc).__name__}: {exc}")
        witness_seconds = time.perf_counter()-tick
    return CertOutcome(passed=rejection is None, status=result.status, cert_reason=result.reason, rejection=rejection,
                       min_jd_lower_bound=result.min_jd_lower_bound,
                       min_margin_lower_bound=result.min_margin_lower_bound, n_uncertain=int(result.n_uncertain),
                       n_infeasible=int(result.n_infeasible), n_cells=int(result.n_cells_checked),
                       witness_cell=result.witness_cell, witness_reference_point=result.witness_reference_point,
                       state_hash=result.state_hash, certificate_version=result.certificate_version,
                       seconds=seconds, witness=record, witness_estimate=estimate, witness_seconds=witness_seconds)


class NumpyBackend:
    """The CPU FP64 reference: every function is the one of `bayesreg.model`."""
    name = "numpy"
    device = "cpu"
    precision = "float64 (longdouble sums, C06-7)"

    def __init__(self, *, cert_tile_size=CERT_TILE_SIZE, module=None, witness_estimate=False):
        from .witness import WitnessSummary
        self.cert_config = certificate_config(cert_tile_size)
        self._module = _model if module is None else module
        missing = [name for name in BACKEND_FUNCTIONS if not callable(getattr(self._module, name, None))]
        if missing:
            raise InputValidationError(f"backend module lacks {missing}")
        if not isinstance(witness_estimate, bool):
            raise InputValidationError("witness_estimate: bool required")
        self.witness_estimate = witness_estimate
        self.certificate_calls = 0
        self.certificate_seconds = 0.
        self.witness_records = 0
        self.witness_seconds = 0.
        self.witness_summary = WitnessSummary()

    # ---- model functions (bayesreg.model names and signatures) ----
    def prepare(self, state):
        prepare = getattr(self._module, "prepare", None)
        return state if prepare is None else prepare(state)

    def evaluate(self, state, u):
        return self._module.evaluate(state, u)

    def gradient(self, state, ev, *, blocks=_model.BLOCK_NAMES):
        return self._module.gradient(state, ev, blocks=blocks)

    def hvp(self, state, ev, v, *, blocks=_model.BLOCK_NAMES):
        return self._module.hvp(state, ev, v, blocks=blocks)

    def gn_diagonal(self, state, ev):
        return self._module.gn_diagonal(state, ev)

    def c_dat(self, state, ev):
        return self._module.c_dat(state, ev)

    def metric_bP(self, state, *, mode="principal", mu=None, nu2=None):
        return self._module.metric_bP(state, mode=mode, mu=mu, nu2=nu2)

    def build_metric(self, state, *, c_dat, b_P, lam_b=0., gamma=0.):
        return self._module.build_metric(state, c_dat=c_dat, b_P=b_P, lam_b=lam_b, gamma=gamma)

    def stable_reduction(self, state, base, trial):
        return self._module.stable_reduction(state, base, trial)

    # ---- certificate: CPU FP64 for every backend ----
    def certify(self, u, grid):
        outcome = certify(u, grid, self.cert_config, witness_estimate=self.witness_estimate)
        self.certificate_calls += 1
        self.certificate_seconds += outcome.seconds
        if outcome.witness is not None:
            self.witness_records += 1
            self.witness_seconds += outcome.witness_seconds
        self.witness_summary.observe(outcome)
        return outcome

    def record(self):
        return dict(backend=self.name, device=self.device, precision=self.precision,
                    model_module=getattr(self._module, "__name__", type(self._module).__name__),
                    certificate=certificate_record(self.cert_config), gpu_port_level=None,
                    witness_record="C05-2b / C16 fields of every JACOBIAN_VIOLATION outcome (always)",
                    witness_estimate=self.witness_estimate)


def module_backend(module, *, name, device, cert_tile_size=CERT_TILE_SIZE, witness_estimate=False):
    """A backend over any module (e.g. a torch FP64 port) with the bayesreg.model names; the
    certificate stays the CPU call of this module."""
    backend = NumpyBackend(cert_tile_size=cert_tile_size, module=module, witness_estimate=witness_estimate)
    backend.name, backend.device = name, device
    backend.precision = getattr(module, "PRECISION", "unspecified")
    return backend


class _HostMetric:
    """A device metric (bayesreg.backends.torch64.TorchMetric) with host float64 free vectors in and out."""

    def __init__(self, metric, device):
        self.device_metric, self._device = metric, device
        self.omega0, self.a, self.b = metric.omega0, metric.a, metric.b

    @property
    def shift(self):
        return self.device_metric.shift

    def solve(self, g):
        return self._device.download(self.device_metric.solve(self._device.upload(g)))

    def matvec(self, v):
        return self._device.download(self.device_metric.matvec(self._device.upload(v)))


class Torch64Backend(NumpyBackend):
    """The torch FP64 backend with the GPU-path stable comparison of C08-8 (see the module docstring).

    device: "cuda", "cpu" or None (cuda when available).  The CPU reference (bayesreg.model) is kept as
    `self._module` (fallback and stable_reduction) and `self.reference` (G0, final GF_B); the certificate is the
    inherited CPU FP64 call.  cache_size: CPU reference evaluations kept for the C08-8(d) cache.
    max_device_states: device states (one per prepared ModelState) kept alive (LRU); since MEM, preparing a state on
    a new level grid also drops the device states of every other grid (the solver never goes back; `_lookup` would
    prepare one again on demand), so record()["device_states"] -- run_pair's hyper.json
    backend_after_solve.device_states -- lists the current grid's states only (FUP1, review MEM minor 3);
    state_options: keyword options of TorchModelState (chunk_cells, spline_chunk, geometry_cache_bytes; OPT1, bitwise
    neutral: jacobian_cache_slots, jacobian_cache_log, speculative_gradient); unknown names are refused here.
    `record()` carries the options as given, their effective values (defaults filled in) and the provenance of every
    live device state (OPT1b, review OPT1_复核_20261030 item 2).  logvol_points: the log-volume point set of the run (a
    name of bayesreg.model.LOGVOL_POINTS; None: the model default), described by record()["comparison"] (the reduction
    provenance) until a state is prepared, so that a record written before the solve (run_pair's config.json and
    provenance.json) names the run's point set (MIN1, review DEF minor 1); after prepare, the last prepared state's.
    A record only: the model of every prepared state is that state's."""
    name = "torch64"
    precision = ("float64 on the device (residuals, geometry, per-tile sums, gradient, HVP, DST); host longdouble "
                 "sums; C08-8 GPU-path comparison with CPU fallback")

    def __init__(self, *, device=None, cert_tile_size=CERT_TILE_SIZE, cache_size=2, max_device_states=3,
                 state_options=None, witness_estimate=False, logvol_points=None):
        super().__init__(cert_tile_size=cert_tile_size, witness_estimate=witness_estimate)
        from ..backends import torch64
        from ..backends.torch64 import stable
        self._T, self._S = torch64, stable
        self.dev = torch64.Device(device)
        self.device = self.dev.type
        self.max_device_states = integer_at_least(max_device_states, "max_device_states")
        self.state_options = dict(state_options or {})
        self.state_options_effective = torch64.resolved_state_options(self.state_options)
        self.comparator = stable.StableComparator(cache_size=cache_size)
        self.reference = NumpyBackend(cert_tile_size=cert_tile_size)
        self._states = OrderedDict()            # id(ModelState) -> (ModelState, TorchModelState, model identity)
        self._tile_size = None
        # the point set described by record()["comparison"] until a state is prepared: the run's (logvol_points) or,
        # not given, the model default (DEF: gauss27+closed27); after prepare, that of the last prepared state
        # (hyper.json backend_after_solve)
        if logvol_points is None:
            logvol_points = _model.LOGVOL_DEFAULT
        elif logvol_points not in _model.LOGVOL_POINTS:
            raise InputValidationError(f"logvol_points: one of {_model.LOGVOL_POINTS} or None (the model default)")
        self._logvol_points = logvol_points

    # ---- device states ----
    def prepare(self, state):
        if not isinstance(state, _model.ModelState):
            raise InputValidationError("prepare: a host ModelState required")
        key = id(state)
        if key in self._states and self._states[key][0] is state:
            self._states.move_to_end(key)
            return state
        self._S.reduction_depth(state.tile_size)
        # MEM (memory only): device states of another level grid can never serve this one (the solver does not go
        # back); dropping them frees their host ModelState and device arrays before this state is uploaded
        for other in [k for k, (host, _, _) in self._states.items() if host.grid.grid_id != state.grid.grid_id]:
            del self._states[other]
        dstate = None
        for host, device_state, _ in reversed(self._states.values()):
            if len(host.samplers) == len(state.samplers) and all(
                    a is b for a, b in zip(host.samplers, state.samplers)):
                dstate = device_state.with_state(state, **self.state_options)
                break
        if dstate is None:
            dstate = self._T.TorchModelState(state, self.dev, **self.state_options)
        self._states[key] = (state, dstate, self._S.model_identity(state))
        self.comparator.forget_other_states(state)      # MEM: CPU Evaluations of other states can never be hit again
        self._tile_size = state.tile_size
        self._logvol_points = state.logvol_points
        while len(self._states) > self.max_device_states:
            self._states.popitem(last=False)
        return state

    def _lookup(self, state):
        entry = self._states.get(id(state))
        if entry is None or entry[0] is not state:
            self.prepare(state)
            entry = self._states[id(state)]
        self._states.move_to_end(id(state))
        return entry[1], entry[2]

    def model_identity(self, state):
        return self._lookup(state)[1]

    def _point(self, state, ev):
        if not isinstance(ev, self._S.HostEvaluation) or ev.state is not state:
            raise InputValidationError("evaluation: a HostEvaluation of this ModelState required "
                                       "(sigma, w, eta frozen)")
        return ev

    # ---- model functions (bayesreg.model names and signatures; host vectors out) ----
    def evaluate(self, state, u):
        """C06-1/7 on the device; a failed device evaluation (InvalidTrial, FloatingPointError, OOM or another
        RuntimeError) is recomputed on the CPU, whose result decides (C08-8(e))."""
        from ..geom import node_field
        dstate, _ = self._lookup(state)
        u_host = adopt_readonly(node_field(u, state.grid))      # MEM: no copy of an owned read-only u (same values)
        try:
            tev = self._T.evaluate(dstate, self.dev.upload(u_host))
            anomaly = None
            if isinstance(tev, InvalidTrial):
                anomaly = f"device InvalidTrial {tev.reason} ({tev.invalid_count})"
        except InputValidationError:
            raise
        except (FloatingPointError, RuntimeError) as exc:
            tev, anomaly = None, f"device exception {type(exc).__name__}: {exc}"
        if anomaly is None:
            return self._S.HostEvaluation.from_device(state, u_host, tev)
        cpu = self._module.evaluate(state, u_host)
        self.comparator.stats.cpu_evaluations += 1
        self.comparator.stats.evaluate_anomalies.append(dict(
            anomaly=anomaly, cpu=cpu.reason if isinstance(cpu, InvalidTrial) else "evaluated"))
        if isinstance(cpu, InvalidTrial):
            return cpu
        return self._S.HostEvaluation.from_cpu(state, cpu, anomaly)

    def gradient(self, state, ev, *, blocks=_model.BLOCK_NAMES):
        ev = self._point(state, ev)
        if ev.device is None:
            return self._module.gradient(state, ev.cpu, blocks=blocks)
        return self.dev.download(self._T.gradient(ev.device.state, ev.device, blocks=blocks))

    def hvp(self, state, ev, v, *, blocks=_model.BLOCK_NAMES):
        ev = self._point(state, ev)
        if ev.device is None:
            return self._module.hvp(state, ev.cpu, v, blocks=blocks)
        if isinstance(v, np.ndarray):
            v = self.dev.upload(v)
        return self.dev.download(self._T.hvp(ev.device.state, ev.device, v, blocks=blocks))

    def gn_diagonal(self, state, ev):
        ev = self._point(state, ev)
        if ev.device is None:
            return self._module.gn_diagonal(state, ev.cpu)
        return self.dev.download(self._T.gn_diagonal(ev.device.state, ev.device))

    def c_dat(self, state, ev):
        ev = self._point(state, ev)
        if ev.device is None:
            return self._module.c_dat(state, ev.cpu)
        return self._T.c_dat(ev.device.state, ev.device)

    def metric_bP(self, state, *, mode="principal", mu=None, nu2=None):
        return self._module.metric_bP(state, mode=mode, mu=mu, nu2=nu2)

    def build_metric(self, state, *, c_dat, b_P, lam_b=0., gamma=0.):
        dstate, _ = self._lookup(state)
        return _HostMetric(self._T.build_metric(dstate, c_dat=c_dat, b_P=b_P, lam_b=lam_b, gamma=gamma), self.dev)

    # ---- comparisons (C08) ----
    def stable_reduction(self, state, base, trial):
        """C08-1..7 on the CPU (both states re-evaluated, cached): for callers that know no required decrease."""
        _, model_id = self._lookup(state)
        points = []
        for k, point in enumerate((self._point(state, base), self._point(state, trial))):
            cpu = self.comparator.cpu_point(state, point, model_id=model_id, base=k == 0)
            if isinstance(cpu, InvalidTrial):
                raise FloatingPointError(f"CPU reference: state not evaluable ({cpu.reason})")
            points.append(cpu)
        return self._module.stable_reduction(state, *points)

    def stable_compare(self, state, base, trial, *, required=0., energy="F_B", reference=None, extra=None,
                       context="unspecified", gf_ratio=None):
        """C08-8: the GPU-path comparison of E(base) and E(trial) with the unified fallback rule.
        A PG candidate the comparison rejects (a reason, or certified < required: run_pg's rule) never gets a gradient,
        so its OPT1 speculative gradient field is released here (memory only, review OPT1_复核_20261030 item 6; were it
        needed after all, `gradient` recomputes it bitwise the same)."""
        base, trial = self._point(state, base), self._point(state, trial)
        dstate, model_id = self._lookup(state)
        if base.device is not None:
            dstate = base.device.state
        red = self.comparator.compare(state, dstate, base, trial, model_id=model_id, required=required,
                                      energy=energy, reference=reference, extra=extra, context=context,
                                      gf_ratio=gf_ratio)
        if context == "pg" and trial.device is not None and (red.reason is not None
                                                             or not red.certified >= float(required)):
            trial.device.release_speculative_gradient()
        return red

    def confirm_gradient_norm(self, state, point, G, target, *, energy="F_B", reference=None, extra_gradient=None,
                              context="unspecified"):
        """C08-8(f): the CPU value when T_B/1.01 <= G <= 1.01 T_B; returns (G used, record or None)."""
        point = self._point(state, point)
        _, model_id = self._lookup(state)
        return self.comparator.confirm_gradient_norm(state, point, G, target, model_id=model_id, energy=energy,
                                                     reference=reference, extra_gradient=extra_gradient,
                                                     context=context)

    def cpu_gradient_norm(self, state, u):
        """GF_B of u on the CPU reference (C08-8(f): the native level's final state)."""
        from .stopping import gf_b
        ev = self.reference.evaluate(state, u)
        if isinstance(ev, InvalidTrial):
            raise FloatingPointError(f"CPU reference: final state not evaluable ({ev.reason})")
        return gf_b(self.reference.gradient(state, ev), state.grid.omega, state.omega0)

    def comparison_summary(self):
        return self.comparator.stats.summary()

    def record(self):
        from ..geom import TILE_SIZE
        out = super().record()
        out.update(device=self.device, device_info=self.dev.describe(), model_module="bayesreg.backends.torch64",
                   cpu_reference="bayesreg.model",
                   comparison=self._S.reduction_provenance(self._tile_size or TILE_SIZE, self._logvol_points),
                   cpu_cache_size=self.comparator.cache_size, state_options=self.state_options,
                   state_options_effective=dict(self.state_options_effective),
                   state_options_note="OPT1 jacobian_cache_slots / jacobian_cache_log / speculative_gradient: 数值逐位不变；"
                                      "只影响显存与速度 (bitwise neutral: device memory and speed only)",
                   device_states=[dstate.provenance() for _, dstate, _ in self._states.values()],
                   gpu_port_level="L0 (FP64, tests/test_bayesreg/torch64); C08-9 L1 gate pending")
        return out


def make_backend(name="numpy", *, device=None, cert_tile_size=CERT_TILE_SIZE, witness_estimate=False, **options):
    """--backend numpy | torch64 (default numpy); witness_estimate: --witness-estimate on|off (both backends)."""
    if name == "numpy":
        if device not in (None, "cpu") or options:
            raise InputValidationError("the numpy backend runs on the cpu and takes no device options")
        return NumpyBackend(cert_tile_size=cert_tile_size, witness_estimate=witness_estimate)
    if name == "torch64":
        return Torch64Backend(device=device, cert_tile_size=cert_tile_size, witness_estimate=witness_estimate,
                              **options)
    raise InputValidationError(f"backend: one of {BACKENDS}")


def assert_finite_vector(value, name):
    array = np.asarray(value)
    if array.dtype != np.float64 or array.ndim != 1 or not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{name}: a finite float64 free vector is required")
    return array
