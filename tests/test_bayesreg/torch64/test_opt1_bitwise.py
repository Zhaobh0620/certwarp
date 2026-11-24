"""OPT1 (C2a Jacobian cache in the C08-8 device comparison, C3 speculative gradient): bitwise equality with the code
before OPT1 (commit fd93113), on the torch cpu device and, when available, cuda.

Reference: tests/test_bayesreg/torch64/_opt1_reference.py, the pre-OPT1 evaluate, gradient, device_gauss_changes and
gpu_fb_change verbatim (test_reference_is_verbatim_fd93113 compares their source with the commit when the repository
history is available).

(i)   unit level, per case (cpu, cuda) x (gauss27, gauss27+corners) x (with channels C = 3, without tissue / structure
      channels C = 1: intensity only; ModelState needs >= 1 channel and a nonempty mask) x (several chunks: 4, 5 or 9)
      x OPT1 options (default: J cached, log J not (OPT1b); log J cached too; cache and speculative gradient off): for
      one base and one trial,
      F_B, energies, total, tile sums and channel residuals of both evaluations; the device tile sums of the
      comparison; Delta F_B and every block; the comparator's A_hat, s_hat, path, guard, certified, reason and blocks
      for four required decreases (GPU path, both band fallbacks, a far one); the gradient (all blocks: the speculative
      field; every single block and a pair: recomputed; after release: recomputed) -- all bitwise equal to the
      reference.  Partial caches (one state released, one chunk dropped) give the same tile sums.
(ii)  lifetime: the Jacobian-cache slots keep the base and the newest trial (LRU with the trial touched before the
      base, gradient a use), an evaluation's caches go with it, one bound per physical device whatever the number of
      Device objects or backends (OPT1b, review item 7), a forged evaluation (dataclasses.replace with another u_cm or
      channel point, or u_cm changed in place) never uses a stale cache; the backend releases the speculative field of
      a PG candidate it rejects, keeps it otherwise (review item 6); the backend records the effective options and
      the provenance of its device states (review item 2).
(iii) nonfinite check: G finite does not imply A finite (overflow constructed); evaluate refuses a state with a
      nonfinite A (so a cached chunk's A was checked finite); a recomputed chunk still checks A.
(iv)  run_pg sequences through solve_direct / solve_proximal on the torch64 backend (the 9^3 synthetic problem of the
      JDS tests with u* amplitude (.8, -.7, .6) voxels, 8 chunks, 6 tiles, c1 = .9 so that evaluated candidates are
      rejected by Armijo; 10 accepted steps): every log row (all fields but timings), the state-hash sequence, the
      energies, the gradient norms, the final theta, the comparison summary and every device Delta F_B with its blocks
      (longdouble; OPT1b) bitwise equal to the pre-OPT1 path; >= 20 candidates with certificate rejections and
      evaluated-and-rejected ones; at least one comparison decided on the device path and no device anomaly (review
      item 5).  cuda: the peak device memory of both runs is recorded.  That these checks catch a rounding-order
      change: test_opt1_mutation.py (review item 3).
(v)   cuda: the measured peak of evaluate + gradient + 2 HVP (one chunk, no HVP cache) within estimate_peak_bytes with
      the OPT1 terms.
The cuda tests are small (<= 9^3 nodes for the solver, the 40 x 38 x 36 medium case for the peak); the GPU may be
shared, nothing here is timed.
"""
from dataclasses import replace
import inspect
import json
import os
import subprocess

import numpy as np
import pytest
import torch

from jd_curl.three_d.boundary import restrict_free
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.validation import InputValidationError

from bayesreg.model import BLOCK_NAMES, ModelState, build_samplers
import bayesreg.backends.torch64 as T
from bayesreg.backends.torch64 import model as TM
from bayesreg.backends.torch64 import stable as S
from bayesreg.backends.torch64.q1 import cofactor
from bayesreg.solve import (Caps, Commit, JDScalingConfig, PGConfig, ProximalParams, Torch64Backend, default_lambda,
                            g0_of, solve_direct, solve_proximal, threshold)

from . import _opt1_reference as R
from ._cases_gauss27 import case, smooth_field  # DEF: "gauss27" (the default before DEF) explicit
from ..model._gauss27 import make_state
from ..model._helpers import aniso, feasible_field
from ..solve import _synthetic as syn

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only"))]
CORNERS = "gauss27+corners"
LD = np.longdouble
OPTIONS = dict(default={}, log=dict(jacobian_cache_log=True),       # OPT1b: the default keeps J only
               off=dict(jacobian_cache_slots=0, speculative_gradient=False))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def ld(value):
    return np.format_float_scientific(LD(value), unique=True)


def digest_array(array):
    return np.ascontiguousarray(np.asarray(array, dtype=np.float64)).tobytes()


# ----------------------------------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------------------------------

def unit_case(name, points):
    """(state, u_base, u_trial, chunk_cells): small synthetic states with several chunks."""
    if name == "aniso_c3":                       # 7 x 6 x 5, C = 3, Student-t, tile 7 (18 tiles), 4 one-slab chunks
        state, u, _ = case("small")
        chunk = 40
    elif name == "aniso_c1":                     # the same grid, intensity only (C = 1)
        g = aniso()
        state = make_state(g, seed=0, n_channels=1, student_t=True, tile_size=7)
        u = feasible_field(g, np.random.default_rng(1))
        chunk = 30                               # slab = 5 x 6 = 30 cells: 4 one-slab chunks
    elif name == "box_c3":                       # 12 x 11 x 10, C = 3, Gaussian, tile 64 (16 tiles), 5 chunks
        g = make_grid_3d(nx=12, ny=11, nz=10, hx=.5, hy=.45, hz=.6, x0=.1, y0=-.4, z0=.3, unit="mm")
        state = make_state(g, seed=3, student_t=False, tile_size=64)
        u = feasible_field(g, np.random.default_rng(4), scale=.12)
        chunk = 220                              # slab = 10 x 11 = 110 cells: 2 slabs per chunk
    elif name == "box_c1":                       # the same grid, intensity only, 9 one-slab chunks
        g = make_grid_3d(nx=12, ny=11, nz=10, hx=.5, hy=.45, hz=.6, x0=.1, y0=-.4, z0=.3, unit="mm")
        state = make_state(g, seed=3, n_channels=1, student_t=True, tile_size=64)
        u = feasible_field(g, np.random.default_rng(4), scale=.12)
        chunk = 110
    else:
        raise KeyError(name)
    if points != "gauss27":
        state = state.replace(logvol_points=points)
    u_trial = u+smooth_field(state.grid, .03, 9)
    return state, u, u_trial, chunk


UNIT_CASES = ["aniso_c3", "aniso_c1", "box_c3", "box_c1"]


def evaluation_parts(ev):
    return [ld(ev.total), ev.energy.hex(), [e.hex() for e in ev.energies["channel"]],
            [ev.energies[name].hex() for name in ("diffusion", "logvol", "curl")], digest_array(ev.tile_sums),
            digest_array(ev.r_host)]


def reduction_parts(red):
    return [red.path, None if red.ared is None else red.ared.hex(), None if red.guard is None else red.guard.hex(),
            None if red.certified is None else red.certified.hex(), red.reason, red.info["fallback_reason"],
            None if red.info["A_hat"] is None else red.info["A_hat"].hex(),
            None if red.info["s_hat"] is None else red.info["s_hat"].hex(),
            None if red.delta is None else ld(red.delta),
            {k: ([ld(c) for c in v] if k == "channel" else ld(v)) for k, v in red.blocks.items()}]


def host(state, tev):
    u = np.array(T.to_numpy(tev.u), order="C")
    u.flags.writeable = False
    return S.HostEvaluation.from_device(state, u, tev)


def pipeline(state, u, u_trial, device, chunk, options, monkeypatch=None):
    """Everything (i) compares, computed by the current code, or by the pre-OPT1 reference when monkeypatch is given
    (the reference functions installed for the comparator's gpu_fb_change)."""
    ds = T.TorchModelState(state, device, chunk_cells=chunk, **options)
    assert len(ds.q1.chunks) >= 4
    evaluate = R.evaluate if monkeypatch is not None else T.evaluate
    gradient = R.gradient if monkeypatch is not None else T.gradient
    tb, tt = evaluate(ds, u), evaluate(ds, u_trial)
    out = dict(base=evaluation_parts(tb), trial=evaluation_parts(tt))
    if monkeypatch is not None:
        monkeypatch.setattr(S, "gpu_fb_change", R.gpu_fb_change)
        tiles = R.device_gauss_changes(ds, tb, tt)
        delta, blocks = R.gpu_fb_change(ds, tb, tt)
    else:
        tiles = S.device_gauss_changes(ds, tb, tt)
        delta, blocks = S.gpu_fb_change(ds, tb, tt)
    out["tiles"] = digest_array(tiles)
    out["delta"] = [ld(delta)]+[[ld(c) for c in blocks["channel"]]]+[ld(blocks[k]) for k in ("diffusion", "logvol",
                                                                                               "curl")]
    hb, ht = host(state, tb), host(state, tt)
    A_hat = float(-delta)
    s_hat = S.gpu_guard(tb.energy, tt.energy)
    comparator = S.StableComparator()
    mid = S.model_identity(state)
    out["compare"] = []
    for required, context in ((0., "outer_F"), (A_hat, "pg"), (abs(A_hat)/2, "pg"), (abs(A_hat)+64*s_hat, "pg")):
        red = comparator.compare(state, ds, hb, ht, model_id=mid, required=required, context=context)
        out["compare"].append(reduction_parts(red))
    if monkeypatch is not None:
        monkeypatch.undo()
    out["gradient_base"] = digest_array(T.to_numpy(gradient(ds, tb)))
    out["gradient_trial"] = digest_array(T.to_numpy(gradient(ds, tt)))
    out["gradient_blocks"] = [digest_array(T.to_numpy(gradient(ds, tt, blocks=(b,)))) for b in BLOCK_NAMES]
    out["gradient_pair"] = digest_array(T.to_numpy(gradient(ds, tt, blocks=("logvol", "curl"))))
    out["gradient_permuted"] = digest_array(T.to_numpy(gradient(ds, tt, blocks=tuple(reversed(BLOCK_NAMES)))))
    return out, ds, tb, tt


# ----------------------------------------------------------------------------------------------
# (0) the reference is the pre-OPT1 code
# ----------------------------------------------------------------------------------------------

def _git_show(path):
    try:
        return subprocess.run(["git", "-C", REPO, "show", f"{R.COMMIT}:{path}"], capture_output=True, text=True,
                              check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _function_source(text, name):
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(f"def {name}("))
    end = start+1
    while end < len(lines) and not (lines[end] and not lines[end][0].isspace() and not lines[end].startswith(")")):
        end += 1
    while lines[end-1] == "":
        end -= 1
    return "\n".join(lines[start:end])


def test_reference_is_verbatim_fd93113():
    sources = {path: _git_show(path) for path in set(R.SOURCES.values())}
    if any(text is None for text in sources.values()):
        pytest.skip(f"repository history with commit {R.COMMIT} not available")
    for name in R.FUNCTIONS:
        expected = _function_source(sources[R.SOURCES[name]], name)
        assert inspect.getsource(getattr(R, name)).rstrip("\n") == expected, name


# ----------------------------------------------------------------------------------------------
# (i) unit level
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("option", list(OPTIONS))
@pytest.mark.parametrize("points", ["gauss27", CORNERS])
@pytest.mark.parametrize("name", UNIT_CASES)
@pytest.mark.parametrize("device", DEVICES)
def test_unit_bitwise(device, name, points, option, monkeypatch):
    state, u, u_trial, chunk = unit_case(name, points)
    options = OPTIONS[option]
    new, ds, tb, tt = pipeline(state, u, u_trial, device, chunk, options)
    old, *_ = pipeline(state, u, u_trial, device, chunk, options, monkeypatch=monkeypatch)
    assert new.keys() == old.keys()
    for key in new:
        assert new[key] == old[key], key
    paths = [row[0] for row in new["compare"]]
    assert "gpu" in paths and "cpu" in paths              # both the GPU decision and the band fallback are covered
    cached = option != "off"
    n_chunks = len(ds.q1.chunks)
    assert (len(tt._jacobians["entries"]) if cached else 0) == (2 if points == CORNERS else 1)*n_chunks*cached
    assert (tt.speculative_field() is not None) == cached
    if cached:                                            # log J kept only with jacobian_cache_log (default off)
        assert all((entry[1] is not None) == (option == "log") for entry in tt._jacobians["entries"].values())
    # after release: recomputed, still bitwise
    tt.release_speculative_gradient()
    tt.release_jacobian_cache()
    assert digest_array(T.to_numpy(T.gradient(ds, tt))) == old["gradient_trial"]
    assert digest_array(S.device_gauss_changes(ds, tb, tt)) == old["tiles"]


@pytest.mark.parametrize("points", ["gauss27", CORNERS])
@pytest.mark.parametrize("device", DEVICES)
def test_partial_caches_bitwise(device, points):
    """One state cached and the other not, a chunk entry missing, log J kept or not: the same device tile sums."""
    state, u, u_trial, chunk = unit_case("box_c3", points)
    ds_ref = T.TorchModelState(state, device, chunk_cells=chunk, jacobian_cache_slots=0)
    expected = digest_array(R.device_gauss_changes(ds_ref, R.evaluate(ds_ref, u), R.evaluate(ds_ref, u_trial)))
    for log in (True, False):
        ds = T.TorchModelState(state, device, chunk_cells=chunk, jacobian_cache_log=log)
        tb, tt = T.evaluate(ds, u), T.evaluate(ds, u_trial)
        assert digest_array(S.device_gauss_changes(ds, tb, tt)) == expected
        del tt._jacobians["entries"][2]                      # one Gauss chunk of the trial recomputed
        if points == CORNERS:
            del tb._jacobians["entries"][("corners", 1)]     # one corner chunk of the base recomputed
        assert digest_array(S.device_gauss_changes(ds, tb, tt)) == expected
        tb.release_jacobian_cache()
        assert digest_array(S.device_gauss_changes(ds, tb, tt)) == expected


# ----------------------------------------------------------------------------------------------
# (ii) lifetime of the caches
# ----------------------------------------------------------------------------------------------

def test_slots_keep_base_and_newest_trial():
    """The solver's pattern: evaluate base, gradient(base); per candidate evaluate trial, compare(base, trial); an
    accepted trial gets gradient(trial) and becomes the base, while an old base may still be referenced elsewhere."""
    state, u, _, chunk = unit_case("aniso_c3", "gauss27")
    ds = T.TorchModelState(state, "cpu", chunk_cells=chunk)
    slots = ds.jacobian_slots
    slots.release_all()                                   # one bound per physical device: start from no holder
    trials = [u+smooth_field(state.grid, a, s) for a, s in ((.02, 3), (.01, 4), (-.01, 5), (.015, 6), (.005, 7))]
    start = T.evaluate(ds, u)                             # held by the caller (solve_direct) all along
    T.gradient(ds, start)
    base = start
    assert slots.holders() == [start]
    kept = []
    for k, w in enumerate(trials):
        trial = T.evaluate(ds, w)
        assert trial.jacobian_cache_nbytes > 0 and base.jacobian_cache_nbytes > 0, k     # base never evicted
        assert len(slots.holders()) <= ds.jacobian_cache_slots
        S.device_gauss_changes(ds, base, trial)
        if k in (1, 3):                                   # accepted: gradient(trial), the trial becomes the base
            T.gradient(ds, trial)
            base = trial
        kept.append(trial)                                # every trial still referenced: eviction, not GC, frees
    assert start.jacobian_cache_nbytes == 0 and set(map(id, slots.holders())) == {id(base), id(kept[-1])}
    # an evaluation's caches go with it
    del start, base, trial, kept
    assert slots.holders() == []
    ds2 = T.TorchModelState(state, "cpu", chunk_cells=chunk)
    assert ds2.jacobian_slots is slots                    # another Device object on the same device: the same bound
    ev = T.evaluate(ds2, u)
    assert ev.jacobian_cache_nbytes == 8*27*state.grid.n_cells and slots.holders() == [ev]     # J only (OPT1b)
    del ev
    assert slots.holders() == []
    logged = T.TorchModelState(state, "cpu", chunk_cells=chunk, jacobian_cache_log=True)
    assert T.evaluate(logged, u).jacobian_cache_nbytes == 8*2*27*state.grid.n_cells
    off = T.TorchModelState(state, "cpu", chunk_cells=chunk, jacobian_cache_slots=0)
    assert T.evaluate(off, u).jacobian_cache_nbytes == 0
    # with_state shares the Device and its slots
    assert ds.with_state(state).jacobian_slots is slots
    # two backends on one device share the bound (before OPT1b each Device object had its own: review item 7)
    first, second = (Torch64Backend(device="cpu", state_options=dict(chunk_cells=chunk)) for _ in range(2))
    assert first.dev is not second.dev
    points = []
    for k in range(4):
        backend = (first, second)[k % 2]
        backend.prepare(state)
        points.append(backend.evaluate(state, trials[k]))
    assert set(map(id, slots.holders())) == {id(p.device) for p in points[-2:]}
    assert all(p.device.jacobian_cache_nbytes == 0 for p in points[:2])


def test_rejected_pg_candidate_releases_its_speculative_field():
    """Review item 6: Torch64Backend.stable_compare releases the speculative gradient field of a PG candidate it
    rejects (run_pg's rule: a reason, or certified < required) and keeps it for an accepted one and for the outer
    comparisons; a released field is recomputed bitwise the reference's."""
    state, u, u_trial, chunk = unit_case("box_c3", "gauss27")
    ref_ds = T.TorchModelState(state, "cpu", chunk_cells=chunk, jacobian_cache_slots=0, speculative_gradient=False)
    reference = {key: digest_array(T.to_numpy(R.gradient(ref_ds, R.evaluate(ref_ds, w))))
                 for key, w in (("u", u), ("trial", u_trial))}
    backend = Torch64Backend(device="cpu", state_options=dict(chunk_cells=chunk))
    backend.prepare(state)
    outcomes = []
    for base_u, trial_u, key in ((u, u_trial, "trial"), (u_trial, u, "u")):
        for required, context in ((0., "pg"), (1e300, "pg"), (1e300, "outer_F"), (0., "outer_E")):
            base, trial = backend.evaluate(state, base_u), backend.evaluate(state, trial_u)
            red = backend.stable_compare(state, base, trial, required=required, context=context)
            rejected = red.reason is not None or not red.certified >= required
            released = trial.device.speculative_field() is None
            assert released == (context == "pg" and rejected), (key, required, context)
            assert base.device.speculative_field() is not None
            assert digest_array(backend.gradient(state, trial)) == reference[key]
            outcomes.append((context, rejected))
    assert ("pg", True) in outcomes and ("pg", False) in outcomes


def test_backend_records_effective_options_and_device_states():
    """Review item 2: the backend record carries the options as given, the effective values (defaults filled in;
    jacobian_cache_log False by default since OPT1b) and the provenance of every live device state; unknown option
    names are refused when the backend is built."""
    from bayesreg.solve.export import dumps
    assert T.resolved_state_options() == dict(chunk_cells=T.CHUNK_CELLS, spline_chunk=T.SPLINE_CHUNK,
                                              geometry_cache_bytes=T.GEOMETRY_CACHE_BYTES, jacobian_cache_slots=2,
                                              jacobian_cache_log=False, speculative_gradient=True)
    with pytest.raises(InputValidationError, match="unknown"):
        Torch64Backend(device="cpu", state_options=dict(jacobian_cache=False))
    state, u, _, chunk = unit_case("aniso_c3", CORNERS)
    options = dict(chunk_cells=chunk, jacobian_cache_slots=0, speculative_gradient=False)
    backend = Torch64Backend(device="cpu", state_options=options)
    record = backend.record()
    effective = record["state_options_effective"]
    assert record["state_options"] == options and record["device_states"] == []
    assert effective == dict(T.resolved_state_options(), **options) and effective["jacobian_cache_log"] is False
    backend.prepare(state)
    point = backend.evaluate(state, u)
    assert point.device.jacobian_cache_nbytes == 0 and point.device.speculative_field() is None
    states = backend.record()["device_states"]
    assert len(states) == 1 and all(states[0][key] == value for key, value in effective.items())
    assert states[0]["logvol_points"] == CORNERS
    dumps(backend.record())                               # JSON-serializable for config / provenance / hyper
    """dataclasses.replace with another u_cm or channel point, or u_cm changed in place: no cached J, no speculative
    field; the comparison and the gradient recompute (bitwise the reference)."""
    state, u, u_trial, chunk = unit_case("aniso_c3", CORNERS)
    ds = T.TorchModelState(state, "cpu", chunk_cells=chunk)
    tb, tt = T.evaluate(ds, u), T.evaluate(ds, u_trial)
    ref_ds = T.TorchModelState(state, "cpu", chunk_cells=chunk, jacobian_cache_slots=0, speculative_gradient=False)
    rb, rt = R.evaluate(ref_ds, u), R.evaluate(ref_ds, u_trial)
    forged = replace(tb, u_cm=tt.u_cm, channel=tt.channel, _cache={})          # tb's caches, tt's state
    assert forged.jacobian_entry(0) is None and forged.speculative_field() is None
    assert tb.jacobian_entry(0) is not None and tb.speculative_field() is not None
    assert digest_array(S.device_gauss_changes(ds, tb, forged)) == digest_array(R.device_gauss_changes(ref_ds, rb, rt))
    assert digest_array(T.to_numpy(T.gradient(ds, forged))) == digest_array(T.to_numpy(R.gradient(ref_ds, rt)))
    other_point = replace(tt, channel=tb.channel)
    assert other_point.jacobian_entry(0) is not None and other_point.speculative_field() is None
    copy = tt.u_cm.clone()
    tt.u_cm.add_(0.)                                      # in-place: the version counter moves, values do not
    assert torch.equal(copy, tt.u_cm)
    assert tt.jacobian_entry(0) is None and tt.speculative_field() is None


# ----------------------------------------------------------------------------------------------
# (iii) the nonfinite check
# ----------------------------------------------------------------------------------------------

def test_finite_G_does_not_imply_finite_A():
    """A = cof(I + G) has products of two entries of F: |G| ~ 1e155 (finite) overflows A to inf."""
    G = torch.zeros((3, 1, 3, 27), dtype=torch.float64)
    G[1, 0, 0] = 1e155
    G[2, 0, 1] = 1e155
    F = G.clone()
    for r in range(3):
        F[r, :, r] += 1.
    A = cofactor(F)
    assert bool(torch.isfinite(G).all()) and not bool(torch.isfinite(A).all())


def _blow_up(q1, scale):
    original = q1.gradient

    def gradient(field, chunk):
        G = original(field, chunk)
        if chunk == q1.chunks[1]:
            G = G.clone()
            G[1, 0, 0] *= scale
            G[2, 0, 1] *= scale
        return G
    return gradient


def test_evaluate_refuses_a_nonfinite_A():
    """A G that is finite but overflows A in one cell: evaluate raises (as before OPT1), so no evaluation -- and no
    Jacobian cache -- exists for it; a cached chunk's A was always checked finite by evaluate."""
    state, u, _, chunk = unit_case("aniso_c3", "gauss27")
    ds = T.TorchModelState(state, "cpu", chunk_cells=chunk)
    ds.q1.gradient = _blow_up(ds.q1, 1e180)
    with pytest.raises(FloatingPointError, match="Gauss-point geometry: nonfinite"):
        T.evaluate(ds, u)
    with pytest.raises(FloatingPointError, match="Gauss-point geometry: nonfinite"):
        R.evaluate(ds, u)
    assert not any(ev.state is ds for ev in ds.jacobian_slots.holders())     # slots are per device, shared


def test_recomputed_chunks_still_check_A():
    """A chunk without a cache entry is recomputed in full (G, A, J) and checked as before OPT1: the same device
    anomaly, with the same message, as the pre-OPT1 comparison (here J of the overflowing cells is -inf, so the message
    is the J <= 0 one).  With a cache entry the A and J of that chunk are not recomputed: only G (finite here) and the
    tile sums are checked, and the overflowing diffusion terms make the tile sums nonfinite, so the comparison is still
    a device anomaly (CPU decision), with another message.  A recomputed G can differ from evaluate's G only on a
    nondeterministic device; for evaluations from `evaluate` the recomputation is bitwise evaluate's (tested above)."""
    state, u, u_trial, chunk = unit_case("aniso_c3", "gauss27")
    ds = T.TorchModelState(state, "cpu", chunk_cells=chunk)
    tb, tt = T.evaluate(ds, u), T.evaluate(ds, u_trial)
    ds.q1.gradient = _blow_up(ds.q1, 1e180)
    with pytest.raises(S.DeviceAnomaly, match=r"device nonfinite values \(0 geometry, [1-9]\d* tile sums\)"):
        S.device_gauss_changes(ds, tb, tt)
    with pytest.raises(S.DeviceAnomaly) as reference:
        R.device_gauss_changes(ds, tb, tt)
    tb.release_jacobian_cache()
    tt.release_jacobian_cache()
    with pytest.raises(S.DeviceAnomaly) as recomputed:
        S.device_gauss_changes(ds, tb, tt)
    assert str(recomputed.value) == str(reference.value)
    # the overflow reaches A: with J kept finite by hand the recomputed chunk is caught by the A check alone
    original = TM._geometry

    def finite_J(dstate, u_cm, chunk_, q1=None):
        G, F, A, J = original(dstate, u_cm, chunk_, q1)
        assert not bool(torch.isfinite(A).all()) or chunk_ != ds.q1.chunks[1]
        return G, F, A, torch.where(torch.isfinite(J), J, torch.ones_like(J))
    S_geometry = S._geometry
    try:
        S._geometry = finite_J
        with pytest.raises(S.DeviceAnomaly, match=r"device nonfinite values \([1-9]\d* geometry"):
            S.device_gauss_changes(ds, tb, tt)
    finally:
        S._geometry = S_geometry


# ----------------------------------------------------------------------------------------------
# (iv) run_pg sequences
# ----------------------------------------------------------------------------------------------

def fold_problem(points):
    """The 9^3 synthetic problem of the JDS tests with u* amplitude (.8, -.7, .6) voxels, tile 100 (6 tiles)."""
    grid = syn.cube_grid(9)
    moving = syn.moving_image(grid.identity())
    fixed = syn.warp_with_spline(moving, grid, syn.true_displacement(grid, (.8, -.7, .6)))
    state = ModelState(grid=grid, omega0=grid.omega, fixed=fixed[None], samplers=build_samplers(moving[None], grid),
                       mask=syn.brain_box(grid, 1), sigma2=np.array([1e-2]), alpha=1e-3, nu=1e-3,
                       tau2=np.full(4, 1.), tissue=syn.uniform_tissue(grid), tile_size=100, logvol_points=points)
    return grid, state


def canonical(value):
    """Exact, comparable form of a log row: floats by hex, longdouble by its shortest repr, timings dropped."""
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in sorted(value.items()) if "seconds" not in k}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return ["ndarray", value.dtype.str, value.shape, digest_array(value).hex()]
    if isinstance(value, (bool, np.bool_)) or value is None or isinstance(value, str):
        return value if not isinstance(value, np.bool_) else bool(value)
    if isinstance(value, np.longdouble):
        return ["ld", ld(value)]
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    return repr(value)


def solve(option, device, points):
    backend = Torch64Backend(device=device, state_options=dict(chunk_cells=64))
    grid, state = fold_problem(points)
    state = backend.prepare(state)
    zero = np.zeros(grid.shape+(3,))
    start = Commit(zero, restrict_free(zero, grid), backend.certify(zero, grid))
    target = threshold(g0_of(backend, state))
    rows = []
    pg = PGConfig(c1=.9, jd=JDScalingConfig(cell_estimate="gauss27+corners"))     # DEF: the estimate before DEF
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
    if option == "a":
        result = solve_direct(backend, state, start, target=target, caps=Caps(max_accepted=10), log=rows.append, pg=pg)
    else:
        result = solve_proximal(backend, state, start, target=target, caps=Caps(max_accepted=10, max_outer=3),
                                params=ProximalParams(lam=default_lambda(grid)), log=rows.append, pg=pg)
    peak = None
    if device == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()-before
    return backend, result, rows, peak


def solve_recording(option, device, points, mp):
    """solve() with the device Delta F_B of every comparison recorded exactly (longdouble total and blocks): the logged
    float64 A_hat can hide a change of the device tile sums (test_opt1_mutation.py), Delta F_B does not."""
    inner = S.gpu_fb_change
    deltas = []

    def recording(dstate, base, trial):
        delta, blocks = inner(dstate, base, trial)
        deltas.append([ld(delta), [ld(c) for c in blocks["channel"]]]+[ld(blocks[k]) for k in ("diffusion", "logvol",
                                                                                                   "curl")])
        return delta, blocks
    mp.setattr(S, "gpu_fb_change", recording)
    return solve(option, device, points)+(deltas,)


@pytest.mark.parametrize("points", ["gauss27", CORNERS])
@pytest.mark.parametrize("option", ["a", "b"])
@pytest.mark.parametrize("device", DEVICES)
def test_run_pg_sequence_bitwise(device, option, points, monkeypatch, record_property):
    with pytest.MonkeyPatch.context() as mp:
        backend, result, rows, peak, deltas = solve_recording(option, device, points, mp)
    R.patch(monkeypatch)
    ref_backend, ref_result, ref_rows, ref_peak, ref_deltas = solve_recording(option, device, points, monkeypatch)
    monkeypatch.undo()
    assert deltas and deltas == ref_deltas                # every device Delta F_B and block, longdouble, bitwise
    candidates = [r for r in rows if r["kind"] == "candidate"]
    evaluated_rejected = [r for r in candidates if r["energy"] is not None and not r["accepted"]]
    certificate_rejected = [r for r in candidates if r["cert_status"] != "certified_feasible"]
    assert len(candidates) >= 20 and evaluated_rejected and certificate_rejected
    assert len(rows) == len(ref_rows)
    for k, (a, b) in enumerate(zip(rows, ref_rows)):
        assert canonical(a) == canonical(b), (k, a.get("kind"))
    hashes = [r["state_hash"] for r in rows if r["kind"] in ("accept", "commit")]
    assert hashes and hashes == [r["state_hash"] for r in ref_rows if r["kind"] in ("accept", "commit")]
    assert [float(e).hex() for e in result.energies] == [float(e).hex() for e in ref_result.energies]
    assert canonical(list(result.gradient_norms)) == canonical(list(ref_result.gradient_norms))
    assert result.status == ref_result.status and result.commit.cert.state_hash == ref_result.commit.cert.state_hash
    assert result.commit.theta.tobytes() == ref_result.commit.theta.tobytes()
    assert canonical(backend.comparison_summary()) == canonical(ref_backend.comparison_summary())
    summary = backend.comparison_summary()
    assert summary["by_context"].get("pg", {}).get("comparisons", 0) > 0
    # review item 5: the device path (C2a cache in the comparison) decided at least once, and nothing fell back for a
    # device anomaly (which would bypass it)
    assert any(not r["fallback"] for r in backend.comparator.stats.records)
    assert summary["device_anomalies"] == dict(evaluate=0, compare=0)
    record = dict(device=device, option=option, points=points, candidates=len(candidates),
                  evaluated_rejected=len(evaluated_rejected), certificate_rejected=len(certificate_rejected),
                  comparisons=summary["comparisons"], cpu_fallback=summary["cpu_fallback"],
                  peak_bytes_opt1=peak, peak_bytes_pre_opt1=ref_peak)
    record_property("opt1_run_pg", json.dumps(record))
    print("OPT1 run_pg", json.dumps(record))


# ----------------------------------------------------------------------------------------------
# (v) cuda peak within the estimate
# ----------------------------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
@pytest.mark.parametrize("points", ["gauss27", CORNERS])
def test_cuda_peak_within_estimate_opt1(points, record_property):
    """One chunk of all 50 505 cells, no HVP cache: device state + evaluate (Jacobian cache, speculative field) +
    gradient + 2 HVP stays within estimate_peak_bytes(live_evaluations=1) with the OPT1 terms; the peak is recorded
    together with that of the pre-OPT1 options (cache and speculative gradient off)."""
    from ..model._helpers import free_random
    state, u, _ = case("medium")
    state = state.replace(logvol_points=points)
    cells = state.grid.n_cells
    v = free_random(state.grid, np.random.default_rng(7))
    peaks = {}
    for label, options in (("opt1", {}), ("pre_opt1", dict(jacobian_cache_slots=0, speculative_gradient=False))):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        ds = T.TorchModelState(state, "cuda", chunk_cells=cells, geometry_cache_bytes=0, **options)
        tev = T.evaluate(ds, u)
        T.gradient(ds, tev)
        T.hvp(ds, tev, v)
        T.hvp(ds, tev, v)
        torch.cuda.synchronize()
        peaks[label] = torch.cuda.max_memory_allocated()-base
        estimate = T.estimate_peak_bytes(state.grid.shape, state.node_index.size, state.n_channels,
                                         chunk_cells=ds.chunk_cells, geometry_cache_bytes=0, live_evaluations=1,
                                         corners=points == CORNERS, **options)
        assert peaks[label] <= estimate["total"], (label, peaks[label], estimate)
        del ds, tev
    record_property("opt1_cuda_peak", json.dumps(dict(points=points, cells=cells, **peaks)))
    print("OPT1 cuda peak", json.dumps(dict(points=points, cells=cells, **peaks)))
