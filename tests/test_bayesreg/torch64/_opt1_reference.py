"""The pre-OPT1 torch64 functions, kept verbatim as the reference of the OPT1 bitwise tests (work package OPT1).

Source: commit fd93113, bayesreg/backends/torch64/model.py (`evaluate`, `gradient`) and
bayesreg/backends/torch64/stable.py (`device_gauss_changes`, `gpu_fb_change`).  The four function bodies below are
byte-for-byte those of fd93113 (checked by test_opt1_bitwise.py::test_reference_is_verbatim_fd93113 when the repository
history is available); only the imports are new.  The helpers they call (_geometry, _corner_geometry, _logvol_scale,
_corner_scale, _corner_tau2, _tile_reduce, _channel_point, _channel_add, _finish, _require_finite, _check, _names,
curl_of, flux_of, stiffness_apply, node_field_cm) are unchanged by OPT1, so these functions compute exactly what the
code before OPT1 computed: an evaluation keeps no Jacobian cache and no speculative gradient, the comparison recomputes
the geometry of both states (gather, DGEMM, cofactor, det) and the gradient recomputes the geometry of its state.

`patch(monkeypatch)` installs them in place of the OPT1 functions (bayesreg.backends.torch64.evaluate / gradient, the
model module's evaluate / gradient, stable.gpu_fb_change / device_gauss_changes), so that the solver runs the pre-OPT1
device path.
"""
import numpy as np
import torch

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError

from bayesreg.model import BLOCK_NAMES
from bayesreg.model.blocks import nonpositive_gauss_jd, nonpositive_logvol_jd
from bayesreg.model.energy import _node_change_sum, _node_sum
from bayesreg.backends.torch64.model import (GAUSS_NAMES, LD, TorchEvaluation, TorchModelState, _channel_add,
                                             _channel_point, _check, _corner_geometry, _corner_scale, _corner_tau2,
                                             _finish, _geometry, _logvol_scale, _names, _require_finite, _tile_reduce,
                                             node_field_cm)
from bayesreg.backends.torch64.q1 import curl_of, flux_of, stiffness_apply
from bayesreg.backends.torch64.stable import DeviceAnomaly, _download, reduction_depth

FUNCTIONS = ("evaluate", "gradient", "device_gauss_changes", "gpu_fb_change")
SOURCES = dict(evaluate="bayesreg/backends/torch64/model.py", gradient="bayesreg/backends/torch64/model.py",
               device_gauss_changes="bayesreg/backends/torch64/stable.py",
               gpu_fb_change="bayesreg/backends/torch64/stable.py")
COMMIT = "fd93113"


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
    corners = dstate.logvol_corners
    cell_sums = torch.empty((4 if corners else 3, n_cells), dtype=torch.float64, device=dev)
    nonpositive = torch.empty(n_cells, dtype=torch.int64, device=dev)
    nonfinite = torch.empty(n_cells, dtype=torch.bool, device=dev)
    if corners:
        nonpositive_corners = torch.empty(n_cells, dtype=torch.int64, device=dev)
    sD, sC = dstate.scale_diffusion[None, None, None, :], dstate.scale_curl
    for chunk in q1.chunks:
        lo, hi = q1.cell_range(chunk)
        G, F, A, J = _geometry(dstate, u_cm, chunk)
        bad = ~torch.isfinite(J)
        for value in (G, A):
            bad |= ~torch.isfinite(value).all(dim=(0, 2))
        nonfinite[lo:hi] = bad.any(dim=1)
        nonpositive[lo:hi] = (J <= 0).sum(dim=1)
        rD = sD*G                                                   # Diffusion.residual
        cell_sums[0, lo:hi] = (rD*rD).sum(dim=(0, 2)).sum(dim=1)
        rJ = _logvol_scale(dstate, chunk)*torch.log(J)              # LogVolume.residual (J > 0 where used)
        cell_sums[1, lo:hi] = (rJ*rJ).sum(dim=1)
        curl = 0.
        for component in curl_of(G):                                # Curl.residual, sum over (point, i)
            rC = sC*component
            curl = curl+rC*rC
        cell_sums[2, lo:hi] = curl.sum(dim=1)
        if corners:                                                 # C05-2a: LogVolumeCorners.residual
            Gc, _, Ac, Jc = _corner_geometry(dstate, u_cm, chunk)
            bad = ~torch.isfinite(Jc)
            for value in (Gc, Ac):
                bad |= ~torch.isfinite(value).all(dim=(0, 2))
            nonfinite[lo:hi] |= bad.any(dim=1)
            nonpositive_corners[lo:hi] = (Jc <= 0).sum(dim=1)
            rJc = _corner_scale(dstate, chunk)*torch.log(Jc)
            cell_sums[3, lo:hi] = (rJc*rJc).sum(dim=1)
    tiles, n_tiles = state.tile_size, dstate.n_tiles
    tile_sums = dstate.device.download(_tile_reduce(cell_sums, tiles, n_tiles, n_cells)).T   # (n_tiles, 3 or 4)
    tile_nonpositive = dstate.device.download(_tile_reduce(nonpositive, tiles, n_tiles, n_cells))
    tile_nonfinite = dstate.device.download(_tile_reduce(nonfinite.to(torch.int64), tiles, n_tiles, n_cells))
    if corners:
        tile_nonpositive_corners = dstate.device.download(_tile_reduce(nonpositive_corners, tiles, n_tiles, n_cells))
    for t in range(n_tiles):
        if tile_nonfinite[t]:
            raise FloatingPointError("Gauss-point geometry: nonfinite numerical result (tile %d)" % t)
        if not corners:
            if tile_nonpositive[t]:
                return nonpositive_gauss_jd(int(tile_nonpositive[t]))
        elif tile_nonpositive[t] or tile_nonpositive_corners[t]:
            return nonpositive_logvol_jd(int(tile_nonpositive[t]), int(tile_nonpositive_corners[t]))
    r_host = dstate.device.download(point.r)
    r_host.flags.writeable = False
    with np.errstate(over="raise", invalid="raise"):
        channel = [_node_sum(state, r_host[c]*r_host[c]) for c in range(state.n_channels)]
        gauss = [LD(0) for _ in GAUSS_NAMES]
        for t in range(n_tiles):                                     # C06-7: ascending tiles
            for k in range(3):
                gauss[k] += LD(tile_sums[t, k])
            if corners:
                gauss[1] += LD(tile_sums[t, 3])                      # C05-2a: right after the tile's Gauss sum
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
    tile_sums = np.ascontiguousarray(tile_sums)
    tile_sums.flags.writeable = False
    return TorchEvaluation(dstate, u_cm, point, total, energy, energies, r_host, tile_sums)


def gradient(dstate, ev, *, blocks=BLOCK_NAMES):
    """C06-4: grad F_B = sum_i K_i^T r_i at ev.u as the device free vector; `blocks` selects terms."""
    _check(dstate, ev)
    names = _names(blocks)
    state = dstate.state
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
                gJ = sJ*(sJ*torch.log(J))/J                         # LogVolume.cotangent(residual)
            if curl:
                gcurl = tuple(sC*(sC*component) for component in curl_of(G))     # Curl.cotangent(residual)
            dstate.q1.scatter_add(out, flux_of(A, gJ, gcurl, G), chunk)
            if logvol and dstate.logvol_corners:                    # C05-2a, after the chunk's Gauss points
                Gc, _, Ac, Jc = _corner_geometry(dstate, ev.u_cm, chunk)
                sJc = _corner_scale(dstate, chunk)
                gJc = sJc*(sJc*torch.log(Jc))/Jc                     # LogVolumeCorners.cotangent(residual)
                dstate.q1_corners.scatter_add(out, flux_of(Ac, gJc, None, Gc), chunk)
    g = _finish(out, "gradient of F_B")
    if "diffusion" in names:
        g = g+(state.alpha/state.omega0)*stiffness_apply(ev.u_cm, state.grid)
    return _require_finite(g, "gradient of F_B")


def device_gauss_changes(dstate, base, trial):
    """C08-8(a)(b): (n_tiles, 3) host array of the device FP64 tile sums of the elementwise changes of the
    diffusion, log-volume and curl blocks (columns in that order; under C05-2a a fourth column, the corner part of
    the log-volume block).  Raises DeviceAnomaly on a device J <= 0 (Gauss point or corner), a nonfinite geometry
    or tile sum, or a device exception (OOM included)."""
    if not isinstance(dstate, TorchModelState):
        raise InputValidationError("dstate: TorchModelState required")
    for ev in (base, trial):
        if not isinstance(ev, TorchEvaluation) or ev.state is not dstate:
            raise InputValidationError("C08-8: two TorchEvaluations of this TorchModelState required")
    reduction_depth(dstate.state.tile_size)
    corners = dstate.logvol_corners
    try:
        q1 = dstate.q1
        n_cells = dstate.grid.n_cells
        dev = base.u_cm.device
        cell_sums = torch.empty((4 if corners else 3, n_cells), dtype=torch.float64, device=dev)
        nonfinite = torch.zeros((), dtype=torch.int64, device=dev)
        nonpositive = torch.zeros((), dtype=torch.int64, device=dev)
        sD, sC = dstate.scale_diffusion[None, None, None, :], dstate.scale_curl
        omega0 = dstate.state.omega0
        for chunk in q1.chunks:
            lo, hi = q1.cell_range(chunk)
            Gb, _, Ab, Jb = _geometry(dstate, base.u_cm, chunk)
            Gt, _, At, Jt = _geometry(dstate, trial.u_cm, chunk)
            for value in (Gb, Ab, Jb, Gt, At, Jt):
                nonfinite += (~torch.isfinite(value)).sum()
            nonpositive += (Jb <= 0).sum()+(Jt <= 0).sum()
            rb, rt = sD*Gb, sD*Gt
            cell_sums[0, lo:hi] = ((rt-rb)*(rt+rb)).sum(dim=(0, 2)).sum(dim=1)
            weight = q1.weights/(omega0*q1.interpolate(dstate.tau2_nodes, chunk))     # omega w_g/(omega0 tau_q^2)
            x = (Jt-Jb)/Jb
            ell = torch.where(x >= -.5, torch.log1p(x), torch.log(Jt/Jb))
            cell_sums[1, lo:hi] = (weight*ell*(torch.log(Jt)+torch.log(Jb))).sum(dim=1)
            curl = None
            for cb, ct in zip(curl_of(Gb), curl_of(Gt)):
                b, t = sC*cb, sC*ct
                d = (t-b)*(t+b)
                curl = d if curl is None else curl+d
            cell_sums[2, lo:hi] = curl.sum(dim=1)
            if corners:                                                 # C05-2a: the corner log-volume term
                Gb, _, Ab, Jb = _corner_geometry(dstate, base.u_cm, chunk)
                Gt, _, At, Jt = _corner_geometry(dstate, trial.u_cm, chunk)
                for value in (Gb, Ab, Jb, Gt, At, Jt):
                    nonfinite += (~torch.isfinite(value)).sum()
                nonpositive += (Jb <= 0).sum()+(Jt <= 0).sum()
                weight = dstate.corner_weights/(omega0*_corner_tau2(dstate, chunk))   # beta_c omega/(8 omega0 tau^2)
                x = (Jt-Jb)/Jb
                ell = torch.where(x >= -.5, torch.log1p(x), torch.log(Jt/Jb))
                cell_sums[3, lo:hi] = (weight*ell*(torch.log(Jt)+torch.log(Jb))).sum(dim=1)
        sums = _tile_reduce(cell_sums, dstate.state.tile_size, dstate.n_tiles, n_cells)
        flags = _download(dstate, torch.stack((nonfinite, nonpositive, (~torch.isfinite(sums)).sum())))
        tiles = _download(dstate, sums).T
    except DeviceAnomaly:
        raise
    except (FloatingPointError, RuntimeError) as exc:                # torch.cuda.OutOfMemoryError included
        raise DeviceAnomaly(f"device exception {type(exc).__name__}: {exc}") from exc
    if flags[1]:
        where = "Gauss points and cell corners (C05-2a)" if corners else "Gauss points"
        raise DeviceAnomaly(f"device J_q <= 0 at {int(flags[1])} {where} of the compared states")
    if flags[0] or flags[2]:
        raise DeviceAnomaly(f"device nonfinite values ({int(flags[0])} geometry, {int(flags[2])} tile sums)")
    return np.ascontiguousarray(tiles)


def gpu_fb_change(dstate, base, trial):
    """Delta F_B of the GPU path (C08-8(a)(b)): (longdouble delta, blocks).  The channel blocks on the host from
    the downloaded channel residuals (C08-1), the Gauss-point blocks from `device_gauss_changes`."""
    state = dstate.state
    channel = [_node_change_sum(state, base.r_host[c], trial.r_host[c]) for c in range(state.n_channels)]
    tiles = device_gauss_changes(dstate, base, trial)
    corners = dstate.logvol_corners
    with np.errstate(over="raise", invalid="raise"):
        gauss = [LD(0) for _ in GAUSS_NAMES]
        for t in range(tiles.shape[0]):                                  # ascending tiles
            for k in range(3):
                gauss[k] += LD(tiles[t, k])
            if corners:
                gauss[1] += LD(tiles[t, 3])                              # C05-2a: right after the tile's Gauss sum
        half = LD(.5)
        changes = [half*s for s in channel]+[half*s for s in gauss]
        delta = LD(0)
        for change in changes:
            delta += change
    blocks = dict(channel=tuple(changes[:state.n_channels]),
                  **{name: change for name, change in zip(GAUSS_NAMES, changes[state.n_channels:])})
    return delta, blocks


def patch(monkeypatch):
    """Run the pre-OPT1 device path: evaluate, gradient and the C08-8 device comparison of commit fd93113."""
    import bayesreg.backends.torch64 as T
    from bayesreg.backends.torch64 import model as TM
    from bayesreg.backends.torch64 import stable as S
    for module in (T, TM):
        monkeypatch.setattr(module, "evaluate", evaluate)
        monkeypatch.setattr(module, "gradient", gradient)
    monkeypatch.setattr(S, "device_gauss_changes", device_gauss_changes)
    monkeypatch.setattr(S, "gpu_fb_change", gpu_fb_change)
