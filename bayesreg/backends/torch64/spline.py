"""FP64 tensor-product not-a-knot cubic spline evaluation on the device (contract C03.5-10, C04-1).

Copied from the mainline torch spline path:
    source  archive/seq_vp_3d/jd_curl/three_d/backends/spline.py (cubic_basis lines 12-34; TorchSpline
            lines 37-41 and 81-127, the torch branch of `sample`)
    commit  eb808be401fde7ed6fee73c2d86212c8c463a513, file sha256
            9c876348dc3a410cb946d0ea1e299d83270bc682a9b7a567ad6f34b21a6cbfde
`cubic_basis` is verbatim.  TorchSpline keeps the frozen SciPy coefficients and knots of one
`interpolation.build_image_sampler` (one object per channel), the bound checks and the per-chunk
gather and contraction of the torch path, operation for operation.  Not copied: the E99 fused Triton
kernel (`fused`, `_sample_fused`, `_torch_chunk`), the FP32 screening sampler (`sample_f32`) and the
`work_control.checkpoint` calls.  Added: `sample_channels`, which evaluates the shared cubic bases once
per chunk for all channels of one grid (all channels share the knots) and then runs the same per-channel
gather and contraction, so its values are bitwise those of `TorchSpline.sample` per channel (tested).

No grid_sample, image prefilter, padding, extrapolation or coordinate clipping.  The span is restricted
only at the closed upper endpoint, not the query itself.  Reference: `jd_curl.three_d.interpolation.
sample_points` (scipy NdBSpline); the two agree to rounding (L0 test, relative <= 1e-12).
"""
import torch

from jd_curl.validation import InputValidationError

SPLINE_CHUNK = 131072     # query points per chunk: (chunk, 4, 4, 4) float64 temporaries of 67 MB each


def cubic_basis(t, x, n):
    span = (torch.searchsorted(t, x.contiguous(), right=True)-1).clamp(3, n-1)
    basis = torch.ones((x.numel(), 1), dtype=x.dtype, device=x.device)
    quadratic = None
    for degree in range(1, 4):
        values, saved = [], torch.zeros_like(x)
        for r in range(degree):
            right = t[span+r+1]-x
            left = x-t[span+1-degree+r]
            temp = basis[:, r]/(right+left)
            values.append(saved+right*temp)
            saved = left*temp
        values.append(saved)
        basis = torch.stack(values, dim=-1)
        if degree == 2:
            quadratic = basis
    indices = span[:, None]+torch.arange(-3, 1, device=x.device)
    zero = torch.zeros_like(x[:, None])
    a, b = t[indices+3]-t[indices], t[indices+4]-t[indices+1]
    # Repeated endpoint knots give zero coefficients, not divisions by zero.
    derivative = (3*torch.cat((zero, quadratic), dim=-1)/torch.where(a != 0, a, 1)
                  -3*torch.cat((quadratic, zero), dim=-1)/torch.where(b != 0, b, 1))
    return indices, basis, derivative


def _contract_chunk(c, bases):
    """The mainline per-chunk gather and contraction: value and xyz gradient at the chunk's points."""
    iz, bz, dz = bases[0]
    iy, by, dy = bases[1]
    ix, bx, dx = bases[2]
    coeff = c[iz[:, :, None, None], iy[:, None, :, None], ix[:, None, None, :]]

    def contract(z, y, x):
        return (coeff*z[:, :, None, None]*y[:, None, :, None]*x[:, None, None, :]).sum(dim=(1, 2, 3))
    value = contract(bz, by, bx)
    gradient = torch.stack((contract(bz, by, dx), contract(bz, dy, bx), contract(dz, by, bx)), dim=-1)
    return value, gradient


class TorchSpline:
    def __init__(self, sampler, upload):
        self._knots = tuple(upload(t) for t in sampler._spline.t)
        self._c = upload(sampler._spline.c)
        self._bounds = tuple((float(a[0]), float(a[-1])) for a in sampler.grid.axes)
        self.grid_id = sampler.grid.grid_id
        self.image_id = sampler.image_id

    def _check(self, points, chunk_size, strict):
        if (not isinstance(points, torch.Tensor) or points.dtype != torch.float64
                or points.device != self._c.device or points.shape[-1:] != (3,) or points.requires_grad):
            raise InputValidationError("spline requires explicit device float64 xyz points, without autograd")
        if not bool(torch.isfinite(points).all()):
            raise InputValidationError("nonfinite spline query")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise InputValidationError("positive integer spline chunk required")
        valid = torch.ones(points.shape[:-1], dtype=torch.bool, device=points.device)
        if strict is not None and (strict.shape != valid.shape or strict.dtype != torch.bool or strict.device != points.device):
            raise InputValidationError("strict query mask mismatch")
        for r, (lo, hi) in enumerate(self._bounds):
            valid &= (points[..., r] >= lo) & (points[..., r] <= hi)
            if strict is not None:
                valid &= ~strict | ((points[..., r] > lo) & (points[..., r] < hi))
        if not bool(valid.all()):
            raise InputValidationError("spline query outside permitted support")

    def bases(self, flat):
        """cubic_basis along z, y, x (coefficient axes) for (n, 3) xyz points."""
        return [cubic_basis(t, flat[:, 2-axis], self._c.shape[axis]) for axis, t in enumerate(self._knots)]

    def sample(self, points, *, chunk_size=SPLINE_CHUNK, strict=None):
        self._check(points, chunk_size, strict)
        flat = points.reshape(-1, 3)
        value = torch.empty(flat.shape[0], device=points.device, dtype=points.dtype)
        gradient = torch.empty_like(flat)
        for start in range(0, len(flat), chunk_size):
            end = min(start+chunk_size, len(flat))
            value[start:end], gradient[start:end] = _contract_chunk(self._c, self.bases(flat[start:end]))
        return value.reshape(points.shape[:-1]), gradient.reshape(points.shape)


def shared_knots(splines):
    """True when every spline has the same knot vectors as the first (same grid: always, in practice)."""
    first = splines[0]._knots
    return all(len(s._knots) == len(first) and all(a.shape == b.shape and bool(torch.equal(a, b))
                                                    for a, b in zip(s._knots, first)) for s in splines[1:])


def sample_channels(splines, points, *, chunk_size=SPLINE_CHUNK, strict=None, shared=None):
    """All channels at the same (n, 3) points: values (C, n), gradients (C, n, 3).

    With shared knots (checked once by the caller via `shared`, or here) the bases are computed once per
    chunk and the per-channel gather and contraction are those of TorchSpline.sample, so the result is
    bitwise [s.sample(points) for s in splines]; otherwise each channel is sampled separately.
    """
    if not splines:
        raise InputValidationError("sample_channels: at least one spline required")
    splines[0]._check(points, chunk_size, strict)
    flat = points.reshape(-1, 3)
    n_channels = len(splines)
    values = torch.empty((n_channels, flat.shape[0]), device=points.device, dtype=torch.float64)
    gradients = torch.empty((n_channels, flat.shape[0], 3), device=points.device, dtype=torch.float64)
    if shared is None:
        shared = shared_knots(splines)
    if not shared:
        for c, spline in enumerate(splines):
            values[c], gradients[c] = spline.sample(points.reshape(-1, 3), chunk_size=chunk_size,
                                                    strict=None if strict is None else strict.reshape(-1))
        return values, gradients
    for start in range(0, len(flat), chunk_size):
        end = min(start+chunk_size, len(flat))
        bases = splines[0].bases(flat[start:end])
        for c, spline in enumerate(splines):
            values[c, start:end], gradients[c, start:end] = _contract_chunk(spline._c, bases)
    return values, gradients
