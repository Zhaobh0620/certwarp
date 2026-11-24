"""ConvexAdam-v4seg (prereg §2.4): the coordinate convention between our arrays and convexAdam's, numpy only (the wrapper imports it in bl_convexadam, the tests in sci310).

Our arrays are (z, y, x) on the fixed grid, 1 mm, and a displacement d is (3, z, y, x) in voxels, pull-back: warped(p) = moving(p + d(p)) (project rules 评测约定).
convexAdam works on arrays in the Learn2Reg v1.1 layout (160, 224, 192) = ours.transpose(2, 1, 0) (m_convexadam.py:22-23), and its displacement disp is (H, W, D, 3) on that layout, in voxels along
that layout's axes: apply_convex warps with map_coordinates(moving, disp.transpose(3, 0, 1, 2) + identity, order 1) (src/convexAdam/apply_convex.py:13-23), so disp[..., c] is the displacement
along the layout's axis c, i.e. along our axis 2 - c.  convex_adam_nnUNet stores the same array, (H, W, D, 3), in disp.nii.gz (convex_adam_nnUNet.py:151-158).
Hence d = disp.transpose(3, 2, 1, 0)[::-1]: the same expression as m_convexadam.py:41 for the MIND arm.
"""
import numpy as np


def to_layout(a):
    """Ours (z, y, x) -> convexAdam's (x, y, z) layout (m_convexadam.py: ours.transpose(2, 1, 0))."""
    return np.ascontiguousarray(np.asarray(a).transpose(2, 1, 0))


def from_layout(a):
    return np.ascontiguousarray(np.asarray(a).transpose(2, 1, 0))


def d_from_disp(disp):
    """convexAdam's disp (H, W, D, 3) on the layout -> our d (3, z, y, x), voxels (m_convexadam.py:41, word for word)."""
    disp = np.asarray(disp)
    if disp.ndim != 4 or disp.shape[-1] != 3:
        raise ValueError(f'disp of shape {disp.shape}: (H, W, D, 3) expected')
    return np.ascontiguousarray(disp.transpose(3, 2, 1, 0)[::-1])


def disp_from_d(d):
    """The inverse of d_from_disp: our d (3, z, y, x) -> the layout's disp (H, W, D, 3).  For the tests' known fields."""
    d = np.asarray(d)
    return np.ascontiguousarray(d[::-1].transpose(3, 2, 1, 0))


def warp_layout(moving_layout, disp):
    """apply_convex's arithmetic (map_coordinates(moving, disp.transpose(3, 0, 1, 2) + identity, order 1), scipy's default mode constant 0) for the tests when convexAdam is not importable."""
    from scipy.ndimage import map_coordinates
    d1, d2, d3, _ = disp.shape
    identity = np.meshgrid(np.arange(d1), np.arange(d2), np.arange(d3), indexing='ij')
    return map_coordinates(np.asarray(moving_layout, dtype=np.float64), np.asarray(disp, dtype=np.float64).transpose(3, 0, 1, 2)+identity, order=1)
