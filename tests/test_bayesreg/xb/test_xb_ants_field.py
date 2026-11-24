"""(丁) (BZ 2026-11-15 ~4:20 AM CST, decision_log that day, item (5); the unseal package draft docs/bayesreg/notes/解封包草稿_20261115.md section 6; its revision-1 difference review
docs/bayesreg/reviews/解封包草稿_修订1_差异复核_20261115.md, minor 4): a synthetic test of the conversion of ANTs' exported displacement field into our convention in xb_core.arm_ants
(at P0 af593248: the arm :853-889, the physical-to-voxel and xyz-to-zyx step :880-883).  Before this file no test of tests/ ran arm_ants past its first step (test_xb_tools.py's coordinate
tests are ConvexAdam's).

ctx.step is replaced by FakeAnts; no ANTs binary runs.  Its 'register' step leaves the files antsRegistrationSyN.sh -t s leaves (the archive's
archive/seq_vp_3d/results/verification/3d_stage_BL2_v1/env/ants/smoke_cmd.md:25-26): <P>0GenericAffine.mat (here the identity), <P>1Warp.nii.gz, <P>1InverseWarp.nii.gz (which arm_ants's
glob <P>[0-9]Warp.nii.gz must not take) and <P>Warped.nii.gz; its 'export_disp' step writes the
DisplacementField[<P>disp.nii.gz] antsApplyTransforms would write: a SimpleITK vector image of a known physical field u (array (z, y, x, 3), the components ITK's physical (x, y, z)) on the grid
of the fixed.nii.gz arm_ants wrote; its 'warp_aat' step the linear warp.  The warps are SimpleITK's own resampling of moving.nii.gz through a DisplacementFieldTransform of u (ITK: T(x) = x + u(x),
output(x) = moving(T(x))).  What is checked:
  - the d arm_ants returns equals the hand-computed d = (u_z, u_y, u_x) on the first axis: shape (3, nz, ny, nx), voxels (disp.npz's key d);
  - with that d our pull-back warp (warped(p) = moving(p + d(p)); map_coordinates order 1, constant 0, the export check's) reproduces SimpleITK's resampling on the whole grid (the
    images are 0 near the faces) and the export check passes (corr > .9999); a d with its components not reversed, or with its sign flipped, does not reproduce it and the export check
    calls it suspicious (the check tells the conventions apart on these fields).

What it cannot check (stated as the difference review's minor 4 asks): on the production path the fixed.nii.gz XB writes for ANTs always has the identity direction, spacing 1 and origin 0
(xb_core.write_sitk_identity, :829-838), so the M of :881 is the identity and this test checks only the axis order and the sign arithmetic of :880-883 (an M transposed or not inverted would
pass it).  It does NOT check the direction semantics of ANTs' own displacement field: in which frame, component order and sign antsApplyTransforms (ANTs 2.6.5's ITK) writes the vectors into
the NIfTI and how sci310's SimpleITK reads them back; here the writer and the reader are both SimpleITK.  Only the per-pair export check (corr >= .999 inside fixed > 0, prereg section 4.5;
P19 stops the arm at the third suspicious pair) can find that, e.g. the ants_s cells of D3 on the development set."""
from pathlib import Path
import re

import numpy as np
import pytest

from . import _world as W

xc = W.xc
P = 'ants_s_'                                    # arm_ants's prefix for the frozen arm ants_s (f"ants_{cfg['variant']}_")
SHAPE = (16, 18, 14)                             # (nz, ny, nx): three different sizes
BORDER = 2                                       # the images are 0 within BORDER voxels of every face: ITK's half-voxel border rule (the edge value) and map_coordinates' constant 0 then agree


def grid(shape=SHAPE):
    return np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing='ij')


def images(shape=SHAPE):
    """fixed, moving (float32): off-centre anisotropic blobs, 0 near the faces."""
    z, y, x = grid(shape)
    inner = np.all([(a >= BORDER) & (a <= n-1-BORDER) for a, n in zip((z, y, x), shape)], axis=0)

    def blob(cz, cy, cx, sz, sy, sx):
        return np.exp(-(((z-cz)/sz)**2+((y-cy)/sy)**2+((x-cx)/sx)**2))
    fixed = (blob(7., 9., 6., 3., 4., 2.5)+.4*blob(10., 5., 9., 2., 2., 2.)+.05)*inner
    moving = (blob(8., 8., 7.5, 3., 4.5, 2.5)+.5*blob(5., 12., 5., 2., 2.5, 2.))*inner
    return fixed.astype(np.float32), moving.astype(np.float32)


def known_u(kind, shape=SHAPE):
    """A physical field u, (z, y, x, 3), components (x, y, z): the components differ from each other and, for 'varying', each depends on another array axis."""
    z, y, x = grid(shape)
    ux, uy, uz = np.full(shape, 2.1), np.full(shape, -1.3), np.full(shape, .7)
    if kind == 'varying':
        ux, uy, uz = ux+.05*(x-(shape[2]-1)/2), uy+.04*(z-(shape[0]-1)/2), uz-.03*(y-(shape[1]-1)/2)
    return np.stack([ux, uy, uz], axis=-1)


class FakeAnts:
    """The three steps arm_ants runs (register, export_disp, warp_aat), as the binaries would leave their files, from the known field u."""

    def __init__(self, u):
        self.u, self.calls = u, []

    def field(self, ref, sign=1.):
        import SimpleITK as sitk
        img = sitk.GetImageFromArray(sign*self.u, isVector=True)
        img.CopyInformation(ref)
        return img

    def warp(self, work, ref, out):
        import SimpleITK as sitk
        tx = sitk.DisplacementFieldTransform(sitk.Cast(self.field(ref), sitk.sitkVectorFloat64))
        sitk.WriteImage(sitk.Resample(sitk.ReadImage(str(work/'moving.nii.gz')), ref, tx, sitk.sitkLinear, 0.), str(work/out))

    def __call__(self, name, cmd, env, cwd, logfile, gpu, *, policy=None, log=print, **kwargs):
        import SimpleITK as sitk
        cmd = [str(c) for c in cmd]
        self.calls.append(dict(name=name, cmd=cmd))
        work = Path(cwd)
        ref = sitk.ReadImage(str(work/'fixed.nii.gz'))                  # the grid arm_ants wrote (write_inputs_sitk)
        out = cmd[cmd.index('-o')+1]
        if name == 'register':
            sitk.WriteImage(self.field(ref), str(work/f'{out}1Warp.nii.gz'))
            sitk.WriteImage(self.field(ref, -1.), str(work/f'{out}1InverseWarp.nii.gz'))
            sitk.WriteTransform(sitk.AffineTransform(3), str(work/f'{out}0GenericAffine.mat'))
            self.warp(work, ref, f'{out}Warped.nii.gz')
        elif name == 'export_disp':
            sitk.WriteImage(self.field(ref), str(work/re.fullmatch(r'DisplacementField\[(.+)\]', out).group(1)))
        elif name == 'warp_aat':
            self.warp(work, ref, out)
        else:
            raise AssertionError(f'arm_ants ran an unexpected step {name}')
        return dict(name=name, cmd=cmd, cwd=str(cwd), returncode=0, seconds_wall=.1, min_mem_available_gb=10., memory_guard_killed=False, memory_events_before=None,
                    memory_events_after=None, kernel_oom_lines=None, fd_spy=dict(passed=True, samples=1, breaches=[]), spy_breach=None)


def run_arm(tmp_path, u):
    fixed, moving = images()
    work = tmp_path/'work'
    work.mkdir()
    fake = FakeAnts(u)
    ctx = xc.Ctx(policy={}, log=lambda *a: None, step=fake, precheck=W.fake_precheck)
    d, own, alts, out = xc.arm_ants('ants_s', xc.ARMS['ants_s'], work, work/'method_log.txt', dict(fixed=fixed, moving=moving), ctx)
    return fixed, moving, d, own, alts, out, fake


@pytest.mark.parametrize('kind', ['translation', 'varying'])
def test_arm_ants_turns_a_known_ants_field_into_our_convention(tmp_path, kind):
    """The hand-computed d: ITK's physical (x, y, z) components, with M = I, are voxels along (x, y, z); ours is (z, y, x) on the first axis, the same sign (both pull back)."""
    u = known_u(kind)
    _, _, d, _, _, out, fake = run_arm(tmp_path, u)
    assert [c['name'] for c in fake.calls] == ['register', 'export_disp', 'warp_aat']
    assert out['transforms'] == ['-t', f'{P}1Warp.nii.gz', '-t', f'{P}0GenericAffine.mat']           # the forward warp, not 1InverseWarp; the warp first, then the affine
    export = fake.calls[1]['cmd']
    assert export[export.index('-r')+1] == 'fixed.nii.gz' and export[export.index('-o')+1] == f'DisplacementField[{P}disp.nii.gz]'
    by_hand = np.stack([u[..., 2], u[..., 1], u[..., 0]])        # d[0] = u_z, d[1] = u_y, d[2] = u_x
    assert d.shape == (3,)+SHAPE and d.dtype == np.float64
    assert np.allclose(d, by_hand, rtol=0, atol=1e-12)
    if kind == 'translation':
        assert np.allclose(d, np.array([.7, -1.3, 2.1]).reshape(3, 1, 1, 1), rtol=0, atol=1e-12)     # (u_x, u_y, u_z) = (2.1, -1.3, .7) written out


@pytest.mark.parametrize('kind', ['translation', 'varying'])
def test_arm_ants_d_pulls_back_as_itks_displacement_field_transform(tmp_path, kind):
    """warped(p) = moving(p + d(p)) with arm_ants's d is SimpleITK's resampling through the field (the arm's own warped and the antsApplyTransforms one), and the export check
    (xb_core.export_check) passes; with the components not reversed, or the sign flipped, it is not, and the export check calls it suspicious."""
    fixed, moving, d, own, alts, _, _ = run_arm(tmp_path, known_u(kind))
    assert set(alts) == {'antsApplyTransforms_linear'}
    for theirs in (own, alts['antsApplyTransforms_linear']):
        assert np.allclose(W.warp(moving, d), theirs, rtol=0, atol=1e-5)
    rec = xc.export_check(fixed, moving, d, own, alts)
    assert not rec['suspicious'] and rec['check']['corr'] > .9999 and rec['check_alt']['antsApplyTransforms_linear']['corr'] > .9999
    for wrong in (d[::-1], -d):                    # (u_x, u_y, u_z) on the first axis; the push-forward sign
        assert not np.allclose(W.warp(moving, wrong), own, rtol=0, atol=1e-3)
        assert xc.export_check(fixed, moving, np.ascontiguousarray(wrong), own, alts)['suspicious']
