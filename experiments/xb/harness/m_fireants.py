"""BL2 smoke, FireANTs arms (env bl_fireants, fused ops). Usage: python -I -B m_fireants.py {greedy|syn} PAIR_NPZ OUT_DIR [K]
BL2VAL (BL2 Amendment 4): optional integer K >= 1 = the equal-time arms' factor: every scale's iteration count is multiplied by K
(scales, optimizer, lr, kernels, smoothing and the package's convergence test unchanged). With K the warm repeat is skipped (one
registration per run, as Amendment 4's cost assumes) and the iterations actually run per scale are counted by wrapping the
registration's own ConvergenceMonitor.reset/converged (the originals are called unchanged). Without K: as the BL2 smoke.
Configuration and export exactly as env/bl_fireants/smoke_cmd.md (= BL1 bl1_register.py, authors' OASIS test configs).
Writes OUT_DIR/{fixed,moving}.nii.gz, disp.npz (d, our convention), own_warped.npz (w: FireANTs' own warped moving,
reg.evaluate, our layout), method_out.json. Cold run exported; a warm repeat in the same process is timed only."""
import json
import os
import sys
import time
from pathlib import Path

T_LAUNCH = float(os.environ['BL2_T_LAUNCH'])
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402
import torch  # noqa: E402
import fireants  # noqa: E402
import fireants.interpolator as fint  # noqa: E402
from fireants.io.image import Image, BatchedImages  # noqa: E402
from fireants.registration.greedy import GreedyRegistration  # noqa: E402
from fireants.registration.syn import SyNRegistration  # noqa: E402

algo, inp, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
assert algo in ('greedy', 'syn')
K = int(sys.argv[4]) if len(sys.argv) > 4 else None                          # BL2VAL: equal-time factor (None = default arm)
assert K is None or K >= 1, K
ITERS = [i*(K or 1) for i in ([250, 200, 100] if algo == 'greedy' else [200, 100, 50])]
RUN_ITERS = []                                                               # BL2VAL: [iterations run, stopped by its own test] per scale
assert fint.FFO_AVAILABLE and fint.fireants_interpolator._use_ffo, 'fused ops not active'
t_imports = time.time()
with np.load(inp) as z:
    fixed, moving = np.array(z['fixed']), np.array(z['moving'])               # (z, y, x)


def write_nii(arr, path):
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr.astype(np.float32)))
    img.SetSpacing((1., 1., 1.)); img.SetOrigin((0., 0., 0.)); img.SetDirection((1., 0., 0., 0., 1., 0., 0., 0., 1.))
    sitk.WriteImage(img, str(path))


write_nii(fixed, out/'fixed.nii.gz'); write_nii(moving, out/'moving.nii.gz')
fi = BatchedImages(Image.load_file(str(out/'fixed.nii.gz')))
mi = BatchedImages(Image.load_file(str(out/'moving.nii.gz')))
torch.cuda.synchronize()
t_ready = time.time()


def run():
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if algo == 'greedy':
        reg = GreedyRegistration([4, 2, 1], ITERS, fi, mi, deformation_type='compositive', optimizer='adam',
                                 optimizer_lr=.25, cc_kernel_size=7, optimizer_params={'beta1': .1, 'scaledown': False},
                                 smooth_grad_sigma=1, smooth_warp_sigma=.25)
    else:
        reg = SyNRegistration([4, 2, 1], ITERS, fi, mi, deformation_type='compositive', optimizer='adam',
                              optimizer_lr=.1, cc_kernel_size=5, smooth_grad_sigma=.5, smooth_warp_sigma=.25)
    if K is not None:                                                        # BL2VAL: count, do not change, the convergence test
        mon = reg.convergence_monitor; _reset, _conv = mon.reset, mon.converged

        def reset():
            RUN_ITERS.append([0, False]); return _reset()

        def converged(loss):
            c = _conv(loss); RUN_ITERS[-1][0] += 1; RUN_ITERS[-1][1] = bool(c); return c
        mon.reset, mon.converged = reset, converged
    reg.optimize()
    torch.cuda.synchronize(); secs = time.perf_counter()-t0
    with torch.no_grad():
        grid = reg.get_warped_coordinates(fi, mi)                             # (1, D, H, W, 3), normalized, align_corners=True
        moved = reg.evaluate(fi, mi)
    g = grid[0].double().cpu().numpy(); own = moved[0, 0].double().cpu().numpy()
    del reg, grid, moved
    return secs, g, own


def to_d(g):
    shape = mi().shape[2:]                                                    # (D, H, W) = (z, y, x)
    idx = np.stack([(g[..., 2]+1)/2*(shape[0]-1), (g[..., 1]+1)/2*(shape[1]-1), (g[..., 0]+1)/2*(shape[2]-1)])
    ident = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij')).astype(np.float64)
    return idx-ident


torch.cuda.reset_peak_memory_stats()
secs_cold, g, own = run()
t_out = time.time()
peak_cold = torch.cuda.max_memory_allocated()
d = to_d(g); del g
assert d.shape == (3,)+fixed.shape
np.savez_compressed(out/'disp.npz', d=d.astype(np.float32))
np.savez_compressed(out/'own_warped.npz', w=own.astype(np.float32))
if K is None:
    torch.cuda.empty_cache()
    secs_warm, g2, _ = run()
    d2 = to_d(g2); del g2
    rep = float(np.abs(d2-d).max())
    rep_fg = float(np.sqrt(((d2-d)**2).sum(0))[fixed > 0].mean())
else:                                                                        # BL2VAL: no warm repeat in the equal-time arms
    secs_warm = rep = rep_fg = None
cfg = (dict(cls='GreedyRegistration', scales=[4, 2, 1], iterations=ITERS, deformation_type='compositive', optimizer='adam',
            optimizer_lr=.25, cc_kernel_size=7, optimizer_params={'beta1': .1, 'scaledown': False}, smooth_grad_sigma=1,
            smooth_warp_sigma=.25, loss='cc (default)')
       if algo == 'greedy' else
       dict(cls='SyNRegistration', scales=[4, 2, 1], iterations=ITERS, deformation_type='compositive', optimizer='adam',
            optimizer_lr=.1, cc_kernel_size=5, smooth_grad_sigma=.5, smooth_warp_sigma=.25, loss='cc (default)'))
(out/'method_out.json').write_text(json.dumps(dict(
    method=f'fireants_{algo}_fused', config=cfg, use_ffo=True, fireants_file=fireants.__file__, torch=torch.__version__,
    device=f'cuda ({torch.cuda.get_device_name(0)})', threads=dict(torch_intra_op=torch.get_num_threads()),
    seconds_registration_cold=secs_cold, seconds_registration_warm=secs_warm, seconds_total=t_out-T_LAUNCH,
    seconds_launch_to_imports=t_imports-T_LAUNCH, seconds_launch_to_ready=t_ready-T_LAUNCH,
    timer_scope='registration object construction + optimize() (as BL1); images on GPU and CUDA context before the timer',
    cuda_peak_allocated_bytes_cold=int(peak_cold), cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
    repeat_max_abs_diff_voxels=rep, repeat_mean_norm_diff_fixed_fg_voxels=rep_fg, warnings=[],
    **({} if K is None else dict(equal_time=dict(
        k=K, iterations=ITERS, iterations_run_per_scale=[n for n, _ in RUN_ITERS], stopped_by_own_test_per_scale=[c for _, c in RUN_ITERS],
        convergence_test='package default (ConvergenceMonitor: tolerance 1e-6 over 10 iterations), unchanged',
        warm_repeat='skipped (BL2VAL: one registration per run)')))), indent=1))
print(f'{algo}: cold {secs_cold:.2f} s, ' + (f'warm {secs_warm:.2f} s, ' if K is None else f'k {K}, iterations {ITERS}, run {RUN_ITERS}, ')
      + f'total {t_out-T_LAUNCH:.2f} s, repeat diff {rep if rep is None else format(rep, ".3e")}', flush=True)
