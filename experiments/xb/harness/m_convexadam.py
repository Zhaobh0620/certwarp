"""BL2 smoke, ConvexAdam (MIND-SSC) arm (env bl_convexadam). Usage: python -I -B m_convexadam.py PAIR_NPZ OUT_DIR
Configuration = package defaults (env/convexadam/smoke_cmd.md): mind_r 1, mind_d 2, lambda 1.25, grid_sp 6, disp_hw 4, 80 Adam
iterations, grid_sp_adam 2, smoothing 0, ic True, no mask, float16. Input in the Learn2Reg v1.1 layout = ours.transpose(2, 1, 0).
Writes disp.npz (our convention), own_warped.npz (apply_convex warp, our layout), method_out.json. Cold run exported; warm
repeat timed only."""
import json
import os
import sys
import time
from pathlib import Path

T_LAUNCH = float(os.environ['BL2_T_LAUNCH'])
import numpy as np  # noqa: E402
import torch  # noqa: E402
import convexAdam  # noqa: E402
from convexAdam.convex_adam_MIND import convex_adam_pt  # noqa: E402
from convexAdam.apply_convex import apply_convex  # noqa: E402

inp, out = Path(sys.argv[1]), Path(sys.argv[2])
t_imports = time.time()
with np.load(inp) as z:
    fixed, moving = np.array(z['fixed'], np.float64), np.array(z['moving'], np.float64)    # (z, y, x) = (192, 224, 160)
F_l2r, M_l2r = fixed.transpose(2, 1, 0).copy(), moving.transpose(2, 1, 0).copy()         # Learn2Reg v1.1 layout (160, 224, 192)
dev = torch.device('cuda')
torch.zeros(1, device=dev); torch.cuda.synchronize()                                     # CUDA context outside the timer
t_ready = time.time()
PARAMS = dict(mind_r=1, mind_d=2, lambda_weight=1.25, grid_sp=6, disp_hw=4, selected_niter=80, selected_smooth=0, grid_sp_adam=2, ic=True,
              use_mask=False, dtype=torch.float16, verbose=False, device=dev)


def run():
    torch.cuda.synchronize(); t0 = time.perf_counter()
    disp = convex_adam_pt(img_fixed=F_l2r, img_moving=M_l2r, **PARAMS)
    torch.cuda.synchronize()
    return time.perf_counter()-t0, disp


torch.cuda.reset_peak_memory_stats()
secs_cold, disp = run()
t_out = time.time()
assert disp.shape == F_l2r.shape+(3,), disp.shape
d = np.ascontiguousarray(disp.transpose(3, 2, 1, 0)[::-1])                                # (3, z, y, x), our axes, voxels
assert d.shape == (3,)+fixed.shape
own = apply_convex(disp=disp, moving=M_l2r).transpose(2, 1, 0)
np.savez_compressed(out/'disp.npz', d=d.astype(np.float32))
np.savez_compressed(out/'own_warped.npz', w=own.astype(np.float32))
peak_cold = torch.cuda.max_memory_allocated()
torch.cuda.empty_cache()
secs_warm, disp2 = run()
d2 = np.ascontiguousarray(disp2.transpose(3, 2, 1, 0)[::-1])
rep = float(np.abs(d2-d).max()); rep_fg = float(np.sqrt(((d2-d)**2).sum(0))[fixed > 0].mean())
cfg = {k: (str(v) if k in ('dtype', 'device') else v) for k, v in PARAMS.items()}
(out/'method_out.json').write_text(json.dumps(dict(
    method='convexadam_mind_defaults', config=cfg, convexadam_file=convexAdam.__file__, torch=torch.__version__,
    input_layout='Learn2Reg v1.1 (160, 224, 192) = ours.transpose(2, 1, 0)',
    device=f'cuda ({torch.cuda.get_device_name(0)})', threads=dict(torch_intra_op=torch.get_num_threads()),
    seconds_registration_cold=secs_cold, seconds_registration_warm=secs_warm, seconds_total=t_out-T_LAUNCH,
    seconds_launch_to_imports=t_imports-T_LAUNCH, seconds_launch_to_ready=t_ready-T_LAUNCH,
    timer_scope='convex_adam_pt(...) incl. MIND features, coupled convex step, Adam and the float16->numpy output; CUDA context before',
    cuda_peak_allocated_bytes_cold=int(peak_cold), cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
    repeat_max_abs_diff_voxels=rep, repeat_mean_norm_diff_fixed_fg_voxels=rep_fg,
    own_warped='convexAdam.apply_convex.apply_convex (map_coordinates order 1, constant 0) on the L2R layout, transposed back',
    warnings=[]), indent=1))
print(f'convexadam: cold {secs_cold:.2f} s, warm {secs_warm:.2f} s, total {t_out-T_LAUNCH:.2f} s, repeat diff {rep:.3e} (fg {rep_fg:.3e})', flush=True)
