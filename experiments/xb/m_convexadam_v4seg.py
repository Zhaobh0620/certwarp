"""ConvexAdam-v4seg (prereg §2.4, P3; env bl_convexadam).  Usage, under the spy (xb_spy.spyrun_argv):
    python -I -B m_convexadam_v4seg.py INPUT_NPZ MAPS_NPZ OUT_DIR
INPUT_NPZ: the derived npz (fixed, moving; our (z, y, x)); MAPS_NPZ: the two v4 argmax label maps (fixed_map, moving_map; int16, 0..35, our layout), written by the driver from HI's probs36
(C22.1-4).  The entry: convexAdam.convex_adam_nnUNet run as its own __main__ (runpy.run_module) with only -f, -m and --result_path, so every other value is its argparse default (lambda 1.25, grid_sp 6,
disp_hw 4, selected_niter 80, selected_smooth 0, grid_sp_adam 2, ic True; convex_adam_nnUNet.py:166-172).  Its inputs are the two maps as NIfTI in the Learn2Reg v1.1 layout (ours.transpose(2, 1, 0),
as m_convexadam.py), identity affine, int16; the function one-hot encodes them itself, the background a class too, the class weights from this pair's own maps (convex_adam_nnUNet.py:19-38).
Before anything, the imported module's file must have the sha256 9dab1091… (b229e52); otherwise exit 3 without running.
Export: disp.nii.gz (H, W, D, 3) -> d (3, z, y, x) (v4seg_convention.d_from_disp = m_convexadam.py:41).  The export check's other side: convexAdam.apply_convex.apply_convex(disp, moving
intensity in the layout), transposed back (own_warped.npz), as m_convexadam.py:43.  No warm repeat (timing information only in BL2; not needed here).
Writes OUT_DIR/{fixed_map,moving_map}.nii.gz, disp.nii.gz (convexAdam's), disp.npz (d), own_warped.npz (w), method_out.json.  Exit 0; 1 when convexAdam itself raised (a method failure); 3 when this
wrapper's own code raised (wrapper_error.json says where; a wrapper error, prereg §4.5 (2)).
"""
import hashlib
import importlib.util
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path

T_LAUNCH = float(os.environ['BL2_T_LAUNCH'])
WRAPPER_ERROR_EXIT = 3
NNUNET_SHA256 = '9dab109177c98cd9ecda72c8b1b114a46c0a7f30ce860c32d4fcf493943d7f58'
DEFAULTS = dict(lambda_weight=1.25, grid_sp=6, disp_hw=4, selected_niter=80, selected_smooth=0, grid_sp_adam=2, ic=True)     # convex_adam_nnUNet.py:166-172, recorded (not passed)


def wrapper_error(out, phase, exc):
    (Path(out)/'wrapper_error.json').write_text(json.dumps(dict(phase=phase, error=repr(exc), traceback=traceback.format_exc()), indent=1))
    print(f'convexadam_v4seg: wrapper error in {phase}: {exc!r}', file=sys.stderr, flush=True)
    return WRAPPER_ERROR_EXIT


def main(argv):
    inp, maps, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    phase = 'inputs'
    try:
        import numpy as np
        import nibabel as nib
        import torch
        import v4seg_convention as conv
        spec = importlib.util.find_spec('convexAdam.convex_adam_nnUNet')
        origin = spec.origin if spec is not None else None
        digest = hashlib.sha256(Path(origin).read_bytes()).hexdigest() if origin else None
        if digest != NNUNET_SHA256:
            raise RuntimeError(f'convexAdam.convex_adam_nnUNet is {origin} with sha256 {digest}, not {NNUNET_SHA256} (b229e52): not run')
        t_imports = time.time()
        with np.load(inp) as z:
            fixed, moving = np.array(z['fixed'], np.float64), np.array(z['moving'], np.float64)
        with np.load(maps) as z:
            fixed_map, moving_map = np.array(z['fixed_map']), np.array(z['moving_map'])
        if fixed_map.shape != fixed.shape or moving_map.shape != fixed.shape or fixed_map.dtype != np.int16 or moving_map.dtype != np.int16:
            raise ValueError(f'maps {fixed_map.shape} {fixed_map.dtype}, {moving_map.shape} {moving_map.dtype} for images {fixed.shape}')
        if fixed_map.min() < 0 or moving_map.min() < 0 or max(fixed_map.max(), moving_map.max()) > 35:
            raise ValueError('a label outside 0..35')
        paths = {}
        for side, a in (('fixed', fixed_map), ('moving', moving_map)):
            paths[side] = out/f'{side}_map.nii.gz'
            nib.save(nib.Nifti1Image(conv.to_layout(a).astype(np.int16), np.eye(4)), str(paths[side]))
        M_layout = conv.to_layout(moving)
        dev = torch.device('cuda')
        torch.zeros(1, device=dev); torch.cuda.synchronize()                    # CUDA context outside the timer (m_convexadam.py:25)
        t_ready = time.time()
        argv_entry = ['convex_adam_nnUNet', '-f', str(paths['fixed']), '-m', str(paths['moving']), '--result_path', str(out)]
    except Exception as exc:                                                    # noqa: BLE001
        return wrapper_error(out, phase, exc)
    phase = 'method'
    try:
        torch.cuda.reset_peak_memory_stats()
        saved = sys.argv
        sys.argv = list(argv_entry)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        runpy.run_module('convexAdam.convex_adam_nnUNet', run_name='__main__', alter_sys=True)
        torch.cuda.synchronize(); secs = time.perf_counter()-t0
        sys.argv = saved
        t_out = time.time()
        peak = torch.cuda.max_memory_allocated()
    except Exception:                                                           # noqa: BLE001
        traceback.print_exc()
        return 1
    phase = 'export'
    try:
        from convexAdam.apply_convex import apply_convex
        disp_path = out/'disp.nii.gz'
        if not disp_path.is_file():
            print('convexadam_v4seg: convex_adam_nnUNet wrote no disp.nii.gz', file=sys.stderr)
            return 1
        disp = np.asarray(nib.load(str(disp_path)).get_fdata(), dtype=np.float64)
        if disp.shape != M_layout.shape+(3,):
            print(f'convexadam_v4seg: disp.nii.gz has shape {disp.shape}, expected {M_layout.shape+(3,)}', file=sys.stderr)
            return 1
        d = conv.d_from_disp(disp)
        own = conv.from_layout(apply_convex(disp=disp, moving=M_layout))
        np.savez_compressed(out/'disp.npz', d=d.astype(np.float32))
        np.savez_compressed(out/'own_warped.npz', w=own.astype(np.float32))
        import convexAdam
        (out/'method_out.json').write_text(json.dumps(dict(
            method='convexadam_v4seg (convexAdam.convex_adam_nnUNet, argparse defaults, v4 argmax maps)', entry=argv_entry, config=DEFAULTS, config_rule='argparse defaults of convex_adam_nnUNet.py:166-172 (no option passed)',
            convex_adam_nnUNet_file=origin, convex_adam_nnUNet_sha256=digest, convexadam_file=convexAdam.__file__, torch=torch.__version__,
            input_layout='Learn2Reg v1.1 = ours.transpose(2, 1, 0); the maps as int16 NIfTI, identity affine', device=f'cuda ({torch.cuda.get_device_name(0)})',
            threads=dict(torch_intra_op=torch.get_num_threads()), seconds_registration_cold=secs, seconds_registration_warm=None, seconds_total=t_out-T_LAUNCH,
            seconds_launch_to_imports=t_imports-T_LAUNCH, seconds_launch_to_ready=t_ready-T_LAUNCH,
            timer_scope='runpy.run_module(convexAdam.convex_adam_nnUNet) incl. its NIfTI reads, one-hot features, the coupled convex step, Adam and its disp.nii.gz write; CUDA context before',
            cuda_peak_allocated_bytes_cold=int(peak), own_warped='convexAdam.apply_convex.apply_convex on the layout, transposed back', warm_repeat='none', warnings=[]), indent=1))
    except Exception as exc:                                                    # noqa: BLE001
        return wrapper_error(out, phase, exc)
    print(f'convexadam_v4seg: {secs:.2f} s, total {t_out-T_LAUNCH:.2f} s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
