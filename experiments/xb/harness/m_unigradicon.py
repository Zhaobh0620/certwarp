"""BL2 smoke, uniGradICON arms (env bl_unigradicon). Usage: python -I -B m_unigradicon.py {io|noio} PAIR_NPZ OUT_DIR [N]
BL2VAL (BL2 Amendment 4): optional integer N >= 1 (io only) = the equal-time arm: the CLI gets --io_iterations N instead of the
default 50 (lr, similarity, everything else unchanged) and the second (warm) CLI run is skipped (one registration per run, as
Amendment 4's cost assumes). Without N: as the BL2 smoke.
Runs the package CLI entry point (unigradicon.main, i.e. `unigradicon-register`) in-process with exactly the flags of
env/bl_unigradicon/smoke_cmd.md: --fixed_modality mri --moving_modality mri --network_weights W, plus --io_iterations None for
noio (io = package default 50 steps, lr 2e-5, LNCC). The only instrumentation: timing wrappers around
unigradicon.get_model_from_model_zoo and icon_registration.itk_wrapper.register_pair (both call the originals unchanged).
Inputs written with itk (identity geometry) as in the record. Export: TransformToDisplacementFieldFilter on the fixed grid.
Writes disp.npz, own_warped.npz (the CLI's warped moving, our layout), method_out.json. The first CLI run is exported; a second
CLI run in the same process (outputs warm_*) is timed only."""
import json
import os
import sys
import time
from pathlib import Path

T_LAUNCH = float(os.environ['BL2_T_LAUNCH'])
import numpy as np  # noqa: E402
import itk  # noqa: E402
import torch  # noqa: E402
import icon_registration  # noqa: E402
import icon_registration.itk_wrapper as iw  # noqa: E402
import unigradicon  # noqa: E402

W = '/home/rmcfz/opt/baselines/unigradicon/weights/unigradicon1.0/Step_2_final.trch'
variant, inp, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
assert variant in ('io', 'noio')
NIO = int(sys.argv[4]) if len(sys.argv) > 4 else None                        # BL2VAL: equal-time --io_iterations (None = default arm)
assert NIO is None or (variant == 'io' and NIO >= 1), (variant, NIO)
os.chdir(out)
t_imports = time.time()
with np.load(inp) as z:
    fixed, moving = np.array(z['fixed'], np.float64), np.array(z['moving'], np.float64)
    for k in ('fixed', 'moving'):
        img = itk.image_from_array(np.ascontiguousarray(z[k].astype(np.float32)))   # array (z, y, x) -> image (x, y, z)
        img.SetSpacing((1., 1., 1.)); img.SetOrigin((0., 0., 0.))                   # direction defaults to identity
        itk.imwrite(img, f'{k}.nii.gz')
t_inputs = time.time()

T = dict(model=[], register_pair=[], main=[])
_orig_model, _orig_rp = unigradicon.get_model_from_model_zoo, iw.register_pair


def timed_model(*a, **k):
    t0 = time.perf_counter(); r = _orig_model(*a, **k); T['model'].append(time.perf_counter()-t0); return r


def timed_rp(*a, **k):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter(); r = _orig_rp(*a, **k)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    T['register_pair'].append(time.perf_counter()-t0); return r


unigradicon.get_model_from_model_zoo = timed_model
iw.register_pair = timed_rp
assert icon_registration.itk_wrapper.register_pair is timed_rp


def cli(prefix):
    argv = ['unigradicon-register', '--fixed', 'fixed.nii.gz', '--fixed_modality', 'mri', '--moving', 'moving.nii.gz',
            '--moving_modality', 'mri', '--transform_out', f'{prefix}.hdf5', '--warped_moving_out', f'{prefix}_warped.nii.gz',
            '--network_weights', W]
    if variant == 'noio':
        argv += ['--io_iterations', 'None']
    elif NIO is not None:                                                    # BL2VAL
        argv += ['--io_iterations', str(NIO)]
    sys.argv = argv
    t0 = time.perf_counter(); unigradicon.main(); T['main'].append(time.perf_counter()-t0)
    return argv


def export(prefix):
    fimg = itk.imread('fixed.nii.gz', itk.F)
    tr = itk.transformread(f'{prefix}.hdf5')[0]
    VF = itk.Image[itk.Vector[itk.F, 3], 3]
    flt = itk.TransformToDisplacementFieldFilter[VF, itk.D].New()
    flt.SetTransform(tr); flt.SetReferenceImage(fimg); flt.SetUseReferenceImage(True); flt.Update()
    v = itk.array_from_image(flt.GetOutput()).astype(np.float64)          # (z, y, x, 3), components (x, y, z)
    return np.moveaxis(v[..., ::-1], -1, 0)                               # (3, z, y, x), our axis order, voxels (spacing 1, identity dir)


tag = f'ug_{variant}'
argv1 = cli(tag)
t_out = time.time()
d = export(tag)
assert d.shape == (3,)+fixed.shape
own = itk.array_from_image(itk.imread(f'{tag}_warped.nii.gz')).astype(np.float64)
np.savez_compressed(out/'disp.npz', d=d.astype(np.float32))
np.savez_compressed(out/'own_warped.npz', w=own.astype(np.float32))
peak1 = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
if NIO is None:
    cli(f'warm_{tag}')
    d2 = export(f'warm_{tag}')
    rep = float(np.abs(d2-d).max()); rep_fg = float(np.sqrt(((d2-d)**2).sum(0))[fixed > 0].mean())
else:                                                                        # BL2VAL: no warm repeat in the equal-time arm
    rep = rep_fg = None
(out/'method_out.json').write_text(json.dumps(dict(
    method=f'unigradicon_{variant}', cli_argv=argv1, unigradicon_file=unigradicon.__file__, torch=torch.__version__, itk=itk.__version__,
    config=dict(io_iterations=(NIO or 50) if variant == 'io' else None, io_lr=2e-5 if variant == 'io' else None, io_sim='lncc', modality='mri',
                weights=W, masks=None, segmentations=None, dice_loss_weight=0.0),
    device=f'cuda ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else 'cpu',
    threads=dict(torch_intra_op=torch.get_num_threads()),
    seconds_registration_cold=T['register_pair'][0], seconds_registration_warm=T['register_pair'][1] if NIO is None else None,
    seconds_model_load=T['model'], seconds_cli_main=T['main'], seconds_total=t_out-T_LAUNCH,
    seconds_launch_to_imports=t_imports-T_LAUNCH, seconds_input_write=t_inputs-t_imports,
    timer_scope='icon_registration.itk_wrapper.register_pair (model to device, resize to 175^3, IO or forward pass, ITK transform '
                'construction); model construction/weight loading and preprocessing outside; total = launch to the CLI having written '
                'its transform and warped image',
    cuda_peak_allocated_bytes_first=int(peak1) if peak1 is not None else None,
    repeat_max_abs_diff_voxels=rep, repeat_mean_norm_diff_fixed_fg_voxels=rep_fg,
    own_warped='CLI --warped_moving_out (itk resample, linear, default value 0)', warnings=[],
    **({} if NIO is None else dict(equal_time=dict(io_iterations=NIO, iterations=[NIO], early_stop='none (fixed number of IO steps)',
                                                   warm_repeat='skipped (BL2VAL: one registration per run)')))), indent=1))
print(f'{variant}: register_pair {T["register_pair"]}, model {T["model"]}, main {T["main"]}, total {t_out-T_LAUNCH:.2f} s, '
      + (f'repeat diff {rep:.3e} (fg {rep_fg:.3e})' if NIO is None else f'io_iterations {NIO}, no warm repeat'), flush=True)
