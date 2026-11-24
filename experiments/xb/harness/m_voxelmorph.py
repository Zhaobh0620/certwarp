"""BL2 smoke, VoxelMorph released model vxm_dense_brain_T1_3D_mse.h5 (env bl_synthmorph, TF). Usage:
    python -I -B m_voxelmorph.py PAIR_NPZ OUT_DIR
As env/voxelmorph/smoke_cmd.md: native layout = flip(ours, (0, 2)).transpose(2, 0, 1) (160, 192, 224), [moving, fixed], [0, 1];
flow (1, 160, 192, 224, 3) native-axis voxels, pull-back; own warp = vxm.networks.Transform(inshape).predict([mv, flow])
(linear, edge clamp). Export d = stack([A(-f1), A(f2), A(-f0)]). The first (cold) register() is exported; a second is timed only."""
import json
import os
import sys
import time
from pathlib import Path

T_LAUNCH = float(os.environ['BL2_T_LAUNCH'])
os.environ.setdefault('VXM_BACKEND', 'tensorflow'); os.environ.setdefault('NEURITE_BACKEND', 'tensorflow')
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import voxelmorph as vxm  # noqa: E402

W = '/home/rmcfz/opt/baselines/voxelmorph/weights_osf_mirror/vxm_dense_brain_T1_3D_mse.h5'
inp, out = Path(sys.argv[1]), Path(sys.argv[2])
t_imports = time.time()
gpus = [str(g) for g in tf.config.list_physical_devices('GPU')]
with np.load(inp) as z:
    fixed, moving = np.array(z['fixed'], np.float32), np.array(z['moving'], np.float32)     # (z, y, x), [0, 1]
to_native = lambda a: np.ascontiguousarray(np.flip(a, (0, 2)).transpose(2, 0, 1))          # noqa: E731
A = lambda a: np.ascontiguousarray(np.flip(a.transpose(1, 2, 0), (0, 2)))                    # noqa: E731
mv, fx = to_native(moving)[None, ..., None], to_native(fixed)[None, ..., None]
model = vxm.networks.VxmDense.load(W, input_model=None)
assert tuple(model.config.params['inshape']) == (160, 192, 224)
t_model = time.time()
t0 = time.perf_counter(); flow = model.register(mv, fx); secs_cold = time.perf_counter()-t0
t_out = time.time()
moved = vxm.networks.Transform((160, 192, 224)).predict([mv, flow])
f = flow[0].astype(np.float64)                                                              # (160, 192, 224, 3)
d = np.stack([A(-f[..., 1]), A(f[..., 2]), A(-f[..., 0])])                                  # (3, z, y, x), our axes, voxels
assert d.shape == (3,)+fixed.shape
own = A(moved[0, ..., 0].astype(np.float64))
np.savez_compressed(out/'disp.npz', d=d.astype(np.float32))
np.savez_compressed(out/'own_warped.npz', w=own.astype(np.float32))
t0 = time.perf_counter(); flow2 = model.register(mv, fx); secs_warm = time.perf_counter()-t0
rep = float(np.abs(flow2.astype(np.float64)-flow.astype(np.float64)).max())
(out/'method_out.json').write_text(json.dumps(dict(
    method='voxelmorph_dense_brain_T1_3D_mse', weights=W,
    weights_sha256_record='8e5fe6bcbca68b4fa867864460315fdaa7e00139cd522379cea79db5e63a9e3c', tensorflow=tf.__version__, tf_gpus=gpus,
    config=dict(inference_only=True, int_steps='stored config', inshape=[160, 192, 224]), voxelmorph_file=vxm.__file__,
    input_layout='native = flip(ours, (0, 2)).transpose(2, 0, 1), (1, 160, 192, 224, 1) [moving, fixed], intensities as stored',
    device='cuda (' + (gpus[0] if gpus else 'none') + ')', threads='TensorFlow default',
    seconds_registration_cold=secs_cold, seconds_registration_warm=secs_warm, seconds_total=t_out-T_LAUNCH,
    seconds_launch_to_imports=t_imports-T_LAUNCH, seconds_model_load=t_model-t_imports,
    timer_scope='VxmDense.register (builds the registration sub-model and runs Keras predict; returns numpy); model load outside',
    repeat_max_abs_diff_flow_voxels=rep, own_warped='vxm.networks.Transform(inshape).predict([mv, flow]) (linear, fill_value=None: edge clamp)',
    warnings=[] if gpus else ['TensorFlow saw no GPU']), indent=1))
print(f'voxelmorph: cold {secs_cold:.3f} s, warm {secs_warm:.3f} s, total {t_out-T_LAUNCH:.2f} s, repeat diff {rep:.3e}', flush=True)
