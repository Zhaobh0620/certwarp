"""MC0 (runs/20261103-mc0/任务书.md; docs/bayesreg/notes/多通道Sequential_可行性与合同草案_20261103.md §9.2): the channels and the K-channel
optimizer of an uncertified ceiling probe.  Not a solver and not a method: an oracle probe on the first 5 development pairs, no certificate in
the loop, its fields enter no result table.

Copied with provenance from F/jd_curl/three_d/preopt.py (F = /home/rmcfz/seq_vp_tag_results/frozen_eb808be, commit eb808be, file sha256 pinned below):
  _gaussian_1d, _smooth, _identity_mesh     lines 128-154, verbatim
  pre_energy_k                              _pre_energy_tensors, _terms, _pre_energy (lines 157-214): only the data term changes, see data_term_k
  run_level_k                               _run_level_default (lines 217-247), verbatim but for the energy function's name
  optimize_k                                preoptimize, _preoptimize_levels (lines 506-614) on the default path (execution None): the level loop,
                                            seed, Adam state and field export unchanged; each level's channels are built before the loop, in a pool
Called from the frozen tree (never copied), with PYTHONPATH=F:<code> (a process never mixes the repository's jd_curl with the frozen one; the
repository's has no preopt or multiresolution): PreoptConfig, nifti.downsample_array, multiresolution.four_level_grids, prolong_displacement,
preopt.boundary_taper.

K = 1 is bit for bit the frozen default path (preoptimize(..., execution=None)), because every torch operation, shape and operand is the same:
grid_sample gets (1, 1, D, H, W), and data_term_k takes the channel out before the residual.  Checked by run_mc0.py self-check-k1.  It is NOT
bit for bit the production path of configuration C (throughput v41: analytic_grad, rounding level by the frozen docstring), so P0 short is not
bitwise A5's seed.

The data term:  D(u) = 1/2 sum_c sum_x (M_c(x + u(x)) - F_c(x))^2,  a channel weight w_c being applied by multiplying both channel images by sqrt(w_c).
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))                  # last: PYTHONPATH=F:<code> must keep the frozen jd_curl first

MAIN_REPO = Path('/home/rmcfz/code/python_projects/seq_vp_tag')     # data/ is an untracked link of the main checkout; a worktree has none
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
FROZEN_COMMIT = 'eb808be401fde7ed6fee73c2d86212c8c463a513'
FROZEN_PREOPT_SHA256 = '367ee3dbe0577affbb4dd5aa6da91f96c7eb578ce3f3136ab4a87543825adccc'    # configC_reference.json freeze_source_py_sha256
STAGE = MAIN_REPO/'data/three_d/stage_L2R_val_v1'
A5_INPUTS = Path('/home/rmcfz/seq_vp_tag_results/d0_v1_a5/inputs/A5/stage_L2R_val_v1')     # read only: P0's images against the ones A5 ran on
ALLOWLIST = ROOT/'configs/bayesreg/dev_subjects.json'
COLOUR_TABLE_SHA256 = 'ee1d8160bf0d6966df8dc8bfe9bf367e7c6069b34a47a3edf18c58839ed85289'
PAIRS = tuple(f'OASIS_OAS1_{n:04d}_MR1__OASIS_OAS1_{n+1:04d}_MR1' for n in range(438, 443))    # the first 5 of selection.json; nothing else runs
PAIR_MEMBERS = ('fixed', 'moving', 'fixed_seg4', 'fixed_seg35', 'moving_seg4', 'moving_seg35')

TIERS = dict(short=dict(), long=dict(iterations=(50, 50, 100, 600)), fine=dict(iterations=(50, 50, 100, 600), sigma_field=.5))
ALL_TIERS, NOT_SHORT = ('short', 'long', 'fine'), ('long', 'fine')
ARMS = dict(P0=dict(k=1, tiers=ALL_TIERS), P1=dict(k=14, tiers=ALL_TIERS), P2=dict(k=35, tiers=NOT_SHORT), P3=dict(k=35, tiers=('long',)))     # plan revision 1: P3 long only
LABELS_35 = tuple(range(1, 36))
GROUPING = 'structure13_v1'

# The PRE of configuration C (experiments/d0/configC_reference.json, reference_protocol.pre_optimizer; a test compares them) but for `seconds`:
# 60 s there would cut the long tiers, so every arm gets the one wall cap of this probe (the brief's 15 minutes).
PRE_C = dict(iterations=(50, 50, 100, 60), lr=.5, cap=.5, sigma_grad=1., sigma_field=1., mu=(1., 1., .01, .003), alpha=(1e-3, 1e-3, 1e-3, 0.),
             kappa_pre=10., j0=.2, beta1=.9, beta2=.99, eps=1e-8)
TAPER = 4
RUN_CAP_SECONDS = 900.
SETUP_WORKERS = 7                               # F's PARALLEL_SETUP_WORKERS; scipy.ndimage releases the GIL; values do not depend on it
SCALE_TOL = 1e-3


@dataclass(frozen=True)
class Cell:
    arm: str
    tier: str
    pair: str

    @property
    def short(self):
        return short_name(self.pair)

    @property
    def run_id(self):
        return f'mc0_{self.arm}_{self.tier}_{self.short}'

    @property
    def unit(self):
        return f'tag-mc0-{self.arm.lower()}-{self.tier}-{self.short}'


def short_name(pair):
    return '-'.join(side.split('_')[2] for side in pair.split('__'))


def require_pair(pair):
    """Only the first 5 development pairs: checked before any file of the pair is opened."""
    if pair not in PAIRS:
        raise SystemExit(f'MC0 runs only the first 5 development pairs {[short_name(p) for p in PAIRS]}; refused: {pair}')
    return pair


def require_cell(arm, tier, pair):
    require_pair(pair)
    if arm not in ARMS or tier not in ARMS[arm]['tiers']:
        raise SystemExit(f'MC0 has no cell arm {arm} tier {tier}')
    return Cell(arm, tier, pair)


def plan_runs():
    """Every run, in order: pair-major; within a pair the arm order rotates with the pair index, the tiers in the brief's order."""
    arms = tuple(ARMS)
    return [Cell(arms[(i+j) % len(arms)], tier, pair)
            for i, pair in enumerate(PAIRS) for j in range(len(arms)) for tier in ARMS[arms[(i+j) % len(arms)]]['tiers']]


def run_dir(root, cell):
    return Path(root)/'runs'/f'{cell.arm}_{cell.tier}'/cell.short


def tier_config(tier):
    return dict(PRE_C, seconds=RUN_CAP_SECONDS, **TIERS[tier])


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def load_module(name, rel):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT/rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_frozen = None


def frozen_api():
    """The frozen tree's functions.  STOP unless jd_curl comes from it and preopt.py has the pinned sha256.  Call it before anything else imports jd_curl."""
    global _frozen
    if _frozen is None:
        import jd_curl
        if FROZEN.resolve() not in Path(jd_curl.__file__).resolve().parents:
            raise SystemExit(f'jd_curl comes from {jd_curl.__file__}; run with PYTHONPATH={FROZEN}:<code>')
        if sha_file(FROZEN/'jd_curl/three_d/preopt.py') != FROZEN_PREOPT_SHA256:
            raise SystemExit(f'{FROZEN}/jd_curl/three_d/preopt.py differs from the pinned sha256')
        from jd_curl.three_d import multiresolution, nifti, preopt
        _frozen = SimpleNamespace(PreoptConfig=preopt.PreoptConfig, preoptimize=preopt.preoptimize, boundary_taper=preopt.boundary_taper,
                                  precision_record=preopt.float32_precision_record, four_level_grids=multiresolution.four_level_grids,
                                  prolong_displacement=multiresolution.prolong_displacement, alphas=multiresolution.TRANSFER_BACKTRACK_ALPHAS,
                                  downsample_array=nifti.downsample_array)
    return _frozen


# ---- channels ------------------------------------------------------------------------------------------------------------------
def make_inputs():
    """experiments/d0/make_inputs.py: the A5 colouring and G_.75 come from there, not copied."""
    return load_module('_mc0_make_inputs', 'experiments/d0/make_inputs.py')


def load_table(path=None):
    """The A5 colour table; STOP unless it has the pinned sha256."""
    mi = make_inputs()
    return mi.load_colour_table(mi.COLOUR_TABLE_A5 if path is None else path, COLOUR_TABLE_SHA256)[0]


def load_groups():
    """The label tuples of structure13_v1: 13 groups, background only label 0."""
    from bayesreg.seg.unet.groups import load_grouping
    grouping = load_grouping(name=GROUPING)
    if grouping.k != 13 or grouping.background_labels != (0,) or sorted(sum(grouping.group_labels, ())) != list(LABELS_35):
        raise SystemExit(f'{GROUPING}: expected 13 groups partitioning the labels 1..35')
    return grouping.group_labels


def _pool_map(fn, items, workers):
    if workers <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='mc0-setup') as pool:
        return list(pool.map(fn, items))


def channels(arm, image, seg35, *, table=None, groups=None, workers=1):
    """(K, D, H, W) float64 channels of one image from its own labels and intensities.  P0: the A5 colouring (protocol §14).  P1: the intensity, then the
    smoothed one-hot sum of each structure group.  P2, P3: the smoothed one-hot of each label 1..35."""
    mi = make_inputs()
    if arm == 'P0':
        return mi.colour_from_table(image, seg35, load_table() if table is None else table)[0][None]
    if arm == 'P1':
        members, offset = (load_groups() if groups is None else groups), 1
    elif arm in ('P2', 'P3'):
        members, offset = tuple((label,) for label in LABELS_35), 0
    else:
        raise SystemExit(f'no channels for arm {arm}')
    out = np.empty((len(members)+offset,)+image.shape, dtype=np.float64)
    if offset:
        out[0] = image

    def fill(c):                                # written in place: the host never holds a second copy of the stack
        out[c+offset] = mi.smooth(np.isin(seg35, members[c]).astype(np.float64))
    _pool_map(fill, range(len(members)), workers)
    return out


def equals_a5_input(pair, fixed_channel, moving_channel, inputs=None):
    """P0's two images against A5's own pair.npz, bit for bit (a record of the arm's formula on the real files); None if that file is not there."""
    path = (A5_INPUTS if inputs is None else inputs)/pair/'pair.npz'
    if not path.is_file():
        return None
    with np.load(path) as z:
        return bool(z['fixed'].tobytes() == fixed_channel.tobytes() and z['moving'].tobytes() == moving_channel.tobytes())


def channel_weights(arm, fixed_seg35):
    """P3: w_l proportional to 1/V_l, V_l the voxels of label l in the fixed image; a label absent there gets 0 (it has no target).  Normalized to mean 1
    over the present labels, for reading only: the scale rule fixes the overall size.  Other arms: None, None."""
    if arm != 'P3':
        return None, None
    voxels = np.array([np.count_nonzero(fixed_seg35 == label) for label in LABELS_35], dtype=np.float64)
    present = voxels > 0
    if not present.any():
        raise SystemExit('P3: no label in the fixed image')
    w = np.zeros(len(LABELS_35))
    w[present] = 1/voxels[present]
    return w/w[present].mean(), voxels


def scale_factor(reference, norm):
    """s such that the data term s D has ||g_D|| = reference at u = 0: the channels are multiplied by sqrt(s)."""
    if not (np.isfinite(reference) and np.isfinite(norm) and reference > 0 and norm > 0):
        raise SystemExit(f'scale rule: gradient norms must be finite and positive (reference {reference}, arm {norm})')
    return float(reference/norm)


def level_grad_norms(F, M, sigma_grad):
    """(||g_D||, ||G_grad g_D||) at u = 0 on one level's channel tensors: the 2-norms of the raw autograd gradient of the data term on the interior nodes (voxel units) and of the
    gradient as the loop smooths it.  A separate zero field: nothing here touches the loop's state."""
    import torch
    import torch.nn.functional as Fn
    D, H, W = F.shape[1:]
    u_int = torch.zeros((3, D-2, H-2, W-2), device=F.device, dtype=F.dtype, requires_grad=True)
    g, = torch.autograd.grad(data_term_k(Fn.pad(u_int, (1, 1, 1, 1, 1, 1)), F, M), u_int)
    return float(torch.linalg.vector_norm(g.double())), float(torch.linalg.vector_norm(_smooth(g, sigma_grad).double()))


def data_grad_norm(fixed, moving, device):
    """||g_D|| at u = 0, native level, the float32 path of the loop: the norm the scale rule equalizes."""
    import torch
    dev = torch.device(device)
    return level_grad_norms(stack_tensor(fixed, dev, torch.float32), stack_tensor(moving, dev, torch.float32), 0.)[0]


def p0_level_norms(fixed, moving, fixed_seg35, moving_seg35, grid, device, sigma_grad, *, table=None, workers=1):
    """P0's level_grad_norms on all four levels: the reference of r_l (plan revision 1)."""
    import torch
    fz = frozen_api()
    dev = torch.device(device)
    stacks = [channels('P0', fixed, fixed_seg35, table=table), channels('P0', moving, moving_seg35, table=table)]
    pyramid = coarse_pyramid(stacks, grid, fz.four_level_grids(grid), fz.downsample_array, workers)
    out = []
    for level in range(4):
        F, M = (stack_tensor(side, dev, torch.float32) for side in (stacks if level == 3 else pyramid[level]))
        raw, smoothed = level_grad_norms(F, M, sigma_grad)
        out.append(dict(level=level, raw=raw, smoothed=smoothed))
    return out


def level_norm_record(arm, reference=None):
    """r_l = ||g_D(arm, l, u = 0)|| / ||g_D(P0, l, u = 0)||, per level (8, 4, 2, 1), raw and as the loop smooths it.  The scale rule fixes r_3 only.  reference None: the arm is P0."""
    reference = arm if reference is None else reference
    def ratio(key):
        return [a[key]/b[key] if b[key] > 0 else None for a, b in zip(arm, reference)]
    return dict(arm=arm, P0=reference, ratio_raw=ratio('raw'), ratio_smoothed=ratio('smoothed'))


def prepare_stacks(arm, fixed, moving, fixed_seg35, moving_seg35, device, *, table=None, groups=None, workers=1):
    """([F, M], record): the arm's two channel stacks, weighted and scaled by the rule.  P0 is its own reference: never multiplied, scale exactly 1."""
    table = load_table() if table is None else table
    p0 = channels('P0', fixed, fixed_seg35, table=table), channels('P0', moving, moving_seg35, table=table)
    reference = data_grad_norm(*p0, device)
    if arm == 'P0':
        return list(p0), dict(D_multiplier=1., channel_multiplier=1., g_norm_P0=reference, g_norm_unscaled=reference, g_norm_scaled=reference,
                              channel_weights=None, fixed_voxels=None)
    del p0
    F = channels(arm, fixed, fixed_seg35, groups=groups, workers=workers)
    M = channels(arm, moving, moving_seg35, groups=groups, workers=workers)
    w, voxels = channel_weights(arm, fixed_seg35)
    if w is not None:
        root = np.sqrt(w)[:, None, None, None]
        F *= root
        M *= root
    unscaled = data_grad_norm(F, M, device)
    s = scale_factor(reference, unscaled)
    F *= np.sqrt(s)
    M *= np.sqrt(s)
    scaled = data_grad_norm(F, M, device)
    if abs(scaled/reference-1) > SCALE_TOL:
        raise SystemExit(f'scale rule: ||g_D|| after scaling {scaled} differs from P0 {reference}')
    return [F, M], dict(D_multiplier=s, channel_multiplier=float(np.sqrt(s)), g_norm_P0=reference, g_norm_unscaled=unscaled, g_norm_scaled=scaled,
                        channel_weights=None if w is None else w.tolist(), fixed_voxels=None if voxels is None else voxels.tolist())


# ---- the copied optimizer (provenance in the module docstring) --------------------------------------------------------------------
def _gaussian_1d(sigma, device, dtype):
    import torch
    r = int(np.ceil(3*sigma)); x = torch.arange(-r, r+1, device=device, dtype=dtype)
    k = torch.exp(-.5*(x/sigma)**2); return k/k.sum()


def _smooth(field, sigma):
    """Separable Gaussian smoothing of a (C, D, H, W) tensor along the three spatial axes (replicate padding)."""
    import torch  # noqa: F401  (verbatim)
    import torch.nn.functional as Fn
    if sigma <= 0:
        return field
    k = _gaussian_1d(sigma, field.device, field.dtype); r = (k.numel()-1)//2
    c = field.shape[0]; out = field[None]
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]; shape[2+axis] = k.numel()
        w = k.view(shape).expand(c, 1, *shape[2:]).contiguous()
        pad = [0, 0, 0, 0, 0, 0]; pad[2*(2-axis)] = pad[2*(2-axis)+1] = r
        out = Fn.conv3d(Fn.pad(out, pad, mode="replicate"), w, groups=c)
    return out[0]


def _identity_mesh(D, H, W, device, dtype):
    """The three meshgrid tensors (z, y, x) of a level, as the per-iteration code built them."""
    import torch
    return torch.meshgrid(torch.arange(D, device=device, dtype=dtype), torch.arange(H, device=device, dtype=dtype),
                          torch.arange(W, device=device, dtype=dtype), indexing="ij")


def data_term_k(u, fixed, moving):
    """1/2 sum_c sum_x (M_c(x + u(x)) - F_c(x))^2.  u (3, D, H, W) voxel units with zero faces; fixed, moving (K, D, H, W).  One grid_sample over the K
    channels.  K = 1: the channel is taken out first, so the residual and the sum run on (D, H, W), the frozen operands."""
    import torch
    import torch.nn.functional as Fn
    D, H, W = fixed.shape[1:]
    zz, yy, xx = _identity_mesh(D, H, W, u.device, u.dtype)
    gx = 2*(xx+u[0])/max(W-1, 1)-1; gy = 2*(yy+u[1])/max(H-1, 1)-1; gz = 2*(zz+u[2])/max(D-1, 1)-1
    grid = torch.stack((gx, gy, gz), dim=-1)[None]
    warped = Fn.grid_sample(moving[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
    if warped.shape[0] == 1:
        warped, fixed = warped[0], fixed[0]
    return .5*((warped-fixed)**2).sum()


def pre_energy_k(u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0):
    """F's _pre_energy with data_term_k: (total tensor, the record floats)."""
    import torch
    import torch.nn.functional as Fn
    u = Fn.pad(u_int, (1, 1, 1, 1, 1, 1))                           # zero faces -> (3, D, H, W)
    hz, hy, hx = spacing[2], spacing[1], spacing[0]
    data = data_term_k(u, fixed, moving)

    def d(axis):
        g = torch.diff(u, dim=1+axis)
        for ax in range(3):
            if ax != axis:
                g = .5*(g.narrow(1+ax, 0, g.shape[1+ax]-1)+g.narrow(1+ax, 1, g.shape[1+ax]-1))
        return g                                                       # (3, D-1, H-1, W-1): components x, y, z
    gz_, gy_, gx_ = d(0), d(1), d(2)                                   # derivatives along z, y, x
    h = (hx, hy, hz)

    def J(c, dd):                                                      # d u_c / d x_dd, physical
        g = (gx_, gy_, gz_)[dd][c]
        return g*(h[c]/h[dd])
    F00 = 1+J(0, 0); F01 = J(0, 1); F02 = J(0, 2)
    F10 = J(1, 0); F11 = 1+J(1, 1); F12 = J(1, 2)
    F20 = J(2, 0); F21 = J(2, 1); F22 = 1+J(2, 2)
    det = F00*(F11*F22-F12*F21)-F01*(F10*F22-F12*F20)+F02*(F10*F21-F11*F20)
    curl = torch.stack((F21-F12, F02-F20, F10-F01))
    geometry = .5*mu*(((det-1)**2).sum()+(curl**2).sum())
    floor = kappa_pre*(torch.relu(j0-det)**2).sum()
    curvature = torch.zeros((), device=u.device, dtype=u.dtype)
    if alpha > 0:
        # seven-point Laplacian of u (mm) on the interior nodes: u_mm,c = h_c u_vox,c; neighbours over the zero faces
        ui = u[:, 1:-1, 1:-1, 1:-1]
        lap = ((u[:, 2:, 1:-1, 1:-1]-2*ui+u[:, :-2, 1:-1, 1:-1])/(hz*hz)
               + (u[:, 1:-1, 2:, 1:-1]-2*ui+u[:, 1:-1, :-2, 1:-1])/(hy*hy)
               + (u[:, 1:-1, 1:-1, 2:]-2*ui+u[:, 1:-1, 1:-1, :-2])/(hx*hx))
        lap = lap*torch.tensor([hx, hy, hz], device=u.device, dtype=u.dtype)[:, None, None, None]
        curvature = .5*alpha*(lap**2).sum()
    total = data+geometry+floor+curvature
    return total, dict(data=float(data), geometry=float(geometry), floor=float(floor), curvature=float(curvature), min_det=float(det.min()))


def run_level_k(u_int, m, v, F, M, h, level, config, started, observer, counters):
    """F's _run_level_default: the original iteration loop of a level.  Returns (first, last, done, stopped)."""
    import torch
    first = last = None; done = 0; stopped = None
    for it in range(config.iterations[level]):
        if perf_counter()-started >= config.seconds:
            stopped = dict(level=level, iteration=it); break
        done = it+1
        total, terms = pre_energy_k(u_int, F, M, h, config.mu[level], config.alpha[level], config.kappa_pre, config.j0)
        if first is None:
            first = terms
        last = dict(terms, total=float(total))
        if not np.isfinite(float(total)):
            raise FloatingPointError("preopt: nonfinite pre-objective")
        grad, = torch.autograd.grad(total, u_int)
        with torch.no_grad():
            g = _smooth(grad, config.sigma_grad)
            m.mul_(config.beta1).add_(g, alpha=1-config.beta1); v.mul_(config.beta2).addcmul_(g, g, value=1-config.beta2)
            t = it+1
            step = config.lr*(m/(1-config.beta1**t))/((v/(1-config.beta2**t)).sqrt()+config.eps)
            largest = float(step.abs().amax())
            scale = 1. if largest <= config.cap else config.cap/largest
            counters["cap_hits"] += scale < 1.
            u_int.sub_(step*scale)
            if config.sigma_field > 0:
                # S30-G3 r1: elastic smoothing of the field (FireANTs' smooth_warp_sigma); it removes the node-level
                # (hourglass) modes that the cell-centre terms cannot see
                u_int.copy_(_smooth(u_int, config.sigma_field))
        if observer is not None:
            observer(dict(level=level, iteration=it, **last))
    return first, last, done, stopped


def stack_tensor(arrays, dev, dtype):
    """(K, ...) float32 on the device, a channel at a time (the host never holds a float32 copy of the whole stack)."""
    import torch
    out = torch.empty((len(arrays),)+tuple(arrays[0].shape), device=dev, dtype=dtype)
    for c, array in enumerate(arrays):
        out[c] = torch.tensor(array, device=dev, dtype=dtype)
    return out


def coarse_pyramid(stacks, grid, grids, downsample, workers):
    """{level: (fixed channels, moving channels)} of the three coarse levels: nifti.downsample_array channel by channel, the same call as F's."""
    K = len(stacks[0])

    def one(task):
        level, side, c = task
        values, got, _ = downsample(stacks[side][c], grid, grids[level].shape)
        if got.grid_id != grids[level].grid_id:
            raise ValueError("mc0: pyramid grids differ")
        return values
    tasks = [(level, side, c) for level in range(3) for side in (0, 1) for c in range(K)]
    results = dict(zip(tasks, _pool_map(one, tasks, workers)))
    return {level: tuple([results[(level, side, c)] for c in range(K)] for side in (0, 1)) for level in range(3)}


def _checkpointing_observer(observer, checkpoint, u_int, level_grid, h):
    """FL: the loop's observer, then checkpoint(level, steps done, u_int, level_grid, h, row) after every step.  run_level_k stays verbatim; a checkpoint only reads u_int."""
    def both(row):
        if observer is not None:
            observer(row)
        checkpoint(row['level'], row['iteration']+1, u_int, level_grid, h, row)
    return both


def optimize_k(stacks, grid, config, device="cuda", started=None, observer=None, workers=1, record_level_norms=False, downsample=None, checkpoint=None):
    """F's preoptimize on the default path, K channels.  stacks: [fixed (K, D, H, W), moving (K, D, H, W)] float64, emptied when the native tensors exist (host
    memory).  started: the clock of config.seconds (default: now).  record_level_norms: each level's ||g_D|| at u = 0 in record['level_g_norm'], not in record['levels'] (the K = 1
    check runs with it on).  downsample: EF passes a bit-for-bit faster nifti.downsample_array (experiments/ef/fast_channels.py); None: the frozen one, as always.  checkpoint: FL passes a reader of the field after each step (experiments/fl/run_fl.py); None: nothing is called, as always.  Returns (u float64 (D, H, W, 3) in mm with zero faces, record); uncertified."""
    import torch
    fz = frozen_api()
    started = perf_counter() if started is None else started
    grids = fz.four_level_grids(grid)
    dev = torch.device(device); dtype = torch.float32
    pyramid = coarse_pyramid(stacks, grid, grids, fz.downsample_array if downsample is None else downsample, workers)
    K = len(stacks[0])
    u_mm = None; levels = []; counters = dict(cap_hits=0); stopped = None; level_norms = []
    for level, level_grid in enumerate(grids):
        tick = perf_counter()
        h = tuple(float(s) for s in level_grid.spacing)                 # (hx, hy, hz)
        seed = np.zeros(level_grid.shape+(3,)) if u_mm is None else fz.prolong_displacement(u_mm, grids[level-1], level_grid)
        # voxel units, interior, (3, D-2, H-2, W-2)
        seed_vox = np.moveaxis(seed[1:-1, 1:-1, 1:-1], -1, 0)/np.array(h)[:, None, None, None]
        u_int = torch.tensor(seed_vox, device=dev, dtype=dtype, requires_grad=True)
        F, M = (stack_tensor(side, dev, dtype) for side in (stacks if level == 3 else pyramid.pop(level)))
        if level == 3:
            stacks.clear()
        if record_level_norms:
            raw, smoothed = level_grad_norms(F, M, config.sigma_grad)
            level_norms.append(dict(level=level, raw=raw, smoothed=smoothed))
        m = torch.zeros_like(u_int); v = torch.zeros_like(u_int)
        first, last, done, stopped = run_level_k(u_int, m, v, F, M, h, level, config, started, observer if checkpoint is None else _checkpointing_observer(observer, checkpoint, u_int, level_grid, h), counters)
        with torch.no_grad():
            out = np.zeros(level_grid.shape+(3,))
            out[1:-1, 1:-1, 1:-1] = np.moveaxis(u_int.detach().double().cpu().numpy(), 0, -1)*np.array(h)
        u_mm = out
        levels.append(dict(level=level, shrink_factor=(8, 4, 2, 1)[level], shape_zyx=list(level_grid.shape), iterations=done,
                           seconds=perf_counter()-tick, first=first, last=last))
        del F, M, m, v, u_int
        if stopped is not None:
            break
    if u_mm.shape != grid.shape+(3,):                                  # time cap on a coarse level: carry to native
        for level in range(levels[-1]["level"]+1, 4):
            u_mm = fz.prolong_displacement(u_mm, grids[level-1], grids[level])
    record = dict(policy="mc0-k-channel-copy-of-fp32-preopt-v1", channels=K, config=asdict(config), device=str(dev), dtype="float32", levels=levels,
                  cap_hits=int(counters["cap_hits"]), stopped=stopped, seconds=perf_counter()-started, **(dict(level_g_norm=level_norms) if record_level_norms else {}),
                  role="uncertified_oracle_probe_field_NOT_a_method_NOT_in_any_result_table", float32_precision=fz.precision_record(torch, dev, config))
    return np.ascontiguousarray(u_mm), record
