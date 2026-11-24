"""R4 的 T88 选项（合同 C25.2–C25.5）：A 的掩膜与 run_level_k_masked，B 的软通道，C1–C3 只是设定里的数。

关着时与现行逐位相同（C25.1-2）：设定等于 SPEC 时 core_for 返回 mc0_core 模块本身（不安装任何替换）、prepare 走 fl_channels.prepare_stacks 原函数。
A 开着时，mc0_core 以新名字再载入一份（load_copy），用 replace_names 把 run_level_k 换成 run_level_k_masked；原模块与 HS 的模块不受影响。
run_level_k_masked 是 mc0_core.run_level_k（:387–418）的逐语句复制，只把场平滑那一句（:411–414）换成 C25.2-3 的混合；测试做 AST 对照。
"""
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for _d in ('mc0', 'ef', 'fl'):
    _p = str(HERE.parent/_d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from scipy.ndimage import binary_dilation, gaussian_filter  # noqa: E402

import r4_common as rc  # noqa: E402


class MaskError(RuntimeError):
    """掩膜的计算抛异常：hx_implementation_error，原因串 mask_error（C25.2-4）。"""
    reason = 'mask_error'


class SoftArgmaxMismatch(RuntimeError):
    """P̄ 的 argmax 与快路径的 L 不逐体素相同：hx_implementation_error，原因串 soft_argmax_mismatch（C25.3-5）。"""
    reason = 'soft_argmax_mismatch'


class SoftSourceMismatch(SoftArgmaxMismatch):
    """读 P̄ 之前 probs36.npz 的 sha256 与读取器核过的不同（C25.3-2 修订 10b）：hx_implementation_error，原因串 soft_probs_sha_mismatch（复核次要 12）。"""
    reason = 'soft_probs_sha_mismatch'


class SoftSourceMissing(SoftArgmaxMismatch):
    """读取器的记录里没有 probs36.npz 的路径或 sha256：hx_implementation_error，原因串 soft_source_missing。"""
    reason = 'soft_source_missing'


class OptionError(RuntimeError):
    """选项开着的臂里别的异常（软通道的读回或 prepare、run_level_k_masked 里的异常）：hx_implementation_error，不静默退路（复核 I1）。reason 由构造时给。"""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


IMPLEMENTATION_ERRORS = (MaskError, SoftArgmaxMismatch, OptionError)


# ---- 再载入（experiments/r2/r2f_core.py:107–133 的逐语句复制；测试对照） -----------------------------------------------------------------------------
def load_copy(name, path, substitutes):
    """The module at `path` executed once more under `name`, with sys.modules[k] = v for each substitute while it imports (so that its `import hs_core` binds this module, and so on); sys.modules is
    restored afterwards (HS's own modules, when imported in the same process, keep theirs).  The copy is registered under `name`."""
    import importlib.util
    import sys
    saved = {k: sys.modules.get(k) for k in substitutes}
    try:
        sys.modules.update(substitutes)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module


def replace_names(module, **names):
    """setattr for each name, after checking that `module` has it (review m1): if HS renames or drops a function, R2 stops here instead of leaving a new attribute nobody calls."""
    missing = sorted(n for n in names if not hasattr(module, n))
    if missing:
        raise AttributeError(f"{getattr(module, '__name__', module)} has no {missing}: a module of HS changed under R2; R2 does not run until it is looked at")
    for name, value in names.items():
        setattr(module, name, value)
    return module


# ---- A：掩膜（C25.2-1、-2） --------------------------------------------------------------------------------------------------------------------------
def ball(radius=rc.MASK_RADIUS):
    """半径 2 体素的球：{d ∈ ℤ³ : |d|² ≤ 4}，33 个偏移。"""
    r = np.arange(-radius, radius+1)
    z, y, x = np.meshgrid(r, r, r, indexing='ij')
    return (z*z+y*y+x*x) <= radius*radius


def native_mask(fixed_labels, moving_labels):
    """m^(3)：B = 1[L_F ∈ 𝒮] ∨ 1[L_M ∈ 𝒮]（不变形）→ 膨胀一次 → G_σ=1（mode nearest，truncate 4）→ clip(0, 1)，float64。只接受两个标签数组。"""
    L_F, L_M = np.asarray(fixed_labels), np.asarray(moving_labels)
    if L_F.shape != L_M.shape or L_F.ndim != 3 or not np.issubdtype(L_F.dtype, np.integer) or not np.issubdtype(L_M.dtype, np.integer):
        raise MaskError(f'两幅图的整数标签图形状不同或类型不对：{L_F.shape} {L_F.dtype}，{L_M.shape} {L_M.dtype}')
    indicator = np.isin(L_F, rc.SMALL_SIX) | np.isin(L_M, rc.SMALL_SIX)
    grown = binary_dilation(indicator, structure=ball(), iterations=1, border_value=0)
    smooth = gaussian_filter(grown.astype(np.float64), sigma=rc.MASK_SIGMA, mode='nearest', truncate=rc.MASK_TRUNCATE)
    return np.clip(smooth, 0, 1)


def level_masks(m3, grid, grids, downsample=None):
    """({level: m^(ℓ) 全网格 float64}, {level: 内部节点 FP32}, 记录)。粗层 = clip(downsample(m^(3), G_3, shape(G_ℓ)), 0, 1)，各层从原生层直接下采样；
    下采样器另开一个 FastDownsample 实例（不与通道共用）。"""
    if downsample is None:
        import fast_channels as fast
        downsample = fast.FastDownsample()
    full = {}
    for level in range(3):
        values, got, _ = downsample(m3, grid, grids[level].shape)
        if got.grid_id != grids[level].grid_id:
            raise MaskError('掩膜的金字塔网格不同')
        full[level] = np.clip(values, 0, 1)
    full[3] = m3
    interior = {level: np.ascontiguousarray(full[level][1:-1, 1:-1, 1:-1].astype(np.float32)) for level in range(4)}
    record = dict(labels=list(rc.SMALL_SIX), dilation=dict(structure='ball |d|^2 <= 4', radius=rc.MASK_RADIUS, offsets=int(ball().sum())), sigma_mask=rc.MASK_SIGMA, truncate=rc.MASK_TRUNCATE,
                  native_sha256=hashlib.sha256(np.ascontiguousarray(m3, dtype=np.float64).tobytes()).hexdigest(),
                  levels={level: dict(sha256_fp32_interior=hashlib.sha256(interior[level].tobytes()).hexdigest(), mean=float(interior[level].mean()), n_positive=int((interior[level] > 0).sum()))
                          for level in range(4)},
                  n_mask=int((m3[1:-1, 1:-1, 1:-1] > 0).sum()))
    return full, interior, record


def masked_level_runner(core, masks, sigma_small):
    """run_level_k 的替身：本层的 (m_ℓ, c_ℓ, σ_s) 从闭包拿。masks: {level: FP32 内部节点数组}。m_ℓ 第一次用时转到 u_int 的设备上（FP32），c_ℓ = 1 − m_ℓ 在设备上算。"""
    pre_energy_k, _smooth = core.pre_energy_k, core._smooth
    from time import perf_counter
    state = {}

    def mask_of(level, like):
        if level not in state:
            import torch
            mask = torch.tensor(masks[level], device=like.device, dtype=torch.float32)
            state[level] = (mask, 1-mask)
        return state[level]

    def run_level_k_masked(u_int, m, v, F, M, h, level, config, started, observer, counters):
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
                    mask, keep = mask_of(level, u_int)
                    a = _smooth(u_int, config.sigma_field)
                    b = _smooth(u_int, sigma_small)
                    u_int.copy_(keep*a+mask*b)
            if observer is not None:
                observer(dict(level=level, iteration=it, **last))
        return first, last, done, stopped
    run_level_k_masked.mask_state = state
    return run_level_k_masked


def core_for(spec, masks=None, *, name='_r4_mc0_core_masked'):
    """A 关着：mc0_core 模块本身（调用的是原函数）。A 开着：mc0_core 的私有再载入，run_level_k 换成 run_level_k_masked。"""
    import mc0_core
    if spec.get('field_mask') is None:
        if spec.get('sigma_small') is not None:
            raise ValueError('field_mask 与 sigma_small 同开同关（C25.1-1）')
        return mc0_core
    if spec['field_mask'] != 'small6' or spec.get('sigma_small') not in (.42, .35) or masks is None:
        raise ValueError(f"A 的设定不对：field_mask {spec.get('field_mask')!r}、sigma_small {spec.get('sigma_small')!r}，或没有掩膜")
    copy = load_copy(name, mc0_core.__file__, {})
    return replace_names(copy, run_level_k=masked_level_runner(copy, masks, spec['sigma_small']))


def preopt_config(fz, pre, spec):
    """C22.2-4 的 N 与 σ_f 照设定：iterations (50, 50, 100, native)，sigma_field；其余是冻结值。"""
    pre = dict(pre)
    pre.update(iterations=(50, 50, 100, int(spec['native'])), sigma_field=float(spec['sigma_field']))
    return fz.PreoptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in pre.items()}), pre


# ---- B：软通道（C25.3） -----------------------------------------------------------------------------------------------------------------------------
def soft_side(path, expected_sha256, shape, fast_labels, workers=1):
    """一侧的软通道：再核 probs36.npz 的 sha256（读取器之前核过一次）→ read_back_probabilities（float64，除以通道和之前核原始和）→ argmax 与快路径的 L 逐体素相同（否则 SoftArgmaxMismatch）
    → mc0p_core.soft_channels（P̄[1:] 逐通道 G_.75，原地写）。返回 ((35, ...) float64, 记录)。"""
    import mc0p_core as mc
    from bayesreg.seg.channels import read_back_probabilities
    got = rc.sha256_file(path)
    if got != expected_sha256:
        raise SoftSourceMismatch(f'{path}：sha256 {got} 不是读取器核过的 {expected_sha256}')
    P, record = read_back_probabilities(path, shape=(36,)+tuple(shape))
    labels = mc.predicted_labels(P)
    n_diff = int((labels != np.asarray(fast_labels)).sum())
    if n_diff:
        raise SoftArgmaxMismatch(f'{Path(path).name}：P̄ 的 argmax 与快路径的 L 有 {n_diff} 个体素不同')
    del labels
    stack = mc.soft_channels(P, workers)
    return stack, dict(probs36_sha256=got, read_back=record, argmax_consistent=True, n_argmax_differs=0)


def prepare_stacks_soft(spec, pin, predicted, table, device, soft, workers=1):
    """fl_channels.prepare_stacks（:38–64）的 'soft' 分支：Q0、权、尺度三段与 :43–46、:53–64 逐语句相同；通道一段换成已算好的软通道 soft = {side: (35, ...)}。
    权与 Q0 只用 argmax 标签（predicted.labels），与 P̄ 无关（C25.3-3）。"""
    import fast_channels as fast
    import mc0_core as core
    import mc0p_core as mc
    SIDES = ('fixed', 'moving')
    labels = predicted.labels
    pool = 2 if workers > 1 else 1
    q0 = core._pool_map(lambda side: fast.q0_image(getattr(pin, side), labels[side], table), SIDES, pool)
    reference = core.data_grad_norm(q0[0], q0[1], device)
    del q0
    record = dict(D_factor=spec['D_factor'], g_norm_Q0=reference, channel_weights=None, fixed_voxels=None, images_equal_a5p_inputs='not compared here: ef_v1 compared the same reader; g_norm_Q0 and D_multiplier tie the two (decide_fl.controls)')
    if spec['channels'] == 'soft':
        F, M = soft['fixed'], soft['moving']
    else:
        raise SystemExit(f"prepare_stacks_soft has no kind {spec['channels']}")
    if spec['weight'] is not None:
        import fl_channels
        w, voxels = mc.channel_weights(spec['weight'], labels['fixed'])
        roots = np.sqrt(w)
        fl_channels.weigh_in_place(F, roots, workers)
        fl_channels.weigh_in_place(M, roots, workers)
        record.update(channel_weights=w.tolist(), fixed_voxels=voxels.tolist())
    unscaled = core.data_grad_norm(F, M, device)
    s = core.scale_factor(reference, unscaled)*spec['D_factor']
    root = np.sqrt(s)
    fast.scale_in_place(F, root, workers)
    fast.scale_in_place(M, root, workers)
    return [F, M], dict(record, D_multiplier=s, channel_multiplier=float(root), g_norm_unscaled=unscaled, g_norm_scaled=None)


def prepare(spec, pin, predicted, table, device, *, soft=None, workers=1):
    """B 关着：fl_channels.prepare_stacks 原函数（通道由 fast_channels.onehot_channels 构造）；B 开着：prepare_stacks_soft。"""
    import fl_channels
    if spec['channels'] == 'onehot':
        return fl_channels.prepare_stacks(spec, pin, predicted, table, device, workers=workers)
    if spec['channels'] == 'soft':
        if soft is None:
            raise ValueError('B 开着却没有软通道')
        return prepare_stacks_soft(spec, pin, predicted, table, device, soft, workers=workers)
    raise ValueError(f"channels {spec['channels']!r}")


def soft_sources(predicted):
    """{side: (probs36 路径, 读取器核过的 sha256)}：取自读取器的记录（快路径的 unet_directory、unet_output_sha256）。缺就是实现错误。"""
    out = {}
    for side in ('fixed', 'moving'):
        rec = predicted.record['sides'][side]
        directory, outputs = rec.get('unet_directory'), rec.get('unet_output_sha256') or {}
        if directory is None or 'probs36.npz' not in outputs:
            raise SoftSourceMissing(f'{side}：读取器的记录没有 probs36.npz 的路径或 sha256（不是快路径）')
        out[side] = (Path(directory)/'probs36.npz', outputs['probs36.npz'])
    return out


def kernel_table(sigmas=(.25, .35, .42, .5, .65)):
    """C25.2-3 的核表：半径、中心权、±1 权、每步每轴方差、棋盘模态每步增益（_gaussian_1d 的公式，numpy 重算）。"""
    rows = {}
    for s in sigmas:
        r = int(np.ceil(3*s))
        x = np.arange(-r, r+1, dtype=np.float64)
        k = np.exp(-.5*(x/s)**2)
        k /= k.sum()
        rows[s] = dict(radius=r, centre=float(k[r]), first=float(k[r+1]), variance=float((k*x*x).sum()), checkerboard=float(abs((k*np.cos(np.pi*x)).sum())))
    return rows
