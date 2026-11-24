"""P2 / P3 (runs/20261106-p2/任务书.md, plan.md; docs/bayesreg/notes/贝叶斯化Hx线_设计_20261105.md §3 (a2), (c), §5.3): the Bayesian parts on the line of Hx24@600.
Not a method: no F_B, no certified descent, only the final field passes the certificate; development pairs only, in no result table.  The numbers, the copied optimizer, the cells and the
reference; reading the files and the units: run_p2.py; the readings: decide_p2.py.

Arms (all from u = 0; everything else frozen Hx24: K = 35 argmax one-hot G_.75, Q3h as the starting weights, D factor 24, sigma_field .5, levels 50/50/100/600, RF05 repair, fuse 1200 s):
  A   the channel weights by a pairwise EM instead of Q3h.  sigma_c^2 by C11-2 with a = b = 0 on the brain domain Omega_B (C03.3-5, the fixed image's Atropos layer):
          sigma_c^2 = R_c / (N_B + 2),   R_c = sum over Omega_B of (M_c(x + u(x)) - F_c(x))^2,  F_c, M_c the unweighted channels;
      w_c = 1/sigma_c^2 normalized to mean 1 over the labels present in the fixed image (V_c > 0) whose R_c > 0; an absent label 0; a present label with R_c = 0 has no finite
      precision: 0, recorded (left_out).  s = 24 ||g_Q0|| / ||g_w|| again at every M step (the rule of fl_channels; 24 a hand-picked constant, not an estimate), ||g_w|| = sqrt(w^T G w)
      with G the Gram matrix of the per-channel data gradients at u = 0 on the native level.  The M steps that are used: the head of each level (level 0 at u = 0; then the level's seed,
      prolonged to the native grid) and the native steps 200 and 400 (after that many native steps; C11-3 with R = (1, 1, 1, 3)); a diagnostic M step on the native field at 600, never used
      (sigma_post, C11-6).  Adam's m, v stay across the native rounds (the design note's choice: the overall scale does not move).
  C   T88's geometry term with a weight per cell: G(u) = mu/2 [sum_K a_K (det F_K - 1)^2 + sum_K |curl_K|^2],  a_K = m / t_K,  t_K = sum_k P_k(K) tau_k^2 / tau_WM^2
      (the classes background, CSF, GM, WM of the fixed image's Atropos layer after C03.2 and C03.5; P_k(K) the mean of the cell's 8 nodes; tau_0^2 = tau_WM^2), m such that the mean of a
      over the Omega_B cells (all 8 nodes in Omega_B) of the level is 1.  Coarse levels: P and Omega_B by C03.6-12 (downsample_array, the simplex renormalized, the mask at >= 1/2).  div and
      curl keep equal weights (b is not learned).  The two ratios tau_CSF^2/tau_WM^2, tau_GM^2/tau_WM^2 by P1's moment matching (p1_core.geometry_sums, tissue_moments; all cells) on the
      field of the pass before.  Three passes of T88: pass 0 with a = 1 (the frozen Hx24, bit for bit: a control), pass 1 with the ratios of pass 0's field, pass 2 with those of pass 1's.
      The ratios of pass 2's field are recorded, not used.  Every pass prepares its channels again, so s is computed again.
  AC  both: C's three passes outside, A's EM inside every pass (restarted from u = 0 each pass).

No function of this module reads a label: the channels come from fast_channels.read_predicted on the PairInput of load_pair (fixed, moving), the tissue from the Atropos layer.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
for _rel in ('fl', 'ef', 'mc0', 'p1'):
    if str(HERE.parent/_rel) not in sys.path:
        sys.path.insert(0, str(HERE.parent/_rel))

import fl_core as fl  # noqa: E402
import fx_core as fx  # noqa: E402
import mc0_core as core  # noqa: E402
import p1_core as p1  # noqa: E402

RESULTS = fl.RESULTS
NATIVE = fx.NATIVE                              # 600
FIRST = fl.FIRST
REST_TIER = f'rest{NATIVE}'
REPAIR_TIER = f'c{NATIVE}'
D_FACTOR = fx.D_FACTOR                          # 24
SPEC = fx.SPEC                                  # frozen Hx24: k 35, sigma_field .5, native 600, onehot, Q3h, D factor 24
ARMS = dict(A=dict(SPEC, em=True, tissue=False), C=dict(SPEC, em=False, tissue=True), AC=dict(SPEC, em=True, tissue=True))
ORDER = ('AC', 'A', 'C')                        # the brief's preference: the first arm through the gate goes to P3
ALL_PAIRS = fx.ALL_PAIRS
EM_NATIVE_STEPS = (200, 400)                    # the native M steps that are used (after that many native steps); plan.md, pinned before P2
EM_POST_STEP = NATIVE                           # the diagnostic M step on the final native field: recorded, never used
EM_SCHEDULE = dict(heads=(0, 1, 2, 3), native=EM_NATIVE_STEPS)
C_PASSES = 3
C_CLASSES = p1.C_CLASSES                        # ('CSF', 'GM'): the two ratios to white matter that are learned
COLLAPSE_FACTOR = 10.                           # C11-5: a sigma_c^2 falling more than 10x between two used M steps is flagged, nothing else
SCALE_TOL = core.SCALE_TOL                      # 1e-3
DEFAULT_SEGMENTER = 'unet35_v1_ens'
REFERENCE_SCHEMA = 'bayesreg.p2.reference.v1'
HS_DECISION = RESULTS/'hs_v1'/'analysis'/'decision_hs.json'     # revision 2 (the lead on the review's I2): the one file HS's reading is read from
HS_ADOPT = '采用新分割器'                                         # runs/20261106-hs/任务书.md: the reading that makes HS's channels the reference configuration
HX_CERTIFIED, HX_A0, HX_IDENTITY, HX_REJECTED = 'hx_certified', 'hx_fallback_configC', 'hx_fallback_identity', 'hx_provenance_rejected'     # C22.5-4's line_status
FALLBACK_STATUSES = (HX_A0, HX_IDENTITY)
THIRD_KEYS_ALLOWED = ('fixed', 'moving', 'initial_u')        # revision 4: what load_pair may have read in the unit of a third certificate
METHOD_KEYS = ('seg35', 'seg4', 'sdlogj_official')     # what a method-level value must hold (revision 2, the review's minor 8)
# revision 2 (the review's I1): what every row of a reference must be, whichever the segmenter: frozen Hx24@600 and RF05 with the fuse
FROZEN_HX24 = dict(k=35, weight='Q3h', D_factor=24., sigma_field=.5, iterations=[50, 50, 100, 600], eps=.05, wall_seconds=1200.)
ROLE = 'P2/P3: Bayesian parts on the Hx24@600 line (EM channel weights, tissue-dependent geometry), predicted channels: a probe, NOT oracle, NOT a method, NOT in any result table'

# the preregistered lines (plan.md)
P2_N = 5
P2_DELTA_MIN = -.004                            # the mean seg35 difference against the reference, at or above
SDLOGJ_MAX = .10                                # the mean official SDlogJ, at or below
P3_LO = -.003                                   # the 19-pair subject-block 95 % interval of the seg35 difference: its lower bound above
P3_MEAN = -.001                                 # and the mean difference at or above
SUPERIOR_LO = 0.                                # the lower bound above it: "优效也成立" (reported, not a gate)
BOOT_N, BOOT_SEED = p1.BOOT_N, p1.BOOT_SEED      # runs/20261105-fd2/recompute.py


class EstimateFailed(ValueError):
    """An M step or a moment ratio without a finite positive value: the arm's estimate failed on this pair (an entry failure, status estimate_failed; C22.5-1 (a))."""


def sha_bytes(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ---- A: the M step, the weights, the scale -----------------------------------------------------------------------------------------------------
def em_weights(R, present, n_brain):
    """C11-2 with a = b = 0: sigma_c^2 = R_c / (N_B + 2); w_c = 1/sigma_c^2 over the present labels with R_c > 0, normalized to mean 1 there.  An absent label: w 0, sigma^2 NaN.
    A present label with R_c = 0 (or not finite): w 0, sigma^2 NaN, in left_out.  Returns (w, sigma2, left_out).  ValueError when no label has a finite precision."""
    R = np.asarray(R, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    ok = present & np.isfinite(R) & (R > 0)
    if not ok.any():
        raise EstimateFailed('EM: no present label with a positive sum of squares on the brain domain')
    sigma2 = np.full(len(R), np.nan)
    sigma2[ok] = R[ok]/(n_brain+2)
    precision = np.zeros(len(R))
    precision[ok] = 1/sigma2[ok]
    w = np.zeros(len(R))
    w[ok] = precision[ok]/precision[ok].mean()
    return w, sigma2, [int(core.LABELS_35[i]) for i in np.flatnonzero(present & ~ok)]


def gram_norm(G, q):
    """sqrt(q^T G q): the 2-norm of sum_c q_c g_c when G_cc' = <g_c, g_c'>."""
    q = np.asarray(q, dtype=np.float64)
    value = float(q @ np.asarray(G, dtype=np.float64) @ q)
    return float(np.sqrt(max(value, 0.)))


def distance_to(w, w_ref):
    """RMS over the labels where both are positive of |log(w / w_ref)| (P1's d_A)."""
    return p1.distance_to_q3h(w, w_ref)


def collapse_flags(previous, current, factor=COLLAPSE_FACTOR):
    """C11-5: the labels whose sigma^2 fell by more than `factor` from one used M step to the next (record only)."""
    out = []
    for i, (a, b) in enumerate(zip(previous, current)):
        if np.isfinite(a) and np.isfinite(b) and b > 0 and a/b > factor:
            out.append(int(core.LABELS_35[i]))
    return out


def residual_sums(host, factor, present, brain_index, coords, workers=1):
    """R_c of the UNWEIGHTED channels on the brain domain: the host arrays are sqrt(factor_c) times them (factor = s0 w0, the frozen preparation), so R_c = R_c(host) / factor_c.
    coords None: u = 0, M itself; otherwise the (3, N_B) index coordinates (z, y, x) of x + u(x) at the brain voxels, trilinear with the value at the nearest face beyond the grid
    (p1_core.warp's interpolation: grid_sample bilinear, border, align_corners True).  float64."""
    from scipy.ndimage import map_coordinates
    F, M = host
    R = np.zeros(len(factor))

    def one(c):
        if not present[c] or not factor[c] > 0:
            return
        f = F[c][brain_index]
        m = M[c][brain_index] if coords is None else map_coordinates(M[c], coords, order=1, mode='nearest', prefilter=False, output=np.float64)
        d = m-f
        R[c] = float((d*d).sum())/factor[c]
    core._pool_map(one, range(len(factor)), workers)
    return R


def brain_coordinates(u_mm, spacing, brain_index):
    """The index coordinates (z, y, x) of x + u(x) at the brain voxels, u (D, H, W, 3) in mm with components (x, y, z)."""
    z, y, x = brain_index
    h = np.asarray(spacing, dtype=np.float64)
    return np.stack((z+u_mm[z, y, x, 2]/h[2], y+u_mm[z, y, x, 1]/h[1], x+u_mm[z, y, x, 0]/h[0]))


def channel_gram(host, present, device):
    """G_cc' = <g_c, g_c'> of the native data gradients at u = 0 of each host channel alone (data_term_k on the interior nodes, float32 autograd as level_grad_norms), summed in float64.
    Only the present labels (the others never weigh).  The gradients are kept on the device for the products, then freed (and the cache emptied on the GPU)."""
    import torch
    import torch.nn.functional as Fn
    dev = torch.device(device)
    F, M = host
    index = [c for c in range(len(present)) if present[c]]
    D, H, W = F.shape[1:]
    Hm = torch.empty((len(index), 3*(D-2)*(H-2)*(W-2)), device=dev, dtype=torch.float32)     # one row per present channel, filled in place: never two copies
    for row, c in enumerate(index):
        Fc = torch.tensor(F[c:c+1], device=dev, dtype=torch.float32)
        Mc = torch.tensor(M[c:c+1], device=dev, dtype=torch.float32)
        u_int = torch.zeros((3, D-2, H-2, W-2), device=dev, dtype=torch.float32, requires_grad=True)
        g, = torch.autograd.grad(core.data_term_k(Fn.pad(u_int, (1, 1, 1, 1, 1, 1)), Fc, Mc), u_int)
        Hm[row] = g.reshape(-1)
        del Fc, Mc, u_int, g
    G = torch.zeros((len(index), len(index)), device=dev, dtype=torch.float64)
    n = Hm.shape[1]
    step = 1 << 21
    for a in range(0, n, step):
        part = Hm[:, a:a+step].double()
        G += part@part.T
    out = np.zeros((len(present), len(present)))
    out[np.ix_(index, index)] = G.cpu().numpy()
    del Hm, G
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return out


class EMWeights:
    """A's weights of one T88 pass.  The host stacks are the frozen preparation (fl_channels.prepare_stacks: Q3h weights w0, scale s0); the device tensors of a level are
    r_c times them, r_c = sqrt(s w_c / (s0 w0_c)).  With no M step (schedule heads () and native ()), r = 1 and nothing is multiplied: the frozen path, bit for bit (a test)."""

    def __init__(self, host, scale, brain, spacing, device, *, schedule=None, workers=1, D=D_FACTOR, gram=None):
        self.host = (host[0], host[1])                          # the arrays themselves, not the list optimize_p2 clears at the native level
        self.w0 = np.asarray(scale['channel_weights'], dtype=np.float64)
        self.s0 = float(scale['D_multiplier'])
        self.g_q0 = float(scale['g_norm_Q0'])
        self.D = float(D)
        self.voxels = np.asarray(scale['fixed_voxels'], dtype=np.float64)
        self.present = self.voxels > 0
        self.factor = self.s0*self.w0
        self.brain_index = np.nonzero(brain)
        self.n_brain = int(len(self.brain_index[0]))
        self.spacing = tuple(float(s) for s in spacing)
        self.schedule = EM_SCHEDULE if schedule is None else schedule
        self.workers = workers
        tick = perf_counter()
        self.G_host = channel_gram(host, self.present, device) if gram is None else gram
        self.seconds_gram = perf_counter()-tick
        q0 = np.where(self.present, 1/self.s0, 0.)
        self.gram_g_norm_w0 = gram_norm(self.G_host, q0)        # ||g_w0||: fl_channels' g_norm_unscaled up to float32 rounding (a check of the Gram matrix)
        self.gram_check = dict(gram=self.gram_g_norm_w0, prepare_stacks=float(scale['g_norm_unscaled']), relative=abs(self.gram_g_norm_w0/float(scale['g_norm_unscaled'])-1), tol=SCALE_TOL)
        self.gram_check['passed'] = bool(self.gram_check['relative'] <= SCALE_TOL)
        self.w, self.s, self.sigma2, self.level = self.w0.copy(), self.s0, None, None
        self.r = np.ones(len(self.w0))
        self.history, self.scale_checks, self.native_marks = [], [], []     # native_marks: the loop's cap_hits and energy terms at each native M step (revision 2, the review's minor 4)

    def g_norm(self, w):
        q = np.where(self.present & (self.factor > 0), w/np.where(self.factor > 0, self.factor, 1.), 0.)
        return gram_norm(self.G_host, q)

    def m_step(self, where, u_mm, *, used=True):
        """One M step at a field (None: u = 0) on the native grid.  used: the weights and the scale change; otherwise only recorded."""
        tick = perf_counter()
        coords = None if u_mm is None else brain_coordinates(u_mm, self.spacing, self.brain_index)
        R = residual_sums(self.host, self.factor, self.present, self.brain_index, coords, self.workers)
        w, sigma2, left_out = em_weights(R, self.present, self.n_brain)
        norm = self.g_norm(w)
        if not (np.isfinite(norm) and norm > 0):
            raise EstimateFailed(f'EM at {where}: ||g_w|| = {norm}')
        s = self.D*self.g_q0/norm
        level = where.get('level') if isinstance(where, dict) else where
        # C11-5, D42 (revision 2, the review's minor 3): the flag compares two rounds of the same level only (the native head, 200, 400); across a level head the ratio is recorded, not flagged
        flags = collapse_flags(self.sigma2, sigma2) if self.sigma2 is not None and used and level == self.level else []
        ratios = None if self.sigma2 is None else [float(a/b) if np.isfinite(a) and np.isfinite(b) and b > 0 else None for a, b in zip(self.sigma2, sigma2)]
        d, n = distance_to(w, self.w0)
        row = dict(where=where, used=used, w=w.tolist(), sigma2=[None if not np.isfinite(v) else float(v) for v in sigma2], R=R.tolist(), left_out=left_out, s=s, g_norm_w=norm,
                   n_brain=self.n_brain, d_to_q3h=d, n_compared=n, collapse_flagged=flags, max_drop_from_previous=max((r for r in ratios or [] if r is not None), default=None), seconds=perf_counter()-tick)
        self.history.append(row)
        if used:
            self.w, self.s, self.sigma2, self.level = w, s, sigma2, level
            self.r = np.where(self.present & (self.factor > 0), np.sqrt(s*w/np.where(self.factor > 0, self.factor, 1.)), 1.)
        return row

    def apply(self, F, M, r_old=None):
        """Scale the device tensors to the current r: in place by r / r_old (r_old None: the tensors hold the host arrays, r_old = 1).  A channel at 0 that has to grow is rebuilt from the host."""
        import torch
        r_old = np.ones(len(self.r)) if r_old is None else r_old
        for c in range(len(self.r)):
            if not self.present[c] or self.r[c] == r_old[c]:
                continue
            for tensor, array in ((F, self.host[0]), (M, self.host[1])):
                if r_old[c] > 0:
                    tensor[c].mul_(float(self.r[c]/r_old[c]))
                else:
                    tensor[c].copy_(torch.tensor(array[c], device=tensor.device, dtype=tensor.dtype)*float(self.r[c]))

    def native_check(self, F, M, where):
        """||g_D|| at u = 0 of the device tensors against D x ||g_Q0||."""
        norm = core.level_grad_norms(F, M, 0.)[0]
        target = self.D*self.g_q0
        row = dict(where=where, g_norm=norm, target=target, relative=abs(norm/target-1), tol=SCALE_TOL, passed=bool(abs(norm/target-1) <= SCALE_TOL))
        self.scale_checks.append(row)
        return row

    def record(self):
        return dict(rule='C11-2, a = b = 0, brain domain; w normalized to mean 1 over the present labels; s = D ||g_Q0|| / sqrt(w^T G w); r = sqrt(s w / (s0 w0)) on the frozen host stacks',
                    schedule=dict(heads=list(self.schedule['heads']), native=list(self.schedule['native']), post=EM_POST_STEP), D_factor=self.D, s0=self.s0, w0=self.w0.tolist(), g_norm_Q0=self.g_q0,
                    n_brain=self.n_brain, gram_check=self.gram_check, seconds_gram=self.seconds_gram, steps=self.history, native_scale_checks=self.scale_checks, native_marks=self.native_marks,
                    adam='m, v kept across the native rounds; native_marks: cumulative cap_hits of the run at the native head and at each native M step, and the energy terms just before and just after the reweighting (the same u)',
                    used_final=dict(w=self.w.tolist(), s=self.s, d_to_q3h=distance_to(self.w, self.w0)[0]))


def native_seed(seed, level, grids, prolong):
    """The level's seed (mm, on the level's grid) on the native grid: None at level 0 (u = 0), the chain of prolong_displacement otherwise."""
    if level == 0:
        return None
    u = seed
    for k in range(level, len(grids)-1):
        u = prolong(u, grids[k], grids[k+1])
    return u


# ---- C: the tissue weights of the cells, the moment ratios -------------------------------------------------------------------------------------
def class_index(class_names, wm_index):
    """{'background': 0, 'CSF': .., 'GM': .., 'WM': ..}: the channels of P~ (background first, then the Atropos classes).  ValueError unless CSF and GM are there and WM is another class."""
    names = list(class_names)
    if not set(C_CLASSES) <= set(names) or names[int(wm_index)] in C_CLASSES:
        raise ValueError(f'Atropos class names {names}, wm_index {wm_index}: CSF, GM and white matter are needed')
    return dict(background=0, CSF=1+names.index('CSF'), GM=1+names.index('GM'), WM=1+int(wm_index))


@dataclass
class TissueLevel:
    """One level: the cell-centre parts of t = P_bg + P_WM + rho_CSF P_CSF + rho_GM P_GM (the classes not named weigh 1 too: tau = tau_WM) and the Omega_B cells."""
    level: int
    shape_cells: tuple
    base: np.ndarray                                    # the sum of the cell probabilities of the classes with tau = tau_WM
    csf: np.ndarray
    gm: np.ndarray
    brain_cells: np.ndarray
    n_brain_cells: int


def tissue_levels(ptilde, brain, grid, grids, downsample, index, workers=1):
    """The four levels of C03.6-12 for the fixed image's tissue: native P~ and Omega_B as given; level < 3: each class through downsample_array, renormalized over the classes,
    the mask at downsample(1[Omega_B]) >= 1/2.  Then the cell centres (p1_core.cell_centre) and the cells with all 8 nodes in the mask (p1_core.brain_cells)."""
    out = []
    K = ptilde.shape[0]
    for level, level_grid in enumerate(grids):
        if level == len(grids)-1:
            P, mask = ptilde, np.asarray(brain, dtype=bool)
        else:
            def down(array):
                values, got, _ = downsample(array, grid, level_grid.shape)
                if got.grid_id != level_grid.grid_id:
                    raise ValueError('tissue pyramid: the coarse grid differs from four_level_grids')
                return values
            parts = core._pool_map(down, [ptilde[k] for k in range(K)]+[np.asarray(brain, dtype=np.float64)], workers)
            stacked = np.stack(parts[:K])
            P = stacked/stacked.sum(axis=0)
            mask = parts[K] >= .5
        cells = p1.cell_centre(P)
        named = {index['CSF'], index['GM']}
        base = sum(cells[k] for k in range(K) if k not in named)
        bc = p1.brain_cells(mask)
        if not bc.any():
            raise ValueError(f'tissue level {level}: no cell inside the brain mask')
        out.append(TissueLevel(level, tuple(cells.shape[1:]), np.ascontiguousarray(base), np.ascontiguousarray(cells[index['CSF']]), np.ascontiguousarray(cells[index['GM']]), bc, int(bc.sum())))
    return out


def cell_weights(level, ratios):
    """(a, record) of one level: t = base + rho_CSF csf + rho_GM gm, a = m / t, m = 1 / mean over the Omega_B cells of 1/t (so that a has mean 1 there).  float64."""
    rho_csf, rho_gm = float(ratios['CSF']), float(ratios['GM'])
    if not (np.isfinite(rho_csf) and np.isfinite(rho_gm) and rho_csf > 0 and rho_gm > 0):
        raise EstimateFailed(f'tissue ratios {ratios}: finite and positive needed')
    t = level.base+rho_csf*level.csf+rho_gm*level.gm
    inv = 1/t
    m = 1/float(inv[level.brain_cells].mean())
    a = m*inv
    return a, dict(level=level.level, m=m, n_brain_cells=level.n_brain_cells, mean_brain=float(a[level.brain_cells].mean()), mean_all=float(a.mean()), min=float(a.min()), max=float(a.max()))


def moment_ratios(u_mm, spacing, ptilde, brain, class_names, wm_index, j0):
    """P1's moment matching on one native field (mm, D, H, W, 3): p1_core.geometry_sums over all cells, tissue_moments; the two ratios to white matter that are learned, and the record."""
    u = p1.voxel_field(u_mm, spacing)
    sums = p1.geometry_sums(u, spacing, tissue=ptilde, brain=brain, j0=j0)
    moments = p1.tissue_moments(sums, class_names, wm_index)
    ratios = {k: float(moments['ratio_to_wm'][k]) for k in C_CLASSES}
    if not all(np.isfinite(v) and v > 0 for v in ratios.values()):
        raise EstimateFailed(f'moment ratios {ratios}: finite and positive needed')
    described = p1.tissue_moments(sums, class_names, wm_index, '_brain')
    return ratios, dict(ratios=ratios, moments=moments, brain_cells=described, min_det=sums['min_det'], n_det_le_0=sums['n_det_le_0'], cells='all (P1, the reading); the Omega_B cells described')


class TissueWeights:
    """C's weights of one T88 pass: per level a (cells) for the ratios of the pass before; None ratios: a = 1 (no weight is passed at all)."""

    def __init__(self, levels, ratios):
        self.levels, self.ratios = levels, ratios
        self.records = []

    def tensor(self, level, dev, dtype):
        import torch
        if self.ratios is None:
            return None
        a, rec = cell_weights(self.levels[level], self.ratios)
        self.records.append(rec)
        return torch.tensor(a, device=dev, dtype=dtype)

    def record(self):
        return dict(ratios_used=self.ratios, levels=self.records, rule='a_K = m / (P_bg + P_WM + rho_CSF P_CSF + rho_GM P_GM) at the cell centre, mean 1 over the Omega_B cells of the level; div and curl equal')


# ---- the hooks and the copied optimizer --------------------------------------------------------------------------------------------------------
class Hooks:
    """What optimize_p2 asks between the frozen statements.  em, tissue: EMWeights, TissueWeights or None."""

    def __init__(self, em=None, tissue=None):
        self.em, self.tissue = em, tissue
        self.counters, self.energy = None, None         # set by optimize_p2: the loop's counters, and per level a callable giving the energy terms at the current u (records only)

    def cap_hits(self):
        return None if self.counters is None else int(self.counters['cap_hits'])

    def upload(self, level, arrays, seed, grids, dev, dtype):
        F, M = (core.stack_tensor(side, dev, dtype) for side in arrays)
        if self.em is not None and level in self.em.schedule['heads']:
            fz = core.frozen_api()
            self.em.m_step(dict(level=level, step=0, at='head'), native_seed(seed, level, grids, fz.prolong_displacement))
            self.em.apply(F, M)
            if level == len(grids)-1:
                self.em.native_check(F, M, dict(level=level, step=0))
                self.em.native_marks.append(dict(level=level, step=0, cap_hits=self.cap_hits()))
        elif self.em is not None:
            self.em.apply(F, M)
        return F, M

    def cell_weight(self, level, dev, dtype):
        return None if self.tissue is None else self.tensor_of(level, dev, dtype)

    def tensor_of(self, level, dev, dtype):
        return self.tissue.tensor(level, dev, dtype)

    def between(self, level, F, M, u_int, level_grid, h):
        """The native rounds of A: a callable(it) for the native level; None elsewhere."""
        if self.em is None or level != 3 or not self.em.schedule['native']:
            return None
        em, hooks = self.em, self

        def at(it):
            if it in em.schedule['native']:
                before = None if hooks.energy is None else hooks.energy()
                u_mm = node_field(u_int, level_grid, h)
                r_old = em.r.copy()
                em.m_step(dict(level=level, step=it, at='native'), u_mm)
                em.apply(F, M, r_old)
                em.native_check(F, M, dict(level=level, step=it))
                em.native_marks.append(dict(level=level, step=it, cap_hits=hooks.cap_hits(), energy_before=before, energy_after=None if hooks.energy is None else hooks.energy()))
        return at


def node_field(u_int, level_grid, h):
    """run_fl.node_field: u float64 (D, H, W, 3) in mm with zero faces from the loop's interior tensor."""
    import torch
    with torch.no_grad():
        out = np.zeros(level_grid.shape+(3,))
        out[1:-1, 1:-1, 1:-1] = np.moveaxis(u_int.detach().double().cpu().numpy(), 0, -1)*np.array(h)
    return out


def pre_energy_p2(u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0, cell_weight=None):
    """mc0_core.pre_energy_k (F's _pre_energy with data_term_k), every statement, with one optional argument: cell_weight (D-1, H-1, W-1) multiplies (det - 1)^2 cell by cell.
    None: the frozen expression, bit for bit (a test)."""
    import torch
    import torch.nn.functional as Fn
    u = Fn.pad(u_int, (1, 1, 1, 1, 1, 1))                           # zero faces -> (3, D, H, W)
    hz, hy, hx = spacing[2], spacing[1], spacing[0]
    data = core.data_term_k(u, fixed, moving)

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
    if cell_weight is None:
        geometry = .5*mu*(((det-1)**2).sum()+(curl**2).sum())
    else:
        geometry = .5*mu*((cell_weight*(det-1)**2).sum()+(curl**2).sum())
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


def run_level_p2(u_int, m, v, F, M, h, level, config, started, observer, counters, cell_weight=None, between=None):
    """mc0_core.run_level_k (F's _run_level_default), every statement, with two optional arguments: cell_weight goes to pre_energy_p2; between(it) is called before iteration it
    (after the wall check).  Both None: the frozen loop."""
    import torch
    first = last = None; done = 0; stopped = None
    for it in range(config.iterations[level]):
        if perf_counter()-started >= config.seconds:
            stopped = dict(level=level, iteration=it); break
        if between is not None:
            between(it)
        done = it+1
        total, terms = pre_energy_p2(u_int, F, M, h, config.mu[level], config.alpha[level], config.kappa_pre, config.j0, cell_weight)
        if first is None:
            first = terms
        last = dict(terms, total=float(total))
        if not np.isfinite(float(total)):
            raise FloatingPointError("preopt: nonfinite pre-objective")
        grad, = torch.autograd.grad(total, u_int)
        with torch.no_grad():
            g = core._smooth(grad, config.sigma_grad)
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
                u_int.copy_(core._smooth(u_int, config.sigma_field))
        if observer is not None:
            observer(dict(level=level, iteration=it, **last))
    return first, last, done, stopped


def _energy_record(u_int, F, M, h, config, level, weight):
    """A callable: the terms of pre_energy_p2 at the loop's current u on this level's tensors, without a graph (a record; it does not touch the loop's state)."""
    def terms():
        import torch
        with torch.no_grad():
            return pre_energy_p2(u_int.detach(), F, M, h, config.mu[level], config.alpha[level], config.kappa_pre, config.j0, weight)[1]
    return terms


def optimize_p2(stacks, grid, config, device="cuda", started=None, observer=None, workers=1, record_level_norms=False, downsample=None, checkpoint=None, hooks=None):
    """mc0_core.optimize_k (F's preoptimize on the default path, K channels), every statement, with one optional argument: hooks (Hooks) uploads a level's tensors, gives the cell
    weights of a level and the callable between iterations.  hooks None: mc0_core.optimize_k itself is called (the frozen path; C's pass 0).  A test runs this copy with hooks that do
    nothing (EM without M steps, a = None) and with a = 1 against optimize_k, bit for bit."""
    if hooks is None:
        return core.optimize_k(stacks, grid, config, device=device, started=started, observer=observer, workers=workers, record_level_norms=record_level_norms, downsample=downsample, checkpoint=checkpoint)
    import torch
    from dataclasses import asdict
    fz = core.frozen_api()
    started = perf_counter() if started is None else started
    grids = fz.four_level_grids(grid)
    dev = torch.device(device); dtype = torch.float32
    pyramid = core.coarse_pyramid(stacks, grid, grids, fz.downsample_array if downsample is None else downsample, workers)
    K = len(stacks[0])
    u_mm = None; levels = []; counters = dict(cap_hits=0); stopped = None; level_norms = []
    hooks.counters = counters
    for level, level_grid in enumerate(grids):
        tick = perf_counter()
        h = tuple(float(s) for s in level_grid.spacing)                 # (hx, hy, hz)
        seed = np.zeros(level_grid.shape+(3,)) if u_mm is None else fz.prolong_displacement(u_mm, grids[level-1], level_grid)
        # voxel units, interior, (3, D-2, H-2, W-2)
        seed_vox = np.moveaxis(seed[1:-1, 1:-1, 1:-1], -1, 0)/np.array(h)[:, None, None, None]
        u_int = torch.tensor(seed_vox, device=dev, dtype=dtype, requires_grad=True)
        F, M = hooks.upload(level, stacks if level == 3 else pyramid.pop(level), seed, grids, dev, dtype)
        if level == 3:
            stacks.clear()
        if record_level_norms:
            raw, smoothed = core.level_grad_norms(F, M, config.sigma_grad)
            level_norms.append(dict(level=level, raw=raw, smoothed=smoothed))
        m = torch.zeros_like(u_int); v = torch.zeros_like(u_int)
        weight = hooks.cell_weight(level, dev, dtype)
        hooks.energy = _energy_record(u_int, F, M, h, config, level, weight)
        between = hooks.between(level, F, M, u_int, level_grid, h)
        first, last, done, stopped = run_level_p2(u_int, m, v, F, M, h, level, config, started, observer if checkpoint is None else core._checkpointing_observer(observer, checkpoint, u_int, level_grid, h), counters, weight, between)
        hooks.energy = between = None                                  # the record's callable holds this level's tensors: let them go with the level
        with torch.no_grad():
            out = np.zeros(level_grid.shape+(3,))
            out[1:-1, 1:-1, 1:-1] = np.moveaxis(u_int.detach().double().cpu().numpy(), 0, -1)*np.array(h)
        u_mm = out
        levels.append(dict(level=level, shrink_factor=(8, 4, 2, 1)[level], shape_zyx=list(level_grid.shape), iterations=done,
                           seconds=perf_counter()-tick, first=first, last=last))
        del F, M, m, v, u_int, weight
        if stopped is not None:
            break
    if u_mm.shape != grid.shape+(3,):                                  # time cap on a coarse level: carry to native
        for level in range(levels[-1]["level"]+1, 4):
            u_mm = fz.prolong_displacement(u_mm, grids[level-1], grids[level])
    record = dict(policy="p2-copy-of-mc0-k-channel-fp32-preopt-with-hooks-v1", channels=K, config=asdict(config), device=str(dev), dtype="float32", levels=levels,
                  cap_hits=int(counters["cap_hits"]), stopped=stopped, seconds=perf_counter()-started, **(dict(level_g_norm=level_norms) if record_level_norms else {}),
                  role=ROLE, float32_precision=fz.precision_record(torch, dev, config))
    return np.ascontiguousarray(u_mm), record


# ---- the channels of a segmenter ---------------------------------------------------------------------------------------------------------------
class ChannelSet:
    """Which registered segmenter's predictions the arms read, and the mark computed for it (mc0p_core.oracle_mark names unet35_v1_ens only).  spec: the reference file's 'channels'."""

    def __init__(self, segmenter_id, spec=None):
        self.segmenter_id, self.spec = segmenter_id, dict(spec or {})

    @property
    def is_default(self):
        return self.segmenter_id == DEFAULT_SEGMENTER

    def source(self):
        import mc0p_core as mc
        if self.is_default:
            return mc.default_unet()
        mi = core.make_inputs()
        s = self.spec
        return mi.UNetSource(pred_root=Path(s['pred_root']), models_root=Path(s['models_root']), image_root=Path(s['image_root']), segmenter_id=self.segmenter_id, registry=core.ROOT/s['registry_rel'], gate=dict(rule='runs/20261106-p2: the reference file names this segmenter (HS adopted it)'))

    def mark(self, keys_read, sides):
        import mc0p_core as mc
        other = sorted(set(keys_read)-set(mc.IMAGE_KEYS))
        images = {'fixed', 'moving'} <= set(keys_read)
        predictions = set(sides) == {'fixed', 'moving'} and all(s.get('segmenter_id') == self.segmenter_id and s.get('policy') == mc.POLICY for s in sides.values())
        return dict(oracle=not (images and predictions and not other), truth_labels_read=bool(other))

    def read_predicted(self, pin, workers=1):
        import fast_channels as fast
        predicted = fast.read_predicted(pin, unet=self.source(), workers=workers)
        predicted.record.update(self.mark(predicted.record['pair_keys_read'], predicted.record['sides']))
        return predicted

    def oracle_of_run(self, record):
        """mc0p_core.oracle_of_run for this segmenter: False after computing the mark again from the run's channel_sources; ValueError otherwise (the field is not read)."""
        sources = record.get('channel_sources') or {}
        mark = self.mark(sources.get('pair_keys_read') or [], sources.get('sides') or {})
        if record.get('oracle') is not False or mark['oracle'] is not False:
            raise ValueError(f"run.json oracle {record.get('oracle')}; computed from its channel_sources for {self.segmenter_id}: {mark}")
        return False


# ---- the reference -----------------------------------------------------------------------------------------------------------------------------
def reference_path(segmenter_id):
    return HERE/f'reference_{segmenter_id}.json'


def reference_rel(segmenter_id):
    return f'experiments/p2/reference_{segmenter_id}.json'


def _json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def pin_line_status(pin):
    """A row's C22.5-4 line_status: `line_status`, or HS's `hs_line_status`; both there must agree; neither (v1's file): hx_certified.  None when they disagree."""
    a, b = pin.get('line_status'), pin.get('hs_line_status')
    if a is not None and b is not None and a != b:
        return None
    return a or b or HX_CERTIFIED


def reference_pair_problems(pin, pair, segmenter_id, pred_root=None):
    """(problems, line_status) of one row of a reference, from the JSON records it names (revision 2, the review's I1; no field, no label is read): that it is this pair, this segmenter,
    oracle false, the frozen Hx24 (k 35, Q3h, D 24, sigma_field .5, 50/50/100/600) repaired by RF05 with the fuse (eps .05, wall_seconds 1200), the repair's source the pinned T88 field,
    and the pins themselves.  hx_certified: the field done and the repair certified at alpha = 1 are required.  A fallback (I2 (b)): the parts that exist are checked the same way, and the
    method's value HS recorded (method_metrics) must hold seg35, seg4 and the official SDlogJ."""
    problems = []
    status = pin_line_status(pin)
    if status is None:
        return [f"line_status {pin.get('line_status')} and hs_line_status {pin.get('hs_line_status')} differ"], None
    if status not in (HX_CERTIFIED,)+FALLBACK_STATUSES:
        return [f'line_status {status}: no output of the line for this pair'], status
    certified = status == HX_CERTIFIED
    fdir, rdir = pin.get('field_dir'), pin.get('repair_dir')
    record = None if not fdir else _json(Path(fdir)/'run.json')
    if record is None:
        if certified:
            problems.append(f'no run.json at the field directory {fdir}')
    else:
        sources = record.get('channel_sources')
        sides = (sources or {}).get('sides') or {}
        checks = dict(pair=record.get('pair') == pair, k=record.get('k') == FROZEN_HX24['k'], weight=record.get('weight') == FROZEN_HX24['weight'], D_factor=record.get('D_factor') == FROZEN_HX24['D_factor'],
                      config=record.get('config') == frozen_t88_config(), channels=record.get('channels') == SPEC['channels'], taper=record.get('taper') == core.TAPER)     # revision 3 (minor 4): T88's every key
        # revision 3 (the follow-up's minor 3): a fallback row whose run stopped before it read its channels (cuda_oom, vram_stop) has channel_sources and oracle None: not refused for that
        checks['oracle'] = record.get('oracle') is False if certified else record.get('oracle') is not True
        if certified or sources is not None:
            checks.update(segmenter=sorted(sides) == ['fixed', 'moving'] and all(s.get('segmenter_id') == segmenter_id for s in sides.values()),
                          policy=sorted(sides) == ['fixed', 'moving'] and all(s.get('policy') == 'all_folds_mean' for s in sides.values()))
            if pred_root is not None:
                checks['pred_root'] = ((sources or {}).get('segmenter') or {}).get('pred_root') == str(pred_root)
        checkpoint = (record.get('checkpoints') or {}).get(str(NATIVE)) or {}
        if certified or checkpoint:
            checks.update(status=not certified or record.get('status') == 'done', field_sha256=checkpoint.get('sha256') == pin.get('field_sha256'), field_u_sha256=checkpoint.get('u_sha256') == pin.get('field_u_sha256'))
        problems += [f'run.json: {k}' for k, ok in checks.items() if not ok]
    repair = None if not rdir else _json(Path(rdir)/'repair.json')
    if repair is None:
        if certified:
            problems.append(f'no repair.json at the repair directory {rdir}')
    else:
        source = repair.get('source') or {}
        metrics = _json(Path(rdir)/'repair_metrics.json') or {}
        checks = dict(pair=repair.get('pair') == pair, oracle=repair.get('oracle') is False, eps=repair.get('eps') == FROZEN_HX24['eps'], config=repair.get('config') == frozen_repair_config(),     # revision 3: RF05's every parameter
                      source_field=source.get('field_sha256') == pin.get('field_sha256'), source_directory=fdir is not None and source.get('directory') is not None and Path(source['directory']) == Path(fdir),
                      attempt=(repair.get('attempt') or {}).get('sha256') == pin.get('attempt_sha256'))
        if certified:
            review = ((metrics.get('after') or {}).get('certificate') or {}).get('accepted_alpha') == 1.
            # revision 3 (HS revision 1b's review, new minor 2): the review certificate fails, a third certificate in a process that reads no label passes: C22.3-9 (d)'s mechanical rule, certified
            third = [] if review else third_certificate_problems(pin, pair)
            checks.update(repaired=metrics.get('repaired_u_sha256') == pin.get('repaired_u_sha256'), certified=review or not third)
            problems += [f'third certificate: {x}' for x in third] if not review and third else []
        problems += [f'repair.json: {k}' for k, ok in checks.items() if not ok]
    if not certified:
        values = pin.get('method_metrics') or {}
        if not all(isinstance(values.get(k), (int, float)) and np.isfinite(values[k]) for k in METHOD_KEYS):
            problems.append(f'{status}: method_metrics without finite {list(METHOD_KEYS)}')
        elif status == HX_A0:                       # revision 3 (the follow-up's minor 5): the same number the candidates' A0 takes, from d0_v1's records
            import rp_core as rp
            a0 = rp.fallback_status(pair)
            if not a0['qualified'] or any(values.get(m) != (a0.get('metrics') or {}).get(m) for m in METHOD_KEYS):
                problems.append(f"{status}: method_metrics are not d0_v1's A0 record of this pair (qualified {a0['qualified']}, {[(a0.get('metrics') or {}).get(m) for m in METHOD_KEYS]})")
    return problems, status


def third_certificate_problems(pin, pair):
    """[] when the row carries a third certificate that decides it as certified (C22.3-9 (d)): the record the pin names (third_certificate_file, its sha256 third_certificate_sha256) is of
    this pair, its process read no label (reads_labels false), with this line's certificate configuration, the third certificate passed at alpha = 1 on the repaired field the row pins
    (repaired_u_sha256), and its output is the repaired field.  Otherwise the reasons."""
    import rp_core as rp
    path = pin.get('third_certificate_file')
    if not path:
        return ['no third certificate record (third_certificate_file)']
    if not Path(path).is_file() or core.sha_file(path) != pin.get('third_certificate_sha256'):
        return [f'{path}: not there or not the pinned sha256']
    record = _json(path) or {}
    third = record.get('third_certificate') or {}
    keys = record.get('keys_read')
    unit = record.get('unit')
    repair = _json(Path(pin['repair_dir'])/'repair.json') if pin.get('repair_dir') else None
    checks = dict(pair=record.get('pair') == core.short_name(pair), no_label=record.get('reads_labels') is False, config=record.get('certificate_config') == dict(rp.CERTIFICATE_C),
                  passed=third.get('passed') is True and (third.get('certificate') or {}).get('accepted_alpha') == 1., field=third.get('repaired_u_sha256') == pin.get('repaired_u_sha256'), output=record.get('output') == 'repair',
                  # revision 4 (the review of revision 3, minor 1): the evidence of no label, not a declaration only: load_pair's keys_read, the unit that certified, not an evaluation unit
                  keys_read=isinstance(keys, list) and {'fixed', 'moving'} <= set(keys) <= set(THIRD_KEYS_ALLOWED),
                  unit=isinstance(unit, str) and unit.startswith('tag-') and '-eval' not in unit,
                  # and the premise of C22.3-9 (d) on this side: the deciding certificate (the repair process) passed, the review one did not
                  premise=repair is not None and repair.get('passed') is True)
    return [k for k, ok in checks.items() if not ok]


def frozen_t88_config():
    """The T88 configuration of frozen Hx24@600 as a run.json holds it: PRE_C, the 900 s wall, 50/50/100/600, sigma_field .5 (fx.Batch.arm_config's value, JSON-normalized)."""
    return json.loads(json.dumps(dict(core.PRE_C, seconds=fl.RUN_CAP_SECONDS, iterations=fl.COARSE_ITERATIONS+(fx.NATIVE,), sigma_field=SPEC['sigma_field'])))


def frozen_repair_config():
    """RF05 with the fuse as a repair.json holds it: every RepairConfig field at eps .05, wall_seconds 1200 (run_fx's repair)."""
    from dataclasses import asdict
    import repair
    return json.loads(json.dumps(asdict(repair.RepairConfig(eps=fl.REPAIR_EPS, wall_seconds=fx.FUSE_SECONDS))))


def expected_pred_root(segmenter_id, reference):
    """The pred_root the channels of this segmenter come from (ChannelSet.source): what a row's channel_sources.segmenter.pred_root must be."""
    return str(ChannelSet(segmenter_id, (reference or {}).get('channels')).source().pred_root)


def hs_reference_problems(reference, decision, decision_sha256, segmenter_id):
    """A reference against HS's own reading (revision 2 I2 (b); revision 3, the follow-up's minor 6: the reference's fallbacks are taken as HS recorded them).  A reference of another
    segmenter than unet35_v1_ens, or any reference with a fallback row, rests on hs_v1/analysis/decision_hs.json: the file the reference names (hs.decision_sha256), reading
    采用新分割器 with this segmenter_id, and every row's line_status that file's per_pair line_status; a fallback row's method_metrics that row's metrics on METHOD_KEYS.
    Problems that concern every row start with 'HS:'; a row's own with '<stage> <short>:'.  [] for v1's reference (no HS behind it)."""
    rows = [(stage, k, v) for stage in ('p2', 'p3') if stage in reference for k, v in reference[stage]['pairs'].items()]
    fallback = [r for r in rows if pin_line_status(r[2]) in FALLBACK_STATUSES]
    if segmenter_id == DEFAULT_SEGMENTER and not fallback:
        return []
    named = (reference.get('hs') or {}).get('decision_sha256')
    if decision is None or named is None or named != decision_sha256:
        return [f"HS: the reference rests on decision_hs.json, the one it names (sha256 {named}, found {decision_sha256})"]
    problems = []
    if decision.get('reading') != HS_ADOPT or decision.get('segmenter_id') != segmenter_id:
        problems.append(f"HS: decision_hs.json reads {decision.get('reading')} with {decision.get('segmenter_id')}, not {HS_ADOPT} with {segmenter_id}")
    hs = {r.get('pair'): r for r in decision.get('per_pair') or []}
    for stage, k, v in rows:
        row = hs.get(k) or {}
        status = pin_line_status(v)
        if row.get('line_status') != status:
            problems.append(f"{stage} {k}: line_status {status}, HS's row {row.get('line_status')}")
        elif status in FALLBACK_STATUSES and any((row.get('metrics') or {}).get(m) != (v.get('method_metrics') or {}).get(m) for m in METHOD_KEYS):
            problems.append(f"{stage} {k}: method_metrics are not HS's row ({[(row.get('metrics') or {}).get(m) for m in METHOD_KEYS]})")
    return problems


def load_reference(segmenter_id, path=None):
    """The reference file of a segmenter: {'schema', 'segmenter_id', 'channels', 'p2': {'pairs': {short: pins}}, 'p3': {...}}.  SystemExit when it is missing or names another segmenter."""
    path = reference_path(segmenter_id) if path is None else Path(path)
    if not path.is_file():
        raise SystemExit(f'{path}: no reference for the segmenter {segmenter_id} (the frozen Hx24 of the same channels); write it first')
    document = json.loads(path.read_text())
    if document.get('schema') != REFERENCE_SCHEMA or document.get('segmenter_id') != segmenter_id:
        raise SystemExit(f"{path}: schema {document.get('schema')}, segmenter {document.get('segmenter_id')}: not the reference of {segmenter_id}")
    for stage, pairs in (('p2', core.PAIRS), ('p3', ALL_PAIRS)):
        names = sorted((document.get(stage) or {}).get('pairs') or {})
        if names != sorted(core.short_name(p) for p in pairs):
            raise SystemExit(f'{path}: {stage} pairs {names} are not the plan')
    if any(document['p2']['pairs'][k] != document['p3']['pairs'][k] for k in document['p2']['pairs']):
        raise SystemExit(f'{path}: the p2 pins of the first 5 pairs differ from their p3 pins (plan.md §3.1: one reference field per pair)')
    if segmenter_id != DEFAULT_SEGMENTER and not {'pred_root', 'models_root', 'image_root', 'registry_rel'} <= set(document.get('channels') or {}):
        raise SystemExit(f'{path}: the channels of {segmenter_id} need pred_root, models_root, image_root, registry_rel')
    return document


# ---- the cells and the batches -----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class P2Cell(fx._Cell):
    PREFIX = 'p2'
    ARMS = ARMS


@dataclass(frozen=True)
class P3Cell(fx._Cell):
    PREFIX = 'p3'
    ARMS = ARMS


@dataclass(frozen=True)
class RefCell(fx._Cell):
    PREFIX = 'ref'
    ARMS = {'Hx24': SPEC}


class _Batch(fx.Batch):
    """What run_fl.py and decide_fl.py ask of fl_core, for one root.  The segmenter and the reference are set at the start of a queue (batch.json) or read from batch.json by a unit."""
    ARMS = ARMS
    FULL, TWO = (NATIVE,), (NATIVE,)
    REPAIR_STEPS = (NATIVE,)
    CELLS_TEXT = ''

    def __init__(self):
        super().__init__()
        self.channels = ChannelSet(DEFAULT_SEGMENTER)

    def use_segmenter(self, segmenter_id, reference=None):
        reference = load_reference(segmenter_id) if reference is None else reference
        self.channels = ChannelSet(segmenter_id, reference.get('channels'))
        self.reference = reference
        return reference

    def segmenter_of_root(self, root):
        """The segmenter of a batch from its batch.json; the default when there is none yet (a unit is only ever started by the queue, after batch.json)."""
        from decide_ef import read_json
        return (read_json(Path(root)/'batch.json') or {}).get('segmenter_id', DEFAULT_SEGMENTER)

    def tier_of(self, pair):
        return FIRST if pair in core.PAIRS else REST_TIER

    def repair_tier_of(self, pair):
        return REPAIR_TIER if pair in core.PAIRS else REST_TIER

    def field_cell(self, arm, pair):
        return self.Cell(arm, self.tier_of(pair), pair)

    def repair_cell(self, arm, pair):
        return self.Cell('R'+arm, self.repair_tier_of(pair), pair)

    def chosen_repairs(self, root):
        return [dict(arm=arm, step=NATIVE) for arm in self.ARMS]

    def check(self, arm, tier, pair, arms, pairs):
        field, repair = arm in arms, arm[:1] == 'R' and arm[1:] in arms
        ok = pair in pairs and ((field and tier == self.tier_of(pair)) or (repair and tier == self.repair_tier_of(pair)))
        if not ok:
            self.refuse(arm, tier, pair)
        return self.Cell(arm, tier, pair)


class P2Batch(_Batch):
    PREFIX, ROOT_NAME, Cell, FIRST_PHASE = 'p2', 'p2_v1', P2Cell, ORDER
    ROLE = ROLE
    CELLS_TEXT = f'arms {list(ARMS)} tier {FIRST} and their repairs R<arm> tier {REPAIR_TIER}, on the first 5 development pairs only'

    def require_cell(self, arm, tier, pair):
        """Checked before any file of the pair is opened."""
        return self.check(arm, tier, pair, ARMS, core.PAIRS)

    def plan_fields(self):
        """Pair-major, the arm order rotating with the pair index (MC0P's rule)."""
        return [self.Cell(ORDER[(i+j) % len(ORDER)], FIRST, pair) for i, pair in enumerate(core.PAIRS) for j in range(len(ORDER))]

    def plan_repairs(self):
        return [self.repair_cell(c.arm, c.pair) for c in self.plan_fields()]

    def all_cells(self, root):
        return self.plan_fields()+self.plan_repairs()


class P3Batch(_Batch):
    PREFIX, ROOT_NAME, Cell = 'p3', 'p3_v1', P3Cell
    ROLE = ROLE
    ARM_FILE = 'p3_arm.json'
    CELLS_TEXT = f'only the arm P2 chose ({ARM_FILE}): tier {FIRST} on the first 5 pairs, {REST_TIER} on the other 14, and its repairs'

    def arm_of(self, root):
        """The arm P2's reading chose, as this batch wrote it once (analysis/p3_arm.json); None before."""
        return (self.read_selection(root, self.ARM_FILE) or {}).get('arm')

    def require_cell(self, arm, tier, pair):
        return self.check(arm, tier, pair, ARMS, ALL_PAIRS)

    def plan_fields_of(self, arm):
        return [self.field_cell(arm, pair) for pair in ALL_PAIRS]

    def plan_repairs_of(self, arm):
        return [self.repair_cell(arm, pair) for pair in ALL_PAIRS]

    def all_cells(self, root):
        arm = self.arm_of(root)
        return [] if arm is None else self.plan_fields_of(arm)+self.plan_repairs_of(arm)

    def chosen_repairs(self, root):
        arm = self.arm_of(root)
        return [] if arm is None else [dict(arm=arm, step=NATIVE)]


class DirectoryBatch:
    """decide_fl.repair_row on a directory given as the root: the reference's repairs of another batch, read with the same function."""
    FIRST = FIRST
    PREFIX = 'p2ref'

    def run_dir(self, root, cell):
        return Path(root)

    def steps_of(self, cell):
        return NATIVE


P2, P3 = P2Batch(), P3Batch()
BATCHES = dict(p2=P2, p3=P3)
