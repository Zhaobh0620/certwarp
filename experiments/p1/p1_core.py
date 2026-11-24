"""P1 (runs/20261105-p1/任务书.md, plan.md; docs/bayesreg/notes/贝叶斯化Hx线_设计_20261105.md §5.2): the numbers of a CPU diagnostic on the 19 repaired
fields of RW.  No registration, no truth label, no metrics file, no Dice.  Pure numpy / scipy here; the reading of the files is run_p1.py.

P1a  per channel c = 1..35 (G_.75 of the argmax of unet35_v1_ens, fast_channels' formula, one channel at a time) the residual rho_c = M_c(x + u(x)) - F_c(x)
     at u = 0 and at the repaired field u_hat, its sums of squares over the grid, over the brain mask Omega_B (contract C03.3-5: the largest 6-connected
     component of the fixed image's Atropos tissue sum > 1/2) and over the support S_c = Omega_B and (F_c > theta or M_c o phi > theta) (C04-6 with the channel
     values themselves: these channels are not square roots).  The M step of C11-2 with a = b = 0 then gives sigma_c^2 on each domain:
         brain           sigma_c^2 = R_c^B / N_B            w_c proportional to 1/R_c^B
         support         sigma_c^2 = R_c^S / N_c            w_c proportional to N_c/R_c^S
         support_shared  sigma_S^2 = sum_c R_c^S / sum_c N_c, every channel the same: w_c = 1
     over the labels present in the fixed image's predicted map (V_c > 0, the labels Q3h weighs), normalized to mean 1 as Q3h.  Per pair the distance
     d_A = RMS_c |log(w_EM / w_Q3h)| (revision 1: the reading of A), the least squares slope beta of log w on log V_c (Q3h: -1/2 exactly; description),
     its R^2, and the ratio sum_g w_EM / sum_g w_Q3h of four label groups.
P1b  on u_hat, per cell centre, det F - 1 and curl by the differences of mc0_core.pre_energy_k (forward along one axis, averaged along the other two), the
     moments tau_k^2 = sum_K P_k(K) (det_K - 1)^2 / sum_K P_k(K) with P the fixed image's Atropos posteriors after C03.2 and C03.5 (background first,
     smoothed, floored) at the cell centre (the mean of its 8 nodes), the ratios tau_k^2 / tau_WM^2 (CSF and GM enter the reading of C), and three forms of
     the curl / volume balance (description).
Controls  (ii) these formulas against pre_energy_k (torch, CPU, float64) on small synthetic fields, relative difference <= 1e-12; (iii) the T88 terms of
     field_0600 recomputed beside the recorded ones, flagged beyond 5 % for D or G.  (i) is a comparison of records: run_p1.py.
"""
import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, map_coordinates

import fast_channels as fast

LABELS = tuple(range(1, 36))
SMALL_SIX = (4, 23, 18, 34, 19, 35)                 # experiments/gate_a/summarize_warm.py
LARGE_SIX = (2, 21, 1, 20, 5, 24)                   # experiments/d0/evaluate.py SEG35_LARGE
CORTEX = (2, 21)
WHITE_MATTER = (1, 20)                              # make_inputs.CEREBRAL_WM
GROUPS = dict(small_six=SMALL_SIX, large_six=LARGE_SIX, cortex=CORTEX, white_matter=WHITE_MATTER)
THETA = .05                                         # D50's default, compared with the channel values themselves
DOMAINS = ('brain', 'support', 'support_shared')
FIELDS = ('u0', 'uhat')
CONTROL_II_TOL = 1e-12
BOOT_N, BOOT_SEED = 10000, 20261105                 # runs/20261105-fd2/recompute.py, runs/20261105-by/check_by.py
N_SUBJECTS = 20

# the preregistered lines (plan.md; revision 1 by the lead's rulings on the review, runs/20261105-p1/plan.md 修订 1)
A_NULL = float(np.log(1.25))                        # A: the 19-pair median of d_A = RMS_c |log(w_EM,brain(u_hat) / w_Q3h)| at or below: "A 是空操作"
BETA_LINE = (-.6, -.4)                              # beta_EM,brain(u_hat), 19-pair median: description only since revision 1
NULL_BAND = (.8, 1.25)                              # C: the block intervals of tau_CSF^2 / tau_WM^2 AND tau_GM^2 / tau_WM^2, each wholly inside: "C 是空操作"
SUBSTANTIAL = 1.5                                   # C: either ratio's 19-pair median >= 1.5 or <= 1/1.5: "C 是实质改动"
C_CLASSES = ('CSF', 'GM')
III_TOL = .05                                       # control (iii): |recomputed / recorded - 1| > .05 for D or G on any pair: abnormal, the readings of A and C held


# ---- channels ------------------------------------------------------------------------------------------------------------------------------------
def label_boxes(L):
    return find_objects(np.asarray(L).astype(np.int32), max_label=len(LABELS))


def onehot_channel(L, index, boxes):
    """fast_channels.onehot_channels(L)[index] bit for bit, alone: (nz, ny, nx) float64, G_.75 of the mask of label LABELS[index], 0 outside its box grown by the kernel radius."""
    L = np.asarray(L)
    out = np.zeros(L.shape, dtype=np.float64)
    box = boxes[index]
    if box is not None:
        crop = fast._grown(box, L.shape, (fast.RADIUS,)*3)
        out[crop] = gaussian_filter((L[crop] == fast.LABELS[index]).astype(np.float64), sigma=fast.SIGMA, mode='nearest', truncate=fast.TRUNCATE)
    return out


# ---- the warp ------------------------------------------------------------------------------------------------------------------------------------
def voxel_field(u_mm, spacing):
    """(D, H, W, 3) mm, components (x, y, z) as optimize_k and the repair store it  ->  (3, D, H, W) voxel units, the u of pre_energy_k (after its padding)."""
    u_mm = np.asarray(u_mm, dtype=np.float64)
    return np.ascontiguousarray(np.moveaxis(u_mm, -1, 0)/np.asarray(spacing, dtype=np.float64)[:, None, None, None])


def sample_coordinates(u):
    """The index coordinates (z, y, x) at which data_term_k samples the moving channels: x + u(x), u (3, D, H, W) voxel units with components (x, y, z)."""
    D, H, W = u.shape[1:]
    zz, yy, xx = np.meshgrid(np.arange(D, dtype=np.float64), np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing='ij')
    return np.stack((zz+u[2], yy+u[1], xx+u[0]))


def warp(M, coords):
    """M(x + u(x)): trilinear, the value at the nearest face beyond the grid (grid_sample bilinear, padding border, align_corners True).  coords None: u = 0, M itself."""
    if coords is None:
        return M
    return map_coordinates(M, coords, order=1, mode='nearest', prefilter=False, output=np.float64)


def channel_sums(F, warped, brain, theta=THETA):
    """The sums of rho^2 = (warped - F)^2 over the grid, the brain mask, the support (brain and (F > theta or warped > theta)), and the support's size."""
    rho = warped-F
    sq = rho*rho
    support = brain & ((F > theta) | (warped > theta))
    return dict(R_grid=float(sq.sum()), R_brain=float(sq[brain].sum()), R_support=float(sq[support].sum()), N_support=int(np.count_nonzero(support)))


# ---- the M step and its reading ------------------------------------------------------------------------------------------------------------------
def q3h_weights(voxels):
    """mc0p_core.channel_weights('Q3h', ...) from the counts: V^(-1/2) on the present labels, mean 1, 0 elsewhere (a test compares)."""
    voxels = np.asarray(voxels, dtype=np.float64)
    present = voxels > 0
    w = np.zeros(len(voxels))
    w[present] = 1/np.sqrt(voxels[present])
    return w/w[present].mean()


def em_weights(sums, voxels, n_brain, domain):
    """(w (35,) with NaN where it is not defined, sigma^2 (35,) or the shared one, the labels left out).  C11-2 with a = b = 0 on the domain; only the labels
    present in the fixed image (V > 0); a label whose sum of squares (or support) is 0 has no finite precision and is left out of the mean.  Normalized to mean 1."""
    voxels = np.asarray(voxels, dtype=np.float64)
    present = voxels > 0
    R_brain = np.array([s['R_brain'] for s in sums])
    R_support = np.array([s['R_support'] for s in sums])
    N = np.array([s['N_support'] for s in sums], dtype=np.float64)
    sigma2 = np.full(len(voxels), np.nan)
    if domain == 'brain':
        ok = present & (R_brain > 0)
        sigma2[ok] = R_brain[ok]/n_brain
    elif domain == 'support':
        ok = present & (N > 0) & (R_support > 0)
        sigma2[ok] = R_support[ok]/N[ok]
    elif domain == 'support_shared':
        ok = present & (N > 0)
        total_R, total_N = float(R_support[ok].sum()), float(N[ok].sum())
        if not (total_R > 0 and total_N > 0):
            ok = np.zeros_like(ok)
        sigma2 = total_R/total_N if ok.any() else None
        w = np.where(ok, 1., np.nan)                 # one sigma for every channel: equal weights, exactly 1 after the normalization
        return w, sigma2, [int(LABELS[i]) for i in np.flatnonzero(present & ~ok)]
    else:
        raise ValueError(f'no domain {domain}')
    left_out = [int(LABELS[i]) for i in np.flatnonzero(present & ~ok)]
    w = np.full(len(voxels), np.nan)
    if ok.any():
        precision = 1/sigma2[ok]
        w[ok] = precision/precision.mean()
    return w, sigma2, left_out


def fit_beta(w, voxels):
    """Least squares log w = a + beta log V over the labels with a finite positive w.  (beta, a, R^2, n); R^2 None when log w does not vary (support_shared)."""
    voxels = np.asarray(voxels, dtype=np.float64)
    ok = np.isfinite(w) & (w > 0) & (voxels > 0)
    n = int(ok.sum())
    if n < 3:
        return dict(beta=None, intercept=None, r2=None, n=n)
    x, y = np.log(voxels[ok]), np.log(w[ok])
    if np.all(y == y[0]):
        return dict(beta=0., intercept=float(y[0]), r2=None, n=n)
    xm, ym = x.mean(), y.mean()
    beta = float(((x-xm)*(y-ym)).sum()/((x-xm)**2).sum())
    a = float(ym-beta*xm)
    residual = y-(a+beta*x)
    return dict(beta=beta, intercept=a, r2=float(1-(residual**2).sum()/((y-ym)**2).sum()), n=n)


def group_ratios(w_em, w_q3h):
    """{group: sum over its labels of w_EM / the same sum of w_Q3h}, the labels where both are finite and positive; None when the group has none.  Both
    weights have mean 1 over their labels, so a ratio above 1 is a larger share of the weight under EM."""
    out = {}
    for name, labels in GROUPS.items():
        idx = [LABELS.index(label) for label in labels]
        keep = [i for i in idx if np.isfinite(w_em[i]) and w_em[i] > 0 and w_q3h[i] > 0]
        out[name] = None if not keep else float(sum(w_em[i] for i in keep)/sum(w_q3h[i] for i in keep))
    return out


def distance_to_q3h(w_em, w_q3h):
    """d_A = RMS over the labels of |log(w_EM / w_Q3h)| (the root of the mean square of the log ratio), the labels where both are finite and positive (both
    weights have mean 1 over the labels present in the fixed image).  (d, n); d None when no label has both."""
    w_em, w_q3h = np.asarray(w_em, dtype=np.float64), np.asarray(w_q3h, dtype=np.float64)
    ok = np.isfinite(w_em) & (w_em > 0) & np.isfinite(w_q3h) & (w_q3h > 0)
    if not ok.any():
        return None, 0
    r = np.log(w_em[ok]/w_q3h[ok])
    return float(np.sqrt(np.mean(r*r))), int(ok.sum())


# ---- the cell-centre terms of pre_energy_k -------------------------------------------------------------------------------------------------------
def cell_terms(u, spacing):
    """(det, curl) at the cell centres of a node block u (3, d, H, W) voxel units, components (x, y, z): pre_energy_k's d(axis), J, det, curl, operation for
    operation.  spacing (hx, hy, hz).  det (d-1, H-1, W-1), curl (3, d-1, H-1, W-1)."""
    hx, hy, hz = (float(s) for s in spacing)

    def narrow(g, ax, start, length):                              # torch's narrow(1 + ax, start, length)
        index = [slice(None)]*g.ndim
        index[1+ax] = slice(start, start+length)
        return g[tuple(index)]

    def d(axis):
        g = np.diff(u, axis=1+axis)
        for ax in range(3):
            if ax != axis:
                g = .5*(narrow(g, ax, 0, g.shape[1+ax]-1)+narrow(g, ax, 1, g.shape[1+ax]-1))
        return g
    gz_, gy_, gx_ = d(0), d(1), d(2)
    h = (hx, hy, hz)

    def J(c, dd):
        g = (gx_, gy_, gz_)[dd][c]
        return g*(h[c]/h[dd])
    F00 = 1+J(0, 0); F01 = J(0, 1); F02 = J(0, 2)
    F10 = J(1, 0); F11 = 1+J(1, 1); F12 = J(1, 2)
    F20 = J(2, 0); F21 = J(2, 1); F22 = 1+J(2, 2)
    det = F00*(F11*F22-F12*F21)-F01*(F10*F22-F12*F20)+F02*(F10*F21-F11*F20)
    curl = np.stack((F21-F12, F02-F20, F10-F01))
    return det, curl


def slabs(n_cells, size):
    edges = list(range(0, n_cells, size))+[n_cells]
    return list(zip(edges[:-1], edges[1:]))


def cell_centre(p):
    """The value of the trilinear interpolant of node values p (C, d, H, W) at the cell centres: the mean of the 8 corners, (C, d-1, H-1, W-1)."""
    s = (p[:, :-1, :-1, :-1]+p[:, :-1, :-1, 1:]+p[:, :-1, 1:, :-1]+p[:, :-1, 1:, 1:]
         + p[:, 1:, :-1, :-1]+p[:, 1:, :-1, 1:]+p[:, 1:, 1:, :-1]+p[:, 1:, 1:, 1:])
    return .125*s


def brain_cells(brain):
    """The cells whose 8 nodes are all in the node mask brain (d, H, W)  ->  (d-1, H-1, W-1) bool."""
    b = np.asarray(brain, dtype=bool)
    return (b[:-1, :-1, :-1] & b[:-1, :-1, 1:] & b[:-1, 1:, :-1] & b[:-1, 1:, 1:] & b[1:, :-1, :-1] & b[1:, :-1, 1:] & b[1:, 1:, :-1] & b[1:, 1:, 1:])


def geometry_sums(u, spacing, *, tissue=None, brain=None, j0=.2, slab=16):
    """The cell-centre sums of a whole field u (3, D, H, W) voxel units with zero faces, slab by slab along z (a cell needs its 8 nodes only, so each cell's
    values are the ones of the whole array).  tissue: (K+1, D, H, W) node probabilities or None.  Returns the totals and, with tissue, per class k:
    sum P_k, sum P_k (det - 1)^2, sum P_k |curl|^2.  brain: a node mask or None; with it the same sums again over the cells with all 8 nodes in it
    (the keys with the suffix _brain; description only)."""
    n_cells = u.shape[1]-1
    parts = ('',) if brain is None else ('', '_brain')
    out = dict(sum_floor0=0., min_det=np.inf, n_det_le_0=0)
    K = None if tissue is None else tissue.shape[0]
    for s in parts:
        out.update({f'n_cells{s}': 0, f'sum_e{s}': 0., f'sum_q{s}': 0.})
        if K is not None:
            out.update({f'sum_p{s}': np.zeros(K), f'sum_pe{s}': np.zeros(K), f'sum_pq{s}': np.zeros(K)})
    for z0, z1 in slabs(n_cells, slab):
        det, curl = cell_terms(u[:, z0:z1+1], spacing)
        e = (det-1)**2
        q = (curl**2).sum(axis=0)
        out['sum_floor0'] += float((np.maximum(j0-det, 0.)**2).sum())
        out['min_det'] = min(out['min_det'], float(det.min()))
        out['n_det_le_0'] += int(np.count_nonzero(det <= 0))
        P = None if K is None else cell_centre(tissue[:, z0:z1+1])
        for s in parts:
            keep = None if s == '' else brain_cells(brain[z0:z1+1])
            ee, qq = (e, q) if keep is None else (e[keep], q[keep])
            out[f'n_cells{s}'] += ee.size
            out[f'sum_e{s}'] += float(ee.sum())
            out[f'sum_q{s}'] += float(qq.sum())
            if K is not None:
                PP = P.reshape(K, -1) if keep is None else P[:, keep]
                out[f'sum_p{s}'] += PP.sum(axis=1)
                out[f'sum_pe{s}'] += (PP*ee.reshape(-1)).sum(axis=1)
                out[f'sum_pq{s}'] += (PP*qq.reshape(-1)).sum(axis=1)
    out['j0'] = j0
    if K is not None:
        for s in parts:
            out.update({k: out[k].tolist() for k in (f'sum_p{s}', f'sum_pe{s}', f'sum_pq{s}')})
    return out


def tissue_moments(sums, class_names, wm_index, part=''):
    """tau_k^2 = sum P_k e / sum P_k, curl moments kappa_k^2 = sum P_k q / sum P_k, ratios to white matter; and the curl / volume balance in three forms, all
    description (P1 does not judge b; revision 1):
      curl_to_det = mean |curl|^2 / mean (det - 1)^2, the design note's formula (a ratio of variances);
      b_per_component = 3 mean (det - 1)^2 / mean |curl|^2, moment matching of precisions with one (det - 1) and three curl components per cell;
      b_helmholtz = 2 mean (det - 1)^2 / mean |curl|^2, the same with the Helmholtz count, one divergence and two curl degrees of freedom per node (B13).
    part '' : all cells; '_brain': the cells inside the brain mask (geometry_sums' keys with that suffix)."""
    names = ['background']+list(class_names)
    p, pe, pq = (np.asarray(sums[k+part]) for k in ('sum_p', 'sum_pe', 'sum_pq'))
    tau2 = pe/p
    kappa2 = pq/p
    wm = 1+int(wm_index)
    mean_e, mean_q = sums['sum_e'+part]/sums['n_cells'+part], sums['sum_q'+part]/sums['n_cells'+part]
    return dict(classes=names, tau2=dict(zip(names, tau2.tolist())), kappa2=dict(zip(names, kappa2.tolist())),
                ratio_to_wm=dict(zip(names, (tau2/tau2[wm]).tolist())), wm_class=names[wm], mean_e=mean_e, mean_q=mean_q, n_cells=sums['n_cells'+part],
                curl_to_det=mean_q/mean_e if mean_e > 0 else None, b_per_component=3*mean_e/mean_q if mean_q > 0 else None, b_helmholtz=2*mean_e/mean_q if mean_q > 0 else None)


# ---- the numpy pre-energy: control (ii) and (iii) ------------------------------------------------------------------------------------------------
def numpy_pre_energy(u_int, F, M, spacing, mu, kappa_pre, j0):
    """pre_energy_k's record with alpha = 0 from the functions of this module: u_int (3, D-2, H-2, W-2) voxel units, F, M (K, D, H, W).  data = 1/2 sum_c R_c over
    the grid, geometry = mu/2 (sum (det-1)^2 + sum |curl|^2), floor = kappa sum relu(j0 - det)^2, min_det."""
    u = np.pad(np.asarray(u_int, dtype=np.float64), ((0, 0), (1, 1), (1, 1), (1, 1)))
    coords = sample_coordinates(u)
    data = 0.
    full = np.ones(F.shape[1:], dtype=bool)
    for c in range(len(F)):
        data += channel_sums(F[c], warp(M[c], coords), full)['R_grid']
    data *= .5
    g = geometry_sums(u, spacing, j0=j0, slab=max(1, (u.shape[1]-1)//3))
    geometry = .5*mu*(g['sum_e']+g['sum_q'])
    floor = kappa_pre*g['sum_floor0']
    return dict(data=data, geometry=geometry, floor=floor, min_det=g['min_det'], total=data+geometry+floor)


def synthetic_case(seed, shape=(9, 11, 10), spacing=(1., 1., 1.), K=3, amplitude=1.6, far=True):
    """A small field and K smooth channels: control (ii).  far: one node displaced 3.5 voxels beyond a face (a clamped sample, and folded cells around it)."""
    rng = np.random.default_rng(seed)
    D, H, W = shape
    u_int = gaussian_filter(rng.normal(size=(3, D-2, H-2, W-2)), (0, .7, .7, .7))*amplitude/.35
    if far:
        u_int[0, 0, 0, 0] = -3.5
    F = gaussian_filter(rng.uniform(size=(K,)+shape), (0, 1., 1., 1.))
    M = gaussian_filter(rng.uniform(size=(K,)+shape), (0, 1., 1., 1.))
    return u_int, F, M, spacing


SETTINGS_FOLDED = ((1., 10., .2), (.003, 10., .2), (1., 1., 1.5))      # (mu, kappa, j0): native mu, and a floor active on most cells
SETTINGS_SMALL = ((1., 10., 1.), (.003, 10., 1.), (1., 1., 1.5))       # a field that does not fold: j0 at 1 keeps the floor on about half the cells
# (seed, shape, spacing, K, amplitude, far, settings, folded): three heavily folded fields with samples beyond the faces, and one of the size of the T88
# fields (|det - 1| well below 1, no fold; revision 1, review minor 5)
CONTROL_II_CASES = ((11, (9, 11, 10), (1., 1., 1.), 3, 1.6, True, SETTINGS_FOLDED, True), (12, (8, 10, 12), (1., 1.25, .75), 2, 1.2, True, SETTINGS_FOLDED, True),
                    (13, (10, 9, 11), (.9, 1., 1.1), 1, 2., True, SETTINGS_FOLDED, True), (14, (10, 11, 9), (1., 1.1, .9), 2, .1, False, SETTINGS_SMALL, False))


def control_ii(pre_energy_k):
    """numpy_pre_energy against pre_energy_k (torch, CPU, float64) on every case and its settings: data, geometry, floor, min_det, total; pass when every relative
    difference is <= 1e-12, the floor is active on every row (so that det is compared, not only its square) and each case folds or not as it is meant to."""
    import torch
    rows, worst = [], 0.
    for seed, shape, spacing, K, amplitude, far, settings, folded in CONTROL_II_CASES:
        u_int, F, M, spacing = synthetic_case(seed, shape, spacing, K, amplitude, far)
        for mu, kappa, j0 in settings:
            ours = numpy_pre_energy(u_int, F, M, spacing, mu, kappa, j0)
            ut = torch.tensor(u_int, dtype=torch.float64)
            _, theirs = pre_energy_k(ut, torch.tensor(F, dtype=torch.float64), torch.tensor(M, dtype=torch.float64), spacing, mu, 0., kappa, j0)
            rel = {k: abs(ours[k]-theirs[k])/max(abs(theirs[k]), 1e-300) for k in ('data', 'geometry', 'floor', 'min_det')}
            rel['total'] = abs(ours['total']-(theirs['data']+theirs['geometry']+theirs['floor']))/abs(theirs['data']+theirs['geometry']+theirs['floor'])
            worst = max(worst, max(rel.values()))
            rows.append(dict(seed=seed, shape=list(shape), spacing=list(spacing), K=K, amplitude=amplitude, mu=mu, kappa=kappa, j0=j0, numpy=ours, torch=dict(theirs), relative=rel,
                             floor_active=theirs['floor'] > 0, folded=theirs['min_det'] <= 0, meant_folded=folded))
    passed = worst <= CONTROL_II_TOL and all(r['floor_active'] and r['folded'] == r['meant_folded'] for r in rows)
    return dict(control='ii', tolerance=CONTROL_II_TOL, max_relative=worst, passed=bool(passed), cases=rows,
                compared='numpy (this module) against experiments/mc0/mc0_core.pre_energy_k, torch CPU float64, alpha 0')


# ---- the 19 pairs: block bootstrap and the reading -----------------------------------------------------------------------------------------------
def block_interval(values, index, *, log=False, n=BOOT_N, seed=BOOT_SEED):
    """The 95 % subject-block bootstrap interval of the mean over pairs (runs/20261105-fd2/recompute.py block): the 20 subjects resampled, pair i = (subject i,
    i + 1) weighs the product of their counts.  index: each value's pair number 0..18.  log: of the mean of log values, returned exponentiated (a geometric mean)."""
    x = np.log(np.asarray(values, dtype=np.float64)) if log else np.asarray(values, dtype=np.float64)
    index = np.asarray(index)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        counts = np.bincount(rng.integers(0, N_SUBJECTS, N_SUBJECTS), minlength=N_SUBJECTS)
        w = counts[index]*counts[index+1]
        if w.sum():
            out.append(float((w*x).sum()/w.sum()))
    lo, hi = np.percentile(out, [2.5, 97.5])
    centre = float(x.mean())
    if log:
        return dict(lo=float(np.exp(lo)), hi=float(np.exp(hi)), centre=float(np.exp(centre)), scale='log (geometric mean)', n=n, seed=seed)
    return dict(lo=float(lo), hi=float(hi), centre=centre, scale='linear (mean)', n=n, seed=seed)


def beta_band(beta_median):
    """Description only since revision 1: where the volume exponent of the EM weights lies against Q3h's -1/2."""
    lo, hi = BETA_LINE
    if beta_median is None:
        return None
    if beta_median < lo:
        return '体积指数比 −½ 陡'
    if beta_median > hi:
        return '体积指数比 −½ 平'
    return '体积指数接近 −½'


def read_a(d_median, beta_median=None):
    """A: the 19-pair median of d_A at or below log 1.25: "A 是空操作", not to P2; above it: to P2.  beta: a band, described beside it."""
    band = beta_band(beta_median)
    if d_median is None:
        return dict(reading='no reading: d_A not defined', to_p2=None, beta_band=band)
    if d_median <= A_NULL:
        return dict(reading='A 是空操作', to_p2=False, beta_band=band, d_median=d_median, line=A_NULL)
    return dict(reading='A 不是空操作', to_p2=True, beta_band=band, d_median=d_median, line=A_NULL)


def read_c(intervals, medians):
    """C: the block intervals (geometric mean) of tau_CSF^2/tau_WM^2 and tau_GM^2/tau_WM^2 both wholly inside [.8, 1.25]: "C 是空操作", not to P2; otherwise either
    median >= 1.5 or <= 1/1.5: "C 是实质改动"; otherwise "C 是弱改动"; both to P2.  intervals, medians: {class: ...} for C_CLASSES."""
    lo, hi = NULL_BAND
    if any((intervals or {}).get(k) is None or (medians or {}).get(k) is None for k in C_CLASSES):
        return dict(reading='no reading: a ratio is not defined', to_p2=None)
    if all(intervals[k]['lo'] >= lo and intervals[k]['hi'] <= hi for k in C_CLASSES):
        return dict(reading='C 是空操作', to_p2=False)
    beyond = [dict(cls=k, median=medians[k], direction=f'{k} 比 WM 松' if medians[k] >= SUBSTANTIAL else f'{k} 比 WM 紧（与设计预期的方向相反）' if k == 'CSF' else f'{k} 比 WM 紧')
              for k in C_CLASSES if medians[k] >= SUBSTANTIAL or medians[k] <= 1/SUBSTANTIAL]
    if beyond:
        return dict(reading='C 是实质改动', to_p2=True, beyond=beyond)
    return dict(reading='C 是弱改动', to_p2=True)


def iii_abnormal(rows, tol=III_TOL):
    """Control (iii)'s preregistered flag (revision 1): the pairs where |recomputed / recorded - 1| > tol for D or G (a ratio that is not defined counts)."""
    out = []
    for r in rows:
        bad = [k for k in ('data', 'geometry') if r['ratio'].get(k) is None or not abs(r['ratio'][k]-1) <= tol]
        if bad:
            out.append(dict(pair=r['pair'], terms=bad, ratio={k: r['ratio'].get(k) for k in ('data', 'geometry')}))
    return out


def decide(controls_passed, d_median, beta_median, c_intervals, c_medians, abnormal=()):
    """The preregistered reading (plan.md, revision 1).  Controls (i), (ii) not passed: no reading.  Control (iii) abnormal on any pair: A and C are read
    but held (to_p2 None), for the lead; nothing stops."""
    if not controls_passed:
        return dict(stopped=True, held=False, A=None, C=None, to_p2=[], text='控制 (i) 或 (ii) 不过：工具停，不读其余')
    a, c = read_a(d_median, beta_median), read_c(c_intervals, c_medians)
    if abnormal:
        return dict(stopped=False, held=True, A=a, C=c, to_p2=None, abnormal=list(abnormal), text='控制 (iii) 异常：A、C 的读数暂缓，交总负责人处理')
    return dict(stopped=False, held=False, A=a, C=c, to_p2=[name for name, r in (('A', a), ('C', c)) if r['to_p2']])


def check_keys_read(keys_read, allowed=('fixed', 'moving', 'initial_u')):
    """The firewall of pair.npz: only the two images (and the all-zero initial_u).  Anything else read: STOP."""
    other = sorted(set(keys_read)-set(allowed))
    if other or not {'fixed', 'moving'} <= set(keys_read):
        raise SystemExit(f'P1: pair.npz members read {sorted(keys_read)}: only {list(allowed)} may be; refused, nothing computed')
    return list(keys_read)
