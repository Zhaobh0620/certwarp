"""V1 (b): numerical attempts to break the three certificate paths near the decision threshold, against exact rational
references, on this machine.

Read-only; synthetic cells only (no data file is opened).  For every test cell (one Q1 cell on a 2x2x2-node grid, the
stored binary64 nodal values are the data; the exact reference is computed with fractions.Fraction from those stored
values and the stored spacing):
  truth   the sign of min J over the closed cell, by exact Bernstein subdivision (best-first, depth <= 60, <= 4000
          boxes): 'pos' (every leaf has positive exact coefficients), 'nonpos' (an exact corner value <= 0 found),
          or 'undet' (budget; such cells are counted, never used for a claim);
  (i)     the edge-hull margin of the kernel the runs used (compiled_enclosure._libm_margins via parallel_margins), the
          SoA kernel (soa_margins), and a pure-Python replica of the same operation sequence (CPython rounds every
          +,-,*,/ separately: no contraction; math.nextafter); checks: kernel == replica == SoA bit for bit; the margin
          <= L_exact, the exact interval evaluation of the same edge-hull expression (outward rounding must enclose it);
          margin > 0 => truth 'pos';
  (ii)    the prescreen bound of the kernel the runs used (certificate_prescreen.prescreen_lower_bounds), a pure-Python
          no-FMA replica of Appendix C's sequence (a)-(f), and an FMA-contracted variant (correctly rounded fma emulated
          with Fraction) to show the comparison can tell the two apart; checks: kernel == no-FMA replica bit for bit;
          |beta_fl - beta_exact| <= B for every coefficient; out <= min beta_exact; out > 0 => truth 'pos';
  (iii)   the compiled fallback arrays (compiled_fallback.fallback_arrays) against the NumPy interval code
          (feasibility._fallback_arrays: one ufunc per operation, so no contraction is possible) bit for bit; exact J at
          the 27 points and exact Bernstein coefficients inside the computed intervals; two levels of the NumPy
          split_bernstein enclose the exact children;
  cell    the per-cell decision of configuration C and of BL1 (tests (i) and (iii) only) by bayesreg's
          cert_eval.cell_pass (the decisions of feasibility._certify_map_parallel, cell by cell; the evaluation's
          cells.matches_map compares it with the map): certified => truth 'pos'; infeasible => truth 'nonpos'; the
          cell's lower bound <= exact J at 27 + 125 points.  The test cell is cell 0 of a 5x5x5-node grid (other nodes 0).
  map     certify_map itself (configuration C, full mode) on 5x5x5 / 6x6x6-node grids with zero faces and a random
          interior field scaled to the map's own threshold: certified => every cell's truth 'pos'; infeasible => some
          cell 'nonpos'.
Families: random cells; affine cancellation cells (det = 1 - a b with a b within a few ulps of 1); near-threshold
scalings t*w at the threshold of each path (t bisected to adjacent binary64 numbers); cells with positive corners and a
negative interior; cells whose J is subnormal (products underflow gradually); non-unit spacing (.7, 1.3, 1.1).
Usage: python -B v1_numerics.py [--quick]
"""
import heapq
import math
import sys
import time
from fractions import Fraction as Fr
from itertools import product

import numpy as np

from jd_curl.three_d.cell_q1 import VERTICES, CellQ1Operators3D, cell_node_indices
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d import feasibility as F
from jd_curl.three_d import interval as iv
from jd_curl.three_d import compiled_enclosure as ce
from jd_curl.three_d import compiled_fallback as cf
from jd_curl.three_d.soa_enclosure import soa_margins
from jd_curl.three_d.certificate_prescreen import prescreen_lower_bounds, G12, G6, G3, SQRT27_UP, ETA
from bayesreg.eval.cert_eval import cell_pass, PROVEN_FALLBACK

STATUS = {0: "certified_feasible", 1: "certified_feasible", 2: "certified_feasible", 3: "infeasible", 4: "uncertain",
          5: "uncertain"}    # cert_eval codes PROVEN_HULL, PROVEN_PRESCREEN, PROVEN_FALLBACK, INFEASIBLE, UNCERTAIN, UNCERTAIN_ARITHMETIC

INF = math.inf
CFG_C = F.PrescreenCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=65536)
CFG_BL1 = F.ParallelEnclosureCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=8192, fallback_workers=2)
HALF = Fr(1, 2)


# ---------------------------------------------------------------------------- exact reference (Fraction)
def exact_F(uc, h, xi):
    U = [[Fr(float(v)) for v in row] for row in uc]
    H = [Fr(float(x)) for x in h]
    X = [Fr(x) for x in xi]
    out = [[Fr(int(r == j)) for j in range(3)] for r in range(3)]
    for j in range(3):
        for a, bits in enumerate(VERTICES):
            if bits[j]:
                continue
            b = a | (1 << j)
            w = Fr(1)
            for k in range(3):
                if k != j:
                    w *= X[k] if bits[k] else 1-X[k]
            for r in range(3):
                out[r][j] += (U[b][r]-U[a][r])/H[j]*w
    return out


def det3(M):
    return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1]) + M[0][1]*(M[1][2]*M[2][0]-M[1][0]*M[2][2])
            + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))


def exact_J27(uc, h):
    """exact J at the closed27 points, storage index 9 iz + 3 iy + ix."""
    vals = [None]*27
    for iz, iy, ix in product(range(3), repeat=3):
        vals[9*iz+3*iy+ix] = det3(exact_F(uc, h, (Fr(ix, 2), Fr(iy, 2), Fr(iz, 2))))
    return vals


def to_bernstein(vals):
    b = list(vals)
    for stride in (1, 3, 9):
        for p in range(27):
            if (p//stride) % 3 == 1:
                b[p] = 2*b[p]-(b[p-stride]+b[p+stride])/2
    return b


def arr(b):
    return np.array(b, dtype=object).reshape(3, 3, 3)          # [iz, iy, ix]


def split(c, axis):
    """exact de Casteljau midpoint split along reference axis 0/1/2 = x/y/z (storage axis 2/1/0)."""
    s = 2-axis
    a, b, d = (np.take(c, j, axis=s) for j in range(3))
    ab, bd = (a+b)*HALF, (b+d)*HALF
    mid = (ab+bd)*HALF
    return np.stack([a, ab, mid], axis=s), np.stack([mid, bd, d], axis=s)


def exact_truth(uc, h, max_depth=60, max_boxes=4000):
    """('pos'|'nonpos'|'undet', exact lower envelope of the leaves or None)."""
    root = arr(to_bernstein(exact_J27(uc, h)))
    heap = [(float(min(root.flat)), 0, root, (Fr(0),)*3, (Fr(1),)*3, 0)]
    serial, boxes, leaves = 0, 0, []
    while heap:
        _, _, c, lo, hi, depth = heapq.heappop(heap)
        boxes += 1
        if boxes > max_boxes:
            return "undet", None
        corners = [c[z, y, x] for z in (0, 2) for y in (0, 2) for x in (0, 2)]
        if min(corners) <= 0:
            return "nonpos", None
        m = min(c.flat)
        if m > 0:
            leaves.append(m)
            continue
        if depth >= max_depth:
            return "undet", None
        axis = max(range(3), key=lambda k: (hi[k]-lo[k], -k))
        left, right = split(c, axis)
        mid = (lo[axis]+hi[axis])/2
        lh, rl = list(hi), list(lo)
        lh[axis], rl[axis] = mid, mid
        for child, a, b in ((left, lo, tuple(lh)), (right, tuple(rl), hi)):
            serial += 1
            heapq.heappush(heap, (float(min(child.flat)), serial, child, a, b, depth+1))
    return "pos", min(leaves)


def exact_samples(uc, h, n=5):
    pts = [Fr(i, n-1) for i in range(n)]
    return [det3(exact_F(uc, h, (x, y, z))) for z in pts for y in pts for x in pts]


def exact_edge_hull_lower(uc, h):
    """L_exact: the edge-hull interval determinant of Lemma A.2 evaluated in exact interval arithmetic."""
    U = [[Fr(float(v)) for v in row] for row in uc]
    H = [Fr(float(x)) for x in h]
    Fl, Fh = [[None]*3 for _ in range(3)], [[None]*3 for _ in range(3)]
    for j in range(3):
        for r in range(3):
            ds = [(U[a | (1 << j)][r]-U[a][r])/H[j] for a in range(8) if not a & (1 << j)]
            Fl[r][j], Fh[r][j] = min(ds)+int(r == j), max(ds)+int(r == j)

    def mul(a, b):
        ps = [a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]]
        return min(ps), max(ps)

    def sub(a, b):
        return a[0]-b[1], a[1]-b[0]

    def add(a, b):
        return a[0]+b[0], a[1]+b[1]
    e = lambda r, j: (Fl[r][j], Fh[r][j])  # noqa: E731
    A = sub(mul(e(1, 1), e(2, 2)), mul(e(1, 2), e(2, 1)))
    B = sub(mul(e(1, 2), e(2, 0)), mul(e(1, 0), e(2, 2)))
    C = sub(mul(e(1, 0), e(2, 1)), mul(e(1, 1), e(2, 0)))
    return add(add(mul(e(0, 0), A), mul(e(0, 1), B)), mul(e(0, 2), C))[0]


# ---------------------------------------------------------------------------- replicas (CPython binary64)
def edge_hull_replica(uc, h, delta=0.):
    nx = math.nextafter

    def out(lo, hi):
        lo, hi = nx(lo, -INF), nx(hi, INF)
        return lo, hi, math.isfinite(lo) and math.isfinite(hi) and lo <= hi

    ok = True
    Fv = {}
    for j in range(3):
        for r in range(3):
            lower, upper = INF, -INF
            for a in range(8):
                if a & (1 << j):
                    continue
                d = uc[a | (1 << j)][r]-uc[a][r]
                ok &= math.isfinite(d)
                lower, upper = min(lower, d), max(upper, d)
            lo, hi, g1 = out(lower, upper)
            lo, hi, g2 = out(lo/h[j], hi/h[j])
            ident = 1. if r == j else 0.
            lo, hi, g3 = out(lo+ident, hi+ident)
            ok &= g1 and g2 and g3
            Fv[r, j] = (lo, hi)

    def mul(a, b):
        x, y, z, w = a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]
        return out(min(min(min(x, y), z), w), max(max(max(x, y), z), w))

    def sub(a, b):
        return out(a[0]-b[1], a[1]-b[0])

    def add(a, b):
        return out(a[0]+b[0], a[1]+b[1])
    gs = []

    def k(t):
        gs.append(t[2])
        return t[:2]
    A = k(sub(k(mul(Fv[1, 1], Fv[2, 2])), k(mul(Fv[1, 2], Fv[2, 1]))))
    B = k(sub(k(mul(Fv[1, 2], Fv[2, 0])), k(mul(Fv[1, 0], Fv[2, 2]))))
    C = k(sub(k(mul(Fv[1, 0], Fv[2, 1])), k(mul(Fv[1, 1], Fv[2, 0]))))
    S = k(add(k(mul(Fv[0, 0], A)), k(mul(Fv[0, 1], B))))
    Jv = k(add(S, k(mul(Fv[0, 2], C))))
    m = k(out(Jv[0]-delta, Jv[1]-delta))
    return m[0], not (ok and all(gs))


def fma(a, b, c):
    return float(Fr(a)*Fr(b)+Fr(c))      # correctly rounded (CPython int true division)


def prescreen_replica(uc, D, contract=False):
    """Appendix C (a)-(f), the kernel's order; contract=True: an FMA-contracted variant (a*b+c -> fma)."""
    sigma = phi = big = 0.
    b = [0.]*27
    for p in range(27):
        Fm = [[0.]*3 for _ in range(3)]
        for r in range(3):
            for j in range(3):
                acc = mag = 0.
                for a in range(8):
                    if contract:
                        acc = fma(uc[a][r], D[j][p][a], acc)
                        mag += abs(uc[a][r]*D[j][p][a])
                    else:
                        term = uc[a][r]*D[j][p][a]
                        acc += term
                        mag += abs(term)
                Fm[r][j] = acc
                if mag > sigma:
                    sigma = mag
            Fm[r][r] += 1.
        for r in range(3):
            for j in range(3):
                if abs(Fm[r][j]) > phi:
                    phi = abs(Fm[r][j])
        f = Fm
        if contract:
            A = fma(f[1][1], f[2][2], -(f[1][2]*f[2][1]))
            Bc = fma(f[1][2], f[2][0], -(f[1][0]*f[2][2]))
            C = fma(f[1][0], f[2][1], -(f[1][1]*f[2][0]))
            v = fma(f[0][2], C, fma(f[0][1], Bc, f[0][0]*A))
        else:
            v = (f[0][0]*(f[1][1]*f[2][2]-f[1][2]*f[2][1]) + f[0][1]*(f[1][2]*f[2][0]-f[1][0]*f[2][2])
                 + f[0][2]*(f[1][0]*f[2][1]-f[1][1]*f[2][0]))
        b[p] = v
        if abs(v) > big:
            big = abs(v)
    for stride in (1, 3, 9):
        for p in range(27):
            if (p//stride) % 3 == 1:
                b[p] = 2.*b[p]-.5*(b[p-stride]+b[p+stride])
    e = G12*(sigma+phi)+ETA
    P = phi+e
    ej = SQRT27_UP*e*(3.*P*P+3.*P*e+e*e)+6.*G6*P*P*P
    bound = 1.01*(27.*ej+135.*G3*big)+ETA
    finite = all(math.isfinite(w) for w in b) and math.isfinite(bound) and math.isfinite(sigma)
    low = min(b)
    return (low-bound if finite else math.nan), b, bound


# ---------------------------------------------------------------------------- one cell through the real code
def cell_setup(uc, h):
    """The test cell is cell 0 of a 5x5x5-node grid (jd_curl's minimum size); the other nodes are 0.  Every quantity
    below is read for cell 0 only (the neighbours share nodes with it and are not test cells)."""
    g = make_grid_3d(nx=5, ny=5, nz=5, hx=h[0], hy=h[1], hz=h[2], unit="mm")
    u = np.zeros(g.shape+(3,))
    for a, (x, y, z) in enumerate(VERTICES):
        u[z, y, x] = uc[a]
    return g, u


def bits(x):
    return np.float64(x).view(np.uint64).item()


def run_cell(uc, h, *, want_truth=True):
    uc = [[float(v) for v in row] for row in uc]
    g, u = cell_setup(uc, h)
    rec = {}
    # (i)
    margin, flag = ce.parallel_margins(u, g, 0.)
    sm, sf = soa_margins(u, g, 0.)
    rm, rflag = edge_hull_replica(uc, g.spacing)
    rec["i_margin"], rec["i_flag"] = float(margin[0]), bool(flag[0])
    rec["i_bitwise_replica"] = (bits(margin[0]) == bits(rm)) and (bool(flag[0]) == rflag)
    rec["i_bitwise_soa"] = (bits(margin[0]) == bits(sm[0])) and (bool(flag[0]) == bool(sf[0]))
    L = exact_edge_hull_lower(uc, g.spacing)
    rec["i_encloses"] = bool(flag[0]) or Fr(float(margin[0])) <= L
    rec["i_L_exact_pos"] = L > 0
    # (ii)
    table = CellQ1Operators3D(g, location="closed27_uniform").coefficients
    out = float(prescreen_lower_bounds(u, g, table, np.array([0], dtype=np.int64), workers=1)[0])
    D = table.tolist()
    rout, rb, rbound = prescreen_replica(uc, D)
    fout, fb, _ = prescreen_replica(uc, D, contract=True)
    rec["ii_out"] = out
    rec["ii_bitwise_replica"] = (math.isnan(out) and math.isnan(rout)) or bits(out) == bits(rout)
    rec["ii_fma_variant_differs"] = not ((math.isnan(out) and math.isnan(fout)) or bits(out) == bits(fout)) or \
        any(bits(x) != bits(y) for x, y in zip(rb, fb))
    beta = to_bernstein(exact_J27(uc, g.spacing))
    if not math.isnan(out):
        rec["ii_error_within_B"] = all(abs(Fr(x)-y) <= Fr(rbound) for x, y in zip(rb, beta))
        rec["ii_out_le_min_beta"] = Fr(out) <= min(beta)
        rec["ii_max_err_over_B"] = float(max(abs(Fr(x)-y) for x, y in zip(rb, beta))/Fr(rbound)) if rbound > 0 else None
    rec["ii_min_beta_pos"] = min(beta) > 0
    # (iii)
    verts = u.reshape(-1, 3)[cell_node_indices(g, slice(0, 1))]
    comp = cf.fallback_arrays(verts, g.spacing, 0., np.geterr())
    nump = F._fallback_arrays(verts, g.spacing, 0., np.geterr())
    if comp is None or nump is None:
        rec["iii_bitwise_numpy"] = comp is None and nump is None
        rec["iii_arrays"] = None
    else:
        same = all(np.array_equal(np.asarray(x).view(np.uint64), np.asarray(y).view(np.uint64))
                   for x, y in ((comp[0].lo, nump[0].lo), (comp[0].hi, nump[0].hi), (comp[1].lo, nump[1].lo),
                                (comp[1].hi, nump[1].hi)))
        rec["iii_bitwise_numpy"] = bool(same)
        J27 = exact_J27(uc, g.spacing)
        plo, phi = comp[0].lo.reshape(-1), comp[0].hi.reshape(-1)
        clo, chi = comp[1].lo.reshape(-1), comp[1].hi.reshape(-1)
        rec["iii_J_enclosed"] = all(Fr(float(plo[p])) <= J27[p] <= Fr(float(phi[p])) for p in range(27))
        rec["iii_beta_enclosed"] = all(Fr(float(clo[p])) <= beta[p] <= Fr(float(chi[p])) for p in range(27))
        # two levels of the NumPy split (the subdivision of _classify) against exact children
        ok, stack = True, [(comp[1][0], arr(beta), 0)]
        while stack:
            ci, ce_, lev = stack.pop()
            if lev == 2:
                continue
            for axis in range(3):
                l1, r1 = iv.split_bernstein(ci, axis)
                l2, r2 = split(ce_, axis)
                for cint, cex in ((l1, l2), (r1, r2)):
                    lo_, hi_ = cint.lo.reshape(-1), cint.hi.reshape(-1)
                    ok &= all(Fr(float(lo_[k])) <= cex.flat[k] <= Fr(float(hi_[k])) for k in range(27))
                    stack.append((cint, cex, lev+1))
        rec["iii_split_enclosed"] = ok
    # per-cell decisions of _certify_map_parallel (configuration C and BL1), cell 0
    for name, cfg in (("C", CFG_C), ("BL1", CFG_BL1)):
        code, lower = cell_pass(u, g, cfg)
        rec[f"{name}_status"] = STATUS[int(code[0])]
        rec[f"{name}_code"] = int(code[0])
        rec[f"{name}_lb"] = None if not np.isfinite(lower[0]) else float(lower[0])
    samples = exact_samples(uc, g.spacing)+exact_J27(uc, g.spacing)
    smin = min(samples)
    for name in ("C", "BL1"):
        lb = rec[f"{name}_lb"]
        rec[f"{name}_lb_le_exact_samples"] = None if lb is None else Fr(lb) <= smin
    rec["exact_sample_min"] = float(smin)
    rec["truth"] = exact_truth(uc, g.spacing)[0] if want_truth else None
    return rec


VIOLATION_KEYS = ("i_bitwise_replica", "i_bitwise_soa", "i_encloses", "ii_bitwise_replica", "ii_error_within_B",
                  "ii_out_le_min_beta", "iii_bitwise_numpy", "iii_J_enclosed", "iii_beta_enclosed", "iii_split_enclosed",
                  "C_lb_le_exact_samples", "BL1_lb_le_exact_samples")


def violations(rec):
    v = [k for k in VIOLATION_KEYS if rec.get(k) is False]
    t = rec["truth"]
    if rec["i_margin"] > 0 and not rec["i_flag"] and not rec["i_L_exact_pos"]:
        v.append("i: margin > 0 but L_exact <= 0")
    if rec["i_margin"] > 0 and not rec["i_flag"] and t == "nonpos":
        v.append("i: margin > 0 but truth nonpos")
    if not math.isnan(rec["ii_out"]) and rec["ii_out"] > 0 and (t == "nonpos" or not rec["ii_min_beta_pos"]):
        v.append("ii: out > 0 but min beta_exact <= 0 or truth nonpos")
    for name in ("C", "BL1"):
        if rec[f"{name}_status"] == "certified_feasible" and t == "nonpos":
            v.append(f"{name}: certified but truth nonpos")
        if rec[f"{name}_status"] == "infeasible" and t == "pos":
            v.append(f"{name}: infeasible but truth pos")
    return v


# ---------------------------------------------------------------------------- families
def affine_cell(M, h):
    """nodal u = (M) applied to physical vertex coordinates, M = grad u (rounded when stored)."""
    uc = []
    for x, y, z in VERTICES:
        X = (x*h[0], y*h[1], z*h[2])
        uc.append([M[r][0]*X[0]+M[r][1]*X[1]+M[r][2]*X[2] for r in range(3)])
    return uc


def scaled(w, t):
    return [[t*v for v in row] for row in w]


def bisect_threshold(w, h, decide, lo=0., hi=1., iters=80):
    """t with decide(t*w) True at lo and False at hi; returns the two adjacent t's (or the closest pair found)."""
    if not decide(scaled(w, lo)) or decide(scaled(w, hi)):
        return None
    for _ in range(iters):
        mid = .5*(lo+hi)
        if mid in (lo, hi):
            break
        if decide(scaled(w, mid)):
            lo = mid
        else:
            hi = mid
    return lo, hi


def single_margin(w, h):
    g, u = cell_setup([[float(v) for v in r] for r in w], h)
    m, f = ce.parallel_margins(u, g, 0.)
    return float(m[0]) > 0 and not f[0]


def single_prescreen(w, h):
    g, u = cell_setup([[float(v) for v in r] for r in w], h)
    t = CellQ1Operators3D(g, location="closed27_uniform").coefficients
    o = float(prescreen_lower_bounds(u, g, t, np.array([0], dtype=np.int64), workers=1)[0])
    return o > 0


def single_cert(cfg):
    def decide(w, h):
        g, u = cell_setup([[float(v) for v in r] for r in w], h)
        return int(cell_pass(u, g, cfg)[0][0]) <= PROVEN_FALLBACK
    return decide


def families(quick):
    rng = np.random.default_rng(20261122)
    H1, HA = (1., 1., 1.), (.7, 1.3, 1.1)
    fam = {}
    n = 6 if quick else 25
    fam["random"] = [(rng.uniform(-s, s, (8, 3)).tolist(), h) for s in (.1, .3, .5) for h in (H1, HA) for _ in range(n)]
    canc = []
    for _ in range(6 if quick else 20):
        a = 1.+rng.uniform(-1e-3, 1e-3)
        b0 = 1./a
        for k in (-2, -1, 0, 1, 2):
            b = float(b0)
            for _ in range(abs(k)):
                b = math.nextafter(b, INF if k > 0 else -INF)
            M = [[0., a, 0.], [b, 0., 0.], [0., 0., 0.]]
            canc.append((affine_cell(M, H1), H1))
    fam["cancellation (det = 1 - ab, ab within 2 ulps of 1)"] = canc
    sub = []
    for kexp in (500, 520, 530, 537, 540, 560, 600):
        c = 2.**-kexp
        for sgn in (1., -1.):
            M = [[-1., c, 0.], [0., -1., sgn*c], [1., 0., -1.]]   # F = [[0,c,0],[0,0,+-c],[1,0,0]]: det = +-c^2 (subnormal for k >= 512)
            sub.append((affine_cell(M, H1), H1))
            sub.append(([[v*(1+2.**-50) for v in row] for row in affine_cell(M, H1)], H1))
    fam["subnormal-scale F entries, stored values rounded (J reaches 0 on the cell)"] = sub
    subpos = []
    for kexp in (500, 520, 530, 537, 540, 560, 600, 1000):
        c = 2.**-kexp
        for c2 in (c, -c, 2.**-30):
            uc = []
            for x, y, z in VERTICES:                       # F = [[y(2^-52 - c), F01, 0], [0, z(2^-52 - c2), F12], [1, 0, 0]]
                uc.append([-x+y*(c if x == 0 else 2.**-52), -y+z*(c2 if y == 0 else 2.**-52), float(x-z)])
            subpos.append((uc, H1))                          # J = F01 F12 F20 = (c(1-x) + 2^-52 x)(c2(1-y) + 2^-52 y): min c*c2 at x = y = 0
    fam["subnormal minimum J (J = F01 F12, min c*c2 at a corner; c = 2^-k, k up to 1000)"] = subpos
    # near-threshold scalings at each path's own threshold
    near = []
    deciders = (("edge-hull (i)", single_margin), ("prescreen (ii)", single_prescreen),
                ("configuration C map", single_cert(CFG_C)), ("BL1 map (i)+(iii)", single_cert(CFG_BL1)))
    m = 3 if quick else 10
    tries = 0
    while len(near) < 4*m*2 and tries < 400:
        tries += 1
        h = H1 if tries % 2 else HA
        w = rng.uniform(-1, 1, (8, 3)).tolist()
        for label, dec in deciders:
            pair = bisect_threshold(w, h, lambda x, d=dec, hh=h: d(x, hh))
            if pair is None:
                continue
            lo, hi = pair
            near.append((scaled(w, lo), h, label+" just certified"))
            near.append((scaled(w, hi), h, label+" just not certified"))
            if sum(1 for x in near if x[2].startswith(label)) >= 2*m:
                pass
    fam["near-threshold scalings (bisected to adjacent t)"] = [(x[0], x[1]) for x in near]
    # positive corners, negative interior (search)
    inner = []
    tries = 0
    while len(inner) < (3 if quick else 10) and tries < 4000:
        tries += 1
        w = rng.uniform(-.9, .9, (8, 3)).tolist()
        J27 = exact_J27(w, H1)
        corners = [J27[9*z+3*y+x] for z in (0, 2) for y in (0, 2) for x in (0, 2)]
        if min(corners) <= 0:
            continue
        b = to_bernstein(J27)
        if min(b) > 0:
            continue
        t, _ = exact_truth(w, H1, max_depth=30, max_boxes=1500)
        if t == "nonpos":
            inner.append((w, H1))
    fam["positive corners, nonpositive interior"] = inner
    return fam


def cell_vertices(u, g, cell):
    ids = cell_node_indices(g, slice(cell, cell+1))[0]
    flat = u.reshape(-1, 3)
    return [flat[i].tolist() for i in ids]


def map_family(quick):
    """certify_map (configuration C, full mode) at its own threshold on zero-face maps; every cell judged exactly."""
    rng = np.random.default_rng(1122)
    n_maps = 2 if quick else 6
    bad = 0
    rows = []
    for k in range(n_maps):
        n = 5 if k % 2 == 0 else 6
        h = (1., 1., 1.) if k % 3 else (.7, 1.3, 1.1)
        g = make_grid_3d(nx=n, ny=n, nz=n, hx=h[0], hy=h[1], hz=h[2], unit="mm")
        w = np.zeros(g.shape+(3,))
        w[1:-1, 1:-1, 1:-1] = .5*rng.uniform(-1, 1, w[1:-1, 1:-1, 1:-1].shape)*np.array([h[0], h[1], h[2]])   # |u| < h/2: every node stays inside
        def decide(t):
            return F.certify_map(t*w, g, config=CFG_C).status == "certified_feasible"
        lo, hi = 0., 1.
        if not decide(lo) or decide(hi):
            rows.append((k, "no threshold in [0, 1]"))
            continue
        for _ in range(80):
            mid = .5*(lo+hi)
            if mid in (lo, hi):
                break
            lo, hi = (mid, hi) if decide(mid) else (lo, mid)
        for t, side in ((lo, "just certified"), (hi, "just not certified")):
            u = t*w
            r = F.certify_map(u, g, config=CFG_C)
            truths = [exact_truth(cell_vertices(u, g, c), g.spacing)[0] for c in range(g.n_cells)]
            n_non = truths.count("nonpos")
            n_und = truths.count("undet")
            ok = not ((r.status == "certified_feasible" and n_non) or
                      (r.status == "infeasible" and r.reason == "JACOBIAN_VIOLATION" and n_non == 0 and n_und == 0))
            lb_ok = True
            if r.min_jd_lower_bound is not None:
                smin = min(min(exact_J27(cell_vertices(u, g, c), g.spacing)) for c in range(g.n_cells))
                lb_ok = Fr(r.min_jd_lower_bound) <= smin
            bad += (not ok) + (not lb_ok)
            rows.append((k, side, f"t={t!r}", r.status, r.reason, f"lb={r.min_jd_lower_bound}",
                         f"paths fast/pre/fb={r.n_fast_cells}/{r.n_prescreen_cells}/{r.n_fallback_cells}",
                         f"cells truth nonpos/undet/total={n_non}/{n_und}/{g.n_cells}", "ok" if ok and lb_ok else "VIOLATION"))
    print(f"\n== family: certify_map at its own threshold (configuration C, full mode, zero faces): {n_maps} maps")
    for row in rows:
        print("   ", *row)
    print(f"   violations: {bad}")
    return bad


def main():
    quick = "--quick" in sys.argv
    t0 = time.time()
    print(f"# arithmetic_supported: numpy {iv.arithmetic_supported()}, numba {ce.arithmetic_supported(np.float64(0.))}; "
          f"parallel_margins execution: {ce.margins_execution()}")
    # the prescreen's table hypothesis (Appendix C): D[j,p,a] = fl(1/h_j) c[j,p,a], c zero or +-(power of two), entries normal
    for h in ((1., 1., 1.), (.7, 1.3, 1.1)):
        g = make_grid_3d(nx=5, ny=5, nz=5, hx=h[0], hy=h[1], hz=h[2], unit="mm")
        D = CellQ1Operators3D(g, location="closed27_uniform").coefficients
        ok = True
        for j in range(3):
            inv = 1./h[j]
            for p_ in range(27):
                for a_ in range(8):
                    x, y, z = (k/2 for k in (p_ % 3, (p_//3) % 3, p_//9))
                    bits = VERTICES[a_]
                    f = [(x, y, z)[k] if bits[k] else 1-(x, y, z)[k] for k in range(3)]
                    c = (1 if bits[j] else -1)*Fr(f[(j+1) % 3])*Fr(f[(j+2) % 3])
                    ok &= Fr(float(D[j, p_, a_])) == Fr(inv)*c
        nz = np.abs(D[D != 0])
        print(f"# prescreen table for spacing {h}: every entry == fl(1/h_j) * c exactly: {ok}; nonzero entries normal: "
              f"{bool(nz.min() >= np.finfo(np.float64).tiny)}; distinct |entries| {sorted(set(nz.tolist()))[:6]}")
    fam = families(quick)
    grand = dict(cells=0, violations=0)
    for name, cells in fam.items():
        stats = {}
        bad = []
        fma_diff = 0
        truth = {"pos": 0, "nonpos": 0, "undet": 0}
        decisions = {}
        min_i_pos_margin = min_ii_pos = None
        maxratio = 0.
        for uc, h in cells:
            rec = run_cell(uc, h)
            truth[rec["truth"]] += 1
            for k in VIOLATION_KEYS:
                if rec.get(k) is not None:
                    stats.setdefault(k, [0, 0])
                    stats[k][0] += 1
                    stats[k][1] += bool(rec[k])
            fma_diff += rec["ii_fma_variant_differs"]
            for nm in ("C", "BL1"):
                decisions[(nm, rec[f"{nm}_status"], rec["truth"])] = decisions.get((nm, rec[f"{nm}_status"], rec["truth"]), 0)+1
            if rec["i_margin"] > 0 and not rec["i_flag"]:
                min_i_pos_margin = rec["i_margin"] if min_i_pos_margin is None else min(min_i_pos_margin, rec["i_margin"])
            if not math.isnan(rec["ii_out"]) and rec["ii_out"] > 0:
                min_ii_pos = rec["ii_out"] if min_ii_pos is None else min(min_ii_pos, rec["ii_out"])
            if rec.get("ii_max_err_over_B") is not None:
                maxratio = max(maxratio, rec["ii_max_err_over_B"])
            v = violations(rec)
            if v:
                bad.append((v, uc, h))
        grand["cells"] += len(cells)
        grand["violations"] += len(bad)
        print(f"\n== family: {name}: {len(cells)} cells; exact truth {truth}")
        for k, (n_, ok) in stats.items():
            print(f"   {k}: {ok}/{n_} true")
        print(f"   prescreen: FMA-contracted variant differs from the kernel in bits on {fma_diff}/{len(cells)} cells; "
              f"max |beta_fl - beta_exact| / B = {maxratio:.3e}")
        print(f"   smallest positive edge-hull margin {min_i_pos_margin}; smallest positive prescreen bound {min_ii_pos}")
        for (nm, st, tr), c in sorted(decisions.items()):
            print(f"   decision {nm}: {st:>20} with truth {tr:>6}: {c}")
        print(f"   violations: {len(bad)}")
        for v, uc, h in bad[:5]:
            print("     ", v, "spacing", h, "u", [[x.hex() for x in row] for row in uc])
    grand["violations"] += map_family(quick)
    # NumPy does not fuse separate ufunc calls: a*b then +c against no-FMA and FMA references
    rng = np.random.default_rng(5)
    a = 1.+rng.uniform(-1e-3, 1e-3, 2000)
    b = 1./a
    c = -np.ones(2000)
    npres = np.add(np.multiply(a, b), c)
    nofma = np.array([x*y+z for x, y, z in zip(a.tolist(), b.tolist(), c.tolist())])
    withfma = np.array([fma(x, y, z) for x, y, z in zip(a.tolist(), b.tolist(), c.tolist())])
    print(f"\n== NumPy ufunc chain multiply then add: equals CPython no-FMA {np.array_equal(npres.view(np.uint64), nofma.view(np.uint64))}; "
          f"cells where an FMA would differ {int(np.count_nonzero(nofma.view(np.uint64) != withfma.view(np.uint64)))}/2000")
    print(f"\n# TOTAL: {grand['cells']} cells, cells with any violation: {grand['violations']}; {time.time()-t0:.0f} s")
    return 0 if grand["violations"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
