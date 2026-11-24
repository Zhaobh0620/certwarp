"""C14 gate (revised): Cov(delta) = H^-1 by whitening, 6 x 6 x 6 nodes (d = 3 N_int = 192), S = 1e5,
for both diffusion-block factors; B6 on the same samples.

Pipeline under test, per sample: b = sum_i B_i eps_i (channel, log-volume and curl adjoints K_i^T; diffusion
either the DST square root sqrt(alpha/omega0) K_Q^1/2 of `stiffness_sqrt_apply` or the explicit K_D^T),
then delta = `laplace.pcg` with the contract settings (x0 = 0, rtol = 1e-3, maxiter = 200, preconditioner
the C09.1 metric of `LaplaceSystem`).  For speed the S = 1e5 right-hand sides are formed in batches with
the dense K_i of the independent oracle (tests/test_bayesreg/model/_helpers.dense_model) and the PCG runs
on column blocks with the dense H and M^-1; test_laplace.py shows the streamed `perturbation_rhs` equals
these dense products to 1e-12 (1e-10 for the DST factor against eigh) and the streamed H and M^-1 equal
the dense ones.  The channel, log-volume and curl noise is shared by the two variants (each variant's
statistics are valid on their own).  Noise: SFC64 (test-only speed; the module uses PCG64 children).

Gate (01_合同 C14): with H = L L^T and z_s = L^T delta_s, C = (1/S) sum z z^T:
  (i)   |v^T C v - 1| <= 5 sqrt(2/S) for the d axes and 20 fixed-seed random unit vectors;
  (ii)  ||C - I||_F <= 1.2 sqrt(d (d + 1) / S);
  (iii) whitening with 1.1 H instead must fail the diagonal check of (i).
B6: mean of delta^T A delta (exact solves of the same b) within 5 standard errors sqrt(2 tr((A H^-1)^2)/S)
of tr(A H^-1), A = K_Q/omega0 and C^T W C/omega0; the PCG (rtol 1e-3) bias of that mean is recorded.

(iii) only rescales the whole covariance (C_wrong = 1.1 C), so it fails by construction.  The power of the
gate against STRUCTURAL sampler errors is checked separately with exact covariances (review M4 item 3,
test_gate_power_against_structural_sampler_errors, no sampling): a sampler that omits block i or doubles its
variance has Cov(delta) = H^-1 (H -/+ K_i^T K_i) H^-1; both (i) and (ii) must flag it with a wide margin for
every block.  A block weight off by sqrt(1.1) is at the resolution limit (deviation .02-.09 against the
tolerance .022) and is recorded only.

Coverage chain: the S = 1e5 gate (marked slow) forms b with the oracle's dense K_i; test_laplace.py ties
`perturbation_rhs` and the streamed H, M^-1 to the same oracle (1e-12; 1e-10 for the DST factor).
"""
import numpy as np
import pytest
import scipy.linalg

from bayesreg.infer import laplace as lp

from . import _cases_gauss27 as cs          # DEF: the C05-2 model ("gauss27", the default before DEF), explicit

S = 100_000
CHUNK = 1000
SEED = 20261030


def _kron_factor(K_D, n_q, m):
    """K_D rows (q, r, j), columns (r', i) = delta_{r r'} G[(q, j), i]: return G (3|Q|, m) after checking it."""
    blocks = K_D.reshape(n_q, 3, 3, 3, m)
    G = blocks[:, 0, :, 0, :]
    for r in range(3):
        for s in range(3):
            assert np.array_equal(blocks[:, r, :, s, :], G if r == s else np.zeros_like(G))
    return G.reshape(3*n_q, m)


def _directions(d):
    directions = []
    for seed in range(20):
        v = np.random.default_rng(1000+seed).normal(size=d)
        directions.append(v/np.linalg.norm(v))
    return np.stack(directions, axis=1)


def test_gate_power_against_structural_sampler_errors(record_property):
    """Exact covariances of wrong samplers (one block omitted or doubled) fail (i) and (ii) by a wide margin."""
    grid, state = cs.cube6()
    u = cs.u_hat_of(grid)
    K, H = cs.dense_blocks(state, u)
    d = H.shape[0]
    Linv = scipy.linalg.solve_triangular(np.linalg.cholesky(H), np.eye(d), lower=True)
    V = np.concatenate([np.eye(d), _directions(d)], axis=1)
    tol_i, tol_ii = 5*np.sqrt(2/S), 1.2*np.sqrt(d*(d+1)/S)
    for name, Kb in K.items():
        G = Kb.T@Kb
        for label, factor in (("omitted", -1.), ("doubled", 1.), ("weight_sqrt1.1", .1)):
            C = Linv@(H+factor*G)@Linv.T              # whitened exact covariance L^T Cov(delta) L
            dev = float(np.max(np.abs(np.einsum("ik,ij,jk->k", V, C, V)-1)))
            frob = float(np.linalg.norm(C-np.eye(d)))
            record_property(f"{name}_{label}_projection_dev", dev)
            record_property(f"{name}_{label}_frobenius", frob)
            if label != "weight_sqrt1.1":
                assert dev > 4*tol_i and frob > 2*tol_ii, (name, label, dev, frob)


@pytest.mark.slow                   # S = 1e5 samples, about 55 s
def test_whitening_gate_and_b6(record_property):
    grid, state = cs.cube6()
    u = cs.u_hat_of(grid)
    system = lp.LaplaceSystem(state, u)
    K, H = cs.dense_blocks(state, u)
    d = 3*grid.n_free_scalar
    m = grid.n_free_scalar
    n_q = grid.n_cells*27
    assert d == 192
    H_sys = cs.dense_operator(system.hvp, d)
    assert np.max(np.abs(H_sys-H)) <= 1e-12*np.max(np.abs(H))
    Minv = cs.dense_operator(system.precondition, d)
    B_D = np.sqrt(state.alpha/state.omega0)*cs.dense_operator(lambda e: lp.stiffness_sqrt_apply(e, grid), d)
    G_T = _kron_factor(K["diffusion"], n_q, m).T
    R_common = np.concatenate([K["channel"].T, K["logvol"].T, K["curl"].T], axis=1)
    L = np.linalg.cholesky(H)
    chol = scipy.linalg.cho_factor(H)
    A = dict(KQ=cs.dense_operator(lambda v: lp.stiffness_apply_scaled(state, v), d),
             CtWC=K["curl"].T@K["curl"]/state.nu)
    assert np.max(np.abs(A["CtWC"]-cs.dense_operator(lambda v: lp.curl_gram_apply(state, v), d))) \
        <= 1e-12*np.max(np.abs(A["CtWC"]))

    rng = np.random.Generator(np.random.SFC64(SEED))
    variants = ("dst_sqrt", "explicit")
    second = {v: np.zeros((d, d)) for v in variants}
    quad = {(v, a, kind): 0. for v in variants for a in A for kind in ("pcg", "exact")}
    max_iter, max_res = {v: 0 for v in variants}, {v: 0. for v in variants}
    eps_common = np.empty((R_common.shape[1], CHUNK))       # noise buffers reused (no per-chunk allocation)
    eps_dst = np.empty((d, CHUNK))
    eps_exp = np.empty((3*n_q, CHUNK))
    explicit = np.empty((d, CHUNK))
    for _ in range(S//CHUNK):
        common = R_common@rng.standard_normal(out=eps_common)
        for r in range(3):
            np.matmul(G_T, rng.standard_normal(out=eps_exp), out=explicit[r*m:(r+1)*m])
        rhs = dict(dst_sqrt=common+B_D@rng.standard_normal(out=eps_dst), explicit=common+explicit)
        for v in variants:
            res = lp.pcg(lambda X: H@X, lambda X: Minv@X, rhs[v], rtol=1e-3, maxiter=200)
            assert np.all(res.converged)
            max_iter[v] = max(max_iter[v], int(res.iterations.max()))
            max_res[v] = max(max_res[v], float(res.rel_residual.max()))
            second[v] += res.x@res.x.T
            exact = scipy.linalg.cho_solve(chol, rhs[v])
            for a, Amat in A.items():
                quad[(v, a, "pcg")] += float(np.sum(res.x*(Amat@res.x)))
                quad[(v, a, "exact")] += float(np.sum(exact*(Amat@exact)))

    tol_i = 5*np.sqrt(2/S)
    tol_ii = 1.2*np.sqrt(d*(d+1)/S)
    directions = _directions(d)
    L_wrong = np.linalg.cholesky(1.1*H)
    Hinv = scipy.linalg.cho_solve(chol, np.eye(d))
    for v in variants:
        cov_delta = second[v]/S
        C = L.T@cov_delta@L
        diag_dev = float(np.max(np.abs(np.diag(C)-1)))
        proj_dev = float(np.max(np.abs(np.einsum("ik,ij,jk->k", directions, C, directions)-1)))
        frob = float(np.linalg.norm(C-np.eye(d)))
        C_wrong = L_wrong.T@cov_delta@L_wrong
        wrong_dev = np.abs(np.diag(C_wrong)-1)
        record_property(f"{v}_axes_max_dev", diag_dev)
        record_property(f"{v}_random_max_dev", proj_dev)
        record_property(f"{v}_frobenius", frob)
        record_property(f"{v}_frobenius_expected", float(np.sqrt(d*(d+1)/S)))
        record_property(f"{v}_1.1H_min_diag_dev", float(wrong_dev.min()))
        record_property(f"{v}_pcg_max_iterations", max_iter[v])
        record_property(f"{v}_pcg_max_rel_residual", max_res[v])
        assert diag_dev <= tol_i and proj_dev <= tol_i            # (i)
        assert frob <= tol_ii                                     # (ii)
        assert np.max(wrong_dev) > tol_i and np.min(wrong_dev) > tol_i     # (iii): every axis fails
        for a, Amat in A.items():
            AH = Amat@Hinv
            trace = float(np.trace(AH))
            se = float(np.sqrt(2*np.trace(AH@AH)/S))
            exact_mean = quad[(v, a, "exact")]/S
            pcg_mean = quad[(v, a, "pcg")]/S
            record_property(f"{v}_{a}_trace_rel_diff", (exact_mean-trace)/trace)
            record_property(f"{v}_{a}_trace_rel_se", se/trace)
            record_property(f"{v}_{a}_pcg_bias_rel", (pcg_mean-exact_mean)/trace)
            assert abs(exact_mean-trace) <= 5*se                  # B6 (exact solves)
            assert abs(pcg_mean-exact_mean) <= 1e-2*trace         # PCG truncation bias (B5-3), recorded
