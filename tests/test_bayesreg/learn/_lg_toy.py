"""A linear Gaussian (LG) population for the C15-9 tests (theory B17 section 0; contract C15-9 gate (i)).  No data/.

Everything here is dense and built from the independent oracle of tests/test_bayesreg/model/_helpers
(`dense_operators`, `selection`: numpy's Gauss-Legendre rule and the scalar Q1 formulas; nothing calls bayesreg's
operators).  Zero-boundary Q1 on a 5 x 5 x 6 node grid, h = (.9, 1.1, 1.2) mm, omega0 = omega (3 N_int = 108,
|Q| = 2160):
    A_alpha = K_Q/omega0,  A_nu = C^T W C/omega0,  Pi_J(tau) = D_div^T diag(c_q/tau_q^2) D_div,  c_q = omega w_g/omega0,
    Pi_0(eta) = alpha A_alpha + nu A_nu + Pi_J(tau),  tau_q^2 = sum_k p_qk tau_k^2,
with a partial-volume tissue layer (K_T = 2 plus a floored background: P^0 = .05, P^1 = .95 s, P^2 = .95 (1 - s),
s a logistic step across x, Q1-interpolated to the Gauss points).  Pairs p = 1..n_T: y_p = J_p u_p + e_p,
u_p ~ N(0, Pi_0(eta*)^-1), e_p ~ N(0, I), J_p = 12 N(0, 1)/sqrt(n) of size (3n/4) x n.  H_p = J_p^T J_p + Pi_0.

Parameterization P2 with tau_0 = 1 fixed and the largest, alpha = alpha_nug = 1e-2 (constant while the learned
tau_k < 1); learned z = (log nu, log tau_1, log tau_2) (tau as standard deviations, the C15-6 components).

  `exact_statistics(eta)`   PairStatistics with exact traces (Sigma_0 = Pi_0^-1, Sigma_p = H_p^-1) and exact .gm
  `mc_statistics(eta, rng)` the same with S = S0 = 16 Gaussian samples per pair (Cholesky), as C14/C15-9 estimate them
  `log_evidence(z)`, `marginal_gradient(z)`  l = sum_p log N(y_p; 0, J_p Pi_0^-1 J_p^T + I) and its gradient in z from
                            the MARGINAL form (d l/d eta_j = 1/2 [tr(M dPi) - v^T dPi v], v = Pi^-1 J^T C^-1 y,
                            M = Pi^-1 J^T C^-1 J Pi^-1) -- derived independently of B17's trace form (as the R1
                            reviewer did)
  `innovation_cov(eta)`     predicted covariance of one MC step in z (delta method; quadratic forms of Gaussian samples:
                            Cov(mean d^T A d, mean d^T B d) = 2 tr(A S B S)/S, the ddof-1 variance side 2 tr(.)/(S-1))
"""
from functools import lru_cache

import numpy as np
import scipy.linalg

from jd_curl.three_d.grid import make_grid_3d

from bayesreg.learn import EBConfig, GMStatistics, PairStatistics, alpha_nugget, eb_update
from bayesreg.solve import Eta

from ..model._helpers import dense_operators, selection

TAU0 = 1.
NU_TRUE = 4.
TAU_TRUE = (.30, .45)
N_PAIRS = 8
S = 16
CONFIG = EBConfig(eb_count="gull_mackay")


def eta_of(z):
    """z = (log nu, log tau_1, log tau_2) -> Eta under P2 (tau_0 = 1 fixed, alpha = alpha_nug(tau))."""
    nu, t1, t2 = np.exp(np.asarray(z, dtype=np.float64))
    tau2 = (TAU0**2, float(t1*t1), float(t2*t2))
    return Eta(alpha_nugget(tau2), float(nu), tau2)


def z_of(eta):
    return np.log([eta.nu, np.sqrt(eta.tau2[1]), np.sqrt(eta.tau2[2])])


def _quad_trace(A, S1, B, S2):
    """tr(A S1 B S2)."""
    return float(np.sum((A@S1)*(B@S2).T))


class Toy:
    """The LG population (see the module docstring); build once with `toy()`."""

    def __init__(self, seed=1729):
        self.seed = seed
        grid = make_grid_3d(nx=5, ny=5, nz=6, hx=.9, hy=1.1, hz=1.2, x0=0., y0=0., z0=0., unit="mm")
        D, B, wq = dense_operators(grid)
        Sel = selection(grid)
        DS = [D[j]@Sel for j in range(3)]                     # (|Q|, N_int): d_j at the Gauss points
        c = wq/grid.omega                                     # omega0 = omega
        kq = sum(DS[j].T@(c[:, None]*DS[j]) for j in range(3))
        A_alpha = np.kron(np.eye(3), kq)                      # K_Q/omega0 (component-major free vector)
        Z = np.zeros_like(DS[0])
        curl = np.concatenate([np.concatenate((Z, -DS[2], DS[1]), axis=1), np.concatenate((DS[2], Z, -DS[0]), axis=1),
                               np.concatenate((-DS[1], DS[0], Z), axis=1)])
        A_nu = curl.T@(np.tile(c, 3)[:, None]*curl)
        x = grid.identity()[..., 0]
        s = 1/(1+np.exp((x-x.mean())/.8))
        nodes = np.stack([np.full(grid.shape, .05), .95*s, .95*(1-s)])
        self.grid, self.n, self.c = grid, 3*Sel.shape[1], c
        self.A_alpha, self.A_nu = .5*(A_alpha+A_alpha.T), .5*(A_nu+A_nu.T)
        self.Ddiv = np.concatenate(DS, axis=1)               # (|Q|, 3 N_int)
        self.P = np.stack([B@nodes[k].ravel() for k in range(3)], axis=1)     # (|Q|, 3) p_qk
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(self.prior(eta_of(np.log([NU_TRUE, *TAU_TRUE]))))
        self.pairs = []
        for _ in range(N_PAIRS):
            J = 12.*rng.standard_normal((3*self.n//4, self.n))/np.sqrt(self.n)
            u = scipy.linalg.solve_triangular(L.T, rng.standard_normal(self.n), lower=False)
            self.pairs.append((J, J.T@J, J@u+rng.standard_normal(J.shape[0])))

    def tau_q2(self, eta):
        return self.P@np.asarray(eta.tau2)

    def prior(self, eta):
        Pi = eta.alpha*self.A_alpha+eta.nu*self.A_nu+self.Ddiv.T@((self.c/self.tau_q2(eta))[:, None]*self.Ddiv)
        return .5*(Pi+Pi.T)

    def rho(self, eta):
        """(|Q|, 3) c_q rho_qk = c_q p_qk (tau_k^2/tau_q^2)^2."""
        ratio = np.asarray(eta.tau2)[None, :]/self.tau_q2(eta)[:, None]
        return self.c[:, None]*self.P*ratio**2

    def btilde(self, eta):
        """B~_k = D_div^T diag(c rho_k) D_div, k = 0..2."""
        w = self.rho(eta)
        return [self.Ddiv.T@(w[:, k][:, None]*self.Ddiv) for k in range(3)]

    # ---- per-pair posteriors ------------------------------------------------------------------------------------
    def _posterior(self, Pi, G, Jy):
        cf = scipy.linalg.cho_factor(G+Pi)
        return cf, scipy.linalg.cho_solve(cf, Jy)

    def exact_statistics(self, eta):
        Pi = self.prior(eta)
        Sig0 = scipy.linalg.cho_solve(scipy.linalg.cho_factor(Pi), np.eye(self.n))
        diag0 = np.sum((self.Ddiv@Sig0)*self.Ddiv, axis=1)
        w_rho, w_p = self.rho(eta), self.c[:, None]*self.P
        g_alpha = eta.alpha*float(np.sum(Sig0*self.A_alpha))
        g_nu = eta.nu*float(np.sum(Sig0*self.A_nu))
        m_prior = tuple(float(w_rho[:, k]@diag0) for k in range(3))
        out = []
        for p, (J, G, y) in enumerate(self.pairs):
            cf, u = self._posterior(Pi, G, J.T@y)
            Sig = scipy.linalg.cho_solve(cf, np.eye(self.n))
            du2 = (self.Ddiv@u)**2
            diag = np.sum((self.Ddiv@Sig)*self.Ddiv, axis=1)
            gm = GMStatistics(n_prior_samples=S, n_prior_converged=S, g_alpha=g_alpha, g_alpha_stderr=0.,
                              g_nu=g_nu, g_nu_stderr=0., m_prior=m_prior, m_prior_stderr=(0.,)*3,
                              m_hat=tuple(float(w_rho[:, k]@du2) for k in range(3)),
                              m_var=tuple(float(w_rho[:, k]@diag) for k in range(3)), m_var_stderr_bound=(0.,)*3,
                              alpha=eta.alpha, nu=eta.nu, tau2=tuple(eta.tau2), count_check=float("nan"),
                              identity_max_rel=0., seed=0)
            out.append(self._pair(p, eta, u, float(np.sum(Sig*self.A_alpha)), 0., float(np.sum(Sig*self.A_nu)), 0.,
                                  tuple(float(w_p[:, k]@du2) for k in range(3)),
                                  tuple(float(w_p[:, k]@diag) for k in range(3)), gm))
        return out

    def _pair(self, p, eta, u, kq_trace, kq_se, curl_trace, curl_se, tau_hat, tau_var, gm):
        n = self.n
        return PairStatistics(pair_id=f"lg{p}", n_free=n, n_curl=n, n_q=1., kq_quadratic=float(u@self.A_alpha@u),
                              curl_quadratic=float(u@self.A_nu@u), kq_trace=kq_trace, kq_trace_stderr=kq_se,
                              curl_trace=curl_trace, curl_trace_stderr=curl_se, tau_hat=tau_hat, tau_var=tau_var,
                              tau_den=tuple(float(x) for x in np.sum(self.c[:, None]*self.P, axis=0)),
                              sigma_used=(1.,), sigma_post=(1.,), n_samples=S, n_converged=S, n_passed=S,
                              pass_rate=1., pass_rate_low=False, gm=gm)

    def mc_statistics(self, eta, rng, S0=S, S_post=S):
        """S0 prior samples N(0, Pi_0^-1) and S_post posterior samples N(0, H_p^-1) per pair (fresh for every pair)."""
        Pi = self.prior(eta)
        L0 = np.linalg.cholesky(Pi)
        w_rho, w_p = self.rho(eta), self.c[:, None]*self.P
        out = []

        def mean_se(values):
            return float(np.mean(values)), float(np.std(values, ddof=1)/np.sqrt(len(values)))
        for p, (J, G, y) in enumerate(self.pairs):
            d0 = scipy.linalg.solve_triangular(L0.T, rng.standard_normal((self.n, S0)), lower=False)
            qa0 = np.einsum("is,ij,js->s", d0, self.A_alpha, d0)
            qn0 = np.einsum("is,ij,js->s", d0, self.A_nu, d0)
            div0 = (self.Ddiv@d0)**2                                          # (|Q|, S0)
            m0 = w_rho.T@div0                                                 # (3, S0)
            cf, u = self._posterior(Pi, G, J.T@y)
            LH = np.linalg.cholesky(G+Pi)
            d = scipy.linalg.solve_triangular(LH.T, rng.standard_normal((self.n, S_post)), lower=False)
            qa = np.einsum("is,ij,js->s", d, self.A_alpha, d)
            qn = np.einsum("is,ij,js->s", d, self.A_nu, d)
            var = np.var(self.Ddiv@(u[:, None]+d), axis=1, ddof=1)
            du2 = (self.Ddiv@u)**2
            m_var = tuple(float(w_rho[:, k]@var) for k in range(3))
            ga, ga_se = mean_se(eta.alpha*qa0)
            gn, gn_se = mean_se(eta.nu*qn0)
            mp = [mean_se(m0[k]) for k in range(3)]
            gm = GMStatistics(n_prior_samples=S0, n_prior_converged=S0, g_alpha=ga, g_alpha_stderr=ga_se, g_nu=gn,
                              g_nu_stderr=gn_se, m_prior=tuple(x[0] for x in mp),
                              m_prior_stderr=tuple(x[1] for x in mp),
                              m_hat=tuple(float(w_rho[:, k]@du2) for k in range(3)), m_var=m_var,
                              m_var_stderr_bound=tuple(v*np.sqrt(2/(S_post-1)) for v in m_var), alpha=eta.alpha,
                              nu=eta.nu, tau2=tuple(eta.tau2), count_check=float("nan"), identity_max_rel=0., seed=0)
            ka, ka_se = mean_se(qa)
            kn, kn_se = mean_se(qn)
            out.append(self._pair(p, eta, u, ka, ka_se, kn, kn_se, tuple(float(w_p[:, k]@du2) for k in range(3)),
                                  tuple(float(w_p[:, k]@var) for k in range(3)), gm))
        return out

    # ---- maps -----------------------------------------------------------------------------------------------------
    def step(self, z, statistics=None, rng=None):
        """One C15-9 update (bayesreg's eb_update) from z with exact (rng None) or MC statistics: (z_new, record)."""
        eta = eta_of(z)
        stats = self.exact_statistics(eta) if rng is None else self.mc_statistics(eta, rng)
        new, record = eb_update(eta, stats, CONFIG)
        return z_of(new), record

    def ratio_step(self, z):
        """The recorded ratio-form values (C15-9 fallback form) of the exact statistics, as a map."""
        _, record = self.step(z)
        c = record["candidates"]
        return np.log([c["nu"]["ratio"], np.sqrt(c["tau_1"]["ratio"]), np.sqrt(c["tau_2"]["ratio"])])

    # ---- evidence (marginal form) ---------------------------------------------------------------------------------
    def log_evidence(self, z):
        eta = eta_of(z)
        Pi = self.prior(eta)
        Pinv = scipy.linalg.cho_solve(scipy.linalg.cho_factor(Pi), np.eye(self.n))
        total = 0.
        for J, _, y in self.pairs:
            C = J@Pinv@J.T+np.eye(J.shape[0])
            cf = scipy.linalg.cho_factor(C)
            total += -.5*y@scipy.linalg.cho_solve(cf, y)-np.sum(np.log(np.diag(cf[0])))-.5*len(y)*np.log(2*np.pi)
        return float(total)

    def marginal_gradient(self, z):
        """d l / dz from the marginal form, alpha held (tau_0 the largest: alpha_nug constant)."""
        eta = eta_of(z)
        Pi = self.prior(eta)
        Pinv = scipy.linalg.cho_solve(scipy.linalg.cho_factor(Pi), np.eye(self.n))
        dPi = [self.A_nu]
        tq2 = self.tau_q2(eta)
        for k in (1, 2):
            dPi.append(-self.Ddiv.T@((self.c*self.P[:, k]/tq2**2)[:, None]*self.Ddiv))     # d Pi / d tau_k^2
        grad = np.zeros(3)
        for J, _, y in self.pairs:
            C = J@Pinv@J.T+np.eye(J.shape[0])
            cf = scipy.linalg.cho_factor(C)
            v = Pinv@J.T@scipy.linalg.cho_solve(cf, y)
            M = Pinv@J.T@scipy.linalg.cho_solve(cf, J@Pinv)
            for j, A in enumerate(dPi):
                grad[j] += .5*(float(np.sum(M*A))-float(v@A@v))
        # chain rule: d/dlog nu = nu d/dnu; d/dlog tau_k = 2 tau_k^2 d/dtau_k^2
        return grad*np.array([eta.nu, 2*eta.tau2[1], 2*eta.tau2[2]])

    # ---- predicted MC noise -------------------------------------------------------------------------------------
    def innovation_cov(self, eta, S0=S, S_post=S):
        """Covariance of one MC step in z at eta (delta method): log nu_new = log N - log Q with N = sum_p (g - nu t);
        log tau_k,new = 1/2 (log tau_k^2 + log sum m_hat - log D_k), D_k = sum_p (m_prior - m_var)."""
        Pi = self.prior(eta)
        Sig0 = scipy.linalg.cho_solve(scipy.linalg.cho_factor(Pi), np.eye(self.n))
        Bt = self.btilde(eta)[1:]
        ops = [self.A_nu, *Bt]                                   # quadratic forms entering N (x nu) and D_1, D_2
        signs = np.array([eta.nu, 1., 1.])
        cov = np.zeros((3, 3))
        N = 0.
        D = np.zeros(2)
        for J, G, y in self.pairs:
            Sig = scipy.linalg.cho_solve(scipy.linalg.cho_factor(G+Pi), np.eye(self.n))
            N += eta.nu*(np.sum(Sig0*self.A_nu)-np.sum(Sig*self.A_nu))
            D += [np.sum(Sig0*B)-np.sum(Sig*B) for B in Bt]
            for a in range(3):
                for b in range(a, 3):
                    # prior side: means of S0 quadratic forms; posterior side: t_nu raw mean (S), m_var ddof 1 (S - 1)
                    post_dof = S_post if a == 0 and b == 0 else (S_post if a == 0 or b == 0 else S_post-1)
                    value = signs[a]*signs[b]*(2*_quad_trace(ops[a], Sig0, ops[b], Sig0)/S0
                                               + 2*_quad_trace(ops[a], Sig, ops[b], Sig)/post_dof)
                    cov[a, b] += value
                    cov[b, a] = cov[a, b]
        grad = np.diag([1/N, -.5/D[0], -.5/D[1]])
        return grad@cov@grad.T


@lru_cache(maxsize=1)
def toy():
    return Toy()
