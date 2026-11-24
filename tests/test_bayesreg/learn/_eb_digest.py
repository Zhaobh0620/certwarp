"""Digests of the eb_count = "dim" path of bayesreg.learn for the C15-9 gate "dim 逐位不变的范围" (contract C15-9,
EBC2): with eb_count = "dim" the arrays (per-pair statistics, Gauss-point sums, posterior qc), the per-round update
records and eta trajectory and the result record are bitwise those of the code before C15-9 was implemented (commit
7579645); provenance gains only the key eb_count (EBConfig.record(); the CLI's args).

Each digest is the sha256 of a canonical JSON form: float by float.hex (exact), int / bool / str / None as is,
numpy arrays by the sha256 of their C-order float64 bytes, tuples as lists, dict keys sorted; wall-clock values are
dropped (keys containing "seconds", and "timing"); the key eb_count of a config record is dropped after checking that
it is "dim" (the one provenance key C15-9 adds).  The cases need no data/, no certificate compilation (a scripted
all-pass / mixed certificate backend with the real C07 configuration) and run in a few seconds:
  gauss    gauss_point_statistics on the aniso 7x6x5 state (with and without a Gauss-point variance);
  pair     pair_statistics (LaplaceSystem -> 16 PCG samples -> posterior_maps -> Gauss sums) on the 6^3 case of
           tests/test_bayesreg/infer with precisions x 400, 13 of 16 samples certified;
  update   eb_update under P2, P2 + learn_tau0, P3, P1 on hand-made statistics with exclusions;
  driver   run_population_eb with injected statistics: 3 rounds to the cap, and a run meeting the criterion;
  config   the EBConfig record of P2 and of P3.
Every EBConfig here passes eb_count = "dim" explicitly (`DIM`): the default is "gull_mackay" since BZ 2026-10-30 (EBD,
C15-9 / D45).  The code of 7579645 has no eb_count field (its update is the C15-1 = dim one), so `DIM` is empty there.
Every ModelState here is given logvol_points = "gauss27" explicitly (`GAUSS27`): the default is "gauss27+closed27"
since work package DEF (2026-10-31; contract 8a4d060 C05-2b, D48).  The code of 7579645 has the option with that
default, the code before C05-2a none (`GAUSS27` is then empty).
Golden values: tests/test_bayesreg/_golden/make_golden_eb.py runs this module on the code of 7579645 (git archive in a
temporary directory; the helpers it imports must be byte-identical there) -> golden_7579645_eb.json; the constants
GOLDEN_7579645_EB of learn/test_eb_count.py are compared with that record (golden_7579645_eb_def.json since the
explicit GAUSS27; golden_7579645_eb_ebd.json, made after the explicit DIM, and golden_7579645_eb.json, made before it,
hold the same digests).
"""
from dataclasses import fields
import hashlib
import json

import numpy as np

from bayesreg.learn import EBConfig, EBPair, MapOutcome, PairStatistics, eb_update, gauss_point_statistics
from bayesreg.learn import pair_statistics, run_population_eb
from bayesreg.learn import population as pop
from bayesreg.model import ModelState
from bayesreg.solve import Eta

from ..infer import _cases as cs
from ..model._helpers import aniso, feasible_field, make_state

TAU2 = (1., .25, .36, .16)
DIM = dict(eb_count="dim") if "eb_count" in {f.name for f in fields(EBConfig)} else {}    # {} on the code of 7579645
GAUSS27 = dict(logvol_points="gauss27") if "logvol_points" in {f.name for f in fields(ModelState)} else {}   # DEF


def _gauss27(state):
    """The state with logvol_points "gauss27" (the default before DEF, explicit since)."""
    return state.replace(**GAUSS27) if GAUSS27 else state


def canonical(obj, *, path=()):
    if isinstance(obj, dict):
        out = {}
        for key in sorted(obj, key=str):
            name = str(key)
            if "seconds" in name or name == "timing":
                continue
            if name == "eb_count" and path and path[-1] in ("config", "eb"):
                if obj[key] != "dim":
                    raise AssertionError(f"eb_count = {obj[key]!r} in a dim digest")
                continue
            out[name] = canonical(obj[key], path=path+(name,))
        return out
    if isinstance(obj, (list, tuple)):
        return [canonical(v, path=path) for v in obj]
    if isinstance(obj, np.ndarray):
        array = np.ascontiguousarray(np.asarray(obj, dtype=np.float64))
        return dict(shape=list(array.shape), sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())
    if isinstance(obj, (bool, np.bool_)) or obj is None or isinstance(obj, str):
        return obj if not isinstance(obj, np.bool_) else bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.longdouble):
        return "ld:"+np.format_float_scientific(obj, unique=True)
    if isinstance(obj, (float, np.floating)):
        return float(obj).hex()
    if isinstance(obj, Eta):
        return canonical(obj.record(), path=path)
    raise TypeError(f"canonical: {type(obj)} at {path}")


def digest(obj):
    return hashlib.sha256(json.dumps(canonical(obj), sort_keys=True).encode()).hexdigest()


# ---- cases ---------------------------------------------------------------------------------------------------

def gauss_case():
    grid = aniso()
    state = _gauss27(make_state(grid, seed=4))
    u = feasible_field(grid, np.random.default_rng(5), .05)
    var = np.random.default_rng(6).uniform(.1, .9, (grid.n_cells, 27))
    fields = ("kq_quadratic", "curl_quadratic", "tau_hat", "tau_var", "tau_den", "n_gauss")    # those of 7579645
    return [{f: getattr(s, f) for f in fields} for s in (gauss_point_statistics(state, u),
                                                         gauss_point_statistics(state, u, var))]


def pair_case():
    grid, state = cs.cube6(factor=400.)
    state = _gauss27(state)
    u = cs.u_hat_of(grid)
    passes = [True]*16
    for k in (3, 9, 14):
        passes[k] = False
    backend = cs.ScriptedCertificate(passes)
    stats, record = pair_statistics(state, u, pair_id="digest_pair", config=EBConfig(n_samples=16, **DIM),
                                    seed=20261030, backend=backend, sigma_post=(1e-3, 2e-3, 3e-3))
    return [stats.record(), record]


def hand_stats(pair_id="p", *, n=300, kq=(2., .5), curl=(1.5, .5), tau_hat=(1., 2., 3., 4.), tau_var=(.5, .5, .5, .5),
               tau_den=(10., 20., 30., 40.), conv=16, passed=16, sigma_post=(.01, .02), n_q=1000.):
    return PairStatistics(pair_id=pair_id, n_free=n, n_curl=n, n_q=n_q, kq_quadratic=kq[0], curl_quadratic=curl[0],
                          kq_trace=kq[1] if conv else float("nan"), kq_trace_stderr=.01,
                          curl_trace=curl[1] if conv else float("nan"), curl_trace_stderr=.01, tau_hat=tau_hat,
                          tau_var=tau_var if passed >= 2 else None, tau_den=tau_den, sigma_used=(.01, .02),
                          sigma_post=sigma_post, n_samples=16, n_converged=conv, n_passed=passed,
                          pass_rate=passed/16, pass_rate_low=passed/16 < .8)


def update_case():
    rows = [hand_stats("a"), hand_stats("b", curl=(2.5, .7), tau_hat=(2., 4., 6., 8.), kq=(3.1, .3)),
            hand_stats("c", conv=0, passed=0), hand_stats("d", passed=1), None,
            hand_stats("e", tau_hat=(1., 2., 3., 400.), tau_var=(.1, .2, .3, .4))]
    p2 = Eta(pop.alpha_nugget(TAU2), 2., TAU2)
    out = []
    for config, eta in ((EBConfig(**DIM), p2), (EBConfig(learn_tau0=True, **DIM), p2),
                        (EBConfig(parametrization="P3", **DIM), Eta(.5, 2., TAU2)),
                        (EBConfig(parametrization="P1", **DIM), Eta(.5, 2., TAU2))):
        new, record = eb_update(eta, rows, config)
        out.append([new.record(), record])
    return out


class _Ready(MapOutcome):
    @property
    def usable(self):
        return self.native and self.status in pop.LAPLACE_ENDINGS


def _pairs(n=2):
    return tuple(EBPair(f"OASIS_OAS1_09{k:02d}_MR1", f"OASIS_OAS1_09{k+50:02d}_MR1", "A", {"check": "digest"})
                 for k in range(n))


def _solve_ready(pair, eta, *, round_index):
    return _Ready("converged_regularized_gradient_B", "digest", True, None, None, [.01], dict(eta=eta))


def _statistics(target):
    count = []

    def statistics(outcome, *, pair_id, config, seed, backend):
        eta = outcome.record["eta"]
        count.append(seed)
        n = 300
        tau2 = eta.tau2
        s = PairStatistics(pair_id=pair_id, n_free=n, n_curl=n, n_q=1e3+len(count), kq_quadratic=1.+.1*len(count),
                           curl_quadratic=n/target(eta.nu), kq_trace=.25, kq_trace_stderr=.01, curl_trace=.0,
                           curl_trace_stderr=0., tau_hat=tuple(t*(10.+.3*k) for k, t in enumerate(tau2)),
                           tau_var=tuple(.01*k for k in range(len(tau2))), tau_den=(10.,)*len(tau2),
                           sigma_used=(.01,), sigma_post=(.01+.001*len(count), .02-.0005*len(count)), n_samples=16,
                           n_converged=16, n_passed=16, pass_rate=1., pass_rate_low=False)
        return s, dict(seed=seed)
    return statistics


def driver_case():
    out = []
    eta_cap = Eta(pop.alpha_nugget(TAU2), 300., TAU2)
    eta_met = Eta(pop.alpha_nugget(TAU2), 104., TAU2)
    for eta0, target in ((eta_cap, lambda nu: 100+.5*(nu-100)), (eta_met, lambda nu: 100+.5*(nu-100))):
        result = run_population_eb(_pairs(), _solve_ready, eta0=eta0, config=EBConfig(**DIM), run_id="digest",
                                   statistics=_statistics(target))
        out.append([result.status, result.wording, result.eta.record(), result.trajectory, result.rounds,
                    result.sigma_prior, result.record])
    return out


def config_case():
    return [dict(config=EBConfig(**DIM).record()), dict(config=EBConfig(parametrization="P3", **DIM).record())]


CASES = dict(gauss=gauss_case, pair=pair_case, update=update_case, driver=driver_case, config=config_case)


def all_digests():
    return {name: digest(build()) for name, build in CASES.items()}
