"""C12-2 level transfer: re-certification of s_k u_prolonged, s_k = .9^k (k = 0..6), first PASS wins; seven
rejections -> transfer_rejected with every row recorded.  The certificate is mocked (a wrapper that rejects
the first m calls on the target grid) so that the back-off sequence is exercised deterministically; every
call still goes through the real certify_map first, so the rows carry real hashes."""
from dataclasses import replace

import numpy as np

from bayesreg.grids import prolong_displacement
from bayesreg.solve import (TRANSFER_ALPHAS, Eta, FixedSigma, NumpyBackend, SolverConfig, solve_multilevel,
                            transfer)
from bayesreg.solve.stopping import TRANSFER_REJECTED, state_hash

from . import _synthetic as syn


class RejectingBackend(NumpyBackend):
    """Certificates on `target` grids: the first `reject` calls are turned into JACOBIAN_VIOLATION."""

    def __init__(self, reject, target_shapes):
        super().__init__()
        self.reject, self.target_shapes, self.seen = reject, set(target_shapes), []

    def certify(self, u, grid):
        outcome = super().certify(u, grid)
        if grid.shape in self.target_shapes:
            self.seen.append(np.max(np.abs(u)))
            if self.reject > 0:
                self.reject -= 1
                return replace(outcome, passed=False, status="infeasible", cert_reason="JACOBIAN_VIOLATION",
                               rejection="JACOBIAN_VIOLATION", min_jd_lower_bound=None)
        return outcome


def coarse_and_fine():
    native, levels, _ = syn.four_levels(26)
    coarse, fine = levels[1].grid, levels[2].grid
    u = syn.solenoidal_displacement(coarse, .8)
    return coarse, fine, u


def test_alphas_are_point_nine_powers():
    assert TRANSFER_ALPHAS == tuple(.9**k for k in range(7)) and len(TRANSFER_ALPHAS) == 7


def test_backoff_takes_the_first_pass():
    coarse, fine, u = coarse_and_fine()
    backend = RejectingBackend(2, {fine.shape})
    seed, outcome, rows = transfer(backend, u, coarse, fine)
    prolonged = prolong_displacement(u, coarse, fine)
    assert [row["k"] for row in rows] == [0, 1, 2] and [row["scale"] for row in rows] == [1., .9, .9**2]
    assert [row["cert"]["passed"] for row in rows] == [False, False, True]
    assert seed.tobytes() == (.9**2*prolonged).tobytes()             # relative to the original seed
    assert outcome.passed and outcome.state_hash == state_hash(seed)
    assert np.allclose(backend.seen, [np.max(np.abs(s*prolonged)) for s in (1., .9, .81)], rtol=0, atol=0)


def test_seven_rejections():
    coarse, fine, u = coarse_and_fine()
    backend = RejectingBackend(7, {fine.shape})
    seed, outcome, rows = transfer(backend, u, coarse, fine)
    assert seed is None and outcome is None
    assert len(rows) == 7 and [row["scale"] for row in rows] == list(TRANSFER_ALPHAS)
    assert all(row["cert"]["rejection"] == "JACOBIAN_VIOLATION" for row in rows)


def test_run_ends_transfer_rejected():
    """Through the level loop: the transfer into level 1 is rejected seven times -> run status
    transfer_rejected, no native output (result.native False, commit None), seven rows in the record."""
    native, levels, _ = syn.four_levels(26)
    backend = RejectingBackend(7, {levels[1].grid.shape})
    result = solve_multilevel(backend, levels, eta=Eta(1., 1., (1., 1., 1., 1.)), sigma=FixedSigma([1.]),
                              config=SolverConfig(rounds=(1, 1, 1, 1)))
    assert result.status == TRANSFER_REJECTED and result.wording == "运行失败"
    assert not result.native and result.commit is None and result.level == 1
    assert len(result.transfers) == 1 and len(result.transfers[0]["rows"]) == 7
    assert result.transfers[0]["accepted_k"] is None
    assert [r["level"] for r in result.rounds] == [0]


def test_run_continues_after_backoff():
    """A transfer accepted at k = 3 starts the next level from .9^3 u_prolonged; max_levels = 2 ends with
    stopped_before_native (development switch) after level 1."""
    native, levels, _ = syn.four_levels(26)
    backend = RejectingBackend(3, {levels[1].grid.shape})
    result = solve_multilevel(backend, levels, eta=Eta(1., 1., (1., 1., 1., 1.)), sigma=FixedSigma([1.]),
                              config=SolverConfig(rounds=(1, 1, 1, 1), max_levels=2))
    assert result.status == "stopped_before_native" and not result.native and result.level == 1
    assert result.transfers[0]["accepted_k"] == 3
    assert [r["level"] for r in result.rounds] == [0, 1] and result.commit.cert.passed
