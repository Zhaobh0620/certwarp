"""Intermediate native states and the profile objective L of one run (contract C12-4a item 8, C16 WST item, C20 D59; WST 2026-11-02,
WARM 2026-11-03: default off, records only; every value of the solve is unchanged).  Used by `run_pair --init-field` (entry and end of
the native level) and by `--checkpoint-accepted N` / `--checkpoint-seconds S` (the states in between).

Profile objective (C11-1 and C11-8 with a = b = 0, sigma at its closed-form minimum for the fixed u): with the variance parameters p,
their counts N_p and S_p = sum of <w_c rho_c^2>_B over the channels of p,

    L(u) = sum_p [ (N_p/2 + 1) log(S_p/(N_p + 2)) + (N_p + 2)/2 ] + Pi_D(u) + Pi_J(u) + Pi_C(u).

brain: one parameter per channel, N_p = N_q.  support: the intensity (N_q) and every structure channel with a non-empty support (N_c).
support_shared: the intensity and one parameter for the non-empty structure channels (N_S).  S_c = 2 sigma_c^2 E_c of an evaluation at u
with any sigma (the evaluation's own sigma is undone), the frozen support v of the level inside E_c.  L compares only within one
model (the same eta, channels, grid and, for support and support_shared, the same v); across levels it means nothing (C11-3).  So the
entry, the checkpoints and the end of one run compare; two seeds, or a seed and a run from u = 0, do not (C12-4a item 8).  S_p = 0 or
not finite (review GATEA_tools M14): log S_p has no value, L is unbounded: `L` is None and `L_undefined` names the parameters; nothing raises.

A checkpoint is a formal commit: the certified state of an accepted step of the native level, written as <directory>/ckpt_<kkk>/disp.npz
(bayesreg.solve.export.write_disp; bayesreg.eval.cli reads such a directory as kind disp).  Due after every N-th accepted step, or at
the first accepted step at least S seconds after the previous checkpoint (after the entry for the first).  No accepted step, no
checkpoint: that is itself the reading (the plan's "stagnation").  Its record: the number, accepted steps so far, seconds, L and its
parts, RMS(u - u_entry) over the brain nodes and over all nodes (mm), the certificate of the state, the write time.  With a checkpoint
configuration the entry state of the native level is written too, as <directory>/entry/disp.npz: a certified state, not a checkpoint (no
number, not in n_checkpoints), so that the entry goes through the same evaluation (WARM).

Records only: whatever goes wrong here (a write, the entry evaluation, L) lands in `errors`; the run, its status and its exit code go on.  The
records take time (one more evaluation at the entry, one file per entry or checkpoint, `write_seconds`): under --wall-seconds a run with
them is a prefix of the run without them.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError, finite_real

from .export import write_disp
from .outer import CommitTracker
from .stopping import state_hash

BLOCKS = ("diffusion", "logvol", "curl")
ENTRY_DIRECTORY = "entry"                # <directory>/entry/disp.npz: the native entry state (not a checkpoint)
L_DEFINITION = ("L = sum_p [(N_p/2 + 1) log(S_p/(N_p + 2)) + (N_p + 2)/2] + diffusion + logvol + curl (C11-1, C11-8; a = b = 0; compare "
                "only within one model and level)")


@dataclass(frozen=True)
class CheckpointConfig:
    """every_accepted N or every_seconds S (exactly one)."""
    every_accepted: int | None = None
    every_seconds: float | None = None

    def __post_init__(self):
        if (self.every_accepted is None) == (self.every_seconds is None):
            raise InputValidationError("checkpoints: exactly one of every_accepted, every_seconds (C12-4a item 8)")
        if self.every_accepted is not None and (not isinstance(self.every_accepted, int) or self.every_accepted < 1):
            raise InputValidationError("checkpoints: every_accepted a positive integer")
        if self.every_seconds is not None:
            object.__setattr__(self, "every_seconds", finite_real(self.every_seconds, "every_seconds", positive=True))

    def record(self):
        return dict(every_accepted=self.every_accepted, every_seconds=self.every_seconds)


def variance_groups(state, channel_sums, n_q_value, level_support=None):
    """[(name, N_p, S_p)] of the variance parameters (module docstring); channel_sums: S_c per channel."""
    domain = getattr(level_support, "sigma_domain", None)
    if level_support is None:
        return [(f"c{c}", n_q_value, float(s)) for c, s in enumerate(channel_sums)]
    nonempty = list(level_support.nonempty)
    groups = [("I", n_q_value, float(channel_sums[0]))]
    if domain == "support":
        n_c = level_support.n_c
        groups += [(f"c{c}", float(n_c[c-1]), float(channel_sums[c])) for c in nonempty]
    elif nonempty:
        groups.append(("S", float(level_support.n_s()), float(sum(channel_sums[c] for c in nonempty))))
    return groups


def profile_record(state, point, n_q_value, level_support=None):
    """The fields of L at one evaluation (`point` evaluated with `state`): S_c, the groups, sigma^2 at its minimum, the prior blocks,
    F_B as evaluated with sigma_used, and L (None, with `L_undefined`, when some S_p is zero or not finite: review GATEA_tools M14)."""
    energies = point.energies
    channel_sums = [2*float(s)*float(e) for s, e in zip(np.asarray(state.sigma2), energies["channel"])]
    groups = variance_groups(state, channel_sums, n_q_value, level_support)
    blocks = {name: float(energies[name]) for name in BLOCKS}
    value = sum(blocks.values())
    hat, undefined = {}, []
    for name, count, total in groups:
        sigma2 = total/(count+2)
        hat[name] = sigma2 if math.isfinite(sigma2) else None
        if sigma2 > 0 and math.isfinite(sigma2):
            value += (count/2+1)*math.log(sigma2)+(count+2)/2
        else:
            undefined.append(name)
    known = (lambda s: s if math.isfinite(s) else None)      # the record stays valid JSON (hyper.json refuses NaN and infinities)
    record = dict(L=None if undefined else value, F_B=float(point.energy), S_c=[known(s) for s in channel_sums],
                  groups=[dict(name=n, N=c, S=known(s)) for n, c, s in groups], sigma2_hat=hat,
                  sigma2_used=[float(s) for s in np.asarray(state.sigma2)], blocks=blocks, logvol_parts=energies.get("logvol_parts"),
                  N_q=n_q_value)
    if undefined:
        record["L_undefined"] = undefined
    return record


class ProfileRecorder:
    """The record of the native level: L at its entry and end, the checkpoints in between.  `directory`, `config`: where and when
    to write checkpoints (None: entry and end only); `log`: the run's log.jsonl writer; `grid`: the native grid; `tag`: the seed tag of
    C16 written into each disp.npz (None without a seed); `clock`: seconds (tests inject it)."""

    def __init__(self, *, grid, directory=None, config=None, log=None, tag=None, clock=time.perf_counter):
        if (directory is None) != (config is None):
            raise InputValidationError("ProfileRecorder: directory and config go together")
        self.grid, self.directory, self.config = grid, None if directory is None else Path(directory), config
        self.log, self.tag, self.clock = (lambda row: None) if log is None else log, tag, clock
        self.entry, self.end, self.checkpoints, self.errors = None, None, [], []
        self.u_entry, self.mask = None, None
        self.accepted = 0
        self.level, self.round = None, None
        self.state, self.level_support, self.n_q = None, None, None
        self.started = None
        self.last_written = None
        self.attempts = 0

    # ---- the round's context ------------------------------------------------------------------------------------------
    def begin_round(self, backend, state, commit, *, level, round_, n_q_value, mask, level_support=None):
        """Called with the round's state before the solve; the first call records L at the entry state (one extra evaluation) and,
        with a checkpoint configuration, writes the entry state.  Nothing here reaches the solve: failures land in `errors`."""
        self.state, self.level, self.round, self.n_q, self.level_support = state, level, round_, n_q_value, level_support
        if self.u_entry is not None:
            return
        self.u_entry, self.mask = commit.u, np.asarray(mask, dtype=bool)
        self.started = self.last_written = self.clock()
        files = {}
        try:
            if self.directory is not None:
                files = self.write_state(self.directory/ENTRY_DIRECTORY, commit.u)
        except Exception as exc:                  # noqa: BLE001  records only (module docstring)
            self.errors.append(f"entry file: {type(exc).__name__}: {exc}")
        try:
            point = backend.evaluate(state, commit.u)
            if isinstance(point, InvalidTrial):
                self.errors.append(f"entry: {point.reason}: {point.message}")
                return
            self.entry = dict(profile_record(state, point, n_q_value, level_support), level=level, round=round_,
                              state_hash=commit.cert.state_hash, min_jd_lower_bound=commit.cert.min_jd_lower_bound, **files)
            self.log(dict(kind="profile", at="entry", **self.entry))
        except Exception as exc:                  # noqa: BLE001  records only
            self.errors.append(f"entry: {type(exc).__name__}: {exc}")

    def tracker(self, commit):
        return ProfileTracker(commit, recorder=self)

    def rms_from_entry(self, u):
        """RMS(u - u_entry) over the brain nodes, mm."""
        delta = np.asarray(u)[self.mask]-np.asarray(self.u_entry)[self.mask]
        return float(np.sqrt(np.mean(np.sum(delta*delta, axis=-1))))

    def rms_from_entry_all(self, u):
        """RMS(u - u_entry) over all nodes of the grid, mm (the zero faces included)."""
        delta = np.asarray(u)-np.asarray(self.u_entry)
        return float(np.sqrt(np.mean(np.sum(delta*delta, axis=-1))))

    def write_state(self, directory, u):
        """`directory`/disp.npz (certified state, formal commit): {directory, state_hash of the file's u, write_seconds}."""
        tick = self.clock()
        directory.mkdir(parents=True)
        digest = state_hash(u)
        write_disp(directory/"disp.npz", u, self.grid, state_hash=digest, **({} if self.tag is None else dict(init_field=self.tag)))
        return dict(directory=str(directory), file_state_hash=digest, write_seconds=self.clock()-tick)

    # ---- accepted steps -----------------------------------------------------------------------------------------------
    def due(self):
        if self.config is None:
            return False
        if self.config.every_accepted is not None:
            return self.accepted % self.config.every_accepted == 0
        return self.clock()-self.last_written >= self.config.every_seconds

    def observe(self, commit, point):
        self.accepted += 1
        if not self.due():
            return
        tick = self.clock()
        k = self.attempts                         # a failed attempt keeps its number and its partial files (hard constraint 4)
        self.attempts += 1
        try:
            files = self.write_state(self.directory/f"ckpt_{k:03d}", commit.u)
            record = dict(k=k, directory=files["directory"], level=self.level, round=self.round, accepted=self.accepted,
                          seconds_from_entry=tick-self.started, rms_from_entry_mm=self.rms_from_entry(commit.u),
                          rms_from_entry_all_mm=self.rms_from_entry_all(commit.u), state_hash=files["file_state_hash"],
                          cert_status=commit.cert.status, min_jd_lower_bound=commit.cert.min_jd_lower_bound,
                          **profile_record(self.state, point, self.n_q, self.level_support))
            record["write_seconds"] = self.clock()-tick
            self.checkpoints.append(record)
            self.log(dict(kind="checkpoint", **record))
        except Exception as exc:                  # noqa: BLE001  records only (module docstring)
            self.errors.append(f"checkpoint {k}: {type(exc).__name__}: {exc}")
        self.last_written = self.clock()

    # ---- end ----------------------------------------------------------------------------------------------------------
    def end_round(self, commit, point):
        """The round's last formal commit and its evaluation: the last call is the native level's end."""
        if point is None or self.u_entry is None:
            return
        try:
            self.end = dict(profile_record(self.state, point, self.n_q, self.level_support), level=self.level, round=self.round,
                            accepted=self.accepted, rms_from_entry_mm=self.rms_from_entry(commit.u),
                            rms_from_entry_all_mm=self.rms_from_entry_all(commit.u), state_hash=commit.cert.state_hash,
                            min_jd_lower_bound=commit.cert.min_jd_lower_bound)
        except Exception as exc:                  # noqa: BLE001  records only
            self.errors.append(f"end: {type(exc).__name__}: {exc}")

    def record(self):
        return dict(definition=L_DEFINITION, entry=self.entry, end=self.end, accepted=self.accepted, checkpoints=self.checkpoints,
                    n_checkpoints=len(self.checkpoints), config=None if self.config is None else self.config.record(),
                    directory=None if self.directory is None else str(self.directory), errors=self.errors)


class ProfileTracker(CommitTracker):
    """CommitTracker that shows every accepted step to its ProfileRecorder (option (a), PG: every accepted step is a formal commit)."""

    def __init__(self, commit, *, recorder):
        super().__init__(commit)
        self.recorder = recorder

    def observe(self, point):
        self.recorder.observe(self.commit, point)


__all__ = ["CheckpointConfig", "ENTRY_DIRECTORY", "L_DEFINITION", "ProfileRecorder", "ProfileTracker", "profile_record", "variance_groups"]
