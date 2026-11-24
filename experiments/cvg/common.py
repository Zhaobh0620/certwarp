"""CVG (L0 convergence diagnosis): shared helpers.  Diagnostic code only; bayesreg/ and jd_curl/ are not modified.

Imports come from the code snapshot named by the environment variable CVG_CODE (a `git archive` export of HEAD's
bayesreg/, jd_curl/ and configs/ under /home/rmcfz/seq_vp_tag_results/cvg_v1/code/<sha>/, made by launch.sh), so that
concurrent edits of the working tree cannot change a running diagnosis; without CVG_CODE the repository itself.

What lives here:
- `level_inputs_from_cache`: the solve_multilevel LevelInput tuple from the prep.py cache (levels kept there are real,
  the others placeholders that carry only the grid: solve_multilevel with max_levels <= kept levels never reads them).
- `IntensitySigma`: the diagnostic arm S (NOT a method configuration): the C11-2 Gaussian M step, then every
  structure / tissue channel's sigma_c^2 replaced by the intensity channel's value of the same M step.
- `CaptureBackend`: bayesreg.solve.NumpyBackend that remembers the last certified u (to save accepted states at
  chosen step counts without touching the solver); every other behaviour is the parent's.
- `RunLog`: the solver's log callable, filtered (accept / jd_scaling / round / transfer / budget rows, and candidate
  rows only when rejected) to log.jsonl, plus the saving of accepted states.
"""
import hashlib
import json
import os
import time
from pathlib import Path
import sys

CODE = os.environ.get("CVG_CODE")
if CODE:
    sys.path.insert(0, CODE)
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from jd_curl.validation import readonly_copy  # noqa: E402
from bayesreg.grids import four_level_grids  # noqa: E402
from bayesreg.solve import Eta, GaussianMStep, LevelInput, NumpyBackend  # noqa: E402
from bayesreg.solve.stopping import state_hash  # noqa: E402

RESULTS = Path("/home/rmcfz/seq_vp_tag_results/cvg_v1")
DATA_ROOT = Path("/home/rmcfz/code/python_projects/seq_vp_tag/data/three_d")     # the repository's data symlink (read only)
CHANNEL_SOURCES = ("structure", "tissue", "none")
SIGMA_MODES = ("em", "intensity")
DIAGNOSTIC_ONLY = "诊断臂，不是方法配置"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_record():
    """Where bayesreg was imported from, and the snapshot's SHA file when there is one."""
    import bayesreg
    root = Path(bayesreg.__file__).resolve().parents[1]
    sha = root/"SNAPSHOT_SHA"
    return dict(cvg_code=CODE, bayesreg_root=str(root), snapshot_sha=sha.read_text().strip() if sha.is_file() else None,
                scripts={p.name: sha256_file(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))})


def write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=1, default=_default)+"\n", encoding="utf-8")


def _default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def load_cache(path):
    cache = np.load(path, allow_pickle=False)
    meta = json.loads(str(cache["meta_json"]))
    return cache, meta


def grids_of(pair, meta):
    grids = four_level_grids(pair.grid)
    for ell, gid in meta["grid_ids"].items():
        if grids[int(ell)].grid_id != gid:
            raise RuntimeError(f"level {ell}: grid id differs from the cache")
    return grids


def channel_arrays(cache, ell, channel_source):
    """(fixed, moving) model channels of level ell exactly as bayesreg.solve.level_inputs builds them from a
    seg.channels.ChannelLevel: [F~] (none) or [F~] + sqrt(P~^k), k = 1..K (tissue: the tissue simplex; structure: the
    structure simplex; background channel 0 dropped, C20 D2 default)."""
    fixed, moving = cache[f"fixed_L{ell}"], cache[f"moving_L{ell}"]
    if channel_source == "none":
        return readonly_copy(fixed[None]), readonly_copy(moving[None])
    layer = "tissue" if channel_source == "tissue" else "structure"
    s_fixed = readonly_copy(np.sqrt(cache[f"{layer}_fixed_L{ell}"][1:]))
    s_moving = readonly_copy(np.sqrt(cache[f"{layer}_moving_L{ell}"][1:]))
    return (readonly_copy(np.concatenate([fixed[None], s_fixed])),
            readonly_copy(np.concatenate([moving[None], s_moving])))


def level_inputs_from_cache(cache, meta, grids, channel_source):
    kept = sorted(int(k) for k in meta["grid_ids"])
    out = []
    for ell, grid in enumerate(grids):
        if ell in kept:
            fixed, moving = channel_arrays(cache, ell, channel_source)
            mask = cache[f"mask_L{ell}"].astype(bool)
            mask.flags.writeable = False
            out.append(LevelInput(ell, grid, fixed, moving, mask, readonly_copy(cache[f"tissue_fixed_L{ell}"])))
        else:
            out.append(LevelInput(ell, grid, None, None, None, None))      # placeholder: grid only
    return tuple(out), kept


def eta_of(meta, nu=None):
    record = meta["eta_tmp"]
    if nu is None:
        return Eta(record["alpha"], record["nu"], tuple(record["tau2"]), source=record["source"])
    return Eta(record["alpha"], float(nu), tuple(record["tau2"]),
               source=f"CVG diagnostic: eta_tmp (P2) with nu = {float(nu):g} (nu_tmp = {record['nu']:.6g})")


class IntensitySigma(GaussianMStep):
    """Diagnostic arm S (诊断臂，不是方法配置): the C11-2 Gaussian M step (a = b = 0, w = 1), then sigma_c^2 of every
    channel c >= 1 set to the intensity channel's sigma_0^2 of the same M step."""
    name = "cvg_intensity_sigma (diagnostic arm, not a method configuration)"

    def _update(self, state, point):
        value = np.array(super()._update(state, point), dtype=np.float64)
        self.last_em = value.copy()
        value[1:] = value[0]
        return value

    def record(self):
        return dict(super().record(), policy=self.name, note=DIAGNOSTIC_ONLY,
                    rule="sigma_c^2 (c >= 1) := sigma_0^2 (intensity) of the same Gaussian M step")


def sigma_policy(mode):
    if mode == "em":
        return GaussianMStep()
    if mode == "intensity":
        return IntensitySigma()
    raise ValueError(mode)


class CaptureBackend(NumpyBackend):
    """NumpyBackend that keeps the last u given to `certify` (diagnostic hook; the parent's behaviour unchanged)."""

    def certify(self, u, grid):
        outcome = super().certify(u, grid)
        self.last_u, self.last_grid, self.last_outcome = u, grid, outcome
        return outcome


class RunLog:
    """The solver log callable: writes a filtered log.jsonl and saves the accepted states at the step counts in
    `save_at` (per level) to states/u_L<level>_k<steps>.npz (state_hash checked against the accept row)."""
    KEEP = ("accept", "jd_scaling", "round", "transfer", "budget", "gf_recheck",
            "warning")                  # JDTI: C09.1-3a certificate_bernstein check (b) warning rows

    def __init__(self, out, backend, save_at=()):
        self.out, self.backend, self.save_at = Path(out), backend, set(int(k) for k in save_at)
        (self.out/"states").mkdir(exist_ok=True)
        self.handle = open(self.out/"log.jsonl", "w", encoding="utf-8")
        self.candidates, self.rejected = 0, {}
        self.accepts = []               # (level, k, tau, energy, gradient_norm, step_ratio, min_jd_lower_bound, iteration, candidates, seconds)
        self.saved = []
        self._tries = 0
        self.t0 = time.perf_counter()

    def __call__(self, row):
        kind = row.get("kind")
        if kind == "candidate":
            self.candidates += 1
            self._tries += 1
            if not row.get("accepted"):
                reason = row.get("reason")
                self.rejected[reason] = self.rejected.get(reason, 0)+1
                self._write(row)
            return
        if kind in self.KEEP:
            self._write(row)
        if kind == "accept":
            k = int(row["accepted_steps"])
            self.accepts.append((row.get("level"), k, row["tau"], row["energy"], row["gradient_norm"],
                                 row["step_ratio"], row["min_jd_lower_bound"], row["iteration"], self._tries,
                                 round(time.perf_counter()-self.t0, 3)))
            self._tries = 0
            if k % 100 == 0:
                self.handle.flush()
            if k in self.save_at:
                self.save(row, f"u_L{row.get('level')}_k{k}.npz")

    def save(self, row, name):
        u = np.asarray(self.backend.last_u)
        digest = state_hash(u)
        if digest != row["state_hash"]:
            raise RuntimeError(f"captured state {digest[:12]} != accept row {row['state_hash'][:12]}")
        np.savez(self.out/"states"/name, u=u, state_hash=digest, level=row.get("level"),
                 accepted_steps=row["accepted_steps"])
        self.saved.append(dict(file=name, state_hash=digest, level=row.get("level"), k=row["accepted_steps"]))

    def _write(self, row):
        self.handle.write(json.dumps(row, default=_default)+"\n")

    def close(self):
        self.handle.close()
