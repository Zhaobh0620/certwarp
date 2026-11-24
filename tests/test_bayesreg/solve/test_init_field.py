"""run_pair --init-field (work packages WST, WARM; contract C12-4a, C16 WST item, D59; default off, path only).
Synthetic data only: the 26^3 pair of solve/test_run_pair.py, channel source tissue, CPU.

Default unchanged:
  * without the switches the seven record files of this tree have the masked sha256 of golden_c16ca4c_wst_records.json (the
    S and the digests measured on the commit before WST by make_golden_sch.py --ref c16ca4c), arm by arm, with the same key
    sets, and the log rows, the numeric fields of hyper.json and qc.json and the disp / final_native arrays have the digests
    of golden_c16ca4c_wst_digests.json (make_golden_swli.py --ref c16ca4c); no new args key;
  * the SolverConfig record, the learn CLI (no such switch) and the probe (no such switch) are unchanged.
Path only: from the seed that is the transfer state of an ordinary run, the native level of the seeded run is bit for bit the
native level of the ordinary run (sigma, G0, T_B, F_B and GF_B traces, statuses, state hashes, exported state).
Sigma and threshold at the seed: level-start M step at the entry state (C11-3), G0 at u = 0 (C13-2).
Seed admitted: from a run_pair directory and from a frozen configuration C directory (a synthetic stand-in with the keys of the
real files); every record of C16 WST item; entry back-off with the real certificate (.9^k; all seven rejected ->
initialization_failed, exit 2, no fallback to u = 0, nothing exported).
Seed refused (init_field_rejected, exit 5, no solve): pair name, subjects, pair sha256, grid id, shape, dtype, non-finite,
nonzero faces, state hash, file sha256, protocol / manifest mismatch, unknown directory, file name.
Oracle: a truth-label seed refused without --init-field-oracle-ok; with it flagged in every record and field_role oracle_seeded
in bayesreg.eval (not comparable by default); arm A1 with oracle false refused; transitive through a seeded run_pair source.
Firewall: the source manifest is read without its label keys; `warped` is never read.
Combinations: per-level caps (only a3 acts), wall budget, a used-up budget (exports the entry state, handover_only),
sigma_domain support_shared, jd_cell_estimate, logvol_points; parse_args refusals (exit 2, no run directory).
WARM (C12-4a item 8 after the reviews): S_p = 0 gives L null and raises nothing (GATEA_tools M14); a failure of the records lands in errors,
the solve does not notice; the entry field; the records leave the path bit for bit, with and without a seed; L at a budget end; both RMS.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jd_curl.three_d.nifti import file_sha256

from bayesreg.cli import run_pair
from bayesreg.eval.fields import COMPARABLE_ROLES, FIELD_ROLES, ROLE_HANDOVER, ROLE_NATIVE, ROLE_ORACLE_SEEDED, describe_run
from bayesreg.grids import prolong_displacement
from bayesreg.solve import (Caps, FixedSigma, GaussianMStep, InitSeed, NumpyBackend, SolverConfig, eta_temporary, g0_of,
                            solve_multilevel, threshold)
from bayesreg.solve import init_field as IF
from bayesreg.solve import jd_scaling as jds
from bayesreg.solve.multilevel import _template
from bayesreg.solve.init_field import INIT_FIELD_REJECTED
from bayesreg.solve.stopping import BUDGET_EXHAUSTED, CAP_REACHED, INITIALIZATION_FAILED, RUN_STATUSES, RUN_WORDING, state_hash

from . import _run_pair_digest as RD
from . import _sch_records as R
from . import _swli_digest as SD
from . import _synthetic as syn
from ..io_seg._helpers import DEV
from .test_run_pair import read
from .test_sch_run_pair import load

ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = ROOT/"tests"/"test_bayesreg"/"_golden"
RECORDS_GOLDEN = GOLDEN_DIR/"golden_c16ca4c_wst_records.json"
DIGESTS_GOLDEN = GOLDEN_DIR/"golden_c16ca4c_wst_digests.json"
BITWISE = pytest.mark.skipif(np.__version__ != "2.0.2", reason="bitwise golden digests recorded with numpy 2.0.2")
COMMON = ("--rounds", "1,1,1,1", "--max-accepted", "2")
TASK = "pg_weak__four__90s"
WST_ARG_KEYS = ("init_field", "start_level", "init_field_oracle_ok")
WST_SOLVER_KEYS = ("start_level", "levels_skipped", "levels_skipped_reason")


class World:
    """The synthetic inputs and one ordinary source run (run_pair), under one module-scoped tmp root."""

    def __init__(self, root):
        self.root = root
        self.base = RD.write_inputs(root)
        self.pair = self.base[self.base.index("--pair")+1]
        self.pair_name = self.pair.split("/")[1]
        self.pair_dir = root/"data"/self.pair
        self.source = self.run("source", *COMMON)
        self.source_seed = self.source/"final_native.npz"
        with np.load(self.source_seed) as z:
            self.u = np.array(z["u"])
        self.grid_id = json.loads(str(np.load(self.source_seed)["grid"]))["grid_id"]
        self.count = 0

    def run(self, name, *options, expect=0):
        jds.clear_cell_bounds_cache()
        out = self.root/f"run_{name}"
        code = run_pair.main([*self.base, "--out", str(out), *options])
        assert code == expect, (name, code)
        return out

    def seeded(self, name, seed, *options, expect=0):
        return self.run(name, "--init-field", str(seed), "--start-level", "3", *options, expect=expect)

    def fresh(self, stem):
        self.count += 1
        return f"{stem}_{self.count}"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return World(tmp_path_factory.mktemp("wst"))


def write_frozen(world, name, u=None, *, stage="D0-A0", pre=None, pair=None, fixed=None, moving=None, grid_id=None,
                 pair_sha=None, recorded_hash=None, recorded_file_sha=None, task_name=None):
    """A frozen configuration C run directory with the keys of the real files (d0_v1/runs/A0/<pair>): final_native.npz (u,
    warped), final_native.json, protocol.json, inputs/<pair>/manifest.json (with label-derived keys that must not be read)."""
    u = world.u if u is None else u
    pair = world.pair_name if pair is None else pair
    run = world.root/f"frozen_{name}"
    task = run/TASK
    task.mkdir(parents=True)
    np.savez_compressed(task/"final_native.npz", u=u, warped=np.zeros(u.shape[:-1]))
    manifest = json.loads((world.pair_dir/"manifest.json").read_text())
    original_sha = manifest["pair_sha256"]
    manifest.update(pair=pair, baseline_label_metrics=dict(seg35=.5, seg4=.6), labels_policy="no labels in loss",
                    fixed_subject=manifest["fixed_subject"] if fixed is None else fixed,
                    moving_subject=manifest["moving_subject"] if moving is None else moving)
    if pair_sha is not None:
        manifest["pair_sha256"] = pair_sha
    if pre is not None:
        manifest["pair_sha256"] = hashlib.sha256(b"recoloured"+name.encode()).hexdigest()
        manifest["preprocessing_d0"] = dict(dict(source_pair_sha256=original_sha,
                                                 colour=dict(fixed=dict(c_by_seg4_label={"1": .3}))), **pre)
    (run/"inputs"/pair).mkdir(parents=True)
    manifest_path = run/"inputs"/pair/"manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    file_sha = file_sha256(task/"final_native.npz")
    certificate = dict(status="certified_feasible", min_jd_lower_bound=.1, n_infeasible=0, n_uncertain=0,
                       state_hash=state_hash(u) if recorded_hash is None else recorded_hash, reason="positive_enclosure_cover",
                       grid_id=world.grid_id if grid_id is None else grid_id, certificate_version="synthetic")
    (task/"final_native.json").write_text(json.dumps(dict(path="final_native.npz", role="formal_native_outer_commit",
                                                          sha256=file_sha if recorded_file_sha is None else recorded_file_sha,
                                                          certificate=certificate)))
    (run/"protocol.json").write_text(json.dumps(dict(
        stage=stage, task=dict(index=0, pair=pair, route="pg_weak", name=TASK if task_name is None else task_name,
                               wall_seconds=90),
        input_sha256={"pair.npz": manifest["pair_sha256"], "manifest.json": file_sha256(manifest_path)},
        pre_optimizer="fp32-v1:synthetic")))
    return task/"final_native.npz"


# ---- default unchanged -------------------------------------------------------------------------------------------------

@BITWISE
def test_default_record_files_are_byte_identical_under_the_mask(tmp_path):
    golden = json.loads(RECORDS_GOLDEN.read_text(encoding="utf-8"))
    assert golden["ref_commit"].startswith("c16ca4c") and golden["gate_failed"] is False
    gen = load("make_golden_sch", "tests/test_bayesreg/_golden/make_golden_sch.py")
    child = gen.run_worktree(tmp_path/"root")
    for arm, record in golden["arms"].items():
        assert record["unclassified"] == [] and record["solve_quantities_in_S"] == [], arm
        texts = child["texts"][arm]
        assert child["codes"][arm] == record["exit_codes"][0], arm
        assert R.masked_digests(texts, record["S"]) == record["digests"], arm
        assert R.key_sets(texts) == record["key_sets"], arm
        args = json.loads(texts["config.json"])["args"]
        assert not set(WST_ARG_KEYS) & set(args), arm


@BITWISE
def test_default_arrays_and_numeric_records_have_the_golden_digests(tmp_path):
    golden = json.loads(DIGESTS_GOLDEN.read_text(encoding="utf-8"))
    assert golden["ref_commit"].startswith("c16ca4c")
    base = RD.write_inputs(tmp_path)
    code, run = RD.run_arm(base, tmp_path/"run_default", ())
    assert code == 0
    assert SD.run_digests(run) == golden["digests"]["runs"]["default"]


def test_solver_config_record_and_other_entry_points_are_unchanged():
    assert not set(WST_SOLVER_KEYS) & set(SolverConfig().record())
    assert SolverConfig().start_level == 0
    from bayesreg.learn import cli
    base = ["--out", "o", "--pairs-root", "p", "--eta0-json", "e"]
    config = cli.solver_config(cli.parse_args(base), syn.cube_grid(9))
    assert config.start_level == 0 and not set(WST_SOLVER_KEYS) & set(config.record())
    with pytest.raises(SystemExit):
        cli.parse_args([*base, "--init-field", "x"])
    probe = load("wst_probe", "experiments/eta/probe.py")
    with pytest.raises(SystemExit):
        probe.parse_args(["--pair", "s/p", "--init-field", "x"])


def test_status_vocabulary():
    assert IF.INIT_FIELD_REJECTED == INIT_FIELD_REJECTED and "收敛" not in IF.INIT_FIELD_REJECTED_WORDING
    assert INIT_FIELD_REJECTED not in RUN_STATUSES and INIT_FIELD_REJECTED not in RUN_WORDING  # default vocabulary unchanged
    assert run_pair.EXIT[INIT_FIELD_REJECTED] == 5
    assert ROLE_ORACLE_SEEDED in FIELD_ROLES and ROLE_ORACLE_SEEDED not in COMPARABLE_ROLES


# ---- path only: the same entry state gives the same native level ---------------------------------------------------------

def test_seeded_native_level_is_the_ordinary_native_level_bit_for_bit():
    native, levels, _ = syn.four_levels(26)
    tissue = levels[3].tissue
    eta = eta_temporary(tissue, levels[3].mask, native)
    caps, rounds = Caps(3, 60), (1, 1, 1, 2)
    jds.clear_cell_bounds_cache()
    backend = NumpyBackend()
    ordinary = solve_multilevel(backend, levels, eta=eta, sigma=GaussianMStep(), config=SolverConfig(rounds=rounds, caps=caps))
    assert ordinary.native and ordinary.transfers[-1]["to_level"] == 3 and ordinary.transfers[-1]["accepted_k"] == 0
    jds.clear_cell_bounds_cache()
    upto2 = solve_multilevel(NumpyBackend(), levels, eta=eta, sigma=GaussianMStep(),
                             config=SolverConfig(rounds=rounds, caps=caps, max_levels=3))
    entry = prolong_displacement(upto2.commit.u, levels[2].grid, levels[3].grid)
    jds.clear_cell_bounds_cache()
    seed = InitSeed(np.asarray(entry), dict(grid_id=native.grid_id, oracle=False, path="test"))
    seeded = solve_multilevel(NumpyBackend(), levels, eta=eta, sigma=GaussianMStep(), init_seed=seed,
                              config=SolverConfig(rounds=rounds, caps=caps, start_level=3))
    assert [r["level"] for r in seeded.rounds] == [3, 3] and seeded.transfers == []
    assert seeded.init_field["accepted_k"] == 0 and seeded.init_field["levels_skipped"] == [0, 1, 2]
    assert [e["level"] for e in seeded.timing["levels"]] == [3]
    rows = [r for r in ordinary.rounds if r["level"] == 3]
    assert len(rows) == len(seeded.rounds)
    for a, b in zip(rows, seeded.rounds):
        for key in ("round", "sigma2_used", "G0", "T_B", "status", "reason", "accepted", "commits", "GF_B", "F_B_start",
                    "F_B_end", "F_B_trace", "GF_B_trace", "state_hash", "min_jd_lower_bound", "sigma2_post", "final"):
            assert a[key] == b[key], key
    assert ordinary.status == seeded.status and ordinary.sigma_post == seeded.sigma_post
    assert ordinary.commit.u.tobytes() == seeded.commit.u.tobytes()


def test_sigma_is_the_m_step_at_the_entry_state_and_g0_is_at_u_zero():
    native, levels, _ = syn.four_levels(26)
    eta = eta_temporary(levels[3].tissue, levels[3].mask, native)
    u = np.asarray(syn.solenoidal_displacement(native, .8))
    seed = InitSeed(u, dict(grid_id=native.grid_id, oracle=False, path="test"))
    result = solve_multilevel(NumpyBackend(), levels, eta=eta, sigma=GaussianMStep(), init_seed=seed,
                              config=SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(1, 60), start_level=3))
    backend = NumpyBackend()
    template = backend.prepare(_template(levels[3], eta, native.omega, 8192, 1))
    sigma2 = GaussianMStep().level_start(template, backend.evaluate(template, u))
    row = result.rounds[0]
    assert row["sigma2_used"] == np.asarray(sigma2).tolist()
    assert result.sigma["trace"][0]["kind"] == "level_start" and result.sigma["trace"][0]["level"] == 3
    state = backend.prepare(template.replace(sigma2=np.asarray(sigma2)))
    g0 = g0_of(backend, state)
    assert row["G0"] == g0 and row["T_B"] == threshold(g0)


def test_solver_config_refusals():
    from jd_curl.validation import InputValidationError
    for kwargs in (dict(start_level=2), dict(start_level=1), dict(start_level=3, max_levels=3)):
        with pytest.raises(InputValidationError):
            SolverConfig(**kwargs)
    native, levels, _ = syn.four_levels(26)
    eta = eta_temporary(levels[3].tissue, levels[3].mask, native)
    seed = InitSeed(np.zeros(native.shape+(3,)), dict(grid_id=native.grid_id, oracle=False))
    with pytest.raises(InputValidationError, match="go together"):
        solve_multilevel(NumpyBackend(), levels, eta=eta, sigma=FixedSigma(np.ones(1)), config=SolverConfig(start_level=3))
    with pytest.raises(InputValidationError, match="go together"):
        solve_multilevel(NumpyBackend(), levels, eta=eta, sigma=FixedSigma(np.ones(1)), init_seed=seed)
    wrong = InitSeed(seed.u, dict(grid_id="other", oracle=False))
    with pytest.raises(InputValidationError, match="native grid"):
        solve_multilevel(NumpyBackend(), levels, eta=eta, sigma=FixedSigma(np.ones(1)), init_seed=wrong,
                         config=SolverConfig(start_level=3))


# ---- a seeded run end to end ---------------------------------------------------------------------------------------------

def check_records(out, *, oracle, source_format, arm):
    """Every record of the C16 WST item."""
    status, config, prov, hyper = (read(out/name) for name in ("status.json", "config.json", "provenance.json", "hyper.json"))
    qc, timing, cert = read(out/"qc.json"), read(out/"timing.json"), read(out/"cert.json")
    assert config["args"]["start_level"] == 3 and config["args"]["init_field_oracle_ok"] is oracle
    assert list(config["args"])[-3:] == list(WST_ARG_KEYS)
    for doc in (config["solver"], prov["solver"], hyper["algorithm"]):
        assert doc["start_level"] == 3 and doc["levels_skipped"] == [0, 1, 2] and "levels_skipped_reason" in doc
    for record in (config["init_field"], prov["init_field"], hyper["init_field"]):
        assert record["oracle"] is oracle and record["source_format"] == source_format and record["arm"] == arm
        assert record["contract"].startswith("C12-4a") and record["state_hash"] and record["file_sha256"]
    assert prov["switches"]["init_field"] == prov["init_field"]["path"] and prov["switches"]["start_level"] == 3
    assert prov["switches"]["init_field_oracle"] is oracle and prov["switches"]["preoptimizer"] is False
    entry = hyper["init_field"]
    assert entry["accepted_k"] == len(entry["entry_rows"])-1 and entry["scale"] == entry["entry_rows"][-1]["scale"]
    assert entry["levels_skipped"] == [0, 1, 2] and "levels 0..2 are not run" in entry["levels_skipped_reason"]
    assert entry["entry_rows"][-1]["cert"]["status"] == "certified_feasible"
    assert qc["init_field_oracle"] is oracle
    for doc in (status["init_field"], timing["init_field"], cert["init_field"]):
        assert doc["oracle"] is oracle and doc["arm"] == arm and doc["entry_state_hash"] == entry["entry_state_hash"]
    assert timing["init_field_load"] >= 0
    with np.load(out/"final_native.npz") as z:
        assert json.loads(str(z["init_field"]))["oracle"] is oracle
    with np.load(out/"disp.npz") as z:
        assert json.loads(str(z["meta"]))["init_field"]["oracle"] is oracle
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    assert rows[0]["kind"] == "init_field" and rows[0]["entry_state_hash"] == entry["entry_state_hash"]
    assert not [r for r in rows if r["kind"] == "transfer"] and {r["level"] for r in rows if r["kind"] == "round"} == {3}
    assert hyper["transfers"] == [] and [e["level"] for e in timing["solve_levels"]["levels"]] == [3]


def test_seed_from_a_run_pair_directory(world):
    out = world.seeded("seeded_bayesreg", world.source_seed, *COMMON)
    status = read(out/"status.json")
    assert status["status"] == CAP_REACHED and status["native"] and status["export"]["certificate_passed"] is True
    check_records(out, oracle=False, source_format=IF.FORMAT_BAYESREG, arm="tissue")
    hyper = read(out/"hyper.json")
    assert hyper["init_field"]["accepted_k"] == 0 and hyper["init_field"]["entry_state_hash"] == state_hash(world.u)
    assert hyper["init_field"]["source_certificate"]["status"] == "certified_feasible"
    assert hyper["rounds"][0]["accepted"] == 2 and hyper["rounds"][0]["F_B_end"] < hyper["rounds"][0]["F_B_start"]
    assert read(out/"cert.json")["status"] == "certified_feasible"
    info = describe_run(out)
    assert info.field_role == ROLE_NATIVE and info.extra["init_field"]["oracle"] is False
    assert info.extra["init_field"]["source_format"] == IF.FORMAT_BAYESREG


def test_seed_from_a_frozen_directory(world):
    seed = write_frozen(world, "ok")
    out = world.seeded("seeded_frozen", seed, *COMMON)
    assert read(out/"status.json")["status"] == CAP_REACHED
    check_records(out, oracle=False, source_format=IF.FORMAT_FROZEN, arm="A0")
    record = read(out/"hyper.json")["init_field"]
    assert record["source_pre_optimizer"] == "fp32-v1:synthetic" and record["source_role"] == "formal_native_outer_commit"
    assert record["source_stage"] == "D0-A0" and record["file_sha256"] == file_sha256(seed)
    assert record["source_pair_sha256"] == read(out/"provenance.json")["pair"]["pair_sha256"]


def test_recoloured_non_oracle_arm_binds_through_source_pair_sha256(world):
    seed = write_frozen(world, "a1ph", stage="D0-A1Ph", pre=dict(arm="A1Ph", oracle=False))
    out = world.seeded("seeded_a1ph", seed, *COMMON)
    check_records(out, oracle=False, source_format=IF.FORMAT_FROZEN, arm="A1Ph")


def folded(world, amplitude):
    """A single interior node moved by -amplitude along z: the corner J of the cell below it is 1 - amplitude."""
    u = np.zeros_like(world.u)
    u[13, 13, 13, 2] = -amplitude
    return u


def test_entry_back_off_with_the_real_certificate(world):
    u = folded(world, 1.05)                        # s = 1: J = -.05 at a corner; s = .9: J = .055
    seed = write_frozen(world, "backoff", u)
    out = world.seeded("seeded_backoff", seed, *COMMON)
    record = read(out/"hyper.json")["init_field"]
    assert record["accepted_k"] == 1 and record["scale"] == .9
    assert record["entry_state_hash"] == state_hash(.9*u)
    first, second = record["entry_rows"]
    assert first["cert"]["status"] == "infeasible" and second["cert"]["status"] == "certified_feasible"
    assert read(out/"status.json")["init_field"]["accepted_k"] == 1


def test_all_seven_rejected_is_a_run_failure_without_a_fallback(world):
    seed = write_frozen(world, "folded", folded(world, 4.))     # .9^6 x 4 > 1: every scaled seed folds
    out = world.seeded("seeded_folded", seed, *COMMON, expect=2)
    status, hyper = read(out/"status.json"), read(out/"hyper.json")
    assert status["status"] == INITIALIZATION_FAILED and status["exit_code"] == 2 and not status["native"]
    assert status["init_field"]["accepted_k"] is None and status["init_field"]["entry_state_hash"] is None
    record = hyper["init_field"]
    assert len(record["entry_rows"]) == 7 and [r["k"] for r in record["entry_rows"]] == list(range(7))
    assert all(r["cert"]["status"] == "infeasible" for r in record["entry_rows"])
    assert hyper["rounds"] == [] and "u = 0 is not used" in status["reason"]
    for name in ("final_native.npz", "disp.npz", "cert.json"):
        assert not (out/name).exists(), name
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["init_field"]            # no candidate, no state at u = 0


def test_an_exception_in_the_solve_keeps_the_seed_in_status(world, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(run_pair, "solve_multilevel", boom)
    out = world.seeded("seeded_exception", world.source_seed, *COMMON, expect=1)
    status = read(out/"status.json")
    assert status["status"] == "exception" and status["init_field"]["oracle"] is False


# ---- refusals of the seed ------------------------------------------------------------------------------------------------

def refused(world, seed, fragment, *options):
    out = world.seeded(world.fresh("refused"), seed, *COMMON, *options, expect=5)
    status = read(out/"status.json")
    assert status["status"] == INIT_FIELD_REJECTED and status["exit_code"] == 5 and fragment in status["reason"], status["reason"]
    for name in ("cmd.txt", "git.txt", "env.txt", "log.txt", "timing.json"):
        assert (out/name).is_file(), name
    for name in ("config.json", "provenance.json", "hyper.json", "log.jsonl", "final_native.npz", "disp.npz", "data.txt"):
        assert not (out/name).exists(), name


def test_refused_other_pair_in_a_frozen_directory(world):
    name = f"{DEV[1]}__{DEV[2]}"
    refused(world, write_frozen(world, "otherpair", pair=name), "is not this pair")


def test_refused_other_subjects(world):
    refused(world, write_frozen(world, "othersubj", moving=DEV[3]), "subjects")


def test_refused_other_pair_sha256(world):
    refused(world, write_frozen(world, "othersha", pair_sha="0"*64), "pair sha256")


def test_refused_other_grid_id(world):
    refused(world, write_frozen(world, "othergrid", grid_id="0"*64), "grid_id")


def test_refused_state_hash_and_file_sha(world):
    refused(world, write_frozen(world, "badhash", recorded_hash="0"*64), "state_hash")
    refused(world, write_frozen(world, "badsha", recorded_file_sha="0"*64), "sha256")


@pytest.mark.parametrize("case,fragment", [("shape", "shape"), ("float32", "float64"), ("nan", "non-finite"), ("face", "six faces")])
def test_refused_bad_arrays(world, case, fragment):
    u = world.u.copy()
    if case == "shape":
        u = u[:-1]
    elif case == "float32":
        u = u.astype(np.float32)
    elif case == "nan":
        u[3, 3, 3, 0] = np.nan
    else:
        u[0, 3, 3, 0] = .1
    refused(world, write_frozen(world, f"array_{case}", u, recorded_hash="ignored"), fragment)


def test_refused_unreadable_files(world):
    seed = write_frozen(world, "corrupt_npz")
    seed.write_bytes(b"not a zip")
    (seed.parent/"final_native.json").write_text(json.dumps(dict(
        json.loads((seed.parent/"final_native.json").read_text()), sha256=file_sha256(seed))))
    refused(world, seed, "unreadable")
    seed = write_frozen(world, "corrupt_json")
    (seed.parent/"final_native.json").write_text("[1, 2]")
    refused(world, seed, "a JSON object")
    seed = write_frozen(world, "corrupt_protocol")
    (seed.parent.parent/"protocol.json").write_text("{not json")
    refused(world, seed, "unreadable")


def test_refused_protocol_and_manifest_mismatch(world):
    seed = write_frozen(world, "wrongtask", task_name="other_task")
    refused(world, seed, "task.name")
    seed = write_frozen(world, "wrongmanifest")
    (seed.parent.parent/"inputs"/world.pair_name/"manifest.json").write_text("{}")
    refused(world, seed, "input_sha256")


def test_refused_stage_without_provenance_and_stage_arm_disagreement(world):
    refused(world, write_frozen(world, "stageA1", stage="D0-A1"), "provenance unknown")
    refused(world, write_frozen(world, "stagearm", stage="D0-A0", pre=dict(arm="A2", oracle=False)), "disagree")
    refused(world, write_frozen(world, "nooracle", stage="D0-A2", pre=dict(arm="A2", oracle="no")), "boolean oracle")


def test_refused_unknown_directory_and_file_name(world):
    bare = world.root/"bare"
    bare.mkdir()
    np.savez(bare/"final_native.npz", u=world.u)
    refused(world, bare/"final_native.npz", "not recognized")
    other = world.root/"other_name.npz"
    np.savez(other, u=world.u)
    refused(world, other, "final_native.npz")


def copy_source(world, name, *, config=None, provenance=None, cert=None, npz=None):
    """The source run's record files in a new directory, each changed by the given function of its document."""
    directory = world.root/f"copied_{name}"
    directory.mkdir()
    for file_name, change in (("config.json", config), ("provenance.json", provenance), ("cert.json", cert)):
        document = read(world.source/file_name)
        if change is not None:
            change(document)
        (directory/file_name).write_text(json.dumps(document))
    with np.load(world.source_seed) as z:
        members = {k: z[k] for k in z.files}
    if npz is not None:
        npz(members)
    np.savez(directory/"final_native.npz", **members)
    return directory/"final_native.npz"


def test_refused_bindings_of_a_run_pair_directory(world):
    other = f"stage_synthetic_v1/{DEV[1]}__{DEV[2]}"

    def set_key(document, key, value):
        document[key] = value
    refused(world, copy_source(world, "pair", config=lambda d: d["args"].update(pair=other)), "is not this pair")
    refused(world, copy_source(world, "subj", provenance=lambda d: d["pair"].update(fixed_subject=DEV[4])), "subjects")
    refused(world, copy_source(world, "sha", provenance=lambda d: d["pair"].update(pair_sha256="0"*64)), "pair sha256")
    refused(world, copy_source(world, "cert", cert=lambda d: set_key(d, "state_hash", "0"*64)), "state_hash")
    refused(world, copy_source(world, "stored", npz=lambda m: m.update(state_hash=np.array("0"*64))), "state_hash")
    refused(world, copy_source(world, "grid", npz=lambda m: m.update(grid=np.array(json.dumps(dict(grid_id="0"*64))))), "grid_id")
    refused(world, copy_source(world, "member", npz=lambda m: m.pop("status")), "lacks")
    refused(world, copy_source(world, "layer", provenance=lambda d: d["channels"].pop("tissue")), "provenance unknown")
    refused(world, copy_source(world, "seen", provenance=lambda d: d["channels"]["tissue"]["fixed"]["accepted"].update(
        test_labels_seen=True)), "--init-field-oracle-ok")
    refused(world, copy_source(world, "upstream", provenance=lambda d: set_key(d, "init_field", dict(oracle=True))),
            "--init-field-oracle-ok")


# ---- oracle --------------------------------------------------------------------------------------------------------------

def test_oracle_seed_is_refused_without_the_flag(world):
    for arm in ("A1", "A3"):
        refused(world, write_frozen(world, f"oracle_{arm}", stage=f"D0-{arm}", pre=dict(arm=arm, oracle=True)),
                "--init-field-oracle-ok")


def test_truth_label_arm_that_says_no_oracle_is_refused(world):
    refused(world, write_frozen(world, "lying", stage="D0-A1", pre=dict(arm="A1", oracle=False)), "inconsistent")
    # also with the flag: the record contradicts itself
    refused(world, write_frozen(world, "lying2", stage="D0-A1", pre=dict(arm="A1", oracle=False)), "inconsistent",
            "--init-field-oracle-ok")


def test_oracle_seed_with_the_flag_is_flagged_everywhere_and_not_comparable(world):
    seed = write_frozen(world, "oracle_ok", stage="D0-A1", pre=dict(arm="A1", oracle=True))
    out = world.seeded("seeded_oracle", seed, *COMMON, "--init-field-oracle-ok")
    check_records(out, oracle=True, source_format=IF.FORMAT_FROZEN, arm="A1")
    info = describe_run(out)
    assert info.field_role == ROLE_ORACLE_SEEDED and ROLE_ORACLE_SEEDED not in COMPARABLE_ROLES
    # transitive: a run seeded by this one is an oracle seed too
    refused(world, out/"final_native.npz", "--init-field-oracle-ok")
    again = world.seeded("seeded_oracle2", out/"final_native.npz", *COMMON, "--init-field-oracle-ok")
    record = read(again/"hyper.json")["init_field"]
    assert record["oracle"] is True and record["source_format"] == IF.FORMAT_BAYESREG
    assert record["source_init_field"]["oracle"] is True and record["source_init_field"]["arm"] == "A1"


def test_flag_with_a_non_oracle_seed_marks_nothing_as_oracle(world):
    out = world.seeded("seeded_flag_clean", world.source_seed, *COMMON, "--init-field-oracle-ok")
    assert read(out/"status.json")["init_field"]["oracle"] is False and read(out/"qc.json")["init_field_oracle"] is False
    assert read(out/"config.json")["args"]["init_field_oracle_ok"] is True
    assert describe_run(out).field_role == ROLE_NATIVE


@pytest.mark.parametrize("change,oracle", [("seen", True), ("marker", True), ("missing", None)])
def test_run_pair_source_layer_provenance(world, change, oracle):
    """A hand-made source record: test_labels_seen / oracle count as oracle; no accepted record is refused."""
    prov = read(world.source/"provenance.json")
    accepted = prov["channels"]["tissue"]["moving"]["accepted"]
    if change == "seen":
        accepted["test_labels_seen"] = True
    elif change == "marker":
        accepted["oracle"] = True
    else:
        del prov["channels"]["tissue"]["moving"]["accepted"]
    if oracle:
        assert IF._bayesreg_oracle(prov) is True
    else:
        with pytest.raises(IF.InitFieldRejected, match="provenance unknown"):
            IF._bayesreg_oracle(prov)
    assert IF._bayesreg_oracle(read(world.source/"provenance.json")) is False


# ---- label firewall ------------------------------------------------------------------------------------------------------

def test_the_source_manifest_is_read_without_its_label_keys(world):
    seed = write_frozen(world, "firewall", stage="D0-A1", pre=dict(arm="A1", oracle=True))
    manifest = IF._read_manifest(seed.parent.parent/"inputs"/world.pair_name/"manifest.json")
    assert "baseline_label_metrics" not in manifest and "labels_policy" not in manifest
    assert "colour" not in manifest["preprocessing_d0"] and manifest["preprocessing_d0"]["oracle"] is True


def test_warped_is_never_read(world, monkeypatch):
    seed = write_frozen(world, "warped")
    pair = _pair_input(world)
    real = np.load

    class Guard:
        def __init__(self, path):
            self.npz = real(path, allow_pickle=False)
            self.files = self.npz.files

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.npz.close()

        def __getitem__(self, key):
            assert key != "warped", "the frozen run's warped image must not be read"
            return self.npz[key]
    monkeypatch.setattr(IF.np, "load", lambda path, allow_pickle=False: Guard(path))
    loaded = IF.load_init_field(seed, pair=pair, pair_name=world.pair, oracle_ok=False)
    assert loaded.record["arm"] == "A0"


def _pair_input(world):
    from bayesreg.io import load_pair
    allowlist = world.base[world.base.index("--allowlist")+1]
    return load_pair(world.pair_dir/"pair.npz", allowlist=allowlist)


# ---- combinations --------------------------------------------------------------------------------------------------------

def test_per_level_caps_only_a3_acts(world):
    out = world.seeded("seeded_caps", world.source_seed, "--rounds", "1,1,1,1", "--max-accepted-per-level", "9,9,9,1")
    hyper = read(out/"hyper.json")
    assert [r["level"] for r in hyper["rounds"]] == [3] and hyper["rounds"][0]["accepted"] == 1
    assert hyper["rounds"][0]["cap_accepted"] == 1
    assert read(out/"config.json")["solver"]["max_accepted_per_level"] == [9, 9, 9, 1]


def test_wall_budget_large_and_used_up(world):
    large = world.seeded("seeded_wall", world.source_seed, *COMMON, "--wall-seconds", "1000000")
    assert read(large/"status.json")["status"] == CAP_REACHED
    tiny = world.seeded("seeded_wall_tiny", world.source_seed, *COMMON, "--wall-seconds", "1e-6")
    status, hyper = read(tiny/"status.json"), read(tiny/"hyper.json")
    assert status["status"] == BUDGET_EXHAUSTED and status["exported"] == state_hash(world.u)
    assert hyper["rounds"][-1]["commits"] == 0 and (tiny/"disp.npz").is_file()
    assert describe_run(tiny).field_role == ROLE_HANDOVER


def test_sigma_domain_support_shared_uses_the_entry_state(world):
    out = world.seeded("seeded_support", world.source_seed, *COMMON, "--sigma-domain", "support_shared")
    hyper = read(out/"hyper.json")
    assert [s["level"] for s in hyper["support_levels"]] == [3]
    assert hyper["support_levels"][0]["seed_state_hash"] == hyper["init_field"]["entry_state_hash"]
    assert read(out/"status.json")["status"] == CAP_REACHED


@pytest.mark.parametrize("options", [("--jd-cell-estimate", "gauss27+corners"), ("--jd-cell-estimate", "certificate_bernstein"),
                                     ("--metric-jd-scaling", "off"), ("--logvol-points", "gauss27")])
def test_other_path_and_model_switches_run(world, options):
    out = world.seeded(world.fresh("seeded_option"), world.source_seed, *COMMON, *options)
    assert read(out/"status.json")["status"] == CAP_REACHED
    assert read(out/"hyper.json")["init_field"]["accepted_k"] == 0


# ---- profile objective and checkpoints (C12-4a item 8) --------------------------------------------------------------------

def test_profile_of_a_seeded_run_entry_and_end(world):
    out = world.seeded("seeded_profile", world.source_seed, *COMMON)
    hyper = read(out/"hyper.json")
    profile = hyper["profile"]
    assert profile["n_checkpoints"] == 0 and profile["errors"] == [] and profile["accepted"] == 2 and profile["config"] is None
    assert read(out/"status.json")["n_checkpoints"] == 0
    entry, end = profile["entry"], profile["end"]
    assert entry["state_hash"] == hyper["init_field"]["entry_state_hash"] == state_hash(world.u)
    assert end["state_hash"] == read(out/"status.json")["exported"] and end["accepted"] == 2 and end["rms_from_entry_mm"] > 0
    row = hyper["rounds"][-1]
    # the M step: sigma at the entry state is the level start; at the end state the round's sigma_post
    assert entry["F_B"] == pytest.approx(row["F_B_start"], rel=1e-12)
    assert end["F_B"] == pytest.approx(row["F_B_end"], rel=1e-12)
    assert [entry["sigma2_hat"][f"c{c}"] for c in range(4)] == pytest.approx(row["sigma2_used"], rel=1e-10)
    assert [end["sigma2_hat"][f"c{c}"] for c in range(4)] == pytest.approx(row["sigma2_post"], rel=1e-10)
    for record in (entry, end):
        n = record["N_q"]
        value = sum(record["blocks"].values())+sum((n/2+1)*np.log(g["S"]/(n+2))+(n+2)/2 for g in record["groups"])
        assert record["L"] == pytest.approx(value, rel=1e-12) and [g["N"] for g in record["groups"]] == [n]*4
    assert end["L"] < entry["L"]                               # two accepted steps with sigma_used, then L at its minimum in sigma
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    assert [r["at"] for r in rows if r["kind"] == "profile"] == ["entry"]
    assert not (out/"checkpoints").exists()


def test_checkpoints_every_accepted_and_the_path_is_unchanged(world):
    plain = world.seeded("seeded_plain_6", world.source_seed, "--rounds", "1,1,1,1", "--max-accepted", "6")
    out = world.seeded("seeded_ckpt", world.source_seed, "--rounds", "1,1,1,1", "--max-accepted", "6", "--checkpoint-accepted", "2")
    for run in (plain, out):
        assert read(run/"status.json")["status"] == CAP_REACHED
    assert read(out/"status.json")["exported"] == read(plain/"status.json")["exported"]      # the same states
    a, b = read(plain/"hyper.json")["rounds"][0], read(out/"hyper.json")["rounds"][0]
    assert a["F_B_trace"] == b["F_B_trace"] and a["GF_B_trace"] == b["GF_B_trace"]
    profile = read(out/"hyper.json")["profile"]
    assert profile["config"] == dict(every_accepted=2, every_seconds=None) and profile["n_checkpoints"] == 3
    assert read(out/"status.json")["n_checkpoints"] == 3
    assert read(out/"config.json")["solver"]["checkpoint"] == dict(every_accepted=2, every_seconds=None)
    assert read(out/"config.json")["args"]["checkpoint_accepted"] == 2
    assert [c["accepted"] for c in profile["checkpoints"]] == [2, 4, 6] and [c["k"] for c in profile["checkpoints"]] == [0, 1, 2]
    trace = a["F_B_trace"]
    for c in profile["checkpoints"]:
        assert c["F_B"] == trace[c["accepted"]] and c["cert_status"] == "certified_feasible" and c["L"] and c["rms_from_entry_mm"] > 0
        directory = out/"checkpoints"/f"ckpt_{c['k']:03d}"
        with np.load(directory/"disp.npz") as z:
            meta = json.loads(str(z["meta"]))
            assert z["d"].shape == (3, 26, 26, 26) and meta["u_state_hash"] == c["state_hash"]
            assert meta["init_field"]["state_hash"] == state_hash(world.u)
        assert describe_run(directory, pair=world.pair_name).kind == "disp"
    with np.load(out/"final_native.npz") as z:
        assert profile["checkpoints"][-1]["state_hash"] == str(z["state_hash"])           # the last accepted step is exported
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    assert sum(r["kind"] == "checkpoint" for r in rows) == 3
    # the plain seeded run and a run without a seed carry no checkpoint rows; records stay under the switch
    assert not (plain/"checkpoints").exists()


def test_max_accepted_zero_gives_l_of_the_seed_without_a_step(world):
    """C12-4a item 8: L of a given field under an arm's eta (the M2V1 field of the plan's rule)."""
    out = world.seeded("seeded_l_only", world.source_seed, "--rounds", "1,1,1,1", "--max-accepted", "0")
    status, hyper = read(out/"status.json"), read(out/"hyper.json")
    assert status["status"] == CAP_REACHED and status["exported"] == state_hash(world.u)
    profile = hyper["profile"]
    assert profile["accepted"] == 0 and profile["entry"]["L"] == pytest.approx(profile["end"]["L"], rel=1e-12)
    assert profile["end"]["rms_from_entry_mm"] == 0. and describe_run(out).field_role == ROLE_HANDOVER


def test_checkpoints_without_a_seed_are_on_the_native_level_only(world):
    out = world.run("ckpt_noseed", *COMMON, "--checkpoint-accepted", "1")
    profile = read(out/"hyper.json")["profile"]
    assert profile["accepted"] == 2 and profile["n_checkpoints"] == 2 and {c["level"] for c in profile["checkpoints"]} == {3}
    assert profile["entry"]["level"] == 3 and "init_field" not in read(out/"status.json")


def test_checkpoint_seconds_with_an_injected_clock(world):
    class Clock:
        t = 0.

        def __call__(self):
            return self.t

    class Point:
        energies = dict(channel=(1., 2.), diffusion=.5, logvol=.25, curl=.125)
        energy = 3.875

    class State:
        sigma2 = np.array([1., 2.])

    class Cert:
        status, min_jd_lower_bound, state_hash = "certified_feasible", .5, "h"

    class Commit:
        u, cert = np.zeros((26, 26, 26, 3)), Cert()

    from bayesreg.solve.checkpoints import CheckpointConfig, ProfileRecorder

    class Backend:
        @staticmethod
        def evaluate(state, u):
            return Point()
    import tempfile
    grid = syn.cube_grid(26)
    clock = Clock()
    with tempfile.TemporaryDirectory() as tmp:
        recorder = ProfileRecorder(grid=grid, directory=Path(tmp)/"c", config=CheckpointConfig(every_seconds=10.), clock=clock)
        recorder.begin_round(Backend, State, Commit, level=3, round_=0, n_q_value=8., mask=np.ones(grid.shape, dtype=bool))
        recorder.observe(Commit, Point)                       # 0 s: not due
        assert recorder.checkpoints == []
        for now, expected in ((9.9, 0), (10., 1), (15., 1), (19.9, 1), (20., 2)):
            clock.t = now
            recorder.observe(Commit, Point)
            assert len(recorder.checkpoints) == expected, now
        record = recorder.record()
    assert record["accepted"] == 6 and [c["accepted"] for c in record["checkpoints"]] == [3, 6]
    entry = record["entry"]
    # S_c = 2 sigma_c^2 E_c; L by hand
    assert entry["S_c"] == [2., 8.] and entry["sigma2_hat"] == dict(c0=.2, c1=.8)
    assert entry["L"] == pytest.approx(.875+5*np.log(.2)+5+5*np.log(.8)+5)


def test_variance_groups_of_the_support_domains():
    from bayesreg.solve.checkpoints import variance_groups

    class Level:
        nonempty = (1, 3)
        n_c = (40., 0., 20.)

        def __init__(self, domain):
            self.sigma_domain = domain

        @staticmethod
        def n_s():
            return 60.
    sums = [10., 4., 0., 2.]
    assert variance_groups(None, sums, 100.) == [("c0", 100., 10.), ("c1", 100., 4.), ("c2", 100., 0.), ("c3", 100., 2.)]
    assert variance_groups(None, sums, 100., Level("support")) == [("I", 100., 10.), ("c1", 40., 4.), ("c3", 20., 2.)]
    assert variance_groups(None, sums, 100., Level("support_shared")) == [("I", 100., 10.), ("S", 60., 6.)]


def test_support_shared_profile_of_a_seeded_run(world):
    out = world.seeded("seeded_support_profile", world.source_seed, *COMMON, "--sigma-domain", "support_shared")
    profile = read(out/"hyper.json")["profile"]
    names = [g["name"] for g in profile["entry"]["groups"]]
    assert names[0] == "I" and set(names) <= {"I", "S"}
    row = read(out/"hyper.json")["rounds"][0]
    assert profile["entry"]["sigma2_hat"]["I"] == pytest.approx(row["sigma2_used"][0], rel=1e-10)
    # L^v is at its minimum in sigma at the entry: sigma_S^2 of the level start (C11-8), so L cannot rise along the accepted steps
    assert profile["entry"]["sigma2_hat"]["S"] == pytest.approx(row["sigma2_S_used"], rel=1e-10) and profile["end"]["L"] < profile["entry"]["L"]


@pytest.mark.parametrize("extra,fragment", [
    (("--checkpoint-accepted", "2", "--checkpoint-seconds", "5", "--wall-seconds", "100"), "give one of them"),
    (("--checkpoint-accepted", "0"), "positive integer"),
    (("--checkpoint-accepted", "2", "--outer", "b"), "option (a)"),
    (("--checkpoint-accepted", "2", "--inner", "lm"), "option (a)"),
    (("--checkpoint-accepted", "2", "--max-levels", "3"), "option (a)"),
    (("--checkpoint-seconds", "5"), "--wall-seconds"),
    (("--checkpoint-seconds", "50", "--wall-seconds", "100"), "at least 3"),
])
def test_checkpoint_switches_refused_in_parse_args(tmp_path, world, capsys, extra, fragment):
    out = tmp_path/"no_run"
    with pytest.raises(SystemExit) as exc:
        run_pair.parse_args([*world.base, "--out", str(out), *extra])
    assert exc.value.code == 2 and fragment in capsys.readouterr().err
    assert not out.exists()


def test_checkpoint_switches_admitted_and_absent_by_default(world):
    args = run_pair.parse_args([*world.base, "--out", "o", "--checkpoint-seconds", "225", "--wall-seconds", "900"])
    assert run_pair.checkpoint_config(args).every_seconds == 225.
    plain = run_pair.parse_args([*world.base, "--out", "o"])
    assert run_pair.checkpoint_config(plain) is None and not {"checkpoint_accepted", "checkpoint_seconds"} & set(vars(plain))
    assert "checkpoint" not in SolverConfig().record()


@pytest.mark.parametrize("extra,fragment", [
    (("--init-field", "SEED"), "go together"),
    (("--start-level", "3"), "go together"),
    (("--init-field-oracle-ok",), "go together"),
    (("--init-field", "SEED", "--init-field-oracle-ok"), "go together"),
    (("--init-field", "SEED", "--start-level", "2"), "only 3"),
    (("--init-field", "SEED", "--start-level", "0"), "only 3"),
    (("--init-field", "MISSING", "--start-level", "3"), "not a file"),
    (("--init-field", "SEED", "--start-level", "3", "--max-levels", "3"), "--max-levels"),
    (("--init-field", "SEED", "--start-level", "3", "--outer", "b"), "option (a)"),
    (("--init-field", "SEED", "--start-level", "3", "--inner", "lm"), "--inner pg"),
    (("--init-field", "SEED", "--start-level", "3", "--wall-seconds", "9", "--level-time-shares", ".1,.1,.1"),
     "--level-time-shares"),
])
def test_refused_in_parse_args_without_a_run_directory(tmp_path, world, capsys, extra, fragment):
    out = tmp_path/"no_run"
    argv = [("MISSING_FILE" if a == "MISSING" else str(world.source_seed) if a == "SEED" else a) for a in extra]
    with pytest.raises(SystemExit) as exc:
        run_pair.parse_args([*world.base, "--out", str(out), *argv])
    assert exc.value.code == 2 and fragment in capsys.readouterr().err
    assert not out.exists()


def test_inner_pg_explicit_is_admitted(world):
    args = run_pair.parse_args([*world.base, "--out", "o", "--inner", "pg", "--init-field", str(world.source_seed),
                                "--start-level", "3"])
    assert args.start_level == 3 and args.init_field == str(world.source_seed) and args.init_field_oracle_ok is False
    assert run_pair.init_field_options(args) == dict(start_level=3)
    plain = run_pair.parse_args([*world.base, "--out", "o"])
    assert run_pair.init_field_options(plain) == {} and not set(WST_ARG_KEYS) & set(vars(plain))


# ---- WARM: the records after the reviews (C12-4a items 7, 8) ------------------------------------------------------------------

class _Point:
    energy = 2.875

    def __init__(self, channel):
        self.energies = dict(channel=channel, diffusion=.5, logvol=.25, curl=.125)


class _State:
    sigma2 = np.array([1., 2.])


def test_profile_record_with_s_zero_or_not_finite_has_no_l_and_raises_nothing():
    from bayesreg.solve.checkpoints import profile_record
    good = profile_record(_State, _Point((1., 2.)), 8.)
    assert good["L"] == pytest.approx(.875+5*np.log(.2)+5+5*np.log(.8)+5) and "L_undefined" not in good
    zero = profile_record(_State, _Point((0., 2.)), 8.)
    assert zero["L"] is None and zero["L_undefined"] == ["c0"] and zero["sigma2_hat"] == dict(c0=0., c1=.8)
    assert zero["F_B"] == 2.875 and zero["groups"][0] == dict(name="c0", N=8., S=0.)
    wild = profile_record(_State, _Point((float("inf"), 2.)), 8.)
    assert wild["L"] is None and wild["L_undefined"] == ["c0"] and wild["sigma2_hat"]["c0"] is None
    assert wild["S_c"] == [None, 8.] and wild["groups"][0]["S"] is None
    json.dumps(wild, allow_nan=False)                 # hyper.json refuses NaN and infinities


def test_a_failure_of_the_records_lands_in_errors_and_the_solve_does_not_notice(world, monkeypatch):
    from bayesreg.solve import checkpoints as CP
    plain = world.seeded("records_plain", world.source_seed, "--rounds", "1,1,1,1", "--max-accepted", "4")

    def no_disk(*args, **kwargs):
        raise OSError("no space left on device")
    monkeypatch.setattr(CP, "write_disp", no_disk)
    out = world.seeded("records_no_disk", world.source_seed, "--rounds", "1,1,1,1", "--max-accepted", "4",
                       "--checkpoint-accepted", "2")
    monkeypatch.undo()
    assert read(out/"status.json")["status"] == read(plain/"status.json")["status"] == CAP_REACHED
    assert read(out/"status.json")["exit_code"] == 0 and read(out/"status.json")["exported"] == read(plain/"status.json")["exported"]
    a, b = read(plain/"hyper.json")["rounds"][0], read(out/"hyper.json")["rounds"][0]
    assert a["F_B_trace"] == b["F_B_trace"] and a["GF_B_trace"] == b["GF_B_trace"]
    profile = read(out/"hyper.json")["profile"]
    assert profile["n_checkpoints"] == 0 and profile["entry"]["L"] and profile["end"]["L"] and read(out/"status.json")["n_checkpoints"] == 0
    assert [e.split(":")[0] for e in profile["errors"]] == ["entry file", "checkpoint 0", "checkpoint 1"]

    def no_l(*args, **kwargs):
        raise ValueError("math domain error")
    monkeypatch.setattr(CP, "profile_record", no_l)
    broken = world.seeded("records_no_l", world.source_seed, "--rounds", "1,1,1,1", "--max-accepted", "4", "--checkpoint-accepted", "2")
    monkeypatch.undo()
    assert read(broken/"status.json")["exported"] == read(plain/"status.json")["exported"]
    profile = read(broken/"hyper.json")["profile"]
    assert profile["entry"] is None and profile["end"] is None and profile["checkpoints"] == []
    assert [e.split(":")[0] for e in profile["errors"]] == ["entry", "checkpoint 0", "checkpoint 1", "end"]


def test_entry_field_with_checkpoints_only(world):
    out = world.seeded("entry_field", world.source_seed, *COMMON, "--checkpoint-accepted", "1")
    hyper = read(out/"hyper.json")
    profile = hyper["profile"]
    directory = out/"checkpoints"/"entry"
    with np.load(directory/"disp.npz") as z:
        meta = json.loads(str(z["meta"]))
        assert z["d"].shape == (3, 26, 26, 26)
    assert meta["u_state_hash"] == hyper["init_field"]["entry_state_hash"] == state_hash(world.u)
    assert meta["init_field"]["oracle"] is False and meta["init_field"]["accepted_k"] == 0 and meta["init_field"]["scale"] == 1.   # C16 (review WARM_tools M1)
    assert meta["init_field"]["entry_state_hash"] == hyper["init_field"]["entry_state_hash"]
    with np.load(out/"checkpoints"/"ckpt_000"/"disp.npz") as z:
        assert json.loads(str(z["meta"]))["init_field"] == meta["init_field"]
    with np.load(out/"disp.npz") as z:                                                                  # the end's tag is the same
        assert json.loads(str(z["meta"]))["init_field"] == meta["init_field"]
    assert profile["entry"]["directory"] == str(directory) and profile["entry"]["file_state_hash"] == meta["u_state_hash"]
    assert profile["entry"]["write_seconds"] >= 0 and profile["n_checkpoints"] == 2 and read(out/"status.json")["n_checkpoints"] == 2
    assert sorted(p.name for p in (out/"checkpoints").iterdir()) == ["ckpt_000", "ckpt_001", "entry"]
    info = describe_run(directory, pair=world.pair_name)
    assert info.kind == "disp" and info.field_role == "disp_only" and info.extra["init_field"]["state_hash"] == state_hash(world.u)
    without_seed = world.run("entry_field_noseed", *COMMON, "--checkpoint-accepted", "1")
    assert (without_seed/"checkpoints"/"entry"/"disp.npz").is_file()
    assert "init_field" not in describe_run(without_seed/"checkpoints"/"entry", pair=world.pair_name).extra
    with np.load(without_seed/"checkpoints"/"entry"/"disp.npz") as z:
        assert json.loads(str(z["meta"]))["u_state_hash"] == read(without_seed/"hyper.json")["profile"]["entry"]["state_hash"]


def test_the_seed_tag_of_the_entry_and_the_checkpoints_after_a_back_off(world):
    """C16: k, scale and the entry's hash in the disp.npz meta, also when the entry is not the seed (review WARM_tools M1); the seed's own hash stays."""
    u = folded(world, 1.05)
    out = world.seeded("tag_backoff", write_frozen(world, "tag_backoff", u), *COMMON, "--checkpoint-accepted", "1")
    record = read(out/"hyper.json")["init_field"]
    assert record["accepted_k"] == 1 and record["scale"] == .9
    for name in ("entry", "ckpt_000"):
        info = describe_run(out/"checkpoints"/name, pair=world.pair_name)
        tag = info.extra["init_field"]
        assert tag["accepted_k"] == 1 and tag["scale"] == .9 and tag["entry_state_hash"] == state_hash(.9*u) and tag["state_hash"] == state_hash(u)
    with np.load(out/"checkpoints"/"entry"/"disp.npz") as z:
        assert json.loads(str(z["meta"]))["u_state_hash"] == state_hash(.9*u)


def test_checkpoints_of_an_oracle_seed_are_oracle_seeded(world):
    seed = write_frozen(world, "oracle_ckpt", stage="D0-A1", pre=dict(arm="A1", oracle=True))
    out = world.seeded("oracle_ckpt", seed, *COMMON, "--init-field-oracle-ok", "--checkpoint-accepted", "1")
    for name in ("entry", "ckpt_000"):
        info = describe_run(out/"checkpoints"/name, pair=world.pair_name)
        assert info.field_role == ROLE_ORACLE_SEEDED and info.field_role not in COMPARABLE_ROLES and info.extra["init_field"]["arm"] == "A1"


def test_a_disp_directory_without_a_seed_keeps_its_role(tmp_path):
    bare = tmp_path/"baseline"
    bare.mkdir()
    np.savez(bare/"disp.npz", d=np.zeros((3, 4, 4, 4)))
    info = describe_run(bare, pair="p")
    assert info.field_role == "disp_only" and "init_field" not in info.extra
    (bare/"disp.npz").write_bytes(b"not a zip")      # read_disp reports it when the field is loaded
    assert describe_run(bare, pair="p").field_role == "disp_only"


def test_truth_label_arm_a5_that_says_no_oracle_is_refused(world):
    refused(world, write_frozen(world, "lying_a5", stage="D0-A5", pre=dict(arm="A5", oracle=False)), "inconsistent")


def _rows(out):
    return [(r["level"], r["round"], r["F_B_trace"], r["GF_B_trace"], r["state_hash"], r["status"], r["sigma2_used"], r["G0"], r["T_B"])
            for r in read(out/"hyper.json")["rounds"]]


def test_checkpoints_on_the_default_path_are_bit_for_bit_the_default_path(world):
    plain = world.run("default_plain", "--rounds", "1,1,1,1", "--max-accepted", "3")
    out = world.run("default_ckpt", "--rounds", "1,1,1,1", "--max-accepted", "3", "--checkpoint-accepted", "1")
    assert _rows(plain) == _rows(out)
    assert read(plain/"status.json")["exported"] == read(out/"status.json")["exported"]
    with np.load(plain/"final_native.npz") as a, np.load(out/"final_native.npz") as b:
        assert a["u"].tobytes() == b["u"].tobytes()
    with np.load(plain/"disp.npz") as a, np.load(out/"disp.npz") as b:
        assert a["d"].tobytes() == b["d"].tobytes()
    assert "profile" not in read(plain/"hyper.json") and read(out/"hyper.json")["profile"]["n_checkpoints"] == 3
    assert not (plain/"checkpoints").exists()
    assert "checkpoint" not in read(plain/"config.json")["solver"] and "n_checkpoints" not in read(plain/"status.json")


def test_the_recorder_does_not_change_a_seeded_native_level(tmp_path):
    from bayesreg.solve.checkpoints import CheckpointConfig, ProfileRecorder
    native, levels, _ = syn.four_levels(26)
    eta = eta_temporary(levels[3].tissue, levels[3].mask, native)
    u = np.asarray(syn.solenoidal_displacement(native, .8))
    config = dict(rounds=(1, 1, 1, 2), caps=Caps(3, 60), start_level=3)

    def solve(**extra):
        jds.clear_cell_bounds_cache()
        seed = InitSeed(u, dict(grid_id=native.grid_id, oracle=False, path="test"))
        return solve_multilevel(NumpyBackend(), levels, eta=eta, sigma=GaussianMStep(), init_seed=seed,
                                config=SolverConfig(**config, **extra.pop("solver", {})), **extra)
    without = solve()
    recorder = ProfileRecorder(grid=native, directory=tmp_path/"checkpoints", config=CheckpointConfig(every_accepted=1))
    recorded = solve(recorder=recorder, solver=dict(checkpoint=CheckpointConfig(every_accepted=1)))
    assert without.profile is None and recorded.profile["errors"] == [] and recorded.profile["n_checkpoints"] > 0
    assert len(without.rounds) == len(recorded.rounds) == 2
    for a, b in zip(without.rounds, recorded.rounds):
        for key in ("round", "sigma2_used", "G0", "T_B", "status", "reason", "accepted", "commits", "GF_B", "F_B_start", "F_B_end",
                    "F_B_trace", "GF_B_trace", "state_hash", "min_jd_lower_bound", "sigma2_post", "final"):
            assert a[key] == b[key], key
    assert without.status == recorded.status and without.commit.u.tobytes() == recorded.commit.u.tobytes()
    assert without.sigma_post == recorded.sigma_post
    entry = recorded.profile["entry"]
    assert entry["level"] == 3 and entry["round"] == 0 and recorded.profile["end"]["round"] == 1     # the last round is the end


def test_profile_at_a_budget_end_has_its_end(world):
    out = world.seeded("budget_end", world.source_seed, *COMMON, "--wall-seconds", "1e-6")
    assert read(out/"status.json")["status"] == BUDGET_EXHAUSTED
    profile = read(out/"hyper.json")["profile"]
    assert profile["accepted"] == 0 and profile["entry"]["L"] == pytest.approx(profile["end"]["L"], rel=1e-12)
    assert profile["end"]["rms_from_entry_mm"] == 0. and profile["end"]["rms_from_entry_all_mm"] == 0. and profile["errors"] == []


def test_rms_over_the_brain_and_over_all_nodes():
    from bayesreg.solve.checkpoints import ProfileRecorder
    grid = syn.cube_grid(9)
    recorder = ProfileRecorder(grid=grid)
    recorder.u_entry = np.zeros(grid.shape+(3,))
    recorder.mask = np.zeros(grid.shape, dtype=bool)
    recorder.mask[4, 4, 4] = True
    u = np.zeros(grid.shape+(3,))
    u[4, 4, 4] = (3., 0., 4.)
    assert recorder.rms_from_entry(u) == 5.
    assert recorder.rms_from_entry_all(u) == pytest.approx(5./np.sqrt(9**3), rel=1e-15)
