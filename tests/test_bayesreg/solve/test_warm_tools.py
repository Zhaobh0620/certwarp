"""WARM tools (experiments/gate_a/run_warm.py, summarize_warm.py; runs/20261103-warm/plan.md), on synthetic files only (tmp_path; no data/, no unit, no solver, no real
result file): a fake machine emulates the units, run_pair's and bayesreg.eval.cli's arguments go through their real parsers.
- the arguments are X2d's plus the seed and the checkpoints, nothing else; both phases fix 900 s / 225 s (the lead's revision before any run);
- the probe: one run, the entry and each checkpoint evaluated, the verdict (go, stagnation, moved below the bar, a failed run), RMS bars on all nodes with the brain-node value alongside;
  its run is that pair's result: the phase pairs launches the other four, never that pair, and its summary covers all five; refused without a probe, after a stagnant probe, admitted
  after a go or with BZ's words;
- the readings of the plan's table, line by line and at full count only; the bars against the references; L and F_B never across seeds in the text;
- the seeds (hashes, stage, oracle, role), the eta record (X2d's), config.json checks, a refused seed, hard constraint 1 on a checkpoint: STOP; a restart launches nothing and evaluates only
  what is missing; a failed evaluation is kept, not rerun;
- --dry-run prints every command line and writes nothing; the word of the gate 12 incident never reaches a real launch.
"""
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pytest

from .test_pilot_tools import PAIRS, ROOT, SHORTS, FakeSystem, head, load


@pytest.fixture(scope="module")
def warm():
    return load("warm_run_warm", "experiments/gate_a/run_warm.py")


@pytest.fixture(scope="module")
def sw():
    return load("warm_summarize_warm", "experiments/gate_a/summarize_warm.py")


@pytest.fixture(scope="module")
def dry_mod():
    return load("warm_m2v1_dry_inputs", "experiments/m2v1/dry_inputs.py")


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1))
    return path


def labels_doc(seg35=.82, cortex=.85, wm=.91, small=.53, large=.89):
    """seg35_per_label with the groups of the plan; seg4_per_label with cortex and white matter."""
    labels = {str(k): seg35 for k in range(1, 36)}
    labels.update({str(k): small for k in (4, 23, 18, 34, 19, 35)})
    labels.update({str(k): large for k in (2, 21, 1, 20, 5, 24)})
    return dict(seg35_per_label=labels, seg4_per_label={"1": cortex, "2": .9, "3": wm, "4": .85})


def write_seed(seeds, pair, *, stage="D0-A1Ph", arm="A1Ph", oracle=False, role="formal_native_outer_commit", status="certified_feasible"):
    """A frozen configuration C directory in the layout of d0_v1_a1p/runs/A1Ph/<pair> (the seed's npz is only hashed here: the fake machine runs nothing)."""
    run = seeds/pair
    task = run/"pg_weak__four__90s"
    task.mkdir(parents=True)
    np.savez(task/"final_native.npz", u=np.zeros((2, 2, 2, 3)))
    from jd_curl.three_d.nifti import file_sha256
    write_json(task/"final_native.json", dict(path="final_native.npz", role=role, sha256=file_sha256(task/"final_native.npz"),
                                              certificate=dict(status=status, state_hash="a"*64, grid_id="g"*64, min_jd_lower_bound=2e-7)))
    write_json(run/"protocol.json", dict(stage=stage, task=dict(pair=pair, name="pg_weak__four__90s", wall_seconds=90)))
    write_json(run/"inputs"/pair/"manifest.json", dict(pair=pair, preprocessing_d0=dict(arm=arm, oracle=oracle, source_pair_sha256="p"*64,
                                                                                         colour=dict(fixed=dict(c_by_seg4_label={"1": .3}))),
                                                       baseline_label_metrics=dict(seg35=.5)))
    write_json(run/"metrics.json", dict(metrics=dict(seg35=.82, seg4=.886, hd95=2., sdlogj_official=.07, sdlogj_brain=.5, **labels_doc()),
                                        certificate=dict(passed=True, min_jd_lower_bound=2e-7), run_id=f"d0v1_A1Ph_{pair[-8:-4]}"))


def write_x2(dry, pair, short):
    write_json(dry/"x2"/"eval"/"d"/short/f"gax2_d_{short}"/"metrics.json", dict(
        metrics=dict(seg35=.809, seg4=.834, hd95=2.2, sdlogj_official=.064, sdlogj_brain=.45), per_label=labels_doc(seg35=.81, cortex=.72, wm=.84, small=.546, large=.798),
        certificate=dict(passed=True, min_jd_lower_bound=1e-5), run_id=f"gax2_d_{short}"))


@pytest.fixture
def inputs(dry_mod, tmp_path):
    dry = tmp_path/"dry"
    dry_mod.main(["--out", str(dry), "--wall-seconds", "60", "--extra", "", "--code-sha", head()])
    for pair, short in zip(PAIRS, SHORTS):
        write_seed(dry/"seeds", pair)
        write_x2(dry, pair, short)
    write_json(dry/"x2"/"eta"/"eta_d.json", dict(schema="bayesreg.gate_a_x2.v1.eta", alpha=.01, nu=1.37, tau=[1., 1., 1., 1.], channel_source="structure",
                                                 sigma_domain="support_shared", support_threshold=.05))
    return dry


def common(dry):
    return ["--synthetic", "--data-root", str(dry/"data"), "--allowlist", str(dry/"dev_subjects.json"), "--tissue-root", str(dry/"seg"), "--structure-root", str(dry/"structure"),
            "--segmenter-registry", str(dry/"registry"/"segmenters.json"), "--seed-root", str(dry/"seeds"), "--eta-json", str(dry/"x2"/"eta"/"eta_d.json"), "--x2-root", str(dry/"x2")]


class WarmFake(FakeSystem):
    """The units of WARM.  run_pair (real parser): the records of C12-4a item 8 with `accepted` steps, `rms` voxels over all nodes (twice that over the brain nodes) and `checkpoints` fields.  bayesreg.eval.cli (real parser): Dice from `dice`,
    a function of (short, field) giving the measures of that state; `failed_cert`: fields whose certificate does not pass; `eval_rc`: units whose evaluation fails."""

    def __init__(self, *, accepted=12, rms=.2, checkpoints=2, dice=None, failed_cert=(), eval_rc=(), status_override=None, config_change=None, l_end=None, **kwargs):
        super().__init__(**kwargs)
        self.accepted, self.rms, self.checkpoints, self.dice = accepted, rms, checkpoints, dice or (lambda short, field: {})
        self.failed_cert, self.eval_rc, self.status_override, self.config_change, self.l_end = set(failed_cert), set(eval_rc), status_override, config_change, l_end

    def run(self, cmd, record):
        unit, log, exit_file, argv = self.split(cmd)
        argv = [str(a) for a in argv]
        self.units.append(unit)
        self.clock += timedelta(seconds=self.step)
        if "-eval-" in unit:
            rc = self.fake_eval(unit, argv)
        else:
            self.fake_run(unit, argv)
            rc = 137 if unit in self.drop else 0
        if log is not None:
            log.write_text(f"fake {unit}\n")
            exit_file.write_text(f"{rc}\n")
        return rc

    def fake_eval(self, unit, argv):
        from bayesreg.eval import cli
        assert argv[2:4] == ["-m", "bayesreg.eval.cli"]
        args = cli.parse_args(argv[4:])
        target = args.runs[0]
        is_field = target.parent.name == "checkpoints"
        assert (args.pair is not None) == is_field and args.method == "bayesreg_warm_v1" and args.out == "metrics.json"
        if unit in self.eval_rc:
            return 3
        field = target.name if is_field else "final"
        short = target.parent.parent.name if is_field else target.name
        values = dict(seg35=.82, seg4=.886, cortex=.85, white_matter=.91, small=.53, large=.89)
        values.update(self.dice(short, field))
        passed = unit not in self.failed_cert
        path = Path(args.out_dir)/args.run_id/"metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(run_id=args.run_id, field_present=True, field_role="disp_only" if is_field else "native_formal",
                                        metrics=dict(seg35=values["seg35"], seg4=values["seg4"], hd95=2., sdlogj_official=.07, sdlogj_brain=.5),
                                        per_label=labels_doc(values["seg35"], values["cortex"], values["white_matter"], values["small"], values["large"]),
                                        certificate=dict(mode="full", status="certified_feasible" if passed else "infeasible", passed=passed, min_jd_lower_bound=1e-6))))
        args.long.write_text("x\n")
        return 0

    def fake_run(self, unit, argv):
        from bayesreg.cli import run_pair as rp
        assert argv[2:4] == ["-m", "bayesreg.cli.run_pair"]
        args = rp.parse_args(argv[4:])
        run = Path(args.out)
        if unit in self.drop:
            return
        run.mkdir()
        recorded = {k: v for k, v in vars(args).items() if not k.startswith("_")}
        eta = json.loads(Path(args.eta_json).read_text())
        final = json.loads((Path(args.init_field).parent/"final_native.json").read_text())
        seed = dict(file_sha256=final["sha256"], state_hash=final["certificate"]["state_hash"], arm="A1Ph", oracle=False, source_format="frozen_configC")
        solver = dict(sigma_domain=args.sigma_domain, start_level=args.start_level, checkpoint=dict(every_accepted=None, every_seconds=args.checkpoint_seconds))
        config = dict(args=recorded, eta=dict(alpha=eta["alpha"], nu=eta["nu"], tau=eta["tau"]), solver=solver, init_field=seed)
        if self.config_change:
            self.config_change(config)
        (run/"config.json").write_text(json.dumps(config))
        status = dict(status="budget_exhausted", wording="预算收尾", exit_code=0, native=True, exported="f"*64, n_checkpoints=self.checkpoints,
                      wall_budget=dict(wall_seconds=args.wall_seconds), init_field=dict(oracle=False))
        if self.status_override:
            status.update(self.status_override)
        (run/"status.json").write_text(json.dumps(status))
        rows = []
        for k in range(self.checkpoints):
            directory = run/"checkpoints"/f"ckpt_{k:03d}"
            directory.mkdir(parents=True)
            (directory/"disp.npz").write_bytes(b"fake")
            rows.append(dict(k=k, directory=str(directory), level=3, round=0, accepted=(k+1)*2, seconds_from_entry=(k+1)*args.checkpoint_seconds, rms_from_entry_mm=self.rms*2*(k+1)/(self.checkpoints+1),
                             rms_from_entry_all_mm=self.rms*(k+1)/(self.checkpoints+1), state_hash=f"{k}"*64, cert_status="certified_feasible", min_jd_lower_bound=1e-6, L=100.-k, F_B=50.-k))
        (run/"checkpoints"/"entry").mkdir(parents=True, exist_ok=True)
        (run/"checkpoints"/"entry"/"disp.npz").write_bytes(b"fake")
        profile = dict(entry=dict(L=100., F_B=50., state_hash=final["certificate"]["state_hash"], min_jd_lower_bound=2e-7, directory=str(run/"checkpoints"/"entry"), write_seconds=.1),
                       end=dict(L=90. if self.l_end is None else self.l_end, F_B=40., accepted=self.accepted, rms_from_entry_mm=self.rms*2, rms_from_entry_all_mm=self.rms, state_hash="e"*64),
                       accepted=self.accepted, checkpoints=rows, n_checkpoints=self.checkpoints, errors=[])
        native = dict(level=3, round=0, status="budget_exhausted", accepted=self.accepted, cap=500, commits=self.accepted, GF_B=2., T_B=1., GF_B_over_T_B=2., min_jd_lower_bound=1e-6, seconds=250.,
                      counters=dict(candidates=self.accepted+20, rejections=dict(JACOBIAN_VIOLATION=20)), sigma2_used=[.02, .03], sigma2_S_used=.03, F_B_trace=[50., 40.], spacing_xyz=[1., 1., 1.],
                      metric=dict(jd_scaling=dict(mode="affine_v1", cell_estimate="certificate", jd_scaling_refresh=3, min_J_a=1e-6, max_fraction_r_lt_1=.2, computations=[])))
        hyper = dict(rounds=[native], transfers=[], profile=profile, support_levels=[dict(level=3, N_q=1000., N_c_over_N_q=[.05], N_c=[50.], empty_support_channels=[], N_S=60., sigma2_S_start=.03)],
                     init_field=dict(seed, accepted_k=0, scale=1., entry_state_hash=final["certificate"]["state_hash"]))
        (run/"hyper.json").write_text(json.dumps(hyper))


@pytest.fixture
def driver(warm, monkeypatch):
    sha = head()

    def prepare(self):
        self.code, self.code_sha = ROOT, sha
        self.allowlist = self.allowlist or self.o.allowlist
    monkeypatch.setattr(warm.WarmArm, "prepare_code", prepare)
    monkeypatch.setattr(warm.WarmArm, "write_env", lambda self: None)

    def go(root, system, dry, phase, *extra):
        return warm.main(["--phase", phase, "--root", str(root), *common(dry), "--wall-seconds", "60", "--checkpoint-seconds", "15", *extra], system=system)
    return go


FIELDS = ("entry", "ckpt000", "ckpt001")


# ---- the arguments -------------------------------------------------------------------------------------------------------------

def test_the_arguments_are_x2d_plus_the_seed_and_the_checkpoints(warm, tmp_path):
    x2 = warm.x2
    real = ["--root", str(tmp_path/"w"), "--data-root", str(tmp_path/"d"), "--allowlist", "A", "--tissue-root", "T"]
    for phase, aid, wall, seconds in (("probe", "warm", 900., 225.), ("pairs", "warm", 900., 225.)):
        queue = warm.WarmQueue(warm.parse_args(["--phase", phase, *real]), FakeSystem())
        arm = queue.arms[aid]
        assert queue.arm_ids == [aid] and arm.prefix == "tag-warm" and queue.o.wall_seconds == wall and arm.checkpoint_seconds == seconds
        other = x2.X2Queue(x2.parse_args(["--root", str(tmp_path/"x"), "--data-root", str(tmp_path/"d"), "--allowlist", "A", "--tissue-root", "T", "--arms", "d", "--wall-seconds", str(wall)]), FakeSystem()).arms["d"]
        pair, eta = PAIRS[0], Path("/some/eta_d.json")
        ours, theirs = arm.run_args(pair, eta), other.run_args(pair, eta)
        seed = warm.SEED_ROOT/pair/"pg_weak__four__90s"/"final_native.npz"
        assert ours[:len(theirs)] == theirs
        assert ours[len(theirs):] == ["--init-field", str(seed), "--start-level", "3", "--checkpoint-seconds", warm.eta.number(seconds)]
        assert "--wall-seconds" in theirs and theirs[theirs.index("--wall-seconds")+1] == warm.eta.number(wall)
        a, b = arm.expected_args(pair, eta), other.expected_args(pair, eta)
        assert {k: a[k] for k in a if k not in b or a[k] != b[k]} == dict(init_field=str(seed), start_level=3, init_field_oracle_ok=False, checkpoint_seconds=seconds)
        assert not [k for k in b if k not in a]
        ac, oc = arm.configuration(), other.configuration()
        assert {k: ac[k] for k in ac if k not in oc or ac[k] != oc[k]} == dict(start_level=3, checkpoint_seconds=seconds, seed_arm="A1Ph", seed_root=str(warm.SEED_ROOT))
        # X2d's objective
        assert a["channel_source"] == "structure" and a["sigma_domain"] == "support_shared" and a["support_threshold"] == .05 and a["logvol_points"] == "gauss27+closed27"
        assert a["segmenter_id"] == "unet35_v1_ens" and a["structure_grouping"] == "structure13_v1" and a["jd_cell_estimate"] == "certificate" and a["rounds"] == "1,1,1,1"
        assert a["device"] == "cuda" and a["backend"] == "torch64" and a["wall_seconds"] == wall


def test_the_code_tree_check_finds_the_switches_in_the_help(warm, capsys):
    from bayesreg.cli import run_pair
    with pytest.raises(SystemExit):
        run_pair.parse_args(["--help"])
    text = capsys.readouterr().out
    assert all(flag in text for flag in warm.NEED_FLAGS)


def test_parse_args_refusals(warm, capsys):
    for argv, fragment in ((["--phase", "probe", "--wall-seconds", "60"], "--synthetic only"), (["--phase", "pairs", "--checkpoint-seconds", "9"], "--synthetic only"),
                           (["--phase", "probe", "--bz-go", "x"], "phase pairs only"), (["--phase", "pairs", "--bz-go", " "], "BZ's words"), (["--phase", "probe", "--segmenter-registry", "x"], "--synthetic only"),
                           (["--phase", "probe", "--no-start-after", "05:00"], "UTC offset"), (["--phase", "other"], "invalid choice")):
        with pytest.raises(SystemExit) as exc:
            warm.parse_args(argv)
        assert exc.value.code == 2 and fragment in capsys.readouterr().err, argv


def test_root_overlap_and_synthetic_roots(warm, tmp_path, capsys):
    for root in (warm.x2.REAL_ROOT, warm.SEED_ROOT, warm.RESULTS/"d0_v1_a1p", warm.RESULTS/"gate_a_x2"/"sub"):
        assert warm.main(["--phase", "probe", "--root", str(root)], system=FakeSystem()) == 2, root
        assert "overlaps" in capsys.readouterr().err
    assert warm.main(["--phase", "probe", "--synthetic", "--root", str(warm.RESULTS/"warm_dry"), "--seed-root", str(tmp_path), "--eta-json", str(tmp_path/"e.json"), "--x2-root", str(tmp_path/"x")], system=FakeSystem()) == 2
    assert "must lie outside" in capsys.readouterr().err
    assert warm.main(["--phase", "probe", "--synthetic", "--root", str(tmp_path/"r")], system=FakeSystem()) == 2     # the real seed root under --synthetic
    assert "synthetic inputs only" in capsys.readouterr().err


def test_a_command_with_the_word_of_the_gate_12_incident_is_not_launched(warm, tmp_path):
    queue = warm.WarmQueue(warm.parse_args(["--phase", "probe", "--root", str(tmp_path/"w"), "--data-root", "D", "--allowlist", "A", "--tissue-root", "T"]), FakeSystem())
    with pytest.raises(warm.Stop, match="gate 12"):
        queue.arms["warm"].launch(command=["python", "-m", warm.FORBIDDEN_WORD], kind="run", name="x", unit="u", memmax_text="9G", memmax_mib=9216, gpu=False, nice=None, runtime=1, quota="100%")
    assert not (tmp_path/"w").exists()


# ---- the probe -----------------------------------------------------------------------------------------------------------------

def test_probe_runs_one_pair_evaluates_every_state_and_says_go(driver, inputs, tmp_path, warm):
    root = tmp_path/"warm"
    system = WarmFake(accepted=12, rms=.2)
    assert driver(root, system, inputs, "probe") == 0, (root/"queue.log").read_text()[-2000:]
    short = SHORTS[-1]                                   # synthetic inputs: the last pair plays the probe's
    run_unit = f"tag-warmdry-warm-{short}"
    assert system.units == [run_unit, f"tag-warmdry-eval-warm-{short}", *(f"tag-warmdry-eval-warm-{short}-{f}" for f in FIELDS)]
    batch = json.loads((root/"batch_probe.json").read_text())
    assert batch["phase"] == "probe" and batch["pairs"] == batch["summary_pairs"] == [PAIRS[-1]] and batch["probe_pair"] == PAIRS[-1]
    assert batch["checkpoint_seconds"] == 15. and batch["method"] == "bayesreg_warm_v1"
    assert batch["arms"]["warm"]["eta"] == dict(alpha=.01, nu=1.37, tau=[1., 1., 1., 1.]) and batch["arms"]["warm"]["eta_file"].endswith("eta_d.json")
    assert batch["arms"]["warm"]["sigma_domain"] == "support_shared" and "诊断点" in batch["arms"]["warm"]["eta_role"]
    seeds = json.loads((root/"seeds.json").read_text())["seeds"]
    assert list(seeds) == list(PAIRS) and seeds[PAIRS[0]]["arm"] == "A1Ph" and seeds[PAIRS[0]]["oracle"] is False and len(seeds[PAIRS[0]]["file_sha256"]) == 64
    assert (root/"eval"/"warm"/short/f"gawarm_warm_{short}"/"metrics.json").is_file()
    for field in ("entry", "ckpt_000", "ckpt_001"):
        assert (root/"eval_fields"/"warm"/short/field/f"gawarm_warm_{short}_{field}"/"metrics.json").is_file(), field
    doc = json.loads((root/"summary_probe.json").read_text())
    assert doc["probe"]["go"] is True and doc["probe"]["label"].startswith("动起来了") and doc["probe"]["accepted"] == 12 and doc["probe"]["rms_all_voxels"] == .2 and doc["probe"]["rms_brain_voxels"] == .4
    assert [s["state"] for s in doc["rows"][0]["states"]] == ["entry", "ckpt_000", "ckpt_001", "final"] and doc["counts"]["fields_not_comparable"] == []
    assert doc["rows"][0]["states"][0]["seconds"] == 0. and doc["rows"][0]["states"][1]["accepted"] == 2 and doc["rows"][0]["accepted_k"] == 0
    text = (root/"summary_probe.md").read_text()
    assert "探针判定" in text and "动起来了" in text and "不跨种子" in text
    assert "probe verdict" in (root/"queue.log").read_text()


@pytest.mark.parametrize("accepted,rms,label,moved", [(5, .1, "动了但没到门槛", True), (5, .03, "动了但没到门槛", True), (3, .5, "动了但没到门槛", True), (9, .05, "动了但没到门槛", True),
                                                      (2, .5, "停滞", False), (5, .029, "停滞", False), (12, .05, "停滞", False), (12, .09, "停滞", False), (0, 0., "停滞", False)])
def test_probe_stagnation_stops_and_the_five_pairs_need_bz(driver, inputs, tmp_path, accepted, rms, label, moved):
    root = tmp_path/"warm"
    assert driver(root, WarmFake(accepted=accepted, rms=rms), inputs, "probe") == 0            # the verdict is a result, not a failure
    doc = json.loads((root/"summary_probe.json").read_text())
    assert doc["probe"]["go"] is False and label in doc["probe"]["label"] and doc["probe"]["moved_below_bar"] is moved
    assert "BZ" in doc["probe"]["label"] and "BZ decides" in (root/"queue.log").read_text()
    system = WarmFake()
    assert driver(root, system, inputs, "pairs") == 8 and system.units == []
    assert "hand to BZ" in (root/"queue.log").read_text() and not (root/"batch_pairs.json").exists()
    system = WarmFake()
    assert driver(root, system, inputs, "pairs", "--bz-go", "BZ 11-03: 继续") == 0, (root/"queue.log").read_text()[-2000:]
    assert [u for u in system.units if "-eval-" not in u] == [f"tag-warmdry-warm-{s}" for s in SHORTS[:-1]]
    assert json.loads((root/"batch_pairs.json").read_text())["probe_gate"]["bz_go"] == "BZ 11-03: 继续"


def test_probe_rms_boundaries_of_the_go(sw):
    def row(accepted, rms, state="ran", exit_code=0, evaluated=True, entry=True):
        known = isinstance(accepted, int) and rms is not None
        return dict(pair=PAIRS[0], run_status="budget_exhausted", state=state, exit_code=exit_code, states=[dict(evaluated=evaluated, comparable=entry)],
                    moved=dict(accepted=accepted, rms_all_voxels=rms, rms_brain_voxels=None, known=known, stagnant=None if not known else accepted < 10 or rms < .1))
    assert sw.probe_verdict(row(10, .1))["go"] is True
    assert sw.probe_verdict(row(9, .1))["go"] is False and sw.probe_verdict(row(9, .1))["moved_below_bar"] is True
    assert sw.probe_verdict(row(10, .0999))["go"] is False and sw.probe_verdict(row(10, .0999))["moved_below_bar"] is False
    for bad in (row(12, .2, state="failed_no_status"), row(12, .2, exit_code=2), row(None, None), row(12, .2, evaluated=False)):
        verdict = sw.probe_verdict(bad)
        assert verdict["go"] is False and verdict["label"] == sw.PROBE_FAILED
    for moved in (row(20, .5, entry=False), row(5, .05, entry=False), row(0, 0., entry=False)):      # review WARM_tools M2
        verdict = sw.probe_verdict(moved)
        assert verdict["go"] is False and verdict["label"] == sw.PROBE_UNDECIDABLE and "不可判" in verdict["label"] and verdict["moved_below_bar"] is False
        assert verdict["accepted"] == moved["moved"]["accepted"]


def test_probe_without_an_evaluated_entry_is_undecidable_and_stops_the_pairs(driver, inputs, tmp_path):
    root = tmp_path/"warm"
    system = WarmFake(accepted=20, rms=.5, eval_rc=[f"tag-warmdry-eval-warm-{SHORTS[-1]}-entry"])
    assert driver(root, system, inputs, "probe") == 0
    doc = json.loads((root/"summary_probe.json").read_text())
    assert doc["probe"]["go"] is False and "不可判" in doc["probe"]["label"] and "总负责人" in doc["probe"]["label"]
    assert "不可判" in (root/"summary_probe.md").read_text()
    assert driver(root, WarmFake(), inputs, "pairs") == 8
    assert "hand to BZ" in (root/"queue.log").read_text()


def test_the_bars_read_the_all_node_rms(sw):
    row = dict(native_accepted=12, states=[dict(state="final", rms_all_voxels=.09, rms_brain_voxels=.5)])
    assert sw.moved_of(row)["stagnant"] is True and sw.moved_of(row)["rms_brain_voxels"] == .5
    row["states"][0].update(rms_all_voxels=.1, rms_brain_voxels=.01)
    assert sw.moved_of(row)["stagnant"] is False
    row["states"][0].update(rms_all_voxels=None, rms_brain_voxels=.5)     # the brain-node value never stands in for it
    assert sw.moved_of(row)["known"] is False and sw.moved_of(row)["stagnant"] is None


def test_failed_probe_run_stops_the_pairs(driver, inputs, tmp_path):
    root = tmp_path/"warm"
    assert driver(root, WarmFake(drop=[f"tag-warmdry-warm-{SHORTS[-1]}"]), inputs, "probe") == 0
    assert (root/"failures"/f"run_warm-{SHORTS[-1]}.json").is_file()
    doc = json.loads((root/"summary_probe.json").read_text())
    assert doc["probe"]["go"] is False and "失败" in doc["probe"]["label"]
    assert driver(root, WarmFake(), inputs, "pairs") == 8


def test_pairs_need_a_probe(driver, inputs, tmp_path):
    root = tmp_path/"warm"
    system = WarmFake()
    assert driver(root, system, inputs, "pairs") == 8 and system.units == []
    assert "after the probe" in (root/"queue.log").read_text()


def test_pairs_after_a_go(driver, inputs, tmp_path, sw):
    root = tmp_path/"warm"
    assert driver(root, WarmFake(), inputs, "probe") == 0
    system = WarmFake()
    assert driver(root, system, inputs, "pairs") == 0, (root/"queue.log").read_text()[-2000:]
    runs = [u for u in system.units if "-eval-" not in u]
    assert runs == [f"tag-warmdry-warm-{s}" for s in SHORTS[:-1]] and len(system.units) == 2*5
    batch = json.loads((root/"batch_pairs.json").read_text())
    assert batch["pairs"] == list(PAIRS[:-1]) and batch["summary_pairs"] == list(PAIRS) and batch["probe_pair"] == PAIRS[-1]
    assert batch["probe_gate"]["go"] is True and batch["probe_gate"]["bz_go"] is None and batch["probe_gate"]["rms_all_voxels"] == .2
    assert batch["arms"]["warm"]["expected_args"] == json.loads((root/"batch_probe.json").read_text())["arms"]["warm"]["expected_args"]
    doc = json.loads((root/"summary_pairs.json").read_text())
    assert doc["readings"]["labels"] == [sw.READING_FEW] and doc["readings"]["comparable"] == 3          # three synthetic pairs: described only
    assert doc["state_means"]["entry"]["n"] == 3 and "ckpt_001" in doc["state_means"] and doc["state_means"]["final"]["n"] == 3
    assert [r["short"] for r in doc["rows"]] == list(SHORTS) and doc["pairs"] == list(PAIRS) and doc["launched_by_this_phase"] == list(PAIRS[:-1])
    assert [p.name for p in (root/"runs").iterdir()] == ["warm"] and sorted(p.name for p in (root/"runs"/"warm").iterdir()) == sorted(SHORTS)
    assert (root/"summary_probe.json").is_file() and (root/"summary_pairs.md").is_file()
    assert "判定" in (root/"summary_pairs.md").read_text()


def test_the_probes_run_is_that_pairs_result_and_nothing_repeats_it(driver, inputs, tmp_path):
    root = tmp_path/"warm"
    short = SHORTS[-1]
    assert driver(root, WarmFake(), inputs, "probe") == 0
    run = root/"runs"/"warm"/short
    kept = {name: (run/name).read_bytes() for name in ("status.json", "hyper.json", "config.json")}
    system = WarmFake(accepted=99)
    assert driver(root, system, inputs, "pairs") == 0, (root/"queue.log").read_text()[-2000:]
    assert system.units and not [u for u in system.units if short in u]
    for phase in ("probe", "pairs"):                             # restarts of either phase, in any order, launch nothing
        again = WarmFake(accepted=99)
        assert driver(root, again, inputs, phase) == 0 and again.units == [], phase
    assert {name: (run/name).read_bytes() for name in kept} == kept
    assert len(list((root/"launch").glob(f"run_warm-{short}_*.txt"))) == 1
    doc = json.loads((root/"summary_pairs.json").read_text())
    assert {r["short"]: r["moved"]["accepted"] for r in doc["rows"]} == {SHORTS[0]: 99, SHORTS[1]: 99, short: 12}


def test_a_lost_probe_run_is_not_relaunched_by_either_phase(driver, inputs, tmp_path):
    import shutil
    root = tmp_path/"warm"
    short = SHORTS[-1]
    assert driver(root, WarmFake(), inputs, "probe") == 0
    shutil.rmtree(root/"eval"/"warm"/short)
    shutil.rmtree(root/"runs"/"warm"/short)                      # the launch record stays
    system = WarmFake()
    assert driver(root, system, inputs, "probe") == 0
    assert not [u for u in system.units if "-eval-" not in u] and (root/"failures"/f"run_warm-{short}.json").is_file()


# ---- the readings of the plan's table -----------------------------------------------------------------------------------------

def _row(short, *, accepted=12, rms=.2, cortex=0., wm=0., seg4=0., seg35=0., large=0., l_entry=100., l_end=90.):
    large_labels = {str(k): dict(entry=.8, final=.8+large) for k in (2, 21, 1, 20, 5, 24)}
    differences = dict(cortex=cortex, white_matter=wm, seg4=seg4, seg35=seg35, large_labels=large_labels, L_entry=l_entry, L_end=l_end)
    return dict(short=short, moved=dict(accepted=accepted, rms_all_voxels=rms, known=True, stagnant=accepted < 10 or rms < .1), differences=differences)


def _five(**kw):
    return [_row(s, **kw) for s in FIVE]


FIVE = ("0438-0439", "0439-0440", "0440-0441", "0441-0442", "0442-0443")


def test_reading_stagnation_takes_precedence_at_three_of_five(sw):
    rows = _five(cortex=-.1)
    for row in rows[:3]:
        row["moved"].update(accepted=5, stagnant=True)
    got = sw.readings(rows)
    assert got["labels"] == [sw.READING_STAGNANT] and got["lines"]["stagnation"]["n"] == 3 and got["lines"]["pull_back"]["reached"] is False
    assert got["lines"]["pull_back"]["falls"]["cortex"]["drop"] == pytest.approx(.1)      # still reported, described only


@pytest.mark.parametrize("k", [1, 2])
def test_reading_one_or_two_stagnant_pairs_are_described_only(sw, k):
    """Review WARM_tools I1: lines 2 to 4 read nothing, whatever the means say; line 1 needs three."""
    for change in (dict(cortex=-.1), dict(seg35=.05), dict()):
        rows = _five(**change)
        for row in rows[:k]:
            row["moved"].update(accepted=5, stagnant=True)
        got = sw.readings(rows)
        assert got["labels"] == [f"只描述（{k}/5 对停滞）"] and got["lines"]["stagnation"]["reached"] is False
        assert not any(got["lines"][key]["reached"] for key in ("pull_back", "compatible", "both_paths")), change


def test_reading_stagnant_pairs_do_not_dilute_the_cortex(sw):
    """The review's counterexample: the three moved pairs are the same in both rows; the two that did not move must not turn the reading into 相容."""
    moved = dict(cortex=-.016, wm=-.002)
    assert sw.readings(_five(**moved))["labels"] == [sw.READING_MIDDLE]
    rows = _five(**moved)
    for row in rows[3:]:
        row["moved"].update(accepted=4, rms_all_voxels=.02, stagnant=True)
        row["differences"].update(cortex=0., white_matter=0.)
    got = sw.readings(rows)
    assert got["lines"]["compatible"]["cortex_gain"] == pytest.approx(-.0096)             # inside the bar .01: the mean alone would read 相容
    assert got["lines"]["compatible"]["reached"] is False and got["labels"] == ["只描述（2/5 对停滞）"]
    assert sw.READING_COMPATIBLE not in got["labels"]


@pytest.mark.parametrize("cortex,seg4,reached", [(-.063, 0., True), (-.0629, 0., False), (0., -.026, True), (0., -.0259, False), (-.1, -.1, True)])
def test_reading_pull_back_bars(sw, cortex, seg4, reached):
    got = sw.readings(_five(cortex=cortex, seg4=seg4))
    assert got["lines"]["pull_back"]["reached"] is reached
    assert (sw.READING_PULLED in got["labels"]) is reached


def test_reading_pull_back_needs_the_same_sign_on_five(sw):
    rows = _five(cortex=-.1)
    rows[0]["differences"]["cortex"] = .01
    got = sw.readings(rows)
    assert got["lines"]["pull_back"]["falls"]["cortex"]["same_sign"] == 4 and got["lines"]["pull_back"]["reached"] is False


def test_reading_compatible(sw):
    got = sw.readings(_five(cortex=-.01, wm=-.01))
    assert got["lines"]["compatible"]["reached"] is True and sw.READING_COMPATIBLE in got["labels"]
    assert sw.readings(_five(cortex=-.0101))["lines"]["compatible"]["reached"] is False
    assert sw.readings(_five(wm=-.0101))["lines"]["compatible"]["reached"] is False
    rows = _five()
    rows[1]["differences"]["L_end"] = 101.
    got = sw.readings(rows)
    assert got["lines"]["compatible"]["L_down"] == 4 and got["lines"]["compatible"]["reached"] is False
    rows = _five()
    rows[1]["differences"]["L_end"] = None                      # L_undefined (M14)
    assert sw.readings(rows)["lines"]["compatible"]["reached"] is False


def test_reading_both_paths(sw):
    got = sw.readings(_five(seg35=.01, large=-.0099))
    assert got["lines"]["both_paths"]["reached"] is True and sw.READING_BOTH in got["labels"]
    assert sw.READING_COMPATIBLE in got["labels"]               # each line on its own
    assert sw.readings(_five(seg35=.0099))["lines"]["both_paths"]["reached"] is False
    assert sw.readings(_five(seg35=.01, large=-.0101))["lines"]["both_paths"]["reached"] is False
    rows = _five(seg35=.02)
    rows[0]["differences"]["seg35"] = -.001
    assert sw.readings(rows)["lines"]["both_paths"]["reached"] is False
    rows = _five(seg35=.02)                                     # each of the six, not their mean
    rows_large = {str(k): dict(entry=.8, final=.8) for k in (2, 21, 1, 20, 5, 24)}
    rows_large["2"] = dict(entry=.8, final=.78)
    rows_large["21"] = dict(entry=.8, final=.82)
    for row in rows:
        row["differences"]["large_labels"] = rows_large
    got = sw.readings(rows)["lines"]["both_paths"]
    assert got["reached"] is False and got["large"]["2"]["gain"] == pytest.approx(-.02)


def test_reading_middle_band_and_too_few(sw):
    got = sw.readings(_five(cortex=-.03, wm=-.03))
    assert got["labels"] == [sw.READING_MIDDLE]
    rows = _five(cortex=-.1)
    rows[4]["differences"] = None                               # a missing pair: described only
    got = sw.readings(rows)
    assert got["labels"] == [sw.READING_FEW] and got["lines"]["pull_back"]["reached"] is False and got["comparable"] == 4


def test_groups_partition_the_35_labels(sw):
    assert sw.SMALL_SIX == (4, 23, 18, 34, 19, 35) and sorted(sw.LARGE_SIX) == [1, 2, 5, 20, 21, 24] and len(sw.OTHER_23) == 23
    assert sorted(sw.SMALL_SIX+sw.LARGE_SIX+sw.OTHER_23) == list(range(1, 36))


def test_summary_level_readings_at_full_count(driver, inputs, tmp_path, sw, monkeypatch):
    """The whole path with REQUIRED_N lowered to the three synthetic pairs: the entry and the end from the evaluator's files, L from the run's records."""
    monkeypatch.setattr(sw, "REQUIRED_N", 3)
    root = tmp_path/"warm"
    # the driver's module has its own loaded summarize_warm; only this module's summary is read below

    def dice(short, field):
        return dict(cortex=.85-.08, seg4=.886-.03) if field == "final" else {}
    assert driver(root, WarmFake(dice=dice), inputs, "probe") == 0
    assert driver(root, WarmFake(dice=dice), inputs, "pairs") == 0
    doc = sw.summary(root, "pairs")
    assert doc["readings"]["labels"] == [sw.READING_PULLED] and doc["readings"]["comparable"] == 3
    assert doc["readings"]["lines"]["pull_back"]["falls"]["cortex"]["drop"] == pytest.approx(.08)
    assert doc["readings"]["lines"]["compatible"]["reached"] is False
    assert doc["rows"][0]["differences"]["L_end"] == 90. and doc["rows"][0]["differences"]["L_entry"] == 100.
    assert doc["state_means"]["entry"]["cortex"] == pytest.approx(.85) and doc["state_means"]["final"]["cortex"] == pytest.approx(.77)
    assert doc["state_means"]["entry"]["small_six"] == pytest.approx(.53) and doc["state_means"]["entry"]["large_six"] == pytest.approx(.89) and doc["state_means"]["entry"]["other_23"] == pytest.approx(.82)
    text = sw.markdown(doc)
    assert sw.READING_PULLED in text and "小六个" in text and "其余 23 个" in text and "SDlogJ 官方" in text and "A1Ph" in text and "X2d" in text
    assert doc["reference_means"]["A1Ph"]["seg4"] == pytest.approx(.886) and doc["seed_check"][0]["max_gap"] == pytest.approx(0.)
    assert doc["bars_check"]["seg4"]["recomputed"] == pytest.approx((.886-.834)/2) and doc["bars_check"]["seg4"]["agrees"] is True
    assert doc["bars_check"]["cortex"]["recomputed"] == pytest.approx((.85-.72)/2) and doc["bars_check"]["cortex"]["agrees"] is False    # synthetic references: the flag works


def test_summary_refuses_a_non_unit_spacing(driver, inputs, tmp_path, sw):
    root = tmp_path/"warm"
    assert driver(root, WarmFake(), inputs, "probe") == 0
    run = root/"runs"/"warm"/SHORTS[-1]
    hyper = json.loads((run/"hyper.json").read_text())
    hyper["rounds"][0]["spacing_xyz"] = [1., 1., 2.]
    (run/"hyper.json").write_text(json.dumps(hyper))
    with pytest.raises(sw.SummaryError, match="voxels"):
        sw.summary(root, "probe")


def test_summary_write_once(driver, inputs, tmp_path, sw):
    root = tmp_path/"warm"
    assert driver(root, WarmFake(), inputs, "probe") == 0
    js, _, state = sw.write(root, "probe")
    assert state == "unchanged" and js == root/"summary_probe.json"
    run = root/"runs"/"warm"/SHORTS[-1]
    hyper = json.loads((run/"hyper.json").read_text())
    hyper["profile"]["end"]["L"] = 80.
    (run/"hyper.json").write_text(json.dumps(hyper))
    js, md, state = sw.write(root, "probe")
    assert "differs" in state and js != root/"summary_probe.json" and md.is_file() and js.name.startswith("summary_probe_")


# ---- checks that STOP ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("change,fragment", [
    (dict(stage="D0-A1Ps"), "stage"), (dict(arm="A4p"), "preprocessing_d0"), (dict(oracle=True), "preprocessing_d0"), (dict(role="certified_native_initializer"), "role"), (dict(status="infeasible"), "certificate")])
def test_a_wrong_seed_stops_before_any_unit(driver, inputs, tmp_path, change, fragment):
    import shutil
    shutil.rmtree(inputs/"seeds"/PAIRS[1])
    write_seed(inputs/"seeds", PAIRS[1], **change)
    root = tmp_path/"warm"
    system = WarmFake()
    assert driver(root, system, inputs, "probe") == 8 and system.units == []
    assert fragment in (root/"queue.log").read_text() and not (root/"batch_probe.json").exists()


def test_a_changed_seed_file_or_missing_seed_stops(driver, inputs, tmp_path):
    seed = inputs/"seeds"/PAIRS[0]/"pg_weak__four__90s"/"final_native.npz"
    seed.write_bytes(seed.read_bytes()+b"x")
    root = tmp_path/"warm"
    assert driver(root, WarmFake(), inputs, "probe") == 8
    assert "sha256" in (root/"queue.log").read_text()
    seed.unlink()
    root = tmp_path/"warm2"
    assert driver(root, WarmFake(), inputs, "probe") == 8
    assert "missing" in (root/"queue.log").read_text()


@pytest.mark.parametrize("name,where", [("manifest", lambda run, pair: run/"inputs"/pair/"manifest.json"), ("protocol", lambda run, pair: run/"protocol.json"),
                                         ("final", lambda run, pair: run/"pg_weak__four__90s"/"final_native.json")])
def test_an_unreadable_seed_record_stops_with_exit_8(warm, driver, inputs, tmp_path, capsys, name, where):
    """Review WARM_tools M5: a STOP as documented, in the driver and in --dry-run, not a traceback."""
    where(inputs/"seeds"/PAIRS[0], PAIRS[0]).write_text("{not json")
    root = tmp_path/"warm"
    system = WarmFake()
    assert driver(root, system, inputs, "probe") == 8 and system.units == []
    assert "STOP: seed of" in (root/"queue.log").read_text()
    capsys.readouterr()
    assert warm.main(["--phase", "probe", "--root", str(tmp_path/"warm2"), *common(inputs), "--dry-run"], system=FakeSystem()) == 8
    assert "STOP: seed of" in capsys.readouterr().err
    where(inputs/"seeds"/PAIRS[0], PAIRS[0]).write_text("[1, 2]")     # valid JSON, not an object
    root = tmp_path/"warm3"
    assert driver(root, WarmFake(), inputs, "probe") == 8
    assert "STOP: seed of" in (root/"queue.log").read_text()


def test_the_seed_filter_drops_label_keys(warm, tmp_path):
    write_seed(tmp_path, PAIRS[0])
    manifest = warm.read_seed_manifest(tmp_path/PAIRS[0]/"inputs"/PAIRS[0]/"manifest.json")
    assert "baseline_label_metrics" not in manifest and "colour" not in manifest["preprocessing_d0"] and manifest["preprocessing_d0"]["oracle"] is False


@pytest.mark.parametrize("key,value", [("nu", 3.87), ("sigma_domain", "brain"), ("support_threshold", .1), ("channel_source", "tissue"), ("alpha", .02)])
def test_an_eta_record_that_is_not_x2d_stops(driver, inputs, tmp_path, key, value):
    path = inputs/"x2"/"eta"/"eta_d.json"
    doc = json.loads(path.read_text())
    doc[key] = value
    path.write_text(json.dumps(doc))
    root = tmp_path/"warm"
    system = WarmFake()
    assert driver(root, system, inputs, "probe") == 8 and system.units == []
    assert "differs from X2d" in (root/"queue.log").read_text()


def test_the_real_eta_file_is_pinned(warm, tmp_path, inputs):
    args = warm.parse_args(["--phase", "probe", "--root", str(tmp_path/"w"), "--eta-json", str(inputs/"x2"/"eta"/"eta_d.json"), "--data-root", "D", "--allowlist", "A", "--tissue-root", "T"])
    queue = warm.WarmQueue(args, FakeSystem())
    with pytest.raises(warm.Stop, match="pinned"):
        queue.eta_value()


@pytest.mark.parametrize("name,change,fragment", [
    ("args", lambda c: c["args"].update(sigma_domain="brain"), "config.json args differ"),
    ("args_seed", lambda c: c["args"].update(init_field="/other"), "config.json args differ"),
    ("args_ckpt", lambda c: c["args"].update(checkpoint_seconds=1.), "config.json args differ"),
    ("seed", lambda c: c["init_field"].update(file_sha256="0"*64), "seed file_sha256"),
    ("oracle", lambda c: c["init_field"].update(oracle=True), "oracle"),
    ("solver", lambda c: c["solver"].update(sigma_domain="brain"), "sigma_domain"),
    ("level", lambda c: c["solver"].update(start_level=0), "start_level"),
    ("rule", lambda c: c["solver"].update(checkpoint=None), "checkpoint"),
    ("eta", lambda c: c["eta"].update(nu=2.), "eta"),
])
def test_a_run_that_differs_from_the_plan_stops(driver, inputs, tmp_path, name, change, fragment):
    root = tmp_path/"warm"
    assert driver(root, WarmFake(config_change=change), inputs, "probe") == 8
    log = (root/"queue.log").read_text()
    assert "STOP" in log and fragment in log, log[-1500:]


def test_a_refused_seed_stops(driver, inputs, tmp_path):
    root = tmp_path/"warm"
    system = WarmFake(status_override=dict(status="init_field_rejected", wording="运行被拒", exit_code=5, native=False, reason="--init-field: x"))
    assert driver(root, system, inputs, "probe") == 8
    assert "the seed was refused" in (root/"queue.log").read_text()


def test_a_checkpoint_whose_certificate_did_not_pass_stops(driver, inputs, tmp_path):
    root = tmp_path/"warm"
    short = SHORTS[-1]
    assert driver(root, WarmFake(failed_cert=[f"tag-warmdry-eval-warm-{short}-ckpt000"]), inputs, "probe") == 8
    log = (root/"queue.log").read_text()
    assert "hard constraint 1" in log and "ckpt000" in log and "probe verdict" not in log


def test_a_failed_field_evaluation_is_kept_and_not_rerun(driver, inputs, tmp_path):
    root = tmp_path/"warm"
    short = SHORTS[-1]
    unit = f"tag-warmdry-eval-warm-{short}-ckpt001"
    assert driver(root, WarmFake(eval_rc=[unit]), inputs, "probe") == 0
    assert (root/"eval_fields"/"warm"/short/"ckpt_001"/"eval_failed.json").is_file()
    doc = json.loads((root/"summary_probe.json").read_text())
    assert doc["counts"]["fields_not_comparable"] == [f"{short}/ckpt_001"] and doc["probe"]["go"] is True
    system = WarmFake()
    assert driver(root, system, inputs, "probe") == 0
    assert system.units == []                                    # kept, not rerun


def test_a_restart_launches_nothing_and_evaluates_what_is_missing(driver, inputs, tmp_path):
    root = tmp_path/"warm"
    short = SHORTS[-1]
    assert driver(root, WarmFake(), inputs, "probe") == 0
    system = WarmFake()
    assert driver(root, system, inputs, "probe") == 0 and system.units == []
    import shutil
    shutil.rmtree(root/"eval_fields"/"warm"/short/"entry")      # the entry never evaluated (the driver was killed before it)
    for record in (root/"launch").glob(f"eval_warm-{short}-entry_*.txt"):
        record.unlink()
    system = WarmFake()
    assert driver(root, system, inputs, "probe") == 0
    assert system.units == [f"tag-warmdry-eval-warm-{short}-entry"]


def test_a_killed_run_is_not_relaunched(driver, inputs, tmp_path):
    root = tmp_path/"warm"
    short = SHORTS[-1]
    assert driver(root, WarmFake(drop=[f"tag-warmdry-warm-{short}"]), inputs, "probe") == 0
    system = WarmFake()
    assert driver(root, system, inputs, "probe") == 0 and system.units == []


def test_the_gate_stops_with_a_busy_machine(driver, inputs, tmp_path):
    root = tmp_path/"warm"
    system = WarmFake(busy=["tag-d0a5-A5-0438-0439.service"])
    assert driver(root, system, inputs, "probe", "--wait-minutes", "0") == 8 and system.units == []
    assert "tag-d0a5" in (root/"queue.log").read_text()
    root = tmp_path/"warm_hold"
    root.mkdir()
    (root/"hold").write_text("")
    system = WarmFake()
    assert driver(root, system, inputs, "probe", "--wait-minutes", "0") == 8 and system.units == []
    assert "hold file" in (root/"queue.log").read_text()


def test_the_gate_of_the_gpu_run_waits_for_the_load_only_there(warm, tmp_path):
    class Loaded(WarmFake):
        def loadavg(self):
            return "3.5 1 1"
    args = warm.parse_args(["--phase", "probe", "--root", str(tmp_path/"w"), "--data-root", "D", "--allowlist", "A", "--tissue-root", "T"])
    arm = warm.WarmQueue(args, Loaded()).arms["warm"]
    assert "1-minute load 3.50" in arm.busy_reason(9216, True) and arm.busy_reason(6144, False) == ""
    assert warm.WarmQueue(args, WarmFake()).arms["warm"].busy_reason(9216, True) == ""


# ---- --dry-run -----------------------------------------------------------------------------------------------------------------

def test_dry_run_prints_every_command_and_writes_nothing(warm, inputs, tmp_path, capsys):
    root = tmp_path/"warm"
    argv = ["--phase", "pairs", "--root", str(root), *common(inputs), "--dry-run", "--wall-seconds", "60", "--checkpoint-seconds", "15"]
    assert warm.main(argv, system=FakeSystem()) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("systemd-run")]
    assert len(lines) == 2*(1+1+1+warm.PLANNED_CHECKPOINTS)
    assert not root.exists()
    for short in SHORTS[:-1]:
        assert f"--unit=tag-warmdry-warm-{short} " in out and f"--unit=tag-warmdry-eval-warm-{short}-ckpt000 " in out
    assert SHORTS[-1] not in "".join(lines)                      # the probe's pair is the probe's run
    run_lines = [line for line in lines if "bayesreg.cli.run_pair" in line]
    assert len(run_lines) == 2 and all("--init-field" in line and "--start-level 3" in line and "--checkpoint-seconds 15" in line and "--sigma-domain support_shared" in line for line in run_lines)
    assert all("eta_d.json" in line for line in run_lines) and all("MemoryMax=9G" in line for line in run_lines)
    assert all("--pair " in line for line in lines if "-ckpt" in line.split()[2] or "-entry" in line.split()[2])
    assert "accepted these arguments" in out


def test_dry_run_of_the_probe_prints_one_pair(warm, inputs, tmp_path, capsys):
    argv = ["--phase", "probe", "--root", str(tmp_path/"warm"), *common(inputs), "--dry-run"]
    assert warm.main(argv, system=FakeSystem()) == 0
    out = capsys.readouterr().out
    assert len([line for line in out.splitlines() if "bayesreg.cli.run_pair" in line]) == 1 and "--wall-seconds 900" in out and "--checkpoint-seconds 225" in out and SHORTS[-1] in out


def test_dry_run_stops_on_a_bad_seed(warm, inputs, tmp_path, capsys):
    (inputs/"seeds"/PAIRS[0]/"pg_weak__four__90s"/"final_native.npz").unlink()
    assert warm.main(["--phase", "probe", "--root", str(tmp_path/"warm"), *common(inputs), "--dry-run"], system=FakeSystem()) == 8
    assert "STOP" in capsys.readouterr().err


def test_real_constants(warm):
    assert warm.REAL_ROOT == Path("/home/rmcfz/seq_vp_tag_results/warm_v1") and warm.SEED_ROOT == Path("/home/rmcfz/seq_vp_tag_results/d0_v1_a1p/runs/A1Ph")
    assert warm.ETA_JSON == Path("/home/rmcfz/seq_vp_tag_results/gate_a_x2/eta/eta_d.json") and warm.PROBE_SHORT == "0441-0442" and warm.PROBE_SHORT in warm.FIRST_FIVE
    assert warm.FIRST_FIVE == FIVE and warm.METHOD == "bayesreg_warm_v1" and "experiments/gate_a/run_x2.py" in warm.DRIVER_SCRIPTS and "experiments/gate_a/run_warm.py" in warm.DRIVER_SCRIPTS
    assert list(warm.ARMS) == [warm.ARM] and warm.ARMS["warm"]["nu"] == 1.37 and warm.ARMS["warm"]["sigma"] == "support_shared" and warm.ARMS["warm"]["theta"] == .05
    assert warm.ARMS["warm"]["wall_seconds"] == 900. and warm.ARMS["warm"]["checkpoint_seconds"] == 225. and warm.PHASES == ("probe", "pairs")


# ---- slow: the real run_pair and bayesreg.eval.cli (numpy backend, CPU, synthetic 26^3 pairs) -----------------------------------------

def write_real_seed(seeds, data, pair, u_scale=.8):
    """A frozen configuration C directory that passes the real binding of C12-4a item 2 on a synthetic pair (the keys of d0_v1_a1p/runs/A1Ph/<pair>)."""
    from jd_curl.three_d.nifti import file_sha256
    from bayesreg.io import load_pair
    from bayesreg.solve.stopping import state_hash
    from . import _synthetic as syn
    source = data/"stage_L2R_val_v1"/pair
    loaded = load_pair(source/"pair.npz", allowlist=str(data.parent/"dev_subjects.json"))
    u = np.array(syn.solenoidal_displacement(loaded.grid, u_scale))
    run = seeds/pair
    task = run/"pg_weak__four__90s"
    task.mkdir(parents=True)
    np.savez_compressed(task/"final_native.npz", u=u, warped=np.zeros(u.shape[:-1]))
    manifest = json.loads((source/"manifest.json").read_text())
    recoloured = "r"*64
    original = manifest["pair_sha256"]
    manifest.update(pair=pair, pair_sha256=recoloured, baseline_label_metrics=dict(seg35=.5), preprocessing_d0=dict(arm="A1Ph", oracle=False, source_pair_sha256=original, colour=dict(fixed={})))
    write_json(run/"inputs"/pair/"manifest.json", manifest)
    write_json(task/"final_native.json", dict(path="final_native.npz", role="formal_native_outer_commit", sha256=file_sha256(task/"final_native.npz"), certificate=dict(
        status="certified_feasible", reason="positive_enclosure_cover", min_jd_lower_bound=.1, n_infeasible=0, n_uncertain=0, state_hash=state_hash(u), grid_id=loaded.grid.grid_id, certificate_version="synthetic")))
    write_json(run/"protocol.json", dict(stage="D0-A1Ph", task=dict(index=0, pair=pair, route="pg_weak", name="pg_weak__four__90s", wall_seconds=90), pre_optimizer="fp32-v1:synthetic",
                                         input_sha256={"pair.npz": recoloured, "manifest.json": file_sha256(run/"inputs"/pair/"manifest.json")}))
    write_json(run/"metrics.json", dict(metrics=dict(seg35=.8, seg4=.85, hd95=2., sdlogj_official=.07, sdlogj_brain=.5, **labels_doc()), certificate=dict(passed=True, min_jd_lower_bound=.1)))


def finite_dice(state):
    return all(isinstance(state["metrics"].get(k), float) for k in ("seg35", "seg4", "small_six", "large_six", "other_23"))


@pytest.mark.slow
def test_real_run_pair_and_evaluator_numpy(warm, sw, dry_mod, tmp_path, monkeypatch):
    """The driver's exact argument lists, with the backend switched to numpy only: seed binding on structure channels and support_shared, checkpoints, the entry field, the evaluator on
    disp directories, the summary on real records.  Each number here is synthetic."""
    from .test_pilot_tools import InlineSystem
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    dry = tmp_path/"dry"
    dry_mod.main(["--out", str(dry), "--wall-seconds", "60", "--extra", "", "--code-sha", head()])
    for pair, short in zip(PAIRS, SHORTS):
        write_real_seed(dry/"seeds", dry/"data", pair)
        write_x2(dry, pair, short)
    write_json(dry/"x2"/"eta"/"eta_d.json", dict(alpha=.01, nu=1.37, tau=[1., 1., 1., 1.], channel_source="structure", sigma_domain="support_shared", support_threshold=.05))
    sha = head()

    def prepare(self):
        self.code, self.code_sha = ROOT, sha
        self.allowlist = self.allowlist or self.o.allowlist
    monkeypatch.setattr(warm.WarmArm, "prepare_code", prepare)
    monkeypatch.setattr(warm.WarmArm, "write_env", lambda self: None)
    original = warm.namespace

    def numpy_namespace(o, spec):
        space = original(o, spec)
        space.backend = "numpy"
        return space
    monkeypatch.setattr(warm, "namespace", numpy_namespace)
    root = tmp_path/"warm"
    system = InlineSystem()
    argv = ["--root", str(root), *common(dry), "--wall-seconds", "9", "--checkpoint-seconds", "3"]
    assert warm.main(["--phase", "probe", *argv], system=system) == 0, (root/"queue.log").read_text()[-3000:]
    short = SHORTS[-1]
    run = root/"runs"/"warm"/short
    status, hyper = json.loads((run/"status.json").read_text()), json.loads((run/"hyper.json").read_text())
    assert status["native"] and status["export"]["certificate_passed"] is True and hyper["init_field"]["accepted_k"] == 0
    assert hyper["init_field"]["arm"] == "A1Ph" and hyper["init_field"]["oracle"] is False and hyper["algorithm"]["start_level"] == 3 and hyper["transfers"] == []
    assert hyper["support_levels"][0]["seed_state_hash"] == hyper["init_field"]["entry_state_hash"] and hyper["profile"]["errors"] == []
    names = [g["name"] for g in hyper["profile"]["entry"]["groups"]]
    assert names[0] == "I" and set(names) <= {"I", "S"}
    doc = json.loads((root/"summary_probe.json").read_text())
    states = doc["rows"][0]["states"]
    assert states[0]["state"] == "entry" and states[-1]["state"] == "final" and all(s["evaluated"] and s["certificate"] == "certified_feasible" for s in states), states
    assert states[0]["field_role"] == "disp_only" and states[-1]["field_role"] in ("native_formal", "handover_only")
    assert all(finite_dice(s) for s in states) and states[0]["L"] is not None and states[0]["rms_all_voxels"] == 0. and states[0]["rms_brain_voxels"] == 0.
    assert doc["rows"][0]["accepted_k"] == 0 and doc["seed_check"][0]["entry_is_seed"] is True
    with np.load(run/"checkpoints"/"entry"/"disp.npz") as z:
        assert json.loads(str(z["meta"]))["u_state_hash"] == hyper["init_field"]["state_hash"]
    if hyper["profile"]["accepted"]:
        assert states[-1]["L"] <= states[0]["L"]                  # one round, sigma at the M step of the entry: L cannot rise along accepted steps
    assert (root/"summary_probe.md").read_text().count("| entry |") >= 1
    entry_doc = json.loads((root/"eval_fields"/"warm"/short/"entry"/f"gawarm_warm_{short}_entry"/"metrics.json").read_text())
    assert entry_doc["init_field"]["accepted_k"] == 0 and entry_doc["init_field"]["scale"] == 1.      # C16 (review WARM_tools M1)
    assert entry_doc["init_field"]["entry_state_hash"] == hyper["init_field"]["entry_state_hash"]
    assert warm.main(["--phase", "pairs", *argv, "--bz-go", "synthetic"], system=InlineSystem()) == 0, (root/"queue.log").read_text()[-3000:]
    doc = json.loads((root/"summary_pairs.json").read_text())
    assert [r["state"] for r in doc["rows"]] == ["ran"]*3 and doc["counts"]["args_mismatch"] == 0 and doc["counts"]["eta_mismatch"] == 0 and doc["readings"]["labels"]


# ---- the probe's combination (review WARM_tools I2): numpy here, torch64 cpu and cuda in tests/test_bayesreg/torch64/test_init_field_torch64.py ----------

def check_probe_combination(warm, dry_mod, tmp_path, monkeypatch, *, backend, device=None):
    """The driver's argument list for a synthetic pair (structure channels, support_shared, the seed, --checkpoint-seconds), only the backend and device set here and no budget acting
    (--wall-seconds 1e6, --max-accepted 3), so that two runs compare bit for bit: without any recorder (run_pair always makes one under --init-field, so solve_multilevel's recorder is
    dropped here) and with the driver's switches.  Then: the run ends and passes the certificate, the records have no errors, the entry and the checkpoint files read as disp directories
    with C16's seed tag, and bayesreg.eval.cli evaluates the entry.  Synthetic numbers only."""
    from bayesreg.cli import run_pair
    from bayesreg.eval import cli
    from bayesreg.eval.fields import describe_run, read_disp
    from bayesreg.solve import jd_scaling as jds
    dry = tmp_path/"dry"
    dry_mod.main(["--out", str(dry), "--wall-seconds", "60", "--extra", "", "--code-sha", head()])
    pair = PAIRS[-1]
    write_real_seed(dry/"seeds", dry/"data", pair)
    eta = write_json(dry/"x2"/"eta"/"eta_d.json", dict(alpha=.01, nu=1.37, tau=[1., 1., 1., 1.], channel_source="structure", sigma_domain="support_shared", support_threshold=.05))
    args = warm.parse_args(["--phase", "probe", "--root", str(tmp_path/"warm"), *common(dry), "--wall-seconds", "1000000", "--checkpoint-seconds", "0.001"])
    arm = warm.WarmQueue(args, FakeSystem()).arms[warm.ARM]
    arm.registry_path, arm.allowlist = dry/"registry"/"segmenters.json", str(dry/"dev_subjects.json")
    arm.o.backend = backend
    if device is not None:
        arm.device = device
    if device == "cuda":
        arm.threads = 4                                          # the real runs'
    argv = [*arm.run_args(pair, eta), "--max-accepted", "3"]
    k = argv.index("--checkpoint-seconds")
    plain = argv[:k]+argv[k+2:]
    assert ("--backend", backend) == tuple(argv[argv.index("--backend"):argv.index("--backend")+2]) and "--sigma-domain" in argv and "--init-field" in argv and "--checkpoint-seconds" not in plain
    real = run_pair.solve_multilevel

    def no_recorder(*a, recorder=None, **kw):
        return real(*a, **kw)
    without, recorded = tmp_path/"without", tmp_path/"recorded"
    with monkeypatch.context() as patch:
        patch.setattr(run_pair, "solve_multilevel", no_recorder)
        jds.clear_cell_bounds_cache()
        assert run_pair.main(["--out", str(without), *plain]) == 0
    jds.clear_cell_bounds_cache()
    assert run_pair.main(["--out", str(recorded), *argv]) == 0
    hyper_a, hyper_b = (json.loads((run/"hyper.json").read_text()) for run in (without, recorded))
    assert "profile" not in hyper_a and hyper_b["profile"]["errors"] == []
    assert len(hyper_a["rounds"]) == len(hyper_b["rounds"]) == 1
    for key in ("level", "round", "status", "accepted", "commits", "sigma2_used", "sigma2_S_used", "G0", "T_B", "F_B_start", "F_B_end", "F_B_trace", "GF_B_trace", "GF_B", "state_hash",
                "min_jd_lower_bound", "counters"):
        assert hyper_a["rounds"][0][key] == hyper_b["rounds"][0][key], key
    for run in (without, recorded):
        status = json.loads((run/"status.json").read_text())
        assert status["native"] and status["export"]["certificate_passed"] is True and status["init_field"]["oracle"] is False
    for name, key in (("final_native.npz", "u"), ("disp.npz", "d")):
        with np.load(without/name) as a, np.load(recorded/name) as b:
            assert a[key].tobytes() == b[key].tobytes(), name
    profile, record = hyper_b["profile"], hyper_b["init_field"]
    assert record["arm"] == "A1Ph" and record["accepted_k"] == 0 and hyper_b["support_levels"][0]["seed_state_hash"] == record["entry_state_hash"] and hyper_b["transfers"] == []
    assert [g["name"] for g in profile["entry"]["groups"]] == ["I", "S"] and profile["entry"]["sigma2_hat"]["S"] == pytest.approx(hyper_b["rounds"][0]["sigma2_S_used"], rel=1e-10)
    assert profile["accepted"] == hyper_b["rounds"][0]["accepted"] == 3 and profile["n_checkpoints"] == 3 and profile["end"]["L"] < profile["entry"]["L"]
    assert profile["end"]["rms_from_entry_all_mm"] > 0 and profile["end"]["rms_from_entry_mm"] > 0
    shape = tuple(hyper_b["rounds"][0]["shape_zyx"])
    states = {"entry": profile["entry"]["state_hash"], **{f"ckpt_{c['k']:03d}": c["state_hash"] for c in profile["checkpoints"]}}
    assert sorted(p.name for p in (recorded/"checkpoints").iterdir()) == sorted(states)
    for name, digest in states.items():
        directory = recorded/"checkpoints"/name
        info = describe_run(directory, pair=pair)
        assert info.kind == "disp" and info.field_role == "disp_only" and info.extra["init_field"]["accepted_k"] == 0 and info.extra["init_field"]["scale"] == 1.
        d, field = read_disp(directory/"disp.npz", shape)
        assert d.shape == (3, *shape) and field["meta"]["u_state_hash"] == digest and field["meta"]["init_field"]["entry_state_hash"] == record["entry_state_hash"]
    ev = tmp_path/"eval_entry"
    code = cli.main([str(recorded/"checkpoints"/"entry"), "--pairs-stage", warm.STAGE, "--out", "metrics.json", "--long", str(tmp_path/"long.csv"), "--out-dir", str(ev), "--method", "bayesreg_warm_v1",
                     "--arm", "WARM-entry", "--run-id", "entry", "--data-root", str(dry/"data"), "--allowlist", str(dry/"dev_subjects.json"), "--pair", pair])
    assert code == 0
    doc = json.loads((ev/"entry"/"metrics.json").read_text())
    assert doc["certificate"]["passed"] is True and doc["init_field"]["accepted_k"] == 0 and isinstance(doc["metrics"]["seg35"], float) and isinstance(doc["metrics"]["seg4"], float)


@pytest.mark.slow
def test_probe_combination_numpy(warm, dry_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    check_probe_combination(warm, dry_mod, tmp_path, monkeypatch, backend="numpy")
