"""The M2 GPU smoke tools (experiments/m2; GS review items 2, 4 and 8), on fabricated records only (no data/, no unit):

- compare_gpu_smoke.py: the field_role column next to the status; only comparable rows (native_formal) sit next to
  A0, a handover_only arm and an arm without an evaluation are listed apart and marked; A0 comes from the
  bayesreg.eval.cli re-evaluation (eval_A0/done.json + one metrics.json) when present, else from
  experiments/d0/evaluate.py with a note saying so;
- gpu_smoke.sh --dry-run refuses missing or real inputs before anything is created; --memory-max must be a size.

Second batch (SMK2: H1-H3, structure channels, certificate cell estimate, C05-2a vs (A') beta_S 1 / 3):
- gpu_smoke.sh --show-args gives exactly the arm options of the work package (common flags, only the log-volume point
  set differs; OPT1 at defaults; the registered registry in real runs, the synthetic staging one only in a dry run);
  the H batch defaults to the v2 root (a per-pair root for another --pair) and the 22528 MiB GPU gate; G and H arms
  are never mixed; --segmenter-registry outside an H dry run, missing or real structure inputs and real reference
  roots in a dry run are refused before anything is created;
- compare_gpu_smoke.py with --ref-root / --ref-arms: the v1 rows marked (v1), the witness-type counts (sample point /
  edge / face / cell interior / none, W2, J <= 0 by point type; "未记录" for code before the record), the rejected
  candidates per level, the blocked levels (受阻收尾 set of the contract), rows of another pair listed apart;
- experiments/m2/dry_inputs.py writes a structure layer that run_pair.prepare_channels admits (staging registry).
"""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT/"experiments"/"m2"/"gpu_smoke.sh"
PAIR = "OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1"


def load_compare():
    spec = importlib.util.spec_from_file_location("compare_gpu_smoke", ROOT/"experiments"/"m2"/"compare_gpu_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


METRICS = dict(seg35=.80, seg4=.85, hd95=2.5, sdlogj_official=.07, sdlogj_brain=.11)


def fabricate(root, *, a0_eval):
    write(root/"a0_metrics.json", dict(schema="d0.metrics.v1", run_id="d0v1_A0", pair=PAIR, method="configC",
                                       status="native_budget_exhausted", metrics_role="formal_native_outer_commit",
                                       classification=dict(**{"class": "ok"}), native_commits=27,
                                       metrics=dict(METRICS, seg35=.79), certificate=dict(status="certified_feasible",
                                                                                         passed=True,
                                                                                         min_jd_lower_bound=.01),
                                       evaluator=dict(script="experiments/d0/evaluate.py")))
    if a0_eval:
        write(root/"eval_A0"/"done.json", dict(done_utc="x"))
        write(root/"eval_A0"/"d0v1_A0"/"metrics.json", dict(
            run_id="d0v1_A0", field_role="native_formal", field_present=True, run_status="native_budget_exhausted",
            metrics=dict(METRICS, seg35=.81), certificate=dict(mode="full", status="certified_feasible", passed=True,
                                                              min_jd_lower_bound=.02)))
    for arm, role, commits in (("G1", "native_formal", 3), ("G2", "handover_only", 0)):
        write(root/arm/"status.json", dict(status="budget_exhausted", wording="预算收尾", exit_code=0, native=True))
        write(root/arm/"config.json", dict(args=dict(pair=f"stage_L2R_val_v1/{PAIR}", channel_source="none",
                                                     logvol_points="gauss27", backend="torch64", device="cuda")))
        write(root/arm/"hyper.json", dict(rounds=[dict(level=3, round=0, status="budget_exhausted", accepted=commits,
                                                       commits=commits)]))
        write(root/arm/"timing.json", dict(total_seconds=2700.))
        write(root/f"eval_{arm}"/f"m2gpu_{arm}"/"metrics.json", dict(
            run_id=f"m2gpu_{arm}", field_role=role, field_present=True, metrics=METRICS,
            certificate=dict(mode="full", status="certified_feasible", passed=True, min_jd_lower_bound=.03)))
    write(root/"G3"/"status.json", dict(status="stopped_before_native", wording="x", exit_code=2, native=False))


def sections(text):
    near, _, apart = text.partition("## 不可比行")
    return near, apart


@pytest.mark.parametrize("a0_eval", [True, False], ids=["a0-reevaluated", "a0-d0-evaluate"])
def test_comparison_field_role_and_a0_source(tmp_path, a0_eval):
    compare = load_compare()
    fabricate(tmp_path, a0_eval=a0_eval)
    out_json, out_md = tmp_path/"comparison.json", tmp_path/"comparison.md"
    assert compare.main(["--root", str(tmp_path), "--arms", "G1,G2,G3", "--a0", str(tmp_path/"a0_metrics.json"),
                         "--out-json", str(out_json), "--out-md", str(out_md)]) == 0
    text = out_md.read_text(encoding="utf-8")
    header = next(line for line in text.splitlines() if line.startswith("| 臂"))
    cells = [c.strip() for c in header.strip("|").split("|")]
    assert cells.index("field_role") == cells.index("状态")+1
    near, apart = sections(text)
    assert "| A0 |" in near and "| G1 |" in near and "| G2 |" not in near and "| G3 |" not in near
    assert "| G2 |" in apart and "**handover_only" in apart and "| G3 |" in apart and "**未评测**" in apart
    document = json.loads(out_json.read_text(encoding="utf-8"))
    flags = {r["arm"]: r["comparable"] for r in [document["a0"]]+document["arms"]}
    assert flags == dict(A0=True, G1=True, G2=False, G3=False)
    a0 = document["a0"]
    assert a0["d0_evaluate"]["metrics"]["seg35"] == .79 and a0["field_role"] == "native_formal"
    if a0_eval:
        assert document["a0_source"] == "bayesreg.eval.cli" and a0["metrics"]["seg35"] == .81
        assert "bayesreg.eval.cli 重评" in text and a0["certificate"]["mode"] == "full"
    else:
        assert document["a0_source"] == "experiments/d0/evaluate.py" and a0["metrics"]["seg35"] == .79
        assert "A0 来自 experiments/d0/evaluate.py" in text and "done.json missing" in a0["fallback_reason"]
    with pytest.raises(SystemExit):                       # nothing is overwritten
        compare.main(["--root", str(tmp_path), "--a0", str(tmp_path/"a0_metrics.json"), "--out-json", str(out_json),
                      "--out-md", str(tmp_path/"other.md")])


def test_comparison_ignores_a_failed_a0_reevaluation(tmp_path):
    compare = load_compare()
    fabricate(tmp_path, a0_eval=True)
    write(tmp_path/"eval_A0"/"d0v1_A0"/"metrics.json", dict(run_id="d0v1_A0", field_present=True, metrics=None,
                                                            error="FieldError: input record differs"))
    record = compare.a0_record(tmp_path/"a0_metrics.json", tmp_path/"eval_A0")
    assert record["source"] == "experiments/d0/evaluate.py" and "evaluation error" in record["fallback_reason"]


def run_script(*args):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=60)


def test_dry_run_requires_synthetic_inputs(tmp_path):
    root = tmp_path/"smoke"
    synthetic = tmp_path/"dry"
    done = run_script("--dry-run", "--root", str(root))
    assert done.returncode == 2 and "are required" in done.stdout and not root.exists()
    for data_root in (ROOT/"data"/"three_d", Path("/home/rmcfz/code/python_projects/seq_vp_tag/data/three_d"),
                      Path("/home/rmcfz/code/python_projects/seq_vp_3d/data/three_d")):
        done = run_script("--dry-run", "--root", str(root), "--data-root", str(data_root), "--tissue-root",
                          str(synthetic/"seg"), "--allowlist", str(synthetic/"dev_subjects.json"))
        assert done.returncode == 2 and "real data" in done.stdout and not root.exists()
    done = run_script("--dry-run", "--root", str(root), "--data-root", str(synthetic/"data"), "--tissue-root",
                      "/home/rmcfz/seq_vp_tag_results/seg/atropos_v1", "--allowlist", str(synthetic/"dev_subjects.json"))
    assert done.returncode == 2 and "real data" in done.stdout and not root.exists()
    done = run_script("--dry-run", "--root", str(root), "--data-root", str(synthetic/"data"), "--tissue-root",
                      str(synthetic/"seg"), "--allowlist", str(synthetic/"dev_subjects.json"))
    assert done.returncode == 2 and "not a dry_inputs.py pair" in done.stdout and not root.exists()


@pytest.mark.parametrize("args", [("--memory-max", "10.5G"), ("--memory-max", "infinity"), ("--min-avail-gib", "8")])
def test_script_refuses_bad_options(tmp_path, args):
    done = run_script("--dry-run", "--root", str(tmp_path/"smoke"), *args)
    assert done.returncode == 2 and not (tmp_path/"smoke").exists()


# ---- SMK2: H arms ------------------------------------------------------------------------------------------------

COMMON = {"--backend": "torch64", "--device": "cuda", "--channel-source": "structure", "--metric-jd-scaling": "affine_v1",
          "--jd-cell-estimate": "certificate", "--outer": "a", "--rounds": "1,1,1,1", "--wall-seconds": "2700",
          "--structure-infer": "never"}
POINTS = {"H1": ("gauss27+corners", None), "H2": ("gauss27+closed27", "1"), "H3": ("gauss27+closed27", "3")}


def shown(stdout):
    """{arm: {option: value or True}} of gpu_smoke.sh --show-args, and the summary line."""
    out, summary = {}, None
    for line in stdout.splitlines():
        if line.startswith("root "):
            summary = line
            continue
        arm, _, command = line.partition(": ")
        tokens = command.split()
        options, k = {}, tokens.index("bayesreg.cli.run_pair")+1
        while k < len(tokens):
            if k+1 < len(tokens) and not tokens[k+1].startswith("--"):
                options[tokens[k]] = tokens[k+1]
                k += 2
            else:
                options[tokens[k]] = True
                k += 1
        out[arm] = options
    return out, summary


def test_show_args_h_arms_match_the_work_package():
    done = run_script("--arms", "H1,H2,H3", "--show-args")
    assert done.returncode == 0, done.stdout
    arms, summary = shown(done.stdout)
    assert sorted(arms) == ["H1", "H2", "H3"]
    for arm, options in arms.items():
        for flag, value in COMMON.items():
            assert options[flag] == value, (arm, flag)
        points, share = POINTS[arm]
        assert options["--logvol-points"] == points
        assert options.get("--logvol-closed27-share") == share
        assert options.get("--closed27-weights") == (None if share is None else "simpson")
        assert options["--pair"] == f"stage_L2R_val_v1/{PAIR}"
        assert options["--out"] == f"/home/rmcfz/seq_vp_tag_results/m2_gpu_smoke_v2/{arm}"
        assert options["--structure-root"] == "/home/rmcfz/seq_vp_tag_results/seg/unet35_v1/pred"
        assert options["--tissue-root"] == "/home/rmcfz/seq_vp_tag_results/seg/atropos_v1"        # tau^2, mask, norm
        for absent in ("--segmenter-registry", "--structure-registry-staging", "--jacobian-cache",
                       "--jacobian-cache-log", "--speculative-gradient"):
            assert absent not in options                                                    # OPT1 defaults, registered
    differing = {flag for flag in set().union(*arms.values())
                 if len({str(arms[a].get(flag)) for a in arms}) > 1}
    assert differing == {"--out", "--logvol-points", "--logvol-closed27-share", "--closed27-weights"}
    assert "m2_gpu_smoke_v2;" in summary and "22528 MiB" in summary and "(G2 G3)" in summary
    g_arms, g_summary = shown(run_script("--show-args").stdout)                             # the G batch is unchanged
    # DEF (2026-10-31): the G arms pass the cell estimate they ran with (gauss27+corners, run_pair's default then)
    # explicitly, since run_pair's default is certificate; nothing else of theirs changed
    assert sorted(g_arms) == ["G1", "G2", "G3"]
    assert all(g_arms[arm]["--jd-cell-estimate"] == "gauss27+corners" for arm in g_arms)
    assert [g_arms[arm]["--logvol-points"] for arm in ("G1", "G2", "G3")] == ["gauss27", "gauss27+corners",
                                                                            "gauss27+corners"]
    assert g_arms["G3"]["--out"] == "/home/rmcfz/seq_vp_tag_results/m2_gpu_smoke/G3" and "9216 MiB" in g_summary
    other = "OASIS_OAS1_0440_MR1__OASIS_OAS1_0441_MR1"
    o_arms, _ = shown(run_script("--arms", "H2", "--pair", f"stage_L2R_val_v1/{other}", "--show-args").stdout)
    assert o_arms["H2"]["--out"] == "/home/rmcfz/seq_vp_tag_results/m2_gpu_smoke_v2_0440-0441/H2"
    assert o_arms["H2"]["--pair"] == f"stage_L2R_val_v1/{other}"


def test_script_refuses_mixed_batches_and_bad_h_inputs(tmp_path):
    root = tmp_path/"smoke"
    for args in (("--arms", "G2,H1"), ("--arms", "H4"), ("--arms", "H1", "--segmenter-registry", str(tmp_path/"r.json")),
                 ("--arms", "G1", "--dry-run", "--segmenter-registry", str(tmp_path/"r.json")),
                 ("--arms", "H1", "--ref-arms", "H2")):
        done = run_script("--root", str(root), *args, "--show-args")
        assert done.returncode == 2 and not root.exists(), (args, done.stdout)
    synthetic = tmp_path/"dry"
    base = ("--dry-run", "--arms", "H1,H2,H3", "--root", str(root), "--data-root", str(synthetic/"data"),
            "--tissue-root", str(synthetic/"seg"), "--allowlist", str(synthetic/"dev_subjects.json"))
    done = run_script(*base)
    assert done.returncode == 2 and "--structure-root and --segmenter-registry" in done.stdout and not root.exists()
    for structure in ("/home/rmcfz/seq_vp_tag_results/seg/unet35_v1/pred", str(ROOT/"data"/"three_d")):
        done = run_script(*base, "--structure-root", structure, "--segmenter-registry", str(synthetic/"r.json"),
                          "--ref-root", str(tmp_path/"v1"))
        assert done.returncode == 2 and "real data" in done.stdout and not root.exists()
    for real in ("/home/rmcfz/seq_vp_tag_results/m2_gpu_smoke", "/home/rmcfz/seq_vp_tag_results/m2_gpu_smoke_v2"):
        done = run_script(*base, "--structure-root", str(synthetic/"structure"), "--segmenter-registry",
                          str(synthetic/"r.json"), "--ref-root", real)
        assert done.returncode == 2 and "must lie outside" in done.stdout and not root.exists()
        done = run_script(*base[:3], "--root", real+"_dry", *base[5:], "--structure-root", str(synthetic/"structure"),
                          "--segmenter-registry", str(synthetic/"r.json"), "--ref-root", str(tmp_path/"v1"))
        assert done.returncode == 2 and "must lie outside" in done.stdout
        assert not Path(real+"_dry").exists()


def load_dry_inputs():
    spec = importlib.util.spec_from_file_location("m2_dry_inputs", ROOT/"experiments"/"m2"/"dry_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_inputs_structure_layer_is_admitted(tmp_path, capsys):
    import logging

    from bayesreg.cli import run_pair
    from bayesreg.io import load_pair
    from bayesreg.seg.structure_layer import STAGING, StructureSource

    dry = load_dry_inputs()
    out = tmp_path/"dry"
    assert dry.main(["--out", str(out)]) == 0                    # 26^3: four levels need >= 5 nodes per axis
    printed = capsys.readouterr().out.split()
    options = dict(zip(printed[::2], printed[1::2]))
    assert options["--structure-root"] == str(out/"structure")
    assert options["--segmenter-registry"] == str(out/"registry"/"segmenters.json")
    pair = load_pair(run_pair.resolve_pair(out/"data", options["--pair"]), allowlist=out/"dev_subjects.json")
    source = StructureSource(root=out/"structure", registry_path=out/"registry"/"segmenters.json",
                             allowlist_path=out/"dev_subjects.json", models_root=out/"no_models",
                             image_root=out/"no_images", allow_staging=True)
    inputs, _, _, prov = run_pair.prepare_channels(pair, "structure", out/"seg", logging.getLogger("smk2"),
                                                   structure=source)
    assert prov["K_S"] == 13 and prov["K_T"] == 3 and inputs[-1].fixed.shape[0] == 14
    s = prov["structure"]
    assert s["registry"]["status"] == STAGING
    for side in ("fixed", "moving"):
        assert s[side]["models"]["policy"] == "all_folds_mean" and s[side]["accepted"]["test_labels_seen"] is False
    root = tmp_path/"smoke"                         # the script admits these inputs (checks only: --show-args)
    done = run_script("--dry-run", "--arms", "H1,H2,H3", "--root", str(root), "--ref-root", str(tmp_path/"v1"),
                      *printed, "--show-args")
    assert done.returncode == 0, done.stdout
    arms, _ = shown(done.stdout)
    for arm in ("H1", "H2", "H3"):
        assert arms[arm]["--segmenter-registry"] == options["--segmenter-registry"]
        assert arms[arm]["--structure-registry-staging"] is True and arms[arm]["--device"] == "cpu"
        assert arms[arm]["--cpu-threads"] == "1" and arms[arm]["--out"] == str(root/arm)
    assert not root.exists()


def candidate(level, iteration, reason="JACOBIAN_VIOLATION", **fields):
    """A log.jsonl candidate row (option (a): no outer index)."""
    return dict(level=level, round=0, kind="candidate", objective="F_B", iteration=iteration, attempt=0,
                accepted=reason is None, reason=reason, **fields)


def fabricate_arm(root, arm, *, role, source, points, levels, witness=None, pair=PAIR, share=None, registry=None,
                  log_rows=()):
    """A finished run directory and its evaluation; levels: [(level, status, accepted, rejections dict)]."""
    if log_rows:
        (root/arm).mkdir(parents=True, exist_ok=True)
        (root/arm/"log.jsonl").write_text("".join(json.dumps(r)+"\n" for r in log_rows), encoding="utf-8")
    status = dict(status="budget_exhausted", wording="预算收尾", exit_code=0, native=True)
    if witness is not None:
        status["witness_summary"] = witness
    if registry is not None:
        status["structure_registry"] = dict(status=registry, path="configs/bayesreg/segmenters.json",
                                            git=dict(tracked=True, clean=True))
    write(root/arm/"status.json", status)
    write(root/arm/"config.json", dict(args=dict(
        pair=f"stage_L2R_val_v1/{pair}", channel_source=source, logvol_points=points,
        logvol_closed27_share=1. if share is None else share, closed27_weights="simpson",     # argparse defaults
        jd_cell_estimate="certificate" if source == "structure" else "gauss27+corners", backend="torch64",
        device="cuda")))
    write(root/arm/"hyper.json", dict(rounds=[dict(level=lv, round=0, status=st, accepted=acc, commits=acc,
                                                   counters=dict(candidates=acc+sum(rej.values()), rejections=rej))
                                              for lv, st, acc, rej in levels]))
    write(root/arm/"timing.json", dict(total_seconds=2700.))
    write(root/f"eval_{arm}"/f"m2gpu_{arm}"/"metrics.json", dict(
        run_id=f"m2gpu_{arm}", field_role=role, field_present=True, metrics=METRICS,
        certificate=dict(mode="full", status="certified_feasible", passed=True, min_jd_lower_bound=.03)))


def test_comparison_h_arms_with_v1_reference(tmp_path):
    compare = load_compare()
    v1, v2 = tmp_path/"v1", tmp_path/"v2"
    fabricate(v1, a0_eval=True)                         # A0 and its re-evaluation live in v1 (G2 there: handover_only)
    fabricate_arm(v1, "G3", role="native_formal", source="tissue", points="gauss27+corners",
                  levels=[(2, "line_search_failed", 156, {"JACOBIAN_VIOLATION": 279, "FEASIBILITY_UNCERTAIN": 5}),
                          (3, "line_search_failed", 35, {"JACOBIAN_VIOLATION": 264})],
                  log_rows=[candidate(2, 155), candidate(2, 156), candidate(2, 156)])     # v1: no witness fields
    witness = dict(certificates=500, jacobian_violations=40,
                   witness_type_counts={"corner": 3, "edge_mid": 4, "edge_interior": 20, "face_interior": 6,
                                        "cell_interior": 2, "None": 5},
                   min_j53_positive=7, min_j53_nonpositive_point_type_counts={"edge_mid": 4, "gauss27": 2})
    log_rows = [candidate(2, 79, witness_type="edge_interior", witness_cell_min_j53=dict(J=-.1, point_type="edge_mid")),
                candidate(2, 80, witness_type="edge_interior", witness_cell_min_j53=dict(J=.02, point_type="gauss27")),
                candidate(2, 80, witness_type="face_interior", witness_cell_min_j53=dict(J=.03, point_type="corner")),
                candidate(2, 80, witness_type=None, witness_cell_min_j53=None),
                candidate(2, 80, reason=None, witness_type=None, witness_cell_min_j53=None),
                dict(level=2, round=0, kind="accept", iteration=80, reason="JACOBIAN_VIOLATION")]
    fabricate_arm(v2, "H1", role="native_formal", source="structure", points="gauss27+corners", witness=witness,
                  registry="registered", log_rows=log_rows,
                  levels=[(0, "converged_gradient", 50, {"NO_STRICT_DECREASE": 3}), (1, "cap_reached", 500, {}),
                          (2, "outer_no_progress", 80, {"JACOBIAN_VIOLATION": 12, "ARMIJO_REJECTED": 1}),
                          (3, "exception", 0, {})])
    fabricate_arm(v2, "H2", role="native_formal", source="structure", points="gauss27+closed27", share=1,
                  witness=dict(certificates=10, jacobian_violations=0, witness_type_counts={}), registry="registered",
                  levels=[(3, "budget_exhausted", 90, {})], pair="OASIS_OAS1_0440_MR1__OASIS_OAS1_0441_MR1")
    out_json, out_md = v2/"comparison.json", v2/"comparison.md"
    assert compare.main(["--root", str(v2), "--arms", "H1,H2,H3", "--ref-root", str(v1), "--ref-arms", "G2,G3",
                         "--a0", str(v1/"a0_metrics.json"), "--a0-eval", str(v1/"eval_A0"),
                         "--out-json", str(out_json), "--out-md", str(out_md)]) == 0
    text = out_md.read_text(encoding="utf-8")
    header = [c.strip() for c in next(line for line in text.splitlines() if line.startswith("| 臂")).strip("|").split("|")]
    for column in ("受阻收尾的层", "各层被拒候选（JV）", "见证点 采样点/棱内/面内/体内", "W2 与 53 点 J ≤ 0", "seg35", "seg4"):
        assert column in header
    near, apart = sections(text)
    assert "| A0 |" in near and "| G3 (v1) |" in near and "| H1 |" in near
    assert "| G2 (v1) |" in apart and "| H2 |" in apart and "| H3 |" in apart       # handover / other pair / missing
    document = json.loads(out_json.read_text(encoding="utf-8"))
    assert document["schema"].endswith(".v3") and document["a0_source"] == "bayesreg.eval.cli"
    assert set(document["blocked_statuses"]) == {"line_search_failed", "trial_budget_exhausted", "outer_no_progress",
                                                 "outer_inner_failed"}
    refs = {r["label"]: r for r in document["reference_arms"]}
    arms = {r["arm"]: r for r in document["arms"]}
    assert refs["G3 (v1)"]["witness"] is None and refs["G3 (v1)"]["reference"] is True
    assert refs["G3 (v1)"]["rejections_by_level"]["2"] == dict(total=284, jacobian_violation=279, by_reason={
        "JACOBIAN_VIOLATION": 279, "FEASIBILITY_UNCERTAIN": 5})
    assert [e["level"] for e in refs["G3 (v1)"]["blocked_levels"]] == [2, 3]
    h1 = arms["H1"]
    assert h1["witness"]["sample_point"] == 7 and h1["witness"]["edge_interior"] == 20
    assert h1["witness"]["face_interior"] == 6 and h1["witness"]["cell_interior"] == 2
    assert h1["witness"]["no_witness"] == 5 and h1["witness"]["w2_min53_positive"] == 7
    assert [e["status"] for e in h1["blocked_levels"]] == ["outer_no_progress"]
    assert [e["status"] for e in h1["other_abnormal_levels"]] == ["exception"]
    last = h1["blocked_levels"][0]["last_iteration_witness"]              # log.jsonl: iteration 80 of level 2
    assert last["jacobian_violations"] == 3 and last["w2_min53_positive"] == 2 and last["outer_iteration"] == [0, 80]
    assert last["types"] == {"edge_interior": 1, "face_interior": 1, "None": 1}
    assert h1["witness_by_level"]["2"]["jacobian_violations"] == 4
    assert h1["witness_by_level"]["2"]["min53_nonpositive_by_point_type"] == {"edge_mid": 1}
    g3_last = refs["G3 (v1)"]["blocked_levels"]
    assert g3_last[0]["last_iteration_witness"]["recorded"] == 0 and g3_last[1]["last_iteration_witness"] is None
    assert h1["options"]["jd_cell_estimate"] == "certificate" and h1["structure_registry"]["status"] == "registered"
    assert h1["options"]["logvol_closed27_share"] is None and arms["H2"]["options"]["logvol_closed27_share"] == 1
    assert arms["H2"]["comparable"] is False and document["pair_mismatch"] == ["OASIS_OAS1_0440_MR1__OASIS_OAS1_0441_MR1"]
    h1_line = next(line for line in near.splitlines() if line.startswith("| H1 |"))
    assert "7（角 3·棱中 4·面心 0·体心 0） / 20 / 6 / 2（无 5；棱上 24）" in h1_line
    assert "W2 7" in h1_line and "edge_mid 4" in h1_line and h1["witness"]["on_edge"] == 24
    assert "L2 13（12）" in h1_line and "另：L3（exception）" in h1_line
    assert ("L2（outer_no_progress；末次迭代 JV 3：None 1, edge_interior 1, face_interior 1；W2 2；53 点 J ≤ 0：无）"
            in h1_line)
    assert "结构层登记 registered" in h1_line and "JD 估计 certificate" in h1_line
    g3_line = next(line for line in near.splitlines() if line.startswith("| G3 (v1) |"))
    assert "未记录" in g3_line and "L2 284（279）; L3 264（264）" in g3_line
    assert "L2（line_search_failed；末次迭代 JV 2，见证点未记录）" in g3_line
    assert "L3（line_search_failed；末次迭代无 JV 记录）" in g3_line
    h2_line = next(line for line in apart.splitlines() if line.startswith("| H2 |"))
    assert "β=1 simpson" in h2_line and "0（角 0·棱中 0·面心 0·体心 0） / 0 / 0 / 0（无 0；棱上 0）" in h2_line
    assert "配对" in text
