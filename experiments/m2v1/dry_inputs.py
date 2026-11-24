"""Synthetic inputs for a dry run of experiments/m2v1/run_batch.py (no data/, no real subject, no label of anyone, no real
result file).

    python -B experiments/m2v1/dry_inputs.py --out DIR [--n 26] [--wall-seconds 120] [--extra "--max-accepted 2"]
        [--sweep-outcome selected|stop|running] [--code-sha SHA]

Writes, under DIR (which must not exist):
  data/stage_L2R_val_v1/<pair>/{pair.npz, manifest.json}  three chained 26^3 synthetic pairs in the Learn2Reg pattern
      (fixed n <- moving n + 1: 0901 <- 0902, 0902 <- 0903, 0903 <- 0904; the plane-wave image of
      tests/test_bayesreg/solve/_synthetic.py warped by a solenoidal field) with synthetic seg35 / seg4 label maps;
      manifests naming this script; data/stage_L2R_val_v1/selection.json lists them
  dev_subjects.json   synthetic allowlist (source synthetic_dry_run: the four pair subjects; synthetic_dry_run_train: the
                      synthetic training subjects 0911-0918 of the two synthetic fold models)
  seg/<subject>/      Atropos-like tissue posteriors with provenance (QC flag true)
  structure/<subject>/, registry/  the structure layer and the staging registry of experiments/m2/dry_inputs.py (its
                      functions, loaded as a private module whose SCRIPT names this script)
  d0/long.csv         synthetic A0 rows (method configC_frozen_eb808be_90s) for the three pairs, in the D0 long format
  a4/long.csv         the same A0 rows and synthetic A4p rows (the A4' long format)
  ref_reeval/         a synthetic re-evaluation of A0 and A4p in the layout of experiments/m2v1/reeval_refs.py, with
                      the long.csv numbers
  sweep/              a synthetic eta sweep root: eta_selected.json and summary.json whose configuration is produced by
                      experiments/eta/sweep.py's own Sweep.configuration() for the dry-run options of this batch (wall
                      seconds and extra as given; structure root and staging registry above), stage_set
                      stage_T0_tune_v1 with synthetic pairs 0905-0910 (disjoint from the batch); code_sha --code-sha
                      (default the repository HEAD); sweep.log ending with "end" (--sweep-outcome selected), a STOP line
                      and no eta_selected.json (stop), or a gate line and no eta_selected.json (running)
The smoke reference is absent (--smoke-root DIR/smoke does not exist: the D47 table shows it as missing).
Prints the run_batch.py options of the dry run.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from jd_curl.three_d.nifti import file_sha256, save_scalar  # noqa: E402

from bayesreg.seg.unet.groups import load_grouping  # noqa: E402

from tests.test_bayesreg.eval._synthetic import labels, seg4_of  # noqa: E402
from tests.test_bayesreg.io_seg._helpers import source_sha  # noqa: E402
from tests.test_bayesreg.solve import _synthetic as syn  # noqa: E402

STAGE = "stage_L2R_val_v1"
SUBJECTS = [f"OASIS_OAS1_{n:04d}_MR1" for n in range(901, 905)]
PAIRS = [(SUBJECTS[k], SUBJECTS[k+1], amp) for k, amp in zip(range(3), (.6, .5, .7))]
SWEEP_PAIRS = [f"OASIS_OAS1_{a:04d}_MR1__OASIS_OAS1_{a+1:04d}_MR1" for a in (905, 907, 909)]
SCRIPT = "experiments/m2v1/dry_inputs.py"
NOTE = f"synthetic dry-run pair ({SCRIPT})"
# every subject that is a fixed image somewhere keeps the unshifted brain (run_pair checks the fixed mask against the
# image, C03); only the last subject (moving only) is shifted, as the moving subject of experiments/eta/dry_inputs.py
SHIFTS = [(0., 0., 0.), (0., 0., 0.), (0., 0., 0.), (.7, -.5, .4)]
ETA = dict(alpha=.01, nu=50., tau=[1., .5, .5, .5])
REF_METHOD = "configC_frozen_eb808be_90s"
LONG = ("set", "split", "method", "arm", "pair", "measure", "value", "run_id")


def _load(name, rel, script=None):
    spec = importlib.util.spec_from_file_location(name, REPO/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if script is not None:
        module.SCRIPT = script
    return module


def write_json(path, document):
    path.write_text(json.dumps(document, indent=1)+"\n", encoding="utf-8")


def write_tissue(out, grid, subject, shift):
    """experiments/eta/dry_inputs.py write_tissue (its record, this script's note)."""
    n = grid.shape[0]
    X = grid.identity()
    c = (n-1)/2
    files = [f"posteriors{k+1:02d}.nii.gz" for k in range(3)]
    seg = out/"seg"/subject
    seg.mkdir(parents=True)
    brain = (np.linalg.norm((X-c-np.asarray(shift))/c, axis=-1) < .8).astype(float)
    for file, share in zip(files, (.2, .3, .5)):
        save_scalar(seg/file, share*brain, grid)
    write_json(seg/"provenance.json", dict(
        layer="tissue", segmenter_id="atropos", model_ids=["ANTs-2.6.5-synthetic-Atropos"], trained_with_labels=False,
        test_labels_seen=False, oracle=False, class_names=["CSF", "GM", "WM"], wm_index=2, posterior_files=files,
        posterior_file_index_base=1, input_sha256={"aligned_norm.nii.gz": source_sha(subject)},
        output_sha256={file: file_sha256(seg/file) for file in files},
        checks={"qc_min_class_fraction_ge_0.05": True}, note=NOTE))


def write_long(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(LONG)
        writer.writerows(rows)


def reference_rows(arm, offset):
    rows = []
    for k, (fixed, moving, _) in enumerate(PAIRS):
        pair = f"{fixed}__{moving}"
        rid = f"dry_{arm}_{k}"
        values = dict(seg35=.5+offset+.01*k, seg4=.6+offset+.01*k, hd95=3.+k, sdlogj_official=.05, sdlogj_brain=.3)
        rows += [("L2R_val", "dev", REF_METHOD, arm, pair, m, repr(v), rid) for m, v in values.items()]
    return rows


def write_reeval(out, rows):
    """A synthetic re-evaluation in the layout of experiments/m2v1/reeval_refs.py (<arm>/<short>/done.json and
    <run_id>/metrics.json with pair, run id, field role, metrics, certificate), with the long.csv numbers."""
    records = {}
    for _, _, _, arm, pair, measure, value, rid in rows:
        records.setdefault((arm, pair, rid), {})[measure] = float(value)
    for (arm, pair, rid), metrics in records.items():
        fixed, _, moving = pair.partition("__")
        ev = out/arm/f"{fixed.split('_')[2]}-{moving.split('_')[2]}"
        (ev/rid).mkdir(parents=True)
        write_json(ev/rid/"metrics.json", dict(
            pair=pair, run_id=rid, method=REF_METHOD, arm=arm, field_role="native_formal" if arm == "A0" else "disp_only",
            metrics=metrics, certificate=dict(mode="full", status="certified_feasible", passed=True, min_jd_lower_bound=.1),
            note=NOTE))
        write_json(ev/"done.json", dict(arm=arm, pair=pair, run_id=rid, metrics=[str(ev/rid/"metrics.json")],
                                        route="configC" if arm == "A0" else "disp_from_final_native", note=NOTE))


def head_sha():
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def write_sweep(out, *, outcome, wall_seconds, extra, code_sha, eta=None):
    """A synthetic eta sweep root at out/sweep (module docstring). The configuration is the eta sweep's own
    Sweep.configuration() under the dry-run options."""
    sweep = out/"sweep"
    sweep.mkdir()
    sweep_module = _load("_m2v1_dry_eta_sweep", "experiments/eta/sweep.py")
    opts = sweep_module.parse_args(["--root", str(sweep), "--dry-run", "--data-root", str(out/"data"), "--allowlist",
                                    str(out/"dev_subjects.json"), "--tissue-root", str(out/"seg"), "--structure-root",
                                    str(out/"structure"), "--segmenter-registry", str(out/"registry"/"segmenters.json"),
                                    "--wall-seconds", str(wall_seconds), "--extra", extra])
    sw = sweep_module.Sweep(opts)
    sw.check_registry()                                   # writes its line to sweep/sweep.log
    configuration = sw.configuration()
    eta = dict(eta or ETA)
    path = [dict(stage=s, chosen="x1", nu=eta["nu"], s_tau=1.) for s in ("s1", "s2", "s3")]
    with open(sweep/"sweep.log", "a", encoding="utf-8") as stream:
        if outcome == "selected":
            write_json(sweep/"eta_selected.json", dict(eta, use="synthetic", selection_path=path, nu_star=eta["nu"],
                                                       configuration=configuration, code_sha=code_sha))
            write_json(sweep/"summary.json", dict(note=NOTE, stage_set="stage_T0_tune_v1", pairs=SWEEP_PAIRS,
                                                  eta_selected=eta, configuration=configuration, code_sha=code_sha))
            stream.write("2026-10-31T21:00:00Z selected eta (synthetic)\n2026-10-31T21:00:00Z end\n")
        elif outcome == "stop":
            stream.write("2026-10-31T21:00:00Z STOP: stage s1: no eligible candidate (synthetic)\n")
        else:
            stream.write("2026-10-31T21:00:00Z gate run s2-x4-0905-0906: (synthetic)\n")
    return sweep


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=26)
    parser.add_argument("--wall-seconds", type=float, default=120.)
    parser.add_argument("--extra", default="--max-accepted 2")
    parser.add_argument("--sweep-outcome", choices=("selected", "stop", "running"), default="selected")
    parser.add_argument("--code-sha", default=None)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    m2 = _load("_m2v1_m2_dry_inputs", "experiments/m2/dry_inputs.py", script=SCRIPT)
    grid = syn.cube_grid(args.n)
    base = syn.moving_image(grid.identity())+.5
    models, grouping = m2.registry_models(), load_grouping()
    rows = []
    for k, (fixed_subject, moving_subject, amplitude) in enumerate(PAIRS):
        fixed = syn.warp_with_spline(base, grid, syn.solenoidal_displacement(grid, amplitude))
        fixed35 = labels(k, shape=grid.shape)
        moving35 = np.roll(labels(k, shape=grid.shape), 1, axis=2)
        name = f"{fixed_subject}__{moving_subject}"
        directory = out/"data"/STAGE/name
        directory.mkdir(parents=True)
        np.savez_compressed(directory/"pair.npz", fixed=fixed, moving=base, initial_u=np.zeros(grid.shape+(3,)),
                            fixed_seg35=fixed35, moving_seg35=moving35, fixed_seg4=seg4_of(fixed35),
                            moving_seg4=seg4_of(moving35))
        write_json(directory/"manifest.json", dict(
            fixed_subject=fixed_subject, moving_subject=moving_subject, shape_zyx=list(grid.shape),
            spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit="mm",
            sources=dict(fixed=dict(sha256=source_sha(fixed_subject)), moving=dict(sha256=source_sha(moving_subject))),
            pair_sha256=file_sha256(directory/"pair.npz"), note=NOTE))
        rows.append(dict(pair=name))
    for subject, shift in zip(SUBJECTS, SHIFTS):
        write_tissue(out, grid, subject, shift)
        m2.write_structure(out/"structure"/subject, grid, subject, models, grouping, shift)
    registry = m2.write_registry(out/"registry", models)
    write_json(out/"data"/STAGE/"selection.json", dict(rule=NOTE, pairs=rows))
    train = [s for _, contact in m2.FOLDS.values() for s in contact]
    write_json(out/"dev_subjects.json", dict(
        schema="bayesreg.dev_subjects.v1", subjects=dict(OASIS=SUBJECTS+train),
        sources=[dict(name="synthetic_dry_run", dataset="OASIS", path=SCRIPT, sha256="0"*64,
                      never_for_training=False, subjects=SUBJECTS),
                 dict(name="synthetic_dry_run_train", dataset="OASIS", path=SCRIPT, sha256="1"*64,
                      never_for_training=False, subjects=train)]))
    (out/"d0").mkdir()
    (out/"a4").mkdir()
    write_long(out/"d0"/"long.csv", reference_rows("A0", 0.))
    write_long(out/"a4"/"long.csv", reference_rows("A0", 0.)+reference_rows("A4p", .005))
    write_reeval(out/"ref_reeval", reference_rows("A0", 0.)+reference_rows("A4p", .005))
    write_sweep(out, outcome=args.sweep_outcome, wall_seconds=args.wall_seconds, extra=args.extra,
                code_sha=args.code_sha or head_sha())
    print(f"--data-root {out/'data'} --allowlist {out/'dev_subjects.json'} --tissue-root {out/'seg'} "
          f"--structure-root {out/'structure'} --segmenter-registry {registry} --sweep-root {out/'sweep'} "
          f"--d0-long {out/'d0'/'long.csv'} --a4-long {out/'a4'/'long.csv'} --smoke-root {out/'smoke'} "
          f"--ref-reeval {out/'ref_reeval'} "
          f"--d47-pair {PAIRS[0][0]}__{PAIRS[0][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
