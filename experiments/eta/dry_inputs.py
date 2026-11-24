"""Synthetic inputs for a dry run of experiments/eta/sweep.py (no data/, no real subject, no label of anyone).

    python -B experiments/eta/dry_inputs.py --out DIR [--n 26]

Writes, under DIR (which must not exist):
  data/stage_T0_tune_v1/<pair>/{pair.npz, manifest.json}  four 26^3 synthetic pairs (the plane-wave image of
      tests/test_bayesreg/solve/_synthetic.py, fixed = moving warped by a solenoidal field of a per-pair amplitude) with
      synthetic seg35 / seg4 label maps; manifests in the mainline format naming this script
  data/stage_T0_tune_v1/selection.json  the four pairs with strata (moving larger x2, moving smaller, balanced), so that the
      sweep's pair rule picks 0901-0902, 0903-0904, 0905-0906 and leaves 0907-0908 out
  dev_subjects.json   a synthetic allowlist: the eight pair subjects (source synthetic_dry_run) and the synthetic training
                      subjects of the two synthetic fold models (source synthetic_dry_run_train)
  seg/<subject>/      Atropos-like tissue posteriors (three classes in an ellipsoid) with a provenance record bound to the
                      manifest's source hashes and checks.qc_min_class_fraction_ge_0.05 = true
  structure/<subject>/  (ETA2) a structure-layer prediction as bayesreg.seg.unet.infer writes it (structure_groups.npz,
                      float16 (K_S + 1, nz, ny, nx), structure13_v1 grouping; provenance.json with policy all_folds_mean
                      of the two synthetic fold models, environment.code naming this script)
  registry/segmenters.json, registry/ensemble.json  a staging registry of the two synthetic fold models of unet35_v1_ens
                      and the ensemble record beside it (registry sha256, member_check passed), admitted by run_pair and
                      the probe only with --structure-registry-staging (the sweep passes it in a dry run only)
The structure layer, the registry and the training subjects are those of experiments/m2/dry_inputs.py (its functions,
loaded from its file as a private module whose SCRIPT names this script). The stage directory is named stage_T0_tune_v1
only so that the sweep's wiring is the real one; it lies under DIR.
Subjects OASIS_OAS1_0901..0908_MR1, training subjects 0911..0918 (no real OASIS id is above 0457). Prints the sweep.py
options of the dry run.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from jd_curl.three_d.nifti import file_sha256, save_scalar  # noqa: E402

from bayesreg.seg.unet.groups import load_grouping  # noqa: E402

from tests.test_bayesreg.eval._synthetic import labels, seg4_of  # noqa: E402
from tests.test_bayesreg.io_seg._helpers import source_sha  # noqa: E402
from tests.test_bayesreg.solve import _synthetic as syn  # noqa: E402

STAGE = "stage_T0_tune_v1"
PAIRS = (("OASIS_OAS1_0901_MR1", "OASIS_OAS1_0902_MR1", "moving larger", .6),
         ("OASIS_OAS1_0903_MR1", "OASIS_OAS1_0904_MR1", "moving smaller", .5),
         ("OASIS_OAS1_0905_MR1", "OASIS_OAS1_0906_MR1", "balanced", .7),
         ("OASIS_OAS1_0907_MR1", "OASIS_OAS1_0908_MR1", "moving larger", .4))
SCRIPT = "experiments/eta/dry_inputs.py"
NOTE = f"synthetic dry-run pair ({SCRIPT})"
FIXED_SHIFT, MOVING_SHIFT = (0., 0., 0.), (.7, -.5, .4)


def write_json(path, document):
    path.write_text(json.dumps(document, indent=1)+"\n", encoding="utf-8")


def m2_inputs():
    """experiments/m2/dry_inputs.py as a private module object (not registered in sys.modules) whose SCRIPT names this
    script, so that the structure provenance and the ensemble note say where they come from."""
    spec = importlib.util.spec_from_file_location("_eta_m2_dry_inputs", REPO/"experiments"/"m2"/"dry_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SCRIPT = SCRIPT
    return module


def write_tissue(out, grid, subject, shift):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=26)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    m2 = m2_inputs()
    grid = syn.cube_grid(args.n)
    moving = syn.moving_image(grid.identity())+.5
    models, grouping = m2.registry_models(), load_grouping()
    rows, subjects = [], []
    for k, (fixed_subject, moving_subject, stratum, amplitude) in enumerate(PAIRS):
        fixed = syn.warp_with_spline(moving, grid, syn.solenoidal_displacement(grid, amplitude))
        fixed35 = labels(k, shape=grid.shape)
        moving35 = np.roll(labels(k, shape=grid.shape), 1, axis=2)
        name = f"{fixed_subject}__{moving_subject}"
        directory = out/"data"/STAGE/name
        directory.mkdir(parents=True)
        np.savez_compressed(directory/"pair.npz", fixed=fixed, moving=moving, initial_u=np.zeros(grid.shape+(3,)),
                            fixed_seg35=fixed35, moving_seg35=moving35, fixed_seg4=seg4_of(fixed35),
                            moving_seg4=seg4_of(moving35))
        write_json(directory/"manifest.json", dict(
            fixed_subject=fixed_subject, moving_subject=moving_subject, shape_zyx=list(grid.shape),
            spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit="mm",
            sources=dict(fixed=dict(sha256=source_sha(fixed_subject)), moving=dict(sha256=source_sha(moving_subject))),
            pair_sha256=file_sha256(directory/"pair.npz"), stratum=stratum, note=NOTE))
        rows.append(dict(pair=name, stratum=stratum))
        subjects += [fixed_subject, moving_subject]
        for subject, shift in ((fixed_subject, FIXED_SHIFT), (moving_subject, MOVING_SHIFT)):
            write_tissue(out, grid, subject, shift)
            m2.write_structure(out/"structure"/subject, grid, subject, models, grouping, shift)
    registry = m2.write_registry(out/"registry", models)
    write_json(out/"data"/STAGE/"selection.json", dict(rule=NOTE, pairs=rows))
    train = [s for _, contact in m2.FOLDS.values() for s in contact]
    write_json(out/"dev_subjects.json", dict(
        schema="bayesreg.dev_subjects.v1", subjects=dict(OASIS=subjects+train),
        sources=[dict(name="synthetic_dry_run", dataset="OASIS", path=SCRIPT, sha256="0"*64,
                      never_for_training=False, subjects=subjects),
                 dict(name="synthetic_dry_run_train", dataset="OASIS", path=SCRIPT, sha256="1"*64,
                      never_for_training=False, subjects=train)]))
    print(f"--data-root {out/'data'} --allowlist {out/'dev_subjects.json'} --tissue-root {out/'seg'} "
          f"--structure-root {out/'structure'} --segmenter-registry {registry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
