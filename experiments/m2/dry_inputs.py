"""Synthetic inputs for a dry run of experiments/m2/gpu_smoke.sh (no data/, no real subject, no label of anyone).

    python -B experiments/m2/dry_inputs.py --out DIR [--n 26]

Writes, under DIR (which must not exist):
  data/stage_L2R_val_v1/<pair>/{pair.npz, manifest.json}  a 26^3 synthetic pair (the plane-wave image of
      tests/test_bayesreg/solve/_synthetic.py, fixed = moving warped by a solenoidal field) with synthetic seg35 / seg4
      label maps (tests/test_bayesreg/eval/_synthetic.py labels), initial_u = 0; manifest in the mainline format
  dev_subjects.json   a synthetic allowlist: the two pair subjects (source synthetic_dry_run) and the synthetic
                      training subjects of the two synthetic fold models (source synthetic_dry_run_train)
  seg/<subject>/      Atropos-like tissue posteriors (three classes in an ellipsoid) with a provenance record bound to the
                      manifest's source hashes
  structure/<subject>/  (SMK2, the H arms) a structure-layer prediction as bayesreg.seg.unet.infer writes it:
                      structure_groups.npz (key "p", float16 (K_S + 1, nz, ny, nx), structure13_v1 grouping: background =
                      1 - a soft brain ellipsoid, the brain split among K_S smooth bumps) and provenance.json
                      (all_folds_mean of the two synthetic fold models; environment.code names this script)
  registry/segmenters.json, registry/ensemble.json  a staging registry of two synthetic fold models of unet35_v1_ens
                      (member sha256 are hashes of tags, label contact = the synthetic training subjects) and the
                      ensemble record beside it (registry sha256, member_check passed), so that run_pair admits it
                      with --structure-registry-staging (the dry run only; real runs use the repository registry)
gpu_smoke.sh --dry-run accepts only these inputs (GS review item 2): it checks the manifest note naming this script, the
allowlist source synthetic_dry_run and, for the H arms, the structure provenance naming this script; it refuses any
input path under data/ or the real segmentation root.
Subjects: OASIS_OAS1_0901_MR1 and OASIS_OAS1_0902_MR1; synthetic training subjects 0911-0918 (no real OASIS id is
above 0457). The stage directory is named stage_L2R_val_v1 only so that the smoke's --pairs-stage wiring is the real
one; it lies under DIR, not under data/.
Prints the gpu_smoke.sh options of the dry run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from jd_curl.three_d.nifti import file_sha256, save_scalar  # noqa: E402

from bayesreg.io import load_segmenter_registry  # noqa: E402
from bayesreg.io.pair import subject_list_sha256  # noqa: E402
from bayesreg.seg.structure_layer import ENSEMBLE_FILE, GROUPS_FILE, PROVENANCE_FILE, SEGMENTER_ID  # noqa: E402
from bayesreg.seg.unet.groups import load_grouping  # noqa: E402

from tests.test_bayesreg.eval._synthetic import labels, seg4_of  # noqa: E402
from tests.test_bayesreg.io_seg._helpers import source_sha  # noqa: E402
from tests.test_bayesreg.solve import _synthetic as syn  # noqa: E402

STAGE = "stage_L2R_val_v1"
FIXED, MOVING = "OASIS_OAS1_0901_MR1", "OASIS_OAS1_0902_MR1"
MOVING_SHIFT = (.7, -.5, .4)
FOLDS = {"unet35_v1_foldA": (0, [f"OASIS_OAS1_{n:04d}_MR1" for n in range(911, 915)]),
         "unet35_v1_foldB": (1, [f"OASIS_OAS1_{n:04d}_MR1" for n in range(915, 919)])}
MEMBERS = 2
SCRIPT = "experiments/m2/dry_inputs.py"


def write_json(path, document):
    path.write_text(json.dumps(document, indent=1)+"\n", encoding="utf-8")


def tag_sha(tag):
    return hashlib.sha256(tag.encode()).hexdigest()


def registry_models():
    """The two synthetic fold models of unet35_v1_ens (schema bayesreg.segmenters.v1 entries)."""
    return {model: dict(segmenter_id=SEGMENTER_ID, fold=fold,
                        member_model_sha256=[tag_sha(f"dry-{model}-{k}") for k in range(MEMBERS)],
                        label_contact_subjects=list(contact),
                        label_contact_subjects_sha256=subject_list_sha256(contact))
            for model, (fold, contact) in FOLDS.items()}


def structure_groups(grid, k, shift=(0., 0., 0.)):
    """(K_S + 1, nz, ny, nx): background = 1 - soft brain ellipsoid, the brain split among K_S smooth bumps (the
    construction of tests/test_bayesreg/solve/test_run_pair_structure.py)."""
    X = grid.identity()
    n = grid.shape[0]
    c = (n-1)/2
    centre = c+np.asarray(shift)
    brain = 1/(1+np.exp((np.linalg.norm((X-centre)/c, axis=-1)-.8)/.05))
    rng = np.random.default_rng(7)
    weights = np.stack([np.exp(-np.sum((X-centre-d)**2, axis=-1)/(2*(.3*c)**2))+1e-3
                        for d in rng.uniform(-.5, .5, (k, 3))*c])
    return np.concatenate([(1-brain)[None], brain*weights/weights.sum(axis=0)])


def write_structure(directory, grid, subject, models, grouping, shift):
    """A prediction directory as bayesreg.seg.unet.infer.segment_subject writes it (structure_groups.npz and
    provenance.json; all_folds_mean of every synthetic fold model)."""
    directory.mkdir(parents=True)
    np.savez_compressed(directory/GROUPS_FILE, p=structure_groups(grid, grouping.k, shift).astype(np.float16))
    model_ids = sorted(models)
    members = [dict(model_id=m, fold=models[m]["fold"], seed=k, member_sha256=sha, temperature=1.2, best_iteration=3)
               for m in model_ids for k, sha in enumerate(models[m]["member_model_sha256"])]
    write_json(directory/PROVENANCE_FILE, dict(
        layer="structure", segmenter_id=SEGMENTER_ID, model_ids=model_ids,
        model_sha256={m: list(models[m]["member_model_sha256"]) for m in model_ids},
        folds={m: models[m]["fold"] for m in model_ids}, trained_with_labels=True, test_labels_seen=False,
        subject=subject, policy="all_folds_mean", members=members, **grouping.record(),
        grouping_check=dict(grouping_status=grouping.status, grouping_status_admitted=True,
                            allow_unverified_grouping=False),
        input_sha256={"aligned_norm.nii.gz": source_sha(subject)},
        output_sha256={GROUPS_FILE: file_sha256(directory/GROUPS_FILE)},
        grid=dict(shape_zyx=list(grid.shape), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.]),
        registry_path="synthetic", registry_sha256="0"*64,
        environment=dict(code=dict(kind="synthetic", script=SCRIPT)), runtime_seconds=0.))


def write_registry(directory, models):
    """registry/segmenters.json (staging) and registry/ensemble.json beside it, as
    bayesreg.seg.unet.registry.register writes them (registry sha256, member_check passed)."""
    directory.mkdir(parents=True)
    path = directory/"segmenters.json"
    write_json(path, dict(schema="bayesreg.segmenters.v1", models=models))
    write_json(directory/ENSEMBLE_FILE, dict(
        format="bayesreg.unet35.ensemble.v1", segmenter_id=SEGMENTER_ID, registry_path=str(path),
        registry_sha256=load_segmenter_registry(path).sha256, fold_models=models, members=[],
        member_check=dict(format="bayesreg.unet35.member_check.v1", passed=True), note=f"synthetic ({SCRIPT})"))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=26)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    n = args.n
    grid = syn.cube_grid(n)
    moving = syn.moving_image(grid.identity())+.5
    fixed = syn.warp_with_spline(moving, grid, syn.solenoidal_displacement(grid, .6))
    fixed35 = labels(0, shape=grid.shape)
    moving35 = np.roll(labels(0, shape=grid.shape), 1, axis=2)
    name = f"{FIXED}__{MOVING}"
    directory = out/"data"/STAGE/name
    directory.mkdir(parents=True)
    np.savez_compressed(directory/"pair.npz", fixed=fixed, moving=moving, initial_u=np.zeros(grid.shape+(3,)),
                        fixed_seg35=fixed35, moving_seg35=moving35, fixed_seg4=seg4_of(fixed35),
                        moving_seg4=seg4_of(moving35))
    write_json(directory/"manifest.json", dict(
        fixed_subject=FIXED, moving_subject=MOVING, shape_zyx=list(grid.shape), spacing_xyz=[1., 1., 1.],
        origin_xyz=[0., 0., 0.], unit="mm", sources=dict(fixed=dict(sha256=source_sha(FIXED)),
                                                        moving=dict(sha256=source_sha(MOVING))),
        pair_sha256=file_sha256(directory/"pair.npz"), note=f"synthetic dry-run pair ({SCRIPT})"))
    train = [s for _, contact in FOLDS.values() for s in contact]
    write_json(out/"dev_subjects.json", dict(
        schema="bayesreg.dev_subjects.v1", subjects=dict(OASIS=[FIXED, MOVING]+train),
        sources=[dict(name="synthetic_dry_run", dataset="OASIS", path=SCRIPT, sha256="0"*64,
                      never_for_training=False, subjects=[FIXED, MOVING]),
                 dict(name="synthetic_dry_run_train", dataset="OASIS", path=SCRIPT, sha256="1"*64,
                      never_for_training=False, subjects=train)]))
    X = grid.identity()
    c = (n-1)/2
    files = [f"posteriors{k+1:02d}.nii.gz" for k in range(3)]
    for subject, shift in ((FIXED, (0., 0., 0.)), (MOVING, MOVING_SHIFT)):
        seg = out/"seg"/subject
        seg.mkdir(parents=True)
        brain = (np.linalg.norm((X-c-np.asarray(shift))/c, axis=-1) < .8).astype(float)
        for file, share in zip(files, (.2, .3, .5)):
            save_scalar(seg/file, share*brain, grid)
        write_json(seg/"provenance.json", dict(
            layer="tissue", segmenter_id="atropos", model_ids=["ANTs-2.6.5-synthetic-Atropos"], trained_with_labels=False,
            test_labels_seen=False, oracle=False, class_names=["CSF", "GM", "WM"], wm_index=2, posterior_files=files,
            posterior_file_index_base=1, input_sha256={"aligned_norm.nii.gz": source_sha(subject)},
            output_sha256={file: file_sha256(seg/file) for file in files}))
    models = registry_models()
    grouping = load_grouping()
    for subject, shift in ((FIXED, (0., 0., 0.)), (MOVING, MOVING_SHIFT)):
        write_structure(out/"structure"/subject, grid, subject, models, grouping, shift)
    registry = write_registry(out/"registry", models)
    print(f"--pair {STAGE}/{name} --data-root {out/'data'} --allowlist {out/'dev_subjects.json'} "
          f"--tissue-root {out/'seg'} --structure-root {out/'structure'} --segmenter-registry {registry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
