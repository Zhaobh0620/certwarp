"""D0: Atropos three-class tissue posteriors (contract C03.1 tissue layer P_T) for the 20 Learn2Reg validation subjects.

Input: each subject's T1 is read with bayesreg.io.pair.load_pair (label firewall: only `fixed`/`moving` are decoded; subject
guard: configs/bayesreg/dev_subjects.json) from a stage_L2R_val_v1 pair in which it appears; when a subject appears in two
pairs, both arrays must be bitwise identical. The array (RAS storage, zyx) is written as a float32 NIfTI with the manifest
grid (RAS affine, 1 mm); mask = image > 0. No label is read by this script.

Atropos (ANTs 2.6.5-gfdce4d2), fixed commands (Atropos defaults except a constant seed, -r 0):
  ImageMath 3 t1_trunc.nii.gz TruncateImageIntensity t1.nii.gz 0.01 0.99 256 mask.nii.gz
  Atropos -d 3 -a t1_trunc.nii.gz -x mask.nii.gz -i KMeans[3] -k Gaussian -c [5,0.001] -m [0.3,1x1x1] -r 0
          -o [labels.nii.gz,posteriors%02d.nii.gz]
  ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = 8.
  Why the truncation (decided 2026-10-30 on OASIS_OAS1_0438_MR1 before any comparison with labels, by a label-free
  criterion): without it the kmeans initialization puts one centre in the sparse bright tail (intensities .43-.86,
  ~0.25% of the mask) and that class keeps 0.28% of the voxels (trial v0); Atropos' own BoxPlot winsorizing (-w) did not
  change this (v1, 0.21%); truncation at the 1%/99% in-mask quantiles gave proportions .23/.43/.34 (v2). The trials are
  kept in <out-root>/../atropos_trials/. QC (label-free, recorded and enforced): every class keeps >= 5% of the mask.
Class order: Atropos orders kmeans classes by ascending intensity (class 1 = darkest). The script checks this with the
posterior-weighted mean intensity and names the classes by that rank: CSF < GM < WM (file posteriors0k <-> class_names[k-1]).
provenance.json follows contract C03.1-2/3 and configs/bayesreg/README.md and is checked with
bayesreg.io.pair.validate_segmentation_provenance (layer tissue) and verify_segmentation_files.

Output: <out-root>/<canonical id>/{t1.nii.gz, mask.nii.gz, labels.nii.gz, posteriors01..03.nii.gz, atropos.log, provenance.json}.
An existing subject directory is never overwritten (the subject is skipped if its provenance.json exists, else the script stops).
Usage: python -B experiments/d0/atropos_seg.py [--out-root DIR] [--subjects ID ...] [--threads 8]
                                               [--stage NAME] [--data-root DIR] [--allowlist PATH]
--stage / --data-root / --allowlist (work package ETA, 2026-10-30; the defaults stage_L2R_val_v1, <repo>/data/three_d and
<repo>/configs/bayesreg/dev_subjects.json are the D0 behaviour bit for bit): the stage whose pairs supply the images, e.g.
stage_T0_tune_v1 for the C05-7 eta sweep (experiments/eta/sweep.py), with the same commands, parameters and checks;
provenance image_source.stage names it. A known stage must hold its full subject count (STAGE_SUBJECTS). A git worktree
has no data/ link, hence --data-root.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bayesreg.io.pair import load_pair, validate_segmentation_provenance, verify_segmentation_files  # noqa: E402
from jd_curl.three_d.nifti import file_sha256  # noqa: E402

ANTS = Path('/home/rmcfz/opt/baselines/ants/ants-2.6.5/bin')
ANTS_VERSION = '2.6.5-gfdce4d2'
MODEL_ID = 'ANTs-2.6.5-gfdce4d2-Atropos'
DATA_ROOT = ROOT/'data/three_d'
DEFAULT_STAGE = 'stage_L2R_val_v1'
STAGE = DATA_ROOT/DEFAULT_STAGE
STAGE_SUBJECTS = {'stage_L2R_val_v1': 20, 'stage_T0_tune_v1': 24}     # full subject count of the known stages
ALLOWLIST = ROOT/'configs/bayesreg/dev_subjects.json'
DEFAULT_OUT = Path('/home/rmcfz/seq_vp_tag_results/seg/atropos_v1')
CLASS_NAMES_BY_RANK = ('CSF', 'GM', 'WM')
PARAMS = dict(truncation='ImageMath TruncateImageIntensity 0.01 0.99 256 in mask (Atropos input only; colour means use the original image)',
              initialization='KMeans[3]', likelihood='Gaussian', convergence='[5,0.001]', mrf='[0.3,1x1x1]',
              use_random_seed=0, mask='image > 0 (float64 array before the float32 cast)', classes=3, dimension=3)


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def subject_images(allowlist, stage=STAGE):
    """{subject: [(pair, side, PairInput)]} over the stage's pairs (firewall loader)."""
    sel = json.loads((stage/'selection.json').read_text())
    found = {}
    for row in sel['pairs']:
        p = load_pair(stage/row['pair']/'pair.npz', allowlist=allowlist)
        for side in ('fixed', 'moving'):
            found.setdefault(getattr(p, side+'_subject'), []).append((row['pair'], side, p))
    return found


def ras_affine(meta):
    aff = np.eye(4)
    aff[:3, :3] = np.diag(meta['spacing_xyz'])
    aff[:3, 3] = meta['origin_xyz']
    return aff


def save(path, zyx, aff, dtype):
    img = nib.Nifti1Image(np.ascontiguousarray(np.asarray(zyx).transpose(2, 1, 0)).astype(dtype), aff)
    img.header.set_xyzt_units('mm')
    nib.save(img, str(path))


def load_zyx(path):
    return np.asarray(nib.load(str(path)).dataobj).transpose(2, 1, 0)


def segment(subject, entries, out_root, threads, stage_name=DEFAULT_STAGE):
    out = out_root/subject
    if (out/'provenance.json').exists():
        print(f'{subject}: provenance.json exists, skipped', flush=True)
        return json.loads((out/'provenance.json').read_text())
    if out.exists():
        raise SystemExit(f'{subject}: {out} exists without provenance.json (incomplete earlier attempt kept; STOP)')
    pair_name, side, pair = entries[0]
    image = getattr(pair, side)
    shas = {f'{n}:{s}': array_sha(getattr(p, s)) for n, s, p in entries}
    srcs = {f'{n}:{s}': p.source_sha256[s] for n, s, p in entries}
    if len(set(shas.values())) != 1 or len(set(srcs.values())) != 1:
        raise SystemExit(f'{subject}: the image differs between pairs {shas} {srcs}')
    out.mkdir(parents=True)
    aff = ras_affine(pair.grid_metadata)
    mask = image > 0
    save(out/'t1.nii.gz', image, aff, np.float32)
    save(out/'mask.nii.gz', mask.astype(np.uint8), aff, np.uint8)
    trunc = [str(ANTS/'ImageMath'), '3', 't1_trunc.nii.gz', 'TruncateImageIntensity', 't1.nii.gz', '0.01', '0.99', '256',
             'mask.nii.gz']
    cmd = [str(ANTS/'Atropos'), '-d', '3', '-a', 't1_trunc.nii.gz', '-x', 'mask.nii.gz', '-i', 'KMeans[3]', '-k', 'Gaussian',
           '-c', '[5,0.001]', '-m', '[0.3,1x1x1]', '-r', '0', '-o', '[labels.nii.gz,posteriors%02d.nii.gz]', '-v', '1']
    env = dict(os.environ, ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(threads))
    started, t0 = utc(), time.perf_counter()
    with open(out/'atropos.log', 'w') as log:
        for c in (trunc, cmd):
            log.write('$ '+' '.join(c)+'\n'); log.flush()
            rc = subprocess.run(c, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise SystemExit(f'{subject}: {Path(c[0]).name} exit {rc} (see {out/"atropos.log"})')
    seconds = time.perf_counter()-t0
    post = np.stack([load_zyx(out/f'posteriors{k:02d}.nii.gz').astype(np.float64) for k in (1, 2, 3)])
    labels = load_zyx(out/'labels.nii.gz')
    total = post.sum(0)
    means = [float((post[k]*image).sum()/post[k].sum()) for k in range(3)]
    rank = np.argsort(np.argsort(means))                      # rank of each file's class by mean intensity
    class_names = [CLASS_NAMES_BY_RANK[r] for r in rank]
    files = [f'posteriors{k:02d}.nii.gz' for k in (1, 2, 3)]
    prov = dict(
        schema='bayesreg.tissue_provenance.v1 (contract C03.1-2/3)', layer='tissue', segmenter_id='atropos',
        model_ids=[MODEL_ID], trained_with_labels=False, test_labels_seen=False, oracle=False,
        class_names=class_names, wm_index=class_names.index('WM'),
        posterior_files=files, posterior_file_index_base=1,
        input_sha256={'aligned_norm.nii.gz': srcs[next(iter(srcs))]},
        output_sha256={name: file_sha256(out/name) for name in files+['labels.nii.gz']},
        subject=subject, dataset='OASIS',
        image_source=dict(stage=stage_name, pairs=[f'{n}:{s}' for n, s, _ in entries],
                          pair_sha256={n: p.provenance['pair_sha256'] for n, _, p in entries},
                          array_sha256_float64=shas[next(iter(shas))], loader='bayesreg.io.pair.load_pair',
                          keys_read=pair.provenance['keys_read'], label_firewall=pair.provenance['label_firewall'],
                          allowlist_sha256=pair.provenance['subject_guard']['allowlist_sha256']),
        atropos_inputs={n: file_sha256(out/n) for n in ('t1.nii.gz', 't1_trunc.nii.gz', 'mask.nii.gz')},
        grid=dict(pair.grid_metadata, storage='zyx arrays, RAS; NIfTI written as xyz with RAS affine'),
        version=f'ANTs {ANTS_VERSION} (ImageMath + Atropos; binaries in {ANTS})', command=[trunc, cmd],
        environment=dict(ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=threads), parameters=PARAMS,
        started_utc=started, seconds=seconds,
        checks=dict(class_mean_intensity_by_file=means, ascending_by_file=bool(np.all(np.diff(means) > 0)),
                    mask_voxels=int(mask.sum()),
                    posterior_sum_max_abs_dev_in_mask=float(np.abs(total[mask]-1).max()),
                    posterior_max_outside_mask=float(np.abs(post[:, ~mask]).max()) if (~mask).any() else 0.,
                    posterior_min=float(post.min()), posterior_max=float(post.max()),
                    label_values=sorted(int(v) for v in np.unique(labels)),
                    volume_fraction_by_file=[float(post[k][mask].sum()/mask.sum()) for k in range(3)]),
        qc_gate='recorded_only')
    fractions = prov['checks']['volume_fraction_by_file']
    prov['checks']['qc_min_class_fraction_ge_0.05'] = bool(min(fractions) >= .05)
    (out/'provenance.json').write_text(json.dumps(prov, indent=1)+'\n')
    if min(fractions) < .05:
        raise SystemExit(f'{subject}: QC failed, class fractions {fractions} (a class below 5%); record kept, STOP')
    for n, s, p in entries:                                   # contract check against every pair the subject is in
        validate_segmentation_provenance(prov, p, side=s, layer='tissue')
    verify_segmentation_files(prov, out)
    print(f'{subject}: {seconds:.1f} s; means {np.round(means, 4).tolist()} -> {class_names}; '
          f'sum dev {prov["checks"]["posterior_sum_max_abs_dev_in_mask"]:.2e}', flush=True)
    return prov


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--subjects', nargs='*')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--stage', default=DEFAULT_STAGE)
    ap.add_argument('--data-root', type=Path, default=DATA_ROOT)
    ap.add_argument('--allowlist', type=Path, default=ALLOWLIST)
    a = ap.parse_args(argv)
    found = subject_images(a.allowlist, a.data_root/a.stage)
    subjects = sorted(found) if not a.subjects else a.subjects
    assert len(found) == STAGE_SUBJECTS.get(a.stage, len(found)) and all(s in found for s in subjects)
    a.out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for s in subjects:
        segment(s, found[s], a.out_root, a.threads, a.stage)
    print(f'done {len(subjects)} subjects in {time.perf_counter()-t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
