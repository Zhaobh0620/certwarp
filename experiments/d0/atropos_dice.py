"""D0, record only: Dice of the Atropos hard labels (atropos_seg.py) against the truth seg4 of each validation subject.

Run only after the posteriors exist (the segmentation never sees labels). Truth classes from seg4 (OASIS: 1 cortex,
2 subcortical grey matter, 3 white matter, 4 CSF): CSF = {4}, GM = {1, 2} (cortical and subcortical grey matter merged),
WM = {3}. Atropos labels: class_names of provenance.json (file index k+1 -> class_names[k]).
Two readings: 'all' (whole volume; Atropos' mask is image > 0, which includes extra-cerebral tissue that seg4 leaves at 0)
and 'in_truth_brain' (voxels with seg4 > 0 only). Macro = mean over the three classes. Not used by any arm or decision.
Usage: python -B experiments/d0/atropos_dice.py [--seg-root DIR] [--out FILE]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT/'data/three_d/stage_L2R_val_v1'
TRUTH = {'CSF': (4,), 'GM': (1, 2), 'WM': (3,)}


def dice(a, b):
    s = a.sum()+b.sum()
    return float(2*np.logical_and(a, b).sum()/s) if s else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg-root', type=Path, default=Path('/home/rmcfz/seq_vp_tag_results/seg/atropos_v1'))
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or a.seg_root/'dice_vs_truth_seg4.json'
    sel = json.loads((STAGE/'selection.json').read_text())
    truth = {}
    for row in sel['pairs']:
        d = STAGE/row['pair']
        man = json.loads((d/'manifest.json').read_text())
        if hashlib.sha256((d/'pair.npz').read_bytes()).hexdigest() != man['pair_sha256']:
            raise SystemExit(f'{row["pair"]}: pair.npz sha256 differs from the manifest')
        with np.load(d/'pair.npz') as z:
            for side in ('fixed', 'moving'):
                truth.setdefault(man[side+'_subject'], np.array(z[side+'_seg4']))
    rows = []
    for subject in sorted(truth):
        sd = a.seg_root/subject
        prov = json.loads((sd/'provenance.json').read_text())
        lab = np.asarray(nib.load(str(sd/'labels.nii.gz')).dataobj).transpose(2, 1, 0)
        t = truth[subject]
        if lab.shape != t.shape:
            raise SystemExit(f'{subject}: shape {lab.shape} != {t.shape}')
        row = dict(subject=subject)
        for scope, region in (('all', np.ones(t.shape, bool)), ('in_truth_brain', t > 0)):
            per = {}
            for k, name in enumerate(prov['class_names']):
                per[name] = dice((lab == k+1) & region, np.isin(t, TRUTH[name]) & region)
            row[scope] = dict(per_class=per, macro=float(np.mean(list(per.values()))))
        rows.append(row)
        print(f"{subject}: all macro {row['all']['macro']:.4f} {json.dumps({k: round(v, 4) for k, v in row['all']['per_class'].items()})}"
              f"  in-brain macro {row['in_truth_brain']['macro']:.4f}", flush=True)
    summary = {scope: dict(macro_mean=float(np.mean([r[scope]['macro'] for r in rows])),
                           per_class_mean={c: float(np.mean([r[scope]['per_class'][c] for r in rows])) for c in TRUTH})
               for scope in ('all', 'in_truth_brain')}
    print(json.dumps(summary, indent=1))
    out.write_text(json.dumps(dict(role='record only (D0 protocol); not an input of any arm or decision',
                                   truth_mapping={k: list(v) for k, v in TRUTH.items()}, rows=rows, summary=summary,
                                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
