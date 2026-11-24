"""A1P (runs/20261102-a1p/plan.md): check the seg35 -> seg4 label assignment against the development labels themselves.

What it does (read only; CPU; no registration, no image is read, no result of any method): for the 20 development subjects
of stage_L2R_val_v1 it counts seg35 label x seg4 label on the voxels of each subject's own label volumes, one count table
per subject (every subject appearing in two pairs is checked to have the same labels in both), and compares them with
  - the seg4 label table /mnt/d/data/seg4_labels.txt (names 1 Cortex, 2 Subcortical-Gray-Matter, 3 White-Matter, 4 CSF), and
  - the assignment `tissue4_ref` of configs/bayesreg/structure_groups_v1.json (status reference_unverified; "inferred, not
    read from the data").
Output: --out JSON (must not exist): the summed table, per seg35 label the seg4 label with the most voxels and its share,
per subject the smallest share, the agreement with tissue4_ref and the voxels with seg4 > 0, seg35 == 0.  Nothing is copied
out of the labels but the counts.

Guard (hard constraints 6, 7): only the pairs of the stage's selection.json, only subjects on the development allowlist that
are not B-line training subjects and not never_for_training (make_inputs.a4_admission); truth labels appear here only as
the check of a label table (never an input of any method).

Usage: python -B experiments/d0/check_seg4_map.py --stage DIR --out FILE [--groups configs/bayesreg/structure_groups_v1.json]
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments/d0'))
import make_inputs as MI  # noqa: E402

SEG4_TABLE = Path('/mnt/d/data/seg4_labels.txt')
SEG4_NAMES = {1: 'Cortex', 2: 'Subcortical-Gray-Matter', 3: 'White-Matter', 4: 'CSF'}
N35, N4 = 36, 5


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_seg4_table(path):
    """{label: name} of a FreeSurfer colour-table file ('#' comments, blank lines)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if parts and not parts[0].startswith('#'):
            out[int(parts[0])] = parts[1]
    return out


def sealed_guard(subjects, allowlist):
    """Contract C23.6-2 (SG, D104 (a)): every subject through the bayesreg guard (bayesreg.io.guard.check_subject, C23.4-2,
    without an unseal object), explicitly and not through the meaning of allows that a4_admission uses.  SystemExit for
    any refusal; reads subject ids only."""
    from bayesreg.io.guard import DevAllowlist, check_subject, load_dev_allowlist
    allow = allowlist if isinstance(allowlist, DevAllowlist) else load_dev_allowlist(allowlist)
    if allow.unseal is not None:
        raise SystemExit('check_seg4_map: an allowlist with an unseal object is refused (development labels only)')
    refused = {s: v.message or v.step for s in subjects for v in [check_subject(s, allow)] if not v.admitted}
    if refused:
        raise SystemExit(f'check_seg4_map: subjects refused by the bayesreg guard (C23.6-2): {refused}')
    return dict(check='bayesreg.io.guard.check_subject (C23.4-2) on every subject, no unseal object', subjects=sorted(subjects),
                allowlist_sha256=allow.sha256)


def as_labels(array, name, top):
    values = np.asarray(array)
    rounded = values.astype(np.int64)
    if not np.array_equal(rounded, values) or rounded.min() < 0 or rounded.max() >= top:
        raise SystemExit(f'{name}: labels must be integers in [0, {top})')
    return rounded


def cross_table(seg35, seg4):
    """(36, 5) counts of (seg35 label, seg4 label) on one subject's voxels."""
    if seg35.shape != seg4.shape:
        raise SystemExit(f'label volumes of shapes {seg35.shape} and {seg4.shape}')
    return np.bincount((N4*seg35+seg4).ravel(), minlength=N35*N4).reshape(N35, N4)


def summarize(tables, reference):
    """tables: {subject: (36, 5) counts}.  reference: {seg35 label (1..35): seg4 label (1..4)} of tissue4_ref."""
    total = sum(tables.values())
    rows, disagree = {}, []
    for label in range(1, N35):
        counts = total[label]
        n = int(counts.sum())
        if n == 0:
            rows[str(label)] = dict(voxels=0, seg4=None, share=None, reference=reference.get(label))
            continue
        best = int(np.argmax(counts))
        worst = min((float(t[label, best])/float(t[label].sum()) for t in tables.values() if t[label].sum() > 0), default=None)
        rows[str(label)] = dict(voxels=n, counts=[int(c) for c in counts], seg4=best, share=float(counts[best])/n,
                                min_subject_share=worst, reference=reference.get(label))
        if reference.get(label) != best:
            disagree.append(dict(seg35=label, data_seg4=best, reference_seg4=reference.get(label),
                                 share=float(counts[best])/n))
    return dict(per_seg35_label=rows, disagreements_with_tissue4_ref=disagree,
                voxels_seg35_0_by_seg4=[int(c) for c in total[0]],
                voxels_seg4_positive_seg35_0=int(total[0, 1:].sum()),
                voxels_seg4_positive=int(total[:, 1:].sum()), voxels_seg35_positive=int(total[1:].sum()),
                voxels_seg35_positive_seg4_0=int(total[1:, 0].sum()))


def reference_map(groups_path):
    document = json.loads(Path(groups_path).read_text())
    grouping = document['groupings']['tissue4_ref']
    out = {}
    for k, group in enumerate(grouping['groups'], start=1):
        for label in group['labels']:
            out[int(label)] = k
    if sorted(out) != list(range(1, N35)):
        raise SystemExit('tissue4_ref: every seg35 label 1..35 exactly once')
    return out, grouping['status'], [g['name'] for g in grouping['groups']]


def run(stage, out, groups, *, allowlist=MI.ALLOWLIST, unet=None, seg4_table=SEG4_TABLE):
    stage, out = Path(stage), Path(out)
    if out.exists():
        raise SystemExit(f'{out}: exists (nothing is overwritten)')
    subjects = MI.stage_subjects(stage)
    admission = MI.a4_admission(subjects, allowlist, unet or MI.UNetSource())
    guard = sealed_guard(subjects, allowlist)
    reference, status, names = reference_map(groups)
    table = read_seg4_table(seg4_table)
    if {k: table.get(k) for k in SEG4_NAMES} != SEG4_NAMES:
        raise SystemExit(f'{seg4_table}: seg4 names {table} differ from {SEG4_NAMES}')
    tables, hashes = {}, {}
    for row in json.loads((stage/'selection.json').read_text())['pairs']:
        directory = stage/row['pair']
        manifest = json.loads((directory/'manifest.json').read_text())
        if any(manifest.get(f'{side}_subject') not in guard['subjects'] for side in ('fixed', 'moving')):
            raise SystemExit(f"{row['pair']}: a subject of its manifest is not one the guard admitted (C23.6-2); not read")
        with np.load(directory/'pair.npz') as z:
            for side in ('fixed', 'moving'):
                subject = manifest[f'{side}_subject']
                seg35 = as_labels(z[f'{side}_seg35'], f"{row['pair']}:{side}_seg35", N35)
                seg4 = as_labels(z[f'{side}_seg4'], f"{row['pair']}:{side}_seg4", N4)
                digest = sha_bytes(seg35.tobytes()+seg4.tobytes())
                if subject in hashes:
                    if hashes[subject] != digest:
                        raise SystemExit(f'{subject}: its labels differ between two pairs')
                    continue
                hashes[subject] = digest
                tables[subject] = cross_table(seg35, seg4)
    if sorted(tables) != sorted(subjects):
        raise SystemExit('the label tables do not cover the stage subjects')
    record = dict(tool='experiments/d0/check_seg4_map.py', tool_sha256=MI.sha_file(Path(__file__)),
                  created_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), stage=str(stage),
                  subjects=subjects, subject_admission=admission, seg4_table=str(seg4_table),
                  seg4_table_sha256=MI.sha_file(seg4_table), seg4_names=table,
                  groups_file=str(groups), groups_sha256=MI.sha_file(groups), tissue4_ref_status=status,
                  tissue4_ref_group_names=names, **summarize(tables, reference))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'x') as stream:
        stream.write(json.dumps(record, indent=1, ensure_ascii=False)+'\n')
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--stage', type=Path, default=MI.STAGE)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--groups', type=Path, default=ROOT/'configs/bayesreg/structure_groups_v1.json')
    ap.add_argument('--seg4-table', type=Path, default=SEG4_TABLE)
    a = ap.parse_args(argv)
    rec = run(a.stage, a.out, a.groups, seg4_table=a.seg4_table)
    print(json.dumps({k: rec[k] for k in ('disagreements_with_tissue4_ref', 'voxels_seg4_positive_seg35_0',
                                          'voxels_seg35_positive_seg4_0', 'voxels_seg4_positive', 'voxels_seg35_positive')},
                     indent=1))
    for label, row in rec['per_seg35_label'].items():
        print(label, row['seg4'], None if row['share'] is None else round(row['share'], 5),
              None if row.get('min_subject_share') is None else round(row['min_subject_share'], 5), row['reference'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
