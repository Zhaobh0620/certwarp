"""LP-C2: the synthetic world of the LPBA40 evaluation entry's tests (contract C26.10: synthetic data and temporary git repositories only).

A temporary code tree (tests/test_bayesreg/sg/_sg_world.make_world: synthetic sealed lists pinned into bayesreg by monkeypatch) holding copies of the pinned code-tree files (the
preregistration, the lists, the frozen files, the c record, the hold-out arm table the LPBA40 one is derived from) and a synthetic LPBA allowlist (its sha256 pinned by
monkeypatch), committed; the results root and the LPBA40 data root in tmp_path (bayesreg.eval.lpba_eval.RESULTS, RAW_ROOT; lpba_common's).  Tiny synthetic NIfTI files with
LPBA40's kind of header (an LAS affine, units unknown): an image and a label map per subject (the label: id 21 on the low half of the stored first axis, 22 on the high half,
plus a few other of the 56 ids).  The stage is written by experiments/lpba/prep_lpba.py's own mm_copy and write_pair (the format of the real one).  The cells of every arm of the
committed LPBA40 arm table (relocated onto the test root) are synthetic records with small displacement fields.  No real image, label or result root is touched."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types

import nibabel as nib
import numpy as np

import bayesreg.io.guard as G
import bayesreg.io.sealed as S
from bayesreg.eval import fields as F
from bayesreg.eval import lpba_eval as E

from ..sg import _sg_world as W

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'lpba', ROOT/'experiments'/'holdout', ROOT/'experiments'/'he'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import lpba_common as lc  # noqa: E402
import prep_lpba as prep  # noqa: E402

hc = lc.hc
SHAPE_XYZ = (12, 16, 16)                                  # native; stored (nz, ny, nx) = (16, 16, 12)
STORED = (16, 16, 12)
AFFINE = np.array([[-1., 0, 0, 90], [0, 1., 0, -126], [0, 0, 1., -72], [0, 0, 0, 1]])
PINNED = (lc.PREREG_REL, *lc.LIST_SHA256, *lc.FROZEN_FILES, lc.C_RECORD_REL, 'configs/frozen/holdout_B_v1_arm_table.json')
CONTACT = ['OASIS_OAS1_0910_MR1', 'OASIS_OAS1_0911_MR1']


def las(data, *, units='unknown', dtype=np.float32, affine=AFFINE):
    img = nib.Nifti1Image(np.asarray(data).astype(dtype), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.header.set_xyzt_units(units)
    return img


def label_data(subject, *, extra=True):
    """Stored (x, y, z): 21 (left) on the low x half, 22 (right) on the high half, inside a box; a few other ids of the 56 when `extra`."""
    lab = np.zeros(SHAPE_XYZ, dtype=np.int16)
    lab[2:6, 3:13, 3:13] = 21
    lab[6:10, 3:13, 3:13] = 22
    if extra:
        n = int(subject[1:])
        lab[2:6, 3:6, 3:5] = 41
        lab[6:10, 3:6, 3:5] = 42
        lab[3:5, 10:13, 10:13] = 181 if n % 2 else 182
    return lab


def image_data(subject):
    rng = np.random.default_rng(int(subject[1:]))
    lab = label_data(subject)
    data = (lab > 0)*rng.uniform(100., 300., SHAPE_XYZ)
    return data


def lpba_document(contact):
    sources = [dict(name='synthetic_dev', dataset='OASIS', path='lists/dev.txt', sha256='a'*64, never_for_training=False, subjects=list(W.DEV)),
               dict(name='synthetic_train', dataset='OASIS', path='lists/train.txt', sha256='b'*64, never_for_training=False, subjects=list(contact)),
               dict(name='lpba_tune', dataset='LPBA40', path='lists/tune.txt', sha256='c'*64, never_for_training=False, subjects=[f'S{n:02d}' for n in range(1, 25)]),
               dict(name='lpba40_test_v1', dataset='LPBA40', path=lc.TEST_SUBJECTS_REL, sha256=lc.LIST_SHA256[lc.TEST_SUBJECTS_REL], never_for_training=False,
                    subjects=[f'S{n:02d}' for n in range(3, 37)])]
    return dict(schema=G.ALLOWLIST_SCHEMA, subjects=dict(OASIS=list(W.DEV)+list(contact), LPBA40=[f'S{n:02d}' for n in range(1, 37)]), sources=sources)


def make(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch, record=False)
    for rel in PINNED:
        (w.repo/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, w.repo/rel)
    path = W.write_json(w.repo/G.LPBA40_ALLOWLIST_REL, lpba_document(CONTACT))
    monkeypatch.setattr(G, 'LPBA40_ALLOWLIST_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    W.commit_all(w.repo, 'the pinned files of the LPBA40 evaluation (test)')
    results, raw, tmp = tmp_path/'results', tmp_path/'lpba_raw', tmp_path/'tmp'
    results.mkdir()
    tmp.mkdir()
    for subject in [f'S{n:02d}' for n in range(1, 41)]:
        directory = raw/subject/'delineation'
        directory.mkdir(parents=True)
        nib.save(las(image_data(subject)), str(directory/'image.nii.gz'))
        nib.save(las(label_data(subject), dtype=np.int16), str(directory/'label.nii.gz'))
    monkeypatch.setenv(hc.TEST_SANDBOX_ENV, '1')
    monkeypatch.setenv('TMPDIR', str(tmp))
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp))
    for module in (lc, hc, E):
        monkeypatch.setattr(module, 'RESULTS', results)
    monkeypatch.setattr(lc, 'RAW_ROOT', raw)
    monkeypatch.setattr(E, 'RAW_ROOT', raw)
    monkeypatch.setattr(hc, 'AUTH_ROOT', w.repo)
    monkeypatch.setattr(prep, 'EXPECTED_SHAPE_ZYX', STORED)
    monkeypatch.setattr(E, 'check_code_identity', lambda pair_set, repo=None: dict(rule_one=dict(pinned='p'*40, stub='test: rule one is tested apart (T13)'),
                                                                                  code=dict(head='h'*40, clean=True, status='')))
    monkeypatch.setattr(E, '_LAST_SPY', None)
    S._reset_cache()
    return types.SimpleNamespace(repo=w.repo, results=results, raw=raw, tmp=tmp, allowlist_path=path)


def build_stage(world, pair_set):
    """The mode's stage by prep_lpba's own functions (P1-P7's format): <root>/stage/_nifti_mm/, <pair>/{pair.npz, manifest.json}."""
    lists = lc.read_lists()
    rows = E.pair_rows(pair_set, lists)
    stage = E.root(pair_set)/'stage'
    c = lc.load_c_record()
    copies = {}
    for _, f, m, _ in rows:
        for s in (f, m):
            if s not in copies:
                copies[s] = prep.mm_copy(world.raw/s/'delineation'/'image.nii.gz', stage/'_nifti_mm'/f'{s}_image.nii.gz')
    for _, f, m, _ in rows:
        prep.write_pair(stage, f, m, copies=copies, c=c.c, c_record=c.record)
    S._reset_cache()
    return rows


def field(shape, seed, scale=.2):
    rng = np.random.default_rng(seed)
    return (scale*rng.standard_normal((3,)+tuple(shape))).astype(np.float64)


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1)+'\n', encoding='utf-8')
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_finalized(pair_set, rows, table, *, drop=(), changes=None):
    """LP-B's step-5 record of the method arm (experiments/hb/hb_review.finalize's form): <root>/review/finalized_outputs.json, a row per pair (cell, terminal, review, line_status,
    part.output); `drop`: pairs left out; `changes`: {pair: {key: value}} over a row."""
    arm = next(a for a in table['arms'] if a['name'] == 'method')
    pairs = {}
    for row in rows:
        if row[3] in drop:
            continue
        cell = Path(arm['record_dir'].format(pair=row[3]))
        term, rev = cell/'terminal.json', cell/'review'/'review.json'
        pairs[row[3]] = dict(cell=str(cell), terminal=dict(path=str(term), sha256=sha(term)), review=dict(path=str(rev), sha256=sha(rev)),
                             line_status=json.loads(term.read_text())['line_status'],
                             part=dict(output={n: (sha(cell/n) if (cell/n).is_file() else None) for n in ('output/final.npz', 'output/disp.npz', 'output/cert.json', 'output/line.json')},
                                       t88_repair={}, state_hash=None, fallback=None, superseded=[], evidence=None), bz_decision=None)
        pairs[row[3]].update((changes or {}).get(row[3], {}))
    path = E.finalized_path(pair_set)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return write_json(path, dict(schema=E.FINALIZED_SCHEMA, pair_set=pair_set, batch='test', pairs=pairs, utc='2026-11-19T22:00:00Z'))


def write_cells(world, pair_set, rows, *, resumed=(), suspicious=(), finalized=True):
    """A written output cell for every arm of the relocated table on every row; configc's pairs in `resumed` from <pair>.resume (result_resume.json); the method's finalized record."""
    table, _ = E.load_arm_table(pair_set, world.repo)
    for row in rows:
        name = row[3]
        for k, arm in enumerate(table['arms']):
            d = field(STORED, 100*row[0]+k)
            base = Path(arm['record_dir'].format(pair=name, arm=arm['name']))
            if arm['name'] == 'method':
                out = base/'output'
                out.mkdir(parents=True, exist_ok=True)
                np.savez(out/'disp.npz', d=d)
                u = F.d_to_u(d, (1., 1., 1.))
                np.savez(out/'final.npz', u=u, state_hash=np.array(F.array_sha256(u)))
                write_json(out/'cert.json', dict(state_hash=F.array_sha256(u), status='certified_feasible'))
                write_json(out/'line.json', dict(line_status='hx_certified'))
                write_json(base/'review'/'review.json', dict(finalized=True))
                write_json(base/'terminal.json', dict(line_status='hx_certified'))
                continue
            work = Path(arm['field']['root'].format(arm=arm['name']))/name
            if arm['name'] == 'configc':
                where = Path(str(work)+'.resume') if name in resumed else work
                disp = where/'output'/'disp.npz'
                disp.parent.mkdir(parents=True, exist_ok=True)
                np.savez(disp, d=d)
                doc = dict(status='ok', output_store=dict(disp=dict(file=str(disp), sha256=sha(disp))))
                write_json(base/('result_resume.json' if name in resumed else 'result.json'), doc)
                if name in resumed:
                    write_json(base/'result.json', dict(status='infra_failure'))
                continue
            disp = work/'disp.npz'
            disp.parent.mkdir(parents=True, exist_ok=True)
            np.savez(disp, d=d)
            write_json(base/'result.json', dict(status='ok', disp=dict(file=str(disp), sha256=sha(disp)),
                                                export_check=dict(suspicious=(arm['name'], name) in suspicious)))
    if finalized:
        write_finalized(pair_set, rows, table)
    return table


def write_manifest(world, pair_set, rows, *, drop=(), extra=None, commit=True):
    """The step-10 manifest (runs/lpba40_test_v1/eval_manifest.txt) and the record-only file, from the entry's own input roles, committed in one commit."""
    table, table_record = E.load_arm_table(pair_set, world.repo)
    cells = E.check_cells(table, rows, repo=world.repo)
    originals = E.image_originals(table, rows)
    roles = E.input_roles(table, rows, cells, originals)
    lines = [f'# {E.SCHEMA_MANIFEST}', f'# prereg_sha256 {E.PREREG_SHA256}', f'# arm_table_sha256 {E.ARM_TABLE_SHA256}']
    for path in sorted(roles):
        if path in drop:
            continue
        lines.append(f'{sha(path)}  {path}')
    if extra:
        lines.append(extra)
    text = '\n'.join(lines)+'\n'
    W.write_text(world.repo/E.MANIFEST_REL, text)
    W.write_text(world.repo/E.RECORDS_REL, f'# record-only lines (test) of the manifest {hashlib.sha256(text.encode()).hexdigest()}\n')     # changes with it: one commit
    if commit:
        W.commit_all(world.repo, 'the step-10 manifest (test)')
    S._reset_cache()
    return roles


def no_units(args, **kwargs):
    import subprocess
    return subprocess.CompletedProcess(args, 0, '', '')
