"""HB's test world (runs/20261109-hb): XB's sandbox (every holdout_common root in a tempfile.mkdtemp(prefix='xbt_') directory, XB_TEST_SANDBOX set, systemd-run a shim that exits 99, SG's real functions on a simulated unseal branch), and
synthetic method cells: HE's own writer (he_core.write_output) on a small grid, a stand-in certificate (a FeasibilityResult of the decision configuration: the comparisons are he_core's own), the contexts of
run_hx.make_holdout_context.  No real subject's file is opened (real ids, synthetic arrays); no unit, no GPU, no frozen run."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace  # noqa: F401

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'hb', ROOT/'experiments'/'he', ROOT/'experiments'/'holdout', ROOT/'experiments'/'xb'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ..xb import _world as XW  # noqa: E402

import hb_core as hb  # noqa: E402
import hb_method  # noqa: E402
import hb_review  # noqa: E402
import he_core as he  # noqa: E402
import he_holdout as hold  # noqa: E402
import run_hx as R  # noqa: E402

from jd_curl.three_d.feasibility import FeasibilityResult  # noqa: E402
from jd_curl.three_d.grid import make_grid_3d  # noqa: E402

hc = XW.hc
SHAPE = XW.SHAPE


def sandbox(tmp_path, monkeypatch, base=None):
    """XB's sandbox (the roots under `base`: the conftest passes XW.roots_base(), never pytest's tmp_path, whose path holds the word pytest the start strings refuse) with HE's hold-out sources;
    HI's hold-out lock (hi_core.HOLDOUT_LOCK) moved there too (A5b item 13: hb_core.hi_union reads it; HI's root is pred_root's parent)."""
    world = XW.sandbox(tmp_path, monkeypatch, base=base)
    monkeypatch.setattr(hc, '_local_end', None)       # A5a diff review n5: no unit end recorded by an earlier test of this process (hb_core.run_unit records one; the gate's local lower bound),
                                                      # as XB's and HP's sandboxes (XW.sandbox does it too; written here so that HB's sandbox does not rest on it)
    monkeypatch.setattr(he, 'HOLDOUT_SOURCES', dict(pred_root=world.results/'seg'/'holdout'/'pred', atropos_root=world.holdout/'atropos'))
    hi = hc._hi()
    monkeypatch.setattr(hi, 'HOLDOUT_LOCK', world.results/'seg'/'hi_holdout_infer.lock')
    return world


def write_hi(subjects=None, *, first=None, residue=(), labels=None):
    """HI's hold-out inference as hi_core leaves it (A5a review I2), in the sandbox: the first run's lock line at hi_core.HOLDOUT_LOCK naming the one root (he_core.HOLDOUT_SOURCES['pred_root']'s
    parent) and, for each subject, pred/<s>/ with stand-in probs36.npz, structure_groups.npz, provenance.json (bytes; a subject of `labels` {subject: int map}: XW.write_prediction's real probs36.npz
    and provenance.json, which ConvexAdam-v4seg reads, and stand-in structure_groups.npz) and its record subjects/<s>.json (subject, output_sha256, provenance_sha256, run).  first: the subjects the first run wrote before it was killed (hi_started.json, hi_stopped.json); the others are written by its one resume into the same root
    (its resume lock's line, hi_started_resume_1.json, hi_record_resume_1.json, run 1), and `residue` (subjects of the resume) lie half written in attempt1/pred/<s>/ (moved out before the resume).
    Default: every subject of the pair list (XW.holdout_rows), one run.  Returns (root, run names of the resume or None)."""
    hi = hc._hi()
    subjects = sorted({s for r in XW.holdout_rows() for s in r}) if subjects is None else list(subjects)
    first = list(subjects) if first is None else list(first)
    root, lock = Path(he.HOLDOUT_SOURCES['pred_root']).parent, Path(hi.HOLDOUT_LOCK)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(dict(out=str(root), event='first_run', utc='2026-11-10T00:00:00Z', pid=1))+'\n', encoding='utf-8')
    (root/'subjects').mkdir(parents=True, exist_ok=True)
    for s in subjects:
        d = root/'pred'/s
        if labels is not None and s in labels:
            XW.write_prediction(root/'pred', s, labels[s])
            (d/'structure_groups.npz').write_bytes(f'{s}/structure_groups.npz'.encode())
        else:
            d.mkdir(parents=True, exist_ok=True)
            for name in hb.HI_PRED_FILES:
                (d/name).write_bytes(f'{s}/{name}'.encode())
        row = dict(subject=s, output_sha256={n: hb.sha(d/n) for n in hb.HI_RECORD_OUTPUTS}, provenance_sha256=hb.sha(d/'provenance.json'), run=0 if s in first else 1)
        (root/'subjects'/f'{s}.json').write_text(json.dumps(row), encoding='utf-8')
    (root/'hi_started.json').write_text(json.dumps(dict(out=str(root))), encoding='utf-8')
    if first == subjects:
        (root/'hi_record.json').write_text(json.dumps(dict(rows={s: {} for s in subjects})), encoding='utf-8')
        return root, None
    (root/'hi_stopped.json').write_text(json.dumps(dict(done=first)), encoding='utf-8')
    for s in residue:
        d = root/'attempt1'/'pred'/s
        d.mkdir(parents=True)
        (d/'probs36.npz').write_bytes(b'half written')
    hi.resume_lock_path(lock, root).write_text(json.dumps(dict(out=str(root), event='authorized_resume', utc='2026-11-10T01:00:00Z', pid=2, resume=dict(previous_out=str(root))))+'\n',
                                               encoding='utf-8')
    names = hi.run_names(1)
    (root/names['started']).write_text(json.dumps(dict(out=str(root))), encoding='utf-8')
    (root/names['record']).write_text(json.dumps(dict(rows={s: {} for s in subjects if s not in first})), encoding='utf-8')
    return root, names


def first_pair():
    return hc.pair_name(*XW.holdout_rows()[0])


def grid(shape=SHAPE):
    nz, ny, nx = shape
    return make_grid_3d(nx=nx, ny=ny, nz=nz, hx=1., hy=1., hz=1., unit='mm')


def config():
    return he.certificate_config(he.frozen_settings())


def result_of(u, g, *, passed=True, lower=.5):
    """A FeasibilityResult of the decision configuration for exactly this array (its state hash); passed False: certified_infeasible, one infeasible cell."""
    return FeasibilityResult(status='certified_feasible' if passed else 'certified_infeasible', min_jd_lower_bound=lower if passed else -.1, min_margin_lower_bound=lower/2, sampled_min_jd=lower+.1,
                             n_cells_checked=int(np.prod([n-1 for n in g.shape])), n_uncertain=0, n_infeasible=0 if passed else 1, n_boxes=0, max_depth=0, witness_cell=None if passed else 3,
                             witness_reference_point=None, reason='test certificate', state_hash=he.array_sha256(np.ascontiguousarray(u)), grid_id=g.grid_id, delta_hex=(0.).hex(), config=config())


class Certify:
    """A stand-in certificate: passes every array unless its state hash is in `fail` (or `fail_zero` and the array is zero); `perturb`: the hashes whose certificate is passed but with another lower bound
    (a review that differs from cert.json: the forced inconsistency of C22.H8 (iv-a)).  Every call recorded."""

    def __init__(self, *, fail=(), fail_zero=False, perturb=()):
        self.fail, self.fail_zero, self.perturb, self.calls = set(fail), fail_zero, set(perturb), []

    def __call__(self, u, g):
        digest = he.array_sha256(np.ascontiguousarray(u))
        self.calls.append(digest)
        if digest in self.fail or (self.fail_zero and not np.any(u)):
            return result_of(u, g, passed=False)
        r = result_of(u, g)
        return replace(r, min_jd_lower_bound=.25) if digest in self.perturb else r

    def pair(self):
        return self, config()


def field(g, scale=.2, seed=1):
    """A small field of the grid with six zero faces (C16)."""
    rng = np.random.default_rng(seed)
    u = np.zeros(g.shape+(3,))
    u[1:-1, 1:-1, 1:-1] = scale*rng.standard_normal((g.shape[0]-2, g.shape[1]-2, g.shape[2]-2, 3))
    return np.ascontiguousarray(u)


def holdout_ctx(tmp_path, monkeypatch, *, unseal=True, check_identity=False):
    """A hold-out pair's context (run_hx.make_holdout_context; the identity not checked: a test context, outside the real results root) and its stage pair (no label member)."""
    pair = first_pair()
    XW.write_stage_pair(hc.STAGE_ROOT, pair, seg=False)
    sim = XW.simulated_unseal(tmp_path, monkeypatch) if unseal else None
    ctx = R.make_holdout_context(pair, device='cpu', check_identity=check_identity)
    return pair, ctx, sim


def write_cell(ctx, *, status=he.HX, u=None, certify=None, review_certificate_in=None, extra_files=True, step4=True):
    """A method cell as HE leaves it: context.json, output/ (he_core.write_output: final.npz, disp.npz, cert.json, line.json), t88/run.json and field_0600.npz, repair/repair.json, repair.npz and
    certified.npz (hx_certified).  Returns (u, line)."""
    out = Path(ctx['out'])
    out.mkdir(parents=True, exist_ok=True)
    he.write_json(R.context_path(ctx), ctx)
    g = grid()
    u = field(g) if u is None else u
    certify = certify or Certify()
    doc = he.certificate_document(certify(u, g), config(), process='test: the decision certificate', array_sha256_of_u=he.array_sha256(u))
    line = dict(R.line_base(ctx), line_status=status, failure=None, fuse=None, fallback=None if status == he.HX else dict(kind='identity' if status == he.FALLBACK_ID else 'configC', state_hash=he.array_sha256(u)),
                rejection=None, third_certificate=None, review_certificate_in=ctx['settings']['review_certificate_in'] if review_certificate_in is None else review_certificate_in, superseded=[])
    if extra_files:
        (out/'t88').mkdir(exist_ok=True)
        np.savez(out/'t88'/he.FIELD_FILE, u=u)
        he.write_json(out/'t88'/'run.json', dict(status='done', labels_sha256=dict(fixed='1'*64, moving='2'*64)))
        (out/'repair').mkdir(exist_ok=True)
        he.write_json(out/'repair'/'repair.json', dict(status='certified'))
        np.savez(out/'repair'/'repair.npz', nodes=np.zeros(1), values=np.zeros(1))
        if status == he.HX:
            np.savez(out/'repair'/'certified.npz', u=u)
    he.write_output(out/he.OUTPUT_DIR, u, g, doc, line)
    if step4 and ctx.get('pair_set') == 'holdout':          # the method queue's record of the pair (its certified check passed, the batch went on): the review's D110 reads it
        rec = hb.record_dir('method', ctx['pair'])
        rec.mkdir(parents=True, exist_ok=True)
        he.write_json(rec/'step4.json', dict(status=status, went_on=True, written_by='test'))
    return u, he.read_json(out/he.OUTPUT_DIR/'line.json')


def manifest(mode, pairs):
    path = mode.manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(schema='test', pairs=hb_review.input_manifest(pairs))))
    return path


def probe(when):
    return dict(when=when, unit='tag-hb-review-test.service', cgroup='/user.slice/tag-hb-review-test.service', events=dict(oom=0, oom_kill=0))


def git(repo, *args):
    return XW.git(repo, *args)


__all__ = ['Certify', 'Path', 'R', 'SHAPE', 'SimpleNamespace', 'XW', 'config', 'field', 'first_pair', 'grid', 'hb', 'hb_method', 'hb_review', 'hc', 'he', 'hold', 'holdout_ctx', 'manifest', 'probe', 'result_of',
           'sandbox', 'subprocess', 'write_cell']
