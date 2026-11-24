"""LP-B's LPBA40 world (runs/20261119-lpba/任务书_B_方法配置C与外部臂.md; contract C26.10 T8, T16, T19): HB's sandbox (hb/conftest.py: every holdout_common root under a tempfile.mkdtemp(prefix='xbt_')
directory, XB_TEST_SANDBOX set, systemd-run a shim exiting 99) with LP-A's synthetic LPBA40 world on top (tests/test_bayesreg/lpba/test_lpba_entries.py's helpers, imported, not copied): a temporary
code tree pinned into bayesreg (synthetic sealed lists, the pinned files copied byte for byte, a synthetic LPBA allowlist pinned by monkeypatch), lpba_common.RESULTS the sandbox's results root,
tiny synthetic LPBA40-like NIfTI images of S01, S02, S05, S06 (no label is ever opened), the formal mode's table cut to S05 <- S06 (LP-A's formal_one), rule one of the formal mode stubbed where
another package's code identity would decide (C26.9-4, LP-C2).  The stage, HI's predictions and Atropos are made by LP-A's own entries (prep_lpba, hi_lpba with tiny members on the CPU,
atropos_lpba with a stand-in for ANTs).  No real subject's file, no unit, no GPU."""
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

from ..hi import _world as HW
from ..lpba import test_lpba_entries as LE
from ..sg import _sg_world as SW
from . import _world as W

R, he, hc, hb, review, hm, hold = W.R, W.he, W.hc, W.hb, W.hb_review, W.hb_method, W.hold
lc, prep, atr, hi_lpba, G, S = LE.lc, LE.prep, LE.atr, LE.hi_lpba, LE.G, LE.S
FORMAL, FEAS = 'lpba40_test', 'lpba40_feas'
PAIR, FEAS_PAIR = 'LPBA40_S05__LPBA40_S06', 'LPBA40_S01__LPBA40_S02'
ONE = [('S05', 'S06')]
CODE = LE.CODE
SUBJECTS = ('S01', 'S02', 'S05', 'S06')


def make(tmp_path, monkeypatch, *, formal_one=True):
    """The world (module docstring) under the sandbox's results root (hc.RESULTS, set by hb/conftest.py).  Returns a namespace (repo, results, raw, allowlist_path, tmp, HW)."""
    w = SW.make_world(tmp_path, monkeypatch, record=False)
    for rel in LE.PINNED:
        (w.repo/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LE.ROOT/rel, w.repo/rel)
    path = SW.write_json(w.repo/G.LPBA40_ALLOWLIST_REL, LE.lpba_document(HW.CONTACT))
    monkeypatch.setattr(G, 'LPBA40_ALLOWLIST_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    raw = tmp_path/'lpba_raw'
    for subject in SUBJECTS:
        directory = raw/subject/'delineation'
        directory.mkdir(parents=True)
        nib.save(LE.las_image(subject), str(directory/'image.nii.gz'))
    results = Path(hc.RESULTS)
    monkeypatch.setattr(lc, 'RESULTS', results)
    monkeypatch.setattr(lc, 'RAW_ROOT', raw)
    monkeypatch.setattr(prep, 'EXPECTED_SHAPE_ZYX', LE.STORED)
    monkeypatch.setattr(hc, 'machine_gate', lambda **kwargs: dict(passed=True, note='test: no machine queried', **kwargs))
    monkeypatch.setattr(hc, 'disk_guard', lambda **kwargs: dict(passed=True, note='test', free_gib={}, **kwargs))
    if formal_one:
        real_table = lc.table
        monkeypatch.setattr(lc, 'table', lambda pair_set, lists=None: (['S05', 'S06'], list(ONE)) if pair_set == FORMAL else real_table(pair_set, lists))
        monkeypatch.setattr(lc, 'formal_code_problem', lambda code, pair_set, dry_run=False: None)
        monkeypatch.setattr(lc, 'rule_one_problem', lambda code: None)
    S._reset_cache()
    return SimpleNamespace(repo=w.repo, results=results, raw=raw, allowlist_path=path, tmp=Path(tmp_path), HW=HW)


PLAN_REL = 'archive/seq_vp_3d/results/verification/3d_stage_HOLDOUT_v1/plan'      # bayesreg.io.sealed._PLAN: three of the four sealed lists lie here in production


def production_lists(world, monkeypatch):
    """LP-G (decision_log 2026-11-20): the synthetic sealed lists moved, unchanged (contents and sha256 synthetic), from lists/ to the relative paths bayesreg.io.sealed reads in production --
    the paths of SW.REAL's SEALED_LISTS, the module's own constant (holdout_B_v1 under splits/bayesreg_v1/; H1, H4, H3 under archive/.../3d_stage_HOLDOUT_v1/plan/) -- and pinned again; plan/
    also gets a synthetic file that is not a list, archive/ one more file.  Only the production paths are taken from the constant: no real list is opened.  Returns a namespace (lists: the four
    paths as sealed.REPO/<path>, plan, other: the other plan file, stray: archive/x.json)."""
    real = {spec['name']: spec['path'] for spec in SW.REAL['SEALED_LISTS']}
    assert sorted(real) == sorted(spec['name'] for spec in S.SEALED_LISTS) and sum(p.startswith(PLAN_REL+'/') for p in real.values()) == 3
    specs = []
    for spec in S.SEALED_LISTS:
        target = world.repo/real[spec['name']]
        target.parent.mkdir(parents=True, exist_ok=True)
        (world.repo/spec['path']).rename(target)
        specs.append(dict(spec, path=real[spec['name']]))
    other = SW.write_text(world.repo/PLAN_REL/'other_plan_file.tsv', 'fixed\tmoving\nOASIS_OAS1_0999_MR1\tOASIS_OAS1_0998_MR1\n')       # synthetic ids (no OASIS subject above 0457)
    stray = SW.write_json(world.repo/'archive'/'x.json', dict(synthetic=True))
    SW.commit_all(world.repo, 'LP-G: the synthetic sealed lists at their production paths')
    SW.pin(monkeypatch, world.repo, specs)
    return SimpleNamespace(lists=[world.repo/spec['path'] for spec in specs], plan=world.repo/PLAN_REL, other=other, stray=stray, archive=world.repo/'archive')


def formal_checkout(world, monkeypatch):
    """LP-H: the production layout of the code tree (C26.9-5, R1.2): the synthetic tree copied to <formal root>/code (lpba_common.roots('lpba40_test').root/code), no data/ in it (no data
    link), bayesreg.io.sealed pinned there (the same lists, paths and sha256), and the working directory there -- the units' --working-directory is the checkout (hb_core.unit_argv).
    world.repo then names the copy.  Returns its path."""
    import shutil
    code = Path(lc.roots(FORMAL).root)/'code'
    shutil.copytree(world.repo, code, symlinks=True)
    assert not (code/'data').exists() and not (code/'data').is_symlink()
    SW.pin(monkeypatch, code, [dict(spec) for spec in S.SEALED_LISTS])
    monkeypatch.chdir(code)
    world.repo = code
    return code


FROZEN_PREPARE = """# synthetic (LP-H test): the frozen prepare's effect on the run directory, as test_hb_configc.fake_prepare leaves it: the slot's pair.npz and manifest.json under
# <run>/inputs/<pair>/, protocol.json, snapshot/ -- read from the private copy's data/three_d/<stage>/<pair>/, where firewall_c.prepare put them.
import json, shutil, sys
from pathlib import Path
run, pair = Path(sys.argv[2]), sys.argv[sys.argv.index('--pair')+1]
slot = Path(__file__).resolve().parents[1]/'data'/'three_d'/'stage_L2R_val_v1'/pair
(run/'inputs'/pair).mkdir(parents=True)
for name in ('pair.npz', 'manifest.json'):
    shutil.copyfile(slot/name, run/'inputs'/pair/name)
(run/'protocol.json').write_text(json.dumps(dict(task=dict(name='task'))))
(run/'snapshot').mkdir()
"""


def synthetic_frozen(tmp, monkeypatch):
    """LP-H: a synthetic frozen tree for the real firewall_c.prepare (frozen_copy, the private copy, the frozen prepare as a child process, the copy's removal): jd_curl/ with nested
    directories, scripts/ with the prepare script (FROZEN_PREPARE) and a shell file; mc0_core.FROZEN set to it (the contexts' paths.frozen and hb_configc's frozen), firewall_c.freeze_hashes
    its .py files' sha256 (the rule of frozen_copy), check_prepared a stand-in that passes (the frozen prepare's outputs are not made here).  Returns its path."""
    from jd_curl.three_d.nifti import file_sha256
    fc = R.fc
    frozen = Path(tmp)/'frozen_tree'
    files = {'jd_curl/__init__.py': '', 'jd_curl/three_d/__init__.py': '', 'jd_curl/three_d/solvers/newton.py': 'X = 1\n', 'jd_curl/three_d/backends/numpy_backend.py': 'Y = 2\n',
             'scripts/run_stage_3d_e48_pg.py': FROZEN_PREPARE, 'scripts/run_queue.sh': '#!/bin/sh\n'}
    for rel, text in files.items():
        (frozen/rel).parent.mkdir(parents=True, exist_ok=True)
        (frozen/rel).write_text(text, encoding='utf-8')
    hashes = {rel: file_sha256(frozen/rel) for rel in files if rel.endswith('.py')}
    monkeypatch.setattr(R.core, 'FROZEN', frozen)
    monkeypatch.setattr(fc, 'freeze_hashes', lambda: dict(hashes))
    monkeypatch.setattr(fc, 'check_prepared_module', lambda: SimpleNamespace(check=lambda *a, **k: dict(ok=True, problems=[], note='test: the frozen prepare is a stand-in')))
    return frozen


def prepare(world, monkeypatch, pair_set=FORMAL, *, hi=True, atropos=True):
    """LP-A's entries on the mode: the stage (prep_lpba), Atropos (atropos_lpba, ANTs stood in), HI (hi_lpba, tiny members on the CPU).  Returns the HI registry (or None)."""
    S._reset_cache()
    assert prep.run(pair_set, code=CODE, log=lambda text: None)['passed']
    registry = None
    if atropos:
        LE.fake_ants(monkeypatch)
        S._reset_cache()
        assert atr.run(pair_set, code=CODE, log=lambda text: None)['passed']
    if hi:
        registry, _ = LE.hi_world(world, monkeypatch)
        S._reset_cache()
        assert hi_lpba.run(pair_set, code=CODE, device='cpu', log=lambda text: None)['passed']
    S._reset_cache()
    return registry


def ctx_of(pair_set=FORMAL, pair=PAIR, **kwargs):
    """An LPBA context (run_hx.make_lpba_context), a test context (the identity not checked: outside the real results root)."""
    S._reset_cache()
    return R.make_lpba_context(pair_set, pair, device='cpu', check_identity=False, **kwargs)


def write_cell(ctx, pin, *, status=he.HX, certify=None, u=None):
    """An LPBA method cell as HE leaves it, on the pair's own grid (hb/_world.write_cell's files): context.json, output/ (he_core.write_output), t88/run.json and field_0600.npz, repair/repair.json,
    repair.npz and certified.npz (hx_certified).  Returns (u, line)."""
    out = Path(ctx['out'])
    out.mkdir(parents=True, exist_ok=True)
    he.write_json(R.context_path(ctx), ctx)
    g = pin.grid
    u = W.field(g) if u is None else u
    certify = certify or W.Certify()
    doc = he.certificate_document(certify(u, g), W.config(), process='test: the decision certificate', array_sha256_of_u=he.array_sha256(u))
    line = dict(R.line_base(ctx, pin), line_status=status, failure=None, fuse=None,
                fallback=None if status == he.HX else dict(kind='identity' if status == he.FALLBACK_ID else 'configC', state_hash=he.array_sha256(u)),
                rejection=None, third_certificate=None, review_certificate_in=ctx['settings']['review_certificate_in'], superseded=[])
    (out/'t88').mkdir(exist_ok=True)
    np.savez(out/'t88'/he.FIELD_FILE, u=u)
    he.write_json(out/'t88'/'run.json', dict(status='done', labels_sha256=dict(fixed='1'*64, moving='2'*64)))
    (out/'repair').mkdir(exist_ok=True)
    he.write_json(out/'repair'/'repair.json', dict(status='certified'))
    np.savez(out/'repair'/'repair.npz', nodes=np.zeros(1), values=np.zeros(1))
    if status == he.HX:
        np.savez(out/'repair'/'certified.npz', u=u)
    he.write_output(out/he.OUTPUT_DIR, u, g, doc, line)
    return u, he.read_json(out/he.OUTPUT_DIR/'line.json')


def detached_git(commit='c'*40):
    """XB's fake git (a clean tree at `commit`) on a detached HEAD (git symbolic-ref -q HEAD exits 1): the feasibility mode's code record (C26.9-1; review minor 5)."""
    import subprocess
    base = W.XW.fake_git(commit=commit)

    def run(argv, **kwargs):
        if 'symbolic-ref' in argv:
            return subprocess.CompletedProcess(argv, 1, '', '')
        return base(argv, **kwargs)
    return run


def git_repo(path):
    """A temporary git repository (the code tree of a store test: the input lock is committed there)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    W.XW.git(path, 'init', '-q')
    W.XW.git(path, 'commit', '-q', '--allow-empty', '-m', 'start')
    return path


def commit(repo, rel):
    W.XW.git(repo, 'add', rel)
    W.XW.git(repo, 'commit', '-q', '-m', f'add {rel}')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


__all__ = ['CODE', 'detached_git', 'FEAS', 'FEAS_PAIR', 'FORMAL', 'ONE', 'PAIR', 'PLAN_REL', 'R', 'S', 'G', 'commit', 'ctx_of', 'git_repo', 'hb', 'hc', 'he', 'hi_lpba', 'hm', 'hold', 'lc', 'make',
           'prep', 'prepare', 'production_lists', 'read', 'review', 'write_cell', 'formal_checkout', 'synthetic_frozen']
