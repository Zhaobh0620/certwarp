"""LP-H (decision_log 2026-11-21 ~2:58 PM CST: the LPBA40 feasibility run3 on F''' 43c99313 stopped at configuration C, S01 <- S02, infra_failure / spy_breach, the ArmSpy refusing an
open of $CODE/data, write false).  The cause: firewall_c.prepare removed its private copy of the frozen tree with shutil.rmtree, whose fd-based walk opens each subdirectory by its bare
name relative to a directory descriptor; the 'open' audit event carries the bare name without the descriptor, so hp_core's hook resolved 'data' (the copy's data/three_d/ slot) against
the working directory, the checkout (<formal root>/code, the units' --working-directory) -- a forbidden root (a code tree's data/).  The fix: firewall_c.remove_tree (absolute paths,
nothing opened); no policy changed.

The world: LP-B's LPBA40 world with the sealed lists at their production paths and the code tree at <formal root>/code with no data/ and as the working directory
(_lpba_world.formal_checkout), a synthetic frozen tree (_lpba_world.synthetic_frozen) for the REAL firewall_c.prepare (the private copy, the frozen prepare as a child process -- a
stand-in script --, the copy's removal).  No real subject's file, no real list, no unit."""
import json
import os
from pathlib import Path
import shutil

import pytest

from . import _lpba_world as L
from . import _world as W

R, he, hc, hb, review, hold, lc, S = L.R, L.he, L.hc, L.hb, L.review, L.hold, L.lc, L.S
FORMAL, FEAS, PAIR, FEAS_PAIR = L.FORMAL, L.FEAS, L.PAIR, L.FEAS_PAIR
import firewall_c as fc  # noqa: E402
import hb_configc as C  # noqa: E402
from .test_hb_configc import Launcher, kernel, probe_of  # noqa: E402

la = hold._lpba()


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'enable_jit', lambda ctx: False)
    w = L.make(tmp_path, monkeypatch)
    w.prod = L.production_lists(w, monkeypatch)
    L.prepare(w, monkeypatch, FORMAL, hi=False, atropos=False)
    L.prepare(w, monkeypatch, FEAS, hi=False, atropos=False)
    w.code = L.formal_checkout(w, monkeypatch)
    w.frozen = L.synthetic_frozen(tmp_path, monkeypatch)
    monkeypatch.setattr(R.core, 'frozen_api', lambda: __import__('types').SimpleNamespace(PreoptConfig=lambda **k: dict(k), four_level_grids=None, downsample_array=None))
    return w


def the_stop(world):
    """The path the feasibility run3 stop names: <checkout>/data, which does not exist (no data link, R1.2)."""
    return str(Path(world.code)/'data')


def no_relative_names(events, world):
    """No event names a path of the checkout that only a bare name resolved against the working directory gives (the copy's subdirectories: jd_curl, three_d, solvers, backends, scripts,
    data, stage_L2R_val_v1, the pair's slot)."""
    bare = {'jd_curl', 'three_d', 'solvers', 'backends', 'scripts', 'data', 'stage_L2R_val_v1', PAIR, FEAS_PAIR}
    return [e for e in events if Path(e['path']).parent == Path(world.code) and Path(e['path']).name in bare]


# ---- the mechanism --------------------------------------------------------------------------------------------------------------------------------
def test_lph_rmtree_names_subdirectories_against_the_working_directory_and_remove_tree_does_not(tmp_path, monkeypatch):
    """Under an ArmSpy whose code tree is the working directory: shutil.rmtree of an own directory holding a data/ subdirectory is refused at <cwd>/data (its fd-based walk opens 'data'
    by its bare name); firewall_c.remove_tree removes the same tree with no breach, nothing opened (its listings are of its own directories by absolute path)."""
    cwd, own = tmp_path/'checkout', tmp_path/'own'
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    policy = la.ArmPolicy(name='lp-h mechanism', write_inside=[own], code_roots=[cwd], results=[tmp_path/'results'])

    def tree():
        for rel in ('jd_curl/three_d/solvers/a.py', 'scripts/b.py', 'data/three_d/stage_L2R_val_v1/x/pair.npz'):
            (own/'copy'/rel).parent.mkdir(parents=True, exist_ok=True)
            (own/'copy'/rel).write_text('x')
        return own/'copy'
    copy = tree()
    S._reset_cache()
    with la.arm_spying(policy) as spy:
        with pytest.raises(la.lc.hp.Breach, match='forbidden open of '+str(cwd/'data')):
            shutil.rmtree(copy)
    assert spy.breaches[0]['resolved'] == str(cwd/'data') and spy.breaches[0]['write'] is False
    shutil.rmtree(copy)                                                                  # outside the spy
    copy = tree()
    S._reset_cache()
    with la.arm_spying(policy) as spy:
        fc.remove_tree(copy)
    assert spy.breaches == [] and spy.n_opens == 0 and not copy.exists()                 # nothing opened at all (an allowed open of the code or an unwatched place is counted, not listed)


def build_tree(top, keep):
    """A tree for remove_tree and shutil.rmtree alike: nested directories, an empty directory chain, a file, links to a directory and a file outside the tree, a link to a directory
    inside the tree, a broken link, a FIFO."""
    for rel in ('a/b/c.txt', 'a/d.txt', 'e.txt', 'empty/deeper/', 'data/three_d/x/pair.npz'):
        path = top/rel
        if rel.endswith('/'):
            path.mkdir(parents=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('x')
    (top/'a'/'link_dir').symlink_to(keep/'sub', target_is_directory=True)
    (top/'link_file').symlink_to(keep/'sub'/'k.txt')
    (top/'a'/'link_inside').symlink_to(top/'data', target_is_directory=True)
    (top/'broken').symlink_to(top/'nowhere')
    os.mkfifo(top/'a'/'fifo')
    return top


def residue(top):
    """What is left of `top`: its relative paths (os.walk without following links), and whether `top` itself is there."""
    if not os.path.lexists(top):
        return None
    left = set()
    for root, dirs, files in os.walk(top):
        left |= {str(Path(root, n).relative_to(top)) for n in dirs+files}
    return left


def test_lph_remove_tree_removes_what_rmtree_removes_and_follows_no_link(tmp_path):
    """Against shutil.rmtree on the same tree: both remove it whole (links, the broken link and the FIFO themselves, nothing they point to); a top that is missing or a link is refused."""
    keep = tmp_path/'outside'
    (keep/'sub').mkdir(parents=True)
    (keep/'sub'/'k.txt').write_text('k')
    ours, theirs = build_tree(tmp_path/'ours', keep), build_tree(tmp_path/'theirs', keep)
    assert residue(ours) == residue(theirs) and len(residue(ours)) >= 15
    fc.remove_tree(ours)
    shutil.rmtree(theirs)
    assert residue(ours) is None and residue(theirs) is None
    assert (keep/'sub'/'k.txt').read_text() == 'k' and sorted(p.name for p in keep.rglob('*')) == ['k.txt', 'sub']
    with pytest.raises(NotADirectoryError):
        fc.remove_tree(tmp_path/'missing')
    (tmp_path/'l').symlink_to(keep, target_is_directory=True)
    with pytest.raises(NotADirectoryError):
        fc.remove_tree(tmp_path/'l')
    (tmp_path/'f').write_text('f')
    with pytest.raises(NotADirectoryError):
        fc.remove_tree(tmp_path/'f')
    assert (keep/'sub'/'k.txt').is_file() and (tmp_path/'l').is_symlink() and (tmp_path/'f').is_file()


@pytest.mark.parametrize('mode', [0o300, 0o500])
def test_lph_remove_tree_raises_where_rmtree_raises_and_leaves_what_it_leaves(tmp_path, mode):
    """The error paths: a subdirectory that cannot be listed (0o300: the walk's scandir fails, onerror raises) or whose entries cannot be removed (0o500: the unlink fails) -- remove_tree
    raises PermissionError where shutil.rmtree does, never silently; both leave the locked directory's contents and touch nothing outside the tree.  What else is left can differ: rmtree
    removes a directory's files as it meets them, remove_tree bottom up (a directory's own files after its subdirectories), so the files removed before the error depend on the order."""
    if os.geteuid() == 0:
        pytest.skip('root ignores the permission bits')
    keep = tmp_path/'outside'
    (keep/'sub').mkdir(parents=True)
    (keep/'sub'/'k.txt').write_text('k')
    tops = []
    for name in ('ours', 'theirs'):
        top = build_tree(tmp_path/name, keep)
        (top/'locked').mkdir()
        (top/'locked'/'f.txt').write_text('f')
        (top/'locked').chmod(mode)
        tops.append(top)
    try:
        with pytest.raises(PermissionError):
            fc.remove_tree(tops[0])
        with pytest.raises(PermissionError):
            shutil.rmtree(tops[1])
    finally:
        for top in tops:
            (top/'locked').chmod(0o700)
    full = residue(build_tree(tmp_path/'reference', keep)) | {'locked', 'locked/f.txt'}
    for top in tops:
        assert {'locked', 'locked/f.txt'} <= residue(top) <= full, top
    assert (keep/'sub'/'k.txt').read_text() == 'k' and sorted(p.name for p in keep.rglob('*')) == ['k.txt', 'sub']


# ---- the configuration C driver: the stop reproduced, and fixed -----------------------------------------------------------------------------------
def configc_batch(pair_set):
    batch = C.roots(pair_set)[0]/'batches'/'test'
    batch.mkdir(parents=True, exist_ok=True)
    return batch


@pytest.fixture
def driver(world, monkeypatch):
    monkeypatch.setattr(he, 'frozen_identity', lambda *a, **k: (dict(test=True), []))
    monkeypatch.setattr(la, 'code_record', lambda pair_set, pinned=None, **k: (dict(commit='c'*40, clean=True, pair_set=pair_set), []))
    S._reset_cache()
    pin, _ = R.admitted_images(L.ctx_of(FEAS, FEAS_PAIR))
    monkeypatch.setattr(W, 'grid', lambda shape=None: pin.grid)                          # the frozen run's stand-in writes a field of the pair's grid
    return world


def run_configc(pair_set=FEAS, pair=FEAS_PAIR):
    S._reset_cache()
    return C.run(pair_set, pair, batch=configc_batch(pair_set), launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel())


def test_lph_the_configuration_c_cell_stops_as_run3_did_with_shutil_rmtree(driver, monkeypatch):
    """The stop reproduced: firewall_c.prepare with shutil.rmtree (the code of F''') in the configuration C driver of the feasibility mode, under its spy, from the checkout: infra_failure,
    kind spy_breach, the breach an open of <checkout>/data, write false, forbidden as a code tree's data/; stage configc (the driver's own record directory); the copy left as evidence."""
    monkeypatch.setattr(fc, 'remove_tree', shutil.rmtree)
    rc = run_configc()
    paths = C.cell_paths(FEAS, FEAS_PAIR)
    r = he.read_json(paths['result'])
    breach = he.read_json(paths['record']/'spy_breach.json')
    assert rc == C.EXIT['infra_failure'] == 20 and (r['status'], r['kind']) == ('infra_failure', 'spy_breach')
    assert breach['pair'] == FEAS_PAIR and breach['pair_set'] == FEAS and breach['exit'] == 20
    assert [(b['path'], b['resolved'], b['write'], b['cls']) for b in breach['breaches']] == [(the_stop(driver), the_stop(driver), False, 'forbidden')]
    assert "a code tree's data/" in breach['breaches'][0]['why'] and 'shutil.py' in r['traceback'] and 'rmtree' in r['traceback']
    assert not Path(the_stop(driver)).exists() and (paths['work']/'frozen_copy').is_dir()


@pytest.mark.parametrize('pair_set, pair', [(FEAS, FEAS_PAIR), (FORMAL, PAIR)])
def test_lph_the_configuration_c_cell_passes_with_the_real_prepare(driver, monkeypatch, pair_set, pair):
    if pair_set == FORMAL:
        S._reset_cache()
        pin, _ = R.admitted_images(L.ctx_of(FORMAL, PAIR))
        monkeypatch.setattr(W, 'grid', lambda shape=None: pin.grid)
    rc = run_configc(pair_set, pair)
    paths = C.cell_paths(pair_set, pair)
    r = he.read_json(paths['result'])
    assert rc == 0 and r['status'] == 'ok', (r.get('error'), r.get('traceback'))
    opened = he.read_json(sorted(paths['record'].glob('opened_files*.json'))[0])
    assert opened['passed'] is True and not list(paths['record'].glob('spy_breach*.json'))
    assert no_relative_names(opened['events'], driver) == [] and not (paths['work']/'frozen_copy').exists()
    assert r['prepare']['frozen_files'] == 5 and 'removed after prepare' in r['prepare']['slot']
    assert not [e for e in opened['events'] if e['resolved'].startswith(str(driver.code/'archive')+'/') and e['resolved'] not in {str(p) for p in hc.sealed_list_paths()}]


# ---- the method's processes: configuration C's fallback with the real prepare (the supervisor, the third certification, the review unit's child) ---------------
class FallbackChild:
    kind = 'test'

    def __init__(self):
        self.calls = []

    def __call__(self, kind, ctx):
        self.calls.append(kind)
        d = Path(ctx['out'])/kind
        d.mkdir(parents=True, exist_ok=True)
        he.write_json(d/'run.json', dict(status='estimate_failed', reason='test (LP-H): no field, configuration C follows', oracle=None, field=None))
        return 0


@pytest.mark.parametrize('remove, breach', [(None, False), (shutil.rmtree, True)])
def test_lph_the_supervisors_configuration_c_fallback_with_the_real_prepare(world, monkeypatch, remove, breach):
    """The supervisor's configuration C fallback (run_hx.configc_fallback: derive_pair, the REAL prepare, the firewall gate, the candidate certified again) under its spy from the checkout:
    no breach; with shutil.rmtree the copy's data/ is refused at <checkout>/data (the fallback records the error and the identity follows -- the spy's breach stops the pair)."""
    if remove is not None:
        monkeypatch.setattr(fc, 'remove_tree', remove)
    monkeypatch.setattr(he, 'segmenter_start_problems', lambda s, registry=None: ([], dict(test=True)))
    ctx = L.ctx_of()
    assert ctx['paths']['frozen'] == str(world.frozen)
    S._reset_cache()
    pin, _ = R.admitted_images(ctx)
    monkeypatch.setattr(W, 'grid', lambda shape=None: pin.grid)
    child = FallbackChild()
    S._reset_cache()
    rc = R.run_pair(ctx, child=child, launcher=Launcher('ok'), certification=W.Certify().pair(), probe=W.probe)
    out = Path(ctx['out'])
    opened = he.read_json(out/'opened_files_supervisor_0.json')
    if breach:
        record = he.read_json(out/'spy_breach_0.json')
        assert rc == he.STOP_QUEUE_EXIT and record['stage'] == 'supervisor' and record['breaches'][0]['resolved'] == the_stop(world)
        return
    line = he.read_json(out/'output'/'line.json')
    assert rc == 0 and opened['passed'] is True and line['line_status'] == he.FALLBACK_C, (opened['breaches'], line.get('fallback'))
    assert no_relative_names(opened['events'], world) == [] and not (out/'fallback_configc'/'frozen_copy').exists()


@pytest.mark.parametrize('remove, breach', [(None, False), (shutil.rmtree, True)])
def test_lph_the_third_certifications_fallback_with_the_real_prepare(world, monkeypatch, remove, breach):
    if remove is not None:
        monkeypatch.setattr(fc, 'remove_tree', remove)
    ctx = L.ctx_of()
    out = Path(ctx['out'])
    out.mkdir(parents=True)
    S._reset_cache()
    pin, _ = R.admitted_images(ctx)
    monkeypatch.setattr(W, 'grid', lambda shape=None: pin.grid)
    u, _ = L.write_cell(ctx, pin)
    S._reset_cache()
    rc = R.third_cert_main(ctx, launcher=Launcher('ok'), certification=W.Certify(fail={he.array_sha256(u)}).pair())
    opened = he.read_json(out/'opened_files_third_0.json')
    if breach:
        record = he.read_json(out/'spy_breach_0.json')
        assert rc == la.SPY_BREACH_EXIT and record['stage'] == 'third' and record['breaches'][0]['resolved'] == the_stop(world)
        return
    line = he.read_json(out/'output'/'line.json')
    assert rc == 0 and opened['passed'] is True and line['line_status'] == he.FALLBACK_C, (opened['breaches'], line.get('fallback'))
    assert no_relative_names(opened['events'], world) == [] and not list(out.glob('fallback_configc*/frozen_copy'))


def test_lph_the_review_units_third_certification_falls_back_with_the_real_prepare(world, monkeypatch):
    """The review unit's child (ReviewSpy over lpba_policy, the code tree the checkout; inside it the ArmSpy of stage third): the third certificate fails, configuration C's fallback with
    the real prepare, no breach in either spy."""
    ctx = L.ctx_of()
    S._reset_cache()
    pin, _ = R.admitted_images(ctx)
    monkeypatch.setattr(W, 'grid', lambda shape=None: pin.grid)
    u, _ = L.write_cell(ctx, pin)
    rec = hb.record_dir('method', PAIR, FORMAL)
    rec.mkdir(parents=True)
    he.write_json(rec/'step4.json', dict(status=he.HX, went_on=True, written_by='test'))
    with monkeypatch.context() as m:
        m.setattr(hb, 'REPO', world.code)
        mode = review.lpba_mode(FORMAL, None)
    S._reset_cache()
    assert review.third_child_main(ctx, mode, launcher=Launcher('ok'), certification=W.Certify(fail={he.array_sha256(u)}).pair()) == 0
    outer = he.read_json(Path(ctx['out'])/'review'/'opened_files_third.json')
    inner = he.read_json(Path(ctx['out'])/'opened_files_third_0.json')
    assert outer['passed'] is True and outer['breaches'] == [] and inner['passed'] is True
    assert he.read_json(Path(ctx['out'])/'output'/'line.json')['line_status'] == he.FALLBACK_C and no_relative_names(inner['events'], world) == []
    assert not list(Path(ctx['out']).glob('fallback_configc*/frozen_copy'))                # the private copy removed
