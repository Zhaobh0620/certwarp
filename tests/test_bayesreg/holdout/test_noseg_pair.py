"""P0S I14 (the lead's ruling 4 of 11-14; prereg revision 2 item 9, checklist item 14; revision 1a item 21): experiments/holdout/noseg_pair.py on synthetic development pairs in the XB sandbox
(the development stage and the results root moved into a temporary directory; real ids of the allowlist, synthetic arrays).  Every member decoded from any npz during a run is recorded by a stand-in
of numpy's NpzFile.__getitem__ (independent of the tool's own guard): the original's label members are never decoded."""
import json
from pathlib import Path

import numpy as np
import pytest

from ..xb import _world as W

import noseg_pair as ns  # noqa: E402   (experiments/holdout is on sys.path through the XB world)

hc = W.hc
PAIR = W.DEV_PAIR
LABEL_KEYS = ('fixed_seg4', 'fixed_seg35', 'moving_seg4', 'moving_seg35')
REAL_ROOT = hc.REAL_RESULTS/ns.ROOT_NAME


def code():
    return dict(root='tree', commit='c'*40, clean=True, status='', untracked=[])


@pytest.fixture
def decoded(monkeypatch):
    """[(file, member)] of every npz member decoded during the test, whatever opened the archive."""
    from numpy.lib import _npyio_impl
    rows = []
    real = _npyio_impl.NpzFile.__getitem__

    def getitem(self, key):
        rows.append((str(self.zip.filename), key))
        return real(self, key)
    monkeypatch.setattr(_npyio_impl.NpzFile, '__getitem__', getitem)
    return rows


@pytest.fixture
def tripwire():
    before = REAL_ROOT.exists()
    yield
    assert REAL_ROOT.exists() == before, 'a test wrote the real item 14 root'


def original_path():
    return Path(hc.DEV_STAGE)/PAIR/'pair.npz'


@pytest.mark.parametrize('initial_u', [True, False])
def test_the_copy_holds_no_label_member_and_the_three_checks_pass(initial_u, decoded, tripwire):
    W.write_stage_pair(hc.DEV_STAGE, PAIR, seg=True, initial_u=initial_u)
    original = original_path()
    with np.load(original) as z:
        assert set(LABEL_KEYS) <= set(z.files)                  # the synthetic original has its label members (listed here by the test, not by the tool)
    decoded.clear()
    out = ns.root()/'t'
    hook, getitem = __import__('bayesreg.io.pair', fromlist=['_open_npz'])._open_npz, np.lib.npyio.NpzFile.__getitem__
    rc, record = ns.run(PAIR, out, code_identity=code, log=lambda m: None)
    assert rc == 0 and record['passed'] is True and record['problems'] == [] and record['steps_not_run'] == []
    assert __import__('bayesreg.io.pair', fromlist=['_open_npz'])._open_npz is hook and np.lib.npyio.NpzFile.__getitem__ is getitem     # both guards taken off after the run
    assert all(record['steps'][s]['passed'] for s in ('load_original', 'strip', 'load_copy', 'load_pair', 'firewall_c', 'external_runner'))
    wanted = ['fixed', 'moving', 'initial_u'] if initial_u else ['fixed', 'moving']
    assert {k for f, k in decoded if f == str(original)} == set(wanted)                           # nothing else of the original was decoded, by anyone
    assert set(record['original_members_asked']) == set(wanted) == set(record['original_members_decoded']) and record['original_refused'] == []
    copy = out/'stage'/PAIR/'pair.npz'
    with np.load(copy) as z, np.load(original) as o:
        assert list(z.files) == wanted and all(z[k].dtype == o[k].dtype and z[k].tobytes() == o[k].tobytes() for k in wanted)
    manifest = json.loads((copy.parent/'manifest.json').read_text())
    from bayesreg.io.pair import MANIFEST_FIELDS
    assert sorted(manifest) == sorted(MANIFEST_FIELDS) and set(manifest['sources']) == {'fixed', 'moving'}            # exactly load_pair's fields (revision 2, minor 4)
    assert manifest['pair_sha256'] == hc.sha256_file(copy) == record['steps']['strip']['pair_sha256'] == record['steps']['load_copy']['pair_sha256']
    assert record['steps']['load_original']['pair_sha256'] == hc.sha256_file(original) != hc.sha256_file(copy)
    load = record['steps']['load_pair']
    assert load['keys_read'] == dict(original=wanted, copy=wanted) and all(load['items'].values())
    fw = record['steps']['firewall_c']
    assert fw['pair_npz_sha256']['from_original'] == fw['pair_npz_sha256']['from_copy'] and all(fw['same_members'].values())
    assert all(c['passed'] for c in fw['firewall_check'].values())
    xb = record['steps']['external_runner']
    assert xb['inputs_sha256']['original'] == xb['inputs_sha256']['copy'] and xb['check_pair_input']['pair_sha256'] == hc.sha256_file(copy)
    assert json.loads((out/ns.RECORD).read_text())['passed'] is True and record['code'] == code()


def test_the_guard_refuses_a_label_member_a_listing_and_an_iteration(tmp_path, decoded):
    from bayesreg.io import pair as pair_module
    W.write_stage_pair(hc.DEV_STAGE, PAIR, seg=True)
    original = original_path()
    other = tmp_path/'other.npz'
    np.savez(other, fixed_seg4=np.zeros(2))
    real = pair_module._open_npz
    with ns.MemberGuard(original) as guard:
        with pair_module._open_npz(original) as npz:
            assert 'initial_u' in npz and 'fixed_seg4' in npz              # membership reads the archive's directory only
            assert npz['fixed'].shape == W.SHAPE
            for action in (lambda: npz['fixed_seg4'], lambda: npz.files, lambda: list(npz), lambda: len(npz), lambda: npz.keys(), lambda: dict(npz)):
                with pytest.raises(ns.MemberRefused):
                    action()
        with pair_module._open_npz(other) as npz:                          # another file opens as before
            assert npz['fixed_seg4'].shape == (2,)
    assert pair_module._open_npz is real
    assert guard.members() == ['fixed'] and [r['refused'] for r in guard.log if 'refused' in r] == ['fixed_seg4']
    assert (str(original), 'fixed_seg4') not in decoded


def test_a_reader_that_asks_for_a_label_member_fails_the_run_before_it_is_decoded(monkeypatch, decoded, tripwire):
    """A mutation of one of the readers under check (firewall_c's): asking the original for a label member is refused by the tool's guard, recorded, and the run does not pass."""
    import firewall_c as fc
    from bayesreg.io import pair as pair_module
    W.write_stage_pair(hc.DEV_STAGE, PAIR, seg=True)
    real = fc.read_image_members

    def greedy(path):
        out = real(path)
        with pair_module._open_npz(path) as npz:
            out['fixed_seg4'] = npz['fixed_seg4']
        return out
    monkeypatch.setattr(fc, 'read_image_members', greedy)
    decoded.clear()
    rc, record = ns.run(PAIR, ns.root()/'greedy', code_identity=code, log=lambda m: None)
    assert rc == 1 and record['passed'] is False and record['steps']['firewall_c']['passed'] is False
    assert [r['refused'] for r in record['original_refused']] == ['fixed_seg4'] and any('fixed_seg4' in p for p in record['problems'])
    assert not [k for f, k in decoded if f == str(original_path()) and k in LABEL_KEYS]


def test_a_runner_that_needs_a_label_member_fails_the_run(monkeypatch, tripwire):
    """What the step is for: a reader of the external runner that still asks for a label member (as the old harness's sanity() did) raises on the copy (KeyError: the stage has no such
    member); the step and the run fail.  (On the original the same read is refused by the guard before it is decoded: the next tests.)"""
    import xb_core as xc
    W.write_stage_pair(hc.DEV_STAGE, PAIR, seg=True)
    real = xc.derive_inputs

    def with_sanity(path, out_path, *, shape):
        if Path(path) != original_path():
            with np.load(path) as z:
                z['fixed_seg4']
        return real(path, out_path, shape=shape)
    monkeypatch.setattr(xc, 'derive_inputs', with_sanity)
    rc, record = ns.run(PAIR, ns.root()/'sanity', code_identity=code, log=lambda m: None)
    step = record['steps']['external_runner']
    assert rc == 1 and step['passed'] is False and 'KeyError' in step['error'] and record['passed'] is False and record['original_refused'] == []


def test_a_reader_that_swallows_the_guards_refusal_still_fails_the_run(monkeypatch, tripwire):
    """The run's last check: a member refused by the guard fails the run even when the reader caught the refusal and went on (its step then passes)."""
    import firewall_c as fc
    from bayesreg.io import pair as pair_module
    W.write_stage_pair(hc.DEV_STAGE, PAIR, seg=True)
    real = fc.read_image_members

    def tries(path):
        with pair_module._open_npz(path) as npz:
            try:
                npz['moving_seg35']
            except (ns.MemberRefused, KeyError):           # the guard on the original; the copy has no such member
                pass
        return real(path)
    monkeypatch.setattr(fc, 'read_image_members', tries)
    rc, record = ns.run(PAIR, ns.root()/'swallowed', code_identity=code, log=lambda m: None)
    assert record['steps']['firewall_c']['passed'] is True and rc == 1 and record['passed'] is False
    assert any('moving_seg35' in p for p in record['problems'])


def test_load_pair_differences_are_problems():
    from types import SimpleNamespace
    a = np.zeros((2, 2), dtype=np.float32)

    def pin(keys, fixed, initial=True):
        return SimpleNamespace(provenance=dict(keys_read=keys, initial_u_present=initial), fixed=fixed, moving=a, grid_metadata=dict(shape_zyx=[2, 2]), fixed_subject='f', moving_subject='m',
                               source_sha256=dict(fixed='1', moving='2'))
    assert ns.compare_loads(pin(['fixed', 'moving'], a), pin(['fixed', 'moving'], a.copy()))['passed'] is True
    result = ns.compare_loads(pin(['fixed', 'moving', 'initial_u'], a), pin(['fixed', 'moving'], a.astype(np.float64), initial=False))
    assert result['passed'] is False and {p.split(': ')[1].split(' ')[0] for p in result['problems']} == {'keys_read', 'initial_u_present', 'fixed'}


def test_a_derived_pair_that_differs_fails_the_firewall_step(monkeypatch, tripwire):
    """firewall_c's derived pair from the copy changed after the fact (moving + 1): firewall_check fails on it, the members and the files differ -- three problems."""
    import firewall_c as fc
    W.write_stage_pair(hc.DEV_STAGE, PAIR, seg=True)
    real = fc.derive_pair

    def derive(pin, out_dir, *, pair):
        record = real(pin, out_dir, pair=pair)
        if Path(out_dir).name == 'from_copy':
            with np.load(Path(out_dir)/'pair.npz') as z:
                arrays = {k: np.array(z[k]) for k in z.files}
            arrays['moving'] = arrays['moving']+1
            np.savez_compressed(Path(out_dir)/'pair.npz', **arrays)
        return record
    monkeypatch.setattr(fc, 'derive_pair', derive)
    rc, record = ns.run(PAIR, ns.root()/'derived', code_identity=code, log=lambda m: None)
    problems = record['steps']['firewall_c']['problems']
    assert rc == 1 and record['steps']['firewall_c']['passed'] is False
    assert any(p.startswith('firewall_check (from_copy)') for p in problems) and 'moving: the derived members differ' in problems
    assert any(p.startswith('the derived pair.npz files differ') for p in problems)


def test_method_inputs_that_differ_fail_the_runner_step(monkeypatch, tripwire):
    """XB's reader gives the copy's moving + 1: the method inputs from the copy are not those from the original (members and file) -- two problems."""
    import xb_core as xc
    W.write_stage_pair(hc.DEV_STAGE, PAIR, seg=True)
    real = xc.read_image_members

    def read(path):
        out = real(path)
        return dict(out, moving=out['moving']+1) if 'stage' in Path(path).parts and ns.ROOT_NAME in str(path) else out
    monkeypatch.setattr(xc, 'read_image_members', read)
    rc, record = ns.run(PAIR, ns.root()/'inputs', code_identity=code, log=lambda m: None)
    problems = record['steps']['external_runner']['problems']
    assert rc == 1 and 'moving: the method inputs differ' in problems and any(p.startswith('the method inputs files differ') for p in problems)


def test_a_copy_that_is_not_the_original_fails_the_strip(monkeypatch, tripwire):
    """A mutation of the writer: the copy's moving is not the original's (one added to it): the strip fails and the steps on the copy do not run."""
    W.write_stage_pair(hc.DEV_STAGE, PAIR, seg=True)
    savez = np.savez
    monkeypatch.setattr(np, 'savez_compressed', lambda f, **arrays: savez(f, **{k: (v+1 if k == 'moving' else v) for k, v in arrays.items()}))
    rc, record = ns.run(PAIR, ns.root()/'shifted', code_identity=code, log=lambda m: None)
    assert rc == 1 and record['steps']['strip']['passed'] is False and any('moving' in p for p in record['problems'])
    assert 'load_copy' not in record['steps'] and set(record['steps_not_run']) >= {'load_copy', 'load_pair', 'firewall_c'}


REFUSALS = {'not a development pair': 'admission', 'not among the 19 (a hold-out row)': 'admission', 'outside the root': 'output', 'the root itself': 'output', 'exists': 'output',
            'not a pair name': 'admission', 'not a pair name, no --out': 'admission', 'not OASIS ids, no --out': 'admission', 'a symbolic link out of the root': 'output', 'a dangling link': 'output', 'a dirty tree': 'code',
            'no HEAD': 'code'}


def refusal_records():
    return sorted((ns.root()/'refusals').glob('*.json'))


@pytest.mark.parametrize('case', sorted(REFUSALS))
def test_refusals_before_any_file_of_the_pair(case, decoded, tripwire, tmp_path):
    """Every refusal before a file of the pair is read: exit 2, one refusal record under <root>/refusals/ with the stage, nothing decoded, no output directory (revision 2: minors 1, 2, 3)."""
    W.write_stage_pair(hc.DEV_STAGE, PAIR, seg=True)
    pair, out, ident = PAIR, ns.root()/case.replace(' ', '_').replace(',', ''), code
    if case == 'not a development pair':
        pair = 'OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1'
    elif case == 'not among the 19 (a hold-out row)':
        pair = 'OASIS_OAS1_0423_MR1__OASIS_OAS1_0099_MR1'           # a row of holdout_B_pairs.tsv (ids only, as tests/test_bayesreg/eval/test_holdout_entry.py uses it): refused as not one of the 19
    elif case == 'outside the root':
        out = tmp_path/'elsewhere'
    elif case == 'the root itself':
        out = ns.root()
        out.mkdir(parents=True)
    elif case == 'exists':
        out.mkdir(parents=True)
    elif case == 'not a pair name':
        pair = 'OASIS_OAS1_0438_MR1'
    elif case == 'not a pair name, no --out':                        # minor 1: the default directory is built from the pair's name
        pair, out = 'not_a_pair', None
    elif case == 'not OASIS ids, no --out':                          # minor 1: <fixed>__<moving>, but no short name for the default directory (hc.short_pair)
        pair, out = 'A__B', None
    elif case == 'a symbolic link out of the root':                  # minor 2: judged on the resolved path
        elsewhere = tmp_path/'elsewhere'
        elsewhere.mkdir()
        ns.root().mkdir(parents=True)
        (ns.root()/'link').symlink_to(elsewhere, target_is_directory=True)
        out = ns.root()/'link'/'run'
    elif case == 'a dangling link':
        ns.root().mkdir(parents=True)
        (ns.root()/'dangling').symlink_to(ns.root()/'gone')
        out = ns.root()/'dangling'
    elif case == 'a dirty tree':                                     # minor 3
        ident = lambda: dict(code(), clean=False, status=' M experiments/holdout/noseg_pair.py')  # noqa: E731
    else:
        ident = lambda: dict(code(), commit=None, clean=False)  # noqa: E731
    decoded.clear()
    lines = []
    rc, record = ns.run(pair, out, code_identity=ident, log=lines.append)
    assert rc == 2 and record is None and lines and lines[0].startswith('refused')
    records = refusal_records()
    assert len(records) == 1 and json.loads(records[0].read_text())['stage'] == REFUSALS[case] and json.loads(records[0].read_text())['schema'] == ns.SCHEMA_REFUSAL
    assert decoded == []
    if out is not None:
        assert not (out/ns.RECORD).exists()
    if case == 'a symbolic link out of the root':
        assert list((tmp_path/'elsewhere').iterdir()) == []
    elif case not in ('exists', 'the root itself', 'a dangling link') and out is not None:
        assert not out.exists()
    if case not in ('exists', 'the root itself', 'a symbolic link out of the root', 'a dangling link'):
        assert [p.name for p in ns.root().iterdir()] == ['refusals']


def synthetic_sealed(tmp_path, monkeypatch, **lists):
    """Synthetic sealed lists (as tests/test_bayesreg/eval/_holdout_world.py writes them): bayesreg.io.sealed.REPO and SEALED_LISTS pinned to a temporary repository; `lists` overrides the ids
    of holdout_B_v1, H1, H4 (each defaults to an id of no development pair)."""
    import bayesreg.io.sealed as S
    from jd_curl.three_d.nifti import file_sha256
    repo = tmp_path/'sealed_repo'
    (repo/'lists').mkdir(parents=True)
    specs = []
    for name, default in (('holdout_B_v1', 'OASIS_OAS1_9001_MR1'), ('H1', 'OASIS_OAS1_9002_MR1'), ('H4', 'OASIS_OAS1_9003_MR1')):
        ids = lists.get(name) or (default,)
        path = repo/'lists'/f'{name}.txt'
        path.write_text(''.join(f'{s}\n' for s in ids), encoding='utf-8')
        specs.append(dict(name=name, path=f'lists/{name}.txt', sha256=file_sha256(path), columns=None, count=len(set(ids))))
    h3 = repo/'lists'/'H3.tsv'
    h3.write_text('fixed\tmoving\nIBSR_01\tIBSR_02\n', encoding='utf-8')
    specs.append(dict(name='H3', path='lists/H3.tsv', sha256=file_sha256(h3), columns=('fixed', 'moving'), count=2))
    monkeypatch.setattr(S, 'REPO', repo)
    monkeypatch.setattr(S, 'SEALED_LISTS', tuple(specs))
    S._reset_cache()


@pytest.mark.parametrize('sealed_in, stage', [('holdout_B_v1', 'admission'), ('H4', 'never')])
def test_a_development_id_on_a_sealed_list_is_refused_at_the_guard(sealed_in, stage, monkeypatch, decoded, tripwire, tmp_path):
    """Revision 2, minor 6: the admission's sealed-list step itself (holdout_common.admit_dev_pair after the 19 pairs): with synthetic sealed lists that hold the pair's moving subject, the
    run is refused there (H1-H4: stage never), before any file of the pair, with its refusal record."""
    W.write_stage_pair(hc.DEV_STAGE, PAIR, seg=True)
    synthetic_sealed(tmp_path, monkeypatch, **{sealed_in: ('OASIS_OAS1_0439_MR1',)})
    decoded.clear()
    rc, record = ns.run(PAIR, ns.root()/'sealed', code_identity=code, log=lambda m: None)
    refusal = json.loads(refusal_records()[0].read_text())
    assert rc == 2 and record is None and refusal['stage'] == stage and 'sealed lists' in refusal['message'] and sealed_in in refusal['message']
    assert decoded == [] and not (ns.root()/'sealed').exists()
    synthetic_sealed(tmp_path/'clean', monkeypatch)                    # the same synthetic lists without the pair's subjects: the run passes the guard
    rc, record = ns.run(PAIR, ns.root()/'clean', code_identity=code, log=lambda m: None)
    assert rc == 0 and record['admission']['sealed_lists'][0]['path'] == 'lists/holdout_B_v1.txt'


def test_a_reader_that_bypasses_the_hook_is_caught_at_numpys_level(monkeypatch, decoded, tripwire):
    """Revision 2, minor 5: a reader under check that opens the original with np.load itself (not through the loader's hook) and asks for a label member: refused before it is decoded,
    recorded at the NpzFile level; the step fails, and the run fails even when the reader swallows the refusal."""
    import firewall_c as fc
    W.write_stage_pair(hc.DEV_STAGE, PAIR, seg=True)
    real = fc.read_image_members

    def bypass(swallow):
        def read(path):
            with np.load(path, allow_pickle=False) as z:
                try:
                    z['fixed_seg35']
                except (ns.MemberRefused, KeyError):
                    if not swallow:
                        raise
            return real(path)
        return read
    for swallow in (False, True):
        monkeypatch.setattr(fc, 'read_image_members', bypass(swallow))
        decoded.clear()
        rc, record = ns.run(PAIR, ns.root()/f'bypass_{swallow}', code_identity=code, log=lambda m: None)
        assert rc == 1 and record['passed'] is False and record['steps']['firewall_c']['passed'] is swallow
        refused = [(r['refused'], r['level']) for r in record['original_refused']]
        assert refused and set(refused) == {('fixed_seg35', 'NpzFile')} and (swallow or len(refused) == 1)      # swallowed: firewall_c reads the original twice (derive, check)
        assert not [k for f, k in decoded if f == str(original_path()) and k in LABEL_KEYS] and any('fixed_seg35' in p for p in record['problems'])


def test_the_default_output_directory(decoded, tripwire):
    """No --out: a new directory <root>/<UTC>_<short pair> (the form d1_expected.md section 8 gives)."""
    W.write_stage_pair(hc.DEV_STAGE, PAIR, seg=True)
    rc, record = ns.run(PAIR, None, code_identity=code, log=lambda m: None)
    out = Path(record['out'])
    assert rc == 0 and out.parent == ns.root().resolve() and out.name.endswith('_0438-0439') and (out/ns.RECORD).is_file()


def test_the_real_root_is_refused_in_the_sandbox():
    assert 'real results root' in ns.out_problem(hc.REAL_RESULTS/ns.ROOT_NAME/'x')
    assert ns.out_problem(ns.root()/'x') is None


def test_the_command_line(decoded, tripwire, monkeypatch):
    W.write_stage_pair(hc.DEV_STAGE, PAIR, seg=True)
    monkeypatch.setattr(hc, 'code_identity', code)                    # this checkout may be dirty while it is worked on; the code check itself is tested above
    assert ns.main(['run', '--out', str(ns.root()/'cli')]) == 0
    record = json.loads((ns.root()/'cli'/ns.RECORD).read_text())
    assert record['passed'] is True and record['pair'] == PAIR == ns.DEFAULT_PAIR and set(record['code']) >= {'commit', 'clean'}
    assert ns.main(['run', '--out', str(ns.root()/'cli')]) == 2                   # a new directory every time
