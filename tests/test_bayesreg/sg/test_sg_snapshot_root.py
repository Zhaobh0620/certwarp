"""Contract C23.7-4 revision 6n (D95 (b), the lead's ruling 2026-11-11 ~3:28 AM CST, BZ can change): where the guard reads the sealed lists (bayesreg.io.sealed.guard_root).

Rule 7 of 6n, a positive and a negative test for each rule:
  1. the four lists all in the code tree: read there; the record, the lists and the cache are byte for byte those of the code before 6n (ff857d06's sealed.py run side by side);
  2. none of them in the code tree and the tree a training snapshot verify_snapshot verifies: read from SNAPSHOT.json's source_repo with the same pins; one more record item (the root and the
     way it was reached); the cache key carries that root; the snapshot is verified once per process;
  3. fail closed otherwise: some lists in the tree; no SNAPSHOT.json; a SNAPSHOT.json that does not verify, cannot be read, has no source_repo, a relative one, one that is not a directory or the
     tree itself; a source_repo that misses a list or holds another one;
  4. the snapshots carry no list (SNAPSHOT_PATHS* unchanged: tests/test_bayesreg/r2/test_r2_smoke.py checks R2's table; the three safety assertions are not touched);
  6. D96 (a) unchanged: through source_repo the guard still refuses a sealed subject.
Synthetic trees (the four lists of tests/test_bayesreg/sg/_sg_world.py's ids, pinned with monkeypatch as the SG world does) for the rules in this process; and real snapshots: make_snapshot of
this worktree's HEAD (a shared clone of it, clean, as the lead's r2_code_<sha8>; the snapshot is of committed code, as make_snapshot always is) with R2's path table, in a mktemp directory
(no test-runner word in its path), then the guard and `cli check-inputs --profile r2_small --require-snapshot` (smoke_r2.prepare's call) in a subprocess from the snapshot's root.  The real
lists are read only by sealed.py, the guard's own reading (C23.6-5): no test opens them or hashes them; a list is made "missing" by renaming it, never by reading it.  CPU; no label, no image.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

import pytest

import bayesreg.io.sealed as S
from bayesreg.io.guard import check_subject, load_dev_allowlist
from bayesreg.seg.unet import snapshot as SNAP
from bayesreg.seg.unet import synthetic
from bayesreg.seg.unet.snapshot import make_snapshot

from . import _sg_world as W

ROOT = W.ROOT
BASE = 'ff857d06'                         # main before revision 6n: its sealed.py is the reference of rule 1
MAIN_DATA = Path('/home/rmcfz/code/python_projects/seq_vp_tag/data')
sys.path.insert(0, str(ROOT/'experiments'/'r2'))
import run_r2  # noqa: E402


@pytest.fixture(autouse=True)
def _forget_the_cache():
    S._reset_cache()
    yield
    S._reset_cache()


# ---- synthetic trees ------------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A synthetic source repository with the four synthetic lists and code.txt, committed; a training snapshot of it (make_snapshot, code.txt only: no list); the pins are the synthetic
    lists' (monkeypatch); the guard's code tree is the snapshot."""
    src = tmp_path/'src'
    src.mkdir()
    W.git(src, 'init', '-q')
    W.write_text(src/'lists/b.txt', ''.join(f'{s}\n' for s in W.HOLDOUT))
    W.write_text(src/'lists/h1.txt', ''.join(f'{s}\n' for s in W.H1))
    W.write_text(src/'lists/h4.txt', '# H4 (synthetic)\n'+''.join(f'{s}\n' for s in W.H4))
    W.write_text(src/'lists/h3.tsv', 'fixed\tmoving\n'+f'{W.H3[0]}\t{W.H3[1]}\n')
    W.write_text(src/'code.txt', 'code\n')
    W.commit_all(src, 'the synthetic source')
    specs = (dict(name='holdout_B_v1', path='lists/b.txt', sha256=W.file_sha256(src/'lists/b.txt'), columns=None, count=len(W.HOLDOUT)),
             dict(name='H1', path='lists/h1.txt', sha256=W.file_sha256(src/'lists/h1.txt'), columns=None, count=len(W.H1)),
             dict(name='H4', path='lists/h4.txt', sha256=W.file_sha256(src/'lists/h4.txt'), columns=None, count=len(W.H4)),
             dict(name='H3', path='lists/h3.tsv', sha256=W.file_sha256(src/'lists/h3.tsv'), columns=('fixed', 'moving'), count=len(W.H3)))
    snap = tmp_path/'snap'
    record = make_snapshot(src, snap, paths=('code.txt',))
    W.pin(monkeypatch, snap, specs)
    return types.SimpleNamespace(src=src, snap=snap, specs=specs, record=record)


def snapshot_document(t):
    return json.loads((t.snap/SNAP.SNAPSHOT_FILE).read_text(encoding='utf-8'))


def rewrite_snapshot_document(t, change):
    document = snapshot_document(t)
    change(document)
    (t.snap/SNAP.SNAPSHOT_FILE).write_text(json.dumps(document), encoding='utf-8')


def test_rule_2_a_verified_snapshot_without_the_lists_reads_them_from_its_source_repo(tree, monkeypatch):
    lists = S.guard_sealed_lists()
    source = Path(tree.record['source_repo'])
    assert source == tree.src.resolve() and not any((tree.snap/spec['path']).exists() for spec in tree.specs)
    assert dict(lists.subjects) == dict(S.load_sealed_lists(tree.src).subjects)
    assert [dict(r) for r in lists.record[:-1]] == S.load_sealed_lists(tree.src).record                                  # the same lists, the same pins
    assert dict(lists.record[-1]) == dict(name='root', path=str(source), sha256=None, subjects=None, read_via=S.VIA_SNAPSHOT, code_tree=str(tree.snap),
                                          snapshot_commit=tree.record['commit'], snapshot_json_sha256=hashlib.sha256((tree.snap/SNAP.SNAPSHOT_FILE).read_bytes()).hexdigest())
    key = S._cache_key()
    assert list(S._CACHE) == [key+((S.VIA_SNAPSHOT, str(source)),)] and S._ROOTS == {key: key+((S.VIA_SNAPSHOT, str(source)),)}      # rule 5: the key carries the root
    calls = []
    monkeypatch.setattr(SNAP, 'verify_snapshot', lambda root: calls.append(root))
    assert S.guard_sealed_lists() is lists and calls == []                                                                 # cached: not verified again in this process


def test_rule_6_d96_a_is_unchanged_through_source_repo(tree):
    """A sealed subject wrongly put on the allowlist: allows False and check_subject refuses it, the lists read through source_repo; a development subject is admitted."""
    allowlist = load_dev_allowlist(W.write_json(tree.src.parent/'mistaken.json', synthetic.allowlist_document([('dev', W.DEV+[W.HOLDOUT[0]], False)])))
    assert allowlist.allows(W.DEV[0]) and not allowlist.allows(W.HOLDOUT[0])
    assert check_subject(W.DEV[0], allowlist).admitted and not check_subject(W.HOLDOUT[0], allowlist).admitted
    assert S.guard_sealed_lists().record[-1]['name'] == 'root'


def test_rule_1_a_full_tree_reads_there_without_the_item(tree, monkeypatch):
    """The source repository itself as the code tree (all four lists there): (src, None); the record has no root item, the key is the key of before."""
    monkeypatch.setattr(S, 'REPO', tree.src)
    S._reset_cache()
    assert S.guard_root() == (tree.src, None)
    lists = S.guard_sealed_lists()
    assert [dict(r) for r in lists.record] == S.load_sealed_lists(tree.src).record and lists.record[-1]['name'] == 'H2'
    assert list(S._CACHE) == [S._cache_key()] and S._ROOTS == {}


def test_rule_1_on_the_real_tree_is_byte_for_byte_the_code_before_6n(tmp_path):
    """The real code tree (all four lists there): the new guard's record, lists and cache key against ff857d06's sealed.py run side by side (both read the lists through their own loader)."""
    source = subprocess.run(['git', 'show', f'{BASE}:bayesreg/io/sealed.py'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    path = tmp_path/'sealed_before_6n.py'
    path.write_text(source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('sealed_before_6n', path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.REPO = S.REPO
    assert 'guard_root' not in source and old.SEALED_LISTS == S.SEALED_LISTS
    before, after = old.guard_sealed_lists(), S.guard_sealed_lists()
    assert json.dumps([dict(r) for r in after.record]) == json.dumps([dict(r) for r in before.record])
    assert {k: sorted(v) for k, v in after.subjects.items()} == {k: sorted(v) for k, v in before.subjects.items()}
    assert list(S._CACHE) == list(old._CACHE) and S._ROOTS == {}


def _placeholder(t):
    path = t.snap/t.specs[0]['path']                                    # a name only: the file is empty, no list is read or copied
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('')


def _source(value):
    return lambda t: rewrite_snapshot_document(t, lambda d: d.update(source_repo=value(t) if callable(value) else value))


def _rename_in_source(t):
    (t.src/t.specs[2]['path']).rename(t.src/(t.specs[2]['path']+'.away'))


def _other_list_in_source(t):
    with (t.src/t.specs[1]['path']).open('a', encoding='utf-8') as stream:          # the synthetic list (not a real one)
        stream.write(f'{synthetic.oasis(990)}\n')


@pytest.mark.parametrize('change, words', [
    (_placeholder, 'only some of the sealed lists are in the code tree'),                                                         # rule 3: some in the tree
    (lambda t: (t.snap/SNAP.SNAPSHOT_FILE).rename(t.snap/'SNAPSHOT.away'), 'has no SNAPSHOT.json'),                               # rule 3: a partial export
    (lambda t: (t.snap/'code.txt').write_text('changed\n'), 'is not a verified training snapshot (CodeNotClean'),               # rule 3: does not verify
    (lambda t: (t.snap/SNAP.SNAPSHOT_FILE).write_text('{"format":', encoding='utf-8'), 'is not a verified training snapshot (JSONDecodeError'),   # cannot be read
    (lambda t: rewrite_snapshot_document(t, lambda d: d.pop('source_repo')), 'is not a verified training snapshot (KeyError'),    # no source_repo
    (_source('relative/src'), 'not an absolute path'),                                                                             # rule 2: absolute
    (_source(None), 'not an absolute path'),
    (_source(lambda t: str(t.src.parent/'no_such_dir')), 'not a directory'),                                                      # rule 2: a directory
    (_source(lambda t: str(t.snap)), 'the code tree itself'),                                                                      # rule 2: not the tree itself
    (_rename_in_source, 'sealed list H4: lists/h4.txt missing'),                                                                   # rule 2: a list missing in source_repo
    (_other_list_in_source, 'sealed list H1: sha256'),                                                                             # rule 2: another list
])
def test_every_other_case_fails_closed(tree, change, words):
    change(tree)
    with pytest.raises(S.SealedListError) as refused:
        S.guard_sealed_lists()
    assert words in str(refused.value), str(refused.value)
    assert S.sealed_cache_is_empty() and S._ROOTS == {}                                                                            # nothing cached


def test_the_positive_for_source_repo_checks_is_the_unchanged_snapshot(tree):
    """The negatives above each change one thing of a snapshot that passes as made (rule 2's positive)."""
    assert S.guard_root()[0] == tree.src.resolve() and S.guard_root()[1]['code_tree'] == str(tree.snap)


# ---- real snapshots of this worktree's HEAD ----------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def real():
    """A shared clone of this worktree at HEAD (clean; with a data link as the lead's checkout has, when the main checkout has one) and a training snapshot of it with R2's path table, in a
    mktemp directory; removed afterwards."""
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True, env=W.GIT_ENV).stdout.strip()
    base = Path(tempfile.mkdtemp(prefix='sg_snapshot_root_'))
    assert 'pytest' not in str(base)
    try:
        src = base/'src'
        subprocess.run(['git', 'clone', '--quiet', '--shared', '--no-checkout', str(ROOT), str(src)], check=True, capture_output=True, env=W.GIT_ENV)
        subprocess.run(['git', '-C', str(src), 'checkout', '--quiet', '--detach', head], check=True, capture_output=True, env=W.GIT_ENV)
        data = MAIN_DATA.is_symlink() and MAIN_DATA.is_dir()
        if data:
            os.symlink(os.readlink(MAIN_DATA), src/'data')
        record = make_snapshot(src, base/'snap', paths=run_r2.SNAPSHOT_PATHS_R2)
        yield types.SimpleNamespace(base=base, src=src, snap=base/'snap', record=record, head=head, data=data)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def child_env(snap=None):
    """smoke_r2.prepare's environment (os.environ, no GPU, no bytecode); PYTHONPATH the snapshot only (or none), so that the snapshot's own bayesreg is imported."""
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if snap is not None:
        env['PYTHONPATH'] = str(snap)
    return env


GUARD = ('import json, sys\nfrom bayesreg.io import sealed\n'
         'lists = sealed.guard_sealed_lists()\n'
         'print(json.dumps(dict(repo=str(sealed.REPO), record=[dict(r) for r in lists.record])))\n')


def test_the_snapshot_carries_no_list_and_the_guard_reads_them_from_source_repo(real):
    assert real.record['commit'] == real.head and real.record['source_repo'] == str(real.src.resolve())
    assert not [spec['path'] for spec in S.SEALED_LISTS if (real.snap/spec['path']).exists()]                                     # rule 4
    assert not [p for p in real.record['files'] if any(p == spec['path'] for spec in S.SEALED_LISTS)]
    done = subprocess.run([sys.executable, '-B', '-c', GUARD], cwd=real.snap, env=child_env(real.snap), capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-2000:]+done.stderr[-4000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out['repo'] == str(real.snap.resolve()) and [r['name'] for r in out['record']] == ['holdout_B_v1', 'H1', 'H4', 'H3', 'H2', 'root']
    assert [r['subjects'] for r in out['record'][:5]] == [100, 100, 40, 18, 4] and [r['sha256'] for r in out['record'][:4]] == [spec['sha256'] for spec in S.SEALED_LISTS]
    root = out['record'][-1]
    assert root['path'] == str(real.src.resolve()) and root['read_via'] == S.VIA_SNAPSHOT and root['code_tree'] == str(real.snap.resolve()) and root['snapshot_commit'] == real.head


def test_check_inputs_from_the_real_snapshot_passes(real):
    """smoke_r2.prepare's call (`cli check-inputs --profile r2_small --require-snapshot`, cwd the snapshot): the allowlist sources against source_repo (pins.source_root_for: its data link),
    LabelGuard over the 106 training subjects with the lists read from source_repo.  Without a data link to give source_repo, skipped (the allowlist sources live under data/)."""
    if not real.data:
        pytest.skip(f'{MAIN_DATA} is not a data link: the allowlist sources cannot be verified')
    argv = [sys.executable, '-B', '-m', 'bayesreg.seg.unet.cli', 'check-inputs', '--profile', 'r2_small', '--require-snapshot']
    done = subprocess.run(argv, cwd=real.snap, env=child_env(), capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout[-3000:]+done.stderr[-4000:]


@pytest.mark.parametrize('how', ['guard', 'check-inputs'])
def test_a_list_missing_in_source_repo_is_refused_from_the_real_snapshot(real, how):
    """Rule 2, negative, real: holdout_B_subjects.txt renamed away in source_repo (not read): SealedListError, from the guard and from check-inputs alike; renamed back afterwards."""
    if how == 'check-inputs' and not real.data:
        pytest.skip(f'{MAIN_DATA} is not a data link')
    path = real.src/S.SEALED_LISTS[0]['path']
    away = path.with_name(path.name+'.away')
    path.rename(away)
    try:
        argv = [sys.executable, '-B', '-c', GUARD] if how == 'guard' else [sys.executable, '-B', '-m', 'bayesreg.seg.unet.cli', 'check-inputs', '--profile', 'r2_small', '--require-snapshot']
        done = subprocess.run(argv, cwd=real.snap, env=child_env(real.snap if how == 'guard' else None), capture_output=True, text=True, timeout=900)
    finally:
        away.rename(path)
    assert done.returncode != 0 and 'SealedListError' in done.stderr and 'sealed list holdout_B_v1: splits/bayesreg_v1/holdout_B_subjects.txt missing' in done.stderr, done.stderr[-3000:]
