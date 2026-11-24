"""LP-G revision 1 (decision_log 2026-11-21 ~2:42 AM CST; the LP-G review's important 1): an LPBA driver of the external arms opens, after its spy is installed, the archive/ files of
xb_core.check_frozen and check_environment (BL2's drive.py and protocol.md, the smoke_cmd.md files, the method scripts, the environment records), while its policy lists the code tree's
archive/ as a forbidden root (xb_drive.policies, LP-C2 integration item 7).  The fix names exactly those files, read only (xb_core.archive_reads; xb_policy rule 1a, read_exact; in an LPBA
mode allowed_exact stays empty, so the inputs named one by one are read only too).

The world: LP-B's LPBA40 world (tests/test_bayesreg/hb/_lpba_world.py) with the sealed lists at their production paths (production_lists), and the code tree is that synthetic tree, as in
production (xb_core.REPO = sealed.REPO = the checkout): the archive/ files at their production relative paths (xb_core's constants, the frozen configuration's lists), synthetic contents
(the method scripts: copies of this checkout's experiments/xb/harness), their pins moved to them; the baselines and the conda environments synthetic files in tmp_path.  The real
check_frozen and check_environment run under the LPBA driver's DriverSpy (conda, pip and git stood in by a runner).  No real archive file, no real list, no conda, no unit."""
import copy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ..hb import _lpba_world as L
from . import _world as W

hc, xc, xb_drive, xb_spy = W.hc, W.xc, W.xb_drive, W.xb_spy
xb_policy = xb_spy.xb_policy
FORMAL, FEAS, PAIR, FEAS_PAIR = L.FORMAL, L.FEAS, L.PAIR, L.FEAS_PAIR
MODES = [(FORMAL, PAIR), (FEAS, FEAS_PAIR)]


def put(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode('utf-8'))
    return xc.sha256(path)


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = L.make(tmp_path, monkeypatch)
    w.prod = L.production_lists(w, monkeypatch)
    return w


@pytest.fixture
def bl2(world, tmp_path, monkeypatch):
    """The BL2 frozen files of check_frozen and check_environment in the synthetic code tree (module docstring).  Returns a namespace (run: the conda/pip/git runner, config)."""
    repo = world.repo
    real = json.loads((xc.REPO/xc.FROZEN_CONFIG_REL).read_text(encoding='utf-8'))       # this checkout's frozen configuration: its lists and arms (a code-tree file; no archive file read)
    config = copy.deepcopy(real)
    drive = put(repo/xc.ARCHIVE_HARNESS/'drive.py', '# synthetic: BL2VAL harness drive.py (LP-G revision 1 test)\n')
    protocol = put(repo/xc.BL2_PROTOCOL, '# synthetic: BL2 protocol.md\n')
    smoke = {record: put(repo/xc.SMOKE_CMD_REL/record/'smoke_cmd.md', f'# synthetic: smoke_cmd.md of {record}\n') for record in xc.SMOKE_CMD_SHA256}
    for name in config['harness']['copies']:
        text = (xc.HERE/'harness'/name).read_bytes()                                    # this checkout's method script (code)
        put(repo/'experiments'/'xb'/'harness'/name, text)
        config['harness']['copies'][name] = put(repo/xc.ARCHIVE_HARNESS/name, text)
    conda_text, pip_text = '# synthetic conda list\nnumpy 2.0.2 py310_0 conda-forge\n', 'numpy==2.0.2\n'
    for rel in config['env_records']['sha256']:
        name = Path(rel).name
        text = conda_text if name == 'conda_list.txt' else pip_text if name == 'pip_freeze.txt' else '{"synthetic": true}\n'
        config['env_records']['sha256'][rel] = put(repo/xc.ENV_RECORDS_REL/rel, text)
    conda, baselines = tmp_path/'conda', tmp_path/'baselines'
    for env, spec in config['environments'].items():
        spec['files'] = {rel: put(conda/'envs'/env/rel, f'synthetic {env} {rel}\n') for rel in spec.get('files') or {}}
    for group, files in config['baseline_files'].items():
        config['baseline_files'][group] = {rel: put(baselines/rel, f'synthetic {rel}\n') for rel in files}
    config_sha = put(repo/xc.FROZEN_CONFIG_REL, json.dumps(config, indent=1)+'\n')
    for name, value in dict(REPO=repo, HERE=repo/'experiments'/'xb', BASELINES=baselines, ENVS=conda/'envs', FROZEN_CONFIG_SHA256=config_sha, ARCHIVE_DRIVE_SHA256=drive,
                            BL2_PROTOCOL_SHA256=protocol, SMOKE_CMD_SHA256=smoke).items():
        monkeypatch.setattr(xc, name, value)
    monkeypatch.setattr(xb_spy, 'BASELINES', baselines)                                 # the driver's read roots (xb_spy.driver_policy reads them at call time)
    monkeypatch.setattr(xb_spy, 'CONDA', conda)
    commits = {str(baselines/c['dir']): c['commit'] for c in config['clones'].values()}

    def run(argv, **kwargs):
        argv = [str(a) for a in argv]
        if argv[1:3] == ['list', '-p']:
            return subprocess.CompletedProcess(argv, 0, conda_text, '')
        if argv[-2:] == ['pip', 'freeze']:
            return subprocess.CompletedProcess(argv, 0, pip_text, '')
        if 'rev-parse' in argv:
            return subprocess.CompletedProcess(argv, 0, commits[argv[argv.index('-C')+1]]+'\n', '')
        if 'status' in argv:
            return subprocess.CompletedProcess(argv, 0, '', '')
        raise AssertionError(argv)
    return SimpleNamespace(run=run, config=config, repo=repo)



def lpba_policy(arm, pair_set, pair):
    """The LPBA driver's policy as xb_drive.run_cell makes it (the sealed lists named one by one, xb_drive.py:302-303)."""
    paths, original = xc.cell_paths(arm, pair_set, pair), xc.pair_path(pair_set, pair)
    return xb_drive.policies(arm, pair_set, pair, paths, original, extra_exact=hc.sealed_list_paths()), original


def checks_under_spy(arm, pair_set, pair, run):
    """The real check_frozen and check_environment (xb_drive._run's steps 4-7) under the LPBA driver's DriverSpy.  Returns (spy, error or None)."""
    doc, original = lpba_policy(arm, pair_set, pair)
    sealed = hc.sealed_list_paths()
    spy = xb_spy.DriverSpy(doc, original=original, counted=sealed).install()
    error = None
    try:
        frozen = xc.check_frozen(arm)
        config = xc.frozen_config()[0]
        environment = xc.check_environment(arm, config=config, run=run)
        assert frozen['passed'] and environment['passed']
    except xb_spy.Breach as exc:
        error = exc
    finally:
        spy.remove()
    return spy, error


def archive_of(world):
    return os.path.realpath(world.repo/'archive')+'/'


@pytest.mark.parametrize('pair_set, pair', MODES)
@pytest.mark.parametrize('arm', xc.LPBA_ARMS)
def test_lpg_r1_the_real_frozen_and_environment_checks_pass_the_lpba_drivers_spy(world, bl2, arm, pair_set, pair):
    spy, error = checks_under_spy(arm, pair_set, pair, bl2.run)
    assert error is None and spy.breaches == [], (error, spy.breaches)
    read = {e['path'] for e in spy.events if not e['write']}
    want = {os.path.realpath(p) for p in xc.archive_reads(arm)}
    assert all(p.startswith(archive_of(world)) for p in want)
    assert {p for p in read if p.startswith(archive_of(world))} == want                # every listed file was read, and nothing else of archive/
    assert all(e['cls'] == 'allowed' for e in spy.events)


@pytest.mark.parametrize('arm', xc.LPBA_ARMS)
def test_lpg_r1_without_the_fix_the_driver_stops_at_bl2s_drive_py(world, bl2, monkeypatch, arm):
    """The stop the review predicted (convexadam_v4seg's first cell; every LPBA arm alike), reproduced: archive_reads taken out of the LPBA policy."""
    monkeypatch.setattr(xc, 'archive_reads', lambda arm, **kwargs: [])
    spy, error = checks_under_spy(arm, FEAS, FEAS_PAIR, bl2.run)
    assert isinstance(error, xb_spy.Breach) and spy.breaches[0]['path'] == os.path.realpath(world.repo/xc.ARCHIVE_HARNESS/'drive.py')
    assert spy.breaches[0]['cls'] == 'forbidden' and 'archive' in spy.breaches[0]['why']


@pytest.mark.parametrize('pair_set, pair', MODES)
def test_lpg_r1_the_lpba_drivers_named_files_are_read_only(world, bl2, pair_set, pair):
    """Rule 1a: in an LPBA mode every file named one by one is read only -- the archive/ files of the checks, the sealed lists, the pair's manifest -- and allowed_exact is empty; a write to
    any of them, the rest of plan/ and the rest of archive/ are refused."""
    for arm in xc.LPBA_ARMS:
        doc, original = lpba_policy(arm, pair_set, pair)
        policy = xb_policy.Policy(doc)
        assert doc['allowed_exact'] == [] and os.path.realpath(original.parent/'manifest.json') in doc['read_exact']
        named = [os.path.realpath(p) for p in [*xc.archive_reads(arm), *hc.sealed_list_paths(), original.parent/'manifest.json']]
        assert set(named) <= set(doc['read_exact'])
        for path in named:
            assert policy.classify(path) == ('allowed', 'named file (read only)'), (arm, path)
            assert policy.classify(path, write=True)[0] != 'allowed', (arm, path)
        others = [world.prod.other, world.prod.stray, world.repo/L.PLAN_REL/'x.py', world.repo/xc.SMOKE_CMD_REL/'niftyreg'/'smoke_cmd.md', world.repo/xc.ARCHIVE_HARNESS/'other.py']
        for path in others:
            assert policy.classify(os.path.realpath(path))[0] == 'forbidden', (arm, path)


def test_lpg_r1_the_hold_outs_and_the_developments_driver_policies_are_unchanged(world, bl2, monkeypatch):
    """No read_exact key outside an LPBA mode (xb_policy rule 1a absent: the order 1-6 as before); the inputs stay in allowed_exact; archive/ is not a forbidden root there."""
    holdout_pair = 'OASIS_OAS1_0960_MR1__OASIS_OAS1_0961_MR1'
    for pair_set, pair in (('dev', W.DEV_PAIR), ('holdout', holdout_pair)):
        paths, original = xc.cell_paths('ants_s', pair_set, pair), xc.pair_path(pair_set, pair)
        doc = xb_drive.policies('ants_s', pair_set, pair, paths, original, extra_exact=hc.sealed_list_paths())
        assert 'read_exact' not in doc and os.path.realpath(original.parent/'manifest.json') in doc['allowed_exact']
        assert not [r for r in doc['forbidden_roots'] if Path(r).name == 'archive']
        policy = xb_policy.Policy(doc)
        assert policy.read_exact == () and 'read_exact' not in policy.document
        assert policy.classify(os.path.realpath(world.repo/xc.ARCHIVE_HARNESS/'drive.py')) == ('allowed', 'a read of the running code tree')


def test_lpg_r1_archive_reads_follow_the_constants_and_the_configuration(world, bl2, monkeypatch):
    """archive_reads: built from xb_core's constants and the frozen configuration as check_frozen and check_environment build them; a configuration that does not hold gives the
    constant files only (check_frozen then refuses before it opens any other)."""
    config = xc.frozen_config()[0]
    for arm in xc.ARMS:
        got = [str(p) for p in xc.archive_reads(arm)]
        assert got[:2] == [str(world.repo/xc.ARCHIVE_HARNESS/'drive.py'), str(world.repo/xc.BL2_PROTOCOL)] and len(got) == len(set(got))
        assert {str(world.repo/xc.ENV_RECORDS_REL/rel) for rel in config['env_records']['sha256']} <= set(got)
        assert {str(world.repo/xc.ARCHIVE_HARNESS/name) for name in config['harness']['copies']} <= set(got)
    monkeypatch.setattr(xc, 'FROZEN_CONFIG_SHA256', '0'*64)
    assert [str(p) for p in xc.archive_reads('ants_s')] == [str(world.repo/xc.ARCHIVE_HARNESS/'drive.py'), str(world.repo/xc.BL2_PROTOCOL),
                                                             str(world.repo/xc.SMOKE_CMD_REL/'ants'/'smoke_cmd.md')]
    with pytest.raises(hc.Refused, match='frozen'):
        xc.check_frozen('ants_s')
