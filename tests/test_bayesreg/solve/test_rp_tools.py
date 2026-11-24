"""RP (runs/20261105-rp/任务书.md, plan.md): experiments/fl/{rp_core,decide_rp,run_rp}.py.  CPU only, synthetic small volumes; nothing here starts a unit (a fake machine; the sandbox's systemd-run fails on top of that).  Real files are read only where a test says so (the pins against d0_v1's JSON records, the bytes of RW's commit).

- the repetitions are RW's batch under other roots and prefixes, in RW's order; their instances of run_fl carry RW's fuse, RW's source, RW's constants; the start strings are RW's but for the names;
- the identity of the code against 85fc80d, on a synthetic git repository, each kind of difference;
- the pins of configuration C's fields; their qualification by d0_v1's records only, each reason it fails; the identity as the second tier;
- the two readings under both conventions of the fuse (strict, code), every branch and every boundary, the method's means by 2c(i), the controls;
- the queue on a fake machine; the dry run in its own process;
- one synthetic repair of each repetition and the identity through the real path, in a subprocess.
"""
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from .test_ef_tools import pipeline_world

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT/'experiments/fl/run_rp.py'
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
REAL_RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
needs_frozen = pytest.mark.skipif(not (FROZEN/'jd_curl/three_d/preopt.py').is_file(), reason='the frozen worktree is not on this machine')


def _load():
    spec = importlib.util.spec_from_file_location('_rp_run', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules['_rp_run'] = module
    spec.loader.exec_module(module)
    return module


RUN = _load()
fxrun, rp, decision, fx, core, fl, MC0 = RUN.fxrun, RUN.rp, RUN.decision, RUN.fx, RUN.core, RUN.fl, RUN.run
RW, RP, RP2, RP3, BATCHES, ALL = fx.RW, rp.RP, rp.RP2, rp.RP3, rp.BATCHES, fx.ALL_PAIRS
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{FROZEN}:{ROOT}')
P0, R0 = core.PAIRS[0], fl.REST_PAIRS[0]
FAILED = fx.pair_of(451)
# rw_v1/runs/RHx24_c600/0439-0440/repair.json "config", copied when this file was written: RW's RF05 with the fuse
RW_CONFIG = {"eps": 0.05, "rho0": 100.0, "rho_factor": 10.0, "radius0": 1, "radius_step": 1, "max_rounds": 5, "max_inner": 300, "memory": 12, "g_tol": 0.001, "f_rtol": 1e-07, "stall_iters": 5, "wall_seconds": 1200.0, "tile_cells": 32768}
# rw_v1/batch.json caps, gate and evaluation
RW_CAPS = {"run_seconds": 900.0, "run_units_seconds": 18000, "device_gib": 9.0, "memory_max": "8G", "nice": 5, "repair_wall_seconds": 1200.0, "repair_runtime_seconds": 1800}
RW_GATE = {"load_limit": 1.5, "gpu_free_mib": 15360, "mem_available_gib": 7.0, "home_floor_gb": 100, "host_floor_gib": 15.0}
RW_EVALUATION = {"numba_threads": 8, "cpu_quota_percent": 800}
# rw_v1/runs/RHx24_c600/0439-0440/repair_metrics.json after.certificate.attempts[0]: the configuration and version every A0 certificate has (d0_v1, checked)
CERT_CONFIG = {"delta": 0.0, "max_depth": 48, "max_boxes": 255, "tile_size": 65536, "policy": "lowest_bound_first_v1", "prepass": "edge_hull_v1", "kernel": "numba-parallel-hoisted-edge-hull-v1", "fallback_workers": 8, "fallback_kernel": "numba-bernstein-fallback-v1", "prescreen_kernel": "numba-float-bernstein-apriori-v1"}
CERT_VERSION = "q1-3d-parallel-hoisted-edge-hull-float-bernstein-prescreen-compiled-bernstein-v1"


@pytest.fixture(autouse=True)
def no_real_machine(monkeypatch):
    """A test never starts a process through the queue's machine: a fake replaces System; the real one raises."""
    def forbidden(*args, **kwargs):
        raise AssertionError('the real System.run was called')
    monkeypatch.setattr(MC0.System, 'run', forbidden)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


# ---- the repetitions are RW's batch -------------------------------------------------------------------------------------------------------------
def test_the_repetitions_are_rws_cells_in_rws_order_under_their_own_names():
    rw = RW.plan_control()+RW.plan_wall()+RW.plan_reused()
    assert len(rw) == 19 and len(RP.plan()) == 38
    for rep, batch in BATCHES.items():
        plan = batch.plan()
        assert [(c.arm, c.tier, c.pair) for c in plan] == [(c.arm, c.tier, c.pair) for c in rw] and batch.all_cells(None) == plan
        assert plan[0].pair == fx.pair_of(439) and [c.pair for c in plan[1:6]] == list(RW.WALL_PAIRS)
        for cell in plan:
            assert cell.run_id == f'rp{rep}_RHx24_{cell.tier}_{cell.short}' and cell.unit == f'tag-rp{rep}-rhx24-{cell.tier}-{cell.short}' and cell.is_repair and cell.base == 'Hx24'
        assert batch.DEFAULT_ROOT == rp.RESULTS/f'rp_v1/r{rep}' and batch.REPAIR_ARM == 'RHx24' and batch.SOURCES_FILE == RW.SOURCES_FILE and batch.REPAIR_EPS == RW.REPAIR_EPS == .05
    assert RW.PREFIX == 'rw' and RW.DEFAULT_ROOT.name == 'rw_v1' and RW.plan_control()[0].run_id == 'rw_RHx24_c600_0439-0440'      # RW's own batch is untouched


@pytest.mark.parametrize('arm,tier,pair', (('Hx24', fl.FIRST, P0), ('RHx24', 'rest600', P0), ('RHx24', 'c600', R0), ('RHx24', 'c600', 'OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1'), ('RO_Hx24', 'c600', P0), ('RNHx24', 'c600', P0)))
def test_a_cell_outside_the_plan_is_refused_before_a_file_opens(arm, tier, pair, monkeypatch):
    monkeypatch.setattr(np, 'load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('a file was opened')))
    for batch in BATCHES.values():
        with pytest.raises(SystemExit, match=f'{batch.PREFIX.upper()} has no cell'):
            batch.require_cell(arm, tier, pair)


def test_the_roots():
    assert RP.root_problem(RP.DEFAULT_ROOT) is None and 'RP writes only to' in RP.root_problem(RP.DEFAULT_ROOT/'r2')
    assert RP.id_problem(RP.DEFAULT_ROOT/'id') is None and RP.id_problem(RP.DEFAULT_ROOT) is not None
    for rep, batch in BATCHES.items():
        cell = batch.plan()[0]
        assert batch.root_problem(RP.rep_root(RP.DEFAULT_ROOT, rep)) is None and batch.root_problem(RW.DEFAULT_ROOT) is not None
        assert batch.out_problem(batch.run_dir(batch.DEFAULT_ROOT, cell), cell) is None and batch.out_problem(RW.run_dir(RW.DEFAULT_ROOT, RW.cell_of(cell.pair)), cell) is not None
    assert RP.a0_field(P0) == REAL_RESULTS/'d0_v1/runs/A0'/P0/'pg_weak__four__90s/final_native.npz' and RP.r1_root() == REAL_RESULTS/'rw_v1'
    with pytest.raises(SystemExit) as refused:
        RUN.main(['rp2', 'run-one', '--arm', 'RHx24', '--tier', 'c600', '--pair', P0, '--out', '/x'])         # no field run in RP
    assert refused.value.code == 2


def test_the_instances_carry_rws_fuse_source_and_constants():
    rw = fxrun.RW_BASE
    for rep, base in RUN.BASES.items():
        assert base.fl is BATCHES[rep] and base is not rw and base.SCRIPT == 'experiments/fl/run_rp.py' and base.QUEUE_UNIT == 'tag-rp-queue'
        assert isinstance(base.repair, fxrun.RepairWithFuse) and base.repair._wall == fx.FUSE_SECONDS == 1200.
        assert asdict(base.repair.RepairConfig(eps=base.fl.REPAIR_EPS)) == asdict(rw.repair.RepairConfig(eps=RW.REPAIR_EPS)) == RW_CONFIG
        assert base.source_of_repair is fxrun.rw_source_of_repair is rw.source_of_repair
        assert base.repair_allowed(base.fl.plan()[3], None) is True
        for name in ('MEMORY_MAX', 'NICE', 'REPAIR_THREADS', 'REPAIR_QUOTA', 'RUN_THREADS', 'EVAL_THREADS', 'EVAL_QUOTA', 'GAP_SECONDS', 'DEFAULT_MIN_HOST_GIB', 'RUN_UNITS_CAP_SECONDS', 'TRUTH_MEMBERS', 'K_STOP'):
            assert getattr(base, name) == getattr(rw, name), name
        assert base.GATE.own_units == ('tag-rp-queue.service',) and {k: getattr(base.GATE, k) for k in ('load_limit', 'gpu_free_mib', 'mem_available_gib', 'host_floor_gib', 'gpu_hold')} == {k: getattr(rw.GATE, k) for k in ('load_limit', 'gpu_free_mib', 'mem_available_gib', 'host_floor_gib', 'gpu_hold')}
        for name in ('run_phase', 'repair_one', 'evaluate_repair', 'evaluate_all', 'source_checked', 'ledger'):
            mine, theirs = getattr(base, name).__code__, getattr(rw, name).__code__
            assert mine.co_code == theirs.co_code and mine.co_filename == theirs.co_filename, name
    assert rw.fl is RW and rw.repair_allowed is fxrun.rw_repair_allowed                  # RW's instance stays RW's


def swap(argv, pairs):
    text = ' '.join(argv)
    for a, b in pairs:
        text = text.replace(a, b)
    return text


def test_the_start_strings_are_rws_but_for_the_names():
    code, rw_root = Path('/r/code'), Path('/r/rw_v1')
    for rep, base in RUN.BASES.items():
        where = Path(f'/r/rp_v1/r{rep}')
        for cell in BATCHES[rep].plan():
            mine = base.run_argv(cell, code, where, 'abc')
            theirs = fxrun.RW_BASE.run_argv(RW.cell_of(cell.pair), code, rw_root, 'abc')
            assert swap(mine, ((f'tag-rp{rep}-', 'tag-rw-'), ('run_rp.py rp' + str(rep), 'run_fx.py rw'), (str(where), str(rw_root)))) == ' '.join(theirs)
            assert '--property=RuntimeMaxSec=1800' in mine and '--property=MemoryMax=8G' in mine and '--nice=5' in mine and '--setenv=NUMBA_NUM_THREADS=8' in mine and '--setenv=CUDA_VISIBLE_DEVICES=' not in mine
        ev, rw_ev = base.eval_argv(code, where, base.EVAL_UNIT), fxrun.RW_BASE.eval_argv(code, rw_root, fxrun.RW_BASE.EVAL_UNIT)
        assert swap(ev, ((f'tag-rp{rep}-', 'tag-rw-'), ('run_rp.py rp' + str(rep), 'run_fx.py rw'), (str(where), str(rw_root)))) == ' '.join(rw_ev) and '--setenv=CUDA_VISIBLE_DEVICES=' in ev
    a0 = RUN.id_eval_argv(code, Path('/r/rp_v1'))
    assert '--unit=tag-rp-eval-id' in a0 and '--setenv=CUDA_VISIBLE_DEVICES=' in a0 and a0[-4:] == ['id', 'evaluate', '--root', '/r/rp_v1/id'] and '--setenv=NUMBA_NUM_THREADS=8' in a0
    queue = RUN.queue_launch()
    assert '--unit=tag-rp-queue' in queue and queue[queue.index('--root')+1] == str(RP.DEFAULT_ROOT) and 'experiments/fl/run_rp.py' in queue
    assert swap(queue, (('tag-rp-queue', 'tag-rw-queue'), ('run_rp.py', 'run_fx.py rw'), (str(RP.DEFAULT_ROOT), str(RW.DEFAULT_ROOT)))) == ' '.join(fxrun.queue_launch('rw'))
    for argv in [queue, a0]+[RUN.BASES[2].run_argv(RP2.plan()[0], code, '/r', 'x')]:
        assert 'py'+'test' not in ' '.join(argv) and not any('seq-vp' in a or 'tag-rw' in a or 'tag-fl' in a for a in argv)


TRUTH_WORDS = ('fixed_seg35', 'moving_seg35', 'fixed_seg4', 'moving_seg4', 'read_pair', 'TRUTH_CHANNEL_MEMBERS')


def test_the_new_files_name_no_truth_member_and_read_truth_only_in_the_evaluation_of_the_identity():
    for rel in ('experiments/fl/rp_core.py', 'experiments/fl/decide_rp.py', 'experiments/fl/run_rp.py'):
        text = (ROOT/rel).read_text()
        assert not any(w in text for w in TRUTH_WORDS), rel
        assert 'systemctl' not in text and 'kill' not in text and 'pkill' not in text
    text = (ROOT/'experiments/fl/run_rp.py').read_text()
    assert text.count('read_truth(') == 1 and text.index('read_truth(') > text.index('def identity_evaluate(') and text.index('read_truth(') < text.index('def r1_input_problems(')
    for rel in ('experiments/fl/rp_core.py', 'experiments/fl/decide_rp.py', 'experiments/fl/run_rp.py'):
        assert 'np.load' not in (ROOT/rel).read_text() and 'certify_ladder' not in (ROOT/rel).read_text().split('def evaluate_identity(')[0], rel       # no field's values are read, and no certificate but the identity's is computed


# Exception (14) of contract C23.9-13 (the lead's ruling, ~6:06 PM CST 11-09, decision_log; option (a) of the SG follow-up report): A RELAXATION of the pin below for
# experiments/d0/evaluate.py only.  Contract C23 revision 6h (SG code review I2) adds to that file exactly these three blocks of text -- stage_pair's definition and
# its two calls at the head of truth_labels and evaluate_run -- written out here verbatim; the file with each block removed once (each must occur exactly once)
# must be 85fc80d's byte for byte.  The other eleven files stay pinned byte for byte.
# Contract C23 revision 6i (SG follow-up review n2, the lead's instruction of 11-09 ~9:29 PM CST) adds two more blocks to the same file, written out the same way:
# run_copy's definition and its call right before the run directory's copy is read; exception (14) covers them as it covers the three of 6h (still a relaxation).
EVALUATE_6H_BLOCKS = (r'''def stage_pair(pair):
    """Contract C23.6-2 revision 6h (the F rows, SG code review I2): the pair named by the caller (--pair is free text) must be a plain name -- no '/', no
    '\\', no '..' -- registered in STAGE/selection.json, and STAGE/<pair> must not be a link nor lie outside STAGE; otherwise SystemExit before anything else
    is read.  So `--pair ../stage_H1_v1/<pair>` cannot reach another stage's labels.  Reads STAGE/selection.json only (pair names, no subject file)."""
    if not isinstance(pair, str) or not pair or any(c in pair for c in ('/', '\\', '\x00')) or '..' in pair or pair == '.':
        raise SystemExit(f'{pair!r}: not a plain pair name; nothing is read (C23.6-2 revision 6h)')
    try:
        rows = json.loads((STAGE/'selection.json').read_text())['pairs']
        names = [r['pair'] if isinstance(r, dict) else r for r in rows]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f'{STAGE}/selection.json cannot be read ({type(exc).__name__}); nothing is read (C23.6-2 revision 6h)') from None
    if pair not in names:
        raise SystemExit(f'{pair}: not a pair of {STAGE}/selection.json; nothing is read (C23.6-2 revision 6h)')
    d = STAGE/pair
    if d.is_symlink() or d.resolve().parent != STAGE.resolve():
        raise SystemExit(f'{d}: a link, or outside {STAGE}; nothing is read (C23.6-2 revision 6h)')
    return pair


''',
r'''    stage_pair(pair)                                                     # C23.6-2 revision 6h: before anything is read
''',
r'''    stage_pair(pair)                                                     # C23.6-2 revision 6h: before anything is read (the run directory's copy too)
''',
r'''def run_copy(run, pair):
    """Contract C23 revision 6i (SG follow-up review n2): the run directory's copy of the pair, <run>/inputs/<pair>/pair.npz, is read only when neither it nor
    <run>/inputs nor <run>/inputs/<pair> is a link and it lies in the run directory; SystemExit otherwise, before it is read.  Returns its path."""
    run = Path(run)
    copy = run/'inputs'/pair/'pair.npz'
    if any(p.is_symlink() for p in (run/'inputs', run/'inputs'/pair, copy)) or not copy.resolve().is_relative_to(run.resolve()):
        raise SystemExit(f'{copy}: a link, or outside the run directory {run}; not read (C23 revision 6i, review n2)')
    return copy


''',
r'''    run_copy(run, pair)                                                  # C23 revision 6i (review n2): the run's copy, not a link, before it is read
''')


def without_the_6h_blocks(data):
    """evaluate.py's bytes with the blocks of revisions 6h and 6i removed; AssertionError unless each occurs exactly once."""
    text = data.decode('utf-8')
    for block in EVALUATE_6H_BLOCKS:
        assert text.count(block) == 1, block[:80]
        text = text.replace(block, '')
    return text.encode('utf-8')


def test_the_files_of_rws_path_in_this_tree_are_byte_for_byte_rws_commit():
    """The tree this test runs in, against 85fc80d: the files the repetitions load (a subprocess lists them in the next test).  experiments/d0/evaluate.py: equal
    after the blocks of revisions 6h and 6i are removed (exception (14), a relaxation; above)."""
    done = subprocess.run(['git', 'cat-file', '-e', f'{rp.BASE_COMMIT}^{{commit}}'], cwd=ROOT, capture_output=True)
    if done.returncode:
        pytest.skip('RW\'s commit is not in this repository')
    for rel in ('experiments/fl/run_fx.py', 'experiments/fl/fx_core.py', 'experiments/fl/decide_fx.py', 'experiments/fl/run_fl.py', 'experiments/fl/decide_fl.py', 'experiments/fl/fl_core.py', 'experiments/fl/rw_sources_v1.json', 'experiments/ef/repair.py', 'experiments/ef/run_ef.py', 'experiments/mc0/run_mc0.py', 'experiments/mc0/mc0_core.py', 'experiments/d0/evaluate.py'):
        theirs = subprocess.run(['git', 'show', f'{rp.BASE_COMMIT}:{rel}'], cwd=ROOT, capture_output=True).stdout
        ours = (ROOT/rel).read_bytes()
        assert theirs == (without_the_6h_blocks(ours) if rel == 'experiments/d0/evaluate.py' else ours), rel


def test_the_6h_exception_removes_only_its_blocks():
    """Exception (14): the narrowing cannot hide another change -- a byte changed outside the blocks, or a block missing, changed or doubled, fails."""
    theirs = subprocess.run(['git', 'show', f'{rp.BASE_COMMIT}:experiments/d0/evaluate.py'], cwd=ROOT, capture_output=True).stdout
    if not theirs:
        pytest.skip('RW\'s commit is not in this repository')
    ours = (ROOT/'experiments/d0/evaluate.py').read_bytes()
    assert without_the_6h_blocks(ours) == theirs and ours != theirs
    assert without_the_6h_blocks(ours.replace(b'def rd(p):', b'def rd(q):', 1)) != theirs                          # a change elsewhere is seen
    for block in EVALUATE_6H_BLOCKS:
        encoded = block.encode('utf-8')
        for broken in (ours.replace(encoded, b'', 1), ours.replace(encoded, encoded[:-2]+b'#\n', 1), ours.replace(encoded, encoded+encoded, 1)):
            with pytest.raises(AssertionError):
                without_the_6h_blocks(broken)


def test_the_6h_calls_are_the_first_lines_of_their_two_functions():
    """SG follow-up diff review n4 (A5b): the exception's blocks are pinned in place, not only in number -- the 2nd and 3rd blocks (the stage_pair calls) are the very next line
    after `def truth_labels(pair):` and after `def evaluate_run(run, arm, pair, run_id, method=None):`; a call moved into another function (or below a read) fails."""
    text = (ROOT/'experiments/d0/evaluate.py').read_text(encoding='utf-8')
    for head, block in (('def truth_labels(pair):\n', EVALUATE_6H_BLOCKS[1]), ('def evaluate_run(run, arm, pair, run_id, method=None):\n', EVALUATE_6H_BLOCKS[2])):
        assert text.count(head) == 1 and text.count(block) == 1 and head+block in text, head
    moved = text.replace(EVALUATE_6H_BLOCKS[1], '', 1).replace('    d = STAGE/pair\n', '    d = STAGE/pair\n'+EVALUATE_6H_BLOCKS[1], 1)
    assert 'def truth_labels(pair):\n'+EVALUATE_6H_BLOCKS[1] not in moved                          # the check sees a call moved below the first read


IMPORTS = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run_rp as run
run.core.load_module('_mc0_evaluate', 'experiments/d0/evaluate.py')
root = Path(sys.argv[2])
files = sorted(str(Path(m.__file__).resolve().relative_to(root)) for m in list(sys.modules.values()) if getattr(m, '__file__', None) and Path(m.__file__).resolve().is_relative_to(root))
print(json.dumps(dict(files=files, torch='torch' in sys.modules)))
"""


def test_every_file_the_repetitions_load_is_under_the_paths_of_the_identity_check():
    done = subprocess.run([PY, '-B', '-c', IMPORTS, str(ROOT/'experiments/fl'), str(ROOT)], env=dict(ENV, PYTHONPATH=str(ROOT)), cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out['files'] and out['torch'] is False
    for rel in out['files']:
        assert rel in rp.OWN_FILES or any(rel == p or rel.startswith(p+'/') for p in rp.IDENTITY_PATHS), rel
    assert {'experiments/fl/run_fx.py', 'experiments/fl/run_fl.py', 'experiments/ef/repair.py', 'experiments/d0/evaluate.py', 'experiments/fl/rp_core.py'} <= set(out['files'])


# ---- the identity of the code, on a synthetic repository ------------------------------------------------------------------------------------------
def git(repo, *args):
    done = subprocess.run(['git', *args], cwd=repo, capture_output=True, text=True, env=dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@t'))
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A repository with RW's kind of files at its base commit, and a frozen tree: a copy of that commit's jd_curl and root .py, as frozen_eb808be is of eb808be."""
    r = tmp_path/'repo'
    for rel in ('experiments/ef/repair.py', 'experiments/fl/run_fx.py', 'bayesreg/x.py', 'configs/bayesreg/a.json', 'docs/n.md', 'jd_curl/three_d/feasibility.py', 'jd_curl/__init__.py', 'setup_hint.py'):
        (r/rel).parent.mkdir(parents=True, exist_ok=True)
        (r/rel).write_text(rel)
    git(r, 'init', '-q')
    git(r, 'add', '-A')
    git(r, 'commit', '-qm', 'base')
    frozen = tmp_path/'frozen'
    for rel in ('jd_curl/three_d/feasibility.py', 'jd_curl/__init__.py', 'setup_hint.py', 'docs/n.md'):
        (frozen/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(r/rel, frozen/rel)
    return r, git(r, 'rev-parse', 'HEAD'), frozen


def commit_own(r):
    for rel in rp.OWN_FILES:
        (r/rel).parent.mkdir(parents=True, exist_ok=True)
        (r/rel).write_text(rel)


def identity_of(r, base, frozen, sha=None):
    return rp.identity(sha or git(r, 'rev-parse', 'HEAD'), base=base, cwd=r, frozen=frozen, frozen_commit=base)


def test_identity_holds_with_only_the_own_files_added(repo):
    r, base, frozen = repo
    commit_own(r)
    (r/'configs/bayesreg/new_registry.json').write_text('{}')
    (r/'configs/bayesreg/登记.json').write_text('{}')                   # a name that git would quote without -z
    (r/'docs/n.md').write_text('changed')                              # not under the paths
    git(r, 'add', '-A')
    git(r, 'commit', '-qm', 'rp')
    record, problems = identity_of(r, base, frozen)
    assert problems == [] and record['same'] is True and sorted(record['added_own']) == sorted(rp.OWN_FILES) and sorted(record['added_other']) == ['configs/bayesreg/new_registry.json', 'configs/bayesreg/登记.json'] and record['n_files_changed'] == 0 and record['n_files_at_base'] == 6
    assert record['frozen']['same'] is True and record['frozen']['n_files'] == 3


@pytest.mark.parametrize('change,word', (('modify', 'M experiments/ef/repair.py'), ('delete', 'D experiments/fl/run_fx.py'), ('add_py', 'A experiments/fl/repair.py: a Python file added'), ('add_bayesreg', 'A bayesreg/y.py: a Python file added'), ('add_scripts', 'A scripts/evaluate_l2r_style2.py: a Python file added'),
                                         ('add_root', 'A sitecustomize.py: a Python file added'), ('modify_root', 'M setup_hint.py'), ('add_unicode', 'A experiments/ef/修复.py: a Python file added'), ('no_own', 'does not hold')))
def test_identity_refuses_each_kind_of_difference(repo, change, word):
    r, base, frozen = repo
    if change != 'no_own':
        commit_own(r)
    if change == 'modify':
        (r/'experiments/ef/repair.py').write_text('changed')
    elif change == 'delete':
        (r/'experiments/fl/run_fx.py').unlink()
    elif change == 'add_py':
        (r/'experiments/fl/repair.py').write_text('shadow')
    elif change == 'add_bayesreg':
        (r/'bayesreg/y.py').write_text('new')
    elif change == 'add_scripts':
        (r/'scripts').mkdir()
        (r/'scripts/evaluate_l2r_style2.py').write_text('x')
    elif change == 'add_root':
        (r/'sitecustomize.py').write_text('x')
    elif change == 'modify_root':
        (r/'setup_hint.py').write_text('changed')
    elif change == 'add_unicode':
        (r/'experiments/ef/修复.py').write_text('x')
    else:
        (r/'scripts').mkdir()
        (r/'scripts/z.txt').write_text('x')
    git(r, 'add', '-A')
    git(r, 'commit', '-qm', 'x')
    record, problems = identity_of(r, base, frozen)
    assert record['same'] is False and any(word in p for p in problems), problems


def test_identity_of_a_commit_that_does_not_exist_is_a_problem(repo):
    r, base, frozen = repo
    record, problems = identity_of(r, base, frozen, sha='f'*40)
    assert problems and 'git diff' in problems[0]


@pytest.mark.parametrize('change,word', (('modify', 'not the same blob'), ('missing', 'missing'), ('extra', 'untracked .py'), ('extra_root', 'untracked .py'), ('pycache', None), ('outside', None)))
def test_the_frozen_tree_must_be_its_commits_blobs(repo, change, word):
    r, base, frozen = repo
    commit_own(r)
    git(r, 'add', '-A')
    git(r, 'commit', '-qm', 'rp')
    if change == 'modify':
        (frozen/'jd_curl/three_d/feasibility.py').write_text('changed')
    elif change == 'missing':
        (frozen/'jd_curl/__init__.py').unlink()
    elif change == 'extra':
        (frozen/'jd_curl/three_d/shadow.py').write_text('x')
    elif change == 'extra_root':
        (frozen/'usercustomize.py').write_text('x')
    elif change == 'pycache':
        (frozen/'jd_curl/__pycache__').mkdir()
        (frozen/'jd_curl/__pycache__/x.py').write_text('x')          # under __pycache__: not importable as a module
    else:
        (frozen/'docs/n.md').write_text('changed')                      # not under the paths the units import from
    record, problems = identity_of(r, base, frozen)
    if word is None:
        assert problems == [] and record['frozen']['same'] is True
    else:
        assert record['same'] is False and record['frozen']['same'] is False and any('the frozen tree' in p and word in p for p in problems), problems


@pytest.mark.skipif(not (FROZEN/'jd_curl').is_dir(), reason='the frozen worktree is not on this machine')
def test_the_real_frozen_tree_is_eb808bes():
    done = subprocess.run(['git', 'cat-file', '-e', f'{core.FROZEN_COMMIT}^{{commit}}'], cwd=ROOT, capture_output=True)
    if done.returncode:
        pytest.skip('eb808be is not in this repository')
    record, problems = rp.frozen_identity()
    assert problems == [] and record['same'] is True and record['n_files'] >= 125 and record['commit'] == core.FROZEN_COMMIT


FROZEN_IMPORTS = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run_rp as run
frozen = Path(sys.argv[2]).resolve()
run.core.frozen_api()
from jd_curl.three_d.feasibility import certify_map
from jd_curl.three_d import jit_cache
from jd_curl.three_d.certificate_prescreen import prescreen_lower_bounds
from jd_curl.three_d.compiled_enclosure import parallel_margins
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.cell_q1 import CellQ1Operators3D
run.run.certificate_config()
run.core.load_module('_mc0_evaluate', 'experiments/d0/evaluate.py')
files = sorted(str(Path(m.__file__).resolve().relative_to(frozen)) for m in list(sys.modules.values()) if getattr(m, '__file__', None) and Path(m.__file__).resolve().is_relative_to(frozen))
print(json.dumps(dict(files=files, torch='torch' in sys.modules)))
"""


@needs_frozen
def test_what_the_units_import_from_the_frozen_tree_is_under_the_paths_of_its_check():
    """With the units' PYTHONPATH (frozen first): frozen_api, the certificate, the JIT cache, what repair.py imports inside its functions, the evaluation's module.  Module-level and these function-level imports only (a limit, plan.md)."""
    done = subprocess.run([PY, '-B', '-c', FROZEN_IMPORTS, str(ROOT/'experiments/fl'), str(FROZEN)], env=ENV, cwd=tempfile.gettempdir(), capture_output=True, text=True)          # not the repository: -c puts the working directory first on sys.path
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out['files'] and out['torch'] is False
    assert all(any(f.startswith(x+'/') for x in rp.FROZEN_PATHS) or '/' not in f for f in out['files']), out['files']
    assert 'jd_curl/three_d/feasibility.py' in out['files'] and 'jd_curl/three_d/preopt.py' in out['files']


# ---- worlds on disk ----------------------------------------------------------------------------------------------------------------------------------
def metrics_of(seg35=.87, sdlogj=.07):
    return dict(seg35=seg35, seg4=seg35+.04, hd95=1.4, sdlogj_official=sdlogj, sdlogj_brain=.55, small_six=.66, large_six=.9, other_23=.9, cortex=.86, white_matter=.93)


def rounds_of(n, wall=False):
    return [dict(round=k, n_free=9000, n_extra_points=0, inner=dict(stop='wall' if wall and k == n-1 else 'inner_cap', iterations=300)) for k in range(n)]


def write_repair(d, run_id, *, seg35=.87, sdlogj=.07, certified=True, wall=False, rounds=1, seconds=60., code_sha='fakesha', config=None, tapered=None, evaluate=True, bound=1e-6, field='f', oracle=False):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d/'repair.npz', nodes=np.arange(3), values=np.zeros((3, 3)))
    short = run_id.split('_')[-1]
    write_json(d/'repair.json', dict(schema='bayesreg.fl.repair.v1', run_id=run_id, status=fx.CERTIFIED if certified else fx.NOT_CERTIFIED, passed=certified, oracle=oracle, reason='certified' if certified else ('wall' if wall else 'round_cap'), eps=.05, config=dict(RW_CONFIG if config is None else config), source=dict(seconds_task=45., tapered_u_sha256=f't_{short}' if tapered is None else tapered),
                                     rounds=rounds_of(rounds, wall), seconds_repair=seconds, seconds_warmup=1.2, seconds_total=seconds+5., attempt=dict(file='repair.npz', sha256=core.sha_file(d/'repair.npz')), code_sha=code_sha, fallback_rule_run=False,
                                     certificate_config=dict(CERT_CONFIG), frozen_commit=core.FROZEN_COMMIT, data=dict(pair_sha256=f'p_{short}', manifest_sha256=f'm_{short}', keys_read=['fixed', 'moving', 'initial_u']), env=dict(python='3.10.20', torch='2.5.1', device='GPU'), device_cap=dict(limit_gib=9.)))
    write_json(d/'queue_done.json', dict(run_id=run_id, rc=0, unit_seconds=seconds+4., state=fx.CERTIFIED if certified else fx.NOT_CERTIFIED, arm_stop=None))
    if evaluate:
        evaluate_repair(d, seg35=seg35, sdlogj=sdlogj, certified=certified, bound=bound, field=field)


def evaluate_repair(d, *, seg35=.87, sdlogj=.07, certified=True, bound=1e-6, field='f'):
    write_json(Path(d)/'repair_metrics.json', dict(before=dict(metrics=metrics_of(seg35+.0005, sdlogj), n_infeasible_at_1=20000), after=dict(metrics=metrics_of(seg35, sdlogj), certificate=dict(accepted_alpha=1. if certified else None, n_infeasible_at_1=0 if certified else 1, attempts=[dict(min_jd_lower_bound=bound if certified else -.26, config=dict(CERT_CONFIG), certificate_version=CERT_VERSION)])),
                                                   change=dict(n_changed_nodes=9000, max_change_mm=.3), repaired_u_sha256=f'{field}_{seg35}_{certified}', evaluator=dict(script='experiments/d0/evaluate.py', sha256='e')))


def write_rw(results, *, same=False, seg35=.874, certified=True, code_sha=None):
    """rw_v1 as R1: RW's commit, its control judged not bit for bit, the 19 repairs."""
    root = results/'rw_v1'
    write_json(root/'batch.json', dict(code_sha=rp.BASE_COMMIT if code_sha is None else code_sha, plan=[]))
    RW.write_selection(root, RW.CONTROL_FILE, dict(same=same, pair='0439-0440'))
    for pair in ALL:
        cell = RW.cell_of(pair)
        write_repair(RW.run_dir(root, cell), cell.run_id, seg35=seg35, certified=certified, code_sha=rp.BASE_COMMIT, rounds=2 if pair in RW.WALL_PAIRS else 1, field='r1')
    return root


def write_rep(results, rep, *, seg35=.874, overrides=None, code_sha='fakesha'):
    """rp_v1/r<rep>, every pair repaired; overrides {pair: keyword arguments of write_repair, or 'missing'}."""
    batch = BATCHES[rep]
    for cell in batch.plan():
        kw = dict(dict(seg35=seg35, code_sha=code_sha, field=f'r{rep}'), **(overrides or {}).get(cell.pair, {})) if (overrides or {}).get(cell.pair) != 'missing' else None
        if kw is not None:
            write_repair(batch.run_dir(RP.rep_root(results/'rp_v1', rep), cell), cell.run_id, **kw)


def u_digest(u):
    return __import__('hashlib').sha256(np.ascontiguousarray(u, dtype=np.float64).view(np.uint8)).hexdigest()


def stage_world(base):
    stage = base/'stage'
    for pair in ALL:
        (stage/pair).mkdir(parents=True, exist_ok=True)
        np.savez(stage/pair/'pair.npz', fixed=np.zeros((6, 7, 8)), moving=np.zeros((6, 7, 8)), fixed_seg35=np.zeros((6, 7, 8), dtype=np.int16))
    return stage


def write_d0(results, stage, *, seg35=.82):
    """d0_v1's A0 records and fields: small u of the image's shape."""
    for pair in ALL:
        d = RP.a0_run_dir(pair, results)
        shape = tuple(MC0.npz_header(stage/pair/'pair.npz', 'fixed')[0])
        u = np.zeros(shape+(3,))
        (d/'pg_weak__four__90s').mkdir(parents=True, exist_ok=True)
        np.savez(d/'pg_weak__four__90s/final_native.npz', u=u, warped=np.zeros(shape))
        sha, state = core.sha_file(d/'pg_weak__four__90s/final_native.npz'), u_digest(u)
        write_json(d/'pg_weak__four__90s/final_native.json', dict(path='final_native.npz', sha256=sha, certificate=dict(status='certified_feasible', min_jd_lower_bound=3e-7, n_infeasible=0, n_uncertain=0, certificate_version=CERT_VERSION, state_hash=state, config=dict(CERT_CONFIG))))
        write_json(d/'metrics.json', dict(run_id=f'd0v1_A0_{core.short_name(pair)}_x', arm='A0', pair=pair, classification=dict(**{'class': 'ok'}, kind='native'), certificate=dict(passed=True, state_hash=state, state_hash_matches_u=True, final_npz_sha256=sha, final_npz_sha256_matches=True), metrics=dict(seg35=seg35, seg4=seg35+.06, hd95=1.6, sdlogj_official=.076, sdlogj_brain=.57),
                                          evaluator=dict(script='experiments/d0/evaluate.py', sha256=rp.D0_EVALUATOR_SHA256)))


def write_identity(results, pairs, *, seg35=.6, certified=True):
    for pair in pairs:
        write_json(RP.id_dir(RP.id_root(results/'rp_v1'), pair)/'identity_metrics.json', dict(pair=pair, oracle=False, certified_at_1=certified, metrics=metrics_of(seg35, 0.)))


def write_pins(results):
    """rw_sources (the tapered pins of the synthetic repairs) under results; returns its path."""
    return write_json(results/'rw_sources.json', dict(pairs={core.short_name(p): dict(tapered_u_sha256=f't_{core.short_name(p)}', field_sha256='x') for p in ALL}))


@pytest.fixture
def results(monkeypatch):
    base_dir = tempfile.mkdtemp(prefix='rpq_')              # not the test runner's tmp_path: its name would fail the start-string check
    path = Path(base_dir)/'results'
    path.mkdir()
    monkeypatch.setattr(fx, 'RESULTS', path)
    for batch in (RW, RP2, RP3):
        monkeypatch.setattr(batch, 'RESULTS', path)
        monkeypatch.setattr(batch, 'DEFAULT_ROOT', path/batch.ROOT_NAME)
    monkeypatch.setattr(RP, 'RESULTS', path)
    monkeypatch.setattr(RP, 'DEFAULT_ROOT', path/'rp_v1')
    monkeypatch.setattr(RW, 'SOURCES_FILE', write_pins(path))
    stage = stage_world(Path(base_dir))
    monkeypatch.setattr(core, 'STAGE', stage)
    write_d0(path, stage)
    evaluate_repair(RW.run_dir(path/'rw_v1', RW.cell_of(RW.CONTROL_PAIR)))          # the pins take RW's evaluation's certificate configuration from rw_v1's control
    monkeypatch.setattr(rp, 'FALLBACK_FILE', write_json(Path(base_dir)/'a0_pins.json', RUN.fallback_pins_from_records(path)))
    yield path
    shutil.rmtree(base_dir)


def written(results, **kw):
    """A complete world: R1, R2, R3 certified, configuration C's fields qualified, batch.json at fakesha."""
    write_rw(results)
    for rep in (2, 3):
        write_rep(results, rep, **kw.get(f'r{rep}', {}))
    write_json(results/'rp_v1/batch.json', dict(code_sha='fakesha', plan=[c.run_id for c in RP.plan()]))
    return results/'rp_v1'


def spoil_a0(results, pair, how='class'):
    d = RP.a0_run_dir(pair, results)
    if how == 'class':
        m = json.loads((d/'metrics.json').read_text())
        m['classification']['class'] = 'method_failure'
        write_json(d/'metrics.json', m)
    else:
        (d/'pg_weak__four__90s/final_native.npz').unlink()


# ---- the readings ----------------------------------------------------------------------------------------------------------------------------------
def test_three_times_19_of_19(results):
    root = written(results)
    d = decision.write_rp(root)
    assert d['reading'] == '3 次都 19/19' and d['statement'] == '在这 19 对上修复稳定（3 次抽样）' and d['reason'] is None and d['n_repairs'] == 57
    assert d['reading_spread'] == '方法级均值的极差 ≤ .001' and d['method_seg35_range'] == pytest.approx(0.) and d['controls']['problems'] == [] and d['controls']['n_configs'] == 1
    assert d['code_convention']['reading'] == d['reading'] and d['code_convention']['reading_spread'] == d['reading_spread'] and d['strict']['n_identity_used'] == d['strict']['n_a0_used'] == 0
    assert all(r['n_certified'] == 19 and r['method'][h]['complete'] and r['method'][h]['substituted_a0'] == [] for r in d['per_rep'] for h in ('strict', 'code'))
    assert {p['n_distinct_repaired_fields'] for p in d['per_pair']} == {3} and d['fallback']['n_qualified'] == 19
    assert [r['rep'] for r in d['per_rep']] == ['R1', 'R2', 'R3'] and d['per_rep'][0]['root'] == str(results/'rw_v1') and set(d['r1_sources'].values()) == {'rw_v1'}
    text = (root/'analysis/summary_rp.md').read_text()
    assert 'seg35' in text and 'seg4' in text and '3 次都 19/19' in text and '3 次抽样' in text and '不是 F_B' in text and '严格口径' in text and '代码口径' in text and '代入恒等 0 次' in text
    assert len((root/'analysis/per_pair_rp.csv').read_text().strip().splitlines()) == 1+57


@pytest.mark.parametrize('delta,word', ((.0009, '方法级均值的极差 ≤ .001'), (.0011, '方法级均值的极差在 .001 与 .005 之间'), (.0049, '方法级均值的极差在 .001 与 .005 之间'), (.0051, '方法级均值的极差 > .005')))
def test_the_lines_of_the_spread(results, delta, word):
    root = written(results, r3=dict(seg35=.874+delta))
    d = decision.write_rp(root)
    assert d['reading'] == '3 次都 19/19' and d['reading_spread'] == word and d['method_seg35_range'] == pytest.approx(delta, abs=1e-12) and d['statement_spread'] == decision.SPREAD_WORDS[word]


def test_the_boundaries_are_exact():
    reps = [dict(rep=f'R{k}', method=dict(strict=dict(complete=True, seg35=v, missing=[]))) for k, v in ((1, 0.), (2, 0.), (3, decision.SPREAD_SMALL))]
    assert decision.read_spread(reps)[0] == decision.READ_SMALL and decision.read_spread(reps)[3] == .001
    reps[2]['method']['strict']['seg35'] = decision.SPREAD_LARGE
    assert decision.read_spread(reps)[0] == decision.READ_MIDDLE
    reps[2]['method']['strict']['seg35'] = .0050001
    assert decision.read_spread(reps)[0] == decision.READ_LARGE
    reps[2]['method']['strict']['seg35'] = float('nan')
    assert decision.read_spread(reps)[0] == decision.UNDECIDABLE


def test_a_failure_is_a_result_and_the_method_takes_configuration_c_there(results):
    root = written(results, r2=dict(overrides={FAILED: dict(certified=False, seg35=.95)}))
    d = decision.write_rp(root)
    assert d['reading'] == '有失败 1/57' and '0451-0452 R2（没有通过证书的状态）' in d['reason'] and '代入配置 C 的场' in d['statement'] and d['code_convention']['reading'] == '有失败 1/57'
    r2 = d['per_rep'][1]
    assert r2['n_certified'] == 18 and r2['method']['strict']['substituted_a0'] == ['0451-0452'] and r2['method']['strict']['seg35'] == pytest.approx((18*.874+.82)/19) and r2['method']['strict']['substituted_identity'] == []
    assert r2['method']['strict']['small_six'] is None and r2['method']['strict']['seg4'] == pytest.approx((18*.914+.88)/19)          # D0's records have no group means: a mean over 18 pairs is not written
    assert r2['repaired']['n_certified'] == 18 and r2['repaired']['seg35'] == pytest.approx(.874)              # the attempt's .95 enters no mean
    assert d['method_seg35_range'] == pytest.approx(.874-(18*.874+.82)/19) and d['reading_spread'] == decision.READ_MIDDLE and d['strict']['n_a0_used'] == 1 and d['strict']['n_identity_used'] == 0
    row = next(p for p in d['per_pair'] if p['pair'] == '0451-0452')
    assert row['n_certified'] == 2 and row['n_failures'] == dict(strict=1, code=1) and row['seg35']['values'] == [.874, None, .874] and row['method_seg35_strict']['values'] == [.874, .82, .874] and row['method_sources']['strict'] == ['repair', 'A0', 'repair'] and row['attempts'][0]['seg35'] == .95
    assert '未认证的尝试' in (root/'analysis/summary_rp.md').read_text()


def test_the_fuse_is_a_failure_strictly_and_a_pass_in_the_codes_convention(results):
    root = written(results, r3=dict(overrides={P0: dict(wall=True, rounds=5, seconds=1201., seg35=.88)}))
    d = decision.write_rp(root)
    assert d['reading'] == '有失败 1/57' and '0438-0439 R3（撞 1200 s 保险丝，证书过了）' in d['reason']
    assert d['code_convention']['reading'] == '3 次都 19/19' and d['code_convention']['reason'] is None
    r3 = d['per_rep'][2]
    assert r3['n_certified'] == 19 and r3['n_fuse'] == 1 and r3['method']['strict']['substituted_a0'] == ['0438-0439'] and r3['method']['code']['substituted_a0'] == []
    assert r3['method']['strict']['seg35'] == pytest.approx((18*.874+.82)/19) and r3['method']['code']['seg35'] == pytest.approx((18*.874+.88)/19)
    row = next(p for p in d['per_pair'] if p['pair'] == '0438-0439')
    assert row['n_certified'] == 3 and row['n_fuse'] == 1 and row['kinds']['R3'] == 'fuse' and row['n_failures'] == dict(strict=1, code=0) and row['method_sources'] == dict(strict=['repair', 'repair', 'A0'], code=['repair', 'repair', 'repair'])


def test_a_fuse_without_a_passing_certificate_is_a_failure_in_both(results):
    root = written(results, r3=dict(overrides={P0: dict(wall=True, rounds=5, seconds=1201., certified=False)}))
    d = decision.write_rp(root)
    assert d['reading'] == d['code_convention']['reading'] == '有失败 1/57' and '证书没过' in d['code_convention']['reason']


@pytest.mark.parametrize('how,word', (('missing', '没有运行'), ('not_evaluated', '没有评测'), ('crashed', '运行失败，没有 repair.json'), ('metrics_error', '评测出错')))
def test_a_repair_not_read_makes_both_readings_undecidable(results, how, word):
    root = written(results)
    cell = RP3.cell_of(FAILED)
    d = RP3.run_dir(RP.rep_root(root, 3), cell)
    if how == 'missing':
        shutil.rmtree(d)
    elif how == 'not_evaluated':
        (d/'repair_metrics.json').unlink()
    elif how == 'crashed':
        (d/'repair_metrics.json').unlink()
        (d/'repair.json').unlink()
        write_json(d/'queue_done.json', dict(run_id=cell.run_id, rc=1, unit_seconds=3., state='failed', arm_stop=None))
    else:
        (d/'repair_metrics.json').unlink()
        write_json(d/'metrics_error.json', dict(error='x'))
    out = decision.write_rp(root)
    for r in (out['strict'], out['code_convention']):
        assert r['reading'] == decision.UNDECIDABLE and word in r['reason'] and '不能读成失败' in r['reason'] and r['reading_spread'] == decision.UNDECIDABLE and 'R3' in r['reason_spread']


@pytest.mark.parametrize('spoil', ('class', 'missing'))
def test_a_failed_pair_whose_configuration_c_does_not_qualify_takes_the_identity(results, spoil):
    root = written(results, r2=dict(overrides={FAILED: dict(certified=False)}))
    spoil_a0(results, FAILED, spoil)
    write_identity(results, [FAILED])
    d = decision.write_rp(root)
    assert d['reading'] == '有失败 1/57' and d['strict']['n_identity_used'] == 1 and d['strict']['n_a0_used'] == 0 and d['per_rep'][1]['method']['strict']['substituted_identity'] == ['0451-0452']
    assert d['per_rep'][1]['method']['strict']['seg35'] == pytest.approx((18*.874+.6)/19) and d['fallback']['not_qualified'][0]['pair'] == '0451-0452' and ('method_failure' in d['fallback']['not_qualified'][0]['why'] or 'no final_native.npz' in d['fallback']['not_qualified'][0]['why'])
    assert '代入恒等 1 次' in (root/'analysis/summary_rp.md').read_text()


def test_a_failed_pair_with_neither_tier_leaves_the_method_undecidable(results):
    root = written(results, r2=dict(overrides={FAILED: dict(certified=False)}))
    spoil_a0(results, FAILED)
    d = decision.write_rp(root)
    assert d['reading'] == '有失败 1/57' and d['reading_spread'] == decision.UNDECIDABLE and '恒等场也没有' in d['reason_spread']


@pytest.mark.parametrize('spoil', ('class', 'missing'))
def test_a_configuration_c_that_does_not_qualify_where_nothing_failed_changes_no_reading(results, spoil):
    root = written(results)
    spoil_a0(results, P0, spoil)
    d = decision.write_rp(root)
    assert d['reading'] == '3 次都 19/19' and d['reading_spread'] == decision.READ_SMALL and d['fallback']['n_qualified'] == 18 and d['strict']['n_identity_used'] == 0


@pytest.mark.parametrize('what,word', (('config', 'do not share one configuration'), ('wall', 'wall_seconds 120.0'), ('code', 'not fakesha'), ('source', 'not the pinned one'), ('r1_code', f'not {rp.BASE_COMMIT}')))
def test_the_controls_make_both_readings_undecidable(results, what, word):
    root = written(results)
    if what in ('config', 'wall'):
        changed = dict(RW_CONFIG, max_inner=301) if what == 'config' else dict(RW_CONFIG, wall_seconds=120.)
        for rep in ((2,) if what == 'config' else (1, 2, 3)):
            for cell in (BATCHES[rep].plan() if rep != 1 else [RW.cell_of(p) for p in ALL]):
                d = (BATCHES[rep].run_dir(RP.rep_root(root, rep), cell) if rep != 1 else RW.run_dir(results/'rw_v1', cell))
                record = json.loads((d/'repair.json').read_text())
                write_json(d/'repair.json', dict(record, config=changed))
    elif what == 'code':
        write_rep(results, 2, code_sha='other')
    elif what == 'source':
        write_rep(results, 3, overrides={P0: dict(tapered='t_other')})
    else:
        d = RW.run_dir(results/'rw_v1', RW.cell_of(P0))
        write_json(d/'repair.json', dict(json.loads((d/'repair.json').read_text()), code_sha='other'))
    d = decision.write_rp(root)
    for r in (d['strict'], d['code_convention']):
        assert r['reading'] == decision.UNDECIDABLE and r['reading_spread'] == decision.UNDECIDABLE and word in r['reason'], r['reason']


def test_a_wrong_oracle_mark_writes_no_reading(results):
    root = written(results, r2=dict(overrides={P0: dict(oracle=None)}))
    with pytest.raises(SystemExit, match='not oracle false'):
        decision.write_rp(root)
    assert not (root/'analysis').exists()


def test_r1_is_only_rw_v1s_own_repairs(results):
    root = written(results)
    shutil.rmtree(results/'rw_v1')
    write_rw(results, same=True)                                     # had RW's control been bit for bit, 13 pairs would be fd2_v1's 120 s repairs: not R1
    d = decision.write_rp(root)
    assert d['reading'] == decision.UNDECIDABLE and '0443-0444 R1（不是 rw_v1 自己的修复（fd2_v1））' in d['reason'] and d['r1_sources']['0443-0444'] == 'fd2_v1' and d['r1_sources']['0438-0439'] == 'rw_v1'


def test_the_per_pair_dispersion(results):
    root = written(results, r2=dict(overrides={P0: dict(rounds=2, seconds=150., bound=4e-8, seg35=.8742)}), r3=dict(overrides={P0: dict(bound=5e-6, seg35=.8738)}))
    d = decision.write_rp(root)
    row = next(p for p in d['per_pair'] if p['pair'] == '0438-0439')
    assert row['rounds']['values'] == [2, 2, 1] and row['rounds']['range'] == 1 and row['seconds_repair']['values'] == [60., 150., 60.] and row['seconds_repair']['range'] == pytest.approx(90.)
    assert row['task_clock']['values'] == [pytest.approx(45.+60.+1.2), pytest.approx(45.+150.+1.2), pytest.approx(45.+60.+1.2)]
    assert row['min_lower_bound_after']['values'] == [1e-6, 4e-8, 5e-6] and row['lower_bound_ratio'] == pytest.approx(5e-6/4e-8)
    assert row['seg35']['range'] == pytest.approx(.0004) and row['seg35']['mean'] == pytest.approx((.874+.8742+.8738)/3)


def test_a_second_reading_keeps_the_first(results):
    root = written(results)
    decision.write_rp(root)
    decision.write_rp(root)
    assert len(list((root/'analysis').glob('previous_*/decision_rp.json'))) == 1


def test_the_decomposition_says_how_much_of_the_range_is_the_fallback(results):
    root = written(results, r2=dict(overrides={FAILED: dict(certified=False)}), r3=dict(overrides={P0: dict(seg35=.8744)}))
    d = decision.write_rp(root)
    dc = d['strict']['decomposition']
    assert dc['n_common'] == 18 and dc['common_seg35']['range'] == pytest.approx(.0004/18) and dc['fallback_shift_seg35']['R1'] == 0. and dc['fallback_shift_seg35']['R3'] == 0.
    assert dc['fallback_shift_seg35']['R2'] == pytest.approx((.82-.874)/19)           # the pair's repaired seg35 in R1 and R3 is .874
    assert d['code_convention']['decomposition']['n_common'] == 18
    assert '分解（只描述）' in (root/'analysis/summary_rp.md').read_text()


def test_a_pair_that_fails_everywhere_has_no_reference_for_the_decomposition(results):
    root = written(results, r2=dict(overrides={FAILED: dict(certified=False)}), r3=dict(overrides={FAILED: dict(certified=False)}))
    d = RW.run_dir(results/'rw_v1', RW.cell_of(FAILED))
    evaluate_repair(d, certified=False)
    out = decision.write_rp(root)
    assert out['strict']['decomposition']['fallback_shift_seg35'] == dict(R1=None, R2=None, R3=None) and out['strict']['decomposition']['n_common'] == 18


def test_a_crash_is_not_read_for_the_readings_and_a_failure_in_the_described_line(results):
    root = written(results)
    cell = RP3.cell_of(FAILED)
    d = RP3.run_dir(RP.rep_root(root, 3), cell)
    (d/'repair_metrics.json').unlink()
    (d/'repair.json').unlink()
    write_json(d/'queue_done.json', dict(run_id=cell.run_id, rc=1, unit_seconds=3., state='failed', arm_stop=None))
    out = decision.write_rp(root)
    c = out['strict']['crash_as_failure']
    assert out['reading'] == decision.UNDECIDABLE and out['reading_spread'] == decision.UNDECIDABLE
    assert c['described_only'] is True and c['n_crashed'] == 1 and c['n_a0_used'] == 1 and c['seg35']['values'][2] == pytest.approx((18*.874+.82)/19) and c['seg35']['range'] == pytest.approx(.874-(18*.874+.82)/19)
    assert out['per_rep'][2]['n_crashed'] == 1 and out['per_rep'][2]['method_crash_as_failure']['strict']['substituted_a0'] == ['0451-0452']
    assert '崩溃按失败代入（只描述，不进判定）：崩溃 1 次' in (root/'analysis/summary_rp.md').read_text()


@pytest.mark.parametrize('what,word', (('certificate_config', 'one repair_certificate_config'), ('frozen_commit', 'one frozen_commit'), ('evaluator', 'one evaluator_sha256'), ('evaluation_certificate', 'one evaluation_certificate'), ('data', 'read different data')))
def test_the_other_things_three_repetitions_must_share(results, what, word):
    root = written(results)
    d = RP2.run_dir(RP.rep_root(root, 2), RP2.cell_of(P0))
    record, metrics = json.loads((d/'repair.json').read_text()), json.loads((d/'repair_metrics.json').read_text())
    if what == 'certificate_config':
        record['certificate_config'] = dict(CERT_CONFIG, max_boxes=64)
    elif what == 'frozen_commit':
        record['frozen_commit'] = 'x'
    elif what == 'evaluator':
        metrics['evaluator']['sha256'] = 'x'
    elif what == 'evaluation_certificate':
        metrics['after']['certificate']['attempts'][0]['certificate_version'] = 'x'
    else:
        record['data']['pair_sha256'] = 'x'
    write_json(d/'repair.json', record)
    write_json(d/'repair_metrics.json', metrics)
    out = decision.write_rp(root)
    assert out['reading'] == decision.UNDECIDABLE and word in out['reason'], out['reason']


def test_the_environment_and_the_device_cap_are_described_not_judged(results):
    root = written(results)
    d = RP3.run_dir(RP.rep_root(root, 3), RP3.cell_of(P0))
    write_json(d/'repair.json', dict(json.loads((d/'repair.json').read_text()), env=dict(python='3.10.20', torch='2.5.1', device='another GPU')))
    out = decision.write_rp(root)
    assert out['reading'] == '3 次都 19/19' and len(out['controls']['described']['env']) == 2 and len(out['controls']['described']['device_cap']) == 1
    assert 'another GPU' in (root/'analysis/summary_rp.md').read_text()


# ---- the fallback: pins, its qualification by the records, the identity ---------------------------------------------------------------------------
def test_the_pins_come_from_the_records_and_say_configuration_c(results):
    pins = RUN.fallback_pins_from_records(results)
    assert pins['schema'] == 'bayesreg.rp.fallback.v1' and set(pins['pairs']) == {core.short_name(p) for p in ALL}
    pin = pins['pairs']['0438-0439']
    assert pins['evaluation_certificate']['config'] == CERT_CONFIG and pins['evaluation_certificate']['certificate_version'] == CERT_VERSION and pin['d0_evaluation']['evaluator_sha256'] == rp.D0_EVALUATOR_SHA256
    assert pin['certificate']['config'] == CERT_CONFIG and pin['certificate']['status'] == 'certified_feasible' and pin['field'] == f'runs/A0/{P0}/pg_weak__four__90s/final_native.npz' and pin['d0_evaluation']['passed'] is True


@pytest.mark.skipif(not (REAL_RESULTS/'d0_v1/runs/A0').is_dir(), reason='d0_v1 is not on this machine')
def test_the_committed_pins_are_d0_v1s_records():
    """Reads d0_v1's JSON records only (no field)."""
    committed = json.loads(rp.FALLBACK_FILE.read_text())
    assert committed == RUN.fallback_pins_from_records(REAL_RESULTS)
    assert len(committed['pairs']) == 19 and all(p['certificate']['status'] == 'certified_feasible' and p['certificate']['config'] == committed['evaluation_certificate']['config'] == CERT_CONFIG and p['certificate']['certificate_version'] == CERT_VERSION and p['certificate']['n_infeasible'] == 0
                                                 and p['d0_evaluation']['passed'] and p['d0_evaluation']['classification'] == 'ok' and p['d0_evaluation']['final_npz_sha256'] == p['file_sha256'] and p['d0_evaluation']['evaluator_sha256'] == rp.D0_EVALUATOR_SHA256 for p in committed['pairs'].values())


def test_write_pins_is_once():
    assert rp.FALLBACK_FILE.is_file()
    with pytest.raises(SystemExit, match='a pin is written once'):
        RUN.main(['write-pins'])


@pytest.mark.parametrize('what,word', ((None, None), ('file', 'not the pinned file'), ('missing', 'no final_native.npz'), ('pin', 'has no pin'), ('record', "the run's certificate"), ('d0', "D0's evaluation"), ('class', "D0's class method_failure"), ('config', 'not configuration C'), ('shape', 'not a float64 field of the image grid'), ('metrics', "D0's metrics are not there"), ('no_d0', 'no D0 metrics.json'), ('state', 'not of the pinned u'),
                                       ('kernel', "not RW's evaluation's"), ('version', "not RW's evaluation's"), ('final_npz', 'not of the pinned file'), ('final_npz_matches', 'not of the pinned file'), ('evaluator', 'not the pinned evaluate.py')))
def test_the_qualification_of_configuration_c_by_the_records(results, what, word):
    pins_path = rp.FALLBACK_FILE
    d = RP.a0_run_dir(P0, results)
    final_path, metrics_path = d/'pg_weak__four__90s/final_native.json', d/'metrics.json'
    final, m = json.loads(final_path.read_text()), json.loads(metrics_path.read_text())
    if what == 'file':
        np.savez(d/'pg_weak__four__90s/final_native.npz', u=np.ones((6, 7, 8, 3)))
    elif what == 'missing':
        (d/'pg_weak__four__90s/final_native.npz').unlink()
    elif what == 'pin':
        pins = json.loads(pins_path.read_text())
        del pins['pairs']['0438-0439']
        pins_path.write_text(json.dumps(pins))
    elif what == 'record':
        final['certificate']['n_infeasible'] = 1
    elif what == 'd0':
        m['certificate']['passed'] = False
    elif what == 'class':
        m['classification']['class'] = 'method_failure'
    elif what == 'config':
        final['certificate']['config']['max_boxes'] = 64
    elif what == 'metrics':
        m['metrics']['seg35'] = None
    elif what == 'no_d0':
        metrics_path.unlink()
    elif what == 'state':
        final['certificate']['state_hash'] = 'x'
    elif what == 'kernel':
        final['certificate']['config']['kernel'] = 'another kernel'
    elif what == 'version':
        final['certificate']['certificate_version'] = 'another version'
    elif what == 'final_npz':
        m['certificate']['final_npz_sha256'] = 'x'
    elif what == 'final_npz_matches':
        m['certificate']['final_npz_sha256_matches'] = False
    elif what == 'evaluator':
        m['evaluator']['sha256'] = 'x'
    elif what == 'shape':
        u = np.zeros((6, 7, 9, 3))
        np.savez(d/'pg_weak__four__90s/final_native.npz', u=u)
        sha = core.sha_file(d/'pg_weak__four__90s/final_native.npz')
        final['sha256'], final['certificate']['state_hash'], m['certificate']['state_hash'] = sha, u_digest(u), u_digest(u)
        pins = json.loads(pins_path.read_text())
        pins['pairs']['0438-0439'].update(file_sha256=sha, u_sha256=u_digest(u))
        pins_path.write_text(json.dumps(pins))
    if what not in ('no_d0', 'missing'):
        write_json(final_path, final)
        write_json(metrics_path, m)
    status = rp.fallback_status(P0, results=results)
    if what is None:
        assert status['qualified'] is True and status['why'] is None and status['metrics']['seg35'] == .82 and status['d0_class'] == 'ok' and "D0's run's own" in status['source'] and 'not computed again' in status['source']
    else:
        assert status['qualified'] is False and word in status['why'], status['why']
    lines = []
    statuses = RUN.fallback_lines(lines.append, results=results, stage=core.STAGE)
    assert len(statuses) == 19 and (statuses['0438-0439']['qualified'] is (what is None)) and ('none' in lines[-1]) is (what is None)


def test_the_qualification_reads_no_field_value_and_computes_no_certificate(results, monkeypatch):
    monkeypatch.setattr(np, 'load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('np.load')))
    import jd_curl.three_d.feasibility as feasibility
    monkeypatch.setattr(feasibility, 'certify_map', lambda *a, **k: (_ for _ in ()).throw(AssertionError('a certificate')))
    assert all(rp.fallback_status(p, results=results)['qualified'] for p in ALL)


def test_the_evaluation_of_the_identity_refuses_another_root(capsys):
    assert RUN.main(['id', 'evaluate', '--root', str(RP.DEFAULT_ROOT/'r2')]) == 2 and 'the identity is evaluated only into' in capsys.readouterr().err
    assert RUN.main(['rp2', 'evaluate', '--root', str(RP.DEFAULT_ROOT/'r3')]) == 2


def test_the_identity_is_evaluated_only_where_configuration_c_does_not_qualify(results, monkeypatch):
    spoil_a0(results, R0, 'class')
    assert RUN.identity_pairs(results) == [R0]
    seen = []
    monkeypatch.setattr(RUN, 'evaluate_identity', lambda d, pair, *a: seen.append(pair) or write_json(d/'identity_metrics.json', dict(certified_at_1=True, oracle=False, metrics=metrics_of(.6, 0.))))
    monkeypatch.setattr(fl, 'read_truth', lambda pair, members: (None, None))
    monkeypatch.setattr(RUN.ANY, 'enable_jit', lambda root: None)
    called = []
    monkeypatch.setattr(core, 'frozen_api', lambda: called.append(1))
    o = RUN.argparse.Namespace(root=str(RP.id_root()))
    assert RUN.identity_evaluate(o, results=results) == 0 and seen == [R0] and called == [1]          # frozen_api first
    assert RUN.identity_evaluate(o, results=results) == 0 and seen == [R0]          # a record is not done again
    assert len(list(RP.id_root().glob('plan_*.json'))) >= 1


# ---- the queue on a fake machine --------------------------------------------------------------------------------------------------------------------
class FakeSystem:
    """The machine: nothing starts; a repair unit writes the records it would, an evaluation unit writes the metrics of what is there."""

    def __init__(self, behaviour=None, not_certified=(), eval_rc=0):
        self.behaviour, self.not_certified, self.eval_rc = behaviour, not_certified, eval_rc
        self.launched, self.checkouts, self.sleeps, self.clock = [], [], [], 0.
        self.load, self.other, self.free_gpu, self.free_home, self.avail, self.free_host = .1, [], 20000., 500., 12., 40.

    def load1(self):
        return self.load

    def units(self):
        return ['tag-rp-queue.service']+list(self.other)

    def gpu_free_mib(self):
        return self.free_gpu

    def home_free_gb(self):
        return self.free_home

    def mem_available_gib(self):
        return self.avail

    def host_free_gib(self):
        return self.free_host

    def now(self):
        return self.clock

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.clock += seconds

    def code_checkout(self, root, commit):
        self.checkouts.append((str(root), commit))
        return Path(root)/'code', commit or 'fakesha'

    def run(self, argv, log):
        unit = next(a for a in argv if a.startswith('--unit=')).split('=', 1)[1]
        self.launched.append(unit)
        assert 'py'+'test' not in ' '.join(argv)
        script = next(i for i, a in enumerate(argv) if a.endswith('experiments/fl/run_rp.py'))
        entry = argv[script+1]
        if entry == 'id':
            write_identity(RP.RESULTS, RUN.identity_pairs())
            return 0
        rep = int(entry[2:])
        batch, where = BATCHES[rep], RP.rep_root(RP.DEFAULT_ROOT, rep)
        if argv[script+2] == 'evaluate':
            for cell in batch.plan():
                d = batch.run_dir(where, cell)
                if (d/'repair.json').is_file() and not (d/'repair_metrics.json').exists():
                    evaluate_repair(d, certified=cell.pair not in self.not_certified, seg35=.874)
            return self.eval_rc
        cell = batch.require_cell(argv[argv.index('--arm')+1], argv[argv.index('--tier')+1], argv[argv.index('--pair')+1])
        assert Path(argv[argv.index('--out')+1]) == batch.run_dir(where, cell)
        self.clock += 60.
        rc = self.behaviour(cell, rep, self) if self.behaviour else None
        if rc is None:
            write_repair(batch.run_dir(where, cell), cell.run_id, certified=cell.pair not in self.not_certified, code_sha=argv[argv.index('--code-sha')+1], evaluate=False)
            rc = 0
        return rc


def options(**kw):
    return RUN.argparse.Namespace(**dict(dict(root=str(RP.DEFAULT_ROOT), commit=None, wait_minutes=30, gap_seconds=5, min_host_gib=15.), **kw))


@pytest.fixture
def machine(results, monkeypatch):
    write_rw(results)
    monkeypatch.setattr(fxrun, 'nx_commit', lambda commit: commit or 'fakesha')
    monkeypatch.setattr(rp, 'identity', lambda sha, **kw: (dict(base=rp.BASE_COMMIT, sha=sha, same=True, entries=[], added_own=list(rp.OWN_FILES)), []))
    monkeypatch.setattr(fxrun, 'safe', lambda fn, out=None, **kw: [])              # the start checks have their own tests above; the fallback's lines run for real
    monkeypatch.setattr(RUN.ANY, 'committed_bytes', lambda sha, rel: (ROOT/rel).read_bytes())
    return results


EVALS = ['tag-rp2-eval', 'tag-rp3-eval']


def plan_units():
    return [c.unit for c in RP.plan()]


def test_the_queue_runs_38_repairs_in_order_then_the_evaluations_and_reads(machine):
    system = FakeSystem()
    assert RUN.queue(options(), system) == 0
    assert system.launched == plan_units()+EVALS and system.checkouts == [(str(RP.DEFAULT_ROOT), 'fakesha')]          # every field of configuration C qualifies: no identity unit
    end = json.loads((RP.DEFAULT_ROOT/'queue_end.json').read_text())
    assert end['reading'] == '3 次都 19/19' and end['reading_spread'] == decision.READ_SMALL and set(end['eval_rc'].values()) == {0} and end['code_convention']['reading'] == '3 次都 19/19' and end['n_identity_used'] == 0 and end['fallback_not_qualified'] == []
    batch = json.loads((RP.DEFAULT_ROOT/'batch.json').read_text())
    assert batch['schema'] == 'bayesreg.rp.batch.v1' and batch['code_sha'] == 'fakesha' and batch['plan'] == [c.run_id for c in RP.plan()] and batch['identity']['same'] is True and all(v['qualified'] for v in batch['fallback_at_start'].values()) and len(batch['fallback_at_start']) == 19
    assert batch['caps'] == RW_CAPS and {k: batch['gate'][k] for k in RW_GATE} == RW_GATE and batch['evaluation'] == RW_EVALUATION and batch['gate']['own_units'] == ['tag-rp-queue.service'] and batch['r1']['code_sha'] == rp.BASE_COMMIT
    assert system.sleeps.count(5) == 36                                 # 18 gaps in each repetition
    again = FakeSystem()
    assert RUN.queue(options(), again) == 0 and again.launched == EVALS       # a restart reruns nothing


def test_a_failure_is_a_result_not_a_stop(machine):
    system = FakeSystem(not_certified=(FAILED,))
    assert RUN.queue(options(), system) == 0
    end = json.loads((RP.DEFAULT_ROOT/'queue_end.json').read_text())
    assert end['reading'] == '有失败 2/57' and '0451-0452 R2' in end['reason'] and '0451-0452 R3' in end['reason'] and end['reading_spread'] == decision.READ_MIDDLE


def test_a_configuration_c_that_does_not_qualify_starts_the_identity_unit_and_the_failure_takes_it(machine):
    spoil_a0(machine, FAILED, 'class')
    system = FakeSystem(not_certified=(FAILED,))
    assert RUN.queue(options(), system) == 0
    assert system.launched == plan_units()+EVALS+['tag-rp-eval-id']
    end = json.loads((RP.DEFAULT_ROOT/'queue_end.json').read_text())
    assert end['fallback_not_qualified'] == ['0451-0452'] and end['n_identity_used'] == 2 and end['reading'] == '有失败 2/57'
    assert json.loads((RP.DEFAULT_ROOT/'batch.json').read_text())['fallback_at_start']['0451-0452']['qualified'] is False


def test_a_crash_is_undecidable_with_exit_8_and_the_rest_goes_on(machine):
    system = FakeSystem(behaviour=lambda cell, rep, s: 1 if (rep, cell.pair) == (2, FAILED) else None)
    assert RUN.queue(options(), system) == 8
    assert system.launched == plan_units()+EVALS
    d = json.loads((RP.DEFAULT_ROOT/'analysis/decision_rp.json').read_text())
    assert d['reading'] == decision.UNDECIDABLE and '0451-0452 R2' in d['reason'] and 'is 不可判' in json.loads((RP.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']


def test_three_crashes_in_a_row_stop_the_repairs_and_what_exists_is_evaluated(machine):
    system = FakeSystem(behaviour=lambda cell, rep, s: 1 if rep == 2 and cell in RP2.plan()[4:7] else None)
    assert RUN.queue(options(), system) == 8
    assert system.launched == plan_units()[:7]+['tag-rp2-eval'] and '3 failures in a row' in json.loads((RP.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']
    assert not RP.rep_root(RP.DEFAULT_ROOT, 3).exists() and json.loads((RP.DEFAULT_ROOT/'analysis/decision_rp.json').read_text())['reading'] == decision.UNDECIDABLE


def test_a_failed_evaluation_says_so(machine):
    system = FakeSystem(eval_rc=1)
    assert RUN.queue(options(), system) == 8
    assert 'an evaluation unit failed' in json.loads((RP.DEFAULT_ROOT/'queue_stop.json').read_text())['reason'] and (RP.DEFAULT_ROOT/'analysis/decision_rp.json').is_file()


@pytest.mark.parametrize('field,value,word', (('other', ['tag-uf-train.service'], 'other units'), ('load', 1.5, 'load'), ('free_gpu', 1000., 'GPU free'), ('free_host', 14., '/mnt/c'), ('other', ['seq-vp-chain.service'], 'other units')))
def test_the_gate_closed_starts_nothing(machine, field, value, word):
    system = FakeSystem()
    setattr(system, field, value)
    assert RUN.queue(options(), system) == 8
    assert system.launched == [] and word in json.loads((RP.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']


def test_the_hold_file_at_the_root_stops_the_next_repair_and_the_next_evaluation(machine):
    system = FakeSystem(behaviour=lambda cell, rep, s: (RP.DEFAULT_ROOT/'hold').write_text('') if cell == RP2.plan()[1] else None)
    assert RUN.queue(options(), system) == 8
    assert system.launched == plan_units()[:2] and 'hold file' in json.loads((RP.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']


def test_the_identity_the_inputs_or_the_fallbacks_records_stop_before_anything_is_checked_out(machine, monkeypatch):
    system = FakeSystem()
    monkeypatch.setattr(rp, 'identity', lambda sha, **kw: (dict(same=False), ['M experiments/ef/repair.py: a file of RW\'s commit']))
    assert RUN.queue(options(), system) == 8
    stop = json.loads((RP.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']
    assert "the code is not RW's" in stop and 'nothing checked out' in stop and system.checkouts == [] and system.launched == [] and not (RP.DEFAULT_ROOT/'batch.json').exists()
    monkeypatch.setattr(rp, 'identity', lambda sha, **kw: (dict(same=True), []))
    monkeypatch.setattr(fxrun, 'safe', lambda fn, out=None, **kw: ['R1 0438-0439: not RW\'s evaluated repair'])
    assert RUN.queue(options(), system) == 8
    assert 'the inputs, before any run' in json.loads((RP.DEFAULT_ROOT/'queue_stop.json').read_text())['reason'] and system.checkouts == [] and len(list(RP.DEFAULT_ROOT.glob('queue_stop_*.json'))) == 1
    monkeypatch.setattr(fxrun, 'safe', lambda fn, out=None, **kw: [])
    rp.FALLBACK_FILE.write_text('not json')
    assert RUN.queue(options(), system) == 8
    assert "configuration C's fields could not be read" in json.loads((RP.DEFAULT_ROOT/'queue_stop.json').read_text())['reason'] and system.checkouts == []


def test_a_driver_whose_rules_are_not_the_batchs_stops(machine, monkeypatch):
    system = FakeSystem()
    monkeypatch.setattr(RUN.ANY, 'committed_bytes', lambda sha, rel: b'other' if rel == 'experiments/fl/decide_rp.py' else (ROOT/rel).read_bytes())
    assert RUN.queue(options(), system) == 8
    assert 'decide_rp.py' in json.loads((RP.DEFAULT_ROOT/'queue_stop.json').read_text())['reason'] and system.launched == []
    assert RUN.main(['decide', '--root', str(RP.DEFAULT_ROOT)]) == 8


def test_a_restart_at_another_commit_stops(machine):
    assert RUN.queue(options(), FakeSystem()) == 0
    system = FakeSystem()
    assert RUN.queue(options(commit='othersha'), system) == 8
    assert 'batch.json is at fakesha' in json.loads((RP.DEFAULT_ROOT/'queue_stop.json').read_text())['reason'] and system.launched == []


def test_the_queue_does_not_write_under_rw_v1_or_d0_v1_and_another_root_touches_nothing(machine, capsys):
    trees = {name: {p: p.read_bytes() for p in (machine/name).rglob('*') if p.is_file()} for name in ('rw_v1', 'd0_v1')}
    assert RUN.queue(options(), FakeSystem()) == 0
    assert {name: {p: p.read_bytes() for p in (machine/name).rglob('*') if p.is_file()} for name in ('rw_v1', 'd0_v1')} == trees
    other = Path(tempfile.mkdtemp(prefix='rpo_'))/'rp_v1'
    assert RUN.queue(options(root=str(other)), FakeSystem()) == 2 and not other.exists() and 'RP writes only to' in capsys.readouterr().err


# ---- the dry run ----------------------------------------------------------------------------------------------------------------------------------
def dry_world(base):
    """A stage, fd2_v1's fields and their pins, rw_v1, d0_v1, with the real names and synthetic content."""
    stage = stage_world(base)
    results = base/'results'
    for pair in ALL:
        fixed, moving = pair.split('__')
        write_json(stage/pair/'manifest.json', dict(pair=pair, fixed_subject=fixed, moving_subject=moving, shape_zyx=[6, 7, 8], spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm', sources=dict(fixed=dict(sha256='0'*64), moving=dict(sha256='0'*64)), pair_sha256='0'*64))
    write_json(stage/'selection.json', dict(pairs=[dict(pair=p) for p in ALL]))
    pins = {}
    for pair in ALL:
        short = core.short_name(pair)
        d = fxrun.fd2.run_dir(results/'fd2_v1', RW.fd2_field_cell(pair))
        d.mkdir(parents=True)
        np.savez(d/'field_0600.npz', u=np.zeros((4, 4, 4, 3)))
        write_json(d/'run.json', dict(status='done', oracle=False, scale=dict(D_factor=24.), checkpoints={'600': dict(file='field_0600.npz', sha256=core.sha_file(d/'field_0600.npz'), u_sha256='u', seconds_task=45.)}))
        write_json(d/'metrics_0600.json', dict(tapered_u_sha256=f't_{short}'))
        r = fxrun.fd2.run_dir(results/'fd2_v1', RW.fd2_repair_cell(pair))
        wall = pair in RW.WALL_PAIRS
        write_repair(r, f'fd2_RHx24_{short}', rounds=2 if wall else 1, wall=wall, config=dict(RW_CONFIG, wall_seconds=120.), code_sha='fd2')
        pins[short] = dict(field_sha256=core.sha_file(d/'field_0600.npz'), tapered_u_sha256=f't_{short}', old_attempt_sha256=json.loads((r/'repair.json').read_text())['attempt']['sha256'])
    write_json(base/'rw_pins.json', dict(pairs=pins))
    write_rw(results)
    write_d0(results, stage)
    write_json(base/'a0_pins.json', RUN.fallback_pins_from_records(results))
    return dict(stage=stage, results=results, sources=base/'rw_pins.json', fallback=base/'a0_pins.json')


def test_the_dry_run_lists_the_38_repairs_and_the_19_fields_of_configuration_c(results, monkeypatch):
    base = Path(tempfile.mkdtemp(prefix='rpw_'))
    world = dry_world(base)
    monkeypatch.setattr(RW, 'SOURCES_FILE', world['sources'])
    monkeypatch.setattr(rp, 'identity', lambda sha, **kw: (dict(n_files_at_base=177, n_files_changed=0, added_own=list(rp.OWN_FILES), added_other=[]), []))
    lines = []
    root = Path(tempfile.mkdtemp(prefix='rpd_'))/'rp_v1'
    assert RUN.dry_run(out=lines.append, root=root, stage=world['stage'], results=world['results'], sources=world['sources'], fallback=world['fallback']) == 0, '\n'.join(lines)
    text = '\n'.join(lines)
    for cell in RP.plan():
        assert cell.run_id in text and cell.unit in text
    assert 'problems: none' in text and 'root exists: False' in text and not root.exists() and 'tag-rp-queue' in text and 'tag-rp-eval-id' in text and 'RuntimeMaxSec=1800' in text and 'estimates' in text and "do not qualify (the identity u = 0 takes their place; not a problem, for the lead): none" in text
    assert sum('qualified by the records: True' in line for line in lines) == 19 and sum(line.startswith('R1 ') for line in lines) == 21
    assert sum(line.startswith('pinned source ') and 'field_sha256' in line and 'tapered_u_sha256 t_' in line for line in lines) == 19 and any(line.startswith('pinned inputs: ') and f"sha256 {core.sha_file(world['sources'])}" in line for line in lines) and any(line.startswith('the frozen tree ') for line in lines)
    spoil_a0(world['results'], P0, 'class')
    lines = []
    assert RUN.dry_run(out=lines.append, root=root, stage=world['stage'], results=world['results'], sources=world['sources'], fallback=world['fallback']) == 0
    assert any("not a problem, for the lead): ['0438-0439']" in line for line in lines) and 'problems: none' in lines[-1]
    monkeypatch.setattr(rp, 'identity', lambda sha, **kw: (dict(), ['M experiments/ef/repair.py']))
    lines = []
    assert RUN.dry_run(out=lines.append, root=root, stage=world['stage'], results=world['results'], sources=world['sources'], fallback=world['fallback']) == 8 and 'M experiments/ef/repair.py' in lines[-1]
    shutil.rmtree(base)


DRY = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
import run_rp as run
world = json.loads(sys.argv[2])
run.RW.SOURCES_FILE = Path(world['sources'])
run.rp.identity = lambda sha, **kw: (dict(), [])
seen, real_header = set(), run.run.npz_header
def header(path, member):
    seen.add(Path(path).name+':'+member)
    return real_header(path, member)
run.run.npz_header = header
def no_load(*args, **kwargs):
    raise AssertionError('np.load in a dry run')
np.load = no_load
lines = []
rc = run.dry_run(out=lines.append, root=Path(world['root']), stage=Path(world['stage']), results=Path(world['results']), sources=Path(world['sources']), fallback=Path(world['fallback']))
print(json.dumps(dict(rc=rc, last=lines[-1], seen=sorted(seen), torch='torch' in sys.modules, root_there=Path(world['root']).exists())))
"""


def test_the_dry_run_in_its_own_process_imports_no_torch_and_opens_no_label():
    base = Path(tempfile.mkdtemp(prefix='rpw_'))
    world = dry_world(base)
    arguments = dict({k: str(v) for k, v in world.items()}, root=str(Path(tempfile.mkdtemp(prefix='rpd_'))/'rp_v1'))
    done = subprocess.run([PY, '-B', '-c', DRY, str(ROOT/'experiments/fl'), json.dumps(arguments)], env=dict(ENV, PYTHONPATH=str(ROOT)), cwd=base, capture_output=True, text=True)
    shutil.rmtree(base)
    assert done.returncode == 0, done.stdout[-1500:]+done.stderr[-2500:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out['rc'] == 0 and out['torch'] is False and 'torch imported by this dry run: False; problems: none' in out['last'] and out['root_there'] is False and set(out['seen']) == {'pair.npz:fixed', 'pair.npz:moving', 'final_native.npz:u'}


# ---- one synthetic repair of each repetition and the identity through the real path ---------------------------------------------------------------
PIPELINE = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
import run_rp as run
import run_fd2
fxrun, fx, fd2, fl, core, rp, decision = run.fxrun, run.fx, run.fxrun.fd2, run.fl, run.core, run.rp, run.decision
RW, RP = fx.RW, rp.RP
world = json.loads(sys.argv[2])
results = Path(world['results'])
fx.RESULTS = results
fd2.DEFAULT_ROOT, fd2.RESULTS = results/'fd2_v1', results
for batch in (RW, rp.RP2, rp.RP3):
    batch.RESULTS, batch.DEFAULT_ROOT = results, results/batch.ROOT_NAME
RP.RESULTS, RP.DEFAULT_ROOT = results, results/'rp_v1'
core.STAGE = Path(world['stage'])
core.SETUP_WORKERS = 2
fd2.COARSE_ITERATIONS = (2, 2, 2)
fx.NATIVE = fd2.NATIVE = 4
fxrun.decision.NATIVE = fxrun.decision.STEP = fxrun.decision.fd2_reading.STEP = 4
fd2.REPAIR_STEPS = (4,)
for spec in (fd2.ARMS['Hx24'], fx.SPEC):
    spec.update(native=4, checkpoints=(4,))
run.mc.SEG_ROOT = Path(world['seg_root'])
unet = core.make_inputs().UNetSource(pred_root=Path(world['pred']))
run.mc.default_unet = lambda: unet
pair = core.PAIRS[0]
short = core.short_name(pair)
for batch in (RW, rp.RP2, rp.RP3):
    batch.CONTROL_PAIR, batch.WALL_PAIRS, batch.REUSED_PAIRS = pair, (), ()
def one(entry, *argv):
    assert entry.main(list(argv)+['--device', 'cpu', '--code-sha', 'x']) == 0
run_fd2.base.enable_jit = lambda root: None
for base in [fxrun.RW_BASE]+list(run.BASES.values()):
    base.enable_jit = lambda root: None
# FD2's own entry writes the Hx24 field (the source)
d = fd2.run_dir(fd2.DEFAULT_ROOT, fd2.Cell('Hx24', fd2.FIRST, pair))
one(run_fd2, 'run-one', '--arm', 'Hx24', '--tier', fd2.FIRST, '--pair', pair, '--out', str(d))
assert run_fd2.base.evaluate_all(run_fd2.argparse.Namespace(root=str(fd2.DEFAULT_ROOT)), cells=[fd2.Cell('Hx24', fd2.FIRST, pair)]) == 0
checkpoint = json.loads((d/'run.json').read_text())['checkpoints']['4']
tapered = json.loads((d/'metrics_0004.json').read_text())['tapered_u_sha256']
RW.SOURCES_FILE = results/'pins.json'
RW.SOURCES_FILE.write_text(json.dumps(dict(pairs={short: dict(field_sha256=checkpoint['sha256'], tapered_u_sha256=tapered)})))
# R1: RW's entry; R2, R3: RP's, with the firewall's spy on every np.load of the repairs
real_load, seen = np.load, set()
class Spy:
    def __init__(self, path, inner):
        self.name, self.inner = Path(path).name, inner
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.inner.close()
    def __contains__(self, key):
        return key in self.inner
    def __getitem__(self, key):
        seen.add(self.name+':'+key)
        if 'seg' in key:
            raise AssertionError('a label member was read: '+key)
        return self.inner[key]
    @property
    def files(self):
        return self.inner.files
np.load = lambda path, **kwargs: Spy(path, real_load(path, **kwargs))
c = RW.Cell('RHx24', 'c4', pair)
one(fxrun, 'rw', 'repair-one', '--arm', 'RHx24', '--tier', 'c4', '--pair', pair, '--out', str(RW.run_dir(RW.DEFAULT_ROOT, c)))
for rep in rp.REPEATS:
    batch = rp.BATCHES[rep]
    cell = batch.Cell('RHx24', 'c4', pair)
    one(run, f'rp{rep}', 'repair-one', '--arm', 'RHx24', '--tier', 'c4', '--pair', pair, '--out', str(batch.run_dir(batch.DEFAULT_ROOT, cell)))
np.load = real_load
assert fxrun.RW_BASE.evaluate_all(fxrun.argparse.Namespace(root=str(RW.DEFAULT_ROOT)), cells=[c]) == 0
for rep in rp.REPEATS:
    assert run.main([f'rp{rep}', 'evaluate', '--root', str(rp.BATCHES[rep].DEFAULT_ROOT)]) == 0
# the second tier: no record of configuration C for this pair here, so it does not qualify; the identity unit's entry evaluates u = 0
rp.FALLBACK_FILE = results/'a0_pins.json'
rp.FALLBACK_FILE.write_text(json.dumps(dict(pairs={})))
run.ALL_PAIRS = rp.ALL_PAIRS = decision.ALL_PAIRS = fxrun.decision.ALL_PAIRS = (pair,)
status = rp.fallback_status(pair)
assert status['qualified'] is False
assert run.main(['id', 'evaluate', '--root', str(RP.id_root())]) == 0
# the reading on this one pair: RW's control judged as rw_v1's was
RW.write_selection(RW.DEFAULT_ROOT, RW.CONTROL_FILE, dict(same=False))
rp.BASE_COMMIT = 'x'
(RP.DEFAULT_ROOT/'batch.json').write_text(json.dumps(dict(code_sha='x')))
decision.N = 3
result = decision.write_rp(RP.DEFAULT_ROOT)
(results/'out.json').write_text(json.dumps(dict(seen=sorted(seen), status_why=status['why'], reading=result['reading'], spread=result['reading_spread'], range=result['method_seg35_range'], problems=result['controls']['problems'], distinct=result['per_pair'][0]['n_distinct_repaired_fields'], identity_used=result['strict']['n_identity_used'], identity_usable=result['identity']['n_usable'])))
print('done')
"""


@needs_frozen
def test_one_repair_of_each_repetition_and_the_identity_through_the_real_path(tmp_path, monkeypatch):
    world = pipeline_world(tmp_path, monkeypatch)
    world['results'] = str(tmp_path/'results')
    done = subprocess.run([PY, '-B', '-c', PIPELINE, str(ROOT/'experiments/fl'), json.dumps(world)], env=ENV, cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0 and 'done' in done.stdout, done.stdout[-3000:]+done.stderr[-4000:]
    results = tmp_path/'results'
    out = json.loads((results/'out.json').read_text())
    short = core.short_name(core.PAIRS[0])
    rw = json.loads((results/f'rw_v1/runs/RHx24_c4/{short}/repair.json').read_text())
    for rep in (2, 3):
        mine = json.loads((results/f'rp_v1/r{rep}/runs/RHx24_c4/{short}/repair.json').read_text())
        assert mine['config'] == rw['config'] and mine['config']['wall_seconds'] == 1200. and mine['run_id'] == f'rp{rep}_RHx24_c4_{short}' and mine['status'] in (fx.CERTIFIED, fx.NOT_CERTIFIED) and mine['fallback_rule_run'] is False and mine['oracle'] is False
        assert mine['source']['directory'].endswith(f'fd2_v1/runs/Hx24_first5/{short}') and mine['source']['tapered_u_sha256'] == rw['source']['tapered_u_sha256'] and mine['data']['keys_read'] == rw['data']['keys_read']
        metrics = json.loads((results/f'rp_v1/r{rep}/runs/RHx24_c4/{short}/repair_metrics.json').read_text())
        assert metrics['repaired_u_sha256'] == json.loads((results/f'rw_v1/runs/RHx24_c4/{short}/repair_metrics.json').read_text())['repaired_u_sha256']          # on this CPU, bit for bit
    assert set(out['seen']) >= {'pair.npz:fixed', 'pair.npz:moving'} and not any('seg' in name.split(':')[1] for name in out['seen'])
    identity = json.loads((results/f'rp_v1/id/{short}/identity_metrics.json').read_text())
    assert identity['certified_at_1'] is True and identity['oracle'] is False and identity['certificate']['attempts'][0]['config']['max_boxes'] == 255 and identity['metrics']['sdlogj_official'] == pytest.approx(0.) and identity['metrics']['seg35'] is not None and 'has no pin' in out['status_why']
    assert out['identity_usable'] == 1 and out['spread'] == decision.READ_SMALL and out['range'] == 0. and out['problems'] == [] and out['distinct'] == 1
    assert (out['reading'], out['identity_used']) in (('3 次都 19/19', 0), ('有失败 3/3', 3))          # the same repaired field 3 times on this CPU, certified or not; not certified, the identity takes its place
