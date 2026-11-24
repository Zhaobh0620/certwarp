"""HB diff review NB1 (decision_log 2026-11-10 ~5:47 AM CST): HB's entries under the production layout.  The units and the queues start `python -B <script>` with the code tree as the working
directory and PYTHONPATH <frozen>:<code tree> (hb_core.unit_argv, hb_core.queue_launch; parsed here from the start strings themselves), so sys.path[0] is the script's directory: only what the
entries put on sys.path themselves is importable -- tests/test_bayesreg/hb/_world.py adds experiments/xb for the tests, and hb_review.other_arms imported xb_core through that.  Every check runs in
a fresh process.  No real root is read: holdout_common's roots are the test sandbox's, set in the child before anything runs, and XB_TEST_SANDBOX makes them refuse a real one; no unit, no git
of the code tree, no pair list (hb_core.rows replaced where the queue would read it)."""
import ast
import json
import os
from pathlib import Path
import subprocess

import pytest

from . import _world as W

he, hc, hb, review = W.he, W.hc, W.hb, W.hb_review
REPO = W.ROOT
ENTRIES = ('experiments/hb/hb_review.py', 'experiments/hb/hb_method.py', 'experiments/hb/hb_configc.py', 'experiments/hb/hb_store.py', 'experiments/he/run_hx.py')
SELF_INSERTED = {'hi_core': 'holdout_common._hi puts experiments/hi on sys.path before it imports hi_core'}
PAIR = 'OASIS_OAS1_9998_MR1__OASIS_OAS1_9999_MR1'      # not a subject of any list: no file of it exists anywhere
ROOT_NAMES = ('RESULTS', 'HOLDOUT_ROOT', 'STAGE_ROOT', 'ATROPOS_ROOT', 'LOCK_ROOT', 'BATCH_ROOT', 'BIG_ROOT', 'DEV_STAGE', 'AUTH_ROOT')


def start_string(kind):
    """(cwd, env) of a start string: the unit's (kind 'unit', the review unit; its third certification's child inherits them) or the queue's (kind 'queue'; the lead runs withdraw the same way)."""
    if kind == 'unit':
        argv = hb.unit_argv('review', PAIR, [REPO/'experiments/hb/hb_review.py', 'unit'], code=REPO, unit_dir=Path('/nonexistent/unit'), res=dict(he.EVALUATION_UNIT), gpu=False,
                            log=Path('/nonexistent/unit.log'))
    else:
        argv = hb.queue_launch('review', ['queue'])
    cwd = next(a.split('=', 1)[1] for a in argv if a.startswith('--working-directory='))
    env = dict(a[len('--setenv='):].split('=', 1) for a in argv if a.startswith('--setenv='))
    assert env['PYTHONPATH'] == f'{he.core.FROZEN}:{REPO}' and Path(cwd) == REPO
    return cwd, dict(env, PATH=os.environ['PATH'], HOME=os.environ.get('HOME', '/tmp'), **{hc.TEST_SANDBOX_ENV: '1'})


def run_child(args, kind='queue', timeout=300):
    cwd, env = start_string(kind)
    return subprocess.run([str(hb.PY), '-B', *map(str, args)], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def no_import_error(done):
    text = done.stdout+done.stderr
    assert 'ModuleNotFoundError' not in text and 'ImportError' not in text, text[-3000:]


def test_the_other_arms_are_xb_s_and_configuration_c():
    """NB1: hb_review writes XB's arm names as a constant (no XB import in the review unit); they are xb_core.ARMS's."""
    import xb_core
    assert set(review.XB_ARMS) == set(xb_core.ARMS) and len(review.XB_ARMS) == len(set(review.XB_ARMS))
    assert review.other_arms() == tuple(sorted({hb.CONFIGC_ARM, *xb_core.ARMS})) and hb.ARM not in review.other_arms()
    assert 'xb_core' not in {n.names[0].name for n in ast.walk(ast.parse((REPO/'experiments/hb/hb_review.py').read_text())) if isinstance(n, ast.Import)}


@pytest.mark.parametrize('script', ENTRIES)
@pytest.mark.parametrize('kind', ['queue', 'unit'])
def test_each_entry_starts_under_the_production_layout(script, kind):
    """Every module an entry imports at its start is importable with the start string's PYTHONPATH and working directory (argparse's --help: the imports run, nothing else)."""
    done = run_child([REPO/script, '--help'], kind)
    no_import_error(done)
    assert done.returncode == 0 and 'usage' in done.stdout, done.stderr[-3000:]


PROBE = r"""
import ast, importlib, importlib.util, json, sys
from pathlib import Path
script, repo, inserted = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), json.loads(sys.argv[3])
sys.path[0] = str(script.parent)                                          # what `python -B <script>` puts first
importlib.import_module(script.stem)
loaded = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)})
mine = [p for p in loaded if p.suffix == '.py' and p.is_relative_to(repo/'experiments')]
lazy = {}
for path in mine:                                                         # the names imported inside the functions of every experiments module the entry loaded (absolute imports)
    names = set()
    for fn in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, ast.Import):
                    names |= {a.name.split('.')[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names.add(node.module.split('.')[0])
    lazy[str(path.relative_to(repo))] = sorted(names - set(sys.stdlib_module_names) - set(inserted))
missing = {rel: [n for n in names if importlib.util.find_spec(n) is None] for rel, names in lazy.items()}
import holdout_common
holdout_common._hi()
late = [n for n in inserted if importlib.util.find_spec(n) is None]
print(json.dumps(dict(scanned=sorted(lazy), names=sorted(set().union(*lazy.values())), missing={k: v for k, v in missing.items() if v}, late=late,
                      xb=importlib.util.find_spec('xb_core') is not None)))
"""


@pytest.mark.parametrize('script', ENTRIES)
def test_every_lazy_import_resolves_under_the_production_layout(script):
    """NB1's kind, for every entry: each name imported inside a function of every experiments module the entry loads is found on the production sys.path once the entry is imported (hi_core:
    after holdout_common._hi puts its directory there); experiments/xb is not on it (the layout is production's, not the tests')."""
    done = run_child(['-c', PROBE, REPO/script, REPO, json.dumps(sorted(SELF_INSERTED))])
    no_import_error(done)
    assert done.returncode == 0, done.stderr[-3000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert {script, 'experiments/holdout/holdout_common.py', 'experiments/he/he_core.py'} <= set(out['scanned']) and {'repair', 'rp_core', 'bayesreg'} <= set(out['names']), out['scanned']      # the scan sees he_core's lazy imports
    assert (out['missing'], out['late'], out['xb']) == ({}, [], False), out


DRIVER = '''
"""Runs hb_review.main in a fresh process with the production sys.path (sys.path[0] the script's directory, the rest from PYTHONPATH), holdout_common's roots the test's sandbox."""
import json, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
sys.path[0] = str(Path(spec['script']).parent)
import hb_core
for name, value in spec['roots'].items():
    setattr(hb_core.hc, name, Path(value))
hb_core.he.HOLDOUT_SOURCES = {k: Path(v) for k, v in spec['sources'].items()}
import hb_review
import hb_method
stubs = spec['stubs']
if 'start_checks' in stubs:
    hb_method.start_checks = lambda commit=None, code_run=None: (dict(code=dict(commit='0'*40)), [])
if 'rows' in stubs:
    hb_core.rows = lambda: [spec['pair']]
if 'context' in stubs:
    hb_review.R.read_context = lambda path: json.loads(Path(path).read_text())
    hb_review.R.cli_context_problem = lambda ctx, results=None: None


def reached(ctx, mode, **kwargs):
    Path(spec['reached']).write_text(json.dumps(dict(mode=mode.name, deny=list(mode.policy.deny))))
    return 0


hb_review.third_child_main = reached
hb_review.review_pair = reached
code = hb_review.main(spec['argv'])
import importlib.util
print(json.dumps(dict(code=code, xb=importlib.util.find_spec('xb_core') is not None)))
'''


@pytest.mark.parametrize('command', ['withdraw', 'queue', 'unit', 'third-cert'])
def test_the_review_reaches_its_hold_out_policy_under_the_production_layout(tmp_path, command):
    """NB1: hb_review.py, as the unit's start string starts it, gets past holdout_mode (the policy with the other arms) in each subcommand that builds it: withdraw is refused at its admission (no
    stop record), the queue at D110 (step 4 not ended; its batch record written), unit and third-cert reach the review and the child's entry (replaced here by a recorder of the policy)."""
    ctx_path = tmp_path/'context.json'
    cell = hb.cell(PAIR)
    if command in ('unit', 'third-cert'):
        cell.mkdir(parents=True)
        ctx = dict(pair=PAIR, pair_set='holdout', out=str(cell), stage_dir=str(hc.STAGE_ROOT))
        (cell/'context.json').write_text(json.dumps(ctx))
        ctx_path.write_text(json.dumps(ctx))
    argv = dict(withdraw=['withdraw', '--pair', PAIR, '--authorization', 'splits/bayesreg_v1/holdout_B_v1_hb_no_resume_1.json'], queue=['queue'],
                unit=['unit', '--pair', PAIR, '--batch', str(tmp_path/'batch')],
                **{'third-cert': ['third-cert', '--context', str(ctx_path), '--batch', str(tmp_path/'batch'), '--mode', 'holdout', '--root', str(hb.review_root())]})[command]
    stubs = dict(withdraw=['start_checks'], queue=['start_checks', 'rows'], unit=['context'], **{'third-cert': ['context']})[command]
    spec = dict(script=str(REPO/'experiments/hb/hb_review.py'), roots={k: str(getattr(hc, k)) for k in ROOT_NAMES}, sources={k: str(v) for k, v in he.HOLDOUT_SOURCES.items()},
                stubs=stubs, pair=PAIR, argv=argv, reached=str(tmp_path/'reached.json'))
    (tmp_path/'spec.json').write_text(json.dumps(spec))
    (tmp_path/'driver.py').write_text(DRIVER)
    done = run_child([tmp_path/'driver.py', tmp_path/'spec.json'], 'unit' if command in ('unit', 'third-cert') else 'queue')
    no_import_error(done)
    assert done.stdout.strip(), done.stderr[-3000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out['xb'] is False, out
    if command == 'withdraw':
        assert out['code'] == hb.EXIT['refused'] and 'no stop record on /home' in done.stderr, done.stderr[-2000:]
    elif command == 'queue':
        batch = he.read_json(next(Path(hc.HOLDOUT_ROOT).glob('batches/hb_review_*/batch.json')))
        assert out['code'] == hb.EXIT['refused'] and 'D110' in batch['stop']['reason'], batch.get('stop')
    else:
        reached = json.loads((tmp_path/'reached.json').read_text())
        assert out['code'] == 0 and reached['mode'] == 'holdout'
        for denied in (Path(hc.HOLDOUT_ROOT)/'fireants_greedy', Path(hc.HOLDOUT_ROOT)/'configc', hc.MAIN_REPO/'data', REPO/'data'):
            assert str(denied) in reached['deny'], denied


def test_the_third_cert_child_refused_before_its_spy_exits_its_own_code(tmp_path):
    """HB diff review n3: a context the child refuses (here: not an HE context) exits THIRD_REFUSED_EXIT, not 2 (argparse's) and not a crash; nothing of a pair opened."""
    bad = tmp_path/'context.json'
    bad.write_text(json.dumps(dict(pair=PAIR)))
    done = run_child([REPO/'experiments/hb/hb_review.py', 'third-cert', '--context', bad, '--batch', tmp_path, '--mode', 'holdout', '--root', tmp_path], 'unit')
    no_import_error(done)
    assert done.returncode == review.THIRD_REFUSED_EXIT and 'not an HE context' in done.stderr, done.stderr[-2000:]
