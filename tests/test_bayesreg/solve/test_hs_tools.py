"""HS (runs/20261106-hs/任务书.md, plan.md): experiments/hs/{hs_core,hs_seg,decide_hs,run_hs}.py.  CPU only, synthetic small volumes; no real image, label, prediction or field is read (the dry run reads headers and records, in its own process); nothing here starts a unit (a fake machine; the sandbox's systemd-run fails on top of that).

- the configuration: the arm, the repair, the certificate, the start strings are Hx24's (fd2_v1, RW) with only the segmenter id changed;
- the composer of (b): 1/2-1/2, the arithmetic of eval_uf.half_and_half, float16 with the argmax kept; (a) stored as v2 stores probs36;
- the registration: the staged document from UF's members (the real member check), (b)'s nnU-Net model, written once; the real reader of the Hx line on the predictions; the registry in FORMAL_REGISTRIES and nothing else in bayesreg/;
- one synthetic UF world (UF's own build, CPU): HS's inference of both candidates reproduces eval_uf.json's rows and counts, the control (segcheck), `register`;
- the reading's lines with their edges; the fallbacks of D74/D75 (each kind), the third certificate; no label in the fallback unit;
- the queues on a fake machine (等 UF, the order of the units, the fallback units only when needed); the dry run in its own process.
"""
import csv
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

import numpy as np
import pytest

from bayesreg.seg import structure_layer
from ..io_seg._helpers import synthetic_folds  # noqa: E402   (C23.9-13 (8), revision 6e)
from .test_d0_a4_tools import env, seg35  # noqa: F401
from . import test_d0_a4_tools as a4
from .test_ef_tools import world_of
from .test_fl_tools import metrics_of, write_json
from . import _import_refs as ir                 # 6o: the "not referred to" check on the AST (P0S)

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT/'experiments/hs/run_hs.py'
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUN = _load('_hs_run', SCRIPT)
hc, hs_seg, decision = RUN.hc, RUN.hs_seg, RUN.decision
fx, fxrun, core, fl, MC0, rp = RUN.fx, RUN.fxrun, RUN.core, RUN.fl, RUN.run, RUN.rp
fd2 = fxrun.fd2
HS, ALL = hc.HS, hc.ALL_PAIRS
P0, R0 = core.PAIRS[0], fl.REST_PAIRS[0]
EVAL_UF = _load('_hs_eval_uf', ROOT/'experiments/uf/eval_uf.py')


@pytest.fixture(autouse=True)
def no_real_machine(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('the real System.run was called')
    monkeypatch.setattr(MC0.System, 'run', forbidden)


HS_STATE = dict(HS.__dict__)                    # the batch object as imported (revision 1b, the follow-up's minor 5)


@pytest.fixture(autouse=True)
def hs_as_imported():
    """After every test the attributes of the HS instance are what they were at import: a monkeypatch on the instance leaves the bound method there as an instance attribute, which would hide any later patch of the class."""
    yield
    HS.__dict__.clear()
    HS.__dict__.update(HS_STATE)


REAL_DATA = tuple(str(p) for p in {core.MAIN_REPO/'data', (core.MAIN_REPO/'data').resolve(), core.STAGE, Path(core.STAGE).resolve()})


@pytest.fixture(autouse=True)
def no_real_stage(monkeypatch):
    """load_pair, the firewall's manifest reader and np.load fail on any file under the real data/ (the stage of the 19 pairs): a test that reaches real data fails instead of reading it (revision 1b, the follow-up's minor 5; the manifest reader: HSF, now that the tests use the real one)."""
    import bayesreg.io.pair as pair_module
    real_load_pair, real_np_load, real_read_manifest = pair_module.load_pair, np.load, pair_module._read_manifest

    def real(path):
        try:
            text = str(Path(os.fspath(path)).resolve())
        except TypeError:
            return False
        return any(text == r or text.startswith(r.rstrip('/')+'/') for r in REAL_DATA)

    def load_pair(path, *args, **kwargs):
        if real(path):
            raise AssertionError(f'a test opened the real stage: {path}')
        return real_load_pair(path, *args, **kwargs)

    def np_load(path, *args, **kwargs):
        if real(path):
            raise AssertionError(f'a test opened a real data file: {path}')
        return real_np_load(path, *args, **kwargs)

    def read_manifest(path):
        if real(path):
            raise AssertionError(f'a test opened a real manifest: {path}')
        return real_read_manifest(path)
    monkeypatch.setattr(pair_module, 'load_pair', load_pair)
    monkeypatch.setattr(np, 'load', np_load)
    monkeypatch.setattr(pair_module, '_read_manifest', read_manifest)


def test_the_guards_of_the_tests_hold(results):
    import bayesreg.io.pair as pair_module
    with pytest.raises(AssertionError, match='real stage'):
        pair_module.load_pair(core.STAGE/ALL[-1]/'pair.npz')
    with pytest.raises(AssertionError, match='real data file'):
        np.load(core.STAGE/ALL[-1]/'pair.npz')
    with pytest.raises(AssertionError, match='real manifest'):
        pair_module._read_manifest(core.STAGE/ALL[-1]/'manifest.json')
    with pytest.raises(AssertionError, match='real manifest'):
        RUN.prediction_problems(lambda text: None, 'UF4')                       # its default stage is the real one (the seg records: the temporary results root)
    HS.read_images = lambda cell: None
    assert 'read_images' in HS.__dict__                                       # hs_as_imported takes it away after this test


# ---- the configuration is Hx24's, only the segmenter changes ------------------------------------------------------------------------------------------
def test_the_arm_is_the_frozen_hx24_of_fd2_and_rw():
    spec = HS.ARMS[hc.HS_ARM]
    assert spec is fx.SPEC and {k: spec[k] for k in ('k', 'sigma_field', 'native', 'checkpoints', 'channels', 'weight', 'D_factor')} == {k: fd2.ARMS['Hx24'][k] for k in ('k', 'sigma_field', 'native', 'checkpoints', 'channels', 'weight', 'D_factor')}
    for pair in ALL:
        mine, theirs = HS.field_cell(pair), fd2.Cell('Hx24', fd2.FIRST if pair in core.PAIRS else 'rest600', pair)
        assert HS.arm_config(mine, HS.sigma_of(mine, None)) == fd2.arm_config(theirs, fd2.sigma_of(theirs, None))
        assert HS.checkpoints_of(mine) == fd2.checkpoints_of(theirs) and HS.steps_of(HS.repair_cell(pair)) == fd2.steps_of(fd2.Cell('RHx24', 'c600' if pair in core.PAIRS else 'rest600', pair)) == 600


def test_the_repair_is_rws_rf05_with_the_fuse():
    mine, rws = RUN.BASE.repair.RepairConfig(eps=fl.REPAIR_EPS), fxrun.RW_BASE.repair.RepairConfig(eps=fl.REPAIR_EPS)
    assert mine == rws and mine.wall_seconds == fx.FUSE_SECONDS == 1200.
    assert RUN.BASE.run.certificate_config() == MC0.certificate_config() and rp.CERTIFICATE_C == dict(delta=0., max_depth=48, max_boxes=255, tile_size=65536)


def test_the_unit_strings_are_nxs_but_for_the_names():
    def props(argv):
        return [a for a in argv if a.startswith(('--property=MemoryMax', '--property=MemorySwapMax', '--property=CPUQuota', '--property=RuntimeMaxSec', '--nice')) or (a.startswith('--setenv=') and 'PYTHONPATH' not in a)]
    for pair in (P0, R0):
        for mine, theirs in ((HS.field_cell(pair), fx.NX.Cell(fx.NX_ARM, fx.NX.FIRST if pair in core.PAIRS else 'rest600', pair)), (HS.repair_cell(pair), fx.NX.Cell(fx.NX.REPAIR_ARM, 'c600' if pair in core.PAIRS else 'rest600', pair))):
            a, b = RUN.BASE.run_argv(mine, Path('/c'), HS.DEFAULT_ROOT, 's'), fxrun.NX_BASE.run_argv(theirs, Path('/c'), fx.NX.DEFAULT_ROOT, 's')
            assert props(a) == props(b) and a[a.index('-B')+2:a.index('-B')+3] == b[b.index('-B')+3:b.index('-B')+4]     # the command name: run-one / repair-one
            assert f'--unit={mine.unit}' in a and mine.unit.startswith('tag-hs-') and 'py'+'test' not in ' '.join(a)


def test_the_cells_and_the_plan():
    assert [c.pair for c in HS.plan_all_fields()] == list(ALL) == [c.pair for c in HS.plan_all_repairs()]
    assert HS.all_cells(None) == HS.plan_all_fields()+HS.plan_all_repairs() and len(HS.all_cells(None)) == 38
    assert HS.repair_cells(HS.chosen_repairs(None)) == [HS.repair_cell(p) for p in core.PAIRS] and HS.rest_cells(None) == ([HS.field_cell(p) for p in fl.REST_PAIRS], [HS.repair_cell(p) for p in fl.REST_PAIRS])
    with pytest.raises(SystemExit, match='HS has no cell'):
        HS.require_cell('SHx24', 'rest600', P0)
    with pytest.raises(SystemExit, match='HS has no cell'):
        HS.require_cell('RHx24', 'c600', P0)
    assert HS.root_problem(HS.DEFAULT_ROOT) is None and 'writes only' in HS.root_problem(HS.DEFAULT_ROOT.parent/'rw_v1')


def added_files():
    """The files bayesreg/ may hold only as files added since ace0903, each a relaxation stated as one (the drift test below and the 6o tests read this one set: P0S revision 2,
    the code review's minor 9)."""
    # EV (C23.12, D106: the hold-out evaluation entry bayesreg.eval.holdout and its decision code), a relaxation stated as one and handed to the lead: these
    # two files may appear only as files added after ace0903 (git status A, never a change of an existing file), and no other file of bayesreg/ or of HS
    # (experiments/hs/) refers to them, so HS's code path cannot reach them.
    ev_new = {'bayesreg/eval/holdout.py', 'bayesreg/eval/holdout_decide.py'}
    # HB (the lead's item 14 of HB's revision round; decision_log 2026-11-10 N3 and ~2:59 AM CST (7)): bayesreg/io/code_identity.py, the code identity shared by the evaluation entry and the
    # hold-out tools; the same relaxation as EV's, stated as one and handed to the lead: an added file only, referred to by no other file of bayesreg/ (EV's holdout.py may) or of experiments/hs/.
    ev_new |= {'bayesreg/io/code_identity.py'}
    # A5b (item 6 of runs/20261110-a5/任务书.md; decision_log 2026-11-10 ~2:45 AM CST N1): bayesreg/io/rejudgment.py, the one reader of the rejudgment records moved from holdout_common so
    # that the evaluation entry can call it.  A relaxation of this test, stated as one and handed to the lead for the contract (the precedents: C23 revision 6i, EV's two files; 6k, HB's
    # code_identity.py): an added file only (git status A since ace0903, the assert below), referred to by no other file of bayesreg/ (EV's holdout.py may) or of experiments/hs/ (the
    # check on the AST below; it was a word of the old list), so the HS path cannot reach it.
    ev_new |= {'bayesreg/io/rejudgment.py'}
    # R2 (runs/20261110-r2; contract C23 revision 6m, for BZ; R2 review I1 and the lead's ruling of 11-10 ~10:59 PM CST): the recipe r2_small, a relaxation stated as one.
    # - bayesreg/seg/unet/small.py: an added file only, as 6k and 6l.  Unlike those, it is referred to inside bayesreg/ (cli.py imports it in the train branch of --profile r2_small), so its
    #   "referred to by no other file" is narrowed to experiments/hs/ and, inside bayesreg/, to cli.py alone (the check on the AST below, 6o).
    # - large.py (the hooks patch_fn, loss_fn, recipe of train_large_member, defaults the calls before them: tests/test_bayesreg/r2/test_r2_recipe.py) and cli.py (the profile r2_small; v1, v2_full,
    #   v3_large unchanged: tests/test_bayesreg/segnet/test_ul.py) are changed files: let through only at the bytes R2's review read, their sha256 pinned here.  Any later change of
    #   either fails this test and needs a relaxation of its own.  HS's path (hs_seg: infer.load_member, predict_logits, ensemble_probabilities; HI: large.cap_gpu_memory, unchanged) reaches none of the three.
    ev_new |= {'bayesreg/seg/unet/small.py'}
    # R3b (runs/20261111-r3; contract revision 10 draft C22.1-8, decode switch default off; the lead's ruling of 11-11 ~3:57 PM CST: R3b's wording of this relaxation):
    # bayesreg/seg/decode.py, the expected-Dice decoding of the small structures, a relaxation stated as one and handed to the lead, as 6k and 6l: an added file only
    # (git status A since ace0903, the assert below), referred to by no other file of bayesreg/ or of experiments/hs/ (the check on the AST below, 6o), so neither
    # bayesreg's own paths nor HS's path can reach it; its callers are experiments/r3 (R3b's calibration tools) and R3a's composition behind the decode switch.
    ev_new |= {'bayesreg/seg/decode.py'}
    # T24 (contract C24.12, revisions 9d, 9d-1, 9d-2; merged into main by 52de5375, 2026-11-19 ~12:37 PM CST; decision_log line 527; contract C23 revision 6p):
    # bayesreg/eval/holdout_tet24.py, the 24-tet proxy's post-hoc wrapper, a relaxation stated as one, as 6k, 6l and 6o: an added file only (git status A since ace0903, the assert
    # below), referred to by no other file of bayesreg/ or of experiments/hs/ (the check on the AST below, 6o), so neither bayesreg's own paths nor HS's path can reach it.  It imports
    # EV's holdout.py and holdout_decide.py and HB's code_identity.py (C24.12-2: the evaluation functions it calls are P1's) and runs only from its own checkout (C24.12-13), so the
    # closure treats it as EV's holdout.py: an added file that refers to the others without being reached itself.
    ev_new |= {'bayesreg/eval/holdout_tet24.py'}
    # LP-C2 (contract C26, revisions 11-11f; merged into main by 7e99377b, 2026-11-20 ~3:23 AM CST; decision_log line 690; contract C23 revision 6q, the p1-repin review's B1):
    # bayesreg/eval/lpba_eval.py, the LPBA40 evaluation entry (C26.7), and bayesreg/eval/lpba_decide.py, its decision module (D146, C26.5-8), a relaxation stated as one, as 6k, 6l, 6o
    # and 6p: added files only (git status A since ace0903, the assert below), referred to by no file of bayesreg/ that is not itself an added file, nor by experiments/hs/ (the check
    # on the AST below, 6o), so neither bayesreg's own paths nor HS's path can reach them.  lpba_eval.py imports holdout.py, holdout_tet24.py (C26.7-5), rejudgment.py and
    # code_identity.py; lpba_decide.py imports holdout_decide.py (D146: reused, not copied); the closure treats both as EV's holdout.py.  6q also corrects 6p: holdout_tet24.py is
    # referred to by the added file lpba_eval.py only (C26.7-5), not by no file at all.
    ev_new |= {'bayesreg/eval/lpba_eval.py', 'bayesreg/eval/lpba_decide.py'}
    return frozenset(ev_new)


def test_the_only_change_in_bayesreg_is_one_registry_file_more():
    assert structure_layer.FORMAL_REGISTRIES[:3] == (structure_layer.REPO_REGISTRY, structure_layer.REPO/'configs/bayesreg/segmenters_v2.json', structure_layer.REPO/'configs/bayesreg/segmenters_nn935_v1.json')
    assert structure_layer.FORMAL_REGISTRIES[3:] == (structure_layer.REPO/hc.REGISTRY_REL, structure_layer.REPO/'configs/bayesreg/segmenters_r2_v1.json') and structure_layer.SINGLE_MODEL_SEGMENTERS == ('nn935_v1',)     # R2 (runs/20261110-r2/plan.md) appended its registry after HS's; the tail pinned whole
    changed = subprocess.run(['git', 'diff', '--name-only', 'ace0903', '--', 'bayesreg', 'jd_curl'], cwd=ROOT, capture_output=True, text=True).stdout.split()
    # C23.9-13 (10), revision 6e: a relaxation, stated as one.  Besides HS's registry change (structure_layer.py), exactly SG's files of bayesreg/ (contract C23),
    # each by name, and nothing in jd_curl/.  HS's behaviour is guarded bit for bit elsewhere: tests/test_bayesreg/hi/test_hs_a_bitwise.py (hs_a's inference on
    # the hold-out path against c027447), tests/test_bayesreg/sg/test_sg_behaviour.py (the guard against c027447's on every non-sealed id), this file's
    # HS tests (the channels of HS = Hx24's, the reader on HS's predictions), and the MEM goldens of run_pair (solve/test_mem_bitwise.py, test_mem_paths_bitwise.py).
    sg_files = {'bayesreg/cli/run_pair.py', 'bayesreg/eval/cli.py', 'bayesreg/eval/truth.py', 'bayesreg/io/guard.py', 'bayesreg/io/pair.py', 'bayesreg/io/sealed.py',
                'bayesreg/io/unseal.py', 'bayesreg/seg/structure_layer.py', 'bayesreg/seg/unet/data.py', 'bayesreg/seg/unet/infer.py', 'bayesreg/seg/unet/synthetic.py'}
    ev_new = set(added_files())                  # the relaxations, each stated as one: added_files() above
    # 6m (the R2 note in added_files()): large.py and cli.py are let through only at the bytes R2's review read
    r2_pinned = {'bayesreg/seg/unet/large.py': 'c6520d62b9fd0bdd5e7bbd9b087039e05385983bdf1bad95e9bdbe8e524574d5', 'bayesreg/seg/unet/cli.py': 'fe74a2b8f57c9de2d9fda0f2fe330fcef50d471da1c91b3b32ac55c816a34ddb'}
    r2_changed = {rel for rel, sha in r2_pinned.items() if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == sha}
    assert r2_changed == set(r2_pinned), sorted(set(r2_pinned)-r2_changed)          # a change of large.py or cli.py after 6m: not let through
    status = dict(line.split('\t', 1)[::-1] for line in subprocess.run(['git', 'diff', '--name-status', 'ace0903', '--', 'bayesreg'], cwd=ROOT, capture_output=True,
                                                                       text=True).stdout.splitlines() if line.strip())
    assert all(status.get(p) == 'A' for p in ev_new & set(changed)), {p: status.get(p) for p in ev_new & set(changed)}
    # 6o (contract C23 revision 6o, its AST item; the lead's ruling 11 of 11-14, P0S): "referred to" is read on the AST (_import_refs.py: every import statement, relative ones resolved by the
    # interpreter's rule, several names, parentheses over lines, inside functions and try blocks; the dynamic imports; and, so that nothing the old word list caught is lost, dotted attribute
    # chains and dotted names or paths inside strings), in place of the word list (review m14; R3b), and closed over the added files (R3b difference review n1: the word list did not read them).
    # - experiments/hs/ reaches none of the added files, small.py included (6m: narrowed from "no other file", cli.py imports it), nor the name train_small_member;
    # - the other files of bayesreg/ reach exactly one added file, small.py, from bayesreg/seg/unet/cli.py alone (6m), and small.py reaches no other added file.
    assert ir.added_reached(ROOT, sorted(ev_new), tops=('experiments/hs',), identifiers=('train_small_member',)) == {}
    reached = ir.added_reached(ROOT, sorted(ev_new), tops=('bayesreg',))
    assert reached == {'bayesreg/seg/unet/small.py': ['bayesreg/seg/unet/cli.py']}, reached
    changed = [p for p in changed if p not in ev_new]
    assert not [p for p in changed if p.startswith('jd_curl/')], changed
    assert changed in ([], ['bayesreg/seg/structure_layer.py'], sorted(sg_files), sorted(sg_files | r2_changed)), sorted(set(changed) ^ (sg_files | r2_changed))


# ---- 6o: the AST check against mutations (P0S; contract C23 revision 6o; R3b difference review n1) ---------------------------------------------
ADDED = tuple(sorted(added_files()))           # the same source as the drift test's ev_new (P0S revision 2, the code review's minor 9)
OLD_WORDS = ('holdout_decide', 'eval.holdout', 'from . import holdout', 'from .holdout import', 'code_identity', 'rejudgment', 'seg.decode', 'seg import decode', 'from .decode',
             'from . import decode')                    # the word list 6o replaced (review m14; R3b), kept here only to show what it missed
OLD_SMALL_WORDS = ('unet.small', 'unet import small', 'train_small_member')     # 6m's word list for experiments/hs/, the same
OLD_MISSED = 7                                          # FORMS[:7]: R3b difference review n1's four forms and the code review's I1 three
DECODE, SMALL = 'bayesreg/seg/decode.py', 'bayesreg/seg/unet/small.py'
FORMS = [                                               # (file, text, what it reaches); the first OLD_MISSED (n1's four, the code review's I1 three) the word list missed
    ('bayesreg/seg/unet/m.py', 'from .. import decode\n', DECODE),
    ('bayesreg/seg/unet/m.py', 'from ..decode import decode_map\n', DECODE),
    ('bayesreg/seg/m.py', 'from . import structure_layer, decode\n', DECODE),
    ('bayesreg/seg/m.py', 'from bayesreg.seg import (\n    structure_layer,\n    decode as d,\n)\n', DECODE),
    ('bayesreg/m.py', "import importlib\nx = 'decode'\nM = importlib.import_module(f'bayesreg.seg.{x}')\n", 'dynamic:<dynamic>'),       # P0S revision 2, the code review's I1
    ('bayesreg/m.py', "import importlib\nx = 'decode'\nM = importlib.import_module('bayesreg.seg.' + x)\n", 'dynamic:<dynamic>'),
    ('bayesreg/m.py', "import importlib\nP = 'bayesreg.seg.'\nM = importlib.import_module(P + 'decode')\n", 'dynamic:<dynamic>'),
    ('bayesreg/m.py', "import importlib\nM = importlib.import_module('.decode', package=P)\n", 'dynamic:<dynamic>'),
    ('bayesreg/m.py', "M = __import__('bayesreg.io', fromlist=['pair', 'rejudgment'])\n", 'bayesreg/io/rejudgment.py'),
    ('bayesreg/m.py', "M = __import__('bayesreg.io', fromlist=names)\n", 'dynamic:<dynamic>'),
    ('bayesreg/seg/unet/m.py', "M = __import__('decode', globals(), None, [], 2)\n", DECODE),
    ('bayesreg/m.py', "M = __import__('x', globals(), None, [], 3)\n", 'dynamic:<unresolvable relative import>'),
    ('bayesreg/seg/m.py', "M = __import__('decode', globals(), None, ['decode_map'], 1)\n", DECODE),
    ('bayesreg/m.py', "import importlib.util\nS = importlib.util.spec_from_file_location('x', ROOT/'bayesreg'/'seg'/name)\n", 'dynamic:<dynamic>'),
    ('bayesreg/m.py', "import runpy\nrunpy.run_path(str(D/'decode.py'))\n", DECODE),
    ('bayesreg/m.py', "from importlib.machinery import SourceFileLoader\nL = SourceFileLoader('m', f'{D}/code_identity.py')\n", 'bayesreg/io/code_identity.py'),
    ('bayesreg/m.py', "import importlib\nM = importlib.reload(module)\n", 'dynamic:<dynamic>'),
    ('bayesreg/eval/m.py', 'from . import (cli,\n              holdout_decide)\n', 'bayesreg/eval/holdout_decide.py'),
    ('bayesreg/cli/m.py', 'from ..io import code_identity\n', 'bayesreg/io/code_identity.py'),
    ('bayesreg/io/m.py', 'from . import pair, rejudgment\n', 'bayesreg/io/rejudgment.py'),
    ('bayesreg/eval/m.py', 'def f():\n    from .holdout import read_truth\n    return read_truth\n', 'bayesreg/eval/holdout.py'),
    ('bayesreg/eval/m.py', 'try:\n    from ..seg import decode\nexcept ImportError:\n    decode = None\n', DECODE),
    ('bayesreg/m.py', 'import bayesreg.seg.decode as d\n', DECODE),
    ('bayesreg/m.py', 'import os, bayesreg.io.rejudgment\n', 'bayesreg/io/rejudgment.py'),
    ('bayesreg/m.py', "import importlib\nM = importlib.import_module('bayesreg.seg.decode')\n", DECODE),
    ('bayesreg/m.py', "import importlib\nM = importlib.import_module('.decode', 'bayesreg.seg')\n", DECODE),
    ('bayesreg/m.py', "import importlib\nM = importlib.import_module('.holdout', package='bayesreg.eval')\n", 'bayesreg/eval/holdout.py'),
    ('bayesreg/m.py', "M = __import__('bayesreg.eval.holdout_decide')\n", 'bayesreg/eval/holdout_decide.py'),
    ('bayesreg/m.py', "import importlib.util\nS = importlib.util.spec_from_file_location('x', ROOT/'bayesreg'/'seg'/'decode.py')\n", DECODE),
    ('bayesreg/m.py', "import importlib\nname = 'x'\nM = importlib.import_module(name)\n", 'dynamic:<dynamic>'),
    ('bayesreg/seg/unet/m.py', 'from .... import x\n', 'import:<unresolvable relative import>'),
    ('bayesreg/m.py', 'import bayesreg\nF = bayesreg.seg.decode.decode_map\n', DECODE),
    ('bayesreg/m.py', "NAME = 'bayesreg.io.code_identity'\nPATH = f'{ROOT}/bayesreg/seg/decode.py'\n", 'bayesreg/io/code_identity.py'),
    ('experiments/hs/x.py', 'from bayesreg.seg.decode import decode_map\n', DECODE),
    ('experiments/hs/x.py', 'from bayesreg.seg.unet import (large,\n    small)\n', SMALL),
    ('experiments/hs/x.py', 'from bayesreg.seg.unet.cli import train_small_member\n', 'identifier:train_small_member'),
    # P0 review (docs/bayesreg/reviews/P0_复核_20261115.md) minor 5: a by-path import whose last constant is only the suffix '.py' (the file name empty, nothing reached, no '<dynamic>');
    # its three forms, then a name that only ends in a constant (the same hole: '_decide' reached nothing), then the constant spellings, which still resolve
    ('bayesreg/m.py', "import importlib.util\nS = importlib.util.spec_from_file_location('m', ROOT/'bayesreg'/'seg'/f'{name}.py')\n", 'dynamic:<dynamic>'),
    ('bayesreg/m.py', "from importlib.machinery import SourceFileLoader\nL = SourceFileLoader('m', (D/name).with_suffix('.py'))\n", 'dynamic:<dynamic>'),
    ('bayesreg/m.py', "import runpy\nrunpy.run_path(str(D/name) + '.py')\n", 'dynamic:<dynamic>'),
    ('bayesreg/m.py', "import runpy\nrunpy.run_path(f'{D}/{x}_decide.py')\n", 'dynamic:<dynamic>'),
    ('bayesreg/m.py', "import runpy\nrunpy.run_path(str(D/x) + '_decide.py')\n", 'dynamic:<dynamic>'),          # T1 review minor 1: pins "the left side of + must end with a separator"
    ('bayesreg/m.py', "import runpy\nrunpy.run_path(D/'.py')\n", 'dynamic:<dynamic>'),
    ('bayesreg/m.py', "import importlib.util\nS = importlib.util.spec_from_file_location('m', ROOT/'bayesreg'/'seg'/f'decode.py')\n", DECODE),
    ('bayesreg/m.py', "from importlib.machinery import SourceFileLoader\nL = SourceFileLoader('m', (D/'decode').with_suffix('.py'))\n", DECODE),
    ('bayesreg/m.py', "from importlib.machinery import SourceFileLoader\nL = SourceFileLoader('m', (D/'decode.txt').with_suffix('.py'))\n", DECODE),     # T1 review minor 2: the suffix replaced
    ('bayesreg/m.py', "import runpy\nrunpy.run_path(str(D) + '/rejudgment.py')\n", 'bayesreg/io/rejudgment.py'),
    ('bayesreg/m.py', "import runpy\nrunpy.run_path('bayesreg/eval/' + 'holdout_decide.py')\n", 'bayesreg/eval/holdout_decide.py'),
    ('bayesreg/eval/m.py', 'from . import holdout_tet24\n', 'bayesreg/eval/holdout_tet24.py'),          # 6p (T24): the added file is reported once referred to
    ('bayesreg/m.py', 'from bayesreg.eval.lpba_eval import main\n', 'bayesreg/eval/lpba_eval.py'),          # 6q (LP-C2)
]
NOT_FORMS = [                                           # what is not a reference: other modules with near names, a module's own name, docstrings and comments, a star import of the package
    ('bayesreg/seg/m.py', 'from . import structure_layer\nfrom bayesreg.seg import channels\n'),
    ('bayesreg/seg/m.py', 'import bayesreg.seg.decoder\nfrom bayesreg.seg.decoders import decode\n'),
    ('bayesreg/eval/m.py', 'from bayesreg.eval import holdout_report\nfrom .cli import main\n'),
    ('bayesreg/m.py', '"""from .decode import x; bayesreg.seg.decode is not imported here."""\n# from bayesreg.seg import decode\ndecode = 1\n'),
    ('bayesreg/m.py', 'from bayesreg.seg import *\n'),
    ('experiments/hs/x.py', 'from bayesreg.seg.unet import large\nimport holdout_common\n'),
    ('experiments/hs/x.py', "import importlib.util\nS = importlib.util.spec_from_file_location('_hs_run_fl', EXPERIMENTS/'fl'/'run_fl.py')\n"),     # run_hs.py:65's form
    ('experiments/hs/x.py', "import importlib.util\nS = importlib.util.spec_from_file_location('_hs_run_fl', (EXPERIMENTS/'fl'/'run_fl').with_suffix('.py'))\n"),     # P0 review minor 5
    ('bayesreg/m.py', "import importlib\nM = importlib.import_module('bayesreg.seg.channels', package=None)\nN = __import__('bayesreg.io', fromlist=None)\n"),
]


def small_tree(tmp_path):
    root = tmp_path/'tree'
    for d in ('bayesreg', 'bayesreg/eval', 'bayesreg/io', 'bayesreg/seg', 'bayesreg/seg/unet', 'bayesreg/cli'):
        (root/d).mkdir(parents=True, exist_ok=True)
        (root/d/'__init__.py').write_text('')
    (root/'experiments'/'hs').mkdir(parents=True)
    for rel in ADDED:
        (root/rel).write_text('"""an added file"""\n')
    return root


def reached_from(root, rel):
    top = 'experiments/hs' if rel.startswith('experiments/hs/') else 'bayesreg'
    return ir.added_reached(root, ADDED, tops=(top,), identifiers=('train_small_member',) if top == 'experiments/hs' else ())


@pytest.mark.parametrize('n', range(len(FORMS)))
def test_6o_every_import_form_is_read(n, tmp_path):
    """6o: each form, alone in a file of a synthetic tree, is reported with its file; the first OLD_MISSED (n1's four, the code review's I1 three) were missed by the old word list."""
    root = small_tree(tmp_path)
    rel, text, want = FORMS[n]
    assert reached_from(root, rel) == {}
    (root/rel).write_text(text)
    got = reached_from(root, rel)
    assert want in got and got[want] == [rel], got
    if n < OLD_MISSED:
        assert not any(word in text for word in OLD_WORDS)
    (root/rel).unlink()
    assert reached_from(root, rel) == {}


@pytest.mark.parametrize('n', range(len(NOT_FORMS)))
def test_6o_what_is_not_a_reference(n, tmp_path):
    root = small_tree(tmp_path)
    rel, text = NOT_FORMS[n]
    (root/rel).write_text(text)
    assert reached_from(root, rel) == {}


def test_6o_the_check_is_closed_over_the_added_files(tmp_path):
    """R3b difference review n1: an added file reached (small.py, by cli.py) that imports another added file makes that one reached too; one not reached may refer to the others (EV's
    holdout.py does) without making them reached."""
    root = small_tree(tmp_path)
    (root/'bayesreg/seg/unet/cli.py').write_text('def train():\n    from . import small\n    return small\n')
    assert ir.added_reached(root, ADDED, tops=('bayesreg',)) == {SMALL: ['bayesreg/seg/unet/cli.py']}
    (root/'bayesreg/eval/holdout.py').write_text('from ..io import code_identity, rejudgment\nfrom ..seg import decode\n')
    assert ir.added_reached(root, ADDED, tops=('bayesreg',)) == {SMALL: ['bayesreg/seg/unet/cli.py']}
    (root/SMALL).write_text('from ..decode import decode_map\n')
    assert ir.added_reached(root, ADDED, tops=('bayesreg',)) == {SMALL: ['bayesreg/seg/unet/cli.py'], DECODE: [SMALL]}
    (root/DECODE).write_text('from ..eval import holdout_decide\n')
    assert ir.added_reached(root, ADDED, tops=('bayesreg',)) == {SMALL: ['bayesreg/seg/unet/cli.py'], DECODE: [SMALL], 'bayesreg/eval/holdout_decide.py': [DECODE]}


REAL_MUTATIONS = [                                      # (file, appended text, (added file or key, referrer) reported, the old word lists missed it): injected into a copy of this tree
    ('bayesreg/eval/cli.py', '\n\ndef _p0s_mutation():\n    from ..seg import (structure_layer,\n                       decode)\n    return decode\n', (DECODE, 'bayesreg/eval/cli.py'), True),
    ('bayesreg/seg/unet/infer.py', '\nfrom .. import decode  # noqa\n', (DECODE, 'bayesreg/seg/unet/infer.py'), True),
    ('bayesreg/io/guard.py', '\nfrom . import pair, rejudgment  # noqa\n', ('bayesreg/io/rejudgment.py', 'bayesreg/io/guard.py'), False),
    ('experiments/hs/hs_seg.py', '\nfrom bayesreg.seg.unet import (large,\n                               small)  # noqa\n', (SMALL, 'experiments/hs/hs_seg.py'), True),
    (SMALL, '\nfrom ..decode import decode_map  # noqa\n', (DECODE, SMALL), True),          # the word list did not read the added files
    ('bayesreg/eval/cli.py', '\nfrom . import holdout_tet24  # noqa\n', ('bayesreg/eval/holdout_tet24.py', 'bayesreg/eval/cli.py'), False),     # 6p (T24); 'from . import holdout' was a word
    ('bayesreg/eval/truth.py', '\nfrom . import lpba_eval  # noqa\n', ('bayesreg/eval/lpba_eval.py', 'bayesreg/eval/truth.py'), True),          # 6q (LP-C2); no old word caught it
]


@pytest.mark.parametrize('n', range(len(REAL_MUTATIONS)))
def test_6o_mutations_of_this_tree_are_caught(n, tmp_path):
    """6o on a copy of this tree's bayesreg/ and experiments/hs/ (*.py, byte for byte): the copy reads as the tree does (the drift test's two assertions); each injected import is reported with its
    file, and which of them the old word lists would have let through is pinned (n1's forms, and an import in an added file, which the word list did not read)."""
    copy = tmp_path/'copy'
    for top in ('bayesreg', 'experiments/hs'):
        for path in sorted((ROOT/top).rglob('*.py')):
            if '__pycache__' not in path.parts:
                target = copy/path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
    clean = {'bayesreg': {SMALL: ['bayesreg/seg/unet/cli.py']}, 'experiments/hs': {}}
    for top, want in clean.items():
        assert ir.added_reached(copy, ADDED, tops=(top,), identifiers=('train_small_member',) if top == 'experiments/hs' else ()) == want
    rel, text, (key, referrer), old_missed = REAL_MUTATIONS[n]
    (copy/rel).write_text((copy/rel).read_text(encoding='utf-8')+text, encoding='utf-8')
    top = 'experiments/hs' if rel.startswith('experiments/hs/') else 'bayesreg'
    got = ir.added_reached(copy, ADDED, tops=(top,), identifiers=('train_small_member',) if top == 'experiments/hs' else ())
    assert got != clean[top] and referrer in got.get(key, []), got
    old_caught = rel not in ADDED and any(word in text for word in OLD_WORDS+(OLD_SMALL_WORDS if top == 'experiments/hs' else ()))
    assert old_caught is not old_missed


def test_the_candidates_ids():
    assert hc.model_ids('UF4') == dict(A='unet35_v4_ens_foldA', B='unet35_v4_ens_foldB', nn=None)
    assert hc.model_ids('UF4_nn') == dict(A='uf_nn_v1_foldA', B='uf_nn_v1_foldB', nn='uf_nn_v1_nnall') and hc.CANDIDATES['UF4_nn']['nn_fold'] == 2
    from bayesreg.io.pair import FORBIDDEN_ID
    assert not any(FORBIDDEN_ID.search(i) for c in hc.CANDIDATES for i in [hc.CANDIDATES[c]['segmenter_id'], *[v for v in hc.model_ids(c).values() if v]])
    assert hc.seg_dir('UF4', Path('/r')) == Path('/r/seg/unet35_v4_ens') and hc.seg_dir('UF4_nn', Path('/r')) == Path('/r/seg/uf_nn_v1') and hc.UF_OUT.name == 'unet35_v4_full'


# ---- UF's reading -------------------------------------------------------------------------------------------------------------------------------
def uf_root(tmp_path, reading='worth_hx24', candidate='UF4', evaluation=True):
    root = tmp_path/'unet35_v4_full'
    if reading is not None:
        write_json(root/hc.UF_DECISION_REL, dict(reading=reading, reading_text='x', candidate=candidate if reading == 'worth_hx24' else None, primary_arm=candidate, delta_seg35=.004, delta_seg4=.001))
    if evaluation:
        write_json(root/hc.UF_EVAL_REL, dict(subjects=[]))
    return root


@pytest.mark.parametrize('reading,candidate,word', ((None, None, '等 UF'), ('not_connected', None, 'HS 不跑'), ('not_decidable', None, 'HS 不跑'), ('worth_hx24', 'UF4', None), ('worth_hx24', 'UF4_nn', None), ('worth_hx24', 'v1', '不是')))
def test_uf_reading(tmp_path, reading, candidate, word):
    waiting, stop, got, _ = hc.uf_reading(uf_root(tmp_path, reading, candidate))
    if word is None:
        assert got == candidate and not waiting and stop is None
    else:
        assert got is None and word in '；'.join(waiting+[stop or ''])


def test_the_dry_run_does_not_open_ufs_files_before_its_reading(tmp_path, monkeypatch):
    root = uf_root(tmp_path, None, evaluation=True)
    opened = []
    real = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda self, *a, **k: (opened.append(self) if str(self).startswith(str(root)) else None, real(self, *a, **k))[1])
    monkeypatch.setattr(HS, 'UF_OUT', root)
    lines = []
    RUN.uf_lines(lines.append)
    assert hc.uf_reading(root)[0] and not opened and 'existence only' in lines[0]


# ---- the composer --------------------------------------------------------------------------------------------------------------------------------
def probabilities(rng, shape, n=36, ties=False):
    p = rng.dirichlet(np.full(n, .3), size=int(np.prod(shape))).T.reshape((n,)+shape)
    if ties:
        p[1, 0, 0, :2] = p[2, 0, 0, :2] = .3
        p[0, 0, 0, :2] = .4
        p[3:, 0, 0, :2] = 0.
    return p


def test_the_mixture_is_half_and_half_element_by_element():
    rng = np.random.default_rng(1)
    shape = (17, 5, 4)
    p16 = probabilities(rng, shape).astype(np.float32).astype(np.float16)
    p_nn = probabilities(rng, shape)
    p_nn /= p_nn.sum(axis=0)
    want = EVAL_UF.half_and_half(p16, p_nn)
    mixed = hs_seg.mix_in_place(p16, p_nn.copy())
    assert (mixed.argmax(axis=0) == want).all()
    a = p16.astype(np.float64)
    assert np.array_equal(mixed, .5*(a/a.sum(axis=0))+.5*p_nn)       # exactly the plain formula
    assert hs_seg.b_differs_from_a(p16, p_nn) == int((want != p16.argmax(axis=0)).sum())
    with pytest.raises(ValueError, match='fixed'):
        hs_seg.mix_in_place(p16, p_nn.copy(), weight=.4)
    with pytest.raises(ValueError, match='float16'):
        hs_seg.mix_in_place(p16.astype(np.float32), p_nn.copy())


def grouping():
    from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping
    return load_grouping(DEFAULT_GROUPS_PATH, 'structure13_v1')


def base_provenance(subject='OASIS_OAS1_0901_MR1'):
    return dict(layer='structure', subject=subject, input_sha256={hs_seg.IMAGE_FILE: 'a'*64})


def test_the_composers_float16_keeps_its_argmax_and_the_reader_reads_it(tmp_path):
    rng = np.random.default_rng(2)
    shape = (6, 5, 4)
    mixed = probabilities(rng, shape, ties=True)
    mixed[:, 1, 1, 1] = 0.
    mixed[5, 1, 1, 1], mixed[6, 1, 1, 1] = .5-1e-6, .5+1e-6           # two values float16 cannot tell apart; the larger is the larger label
    labels = mixed.argmax(axis=0)
    assert (mixed.astype(np.float16).argmax(axis=0) != labels).any()
    row = hs_seg.write_prediction('OASIS_OAS1_0901_MR1', mixed, labels, out_root=tmp_path, grouping=grouping(), provenance=base_provenance())
    assert row['n_reader_not_own'] == 0 and row['n_float16_ties_fixed'] >= 1
    stored = np.load(tmp_path/'OASIS_OAS1_0901_MR1'/'probs36.npz')['p']
    assert stored.dtype == np.float16 and (stored.argmax(axis=0) == labels).all()
    prov = json.loads((tmp_path/'OASIS_OAS1_0901_MR1'/'provenance.json').read_text())
    assert prov['output_sha256'] == row['output_sha256'] and prov['outputs']['probs36']['dtype'] == 'float16'
    with pytest.raises(FileExistsError):
        hs_seg.write_prediction('OASIS_OAS1_0901_MR1', mixed, labels, out_root=tmp_path, grouping=grouping(), provenance=base_provenance())


def test_a_stores_exactly_what_v2_stores(tmp_path):
    rng = np.random.default_rng(3)
    probs = probabilities(rng, (6, 5, 4)).astype(np.float32)
    p16 = probs.astype(np.float16)
    row = hs_seg.write_prediction('OASIS_OAS1_0901_MR1', probs, p16.argmax(axis=0), out_root=tmp_path, grouping=grouping(), provenance=base_provenance())
    stored = np.load(tmp_path/'OASIS_OAS1_0901_MR1'/'probs36.npz')['p']
    assert row['n_float16_ties_fixed'] == 0 and stored.tobytes() == p16.tobytes()
    from bayesreg.seg.unet.groups import group_probabilities
    assert np.load(tmp_path/'OASIS_OAS1_0901_MR1'/'structure_groups.npz')['p'].tobytes() == group_probabilities(probs, grouping(), dtype=np.float32).astype(np.float16).tobytes()     # infer.segment_subject's groups


def test_a_check_that_fails_keeps_the_files_without_provenance(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    probs = probabilities(rng, (6, 5, 4))
    labels = probs.argmax(axis=0)
    wrong = labels.copy()
    wrong[0, 0, 0] = (wrong[0, 0, 0]+1) % 36
    monkeypatch.setattr(hs_seg.nx_seg, 'float16_keeping_argmax', lambda p, l: (p.astype(np.float16), 0))
    with pytest.raises(hs_seg.HSSegFailed, match='other than the argmax'):
        hs_seg.write_prediction('OASIS_OAS1_0901_MR1', probs, wrong, out_root=tmp_path, grouping=grouping(), provenance=base_provenance())
    d = tmp_path/'OASIS_OAS1_0901_MR1'
    assert (d/'probs36.npz').is_file() and not (d/'provenance.json').exists() and 'not rerun' in json.loads((d/'check_failed.json').read_text())['note']


# ---- the reading's lines --------------------------------------------------------------------------------------------------------------------------
def reference_of(values=None, problem=None):
    values = values or {}
    return {p: dict(metrics=metrics_of(values.get(p, .874)), problem=problem if p == P0 else None) for p in ALL}


def rows_of(values, *, sdlogj=.07, status=hc.HX_CERTIFIED, kind='certified', override=None):
    rows = {p: dict(pair=core.short_name(p), kind=kind, why=None, line_status=status, metrics=metrics_of(values.get(p, .876), sdlogj)) for p in ALL}
    for pair, change in (override or {}).items():
        rows[pair] = dict(rows[pair], **change)
    return rows


def test_block_interval_is_fd2s():
    rng = np.random.default_rng(7)
    d = rng.normal(.002, .004, 19)
    lo, hi = decision.block_interval(d, list(range(19)))
    # the recompute of FD2 (runs/20261105-fd2/recompute.py block), written out
    rng2, out = np.random.default_rng(20261105), []
    index = np.arange(19)
    while len(out) < 10000:
        counts = np.bincount(rng2.integers(0, 20, 20), minlength=20)
        w = counts[index]*counts[index+1]
        if w.sum():
            out.append(float((w*d).sum()/w.sum()))
    assert (lo, hi) == tuple(float(v) for v in np.percentile(out, [2.5, 97.5]))


def test_adopt_on_the_lines():
    reading, reason, compare = decision.read_hs(rows_of({}), reference_of())
    assert reading == hc.READ_ADOPT and reason is None and abs(compare['seg35']['mean']-.002) < 1e-12 and compare['seg35']['lower'] == pytest.approx(.002)


def test_a_mean_of_exactly_zero_is_adopted_and_below_is_not():
    assert decision.read_hs(rows_of({p: .874 for p in ALL}), reference_of())[0] == hc.READ_ADOPT
    reading, reason, _ = decision.read_hs(rows_of({p: .874-1e-6 for p in ALL}), reference_of())
    assert reading == hc.READ_NOT and 'Δ 均值' in reason


def test_the_lower_bound_must_be_above_minus_003(monkeypatch):
    rows = rows_of({p: .874+.002+(.03 if i % 2 else -.03) for i, p in enumerate(ALL)})
    reading, reason, compare = decision.read_hs(rows, reference_of())
    assert compare['seg35']['mean'] > 0 and compare['seg35']['lower'] < -.003 and reading == hc.READ_NOT and '下界' in reason
    monkeypatch.setattr(decision, 'block_interval', lambda d, index: (-.003, .01))
    assert '下界 -0.00300 ≤ -0.003' in decision.read_hs(rows_of({}), reference_of())[1]
    monkeypatch.setattr(decision, 'block_interval', lambda d, index: (-.003+1e-9, .01))
    assert decision.read_hs(rows_of({}), reference_of())[0] == hc.READ_ADOPT


@pytest.mark.parametrize('sdlogj,word', ((.10, hc.READ_ADOPT), (.1000001, hc.READ_NOT)))
def test_the_sdlogj_line(sdlogj, word):
    assert decision.read_hs(rows_of({}, sdlogj=sdlogj), reference_of())[0] == word


def test_a_fallback_pair_counts_at_its_fallback_value():
    rows = rows_of({}, override={R0: dict(line_status=hc.HX_A0, kind='fallback', metrics=metrics_of(.80))})
    reading, reason, compare = decision.read_hs(rows, reference_of())
    assert compare['seg35']['n'] == 19 and compare['seg35']['per_pair'][core.short_name(R0)] == pytest.approx(.80-.874) and reading == hc.READ_NOT and 'Δ 均值' in reason


@pytest.mark.parametrize('change,word', ((dict(kind=decision.NOT_RUN, line_status=None, metrics=None, why='x 场没有运行'), '没有运行'), (dict(kind=decision.NOT_READ, line_status=None, metrics=None, why='x 修复没有评测'), '没有评测'),
                                         (dict(kind='rejected', line_status=hc.HX_REJECTED, metrics=None, why='x 通道被拒'), '输入错误')))
def test_not_read_and_refused_pairs_are_undecidable(change, word):
    reading, reason, _ = decision.read_hs(rows_of({}, override={R0: change}), reference_of())
    assert reading == hc.UNDECIDABLE and word in reason


def test_rw_not_19_of_19_is_undecidable():
    reading, reason, _ = decision.read_hs(rows_of({}), reference_of(problem='0438-0439 RW 没有通过证书'))
    assert reading == hc.UNDECIDABLE and 'RW 没有通过证书' in reason


def test_nan_is_undecidable():
    reading, _, _ = decision.read_hs(rows_of({P0: float('nan')}), reference_of())
    assert reading == hc.UNDECIDABLE


def test_an_sdlogj_missing_on_one_pair_is_undecidable():
    rows = rows_of({}, override={R0: dict(metrics=dict(metrics_of(.876), sdlogj_official=None))})
    reading, reason, compare = decision.read_hs(rows, reference_of())
    assert compare['hs_sdlogj_official'] is None and reading == hc.UNDECIDABLE and 'SDlogJ' in reason


def test_rw_records_must_be_the_ones_pinned_at_the_start(results):
    write_rw(results)
    pins = decision.rw_pins(results)
    assert len(pins) == 19 and all(v['repair.json'] and v['repair_metrics.json'] for v in pins.values())
    assert all(r['problem'] is None for r in decision.rw_reference(results, pins).values())
    d = fx.RW.run_dir(results/'rw_v1', fx.RW.cell_of(R0))
    (d/'repair_metrics.json').write_text((d/'repair_metrics.json').read_text()+' ')
    reference = decision.rw_reference(results, pins)
    assert '钉住' in reference[R0]['problem'] and sum(r['problem'] is not None for r in reference.values()) == 1


def test_the_expectation_is_the_slope_times_the_pair_level_delta():
    subjects = sorted({s for p in ALL for s in p.split('__')})
    v1 = {s: dict(seg35=.926) for s in subjects}
    cand = {s: dict(seg35=.926+(.004 if k % 2 else .002)) for k, s in enumerate(subjects)}
    for arm in ('UF4', 'UF4_nn'):
        e = decision.expectation(dict(rows={arm: cand}, reference_rows=dict(reference_v1=v1)), arm, .874)
        d19 = np.mean([((cand[a]['seg35']+cand[b]['seg35'])-(v1[a]['seg35']+v1[b]['seg35']))/2 for a, b in (p.split('__') for p in ALL)])
        assert e['pairs19']['seg_level_delta'] == pytest.approx(d19) and e['pairs19']['expected_delta'] == pytest.approx(.890*d19) and e['expected_hs_seg35'] == pytest.approx(.874+.890*d19) and e['slope'] == .890
        assert e['first5']['expected_delta'] == pytest.approx(.890*np.mean([((cand[a]['seg35']+cand[b]['seg35'])-(v1[a]['seg35']+v1[b]['seg35']))/2 for a, b in (p.split('__') for p in core.PAIRS)]))
    missing = decision.expectation(dict(rows=dict(UF4={}), reference_rows=dict(reference_v1=v1)), 'UF4', .874)
    assert missing['pairs19']['expected_delta'] is None


# ---- the pairs on disk: what each kind is, the fallbacks ---------------------------------------------------------------------------------------------
@pytest.fixture
def results(monkeypatch):
    base_dir = tempfile.mkdtemp(prefix='hsq_')              # not the test runner's tmp_path: its name would fail the start-string check
    path = Path(base_dir)/'results'
    path.mkdir()
    monkeypatch.setattr(HS, 'RESULTS', path)
    monkeypatch.setattr(HS, 'DEFAULT_ROOT', path/HS.ROOT_NAME)
    monkeypatch.setattr(HS, 'UF_OUT', path/'seg'/'unet35_v4_full')
    for batch in (fx.RW,):
        monkeypatch.setattr(batch, 'RESULTS', path)
        monkeypatch.setattr(batch, 'DEFAULT_ROOT', path/batch.ROOT_NAME)
    monkeypatch.setattr(fx, 'RESULTS', path)
    monkeypatch.setattr(HS, 'registry_path', lambda repo=None: path/'segmenters_hs_v1.json')
    yield path
    shutil.rmtree(base_dir)


def rounds_of(n, wall=False):
    return [dict(round=k, n_free=9000, n_extra_points=0, inner=dict(stop='wall' if wall and k == n-1 else 'inner_cap', iterations=300), seconds_solve=30., seconds_certificate=1.) for k in range(n)]


def write_field(root, pair, *, status='done', evaluate=True, seg35=.85, refusal=None, run_json=True):
    cell = HS.field_cell(pair)
    d = HS.run_dir(root, cell)
    d.mkdir(parents=True, exist_ok=True)
    if run_json:
        write_json(d/'run.json', dict(status=status, reason=None, oracle=False, run_id=cell.run_id, pair=pair, k=35, weight='Q3h', D_factor=24., channels='onehot', taper=core.TAPER,
                                      channel_sources=dict(oracle=False, segmenter=dict(segmenter_id='unet35_v4_ens', pred_root=str(hc.seg_dir('UF4', HS.RESULTS)/'pred')), sides={side: dict(segmenter_id='unet35_v4_ens', policy='all_folds_mean') for side in ('fixed', 'moving')}),
                                      config=json.loads(json.dumps(HS.arm_config(cell, .5))), scale=dict(D_factor=24.), scale_check=dict(passed=True), optimizer=dict(levels=[dict(seconds=1.)]*4),
                                      checkpoints={'600': dict(step=600, file='field_0600.npz', sha256='s', u_sha256='u', seconds_task=45., seconds_io=1., terms=dict(data=1., geometry=1., floor=0., curvature=0., min_det=.1, total=2.))}))
    write_json(d/'queue_done.json', dict(run_id=cell.run_id, rc=0 if run_json else 1, unit_seconds=50., state=status if run_json else 'failed', arm_stop=None))
    if refusal:
        write_json(d/hc.REFUSAL_FILE, dict(kind=refusal, message=f'{refusal} message'))
    if evaluate and run_json and status == 'done':
        write_json(d/'metrics_0600.json', dict(oracle=False, certificate=dict(accepted_alpha=None, n_infeasible_at_1=9000, attempts=[dict(alpha=1., min_jd_lower_bound=-.1)]), metrics=metrics_of(seg35), tapered_u_sha256='t', seconds_certificate=4., seconds_metrics=8.))


def repair_config():
    """RF05 with the fuse as run_fl.repair_one records it (asdict of the RepairConfig the HS instance builds)."""
    from dataclasses import asdict
    return json.loads(json.dumps(asdict(RUN.BASE.repair.RepairConfig(eps=fl.REPAIR_EPS))))


def write_repair(root, pair, *, passed=True, review=None, evaluate=True, crashed=False, seg35=.876, rounds=1, wall=False):
    """review: the evaluation unit's certificate; by default it agrees with the repair's own."""
    review = passed if review is None else review
    cell = HS.repair_cell(pair)
    d = HS.run_dir(root, cell)
    d.mkdir(parents=True, exist_ok=True)
    write_json(d/'queue_done.json', dict(run_id=cell.run_id, rc=1 if crashed else 0, unit_seconds=60., state='failed' if crashed else ('certified' if passed else fx.NOT_CERTIFIED), arm_stop=None))
    if crashed:
        return
    np.savez(d/'repair.npz', nodes=np.arange(3), values=np.zeros((3, 3)))
    write_json(d/'repair.json', dict(status=fx.CERTIFIED if passed else fx.NOT_CERTIFIED, passed=passed, oracle=False, run_id=cell.run_id, pair=pair, reason='certified' if passed else 'round_cap', eps=.05, seconds_repair=60., seconds_warmup=2., source=dict(seconds_task=45., field_sha256='s', directory=str(HS.run_dir(root, HS.field_cell(pair)))), rounds=rounds_of(rounds, wall), attempt=dict(file='repair.npz', sha256=core.sha_file(d/'repair.npz'), n_free_nodes=3), config=repair_config()))
    if evaluate:
        write_json(d/'repair_metrics.json', dict(before=dict(metrics=metrics_of(seg35+.0005), n_infeasible_at_1=20000), after=dict(metrics=metrics_of(seg35), certificate=dict(accepted_alpha=1. if review else None, n_infeasible_at_1=0 if review else 1, attempts=[dict(min_jd_lower_bound=1e-6 if review else -.2)])), change=dict(), repaired_u_sha256='r'))


def all_certified(root, **kw):
    for pair in ALL:
        write_field(root, pair)
        write_repair(root, pair, **kw)


def write_rw(results, seg35=.874):
    """rw_v1 as RW left it: the control judged not bit for bit, 19 repairs of its own, all certified by both."""
    root = results/'rw_v1'
    fx.RW.write_selection(root, fx.RW.CONTROL_FILE, dict(same=False))
    for pair in ALL:
        cell = fx.RW.cell_of(pair)
        d = fx.RW.run_dir(root, cell)
        d.mkdir(parents=True, exist_ok=True)
        write_json(d/'queue_done.json', dict(run_id=cell.run_id, rc=0, unit_seconds=70., state='certified', arm_stop=None))
        write_json(d/'repair.json', dict(status='certified', passed=True, oracle=False, run_id=cell.run_id, reason='certified', seconds_repair=60., seconds_warmup=2., source=dict(seconds_task=45.), rounds=rounds_of(1), attempt=dict(sha256='x'), config=dict(wall_seconds=1200.)))
        write_json(d/'repair_metrics.json', dict(before=dict(metrics=metrics_of(seg35), n_infeasible_at_1=1), after=dict(metrics=metrics_of(seg35), certificate=dict(accepted_alpha=1., n_infeasible_at_1=0, attempts=[dict(min_jd_lower_bound=1e-6)])), change=dict(), repaired_u_sha256='r'))
    return root


def test_rw_reference_reads_19_of_19(results):
    write_rw(results)
    reference = decision.rw_reference(results)
    assert all(reference[p]['problem'] is None and reference[p]['metrics']['seg35'] == .874 for p in ALL)


def test_each_kind_of_pair(results):
    root = HS.DEFAULT_ROOT
    write_field(root, ALL[0]); write_repair(root, ALL[0])
    write_field(root, ALL[1], status='nonfinite')
    write_field(root, ALL[2], run_json=False, refusal='alignment_qc')
    write_field(root, ALL[3], run_json=False, refusal='provenance_rejected')
    write_field(root, ALL[4]); write_repair(root, ALL[4], crashed=True)
    write_field(root, ALL[5]); write_repair(root, ALL[5], passed=False)
    write_field(root, ALL[6]); write_repair(root, ALL[6], review=False)
    write_field(root, ALL[7], evaluate=False)
    write_field(root, ALL[8])
    write_field(root, ALL[9]); write_repair(root, ALL[9], evaluate=False)
    write_field(root, ALL[10], run_json=False)
    write_field(root, ALL[11]); write_repair(root, ALL[11], rounds=2, wall=True)
    write_field(root, ALL[13]); write_repair(root, ALL[13], passed=False, review=True)
    write_field(root, ALL[14]); write_repair(root, ALL[14], passed=False, evaluate=False)
    kinds = [decision.classify(root, p) for p in ALL]
    assert [k for k, _ in kinds[:15]] == ['certified', 'fallback', 'fallback', 'rejected', 'fallback', 'fallback', 'disagree', 'not_read', 'not_run', 'not_read', 'fallback', 'certified', 'not_run', 'disagree', 'not_read']
    assert '修复进程的证书不过，评测单元的复核证书通过' in kinds[13][1] and '复核证书没有' in kinds[14][1]
    assert '入口失败（状态 nonfinite' in kinds[1][1] and '对齐 QC 不过' in kinds[2][1] and '运行出错' in kinds[4][1] and '没有通过证书' in kinds[5][1] and 'D74-1' in kinds[5][1] and '状态 failed' in kinds[10][1]
    assert [decision.needs_fallback(root, p) for p in ALL[:7]] == [False, True, True, False, True, True, True]
    row = decision.method_row(root, ALL[11])
    assert row['line_status'] == hc.HX_CERTIFIED and row['wall_hit'] is True and row['rounds'] == 2      # (c-ii): a fuse hit then a pass is a pass
    assert decision.method_row(root, ALL[1])['kind'] == decision.NOT_READ and '还没有判' in decision.method_row(root, ALL[1])['why']


def write_fallback(root, pair, **record):
    write_json(HS.fallback_root(root)/core.short_name(pair)/'fallback.json', dict(dict(pair=core.short_name(pair), reads_labels=False), **record))


def test_the_fallbacks_take_their_metrics(results, monkeypatch):
    root = HS.DEFAULT_ROOT
    write_field(root, ALL[1], status='nonfinite')
    write_field(root, ALL[2]); write_repair(root, ALL[2], passed=False)
    write_field(root, ALL[3]); write_repair(root, ALL[3], review=False)
    write_field(root, ALL[4]); write_repair(root, ALL[4], review=False)
    write_fallback(root, ALL[1], output='A0')
    write_fallback(root, ALL[2], output='identity', a0_why='not qualified')
    write_fallback(root, ALL[3], output='repair', third_certificate=dict(passed=True))
    write_fallback(root, ALL[4], output='A0', third_certificate=dict(passed=False))
    write_json(HS.id_root(root)/core.short_name(ALL[2])/'identity_metrics.json', dict(certified_at_1=True, oracle=False, metrics=metrics_of(.6)))
    monkeypatch.setattr(rp, 'fallback_status', lambda pair, results=None, **kw: dict(qualified=True, why=None, metrics=metrics_of(.83)))
    rows = {p: decision.method_row(root, p) for p in ALL[1:5]}
    assert rows[ALL[1]]['line_status'] == hc.HX_A0 and rows[ALL[1]]['metrics']['seg35'] == .83 and '入口失败' in rows[ALL[1]]['why']
    assert rows[ALL[2]]['line_status'] == hc.HX_IDENTITY and rows[ALL[2]]['metrics']['seg35'] == .6
    assert rows[ALL[3]]['line_status'] == hc.HX_CERTIFIED and rows[ALL[3]]['third_certificate'] is True
    assert rows[ALL[4]]['line_status'] == hc.HX_A0 and '第三次认证不过' in rows[ALL[4]]['why']
    monkeypatch.setattr(rp, 'fallback_status', lambda pair, results=None, **kw: dict(qualified=False, why='gone', metrics={}))
    assert decision.method_row(root, ALL[1])['kind'] == decision.NOT_READ                  # recorded A0 that no longer qualifies: not read
    (HS.id_root(root)/core.short_name(ALL[2])/'identity_metrics.json').unlink()
    assert decision.method_row(root, ALL[2])['kind'] == decision.NOT_READ


class Grid:
    shape = (4, 4, 4)
    spacing = (1., 1., 1.)


class Result:
    def __init__(self, ok):
        self.status, self.reason, self.min_jd_lower_bound, self.n_cells_checked, self.n_infeasible, self.n_uncertain, self.certificate_version = ('certified_feasible' if ok else 'certified_infeasible'), None, (1e-3 if ok else -.1), 27, 0 if ok else 1, 0, 'v'
        self.config = types.SimpleNamespace()


def fake_certify(fails=()):
    def certify(u, grid, config=None):
        return Result(not any(np.array_equal(u, f) for f in fails) and not (np.asarray(u) == 7.).any())
    return certify


@pytest.fixture
def no_labels(monkeypatch):
    """The fallback unit's process: the firewall view gives the grid; reading a truth label fails the test."""
    def forbidden(*a, **k):
        raise AssertionError('a truth label was read')
    monkeypatch.setattr(fl, 'read_truth', forbidden)
    monkeypatch.setattr(HS, 'read_images', lambda cell: types.SimpleNamespace(grid=Grid(), provenance=dict(keys_read=['fixed', 'moving', 'initial_u'])))
    monkeypatch.setattr(MC0, 'asdict', lambda c: {})
    return forbidden


def test_the_fallback_unit_decides_each_case_and_reads_no_label(results, monkeypatch, no_labels):
    root = HS.DEFAULT_ROOT
    a0 = np.full((4, 4, 4, 3), .01)
    monkeypatch.setattr(rp, 'fallback_status', lambda pair, results=None, **kw: dict(qualified=pair != ALL[2], why=None if pair != ALL[2] else 'not qualified', file_sha256='f', run_id='r', d0_class='ok', metrics=metrics_of(.83)))
    monkeypatch.setattr(rp.RP, 'fallback_pins', lambda path=None: {core.short_name(p): dict(u_sha256=RUN.u_sha256(a0)) for p in ALL})
    monkeypatch.setattr(rp.RP, 'a0_field', lambda pair, results=None: results_npz)
    results_npz = HS.DEFAULT_ROOT.parent/'a0.npz'
    results_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(results_npz, u=a0)
    write_field(root, ALL[1], status='nonfinite')
    write_field(root, ALL[2], status='nonfinite')
    write_field(root, ALL[3]); write_repair(root, ALL[3], review=False)
    monkeypatch.setattr(RUN.BASE, 'source_checked', lambda cell, fz, grid: (np.zeros((4, 4, 4, 3)), dict(directory='d')))
    monkeypatch.setattr(RUN.BASE.efrun, 'repaired_field', lambda u, d, record: u+.02)
    passing = fake_certify()
    first = RUN.decide_fallback(root, ALL[1], fz=None, certify=passing, config=None)
    assert first['output'] == 'A0' and first['reads_labels'] is False and first['a0_certificate']['accepted_alpha'] == 1.
    assert first['keys_read'] == ['fixed', 'moving', 'initial_u'] and first['unit'] == RUN.FALLBACK_UNIT and first['process']['pid'] > 0      # revision 1d: what P2 revision 4 asks of the record
    assert RUN.decide_fallback(root, ALL[2], fz=None, certify=passing, config=None)['output'] == 'identity'
    refuse_a0 = fake_certify(fails=(a0,))
    record = RUN.decide_fallback(root, ALL[1], fz=None, certify=refuse_a0, config=None)
    assert record['output'] == 'identity' and 'does not pass the certificate here' in record['a0_why']
    third = RUN.decide_fallback(root, ALL[3], fz=None, certify=passing, config=None)
    assert third['output'] == 'repair' and third['third_certificate']['passed'] is True
    third = RUN.decide_fallback(root, ALL[3], fz=None, certify=fake_certify(fails=(np.full((4, 4, 4, 3), .02),)), config=None)
    assert third['output'] == 'A0' and third['third_certificate']['passed'] is False
    monkeypatch.setattr(rp.RP, 'fallback_pins', lambda path=None: {core.short_name(p): dict(u_sha256='other') for p in ALL})
    assert 'not the pinned state' in RUN.decide_fallback(root, ALL[1], fz=None, certify=passing, config=None)['a0_why']


def test_the_reading_on_disk_with_a_fallback(results, monkeypatch):
    root = HS.DEFAULT_ROOT
    write_rw(results)
    all_certified(root, seg35=.880)
    shutil.rmtree(HS.run_dir(root, HS.repair_cell(ALL[5])))
    write_repair(root, ALL[5], passed=False)
    write_fallback(root, ALL[5], output='A0')
    monkeypatch.setattr(rp, 'fallback_status', lambda pair, results=None, **kw: dict(qualified=True, why=None, metrics=metrics_of(.85)))
    monkeypatch.setattr(hc, 'uf_reading', lambda uf_out=None: ([], None, 'UF4', dict(reading='worth_hx24', primary_arm='UF4', candidate='UF4', delta_seg35=.004, delta_seg4=.001)))
    d = decision.write_hs(root, results=results)
    assert d['reading'] == hc.READ_ADOPT and d['counts']['by_status'][hc.HX_A0] == [core.short_name(ALL[5])] and d['counts']['n_fallback'] == 1
    assert d['method']['n'] == 19 and d['method']['seg35'] == pytest.approx((18*.880+.85)/19) and d['certified_only']['n'] == 18 and d['certified_only']['seg35'] == pytest.approx(.880)
    text = (root/'analysis/summary_hs.md').read_text()
    assert '采用新分割器' in text and '退路 A0（配置 C）' in text and '离 .88 差' in text and '不是 F_B' in text and '含退路' in text
    rows = list(csv.reader((root/'analysis/per_pair_hs.csv').open()))
    assert rows[0][:3] == ['pair', 'line_status', 'kind'] and len(rows) == 20


# ---- what HS gives P2: the decision file and the reference file, read by P2's own readers ------------------------------------------------------------
P2_COMMIT = '83af310'                           # the branch p2-tools: run_p2.hs_problem, p2_core.load_reference, decide_p2.reference_rows (runs/20261106-p2/plan.md §3, §3.1); 6cae24e before revision 1, 32aa28c in revision 1, 2b7cceb in 1b and 1c, 2488907 in 1d


@pytest.fixture(scope='module')
def p2_tools(tmp_path_factory):
    """P2's modules: the repository's when they are merged, else the files of p2-tools at P2_COMMIT in a copy of experiments/ (the other directories linked).  Skipped when neither is there."""
    import importlib
    base = ROOT/'experiments'/'p2'
    if not (base/'p2_core.py').is_file():
        experiments = tmp_path_factory.mktemp('p2tools')/'experiments'
        (experiments/'p2').mkdir(parents=True)
        for d in (ROOT/'experiments').iterdir():
            if d.is_dir() and d.name != 'p2':
                (experiments/d.name).symlink_to(d)
        for name in ('p2_core.py', 'decide_p2.py', 'run_p2.py'):
            done = subprocess.run(['git', 'show', f'{P2_COMMIT}:experiments/p2/{name}'], cwd=ROOT, capture_output=True)
            if done.returncode:
                pytest.skip(f'P2 tools not on this machine: {done.stderr.decode()[:200]}')
            (experiments/'p2'/name).write_bytes(done.stdout)
        base = experiments/'p2'
    sys.path.insert(0, str(base))
    try:
        modules = types.SimpleNamespace(**{name: importlib.import_module(name) for name in ('p2_core', 'decide_p2', 'run_p2')})
    finally:
        sys.path.remove(str(base))
    return modules


def adopted_root(results, monkeypatch, *, fallback_pair=None):
    root = HS.DEFAULT_ROOT
    write_rw(results)
    all_certified(root, seg35=.880)
    if fallback_pair is not None:
        shutil.rmtree(HS.run_dir(root, HS.repair_cell(fallback_pair)))
        write_repair(root, fallback_pair, passed=False)
        write_fallback(root, fallback_pair, output='A0')
        monkeypatch.setattr(rp, 'fallback_status', lambda pair, results=None, **kw: dict(qualified=True, why=None, metrics=metrics_of(.879)))
    monkeypatch.setattr(hc, 'uf_reading', lambda uf_out=None: ([], None, 'UF4', dict(reading='worth_hx24', primary_arm='UF4', candidate='UF4', delta_seg35=.004, delta_seg4=.001)))
    monkeypatch.setattr(HS, 'candidate', lambda uf_out=None: 'UF4')
    document = decision.write_hs(root, results=results)
    assert document['reading'] == hc.READ_ADOPT and document['segmenter_id'] == 'unet35_v4_ens'
    return root, document


def test_p2_reads_hs_decision_file(results, monkeypatch, p2_tools):
    root, document = adopted_root(results, monkeypatch)
    hs_problem = p2_tools.run_p2.hs_problem
    path = root/'analysis'/'decision_hs.json'
    assert p2_tools.run_p2.HS_ADOPT == hc.READ_ADOPT and hs_problem('unet35_v4_ens', path=path) is None
    assert 'unet35_v4_ens' in hs_problem('unet35_v1_ens', path=path)
    write_json(path, dict(json.loads(path.read_text()), reading=hc.READ_NOT))
    assert hs_problem('unet35_v1_ens', path=path) is None and hs_problem('unet35_v4_ens', path=path) is not None


def test_the_reference_for_p2_is_read_by_p2s_readers(results, monkeypatch, p2_tools, tmp_path):
    import hs_reference as href
    root, decision_document = adopted_root(results, monkeypatch)
    reference = href.reference_document(root, decision_document)
    path = href.write(tmp_path/'reference_unet35_v4_ens.json', reference)
    assert href.write(path, reference) == path                                    # the same content again: accepted
    loaded = p2_tools.p2_core.load_reference('unet35_v4_ens', path)
    assert loaded['schema'] == p2_tools.p2_core.REFERENCE_SCHEMA and sorted(loaded['p2']['pairs']) == sorted(core.short_name(p) for p in core.PAIRS) and len(loaded['p3']['pairs']) == 19
    for stage in ('p2', 'p3'):
        rows = p2_tools.decide_p2.reference_rows(loaded, stage, 'unet35_v4_ens', hs_decision=root/'analysis'/'decision_hs.json')
        assert all(r['kind'] == p2_tools.decide_p2.CERTIFIED and r['pinned'] is True for r in rows.values()) and len(rows) == (5 if stage == 'p2' else 19)
        assert all(r['repair']['after']['seg35'] == .880 for r in rows.values())
    pin = loaded['p3']['pairs'][core.short_name(R0)]
    assert pin['repair_dir'] == str(HS.run_dir(root, HS.repair_cell(R0))) and pin['tier'] == 'rest600' and pin['field_dir'] == str(HS.run_dir(root, HS.field_cell(R0))) and pin['hs_line_status'] == hc.HX_CERTIFIED
    assert all(pin[k] for k in ('attempt_sha256', 'repaired_u_sha256', 'field_sha256', 'field_u_sha256', 'repair_json_sha256', 'repair_metrics_sha256', 'run_json_sha256')) and href.fallback_rows(loaded) == [] and href.third_certificate_rows(loaded) == []
    channel = p2_tools.p2_core.ChannelSet('unet35_v4_ens', loaded['channels']).source()
    ours = RUN.hs_source()
    assert (Path(channel.pred_root), Path(channel.models_root), Path(channel.image_root), channel.segmenter_id) == (Path(ours.pred_root), Path(ours.models_root), Path(ours.image_root), ours.segmenter_id)
    assert Path(channel.registry) == core.ROOT/hc.REGISTRY_REL == hc.HSBatch.registry_path(HS)        # the fixture points HS's at a temporary file; the class's is the repository's
    reference['p2']['source'] += ' changed'
    with pytest.raises(href.ReferenceRefused, match='other content'):
        href.write(path, reference)


def test_a_reference_with_every_kind_of_fallback_row_is_read_by_p2(results, monkeypatch, p2_tools, tmp_path):
    """The three ways a pair of HS reaches a fallback, in one reference read by P2's readers at 2b7cceb: a repair that did not pass (A0), a field that is not done (nonfinite, A0), and the alignment QC refused (no channel read, the identity).
    P2 reads every row; a fallback row's value is HS's method value, decision_hs.json's per_pair metrics, key for key (P2 plan §3.1, revision 2)."""
    import hs_reference as href
    root = HS.DEFAULT_ROOT
    write_rw(results)
    all_certified(root, seg35=.880)
    failed, entry, refused = ALL[2], ALL[7], ALL[12]
    for pair in (failed, entry, refused):
        shutil.rmtree(HS.run_dir(root, HS.repair_cell(pair)))
        shutil.rmtree(HS.run_dir(root, HS.field_cell(pair)))
    write_field(root, failed); write_repair(root, failed, passed=False)
    write_field(root, entry, status='nonfinite')
    cell = HS.field_cell(refused)
    d = HS.run_dir(root, cell)
    d.mkdir(parents=True)
    write_json(d/'run.json', dict(schema='bayesreg.hs.entry_refused.v1', run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=refused, status=RUN.ENTRY_QC, reason='EF x', oracle=False, channel_sources=None, checkpoints={}, field=None))
    write_json(d/'queue_done.json', dict(run_id=cell.run_id, rc=0, unit_seconds=20., state=RUN.ENTRY_QC, arm_stop=None))
    write_json(d/hc.REFUSAL_FILE, dict(kind='alignment_qc', message='EF x'))
    write_fallback(root, failed, output='A0')
    write_fallback(root, entry, output='A0')
    write_fallback(root, refused, output='identity', a0_why='not qualified')
    write_json(HS.id_root(root)/core.short_name(refused)/'identity_metrics.json', dict(certified_at_1=True, oracle=False, metrics=metrics_of(.874)))
    monkeypatch.setattr(rp, 'fallback_status', lambda pair, results=None, **kw: dict(qualified=True, why=None, metrics=metrics_of(.879 if pair == failed else .878)))
    monkeypatch.setattr(hc, 'uf_reading', lambda uf_out=None: ([], None, 'UF4', dict(reading='worth_hx24', primary_arm='UF4', candidate='UF4', delta_seg35=.004, delta_seg4=.001)))
    monkeypatch.setattr(HS, 'candidate', lambda uf_out=None: 'UF4')
    document = decision.write_hs(root, results=results)
    assert document['reading'] == hc.READ_ADOPT and document['counts']['by_status'][hc.HX_A0] == [core.short_name(failed), core.short_name(entry)] and document['counts']['by_status'][hc.HX_IDENTITY] == [core.short_name(refused)]
    reference = href.reference_document(root, document)
    rows_ref = reference['p3']['pairs']
    assert rows_ref[core.short_name(refused)]['field_dir'] is None and rows_ref[core.short_name(refused)]['repair_dir'] is None and rows_ref[core.short_name(entry)]['repair_dir'] is None and rows_ref[core.short_name(entry)]['field_dir'] is not None
    assert all('method_metrics' in rows_ref[core.short_name(p)] for p in (failed, entry, refused)) and 'method_metrics' not in rows_ref[core.short_name(ALL[0])]
    loaded = p2_tools.p2_core.load_reference('unet35_v4_ens', href.write(tmp_path/'reference_unet35_v4_ens.json', reference))
    decision_path = root/'analysis'/'decision_hs.json'
    assert loaded['hs']['decision_sha256'] == core.sha_file(decision_path)
    per_pair = {r['pair']: r for r in json.loads(decision_path.read_text())['per_pair']}
    for stage in ('p2', 'p3'):
        rows = p2_tools.decide_p2.reference_rows(loaded, stage, 'unet35_v4_ens', hs_decision=decision_path)
        assert all(r['kind'] in (p2_tools.decide_p2.CERTIFIED, p2_tools.decide_p2.FALLBACK) and r['pinned'] for r in rows.values()), {r['short']: r['why'] for r in rows.values() if not r['pinned']}
        for pair, r in rows.items():
            if r['kind'] == p2_tools.decide_p2.FALLBACK:
                assert r['value'] == per_pair[r['short']]['metrics'] and r['line_status'] == per_pair[r['short']]['line_status'] and all(r['value'][m] == per_pair[r['short']]['metrics'][m] for m in p2_tools.p2_core.METHOD_KEYS)
            else:
                assert r['value'] == per_pair[r['short']]['metrics'] and per_pair[r['short']]['line_status'] == hc.HX_CERTIFIED       # a certified row: the repaired field's metrics, the same HS used
    assert sorted(r['short'] for r in p2_tools.decide_p2.reference_rows(loaded, 'p3', 'unet35_v4_ens', hs_decision=decision_path).values() if r['kind'] == p2_tools.decide_p2.FALLBACK) == sorted(core.short_name(p) for p in (failed, entry, refused))
    other = tmp_path/'other_decision.json'
    other.write_text(decision_path.read_text()+' ')
    rows = p2_tools.decide_p2.reference_rows(loaded, 'p3', 'unet35_v4_ens', hs_decision=other)                 # not the decision file the reference names: P2 reads none of its rows (2488907: a reference of another segmenter rests on decision_hs.json as a whole)
    assert all(r['kind'] == p2_tools.decide_p2.NOT_READ for r in rows.values())


def test_no_reference_without_adoption_and_a_fallback_pair_is_named(results, monkeypatch, p2_tools, tmp_path):
    import hs_reference as href
    root, decision_document = adopted_root(results, monkeypatch, fallback_pair=R0)
    reference = href.reference_document(root, decision_document)
    assert href.fallback_rows(reference) == [core.short_name(R0)] and href.third_certificate_rows(reference) == [] and reference['p3']['pairs'][core.short_name(R0)]['hs_line_status'] == hc.HX_A0
    rows = p2_tools.decide_p2.reference_rows(p2_tools.p2_core.load_reference('unet35_v4_ens', href.write(tmp_path/'r.json', reference)), 'p3', 'unet35_v4_ens', hs_decision=root/'analysis'/'decision_hs.json')
    hs_row = next(r for r in json.loads((root/'analysis'/'decision_hs.json').read_text())['per_pair'] if r['pair'] == core.short_name(R0))
    assert rows[R0]['kind'] == p2_tools.decide_p2.FALLBACK and rows[R0]['value'] == hs_row['metrics'] and rows[R0]['line_status'] == hc.HX_A0     # P2 takes HS's own method value (2b7cceb)
    with pytest.raises(href.ReferenceRefused, match='does not read'):
        href.reference_document(root, dict(decision_document, reading=hc.READ_NOT))
    with pytest.raises(href.ReferenceRefused, match='names'):
        href.reference_document(root, dict(decision_document, segmenter_id='uf_nn_v1'))


def test_write_reference_takes_only_the_decision_the_records_give(results, monkeypatch, p2_tools, tmp_path):
    root, _ = adopted_root(results, monkeypatch)
    repo = tmp_path/'repo'
    (repo/'experiments'/'p2').mkdir(parents=True)
    o = types.SimpleNamespace(root=str(root))
    assert RUN.write_reference(o, repo=repo) == 0
    written = repo/'experiments'/'p2'/'reference_unet35_v4_ens.json'
    assert p2_tools.p2_core.load_reference('unet35_v4_ens', written)['segmenter_id'] == 'unet35_v4_ens'
    path = root/'analysis'/'decision_hs.json'
    write_json(path, dict(json.loads(path.read_text()), reason='changed by hand'))
    assert RUN.write_reference(o, repo=repo) == 8


def test_write_reference_writes_the_canonical_root(results, monkeypatch, p2_tools, tmp_path):
    """--root another spelling of hs_v1 (a link): the pins use HS.DEFAULT_ROOT, the spelling of repair.json's source.directory, so P2's check still holds (revision 1c, minor 3)."""
    root, _ = adopted_root(results, monkeypatch)
    link = tmp_path/'hs_link'
    link.symlink_to(root)
    repo = tmp_path/'repo'
    (repo/'experiments'/'p2').mkdir(parents=True)
    assert RUN.write_reference(types.SimpleNamespace(root=str(link)), repo=repo) == 0
    written = json.loads((repo/'experiments'/'p2'/'reference_unet35_v4_ens.json').read_text())
    pins = [v for stage in ('p2', 'p3') for v in written[stage]['pairs'].values()]
    assert all(v['field_dir'].startswith(str(HS.DEFAULT_ROOT)+'/') and v['repair_dir'].startswith(str(HS.DEFAULT_ROOT)+'/') for v in pins) and not any(str(link) in json.dumps(v) for v in pins)
    rows = p2_tools.decide_p2.reference_rows(p2_tools.p2_core.load_reference('unet35_v4_ens', repo/'experiments'/'p2'/'reference_unet35_v4_ens.json'), 'p3', 'unet35_v4_ens', hs_decision=root/'analysis'/'decision_hs.json')
    assert all(r['kind'] == p2_tools.decide_p2.CERTIFIED for r in rows.values())


def test_a_row_of_the_third_certificate_is_not_called_a_fallback(results, monkeypatch, capsys, tmp_path):
    import hs_reference as href
    root = HS.DEFAULT_ROOT
    write_rw(results)
    all_certified(root, seg35=.880)
    shutil.rmtree(HS.run_dir(root, HS.repair_cell(R0)))
    write_repair(root, R0, seg35=.880, review=False)
    write_fallback(root, R0, **third_record(R0))
    monkeypatch.setattr(hc, 'uf_reading', lambda uf_out=None: ([], None, 'UF4', dict(reading='worth_hx24', primary_arm='UF4', candidate='UF4', delta_seg35=.004, delta_seg4=.001)))
    monkeypatch.setattr(HS, 'candidate', lambda uf_out=None: 'UF4')
    document = decision.write_hs(root, results=results)
    reference = href.reference_document(root, document)
    assert href.third_certificate_rows(reference) == [core.short_name(R0)] and href.fallback_rows(reference) == [] and 'method_metrics' not in reference['p3']['pairs'][core.short_name(R0)]
    repo = tmp_path/'repo'
    (repo/'experiments'/'p2').mkdir(parents=True)
    assert RUN.write_reference(types.SimpleNamespace(root=str(root)), repo=repo) == 0
    out = capsys.readouterr().out
    assert 'third certificate' in out and 'fallback rows' not in out


def third_record(pair, **changes):
    """What decide_fallback writes for a pair whose third certificate passed (run_hs.py): reads_labels false, configuration C, the repaired field's sha256 (the fakes' 'r'), output repair."""
    return dict(dict(kind='disagree', reads_labels=False, certificate_config=dict(rp.CERTIFICATE_C), output='repair', keys_read=['fixed', 'moving', 'initial_u'], unit=RUN.FALLBACK_UNIT,
                     third_certificate=dict(passed=True, certificate=dict(accepted_alpha=1., status_at_1='certified_feasible'), repaired_u_sha256='r')), **changes)


def test_the_third_certificate_record_says_which_keys_and_which_unit():
    """revision 1d: the record of the fallback unit holds load_pair's keys_read (within fixed, moving, initial_u) and the certifying unit (the fallback unit, not an evaluation unit); write-reference names a record that does not."""
    good = dict(third_record(R0), pair=core.short_name(R0))
    assert RUN.third_record_problems(good) == [] and RUN.third_record_problems(None) == ['no record']
    for keys in (['fixed', 'moving', 'moving_seg35'], [], ['fixed', 'initial_u'], None):
        assert RUN.third_record_problems(dict(good, keys_read=keys)) == ['keys_read']
    assert RUN.third_record_problems(dict(good, keys_read=['moving', 'fixed'])) == []
    for unit in ('tag-hs-eval', 'tag-hs-eval-r', RUN.ID_EVAL_UNIT, RUN.SEGCHECK_UNIT, None, 'hs-fallback'):
        assert RUN.third_record_problems(dict(good, unit=unit)) == ['unit']
    assert RUN.third_record_problems(dict(good, reads_labels=True, output='A0')) == ['no_label', 'output']


def test_a_row_of_the_third_certificate_is_read_certified_by_p2_with_its_record(results, monkeypatch, p2_tools, tmp_path):
    """revision 1d: the row hx_certified by the third certificate carries third_certificate_file and third_certificate_sha256; P2 2488907 reads it certified, its value the repaired field's; without either key, or with the record changed, not read."""
    import hs_reference as href
    root = HS.DEFAULT_ROOT
    write_rw(results)
    all_certified(root, seg35=.880)
    shutil.rmtree(HS.run_dir(root, HS.repair_cell(R0)))
    write_repair(root, R0, seg35=.880, review=False)
    write_fallback(root, R0, **third_record(R0))
    monkeypatch.setattr(hc, 'uf_reading', lambda uf_out=None: ([], None, 'UF4', dict(reading='worth_hx24', primary_arm='UF4', candidate='UF4', delta_seg35=.004, delta_seg4=.001)))
    monkeypatch.setattr(HS, 'candidate', lambda uf_out=None: 'UF4')
    document = decision.write_hs(root, results=results)
    assert document['reading'] == hc.READ_ADOPT and next(r for r in document['per_pair'] if r['pair'] == core.short_name(R0))['line_status'] == hc.HX_CERTIFIED
    reference = href.reference_document(root, document)
    pin = reference['p3']['pairs'][core.short_name(R0)]
    record = HS.fallback_root(root)/core.short_name(R0)/'fallback.json'
    assert pin['third_certificate_file'] == str(record) and pin['third_certificate_sha256'] == core.sha_file(record) and 'method_metrics' not in pin and pin['certified_at_1'] is False
    pinned = json.loads(record.read_text())
    assert pinned['keys_read'] == ['fixed', 'moving', 'initial_u'] and pinned['unit'] == RUN.FALLBACK_UNIT and RUN.third_record_problems(pinned) == []      # the sha256 pins these too
    assert all('third_certificate_file' not in v for k, v in reference['p3']['pairs'].items() if k != core.short_name(R0))
    decision_path = root/'analysis'/'decision_hs.json'

    def p3_row(document, name):
        loaded = p2_tools.p2_core.load_reference('unet35_v4_ens', href.write(tmp_path/name, document))
        return p2_tools.decide_p2.reference_rows(loaded, 'p3', 'unet35_v4_ens', hs_decision=decision_path)
    rows = p3_row(reference, 'reference_unet35_v4_ens.json')
    assert rows[R0]['kind'] == p2_tools.decide_p2.CERTIFIED and rows[R0]['value']['seg35'] == .880 and all(r['kind'] == p2_tools.decide_p2.CERTIFIED for r in rows.values())
    for key in ('third_certificate_file', 'third_certificate_sha256'):
        broken = json.loads(json.dumps(reference))
        for stage in ('p2', 'p3'):
            if core.short_name(R0) in broken[stage]['pairs']:
                del broken[stage]['pairs'][core.short_name(R0)][key]
        rows = p3_row(broken, f'without_{key}.json')
        assert rows[R0]['kind'] == p2_tools.decide_p2.NOT_READ and rows[ALL[0]]['kind'] == p2_tools.decide_p2.CERTIFIED
    original = record.read_text()
    for key in ('keys_read', 'unit'):                                               # revision 1e, P2 revision 4 (83af310): a record without either key, its reference written afresh (the sha256 holds): not read
        write_json(record, {k: v for k, v in json.loads(original).items() if k != key})
        rows = p3_row(href.reference_document(root, document), f'record_without_{key}.json')
        assert rows[R0]['kind'] == p2_tools.decide_p2.NOT_READ and f'third certificate: {key}' in rows[R0]['why'] and rows[ALL[0]]['kind'] == p2_tools.decide_p2.CERTIFIED
    write_json(record, dict(json.loads(original), unit='tag-hs-eval'))                 # the certifying unit an evaluation unit
    assert p3_row(href.reference_document(root, document), 'record_eval_unit.json')[R0]['kind'] == p2_tools.decide_p2.NOT_READ
    record.write_text(original)
    assert p3_row(href.reference_document(root, document), 'record_again.json')[R0]['kind'] == p2_tools.decide_p2.CERTIFIED
    write_json(record, dict(json.loads(original), reads_labels=True))         # the record changed after the reference was written: its sha256 no longer holds
    rows = p3_row(reference, 'after_the_record_changed.json')
    assert rows[R0]['kind'] == p2_tools.decide_p2.NOT_READ


def test_a_wrong_oracle_mark_writes_no_reading(results, monkeypatch):
    root = HS.DEFAULT_ROOT
    write_rw(results)
    all_certified(root)
    path = HS.run_dir(root, HS.field_cell(P0))/'run.json'
    write_json(path, dict(json.loads(path.read_text()), oracle=True))
    with pytest.raises(SystemExit, match='oracle false'):
        decision.decide(root, results=results)


# ---- the mark and the source of the channels --------------------------------------------------------------------------------------------------------
def test_the_mark_is_the_candidates():
    images = ['fixed', 'moving', 'initial_u']
    for sid in ('unet35_v4_ens', 'uf_nn_v1'):
        sides = {s: dict(segmenter_id=sid, policy='all_folds_mean') for s in ('fixed', 'moving')}
        assert RUN.hs_oracle_mark(images, sides, sid) == dict(oracle=False, truth_labels_read=False)
        assert RUN.hs_oracle_mark(images+['moving_seg35'], sides, sid)['oracle'] is True
        assert RUN.hs_oracle_mark(images, dict(sides, fixed=dict(segmenter_id='unet35_v1_ens', policy='all_folds_mean')), sid)['oracle'] is True
        assert RUN.hs_oracle_mark(images, dict(sides, fixed=dict(segmenter_id=sid, policy='cross_fit_other_folds')), sid)['oracle'] is True


def refusal_of(fn, *args, **kwargs):
    with pytest.raises(SystemExit) as raised:
        fn(*args, **kwargs)
    return str(raised.value)


def test_the_kind_of_a_refusal(tmp_path):
    """alignment_qc only for the QC that was computed and failed (fast_channels.read_side); the other refusals that name the alignment QC are input errors (revision 1, review important 1).  The messages are the code's own where it can be called."""
    for dice in (0.7123456789, None, 1e-05, 0.):
        assert RUN.refusal_kind(f"EF {P0.split('__')[0]} <- {P0.split('__')[1]}:moving: alignment QC {dice} < {0.8} (protocol §12.4 item 5)") == 'alignment_qc'
    source = (ROOT/'experiments/ef/fast_channels.py').read_text()
    assert source.count('raise SystemExit(f"EF {pin.fixed_subject} <- {pin.moving_subject}:{side}: alignment QC {qc[\'dice\']} < {qc[\'threshold\']} (protocol §12.4 item 5)")') == 1     # the wording the pattern relies on
    mi = core.make_inputs()
    pin = types.SimpleNamespace(fixed_subject=P0.split('__')[0], moving_subject=P0.split('__')[1])
    (tmp_path/'atropos'/pin.fixed_subject).mkdir(parents=True)
    broken = tmp_path/'broken'/pin.fixed_subject
    broken.mkdir(parents=True)
    (broken/'provenance.json').write_text('{not json')
    messages = [refusal_of(mi._dice, np.zeros(2, bool), np.zeros(3, bool), 'alignment QC'),                    # make_inputs.py: arrays of other shapes
                refusal_of(mi.atropos_wm, None, pin, 'fixed'),                                                 # no seg_root
                refusal_of(mi.atropos_wm, tmp_path/'atropos', pin, 'fixed'),                                   # no Atropos provenance.json
                refusal_of(mi.atropos_wm, tmp_path/'broken', pin, 'fixed'),                                    # the Atropos posteriors refused
                f'A4p fixed {pin.fixed_subject}: grouping structure13_v1 has no group cerebral_wm of seg35 labels (1, 20) (alignment QC, protocol §12.4 item 5)',      # fast_channels.read_side, the grouping
                'A4p fixed s: structure layer refused (ProvenanceRejected: x)', 'A4p: subjects refused (D24): {}', 'probs36.npz: a voxel with zero total probability', 'A4p: the repository registry x must be committed and clean']
    assert 'arrays of shape' in messages[0] and 'none given' in messages[1] and 'no provenance.json; alignment QC' in messages[2] and 'Atropos posteriors refused' in messages[3] and all('alignment QC' in m for m in messages[:5])
    for message in messages:
        assert RUN.refusal_kind(message) == 'provenance_rejected', message
    assert "no group {mi.UNET_WM_GROUP} of seg35 labels {mi.CEREBRAL_WM} (alignment QC, protocol §12.4 item 5)" in source


def test_a_load_pair_refusal_is_an_input_error_on_any_pair_the_last_one_too(results, monkeypatch):
    pair = ALL[-1]
    cell = HS.field_cell(pair)
    d = HS.run_dir(HS.DEFAULT_ROOT, cell)

    def refuse(c):
        raise SystemExit('SubjectSealed: OASIS_OAS1_0457_MR1 is not on the allowlist')
    monkeypatch.setattr(HS, 'read_images', refuse)                  # on the instance: an earlier monkeypatch may have left an instance attribute that shadows the class
    monkeypatch.setattr(RUN.BASE, 'run_one', lambda o: HS.read_images(HS.require_cell(o.arm, o.tier, o.pair)))         # run_fl.run_field reads the images first
    with pytest.raises(SystemExit, match='SubjectSealed'):
        RUN.run_one(types.SimpleNamespace(arm=cell.arm, tier=cell.tier, pair=pair, out=str(d), code_sha='s', device='cpu'))
    refusal = json.loads((d/hc.REFUSAL_FILE).read_text())
    assert refusal['kind'] == 'provenance_rejected' and refusal['stage'] == 'load_pair' and HS.__dict__['read_images'] is refuse      # run_one put back what was there
    write_json(d/'queue_done.json', dict(run_id=cell.run_id, rc=1, unit_seconds=3., state='failed', arm_stop='K = 35 run failed without a record'))
    assert decision.classify(HS.DEFAULT_ROOT, pair)[0] == 'rejected' and decision.method_row(HS.DEFAULT_ROOT, pair)['line_status'] == hc.HX_REJECTED


def test_an_alignment_qc_refusal_is_an_entry_failure_and_any_other_refusal_is_raised(results, monkeypatch):
    cell = HS.field_cell(P0)
    d = HS.run_dir(HS.DEFAULT_ROOT, cell)

    def refuse(message):
        def run_one(o):
            Path(o.out).mkdir(parents=True, exist_ok=True)
            RUN.run.write_json(Path(o.out)/hc.REFUSAL_FILE, dict(kind=RUN.refusal_kind(message), message=message))
            raise SystemExit(message)
        return run_one
    o = types.SimpleNamespace(arm=cell.arm, tier=cell.tier, pair=cell.pair, out=str(d), code_sha='s', device='cpu')
    monkeypatch.setattr(RUN.BASE, 'run_one', refuse(f"EF {P0.split('__')[0]} <- {P0.split('__')[1]}:fixed: alignment QC 0.7 < 0.8 (protocol §12.4 item 5)"))
    assert RUN.run_one(o) == 0
    record = json.loads((d/'run.json').read_text())
    assert record['status'] == RUN.ENTRY_QC and record['oracle'] is False and record['checkpoints'] == {} and record['channel_sources'] is None
    assert RUN.run.arm_stop_reason(cell, record, 30., k=35, vram_limit=9*2**30) is None           # the arm goes on
    kind, why = decision.classify(HS.DEFAULT_ROOT, P0)
    assert kind == 'fallback' and '对齐 QC 不过' in why
    other = HS.run_dir(HS.DEFAULT_ROOT, HS.field_cell(ALL[1]))
    monkeypatch.setattr(RUN.BASE, 'run_one', refuse('A4p fixed s: structure layer refused (ProvenanceRejected: x)'))
    with pytest.raises(SystemExit, match='structure layer refused'):
        RUN.run_one(types.SimpleNamespace(arm=cell.arm, tier=HS.field_cell(ALL[1]).tier, pair=ALL[1], out=str(other), code_sha='s', device='cpu'))
    assert not (other/'run.json').exists() and json.loads((other/hc.REFUSAL_FILE).read_text())['kind'] == 'provenance_rejected'


# ---- the real reader of the Hx line on HS's predictions ------------------------------------------------------------------------------------------------
def hs_world(env, monkeypatch, candidate):  # noqa: F811
    """The pair of the synthetic stage; the candidate registered in the test's repository (members of made-up sha256); its predictions written by write_prediction; the reader's source pointed there."""
    pair = world_of(env, monkeypatch)
    root = env['root']
    contact = sorted(env['fold0']+env['fold1'])                                                  # env's training draw (C23.9-13 (8), revision 6e)
    members = [dict(fold=f, seed=s, member_sha256=f'{k:x}'*64, label_contact_subjects=contact) for k, (f, s) in enumerate(hc.UF_MEMBERS, 1)]
    document = hs_seg.registry_document(candidate, members=members, nn_sha256='e'*64 if candidate == 'UF4_nn' else None, nn_contact=contact if candidate == 'UF4_nn' else None)
    registry = root/'repo'/'configs'/'segmenters_hs_v1.json'
    registry.write_text(hs_seg.document_text(document))
    a4.commit_registry(registry)
    monkeypatch.setattr(structure_layer, 'FORMAL_REGISTRIES', structure_layer.FORMAL_REGISTRIES+(registry,))
    from bayesreg.io.pair import load_segmenter_registry
    from bayesreg.seg.unet.groups import require_inference_grouping
    loaded = load_segmenter_registry(registry)
    g = grouping()
    for subject in pair.split('__'):
        p = a4.posterior(subject)
        models = dict(loaded.models)
        member_rows = [dict(model_id=m, member_sha256=s) for m in sorted(models) for s in models[m]['member_model_sha256']]
        prov = hs_seg.provenance_of(subject, candidate=candidate, registry=loaded, models=models, policy='all_folds_mean', members=member_rows, meta=dict(sha256=a4.source_sha(subject), grid_shape_zyx=list(a4.SHAPE), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.]),
                                    norm={}, grouping=g, grouping_check=require_inference_grouping(g), registry_rel=str(registry), staged_sha256=loaded.sha256, code=dict(commit='c'), environment={}, command=['test'], composition=dict(rule='test'), seconds=0.)
        hs_seg.write_prediction(subject, p, p.astype(np.float16).argmax(axis=0), out_root=root/'hs'/'pred', grouping=g, provenance=prov)
    mi = core.make_inputs()
    source = mi.UNetSource(pred_root=root/'hs'/'pred', folds=env['unet'].folds, folds_sha256=env['unet'].folds_sha256, models_root=root/'models', image_root=root/'images', segmenter_id=hc.CANDIDATES[candidate]['segmenter_id'], registry=registry, gate=dict(rule='test'))
    monkeypatch.setattr(RUN, 'hs_source', lambda: source)
    monkeypatch.setattr(HS, 'candidate', lambda uf_out=None: candidate)
    return pair, source


@pytest.mark.parametrize('candidate', ('UF4', 'UF4_nn'))
@synthetic_folds('env')
def test_the_real_reader_accepts_hs_predictions_and_reads_their_argmax(env, monkeypatch, candidate):  # noqa: F811
    pair, source = hs_world(env, monkeypatch, candidate)
    pin = HS.read_images(HS.field_cell(pair))
    predicted = RUN.BASE.fast.read_predicted(pin)
    for side, subject in (('fixed', pin.fixed_subject), ('moving', pin.moving_subject)):
        assert (predicted.labels[side] == a4.posterior(subject).astype(np.float16).argmax(axis=0)).all() and predicted.record['sides'][side]['read_path'] == 'fast'
        assert predicted.record['sides'][side]['segmenter_id'] == hc.CANDIDATES[candidate]['segmenter_id'] and sorted(predicted.record['sides'][side]['model_ids']) == sorted(v for v in hc.model_ids(candidate).values() if v)
    assert predicted.record['oracle'] is False and predicted.record['truth_labels_read'] is False
    assert RUN.BASE.fast._module.read_predicted(pin, unet=source).record['oracle'] is True        # mc0p_core.oracle_mark names unet35_v1_ens: why HS computes its own


@synthetic_folds('env')
def test_the_channels_of_hs_are_hx24s_on_the_same_labels(env, monkeypatch):  # noqa: F811
    pair, _ = hs_world(env, monkeypatch, 'UF4')
    pin = HS.read_images(HS.field_cell(pair))
    predicted = RUN.BASE.fast.read_predicted(pin)
    channels = RUN.BASE.channels
    (a, b), sa = channels.prepare_stacks(HS.ARMS[hc.HS_ARM], pin, predicted, core.load_table(), 'cpu')
    (c, d), sb = channels.prepare_stacks(fd2.ARMS['Hx24'], pin, predicted, core.load_table(), 'cpu')
    assert a.tobytes() == c.tobytes() and b.tobytes() == d.tobytes() and sa == sb


def test_a_refusal_is_recorded_with_its_kind_and_raised(env, monkeypatch, tmp_path):  # noqa: F811
    pair, source = hs_world(env, monkeypatch, 'UF4')
    registry = Path(source.registry)
    registry.write_text(registry.read_text()+' ')                  # not committed and clean any more
    pin = HS.read_images(HS.field_cell(pair))
    RUN.CURRENT['out'] = tmp_path/'run'
    try:
        with pytest.raises(SystemExit, match='committed and clean'):
            RUN.BASE.fast.read_predicted(pin)
    finally:
        RUN.CURRENT.pop('out')
    assert json.loads((tmp_path/'run'/hc.REFUSAL_FILE).read_text())['kind'] == 'provenance_rejected'


def test_the_registry_document(tmp_path):
    contact = sorted(a4.FOLD0+a4.FOLD1)
    members = [dict(fold=f, seed=s, member_sha256=f'{k:x}'*64, label_contact_subjects=contact) for k, (f, s) in enumerate(hc.UF_MEMBERS, 1)]
    a = hs_seg.registry_document('UF4', members=members)
    assert sorted(a['models']) == ['unet35_v4_ens_foldA', 'unet35_v4_ens_foldB'] and a['models']['unet35_v4_ens_foldA']['member_model_sha256'] == ['1'*64, '2'*64] and a['models']['unet35_v4_ens_foldB']['fold'] == 1
    b = hs_seg.registry_document('UF4_nn', members=members, nn_sha256='e'*64, nn_contact=contact)
    assert b['models']['uf_nn_v1_nnall'] == dict(segmenter_id='uf_nn_v1', fold=2, member_model_sha256=['e'*64], label_contact_subjects=contact, label_contact_subjects_sha256=b['models']['uf_nn_v1_foldA']['label_contact_subjects_sha256'])
    hs_seg.validated(hs_seg.document_text(b), tmp_path/'x.json')
    with pytest.raises(hs_seg.HSSegFailed, match='label contact'):
        hs_seg.registry_document('UF4_nn', members=members, nn_sha256='e'*64, nn_contact=contact[1:])
    with pytest.raises(hs_seg.HSSegFailed, match='needs the nnU-Net'):
        hs_seg.registry_document('UF4_nn', members=members)
    with pytest.raises(Exception, match='seeds'):
        hs_seg.registry_document('UF4', members=members[:1]+members[2:])


# ---- one synthetic UF world: HS's inference reproduces eval_uf.json --------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def uf_world(tmp_path_factory):
    import torch
    torch.set_num_threads(1)
    run_uf = _load('_hs_run_uf', ROOT/'experiments/uf/run_uf.py')
    from bayesreg.seg.unet.data import load_full_folds
    work = tmp_path_factory.mktemp('hs_uf_world')
    environ = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    synthetic = run_uf.build_synthetic(work, environ)
    run_uf.members_plan(synthetic.spec, load_full_folds(work/'folds_full_v2.json'))
    run_uf.run_chain(synthetic.spec)
    evaluation = json.loads((synthetic.spec.out/'eval_uf'/'eval_uf.json').read_text())
    return types.SimpleNamespace(s=synthetic, out=synthetic.spec.out, evaluation=evaluation, subjects=evaluation['subjects'], work=work)


def seg_kwargs(w, candidate, tmp_path):
    from bayesreg.seg.unet.data import OasisFiles, load_full_folds
    from bayesreg.seg.unet.pins import load_synthetic_inputs
    split, allowlist, _ = load_synthetic_inputs(w.work/'folds_full_v2.json', w.s.allowlist, data_root=w.s.data, training_source=a4_source(), expect_counts=(5, 1), loader=load_full_folds)
    return dict(subjects=w.subjects, seg_dir=tmp_path/candidate, uf_out=w.out, nn_out=w.s.nn_out, evaluation=w.evaluation, split=split, allowlist=allowlist, training_source=a4_source(), pinned_folds_sha256=None, files=OasisFiles(w.s.data), device='cpu', code=dict(commit='test'), command=['test'], log=lambda _: None)


def a4_source():
    sys.path.insert(0, str(ROOT/'experiments/unv2'))
    import synthetic_tree
    return synthetic_tree.SOURCE


@pytest.fixture
def stage_nn_unchecked(monkeypatch):
    """The synthetic nnU-Net contact is not the synthetic training set of the folds file: the contact check has its own tests (FX's); here only its call is bypassed."""
    original = hs_seg.stage
    monkeypatch.setattr(hs_seg, 'stage', lambda *a, **k: original(*a, **dict(k, contact_check=False)))


@pytest.mark.parametrize('candidate', ('UF4', 'UF4_nn'))
def test_hs_reproduces_eval_ufs_rows_and_counts(uf_world, tmp_path, candidate, stage_nn_unchecked, monkeypatch):
    import eval_seg
    reads = []
    real = eval_seg.read_truth
    monkeypatch.setattr(eval_seg, 'read_truth', lambda *a, **k: (reads.append(a), real(*a, **k))[1])
    kwargs = seg_kwargs(uf_world, candidate, tmp_path)
    record = hs_seg.infer_all(candidate, **kwargs)
    assert not reads                                                # the inference reads no truth
    assert record['passed'] is True and record['n_reader_not_own_total'] == 0 and sorted(record['rows']) == sorted(uf_world.subjects)
    staged = tmp_path/candidate/hc.STAGED_REGISTRY
    registry = json.loads(staged.read_text())
    assert sorted(registry['models']) == sorted(v for v in hc.model_ids(candidate).values() if v) and record['staged']['evaluated_members'] == {n: m['member_sha256'] for n, m in uf_world.evaluation['members'].items()}
    for subject in uf_world.subjects:
        prov = json.loads((tmp_path/candidate/'pred'/subject/'provenance.json').read_text())
        assert prov['segmenter_id'] == hc.CANDIDATES[candidate]['segmenter_id'] and prov['policy'] == 'all_folds_mean' and prov['registry_sha256'] == record['staged']['sha256'] and [m['model_id'] for m in prov['members']][:4] == [hc.model_ids(candidate)['A']]*2+[hc.model_ids(candidate)['B']]*2
        assert all(m['checkpoint_choice'] == 'last' for m in prov['members'][:4]) and 'best_iteration' not in prov['members'][0] and prov['allowlist_sha256'] == kwargs['allowlist'].sha256 and prov['version']['bayesreg']
        assert record['rows'][subject]['voxels_b_differs_from_a'] == uf_world.evaluation['per_subject_extra'][subject]['voxels_b_differs_from_a']
    check = hs_seg.segcheck(candidate, seg_dir=tmp_path/candidate, evaluation=uf_world.evaluation, subjects=uf_world.subjects, data_root=uf_world.s.data, n_subjects=len(uf_world.subjects), log=lambda _: None)
    assert reads and check['passed'] is True and check['n_rows_equal'] == len(uf_world.subjects) and check['mean_difference'] == dict(seg35=0., seg4=0.) and check['n_count_equal'] == len(uf_world.subjects)
    assert all(check['per_subject'][x]['truth_sha256'] == uf_world.evaluation['truth_sha256'][x] and check['per_subject'][x]['truth_same_as_uf'] is True for x in uf_world.subjects)
    with pytest.raises(FileExistsError):
        hs_seg.segcheck(candidate, seg_dir=tmp_path/candidate, evaluation=uf_world.evaluation, subjects=uf_world.subjects, data_root=uf_world.s.data, n_subjects=2, log=lambda _: None)
    with pytest.raises(FileExistsError):
        hs_seg.infer_all(candidate, **kwargs)
    out = tmp_path/'segmenters_hs_v1.json'
    registry_loaded = hs_seg.register(candidate, seg_dir=tmp_path/candidate, registry_path=out, subjects=uf_world.subjects)
    assert out.read_bytes() == staged.read_bytes() and registry_loaded.sha256 == record['staged']['sha256']
    hs_seg.register(candidate, seg_dir=tmp_path/candidate, registry_path=out, subjects=uf_world.subjects)      # the same bytes: accepted
    out.write_text(out.read_text()+' ')
    with pytest.raises(hs_seg.nx_seg.RegistrationRefused):
        hs_seg.register(candidate, seg_dir=tmp_path/candidate, registry_path=out, subjects=uf_world.subjects)


def test_a_control_that_does_not_reproduce_fails_and_nothing_registers(uf_world, tmp_path):
    kwargs = seg_kwargs(uf_world, 'UF4', tmp_path)
    hs_seg.infer_all('UF4', **kwargs)
    evaluation = json.loads(json.dumps(uf_world.evaluation))
    first = uf_world.subjects[0]
    evaluation['rows']['UF4'][first]['seg35'] += .01
    check = hs_seg.segcheck('UF4', seg_dir=tmp_path/'UF4', evaluation=evaluation, subjects=uf_world.subjects, data_root=uf_world.s.data, n_subjects=len(uf_world.subjects), log=lambda _: None)
    assert check['passed'] is False and check['mean_difference']['seg35'] == pytest.approx(-.01/len(uf_world.subjects)) and check['n_rows_equal'] == len(uf_world.subjects)-1
    with pytest.raises(hs_seg.HSSegFailed, match='segcheck'):
        hs_seg.register('UF4', seg_dir=tmp_path/'UF4', registry_path=tmp_path/'r.json', subjects=uf_world.subjects)
    assert not (tmp_path/'r.json').exists()


def test_a_control_on_other_truth_files_than_ufs_fails(uf_world, tmp_path):
    hs_seg.infer_all('UF4', **seg_kwargs(uf_world, 'UF4', tmp_path))
    evaluation = json.loads(json.dumps(uf_world.evaluation))
    evaluation['truth_sha256'][uf_world.subjects[1]]['seg4'] = '0'*64
    check = hs_seg.segcheck('UF4', seg_dir=tmp_path/'UF4', evaluation=evaluation, subjects=uf_world.subjects, data_root=uf_world.s.data, n_subjects=len(uf_world.subjects), evaluation_sha256='e'*64, log=lambda _: None)
    assert check['mean_difference'] == dict(seg35=0., seg4=0.) and check['passed'] is False and check['per_subject'][uf_world.subjects[1]]['truth_same_as_uf'] is False and check['eval_uf_sha256'] == 'e'*64
    (tmp_path/'UF4'/hc.SEGCHECK_RECORD).unlink()
    del evaluation['truth_sha256']                                   # eval_uf.json without the sha256 of the truth: not passed either (revision 1b, the follow-up's minor 3)
    check = hs_seg.segcheck('UF4', seg_dir=tmp_path/'UF4', evaluation=evaluation, subjects=uf_world.subjects, data_root=uf_world.s.data, n_subjects=len(uf_world.subjects), log=lambda _: None)
    assert check['mean_difference'] == dict(seg35=0., seg4=0.) and check['passed'] is False and not any(d['truth_same_as_uf'] for d in check['per_subject'].values())


def test_stage_refuses_members_other_than_eval_ufs(uf_world, tmp_path):
    kwargs = seg_kwargs(uf_world, 'UF4', tmp_path)
    evaluation = json.loads(json.dumps(uf_world.evaluation))
    evaluation['members']['foldB_s1']['member_sha256'] = '0'*64
    with pytest.raises(hs_seg.HSSegFailed, match='not those eval_uf.json evaluated'):
        hs_seg.stage('UF4', seg_dir=tmp_path/'s', uf_out=uf_world.out, evaluation=evaluation, split=kwargs['split'], allowlist=kwargs['allowlist'], training_source=kwargs['training_source'], pinned_folds_sha256=None, nn_out=uf_world.s.nn_out)
    hs_seg.stage('UF4', seg_dir=tmp_path/'t', uf_out=uf_world.out, evaluation=uf_world.evaluation, split=kwargs['split'], allowlist=kwargs['allowlist'], training_source=kwargs['training_source'], pinned_folds_sha256=None, nn_out=uf_world.s.nn_out)
    (tmp_path/'t'/hc.STAGED_REGISTRY).write_text('{}\n')
    with pytest.raises(hs_seg.HSSegFailed, match='written once'):
        hs_seg.stage('UF4', seg_dir=tmp_path/'t', uf_out=uf_world.out, evaluation=uf_world.evaluation, split=kwargs['split'], allowlist=kwargs['allowlist'], training_source=kwargs['training_source'], pinned_folds_sha256=None, nn_out=uf_world.s.nn_out)


def test_a_wrong_nnunet_file_stops_the_composer(uf_world, tmp_path, stage_nn_unchecked):
    kwargs = seg_kwargs(uf_world, 'UF4_nn', tmp_path)
    copy = tmp_path/'uf'
    shutil.copytree(uf_world.out, copy, symlinks=True, ignore=shutil.ignore_patterns('snapshot'))
    path = copy/'nn'/'probs'/uf_world.subjects[0]/'probs36.npz'
    path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(hs_seg.HSSegFailed, match="not the file UF's gate passed"):
        hs_seg.infer_all('UF4_nn', **dict(kwargs, uf_out=copy))


# ---- the queues on a fake machine ---------------------------------------------------------------------------------------------------------------
class FakeSystem:
    def __init__(self, *, behaviour=None, not_done=(), not_passed=()):
        self.launched, self.clock, self.behaviour, self.not_done, self.not_passed = [], 0., behaviour, set(not_done), set(not_passed)
        self.load, self.free_gpu, self.free_home, self.avail, self.free_host, self.other = .2, 23000., 500., 12., 70., []

    def load1(self):
        return self.load

    def units(self):
        return list(self.other)

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
        self.clock += seconds

    def code_checkout(self, root, commit):
        return Path(root)/'code', commit or 'fakesha'

    def run(self, argv, log):
        unit = next(a for a in argv if a.startswith('--unit=')).split('=', 1)[1]
        self.launched.append(unit)
        assert 'py'+'test' not in ' '.join(argv)
        root = HS.DEFAULT_ROOT
        if self.behaviour:
            rc = self.behaviour(unit, argv, self)
            if rc is not None:
                return rc
        if '--arm' in argv:
            cell = HS.require_cell(argv[argv.index('--arm')+1], argv[argv.index('--tier')+1], argv[argv.index('--pair')+1])
            if cell.is_repair:
                write_repair(root, cell.pair, evaluate=False, passed=cell.pair not in self.not_passed)
            else:
                write_field(root, cell.pair, evaluate=False, status='nonfinite' if cell.pair in self.not_done else 'done')
            return 0
        if unit in ('tag-hs-eval', 'tag-hs-eval-r'):
            for pair in ALL:
                d = HS.run_dir(root, HS.field_cell(pair))
                if (d/'run.json').is_file() and json.loads((d/'run.json').read_text())['status'] == 'done' and not (d/'metrics_0600.json').exists():
                    write_json(d/'metrics_0600.json', dict(oracle=False, certificate=dict(accepted_alpha=None, n_infeasible_at_1=9000, attempts=[dict(alpha=1., min_jd_lower_bound=-.1)]), metrics=metrics_of(.85), tapered_u_sha256='t', seconds_certificate=4., seconds_metrics=8.))
                r = HS.run_dir(root, HS.repair_cell(pair))
                if (r/'repair.json').is_file() and not (r/'repair_metrics.json').exists():
                    passed = json.loads((r/'repair.json').read_text())['passed']
                    write_json(r/'repair_metrics.json', dict(before=dict(metrics=metrics_of(.88), n_infeasible_at_1=1), after=dict(metrics=metrics_of(.88), certificate=dict(accepted_alpha=1. if passed else None, n_infeasible_at_1=0 if passed else 1, attempts=[dict(min_jd_lower_bound=1e-6 if passed else -.1)])), change=dict(), repaired_u_sha256='r'))
        if unit == 'tag-hs-fallback':
            for pair in ALL:
                if decision.needs_fallback(root, pair):
                    write_fallback(root, pair, output='identity')
        if unit == 'tag-hs-eval-id':
            for pair in RUN.identity_pairs(root):
                write_json(HS.id_root(root)/core.short_name(pair)/'identity_metrics.json', dict(certified_at_1=True, oracle=False, metrics=metrics_of(.6)))
        return 0


def options(phase, **kw):
    return RUN.argparse.Namespace(**dict(dict(phase=phase, root=str(HS.DEFAULT_ROOT), commit=None, wait_minutes=30, gap_seconds=5, min_host_gib=15.), **kw))


@pytest.fixture
def machine(results, monkeypatch):
    monkeypatch.setattr(fxrun, 'safe', lambda fn, out=None, **kw: [])            # the start checks have their own tests
    monkeypatch.setattr(RUN.BASE, 'committed_bytes', lambda sha, rel: HS.registry_path().read_bytes() if rel == hc.REGISTRY_REL and HS.registry_path().is_file() else (ROOT/rel).read_bytes())
    monkeypatch.setattr(fxrun, 'nx_commit', lambda commit: commit or 'fakesha')
    monkeypatch.setattr(RUN, 'code_drift', lambda seg_sha, sha: ([], [], None))       # its own tests below
    write_rw(results)
    write_json(results/'rw_v1'/'analysis'/'decision_rw.json', dict(reading=fxrun.decision.READ_RW_ALL))
    return results


def uf_done(results, candidate='UF4'):
    write_json(HS.UF_OUT/hc.UF_DECISION_REL, dict(reading='worth_hx24', reading_text='值得接进 Hx24', candidate=candidate, primary_arm=candidate, delta_seg35=.004, delta_seg4=.001))
    subjects = sorted({s for p in ALL for s in p.split('__')})
    write_json(HS.UF_OUT/hc.UF_EVAL_REL, dict(subjects=subjects, rows={candidate: {s: dict(seg35=.93) for s in subjects}}, reference_rows=dict(reference_v1={s: dict(seg35=.926) for s in subjects})))


def seg_done(results, candidate='UF4', passed=True, register=True):
    sd = hc.seg_dir(candidate, results)
    subjects = sorted({s for p in ALL for s in p.split('__')})
    write_json(sd/hc.INFER_RECORD, dict(candidate=candidate, subjects=subjects, rows={s: {} for s in subjects}, passed=True))
    write_json(sd/hc.SEGCHECK_RECORD, dict(candidate=candidate, passed=passed, mean_difference=dict(seg35=0., seg4=0.), n_rows_equal=20, n_count_equal=20, tolerance=2e-4))
    (sd/hc.STAGED_REGISTRY).write_text('{"staged": true}\n')
    if register:
        HS.registry_path().write_text('{"staged": true}\n')


def test_every_queue_waits_for_uf(machine):
    for phase in ('seg', 'fields'):
        system = FakeSystem()
        assert RUN.queue(options(phase), system) == 8
        assert system.launched == [] and '等 UF' in json.loads((HS.DEFAULT_ROOT/'queue_stop.json').read_text())['reason'] and 'nothing checked out' in json.loads((HS.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']


def test_a_reading_that_does_not_connect_runs_nothing(machine):
    write_json(HS.UF_OUT/hc.UF_DECISION_REL, dict(reading='not_connected', reading_text='不接', candidate=None))
    system = FakeSystem()
    assert RUN.queue(options('seg'), system) == 8 and system.launched == [] and '本包作废' in json.loads((HS.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']


def test_the_seg_queue_runs_the_inference_then_the_control_and_hands_over(machine):
    uf_done(machine)

    def behaviour(unit, argv, s):
        sd = hc.seg_dir('UF4', machine)
        if unit == 'tag-hs-infer':
            assert 'CUDA_VISIBLE_DEVICES' not in ' '.join(argv) and f'--setenv=PYTHONPATH={HS.DEFAULT_ROOT}/seg/code' in argv
            subjects = sorted({x for p in ALL for x in p.split('__')})
            write_json(sd/hc.INFER_RECORD, dict(candidate='UF4', subjects=subjects, rows={x: {} for x in subjects}, passed=True))
            return 0
        if unit == 'tag-hs-segcheck':
            assert '--setenv=CUDA_VISIBLE_DEVICES=' in argv
            write_json(sd/hc.SEGCHECK_RECORD, dict(candidate='UF4', passed=True, mean_difference=dict(seg35=0., seg4=0.), n_rows_equal=20, n_count_equal=20, tolerance=2e-4))
            return 0
    system = FakeSystem(behaviour=behaviour)
    assert RUN.queue(options('seg'), system) == 0
    assert system.launched == ['tag-hs-infer', 'tag-hs-segcheck'] and json.loads((HS.DEFAULT_ROOT/'seg_end.json').read_text())['segcheck_passed'] is True
    assert 'register' in (HS.DEFAULT_ROOT/'queue.log').read_text() and not HS.registry_path().exists() and (hc.seg_dir('UF4', machine)/RUN.INFER_LAUNCHED).is_file()


def test_a_launched_inference_that_left_no_record_is_not_launched_again(machine):
    uf_done(machine)
    system = FakeSystem(behaviour=lambda unit, argv, s: 1 if unit == 'tag-hs-infer' else None)
    assert RUN.queue(options('seg'), system) == 8 and system.launched == ['tag-hs-infer']
    system = FakeSystem()
    assert RUN.queue(options('seg'), system) == 8 and system.launched == [] and 'launched before' in json.loads((HS.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']


def seg_record_writer(machine, *, segcheck_rc=0):
    """A behaviour of the fake machine for the seg phase: the inference writes its record; the control writes its record unless it fails (rc 1, nothing written)."""
    def behaviour(unit, argv, s):
        sd = hc.seg_dir('UF4', machine)
        subjects = sorted({x for p in ALL for x in p.split('__')})
        if unit == 'tag-hs-infer':
            write_json(sd/hc.INFER_RECORD, dict(candidate='UF4', subjects=subjects, rows={x: {} for x in subjects}, passed=True))
            return 0
        if unit == 'tag-hs-segcheck':
            if segcheck_rc:
                return segcheck_rc
            write_json(sd/hc.SEGCHECK_RECORD, dict(candidate='UF4', passed=True, mean_difference=dict(seg35=0., seg4=0.), n_rows_equal=20, n_count_equal=20, tolerance=2e-4))
            return 0
    return behaviour


def test_a_launched_control_that_left_no_record_reads_the_truth_again_only_with_the_leads_words(machine):
    uf_done(machine)
    sd = hc.seg_dir('UF4', machine)
    system = FakeSystem(behaviour=seg_record_writer(machine, segcheck_rc=1))
    assert RUN.queue(options('seg'), system) == 8 and system.launched == ['tag-hs-infer', 'tag-hs-segcheck'] and (sd/RUN.SEGCHECK_LAUNCHED).is_file()
    system = FakeSystem(behaviour=seg_record_writer(machine))
    assert RUN.queue(options('seg'), system) == 8 and system.launched == [] and "needs the lead's words" in json.loads((HS.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']
    words = 'BZ 原话：再读一次（2026-11-06 3:00 am CST）'
    system = FakeSystem(behaviour=seg_record_writer(machine))
    assert RUN.queue(options('seg', segcheck_reread=words), system) == 0 and system.launched == ['tag-hs-segcheck']
    rereads = json.loads((sd/RUN.SEGCHECK_REREADS).read_text())
    assert len(rereads) == 1 and rereads[0]['words'] == words and (sd/rereads[0]['previous_launch']).is_file() and json.loads((sd/RUN.SEGCHECK_LAUNCHED).read_text())['approved_reread'] == words
    system = FakeSystem()
    assert RUN.queue(options('seg', segcheck_reread=words), system) == 8 and system.launched == [] and 'no launched control' in json.loads((HS.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']
    assert RUN.queue(options('fields', segcheck_reread=words), FakeSystem()) == 2


def test_the_inference_waits_for_21_gib_of_the_gpu(machine):
    uf_done(machine)
    system = FakeSystem(behaviour=seg_record_writer(machine))
    system.free_gpu = 21*1024-1.
    assert RUN.queue(options('seg', wait_minutes=0), system) == 8 and system.launched == [] and 'GPU free < 21504' in json.loads((HS.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']


def test_the_code_of_the_fields_is_the_seg_phases_but_for_the_registration(monkeypatch):
    def git_says(names):
        monkeypatch.setattr(rp, 'git', lambda *args, cwd=None: (0, '\0'.join(names)+'\0', '') if args[:4] == ('diff', '--no-renames', '--name-only', '-z') else (1, '', f'not the diff of revision 1b and HSF: {args}'))     # renames off: both paths of a move are listed; -z: not quoted (HSF)
    git_says([hc.REGISTRY_REL, 'runs/20261106-hs/plan.md', 'docs/decision_log.md', 'experiments/p2/p2_core.py', 'tests/test_bayesreg/solve/test_x.py'])
    changed, problems, approval = RUN.code_drift('a'*40, 'b'*40)
    assert len(changed) == 5 and problems == [] and approval is None
    for bad in ('experiments/fl/run_fl.py', 'bayesreg/seg/structure_layer.py', 'experiments/hs/hs_core.py', 'configs/bayesreg/dev_subjects.json', 'jd_curl/three_d/x.py', 'setup.py', 'experiments/p1/p1_core.py'):
        git_says([hc.REGISTRY_REL, bad])
        assert RUN.code_drift('a'*40, 'b'*40)[1] and bad in RUN.code_drift('a'*40, 'b'*40)[1][0]
    assert RUN.code_drift(None, 'b'*40)[1] == ['batch_seg.json has no code_sha: the seg phase has not run']
    monkeypatch.setattr(rp, 'git', lambda *args, cwd=None: (128, '', 'bad revision'))
    assert 'bad revision' in RUN.code_drift('a'*40, 'b'*40)[1][0]


# ---- the drift approval (HSF): a git repository of the test's own, real diffs ------------------------------------------------------------------------
APPROVABLE = 'experiments/hs/run_hs.py'
REVIEW = 'docs/bayesreg/reviews/HSF_复核.md'


def drift_repo(tmp_path):
    """(git, repo, the seg phase's commit): a fresh work tree (local identity, no hooks) with run_hs.py, hs_seg.py, the registry and the plan committed, as the seg phase's commit."""
    repo = tmp_path/'repo'
    repo.mkdir()

    def git(*args):
        return subprocess.run(['git', '-C', str(repo), '-c', 'user.name=test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null', *args], check=True, capture_output=True, text=True).stdout.strip()
    git('init', '-q')
    return git, repo, commit(git, repo, {APPROVABLE: 'line 1\nline 2\n', 'experiments/hs/hs_seg.py': 'seg 1\n', hc.REGISTRY_REL: '{}\n', 'runs/20261106-hs/plan.md': 'plan\n'})


def commit(git, repo, files):
    for rel, text in files.items():
        (repo/rel).parent.mkdir(parents=True, exist_ok=True)
        (repo/rel).write_text(text)
    git('add', '-A')
    git('commit', '-q', '-m', 'x')
    return git('rev-parse', 'HEAD')


def diff_sha(repo, a, b, rel):
    """The sha256 of the diff's bytes, computed here with the command written out (not RUN's helper)."""
    return hashlib.sha256(subprocess.run(['git', '-C', str(repo), 'diff', '--no-renames', '--full-index', a, b, '--', rel], check=True, capture_output=True).stdout).hexdigest()


def test_the_diff_of_an_entry_names_the_blobs_in_full(tmp_path):
    """The approval pins the diff with the blob ids in full: abbreviated, their length grows with the repository's object count, and so would the bytes (HSF)."""
    git, repo, seg = drift_repo(tmp_path)
    fix = commit(git, repo, {APPROVABLE: 'line 1\nline 2, fixed\n'})
    full = subprocess.run(['git', '-C', str(repo), 'diff', '--no-renames', '--full-index', seg, fix, '--', APPROVABLE], check=True, capture_output=True).stdout
    index = [line for line in full.decode().splitlines() if line.startswith('index ')]
    assert len(index) == 1 and len(index[0].split()[1]) == 2*40+2 and RUN.drift_diff_sha256(seg, fix, APPROVABLE, repo) == (hashlib.sha256(full).hexdigest(), None)


def approval_text(seg_sha, entries, **extra):
    return json.dumps(dict(dict(schema=RUN.DRIFT_APPROVAL_SCHEMA, seg_code_sha=seg_sha, entries=entries), **extra), indent=1, ensure_ascii=False)+'\n'


def test_the_drift_approval_lets_through_only_the_reviewed_diff_of_run_hs(tmp_path):
    git, repo, seg = drift_repo(tmp_path)
    fix = commit(git, repo, {APPROVABLE: 'line 1\nline 2, fixed\n', 'tests/test_x.py': 't\n', hc.REGISTRY_REL: '{"r": 1}\n'})
    changed, problems, approval = RUN.code_drift(seg, fix, repo=repo)                      # no approval: STOP as before
    assert approval is None and APPROVABLE in problems[0] and f'no {RUN.DRIFT_APPROVAL_REL} committed' in problems[0] and set(changed) == {APPROVABLE, 'tests/test_x.py', hc.REGISTRY_REL}
    entry = dict(path=APPROVABLE, diff_sha256=diff_sha(repo, seg, fix, APPROVABLE), review=REVIEW)
    (repo/RUN.DRIFT_APPROVAL_REL).write_text(approval_text(seg, [entry]))                 # only in the working tree: STOP
    changed, problems, approval = RUN.code_drift(seg, fix, repo=repo)
    assert approval is None and 'only in the working tree does not count' in problems[0]
    ok = commit(git, repo, {RUN.DRIFT_APPROVAL_REL: approval_text(seg, [entry]), REVIEW: 'the review\n'})
    changed, problems, approval = RUN.code_drift(seg, ok, repo=repo)                       # committed with the fields: through, and recorded
    assert problems == [] and RUN.DRIFT_APPROVAL_REL in changed and REVIEW in changed       # unquoted (-z): the review's name is not ASCII
    assert approval == dict(file=RUN.DRIFT_APPROVAL_REL, sha256=hashlib.sha256((repo/RUN.DRIFT_APPROVAL_REL).read_bytes()).hexdigest(), commit=ok, seg_code_sha=seg, entries=[entry], approved=[APPROVABLE], review_sha256={REVIEW: hashlib.sha256(b'the review\n').hexdigest()})
    more = commit(git, repo, {APPROVABLE: 'line 1\nline 2, fixed\nline 3\n'})               # one line more: STOP
    changed, problems, approval = RUN.code_drift(seg, more, repo=repo)
    assert approval is None and 'the sha256 of its diff' in problems[0] and APPROVABLE in problems[0]


@pytest.mark.parametrize('case', ('other_path', 'seg_sha', 'hs_seg_listed', 'hs_seg_unlisted', 'schema', 'extra_key', 'upper_hex', 'duplicate_key', 'not_json', 'no_entries', 'twice', 'empty_review'))
def test_a_drift_approval_that_does_not_hold_stops_as_before(tmp_path, case):
    git, repo, seg = drift_repo(tmp_path)
    files = {APPROVABLE: 'line 1\nline 2, fixed\n'}
    if case.startswith('hs_seg'):
        files['experiments/hs/hs_seg.py'] = 'seg 2\n'
    fix = commit(git, repo, files)
    entry = dict(path=APPROVABLE, diff_sha256=diff_sha(repo, seg, fix, APPROVABLE), review=REVIEW)
    seg_entry = dict(path='experiments/hs/hs_seg.py', diff_sha256=diff_sha(repo, seg, fix, 'experiments/hs/hs_seg.py'), review=REVIEW)
    text = dict(other_path=approval_text(seg, [entry, dict(entry, path='experiments/hs/hs_core.py')]), seg_sha=approval_text(fix, [entry]), hs_seg_listed=approval_text(seg, [entry, seg_entry]), hs_seg_unlisted=approval_text(seg, [entry]),
                schema=approval_text(seg, [entry]).replace('drift_approval.v1', 'drift_approval.v2'), extra_key=approval_text(seg, [entry], note='x'), upper_hex=approval_text(seg, [dict(entry, diff_sha256=entry['diff_sha256'].upper())]),
                duplicate_key=approval_text(seg, [entry]).replace('"schema"', f'"seg_code_sha": "{seg}", "schema"', 1), not_json='{"schema": ', no_entries=approval_text(seg, []), twice=approval_text(seg, [entry, entry]),
                empty_review=approval_text(seg, [dict(entry, review=' ')]))[case]
    if case == 'duplicate_key':
        assert json.loads(text)['seg_code_sha'] == seg                                      # json alone would take it
    ok = commit(git, repo, {RUN.DRIFT_APPROVAL_REL: text})
    changed, problems, approval = RUN.code_drift(seg, ok, repo=repo)
    word = dict(other_path='the whole approval is void', seg_sha="is not batch_seg.json's", hs_seg_listed='the whole approval is void', hs_seg_unlisted="no entry for ['experiments/hs/hs_seg.py']", schema='not the schema', extra_key='not the schema',
                upper_hex='an entry is not', duplicate_key='duplicate keys', not_json='not a JSON object', no_entries='not a non-empty list', twice='named twice', empty_review='an entry is not')[case]
    assert approval is None and len(problems) == 1 and problems[0].startswith(f'changed since the seg phase ({seg[:7]}): ') and word in problems[0], problems


def test_a_forbidden_path_that_git_would_quote_is_caught(tmp_path):
    """HSF: without -z, git quotes a path that is not ASCII ("experiments/hs/\\350..."), which no longer starts with its directory."""
    git, repo, seg = drift_repo(tmp_path)
    fix = commit(git, repo, {'experiments/hs/说明.txt': 'x\n', 'docs/说明.md': 'y\n'})
    changed, problems, approval = RUN.code_drift(seg, fix, repo=repo)
    assert approval is None and set(changed) == {'experiments/hs/说明.txt', 'docs/说明.md'} and "['experiments/hs/说明.txt']" in problems[0]


def test_drift_approval_prints_the_file_that_lets_the_fields_through(results, tmp_path, capsys):
    git, repo, seg = drift_repo(tmp_path)
    fix = commit(git, repo, {APPROVABLE: 'line 1\nline 2, fixed\n', REVIEW: 'the review\n'})
    write_json(HS.DEFAULT_ROOT/'batch_seg.json', dict(code_sha=seg))
    before = sorted(p.relative_to(results) for p in results.rglob('*'))
    o = types.SimpleNamespace(root=str(HS.DEFAULT_ROOT), review=REVIEW)
    assert RUN.print_drift_approval(o, repo=repo) == 0
    printed = capsys.readouterr()
    assert json.loads(printed.out) == dict(schema=RUN.DRIFT_APPROVAL_SCHEMA, seg_code_sha=seg, entries=[dict(path=APPROVABLE, diff_sha256=diff_sha(repo, seg, fix, APPROVABLE), review=REVIEW)]) and f'HEAD {fix}' in printed.err
    assert sorted(p.relative_to(results) for p in results.rglob('*')) == before and 'drift_approval.json' not in git('status', '--porcelain')       # nothing written
    ok = commit(git, repo, {RUN.DRIFT_APPROVAL_REL: printed.out})
    changed, problems, approval = RUN.code_drift(seg, ok, repo=repo)
    assert problems == [] and approval['approved'] == [APPROVABLE] and approval['commit'] == ok
    commit(git, repo, {'experiments/hs/hs_seg.py': 'seg 2\n'})                              # a change that may not be approved: exit 8, said on stderr
    assert RUN.print_drift_approval(o, repo=repo) == 8 and 'may not be approved' in capsys.readouterr().err
    assert RUN.print_drift_approval(types.SimpleNamespace(root=str(results/'other'), review=REVIEW), repo=repo) == 2


def test_the_fields_queue_records_the_drift_approval_and_stops_on_drift(machine, monkeypatch):
    uf_done(machine)
    seg_done(machine)
    record = dict(file=RUN.DRIFT_APPROVAL_REL, sha256='a'*64, commit='fakesha', seg_code_sha='s'*40, entries=[dict(path=APPROVABLE, diff_sha256='d'*64, review=REVIEW)], approved=[APPROVABLE], review_sha256={REVIEW: 'e'*64})
    monkeypatch.setattr(RUN, 'code_drift', lambda seg_sha, sha: ([APPROVABLE, 'tests/x.py'], ['changed since the seg phase (sssssss): [...]; not approved: no entry'], None))
    system = FakeSystem()
    assert RUN.queue(options('fields'), system) == 8 and system.launched == [] and 'not approved' in json.loads((HS.DEFAULT_ROOT/'queue_stop.json').read_text())['reason'] and not (HS.DEFAULT_ROOT/'batch.json').exists()
    monkeypatch.setattr(RUN, 'code_drift', lambda seg_sha, sha: ([APPROVABLE, 'tests/x.py'], [], record))
    system = FakeSystem()
    assert RUN.queue(options('fields'), system) == 0 and system.launched == fields_order()
    batch = json.loads((HS.DEFAULT_ROOT/'batch.json').read_text())
    assert batch['drift_approval'] == record and batch['changed_since_seg'] == [APPROVABLE, 'tests/x.py']


def test_the_control_unit_reads_the_truth_only_when_its_queue_launched_it(results, monkeypatch, capsys):
    """run_hs.py segcheck by hand: no segcheck_launched.json of the seg queue for this code, nothing is read (revision 1b, the follow-up's minor 1)."""
    monkeypatch.setattr(HS, 'candidate', lambda uf_out=None: 'UF4')
    called = []
    monkeypatch.setattr(RUN, 'seg_inputs', lambda source_root=None: (called.append('inputs'), (['s'], None, None, {}))[1])
    monkeypatch.setattr(hs_seg, 'segcheck', lambda *a, **k: (called.append('segcheck'), dict(passed=True))[1])
    monkeypatch.setattr(core, 'sha_file', lambda path: 'e'*64)
    write_json(HS.UF_OUT/hc.UF_EVAL_REL, dict(subjects=['s']))
    o = types.SimpleNamespace(code_sha='abc')
    assert RUN.segcheck_one(o) == 8 and called == [] and 'no segcheck_launched.json' in capsys.readouterr().err
    write_json(HS.seg_dir('UF4')/RUN.SEGCHECK_LAUNCHED, dict(unit=RUN.SEGCHECK_UNIT, code_sha='other'))
    assert RUN.segcheck_one(o) == 8 and called == []
    write_json(HS.seg_dir('UF4')/RUN.SEGCHECK_LAUNCHED, dict(unit=RUN.SEGCHECK_UNIT, code_sha='abc'))
    assert RUN.segcheck_one(o) == 0 and called == ['inputs', 'segcheck']


def test_the_control_unit_starts_once(results, monkeypatch, capsys):
    """segcheck_started.json is written once (O_EXCL) before any truth label: a second start of the unit by hand, with the same launch marker, reads nothing (revision 1c, minor 1)."""
    monkeypatch.setattr(HS, 'candidate', lambda uf_out=None: 'UF4')
    called = []
    monkeypatch.setattr(RUN, 'seg_inputs', lambda source_root=None: (called.append('inputs'), (['s'], None, None, {}))[1])
    monkeypatch.setattr(hs_seg, 'segcheck', lambda *a, **k: (called.append('segcheck'), dict(passed=False))[1])      # a control that crashed or failed leaves no way round
    monkeypatch.setattr(core, 'sha_file', lambda path: 'e'*64)
    write_json(HS.UF_OUT/hc.UF_EVAL_REL, dict(subjects=['s']))
    write_json(HS.seg_dir('UF4')/RUN.SEGCHECK_LAUNCHED, dict(unit=RUN.SEGCHECK_UNIT, code_sha='abc'))
    o = types.SimpleNamespace(code_sha='abc')
    assert RUN.segcheck_one(o) == 8 and called == ['inputs', 'segcheck'] and json.loads((HS.seg_dir('UF4')/RUN.SEGCHECK_STARTED).read_text())['code_sha'] == 'abc'
    assert RUN.segcheck_one(o) == 8 and called == ['inputs', 'segcheck'] and 'started before' in capsys.readouterr().err


def test_an_approved_reread_moves_the_start_aside_too(machine):
    uf_done(machine)
    sd = hc.seg_dir('UF4', machine)

    def crash(unit, argv, s):
        if unit == 'tag-hs-segcheck':
            (sd/RUN.SEGCHECK_STARTED).write_text('{}\n')                        # the unit started, then crashed
            return 1
        return seg_record_writer(machine)(unit, argv, s)
    assert RUN.queue(options('seg'), FakeSystem(behaviour=crash)) == 8
    assert RUN.queue(options('seg', segcheck_reread='BZ 原话'), FakeSystem(behaviour=seg_record_writer(machine))) == 0
    rereads = json.loads((sd/RUN.SEGCHECK_REREADS).read_text())
    assert not (sd/RUN.SEGCHECK_STARTED).exists() and (sd/rereads[0]['previous_start']).read_text() == '{}\n' and (sd/rereads[0]['previous_launch']).is_file()


def test_an_approved_reread_is_not_used_when_the_gate_stays_closed(machine):
    uf_done(machine)
    sd = hc.seg_dir('UF4', machine)
    system = FakeSystem(behaviour=seg_record_writer(machine, segcheck_rc=1))
    assert RUN.queue(options('seg'), system) == 8
    marker = (sd/RUN.SEGCHECK_LAUNCHED).read_text()
    system = FakeSystem(behaviour=seg_record_writer(machine))
    system.load = 2.
    assert RUN.queue(options('seg', segcheck_reread='BZ 原话', wait_minutes=0), system) == 8 and system.launched == [] and 'load' in json.loads((HS.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']
    assert (sd/RUN.SEGCHECK_LAUNCHED).read_text() == marker and not (sd/RUN.SEGCHECK_REREADS).exists() and not list(sd.glob('segcheck_launched_*.json'))      # nothing used, nothing renamed


def test_a_load_pair_refusal_in_the_repair_unit_is_an_input_error(results, monkeypatch):
    """repair_one is guarded as run_one: load_pair refusing the pair writes entry_refusal.json beside the repair; classify reads it refused, not a repair that crashed (revision 1b, the follow-up's minor 4)."""
    pair = ALL[4]
    root = HS.DEFAULT_ROOT
    write_field(root, pair)
    cell = HS.repair_cell(pair)
    d = HS.run_dir(root, cell)

    def refuse(c):
        raise SystemExit('the sha256 of pair.npz differs from the manifest')
    monkeypatch.setattr(HS, 'read_images', refuse)
    monkeypatch.setattr(RUN.BASE, 'repair_one', lambda o: HS.read_images(HS.require_cell(o.arm, o.tier, o.pair)))      # run_fl.repair_one reads the images (fl.read_images(cell)) before the field
    with pytest.raises(SystemExit, match='sha256 of pair.npz'):
        RUN.repair_one(types.SimpleNamespace(arm=cell.arm, tier=cell.tier, pair=pair, out=str(d), code_sha='s', device='cpu'))
    assert json.loads((d/hc.REFUSAL_FILE).read_text())['stage'] == 'load_pair' and HS.__dict__['read_images'] is refuse
    write_json(d/'queue_done.json', dict(run_id=cell.run_id, rc=1, unit_seconds=3., state='failed', arm_stop=None))
    kind, why = decision.classify(root, pair)
    assert kind == 'rejected' and '修复单元的读图被拒' in why and decision.method_row(root, pair)['line_status'] == hc.HX_REJECTED


def test_the_seg_queue_does_not_rerun_a_started_inference_and_a_failed_control_stops(machine):
    uf_done(machine)
    sd = hc.seg_dir('UF4', machine)
    (sd/'pred').mkdir(parents=True)
    system = FakeSystem()
    assert RUN.queue(options('seg'), system) == 8 and system.launched == [] and 'not rerun' in json.loads((HS.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']
    shutil.rmtree(sd/'pred')
    seg_done(machine, passed=False, register=False)
    system = FakeSystem()
    assert RUN.queue(options('seg'), system) == 8 and system.launched == [] and 'did not pass' in json.loads((HS.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']


@pytest.mark.parametrize('what,word', (('segcheck', '等分割'), ('registry', '等登记'), ('rw', '等 RW')))
def test_the_fields_queue_waits_for_each_step_by_name(machine, what, word):
    uf_done(machine)
    seg_done(machine)
    {'segcheck': hc.seg_dir('UF4', machine)/hc.SEGCHECK_RECORD, 'registry': HS.registry_path(), 'rw': machine/'rw_v1'/'analysis'/'decision_rw.json'}[what].unlink()
    system = FakeSystem()
    assert RUN.queue(options('fields'), system) == 8 and system.launched == [] and word in json.loads((HS.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']


def test_a_registry_that_is_not_the_staged_bytes_stops_before_the_checkout(machine):
    uf_done(machine)
    seg_done(machine)
    HS.registry_path().write_text('{"other": true}\n')
    system = FakeSystem()
    assert RUN.queue(options('fields'), system) == 8 and system.launched == [] and 'not the staged bytes' in json.loads((HS.DEFAULT_ROOT/'queue_stop.json').read_text())['reason']


def fields_order(fallback=False, identity=False, skip=()):
    return [HS.field_cell(p).unit for p in ALL]+['tag-hs-eval']+[HS.repair_cell(p).unit for p in ALL if p not in skip]+['tag-hs-eval-r']+(['tag-hs-fallback'] if fallback else [])+(['tag-hs-eval-id'] if identity else [])


def test_the_fields_queue_all_certified(machine, monkeypatch):
    uf_done(machine)
    seg_done(machine)
    system = FakeSystem()
    assert RUN.queue(options('fields'), system) == 0
    assert system.launched == fields_order()
    end = json.loads((HS.DEFAULT_ROOT/'queue_end.json').read_text())
    assert end['reading'] == hc.READ_ADOPT and end['counts']['n_fallback'] == 0
    p = json.loads((HS.DEFAULT_ROOT/'analysis'/'prediction_hs.json').read_text())
    assert p['pairs19']['seg_level_delta'] == pytest.approx(.004) and p['pairs19']['expected_delta'] == pytest.approx(.890*.004)
    batch = json.loads((HS.DEFAULT_ROOT/'batch.json').read_text())
    assert batch['candidate'] == 'UF4' and batch['segmenter_id'] == 'unet35_v4_ens' and batch['caps']['repair_wall_seconds'] == 1200. and len(batch['rw_pins']) == 19 and batch['changed_since_seg'] == [] and batch['drift_approval'] is None and batch['uf_eval_sha256'] == core.sha_file(HS.UF_OUT/hc.UF_EVAL_REL)
    assert p['eval_uf_sha256'] == core.sha_file(HS.UF_OUT/hc.UF_EVAL_REL)


def test_the_fields_queue_with_a_failed_entry_and_a_failed_repair_runs_the_fallbacks(machine):
    uf_done(machine)
    seg_done(machine)
    system = FakeSystem(not_done={ALL[3]}, not_passed={ALL[7]})
    assert RUN.queue(options('fields'), system) == 0
    assert system.launched == fields_order(fallback=True, identity=True, skip=(ALL[3],))
    d = json.loads((HS.DEFAULT_ROOT/'analysis'/'decision_hs.json').read_text())
    assert d['counts']['by_status'][hc.HX_IDENTITY] == [core.short_name(ALL[3]), core.short_name(ALL[7])] and d['counts']['n_entry_failure'] == 1 and d['reading'] == hc.READ_NOT


def test_a_stopped_queue_leaves_the_rest_not_run_and_the_reading_undecidable(machine):
    uf_done(machine)
    seg_done(machine)
    system = FakeSystem(behaviour=lambda unit, argv, s: 1 if unit.startswith('tag-hs-shx24') and unit.endswith(('0440-0441', '0441-0442', '0442-0443')) else None)
    assert RUN.queue(options('fields'), system) == 8
    d = json.loads((HS.DEFAULT_ROOT/'analysis'/'decision_hs.json').read_text())
    assert d['reading'] == hc.UNDECIDABLE and '没有运行' in d['reason']


# ---- the checks before the fields ----------------------------------------------------------------------------------------------------------------------
def image_sha(subject):
    """The sha256 a synthetic manifest names for the source image of `subject` (64 hex, as the firewall's reader requires)."""
    return hashlib.sha256(f'the source image of {subject}'.encode()).hexdigest()


def write_manifests(stage, pairs=ALL):
    """<stage>/<pair>/manifest.json in the mainline format (test_d0_a4_tools.write_stage's keys, with label-derived ones the firewall drops), for the real bayesreg.io.pair._read_manifest (HSF: no stand-in of its return value)."""
    for pair in pairs:
        fixed, moving = pair.split('__')
        write_json(Path(stage)/pair/'manifest.json', dict(pair=pair, fixed_subject=fixed, moving_subject=moving, shape_zyx=[5, 6, 7], spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm',
                                                          sources=dict(fixed=dict(sha256=image_sha(fixed)), moving=dict(sha256=image_sha(moving)), fixed_seg35=dict(sha256='c'*64)),
                                                          pair_sha256=hashlib.sha256(pair.encode()).hexdigest(), baseline_label_metrics=dict(seg35=.5), preprocessing='synthetic'))
    return Path(stage)


def fields_world(results, monkeypatch, tmp_path, candidate='UF4'):
    """The 20 predictions as the seg phase leaves them (files of a few bytes), Atropos beside, a synthetic stage whose manifest.json files the real firewall reader reads (HSF: the stand-in of its return value had another shape than the reader)."""
    subjects = sorted({s for p in ALL for s in p.split('__')})
    sd = hc.seg_dir(candidate, results)
    write_json(sd/hc.INFER_RECORD, dict(candidate=candidate, subjects=subjects, rows={s: {} for s in subjects}, passed=True))
    write_json(sd/hc.SEGCHECK_RECORD, dict(candidate=candidate, passed=True, mean_difference=dict(seg35=0., seg4=0.)))
    atropos = tmp_path/'atropos'
    for subject in subjects:
        d = sd/'pred'/subject
        write_json(d/'provenance.json', dict(segmenter_id=hc.CANDIDATES[candidate]['segmenter_id'], policy='all_folds_mean', test_labels_seen=False, registry_sha256='r'*64, input_sha256={hs_seg.IMAGE_FILE: image_sha(subject)}))
        (d/'probs36.npz').write_bytes(b'x')
        (d/'structure_groups.npz').write_bytes(b'x')
        write_json(atropos/subject/'provenance.json', {})
    stage = write_manifests(tmp_path/'stage')
    return sd, atropos, subjects, stage


def test_the_predictions_against_the_real_readers_manifest(results, monkeypatch, tmp_path):
    """HSF (the fields queue STOPped at 10:20:42 am CST 11-06, KeyError: 'sources'): the firewall's reader gives source_sha256 = {side: sha256}, not sources; prediction_problems on its real output.
    Before the fix this raised KeyError; after it, no problem, and a prediction of another image is named."""
    sd, atropos, subjects, stage = fields_world(results, monkeypatch, tmp_path)
    import bayesreg.io.pair as pair_module
    _, _, selected = pair_module._read_manifest(stage/ALL[0]/'manifest.json')
    fixed, moving = ALL[0].split('__')
    assert sorted(selected) == ['fixed_subject', 'moving_subject', 'pair_sha256', 'source_sha256'] and selected['source_sha256'] == dict(fixed=image_sha(fixed), moving=image_sha(moving))

    def check():
        return RUN.prediction_problems(lambda text: None, 'UF4', seg_root=atropos, registry_sha256='r'*64, stage=stage)
    assert check() == []
    path = sd/'pred'/subjects[5]/'provenance.json'
    write_json(path, dict(json.loads(path.read_text()), input_sha256={hs_seg.IMAGE_FILE: image_sha('another subject')}))
    assert check() == [f'{subjects[5]}: the prediction was made from another image than the manifests name']
    write_json(path, dict(json.loads(path.read_text()), input_sha256={hs_seg.IMAGE_FILE: image_sha(subjects[5])}))
    manifest = stage/ALL[0]/'manifest.json'
    document = json.loads(manifest.read_text())
    write_json(manifest, dict(document, sources=dict(document['sources'], moving=dict(sha256=image_sha('another subject')))))
    assert check() == [f'{moving}: the prediction was made from another image than the manifests name']        # the manifests of the subject disagree with its prediction


def test_the_predictions_checked_before_the_fields(results, monkeypatch, tmp_path):
    sd, atropos, subjects, stage = fields_world(results, monkeypatch, tmp_path)

    def check(sha='r'*64):
        return RUN.prediction_problems(lambda text: None, 'UF4', seg_root=atropos, registry_sha256=sha, stage=stage)
    assert check() == []
    assert check('s'*64) == [f'{s}: the prediction is not the registered candidate' for s in subjects]
    path = sd/'pred'/subjects[3]/'provenance.json'
    write_json(path, dict(json.loads(path.read_text()), input_sha256={hs_seg.IMAGE_FILE: 'other'}))
    assert check() == [f'{subjects[3]}: the prediction was made from another image than the manifests name']
    write_json(path, dict(json.loads(path.read_text()), input_sha256={hs_seg.IMAGE_FILE: image_sha(subjects[3])}, policy='cross_fit_other_folds'))
    assert check() == [f'{subjects[3]}: the prediction is not the registered candidate']
    write_json(path, dict(json.loads(path.read_text()), policy='all_folds_mean'))
    (sd/'pred'/subjects[0]/'structure_groups.npz').unlink()
    shutil.rmtree(atropos/subjects[1])
    assert check() == [f"{subjects[0]}: missing ['structure_groups.npz']", f"{subjects[1]}: missing ['Atropos {atropos/subjects[1]}']"]
    write_json(sd/hc.SEGCHECK_RECORD, dict(candidate='UF4', passed=False))
    assert any('segcheck.json' in p for p in check())


def test_the_registry_checked_before_the_fields(results, monkeypatch, tmp_path):
    contact = sorted(a4.FOLD0+a4.FOLD1)
    members = [dict(fold=f, seed=s, member_sha256=f'{k:x}'*64, label_contact_subjects=contact) for k, (f, s) in enumerate(hc.UF_MEMBERS, 1)]
    text = hs_seg.document_text(hs_seg.registry_document('UF4', members=members))
    registry = tmp_path/'repo'/'configs'/'segmenters_hs_v1.json'
    registry.parent.mkdir(parents=True)
    monkeypatch.setattr(HS, 'registry_path', lambda repo=None: registry)
    sd = hc.seg_dir('UF4', results)
    sd.mkdir(parents=True)
    (sd/hc.STAGED_REGISTRY).write_text(text)
    assert RUN.registry_lines(lambda t: None, 'UF4') == ['等登记：没有 '+hc.REGISTRY_REL]
    registry.write_text(text)
    a4.commit_registry(registry)
    assert any('FORMAL_REGISTRIES' in p for p in RUN.registry_lines(lambda t: None, 'UF4'))
    monkeypatch.setattr(structure_layer, 'FORMAL_REGISTRIES', structure_layer.FORMAL_REGISTRIES+(registry,))
    assert RUN.registry_lines(lambda t: None, 'UF4') == []
    assert any('is not the candidate UF4_nn' in p for p in RUN.registry_lines(lambda t: None, 'UF4_nn'))
    registry.write_text(text+' ')
    problems = RUN.registry_lines(lambda t: None, 'UF4')
    assert any('committed and clean' in p for p in problems) and any('not the staged bytes' in p for p in problems)


# ---- the start strings and the dry run ---------------------------------------------------------------------------------------------------------------
def test_the_start_strings():
    for phase in ('seg', 'fields'):
        argv = RUN.queue_launch(phase)
        assert argv[:4] == ['systemd-run', '--user', '--unit=tag-hs-queue', '--collect'] and f'--working-directory={core.MAIN_REPO}' in argv and argv[-8:-6] == ['--phase', phase] and 'py'+'test' not in ' '.join(argv)
    infer = RUN.seg_unit_argv(RUN.INFER_UNIT, Path('/c'), ['infer-one'], log=Path('/l'), gpu=True, runtime=RUN.INFER_RUNTIME)
    assert '--setenv=PYTHONPATH=/c' in infer and not any(a.startswith('--setenv=CUDA_VISIBLE_DEVICES') for a in infer) and '--property=MemoryMax=8G' in infer
    check = RUN.seg_unit_argv(RUN.SEGCHECK_UNIT, Path('/c'), ['segcheck'], log=Path('/l'), gpu=False, runtime=RUN.SEGCHECK_RUNTIME)
    assert '--setenv=CUDA_VISIBLE_DEVICES=' in check
    fallback = RUN.BASE.unit_argv(RUN.FALLBACK_UNIT, Path('/c'), ['fallback', '--root', '/r'], log=Path('/l'), gpu=False)
    assert f'--setenv=PYTHONPATH={core.FROZEN}:/c' in fallback and '--setenv=CUDA_VISIBLE_DEVICES=' in fallback
    with pytest.raises(ValueError):
        MC0.check_launch(['x', 'py'+'test'])


def test_the_seg_inputs_are_pinned_and_verified_against_the_main_checkout(monkeypatch):
    import bayesreg.seg.unet.pins as pins
    import eval_seg
    seen = {}

    def pinned(folds, allowlist, **kw):
        seen.update(kw, folds=folds, allowlist=allowlist)
        return 'split', 'allow', dict(code='c')
    monkeypatch.setattr(pins, 'load_pinned_full_inputs', pinned)
    subjects = sorted({s for p in ALL for s in p.split('__')})
    monkeypatch.setattr(eval_seg, 'admit_subjects', lambda allowlist, source, sealed, training: (seen.update(source=source), list(reversed(subjects)))[1])
    got = RUN.seg_inputs()
    assert got[0] == subjects and seen['source_root'] == core.MAIN_REPO and seen['source'] == 'stage_L2R_val_v1' and seen['folds'] == pins.DEFAULT_FULL_FOLDS and seen['expect_counts'] == (98, 8)
    monkeypatch.setattr(eval_seg, 'admit_subjects', lambda *a: subjects[:19])
    with pytest.raises(hs_seg.HSSegFailed, match='not the 20'):
        RUN.seg_inputs()


def test_the_dry_run_runs_the_fields_input_check_once_the_seg_phase_is_done_and_the_registry_committed(results, monkeypatch, tmp_path):
    """HSF: the dry run did not call fields_input_problems, so the KeyError was first met by the queue.  Now: before the seg phase is done or the registry committed, what it waits for (not a problem);
    after, fields_input_problems as the queue runs it, every line and problem printed and counted, an exception a problem as in the queue (fxrun.safe)."""
    calls = []

    def check(out, **kw):
        calls.append(kw)
        out('a line of the check')
        return ['problem 1', 'problem 2']
    monkeypatch.setattr(RUN, 'fields_input_problems', check)
    registry = tmp_path/'repo'/'configs'/'segmenters_hs_v1.json'
    monkeypatch.setattr(HS, 'registry_path', lambda repo=None: registry)
    lines = []
    assert RUN.fields_check_lines(lines.append, 'UF4') == [] and calls == [] and any(line.startswith('等分割') for line in lines) and any(line.startswith('等登记') for line in lines)
    sd = hc.seg_dir('UF4', results)
    write_json(sd/hc.INFER_RECORD, dict(passed=True))
    write_json(sd/hc.SEGCHECK_RECORD, dict(passed=False))
    lines.clear()
    assert RUN.fields_check_lines(lines.append, 'UF4') == [] and calls == [] and any(line.startswith('等分割') for line in lines)
    write_json(sd/hc.SEGCHECK_RECORD, dict(passed=True))
    registry.parent.mkdir(parents=True)
    registry.write_text('{}\n')
    lines.clear()
    assert RUN.fields_check_lines(lines.append, 'UF4') == [] and calls == [] and [line[:3] for line in lines] == ['等登记'] and 'tracked False' in lines[0]      # there, not committed
    a4.commit_registry(registry)
    lines.clear()
    assert RUN.fields_check_lines(lines.append, 'UF4', stage='S', allowlist='A', results='R') == ['problem 1', 'problem 2'] and calls == [dict(candidate='UF4', stage='S', allowlist='A', results='R')]
    assert lines[1:] == ['[fields inputs] a line of the check', '[fields inputs] problem: problem 1', '[fields inputs] problem: problem 2', "--- fields_input_problems: 2 problem(s): ['problem 1', 'problem 2']"] and "input check" in lines[0]
    monkeypatch.setattr(RUN, 'fields_input_problems', lambda out, **kw: {}['sources'])
    lines.clear()
    assert RUN.fields_check_lines(lines.append, 'UF4') == ["KeyError: 'sources'"]          # the STOP of 10:20:42 am CST, now a problem of the dry run
    monkeypatch.setattr(RUN, 'fields_input_problems', lambda out, **kw: [])
    lines.clear()
    assert RUN.fields_check_lines(lines.append, 'UF4') == [] and lines[-1] == '--- fields_input_problems: 0 problem(s) (none)'


def test_a_queue_on_another_root_touches_nothing(tmp_path, capsys):
    assert RUN.queue(options('seg', root=str(tmp_path/'hs_other'))) == 2 and not (tmp_path/'hs_other').exists()


def test_the_dry_run_in_its_own_process():
    there = HS.DEFAULT_ROOT.exists()
    shim = Path(tempfile.mkdtemp(prefix='hsdry_'))
    (shim/'systemd-run').write_text('#!/bin/sh\nexit 99\n')
    (shim/'systemd-run').chmod(0o755)
    environ = dict(os.environ, PATH=f"{shim}:{os.environ['PATH']}", CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    try:
        done = subprocess.run([PY, '-B', str(SCRIPT), '--dry-run'], env=environ, capture_output=True, text=True, timeout=240, cwd=str(ROOT))
    finally:
        shutil.rmtree(shim)
    text = done.stdout
    assert done.returncode == 0, text[-3000:]+done.stderr[-2000:]
    assert 'torch imported by this dry run: False; problems: none' in text and 'candidate UF4:' in text and 'candidate UF4_nn:' in text and 'uf_nn_v1_nnall (fold 2' in text and '--phase seg' in text and '--phase fields' in text and '--unit=tag-hs-eval-r ' in text
    if not hc.uf_paths()['decision'].is_file():
        assert '等 UF' in text
    if 'UF 的候选：' in text:                                         # HSF: the fields queue's input check runs, or what it waits for is printed
        assert ("--- the fields queue's input check" in text and '--- fields_input_problems: 0 problem(s) (none)' in text and '[fields inputs] seg phase: infer_record.json passed True' in text) or '还没到场的输入核对' in text
    assert HS.DEFAULT_ROOT.exists() == there                         # the dry run creates no root
