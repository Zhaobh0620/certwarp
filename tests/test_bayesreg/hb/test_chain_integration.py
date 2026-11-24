"""The whole hold-out chain in one sandbox (runs/20261110-a5/任务书.md item 9; A5b item 15; A5b review I3 and minors 1, 2): each step reads only what the step before it wrote --
the stage (HP's own run, experiments/holdout/prep_stage.run_holdout, through SG's unseal object), the method arm's queue (HE played by test_hb_queues' FakeSystem, which writes HE's
own files through he_core.write_output and HE's t88/repair records), the step-5 review and its finalized record (hb_review.queue), the move to D: (hb_store.move_pair, move_resume),
the configuration C arm (hb_configc.run with test_hb_configc's stand-ins of the frozen run), the external arms of the frozen main family (xb_drive.run_cell with the XB world's
FakeMethod; ConvexAdam-v4seg on HI's predictions and the method's pinned T88 run.json), the step-8 input lock (hb_store.input_lock), the repair ablation's preparation
(hb_prerepair.run), BZ's (ii) record, the D108 manifest (the lead's step: the entry's own input_roles), the evaluation entry (bayesreg.eval.holdout, with rule one on a committed
runs/holdout_B_v1/pinned_code.json) and the decision code (holdout_decide).  The arm table is configs/frozen/holdout_B_v1_arm_table.json with its roots moved (method and main family).
Paths taken: configuration C and four XB arms; the method's fallback one; an infra_failure of configuration C resumed; a stopped XB arm (P19); the step-5 (甲) and a withdrawal;
a move-resume; the method's P20 written by BZ's no_resume; ConvexAdam-v4seg's P20 on pairs without a method output (refused, rerun, refused, BZ's no_resume of the newest copy);
a configuration C needs_lead_rejudgment written by a rejudgment record; HI resumed into its one root; F4's (ii).  PQ's two units in the hold-out mode and the entry's reading of their
outputs (release, pq_states, Ev1, Ev2, the input-lock check): tests/test_bayesreg/pqh/test_pqh_to_entry.py (PQH's hold-out world: its code tree, registry and predictions).
Every interface has a case where one side changes one field name (or path) and the chain fails with a message that names the renamed key read as missing at its place.
HB's sandbox (hb/conftest.py: every root under a tempfile.mkdtemp(prefix='xbt_') directory, XB_TEST_SANDBOX set, systemd-run a shim exiting 99); real hold-out ids (the pair
list's first rows; ids only) with synthetic arrays and labels; SG's real loaders on the simulated unseal branch."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.nifti import file_sha256, save_scalar

from bayesreg.eval import holdout as H
from bayesreg.eval import holdout_decide as D
from bayesreg.io import code_identity as CI
from bayesreg.io import rejudgment as RJ

from ..pq import _world as PW
from . import _world as W
from .test_hb_configc import Launcher, fake_prepare, kernel, probe_of
from .test_hb_prerepair import overwrite, repair_inputs
from .test_hb_queues import FakeSystem

R, he, hc, hb, review, hm, XW = W.R, W.he, W.hc, W.hb, W.hb_review, W.hb_method, W.XW
import hb_configc as C  # noqa: E402
import hb_prerepair as P  # noqa: E402
import hb_store  # noqa: E402
import xb_core as xc  # noqa: E402

FROZEN_TABLE = Path(H.REPO)/'configs'/'frozen'/'holdout_B_v1_arm_table.json'   # A5b review minor 2: the chain's arm table is the frozen one (its roots moved into the sandbox)
REAL_HOME, REAL_BIG = '/home/rmcfz/seq_vp_tag_results/holdout_B_v1', '/mnt/d/seq_vp_tag_holdout_B_v1'
XB_MAIN = ('ants_s', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg')         # the frozen main family's four XB arms (ants_s run by the FakeMethod through arm_py, as test_xb_run does)
REPLAY_PASS = '逐位相同 19/19'


def quiet(*a, **k):
    pass


def no_units(argv, **kwargs):
    import subprocess
    return subprocess.CompletedProcess(argv, 0, '', '')


class ChainSystem(FakeSystem):
    """FakeSystem's HE stand-in, and HE's t88/run.json and repair/repair.json as HE writes them for a pair whose T88 finished and whose repair child got past source_checked; a pair both of whose
    subjects have HI labels (the chain's ConvexAdam-v4seg pair): run.json names labels_sha256 of HI's argmax (ConvexAdam-v4seg checks its maps against them, prereg §2.4)."""

    def __init__(self, outcomes, *, labels=None, **kwargs):
        super().__init__(outcomes, **kwargs)
        self.labels = dict(labels or {})

    def method(self, pair, outcome):
        super().method(pair, outcome)
        if outcome.get('line') in he.OUTPUT_STATUSES:
            repair_inputs(pair)
            fixed, moving = pair.split('__')
            if fixed in self.labels and moving in self.labels:
                path = hb.cell(pair)/'t88'/'run.json'
                doc = json.loads(path.read_text())
                doc['labels_sha256'] = xc.labels_digest({'fixed': np.asarray(self.labels[fixed], np.int16), 'moving': np.asarray(self.labels[moving], np.int16)})
                overwrite(path, doc)


def oasis_labels(seed, lookup, shape):
    rng = np.random.default_rng(seed)
    seg35 = np.zeros(shape)
    seg35[1:-1, 1:-1, 1:-1] = rng.integers(1, 36, (shape[0]-2, shape[1]-2, shape[2]-2))
    return dict(seg35=seg35, seg4=lookup[seg35.astype(int)].astype(np.float64))


class Chain:
    """One hold-out of `n` pairs in HB's sandbox; the steps in order (each a method); `hooks`: {point: callable(chain)} run at named points (the interface cases)."""

    def __init__(self, tmp_path, monkeypatch, *, n=2, hooks=None):
        self.tmp, self.mp, self.hooks = Path(tmp_path), monkeypatch, dict(hooks or {})
        self.pairs = [hc.pair_name(*r) for r in XW.holdout_rows()[:n]]
        self.rows = tuple((i, *p.split('__'), p) for i, p in enumerate(self.pairs))
        mp = monkeypatch
        mp.setattr(R.core, 'frozen_api', lambda: SimpleNamespace(boundary_taper=PW.frozen_formula_taper))
        mp.setattr(hm, 'PREREG_SHA256', hb.sha(hb.REPO/hm.PREREG_REL))
        mp.setattr(hb, 'code_record', lambda commit=None, run=None: dict(commit='1'*40, head='1'*40, clean=True, pinned_code=dict(stub='HB, XB: rule one stubbed; the entry runs it')))
        mp.setattr(xc, 'PREREG_SHA256', xc.sha256(xc.REPO/xc.PREREG_REL))
        XW.fake_checks(monkeypatch)
        mp.setitem(xc.ARM_FUNCTIONS, 'ants', xc.ARM_FUNCTIONS['py'])                 # ants_s through arm_py and the FakeMethod (test_xb_run's way with niftyreg)
        mp.setitem(xc.ARMS, 'ants_s', dict(xc.ARMS['ants_s'], script='harness/m_convexadam.py', args=[], post_args=[], env='bl_convexadam'))
        mp.setattr(C.fc, 'prepare', fake_prepare)
        mp.setattr(he, 'frozen_identity', lambda *a, **k: (dict(test=True), []))
        self.sim = XW.simulated_unseal(self.tmp, monkeypatch)
        self.repo = self.sim.repo
        mp.setattr(hc, 'AUTH_ROOT', self.repo)
        mp.setattr(hb, 'rows', lambda: list(self.pairs))
        self.stage_record = self.prepare_stage()
        # HI's hold-out inference resumed into its one root (A5a review I2; A5b item 13): the first run killed after about half the subjects (the first pair's fixed subject among them), the
        # first pair's moving subject left half written (moved to attempt1/pred/ before the resume), the resume writing the rest; the first pair's two predictions real (ConvexAdam-v4seg reads them)
        everyone = sorted({s for r in XW.holdout_rows() for s in r})
        f1, m1 = self.pairs[0].split('__')
        mine = sorted({s for p in self.pairs for s in p.split('__')})           # ConvexAdam-v4seg reads the chain pairs' predictions: real probs36 for them
        self.labels = {s: np.random.default_rng(20+i).integers(0, 36, XW.SHAPE) for i, s in enumerate(mine)}
        first = [s for i, s in enumerate(everyone) if s != m1 and (s == f1 or i % 2 == 0)]
        self.hi_root, self.hi_names = W.write_hi(everyone, first=first, residue=[m1], labels=self.labels)
        self.k = {}
        self.pin_code()
        self.patch_entry()

    def prepare_stage(self):
        """Step 1 (prereg §5.2; A5b review I3): the hold-out stage by HP's own run, experiments/holdout/prep_stage.run_holdout -- the admission through SG's unseal object (the
        simulated unseal branch), the lock, HP's spy, write_pair's pair.npz (fixed, moving, initial_u) and manifest.json for every row of the pair list -- from synthetic aligned_norm images of
        the 100 subjects at the chain's grid (bayesreg.seg.unet.synthetic.write_subjects; the chain's truth labels are another data root, written by patch_entry).  Stood in: the
        development check's decision (tests/test_bayesreg/hp/_world.decide), HP's rule one and two (the chain's code tree is not HP's checkout), the machine gate."""
        import hp_core
        import prep_stage
        from bayesreg.seg.unet import synthetic
        from ..hp import _world as HPW
        mp, data = self.mp, self.tmp/'hp_data'
        synthetic.write_subjects(data, sorted({s for r in XW.holdout_rows() for s in r}), shape_xyz=tuple(XW.SHAPE[::-1]))
        mp.setattr(prep_stage, 'EXPECTED_SHAPE_ZYX', tuple(XW.SHAPE))
        mp.setattr(hp_core, 'gate_identity', lambda commit, **kw: (dict(passed=True, gate=commit, note='chain test: HP rule two stood in'), []))
        mp.setattr(hp_core, 'pinned_identity', lambda: (dict(passed=True, pinned=HPW.CODE['commit'], note='chain test: HP rule one stood in'), []))
        mp.setattr(hc, 'machine_gate', lambda **kw: dict(passed=True, note='chain test: no machine queried', **kw))
        HPW.decide(mp)
        tsv = self.repo/hc.PAIRS_TSV_REL                                 # the unseal branch holds the pair list: its bytes copied as the sealed lists are (ids only)
        tsv.parent.mkdir(parents=True, exist_ok=True)
        tsv.write_bytes((Path(hb.REPO)/hc.PAIRS_TSV_REL).read_bytes())
        self.commit('the pair list on the unseal branch (HP reads it from the sealed tree)')
        XW.sealed_module._reset_cache()
        record = prep_stage.run_holdout(unseal_path=self.sim.record, code=HPW.CODE, data_root=data, log=quiet)
        XW.sealed_module._reset_cache()
        assert record['passed'] and record['stage_summary']['pairs'] == len(XW.holdout_rows()), record.get('stopped')
        return record

    def hook(self, name):
        if name in self.hooks:
            self.hooks[name](self)

    # the code tree: the arm table, the preregistration, the pair list (before P1), then runs/holdout_B_v1/pinned_code.json (C24.3-2) --------------------------------------------
    def arm_table(self):
        """The frozen arm table (configs/frozen/holdout_B_v1_arm_table.json; A5b review minor 2) with its two roots moved into the sandbox (hc.HOLDOUT_ROOT, hc.BIG_ROOT): the method arm and
        the main family, every key as frozen; the seven descriptive arms and the three of the 90 s tier are left out (the chain runs no cell of theirs; their specs are XB's with another
        name and family: test_the_chain_s_arm_table_is_the_frozen_one checks it)."""
        text = FROZEN_TABLE.read_text(encoding='utf-8').replace(REAL_HOME, str(hc.HOLDOUT_ROOT)).replace(REAL_BIG, str(hc.BIG_ROOT))
        table = json.loads(text)
        keep = {'method', *table['main_family']}
        table['arms'] = [a for a in table['arms'] if a['name'] in keep]
        assert table['stage_root'] == str(hc.STAGE_ROOT) and {a['name'] for a in table['arms']} == {'method', 'configc', *XB_MAIN}
        return table

    def git(self, *args):
        return XW.git(self.repo, *args)

    def commit(self, message):
        self.git('add', '-A')
        self.git('commit', '-q', '--allow-empty', '-m', message)

    def head(self):
        return self.git('rev-parse', 'HEAD').stdout.strip()

    def pin_code(self):
        self.table = self.arm_table()
        (self.repo/H.ARM_TABLE_REL).parent.mkdir(parents=True, exist_ok=True)
        (self.repo/H.ARM_TABLE_REL).write_text(json.dumps(self.table, indent=1), encoding='utf-8')
        self.table_sha = file_sha256(self.repo/H.ARM_TABLE_REL)
        (self.repo/H.PREREG_REL).parent.mkdir(parents=True, exist_ok=True)
        (self.repo/H.PREREG_REL).write_text(f'# synthetic preregistration of the chain test\n\narm table sha256 {self.table_sha}\n', encoding='utf-8')
        (self.repo/H.PAIRS_REL).write_text('fixed\tmoving\n'+''.join(f'{r[1]}\t{r[2]}\n' for r in self.rows), encoding='utf-8')
        # runs/ is a top-level directory of the real tree at P1; rule one reads a top-level directory P1 does not have as code (C24.3-3): the records under runs/holdout_B_v1/ need it there
        (self.repo/'runs').mkdir(exist_ok=True)
        (self.repo/'runs'/'README.md').write_text('runs (chain test)\n', encoding='utf-8')
        self.commit('P1: the arm table, the preregistration, the pairs (the code of the hold-out)')
        self.p1 = self.head()
        line = f'| 2026-11-10 | the pinned commit {self.p1} (chain test) |'
        (self.repo/CI.PINNED_RECORD_REL).parent.mkdir(parents=True, exist_ok=True)
        (self.repo/CI.PINNED_RECORD_REL).write_text(json.dumps(dict(schema=CI.PINNED_RECORD_SCHEMA, commit=self.p1, decision_log_line=line, written_by='lead', time_local='11-10 7 PM CST',
                                                                     utc='2026-11-11T01:00:00Z', reason='chain test')), encoding='utf-8')
        with (self.repo/CI.LATER_DECISION_LOG).open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
        self.commit('runs/holdout_B_v1/pinned_code.json and its decision log line, one commit (C24.3-9)')

    def authorize(self, tool, kind, pair, *, arm, **fields):
        """holdout_common's committed record (splits/bayesreg_v1/holdout_B_v1_<tool>_<kind>_<k>.json) on BZ's behalf."""
        key = (tool, kind)
        self.k[key] = self.k.get(key, 0)+1
        rel = f'splits/bayesreg_v1/holdout_B_v1_{tool}_{kind}_{self.k[key]}.json'
        doc = dict(schema=f'bayesreg.{tool}.{kind}.v1', written_by="lead on BZ's behalf", words='（合成测试）BZ：照办', time_local='11-10 7 PM CST', utc='2026-11-11T01:00:00Z',
                   reason='chain test', tool=tool, arm=arm, pair=pair, pair_set='holdout')
        doc.update(fields)
        (self.repo/rel).write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
        self.commit(f'authorization {rel}')
        return rel

    # the entry's pins in the sandbox --------------------------------------------------------------------------------------------------------------------------
    def patch_entry(self):
        mp, results = self.mp, Path(hc.RESULTS)
        self.data = self.tmp/'data'
        lookup = H.seg4_map()
        from bayesreg.io.pair import _read_manifest
        grid, _, _ = _read_manifest(Path(hc.STAGE_ROOT)/self.pairs[0]/'manifest.json')      # the truth on the stage's grid (HP's manifest: the images' affine)
        for k, s in enumerate(sorted({s for r in self.rows for s in r[1:3]})):
            labels = oasis_labels(30+k, lookup, XW.SHAPE)
            for kind in ('seg35', 'seg4'):
                save_scalar(self.data/s/H.LABEL_FILES[kind], labels[kind], grid)
        for key, value in dict(PAIRS_SHA256=file_sha256(self.repo/H.PAIRS_REL), N_PAIRS=len(self.rows), PREREG_SHA256=file_sha256(self.repo/H.PREREG_REL), ARM_TABLE_SHA256=self.table_sha,
                               DATA_ROOT=self.data, RESULTS=results, HOLDOUT_RESULTS=Path(hc.HOLDOUT_ROOT), EVAL_DIR=Path(hc.HOLDOUT_ROOT)/'eval',
                               MASTER_LOCK=Path(hc.HOLDOUT_ROOT)/'eval'/'holdout_eval.lock', PAIR_LOCK_DIR=Path(hc.HOLDOUT_ROOT)/'eval'/'pair_locks',
                               TRUTH_GATE_OUT=results/'ev'/'truth_gate_v1', TRUTH_GATE_LOCK=results/'ev'/'truth_gate.lock', TRUTH_GATE_PLAN_SHA256='f'*64,
                               REVIEW_ROOT=hb.review_root(), PREREPAIR_ROOT=P.record_root(), PREREPAIR_BIG=P.big_root()).items():
            mp.setattr(H, key, value)
        mp.setattr(H, 'replay_decision_problem', lambda decision=None, *, repo=None: None if isinstance(decision, dict) and decision.get('reading') == REPLAY_PASS
                   else 'PQ_REPLAY_DECISION (the chain test stands in for pqh_replay; its own tests run it)')
        mp.setattr(H, 'imported_from', lambda root: True)

    def gates(self):
        """The truth gate's decision as the gate leaves it on P1 (result, lock) and the replay gate's (its check stood in), both with code_commit P1 (rule two from P1)."""
        import nibabel
        out = Path(H.TRUTH_GATE_OUT)
        out.mkdir(parents=True, exist_ok=True)
        counts = dict(reader=dict(subjects=20, identical=20), metrics=dict(pairs=19, identical=19))
        (out/'truth_gate.json').write_text(json.dumps(dict(schema=H.SCHEMA_GATE, reading=H.GATE_READING_PASS, plan=dict(sha256='f'*64), counts=counts, code=dict(head=self.p1, clean=True))))
        Path(H.TRUTH_GATE_LOCK).write_text(json.dumps(dict(code_head=self.p1))+'\n')
        decision = dict(result_json=str(out/'truth_gate.json'), result_sha256=file_sha256(out/'truth_gate.json'), reading=H.GATE_READING_PASS, code_commit=self.p1,
                        environment=dict(numpy=np.__version__, nibabel=nibabel.__version__), counts=counts, plan_sha256='f'*64, decided_by='chain test', review='chain test')
        return decision, dict(result_json='x', result_sha256='e'*64, reading=REPLAY_PASS, code_commit=self.p1, plan_sha256='d'*64, decided_by='t', review='t')

    # the steps ------------------------------------------------------------------------------------------------------------------------------------------------------
    def method(self, outcomes, *, expect=0):
        assert hm.queue(pairs=list(self.pairs), system=ChainSystem(outcomes, labels=self.labels), echo=quiet) == expect
        self.hook('after_method')

    def review(self, outcomes, *, expect=0):
        code = review.queue(system=FakeSystem(outcomes), echo=quiet)
        assert code == expect, code
        self.hook('after_review')

    def move(self, *, fail_at=None):
        for p in self.pairs:
            if fail_at == p:
                import shutil
                real, calls = shutil.copyfile, []

                def copyfile(src, dst, *a, **k):
                    calls.append(src)
                    if len(calls) == 3:
                        raise OSError(28, 'No space left on device (chain test)')
                    return real(src, dst, *a, **k)
                self.mp.setattr(shutil, 'copyfile', copyfile)
                with pytest.raises(hc.InfrastructureFailure):
                    hb_store.move_pair(p, disk=lambda: None)
                self.mp.setattr(shutil, 'copyfile', real)
                move = hb.record_dir('method', p)/'move.json'
                rel = self.authorize('hb', 'resume', p, arm='move', lock_sha256=hb.sha(move), previous_record=str(move), previous_record_sha256=hb.sha(move))
                assert hb_store.move_resume(p, rel, disk=lambda: None)['passed']
            else:
                assert hb_store.move_pair(p, disk=lambda: None)['passed']
        self.hook('after_move')

    def configc(self, pair, *, probe=None, kern=None, resume_rel=None, expect=0):
        batch = C.roots('holdout')[0]/'batches'/'chain'
        batch.mkdir(parents=True, exist_ok=True)
        code = C.run('holdout', pair, batch=batch, resume_rel=resume_rel, launcher=Launcher('ok'), certify=W.Certify(), probe=probe or probe_of(), kernel=kern or kernel())
        assert code == expect, code
        return C.cell_paths('holdout', pair)

    def xb(self, arms=XB_MAIN, pairs=None):
        """The external arms (xb_drive.run_cell with the XB world's FakeMethod); ConvexAdam-v4seg with --method-run the method's T88 run.json HB's records pin (on D:), HI's predictions read
        through check_prediction and check_hi_record."""
        for arm in arms:
            for p in (pairs or self.pairs):
                XW.sealed_module._reset_cache()
                method_run = xc.method_t88_run('holdout', p) if xc.ARMS[arm]['kind'] == 'v4seg' else None
                if xc.ARMS[arm]['kind'] == 'v4seg' and method_run is None:
                    self.xb_p20(arm, p)
                    continue
                assert XW.run_cell(arm, 'holdout', p, method_run=method_run, deps=XW.deps()) == 0, (arm, p)
        self.hook('after_xb')

    def input_lock(self):
        """Step 8 (hb_store.input_lock): it reads the pair list's 100 subjects (hb.rows over the whole list) in HI's resumed root (written at the start: the union of the two runs, A5b item 13;
        stand-in files but the first pair's)."""
        everyone = [hc.pair_name(*r) for r in XW.holdout_rows()]
        self.mp.setattr(hb, 'rows', lambda: list(everyone))
        hb_store.input_lock(P.LOCK_REL, repo=self.repo)
        self.mp.setattr(hb, 'rows', lambda: list(self.pairs))
        self.hook('after_input_lock')
        self.commit('step 8: the input lock')

    def xb_p20(self, arm, pair):
        """An external arm's P20 (prereg revision 2 item 17 (f); A5a; A5b review: the external arm's P20 in the chain): ConvexAdam-v4seg on a pair whose method arm left no T88 run.json in
        HB's records (no output: HE's P20, a step-5 implementation error, a withdrawal; or the move record unreadable) is refused before its lock (stage method_run); BZ's committed rerun
        authorization; refused again; the cell's refusal records copied into its record directory (holdout_common.keep_refusals, as xb_chain does); BZ's committed no_resume naming the
        newest copy, its reason naming P20.  XB's own state reads it final (xb_chain.cell_state); the entry's p20_cell writes it (no output, bz_no_rerun)."""
        for k in (1, 2):
            XW.sealed_module._reset_cache()
            assert XW.run_cell(arm, 'holdout', pair, deps=XW.deps()) == xc.EXIT['refused'], (arm, pair, k)
            if k == 1:
                XW.authorize('rerun', arm, 'holdout', pair)
        paths = xc.cell_paths(arm, 'holdout', pair)
        hc.keep_refusals(xc.roots('holdout')[0], paths['record'], arm=arm, pair=pair, pair_set='holdout')
        _, draft = XW.xb_drive.authorization_draft('no_resume', arm, 'holdout', pair, written_by='BZ', words='（合成测试）BZ：P20，这一对不再跑', time_local='11-10 10 PM CST',
                                                   reason="P20: the method arm has no output on this pair, ConvexAdam-v4seg cannot read the method's labels; BZ does not let it run again")
        XW.commit_authorization('holdout', 'no_resume', draft)
        import xb_chain
        assert xb_chain.cell_state(arm, 'holdout', pair)[0] == 'final'
        self.p20_cells = getattr(self, 'p20_cells', [])+[(arm, pair)]

    def xb_v4seg(self, pair, *, expect=0):
        """ConvexAdam-v4seg on a hold-out pair (A5b; A5a review n6): xb_drive.run_cell with --method-run the method's T88 run.json pinned by HB's finalized and move records (on D:), HI's
        predictions of both subjects checked (check_prediction), then HI's subject records (check_hi_record) -- the hold-out wiring the XB tests check only by its source text."""
        XW.sealed_module._reset_cache()
        method = XW.FakeMethod()
        code = XW.run_cell('convexadam_v4seg', 'holdout', pair, method_run=xc.method_t88_run('holdout', pair), deps=XW.deps(step=method))
        assert code == expect, code
        return method

    def prerepair(self):
        XW.sealed_module._reset_cache()
        assert P.run(repo=self.repo, gate=lambda: dict(passed=True), echo=quiet) == 0
        self.hook('after_prerepair')

    def bz_ii(self, rel=H.PQ_METHOD_SIDE_REL):
        (self.repo/rel).parent.mkdir(parents=True, exist_ok=True)
        (self.repo/rel).write_text(json.dumps(dict(schema=H.SCHEMA_PQ_METHOD_SIDE, written_by="lead on BZ's behalf", words='（合成测试）F4 只描述', time_local='11-10 7 PM CST',
                                                   utc='2026-11-11T01:00:00Z', reason='chain test: a method-side control of PQ failed (synthetic)', choice='ii'), ensure_ascii=False))
        self.commit("BZ's (ii)")
        self.ii = rel

    def manifest(self):
        """Step 10 (the lead's D108 list, built with the entry's own input_roles): the inputs only, and the record-only file, one commit."""
        self.hook('before_manifest')
        cache = {}
        table = H.validate_arm_table(json.loads((self.repo/H.ARM_TABLE_REL).read_text()))
        cells = H.check_cells(table, self.rows, repo=self.repo, cache=cache)
        ablation = H.check_ablation(table, self.rows, cells, cache)
        inputs = sorted(H.needed_inputs(table, self.rows, cells, ablation=ablation))
        lines = [f'# {H.SCHEMA_MANIFEST}', f'# prereg_sha256 {file_sha256(self.repo/H.PREREG_REL)}', f'# arm_table_sha256 {self.table_sha}',
                 f'# unseal_sha256 {file_sha256(self.repo/H.UNSEAL_REL)}', f'# {H.PQ_HEADER_BZ_II} {self.ii}']+[f'{file_sha256(p)}  {p}' for p in inputs]
        (self.repo/H.MANIFEST_REL).write_text('\n'.join(lines)+'\n', encoding='utf-8')
        (self.repo/H.RECORDS_REL).write_text(f'# record-only lines (chain test)\n{file_sha256(self.repo/P.LOCK_REL)}  {self.repo/P.LOCK_REL}\n', encoding='utf-8')
        self.commit('step 10: the D108 manifest and the record-only file')
        return inputs

    def evaluate(self, out='run_1'):
        decision, replay = self.gates()
        XW.sealed_module._reset_cache()
        admission, context = H.open_holdout(Path(H.EVAL_DIR)/out, repo=self.repo, decision=decision, pq_replay=replay, run=no_units)
        code = H.evaluate_holdout(admission, context)
        return code, context

    def records(self, context):
        return {r[3]: json.loads((context['out']/'pairs'/f'{r[3]}.json').read_text()) for r in self.rows}


def hx_chain(chain, *, review_outcomes=None, method_outcomes=None, move_fail=None):
    chain.method(method_outcomes or [dict(line=he.HX), dict(line=he.FALLBACK_C)])
    chain.review(review_outcomes or [dict(review='finalize'), dict(review='finalize')])
    chain.move(fail_at=move_fail)


# ---- the main chain ------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_chain_end_to_end_with_configc_resumed_an_xb_arm_stopped_the_fallback_and_f4_ii(tmp_path, monkeypatch):
    chain = Chain(tmp_path, monkeypatch)
    p1, p2 = chain.pairs
    hx_chain(chain)
    # configuration C: p1 ok; p2 infra_failure (OOM evidence), then BZ's resume, then ok (prereg §5.4)
    chain.configc(p1)
    paths = chain.configc(p2, probe=probe_of(dict(oom=1, oom_kill=1)), expect=C.EXIT['infra_failure'])
    first = paths['result']
    rel = chain.authorize('hb', 'resume', p2, arm='configc', lock_sha256=hb.sha(paths['lock']), previous_record=str(first), previous_record_sha256=hb.sha(first))
    chain.configc(p2, resume_rel=rel)
    # XB: three arms on both pairs; fireants_syn stopped after p1 (P19: XB's arm_stopped.json in XB's format, the lead's committed copy with unrun_pairs)
    chain.xb(arms=('ants_s', 'fireants_greedy', 'convexadam_v4seg'))
    chain.xb(arms=('fireants_syn',), pairs=[p1])
    xb_stop = xc.cell_paths('fireants_syn', 'holdout', p1)['arm_stop']
    hc.write_json_once(xb_stop, dict(schema=xc.SCHEMA.format('arm_stop'), arm='fireants_syn', pair_set='holdout', utc=hc.utc(), suspicious_pairs=['a', 'b', p1], at_pair=p1,
                                     rule='prereg §4.5 P19 (chain test: three suspicious pairs stand in)'))
    xb_doc = json.loads(xb_stop.read_text())
    stop_rel = 'runs/holdout_B_v1/arm_stop_fireants_syn.json'
    (chain.repo/stop_rel).write_text(json.dumps(dict({k: xb_doc[k] for k in H.XB_STOP_FIELDS}, schema=H.SCHEMA_STOP, written_by='lead', unrun_pairs=[p2],
                                                     arm_stopped=dict(path=str(xb_stop), sha256=hb.sha(xb_stop)))))
    chain.commit('the lead: the committed stop record of fireants_syn')
    chain.input_lock()
    chain.prerepair()
    chain.bz_ii()
    chain.manifest()
    code, context = chain.evaluate()
    assert code == 0, {k: json.loads((context['out']/'holdout_eval.json').read_text())[k] for k in ('failed', 'stopped')}
    line = json.loads(Path(H.MASTER_LOCK).read_text())
    assert line['pinned_commit'] == chain.p1 and line['ablation']['pairs'] == sorted([p1, p2])          # rule one on the committed pinned record; the ablation decided before any truth
    records = chain.records(context)
    a1, a2 = records[p1]['arms'], records[p2]['arms']
    assert a1['method']['terminal_status'] == he.HX and a2['method']['terminal_status'] == he.FALLBACK_C and a1['method']['move']['files']['output/disp.npz']['finalized_sha256']
    assert a2['configc']['outcome'] == 'output' and a2['configc']['terminal'].endswith('result_resume.json') and a1['configc']['outcome'] == 'output'
    assert a2['fireants_syn']['no_output_class'] == 'arm_stopped' and a2['fireants_syn']['stop_record']['arm_stopped']['path'] == str(xb_stop)
    assert all(records[p]['ablation']['in_ablation'] and records[p]['ablation']['complete'] for p in (p1, p2)) and all(records[p]['pq']['method_side'] == 'bz_ii' for p in (p1, p2))
    collected = D.collect([context['out']], chain.rows)
    decided = D.decide(collected, chain.table, chain.rows, sdlogj_guard=.02, idx=D.draw_idx(n=2, b=200))
    assert decided['F1']['arms']['fireants_syn']['verdict'] == D.NOT_JUDGED and decided['F4']['reading'] == D.F4_METHOD
    assert decided['ablation']['tables']['main']['pairs'] == [p1] and decided['ablation']['tables']['fallback']['pairs'] == [p2]


def test_the_step5_jia_a_withdrawal_and_a_move_resume(tmp_path, monkeypatch):
    chain = Chain(tmp_path, monkeypatch)
    p1, p2 = chain.pairs
    chain.method([dict(line=he.HX), dict(line=he.HX)])
    # p1: the review stops (review_not_finalized, its /home copy); BZ withdraws it (C22.5-6 (b), (甲); hb_review.withdraw)
    assert review.queue(system=FakeSystem([dict(review='stop', exit=21)]), echo=quiet) == 21
    copy = review.home_records(p1)/'review_not_finalized.json'
    rel = chain.authorize('hb', 'no_resume', p1, arm='review', record=str(copy), record_sha256=hb.sha(copy), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (b): 撤回这一对的输出')
    review.withdraw(p1, rel)
    # p2: the (d) chain ends in hx_implementation_error at step 5; BZ's (甲) of arm review on the /home copy of review.json
    assert review.queue(system=FakeSystem([dict(review='implementation')]), echo=quiet) != 0
    home = review.home_records(p2)/'review.json'
    chain.authorize('hb', 'no_resume', p2, arm='review', record=str(home), record_sha256=hb.sha(home), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (a): 补成恒等')
    assert review.queue(system=FakeSystem([]), echo=quiet) == 0
    chain.move(fail_at=p1)                                                             # p1 half moved, then move-resume with BZ's resume (arm move)
    for p in chain.pairs:
        chain.configc(p)
    chain.xb()
    chain.input_lock()
    chain.prerepair()
    chain.bz_ii()
    chain.manifest()
    code, context = chain.evaluate()
    assert code == 0
    records = chain.records(context)
    m1, m2 = records[p1]['arms']['method'], records[p2]['arms']['method']
    assert m1['outcome'] == m2['outcome'] == 'no_output' and m1['no_output_class'] == m2['no_output_class'] == 'implementation'
    assert m1['authorization']['arm'] == m2['authorization']['arm'] == 'review' and m1['move']['finalized']['bz_decision']['step'] == 5
    moves = hb.record_dir('method', p1)
    assert json.loads((moves/H.MOVE_NAME).read_text())['passed'] is False and json.loads((moves/H.MOVE_RESUME_NAME).read_text())['passed'] is True
    assert m1['move']['files']['terminal.json']['record'] in (str(moves/H.MOVE_NAME), str(moves/H.MOVE_RESUME_NAME))     # read through move.json's ok rows plus move_resume.json's
    assert all(not records[p]['ablation']['in_ablation'] for p in chain.pairs)        # no method output: no repaired field to compare with
    for p in chain.pairs:                                                               # ConvexAdam-v4seg has no method labels on these pairs: its P20 written by BZ's no_resume
        v4 = records[p]['arms']['convexadam_v4seg']
        assert v4['outcome'] == 'no_output' and v4['no_output_class'] == 'bz_no_rerun' and v4['cause'] == 'P20' and v4['authorization']['newest_refusal_copy']['seq'] == 2


def test_the_methods_p20_and_a_configc_rejudgment(tmp_path, monkeypatch):
    chain = Chain(tmp_path, monkeypatch)
    p1, p2 = chain.pairs
    # p1: HE refuses the pair (hx_provenance_rejected, P20); BZ does not let it rerun: no_resume naming P20 (hb_method.py no-resume), state final
    chain.method([dict(line=he.REJECTED)], expect=hb.EXIT['implementation'])
    step = hm.step_record(p1)
    rel = chain.authorize('hb', 'no_resume', p1, arm='method', record=str(step), record_sha256=hb.sha(step), note=hc.NO_RESUME_NOTE, reason='P20：BZ 不批重跑')
    hm.no_resume(p1, rel)
    chain.method([dict(line=he.HX)])
    chain.review([dict(review='finalize')])
    chain.move()
    # configuration C on p2 ends needs_lead_rejudgment (its evidence unreadable); the lead's committed rejudgment: unit_limit
    chain.configc(p1)
    paths = chain.configc(p2, kern=kernel(False), expect=C.EXIT['needs_lead_rejudgment'])
    rec = C.effective_path(paths)
    folder = chain.repo/RJ.REJUDGMENT_DIR_REL/'configc'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/f'{p2}_1.json').write_text(json.dumps(dict(schema=RJ.REJUDGMENT_SCHEMA, arm='configc', pair=p2, pair_set='holdout', status='needs_lead_rejudgment',
                                                       record=dict(path=str(rec), sha256=hb.sha(rec)), evidence=[], judgment='unit_limit', reason='the unit\'s evidence (chain test)',
                                                       written_by='lead', words='', time_local='11-10 7 PM CST', utc='2026-11-11T01:00:00Z')))
    chain.commit('the lead: a rejudgment of configc')
    assert C.state('holdout', p2) == 'done'                                            # the configuration C queue's own reading of the same record (one reader)
    chain.xb()
    chain.input_lock()
    chain.prerepair()
    chain.bz_ii()
    chain.manifest()
    code, context = chain.evaluate()
    assert code == 0
    records = chain.records(context)
    m1 = records[p1]['arms']['method']
    assert records[p1]['arms']['convexadam_v4seg']['cause'] == 'P20' and records[p2]['arms']['convexadam_v4seg']['outcome'] == 'output'      # the external arm's P20 beside the method's
    assert m1['outcome'] == 'no_output' and m1['no_output_class'] == 'bz_no_rerun' and m1['cause'] == 'P20' and m1['move']['finalized']['bz_decision']['step'] == 4
    c2 = records[p2]['arms']['configc']
    assert c2['no_output_class'] == 'unit_limit' and c2['rejudgment']['judgment'] == 'unit_limit' and c2['cause'] == 'rejudgment'


# ---- every interface: one side changes one field name (or path); the chain fails and names it ---------------------------------------------------------------------------
def note_message(case, message):
    """The failure message of an interface case, appended to $CHAIN_MESSAGES when it is set (for the implementation notes; nothing is written otherwise)."""
    import os
    target = os.environ.get('CHAIN_MESSAGES')
    if target:
        with open(target, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(case=case, message=message), ensure_ascii=False)+'\n')


def rename(path, old, new, *, rows=None):
    """Rename key `old` to `new` in a JSON record (at the top, or in each row of `rows`)."""
    doc = json.loads(Path(path).read_text())
    for target in ([doc] if rows is None else doc[rows]):
        if old in target:
            target[new] = target.pop(old)
    Path(path).chmod(0o644)                                              # some records are written read-only
    Path(path).write_text(json.dumps(doc))
    return doc


def break_finalized_cell(chain):
    """HB's finalized record names the cell `cell_dir` instead of `cell`."""
    doc = json.loads(finalized_path().read_text())
    row = doc['pairs'][chain.pairs[0]]
    row['cell_dir'] = row.pop('cell')
    finalized_path().chmod(0o644)
    finalized_path().write_text(json.dumps(doc))


def finalized_path():
    return hb.review_root()/'finalized_outputs.json'


def break_review_json(chain):
    """HB's review.json says `final` instead of `finalized` (and its finalized record names the new bytes, as HB would)."""
    p1 = chain.pairs[0]
    path = hb.cell(p1)/'review'/'review.json'
    rename(path, 'finalized', 'final')
    doc = json.loads(finalized_path().read_text())
    doc['pairs'][p1]['review']['sha256'] = hb.sha(path)
    finalized_path().chmod(0o644)
    finalized_path().write_text(json.dumps(doc))


CASES = {
    # (point, change, what the failure must say -- the renamed key read as missing, at its place (A5b review minor 1: not a word the message holds anyway) --, where it fails: 'open' (the
    # entry's (b), check_cells, refuses before any truth; the lead's manifest step runs the same function), 'ablation' (the pair leaves the ablation), 'prerepair')
    'finalized.cell': ('after_review', break_finalized_cell, 'finalized_outputs.json: no row with a cell for {p1}', 'open'),
    'move.dst': ('after_move', lambda c: rename(hb.record_dir('method', c.pairs[0])/'move.json', 'dst', 'destination', rows='files'),
                 "move.json: files[0] ", 'open'),
    'review.finalized': ('after_review', break_review_json, 'review/review.json: finalized = None is not one of [True]', 'open'),
    'configc.output_store': ('after_xb', lambda c: rename(C.cell_paths('holdout', c.pairs[0])['result'], 'output_store', 'stored_output'),
                             'result.json: output_store.disp.file = None / output_store.disp.sha256 = None', 'open'),
    'xb.disp': ('after_xb', lambda c: rename(xc.cell_paths('ants_s', 'holdout', c.pairs[0])['result'], 'disp', 'displacement'), 'ants_s/{p1}/result.json: disp.file = None / disp.sha256 = None', 'open'),
    'xb.export_check': ('after_xb', lambda c: rename(xc.cell_paths('fireants_greedy', 'holdout', c.pairs[0])['result'], 'export_check', 'export'),
                        'fireants_greedy/{p1}/result.json: export_check.suspicious is not a bool', 'open'),
    'prerepair.checks': ('after_prerepair', lambda c: rename(P.paths(c.pairs[0])[0], 'checks', 'tests'),
                         "prerepair.json: the preparation did not pass: checks ['field_sha256', 'u_sha256', 'tapered_u_sha256'] not passed; problems []", 'ablation'),
    'input_lock.sha256': ('after_input_lock', lambda c: rename(c.repo/P.LOCK_REL, 'sha256', 'digest', rows='rows'), "t88/field_0600.npz: the input lock's row None is not its sha256", 'prerepair'),
}


@pytest.mark.parametrize('case', sorted(CASES))
def test_an_interface_with_one_field_renamed_fails_and_names_it(tmp_path, monkeypatch, case):
    point, change, said, where = CASES[case]
    chain = Chain(tmp_path, monkeypatch, hooks={point: change})
    said = said.format(p1=chain.pairs[0])
    hx_chain(chain)
    for p in chain.pairs:
        chain.configc(p)
    chain.xb()
    chain.input_lock()
    chain.prerepair()
    chain.bz_ii()
    if where == 'prerepair':
        record = json.loads(P.paths(chain.pairs[0])[0].read_text())
        note_message(case, record['problems'])
        assert record['passed'] is False and said in record['problems'], record['problems']
        return
    if where == 'open':
        with pytest.raises(H.HoldoutRefused) as info:
            chain.manifest()                                                   # the lead's list is built by the entry's own reading of the cells: it refuses there already
            chain.evaluate()
        note_message(case, str(info.value))
        assert said in str(info.value), str(info.value)[:2000]
        assert not Path(H.MASTER_LOCK).exists()
        if case == 'move.dst':                                                # A5b diff review minor 3: both readers name the move record's format, not a move not done
            assert "has no ['dst']: the move record's format" in str(info.value)
            refused = [json.loads(q.read_text()) for q in XW.refusals('holdout')]
            assert any(r.get('stage') == 'method_run' and "(t88/run.json) has no dst: the move record's format" in json.dumps(r) for r in refused), refused
        return
    chain.manifest()
    code, context = chain.evaluate()
    reason = chain.records(context)[chain.pairs[0]]['ablation']
    note_message(case, reason['reason'])
    assert code == 0 and reason['in_ablation'] is False and said in reason['reason'], reason


@pytest.mark.parametrize('case', ['ii_path', 'stop_pointer', 'pinned_key', 'rejudgment_key'])
def test_the_committed_records_with_one_field_or_path_changed_fail_and_name_it(tmp_path, monkeypatch, case):
    chain = Chain(tmp_path, monkeypatch)
    p1, p2 = chain.pairs
    hx_chain(chain)
    chain.configc(p1)
    if case == 'rejudgment_key':
        paths = chain.configc(p2, kern=kernel(False), expect=C.EXIT['needs_lead_rejudgment'])
        rec = C.effective_path(paths)
        folder = chain.repo/RJ.REJUDGMENT_DIR_REL/'configc'
        folder.mkdir(parents=True, exist_ok=True)
        (folder/f'{p2}_1.json').write_text(json.dumps(dict(schema=RJ.REJUDGMENT_SCHEMA, arm='configc', pair=p2, pair_set='holdout', status='needs_lead_rejudgment',
                                                           record=dict(path=str(rec), sha256=hb.sha(rec)), evidence=[], verdict='unit_limit', reason='r', written_by='lead', words='',
                                                           time_local='x', utc='2026-11-11T01:00:00Z')))
        chain.commit('a rejudgment whose judgment key is called verdict')
        said = f"{p2}_1.json: exactly the keys ["                                     # the record's keys listed (quotes escaped in the nested message)
    else:
        chain.configc(p2)
    if case == 'stop_pointer':
        chain.xb(arms=('ants_s', 'fireants_greedy', 'convexadam_v4seg'))
        chain.xb(arms=('fireants_syn',), pairs=[p1])
        xb_stop = xc.cell_paths('fireants_syn', 'holdout', p1)['arm_stop']
        hc.write_json_once(xb_stop, dict(schema=xc.SCHEMA.format('arm_stop'), arm='fireants_syn', pair_set='holdout', utc=hc.utc(), suspicious_pairs=['a', 'b', p1], at_pair=p1, rule='P19'))
        xb_doc = json.loads(xb_stop.read_text())
        (chain.repo/'runs/holdout_B_v1/arm_stop_fireants_syn.json').write_text(json.dumps(dict({k: xb_doc[k] for k in H.XB_STOP_FIELDS}, schema=H.SCHEMA_STOP, written_by='lead',
                                                                                              unrun_pairs=[p2], xb_record=dict(path=str(xb_stop), sha256=hb.sha(xb_stop)))))
        chain.commit('a stop record whose pointer is called xb_record')
        said = "runs/holdout_B_v1/arm_stop_fireants_syn.json: arm_stopped {path, sha256} of XB's arm_stopped.json required"
    else:
        chain.xb()
    chain.input_lock()
    chain.prerepair()
    if case == 'ii_path':
        chain.bz_ii('runs/holdout_B_v1/pq_method_side.json')
        said = f"pq_method_side_decision 'runs/holdout_B_v1/pq_method_side.json': BZ's (ii) record is {H.PQ_METHOD_SIDE_REL} and no other path"
    else:
        chain.bz_ii()
    if case == 'pinned_key':
        doc = json.loads((chain.repo/CI.PINNED_RECORD_REL).read_text())
        doc['pinned'] = doc.pop('commit')
        (chain.repo/CI.PINNED_RECORD_REL).write_text(json.dumps(doc))
        chain.commit('the pinned record changed (after P1: refused by rule one too)')
        said = f"{CI.PINNED_RECORD_REL}: exactly the keys ['schema', 'commit',"
    with pytest.raises(H.HoldoutRefused) as info:
        chain.manifest()
        chain.evaluate()
    note_message(case, str(info.value))
    assert said in str(info.value), str(info.value)[:2000]
    assert not Path(H.MASTER_LOCK).exists()


# ---- HI resumed into its one root, read by every reader; ConvexAdam-v4seg on a hold-out cell (A5b item 13; the lead's message of 11-10 evening, items 2 and 3) ------------------------------
def test_hi_resumed_into_its_one_root_is_read_by_he_s_queue_pq_convexadam_v4seg_and_the_input_lock(tmp_path, monkeypatch):
    """The chain's HI root was written by a first run killed half way and its one resume (the first pair's fixed subject by the first run, its moving subject by the resume after its half
    prediction was moved to attempt1/pred/).  HE's queue reads it as one (batch.json's hi); ConvexAdam-v4seg on that pair goes through check_prediction and check_hi_record (both records, run 0 and
    run 1) to an output whose maps are the method's labels; PQH's hi_predictions reads the union under its spy with hi_files allowed (the resume's records among them, nothing breached); the
    input lock rows each record; nothing under attempt1/ is opened by any of them."""
    chain = Chain(tmp_path, monkeypatch)
    p1, _ = chain.pairs
    f1, m1 = p1.split('__')
    hx_chain(chain)
    batch = json.loads(next(Path(hc.batch_root()).glob('hb_method_*/batch.json')).read_text())
    assert batch['hi']['runs'] == [str(chain.hi_root)] and batch['hi']['missing'] == [] and batch['hi']['residue'] == []
    assert batch['hi']['records'][f1]['run'] == 0 and batch['hi']['records'][m1]['run'] == 1 and (chain.hi_root/'attempt1'/'pred'/m1).is_dir()
    method = chain.xb_v4seg(p1)
    doc = json.loads(xc.cell_paths('convexadam_v4seg', 'holdout', p1)['result'].read_text())
    records = doc['checks']['hi_records']
    assert doc['status'] == 'ok' and len(method.calls) == 1 and doc['maps']['method_labels_check']['equal'] is True
    assert records['fixed']['run'] == 0 and records['moving']['run'] == 1 and records['moving']['record'] == str(chain.hi_root/'subjects'/f'{m1}.json')
    assert doc['checks']['method_run']['path'] == str(Path(hc.big_dir('method', p1))/'t88'/'run.json')
    import pqh_inputs as hin
    from pqh_firewall import HoldoutRules, spy
    everyone = sorted({s for r in XW.holdout_rows() for s in r})
    lock = Path(hc._hi().HOLDOUT_LOCK)
    rules = HoldoutRules(unit='out', mode='holdout', watched=(Path(hc.RESULTS),), files=hin.hi_files(lock, everyone), patterns=[(lock.parent, hin.HI_LOCK_PATTERN)])
    with spy(rules) as watch:
        rows = hin.hi_predictions(lock, everyone)
    opened = {Path(r['path']) for r in watch.opened}
    assert watch.breaches == [] and chain.hi_root/chain.hi_names['record'] in opened and not any('attempt1' in q.parts for q in opened)
    assert rows[m1]['pred_dir'] == chain.hi_root/'pred'/m1 and rows[m1]['record_sha256'] == records['moving']['record_sha256'] and rows[f1]['run'] == str(chain.hi_root)
    assert rows[m1]['output_sha256']['probs36.npz'] == hb.sha(chain.hi_root/'pred'/m1/'probs36.npz') == doc['checks']['predictions']['moving']['probs36_sha256']
    chain.input_lock()
    lock_doc = json.loads((chain.repo/P.LOCK_REL).read_text())
    hi_rows = {r['subject']: r for r in lock_doc['rows'] if r['kind'] == 'hi_record'}
    assert len(hi_rows) == 100 and hi_rows[f1]['run'] == 0 and hi_rows[m1]['run'] == 1 and hi_rows[m1]['sha256'] == records['moving']['record_sha256']
    assert not any('attempt1' in str(r.get('path')) for r in lock_doc['rows'])


@pytest.mark.parametrize('how', ['record_lost', 'record_other_sha'])
def test_convexadam_v4seg_refuses_a_hold_out_subject_whose_hi_record_does_not_hold(tmp_path, monkeypatch, how):
    """xb_core.check_hi_record on the hold-out cell (A5a review I2 point 3): the moving subject's HI record lost, or naming another probs36 sha256 -> refused (stage provenance) before the lock,
    the method not run, the refusal naming the record."""
    chain = Chain(tmp_path, monkeypatch)
    p1, _ = chain.pairs
    m1 = p1.split('__')[1]
    hx_chain(chain)
    record = chain.hi_root/'subjects'/f'{m1}.json'
    if how == 'record_lost':
        record.unlink()
    else:
        doc = json.loads(record.read_text())
        doc['output_sha256']['probs36.npz'] = '0'*64
        record.write_text(json.dumps(doc))
    method = chain.xb_v4seg(p1, expect=xc.EXIT['refused'])
    assert method.calls == [] and not xc.cell_paths('convexadam_v4seg', 'holdout', p1)['lock'].exists()
    refusals = [json.loads(q.read_text()) for q in XW.refusals('holdout')]
    assert any(r.get('stage') == 'provenance' and str(record) in json.dumps(r) for r in refusals), refusals


def test_the_chain_s_arm_table_is_the_frozen_one(tmp_path, monkeypatch):
    """A5b review minor 2: the chain's table is configs/frozen/holdout_B_v1_arm_table.json with its two roots moved, the method arm and the main family key for key; every arm it leaves out
    (descriptive, 90 s tier) is an XB arm whose spec is an included XB arm's with its name and family put back -- so the chain's cells exercise every arm's reading."""
    chain = Chain(tmp_path, monkeypatch)
    frozen = json.loads(FROZEN_TABLE.read_text(encoding='utf-8').replace(REAL_HOME, str(hc.HOLDOUT_ROOT)).replace(REAL_BIG, str(hc.BIG_ROOT)))
    mine = {a['name']: a for a in chain.table['arms']}
    assert all(mine[a['name']] == a for a in frozen['arms'] if a['name'] in mine) and chain.table['main_family'] == frozen['main_family']
    template = mine['fireants_greedy']
    for arm in frozen['arms']:
        if arm['name'] not in mine:
            assert json.dumps(dict(arm, name='fireants_greedy', family='main'), sort_keys=True).replace(arm['name'], 'fireants_greedy') == json.dumps(template, sort_keys=True), arm['name']
    assert H.validate_arm_table(json.loads((chain.repo/H.ARM_TABLE_REL).read_text())) is not None
