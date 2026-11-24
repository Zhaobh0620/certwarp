"""A5b review I3 (the lead's ruling, 11-10 ~9:22 PM CST): PQ's two units in the hold-out mode write what the evaluation entry reads -- the interface the chain test
(tests/test_bayesreg/hb/test_chain_integration.py) took through BZ's (ii) only.  On PQH's hold-out world (W.holdout_world: HI's run, HB's finalized record of the method cell, SG's hold-out
admission stood in by the development one as in test_pqh_resume.py): the members unit runs with its inference stood in (the registered prediction and the development V), the PQ unit runs
for real and writes pq.json, pq.npz, baselines.npz and pq_outputs_manifest.txt; the lead's release (pq_outputs_manifest.sha256 and pq_release.md, one commit) and the step-8 input lock (HB's
format: the subjects' prediction files that HI's records name, the method's field_0600.npz) are committed in the code tree; then the entry's own functions read them in its order: (c)'s
check_pq_release and pq_states, Ev1 (pq_ev1), read_input_lock, and the pair's evaluate_pq (Ev2 against the d the entry reads from the method's output/disp.npz, the input-lock check, the
acceptance numbers on the world's truth).  The first two tests do not go through open_holdout: the entry's PQ functions are called with the inputs the D108 manifest would list (PQ's list
and every file it names).

U5 (解封包定稿_20261115.md U5; decision_log 2026-11-14 SM3 ruling (1), 2026-11-15 ~4:20 AM (10); A5b diff review: "PQ 一段没走 open_holdout 全程"): the two synthetic worlds as one.  The
PQH world above runs its units in the production order -- HB's step 5 finalizes the method cell in its working cell, hb_store.move_pair moves it whole to D: (move.json), PQ's units find the
method's files through that move record -- and the evaluation entry's synthetic world (tests/test_bayesreg/eval/_holdout_world.World: SG's real admission on synthetic sealed lists naming the
PQ world's two ids, BZ's synthetic unseal record, rule one stubbed as there, five main arms in XB's format) is built around what PQ wrote: the arm table's method arm and pq section are the frozen
table's (configs/frozen/holdout_B_v1_arm_table.json, its two roots moved into the sandbox), the stage is the one PQ read, the truth the PQ world's labels written as NIfTI on the stage's grid; the
lead's release, the step-8 input lock and the D108 manifest (the entry's own input_roles over PQ's real list) are committed in the entry's code tree.  Then open_holdout and evaluate_holdout run
in full: (c) reads PQ's list, the spy lets exactly the listed PQ files through, the pair's record holds PQ's Ev1, Ev2, input lock and numbers.  Every assertion of the two tests above is made
again on the pair's record; the entry's own tests of PQ on the synthetic PQ files (tests/test_bayesreg/eval/test_holdout_entry.py) are unchanged.  The PQ units run on the PQH world's code tree
(its registry, allowlist and the development admission standing in for SG's), the entry on its own; the two trees share every data root."""
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from jd_curl.three_d.nifti import save_scalar

import bayesreg.io.sealed as S
from bayesreg.eval import holdout as H
from bayesreg.eval import holdout_decide as D

from ..eval import _holdout_world as EW
from ..eval import _synthetic as syn
from . import _world as W
from .test_pqh_resume import members_world, run_members, world  # noqa: F401 - the fixtures of the hold-out world with the members' inference stood in

hp, hin, hu, io_, hc = W.hp, W.hin, W.hu, W.io_, W.hc
PAIR = W.PAIR


def release(layout, repo, *, note=None):
    """The lead's release (C22.H4-5): runs/holdout_B_v1/pq_outputs_manifest.sha256 ('<sha256>  <the list>') and pq_release.md ('放行：<sha256>'), one commit."""
    manifest = Path(layout.pq_small)/hu.MANIFEST
    sha = io_.sha_file(manifest)
    for rel, text in ((H.PQ_RELEASE_SHA_REL, f'{sha}  {manifest}\n'), (H.PQ_RELEASE_NOTE_REL, (note or '放行：{sha}').format(sha=sha)+'\n')):
        (repo/rel).parent.mkdir(parents=True, exist_ok=True)
        (repo/rel).write_text(text, encoding='utf-8')
    W.PW.commit_all(repo, 'the lead: PQ released (test)')
    return manifest, sha


def input_lock(layout, repo, cell):
    """Step 8's input lock in hb_store.input_lock's format: kind prediction rows of each subject's probs36, structure_groups, provenance (the files HI's records name, by sha256) and the
    method's t88/field_0600.npz (kind method_field), committed."""
    rows = []
    for s, r in sorted(hin.hi_predictions(layout.hi_lock, layout.subjects).items()):
        for name in H.PQ_INPUT_KEYS:
            path = Path(r['pred_dir'])/name
            rows.append(dict(kind='prediction', subject=s, path=str(path), sha256=io_.sha_file(path)))
    field = Path(cell)/'t88'/'field_0600.npz'
    rows.append(dict(kind='method_field', pair=PAIR, path=str(field), sha256=io_.sha_file(field), line_status='hx_certified'))
    path = repo/H.INPUT_LOCK_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(schema=H.INPUT_LOCK_SCHEMA, finalized_outputs=None, rows=rows, utc='2026-11-10T00:00:00Z', rule='test: hb_store.input_lock\'s rows')), encoding='utf-8')
    W.PW.commit_all(repo, 'step 8: the input lock (test)')


def listed_inputs(manifest, sha):
    """What the D108 manifest lists of PQ: the list at its sha256 and every file it names at the sha256 it names (check_manifest compares the two)."""
    inputs = {str(manifest): sha}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest, path = line.split('  ', 1)
        inputs[path] = digest
    return inputs


def entry_reads(m, repo, *, note=None):
    """PQ's units, the release, the input lock; then the entry's (c) for PQ and the pair's PQ evaluation.  Returns (release record, states, the pair's PQ record)."""
    layout = m.layout
    assert run_members(m) == 0                                                  # the members unit (its inference stood in)
    assert W.run_out(layout, deps=W.deps()) == 0                                # the PQ unit, for real, in the hold-out mode
    final = json.loads(Path(layout.finalized_path).read_text())
    cell = Path(final['pairs'][PAIR]['cell'])
    manifest, sha = release(layout, repo, note=note)
    input_lock(layout, repo, cell)
    inputs = listed_inputs(manifest, sha)
    table = dict(arms=[dict(name='method', family='method')], pq=dict(small_root=str(layout.pq_small), big_root=str(layout.pq_big)))
    rel = H.check_pq_release(table, {}, inputs, repo=repo)                     # (c): the release from the code tree, the list parsed
    row = (0, *PAIR.split('__'), PAIR)
    cell_record = H.Cell('method', PAIR, 0, True, 'output', None, 'hx_certified', False, [], {}, field_dir=str(cell/'output'))
    states = H.pq_ev1(H.pq_states(table, [row], {('method', PAIR): cell_record}, rel), rel, inputs)
    rel['input_lock'] = H.read_input_lock(repo)
    grid, metadata, _ = H._read_manifest(Path(layout.stage)/PAIR/'manifest.json')
    labs, _, _ = m.w.truth(PAIR)                                               # the world's truth of the pair (the entry would read it after its lock)
    truth = H.truth_of_pair(dict(seg35=labs['fixed_seg35'], seg4=labs['fixed_seg4']), dict(seg35=labs['moving_seg35'], seg4=labs['moving_seg4']))
    info = H.describe_run(cell/'output', pair=PAIR, kind='disp')                # the entry's own reading of the method's field (evaluate_pair)
    d, _, _ = H.load_field(info, grid.shape, grid.spacing)
    record = {}
    context = dict(pq_release=rel, pq_states=states, listed=inputs)
    H.evaluate_pq(context, row, metadata, truth, (d, H.array_sha256(d)), record)
    return rel, states, record['pq'], inputs


@pytest.fixture
def chain(members_world, world):                                                # noqa: F811
    members_world.w = world.w
    return members_world


def test_pq_s_two_units_write_what_the_entry_reads_states_ev1_ev2_the_input_lock_and_the_numbers(chain):
    repo = Path(chain.layout.code).resolve()
    rel, states, pq, inputs = entry_reads(chain, repo)
    assert rel['method_side'] == 'released' and rel['passed'] is True and rel['problems'] == []
    state = states[PAIR]
    assert state['state'] == 'present' and set(state['files']) == set(H.PQ_FILES) and state['ev1']['passed'] is True
    assert all(Path(state['files'][n]).is_relative_to(Path(chain.layout.pq_small if root == 'small_root' else chain.layout.pq_big)) for n, root in H.PQ_FILES.items())
    assert pq['ev2']['passed'] is True and pq['ev2']['d_sha256_in_pq_json'] == pq['ev2']['d_sha256']                # Ev2: pq.json's d is the d the entry reads
    assert pq['input_lock']['passed'] is True and len(pq['input_lock']['rows']) == 7 and all(r['same'] for r in pq['input_lock']['rows'])
    q = pq['acceptance']['scores']['omega_e']['q']
    assert pq['complete'] is True and pq['acceptance']['masks']['omega_e'] > 0 and 0. <= q['auroc'] <= 1. and np.isfinite(q['ause'])
    record = dict(pair=PAIR, complete=True, pq=pq, arms=dict(method=dict(outcome='output', terminal_status='hx_certified')))
    [row] = D.pq_rows({PAIR: record}, dict(arms=[dict(name='method', family='method')]), [(0, *PAIR.split('__'), PAIR)])
    assert row['method_output'] is True and row['pq']['state'] == 'present' and row['pq']['ev2']['passed'] is True        # holdout_decide reads the same record (F4)


def test_a_release_note_that_does_not_hold_is_ev1_not_a_refusal(chain):
    """Ev1 (C22.H7-1): a release note that is not the one line 放行：<the list's sha256> is recorded (passed false), never refused; F4 reads the pair's PQ as 不判（评测侧）."""
    repo = Path(chain.layout.code).resolve()
    rel, states, pq, _ = entry_reads(chain, repo, note='放行：{sha}，但再看看')
    assert rel['passed'] is False and rel['problems'] and states[PAIR]['ev1']['passed'] is False and pq['ev1']['passed'] is False


# ---- U5: the two synthetic worlds as one (module docstring) -------------------------------------------------------------------------------------------------------------------------------
FROZEN_TABLE = W.ROOT/'configs'/'frozen'/'holdout_B_v1_arm_table.json'
REAL_HOME, REAL_BIG = '/home/rmcfz/seq_vp_tag_results/holdout_B_v1', '/mnt/d/seq_vp_tag_holdout_B_v1'      # the frozen table's two roots (moved into the sandbox, as the chain test does)
ENTRY_HOME_ROOTS = ('EVAL_DIR', 'MASTER_LOCK', 'PAIR_LOCK_DIR', 'REVIEW_ROOT', 'PREREPAIR_ROOT')
REAL_ENTRY = {k: Path(getattr(H, k)) for k in ('HOLDOUT_RESULTS', 'PREREPAIR_BIG')+ENTRY_HOME_ROOTS}      # the entry's own roots as pinned (read at import, before any test moves them)


def hb_store():
    """experiments/hb/hb_store.py (HB's move of the method cell to D:), imported as HB's tests do (its flat name; hb_core with it)."""
    folder = str(W.ROOT/'experiments'/'hb')
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import hb_store as module
    return module


def entry_roots():
    """The entry's roots in the sandbox by the frozen table's rule: each pinned path under HOLDOUT_RESULTS (= the table's /home root) moved to holdout_common's HOLDOUT_ROOT, under the table's D:
    root moved to its BIG_ROOT -- so the master lock is where PQ's unit looks for it (pqh_inputs.Layout.ev_master_lock)."""
    home = {k: Path(hc.HOLDOUT_ROOT)/REAL_ENTRY[k].relative_to(REAL_ENTRY['HOLDOUT_RESULTS']) for k in ENTRY_HOME_ROOTS}
    return dict(home, HOLDOUT_RESULTS=Path(hc.HOLDOUT_ROOT), PREREPAIR_BIG=Path(hc.BIG_ROOT)/REAL_ENTRY['PREREPAIR_BIG'].relative_to(REAL_BIG))


def step5_and_move(chain):
    """HB's step 5 and the move to D:, before PQ (prereg §5.2; C22.H1-1, -3; pqh_pins.METHOD_ARM): the method cell of the PQH world (HE's output with HB's review records, as HE and HB write
    them) is the working cell holdout_B_v1/work/method/<pair>/ (a byte copy), finalized_outputs.json names it (hb_review.finalize's row), then hb_store.move_pair itself moves it whole to D:
    (holdout_common.move_files) with its record holdout_B_v1/method/<pair>/move.json.  Returns the cell on D:."""
    store, layout = hb_store(), chain.layout
    work = Path(store.hb.cell(PAIR))
    final = json.loads(Path(layout.finalized_path).read_text(encoding='utf-8'))
    shutil.copytree(final['pairs'][PAIR]['cell'], work)
    W.finalized(layout, {PAIR: W.finalized_row(work)})
    assert store.hb.review_root()/'finalized_outputs.json' == Path(layout.finalized_path) and store.hb.record_dir('method', PAIR) == Path(layout.method_records)/PAIR
    moved = store.move_pair(PAIR, disk=lambda: None)
    dst = Path(hc.big_dir('method', PAIR))
    assert moved['passed'] is True and not work.exists() and store.where(PAIR) == dst and (dst/'output'/'disp.npz').is_file()
    return dst


class EntryWorld(EW.World):
    """The evaluation entry's synthetic world around the PQH world after its units ran (module docstring): one pair, the PQ world's; the entry's tree under <tmp>/entry/."""

    def __init__(self, tmp_path, chain, *, note=None):
        self.chain, self.note, self.layout = chain, note, chain.layout
        self.subjects = tuple(PAIR.split('__'))
        self.pair_rows = (self.subjects,)
        self.tmp = Path(tmp_path)/'entry'
        self.repo, self.data, self.results = self.tmp/'repo', self.tmp/'data', self.tmp/'results'
        self.lookup = H.seg4_map()
        self.grid = chain.w.grid
        self.rows = ((0, *self.subjects, PAIR),)
        labs, _, _ = chain.w.truth(PAIR)                                           # the PQ world's truth of the pair (its stage's label members)
        self.labels = {s: dict(seg35=labs[f'{side}_seg35'], seg4=labs[f'{side}_seg4']) for side, s in zip(('fixed', 'moving'), self.subjects)}
        self.reads, self.bz_ii, self.synced = [], None, False                    # synced stays False: the method cell is HB's, never rewritten by the world
        self.review_root = entry_roots()['REVIEW_ROOT']
        self._build()

    def roots(self):
        return {k: v for k, v in entry_roots().items() if k in ('REVIEW_ROOT', 'PREREPAIR_ROOT', 'PREREPAIR_BIG')}

    def write_table(self, table=None):
        """The frozen table's method arm and pq section (roots moved), the eval world's five main arms (XB's format), the stage PQ read."""
        if table is None:
            frozen = json.loads(FROZEN_TABLE.read_text(encoding='utf-8').replace(REAL_HOME, str(hc.HOLDOUT_ROOT)).replace(REAL_BIG, str(hc.BIG_ROOT)))
            prod = hin.holdout_layout(self.layout.code, pairs=(PAIR,))                     # PQ's production layout on the sandbox's roots: its roots are the frozen table's, and the world's
            assert frozen['pq'] == dict(small_root=str(prod.pq_small), big_root=str(prod.pq_big)) == dict(small_root=str(self.layout.pq_small), big_root=str(self.layout.pq_big))
            base = EW.arm_table(self.tmp)
            table = dict(base, stage_root=str(self.layout.stage), pq=frozen['pq'], arms=[a for a in frozen['arms'] if a['family'] == 'method']+[a for a in base['arms'] if a['family'] == 'main'])
        super().write_table(table)

    def field(self, arm_name, row):
        k = EW.MAIN.index(arm_name)+1
        return syn.smooth_d(shape=tuple(self.grid.shape), amplitude=(.5-.05*k, -.4+.05*row[0], .3), freq=1)

    def pq_list(self):
        return Path(self.layout.pq_small)/hu.MANIFEST

    def _build(self):
        self.write_code_tree()
        for subject, arrays in self.labels.items():
            for kind in ('seg35', 'seg4'):
                save_scalar(self.data/subject/H.LABEL_FILES[kind], arrays[kind], self.grid)
        for row in self.rows:
            for arm in self.table['arms']:
                if arm['family'] == 'main':
                    self.write_cell(arm, row, status='ok')
        (self.tmp/'moves').mkdir()
        (self.tmp/'moves'/'move_0.json').write_text('{}', encoding='utf-8')
        EW.commit(self.repo, 'the code tree of the hold-out (U5)')
        input_lock(self.layout, self.repo, hb_store().where(PAIR))               # step 8, the method's field where hb_store.where names it now (D:)
        release(self.layout, self.repo, note=self.note)                          # the lead's release of PQ's list (one commit)
        self.write_manifest()                                                     # the D108 manifest: the entry's own input_roles over PQ's real list

    def patch(self, monkeypatch, **kwargs):
        super().patch(monkeypatch, **kwargs)
        for key, value in entry_roots().items():
            monkeypatch.setattr(H, key, value)


@pytest.fixture
def merged(chain, tmp_path, monkeypatch):
    """build(note=None): HB's step 5 and move, PQ's two units (for real, in the hold-out mode), then the entry's world around them, patched."""
    monkeypatch.setattr(H, '_LAST_SPY', None)

    def build(note=None):
        step5_and_move(chain)
        assert run_members(chain) == 0
        assert W.run_out(chain.layout, deps=W.deps()) == 0
        entry = EntryWorld(tmp_path, chain, note=note)
        entry.patch(monkeypatch)
        return entry
    yield build
    H.stop_spy()
    S._reset_cache()


def evaluated(entry):
    admission, context = entry.open()
    code = H.evaluate_holdout(admission, context)
    return code, context, json.loads((context['out']/'pairs'/f'{PAIR}.json').read_text(encoding='utf-8')), json.loads((context['out']/'holdout_eval.json').read_text(encoding='utf-8'))


def test_u5_pq_s_units_write_what_open_holdout_reads_in_full(merged):
    """PQ's units write, the entry's open_holdout and evaluate_holdout read, in one sandbox: every assertion of the direct test above on the pair's record; then what only the whole entry does --
    the D108 list from the entry's own input_roles with PQ's list and files at PQ's sha256, the spy letting exactly them through, PQ and the entry reading the method's files at the same places
    through the same move record, the truth read for this pair only, and the master lock the one PQ's unit stops at (C22.H3-3)."""
    entry = merged()
    layout = entry.layout
    code, context, record, summary = evaluated(entry)
    assert code == 0 and summary['failed'] == [] and summary['stopped'] is None and record['complete'] is True
    # -- the direct test's assertions, on the record evaluate_holdout wrote
    rel, states, pq = context['pq_release'], context['pq_states'], record['pq']
    assert rel['method_side'] == 'released' and rel['passed'] is True and rel['problems'] == []
    state = states[PAIR]
    assert state['state'] == 'present' and set(state['files']) == set(H.PQ_FILES) and state['ev1']['passed'] is True
    assert all(Path(state['files'][n]).is_relative_to(Path(layout.pq_small if root == 'small_root' else layout.pq_big)) for n, root in H.PQ_FILES.items())
    assert pq['ev2']['passed'] is True and pq['ev2']['d_sha256_in_pq_json'] == pq['ev2']['d_sha256']                # Ev2: pq.json's d is the d the entry reads
    assert pq['input_lock']['passed'] is True and len(pq['input_lock']['rows']) == 7 and all(r['same'] for r in pq['input_lock']['rows'])
    q = pq['acceptance']['scores']['omega_e']['q']
    assert pq['complete'] is True and pq['acceptance']['masks']['omega_e'] > 0 and 0. <= q['auroc'] <= 1. and np.isfinite(q['ause'])
    [row] = D.pq_rows({PAIR: record}, entry.table, entry.rows)
    assert row['method_output'] is True and row['pq']['state'] == 'present' and row['pq']['ev2']['passed'] is True        # holdout_decide reads the same record (F4)
    # -- (c): the D108 list is the entry's input_roles over PQ's real list; PQ's list and its three files at the sha256 PQ's list and the release name
    listed, pq_list = context['listed'], str(entry.pq_list())
    assert H.classify_listed(pq_list, H.input_roles(entry.table, entry.rows, context['cells'], states, pq_list, context['ablation'])) == 'pq_list'
    assert listed[pq_list] == rel['released_sha256'] == io_.sha_file(pq_list) and rel['list'] == pq_list
    assert {p: listed[p] for p in state['files'].values()} == {p: rel['entries'][p] for p in state['files'].values()} == {p: io_.sha_file(p) for p in state['files'].values()}
    assert record['arms']['method']['outcome'] == 'output' and record['arms']['method']['terminal_status'] == 'hx_certified'
    # -- the spy: PQ's list read in (c), its three files in the evaluation, each by the path PQ's list names; nothing refused
    opened = summary['opened_files']
    assert opened['refused'] == []
    seen = {(e['path'], e['phase']) for e in opened['events'] if e['kind'] == 'input'}
    assert (os.path.realpath(pq_list), 'admission') in seen
    assert all((os.path.realpath(p), 'evaluation') in seen for p in state['files'].values())
    assert {os.path.realpath(r) for r in (layout.pq_small, layout.pq_big)} <= set(opened['watched_dirs'])
    # -- the method's files: PQ found them through hb_store's move record, the entry through the same record; the same files on D:
    dst = Path(hc.big_dir('method', PAIR))
    document = json.loads(Path(state['files']['pq.json']).read_text(encoding='utf-8'))
    move = record['arms']['method']['move']
    assert document['moves']['move_record']['path'] == str(Path(layout.method_records)/PAIR/hp.MOVE_RECORD_NAME) == move['files']['output/disp.npz']['record']
    assert document['moves']['move_record']['sha256'] == io_.sha_file(document['moves']['move_record']['path']) and document['moves']['move_record']['passed'] is True
    assert sorted(document['moves']['moved']) == sorted(document['method_files'])                                   # PQ found every method file through the move record (none at the working cell)
    both = sorted(set(document['method_files']) & set(move['files']))
    assert {'output/final.npz', 'output/disp.npz', 'output/cert.json', 'output/line.json', 'terminal.json'} <= set(both)
    for rel_path in both:
        mine, theirs = document['method_files'][rel_path], move['files'][rel_path]
        assert Path(mine['path']) == Path(theirs['dst']) == dst/rel_path and mine['sha256'] == theirs['finalized_sha256'] == theirs['sha256_after'], rel_path
    # -- the truth: this pair's four label files, read in the evaluation only
    assert sorted(entry.reads) == sorted(str(entry.data/s/H.LABEL_FILES[k]) for s in entry.subjects for k in ('seg35', 'seg4'))
    # -- the order (C22.H3-3): the entry's master lock is the one PQ's unit looks for; once it is there the PQ unit refuses before anything
    prod = hin.holdout_layout(layout.code, pairs=(PAIR,))
    assert Path(H.MASTER_LOCK) == Path(prod.ev_master_lock) == Path(layout.ev_master_lock) and Path(H.MASTER_LOCK).is_file()
    assert REAL_ENTRY['HOLDOUT_RESULTS'] == Path(REAL_HOME) and REAL_ENTRY['MASTER_LOCK'] == Path(REAL_HOME)/hp.EV_EVAL_DIR_NAME/hp.EV_MASTER_LOCK_NAME
    assert W.run_out(layout, deps=W.deps()) == hu.EXIT['refused'] and io_.read_json(W.refusals(layout)[-1])['stage'] == 'order'


def test_u5_a_release_note_that_does_not_hold_is_ev1_through_open_holdout_never_a_refusal(merged):
    """The direct test's Ev1 case through open_holdout: (c) records the release note that does not hold (passed false), admits, and the evaluation goes on; the pair's PQ reads Ev1 not passed
    (F4 不判（评测侧）), its F1-F3 arms are evaluated as usual."""
    entry = merged(note='放行：{sha}，但再看看')
    code, context, record, summary = evaluated(entry)
    rel, states, pq = context['pq_release'], context['pq_states'], record['pq']
    assert code == 0 and summary['stopped'] is None and record['complete'] is True
    assert rel['passed'] is False and rel['problems'] and states[PAIR]['ev1']['passed'] is False and pq['ev1']['passed'] is False
    assert all(record['arms'][a['name']]['outcome'] == 'output' for a in entry.table['arms'])
