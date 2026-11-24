"""MC0P's one approved retry (runs/20261104-mc0p/plan.md revision 2): `queue --retry-of`, `decide --merge`; CPU only, made-up records and synthetic small volumes, no real image, label, prediction or field.

- the plan: the control and the cells not done in the first batch, from its records, in the lead's order; only this first batch and this new root;
- the first batch is read only: every entry that writes refuses a path under it, and a whole fake batch leaves its bytes as they were;
- the strict gate: no allow list, no hold for the chain; the control compared, both ways, neither a failure;
- the merged reading: at least three branches, the same rules; v1 wins where both have a cell; source_root; the first batch's own reading quoted first;
- the dry run; one real synthetic retry through run-one, evaluate and the merged reading, in a subprocess.
"""
import csv
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from ..io_seg._helpers import write_json
from .test_mc0p_tools import EARLY, FROZEN, MC0, MI, PY, ROOT, RUN, Machine, chain_files, core, decision, dry_world, env, mc, needs_frozen, no_real_results, pipeline_world, snapshot  # noqa: F401

PAIRS = core.PAIRS
OOM_IN_V1 = (('Q2', 'long', PAIRS[3]), ('Q2s', 'long', PAIRS[4]), ('Q3', 'long', PAIRS[0]), ('Q3x3', 'long', PAIRS[0]), ('Q3h', 'long', PAIRS[0]))
SKIPPED_IN_V1 = (('Q2', 'fine', PAIRS[3]), ('Q2', 'long', PAIRS[4]), ('Q2', 'fine', PAIRS[4]))+tuple((arm, 'long', p) for arm in ('Q3', 'Q3x3', 'Q3h') for p in PAIRS[1:])
NOT_DONE = set(OOM_IN_V1+SKIPPED_IN_V1)
BASE = .85
FIELD = np.arange(6.).reshape(1, 1, 2, 3)
READING_V1 = dict(reading=decision.READ_SEE, reading_basis='effect', scope_note='没读到的臂（这些档不能算"看不到"）：long：Q2', tiers=dict(
    long=dict(comparable=True, best_arm='Q1', delta_seg35=.01736835, delta_seg4=.00977012, positive_seg35=5, positive_seg4=5, no_headroom=False, headroom=True, excluded=dict(Q2='x'), reason=None),
    fine=dict(comparable=False, best_arm=None, delta_seg35=None, delta_seg4=None, positive_seg35=None, positive_seg4=None, no_headroom=False, headroom=False, excluded=dict(Q2='x'), reason='没有符合条件的 K > 1 臂')))


def metrics_of(seg35, sd=.07):
    metrics = {m: .5 for m in decision.MEASURES}
    metrics.update(seg35=seg35, seg4=seg35+.05, sdlogj_official=sd)
    return dict(metrics=metrics, certificate=dict(accepted_alpha=1., n_infeasible_at_1=0, status_at_1='certified_feasible'))


def write_cell(root, key, *, status='done', seg35=BASE, retry_of=None, field=None, evaluated=True, oracle=False):
    cell = mc.Cell(*key)
    d = core.run_dir(root, cell)
    d.mkdir(parents=True)
    record = dict(run_id=cell.run_id, arm=key[0], tier=key[1], pair=key[2], status=status, oracle=oracle, scale=dict(images_equal_a5p_inputs=True, D_multiplier=2., D_factor=1.), field=None if field is None else dict(u_sha256='a'*64))
    if retry_of is not None:
        record['retry_of'] = retry_of
    write_json(d/'run.json', record)
    if field is not None:
        np.savez(d/'field.npz', u=field)
    if evaluated and status == 'done':
        write_json(d/'metrics.json', metrics_of(seg35))
    write_json(d/'queue_done.json', dict(run_id=cell.run_id, unit='x', rc=0, unit_seconds=10., state=status, arm_stop='cuda_oom' if status == 'cuda_oom' else None))


def delta_of(values, arm, tier, pair):
    """values {(arm, tier): differences per pair, or one number}."""
    d = values.get((arm, tier), -.1 if arm != 'Q0' else 0.)
    return d if isinstance(d, (int, float)) else d[PAIRS.index(pair)]


def first_batch(v1, values):
    for cell in mc.plan_runs():
        key = (cell.arm, cell.tier, cell.pair)
        if key in OOM_IN_V1:
            write_cell(v1, key, status='cuda_oom')
        elif key not in NOT_DONE:
            write_cell(v1, key, seg35=BASE+delta_of(values, *key), field=FIELD if mc.Cell(*key) == mc.Cell(mc.CONTROL.arm, mc.CONTROL.tier, mc.CONTROL.pair) else None)
    (v1/'analysis').mkdir()
    write_json(v1/'analysis/decision_mc0p.json', READING_V1)
    (v1/'analysis/long.csv').write_text('the first batch\n')


@pytest.fixture
def world(monkeypatch):
    base = Path(tempfile.mkdtemp(prefix='mc0pr_'))      # not the test runner's tmp_path: its name would fail the start-string check
    v1, v1b = base/'mc0p_v1', base/'mc0p_v1b'
    monkeypatch.setattr(mc, 'RETRY_OF_ROOT', v1)
    monkeypatch.setattr(mc, 'RETRY_ROOT', v1b)
    v1.mkdir()
    yield v1, v1b
    shutil.rmtree(base)


def options(root, retry_of=None):
    return RUN.argparse.Namespace(root=str(root), commit=None, wait_minutes=30, retry_of=None if retry_of is None else str(retry_of))


def lead_order():
    deciding = [('Q2', 'long', PAIRS[3]), ('Q2', 'long', PAIRS[4]), ('Q2', 'fine', PAIRS[3]), ('Q2', 'fine', PAIRS[4]), ('Q2s', 'long', PAIRS[4])]
    return [(mc.CONTROL.arm, mc.CONTROL.tier, mc.CONTROL.pair)]+deciding+[(arm, 'long', p) for arm in ('Q3', 'Q3x3', 'Q3h') for p in PAIRS]


# ---- the plan --------------------------------------------------------------------------------------------------------------------------
def test_the_constants_are_the_approved_paths():
    assert mc.RETRY_OF_ROOT == Path('/home/rmcfz/seq_vp_tag_results/mc0p_v1') and mc.RETRY_ROOT == Path('/home/rmcfz/seq_vp_tag_results/mc0p_v1b') and mc.EXPECTED_NOT_DONE == 20
    assert (mc.CONTROL.arm, mc.CONTROL.tier, mc.CONTROL.short) == ('Q2', 'long', '0438-0439')


def test_the_plan_is_the_control_and_the_cells_not_done_in_the_lead_order(world):
    v1, _ = world
    first_batch(v1, {})
    plan = mc.retry_plan(v1)
    assert [(c.arm, c.tier, c.pair) for c in plan] == lead_order() and len(plan) == 21 and len(NOT_DONE) == 20
    assert plan[0] == mc.CONTROL and plan[0].run_id == 'mc0p_Q2_long_0438-0439' and plan[0].unit == 'tag-mc0pb-q2-long-0438-0439'
    assert core.run_dir('/r', plan[1]) == Path('/r/runs/Q2_long/0441-0442')              # the layout of the first batch
    done = [c for c in mc.plan_runs() if (c.arm, c.tier, c.pair) not in NOT_DONE and c.run_id != mc.CONTROL.run_id]
    assert len(done) == 24 and not {c.run_id for c in done} & {c.run_id for c in plan}


def test_the_plan_comes_from_the_records(world):
    v1, _ = world
    first_batch(v1, {})
    d = core.run_dir(v1, mc.Cell('Q1', 'long', PAIRS[2]))                                 # another status counts as not done, whatever it is
    write_json(d/'run.json', dict(status='time_cap'))
    shutil.rmtree(core.run_dir(v1, mc.Cell('Q0', 'fine', PAIRS[1])))                      # no record: not done
    keys = [(c.arm, c.tier, c.pair) for c in mc.retry_plan(v1)]
    assert len(keys) == 23 and keys[1] == ('Q0', 'fine', PAIRS[1]) and keys[2] == ('Q1', 'long', PAIRS[2])


def test_a_control_not_done_in_the_first_batch_stops(world):
    v1, _ = world
    with pytest.raises(SystemExit, match='control'):
        mc.retry_plan(v1)


@pytest.mark.parametrize('case', ('other first batch', 'other new root', 'v1 as new root', 'v1 without retry', 'new root without retry'))
def test_only_this_first_batch_and_this_new_root(world, case, capsys):
    v1, v1b = world
    first_batch(v1, {})
    before = snapshot(v1)
    other = v1.parent/'mc0p_other'
    root, retry_of = {'other first batch': (v1b, other), 'other new root': (other, v1), 'v1 as new root': (v1, v1), 'v1 without retry': (v1, None), 'new root without retry': (v1b, None)}[case]
    assert RUN.queue(options(root, retry_of), RetrySystem(v1, v1b)) == 2
    assert capsys.readouterr().err and not v1b.exists() and not other.exists() and snapshot(v1) == before and not (v1/'queue.lock').exists()    # before the lock file


def test_the_real_roots_pass_and_others_do_not():
    assert mc.retry_problem('/home/rmcfz/seq_vp_tag_results/mc0p_v1b', '/home/rmcfz/seq_vp_tag_results/mc0p_v1') is None
    assert mc.retry_problem('/home/rmcfz/seq_vp_tag_results/mc0p_v1c', '/home/rmcfz/seq_vp_tag_results/mc0p_v1')
    assert mc.retry_problem('/home/rmcfz/seq_vp_tag_results/mc0p_v1b', '/home/rmcfz/seq_vp_tag_results/mc0_v1')


def test_a_run_of_the_retry_refuses_a_done_cell_and_another_root(world):
    v1, v1b = world
    first_batch(v1, {})

    def one(arm, tier, pair, out):
        return RUN.run_one(RUN.argparse.Namespace(arm=arm, tier=tier, pair=pair, out=str(out), device='cpu', code_sha='x', retry_of=str(v1)))
    with pytest.raises(SystemExit, match='not run again'):
        one('Q1', 'long', PAIRS[0], core.run_dir(v1b, mc.Cell('Q1', 'long', PAIRS[0])))
    with pytest.raises(SystemExit, match='goes to'):
        one('Q2', 'long', PAIRS[3], core.run_dir(v1.parent/'mc0p_other', mc.Cell('Q2', 'long', PAIRS[3])))


# ---- the first batch is read only ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('argv', (['decide'], ['evaluate'], ['queue']))
def test_the_entries_that_write_refuse_the_first_batch(world, argv, capsys):
    v1, _ = world
    first_batch(v1, {})
    before = snapshot(v1)
    assert RUN.main(argv+['--root', str(v1)]) == 2
    assert 'read only' in capsys.readouterr().err and snapshot(v1) == before


def test_a_run_refuses_to_write_under_the_first_batch(world, capsys):
    v1, _ = world
    first_batch(v1, {})
    before = snapshot(v1)
    out = core.run_dir(v1, mc.Cell('Q2', 'long', PAIRS[3]))/'again'
    assert RUN.main(['run-one', '--arm', 'Q2', '--tier', 'long', '--pair', PAIRS[3], '--out', str(out)]) == 2
    assert 'read only' in capsys.readouterr().err and snapshot(v1) == before
    with pytest.raises(SystemExit, match='read only'):
        decision.write_analysis(v1)


# ---- the fake machine -----------------------------------------------------------------------------------------------------------------------
class RetrySystem(Machine):
    """A quiet machine; run units write what a real run would, the evaluation writes metrics for each record that is done."""

    def __init__(self, v1, v1b, values=None, *, behaviour=None, control_field=FIELD):
        super().__init__()
        self.launched, self.argvs, self.clock, self.v1, self.v1b, self.values, self.behaviour, self.control_field, self.evaluate = [], {}, 0., v1, v1b, values or {}, behaviour, control_field, None

    def units(self):
        return [f'{RUN.RETRY_QUEUE_UNIT}.service']+list(self.other)

    def now(self):
        return self.clock

    def sleep(self, seconds):
        self.clock += seconds

    def code_checkout(self, root, commit):
        return Path(root)/'code', 'fakesha'

    def run(self, argv, log):
        unit = next(a for a in argv if a.startswith('--unit=')).split('=', 1)[1]
        self.launched.append(unit)
        self.argvs[unit] = argv
        if unit == RUN.RETRY_K1_UNIT:
            MC0.write_json(argv[argv.index('--out')+1], dict(bitwise=True))
            return 0
        if unit == RUN.RETRY_EVAL_UNIT:
            if self.evaluate:
                return self.evaluate(argv)
            for cell in mc.plan_runs():
                d = core.run_dir(self.v1b, cell)
                record = MC0.read_json(d/'run.json')
                if record is not None and record['status'] == 'done' and not (d/'metrics.json').exists():
                    write_json(d/'metrics.json', metrics_of(BASE+delta_of(self.values, cell.arm, cell.tier, cell.pair)))
            return 0
        cell = mc.require_cell(argv[argv.index('--arm')+1], argv[argv.index('--tier')+1], argv[argv.index('--pair')+1])
        assert argv[argv.index('--retry-of')+1] == str(self.v1)
        rc, status, seconds = self.behaviour(cell) if self.behaviour else (0, 'done', 10.)
        self.clock += seconds
        out = Path(argv[argv.index('--out')+1])
        if status is None:                                 # failed without a record
            return rc
        MC0.write_json(out/'run.json', dict(run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, status=status, oracle=False, retry_of=str(mc.RETRY_OF_ROOT), scale=dict(images_equal_a5p_inputs=True, D_multiplier=2., D_factor=1.)))
        if status == 'done' and (cell.arm, cell.tier, cell.pair) == (mc.CONTROL.arm, mc.CONTROL.tier, mc.CONTROL.pair) and self.control_field is not None:
            np.savez(out/'field.npz', u=self.control_field)
        return rc


SMALL = {('Q1', 'long'): .002, ('Q2', 'long'): .002, ('Q2', 'fine'): .002, ('Q2s', 'long'): .002}


# ---- the gate --------------------------------------------------------------------------------------------------------------------------------
def test_the_gate_of_the_retry_has_no_allow_list_and_no_hold():
    gate = RUN.RETRY_GATE
    assert gate.own_units == ('tag-mc0pb-queue.service',) and gate.gpu_hold is None and gate.load_limit == 1.5 and gate.gpu_free_mib == 15*1024 and gate.mem_available_gib == 7.


def quiet():
    m = Machine()
    m.other, m.load, m.free_gpu, m.avail = ['tag-mc0pb-queue.service'], 1.49, 15*1024., 7.
    return m


def test_the_gate_of_the_retry_ignores_the_last_member_and_the_deadline(tmp_path, monkeypatch):
    chain_files(tmp_path, monkeypatch, log='x waiting (member): y\n', state=('train_B1.started', 'register.done'))
    monkeypatch.setattr(RUN, 'wall_clock', lambda: EARLY.replace(day=9))
    assert RUN.unv2_hold_reason() is not None                                               # the first batch's hold would close
    assert MC0.busy_reason(quiet(), tmp_path, True, RUN.RETRY_GATE) is None


@pytest.mark.parametrize('field,value,word', (('load', 1.5, 'load'), ('free_gpu', 15*1024-1., 'GPU free'), ('avail', 6.99, 'MemAvailable'), ('free_home', 99., '/home'),
                                              ('other', ['tag-mc0pb-queue.service', 'tag-unv2-chain.service'], 'other units'), ('other', ['tag-mc0p-queue.service'], 'other units'),
                                              ('other', ['seq-vp-x.service'], 'other units'), ('other', ['tag-seed-queue.service'], 'other units'), ('other', ['systemctl failed with exit 1'], 'other units')))
def test_the_gate_of_the_retry_blocks(tmp_path, field, value, word):
    m = quiet()
    setattr(m, field, value)
    assert word in MC0.busy_reason(m, tmp_path, True, RUN.RETRY_GATE)


def test_next_to_the_training_the_retry_waits_30_minutes_and_stops(world):
    v1, v1b = world
    first_batch(v1, {})
    system = RetrySystem(v1, v1b)
    system.other = ['tag-unv2-chain.service']
    assert RUN.queue(options(v1b, v1), system) == 8
    assert system.launched == [] and 'tag-unv2-chain' in json.loads((v1b/'queue_stop.json').read_text())['reason'] and system.clock >= 30*60


# ---- the queue ------------------------------------------------------------------------------------------------------------------------------
def test_the_retry_runs_its_plan_once_and_leaves_the_first_batch_as_it_was(world):
    v1, v1b = world
    first_batch(v1, SMALL)
    before = snapshot(v1)
    system = RetrySystem(v1, v1b, SMALL)
    assert RUN.queue(options(v1b, v1), system) == 0
    assert system.launched[0] == RUN.RETRY_K1_UNIT == 'tag-mc0pb-k1check' and system.launched[-1] == RUN.RETRY_EVAL_UNIT == 'tag-mc0pb-eval'
    assert system.launched[1:-1] == [mc.RetryCell(*key).unit for key in lead_order()] and all(u.startswith('tag-mc0pb-') for u in system.launched)
    assert snapshot(v1) == before
    batch = json.loads((v1b/'batch.json').read_text())
    assert batch['retry_of'] == str(v1) and '总负责人' in batch['approval'] and batch['not_done_in_first_batch'] == 20 and batch['gate']['chain_hold'] is None and batch['gate']['own_units'] == ['tag-mc0pb-queue.service'] and len(batch['plan']) == 21
    assert batch['caps']['device_gib'] == 9. and batch['caps']['run_seconds'] == 900.
    assert 'retry of' in (v1b/'queue.log').read_text() and json.loads((v1b/'control.json').read_text())['control_bitwise'] is True
    end = json.loads((v1b/'queue_end.json').read_text())
    assert end['reading'] == decision.READ_NONE and (v1b/'analysis/decision_mc0p_merged.json').is_file() and not (v1b/'analysis/decision_mc0p.json').exists()
    again = RetrySystem(v1, v1b, SMALL)
    assert RUN.queue(options(v1b, v1), again) == 0
    assert again.launched == [RUN.RETRY_EVAL_UNIT] and snapshot(v1) == before


def test_another_oom_stops_the_arm_as_before_and_the_record_stays(world):
    v1, v1b = world
    first_batch(v1, SMALL)

    def behaviour(cell):
        return 0, 'cuda_oom' if (cell.arm, cell.tier, cell.pair) == ('Q2', 'long', PAIRS[3]) else 'done', 10.
    system = RetrySystem(v1, v1b, SMALL, behaviour=behaviour)
    assert RUN.queue(options(v1b, v1), system) == 0
    assert sum(u.startswith('tag-mc0pb-q2-') for u in system.launched) == 2 and sum(u.startswith('tag-mc0pb-q2s-') for u in system.launched) == 1     # the control, then that one; the arm stops
    assert json.loads((core.run_dir(v1b, mc.Cell('Q2', 'long', PAIRS[3]))/'run.json').read_text())['status'] == 'cuda_oom'
    result = json.loads((v1b/'analysis/decision_mc0p_merged.json').read_text())
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ and 'Q2' in result['tiers']['long']['excluded'] and result['reading_basis'] == 'not_read'
    assert list(result['arms_stopped']['v1b']) == ['Q2']


@pytest.mark.parametrize('rc,status', ((0, 'cuda_oom'), (1, None)))
def test_the_control_failing_stops_no_arm_and_the_cells_of_the_reading_go_on(world, rc, status):
    """Plan revision 3: the control is a record only.  With a record (device out of memory) or without (the unit failed)."""
    v1, v1b = world
    first_batch(v1, SMALL)
    before = snapshot(v1)

    def behaviour(cell):
        return (rc, status, 10.) if (cell.arm, cell.tier, cell.pair) == lead_order()[0] else (0, 'done', 10.)
    system = RetrySystem(v1, v1b, SMALL, behaviour=behaviour)
    assert RUN.queue(options(v1b, v1), system) == 0
    assert system.launched[1:-1] == [mc.RetryCell(*key).unit for key in lead_order()]      # all 21, the 4 Q2 cells of the reading among them
    done = json.loads((core.run_dir(v1b, mc.CONTROL)/'queue_done.json').read_text())
    assert done['arm_stop'] is None and done['control_stop'] and done['state'] == (status or 'failed') and 'no arm stops' in (v1b/'queue.log').read_text()
    result = json.loads((v1b/'analysis/decision_mc0p_merged.json').read_text())
    assert result['arms_stopped']['v1b'] == {} and result['reading'] == decision.READ_NONE and result['reading_basis'] == 'effect' and result['control']['control_bitwise'] is None and result['control']['control_stop'] == done['control_stop']
    assert result['merged']['cells'][mc.CONTROL.run_id] == str(v1) and result['merged']['cells']['mc0p_Q2_long_0441-0442'] == str(v1b)
    assert '没有比较' in (v1b/'analysis/summary_mc0p_merged.md').read_text() and '不停臂' in (v1b/'analysis/summary_mc0p_merged.md').read_text() and snapshot(v1) == before


def test_if_k35_really_cannot_run_the_next_q2_cell_stops_the_arm(world):
    v1, v1b = world
    first_batch(v1, SMALL)

    def behaviour(cell):
        return 0, 'cuda_oom' if cell.arm == 'Q2' else 'done', 10.
    system = RetrySystem(v1, v1b, SMALL, behaviour=behaviour)
    assert RUN.queue(options(v1b, v1), system) == 0
    assert sum(u.startswith('tag-mc0pb-q2-') for u in system.launched) == 2                  # the control, then one more run, as the review says
    stopped = json.loads((v1b/'queue_end.json').read_text())['ledger']['arms_stopped']
    assert list(stopped) == ['Q2'] and stopped['Q2']['run_id'] == 'mc0p_Q2_long_0441-0442'


def test_a_failure_of_the_control_without_a_record_still_counts_as_a_failure_in_a_row_and_the_hand_path_works(world):
    v1, v1b = world
    first_batch(v1, SMALL)
    system = RetrySystem(v1, v1b, SMALL, behaviour=lambda cell: (1, None, 5.))
    assert RUN.queue(options(v1b, v1), system) == 8
    assert len(system.launched) == 1+3 and '3 failures' in json.loads((v1b/'queue_stop.json').read_text())['reason']
    assert not (v1b/'analysis').exists()
    assert RUN.main(['decide', '--root', str(v1b), '--merge', str(v1)]) == 0                  # plan's hand path after such a STOP
    result = json.loads((v1b/'analysis/decision_mc0p_merged.json').read_text())
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ and result['merged']['statuses']['mc0p_Q2_long_0441-0442'] == 'failed'


def test_a_described_cell_failing_without_a_record_still_gets_the_merged_reading(world):
    """The review's case: only queue_done.json, no run.json, in an arm that is only described: the row says nothing, the reading is written."""
    v1, v1b = world
    first_batch(v1, SMALL)
    missing = ('Q3', 'long', PAIRS[1])

    def behaviour(cell):
        return (1, None, 5.) if (cell.arm, cell.tier, cell.pair) == missing else (0, 'done', 10.)
    system = RetrySystem(v1, v1b, SMALL, behaviour=behaviour)
    assert RUN.queue(options(v1b, v1), system) == 0
    d = core.run_dir(v1b, mc.Cell(*missing))
    assert (d/'queue_done.json').is_file() and not (d/'run.json').exists()
    result = json.loads((v1b/'analysis/decision_mc0p_merged.json').read_text())
    assert result['reading'] == decision.READ_NONE and result['merged']['statuses']['mc0p_Q3_long_0439-0440'] == 'failed' and result['merged']['cells']['mc0p_Q3_long_0439-0440'] == str(v1b)
    assert list(result['arms_stopped']['v1b']) == ['Q3'] and json.loads((v1b/'queue_end.json').read_text())['reading'] == decision.READ_NONE and not (v1b/'queue_stop.json').exists()


def test_write_merged_takes_a_cell_with_queue_done_alone(world, capsys):
    v1, v1b = world
    first_batch(v1, SMALL)
    v1b.mkdir()
    for key in lead_order():
        write_cell(v1b, key, seg35=BASE+delta_of(SMALL, *key), retry_of=str(mc.RETRY_OF_ROOT), field=FIELD if key == lead_order()[0] else None)
    d = core.run_dir(v1b, mc.Cell('Q3h', 'long', PAIRS[2]))
    (d/'run.json').unlink()
    (d/'metrics.json').unlink()
    write_json(d/'queue_done.json', dict(run_id='mc0p_Q3h_long_0440-0441', unit='x', rc=1, unit_seconds=5., state='failed', arm_stop='K = 35 run failed without a record', ended_utc='x'))
    assert RUN.main(['decide', '--root', str(v1b), '--merge', str(v1)]) == 0
    result = json.loads((v1b/'analysis/decision_mc0p_merged.json').read_text())
    assert result['reading'] == decision.READ_NONE and result['merged']['statuses']['mc0p_Q3h_long_0440-0441'] == 'failed'
    with open(v1b/'analysis/long.csv') as f:
        assert not [r for r in csv.DictReader(f) if r['run_id'] == 'mc0p_Q3h_long_0440-0441']


def test_a_merged_reading_that_refuses_writes_queue_stop(world):
    v1, v1b = world
    first_batch(v1, SMALL)
    (v1/'analysis/decision_mc0p.json').unlink()
    system = RetrySystem(v1, v1b, SMALL)
    assert RUN.queue(options(v1b, v1), system) == 8
    assert 'the merged reading' in json.loads((v1b/'queue_stop.json').read_text())['reason'] and 'no reading to quote' in (v1b/'queue.log').read_text()
    assert not (v1b/'queue_end.json').exists() and not (v1b/'analysis/decision_mc0p_merged.json').exists()


def test_the_cap_costs_description_first(world):
    v1, v1b = world
    first_batch(v1, SMALL)
    system = RetrySystem(v1, v1b, SMALL, behaviour=lambda cell: (0, 'done', 800.))
    assert RUN.queue(options(v1b, v1), system) == 0
    runs = system.launched[1:-1]
    assert len(runs) == 12 and runs[:6] == [mc.RetryCell(*key).unit for key in lead_order()[:6]] and 'cap of' in json.loads((v1b/'queue_end.json').read_text())['end']


def test_the_start_strings_of_the_retry(world):
    v1, v1b = world
    first_batch(v1, {})
    code = v1b/'code'
    strings = [RUN.queue_launch(v1b, retry_of=v1), RUN.k1_argv(code, v1b, RUN.RETRY_K1_UNIT), RUN.eval_argv(code, v1b, RUN.RETRY_EVAL_UNIT)]+[RUN.run_argv(c, code, v1b, 'sha', v1) for c in mc.retry_plan(v1)]
    for argv in strings:
        text = shlex.join(argv)
        assert 'py'+'test' not in text and text.startswith('systemd-run --user --unit=tag-mc0pb-') and '--property=MemoryMax=8G' in text and 'run_mc0p.py' in text
    assert f'queue --root {v1b} --retry-of {v1}' in shlex.join(strings[0])
    assert all(f'--retry-of {v1}' in shlex.join(argv) for argv in strings[3:])
    evaluation = shlex.join(strings[2])
    assert f'evaluate --root {v1b}' in evaluation and '--setenv=CUDA_VISIBLE_DEVICES=' in evaluation and '--setenv=NUMBA_NUM_THREADS=1' in evaluation and '--property=CPUQuota=100%' in evaluation
    assert RUN.EVAL_POOL <= 4 and '--retry-of' not in shlex.join(RUN.queue_launch())       # the first batch's string has not changed


# ---- the control -----------------------------------------------------------------------------------------------------------------------------
def test_the_control_compares_u_both_ways_and_says_when_it_could_not(world):
    v1, v1b = world
    first_batch(v1, {})
    assert decision.control_record(v1, v1b)['control_bitwise'] is None                      # no retry yet
    d = core.run_dir(v1b, mc.CONTROL)
    d.mkdir(parents=True)
    np.savez(d/'field.npz', u=FIELD)
    same = decision.control_record(v1, v1b)
    assert same['control_bitwise'] is True and same['u_sha256_v1'] == same['u_sha256_v1b'] and same['cell'] == 'mc0p_Q2_long_0438-0439'
    changed = FIELD.copy()
    changed[0, 0, 0, 0] = np.nextafter(changed[0, 0, 0, 0], 1.)
    np.savez(d/'field.npz', u=changed)
    other = decision.control_record(v1, v1b)
    assert other['control_bitwise'] is False and other['u_sha256_v1'] != other['u_sha256_v1b']


def test_a_control_that_differs_is_no_failure_and_changes_no_reading(world):
    v1, v1b = world
    first_batch(v1, SMALL)
    changed = FIELD+1e-9
    same, different = RetrySystem(v1, v1b, SMALL), None
    assert RUN.queue(options(v1b, v1), same) == 0
    reading = json.loads((v1b/'analysis/decision_mc0p_merged.json').read_text())
    shutil.rmtree(v1b)
    different = RetrySystem(v1, v1b, SMALL, control_field=changed)
    assert RUN.queue(options(v1b, v1), different) == 0
    result = json.loads((v1b/'analysis/decision_mc0p_merged.json').read_text())
    assert reading['control']['control_bitwise'] is True and result['control']['control_bitwise'] is False
    assert result['reading'] == reading['reading'] and result['tiers'] == reading['tiers']
    text = (v1b/'analysis/summary_mc0p_merged.md').read_text()
    assert 'control_bitwise: false' in text and '设备不可重复' in text and '不算失败' in text and '不改读数' in text


# ---- the merged reading -----------------------------------------------------------------------------------------------------------------------
def merged_with(world, values, retry_values=None, retry=None):
    """The first batch, the retry's cells as given (default: all done), and write_merged."""
    v1, v1b = world
    first_batch(v1, values)
    v1b.mkdir()
    for key in lead_order() if retry is None else retry:
        write_cell(v1b, key, seg35=BASE+delta_of(values if retry_values is None else retry_values, *key), retry_of=str(mc.RETRY_OF_ROOT), field=FIELD if key == lead_order()[0] else None)
    return decision.write_merged(v1b, v1)


def test_merged_the_first_batch_alone_was_not_read_and_now_it_is_no_headroom(world):
    v1, v1b = world
    alone = None
    result = merged_with(world, SMALL)
    alone = decision.decide(decision.collect(v1))
    assert alone['reading'] == decision.READ_MIDDLE_NOT_READ                                 # the gaps of the first batch
    assert result['reading'] == decision.READ_NONE and result['reading_basis'] == 'effect' and result['scope_note'] is None and result['schema'] == decision.MERGED_SCHEMA
    assert all(r['comparable'] and r['no_headroom'] for r in result['tiers'].values())


def test_merged_headroom_from_an_arm_of_the_first_batch_stays(world):
    result = merged_with(world, {('Q1', 'long'): .02})
    assert result['reading'] == decision.READ_SEE and result['tiers']['long']['best_arm'] == 'Q1' and result['tiers']['fine']['comparable'] is True


def test_merged_headroom_can_come_from_the_retried_cells(world):
    values = dict(SMALL)
    values[('Q2', 'fine')] = .03
    result = merged_with(world, values)
    assert result['reading'] == decision.READ_SEE and result['tiers']['fine']['headroom'] and result['tiers']['fine']['best_arm'] == 'Q2'
    assert result['merged']['cells']['mc0p_Q2_fine_0442-0443'].endswith('mc0p_v1b') and result['merged']['cells']['mc0p_Q2_fine_0438-0439'].endswith('mc0p_v1')


def test_merged_middle_band_of_the_effect(world):
    values = dict(SMALL)
    values[('Q2', 'long')] = .007
    result = merged_with(world, values)
    assert result['reading'] == decision.READ_MIDDLE and result['reading_basis'] == 'effect'


def test_merged_a_cell_the_retry_did_not_reach_is_the_first_batchs_failure_and_is_not_read(world):
    result = merged_with(world, SMALL, retry=[key for key in lead_order() if key != ('Q2s', 'long', PAIRS[4])])
    assert result['reading'] == decision.READ_MIDDLE_NOT_READ and result['reading_basis'] == 'not_read' and 'Q2s' in result['tiers']['long']['excluded']
    assert result['merged']['statuses']['mc0p_Q2s_long_0442-0443'] == 'cuda_oom' and result['merged']['cells']['mc0p_Q2s_long_0442-0443'].endswith('mc0p_v1')


def test_merged_the_rules_are_the_same_q0_over_the_bar_is_not_comparable(world):
    v1, v1b = world
    first_batch(v1, SMALL)
    v1b.mkdir()
    for key in lead_order():
        write_cell(v1b, key, seg35=BASE+delta_of(SMALL, *key), retry_of=str(mc.RETRY_OF_ROOT), field=FIELD if key == lead_order()[0] else None)
    for pair in PAIRS:
        path = core.run_dir(v1, mc.Cell('Q0', 'fine', pair))/'metrics.json'
        record = json.loads(path.read_text())
        record['metrics']['sdlogj_official'] = .11
        path.write_text(json.dumps(record))
    result = decision.write_merged(v1b, v1)
    assert not result['tiers']['fine']['comparable'] and result['reading'] == decision.READ_MIDDLE_NOT_READ


def test_where_both_have_the_cell_the_first_batchs_is_used(world):
    v1, v1b = world
    retry_values = dict(SMALL)
    retry_values[('Q2', 'long')] = .4                                                         # only the control is on both sides, and it would change the mean a lot
    result = merged_with(world, SMALL, retry_values=retry_values)
    key = (mc.CONTROL.arm, mc.CONTROL.tier, mc.CONTROL.pair)
    rows = decision.merge_rows(decision.collect(v1), decision.collect(v1b), v1, v1b)
    assert rows[key]['source_root'] == str(v1) and rows[key]['metrics']['seg35'] == pytest.approx(BASE+.002)
    assert result['control']['metrics']['v1']['seg35'] == pytest.approx(BASE+.002) and result['control']['metrics']['v1b']['seg35'] == pytest.approx(BASE+.4)
    assert rows[('Q2', 'long', PAIRS[0])]['status'] == 'done'


def test_the_merged_files_quote_the_first_batch_and_say_the_two_conditions(world):
    v1, v1b = world
    merged_with(world, {('Q1', 'long'): .02})
    before = snapshot(v1)
    text = (v1b/'analysis/summary_mc0p_merged.md').read_text()
    head = text.split('# MC0P 合并读数')[0]
    assert head.startswith('**v1 自己的读数（原样') and decision.READ_SEE in head and 'Q1' in head and '+.0174 / +.0098' in head and '5/5' in head and 'long：Q2' in head
    assert '与训练同跑' in head and 'v1b 单独' in head and 'T88 的迭代数固定' in head and '不是计时运行' in head and 'control_bitwise: true' in head and '〔推断〕' in head
    assert '来自 v1b' in head and 'mc0p_Q3h_long_0442-0443' in head
    result = json.loads((v1b/'analysis/decision_mc0p_merged.json').read_text())
    assert result['first_batch_reading']['reading'] == decision.READ_SEE and result['first_batch_reading']['tiers']['long']['delta_seg35'] == .01736835 and result['oracle'] is False and result['conditions'] == decision.CONDITIONS
    with open(v1b/'analysis/long.csv') as f:
        rows = list(csv.DictReader(f))
    sources = {(r['arm'], r['tier'], r['pair']): r['source_root'] for r in rows}
    assert sources[('Q1', 'long', PAIRS[0])] == str(v1) and sources[('Q2', 'long', PAIRS[0])] == str(v1) and sources[('Q2', 'long', PAIRS[3])] == str(v1b) and all(r['oracle'] == 'false' for r in rows)
    assert len({(r['arm'], r['tier'], r['pair']) for r in rows}) == 45
    assert (v1/'analysis/long.csv').read_text() == 'the first batch\n' and not list((v1/'analysis').glob('previous_*')) and snapshot(v1) == before
    decision.write_merged(v1b, v1)                                                            # again: the earlier version set aside in the retry's root, none in the first batch
    assert len(list((v1b/'analysis').glob('previous_*/decision_mc0p_merged.json'))) == 1 and snapshot(v1) == before


@pytest.mark.parametrize('change', ('no retry_of', 'other retry_of', 'oracle true'))
def test_a_record_that_does_not_belong_writes_nothing(world, change):
    v1, v1b = world
    first_batch(v1, SMALL)
    v1b.mkdir()
    for key in lead_order():
        write_cell(v1b, key, seg35=BASE, retry_of=str(mc.RETRY_OF_ROOT))
    path = core.run_dir(v1b, mc.Cell('Q2', 'long', PAIRS[3]))/'run.json'
    record = json.loads(path.read_text())
    if change == 'no retry_of':
        del record['retry_of']
    elif change == 'other retry_of':
        record['retry_of'] = '/somewhere/else'
    else:
        record['oracle'] = True
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match='nothing written'):
        decision.write_merged(v1b, v1)
    assert not (v1b/'analysis').exists()


def test_the_decide_entry_of_the_retry(world, capsys):
    v1, v1b = world
    first_batch(v1, SMALL)
    v1b.mkdir()
    for key in lead_order():
        write_cell(v1b, key, seg35=BASE+delta_of(SMALL, *key), retry_of=str(mc.RETRY_OF_ROOT), field=FIELD if key == lead_order()[0] else None)
    assert RUN.main(['decide', '--root', str(v1b)]) == 2 and not (v1b/'analysis').exists()     # a part of the cells would give a misleading reading
    assert RUN.main(['decide', '--root', str(v1b), '--merge', str(v1.parent/'mc0p_other')]) == 2
    other = v1.parent/'mc0p_other'
    other.mkdir()
    assert RUN.main(['decide', '--root', str(other), '--merge', str(v1)]) == 2 and not (other/'analysis').exists()
    assert not (v1b/'analysis').exists()
    assert RUN.main(['decide', '--root', str(v1b), '--merge', str(v1)]) == 0 and (v1b/'analysis/decision_mc0p_merged.json').is_file()
    with pytest.raises(SystemExit):
        decision.write_merged(other, v1)
    capsys.readouterr()


# ---- the dry run ---------------------------------------------------------------------------------------------------------------------------
def test_the_dry_run_of_the_retry_prints_the_plan_and_writes_nothing(world, tmp_path, monkeypatch):
    v1, v1b = world
    first_batch(v1, {})
    before = snapshot(v1)
    lines = []
    assert RUN.dry_run(root=v1b, retry_of=v1, out=lines.append, **dry_world(tmp_path, monkeypatch)) == 0
    text = '\n'.join(lines)
    assert '24 done' not in text and '25 done, 5 cuda_oom, 15 never run, 0 other; 20 cells to run (the lead counted 20)' in text and 'problems: none' in text and decision.READ_SEE in text
    rows = [line for line in lines if '| tag-mc0pb-' in line]
    assert len(rows) == 21 and 'tag-mc0pb-q2-long-0438-0439' in rows[0] and 'tag-mc0pb-q3h-long-0442-0443' in rows[-1]
    assert 'no allow list, no hold for the chain' in text and f'--retry-of {v1}' in text and 'py'+'test' not in text.replace(str(tmp_path), '')
    assert snapshot(v1) == before and not v1b.exists()


def test_the_dry_run_of_the_retry_says_when_the_count_differs(world, tmp_path, monkeypatch):
    v1, v1b = world
    first_batch(v1, {})
    write_json(core.run_dir(v1, mc.Cell('Q1', 'long', PAIRS[2]))/'run.json', dict(status='time_cap'))
    lines = []
    assert RUN.dry_run(root=v1b, retry_of=v1, out=lines.append, **dry_world(tmp_path, monkeypatch)) == 8
    assert '21 cells not done' in lines[-1]


def test_the_dry_run_entry_refuses_another_first_batch(capsys):
    assert RUN.main(['--dry-run', '--retry-of', '/somewhere/mc0p_v1']) == 2
    assert 'only' in capsys.readouterr().err


# ---- one real synthetic retry -------------------------------------------------------------------------------------------------------------
RETRY_PIPELINE = """
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import mc0_core as core
core.frozen_api()
import mc0p_core as mc
import run_mc0p as run
world = json.loads(sys.argv[2])
v1, v1b = Path(world['root']), Path(world['root']).parent/'mc0p_v1b'
core.STAGE = Path(world['stage'])
core.SETUP_WORKERS = 1
core.TIERS.update(long=dict(iterations=(2, 2, 2, 4)), fine=dict(iterations=(2, 2, 2, 4), sigma_field=.5))
mc.SEG_ROOT, mc.A5P_INPUTS, mc.MC0_ROOT = Path(world['seg_root']), Path(world['a5p']), Path(world['mc0_root'])
mi = core.make_inputs()
mc.default_unet = lambda: mi.UNetSource(pred_root=Path(world['pred']))
pair = core.PAIRS[0]
for arm, tier in (('Q2', 'long'), ('Q1', 'long')):
    assert run.main(['run-one', '--arm', arm, '--tier', tier, '--pair', pair, '--out', str(core.run_dir(v1, mc.Cell(arm, tier, pair))), '--device', 'cpu', '--code-sha', 'first']) == 0
assert run.main(['evaluate', '--root', str(v1)]) == 0
assert run.main(['decide', '--root', str(v1)]) == 0
mc.RETRY_OF_ROOT, mc.RETRY_ROOT = v1, v1b
def tree():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(v1.rglob('*')) if p.is_file()}
before = tree()
for arm, tier in (('Q2', 'long'), ('Q0', 'long'), ('Q2s', 'long')):
    assert run.main(['run-one', '--arm', arm, '--tier', tier, '--pair', pair, '--out', str(core.run_dir(v1b, mc.Cell(arm, tier, pair))), '--device', 'cpu', '--code-sha', 'second', '--retry-of', str(v1)]) == 0
assert run.main(['evaluate', '--root', str(v1b)]) == 0
assert run.main(['decide', '--root', str(v1b), '--merge', str(v1)]) == 0
assert tree() == before, 'the first batch changed'
"""


@needs_frozen
def test_a_real_retry_through_everything(tmp_path, monkeypatch):
    world = pipeline_world(tmp_path, monkeypatch)
    env_ = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{FROZEN}:{ROOT}')
    done = subprocess.run([PY, '-B', '-c', RETRY_PIPELINE, str(ROOT/'experiments/mc0'), json.dumps(world)], env=env_, cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout+done.stderr
    v1, v1b = Path(world['root']), Path(world['root']).parent/'mc0p_v1b'
    for arm, tier, k in (('Q2', 'long', 35), ('Q0', 'long', 1), ('Q2s', 'long', 35)):
        d = core.run_dir(v1b, mc.Cell(arm, tier, PAIRS[0]))
        run, metrics = json.loads((d/'run.json').read_text()), json.loads((d/'metrics.json').read_text())
        assert run['retry_of'] == str(v1) and isinstance(run['retry_of'], str) and run['schema'] == 'bayesreg.mc0p.run.v1' and run['run_id'] == f'mc0p_{arm}_{tier}_0438-0439' and (run['arm'], run['tier'], run['pair'], run['k']) == (arm, tier, PAIRS[0], k)
        assert run['oracle'] is False and metrics['oracle'] is False and run['channel_sources']['oracle'] is False and run['data']['pair_sha256'] and run['status'] == 'done' and mc.oracle_of_run(run) is False
    assert 'retry_of' not in json.loads((core.run_dir(v1, mc.Cell('Q2', 'long', PAIRS[0]))/'run.json').read_text())
    result = json.loads((v1b/'analysis/decision_mc0p_merged.json').read_text())
    assert result['control']['control_bitwise'] is True and result['control']['u_sha256_v1'] == json.loads((core.run_dir(v1, mc.CONTROL)/'run.json').read_text())['field']['u_sha256']    # the same cell, the same bytes on a CPU
    assert result['merged']['cells']['mc0p_Q2_long_0438-0439'] == str(v1) and result['merged']['cells']['mc0p_Q0_long_0438-0439'] == str(v1b) and result['first_batch_reading']['reading'] == decision.READ_UNDECIDABLE
    with open(v1b/'analysis/long.csv') as f:
        sources = {(r['arm'], r['pair']): r['source_root'] for r in csv.DictReader(f)}
    assert sources[('Q1', PAIRS[0])] == str(v1) and sources[('Q0', PAIRS[0])] == str(v1b) and sources[('Q2', PAIRS[0])] == str(v1)
    assert (v1b/'analysis/summary_mc0p_merged.md').read_text().startswith('**v1 自己的读数')
