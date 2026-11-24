"""TC (runs/20261104-tc/plan.md; docs/bayesreg/notes/提速与换预分割器规划_20261104.md §5): experiments/tc/; CPU only, small synthetic volumes and made-up records; no solver, no GPU, no unit.

- the rule that takes a commit for a time, and what it checks of the two records;
- the comparison of the commit sequences with the original;
- the reading: every branch, its bars, its premises; only S_P decides; the two difference tables;
- the gate; only the first 5 pairs; the start strings;
- nothing is written to an existing result directory; the cleanup only under tc_v1 and only after a complete curve;
- the queue on a fake machine, with the real evaluation, curve, comparison, cleanup and reading on synthetic fields;
- the time ledger; the measurement's refusals, its timers, the segmenter part on a synthetic world, the entry part on a synthetic world (needs the frozen worktree).
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/tc'))
sys.path.insert(1, str(ROOT/'experiments/seed'))

import decide_tc as DEC  # noqa: E402
import inject_run as INJ  # noqa: E402
import measure_tc as MEASURE  # noqa: E402
import run_seed as SEED  # noqa: E402
import run_seed19 as S19  # noqa: E402
import run_tc as TC  # noqa: E402
import tc_curve as CURVE  # noqa: E402

from .test_mc0p_tools import FROZEN, PY, needs_frozen, pipeline_world  # noqa: E402
from .test_seed19_tools import make_entry as make_s19_entry  # noqa: E402
from .test_seed_tools import SHAPE, FakeSystem, make_entry as make_seed_entry, small_field, write_json  # noqa: E402

PAIRS = TC.PAIRS
ELAPSED = (8., 14., 15., 29., 31., 61., 89.5, 90., 121., 151., 211., 269., 275.)
REAL_SYSTEMS = (TC.System, SEED.System, S19.System)


class NoRealMachine:
    def __init__(self, *args, **kwargs):
        raise AssertionError('a queue on the real machine inside a test: a guard has regressed')


@pytest.fixture(autouse=True)
def no_real_machine(monkeypatch):
    """Every queue here runs on a fake machine, passed in.  A guard that regresses must fail the test, never start a real queue (2026-11-04, runs/20261104-seed19/plan.md revision 1)."""
    for module in (TC, SEED, S19):
        monkeypatch.setattr(module, 'System', NoRealMachine)


@pytest.fixture
def neutral():
    """A temporary directory whose path does not hold the word of the test runner: check_launch would refuse the start strings under tmp_path, as it should."""
    with tempfile.TemporaryDirectory(prefix='tc_tools_') as d:
        yield Path(d)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(root):
    return {str(p): sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def json_load(path):
    return json.loads(Path(path).read_text())


# ---- a synthetic world: truth, A5P's inputs, the entries, the originals --------------------------------------------------------------------------------------------------------------------------
def labels_of(index):
    """Blocks of a few voxels, each with its own label, moved one voxel from one subject to the next: the Dice is neither 0 nor 1 and changes with the field."""
    z, y, x = np.indices(SHAPE)
    block = ((z//3)*9+(y//3)*3+x//4+1) % 36
    return np.roll(block, index, axis=2).astype(np.int16)


def write_pair(d, pair, index, extra=None):
    fixed, moving = labels_of(index), labels_of(index+1)
    rng = np.random.default_rng(index)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d/'pair.npz', fixed=rng.uniform(.1, .7, SHAPE), moving=rng.uniform(.1, .7, SHAPE), fixed_seg35=fixed, moving_seg35=moving, fixed_seg4=(fixed % 5).astype(np.int16), moving_seg4=(moving % 5).astype(np.int16))
    a, b = pair.split('__')
    write_json(d/'manifest.json', dict(pair=pair, pair_sha256=sha(d/'pair.npz'), fixed_subject=a, moving_subject=b, shape_zyx=list(SHAPE), spacing_xyz=[1., 1., 1.], **(extra or {})))


def commit_field(pair_index, k, moved=0.):
    """The field of commit k (1-based): a small interior field growing with k; moved != 0: another path from there on."""
    return small_field(pair_index)*.5*k*(1.+moved)


def write_task(run, arm, pair, *, elapsed=ELAPSED, diverge_at=None):
    """A task directory as the frozen solver leaves it, before the cleanup: the commits with their records, the level files, the final map."""
    run = Path(run)
    index = PAIRS.index(pair)
    task = run/'pg_weak__four__330s'
    task.mkdir(parents=True, exist_ok=True)
    commits, history = [], []
    for k, t in enumerate(elapsed, 1):
        u = commit_field(index, k, .1 if diverge_at is not None and k >= diverge_at else 0.)
        name = f'committed_{k:04d}_L3.npz'
        np.savez(task/name, u=u)
        state = hashlib.sha256(np.ascontiguousarray(u).view(np.uint8)).hexdigest()
        commits.append(dict(path=name, sha256=sha(task/name), level=3, native=True, role='formal_outer_commit', certificate=dict(status='certified_feasible', min_jd_lower_bound=1e-4, n_infeasible=0, n_uncertain=0, state_hash=state), solver_commit_seconds=t, visible_seconds=t+50.))
        history.append(dict(level=3, outer_accepted=True, elapsed_seconds=t, D_candidate=2000.-10*k, F_candidate=2000.-10*k, GF_candidate=5.-.1*k, D_start=2000., F_start=2000., inner_accepted_steps=5))
        if k == 3:                                       # a rejected outer row is not a commit
            history.append(dict(level=3, outer_accepted=False, elapsed_seconds=t+.1))
    write_json(task/'commits.json', commits)
    write_json(task/'history.json', history)
    u0 = small_field(index)*.01
    np.savez(task/'level_3_preoptimizer_done.npz', u=u0)
    np.savez(task/'level_3_level_prepared.npz', u=u0)
    entry = SEED.read_json(run/'seed_entry.json') or {}
    done = dict(event='preoptimizer_done', level=3, path='level_3_preoptimizer_done.npz', sha256=sha(task/'level_3_preoptimizer_done.npz'), preoptimizer=dict(policy=INJ.POLICY, raw_hash=entry.get('u_sha256'), seed_hash=entry.get('tapered_sha256'), taper_L=4))
    prepared = dict(event='level_prepared', level=3, path='level_3_level_prepared.npz', sha256=sha(task/'level_3_level_prepared.npz'), prepare_seconds=49., transfer_recovery=dict(status='ready', accepted_alpha=1., attempts=[dict(alpha=1., certificate_passed=True, certificate=dict(min_jd_lower_bound=.01, n_infeasible=0, state_hash='h'))]))
    write_json(task/'levels.json', [done, prepared])
    final = commit_field(index, len(elapsed), .1 if diverge_at is not None else 0.)
    np.savez(task/'final_native.npz', u=final, warped=np.zeros(SHAPE))
    write_json(task/'final_native.json', dict(path='final_native.npz', sha256=sha(task/'final_native.npz'), role='formal_native_outer_commit', certificate=dict(status='certified_feasible', min_jd_lower_bound=1e-4, n_infeasible=0, n_uncertain=0, state_hash=hashlib.sha256(np.ascontiguousarray(final).view(np.uint8)).hexdigest())))
    write_json(task/'summary.json', dict(status='native_budget_exhausted', success=False, metrics_role='formal_native_outer_commit', native_commits=len(elapsed), e2e_seconds=327., GF=1., GF_threshold=1e-5, solve_call_wall_seconds=326., final_export_seconds=.4, stored_array_io_seconds=.4,
                                         backend_timings=dict(gradient=40., poisson=20.), counts=dict(inner_trials=600, outer_accepted=len(elapsed)), evaluation_counts=dict(gpu=300), backend_timing_scope='inclusive nested timers; do NOT sum as total wall', costs=dict(levels=[dict(level=3, rejected_reasons=dict(JACOBIAN_VIOLATION=200))])))
    write_json(run/'completion.json', dict(seconds=327.))
    (run/'inputs'/pair).mkdir(parents=True, exist_ok=True)
    return task


class World:
    """base/results: A5P's inputs, the entries, the originals of the 15 cells.  base/stage: the truth.  ev: the real evaluate.py pointed at this stage."""

    def __init__(self, base, monkeypatch):
        self.base = Path(base)
        self.results = self.base/'results'
        self.stage = self.base/'stage'
        self.ev = SEED.load_evaluate()
        monkeypatch.setattr(self.ev, 'STAGE', self.stage)
        inputs = self.results/SEED.ARMS['S_P']['inputs']
        files = {}
        for index, pair in enumerate(PAIRS):
            write_pair(self.stage/pair, pair, index)
            d = inputs/'stage_L2R_val_v1'/pair
            write_pair(d, pair, index, dict(preprocessing_d0=dict(arm='A5P', oracle=False, source_pair_sha256='orig'+pair, source_manifest_sha256='origm'+pair, recipe='synthetic')))
            files[f'stage_L2R_val_v1/{pair}/pair.npz'] = sha(d/'pair.npz')
            files[f'stage_L2R_val_v1/{pair}/manifest.json'] = sha(d/'manifest.json')
            make_seed_entry(self.results, 'S_P', pair, seed=index)
            make_s19_entry(self.results, 'S_K1', pair, seed=100+index)
        write_json(inputs/'d0_inputs.json', dict(arm='A5P', oracle=False, files_sha256=files))
        write_json(self.stage/'selection.json', dict(pairs=[dict(pair=p) for p in PAIRS]))       # C23.6-2 revision 6h: d0/evaluate admits only the stage's registered pairs
        for arm, pair in TC.plan_runs():
            self.make_original(arm, pair)
        self.root = self.results/'tc_v1'

    def protocol(self, arm, pair):
        d = self.results/SEED.ARMS['S_P']['inputs']/'stage_L2R_val_v1'/pair
        return dict(task=dict(name='pg_weak__four__330s', pair=pair), stage=TC.ARMS[arm]['stage'], input_sha256={'pair.npz': sha(d/'pair.npz'), 'manifest.json': sha(d/'manifest.json')})

    def make_original(self, arm, pair):
        run = TC.original_dir(arm, pair, self.results)
        run.mkdir(parents=True)
        write_json(run/'protocol.json', self.protocol(arm, pair))
        _, entry = TC.cell_files(arm, pair, self.results)
        if entry is not None:
            write_json(run/'seed_entry.json', entry)
            shutil.copyfile(SEED.HERE/'inject_run.py', run/'seed_inject.py')
        write_task(run, arm, pair)
        shutil.copyfile(self.stage/pair/'pair.npz', run/'inputs'/pair/'pair.npz')
        write_json(run/'metrics.json', self.ev.evaluate_run(run, arm, pair, f'orig_{arm}_{pair}', method='m'))
        (run/'cmd.txt').write_text('# made up\n')

    def options(self, **changes):
        return SimpleNamespace(**dict(dict(root=str(self.root), wait_minutes=30, min_host_gib=15.), **changes))


@pytest.fixture
def world(neutral, monkeypatch):
    return World(neutral, monkeypatch)


class TCMachine(FakeSystem):
    """A fake machine: `launch` writes what the frozen solver would write; evaluate and curve are the real steps, in this process."""

    def __init__(self, world):
        super().__init__()
        self.world = world
        self.diverge, self.fail_curve, self.bad_trace, self.exit_measure, self.infrastructure = {}, set(), set(), 0, set()

    def prepare(self, argv, target, log, results=None):
        run = Path(argv[argv.index('prepare')+1])
        arm, pair = run.parent.name, run.name
        run.mkdir(parents=True)
        assert argv[argv.index('--stage')+1] == TC.ARMS[arm]['stage'] and argv[argv.index('--pair')+1] == pair
        write_json(run/'protocol.json', self.world.protocol(arm, pair))
        (run/'inputs'/pair).mkdir(parents=True)
        shutil.copyfile(self.world.stage/pair/'pair.npz', run/'inputs'/pair/'pair.npz')
        return 0

    def check_prepared(self, run, pair, stage, pair_sha, manifest_sha, entry):
        assert entry['seconds'] == 330 and entry['changes'] == SEED.CHANGES and Path(entry['inputs_root']).name == 'A5P'
        return dict(ok=not self.fail_check, problems=['made up'] if self.fail_check else [])

    def launch(self, argv, log):
        unit = argv[2].split('=')[1]
        self.launched.append(unit)
        if unit == TC.MEASURE_UNIT:
            return self.exit_measure
        run = Path(argv[argv.index('--wait')+2].split('=')[1]).parent
        arm, pair = run.parent.name, run.name
        assert ('seed_inject.py' in argv[-1]) == TC.ARMS[arm]['injected'] and ('snapshot/scripts/run_stage_3d_e48_pg.py' in argv[-1]) != TC.ARMS[arm]['injected']
        if unit not in self.infrastructure:
            task = write_task(run, arm, pair, diverge_at=self.diverge.get((arm, pair)))
            if unit in self.bad_trace:
                levels = json_load(task/'levels.json')
                levels[0]['preoptimizer']['raw_hash'] = 'wrong'
                write_json(task/'levels.json', levels)
        (run/'exit_code').write_text('0\n')
        return 0

    def classify(self, run, t0):
        unit = (Path(run)/'service_unit.txt').read_text().strip()
        if unit in self.infrastructure:
            return {'class': 'infrastructure', 'kind': 'k', 'reason': 'r'}
        record = {'class': self.classes.get(unit, 'ok'), 'kind': 'k', 'reason': 'r'}
        write_json(Path(run)/'classification.json', record)
        return record

    def evaluate(self, run, arm, pair, run_id):
        return TC.evaluate_one(SimpleNamespace(run=str(run), arm=arm, pair=pair, run_id=run_id), ev=self.world.ev)

    def curve(self, run, arm, pair, run_id):
        if (arm, pair) in self.fail_curve:
            return 8
        return TC.curve_one(SimpleNamespace(run=str(run), arm=arm, pair=pair, run_id=run_id), ev=self.world.ev, results=self.world.results)

    def uptime(self):
        return '100.0'


def start_queue(world, machine, **changes):
    return TC.queue(world.options(**changes), system=machine, results=world.results, repo=ROOT, ev=world.ev)


# ---- the rule ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_times_are_the_briefs():
    assert CURVE.TIMES == (0, 15, 30, 45, 60, 90, 120, 150, 210, 270) and TC.MAX_FIELDS == 11


@pytest.mark.parametrize('t,position', [(0, 0), (15, 2), (14.9, 1), (30, 3), (45, 4), (60, 4), (90, 7), (89.9, 6), (120, 7), (150, 8), (210, 9), (270, 11), (1000, 12), (7.9, None)])
def test_the_commit_of_a_time(t, position):
    """Not later than t: exactly t counts; t = 0 is the first commit; before the first commit there is none."""
    assert CURVE.select_commit(list(ELAPSED), t) == position


def test_no_commit_at_all():
    assert CURVE.select_commit([], 0) is None and CURVE.select_commit([], 90) is None


def make_records(task, commits, rows):
    task.mkdir(parents=True, exist_ok=True)
    write_json(task/'commits.json', commits)
    write_json(task/'history.json', rows)


def commit_row(t, **changes):
    return dict(dict(path='c', sha256='s', level=3, native=True, role='formal_outer_commit', solver_commit_seconds=t, visible_seconds=t+50., certificate=dict(status='certified_feasible', min_jd_lower_bound=1e-4, n_infeasible=0, n_uncertain=0, state_hash='h')), **changes)


def history_row(t, **changes):
    return dict(dict(level=3, outer_accepted=True, elapsed_seconds=t, D_candidate=1., F_candidate=1., GF_candidate=1.), **changes)


def test_the_clock_is_the_native_rows_only(neutral):
    """Other levels and rejected rows are left out; the visible seconds are recorded beside, not used."""
    make_records(neutral/'t', [commit_row(1., level=2, native=False), commit_row(5.), commit_row(9.)], [history_row(1., level=2), history_row(5.), history_row(7., outer_accepted=False), history_row(9.)])
    commits = CURVE.native_commits(neutral/'t')
    assert [c['elapsed_seconds'] for c in commits] == [5., 9.] and [c['visible_seconds'] for c in commits] == [55., 59.] and [c['index'] for c in commits] == [1, 2]


@pytest.mark.parametrize('commits,rows,word', [
    ([commit_row(5.)], [history_row(5.), history_row(6.)], 'accepted'),
    ([commit_row(5.)], [history_row(5.5)], 'solver_commit_seconds'),
    ([commit_row(5.), commit_row(5.)], [history_row(5.), history_row(5.)], 'after'),
    ([commit_row(5., role='x')], [history_row(5.)], 'role'),
    ([commit_row(float('nan'))], [history_row(float('nan'))], 'elapsed'),
])
def test_records_that_do_not_agree_stop_the_curve(neutral, commits, rows, word):
    make_records(neutral/'t', commits, rows)
    with pytest.raises(CURVE.CurveError, match=word):
        CURVE.native_commits(neutral/'t')


@pytest.mark.parametrize('changes', [dict(status='uncertified'), dict(min_jd_lower_bound=0.), dict(min_jd_lower_bound=None), dict(n_infeasible=1), dict(n_uncertain=1)])
def test_a_certificate_does_not_pass_on_a_zero_or_an_undecided_cell(changes):
    good = dict(status='certified_feasible', min_jd_lower_bound=1e-4, n_infeasible=0, n_uncertain=0)
    assert CURVE.certificate_passed(good) and not CURVE.certificate_passed(dict(good, **changes)) and not CURVE.certificate_passed(None)


# ---- the comparison with the original --------------------------------------------------------------------------------------------------------------------------------------------------------
def seq(*names, D=100.):
    return [dict(sha256=n, state_hash='s'+n, elapsed_seconds=float(k), D=D+k, F=D+k, GF=1., min_jd_lower_bound=1e-4) for k, n in enumerate(names, 1)]


def test_identical_sequences():
    r = CURVE.compare_sequences(seq('a', 'b'), seq('a', 'b'))
    assert r['identical'] and r['common_prefix'] == 2 == r['n_new'] == r['n_old'] and r['first_difference'] is None and r['tail'] is None and not r['one_is_prefix_of_the_other']


def test_the_first_difference_says_where_and_how_much():
    new, old = seq('a', 'x', 'c'), seq('a', 'b', 'c', D=90.)
    r = CURVE.compare_sequences(new, old)
    assert not r['identical'] and r['common_prefix'] == 1 and r['first_difference']['commit'] == 2 and r['first_difference']['sha256_new'] == 'x' and r['first_difference']['sha256_old'] == 'b' and r['first_difference']['D_new_minus_old'] == 10.


@pytest.mark.parametrize('new,old,longer', [(('a', 'b', 'c'), ('a', 'b'), 'new'), (('a', 'b'), ('a', 'b', 'c', 'd'), 'old')])
def test_one_run_went_on_longer(new, old, longer):
    r = CURVE.compare_sequences(seq(*new), seq(*old))
    assert not r['identical'] and r['one_is_prefix_of_the_other'] and r['first_difference'] is None and r['common_prefix'] == 2 and r['tail']['longer'] == longer and r['tail']['n_extra'] == abs(len(new)-len(old))


def test_equal_fields_in_other_files_are_told_apart():
    new, old = seq('a'), seq('b')
    new[0]['state_hash'] = old[0]['state_hash'] = 'same'
    r = CURVE.compare_sequences(new, old)
    assert r['common_prefix'] == 0 and r['state_common_prefix'] == 1


# ---- the reading -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def fake_point(t, seg35, seg4=.9, **changes):
    metrics = dict(seg35=seg35, seg4=seg4, sdlogj_official=.07, sdlogj_brain=.6, small_six=.5, large_six=.8, other_23=.8, cortex=.8, white_matter=.9, seg4_per_label={'1': .8, '3': .9}, seg35_per_label={str(k): seg35 for k in range(1, 36)})
    return dict(dict(t=t, commit=1, elapsed_seconds=float(t) if t != CURVE.FINAL else 275., visible_seconds=50., certificate_passed=True, D=1., GF=1., min_jd_lower_bound=1e-4, metrics=metrics), **changes)


def fake_record(arm, at90=.84, final=.84, **metrics_changes):
    points = [fake_point(t, at90 if t >= 90 else at90-.01) for t in CURVE.TIMES]
    metrics = dict(dict(classification={'class': 'ok'}, certificate=dict(passed=True, state_hash_matches_u=True, final_npz_sha256_matches=True), seed=dict(passed=True, problems=[])), **metrics_changes)
    return dict(curve=dict(complete=True, self_check=dict(passed=True), points=points, final=fake_point(CURVE.FINAL, final, is_last_commit=True), n_native_commits=13, run_id='x'), metrics=metrics)


def reading_runs(delta=.003, at90=.84, arm='S_P'):
    return {(arm, pair): fake_record(arm, at90=at90, final=at90+(delta[k] if isinstance(delta, (list, tuple)) else delta)) for k, pair in enumerate(PAIRS)}


def read(runs):
    return DEC.decide(runs, list(PAIRS), TC.short_name)


@pytest.mark.parametrize('delta,code', [(-.001, DEC.PLATEAU), (0., DEC.PLATEAU), (.0015, DEC.PLATEAU), (.0025, DEC.MIDDLE), (.0045, DEC.MIDDLE), (.0055, DEC.RISING), (.02, DEC.RISING)])
def test_the_table_of_the_reading(delta, code):
    assert read(reading_runs(delta))['reading_code'] == code


@pytest.mark.parametrize('value,code', [(.002, DEC.PLATEAU), (.00200001, DEC.MIDDLE), (.00499999, DEC.MIDDLE), (.005, DEC.RISING)])
def test_the_bars_are_included(value, code):
    assert DEC.label_of(value) == code


def test_the_reading_is_the_mean_over_the_pairs_unrounded():
    deltas = [0., 0., 0., 0., .0101]                     # mean .00202, above .002 though 4 pairs are flat
    result = read(reading_runs(deltas))
    assert result['reading_code'] == DEC.MIDDLE and result['delta_seg35']['positive'] == 1 and result['delta_seg35']['mean'] == pytest.approx(.00202, abs=1e-12)


def premise_case(change):
    runs = reading_runs(.01)
    rec = runs[('S_P', PAIRS[2])]
    change(rec)
    return runs


def drop_point_90(rec):
    next(p for p in rec['curve']['points'] if p['t'] == 90).update(commit=None, metrics=None)


@pytest.mark.parametrize('change,word', [
    (lambda r: r.update(metrics=None), 'metrics'),
    (lambda r: r['metrics'].update(classification={'class': 'handover_only'}), '分类'),
    (lambda r: r['metrics']['certificate'].update(passed=False), '证书'),
    (lambda r: r['metrics']['certificate'].update(state_hash_matches_u=False), '证书'),
    (lambda r: r['metrics']['seed'].update(passed=False), '入口'),
    (lambda r: r.pop('curve'), 'curve'),
    (lambda r: r['curve'].update(complete=False), 'curve'),
    (lambda r: r['curve']['self_check'].update(passed=False), 'curve'),
    (drop_point_90, '没有提交'),
    (lambda r: next(p for p in r['curve']['points'] if p['t'] == 90).update(certificate_passed=False), '证书'),
    (lambda r: r['curve']['final'].update(elapsed_seconds=90.), '不晚于'),
    (lambda r: r['curve']['final'].update(certificate_passed=False), '证书'),
    (lambda r: r['curve']['final'].update(is_last_commit=False), '最后一次提交'),
])
def test_a_premise_not_met_is_not_decidable(change, word):
    result = read(premise_case(change))
    assert result['reading_code'] == DEC.NOT_DECIDABLE and word in result['reading'] and '0440-0441' in result['reading']


def test_a_missing_pair_is_not_decidable():
    runs = reading_runs(.01)
    del runs[('S_P', PAIRS[4])]
    result = read(runs)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and result['delta_seg35']['mean'] is None and result['delta_seg35']['n'] == 4


def test_only_s_p_decides():
    runs = reading_runs(.01)
    with_others = {**runs, **reading_runs(0., arm='S_K1'), **reading_runs(-.02, arm='B1TM')}
    for pair in PAIRS:                                   # the descriptive arms may even be broken
        with_others[('B1TM', pair)]['metrics']['classification'] = {'class': 'method_failure'}
    assert read(with_others)['reading_code'] == read(runs)['reading_code'] == DEC.RISING
    assert read(reading_runs(.01, arm='S_K1'))['reading_code'] == DEC.NOT_DECIDABLE


def test_the_times_30_and_60_are_described_with_their_own_t():
    """Review I1: the words of the reading speak of 90 s; a row of t = 30 must not say '90 s 之后还在涨' under a reading of '90 s 之后基本是平台'."""
    result = read(reading_runs(.001))                    # 30 and 60 lie .01 below the value at 90
    described = result['described_times']
    assert set(described) == {'30', '60'} and result['reading_code'] == DEC.PLATEAU
    assert described['30']['delta']['mean'] == pytest.approx(.011) and described['30']['delta_seg4']['mean'] == pytest.approx(0.)
    assert described['30']['words'] == '30 s 之后还在涨' and described['60']['words'] == '60 s 之后还在涨'


@pytest.mark.parametrize('delta,words', [(-.009, '30 s 之后基本是平台'), (-.007, '中间带')])
def test_the_other_words_of_a_described_time(delta, words):
    assert read(reading_runs(delta))['described_times']['30']['words'] == words


def test_the_reading_says_seg4_too_and_which_clock():
    """Review m5, I3: the text that goes into queue.log."""
    text = read(reading_runs(.01))['reading']
    assert 'Δ(seg35)' in text and 'Δ(seg4)' in text and '不进判定' in text and '原生层时刻' in text


def test_a_value_missing_with_every_premise_met_is_not_decidable():
    """Review m1: no TypeError on a mean that is None."""
    runs = reading_runs(.01)
    runs[('S_P', PAIRS[1])]['curve']['final']['metrics'].pop('seg35')
    result = read(runs)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and '不可算' in result['reading'] and result['delta_seg35']['n'] == 4


# ---- the three clocks (plan revision 1, review I3) -------------------------------------------------------------------------------------------------------------------------------------------
def test_the_real_total_time_of_each_arm():
    assert DEC.real_total_seconds('S_P', 140.) == 140.-47.+60.+86. == 239. and DEC.real_total_seconds('S_K1', 140.) == 140.-47.+38.+86. and DEC.real_total_seconds('B1TM', 95.) == 95.+86. and DEC.real_total_seconds('S_P', None) is None


def test_the_tables_and_the_summary_say_the_three_clocks(world, neutral):
    root = neutral/'clock_tree'/'tc_v1'
    runs = {**reading_runs(.01), **reading_runs(0., arm='S_K1'), **reading_runs(0., arm='B1TM')}
    for rec in runs.values():
        for p in rec['curve']['points']:
            p['visible_seconds'] = p['t']+50.
        rec['curve']['final']['visible_seconds'] = 325.
    for pair in PAIRS:                                   # B1TM's task clock runs about 45 s behind
        rec = runs[('B1TM', pair)]
        for p in rec['curve']['points']:
            p['visible_seconds'] = p['t']+5.
    write_fake_batch(root, runs)
    result = TC.write_analysis(root, ev=world.ev)
    row = next(r for r in result['tables']['S_P'] if r['t'] == 90)
    assert row['visible_seconds'] == 140. and row['real_total_seconds'] == 239.
    diff = next(r for r in result['differences']['S_P(t) − B1TM(t)'] if r['t'] == 90)
    assert diff['task_clock_a'] == 140. and diff['task_clock_b'] == 95. and diff['real_total_a'] == 239. and diff['real_total_b'] == 181.
    text = (root/'analysis/summary_tc.md').read_text()
    assert '三种时钟' in text and '任务时钟约 140 s' in text and '真实总时间约 239 s' in text and '| 90 | 140.0 | 239 |' in text and '140 / 95' in text and '239 / 181' in text and set(result['clocks']) == {'native', 'task', 'real'}
    note = DEC.NOTE
    assert '任务时钟相同的是 final 行' in note and '不是同一时刻' not in note and '不能读成' in note


def test_the_anomalies_look_at_the_final_map_too():
    """Review m11."""
    runs = reading_runs(.01)
    runs[('S_P', PAIRS[0])]['curve']['final']['metrics']['sdlogj_official'] = .2
    rows = DEC.anomalies(runs, [('S_P', p) for p in PAIRS], TC.short_name)
    assert [r['pair'] for r in rows] == ['0438-0439'] and '终态官方 SDlogJ 0.2000' in rows[0]['what']


def different_values(seg35):
    rec = fake_record('x')
    for point in rec['curve']['points']+[rec['curve']['final']]:
        point['metrics'].update(seg35=seg35, seg4=seg35+.05, cortex=seg35, seg4_per_label={'1': seg35, '3': .9}, seg35_per_label={str(k): seg35 for k in range(1, 36)})
    return rec


def test_the_two_difference_tables(world):
    runs = {}
    for k, pair in enumerate(PAIRS):
        runs[('S_P', pair)] = different_values(.85)
        runs[('S_K1', pair)] = different_values(.84 if k else .86)                  # 4 of 5 positive
        runs[('B1TM', pair)] = different_values(.83)
    table = DEC.difference_table(runs, 'S_P', 'S_K1', list(PAIRS), world.ev)
    assert [row['t'] for row in table] == list(CURVE.TIMES)+[CURVE.FINAL]
    for row in table:
        for key in ('seg35', 'seg4', 'cortex', 'small_six'):
            assert row[key]['mean'] == pytest.approx((4*.01-.01)/5) and row[key]['positive'] == 4 and row[key]['n'] == 5
    table = DEC.difference_table(runs, 'S_P', 'B1TM', list(PAIRS), world.ev)
    assert all(row['seg35']['mean'] == pytest.approx(.02) and row['small_six']['positive'] == 5 for row in table)


def write_fake_batch(root, runs):
    for (arm, pair), rec in runs.items():
        d = root/'runs'/arm/pair
        d.mkdir(parents=True)
        for name in ('curve', 'metrics'):
            if rec.get(name) is not None:
                write_json(d/f'{name}.json', rec[name])


def test_the_written_analysis_reports_seg35_and_seg4_and_keeps_the_earlier_version(world, neutral):
    root = neutral/'analysis_tree'/'tc_v1'
    runs = {**reading_runs(.01), **reading_runs(0., arm='S_K1'), **reading_runs(0., arm='B1TM')}
    write_fake_batch(root, runs)
    result = TC.write_analysis(root, ev=world.ev)
    text = (root/'analysis/summary_tc.md').read_text()
    assert result['reading_code'] == DEC.RISING and 'seg35' in text and 'seg4' in text and '皮层' in text and '小六个' in text and 'S_P(t) − S_K1(t)' in text and 'S_P(t) − B1TM(t)' in text
    assert set(result['tables']) == {'S_P', 'S_K1', 'B1TM'} and set(result['differences']) == {'S_P(t) − S_K1(t)', 'S_P(t) − B1TM(t)'}
    decision = json_load(root/'analysis/decision_tc.json')
    assert decision['reading_code'] == DEC.RISING and decision['reading_arm'] == 'S_P' and (root/'analysis/long_curve.csv').read_text().count('\n') > 100
    again = TC.write_analysis(root, ev=world.ev)
    assert again['previous_summary_moved_to'] is not None and (Path(again['previous_summary_moved_to'])/'decision_tc.json').is_file()


# ---- the gate --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Gate(FakeSystem):
    pass


@pytest.mark.parametrize('change,word', [
    (dict(unit_list=['tag-mc0p-queue.service']), 'other units'),
    (dict(unit_list=['seq-vp-chain.service']), 'other units'),
    (dict(load=1.5), 'load'),
    (dict(gib={'/home': 99., '/mnt/c': 500.}), '/home'),
    (dict(gib={'/home': 500., '/mnt/c': 14.9}), '/mnt/c'),
    (dict(gpu='compute pids [1]'), 'GPU busy'),
    (dict(mib=15*1024-1), 'GPU free'),
])
def test_each_reason_closes_the_gate(neutral, change, word):
    machine = Gate()
    assert TC.busy_reason(machine, neutral, 15.) is None
    for key, value in change.items():
        setattr(machine, key, value)
    assert word in TC.busy_reason(machine, neutral, 15.)


def test_the_gate_is_open_at_the_floors_and_for_its_own_queue(neutral):
    machine = Gate()
    machine.load, machine.gib, machine.mib, machine.unit_list = 1.49, {'/home': 100., '/mnt/c': 15.}, 15*1024, [f'{TC.QUEUE_UNIT}.service']
    assert TC.busy_reason(machine, neutral, 15.) is None


def test_the_hold_file_closes_the_gate(neutral):
    (neutral/'hold').write_text('')
    assert TC.busy_reason(Gate(), neutral, 15.) == 'hold file'


def test_a_closed_gate_stops_after_its_time(neutral):
    machine = Gate()
    machine.load = 5.
    lines = []
    with pytest.raises(TC.GateClosed, match='gate closed for 30 minutes'):
        TC.wait_for_gate(machine, neutral, 30, lines.append, 15.)
    assert machine.clock >= 30*60 and all('load' in line for line in lines)


def test_the_queue_stops_when_the_gate_stays_closed_and_nothing_runs(world):
    machine = TCMachine(world)
    machine.load = 5.
    assert start_queue(world, machine) == 8
    assert 'gate closed' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == [] and not (world.root/'runs').exists()


# ---- only the first 5 pairs, the arms of the plan ----------------------------------------------------------------------------------------------------------------------------------------------
def test_the_plan_is_15_runs_s_p_first():
    plan = TC.plan_runs()
    assert len(plan) == 15 and [a for a, _ in plan] == ['S_P']*5+['S_K1']*5+['B1TM']*5 and all(p in PAIRS for _, p in plan) and len(set(plan)) == 15
    assert PAIRS == tuple(f'OASIS_OAS1_{n:04d}_MR1__OASIS_OAS1_{n+1:04d}_MR1' for n in range(438, 443))


@pytest.mark.parametrize('arm,pair', [('S_P', 'OASIS_OAS1_0443_MR1__OASIS_OAS1_0444_MR1'), ('S_K1', 'OASIS_OAS1_0456_MR1__OASIS_OAS1_0457_MR1'), ('B1TM', 'OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1'), ('S_T', PAIRS[0]), ('A5P', PAIRS[0])])
def test_other_cells_are_refused_before_any_file(arm, pair, neutral):
    for call in (lambda: TC.require_cell(arm, pair), lambda: TC.run_dir(neutral, arm, pair), lambda: TC.cell_files(arm, pair, neutral), lambda: TC.original_dir(arm, pair, neutral), lambda: TC.prepare_argv(arm, pair, neutral), lambda: TC.unit_argv(arm, pair, neutral, neutral), lambda: TC.unit_name(arm, pair)):
        with pytest.raises(SystemExit):
            call()


@pytest.mark.parametrize('step', ['evaluate-one', 'curve-one'])
def test_the_steps_refuse_another_pair(step, neutral):
    with pytest.raises(SystemExit, match='first 5'):
        TC.main([step, str(neutral/'x'), '--arm', 'S_P', '--pair', 'OASIS_OAS1_0443_MR1__OASIS_OAS1_0444_MR1', '--run-id', 'x'])


def test_the_subjects_are_on_the_development_allowlist_and_off_the_training_and_holdout_lists():
    sys.path.insert(0, str(ROOT))
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')
    held = set((ROOT/'splits/bayesreg_v1/holdout_B_subjects.txt').read_text().split())
    train = set((ROOT/'splits/bayesreg_v1/train_subjects.txt').read_text().split())
    for pair in PAIRS:
        a, b = pair.split('__')
        check_pair_subjects(a, b, allowlist)
        assert not {a, b} & held and not {a, b} & train


# ---- the start strings -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_start_strings_hold_no_forbidden_word_and_the_right_names():
    root = Path('/x/tc_v1')
    for argv in [TC.queue_launch(root), TC.measure_argv(root)]+[TC.unit_argv(a, p, root/'runs'/a/p, root) for a, p in TC.plan_runs()]+[TC.prepare_argv(a, p, root/'runs'/a/p) for a, p in TC.plan_runs()]:
        assert 'pytest' not in ' '.join(argv) and 'seq-vp' not in ' '.join(argv)
    assert TC.queue_launch(root)[2] == '--unit=tag-tc-queue' and TC.measure_argv(root)[2] == '--unit=tag-tc-measure'
    assert [TC.unit_argv(a, p, root/'r', root)[2] for a, p in TC.plan_runs()][:1] == ['--unit=tag-tc-S_P-0438-0439'] and TC.unit_name('S_K1', PAIRS[4]) == 'tag-tc-S_K1-0442-0443' and TC.unit_name('B1TM', PAIRS[1]) == 'tag-tc-B1TM-0439-0440'
    assert TC.new_run_id('S_P', PAIRS[0]).startswith('tc_S_P_0438-0439_')
    assert TC.default_root().name == 'tc_v1' and TC.default_root().parent == Path('/home/rmcfz/seq_vp_tag_results')
    with pytest.raises(ValueError, match='forbidden'):
        SEED.check_launch(['x', '/tmp/pytest-of-me/y'])


def test_the_units_are_the_originals_but_for_the_name():
    """The same resources, watchdog and environment as the entries' own strings: S_P, S_K1 with the wrapper; B1TM with the frozen entry."""
    root, run = Path('/x/tc_v1'), Path('/x/tc_v1/runs/A/P')
    a = TC.unit_argv('S_P', PAIRS[0], run, root)
    b = SEED.unit_argv('S_P', PAIRS[0], run, root, unit='tag-tc-S_P-0438-0439')
    assert a == b
    assert TC.unit_argv('S_K1', PAIRS[0], run, root) == S19.core.unit_argv('S_K1', PAIRS[0], run, root, unit='tag-tc-S_K1-0438-0439')
    c = TC.unit_argv('B1TM', PAIRS[0], run, root)
    assert c[:-1] == [t if not t.startswith('--unit=') else '--unit=tag-tc-B1TM-0438-0439' for t in b[:-1]]
    assert f"'{run}/snapshot/scripts/run_stage_3d_e48_pg.py' run '{run}'" in c[-1] and 'seed_inject' not in c[-1] and f'{SEED.WATCHDOG}s' in c[-1]
    assert any(t == '--property=MemoryMax=12G' for t in c) and any(t == '--property=CPUQuota=1200%' for t in c)


def test_the_prepare_lines_are_the_originals_with_their_stages():
    run = Path('/x/r')
    assert TC.prepare_argv('S_P', PAIRS[0], run) == SEED.prepare_argv('S_P', PAIRS[0], run) and TC.prepare_argv('S_K1', PAIRS[0], run) == S19.prepare_argv('S_K1', PAIRS[0], run)
    argv = TC.prepare_argv('B1TM', PAIRS[0], run)
    assert argv[argv.index('--stage')+1] == 'B1-B1TM' and argv[argv.index('--geometry-weight-levels')+1] == '1,1,0.1,0.01' and argv[argv.index('--seconds')+1] == '330'


def write_original_cmd(world, arm, pair, tokens_change=None):
    d = TC.original_dir(arm, pair, world.results)
    prepare = [str(t) for t in TC.prepare_argv(arm, pair, d, world.results)]
    unit = [str(t).replace(str(world.root), str(d.parents[2])) for t in TC.unit_argv(arm, pair, d, world.root)]
    old_unit = 'tag-old-unit'
    unit[2] = f'--unit={old_unit}'
    if tokens_change:
        tokens_change(prepare)
    (d/'cmd.txt').write_text(f'# prepare\n{shlex.join(prepare)}\n# run\n{shlex.join(unit)}\n')
    (d/'service_unit.txt').write_text(old_unit+'\n')


def test_the_command_lines_are_compared_with_the_originals(world):
    write_original_cmd(world, 'B1TM', PAIRS[0])
    assert TC.command_difference('B1TM', PAIRS[0], world.results) == {}

    def other(tokens):
        tokens[tokens.index('--seconds')+1] = '90'
    write_original_cmd(world, 'B1TM', PAIRS[1], other)
    assert set(TC.command_difference('B1TM', PAIRS[1], world.results)) == {'prepare'}


# ---- nothing is written to an existing result directory --------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('name', ['seed_v1', 'seed_v1b', 'b1_v1', 'mc0p_v1', 'tc_v2', 'tc_v1/runs'])
def test_the_only_root_is_tc_v1(name, world):
    assert TC.root_problem(world.results/name, world.results) is not None and TC.root_problem(world.root, world.results) is None


@pytest.mark.parametrize('name', ['seed_v1', 'seed_v1b', 'b1_v1'])
@pytest.mark.parametrize('command', ['queue', 'decide', 'ledger'])
def test_the_commands_refuse_another_root(name, command, world, capsys):
    before = tree(world.results)
    assert TC.main([command, '--root', str(world.results/name)]) == 2
    assert 'writes only to' in capsys.readouterr().err and tree(world.results) == before


@pytest.mark.parametrize('step', ['evaluate-one', 'curve-one'])
@pytest.mark.parametrize('arm', ['S_P', 'S_K1', 'B1TM'])
def test_the_steps_refuse_an_original_run(step, arm, world, capsys):
    before = tree(world.results)
    assert TC.main([step, str(TC.original_dir(arm, PAIRS[0], world.results)), '--arm', arm, '--pair', PAIRS[0], '--run-id', 'x']) == 2
    assert 'writes only to' in capsys.readouterr().err and tree(world.results) == before


def test_the_cleanup_refuses_an_original_even_with_a_complete_curve(world):
    d = TC.original_dir('S_P', PAIRS[0], world.results)
    write_json(d/'curve.json', dict(complete=True, self_check=dict(passed=True), run_dir=str(d)))
    before = tree(world.results)
    for runs_root in (d.parents[1], world.root/'runs'):
        with pytest.raises(CURVE.CurveError, match='nothing deleted'):
            CURVE.cleanup_after_curve(d, runs_root)
    assert tree(world.results) == before


def own_run(world, arm='S_P', pair=PAIRS[0]):
    run = TC.run_dir(world.root, arm, pair)
    run.mkdir(parents=True)
    write_json(run/'protocol.json', world.protocol(arm, pair))
    _, entry = TC.cell_files(arm, pair, world.results)
    if entry is not None:
        write_json(run/'seed_entry.json', entry)
    task = write_task(run, arm, pair)
    shutil.copyfile(world.stage/pair/'pair.npz', run/'inputs'/pair/'pair.npz')
    return run, task


@pytest.mark.parametrize('curve_doc', [None, dict(complete=False, self_check=dict(passed=True)), dict(complete=True, self_check=dict(passed=False)), dict(complete=True), dict(complete=True, self_check=dict(passed=True)), dict(complete=True, self_check=dict(passed=True), run_dir='/elsewhere/run')])
def test_the_cleanup_waits_for_a_complete_curve_of_this_run(curve_doc, world):
    run, task = own_run(world)
    if curve_doc is not None:
        write_json(run/'curve.json', curve_doc)
    before = tree(run)
    with pytest.raises(CURVE.CurveError, match='commits stay'):
        CURVE.cleanup_after_curve(run, world.root/'runs')
    assert tree(run) == before


def test_the_cleanup_compares_the_run_id_of_the_curve(world):
    run, _ = own_run(world)
    (run/'run_id').write_text('tc_this\n')
    write_json(run/'curve.json', dict(complete=True, self_check=dict(passed=True), run_dir=str(run), run_id='tc_other'))
    before = tree(run)
    with pytest.raises(CURVE.CurveError, match='run_id tc_other'):
        CURVE.cleanup_after_curve(run, world.root/'runs')
    assert tree(run) == before


def test_the_cleanup_refuses_a_task_directory_that_is_a_symbolic_link(world):
    """Review I2, the first counter-example: <run>/<task> points at the task directory of an original."""
    run, task = own_run(world)
    shutil.rmtree(task)
    original = TC.original_dir('S_P', PAIRS[0], world.results)/'pg_weak__four__330s'
    task.symlink_to(original)
    write_json(run/'curve.json', dict(complete=True, self_check=dict(passed=True), run_dir=str(run)))
    before = tree(world.results/'seed_v1')
    with pytest.raises(CURVE.CurveError, match='not a child of this run'):
        CURVE.cleanup_after_curve(run, world.root/'runs')
    assert tree(world.results/'seed_v1') == before and len(list(original.glob('committed_*.npz'))) == len(ELAPSED)


def test_the_cleanup_refuses_a_task_name_that_leaves_the_run(world):
    """Review I2, the second counter-example: `..` in protocol.json leads to the task directory of an original."""
    run, _ = own_run(world)
    protocol = json_load(run/'protocol.json')
    protocol['task']['name'] = f'../../../../seed_v1/runs/S_P/{PAIRS[0]}/pg_weak__four__330s'
    write_json(run/'protocol.json', protocol)
    assert CURVE.task_dir(run).resolve() == TC.original_dir('S_P', PAIRS[0], world.results).resolve()/'pg_weak__four__330s'     # the name really leads there
    write_json(run/'curve.json', dict(complete=True, self_check=dict(passed=True), run_dir=str(run)))
    before = tree(world.results/'seed_v1')
    with pytest.raises(CURVE.CurveError, match='not a child of this run'):
        CURVE.cleanup_after_curve(run, world.root/'runs')
    assert tree(world.results/'seed_v1') == before


def test_the_cleanup_takes_only_the_intermediate_maps_that_are_recorded(world):
    run, task = own_run(world)
    write_json(run/'curve.json', dict(complete=True, self_check=dict(passed=True), run_dir=str(run)))
    (task/'other.npz').write_bytes(b'x')
    (task/'committed_9999_L3.npz').write_bytes(b'not in commits.json')
    names = {p.name for p in task.iterdir()}
    n = CURVE.cleanup_after_curve(run, world.root/'runs')
    left = {p.name for p in task.iterdir()}
    assert n == len(ELAPSED)+2 and left == {x for x in names if not x.startswith(('committed_', 'level_'))}|{'committed_9999_L3.npz'} and 'final_native.npz' in left and 'other.npz' in left and 'commits.json' in left
    record = json_load(run/'cleanup.json')
    assert record['deleted_bytes'] > 0 and len(record['deleted']) == n and all(row['sha256'] for row in record['deleted']) and {row['name'] for row in record['kept_npz']} == {'final_native.npz', 'other.npz', 'committed_9999_L3.npz'}
    assert [row['name'] for row in record['not_deleted_unrecorded']] == ['committed_9999_L3.npz']


# ---- the curve of one run, real functions on synthetic fields ---------------------------------------------------------------------------------------------------------------------------------
def evaluate_own(world, run, arm='S_P', pair=PAIRS[0]):
    write_json(run/'tc_arm.json', dict(arm=arm))
    write_json(run/'classification.json', {'class': 'ok', 'kind': 'k', 'reason': 'r'})
    write_json(run/'seed_arm.json', dict(arm=arm))
    assert TC.evaluate_one(SimpleNamespace(run=str(run), arm=arm, pair=pair, run_id='r'), ev=world.ev) == 0


def run_curve(world, run, arm='S_P', pair=PAIRS[0]):
    return TC.curve_one(SimpleNamespace(run=str(run), arm=arm, pair=pair, run_id='r'), ev=world.ev, results=world.results)


def test_the_curve_takes_the_commits_of_the_rule_and_agrees_with_the_evaluation(world):
    run, _ = own_run(world)
    evaluate_own(world, run)
    assert run_curve(world, run) == 0
    c = json_load(run/'curve.json')
    want = {0: 1, 15: 3, 30: 4, 45: 5, 60: 5, 90: 8, 120: 8, 150: 9, 210: 10, 270: 12}
    assert {p['t']: p['commit'] for p in c['points']} == want and c['final']['commit'] == 13 and c['final']['is_last_commit'] is True
    assert c['n_fields_evaluated'] == len(set(want.values()))+1 == 9 and c['complete'] and c['self_check']['passed']
    by = {p['t']: p for p in c['points']}
    assert by[45]['metrics'] == by[60]['metrics'] and by[90]['elapsed_seconds'] == 90. and by[90]['visible_seconds'] == 140. and by[90]['age_seconds'] == 0. and by[0]['age_seconds'] is None
    assert set(by[90]['metrics']) >= {'seg35', 'seg4', 'sdlogj_official', 'sdlogj_brain', 'small_six', 'large_six', 'other_23', 'cortex', 'white_matter'}
    assert by[90]['metrics']['cortex'] == by[90]['metrics']['seg4_per_label']['1'] and by[0]['metrics']['seg35'] != by[270]['metrics']['seg35']
    assert c['clock'] == CURVE.CLOCK and c['final']['metrics']['seg35'] == json_load(run/'metrics.json')['metrics']['seg35']


def test_the_groups_are_means_of_the_per_label_dice(world):
    labs, manifest, _ = world.ev.truth_labels(PAIRS[0])
    metrics = CURVE.evaluate_field(world.ev, commit_field(0, 3), labs, tuple(manifest['spacing_xyz']))
    per = metrics['seg35_per_label']
    for name, labels in world.ev.A5_GROUPS.items():
        values = [per[k] for k in labels if per.get(k) is not None]
        assert metrics[name] == pytest.approx(float(np.mean(values)))
    assert metrics['seg35'] == pytest.approx(float(np.mean([v for v in per.values() if v is not None])))


@pytest.mark.parametrize('damage,word', [
    (lambda task: (task/'committed_0001_L3.npz').unlink(), 'no such file'),
    (lambda task: np.savez(task/'committed_0008_L3.npz', u=np.zeros(SHAPE+(3,))), 'sha256'),
    (lambda task: write_json(task/'commits.json', [dict(c, certificate=dict(c['certificate'], status='uncertified')) for c in json_load(task/'commits.json')]), 'did not pass'),
    (lambda task: write_json(task/'commits.json', [dict(c, certificate=dict(c['certificate'], state_hash='x')) for c in json_load(task/'commits.json')]), 'state hash'),
    (lambda task: write_json(task/'history.json', json_load(task/'history.json')[1:]), 'native commits'),
    (lambda task: write_json(task/'final_native.json', dict(json_load(task/'final_native.json'), certificate=dict(status='certified_feasible', min_jd_lower_bound=1e-4, n_infeasible=0, n_uncertain=0, state_hash='x'))), 'state hash'),
])
def test_a_curve_that_cannot_be_trusted_stops_and_writes_nothing(damage, word, world, capsys):
    run, task = own_run(world)
    evaluate_own(world, run)
    damage(task)
    assert run_curve(world, run) == 8
    assert word in capsys.readouterr().out and not (run/'curve.json').exists() and not (run/'reproduction.json').exists()
    assert len(list(task.glob('committed_*.npz'))) >= len(ELAPSED)-1


def test_the_curve_agrees_with_metrics_json_or_stops(world, capsys):
    run, _ = own_run(world)
    evaluate_own(world, run)
    doc = json_load(run/'metrics.json')
    doc['metrics']['seg35'] += 1e-9
    write_json(run/'metrics.json', doc)
    assert run_curve(world, run) == 8 and 'differs from metrics.json' in capsys.readouterr().out and not (run/'curve.json').exists()


def test_the_reproduction_record(world):
    run, _ = own_run(world)
    evaluate_own(world, run)
    before = tree(TC.original_dir('S_P', PAIRS[0], world.results))
    assert run_curve(world, run) == 0
    r = json_load(run/'reproduction.json')
    assert r['sequence']['identical'] and r['curve_applies_to_original'] and r['start']['entry_same'] and r['start']['initializer_same'] and r['final']['same_file'] and r['final']['seg35_new_minus_old'] == 0.
    assert tree(TC.original_dir('S_P', PAIRS[0], world.results)) == before
    write_task(run, 'S_P', PAIRS[0], diverge_at=5)
    evaluate_own(world, run)
    (run/'curve.json').unlink()
    assert run_curve(world, run) == 0
    r = json_load(run/'reproduction.json')
    assert not r['sequence']['identical'] and r['sequence']['common_prefix'] == 4 and r['sequence']['first_difference']['commit'] == 5 and not r['final']['same_file'] and r['final']['field_difference']['max_abs_mm'] > 0 and r['curve_applies_to_original_up_to_commit'] == 4
    assert r['commit_at_time']['30'] == dict(commit_new=4, commit_old=4, elapsed_new=29., elapsed_old=29., same_field=True)
    assert r['commit_at_time']['90']['commit_new'] == r['commit_at_time']['90']['commit_old'] == 8 and r['commit_at_time']['90']['same_field'] is False       # the same position, past the common prefix


def test_the_same_sequence_at_another_pace_takes_another_commit_at_90(world):
    """Review m7: the sequence alone does not say the two runs have the same commit at the same t."""
    run, _ = own_run(world)
    slower = tuple(t*1.2 for t in ELAPSED)
    write_task(run, 'S_P', PAIRS[0], elapsed=slower)
    evaluate_own(world, run)
    assert run_curve(world, run) == 0
    r = json_load(run/'reproduction.json')
    assert r['sequence']['identical'] and r['curve_applies_to_original']
    assert r['commit_at_time']['90']['commit_old'] == 8 and r['commit_at_time']['90']['commit_new'] == 6 and r['commit_at_time']['90']['same_field'] is False and r['commit_at_time']['90']['elapsed_new'] == pytest.approx(61.*1.2)
    assert DEC.reproduction_rows({('S_P', PAIRS[0]): dict(reproduction=r)}, [('S_P', PAIRS[0])], TC.short_name)[0]['commit_at_90_new'] == 6


@pytest.mark.parametrize('arm', ['S_P', 'S_K1', 'B1TM'])
def test_each_arm_has_its_original(arm, world):
    assert TC.original_dir(arm, PAIRS[0], world.results).parts[-4:-2] == ({'S_P': 'seed_v1', 'S_K1': 'seed_v1b', 'B1TM': 'b1_v1'}[arm], 'runs')


# ---- the entries and the originals -----------------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_cells_take_their_batches_entries(world):
    files, entry = TC.cell_files('S_P', PAIRS[0], world.results)
    assert entry['source_arm'] == 'Q2' and entry['source_k'] == 35 and entry['oracle'] is False and entry['charge_seconds'] == 47. and files[3]['pair'] == PAIRS[0]
    _, entry = TC.cell_files('S_K1', PAIRS[0], world.results)
    assert entry['source_arm'] == 'Q0' and entry['source_k'] == 1 and '/mc0p_v1/runs/Q0_long/' in entry['field_path']
    assert TC.cell_files('B1TM', PAIRS[0], world.results)[1] is None
    _, entry = TC.cell_files('S_P', PAIRS[3], world.results)
    assert '/mc0p_v1b/runs/Q2_long/' in entry['field_path']


@pytest.mark.parametrize('change,word', [
    (lambda d: (d/'metrics.json').unlink(), 'metrics.json'),
    (lambda d: (d/'pg_weak__four__330s/commits.json').unlink(), 'commits.json'),
    (lambda d: write_json(d/'protocol.json', dict(json_load(d/'protocol.json'), input_sha256={})), 'inputs differ'),
    (lambda d: write_json(d/'protocol.json', dict(json_load(d/'protocol.json'), stage='other')), 'stage'),
    (lambda d: write_json(d/'seed_entry.json', dict(json_load(d/'seed_entry.json'), u_sha256='x')), 'u_sha256'),
    (lambda d: write_json(d/'seed_entry.json', dict(json_load(d/'seed_entry.json'), charge_seconds=0.)), 'charge_seconds'),
    (lambda d: (d/'seed_inject.py').write_text('other'), 'wrapper'),
])
def test_an_original_that_is_not_this_cell_stops_the_start(change, word, world):
    change(TC.original_dir('S_K1', PAIRS[2], world.results))
    machine = TCMachine(world)
    assert start_queue(world, machine) == 8
    reason = json_load(world.root/'queue_stop.json')['reason']
    assert word in reason and 'start checks' in reason and machine.launched == []


def test_a_missing_entry_stops_the_start(world):
    (S19.entry_dir('S_K1', PAIRS[1], world.results)/'field.npz').unlink()
    machine = TCMachine(world)
    assert start_queue(world, machine) == 8 and machine.launched == [] and 'start checks' in json_load(world.root/'queue_stop.json')['reason']


# ---- the queue on a fake machine, with the real steps ----------------------------------------------------------------------------------------------------------------------------------------
def test_the_whole_queue(world, capsys):
    machine = TCMachine(world)
    machine.diverge[('B1TM', PAIRS[1])] = 7
    originals = {arm: tree(world.results/spec['original']) for arm, spec in TC.ARMS.items()}
    inputs = tree(world.results/SEED.ARMS['S_P']['inputs'])
    assert start_queue(world, machine) == 0
    assert machine.launched == [TC.unit_name(a, p) for a, p in TC.plan_runs()]+[TC.MEASURE_UNIT]
    assert machine.sleeps.count(SEED.BETWEEN_RUNS_SECONDS) == 15          # 14 between the runs, 1 before the measurement
    assert {arm: tree(world.results/spec['original']) for arm, spec in TC.ARMS.items()} == originals and tree(world.results/SEED.ARMS['S_P']['inputs']) == inputs
    for arm, pair in TC.plan_runs():
        run = TC.run_dir(world.root, arm, pair)
        task = run/'pg_weak__four__330s'
        assert not list(task.glob('committed_*.npz')) and not list(task.glob('level_*.npz')) and (task/'final_native.npz').is_file() and (task/'commits.json').is_file() and (task/'history.json').is_file()
        for name in ('curve.json', 'reproduction.json', 'cleanup.json', 'metrics.json', 'queue_done.json', 'cmd.txt', 'git.txt', 'env.txt', 'data.txt', 'seed', 'tc_arm.json', 'run_id', 'prepared_check.json'):
            assert (run/name).is_file(), name
        assert (run/'run_id').read_text().startswith(f'tc_{arm}_') and json_load(run/'queue_done.json')['state'] == 'ok' and (run/'seed_inject.py').exists() == TC.ARMS[arm]['injected']
        assert json_load(run/'metrics.json')['method'] == TC.ARMS[arm]['method'] and json_load(run/'curve.json')['complete']
        identical = json_load(run/'reproduction.json')['sequence']['identical']
        assert identical == ((arm, pair) != ('B1TM', PAIRS[1]))
    decision = json_load(world.root/'analysis/decision_tc.json')
    assert decision['reading_code'] != DEC.NOT_DECIDABLE and decision['delta_seg35']['n'] == 5
    text = (world.root/'analysis/summary_tc.md').read_text()
    assert '第 7 次提交' in text and 'S_K1' in text
    assert any(r['arm'] == 'B1TM' and not r['identical'] and r['first_difference']['commit'] == 7 for r in decision['reproduction'])
    batch = json_load(world.root/'batch.json')
    assert len(batch['plan']) == 15 and batch['originals'] == dict(S_P='seed_v1', S_K1='seed_v1b', B1TM='b1_v1')


def test_a_second_start_runs_nothing_again(world):
    machine = TCMachine(world)
    assert start_queue(world, machine) == 0
    first = tree(world.root/'runs')
    (world.root/measure_tc_result()).parent.mkdir(parents=True, exist_ok=True)
    write_json(world.root/measure_tc_result(), dict(done=True))
    again = TCMachine(world)
    assert start_queue(world, again) == 0
    assert again.launched == [] and tree(world.root/'runs') == first and len(json_load(world.root/'batch.json')['starts']) == 2


def measure_tc_result():
    return Path(MEASURE.OUT_NAME)/MEASURE.RESULT_FILE


def test_a_curve_failure_stops_the_queue_and_keeps_the_commits(world):
    machine = TCMachine(world)
    machine.fail_curve.add(('S_P', PAIRS[1]))
    assert start_queue(world, machine) == 8
    assert machine.launched == [TC.unit_name('S_P', PAIRS[0]), TC.unit_name('S_P', PAIRS[1])] and 'curve failed' in json_load(world.root/'queue_stop.json')['reason']
    task = TC.run_dir(world.root, 'S_P', PAIRS[1])/'pg_weak__four__330s'
    assert len(list(task.glob('committed_*.npz'))) == len(ELAPSED) and not (TC.run_dir(world.root, 'S_P', PAIRS[1])/'cleanup.json').exists()
    assert json_load(world.root/'analysis/decision_tc.json')['reading_code'] == DEC.NOT_DECIDABLE      # what finished is summed up


def test_an_entry_not_injected_as_recorded_stops_the_queue_and_keeps_the_commits(world):
    machine = TCMachine(world)
    machine.bad_trace.add(TC.unit_name('S_K1', PAIRS[0]))
    assert start_queue(world, machine) == 8
    assert 'not injected as recorded' in json_load(world.root/'queue_stop.json')['reason'] and TC.MEASURE_UNIT not in machine.launched
    assert len(list((TC.run_dir(world.root, 'S_K1', PAIRS[0])/'pg_weak__four__330s').glob('committed_*.npz'))) == len(ELAPSED)


def test_a_method_failure_is_kept_whole_and_the_queue_goes_on(world):
    machine = TCMachine(world)
    machine.classes[TC.unit_name('B1TM', PAIRS[0])] = 'method_failure'
    assert start_queue(world, machine) == 0
    run = TC.run_dir(world.root, 'B1TM', PAIRS[0])
    assert len(list((run/'pg_weak__four__330s').glob('committed_*.npz'))) == len(ELAPSED) and not (run/'curve.json').exists() and not (run/'cleanup.json').exists() and json_load(run/'queue_done.json')['state'] == 'method_failure'
    assert len(machine.launched) == 16


def test_a_method_failure_of_s_p_makes_the_reading_not_decidable(world):
    machine = TCMachine(world)
    machine.classes[TC.unit_name('S_P', PAIRS[0])] = 'method_failure'
    assert start_queue(world, machine) == 0
    assert json_load(world.root/'analysis/decision_tc.json')['reading_code'] == DEC.NOT_DECIDABLE


def test_three_infrastructure_failures_in_a_row_stop_the_queue(world):
    machine = TCMachine(world)
    machine.infrastructure = {TC.unit_name('S_P', p) for p in PAIRS[:3]}
    assert start_queue(world, machine) == 8
    assert len(machine.launched) == 3 and 'infrastructure' in json_load(world.root/'queue_stop.json')['reason']


def test_an_incomplete_run_is_never_rerun(world):
    TC.run_dir(world.root, 'S_P', PAIRS[0]).mkdir(parents=True)
    machine = TCMachine(world)
    assert start_queue(world, machine) == 8 and machine.launched == [] and 'incomplete attempt' in json_load(world.root/'queue_stop.json')['reason']


def test_a_run_that_ended_is_post_processed_on_a_new_start_not_rerun(world):
    machine = TCMachine(world)
    machine.fail_curve.add(('S_P', PAIRS[0]))
    assert start_queue(world, machine) == 8 and machine.launched == [TC.unit_name('S_P', PAIRS[0])]
    again = TCMachine(world)
    assert start_queue(world, again) == 0
    assert TC.unit_name('S_P', PAIRS[0]) not in again.launched and len(again.launched) == 15
    assert (TC.run_dir(world.root, 'S_P', PAIRS[0])/'curve.json').is_file() and (TC.run_dir(world.root, 'S_P', PAIRS[0])/'cleanup.json').is_file()


def test_a_start_after_the_cleanup_does_not_ask_for_the_commits_again(world):
    machine = TCMachine(world)
    assert start_queue(world, machine) == 0
    run = TC.run_dir(world.root, 'S_P', PAIRS[0])
    (run/'queue_done.json').unlink()                     # the driver died between the cleanup and this file
    (world.root/measure_tc_result()).parent.mkdir(parents=True, exist_ok=True)
    write_json(world.root/measure_tc_result(), dict(done=True))
    curve_before = sha(run/'curve.json')
    again = TCMachine(world)
    assert start_queue(world, again) == 0
    assert again.launched == [] and json_load(run/'queue_done.json')['state'] == 'ok' and sha(run/'curve.json') == curve_before


def test_the_command_line_of_the_two_steps(world, monkeypatch):
    """The parser and the guards on the way the queue calls them (a subprocess there; here in this process, with this world's stage and root)."""
    monkeypatch.setattr(SEED, 'RESULTS', world.results)
    monkeypatch.setattr(TC, 'load_evaluate', lambda: world.ev)
    run, task = own_run(world)
    write_json(run/'tc_arm.json', dict(arm='S_P'))
    write_json(run/'classification.json', {'class': 'ok', 'kind': 'k', 'reason': 'r'})
    write_json(run/'seed_arm.json', dict(arm='S_P'))
    assert TC.main(['evaluate-one', str(run), '--arm', 'S_P', '--pair', PAIRS[0], '--run-id', 'tc_x']) == 0 and json_load(run/'metrics.json')['run_id'] == 'tc_x'
    assert TC.main(['curve-one', str(run), '--arm', 'S_P', '--pair', PAIRS[0], '--run-id', 'tc_x']) == 0 and json_load(run/'curve.json')['run_id'] == 'tc_x' and json_load(run/'reproduction.json')['sequence']['identical']
    assert len(list(task.glob('committed_*.npz'))) == len(ELAPSED)                 # curve-one deletes nothing


def test_a_measurement_that_fails_stops_after_the_reading_and_is_not_rerun(world):
    machine = TCMachine(world)
    machine.exit_measure = 8
    assert start_queue(world, machine) == 8
    assert machine.launched[-1] == TC.MEASURE_UNIT and 'the reading stands' in json_load(world.root/'queue_stop.json')['reason'] and (world.root/'analysis/decision_tc.json').is_file()


def test_the_prepared_check_stops_the_queue(world):
    machine = TCMachine(world)
    machine.fail_check = True
    assert start_queue(world, machine) == 8 and machine.launched == [] and 'differs from the configuration' in json_load(world.root/'queue_stop.json')['reason']


def test_another_driver_on_the_root_is_refused(world):
    import fcntl
    world.root.mkdir(parents=True)
    with open(world.root/'queue.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert start_queue(world, TCMachine(world)) == 2


def test_a_batch_of_another_definition_stops(world):
    machine = TCMachine(world)
    world.root.mkdir(parents=True)
    write_json(world.root/'batch.json', dict(schema='bayesreg.tc.batch.v1', seconds=90))
    assert start_queue(world, machine) == 8 and 'batch.json differs' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == []


def test_a_deletion_that_fails_stops_the_queue_with_its_record(world, monkeypatch):
    """Review m4: an OSError of the cleanup is a STOP with queue_stop.json, not a driver that dies."""
    def refuses(run, runs_root):
        raise OSError('made up')
    monkeypatch.setattr(CURVE, 'cleanup_after_curve', refuses)
    machine = TCMachine(world)
    assert start_queue(world, machine) == 8
    reason = json_load(world.root/'queue_stop.json')['reason']
    assert 'cleanup' in reason and 'made up' in reason and len(machine.launched) == 1
    assert len(list((TC.run_dir(world.root, 'S_P', PAIRS[0])/'pg_weak__four__330s').glob('committed_*.npz'))) == len(ELAPSED)


@pytest.mark.parametrize('floor', [14.9, 0., -1., float('nan')])
def test_the_floor_of_the_disk_cannot_be_lowered(floor, world, capsys):
    """Review m10, r4: nan too, `free < nan` would open the gate."""
    machine = TCMachine(world)
    assert start_queue(world, machine, min_host_gib=floor) == 2
    assert 'at least 15' in capsys.readouterr().err and machine.launched == [] and not world.root.exists()


def test_a_file_without_a_record_is_seen_in_the_log_and_the_anomalies(world):
    """Review r3: it stays, and someone reads about it."""
    class Leaves(TCMachine):
        def launch(self, argv, log):
            code = super().launch(argv, log)
            if argv[2].split('=')[1] == TC.unit_name('S_P', PAIRS[0]):
                (TC.run_dir(world.root, 'S_P', PAIRS[0])/'pg_weak__four__330s/committed_0014_L3.npz.partial.npz').write_bytes(b'x')
            return code
    machine = Leaves(world)
    assert start_queue(world, machine) == 0
    task = TC.run_dir(world.root, 'S_P', PAIRS[0])/'pg_weak__four__330s'
    assert [p.name for p in task.glob('committed_*.npz')] == ['committed_0014_L3.npz.partial.npz']
    assert 'committed_0014_L3.npz.partial.npz' in (world.root/'queue.log').read_text()
    decision = json_load(world.root/'analysis/decision_tc.json')
    assert any(row['arm'] == 'S_P' and row['pair'] == '0438-0439' and 'committed_0014_L3.npz.partial.npz' in row['what'] for row in decision['anomalies'])
    assert 'committed_0014_L3.npz.partial.npz' in (world.root/'analysis/summary_tc.md').read_text()


def test_a_higher_floor_reaches_the_gate_of_the_queue(world):
    machine = TCMachine(world)
    machine.gib = {'/home': 500., '/mnt/c': 16.}
    assert start_queue(world, machine, min_host_gib=20.) == 8
    assert '/mnt/c free < 20' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == []


def test_the_real_steps_call_this_script_with_their_arguments(neutral, monkeypatch):
    """The real System of this batch (the class itself, no machine behind it): evaluate and curve start run_tc.py with the step's name, in the run's log, nothing else; the inputs check reads this batch's root."""
    seen = []
    real = REAL_SYSTEMS[0]()
    with monkeypatch.context() as patch:                 # only this block: the guard of no_real_machine stays
        patch.setattr(TC.subprocess, 'run', lambda argv, **kwargs: seen.append(argv) or SimpleNamespace(returncode=5))
        assert real.evaluate(neutral, 'S_P', PAIRS[0], 'r') == 5 and real.curve(neutral, 'B1TM', PAIRS[1], 'q') == 5
    assert seen[0][:2] == [SEED.PY, '-B'] and seen[0][2].endswith('experiments/tc/run_tc.py') and seen[0][3:] == ['evaluate-one', str(neutral), '--arm', 'S_P', '--pair', PAIRS[0], '--run-id', 'r']
    assert seen[1][3:] == ['curve-one', str(neutral), '--arm', 'B1TM', '--pair', PAIRS[1], '--run-id', 'q'] and (neutral/'log.txt').exists()
    assert any('no d0_inputs.json' in problem for problem in real.inputs_problems('B1TM', ROOT, neutral))


# ---- the time ledger -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_ledger_reads_and_sums_and_writes_only_where_told(world, neutral):
    before = tree(world.results)
    sources = [('seed_v1', TC.original_dir('S_P', pair, world.results)) for pair in PAIRS[:2]]
    ledger = CURVE.write_ledger(sources, neutral/'out')
    assert tree(world.results) == before and ledger['n_runs'] == 2 and ledger['statistics']['backend_timings']['gradient'] == dict(mean=40., min=40., max=40., n=2) and ledger['statistics']['counts']['inner_trials']['mean'] == 600.
    assert ledger['statistics']['rejected_reasons']['JACOBIAN_VIOLATION']['mean'] == 200. and ledger['native_seconds_per_accepted_step']['mean'] == pytest.approx(275./(5*len(ELAPSED)))
    text = (neutral/'out/time_ledger_sp19.md').read_text()
    assert '不能相加' in text and 'gradient' in text and json_load(neutral/'out/time_ledger_sp19.json')['n_runs'] == 2


def test_the_ledger_has_the_19_runs_of_s_p(neutral):
    sources = TC.ledger_sources(neutral)
    assert len(sources) == 19 and [s[0] for s in sources] == ['seed_v1']*5+['seed_v1b']*14 and [Path(s[1]).name for s in sources] == list(PAIRS)+list(S19.REST_PAIRS) and all(Path(s[1]).parent.name == 'S_P' for s in sources)


# ---- the measurement -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('path', ['/x/seed_v1/measure', '/x/mc0p_v1/runs/Q2_long/0438-0439', '/x/seg/unet35_v1/pred', '/x/tc_v1', '/x/tc_v1/runs/measure', '/x/tc_v1/measure', '/home/rmcfz/seq_vp_tag_results/seed_v1/measure'])
def test_the_measurement_refuses_every_other_directory(path, capsys):
    """Review m8: the resolved path, not only the names: /x/tc_v1/measure too."""
    assert MEASURE.out_problem(path) is not None and MEASURE.main(['--out', path, '--device', 'cpu']) == 2 and 'writes only to' in capsys.readouterr().err and not Path(path).exists()


def test_the_measurement_takes_its_directory_only(neutral, monkeypatch):
    """On a made-up results directory: the real one holds measure_0438.json once the unit has run (review R1)."""
    monkeypatch.setattr(MEASURE, 'RESULTS', neutral)
    assert MEASURE.out_problem(neutral/'tc_v1/measure') is None and MEASURE.out_problem(neutral/'tc_v1/../tc_v1/measure') is None
    assert MEASURE.out_problem(neutral/'other/tc_v1/measure') is not None


def test_the_measurement_is_not_run_twice(neutral, capsys, monkeypatch):
    monkeypatch.setattr(MEASURE, 'RESULTS', neutral)
    out = neutral/'tc_v1/measure'
    assert MEASURE.out_problem(out) is None
    out.mkdir(parents=True)
    (out/MEASURE.RESULT_FILE).write_text('{}')
    assert MEASURE.main(['--out', str(out), '--device', 'cpu']) == 2 and 'not run twice' in capsys.readouterr().err


def test_the_subject_and_the_pair_of_the_measurement_are_the_briefs():
    assert MEASURE.SUBJECT == 'OASIS_OAS1_0438_MR1' and MEASURE.PAIR == PAIRS[0]


class FakeMemory:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.append('reset')

    def peak_gib(self):
        self.calls.append('peak')
        return 7.

    def release(self):
        self.calls.append('release')


def test_the_device_cache_goes_after_a_part_that_failed_and_the_next_part_runs(neutral):
    """Review m9."""
    def fails():
        raise RuntimeError('made up out of memory')
    memory = FakeMemory()
    record = dict(failures=[], first=None, second=None)
    MEASURE.run_parts(record, neutral, (('first', fails), ('second', lambda: dict(status='done'))), memory=memory, started=0.)
    assert memory.calls == ['reset', 'release', 'reset', 'peak', 'release']
    assert record['first'] is None and record['second'] == dict(status='done', device_peak_allocated_gib_of_part=7.) and record['failures'][0]['part'] == 'first' and 'made up out of memory' in record['failures'][0]['traceback']
    assert json_load(neutral/MEASURE.RESULT_FILE)['second']['status'] == 'done'


def test_a_release_that_fails_loses_neither_the_record_nor_the_next_part(neutral):
    """Review r5."""
    class Refuses(FakeMemory):
        def release(self):
            super().release()
            if self.calls.count('release') == 1:
                raise RuntimeError('made up device error')
    memory = Refuses()
    record = dict(failures=[], first=None, second=None)
    MEASURE.run_parts(record, neutral, (('first', lambda: dict(a=1)), ('second', lambda: dict(b=2))), memory=memory, started=0.)
    assert record['first']['a'] == 1 and record['second']['b'] == 2 and memory.calls.count('release') == 2
    assert [(row['part'], row['step']) for row in record['failures']] == [('first', 'release')] and 'made up device error' in record['failures'][0]['traceback']
    assert json_load(neutral/MEASURE.RESULT_FILE)['failures'][0]['part'] == 'first'


def test_the_parts_on_the_cpu_ask_nothing_of_a_device(neutral):
    record = dict(failures=[], first=None)
    MEASURE.run_parts(record, neutral, (('first', lambda: dict(a=1)),), memory=None, started=0.)
    assert record['first'] == dict(a=1) and record['failures'] == []


def test_the_timers_put_everything_back_and_pass_exceptions():
    module = SimpleNamespace(f=lambda x: x+1, g=lambda: 1/0)
    original_f, original_g = module.f, module.g
    sink, ticks, synced = {}, iter(range(100)), []
    with MEASURE.timed(module, ('f', 'g'), sink, clock=lambda: float(next(ticks)), after=lambda: synced.append(1)):
        assert module.f(1) == 2 and module.f(2) == 3
        with pytest.raises(ZeroDivisionError):
            module.g()
    assert module.f is original_f and module.g is original_g and sink == dict(f=[1., 1.], g=[1.]) and len(synced) == 3


def test_the_summary_of_the_segmenter_steps():
    sink = dict(load_member=[1., 1.], normalize_image=[.5], predict_logits=[2., 2.], ensemble_probabilities=[10.], group_probabilities=[1.], _save_npz=[3., 2.], file_sha256=[.5, .5], read_image=[1.], check_subject_admitted=[.1], select_fold_models=[.1], require_inference_grouping=[.1], find_member_files=[.2], write_json=[.1])
    s = MEASURE.segmenter_summary(sink, 20.)
    assert s['calibration_seconds'] == pytest.approx(3.5) and s['forward_seconds'] == [2., 2.] and s['forward_seconds_total'] == 4. and s['write_seconds_total'] == 5. and s['hash_all_member_files_seconds'] == .2 and s['other_seconds'] == pytest.approx(20.-(.3+.2+1.+10.+1.+5.+1.+.1))


def test_the_timed_names_exist_on_the_real_modules():
    """A name that no longer exists would raise at the start of the unit, after the 15 runs."""
    from bayesreg.seg.unet import infer
    assert all(hasattr(infer, name) for name in MEASURE.SEGMENTER_STEPS)
    run = MEASURE.load_mc0p()
    mi = run.core.make_inputs()
    assert all(hasattr(mi, name) for name in MEASURE.MAKE_INPUTS_STEPS) and all(hasattr(run.mc, name) for name in MEASURE.MC_STEPS) and all(hasattr(run.core, name) for name in MEASURE.CORE_STEPS)


def test_the_segmenter_part_on_a_synthetic_world(neutral):
    """The real segment_subject under the timers, on the CPU, on the synthetic subjects of the U-Net's own dry run: the same files as segment_subject without timers, in its own directory, removed afterwards."""
    from bayesreg.io.guard import load_dev_allowlist
    from bayesreg.io.pair import load_segmenter_registry
    from bayesreg.seg.unet.cli import DEFAULT_GROUPS, dry_run
    from bayesreg.seg.unet.data import OasisFiles
    from bayesreg.seg.unet.groups import load_grouping
    from bayesreg.seg.unet.synthetic import oasis
    work = neutral/'unet'
    summary = dry_run(work=work, device='cpu', shape=(24, 24, 24), config=dict(max_iters=2, eval_every=2, warmup_iters=1, min_iters=2, log_every=1), log=lambda *_: None)
    assert summary['ok']
    subject = oasis(907)
    recorded = work/'models/pred'/subject/'provenance.json'
    before = tree(work/'models/pred')
    out = neutral/'tc_v1/measure'
    out.mkdir(parents=True)
    part = MEASURE.segmenter_part(out, 'cpu', subject=subject, files=OasisFiles(work/'data'), registry=load_segmenter_registry(work/'segmenters.json'), allowlist=load_dev_allowlist(work/'dev_subjects.json'), models_root=work/'models', grouping=load_grouping(DEFAULT_GROUPS), recorded=recorded)
    assert part['n_members'] == 4 == len(part['forward_seconds']) == len(part['member_load_seconds']) and len(part['write_seconds']) == 2 and part['calibration_seconds'] > 0 and part['read_image_seconds'] > 0 and part['policy'] == 'all_folds_mean'
    assert part['outputs_equal_recorded'] is True and part['file_bytes']['probs36.npz'] > 0 and not (out/'pred_tmp').exists() and tree(work/'models/pred') == before


ENTRY_PART = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import measure_tc as M
run = M.load_mc0p()
core, mc = run.core, run.mc
core.frozen_api()
world = json.loads(sys.argv[2])
core.STAGE = Path(world['stage'])
core.SETUP_WORKERS = 1
core.TIERS.update(long=dict(iterations=(2, 2, 2, 4)))
mc.SEG_ROOT, mc.A5P_INPUTS, mc.MC0_ROOT = Path(world['seg_root']), Path(world['a5p']), Path(world['mc0_root'])
mi = core.make_inputs()
mc.default_unet = lambda: mi.UNetSource(pred_root=Path(world['pred']))
out = Path(world['out'])
out.mkdir(parents=True)
real = (mi.colour_from_table, core.optimize_k)
part = M.entry_part(out, 'cpu', recorded=None)
assert (mi.colour_from_table, core.optimize_k) == real
print(json.dumps(part))
"""


@needs_frozen
def test_the_entry_part_on_a_synthetic_world(neutral, monkeypatch):
    """run_mc0p.run_one, Q2 long, under the timers, on the CPU, on MC0P's synthetic world: every step timed, u hashed, nothing left in the scratch directory."""
    world = pipeline_world(neutral, monkeypatch)
    world['out'] = str(neutral/'tc_v1/measure')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{FROZEN}:{ROOT}')
    done = subprocess.run([PY, '-B', '-c', ENTRY_PART, str(ROOT/'experiments/tc'), json.dumps(world)], env=env, cwd=neutral, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:]+done.stderr[-2000:]
    part = json.loads(done.stdout.splitlines()[-1])
    assert part['status'] == 'done' and part['k'] == 35 and len(part['u_sha256']) == 64 and part['u_equal_recorded'] is None and part['seconds_channels'] > 0
    steps = part['steps_seconds']
    assert set(steps) >= {'read_images', 'read_predicted', 'a4_admission', 'pair_structure_layer', 'unet_posterior', 'atropos_wm', 'alignment_qc', 'q0_channels', 'prepare_stacks', 'channels', 'data_grad_norm', 'p0_level_norms', 'coarse_pyramid', 'optimize_k', 'load_table', 'equals_a5_input'}
    assert len(steps['unet_posterior']) == 2 and len(steps['optimize_k']) == 1 and [row['iterations'] for row in part['levels']] == [2, 2, 2, 4]
    assert not (Path(world['out'])/'q2_tmp').exists()
