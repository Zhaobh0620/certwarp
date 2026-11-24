"""S_Y (runs/20261105-sy/任务书.md, plan.md): experiments/sy/; CPU only, small synthetic volumes and made-up records; no solver, no GPU, no unit.

- the source: the pinned sha256, fd2_v1's records, the channels of unet35_v1_ens; the band, both branches; the hold, a constant;
- the reading: every branch, its bars, its premises, the certificate computed from the field;
- the gate, closed behind a training; the plan's cells; the start strings; no truth key on the registration path;
- the queue on a fake machine, with the real evaluation, curve, cleanup and reading on synthetic fields: nothing written outside sy_v1; a run that does not end ok stops it;
- the dry run: no torch, no truth, nothing written outside its scratch directory.
"""
import ast
import hashlib
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
sys.path.insert(0, str(ROOT/'experiments/sy'))
sys.path.insert(1, str(ROOT/'experiments/s2'))
sys.path.insert(2, str(ROOT/'experiments/seed'))
sys.path.insert(3, str(ROOT/'experiments/tc'))

import decide_sy as DEC  # noqa: E402
import inject_run as INJ  # noqa: E402
import run_s2 as S2  # noqa: E402
import run_seed as SEED  # noqa: E402
import run_seed19 as S19  # noqa: E402
import run_sy as RUN  # noqa: E402
import s2_core as S2CORE  # noqa: E402
import sy_core as CORE  # noqa: E402
import tc_curve as CURVE  # noqa: E402

from .test_mc0p_tools import PY  # noqa: E402
from .test_s2_tools import ELAPSED, GROUPS, SHAPE, band_node, fake_curve, fake_metrics, interior_nodes, json_load, moved_values, raw_field, repaired_of, sha, task_records, tree, write_pair, write_task  # noqa: E402
from .test_seed_tools import FakeSystem, predicted_sources, write_json  # noqa: E402

PAIRS = CORE.PAIRS
TASK = 'pg_weak__four__330s'
REAL_RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
HX24_MEANS, S_X_MEANS, S_P_MEANS = (.8617, .9043), (.8526, .9011), (.8430, .8984)
needs_fd2 = pytest.mark.skipif(not (REAL_RESULTS/'fd2_v1/runs/RHx24_c600').is_dir(), reason='the records of FD2 are not on this machine')


class NoRealMachine:
    def __init__(self, *args, **kwargs):
        raise AssertionError('a queue on the real machine inside a test: a guard has regressed')


@pytest.fixture(autouse=True)
def no_real_machine(monkeypatch):
    """Every queue here runs on a fake machine, passed in.  A guard that regresses must fail the test, never start a real queue."""
    for module in (RUN, S2, SEED, S19):
        monkeypatch.setattr(module, 'System', NoRealMachine)


@pytest.fixture
def neutral():
    """A temporary directory whose path does not hold the word of the test runner: check_launch would refuse the start strings under tmp_path, as it should."""
    with tempfile.TemporaryDirectory(prefix='sy_tools_') as d:
        yield Path(d)


def metrics_of(means, *, hd95=1.7, sdlogj=.07, small=.66):
    return dict(fake_metrics(means[0], means[1], sdlogj=sdlogj, small=small), hd95=hd95)


def final_field(seed):
    rng = np.random.default_rng(seed)
    u = np.zeros(SHAPE+(3,))
    u[1:-1, 1:-1, 1:-1] = rng.normal(size=(SHAPE[0]-2, SHAPE[1]-2, SHAPE[2]-2, 3))*.02
    return u


def attempts_row(k, reason):
    return dict(level=3, visible_seconds=50.+k*20., elapsed_seconds=float(k % 3), reason=reason, accepted=reason is None)


def write_final(task, seed):
    """The exported map of a finished run, with the solver's record."""
    u = final_field(seed)
    np.savez(task/'final_native.npz', u=u, warped=np.zeros(SHAPE))
    write_json(task/'final_native.json', dict(path='final_native.npz', sha256=sha(task/'final_native.npz'), role='formal_native_outer_commit', certificate=dict(status='certified_feasible', min_jd_lower_bound=1e-4, n_infeasible=0, n_uncertain=0, state_hash=CORE.array_sha(u))))
    write_json(task/'inner_attempts.json', [attempts_row(k, 'JACOBIAN_VIOLATION' if k % 2 else None) for k in range(12)])
    return u


class World:
    """base/results: A5P's inputs, fd2_v1's Hx24 and its repair, S_P's and S_X's records.  base/stage: the truth.  ev: the real evaluate.py pointed at this stage."""

    def __init__(self, base, monkeypatch):
        self.base = Path(base)
        self.results = self.base/'results'
        self.stage = self.base/'stage'
        self.root = self.results/CORE.ROOT_NAME
        self.ev = SEED.load_evaluate()
        monkeypatch.setattr(self.ev, 'STAGE', self.stage)
        self.pins = {}
        monkeypatch.setattr(CORE, 'PINS', self.pins)
        inputs = self.results/SEED.ARMS['S_P']['inputs']
        files = {}
        for index, pair in enumerate(PAIRS):
            write_pair(self.stage/pair, pair, index)
            d = inputs/'stage_L2R_val_v1'/pair
            write_pair(d, pair, index, dict(preprocessing_d0=dict(arm='A5P', oracle=False, source_pair_sha256='orig'+pair, source_manifest_sha256='origm'+pair, recipe='synthetic')))
            files[f'stage_L2R_val_v1/{pair}/pair.npz'] = sha(d/'pair.npz')
            files[f'stage_L2R_val_v1/{pair}/manifest.json'] = sha(d/'manifest.json')
        write_json(inputs/'d0_inputs.json', dict(arm='A5P', oracle=False, files_sha256=files))
        write_json(self.stage/'selection.json', dict(pairs=[dict(pair=p) for p in PAIRS]))       # C23.6-2 revision 6h: d0/evaluate admits only the stage's registered pairs
        self.raw, self.nodes, self.values = {}, {}, {}
        for pair in PAIRS:
            self.make_source(pair)
            self.make_s_p(pair)
            self.make_s_x(pair)

    def make_source(self, pair, *, nodes=None, values=None, seconds_repair=121.3):
        """Hx24's field run and its repair in fd2_v1, as run_fl.py writes them, and their pins."""
        seed = PAIRS.index(pair)
        raw = raw_field(seed)
        nodes = interior_nodes(seed) if nodes is None else nodes
        values = moved_values(raw, nodes, seed) if values is None else values
        self.raw[pair], self.nodes[pair], self.values[pair] = raw, nodes, values
        short = CORE.short_name(pair)
        f, d = CORE.field_dir(pair, self.results), CORE.repair_dir(pair, self.results)
        f.mkdir(parents=True, exist_ok=True)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(f/CORE.FIELD_FILE, u=raw)
        a, b = pair.split('__')
        data = dict(pair_sha256='orig'+pair, manifest_sha256='origm'+pair, fixed_subject=a, moving_subject=b)
        write_json(f/'run.json', dict(schema=CORE.RUN_SCHEMA, run_id=f'fd2_Hx24_first5_{short}', arm='Hx24', tier='first5', pair=pair, k=35, channels='onehot', D_factor=24., native_iterations=600, oracle=False, status='done', channel_sources=predicted_sources(), data=data, checkpoints={'600': dict(step=600, file=CORE.FIELD_FILE, sha256=sha(f/CORE.FIELD_FILE), seconds_task=45.)}))
        np.savez(d/'repair.npz', nodes=nodes, values=values)
        tapered, repaired = CORE.array_sha(CORE.s2.tapered(raw)), CORE.array_sha(repaired_of(raw, nodes, values))
        write_json(d/'repair.json', dict(schema=CORE.REPAIR_SCHEMA, run_id=f'fd2_RHx24_c600_{short}', arm='RHx24', tier='c600', pair=pair, oracle=False, status='certified', passed=True, eps=.05, data=data, seconds_repair=seconds_repair, seconds_warmup=1.2, source=dict(directory=str(f), file=CORE.FIELD_FILE, field_sha256=sha(f/CORE.FIELD_FILE), tapered_u_sha256=tapered, seconds_task=45.),
                                          attempt=dict(file='repair.npz', sha256=sha(d/'repair.npz')), certificate=dict(status='certified_feasible', min_jd_lower_bound=1e-4, n_infeasible=0, n_uncertain=0, state_hash=repaired)))
        write_json(d/'repair_metrics.json', dict(schema=CORE.REPAIR_METRICS_SCHEMA, run_id=f'fd2_RHx24_c600_{short}', pair=pair, oracle=False, repaired_u_sha256=repaired, after=dict(metrics=metrics_of(HX24_MEANS), certificate=dict(accepted_alpha=1., status_at_1='certified_feasible', n_infeasible_at_1=0, n_uncertain_at_1=0)), evaluator=dict(script='experiments/d0/evaluate.py', sha256=sha(self.ev.__file__))))
        self.pin(pair, tapered_u=tapered, repaired_u=repaired)

    def pin(self, pair, **also):
        """What the table of sy_core would say about these files."""
        f, d = CORE.field_dir(pair, self.results), CORE.repair_dir(pair, self.results)
        short = CORE.short_name(pair)
        self.pins[short] = dict(self.pins.get(short, {}), raw_field=sha(f/CORE.FIELD_FILE), repair_npz=sha(d/'repair.npz'), run_json=sha(f/'run.json'), repair_json=sha(d/'repair.json'), repair_metrics_json=sha(d/'repair_metrics.json'), **also)

    def change(self, pair, which, change):
        """One record of fd2_v1 changed by `change(doc)`, and pinned as it now is: only source_problems can object."""
        d = dict(run=CORE.field_dir(pair, self.results)/'run.json', repair=CORE.repair_dir(pair, self.results)/'repair.json', metrics=CORE.repair_dir(pair, self.results)/'repair_metrics.json')[which]
        doc = json_load(d)
        change(doc)
        write_json(d, doc)
        self.pin(pair)

    def make_run_records(self, run, *, unit, metrics, entry, seed, alpha=1.):
        """A finished run of the frozen path, as seed_v1 or s2_v1 hold it."""
        task = Path(run)/TASK
        task.mkdir(parents=True)
        pair = Path(run).name
        write_json(Path(run)/'protocol.json', dict(task=dict(name=TASK, pair=pair), stage='SEED-S_P'))
        u = write_final(task, seed)
        recorded = json_load(task/'final_native.json')
        write_json(Path(run)/'metrics.json', dict(arm=Path(run).parent.name, pair=pair, classification={'class': 'ok'}, status='native_budget_exhausted', e2e_seconds=327., native_commits=13, GF=3., GF_threshold=1e-5, metrics=metrics, certificate=dict(status='certified_feasible', passed=True, state_hash=CORE.array_sha(u), state_hash_matches_u=True, final_npz_sha256=recorded['sha256'], final_npz_sha256_matches=True, min_jd_lower_bound=1e-4), seed=dict(accepted_alpha=alpha), self_check=dict(passed=True), run_labels_equal_truth=True, evaluator=dict(sha256=sha(self.ev.__file__))))
        write_json(Path(run)/'seed_entry.json', dict(entry, pair=pair, oracle=False, shape_zyx=list(SHAPE), source_k=35))
        write_json(Path(run)/'cleanup.json', dict(deleted_bytes=12*2**30))
        (Path(run)/'service_unit.txt').write_text(unit+'\n')
        task_records(task)
        write_json(task/'levels.json', [dict(event='level_prepared', transfer_recovery=dict(attempts=[dict(certificate=dict(grid_id='grid'))]))])
        return task

    def make_s_p(self, pair):
        run = CORE.reference_dir(pair, self.results)
        self.make_run_records(run, unit=f'tag-seed-S_P-{CORE.short_name(pair)}', metrics=metrics_of(S_P_MEANS, hd95=1.9, sdlogj=.08, small=.59), entry=dict(charge_seconds=47., field_path='x', field_sha256='x', source_seconds_total=101.), seed=PAIRS.index(pair)+60, alpha=.9 if pair == PAIRS[1] else 1.)
        (run/'cmd.txt').write_text(f"# prepare\n{shlex.join(SEED.prepare_argv('S_P', pair, run, self.results))}\n# run\n{shlex.join(SEED.unit_argv('S_P', pair, run, run.parents[2]))}\n")
        again = self.results/'tc_v1/runs/S_P'/pair
        again.mkdir(parents=True)
        write_json(again/'curve.json', fake_curve(.84, offset=50.))
        write_json(again/'protocol.json', dict(task=dict(name=TASK)))
        (again/TASK).mkdir()
        task_records(again/TASK)

    def make_s_x(self, pair):
        run = CORE.s_x_dir(pair, self.results)
        self.make_run_records(run, unit=f'tag-s2-S_X-{CORE.short_name(pair)}', metrics=metrics_of(S_X_MEANS, hd95=1.68, sdlogj=.075, small=.63), entry=dict(charge_seconds=152., source_seconds_total=140., seconds_repair=11., seconds_warmup=1.3), seed=PAIRS.index(pair)+80)
        doc = fake_curve(.85, offset=155.)
        doc['final']['elapsed_seconds'] = 172.                # S_X's native level ends early: its rows past 172 s are its final map
        write_json(run/'curve.json', doc)
        d = self.results/'fl_v1/runs/RHx_c2400'/CORE.short_name(pair)
        d.mkdir(parents=True)
        write_json(d/'repair_metrics.json', dict(after=dict(metrics=metrics_of((.8507, .8827), hd95=1.77, sdlogj=.056, small=.64))))

    def certifier(self, *, passes=True):
        def certify(seed, grid, config=None):
            assert config == 'configuration C' and seed.shape == SHAPE+(3,)
            return SimpleNamespace(status='certified_feasible' if passes else 'infeasible', reason='r', min_jd_lower_bound=1e-4 if passes else -.1, n_cells_checked=100, n_infeasible=0 if passes else 3, n_uncertain=0, state_hash=CORE.array_sha(seed), grid_id='grid', certificate_version='v')

        def frozen_taper(u, grid, width):
            return CORE.s2.tapered(u, width), None

        def make_grid(**kw):
            return SimpleNamespace(**kw)
        return certify, 'configuration C', frozen_taper, make_grid

    def options(self, **changes):
        return SimpleNamespace(**dict(dict(root=str(self.root), wait_minutes=30, min_host_gib=15.), **changes))

    def manifest(self, pair=PAIRS[0]):
        return CORE.input_files(pair, self.results)[3]


@pytest.fixture
def world(neutral, monkeypatch):
    return World(neutral, monkeypatch)


# ---- the fake machine -------------------------------------------------------------------------------------------------------------------------------------------------------------------
class SYMachine(FakeSystem):
    """A fake machine: `launch` writes what the frozen solver would write; evaluate, certify, curve and the taper check are the real steps, in this process."""

    def __init__(self, world, *, passes=True, final_passes=True):
        super().__init__()
        self.world, self.passes, self.final_passes = world, passes, final_passes
        self.bad_raw, self.bad_alpha, self.bad_state, self.fail_curve, self.fail_certify, self.infrastructure, self.empty = set(), set(), set(), set(), set(), set(), set()
        self.taper_checks = 0
        self.taper_certifier = None
        self.certified = []
        self.after_taper_check = None

    def prepare(self, argv, target, log, results=None):
        run = Path(argv[argv.index('prepare')+1])
        arm, pair = run.parent.name, run.name
        run.mkdir(parents=True)
        assert argv[argv.index('--stage')+1] == CORE.stage_name(arm) and argv[argv.index('--pair')+1] == pair and Path(target).name == 'stage_L2R_val_v1'
        write_json(run/'protocol.json', dict(task=dict(name=TASK, pair=pair), stage=CORE.stage_name(arm)))
        (run/'inputs'/pair).mkdir(parents=True)
        shutil.copyfile(self.world.stage/pair/'pair.npz', run/'inputs'/pair/'pair.npz')
        return 0

    def check_prepared(self, run, pair, stage, pair_sha, manifest_sha, entry):
        assert entry['seconds'] == 330 and entry['changes'] == SEED.CHANGES and Path(entry['inputs_root']).name == 'A5P' and stage == CORE.stage_name(CORE.ARM)
        return dict(ok=not self.fail_check, problems=['made up'] if self.fail_check else [])

    def launch(self, argv, log):
        unit = argv[2].split('=')[1]
        self.launched.append(unit)
        assert 'seed_inject.py' in argv[-1]
        run = Path(argv[argv.index('--wait')+2].split('=')[1]).parent
        if unit not in self.infrastructure:
            write_task(run, elapsed=() if unit in self.empty else ELAPSED, bad_raw=unit in self.bad_raw, alpha=.9 if unit in self.bad_alpha else 1., state_hash='another field' if unit in self.bad_state else None)
            write_json(run/TASK/'inner_attempts.json', [attempts_row(k, 'JACOBIAN_VIOLATION' if k % 3 == 0 else None) for k in range(12)])
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
        return RUN.evaluate_one(SimpleNamespace(run=str(run), arm=arm, pair=pair, run_id=run_id), ev=self.world.ev)

    def certify(self, run, arm, pair, run_id):
        self.certified.append(pair)
        if pair in self.fail_certify:
            return 8
        return RUN.certify_one(SimpleNamespace(run=str(run), arm=arm, pair=pair, run_id=run_id), results=self.world.results, certifier=self.world.certifier(passes=self.final_passes))

    def curve(self, run, arm, pair, run_id):
        if pair in self.fail_curve:
            return 8
        return RUN.curve_one(SimpleNamespace(run=str(run), arm=arm, pair=pair, run_id=run_id), ev=self.world.ev)

    def taper_check(self, out, log_path):
        self.taper_checks += 1
        record = RUN.taper_check(CORE.plan_runs(), results=self.world.results, out=lambda text: None, certifier=self.taper_certifier or self.world.certifier(passes=self.passes))
        write_json(out, record)
        if self.after_taper_check:
            self.after_taper_check()
        return 0 if record['passed'] else 8

    def uptime(self):
        return '100.0'


def start_queue(world, machine, **changes):
    return RUN.queue(world.options(**changes), system=machine, results=world.results, repo=ROOT, ev=world.ev)


UNITS = [CORE.unit_name(a, p) for a, p in CORE.plan_runs()]


# ---- the pins and the source -----------------------------------------------------------------------------------------------------------------------------------------------------------------
def source_error(world, pair=PAIRS[0]):
    with pytest.raises(CORE.EntryError) as caught:
        CORE.source_of(pair, world.manifest(pair), world.results)
    return str(caught.value)


def test_the_inputs_are_a5ps_as_s_p(world):
    d, _, _, manifest = CORE.input_files(PAIRS[0], world.results)
    assert d == world.results/'d0_v1_a5/inputs/A5P/stage_L2R_val_v1'/PAIRS[0] and manifest['preprocessing_d0']['arm'] == 'A5P' and manifest['preprocessing_d0']['oracle'] is False


def test_a_cell_gives_its_entry(world):
    source = CORE.source_of(PAIRS[0], world.manifest(), world.results)
    assert source['source_root'] == 'fd2_v1' and source['source_tier'] == 'Hx24@600' and source['oracle'] is False and source['charge_seconds'] == 47. and source['repair_eps'] == .05 and source['raw_path'] == CORE.field_dir(PAIRS[0], world.results)/'field_0600.npz'


def test_the_table_of_pins_has_the_first_5_pairs_and_every_key():
    keys = {'raw_field', 'repair_npz', 'run_json', 'repair_json', 'repair_metrics_json', 'tapered_u', 'repaired_u'}
    assert set(CORE.PINS) == {CORE.short_name(p) for p in PAIRS} and all(set(row) == keys and all(len(v) == 64 and int(v, 16) >= 0 for v in row.values()) for row in CORE.PINS.values())
    every = [v for row in CORE.PINS.values() for v in row.values()]
    assert len(set(every)) == len(every)


@needs_fd2
def test_the_pins_are_fd2s_real_records_and_their_format():
    """Only the small records are read here (and hashed); the two field files are hashed by the dry run."""
    for pair in PAIRS:
        pin, f, d = CORE.PINS[CORE.short_name(pair)], CORE.field_dir(pair), CORE.repair_dir(pair)
        assert sha(f/'run.json') == pin['run_json'] and sha(d/'repair.json') == pin['repair_json'] and sha(d/'repair_metrics.json') == pin['repair_metrics_json']
        source = CORE.source_of(pair, CORE.input_files(pair)[3])
        assert source['raw_sha256'] == pin['raw_field'] and source['repair_sha256'] == pin['repair_npz'] and source['hold']['seconds'] == 47.
    clocks = [CORE.source_of(p, CORE.input_files(p)[3])['hold']['real_task_clock'] for p in PAIRS]
    assert round(float(np.mean(clocks)), 1) == 143.6 and round(max(clocks), 1) == 170.3       # runs/20261105-fd2/result.md


def test_the_field_comes_from_fd2_v1_only(world):
    assert CORE.field_dir(PAIRS[0], world.results) == world.results/'fd2_v1/runs/Hx24_first5/0438-0439' and CORE.repair_dir(PAIRS[0], world.results) == world.results/'fd2_v1/runs/RHx24_c600/0438-0439'
    assert 'fd_v1' not in CORE.field_dir(PAIRS[0], world.results).parts and CORE.FD2_ROOT == 'fd2_v1' and (CORE.SOURCE_ARM, CORE.REPAIR_ARM, CORE.STEP, CORE.D_FACTOR) == ('Hx24', 'RHx24', 600, 24.)


@pytest.mark.parametrize('name', ['run.json', 'repair.json', 'repair_metrics.json'])
def test_a_record_changed_after_the_pins_is_refused(world, name):
    d = (CORE.field_dir if name == 'run.json' else CORE.repair_dir)(PAIRS[0], world.results)
    (d/name).write_text((d/name).read_text()+'\n')
    assert 'the pin' in source_error(world) and name in source_error(world)


def test_a_changed_raw_field_is_refused(world):
    d = CORE.field_dir(PAIRS[0], world.results)
    np.savez(d/CORE.FIELD_FILE, u=raw_field(9))
    with pytest.raises(CORE.EntryError, match='sha256'):
        CORE.construct(CORE.ARM, PAIRS[0], world.manifest(), world.results)


def test_a_changed_repair_file_is_refused(world):
    d = CORE.repair_dir(PAIRS[0], world.results)
    np.savez(d/'repair.npz', nodes=world.nodes[PAIRS[0]], values=world.values[PAIRS[0]]+1.)
    with pytest.raises(CORE.EntryError, match='sha256'):
        CORE.construct(CORE.ARM, PAIRS[0], world.manifest(), world.results)


def test_a_missing_file_is_missing(world):
    (CORE.repair_dir(PAIRS[0], world.results)/'repair.json').unlink()
    with pytest.raises(CORE.EntryMissing, match='no such file'):
        CORE.source_of(PAIRS[0], world.manifest(), world.results)


@pytest.mark.parametrize('which,key,value,word', [
    ('run', 'schema', 'x', 'schema'), ('run', 'run_id', 'x', 'run_id'), ('run', 'arm', 'Hx16', 'arm'), ('run', 'tier', 'rest600', 'tier'), ('run', 'pair', PAIRS[1], 'pair'), ('run', 'k', 36, ' k '), ('run', 'channels', 'truth', 'channels'), ('run', 'D_factor', 16., 'D factor'), ('run', 'native_iterations', 1200, 'native iterations'), ('run', 'oracle', True, 'oracle'), ('run', 'status', 'failed', 'status'),
    ('repair', 'schema', 'x', 'schema'), ('repair', 'run_id', 'x', 'run_id'), ('repair', 'arm', 'RHx16', 'arm'), ('repair', 'tier', 'rest600', 'tier'), ('repair', 'pair', PAIRS[1], 'pair'), ('repair', 'oracle', True, 'oracle'), ('repair', 'status', 'failed_back_to_frozen_rule', 'status'), ('repair', 'passed', False, 'passed'), ('repair', 'eps', .1, 'eps'),
    ('metrics', 'schema', 'x', 'schema'), ('metrics', 'run_id', 'x', 'run_id'), ('metrics', 'pair', PAIRS[1], 'pair'), ('metrics', 'oracle', True, 'oracle'), ('metrics', 'repaired_u_sha256', 'f'*64, 'repaired_u_sha256'),
])
def test_records_of_something_else_are_refused(world, which, key, value, word):
    world.change(PAIRS[0], which, lambda doc: doc.update({key: value}))
    assert word in source_error(world)


@pytest.mark.parametrize('change,word', [
    (lambda doc: doc['certificate'].update(status='infeasible'), 'did not pass'),
    (lambda doc: doc['certificate'].update(min_jd_lower_bound=0.), 'did not pass'),
    (lambda doc: doc['certificate'].update(n_uncertain=1), 'did not pass'),
    (lambda doc: doc['certificate'].update(state_hash='f'*64), 'state_hash'),
    (lambda doc: doc['source'].update(directory='/elsewhere'), 'source directory'),
    (lambda doc: doc['source'].update(file='field_2400.npz'), 'source file'),
    (lambda doc: doc['source'].update(field_sha256='f'*64), 'source field_sha256'),
    (lambda doc: doc['source'].update(tapered_u_sha256='f'*64), 'source tapered_u_sha256'),
    (lambda doc: doc['attempt'].update(sha256='f'*64), 'attempt sha256'),
    (lambda doc: doc.update(seconds_repair=None), 'seconds_repair'),
    (lambda doc: doc.update(seconds_warmup=float('nan')), 'seconds_warmup'),
])
def test_the_repair_record_must_be_this_repair(world, change, word):
    world.change(PAIRS[0], 'repair', change)
    assert word in source_error(world)


@pytest.mark.parametrize('change', [lambda c: c.update(accepted_alpha=.9), lambda c: c.update(status_at_1='infeasible'), lambda c: c.update(n_infeasible_at_1=1), lambda c: c.update(n_uncertain_at_1=1)])
def test_the_evaluation_must_have_certified_it_at_alpha_one(world, change):
    world.change(PAIRS[0], 'metrics', lambda doc: change(doc['after']['certificate']))
    assert "the evaluation's certificate at alpha = 1" in source_error(world)


@pytest.mark.parametrize('change,word', [
    (lambda s: s.update(oracle=True), 'oracle'),
    (lambda s: s.update(truth_labels_read=True), 'truth_labels_read'),
    (lambda s: s.update(pair_keys_read=['fixed', 'moving', 'fixed_seg35']), 'pair_keys_read'),
    (lambda s: s['sides']['fixed'].update(segmenter_id='unet35_v1'), 'sides'),
    (lambda s: s['sides']['moving'].update(segmenter_id='nnunet_v1'), 'sides'),
    (lambda s: s['sides']['fixed'].update(policy='member_0'), 'sides'),
])
def test_the_entry_is_bound_to_the_ensembles_predictions(world, change, word):
    """The brief: bound to unet35_v1_ens; a change of segmenter makes this entry another entry."""
    world.change(PAIRS[0], 'run', lambda doc: change(doc['channel_sources']))
    assert word in source_error(world)


@pytest.mark.parametrize('which', ['run', 'repair'])
@pytest.mark.parametrize('key', ['pair_sha256', 'fixed_subject'])
def test_a_field_of_other_images_is_refused(world, which, key):
    world.change(PAIRS[0], which, lambda doc: doc['data'].update({key: 'other'}))
    assert 'the field' in source_error(world)


# ---- the band: both branches ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_no_node_in_the_band_gives_the_repaired_field_bit_for_bit(world):
    prime, source, record = CORE.construct(CORE.ARM, PAIRS[0], world.manifest(), world.results)
    assert record['n_nodes_in_band'] == 0 and record['bitwise_equal_to_repaired'] is True and record['min_face_distance'] >= CORE.s2.TAPER
    assert CORE.s2.tapered(prime).tobytes() == repaired_of(world.raw[PAIRS[0]], world.nodes[PAIRS[0]], world.values[PAIRS[0]]).tobytes() and source['repaired_sha256'] == CORE.array_sha(CORE.s2.tapered(prime))


def test_a_node_at_the_first_row_with_w_one_is_not_in_the_band(world):
    raw = world.raw[PAIRS[0]]
    nodes = np.unique(np.append(interior_nodes(), band_node(4)))
    world.make_source(PAIRS[0], nodes=nodes, values=moved_values(raw, nodes, unmoved=0))
    _, _, record = CORE.construct(CORE.ARM, PAIRS[0], world.manifest(), world.results)
    assert record['n_nodes_in_band'] == 0 and record['min_face_distance'] == 4


def in_band(world, pair=PAIRS[3], distance=2, unmoved=0):
    raw = world.raw[pair]
    nodes = np.sort(np.append(interior_nodes(), band_node(distance)))
    values = moved_values(raw, nodes, unmoved=0)
    if unmoved:
        at = int(np.flatnonzero(nodes == band_node(distance))[0])
        values[at] = CORE.s2.tapered(raw).reshape(-1, 3)[nodes[at]]
    world.make_source(pair, nodes=nodes, values=values)


@pytest.mark.parametrize('distance', [1, 2, 3])
def test_a_repair_node_in_the_band_is_refused_with_its_counts(world, distance):
    in_band(world, distance=distance)
    with pytest.raises(CORE.BandError) as caught:
        CORE.construct(CORE.ARM, PAIRS[3], world.manifest(PAIRS[3]), world.results)
    assert caught.value.record['n_nodes_in_band'] == 1 and caught.value.record['min_face_distance'] == distance and caught.value.record['n_changed_nodes_in_band'] == 1


def test_a_free_node_in_the_band_is_refused_even_when_it_did_not_move(world):
    in_band(world, unmoved=1)
    with pytest.raises(CORE.BandError) as caught:
        CORE.construct(CORE.ARM, PAIRS[3], world.manifest(PAIRS[3]), world.results)
    assert caught.value.record['n_nodes_in_band'] == 1 and caught.value.record['n_changed_nodes_in_band'] == 0


def test_the_check_stops_on_a_node_in_the_band_and_reports_all_the_cells(world):
    in_band(world, pair=PAIRS[2])
    lines = []
    record = RUN.taper_check(CORE.plan_runs(), results=world.results, out=lines.append, certifier=world.certifier())
    assert record['passed'] is False and record['band_refusals'] == ['0440-0441'] and record['n_nodes_in_band'] == 1 and len(record['rows']) == 5 and sum('REFUSED (band)' in line for line in lines) == 1


def test_a_node_in_the_band_stops_the_queue_before_any_entry_and_any_run(world):
    in_band(world)
    machine = SYMachine(world)
    assert start_queue(world, machine) == 8
    stop = json_load(world.root/'queue_stop.json')['reason']
    assert 'taper check did not pass' in stop and '0441-0442' in stop and 'band' in stop and machine.launched == [] and not (world.root/'entries').exists() and not (world.root/'runs').exists()


# ---- the hold --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('seconds_repair', [13., 121.3, 400.])
def test_the_hold_is_47_seconds_whatever_the_entry_took(world, seconds_repair):
    """S_X's rule (the median of the records, capped at 200 s) is not this arm's.  The real time stays next to it, for the time account."""
    world.make_source(PAIRS[0], seconds_repair=seconds_repair)
    entry = CORE.write_entry(world.root, CORE.ARM, PAIRS[0], world.manifest(), world.results)
    assert entry['charge_seconds'] == 47. == SEED.ENTRY_CHARGE_SECONDS == S2CORE.HOLD_S_P and entry['hold']['seconds'] == 47. and entry['hold']['real_task_clock'] == pytest.approx(45.+seconds_repair+1.2) and entry['seconds_repair'] == seconds_repair and entry['source_seconds_total'] == 45.
    assert 'not the cost of the entry' in entry['hold']['rule']


def test_the_runs_have_the_hold_and_the_units_are_s_ps(world):
    machine = SYMachine(world)
    assert start_queue(world, machine) == 0
    for pair in PAIRS:
        run = CORE.run_dir(world.root, CORE.ARM, pair)
        assert json_load(run/'seed_entry.json')['charge_seconds'] == 47. and json_load(run/'sy_arm.json')['charge_seconds'] == 47. and json_load(run/'seed_arm.json')['charge_seconds'] == 47. and json_load(run/'metrics.json')['seed']['charge_seconds'] == 47.
        assert 'tag-sy-S_Y-' in (run/'service_unit.txt').read_text() and 'real task clock' in (run/'data.txt').read_text()


def test_the_reading_says_what_the_hold_is_not(world):
    assert start_queue(world, SYMachine(world)) == 0
    summary = (world.root/'analysis/summary_sy.md').read_text()
    assert '持时 47 s，与 S_P 相同，这不是入口的成本' in summary and '入口真实任务时钟' in summary and '截断值' in summary


def test_the_time_account_takes_the_real_entry_and_not_the_hold(world):
    assert start_queue(world, SYMachine(world)) == 0
    t = json_load(world.root/'analysis/decision_sy.json')['description']['time']
    entry = 45.+121.3+1.2
    assert t['hx24']['entry'] == pytest.approx(entry) and t['hx24']['total'] == pytest.approx(entry+91.+1.5)
    e2e = 327.
    assert t['s_y']['n'] == 5 and t['s_y']['entry'] == pytest.approx(entry) and t['s_y']['native'] == pytest.approx(e2e-47.) and t['s_y']['total'] == pytest.approx(entry+e2e-47.+91.+1.5+16.5)
    assert t['s_x']['entry'] == pytest.approx(152.3) and t['s_x']['native'] == pytest.approx(327.-152.) and t['s_p']['entry'] == 60. and t['s_p']['total'] == pytest.approx(60.+327.-47.+91.+16.5)


# ---- the reading ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def short(pair):
    return CORE.short_name(pair)


def fake_hx24(seg35=.8617, seg4=.9043):
    return {p: dict(metrics=metrics_of((seg35, seg4)), certificate=dict(accepted_alpha=1., status_at_1='certified_feasible', n_infeasible_at_1=0, n_uncertain_at_1=0), repaired_u_sha256='h', evaluator_sha256='ev') for p in PAIRS}


def fake_run(seg35, *, seg4=.91, sdlogj=.08, alpha=1., hold=47., hash_='h', cls='ok', solver=True, field=True, field_state='s', field_file='f', recorded_attempt=None, seed_passed=True, done='ok', evaluator='ev', injected=47.03, pair=None, self_check=True):
    certificate = dict(status='certified_feasible', passed=solver, state_hash='s', state_hash_matches_u=True, final_npz_sha256='f', final_npz_sha256_matches=True)
    seed = dict(passed=seed_passed, problems=[], accepted_alpha=alpha, charge_seconds=hold, injected_seconds=injected, repaired_u_sha256=hash_, transfer_attempts=[dict(alpha=1., state_hash=hash_ if recorded_attempt is None else recorded_attempt)])
    rec = dict(metrics=dict(arm=CORE.ARM, pair=pair, metrics=metrics_of((seg35, seg4), sdlogj=sdlogj), classification={'class': cls}, certificate=certificate, seed=seed, evaluator=dict(sha256=evaluator), self_check=dict(passed=self_check), run_labels_equal_truth=True, status='native_budget_exhausted'))
    if done is not None:
        rec['queue_done'] = dict(state=done)
    if field is not None:
        rec['field_certificate'] = dict(status='certified_feasible' if field else 'infeasible', passed=field, state_hash=field_state, file_sha256=field_file, min_jd_lower_bound=1e-4 if field else -.1, n_infeasible=0 if field else 2, n_uncertain=0)
    return rec


def read(deltas, base=.8617, **changes):
    runs = {(CORE.ARM, p): fake_run(base+d, **changes) for p, d in zip(PAIRS, deltas)}
    return DEC.decide(runs, fake_hx24(base), PAIRS, short, GROUPS, pin=None)


@pytest.mark.parametrize('deltas,code', [
    ((.01, .01, .01, .01, .01), DEC.ADDS),
    ((.03, .01, .01, .01, -.01), DEC.ADDS),
    ((.03, .01, .01, -.01, -.01), DEC.MIDDLE),
    ((.004, .004, .004, .004, .004), DEC.MIDDLE),
    ((.0019, .0019, .0019, .0019, .0019), DEC.NONE),
    ((.001, .001, .001, .001, .001), DEC.NONE),
    ((-.01, -.01, -.01, -.01, -.01), DEC.NONE),
    ((.05, -.04, -.04, .01, .01), DEC.NONE),
])
def test_the_table_of_the_reading(deltas, code):
    assert read(deltas)['reading_code'] == code


@pytest.mark.parametrize('delta,positive,code', [(.005, 4, DEC.ADDS), (.005, 5, DEC.ADDS), (.005, 3, DEC.MIDDLE), (.0049999, 5, DEC.MIDDLE), (.002, 5, DEC.NONE), (.002, 0, DEC.NONE), (.0020001, 5, DEC.MIDDLE), (.0020001, 0, DEC.MIDDLE), (-.1, 0, DEC.NONE), (.1, 3, DEC.MIDDLE), (.1, 4, DEC.ADDS)])
def test_the_bars_are_included(delta, positive, code):
    assert DEC.label_of(delta, positive) == code


def test_the_words_of_the_brief():
    assert DEC.READINGS == {DEC.ADDS: '原生层还能加 seg35', DEC.NONE: '原生层加不了 seg35', DEC.MIDDLE: '中间带', DEC.NOT_DECIDABLE: '不可判'} and (DEC.ADD_BAR, DEC.NO_BAR, DEC.MIN_POSITIVE, DEC.SDLOGJ_MAX) == (.005, .002, 4, .10)


def test_delta_is_paired_against_the_same_repaired_field():
    runs = {(CORE.ARM, p): fake_run(.86+i*.001) for i, p in enumerate(PAIRS)}
    hx24 = {p: dict(fake_hx24()[p], metrics=metrics_of((.85+i*.002, .9))) for i, p in enumerate(PAIRS)}
    result = DEC.decide(runs, hx24, PAIRS, short, GROUPS, pin=None)
    assert result['delta_seg35']['per_pair'][PAIRS[0]] == pytest.approx(.01) and result['delta_seg35']['per_pair'][PAIRS[4]] == pytest.approx(.86+.004-.858) and result['delta_seg35']['n'] == 5


def test_the_reading_says_seg4_and_sdlogj_without_using_them():
    result = read((.01,)*5, seg4=.5)
    assert result['reading_code'] == DEC.ADDS and 'Δ(seg4)' in result['reading'] and '官方 SDlogJ' in result['reading'] and '都只描述' in result['reading'] and result['delta_seg4']['mean'] < 0


@pytest.mark.parametrize('change,word', [
    (dict(cls='method_failure'), '运行失败'),
    (dict(cls='handover_only'), '运行失败'),
    (dict(solver=False), '终态证书的记录未通过'),
    (dict(field=None), '评测单元从场算的证书'),
    (dict(field=False), '评测单元从场算的证书未通过'),
    (dict(field_state='other'), '不是这次运行导出的场'),
    (dict(field_file='other'), '不是这次运行导出的场'),
    (dict(seed_passed=False), '入口核对未通过'),
    (dict(alpha=.9), '转移 α'),
    (dict(recorded_attempt='other'), '转移的 state hash'),
    (dict(hash_='other'), 'fd2_v1 记录的修复场'),
    (dict(hold=152.), '持时 152.0 s'),
    (dict(done=None), '运行没有以 ok 结束（queue_done None）'),
    (dict(done='handover_only'), '运行没有以 ok 结束'),
    (dict(evaluator='other'), '评测代码 other'),
    (dict(evaluator=None), '评测代码 None'),
])
def test_a_premise_of_one_run_not_met_is_not_decidable(change, word):
    runs = {(CORE.ARM, p): fake_run(.87) for p in PAIRS}
    runs[(CORE.ARM, PAIRS[2])] = fake_run(.87, **change)
    result = DEC.decide(runs, fake_hx24(), PAIRS, short, GROUPS, pin=None)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and word in result['reading'] and '0440-0441' in result['reading']


def test_the_mean_sdlogj_is_a_premise_and_one_high_is_only_an_anomaly():
    assert read((.01,)*5, sdlogj=.1001)['reading_code'] == DEC.NOT_DECIDABLE and read((.01,)*5, sdlogj=.10)['reading_code'] == DEC.ADDS
    runs = {(CORE.ARM, p): fake_run(.8717, sdlogj=.06) for p in PAIRS}
    runs[(CORE.ARM, PAIRS[0])] = fake_run(.8717, sdlogj=.12)
    assert DEC.decide(runs, fake_hx24(), PAIRS, short, GROUPS, pin=None)['reading_code'] == DEC.ADDS
    assert [a['pair'] for a in DEC.anomalies(runs, CORE.plan_runs(), short) if '0.1200' in a['what']] == ['0438-0439']


def test_nan_is_not_a_reading():
    runs = {(CORE.ARM, p): fake_run(.87) for p in PAIRS}
    runs[(CORE.ARM, PAIRS[0])] = fake_run(float('nan'))
    assert DEC.decide(runs, fake_hx24(), PAIRS, short, GROUPS, pin=None)['reading_code'] == DEC.NOT_DECIDABLE
    runs[(CORE.ARM, PAIRS[0])] = fake_run(.87, sdlogj=float('nan'))
    assert DEC.decide(runs, fake_hx24(), PAIRS, short, GROUPS, pin=None)['reading_code'] == DEC.NOT_DECIDABLE


def test_a_missing_pair_is_not_decidable():
    runs = {(CORE.ARM, p): fake_run(.87) for p in PAIRS[:4]}
    result = DEC.decide(runs, fake_hx24(), PAIRS, short, GROUPS, pin=None)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and '运行失败' in result['reading'] and '0442-0443' in result['reading'] and result['delta_seg35']['mean'] is None


def test_a_final_map_without_values_is_not_decidable():
    runs = {(CORE.ARM, p): fake_run(.87) for p in PAIRS}
    runs[(CORE.ARM, PAIRS[0])]['metrics']['metrics'] = None
    result = DEC.decide(runs, fake_hx24(), PAIRS, short, GROUPS, pin=None)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and '不可算' in result['reading']


def test_the_reference_is_a_premise():
    runs = {(CORE.ARM, p): fake_run(.87) for p in PAIRS}
    hx24 = fake_hx24()
    del hx24[PAIRS[1]]
    assert '没有 repair_metrics.json' in DEC.decide(runs, hx24, PAIRS, short, GROUPS, pin=None)['reading']
    hx24 = fake_hx24()
    hx24[PAIRS[0]]['certificate']['status_at_1'] = 'infeasible'
    assert '评测单元在 α = 1 的证书未通过' in DEC.decide(runs, hx24, PAIRS, short, GROUPS, pin=None)['reading']


def test_the_pin_of_the_reference():
    runs = {(CORE.ARM, p): fake_run(.87) for p in PAIRS}
    assert DEC.decide(runs, fake_hx24(), PAIRS, short, GROUPS)['reading_code'] != DEC.NOT_DECIDABLE
    assert '不是钉住的' in DEC.decide(runs, fake_hx24(.85), PAIRS, short, GROUPS)['reading']


def test_the_values_have_hd95_and_the_groups_are_the_evaluations():
    metrics = metrics_of((.8, .9), hd95=1.5, small=.6)
    assert DEC.value_of(metrics, 'hd95', GROUPS) == 1.5 and DEC.value_of(metrics, 'small_six', GROUPS) == pytest.approx(.6) and DEC.value_of(metrics, 'cortex', GROUPS) == .85
    assert 'hd95' in DEC.DIFFERENCE_KEYS and {'seg4', 'cortex', 'white_matter', 'large_six', 'small_six', 'other_23', 'sdlogj_official', 'sdlogj_brain'} <= set(DEC.DIFFERENCE_KEYS)
    assert DEC.paired({PAIRS[0]: None}, {PAIRS[0]: metrics}, PAIRS[:1], 'seg35', GROUPS)['mean'] is None and DEC.paired({PAIRS[0]: metrics}, {PAIRS[0]: metrics}, PAIRS[:1], 'hd95', GROUPS)['mean'] == 0.


def test_the_windows_count_on_the_task_clock_from_the_first_trial(neutral):
    """inner_attempts.json's elapsed_seconds starts again at each outer step; visible_seconds does not.  A wrong clock would put every trial in the first window (it did, before the dry run's comparison with the review of S_X)."""
    run = neutral/'run'
    (run/TASK).mkdir(parents=True)
    write_json(run/'protocol.json', dict(task=dict(name=TASK)))
    assert DEC.rejection_windows(run) is None
    rows = [dict(level=3, visible_seconds=100.+s, elapsed_seconds=1., reason=r) for s, r in ((0., 'JACOBIAN_VIOLATION'), (14.9, None), (15., 'JACOBIAN_VIOLATION'), (159.9, None), (160., 'FEASIBILITY_UNCERTAIN'), (200., 'JACOBIAN_VIOLATION'))]+[dict(level=2, visible_seconds=1., elapsed_seconds=1., reason='JACOBIAN_VIOLATION')]
    write_json(run/TASK/'inner_attempts.json', rows)
    windows = DEC.rejection_windows(run)
    assert windows['0–15 s'] == [2, 1] and windows['15–30 s'] == [1, 1] and windows['120–175 s'] == [2, 0] and windows['175 s 之后'] == [1, 1] and windows['0–160 s'] == [4, 2]
    pooled = DEC.pooled_windows({PAIRS[0]: dict(windows=windows), PAIRS[1]: dict(windows=windows)}, PAIRS[:2])
    assert pooled['0–15 s'] == .5 and pooled['30–60 s'] is None and DEC.pooled_windows({PAIRS[0]: dict(windows=windows)}, PAIRS[:2]) is None


# ---- the certificate from the field ------------------------------------------------------------------------------------------------------------------------------------------------------------
def finished_run(world, pair=PAIRS[0]):
    run = CORE.run_dir(world.root, CORE.ARM, pair)
    run.mkdir(parents=True)
    write_json(run/'protocol.json', dict(task=dict(name=TASK)))
    write_json(run/'seed_entry.json', CORE.write_entry(world.base/'scratch_entries', CORE.ARM, pair, world.manifest(pair), world.results))
    (run/'run_id').write_text('rid\n')
    return run, write_task(run)


def options_of(run, pair=PAIRS[0]):
    return SimpleNamespace(run=str(run), arm=CORE.ARM, pair=pair, run_id='rid')


def test_the_certificate_is_computed_from_the_field_and_not_read(world):
    """The solver's record says nothing here: the fake certifier sees the field of the file, and says what it says."""
    run, task = finished_run(world)
    seen = []
    certify, config, taper, grid = world.certifier(passes=False)

    def spy(seed, g, config=None):
        seen.append(CORE.array_sha(seed))
        return certify(seed, g, config=config)
    assert RUN.certify_one(options_of(run), results=world.results, certifier=(spy, config, taper, grid)) == 0
    doc = json_load(run/'field_certificate.json')
    with np.load(task/'final_native.npz') as z:
        state = CORE.array_sha(np.array(z['u']))
    assert seen == [state] and doc['passed'] is False and doc['status'] == 'infeasible' and doc['field_state_hash'] == state and doc['file_sha256'] == sha(task/'final_native.npz') and json_load(task/'final_native.json')['certificate']['status'] == 'certified_feasible' and doc['lower_bound_as_the_solver'] is False


def test_a_certificate_of_another_state_does_not_pass(world):
    run, _ = finished_run(world)
    certify, config, taper, grid = world.certifier()

    def other(seed, g, config=None):
        return SimpleNamespace(**dict(vars(certify(seed, g, config=config)), state_hash='another'))
    assert RUN.certify_one(options_of(run), results=world.results, certifier=(other, config, taper, grid)) == 0
    assert json_load(run/'field_certificate.json')['passed'] is False


def test_a_zero_bound_does_not_pass(world):
    run, _ = finished_run(world)
    certify, config, taper, grid = world.certifier()

    def zero(seed, g, config=None):
        return SimpleNamespace(**dict(vars(certify(seed, g, config=config)), min_jd_lower_bound=0.))
    assert RUN.certify_one(options_of(run), results=world.results, certifier=(zero, config, taper, grid)) == 0
    assert json_load(run/'field_certificate.json')['passed'] is False


def test_a_run_without_an_exported_map_has_no_certificate(world):
    run, task = finished_run(world)
    (task/'final_native.npz').unlink()
    assert RUN.certify_one(options_of(run), results=world.results, certifier=world.certifier()) == 8 and not (run/'field_certificate.json').exists()


def test_the_certificate_opens_only_the_final_map(world, monkeypatch):
    run, _ = finished_run(world)
    opened = []
    real = np.load

    def spy(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', spy)
    RUN.certify_one(options_of(run), results=world.results, certifier=world.certifier())
    assert opened == ['final_native.npz']


def test_the_references_are_certified_from_their_fields_too(world):
    """The brief: every 过证书 of the summary from the evaluation.  S_P's and S_X's records are the solvers'; the taper check recomputes them, before the first run, and only reads."""
    before = tree(world.results/'seed_v1'), tree(world.results/'s2_v1')
    record = RUN.taper_check(CORE.plan_runs(), results=world.results, out=lambda text: None, certifier=world.certifier())
    assert record['passed'] is True and [(r['arm'], r['short']) for r in record['references']] == [(arm, CORE.short_name(p)) for p in PAIRS for arm in ('S_P', 'S_X')] and all(r['passed'] for r in record['references']) and record['references_failed'] == []
    assert (tree(world.results/'seed_v1'), tree(world.results/'s2_v1')) == before


def test_a_reference_without_its_field_is_refused_by_the_check_and_stops_the_queue_at_the_cheap_checks(world):
    (CORE.reference_dir(PAIRS[1], world.results)/TASK/'final_native.npz').unlink()
    record = RUN.taper_check(CORE.plan_runs(), results=world.results, out=lambda text: None, certifier=world.certifier())
    assert record['passed'] is False and record['references_failed'] == ['S_P 0439-0440'] and all(r['passed'] for r in record['rows'])
    machine = SYMachine(world)
    assert start_queue(world, machine) == 8
    assert 'S_P 0439-0440: no final_native.npz' in json_load(world.root/'queue_stop.json')['reason'] and machine.taper_checks == 0 and machine.launched == []


def test_a_reference_that_does_not_pass_stops_the_queue_before_any_run(world):
    with np.load(CORE.s_x_dir(PAIRS[2], world.results)/TASK/'final_native.npz') as z:
        bad = CORE.array_sha(np.array(z['u']))
    certify, config, taper, grid = world.certifier()

    def fails_on_it(seed, g, config=None):
        result = certify(seed, g, config=config)
        return SimpleNamespace(**dict(vars(result), status='infeasible', min_jd_lower_bound=-.1, n_infeasible=1)) if result.state_hash == bad else result
    machine = SYMachine(world)
    machine.taper_certifier = (fails_on_it, config, taper, grid)
    assert start_queue(world, machine) == 8
    stop = json_load(world.root/'queue_stop.json')['reason']
    assert 'S_X 0440-0441, final map: infeasible' in stop and machine.launched == [] and not (world.root/'entries').exists()


def test_the_taper_check_fails_a_field_that_does_not_pass_or_is_not_the_field(world):
    record = RUN.taper_check(CORE.plan_runs(), results=world.results, out=lambda text: None, certifier=world.certifier(passes=False))
    assert record['passed'] is False and all('the certificate did not pass' in ' '.join(r['problems']) for r in record['rows'])
    certify, config, taper, grid = world.certifier()

    def other(seed, g, config=None):
        return SimpleNamespace(**dict(vars(certify(seed, g, config=config)), state_hash='another'))

    def other_grid(seed, g, config=None):
        return SimpleNamespace(**dict(vars(certify(seed, g, config=config)), grid_id='another'))
    assert 'state hash' in ' '.join(RUN.taper_check(CORE.plan_runs()[:1], results=world.results, out=lambda t: None, certifier=(other, config, taper, grid))['rows'][0]['problems'])
    assert 'grid id' in ' '.join(RUN.taper_check(CORE.plan_runs()[:1], results=world.results, out=lambda t: None, certifier=(other_grid, config, taper, grid))['rows'][0]['problems'])


def test_the_certificate_configuration_is_s2s():
    """frozen_certifier is run_s2's, which has configuration C in the words of run_mc0.certificate_config (test_s2_tools); certify_map's default is not it."""
    assert RUN.rs2.frozen_certifier is S2.frozen_certifier and 'PrescreenCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=65536)' in (ROOT/'experiments/s2/run_s2.py').read_text()


# ---- the gate ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('change,word', [
    (dict(unit_list=['tag-nn-train.service']), 'other units'),
    (dict(unit_list=['seq-vp-chain.service']), 'other units'),
    (dict(unit_list=['tag-s2-queue.service']), 'other units'),
    (dict(load=1.5), 'load'),
    (dict(gib={'/home': 99., '/mnt/c': 500.}), '/home'),
    (dict(gib={'/home': 500., '/mnt/c': 14.9}), '/mnt/c'),
    (dict(gpu='compute pids [1]'), 'GPU busy'),
    (dict(mib=15*1024-1), 'GPU free'),
])
def test_each_reason_closes_the_gate(neutral, change, word):
    machine = FakeSystem()
    assert RUN.busy_reason(machine, neutral, 15.) is None
    for key, value in change.items():
        setattr(machine, key, value)
    assert word in RUN.busy_reason(machine, neutral, 15.)


def test_the_gate_is_as_s2s(neutral):
    for change in (dict(load=1.5), dict(mib=15*1024-1), dict(gpu='x'), dict(unit_list=['tag-fl-queue.service'])):
        machine = FakeSystem()
        for key, value in change.items():
            setattr(machine, key, value)
        assert RUN.busy_reason(machine, neutral, 15.) == S2.busy_reason(machine, neutral, 15.)
    assert (RUN.GPU_FREE_MIB, RUN.DEFAULT_MIN_HOST_GIB) == (S2.GPU_FREE_MIB, S2.DEFAULT_MIN_HOST_GIB) == (15*1024, 15.)


def test_the_gate_is_open_at_the_floors_and_for_its_own_queue(neutral):
    machine = FakeSystem()
    machine.load, machine.gib, machine.mib, machine.unit_list = 1.49, {'/home': 100., '/mnt/c': 15.}, 15*1024, ['tag-sy-queue.service']
    assert RUN.busy_reason(machine, neutral, 15.) is None


def test_the_hold_file_closes_the_gate(neutral):
    (neutral/'hold').write_text('')
    assert RUN.busy_reason(FakeSystem(), neutral, 15.) == 'hold file'


def test_a_closed_gate_stops_after_its_time(neutral):
    machine = FakeSystem()
    machine.load = 5.
    lines = []
    with pytest.raises(RUN.GateClosed, match='gate closed for 30 minutes'):
        RUN.wait_for_gate(machine, neutral, 30, lines.append, 15.)
    assert machine.clock >= 30*60 and all('load' in line for line in lines)


@pytest.mark.parametrize('how', ['hold', 'load', 'nn_unit', 'nn_gpu'])
def test_a_closed_gate_keeps_the_certificates_and_the_entries_from_starting(world, how):
    """While nnU-Net trains the gate is closed twice over: its unit, and its compute process."""
    machine = SYMachine(world)
    world.root.mkdir(parents=True)
    if how == 'hold':
        (world.root/'hold').write_text('')
    elif how == 'load':
        machine.load = 5.
    elif how == 'nn_unit':
        machine.unit_list = ['tag-nn-train.service']
    else:
        machine.gpu = 'compute pids [1], utilization 97.0%'
    assert start_queue(world, machine) == 8
    assert 'gate closed' in json_load(world.root/'queue_stop.json')['reason'] and machine.taper_checks == 0 and not (world.root/'entries').exists() and not (world.root/'runs').exists() and machine.launched == []
    assert not list(world.root.glob('taper_check_*')) and not (world.root/'batch.json').exists()


def test_the_certificates_wait_for_the_gate_and_then_start(world):
    machine = SYMachine(world)
    machine.unit_list = ['tag-nn-train.service']
    real_sleep = machine.sleep

    def sleep(seconds):                                  # the training ends after a while
        real_sleep(seconds)
        if machine.clock >= 600:
            machine.unit_list = []
    machine.sleep = sleep
    assert start_queue(world, machine) == 0
    assert machine.taper_checks == 1 and 'waiting: other units' in (world.root/'queue.log').read_text() and len(machine.launched) == 5


def test_the_cheap_checks_come_before_the_gate(world):
    """A missing file stops at once: it does not wait 30 minutes behind a closed gate first."""
    (world.results/'tc_v1/runs/S_P'/PAIRS[2]/'curve.json').unlink()
    machine = SYMachine(world)
    machine.load = 5.
    assert start_queue(world, machine) == 8
    assert 'no curve.json' in json_load(world.root/'queue_stop.json')['reason'] and machine.clock == 0.


def test_the_gate_stays_closed_between_runs(world):
    """Each run stands behind the gate: a training started between two runs stops the queue; what ran stays."""
    machine = SYMachine(world)
    real = machine.launch

    def launch(argv, log):
        code = real(argv, log)
        if len(machine.launched) == 2:
            machine.unit_list = ['tag-nn-train.service']
        return code
    machine.launch = launch
    assert start_queue(world, machine) == 8
    assert len(machine.launched) == 2 and 'gate closed' in json_load(world.root/'queue_stop.json')['reason'] and json_load(world.root/'analysis/decision_sy.json')['reading_code'] == DEC.NOT_DECIDABLE


# ---- the plan's cells, the start strings -----------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_plan_is_the_briefs():
    assert CORE.plan_runs() == [('S_Y', p) for p in PAIRS] and PAIRS == tuple(f'OASIS_OAS1_{n:04d}_MR1__OASIS_OAS1_{n+1:04d}_MR1' for n in range(438, 443)) and CORE.ROOT_NAME == 'sy_v1'


@pytest.mark.parametrize('arm,pair', [('S_Y', S2CORE.EXCEPTION_PAIR), ('S_Y', 'OASIS_OAS1_0446_MR1__OASIS_OAS1_0447_MR1'), ('S_Y', 'OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1'), ('S_X', PAIRS[0]), ('S_F', PAIRS[0]), ('S_P', PAIRS[0])])
def test_other_cells_are_refused_before_any_file(arm, pair, neutral):
    for call in (lambda: CORE.require_cell(arm, pair), lambda: CORE.run_dir(neutral, arm, pair), lambda: CORE.entry_paths(neutral, arm, pair), lambda: RUN.prepare_argv(arm, pair, neutral), lambda: CORE.unit_name(arm, pair), lambda: RUN.unit_argv(arm, pair, neutral, neutral), lambda: CORE.construct(arm, pair, {}, neutral)):
        with pytest.raises(SystemExit):
            call()
    if arm == 'S_Y':
        for call in (lambda: CORE.field_dir(pair, neutral), lambda: CORE.repair_dir(pair, neutral), lambda: CORE.source_of(pair, {}, neutral)):
            with pytest.raises(SystemExit):
                call()


@pytest.mark.parametrize('step', ['evaluate-one', 'certify-one', 'curve-one'])
def test_the_steps_refuse_another_pair(step, neutral):
    with pytest.raises(SystemExit, match='S_Y runs'):
        RUN.main([step, str(neutral/'x'), '--arm', 'S_Y', '--pair', 'OASIS_OAS1_0443_MR1__OASIS_OAS1_0444_MR1', '--run-id', 'x'])


def test_the_subjects_are_on_the_development_allowlist_and_off_the_training_and_holdout_lists():
    sys.path.insert(0, str(ROOT))
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')
    held = set((ROOT/'splits/bayesreg_v1/holdout_B_subjects.txt').read_text().split())
    train = set((ROOT/'splits/bayesreg_v1/train_subjects.txt').read_text().split())
    for _, pair in CORE.plan_runs():
        a, b = pair.split('__')
        check_pair_subjects(a, b, allowlist)
        assert not {a, b} & held and not {a, b} & train


def test_the_start_strings_hold_no_forbidden_word_and_the_right_names():
    root = Path('/x/sy_v1')
    for argv in [RUN.queue_launch(root)]+[RUN.unit_argv(a, p, root/'runs'/a/p, root) for a, p in CORE.plan_runs()]+[RUN.prepare_argv(a, p, root/'runs'/a/p) for a, p in CORE.plan_runs()]:
        assert 'pytest' not in ' '.join(argv) and 'seq-vp' not in ' '.join(argv)
    assert RUN.queue_launch(root)[2] == '--unit=tag-sy-queue' and CORE.unit_name('S_Y', PAIRS[0]) == 'tag-sy-S_Y-0438-0439' and RUN.new_run_id('S_Y', PAIRS[0]).startswith('sy_S_Y_0438-0439_') and 'experiments/sy/run_sy.py' in RUN.queue_launch(root)


def test_the_units_are_s_ps_but_for_the_name():
    root = Path('/x/sy_v1')
    run = root/'runs/S_Y'/PAIRS[0]
    argv, other = RUN.unit_argv('S_Y', PAIRS[0], run, root), SEED.unit_argv('S_P', PAIRS[0], run, root)
    assert [a for a in argv if not a.startswith('--unit=')] == [a for a in other if not a.startswith('--unit=')]
    assert '--property=MemoryMax=12G' in argv and 'seed_inject.py' in argv[-1] and f'{SEED.WATCHDOG}s' in argv[-1]


def test_the_prepare_lines_are_s_ps_with_their_stages():
    run = Path('/x/sy_v1/runs/S_Y')/PAIRS[0]
    argv, other = RUN.prepare_argv('S_Y', PAIRS[0], run), SEED.prepare_argv('S_P', PAIRS[0], run)
    assert [a.replace('SY-S_Y', 'SEED-S_P') for a in argv] == other and '330' in argv and '1,1,0.1,0.01' in argv


def test_the_only_root_is_sy_v1(world, capsys):
    other = world.results/'tc_v1'
    before = tree(other)
    assert RUN.queue(world.options(root=str(other)), system=SYMachine(world), results=world.results, repo=ROOT, ev=world.ev) == 2
    assert tree(other) == before and 'writes only to' in capsys.readouterr().err
    assert CORE.root_problem(world.root, world.results) is None and CORE.root_problem(world.base/'sy_v2', world.results) and CORE.root_problem(world.results/'s2_v1', world.results)


@pytest.mark.parametrize('value', [14.9, float('nan')])
def test_the_floor_cannot_be_lowered(world, value, capsys):
    assert RUN.queue(world.options(min_host_gib=value), system=SYMachine(world), results=world.results, repo=ROOT, ev=world.ev) == 2
    assert not world.root.exists() and '--min-host-gib' in capsys.readouterr().err


def test_the_commands_refuse_another_root(neutral, capsys):
    assert RUN.main(['decide', '--root', str(neutral/'s2_v1')]) == 2
    assert 'writes only to' in capsys.readouterr().err


@pytest.mark.parametrize('step', ['evaluate-one', 'certify-one', 'curve-one'])
def test_the_steps_refuse_a_run_elsewhere(step, neutral, capsys):
    assert RUN.main([step, str(neutral/'s2_v1/runs/S_X'/PAIRS[0]), '--arm', 'S_Y', '--pair', PAIRS[0], '--run-id', 'x']) == 2
    assert 'writes only to' in capsys.readouterr().err


# ---- the firewall -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def string_constants(path):
    return {n.value for n in ast.walk(ast.parse(Path(path).read_text())) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def test_no_truth_key_on_the_registration_path():
    for name in ('sy_core.py', 'run_sy.py', 'decide_sy.py'):
        strings = string_constants(ROOT/'experiments/sy'/name)
        assert not [k for k in ('fixed_seg35', 'moving_seg35', 'fixed_seg4', 'moving_seg4') if k in strings], name
    source = (ROOT/'experiments/sy/sy_core.py').read_text()
    assert 'truth_labels' not in source and 'np.load' not in source          # the field files are s2_core's readers


def test_a_construction_opens_only_the_two_field_files(world, monkeypatch):
    """The counterexample is in the fixture: every pair.npz there holds truth labels.  Any np.load of anything else here fails."""
    opened = []
    real = np.load

    def spy(path, *args, **kwargs):
        opened.append(Path(path))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', spy)
    CORE.construct(CORE.ARM, PAIRS[0], world.manifest(), world.results)
    assert [p.name for p in opened] == [CORE.FIELD_FILE, 'repair.npz'] and all(p.is_relative_to(world.results/'fd2_v1') for p in opened)


def test_the_taper_check_never_opens_a_pair_file(world, monkeypatch):
    opened = []
    real = np.load

    def spy(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', spy)
    RUN.taper_check(CORE.plan_runs(), results=world.results, out=lambda text: None, certifier=world.certifier())
    assert set(opened) == {CORE.FIELD_FILE, 'repair.npz', 'final_native.npz'}


def test_only_the_steps_after_a_run_read_truth_labels():
    tree_ = ast.parse((ROOT/'experiments/sy/run_sy.py').read_text())
    functions = {n.name: n for n in ast.walk(tree_) if isinstance(n, ast.FunctionDef)}
    readers = [name for name, n in functions.items() if any(isinstance(x, ast.Attribute) and x.attr in ('evaluate_run', 'curve_of_run', 'truth_labels') for x in ast.walk(n))]
    assert sorted(readers) == ['curve_one', 'evaluate_one']


# ---- the cleanup ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def own_run(world, pair=PAIRS[0]):
    run, task = finished_run(world, pair)
    write_json(run/'curve.json', dict(complete=True, self_check=dict(passed=True), run_dir=str(run), run_id='rid'))
    return run, task


def test_the_cleanup_takes_only_the_recorded_intermediate_maps(world):
    run, task = own_run(world)
    np.savez(task/'committed_9999_L3.npz', u=np.zeros(1))    # no record of it
    (task/'other.txt').write_text('x')
    n = CORE.cleanup_after_curve(run, world.root/'runs', CURVE)
    record = json_load(run/'cleanup.json')
    assert n == len(ELAPSED)+2 and (task/'final_native.npz').is_file() and (task/'other.txt').is_file() and (task/'committed_9999_L3.npz').is_file() and not (task/'committed_0001_L3.npz').exists() and not (task/'level_3_preoptimizer_done.npz').exists()
    assert [r['name'] for r in record['not_deleted_unrecorded']] == ['committed_9999_L3.npz'] and record['deleted_bytes'] > 0 and '(class ok only' in record['rule']


def test_the_cleanup_does_what_s2s_does(world):
    """The same tree under s2_v1 and sy_v1: the same files go, the same record, but for the rule's words."""
    run, task = own_run(world)
    s2_root = world.base/'s2_copy'/S2CORE.ROOT_NAME
    copy = s2_root/'runs/S_F'/PAIRS[0]
    shutil.copytree(run, copy)
    doc = json_load(copy/'curve.json')
    doc['run_dir'] = str(copy)
    write_json(copy/'curve.json', doc)
    n1, n2 = CORE.cleanup_after_curve(run, world.root/'runs', CURVE), S2CORE.cleanup_after_curve(copy, s2_root/'runs', CURVE)
    a, b = json_load(run/'cleanup.json'), json_load(copy/'cleanup.json')
    a.pop('rule'), b.pop('rule')
    assert n1 == n2 and a == b and sorted(p.name for p in task.iterdir()) == sorted(p.name for p in (copy/task.name).iterdir())


def test_the_cleanup_refuses_anything_outside_sy_v1(world):
    original = CORE.s_x_dir(PAIRS[0], world.results)
    write_json(original/'curve.json', dict(complete=True, self_check=dict(passed=True), run_dir=str(original), run_id=None))
    before = tree(original)
    for runs_root in (world.results/'s2_v1/runs', world.results/'seed_v1/runs', world.root/'runs'):
        with pytest.raises(CURVE.CurveError, match='not a run directory'):
            CORE.cleanup_after_curve(original, runs_root, CURVE)
    assert tree(original) == before


@pytest.mark.parametrize('doc', [None, dict(complete=False, self_check=dict(passed=True)), dict(complete=True, self_check=dict(passed=False))])
def test_the_cleanup_waits_for_a_complete_curve(world, doc):
    run, task = own_run(world)
    (run/'curve.json').unlink()
    if doc is not None:
        write_json(run/'curve.json', dict(doc, run_dir=str(run), run_id='rid'))
    before = tree(task)
    with pytest.raises(CURVE.CurveError):
        CORE.cleanup_after_curve(run, world.root/'runs', CURVE)
    assert tree(task) == before


def test_the_cleanup_refuses_a_task_directory_elsewhere(world):
    run, task = own_run(world)
    outside = world.base/'outside_task'
    shutil.move(task, outside)
    (run/task.name).symlink_to(outside)
    before = tree(outside)
    with pytest.raises(CURVE.CurveError, match='not a child'):
        CORE.cleanup_after_curve(run, world.root/'runs', CURVE)
    assert tree(outside) == before


# ---- the entry file ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_entry_file_is_what_the_wrapper_takes_and_tapers_to_the_repaired_field(world):
    entry = CORE.write_entry(world.root, CORE.ARM, PAIRS[0], world.manifest(), world.results)
    u = INJ.load_field(entry)
    assert CORE.array_sha(CORE.s2.tapered(u)) == entry['repaired_u_sha256'] == entry['tapered_sha256'] and entry['source_tier'] == 'Hx24@600' and entry['source_root'] == 'fd2_v1' and entry['oracle'] is False and entry['evaluation_lower_bound'] == 1e-4


def test_the_entry_is_written_once_and_never_overwritten(world):
    first = CORE.write_entry(world.root, CORE.ARM, PAIRS[0], world.manifest(), world.results)
    before = tree(world.root/'entries')
    assert CORE.write_entry(world.root, CORE.ARM, PAIRS[0], world.manifest(), world.results) == first and tree(world.root/'entries') == before
    field = CORE.entry_paths(world.root, CORE.ARM, PAIRS[0])[0]
    np.savez(field, u=raw_field(9))
    with pytest.raises(CORE.EntryError, match='nothing overwritten'):
        CORE.write_entry(world.root, CORE.ARM, PAIRS[0], world.manifest(), world.results)


def test_an_incomplete_entry_is_kept(world):
    field, record = CORE.entry_paths(world.root, CORE.ARM, PAIRS[0])
    field.parent.mkdir(parents=True)
    np.savez(field, u=raw_field(9))
    with pytest.raises(CORE.EntryError, match='incomplete entry'):
        CORE.write_entry(world.root, CORE.ARM, PAIRS[0], world.manifest(), world.results)
    assert not record.exists()


def test_nothing_of_a_refused_entry_is_written(world):
    in_band(world, pair=PAIRS[0])
    with pytest.raises(CORE.BandError):
        CORE.write_entry(world.root, CORE.ARM, PAIRS[0], world.manifest(), world.results)
    assert not (world.root/'entries').exists()


# ---- the queue on a fake machine, the real steps -----------------------------------------------------------------------------------------------------------------------------------------------
READ_ONLY = ('fd2_v1', 's2_v1', 'seed_v1', 'tc_v1', 'd0_v1_a5')


def read_only_trees(world):
    return {name: tree(world.results/name) for name in READ_ONLY if (world.results/name).exists()}


def test_the_queue_runs_the_plan_and_writes_only_under_its_root(world):
    machine = SYMachine(world)
    before = read_only_trees(world)
    assert start_queue(world, machine) == 0
    assert machine.launched == UNITS and read_only_trees(world) == before and machine.taper_checks == 1 and machine.certified == list(PAIRS)
    assert machine.sleeps.count(SEED.BETWEEN_RUNS_SECONDS) == 4
    for arm, pair in CORE.plan_runs():
        run = CORE.run_dir(world.root, arm, pair)
        entry, metrics = json_load(run/'seed_entry.json'), json_load(run/'metrics.json')
        assert entry['field_path'] == str(CORE.entry_paths(world.root, arm, pair)[0]) and (run/'seed_inject.py').read_bytes() == (SEED.HERE/'inject_run.py').read_bytes()
        assert metrics['seed']['passed'] is True and metrics['seed']['accepted_alpha'] == 1. and metrics['seed']['repaired_u_sha256'] == entry['repaired_u_sha256'] == CORE.PINS[CORE.short_name(pair)]['repaired_u'] and metrics['sy']['arm'] == arm and metrics['method'] == RUN.method_name(arm)
        assert json_load(run/'field_certificate.json')['passed'] is True and json_load(run/'field_certificate.json')['state_hash'] == metrics['certificate']['state_hash']
        assert json_load(run/'curve.json')['complete'] is True and json_load(run/'queue_done.json')['state'] == 'ok' and (run/'run_id').read_text().startswith(f'sy_{arm}_')
        assert not list((run/TASK).glob('committed_*')) and not list((run/TASK).glob('level_*')) and (run/TASK/'final_native.npz').is_file() and json_load(run/'cleanup.json')['deleted']
        assert (run/'cmd.txt').is_file() and (run/'git.txt').is_file() and (run/'data.txt').is_file() and (run/'env.txt').is_file() and (run/'seed').is_file() and (run/'sy_arm.json').is_file() and 'unet35_v1_ens' in (run/'data.txt').read_text()
    log = (world.root/'queue.log').read_text()
    assert 'taper check passed on 5 cells and 10 final maps' in log and 'reading:' in log and 'Handed to the lead' in log and not (world.root/'queue_stop.json').exists()


def test_the_analysis_reports_what_the_brief_asks(world):
    assert start_queue(world, SYMachine(world)) == 0
    out = world.root/'analysis'
    decision, summary = json_load(out/'decision_sy.json'), (out/'summary_sy.md').read_text()
    assert decision['reading_arm'] == 'S_Y' and decision['delta_seg35']['n'] == 5 and decision['delta_seg4']['n'] == 5 and decision['reading_code'] != DEC.NOT_DECIDABLE
    d = decision['description']
    for name in ('against_hx24', 'against_s_p', 'against_s_x'):
        assert set(d[name]) == set(DEC.DIFFERENCE_KEYS) and d[name]['seg35']['n'] == 5 and d[name]['hd95']['n'] == 5
    assert d['s_x_against_its_entry']['seg35']['mean'] == pytest.approx(.8526-.8507) and d['s_x_against_its_entry']['seg4']['mean'] == pytest.approx(.9011-.8827) and [r['t'] for r in d['s_x_against_its_entry_by_t']][-1] == CURVE.FINAL
    assert d['against_s_p_alpha_one']['seg35']['n'] == 4 and d['s_p_alpha']['0439-0440'] == .9 and d['certificates']['s_y'] == 5 and d['certificates']['hx24'] == 5 and {k: d['certificates'][k]['passed'] for k in ('S_P', 'S_X')} == dict(S_P=5, S_X=5) and d['certificates']['S_P']['file'].startswith('taper_check_')
    assert d['spread']['seg35'] is not None and d['spread']['seg35']['low'] <= decision['delta_seg35']['mean'] <= d['spread']['seg35']['high'] and d['bounds']['s_y_final'] == pytest.approx(1e-4) and d['time']['wall'] == 5 and d['time']['n_entries'] == 5
    assert [r['t'] for r in d['same_t_s_x']][-1] == CURVE.FINAL and [r['t'] for r in d['same_t_s_p']][-1] == CURVE.FINAL and [r['t'] for r in d['against_entry']][-1] == CURVE.FINAL and d['against_entry'][-1]['seg35']['mean'] == pytest.approx(d['against_hx24']['seg35']['mean'])
    assert d['rejection']['s_y']['n'] == 5 and d['rejection']['s_y']['rejected_share'] == pytest.approx(200/500) and d['rejection']['s_x']['rejected_share'] == .375 and d['rejection']['windows']['s_y']['0–160 s'] == pytest.approx(3/8) and d['rejection']['windows']['s_p']['0–160 s'] == pytest.approx(4/8)
    assert [row['elapsed'] for row in d['native']] == [ELAPSED[-1]]*5 and d['native'][0]['s_x_elapsed'] == 172.
    assert d['means']['Hx24']['seg35'] == pytest.approx(.8617) and d['means']['S_P']['seg4'] == pytest.approx(.8984) and d['means']['S_X']['hd95'] == pytest.approx(1.68)
    for word in ('seg35', 'seg4', '皮层', '白质', '小六个', '大六个', '其余 23 个', 'HD95', '官方 SDlogJ', '脑内 SDlogJ', '被拒比例', '首次试探后的秒数', '同 t', '0439-0440', '它证伪不了什么', '.88 的线不作门', '预算收尾', '开发集', 'S_Y 终态 − S_P 终态', 'S_Y 终态 − S_X 终态', '都是另一时段', '不是 B 线收敛', '选配用过', '从 4 个新臂里取的', '选择效应', '进方法先改合同', '评测单元从场算', '〔估算', '下一步不预先写', '其余 14 对', 'S_Y(t) − Hx24 修复后', 't = 210、270', 'S_X 终态 − Hx@2400 修复后', 'S_X(t) − Hx@2400', '平均 167.5 s', '5 对的修复撞了 120 s 的墙', '少记约 120 s', 'S_X 的持时 152 s', '去掉 0439-0440（0.9）', 'σ_d', '95% t 区间', '第一次提交下界', '120 s 墙版修复', 'RW', 'unet35_v1_ens'):
        assert word in summary, word
    assert {row['arm'] for row in map(dict, __import__('csv').DictReader(open(out/'long_curve.csv')))} == {'S_Y', 'S_X_s2_v1', 'S_P_tc_v1'}
    RUN.write_analysis(world.root, ev=world.ev, results=world.results)
    assert len(list(out.glob('previous_*'))) == 1 and (out/'summary_sy.md').is_file()


def test_the_reading_stands_on_the_real_steps(world):
    assert start_queue(world, SYMachine(world)) == 0
    decision = json_load(world.root/'analysis/decision_sy.json')
    deltas = [json_load(CORE.run_dir(world.root, 'S_Y', p)/'metrics.json')['metrics']['seg35']-.8617 for p in PAIRS]
    assert decision['delta_seg35']['mean'] == pytest.approx(float(np.mean(deltas))) and decision['delta_seg35']['positive'] == sum(d > 0 for d in deltas) and decision['reading_code'] == DEC.label_of(float(np.mean(deltas)), sum(d > 0 for d in deltas))


def test_a_certificate_that_fails_stops_the_queue_before_any_entry_and_any_run(world):
    machine = SYMachine(world, passes=False)
    assert start_queue(world, machine) == 8
    assert 'the certificate did not pass' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == [] and not (world.root/'entries').exists()


def test_a_missing_reference_stops_before_the_check(world):
    (world.results/'tc_v1/runs/S_P'/PAIRS[2]/'curve.json').unlink()
    (CORE.s_x_dir(PAIRS[3], world.results)/'curve.json').unlink()
    machine = SYMachine(world)
    assert start_queue(world, machine) == 8
    stop = json_load(world.root/'queue_stop.json')['reason']
    assert 'no curve.json' in stop and "S_X 0441-0442: no ['curve.json']" in stop and machine.taper_checks == 0 and machine.launched == []


def test_a_frozen_or_input_problem_or_a_broken_pin_stops_before_the_check(world):
    machine = SYMachine(world)
    machine.frozen, machine.inputs = ['frozen made up'], ['inputs made up']
    d = CORE.repair_dir(PAIRS[4], world.results)
    (d/'repair.json').write_text((d/'repair.json').read_text()+'\n')
    assert start_queue(world, machine) == 8
    stop = json_load(world.root/'queue_stop.json')['reason']
    assert 'frozen made up' in stop and 'inputs made up' in stop and 'the pin' in stop and machine.taper_checks == 0 and machine.launched == []


@pytest.mark.parametrize('attribute', ['bad_raw', 'bad_alpha', 'bad_state'])
def test_an_entry_not_injected_as_recorded_stops_and_keeps_the_run(world, attribute):
    machine = SYMachine(world)
    getattr(machine, attribute).add(UNITS[0])
    assert start_queue(world, machine) == 8
    assert machine.launched == UNITS[:1] and 'not injected as recorded' in json_load(world.root/'queue_stop.json')['reason'] and machine.certified == []
    run = CORE.run_dir(world.root, 'S_Y', PAIRS[0])
    assert list((run/TASK).glob('committed_*')) and not (run/'queue_done.json').exists() and not (run/'cleanup.json').exists()
    assert json_load(world.root/'analysis/decision_sy.json')['reading_code'] == DEC.NOT_DECIDABLE


def test_a_final_map_the_evaluation_does_not_certify_stops_and_keeps_the_run(world):
    """The solver's record says it passed; the field says otherwise: a person looks.  The commits stay."""
    machine = SYMachine(world, final_passes=False)
    assert start_queue(world, machine) == 8
    assert machine.launched == UNITS[:1] and 'computed from the final map did not pass' in json_load(world.root/'queue_stop.json')['reason']
    run = CORE.run_dir(world.root, 'S_Y', PAIRS[0])
    assert list((run/TASK).glob('committed_*')) and not (run/'cleanup.json').exists() and not (run/'curve.json').exists() and json_load(run/'metrics.json')['certificate']['passed'] is True
    decision = json_load(world.root/'analysis/decision_sy.json')
    assert decision['reading_code'] == DEC.NOT_DECIDABLE and '评测单元从场算的证书未通过' in decision['reading']


def test_a_failed_certificate_process_stops_and_keeps_the_commits(world):
    machine = SYMachine(world)
    machine.fail_certify.add(PAIRS[0])
    assert start_queue(world, machine) == 8
    run = CORE.run_dir(world.root, 'S_Y', PAIRS[0])
    assert list((run/TASK).glob('committed_*')) and 'from its field failed' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == UNITS[:1]


def test_a_failed_curve_stops_and_keeps_the_commits(world):
    machine = SYMachine(world)
    machine.fail_curve.add(PAIRS[0])
    assert start_queue(world, machine) == 8
    run = CORE.run_dir(world.root, 'S_Y', PAIRS[0])
    assert list((run/TASK).glob('committed_*')) and 'curve failed' in json_load(world.root/'queue_stop.json')['reason'] and (run/'metrics.json').is_file()


def test_a_failed_prepare_check_stops(world):
    machine = SYMachine(world)
    machine.fail_check = True
    assert start_queue(world, machine) == 8
    assert 'differs from the configuration' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == []


@pytest.mark.parametrize('cls', ['method_failure', 'handover_only'])
def test_a_run_that_does_not_end_ok_stops_the_queue_and_stays_whole(world, cls):
    """失败即停 (the brief), read literally: S2 went on after a method failure; here the reading already cannot be decided and a person looks.  A later start stops at the same place and reruns nothing."""
    machine = SYMachine(world)
    machine.classes[UNITS[1]] = cls
    if cls == 'handover_only':
        machine.empty.add(UNITS[1])
    assert start_queue(world, machine) == 8
    assert machine.launched == UNITS[:2]
    stop = json_load(world.root/'queue_stop.json')['reason']
    assert f'0439-0440 ended {cls}' in stop and 'stops the queue' in stop
    run = CORE.run_dir(world.root, 'S_Y', PAIRS[1])
    assert list((run/TASK).glob('level_*')) and not (run/'cleanup.json').exists() and json_load(run/'queue_done.json')['state'] == cls
    decision = json_load(world.root/'analysis/decision_sy.json')
    assert decision['reading_code'] == DEC.NOT_DECIDABLE and '0439-0440' in decision['reading']
    again = SYMachine(world)
    done = tree(world.root/'runs')
    assert start_queue(world, again) == 8
    assert again.launched == [] and tree(world.root/'runs') == done and f'0439-0440 ended {cls}' in json_load(world.root/'queue_stop.json')['reason']


def test_an_infrastructure_failure_stops_the_queue_too(world):
    machine = SYMachine(world)
    machine.infrastructure.add(UNITS[0])
    assert start_queue(world, machine) == 8
    assert machine.launched == UNITS[:1] and '0438-0439 ended infrastructure' in json_load(world.root/'queue_stop.json')['reason'] and json_load(CORE.run_dir(world.root, 'S_Y', PAIRS[0])/'queue_done.json')['state'] == 'infrastructure'


def test_a_curve_that_fails_on_a_handover_only_run_changes_nothing_of_the_stop(world):
    machine = SYMachine(world)
    machine.classes[UNITS[0]] = 'handover_only'
    machine.empty.add(UNITS[0])
    machine.fail_curve.add(PAIRS[0])
    assert start_queue(world, machine) == 8
    assert 'handover_only, no curve' in (world.root/'queue.log').read_text() and '0438-0439 ended handover_only' in json_load(world.root/'queue_stop.json')['reason']


def test_a_finished_run_is_not_rerun(world):
    assert start_queue(world, SYMachine(world)) == 0
    again = SYMachine(world)
    done = tree(world.root/'runs')
    assert start_queue(world, again) == 0
    assert again.launched == [] and tree(world.root/'runs') == done and len(json_load(world.root/'batch.json')['starts']) == 2


def test_an_incomplete_attempt_stops_and_stays(world):
    run = CORE.run_dir(world.root, 'S_Y', PAIRS[0])
    run.mkdir(parents=True)
    (run/'something').write_text('x')
    machine = SYMachine(world)
    assert start_queue(world, machine) == 8
    assert 'incomplete attempt' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == [] and (run/'something').is_file()


def test_a_second_driver_is_refused(world):
    world.root.mkdir(parents=True)
    import fcntl
    held = open(world.root/'queue.lock', 'w')
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert start_queue(world, SYMachine(world)) == 2
    finally:
        held.close()


def test_the_batch_pins_the_readings_code_and_the_source_table(world):
    """A changed decide_sy.py or sy_core.py is a STOP at the next start and is written in the summary of a later `decide`."""
    assert start_queue(world, SYMachine(world)) == 0
    batch = json_load(world.root/'batch.json')
    assert batch['decide_sha256'] == sha(DEC.__file__) and batch['pins_sha256'] == sha(CORE.__file__) and batch['hold'] == 47. and batch['source'] == 'Hx24@600'
    assert json_load(world.root/'analysis/decision_sy.json')['decide_code']['same'] is True
    write_json(world.root/'batch.json', dict(batch, decide_sha256='f'*64))
    result = RUN.write_analysis(world.root, ev=world.ev, results=world.results)
    assert result['decide_code'] == dict(at_first_start='f'*64, now=sha(DEC.__file__), code_changed=[], same=False) and '判定代码或它依赖的代码与队列第一次启动时不同' in (world.root/'analysis/summary_sy.md').read_text()
    machine = SYMachine(world)
    assert start_queue(world, machine) == 8
    assert 'batch.json differs' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == []


def test_the_real_system_calls_its_own_script(world, monkeypatch):
    """The real class with subprocess replaced: nothing starts."""
    monkeypatch.undo()
    seen = []

    def fake_run(argv, **kwargs):
        seen.append((argv, kwargs.get('env')))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(RUN.subprocess, 'run', fake_run)
    system = RUN.System()
    run = world.base/'neutral_root/runs/S_Y'/PAIRS[0]
    run.mkdir(parents=True)
    assert system.evaluate(run, 'S_Y', PAIRS[0], 'rid') == 0 and system.curve(run, 'S_Y', PAIRS[0], 'rid') == 0 and system.certify(run, 'S_Y', PAIRS[0], 'rid') == 0
    assert system.taper_check(world.base/'neutral_root/out.json', world.base/'log.txt') == 0
    (a, _), (b, _), (c, env), (d, env_check) = seen
    assert a[2].endswith('experiments/sy/run_sy.py') and a[3] == 'evaluate-one' and b[3] == 'curve-one' and c[3] == 'certify-one' and d[3] == 'taper-check' and d[2].endswith('experiments/sy/run_sy.py')
    for e in (env, env_check):
        assert e['CUDA_VISIBLE_DEVICES'] == '' and e['NUMBA_NUM_THREADS'] == str(RUN.CERTIFICATE_THREADS) and e['JD_CURL_JIT_CACHE'] == str(world.base/'neutral_root/cache/numba_jit')


def test_nothing_here_starts_a_unit():
    """Only `launch` of the machine and the printed strings have systemd-run; the steps do not."""
    source = (ROOT/'experiments/sy/sy_core.py').read_text()
    assert 'systemd-run' not in source and 'subprocess' not in source
    assert 'systemd-run' not in (ROOT/'experiments/sy/decide_sy.py').read_text()
    assert shlex.join(RUN.queue_launch(Path('/x/sy_v1'))).startswith('systemd-run --user --unit=tag-sy-queue --collect')


def test_s2s_files_are_only_imported():
    """Nothing here assigns to s2_core, decide_s2, run_s2: S2's and S_X's readings cannot move through this package."""
    for name in ('sy_core.py', 'run_sy.py', 'decide_sy.py'):
        tree_ = ast.parse((ROOT/'experiments/sy'/name).read_text())
        for node in ast.walk(tree_):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    assert not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id in ('s2', 'rs2', 'ds2')), name


# ---- the dry run ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def fake_prepare(tokens, cwd, env):
    """As the frozen prepare leaves a run directory, for what the dry run reads afterwards."""
    run = Path(tokens[tokens.index('prepare')+1])
    run.mkdir(parents=True)
    write_json(run/'protocol.json', dict(task=dict(name=TASK, pair=run.name), stage=tokens[tokens.index('--stage')+1]))
    return SimpleNamespace(returncode=0, stdout='', stderr='')


@pytest.fixture
def dry_world(world, monkeypatch):
    frozen = SEED.frozen_dir(world.results)
    (frozen/'data/three_d').mkdir(parents=True)
    (frozen/'data/three_d/stage_L2R_val_v1').symlink_to(world.stage)

    def frozen_copy(target, results=None):
        (target/'scripts').mkdir(parents=True)
        return 235
    monkeypatch.setattr(SEED, 'frozen_copy', frozen_copy)
    return world


def dry(world, machine=None, **changes):
    lines = []
    code = RUN.dry_run(out=lines.append, results=world.results, repo=ROOT, system=SYMachine(world) if machine is None else machine, ev=world.ev, pins={}, prepare=fake_prepare, torch_counts=False, **changes)
    return code, lines


def test_the_dry_run_passes_and_writes_nothing_outside_its_scratch(dry_world, monkeypatch):
    world = dry_world
    opened = []
    real = np.load

    def spy(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', spy)

    def no_truth(*args, **kwargs):
        raise AssertionError('the dry run read the truth labels')
    monkeypatch.setattr(world.ev, 'truth_labels', no_truth)
    monkeypatch.setitem(sys.modules, 'torch', None)      # an import of torch in the dry run would raise here (this test's process may already hold it)
    before = tree(world.results)
    code, lines = dry(world)
    text = '\n'.join(lines)
    assert code == 0 and 'problems: none' in lines[-1] and tree(world.results) == before and not world.root.exists()
    assert 'pair.npz' not in opened and set(opened) == {CORE.FIELD_FILE, 'repair.npz', 'field.npz'}
    assert all(f'S_Y {CORE.short_name(p)}: raw ' in text for p in PAIRS) and text.count('in the band 0 (moved 0)') == 5 and 'gate, as it stands now' in text and 'open' in text and 'code none' in text


def test_the_dry_run_reports_a_node_in_the_band_and_does_not_go_round_it(dry_world):
    world = dry_world
    in_band(world, pair=PAIRS[3])
    code, lines = dry(world)
    text = '\n'.join(lines)
    assert code == 8 and 'S_Y 0441-0442: BAND:' in text and "'n_nodes_in_band': 1" in text and 'problems: ' in lines[-1] and 'S_Y 0441-0442: BAND:' in lines[-1] and 'none' not in lines[-1] and '# item 4' not in text and '# item 5' in text


def test_the_dry_run_says_the_gate_is_closed_behind_the_training(dry_world):
    machine = SYMachine(dry_world)
    machine.unit_list, machine.gpu = ['tag-nn-train.service'], 'compute pids [1]'
    code, lines = dry(dry_world, machine)
    assert code == 0 and "closed: other units: ['tag-nn-train.service']" in lines and machine.launched == [] and machine.taper_checks == 0 and machine.certified == []


def test_the_dry_run_refuses_a_scratch_that_exists(dry_world, neutral):
    with pytest.raises(SystemExit, match='exists'):
        RUN.dry_run(scratch=neutral, results=dry_world.results, repo=ROOT, system=SYMachine(dry_world), ev=dry_world.ev, pins={}, prepare=fake_prepare, torch_counts=False)


def test_the_dry_run_checks_the_rehearsal_against_the_documents(dry_world):
    with pytest.raises(SystemExit, match='the documents say'):
        RUN.dry_run(out=lambda text: None, results=dry_world.results, repo=ROOT, system=SYMachine(dry_world), ev=dry_world.ev, pins=dict(s_p=(.03, .0059)), prepare=fake_prepare, torch_counts=False)


def test_the_tools_do_not_import_torch(neutral):
    code = f"import sys\nsys.path.insert(0, {str(ROOT/'experiments/sy')!r})\nimport run_sy\nprint('torch' in sys.modules, 'jd_curl' in sys.modules)\n"
    done = subprocess.run([PY, '-B', '-c', code], env={'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, capture_output=True, text=True, cwd=neutral)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ['False', 'False']


@pytest.mark.parametrize('argv,word', [(['certify-one', '/x', '--arm', 'S_X', '--pair', PAIRS[0], '--run-id', 'x'], 'no arm'), (['evaluate-one', '/x', '--arm', 'S_Y', '--pair', 'x', '--run-id', 'x'], 'S_Y runs')])
def test_the_commands_refuse_bad_arguments_before_any_file(argv, word):
    with pytest.raises(SystemExit, match=word):
        RUN.main(argv)


def test_there_is_no_option_for_another_source_or_another_hold():
    for words in (['queue', '--hold', '60'], ['queue', '--sx-arm', 'S'], ['taper-check', '--arms', 'S_F']):
        with pytest.raises(SystemExit):
            RUN.main(words)


def test_the_sha_helper_is_the_bytes_of_the_array():
    u = raw_field(3)
    assert CORE.array_sha(u) == hashlib.sha256(u.tobytes()).hexdigest()


# ---- the review of the tools (docs/bayesreg/reviews/SY_tools_复核_20261105.md) ----------------------------------------------------------------------------------------------------------
def test_a_stop_after_the_fifth_run_is_not_decidable_and_a_later_start_only_post_processes(world):
    """Important 1: the fifth pair's curve fails after its metrics and its certificate are written; the summary of the STOP must not give a line.  A later start does the steps after the run again, starts no unit, and then reads."""
    machine = SYMachine(world)
    machine.fail_curve.add(PAIRS[4])
    assert start_queue(world, machine) == 8 and machine.launched == UNITS
    run = CORE.run_dir(world.root, 'S_Y', PAIRS[4])
    assert (run/'metrics.json').is_file() and (run/'field_certificate.json').is_file() and not (run/'queue_done.json').exists() and list((run/TASK).glob('committed_*'))
    decision = json_load(world.root/'analysis/decision_sy.json')
    assert decision['reading_code'] == DEC.NOT_DECIDABLE and 'S_Y 0442-0443: 运行没有以 ok 结束（queue_done None）' in decision['reading']
    again = SYMachine(world)
    assert start_queue(world, again) == 0
    assert again.launched == [] and again.certified == [PAIRS[4]] and json_load(run/'queue_done.json')['state'] == 'ok' and (run/'curve.json').is_file() and not list((run/TASK).glob('committed_*'))
    assert json_load(world.root/'analysis/decision_sy.json')['reading_code'] != DEC.NOT_DECIDABLE


def test_a_final_map_that_does_not_pass_stops_again_on_a_later_start_and_its_record_is_kept(world):
    """Minor 13, 2: the later start certifies the same field again, stops again, starts no unit; the first record that did not pass stays under its time."""
    assert start_queue(world, SYMachine(world, final_passes=False)) == 8
    run = CORE.run_dir(world.root, 'S_Y', PAIRS[0])
    first = json_load(run/'field_certificate.json')
    again = SYMachine(world, final_passes=False)
    assert start_queue(world, again) == 8
    assert again.launched == [] and again.certified == [PAIRS[0]] and 'computed from the final map did not pass' in json_load(world.root/'queue_stop.json')['reason']
    kept = list(run.glob('field_certificate_*.json'))
    assert len(kept) == 1 and json_load(kept[0]) == first and json_load(run/'field_certificate.json')['passed'] is False


def test_a_record_that_passed_is_rewritten_and_one_that_did_not_is_kept(neutral):
    path = neutral/'field_certificate.json'
    assert RUN.keep_if_not_passed(path, lambda doc: doc.get('passed') is True) is None
    write_json(path, dict(passed=True))
    assert RUN.keep_if_not_passed(path, lambda doc: doc.get('passed') is True) is None and path.is_file()
    write_json(path, dict(passed=False))
    first = RUN.keep_if_not_passed(path, lambda doc: doc.get('passed') is True)
    write_json(path, dict(passed=False, n=2))
    second = RUN.keep_if_not_passed(path, lambda doc: doc.get('passed') is True)
    assert first != second and json_load(first) == dict(passed=False) and json_load(second) == dict(passed=False, n=2) and not path.exists()


def test_a_metrics_record_that_did_not_pass_is_kept_on_a_later_start(world):
    machine = SYMachine(world)
    machine.bad_raw.add(UNITS[0])
    assert start_queue(world, machine) == 8
    run = CORE.run_dir(world.root, 'S_Y', PAIRS[0])
    first = json_load(run/'metrics.json')
    again = SYMachine(world)
    assert start_queue(world, again) == 8 and again.launched == []
    kept = list(run.glob('metrics_*.json'))
    assert len(kept) == 1 and json_load(kept[0]) == first and first['seed']['passed'] is False


def test_a_known_failure_is_said_before_the_gate_and_the_certificates(world):
    """Minor 1: the second start stops at once, even with the gate closed; it checks nothing and certifies nothing."""
    machine = SYMachine(world)
    machine.classes[UNITS[1]] = 'method_failure'
    assert start_queue(world, machine) == 8
    again = SYMachine(world)
    again.load = 5.
    assert start_queue(world, again) == 8
    assert again.clock == 0. and again.taper_checks == 0 and again.launched == [] and '0439-0440 ended method_failure' in json_load(world.root/'queue_stop.json')['reason'] and len(list(world.root.glob('taper_check_*'))) == 1
    assert len(list(world.root.glob('queue_stop_*.json'))) == 1


def test_a_final_map_without_u_has_no_certificate(world):
    run, task = finished_run(world)
    np.savez(task/'final_native.npz', v=np.zeros(3))
    assert RUN.certify_one(options_of(run), results=world.results, certifier=world.certifier()) == 8 and not (run/'field_certificate.json').exists()


def test_the_certificate_says_whether_the_grid_and_the_version_are_the_solvers(world):
    run, task = finished_run(world)
    doc = json_load(task/'final_native.json')
    doc['certificate'].update(grid_id='grid', certificate_version='v')
    write_json(task/'final_native.json', doc)
    assert RUN.certify_one(options_of(run), results=world.results, certifier=world.certifier()) == 0
    record = json_load(run/'field_certificate.json')
    assert record['same_grid_as_the_solver'] is True and record['same_version_as_the_solver'] is True
    doc['certificate'].update(grid_id='another')
    write_json(task/'final_native.json', doc)
    RUN.certify_one(options_of(run), results=world.results, certifier=world.certifier())
    assert json_load(run/'field_certificate.json')['same_grid_as_the_solver'] is False


def test_the_latest_taper_check_gives_the_references(neutral):
    write_json(neutral/'taper_check_20261105T180000Z.json', dict(references=[dict(arm='S_P', passed=False), dict(arm='S_X', passed=True)]))
    write_json(neutral/'taper_check_20261105T190000Z.json', dict(references=[dict(arm='S_P', passed=True), dict(arm='S_X', passed=True)]))
    got = DEC.reference_certificates(neutral)
    assert got['S_P'] == dict(n=1, passed=1, file='taper_check_20261105T190000Z.json') and got['S_X']['passed'] == 1
    assert DEC.reference_certificates(neutral/'none')['S_P'] == dict(n=0, passed=0, file=None)


def test_a_change_of_the_code_the_reading_goes_through_stops_a_later_start(world):
    """Important 2: batch.json pins run_sy.py and the files the runs and the reading import; the summary of a later `decide` names them."""
    assert start_queue(world, SYMachine(world)) == 0
    batch = json_load(world.root/'batch.json')
    assert set(batch['code_sha256']) == set(DEC.CODE_FILES) and batch['code_sha256']['experiments/sy/run_sy.py'] == sha(RUN.__file__) and batch['code_sha256']['experiments/d0/evaluate.py'] == sha(world.ev.__file__)
    assert {'experiments/sy/run_sy.py', 'experiments/s2/decide_s2.py', 'experiments/s2/s2_core.py', 'experiments/seed/inject_run.py', 'experiments/d0/classify_run.py', 'experiments/d0/check_prepared.py', 'experiments/tc/tc_curve.py'} <= set(DEC.CODE_FILES)
    write_json(world.root/'batch.json', dict(batch, code_sha256=dict(batch['code_sha256'], **{'experiments/d0/evaluate.py': 'f'*64})))
    result = RUN.write_analysis(world.root, ev=world.ev, results=world.results)
    assert result['decide_code']['same'] is False and result['decide_code']['code_changed'] == ['experiments/d0/evaluate.py'] and 'experiments/d0/evaluate.py' in (world.root/'analysis/summary_sy.md').read_text()
    machine = SYMachine(world)
    assert start_queue(world, machine) == 8
    stop = json_load(world.root/'queue_stop.json')['reason']
    assert 'batch.json differs' in stop and "'code_sha256': ['experiments/d0/evaluate.py']" in stop and machine.launched == []


def test_the_reading_by_hand_checks_the_pins(world):
    """Minor 6: `decide` reads fd2_v1's repair_metrics.json; a record changed after the pins is refused, nothing written."""
    assert start_queue(world, SYMachine(world)) == 0
    before = tree(world.root/'analysis')
    d = CORE.repair_dir(PAIRS[2], world.results)
    (d/'repair_metrics.json').write_text((d/'repair_metrics.json').read_text()+'\n')
    with pytest.raises(SystemExit, match='the pin'):
        RUN.write_analysis(world.root, ev=world.ev, results=world.results)
    assert tree(world.root/'analysis') == before


def test_a_reference_without_a_finite_value_is_not_decidable():
    """Minor 5: not a traceback."""
    runs = {(CORE.ARM, p): fake_run(.87) for p in PAIRS}
    hx24 = fake_hx24()
    hx24[PAIRS[3]]['metrics'] = dict(hx24[PAIRS[3]]['metrics'], seg35=None)
    result = DEC.decide(runs, hx24, PAIRS, short, GROUPS)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and 'Hx24 0441-0442: seg35、seg4 不是有限数' in result['reading']


@pytest.mark.parametrize('change,word', [
    (dict(self_check=False), 'evaluate.py 的自检'),
    (dict(pair='OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1'), 'metrics.json 记的是'),
    (dict(injected=49.), '实测持时 49.00 s'),
])
def test_the_self_checks_and_the_measured_hold_are_anomalies(change, word):
    """Minors 3, 4: described, not premises."""
    runs = {(CORE.ARM, p): fake_run(.87, pair=p) for p in PAIRS}
    assert not [a for a in DEC.anomalies(runs, CORE.plan_runs(), short) if 'curve' not in a['what']]
    runs[(CORE.ARM, PAIRS[0])] = fake_run(.87, **dict(dict(pair=PAIRS[0]), **change))
    found = [a for a in DEC.anomalies(runs, CORE.plan_runs(), short) if word in a['what']]
    assert [a['pair'] for a in found] == ['0438-0439'] and DEC.decide(runs, fake_hx24(), PAIRS, short, GROUPS)['reading_code'] != DEC.NOT_DECIDABLE


def test_the_labels_and_the_grid_of_the_records_are_anomalies():
    runs = {(CORE.ARM, p): fake_run(.87, pair=p) for p in PAIRS}
    runs[(CORE.ARM, PAIRS[1])]['metrics']['run_labels_equal_truth'] = False
    runs[(CORE.ARM, PAIRS[2])]['field_certificate'].update(same_grid_as_the_solver=False)
    whats = {a['pair']: a['what'] for a in DEC.anomalies(runs, CORE.plan_runs(), short) if 'curve' not in a['what']}
    assert 'run_labels_equal_truth' in whats['0439-0440'] and '网格或版本' in whats['0440-0441']


def test_the_spread_of_delta():
    row = dict(per_pair={p: v for p, v in zip(PAIRS, (.0063, -.0023, .0027, .0049, -.0022))}, mean=float(np.mean((.0063, -.0023, .0027, .0049, -.0022))))
    got = DEC.spread(row)
    assert got['sd'] == pytest.approx(.00399, abs=5e-5) and got['mde'] == pytest.approx(1.66*got['sd']) and got['low'] < 0 < got['high'] and DEC.spread(dict(row, mean=None)) is None
    assert DEC.spread(dict(per_pair=dict(list(row['per_pair'].items())[:4]), mean=.001)) is None


def test_the_summary_holds_no_number_written_before_the_runs():
    """Minor 10: the notes are made from the records."""
    source = (ROOT/'experiments/sy/decide_sy.py').read_text()
    for text in ('约 277', '约 172', '约 144', '约 97', '约 105', '152 s', 'α = .9', '终态都是预算收尾', '3 对的修复'):
        assert text not in source, text
