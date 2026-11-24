"""S2 (runs/20261105-s2/任务书.md, plan.md): experiments/s2/; CPU only, small synthetic volumes and made-up records; no solver, no GPU, no unit.

- raw' and the second taper: bit for bit against the frozen function; a repair node in the band is refused;
- the guards of an entry: sha256, shape, the pair, only from the named root; S_X's pins and its hold;
- the reading: every branch, its bars, its premises; only S_F decides;
- the gate; the plan's cells; the start strings; no truth key on the registration path;
- the queue on a fake machine, with the real evaluation, curve, cleanup and reading on synthetic fields: nothing written outside s2_v1.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/s2'))
sys.path.insert(1, str(ROOT/'experiments/seed'))
sys.path.insert(2, str(ROOT/'experiments/tc'))

import decide_s2 as DEC  # noqa: E402
import inject_run as INJ  # noqa: E402
import run_s2 as S2  # noqa: E402
import run_seed as SEED  # noqa: E402
import run_seed19 as S19  # noqa: E402
import s2_core as CORE  # noqa: E402
import tc_curve as CURVE  # noqa: E402

from .test_mc0p_tools import FROZEN, PY, needs_frozen  # noqa: E402
from .test_seed_tools import FakeSystem, predicted_sources, write_json  # noqa: E402

SHAPE = (12, 13, 14)                                     # (8, 9, 10) of the other tests has no node with w = 1
PAIRS = CORE.PAIRS
ELAPSED = (8., 14., 15., 29., 31., 61., 89.5, 90., 121., 151., 211., 269., 275.)
RF, RL = 'RF05_fine', 'RL05_long'
GROUPS = dict(small_six=(18, 19, 23, 34, 35, 4), large_six=(1, 2, 5, 20, 21, 24), other_23=tuple(k for k in range(1, 36) if k not in (18, 19, 23, 34, 35, 4, 1, 2, 5, 20, 21, 24)))


class NoRealMachine:
    def __init__(self, *args, **kwargs):
        raise AssertionError('a queue on the real machine inside a test: a guard has regressed')


@pytest.fixture(autouse=True)
def no_real_machine(monkeypatch):
    """Every queue here runs on a fake machine, passed in.  A guard that regresses must fail the test, never start a real queue."""
    for module in (S2, SEED, S19):
        monkeypatch.setattr(module, 'System', NoRealMachine)


@pytest.fixture
def neutral():
    """A temporary directory whose path does not hold the word of the test runner: check_launch would refuse the start strings under tmp_path, as it should."""
    with tempfile.TemporaryDirectory(prefix='s2_tools_') as d:
        yield Path(d)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(root):
    return {str(p): sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def json_load(path):
    return json.loads(Path(path).read_text())


# ---- fields ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def raw_field(seed=1):
    """Zero faces, nonzero in the band: its taper differs from it."""
    rng = np.random.default_rng(seed)
    u = np.zeros(SHAPE+(3,))
    u[1:-1, 1:-1, 1:-1] = rng.normal(size=(SHAPE[0]-2, SHAPE[1]-2, SHAPE[2]-2, 3))*.1
    return u


def interior_nodes(seed=1, n=9):
    """Sorted node ids with w = 1."""
    rng = np.random.default_rng(100+seed)
    z, y, x = np.indices(SHAPE)
    ok = np.minimum.reduce([np.minimum(c, s-1-c) for c, s in zip((z, y, x), SHAPE)]) >= CORE.TAPER
    return np.sort(rng.choice(np.flatnonzero(ok.ravel()), size=n, replace=False))


def band_node(distance=2):
    return int(np.ravel_multi_index((distance, 6, 7), SHAPE))


def moved_values(raw, nodes, seed=1, unmoved=2):
    """The repair's values at these nodes: the tapered field plus a small change, the last `unmoved` nodes as they were (a free node need not move)."""
    rng = np.random.default_rng(200+seed)
    start = CORE.tapered(raw).reshape(-1, 3)[nodes]
    values = start+rng.normal(size=start.shape)*.01
    if unmoved:
        values[-unmoved:] = start[-unmoved:]
    return values


def repaired_of(raw, nodes, values):
    v = CORE.tapered(raw)
    v.reshape(-1, 3)[nodes] = values
    return v


def commit_field(seed, k):
    rng = np.random.default_rng(seed)
    u = np.zeros(SHAPE+(3,))
    u[1:-1, 1:-1, 1:-1] = rng.normal(size=(SHAPE[0]-2, SHAPE[1]-2, SHAPE[2]-2, 3))*.02
    return u*.5*k


# ---- a synthetic world ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def labels_of(index):
    z, y, x = np.indices(SHAPE)
    block = ((z//3)*9+(y//3)*3+x//4+1) % 36
    return np.roll(block, index, axis=2).astype(np.int16)


def write_pair(d, pair, index, extra=None):
    fixed, moving = labels_of(index), labels_of(index+1)
    rng = np.random.default_rng(index)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d/'pair.npz', fixed=rng.uniform(.1, .7, SHAPE), moving=rng.uniform(.1, .7, SHAPE), fixed_seg35=fixed, moving_seg35=moving, fixed_seg4=(fixed % 5).astype(np.int16), moving_seg4=(moving % 5).astype(np.int16))
    a, b = pair.split('__')
    write_json(d/'manifest.json', dict(pair=pair, pair_sha256=sha(d/'pair.npz'), fixed_subject=a, moving_subject=b, shape_zyx=list(SHAPE), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], **(extra or {})))


def fake_metrics(seg35=.84, seg4=.89, sdlogj=.08, cortex=.85, small=.59):
    per35 = {str(k): (small if k in GROUPS['small_six'] else seg35) for k in range(1, 36)}
    return dict(seg35=seg35, seg4=seg4, sdlogj_official=sdlogj, sdlogj_brain=.45, seg4_per_label={'1': cortex, '2': .9, '3': .91, '4': .88}, seg35_per_label=per35)


def task_records(task, n=3):
    write_json(task/'summary.json', dict(status='native_budget_exhausted', native_commits=n, costs=dict(levels=[dict(level=3, attempts=600, inner_accepted=375, rejected_reasons=dict(JACOBIAN_VIOLATION=225))])))
    write_json(task/'history.json', [dict(level=3, outer_accepted=True, elapsed_seconds=float(10*k), inner_accepted_steps=5) for k in range(1, n+1)])


class World:
    """base/results: A5P's inputs, ef_v1's repairs and their sources, S_P's records.  base/stage: the truth.  ev: the real evaluate.py pointed at this stage."""

    def __init__(self, base, monkeypatch):
        self.base = Path(base)
        self.results = self.base/'results'
        self.stage = self.base/'stage'
        self.root = self.results/CORE.ROOT_NAME
        self.ev = SEED.load_evaluate()
        monkeypatch.setattr(self.ev, 'STAGE', self.stage)
        inputs = self.results/SEED.ARMS['S_P']['inputs']
        files = {}
        self.pairs = PAIRS+(CORE.EXCEPTION_PAIR,)
        for index, pair in enumerate(self.pairs):
            write_pair(self.stage/pair, pair, index)
            d = inputs/'stage_L2R_val_v1'/pair
            write_pair(d, pair, index, dict(preprocessing_d0=dict(arm='A5P', oracle=False, source_pair_sha256='orig'+pair, source_manifest_sha256='origm'+pair, recipe='synthetic')))
            files[f'stage_L2R_val_v1/{pair}/pair.npz'] = sha(d/'pair.npz')
            files[f'stage_L2R_val_v1/{pair}/manifest.json'] = sha(d/'manifest.json')
        write_json(inputs/'d0_inputs.json', dict(arm='A5P', oracle=False, files_sha256=files))
        write_json(self.stage/'selection.json', dict(pairs=[dict(pair=p) for p in self.pairs]))       # C23.6-2 revision 6h: d0/evaluate admits only the stage's registered pairs
        self.raw, self.nodes, self.values = {}, {}, {}
        for arm in CORE.QUEUE_ARMS:
            for pair in CORE.ARMS[arm]['pairs']:
                self.make_repair(arm, pair)
        for pair in self.pairs:
            self.make_reference(pair)

    def seed_of(self, arm, pair):
        return self.pairs.index(pair)+(0 if arm == 'S_F' else 50)

    def make_repair(self, arm, pair, *, nodes=None, values=None, **changes):
        """The source run and the repair of ef_v1 for one cell; `changes`: into repair.json."""
        seed = self.seed_of(arm, pair)
        raw = raw_field(seed)
        nodes = interior_nodes(seed) if nodes is None else nodes
        values = moved_values(raw, nodes, seed) if values is None else values
        self.raw[(arm, pair)], self.nodes[(arm, pair)], self.values[(arm, pair)] = raw, nodes, values
        spec = CORE.ARMS[arm]
        short = CORE.short_name(pair)
        s, d = CORE.source_dir(arm, pair, self.results), CORE.repair_dir(arm, pair, self.results)
        s.mkdir(parents=True, exist_ok=True)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(s/'field.npz', u=raw)
        a, b = pair.split('__')
        write_json(s/'run.json', dict(schema=CORE.SOURCE_SCHEMA, run_id=f"mc0p_Q2_{spec['tier']}_{short}", pair=pair, arm='Q2', tier=spec['tier'], k=35, oracle=False, status='done', seconds_total=44., channel_sources=predicted_sources(), field=dict(file='field.npz', sha256=sha(s/'field.npz'), u_sha256=CORE.array_sha(raw)),
                                      data=dict(pair_sha256='orig'+pair, manifest_sha256='origm'+pair, fixed_subject=a, moving_subject=b)))
        write_json(s/'metrics.json', dict(metrics=fake_metrics(.82, .85, small=.5), at_accepted_alpha=dict(alpha=.9, seg35=.80, seg4=.84), certificate=dict(accepted_alpha=.9)))
        np.savez(d/'repair.npz', nodes=nodes, values=values)
        repaired = CORE.array_sha(repaired_of(raw, nodes, values))
        record = dict(schema=CORE.REPAIR_SCHEMA, run_id=f"ef_{spec['ef_arm']}_{spec['tier']}_{short}", arm=spec['ef_arm'], tier=spec['tier'], pair=pair, oracle=False, status='certified', passed=True, eps=.05, source=dict(directory=str(s), run_id=f"mc0p_Q2_{spec['tier']}_{short}", field_sha256=sha(s/'field.npz'), tapered_u_sha256=CORE.array_sha(CORE.tapered(raw))),
                      certificate=dict(status='certified_feasible', min_jd_lower_bound=1e-4, n_infeasible=0, n_uncertain=0, state_hash=repaired), attempt=dict(file='repair.npz', sha256=sha(d/'repair.npz')), seconds_repair=13., seconds_warmup=12.4)
        record.update(changes)
        write_json(d/'repair.json', record)
        write_json(d/'repair_metrics.json', dict(schema=CORE.REPAIR_METRICS_SCHEMA, run_id=record['run_id'], pair=pair, oracle=False, fallback=None, repaired_u_sha256=repaired, after=dict(metrics=fake_metrics(.84, .88, small=.58), certificate=dict(accepted_alpha=1., status_at_1='certified_feasible', n_infeasible_at_1=0, n_uncertain_at_1=0))))

    def make_reference(self, pair):
        """S_P's records, as seed_v1, seed_v1b, tc_v1 hold them; S_Pp's source field is S_P's entry on its pairs."""
        run = CORE.reference_dir(pair, self.results)
        task = run/'pg_weak__four__330s'
        task.mkdir(parents=True)
        write_json(run/'protocol.json', dict(task=dict(name=task.name, pair=pair), stage='SEED-S_P'))
        write_json(run/'metrics.json', dict(metrics=fake_metrics(.8430, .8984), certificate=dict(passed=True), seed=dict(accepted_alpha=.9 if pair in CORE.S_PP_PAIRS else 1.)))
        field = CORE.source_dir('S_Pp', pair, self.results)/'field.npz' if pair in CORE.S_PP_PAIRS else self.results/'mc0p_v1/runs/Q2_long'/CORE.short_name(pair)/'field.npz'
        if pair not in CORE.S_PP_PAIRS:
            field.parent.mkdir(parents=True)
            write_json(field.parent/'metrics.json', dict(metrics=fake_metrics(.82, .85, small=.5), at_accepted_alpha=dict(alpha=1., seg35=.82, seg4=.85), certificate=dict(accepted_alpha=1.)))
            np.savez(field, u=raw_field(77))
        write_json(run/'seed_entry.json', dict(field_path=str(field), field_sha256=sha(field), charge_seconds=47., pair=pair, oracle=False, shape_zyx=list(SHAPE), source_k=35))
        write_json(run/'cleanup.json', dict(deleted_bytes=12*2**30))
        task_records(task)
        write_json(task/'levels.json', [dict(event='level_prepared', transfer_recovery=dict(attempts=[dict(certificate=dict(grid_id='grid'))]))])
        if pair in PAIRS:
            again = self.results/'tc_v1/runs/S_P'/pair
            again.mkdir(parents=True)
            write_json(again/'curve.json', fake_curve(.84, offset=50.))
            write_json(again/'protocol.json', dict(task=dict(name=task.name)))
            (again/task.name).mkdir()
            task_records(again/task.name)

    def certifier(self, *, passes=True):
        def certify(seed, grid, config=None):
            assert config == 'configuration C' and seed.shape == SHAPE+(3,)
            return SimpleNamespace(status='certified_feasible' if passes else 'infeasible', reason='r', min_jd_lower_bound=1e-4 if passes else -.1, n_cells_checked=100, n_infeasible=0 if passes else 3, n_uncertain=0, state_hash=CORE.array_sha(seed), grid_id='grid', certificate_version='v')

        def frozen_taper(u, grid, width):
            return CORE.tapered(u, width), None

        def make_grid(**kw):
            return SimpleNamespace(**kw)
        return certify, 'configuration C', frozen_taper, make_grid

    def options(self, **changes):
        return SimpleNamespace(**dict(dict(root=str(self.root), wait_minutes=30, min_host_gib=15.), **changes))

    def files(self, pair):
        return CORE.input_files(pair, self.results)


@pytest.fixture
def world(neutral, monkeypatch):
    return World(neutral, monkeypatch)


def fake_point(t, seg35, offset=50., **changes):
    metrics = dict(seg35=seg35, seg4=.89, sdlogj_official=.08, sdlogj_brain=.45, small_six=.59, large_six=.87, other_23=.89, cortex=.85, white_matter=.9)
    return dict(dict(t=t, commit=1 if t == 0 else 2+t//30, elapsed_seconds=float(t), visible_seconds=float(t)+offset, D=1000., GF=1., min_jd_lower_bound=1e-4, certificate_passed=True, metrics=metrics), **changes)


def fake_curve(seg35, offset=50.):
    return dict(schema=CURVE.SCHEMA_CURVE, run_id='fake', complete=True, self_check=dict(passed=True), points=[fake_point(t, seg35, offset) for t in CURVE.TIMES], final=dict(fake_point(270, seg35, offset), t=CURVE.FINAL, elapsed_seconds=275., visible_seconds=275.+offset, is_last_commit=True), n_native_commits=13)


# ---- the fake machine -------------------------------------------------------------------------------------------------------------------------------------------------------------------
def write_task(run, *, elapsed=ELAPSED, bad_raw=False, alpha=1., state_hash=None):
    """A task directory as the frozen solver leaves it, before the cleanup."""
    run = Path(run)
    entry = json_load(run/'seed_entry.json')
    task = run/'pg_weak__four__330s'
    task.mkdir(parents=True, exist_ok=True)
    seed = int(entry['u_sha256'][:6], 16)
    commits, history = [], []
    for k, t in enumerate(elapsed, 1):
        u = commit_field(seed, k)
        name = f'committed_{k:04d}_L3.npz'
        np.savez(task/name, u=u)
        commits.append(dict(path=name, sha256=sha(task/name), level=3, native=True, role='formal_outer_commit', certificate=dict(status='certified_feasible', min_jd_lower_bound=1e-4, n_infeasible=0, n_uncertain=0, state_hash=CORE.array_sha(u)), solver_commit_seconds=t, visible_seconds=t+entry['charge_seconds']+3.))
        history.append(dict(level=3, outer_accepted=True, elapsed_seconds=t, D_candidate=2000.-10*k, F_candidate=2000.-10*k, GF_candidate=5.-.1*k, D_start=2000., F_start=2000., inner_accepted_steps=5))
    write_json(task/'commits.json', commits)
    write_json(task/'history.json', history)
    np.savez(task/'level_3_preoptimizer_done.npz', u=commit_field(seed, 0))
    np.savez(task/'level_3_level_prepared.npz', u=commit_field(seed, 0))
    repaired = entry['repaired_u_sha256'] if state_hash is None else state_hash
    attempts = [dict(alpha=1., certificate_passed=alpha == 1., certificate=dict(min_jd_lower_bound=1e-4 if alpha == 1. else -.1, n_infeasible=0 if alpha == 1. else 2, state_hash=repaired))]+([dict(alpha=alpha, certificate_passed=True, certificate=dict(min_jd_lower_bound=1e-3, n_infeasible=0, state_hash='scaled'))] if alpha != 1. else [])
    write_json(task/'levels.json', [dict(event='preoptimizer_done', level=3, path='level_3_preoptimizer_done.npz', sha256=sha(task/'level_3_preoptimizer_done.npz'), preoptimizer=dict(policy=INJ.POLICY, raw_hash='wrong' if bad_raw else entry['u_sha256'], seed_hash=entry['tapered_sha256'], taper_L=4, charge_seconds=entry['charge_seconds'])),
                                    dict(event='level_prepared', level=3, path='level_3_level_prepared.npz', sha256=sha(task/'level_3_level_prepared.npz'), prepare_seconds=49., transfer_recovery=dict(status='ready', accepted_alpha=alpha, attempts=attempts))])
    final = commit_field(seed, len(elapsed))
    np.savez(task/'final_native.npz', u=final, warped=np.zeros(SHAPE))
    write_json(task/'final_native.json', dict(path='final_native.npz', sha256=sha(task/'final_native.npz'), role='formal_native_outer_commit', certificate=dict(status='certified_feasible', min_jd_lower_bound=1e-4, n_infeasible=0, n_uncertain=0, state_hash=CORE.array_sha(final))))
    write_json(task/'summary.json', dict(status='native_budget_exhausted', success=False, metrics_role='formal_native_outer_commit', native_commits=len(elapsed), e2e_seconds=327., GF=1., GF_threshold=1e-5, costs=dict(levels=[dict(level=3, attempts=500, inner_accepted=300, rejected_reasons=dict(JACOBIAN_VIOLATION=200))])))
    write_json(run/'completion.json', dict(seconds=327.))
    return task


class S2Machine(FakeSystem):
    """A fake machine: `launch` writes what the frozen solver would write; evaluate, curve and the taper check are the real steps, in this process."""

    def __init__(self, world, *, passes=True):
        super().__init__()
        self.world, self.passes = world, passes
        self.bad_raw, self.bad_alpha, self.bad_state, self.fail_curve, self.infrastructure, self.empty = set(), set(), set(), set(), set(), set()
        self.taper_checks = []
        self.after_taper_check = None
        self.no_commits = set()

    def prepare(self, argv, target, log, results=None):
        run = Path(argv[argv.index('prepare')+1])
        arm, pair = run.parent.name, run.name
        run.mkdir(parents=True)
        assert argv[argv.index('--stage')+1] == CORE.stage_name(arm) and argv[argv.index('--pair')+1] == pair and Path(target).name == 'stage_L2R_val_v1'
        write_json(run/'protocol.json', dict(task=dict(name='pg_weak__four__330s', pair=pair), stage=CORE.stage_name(arm)))
        (run/'inputs'/pair).mkdir(parents=True)
        shutil.copyfile(self.world.stage/pair/'pair.npz', run/'inputs'/pair/'pair.npz')
        return 0

    def check_prepared(self, run, pair, stage, pair_sha, manifest_sha, entry):
        assert entry['seconds'] == 330 and entry['changes'] == SEED.CHANGES and Path(entry['inputs_root']).name == 'A5P' and stage == CORE.stage_name(Path(run).parent.name)
        return dict(ok=not self.fail_check, problems=['made up'] if self.fail_check else [])

    def launch(self, argv, log):
        unit = argv[2].split('=')[1]
        self.launched.append(unit)
        assert 'seed_inject.py' in argv[-1]
        run = Path(argv[argv.index('--wait')+2].split('=')[1]).parent
        if unit not in self.infrastructure:
            task = write_task(run, elapsed=() if unit in self.empty else ELAPSED, bad_raw=unit in self.bad_raw, alpha=.9 if unit in self.bad_alpha else 1., state_hash='another field' if unit in self.bad_state else None)
            if unit in self.no_commits:                  # the frozen path writes commits.json at each outer record: a run without one has no such file at all
                (task/'commits.json').unlink()
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
        return S2.evaluate_one(SimpleNamespace(run=str(run), arm=arm, pair=pair, run_id=run_id), ev=self.world.ev)

    def curve(self, run, arm, pair, run_id):
        if (arm, pair) in self.fail_curve:
            return 8
        return S2.curve_one(SimpleNamespace(run=str(run), arm=arm, pair=pair, run_id=run_id), ev=self.world.ev)

    def taper_check(self, out, sx, log_path, arms):
        self.taper_checks.append(list(arms))
        record = S2.taper_check([(a, p) for a in arms for p in CORE.ARMS[a]['pairs']], results=self.world.results, sx=sx, out=lambda text: None, certifier=self.world.certifier(passes=self.passes))
        write_json(out, record)
        if self.after_taper_check:
            self.after_taper_check()
        return 0 if record['passed'] else 8

    def uptime(self):
        return '100.0'


def start_queue(world, machine, **changes):
    return S2.queue(world.options(**changes), system=machine, results=world.results, repo=ROOT, ev=world.ev)


# ---- raw' and the second taper -----------------------------------------------------------------------------------------------------------------------------------------------------------
def prime_of(seed=1, nodes=None, values=None):
    raw = raw_field(seed)
    nodes = interior_nodes(seed) if nodes is None else nodes
    values = moved_values(raw, nodes, seed) if values is None else values
    repaired = repaired_of(raw, nodes, values)
    return raw, nodes, values, repaired, dict(tapered_sha256=CORE.array_sha(CORE.tapered(raw)), repaired_sha256=CORE.array_sha(repaired))


def test_the_taper_of_raw_prime_is_the_repaired_field_bit_for_bit():
    raw, nodes, values, repaired, hashes = prime_of()
    prime, record = CORE.raw_prime(raw, nodes, values, **hashes)
    assert CORE.tapered(prime).tobytes() == repaired.tobytes() and SEED.boundary_taper_sha256(prime, SHAPE) == hashes['repaired_sha256']
    assert record['bitwise_equal_to_repaired'] is True and record['taper_of_raw_prime_sha256'] == hashes['repaired_sha256'] and record['raw_prime_sha256'] == CORE.array_sha(prime)
    assert record['n_repair_nodes'] == 9 and record['n_changed_nodes'] == 7 and record['n_nodes_in_band'] == 0 and record['min_face_distance'] >= CORE.TAPER
    outside = np.setdiff1d(np.arange(int(np.prod(SHAPE))), nodes)                # only the repair's nodes differ from the raw field
    assert np.array_equal(prime.reshape(-1, 3)[outside], raw.reshape(-1, 3)[outside]) and np.array_equal(prime.reshape(-1, 3)[nodes], values)
    assert prime.flags.writeable is False


def test_injecting_the_repaired_field_as_it_is_would_not_be_the_certified_field():
    """Why raw' exists: the frozen path tapers what it is given; the repaired field's band would be w^2, and its hash another."""
    _, _, _, repaired, hashes = prime_of()
    again = CORE.tapered(repaired)
    assert CORE.array_sha(again) != hashes['repaired_sha256'] and CORE.array_sha(repaired) == hashes['repaired_sha256']
    w = CORE.taper_weights(SHAPE)
    assert np.array_equal(again[w == 1.], repaired[w == 1.]) and np.any(again[w < 1.] != repaired[w < 1.])


@pytest.mark.parametrize('distance', [1, 2, 3])
def test_a_repair_node_in_the_band_is_refused(distance):
    """The counterexample: the same construction with one node in the band is NOT the repaired field: the check must stop, not use it."""
    raw = raw_field()
    node = band_node(distance)
    nodes = np.sort(np.append(interior_nodes(), node))
    values = moved_values(raw, nodes, unmoved=0)
    repaired = repaired_of(raw, nodes, values)
    naive = np.array(raw, copy=True)
    naive.reshape(-1, 3)[nodes] = values
    assert CORE.array_sha(CORE.tapered(naive)) != CORE.array_sha(repaired)       # what using it would have meant
    with pytest.raises(CORE.BandError, match="1 repair nodes in the taper's band .1 of them moved") as refused:
        CORE.raw_prime(raw, nodes, values, tapered_sha256=CORE.array_sha(CORE.tapered(raw)), repaired_sha256=CORE.array_sha(repaired))
    assert refused.value.record['n_nodes_in_band'] == 1 and refused.value.record['min_face_distance'] == distance


def test_a_free_node_in_the_band_is_refused_even_when_it_did_not_move():
    raw = raw_field()
    nodes = np.sort(np.append(interior_nodes(), band_node()))
    values = moved_values(raw, nodes, unmoved=0)
    position = int(np.flatnonzero(nodes == band_node())[0])
    values[position] = CORE.tapered(raw).reshape(-1, 3)[band_node()]
    with pytest.raises(CORE.BandError, match='.0 of them moved') as refused:
        CORE.raw_prime(raw, nodes, values, tapered_sha256=CORE.array_sha(CORE.tapered(raw)), repaired_sha256=CORE.array_sha(repaired_of(raw, nodes, values)))
    assert refused.value.record['n_nodes_in_band'] == 1 and refused.value.record['n_changed_nodes_in_band'] == 0


def test_a_node_at_the_first_row_with_w_one_is_not_in_the_band():
    raw = raw_field()
    nodes = np.array([band_node(CORE.TAPER)])
    values = moved_values(raw, nodes, unmoved=0)
    _, record = CORE.raw_prime(raw, nodes, values, tapered_sha256=CORE.array_sha(CORE.tapered(raw)), repaired_sha256=CORE.array_sha(repaired_of(raw, nodes, values)))
    assert record['n_nodes_in_band'] == 0 and record['min_face_distance'] == CORE.TAPER


@pytest.mark.parametrize('name', ['tapered_sha256', 'repaired_sha256'])
def test_a_field_that_is_not_the_recorded_one_is_refused(name):
    raw, nodes, values, _, hashes = prime_of()
    hashes[name] = 'f'*64
    with pytest.raises(CORE.EntryError, match='the taper of the raw field|the repaired field'):
        CORE.raw_prime(raw, nodes, values, **hashes)


def test_a_raw_field_that_was_already_tapered_is_refused():
    """A checkpoint saved after the taper would be tapered twice: its hash is not the one the repair started from."""
    raw, nodes, values, _, hashes = prime_of()
    with pytest.raises(CORE.EntryError, match='the taper of the raw field'):
        CORE.raw_prime(CORE.tapered(raw), nodes, values, **hashes)


@needs_frozen
def test_the_arithmetic_is_the_frozen_functions(neutral):
    """The frozen boundary_taper itself on raw': the repaired field's bytes, and the same weights.  In a process of its own: the frozen tree is not on this sys.path."""
    raw, nodes, values, repaired, hashes = prime_of()
    prime, _ = CORE.raw_prime(raw, nodes, values, **hashes)
    np.save(neutral/'prime.npy', prime)
    code = ('import hashlib, numpy as np\nfrom types import SimpleNamespace\nfrom jd_curl.three_d.preopt import boundary_taper\n'
            f"u = np.load({str(neutral/'prime.npy')!r})\nseed, w = boundary_taper(u, SimpleNamespace(shape=u.shape[:3]), 4)\n"
            f"np.save({str(neutral/'w.npy')!r}, w)\nprint(hashlib.sha256(seed.tobytes()).hexdigest())\n")
    done = subprocess.run([PY, '-B', '-c', code], env={'PYTHONPATH': str(FROZEN), 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, capture_output=True, text=True, cwd=neutral)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == hashes['repaired_sha256'] == CORE.array_sha(repaired)
    assert np.load(neutral/'w.npy').tobytes() == CORE.taper_weights(SHAPE).tobytes()


def test_the_sources_are_ef_cores():
    sys.path.insert(0, str(ROOT/'experiments/ef'))
    sys.path.insert(1, str(ROOT/'experiments/mc0'))
    import ef_core
    assert CORE.SOURCES == ef_core.SOURCES and CORE.EXCEPTION_PAIR == ef_core.EXCEPTION_PAIR and CORE.S_PP_PAIRS == ef_core.LONG_REPAIR_PAIRS and CORE.EPS == ef_core.REPAIR_ARMS['RF05']['eps'] == ef_core.REPAIR_ARMS['RL05']['eps']


def test_the_certificate_configuration_is_the_main_lines():
    """frozen_certifier builds it with the frozen tree; the source says configuration C in the words of run_mc0.certificate_config."""
    want = 'PrescreenCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=65536)'
    assert want in (ROOT/'experiments/s2/run_s2.py').read_text() and want in (ROOT/'experiments/mc0/run_mc0.py').read_text()


# ---- the guards of an entry -------------------------------------------------------------------------------------------------------------------------------------------------------------
def source_error(world, arm='S_F', pair=PAIRS[0]):
    with pytest.raises(CORE.EntryError) as refused:
        CORE.construct(arm, pair, world.files(pair)[3], world.results)
    return str(refused.value)


def test_a_cell_gives_its_entry(world):
    for arm, pair in CORE.plan_runs():
        prime, source, record = CORE.construct(arm, pair, world.files(pair)[3], world.results)
        assert CORE.array_sha(CORE.tapered(prime)) == source['repaired_sha256'] and record['n_nodes_in_band'] == 0 and source['charge_seconds'] == 47.


def test_a_changed_raw_field_is_refused(world):
    np.savez(CORE.source_dir('S_F', PAIRS[0], world.results)/'field.npz', u=raw_field(9))
    assert 'sha256' in source_error(world)


def test_a_changed_repair_file_is_refused(world):
    d = CORE.repair_dir('S_F', PAIRS[0], world.results)
    with np.load(d/'repair.npz') as z:
        nodes, values = np.array(z['nodes']), np.array(z['values'])
    np.savez(d/'repair.npz', nodes=nodes, values=values+1e-9)
    assert 'sha256' in source_error(world)


def repin_repair(world, nodes, values, arm='S_F', pair=PAIRS[0]):
    """A repair file the records agree with, for what the file itself must say."""
    d = CORE.repair_dir(arm, pair, world.results)
    np.savez(d/'repair.npz', nodes=nodes, values=values)
    doc = json_load(d/'repair.json')
    doc['attempt']['sha256'] = sha(d/'repair.npz')
    write_json(d/'repair.json', doc)


def test_a_wrong_shape_is_refused(world):
    d = CORE.source_dir('S_F', PAIRS[0], world.results)
    np.savez(d/'field.npz', u=np.zeros((12, 13, 15, 3)))
    doc = json_load(CORE.repair_dir('S_F', PAIRS[0], world.results)/'repair.json')
    doc['source']['field_sha256'] = sha(d/'field.npz')
    write_json(CORE.repair_dir('S_F', PAIRS[0], world.results)/'repair.json', doc)
    run = json_load(d/'run.json')
    run['field']['sha256'] = sha(d/'field.npz')
    write_json(d/'run.json', run)
    assert 'shape' in source_error(world)


@pytest.mark.parametrize('nodes,values,word', [
    (np.array([5, 3], dtype=np.int64), np.zeros((2, 3)), 'sorted'),
    (np.array([3, 3], dtype=np.int64), np.zeros((2, 3)), 'sorted'),
    (np.array([int(np.prod(SHAPE))], dtype=np.int64), np.zeros((1, 3)), 'sorted'),
    (np.array([3, 4], dtype=np.int64), np.zeros((3, 3)), 'values'),
    (np.array([3., 4.]), np.zeros((2, 3)), 'nodes'),
    (np.array([3], dtype=np.int64), np.array([[np.nan, 0., 0.]]), 'finite'),
])
def test_a_repair_file_that_is_not_one_is_refused(world, nodes, values, word):
    repin_repair(world, nodes, values)
    assert word in source_error(world)


def test_a_repair_file_with_other_members_is_refused(world):
    d = CORE.repair_dir('S_F', PAIRS[0], world.results)
    np.savez(d/'repair.npz', nodes=np.array([3]), values=np.zeros((1, 3)), extra=np.zeros(1))
    doc = json_load(d/'repair.json')
    doc['attempt']['sha256'] = sha(d/'repair.npz')
    write_json(d/'repair.json', doc)
    assert 'members' in source_error(world)


@pytest.mark.parametrize('key,value,word', [
    ('pair', PAIRS[1], 'pair'),
    ('arm', 'RF01', 'arm'),
    ('tier', 'long', 'tier'),
    ('eps', .01, 'eps'),
    ('oracle', True, 'oracle'),
    ('passed', False, 'passed'),
    ('status', 'failed_back_to_frozen_rule', 'status'),
    ('run_id', 'ef_RF05_fine_0439-0440', 'run_id'),
    ('certificate', dict(status='infeasible', min_jd_lower_bound=-.1, n_infeasible=3, n_uncertain=0, state_hash='x'), 'did not pass'),
])
def test_a_repair_record_of_something_else_is_refused(world, key, value, word):
    d = CORE.repair_dir('S_F', PAIRS[0], world.results)
    doc = json_load(d/'repair.json')
    doc[key] = value
    write_json(d/'repair.json', doc)
    assert word in source_error(world)


def test_a_repair_of_another_pair_is_refused_even_with_its_files_moved_here(world):
    """The files of 0439-0440 in the directory of 0438-0439: every record says the other pair, and the arrays hash to another field."""
    here, there = CORE.repair_dir('S_F', PAIRS[0], world.results), CORE.repair_dir('S_F', PAIRS[1], world.results)
    for name in ('repair.json', 'repair.npz', 'repair_metrics.json'):
        shutil.copyfile(there/name, here/name)
    assert 'pair' in source_error(world) and 'run_id' in source_error(world)


@pytest.mark.parametrize('key,value', [('pair', PAIRS[1]), ('arm', 'Q0'), ('tier', 'long'), ('k', 1), ('oracle', True), ('status', 'failed'), ('run_id', 'mc0p_Q2_fine_0439-0440')])
def test_a_source_run_of_something_else_is_refused(world, key, value):
    d = CORE.source_dir('S_F', PAIRS[0], world.results)
    doc = json_load(d/'run.json')
    doc[key] = value
    write_json(d/'run.json', doc)
    assert f'source run.json {key}' in source_error(world)


def test_a_source_with_truth_channels_is_refused(world):
    d = CORE.source_dir('S_F', PAIRS[0], world.results)
    doc = json_load(d/'run.json')
    doc['channel_sources'] = predicted_sources(truth_labels_read=True)
    write_json(d/'run.json', doc)
    assert 'truth_labels_read' in source_error(world)


def test_a_source_of_other_images_is_refused(world):
    d = CORE.source_dir('S_F', PAIRS[0], world.results)
    doc = json_load(d/'run.json')
    doc['data']['pair_sha256'] = 'another'
    write_json(d/'run.json', doc)
    assert 'original pair' in source_error(world)


def test_the_entry_comes_only_from_the_named_root(world):
    """repair.json names a source elsewhere, with a good copy of the field there: refused before it is opened."""
    d = CORE.repair_dir('S_F', PAIRS[0], world.results)
    doc = json_load(d/'repair.json')
    elsewhere = world.base/'elsewhere'
    shutil.copytree(CORE.source_dir('S_F', PAIRS[0], world.results), elsewhere)
    doc['source']['directory'] = str(elsewhere)
    write_json(d/'repair.json', doc)
    assert 'may only come from' in source_error(world)


def test_the_repair_metrics_must_say_alpha_one_and_the_same_field(world):
    d = CORE.repair_dir('S_F', PAIRS[0], world.results)
    doc = json_load(d/'repair_metrics.json')
    doc['after']['certificate']['accepted_alpha'] = .9
    write_json(d/'repair_metrics.json', doc)
    assert 'alpha = 1' in source_error(world)
    doc['after']['certificate']['accepted_alpha'] = 1.
    doc['repaired_u_sha256'] = 'f'*64
    write_json(d/'repair_metrics.json', doc)
    assert 'repaired hash' in source_error(world)


def test_a_missing_file_is_missing(world):
    (CORE.repair_dir('S_F', PAIRS[0], world.results)/'repair_metrics.json').unlink()
    with pytest.raises(CORE.EntryMissing):
        CORE.construct('S_F', PAIRS[0], world.files(PAIRS[0])[3], world.results)


def test_the_entry_file_is_what_the_wrapper_takes_and_the_frozen_path_tapers_to_the_repaired_field(world):
    entry = CORE.write_entry(world.root, 'S_F', PAIRS[0], world.files(PAIRS[0])[3], world.results)
    assert set(INJ.ENTRY_KEYS) <= set(entry) and Path(entry['field_path']).parent == world.root/'entries/S_F/0438-0439' and entry['charge_seconds'] == 47. and entry['oracle'] is False
    u = INJ.load_field(entry)
    assert CORE.array_sha(u) == entry['u_sha256'] and SEED.boundary_taper_sha256(u, SHAPE) == entry['tapered_sha256'] == entry['repaired_u_sha256']
    held = []
    tick = iter(range(100))
    injected = INJ.make_injected(entry, u, clock=lambda: next(tick), wait=held.append)
    got, record = injected(None, None, SimpleNamespace(shape=SHAPE))
    assert CORE.array_sha(got) == entry['u_sha256'] and record['charge_seconds'] == 47. and record['policy'] == INJ.POLICY and held and held[0] <= 47.


def test_the_entry_is_written_once_and_never_overwritten(world):
    pair = PAIRS[0]
    first = CORE.write_entry(world.root, 'S_F', pair, world.files(pair)[3], world.results)
    before = tree(world.root)
    assert CORE.write_entry(world.root, 'S_F', pair, world.files(pair)[3], world.results) == first and tree(world.root) == before
    field, _ = CORE.entry_paths(world.root, 'S_F', pair)
    np.savez(field, u=raw_field(5))
    kept = tree(world.root)
    with pytest.raises(CORE.EntryError, match='nothing overwritten'):
        CORE.write_entry(world.root, 'S_F', pair, world.files(pair)[3], world.results)
    assert tree(world.root) == kept


def test_nothing_of_a_refused_entry_is_written(world):
    raw = world.raw[('S_F', PAIRS[0])]
    nodes = np.sort(np.append(interior_nodes(), band_node()))
    world.make_repair('S_F', PAIRS[0], nodes=nodes, values=moved_values(raw, nodes, unmoved=0))
    with pytest.raises(CORE.BandError):
        CORE.write_entry(world.root, 'S_F', PAIRS[0], world.files(PAIRS[0])[3], world.results)
    assert not world.root.exists()


# ---- the check, with a certificate that is told -------------------------------------------------------------------------------------------------------------------------------------------
def checked(world, cells=None, **kw):
    lines = []
    record = S2.taper_check(CORE.plan_runs() if cells is None else cells, results=world.results, out=lines.append, certifier=world.certifier(**{k: kw.pop(k) for k in list(kw) if k == 'passes'}), **kw)
    return record, lines


def test_the_check_passes_every_cell_and_writes_nothing(world):
    before = tree(world.results)
    record, lines = checked(world)
    assert record['passed'] is True and len(record['rows']) == 7 and record['n_nodes_in_band'] == 0 and record['band_refusals'] == [] and tree(world.results) == before
    assert all(r['frozen_taper_is_repaired'] and r['certificate']['state_hash'] == r['repaired_u_sha256'] for r in record['rows'])
    assert all('in the band 0' in line and 'passed True' in line for line in lines)


def test_the_check_stops_on_a_node_in_the_band_and_reports_all_the_cells(world):
    raw = world.raw[('S_F', PAIRS[2])]
    nodes = np.sort(np.append(interior_nodes(), band_node()))
    world.make_repair('S_F', PAIRS[2], nodes=nodes, values=moved_values(raw, nodes, unmoved=0))
    record, lines = checked(world)
    assert record['passed'] is False and record['band_refusals'] == ['0440-0441'] and record['n_nodes_in_band'] == 1 and len(record['rows']) == 7 and sum('REFUSED (band)' in line for line in lines) == 1


def test_the_check_fails_a_field_the_certificate_does_not_pass(world):
    record, _ = checked(world, passes=False)
    assert record['passed'] is False and all('the certificate did not pass' in ' '.join(r['problems']) for r in record['rows'])


def test_the_check_fails_when_a_zero_bound_is_all_there_is(world):
    certify, config, taper, grid = world.certifier()

    def zero(seed, g, config=None):
        return SimpleNamespace(**dict(vars(certify(seed, g, config=config)), min_jd_lower_bound=0.))
    record = S2.taper_check(CORE.plan_runs(), results=world.results, out=lambda t: None, certifier=(zero, config, taper, grid))
    assert record['passed'] is False


def test_the_check_fails_another_state_hash_and_another_grid(world):
    certify, config, taper, grid = world.certifier()

    def other(seed, g, config=None):
        return SimpleNamespace(**dict(vars(certify(seed, g, config=config)), state_hash='another'))

    def other_grid(seed, g, config=None):
        return SimpleNamespace(**dict(vars(certify(seed, g, config=config)), grid_id='another'))
    assert 'state hash' in ' '.join(S2.taper_check(CORE.plan_runs()[:1], results=world.results, out=lambda t: None, certifier=(other, config, taper, grid))['rows'][0]['problems'])
    assert 'grid id' in ' '.join(S2.taper_check(CORE.plan_runs()[:1], results=world.results, out=lambda t: None, certifier=(other_grid, config, taper, grid))['rows'][0]['problems'])


def test_s_pp_must_start_from_s_ps_own_field(world):
    """"Only alpha differs": a source that is not the field S_P was given is refused."""
    run = CORE.reference_dir(CORE.S_PP_PAIRS[0], world.results)
    doc = json_load(run/'seed_entry.json')
    doc['field_sha256'] = 'another'
    write_json(run/'seed_entry.json', doc)
    record, _ = checked(world, cells=[('S_Pp', CORE.S_PP_PAIRS[0])])
    assert record['passed'] is False and "S_P's entry on this pair" in ' '.join(record['rows'][0]['problems'])


def test_the_trace_needs_alpha_one_on_the_repaired_field(world):
    run = world.base/'run'
    run.mkdir()
    entry = CORE.write_entry(world.root, 'S_F', PAIRS[0], world.files(PAIRS[0])[3], world.results)
    write_json(run/'seed_entry.json', entry)
    write_json(run/'protocol.json', dict(task=dict(name='pg_weak__four__330s')))
    write_task(run)
    assert CORE.entry_trace(run)[1] == []
    assert CORE.entry_trace(run)[0]['passed'] is True and CORE.entry_trace(run)[0]['repaired_u_sha256'] == entry['repaired_u_sha256']
    write_task(run, alpha=.9)
    problems = CORE.entry_trace(run)[1]
    assert any('accepted alpha 0.9' in p for p in problems) and CORE.entry_trace(run)[0]['passed'] is False
    write_task(run, state_hash='another')
    assert any('first transfer attempt' in p for p in CORE.entry_trace(run)[1])
    write_task(run, bad_raw=True)
    assert any('raw_hash' in p for p in CORE.entry_trace(run)[1])


# ---- S_X -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def write_fl_records(world, pair, k, *, arm, checkpoint, seconds_task, seconds_repair, seconds_warmup=12.4):
    """One cell of fl_v1, in the format of experiments/fl/run_fl.py run_field, repair_one, evaluate_repair: read there, not guessed."""
    root = world.results/CORE.FL_ROOT
    short = CORE.short_name(pair)
    field, d = CORE.fl_field_dir(root, arm, pair), CORE.fl_repair_dir(root, arm, checkpoint, pair)
    field.mkdir(parents=True, exist_ok=True)
    d.mkdir(parents=True, exist_ok=True)
    raw = raw_field(300+k)
    nodes = interior_nodes(300+k)
    values = moved_values(raw, nodes, 300+k)
    name = f'field_{checkpoint:04d}.npz'
    np.savez(field/name, u=raw)
    np.savez(d/'repair.npz', nodes=nodes, values=values)
    a, b = pair.split('__')
    data = dict(pair_sha256='orig'+pair, manifest_sha256='origm'+pair, fixed_subject=a, moving_subject=b)
    write_json(field/'run.json', dict(schema=CORE.FL_RUN_SCHEMA, run_id=f'fl_{arm}_first5_{short}', arm=arm, tier='first5', pair=pair, k=CORE.SX_K.get(arm, 35), channels='onehot+intensity' if arm == 'I' else 'onehot', oracle=False, status='done', channel_sources=predicted_sources(), data=data,
                                      checkpoints={str(checkpoint): dict(step=checkpoint, file=name, sha256=sha(field/name), u_sha256=CORE.array_sha(raw), seconds_task=seconds_task)}))
    repaired = CORE.array_sha(repaired_of(raw, nodes, values))
    run_id = f'fl_R{arm}_c{checkpoint}_{short}'
    write_json(d/'repair.json', dict(schema=CORE.FL_REPAIR_SCHEMA, run_id=run_id, arm=f'R{arm}', tier=f'c{checkpoint}', pair=pair, oracle=False, status='certified', passed=True, reason='certified', eps=.05, data=data, source=dict(directory=str(field), file=name, run_id=f'fl_{arm}_first5_{short}', field_sha256=sha(field/name), tapered_u_sha256=CORE.array_sha(CORE.tapered(raw)), seconds_task=seconds_task),
                                     certificate=dict(status='certified_feasible', min_jd_lower_bound=1e-4, n_infeasible=0, n_uncertain=0, state_hash=repaired), attempt=dict(file='repair.npz', sha256=sha(d/'repair.npz'), n_free_nodes=len(nodes)), seconds_repair=seconds_repair, seconds_warmup=seconds_warmup))
    write_json(d/'repair_metrics.json', dict(schema=CORE.FL_REPAIR_METRICS_SCHEMA, run_id=run_id, arm=f'R{arm}', tier=f'c{checkpoint}', pair=pair, eps=.05, oracle=False, repaired_u_sha256=repaired, after=dict(metrics=fake_metrics(.85, .89, small=.6), certificate=dict(accepted_alpha=1., status_at_1='certified_feasible', n_infeasible_at_1=0, n_uncertain_at_1=0))))


def pin(world, arm, checkpoint, **changes):
    """The pins file as the generator writes it, and the five options."""
    root = world.results/CORE.FL_ROOT
    doc = CORE.sx_pins(root, arm, checkpoint)
    doc.update(changes)
    pins = world.base/'pins.json'
    write_json(pins, doc)
    return CORE.SXOptions(str(root), arm, checkpoint, str(pins), sha(pins))


def make_sx(world, arm='S', checkpoint=2400, tasks=(60.2, 61., 62.1, 70., 80.), repairs=(10., 10., 10., 10., 10.), **changes):
    for k, pair in enumerate(PAIRS):
        write_fl_records(world, pair, k, arm=arm, checkpoint=checkpoint, seconds_task=tasks[k], seconds_repair=repairs[k])
    return pin(world, arm, checkpoint, **changes)


def repin(options, change):
    doc = json_load(options.pins)
    change(doc)
    write_json(options.pins, doc)
    return CORE.SXOptions(options.root, options.arm, options.checkpoint, options.pins, sha(options.pins))


def fl_paths(world, options, pair=PAIRS[0]):
    root = world.results/CORE.FL_ROOT
    return CORE.fl_field_dir(root, options.arm, pair), CORE.fl_repair_dir(root, options.arm, options.checkpoint, pair)


def test_s_x_needs_all_its_options_or_none():
    assert CORE.sx_options() is None
    assert CORE.sx_options('r', 'S', 2400, 'p', 'h') == CORE.SXOptions('r', 'S', 2400, 'p', 'h')
    for missing in range(5):
        given = ['r', 'S', 2400, 'p', 'h']
        given[missing] = None
        with pytest.raises(SystemExit, match='all of'):
            CORE.sx_options(*given)


def test_s_x_is_only_there_when_asked_for():
    assert [a for a, _ in CORE.plan_runs()] == ['S_Pp']*2+['S_F']*5
    assert [a for a, _ in CORE.plan_runs(with_x=True)] == ['S_Pp']*2+['S_F']*5+['S_X']*5


def test_s_x_gives_its_entries_and_the_hold(world):
    options = make_sx(world)
    prime, source, record = CORE.construct('S_X', PAIRS[0], world.files(PAIRS[0])[3], world.results, options)
    assert CORE.array_sha(CORE.tapered(prime)) == source['repaired_sha256'] and record['n_nodes_in_band'] == 0
    assert source['charge_seconds'] == 73. and source['source_root'] == 'fl_v1' and source['source_tier'] == 'S@2400' and source['oracle'] is False and source['source_k'] == 35 and source['evaluation_lower_bound'] == 1e-4
    assert source['hold']['median_without_warmup'] == pytest.approx(72.1) and source['hold']['median_with_warmup'] == pytest.approx(84.5) and source['seconds_entry'] == 60.2 and source['seconds_repair'] == 10. and source['seconds_warmup'] == 12.4


def seconds_rows(tasks, repair=0., warmup=12.4):
    return {p: dict(seconds_task=s, seconds_repair=repair, seconds_warmup=warmup) for p, s in zip(PAIRS, tasks)}


@pytest.mark.parametrize('seconds,hold', [((10., 20., 30., 40., 50.), 47.), ((46., 46., 46.9, 47., 90.), 47.), ((47., 47., 47.1, 48., 90.), 48.), ((60.5, 61., 61.2, 200., 200.), 62.)])
def test_the_hold_is_the_rounded_up_median_and_not_below_47(seconds, hold):
    assert CORE.sx_hold(seconds_rows(seconds))['seconds'] == hold


def test_the_hold_counts_the_repair_and_not_the_warm_up():
    """seconds_task + seconds_repair, as S_F's 57-70 s; with the warm-up it is only reported."""
    record = CORE.sx_hold(seconds_rows((60., 60., 60., 60., 60.), repair=13.2, warmup=12.4))
    assert record['seconds'] == 74. and record['median_without_warmup'] == pytest.approx(73.2) and record['median_with_warmup'] == pytest.approx(85.6) and 'no warm-up' in record['rule'] and set(record['with_warmup']) == {CORE.short_name(p) for p in PAIRS}


def test_the_hold_is_at_most_200_seconds():
    """The lead, plan revision 1.  X at 4800 steps (about 273 s of task clock before its repair) is no entry of S_X; 299 s would have let a native level of 30 s through."""
    assert CORE.SX_MAX_HOLD == 200. and CORE.sx_hold(seconds_rows((199.5,)*5))['seconds'] == 200.
    for seconds in (200.1, 285., 299.):
        with pytest.raises(CORE.EntryError, match='t about 126 s'):
            CORE.sx_hold(seconds_rows((seconds,)*5))


def test_a_hold_above_the_line_stops_s_x_at_its_records(world):
    options = make_sx(world, arm='X', checkpoint=4800, tasks=(273.,)*5, repairs=(13.,)*5)
    with pytest.raises(CORE.EntryError, match='t about 126 s'):
        CORE.cell_source('S_X', PAIRS[0], world.files(PAIRS[0])[3], world.results, options)


def test_s_x_needs_its_options(world):
    with pytest.raises(CORE.EntryError, match='S_X needs'):
        CORE.cell_source('S_X', PAIRS[0], world.files(PAIRS[0])[3], world.results, None)


def test_the_pin_of_the_pins_file(world):
    options = make_sx(world)
    with pytest.raises(CORE.EntryError, match='sx-pins-sha256'):
        CORE.sx_rows(CORE.SXOptions(options.root, options.arm, options.checkpoint, options.pins, 'f'*64), world.results)
    assert set(CORE.sx_rows(options, world.results)) == set(PAIRS)
    Path(options.pins).write_text(Path(options.pins).read_text()+' ')
    with pytest.raises(CORE.EntryError, match='sx-pins-sha256'):
        CORE.sx_rows(options, world.results)


def test_s_x_only_from_fl_v1(world):
    options = make_sx(world)
    with pytest.raises(CORE.EntryError, match='may only come from'):
        CORE.sx_rows(CORE.SXOptions(str(world.results/'ef_v1'), options.arm, options.checkpoint, options.pins, options.pins_sha256), world.results)


def test_s_x_refuses_a_path_outside_the_root(world):
    options = make_sx(world)
    outside = world.base/'outside'/'0438-0439'
    outside.mkdir(parents=True)
    bad = repin(options, lambda doc: doc['pairs']['0438-0439'].update(raw_field=str(outside/'field.npz')))
    with pytest.raises(CORE.EntryError, match='may only come from'):
        CORE.sx_rows(bad, world.results)


def test_s_x_refuses_a_link_that_leaves_the_root(world):
    options = make_sx(world)
    outside = world.base/'outside'
    outside.mkdir()
    link = world.results/CORE.FL_ROOT/'runs'/'0438-0439'
    link.symlink_to(outside)
    bad = repin(options, lambda doc: doc['pairs']['0438-0439'].update(repair_npz=str(link/'repair.npz')))
    with pytest.raises(CORE.EntryError, match='may only come from'):
        CORE.sx_rows(bad, world.results)


def test_s_x_refuses_the_files_of_another_pair(world):
    options = make_sx(world)
    other = json_load(options.pins)['pairs']['0439-0440']['raw_field']
    bad = repin(options, lambda doc: doc['pairs']['0438-0439'].update(raw_field=other))
    with pytest.raises(CORE.EntryError, match='reads only from'):
        CORE.sx_rows(bad, world.results)


def test_s_x_refuses_a_swapped_pin(world):
    options = make_sx(world)
    doc = json_load(options.pins)['pairs']
    bad = repin(options, lambda d: d['pairs']['0438-0439'].update(raw_field_sha256=doc['0439-0440']['raw_field_sha256']))
    with pytest.raises(CORE.EntryError, match='sha256'):
        CORE.construct('S_X', PAIRS[0], world.files(PAIRS[0])[3], world.results, bad)


@pytest.mark.parametrize('arm', ['O', 'RF05', 'S_F', 'RF150'])
def test_s_x_refuses_the_oracle_arm_and_other_names(world, arm):
    options = CORE.SXOptions(str(world.results/CORE.FL_ROOT), arm, 2400, str(world.base/'pins.json'), 'f'*64)
    with pytest.raises(CORE.EntryError, match='--sx-arm'):
        CORE.sx_rows(options, world.results)
    with pytest.raises(CORE.EntryError, match='--sx-arm'):
        CORE.sx_pins(world.results/CORE.FL_ROOT, arm, 2400)


def test_arm_s_with_the_path_of_the_oracle_arm_is_refused(world):
    """The review's measurement: --sx-arm S, raw_field in O's directory, a pins file saying oracle false: the path decides, whatever the pins say."""
    options = make_sx(world)
    root = world.results/CORE.FL_ROOT
    other = root/'runs/O_2400/0438-0439'
    other.mkdir(parents=True)
    bad = repin(options, lambda doc: doc['pairs']['0438-0439'].update(raw_field=str(other/'field.npz')))
    with pytest.raises(CORE.EntryError, match='reads only from'):
        CORE.sx_rows(bad, world.results)


@pytest.mark.parametrize('key', ['raw_field', 'repair_npz'])
def test_a_path_through_oracle_isolated_is_refused(world, key):
    options = make_sx(world)
    root = world.results/CORE.FL_ROOT
    isolated = root/CORE.FL_ORACLE_DIR/'runs/O_first5/0438-0439/field_2400.npz'
    bad = repin(options, lambda doc: doc['pairs']['0438-0439'].update({key: str(isolated)}))
    with pytest.raises(CORE.EntryError, match='oracle arm'):
        CORE.sx_rows(bad, world.results)


def test_a_link_into_oracle_isolated_is_refused(world):
    """The arm's own directory name, a link to the quarantine: the resolved path is what counts."""
    options = make_sx(world)
    root = world.results/CORE.FL_ROOT
    field, _ = fl_paths(world, options)
    isolated = root/CORE.FL_ORACLE_DIR/'runs/O_first5/0438-0439'
    isolated.parent.mkdir(parents=True)
    shutil.move(field, isolated)
    field.symlink_to(isolated)
    with pytest.raises(CORE.EntryError, match='oracle arm'):
        CORE.sx_rows(options, world.results)


def changed_record(world, options, which, change):
    """One record of fl_v1 changed, the pins written again by the generator: only the record's own content can refuse it."""
    field, d = fl_paths(world, options)
    path = dict(run=field/'run.json', repair=d/'repair.json', metrics=d/'repair_metrics.json')[which]
    doc = json_load(path)
    change(doc)
    write_json(path, doc)
    return pin(world, options.arm, options.checkpoint)


@pytest.mark.parametrize('which,change,word', [
    ('run', lambda doc: doc.update(oracle=True), 'run.json oracle'),
    ('run', lambda doc: doc.update(channels='truth'), 'run.json channels'),
    ('run', lambda doc: doc.update(channel_sources=predicted_sources(truth_labels_read=True)), 'truth_labels_read'),
    ('run', lambda doc: doc.update(channel_sources=predicted_sources(oracle=True)), 'channel_sources oracle'),
    ('run', lambda doc: doc.pop('channel_sources'), 'no channel_sources'),
    ('run', lambda doc: doc.update(status='time_cap'), 'run.json status'),
    ('run', lambda doc: doc.update(pair=PAIRS[1]), 'run.json pair'),
    ('run', lambda doc: doc.update(arm='T'), 'run.json arm'),
    ('run', lambda doc: doc.update(tier='rest2400'), 'run.json tier'),
    ('run', lambda doc: doc.update(k=36), 'run.json k'),
    ('run', lambda doc: doc.update(schema='bayesreg.ef.run.v1'), 'run.json schema'),
    ('run', lambda doc: doc.update(checkpoints={}), 'seconds_task'),
    ('repair', lambda doc: doc.update(oracle=True), 'repair.json oracle'),
    ('repair', lambda doc: doc.update(status='failed_back_to_frozen_rule'), 'repair.json status'),
    ('repair', lambda doc: doc.update(passed=False), 'repair.json passed'),
    ('repair', lambda doc: doc.update(eps=.01), 'repair.json eps'),
    ('repair', lambda doc: doc.update(arm='RT'), 'repair.json arm'),
    ('repair', lambda doc: doc.update(tier='c4800'), 'repair.json tier'),
    ('repair', lambda doc: doc['source'].update(directory='/elsewhere'), 'source directory'),
    ('repair', lambda doc: doc['source'].update(seconds_task=1.), 'source seconds_task'),
    ('repair', lambda doc: doc['certificate'].update(n_infeasible=1), 'did not pass'),
    ('repair', lambda doc: doc.update(seconds_repair=float('nan')), 'seconds_repair'),
    ('repair', lambda doc: doc.update(seconds_repair=-1.), 'seconds_repair'),
    ('repair', lambda doc: doc.pop('seconds_warmup'), 'seconds_warmup'),
    ('metrics', lambda doc: doc.update(oracle=True), 'repair_metrics.json oracle'),
    ('metrics', lambda doc: doc.update(schema='x'), 'repair_metrics.json schema'),
    ('metrics', lambda doc: doc.update(repaired_u_sha256='f'*64), 'certificate state_hash'),
    ('metrics', lambda doc: doc['after']['certificate'].update(accepted_alpha=.9), 'alpha = 1'),
])
def test_fls_records_of_something_else_are_refused(world, which, change, word):
    bad = changed_record(world, make_sx(world), which, change)
    with pytest.raises(CORE.EntryError, match=word):
        CORE.sx_rows(bad, world.results)


def test_a_record_changed_after_the_pins_is_refused(world):
    options = make_sx(world)
    field, d = fl_paths(world, options)
    for path in (field/'run.json', d/'repair.json', d/'repair_metrics.json'):
        keep = path.read_text()
        path.write_text(keep+' ')
        with pytest.raises(CORE.EntryError, match='!= the pins'):
            CORE.sx_rows(options, world.results)
        path.write_text(keep)
    assert set(CORE.sx_rows(options, world.results)) == set(PAIRS)


@pytest.mark.parametrize('key', ['pair_sha256', 'fixed_subject'])
@pytest.mark.parametrize('which', ['run', 'repair'])
def test_s_xs_field_of_other_images_is_refused(world, which, key):
    bad = changed_record(world, make_sx(world), which, lambda doc: doc['data'].update({key: 'another'}))
    with pytest.raises(CORE.EntryError, match='original pair|subjects'):
        CORE.cell_source('S_X', PAIRS[0], world.files(PAIRS[0])[3], world.results, bad)


def test_the_oracle_mark_of_an_entry_comes_from_the_record(world):
    """No literal in the record of an entry: an S_X or S_F source whose record said anything but false never got here."""
    source = (ROOT/'experiments/s2/s2_core.py').read_text()
    assert "oracle=source['oracle']" in source and 'oracle=False, charge_seconds' not in source
    options = make_sx(world)
    entry = CORE.write_entry(world.root, 'S_X', PAIRS[0], world.files(PAIRS[0])[3], world.results, options)
    assert entry['oracle'] is False and entry['hold']['seconds'] == 73. and entry['seconds_warmup'] == 12.4 and entry['evaluation_lower_bound'] == 1e-4
    prime, picked, record = CORE.construct('S_X', PAIRS[0], world.files(PAIRS[0])[3], world.results, options)
    assert CORE.entry_record('S_X', PAIRS[0], CORE.entry_paths(world.root, 'S_X', PAIRS[0])[0], prime, dict(picked, oracle='from the record'), record, world.files(PAIRS[0])[3])['oracle'] == 'from the record'


def test_the_pins_generator_reads_fls_records_and_writes_nothing(world):
    options = make_sx(world)
    root = world.results/CORE.FL_ROOT
    before = tree(root)
    lines = []
    assert S2.print_pins(SimpleNamespace(sx_root=str(root), sx_arm='S', sx_checkpoint=2400), out=lines.append, results=world.results) == 0
    doc = json.loads(lines[0])
    assert tree(root) == before and doc == json_load(options.pins) and doc['oracle'] is False
    assert doc['pairs']['0438-0439']['raw_field'] == str(root/'runs/S_first5/0438-0439/field_2400.npz') and doc['pairs']['0438-0439']['repair_npz'] == str(root/'runs/RS_c2400/0438-0439/repair.npz') and 'seconds_task' not in doc['pairs']['0438-0439']
    with pytest.raises(SystemExit, match='S_X reads only'):
        S2.print_pins(SimpleNamespace(sx_root=str(world.results/'ef_v1'), sx_arm='S', sx_checkpoint=2400), out=lines.append, results=world.results)
    with pytest.raises(SystemExit, match='no \\['):
        S2.print_pins(SimpleNamespace(sx_root=str(root), sx_arm='T', sx_checkpoint=2400), out=lines.append, results=world.results)
    with pytest.raises(SystemExit, match='--sx-arm'):
        S2.print_pins(SimpleNamespace(sx_root=str(root), sx_arm='O', sx_checkpoint=2400), out=lines.append, results=world.results)


@pytest.mark.parametrize('step', [600, 1200, 2000])
def test_s_x_only_from_the_checkpoints_fl_repairs(world, step):
    good = make_sx(world)
    with pytest.raises(CORE.EntryError, match='--sx-checkpoint'):
        CORE.sx_rows(CORE.SXOptions(good.root, good.arm, step, good.pins, good.pins_sha256), world.results)


def test_s_x_refuses_oracle_marks_and_another_arm_in_the_pins(world):
    good = make_sx(world)
    with pytest.raises(CORE.EntryError, match='oracle'):
        CORE.sx_rows(pin(world, 'S', 2400, oracle=True), world.results)
    good = pin(world, 'S', 2400)
    with pytest.raises(CORE.EntryError, match='arm'):
        CORE.sx_rows(CORE.SXOptions(good.root, 'T', good.checkpoint, good.pins, good.pins_sha256), world.results)
    with pytest.raises(CORE.EntryError, match='checkpoint'):
        CORE.sx_rows(CORE.SXOptions(good.root, good.arm, 4800, good.pins, good.pins_sha256), world.results)


@pytest.mark.parametrize('change,word', [
    (lambda doc: doc['pairs'].pop('0438-0439'), 'exactly the first 5'),
    (lambda doc: doc['pairs'].update({'0455-0456': {}}), 'exactly the first 5'),
    (lambda doc: doc['pairs']['0438-0439'].pop('repaired_u_sha256'), 'missing'),
    (lambda doc: doc['pairs']['0438-0439'].pop('run_json_sha256'), 'missing'),
])
def test_s_x_refuses_an_incomplete_pins_file(world, change, word):
    with pytest.raises(CORE.EntryError, match=word):
        CORE.sx_rows(repin(make_sx(world), change), world.results)


def fl_source_dir():
    """experiments/fl once FL is merged; before that the review's worktree, by S2_FL_SOURCE."""
    here = os.environ.get('S2_FL_SOURCE') or ROOT/'experiments/fl'
    return Path(here) if (Path(here)/'fl_core.py').is_file() else None


@pytest.mark.skipif(fl_source_dir() is None, reason='FL is not merged yet (S2_FL_SOURCE names its worktree)')
def test_the_format_is_fls_own():
    """A tripwire for when FL is merged: what S_X assumes of FL's records, against FL's source."""
    d = fl_source_dir()
    core_text, run_text = (d/'fl_core.py').read_text(), (d/'run_fl.py').read_text()
    for word in (f"ORACLE_DIR = '{CORE.FL_ORACLE_DIR}'", f"FIRST = '{CORE.FL_FIRST}'", "REPAIR_STEPS = (2400, 4800)", "base/'runs'/f'{cell.arm}_{cell.tier}'/cell.short", "f'fl_{self.arm}_{self.tier}_{self.short}'"):
        assert word in core_text, word
    for name, k in CORE.SX_K.items():
        assert f"{name}=dict(k={k}," in core_text, name
    assert 'O=dict(k=35' in core_text and "channels='truth'" in core_text
    for word in (f"schema='{CORE.FL_RUN_SCHEMA}'", f"schema='{CORE.FL_REPAIR_SCHEMA}'", f"schema='{CORE.FL_REPAIR_METRICS_SCHEMA}'", "name = f'field_{done:04d}.npz'", "seconds_task=seconds_task", "checkpoints=keeper.records", "np.savez(out/'repair.npz', nodes=result.attempt_nodes, values=result.attempt_values)", "seconds_warmup=seconds_warmup_numba+seconds_warmup_device", "seconds_repair=seconds_repair", "attempt=dict(file='repair.npz', sha256=", "source=source", "field_sha256=sha, tapered_u_sha256=digest", "seconds_task=seconds_task)", "repaired_u_sha256=hashlib.sha256"):
        assert word in run_text, word


def test_s_x_stops_on_a_node_in_the_band(world):
    options = make_sx(world)
    d = Path(json_load(options.pins)['pairs']['0438-0439']['repair_npz'])
    raw = np.load(fl_paths(world, options)[0]/'field_2400.npz')['u']
    nodes = np.sort(np.append(interior_nodes(), band_node()))
    values = moved_values(raw, nodes, unmoved=0)
    np.savez(d, nodes=nodes, values=values)
    repaired = CORE.array_sha(repaired_of(raw, nodes, values))
    changed_record(world, options, 'repair', lambda doc: (doc['attempt'].update(sha256=sha(d)), doc['certificate'].update(state_hash=repaired)))
    bad = changed_record(world, options, 'metrics', lambda doc: doc.update(repaired_u_sha256=repaired))
    record = S2.taper_check([('S_X', p) for p in PAIRS], results=world.results, sx=bad, out=lambda t: None, certifier=world.certifier())
    assert record['passed'] is False and record['band_refusals'] == ['0438-0439']


# ---- the reading ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def fake_run(pair, seg35, *, seg4=.89, sdlogj=.08, alpha=1., hash_='h', state=None, cls='ok', passed=True, seed_passed=True, bound=True, npz=True, recorded=None):
    metrics = dict(metrics=fake_metrics(seg35, seg4, sdlogj), classification={'class': cls}, certificate=dict(passed=passed, state_hash_matches_u=bound, final_npz_sha256_matches=npz), seed=dict(passed=seed_passed, problems=[] if seed_passed else ['made up'], accepted_alpha=alpha, repaired_u_sha256=hash_, transfer_attempts=[dict(alpha=1., passed=True, state_hash=hash_ if state is None else state)]))
    return dict(metrics=metrics, repair_metrics=dict(repaired_u_sha256=hash_ if recorded is None else recorded))


def fake_references(seg35=.84, seg4=.89):
    return {pair: dict(metrics=dict(metrics=fake_metrics(seg35, seg4), certificate=dict(passed=True), seed=dict(accepted_alpha=1.))) for pair in PAIRS}


def short(pair):
    return CORE.short_name(pair)


def read(deltas, **changes):
    runs = {('S_F', pair): fake_run(pair, .84+d, **changes) for pair, d in zip(PAIRS, deltas)}
    return DEC.decide(runs, fake_references(), PAIRS, short, GROUPS, pin=None)


@pytest.mark.parametrize('deltas,code', [
    ((.003, .003, .003, .003, .003), DEC.KEPT),
    ((.01, .01, .01, .01, -.01), DEC.KEPT),                     # mean +.006, 4/5
    ((.002, .002, .002, .002, .002), DEC.KEPT),                 # the bar included
    ((.0019, .0019, .0019, .0019, .0019), DEC.MIDDLE),          # just under the bar
    ((.01, .01, .01, -.001, -.001), DEC.MIDDLE),                # mean +.0056, 3/5: not the first line, not <= 2/5
    ((.001, .001, .001, .001, .001), DEC.MIDDLE),               # small and 5/5
    ((.02, .02, -.001, -.001, -.001), DEC.NOT_VISIBLE),         # mean +.0034, 2/5
    ((0., 0., 0., 0., 0.), DEC.NOT_VISIBLE),                    # the bar included: Delta <= 0
    ((.001, .001, .001, .001, -.004), DEC.NOT_VISIBLE),         # mean 0, 4/5
    ((-.003, -.003, -.003, -.003, -.003), DEC.NOT_VISIBLE),
])
def test_the_table_of_the_reading(deltas, code):
    result = read(deltas)
    assert result['reading_code'] == code and result['reading'].startswith(DEC.READINGS[code]) and result['delta_seg35']['positive'] == sum(d > 1e-15 for d in deltas)


@pytest.mark.parametrize('delta,positive,code', [(.002, 4, DEC.KEPT), (.002, 3, DEC.MIDDLE), (.0019999, 5, DEC.MIDDLE), (0., 5, DEC.NOT_VISIBLE), (1e-9, 3, DEC.MIDDLE), (.1, 2, DEC.NOT_VISIBLE), (-.1, 5, DEC.NOT_VISIBLE)])
def test_the_bars_are_included(delta, positive, code):
    assert DEC.label_of(delta, positive) == code


def test_the_words_of_the_plan():
    assert DEC.READINGS == {'kept': '细档的好处过了原生层还在，扩到其余 14 对要交 BZ 或总负责人', 'not_visible': '过了原生层看不到', 'middle': '中间带', 'not_decidable': '不可判'}


def test_the_reading_says_seg4_and_sdlogj_without_using_them():
    result = read((.003,)*5, seg4=.80)
    assert result['reading_code'] == DEC.KEPT and '不进判定' in result['reading'] and result['delta_seg4']['mean'] < -.08 and result['mean_sdlogj_official'] == pytest.approx(.08)


@pytest.mark.parametrize('change,word', [
    (dict(cls='method_failure'), '分类'),
    (dict(passed=False), '终态证书'),
    (dict(seed_passed=False), '入口核对'),
    (dict(alpha=.9), '转移 α'),
    (dict(state='another field'), 'state hash'),
    (dict(bound=False), '绑定'),
    (dict(npz=False), '绑定'),
    (dict(recorded='another field'), '修复记录'),
])
def test_a_premise_of_one_run_not_met_is_not_decidable(change, word):
    runs = {('S_F', pair): fake_run(pair, .85) for pair in PAIRS}
    runs[('S_F', PAIRS[2])] = fake_run(PAIRS[2], .85, **change)
    result = DEC.decide(runs, fake_references(), PAIRS, short, GROUPS, pin=None)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and word in result['reading'] and '0440-0441' in result['reading']


def test_the_mean_sdlogj_is_a_premise():
    assert read((.003,)*5, sdlogj=.1)['reading_code'] == DEC.KEPT                  # the bar included
    result = read((.003,)*5, sdlogj=.1001)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and 'SDlogJ' in result['reading']


def test_one_high_sdlogj_is_only_an_anomaly():
    runs = {('S_F', pair): fake_run(pair, .843) for pair in PAIRS}
    runs[('S_F', PAIRS[0])] = fake_run(PAIRS[0], .843, sdlogj=.12)
    assert DEC.decide(runs, fake_references(), PAIRS, short, GROUPS, pin=None)['reading_code'] == DEC.KEPT
    runs[('S_F', PAIRS[0])]['metrics'].update(status='native_budget_exhausted')
    what = DEC.anomalies(runs, [('S_F', PAIRS[0])], short)
    assert any('SDlogJ' in a['what'] for a in what)


def test_a_run_without_the_evaluations_record_of_the_repaired_field_is_not_decidable():
    runs = {('S_F', pair): fake_run(pair, .85) for pair in PAIRS}
    del runs[('S_F', PAIRS[0])]['repair_metrics']
    result = DEC.decide(runs, fake_references(), PAIRS, short, GROUPS, pin=None)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and '修复记录' in result['reading']


def test_a_missing_pair_is_not_decidable():
    runs = {('S_F', pair): fake_run(pair, .85) for pair in PAIRS[:4]}
    result = DEC.decide(runs, fake_references(), PAIRS, short, GROUPS, pin=None)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and '没有 metrics.json' in result['reading']


def test_a_final_map_without_values_is_not_decidable():
    runs = {('S_F', pair): fake_run(pair, .85) for pair in PAIRS}
    del runs[('S_F', PAIRS[0])]['metrics']['metrics']['seg35']
    result = DEC.decide(runs, fake_references(), PAIRS, short, GROUPS, pin=None)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and 'Δ 不可算' in result['reading']


def test_the_reference_is_a_premise():
    runs = {('S_F', pair): fake_run(pair, .85) for pair in PAIRS}
    missing = fake_references()
    del missing[PAIRS[1]]['metrics']
    assert '没有 metrics.json' in DEC.decide(runs, missing, PAIRS, short, GROUPS, pin=None)['reading']
    failed = fake_references()
    failed[PAIRS[1]]['metrics']['certificate']['passed'] = False
    assert '终态证书' in DEC.decide(runs, failed, PAIRS, short, GROUPS, pin=None)['reading']


def test_the_pin_of_the_reference():
    runs = {('S_F', pair): fake_run(pair, .85) for pair in PAIRS}
    result = DEC.decide(runs, fake_references(.8430, .8984), PAIRS, short, GROUPS)
    assert result['reading_code'] == DEC.KEPT and DEC.S_P_PIN == (.8430, .8984)
    assert '钉住' in DEC.decide(runs, fake_references(.8431, .8984), PAIRS, short, GROUPS)['reading']


def test_only_s_f_decides():
    runs = {('S_F', pair): fake_run(pair, .84) for pair in PAIRS}
    alone = DEC.decide(runs, fake_references(), PAIRS, short, GROUPS, pin=None)
    runs.update({('S_X', pair): fake_run(pair, .90) for pair in PAIRS})
    runs.update({('S_Pp', pair): fake_run(pair, .90) for pair in PAIRS[1:2]})
    assert DEC.decide(runs, fake_references(), PAIRS, short, GROUPS, pin=None) == alone and alone['reading_code'] == DEC.NOT_VISIBLE
    assert [arm for arm, spec in CORE.ARMS.items() if spec['decides']] == ['S_F']


def test_there_is_no_gate_at_088():
    source = (ROOT/'experiments/s2/decide_s2.py').read_text()
    assert '.88 的线不作门' in source and 'GAIN_BAR, MIN_POSITIVE, NULL_POSITIVE = .002, 4, 2' in source
    tree_ = ast.parse(source)
    assert not [n.value for n in ast.walk(tree_) if isinstance(n, ast.Constant) and n.value == .88]


def test_the_groups_of_a_metrics_dict():
    metrics = fake_metrics(.84, .89, cortex=.8, small=.5)
    got = DEC.group_values(metrics, GROUPS)
    assert got['cortex'] == .8 and got['white_matter'] == .91 and got['small_six'] == .5 and got['large_six'] == pytest.approx(.84) and got['other_23'] == pytest.approx(.84) and got['seg35'] == .84 and got['sdlogj_brain'] == .45
    assert DEC.group_values(None, GROUPS)['seg35'] is None


def test_the_groups_are_the_evaluations():
    ev = SEED.load_evaluate()
    assert {name: sorted(labels) for name, labels in ev.A5_GROUPS.items()} == {name: sorted(labels) for name, labels in GROUPS.items()} and len(ev.A5_GROUPS['other_23']) == 23


def test_the_state_at_a_task_clock():
    rec = dict(curve=fake_curve(.84, offset=50.))
    assert DEC.at_task_clock(rec, 40.) is None and DEC.at_task_clock(rec, 50.)['t'] == 0 and DEC.at_task_clock(rec, 64.9)['t'] == 0 and DEC.at_task_clock(rec, 65.)['t'] == 15 and DEC.at_task_clock(rec, 330.)['t'] == CURVE.FINAL


def test_a_longer_hold_has_no_state_at_an_early_clock():
    a, b = {PAIRS[0]: dict(curve=fake_curve(.85, offset=90.))}, {PAIRS[0]: dict(curve=fake_curve(.84, offset=50.))}
    rows = DEC.same_clock_table(a, b, PAIRS[:1], clocks=(60, 100, 330))
    assert (rows[0]['n_a'], rows[0]['n_b']) == (0, 1) and rows[0]['seg35']['mean'] is None
    assert rows[1]['n_a'] == 1 and rows[1]['seg35']['mean'] == pytest.approx(.01) and rows[1]['age_a'] == pytest.approx(10.) and rows[2]['seg35']['n'] == 1


def test_the_same_t_table_pairs_the_runs():
    a, b = {p: dict(curve=fake_curve(.85)) for p in PAIRS}, {p: dict(curve=fake_curve(.84)) for p in PAIRS}
    rows = DEC.same_t_table(a, b, PAIRS)
    assert [r['t'] for r in rows] == list(CURVE.TIMES)+[CURVE.FINAL] and all(r['seg35']['mean'] == pytest.approx(.01) and r['seg35']['positive'] == 5 for r in rows)


def test_the_rejections_are_read_from_the_runs_records(neutral):
    run = neutral/'run'
    (run/'t').mkdir(parents=True)
    write_json(run/'protocol.json', dict(task=dict(name='t')))
    write_json(run/'t/summary.json', dict(native_commits=75, status='s', costs=dict(levels=[dict(level=2, attempts=1), dict(level=3, attempts=604, inner_accepted=375, rejected_reasons=dict(JACOBIAN_VIOLATION=229, OTHER=3))])))
    write_json(run/'t/history.json', [dict(level=3, elapsed_seconds=10., inner_accepted_steps=5), dict(level=3, elapsed_seconds=25., inner_accepted_steps=5), dict(level=2, elapsed_seconds=99., inner_accepted_steps=50)])
    row = DEC.rejection_row(run)
    assert row['attempts'] == 604 and row['jacobian_rejected'] == 229 and row['rejected_share'] == 229/604 and row['other_rejected'] == dict(OTHER=3) and row['native_commits'] == 75 and row['accepted_steps'] == 10 and row['seconds_per_accepted_step'] == 2.5
    assert row['other_rejected_total'] == 3
    assert DEC.rejection_means([dict(arm=row)], 'arm')['other_rejected_total'] == 3 and DEC.rejection_means([dict(arm=row)], 'arm')['n'] == 1


@pytest.mark.parametrize('name', ['summary.json', 'history.json'])
def test_a_run_without_its_records_has_no_rejection_row(neutral, name):
    run = neutral/'run'
    (run/'t').mkdir(parents=True)
    write_json(run/'protocol.json', dict(task=dict(name='t')))
    write_json(run/'t/summary.json', dict(costs=dict(levels=[])))
    write_json(run/'t/history.json', [])
    assert DEC.rejection_row(run)['attempts'] is None and DEC.rejection_row(run)['rejected_share'] is None and DEC.rejection_row(run)['seconds_per_accepted_step'] is None
    (run/'t'/name).unlink()
    assert DEC.rejection_row(run) is None


# ---- the gate ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Gate(FakeSystem):
    pass


@pytest.mark.parametrize('change,word', [
    (dict(unit_list=['tag-ul-train.service']), 'other units'),
    (dict(unit_list=['seq-vp-chain.service']), 'other units'),
    (dict(load=1.5), 'load'),
    (dict(gib={'/home': 99., '/mnt/c': 500.}), '/home'),
    (dict(gib={'/home': 500., '/mnt/c': 14.9}), '/mnt/c'),
    (dict(gpu='compute pids [1]'), 'GPU busy'),
    (dict(mib=15*1024-1), 'GPU free'),
])
def test_each_reason_closes_the_gate(neutral, change, word):
    machine = Gate()
    assert S2.busy_reason(machine, neutral, 15.) is None
    for key, value in change.items():
        setattr(machine, key, value)
    assert word in S2.busy_reason(machine, neutral, 15.)


def test_the_gate_is_open_at_the_floors_and_for_its_own_queue(neutral):
    machine = Gate()
    machine.load, machine.gib, machine.mib, machine.unit_list = 1.49, {'/home': 100., '/mnt/c': 15.}, 15*1024, ['tag-s2-queue.service']
    assert S2.busy_reason(machine, neutral, 15.) is None


def test_the_hold_file_closes_the_gate(neutral):
    (neutral/'hold').write_text('')
    assert S2.busy_reason(Gate(), neutral, 15.) == 'hold file'


def test_a_closed_gate_stops_after_its_time(neutral):
    machine = Gate()
    machine.load = 5.
    lines = []
    with pytest.raises(S2.GateClosed, match='gate closed for 30 minutes'):
        S2.wait_for_gate(machine, neutral, 30, lines.append, 15.)
    assert machine.clock >= 30*60 and all('load' in line for line in lines)


def test_the_queue_stops_when_the_gate_stays_closed_and_nothing_runs(world):
    machine = S2Machine(world)
    machine.load = 5.
    assert start_queue(world, machine) == 8
    assert 'gate closed' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == [] and not (world.root/'runs').exists() and machine.taper_checks == [] and not (world.root/'entries').exists()


# ---- the plan's cells, the start strings -----------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_plan_is_the_briefs():
    assert CORE.plan_runs()[:2] == [('S_Pp', PAIRS[1]), ('S_Pp', CORE.EXCEPTION_PAIR)]
    assert CORE.plan_runs()[2:] == [('S_F', p) for p in PAIRS]
    assert PAIRS == tuple(f'OASIS_OAS1_{n:04d}_MR1__OASIS_OAS1_{n+1:04d}_MR1' for n in range(438, 443)) and CORE.EXCEPTION_PAIR in S19.REST_PAIRS
    assert (CORE.ARMS['S_Pp']['ef_arm'], CORE.ARMS['S_Pp']['tier'], CORE.ARMS['S_F']['ef_arm'], CORE.ARMS['S_F']['tier']) == ('RL05', 'long', 'RF05', 'fine') and CORE.HOLD_S_P == SEED.ENTRY_CHARGE_SECONDS == 47.


@pytest.mark.parametrize('arm,pair', [('S_F', CORE.EXCEPTION_PAIR), ('S_Pp', PAIRS[0]), ('S_X', CORE.EXCEPTION_PAIR), ('S_F', 'OASIS_OAS1_0443_MR1__OASIS_OAS1_0444_MR1'), ('S_F', 'OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1'), ('S_P', PAIRS[0]), ('S_L', PAIRS[0]), ('O', PAIRS[0])])
def test_other_cells_are_refused_before_any_file(arm, pair, neutral):
    for call in (lambda: CORE.require_cell(arm, pair), lambda: CORE.run_dir(neutral, arm, pair), lambda: CORE.repair_dir(arm, pair, neutral), lambda: CORE.source_dir(arm, pair, neutral), lambda: CORE.entry_paths(neutral, arm, pair), lambda: S2.prepare_argv(arm, pair, neutral), lambda: CORE.unit_name(arm, pair), lambda: S2.unit_argv(arm, pair, neutral, neutral), lambda: CORE.cell_source(arm, pair, {}, neutral)):
        with pytest.raises(SystemExit):
            call()


@pytest.mark.parametrize('step', ['evaluate-one', 'curve-one'])
def test_the_steps_refuse_another_pair(step, neutral):
    with pytest.raises(SystemExit, match='S2 runs'):
        S2.main([step, str(neutral/'x'), '--arm', 'S_F', '--pair', 'OASIS_OAS1_0443_MR1__OASIS_OAS1_0444_MR1', '--run-id', 'x'])


def test_the_subjects_are_on_the_development_allowlist_and_off_the_training_and_holdout_lists():
    sys.path.insert(0, str(ROOT))
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')
    held = set((ROOT/'splits/bayesreg_v1/holdout_B_subjects.txt').read_text().split())
    train = set((ROOT/'splits/bayesreg_v1/train_subjects.txt').read_text().split())
    for _, pair in CORE.plan_runs(with_x=True):
        a, b = pair.split('__')
        check_pair_subjects(a, b, allowlist)
        assert not {a, b} & held and not {a, b} & train


def test_the_start_strings_hold_no_forbidden_word_and_the_right_names():
    root = Path('/x/s2_v1')
    options = CORE.SXOptions('/x/fl_v1', 'S', 2400, '/x/pins.json', 'f'*64)
    for argv in [S2.queue_launch(root), S2.queue_launch(root, sx=options)]+[S2.unit_argv(a, p, root/'runs'/a/p, root) for a, p in CORE.plan_runs(with_x=True)]+[S2.prepare_argv(a, p, root/'runs'/a/p) for a, p in CORE.plan_runs(with_x=True)]:
        assert 'pytest' not in ' '.join(argv) and 'seq-vp' not in ' '.join(argv)
    assert S2.queue_launch(root)[2] == '--unit=tag-s2-queue' and CORE.unit_name('S_F', PAIRS[0]) == 'tag-s2-S_F-0438-0439' and CORE.unit_name('S_Pp', CORE.EXCEPTION_PAIR) == 'tag-s2-S_Pp-0455-0456' and CORE.unit_name('S_X', PAIRS[4]) == 'tag-s2-S_X-0442-0443'
    assert S2.new_run_id('S_F', PAIRS[0]).startswith('s2_S_F_0438-0439_')
    with pytest.raises(ValueError, match='forbidden'):
        SEED.check_launch(['x', '/tmp/pytest-of-me/s2_v1'])


def test_the_units_are_s_ps_but_for_the_name():
    root = Path('/x/s2_v1')
    run = root/'runs/S_F'/PAIRS[0]
    argv = S2.unit_argv('S_F', PAIRS[0], run, root)
    other = SEED.unit_argv('S_P', PAIRS[0], run, root)
    assert [a for a in argv if not a.startswith('--unit=')] == [a for a in other if not a.startswith('--unit=')]
    assert '--property=MemoryMax=12G' in argv and argv[argv.index('--wait')+2] == f'--working-directory={run}/snapshot' and 'seed_inject.py' in argv[-1] and f'{SEED.WATCHDOG}s' in argv[-1]


def test_the_prepare_lines_are_s_ps_with_their_stages():
    run = Path('/x/s2_v1/runs/S_F')/PAIRS[0]
    argv = S2.prepare_argv('S_F', PAIRS[0], run)
    other = SEED.prepare_argv('S_P', PAIRS[0], run)
    assert [a.replace('S2-S_F', 'SEED-S_P') for a in argv] == other and '330' in argv and '1,1,0.1,0.01' in argv


def test_the_queue_string_carries_s_xs_options_only_when_given():
    root = Path('/x/s2_v1')
    assert not [a for a in S2.queue_launch(root) if a.startswith('--sx')]
    tokens = S2.queue_launch(root, sx=CORE.SXOptions('/x/fl_v1', 'S', 2400, '/x/pins.json', 'f'*64))
    assert tokens[tokens.index('--sx-arm')+1] == 'S' and tokens[tokens.index('--sx-checkpoint')+1] == '2400' and tokens[tokens.index('--sx-pins-sha256')+1] == 'f'*64 and tokens[tokens.index('--sx-root')+1] == '/x/fl_v1'


def test_the_only_root_is_s2_v1(world, capsys):
    other = world.results/'tc_v1'
    before = tree(other)
    assert S2.queue(world.options(root=str(other)), system=S2Machine(world), results=world.results, repo=ROOT, ev=world.ev) == 2
    assert tree(other) == before and 'writes only to' in capsys.readouterr().err
    assert CORE.root_problem(world.root, world.results) is None and CORE.root_problem(world.base/'s2_v2', world.results)


@pytest.mark.parametrize('value', [14.9, float('nan')])
def test_the_floor_cannot_be_lowered(world, value, capsys):
    assert S2.queue(world.options(min_host_gib=value), system=S2Machine(world), results=world.results, repo=ROOT, ev=world.ev) == 2
    assert not world.root.exists() and '--min-host-gib' in capsys.readouterr().err


def test_the_commands_refuse_another_root(neutral, capsys):
    assert S2.main(['decide', '--root', str(neutral/'tc_v1')]) == 2
    assert 'writes only to' in capsys.readouterr().err


@pytest.mark.parametrize('step', ['evaluate-one', 'curve-one'])
def test_the_steps_refuse_a_run_elsewhere(step, neutral, capsys):
    assert S2.main([step, str(neutral/'tc_v1/runs/S_F'/PAIRS[0]), '--arm', 'S_F', '--pair', PAIRS[0], '--run-id', 'x']) == 2
    assert 'writes only to' in capsys.readouterr().err


# ---- the firewall -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
TRUTH_KEYS = ('fixed_seg35', 'moving_seg35', 'fixed_seg4', 'moving_seg4', 'seg35', 'seg4')


def string_constants(path):
    return {n.value for n in ast.walk(ast.parse(Path(path).read_text())) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def test_no_truth_key_on_the_registration_path():
    """s2_core reads fields only; run_s2 names no label key either: only ev.truth_labels, in the two steps after a run, reads labels."""
    for name in ('s2_core.py', 'run_s2.py'):
        strings = string_constants(ROOT/'experiments/s2'/name)
        assert not [k for k in ('fixed_seg35', 'moving_seg35', 'fixed_seg4', 'moving_seg4') if k in strings], name
    source = (ROOT/'experiments/s2/s2_core.py').read_text()
    assert not re.search(r'(?<!re)pair\.npz', source.split('"""', 2)[2]) and 'truth_labels' not in source and 'np.load' in source      # repair.npz is not pair.npz
    wrapper = (ROOT/'experiments/seed/inject_run.py').read_text()
    assert 'seg35' not in wrapper and 'seg4' not in wrapper and 'pair.npz' not in wrapper


def test_a_construction_opens_only_the_two_field_files(world, monkeypatch):
    """The counterexample is in the fixture: every pair.npz there holds truth labels.  Any np.load of anything else here fails."""
    opened = []
    real = np.load

    def spy(path, *args, **kwargs):
        opened.append(Path(path))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', spy)
    manifest = world.files(PAIRS[0])[3]
    CORE.construct('S_F', PAIRS[0], manifest, world.results)
    assert [p.name for p in opened] == ['field.npz', 'repair.npz'] and all(p.is_relative_to(world.results) for p in opened)
    np.load(world.stage/PAIRS[0]/'pair.npz').close()     # the spy does see a label file
    assert opened[-1].name == 'pair.npz'


def test_the_taper_check_never_opens_a_pair_file(world, monkeypatch):
    opened = []
    real = np.load

    def spy(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', spy)
    checked(world)
    assert set(opened) == {'field.npz', 'repair.npz'}


def test_only_the_steps_after_a_run_read_truth_labels():
    tree_ = ast.parse((ROOT/'experiments/s2/run_s2.py').read_text())
    functions = {n.name: n for n in ast.walk(tree_) if isinstance(n, ast.FunctionDef)}
    readers = [name for name, n in functions.items() if any(isinstance(x, ast.Attribute) and x.attr in ('evaluate_run', 'curve_of_run', 'truth_labels') for x in ast.walk(n))]
    assert sorted(readers) == ['curve_one', 'evaluate_one']


# ---- the cleanup ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def own_run(world, arm='S_F', pair=PAIRS[0]):
    run = CORE.run_dir(world.root, arm, pair)
    run.mkdir(parents=True)
    write_json(run/'protocol.json', dict(task=dict(name='pg_weak__four__330s')))
    entry = CORE.write_entry(world.base/'scratch_entries', arm, pair, world.files(pair)[3], world.results)
    write_json(run/'seed_entry.json', entry)
    (run/'run_id').write_text('rid\n')
    task = write_task(run)
    write_json(run/'curve.json', dict(complete=True, self_check=dict(passed=True), run_dir=str(run), run_id='rid'))
    return run, task


def test_the_cleanup_takes_only_the_recorded_intermediate_maps(world):
    run, task = own_run(world)
    np.savez(task/'committed_9999_L3.npz', u=np.zeros(1))    # no record of it
    (task/'other.txt').write_text('x')
    n = CORE.cleanup_after_curve(run, world.root/'runs', CURVE)
    record = json_load(run/'cleanup.json')
    assert n == len(ELAPSED)+2 and (task/'final_native.npz').is_file() and (task/'other.txt').is_file() and (task/'committed_9999_L3.npz').is_file() and not (task/'committed_0001_L3.npz').exists() and not (task/'level_3_preoptimizer_done.npz').exists()
    assert [r['name'] for r in record['not_deleted_unrecorded']] == ['committed_9999_L3.npz'] and record['deleted_bytes'] > 0


def test_the_cleanup_does_what_tcs_does(world):
    """The same tree under tc_v1 and s2_v1: the same files go, the same record, but for the rule's words."""
    run, task = own_run(world)
    tc_root = world.base/'tc_copy'/CURVE.ROOT_NAME
    copy = tc_root/'runs/S_P'/PAIRS[0]
    shutil.copytree(run, copy)
    doc = json_load(copy/'curve.json')
    doc['run_dir'] = str(copy)
    write_json(copy/'curve.json', doc)
    n1, n2 = CORE.cleanup_after_curve(run, world.root/'runs', CURVE), CURVE.cleanup_after_curve(copy, tc_root/'runs')
    a, b = json_load(run/'cleanup.json'), json_load(copy/'cleanup.json')
    a.pop('rule'), b.pop('rule')
    assert n1 == n2 and a == b and sorted(p.name for p in task.iterdir()) == sorted(p.name for p in (copy/task.name).iterdir())


def test_the_cleanup_refuses_anything_outside_s2_v1(world):
    original = CORE.reference_dir(PAIRS[0], world.results)
    write_json(original/'curve.json', dict(complete=True, self_check=dict(passed=True), run_dir=str(original), run_id=None))
    before = tree(original)
    for runs_root in (world.results/'seed_v1/runs', world.results/'tc_v1/runs'):
        with pytest.raises(CURVE.CurveError, match='not a run directory'):
            CORE.cleanup_after_curve(original, runs_root, CURVE)
    with pytest.raises(CURVE.CurveError, match='not a run directory'):
        CORE.cleanup_after_curve(original, world.root/'runs', CURVE)
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


def test_the_cleanup_compares_the_run_and_its_id(world):
    run, task = own_run(world)
    write_json(run/'curve.json', dict(complete=True, self_check=dict(passed=True), run_dir=str(run), run_id='another'))
    with pytest.raises(CURVE.CurveError, match='run_id'):
        CORE.cleanup_after_curve(run, world.root/'runs', CURVE)
    write_json(run/'curve.json', dict(complete=True, self_check=dict(passed=True), run_dir='/elsewhere', run_id='rid'))
    with pytest.raises(CURVE.CurveError, match='is of the run'):
        CORE.cleanup_after_curve(run, world.root/'runs', CURVE)


def test_the_cleanup_refuses_a_task_directory_elsewhere(world):
    run, task = own_run(world)
    outside = world.base/'outside_task'
    shutil.move(task, outside)
    (run/task.name).symlink_to(outside)
    before = tree(outside)
    with pytest.raises(CURVE.CurveError, match='not a child'):
        CORE.cleanup_after_curve(run, world.root/'runs', CURVE)
    assert tree(outside) == before


# ---- the queue on a fake machine, the real steps -----------------------------------------------------------------------------------------------------------------------------------------------
READ_ONLY = ('ef_v1', 'mc0p_v1', 'mc0p_v1b', 'mc0p_v1c', 'seed_v1', 'seed_v1b', 'tc_v1', 'd0_v1_a5')


def read_only_trees(world):
    return {name: tree(world.results/name) for name in READ_ONLY if (world.results/name).exists()}


def test_the_queue_runs_the_plan_and_writes_only_under_its_root(world):
    machine = S2Machine(world)
    before = read_only_trees(world)
    assert start_queue(world, machine) == 0
    assert machine.launched == [CORE.unit_name(a, p) for a, p in CORE.plan_runs()] and read_only_trees(world) == before and machine.taper_checks == [['S_Pp', 'S_F']]
    assert machine.sleeps.count(SEED.BETWEEN_RUNS_SECONDS) == 6
    for arm, pair in CORE.plan_runs():
        run = CORE.run_dir(world.root, arm, pair)
        entry, metrics = json_load(run/'seed_entry.json'), json_load(run/'metrics.json')
        assert entry['field_path'] == str(CORE.entry_paths(world.root, arm, pair)[0]) and entry['charge_seconds'] == 47. and (run/'seed_inject.py').read_bytes() == (SEED.HERE/'inject_run.py').read_bytes()
        assert metrics['seed']['passed'] is True and metrics['seed']['accepted_alpha'] == 1. and metrics['seed']['repaired_u_sha256'] == entry['repaired_u_sha256'] and metrics['s2']['arm'] == arm and metrics['method'] == S2.method_name(arm)
        assert json_load(run/'curve.json')['complete'] is True and json_load(run/'queue_done.json')['state'] == 'ok' and (run/'run_id').read_text().startswith(f's2_{arm}_')
        assert not list((run/'pg_weak__four__330s').glob('committed_*')) and not list((run/'pg_weak__four__330s').glob('level_*')) and (run/'pg_weak__four__330s/final_native.npz').is_file() and json_load(run/'cleanup.json')['deleted']
        assert (run/'cmd.txt').is_file() and (run/'git.txt').is_file() and (run/'data.txt').is_file() and (run/'env.txt').is_file() and (run/'seed').is_file() and (run/'s2_arm.json').is_file()
    log = (world.root/'queue.log').read_text()
    assert 'taper check passed on 7 cells' in log and 'reading:' in log and 'Handed to the lead' in log and not (world.root/'queue_stop.json').exists()


def test_the_analysis_says_seg35_and_seg4_and_keeps_the_earlier_version(world):
    assert start_queue(world, S2Machine(world)) == 0
    out = world.root/'analysis'
    decision, summary = json_load(out/'decision_s2.json'), (out/'summary_s2.md').read_text()
    assert decision['reading_arm'] == 'S_F' and decision['reading_code'] == DEC.NOT_VISIBLE and decision['delta_seg35']['n'] == 5 and decision['delta_seg4']['n'] == 5 and decision['delta_seg35']['mean'] < 0
    d = decision['description']
    assert set(d['final_differences']) == set(DEC.DIFFERENCE_KEYS) and [r['t'] for r in d['same_t']][-1] == CURVE.FINAL and [r['task_clock'] for r in d['same_clock']] == list(DEC.TASK_CLOCKS) and len(d['retention']) == 7 and d['without_alpha_pair']['seg35']['n'] == 4 and d['alpha_pair']['seg35']['n'] == 1 and d['s_x']['ran'] is False
    assert d['rejection']['arm']['rejected_share'] == .4 and d['rejection']['s_p']['rejected_share'] == .375 and d['rejection']['s_p_tc']['n'] == 5 and d['rejection']['arm']['native_commits'] == len(ELAPSED)
    seg35, cortex = d['retention'][0], d['retention'][2]
    assert seg35['delta_entry']['mean'] == pytest.approx(.02) and seg35['delta_entry_accepted']['mean'] == pytest.approx(.024) and seg35['delta_entry_accepted']['per_pair'][PAIRS[1]] == pytest.approx(.04) and d['retention'][1]['delta_entry_accepted']['mean'] == pytest.approx(.032)
    assert seg35['kept'] == pytest.approx(seg35['delta_final']['mean']/.02) and seg35['kept_accepted'] == pytest.approx(seg35['delta_final']['mean']/.024) and cortex['delta_entry_accepted'] is None and cortex['kept_accepted'] is None
    split = d['alpha_pair_split']
    assert split['s_f_minus_s_pp']['seg35']['n'] == 1 and split['s_pp_minus_s_p']['seg35']['n'] == 1 and split['s_f_minus_s_pp']['seg35']['mean']+split['s_pp_minus_s_p']['seg35']['mean'] == pytest.approx(d['alpha_pair']['seg35']['mean'])
    assert d['rejection']['arm']['other_rejected_total'] == 0
    assert [r['pair'] for r in d['s_pp']['rows']] == ['0439-0440', '0455-0456'] and d['s_pp']['rows'][0]['seg35_entry_gain'] == pytest.approx(.04) and d['s_pp']['rows'][0]['s_p_accepted_alpha'] == .9 and d['s_pp']['rows'][0]['s_pp_accepted_alpha'] == 1.
    for word in ('seg35', 'seg4', '皮层', '白质', '小六个', '大六个', '其余 23 个', '官方 SDlogJ', '脑内 SDlogJ', '被拒比例', '留存', '同任务时钟', '同 t', '0439-0440', 'S_Pp', '没有跑', '它证伪不了什么', '.88 的线不作门', '预算收尾', '开发集', 'Δ 入口（S_P 接受 α）', '留存（接受 α）', 'S_F − S_Pp', '其他拒绝', '不能和 TC 结果的 217/673', '不是判定的 Δ', '同任务时钟（S_P 是 tc_v1'):
        assert word in summary, word
    assert {row['arm'] for row in map(dict, __import__('csv').DictReader(open(out/'long_curve.csv')))} == {'S_F', 'S_Pp', 'S_P_tc_v1'}
    S2.write_analysis(world.root, ev=world.ev, results=world.results)
    assert len(list(out.glob('previous_*'))) == 1 and (out/'summary_s2.md').is_file()


def test_the_reading_stands_on_the_real_steps(world):
    """The pinned references of the world: S_P's final map is .8430 / .8984 on every pair, the synthetic S_F's Dice is far from it: the code reads what it reads, and the same from the files by hand."""
    assert start_queue(world, S2Machine(world)) == 0
    decision = json_load(world.root/'analysis/decision_s2.json')
    deltas = [json_load(CORE.run_dir(world.root, 'S_F', p)/'metrics.json')['metrics']['seg35']-.8430 for p in PAIRS]
    assert decision['delta_seg35']['mean'] == pytest.approx(float(np.mean(deltas))) and decision['delta_seg35']['positive'] == sum(d > 0 for d in deltas)


def test_a_node_in_the_band_stops_the_queue_before_any_entry_and_any_run(world):
    raw = world.raw[('S_F', PAIRS[3])]
    nodes = np.sort(np.append(interior_nodes(), band_node()))
    world.make_repair('S_F', PAIRS[3], nodes=nodes, values=moved_values(raw, nodes, unmoved=0))
    machine = S2Machine(world)
    assert start_queue(world, machine) == 8
    stop = json_load(world.root/'queue_stop.json')['reason']
    assert 'taper check did not pass' in stop and '0441-0442' in stop and 'band' in stop and machine.launched == [] and not (world.root/'entries').exists() and not (world.root/'runs').exists()


def test_a_certificate_that_fails_stops_the_queue_before_any_entry_and_any_run(world):
    machine = S2Machine(world, passes=False)
    assert start_queue(world, machine) == 8
    assert 'the certificate did not pass' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == [] and not (world.root/'entries').exists()


def test_a_missing_reference_stops_before_the_check(world):
    (world.results/'tc_v1/runs/S_P'/PAIRS[2]/'curve.json').unlink()
    (CORE.reference_dir(CORE.EXCEPTION_PAIR, world.results)/'metrics.json').unlink()
    machine = S2Machine(world)
    assert start_queue(world, machine) == 8
    stop = json_load(world.root/'queue_stop.json')['reason']
    assert 'no curve.json' in stop and '0455-0456' in stop and machine.taper_checks == [] and machine.launched == []


def test_a_frozen_or_input_problem_stops_before_the_check(world):
    machine = S2Machine(world)
    machine.frozen, machine.inputs = ['frozen made up'], ['inputs made up']
    assert start_queue(world, machine) == 8
    stop = json_load(world.root/'queue_stop.json')['reason']
    assert 'frozen made up' in stop and 'inputs made up' in stop and machine.taper_checks == [] and machine.launched == []


@pytest.mark.parametrize('attribute', ['bad_raw', 'bad_alpha', 'bad_state'])
def test_an_entry_not_injected_as_recorded_stops_and_keeps_the_run(world, attribute):
    """The walk of the injection path: S_Pp goes first, and a transfer that is not alpha = 1 on the repaired field stops before S_F starts."""
    machine = S2Machine(world)
    unit = CORE.unit_name('S_Pp', CORE.S_PP_PAIRS[0])
    getattr(machine, attribute).add(unit)
    assert start_queue(world, machine) == 8
    assert machine.launched == [unit] and 'not injected as recorded' in json_load(world.root/'queue_stop.json')['reason']
    run = CORE.run_dir(world.root, 'S_Pp', CORE.S_PP_PAIRS[0])
    assert list((run/'pg_weak__four__330s').glob('committed_*')) and not (run/'queue_done.json').exists() and not (run/'cleanup.json').exists()
    assert json_load(world.root/'analysis/decision_s2.json')['reading_code'] == DEC.NOT_DECIDABLE


def test_a_failed_curve_stops_and_keeps_the_commits(world):
    machine = S2Machine(world)
    machine.fail_curve.add(('S_F', PAIRS[0]))
    assert start_queue(world, machine) == 8
    run = CORE.run_dir(world.root, 'S_F', PAIRS[0])
    assert list((run/'pg_weak__four__330s').glob('committed_*')) and 'curve failed' in json_load(world.root/'queue_stop.json')['reason'] and (run/'metrics.json').is_file()


def test_a_failed_prepare_check_stops(world):
    machine = S2Machine(world)
    machine.fail_check = True
    assert start_queue(world, machine) == 8
    assert 'differs from the configuration' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == []


def test_a_method_failure_is_kept_whole_and_the_queue_goes_on(world):
    machine = S2Machine(world)
    machine.classes[CORE.unit_name('S_F', PAIRS[1])] = 'method_failure'
    assert start_queue(world, machine) == 0
    run = CORE.run_dir(world.root, 'S_F', PAIRS[1])
    assert list((run/'pg_weak__four__330s').glob('committed_*')) and not (run/'curve.json').exists() and json_load(run/'queue_done.json')['state'] == 'method_failure'
    decision = json_load(world.root/'analysis/decision_s2.json')
    assert decision['reading_code'] == DEC.NOT_DECIDABLE and '0439-0440' in decision['reading'] and len(machine.launched) == 7


def test_three_infrastructure_failures_in_a_row_stop(world):
    machine = S2Machine(world)
    machine.infrastructure |= {CORE.unit_name(a, p) for a, p in CORE.plan_runs()[:3]}
    assert start_queue(world, machine) == 8
    assert '3 infrastructure failures' in json_load(world.root/'queue_stop.json')['reason'] and len(machine.launched) == 3


def test_a_finished_run_is_not_rerun_and_an_incomplete_one_stops(world):
    machine = S2Machine(world)
    assert start_queue(world, machine) == 0
    again = S2Machine(world)
    done = tree(world.root/'runs')
    assert start_queue(world, again) == 0
    assert again.launched == [] and tree(world.root/'runs') == done
    assert len(json_load(world.root/'batch.json')['starts']) == 2


def test_an_incomplete_attempt_stops_and_stays(world):
    run = CORE.run_dir(world.root, 'S_Pp', CORE.S_PP_PAIRS[0])
    run.mkdir(parents=True)
    (run/'something').write_text('x')
    machine = S2Machine(world)
    assert start_queue(world, machine) == 8
    assert 'incomplete attempt' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == [] and (run/'something').is_file()


def test_a_second_driver_is_refused(world):
    world.root.mkdir(parents=True)
    import fcntl
    held = open(world.root/'queue.lock', 'w')
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert start_queue(world, S2Machine(world)) == 2
    finally:
        held.close()


def test_s_x_joins_with_its_options_and_its_hold(world):
    machine = S2Machine(world)
    assert start_queue(world, machine) == 0
    options = make_sx(world)
    second = S2Machine(world)
    before = tree(world.root/'runs/S_F')
    assert start_queue(world, second, sx_root=options.root, sx_arm=options.arm, sx_checkpoint=options.checkpoint, sx_pins=options.pins, sx_pins_sha256=options.pins_sha256) == 0
    assert second.launched == [CORE.unit_name('S_X', p) for p in PAIRS] and tree(world.root/'runs/S_F') == before and second.taper_checks == [['S_Pp', 'S_F', 'S_X']]
    for pair in PAIRS:
        run = CORE.run_dir(world.root, 'S_X', pair)
        entry = json_load(run/'seed_entry.json')
        assert entry['charge_seconds'] == 73. and entry['source_root'] == 'fl_v1' and entry['source_tier'] == 'S@2400' and json_load(run/'metrics.json')['seed']['passed'] is True and json_load(run/'s2_arm.json')['charge_seconds'] == 73.
    assert json_load(world.root/'batch.json')['starts'][-1]['s_x']['arm'] == 'S'
    decision = json_load(world.root/'analysis/decision_s2.json')
    assert decision['description']['s_x']['ran'] is True and decision['description']['s_x']['hold_seconds'] == [73.] and decision['description']['s_x']['against_s_f']['seg35']['n'] == 5 and decision['description']['s_x']['hold']['median_with_warmup'] == pytest.approx(84.5)
    summary = (world.root/'analysis/summary_s2.md').read_text()
    assert 'S_X 终态 − S_F 终态' in summary and 'S_X 记满，S_F、S_P 少记 10–23 s，同任务时钟的差对 S_X 偏不利' in summary and '含 warm-up 84.5 s' in summary


def test_s_x_with_a_wrong_pin_stops_before_any_run(world):
    options = make_sx(world)
    machine = S2Machine(world)
    assert start_queue(world, machine, sx_root=options.root, sx_arm=options.arm, sx_checkpoint=options.checkpoint, sx_pins=options.pins, sx_pins_sha256='f'*64) == 8
    assert 'sx-pins-sha256' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == [] and machine.taper_checks == []


def sx_changes(options):
    return dict(sx_root=options.root, sx_arm=options.arm, sx_checkpoint=options.checkpoint, sx_pins=options.pins, sx_pins_sha256=options.pins_sha256)


def test_a_changed_hold_after_the_entries_exist_is_refused(world):
    options = make_sx(world)
    first = S2Machine(world)

    def close_the_gate():                                # the entries get written, then the gate of the first run stays closed: nothing runs
        first.load = 5.
    first.after_taper_check = close_the_gate
    assert start_queue(world, first, **sx_changes(options)) == 8
    assert CORE.entry_paths(world.root, 'S_X', PAIRS[0])[0].is_file() and first.launched == [] and 'gate closed' in json_load(world.root/'queue_stop.json')['reason']
    for pair in PAIRS:
        d = CORE.fl_repair_dir(world.results/CORE.FL_ROOT, 'S', 2400, pair)
        doc = json_load(d/'repair.json')
        doc['seconds_repair'] = 100.
        write_json(d/'repair.json', doc)
    slower = pin(world, 'S', 2400)
    second = S2Machine(world)
    assert start_queue(world, second, **sx_changes(slower)) == 8
    assert 'nothing overwritten' in json_load(world.root/'queue_stop.json')['reason'] and second.launched == []


# ---- the gate before the certificates (review I1) ------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('how', ['hold', 'load', 'unit', 'gpu'])
def test_a_closed_gate_keeps_the_certificates_and_the_entries_from_starting(world, how):
    machine = S2Machine(world)
    world.root.mkdir(parents=True)
    if how == 'hold':
        (world.root/'hold').write_text('')
    elif how == 'load':
        machine.load = 5.
    elif how == 'unit':
        machine.unit_list = ['tag-fl-queue.service']
    else:
        machine.gpu = 'compute pids [1]'
    assert start_queue(world, machine) == 8
    assert 'gate closed' in json_load(world.root/'queue_stop.json')['reason'] and machine.taper_checks == [] and not (world.root/'entries').exists() and not (world.root/'runs').exists() and machine.launched == []
    assert not list(world.root.glob('taper_check_*')) and not (world.root/'batch.json').exists()


def test_the_certificates_wait_for_the_gate_and_then_start(world):
    machine = S2Machine(world)
    machine.load = 5.
    real_sleep = machine.sleep

    def sleep(seconds):                                  # the other job ends after a while
        real_sleep(seconds)
        if machine.clock >= 600:
            machine.load = .5
    machine.sleep = sleep
    assert start_queue(world, machine) == 0
    assert machine.taper_checks == [['S_Pp', 'S_F']] and 'waiting: 1-minute load' in (world.root/'queue.log').read_text() and len(machine.launched) == 7


def test_the_cheap_checks_come_before_the_gate(world):
    """A missing file stops at once: it does not wait 30 minutes behind a closed gate first."""
    (world.results/'tc_v1/runs/S_P'/PAIRS[2]/'curve.json').unlink()
    machine = S2Machine(world)
    machine.load = 5.
    assert start_queue(world, machine) == 8
    assert 'no curve.json' in json_load(world.root/'queue_stop.json')['reason'] and machine.clock == 0.


# ---- handover_only and the rule of the two empty pairs -----------------------------------------------------------------------------------------------------------------------------------------
def empty_units(machine, *pairs, cls='handover_only'):
    for pair in pairs:
        unit = CORE.unit_name('S_F', pair)
        machine.empty.add(unit)
        machine.classes[unit] = cls


def test_a_handover_only_run_is_kept_whole(world):
    """classify_run.py says kept whole; hard rule 4: only class ok is cleaned."""
    machine = S2Machine(world)
    empty_units(machine, PAIRS[0])
    assert start_queue(world, machine) == 0
    run = CORE.run_dir(world.root, 'S_F', PAIRS[0])
    task = run/'pg_weak__four__330s'
    assert json_load(run/'queue_done.json')['state'] == 'handover_only' and not (run/'cleanup.json').exists() and list(task.glob('level_*')) and (task/'final_native.npz').is_file() and CORE.native_commit_count(run) == 0
    assert len(machine.launched) == 7
    decision = json_load(world.root/'analysis/decision_s2.json')
    assert decision['reading_code'] == DEC.NOT_DECIDABLE and '分类 handover_only' in decision['reading']
    ok = CORE.run_dir(world.root, 'S_F', PAIRS[1])
    assert (ok/'cleanup.json').is_file() and not list((ok/'pg_weak__four__330s').glob('level_*'))


def test_a_curve_that_fails_on_a_handover_only_run_stops_nothing(world):
    machine = S2Machine(world)
    empty_units(machine, PAIRS[2])
    machine.fail_curve.add(('S_F', PAIRS[2]))
    assert start_queue(world, machine) == 0
    run = CORE.run_dir(world.root, 'S_F', PAIRS[2])
    assert len(machine.launched) == 7 and not (run/'curve.json').exists() and 'handover_only, no curve' in (world.root/'queue.log').read_text()


@pytest.mark.parametrize('first,second', [('handover_only', 'handover_only'), ('method_failure', 'handover_only'), ('handover_only', 'method_failure'), ('method_failure', 'method_failure')])
def test_two_empty_s_f_pairs_stop_the_queue(world, first, second):
    """The rule of plan revision 1 (the review's minor 2).  What ran stays; the reading is not decidable; 3 pairs and about 22 minutes are saved."""
    machine = S2Machine(world)
    empty_units(machine, PAIRS[0], cls=first)
    empty_units(machine, PAIRS[1], cls=second)
    assert start_queue(world, machine) == 8
    assert machine.launched == [CORE.unit_name(a, p) for a, p in CORE.plan_runs()[:4]]
    reason = json_load(world.root/'queue_stop.json')['reason']
    assert 'both 0 native commits' in reason and 'plan revision 1' in reason
    assert json_load(world.root/'analysis/decision_s2.json')['reading_code'] == DEC.NOT_DECIDABLE
    for pair in PAIRS[:2]:
        assert list((CORE.run_dir(world.root, 'S_F', pair)/'pg_weak__four__330s').glob('level_*')) and not (CORE.run_dir(world.root, 'S_F', pair)/'cleanup.json').exists()


@pytest.mark.parametrize('cls', ['handover_only', 'method_failure'])
def test_two_s_f_pairs_without_commits_json_stop_the_queue(world, cls):
    """The other branch of the rule (the review of the revision, minor 2): a real run with no outer record has no commits.json, not an empty one.  It counts as 0; there is no curve, and that stops nothing but the rule."""
    machine = S2Machine(world)
    empty_units(machine, PAIRS[0], PAIRS[1], cls=cls)
    machine.no_commits |= {CORE.unit_name('S_F', pair) for pair in PAIRS[:2]}
    assert start_queue(world, machine) == 8
    assert machine.launched == [CORE.unit_name(a, p) for a, p in CORE.plan_runs()[:4]]
    reason = json_load(world.root/'queue_stop.json')['reason']
    assert 'both 0 native commits' in reason and 'plan revision 1' in reason
    assert json_load(world.root/'analysis/decision_s2.json')['reading_code'] == DEC.NOT_DECIDABLE
    for pair in PAIRS[:2]:
        run = CORE.run_dir(world.root, 'S_F', pair)
        task = run/'pg_weak__four__330s'
        assert not (task/'commits.json').exists() and CORE.native_commit_count(run) == 0 and not (run/'curve.json').exists() and not (run/'cleanup.json').exists() and list(task.glob('level_*')) and json_load(run/'queue_done.json')['state'] == cls


def test_one_pair_without_commits_json_does_not_stop_the_queue(world):
    machine = S2Machine(world)
    empty_units(machine, PAIRS[0])
    machine.no_commits.add(CORE.unit_name('S_F', PAIRS[0]))
    assert start_queue(world, machine) == 0 and len(machine.launched) == 7
    assert 'handover_only, no curve' in (world.root/'queue.log').read_text() and json_load(world.root/'analysis/decision_s2.json')['reading_code'] == DEC.NOT_DECIDABLE


def test_the_count_of_native_commits(neutral):
    run = neutral/'run'
    (run/'t').mkdir(parents=True)
    assert CORE.native_commit_count(run) == 0                # no protocol.json either
    write_json(run/'protocol.json', dict(task=dict(name='t')))
    assert CORE.native_commit_count(run) == 0                # no commits.json
    write_json(run/'t/commits.json', [])
    assert CORE.native_commit_count(run) == 0
    write_json(run/'t/commits.json', [dict(level=3, native=True), dict(level=2, native=False), dict(level=3, native=False), dict(level=3, native=True)])
    assert CORE.native_commit_count(run) == 2


def test_the_cleanup_says_only_class_ok(world):
    run, _ = own_run(world)
    CORE.cleanup_after_curve(run, world.root/'runs', CURVE)
    rule = json_load(run/'cleanup.json')['rule']
    assert '(class ok only' in rule and 'classes ok and handover_only' not in rule


def test_a_second_start_does_not_step_over_the_rule(world):
    machine = S2Machine(world)
    empty_units(machine, PAIRS[0], PAIRS[1])
    assert start_queue(world, machine) == 8
    again = S2Machine(world)
    assert start_queue(world, again) == 8
    assert again.launched == [] and 'both 0 native commits' in json_load(world.root/'queue_stop.json')['reason']


@pytest.mark.parametrize('empty', [(PAIRS[0],), (PAIRS[1],), (PAIRS[2], PAIRS[3])])
def test_the_rule_needs_the_first_two_pairs_both_empty(world, empty):
    machine = S2Machine(world)
    empty_units(machine, *empty)
    assert start_queue(world, machine) == 0 and len(machine.launched) == 7


def test_an_infrastructure_failure_does_not_count_for_the_rule(world):
    machine = S2Machine(world)
    empty_units(machine, PAIRS[1])
    machine.infrastructure.add(CORE.unit_name('S_F', PAIRS[0]))
    assert start_queue(world, machine) == 0 and len(machine.launched) == 7


def test_empty_s_pp_runs_do_not_count_for_the_rule(world):
    machine = S2Machine(world)
    for pair in CORE.S_PP_PAIRS:
        unit = CORE.unit_name('S_Pp', pair)
        machine.empty.add(unit)
        machine.classes[unit] = 'handover_only'
    assert start_queue(world, machine) == 0 and len(machine.launched) == 7


# ---- an empty repair --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
EMPTY_NODES, EMPTY_VALUES = np.empty(0, dtype=np.int64), np.empty((0, 3))


def test_an_empty_repair_gives_raw_itself():
    """repair_field on a field that already passes (already_certified) stores two empty arrays: raw' is raw, recorded, nothing stops."""
    raw = raw_field()
    tapered_sha = CORE.array_sha(CORE.tapered(raw))
    prime, record = CORE.raw_prime(raw, EMPTY_NODES, EMPTY_VALUES, tapered_sha256=tapered_sha, repaired_sha256=tapered_sha)
    assert prime.tobytes() == raw.tobytes() and record['empty_repair'] is True and record['n_repair_nodes'] == 0 and record['n_nodes_in_band'] == 0 and record['min_face_distance'] is None and record['taper_of_raw_prime_sha256'] == tapered_sha


def test_an_empty_repair_of_a_cell_goes_through_the_file_and_the_check(world):
    world.make_repair('S_F', PAIRS[0], nodes=EMPTY_NODES, values=EMPTY_VALUES)
    d = CORE.repair_dir('S_F', PAIRS[0], world.results)
    nodes, values = CORE.read_repair(d/'repair.npz', sha(d/'repair.npz'), SHAPE)
    assert nodes.shape == (0,) and values.shape == (0, 3)
    entry = CORE.write_entry(world.root, 'S_F', PAIRS[0], world.files(PAIRS[0])[3], world.results)
    assert entry['construction']['empty_repair'] is True and entry['u_sha256'] == CORE.array_sha(world.raw[('S_F', PAIRS[0])])
    record, lines = checked(world, cells=[('S_F', PAIRS[0])])
    assert record['passed'] is True and 'nodes 0 (moved 0)' in lines[0]
    other = CORE.write_entry(world.root, 'S_F', PAIRS[1], world.files(PAIRS[1])[3], world.results)
    assert other['construction']['empty_repair'] is False


def test_an_empty_repair_with_a_record_of_another_field_is_refused(world):
    world.make_repair('S_F', PAIRS[0], nodes=EMPTY_NODES, values=EMPTY_VALUES)
    d = CORE.repair_dir('S_F', PAIRS[0], world.results)
    doc, metrics = json_load(d/'repair.json'), json_load(d/'repair_metrics.json')
    doc['certificate']['state_hash'] = metrics['repaired_u_sha256'] = 'f'*64
    write_json(d/'repair.json', doc)
    write_json(d/'repair_metrics.json', metrics)
    with pytest.raises(CORE.EntryError, match='the repaired field'):
        CORE.construct('S_F', PAIRS[0], world.files(PAIRS[0])[3], world.results)


# ---- the trace, the reading's code ------------------------------------------------------------------------------------------------------------------------------------------------------------
def traced_run(world):
    run = world.base/'run'
    run.mkdir()
    entry = CORE.write_entry(world.root, 'S_F', PAIRS[0], world.files(PAIRS[0])[3], world.results)
    write_json(run/'seed_entry.json', entry)
    write_json(run/'protocol.json', dict(task=dict(name='pg_weak__four__330s')))
    write_task(run)
    assert CORE.entry_trace(run)[1] == []
    return run, entry, run/'pg_weak__four__330s/levels.json'


def test_the_trace_refuses_a_seed_that_is_not_the_repaired_field(world):
    run, _, levels = traced_run(world)
    doc = json_load(levels)
    doc[0]['preoptimizer']['seed_hash'] = 'another'
    write_json(levels, doc)
    assert any('seed_hash another != the repaired field' in p for p in CORE.entry_trace(run)[1])


def test_the_trace_refuses_a_hold_that_is_not_the_records(world):
    run, _, levels = traced_run(world)
    doc = json_load(levels)
    doc[0]['preoptimizer']['charge_seconds'] = 46.
    write_json(levels, doc)
    assert any('the hold 46.0 != seed_entry.json 47.0' in p for p in CORE.entry_trace(run)[1])


def test_the_trace_refuses_an_entry_whose_two_hashes_differ(world):
    run, entry, _ = traced_run(world)
    write_json(run/'seed_entry.json', dict(entry, repaired_u_sha256='another'))
    assert any('seed_entry.json: repaired_u_sha256 another' in p for p in CORE.entry_trace(run)[1])


def test_the_trace_refuses_another_lower_bound_than_the_evaluations(world):
    """The same certificate, bit for bit (review minor 6): another bound means another configuration on the frozen path."""
    run, _, levels = traced_run(world)
    doc = json_load(levels)
    doc[1]['transfer_recovery']['attempts'][0]['certificate']['min_jd_lower_bound'] = 2e-4
    write_json(levels, doc)
    assert any("the transfer certificate's lower bound 0.0002" in p for p in CORE.entry_trace(run)[1])


def test_the_batch_pins_the_readings_code(world):
    """Review minor 4: a changed decide_s2.py is a STOP at the next start and is written in the summary of a later `decide`."""
    assert start_queue(world, S2Machine(world)) == 0
    batch = json_load(world.root/'batch.json')
    assert batch['decide_sha256'] == sha(DEC.__file__)
    assert json_load(world.root/'analysis/decision_s2.json')['decide_code']['same'] is True
    write_json(world.root/'batch.json', dict(batch, decide_sha256='f'*64))
    result = S2.write_analysis(world.root, ev=world.ev, results=world.results)
    assert result['decide_code'] == dict(at_first_start='f'*64, now=sha(DEC.__file__), same=False) and '判定代码与队列第一次启动时不同' in (world.root/'analysis/summary_s2.md').read_text()
    machine = S2Machine(world)
    assert start_queue(world, machine) == 8
    assert 'batch.json differs' in json_load(world.root/'queue_stop.json')['reason'] and 'decide_sha256' in json_load(world.root/'queue_stop.json')['reason'] and machine.launched == []


def test_a_rehearsal_without_a_batch_says_nothing_of_the_code(world):
    (world.root).mkdir(parents=True)
    result = S2.write_analysis(world.root, ev=world.ev, results=world.results)
    assert result['decide_code']['same'] is None and result['decide_code']['at_first_start'] is None and '判定代码与队列第一次启动时不同' not in (world.root/'analysis/summary_s2.md').read_text()


def test_the_real_system_calls_its_own_script(world, monkeypatch):
    """The real class with subprocess replaced: nothing starts."""
    monkeypatch.undo()
    seen = []

    def fake_run(argv, **kwargs):
        seen.append((argv, kwargs.get('env')))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(S2.subprocess, 'run', fake_run)
    system = S2.System()
    run = world.base/'neutral_run'
    run.mkdir()
    assert system.evaluate(run, 'S_F', PAIRS[0], 'rid') == 0 and system.curve(run, 'S_F', PAIRS[0], 'rid') == 0
    assert system.taper_check(world.base/'out.json', None, world.base/'log.txt', ['S_Pp', 'S_F']) == 0
    (a, _), (b, _), (c, env) = seen
    assert a[2].endswith('experiments/s2/run_s2.py') and a[3] == 'evaluate-one' and b[3] == 'curve-one' and c[3] == 'taper-check' and c[c.index('--arms')+1:c.index('--json')] == ['S_Pp', 'S_F']
    assert env['CUDA_VISIBLE_DEVICES'] == '' and env['NUMBA_NUM_THREADS'] == str(S2.TAPER_THREADS) and env['JD_CURL_JIT_CACHE'] == str(world.base/'cache/numba_jit') and env['JD_CURL_FIRST_USE_CACHE'] == str(world.base/'cache/first_use')


def test_nothing_here_starts_a_unit():
    """Only `launch` of the machine and the printed strings have systemd-run; the steps do not."""
    source = (ROOT/'experiments/s2/s2_core.py').read_text()
    assert 'systemd-run' not in source and 'subprocess' not in source
    assert 'systemd-run' not in (ROOT/'experiments/s2/decide_s2.py').read_text()
    assert shlex.join(S2.queue_launch(Path('/x/s2_v1'))).startswith('systemd-run --user --unit=tag-s2-queue --collect')


@pytest.mark.parametrize('argv,word', [(['taper-check', '--arms', 'S_Q'], 'only the arms'), (['taper-check', '--sx-arm', 'S'], 'all of'), (['queue', '--sx-arm', 'S'], 'all of')])
def test_the_commands_refuse_bad_arguments_before_any_file(argv, word):
    with pytest.raises(SystemExit, match=word):
        S2.main(argv)
