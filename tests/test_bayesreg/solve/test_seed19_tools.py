"""SEED19 (runs/20261104-seed19/任务书.md, plan.md; docs/bayesreg/05_D0协议.md §17): experiments/mc0/run_mc0pc.py, experiments/seed/run_seed19.py, decide_seed19.py; CPU only, made-up records and synthetic small volumes, no real image, label, prediction or field, no GPU, no unit.

- the entry map: the wrong pair, an oracle record, the wrong root, a retry_of, a failed run: refused; only the 14 pairs (S_K1: only the first 5);
- nothing is written to seed_v1, mc0p_v1, mc0p_v1b; evaluate-one only to this batch's run; the old entries refuse the new roots;
- the disk floor: --min-host-gib in the new entries only, SEED's 26 GiB unchanged; a closed gate stops, what has finished is summed up;
- S_K1 does not run unless asked for; --only-s-k1 runs it without the entries of S_P;
- the reading: every branch, not decidable and missing pairs among them; S_P of the first 5 only from seed_v1;
- the start strings; the queues on a fake machine; the mc0p_v1c control; one real synthetic cell through run-one, evaluate and the summary, with every first-5 guard switched off.
"""
import csv
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

from .test_mc0p_tools import FROZEN, MI, PY, ROOT, Machine, core, decision, mc, needs_frozen, pipeline_world, RUN as MC0P  # noqa: F401
from .test_seed_tools import DROP, LONG_CONFIG, FakeSystem, fake_levels, metric_values, predicted_sources, sha, small_field, write_json

sys.path.insert(0, str(ROOT/'experiments/seed'))

import decide_seed as DEC  # noqa: E402
import decide_seed19 as D19  # noqa: E402
import run_mc0 as MC0  # noqa: E402
import run_mc0pc as C  # noqa: E402
import run_seed as SEED  # noqa: E402
import run_seed19 as S19  # noqa: E402

REAL_S19_SYSTEM, REAL_MC0_SYSTEM = S19.System, MC0.System
REST, FIRST = S19.REST_PAIRS, S19.FIRST_PAIRS
ALL = FIRST+REST
SHAPE = (8, 9, 10)
RESULTS = Path('/made/up/results')


class NoRealMachine:
    def __init__(self, *args, **kwargs):
        raise AssertionError('a queue on the real machine inside a test: a guard has regressed')


@pytest.fixture(autouse=True)
def no_real_machine(monkeypatch):
    """Every queue here runs on a fake machine, passed in.  A write guard or a parser that regresses must fail the test, never start a real queue: on 2026-11-04 a mutation of the old entry's guard
    (`run_seed.py queue --root <temporary>/seed_v1b`) went on to the real System and started a real solver unit (runs/20261104-seed19/plan.md revision 1)."""
    for module in (SEED, S19, MC0):
        monkeypatch.setattr(module, 'System', NoRealMachine)


# ---- a synthetic results tree -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def make_inputs(results):
    root = results/S19.INPUTS
    files = {}
    for pair in ALL:
        d = root/'stage_L2R_val_v1'/pair
        d.mkdir(parents=True)
        np.savez(d/'pair.npz', fixed=np.zeros(SHAPE), moving=np.zeros(SHAPE))
        fixed, moving = pair.split('__')
        write_json(d/'manifest.json', dict(pair=pair, pair_sha256=sha(d/'pair.npz'), fixed_subject=fixed, moving_subject=moving, shape_zyx=list(SHAPE), preprocessing_d0=dict(arm='A5P', oracle=False, source_pair_sha256='orig'+pair, source_manifest_sha256='origm'+pair, recipe='synthetic')))
        files[f'stage_L2R_val_v1/{pair}/pair.npz'] = sha(d/'pair.npz')
        files[f'stage_L2R_val_v1/{pair}/manifest.json'] = sha(d/'manifest.json')
    write_json(root/'d0_inputs.json', dict(arm='A5P', oracle=False, files_sha256=files))


def make_entry(results, the_arm, the_pair, *, seed=1, directory=None, **overrides):
    arm, pair = the_arm, the_pair
    spec = S19.ARMS[arm]
    d = S19.entry_dir(arm, pair, results) if directory is None else directory
    d.mkdir(parents=True, exist_ok=True)
    u = small_field(seed)
    np.savez(d/'field.npz', u=u)
    fixed, moving = pair.split('__')
    record = dict(schema=S19.MC0P_SCHEMA, config=dict(LONG_CONFIG), run_id=spec['source_prefix']+core.short_name(pair), arm=spec['source_arm'], tier='long', pair=pair, k=spec['k'], oracle=False, status='done', seconds_total=61., code_sha='x', channel_sources=predicted_sources(),
                  field=dict(file='field.npz', sha256=sha(d/'field.npz'), u_sha256=hashlib.sha256(u.tobytes()).hexdigest()), data=dict(pair_sha256='orig'+pair, manifest_sha256='origm'+pair, fixed_subject=fixed, moving_subject=moving))
    record.update(overrides)
    write_json(d/'run.json', {k: v for k, v in record.items() if v is not DROP})
    write_json(d/'metrics.json', dict(tapered_u_sha256=SEED.boundary_taper_sha256(u, u.shape[:3]), metrics=dict(seg35=.86, seg4=.88, sdlogj_official=.05), at_accepted_alpha=dict(alpha=1., seg35=.86, seg4=.88), certificate=dict(accepted_alpha=1.)))
    return d


def seed_block(results, arm='S_P'):
    return dict(passed=True, problems=[], accepted_alpha=1., entry_D_at_accepted_alpha=2000., last_D=300., arm=dict(seconds=SEED.SECONDS, changes=SEED.CHANGES, inputs_root=str(S19.inputs_root(arm, results)), charge_seconds=SEED.ENTRY_CHARGE_SECONDS))


def s_p_run(results, seg35=.85, seg4=.9, sdlogj=.08, arm='S_P', pair=REST[0], **changes):
    run = dict(run_id=f'seedb_{arm}_{core.short_name(pair)}', arm=arm, pair=pair, method='m', classification={'class': 'ok', 'kind': 'none', 'reason': ''}, status='native_budget_exhausted', metrics=metric_values(seg35, seg4, sdlogj), native_commits=75, e2e_seconds=327.,
               certificate=dict(passed=True, status='certified_feasible', min_jd_lower_bound=1e-4, state_hash_matches_u=True, final_npz_sha256_matches=True), self_check=dict(passed=True), seed=seed_block(results, arm))
    run.update(changes)
    return {k: v for k, v in run.items() if v is not DROP}


def control_run(seg35=D19.PINS['B1TM'][0], seg4=D19.PINS['B1TM'][1], sdlogj=.08):
    return dict(run_id='b1', metrics=metric_values(seg35, seg4, sdlogj), certificate=dict(passed=True, min_jd_lower_bound=1e-4), native_commits=85, e2e_seconds=327.)


def make_controls(results, b1tm=D19.PINS['B1TM']):
    """b1_v1: B1TM and B1W_v2 on the 19 pairs: the run's metrics.json, its record, analysis/long.csv."""
    b1 = results/'b1_v1'
    rows = []
    for arm, (seg35, seg4) in dict(B1TM=b1tm, B1W_v2=(.85, .91)).items():
        for pair in ALL:
            run_id = f'b1_{arm}_{core.short_name(pair)}'
            m = dict(run_id=run_id, arm=arm, pair=pair, method='b1', metrics=metric_values(seg35, seg4), certificate=dict(passed=True, min_jd_lower_bound=1e-4), self_check=dict(passed=True), native_commits=85, e2e_seconds=327.,
                     b1=dict(seconds=SEED.SECONDS, changes=SEED.CHANGES, inputs_root=str(S19.inputs_root('S_P', results))))
            write_json(b1/'runs'/arm/pair/'metrics.json', m)
            for key in ('seg35', 'seg4', 'dice_official', 'hd95', 'sdlogj_official', 'sdlogj_brain', 'num_foldings_official', 'pct_J_lt_half_brain', 'cell_centre_J_nonpositive'):
                rows.append(dict(set='L2R_val', split='dev', method='b1', arm=arm, pair=pair, measure=key, value=m['metrics'][key], run_id=run_id))
            for key, value in (('cert_passed', 1), ('cert_min_jd_lower_bound', 1e-4), ('self_check_passed', 1), ('native_commits', 85), ('e2e_seconds', 327.)):
                rows.append(dict(set='L2R_val', split='dev', method='b1', arm=arm, pair=pair, measure=key, value=value, run_id=run_id))
    (b1/'analysis').mkdir(parents=True, exist_ok=True)
    with open(b1/'analysis/long.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['set', 'split', 'method', 'arm', 'pair', 'measure', 'value', 'run_id'])
        w.writeheader()
        w.writerows(rows)


def make_first5(results, means=D19.FIRST5_PIN):
    for pair in FIRST:
        write_json(results/S19.FIRST_ROOT/'runs/S_P'/pair/'metrics.json', s_p_run(results, means[0], means[1], pair=pair, run_id=f'seed_S_P_{core.short_name(pair)}', method='seed_v1'))


def make_results(base, entries=True):
    results = Path(base)/'results'
    make_inputs(results)
    make_controls(results)
    make_first5(results)
    if entries:
        for k, pair in enumerate(REST):
            make_entry(results, 'S_P', pair, seed=k)
        for k, pair in enumerate(FIRST):
            make_entry(results, 'S_K1', pair, seed=100+k)
    return results


@pytest.fixture
def neutral():
    """A temporary directory whose path does not hold the word of the test runner: check_launch would refuse the start strings under tmp_path, as it should."""
    with tempfile.TemporaryDirectory(prefix='seed19_tools_') as d:
        yield Path(d)


def tree(root):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(root).rglob('*')) if p.is_file()}


# ---- the plan and the map -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_14_pairs_are_the_briefs_and_both_entries_agree():
    assert len(REST) == 14 and REST == C.REST_PAIRS
    assert core.short_name(REST[0]) == '0443-0444' and core.short_name(REST[-1]) == '0456-0457'
    assert not set(REST) & set(FIRST) and FIRST == core.PAIRS
    selection = core.MAIN_REPO/'data/three_d/stage_L2R_val_v1/selection.json'
    if selection.is_file():
        assert tuple(row['pair'] for row in json.loads(selection.read_text())['pairs']) == ALL


def test_the_subjects_are_on_the_development_allowlist_and_off_the_training_and_holdout_lists():
    subjects = {side for pair in ALL for side in pair.split('__')}
    assert len(subjects) == 20
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')
    for pair in ALL:
        check_pair_subjects(*pair.split('__'), allowlist)
    for name in ('holdout_B_subjects.txt', 'train_subjects.txt'):
        listed = set((ROOT/'splits/bayesreg_v1'/name).read_text().split())
        assert listed and not subjects & listed, name


def test_the_plan_of_mc0p_v1c_is_the_control_and_the_14_pairs():
    plan = C.plan_runs()
    assert len(plan) == 15 and plan[0] == C.CONTROL and C.CONTROL.short == '0438-0439'
    assert [c.pair for c in plan[1:]] == list(REST) and {(c.arm, c.tier) for c in plan} == {('Q2', 'long')}
    assert plan[1].run_id == 'mc0p_Q2_long_0443-0444' and plan[1].unit == 'tag-mc0pc-q2-long-0443-0444'
    assert core.run_dir('/r', plan[1]) == Path('/r/runs/Q2_long/0443-0444')
    assert not plan[1].unit.startswith('seq-vp-')


@pytest.mark.parametrize('arm, tier, pair', (('Q0', 'long', REST[0]), ('Q2', 'fine', REST[0]), ('Q2', 'long', FIRST[1]), ('Q2', 'long', 'OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1')))
def test_mc0p_v1c_refuses_other_cells_before_any_file(arm, tier, pair):
    with pytest.raises(SystemExit, match='refused'):
        C.require_cell(arm, tier, pair)


@pytest.mark.parametrize('name', (S19.FIRST_ROOT, 'mc0p_v1', 'mc0p_v1b', 'other'))
def test_mc0p_v1c_writes_only_to_its_root(name, capsys, monkeypatch, neutral):
    monkeypatch.setattr(C, 'DEFAULT_ROOT', neutral/'mc0p_v1c')
    target = neutral/name
    target.mkdir()
    (target/'x').write_text('kept')
    before = tree(target)
    for argv in (['queue', '--root', str(target)], ['evaluate', '--root', str(target)], ['summarize', '--root', str(target)]):
        assert C.main(argv) == 2
        assert 'writes only to' in capsys.readouterr().err
    with pytest.raises(SystemExit, match='only to'):
        C.main(['run-one', '--arm', 'Q2', '--tier', 'long', '--pair', REST[0], '--out', str(target/'runs/Q2_long/0443-0444')])
    assert tree(target) == before and not (neutral/'mc0p_v1c').exists()


def test_seed19_plan_is_s_p_on_14_and_s_k1_only_with_the_switch_and_then_first():
    assert S19.plan_runs() == [('S_P', p) for p in REST]
    assert S19.plan_runs(True) == [('S_K1', p) for p in FIRST]+[('S_P', p) for p in REST]        # with the switch S_K1 comes first
    assert S19.run_dir('/r', 'S_P', REST[0]) == Path('/r/runs/S_P')/REST[0] and S19.run_dir('/r', 'S_K1', FIRST[0]) == Path('/r/runs/S_K1')/FIRST[0]
    assert S19.unit_name('S_P', REST[0]) == 'tag-seedb-S_P-0443-0444' and S19.new_run_id('S_P', REST[0]).startswith('seedb_S_P_0443-0444_')
    assert S19.method_name('S_P') == SEED.method_name('S_P') and S19.method_name('S_K1').endswith('entryQ0long')


@pytest.mark.parametrize('arm, pair', [('S_P', p) for p in FIRST]+[('S_K1', REST[0]), ('S_T', REST[0]), ('S_P', 'OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1')])
def test_seed19_refuses_other_cells_before_any_file(arm, pair):
    for call in (lambda: S19.run_dir('/r', arm, pair), lambda: S19.input_files(arm, pair, '/nonexistent'), lambda: S19.entry_dir(arm, pair, '/nonexistent'), lambda: S19.prepare_argv(arm, pair, '/r/x', '/nonexistent')):
        with pytest.raises(SystemExit, match='SEED19'):
            call()


def test_the_entry_map_is_fixed():
    assert S19.entry_dir('S_P', REST[3], '/r') == Path('/r/mc0p_v1c/runs/Q2_long')/core.short_name(REST[3])
    assert S19.entry_dir('S_K1', FIRST[3], '/r') == Path('/r/mc0p_v1/runs/Q0_long')/core.short_name(FIRST[3])
    assert S19.ARMS['S_P']['k'] == 35 and S19.ARMS['S_K1']['k'] == 1 and S19.ORACLE is False
    assert S19.ARMS['S_P']['inputs'] == SEED.ARMS['S_P']['inputs'] == 'd0_v1_a5/inputs/A5P'


def files_of(results, arm='S_P', pair=REST[0]):
    return S19.input_files(arm, pair, results)[3]


def test_a_good_entry_is_taken(tmp_path):
    results = make_results(tmp_path)
    e = S19.check_entry('S_P', REST[0], files_of(results), results)
    assert e['source_root'] == 'mc0p_v1c' and e['source_k'] == 35 and e['oracle'] is False and e['charge_seconds'] == 47. and e['source_run_id'] == 'mc0p_Q2_long_0443-0444' and not e['retry_of']
    k1 = S19.check_entry('S_K1', FIRST[0], files_of(results, 'S_K1', FIRST[0]), results)
    assert k1['source_root'] == 'mc0p_v1' and k1['source_k'] == 1 and k1['source_arm'] == 'Q0'
    assert set(SEED.inject_run.ENTRY_KEYS) <= set(e)                 # what the wrapper reads


@pytest.mark.parametrize('change, word', ((dict(oracle=True), 'oracle'), (dict(oracle=None), 'oracle'), (dict(status='cuda_oom'), 'status'), (dict(k=1), 'K'), (dict(arm='Q2s'), 'arm'), (dict(tier='fine'), 'tier'), (dict(pair=REST[1]), 'pair'), (dict(run_id='mc0p_Q2_long_0444-0445'), 'run_id'),
                                          (dict(retry_of='/x/mc0p_v1'), 'retry_of'), (dict(schema='bayesreg.mc0.run.v1'), 'schema'), (dict(channel_sources=DROP), 'channel_sources'), (dict(channel_sources=predicted_sources(truth_labels_read=True)), 'truth_labels_read'),
                                          (dict(config=dict(LONG_CONFIG, iterations=[50, 50, 100, 60])), 'long tier'), (dict(data=dict(pair_sha256='orig'+REST[1], manifest_sha256='origm'+REST[1], fixed_subject='a', moving_subject='b')), 'original pair')))
def test_a_wrong_record_is_refused(tmp_path, change, word):
    results = make_results(tmp_path)
    make_entry(results, 'S_P', REST[0], **change)
    with pytest.raises(S19.EntryError, match=word):
        S19.check_entry('S_P', REST[0], files_of(results), results)


def test_a_field_of_the_wrong_pair_is_refused(tmp_path):
    """The record and the field of the next pair put in this pair's directory."""
    results = make_results(tmp_path)
    source = S19.entry_dir('S_P', REST[1], results)
    target = S19.entry_dir('S_P', REST[0], results)
    shutil.rmtree(target)
    shutil.copytree(source, target)
    with pytest.raises(S19.EntryError, match='pair'):
        S19.check_entry('S_P', REST[0], files_of(results), results)


@pytest.mark.parametrize('root', ('mc0p_v1', 'mc0p_v1b'))
def test_an_entry_under_another_root_is_not_seen(tmp_path, root):
    """The path has no switch: the same pair under another batch is simply not there."""
    results = make_results(tmp_path, entries=False)
    make_entry(results, 'S_P', REST[0], directory=results/root/'runs/Q2_long'/core.short_name(REST[0]))
    with pytest.raises(S19.EntryMissing, match='入口待 mc0p_v1c'):
        S19.check_entry('S_P', REST[0], files_of(results), results)


def test_s_k1_takes_only_q0_of_the_first_batch(tmp_path):
    results = make_results(tmp_path)
    for change, word in ((dict(k=35), 'K'), (dict(arm='Q2'), 'arm'), (dict(oracle=True), 'oracle')):
        make_entry(results, 'S_K1', FIRST[0], **change)
        with pytest.raises(S19.EntryError, match=word):
            S19.check_entry('S_K1', FIRST[0], files_of(results, 'S_K1', FIRST[0]), results)


def test_sha256_shape_and_taper_are_checked(tmp_path):
    results = make_results(tmp_path)
    d = make_entry(results, 'S_P', REST[0])
    np.savez(d/'field.npz', u=small_field(77))
    with pytest.raises(S19.EntryError, match='sha256'):
        S19.check_entry('S_P', REST[0], files_of(results), results)
    d = make_entry(results, 'S_P', REST[0])
    write_json(d/'metrics.json', dict(tapered_u_sha256='other'))
    with pytest.raises(S19.EntryError, match='tapered'):
        S19.check_entry('S_P', REST[0], files_of(results), results)
    d = make_entry(results, 'S_P', REST[0])
    u = np.zeros((4, 4, 4, 3))
    np.savez(d/'field.npz', u=u)
    write_json(d/'run.json', dict(json.loads((d/'run.json').read_text()), field=dict(file='field.npz', sha256=sha(d/'field.npz'), u_sha256='x')))
    with pytest.raises(S19.EntryError, match='shape'):
        S19.check_entry('S_P', REST[0], files_of(results), results)


def test_inputs_that_are_not_a5p_are_refused(tmp_path):
    results = make_results(tmp_path, entries=False)
    d = results/S19.INPUTS/'stage_L2R_val_v1'/REST[0]
    np.savez(d/'pair.npz', fixed=np.ones(SHAPE))
    with pytest.raises(S19.EntryError, match='differ'):
        S19.input_files('S_P', REST[0], results)


# ---- the disk floor ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Disk(FakeSystem):
    def __init__(self, mnt_c):
        super().__init__()
        self.gib = {'/home': 500., '/mnt/c': mnt_c}


@pytest.mark.parametrize('free, old, new', ((14.9, True, True), (15., True, False), (25.9, True, False), (26., False, False)))
def test_the_floor_of_the_new_entry_and_the_unchanged_one_of_seed(tmp_path, free, old, new):
    old_reason = SEED.busy_reason(Disk(free), tmp_path)
    new_reason = SEED.busy_reason(Disk(free), tmp_path, queue_unit=S19.QUEUE_UNIT, host_floor_gib=S19.DEFAULT_MIN_HOST_GIB)
    assert bool(old_reason) is old and bool(new_reason) is new
    if old:
        assert old_reason == '/mnt/c free < 10 GiB + 16 GiB pending'
    if new:
        assert new_reason == '/mnt/c free < 15.0 GiB'


def test_seed_wait_for_gate_keeps_26(tmp_path):
    with pytest.raises(SEED.Stop, match='10 GiB \\+ 16 GiB pending'):
        SEED.wait_for_gate(Disk(20.), tmp_path, 0, lambda text: None)


def test_seed_keeps_its_gate_and_has_no_floor_option(capsys):
    assert SEED.STORAGE_RESERVE_GIB+SEED.PENDING_GIB == 26
    with pytest.raises(SystemExit):
        SEED.main(['queue', '--min-host-gib', '15'])
    assert 'unrecognized arguments' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        MC0P.main(['queue', '--min-host-gib', '15'])
    assert 'unrecognized arguments' in capsys.readouterr().err


def test_the_queue_units_of_the_gates_are_their_own(tmp_path):
    machine = Disk(100.)
    machine.unit_list = [f'{SEED.QUEUE_UNIT}.service']
    assert SEED.busy_reason(machine, tmp_path) is None
    assert 'other units' in SEED.busy_reason(machine, tmp_path, queue_unit=S19.QUEUE_UNIT, host_floor_gib=15.)
    machine.unit_list = [f'{S19.QUEUE_UNIT}.service']
    assert 'other units' in SEED.busy_reason(machine, tmp_path)
    assert SEED.busy_reason(machine, tmp_path, queue_unit=S19.QUEUE_UNIT, host_floor_gib=15.) is None


class Plain(Machine):
    def host_free_gib(self):
        raise AssertionError('a gate without a floor does not ask for /mnt/c')


class HostMachine(Machine):
    def __init__(self, host=100.):
        super().__init__()
        self.host = host

    def host_free_gib(self):
        return self.host


def test_the_mc0_gates_do_not_ask_for_the_disk_and_mc0p_v1c_does(tmp_path):
    for gate in (MC0.MC0_GATE, MC0P.GATE, MC0P.RETRY_GATE):
        assert gate.host_floor_gib is None
    machine = Plain()
    machine.free_gpu = 24000.
    assert MC0.busy_reason(machine, tmp_path, True) is None
    assert MC0.busy_reason(machine, tmp_path, False, MC0P.RETRY_GATE) is None
    gate = C.replace(C.GATE, host_floor_gib=15.)
    high, low = HostMachine(15.), HostMachine(14.9)
    assert (gate.load_limit, gate.gpu_free_mib, gate.mem_available_gib, gate.gpu_hold) == (1.5, 15*1024, 7., None) and gate.own_units == ('tag-mc0pc-queue.service',)
    high.free_gpu = low.free_gpu = 24000.
    assert MC0.busy_reason(high, tmp_path, True, gate) is None
    assert MC0.busy_reason(low, tmp_path, True, gate) == '/mnt/c free < 15.0 GiB'


def test_the_real_disk_reading_keeps_the_gate_closed_on_an_error(monkeypatch):
    def fail(path):
        raise OSError('no reading')
    monkeypatch.setattr(MC0.shutil, 'disk_usage', fail)
    assert REAL_MC0_SYSTEM().host_free_gib() == 0.


# ---- the start strings ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_start_strings_hold_no_forbidden_word_and_the_right_names():
    argv = S19.queue_launch()
    text = shlex.join(argv)
    assert 'pytest' not in text and '--unit=tag-seedb-queue' in text and 'run_seed19.py queue' in text and '--min-host-gib 15' in text and '--root /home/rmcfz/seq_vp_tag_results/seed_v1b' in text
    assert '--include-s-k1' not in text and '--include-s-k1' in shlex.join(S19.queue_launch(include_s_k1=True))
    argv = C.queue_launch()
    text = shlex.join(argv)
    assert 'pytest' not in text and '--unit=tag-mc0pc-queue' in text and 'run_mc0pc.py queue --root /home/rmcfz/seq_vp_tag_results/mc0p_v1c' in text and '--min-host-gib 15' in text
    root = C.DEFAULT_ROOT
    for line in [C.run_argv(c, root/'code', root, 'sha') for c in C.plan_runs()]+[C.eval_argv(root/'code', root)]+[MC0P.k1_argv(root/'code', root, C.K1_UNIT)]:
        assert 'pytest' not in shlex.join(str(a) for a in line) and any(str(a).startswith('--unit=tag-mc0pc-') for a in line)
    for arm, pair in S19.plan_runs(True):
        run = S19.run_dir(S19.default_root(), arm, pair)
        for line in (S19.unit_argv(arm, pair, run, S19.default_root()), S19.prepare_argv(arm, pair, run)):
            assert 'pytest' not in shlex.join(str(a) for a in line)
        assert S19.unit_argv(arm, pair, run, S19.default_root())[2] == f'--unit=tag-seedb-{arm}-{core.short_name(pair)}'
    with pytest.raises(ValueError, match='forbidden'):
        S19.unit_argv('S_P', REST[0], '/x/pytest-1/run', '/x')
    with pytest.raises(ValueError, match='forbidden'):
        C.run_argv(C.plan_runs()[1], '/x/pytest-1/code', '/x')


def test_the_units_of_the_run_are_b1tms_but_for_the_name_and_the_old_default_stays():
    """S_P here against S_P of the first 5 pairs: the same resources, environment, watchdog, options."""
    new = S19.unit_argv('S_P', REST[0], '/r/x', '/r')
    old = SEED.unit_argv('S_P', FIRST[0], '/r/x', '/r')
    assert [a for a in new if not a.startswith('--unit=')] == [a for a in old if not a.startswith('--unit=')] and old[2] == '--unit=tag-seed-S_P-0438-0439'
    assert S19.prepare_argv('S_P', REST[0], '/r/x')[:-1] == SEED.prepare_argv('S_P', FIRST[0], '/r/x')[:-1]
    assert S19.prepare_argv('S_K1', FIRST[0], '/r/x')[-3:] == ['SEED-S_K1', '--pair', FIRST[0]]
    args = C.run_argv(C.plan_runs()[1], '/r/code', '/r', 'sha')
    assert '--property=MemoryMax=8G' in args and '--nice=5' in args and 'retry' not in ' '.join(args)


def test_the_optimizer_and_the_tier_are_mc0ps():
    assert core.tier_config('long')['iterations'] == (50, 50, 100, 600) and core.tier_config('long')['seconds'] == 900. and mc.VRAM_LIMIT_BYTES == 9*2**30 and mc.ARMS['Q2']['k'] == 35 and mc.ARMS['Q2']['D_factor'] == 1.
    assert SEED.ENTRY_CHARGE_SECONDS == 47. and SEED.SECONDS == 330 and SEED.GEOMETRY == [1., 1., .1, .01]


def test_the_scale_rule_reads_no_optimized_field_of_q0():
    """Q0's ||g_D|| and the level norms come from the images at u = 0: nothing in these functions opens a field or a Q0 run directory."""
    import inspect
    for fn in (mc.prepare_stacks, core.data_grad_norm, core.level_grad_norms, core.p0_level_norms):
        source = inspect.getsource(fn)
        assert 'field.npz' not in source and 'run_dir' not in source and 'np.load' not in source
    assert 'torch.zeros' in inspect.getsource(core.level_grad_norms)


# ---- the reading -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def reading_runs(deltas, **changes):
    """S_P of the 14 pairs at B1TM + delta, B1TM at the pin."""
    runs = {('B1TM', p): control_run() for p in ALL}
    for pair, delta in zip(REST, deltas):
        if delta is not None:
            runs[('S_P', pair)] = s_p_run(RESULTS, D19.PINS['B1TM'][0]+delta, D19.PINS['B1TM'][1]+delta/2, pair=pair, **changes)
    return runs


def code_of(deltas, **changes):
    return D19.reading_one(reading_runs(deltas, **changes), [], RESULTS)


@pytest.mark.parametrize('deltas, code', (([.006]*14, D19.REPLICATED),                                  # well above, 14/14
                                          ([.0085]*11+[-.001]*3, D19.REPLICATED),                         # 11/14 is enough
                                          ([.012]*10+[-.003]*4, D19.MIDDLE),                              # the mean above, 10/14: not enough to replicate, more than 7
                                          ([.004]*14, D19.MIDDLE),                                        # between the bars, 14/14
                                          ([.0019]*14, D19.NOT_REPLICATED),                               # the bars themselves: test_the_bars_of_reading_one_are_included
                                          ([.001]*14, D19.NOT_REPLICATED),
                                          ([.05]*7+[-.001]*7, D19.NOT_REPLICATED),                        # at most 7/14, whatever the mean
                                          ([.05]*8+[-.001]*6, D19.MIDDLE),                                # 8/14 is more than 7 and less than 11
                                          ([0.]*14, D19.NOT_REPLICATED)))                                 # strictly positive
def test_the_table_of_reading_one(deltas, code):
    assert code_of(deltas)['code'] == code


def test_positive_is_strictly_above_zero():
    assert code_of([.0075]*10+[0.]*4)['code'] == D19.MIDDLE              # the mean above the bar, 10 positive: a difference of exactly 0 is not positive
    assert code_of([.0075]*11+[0.]*3)['code'] == D19.REPLICATED


def test_the_bars_of_reading_one_are_included(monkeypatch):
    runs = reading_runs([.005]*14)
    assert D19.reading_one(runs, [], RESULTS)['delta_seg35']['positive'] == 14
    monkeypatch.setattr(D19, 'paired', lambda *a, **k: dict(per_pair={}, mean=.005, median=.005, positive=11, n=14))
    assert D19.reading_one(runs, [], RESULTS)['code'] == D19.REPLICATED
    monkeypatch.setattr(D19, 'paired', lambda *a, **k: dict(per_pair={}, mean=.002, median=.002, positive=14, n=14))
    assert D19.reading_one(runs, [], RESULTS)['code'] == D19.NOT_REPLICATED
    monkeypatch.setattr(D19, 'paired', lambda *a, **k: dict(per_pair={}, mean=.00200001, median=.002, positive=14, n=14))
    assert D19.reading_one(runs, [], RESULTS)['code'] == D19.MIDDLE
    monkeypatch.setattr(D19, 'paired', lambda *a, **k: dict(per_pair={}, mean=.00499999, median=.005, positive=14, n=14))
    assert D19.reading_one(runs, [], RESULTS)['code'] == D19.MIDDLE


@pytest.mark.parametrize('n_missing, code', ((0, D19.REPLICATED), (1, D19.REPLICATED), (2, D19.REPLICATED), (3, D19.NOT_DECIDABLE), (14, D19.NOT_DECIDABLE)))
def test_at_least_12_pairs_with_a_certified_final_field(n_missing, code):
    result = code_of([.01]*(14-n_missing)+[None]*n_missing)
    assert result['code'] == code and result['n'] == 14-n_missing and len(result['excluded']) == n_missing


def test_with_missing_pairs_the_positive_count_stays_absolute():
    assert code_of([.01]*10+[-.001]*2+[None]*2)['code'] == D19.MIDDLE                # 10 positive, 12 counted: not 11
    assert code_of([.01]*11+[-.001]*1+[None]*2)['code'] == D19.REPLICATED
    assert code_of([.03]*7+[-.001]*5+[None]*2)['code'] == D19.NOT_REPLICATED


@pytest.mark.parametrize('change, word', ((dict(classification={'class': 'handover_only'}), '分类'), (dict(native_commits=0), '原生提交'), (dict(metrics=None), '无终态场'), (dict(certificate=dict(passed=False)), '终态证书'), (dict(seed=dict(passed=False, problems=['x'], arm={})), '入口核对')))
def test_a_run_that_does_not_count_on_three_pairs_makes_it_not_decidable(change, word):
    runs = reading_runs([.01]*14)
    for pair in REST[:3]:
        runs[('S_P', pair)] = s_p_run(RESULTS, .9, .95, pair=pair, **change)
    result = D19.reading_one(runs, [], RESULTS)
    assert result['code'] == D19.NOT_DECIDABLE and result['n'] == 11 and word in json.dumps(result['excluded'], ensure_ascii=False)


def test_a_run_of_another_configuration_does_not_count():
    runs = reading_runs([.01]*14)
    block = seed_block(RESULTS)
    block['arm'] = dict(block['arm'], charge_seconds=61.)
    runs[('S_P', REST[0])] = s_p_run(RESULTS, .9, .95, pair=REST[0], seed=block)
    result = D19.reading_one(runs, [], RESULTS)
    assert result['n'] == 13 and '配置' in json.dumps(result['excluded'], ensure_ascii=False)


def test_sdlogj_over_the_bar_is_not_decidable_and_at_the_bar_is_read(monkeypatch):
    runs = reading_runs([.01]*14)
    for pair in REST:
        runs[('S_P', pair)]['metrics']['sdlogj_official'] = .1001
    assert D19.reading_one(runs, [], RESULTS)['code'] == D19.NOT_DECIDABLE
    for pair in REST:
        runs[('S_P', pair)]['metrics']['sdlogj_official'] = .125
    monkeypatch.setattr(D19, 'SDLOGJ_MAX', .125)                       # a number the mean of 14 equal values gives exactly: the bar itself is read
    assert D19.reading_one(runs, [], RESULTS)['code'] == D19.REPLICATED


def test_a_reference_problem_and_a_control_without_certificate_are_not_decidable():
    runs = reading_runs([.01]*14)
    assert D19.reading_one(runs, ['made up'], RESULTS)['code'] == D19.NOT_DECIDABLE
    for pair in REST[:3]:
        runs[('B1TM', pair)] = dict(control_run(), certificate=dict(passed=False))
    assert D19.reading_one(runs, [], RESULTS)['code'] == D19.NOT_DECIDABLE


def target_runs(seg35=.86, sdlogj=.10, n=19):
    runs = {('B1TM', p): control_run() for p in ALL}
    for pair in ALL[:n]:
        runs[('S_P', pair)] = s_p_run(RESULTS, seg35, .9, sdlogj, pair=pair)
    return runs


def test_the_bars_of_the_target_are_included(monkeypatch):
    monkeypatch.setattr(D19, 'TARGET', .875)                           # numbers the mean of 19 equal values gives exactly
    monkeypatch.setattr(D19, 'SDLOGJ_MAX', .125)
    assert D19.reading_two(target_runs(.875, .125), [], RESULTS)['code'] == D19.REACHED


@pytest.mark.parametrize('seg35, sdlogj, code', ((.8601, .0999, D19.REACHED), (.8599, .08, D19.NOT_REACHED), (.87, .1001, D19.NOT_REACHED)))
def test_the_target_on_the_19_pairs(seg35, sdlogj, code):
    result = D19.reading_two(target_runs(seg35, sdlogj), [], RESULTS)
    assert result['code'] == code and result['n'] == 19 and result['certificates'] == 19


def test_not_reached_says_by_how_much():
    result = D19.reading_two(target_runs(.85, .08), [], RESULTS)
    assert result['gaps']['seg35'] == pytest.approx(.01) and result['gaps']['sdlogj_official'] == 0. and '差 seg35 0.0100' in result['text'] and result['stretch_gap'] == pytest.approx(.03)


@pytest.mark.parametrize('change, word', ((dict(certificate=dict(passed=False)), '终态证书'), (dict(certificate=dict(passed=True, state_hash_matches_u=False, final_npz_sha256_matches=True)), '终态证书'), (dict(metrics=None), '无终态场'),
                                          (dict(seed=dict(passed=False, problems=['x'], arm={})), '入口核对')))
def test_a_run_without_a_certified_final_field_is_a_missing_pair(change, word):
    """Plan revision 1: not 'not reached': only the count."""
    runs = target_runs(.87, .08)
    runs[('S_P', REST[0])] = s_p_run(RESULTS, .87, .9, .08, pair=REST[0], **change)
    result = D19.reading_two(runs, [], RESULTS)
    assert result['code'] == D19.NOT_JUDGED and result['n'] == 18 and result['gaps'] is None and '18/19' in result['text'] and word in result['missing'][core.short_name(REST[0])]


def test_reading_two_checks_the_charge_too():
    runs = target_runs(.87, .08)
    block = seed_block(RESULTS)
    block['arm'] = dict(block['arm'], charge_seconds=61.)
    runs[('S_P', REST[0])] = s_p_run(RESULTS, .87, .9, .08, pair=REST[0], seed=block)
    result = D19.reading_two(runs, [], RESULTS)
    assert result['code'] == D19.NOT_JUDGED and result['n'] == 18 and '配置' in result['missing'][core.short_name(REST[0])]


def test_a_missing_pair_gives_only_the_count():
    result = D19.reading_two(target_runs(.90, .08, n=18), [], RESULTS)
    assert result['code'] == D19.NOT_JUDGED and result['n'] == 18 and result['gaps'] is None and '18/19' in result['text'] and '不判' in result['text']
    assert D19.reading_two(target_runs(.90, .08), ['seed_v1 made up'], RESULTS)['code'] == D19.NOT_JUDGED


def k1_runs(s_k1_minus_b1tm, s_p_minus_s_k1):
    base = D19.PINS['B1TM'][0]
    runs = {('B1TM', p): control_run() for p in FIRST}
    for k, pair in enumerate(FIRST):
        a, b = (s_k1_minus_b1tm[k], s_p_minus_s_k1[k])
        runs[('S_K1', pair)] = s_p_run(RESULTS, base+a, .9, arm='S_K1', pair=pair)
        runs[('S_P', pair)] = s_p_run(RESULTS, base+a+b, .9, pair=pair)
    return runs


@pytest.mark.parametrize('a, b, code', (([.0]*5, [.006]*5, D19.K1_CHANNELS),                           # S_P - S_K1 >= .005, 5/5
                                        ([.0]*5, [.0125]*4+[-.01]*1, D19.K1_CHANNELS),                  # 4/5 is enough
                                        ([.0]*5, [.03]*3+[-.01]*2, D19.K1_MIDDLE),                      # the mean is high, 3/5
                                        ([.006]*5, [.0019]*5, D19.K1_STEPS),                            # S_K1 - B1TM >= .005, S_P - S_K1 <= .002
                                        ([.006]*5, [.003]*5, D19.K1_MIDDLE),                            # between
                                        ([.004]*5, [.001]*5, D19.K1_MIDDLE)))                           # S_K1 - B1TM below the bar
def test_the_reading_of_s_k1(tmp_path, a, b, code):
    assert D19.reading_s_k1(k1_runs(a, b), tmp_path, [], RESULTS)['code'] == code


@pytest.mark.parametrize('a, b, positive, code', ((.005, .0, 4, D19.K1_CHANNELS), (.00499999, .005, 5, D19.K1_MIDDLE), (.002, .005, 5, D19.K1_STEPS), (.00200001, .005, 5, D19.K1_MIDDLE), (.002, .00499999, 5, D19.K1_MIDDLE)))
def test_the_bars_of_s_k1_are_included(tmp_path, monkeypatch, a, b, positive, code):
    runs = k1_runs([.0]*5, [.006]*5)
    monkeypatch.setattr(D19, 'paired', lambda runs, arm, ref, key, pairs: dict(per_pair={}, mean=a if (arm, ref) == ('S_P', 'S_K1') else b, median=0., positive=positive, n=5))
    assert D19.reading_s_k1(runs, tmp_path, [], RESULTS)['code'] == code


def test_s_k1_not_run_is_not_read_and_a_partial_one_is_not_decidable(tmp_path):
    assert D19.reading_s_k1({}, tmp_path, [], RESULTS)['code'] == D19.K1_NOT_RUN
    runs = k1_runs([.0]*5, [.006]*5)
    del runs[('S_K1', FIRST[2])]
    result = D19.reading_s_k1(runs, tmp_path, [], RESULTS)
    assert result['code'] == D19.K1_NOT_DECIDABLE and core.short_name(FIRST[2]) in result['text']
    (tmp_path/'runs/S_K1'/FIRST[0]).mkdir(parents=True)                 # started, nothing evaluated
    assert D19.reading_s_k1({}, tmp_path, [], RESULTS)['code'] == D19.K1_NOT_DECIDABLE


# ---- the written analysis ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def write_trace(run, D_entry=2000., D_final=300.):
    task = Path(run)/'pg_weak__four__330s'
    write_json(Path(run)/'protocol.json', dict(task=dict(name=task.name)))
    write_json(task/'levels.json', [dict(event='level_prepared', level=3, visible_seconds=50., transfer_recovery=dict(accepted_alpha=.9)), dict(event='level_completed', level=3, visible_seconds=325.)])
    write_json(task/'history.json', [dict(level=3, outer_accepted=True, D_start=D_entry, D_candidate=D_entry-10.), dict(level=3, outer_accepted=True, D_start=D_entry-10., D_candidate=D_final)])


def analysis_tree(base, delta=.01, n=14):
    results = make_results(base)
    root = results/'seed_v1b'
    for pair in REST[:n]:
        write_json(root/'runs/S_P'/pair/'metrics.json', s_p_run(results, D19.PINS['B1TM'][0]+delta, D19.PINS['B1TM'][1]+delta/2, pair=pair, method=S19.method_name('S_P')))
        write_trace(root/'runs/S_P'/pair)
    return results, root


def test_the_written_analysis_reports_seg35_and_seg4_and_keeps_the_earlier_version(tmp_path):
    results, root = analysis_tree(tmp_path)
    result = D19.write_analysis(root, results)
    assert result['reading_one']['code'] == D19.REPLICATED and result['reading_one']['n'] == 14 and result['reference_problems'] == [] and result['first5_problems'] == []
    assert result['reading_two']['code'] in (D19.REACHED, D19.NOT_REACHED) and result['reading_two']['n'] == 19 and result['reading_s_k1']['code'] == D19.K1_NOT_RUN and result['described_s_k1'] == []
    text = (root/'analysis/summary_seed19.md').read_text()
    assert f"19 对 seg35 差 {D19.STRETCH-result['reading_two']['seg35']:.4f}" in (root/'analysis/summary_seed19.md').read_text() and '.88（只描述，不判）' in (root/'analysis/summary_seed19.md').read_text()
    for word in ('其余 14 对复现', 'seg4', '小六个', '皮质下', 'CSF', 'A5Pv2', '0439-0440', '0440-0441', '转移 α', '入口 D', '终态 D', '原生层 s', 'α = 1', '不是留出集'):
        assert word in text, word
    stored = json.loads((root/'analysis/decision_seed19.json').read_text())
    assert stored['reading_one']['code'] == D19.REPLICATED and stored['schema'] == D19.SCHEMA
    row = next(r for r in stored['described']['rows'] if r['pair'] == '0443-0444')
    assert row['alpha'] == .9 and row['D_entry'] == 2000. and row['D_final'] == 300. and row['native_seconds'] == 275. and row['entry']['root'] == 'mc0p_v1c' and row['entry']['seg35'] == .86
    assert [r['pair'] for r in stored['described']['subject_0440']] == ['0439-0440', '0440-0441']
    assert stored['described']['means']['all_19']['B1W_v2']['n'] == 19 and stored['described']['against_a5pv2']['rest_14']['seg35']['n'] == 14
    with open(root/'analysis/long.csv', newline='') as f:
        ids = {row['run_id'] for row in csv.DictReader(f)}
    assert any(i.startswith('seedb_') for i in ids) and any(i.startswith('seed_S_P') for i in ids)
    D19.write_analysis(root, results)
    assert any(p.name.startswith('previous_') for p in (root/'analysis').iterdir())


def test_the_first_5_come_only_from_seed_v1_and_seed_v1_is_not_touched(tmp_path):
    results, root = analysis_tree(tmp_path)
    before = tree(results/S19.FIRST_ROOT)
    plain = D19.write_analysis(root, results)['reading_two']
    for pair in FIRST:                                                  # decoys under this batch's root: must not be read for S_P
        write_json(root/'runs/S_P'/pair/'metrics.json', s_p_run(results, .99, .99, pair=pair, run_id='decoy'))
    again = D19.write_analysis(root, results)
    assert again['reading_two']['seg35'] == plain['seg35'] and 'decoy' not in {r['run_id'] for r in again['described']['rows']}
    assert tree(results/S19.FIRST_ROOT) == before
    assert D19.seed_root_of(FIRST[0], root, results) == results/S19.FIRST_ROOT and D19.seed_root_of(REST[0], root, results) == root


def test_the_reading_before_any_run_is_not_decidable_and_counts_5(tmp_path):
    results = make_results(tmp_path, entries=False)
    root = results/'seed_v1b'
    result = D19.write_analysis(root, results)
    assert result['reading_one']['code'] == D19.NOT_DECIDABLE and result['reading_two']['code'] == D19.NOT_JUDGED and result['reading_two']['n'] == 5


def test_a_baseline_that_is_not_the_pinned_one_is_not_decidable(tmp_path):
    results, root = analysis_tree(tmp_path)
    make_controls(results, b1tm=(.85, .9083))
    result = D19.write_analysis(root, results)
    assert result['reading_one']['code'] == D19.NOT_DECIDABLE and any('钉住' in p for p in result['reference_problems'])


def test_a_baseline_of_another_configuration_is_not_decidable(tmp_path):
    results, root = analysis_tree(tmp_path)
    path = results/'b1_v1/runs/B1TM'/REST[0]/'metrics.json'
    doc = json.loads(path.read_text())
    doc['b1']['seconds'] = 90
    path.write_text(json.dumps(doc))
    assert D19.write_analysis(root, results)['reading_one']['code'] == D19.NOT_DECIDABLE


def test_first_5_of_seed_v1_that_differ_from_the_record_are_not_judged(tmp_path):
    results, root = analysis_tree(tmp_path)
    make_first5(results, means=(.84, .8984))
    result = D19.write_analysis(root, results)
    assert result['first5_problems'] and result['reading_two']['code'] == D19.NOT_JUDGED and result['reading_one']['code'] == D19.REPLICATED       # reading one has only the 14 pairs


@pytest.mark.parametrize('name', (S19.FIRST_ROOT, 'mc0p_v1', 'mc0p_v1b', 'mc0p_v1c'))
def test_the_reading_refuses_to_write_under_the_batches_it_reads(tmp_path, name):
    results = make_results(tmp_path, entries=False)
    (results/name).mkdir(exist_ok=True)
    before = tree(results/name)
    with pytest.raises(SystemExit, match='only reads'):
        D19.write_analysis(results/name, results)
    with pytest.raises(SystemExit, match='only reads'):
        D19.write_analysis(results/name/'sub', results)
    assert tree(results/name) == before


def test_the_pins_and_thresholds_are_the_briefs():
    assert D19.PINS == dict(B1TM=(.8468, .9083)) and D19.PIN_TOL == DEC.PIN_TOL
    assert (D19.REPLICATE_BAR, D19.NOT_REPLICATE_BAR, D19.REPLICATE_POSITIVE, D19.NOT_REPLICATE_POSITIVE, D19.MIN_PAIRS, D19.SDLOGJ_MAX, D19.TARGET, D19.STRETCH) == (.005, .002, 11, 7, 12, .10, .86, .88)
    assert (D19.K1_BAR, D19.K1_NEAR_BAR, D19.K1_POSITIVE, D19.K1_N) == (.005, .002, 4, 5)


# ---- the queue of seed_v1b on a fake machine -------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Fake19(FakeSystem):
    def __init__(self, results):
        super().__init__()
        self.results = results

    def evaluate(self, run, arm, pair, run_id):
        if self.fail_eval:
            return 1
        trace, _ = SEED.injection_trace(run)
        write_json(Path(run)/'metrics.json', s_p_run(self.results, D19.PINS['B1TM'][0]+.01, D19.PINS['B1TM'][1]+.005, arm=arm, pair=pair, run_id=run_id, method=S19.method_name(arm), seed=dict(trace, arm=json.loads((Path(run)/'seed_arm.json').read_text()))))
        return 0


def queue_options(root, **changes):
    return SimpleNamespace(**dict(dict(root=str(root), wait_minutes=30, min_host_gib=S19.DEFAULT_MIN_HOST_GIB, include_s_k1=False, only_s_k1=False), **changes))


@pytest.fixture
def machine(neutral):
    results = make_results(neutral)
    return SimpleNamespace(results=results, root=results/'seed_v1b', repo=neutral/'no_repo', system=Fake19(results))


def run_queue(machine, **changes):
    return S19.queue(queue_options(machine.root, **changes), machine.system, machine.results, machine.repo)


def test_the_queue_runs_s_p_on_the_14_pairs_once_writes_the_reading_and_nothing_else(machine):
    others = {name: tree(machine.results/name) for name in (S19.FIRST_ROOT, 'mc0p_v1', 'mc0p_v1c')}
    assert run_queue(machine) == 0
    assert machine.system.launched == [S19.unit_name('S_P', p) for p in REST]
    assert not (machine.root/'runs/S_K1').exists()
    assert (machine.root/'runs/S_P'/REST[0]/'seed_inject.py').read_bytes() == (ROOT/'experiments/seed/inject_run.py').read_bytes()
    entry = json.loads((machine.root/'runs/S_P'/REST[0]/'seed_entry.json').read_text())
    assert '/mc0p_v1c/runs/Q2_long/0443-0444/field.npz' in entry['field_path'] and entry['oracle'] is False and entry['charge_seconds'] == 47.
    arm = json.loads((machine.root/'runs/S_P'/REST[0]/'seed_arm.json').read_text())
    assert arm['entry_root'] == 'mc0p_v1c' and arm['seconds'] == 330 and arm['charge_seconds'] == 47. and arm['input_oracle'] is False
    assert (machine.root/'runs/S_P'/REST[0]/'run_id').read_text().startswith('seedb_S_P_0443-0444_')
    decision_doc = json.loads((machine.root/'analysis/decision_seed19.json').read_text())
    assert decision_doc['reading_one']['code'] == D19.REPLICATED and decision_doc['reading_s_k1']['code'] == D19.K1_NOT_RUN
    assert {name: tree(machine.results/name) for name in others} == others
    assert 'S_K1 False' in (machine.root/'queue.log').read_text()
    launched = list(machine.system.launched)
    assert run_queue(machine) == 0 and machine.system.launched == launched          # a restart reruns nothing


def test_with_the_switch_s_k1_runs_first_in_its_own_directory(machine):
    assert run_queue(machine, include_s_k1=True) == 0
    assert machine.system.launched == [S19.unit_name('S_K1', p) for p in FIRST]+[S19.unit_name('S_P', p) for p in REST]
    entry = json.loads((machine.root/'runs/S_K1'/FIRST[0]/'seed_entry.json').read_text())
    assert '/mc0p_v1/runs/Q0_long/0438-0439/field.npz' in entry['field_path'] and entry['source_k'] == 1
    assert json.loads((machine.root/'analysis/decision_seed19.json').read_text())['reading_s_k1']['code'] != D19.K1_NOT_RUN
    assert json.loads((machine.root/'batch.json').read_text())['plan'][0] == 'S_K1 0438-0439' and json.loads((machine.root/'batch.json').read_text())['include_s_k1'] is True
    stored = json.loads((machine.root/'analysis/decision_seed19.json').read_text())
    assert [r['pair'] for r in stored['described_s_k1']] == list(map(core.short_name, FIRST)) and stored['described_s_k1'][0]['entry']['root'] == 'mc0p_v1' and stored['described_s_k1'][0]['entry']['seg35'] == .86
    text = (machine.root/'analysis/summary_seed19.md').read_text()
    assert '只描述：S_K1' in text and 'seg4（只描述）' in text


def test_the_switch_on_a_later_start_adds_only_s_k1_and_batch_json_says_so(machine):
    assert run_queue(machine) == 0
    assert run_queue(machine, include_s_k1=True) == 0
    assert machine.system.launched == [S19.unit_name('S_P', p) for p in REST]+[S19.unit_name('S_K1', p) for p in FIRST]
    batch = json.loads((machine.root/'batch.json').read_text())
    assert batch['include_s_k1'] is True and batch['plan'][0] == 'S_K1 0438-0439' and [s['include_s_k1'] for s in batch['starts']] == [False, True] and len(batch['starts'][0]['plan']) == 14


@pytest.mark.parametrize('key, value', (('charge_seconds', 61.), ('seconds', 240), ('frozen_commit', 'other'), ('entry_roots', dict(S_P='mc0p_v1b', S_K1='mc0p_v1')), ('schema', 'other')))
def test_a_restart_on_another_batch_stops_before_any_run(machine, key, value):
    assert run_queue(machine) == 0
    path = machine.root/'batch.json'
    doc = json.loads(path.read_text())
    doc[key] = value
    path.write_text(json.dumps(doc))
    launched = list(machine.system.launched)
    assert run_queue(machine, include_s_k1=True) == 8
    assert 'batch.json differs' in json.loads((machine.root/'queue_stop.json').read_text())['reason'] and machine.system.launched == launched


def test_only_s_k1_is_off_by_default_and_excludes_the_other_switch(capsys):
    assert S19.plan_runs(only_s_k1=True) == [('S_K1', p) for p in FIRST]
    with pytest.raises(SystemExit):
        S19.plan_runs(True, True)
    with pytest.raises(SystemExit):
        S19.main(['queue', '--include-s-k1', '--only-s-k1'])
    assert 'not allowed with' in capsys.readouterr().err
    assert '--only-s-k1' in shlex.join(S19.queue_launch(only_s_k1=True)) and '--only-s-k1' not in shlex.join(S19.queue_launch()) and '--only-s-k1' not in shlex.join(S19.queue_launch(include_s_k1=True))


def test_only_s_k1_runs_without_the_entries_of_s_p_and_then_s_p_can_follow(neutral):
    results = make_results(neutral, entries=False)
    for k, pair in enumerate(FIRST):
        make_entry(results, 'S_K1', pair, seed=100+k)
    machine = SimpleNamespace(results=results, root=results/'seed_v1b', repo=neutral/'no_repo', system=Fake19(results))
    assert run_queue(machine, include_s_k1=True) == 8                  # the default rule stands: an entry of S_P missing, nothing runs
    assert not machine.system.launched and '入口待 mc0p_v1c' in json.loads((machine.root/'queue_stop.json').read_text())['reason']
    assert run_queue(machine) == 8 and not machine.system.launched
    assert run_queue(machine, only_s_k1=True) == 0
    assert machine.system.launched == [S19.unit_name('S_K1', p) for p in FIRST] and not (machine.root/'runs/S_P').exists()
    decision_doc = json.loads((machine.root/'analysis/decision_seed19.json').read_text())
    assert decision_doc['reading_s_k1']['code'] != D19.K1_NOT_RUN and decision_doc['reading_one']['code'] == D19.NOT_DECIDABLE and decision_doc['reading_two']['code'] == D19.NOT_JUDGED
    batch = json.loads((machine.root/'batch.json').read_text())
    assert batch['only_s_k1'] is True and len(batch['plan']) == 5
    for k, pair in enumerate(REST):                                     # the entries come later: S_P alone follows, S_K1 stays as it ran
        make_entry(results, 'S_P', pair, seed=k)
    launched = list(machine.system.launched)
    assert run_queue(machine) == 0
    assert machine.system.launched == launched+[S19.unit_name('S_P', p) for p in REST]


def test_only_s_k1_still_checks_its_own_entries(neutral):
    results = make_results(neutral, entries=False)
    for k, pair in enumerate(FIRST[:4]):
        make_entry(results, 'S_K1', pair, seed=100+k)
    make_entry(results, 'S_K1', FIRST[4], oracle=True)
    machine = SimpleNamespace(results=results, root=results/'seed_v1b', repo=neutral/'no_repo', system=Fake19(results))
    assert run_queue(machine, only_s_k1=True) == 8
    assert not machine.system.launched and 'oracle' in json.loads((machine.root/'queue_stop.json').read_text())['reason']


@pytest.mark.parametrize('name', (S19.FIRST_ROOT, 'mc0p_v1', 'mc0p_v1b', 'mc0p_v1c'))
def test_the_queue_refuses_every_other_root_before_touching_it(machine, name, capsys):
    target = machine.results/name
    target.mkdir(exist_ok=True)
    before = tree(target)
    assert S19.queue(queue_options(target), machine.system, machine.results, machine.repo) == 2
    assert 'writes only to' in capsys.readouterr().err and tree(target) == before and not (target/'queue.lock').exists() and not machine.system.launched


def test_a_missing_entry_stops_before_any_run_and_says_what_is_missing(neutral):
    results = make_results(neutral, entries=False)
    machine = SimpleNamespace(results=results, root=results/'seed_v1b', repo=neutral/'no_repo', system=Fake19(results))
    assert run_queue(machine) == 8
    assert not machine.system.launched and '入口待 mc0p_v1c' in json.loads((machine.root/'queue_stop.json').read_text())['reason'] and not (machine.root/'runs').exists()


def test_a_missing_first_5_or_control_stops_before_any_run(machine):
    (machine.results/S19.FIRST_ROOT/'runs/S_P'/FIRST[0]/'metrics.json').unlink()
    (machine.results/'b1_v1/runs/B1TM'/REST[0]/'metrics.json').unlink()
    assert run_queue(machine) == 8
    reason = json.loads((machine.root/'queue_stop.json').read_text())['reason']
    assert 'seed_v1 S_P 0438-0439' in reason and 'control B1TM 0443-0444' in reason and not machine.system.launched


def test_a_gate_below_the_floor_waits_stops_and_sums_up_what_finished(machine):
    system = machine.system

    class Falling(Fake19):
        """/mnt/c falls under the floor after the second run."""
        def free_gib(self, path):
            return 14. if path == '/mnt/c' and len(self.launched) >= 2 else 500.
    falling = Falling(machine.results)
    machine.system = falling
    assert system is not falling
    assert run_queue(machine) == 8
    assert len(falling.launched) == 2 and falling.sleeps.count(SEED.POLL_SECONDS) == 60
    stop = json.loads((machine.root/'queue_stop.json').read_text())
    assert '/mnt/c free < 15.0 GiB' in stop['reason']
    result = json.loads((machine.root/'analysis/decision_seed19.json').read_text())
    assert result['reading_one']['code'] == D19.NOT_DECIDABLE and result['reading_two']['code'] == D19.NOT_JUDGED and result['reading_two']['n'] == 7
    assert sorted(p.name for p in (machine.root/'runs/S_P').iterdir()) == sorted(REST[:2])
    machine.system = Fake19(machine.results)                             # the disk back: only the rest run; the old stop stays under its time
    assert run_queue(machine) == 0
    assert machine.system.launched == [S19.unit_name('S_P', p) for p in REST[2:]]
    assert list(machine.root.glob('queue_stop_*.json')) and not (machine.root/'queue_stop.json').exists()


def test_a_lower_floor_on_the_command_line_opens_the_gate(machine):
    class Low(Fake19):
        def free_gib(self, path):
            return 12. if path == '/mnt/c' else 500.
    machine.system = Low(machine.results)
    assert run_queue(machine, min_host_gib=11.) == 0 and len(machine.system.launched) == 14
    batch = json.loads((machine.root/'batch.json').read_text())
    assert batch['min_host_gib'] == 11. and batch['include_s_k1'] is False and batch['only_s_k1'] is False


def test_a_stop_that_is_no_gate_writes_no_reading(machine):
    machine.system.bad_trace = {S19.unit_name('S_P', REST[0])}
    assert run_queue(machine) == 8
    assert 'not injected as recorded' in json.loads((machine.root/'queue_stop.json').read_text())['reason'] and not (machine.root/'analysis').exists()


def test_three_infrastructure_failures_stop(machine):
    machine.system.classes = {S19.unit_name('S_P', p): 'infrastructure' for p in REST[:3]}
    assert run_queue(machine) == 8
    assert 'infrastructure' in json.loads((machine.root/'queue_stop.json').read_text())['reason'] and len(machine.system.launched) == 3


class EvaluationStub:
    def __init__(self):
        self.calls = []

    def evaluate_run(self, run, arm, pair, run_id, method=None):
        self.calls.append((str(run), arm, pair))
        return dict(metrics=None)


@pytest.mark.parametrize('name', (S19.FIRST_ROOT, 'mc0p_v1', 'mc0p_v1b', 'mc0p_v1c', 'other'))
def test_evaluate_one_refuses_every_run_outside_this_batch(neutral, monkeypatch, capsys, name):
    """The counterexample of review important 1: seed_v1's metrics.json must stay."""
    monkeypatch.setattr(SEED, 'RESULTS', neutral)
    stub = EvaluationStub()
    monkeypatch.setattr(SEED, 'load_evaluate', lambda: stub)
    for arm, pair in (('S_K1', FIRST[0]), ('S_P', REST[0])):
        run = neutral/name/'runs'/arm/pair
        run.mkdir(parents=True)
        (run/'metrics.json').write_text('{"original": true}')
        assert S19.main(['evaluate-one', str(run), '--arm', arm, '--pair', pair, '--run-id', 'x']) == 2
        assert 'writes only to' in capsys.readouterr().err and json.loads((run/'metrics.json').read_text()) == dict(original=True)
    assert not stub.calls


def test_evaluate_one_refuses_another_arm_or_pair_inside_this_batch_and_takes_its_own(neutral, monkeypatch, capsys):
    monkeypatch.setattr(SEED, 'RESULTS', neutral)
    stub = EvaluationStub()
    monkeypatch.setattr(SEED, 'load_evaluate', lambda: stub)
    run = S19.run_dir(neutral/'seed_v1b', 'S_P', REST[0])
    fake_levels(run, dict(u_sha256='a', tapered_sha256='b'))
    write_json(run/'seed_arm.json', dict(arm='S_P'))
    assert S19.main(['evaluate-one', str(run), '--arm', 'S_P', '--pair', REST[1], '--run-id', 'x']) == 2
    assert S19.main(['evaluate-one', str(run), '--arm', 'S_K1', '--pair', FIRST[0], '--run-id', 'x']) == 2
    assert not stub.calls and not (run/'metrics.json').exists()
    with pytest.raises(SystemExit, match='SEED19'):
        S19.main(['evaluate-one', str(run), '--arm', 'S_P', '--pair', FIRST[0], '--run-id', 'x'])
    assert S19.main(['evaluate-one', str(run), '--arm', 'S_P', '--pair', REST[0], '--run-id', 'x']) == 0
    assert stub.calls == [(str(run), 'S_P', REST[0])] and (run/'metrics.json').is_file()


def test_the_step_of_the_queue_passes_the_guard_of_evaluate_one(neutral, monkeypatch):
    """The command line System.evaluate starts, handed to main: the queue's own step is not refused."""
    monkeypatch.setattr(SEED, 'RESULTS', neutral)
    stub = EvaluationStub()
    monkeypatch.setattr(SEED, 'load_evaluate', lambda: stub)
    run = S19.run_dir(S19.default_root(), 'S_K1', FIRST[0])
    fake_levels(run, dict(u_sha256='a', tapered_sha256='b'))
    write_json(run/'seed_arm.json', dict(arm='S_K1'))
    started = []
    with monkeypatch.context() as patch:
        patch.setattr(S19.subprocess, 'run', lambda argv, **kwargs: started.append(argv) or SimpleNamespace(returncode=0))
        assert REAL_S19_SYSTEM().evaluate(run, 'S_K1', FIRST[0], 'seedb_x') == 0
    argv = [str(a) for a in started[0]]
    assert argv[2].endswith('run_seed19.py') and S19.main(argv[3:]) == 0 and stub.calls == [(str(run), 'S_K1', FIRST[0])]


@pytest.mark.parametrize('command', ('queue', 'decide'))
def test_the_old_seed_entry_refuses_the_new_root(neutral, capsys, command):
    root = neutral/'seed_v1b'
    assert SEED.main([command, '--root', str(root)]) == 2
    assert 'run_seed19.py' in capsys.readouterr().err and not root.exists()


@pytest.mark.parametrize('command', ('queue', 'evaluate', 'decide'))
def test_the_old_mc0p_entry_refuses_the_new_root(neutral, capsys, command):
    root = neutral/'mc0p_v1c'
    assert MC0P.main([command, '--root', str(root)]) == 2
    assert 'run_mc0pc.py' in capsys.readouterr().err and not root.exists()


def test_evaluate_one_uses_the_evaluation_and_adds_the_entry_record(neutral, monkeypatch):
    run = neutral/'run'
    fake_levels(run, dict(u_sha256='a', tapered_sha256='b'))
    write_json(run/'seed_arm.json', dict(arm='S_P'))
    calls = []

    class Ev:
        def evaluate_run(self, the_run, arm, pair, run_id, method=None):
            calls.append((arm, pair, run_id, method))
            return dict(metrics=None)
    monkeypatch.setattr(SEED, 'load_evaluate', lambda: Ev())
    assert S19.evaluate_one(SimpleNamespace(run=str(run), arm='S_P', pair=REST[0], run_id='seedb_x')) == 0
    assert calls == [('S_P', REST[0], 'seedb_x', S19.method_name('S_P'))]
    stored = json.loads((run/'metrics.json').read_text())
    assert stored['seed']['passed'] is True and stored['seed']['arm'] == dict(arm='S_P')
    with pytest.raises(SystemExit):
        S19.evaluate_one(SimpleNamespace(run=str(run), arm='S_P', pair=FIRST[0], run_id='x'))


# ---- the queue of mc0p_v1c on a fake machine -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
FIELD = np.arange(6.).reshape(1, 1, 2, 3)


class CMachine(HostMachine):
    """A quiet machine; run units write what a real run would, the evaluation a metrics.json per field."""

    def __init__(self, control_root=None, *, behaviour=None, control_field=FIELD, host=100.):
        super().__init__(host)
        self.launched, self.argvs, self.clock, self.behaviour, self.control_field, self.sleeps = [], {}, 0., behaviour or {}, control_field, 0

    def units(self):
        return [f'{C.QUEUE_UNIT}.service']+list(self.other)

    def now(self):
        return self.clock

    def sleep(self, seconds):
        self.clock += seconds
        self.sleeps += 1

    def code_checkout(self, root, commit):
        return Path(root)/'code', 'fakesha'

    def run(self, argv, log):
        unit = next(a for a in argv if a.startswith('--unit=')).split('=', 1)[1]
        self.launched.append(unit)
        self.argvs[unit] = argv
        out = Path(argv[argv.index('--out')+1]) if '--out' in argv else None
        if unit == C.K1_UNIT:
            write_json(out, dict(bitwise=self.behaviour.get('k1', True)))
            return 0
        if unit == C.EVAL_UNIT:
            root = Path(argv[argv.index('--root')+1])
            for cell in C.plan_runs():
                d = core.run_dir(root, cell)
                if (d/'field.npz').is_file():
                    write_json(d/'metrics.json', dict(metrics=dict({m: .5 for m in decision.MEASURES}, seg35=.8, seg4=.85, sdlogj_official=.05), at_accepted_alpha=dict(alpha=1., seg35=.8, seg4=.85), certificate=dict(accepted_alpha=1., n_infeasible_at_1=0, status_at_1='certified_feasible')))
            return self.behaviour.get('eval_rc', 0)
        self.clock += 61.
        pair = argv[argv.index('--pair')+1]
        how = self.behaviour.get(pair, 'done')
        if how == 'no_record':
            return 1
        if how in ('host_low',):
            self.host = 1.
            how = 'done'
        record = dict(run_id=f'mc0p_Q2_long_{core.short_name(pair)}', arm='Q2', tier='long', pair=pair, k=35, oracle=False, status=how, reason=None, scale=dict(images_equal_a5p_inputs=True, D_multiplier=2., D_factor=1.), device_peak_reserved_gib=6., seconds_total=61.)
        if how == 'not_a5p':
            record['status'], record['scale'] = 'input_mismatch', dict(images_equal_a5p_inputs=False)
        if how == 'oracle':
            record['status'], record['oracle'] = 'done', None
        if record['status'] == 'done':
            np.savez(out/'field.npz', u=self.control_field if pair == C.CONTROL.pair else np.arange(6.).reshape(1, 1, 2, 3)+1.)
        write_json(out/'run.json', record)
        return 0


@pytest.fixture
def c_world(neutral, monkeypatch):
    root, v1b = neutral/'mc0p_v1c', neutral/'mc0p_v1b'
    monkeypatch.setattr(C, 'DEFAULT_ROOT', root)
    monkeypatch.setattr(C, 'CONTROL_ROOT', v1b)
    d = core.run_dir(v1b, C.CONTROL)
    d.mkdir(parents=True)
    np.savez(d/'field.npz', u=FIELD)
    return SimpleNamespace(root=root, v1b=v1b)


def c_options(root, **changes):
    return SimpleNamespace(**dict(dict(root=str(root), commit=None, wait_minutes=30, min_host_gib=C.DEFAULT_MIN_HOST_GIB), **changes))


def test_the_queue_of_mc0p_v1c_runs_the_15_and_leaves_v1b_alone(c_world):
    before = tree(c_world.v1b)
    system = CMachine()
    assert C.queue(c_options(c_world.root), system) == 0
    assert system.launched == [C.K1_UNIT]+[c.unit for c in C.plan_runs()]+[C.EVAL_UNIT]
    assert tree(c_world.v1b) == before
    control = json.loads((c_world.root/'control.json').read_text())
    assert control['control_bitwise'] is True and control['u_sha256_v1b'] == control['u_sha256_v1c']
    batch = json.loads((c_world.root/'batch.json').read_text())
    assert batch['gate']['host_floor_gib'] == 15. and batch['gate']['load_limit'] == 1.5 and batch['plan'][0] == C.CONTROL.run_id and len(batch['plan']) == 15 and batch['schema'] == C.BATCH_SCHEMA
    end = json.loads((c_world.root/'queue_end.json').read_text())
    assert end['eval_rc'] == 0 and len(end['ledger']['states']) == 15 and end['control']['control_bitwise'] is True
    text = (c_world.root/'analysis/summary_mc0pc.md').read_text()
    assert 'mc0p_Q2_long_0456-0457' in text and 'control_bitwise True' in text and '接受 α' in text
    with open(c_world.root/'analysis/long.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    assert {r['oracle'] for r in rows} == {'false'} and {r['arm'] for r in rows} == {'Q2'}
    launched = list(system.launched)
    assert C.queue(c_options(c_world.root), system) == 0
    assert system.launched[len(launched):] == [C.EVAL_UNIT]                 # a restart reruns no cell


def test_a_control_that_differs_is_recorded_and_stops_nothing(c_world):
    system = CMachine(control_field=FIELD+1.)
    assert C.queue(c_options(c_world.root), system) == 0
    control = json.loads((c_world.root/'control.json').read_text())
    assert control['control_bitwise'] is False and control['u_sha256_v1b'] != control['u_sha256_v1c'] and len(system.launched) == 17


def test_a_control_out_of_memory_stops_no_arm(c_world):
    system = CMachine(behaviour={C.CONTROL.pair: 'cuda_oom'})
    assert C.queue(c_options(c_world.root), system) == 0
    done = json.loads((core.run_dir(c_world.root, C.CONTROL)/'queue_done.json').read_text())
    assert done['arm_stop'] is None and 'cuda_oom' in done['control_stop'] and len(system.launched) == 17
    control = json.loads((c_world.root/'control.json').read_text())
    assert control['control_bitwise'] is None and 'cuda_oom' in control['control_stop']


def test_an_out_of_memory_on_a_pair_stops_the_arm_and_says_what_is_missing(c_world):
    system = CMachine(behaviour={REST[2]: 'cuda_oom'})
    assert C.queue(c_options(c_world.root), system) == 0
    assert [u for u in system.launched if 'q2-long' in u] == [c.unit for c in C.plan_runs()[:4]]
    end = json.loads((c_world.root/'queue_end.json').read_text())
    assert 'cuda_oom' in end['ledger']['arms_stopped']['Q2']['reason'] and 'arm stopped' in end['end'] and 'cuda_oom' in end['end']
    assert 'arm stopped' in (c_world.root/'queue.log').read_text().splitlines()[-1] and 'arm stopped' in (c_world.root/'analysis/summary_mc0pc.md').read_text()
    assert '没有运行' in (c_world.root/'analysis/summary_mc0pc.md').read_text() and 'skip mc0p_Q2_long_0446-0447' in (c_world.root/'queue.log').read_text()


def test_images_that_are_not_a5ps_or_a_record_that_is_not_oracle_false_stop_without_evaluation(c_world):
    system = CMachine(behaviour={REST[0]: 'not_a5p'})
    assert C.queue(c_options(c_world.root), system) == 8
    assert C.EVAL_UNIT not in system.launched and 'A5P' in json.loads((c_world.root/'queue_stop.json').read_text())['reason']


def test_a_record_with_another_oracle_mark_stops(c_world):
    system = CMachine(behaviour={REST[1]: 'oracle'})
    assert C.queue(c_options(c_world.root), system) == 8
    assert C.EVAL_UNIT not in system.launched and 'oracle' in json.loads((c_world.root/'queue_stop.json').read_text())['reason']


def test_a_k1_check_that_fails_stops_before_any_run(c_world):
    system = CMachine(behaviour=dict(k1=False))
    assert C.queue(c_options(c_world.root), system) == 8
    assert system.launched == [C.K1_UNIT]


def test_a_failure_without_a_record_stops_the_arm_at_the_first_pair(c_world):
    system = CMachine(behaviour={C.CONTROL.pair: 'no_record', REST[0]: 'no_record'})
    assert C.queue(c_options(c_world.root), system) == 0                   # a K = 35 failure without a record stops the arm at the first pair; the queue ends with the evaluation
    assert 'without a record' in json.loads((c_world.root/'queue_end.json').read_text())['ledger']['arms_stopped']['Q2']['reason']
    assert [u for u in system.launched if 'q2-long' in u] == [c.unit for c in C.plan_runs()[:2]]


def test_below_the_floor_it_waits_stops_and_evaluates_what_finished(c_world):
    system = CMachine(behaviour={REST[0]: 'host_low'})
    assert C.queue(c_options(c_world.root), system) == 8
    stop = json.loads((c_world.root/'queue_stop.json').read_text())
    assert '/mnt/c free < 15.0 GiB' in stop['reason'] and system.sleeps >= 58
    assert system.launched == [C.K1_UNIT]+[c.unit for c in C.plan_runs()[:2]]+[C.EVAL_UNIT]
    assert (core.run_dir(c_world.root, C.plan_runs()[1])/'metrics.json').is_file()
    end = json.loads((c_world.root/'queue_end.json').read_text())
    assert end['end'].startswith('STOP') and (c_world.root/'analysis/summary_mc0pc.md').is_file()
    system.host = 100.
    assert C.queue(c_options(c_world.root), system) == 0                   # the disk back: only the rest; the old stop stays under its time
    assert list(c_world.root.glob('queue_stop_*.json')) and not (c_world.root/'queue_stop.json').exists()
    assert system.launched.count(C.plan_runs()[1].unit) == 1 and len([u for u in system.launched if 'q2-long' in u]) == 15


def test_a_gate_closed_for_another_reason_does_not_evaluate(c_world):
    system = CMachine()
    system.other = ['tag-other.service']
    assert C.queue(c_options(c_world.root), system) == 8
    assert not system.launched and 'other units' in json.loads((c_world.root/'queue_stop.json').read_text())['reason']


def test_a_failed_evaluation_stops_with_the_summary_written(c_world):
    system = CMachine(behaviour=dict(eval_rc=1))
    assert C.queue(c_options(c_world.root), system) == 8
    assert (c_world.root/'analysis/summary_mc0pc.md').is_file() and 'tag-mc0pc-eval' in json.loads((c_world.root/'queue_stop.json').read_text())['reason']


def test_a_batch_at_other_code_stops(c_world):
    system = CMachine()
    c_world.root.mkdir()
    write_json(c_world.root/'batch.json', dict(code_sha='other', plan=[]))
    assert C.queue(c_options(c_world.root), system) == 8
    assert not system.launched


@pytest.mark.parametrize('call', (lambda: SEED.queue(SimpleNamespace(root='/nonexistent/seed_x', wait_minutes=30)), lambda: S19.queue(queue_options(Path('/nonexistent/seed_v1b'))), lambda: MC0.queue(SimpleNamespace(root='/nonexistent/mc0_x', commit=None, wait_minutes=30))))
def test_the_safety_net_of_this_file_holds(call):
    """No queue of these tests can reach the real machine."""
    with pytest.raises(AssertionError, match='real machine'):
        call()


def test_the_old_entries_still_refuse_the_14_pairs():
    with pytest.raises(SystemExit, match='refused'):
        mc.require_cell('Q2', 'long', REST[0])
    with pytest.raises(SystemExit, match='refused'):
        mc.read_images(REST[0])
    with pytest.raises(SystemExit, match='refused'):
        MC0.read_pair(REST[0], ())
    with pytest.raises(SystemExit, match='SEED runs only'):
        SEED.run_dir('/r', 'S_P', REST[0])
    assert MC0P.run_one.__kwdefaults__ == dict(cell=None, images=None) and MC0P.evaluate_all.__kwdefaults__ == dict(plan=None, read_pair=None)
    assert [c.pair for c in mc.plan_runs()] == [c.pair for c in mc.plan_runs() if c.pair in core.PAIRS] and len(mc.plan_runs()) == 45
    assert SEED.plan_runs() == [(a, p) for a in SEED.ARMS for p in SEED.PAIRS] and len(SEED.plan_runs()) == 10 and tuple(SEED.ARMS) == ('S_P', 'S_T')


# ---- one real synthetic cell, every first-5 guard off -------------------------------------------------------------------------------------------------------------------------------------------------------------
C_PIPELINE = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import mc0_core as core
core.frozen_api()
import mc0p_core as mc
import run_mc0pc as C
world = json.loads(sys.argv[2])
root = Path(world['root']).parent/'mc0p_v1c'
C.DEFAULT_ROOT = root
core.STAGE = Path(world['stage'])
core.SETUP_WORKERS = 1
core.TIERS.update(long=dict(iterations=(2, 2, 2, 4)))
mc.SEG_ROOT, mc.A5P_INPUTS = Path(world['seg_root']), Path(world['a5p'])
mi = core.make_inputs()
mc.default_unet = lambda: mi.UNetSource(pred_root=Path(world['pred']))
core.PAIRS = ()                                                 # the path of this batch never asks the first-5 guard
out = core.run_dir(root, C.CONTROL)
assert C.main(['run-one', '--arm', 'Q2', '--tier', 'long', '--pair', C.CONTROL.pair, '--out', str(out), '--device', 'cpu', '--code-sha', 'x']) == 0
assert C.main(['evaluate', '--root', str(root)]) == 0
assert C.main(['summarize', '--root', str(root)]) == 0
run = json.loads((out/'run.json').read_text())
assert run['status'] == 'done' and run['oracle'] is False and run['k'] == 35 and 'retry_of' not in run and run['run_id'] == 'mc0p_Q2_long_0438-0439' and run['scale']['images_equal_a5p_inputs'] is True
assert run['channel_sources']['truth_labels_read'] is False
metrics = json.loads((out/'metrics.json').read_text())
assert metrics['oracle'] is False and metrics['tapered_u_sha256']
print(json.dumps(dict(alpha=metrics['certificate']['accepted_alpha'])))
"""


@needs_frozen
def test_one_real_cell_through_the_new_entry(tmp_path, monkeypatch):
    world = pipeline_world(tmp_path, monkeypatch)
    env_ = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{FROZEN}:{ROOT}')
    done = subprocess.run([PY, '-B', '-c', C_PIPELINE, str(ROOT/'experiments/mc0'), json.dumps(world)], env=env_, cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:]+done.stderr[-3000:]
    root = tmp_path/'mc0p_v1c'
    assert (root/'analysis/summary_mc0pc.md').is_file() and (root/'analysis/long.csv').is_file()
    run = json.loads((core.run_dir(root, C.CONTROL)/'run.json').read_text())
    assert SEED.channel_sources_problems(run) == [] and run['schema'] == S19.MC0P_SCHEMA and not run.get('retry_of')       # what SEED19's entry check asks of a record of this path
    assert (run['arm'], run['tier'], run['k'], run['oracle'], run['status']) == ('Q2', 'long', 35, False, 'done') and run['field']['file'] == 'field.npz' and run['data']['pair_sha256']


# ---- the dry run of mc0p_v1c on stand-ins ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def c_dry_world(tmp_path):
    stage = tmp_path/'stage_L2R_val_v1'
    stage.mkdir()
    write_json(stage/'selection.json', dict(pairs=[dict(pair=p) for p in ALL]))
    a5p = tmp_path/'A5P'
    grid = (6, 7, 8)
    from jd_curl.three_d.nifti import file_sha256 as hashed
    subjects = set()
    for pair in REST:
        fixed, moving = pair.split('__')
        subjects |= {fixed, moving}
        for directory in (stage/pair, a5p/pair):
            directory.mkdir(parents=True)
            np.savez(directory/'pair.npz', fixed=np.zeros(grid), moving=np.zeros(grid))
        write_json(stage/pair/'manifest.json', dict(shape_zyx=list(grid), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm', pair_sha256=hashed(stage/pair/'pair.npz'), fixed_subject=fixed, moving_subject=moving, sources=dict(fixed=dict(sha256='a'*64), moving=dict(sha256='b'*64))))
    for subject in subjects:
        p = tmp_path/'pred'/subject
        p.mkdir(parents=True)
        np.savez(p/'probs36.npz', p=np.zeros((36,)+grid, dtype=np.float16))
        (p/'structure_groups.npz').write_bytes(b'x')
        write_json(p/'provenance.json', dict(segmenter_id='unet35_v1_ens', policy='all_folds_mean', test_labels_seen=False))
        (tmp_path/'atropos'/subject).mkdir(parents=True)
        write_json(tmp_path/'atropos'/subject/'provenance.json', {})
    v1b = tmp_path/'mc0p_v1b'
    write_json(core.run_dir(v1b, C.CONTROL)/'run.json', dict(status='done', field=dict(u_sha256='abc')))
    mi = core.make_inputs()
    return dict(root=tmp_path/'mc0p_v1c', stage=stage, a5p=a5p, seg_root=tmp_path/'atropos', control_root=v1b, unet=mi.UNetSource(pred_root=tmp_path/'pred'))


def test_the_dry_run_of_mc0p_v1c_prints_the_15_and_writes_nothing(neutral):
    world = c_dry_world(neutral)
    lines = []
    before = tree(neutral)
    assert C.dry_run(out=lines.append, **world) == 0
    text = '\n'.join(lines)
    for cell in C.plan_runs():
        assert cell.run_id in text and cell.unit in text
    assert 'problems: none' in text and "pairs 6..19 == the plan's 14 pairs: True" in text and "torch imported" in text and 'not needed' in text and '/mnt/c >= 15.0 GiB' in text
    assert tree(neutral) == before and not world['root'].exists()


def test_the_dry_run_of_mc0p_v1c_says_what_is_wrong(neutral):
    world = c_dry_world(neutral)
    shutil.rmtree(world['a5p']/REST[1])
    shutil.rmtree(world['control_root'])
    lines = []
    assert C.dry_run(out=lines.append, **world) == 8
    text = lines[-1]
    assert "A5P's input is missing" in text and 'the control has no finished run' in text
