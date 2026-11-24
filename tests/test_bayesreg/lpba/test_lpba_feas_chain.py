"""LP-C2 item 3 (contract C26.5-9, C26.8-5; prereg §7.2, §12 item 14; the review's I1, I2, minor 2-4): experiments/lpba/feas_chain.py with a stand-in for the units (no unit is
started, no GPU): each step's records (and the unit's end probe) are written by the stand-in in the synthetic world's feasibility roots, and the chain's stop conditions are read from
them -- the WM ratio band (also when atropos_lpba exits 8), both directions' QC fallback, method (1), the external arms' status and export check, the no-truth evaluation, the disk
(per subject x 34/2, per pair x 17/2), the unit's peak, the code commit every record names; the machine gate polled through holdout_common.wait_for on a test clock; a missing command
refuses before the first unit; new roots.  LP-C2 integration item 5: LP-D's check (a) and LP-B's queues wired (their entries' own start strings, waited; their batch records'
code commit; the per-pair units' end probes; the review's finalized outputs; the feasibility input lock)."""
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import types

import pytest

import bayesreg.io.sealed as S
from bayesreg.eval import lpba_eval as E

from . import _eval_world as EW

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments'/'lpba'))
import feas_chain as FC  # noqa: E402

lc, hc = EW.lc, EW.hc
PAIRS = ('LPBA40_S01__LPBA40_S02', 'LPBA40_S02__LPBA40_S01')
STEPS = FC.STEPS                    # integration item 5: every step has its command
GIB = 2**30
HEAD = 'c'*40
BY_NAME = {s.name: s for s in FC.STEPS}
PROBE_OUT = re.compile(r'/([a-z0-9_]+)_probe_end\.json')


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = EW.make(tmp_path, monkeypatch)
    monkeypatch.setattr(hc, 'disk_guard', lambda **kwargs: dict(passed=True, note='test', **kwargs))
    shell = tmp_path/'cgroup_shell'
    shell.write_text('0::/user.slice/user-1000.slice/session-3.scope\n')
    CGROUP[0] = shell
    yield w
    CGROUP[0] = None
    S._reset_cache()


def plenty(path):
    return types.SimpleNamespace(free=500*GIB)


def write(path, doc):
    return EW.write_json(path, doc)


class StandIn:
    """Writes each step's records as its entry would, with the given changes (keyed by step name): wm (subject -> ratio), method (pair -> (line status, failure class, QC dice)),
    arms (arm -> pair -> (status, suspicious)), eval_labels (label files opened), exit (step -> code), commit (step -> the code commit its record names), peak (step -> bytes or None),
    events (step -> memory.events or None); the per-pair units of LP-B's queues: inner (step -> dict(peak, max, events) changes); review (pair -> finalized entry changes or None to leave the
    pair out); drop (record names not written: 'devcheck_a', 'input_lock', 'batch:<step>', 'probe:<step>'); probe_max (step -> the probe's memory.max bytes); inner's 'schema'
    (the first unit's unit_end.json of another schema) and 'missing' (that many units write none).  The step is known by its end probe's path in the unit's command."""

    def __init__(self, *, wm=None, method=None, arms=None, eval_labels=(), exit=None, commit=None, peak=None, events=None, inner=None, review=None, drop=(), batch_commit=None,
                 devcheck_passed=True, probe_max=None):
        self.wm = dict(S01=1.02, S02=.97) if wm is None else wm
        self.method = method or {p: ('hx_certified', None, [.9, .88]) for p in PAIRS}
        self.arms = arms or {}
        self.eval_labels, self.exit, self.commit, self.peak, self.events, self.units = list(eval_labels), exit or {}, commit or {}, peak or {}, events or {}, []
        self.inner, self.review, self.drop, self.batch_commit, self.devcheck_passed = inner or {}, review or {}, set(drop), batch_commit or {}, devcheck_passed
        self.probe_max = probe_max or {}

    def batch(self, name, directory, **extra):
        if f'batch:{name}' not in self.drop:
            write(FC.places()['root']/'batches'/directory/'batch.json', dict(code=dict(commit=self.batch_commit.get(name, HEAD), clean=True), **extra))

    def inner_units(self, step, cells):
        """The in-unit supervisor's unit_end.json of each per-pair unit (xb_unit.py's), its probe's memory.max the formal per-pair unit's."""
        change = self.inner.get(step.name, {})
        limit = FC.limit_bytes(step.inner_memory())
        for k, cell in enumerate(cells[:len(cells)-change.get('missing', 0)]):
            probe = dict(peak_bytes=change.get('peak', GIB), max_bytes=change.get('max', limit), events=change.get('events', dict(oom=0, oom_kill=0)))
            schema = change.get('schema', FC.XB_UNIT_END_SCHEMA) if k == 0 else FC.XB_UNIT_END_SCHEMA
            write(cell/'unit'/'unit_end.json', dict(schema=schema, unit=f'tag-lp-feas-x-{cell.name}', probe=probe))

    def __call__(self, argv):
        unit = next(a.split('=', 1)[1] for a in argv if a.startswith('--unit='))
        name = PROBE_OUT.search(argv[-1]).group(1)
        step = BY_NAME[name]
        assert unit == f'tag-lp-feas-{step.tool}'
        self.units.append(unit)
        r = lc.roots('lpba40_feas')
        root = FC.places()['root']
        _, arms = FC._arm_table()
        code = dict(commit=self.commit.get(name, HEAD), clean=True)
        if name == 'prep':
            write(r.stage/lc.PREP_RECORD, dict(passed=True, pair_set='lpba40_feas', stage_summary=dict(pairs=[dict(pair=p) for p in PAIRS]), code=code))
        elif name == 'devcheck_a':
            if 'devcheck_a' not in self.drop:
                write(root.joinpath(*FC.DEVCHECK_A_RECORD), dict(passed=self.devcheck_passed, code=code))
        elif name == 'hi':
            write(r.hi_out/'hi_record.json', dict(passed=True, rows=[{}, {}], code=code))
        elif name == 'atropos':
            out = [s for s, x in self.wm.items() if not .8 <= x <= 1.25]
            write(r.atropos/'atropos_record.json', dict(passed=not out, wm={s: dict(wm_ratio=lc.number(x)) for s, x in self.wm.items()},
                                                        feasibility_stop=dict(stop=bool(out), subjects=out), code=code))
        elif name == 'method':
            for pair, (status, failure, dice) in self.method.items():
                cell = Path(arms['method']['record_dir'].format(pair=pair))
                write(cell/'terminal.json', dict(line_status=status, code_commit=code['commit']))
                write(cell/'output'/'line.json', dict(line_status=status, failure=None if failure is None else {'class': failure}))
                write(cell/'t88'/'run.json', dict(alignment_qc=[dict(dice=d, passed=d >= .8) for d in dice], side_refusals={}))
            self.batch(name, 'hb_method_20261119T000000Z', kind='method')
            self.inner_units(step, [root/'method'/p for p in PAIRS])
        elif name == 'review':
            pairs = {}
            for pair in PAIRS:
                entry = dict(line_status=self.method[pair][0], review=dict(path=f'/x/{pair}/review.json', sha256='0'*64))
                change = self.review.get(pair, {})
                if change is not None:
                    pairs[pair] = dict(entry, **change)
            write(E.finalized_path('lpba40_feas'), dict(pair_set='lpba40_feas', pairs=pairs, code=code))
            self.batch(name, 'hb_review_20261119T000000Z', kind='review')
            self.inner_units(step, [root/'review'/p for p in PAIRS])
        elif name == 'input_lock':
            if 'input_lock' not in self.drop:
                rows = [dict(kind='prediction', subject=s, path=f'/pred/{s}/{n}', sha256='1'*64) for s in lc.FEAS_SUBJECTS for n in ('probs36.npz', 'structure_groups.npz', 'provenance.json')]
                rows += [dict(kind='hi_record', subject=s, path=f'/hi/{s}.json', sha256='2'*64) for s in lc.FEAS_SUBJECTS]
                rows += [dict(kind='method_field', pair=p, path=f'/f/{p}', sha256='3'*64, bytes=1, line_status=self.method[p][0]) for p in PAIRS]
                write(root/FC.INPUT_LOCK_NAME, dict(schema='bayesreg.hb.input_lock.v1', pair_set='lpba40_feas', rows=rows))
        elif name in ('configc', 'ants_s', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg'):
            for pair in PAIRS:
                status, suspicious = self.arms.get(name, {}).get(pair, ('ok', False))
                doc = dict(status=status, code=code)
                if arms[name]['export_check'] is not None:
                    doc['export_check'] = dict(suspicious=suspicious)
                write(Path(arms[name]['record_dir'].format(pair=pair, arm=name))/'result.json', doc)
            if name == 'configc':
                self.batch(name, 'hb_configc_20261119T000000Z', kind='configc')
            else:
                self.batch(name, f'xb_20261119T00000{FC.XB_STEP_ARMS.index(name)}Z', arms=[name])
            self.inner_units(step, [root/name/p for p in PAIRS])
        elif name == 'eval':
            out = FC.places()['eval_out']
            write(out/'lpba_eval.json', dict(stopped=None, failed=[], not_reached=[], label_files_opened=self.eval_labels, truth_mode=False,
                                             record=dict(identity=dict(code=dict(head=code['commit'], clean=True)))))
            for pair in PAIRS:
                entry = dict(outcome='output', metrics=dict(sdlogj_official=.05, sdlogj_brain=.1), certificate=dict(passed=True), tet24=dict(x=1))
                write(out/'pairs'/f'{pair}.json', dict(pair=pair, complete=True, truth_read=False, grid=dict(shape_zyx=list(FC.FULL_GRID)), brain=dict(nodes=10),
                                                       arms={a: entry for a in arms}))
        peak = self.peak.get(name, GIB//2)
        events = self.events.get(name, dict(low=0, high=0, max=0, oom=0, oom_kill=0))
        if f'probe:{name}' not in self.drop:
            write(FC.probe_path(step), dict(peak_bytes=peak, max_bytes=self.probe_max.get(name, FC.limit_bytes(FC.memory_max_of(step))), events=events,
                                            unread={} if peak is not None else {'memory.peak': '读不到 (test)'}))
        return dict(returncode=self.exit.get(name, 0), stdout='', stderr='')


CHECKS = FC.CHECKS


def clean(repo=None):
    return dict(commit=HEAD, clean=True, status='')


def chain(stand_in, **kwargs):
    kwargs.setdefault('machine', lambda **kw: dict(passed=True))
    kwargs.setdefault('checks', CHECKS)
    kwargs.setdefault('cgroup', CGROUP[0])
    kwargs.setdefault('code_identity', clean)
    return FC.run_chain(steps=STEPS, runner=stand_in, usage=kwargs.pop('usage', plenty), log=lambda text: None, **kwargs)


CGROUP = [None]


def test_the_steps_and_their_units(world):
    names = [s.name for s in FC.STEPS]
    assert names[:4] == ['prep', 'devcheck_a', 'hi', 'atropos'] and names[-1] == 'eval'
    assert {'method', 'review', 'configc', 'ants_s', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg'} <= set(names)
    eval_step = FC.STEPS[-1]
    assert FC.places()['eval_out'].parent == E.eval_dir('lpba40_feas')
    argv = FC.unit_argv(eval_step, code=Path('/code'))
    # integration review minor 3: the evaluation's unit is lpba_eval.launch_line's (the hold-out's tag-hb-eval spec), not the chain's own
    assert '--unit=tag-lp-feas-eval' in argv and {'--property=MemorySwapMax=4G', '--property=RuntimeMaxSec=259200', '--property=CPUQuota=800%', '--property=MemoryMax=10G',
                                                  '--property=OOMPolicy=continue', '--nice=5', '--setenv=NUMBA_NUM_THREADS=2'} <= set(argv) and argv[-3:-1] == ['/bin/bash', '-c']
    assert argv[-1].startswith(f"{E.PY} -B -m bayesreg.eval.lpba_eval evaluate --pair-set lpba40_feas --out-dir {FC.places()['eval_out']}; ec=$?;")
    assert 'experiments/he/unit_probe.py' in argv[-1] and str(FC.probe_path(eval_step)) in argv[-1] and argv[-1].endswith('exit $ec')
    assert all(FC.unit_argv(s, code=Path('/code'))[2].startswith('--unit=tag-lp-feas-') for s in STEPS)
    assert FC.unit_argv(FC.STEPS[0], code=Path('/code'))[-1].startswith('/home/rmcfz/miniconda3/envs/sci310/bin/python -B /code/experiments/lpba/prep_lpba.py --pair-set lpba40_feas;')
    assert FC.limit_bytes('10G') == 10*GIB and FC.limit_bytes('512M') == 512*2**20
    assert names == ['prep', 'devcheck_a', 'hi', 'atropos', 'method', 'review', 'convexadam_v4seg', 'input_lock', 'configc', 'ants_s', 'fireants_greedy', 'fireants_syn', 'eval']   # prereg §8.2


def test_the_lp_b_and_lp_d_steps_run_their_entries_own_start_strings(world):
    """Integration item 5: an LP-B step's unit is its entry's own start string (hb_core.queue_launch, xb_chain.chain_launch_string) -- the unit its machine gate exempts, MemoryMax,
    nice, PYTHONPATH -- with --wait, the log and the end probe; LP-D's (a) runs lpdev_a.py run in its own unit tag-lp-feas-deva; the constants the checks read are LP-B's and LP-D's."""
    sys.path.insert(0, str(ROOT/'experiments'/'hb'))
    sys.path.insert(0, str(ROOT/'experiments'/'xb'))
    sys.path.insert(0, str(ROOT/'experiments'/'lpba'/'devcheck'))
    import hb_core
    import lpdev_a
    import lpdev_common
    import xb_chain
    import xb_core
    import xb_unit
    code = Path('/code')
    for name, kind in (('method', 'method'), ('review', 'review'), ('configc', 'configc')):
        step = BY_NAME[name]
        base = hb_core.queue_launch(kind, ['queue', '--pair-set', 'lpba40_feas'], repo=code, pair_set='lpba40_feas')
        argv = FC.unit_argv(step, code=code)
        assert f'--unit={hb_core.queue_unit(kind, "lpba40_feas")}' in argv and argv.index('--wait') == argv.index(f'--unit=tag-lp-feas-{step.tool}')+1
        assert [a for a in argv[:-3] if a not in ('--wait',) and not a.startswith('--property=Standard')] == base[:base.index(str(hb_core.PY))]
        assert argv[-3:-1] == ['/bin/bash', '-c'] and argv[-1].startswith(' '.join(base[base.index(str(hb_core.PY)):])) and str(FC.probe_path(step)) in argv[-1]
        assert FC.memory_max_of(step) == '2G' and FC.limit_bytes(step.inner_memory()) == FC.limit_bytes(dict(method='12G', review='8G', configc='12G')[name])
    for arm in FC.XB_STEP_ARMS:
        step = BY_NAME[arm]
        base = xb_chain.chain_launch_string('lpba40_feas', [arm], code=code)
        argv = FC.unit_argv(step, code=code)
        assert step.tool == 'x-chain' and f'--unit={xb_core.chain_unit("lpba40_feas")}' in argv and '--wait' in argv
        assert argv[-1].startswith(f'{xb_core.PY} -B /code/experiments/xb/xb_chain.py queue --pair-set lpba40_feas --arms {arm}; ec=$?;')
        assert step.inner_memory() == '10G'
    assert sorted(FC.XB_STEP_ARMS) == sorted(xb_core.LPBA_ARMS) and FC.XB_UNIT_END_SCHEMA == xb_unit.SCHEMA.format('end')
    devcheck = BY_NAME['devcheck_a']
    assert FC.unit_argv(devcheck, code=code)[2] == f'--unit={lpdev_a.lc.unit_name("lpba40_feas", lpdev_a.UNIT_TOOL)}'
    assert f'--property=RuntimeMaxSec={lpdev_a.RUNTIME_S}' in FC.unit_argv(devcheck, code=code)          # review minor 3: lpdev_a.launch_line's own unit
    assert FC.places()['root'].joinpath(*FC.DEVCHECK_A_RECORD) == lpdev_common.roots().a/lpdev_a.RECORD
    lock = BY_NAME['input_lock']
    assert lock.commit_from == 'tree' and FC.step_command(lock, code=code)[2:] == ['/code/experiments/hb/hb_store.py', 'lpba-input-lock', '--pair-set', 'lpba40_feas']
    assert f'--setenv=PYTHONPATH={hb_core.he.core.FROZEN}:/code' in FC.unit_argv(lock, code=code)
    assert "/'hb_input_lock.json'" in (ROOT/'experiments'/'hb'/'hb_store.py').read_text(encoding='utf-8') and FC.INPUT_LOCK_NAME == 'hb_input_lock.json'
    assert all(FC.BATCH_PATTERNS[k] == f'hb_{k}_*' for k in ('method', 'review', 'configc'))
    with pytest.raises(ValueError):
        FC.split_start(['systemd-run', '--user', '--unit=tag-lp-feas-he-queue', '--wait', 'python'], unit='tag-lp-feas-he-queue')
    with pytest.raises(ValueError):
        FC.split_start(['systemd-run', '--user', '--unit=tag-lp-he-queue', 'python'], unit='tag-lp-feas-he-queue')


def test_a_missing_command_refuses_before_any_unit(world):
    stand_in = StandIn()
    assert FC.missing_commands() == [] and FC.main(['check'], out=lambda text: None) == 0              # integration item 5: every step's command is in the code tree
    steps = tuple(replace(s, argv=None, launch=None) if s.name in ('devcheck_a', 'review') else s for s in FC.STEPS)
    code, record = FC.run_chain(steps=steps, runner=stand_in, machine=lambda **kw: dict(passed=True), usage=plenty, code_identity=clean, cgroup=CGROUP[0], log=lambda text: None)
    assert code == 2 and stand_in.units == [] and record['refused']['missing_commands'] == ['devcheck_a (LP-D (prereg §12 item 4 (a)))', 'review (LP-B)']
    assert not FC.places()['chain'].exists()
    code, record = FC.run_chain(steps=STEPS, runner=stand_in, usage=plenty, code_identity=lambda repo=None: dict(commit=HEAD, clean=False), cgroup=CGROUP[0],
                                log=lambda text: None)
    assert code == 2 and 'not clean' in record['refused']['code'] and stand_in.units == []        # C26.9-1: from a clean checkout


def test_the_whole_chain_passes(world):
    stand_in = StandIn(method={PAIRS[0]: ('hx_fallback_configC', 'segmentation_qc', [.62, .9]), PAIRS[1]: ('hx_certified', None, [.45, .9])})
    code, record = chain(stand_in)
    assert code == 0 and record['passed'] and len(stand_in.units) == len(STEPS)
    method = next(r for r in record['steps'] if r['step'] == 'method')['check']
    assert method['qc_below_050'] == [PAIRS[1]] and method['cells'][PAIRS[0]]['fallback_by_qc'] is True        # one direction's QC fallback, a QC below .50: recorded, no stop
    assert all(r['code_commit'] == HEAD and r['peak']['peak_bytes'] == GIB//2 for r in record['steps'])
    inner = {r['step']: r['inner_units'] for r in record['steps']}
    assert all(len(inner[s.name]['units']) == 2 and inner[s.name]['skipped'] == [] for s in STEPS if s.inner_memory is not None)
    assert all(inner[s.name] is None for s in STEPS if s.inner_memory is None)
    review = next(r for r in record['steps'] if r['step'] == 'review')['check']
    assert review['pairs'] == {PAIRS[0]: 'hx_fallback_configC', PAIRS[1]: 'hx_certified'} and len(review['batches']) == 1
    assert next(r for r in record['steps'] if r['step'] == 'input_lock')['tree_after']['commit'] == HEAD
    assert [len(next(r for r in record['steps'] if r['step'] == a)['check']['batches']) for a in FC.XB_STEP_ARMS] == [1, 1, 1, 1]   # each XB step its own batch (by arms)
    assert hc.local_unit_end()['unit'] == 'tag-lp-feas-eval'                                              # review I1: holdout_common.note_unit_end after every unit
    assert [r['unit_end']['unit'] for r in record['steps']] == [f'tag-lp-feas-{s.tool}' for s in STEPS]
    saved = json.loads((FC.places()['chain']/'chain.json').read_text())
    assert saved['passed'] is True and not (FC.places()['chain']/'chain_stopped.json').exists()
    code, record = chain(StandIn())                                                                       # the roots are no longer new
    assert code == 2 and 'new roots' in record['refused']['roots']


@pytest.mark.parametrize('case, step, to, kind', [('wm', 'atropos', 'BZ', 'wm_ratio'), ('wm_missing', 'atropos', 'lead', 'wm_ratio_missing'),
                                                  ('both_qc', 'method', 'BZ', 'method_3'), ('rejected', 'method', 'BZ', 'method_1'),
                                                  ('implementation', 'method', 'BZ', 'method_1'), ('suspicious', 'ants_s', 'lead', 'export'),
                                                  ('wrapper', 'configc', 'lead', 'arm_status'), ('labels', 'eval', 'lead', 'label'), ('exit', 'hi', 'lead', 'exit'),
                                                  ('disk', 'disk', 'BZ', 'disk'), ('commit', 'hi', 'lead', 'code_commit'), ('oom', 'method', 'BZ', 'memory_oom'),
                                                  ('peak', 'method', 'lead', 'memory_at_limit'), ('peak_unread', 'configc', 'lead', 'memory_unread'),
                                                  ('events_unread', 'hi', 'lead', 'memory_unread'),
                                                  # integration item 5: LP-D's (a) and LP-B's steps
                                                  ('devcheck_not_passed', 'devcheck_a', 'lead', 'check'), ('devcheck_missing', 'devcheck_a', 'lead', 'check'),
                                                  ('inner_oom', 'method', 'BZ', 'memory_oom'), ('inner_at_limit', 'fireants_syn', 'lead', 'memory_at_limit'),
                                                  ('inner_oom_exit', 'ants_s', 'BZ', 'memory_oom'),
                                                  ('inner_max', 'configc', 'lead', 'memory_max'), ('inner_unread', 'review', 'lead', 'memory_unread'),
                                                  ('review_pair_missing', 'review', 'lead', 'finalized'), ('review_no_record', 'review', 'lead', 'finalized'),
                                                  ('batch_missing', 'ants_s', 'lead', 'batch'), ('batch_commit', 'method', 'lead', 'code_commit'),
                                                  ('cell_commit', 'convexadam_v4seg', 'lead', 'code_commit'), ('lock_missing', 'input_lock', 'lead', 'input_lock'),
                                                  ('tree_moved', 'input_lock', 'lead', 'code_commit'),
                                                  # integration review minors 1, 4, 5
                                                  ('exit_oom', 'hi', 'BZ', 'memory_oom'), ('probe_max', 'atropos', 'lead', 'memory_max'),
                                                  ('inner_schema', 'review', 'lead', 'inner_count'), ('inner_missing', 'method', 'lead', 'inner_count')])
def test_each_stop_condition(world, case, step, to, kind):
    kwargs = {}
    stand_in = {'wm': lambda: StandIn(wm=dict(S01=1.3, S02=1.), exit=dict(atropos=8)),             # review I2: atropos_lpba exits 8 on the band; the check's stop to BZ is the stop
                'wm_missing': lambda: StandIn(wm=dict(S01=1.)),
                'both_qc': lambda: StandIn(method={p: ('hx_fallback_configC', 'segmentation_qc', [.7, .9]) for p in PAIRS}),
                'rejected': lambda: StandIn(method={PAIRS[0]: ('hx_provenance_rejected', None, []), PAIRS[1]: ('hx_certified', None, [.9])}),
                'implementation': lambda: StandIn(method={PAIRS[0]: ('hx_certified', None, [.9]), PAIRS[1]: ('hx_implementation_error', None, [])}),
                'suspicious': lambda: StandIn(arms=dict(ants_s={PAIRS[1]: ('ok', True)})),
                'wrapper': lambda: StandIn(arms=dict(configc={PAIRS[0]: ('wrapper_error', None)})),
                'labels': lambda: StandIn(eval_labels=['/x/label.nii.gz']),
                'exit': lambda: StandIn(exit=dict(hi=1)),
                'disk': lambda: StandIn(),
                'commit': lambda: StandIn(commit=dict(hi='d'*40)),
                'oom': lambda: StandIn(events=dict(method=dict(oom=1, oom_kill=1))),
                'peak': lambda: StandIn(peak=dict(method=12*GIB)),                  # at MemoryMax, no OOM evidence: the page cache can reach it -- for the lead to look at
                'peak_unread': lambda: StandIn(peak=dict(configc=None)),
                'events_unread': lambda: StandIn(events=dict(hi=None)),
                'devcheck_not_passed': lambda: StandIn(devcheck_passed=False),
                'devcheck_missing': lambda: StandIn(drop={'devcheck_a'}),
                'inner_oom': lambda: StandIn(inner=dict(method=dict(events=dict(oom=1, oom_kill=1)))),
                'inner_at_limit': lambda: StandIn(inner=dict(fireants_syn=dict(peak=10*GIB))),
                'inner_oom_exit': lambda: StandIn(inner=dict(ants_s=dict(events=dict(oom=1, oom_kill=1))), exit=dict(ants_s=20), arms=dict(ants_s={PAIRS[0]: ('infra_failure', False)})),
                'inner_max': lambda: StandIn(inner=dict(configc=dict(max=8*GIB))),
                'inner_unread': lambda: StandIn(inner=dict(review=dict(peak=None))),
                'review_pair_missing': lambda: StandIn(review={PAIRS[1]: None}),
                'review_no_record': lambda: StandIn(review={PAIRS[0]: dict(review=None)}),
                'batch_missing': lambda: StandIn(drop={'batch:ants_s'}),
                'batch_commit': lambda: StandIn(batch_commit=dict(method='d'*40)),
                'cell_commit': lambda: StandIn(commit=dict(convexadam_v4seg='d'*40)),
                'lock_missing': lambda: StandIn(drop={'input_lock'}),
                'tree_moved': lambda: StandIn(),
                'exit_oom': lambda: StandIn(exit=dict(hi=137), events=dict(hi=dict(oom=1, oom_kill=1))),     # minor 1: the unit's own OOM evidence after an exit, to BZ
                'probe_max': lambda: StandIn(probe_max=dict(atropos=4*GIB)),                              # minor 4: memory.max is not the unit's MemoryMax 8G
                'inner_schema': lambda: StandIn(inner=dict(review=dict(schema='bayesreg.other.v1'))),     # minor 5: skipped, one record for two pairs
                'inner_missing': lambda: StandIn(inner=dict(method=dict(missing=1)))}[case]()
    if case == 'tree_moved':                                                    # the code tree's HEAD after the input lock's unit is not the chain's
        calls = []

        def moving(repo=None):
            calls.append(repo)
            return dict(commit=HEAD if len(calls) == 1 else 'e'*40, clean=True, status='')
        kwargs['code_identity'] = moving
    if case == 'disk':
        kwargs['usage'] = lambda path: types.SimpleNamespace(free=19.9*GIB)        # under the 20 GiB floor before anything is added
    code, record = chain(stand_in, **kwargs)
    assert code == 4 and record['stop']['step'] == step and record['stop']['to'].startswith(to) and record['stop']['kind'] == kind, record['stop']
    stopped = json.loads((FC.places()['chain']/'chain_stopped.json').read_text())
    assert stopped['step'] == step
    tools = {s.name: s.tool for s in FC.STEPS}
    assert stand_in.units[-1] == f'tag-lp-feas-{tools[step]}' if case != 'disk' else stand_in.units == ['tag-lp-feas-prep']      # nothing after the step that stopped


def test_the_machine_gate_is_polled_on_the_test_clock(world):
    """Review I1: the machine gate through holdout_common.wait_for -- refused twice (the 10 s gap), then passed: the chain sleeps and goes on; refused for 15 minutes: precheck_timeout."""
    clock = dict(t=0.)
    sleeps, calls = [], []

    def machine(**kw):
        calls.append(kw)
        if len(calls) <= 2:
            raise hc.Refused('machine', 'the previous tag- unit ended 4 s ago (test)')
        return dict(passed=True)

    def sleep(seconds):
        sleeps.append(seconds)
        clock['t'] += seconds
    wait = dict(now=lambda: clock['t'], sleep=sleep, poll=30.)
    code, record = chain(StandIn(), machine=machine, wait=wait)
    assert code == 0 and sleeps == [30., 30.] and calls[0] == dict(own_units=())                           # nothing exempted: the chain is in no unit
    FC.retire('polled')
    clock['t'], sleeps[:] = 0., []
    stand_in = StandIn()

    def busy(**kw):
        raise hc.Refused('machine', 'other units are active (test)')
    code, record = chain(stand_in, machine=busy, wait=wait)
    assert code == 4 and record['stop']['kind'] == 'precheck_timeout' and stand_in.units == [] and sum(sleeps) >= hc.GATE_WAIT_SECONDS


def test_an_exception_of_the_chain_is_a_stop_with_its_record(world):
    def broken():
        raise KeyError('a record without its key (test)')
    code, record = chain(StandIn(), checks=dict(CHECKS, hi=broken))
    assert code == 4 and record['stop']['step'] == 'hi' and record['stop']['kind'] == 'exception' and 'KeyError' in record['stop']['reason']
    assert json.loads((FC.places()['chain']/'chain_stopped.json').read_text())['kind'] == 'exception'


def test_the_disk_extrapolates_per_subject_and_per_pair(world, monkeypatch):
    r = lc.roots('lpba40_feas')
    (r.hi_out).mkdir(parents=True)
    (r.hi_out/'x.bin').write_bytes(b'0'*1000)                                  # per subject: x 34/2
    (r.stage/lc.MM_DIR).mkdir(parents=True)
    (r.stage/lc.MM_DIR/'S01_image.nii.gz').write_bytes(b'0'*100)                 # per subject
    (r.stage/'LPBA40_S01__LPBA40_S02').mkdir(parents=True)
    (r.stage/'LPBA40_S01__LPBA40_S02'/'pair.npz').write_bytes(b'0'*200)          # per pair: x 17/2
    record = FC.check_disk(plenty)
    assert record['extrapolated_formal_gib']*GIB == pytest.approx(1100*17+200*8.5)


def test_the_hold_file_stops_before_the_unit(world):
    hold = lc.roots('lpba40_feas').hold
    hold.parent.mkdir(parents=True, exist_ok=True)
    hold.write_text('')
    stand_in = StandIn()
    code, record = chain(stand_in)
    assert code == 4 and record['stop']['kind'] == 'hold' and stand_in.units == []


def test_retire_keeps_the_old_roots(world):
    p = FC.places()
    (p['root']/'stage').mkdir(parents=True)
    p['hi_out'].mkdir(parents=True)
    record = FC.retire('try1')
    assert not p['root'].exists() and not p['hi_out'].exists()
    assert (p['root'].with_name('lpba40_feas_v1_try1')/'stage').is_dir() and (p['root'].with_name('lpba40_feas_v1_try1')/'retired.json').is_file()
    assert len(record['retired']) == 2 and FC.roots_problem() is None
    with pytest.raises(ValueError):
        FC.retire('../x')


# ---- the diff review: the chain in no unit (I, the lead's (a)), the formal units' MemoryMax (minor 1), the hold after the wait (minor 2), the exit code kept (minor 3) ----------------
@pytest.mark.parametrize('line', ['0::/user.slice/user-1000.slice/user@1000.service/app.slice/tag-lp-feas-chain.service',
                                  '0::/user.slice/user-1000.slice/user@1000.service/app.slice/tag-x.service/sub'])
def test_inside_a_tag_unit_the_chain_refuses_and_writes_nothing(world, tmp_path, line):
    fake = tmp_path/'cgroup_unit'
    fake.write_text(line+'\n')
    stand_in = StandIn()
    code, record = chain(stand_in, cgroup=fake)
    assert code == 8 and stand_in.units == [] and record['refused']['unit'].startswith('tag-') and not FC.places()['chain'].exists()
    assert FC.chain_unit(CGROUP[0]) is None and FC.chain_unit(tmp_path/'missing').startswith('(unreadable')


def test_the_steps_units_are_the_formal_units(world, monkeypatch):
    """Diff review minor 1, extended by integration review minor 3: every step's feasibility unit is its entry's own start string and equals the formal mode's but for the unit name,
    the pair set and the log -- MemoryMax, MemorySwapMax, RuntimeMaxSec, CPUQuota (Atropos), OOMPolicy and the swap of the evaluation, nice, the environment, the command."""
    real = FC.formal_specs()
    memory = {n: next(o for o in spec['options'] if o.startswith('--property=MemoryMax=')).split('=', 2)[2] for n, spec in real.items() if spec is not None}
    assert memory['prep'] == memory['atropos'] == memory['devcheck_a'] == '8G' and memory['hi'] == '12G' and memory['eval'] == E.UNIT_MEMORY_MAX == '10G'
    assert all(memory[n] == '2G' for n in ('method', 'review', 'configc', 'ants_s', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg'))   # LP-B's queue and chain units (lpba40_test)
    assert '--property=CPUQuota=800%' in real['atropos']['options'] and '--property=CPUQuota=800%' not in real['prep']['options']
    assert {'--property=MemorySwapMax=4G', '--property=RuntimeMaxSec=259200', '--property=OOMPolicy=continue', '--property=CPUQuota=800%'} <= set(real['eval']['options'])
    assert real['eval']['command'][2:7] == ['-m', 'bayesreg.eval.lpba_eval', 'evaluate', '--pair-set', '<pair_set>']
    assert real['input_lock'] is None and 'input_lock' in FC.NOT_FORMAL                                # no formal unit (the lead's command)
    assert FC.spec_problems(FC.STEPS) == []                                                            # every step's is known and equal
    changed = dict(real, method=dict(real['method'], options=sorted(real['method']['options']+['--property=CPUQuota=100%'])))
    assert FC.spec_problems(FC.STEPS, changed) == ["method: the feasibility unit is not the formal unit's: options {'feasibility': [], 'formal': ['--property=CPUQuota=100%']}, command None"]
    stand_in = StandIn()
    hi = dict(real['hi'], options=[o.replace('MemoryMax=12G', 'MemoryMax=16G') for o in real['hi']['options']])
    code, record = chain(stand_in, formal=dict(real, hi=hi))
    assert code == 2 and stand_in.units == [] and record['refused']['unit_spec'] == [
        "hi: the feasibility unit is not the formal unit's: options {'feasibility': ['--property=MemoryMax=12G'], 'formal': ['--property=MemoryMax=16G']}, command None"]
    # an entry whose feasibility line drifted from its formal one (here: Atropos without its CPUQuota in the feasibility mode) is refused before any unit
    import atropos_lpba
    original = atropos_lpba.launch_line
    monkeypatch.setattr(atropos_lpba, 'launch_line', lambda pair_set, **kw: original(pair_set, **kw).replace(' --property=CPUQuota=800%', '') if pair_set == 'lpba40_feas'
                        else original(pair_set, **kw))
    problems = FC.spec_problems(FC.STEPS)
    assert problems == ["atropos: the feasibility unit is not the formal unit's: options {'feasibility': [], 'formal': ['--property=CPUQuota=800%']}, command None"]


def test_the_hold_file_after_the_wait_stops_before_the_unit(world):
    hold = lc.roots('lpba40_feas').hold

    def machine(**kw):
        hold.parent.mkdir(parents=True, exist_ok=True)
        hold.write_text('')                                                  # put while the gate waited
        return dict(passed=True)
    stand_in = StandIn()
    code, record = chain(stand_in, machine=machine)
    assert code == 4 and record['stop']['kind'] == 'hold' and stand_in.units == []


def test_an_exception_of_a_check_after_a_failed_unit_keeps_the_exit_code(world):
    def broken():
        raise KeyError('a record without its key (test)')
    code, record = chain(StandIn(exit=dict(hi=3)), checks=dict(CHECKS, hi=broken))
    assert code == 4 and record['stop']['step'] == 'hi' and record['stop']['kind'] == 'exception' and 'exit 3' in record['stop']['reason'] and 'KeyError' in record['stop']['reason']


# ---- the integration review (0/1/8): minors 1, 2, 4, 5 -------------------------------------------------------------------------------------------------------------------------------
def test_minor1_an_exit_keeps_the_per_pair_units_stop_and_reads_the_units_probe(world):
    """Minor 1: after an exit that is not 0 the per-pair units' stop is kept (row['inner_units'], row['inner_stop']) and named in the exit's stop; the unit's own probe is read first
    (no OOM evidence there: recorded, the exit's stop follows; no probe at all: recorded as unread)."""
    code, record = chain(StandIn(inner=dict(configc=dict(max=8*GIB)), exit=dict(configc=3)))
    assert code == 4 and record['stop']['step'] == 'configc' and record['stop']['kind'] == 'exit' and record['stop']['to'].startswith('lead')
    assert 'exit 3' in record['stop']['reason'] and 'the per-pair units: memory_max' in record['stop']['reason']
    row = next(r for r in record['steps'] if r['step'] == 'configc')
    assert row['inner_stop']['kind'] == 'memory_max' and row['inner_units']['stop']['kind'] == 'memory_max'
    assert row['probe_on_exit']['read'] is True and row['probe_on_exit']['events'] == dict(oom=0, oom_kill=0)
    FC.retire('minor1')
    code, record = chain(StandIn(exit=dict(hi=1), drop={'probe:hi'}))
    assert code == 4 and record['stop']['step'] == 'hi' and record['stop']['kind'] == 'exit'
    assert next(r for r in record['steps'] if r['step'] == 'hi')['probe_on_exit']['read'] is False


def test_minor1_oom_evidence_of_a_per_pair_unit_goes_to_bz_before_the_other_stops_of_its_units(world):
    """Minor 1 (and the per-pair units read as one set): one unit's MemoryMax is wrong, the other has OOM evidence -> the stop is the OOM's, to BZ."""
    step = BY_NAME['configc']
    before = FC.unit_ends()
    root = FC.places()['root']
    limit = FC.limit_bytes(step.inner_memory())
    for k, (maximum, events) in enumerate([(8*GIB, dict(oom=0, oom_kill=0)), (limit, dict(oom=1, oom_kill=1))]):
        write(root/'configc'/PAIRS[k]/'unit'/'unit_end.json', dict(schema=FC.XB_UNIT_END_SCHEMA, unit='u', probe=dict(peak_bytes=GIB, max_bytes=maximum, events=events)))
    with pytest.raises(FC.Stop) as caught:
        FC.check_inner_units(step, before, n_pairs=2)
    assert caught.value.to == 'BZ' and caught.value.kind == 'memory_oom'


def test_minor5_a_unit_end_of_another_schema_is_skipped_recorded_and_counted_out(world):
    step = BY_NAME['ants_s']
    before = FC.unit_ends()
    root = FC.places()['root']
    probe = dict(peak_bytes=GIB, max_bytes=FC.limit_bytes(step.inner_memory()), events=dict(oom=0, oom_kill=0))
    for pair in PAIRS:
        write(root/'ants_s'/pair/'unit'/'unit_end.json', dict(schema=FC.XB_UNIT_END_SCHEMA, unit='u', probe=probe))
    (root/'ants_s'/'stray'/'unit').mkdir(parents=True)
    (root/'ants_s'/'stray'/'unit'/'unit_end.json').write_text('not json')
    record = FC.check_inner_units(step, before, n_pairs=2)
    assert len(record['units']) == 2 and record['skipped'] == [dict(unit_end=str(root/'ants_s'/'stray'/'unit'/'unit_end.json'), schema=None)]
    with pytest.raises(FC.Stop) as caught:
        FC.check_inner_units(step, before, n_pairs=3)
    assert caught.value.kind == 'inner_count' and caught.value.to.startswith('lead') and 'stray' in caught.value.reason


@pytest.mark.parametrize('line', ['0::/user.slice/user-1000.slice/user@1000.service/app.slice/tag-lp-feas-chain.scope',
                                  '0::/user.slice/user-1000.slice/user@1000.service/app.slice/tag-x.scope/sub'])
def test_minor2_inside_a_tag_scope_the_chain_refuses_too(world, tmp_path, line):
    fake = tmp_path/'cgroup_scope'
    fake.write_text(line+'\n')
    stand_in = StandIn()
    code, record = chain(stand_in, cgroup=fake)
    assert code == 8 and stand_in.units == [] and record['refused']['unit'].endswith('.scope') and not FC.places()['chain'].exists()
    other = tmp_path/'cgroup_other'
    other.write_text('0::/user.slice/user-1000.slice/user@1000.service/app.slice/run-u12.scope\n')
    assert FC.chain_unit(other) is None


def test_minor3_an_entrys_launch_line_must_wait_and_log_to_the_chains_log(world):
    log = '/x/step_unit.log'
    good = ['systemd-run', '--user', '--unit=tag-lp-feas-prep', '--wait', '--collect', f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}', 'python']
    options, command = FC.split_start(good, unit='tag-lp-feas-prep', log=log)
    assert options == ['--unit=tag-lp-feas-prep', '--collect'] and command == ['python']
    for bad in ([a for a in good if a != '--wait'], [a.replace(log, '/y.log') for a in good], good[:5]+good[6:]):
        with pytest.raises(ValueError):
            FC.split_start(bad, unit='tag-lp-feas-prep', log=log)
