"""LP-F (decision_log 2026-11-20, the lead's rulings of ~4:08 and ~4:24 AM CST and the LP-F review rulings of ~3:42 PM CST; LPBA40 prereg §8.6; contract C26.8-7):
experiments/lpba/dxg_monitor.py on synthetic inputs only.  The kernel journal, dmesg and the user journal are fake texts and JSON entries; the supervisor records are written by the module's
own fixture helpers; every root is lpba_common.RESULTS moved into tmp_path (holdout_common's test sandbox set), so the real hold file and the real results root are never touched; no unit,
no nvidia-smi, no journalctl is run (the measuring and the watching are given fakes)."""
import ast
import json
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'lpba', ROOT/'experiments'/'lpba'/'devcheck', ROOT/'experiments'/'xb', ROOT/'experiments'/'he'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import dxg_monitor as M  # noqa: E402

lc, hc = M.lc, M.lc.hc
COMMIT = M.SELF_COMMIT
ARGS = ['--since', '900', '--delta', '0.7,0.85', '--start-epoch', '0', '--commit', COMMIT]
OWN_UNIT = M.own_unit                                     # the module's own, before the sandbox fixture replaces it


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setenv(hc.TEST_SANDBOX_ENV, '1')
    monkeypatch.setattr(lc, 'RESULTS', tmp_path/'results')
    monkeypatch.setattr(M, 'own_unit', lambda cgroup=None: None)
    return tmp_path/'results'


def kern(*lines, journal=True):
    head = '[    0.000000] host kernel: Linux version x' if journal else '[    0.000000] Linux version x'
    return '\n'.join([head]+[M._kline(t, x, journal) for t, x in lines])


def run(mode, lines=(), user=(), *, active=False, deltas=(.70, .85), since=900., dmesg=None, may_hold=True, logs=(), now_uptime=4000., now_epoch=1e12, defer=(),
        start_epoch=0.):
    return M.evaluate(mode, since=since, deltas=deltas, active=active, start_epoch=start_epoch, commit=COMMIT, kernel_text=kern(*lines),
                      dmesg_text=kern(*lines, journal=False) if dmesg is None else dmesg, user_entries=list(user), now_uptime=now_uptime, now_epoch=now_epoch, boot_offset=0.,
                      logs=logs, may_hold=may_hold, defer=defer)


def pair(f, m):
    return lc.pair_name(f, m)


# ---- the self-test --------------------------------------------------------------------------------------------------------------------------------
def test_the_selftest_passes_every_case_and_leaves_the_real_roots_alone(sandbox):
    """The module's own synthetic cases (one per class and rule, and one per review blocker): all pass; lpba_common.RESULTS is restored afterwards and its temporary root removed."""
    before = lc.RESULTS
    lines = []
    n_pass, n_fail, rows = M.selftest(log=lines.append)
    assert n_fail == 0 and n_pass == len(rows) >= 52, [r for r in rows if not r['passed']]
    assert lc.RESULTS == before and not (sandbox/'lpba40_test_v1'/'hold').exists()
    names = ' '.join(r['case'] for r in rows)
    for needed in ('W at the GPU step end', 'EAGAIN inside the GPU step', 'EAGAIN in the cell window, outside', 'CA', 'W variant', 'qai variant', 'EAGAIN variant',
                   'non-dxg line in a cell window', 'non-dxg line outside', 'drop_caches: 1', 'drop_caches: 3', 'dmesg unreadable', 'kernel journal unreadable',
                   'short of dmesg', 'last evaluation', '(B1 (a))', '(B1 (b))', '(B1 (c))', '(B2', 'I2', 'tag-lp-eval', 'I1a', 'I1b', 'I1c', 'I1d'):
        assert needed in names, needed


def test_the_selftest_command_exits_0(sandbox, capsys):
    assert M.main(['selftest']) == 0
    assert capsys.readouterr().out.strip().endswith('0 failed')


# ---- constants against their sources --------------------------------------------------------------------------------------------------------------
def test_constants_are_the_sources_ones():
    import he_core
    import lpdev_common
    import xb_core
    assert M.DEVCHECK_DIR == lpdev_common.DEVCHECK_DIR
    assert M.OUTPUT_STATUSES == he_core.OUTPUT_STATUSES
    assert M.XB_ARMS == xb_core.LPBA_ARMS
    log = (ROOT/'docs'/'decision_log.md').read_text(encoding='utf-8')
    assert f'`{M.W_TEXT}`' in log and f'`{M.P_TEXT}`' in log                  # decision_log line 498: the m version's W and P texts
    assert all(t.split('dxgk: ', 1)[1].split(': ', 1)[1] in log for t in M.EAGAIN_TEXTS)     # BZ 11-18 (2): vmbus_sendpacket failed: fffffff5, create_process failed -11
    assert M.SOURCE['evaluator']['sha256'].startswith('4e892819') and M.SOURCE['watcher']['sha256'].startswith('eed0b674')     # decision_log line 500
    assert M.DRIFT == .15 and M.TOL == 1. and M.POLL_S == 20. and M.EXIT_ERROR == 4
    assert set(M.USER_FIELDS) >= {'__MONOTONIC_TIMESTAMP', 'USER_UNIT', '_SYSTEMD_USER_UNIT', 'MESSAGE'} and f'--output-fields={",".join(M.USER_FIELDS)}' in M.UCMD


def test_the_hold_is_lpba_commons_one_for_every_mode(sandbox):
    for mode in M.MODES:
        assert M.hold_path(mode) == lc.roots('lpba40_test').hold == sandbox/'lpba40_test_v1'/'hold'
    assert M.mode_root('lpba40_test') == lc.roots('lpba40_test').root and M.mode_root('lpba40_feas') == lc.roots('lpba40_feas').root
    assert M.mode_root('devcheck') == sandbox/'lpba40_devcheck_v1'
    with pytest.raises(SystemExit):
        M.check_mode('holdout')


# ---- units of each mode ------------------------------------------------------------------------------------------------------------------------
def test_unit_names_of_each_mode_from_the_tools_own_builders(sandbox):
    import lpba_arms
    import xb_core
    from bayesreg.eval import lpba_eval
    p = pair('S03', 'S04')
    for mode in ('lpba40_test', 'lpba40_feas'):
        cases = {lpba_arms.unit_name('he', mode, p): ('cell', 'method'), lpba_arms.unit_name('c', mode, p): ('cell', 'configc'),
                 lpba_arms.unit_name('review', mode, p): ('cell', 'review'), xb_core.unit_name('fireants_syn', p, mode): ('cell', 'xb'),
                 xb_core.unit_name('convexadam_v4seg', p, mode): ('cell', 'xb'), xb_core.unit_name('ants_s', p, mode): ('cell', 'xb'),
                 lpba_arms.unit_prefix('he-c', mode)+'-S03-S04': ('cell', 'nested'), lpba_arms.unit_prefix('he-c3', mode)+'-S03-S04': ('cell', 'nested'),
                 lc.unit_name(mode, 'hi'): ('cell', 'hi'), lc.unit_name(mode, 'prep'): ('cell', 'other'), lc.unit_name(mode, 'atropos'): ('cell', 'other'),
                 lpba_eval.OWN_UNITS[mode][0]: ('cell', 'other'),                               # step 11's evaluation unit: a CPU cell (the lead's ruling, LP-F review)
                 lpba_arms.unit_name('he', mode): ('driver', 'driver'), lpba_arms.unit_name('c', mode): ('driver', 'driver'),
                 lpba_arms.unit_name('review', mode): ('driver', 'driver'), xb_core.chain_unit(mode): ('driver', 'driver'),
                 'tag-hb-x-chain': (None, None), 'seq-vp-x': (None, None), 'tag-evalspec': (None, None)}
        for name, want in cases.items():
            assert M.unit_of(mode, name)[:2] == want, (mode, name)
        assert M.unit_of(mode, xb_core.unit_name('fireants_syn', p, mode))[2] == dict(arm='fireants_syn', f='S03', m='S04', pair=p)
        assert M.expected_unit(mode, 'xb', 'fireants_syn', p) == xb_core.unit_name('fireants_syn', p, mode)
        for k, part in (('method', 'he'), ('configc', 'c'), ('review', 'review')):
            assert M.expected_unit(mode, k, 'method', p) == lpba_arms.unit_name(part, mode, p)
    assert lpba_eval.OWN_UNITS['lpba40_test'] == ('tag-lp-eval',)
    assert M.unit_of('lpba40_test', 'tag-lp-feas-hi')[0] is None and M.unit_of('lpba40_test', 'tag-lp-devb')[0] is None
    assert M.unit_of('lpba40_feas', 'tag-lp-hi')[0] is None
    assert M.unit_of('devcheck', 'tag-lp-devb')[:2] == ('cell', 'devb') and M.unit_of('devcheck', 'tag-lp-devc')[:2] == ('cell', 'devc')
    assert M.unit_of('devcheck', 'tag-he-c-0438-0439')[:2] == ('cell', 'nested') and M.unit_of('devcheck', 'tag-lp-hi')[0] is None


def test_a_development_nested_unit_name_is_masked_not_refused(sandbox):
    """(c)'s nested tag-he-c-<short> carries a development pair's numbers: masked in every output (the id guard stays the last line of defence, never tripped)."""
    doc, events = run('devcheck', user=M._window('tag-lp-devc', 1000., 1200.)+M._window('tag-he-c-0438-0439', 1100., 1150.), may_hold=False, defer=M.DEFERRED_KINDS)
    assert not events and sorted(c['id'] for c in doc['cells']) == ['tag-he-c-<id>#1', 'tag-lp-devc#1'] and '0438' not in json.dumps(doc)
    assert doc['deferred'] == ["tag-lp-devc#1: ['devcheck_c.json'] not written at the last evaluation (the step after the unit writes it)"]


# ---- B1: two sources -----------------------------------------------------------------------------------------------------------------------------
def test_supervisor_records_on_disk_are_a_source_of_their_own(sandbox):
    """B1: the supervisor directories of the mode root (started since --since; one level at a time) are cells even with an empty user journal; a directory started before the step is
    not; an unreadable directory made since --start-epoch is an event."""
    root = M.mode_root('lpba40_test')
    M._xb_cell(root, 'fireants_syn', pair('S03', 'S04'), (1000., 30.), win=(995., 1032.))
    M._xb_cell(root, 'ants_s', pair('S05', 'S06'), (500., 30.), win=(495., 532.))           # before the step (since 900)
    M._hb_cell(root, 'method', pair('S07', 'S08'), (1100., 1200.))
    (root/'method'/pair('S07', 'S08')/'unit_review').mkdir(parents=True)                        # the review's directory, not readable yet
    disk = M.disk_cells('lpba40_test', 900., 0.)
    assert [(d['kind'], d['unit'], d['start'], d['end']) for d in disk if not d['unreadable']] == [
        ('method', 'tag-lp-he-S07-S08', 1100., 1200.), ('xb', 'tag-lp-x-fireants_syn-S03-S04', 995., 1032.)]
    assert [(d['kind'], d['unit']) for d in disk if d['unreadable']] == [('review', 'tag-lp-review-S07-S08')]
    doc, events = run('lpba40_test', [(1010., 'hv_balloon: unhandled message: type: 7')])
    texts = doc['events']
    assert any('x-fireants_syn-S03-S04#1: seen only in the supervisor records' in t for t in texts)
    assert any('he-S07-S08#1: seen only in the supervisor records' in t for t in texts)
    assert any('review-S07-S08#1: supervisor directory unit_review made since the step start without a readable start' in t for t in texts)
    assert any('non-dxg kernel line' in t for t in texts) and M.hold_path('lpba40_test').exists()
    assert M.disk_cells('lpba40_test', 900., 1e12) == [d for d in disk if not d['unreadable']]      # made before --start-epoch: not an event


def test_a_cell_in_one_source_only_waits_its_grace_while_active(sandbox):
    root = M.mode_root('lpba40_test')
    M._xb_cell(root, 'fireants_syn', pair('S03', 'S04'), (1000., 30.), win=(995., None))
    doc, events = run('lpba40_test', active=True, now_uptime=1000.)                 # 5 s after the start: not yet an event
    assert not events
    doc, events = run('lpba40_test', active=True, now_uptime=995.+M.GRACE_S)
    assert any('seen only in the supervisor records' in t for t in doc['events'])


def test_w_in_a_configc_cell_holds_since_its_records_time_no_gpu_segment(sandbox):
    """I2: configuration C's records time no GPU segment on the uptime clock: a W at the supervisor's child exit holds (the 46e49cd5 version recorded it); and its m.2 records."""
    root = M.mode_root('lpba40_test')
    rec = M._hb_cell(root, 'configc', pair('S07', 'S08'), (2000., 2100.))
    user = M._window('tag-lp-c-S07-S08', 1999.9, 2100.5)
    doc, events = run('lpba40_test', [(2100.75, M.W_TEXT)], user)         # readings 2100.05 / 2099.90: the second inside both windows, the first after unit_end.uptime
    assert any('outside every cell window or split' in t for t in doc['events'])
    M.hold_path('lpba40_test').unlink()
    doc, events = run('lpba40_test', [(2050., M.W_TEXT)], user)
    assert doc['events'] == ['dxg: tag-lp-c-S07-S08#1: W line at kernel 2050.000000: W/P lines and the rule fails: a False (GPU segments 0), b failed [], c failed []']
    M.hold_path('lpba40_test').unlink()
    (rec/'result.json').unlink()
    doc, events = run('lpba40_test', [], user)
    assert doc['events'] == ["state: last evaluation (the step no longer active): tag-lp-c-S07-S08#1 started and ['result.json'] not written (m.2)"]


def test_a_cell_whose_records_are_younger_than_three_seconds_is_not_settled_while_active(sandbox):
    root = M.mode_root('lpba40_test')
    M._xb_cell(root, 'fireants_greedy', pair('S03', 'S04'), (1000., 30.))
    user = M._window('tag-lp-x-fireants_greedy-S03-S04', 995., 1032.)
    doc, events = run('lpba40_test', [(1030.8, M.W_TEXT)], user, active=True, now_uptime=1040., now_epoch=__import__('time').time())
    assert not events and doc['wp_pending'] == 1 and not doc['cells'][0]['settled']


def test_the_two_readings_use_both_deltas_and_the_tolerance_is_one_second(sandbox):
    """W at kernel t: the GPU step ends at 1030.  δ (.70, .85): t = 1031.8 reads 1031.10 / 1030.95 -- within 1 s of 1030 only for the second reading -> hold; t = 1031.6 reads
    1030.90 / 1030.75 -> both within -> recorded.  The supervisor window ends at 1032, the journal's at 1032.5."""
    root = M.mode_root('lpba40_test')
    M._xb_cell(root, 'fireants_syn', pair('S03', 'S04'), (1000., 30.))
    user = M._window('tag-lp-x-fireants_syn-S03-S04', 995., 1032.5)
    doc, events = run('lpba40_test', [(1031.6, M.W_TEXT)], user)
    assert not events
    doc, events = run('lpba40_test', [(1031.8, M.W_TEXT)], user)
    assert any('a False' in e for e in doc['events'])


def test_feasibility_mode_reads_its_own_root_and_units(sandbox):
    root = M.mode_root('lpba40_feas')
    M._xb_cell(root, 'convexadam_v4seg', pair('S01', 'S02'), (1000., 30.), mode='lpba40_feas')
    user = M._window('tag-lp-feas-x-convexadam_v4seg-S01-S02', 995., 1032.) + M._window('tag-lp-x-convexadam_v4seg-S01-S02', 1500., 1600.)
    doc, events = run('lpba40_feas', [(1015., sorted(M.EAGAIN_TEXTS)[0])], user)
    assert not events and doc['eagain']['recorded'] == 1 and [c['id'] for c in doc['cells']] == ['tag-lp-feas-x-convexadam_v4seg-S01-S02#1']
    assert doc['foreign_units'] == ['tag-lp-x-convexadam_v4seg-S01-S02']


def test_the_hold_is_made_with_its_directory_when_the_formal_root_is_not_there_yet(sandbox):
    """The development checks and the feasibility run before <RESULTS>/lpba40_test_v1 exists: an event still leaves the hold (its directory made, nothing else in it)."""
    assert not (sandbox/'lpba40_test_v1').exists()
    doc, events = run('devcheck', [(1500., M.W_TEXT)])
    assert events and doc['hold_touched_now'] and M.hold_path('devcheck').is_file() and [p.name for p in (sandbox/'lpba40_test_v1').iterdir()] == ['hold']


def test_no_hold_option_and_files_never_touch_the_hold(sandbox, tmp_path):
    doc, events = run('lpba40_test', [(1500., M.W_TEXT)], may_hold=False)
    assert events and not M.hold_path('lpba40_test').exists() and doc['may_hold'] is False
    kf, df, uf = tmp_path/'k.txt', tmp_path/'d.txt', tmp_path/'u.json'
    kf.write_text(kern((1500., M.W_TEXT)))
    df.write_text(kern((1500., M.W_TEXT), journal=False))
    uf.write_text('')
    rc = M.main(['eval', '--mode', 'lpba40_test', *ARGS, '--journal-file', str(kf), '--dmesg-file', str(df), '--user-journal-file', str(uf)])
    assert rc == 3 and not M.hold_path('lpba40_test').exists()


# ---- B3: arguments, exits ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('bad', [('--delta', '0.9,0.8'), ('--delta', '-0.1,-0.8'), ('--delta', 'x'), ('--since', 'nan?'), ('--since', 'nan'), ('--start-epoch', 'inf'),
                                 ('--start-epoch', ''), ('--delta', '0.7,inf'), ('--commit', 'HEAD'), ('--ack-before', 'abc')])
def test_eval_with_an_argument_that_does_not_parse_holds_and_exits_4(sandbox, bad, capsys):
    """B3 (a): the m version's evaluator exits 4 and holds on a δ (or any value) it cannot use; so does eval (not in file mode)."""
    argv = list(ARGS) + ['--ack-before', '0'*64]
    argv[argv.index(bad[0])+1] = bad[1]
    assert M.main(['eval', '--mode', 'lpba40_test', *argv]) == M.EXIT_ERROR
    assert M.hold_path('lpba40_test').exists() and json.loads(capsys.readouterr().out)['hold_touched_now'] is True


def test_watch_with_a_bad_delta_holds_writes_event_and_end_and_exits_4(sandbox, tmp_path, monkeypatch):
    """B3 (a): `watch … --delta 0.9,0.8` used to end by SystemExit with no hold and no poll.txt."""
    monkeypatch.setattr(M, 'evaluate_now', lambda *a, **k: pytest.fail('no evaluation with a bad δ'))
    argv = list(ARGS)
    argv[argv.index('--delta')+1] = '0.9,0.8'
    assert M.main(['watch', '--mode', 'lpba40_test', *argv, '--out', str(tmp_path/'w'), '--watch-pid', '1']) == M.EXIT_ERROR
    poll = (tmp_path/'w'/'poll.txt').read_text().splitlines()
    assert M.hold_path('lpba40_test').exists() and poll[0].startswith('EVENT') and 'rc=4' in poll[0] and poll[-1].startswith('END')


def test_watch_ended_by_anything_holds(sandbox, tmp_path, monkeypatch):
    """B3 (a): a SystemExit or KeyboardInterrupt inside the loop holds, writes EVENT and END, exits 4."""
    monkeypatch.setattr(M, 'watched_active', lambda units, pids: True)
    monkeypatch.setattr(M, 'uptime', lambda: 1000.)

    def boom(*a, **k):
        raise KeyboardInterrupt()
    monkeypatch.setattr(M, 'evaluate_now', boom)
    assert M.main(['watch', '--mode', 'lpba40_test', *ARGS, '--out', str(tmp_path/'w'), '--watch-unit', 'tag-lp-hi']) == M.EXIT_ERROR
    poll = (tmp_path/'w'/'poll.txt').read_text().splitlines()
    assert M.hold_path('lpba40_test').exists() and 'KeyboardInterrupt' in poll[-2] and poll[-1].startswith('END')


def test_refusals_before_the_start_are_listed_not_counted_and_later_ones_hold(sandbox, tmp_path, monkeypatch):
    """B3 (b) (R1): refusal records count from --start-epoch on; the earlier ones are written into refusals_before.json at the watch's start."""
    import os
    root = M.mode_root('lpba40_test')
    old = M._write(root/'batches'/'xb_1'/'refusals'/'old.json', '{}')
    os.utime(old, (1000., 1000.))
    hi = M._window('tag-lp-prep', 1000., 1100.)                     # a CPU unit with no record to check
    doc, events = run('lpba40_test', user=hi, start_epoch=2000.)
    assert not events and doc['refusals_since_start'] == 0
    M._write(root/'batches'/'xb_2'/'refusals'/'new.json', '{}')
    doc, events = run('lpba40_test', user=hi, start_epoch=2000.)
    assert doc['refusals_since_start'] == 1 and any('refusal record new.json' in e for e in doc['events'])
    M.hold_path('lpba40_test').unlink()
    seq = iter([True, False])
    monkeypatch.setattr(M, 'watched_active', lambda units, pids: next(seq, False))
    monkeypatch.setattr(M, 'uptime', lambda: 1000.)
    monkeypatch.setattr(M.time, 'sleep', lambda s: None)
    monkeypatch.setattr(M, 'evaluate_now', lambda args, may_hold, **k: (dict(schema=M.SCHEMA, events=[], cells=[]), []))
    argv = list(ARGS)
    argv[argv.index('--start-epoch')+1] = '2000'
    assert M.main(['watch', '--mode', 'lpba40_test', *argv, '--out', str(tmp_path/'w'), '--watch-pid', '1']) == 0
    listed = json.loads((tmp_path/'w'/'refusals_before.json').read_text())
    assert [Path(r['path']).name for r in listed['records']] == ['old.json'] and listed['records'][0]['sha256'] == M.sha(old)


# ---- I1: the deferred devc --------------------------------------------------------------------------------------------------------------------
def test_devc_is_deferred_at_the_watchs_last_evaluation_and_judged_by_the_eval_after_compare(sandbox, tmp_path, monkeypatch):
    """I1: watch (devcheck mode) writes the devc cell's pending W and its missing devcheck_c.json to pending.json and holds nothing; the eval after lpdev_c.py compare (no deferral)
    holds: the W has no GPU segment to belong to, and a record that did not pass or is missing holds too."""
    user = M._window('tag-lp-devc', 1000., 1100.)
    lines = [(1050., M.W_TEXT)]
    seq = iter([True, True, False])
    monkeypatch.setattr(M, 'watched_active', lambda units, pids: next(seq, False))
    monkeypatch.setattr(M, 'uptime', lambda: 1200.)
    monkeypatch.setattr(M.time, 'sleep', lambda s: None)
    monkeypatch.setattr(M, 'read_kernel_journal', lambda: kern(*lines))
    monkeypatch.setattr(M, 'read_dmesg', lambda: kern(*lines, journal=False))
    monkeypatch.setattr(M, 'read_user_journal', lambda start_epoch: (list(user), None))
    out = tmp_path/'w'
    assert M.main(['watch', '--mode', 'devcheck', *ARGS, '--out', str(out), '--watch-unit', 'tag-lp-devc']) == 0
    pending = json.loads((out/'pending.json').read_text())
    assert len(pending['deferred']) == 2 and not M.hold_path('devcheck').exists() and 'deferred 2' in (out/'poll.txt').read_text().splitlines()[-1]
    M._write(M.mode_root('devcheck')/'c'/'devcheck_c.json', dict(passed=True))
    assert M.main(['eval', '--mode', 'devcheck', *ARGS, '--active', '0']) == M.EXIT_EVENT
    assert M.hold_path('devcheck').exists()


# ---- δ ------------------------------------------------------------------------------------------------------------------------------------------
def test_delta_is_measured_from_the_new_qai_line_and_refuses_what_it_cannot_vouch_for():
    qai = next(iter(M.QAI_TEXTS))
    state = dict(lines=[(10., qai)], up=iter([100., 100.2, 200., 200.1, 300., 300.3]))

    def journal():
        return kern(*state['lines'])

    def poke():
        state['lines'].append((state['last_u']+.9, qai))         # the new qai line at kernel time u0 + .9

    def up():
        state['last_u'] = next(state['up'])
        return state['last_u']
    doc = M.measure_delta(3, read_uptime=up, poke=poke, read_journal=journal, sleep=lambda s: None, active=lambda: [])
    # try 1: u0 100, u1 100.2, t 100.9 -> [100.9-100.2-.01, 100.9-100] = [.69, .9]; try 2: [.79, .9]; try 3: [.59, .9]
    assert [x['lo'] for x in doc['tries']] == [.69, .79, .59] and [x['hi'] for x in doc['tries']] == [.9, .9, .9]
    assert doc['delta_lo'] == .59 and doc['delta_hi'] == .9 and doc['delta_hi_live'] == round(.9+M.DRIFT, 4) and doc['watch_arg'] == '0.4400,1.0500'      # revision 4a: both ends
    assert doc['delta_lo_live'] == round(.59-M.DRIFT, 4)
    with pytest.raises(SystemExit, match='units active'):
        M.measure_delta(2, read_uptime=up, poke=poke, read_journal=journal, sleep=lambda s: None, active=lambda: ['tag-lp-hi.service'])
    with pytest.raises(SystemExit, match='no new qai line'):
        M.measure_delta(2, read_uptime=lambda: 1., poke=lambda: None, read_journal=journal, sleep=lambda s: None, active=lambda: [])      # no new qai line
    with pytest.raises(SystemExit, match='could not be read'):
        M.measure_delta(2, read_uptime=lambda: 1., poke=lambda: None, read_journal=lambda: None, sleep=lambda s: None, active=lambda: [])    # review minor 1
    with pytest.raises(SystemExit, match='at least 2'):
        M.measure_delta(1, read_uptime=lambda: 1., poke=lambda: None, read_journal=journal, sleep=lambda s: None, active=lambda: [])      # revision 4a: --tries >= 2
    late = dict(lines=[(10., qai)], up=iter([500., 500.1, 600., 600.1]), ts=iter([499.5, 599.5]))

    def poke_early():
        late['lines'].append((next(late['ts']), qai))              # a new line on the kernel clock before u0: δ < 0 -- revision 4: accepted as it is (signed), not truncated
    early = M.measure_delta(2, read_uptime=lambda: next(late['up']), poke=poke_early, read_journal=lambda: kern(*late['lines']), sleep=lambda s: None, active=lambda: [])
    assert (early['delta_lo'], early['delta_hi']) == (-0.61, -0.5)
    assert M.merge_delta(dict(delta_lo=.6, delta_hi=.7), dict(delta_lo=.65, delta_hi=.8)) == dict(delta_lo=.6, delta_hi=.8, arg='0.6000,0.8000', n=2, spread=-.05)
    assert M.merge_delta(dict(delta_lo=.6, delta_hi=.7, delta_hi_live=.85), dict(delta_lo=.65, delta_hi=.8, delta_hi_live=.95))['watch_arg'] == '0.6000,0.9500'
    three = M.merge_delta(dict(delta_lo=.6, delta_hi=.7, delta_hi_live=.85), dict(delta_lo=.55, delta_hi=.8, delta_hi_live=.95), dict(delta_lo=.65, delta_hi=.9, delta_hi_live=1.05))
    assert three['arg'] == '0.5500,0.9000' and three['watch_arg'] == '0.5500,1.0500' and three['n'] == 3        # revision 2 diff review minor 3: pre, pre2, post
    with pytest.raises(SystemExit):
        M.merge_delta(dict(delta_lo=.6, delta_hi=.7))
    assert M.merge_delta(dict(delta_lo=-.1, delta_hi=.7), dict(delta_lo=.65, delta_hi=.8))['arg'] == '-0.1000,0.8000'      # revision 4: signed


# ---- not a unit; the watcher's place ------------------------------------------------------------------------------------------------------------
def test_the_monitor_refuses_inside_a_tag_or_seq_vp_unit(tmp_path, monkeypatch, sandbox):
    cg = tmp_path/'cgroup'
    for text, inside in (('0::/user.slice/user-1000.slice/session-93.scope\n', False),
                         ('0::/user.slice/user-1000.slice/user@1000.service/app.slice/tag-lp-he-queue.service\n', True),
                         ('0::/user.slice/user-1000.slice/user@1000.service/app.slice/run-u1.scope\n', False),
                         ('0::/user.slice/user-1000.slice/user@1000.service/app.slice/tag-lp-x.scope\n', True),
                         ('0::/user.slice/user-1000.slice/user@1000.service/app.slice/seq-vp-a.service\n', True)):
        cg.write_text(text)
        assert (OWN_UNIT(cg) is not None) is inside, text
    assert OWN_UNIT(tmp_path/'missing').startswith('(unreadable')
    monkeypatch.setattr(M, 'own_unit', lambda cgroup=None: 'tag-lp-x.service')
    with pytest.raises(SystemExit, match='never in a unit'):
        M.main(['delta', '--out', str(tmp_path/'d.json')])
    assert not M.hold_path('lpba40_test').exists()
    assert M.main(['watch', '--mode', 'lpba40_test', *ARGS, '--out', str(tmp_path/'w'), '--watch-pid', '1']) == M.EXIT_ERROR      # minor 5: a refused watch holds
    assert M.hold_path('lpba40_test').exists() and not (tmp_path/'w').exists() and not (tmp_path/'d.json').exists()


def test_the_watchers_directory_is_outside_every_mode_root_and_code_tree(sandbox, tmp_path):
    for bad in (lc.roots('lpba40_feas').root/'monitor', lc.roots('lpba40_test').root/'m', lc.roots('lpba40_test').hi_out/'x', sandbox/'lpba40_devcheck_v1'/'m',
                lc.REPO/'runs'/'x', Path(hc.MAIN_REPO)/'x'):
        assert M.out_problem(bad), bad
    assert M.out_problem(sandbox/'lpba40_dxg_monitor_v1'/'step2') is None and M.out_problem(tmp_path/'w') is None
    assert M.main(['watch', '--mode', 'lpba40_test', *ARGS, '--out', str(lc.roots('lpba40_feas').root/'m'), '--watch-pid', '1']) == M.EXIT_ERROR
    assert M.hold_path('lpba40_test').exists() and not lc.roots('lpba40_feas').root.exists()


def test_watch_polls_until_the_step_ends_writes_its_lines_and_holds_on_an_evaluator_exception(sandbox, tmp_path, monkeypatch):
    out = tmp_path/'w'
    seq = iter([True, True, True, False])                  # the first answer ends the wait for the start (--wait-start)
    monkeypatch.setattr(M, 'watched_active', lambda units, pids: next(seq))
    monkeypatch.setattr(M, 'uptime', lambda: 1000.)
    calls = []

    def fake(args, may_hold, **kw):
        calls.append((args.active, may_hold))
        if len(calls) == 2:
            raise RuntimeError('boom')
        return dict(schema=M.SCHEMA, active=bool(args.active), counts_since={}, events=[], cells=[], hold_present=False), []
    monkeypatch.setattr(M, 'evaluate_now', fake)
    monkeypatch.setattr(M.time, 'sleep', lambda s: None)
    rc = M.main(['watch', '--mode', 'lpba40_test', *ARGS, '--out', str(out), '--watch-unit', 'tag-lp-hi'])
    assert rc == 0 and calls == [(1, True), (1, True), (0, False), (0, True)]        # the last evaluation: a look first (no hold), then the judged one (review minor 3)
    poll = (out/'poll.txt').read_text().splitlines()
    assert sum(x.startswith('EVENT') for x in poll) == 1 and 'rc=4' in [x for x in poll if x.startswith('EVENT')][0] and poll[-1].startswith('END')
    assert len((out/'dxg_live.jsonl').read_text().splitlines()) == 3 and M.hold_path('lpba40_test').exists()
    assert json.loads((out/'watch_start.json').read_text())['source']['evaluator']['sha256'] == M.SOURCE['evaluator']['sha256']


def test_watch_started_before_its_unit_that_never_starts_holds_and_exits_5(sandbox, tmp_path, monkeypatch):
    """--wait-start runs out: an EVENT and the hold (review minor 2), one last evaluation (active 0), END, exit 5; the wait is bounded."""
    out = tmp_path/'w'
    monkeypatch.setattr(M, 'watched_active', lambda units, pids: False)
    monkeypatch.setattr(M, 'uptime', lambda: 1000.)
    monkeypatch.setattr(M.time, 'sleep', lambda s: None)
    calls = []
    monkeypatch.setattr(M, 'evaluate_now', lambda args, may_hold, **kw: (calls.append(args.active) or (dict(schema=M.SCHEMA, events=[], cells=[]), [])))
    assert M.main(['watch', '--mode', 'devcheck', *ARGS, '--out', str(out), '--watch-unit', 'tag-lp-devb', '--wait-start', '6']) == M.EXIT_NOT_STARTED
    poll = (out/'poll.txt').read_text().splitlines()
    assert calls == [0, 0] and poll[0].startswith('EVENT') and 'NOT_STARTED' in poll[0] and poll[-1].startswith('END') and M.hold_path('devcheck').exists()


def test_watch_needs_something_to_watch(sandbox, tmp_path):
    assert M.main(['watch', '--mode', 'lpba40_test', *ARGS, '--out', str(tmp_path/'w')]) == M.EXIT_ERROR
    assert M.hold_path('lpba40_test').exists() and (tmp_path/'w'/'poll.txt').read_text().splitlines()[-1].startswith('END')


def test_the_monitor_never_starts_or_stops_a_unit_and_never_removes_the_hold():
    """Static: no systemd-run, no systemctl start/stop/kill, a signal-0 probe only; unlink / rmdir only in the self-test's own temporary root."""
    src = (ROOT/'experiments'/'lpba'/'dxg_monitor.py').read_text()
    assert "'systemd-run'" not in src and 'os.remove' not in src and src.count('os.kill(') == 1 and 'os.kill(int(p), 0)' in src
    tree = ast.parse(src)
    commands = [n for n in ast.walk(tree) if isinstance(n, (ast.List, ast.Tuple)) and any(isinstance(e, ast.Constant) and e.value == 'systemctl' for e in n.elts)]
    assert commands and all(not ({e.value for e in n.elts if isinstance(e, ast.Constant)} & {'start', 'stop', 'kill', 'restart', 'reset-failed'}) for n in commands)
    unlinks = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ('unlink', 'rmdir')]
    allowed = {id(n) for f in tree.body if isinstance(f, ast.FunctionDef) and f.name in ('selftest', '_rmtree') for n in ast.walk(f)}
    assert unlinks and all(id(n) in allowed for n in unlinks)


def test_eval_or_watch_whose_arguments_do_not_parse_at_all_hold(sandbox, tmp_path):
    """B3 (a): an argparse failure of eval or watch (a missing option) holds and exits 4; --help and the other subcommands are not touched."""
    assert M.main(['eval', '--mode', 'lpba40_test', '--since', '900']) == M.EXIT_ERROR and M.hold_path('lpba40_test').exists()
    M.hold_path('lpba40_test').unlink()
    assert M.main(['watch', '--mode', 'lpba40_test', *ARGS]) == M.EXIT_ERROR and M.hold_path('lpba40_test').exists()
    M.hold_path('lpba40_test').unlink()
    with pytest.raises(SystemExit):
        M.main(['eval', '--mode', 'lpba40_test', '--no-hold'])
    with pytest.raises(SystemExit):
        M.main(['delta'])
    assert not M.hold_path('lpba40_test').exists()


# ---- revision 2 -----------------------------------------------------------------------------------------------------------------------------------
def test_before_writes_the_same_bytes_as_watch_and_ack_before_releases_only_that_file(sandbox, tmp_path, monkeypatch, capsys):
    """R3 (the revision 1 diff review B1): a cell of the root that started before --since with a status not ok holds every evaluation until --ack-before names cells_before.json's
    sha256; `before` writes the bytes watch writes; another sha256 does not release it."""
    root = M.mode_root('lpba40_test')
    M._xb_cell(root, 'fireants_syn', pair('S03', 'S04'), (500., 30.), win=(495., 532.), status='method_failure')
    assert M.main(['before', '--mode', 'lpba40_test', '--since', '900', '--start-epoch', '0', '--out', str(tmp_path/'cb.json')]) == 0
    digest = capsys.readouterr().out.split()[0]
    rows = json.loads((tmp_path/'cb.json').read_text())['rows']
    assert len(rows) == 1 and rows[0]['status'] == 'method_failure' and rows[0]['problem'] is True
    doc, events = run('lpba40_test', user=M._window('tag-lp-hi', 1000., 1100.), may_hold=False)
    assert any('started before the step are not ok' in e for e in doc['events']) and doc['cells_before']['problems_sha256'] == digest
    doc, events = M.evaluate('lpba40_test', since=900., deltas=(.7, .85), active=False, start_epoch=0., commit=COMMIT, kernel_text=kern(), dmesg_text=kern(journal=False),
                             user_entries=M._window('tag-lp-hi', 1000., 1100.), now_uptime=4000., now_epoch=1e12, boot_offset=0., ack_before=digest, may_hold=False)
    assert not any('before' in e for e in doc['events']) and doc['cells_before']['acknowledged'] is True
    seq = iter([True, False])
    monkeypatch.setattr(M, 'watched_active', lambda units, pids: next(seq, False))
    monkeypatch.setattr(M, 'uptime', lambda: 1000.)
    monkeypatch.setattr(M.time, 'sleep', lambda s: None)
    monkeypatch.setattr(M, 'evaluate_now', lambda args, may_hold, **k: (dict(schema=M.SCHEMA, events=[], cells=[]), []))
    assert M.main(['watch', '--mode', 'lpba40_test', *ARGS, '--out', str(tmp_path/'w'), '--watch-pid', '1']) == 0
    assert (tmp_path/'w'/'cells_before.json').read_bytes() == (tmp_path/'cb.json').read_bytes()


def test_an_ack_file_given_to_eval_releases_only_the_lead_kinds_without_kernel_events(sandbox, tmp_path, capsys):
    """I1 (i): eval --ack F; a lead entry for a status in a cell without kernel events is accepted, one for a W line (kernel) is not; with BZ's words it is."""
    root = M.mode_root('lpba40_test')
    M._xb_cell(root, 'fireants_syn', pair('S03', 'S04'), (1000., 30.), win=(995., 1032.), status='method_failure')
    user = M._window('tag-lp-x-fireants_syn-S03-S04', 995., 1032.4)
    uf, kf, df, ack = tmp_path/'u.json', tmp_path/'k.txt', tmp_path/'d.txt', tmp_path/'ack.json'
    uf.write_text('\n'.join(json.dumps(e) for e in user))
    cell = 'tag-lp-x-fireants_syn-S03-S04#1'
    files = ['--journal-file', str(kf), '--dmesg-file', str(df), '--user-journal-file', str(uf), '--ack', str(ack)]

    def ev(lines, entries, capsys=None):
        kf.write_text(kern(*lines))
        df.write_text(kern(*lines, journal=False))
        ack.write_text(json.dumps(dict(schema=M.ACK_SCHEMA, entries=entries)))
        return M.evaluate('lpba40_test', since=900., deltas=(.7, .85), active=False, start_epoch=0., commit=COMMIT, kernel_text=kf.read_text(), dmesg_text=df.read_text(),
                          user_entries=user, now_uptime=4000., now_epoch=1e12, boot_offset=0., ack_path=ack, may_hold=False)[0]
    def ids(doc):
        return {(e['kind'], e['cell']): e['id'] for e in doc['event_items']}
    base = ids(ev([], []))
    lead = dict(event_id=base[('status', cell)], cell=cell, kind='status', decision_log='docs/decision_log.md line 759', bz_words='')
    assert ev([], [lead])['events'] == []
    assert ev([], [dict(lead, event_id='0'*16)])['events'] != []                # an id names one event: another id releases nothing
    assert ev([], [dict(lead, kind='wp')])['ack']['rejected'][0]['why'].startswith('event_id')      # cell and kind must be the event's
    with_w = ids(ev([(1012., M.W_TEXT)], []))
    assert with_w[('status', cell)] == base[('status', cell)]                    # the status event's id does not depend on the kernel lines
    w = dict(lead, event_id=with_w[('wp', cell)], kind='wp')
    doc = ev([(1012., M.W_TEXT)], [lead, w])
    assert len(doc['ack']['rejected']) == 2 and len(doc['events']) == 2          # a kernel event in the cell: neither the status nor the W the lead's own
    doc = ev([(1012., M.W_TEXT)], [dict(lead, bz_words='BZ 原话'), dict(w, bz_words='BZ 原话')])
    assert doc['events'] == [] and len(doc['acked']) == 2
    doc = ev([(1012., M.W_TEXT), (1014., M.W_TEXT)], [dict(lead, bz_words='BZ 原话'), dict(w, bz_words='BZ 原话')])
    assert len(doc['events']) == 1 and 'kernel 1014.000000' in doc['events'][0]     # a later W of the same kind and cell is judged anew
    kf.write_text(kern())
    df.write_text(kern(journal=False))
    ack.write_text(json.dumps(dict(schema=M.ACK_SCHEMA, entries=[lead])))
    assert M.main(['eval', '--mode', 'lpba40_test', *ARGS, *files]) == 0, capsys.readouterr().out[-3000:]
    ack.write_text('not json')
    assert M.main(['eval', '--mode', 'lpba40_test', *ARGS, *files]) == M.EXIT_EVENT          # an ack file that does not parse is an event


def test_eval_pending_and_expect_unit_check_the_journal_only_cells(sandbox, tmp_path):
    """I2 (the revision 1 diff review): the eval after compare reads the units of pending.json; an empty user journal, or the unit missing from it, holds."""
    pending = tmp_path/'pending.json'
    pending.write_text(json.dumps(dict(units=['tag-lp-devc'], deferred=['tag-lp-devc#1: x'])))
    M._write(M.mode_root('devcheck')/'c'/'devcheck_c.json', dict(passed=True))
    uf, kf, df = tmp_path/'u.json', tmp_path/'k.txt', tmp_path/'d.txt'
    kf.write_text(kern())
    df.write_text(kern(journal=False))
    files = ['--journal-file', str(kf), '--dmesg-file', str(df), '--user-journal-file', str(uf)]
    uf.write_text('')
    assert M.main(['eval', '--mode', 'devcheck', *ARGS, *files, '--pending', str(pending)]) == M.EXIT_EVENT
    uf.write_text('\n'.join(json.dumps(e) for e in M._window('tag-lp-devb', 1000., 1100.)))
    assert M.main(['eval', '--mode', 'devcheck', *ARGS, *files, '--pending', str(pending)]) == M.EXIT_EVENT      # devc not in the journal
    uf.write_text('\n'.join(json.dumps(e) for e in M._window('tag-lp-devc', 1000., 1100.)))
    assert M.main(['eval', '--mode', 'devcheck', *ARGS, *files, '--pending', str(pending)]) == 0
    assert M.main(['eval', '--mode', 'devcheck', *ARGS, *files, '--expect-unit', 'tag-lp-devb']) == M.EXIT_EVENT
    pending.write_text('{}')
    assert M.main(['eval', '--mode', 'devcheck', *ARGS, *files, '--pending', str(pending)]) == M.EXIT_ERROR


def _watch_once(monkeypatch, tmp_path, out, lines, user, *, since='900', extra=(), delta='0.7,0.85'):
    """One watch over fake sources: active for one poll, then the last evaluation."""
    seq = iter([True, True, False])
    monkeypatch.setattr(M, 'watched_active', lambda units, pids: next(seq, False))
    monkeypatch.setattr(M, 'uptime', lambda: 4000.)
    monkeypatch.setattr(M.time, 'sleep', lambda s: None)
    monkeypatch.setattr(M, 'read_kernel_journal', lambda: kern(*lines))
    monkeypatch.setattr(M, 'read_dmesg', lambda: kern(*lines, journal=False))
    monkeypatch.setattr(M, 'read_user_journal', lambda start_epoch: (list(user), None))
    argv = ['watch', '--mode', 'lpba40_test', '--since', since, '--delta', delta, '--start-epoch', '0', '--commit', COMMIT, '--out', str(out), '--watch-unit', 'tag-lp-hi',
            '--ack', str(tmp_path/'ack.json'), *extra]
    return M.main(argv)


def test_a_watcher_that_takes_over_scans_from_the_same_start(sandbox, tmp_path, monkeypatch):
    """I1 (ii): --continue-from: another --since, a narrower δ, another ack file or an old watcher still running are refused (hold, exit 4); with the same start a kernel line written after
    the old watcher's last poll and before the new one started is judged by the new one -- no kernel line falls outside every scan."""
    user = M._window('tag-lp-hi', 1000., 3000.)
    M._write(lc.roots('lpba40_test').hi_out/'hi_record.json', dict(passed=True))
    old = tmp_path/'w1'
    assert _watch_once(monkeypatch, tmp_path, old, [], user) == 0
    assert not M.hold_path('lpba40_test').exists()
    later = [(2500., 'hv_balloon: unhandled message: type: 7')]          # in hi's window, written after the old watcher ended
    for extra, since, delta in (((), '950', '0.7,0.85'), ((), '900', '0.75,0.85'), (('--ack', str(tmp_path/'other.json')), '900', '0.7,0.85')):
        assert _watch_once(monkeypatch, tmp_path, tmp_path/f'r{since}{delta}', later, user, since=since, delta=delta, extra=('--continue-from', str(old), *extra)) == M.EXIT_ERROR
        assert M.hold_path('lpba40_test').exists()
        M.hold_path('lpba40_test').unlink()
    new = tmp_path/'w2'
    assert _watch_once(monkeypatch, tmp_path, new, later, user, delta='0.6,0.9', extra=('--continue-from', str(old))) == 0
    assert M.hold_path('lpba40_test').exists() and 'non-dxg kernel line' in (new/'poll.txt').read_text()
    M.hold_path('lpba40_test').unlink()
    running = tmp_path/'w3'
    running.mkdir()
    (running/'watch_start.json').write_text((old/'watch_start.json').read_text())
    (running/'poll.txt').write_text('a poll line\n')
    assert _watch_once(monkeypatch, tmp_path, tmp_path/'w4', [], user, extra=('--continue-from', str(running))) == M.EXIT_ERROR


def test_the_lead_acknowledges_while_the_watcher_runs(sandbox, tmp_path, monkeypatch):
    """I1 (i), (iv): the watcher holds on a method failure; the lead writes the ack file and the next poll takes the event off (the hold may then be removed and is not touched again)."""
    root = M.mode_root('lpba40_test')
    M._xb_cell(root, 'fireants_syn', pair('S03', 'S04'), (1000., 30.), win=(995., 1032.), status='method_failure')
    user = M._window('tag-lp-x-fireants_syn-S03-S04', 995., 1032.4) + [M._user(990., 'tag-lp-x-chain', 'Started tag-lp-x-chain.service - c')]
    ack = tmp_path/'ack.json'
    out = tmp_path/'w'
    polls = iter([True, True, True, True, False])
    calls = {'n': 0}

    def active(units, pids):
        calls['n'] += 1
        if calls['n'] == 3:                      # after the first poll held: the lead reads the event's id, disposes, writes the ack, removes the hold
            first = json.loads((out/'dxg_live.jsonl').read_text().splitlines()[0])
            item = next(e for e in first['event_items'] if e['kind'] == 'status')
            ack.write_text(json.dumps(dict(schema=M.ACK_SCHEMA, entries=[dict(event_id=item['id'], cell=item['cell'], kind='status',
                                                                              decision_log='docs/decision_log.md line 759 (method failure)', bz_words='')])))
            M.hold_path('lpba40_test').unlink()
        return next(polls, False)
    monkeypatch.setattr(M, 'watched_active', active)
    monkeypatch.setattr(M, 'uptime', lambda: 4000.)
    monkeypatch.setattr(M.time, 'sleep', lambda s: None)
    monkeypatch.setattr(M, 'read_kernel_journal', lambda: kern())
    monkeypatch.setattr(M, 'read_dmesg', lambda: kern(journal=False))
    monkeypatch.setattr(M, 'read_user_journal', lambda start_epoch: (list(user), None))
    assert M.main(['watch', '--mode', 'lpba40_test', *ARGS, '--out', str(out), '--watch-unit', 'tag-lp-x-chain', '--ack', str(ack)]) == 0
    lines = [json.loads(x) for x in (out/'dxg_live.jsonl').read_text().splitlines()]
    assert lines[0]['hold_touched_now'] is True and any('effective status not ok' in e for e in lines[0]['events'])
    assert all(not x['events'] and len(x['acked']) == 1 for x in lines[1:]) and not M.hold_path('lpba40_test').exists()


def test_a_watcher_stopped_by_sigterm_holds_and_writes_its_end(sandbox, tmp_path, monkeypatch):
    """I1 (iv): kill <watch.pid> (SIGTERM) ends the watcher through the same path: hold, EVENT, END exit 4, so that a takeover (--continue-from) finds the END line."""
    import os
    import signal as sig
    monkeypatch.setattr(M, 'watched_active', lambda units, pids: True)
    monkeypatch.setattr(M, 'uptime', lambda: 1000.)

    def term(*a, **k):
        os.kill(os.getpid(), sig.SIGTERM)
        return dict(schema=M.SCHEMA, events=[], cells=[]), []
    monkeypatch.setattr(M, 'evaluate_now', term)
    old = sig.getsignal(sig.SIGTERM)
    try:
        assert M.main(['watch', '--mode', 'lpba40_test', *ARGS, '--out', str(tmp_path/'w'), '--watch-unit', 'tag-lp-hi']) == M.EXIT_ERROR
    finally:
        sig.signal(sig.SIGTERM, old)
    poll = (tmp_path/'w'/'poll.txt').read_text().splitlines()
    assert M.hold_path('lpba40_test').exists() and 'signal' in poll[-2] and poll[-1].startswith('END') and 'exit=4' in poll[-1]
    assert json.loads((tmp_path/'w'/'watch_end.json').read_text())['exit'] == M.EXIT_ERROR


# ---- revision 3 -----------------------------------------------------------------------------------------------------------------------------------
def test_the_watchers_of_one_step_share_a_monitor_root(sandbox, tmp_path, monkeypatch):
    """The lead's ruling ~4:38 PM CST (minor): under $MON an earlier watcher of the same mode refuses a new one without --continue-from; one without its END line refuses any new one;
    a refused watcher writes nothing into the other's directory."""
    mon = tmp_path/'lpba40_dxg_monitor_v1'/'s04_method'
    user = M._window('tag-lp-hi', 1000., 3000.)
    M._write(lc.roots('lpba40_test').hi_out/'hi_record.json', dict(passed=True))
    assert _watch_once(monkeypatch, tmp_path, mon, [], user) == 0
    before = (mon/'poll.txt').read_text()
    assert _watch_once(monkeypatch, tmp_path, mon, [], user) == M.EXIT_ERROR                    # the same directory again
    assert (mon/'poll.txt').read_text() == before and M.hold_path('lpba40_test').exists()
    M.hold_path('lpba40_test').unlink()
    assert _watch_once(monkeypatch, tmp_path, mon/'r2', [], user) == M.EXIT_ERROR              # a takeover without --continue-from
    M.hold_path('lpba40_test').unlink()
    assert _watch_once(monkeypatch, tmp_path, mon/'r3', [], user, extra=('--continue-from', str(mon))) == 0
    (mon/'r3'/'poll.txt').write_text((mon/'r3'/'poll.txt').read_text()+'a poll line after END\n')      # as if r3 still ran
    assert _watch_once(monkeypatch, tmp_path, mon/'r4', [], user, extra=('--continue-from', str(mon))) == M.EXIT_ERROR
    other = tmp_path/'lpba40_dxg_monitor_v1'/'feas'
    assert _watch_once(monkeypatch, tmp_path, other, [], user) == 0                              # another step's root: no conflict


def test_event_ids_are_stable_and_name_one_event(sandbox):
    """B1 of the revision 2 diff review: the same inputs give the same ids; the id of a running cell's event does not change when the cell ends; two W lines have two ids."""
    root = M.mode_root('lpba40_test')
    M._xb_cell(root, 'fireants_syn', pair('S03', 'S04'), (1000., 30.), win=(995., 1032.), status='method_failure')
    user = M._window('tag-lp-x-fireants_syn-S03-S04', 995., 1032.4)
    lines = [(1580., M.W_TEXT), (1590., M.W_TEXT)]
    a = run('lpba40_test', lines, user, may_hold=False)[0]['event_items']
    b = run('lpba40_test', lines, user, may_hold=False)[0]['event_items']
    assert [e['id'] for e in a] == [e['id'] for e in b] and len({e['id'] for e in a}) == len(a) == 3
    assert len([e for e in a if e['kind'] == 'wp']) == 2
    jitter = M.evaluate('lpba40_test', since=900., deltas=(.7, .85), active=False, start_epoch=0., commit=COMMIT, kernel_text=kern(*lines), dmesg_text=kern(*lines, journal=False),
                        user_entries=user, now_uptime=4000.3, now_epoch=1e12+7, boot_offset=3e-7, may_hold=False)[0]['event_items']
    assert [e['id'] for e in jitter] == [e['id'] for e in a]          # the boot offset jitters between polls; the ids take the journal's own microseconds


def test_one_source_event_acknowledged_leaves_the_other_source_events_holding(sandbox):
    """Revision 2 diff review A1: BZ acknowledges "kernel journal earliest line after the start" by its event_id; "the journal short of dmesg" and "0 user entries" of the same
    evaluation (same kind source, no cell) still hold."""
    import os
    ackf = M.mode_root('lpba40_test').parent/'ack_a1.json'
    late = '\n'.join(['[  950.000000] host kernel: x'])                     # the journal's first line after --since 900
    dm = '\n'.join(['[    0.000000] Linux version x', M._kline(1500., M.W_TEXT, False)])
    args = dict(since=900., deltas=(.7, .85), active=False, start_epoch=0., commit=COMMIT, kernel_text=late, dmesg_text=dm, user_entries=[], now_uptime=4000., now_epoch=1e12,
                boot_offset=0.)
    doc = M.evaluate('lpba40_test', may_hold=False, **args)[0]
    first = next(e for e in doc['event_items'] if 'earliest line' in e['text'])
    assert sum(e['kind'] == 'source' for e in doc['event_items']) == 3
    M._write(ackf, dict(schema=M.ACK_SCHEMA, entries=[dict(event_id=first['id'], cell='-', kind='source', decision_log='line 7xx', bz_words='BZ 原话')]))
    doc, events = M.evaluate('lpba40_test', ack_path=ackf, **args)
    assert len(doc['acked']) == 1 and len(events) == 2 and M.hold_path('lpba40_test').exists()
    assert {e['kind'] for e in events} == {'source'} and not any('earliest line' in e['text'] for e in events)
    os.remove(ackf)


# ---- revision 3b ----------------------------------------------------------------------------------------------------------------------------------
def test_a_condition_acknowledged_once_holds_again_when_it_comes_back(sandbox, tmp_path, monkeypatch):
    """Revision 3 diff review I1: the watcher carries each condition's episode start from poll to poll.  BZ acknowledges "dmesg not readable"; while it lasts the acknowledgement holds;
    it clears; it comes back: a new episode, a new event id, hold."""
    user = M._window('tag-lp-hi', 1000., 3000.)
    M._write(lc.roots('lpba40_test').hi_out/'hi_record.json', dict(passed=True))
    ack = tmp_path/'ack.json'
    out = tmp_path/'w'
    plan = iter([(True, False), (True, False), (True, True), (True, False), (False, True)])      # (active, dmesg readable) per poll
    state = {'poll': 0, 'readable': False, 'up': 4000.}

    def active(units, pids):
        state['poll'] += 1
        if state['poll'] == 1:
            return True                         # the wait for the start
        act, readable = next(plan, (False, True))
        state['readable'] = readable
        state['up'] += 20.
        if state['poll'] == 3:                  # after the first poll held: BZ's words recorded, the hold removed
            first = json.loads((out/'dxg_live.jsonl').read_text().splitlines()[0])
            item = next(e for e in first['event_items'] if 'dmesg not readable' in e['text'])
            ack.write_text(json.dumps(dict(schema=M.ACK_SCHEMA, entries=[dict(event_id=item['id'], cell='-', kind='source', decision_log='line 7xx', bz_words='BZ 原话')])))
            M.hold_path('lpba40_test').unlink()
        return act
    monkeypatch.setattr(M, 'watched_active', active)
    monkeypatch.setattr(M, 'uptime', lambda: state['up'])
    monkeypatch.setattr(M.time, 'sleep', lambda s: None)
    monkeypatch.setattr(M, 'read_kernel_journal', lambda: kern())
    monkeypatch.setattr(M, 'read_dmesg', lambda: kern(journal=False) if state['readable'] else None)
    monkeypatch.setattr(M, 'read_user_journal', lambda start_epoch: (list(user), None))
    M.main(['watch', '--mode', 'lpba40_test', *ARGS, '--out', str(out), '--watch-unit', 'tag-lp-hi', '--ack', str(ack), '--interval', '0'])
    lines = [json.loads(x) for x in (out/'dxg_live.jsonl').read_text().splitlines()]
    dm = [[e['id'] for e in x['event_items'] if 'dmesg not readable' in e['text']] for x in lines]
    acked = [[e['id'] for e in x['acked'] if 'dmesg not readable' in e['text']] for x in lines]
    assert dm[0] and acked[1] == dm[0] and not dm[1]                      # poll 1 held; poll 2 the same episode, acknowledged
    assert not dm[2] and not acked[2]                                     # poll 3 readable: the condition cleared
    assert dm[3] and dm[3] != dm[0] and M.hold_path('lpba40_test').exists()     # poll 4 back: a new id, hold


def test_problem_rows_carry_their_reading_and_unit_end(sandbox):
    """Revision 3 diff review minor 2: a problem row whose unit reading changes gives another problems_sha256."""
    root = M.mode_root('lpba40_test')
    rec = M._xb_cell(root, 'fireants_syn', pair('S03', 'S04'), (500., 30.), win=(495., 532.), status='method_failure')
    a = M.before_cells('lpba40_test', 900., 0.)
    M._write(rec/'unit'/'unit_reading.json', dict(status='infra_failure', kind='x', systemd_result='failed', systemd_run_exit=1))
    b = M.before_cells('lpba40_test', 900., 0.)
    assert a['problems'] == b['problems'] == 1 and a['problems_sha256'] != b['problems_sha256'] and a['rows'][0]['reading_sha256'] != b['rows'][0]['reading_sha256']


# ---- revision 4: a signed δ ---------------------------------------------------------------------------------------------------------------------------
def _measure(rows, tries=None):
    """measure_delta over fake tries: rows = [(u0, u1, t)] (the new qai line's kernel time t)."""
    qai = next(iter(M.QAI_TEXTS))
    state = dict(lines=[(1., qai)], ups=iter([x for u0, u1, _ in rows for x in (u0, u1)]), ts=iter([t for _, _, t in rows]))

    def poke():
        state['lines'].append((next(state['ts']), qai))
    return M.measure_delta(tries or len(rows), read_uptime=lambda: next(state['ups']), poke=poke, read_journal=lambda: kern(*state['lines']), sleep=lambda s: None,
                           active=lambda: [])


def test_delta_takes_this_boots_negative_offset():
    """Revision 4 (decision_log 2026-11-20 ~6:45 PM CST): the three first tries of runs/20261120-lpba-run/delta_diag_20261120.jsonl -- the new qai line stamped about 31.6 s before u0 --
    give a negative δ, written as it is (no floor, no truncation)."""
    doc = _measure([(13388.66, 13388.76, 13357.085771), (13463.82, 13463.91, 13432.223184), (13539.03, 13539.12, 13507.439083)])
    assert [x['lo'] for x in doc['tries']] == [-31.6842, -31.6968, -31.6909] and [x['hi'] for x in doc['tries']] == [-31.5742, -31.5968, -31.5909]
    assert doc['delta_lo'] == -31.6968 and doc['delta_hi'] == -31.5742 and doc['watch_arg'] == '-31.8468,-31.4242' and doc['spread'] <= M.DELTA_AGREE


def test_delta_refuses_tries_that_do_not_agree():
    with pytest.raises(SystemExit, match='do not agree'):
        _measure([(100., 100.1, 68.4), (200., 200.1, 170.0)])               # δ about -31.6, then about -30.0: max lo - min hi = 1.49 s
    doc = _measure([(100., 100.1, 68.4), (200., 200.1, 168.7)])              # .3 s apart: they agree
    assert doc['spread'] <= M.DELTA_AGREE


def test_merge_delta_and_the_command_line_take_a_negative_delta(sandbox, tmp_path):
    m = M.merge_delta(dict(delta_lo=-31.69, delta_hi=-31.58, delta_lo_live=-31.84, delta_hi_live=-31.43),
                      dict(delta_lo=-31.70, delta_hi=-31.57, delta_lo_live=-31.85, delta_hi_live=-31.42))
    assert m['arg'] == '-31.7000,-31.5700' and m['watch_arg'] == '-31.8500,-31.4200' and m['spread'] == round(-31.69 - -31.58, 4)
    with pytest.raises(SystemExit, match='交总负责人'):                  # revision 4a: measurements of one step that do not agree are not merged
        M.merge_delta(dict(delta_lo=-31.69, delta_hi=-31.58), dict(delta_lo=-30.10, delta_hi=-30.00))
    with pytest.raises(SystemExit):
        M.merge_delta(dict(delta_lo=-31.5, delta_hi=-31.6), dict(delta_lo=-31.5, delta_hi=-31.6))
    assert M.parse_delta('-31.69,-31.57') == (-31.69, -31.57)
    with pytest.raises(ValueError):
        M.parse_delta('-31.57,-31.69')
    kf, df, uf = tmp_path/'k.txt', tmp_path/'d.txt', tmp_path/'u.json'
    kf.write_text(kern())
    df.write_text(kern(journal=False))
    uf.write_text('\n'.join(json.dumps(e) for e in M._window('tag-lp-prep', 1000., 1100.)))
    files = ['--journal-file', str(kf), '--dmesg-file', str(df), '--user-journal-file', str(uf)]
    for form in (['--delta', '-31.69,-31.57'], ['--delta=-31.69,-31.57']):
        argv = ['--since', '900', *form, '--start-epoch', '0', '--commit', COMMIT]
        assert M.main(['eval', '--mode', 'lpba40_test', *argv, *files]) == 0
    assert not M.hold_path('lpba40_test').exists()


def test_a_negative_delta_reads_the_windows_as_a_positive_one(sandbox):
    """The same events with δ = +.7..+.85 and δ = -31.69..-31.57 (each kernel time shifted by the difference): the same judgements; the scan starts |δ_lo| before SINCE."""
    root = M.mode_root('lpba40_test')
    M._xb_cell(root, 'fireants_syn', pair('S03', 'S04'), (1000., 30.), win=(995., 1032.))
    user = M._window('tag-lp-x-fireants_syn-S03-S04', 995., 1032.4)
    targets = [(1030.1, M.W_TEXT), (1012., M.W_TEXT), (1015., sorted(M.EAGAIN_TEXTS)[0]), (1010., 'hv_balloon: unhandled message: type: 7'), (1580., M.W_TEXT)]
    pos = run('lpba40_test', [(round(u+.775, 6), x) for u, x in targets], user, deltas=(.7, .85), may_hold=False)[0]
    neg = run('lpba40_test', [(round(u-31.63, 6), x) for u, x in targets], user, deltas=(-31.69, -31.57), may_hold=False)[0]
    assert [(e['kind'], e['cell']) for e in pos['event_items']] == [(e['kind'], e['cell']) for e in neg['event_items']]
    assert pos['judged'][next(iter(pos['judged']))]['passes'] is False and pos['eagain']['recorded'] == neg['eagain']['recorded'] == 1
    assert pos['kernel_scan_from'] == 900. and neg['kernel_scan_from'] == 900. - 31.69


def test_nvidia_smi_is_the_machine_gates_full_path():
    import hi_core
    src = (ROOT/'experiments'/'lpba'/'dxg_monitor.py').read_text()
    assert "subprocess.run([str(hc._hi().SMI)]" in src and "['nvidia-smi']" not in src and str(hi_core.SMI) == '/usr/lib/wsl/lib/nvidia-smi'


# ---- revision 5: a record directory of this step whose unit has not started yet (feas-r2, 2026-11-21 13:01:29 CST) ---------------------------------------
def _feas_gap(root):
    """feas-r2's two cells: S01<-S02's method cell done (step4.json, unit_end.json), S02<-S01's record directory made by the queue before its machine gate, no unit/ yet."""
    M._hb_cell(root, 'method', pair('S01', 'S02'), (2000., 2100.), mode='lpba40_feas')
    nxt = root/'method'/pair('S02', 'S01')
    nxt.mkdir(parents=True)
    return nxt


FEAS_USER = M._window('tag-lp-feas-he-S01-S02', 1999.9, 2100.4)


def test_a_record_directory_made_since_the_step_start_without_its_unit_yet_is_not_a_cell_before_the_step(sandbox):
    """feas-r2 13:01:29 CST: method/LPBA40_S02__LPBA40_S01/ made at 13:01:19.390 (with S01<-S02's step4.json), unit/ and unit_start.json at 13:01:49.74; the evaluations between
    read it as a cell before the step with its records missing (dxg_monitor.py:719-721 at F'') and held.  Now: no event, listed as not started -- for the whole gap, which the
    method queue's machine gate can stretch to 15 min (no grace: the directory's creation time is the proof)."""
    root = M.mode_root('lpba40_feas')
    epoch = time.time() - 5.
    nxt = _feas_gap(root)
    for now in (4000., 4300., 4900.):                     # 0, 5 and 15 minutes into the gap
        doc, events = run('lpba40_feas', user=FEAS_USER, active=True, now_uptime=now, start_epoch=epoch)
        assert not events and doc['events'] == [], doc['events']
        assert doc['cells_before'] == dict(problems_sha256=M.problems_sha256('lpba40_feas', []), rows=0, problems=0, acknowledged=False, not_started=1)
    before = M.before_cells('lpba40_feas', 900., epoch)
    assert before['rows'] == [] and [(x['kind'], x['record']) for x in before['not_started']] == [('method', str(nxt))] and before['not_started'][0]['birth'] >= epoch
    M._supervisor(nxt/'unit', 'tag-lp-feas-he-S02-S01', (2200., None))                        # the unit starts: a cell of this step (disk_cells), still no row
    assert M.before_cells('lpba40_feas', 900., epoch)['rows'] == [] and M.before_cells('lpba40_feas', 900., epoch)['not_started'] == []


QUEUE = M._window('tag-lp-feas-he-queue', 1990.) + [M._user(2400., 'tag-lp-feas-he-queue', 'tag-lp-feas-he-queue.service: Deactivated successfully.'),
                                                   M._user(2400.1, 'tag-lp-feas-he-queue', 'tag-lp-feas-he-queue.service: Consumed 1s CPU time.')]


def test_the_method_queue_ended_by_sigterm_in_its_gate_holds_at_the_last_evaluation(sandbox):
    """Revision 5a (B1 of the revision 5 review): the queue made method/<pair>/ and waited at its machine gate; SIGTERM (systemctl stop, kill) ended it -- no STOP line, no
    failure line (systemd reads SIGTERM of a simple service as a clean exit), no refusal.  While the step runs: listed, no event; at the last evaluation: one not_started event, a
    hold, not on the lead's white list."""
    root = M.mode_root('lpba40_feas')
    epoch = time.time() - 5.
    nxt = _feas_gap(root)
    doc, events = run('lpba40_feas', user=FEAS_USER + QUEUE, active=True, now_uptime=2300., start_epoch=epoch)
    assert doc['events'] == [] and doc['cells_before']['not_started'] == 1
    hold = M.hold_path('lpba40_feas')
    assert not hold.exists()
    doc, events = run('lpba40_feas', user=FEAS_USER + QUEUE, active=False, start_epoch=epoch)
    assert len(doc['events']) == 1 and f'record directory {M.mask(str(nxt))} made since the step start' in doc['events'][0] and 'no unit attempt in it' in doc['events'][0]
    item, = doc['event_items']
    assert item['kind'] == 'not_started' and item['lead'] is False and item['cell'] == f"method/{nxt.name}"
    assert doc['last_eval_open']['not_started'] == [M.mask(str(nxt))] and doc['hold_touched_now'] is True and hold.exists()
    assert doc['cells_before']['problems'] == 0 and not any('started before the step' in e for e in doc['events'])
    entry = dict(event_id=item['id'], cell=item['cell'], kind='not_started', decision_log='2026-11-21 x', bz_words='')     # the lead alone cannot release it
    acc, rej = M.judge_acks([entry], [dict(item, category='state', text=item['text'])], {}, False)
    assert acc == [] and len(rej) == 1 and "BZ's words needed" in rej[0]['why']
    acc, rej = M.judge_acks([dict(entry, bz_words='按建议')], [dict(item, category='state', text=item['text'])], {}, False)
    assert len(acc) == 1 and rej == []


def test_a_normal_close_leaves_not_started_empty_and_no_event(sandbox):
    """The same step when S02<-S01's unit ran and ended: a queue goes on only after systemd-run --wait returns, so the last evaluation finds every directory with its attempt."""
    root = M.mode_root('lpba40_feas')
    epoch = time.time() - 5.
    _feas_gap(root)
    M._hb_cell(root, 'method', pair('S02', 'S01'), (2200., 2300.), mode='lpba40_feas')
    doc, events = run('lpba40_feas', user=FEAS_USER + M._window('tag-lp-feas-he-S02-S01', 2199.9, 2300.4) + QUEUE, active=False, start_epoch=epoch)
    assert doc['events'] == [] and not events and doc['cells_before']['not_started'] == 0 and doc['last_eval_open']['not_started'] == []
    assert not M.hold_path('lpba40_feas').exists()


@pytest.mark.parametrize('kind,base', [('method', 'method'), ('configc', 'configc'), ('xb', 'fireants_syn'), ('xb', 'convexadam_v4seg')])
def test_every_kind_of_record_directory_made_before_the_step_without_any_attempt_still_holds(sandbox, kind, base):
    """Stricter, not looser: the same directory made before --start-epoch (a cell refused before its lock in an earlier run) is a problem row and holds until --ack-before; made
    since, it is not.  The review's directories are the method's: no unit_review* there is no row either way."""
    root = M.mode_root('lpba40_test')
    rec = root/base/pair('S03', 'S04')
    rec.mkdir(parents=True)
    made = M.birth(rec)
    assert made is not None
    late = M.before_cells('lpba40_test', 900., made + 1.)                                     # --start-epoch after the directory's creation: made before the step
    assert [(r['kind'], r['record'], r['missing'], r['problem']) for r in late['rows']] == [(kind, str(rec), ['unit*'], True)] and late['not_started'] == []
    doc, events = run('lpba40_test', user=M._window('tag-lp-hi', 1000., 1100.), start_epoch=made + 1., may_hold=False)
    assert any('1 cell(s) that started before the step are not ok' in e for e in doc['events']) and doc['cells_before']['problems_sha256'] == late['problems_sha256']
    early = M.before_cells('lpba40_test', 900., made - 1.)                                    # made since the step start
    assert early['rows'] == [] and [x['record'] for x in early['not_started']] == [str(rec)]
    doc, events = run('lpba40_test', user=M._window('tag-lp-hi', 1000., 1100.), start_epoch=made - 1., may_hold=False)
    assert not any('started before the step' in e for e in doc['events'])


def test_a_creation_time_that_cannot_be_read_keeps_the_problem_row(sandbox, monkeypatch):
    root = M.mode_root('lpba40_feas')
    epoch = time.time() - 5.
    nxt = _feas_gap(root)
    monkeypatch.setattr(M, 'birth', lambda p: None)
    before = M.before_cells('lpba40_feas', 900., epoch)
    assert [(r['record'], r['problem']) for r in before['rows']] == [(str(nxt), True)] and before['not_started'] == []
    doc, events = run('lpba40_feas', user=FEAS_USER, active=True, start_epoch=epoch, may_hold=False)
    assert any('started before the step are not ok' in e for e in doc['events'])


def test_the_mtime_of_an_old_directory_does_not_make_it_this_steps(sandbox):
    """mtime and ctime move when anything is written into the directory (a refusal copied in, a lock beside it); the creation time does not."""
    root = M.mode_root('lpba40_test')
    rec = root/'configc'/pair('S05', 'S06')
    rec.mkdir(parents=True)
    made = M.birth(rec)
    time.sleep(.05)
    (rec/'refusal_copy.json').write_text('{}')                  # written after the step start
    assert M.mtime(rec) > made + .02 and M.birth(rec) == made
    epoch = made + .01                                          # the step started between the creation and the write
    assert M.mtime(rec) >= epoch
    assert [r['record'] for r in M.before_cells('lpba40_test', 900., epoch)['rows']] == [str(rec)]


def _stat_birth(path):
    """coreutils stat's own birth time of path (not following a link), to the nanosecond: %W seconds and %w's fraction."""
    import subprocess
    w, human = subprocess.run(['/usr/bin/stat', '-c', '%W|%w', str(path)], capture_output=True, text=True, check=True).stdout.strip().split('|')
    return int(w), int(human.split(' ')[1].split('.')[1])


def test_birth_reads_statx(tmp_path):
    t0 = time.time()
    d = tmp_path/'lpf5_birth'
    d.mkdir()
    t1 = time.time()
    b = M.birth(d)
    assert b is not None and t0 - .05 <= b <= t1 + .01             # the file system's coarse clock may lag by a tick
    assert M.birth(d/'missing') is None


def test_birth_is_stx_btime_to_the_nanosecond_and_does_not_move(tmp_path):
    """Revision 5a (the revision 5 review m3): pins the offset 80 -- after a write (mtime, ctime) and a listing (atime) the birth is unchanged and equals coreutils stat's."""
    d = tmp_path/'lpf5_btime'
    d.mkdir()
    b0 = M.birth(d)
    time.sleep(.05)
    (d/'x.json').write_text('{}')
    list(d.iterdir())
    time.sleep(.05)
    os_stat = d.stat()
    assert os_stat.st_mtime > b0 + .02 and os_stat.st_ctime > b0 + .02
    sec, nsec = _stat_birth(d)
    assert M.birth(d) == b0 and abs(b0 - (sec + nsec*1e-9)) < 1e-6


def test_birth_is_none_where_statx_gives_no_btime():
    """procfs answers statx with rc 0 and a mask without STATX_BTIME (its btime field zero): read as unknown, not as 1970 (the mask check; the revision 5 review m3)."""
    assert M.birth('/proc/self') is None and M.birth('/proc') is None


def test_birth_of_a_symbolic_link_is_the_links_own(tmp_path):
    """AT_SYMLINK_NOFOLLOW (the revision 5 review m3; its limit m2 is in the Known limits): a link made later than its target reads the link's own birth."""
    target = tmp_path/'target'
    target.mkdir()
    time.sleep(.05)
    link = tmp_path/'link'
    link.symlink_to(target)
    sec, nsec = _stat_birth(link)
    assert M.birth(link) > M.birth(target) + .02 and abs(M.birth(link) - (sec + nsec*1e-9)) < 1e-6


def test_before_prints_the_number_not_started(sandbox, tmp_path, capsys):
    root = M.mode_root('lpba40_test')
    (root/'method'/pair('S03', 'S04')).mkdir(parents=True)
    epoch = time.time() - 5.
    assert M.main(['before', '--mode', 'lpba40_test', '--since', '900', '--start-epoch', repr(epoch), '--out', str(tmp_path/'cb.json')]) == 0
    out = capsys.readouterr().out
    assert 'problems 0  rows 0  not_started 1' in out
