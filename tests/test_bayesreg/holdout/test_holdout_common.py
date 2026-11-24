"""experiments/holdout/holdout_common.py: the pinned pair list, the admission (SG's real functions: bayesreg.io.unseal.load_unseal and bind_unseal, bayesreg.io.guard.check_subject, on a simulated
unseal branch in a temporary git repository; and on the code tree as it is, which has no record), the disks, the machine gate, the lock, the move to D:, the resume record.  Ids only and synthetic
files in tmp_path; no subject's file is opened (the admission tests watch it with the driver's spy)."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from ..xb import _world as W

hc, xc, xb_spy = W.hc, W.xc, W.xb_spy
sealed_module = W.sealed_module


def watching(tmp_path):
    """A spy that refuses any open outside the code tree, bayesreg's code tree (the simulated branch when pinned), conda and the system (the hold-out stage, the data, the results root are forbidden),
    with a synthetic file at a hold-out pair's stage path; the sealed lists counted at every read (C23.6-5)."""
    row = W.holdout_rows()[0]
    original = hc.stage_pair_path(hc.pair_name(*row))
    if not original.exists():
        W.write_stage_pair(hc.STAGE_ROOT, hc.pair_name(*row), seg=False)
    tree = [] if Path(sealed_module.REPO).resolve() == xc.REPO.resolve() else [Path(sealed_module.REPO)]
    lists = hc.sealed_list_paths()
    policy = xb_spy.driver_policy(work=tmp_path/'none', record_dir=tmp_path/'none', lock_dir=tmp_path/'none', code_root=xc.REPO, results=hc.RESULTS, main_repo=hc.MAIN_REPO, holdout_root=hc.HOLDOUT_ROOT,
                                  extra_exact=lists, extra_read=tree)
    return xb_spy.DriverSpy(policy, original=original, counted=lists)


def assert_nothing_read(spy):
    record = spy.record(expect_original=False)
    assert record['breaches'] == [] and record['original_opens_outside'] == [] and record['original_opens_inside_reader'] == 0, record
    return record


# ---- the pair list -------------------------------------------------------------------------------------------------------------------------------
def test_the_pair_list_is_the_pinned_one_in_its_order():
    rows = hc.holdout_pairs()
    assert len(rows) == 50 and rows[0] == ('OASIS_OAS1_0423_MR1', 'OASIS_OAS1_0099_MR1')
    sealed = hc.sealed_lists()
    assert {s for r in rows for s in r} == set(sealed.subjects['holdout_B_v1'])           # C23.12-2 (a): the 100 ids are exactly the hold-out list


def test_a_changed_pair_list_is_refused(tmp_path):
    repo = tmp_path/'repo'
    (repo/'splits/bayesreg_v1').mkdir(parents=True)
    text = (hc.REPO/hc.PAIRS_TSV_REL).read_text()
    (repo/hc.PAIRS_TSV_REL).write_text(text.replace('OASIS_OAS1_0423_MR1\t', 'OASIS_OAS1_0099_MR1\t', 1))
    with pytest.raises(hc.Refused) as e:
        hc.holdout_pairs(repo)
    assert e.value.stage == 'pairs'


# ---- the hold-out admission ---------------------------------------------------------------------------------------------------------------------
def test_without_a_record_every_holdout_pair_is_refused_at_the_unseal_and_nothing_is_read(tmp_path):
    fixed, moving = W.holdout_rows()[0]
    assert sealed_module.sealed_cache_is_empty()
    with watching(tmp_path) as spy:
        with pytest.raises(hc.Refused) as e:
            hc.admit_holdout_pair(fixed, moving)                 # the code tree as it is: no unseal record
    assert e.value.stage == 'unseal'
    record = assert_nothing_read(spy)
    assert record['n_sealed_list_reads'] >= 8                    # C23.6-5: the guard's first read happened under the spy, every read recorded (each list twice: sha256 and text)


def test_the_real_loader_admits_a_row_in_its_direction(tmp_path, monkeypatch):
    sim = W.simulated_unseal(tmp_path, monkeypatch)
    fixed, moving = W.holdout_rows()[3]
    with watching(tmp_path) as spy:
        record, bound = hc.admit_holdout_pair(fixed, moving)
    assert_nothing_read(spy)
    from bayesreg.io.unseal import bound_problem
    assert record['pairs_tsv']['row'] == 4 and record['pairs_tsv']['sha256'] == hc.PAIRS_TSV_SHA256
    assert record['unseal']['sealed_list'] == 'holdout_B_v1' and record['unseal']['written_by'] == "lead on BZ's behalf" and record['unseal']['n_subjects'] == 100
    assert record['check_subject']['fixed']['admitted_by_unseal'] and bound.unseal is not None and bound_problem(bound) is None and bound.sha256 == sim.allowlist.sha256
    assert Path(bound.unseal.path).resolve() == (sim.repo/hc.UNSEAL_REL).resolve()


def refusal(fixed, moving, tmp_path):
    with watching(tmp_path) as spy:
        with pytest.raises(hc.Refused) as e:
            hc.admit_holdout_pair(fixed, moving)
    assert_nothing_read(spy)
    return e.value


def test_refusals_read_nothing(tmp_path, monkeypatch):
    W.simulated_unseal(tmp_path, monkeypatch)
    rows = W.holdout_rows()
    sealed = hc.sealed_lists()
    (f0, m0), (f1, m1) = rows[0], rows[1]
    e = refusal(f0, m1, tmp_path)                                                                                    # two hold-out subjects, not a row
    assert e.stage == 'admission' and 'not a row' in str(e)
    e = refusal(m0, f0, tmp_path)                                                                                    # the row the other way round
    assert e.stage == 'admission' and 'other way round' in str(e)
    h1 = sorted(sealed.subjects['H1'])[0]
    h4 = sorted(sealed.subjects['H4'])[0]
    assert refusal(f0, h1, tmp_path).stage == 'never'                                                                # H1: never
    assert refusal(h4, m0, tmp_path).stage == 'never'
    assert refusal('LPBA40_S37', m0, tmp_path).stage == 'admission'                                                  # H2 (not even an OASIS id)
    assert refusal(f0, 'OASIS_OAS1_0438_MR1', tmp_path).stage == 'admission'                                         # a development subject is not a hold-out subject
    h3 = sorted(sealed.subjects['H3'])[0]
    assert refusal(f0, h3, tmp_path).stage in ('admission', 'never')                                                 # m12: H3 (IBSR18) too


@pytest.mark.parametrize('case', ['no record', 'uncommitted', 'not in the decision log', 'written_by'])
def test_a_record_the_loader_refuses_refuses_the_pair(tmp_path, monkeypatch, case):
    kw = {'no record': dict(record=False), 'uncommitted': dict(commit=False), 'not in the decision log': dict(log=False), 'written_by': dict(written_by='someone')}[case]
    W.simulated_unseal(tmp_path, monkeypatch, **kw)
    fixed, moving = W.holdout_rows()[0]
    assert refusal(fixed, moving, tmp_path).stage == 'unseal'


def test_a_bound_allowlist_made_otherwise_is_refused_by_the_guard(tmp_path, monkeypatch):
    """C23.3-5: only bind_unseal's object counts; a replace() of it carries the same unseal object and is refused by check_subject (and so by the admission's re-check in the driver)."""
    from dataclasses import replace
    from bayesreg.io.guard import check_subject
    W.simulated_unseal(tmp_path, monkeypatch)
    fixed, moving = W.holdout_rows()[0]
    record, bound = hc.admit_holdout_pair(fixed, moving)
    assert check_subject(fixed, bound).admitted
    assert not check_subject(fixed, replace(bound)).admitted


def test_the_development_admission():
    fixed, moving = W.DEV_PAIR.split('__')
    record, allowlist = hc.admit_dev_pair(fixed, moving)
    assert record['pair_set'] == 'dev' and record['guard']['fixed_subject'] == fixed
    with pytest.raises(hc.Refused):
        hc.admit_dev_pair('OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0440_MR1')        # not one of the 19
    f0, m0 = W.holdout_rows()[0]
    with pytest.raises(hc.Refused):
        hc.admit_dev_pair(f0, m0, dev_pairs=(hc.pair_name(f0, m0),))           # a hold-out subject is never a development one


# ---- disks, machine ------------------------------------------------------------------------------------------------------------------------------
def test_disk_guard_both_sides():
    usage = lambda free: (lambda path: type('U', (), dict(free=free[str(path)]*2**30))())      # noqa: E731
    ok = {'/mnt/d': 50., '/mnt/c': 30., '/home': 900.}
    assert hc.disk_guard(need_d_gib=5., need_home_gib=5., usage=usage(ok))['passed']
    with pytest.raises(hc.InfrastructureFailure):
        hc.disk_guard(need_d_gib=11., need_home_gib=1., usage=usage(ok))                     # D: 11 + 40 > 50
    with pytest.raises(hc.InfrastructureFailure):
        hc.disk_guard(need_d_gib=0., need_home_gib=11., usage=usage(ok))                     # C: 11 + 20 > 30
    with pytest.raises(hc.InfrastructureFailure):
        hc.disk_guard(need_d_gib=0., need_home_gib=1., usage=usage({**ok, '/home': 10.}))     # /home too


BOOT = 'b'*32
PROBE = json.dumps({'__REALTIME_TIMESTAMP': '1', '__MONOTONIC_TIMESTAMP': '5000000', '_BOOT_ID': BOOT})+'\n'      # journalctl -n 1: the newest entry of the user journal (any unit)


def fake_machine(units='', util='0', journal='', journal_rc=0, probe=PROBE):
    calls = []

    def run(argv, **kwargs):
        if argv[:2] == ['systemctl', '--user']:
            return subprocess.CompletedProcess(argv, 0, units, '')
        if argv[0] == 'journalctl':                                                     # A5a: the user journal (its liveness probe -n 1, then the window)
            calls.append(list(argv))
            text = probe if '-n' in argv else journal
            return subprocess.CompletedProcess(argv, journal_rc, text, '' if journal_rc == 0 else 'no journal (test)')
        if '--query-compute-apps=pid' in argv:
            return subprocess.CompletedProcess(argv, 0, '', '')
        return subprocess.CompletedProcess(argv, 0, util+'\n', '')
    run.calls = calls
    return run


def journal_line(unit, mono, *, field='USER_UNIT', boot=BOOT):
    """One entry as journalctl -o json --output-fields=... prints it (no MESSAGE: A5a review m6)."""
    return json.dumps({'__REALTIME_TIMESTAMP': str(int((1_800_000_000.+mono)*1e6)), '__MONOTONIC_TIMESTAMP': str(int(mono*1e6)), '_BOOT_ID': boot, field: f'{unit}.service',
                       **({'_SYSTEMD_USER_UNIT': 'init.scope'} if field == 'USER_UNIT' else {})})+'\n'


@pytest.fixture
def machine(tmp_path, monkeypatch):
    meminfo, loadavg = tmp_path/'meminfo', tmp_path/'loadavg'
    meminfo.write_text('MemAvailable:   11000000 kB\n')
    loadavg.write_text('0.20 0.1 0.1 1/1 1\n')
    monkeypatch.setattr(hc, '_local_end', None)                                        # no unit of this process yet
    return dict(meminfo=meminfo, loadavg=loadavg, now=1000., boot=BOOT)


def test_the_machine_gate_waits_ten_seconds_after_the_previous_unit(machine):
    """A5a (decision_log 2026-11-10 ~3:16 AM CST; runs/20261107-xb/limit_experiment_result.md revision 1 item 2: dmesg runs ~4 s ahead of /proc/uptime here; A5a review I1, m1, m6): the gate refuses
    while the previous tag- unit of this project wrote its last journal entry (its end) less than 10 s ago, on the monotonic clock; the parent units are not "previous"; seq-vp-* units are the other
    line's; an entry of another boot ended before this boot; only the fields the gate needs are asked; an unreadable journal, a line that is not a JSON object or an entry without its stamps refuses."""
    kw = machine
    recent = journal_line('tag-hb-x-ants_s-0438-0439', 996.)
    old = journal_line('tag-hb-xl-runtime-ants_s', 989.5)
    run = fake_machine(journal=old)
    record = hc.machine_gate(run=run, **kw)
    assert record['passed'] and record['previous_unit']['unit'] == 'tag-hb-xl-runtime-ants_s' and record['previous_unit']['age_seconds'] == 10.5 and record['unit_gap_seconds'] == 10.
    assert all(f'--output-fields={",".join(hc.JOURNAL_FIELDS)}' in c and 'MESSAGE' not in ' '.join(c) for c in run.calls) and [('-n' in c) for c in run.calls] == [True, False]
    with pytest.raises(hc.Refused, match='4.0 s ago'):
        hc.machine_gate(run=fake_machine(journal=old+recent), **kw)
    with pytest.raises(hc.Refused, match='last journal entry'):
        hc.machine_gate(run=fake_machine(journal=journal_line('tag-hb-x-ants_s-0438-0439', 999., field='_SYSTEMD_USER_UNIT')), **kw)
    assert hc.machine_gate(run=fake_machine(journal=journal_line('seq-vp-chain', 999.)), **kw)['previous_unit'] is None
    assert hc.machine_gate(run=fake_machine(journal=journal_line('tag-x', 999., boot='c'*32)), **kw)['previous_unit'] is None      # another boot
    assert hc.machine_gate(run=fake_machine(journal=recent), parent_units=('tag-hb-x-ants_s-0438-0439',), **kw)['passed']
    assert hc.machine_gate(run=fake_machine(journal=''), **kw)['previous_unit'] is None             # the journal readable, no unit in the window
    with pytest.raises(hc.Refused, match='journalctl exited'):
        hc.machine_gate(run=fake_machine(journal_rc=1), **kw)
    with pytest.raises(hc.Refused, match='not a JSON object'):
        hc.machine_gate(run=fake_machine(journal='-- No entries --\n'), **kw)
    with pytest.raises(hc.Refused, match='MONOTONIC'):
        hc.machine_gate(run=fake_machine(journal=json.dumps(dict(USER_UNIT='tag-x.service', _BOOT_ID=BOOT))+'\n'), **kw)

    def broken(argv, **kwargs):
        if argv[0] == 'journalctl':
            raise OSError('no journalctl (test)')
        return fake_machine()(argv, **kwargs)
    with pytest.raises(hc.Refused, match='could not be read'):
        hc.machine_gate(run=broken, **kw)


def test_an_empty_or_missing_journal_is_refused(machine, tmp_path, monkeypatch):
    """A5a review I1: journalctl exits 0 and prints nothing when it finds no journal file (an empty --directory; a user journal never kept): the liveness probe (-n 1 must give one entry) refuses it,
    never read as "no unit in the window".  Once with a stand-in, once with the real journalctl on an empty directory of the test (read only)."""
    with pytest.raises(hc.Refused, match='gave 0 entries'):
        hc.machine_gate(run=fake_machine(journal='', probe=''), **machine)
    empty = tmp_path/'empty_journal'
    empty.mkdir()
    monkeypatch.setattr(hc, 'JOURNAL_SOURCE', ('--directory', str(empty)))
    with pytest.raises(hc.Refused, match='gave 0 entries'):
        hc.journal_alive()                                                              # the real journalctl: exit 0, no output
    with pytest.raises(hc.Refused, match='gave 0 entries'):
        hc.last_unit_end(now=1000., boot=BOOT)
    # A5a diff review n2: a journal directory that does not exist -- the real journalctl exits non-zero ("Failed to open ..."): refused by the exit-code branch, before any entry is read
    missing = tmp_path/'no_such_journal_dir'
    monkeypatch.setattr(hc, 'JOURNAL_SOURCE', ('--directory', str(missing)))
    for probe in (hc.journal_alive, lambda: hc.last_unit_end(now=1000., boot=BOOT)):
        with pytest.raises(hc.Refused, match='journalctl exited [1-9]') as refused:
            probe()
        assert refused.value.stage == 'machine'
    assert not missing.exists()


def test_the_queues_own_record_holds_when_journald_is_slow(machine):
    """A5a review I1: the journal shows the previous unit's last line a minute ago (journald has not taken its end in), but this queue recorded the end of the unit it ran 3 s ago (note_unit_end):
    refused; 10.5 s ago: passes.  A record of another boot does not count.  xb_chain.run_one and hb_core.run_unit record it right after systemd-run --wait returns."""
    journal = journal_line('tag-hb-c-0438-0439', 940.)
    hc.note_unit_end('tag-hb-c-0438-0439', monotonic=997.)
    hc._local_end['boot_id'] = BOOT
    with pytest.raises(hc.Refused, match="this queue's previous unit tag-hb-c-0438-0439 ended 3.0 s ago"):
        hc.machine_gate(run=fake_machine(journal=journal), **machine)
    hc.note_unit_end('tag-hb-c-0438-0439', monotonic=989.5)
    hc._local_end['boot_id'] = BOOT
    record = hc.machine_gate(run=fake_machine(journal=journal), **machine)
    assert record['passed'] and record['local_unit_end']['unit'] == 'tag-hb-c-0438-0439'
    hc.note_unit_end('tag-x', monotonic=999.)
    hc._local_end['boot_id'] = 'c'*32
    assert hc.machine_gate(run=fake_machine(journal=journal), **machine)['passed']


def test_this_processs_own_unit_is_not_the_previous_one(machine, tmp_path):
    """A5a review m2: a queue running in its own unit (here tag-hb-x-chain, named by a stand-in /proc/self/cgroup) writes journal lines of that unit; they are not "the previous unit".  The same lines
    count when this process is not in that unit (the own unit is only what /proc/self/cgroup shows: hi_core.units_of_this_process)."""
    journal = journal_line('tag-hb-x-chain', 999., field='_SYSTEMD_USER_UNIT')
    inside = tmp_path/'cgroup_inside'
    inside.write_text('0::/user.slice/user-1000.slice/user@1000.service/app.slice/tag-hb-x-chain.service\n')
    outside = tmp_path/'cgroup_outside'
    outside.write_text('0::/user.slice/user-1000.slice/session-1.scope\n')
    chain_active = 'tag-hb-x-chain.service loaded active running chain\n'
    record = hc.machine_gate(run=fake_machine(units=chain_active, journal=journal), own_units=('tag-hb-x-chain',), cgroup=inside, **machine)
    assert record['passed'] and record['own_units'] == ['tag-hb-x-chain'] and record['previous_unit'] is None
    with pytest.raises(hc.Refused):
        hc.machine_gate(run=fake_machine(units='', journal=journal), own_units=('tag-hb-x-chain',), cgroup=outside, **machine)


def test_machine_gate(machine):
    kw = machine
    assert hc.machine_gate(run=fake_machine(), **kw)['passed']
    chain = 'tag-hb-x-chain.service loaded active running chain\n'
    assert hc.machine_gate(run=fake_machine(chain), parent_units=('tag-hb-x-chain',), **kw)['passed']
    with pytest.raises(hc.Refused):
        hc.machine_gate(run=fake_machine(chain), **kw)                                       # the chain is another unit unless named as the parent
    with pytest.raises(hc.Refused):
        hc.machine_gate(run=fake_machine('seq-vp-x.service loaded active running y\n'), parent_units=('tag-hb-x-chain',), **kw)
    with pytest.raises(hc.Refused):
        hc.machine_gate(run=fake_machine(util='55'), **kw)
    kw['meminfo'].write_text('MemAvailable:   4000000 kB\n')
    with pytest.raises(hc.Refused):
        hc.machine_gate(run=fake_machine(), **kw)
    kw['meminfo'].write_text('MemAvailable:   11000000 kB\n')
    kw['loadavg'].write_text('3.0 1 1 1/1 1\n')
    with pytest.raises(hc.Refused):
        hc.machine_gate(run=fake_machine(), **kw)


def test_wait_for_times_out_as_infrastructure():
    clock = [0.]

    def check():
        raise hc.Refused('machine', 'busy')
    with pytest.raises(hc.InfrastructureFailure) as e:
        hc.wait_for(check, seconds=60, poll=30, now=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0]+s), log=lambda m: None)
    assert e.value.kind == 'precheck_timeout'


# ---- lock, move ----------------------------------------------------------------------------------------------------------------------------------
def test_the_lock_is_taken_once(tmp_path):
    lock = tmp_path/'locks'/'a.lock'
    hc.check_once(lock)
    hc.take_lock(lock, hc.lock_line(arm='x'))
    with pytest.raises(hc.Refused):
        hc.check_once(lock)
    with pytest.raises(hc.Refused):
        hc.take_lock(lock, hc.lock_line(arm='x'))
    line = json.loads(lock.read_text().splitlines()[0])
    assert line['arm'] == 'x' and line['pid'] == os.getpid() and 'boot_id' in line


def test_move_to_d(tmp_path):
    src, dst = tmp_path/'work', tmp_path/'d'/'cell'
    src.mkdir()
    (src/'disp.npz').write_bytes(b'0123456789')
    (src/'own_warped.npz').write_bytes(b'abc')
    record = hc.move_files(src, dst, tmp_path/'moves.json', names=['disp.npz', 'own_warped.npz'])
    assert record['passed'] and not (src/'disp.npz').exists() and (dst/'disp.npz').read_bytes() == b'0123456789'
    assert json.loads((tmp_path/'moves.json').read_text())['files'][0]['sha256_before'] == record['files'][0]['sha256_after']


def test_a_move_whose_sha256_differs_keeps_both_and_stops(tmp_path, monkeypatch):
    src, dst = tmp_path/'work', tmp_path/'d'
    src.mkdir()
    (src/'disp.npz').write_bytes(b'0123456789')
    real = shutil.copyfile

    def corrupt(a, b):
        real(a, b)
        Path(b).write_bytes(b'0123456780')
    monkeypatch.setattr(hc.shutil, 'copyfile', corrupt)
    with pytest.raises(hc.InfrastructureFailure) as e:
        hc.move_files(src, dst, tmp_path/'moves.json')
    assert e.value.kind == 'move_mismatch' and (src/'disp.npz').exists() and (dst/'disp.npz').exists()
    assert json.loads((tmp_path/'moves.json').read_text())['passed'] is False


def test_code_identity_looks_for_untracked_files_where_the_driver_imports_from():
    """m9: the untracked files under experiments/ (hi_core, he/unit_probe and their imports, not only xb and holdout), bayesreg/, jd_curl/ and configs/ make the tree not clean."""
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        if 'rev-parse' in argv:
            return subprocess.CompletedProcess(argv, 0, 'c'*40+'\n', '')
        if 'ls-files' in argv:
            return subprocess.CompletedProcess(argv, 0, 'experiments/he/new_probe.py\n', '')
        return subprocess.CompletedProcess(argv, 0, '', '')
    record = hc.code_identity(repo=W.ROOT, run=run)
    listing = next(c for c in calls if 'ls-files' in c)
    assert listing[listing.index('--')+1:] == ['experiments', 'bayesreg', 'jd_curl', 'configs'] and '--literal-pathspecs' in listing
    assert record['clean'] is False and record['untracked'] == ['experiments/he/new_probe.py']


def test_a_move_that_stops_half_way_writes_its_partial_record(tmp_path, monkeypatch):
    """m4: an OSError in the middle (D: full) writes the move record with the files moved so far and the error, keeps the file at hand, and is an infrastructure failure, not a wrapper error."""
    src, dst = tmp_path/'work', tmp_path/'d'
    src.mkdir()
    (src/'disp.npz').write_bytes(b'0123456789')
    (src/'own_warped.npz').write_bytes(b'abc')
    real = shutil.copyfile

    def full(a, b):
        if Path(a).name == 'own_warped.npz':
            raise OSError(28, 'No space left on device')
        return real(a, b)
    monkeypatch.setattr(hc.shutil, 'copyfile', full)
    with pytest.raises(hc.InfrastructureFailure) as e:
        hc.move_files(src, dst, tmp_path/'moves.json', names=['disp.npz', 'own_warped.npz'])
    assert e.value.kind == 'move_error' and xc.classify(e.value) == ('infra_failure', 'move_error')
    record = json.loads((tmp_path/'moves.json').read_text())
    assert record['passed'] is False and 'No space left' in record['error']
    assert [(r['file'], r['ok']) for r in record['files']] == [('disp.npz', True), ('own_warped.npz', False)] and record['files'][1]['problem'] == 'copying'
    assert not (src/'disp.npz').exists() and (dst/'disp.npz').exists() and (src/'own_warped.npz').exists()


# ---- the committed authorizations (resume, rerun, no_resume; I1-I3) ------------------------------------------------------------------------------
def resume_world(tmp_path, *, previous=dict(status='infra_failure'), commit=True, pair_set='holdout', name=None, **override):
    """A first run's lock and record of cell (a, p), and a resume authorization for it in the sandbox's authorization repository (hc.AUTH_ROOT)."""
    lock = tmp_path/'locks'/'xb__a__p.lock'
    hc.take_lock(lock, hc.lock_line(arm='a'))
    record = tmp_path/'rec'/'result.json'
    record.parent.mkdir()
    if previous is not None:
        record.write_text(json.dumps(previous))
    doc = dict(schema='bayesreg.xb.resume.v1', written_by="lead on BZ's behalf", words='（测试）补跑', time_local='2026-11-07 6 PM CST', utc='2026-11-07T23:00:00Z', reason='test', tool='xb', arm='a', pair='p',
               pair_set=pair_set, lock_sha256=hc.sha256_file(lock), previous_record=str(record), previous_record_sha256=hc.sha256_file(record) if previous is not None else None)
    doc.update(override)
    path = W.commit_authorization(pair_set, 'resume', doc, commit=commit) if name is None else W.write_json(hc.authorization_dir(pair_set)/name, doc)
    if name is not None and commit:
        W.git(W.auth_repo(), 'add', '-A')
        W.git(W.auth_repo(), 'commit', '-q', '-m', 'named')
    kw = dict(tool='xb', arm='a', pair='p', pair_set=pair_set, lock=lock, previous_record=record,
              complete=lambda: record.is_file() and xc.complete(json.loads(record.read_text())))
    return path, kw


def test_a_resume_record_that_holds(tmp_path):
    path, kw = resume_world(tmp_path)
    assert path.parent == hc.AUTH_ROOT/'splits'/'bayesreg_v1' and path.name == 'holdout_B_v1_xb_resume_1.json'
    resume = hc.load_resume(path, **kw)
    assert resume.lock_path.endswith('.lock.resume') and resume.previous_record == str(kw['previous_record']) and resume.document['git']['tracked']
    hc.check_once(kw['lock'], resume=resume)
    dev, kw_dev = resume_world(tmp_path/'dev', pair_set='dev')
    assert dev.parent == hc.AUTH_ROOT/'runs'/'20261107-xb'/'authorizations' and dev.name == 'dev_xb_resume_1.json'          # m15
    assert hc.load_resume(dev, **kw_dev).path == str(dev)


@pytest.mark.parametrize('case', ['uncommitted', 'changed after the commit', 'complete', 'writer', 'no reason', 'lock', 'resumed', 'cell', 'pair set', 'dev name in the hold-out directory'])
def test_resume_refusals(tmp_path, case):
    if case == 'uncommitted':
        path, kw = resume_world(tmp_path, commit=False)
    elif case == 'changed after the commit':
        path, kw = resume_world(tmp_path)
        path.write_text(path.read_text().replace('"test"', '"edited"'))
    elif case == 'complete':
        path, kw = resume_world(tmp_path, previous=dict(status='method_failure'))                 # a method failure is final: not rerun
    elif case == 'writer':
        path, kw = resume_world(tmp_path, written_by='someone')
    elif case == 'no reason':
        path, kw = resume_world(tmp_path, reason=' ')
    elif case == 'lock':
        path, kw = resume_world(tmp_path, lock_sha256='0'*64)
    elif case == 'resumed':
        path, kw = resume_world(tmp_path)
        hc.take_lock(hc.resume_lock_path(kw['lock']), hc.lock_line(arm='a'))                      # at most once
    elif case == 'cell':
        path, kw = resume_world(tmp_path, arm='b')
    elif case == 'pair set':
        path, kw = resume_world(tmp_path, pair_set='holdout')
        kw['pair_set'] = 'dev'
    else:
        path, kw = resume_world(tmp_path, name='dev_xb_resume_1.json')
    with pytest.raises(hc.Refused) as e:
        hc.load_resume(path, **kw)
    assert e.value.stage == 'resume'


def test_the_dry_run_summary_takes_only_the_p0_refusals_as_expected(tmp_path, monkeypatch):
    """A5a review m8: every hold-out entry's dry-run record carries passed, problems, expected, unexpected and p0_ok the same way.  Expected on P0: rule one failing only for want of the pinned
    record (bayesreg.io.code_identity.pinned_record's words), no unseal record, a development check's decision still None; anything else is unexpected."""
    unpinned = 'runs/holdout_B_v1/pinned_code.json is not committed at HEAD (fatal: no such path): no pinned commit (it is never HEAD by default)'
    assert hc.rule_one_kind([unpinned]) == 'rule_one_unpinned' and hc.rule_one_kind([unpinned, 'the code differs']) == 'rule_one' and hc.rule_one_kind([]) == 'rule_one'
    empty = hc.dry_run_summary([])
    assert empty['passed'] is True and empty['problems'] == [] and empty['p0_ok'] is True and empty['expected'] == empty['unexpected'] == []
    p0 = hc.dry_run_summary([(hc.rule_one_kind([unpinned]), f'rule one: {[unpinned]}'), ('no_unseal_record', 'refused (unseal): none'), ('decision_pending', 'decision: None')])
    assert p0['passed'] is False and p0['p0_ok'] is True and [e['kind'] for e in p0['expected']] == ['rule_one_unpinned', 'no_unseal_record', 'decision_pending'] and len(p0['problems']) == 3
    other = hc.dry_run_summary([('rule_one', 'rule one: the code differs'), ('problem', 'preregistration: None'), ('refused_admission', 'refused (admission): x')])
    assert other['p0_ok'] is False and [e['problem'] for e in other['unexpected']] == ['rule one: the code differs', 'preregistration: None', 'refused (admission): x']
    from bayesreg.io import sealed as sealed_module
    monkeypatch.setattr(sealed_module, 'REPO', tmp_path)
    assert hc.unseal_record_exists() is False
    (tmp_path/hc.UNSEAL_REL).parent.mkdir(parents=True)
    (tmp_path/hc.UNSEAL_REL).write_text('{}')
    assert hc.unseal_record_exists() is True


def test_the_rerun_and_no_resume_authorizations(tmp_path):
    """I1: rerun_authorization: None without refusal records; Refused without an authorization; a committed one must list exactly the counted records (stage rerun left out); I3:
    no_resume_authorization holds only for the record and sha256 it names, committed and unchanged, of a record that is not complete."""
    root = tmp_path/'records'
    assert hc.rerun_authorization(root, tool='xb', arm='a', pair='p', pair_set='dev') is None
    first = hc.write_refusal(hc.batch_root(root)/'xb_1', dict(arm='a', pair='p', pair_set='dev', stage='admission'))
    hc.write_refusal(hc.batch_root(root)/'xb_1', dict(arm='a', pair='q', pair_set='dev', stage='admission'))           # another cell
    hc.write_refusal(hc.batch_root(root)/'xb_2', dict(arm='a', pair='p', pair_set='dev', stage='rerun'))              # refused for lacking the authorization: not counted
    with pytest.raises(hc.Refused) as e:
        hc.rerun_authorization(root, tool='xb', arm='a', pair='p', pair_set='dev')
    assert e.value.stage == 'rerun'
    common = dict(schema='bayesreg.xb.rerun.v1', written_by='BZ', words='可以重跑', time_local='2026-11-09 1 PM CST', utc='2026-11-09T19:00:00Z', reason='test', tool='xb', arm='a', pair='p', pair_set='dev')
    W.commit_authorization('dev', 'rerun', dict(common, refusals=[]))
    with pytest.raises(hc.Refused):
        hc.rerun_authorization(root, tool='xb', arm='a', pair='p', pair_set='dev')
    path = W.commit_authorization('dev', 'rerun', dict(common, refusals=[dict(path=str(first), sha256=hc.sha256_file(first))]))
    assert hc.rerun_authorization(root, tool='xb', arm='a', pair='p', pair_set='dev')['path'] == str(path)
    record = tmp_path/'rec'/'result.json'
    record.parent.mkdir()
    record.write_text(json.dumps(dict(status='infra_failure')))
    doc = dict(common, schema='bayesreg.xb.no_resume.v1', record=str(record), record_sha256=hc.sha256_file(record), note='未批补跑')
    doc.pop('refusals', None)
    W.commit_authorization('dev', 'no_resume', doc)
    kw = dict(tool='xb', arm='a', pair='p', pair_set='dev', record=record)
    assert hc.no_resume_authorization(**kw, complete=lambda: False) is not None
    assert hc.no_resume_authorization(**kw, complete=lambda: True) is None
    assert hc.no_resume_authorization(**dict(kw, record=tmp_path/'other.json'), complete=lambda: False) is None
    record.write_text(json.dumps(dict(status='infra_failure', changed=True)))
    assert hc.no_resume_authorization(**kw, complete=lambda: False) is None                 # the record is not the one it names any more
