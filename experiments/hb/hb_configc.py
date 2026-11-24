"""HB: the configuration C arm (preregistration §2.2; runs/20261109-hb/任务书.md item 5): the frozen solver's configuration C on the hold-out pairs, as an arm of its own, through HE's fallback-one code
(experiments/he/firewall_c.py; unchanged by HB -- LP-H (decision_log 2026-11-21) changed only how prepare removes its private copy, firewall_c.remove_tree): the derived pair (fixed, moving,
initial_u byte for byte, the label members zero), the frozen prepare from a private copy, check_prepared, the firewall gate,
the frozen worker in this unit (firewall_c.DirectLauncher), experiments/d0/classify_run.py (unchanged), D75's candidates certified again in this process (it reads no label).

    python -B experiments/hb/hb_configc.py dry-run [--pair-set holdout|dev]
    python -B experiments/hb/hb_configc.py queue --pair-set holdout|dev [--commit SHA] [--pairs P,...] [--resume PAIR=REL ...]
    python -B experiments/hb/hb_configc.py launch-string --pair-set ...
    python -B experiments/hb/hb_configc.py run --pair P --pair-set holdout|dev --batch DIR [--resume REL]     the unit's process (tag-hb-c-<ffff>-<mmmm>)

The driver, in this order (nothing of a subject is opened before the lock but the ids, the manifest's whitelisted fields and the container's sha256 through load_pair's guard):
  the admission (holdout_common.admit_holdout_pair: load_unseal, bind_unseal in this process, H1-H4 never, check_subject, the pair list in its direction; a development pair: admit_dev_pair); the frozen
  tree's identity (he_core.frozen_identity); the cell's lock (locks/hb__configc__<pair>.lock, O_EXCL); then load_pair with the bound allowlist, derive_pair, prepare, firewall_check; the frozen run in
  this unit (the unit's memory.events before and after, the kernel's lines in the window); classify_run; then:
  the step's evidence (holdout_common.classify_unit_end, rule (乙), need dmesg and the probe; prereg revision 1c item 1): OOM evidence -> infra_failure; unreadable -> needs_lead_rejudgment; a dxg line
  -> infra_failure; classify_run's class: ok, handover_only -> D75's candidates (final_native.npz, else the certified native seed), each certified again (configuration C's certificate, CPU FP64, this
  process) and the first that passes is the output (output/: final.npz, disp.npz, cert.json, line.json of he_core.write_output) -> ok; none passes -> method_failure (no certified state, §4.5 (1));
  method_failure: timeout (the 300 s watchdog) -> unit_limit (class (3), revision 1c item 1), the others -> method_failure; infrastructure -> infra_failure; unsure -> needs_lead_rejudgment.  An
  exception of this code after the lock -> wrapper_error (the batch stops; prereg revision 1a item 2: a rerun after the fix, by a committed resume record).
  Class ok: the intermediate commits of the task directory (committed_*.npz, level_*.npz; D0's cleanup rule, run_queue.sh) listed with their sha256 (intermediates.tsv) and deleted; handover_only and
  every failure: kept whole (prereg §5.3).  A hold-out cell then moves whole to /mnt/d/seq_vp_tag_holdout_B_v1/configc/<pair>/ (holdout_common.move_files, sha256 before and after).
  result.json (holdout_B_v1/configc/<pair>/): the terminal record (status, the classification, the candidates and the certificate, the output's files on D:, the firewall, the move).
The queue: the start checks (hb_method.start_checks for the hold-out), the pair list, per pair the gate (15 minutes), the disks, the unit (hb_core.CONFIGC_UNIT: 12G, swap 4G, OOMPolicy continue,
CPUQuota 1200 %, NUMBA 8, RuntimeMaxSec 1800; the GPU), its reading: a record the unit's evidence contradicts (OOM evidence of the unit) keeps its file and gets overridden.json (xb_chain's I2);
no record after the lock -> written here from the reading.  Continues: ok, method_failure, unit_limit; stops: infra_failure, needs_lead_rejudgment, wrapper_error, a refusal.
A stopped cell runs again once, by a committed resume record (--resume PAIR=REL; prereg §5.4): its records result_resume.json (overridden_resume.json), its own lock, work and D: directories with the
suffix .resume; the resume and the first run are one chain (chain()).  A resumed run that fails again (infra_failure, wrapper_error) is written: no output, no third run (runs/20261109-align item 3);
the batch stops at that moment and goes on at the next start.  BZ's committed 未批补跑 record (no_resume, its record the first run's result.json or overridden.json) also makes the cell written.

LP-B (contract C26.5-5, C26.4 L32, C26.10 T19; LPBA40 preregistration 06121278... §4.2, §8.2 step 8; brief item 2): --pair-set lpba40_test | lpba40_feas runs the arm on an LPBA40 mode.  The
admission: he_holdout.lpba40_pre_manifest (the mode's table in its direction, the pins, the pinned LPBA allowlist, check_subject: C26.6-2 steps 1-3), the manifest's subjects, then load_pair with that
allowlist (L32).  The code: lpba_arms.code_record (formal: rule one, LP-C2's, C26.9-4, fail closed; feasibility: a clean checkout); the pins (hb_method.lpba_pins_problem: the preregistration, the
frozen files, configC_reference.json d3313868..., the c record).  The derived pair, the frozen worker, D75 and the certificate as above (the derived manifest has no label_sets, the label members
zero; lam = .1/L² of the LPBA40 grid, firewall_c.frozen_weights).  Places: <root>/configc/<pair>/ (records), <root>/work/configc/<pair>/ (the run and output/), nothing moves (prereg §11): the record
result['output_store'] is written all the same, of the same shape as the hold-out's after its move (hb_configc.py's move branch), its files the work cell's output/ (<pair>.resume/output/ for a resumed
run) -- the LPBA40 arm table's field key output_store.disp.file (decision_log line 532 (1)).  Units tag-lp-c-<Sff>-<Smm> (tag-lp-feas-c-...), the queue tag-lp-c-queue; the hold file lpba40_test_v1/hold.
A rerun, resume or no_resume record of an LPBA mode is read only when holdout_common knows its authorizations (C26.8-3, D148: LP-C2's; the feasibility mode has none): otherwise refused.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hb_core as hb  # noqa: E402

import numpy as np  # noqa: E402

import firewall_c as fc  # noqa: E402
import hb_method  # noqa: E402

hc, he = hb.hc, hb.he
ARM = hb.CONFIGC_ARM
DEV_ROOT_NAME = 'hb_dev_v1'                     # development runs under holdout_common.RESULTS (read at call time): records, work, store all on /home; never D:
CONTINUES = ('ok', 'method_failure', 'unit_limit')
STATUSES = CONTINUES+('infra_failure', 'needs_lead_rejudgment', 'wrapper_error')
EXIT = dict(ok=0, method_failure=1, unit_limit=1, infra_failure=20, needs_lead_rejudgment=25, wrapper_error=23, refused=22)
INTERMEDIATE_PATTERNS = ('committed_*.npz', 'level_*.npz')      # experiments/d0/run_queue.sh: the intermediate commit maps of the task directory


class WrapperError(Exception):
    pass


PAIR_SETS = ('holdout', 'dev')+hb.hold.LPBA_PAIR_SETS


def roots(pair_set):
    if pair_set == 'holdout':
        return Path(hc.HOLDOUT_ROOT), Path(hc.BIG_ROOT)
    if pair_set == 'dev':
        return Path(hc.guard_root(Path(hc.RESULTS)/DEV_ROOT_NAME)), None
    if hb.is_lpba(pair_set):                            # LP-B: the mode's root; nothing moves (prereg §11)
        return Path(hb.lpba().root(pair_set)), None
    raise hc.Refused('admission', f'pair set {pair_set!r}: one of {PAIR_SETS}')


def hold_file(pair_set):
    """The hold file the queue waits on: the hold-out's and the development's <root>/hold as before; an LPBA mode's lpba40_test_v1/hold (C26.2-3)."""
    return hb.hold_path(pair_set) if hb.is_lpba(pair_set) else roots(pair_set)[0]/'hold'


def authorizations_problem(pair_set):
    """LP-B: None for the hold-out and the development; for an LPBA mode, why its committed authorizations cannot be read (hb_core.lpba_resume_problem), or None."""
    return hb.lpba_resume_problem(pair_set) if hb.is_lpba(pair_set) else None


def spy_breach_problem(pair_set, pair):
    """LP-B diff review minor 3: an LPBA cell with a spy breach record (its record directory, its work cells) is not rerun or resumed by an authorization (lpba_arms.spy_breach_problem); None
    otherwise and for the hold-out and the development."""
    if not hb.is_lpba(pair_set):
        return None
    paths = cell_paths(pair_set, pair)
    return hb.lpba().spy_breach_problem(paths['record'], paths['work'], paths['work'].with_name(paths['work'].name+'.resume'))


def rerun_authorization(pair_set, pair):
    """holdout_common.rerun_authorization of the cell; an LPBA mode without known authorizations: None when the cell has no refusal record, else Refused (stage rerun).  An LPBA cell with a spy
    breach record: Refused (stage rerun) whatever the authorizations (diff review minor 3)."""
    breach = spy_breach_problem(pair_set, pair)
    if breach:
        raise hc.Refused(hc.RERUN_STAGE, f'{ARM} {pair} ({pair_set}): {breach}')
    problem = authorizations_problem(pair_set)
    if problem is not None:
        prior = hc.cell_refusals(roots(pair_set)[0], arm=ARM, pair=pair, pair_set=pair_set)
        if prior:
            raise hc.Refused(hc.RERUN_STAGE, f'{ARM} {pair} ({pair_set}) was refused before its lock {len(prior)} time(s): {problem}')
        return None
    return hc.rerun_authorization(roots(pair_set)[0], tool=hb.TOOL, arm=ARM, pair=pair, pair_set=pair_set)


def cell_paths(pair_set, pair):
    record_root, big = roots(pair_set)
    return dict(record=Path(hc.record_dir(ARM, pair, record_root)), work=Path(hc.work_dir(ARM, pair, record_root)), store=None if big is None else Path(hc.big_dir(ARM, pair, big)),
                lock=Path(record_root)/'locks'/f'hb__{ARM}__{pair}.lock', result=Path(hc.record_dir(ARM, pair, record_root))/'result.json',
                overridden=Path(hc.record_dir(ARM, pair, record_root))/'overridden.json', batches=Path(record_root)/'batches')


RESUMED_WRITTEN = ('infra_failure', 'wrapper_error')     # prereg §5.4 and runs/20261109-align item 3: a resumed run that fails again (infrastructure, our wrapper) is written: no output, no third run


CONFIGC_D_NEED_GIB = 5.5                         # HB review m12 (ii): a failed or handover_only run is kept whole and moved to D: (about 5.5 GB, hb_core.UNIT_NEED_GIB's note); the worst case


def load_resume(pair_set, pair, rel):
    """holdout_common.load_resume for a configuration C cell (the driver and, before the unit, the queue: HB review I4): tool hb, the cell's lock, the first run's effective record, not complete
    (a continuing class, or a resumed chain already)."""
    paths = cell_paths(pair_set, pair)
    if spy_breach_problem(pair_set, pair):              # LP-B diff review minor 3: a spy breach goes to BZ, never through a resume
        raise hc.Refused('resume', f'{ARM} {pair} ({pair_set}): {spy_breach_problem(pair_set, pair)}')
    if authorizations_problem(pair_set) is not None:
        raise hc.Refused('resume', authorizations_problem(pair_set))
    return hc.load_resume(rel, tool=hb.TOOL, arm=ARM, pair=pair, pair_set=pair_set, lock=paths['lock'], previous_record=first_record(paths),
                          complete=lambda: chain(paths)[1] or (effective(paths) or {}).get('status') in CONTINUES)


def first_record(paths):
    """The first run's record path: overridden.json (the unit's evidence overrode the driver's record) when there, else result.json (may not exist)."""
    return paths['overridden'] if paths['overridden'].is_file() else paths['result']


def resumed_paths(paths):
    return dict(result=paths['record']/'result_resume.json', overridden=paths['record']/'overridden_resume.json')


def effective_path(paths):
    """The path of the cell's effective record (chain's): the resumed run's overridden or result record, else the first run's."""
    r = resumed_paths(paths)
    for path in (r['overridden'], r['result']):
        if path.is_file():
            return path
    return first_record(paths)


def chain(paths):
    """(the cell's effective record or None, resumed): the resumed run's record (its overridden record first) when the cell was resumed, else the first run's (prereg §5.4: the resume and the first
    run are one chain)."""
    r = resumed_paths(paths)
    for path in (r['overridden'], r['result']):
        doc = hb.read_json(path)
        if doc is not None:
            return doc, True
    return hb.read_json(first_record(paths)), False


def effective(paths):
    return chain(paths)[0]


def admit(pair_set, pair):
    fixed, moving = hc.split_pair(pair)
    hc.safe_name(pair, 'pair')
    if pair_set == 'holdout':
        from bayesreg.io.pair import _read_manifest
        record, allowlist = hc.admit_holdout_pair(fixed, moving)
        path = Path(hc.stage_pair_path(pair))
        _, _, selected = _read_manifest(path.parent/'manifest.json')           # HB review m12 (i): as he_holdout.admitted_images, the stage's manifest names this pair
        if (selected['fixed_subject'], selected['moving_subject']) != (fixed, moving):
            raise hc.Refused('manifest', f"the stage's manifest names {selected['fixed_subject']} <- {selected['moving_subject']}, not the pair {pair}")
        return record, allowlist, path
    if hb.is_lpba(pair_set):                            # LP-B (C26.5-5, L32): C26.6-2 steps 1-3, the manifest's subjects; load_pair (the run) with the pinned LPBA allowlist
        from bayesreg.io.pair import _read_manifest
        stage = hb.hold.lpba_stage_dir(pair_set)
        try:
            record, allowlist = hb.hold.lpba40_pre_manifest(pair_set, pair, stage)
        except hb.hold.HoldoutAdmissionRefused as exc:
            raise hc.Refused(getattr(exc, 'stage', 'admission'), str(exc)) from None
        path = Path(stage)/pair/'pair.npz'
        _, _, selected = _read_manifest(path.parent/'manifest.json')
        if (selected['fixed_subject'], selected['moving_subject']) != (record['fixed_subject'], record['moving_subject']):
            raise hc.Refused('manifest', f"the stage's manifest names {selected['fixed_subject']} <- {selected['moving_subject']}, not the pair {pair}")
        return record, allowlist, path
    record, allowlist = hc.admit_dev_pair(fixed, moving)
    return record, allowlist, Path(hc.DEV_STAGE)/pair/'pair.npz'


def certify_config():
    """Configuration C's certificate of the method (C22.4-1), from HE's frozen settings: the same configuration certifies this arm's output."""
    return he.certificate_config(he.frozen_settings())


def step_probe():
    import unit_probe
    return he.safe_probe('step', unit_probe.probe)


def kernel_window(t0):
    done = hb.System().dmesg()
    return done, hc.parse_oom(done, t0), hc.dxg_lines(done, t0)


def intermediates(run_dir, manifest):
    """Class ok: D0's intermediate commit maps listed (path, size, sha256: the cleanup manifest's format) and deleted; returns the record."""
    task = fc.task_dir(run_dir)
    rows = []
    for pattern in INTERMEDIATE_PATTERNS:
        for p in sorted(task.glob(pattern)):
            rows.append((str(p), p.stat().st_size, hb.sha(p)))
    with Path(hc.guard_root(manifest)).open('x', encoding='utf-8') as stream:          # written once
        stream.write('path\tsize_bytes\tsha256\n'+''.join(f'{p}\t{s}\t{h}\n' for p, s, h in rows))
    for p, _, _ in rows:
        Path(p).unlink()
    return dict(manifest=str(manifest), manifest_sha256=hb.sha(manifest), n_deleted=len(rows), bytes_deleted=sum(s for _, s, _ in rows),
                rule='class ok: the intermediate commits (committed_*.npz, level_*.npz of the task directory) recorded and deleted (D0 run_queue.sh); failures and handover_only kept whole (prereg §5.3)')


def lpba_policy_of(pair_set, pair, paths, batch):
    """LP-B review I3: the allow list of the configuration C arm's driver on an LPBA pair (lpba_arms.ArmPolicy; experiments/lpba/实现说明_LP-B.md): its work cell and the .resume one, its records,
    this run's batch directory (its refusal record; diff review minor 3: no other batch is written), the cache beside the cells (read and written); its lock and the resume's lock; the stage pair's
    pair.npz and manifest.json (read); the batches root read only (the earlier refusal records: their sequence numbers, the rerun check); the code trees (this checkout, bayesreg's, the frozen
    tree: its private copy is made from its jd_curl/ and scripts/; its results/, docs/, figures/ forbidden, diff review I3-a); everything else under the results root and the data roots refused,
    the libraries not watched."""
    la = hb.lpba()
    from bayesreg.io import sealed
    stage = Path(hb.hold.lpba_stage_dir(pair_set))/hc.safe_name(pair, 'pair')
    work = paths['work']
    return la.ArmPolicy(name=f'{pair_set} configc {pair}', read_exact=[stage/'pair.npz', stage/'manifest.json'], read_inside=[paths['batches']],
                        write_inside=[work, work.with_name(work.name+'.resume'), paths['record'], Path(batch), work.parent/'cache'],
                        write_exact=[paths['lock'], hc.resume_lock_path(paths['lock'])], results=la.results_roots()+[Path(he.RESULTS)], code_roots=[hb.REPO, Path(sealed.REPO)],
                        frozen_roots=[Path(he.core.FROZEN)])


def lpba_spied_run(pair_set, pair, *, batch, **kwargs):
    """LP-B review I3 (C26.8-5): run() of an LPBA pair under the allow-list spy (lpba_arms.arm_spying).  A breach before the lock is a refusal record (stage spy); after it, run() records
    infra_failure, kind spy_breach.  The spy's record <record>/opened_files[_k].json; a breach (also one a library swallowed) <record>/spy_breach[_k].json, which stops the queue."""
    la = hb.lpba()
    paths = cell_paths(pair_set, pair)
    try:
        policy = lpba_policy_of(pair_set, pair, paths, batch)
        with la.arm_spying(policy) as spy:
            try:
                rc = run(pair_set, pair, batch=batch, _spied=True, _spy=spy, **kwargs)
            except la.lc.hp.Breach as exc:
                hc.write_refusal(batch, dict(tool=hb.TOOL, arm=ARM, pair=pair, pair_set=pair_set, stage='spy', reason=str(exc), seq=next_refusal_seq(pair_set, pair)))
                rc = EXIT['refused']
            summary = spy.summary()
    except hc.Refused as exc:
        print(f'refused: {exc}', file=sys.stderr)
        return EXIT['refused']
    paths['record'].mkdir(parents=True, exist_ok=True)
    hb.write_once(hb.numbered(paths['record'], 'opened_files'), summary)
    if summary['passed']:
        return rc
    hb.write_once(hb.numbered(paths['record'], 'spy_breach'), dict(pair=pair, pair_set=pair_set, breaches=summary['breaches'], exit=rc, utc=hb.utc(),
                                                                  rule='LP-B review I3, C26.8-5: a read outside the allow list stops the arm (a tool error first)'))
    return rc if rc != 0 else EXIT['infra_failure']


def run(pair_set, pair, *, batch, resume_rel=None, pinned=None, launcher=None, certify=None, probe=None, kernel=None, move=True, _spied=False, _spy=None):
    """The driver (module docstring).  Returns the exit code; result.json written after the lock.  launcher, certify, probe, kernel: the tests' stand-ins (the frozen run, the certificate, the unit probe,
    the kernel's window).  LP-B: an LPBA pair runs under the allow-list spy (lpba_spied_run; `_spy` the installed spy: a breach it saw -- also one a library swallowed -- keeps the intermediate
    commits, diff review minor 1: the failure's evidence whole)."""
    if hb.is_lpba(pair_set) and not _spied:
        return lpba_spied_run(pair_set, pair, batch=batch, resume_rel=resume_rel, pinned=pinned, launcher=launcher, certify=certify, probe=probe, kernel=kernel, move=move)
    paths = cell_paths(pair_set, pair)
    probe = probe or step_probe
    kernel = kernel or kernel_window
    refuse_doc = dict(tool=hb.TOOL, arm=ARM, pair=pair, pair_set=pair_set)
    if Path(paths['batches']).resolve() not in Path(batch).resolve().parents:       # the refusal records go where the next attempt looks for them (XB's I1)
        print(f'refused: the batch {batch} is not under {paths["batches"]}', file=sys.stderr)
        return EXIT['refused']
    try:
        if pair_set == 'holdout':
            code = hb.code_record(pinned)                  # the batch's pinned commit (decision_log 2026-11-09 df8541d)
            problem = hb_method.prereg_problem()
            if problem:
                raise hc.Refused('preregistration', problem)
        elif hb.is_lpba(pair_set):                      # LP-B (C26.9, C26.6-3): the code, then the pins, before the admission
            code, problems = hb.lpba().code_record(pair_set, pinned)
            if problems:
                raise hc.Refused('code', f'{problems}')
            problem = hb_method.lpba_pins_problem()
            if problem:
                raise hc.Refused('pins', problem)
        else:
            code = hc.code_identity()
        rerun = rerun_authorization(pair_set, pair)     # XB's I1: refused before its lock -> a committed rerun authorization first
        record, allowlist, pair_path = admit(pair_set, pair)
        frozen_record, problems = he.frozen_identity()
        if problems:
            raise hc.Refused('frozen', f'the frozen tree: {problems}')
        resume = None
        if resume_rel is not None:
            resume = load_resume(pair_set, pair, resume_rel)
        hc.check_once(paths['lock'], resume=resume)
    except (hc.Refused, hb.Stop) as exc:
        hc.write_refusal(batch, dict(refuse_doc, stage=getattr(exc, 'stage', 'start'), reason=str(exc), seq=next_refusal_seq(pair_set, pair)))
        return EXIT['refused']
    lock = Path(resume.lock_path) if resume is not None else paths['lock']
    hc.take_lock(lock, hc.lock_line(arm=ARM, pair=pair, pair_set=pair_set, batch=str(batch), code=code.get('commit'), configc_reference_sha256=hb.sha(fc.CONFIGC_REFERENCE), unit=os.environ.get('HB_UNIT')))
    paths['record'].mkdir(parents=True, exist_ok=True)
    work = paths['work'] if resume is None else paths['work'].with_name(paths['work'].name+'.resume')
    result = dict(schema=hb.SCHEMA.format('configc_result'), arm=ARM, pair=pair, pair_set=pair_set, status=None, kind=None, admission=record, frozen=frozen_record, code=code, rerun_authorization=rerun,
                  lock=dict(path=str(lock), sha256=hb.sha(lock)), resume=None if resume is None else dict(path=resume.path, sha256=resume.sha256), work=str(work), utc_start=hb.utc())
    try:
        from bayesreg.io.pair import load_pair
        work.mkdir(parents=True, exist_ok=False)
        pin = load_pair(pair_path, allowlist=allowlist)
        result['firewall'] = fc.derive_pair(pin, work/'derived', pair=pair)
        prepared = fc.prepare(work/'derived', work/'run', pair=pair, scratch=work, frozen=he.core.FROZEN)
        result['prepare'] = {k: prepared[k] for k in ('slot', 'frozen_files', 'seconds')}
        gate, problems = fc.firewall_check(pin, work/'derived', work/'run', pair=pair)
        result['firewall']['gate'] = gate
        if problems:
            raise WrapperError(f'the firewall gate: {problems}')
        before = probe()
        t0 = fc.uptime()
        launch = (launcher or fc.DirectLauncher(work.parent/'cache'))(work/'run', hb.lpba().short(pair) if hb.is_lpba(pair_set) else hc.short_pair(pair))      # LP-B: S05-S06
        after = probe()
        text, oom, dxg = kernel(t0)
        events = hc.events_delta((before or {}).get('events'), (after or {}).get('events'))
        exit_code = (work/'run'/'exit_code').read_text().strip() if (work/'run'/'exit_code').is_file() else None
        result['classification'] = cls = fc.classify(work/'run', dxg=dxg or 0, oom=len(oom or []), watchdog=fc.WATCHDOG_SECONDS)     # classify_run unchanged; the counts are read below too
        step_kill = exit_code == '137' and cls.get('kind') != 'timeout'                 # a -9 of the frozen worker that is not the watchdog's (its kill after the grace is the watchdog's)
        status_u, kind_u, evidence = hc.classify_unit_end(unit=os.environ.get('HB_UNIT'), systemd_result=None, oom_lines=oom, events=events, journal=None, step_kill=step_kill,
                                                         unit_cgroup=(after or {}).get('cgroup'), need=('dmesg', 'end_probe'))
        result['launch'] = dict(launch, kernel=dict(dxg_lines=dxg, oom_lines=oom, dmesg_read=text is not None), step_reading=dict(status=status_u, kind=kind_u, evidence=evidence), exit_code=exit_code)
        if status_u is not None:
            result.update(status=status_u, kind=kind_u)
        elif dxg:
            result.update(status='infra_failure', kind='dxg')
        elif cls.get('class') in fc.CANDIDATE_CLASSES:
            config = certify_config()
            if certify is None:
                from jd_curl.three_d.feasibility import certify_map

                def certify(u, grid):
                    return certify_map(u, grid, config=config)
            result['candidates'] = []
            for name, path, crec in fc.candidates(work/'run', cls):
                row = dict(file=name, path=str(path), **{k: crec.get(k) for k in ('recorded_sha256', 'file_sha256', 'role')})
                try:
                    u = fc.load_candidate(path, crec, pin.grid)
                    problem = he.field_problem(u, pin.grid)
                    if problem:
                        raise ValueError(problem)
                    digest = he.array_sha256(u)
                    doc = he.certificate_document(certify(u, pin.grid), config, process=f'the configuration C arm (hb_configc.py, pid {os.getpid()}; reads no label): certified again', array_sha256_of_u=digest)
                    row.update(state_hash=digest, passed=he.passes(doc) and doc['state_hash'] == digest, status=doc['status'], min_jd_lower_bound=doc['min_jd_lower_bound'])
                except (OSError, ValueError, KeyError) as exc:
                    row.update(passed=False, error=f'{type(exc).__name__}: {exc}')
                result['candidates'].append(row)
                if row['passed']:
                    line = dict(schema=hb.SCHEMA.format('configc_line'), arm=ARM, pair=pair, line_status='configc_certified', candidate=name, classification=cls, grid=he.grid_record(pin.grid),
                                rule='prereg §2.2, D75: the first candidate that passes configuration C\'s certificate, certified again in this process')
                    result['output'] = he.write_output(work/'output', u, pin.grid, doc, line)
                    result.update(status='ok', kind=cls.get('class'))
                    break
            else:
                result.update(status='method_failure', kind='no_certified_state')
        elif cls.get('class') == 'method_failure':
            result.update(status='unit_limit' if cls.get('kind') == 'timeout' else 'method_failure', kind=f"configc_{cls.get('kind')}")
        elif cls.get('class') == 'infrastructure':
            result.update(status='infra_failure', kind=f"classify_run_{cls.get('kind')}")
        else:
            result.update(status='needs_lead_rejudgment', kind=f"classify_run_{cls.get('class')}_{cls.get('kind')}")
        if result['status'] == 'ok' and cls.get('class') == 'ok':
            if _spy is not None and _spy.breaches:      # LP-B diff review minor 1: the spy stopped a read that some code swallowed: nothing deleted (the evidence whole; lpba_spied_run stops the queue)
                result['intermediates'] = dict(kept=True, n_breaches=len(_spy.breaches), rule='LP-B diff review minor 1: a spy breach after the lock keeps every file of the run (project rules, hard rule 4)')
            else:
                result['intermediates'] = intermediates(work/'run', paths['record']/('intermediates.tsv' if resume is None else 'intermediates_resume.tsv'))
    except Exception as exc:                            # noqa: BLE001 - our code after the lock: wrapper_error (prereg revision 1a item 2, 1b item 5)
        result.update(status='wrapper_error', kind=type(exc).__name__, error=f'{type(exc).__name__}: {exc}'[:2000], traceback=traceback.format_exc()[-4000:])
        if hb.is_lpba(pair_set) and isinstance(exc, hb.lpba().lc.hp.Breach):
            result.update(status='infra_failure', kind='spy_breach')        # LP-B review I3: the allow-list spy stopped a read after the lock (as XB's driver: infra_failure, spy_breach)
    if _spy is not None and _spy.breaches and (result['status'], result['kind']) != ('infra_failure', 'spy_breach'):
        result['superseded'] = dict(status=result['status'], kind=result['kind'])     # diff review minor 1: a breach some code swallowed (as XB's spy_not_passed): infra_failure, the files kept
        result.update(status='infra_failure', kind='spy_breach')
    if move and paths['store'] is not None and work.exists():
        store = paths['store'] if resume is None else paths['store'].with_name(paths['store'].name+'.resume')
        there = sorted(str(q.relative_to(store)) for q in store.rglob('*') if q.is_file() or q.is_symlink()) if store.exists() else []
        try:
            if there:                                   # HB review I3: nothing moves into a destination that holds files; the work directory stays whole on /home
                raise hc.InfrastructureFailure('move_destination_exists', f'{len(there)} files already at {store} ({there[:5]}): nothing moved; to the lead')
            result['move'] = hc.move_files(work, store, paths['record']/('move.json' if resume is None else 'move_resume.json'))
            hb.remove_empty_tree(work)
            if result.get('output'):
                result['output_store'] = {k: dict(v, file=str(store/'output'/v['file'])) for k, v in result['output'].items()}
        except hc.InfrastructureFailure as exc:
            result.update(status='infra_failure', kind=exc.kind, move_error=str(exc))
    if hb.is_lpba(pair_set) and result.get('output'):   # LP-B (C26.5-5, decision_log line 532 (1)): nothing moves, the output_store is written all the same, the work cell's output/ (.resume's for a resume)
        result['output_store'] = {k: dict(v, file=str(work/'output'/v['file'])) for k, v in result['output'].items()}
    result['utc_end'] = hb.utc()
    paths['record'].mkdir(parents=True, exist_ok=True)
    hb.write_once(paths['result'] if resume is None else resumed_paths(paths)['result'], result)
    return EXIT[result['status']]


# ---- the queue ----------------------------------------------------------------------------------------------------------------------------------
def _refusal_seq(path):
    doc = hb.read_json(path)
    seq = doc.get('seq') if isinstance(doc, dict) else None
    return seq if isinstance(seq, int) and not isinstance(seq, bool) and seq >= 1 else None


def next_refusal_seq(pair_set, pair):
    """HB diff review n4: the sequence number of this cell's next refusal record (one more than the largest of its refusal records, those of stage rerun included; 1 for the first).  A record
    without one is passed over here (a refusal is always written) and refused by refusals_by_seq; when the records cannot be listed the number is None (the record is still written; the newest is
    then not known and refusals_by_seq refuses, to the lead)."""
    try:
        rows = hc.cell_refusals(roots(pair_set)[0], arm=ARM, pair=pair, pair_set=pair_set, counted=False)
    except (hc.Refused, OSError):
        return None
    return 1+max((q for q in (_refusal_seq(Path(r['path'])) for r in rows) if q is not None), default=0)


def refusals_by_seq(paths):
    """[(seq, path)] by the refusal records' sequence numbers (never mtime or the name; HB diff review n4); Refused when one has none or two share one: holdout_common.refusals_by_seq, which is
    bayesreg.io.rejudgment.by_seq (A5b review minor 6: one ordering)."""
    return hc.refusals_by_seq(paths)


def keep_refusals(pair_set, pair, paths=None):
    """The lead's item 15 (decision_log 2026-11-10 ~2:53 AM CST): every refusal record of this cell before its lock (holdout_common.write_refusal under <record root>/batches/*/refusals/;
    holdout_common.cell_refusals, those of stage rerun included) copied once into the cell's record directory, refusals/<batch>__<name>, byte for byte, with <copy>.sha256 ("<sha256>  <source>");
    the evaluation entry reads the record directory only.  Returns the copies (path, sha256, source, seq), oldest first by the records' sequence numbers (refusals_by_seq)."""
    paths = cell_paths(pair_set, pair) if paths is None else paths
    out = []
    rows = {r['path']: r for r in hc.cell_refusals(roots(pair_set)[0], arm=ARM, pair=pair, pair_set=pair_set, counted=False)}
    for seq, src in refusals_by_seq(rows):
        r = rows[str(src)]
        copy = paths['record']/'refusals'/f'{src.parent.parent.name}__{src.name}'
        if not copy.exists():
            copy.parent.mkdir(parents=True, exist_ok=True)
            with open(src, 'rb') as stream:
                data = stream.read()
            with open(copy, 'xb') as stream:
                stream.write(data)
            with open(copy.with_name(copy.name+'.sha256'), 'x', encoding='utf-8') as stream:
                stream.write(f'{hb.sha(copy)}  {src}\n')
        if hb.sha(copy) != r['sha256']:
            raise hc.Refused('refusal_copy', f'{copy} is not the refusal record {src} (sha256)')
        out.append(dict(path=str(copy), sha256=r['sha256'], source=str(src), seq=seq))
    return out


def declined_refusal(pair_set, pair, paths=None):
    """The lead's item 15 (P20): a cell refused before its lock, run again by a rerun authorization and refused again, is written (no output, P20) by BZ's committed no_resume record (tool hb, arm
    configc, note 未批补跑) whose record is the newest refusal record of the cell -- by the records' sequence numbers (HB diff review n4; never mtime) -- its copy in the record directory, or the
    refusal record itself, with that sha256, its reason naming P20 (bayesreg.io.rejudgment.p20_reason_problem: A5b review I2, the evaluation entry's criterion).  None or the authorization;
    Refused when the copies' sequence numbers do not tell the newest."""
    paths = cell_paths(pair_set, pair) if paths is None else paths
    copies = sorted((paths['record']/'refusals').glob('*.json')) if (paths['record']/'refusals').is_dir() else []
    if not copies or authorizations_problem(pair_set) is not None:
        return None
    newest = refusals_by_seq(copies)[-1][1]
    source = (newest.with_name(newest.name+'.sha256').read_text(encoding='utf-8').split('  ', 1)+[''])[1].strip() if newest.with_name(newest.name+'.sha256').is_file() else ''
    for record in (newest, Path(source) if source else None):
        if record is None:
            continue
        auth = hc.no_resume_authorization(tool=hb.TOOL, arm=ARM, pair=pair, pair_set=pair_set, record=record, complete=lambda: False,
                                          accept=lambda a: a.document.get('record_sha256') == hb.sha(newest) and hb.rejudgment_io.p20_reason_problem(a.document.get('reason')) is None)
        if auth is not None:
            return auth
    return None


def declined(pair_set, pair, paths=None):
    """BZ's 未批补跑 for a cell that stopped the batch (holdout_common.no_resume_authorization: tool hb, arm configc, its record the first run's effective record with that sha256): None or the record."""
    paths = cell_paths(pair_set, pair) if paths is None else paths
    if not first_record(paths).is_file() or authorizations_problem(pair_set) is not None:
        return None
    return hc.no_resume_authorization(tool=hb.TOOL, arm=ARM, pair=pair, pair_set=pair_set, record=first_record(paths), complete=lambda: False)


def state(pair_set, pair):
    """done (a record that lets the batch go on: a continuing class; a resumed run that failed again, written (RESUMED_WRITTEN); a stopped first run with BZ's committed 未批补跑 record), stopped (a
    record or a lock that stopped the batch: a rerun only by a committed resume record), not_run."""
    paths = cell_paths(pair_set, pair)
    doc, resumed = chain(paths)
    if doc is not None:
        if doc.get('status') in CONTINUES or (resumed and doc.get('status') in RESUMED_WRITTEN) or (not resumed and declined(pair_set, pair, paths) is not None):
            return 'done'
        accept = hb.CONFIGC_RESUMED_JUDGMENTS if resumed else hb.CONFIGC_JUDGMENTS
        if doc.get('status') == 'needs_lead_rejudgment' and hb.rejudged(ARM, pair, effective_path(paths), accept, pair_set=pair_set)[0] is not None:
            return 'done'                               # item 13: a committed rejudgment of the effective record (unit_limit by the lead, no_output by BZ; after a resume infrastructure too, N2 (i))
        return 'stopped'
    if paths['lock'].exists():
        return 'done' if declined(pair_set, pair, paths) is not None else 'stopped'
    if declined_refusal(pair_set, pair, paths) is not None:
        return 'done'                                   # item 15: refused before its lock (P20) after a rerun; BZ's no_resume: written, no output
    return 'not_run'


def run_one(pair_set, pair, system, *, batch, log, resume_rel=None, pinned=None):
    paths = cell_paths(pair_set, pair)
    try:
        hb.gate_of(system, 'configc', pair_set)
        system.disk(CONFIGC_D_NEED_GIB if pair_set == 'holdout' else 0., hb.UNIT_NEED_GIB['configc'][1])
    except (hc.InfrastructureFailure, hc.Refused) as exc:
        raise hb.Stop(hb.EXIT['infra'], f'{pair}: before the unit: {exc}') from None
    paths['record'].mkdir(parents=True, exist_ok=True)
    unit_dir = hb.attempt_dir(paths['record']/'unit')
    inner = [HERE/'hb_configc.py', 'run', '--pair', pair, '--pair-set', pair_set, '--batch', str(batch), *(['--resume', resume_rel] if resume_rel else []), *(['--pinned', pinned] if pinned else [])]
    reading = hb.run_unit('configc', pair, inner, system, batch=batch, unit_dir=unit_dir, res=dict(hb.CONFIGC_UNIT), gpu=True, code=hb.REPO, pair_set=pair_set)
    own = paths if not resume_rel else dict(paths, **resumed_paths(paths))      # the resumed run's records: result_resume.json, overridden_resume.json
    doc = hb.read_json(own['result'])
    if doc is None:
        locked = Path(hc.resume_lock_path(paths['lock'])).exists() if resume_rel else paths['lock'].exists()        # HB review I4: a resume's own lock is <lock>.resume
        if not locked:
            kept = keep_refusals(pair_set, pair, paths)     # item 15: the refusal records copied into the cell's record directory
            raise hb.Stop(hb.EXIT['refused'], f'{pair}: the driver stopped before its {"resume " if resume_rel else ""}lock (its refusal record in {batch}/refusals/, copied to '
                                              f'{kept[-1]["path"] if kept else "-"}); nothing else written here')
        status = reading['status'] or 'needs_lead_rejudgment'
        doc = dict(schema=hb.SCHEMA.format('configc_result'), arm=ARM, pair=pair, pair_set=pair_set, status=status, kind=reading['status_kind'] or 'no_record', unit_reading=reading,
                   written_by='hb_configc.py queue (the driver left no record after its lock)', utc=hb.utc())
        hb.write_once(own['result'], doc)
    elif reading['status'] in ('infra_failure', 'needs_lead_rejudgment') and doc.get('status') not in ('infra_failure', 'needs_lead_rejudgment'):
        over = dict(schema=hb.SCHEMA.format('configc_result'), arm=ARM, pair=pair, pair_set=pair_set, status=reading['status'], kind=reading['status_kind'], unit_reading=reading,
                    overrides=dict(path=str(own['result']), sha256=hb.sha(own['result']), status=doc.get('status')), output=None, resumed=bool(resume_rel),
                    written_by='hb_configc.py queue (the unit\'s evidence contradicts the driver\'s record; xb_chain I2)', utc=hb.utc())
        hb.write_once(own['overridden'], over)
        doc = over
    log(f'{pair}: {doc.get("status")} ({doc.get("kind")})')
    breaches = sorted(paths['record'].glob('spy_breach*.json')) if hb.is_lpba(pair_set) else []
    if breaches:                                        # LP-B review I3 (C26.8-5): the driver's allow-list spy stopped a read: the batch stops whatever the record says
        raise hb.Stop(hb.EXIT['stop'], f'{pair}: the allow-list spy stopped a read ({[str(b) for b in breaches]}): the batch stops (C26.8-5)')
    if doc.get('status') in CONTINUES:
        return doc['status']
    if resume_rel and doc.get('status') in RESUMED_WRITTEN:
        log(f'{pair}: the resumed run failed again ({doc.get("status")}): written, no output, no third run (prereg §5.4)')
    code = hb.EXIT['infra'] if doc.get('status') == 'infra_failure' else hb.EXIT['rejudge'] if doc.get('status') == 'needs_lead_rejudgment' else hb.EXIT['stop']
    raise hb.Stop(code, f'{pair}: {doc.get("status")} ({doc.get("kind")}): the batch stops')


def queue(pair_set, *, pairs=None, resumes=None, commit=None, system=None, echo=print, code_run=None):
    system = hb.System() if system is None else system
    resumes = dict(resumes or {})
    if pair_set == 'holdout':
        record, problems = hb_method.start_checks(commit, code_run)
        if problems:
            echo(f'refused at the start: {problems}')
            return hb.EXIT['refused']
        plan_all = hb.rows()
    elif hb.is_lpba(pair_set):                          # LP-B: the code, the settings, the pins (hb_method.lpba_start_checks); the mode's table
        roots(pair_set)
        record, problems = hb_method.checks_of(pair_set, commit, code_run)
        if problems:
            echo(f'refused at the start: {problems}')
            return hb.EXIT['refused']
        plan_all = hb.rows_of(pair_set)
    else:
        record, plan_all = dict(code=hc.code_identity(run=code_run)), list(hc.DEV_PAIRS)
    batch = hb.attempt_dir(roots(pair_set)[0]/'batches'/f'hb_configc_{hb.stamp()}')
    batch.mkdir(parents=True)
    log = hb.Log(batch, echo)
    record.update(schema=hb.SCHEMA.format('batch'), kind='configc', pair_set=pair_set, utc_start=hb.utc(), cells=[], unit=dict(hb.CONFIGC_UNIT), configc_reference_sha256=hb.sha(fc.CONFIGC_REFERENCE))
    code = 0
    try:
        plan = plan_all if not pairs else [p for p in plan_all if p in pairs]
        if pairs and len(plan) != len(pairs):
            raise hb.Stop(hb.EXIT['refused'], f'pairs not in the plan: {sorted(set(pairs)-set(plan))}')
        log(f'start: {len(plan)} pairs')
        for pair in plan:
            st = state(pair_set, pair)
            record['cells'].append(dict(pair=pair, state_before=st))
            if st == 'done':
                continue
            resume_rel = resumes.get(pair)
            if st == 'stopped' and resume_rel is not None:
                try:                                    # HB review I4 (as hb_method.queue): the resume record holds before the unit starts; nothing is written otherwise
                    record['cells'][-1]['resume'] = dict(path=load_resume(pair_set, pair, resume_rel).path)
                except hc.Refused as exc:
                    raise hb.Stop(hb.EXIT['refused'], f'{pair}: {exc}') from None
            elif resume_rel is not None:
                raise hb.Stop(hb.EXIT['refused'], f'{pair}: a resume record for a cell that is {st}, not stopped')
            if st == 'stopped' and resume_rel is None:
                raise hb.Stop(hb.EXIT['resume'], f'{pair}: a lock or a record that stopped the batch: a rerun needs a committed resume record (prereg §5.4)')
            if st == 'not_run':                         # XB's I1 (C23.12-6, D107): a cell refused before its lock runs again only with a committed rerun authorization listing those refusal records
                try:
                    record['cells'][-1]['rerun_authorization'] = rerun_authorization(pair_set, pair)
                except hc.Refused as exc:
                    raise hb.Stop(hb.EXIT['refused'], f'{pair}: {exc}') from None
            while hold_file(pair_set).exists():
                system.sleep(60)
            record['cells'][-1]['status'] = run_one(pair_set, pair, system, batch=batch, log=log, resume_rel=resume_rel,
                                                    pinned=(record.get('code') or {}).get('commit') if pair_set == 'holdout' or hb.is_lpba(pair_set) else None)
    except hb.Stop as exc:
        code = exc.code
        record['stop'] = dict(code=code, reason=str(exc))
        log(f'STOP ({code}): {exc}')
    except hc.Refused as exc:
        code = hb.EXIT['refused']
        record['stop'] = dict(code=code, reason=str(exc))
        log(f'STOP ({code}): {exc}')
    except Exception as exc:                            # noqa: BLE001 - HB review m11: the batch record is written whatever stopped the queue
        code = hb.EXIT['stop']
        record['stop'] = dict(code=code, reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
        log(f'STOP ({code}): an exception of the queue: {type(exc).__name__}: {exc}')
    record.update(utc_end=hb.utc(), exit=code)
    hb.write_once(batch/'batch.json', record)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    d = sub.add_parser('dry-run')
    d.add_argument('--pair-set', choices=PAIR_SETS, default='holdout')
    q = sub.add_parser('queue')
    q.add_argument('--pair-set', choices=PAIR_SETS, required=True)
    q.add_argument('--commit', default=None)
    q.add_argument('--pairs', default=None)
    q.add_argument('--resume', action='append', default=[])
    ls = sub.add_parser('launch-string')
    ls.add_argument('--pair-set', choices=PAIR_SETS, required=True)
    r = sub.add_parser('run')
    r.add_argument('--pair', required=True)
    r.add_argument('--pair-set', choices=PAIR_SETS, required=True)
    r.add_argument('--batch', required=True)
    r.add_argument('--resume', default=None)
    r.add_argument('--pinned', default=None)
    o = ap.parse_args(argv)
    if o.command == 'dry-run':
        if o.pair_set == 'holdout' or hb.is_lpba(o.pair_set):
            record, problems = hb_method.checks_of(o.pair_set)
        else:
            record, problems = dict(code=hc.code_identity()), []
        print(json.dumps(dict(record, problems=problems), indent=1, default=str))
        return hb.EXIT['refused'] if problems else 0
    if o.command == 'launch-string':
        print(' '.join(hb.queue_launch('configc', ['queue', '--pair-set', o.pair_set], pair_set=o.pair_set if hb.is_lpba(o.pair_set) else 'holdout')))
        return 0
    if o.command == 'queue':
        return queue(o.pair_set, pairs=o.pairs.split(',') if o.pairs else None, resumes=hb_method.parse_resumes(o.resume), commit=o.commit)
    os.environ.setdefault('HB_UNIT', hb.unit_name('configc', o.pair, o.pair_set if hb.is_lpba(o.pair_set) else 'holdout'))
    return run(o.pair_set, o.pair, batch=Path(o.batch), resume_rel=o.resume, pinned=o.pinned)


if __name__ == '__main__':
    sys.exit(main())
