"""PQH: the two hold-out PQ units (contract C22.H3-C22.H6, revision 7; D113 (a), D114 (a), D117 (a), D121 (a), D123 (b)) and their dev mode (the smoke's segments (i), (ii), C22.H8, D124 (a)).

members (tag-hb-pq-members, GPU; C22.H5, D121 (a)): the four registered members of unet35_v4_ens inferred again on the 100 hold-out subjects, V = sum_c Var_m P_m^c per voxel (float32), with
    experiments/pq/pq_members.mean_and_variance and subject_record (not copied: D121 (a)); Hm5 against HI's probs36 (agreement >= .9999).  Admission as HI's infer_holdout steps 1-10 with the same unseal
    record (hi_core.load_unseal, bind, admit, check_registry, check_decision, check_environment; D24 through pq_members.plan_for -> hs_seg.uf_plan): every subject passes before the first image is read.
out (tag-hb-pq-out, CPU; C22.H3, H4, H6): every pair of holdout_B_pairs.tsv, after the finalization record of step 5: Hm1-Hm4 on all pairs with an output first (any failure: the unit stops, pq_refused.json,
    no PQ output, BZ decides, C22.H4-2), then PQ and the label-free baselines with the development package's functions (pqh_inputs.pq_arrays = pq_io.compute_pair's order); pq/<pair>/pq.json (on /home),
    pq.npz and baselines.npz (on D:); a pair without the method's output pq_absent.json; a pair whose PQ fails half way pq_failed.json (kept, not rerun).  line.json is never changed (D114 (a)).
Both, before anything of a subject is read (in this order): the spy over an empty sealed-list cache (C23.6-5; pqh_firewall), plan.md and (hold-out) the preregistration pinned, the frozen settings' sha256,
    the process inside its own unit, the finalization record and the code tree (code_check: the detached checkout of C22.H0-4, the code directories those of the record's commit by the global rule
    of 11-09 as rule one (unit_code_identity: holdout_common.pinned_code_problems, P1 from runs/holdout_B_v1/pinned_code.json); the smoke's checkout he_smoke_v3/code at its HEAD, no tracked
    file modified (I5, smoke_code_identity); the code imported from it (m7)),
    refusals before an earlier lock covered by a committed rerun authorization (holdout_common.rerun_authorization, I1), the lock (once; a resume only through holdout_common.load_resume,
    D123 (b); a rerun after a failed control only through a committed authorization of kind rerun listing that run's stop records, at most once, C22.H4-2 (i)), the admission (SG through hi_core /
    holdout_common), the order (C22.H3-3), the disks (holdout_common.disk_guard, both sides), the machine gate (holdout_common.machine_gate, waited for up to 15 minutes; MemAvailable of C22.H5-2 / H6-3).
    A refusal before the lock writes one record under pq/batches/<batch>/refusals/ (holdout_common.write_refusal), never into the unit's records -- except a control (Hm1-Hm6; the sealed-list cache
    filled before the spy and a breach are Hm6): pq_refused.json in the unit's record directory, BZ decides (review m3, C22.H4-2 last item, C22.H4-3 item 2).
A resume (D123 (b)) computes the pairs or subjects the killed run did not finish; the ones it left half written, and the pq_failed pairs to compute again, are those its committed resume record lists
    (resume_items: `recompute: ...` in its reason), renamed whole to <name>.attempt<k> first and kept (keep_attempt; I2).  After the lock no exception leaves a unit: the opened files and the records
    are always written (write_record).  Every write goes through pqh_inputs.guard_write (I7).
Where a run writes (pqh_inputs.run_dirs): the first run and its resume pq/members/ and pq/<pair>/ (records on /home, V and the npz on D:); a rerun after a failed control beside them (members_rerun/, rerun/);
    nothing is overwritten.  The sha256 list (pq_outputs_manifest.txt) names every file of the run that counts by its absolute path: it is what the evaluation entry reads (Ev1).
The dev mode takes the development allowlist instead of the unseal object, the pair 0438-0439, the registered development predictions, and writes only under he_smoke_v3/pq/ (its own locks); it never resumes
    or reruns (C22.H8: a smoke that does not pass is done again in a new root).
Exit codes: 0 done; 2 refused before the lock; 8 stopped (a control failed -- pq_refused.json, before or after the lock -- or an error after it; the records say which); 20 an infrastructure refusal
before the lock.
"""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pq_io as io_  # noqa: E402
import pq_pins as pins  # noqa: E402
import pqh_firewall as hfw  # noqa: E402
import pqh_inputs as hin  # noqa: E402
import pqh_pins as hp  # noqa: E402
from pq_core import PQRefused  # noqa: E402

REPO = HERE.parents[1]
EXIT = dict(ok=0, refused=2, stopped=8, infra=20)
STOPPED = 'pq_refused.json'
CONTROL_STAGES = ('hm1', 'hm2', 'hm3', 'hm4', 'hm5', 'hm6')
OUT_RECORD, OUT_RESUME_RECORD = 'pq_record.json', 'pq_record.resume.json'
MANIFEST, MANIFEST_RESUME = 'pq_outputs_manifest.txt', 'pq_outputs_manifest.resume.txt'
PQ_JSON, PQ_NPZ, BASELINES, ABSENT, FAILED = 'pq.json', 'pq.npz', 'baselines.npz', 'pq_absent.json', 'pq_failed.json'
RELEASE_SHA_REL, RELEASE_NOTE_REL = 'runs/holdout_B_v1/pq_outputs_manifest.sha256', 'runs/holdout_B_v1/pq_release.md'     # 〔对齐〕 the lead's release (C22.H4-5); EV's Ev1 reads them


def hc():
    return hin.hc()


def hi():
    import hi_core
    return hi_core


def utc():
    return hc().utc()


@dataclass
class Deps:
    """What a unit asks of the machine and of the pinned inputs (the tests replace them; None: the real one)."""
    gate: object = None                     # (unit) -> record
    disk_usage: object = None               # shutil.disk_usage
    git: object = None                      # subprocess.run for bayesreg.io.code_identity's git calls (both modes)
    identity: object = None                 # (layout) -> (code record, problems): replaces unit_code_identity / smoke_code_identity (tests)
    imports: object = None                  # (layout) -> [problems]: import_problems (m7: bayesreg, this package, holdout_common, hi_core from the code tree)
    in_unit: object = None                  # (unit name) -> bool
    probe: object = None                    # (when) -> record (experiments/he/unit_probe.probe)
    allowlist: object = None                # () -> (DevAllowlist, record): the pinned allowlist with its sources re-hashed (members unit)
    files: object = None                    # () -> OasisFiles of the data root (members unit)
    device: object = None                   # (device, free_min_gib) -> torch.device
    environment: object = None              # (device, decision) -> record
    decision: object = None                 # () -> HI's development-check decision (hi_core.check_decision)
    sealed_training: object = None          # () -> (SealedLists, training subjects) for hi_core.admit (members unit)
    taper: object = None                    # (u, grid, L) -> (u_tapered, w): the frozen tree's boundary_taper
    log: object = print
    extra: dict = field(default_factory=dict)


class Refusal(Exception):
    """A refusal before the lock: stage and class (refusal, or infrastructure: the machine, the disks, a breach of the spy there)."""

    def __init__(self, stage, message, cls='refusal'):
        super().__init__(f'{stage}: {message}')
        self.stage, self.cls = stage, cls


# ---- the checks before the lock -----------------------------------------------------------------------------------------------------------------
def prereg_problem(code):
    path = Path(code)/hp.PREREG_REL
    if hp.PREREG_SHA256 is None:
        return 'PREREG_SHA256 is None: the preregistration is not pinned in this code (the lead fills it after BZ adopts revision 2; C22.H3-6)'
    digest = io_.sha_file(path) if path.is_file() else None
    return None if digest == hp.PREREG_SHA256 else f'{hp.PREREG_REL}: sha256 {digest} is not the pinned {hp.PREREG_SHA256}'


# ---- the code identity (rule one for the units; HB's shared functions since HB's merge d008f50: holdout_common.pinned_code_problems -> bayesreg.io.code_identity) -----------------------------
def unit_code_identity(repo, run=None):
    """Rule one (contract revision 9 draft C24.3; decision_log 2026-11-10 ~2:59 AM CST (1)-(4)): holdout_common.pinned_code_problems -> bayesreg.io.code_identity.unit_identity_problems.
    P1 only from the committed runs/holdout_B_v1/pinned_code.json -- never HEAD, and no longer the finalization record's code.commit (the lead's instruction after HB's merge); every commit of
    P1..HEAD on its own (authorizations and records added once, the decision log only appended to, the allowlist modified once with the unseal record: allowlist_inclusion_problems); the code
    (the five directories, the root's importable files, new top-level directories) unchanged; a clean working tree.
    Returns (code record, problems); the record maps HB's fields as hb_core.code_record does (review NM1): commit = P1, head = HEAD, clean = passed; `identity` is HB's whole record."""
    record, problems = hc().pinned_code_problems(None, repo=repo, run=run)
    return dict(commit=record.get('pinned'), head=record.get('head'), clean=bool(record.get('passed')) and not problems, identity=record,
                rule='rule one: holdout_common.pinned_code_problems (P1 from runs/holdout_B_v1/pinned_code.json)'), problems


def checkout_code_identity(repo, run=None):
    """The smoke's checkout and the replay gate's run (neither has a pinned record): the checkout's code is its own HEAD's -- bayesreg.io.code_identity.gate_identity_problems from HEAD to HEAD,
    i.e. its working-tree part: no tracked change in the code directories or at the root, no untracked or ignored file in the code directories but a compiled file beside its source, no
    untracked or ignored importable file at the root.  Returns (code record with commit = head = HEAD and clean = passed, problems)."""
    from bayesreg.io import code_identity
    record, problems = code_identity.gate_identity_problems('HEAD', repo=repo, run=run)
    return dict(commit=record.get('head'), head=record.get('head'), clean=bool(record.get('passed')) and not problems, identity=record,
                rule="the checkout at its own HEAD: bayesreg.io.code_identity.gate_identity_problems('HEAD') (its working-tree checks)"), problems


def smoke_code_identity(repo, run=None):
    """The smoke's three segments (dev mode; review 3 r1, the lead's ruling: the stricter reading): checkout_code_identity and, as the working-tree part of rule one, no tracked file of the whole
    tree modified or staged (git status --porcelain=v1 -z --untracked-files=no; docs/, runs/, splits/ included).  Returns (code record with tracked_changes, problems)."""
    from bayesreg.io import code_identity
    code, problems = checkout_code_identity(repo, run)
    git, _ = code_identity.git_runner(Path(repo), run)
    rc, out, err = git('status', '--porcelain=v1', '-z', '--untracked-files=no')
    changes = [e for e in out.split('\0') if e] if rc == 0 else None
    problems = list(problems)
    if rc != 0:
        problems.append(f'git status: exit {rc} {err}')
    elif changes:
        problems.append(f'tracked files modified or staged in the working tree: {changes[:10]} (the smoke runs from a clean checkout, C22.H8; review 3 r1)')
    return dict(code, clean=code['clean'] and not problems, tracked_changes=changes,
                rule=code['rule']+'; and no tracked file of the whole tree modified (git status --untracked-files=no; review 3 r1)'), problems


def code_check(layout, deps, finalized, checks):
    """The code tree of a run (C22.H0-4; I5; review NM1).  Hold-out: the detached checkout holdout_B_v1/code, by rule one (unit_code_identity; P1 from runs/holdout_B_v1/pinned_code.json).
    Dev (the smoke, C22.H8 "冒烟各单元也从一份分离检出跑"): the smoke root's checkout he_smoke_v3/code at its own HEAD with no tracked file modified anywhere (smoke_code_identity).  Both: the code imported from that tree
    (import_problems, m7).  Fills checks['code'] ({commit, head, clean, identity, rule, checkout}); Refusal('code') otherwise.  deps.identity (layout) -> (record, problems) replaces both in tests."""
    h = hc()
    checkout = Path(h.HOLDOUT_ROOT)/'code' if layout.mode == 'holdout' else Path(hp.SMOKE_ROOT)/'code'
    if deps.identity is not None:
        code, problems = deps.identity(layout)
    elif layout.mode == 'holdout':
        code, problems = unit_code_identity(layout.code, deps.git)
    else:
        code, problems = smoke_code_identity(layout.code, deps.git)
    checks['code'] = dict(code, checkout=str(checkout),
                          finalization_record_code=None if not finalized else (finalized[0].get('code') or {}).get('commit'))     # recorded only: P1 is the pinned record's
    if Path(layout.code).resolve() != checkout.resolve():
        raise Refusal('code', f'{layout.code} is not the detached checkout {checkout} (C22.H0-4; C22.H8 for the smoke)')
    if problems:
        raise Refusal('code', f'{problems}')
    problems = deps.imports(layout) if deps.imports is not None else import_problems(layout)
    checks['imports'] = dict(problems=problems, rule='m7: bayesreg, this package, holdout_common and hi_core imported from the checkout; jd_curl from it (members) or from the frozen tree (out, the replay)')
    if problems:
        raise Refusal('code', f'{problems}')
    return checks['code']


def import_problems(layout):
    """m7 (HE's pythonpath_problems' way): the modules this process runs come from the checkout -- bayesreg, pqh_units (this package), pq_core, holdout_common, hi_core and, when loaded, pq_io,
    pq_members, mc0_core; jd_curl from the checkout or from the frozen tree (the out unit and the replay run with PYTHONPATH=<frozen>:<code>, mc0_core.frozen_api).  [] or the problems."""
    import bayesreg
    import jd_curl
    import pq_core
    code = Path(layout.code).resolve()
    modules = dict(bayesreg=bayesreg, pqh_units=sys.modules[__name__], pq_core=pq_core, holdout_common=hc(), hi_core=hi())
    modules.update({name: sys.modules[name] for name in ('pq_io', 'pq_members', 'mc0_core', 'pqh_inputs', 'pqh_firewall') if name in sys.modules})
    problems = [f'{name} is imported from {Path(m.__file__).resolve()}, not from {code}' for name, m in sorted(modules.items()) if code not in Path(m.__file__).resolve().parents]
    where = Path(jd_curl.__file__).resolve()
    if code not in where.parents and Path(hp.FROZEN).resolve() not in where.parents:
        problems.append(f'jd_curl is imported from {where}, neither {code} nor the frozen tree {hp.FROZEN}')
    return problems


def unit_names(mode):
    return hp.UNITS if mode == 'holdout' else hp.DEV_UNITS


def unit_name_of(layout, unit):
    """The unit this run must be in: the layout's own out unit (SM3: the dev mode's (iv-b), (iv-c), pqh_pins.DEV_OUT_UNITS), else unit_names(mode)[unit]."""
    if unit == 'out' and getattr(layout, 'out_unit', None):
        return layout.out_unit
    return unit_names(layout.mode)[unit]


def own_unit_problem(unit_name, deps):
    inside = deps.in_unit(unit_name) if deps.in_unit is not None else bool(hi().units_of_this_process(units=(unit_name,)))
    return None if inside else f'run it in its unit {unit_name} (run_pqh.py launch prints and starts it)'


def gate_of(unit, mode, deps, unit_name=None):
    """The machine gate inside the unit: holdout_common.machine_gate with this process's own unit (`unit_name`, default unit_names(mode)[unit]; SM3: the caller passes the unit the process
    really runs in -- the P3 replay's tag-hb-pq-p3-replay, the (iv) segments' out units -- since an own unit the process is not in counts as another active unit and the gate never opens) and, in
    the dev mode only, the smoke's v3 queue as a parent unit (pqh_pins.DEV_PARENT_UNITS: it waits for this unit, the way xb_drive's chain does)."""
    if deps.gate is not None:
        return deps.gate(unit)
    h = hc()
    unit_name = unit_names(mode)[unit] if unit_name is None else unit_name
    parents = tuple(hp.DEV_PARENT_UNITS) if mode == 'dev' else ()
    return h.wait_for(lambda: h.machine_gate(own_units=(unit_name,), parent_units=parents, mem_min_gib=hp.GATES[unit]['mem_available_gib']), log=deps.log)


def probe(when, deps):
    if deps.probe is not None:
        return deps.probe(when)
    if str(HERE.parent/'he') not in sys.path:
        sys.path.append(str(HERE.parent/'he'))
    import unit_probe
    try:
        return unit_probe.probe(when)
    except Exception as exc:                        # noqa: BLE001 - a probe never fails a run (unit_probe's rule)
        return dict(when=when, error=f'{type(exc).__name__}: {exc}')


def previous_stop_records(layout, unit):
    """The records of this unit's runs that stopped on a control (C22.H4-2: pq_refused*.json; a members record whose stage is a control or whose spy failed): the first run's and -- review NM4 --
    the control rerun's (a rerun stopped before its lock writes pq_refused.json in its own directory; the authorization that let it start no longer covers the stops, a new one must list them all)."""
    names = []
    for kind in ('first_run', 'control_rerun'):
        small, _ = hin.run_dirs(layout, unit, kind)
        names += sorted(small.glob('pq_refused*.json')) if small.is_dir() else []      # m3: a stop before the lock may sit beside an earlier one (stop_record_path, write_record)
        if unit == 'members':
            names.append(small/hin.MEMBERS_RECORD)
    return [p for p in names if p.is_file() and _is_stop(p)]


def _is_stop(path):
    if path.name.startswith('pq_refused'):
        return True
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return True
    return document.get('stage') in CONTROL_STAGES or (document.get('spy') or {}).get('m6') is not True


def control_rerun(layout, unit, lock):
    """C22.H4-2 (i), C22.H8 拒绝之后的重跑: after a run that stopped on a control, the unit runs again only with a committed authorization of kind rerun (holdout_common.find_authorizations) whose
    `refusals` list exactly the stop records (path and sha256; previous_stop_records: the first run's and a stopped rerun's own, review NM4), at most once (its own lock <lock>.rerun taken: a
    second rerun is refused).  None when there is no such stop; else the authorization and the lock; Refusal otherwise.  The dev mode never reruns (C22.H8: a new root)."""
    stops = previous_stop_records(layout, unit)
    if not stops:
        return None
    if layout.mode != 'holdout':
        raise Refusal('rerun', f'{[str(p) for p in stops]}: the dev mode does not rerun (C22.H8: a smoke that does not pass is done again in a new root)')
    own = Path(lock).with_name(Path(lock).name+'.rerun')
    if own.exists():
        raise Refusal('rerun', f'{own} exists: the unit was rerun once after a control failed; no second rerun (C22.H8)')
    h = hc()
    want = sorted((str(p), io_.sha_file(p)) for p in stops)
    valid, refused = h.find_authorizations(kind='rerun', tool=hp.TOOL, arm=hp.PQ_ARM, pair=unit, pair_set=layout.pair_set)
    for auth in valid:
        listed = auth.document['refusals']
        if isinstance(listed, list) and all(isinstance(r, dict) for r in listed) and sorted((r.get('path'), r.get('sha256')) for r in listed) == want:
            return dict(path=auth.path, sha256=auth.sha256, stops=[dict(path=p, sha256=s) for p, s in want], lock=str(own))
    raise Refusal('rerun', f'the last run of {unit} stopped on a control ({want}); it runs again only with a committed authorization of kind rerun listing exactly these records (C22.H4-2 (i)); '
                           f'found {[a.path for a in valid]}, refused {refused}')


def write_refusal(layout, unit, document, log):
    """I1 (holdout_common.write_refusal): one record per refusal before the lock under <pq>/batches/<unit>_<utc>_<pid>/refusals/."""
    batch = Path(layout.batch_root)/f"{unit}_{utc().replace(':', '').replace('-', '')}_{os.getpid()}"
    hin.guard_write(batch)
    path = hc().write_refusal(batch, dict(document, tool=hp.TOOL, arm=hp.PQ_ARM, pair=unit, pair_set=layout.pair_set))
    log(f'refusal record {path}')
    return path


def refuse(layout, unit, spy, checks, exc, log, plan=None):
    """A refusal before the lock.  A control stage (Hm1-Hm6; the sealed-list cache filled before the spy and a breach of the spy are Hm6: C22.H4-3 allowed item 2, review m3) stops the unit with
    pq_refused.json in its record directory, BZ decides (C22.H4-2 last item; exit 8); any other stage writes holdout_common's refusal record beside the unit (I1; exit 2, or 20 for the machine)."""
    cls, stage = getattr(exc, 'cls', 'refusal'), getattr(exc, 'stage', 'refusal')
    if stage in CONTROL_STAGES:
        return stop_before_lock(layout, unit, spy, checks, stage, str(exc), log, plan)
    document = dict(stage=stage, refusal_class=cls, reason=str(exc)[:4000], mode=layout.mode, unit=unit, checks_run=sorted(checks), checks=checks, spy=None if spy is None else spy.record(),
                    rule='I1 (holdout_common): a refusal before the lock is recorded beside the unit, never in its records; a hold-out run after it needs a committed rerun authorization listing this record')
    write_refusal(layout, unit, document, log)
    log(f'{unit}: refused before the lock ({stage}): {exc}')
    return EXIT['infra'] if cls == 'infrastructure' else EXIT['refused']


def stop_record_path(directory):
    """pq_refused.json in the run's record directory, or pq_refused_<k>.json beside an earlier one (never overwritten)."""
    directory = Path(directory)
    if not (directory/STOPPED).exists():
        return directory/STOPPED
    k = 2
    while (directory/f'pq_refused_{k}.json').exists():
        k += 1
    return directory/f'pq_refused_{k}.json'


def stop_before_lock(layout, unit, spy, checks, stage, error, log, plan=None):
    """m3: a control failed before the lock (C22.H4-2 Hm1-Hm6; C22.H4-3 item 2 for the sealed-list cache).  pq_refused.json in the record directory this run would have written (the first run's,
    or the control rerun's when one was planned), with the spy's record; nothing of PQ is written; BZ decides ((i) a rerun through a committed authorization listing this record, (ii) F4 described
    only).  Exit 8."""
    if unit in ('members', 'out'):
        small, _ = hin.run_dirs(layout, unit, (plan or {}).get('kind', 'first_run'))
    else:
        small = Path(layout.pq_small)/unit                     # the smoke's P3 replay: its own directory
    path = stop_record_path(small)
    document = dict(schema=hp.SCHEMA.format('stopped'), unit=unit, mode=layout.mode, stage=stage, error=str(error)[:4000], before_lock=True, utc=utc(), checks_run=sorted(checks),
                    spy=None if spy is None else spy.record(),
                    rule='C22.H4-2 last item (review m3): a control failed before the lock -- the sealed-list cache filled before the spy (C22.H4-3 item 2: "名单缓存在装 spy 之前已经有值"), a breach, '
                         'Hm1-Hm5 -- the unit stops; BZ decides before the truth ((i) a rerun through a committed authorization listing this record, (ii) F4 described only)')
    write_record(path, document, log)
    log(f'{unit}: stopped before the lock ({stage}): {error}; {path}')
    return EXIT['stopped']


def as_refusal(exc):
    if isinstance(exc, Refusal):
        return exc
    if isinstance(exc, hfw.Breach):
        return Refusal('hm6', str(exc))                 # m3: a breach before the lock is Hm6 (C22.H4-2), to BZ
    if isinstance(exc, PQRefused):
        return Refusal(exc.stage, str(exc))
    if isinstance(exc, SystemExit):                     # m7: mc0_core.frozen_api refuses with SystemExit
        return Refusal('frozen', f'SystemExit: {exc}')
    return Refusal(getattr(exc, 'stage', type(exc).__name__), f'{type(exc).__name__}: {exc}')     # HIRefused, SubjectSealed, PinnedInputRefused, holdout_common.Refused, ...


# ---- the writes (I7: every one through pqh_inputs.guard_write first) ------------------------------------------------------------------------------
def write_once(path, data=None, *, text=None):
    return io_.write_once(hin.guard_write(path), data, text=text)


def save_npz_once(path, **arrays):
    return io_.save_npz_once(hin.guard_write(path), **arrays)


def write_json_once(path, document):
    return hc().write_json_once(hin.guard_write(path), document)


def make_dir(path):
    Path(hin.guard_write(path)).mkdir(parents=True, exist_ok=True)
    return Path(path)


def write_record(path, document, log=print):
    """A record after the lock that must be written (I2: "opened_files 与记录一定写"): at its name, or -- if that name is taken or the write fails -- beside it as <stem>.<utc>.<pid>.json; a failure of
    that too is logged, never raised.  Returns the path written or None."""
    path = Path(path)
    try:
        return write_json_once(path, document)
    except hc().RealRootInTest:
        raise
    except Exception as exc:                        # noqa: BLE001 - the record is written beside it
        other = path.with_name(f"{path.stem}.{utc().replace(':', '').replace('-', '')}.{os.getpid()}.json")
        try:
            return write_json_once(other, dict(document, written_beside=str(path), why=f'{type(exc).__name__}: {exc}'))
        except Exception as exc2:                   # noqa: BLE001
            log(f'RECORD NOT WRITTEN {path}: {type(exc).__name__}: {exc}; {other}: {type(exc2).__name__}: {exc2}')
            return None


def keep_attempt(path):
    """I2: a directory the killed run left (or a failed pair the resume recomputes) is renamed whole to <name>.attempt<k> (k = 1, 2, ...), kept, never deleted (HB's way).  Returns the new path."""
    path = Path(path)
    k = 1
    while path.with_name(f'{path.name}.attempt{k}').exists():
        k += 1
    target = path.with_name(f'{path.name}.attempt{k}')
    hin.guard_write(path)
    hin.guard_write(target)
    os.rename(path, target)
    return target


# ---- the rules of the spy (C22.H4-3) --------------------------------------------------------------------------------------------------------------
def source_roots():
    """Where the pinned allowlist's sources are re-hashed: the main checkout (pqh_units._pinned_allowlist and holdout_common's admission use MAIN_REPO: data/ is its untracked link)."""
    return sorted({Path(hp.MAIN_REPO), Path(hc().MAIN_REPO)}, key=str)


def allowlist_source_files(layout):
    """B2: every source file of the allowlist, under every root its sources are verified against (source_roots), path by path as the allowlist names them (its sources' `path`; the code tree's
    configs/bayesreg/dev_subjects.json and, if different, bayesreg's own tree's, read here before the spy: configuration, not a subject file).  On the unseal branch the hold-out source is one of them."""
    from bayesreg.io import sealed
    files = []
    for tree in sorted({Path(layout.code), Path(sealed.REPO)}, key=str):
        try:
            sources = json.loads((tree/pins.ALLOWLIST_REL).read_text(encoding='utf-8')).get('sources') or []
        except (OSError, ValueError, AttributeError):
            continue                                            # the loader refuses it later, under the spy
        files += [Path(root)/s['path'] for root in source_roots() for s in sources if isinstance(s, dict) and isinstance(s.get('path'), str)]
    return sorted(set(files), key=str)


def common_files(layout):
    """The files both units may read (C22.H4-1 table: configs, splits, the unseal record and the decision log only through load_unseal, plan.md, the preregistration, the finalization record), the four
    sealed lists by path (revision 7c-7d), and the allowlist's source files where they are re-hashed (allowlist_source_files: B2, C22.H4-3 revision 7b n11)."""
    from bayesreg.io import sealed
    trees = sorted({Path(layout.code), Path(sealed.REPO)}, key=str)
    rel = [hp.PLAN_REL, hp.PREREG_REL, hp.SEG4_TABLE_REL, hp.ANCHOR['registry'], pins.ALLOWLIST_REL, 'configs/bayesreg/structure_groups_v1.json', pins.FOLDS_REL,
           'splits/bayesreg_v1/holdout_B_pairs.tsv', 'splits/bayesreg_v1/train_subjects.txt', 'splits/bayesreg_v1/unseal_holdout_B_v1.json', 'docs/decision_log.md']
    files = [t/r for t in trees for r in rel]+[layout.finalized]
    files += hfw.sealed_list_files(layout.code)
    files += allowlist_source_files(layout)
    return files


def rules_for(layout, unit):
    """The rules of one unit in its mode (C22.H4-1, C22.H5-3); the out unit adds the method's files after reading the finalization record (pqh_firewall.HoldoutRules.allow_files)."""
    h = hc()
    files = common_files(layout)
    own_lock = Path(layout.members_lock if unit == 'members' else layout.out_lock)
    files += [own_lock, own_lock.with_name(own_lock.name+'.rerun'), h.resume_lock_path(own_lock)]
    small, big = Path(layout.pq_small), Path(layout.pq_big)
    if unit == 'members':
        dirs = [small/'members', big/'members', small/'members_rerun', big/'members_rerun']
        files += [Path(layout.images)/s/'aligned_norm.nii.gz' for s in layout.subjects]           # only through the bound OasisFiles (C23.4-1); no other file of a subject
        files += [Path(layout.models_root)/name/'member.pt' for _, name, _ in hp.MEMBERS]           # the four files, not their directories (C22.Q1-3)
        if layout.mode == 'holdout':                                                                # B1: hi_core.check_decision reads the development check's record and its lock line
            h_i = hi()
            files += [Path(h_i.DEVCHECK_OUT)/h_i.DEVCHECK_RECORD, Path(h_i.DEVCHECK_LOCK)]          # (the lead's ruling (1) of 11-09; C22.H4-3's note with the next revision)
    elif getattr(layout, 'out_subdir', None):
        # SM3 revision 2 (review minor 10): a dev (iv-b), (iv-c) out unit: its own run directories (pqh_inputs.run_dirs, first run and rerun) and the members runs it reads; not (ii)'s pq/
        # root, nor the other segment's directories
        dirs = [*hin.run_dirs(layout, 'out', 'first_run'), *hin.run_dirs(layout, 'out', 'control_rerun'), small/'members', big/'members', small/'members_rerun', big/'members_rerun']
    else:
        dirs = [small, big]
    if unit != 'members':
        if layout.method_records is not None:
            files += [Path(layout.method_records)/p/n for p in layout.pairs for n in (hp.MOVE_RECORD_NAME, hp.MOVE_RESUME_NAME)]     # hb_store.move_pair's and move_resume's records
        if layout.dev_cell is not None:
            files += [Path(layout.dev_cell)/hp.TERMINAL_NAME, Path(layout.dev_cell)/hp.REVIEW_RECORD_REL]     # dev: the copy's records stand for the finalization record
        files += [Path(layout.stage)/p/n for p in layout.pairs for n in ('pair.npz', 'manifest.json')]
    dirs.append(Path(layout.batch_root))
    patterns = [(Path(h.authorization_dir(layout.pair_set)), rf'{h.AUTH_PREFIX[layout.pair_set]}{hp.TOOL}_(resume|rerun|no_resume)_[0-9]{{1,3}}\.json')]
    if layout.mode == 'holdout':
        patterns.append((Path(layout.hi_lock).parent, hin.HI_LOCK_PATTERN))
    else:
        files += [Path(layout.dev_pred_root)/s/n for s in layout.subjects for n in ('probs36.npz', 'structure_groups.npz', 'provenance.json')]
        if unit == 'members':
            files += [Path(hp.PQ_V1)/'members'/hin.MEMBERS_RECORD]                                   # smoke (i): V against the development reading's (C22.H8 (i))
            files += [Path(hp.PQ_V1)/'members'/s/io_.VAR_FILE for s in layout.subjects]
    code = Path(layout.code)
    from bayesreg.io import sealed
    return hfw.HoldoutRules(unit=unit, mode=layout.mode, watched=layout.extra.get('watched', hp.WATCHED), unwatched=layout.extra.get('unwatched', (hp.PYTHON_ENV,)+hp.RUNTIME_CACHE_DIRS),
                            files=files, dirs=dirs, patterns=patterns, forbidden_dirs=layout.forbidden_dirs, code_trees=(code, REPO, Path(sealed.REPO), Path(hp.FROZEN)),
                            result_roots=[t/d for t in (code, Path(sealed.REPO), Path(hp.MAIN_REPO)) for d in ('docs', 'runs')],
                            pinned_documents=[t/r for t in (code, Path(sealed.REPO)) for r in (hp.PLAN_REL, hp.PREREG_REL, 'docs/decision_log.md')])     # load_unseal reads bayesreg's tree's log


def preload(unit):
    """Imports before the spy (an import that reads a file would be recorded as the unit's reading), as run_pq.preload."""
    import bayesreg.eval.fields  # noqa: F401
    import bayesreg.eval.truth  # noqa: F401
    import bayesreg.io.guard  # noqa: F401
    import bayesreg.io.pair  # noqa: F401
    import bayesreg.io.sealed  # noqa: F401
    import bayesreg.io.unseal  # noqa: F401
    import bayesreg.seg.channels  # noqa: F401
    import bayesreg.seg.structure_layer  # noqa: F401
    import jd_curl.three_d.nifti  # noqa: F401
    import jd_curl.validation  # noqa: F401
    hi()
    hc()
    if unit == 'members':
        import pq_members  # noqa: F401
        import hs_seg  # noqa: F401
        import torch  # noqa: F401
        import bayesreg.seg.unet.data  # noqa: F401
        import bayesreg.seg.unet.infer  # noqa: F401
        import bayesreg.seg.unet.pins  # noqa: F401
        import bayesreg.seg.unet.train  # noqa: F401
    else:
        import fast_channels  # noqa: F401
        import mc0_core
        mc0_core.make_inputs()


def common_checks(layout, unit, deps, checks):
    """Plan, preregistration, settings, own unit, finalization record and the checkout (C22.H3-3, C22.H0-4).  Fills `checks`; Refusal otherwise.  Returns the finalization record (document, sha256)."""
    checks['plan'] = io_.check_plan(layout.code)
    problem = prereg_problem(layout.code)
    checks['preregistration'] = dict(path=hp.PREREG_REL, pinned=hp.PREREG_SHA256, problem=problem)
    if problem and layout.mode == 'holdout':
        raise Refusal('preregistration', problem)
    checks['settings_sha256'] = hp.settings_sha256()
    problem = own_unit_problem(unit_name_of(layout, unit), deps)
    if problem:
        raise Refusal('unit', problem)
    finalized = None
    if unit == 'out' or layout.mode == 'holdout':      # the dev members unit (smoke (i)) runs before the review unit of segment (ii) wrote its record
        finalized = hin.read_finalized(layout)
        checks['finalized'] = dict(path=str(layout.finalized), sha256=finalized[1], code=finalized[0].get('code'))
    else:
        checks['finalized'] = dict(path=None, sha256=None, note='dev members unit: no finalization record needed (C22.H8 (i))')
    code_check(layout, deps, finalized, checks)
    return finalized


def lock_plan(layout, unit, lock, resume_rel, checks, complete):
    """Which lock this run takes and where it writes: the first run's, a resume's (holdout_common.load_resume; D123 (b)), or a control rerun's (control_rerun).  Refusal otherwise."""
    h = hc()
    if layout.mode == 'holdout':
        try:
            checks['rerun_authorization'] = h.rerun_authorization(Path(layout.pq_small), tool=hp.TOOL, arm=hp.PQ_ARM, pair=unit, pair_set=layout.pair_set)
        except h.Refused as exc:
            raise Refusal(exc.stage, str(exc)) from None
    rerun = control_rerun(layout, unit, lock)
    checks['control_rerun'] = rerun
    if rerun is not None:
        if resume_rel is not None:
            raise Refusal('rerun', 'a control rerun and a resume at once')
        return dict(kind='control_rerun', lock=Path(rerun['lock']), authorization=rerun)
    resume = None
    if resume_rel is not None:
        if layout.mode != 'holdout':
            raise Refusal('resume', 'the dev mode never resumes (C22.H8)')
        small, _ = hin.run_dirs(layout, unit, 'first_run')
        previous = small/(hin.MEMBERS_RECORD if unit == 'members' else OUT_RECORD)
        try:
            resume = h.load_resume(resume_rel, tool=hp.TOOL, arm=hp.PQ_ARM, pair=unit, pair_set=layout.pair_set, lock=lock, previous_record=previous, complete=complete)
        except h.Refused as exc:
            raise Refusal(exc.stage, str(exc)) from None
        checks['resume'] = dict(path=resume.path, sha256=resume.sha256, previous_record=resume.previous_record)
    try:
        h.check_once(lock, resume=resume)
    except h.Refused as exc:
        raise Refusal(exc.stage, str(exc)) from None
    if resume is not None:
        return dict(kind='resume', lock=Path(resume.lock_path), authorization=dict(path=resume.path, sha256=resume.sha256), recompute=resume_items(resume.document))
    return dict(kind='first_run', lock=Path(lock), authorization=None)


RECOMPUTE = re.compile(r'^recompute:[ \t]*(.*?)[ \t]*$', re.MULTILINE)


def resume_items(document):
    """I2 (the lead's ruling (3) of 11-09: "只重算补跑授权里列出的对"): what a resume computes again, named in its authorization.  holdout_common's resume record has a fixed set of keys
    (AUTH_KEYS['resume']), so the list is one line of its `reason`: `recompute: <item>, <item>` -- pairs (<fixed>__<moving>) for the out unit, subjects for the members unit -- or `recompute: none`.
    The pairs or subjects the killed run left half written must be listed (keep_attempt renames them first); a pq_failed pair is computed again only when listed; a written one never.
    Refusal('resume') without exactly one such line."""
    lines = RECOMPUTE.findall(str(document.get('reason') or ''))
    if len(lines) != 1:
        raise Refusal('resume', f"the resume record's reason must hold exactly one line 'recompute: <item>, ...' or 'recompute: none' (I2: only the listed pairs or subjects are computed again); "
                                f'found {len(lines)}')
    text = lines[0].strip()
    if text == 'none':
        return []
    items = [t.strip() for t in text.split(',')]
    if not text or any(not t for t in items):
        raise Refusal('resume', f"the resume record's line 'recompute: {text}' names an empty item: write the pairs or subjects, or 'recompute: none' (review NM7)")
    return items


def disk_and_machine(layout, unit, deps, checks, unit_name=None):
    """The disks and the machine gate (gate_of) of this run; `unit_name` the unit the process runs in (default unit_name_of(layout, unit); the P3 replay passes its own, SM3)."""
    h = hc()
    try:
        checks['disk'] = h.disk_guard(need_d_gib=hp.DISK_NEED_GIB[unit]['d'] if layout.mode == 'holdout' else 0., need_home_gib=hp.DISK_NEED_GIB[unit]['home'], usage=deps.disk_usage)
        checks['machine'] = gate_of(unit, layout.mode, deps, unit_name=unit_name_of(layout, unit) if unit_name is None else unit_name)
    except (h.InfrastructureFailure, h.Refused) as exc:
        raise Refusal(getattr(exc, 'kind', None) or getattr(exc, 'stage', 'machine'), str(exc), cls='infrastructure') from None


def take_lock(plan, line):
    h = hc()
    hin.guard_write(plan['lock'])
    try:
        h.take_lock(plan['lock'], h.lock_line(**line))
    except h.Refused as exc:
        raise Refusal(exc.stage, str(exc)) from None
    return dict(path=str(plan['lock']), sha256=io_.sha_file(plan['lock']), kind=plan['kind'], authorization=plan['authorization'])


def host():
    import platform
    import resource
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, pid=os.getpid())


# ---- the members unit -------------------------------------------------------------------------------------------------------------------------------
def members_admission(layout, deps, checks, subjects=None, rules=None):
    """The hold-out: HI's steps 2, 5-7 with the same unseal record (hi_core.load_unseal, bind -- the allowlist and the OasisFiles bound --, admit, check_registry; D24 through pq_members.plan_for ->
    hs_seg.uf_plan; the four member files re-hashed), HI's predictions complete, the decision of HI's development check; the dev mode: the pinned allowlist, the two smoke subjects, the registered
    predictions.  Returns (subjects, allowlist, files, registry, plans, references, decision)."""
    import pq_members as members
    from bayesreg.io.pair import load_segmenter_registry
    h_i = hi()
    code = Path(layout.code)
    unseal = None
    if layout.mode == 'holdout':                    # BZ's unseal record first among the subjects' inputs (HI's step 2; review m9: before it the unit read only the finalization record, the
                                                    # authorizations, its stop and resume records and, for a resume, its own member.json -- no file of a subject)
        from bayesreg.io import sealed
        unseal = h_i.load_unseal(Path(sealed.REPO)/h_i.UNSEAL_REL, anchor=hp.ANCHOR)
        if sorted(unseal.subjects) != list(layout.subjects):
            raise Refusal('admission', f'the unseal record holds {len(unseal.subjects)} subjects, the pair list {len(layout.subjects)}: not the same subjects')
    allowlist, allowlist_record = deps.allowlist() if deps.allowlist is not None else _pinned_allowlist(code)
    files = deps.files() if deps.files is not None else _oasis_files(layout.images)
    registry = load_segmenter_registry(code/hp.ANCHOR['registry'])
    checks['registry'] = h_i.check_registry(registry, hp.ANCHOR)
    available = members.member_files(layout.models_root)
    checks['member_files'] = sorted(available)
    if layout.mode == 'holdout':
        allowlist, files = h_i.bind(unseal, allowlist, files)
        sealed_lists, training = deps.sealed_training() if deps.sealed_training is not None else _sealed_training(code)
        subjects = list(layout.subjects) if subjects is None else list(subjects)
        checks['unseal'] = unseal.record()
        checks['admission'] = {s: h_i.admit(s, unseal=unseal, allowlist=allowlist, sealed=sealed_lists, training=training) for s in subjects}
        plans = {s: members.plan_for(registry, s, allowlist, available) for s in subjects}
        if rules is not None:
            rules.allow_files(hin.hi_files(layout.hi_lock, layout.subjects), "HI's runs of its lock: the records and the predictions of the unsealed subjects (Hm5)")
        references = hin.hi_predictions(layout.hi_lock, subjects)
        checks['hi'] = dict(lock=str(layout.hi_lock), runs=sorted({r['run'] for r in references.values()}))
        decision = deps.decision() if deps.decision is not None else h_i.check_decision(h_i.DEVCHECK_DECISION)
        checks['decision'] = decision
    else:
        subjects = list(layout.subjects) if subjects is None else list(subjects)
        plans = {s: members.plan_for(registry, s, allowlist, available) for s in subjects}
        references = {s: dict(pred_dir=Path(layout.dev_pred_root)/s, want=layout.v4[s]) for s in subjects}
        decision = None
    checks['allowlist'] = {k: v for k, v in (allowlist_record or {}).items() if k != 'code'}
    checks['plans'] = {s: dict(policy=p[2], members=[sha for _, sha, _ in p[0]]) for s, p in plans.items()}
    return subjects, allowlist, files, registry, plans, references, decision


def _pinned_allowlist(code):
    from bayesreg.seg.unet.pins import load_pinned_allowlist
    return load_pinned_allowlist(Path(code)/pins.ALLOWLIST_REL, source_root=hp.MAIN_REPO)


def _oasis_files(root):
    from bayesreg.seg.unet.data import OasisFiles
    return OasisFiles(root=Path(root))


def _sealed_training(code):
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.seg.unet.data import load_folds
    return load_sealed_lists(Path(code)), sorted(load_folds(Path(code)/pins.FOLDS_REL).training_subjects)


def reference_prediction(layout, subject, reference):
    """The probs36 Hm5 compares with: hold-out, HI's (the sha256 of HI's subject record, named by HI's provenance); dev, the registered prediction with plan.md's sha256 (pq_members.registered)."""
    d = Path(reference['pred_dir'])
    if layout.mode == 'holdout':
        want = reference['output_sha256']['probs36.npz']
        if io_.sha_file(d/'probs36.npz') != want:
            raise PQRefused('hm5', f"{d}/probs36.npz is not the sha256 of HI's subject record")
        prov = io_.read_json(d/'provenance.json')
        if (prov.get('output_sha256') or {}).get('probs36.npz') != want:
            raise PQRefused('hm5', f'{d}/provenance.json does not name this probs36.npz')
        with np.load(d/'probs36.npz', allow_pickle=False) as z:
            stored = np.array(z['p'])
        return stored, prov
    import pq_members as members
    return members.registered(layout.dev_pred_root, subject, reference['want'])


def members_complete(layout):
    def complete():
        try:
            hin.members_view(layout, layout.subjects)
            return True
        except PQRefused:
            return False
    return complete


def members_carried(small, big):
    """A resume (D123 (b)): the subjects the resumed run finished but whose rows are not in its record (it was killed before writing it): member.json with Hm5 passed and var.npz with its sha256.
    They are carried into the resume's record; the first record's own rows stay there (pqh_inputs.members_view takes both)."""
    first = small/hin.MEMBERS_RECORD
    in_first = set((json.loads(first.read_text(encoding='utf-8')).get('subjects') or {})) if first.is_file() else set()
    out = {}
    if not big.is_dir():
        return out
    for d in sorted(p for p in big.iterdir() if p.is_dir() and '.attempt' not in p.name):
        record = d/'member.json'
        if d.name in in_first or not record.is_file():
            continue
        row = json.loads(record.read_text(encoding='utf-8'))
        if row.get('m5') is True and (d/io_.VAR_FILE).is_file() and io_.sha_file(d/io_.VAR_FILE) == row.get('var_sha256'):
            out[d.name] = dict(m5=True, agreement=row.get('agreement'), n_differ=row.get('n_differ'), max_abs_difference_to_registered=row.get('max_abs_difference_to_registered'),
                               var_sha256=row.get('var_sha256'), image_sha256=row.get('image_sha256'), carried_from=str(record), carried_sha256=io_.sha_file(record))
    return out


def members_resume_plan(layout, small, big, recompute):
    """I2 for the members unit: (todo, half): the subjects a resume infers -- every subject of the layout neither in the first record nor carried -- and among them the ones whose directory the
    killed run left half written (renamed whole by keep_attempt after the lock).  A half-written subject must be listed in the resume record (resume_items); a listed subject that is done is refused."""
    carried = members_carried(small, big)
    first = small/hin.MEMBERS_RECORD
    done = set(carried) | (set((json.loads(first.read_text(encoding='utf-8')).get('subjects') or {})) if first.is_file() else set())
    todo = [s for s in layout.subjects if s not in done]
    half = [s for s in todo if (big/s).exists()]
    problems = [f'{s} is done: never inferred again' for s in recompute if s in done]
    problems += [f'{s} is not a subject of this unit' for s in recompute if s not in layout.subjects]
    problems += [f'{s}: the killed run left {big/s}; the resume record does not list it (recompute: ...)' for s in half if s not in recompute]
    if problems:
        raise Refusal('resume', f'{problems[:10]}')
    return carried, todo, half


def members_unit(layout, *, device='cuda', deps=None, resume_rel=None):
    """The members unit (module docstring).  Returns the exit code.  After the lock nothing escapes the unit: an exception (SystemExit included) is recorded, the opened files and the record are
    written (write_record; I2)."""
    import pq_members as members
    deps = Deps() if deps is None else deps
    log = deps.log
    tick = perf_counter()
    preload('members')
    checks = {}
    problem = hfw.sealed_cache_problem('members')
    if problem:
        return refuse(layout, 'members', None, checks, Refusal('hm6', problem), log)
    rules = rules_for(layout, 'members')
    spy = hfw.spy(rules)
    spy.install()
    plan, small, resume, carried, half, attempts, lock, dev = None, None, False, {}, [], [], None, None
    rows, error, stage, peak, reference, subjects = {}, None, None, None, None, []
    try:
        try:
            checks['probe_start'] = probe('start', deps)
            common_checks(layout, 'members', deps, checks)
            plan = lock_plan(layout, 'members', layout.members_lock, resume_rel, checks, members_complete(layout))
            small, big = hin.run_dirs(layout, 'members', plan['kind'])
            resume = plan['kind'] == 'resume'
            todo = None
            if resume:
                carried, todo, half = members_resume_plan(layout, small, big, plan['recompute'])
            subjects, allowlist, files, registry, plans, references, decision = members_admission(layout, deps, checks, todo, rules)
            if not resume and small.exists() and [q for q in small.iterdir() if not q.name.startswith('pq_refused')]:
                raise Refusal('once', f'{small} exists: a run of the members unit writes a new directory (only stop records before a lock may be there, review NM4)')
            existing = [s for s in subjects if (big/s).exists() and s not in half]
            if existing:
                raise Refusal('once', f'subject directories exist already under {big}: {existing[:5]} (never overwritten; a subject is never inferred twice)')
            disk_and_machine(layout, 'members', deps, checks)
            dev = deps.device(device, hp.GATES['members']['gpu_free_min_gib']) if deps.device is not None else hi().prepare_device(device, free_min_gib=hp.GATES['members']['gpu_free_min_gib'])
            if layout.mode == 'holdout':
                checks['environment'] = deps.environment(dev, decision) if deps.environment is not None else hi().check_environment(dev, decision)
            else:
                checks['environment'] = deps.environment(dev, None) if deps.environment is not None else hi().environment_of(dev)
            lock = take_lock(plan, dict(tool=hp.TOOL, unit='members', mode=layout.mode, kind=plan['kind'], subjects=len(subjects), records=str(small), settings_sha256=checks['settings_sha256'],
                                        code_commit=checks['code'].get('commit'), code_head=checks['code'].get('head'), finalized_sha256=checks['finalized']['sha256'],
                                        unseal_sha256=(checks.get('unseal') or {}).get('sha256'),
                                        recompute=plan.get('recompute')))
        except (Exception, SystemExit) as exc:      # noqa: BLE001 - every exception before the lock is a refusal record (I1) or, for a control, pq_refused.json (m3)
            return refuse(layout, 'members', spy, checks, as_refusal(exc), log, plan)
        try:
            for s in half:                                       # I2: the half-written subjects of the killed run, renamed whole and kept
                attempts.append(dict(subject=s, src=str(big/s), dst=str(keep_attempt(big/s))))
            make_dir(small)
            write_record(small/('members_started.json' if not resume else 'members_started.resume.json'),
                         dict(schema=hp.SCHEMA.format('members_started'), lock=lock, checks=checks, subjects=subjects, carried=sorted(carried), attempts=attempts, utc=utc()), log)
            for subject in subjects:
                t0 = perf_counter()
                plan_s, models, policy = plans[subject]
                stored, prov = reference_prediction(layout, subject, references[subject])
                mean, V, meta, norm, member_rows = members.mean_and_variance(subject, files=files, models=models, plan=plan_s, device=dev)
                row = members.subject_record(subject, mean=mean, V=V, meta=meta, norm=norm, members=member_rows, stored=stored, prov=prov, out_dir=hin.guard_write(big/subject),
                                             seconds=perf_counter()-t0)
                rows[subject] = dict(row, policy=policy)
                log(f"{subject}: agreement {row['agreement']:.6f}, Hm5 {row['m5']}, {row['seconds']:.1f} s")
                del mean, V, stored
                if not row['m5']:
                    stage, error = 'hm5', f"{subject}: Hm5 not passed (agreement {row['agreement']}, image of the provenance {row['image_matches_provenance']}): the unit stops (C22.H5-5)"
                    break
        except hfw.Breach as exc:
            stage, error = 'hm6', str(exc)
        except PQRefused as exc:
            stage, error = (exc.stage if exc.stage in CONTROL_STAGES else 'hm5'), str(exc)
        except (Exception, SystemExit) as exc:      # noqa: BLE001 - kept, not rerun; an infrastructure failure is resumed only through a committed resume record (D123 (b))
            stage, error = 'error', f'{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}'
        if layout.mode == 'dev':
            try:
                reference = dev_reference(rows, big)
            except Exception as exc:                # noqa: BLE001 - recorded; the smoke's reading judges it
                reference = dict(error=f'{type(exc).__name__}: {exc}', equal=False)
        try:
            import torch
            if getattr(dev, 'type', None) == 'cuda':
                peak = dict(device_peak_allocated_gib=torch.cuda.max_memory_allocated(dev)/2**30, device_peak_reserved_gib=torch.cuda.max_memory_reserved(dev)/2**30)
        except Exception:                           # noqa: BLE001
            peak = None
    finally:
        spy.remove()
    return members_records(layout, spy, checks, plan, small, resume, rows, subjects, carried, attempts, error, stage, peak, reference, dev, lock, deps, tick)


def members_records(layout, spy, checks, plan, small, resume, rows, subjects, carried, attempts, error, stage, peak, reference, dev, lock, deps, tick):
    """The members unit's records after the lock, each written (write_record: beside its name if need be, never raised; I2)."""
    import pq_members as members
    log = deps.log
    opened = spy.record()
    write_record(small/('opened_files_members.json' if not resume else 'opened_files_members.resume.json'),
                 dict(opened, mode=layout.mode, unwatched='the files the subprocesses (git, systemctl, nvidia-smi, dmesg) open are not watched'), log)
    try:
        document = members.members_document(rows, code=checks.get('code'), environment=checks.get('environment'), spy_record=opened, device=dev, seconds=perf_counter()-tick)
    except Exception as exc:                        # noqa: BLE001 - the record is still written
        document = dict(subjects=dict(rows), spy=dict(m6=opened['m6'], n_files=opened.get('n_files'), breaches=opened.get('breaches')), document_error=f'{type(exc).__name__}: {exc}')
        stage, error = stage or 'error', error or document['document_error']
    document['subjects'].update(carried)
    passed = error is None and bool(opened['m6']) and sorted(rows) == sorted(subjects) and all(r['m5'] for r in rows.values())
    if not opened['m6'] and stage is None:
        stage, error = 'hm6', f"the spy recorded breaches {opened['breaches']}"
    document.update(schema=hp.SCHEMA.format('members'), contract=hp.CONTRACT, mode=layout.mode, kind=plan['kind'], lock=lock, settings_sha256=checks['settings_sha256'], error=error, stage=stage,
                    m5_passed=passed, carried=sorted(carried), attempts=attempts, recompute=plan.get('recompute'), checks={k: v for k, v in checks.items() if k != 'admission'},
                    admission=checks.get('admission'), device_peak=peak, host=host(), probe_end=probe('end', deps), utc=utc())
    if layout.mode == 'dev':
        document['dev_reference'] = reference
    if stage in CONTROL_STAGES:
        write_record(stop_record_path(small), dict(schema=hp.SCHEMA.format('stopped'), unit='members', stage=stage, error=error, utc=utc(),
                                                   rule='C22.H4-2 last item, C22.H5-5: the unit stops; BZ decides ((i) a rerun through a committed authorization, (ii) F4 described only)'), log)
    elif stage is not None:
        write_record(small/('members_stopped.json' if not resume else 'members_stopped.resume.json'),
                     dict(schema=hp.SCHEMA.format('members_stopped'), stage=stage, error=error, done=sorted(rows), utc=utc(), note='kept, not rerun; a resume needs a committed resume record (D123 (b))'), log)
    write_record(small/(hin.MEMBERS_RECORD if not resume else hin.MEMBERS_RESUME_RECORD), document, log)
    log(f"members: Hm5 {document['m5_passed']}, spy m6 {opened['m6']}, error {error}")
    return EXIT['ok'] if passed else EXIT['stopped']


def dev_reference(rows, big):
    """C22.H8 (i): V against the development reading's (pq_v1/members), per subject; recorded here, judged by the smoke's reading.  The reference var.npz is bound to the sha256 its record holds
    (members_record.json var_sha256); then the arrays are compared bit for bit (dtype, shape, bytes): two np.savez files of the same array differ in the zip entry's time stamp, so the file sha256 of
    this run's var.npz is not compared."""
    path = Path(hp.PQ_V1)/'members'/hin.MEMBERS_RECORD
    if not path.is_file():
        return dict(path=str(path), present=False, equal=False)
    recorded = json.loads(path.read_text(encoding='utf-8')).get('subjects') or {}
    out = {}
    for s in rows:
        ref = Path(hp.PQ_V1)/'members'/s/io_.VAR_FILE
        bound = ref.is_file() and io_.sha_file(ref) == (recorded.get(s) or {}).get('var_sha256')
        row = dict(reference=str(ref), reference_bound=bound, equal=False)
        if bound and (Path(big)/s/io_.VAR_FILE).is_file():
            with np.load(ref, allow_pickle=False) as z:
                theirs = np.array(z['V'])
            with np.load(Path(big)/s/io_.VAR_FILE, allow_pickle=False) as z:
                mine = np.array(z['V'])
            row.update(equal=bool(mine.dtype == theirs.dtype and mine.shape == theirs.shape and mine.tobytes() == theirs.tobytes()), mine_array_sha256=io_.sha_array(mine),
                       reference_array_sha256=io_.sha_array(theirs))
        out[s] = row
    return dict(path=str(path), sha256=io_.sha_file(path), subjects=out, equal=bool(out) and all(r['equal'] for r in out.values()),
                rule='C22.H8 (i): the reference var.npz bound to pq_v1/members/members_record.json, then V bit for bit')


# ---- the PQ unit --------------------------------------------------------------------------------------------------------------------------------------
def admit_pairs(layout, checks):
    """Every pair through SG: holdout_common.admit_holdout_pair (load_unseal, bind_unseal, check_subject C23.4-2 in full, the pinned pair row in its direction) or admit_dev_pair.  {pair: allowlist}."""
    h = hc()
    out, records = {}, {}
    for pair in layout.pairs:
        fixed, moving = h.split_pair(pair)
        record, allowlist = h.admit_holdout_pair(fixed, moving) if layout.mode == 'holdout' else h.admit_dev_pair(fixed, moving)
        out[pair], records[pair] = allowlist, record
    checks['admission'] = records
    return out


class PairInputs:
    """The inputs of one pair, read again in each pass (pq_io.check_pair / compute_pair's way)."""

    def __init__(self, layout, pair, entry, paths, allowlist, deps, hi_rows):
        self.layout, self.pair, self.entry, self.paths, self.allowlist, self.deps, self.hi_rows = layout, pair, entry, paths, allowlist, deps, hi_rows

    def open(self):
        """load_pair with the bound allowlist, Hm1, Hm4's first item: (pin, v, field record, disp record)."""
        pin = hin.load_pair_checked(self.layout, self.pair, self.allowlist)
        v, field_record = hin.bind_output(self.layout, self.pair, self.entry, self.paths, pin.grid, self.deps.taper)
        line = json.loads(self.paths[hp.HE_FILES['line']].read_text(encoding='utf-8'))
        _, disp = hin.check_disp(v, pin, self.paths, line)
        return pin, v, field_record, disp

    def reader(self, pin, side, labels):
        """A () -> (P, L, record) reader of one side (Hm2, Hm3): the structure layer of the root HI wrote this subject into, with the bound allowlist."""
        subject = pin.fixed_subject if side == 'fixed' else pin.moving_subject
        if self.layout.mode == 'holdout':
            row = self.hi_rows[subject]
            root, want = Path(row['pred_dir']).parent, dict(probs36=row['output_sha256']['probs36.npz'], structure_groups=row['output_sha256']['structure_groups.npz'])
        else:
            root, want = Path(self.layout.dev_pred_root), self.layout.v4[subject]._asdict()
        layer = hin.structure_layer(self.layout.code, pin, root, self.allowlist)
        self.layer_record = dict(registry=dict(path=layer.registry.path, sha256=layer.registry.sha256), preflight=layer.preflight)     # D24's pair-level check, for pq.json
        return lambda: hin.read_side(layer, pin, side, want=want, labels_sha256=labels)


def pair_written(small, pair):
    return (small/pair/PQ_JSON).is_file() or (small/pair/ABSENT).is_file()


def out_resume_plan(layout, small, big, recompute):
    """I2 for the out unit: (todo, again): the pairs a resume writes -- the pairs without pq.json or pq_absent.json and without pq_failed.json, plus the pq_failed pairs the resume record lists --
    and the pairs whose directories are renamed whole first (keep_attempt): the killed run's half-written pairs (they must be listed) and the listed pq_failed pairs.  A written pair is never
    computed again.  Refusal('resume') otherwise."""
    problems = [f'{p} is not a pair of this unit' for p in recompute if p not in layout.pairs]
    todo, again = [], []
    for p in layout.pairs:
        there = [d for d in (small/p, big/p) if d.exists()]
        if pair_written(small, p):
            if p in recompute:
                problems.append(f'{p} is written (pq.json or pq_absent.json): never computed again')
            continue
        failed = (small/p/FAILED).is_file()
        if failed and p not in recompute:
            continue                                            # C22.H6-3: pq_failed kept, not rerun unless the resume record names it
        if there and p not in recompute:
            problems.append(f'{p}: the killed run left {[str(d) for d in there]}; the resume record does not list it (recompute: ...)')
            continue
        todo.append(p)
        if there:
            again.append(p)
    if problems:
        raise Refusal('resume', f'{problems[:10]}')
    return todo, again


def out_unit(layout, *, deps=None, resume_rel=None):
    """The PQ unit (module docstring).  Returns the exit code.  After the lock nothing escapes the unit: an exception (SystemExit included) is recorded, the opened files and the record are
    written (write_record; I2)."""
    deps = Deps() if deps is None else deps
    log = deps.log
    tick = perf_counter()
    preload('out')
    checks = {}
    problem = hfw.sealed_cache_problem('out')
    if problem:
        return refuse(layout, 'out', None, checks, Refusal('hm6', problem), log)
    rules = rules_for(layout, 'out')
    spy = hfw.spy(rules)
    spy.install()
    plan, small, big, resume, lock, again, attempts = None, None, None, False, None, [], []
    controls = dict(hm1=None, hm2=None, hm3=None, hm4=None)
    error, stage, documents, failed, absent, pass1, todo, finalized, unwritten = None, None, {}, {}, [], {}, [], None, {}
    try:
        try:
            checks['probe_start'] = probe('start', deps)
            finalized, _ = common_checks(layout, 'out', deps, checks)
            if layout.ev_master_lock is not None and Path(layout.ev_master_lock).exists():
                raise Refusal('order', f'{layout.ev_master_lock} exists: the evaluation has started; PQ is not computed after it (C22.H3-3)')
            pairs_out = [p for p in layout.pairs if finalized['pairs'][p]['line_status'] in hp.LINE_STATUSES_WITH_OUTPUT]
            first_small, _ = hin.run_dirs(layout, 'out', 'first_run')
            plan = lock_plan(layout, 'out', layout.out_lock, resume_rel, checks, lambda: all(pair_written(first_small, p) for p in layout.pairs))
            small, big = hin.run_dirs(layout, 'out', plan['kind'])
            resume = plan['kind'] == 'resume'
            if resume:
                todo, again = out_resume_plan(layout, small, big, plan['recompute'])
            else:
                todo = list(layout.pairs)
            allowlists = admit_pairs(layout, checks)            # SG first (C22.H4-4): without the unseal object nothing of a subject is looked at
            subjects = sorted({s for p in pairs_out for s in p.split('__')})
            members_rows, members_records = hin.members_view(layout, layout.subjects if layout.mode == 'holdout' else subjects)     # C22.H3-3: the members' record complete (100)
            checks['members'] = members_records
            hi_rows = None
            if layout.mode == 'holdout':
                rules.allow_files(hin.hi_files(layout.hi_lock, layout.subjects), "HI's runs of its lock: the records and the predictions of the unsealed subjects (Hm2)")
                hi_rows = hin.hi_predictions(layout.hi_lock, layout.subjects)                       # C22.H3-3: HI's inference complete over its runs (100)
                checks['hi'] = {s: dict(run=r['run'], record_sha256=r['record_sha256']) for s, r in hi_rows.items()}
            inputs, moves = {}, {}
            for p in pairs_out:
                paths, moved = hin.method_paths(layout, p, finalized['pairs'][p])
                rules.allow_files(paths.values(), f"the method's files of {p} named by the finalization record (C22.H4-1)")
                inputs[p], moves[p] = PairInputs(layout, p, finalized['pairs'][p], paths, allowlists[p], deps, hi_rows), moved
            checks['moves'] = moves
            if not resume:
                taken = [p for p in todo if (small/p).exists() or (big/p).exists()]
                if taken:
                    raise Refusal('once', f'pair directories exist already under {small} or {big}: {taken[:5]} (never overwritten)')
            disk_and_machine(layout, 'out', deps, checks)
            if deps.taper is None:
                import mc0_core
                deps.taper = mc0_core.frozen_api().boundary_taper           # SystemExit when jd_curl is not the frozen tree's: a refusal (m7)
                checks['frozen'] = dict(root=str(mc0_core.FROZEN), commit=mc0_core.FROZEN_COMMIT, preopt_sha256=mc0_core.FROZEN_PREOPT_SHA256, taper='preopt.boundary_taper (C22.Q1-2, L = 4)')
            else:
                checks['frozen'] = dict(taper=getattr(deps.taper, '__qualname__', str(deps.taper)), note='not the frozen tree (a test)')
            lookup = io_.load_lookup(layout.code, hp.SEG4_TABLE_SHA256)
            lock = take_lock(plan, dict(tool=hp.TOOL, unit='out', mode=layout.mode, kind=plan['kind'], pairs=len(todo), records=str(small), settings_sha256=checks['settings_sha256'],
                                        code_commit=checks['code'].get('commit'), code_head=checks['code'].get('head'), finalized_sha256=checks['finalized']['sha256'],
                                        recompute=plan.get('recompute')))
        except (Exception, SystemExit) as exc:      # noqa: BLE001 - every exception before the lock is a refusal record (I1) or, for a control, pq_refused.json (m3)
            return refuse(layout, 'out', spy, checks, as_refusal(exc), log, plan)
        try:
            for p in again:                                      # I2: the killed run's half-written pairs and the listed pq_failed pairs, renamed whole and kept
                attempts += [dict(pair=p, src=str(d), dst=str(keep_attempt(d))) for d in (small/p, big/p) if d.exists()]
            for p in todo:                                       # pass 1: Hm1-Hm4 on every pair with an output; nothing computed or written (C22.H4-2)
                if p not in inputs:
                    continue
                pin, v, field_record, disp = inputs[p].open()
                labels = field_record['labels_sha256']
                inputs[p].reader(pin, 'fixed', labels)()
                top = hin.check_hm4_top(inputs[p].reader(pin, 'moving', labels), v, pin)
                pass1[p] = dict(state_hash=field_record['state_hash'], line_status=field_record['line_status'], d_sha256=disp['d_sha256'], labels_checked=labels is not None, n_outside=top['n_outside'])
                log(f'{p}: Hm1-Hm4 passed')
            controls.update(hm1=True, hm2=True, hm3=True, hm4=True)
        except PQRefused as exc:
            stage, error = (exc.stage if exc.stage in CONTROL_STAGES else 'hm1'), str(exc)
            controls[stage] = False
        except hfw.Breach as exc:
            stage, error = 'hm6', str(exc)
        except (Exception, SystemExit) as exc:      # noqa: BLE001
            stage, error = 'error', f'{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}'
        if error is None:
            for p in todo:                                       # pass 2: compute and write, pair after pair
                try:
                    if p not in inputs:
                        write_once(small/p/ABSENT, dict(schema=hp.SCHEMA.format('absent'), pair=p, line_status=finalized['pairs'][p]['line_status'], finalized_sha256=checks['finalized']['sha256'],
                                                        reason='the method has no output for this pair (C22.H6-3)', utc=utc()))
                        absent.append(p)
                        continue
                    documents[p] = compute_pair(layout, p, inputs[p], members_rows, lookup, checks, lock, small, big)
                    log(f"{p}: S35^ {documents[p]['numbers']['S35_hat']} (the method's own number, never in a Dice table)")
                except hfw.Breach as exc:
                    stage, error = 'hm6', str(exc)
                    break
                except PQRefused as exc:
                    stage, error = (exc.stage if exc.stage in CONTROL_STAGES else 'hm1'), str(exc)
                    controls[stage] = False
                    break
                except hc().RealRootInTest:
                    raise
                except (Exception, SystemExit) as exc:  # noqa: BLE001 - pq_failed.json, kept, not rerun; the next pair goes on
                    failed[p] = f'{type(exc).__name__}: {exc}'
                    try:
                        write_once(small/p/FAILED, dict(schema=hp.SCHEMA.format('failed'), pair=p, error=failed[p], traceback=traceback.format_exc()[-3000:], utc=utc(),
                                                        rule='C22.H6-3: kept, not rerun automatically; an infrastructure failure only through a committed resume record, at most once'))
                    except Exception as exc2:       # noqa: BLE001 - I2: never out of the unit; the record says it
                        unwritten[p] = f'{type(exc2).__name__}: {exc2}'
    finally:
        spy.remove()
    return out_records(layout, spy, checks, plan, small, big, resume, lock, controls, error, stage, documents, failed, absent, pass1, todo, attempts, unwritten, deps, tick)


def out_records(layout, spy, checks, plan, small, big, resume, lock, controls, error, stage, documents, failed, absent, pass1, todo, attempts, unwritten, deps, tick):
    """The PQ unit's records after the lock, each written (write_record; I2), then the sha256 list when the run is complete (C22.H3-4)."""
    log = deps.log
    opened = spy.record()
    write_record(small/('opened_files_out.json' if not resume else 'opened_files_out.resume.json'), dict(opened, mode=layout.mode, unwatched='subprocesses (git, systemctl, dmesg) are not watched'), log)
    controls['hm5'] = True
    controls['hm6'] = bool(opened['m6']) and stage != 'hm6'
    if not opened['m6'] and stage is None:
        stage, error = 'hm6', f"the spy recorded breaches {opened['breaches']}"
    if stage in CONTROL_STAGES:
        write_record(stop_record_path(small), dict(schema=hp.SCHEMA.format('stopped'), unit='out', stage=stage, error=error, utc=utc(), no_output=not documents,
                                                   written_before_the_stop=sorted(documents),
                                                   rule='C22.H4-2: any Hm1-Hm6 failure stops the unit; no PQ output counts (review m10: the pairs written before the stop stay on disk, out of any list); '
                                                        'BZ decides before the truth ((i) a rerun through a committed authorization, (ii) F4 described only)'), log)
    written = all(p in documents or p in failed or p in absent for p in todo)
    complete = stage is None and error is None and written and not unwritten and all(controls[k] is True for k in ('hm1', 'hm2', 'hm3', 'hm4', 'hm5', 'hm6'))
    record = dict(schema=hp.SCHEMA.format('out'), contract=hp.CONTRACT, mode=layout.mode, kind=plan['kind'], lock=lock, settings_sha256=checks['settings_sha256'], controls=controls, stage=stage,
                  error=error, pass1=pass1, written=sorted(documents), failed=failed, failed_record_not_written=unwritten, absent=absent, todo=todo, recompute=plan.get('recompute'), attempts=attempts,
                  complete=complete, checks={k: v for k, v in checks.items() if k != 'admission'}, admission=checks.get('admission'), host=host(), seconds=perf_counter()-tick,
                  probe_end=probe('end', deps), utc=utc(),
                  rule="C22.H3-4: written = Hm1-Hm6 passed and every pair with an output has pq.json or pq_failed.json; then the lead's release (one commit with the list's sha256)")
    write_record(small/(OUT_RECORD if not resume else OUT_RESUME_RECORD), record, log)
    if not complete:
        log(f'out: stopped at {stage}: {error}; no sha256 list (C22.H3-4: BZ decides before the truth)')
        return EXIT['stopped']
    try:
        manifest = write_manifest(layout, small, big, resume)
    except Exception as exc:                        # noqa: BLE001 - I2: recorded, never out of the unit
        write_record(small/'pq_manifest_not_written.json', dict(error=f'{type(exc).__name__}: {exc}', utc=utc()), log)
        return EXIT['stopped']
    sha = io_.sha_file(manifest)
    log(f'out: done; the list {manifest}, sha256 {sha}')
    log('the lead: read the opened_files of both units; then commit in ONE commit, before the input lock of step 8 (C22.H4-5):')
    log(f'  {RELEASE_SHA_REL} :  {sha}  {manifest}')
    log(f'  {RELEASE_NOTE_REL}              :  放行：{sha}')
    return EXIT['ok']


def compute_pair(layout, pair, inputs, members_rows, lookup, checks, lock, small, big):
    """Pass 2 of one pair: the inputs again (every control again), the arrays (pqh_inputs.pq_arrays), the three files once (C22.H6-1)."""
    tick = perf_counter()
    pin, v, field_record, disp = inputs.open()
    labels = field_record['labels_sha256']
    shape = tuple(pin.grid.shape)
    V_F = hin.read_variance(members_rows, pin.fixed_subject, shape)
    V_M = hin.read_variance(members_rows, pin.moving_subject, shape)
    pq, base, extra = hin.pq_arrays(v=v, pin=pin, read_fixed=inputs.reader(pin, 'fixed', labels), read_moving=inputs.reader(pin, 'moving', labels), V_F=V_F, V_M=V_M, lookup=lookup)
    if extra['d_sha256'] != disp['d_sha256']:
        raise PQRefused('hm4', f"{pair}: d of the computation {extra['d_sha256']} is not Hm4's {disp['d_sha256']}")
    save_npz_once(big/pair/PQ_NPZ, **pq)
    save_npz_once(big/pair/BASELINES, **base)
    files = {PQ_NPZ: dict(path=str(big/pair/PQ_NPZ), keys=['q', 'q4'], dtype='float32', sha256=io_.sha_file(big/pair/PQ_NPZ)),
             BASELINES: dict(path=str(big/pair/BASELINES), keys=list(base), dtypes={k: str(a.dtype) for k, a in base.items()}, sha256=io_.sha_file(big/pair/BASELINES))}
    document = dict(schema=hp.SCHEMA.format('pq'), pair=pair, mode=layout.mode, contract=hp.CONTRACT, plan_sha256=hp.PLAN_SHA256, settings_sha256=checks['settings_sha256'],
                    code=dict(commit=checks['code'].get('commit'), head=checks['code'].get('head'), clean=checks['code'].get('clean')),     # NM1: commit = P1 (hold-out) or HEAD (smoke)
                    lock=dict(path=lock['path'], sha256=lock['sha256']), line_status=field_record['line_status'],
                    role="PQ: an added output of the method (C22.H2); its Dice are the method's own numbers and enter no Dice table (C22.H6-5); not a registration posterior, not a MAP, not a segmentation",
                    method=dict(field_record, finalized=checks['finalized']),
                    method_files={rel: dict(path=str(p), sha256=field_record['files'].get(rel)) for rel, p in inputs.paths.items()},
                    moves=(checks.get('moves') or {}).get(pair),                                           # move.json and, after a half move, move_resume.json (path, sha256; review NI1)
                    hm=dict(hm1=True, hm2=True, hm3=True, hm4=True, rebuilt_equals_final=field_record.get('rebuilt_equals_final'), labels_checked_against_run_json=labels is not None,
                            labels_absent_reason=field_record.get('labels_sha256_absent_reason')),
                    inputs=dict(fixed=extra['fixed'], moving=extra['moving'], seg4_table_sha256=hp.SEG4_TABLE_SHA256, structure_layer=getattr(inputs, 'layer_record', None),
                                variance={s: dict(sha256=members_rows[s].get('var_sha256'), path=members_rows[s].get('var_path')) for s in (pin.fixed_subject, pin.moving_subject)},
                                data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], keys_read=list(pin.provenance['keys_read']))),
                    sampling=dict(rule='iota: warp_labels_l2r of the voxel numbers + 1 (scripts/evaluate_l2r_style.py), order 0, outside -> background (C22.Q2-1, 2)', d_sha256=disp['d_sha256'],
                                  d_hash_rule=disp['rule'], n_outside=extra['n_outside'], spacing_xyz=disp['spacing_xyz'], m4='argmax P~_M = warp_labels_l2r(L_M, d) bit for bit'),
                    numbers=extra['numbers'], tail_mass=extra['tail_mass'], q_range=extra['q_range'], eps_em=extra['eps_em'], files=files, seconds=perf_counter()-tick, utc=utc())
    write_once(small/pair/PQ_JSON, document)
    return document


def write_manifest(layout, small, big, resume):
    """C22.H4-5: one sha256 line per file that counts (sha256sum format, absolute paths, sorted): the run's pair directories on both roots and its records, and the members run that counts
    (pqh_inputs.members_dirs) with its V files.  Refusal records, other runs' directories, the half-written directories a resume renamed (<name>.attempt<k>, review NM5) and the list itself are left
    out, and so are stop records before a lock left in a rerun's members directory (pq_refused*.json, review 2 n3 (a)).  A resume writes its own list over the whole run."""
    m_small, m_big = hin.members_dirs(layout)
    paths = [f for root in (m_small, m_big) if root.is_dir() for f in root.rglob('*')
             if f.is_file() and not any('.attempt' in part for part in f.relative_to(root).parts) and not f.name.startswith('pq_refused')]     # NM5; n3 (a): stop records are not outputs
    for p in layout.pairs:
        paths += [f for root in (small/p, big/p) if root.is_dir() for f in root.rglob('*') if f.is_file()]
    paths += [small/n for n in (OUT_RECORD, OUT_RESUME_RECORD, 'opened_files_out.json', 'opened_files_out.resume.json') if (small/n).is_file()]
    rows = sorted({str(f): io_.sha_file(f) for f in paths}.items())
    return write_once(small/(MANIFEST if not resume else MANIFEST_RESUME), text=''.join(f'{sha}  {path}\n' for path, sha in rows))
