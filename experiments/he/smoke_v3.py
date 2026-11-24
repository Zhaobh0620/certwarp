"""SM3 (runs/20261114-p0/任务书.md, the lead's rulings 5 and 7; contract C22.H8 冒烟 with revisions 7a-7c, C24.3-10; preregistration revision 2 item 9 step 15): the smoke he_smoke_v3 after its A segment.

The A segment (HE with C off, D85) is smoke_he.py's queue on the same root, started with an explicit --root (smoke_he.DEFAULT_ROOT is still he_smoke_v2):
    smoke_he.py queue --root /home/rmcfz/seq_vp_tag_results/he_smoke_v3
This file runs D120's four segments after it, on copies of the A segment's arm a (runs/20261109-hb/smoke_plan.md, version 3), from the same detached checkout <root>/code:

    python -B experiments/he/smoke_v3.py --dry-run --root R
        the steps, their start strings, the gate, the root pin, the injections' fixed values, the A segment's state, the three spellings of PQ's /mnt/d root (the lead's ruling 5); nothing written,
        torch not imported, no large file read
    smoke_v3.py launch-string --root R              the lead's start string of the queue (systemd-run, unit tag-he-smoke-v3-queue, working directory <root>/code)
    smoke_v3.py queue --root R [--wait-minutes 30] [--gap-seconds 15] [--min-host-gib 15]
        the queue (in its unit), one step at a time, each after the gate (smoke_he.GATE with this queue's unit; the review units ask the GPU idle and MemAvailable >= 10 GiB, C22.H1-8; the PQ units their
        own gates' MemAvailable); a step that exits non-zero or leaves no expected record stops the queue (v3/queue_stop*.json); nothing is rerun; a root whose v3 queue launched a step never starts it
        again.  The steps:
          copy          (in the queue's process) the A segment's arm a, method-side files only, copied to <root>/<segment>/a/0438-0439 for ii, iv-a, iv-b, iv-c, sha256 before and after
                        (<segment>/copy.json); each copy's context rebuilt by run_hx.make_context (out the copy, code_head this checkout's HEAD): (ii) the frozen values, (iv-*) the injection values
                        (injection_settings: the frozen values with the segment's fixed injection and the copy's state hash, so settings_frozen false and the context marked test_context, which
                        every production command line refuses); v3/baseline.json: the sha256 of every file of the A arm and of he_smoke_v2
          i-members     PQ (i): run_pqh.py dry-run members-dev (0 problems) then run_pqh.py launch members-dev (tag-hb-pq-members-dev, GPU)
          ii-review     (ii) hb_review.py dev-unit on the (ii) copy (tag-he-smoke-ii-review; he_core.EVALUATION_UNIT, no GPU)
          ii-pq         (ii) PQ: run_pqh.py out-dev (tag-hb-pq-out-dev)
          iii-replay    (iii) run_pqh.py p3-replay (tag-hb-pq-p3-replay)
          iva-review    (iv-a) this file's forced-review (tag-he-smoke-iva-review): the review certificate perturbed for the copy's array only
          ivb-review    (iv-b) the same, and the third certification forced not to pass for that array only: configuration C in tag-he-c3-0438-0439 for real (GPU)
          ivb-pq        PQ on (iv-b)'s fallback pair: run_pqh.py out-dev-ivb (tag-hb-pq-out-dev-ivb; the lead's default "加", smoke_plan_pq_segments.md §4 item 1)
          ivc-review    (iv-c) the same as (iv-b), and configuration C's launcher after that failed third certificate starts no frozen run (no candidate): the identity
          ivc-pq        PQ on (iv-c)'s fallback pair: run_pqh.py out-dev-ivc
          v3-check      this file's check (CPU): the reading of every segment, smoke_v3_report.json / .md
    smoke_v3.py forced-review --root R --segment iv-a|iv-b|iv-c --batch-sha SHA       a unit of the queue (hb_review.review_pair in its development mode, with the injection)
    smoke_v3.py forced-third --context FILE --root R --segment iv-a|iv-b|iv-c          the third certification's child of a forced review (hb_review.third_child_main, with the injection)
    smoke_v3.py check --root R                                                          read only on the records (no truth, no evaluation record; the A arm and he_smoke_v2 hashed again)

The injections (C22.H8 (iv), revisions 7b, 7c; 01_合同.md:3417-3428).  Only in this file (no production module imports it; a test checks), only through the interfaces hb_review already has
(review_pair's certification= and third=, third_child_main's certification= and launcher=), only on a context whose values are exactly injection_settings(segment, the copy's state hash) under
<results>/he_smoke*/<iv-a|iv-b|iv-c>/a/0438-0439 (injection_context_problem), and only on the one array whose C-order sha256 is that state hash (InjectedCertify computes the hash of every array it is
given; another array is certified by the real certificate, unchanged).  The fixed values: the review certificate's `reason` (a compared field, C22.H8 (iv-a) "改了某个比较字段") set to PERTURB_VALUE;
the third certificate's status set to FORCED_STATUS (not jd_curl's: no reader can take it for a certificate) and its reason to FORCED_REASON; configuration C's launcher NoCandidateLauncher, which
starts nothing and writes he_smoke_v3_no_candidate.json in the run directory, only after this process forced the target's third certificate and only for the fallback_configc_after_third_<k>/run of
this copy.  The fallback's candidates, the identity and the new output's review use the real certificate.  Every process writes where the injection acted and how often (<segment>/injection/).
Exit codes: 0 done; 2 usage or a refused start; 8 a STOP or a failed check; the review units return hb_review's codes; the third certification's child hb_review's (THIRD_REFUSED_EXIT 3 before its spy).
"""
import argparse
import dataclasses
from dataclasses import dataclass, replace
import fcntl
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import traceback
from types import MappingProxyType

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import smoke_he as S  # noqa: E402

R, he, fc, run = S.R, S.he, S.fc, S.run
core = he.core

SCRIPT = 'experiments/he/smoke_v3.py'
SMOKE_V3 = HERE/'smoke_v3.py'
PY = fc.PY
SCHEMA = 'bayesreg.he.smoke_v3.{}.v1'
BRIEF = 'runs/20261114-p0/任务书.md (SM3)'
PLAN_REL = 'runs/20261109-hb/smoke_plan.md'     # version 3: the HB and PQH plans merged (the lead pins its sha256 in decision_log before the queue starts; the report records it)
QUEUE_UNIT = 'tag-he-smoke-v3-queue'            # = pqh_pins.DEV_PARENT_UNITS[0] (a test compares)
V3_DIR = 'v3'                                   # <root>/v3: this queue's own records, beside the A segment's (queue.lock, queue.log, steps/, batch.json are the A queue's)
REPORT = 'smoke_v3_report'
SEGMENTS = ('ii', 'iv-a', 'iv-b', 'iv-c')
SOURCE_ARM = 'a'                                # the A segment's production launcher arm (C22.H8 修订 7c)
COPY_DIRS = ('output', 't88', 'repair')         # the method-side files (C22.H8 修订 7a): output/, t88/, repair/, admission.json; never an evaluation record
COPY_FILES = ('admission.json',)
OUTPUT_FILES = ('cert.json', 'disp.npz', 'final.npz', 'line.json')
NOT_COPIED_WORDS = ('metrics', 'evaluation', 'labels', 'smoothness', 'review_mismatch', 'third_certificate', 'identity_eval')     # a copied name holding one of these: refused (C22.H8 修订 7a)

# ---- the injections (C22.H8 (iv); the fixed values, written here once) -----------------------------------------------------------------------------
INJECTIONS = MappingProxyType({'iv-a': ('review',), 'iv-b': ('review', 'third'), 'iv-c': ('review', 'third', 'configc')})
PERTURB_FIELD = 'reason'                        # he_core.COMPARED holds it; changing it changes no number and no pass (he_core.passes reads status, the lower bound and the counts)
PERTURB_VALUE = 'he_smoke_v3 injection (C22.H8 (iv)): the review certificate changed in this one compared field; not a certificate'
FORCED_STATUS = 'he_smoke_v3_forced_not_passed'
FORCED_REASON = 'he_smoke_v3 injection (C22.H8 (iv-b), (iv-c)): the third certification of this array forced not to pass; not a certificate'
NO_CANDIDATE_KIND = 'he_smoke_v3_no_candidate'
NO_CANDIDATE_RECORD = 'he_smoke_v3_no_candidate.json'
INJECTION_RULE = ('C22.H8 (iv) revisions 7b, 7c: only in experiments/he/smoke_v3.py, only through hb_review.review_pair (certification=, third=) and hb_review.third_child_main (certification=, '
                  'launcher=), only on the copy whose output has this state hash; the fallback candidates, the identity and the new output are certified by the real certificate')


# ---- small helpers --------------------------------------------------------------------------------------------------------------------------------
def _review():
    """hb_review (experiments/hb), imported when needed (the queue and the dry run never import it)."""
    path = str(EXPERIMENTS/'hb')
    if path not in sys.path:
        sys.path.insert(0, path)
    import hb_review
    return hb_review


def _pins():
    """pqh_pins (experiments/pq): the fixed paths and unit names of PQ's dev mode."""
    path = str(EXPERIMENTS/'pq')
    if path not in sys.path:
        sys.path.append(path)
    import pqh_pins
    return pqh_pins


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return he.read_json(path)


def v3_dir(root):
    return Path(root)/V3_DIR


def segment_root(root, segment):
    return Path(root)/segment


def cell_dir(root, segment):
    """The copy of segment `segment`: <root>/<segment>/a/0438-0439 (pqh_pins.dev_cell_rel for ii, iv-b, iv-c)."""
    if segment not in SEGMENTS:
        raise ValueError(f'segment {segment!r}: one of {SEGMENTS}')
    return Path(root)/segment/SOURCE_ARM/S.SHORT


def copy_record_path(root, segment):
    return segment_root(root, segment)/'copy.json'


def tree_rows(directory):
    """{relative path: {bytes, sha256}} of every regular file under `directory` (a symbolic link: its target, not followed); None when it is not a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    rows = {}
    for top, dirs, files in os.walk(directory):
        for name in sorted(files)+sorted(d for d in dirs if os.path.islink(os.path.join(top, d))):
            path = Path(top)/name
            rel = str(path.relative_to(directory))
            if path.is_symlink():
                rows[rel] = dict(link=os.readlink(path))
            elif path.is_file():
                rows[rel] = dict(bytes=path.stat().st_size, sha256=sha_file(path))
    return dict(sorted(rows.items()))


def numbered(directory, stem):
    """<directory>/<stem>.json, or <stem>_<k>.json with k the first free number."""
    path, k = Path(directory)/f'{stem}.json', 0
    while path.exists():
        path, k = Path(directory)/f'{stem}_{k}.json', k+1
    return path


# ---- the injections ---------------------------------------------------------------------------------------------------------------------------------
class InjectedCertify:
    """certify(u, grid) of the real certificate, changed only for the one array whose C-order sha256 (he_core.array_sha256, the certificate's state hash convention) is `target`.
    kind 'perturb': the result's PERTURB_FIELD set to PERTURB_VALUE (the review of (iv-*)); 'fail': its status FORCED_STATUS and reason FORCED_REASON (the third certification of (iv-b), (iv-c));
    None: never changed (the third certification of (iv-a): counted only).  Every call recorded (calls)."""

    def __init__(self, certify, *, target, kind, role):
        if kind not in (None, 'perturb', 'fail'):
            raise ValueError(f'kind {kind!r}: None, perturb or fail')
        if not (isinstance(target, str) and re.fullmatch(r'[0-9a-f]{64}', target)):
            raise ValueError(f'target {target!r}: a sha256')
        self.certify, self.target, self.kind, self.role, self.calls = certify, target, kind, role, []

    def __call__(self, u, grid, **kwargs):
        digest = he.array_sha256(u)
        result = self.certify(u, grid, **kwargs)
        row = dict(n=len(self.calls), state_hash=digest, on_target=digest == self.target, applied=False, field=None)
        if self.kind is not None and digest == self.target:
            if not dataclasses.is_dataclass(result):
                raise TypeError(f'{type(result).__name__}: the certificate result is not a dataclass (jd_curl FeasibilityResult)')
            if self.kind == 'perturb':
                real = getattr(result, PERTURB_FIELD)
                result = replace(result, **{PERTURB_FIELD: PERTURB_VALUE})
                row.update(applied=real != PERTURB_VALUE, field=PERTURB_FIELD, real=real, forced=PERTURB_VALUE)
            else:
                real, real_passed = result.status, he.passes(result)
                result = replace(result, status=FORCED_STATUS, reason=FORCED_REASON)
                row.update(applied=True, field='status', real=real, real_passed=real_passed, forced=FORCED_STATUS)
        self.calls.append(row)
        return result

    def forced_failure(self):
        """Whether this process forced the target's certificate not to pass (kind fail, applied on the target)."""
        return self.kind == 'fail' and any(r['applied'] and r['on_target'] for r in self.calls)

    def record(self):
        return dict(role=self.role, kind=self.kind, target_state_hash=self.target, n_calls=len(self.calls), n_on_target=sum(r['on_target'] for r in self.calls),
                    n_applied=sum(r['applied'] for r in self.calls), other_states=sorted({r['state_hash'] for r in self.calls if not r['on_target']}), calls=list(self.calls), rule=INJECTION_RULE)


class NoCandidateLauncher:
    """(iv-c): configuration C's launcher after the target's failed third certificate: it starts no frozen run (run_hx.configc_fallback then classifies a run that never ran: no candidate, D75) and
    writes NO_CANDIDATE_RECORD in the run directory.  Only when `certify` forced the target's third certificate in this process and the run is <out>/fallback_configc_after_third_<k>/run; any other
    call goes to the real launcher (run_hx.third_launcher: tag-he-c3)."""
    kind = NO_CANDIDATE_KIND
    unit_prefix = None

    def __init__(self, real, certify, out):
        self.real, self.certify, self.out, self.calls = real, certify, Path(out), []

    def __call__(self, run_dir, short):
        run_dir = Path(run_dir)
        here = (run_dir.name == 'run' and run_dir.parent.name.startswith('fallback_configc_after_third_') and run_dir.parent.parent.resolve() == self.out.resolve())
        if not (here and self.certify.forced_failure()):
            self.calls.append(dict(run=str(run_dir), applied=False, why='not after a forced third certificate of the target, or not this copy\'s fallback after it'))
            return self.real(run_dir, short)
        doc = dict(schema=SCHEMA.format('no_candidate'), run=str(run_dir), short=short, target_state_hash=self.certify.target, started=False, utc=he.utc(),
                   rule='C22.H8 (iv-c): configuration C has no usable candidate -- no frozen run is started (an injection of he_smoke_v3); the identity follows with the real certificate')
        he.write_json(run_dir/NO_CANDIDATE_RECORD, doc)
        self.calls.append(dict(run=str(run_dir), applied=True))
        return dict(kind=self.kind, started=False, unit=None, argv=None, returncode=None, stderr_tail='', probes=fc.probes_of(run_dir), injection=doc)

    def record(self):
        return dict(role='configc_launcher', kind=self.kind, target_state_hash=self.certify.target, n_calls=len(self.calls), n_applied=sum(c['applied'] for c in self.calls), calls=list(self.calls))


def injection_settings(segment, target):
    """The values of a forced copy (iv-*): the frozen values, the name changed and the segment's injection recorded (so settings_frozen false: make_context marks the context test_context and every
    production command line refuses it, run_hx.cli_context_problem)."""
    if segment not in INJECTIONS:
        raise ValueError(f'segment {segment!r}: one of {sorted(INJECTIONS)}')
    kinds = INJECTIONS[segment]
    s = he.frozen_settings()
    s['name'] = f'he_smoke_v3 ({segment}) forced inconsistency (not the method; contract C22.H8 (iv))'
    s['smoke_injection'] = dict(segment=segment, target_state_hash=target, injections=list(kinds), review=dict(field=PERTURB_FIELD, value=PERTURB_VALUE),
                                third=dict(status=FORCED_STATUS, reason=FORCED_REASON) if 'third' in kinds else None,
                                configc=dict(launcher=NO_CANDIDATE_KIND, record=NO_CANDIDATE_RECORD) if 'configc' in kinds else None, plan=PLAN_REL, brief=BRIEF, rule=INJECTION_RULE)
    return s


def injection_context_problem(ctx, root, segment, *, results=None, stage=None):
    """None, or why the forced entries (forced-review, forced-third) may not run this context: exactly injection_settings(segment, the copy's recorded state hash), settings_frozen false, the identity
    checked, marked test_context, the smoke's pair and stage, the development pair set, the unit launcher, its output the copy <results>/he_smoke*/<segment>/a/0438-0439."""
    if segment not in INJECTIONS:
        return f'segment {segment!r}: the forced entries run only {sorted(INJECTIONS)}'
    problem = S.root_problem(root, results)
    if problem:
        return problem
    cell = cell_dir(root, segment)
    if Path(ctx.get('out', '')).resolve() != cell.resolve():
        return f"{ctx.get('out')}: a forced context's output is the copy {cell}"
    copy = read(copy_record_path(root, segment)) or {}
    target = copy.get('target_state_hash')
    if not isinstance(target, str):
        return f'{copy_record_path(root, segment)}: no copy record with its state hash'
    if he.canonical(ctx.get('settings')) != he.canonical(injection_settings(segment, target)):
        return f'the values are not exactly injection_settings({segment!r}, {target[:12]}…)'
    if ctx.get('settings_frozen') is not False or ctx.get('check_identity') is not True or ctx.get('test_context') != R.TEST_CONTEXT:
        return f"settings_frozen {ctx.get('settings_frozen')}, check_identity {ctx.get('check_identity')}, test_context {ctx.get('test_context')!r}: not a forced context"
    if ctx.get('pair') != S.PAIR or Path(ctx.get('stage_dir', '')) != Path(S.STAGE if stage is None else stage) or R.hold.pair_set_of(ctx) != 'dev':
        return f"the pair {ctx.get('pair')} of {ctx.get('stage_dir')} ({R.hold.pair_set_of(ctx)}): the smoke runs {S.PAIR} of the development stage only"
    if (ctx.get('launcher') or {}).get('kind') != 'unit':
        return 'a forced context launches configuration C in a unit of its own'
    return None


def write_injection_record(root, segment, role, document, rc):
    directory = segment_root(root, segment)/'injection'
    directory.mkdir(parents=True, exist_ok=True)
    return S.write_once(numbered(directory, role), dict(document, schema=SCHEMA.format('injection'), segment=segment, exit=rc, pid=os.getpid(), utc=he.utc()))


class SmokeThirdChild:
    """hb_review.ThirdChild for a forced review: python -B smoke_v3.py forced-third --context FILE --root R --segment S, this process's environment, no timeout of its own (the review unit's
    RuntimeMaxSec), the log in the cell's review/third-cert.log."""

    def __init__(self, root, segment):
        self.root, self.segment = Path(root), segment

    def argv(self, ctx):
        return fc.check_launch([PY, '-B', str(SMOKE_V3), 'forced-third', '--context', str(R.context_path(ctx)), '--root', str(self.root), '--segment', self.segment])

    def __call__(self, ctx):
        argv = self.argv(ctx)
        (Path(ctx['out'])/'review').mkdir(exist_ok=True)
        with (Path(ctx['out'])/'review'/'third-cert.log').open('a') as f:
            return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT).returncode


def dev_mode(review, root, segment, ctx, batch_sha):
    seg = segment_root(root, segment)
    return review.Mode('dev', seg/'review', batch_sha, review.dev_policy(seg, Path(ctx['stage_dir'])/ctx['pair']), segment=seg)


def forced_review_main(root, segment, batch_sha, *, results=None, stage=None, certification=None, third=None, probe=None):
    """A forced review (iv-*): the copy's context checked (injection_context_problem, hb_review.dev_root_problem), the input manifest as hb_review.py dev-unit writes it, then
    hb_review.review_pair in the development mode with the review certificate perturbed for the target only and this file's third certification child.  certification, third, probe: the tests'
    stand-ins of the real certificate (it is wrapped all the same), the child and the unit probe."""
    review = _review()
    cell = cell_dir(root, segment)
    try:
        ctx = R.read_context(cell/'context.json')
    except SystemExit as exc:
        print(f'refused: {exc}', file=sys.stderr)
        return 2
    problem = injection_context_problem(ctx, root, segment, results=results, stage=stage) or review.dev_root_problem(segment_root(root, segment), cell, results=results)
    if problem is None and ctx.get('code_head') != batch_sha:
        problem = f"the copy's code_head {ctx.get('code_head')} is not the batch's {batch_sha}"
    if problem:
        print(f'refused: {problem}', file=sys.stderr)
        return 2
    mode = dev_mode(review, root, segment, ctx, batch_sha)
    if not mode.manifest().is_file():                   # as hb_review.py dev-unit
        review.hb.write_once(mode.manifest(), dict(schema=review.hb.SCHEMA.format('input_manifest'), mode='dev', pairs={ctx['pair']: dict(step4='output', cell=str(cell),
                                                                                                                                           files=review.hb.file_rows(cell, review.hb.INPUT_FILES))},
                                                    utc=review.hb.utc(), written_by=SCRIPT))
    real, config = R.certifier(ctx) if certification is None else certification
    injected = InjectedCertify(real, target=ctx['settings']['smoke_injection']['target_state_hash'], kind='perturb', role='review')
    rc = review.review_pair(ctx, mode, batch=str(mode.root), certification=(injected, config), third=SmokeThirdChild(root, segment) if third is None else third, probe=probe)
    write_injection_record(root, segment, 'review', injected.record(), rc)
    return rc


def forced_third_main(context, root, segment, *, results=None, stage=None, certification=None, launcher=None):
    """The third certification's child of a forced review: refused before its spy -> hb_review.THIRD_REFUSED_EXIT (as hb_review.py third-cert); then hb_review.third_child_main with the
    certificate forced not to pass for the target ((iv-b), (iv-c); counted only in (iv-a)) and, in (iv-c), NoCandidateLauncher.  certification, launcher: the tests' stand-ins of the real ones."""
    review = _review()
    try:
        ctx = R.read_context(context)
        problem = injection_context_problem(ctx, root, segment, results=results, stage=stage) or review.dev_root_problem(segment_root(root, segment), ctx['out'], results=results)
        if problem:
            print(f'refused: {problem}', file=sys.stderr)
            return review.THIRD_REFUSED_EXIT
        mode = dev_mode(review, root, segment, ctx, None)
        kinds = INJECTIONS[segment]
        real, config = R.certifier(ctx) if certification is None else certification
        injected = InjectedCertify(real, target=ctx['settings']['smoke_injection']['target_state_hash'], kind='fail' if 'third' in kinds else None, role='third')
        chosen = None
        if 'configc' in kinds:
            chosen = NoCandidateLauncher(R.third_launcher(ctx) if launcher is None else launcher, injected, ctx['out'])
        elif launcher is not None:
            chosen = launcher
    except (Exception, SystemExit) as exc:              # noqa: BLE001 - before the spy and anything of the pair: refused, as hb_review.py third-cert
        print(f'refused before the spy: {type(exc).__name__}: {exc}', file=sys.stderr)
        return review.THIRD_REFUSED_EXIT
    rc = review.third_child_main(ctx, mode, launcher=chosen, certification=(injected, config))
    document = injected.record()
    if isinstance(chosen, NoCandidateLauncher):
        document['launcher'] = chosen.record()
    write_injection_record(root, segment, 'third', document, rc)
    return rc


# ---- the steps -----------------------------------------------------------------------------------------------------------------------------------
EVALUATION = dict(he.EVALUATION_UNIT)
REVIEW_GATE_GIB = he.FROZEN_RUN_GATE_GIB        # C22.H1-8: MemAvailable >= 10 GiB before a review unit ((iv-b)'s can start configuration C)


@dataclass(frozen=True)
class Step:
    name: str
    kind: str                                   # copy | pq | review | forced-review | check
    segment: str = None
    gpu: bool = False                           # the gate asks the GPU free (the review units: C22.H1-8 "GPU 空"; the members unit; (iv-b)'s tag-he-c3)
    command: str = None                         # run_pqh.py's subcommand (kind pq)
    mem_gib: float = he.OTHER_GATE_GIB          # the gate's MemAvailable

    @property
    def unit(self):
        if self.kind == 'copy':
            return None
        if self.kind == 'pq':
            p = _pins()
            if self.command == 'members-dev':
                return p.DEV_UNITS['members']
            if self.command == 'p3-replay':
                return p.DEV_UNITS['p3_replay']
            return p.DEV_OUT_UNITS[self.segment]
        return f'tag-he-smoke-{self.name}'


STEPS = (Step('copy', 'copy'),
         Step('i-members', 'pq', None, True, 'members-dev', 8.),
         Step('ii-review', 'review', 'ii', True, None, REVIEW_GATE_GIB),
         Step('ii-pq', 'pq', 'ii', False, 'out-dev', 9.),
         Step('iii-replay', 'pq', None, False, 'p3-replay', 9.),
         Step('iva-review', 'forced-review', 'iv-a', True, None, REVIEW_GATE_GIB),
         Step('ivb-review', 'forced-review', 'iv-b', True, None, REVIEW_GATE_GIB),
         Step('ivb-pq', 'pq', 'iv-b', False, 'out-dev-ivb', 9.),
         Step('ivc-review', 'forced-review', 'iv-c', True, None, REVIEW_GATE_GIB),
         Step('ivc-pq', 'pq', 'iv-c', False, 'out-dev-ivc', 9.),
         Step('v3-check', 'check'))


def step_named(name):
    return next(s for s in STEPS if s.name == name)


def unit_argv(step, code, root, sha):
    """The start string of a review or check unit (smoke_he.step_argv's way: run_mc0.unit_argv with he_core.EVALUATION_UNIT, OOMPolicy after MemorySwapMax, he_core.unit_argv_problems)."""
    code, root = Path(code), Path(root)
    if step.kind == 'review':
        inner, script = ['dev-unit', '--cell', str(cell_dir(root, step.segment)), '--root', str(segment_root(root, step.segment)), '--batch-sha', sha], 'experiments/hb/hb_review.py'
    elif step.kind == 'forced-review':
        inner, script = ['forced-review', '--root', str(root), '--segment', step.segment, '--batch-sha', sha], SCRIPT
    elif step.kind == 'check':
        inner, script = ['check', '--root', str(root)], SCRIPT
    else:
        raise ValueError(f'{step.name}: not a unit of this file')
    res = EVALUATION
    argv = run.unit_argv(step.unit, code, inner, log=v3_dir(root)/f'{step.unit}.log', gpu=False, numba=res['numba'], quota=res['quota'], runtime=res['runtime'], script=script,
                         memory_max=res['memory_max'], nice=res['nice'])
    argv.insert(argv.index('--property=MemorySwapMax=4G')+1, f"--property=OOMPolicy={res['oom_policy']}")
    problems = he.unit_argv_problems(argv, res)
    if problems:
        raise ValueError(f'{step.unit}: the start string is not the unit he_core names: {problems}')
    return run.check_launch(argv)


def pq_argv(step, code, what):
    """run_pqh.py dry-run|launch <command>, from the checkout (its own systemd-run --wait: the unit, its specification, xb_unit.py)."""
    return run.check_launch([PY, '-B', str(Path(code)/'experiments/pq/run_pqh.py'), what, step.command])


def pq_dir(root, segment=None):
    """The PQ dev mode's record directory of a segment's out unit (pqh_inputs.run_dirs: pq/ for (ii), pq/<segment>/ for (iv-b), (iv-c))."""
    pq = Path(root)/'pq'
    return pq if segment in (None, 'ii') else pq/segment


def pq_big(root, segment=None):
    big = Path(root)/'pq'/'big'
    return big if segment in (None, 'ii') else big/segment


# the PQ units' record names (pqh_units / pqh_inputs; a test compares)
OUT_RECORD, PQ_MANIFEST, PQ_JSON, PQ_NPZ, BASELINES = 'pq_record.json', 'pq_outputs_manifest.txt', 'pq.json', 'pq.npz', 'baselines.npz'
MEMBERS_RECORD, OPENED_MEMBERS, OPENED_OUT = 'members_record.json', 'opened_files_members.json', 'opened_files_out.json'


def expected_files(step, root):
    """What a step must leave: a missing one stops the queue as a non-zero exit does."""
    root = Path(root)
    if step.kind == 'copy':
        return [copy_record_path(root, s) for s in SEGMENTS]+[v3_dir(root)/'baseline.json']
    if step.kind == 'pq':
        if step.command == 'members-dev':
            return [root/'pq'/'members'/MEMBERS_RECORD]
        if step.command == 'p3-replay':
            return [root/'pq'/'p3_replay'/'p3_replay.json']
        return [pq_dir(root, step.segment)/OUT_RECORD, pq_dir(root, step.segment)/PQ_MANIFEST]
    if step.kind in ('review', 'forced-review'):
        return [cell_dir(root, step.segment)/'review'/'review.json', cell_dir(root, step.segment)/'terminal.json']
    return [root/f'{REPORT}.json']


def queue_launch(root):
    """The lead's start string of the queue: from the smoke's checkout <root>/code (whose HEAD is the A segment's commit), PYTHONPATH <frozen>:<code> as the units'."""
    code = Path(root)/'code'
    return run.check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT}', '--collect', f'--working-directory={code}', '--property=MemoryMax=4G', '--nice=5',
                             '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=NUMEXPR_NUM_THREADS=1',
                             '--setenv=CUDA_VISIBLE_DEVICES=', f'--setenv=PYTHONPATH={core.FROZEN}:{code}', PY, '-B', str(code/SCRIPT), 'queue', '--root', str(root)])


GATE = replace(S.GATE, own_units=(f'{QUEUE_UNIT}.service',))


def step_gate(step, gate):
    return replace(gate, mem_available_gib=max(gate.mem_available_gib or 0., step.mem_gib))


# ---- the queue's preconditions -------------------------------------------------------------------------------------------------------------------
def root_problem(root, results=None):
    """A v3 root: smoke_he's rule (<results>/he_smoke*) and PQ's pinned root (pqh_pins.SMOKE_ROOT: the dev mode's paths are code constants; a new root changes them, C24.4-5)."""
    problem = S.root_problem(root, results)
    if problem:
        return problem
    pinned = Path(_pins().SMOKE_ROOT)
    if Path(root).resolve() != pinned.resolve():
        return f'{root}: PQ\'s dev mode is pinned to {pinned} (pqh_pins.SMOKE_ROOT, DEV_PQ_ROOT and its locks); a new root changes them first (C24.4-5)'
    return None


def latest_report(root):
    """The A segment's newest report (smoke_he.check writes smoke_report.json, then smoke_report_<k>.json): (path, document) or (None, None)."""
    root = Path(root)
    paths = [root/'smoke_report.json']+sorted(root.glob('smoke_report_[0-9]*.json'), key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    paths = [p for p in paths if p.is_file()]
    return (paths[-1], read(paths[-1])) if paths else (None, None)


def a_segment(root):
    """The A segment as the v3 queue takes it: its batch (code_sha), its queue_end.json, its newest report with no item 不过 or 不过（待查）, C off (tissue_prior off), and arm a's line status."""
    root = Path(root)
    batch, end = read(root/'batch.json') or {}, read(root/'queue_end.json')
    path, report = latest_report(root)
    problems = []
    sha = batch.get('code_sha')
    if not (isinstance(sha, str) and re.fullmatch(r'[0-9a-f]{40}', sha)):
        problems.append(f"{root/'batch.json'}: no code_sha (the A segment did not start)")
    if end is None:
        problems.append(f"{root/'queue_end.json'}: the A segment's queue did not end")
    bad = []
    if report is None:
        problems.append('no smoke_report*.json: the A segment was not read')
    else:
        bad = [f"{i.get('item')} {i.get('name')}: {i.get('verdict')}" for i in report.get('items') or [] if i.get('verdict') in (S.FAIL, S.OPEN)]
        if bad or not report.get('items'):
            problems.append(f'{path.name}: items not passed {bad or "(no items)"}: (ii) and (iv) wait for the A segment (C22.H8 修订 7c)')
        if report.get('tissue_prior') != 'off':
            problems.append(f"{path.name}: tissue_prior {report.get('tissue_prior')!r}: the A segment is HE with C off (D85)")
        if report.get('code_sha') != sha:
            problems.append(f"{path.name}: code_sha {report.get('code_sha')} is not the batch's {sha}")
    line = read(S.run_dir(root, SOURCE_ARM)/'output'/'line.json') or {}
    if line.get('line_status') != he.HX:
        problems.append(f"arm a's line status {line.get('line_status')!r}: the plan's expectations are for hx_certified (C22.H8 (ii) 修订 7c: a fallback pair goes by C22.H4-1's row; to the lead)")
    return dict(code_sha=sha, report=None if path is None else str(path), report_sha256=None if path is None else sha_file(path), counts=(report or {}).get('counts'),
                summary=(report or {}).get('summary'), not_passed=bad, line_status=line.get('line_status'), problems=problems)


class System(S.System):
    """smoke_he.System (run_mc0's machine, the uptime, the identity) and the checkout's git state (tests replace it)."""

    def checkout(self, code):
        def git(*args):
            done = subprocess.run(['git', *args], cwd=code, capture_output=True, text=True)
            return done.returncode, done.stdout.strip(), done.stderr.strip()
        rc, head, err = git('rev-parse', 'HEAD')
        rc2, status, err2 = git('status', '--porcelain=v1', '--untracked-files=no')
        return dict(code=str(code), head=head if rc == 0 else None, clean=rc2 == 0 and status == '', status=status or err or err2, this_process_tree=str(Path(core.ROOT).resolve()),
                    runs_from_it=Path(core.ROOT).resolve() == Path(code).resolve())


def checkout_problems(record, sha):
    problems = []
    if not record.get('runs_from_it'):
        problems.append(f"this queue runs from {record.get('this_process_tree')}, not the smoke's checkout {record.get('code')} (C22.H8: every unit from one detached checkout)")
    if record.get('head') != sha:
        problems.append(f"{record.get('code')}: HEAD {record.get('head')} is not the A segment's {sha}")
    if not record.get('clean'):
        problems.append(f"{record.get('code')}: tracked files changed ({str(record.get('status'))[:300]})")
    return problems


def plan_record(code):
    path = Path(code)/PLAN_REL
    return dict(path=PLAN_REL, sha256=sha_file(path) if path.is_file() else None, rule='the lead pins this sha256 in decision_log before the queue starts (preregistration revision 2 item 9 step 15)')


# ---- the copies ----------------------------------------------------------------------------------------------------------------------------------
def source_listing(src):
    """The method-side files of the A arm (relative paths) and the problems that refuse the copy: output/ not exactly OUTPUT_FILES, a missing directory or file, a link, an evaluation-like name."""
    src = Path(src)
    rels, problems = [], []
    for name in COPY_FILES:
        (rels if (src/name).is_file() else problems).append(name if (src/name).is_file() else f'{src/name}: not there')
    for d in COPY_DIRS:
        if not (src/d).is_dir() or (src/d).is_symlink():
            problems.append(f'{src/d}: not a directory')
            continue
        for path in sorted((src/d).rglob('*')):
            rel = str(path.relative_to(src))
            if path.is_symlink():
                problems.append(f'{rel}: a link')
            elif path.is_file():
                rels.append(rel)
    out = sorted(Path(r).name for r in rels if r.startswith('output/'))
    if out != sorted(OUTPUT_FILES):
        problems.append(f'output/ holds {out}, not exactly {list(OUTPUT_FILES)} (a review record of the A segment\'s evaluation: the copy waits, to the lead)')
    named = [r for r in rels if any(w in Path(r).name for w in NOT_COPIED_WORDS)]
    if named:
        problems.append(f'evaluation-like names among the method-side files: {named}')
    return rels, problems


def npz_u(path):
    with np.load(path, allow_pickle=False) as z:
        return np.array(z['u'])


def copy_segment(root, segment, *, stage=None):
    """One copy (module docstring).  run_mc0.Stop when a check does not hold; the record <segment>/copy.json is written last, once."""
    root = Path(root)
    src, dst = S.run_dir(root, SOURCE_ARM), cell_dir(root, segment)
    record_path = copy_record_path(root, segment)
    if record_path.exists() or dst.exists():
        raise run.Stop(f'{segment}: {dst} or {record_path} is there: a copy is made once')
    rels, problems = source_listing(src)
    if problems:
        raise run.Stop(f'{segment}: the A arm cannot be copied: {problems}')
    before = {rel: sha_file(src/rel) for rel in rels}
    for rel in rels:
        (dst/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src/rel, dst/rel)
    files, same = [], True
    for rel in rels:
        after, dest = sha_file(src/rel), sha_file(dst/rel)
        row = dict(file=rel, bytes=(dst/rel).stat().st_size, sha256_before=before[rel], sha256_after_source=after, sha256_dest=dest, same=before[rel] == after == dest)
        same &= row['same']
        files.append(row)
    line = read(dst/'output'/'line.json') or {}
    target = he.array_sha256(npz_u(dst/'output'/'final.npz'))
    line_hash = ((line.get('output') or {}).get('final') or {}).get('state_hash')
    if not same or target != line_hash:
        raise run.Stop(f'{segment}: the copy does not hold (files the same: {same}; final.npz state hash {target}, line.json {line_hash})')
    settings = he.frozen_settings() if segment == 'ii' else injection_settings(segment, target)
    ctx = R.make_context(S.STAGE if stage is None else stage, S.PAIR, dst, device='cuda', settings=settings, launcher='unit', check_identity=True)
    he.write_json(R.context_path(ctx), ctx)
    top = sorted(p.name for p in src.iterdir() if p.name not in COPY_DIRS+COPY_FILES)
    doc = dict(schema=SCHEMA.format('copy'), segment=segment, source=str(src), dest=str(dst), files=files, all_same=same, target_state_hash=target, line_state_hash=line_hash,
               line_status=line.get('line_status'), context=dict(path=str(R.context_path(ctx)), sha256=sha_file(R.context_path(ctx)), settings_sha256=ctx['settings_sha256'],
                                                             settings_frozen=ctx['settings_frozen'], test_context=ctx['test_context'], code_head=ctx['code_head']),
               source_context_sha256=sha_file(src/'context.json') if (src/'context.json').is_file() else None, not_copied=top, utc=he.utc(),
               rule='C22.H8 修订 7a, 7c: the method-side files of the A segment\'s arm a (output/, t88/, repair/, admission.json), sha256 before and after; never an evaluation record; the context '
                    'rebuilt by run_hx.make_context (out the copy, code_head this checkout)')
    S.write_once(record_path, doc)
    return doc


def copy_all(root, *, results=None, stage=None, log=print):
    """The copy step: v3/baseline.json (every file of the A arm and of he_smoke_v2 by sha256, before any copy), then the four copies."""
    root = Path(root)
    v2 = S.results_of(results)/'he_smoke_v2'
    S.write_once(v3_dir(root)/'baseline.json', dict(schema=SCHEMA.format('baseline'), a_arm=dict(root=str(S.run_dir(root, SOURCE_ARM)), files=tree_rows(S.run_dir(root, SOURCE_ARM))),
                                                     he_smoke_v2=dict(root=str(v2), files=tree_rows(v2)), utc=he.utc(),
                                                     rule='C22.H8 修订 7c: the A arm\'s original directory and he_smoke_v2 are read only; the check hashes them again'))
    out = {}
    for segment in SEGMENTS:
        out[segment] = copy_segment(root, segment, stage=stage)
        log(f"copy {segment}: {len(out[segment]['files'])} files, state hash {out[segment]['target_state_hash'][:12]}…")
    return out


def before_record(root, step):
    """What a step's check compares afterwards: before a PQ out unit, the copy's every file (D114 (a): line.json, review.json, terminal.json unchanged by PQ)."""
    if step.kind == 'pq' and step.segment is not None:
        return dict(cell=tree_rows(cell_dir(root, step.segment)))
    return None


# ---- the queue ---------------------------------------------------------------------------------------------------------------------------------------
class GateClosed(run.Stop):
    """run_mc0.wait_for_gate's STOP: the machine gate stayed closed for --wait-minutes.  Not a failure: before the first launched step the same start string starts the queue again (the
    lead's ruling B2 on the plans' review, 11-14); every other stop goes to BZ."""


class GateReading:
    """The system as run_mc0.wait_for_gate sees it, its MemAvailable readings kept: after the gate opened, `mem` is the reading of its last, passing check (busy_reason reads MemAvailable on
    every check, the step's gate asks it; SM3 revision 3).  Everything else goes to the system."""

    def __init__(self, system):
        self.system, self.mem = system, None

    def __getattr__(self, name):
        return getattr(self.system, name)

    def mem_available_gib(self):
        self.mem = self.system.mem_available_gib()
        return self.mem


def queue_body(o, system, root, log, *, results=None, stage=None):
    root = Path(root)
    d = v3_dir(root)
    launched = sorted(p.name for p in (d/'steps').glob('*.launched.json'))
    if launched:
        raise run.Stop(f'v3 steps were launched in this root before ({launched}): nothing is rerun; a new root (C22.H8)')
    earlier = [(p.name, (read(p) or {}).get('kind')) for p in sorted(d.glob('queue_stop*.json'))]
    other = [x for x in earlier if x[1] != 'gate']
    if other:                                           # SM3 revision 3 (the lead's ruling B2): a refusal at the start, an exception or anything but the machine gate: to BZ, not started again
        raise run.Stop(f'earlier starts of this queue stopped for other than the machine gate ({other}): only a gate that stayed closed before the first step starts again; to BZ')
    a = a_segment(root)
    if a['problems']:
        raise run.Stop(f"the A segment, before any step: {a['problems']}")
    sha, code = a['code_sha'], root/'code'
    checkout = system.checkout(code)
    problems = checkout_problems(checkout, sha)
    _, identity = system.code_identity(code)
    problems += [f'identity: {p}' for p in identity]
    if problems:
        raise run.Stop(f'the checkout, before any step: {problems}')
    gate = replace(GATE, host_floor_gib=o.min_host_gib)
    batch = dict(schema=SCHEMA.format('batch'), root=str(root), code_sha=sha, code=str(code), checkout=checkout, a_segment=a, plan=plan_record(code),
                 steps=[dict(name=s.name, kind=s.kind, segment=s.segment, unit=s.unit, gpu=s.gpu, command=s.command, mem_gib=s.mem_gib) for s in STEPS], segments=list(SEGMENTS),
                 injections={k: list(v) for k, v in INJECTIONS.items()}, gate=dict(own_units=list(gate.own_units), load_limit=gate.load_limit, gpu_free_mib=gate.gpu_free_mib,
                                                                                    host_floor_gib=gate.host_floor_gib, home_floor_gb=run.HOME_FLOOR_GB),
                 units=dict(review=EVALUATION, pq=dict(members=dict(_pins().UNIT_SPECS['members']), out=dict(_pins().UNIT_SPECS['out']))), gap_seconds=o.gap_seconds, wait_minutes=o.wait_minutes,
                 started_utc=run.utc())
    old = read(d/'batch.json')
    if old is None:
        S.write_once(d/'batch.json', batch)
    else:                                               # SM3 revision 2 (review I2): an earlier start stopped before its first step (the gate stayed closed): taken again only on the same code and plan
        same = dict(code_sha=old.get('code_sha') == sha, plan=(old.get('plan') or {}).get('sha256') == batch['plan']['sha256'],
                    steps=[s.get('name') for s in old.get('steps') or []] == [s.name for s in STEPS])
        if not all(same.values()):
            raise run.Stop(f"v3/batch.json is at {old.get('code_sha')} with the plan {(old.get('plan') or {}).get('sha256')}; this start is at {sha} with {batch['plan']['sha256']} "
                           f'({same}): start it from the code and the plan of the earlier start, or a new root')
        S.write_once(numbered(d, 'restart'), dict(batch_sha256=sha_file(d/'batch.json'), checkout=checkout, a_segment=a, utc=run.utc(),
                                                  options=dict(wait_minutes=o.wait_minutes, gap_seconds=o.gap_seconds, min_host_gib=o.min_host_gib),     # revision 3: recorded, not compared
                                                  rule='SM3 revision 2 (review I2): no step had been launched; the same code and plan; the same start string'))
        log('batch.json there (an earlier start stopped before its first step): the same code and plan, taken as it is')
    log(f"start: {len(STEPS)} steps at {sha}; the A segment's report {a['report']}; one at a time; a failure stops the queue")
    rcs = {}
    for step in STEPS:
        this_gate = step_gate(step, gate)
        reading = GateReading(system)                                                     # revision 3 (diff review 2): the MemAvailable the gate opened on
        try:
            run.wait_for_gate(reading, root, step.gpu, o.wait_minutes, log, this_gate)
        except run.Stop as exc:                         # only a gate that stayed closed (B2)
            raise GateClosed(str(exc)) from None
        argv = unit_argv(step, code, root, sha) if step.kind in ('review', 'forced-review', 'check') else None
        launch = dict(dry_run=pq_argv(step, code, 'dry-run'), launch=pq_argv(step, code, 'launch')) if step.kind == 'pq' else None
        S.write_once(d/'steps'/f'{step.name}.launched.json', dict(step=step.name, kind=step.kind, segment=step.segment, unit=step.unit, argv=argv, pq=launch, gpu=step.gpu,
                                                                  gate_mem_available_gib=this_gate.mem_available_gib, mem_available_gib_at_gate=reading.mem,
                                                                  mem_available_gib_at_launch=system.mem_available_gib(),
                                                                  uptime_before=system.uptime(), before=before_record(root, step), utc=run.utc()))
        log(f'launch {step.name} ({step.unit or "in the queue process"})')
        tick = system.now()
        error = None
        if step.kind == 'copy':
            try:
                copy_all(root, results=results, stage=stage, log=log)
                rc = 0
            except run.Stop as exc:
                rc, error = 8, str(exc)
            except Exception as exc:                    # noqa: BLE001 - recorded; the queue stops
                rc, error = 8, f'{type(exc).__name__}: {exc}'
        elif step.kind == 'pq':
            rc = system.run(launch['dry_run'], d/f'{step.name}.dry_run.log')
            if rc != 0:
                error = f'run_pqh.py dry-run {step.command}: exit {rc} (problems: see {d/f"{step.name}.dry_run.log"})'
            else:
                rc = system.run(launch['launch'], d/f'{step.name}.launch.log')
        else:
            rc = system.run(argv, d/f'{step.unit}.systemd_run.log')
        missing = [str(p) for p in expected_files(step, root) if not p.is_file()]
        rcs[step.name] = rc
        S.write_once(d/'steps'/f'{step.name}.done.json', dict(step=step.name, rc=rc, error=error, missing=missing, seconds=system.now()-tick, uptime_after=system.uptime(), utc=run.utc()))
        log(f'{step.name}: exit {rc}{f", {error}" if error else ""}{f", missing {missing}" if missing else ""}')
        if rc != 0 or missing:
            raise run.Stop(f'{step.name} exited {rc}{f" ({error})" if error else ""}{f" and left no {missing}" if missing else ""}: the queue stops here (nothing is rerun; C22.H8: 不过（待查）, a new root)')
        system.sleep(o.gap_seconds)
    report = read(root/f'{REPORT}.json') or {}
    S.write_once(d/'queue_end.json', dict(rc=rcs, summary=report.get('summary'), utc=run.utc()))
    log(f"check: {report.get('summary')}.  Handed to the lead")
    return 0


def queue(o, system=None, *, results=None, stage=None):
    root = Path(o.root)
    problem = root_problem(root, results)
    if problem:                                         # before the lock: nothing is created
        print(problem, file=sys.stderr)
        return 2
    d = v3_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    lock = open(d/'queue.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f'another driver is on {d}', file=sys.stderr)
        return 2

    def log(text):
        with open(d/'queue.log', 'a') as f:
            f.write(f'{run.utc()} {text}\n')
    try:
        return _queue_records(o, system, root, d, log, results=results, stage=stage)
    finally:
        lock.close()                                    # revision 3: the lock released when the queue ends, whatever still refers to its frame


def _queue_records(o, system, root, d, log, *, results=None, stage=None):
    """queue_body and the records of a STOP (kind gate, stop or exception; the steps launched)."""
    try:
        return queue_body(o, System() if system is None else system, root, log, results=results, stage=stage)
    except run.Stop as stop:
        kind = 'gate' if isinstance(stop, GateClosed) else 'stop'                        # revision 3 (B2): the machine gate, or anything else (to BZ)
        log(f'STOP ({kind}): {stop}')
        S.write_once(numbered(d, 'queue_stop'), dict(kind=kind, reason=str(stop), launched=launched_names(root), utc=run.utc()))
        return 8
    except Exception as exc:                            # noqa: BLE001 - SM3 revision 2 (review I2): whatever stopped the queue is recorded
        log(f'STOP (an exception of the queue): {type(exc).__name__}: {exc}')
        S.write_once(numbered(d, 'queue_stop'), dict(kind='exception', reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:], launched=launched_names(root),
                                                     utc=run.utc()))
        return 8


def launched_names(root):
    """The steps launched so far (v3/steps/*.launched.json): any -> the root is used (C22.H8: a new root); none, and every earlier stop the machine gate's (kind gate) -> the same start
    string may start the queue again (revision 3, the lead's ruling B2: a refusal at the start or an exception goes to BZ)."""
    return sorted(p.name for p in (v3_dir(root)/'steps').glob('*.launched.json'))


# ---- the check ---------------------------------------------------------------------------------------------------------------------------------
PASS, FAIL, OPEN, DESCRIBE = S.PASS, S.FAIL, S.OPEN, S.DESCRIBE


def item(key, name, criterion, verdict, values, note=None):
    return dict(item=key, name=name, criterion=criterion, verdict=verdict, values=values, note=note)


def judged(key, name, criterion, checks, values=None):
    """过 when every check holds; otherwise 不过（待查） (C22.H8 修订 7a: no reason guessed, nothing rerun)."""
    return item(key, name, criterion, PASS if checks and all(v is True for v in checks.values()) else OPEN, dict(values or {}, checks=checks),
                note=None if checks and all(v is True for v in checks.values()) else '不同：只记录，不解释原因；不过就不解封（C22.H8）')


def missing(key, name, criterion, what):
    return item(key, name, criterion, FAIL, dict(missing=what), note='a record this item reads is missing')


def sealed_paths():
    return set(_review().hc.sealed_list_paths())


def sealed_checks(opened_realpaths, sealed):
    """Every sealed list opened (C23.2's four, each process, counted only) and no other file of their directories."""
    dirs = {str(Path(p).parent) for p in sealed}
    others = sorted(p for p in opened_realpaths if str(Path(p).parent) in dirs and p not in sealed)
    return dict(all_four=bool(sealed) and sealed <= set(opened_realpaths), no_other_file_of_their_directories=not others), others


def latest_unit(root, unit):
    """The newest record directory of a PQ unit (run_pqh.launch: pq/units/<unit>_<utc>/), its unit_reading.json."""
    dirs = sorted((Path(root)/'pq'/'units').glob(f'{unit}_*'))
    if not dirs:
        return None, None
    return dirs[-1], read(dirs[-1]/'unit_reading.json')


def unit_reading_ok(reading):
    return isinstance(reading, dict) and reading.get('status') is None and reading.get('returncode') == 0


def v1_copies(root):
    criterion = ('each segment\'s copy.json: every method-side file of the A arm the same before, after and in the copy (sha256), output/ exactly the four files, no evaluation record; the copy\'s '
                 'final.npz state hash = its line.json\'s = the injection\'s target; line status hx_certified; context.json the recorded one: (ii) the frozen values (settings_frozen true, no '
                 'test_context), (iv-*) exactly injection_settings(segment, target) (settings_frozen false, test_context marked); code_head the checkout\'s')
    batch = read(v3_dir(root)/'batch.json') or {}
    rels, problems = source_listing(S.run_dir(root, SOURCE_ARM))
    checks, rows = {}, {}
    for segment in SEGMENTS:
        doc = read(copy_record_path(root, segment))
        ctx_path = cell_dir(root, segment)/'context.json'
        ctx = read(ctx_path)
        if doc is None or ctx is None:
            return missing('V1', 'copies', criterion, [str(p) for p, x in ((copy_record_path(root, segment), doc), (ctx_path, ctx)) if x is None])
        target = doc.get('target_state_hash')
        frozen = segment == 'ii'
        want = he.frozen_settings() if frozen else injection_settings(segment, target)
        checks[f'{segment}_files'] = doc.get('all_same') is True and all(r.get('same') is True for r in doc.get('files') or []) and sorted(r['file'] for r in doc.get('files') or []) == sorted(rels) \
            and not problems
        checks[f'{segment}_target'] = isinstance(target, str) and target == doc.get('line_state_hash') and doc.get('line_status') == he.HX
        checks[f'{segment}_context'] = (sha_file(ctx_path) == (doc.get('context') or {}).get('sha256') and he.canonical(ctx.get('settings')) == he.canonical(want)
                                        and ctx.get('settings_frozen') is frozen and (ctx.get('test_context') is None) == frozen and ctx.get('check_identity') is True
                                        and ctx.get('code_head') == batch.get('code_sha') and Path(ctx.get('out', '')) == cell_dir(root, segment))
        rows[segment] = dict(n_files=len(doc.get('files') or []), target_state_hash=target, line_status=doc.get('line_status'), settings_sha256=ctx.get('settings_sha256'),
                             settings_frozen=ctx.get('settings_frozen'), not_copied=doc.get('not_copied'))
    return judged('V1', 'copies', criterion, checks, dict(segments=rows, source_problems=problems))


def v2_untouched(root, results=None):
    criterion = 'C22.H8 修订 7c: the A arm\'s original directory and he_smoke_v2: every file\'s sha256 now the same as in v3/baseline.json (written before the copies); nothing added or removed'
    base = read(v3_dir(root)/'baseline.json')
    if base is None:
        return missing('V2', 'untouched', criterion, [str(v3_dir(root)/'baseline.json')])
    a_now, v2_now = tree_rows(S.run_dir(root, SOURCE_ARM)), tree_rows(Path(base['he_smoke_v2']['root']))

    def diff(old, new):
        old, new = old or {}, new or {}
        return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    a_diff, v2_diff = diff(base['a_arm']['files'], a_now), diff(base['he_smoke_v2']['files'], v2_now)
    checks = dict(a_arm=base['a_arm']['files'] is not None and not a_diff, he_smoke_v2=not v2_diff)
    return judged('V2', 'untouched', criterion, checks, dict(a_arm_files=len(a_now or {}), he_smoke_v2_files=len(v2_now or {}), a_arm_differs=a_diff[:50], he_smoke_v2_differs=v2_diff[:50]))


def v3_members(root, sha):
    criterion = ('PQ (i) (smoke_plan v3 §2 (i)): the unit\'s reading status null and exit 0; members_record.json m5_passed true, both subjects m5 true; dev_reference.equal true (V against '
                 'pq_v1/members bit for bit); opened_files_members.json m6 true, no breach; the four sealed lists among its opened files and no other file of their directory; the lock '
                 'pq/pq_members_dev.lock there; checks.code clean, the checkout <root>/code, the commit the A segment\'s; checks.imports.problems empty; no pq_refused*.json.  The runtime caches '
                 'used, the device peak, the seconds: described')
    pins = _pins()
    pq = Path(root)/'pq'
    rec, opened = read(pq/'members'/MEMBERS_RECORD), read(pq/'members'/OPENED_MEMBERS)
    unit_dir, reading = latest_unit(root, pins.DEV_UNITS['members'])
    if rec is None or opened is None:
        return missing('V3', 'pq_members', criterion, [str(p) for p, x in ((pq/'members'/MEMBERS_RECORD, rec), (pq/'members'/OPENED_MEMBERS, opened)) if x is None])
    sealed = sealed_paths()
    sealed_ok, others = sealed_checks([os.path.realpath(f['path']) for f in opened.get('files') or []], sealed)
    code = (rec.get('checks') or {}).get('code') or {}
    subjects = rec.get('subjects') or {}
    checks = dict(unit=unit_reading_ok(reading), m5=rec.get('m5_passed') is True and sorted(subjects) == sorted(pins.DEV_SMOKE_SUBJECTS) and all(r.get('m5') is True for r in subjects.values()),
                  dev_reference=(rec.get('dev_reference') or {}).get('equal') is True, m6=opened.get('m6') is True and opened.get('breaches') == [], **{f'sealed_{k}': v for k, v in sealed_ok.items()},
                  lock=(pq/'pq_members_dev.lock').is_file(), code=code.get('clean') is True and Path(str(code.get('checkout'))) == Path(root)/'code' and code.get('commit') == sha,
                  imports=((rec.get('checks') or {}).get('imports') or {}).get('problems') == [], no_refusal=not list((pq/'members').glob('pq_refused*.json')))
    return judged('V3', 'pq_members', criterion, checks, dict(unit_dir=None if unit_dir is None else str(unit_dir), reading=reading, other_list_files=others,
                                                            described=dict(device_peak=rec.get('device_peak'), seconds=rec.get('seconds'), n_files=opened.get('n_files'),
                                                                           rules=opened.get('rules'), runtime_cache_dirs=pins.RUNTIME_CACHE_DIRS,
                                                                           note='the lead\'s reading of C22.H4-3 修订 7c (11-14, B1): RUNTIME_CACHE_DIRS as it is at P0 is the pin; a cache '
                                                                                'directory used outside it is a breach (m6 false: V3 does not pass); the directories used are described, the '
                                                                                'constant is not changed after the smoke (tightening it is a P0′)')))


def review_checks(root, segment, sha):
    """The records every review segment has: (checks, values, documents) or a missing list."""
    cell = cell_dir(root, segment)
    rev = cell/'review'
    doc, term, spy = read(rev/'review.json'), read(cell/'terminal.json'), read(rev/'opened_files.json')
    if doc is None or term is None or spy is None:
        return None, [str(p) for p, x in ((rev/'review.json', doc), (cell/'terminal.json', term), (rev/'opened_files.json', spy)) if x is None]
    sealed = sealed_paths()
    sealed_ok, others = sealed_checks(spy.get('opened_realpath') or [], sealed)
    checks = dict(finalized=doc.get('finalized') is True and doc.get('mode') == 'dev', terminal=term.get('line_status') == doc.get('line_status') and (term.get('review_record') or {}).get('sha256') == sha_file(rev/'review.json'),
                  code_head=doc.get('code_head') == sha, spy=spy.get('passed') is True and spy.get('breaches') == [] and spy.get('positive_condition') is True and spy.get('evaluation_records_opened') == [],
                  **{f'sealed_{k}': v for k, v in sealed_ok.items()}, no_stop_record=not list(rev.glob('review_not_finalized*.json')))
    values = dict(line_status=doc.get('line_status'), rounds=len(doc.get('rounds') or []), consistent=doc.get('consistent'), state_hash=doc.get('state_hash'), other_list_files=others,
                  n_sealed_list_opens=spy.get('n_sealed_list_opens'), unit=doc.get('unit'),
                  described=dict(context_settings_are_pinned=doc.get('context_settings_are_pinned'), review_certificate_in=doc.get('review_certificate_in'), settings_sha256=doc.get('settings_sha256'),
                                 rule='the development mode of review_pair checks neither against the pins (C22.H1-7 last item: recorded, not refused)'))
    unit = doc.get('unit') or {}
    checks['unit_probes'] = all(isinstance(unit.get(k), dict) and bool(unit.get(k)) for k in ('start', 'end'))           # SM3 revision 2 (review minor 3): the probes are there
    return (checks, values, dict(doc=doc, term=term, spy=spy)), None


def v4_review_ii(root, sha):
    criterion = ('(ii) (smoke_plan v3 §2 (ii)): review.json finalized in the development mode, consistent in one round, the line status the copy\'s (hx_certified) in review.json and terminal.json, '
                 'terminal.json naming this review.json; code_head the checkout\'s; opened_files.json passed with its positive condition, no breach, no evaluation record; the four sealed lists '
                 'among its opened files and no other file of their directory; no review_not_finalized; the copy\'s method-side files the same as copied (sha256); review.json\'s unit.start and '
                 'unit.end probes there (revision 2); context_settings_are_pinned, review_certificate_in and settings_sha256 described (the development mode checks neither)')
    got, absent = review_checks(root, 'ii', sha)
    if got is None:
        return missing('V4', 'review_ii', criterion, absent)
    checks, values, docs = got
    copy = read(copy_record_path(root, 'ii')) or {}
    cell = cell_dir(root, 'ii')
    checks.update(consistent=docs['doc'].get('consistent') is True and values['rounds'] == 1 and docs['doc'].get('line_status') == copy.get('line_status') == he.HX,
                  copy_unchanged=all(sha_file(cell/r['file']) == r['sha256_dest'] for r in copy.get('files') or []) and bool(copy.get('files')))
    return judged('V4', 'review_ii', criterion, checks, values)


def pq_out_checks(root, segment, sha):
    """PQ's out unit on one segment's copy (smoke_plan v3 §2 (ii) PQ, §2 (iv) PQ): (checks, values) or a missing list."""
    pins = _pins()
    small = pq_dir(root, segment)
    cell = cell_dir(root, segment)
    rec, pqj, opened = read(small/OUT_RECORD), read(small/S.PAIR/PQ_JSON), read(small/OPENED_OUT)
    line = read(cell/'output'/'line.json') or {}
    review = read(cell/'review'/'review.json') or {}
    if rec is None or pqj is None or opened is None:
        return None, [str(p) for p, x in ((small/OUT_RECORD, rec), (small/S.PAIR/PQ_JSON, pqj), (small/OPENED_OUT, opened)) if x is None]
    unit_dir, reading = latest_unit(root, pins.DEV_OUT_UNITS[segment])
    controls = rec.get('controls') or {}
    hm = pqj.get('hm') or {}
    status = line.get('line_status')
    if status == he.HX:
        binding = hm.get('rebuilt_equals_final') is True and hm.get('labels_checked_against_run_json') is True
    else:
        fallback = ((pqj.get('method') or {}).get('fallback') or {})
        binding = (hm.get('rebuilt_equals_final') is None and fallback.get('kind') == ((line.get('fallback') or {}).get('kind')) and fallback.get('state_hash') == review.get('state_hash')
                   and fallback.get('certificate_passed') is True)
    manifest = small/PQ_MANIFEST
    rows = [ln.split('  ', 1) for ln in manifest.read_text(encoding='utf-8').splitlines()] if manifest.is_file() else []
    launched = read(v3_dir(root)/'steps'/f"{ {'ii': 'ii-pq', 'iv-b': 'ivb-pq', 'iv-c': 'ivc-pq'}[segment] }.launched.json") or {}
    before = ((launched.get('before') or {}).get('cell')) or {}
    now = tree_rows(cell) or {}
    unchanged = bool(before) and all(now.get(k) == v for k, v in before.items()) and all(now.get(k) == before.get(k) for k in ('output/line.json', 'review/review.json', 'terminal.json'))
    code = (rec.get('checks') or {}).get('code') or {}
    members = [m for m in (opened.get('pair_members') or {}).values()]
    sealed_ok, others = sealed_checks([os.path.realpath(f['path']) for f in opened.get('files') or [] if isinstance(f, dict) and isinstance(f.get('path'), str)], sealed_paths())     # C22.H8 (ii) 修订 7d (review I1)
    checks = dict(unit=unit_reading_ok(reading), complete=rec.get('complete') is True and all(controls.get(k) is True for k in ('hm1', 'hm2', 'hm3', 'hm4', 'hm5', 'hm6')),
                  line_status=pqj.get('line_status') == status == review.get('line_status'), binding=binding,
                  d_sha256=(pqj.get('sampling') or {}).get('d_sha256') == ((line.get('output') or {}).get('disp') or {}).get('d_sha256') is not None,
                  m6=opened.get('m6') is True and opened.get('breaches') == [] and bool(members) and all(set(m) <= {'fixed', 'moving', 'initial_u'} for m in members),
                  manifest=bool(rows) and all(len(r) == 2 and Path(r[1]).is_file() and sha_file(r[1]) == r[0] for r in rows), d114_cell_unchanged=unchanged,
                  code=code.get('clean') is True and Path(str(code.get('checkout'))) == Path(root)/'code' and code.get('commit') == sha, no_refusal=not list(small.glob('pq_refused*.json')),
                  **{f'sealed_{k}': v for k, v in sealed_ok.items()})
    values = dict(unit_dir=None if unit_dir is None else str(unit_dir), reading=reading, line_status=status, hm=hm, n_manifest_rows=len(rows), state_hash=review.get('state_hash'), other_list_files=others,
                  numbers=pqj.get('numbers'), described='S35^ and the like are the method\'s own numbers (C22.H6-5), never in a Dice table')
    return (checks, values), None


def v5_pq_ii(root, sha):
    criterion = ('(ii) PQ (smoke_plan v3 §2 (ii) PQ, items 1-8): the unit\'s reading status null and exit 0; pq_record.json complete, controls hm1-hm6 true; pq.json hm.rebuilt_equals_final and '
                 'labels_checked_against_run_json true (hx_certified; Hm3 labels from the copy\'s t88/run.json), sampling.d_sha256 the copy\'s line.json output.disp.d_sha256; '
                 'opened_files_out.json m6, no breach, pair.npz read for fixed, moving, initial_u only; the four sealed lists among its files (realpath) and no other file of their directory '
                 '(C22.H8 (ii) 修订 7d; revision 2); every line of pq_outputs_manifest.txt the file\'s sha256; the copy\'s files the same before and after (D114 (a)); checks.code clean, the '
                 'checkout, the commit; no pq_refused')
    got, absent = pq_out_checks(root, 'ii', sha)
    if got is None:
        return missing('V5', 'pq_ii', criterion, absent)
    return judged('V5', 'pq_ii', criterion, *got)


# ---- the lead's ruling 5: PQ's outputs through the evaluation entry's own reading functions (development mode), and PQ's /mnt/d root spelled three times --------
def mnt_d_spellings(repo=None):
    """The three places that spell PQ's roots on the hold-out (decision_log 2026-11-10 ~11:16 PM (b): "真实 /mnt/d 路径写法三处一致"): the arm table's pq section (configs/frozen/
    holdout_B_v1_arm_table.json, what bayesreg.eval.holdout reads and its spy watches), PQH's hold-out layout (pqh_inputs.holdout_layout, built without the pair list), holdout_common's roots
    (HOLDOUT_ROOT and REAL_BIG, under pqh_pins.PQ_ARM, the write guard's real root).  (record, problems).  Nothing of the hold-out is read: the pair list is not opened."""
    from bayesreg.eval import holdout as H
    pins = _pins()
    import pqh_inputs as hin
    hc = _review().hc
    repo = Path(core.ROOT if repo is None else repo)
    problems = []
    raw = (repo/H.ARM_TABLE_REL).read_bytes()
    table = H.validate_arm_table(H.strict_json(raw, H.ARM_TABLE_REL))
    layout = hin.holdout_layout(repo, pairs=())
    rows = dict(arm_table=dict(small=table['pq']['small_root'], big=table['pq']['big_root'], path=H.ARM_TABLE_REL, sha256=hashlib.sha256(raw).hexdigest(), pinned=H.ARM_TABLE_SHA256),
                pqh_layout=dict(small=str(layout.pq_small), big=str(layout.pq_big)),
                holdout_common=dict(small=str(Path(hc.HOLDOUT_ROOT)/pins.PQ_ARM), big=str(Path(hc.REAL_BIG)/pins.PQ_ARM)))
    for key in ('small', 'big'):
        spelled = {name: row[key] for name, row in rows.items()}
        if len(set(spelled.values())) != 1:
            problems.append(f'PQ\'s {key} root is spelled differently: {spelled}')
    if not rows['arm_table']['big'].startswith(str(Path(hc.REAL_BIG))+'/'):
        problems.append(f"the arm table's big root {rows['arm_table']['big']} is not under {hc.REAL_BIG}")
    if str(Path(hc.REAL_BIG)) not in {str(Path(p)) for p in pins.REAL_WRITE_ROOTS}:
        problems.append(f'{hc.REAL_BIG} is not among pqh_pins.REAL_WRITE_ROOTS')
    if H.ARM_TABLE_SHA256 is not None and rows['arm_table']['sha256'] != H.ARM_TABLE_SHA256:
        problems.append(f'{H.ARM_TABLE_REL}: sha256 {rows["arm_table"]["sha256"]} is not the pinned {H.ARM_TABLE_SHA256}')
    return dict(rows, rule='decision_log 2026-11-10 ~11:16 PM CST (b): the real /mnt/d path written the same in the three places'), problems


def entry_reading(root, segment):
    """The evaluation entry's PQ functions on the development outputs of one segment (the lead's ruling 5): the arm table's method arm with PQ's roots replaced by this segment's dev roots;
    the list pq_outputs_manifest.txt parsed with the entry's rule (_ENTRY) as check_pq_release does; `listed` the files' sha256 now (what the D108 manifest would list); then pq_states (present),
    pq_ev1, read_pq_outputs (the bytes against `listed`, the arrays on the grid) and Ev2 (pq.json's sampling.d_sha256, the entry's key, against the d of the copy's disp.npz, float64 C-order).
    Not through open_holdout: its rule one, admission, preregistration, arm table pin and manifest read hold-out files (C22.H8: the smoke reads none); no truth (evaluate_pq is not called).
    SM3 revision 2 (review minor 2): in the development mode Ev1 is nearly a tautology -- `listed` is computed from the same files now, so it only says that PQ's list names each file by its
    sha256 (the manifest check of V5 says it too); what it adds is the entry's own code path.  Besides, the keys the entry's input-lock check reads (pq_input_lock_check: inputs.<side>.
    probs36_sha256, structure_groups_sha256, provenance_sha256 and method_files['t88/field_0600.npz'].sha256) are in pq.json, each a sha256."""
    from bayesreg.eval import fields
    from bayesreg.eval import holdout as H
    small, big = pq_dir(root, segment), pq_big(root, segment)
    raw = (Path(core.ROOT)/H.ARM_TABLE_REL).read_bytes()
    table = H.validate_arm_table(H.strict_json(raw, H.ARM_TABLE_REL))
    method = H.method_arm(table)
    dev_table = dict(table, arms=[method], pq=dict(small_root=str(small), big_root=str(big)))
    fixed, moving = S.PAIR.split('__')
    row = (0, fixed, moving, S.PAIR)
    line = read(cell_dir(root, segment)/'output'/'line.json') or {}
    cells = {(method['name'], S.PAIR): SimpleCell('output' if line.get('line_status') in he.OUTPUT_STATUSES else 'none', line.get('line_status'))}
    text = (small/PQ_MANIFEST).read_text(encoding='utf-8')
    entries, bad = {}, []
    for number, ln in enumerate(text.splitlines(), 1):
        m = H._ENTRY.fullmatch(ln)
        if not m or m.group(2) in entries:
            bad.append(number)
            continue
        entries[m.group(2)] = m.group(1)
    release = dict(method_side='released', entries=entries, passed=not bad, problems=[f'lines {bad}'] if bad else [], list=str(small/PQ_MANIFEST), released_sha256=sha_file(small/PQ_MANIFEST),
                   note='development: no committed release (the entry reads PQ_RELEASE_SHA_REL on the hold-out); the list parsed with the entry\'s rule')
    listed = {path: sha_file(path) for path in entries if Path(path).is_file()}
    states = H.pq_ev1(H.pq_states(dev_table, [row], cells, release), release, listed)
    state = states[S.PAIR]
    out = dict(state=state.get('state'), reason=state.get('reason'), ev1=state.get('ev1'), files=state.get('files'))
    if state.get('state') != 'present':
        return out, [f"pq_states: {state.get('state')} ({state.get('reason')})"]
    grid_shape = tuple(npz_u(cell_dir(root, segment)/'output'/'final.npz').shape[:3])
    document, pq, base = H.read_pq_outputs(state, listed, grid_shape)
    d, _ = fields.read_disp(cell_dir(root, segment)/'output'/'disp.npz', grid_shape)
    d_now = fields.array_sha256(np.ascontiguousarray(d, dtype=np.float64))
    ev2 = dict(pq_json=H.dotted(document, H.PQ_D_SHA256_KEY), disp_npz=d_now, line_json=((line.get('output') or {}).get('disp') or {}).get('d_sha256'))
    lock_keys = {f'inputs.{side}.{key}': H.dotted(document, f'inputs.{side}.{key}') for side in ('fixed', 'moving') for key in H.PQ_INPUT_KEYS.values()}
    lock_keys["method_files['t88/field_0600.npz'].sha256"] = ((document.get('method_files') or {}).get('t88/field_0600.npz') or {}).get('sha256')
    out.update(arrays={k: list(v.shape) for k, v in dict(pq, **base).items()}, ev2=ev2, input_lock_keys=lock_keys,
               ev1_note='development: listed is computed from the same files now (nearly a tautology); the entry\'s code path is what is exercised')
    problems = []
    bad = {k: v for k, v in lock_keys.items() if not (isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v))}
    if bad:
        problems.append(f"the keys of the entry's input-lock check are not sha256 in pq.json: {bad}")
    if (state.get('ev1') or {}).get('passed') is not True:
        problems.append(f"Ev1: {state.get('ev1')}")
    if len(set(ev2.values())) != 1:
        problems.append(f'Ev2: {ev2}')
    return out, problems


@dataclass(frozen=True)
class SimpleCell:
    """The two attributes of bayesreg.eval.holdout's cell that pq_states reads."""
    outcome: str
    status: str


def v6_entry(root, segments=('ii',)):
    criterion = ('the lead\'s ruling 5 (decision_log 2026-11-10 ~11:04 and ~11:16 PM, (b)) in the development mode: for each PQ segment, the evaluation entry\'s own functions on the dev outputs '
                 '(bayesreg.eval.holdout pq_states present, pq_ev1 passed -- nearly a tautology in the development mode --, read_pq_outputs on the grid, Ev2 d the same in pq.json, disp.npz and '
                 'line.json, the keys of the entry\'s input-lock check in pq.json each a sha256); and PQ\'s roots spelled the same in '
                 'the arm table, PQH\'s hold-out layout and holdout_common (/mnt/d/seq_vp_tag_holdout_B_v1/pq).  open_holdout itself is not run here (its first steps read hold-out files): the '
                 'admission, the committed release, the D108 manifest and the spy on the hold-out roots are left to the synthetic tests and the hold-out evaluation (to the lead)')
    checks, values = {}, {}
    try:
        record, problems = mnt_d_spellings()
        checks['mnt_d_three_places'] = not problems
        values['mnt_d'] = dict(record, problems=problems)
    except Exception as exc:                            # noqa: BLE001 - recorded, the item does not pass
        checks['mnt_d_three_places'] = False
        values['mnt_d'] = dict(error=f'{type(exc).__name__}: {exc}'[:1000])
    for segment in segments:
        try:
            out, problems = entry_reading(root, segment)
            checks[f'entry_{segment}'] = not problems
            values[segment] = dict(out, problems=problems)
        except Exception as exc:                        # noqa: BLE001
            checks[f'entry_{segment}'] = False
            values[segment] = dict(error=f'{type(exc).__name__}: {exc}'[:1000])
    return judged('V6', 'pq_through_the_entry', criterion, checks, values)


def v7_replay(root, sha):
    criterion = ('(iii) (smoke_plan v3 §2 (iii)): the unit\'s reading status null and exit 0; p3_replay.json reading 逐位相同, its 10 rows all equal and file_listed; spy_m6 true; checks.code '
                 'clean, the checkout, the commit; no truth, no evaluation record read')
    pins = _pins()
    rec = read(Path(root)/'pq'/'p3_replay'/'p3_replay.json')
    if rec is None:
        return missing('V7', 'p3_replay', criterion, [str(Path(root)/'pq'/'p3_replay'/'p3_replay.json')])
    unit_dir, reading = latest_unit(root, pins.DEV_UNITS['p3_replay'])
    rows = rec.get('rows') or {}
    code = (rec.get('checks') or {}).get('code') or {}
    checks = dict(unit=unit_reading_ok(reading), reading=rec.get('reading') == '逐位相同', rows=len(rows) == 10 and all(r.get('equal') is True and r.get('file_listed') is True for r in rows.values()),
                  spy=rec.get('spy_m6') is True, code=code.get('clean') is True and Path(str(code.get('checkout'))) == Path(root)/'code' and code.get('commit') == sha)
    return judged('V7', 'p3_replay', criterion, checks, dict(unit_dir=None if unit_dir is None else str(unit_dir), reading=reading, error=rec.get('error')))


def injection_records(root, segment, role):
    return [read(p) for p in sorted((segment_root(root, segment)/'injection').glob(f'{role}*.json'))]


def forced_review_item(key, root, segment, sha):
    """(iv-a), (iv-b), (iv-c) (smoke_plan v3 §2 (iv); C22.H8 (iv) 修订 7a-7c): the records of the expected loop."""
    names = {'iv-a': 'forced_iva', 'iv-b': 'forced_ivb', 'iv-c': 'forced_ivc'}
    criterion = {
        'iv-a': ('(iv-a): two rounds; round 1 on the copy\'s array (the target), its review differing from cert.json in `reason` only (the injection\'s value), output/review_mismatch_0.json; '
                 'the third certification\'s child exit 0, output/third_certificate.json of the target passed (the real certificate); round 2 on the same array, differing again, resolved by that '
                 'record; finalized hx_certified; the review injection applied twice, on the target only; the child\'s injection never applied'),
        'iv-b': ('(iv-b): round 1 as (iv-a); the third certificate of the target forced not to pass (FORCED_STATUS) -> output_superseded_0 (its final.npz the target, its third_certificate.json passed '
                 'false); fallback_configc_after_third_0/fallback.json kind configC, its candidate certified by the real certificate (passes, the new output\'s state hash), launched in '
                 'tag-he-c3-0438-0439 with D0\'s unit (MemoryMax 12G, MemorySwapMax 4G, CPUQuota 1200 %, OOMPolicy continue), its two probes (unit_probe_start.json, unit_probe_end.json) there; '
                 'round 2 on the new array, consistent; finalized hx_fallback_configC '
                 'bound to that fallback.json; the review injection applied once, the child\'s once, both on the target only'),
        'iv-c': ('(iv-c): as (iv-b) to the failed third certificate; configuration C\'s launcher started nothing (he_smoke_v3_no_candidate.json in fallback_configc_after_third_0/run, the '
                 'fallback.json without a candidate); fallback_identity_after_third_0/fallback.json kind identity certified by the real certificate (u = 0); round 2 consistent; finalized '
                 'hx_fallback_identity; the launcher injection applied once')}[segment]
    criterion += ('; in every (iv) segment, as V4: the review unit\'s identity, code_head, spy, the four sealed lists, no review_not_finalized, review.json\'s unit.start and unit.end probes '
                  'there (revision 2); one third certification\'s child, exit 0, its opened_files_third passed')
    got, absent = review_checks(root, segment, sha)
    if got is None:
        return missing(key, names[segment], criterion, absent)
    checks, values, docs = got
    doc = docs['doc']
    rounds = doc.get('rounds') or []
    copy = read(copy_record_path(root, segment)) or {}
    target = copy.get('target_state_hash')
    cell = cell_dir(root, segment)
    review_inj, third_inj = injection_records(root, segment, 'review'), injection_records(root, segment, 'third')
    r0 = rounds[0] if rounds else {}
    checks['two_rounds'] = len(rounds) == 2 and r0.get('state_hash') == target and r0.get('consistent') is False and set((r0.get('differences') or {})) == {PERTURB_FIELD} \
        and (r0.get('review') or {}).get(PERTURB_FIELD) == PERTURB_VALUE and r0.get('third_returncode') == 0
    rec = review_inj[-1] if review_inj else {}
    third = third_inj[-1] if third_inj else {}
    third_files = sorted(p.name for p in (cell/'review').glob('opened_files_third*.json'))
    checks['third_child_spy'] = len(third_files) == 1 and (read(cell/'review'/third_files[0]) or {}).get('passed') is True if third_files else False
    checks['one_child'] = len(third_inj) == 1 and third.get('exit') == 0
    if segment == 'iv-a':
        r1 = rounds[1] if len(rounds) == 2 else {}
        tc = read(cell/'output'/'third_certificate.json') or {}
        checks.update(mismatch=(cell/'output'/'review_mismatch_0.json').is_file() and not (cell/'output'/'review_mismatch_1.json').exists(),
                      third_passed=tc.get('passed') is True and tc.get('output_state_hash') == target and (tc.get('certificate') or {}).get('status') == he.CERT_PASS,
                      resolved=r1.get('state_hash') == target and r1.get('consistent') is False and (r1.get('third_certificate') or {}).get('passed') is True,
                      line_status=doc.get('line_status') == docs['term'].get('line_status') == he.HX and doc.get('state_hash') == target,
                      injections=rec.get('target_state_hash') == target and rec.get('n_applied') == 2 and rec.get('n_on_target') == 2 and rec.get('other_states') == []
                      and third.get('kind') is None and third.get('n_applied') == 0 and third.get('n_on_target') == 1)
    else:
        sup = cell/'output_superseded_0'
        tc = read(sup/'third_certificate.json') or {}
        fb_c = read(cell/'fallback_configc_after_third_0'/'fallback.json') or {}
        r1 = rounds[1] if len(rounds) == 2 else {}
        new_hash = doc.get('state_hash')
        superseded_ok = sup.is_dir() and (sup/'final.npz').is_file() and he.array_sha256(npz_u(sup/'final.npz')) == target and (sup/'review_mismatch_0.json').is_file()
        checks.update(superseded=superseded_ok,
                      third_forced=tc.get('passed') is False and tc.get('output_state_hash') == target and (tc.get('certificate') or {}).get('status') == FORCED_STATUS
                      and (tc.get('certificate') or {}).get('reason') == FORCED_REASON,
                      round2=r1.get('state_hash') == new_hash != target and r1.get('consistent') is True and (r1.get('fallback_json') or {}).get('state_hash') == new_hash)
        if segment == 'iv-b':
            cand = next((c for c in fb_c.get('candidates') or [] if c.get('passed')), {})
            launch = fb_c.get('launch') or {}
            argv = launch.get('argv') or []
            checks.update(configc=fb_c.get('kind') == 'configC' and fb_c.get('state_hash') == new_hash and he.passes(fb_c.get('certificate') or {}) and cand.get('state_hash') == new_hash,
                          tag_he_c3=launch.get('kind') == 'unit' and launch.get('unit') == f'{R.THIRD_UNIT_PREFIX}-{S.SHORT}' and all(r in argv for r in fc.UNIT_RESOURCES) and fc.UNIT_OOM_POLICY in argv,
                          tag_he_c3_probes=all(isinstance((launch.get('probes') or {}).get(n), dict) and bool(launch['probes'][n]) for n in fc.PROBE_FILES),     # review minor 3
                          line_status=doc.get('line_status') == docs['term'].get('line_status') == he.FALLBACK_C,
                          injections=rec.get('target_state_hash') == target and rec.get('n_applied') == 1 and rec.get('n_on_target') == 1 and new_hash in (rec.get('other_states') or [])
                          and third.get('kind') == 'fail' and third.get('n_applied') == 1 and third.get('n_on_target') == 1 and new_hash in (third.get('other_states') or []))
            values.update(tag_he_c3=dict(unit=launch.get('unit'), returncode=launch.get('returncode'), probes=launch.get('probes'), seconds=launch.get('seconds')), candidates=fb_c.get('candidates'))
        else:
            fb_i = read(cell/'fallback_identity_after_third_0'/'fallback.json') or {}
            zero = he.array_sha256(np.zeros(npz_u(cell/'output'/'final.npz').shape))
            launcher = third.get('launcher') or {}
            checks.update(no_candidate=(cell/'fallback_configc_after_third_0'/'run'/NO_CANDIDATE_RECORD).is_file() and (fb_c.get('launch') or {}).get('kind') == NO_CANDIDATE_KIND
                          and fb_c.get('state_hash') is None and bool(fb_c.get('error')),
                          identity=fb_i.get('kind') == 'identity' and fb_i.get('state_hash') == new_hash == zero and he.passes(fb_i.get('certificate') or {}),
                          line_status=doc.get('line_status') == docs['term'].get('line_status') == he.FALLBACK_ID,
                          injections=rec.get('target_state_hash') == target and rec.get('n_applied') == 1 and rec.get('n_on_target') == 1 and third.get('kind') == 'fail'
                          and third.get('n_applied') == 1 and third.get('n_on_target') == 1 and launcher.get('n_applied') == 1 and launcher.get('n_calls') == 1)
    values.update(target_state_hash=target, injection=dict(review=rec, third=third))
    return judged(key, names[segment], criterion, checks, values)


def pq_iv_item(key, root, segment, sha):
    criterion = (f'PQ on ({segment})\'s fallback pair (the lead\'s default "加"; smoke_plan v3 §2 (iv) PQ): as V5, with Hm1 by C22.H4-1\'s row for a fallback pair (rebuilt_equals_final null; '
                 'pq.json\'s fallback kind the line\'s, its state hash the review\'s, its certificate passed)')
    got, absent = pq_out_checks(root, segment, sha)
    if got is None:
        return missing(key, f'pq_{segment.replace("-", "")}', criterion, absent)
    return judged(key, f'pq_{segment.replace("-", "")}', criterion, *got)


def v13_units(root):
    criterion = ('every review unit\'s start string carries he_core.EVALUATION_UNIT once (he_core.unit_argv_problems) with OOMPolicy continue and no GPU; every step\'s MemAvailable read when '
                 'its gate opened (launched.json mem_available_gib_at_gate) and the threshold written at least its gate (review units >= 10 GiB, C22.H1-8; revision 2 review minor 4, revision 3); '
                 'the reading again at launch, the units\' probes and the PQ units\' readings described')
    checks, rows = {}, {}
    for step in STEPS:
        launched = read(v3_dir(root)/'steps'/f'{step.name}.launched.json')
        if launched is None:
            return missing('V13', 'units', criterion, [f'v3/steps/{step.name}.launched.json'])
        if step.kind in ('review', 'forced-review', 'check'):
            argv = launched.get('argv') or []
            checks[step.name] = he.unit_argv_problems(argv, EVALUATION) == [] and '--property=OOMPolicy=continue' in argv and '--setenv=CUDA_VISIBLE_DEVICES=' in argv
        at_gate = launched.get('mem_available_gib_at_gate')                                 # revision 3 (diff review 2): the reading the gate opened on; the one read again at launch is described
        checks[f'{step.name}_gate'] = (isinstance(at_gate, (int, float)) and not isinstance(at_gate, bool) and at_gate >= step.mem_gib
                                       and isinstance(launched.get('gate_mem_available_gib'), (int, float)) and launched['gate_mem_available_gib'] >= step.mem_gib)
        rows[step.name] = dict(unit=launched.get('unit'), gate_mem_available_gib=launched.get('gate_mem_available_gib'), mem_available_gib_at_gate=at_gate,
                               mem_available_gib_at_launch=launched.get('mem_available_gib_at_launch'),
                               seconds=(read(v3_dir(root)/'steps'/f'{step.name}.done.json') or {}).get('seconds'))
        if step.segment and step.kind in ('review', 'forced-review'):
            rows[step.name]['probe'] = ((read(cell_dir(root, step.segment)/'review'/'review.json') or {}).get('unit'))
    return judged('V13', 'units', criterion, checks, dict(steps=rows))


def check(root, *, results=None):
    """The reading of v3 (items V1-V13); writes <root>/smoke_v3_report(.json|.md) (numbered if there).  Reads no truth and no evaluation record."""
    root = Path(root)
    batch = read(v3_dir(root)/'batch.json') or {}
    sha = batch.get('code_sha')
    items = []
    for fn in (lambda: v1_copies(root), lambda: v2_untouched(root, results), lambda: v3_members(root, sha), lambda: v4_review_ii(root, sha), lambda: v5_pq_ii(root, sha),
               lambda: v6_entry(root, ('ii', 'iv-b', 'iv-c')), lambda: v7_replay(root, sha), lambda: forced_review_item('V8', root, 'iv-a', sha),
               lambda: forced_review_item('V9', root, 'iv-b', sha), lambda: pq_iv_item('V10', root, 'iv-b', sha), lambda: forced_review_item('V11', root, 'iv-c', sha),
               lambda: pq_iv_item('V12', root, 'iv-c', sha), lambda: v13_units(root)):
        try:
            items.append(fn())
        except Exception as exc:                        # noqa: BLE001 - an item that cannot be read does not pass
            items.append(item(f'V{len(items)+1}', 'unreadable', 'the item raised', FAIL, dict(error=f'{type(exc).__name__}: {exc}'[:2000])))
    counts = {v: sum(i['verdict'] == v for i in items) for v in (PASS, FAIL, OPEN, DESCRIBE)}
    summary = '; '.join(f"{i['item']} {i['verdict']}" for i in items)
    plan = batch.get('plan') or {}
    report = dict(schema=SCHEMA.format('report'), root=str(root), pair=S.PAIR, code_sha=sha, plan=plan, a_segment=batch.get('a_segment'), items=items, counts=counts, summary=summary,
                  wording='开发集单次（0438-0439）；输出是认证终态，不是正式状态，不是 F_B；(iv) 的结果是注入的强制不一致，不是方法的读数；PQ 的数是方法自己的数，不进 Dice 表；不读真值、不读任何评测记录',
                  lead_checks=['batch.code_sha equals the P0 commit in decision_log (C24.3-10; the HEAD of <root>/code)', f"plan.sha256 {plan.get('sha256')} equals the one pinned in decision_log"],
                  utc=he.utc())
    k = 0
    while (root/(REPORT+('' if k == 0 else f'_{k}')+'.json')).exists():
        k += 1
    name = REPORT+('' if k == 0 else f'_{k}')
    S.write_once(root/f'{name}.json', report)
    with (root/f'{name}.md').open('x', encoding='utf-8') as f:
        f.write(report_markdown(report))
    print(f'smoke v3 check: {summary}')
    return report


def report_markdown(report):
    lines = [f"# he_smoke_v3 D120 段读数（{report['pair']}）", '', f"> check 写于 {report['utc']}（UTC）；代码 {report['code_sha']}；根目录 {report['root']}；计划 {report['plan'].get('sha256')}。",
             f"> {report['wording']}。", '', f"计数：{'、'.join(f'{k} {v}' for k, v in report['counts'].items())}。", '', '| 项 | 名称 | 判定 | 判据 | 核对 |', '|---|---|---|---|---|']
    for i in report['items']:
        checks = (i['values'] or {}).get('checks', i['values']) if isinstance(i['values'], dict) else i['values']
        lines.append(f"| {i['item']} | {i['name']} | {i['verdict']} | {i['criterion'][:300].replace('|', '/')} | {S._short(checks).replace('|', '/')} |")
    lines += ['', '总负责人另核：'+'；'.join(report['lead_checks'])+'。', '"不过（待查）"只记录，不解释原因；任一项不过就不解封，开新根再做（C22.H8）。']
    return '\n'.join(lines)+'\n'


# ---- the dry run --------------------------------------------------------------------------------------------------------------------------------
def dry_run(root, *, results=None, out=print):
    """Nothing written, torch not imported, no large file read: the root pin, the A segment's state (its JSON records), the steps and their start strings, the injections, /mnt/d's three spellings."""
    root = Path(root)
    problems = []
    out(f'he_smoke_v3 D120 segments, dry run ({run.utc()}): nothing written, no unit, no GPU, no truth; JSON records only')
    problem = root_problem(root, results)
    out(f'root {root}: {"ok" if problem is None else problem}')
    if problem:
        problems.append(problem)
    if (v3_dir(root)/'steps').is_dir() and list((v3_dir(root)/'steps').glob('*.launched.json')):
        problems.append(f'{v3_dir(root)}: v3 steps were launched there: the queue never starts again')
    stops = [(p.name, (read(p) or {}).get('kind')) for p in sorted(v3_dir(root).glob('queue_stop*.json'))] if v3_dir(root).is_dir() else []
    if [x for x in stops if x[1] != 'gate']:
        problems.append(f'earlier starts stopped for other than the machine gate {stops}: to BZ (B2)')
    a = a_segment(root) if root.is_dir() else dict(problems=['the root is not there: the A segment has not run'], code_sha=None)
    out(f"A segment: code {a.get('code_sha')}; report {a.get('report')}; counts {a.get('counts')}; arm a {a.get('line_status')}; {'ok' if not a['problems'] else a['problems']}")
    problems += [f'A segment: {x}' for x in a['problems']]          # the queue would refuse now (expected before the A segment has ended and read pass)
    out(f'steps ({len(STEPS)}), one at a time, the code at {root/"code"}:')
    sha = a.get('code_sha') or '<A segment commit>'
    for step in STEPS:
        if step.kind in ('review', 'forced-review', 'check'):
            text = shlex.join(unit_argv(step, root/'code', root, sha))
        elif step.kind == 'pq':
            text = shlex.join(pq_argv(step, root/'code', 'dry-run'))+'  then  '+shlex.join(pq_argv(step, root/'code', 'launch'))
        else:
            text = 'in the queue process: v3/baseline.json, then <segment>/a/0438-0439 and <segment>/copy.json for '+', '.join(SEGMENTS)
        out(f'  {step.name:11s} unit {step.unit}; gate: GPU {"free" if step.gpu else "not asked"}, MemAvailable >= {step.mem_gib:g} GiB; expects {[str(p) for p in expected_files(step, root)]}')
        out(f'      {text}')
    out(f'injections: review {PERTURB_FIELD} := {PERTURB_VALUE!r}; third status := {FORCED_STATUS!r}; configuration C launcher {NO_CANDIDATE_KIND} (iv-c); {dict(INJECTIONS)}')
    try:
        record, mnt = mnt_d_spellings()
        out(f"PQ's roots (the lead's ruling 5): {dict((k, v) for k, v in record.items() if k != 'rule')}; {'the same in the three places' if not mnt else mnt}")
        problems += mnt
    except Exception as exc:                            # noqa: BLE001
        problems.append(f'/mnt/d spellings: {type(exc).__name__}: {exc}')
    out('the lead\'s start string of the queue (after the A segment\'s check read 过 and the plan\'s sha256 is in decision_log): '+shlex.join(queue_launch(root)))
    out(f'plan {PLAN_REL}: sha256 {plan_record(core.ROOT)["sha256"]} (in this checkout)')
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


# ---- the entries ---------------------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['--dry-run']:
        d = argparse.ArgumentParser(description='he_smoke_v3: the dry run')
        d.add_argument('--dry-run', action='store_true')
        d.add_argument('--root', required=True)
        return dry_run(Path(d.parse_args(argv).root))
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('launch-string').add_argument('--root', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', required=True)
    q.add_argument('--wait-minutes', type=int, default=30)                  # D126: a review unit's gate waits 30 minutes, then STOP
    q.add_argument('--gap-seconds', type=int, default=15)                   # A5a m14: >= 10 s after the previous unit's end
    q.add_argument('--min-host-gib', type=float, default=15.)
    c = sub.add_parser('check')
    c.add_argument('--root', required=True)
    fr = sub.add_parser('forced-review')
    fr.add_argument('--root', required=True)
    fr.add_argument('--segment', choices=sorted(INJECTIONS), required=True)
    fr.add_argument('--batch-sha', required=True)
    ft = sub.add_parser('forced-third')
    ft.add_argument('--context', required=True)
    ft.add_argument('--root', required=True)
    ft.add_argument('--segment', choices=sorted(INJECTIONS), required=True)
    o = ap.parse_args(argv)
    if o.command == 'launch-string':
        problem = root_problem(o.root)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        print(shlex.join(queue_launch(o.root)))
        return 0
    if o.command == 'queue':
        return queue(o)
    if o.command == 'check':
        problem = S.root_problem(o.root)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        check(o.root)
        return 0
    if o.command == 'forced-review':
        return forced_review_main(Path(o.root), o.segment, o.batch_sha)
    return forced_third_main(o.context, Path(o.root), o.segment)


if __name__ == '__main__':
    sys.exit(main())
