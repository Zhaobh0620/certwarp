"""HES (runs/20261106-he/任务书_HES冒烟驱动.md; pre-registration runs/20261106-he/smoke_plan.md): the GPU smoke of the HE entry on one development pair, 0438-0439, and the reading of its
nine items (docs/bayesreg/notes/HE_防火墙干净入口_20261106.md section 6, revision 1; review docs/bayesreg/reviews/HE_tools_复核_20261106.md section 8).  The entry's own files (run_hx.py, he_core.py,
firewall_c.py, spies.py, spy_pair.py, configc_control.py) are used as they are; nothing here changes a number of the method.
HE2 (runs/20261107-he2/任务书.md item 7; pre-registration runs/20261107-he2/smoke_plan.md; contract C22.9-4 (xv), C22.B4-6, D85): the smoke of the method Hx24-RF05-F1200-v4-C3 on a new root, he_smoke_v2.
The references are HS's T88 field of the pair (hs_v1, C off: pass 0, u_sha256 b6f061da...) and its repair, and P3's C field (p3_v1, pass 2, u_sha256 edf3b039...), its ratios and its repair; items 1-9 as in
v1 (item 3 is C22.B4-6 (a), pass 0), items 10-13 are C22.B4-6 (b)-(e).

    python -B experiments/he/smoke_he.py --dry-run [--root R]
        the units in their order and their start strings, the reference files of items 3-5, 10, 11, 13 with the sha256 their records name (no large file is read or hashed: JSON records and npy headers only),
        the estimates, the identity of this checkout, the admission of the pair (names and manifest only; pair.npz is not opened).  torch is not imported; nothing is written.
    smoke_he.py queue --root /home/rmcfz/seq_vp_tag_results/he_smoke_v2 [--commit SHA] [--wait-minutes 30] [--gap-seconds 5] [--min-host-gib 15]
        the lead starts it (systemd-run, unit tag-he-smoke-queue; queue_launch() prints the string).  The code of the units: a detached checkout <root>/code of HEAD (--commit, if given, must be HEAD).
        One unit at a time, each after the gate (the hold file <root>/hold, no other tag-* or seq-vp-* unit, 1-minute load < 1.5, /home and /mnt/c free, MemAvailable, the GPU free for GPU units):
          (a)  tag-he-smoke-a         run_hx.py pair, the frozen values, launcher unit (production);    tag-he-smoke-a-eval  run_hx.py evaluate
          (b)  tag-he-smoke-b         spy_pair.py, the same pair, its own directory (the supervisor and its children in one process under the spies)
          (c)  tag-he-smoke-c         the forced fallback one through the function interface, launcher unit;   tag-he-smoke-c-eval
          (d)  tag-he-smoke-d         the same, launcher direct (the frozen run in this unit);                 tag-he-smoke-d-eval
               tag-he-smoke-check     smoke_he.py check (CPU)
        A unit that exits non-zero or leaves no expected record stops the queue (queue_stop.json); nothing is rerun: a root in which a unit was launched never starts again.
    smoke_he.py check --root R          read only on the outputs (the truth of the pair once more, for item 4's recomputation, after evaluation.json exists): items 1-13 judged by code,
                                        <root>/smoke_report.json and smoke_report.md (a later check writes smoke_report_<k>.*; nothing is overwritten).
    smoke_he.py forced --launcher unit|direct --out <root>/c|d/0438-0439 [--device cuda]          a unit of the queue: (c) or (d)
    smoke_he.py child t88|repair|third-cert --context FILE                                      the children of a forced run (run_hx.main's own branch, with the forced context's check)
    smoke_he.py evaluate-forced --context FILE                                                  the evaluation unit of a forced run

The forced context (c, d).  The repair cannot fail on demand with the frozen values, and the command line of run_hx.py refuses any context whose values are not the frozen ones under the results root
(cli_context_problem).  So, as the review's new minor 3 and HE revision 2 allow: make_context with forced_settings() (the frozen values but the repair's max_rounds 1 and max_inner 0: round 0 certifies the
tapered T88 field unchanged, so the repair fails, round_cap, whenever that field does not already pass), check_identity true, marked test_context by make_context, settings_frozen false in every record.
Its processes go through run_hx's functions (run_pair, t88_main, repair_main, third_cert_main, evaluate_main) from the entries above, which refuse any context but exactly this one
(forced_context_problem: the forced values byte for byte, the identity checked, the smoke pair and stage, <results>/he_smoke*/c|d/0438-0439 with its launcher).  Exit codes: 0 done, 2 usage or a refused
start, 8 a STOP or a failed check.
"""
import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import run_hx as R  # noqa: E402

he, fc = R.he, R.fc
core = he.core
import run_mc0 as run  # noqa: E402  (experiments/mc0 is on sys.path through he_core: the gate, the checkout, Stop)

SCRIPT = 'experiments/he/smoke_he.py'
SMOKE = HERE/'smoke_he.py'
PY = fc.PY
SCHEMA = 'bayesreg.he.smoke.{}.v1'
PAIR = core.PAIRS[0]                            # OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1: hs_v1 (SHx24, RSHx24), p3_v1 (C, RC) and d0_v1 (A0) have it
SHORT = he.short_name(PAIR)
STAGE = core.STAGE                              # the main checkout's data/three_d/stage_L2R_val_v1 (a checkout of the code has no data/ link)
RESULTS = he.RESULTS
ROOT_PREFIX = 'he_smoke'
DEFAULT_ROOT = RESULTS/'he_smoke_v2'            # HE2: a new root (smoke_plan v1 section 3 item 13: a root in which a unit was launched never starts again; he_smoke_v1 ran on 2026-11-06)
# The references (C22.9-4 (xv), C22.B4-6), written as the contract gives them; the records they come from are checked against them (reference_records, the dry run, the queue's input check, the check)
REF_PASS0_U_SHA256 = 'b6f061daa0eb8ce2131abea9218e2010a9f9e7a1057f91ec25eb227809adde2f'     # hs_v1/runs/SHx24_first5/0438-0439 run.json field.u_sha256 (code d5403e5): the T88 field of v4, C off
REF_PASS2_U_SHA256 = 'edf3b0392269ff347ae486d5f408abaa36a03f867fc4e25af6f6bf3f523dcf04'     # p3_v1/runs/C_first5/0438-0439 run.json checkpoints."600".u_sha256 (code 090c985): C's pass 2
NATIVE = 600                                    # the native step of the references' checkpoints (fx_core.NATIVE; a test compares)
QUEUE_UNIT = 'tag-he-smoke-queue'
REPORT = 'smoke_report'
FORCED_ARMS = {'c': 'unit', 'd': 'direct'}
ARMS = ('a', 'b', 'c', 'd')

# ---- the forced values (c, d) --------------------------------------------------------------------------------------------------------------------
FORCED_REPAIR = dict(max_rounds=1, max_inner=0)
FORCED_NAME = 'HES forced repair failure (not the method; runs/20261106-he/smoke_plan.md)'


def forced_settings():
    """The frozen values but the repair's max_rounds 1 and max_inner 0 (with no inner step, round 0 certifies v = the tapered T88 field: the repair fails, round_cap, unless that field already passes), the
    name, and a note of the change.  Nothing else differs: the T88, the certificate, configuration C, the children's timeouts are the frozen ones."""
    s = he.frozen_settings()
    s['name'] = FORCED_NAME
    s['repair'] = dict(s['repair'], **FORCED_REPAIR)
    s['smoke_forcing'] = dict(brief='runs/20261106-he/任务书_HES冒烟驱动.md (c)(d)', change=f'repair {FORCED_REPAIR} (frozen: max_rounds 5, max_inner 300)',
                              why='force the repair to fail so that fallback one (configuration C) runs on a real pair; settings_frozen false in every record')
    return s


def results_of(results=None):
    return Path(RESULTS if results is None else results)


def run_dir(root, arm):
    return Path(root)/arm/SHORT


def forced_out_problem(out, launcher, results=None):
    """None, or why `out` is not the directory of a forced run: <results>/he_smoke*/<c|d>/0438-0439, c with the unit launcher and d with the direct one."""
    out = Path(out).resolve()
    arm, root = out.parent.name, out.parent.parent
    if out.name != SHORT or arm not in FORCED_ARMS or not root.name.startswith(ROOT_PREFIX) or root.parent != results_of(results).resolve():
        return f'{out}: a forced run writes only to {results_of(results)}/{ROOT_PREFIX}*/<c|d>/{SHORT}'
    if FORCED_ARMS[arm] != launcher:
        return f'{out}: the arm {arm} runs with the launcher {FORCED_ARMS[arm]}, not {launcher}'
    return None


def forced_context(out, *, launcher, device='cuda', results=None, paths=None):
    """The context of (c) or (d): make_context (the function interface) with forced_settings(), the identity checked; make_context marks it test_context."""
    problem = forced_out_problem(out, launcher, results)
    if problem:
        raise ValueError(problem)
    return R.make_context(STAGE, PAIR, out, device=device, settings=forced_settings(), paths=paths, launcher=launcher, check_identity=True)


def forced_context_problem(ctx, results=None):
    """None, or why the forced entries (child, evaluate-forced) may not run this context: they take exactly the forced one, nothing else."""
    if he.canonical(ctx.get('settings')) != he.canonical(forced_settings()):
        return 'the values are not exactly forced_settings()'
    if ctx.get('settings_frozen') is not False or ctx.get('check_identity') is not True or ctx.get('test_context') != R.TEST_CONTEXT:
        return f"settings_frozen {ctx.get('settings_frozen')}, check_identity {ctx.get('check_identity')}, test_context {ctx.get('test_context')!r}: not the forced context"
    if ctx.get('pair') != PAIR or Path(ctx.get('stage_dir', '')) != Path(STAGE):
        return f"the pair {ctx.get('pair')} of {ctx.get('stage_dir')}: the smoke runs {PAIR} of {STAGE} only"
    return forced_out_problem(ctx.get('out', ''), (ctx.get('launcher') or {}).get('kind'), results)


class ForcedChild:
    """run_hx.SubprocessChild for the forced context: python -B smoke_he.py child <stage> --context FILE, the timeouts of the settings, this process's environment."""
    kind = 'subprocess_forced'

    def __call__(self, stage, ctx):
        timeout = {'t88': ctx['settings']['children']['t88_timeout_seconds'], 'repair': ctx['settings']['children']['repair_timeout_seconds']}.get(stage)
        argv = fc.check_launch([PY, '-B', str(SMOKE), 'child', stage, '--context', str(R.context_path(ctx))])
        log = Path(ctx['out'])/f'{stage}.log'
        with log.open('a') as f:
            try:
                return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                f.write(f'\n{he.utc()} killed at the timeout {timeout} s\n')
                return 'timeout'


class ForcedThirdChild:
    """run_hx.ThirdCertChild for the forced context."""

    def __call__(self, ctx):
        argv = fc.check_launch([PY, '-B', str(SMOKE), 'child', 'third-cert', '--context', str(R.context_path(ctx))])
        with (Path(ctx['out'])/'third-cert.log').open('a') as f:
            return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT).returncode


def forced_main(o):
    problem = R.out_problem(o.out) or forced_out_problem(o.out, o.launcher)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    ctx = forced_context(o.out, launcher=o.launcher, device=o.device)
    return R.run_pair(ctx, child=ForcedChild())


def read_forced_context(path):
    ctx = R.read_context(path)
    problem = forced_context_problem(ctx)
    if problem:
        print(f'refused: {problem}', file=sys.stderr)
        return None
    return ctx


def child_main(stage, path):
    """run_hx.main's branch of the stage, with forced_context_problem in place of cli_context_problem."""
    ctx = read_forced_context(path)
    if ctx is None:
        return 8
    if stage in ('t88', 'repair'):
        _, problems = R.check_identity(ctx, ctx['out'], f'identity_{stage}.json')
        if problems:
            return 8
        return R.t88_main(ctx) if stage == 't88' else R.repair_main(ctx)
    return R.third_cert_main(ctx)


def evaluate_forced_main(path):
    ctx = read_forced_context(path)
    if ctx is None:
        return 8
    return R.evaluate_main(ctx, third=ForcedThirdChild())


# ---- the units --------------------------------------------------------------------------------------------------------------------------------------
SUPERVISOR = dict(he.SUPERVISOR_UNIT)
# Revision 1 (review important 1 and minor 7; the lead's ruling): MemoryMax 12G and MemorySwapMax 4G as D0's frozen unit ((d): the frozen run shares the unit with the supervisor), the machine guarded by d's
# gate (MemAvailable >= D_MEM_AVAILABLE_GIB); OOMPolicy=continue, so that a child killed by the OOM killer leaves the supervisor alive (the user manager's default is stop: the whole unit would go);
# 4800 s > the bound of item 7.  HE2 (H2): the values are he_core's (SUPERVISOR_UNIT, in the frozen settings' units), the production supervisor's.
EVALUATION = dict(he.EVALUATION_UNIT)


@dataclass(frozen=True)
class Step:
    name: str
    arm: str
    kind: str                                   # pair | evaluate | spy | forced | evaluate-forced | check
    gpu: bool

    @property
    def unit(self):
        return f'tag-he-smoke-{self.name}'


STEPS = (Step('a', 'a', 'pair', True), Step('a-eval', 'a', 'evaluate', False), Step('b', 'b', 'spy', True), Step('c', 'c', 'forced', True), Step('c-eval', 'c', 'evaluate-forced', False),
         Step('d', 'd', 'forced', True), Step('d-eval', 'd', 'evaluate-forced', False), Step('check', 'check', 'check', False))


def step_argv(step, code, root):
    """The start string of one unit (run_mc0.unit_argv: PYTHONPATH <frozen>:<code>, single-threaded BLAS, the log appended)."""
    code, root = Path(code), Path(root)
    d = run_dir(root, step.arm)
    log = root/step.arm/f'{step.unit}.log'
    pair = ['--stage-dir', str(STAGE), '--pair', PAIR, '--out', str(d), '--device', 'cuda']
    inner, script, res = {
        'pair': (['pair', *pair, '--launcher', 'unit'], 'experiments/he/run_hx.py', SUPERVISOR),
        'spy': ([*pair, '--launcher', 'unit'], 'experiments/he/spy_pair.py', SUPERVISOR),
        'forced': (['forced', '--launcher', FORCED_ARMS.get(step.arm, '-'), '--out', str(d), '--device', 'cuda'], SCRIPT, SUPERVISOR),
        'evaluate': (['evaluate', '--context', str(d/'context.json')], 'experiments/he/run_hx.py', EVALUATION),
        'evaluate-forced': (['evaluate-forced', '--context', str(d/'context.json')], SCRIPT, EVALUATION),
        'check': (['check', '--root', str(root)], SCRIPT, EVALUATION),
    }[step.kind]
    argv = run.unit_argv(step.unit, code, inner, log=log, gpu=step.gpu, numba=res['numba'], quota=res['quota'], runtime=res['runtime'], script=script, memory_max=res['memory_max'], nice=res['nice'])
    if res.get('oom_policy'):                           # revision 1: inserted after MemorySwapMax; run_mc0.unit_argv is not changed (HE2 review minor 14: the evaluation and check units too)
        argv.insert(argv.index('--property=MemorySwapMax=4G')+1, f"--property=OOMPolicy={res['oom_policy']}")
    problems = he.unit_argv_problems(argv, res)         # HE2 review minors 8, 11: the string carries the unit's values (he_core.SUPERVISOR_UNIT, EVALUATION_UNIT), its MemorySwapMax included
    if problems:
        raise ValueError(f'{step.unit}: the start string is not the unit he_core names: {problems}')
    return run.check_launch(argv)


def expected_files(step, root):
    """What a unit must leave: a missing one stops the queue as a non-zero exit does."""
    d = run_dir(root, step.arm)
    return {'pair': [d/'output/line.json'], 'forced': [d/'output/line.json'], 'spy': [d/'spy.json', d/'output/line.json'], 'evaluate': [d/'evaluation.json'],
            'evaluate-forced': [d/'evaluation.json'], 'check': [Path(root)/f'{REPORT}.json']}[step.kind]


def queue_launch(root=DEFAULT_ROOT, commit=None):
    """The lead's start string of the queue, from the checkout whose code prints it (core.ROOT: the tree this file runs from; SM3 revision 3, the plans' review I3 -- the A segment of
    he_smoke_v3 starts from P0's detached checkout, so that the queue's --commit check (queue_body: git rev-parse HEAD in core.ROOT) reads that checkout's HEAD, not the main checkout's, which
    moves with every later document commit).  The working directory and the script are that tree's."""
    return run.check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT}', '--collect', f'--working-directory={core.ROOT}', '--property=MemoryMax=2G', '--nice=5', '--setenv=PYTHONDONTWRITEBYTECODE=1',
                             '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=CUDA_VISIBLE_DEVICES=', PY, '-B', str(core.ROOT/SCRIPT), 'queue', '--root', str(root),
                             *(['--commit', commit] if commit else [])])


GATE = run.Gate(own_units=(f'{QUEUE_UNIT}.service',), load_limit=run.LOAD_LIMIT, gpu_free_mib=15*1024, mem_available_gib=he.OTHER_GATE_GIB, host_floor_gib=15.)     # mc0p's RETRY_GATE (RW, FL) with /mnt/c
D_MEM_AVAILABLE_GIB = he.FROZEN_RUN_GATE_GIB    # revision 1 (the lead): (d)'s gate, 10 GiB; HE2 (the gate of c): every unit in which configuration C's frozen run can start
FROZEN_RUN_KINDS = ('pair', 'spy', 'forced', 'evaluate', 'evaluate-forced')     # HE2: the supervisor (its fallback one: a nested unit for a, b, c, this unit for d) and the evaluation (H3: after a failed third
                                                                               # certificate, a nested unit); only the check cannot start it


def step_gate(step, gate):
    """The gate before a unit.  HE2 (the gate of c; runs/20261107-he2/任务书.md item 6): every unit whose process can start configuration C's frozen run asks MemAvailable >= D_MEM_AVAILABLE_GIB (in v1 only (d)
    did, while (c)'s nested unit has the same 12G); the check the gate as it is.  A gate gives a cgroup no room: it asks the machine to be idle enough at launch (he_core.FROZEN_RUN_GATE_GIB)."""
    return replace(gate, mem_available_gib=max(gate.mem_available_gib or 0., D_MEM_AVAILABLE_GIB)) if step.kind in FROZEN_RUN_KINDS else gate


# ---- small helpers -------------------------------------------------------------------------------------------------------------------------------------
def read_json(path):
    return he.read_json(path)


def write_once(path, value):
    """A record written once and atomically: a temporary file, then a hard link (fails if the record is there), then the temporary removed."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=he._default)+'\n')
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()
    return path


def git_commit(commit, cwd=None):
    done = subprocess.run(['git', 'rev-parse', '--verify', f"{commit or 'HEAD'}^{{commit}}"], cwd=core.ROOT if cwd is None else cwd, capture_output=True, text=True)
    if done.returncode:
        raise run.Stop(f"git rev-parse {commit or 'HEAD'}: {done.stderr.strip()}")
    return done.stdout.strip()


def root_problem(root, results=None):
    root = Path(root)
    if not root.name.startswith(ROOT_PREFIX) or root.resolve().parent != results_of(results).resolve():
        return f'{root}: a smoke root is {results_of(results)}/{ROOT_PREFIX}*'
    return None


def launched_steps(root):
    return sorted(p.name for p in (Path(root)/'steps').glob('*.launched.json'))


class System(run.System):
    """run_mc0.System and what the smoke asks besides (tests replace it)."""

    def uptime(self):
        return fc.uptime()

    def code_identity(self, code):
        return he.identity(code, frozen=core.FROZEN, check_pythonpath=False)


# ---- the references (items 3-5, 10, 11, 13) --------------------------------------------------------------------------------------------------------------
def reference_paths(results=None, stage=None):
    """HE2 (C22.9-4 (xv), C22.B4-6): hs_v1's T88 field and repair of the pair (v4, C off) in the place of fd2_v1's and rw_v1's; p3_v1's C field and repair (v4, C on); d0_v1's A0; the original manifest."""
    r = results_of(results)
    hs = r.joinpath('hs_v1', 'runs', 'SHx24_first5', SHORT)
    hs_r = r.joinpath('hs_v1', 'runs', 'RSHx24_c600', SHORT)
    p3 = r.joinpath('p3_v1', 'runs', 'C_first5', SHORT)
    p3_r = r.joinpath('p3_v1', 'runs', 'RC_c600', SHORT)
    d0 = r.joinpath('d0_v1', 'runs', 'A0', PAIR)
    return dict(hs_dir=hs, hs_run=hs/'run.json', hs_field=hs/he.FIELD_FILE, hs_repair_dir=hs_r, hs_repair=hs_r/'repair.json', hs_metrics=hs_r/'repair_metrics.json',
                p3_dir=p3, p3_run=p3/'run.json', p3_field=p3/he.FIELD_FILE, p3_repair_dir=p3_r, p3_repair=p3_r/'repair.json', p3_metrics=p3_r/'repair_metrics.json', d0_dir=d0,
                d0_classification=d0/'classification.json', manifest=Path(STAGE if stage is None else stage)/PAIR/'manifest.json')


def d0_task(paths):
    """D0 A0's task directory (its protocol.json), its final_native.json and .npz."""
    task = fc.task_dir(paths['d0_dir'])
    return task, task/fc.FINAL_RECORD, task/fc.FINAL


def original_lam(manifest_path):
    """Item 5, outside the method: the one key weights.lam of the ORIGINAL manifest, read with plain json (HE revision 1, minor 8); nothing else of it is used."""
    lam = json.loads(Path(manifest_path).read_text())['weights']['lam']
    return dict(path=str(manifest_path), lam=lam, lam_hex=float(lam).hex(), rule='plain json, the one key weights.lam, in the check (not in a process of the method)')


def p3_ratios(p3):
    """P3's ratios used in pass 1 and pass 2 (run.json passes[k].tissue.ratios_used) and its pass u_sha256s."""
    passes = p3.get('passes') or []
    return dict(u_sha256=[r.get('u_sha256') for r in passes], ratios_used=[(r.get('tissue') or {}).get('ratios_used') for r in passes][1:], last_estimate=(passes[-1].get('estimate') or {}).get('ratios') if passes else None)


def reference_records(results=None, stage=None):
    """The reference values the records name (no large file is read or hashed); KeyError/OSError/ValueError when one is missing, ValueError when a record does not name the contract's hash."""
    p = reference_paths(results, stage)
    hs, hs_r, hs_m = read_json(p['hs_run']), read_json(p['hs_repair']), read_json(p['hs_metrics'])
    p3, p3_r, p3_m = read_json(p['p3_run']), read_json(p['p3_repair']), read_json(p['p3_metrics'])
    classification = read_json(p['d0_classification'])
    task, final_record, final = d0_task(p)
    final_json = read_json(final_record)
    named = ('hs_run', 'hs_repair', 'hs_metrics', 'p3_run', 'p3_repair', 'p3_metrics', 'd0_classification')
    if None in (hs, hs_r, hs_m, p3, p3_r, p3_m, classification, final_json) or not p['hs_field'].is_file() or not p['p3_field'].is_file() or not final.is_file():
        missing = [str(p[k]) for k in named if read_json(p[k]) is None]+[str(x) for x in (p['hs_field'], p['p3_field'], final_record, final) if not x.is_file()]
        raise FileNotFoundError(f'reference records missing: {missing}')
    p3_check = (p3.get('checkpoints') or {}).get(str(NATIVE)) or {}
    ratios = p3_ratios(p3)
    problems = [what for what, ok in (
        (f"hs_v1 run.json field.u_sha256 {hs['field'].get('u_sha256')} is not the contract's {REF_PASS0_U_SHA256}", hs['field'].get('u_sha256') == REF_PASS0_U_SHA256),
        (f"p3_v1 run.json checkpoints.{NATIVE}.u_sha256 {p3_check.get('u_sha256')} or field.u_sha256 {(p3.get('field') or {}).get('u_sha256')} is not the contract's {REF_PASS2_U_SHA256}",
         p3_check.get('u_sha256') == REF_PASS2_U_SHA256 == (p3.get('field') or {}).get('u_sha256')),
        (f"p3_v1's pass 0 {ratios['u_sha256'][:1]} is not hs_v1's T88 field {REF_PASS0_U_SHA256}", ratios['u_sha256'][:1] == [REF_PASS0_U_SHA256]),
        (f"p3_v1's ratios used {ratios['ratios_used']}: two passes with CSF and GM needed", len(ratios['ratios_used']) == 2 and all(isinstance(r, dict) and set(r) == {'CSF', 'GM'} for r in ratios['ratios_used'])),
        ('hs_v1 and p3_v1 repaired other fields than their T88 records name', hs_r['source'].get('field_sha256') == hs['field'].get('sha256') and p3_r['source'].get('field_sha256') == (p3.get('field') or {}).get('sha256')))
        if not ok]
    if problems:
        raise ValueError(f'reference records: {problems}')
    metrics = lambda m: {k: ((m.get('after') or {}).get('metrics') or {}).get(k) for k in ('seg35', 'seg4', 'sdlogj_official')}     # noqa: E731
    return dict(
        t88=dict(path=str(p['hs_field']), file_sha256=hs['field']['sha256'], u_sha256=hs['field']['u_sha256'], status=hs.get('status'), env=hs.get('env'), workers=hs.get('workers'),
                 repair_source_field_sha256=hs_r['source']['field_sha256'], repair_source_tapered_u_sha256=hs_r['source'].get('tapered_u_sha256'), seconds_total=hs.get('seconds_total')),
        hs=dict(repair=str(p['hs_repair']), metrics=str(p['hs_metrics']), repaired_u_sha256=hs_m.get('repaired_u_sha256'), passed=hs_r.get('passed'), rounds=len(hs_r.get('rounds') or []),
                seconds_total=hs_r.get('seconds_total'), values=metrics(hs_m)),
        p3=dict(path=str(p['p3_field']), file_sha256=(p3.get('field') or {}).get('sha256'), u_sha256=p3_check.get('u_sha256'), status=p3.get('status'), segmenter_id=p3.get('segmenter_id'), env=p3.get('env'),
                seconds_total=p3.get('seconds_total'), host_peak_rss_gib=p3.get('host_peak_rss_gib'), device_peak_reserved_gib=p3.get('device_peak_reserved_gib'), pass_u_sha256=ratios['u_sha256'],
                ratios_used=ratios['ratios_used'], last_estimate=ratios['last_estimate']),
        p3_repair=dict(repair=str(p['p3_repair']), metrics=str(p['p3_metrics']), passed=p3_r.get('passed'), rounds=len(p3_r.get('rounds') or []), seconds_total=p3_r.get('seconds_total'),
                       initial_certificate={k: (p3_r.get('initial_certificate') or {}).get(k) for k in ('status', 'n_infeasible', 'state_hash')}, repaired_u_sha256=p3_m.get('repaired_u_sha256'), values=metrics(p3_m)),
        d0=dict(final_native=str(final), file_sha256=final_json.get('sha256'), state_hash=(final_json.get('certificate') or {}).get('state_hash'), role=final_json.get('role'),
                classification=classification.get('class'), whole_wall_seconds=(classification.get('evidence') or {}).get('whole_wall_seconds')),
        manifest=original_lam(p['manifest']))


def admission_problems(stage=None, allowlist=None):
    """The pair's admission without opening pair.npz: the sealed lists on the names and the manifest's subjects (run_hx.sealed_admission), then the allowlist's guard on them (load_pair's first step)."""
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    from bayesreg.io.pair import _read_manifest
    stage = Path(STAGE if stage is None else stage)
    ctx = dict(stage_dir=str(stage), pair=PAIR)
    try:
        R.sealed_admission(ctx)
        _, _, selected = _read_manifest(stage/PAIR/'manifest.json')
        check_pair_subjects(selected['fixed_subject'], selected['moving_subject'], load_dev_allowlist(core.ALLOWLIST if allowlist is None else allowlist))
    except R.admission_refusals() as exc:
        return [f'admission: {type(exc).__name__}: {exc}'[:1000]]
    return []


def reference_problems(refs, *, anchor=True):
    """The checks beyond reference_records' own, asked by the queue's input check and the dry run alike (HE2 review minor 10): the frozen settings' segmenter anchor (anchor=False: the caller asks it),
    hs_v1's field done and the one its repair read, the P3 reference a done field of the anchor's segmenter, P3's field not passing its certificate before the repair (item 13's premise)."""
    problems = [f'anchor: {x}' for x in he.segmenter_start_problems(he.frozen_settings())[0]] if anchor else []
    t, c, w = refs['t88'], refs['p3'], refs['p3_repair']
    if t['repair_source_field_sha256'] != t['file_sha256'] or t['status'] != 'done':
        problems.append('item 3: the reference field records disagree')
    if c['status'] != 'done' or c['segmenter_id'] != he.CONTRACT_SEGMENTER['segmenter_id']:
        problems.append("item 10: the P3 reference is not a done field of the anchor's segmenter")
    if w['initial_certificate'].get('status') in (None, he.CERT_PASS):
        problems.append('item 13: the P3 reference field passed its certificate before the repair: the forcing would not force')
    return problems


def input_problems(results=None, stage=None, allowlist=None):
    problems = []
    try:
        problems += reference_problems(reference_records(results, stage))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        problems.append(f'references: {type(exc).__name__}: {exc}'[:1000])
    return problems+admission_problems(stage, allowlist)


# ---- the queue ------------------------------------------------------------------------------------------------------------------------------------------
def keep_old(root, names=('queue_stop',)):
    """Evidence kept under a new name before a start can write its own (run_mc0.queue writes queue_stop.json on a STOP): <name>_<k>.json, k the first free number."""
    for name in names:
        if (root/f'{name}.json').exists():
            k = 0
            while (root/f'{name}_{k}.json').exists():
                k += 1
            (root/f'{name}.json').rename(root/f'{name}_{k}.json')


def plan_names():
    return [s.unit for s in STEPS]


def queue_body(o, system, root, log, *, results=None, stage=None):
    root = Path(root)
    keep_old(root)                                      # first: a refusal below must not overwrite an earlier stop
    launched = launched_steps(root)
    if launched:
        raise run.Stop(f'units were launched in this root before ({launched}): a smoke root runs once and nothing is rerun; start a new root')
    gate = replace(GATE, host_floor_gib=o.min_host_gib)
    head = git_commit('HEAD', cwd=core.ROOT)            # SM3 revision 3: the HEAD of the tree this queue runs from (P0's detached checkout for he_smoke_v3), never the main checkout's
    sha = git_commit(o.commit, cwd=core.ROOT) if o.commit else head
    if sha != head:
        raise run.Stop(f'--commit {sha} is not HEAD {head}: the queue runs the code it checks out; start it from a checkout at that commit')
    problems = input_problems(results, stage)
    if problems:
        raise run.Stop(f'the inputs, before any unit: {problems}; nothing checked out')
    code, sha = system.code_checkout(root, sha)
    _, problems = system.code_identity(code)
    if problems:
        raise run.Stop(f'the identity of {code}, before any unit: {problems}')
    batch = read_json(root/'batch.json')
    value = dict(schema=SCHEMA.format('batch'), code_sha=sha, code=str(code), pair=PAIR, stage=str(STAGE), plan=plan_names(), frozen_settings_sha256=he.settings_sha256(he.frozen_settings()),
                 forced_settings_sha256=he.settings_sha256(forced_settings()), forced_repair=FORCED_REPAIR, gate=dict(own_units=list(gate.own_units), load_limit=gate.load_limit, gpu_free_mib=gate.gpu_free_mib,
                 mem_available_gib=gate.mem_available_gib, mem_available_gib_frozen_run=D_MEM_AVAILABLE_GIB, frozen_run_kinds=list(FROZEN_RUN_KINDS), home_floor_gb=run.HOME_FLOOR_GB, host_floor_gib=gate.host_floor_gib),
                 units=dict(supervisor=SUPERVISOR, evaluation=EVALUATION, frozen_c=dict(resources=list(fc.UNIT_RESOURCES), oom_policy=fc.UNIT_OOM_POLICY)),
                 references=dict(pass0_u_sha256=REF_PASS0_U_SHA256, pass2_u_sha256=REF_PASS2_U_SHA256),
                 gap_seconds=o.gap_seconds, wait_minutes=o.wait_minutes, frozen_commit=core.FROZEN_COMMIT, started_utc=run.utc())
    if batch is None:
        write_once(root/'batch.json', value)
    elif batch.get('code_sha') != sha or batch.get('plan') != value['plan']:
        raise run.Stop(f"batch.json is at {batch.get('code_sha')} with the plan {batch.get('plan')}; this start is at {sha}")
    log(f'start: {len(STEPS)} units {plan_names()} at {sha}; the pair {SHORT}; one at a time; a failure stops the queue')
    rcs = {}
    for step in STEPS:
        this_gate = step_gate(step, gate)
        run.wait_for_gate(system, root, step.gpu, o.wait_minutes, log, this_gate)
        (root/step.arm).mkdir(parents=True, exist_ok=True)
        argv = step_argv(step, code, root)
        write_once(root/'steps'/f'{step.name}.launched.json', dict(step=step.name, unit=step.unit, kind=step.kind, gpu=step.gpu, argv=argv, gate_mem_available_gib=this_gate.mem_available_gib, utc=run.utc(),
                                                                   mem_available_gib_at_launch=system.mem_available_gib(), uptime_before=system.uptime()))     # HE2 clean: the reading at launch, not only the threshold
        log(f'launch {step.unit}')
        tick = system.now()
        rc = system.run(argv, root/step.arm/f'{step.unit}.systemd_run.log')
        missing = [str(p) for p in expected_files(step, root) if not p.is_file()]
        rcs[step.unit] = rc
        write_once(root/'steps'/f'{step.name}.done.json', dict(step=step.name, unit=step.unit, rc=rc, missing=missing, seconds=system.now()-tick, uptime_after=system.uptime(), utc=run.utc()))
        log(f'{step.unit}: exit {rc}{f", missing {missing}" if missing else ""}')
        if rc != 0 or missing:
            raise run.Stop(f'{step.unit} exited {rc}{f" and left no {missing}" if missing else ""}: the queue stops here (nothing is rerun)')
        system.sleep(o.gap_seconds)
    report = read_json(root/f'{REPORT}.json') or {}
    summary = report.get('summary')
    log(f'check: {summary}.  Handed to the lead: no next step starts here')
    write_once(root/'queue_end.json', dict(rc=rcs, summary=summary, utc=run.utc()))
    return 0


def queue(o, system=None, *, results=None, stage=None):
    problem = root_problem(o.root)
    if problem:                                         # before the lock: nothing is created
        print(problem, file=sys.stderr)
        return 2
    return run.queue(o, System() if system is None else system, body=lambda o, s, root, log: queue_body(o, s, root, log, results=results, stage=stage), prefix=ROOT_PREFIX)


# ---- the check: items 1-9 ------------------------------------------------------------------------------------------------------------------------------
PASS, FAIL, OPEN, DESCRIBE = '过', '不过', '不过（待查）', '只描述'
T88_TIMEOUT, REPAIR_TIMEOUT = 1200, 1800        # he_core frozen children's timeouts (a test compares)
FULL_CERTIFICATES, CERTIFICATE_SECONDS = 3, 120.         # 〔实现员定〕item 7: the candidate's, the identity's and one spare, each <= 120 s with the JIT load (rw_v1: about 1 s once compiled)
FALLBACK_SETUP_SECONDS = 300.                   # 〔实现员定〕item 7: the derived pair, the private copy, prepare, check_prepared, classify, dmesg
OOM_POLICY = 'continue'                         # revision 1 (review important 1): item 7 asks it of the four supervisor units
DEVICE_LIMIT_GIB = 9.
GPU_OOM_STATUSES = ('cuda_oom', 'vram_stop')    # revision 1 (review minor 6): run_hx.t88_main's statuses of a device that ran out (the 9 GiB cap of limit_device_memory or vram_guard)
EXPECTED_LOADS = {he.HX: 3, he.FALLBACK_C: 5}   # HE note section 6 item 2: the supervisor, t88, repair one load_pair each; fallback one also derive_pair and firewall_check


def item(n, name, criterion, verdict, values, note=None):
    return dict(item=n, name=name, criterion=criterion, verdict=verdict, values=values, note=note)


def missing_item(n, name, criterion, what):
    return item(n, name, criterion, FAIL, dict(missing=what), note='a record this item reads is missing')


def _gib(text):
    m = re.fullmatch(r'([0-9.]+)([KMGT])?', str(text))
    if not m:
        return None
    return float(m.group(1))*{None: 2**-30, 'K': 2**-20, 'M': 2**-10, 'G': 1., 'T': 2**10}[m.group(2)]


def unit_property(argv, name):
    prefix = f'--property={name}='
    return next((a[len(prefix):] for a in argv or [] if a.startswith(prefix)), None)


def line_of(d):
    return read_json(Path(d)/'output/line.json')


def item1_identity(root, batch):
    criterion = ('every identity*.json of a, b, c, d: same true, problems [], repo.head and pinned_head the commit batch.json recorded when the queue started, tracked_clean and code_paths_clean true; '
                 'identity.json in each; context.json code_head that commit; a and b with the frozen values, c and d with exactly forced_settings()')
    if batch is None or not batch.get('code_sha'):
        return missing_item(1, 'identity', criterion, ['batch.json'])
    sha, ok, rows = batch['code_sha'], True, {}
    forced_sha = he.settings_sha256(forced_settings())
    for arm in ARMS:
        d = run_dir(root, arm)
        ctx = read_json(d/'context.json') or {}
        files = sorted(d.glob('identity*.json'))
        arm_rows = {}
        for f in files:
            r = read_json(f) or {}
            repo = r.get('repo') or {}
            row = dict(same=r.get('same'), problems=r.get('problems'), head=repo.get('head'), pinned_head=r.get('pinned_head'), tracked_clean=repo.get('tracked_clean'), code_paths_clean=repo.get('code_paths_clean'),
                       checked_utc=r.get('checked_utc'))
            row['ok'] = (row['same'] is True and row['problems'] == [] and row['head'] == sha and row['pinned_head'] == sha and row['tracked_clean'] is True and row['code_paths_clean'] is True)
            ok &= row['ok']
            arm_rows[f.name] = row
        settings_ok = (ctx.get('settings_frozen') is True) if arm in ('a', 'b') else (ctx.get('settings_frozen') is False and ctx.get('settings_sha256') == forced_sha)
        present = (d/'identity.json').is_file()
        ok &= present and ctx.get('code_head') == sha and settings_ok and ctx.get('check_identity') is True
        rows[arm] = dict(records=arm_rows, identity_json=present, code_head=ctx.get('code_head'), settings_frozen=ctx.get('settings_frozen'), settings_sha256=ctx.get('settings_sha256'), settings_ok=settings_ok)
    return item(1, 'identity', criterion, PASS if ok else FAIL, dict(code_sha=sha, forced_settings_sha256=forced_sha, arms=rows))


def item2_spy(root):
    criterion = ('b/spy.json: passed and positive_condition true, run_rc 0, no breach; len(loader_calls) = reads_seen.fixed = reads_seen.moving = reads_at_the_class.fixed = .moving; '
                 'forbidden_access and evaluation_records_opened empty; the line status of a (production, subprocess children) = that of b; the seconds only described')
    spy = read_json(run_dir(root, 'b')/'spy.json')
    la, lb = line_of(run_dir(root, 'a')), line_of(run_dir(root, 'b'))
    if spy is None or la is None or lb is None:
        return missing_item(2, 'spy', criterion, [n for n, x in (('b/spy.json', spy), ('a/output/line.json', la), ('b/output/line.json', lb)) if x is None])
    loads = len(spy.get('loader_calls') or [])
    seen, at = spy.get('reads_seen') or {}, spy.get('reads_at_the_class') or {}
    counts = loads == seen.get('fixed') == seen.get('moving') == at.get('fixed') == at.get('moving')
    ok = (spy.get('passed') is True and spy.get('positive_condition') is True and spy.get('run_rc') == 0 and spy.get('breach') is None and counts and spy.get('forbidden_access') == []
          and spy.get('evaluation_records_opened') == [] and la.get('line_status') == lb.get('line_status'))
    expected = EXPECTED_LOADS.get(lb.get('line_status'))
    values = dict(passed=spy.get('passed'), positive_condition=spy.get('positive_condition'), run_rc=spy.get('run_rc'), breach=spy.get('breach'), loader_calls=spy.get('loader_calls'), reads_seen=seen,
                  reads_at_the_class=at, forbidden_access=spy.get('forbidden_access'), evaluation_records_opened=spy.get('evaluation_records_opened'), line_status_a=la.get('line_status'),
                  line_status_b=lb.get('line_status'), seconds_total_a=la.get('seconds_total'), seconds_total_b=lb.get('seconds_total'),
                  described=dict(expected_loads_for_the_line_status=expected, loads_as_expected=None if expected is None else loads == expected))
    return item(2, 'spy', criterion, PASS if ok else FAIL, values)


def npz_u(path):
    with np.load(path, allow_pickle=False) as z:
        return np.array(z['u'])


def unit_values(argv):
    """The unit's parameters a T88 may see: NUMBA_NUM_THREADS, CPUQuota, MemoryMax (None where the string has none)."""
    setenv = '--setenv=NUMBA_NUM_THREADS='
    return dict(NUMBA_NUM_THREADS=next((a[len(setenv):] for a in argv or [] if a.startswith(setenv)), None), CPUQuota=unit_property(argv, 'CPUQuota'), MemoryMax=unit_property(argv, 'MemoryMax'))


def reference_unit(ref_dir):
    """fd2_v1's own start record of its field unit (cmd.txt, the systemd-run string): the unit values, or None with a note."""
    cmd = Path(ref_dir)/'cmd.txt'
    if not cmd.is_file():
        return dict(values=None, note=f'{cmd} is not there')
    return dict(values=unit_values(shlex.split(cmd.read_text())), source=str(cmd))


ENV_KEYS = ('torch', 'cuda', 'device', 'cudnn', 'deterministic_algorithms', 'cublas_workspace_config')


def env_of(rec):
    return {k: (rec.get('env') or {}).get(k) for k in ENV_KEYS}


def pass_u(record, k):
    """passes[k].u_sha256 of a T88 run.json, or None."""
    passes = (record or {}).get('passes') or []
    return passes[k].get('u_sha256') if len(passes) > k else None


def item3_pass0(root, results=None):
    """HE2: C22.B4-6 (a) in the place of v1's item 3 (the T88 field against fd2_v1's): with C on, the field the method keeps is pass 2's (item 10); pass 0 is the frozen T88 (C off), whose array is not kept
    (C22.B1-9), so its u_sha256 (sha256 of the C-order bytes, the same convention as the reference's) is compared."""
    criterion = ("C22.B4-6 (a): a's T88 pass 0 (a = 1, the frozen T88: C off) against hs_v1's T88 field of the pair (v4, C off): the u_sha256 that a's t88/run.json records for passes[0] equal to the sha256 of the "
                 "C-order bytes of hs_v1's stored array: 过; different: 不过（待查）, both hashes, the workers, the device, the torch version and both units' NUMBA_NUM_THREADS, CPUQuota, MemoryMax recorded, no "
                 f"reason guessed; the reference array not the one hs_v1's run.json names (field.u_sha256) or not the contract's {REF_PASS0_U_SHA256[:8]}…: 不过; a's record without pass 0: 不过.  "
                 "b, c, d's pass 0 only described")
    p = reference_paths(results)
    ours_dir = run_dir(root, 'a')/'t88'
    ours_rec, ref_rec = read_json(ours_dir/'run.json'), read_json(p['hs_run'])
    ref_file = p['hs_field']
    if ours_rec is None or ref_rec is None or not ref_file.is_file():
        return missing_item(3, 't88_pass0', criterion, [str(x) for x in (ours_dir/'run.json', p['hs_run'], ref_file) if not Path(x).is_file()])
    ours_u = pass_u(ours_rec, 0)
    if ours_u is None:
        return missing_item(3, 't88_pass0', criterion, ['a/t88/run.json passes[0].u_sha256'])
    b = npz_u(ref_file)
    ref_u = he.array_sha256(b)
    reference_is_recorded = ref_u == (ref_rec.get('field') or {}).get('u_sha256') == REF_PASS0_U_SHA256        # revision 1, review minor 4; HE2: and the contract's
    same = ours_u == ref_u
    launched, _ = step_records(root, 'a')
    ctx = read_json(run_dir(root, 'a')/'context.json') or {}
    others = {arm: dict(u_sha256=pass_u(read_json(run_dir(root, arm)/'t88/run.json'), 0), same_as_a=None) for arm in ('b', 'c', 'd')}
    for row in others.values():
        row['same_as_a'] = None if row['u_sha256'] is None else row['u_sha256'] == ours_u
    values = dict(ours=dict(record=str(ours_dir/'run.json'), u_sha256=ours_u, n_passes=len(ours_rec.get('passes') or []), tissue_prior=ours_rec.get('tissue_prior'), workers=ctx.get('workers'), env=env_of(ours_rec),
                            unit=dict(values=unit_values((launched or {}).get('argv')), source='steps/a.launched.json')),
                  reference=dict(path=str(ref_file), file_sha256=core.sha_file(ref_file), u_sha256=ref_u, recorded_u_sha256=(ref_rec.get('field') or {}).get('u_sha256'), contract_u_sha256=REF_PASS0_U_SHA256,
                                 dtype=str(b.dtype), shape=list(b.shape), workers=ref_rec.get('workers'), workers_note=None if 'workers' in ref_rec else "hs_v1's run.json records no workers", env=env_of(ref_rec),
                                 unit=reference_unit(p['hs_dir'])),
                  hashes_equal=same, reference_is_the_recorded_field=reference_is_recorded, contract='C22.B4-6 (a)', described=dict(b_c_d_against_a=others))
    if not reference_is_recorded:
        return item(3, 't88_pass0', criterion, FAIL, values, note='参照不是记录里的那个场（hs_v1 run.json 的 u_sha256，合同的 b6f061da…）')
    return item(3, 't88_pass0', criterion, PASS if same else OPEN, values, note=None if same else '不同：只记录，不解释原因')


def item3_field_off(root, results=None):
    """HB (C22 卷首修订 8, D85: the new HE smoke of the method with C off; runs/20261109-hb/smoke_plan.md): item 3 when a's settings have tissue_prior off.  There is no pass 0: the T88 child runs
    mc0_core.optimize_k once (HS's path), so a's own field (t88/field_0600.npz) is compared with hs_v1's T88 field of the pair bit for bit, and its record names the same u_sha256."""
    criterion = ("C22 卷首修订 8 / D85 (C off): a's t88/run.json says tissue_prior off, no passes, and its field (t88/field_0600.npz: dtype, shape, C-order bytes) is hs_v1's T88 field of the pair "
                 f"(SHx24_first5/0438-0439/field_0600.npz, u_sha256 {REF_PASS0_U_SHA256[:8]}…): 过; different: 不过（待查）, both hashes, the workers, the device, the torch version and both units' "
                 "NUMBA_NUM_THREADS, CPUQuota, MemoryMax recorded, no reason guessed; the reference not the one hs_v1's run.json names or not the contract's: 不过; a's own record not consistent (field.u_sha256 "
                 "and the file's array not one hash, or tissue_prior not off, or passes present): 不过.  b, c, d's fields described")
    p = reference_paths(results)
    ours_dir = run_dir(root, 'a')/'t88'
    ours_rec, ref_rec = read_json(ours_dir/'run.json'), read_json(p['hs_run'])
    ours_file, ref_file = ours_dir/he.FIELD_FILE, p['hs_field']
    if ours_rec is None or ref_rec is None or not ours_file.is_file() or not ref_file.is_file():
        return missing_item(3, 't88_field_c_off', criterion, [str(x) for x in (ours_dir/'run.json', p['hs_run'], ours_file, ref_file) if not Path(x).is_file()])
    a, b = npz_u(ours_file), npz_u(ref_file)
    same = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    ours_u, ref_u = he.array_sha256(a), he.array_sha256(b)
    reference_is_recorded = ref_u == (ref_rec.get('field') or {}).get('u_sha256') == REF_PASS0_U_SHA256
    consistent = ours_rec.get('tissue_prior') == 'off' and ours_rec.get('passes') is None and ours_u == (ours_rec.get('field') or {}).get('u_sha256')
    launched, _ = step_records(root, 'a')
    ctx = read_json(run_dir(root, 'a')/'context.json') or {}
    others = {}
    for arm in ('b', 'c', 'd'):
        u = ((read_json(run_dir(root, arm)/'t88/run.json') or {}).get('field') or {}).get('u_sha256')
        others[arm] = dict(u_sha256=u, same_as_a=None if u is None else u == ours_u)
    values = dict(ours=dict(path=str(ours_file), file_sha256=core.sha_file(ours_file), u_sha256=ours_u, recorded_u_sha256=(ours_rec.get('field') or {}).get('u_sha256'), tissue_prior=ours_rec.get('tissue_prior'),
                            dtype=str(a.dtype), shape=list(a.shape), workers=ctx.get('workers'), env=env_of(ours_rec), unit=dict(values=unit_values((launched or {}).get('argv')), source='steps/a.launched.json')),
                  reference=dict(path=str(ref_file), file_sha256=core.sha_file(ref_file), u_sha256=ref_u, recorded_u_sha256=(ref_rec.get('field') or {}).get('u_sha256'), contract_u_sha256=REF_PASS0_U_SHA256,
                                 dtype=str(b.dtype), shape=list(b.shape), env=env_of(ref_rec), unit=reference_unit(p['hs_dir'])),
                  arrays_equal=same, reference_is_the_recorded_field=reference_is_recorded, ours_consistent=consistent, contract='C22 卷首修订 8, D85', described=dict(b_c_d_against_a=others))
    if not reference_is_recorded:
        return item(3, 't88_field_c_off', criterion, FAIL, values, note='参照不是记录里的那个场（hs_v1 run.json 的 u_sha256，合同的 b6f061da…）')
    if not consistent:
        return item(3, 't88_field_c_off', criterion, FAIL, values, note='a 自己的记录对不上（tissue_prior、passes、field.u_sha256 与文件里的数组）')
    return item(3, 't88_field_c_off', criterion, PASS if same else OPEN, values, note=None if same else '不同：只记录，不解释原因')


def item13_forcing_off(root, results=None):
    """HB: item 13 with C off: the forcing (max_rounds 1, max_inner 0) needs HS's T88 field not to pass its certificate before the repair (hs_v1 RSHx24_c600's initial certificate)."""
    criterion = ("C off (C22 卷首修订 8): the forcing (max_rounds 1, max_inner 0) needs the T88 field not to pass its certificate before the repair: hs_v1/runs/RSHx24_c600/0438-0439/repair.json "
                 "initial_certificate not certified_feasible; c and d's own repair.json initial_certificate not a passing one (C07-3) and their reason round_cap.  Their initial state hash against HS's: "
                 "described")
    ref_path = reference_paths(results)['hs_repair']
    hs = read_json(ref_path)
    if hs is None:
        return missing_item(13, 'forcing_precondition_c_off', criterion, [str(ref_path)])
    ref = hs.get('initial_certificate') or {}
    ok = ref.get('status') not in (None, he.CERT_PASS)
    rows = {}
    for arm in ('c', 'd'):
        repair = read_json(run_dir(root, arm)/'repair/repair.json')
        if repair is None:
            return missing_item(13, 'forcing_precondition_c_off', criterion, [f'{arm}/repair/repair.json'])
        initial = repair.get('initial_certificate') or {}
        rows[arm] = dict(status=initial.get('status'), n_infeasible=initial.get('n_infeasible'), state_hash=initial.get('state_hash'), reason=repair.get('reason'),
                         same_state_as_hs=None if initial.get('state_hash') is None else initial.get('state_hash') == ref.get('state_hash'))
        ok &= bool(initial) and not he.passes(initial) and repair.get('reason') == 'round_cap'
    return item(13, 'forcing_precondition_c_off', criterion, PASS if ok else FAIL, dict(reference={k: ref.get(k) for k in ('status', 'n_infeasible', 'state_hash')}, arms=rows, contract='C22 卷首修订 8'))


def not_applicable(n, name, why):
    return item(n, name, why, DESCRIBE, dict(applies=False))


def c_off(root):
    """Whether the smoke's production arm ran with tissue_prior off: what its T88 record says it ran (the context's settings when there is no record)."""
    mode = (read_json(run_dir(root, 'a')/'t88/run.json') or {}).get('tissue_prior')
    if mode is None:
        mode = ((read_json(run_dir(root, 'a')/'context.json') or {}).get('settings') or {}).get('tissue_prior')
    return mode == 'off'


def item10_pass2(root, results=None):
    criterion = ("C22.B4-6 (b): a/t88/field_0600.npz (pass 2's field, the one kept) against p3_v1/runs/C_first5/0438-0439/field_0600.npz: the stored array u (dtype, shape, C-order bytes) the same: 过; "
                 "different: 不过（待查）, both hashes recorded, no reason guessed; the reference array not the one p3_v1's run.json names (checkpoints.600.u_sha256 and field.u_sha256) or not the contract's "
                 f"{REF_PASS2_U_SHA256[:8]}…: 不过; a's own record not consistent (field.u_sha256, passes[2].u_sha256 and the file's array not one hash, or not three passes): 不过.  b, c, d's kept fields described")
    p = reference_paths(results)
    ours_dir = run_dir(root, 'a')/'t88'
    ours_rec, ref_rec = read_json(ours_dir/'run.json'), read_json(p['p3_run'])
    ours_file, ref_file = ours_dir/he.FIELD_FILE, p['p3_field']
    if ours_rec is None or ref_rec is None or not ours_file.is_file() or not ref_file.is_file():
        return missing_item(10, 't88_pass2', criterion, [str(x) for x in (ours_dir/'run.json', p['p3_run'], ours_file, ref_file) if not Path(x).is_file()])
    a, b = npz_u(ours_file), npz_u(ref_file)
    same = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    ref_u = he.array_sha256(b)
    recorded = ((ref_rec.get('checkpoints') or {}).get(str(NATIVE)) or {}).get('u_sha256')
    reference_is_recorded = ref_u == recorded == (ref_rec.get('field') or {}).get('u_sha256') == REF_PASS2_U_SHA256
    ours_u = he.array_sha256(a)
    consistent = len(ours_rec.get('passes') or []) == 3 and ours_u == (ours_rec.get('field') or {}).get('u_sha256') == pass_u(ours_rec, 2)
    others = {}
    for arm in ('b', 'c', 'd'):
        u = ((read_json(run_dir(root, arm)/'t88/run.json') or {}).get('field') or {}).get('u_sha256')
        others[arm] = dict(u_sha256=u, same_as_a=None if u is None else u == ours_u)
    values = dict(ours=dict(path=str(ours_file), file_sha256=core.sha_file(ours_file), u_sha256=ours_u, recorded_u_sha256=(ours_rec.get('field') or {}).get('u_sha256'), pass2_u_sha256=pass_u(ours_rec, 2),
                            dtype=str(a.dtype), shape=list(a.shape), env=env_of(ours_rec)),
                  reference=dict(path=str(ref_file), file_sha256=core.sha_file(ref_file), u_sha256=ref_u, recorded_u_sha256=recorded, contract_u_sha256=REF_PASS2_U_SHA256, dtype=str(b.dtype), shape=list(b.shape),
                                 env=dict((ref_rec.get('env') or {})), unit=reference_unit(p['p3_dir'])),
                  arrays_equal=same, reference_is_the_recorded_field=reference_is_recorded, ours_consistent=consistent, contract='C22.B4-6 (b)', described=dict(b_c_d_against_a=others))
    if not reference_is_recorded:
        return item(10, 't88_pass2', criterion, FAIL, values, note='参照不是记录里的那个场（p3_v1 run.json 的 u_sha256，合同的 edf3b039…）')
    if not consistent:
        return item(10, 't88_pass2', criterion, FAIL, values, note='a 自己的记录对不上（三遍、field.u_sha256、passes[2].u_sha256、文件里的数组）')
    return item(10, 't88_pass2', criterion, PASS if same else OPEN, values, note=None if same else '不同：只记录，不解释原因')


def item11_ratios(root, results=None):
    criterion = ("C22.B4-6 (c): the ratios a used in pass 1 and pass 2 (t88/run.json passes[1], passes[2] tissue.ratios_used) equal p3_v1's of the pair, number for number (JSON): 过; different: 不过（待查）; a "
                 "record without them: 不过.  The estimate on pass 2's field, pass 1's u_sha256 against p3's, b, c, d's ratios: described")
    p = reference_paths(results)
    ours, ref = read_json(run_dir(root, 'a')/'t88/run.json'), read_json(p['p3_run'])
    if ours is None or ref is None:
        return missing_item(11, 'c_ratios', criterion, [str(x) for x in (run_dir(root, 'a')/'t88/run.json', p['p3_run']) if not Path(x).is_file()])
    mine, theirs = p3_ratios(ours), p3_ratios(ref)
    if len(mine['ratios_used']) != 2 or None in mine['ratios_used']:
        return missing_item(11, 'c_ratios', criterion, ['a/t88/run.json passes[1..2].tissue.ratios_used'])
    same = canonical_json(mine['ratios_used']) == canonical_json(theirs['ratios_used'])
    others = {arm: p3_ratios(read_json(run_dir(root, arm)/'t88/run.json') or {})['ratios_used'] for arm in ('b', 'c', 'd')}
    values = dict(ours=mine, reference=theirs, ratios_equal=same, contract='C22.B4-6 (c)',
                  described=dict(pass1_same_as_p3=None if len(mine['u_sha256']) < 2 or len(theirs['u_sha256']) < 2 else mine['u_sha256'][1] == theirs['u_sha256'][1],
                                 last_estimate_same_as_p3=canonical_json(mine['last_estimate']) == canonical_json(theirs['last_estimate']),
                                 b_c_d_same_as_a={arm: canonical_json(r) == canonical_json(mine['ratios_used']) for arm, r in others.items()}))
    return item(11, 'c_ratios', criterion, PASS if same else OPEN, values, note=None if same else '不同：只记录，不解释原因')


ORDER_RULE = 'the review certificate of the output before any label was read'


def recompute_metrics(ctx, out_dir, *, stage=None):
    """Item 4: the metrics once more, by bayesreg.eval.metrics from disp.npz, the truth through bayesreg.eval.truth (sealed lists, allowlist) as the evaluation unit reads it.  The check's own read; only of the
    smoke's pair of the smoke's stage (revision 1, review minor 3: not whatever a context.json names), ValueError before anything is opened otherwise."""
    from bayesreg.eval import fields, metrics
    from bayesreg.eval import truth as bench_truth
    stage_dir = Path(ctx['stage_dir'])
    if ctx.get('pair') != PAIR or stage_dir != Path(STAGE if stage is None else stage):
        raise ValueError(f"the recomputation reads the truth of {PAIR} of {STAGE if stage is None else stage} only, not {ctx.get('pair')} of {stage_dir}")
    sealed = bench_truth.load_sealed_lists()
    pair_dir, grid, _, guard = bench_truth.admit_pair(stage_dir.parent, stage_dir.name, ctx['pair'], ctx['paths']['allowlist'], sealed)
    d, disp_record = fields.read_disp(Path(out_dir)/'output/disp.npz', grid.shape)
    scheme = bench_truth.scheme_for(guard, pair_dir)
    labels, _ = bench_truth.load_truth(pair_dir, scheme, grid, guard)
    scalars, per_label, official = metrics.evaluate_field(d, labels, scheme, tuple(grid.spacing))
    return dict(metrics=scalars, per_label=per_label, official_row=official, d_sha256=disp_record['d_sha256'])


def canonical_json(value):
    return json.dumps(json.loads(json.dumps(value, default=he._default)), sort_keys=True)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def item4_repair_and_evaluation(root, results=None, recompute=None, stage=None):
    criterion = ('a: line_status hx_certified; repair.json passed true with its rounds; output/cert.json passes (C07-3) and is of final.npz; evaluation.json review consistent and the review certificate the same '
                 'as cert.json field by field (he_core.same_certificate); the order: evaluation.json order is the code\'s rule, evaluation_started.json (the marker before the truth) names the review\'s state '
                 'hash, and identity_eval_<k>.json <= the marker <= evaluation.json in time; seg35 and seg4 both reported; the metrics from disp.npz once more (bayesreg.eval.metrics, the check\'s own read of '
                 'the truth) equal to evaluation.json.  Only described: P3\'s repair of the same pair (v4, C on: the same configuration; p3_v1 RC_c600) and HS\'s (v4, C off; hs_v1 RSHx24_c600); repairs are not '
                 'reproducible bit for bit')
    d = run_dir(root, 'a')
    line, repair, ev, ctx = line_of(d), read_json(d/'repair/repair.json'), read_json(d/'evaluation.json'), read_json(d/'context.json')
    if None in (line, repair, ev, ctx) or not (d/'output/final.npz').is_file():
        return missing_item(4, 'repair_and_evaluation', criterion, [n for n, x in (('output/line.json', line), ('repair/repair.json', repair), ('evaluation.json', ev), ('context.json', ctx)) if x is None])
    checks, values = {}, {}
    checks['line_status'] = line.get('line_status') == he.HX
    rounds = repair.get('rounds')
    checks['repair_passed'] = repair.get('passed') is True and isinstance(rounds, list)
    try:
        u, _, cert = he.read_output(d/'output')
        checks['certificate'] = he.passes(cert) and cert.get('state_hash') == he.array_sha256(u)
    except (OSError, ValueError, KeyError) as exc:
        cert, checks['certificate'] = None, False
        values['output_error'] = f'{type(exc).__name__}: {exc}'
    rc = ev.get('review_certificate') or {}
    same, diffs = he.same_certificate(rc.get('review') or {}, cert or {})
    checks['review_consistent'] = rc.get('consistent') is True and same
    marker = read_json(d/R.TRUTH_MARKER)
    starts = sorted(d.glob('identity_eval_[0-9]*.json'), key=lambda p: p.stat().st_mtime_ns)
    times = [starts[-1].stat().st_mtime_ns if starts else None, (d/R.TRUTH_MARKER).stat().st_mtime_ns if marker is not None else None, (d/'evaluation.json').stat().st_mtime_ns]
    checks['order'] = (str(ev.get('order', '')).startswith(ORDER_RULE) and marker is not None and marker.get('state_hash') == (rc.get('review') or {}).get('state_hash') and None not in times
                       and times[0] <= times[1] <= times[2])
    m = ev.get('metrics') or {}
    checks['seg35_and_seg4'] = finite(m.get('seg35')) and finite(m.get('seg4'))
    try:
        again = recompute(ctx, d) if recompute is not None else recompute_metrics(ctx, d, stage=stage)
        checks['recomputed_equal'] = all(canonical_json(again[k]) == canonical_json(ev.get(k)) for k in ('metrics', 'per_label', 'official_row'))
        values['recomputed'] = dict(seg35=again['metrics'].get('seg35'), seg4=again['metrics'].get('seg4'), d_sha256=again.get('d_sha256'),
                                    differing=[k for k in ('metrics', 'per_label', 'official_row') if canonical_json(again[k]) != canonical_json(ev.get(k))])
    except Exception as exc:                            # noqa: BLE001 - no recomputation is a failed item, with its reason
        checks['recomputed_equal'] = False
        values['recompute_error'] = f'{type(exc).__name__}: {exc}'[:1000]
    paths = reference_paths(results)
    described = dict(output_state_hash=(cert or {}).get('state_hash'), note='开发集单次；这一对的 Dice 是 P3 这一对的重复，不算选择读数；seg35 与 seg4 都报')
    for name, key in (('p3', 'p3_metrics'), ('hs', 'hs_metrics')):
        ref = read_json(paths[key]) or {}
        after = (ref.get('after') or {}).get('metrics') or {}
        described.update({f'{name}_seg35': after.get('seg35'), f'{name}_seg4': after.get('seg4'), f'{name}_sdlogj_official': after.get('sdlogj_official'),
                          f'seg35_minus_{name}': None if not (finite(m.get('seg35')) and finite(after.get('seg35'))) else m['seg35']-after['seg35'],
                          f'seg4_minus_{name}': None if not (finite(m.get('seg4')) and finite(after.get('seg4'))) else m['seg4']-after['seg4'],
                          f'{name}_repaired_u_sha256': ref.get('repaired_u_sha256'), f'same_field_as_{name}': None if cert is None else cert.get('state_hash') == ref.get('repaired_u_sha256')})
    values.update(checks=checks, line_status=line.get('line_status'), n_rounds=len(rounds) if isinstance(rounds, list) else None, repair_reason=repair.get('reason'), review_differences=diffs,
                  seg35=m.get('seg35'), seg4=m.get('seg4'), sdlogj_official=m.get('sdlogj_official'), times_ns=dict(identity_eval=times[0], marker=times[1], evaluation=times[2]), described=described)
    return item(4, 'repair_and_evaluation', criterion, PASS if all(checks.values()) else FAIL, values)


def dice_none(labels):
    """labels.json of the frozen worker on zero labels: every Dice of final and baseline None (macro and per label), no class defined."""
    if not isinstance(labels, dict) or not isinstance(labels.get('final'), dict):
        return False
    for part in ('final', 'baseline'):
        for k in ('seg4', 'seg35'):
            r = (labels.get(part) or {}).get(k)
            if not isinstance(r, dict) or r.get('macro_dice') is not None or r.get('defined_classes') not in (0, None) or any(row.get('dice') is not None for row in r.get('per_label') or []):
                return False
    return True


def brain_none(smoothness):
    final = (smoothness or {}).get('final')
    return isinstance(final, dict) and final.get('n_brain_cells') == 0 and all(final.get(k) is None for k in ('sdlogj_brain', 'pct_J_lt_half_brain', 'pct_J_gt_two_brain'))


def disk_bytes(path):
    path = Path(path)
    if not path.exists():
        return None
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file() and not f.is_symlink())


def item5_fallback(root, arm, results=None, stage=None):
    criterion = ("the forced context (forced_settings(), settings_frozen false, the identity checked, test_context); the repair failed (repair_not_certified); line_status hx_fallback_configC with "
                 "configuration C's final_native.npz; fallback.firewall.gate passed; check_prepared ok; the class ok; final_native certified again by the supervisor (passes, the state hash its array's); the "
                 "frozen worker's labels.json every Dice None and smoothness.json's brain items None; exit_code 0 and the whole run (first-use dry pass included) within the 300 s watchdog; the launcher "
                 "of the arm; the ORIGINAL manifest's weights.lam (plain json, one key, here) equal to fallback.firewall.lam_hex bit for bit.  Only described: D0 A0's final_native of the pair")
    d = run_dir(root, arm)
    ctx, line, repair = read_json(d/'context.json'), line_of(d), read_json(d/'repair/repair.json')
    runp = d/'fallback_configc/run'
    if None in (ctx, line, repair):
        return missing_item(5, f'fallback_one_{arm}', criterion, [n for n, x in (('context.json', ctx), ('output/line.json', line), ('repair/repair.json', repair)) if x is None])
    fb = line.get('fallback') if isinstance(line.get('fallback'), dict) else {}      # revision 1, review minor 2: no fallback (the forcing did not force, or another line) is judged on the checks, not as missing
    checks, values = {}, dict(fallback_present=bool(fb))
    checks['forced_context'] = (he.canonical(ctx.get('settings')) == he.canonical(forced_settings()) and ctx.get('settings_frozen') is False and ctx.get('check_identity') is True
                                and ctx.get('test_context') == R.TEST_CONTEXT and line.get('settings_frozen') is False)
    checks['repair_failed'] = repair.get('passed') is False and (line.get('failure') or {}).get('class') == 'repair_not_certified'
    checks['line_status'] = line.get('line_status') == he.FALLBACK_C and fb.get('kind') == 'configC' and fb.get('file') == fc.FINAL
    firewall = fb.get('firewall') or {}
    checks['gate_one'] = (firewall.get('gate') or {}).get('passed') is True
    prepared = read_json(runp/'prepared_check.json') or {}
    checks['check_prepared'] = prepared.get('ok') is True                  # firewall_c.prepare writes it; a failed check_prepared raises there and no candidate follows
    checks['class_ok'] = (fb.get('configC_classification') or {}).get('class') == 'ok'
    cert = fb.get('certificate') or {}
    checks['certified_again'] = he.passes(cert) and cert.get('state_hash') == fb.get('state_hash') and cert.get('state_hash_matches_array') is True
    labels = smoothness = None
    try:
        task = fc.task_dir(runp)
        labels, smoothness = read_json(task/'labels.json'), read_json(task/'smoothness.json')
    except (OSError, KeyError, ValueError) as exc:
        values['task_error'] = f'{type(exc).__name__}: {exc}'
    checks['labels_none'] = dice_none(labels) and brain_none(smoothness)
    classification = read_json(runp/'classification.json') or {}
    evidence = classification.get('evidence') or {}
    exit_code = (runp/'exit_code').read_text().strip() if (runp/'exit_code').is_file() else None
    wall = evidence.get('whole_wall_seconds')
    checks['watchdog'] = exit_code == '0' and finite(wall) and wall < fc.WATCHDOG_SECONDS
    checks['launcher'] = (fb.get('launch') or {}).get('kind') == FORCED_ARMS[arm]
    try:
        lam = original_lam(reference_paths(results, stage)['manifest'])
        checks['lam'] = lam['lam_hex'] == firewall.get('lam_hex')
    except (OSError, KeyError, ValueError, TypeError) as exc:
        lam, checks['lam'] = dict(error=f'{type(exc).__name__}: {exc}'), False
    d0 = {}
    try:
        task0, record0, _ = d0_task(reference_paths(results, stage))
        r0 = read_json(record0) or {}
        d0 = dict(file_sha256=r0.get('sha256'), state_hash=(r0.get('certificate') or {}).get('state_hash'))
    except (OSError, KeyError, ValueError) as exc:
        d0 = dict(error=f'{type(exc).__name__}: {exc}')
    t88_env = (read_json(d/'t88/run.json') or {}).get('env') or {}
    values.update(checks=checks, line_status=line.get('line_status'), failure=line.get('failure'), repair_reason=repair.get('reason'), gate=firewall.get('gate'), classification=fb.get('configC_classification'),
                  candidates=fb.get('candidates'), exit_code=exit_code, whole_wall_seconds=wall, watchdog_seconds=fc.WATCHDOG_SECONDS, launch=fb.get('launch'), lam=dict(original=lam, derived_hex=firewall.get('lam_hex')),
                  described=dict(d0_a0=d0, ours=dict(file_sha256=fb.get('final_native_sha256'), state_hash=fb.get('state_hash')),
                                 file_equal_to_d0=None if 'error' in d0 else d0.get('file_sha256') == fb.get('final_native_sha256'),
                                 array_equal_to_d0=None if 'error' in d0 else d0.get('state_hash') == fb.get('state_hash'),
                                 derived_bytes=disk_bytes(d/'fallback_configc/derived'), run_bytes=disk_bytes(runp), t88_host_peak_rss_gib=t88_env.get('host_peak_rss_gib'),
                                 t88_device_peak_reserved_gib=t88_env.get('device_peak_reserved_gib'), repair_host_peak_rss_gib=((repair.get('env') or {}).get('host_peak_rss_gib')),
                                 prepare_seconds=(fb.get('prepare') or {}).get('seconds'), launch_seconds=(fb.get('launch') or {}).get('seconds')))
    return item(5, f'fallback_one_{arm}', criterion, PASS if all(checks.values()) else FAIL, values)


def dmesg_text():
    done = subprocess.run(['dmesg'], capture_output=True, text=True)
    return None if done.returncode else done.stdout


def dmesg_window(text, start, end):
    """(dxg, oom) between two uptimes (firewall_c.dmesg_counts' patterns, bounded above), (None, None) when dmesg cannot be read or the window is not known."""
    if text is None or start is None or end is None:
        return None, None
    stamp = re.compile(r'^\[ *([0-9.]+)\]')
    dxg = oom = 0
    for line in text.splitlines():
        m = stamp.match(line)
        if m and start <= float(m.group(1)) <= end:
            dxg += bool(re.search('create_allocation failed', line))
            oom += bool(re.search('Out of memory|oom-kill|Killed process', line))
    return dxg, oom


def step_records(root, name):
    return read_json(Path(root)/'steps'/f'{name}.launched.json'), read_json(Path(root)/'steps'/f'{name}.done.json')


def item6_device(root, text):
    criterion = f'b (in-process: t88 and the repair in one process): the device peak (reserved, allocated) of its records; dmesg in the window of the unit (if readable): no dxg or OOM line.  Only described; above {DEVICE_LIMIT_GIB:g} GiB or a driver-level OOM: 不过'
    d = run_dir(root, 'b')
    t88, repair = read_json(d/'t88/run.json'), read_json(d/'repair/repair.json')
    if t88 is None or repair is None:
        return missing_item(6, 'device_memory', criterion, [n for n, x in (('b/t88/run.json', t88), ('b/repair/repair.json', repair)) if x is None])
    rows = {name: {k: (r.get('env') or {}).get(k) for k in ('device_peak_allocated_gib', 'device_peak_reserved_gib', 'host_peak_rss_gib', 'device')} for name, r in (('t88', t88), ('repair', repair))}
    peaks = [v for r in rows.values() for k, v in r.items() if k.startswith('device_peak') and finite(v)]
    launched, done = step_records(root, 'b')
    dxg, oom = dmesg_window(text, (launched or {}).get('uptime_before'), (done or {}).get('uptime_after'))
    device_out = t88.get('status') in GPU_OOM_STATUSES or 'OutOfMemory' in str(repair.get('error') or '')       # revision 1, review minor 6: the 9 GiB cap makes torch raise before a peak above it is seen
    bad = not peaks or max(peaks) > DEVICE_LIMIT_GIB or bool(dxg) or bool(oom) or device_out
    return item(6, 'device_memory', criterion, FAIL if bad else DESCRIBE, dict(records=rows, peak_gib=max(peaks) if peaks else None, dmesg_read=text is not None, dxg_lines=dxg, oom_lines=oom,
                                                                                t88_status=t88.get('status'), repair_error=repair.get('error'), device_out_of_memory=device_out))


def frozen_unit_gib(name):
    """The frozen unit's limit (firewall_c.UNIT_RESOURCES, D0's run_queue.sh) in GiB."""
    return _gib(next(r.split('=', 2)[2] for r in fc.UNIT_RESOURCES if r.startswith(f'--property={name}=')))


def item7_units(root):
    bound = T88_TIMEOUT+REPAIR_TIMEOUT+fc.WATCHDOG_SECONDS+fc.KILL_GRACE+FULL_CERTIFICATES*CERTIFICATE_SECONDS+FALLBACK_SETUP_SECONDS
    frozen_mem, frozen_swap = frozen_unit_gib('MemoryMax'), frozen_unit_gib('MemorySwapMax')
    criterion = (f'the start strings of the supervisor units a, b, c, d: RuntimeMaxSec > T88 {T88_TIMEOUT} + repair {REPAIR_TIMEOUT} + watchdog {fc.WATCHDOG_SECONDS} + {fc.KILL_GRACE} + {FULL_CERTIFICATES} full '
                 f'certificates x {CERTIFICATE_SECONDS:g} s + {FALLBACK_SETUP_SECONDS:g} s = {bound:g} s, and OOMPolicy={OOM_POLICY} (absent: 不过); d (DirectLauncher: the frozen run in this unit) MemoryMax >= '
                 f'{frozen_mem:g} GiB and MemorySwapMax >= {frozen_swap:g} GiB (D0\'s frozen unit).  HE2 (the gate of c): every unit whose process can start configuration C\'s frozen run ({", ".join(FROZEN_RUN_KINDS)}: '
                 f'a, a-eval, b, c, c-eval, d, d-eval) gated at MemAvailable >= {D_MEM_AVAILABLE_GIB:g} GiB (steps/<unit>.launched.json).  HE2 (H2): systemd\'s properties of the supervisor units as their '
                 'own probe read them at the start (output/line.json unit.matches_at_start: MemoryMax, MemorySwapMax, OOMPolicy against the settings\' unit): a reading that differs is 不过; 读不到 (None) is '
                 'recorded, not judged.  HE2 revision 1 (review minors 9, 14): the kernel\'s limits of each supervisor unit\'s cgroup as its probe read them at the start (unit.kernel_matches_at_start: '
                 'memory.max, memory.swap.max against the settings\' unit) the same way (differs: 不过; not read: not judged); the evaluation units a-eval, c-eval, d-eval: OOMPolicy=continue in their strings, '
                 'and their readings (evaluation.json unit) judged the same way against units.evaluation.  Otherwise the strings and the queue\'s record of the gate (the smoke\'s own constants; a holdout '
                 'queue checks its units again)')
    rows, ok = {}, True

    def readings(unit):
        systemd, kernel = unit.get('matches_at_start') or {}, unit.get('kernel_matches_at_start') or {}
        return systemd, kernel, not any(v is False for v in list(systemd.values())+list(kernel.values()))
    for arm in ARMS:
        launched, _ = step_records(root, arm)
        if launched is None:
            return missing_item(7, 'units', criterion, [f'steps/{arm}.launched.json'])
        argv = launched.get('argv') or []
        runtime, memory, swap, oom = (unit_property(argv, k) for k in ('RuntimeMaxSec', 'MemoryMax', 'MemorySwapMax', 'OOMPolicy'))
        unit = (line_of(run_dir(root, arm)) or {}).get('unit') or {}
        matches, kernel, agree = readings(unit)
        rows[arm] = dict(RuntimeMaxSec=runtime, MemoryMax=memory, MemorySwapMax=swap, OOMPolicy=oom, CPUQuota=unit_property(argv, 'CPUQuota'), gate_mem_available_gib=launched.get('gate_mem_available_gib'),
                         read_at_start=matches, kernel_at_start=kernel, systemd_at_start=((unit.get('start') or {}).get('systemd')), systemd_unread=((unit.get('start') or {}).get('systemd_unread')),
                         kernel_limits_at_start={k: (unit.get('start') or {}).get(k) for k in ('max_bytes', 'swap_max_bytes')})
        ok &= runtime is not None and float(runtime) > bound and oom == OOM_POLICY and agree
        if arm == 'd':
            ok &= (_gib(memory) is not None and _gib(memory) >= frozen_mem and _gib(swap) is not None and _gib(swap) >= frozen_swap)
    evaluations = {}
    for step in STEPS:
        if step.kind not in ('evaluate', 'evaluate-forced'):
            continue
        launched, _ = step_records(root, step.name)
        if launched is None:
            return missing_item(7, 'units', criterion, [f'steps/{step.name}.launched.json'])
        unit = (read_json(run_dir(root, step.arm)/'evaluation.json') or {}).get('unit') or {}
        matches, kernel, agree = readings(unit)
        evaluations[step.name] = dict(OOMPolicy=unit_property(launched.get('argv') or [], 'OOMPolicy'), read_at_start=matches, kernel_at_start=kernel)
        ok &= evaluations[step.name]['OOMPolicy'] == OOM_POLICY and agree
    gates, at_launch = {}, {}
    for step in STEPS:
        launched, _ = step_records(root, step.name)
        if launched is None:
            return missing_item(7, 'units', criterion, [f'steps/{step.name}.launched.json'])
        gates[step.name] = launched.get('gate_mem_available_gib')
        at_launch[step.name] = launched.get('mem_available_gib_at_launch')         # HE2 clean: the reading at launch, described only (he_smoke_v2's records have none)
        if step.kind in FROZEN_RUN_KINDS:
            ok &= finite(gates[step.name]) and gates[step.name] >= D_MEM_AVAILABLE_GIB
    read = [arm for arm, r in rows.items() if r['read_at_start'] and all(v is not None for v in r['read_at_start'].values())]
    kernel_read = [arm for arm, r in rows.items() if r['kernel_at_start'] and all(v is not None for v in r['kernel_at_start'].values())]
    return item(7, 'units', criterion, PASS if ok else FAIL, dict(units=rows, evaluations=evaluations, gates=gates, described=dict(mem_available_gib_at_launch=at_launch),
                                                                runtime_bound_seconds=bound, memory_bound_gib_d=frozen_mem, swap_bound_gib_d=frozen_swap,
                                                                gate_bound_gib_frozen_run=D_MEM_AVAILABLE_GIB, systemd_read_for=read, kernel_read_for=kernel_read,
                                                                note='OOMPolicy as the strings set it and, where they could be read, as systemd applied it and the kernel limits the cgroup; no child was killed by the '
                                                                     'OOM killer in this smoke (item 12 reads the cgroups\' memory.events)'))


def item8_dmesg(root):
    criterion = 'c, d: whether the supervisor read dmesg (fallback.launch.dmesg_read); when it could not, dxg and OOM recorded None and handed to classify_run as 0.  Only described'
    rows = {}
    for arm in ('c', 'd'):
        d = run_dir(root, arm)
        fb = ((line_of(d) or {}).get('fallback') or {})
        launch = fb.get('launch')
        if launch is None:
            return missing_item(8, 'dmesg', criterion, [f'{arm}/output/line.json fallback.launch'])
        evidence = (read_json(d/'fallback_configc/run/classification.json') or {}).get('evidence') or {}
        rows[arm] = dict(dmesg_read=launch.get('dmesg_read'), dxg_lines=launch.get('dxg_lines'), oom_lines=launch.get('oom_lines'), classify_dxg=evidence.get('dxg_lines'), classify_oom=evidence.get('oom_lines'),
                         none_as_zero=None if launch.get('dmesg_read') else (launch.get('dxg_lines') is None and launch.get('oom_lines') is None and evidence.get('dxg_lines') == 0 and evidence.get('oom_lines') == 0))
    return item(8, 'dmesg', criterion, DESCRIBE, rows)


def time_identity(code_root=None, repeat=2):
    """Item 9: one identity check (the frozen tree and the checkout) timed in this process, as every HE process does it."""
    out = []
    for _ in range(repeat):
        tick = perf_counter()
        he.identity(core.ROOT if code_root is None else code_root, frozen=core.FROZEN, check_pythonpath=False)
        out.append(perf_counter()-tick)
    return out


def item9_timing(root, measured):
    criterion = ('the identity checks of each process (start and end): their records per run directory with checked_utc; the supervisor\'s first one from the file times (started.json -> identity.json); '
                 'one check timed by this process.  Only described')
    rows = {}
    for arm in ARMS:
        d = run_dir(root, arm)
        files = sorted(d.glob('identity*.json'))
        if not files:
            return missing_item(9, 'timing', criterion, [f'{arm}/identity*.json'])
        first = (d/'identity.json').stat().st_mtime-(d/'started.json').stat().st_mtime if (d/'identity.json').is_file() and (d/'started.json').is_file() else None
        rows[arm] = dict(n_checks=len(files), records={f.name: (read_json(f) or {}).get('checked_utc') for f in files}, supervisor_first_check_seconds=first)
    return item(9, 'timing', criterion, DESCRIBE, dict(arms=rows, measured_seconds=measured, total_checks=sum(r['n_checks'] for r in rows.values())))


def cgroup_row(probe):
    """The probe's numbers a reader needs (unit_probe.py), 读不到 as recorded."""
    probe = probe or {}
    stat = probe.get('memory_stat') or {}
    gib = lambda v: v/2**30 if isinstance(v, int) else None     # noqa: E731
    return dict(unit=probe.get('unit'), in_tag_unit=probe.get('in_tag_unit'), peak_gib=gib(probe.get('peak_bytes')), max=probe.get('max_bytes'),
                swap_max=probe.get('swap_max_bytes'), events=probe.get('events'), unread=probe.get('unread') or probe.get('cgroup_unread') or probe.get('error'), systemd=probe.get('systemd'),
                systemd_unread=probe.get('systemd_unread'), anon_gib=gib(stat.get('anon')), file_gib=gib(stat.get('file')), memory_stat_unread=probe.get('memory_stat_unread'))     # HE2 clean: described only


def oom_killed(probe):
    return bool(((probe or {}).get('events') or {}).get('oom_kill'))


def item12_memory(root):
    criterion = ("C22.B4-6 (d), only described: the processes' peaks of a, b, c, d (t88, repair: host_peak_rss_gib, device peaks); each supervisor unit's cgroup at its end (output/line.json unit: memory.peak, "
                 "memory.events, the limits) and systemd's properties at its start; the frozen run's unit (fallback_configc/run/unit_probe_*.json: c's nested unit, d's own unit); the evaluation units "
                 "(evaluation.json unit).  The numbers are the kernel's and systemd's as read (unit_probe.py), 读不到 where they could not be read; they are evidence of the limits, not a proof that a limit "
                 "acted.  不过 only when the unit record the code writes is missing from a line, or a cgroup's memory.events counts an oom_kill")
    rows, bad, missing = {}, [], []
    for arm in ARMS:
        d = run_dir(root, arm)
        line = line_of(d)
        if line is None or 'unit' not in line:
            missing.append(f'{arm}/output/line.json unit')
            continue
        t88, repair = read_json(d/'t88/run.json') or {}, read_json(d/'repair/repair.json') or {}
        unit = line['unit'] or {}
        row = dict(t88=dict((t88.get('env') or {}), seconds_total=t88.get('seconds_total'), seconds_tissue=t88.get('seconds_tissue')), repair=dict((repair.get('env') or {}), seconds_total=repair.get('seconds_total')),
                   supervisor_end=cgroup_row(unit.get('end')), supervisor_start_systemd=(unit.get('start') or {}).get('systemd'), matches_at_start=unit.get('matches_at_start'))
        row['t88'] = {k: row['t88'].get(k) for k in ('host_peak_rss_gib', 'device_peak_reserved_gib', 'device_peak_allocated_gib', 'seconds_total', 'seconds_tissue')}
        row['repair'] = {k: row['repair'].get(k) for k in ('host_peak_rss_gib', 'device_peak_reserved_gib', 'device_peak_allocated_gib', 'seconds_total')}
        if oom_killed(unit.get('end')):
            bad.append(f'{arm}: the supervisor unit counts an oom_kill')
        probes = fc.probes_of(d/'fallback_configc/run') if (d/'fallback_configc/run').is_dir() else {}
        if probes:
            row['frozen_run'] = {name: cgroup_row(p) for name, p in probes.items()}
            if any(oom_killed(p) for p in probes.values()):
                bad.append(f'{arm}: the frozen run\'s cgroup counts an oom_kill')
        ev = read_json(d/'evaluation.json')
        if ev is not None:
            row['evaluation_end'] = cgroup_row((ev.get('unit') or {}).get('end'))
            if oom_killed((ev.get('unit') or {}).get('end')):
                bad.append(f'{arm}: the evaluation unit counts an oom_kill')
        rows[arm] = row
    if missing:
        return missing_item(12, 'memory_evidence', criterion, missing)
    return item(12, 'memory_evidence', criterion, FAIL if bad else DESCRIBE, dict(arms=rows, oom_kills=bad, contract='C22.B4-6 (d)',
                                                                                  note='the processes\' peaks do not add up to the unit\'s: memory.peak is the cgroup\'s (every process of the unit)'))


def item13_forcing(root, results=None):
    criterion = ("C22.B4-6 (e): the forcing (max_rounds 1, max_inner 0) needs C's field not to pass its certificate before the repair: p3_v1/runs/RC_c600/0438-0439/repair.json initial_certificate not "
                 "certified_feasible (the reference: infeasible, 21462 cells); c and d's own repair.json initial_certificate not a passing one (C07-3) and their reason round_cap.  Their initial state hash "
                 "against p3's: described")
    p3 = read_json(reference_paths(results)['p3_repair'])
    rows, ok = {}, True
    if p3 is None:
        return missing_item(13, 'forcing_precondition', criterion, [str(reference_paths(results)['p3_repair'])])
    ref = p3.get('initial_certificate') or {}
    ok &= ref.get('status') not in (None, he.CERT_PASS)
    for arm in ('c', 'd'):
        repair = read_json(run_dir(root, arm)/'repair/repair.json')
        if repair is None:
            return missing_item(13, 'forcing_precondition', criterion, [f'{arm}/repair/repair.json'])
        initial = repair.get('initial_certificate') or {}
        rows[arm] = dict(status=initial.get('status'), n_infeasible=initial.get('n_infeasible'), state_hash=initial.get('state_hash'), reason=repair.get('reason'),
                         same_state_as_p3=None if initial.get('state_hash') is None else initial.get('state_hash') == ref.get('state_hash'))
        ok &= bool(initial) and not he.passes(initial) and repair.get('reason') == 'round_cap'
    return item(13, 'forcing_precondition', criterion, PASS if ok else FAIL, dict(reference={k: ref.get(k) for k in ('status', 'n_infeasible', 'state_hash')}, arms=rows, contract='C22.B4-6 (e)'))


def check(root, *, results=None, stage=None, recompute=None, dmesg=None, measure=None):
    """Items 1-13 from the outputs; writes <root>/smoke_report(.json|.md) (numbered if there).  Returns the report."""
    root = Path(root)
    batch = read_json(root/'batch.json')
    text = dmesg_text() if dmesg is None else dmesg()
    measured = (measure or (lambda: time_identity(Path(batch['code']) if batch and batch.get('code') else None)))()
    off = c_off(root)                                   # HB: the method with C off (C22 卷首修订 8): item 3 is the field against HS's, items 10, 11 do not apply, item 13 reads HS's repair
    items = [item1_identity(root, batch), item2_spy(root), item3_field_off(root, results) if off else item3_pass0(root, results), item4_repair_and_evaluation(root, results, recompute, stage),
             item5_fallback(root, 'c', results, stage), item5_fallback(root, 'd', results, stage), item6_device(root, text), item7_units(root), item8_dmesg(root), item9_timing(root, measured),
             not_applicable(10, 't88_pass2', 'C off (C22 卷首修订 8): there is no pass 2') if off else item10_pass2(root, results),
             not_applicable(11, 'c_ratios', 'C off (C22 卷首修订 8): there are no ratios of C') if off else item11_ratios(root, results),
             item12_memory(root), item13_forcing_off(root, results) if off else item13_forcing(root, results)]
    counts = {v: sum(i['verdict'] == v for i in items) for v in (PASS, FAIL, OPEN, DESCRIBE)}
    summary = '; '.join(f"{i['item']}{'' if i['item'] != 5 else i['name'][-2:]} {i['verdict']}" for i in items)
    method = he.NAME if off else he.HE2_NAME
    report = dict(schema=SCHEMA.format('report'), root=str(root), pair=PAIR, code_sha=(batch or {}).get('code_sha'), items=items, counts=counts, summary=summary, tissue_prior='off' if off else 'moment3',
                  wording=f'开发集单次（0438-0439）；输出是认证终态，不是正式状态，不是 F_B；方法 {method}；a 的 Dice 是 {"HS" if off else "P3"} 这一对的重复，c、d 的是 D0 A0 的重复，都不算选择读数；seg35 与 seg4 都报',
                  truth_read='item 4: the check read the truth of the pair once more, after evaluation.json existed (bayesreg.eval.truth), to recompute the metrics', utc=he.utc())
    k = 0
    while (root/(REPORT+('' if k == 0 else f'_{k}')+'.json')).exists() or (root/(REPORT+('' if k == 0 else f'_{k}')+'.md')).exists():
        k += 1
    name = REPORT+('' if k == 0 else f'_{k}')
    write_once(root/f'{name}.json', report)
    with (root/f'{name}.md').open('x', encoding='utf-8') as f:
        f.write(report_markdown(report))
    print(f'smoke check: {summary}')
    return report


def _short(value, n=220):
    text = json.dumps(value, ensure_ascii=False, default=he._default)
    return text if len(text) <= n else text[:n-1]+'…'


def report_markdown(report):
    lines = [f"# HES 冒烟读数（{report['pair']}）", '', f"> check 写于 {report['utc']}（UTC）；代码 {report['code_sha']}；根目录 {report['root']}。", f"> {report['wording']}。", f"> {report['truth_read']}。", '',
             f"计数：{'、'.join(f'{k} {v}' for k, v in report['counts'].items())}。", '', '| 项 | 名称 | 判定 | 判据 | 读到的值（摘要；全文见 JSON） |', '|---|---|---|---|---|']
    for i in report['items']:
        values = i['values']
        brief = values.get('checks', values) if isinstance(values, dict) else values
        lines.append(f"| {i['item']} | {i['name']} | {i['verdict']} | {i['criterion'][:300].replace('|', '/')} | {_short(brief).replace('|', '/')} |")
    lines += ['', '只描述的项不判过；"不过（待查）"只记录，不解释原因。']
    return '\n'.join(lines)+'\n'


# ---- estimates and the dry run ----------------------------------------------------------------------------------------------------------------------------
PROCESS_SECONDS, IDENTITY_SECONDS, JIT_SECONDS = 15., 10., 30.      # 〔估算〕a process's imports; one identity check; a cold numba cache in a new directory
FORCED_REPAIR_SECONDS, FALLBACK_SECONDS, FIRST_USE_SECONDS, EVAL_SECONDS, CHECK_SECONDS = 40., 150., 60., 30., 120.     # 〔估算〕nobody has run these on a real pair


def estimates(results=None):
    """〔估算〕seconds per unit, from p3_v1's C field unit of the pair (three passes, seconds_total), p3_v1's repair of it (seconds_total) and D0 A0's frozen run (whole wall), and the constants above (HE2: C on, so
    P3's records in the place of fd2_v1's and rw_v1's)."""
    try:
        r = reference_records(results)
    except (OSError, KeyError, ValueError, TypeError):
        return None
    t88, repair, frozen = r['p3']['seconds_total'], r['p3_repair']['seconds_total'], r['d0']['whole_wall_seconds']
    if None in (t88, repair, frozen):
        return None
    supervisor = 3*PROCESS_SECONDS+6*IDENTITY_SECONDS
    evaluation = PROCESS_SECONDS+2*IDENTITY_SECONDS+JIT_SECONDS+EVAL_SECONDS
    forced = supervisor+t88+FORCED_REPAIR_SECONDS+FALLBACK_SECONDS+frozen+FIRST_USE_SECONDS+JIT_SECONDS
    units = {'a': supervisor+t88+repair+JIT_SECONDS, 'a-eval': evaluation, 'b': supervisor+t88+repair+JIT_SECONDS, 'c': forced, 'c-eval': evaluation, 'd': forced, 'd-eval': evaluation, 'check': CHECK_SECONDS}
    return dict(t88=t88, repair=repair, frozen=frozen, units=units, gpu_seconds=sum(units[k] for k in 'abcd'), total_seconds=sum(units.values())+5*len(units))


def dry_run(*, root=DEFAULT_ROOT, results=None, stage=None, out=print):
    """See the module docstring: nothing written, torch not imported, no large file read or hashed."""
    root = Path(root)
    problems = []
    out(f'HES dry run ({run.utc()}): no unit, no GPU, no truth label, nothing written; JSON records and npy headers only')
    out(f'pair {PAIR} of {STAGE}; root {root} (exists: {root.exists()}); launched units there: {launched_steps(root) if root.exists() else []}')
    if root.exists() and launched_steps(root):
        problems.append(f'{root}: units were launched there: a smoke root runs once')
    problem = root_problem(root)
    if problem:
        problems.append(problem)
    frozen, forced = he.frozen_settings(), forced_settings()
    changed = sorted(k for k in set(frozen) | set(forced) if he.canonical(frozen.get(k)) != he.canonical(forced.get(k)))
    out(f"settings: frozen sha256 {he.settings_sha256(frozen)} ({frozen['name']}: segmenter {frozen['segmenter']['segmenter_id']}, tissue_prior {frozen['tissue_prior']}, {frozen['c']['passes']} passes; "
        f"{frozen['contract']}); forced sha256 {he.settings_sha256(forced)}, top-level keys changed {changed}, repair {FORCED_REPAIR}")
    if not he.is_frozen(frozen) or he.is_frozen(forced) or changed != ['name', 'repair', 'smoke_forcing']:
        problems.append(f'the forced values are not the frozen ones changed in name, repair and smoke_forcing only: {changed}')
    anchor_problems, anchor = he.segmenter_start_problems(frozen)
    out(f"segmenter anchor (C22.9-4 (xv)): {'ok' if not anchor_problems else anchor_problems}; registry {anchor['registry_rel']} sha256 {anchor['registry_sha256']}; models {anchor['model_segmenter_ids']}")
    problems += [f'anchor: {x}' for x in anchor_problems]
    record, id_problems = he.identity(core.ROOT, frozen=core.FROZEN, check_pythonpath=False)
    out(f"identity of this checkout {core.ROOT} (HEAD {(record.get('repo') or {}).get('head')}; PYTHONPATH not checked here, the units set <frozen>:<root>/code): {'ok' if not id_problems else id_problems}")
    problems += [f'identity: {p}' for p in id_problems]
    adm = admission_problems(stage)
    out(f"admission (sealed lists on the names and the manifest, then the allowlist; pair.npz not opened): {'ok' if not adm else adm}")
    problems += adm
    out('--- references (the sha256 the records name; nothing recomputed)')
    try:
        refs = reference_records(results, stage)            # it checks that the records name the contract's hashes (REF_PASS0_U_SHA256, REF_PASS2_U_SHA256)
        t = refs['t88']
        shape, dtype = run.npz_header(t['path'], 'u')
        out(f"item 3 (C22.B4-6 (a), pass 0): {t['path']}: file sha256 {t['file_sha256']} (hs_v1's repair read it as {t['repair_source_field_sha256']}: "
            f"{'same' if t['repair_source_field_sha256'] == t['file_sha256'] else 'DIFFERENT'}), u_sha256 {t['u_sha256']} (the contract's {REF_PASS0_U_SHA256[:8]}…), u {tuple(shape)} {dtype} (npy header), "
            f"status {t['status']}, env {t['env']}, workers {t['workers']}")
        c = refs['p3']
        shape2, dtype2 = run.npz_header(c['path'], 'u')
        out(f"item 10 (C22.B4-6 (b), pass 2): {c['path']}: file sha256 {c['file_sha256']}, u_sha256 {c['u_sha256']} (the contract's {REF_PASS2_U_SHA256[:8]}…), u {tuple(shape2)} {dtype2} (npy header), status {c['status']}, "
            f"segmenter {c['segmenter_id']}; its three passes {[x[:8]+'…' for x in c['pass_u_sha256']]}, host peak {c['host_peak_rss_gib']} GiB, device reserved {c['device_peak_reserved_gib']} GiB, {c['seconds_total']} s")
        out(f"item 11 (C22.B4-6 (c)): p3_v1's ratios used in passes 1 and 2: {c['ratios_used']}; the estimate on pass 2's field (described): {c['last_estimate']}")
        w, h = refs['p3_repair'], refs['hs']
        out(f"item 4 (only described): {w['repair']} (P3: passed {w['passed']}, {w['rounds']} rounds, {w['seconds_total']} s) and {h['repair']} (HS: passed {h['passed']}, {h['rounds']} rounds); "
            f"their repair_metrics.json (repaired_u_sha256 {w['repaired_u_sha256']}, {h['repaired_u_sha256']}; their seg35 and seg4 are read by the check only)")
        out(f"item 13 (C22.B4-6 (e)): P3's repair of the pair before repair: {w['initial_certificate']} (the forcing needs a field that does not pass)")
        checked = reference_problems(refs, anchor=False)      # the same checks as the queue's input check (HE2 review minor 10); the anchor is asked above
        out(f"the references' checks (items 3, 10, 13; the queue asks them too): {'ok' if not checked else checked}")
        problems += checked
        z = refs['d0']
        shape0, dtype0 = run.npz_header(z['final_native'], 'u')
        out(f"item 5 (only described): {z['final_native']}: file sha256 {z['file_sha256']}, state hash {z['state_hash']}, role {z['role']}, class {z['classification']}, whole wall {z['whole_wall_seconds']} s, u {tuple(shape0)} {dtype0}")
        m = refs['manifest']
        out(f"item 5 (judged): the original manifest {m['path']}: weights.lam {m['lam']!r} = {m['lam_hex']} (one key, plain json); the derived pair must carry the same lam_hex")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        problems.append(f'references: {type(exc).__name__}: {exc}'[:1000])
        out(f'references: {problems[-1]}')
    e = estimates(results)
    if e is not None:
        out(f"estimates〔估算〕: T88 with C's three passes {e['t88']:.0f} s (p3_v1 C_first5 seconds_total), repair {e['repair']:.0f} s (p3_v1 RC_c600), configuration C {e['frozen']:.0f} s (d0_v1 whole wall); units "
            + ', '.join(f'{k} {v/60:.1f} min' for k, v in e['units'].items())+f"; GPU units {e['gpu_seconds']/60:.0f} min; all with the gaps {e['total_seconds']/60:.0f} min, plus any wait at the gate")
    gate = GATE
    out(f'gate: own unit {gate.own_units}, load < {gate.load_limit}, GPU free >= {gate.gpu_free_mib} MiB (GPU units), MemAvailable >= {D_MEM_AVAILABLE_GIB} GiB for the units that can start the frozen run '
        f'({", ".join(s.name for s in STEPS if s.kind in FROZEN_RUN_KINDS)}), {gate.mem_available_gib} GiB for the others, /home >= {run.HOME_FLOOR_GB} GB, '
        f'/mnt/c >= {gate.host_floor_gib} GiB, no other tag-* or seq-vp-* unit, no {root}/hold; a STOP after 30 minutes closed; the supervisor units: MemoryMax {SUPERVISOR["memory_max"]}, MemorySwapMax 4G, OOMPolicy={SUPERVISOR["oom_policy"]}')
    out(f'/home free {shutil.disk_usage("/home").free/2**30:.0f} GB; /mnt/c free {shutil.disk_usage("/mnt/c").free/2**30:.0f} GiB')
    out(f'--- start strings (the queue: the lead, from the checkout that runs this dry run, {core.ROOT}, whose HEAD is the commit to run; GPU alone; the units: the queue, one at a time, the code at $ROOT/code)')
    out(shlex.join(queue_launch(root)))
    code = root/'code'
    for step in STEPS:
        out(f'{step.name:7s} {shlex.join(step_argv(step, code, root))}')
    out(f"nested (c only, by firewall_c.UnitLauncher): systemd-run --user --unit=tag-he-c-{SHORT} {' '.join(fc.UNIT_RESOURCES)} {fc.UNIT_OOM_POLICY} ... (the frozen run, D0's unit; HE2: OOMPolicy and the "
        f"unit probe {list(fc.PROBE_FILES)}); after a failed third certificate (H3): --unit={R.THIRD_UNIT_PREFIX}-{SHORT}, the same properties")
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


# ---- the entries -------------------------------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['--dry-run']:
        d = argparse.ArgumentParser(description='HES: the dry run')
        d.add_argument('--dry-run', action='store_true')
        d.add_argument('--root', default=str(DEFAULT_ROOT))
        return dry_run(root=Path(d.parse_args(argv).root))
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(DEFAULT_ROOT))
    q.add_argument('--commit', default=None)
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--gap-seconds', type=int, default=5)
    q.add_argument('--min-host-gib', type=float, default=15.)
    c = sub.add_parser('check')
    c.add_argument('--root', required=True)
    f = sub.add_parser('forced')
    f.add_argument('--launcher', choices=('unit', 'direct'), required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--device', default='cuda')
    ch = sub.add_parser('child')
    ch.add_argument('stage', choices=('t88', 'repair', 'third-cert'))
    ch.add_argument('--context', required=True)
    ev = sub.add_parser('evaluate-forced')
    ev.add_argument('--context', required=True)
    o = ap.parse_args(argv)
    if o.command == 'queue':
        return queue(o)
    if o.command == 'check':
        problem = root_problem(o.root)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        check(o.root)
        return 0
    if o.command == 'forced':
        return forced_main(o)
    if o.command == 'child':
        return child_main(o.stage, o.context)
    return evaluate_forced_main(o.context)


if __name__ == '__main__':
    sys.exit(main())
