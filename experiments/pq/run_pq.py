"""PQ (contract C22.Q, revision 5 adopted by BZ 2026-11-07; runs/20261107-pq/plan.md, final, sha256 pinned in pq_pins.PLAN_SHA256; runs/20261107-pq/任务书_工具.md): the four units of plan.md section 12.

    python -B experiments/pq/run_pq.py members --dry-run | out --dry-run | eval --dry-run | decide --dry-run
        CPU; no image, no truth, no GPU, nothing written; the checks a unit makes before it reads anything, the start string, the estimates.
    run_pq.py members [--device cuda]   (unit tag-pq-members, GPU; D91 (a))   the four members inferred again, V per subject, m5
    run_pq.py out                       (unit tag-pq-out, CPU)                  m1-m4 on all 19 pairs, then PQ and the label-free baselines, opened_files.json, out_record.json, the sha256 list (m6)
        -> the lead reads opened_files.json, releases, and commits runs/20261107-pq/outputs_manifest.sha256 and runs/20261107-pq/release.md in one commit (the printed lines).
    run_pq.py eval                      (unit tag-pq-eval, CPU)                 the release, e2, the guard, the lock, the truth once per pair, e1
    run_pq.py decide                    (the lead, CPU)                          the reading of plan.md section 6 and the descriptions of section 7 (analysis/)

The units run from the main checkout (clean, its HEAD recorded); out and eval with PYTHONPATH=<frozen>:<code> (boundary_taper of the frozen tree, C22.Q1-2).  A unit refuses to run outside its own systemd unit
(/proc/self/cgroup), beside any other active tag-* or seq-vp-* unit, with less than 100 GB free on /home, or with too little memory; members also with a busy GPU.  Every output directory is new; nothing is
overwritten, a failure is kept and not rerun (project rules, hard rule 4).  Exit codes: 0 done, 2 usage, 8 refused or stopped.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

import numpy as np  # noqa: E402

import pq_core as core  # noqa: E402
import pq_firewall as fw  # noqa: E402
import pq_io as io_  # noqa: E402
import pq_pins as pins  # noqa: E402
import pq_stats as stats  # noqa: E402
from pq_core import PQRefused  # noqa: E402

PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
SCRIPT = 'experiments/pq/run_pq.py'
UNITS = dict(members='tag-pq-members', out='tag-pq-out', eval='tag-pq-eval')
MEMORY_MAX = dict(members='12G', out='11G', eval='11G')
MEM_AVAILABLE_GIB = dict(members=8., out=9., eval=9.)
HOME_FREE_GB = 100.
GPU_UTIL_MAX = 20
# Free device memory needed before the members unit starts (revision 1, review minor 7; revision 2, diff review minor 6: recomputed by the tensors alive together, on the v4 members' own record).
# Base: TTA ran the registered inference with these same four v4 members (its arm unflipped is hs_seg.uf4_probabilities) and recorded gpu_peak_allocated_gib 4.95 GiB
# (/home/rmcfz/seq_vp_tag_results/segprobe_v1/tta_v4/flip_tta.json; that peak covers the flip pass too, so it bounds the inference alone from above).  One float64 (36, 192, 224, 160) stack is 1.85 GiB.
# 〔估算〕alive together, the largest of:
#   (a) a member's forward pass (members 2-4): the inference's working set <= 4.95 + the float64 sums of p_m and p_m^2 (3.69)                                   = about 8.6 GiB allocated;
#   (b) right after it: the logits .92 + the float32 softmax .92 + p_m in float64 1.85 + the two sums 3.69 + the ensemble's float32 running total .92          = about 8.3 GiB;
#   (c) after the loop, the variance in place (div_, addcmul_): two float64 stacks                                                                            = about 3.7 GiB.
# So about 8.6 GiB allocated; with TTA's reserved 6.55 GiB in place of its allocated, about 10.2 GiB reserved.  Threshold 12 GiB (was 14 in revision 1): about 1.8 GiB over the reserved estimate,
# about 3.4 GiB over the allocated one (PyTorch frees its cache before an out-of-memory error, so the allocated peak is what counts).  The cap of the process stays hs_seg's 20 GiB.  The run records
# device_peak_allocated_gib / device_peak_reserved_gib (members_record.json device_peak), against which this estimate is checked afterwards.
GPU_FREE_MIN_GIB, GPU_CAP_GIB = 12., 20.
SMI = Path('/usr/lib/wsl/lib/nvidia-smi')
GATE_PATTERNS = ('tag-*', 'seq-vp-*')
ESTIMATES = dict(members='〔估算〕GPU 10-15 分钟（20 人 x 4 成员；TTA 的 20 x 4 x 2 方向约 15 分钟，plan.md 第 11 节）；磁盘 V 20 张约 .55 GB',
                 out='〔估算〕CPU 两遍：m1-m4 全部 19 对约 10-25 分钟，计算与写盘 19 对约 15-40 分钟（全尺寸合成数组上的纯计算约 15 s 一对，余下是读 probs36 与 structure_groups 三遍）；'
                     '主机峰值约 5-6.5 GiB（两侧 float64 后验各约 2 GB）；磁盘约 4.8 GB（每对 pq.npz 约 55 MB、baselines.npz 约 200 MB）',
                 eval='〔估算〕CPU 每对约 30-90 s（真值、v 再绑定一次、10 个分数 x 3 种掩膜，全尺寸合成数组上 score_record 每次约 .6 s），19 对约 15-30 分钟；磁盘 eval.json 与 e.npz 约 .1-.3 GB',
                 decide='〔估算〕CPU 几分钟（B 类 8 个基线各 10000 次 block_interval，A 类 10000 次有效抽样）')


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def host_record():
    """The environment and the host peak of this process (C22.Q5 运行): Python, numpy, scipy versions; ru_maxrss (a running maximum, GiB)."""
    import platform
    import resource
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, pid=os.getpid())


# ---- the start strings ------------------------------------------------------------------------------------------------------------------------------
def start_string(unit, repo=pins.MAIN_REPO, frozen=pins.FROZEN, device='cuda'):
    env = dict(PYTHONDONTWRITEBYTECODE=1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1,
               PYTHONPATH=str(repo) if unit == 'members' else f'{frozen}:{repo}')
    tail = ['members', '--device', device] if unit == 'members' else [unit]
    return ['systemd-run', '--user', f'--unit={UNITS[unit]}', '--collect', f'--working-directory={repo}', f'--property=MemoryMax={MEMORY_MAX[unit]}', '--nice=10',
            *[f'--setenv={k}={v}' for k, v in env.items()], PY, '-B', SCRIPT, *tail]


# ---- the machine and the code -----------------------------------------------------------------------------------------------------------------------
def in_own_unit(unit, cgroup='/proc/self/cgroup'):
    try:
        text = Path(cgroup).read_text(encoding='utf-8')
    except OSError:
        return False
    return re.search(rf'/{re.escape(UNITS[unit])}\.service(/|$)', text, re.MULTILINE) is not None


def mem_available_gib(meminfo='/proc/meminfo'):
    for line in Path(meminfo).read_text(encoding='utf-8').splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/2**20
    return 0.


def machine_gate(unit, *, run=subprocess.run, meminfo='/proc/meminfo', home='/home', smi=SMI):
    """No other active or activating tag-* / seq-vp-* unit, /home >= 100 GB free, MemAvailable; for members also a quiet GPU.  PQRefused('machine') otherwise."""
    record = dict(unit=UNITS[unit], checked_utc=utc())
    try:
        listing = run(['systemctl', '--user', 'list-units', '--state=active,activating', '--no-legend', '--plain', *GATE_PATTERNS], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PQRefused('machine', f'systemctl could not be queried ({exc})') from None
    if listing.returncode != 0:
        raise PQRefused('machine', f'systemctl --user list-units failed ({listing.stderr.strip()})')
    active = sorted(line.split()[0] for line in listing.stdout.splitlines() if line.strip())
    others = [u for u in active if u.removesuffix('.service') != UNITS[unit]]
    record.update(active_units=active, other_units=others)
    if others:
        raise PQRefused('machine', f'other units are active: {others}; this unit runs alone')
    free_gb = shutil.disk_usage(home).free/1e9
    available = mem_available_gib(meminfo)
    record.update(home_free_gb=free_gb, mem_available_gib=available)
    if free_gb < HOME_FREE_GB:
        raise PQRefused('machine', f'/home has {free_gb:.0f} GB free, {HOME_FREE_GB:.0f} GB required (project rules)')
    if available < MEM_AVAILABLE_GIB[unit]:
        raise PQRefused('machine', f'MemAvailable {available:.1f} GiB < {MEM_AVAILABLE_GIB[unit]} GiB')
    if unit == 'members':
        try:
            apps = run([str(smi), '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, timeout=60)
            util = run([str(smi), '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            raise PQRefused('machine', f'nvidia-smi could not be queried ({exc})') from None
        pids = [p.strip() for p in apps.stdout.splitlines() if p.strip().isdigit()]
        reading = (util.stdout.strip().splitlines() or [''])[0].strip()
        if apps.returncode != 0 or util.returncode != 0 or not reading.isdigit():
            raise PQRefused('machine', f'nvidia-smi gave no reading ({reading!r})')
        record.update(gpu_compute_pids=pids, gpu_utilization_percent=int(reading))
        if pids or int(reading) > GPU_UTIL_MAX:
            raise PQRefused('machine', f'GPU busy (pids {pids}, utilization {reading} %)')
    record['passed'] = True
    return record


def code_state(repo=REPO, *, frozen_required):
    """HEAD and cleanliness of the code tree, where bayesreg and this package were imported from; with frozen_required, jd_curl from the frozen tree (mc0_core.frozen_api: preopt.py's pinned sha256)."""
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, timeout=60)
    head, status = git('rev-parse', 'HEAD'), git('status', '--porcelain')
    import bayesreg
    import jd_curl
    record = dict(root=str(repo), commit=head.stdout.strip() or None, clean=head.returncode == 0 and status.returncode == 0 and not status.stdout.strip(), dirty=status.stdout.splitlines()[:20],
                  bayesreg=str(Path(bayesreg.__file__).resolve().parent), jd_curl=str(Path(jd_curl.__file__).resolve().parent), pq=str(HERE),
                  files={p.name: io_.sha_file(p) for p in sorted(HERE.glob('*.py'))})
    problems = []
    if not record['clean']:
        problems.append(f"the code tree {repo} is not clean at {record['commit']}: {record['dirty'][:5]}")
    if not Path(record['bayesreg']).is_relative_to(Path(repo).resolve()) or not HERE.is_relative_to(Path(repo).resolve()):
        problems.append(f"bayesreg ({record['bayesreg']}) or this package ({HERE}) is not from {repo}")
    if frozen_required:
        sys.path.insert(1, str(HERE.parent/'mc0'))
        import mc0_core
        try:
            mc0_core.frozen_api()
            record['frozen'] = dict(root=str(mc0_core.FROZEN), commit=mc0_core.FROZEN_COMMIT, preopt_sha256=mc0_core.FROZEN_PREOPT_SHA256)
        except SystemExit as exc:
            problems.append(f'the frozen tree: {exc}')
    if problems:
        raise PQRefused('code', str(problems))
    return record


def frozen_taper():
    import mc0_core
    return mc0_core.frozen_api().boundary_taper


def production_context(*, taper=None):
    return io_.Context(repo=REPO, source_root=pins.MAIN_REPO, results=pins.RESULTS, images=pins.DATA_IMAGES, taper=taper)


SEALED_CACHE_POPULATED = 'sealed list cache already populated'


def sealed_cache_problem(unit):
    """C23.6-5 (SG, revision 6g; the read-only query of C23.2-3): before the spy is installed, the guard's cache of the sealed lists must be empty, so that the guard's
    first read of the lists happens under the spy and is recorded (the lists are allowed items, C22.Q1-3).  None, or the reason to stop."""
    from bayesreg.io.sealed import sealed_cache_is_empty
    if sealed_cache_is_empty():
        return None
    return f'{unit}: {SEALED_CACHE_POPULATED} before the spy was installed: the guard would read the lists unseen (C23.6-5); a unit runs in a process of its own'


def rules_for(ctx, unit):
    member_files = [Path(ctx.results)/pins.MODELS_ROOT_REL/name/'member.pt' for _, name, _ in pins.MEMBERS]
    return fw.Rules(repo=ctx.repo, results=ctx.results, data=Path(ctx.source_root)/'data', images=ctx.images, pairs=[pins.pair_name(s) for s in ctx.shorts], subjects=ctx.subjects(),
                    member_files=member_files, p3_dirs={s: (ctx.p3[s].t88_dir, ctx.p3[s].repair_dir) for s in ctx.shorts}, unit=unit, pq_root=ctx.pq_root, pred_root=ctx.pred_root,
                    p3_root=Path(ctx.results)/pins.P3_ROOT_REL, extra_allowed=[Path(ctx.repo)/pins.PLAN_REL])


def preload(unit):
    """Import, before the spy is installed, every module the unit will call, so that what the spy records is the unit's reading, not an import's (an import that read a file would be recorded too)."""
    import bayesreg.io.guard  # noqa: F401
    import bayesreg.io.pair  # noqa: F401
    import bayesreg.seg.channels  # noqa: F401
    import bayesreg.seg.structure_layer  # noqa: F401
    import jd_curl.three_d.nifti  # noqa: F401
    import jd_curl.validation  # noqa: F401
    if unit == 'members':
        import pq_members  # noqa: F401   (puts experiments/hs on the path)
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


def allowlist_of(ctx):
    from bayesreg.io.guard import load_dev_allowlist, verify_allowlist_sources
    allowlist = load_dev_allowlist(Path(ctx.repo)/pins.ALLOWLIST_REL)
    try:
        sources = verify_allowlist_sources(allowlist, ctx.source_root)
    except Exception as exc:          # InputValidationError
        raise PQRefused('allowlist', f'the allowlist sources do not verify against {ctx.source_root}: {exc}') from None
    return allowlist, dict(path=pins.ALLOWLIST_REL, sha256=allowlist.sha256, sources=sources)


# ---- a breach before any output ----------------------------------------------------------------------------------------------------------------------
WINDOW_BREACH = '越界、尚无输出'


def window_breach(ctx, unit, spy, exc, *, opened_path, record_path, record, log):
    """Revision 2 (diff review minor 5): a breach in the spy's window before the unit's inner try (the plan, the gate, the code, the members' record, the device): the spy's record and a record marked
    越界、尚无输出 are written (project rules, hard rule 4: a failure is kept), at the unit's own names, or -- when a name is taken -- at pq_v1/window_breach_<unit>_<utc>.json; the unit stops with 8 and is not rerun
    (its own names now exist)."""
    opened = dict(spy.record(), status=WINDOW_BREACH, unwatched='the files the subprocesses (git, systemctl, nvidia-smi) open are not watched')
    document = dict(record, status=WINDOW_BREACH, stage='m6', passed=False, error=f'{type(exc).__name__}: {exc}', host=host_record(), utc=utc())
    try:
        io_.write_once(opened_path, opened)
        io_.write_once(record_path, document)
        where = [str(opened_path), str(record_path)]
    except FileExistsError:
        fallback = ctx.pq_root/f"window_breach_{unit}_{utc().replace(':', '')}.json"
        io_.write_once(fallback, dict(opened_files=opened, record=document))
        where = [str(fallback)]
    log(f'{unit}: {WINDOW_BREACH}: {exc}; recorded in {where}')
    return 8


# ---- the members unit ------------------------------------------------------------------------------------------------------------------------------
def members_unit(ctx, *, device, allowlist=None, files=None, gate=None, code=None, log=print):
    """The members unit (needs a new pq_v1/members/).  The tests pass allowlist, files, gate and code; the real run takes the pinned allowlist (bayesreg.seg.unet.pins), /mnt/d/data and the machine.
    Revision 1 (review minor 1): the spy is installed right after the imports, before the first file the unit opens (the plan), so that opened_files_members.json holds every file this process opened;
    what the subprocesses (git, systemctl, nvidia-smi) open is not watched."""
    import pq_members as members
    from bayesreg.io.pair import load_segmenter_registry
    tick = perf_counter()
    preload('members')
    root = ctx.pq_root/'members'
    problem = sealed_cache_problem('members')
    if problem:
        raise PQRefused('spy', problem)
    spy = fw.Spy(rules_for(ctx, 'members'))
    rows, error, environment, plan, dev, peak = {}, None, None, None, None, None
    try:
        with spy:
            plan = io_.check_plan(ctx.repo)
            if root.exists():
                raise PQRefused('once', f'{root} exists: the members unit runs once; nothing is overwritten or rerun')
            gate = machine_gate('members') if gate is None else gate
            code = code_state(ctx.repo, frozen_required=False) if code is None else code
            import torch
            dev = torch.device(device)
            if dev.type == 'cuda':
                from bayesreg.seg.unet.large import cap_gpu_memory
                free, _ = torch.cuda.mem_get_info(dev)
                if free < GPU_FREE_MIN_GIB*2**30:
                    raise PQRefused('machine', f'{free/2**30:.1f} GiB free on {dev}, {GPU_FREE_MIN_GIB} GiB required (GPU_FREE_MIN_GIB: the estimate of the peak and its margin)')
                cap_gpu_memory(dev, GPU_CAP_GIB)
            from bayesreg.seg.unet.train import environment_record
            environment = {k: v for k, v in environment_record(dev).items() if k != 'code'}
            if files is None:
                from bayesreg.seg.unet.data import OasisFiles
                files = OasisFiles(root=ctx.images)
            root.mkdir(parents=True)
            try:
                if allowlist is None:
                    from bayesreg.seg.unet.pins import load_pinned_allowlist
                    allowlist, _ = load_pinned_allowlist(Path(ctx.repo)/pins.ALLOWLIST_REL, source_root=ctx.source_root)
                registry = load_segmenter_registry(Path(ctx.repo)/ctx.anchor['registry'])
                rows = members.run_members(ctx, allowlist=allowlist, registry=registry, files=files, device=dev, log=log)
            except Exception as exc:                          # every failure is kept, never rerun (a Breach is an AssertionError)
                error = f'{type(exc).__name__}: {exc}'
            if dev.type == 'cuda':
                peak = dict(device_peak_allocated_gib=torch.cuda.max_memory_allocated(dev)/2**30, device_peak_reserved_gib=torch.cuda.max_memory_reserved(dev)/2**30)
    except fw.Breach as exc:                                  # revision 2 (diff review minor 5): a breach before members/ has any output
        return window_breach(ctx, 'members', spy, exc, opened_path=root/'opened_files_members.json', record_path=root/io_.MEMBERS_RECORD,
                             record=dict(schema=members.MEMBER_SCHEMA, plan=plan, gate=gate, code=code, subjects={}, m5_passed=False, spy=dict(m6=False)), log=log)
    record = spy.record()
    io_.write_once(root/'opened_files_members.json', dict(record, unwatched='the files the subprocesses (git, systemctl, nvidia-smi) open are not watched'))
    document = members.members_document(rows, code=code, environment=environment, spy_record=record, device=dev, seconds=perf_counter()-tick)
    document.update(plan=plan, gate=gate, error=error, host=host_record(), device_peak=peak)
    if error is not None:
        document['m5_passed'] = False
    members.write_members_record(ctx, document)
    log(f"members: m5 {document['m5_passed']}, spy m6 {record['m6']}, error {error}")
    return 0 if error is None and document['m5_passed'] and record['m6'] else 8


# ---- the PQ unit -----------------------------------------------------------------------------------------------------------------------------------
def out_unit(ctx, *, gate=None, code=None, log=print):
    """The PQ unit: m1-m4 for all pairs (pass 1), then PQ and the baselines (pass 2), opened_files.json, out_record.json and the sha256 list.  Revision 1 (review minor 1): the spy from the first file the
    unit opens (the plan, the members' record) to the last input; the subprocesses (git, systemctl) are not watched."""
    tick = perf_counter()
    preload('out')
    runs = ctx.pq_root/'runs'
    problem = sealed_cache_problem('out')
    if problem:
        raise PQRefused('spy', problem)
    spy = fw.Spy(rules_for(ctx, 'out'))
    checks, documents, error, stage, members_record, plan = {}, {}, None, None, {}, None
    try:
        with spy:
            plan = io_.check_plan(ctx.repo)
            if runs.exists() or (ctx.pq_root/'out_record.json').exists():
                raise PQRefused('once', f'{runs} exists: the PQ unit runs once; nothing is overwritten or rerun')
            gate = machine_gate('out') if gate is None else gate
            code = code_state(ctx.repo, frozen_required=True) if code is None else code
            if ctx.taper is None:
                ctx.taper = frozen_taper()
            members_record = io_.members_record(ctx)
            runs.mkdir(parents=True)
            try:
                allowlist, _ = allowlist_of(ctx)
                lookup = io_.load_lookup(ctx.repo, ctx.seg4_sha256)
                for short in ctx.shorts:
                    checks[short] = io_.check_pair(ctx, short, allowlist, lookup)
                    log(f'{short}: m1-m4 passed')
                for short in ctx.shorts:
                    documents[short] = io_.compute_pair(ctx, short, allowlist, lookup, members_record, runs/short)
                    log(f"{short}: S35^ {documents[short]['numbers']['S35_hat']}, S4^ {documents[short]['numbers']['S4_hat']} (the method's own numbers, no Dice table)")
            except PQRefused as exc:
                error, stage = str(exc), exc.stage
            except fw.Breach as exc:
                error, stage = str(exc), 'm6'
            except Exception as exc:                          # kept, not rerun
                error, stage = f'{type(exc).__name__}: {exc}', 'error'
    except fw.Breach as exc:                                  # revision 2 (diff review minor 5): a breach before runs/ has any output
        return window_breach(ctx, 'out', spy, exc, opened_path=ctx.pq_root/'opened_files.json', record_path=ctx.pq_root/'out_record.json',
                             record=dict(schema='bayesreg.pq.out_record.v1', plan=plan, contract=pins.CONTRACT, pairs=[], checks={}, controls=dict(m1=False, m2=False, m3=False, m4=False, m5=False, m6=False),
                                         gate=gate, code=code), log=log)
    record = spy.record()
    members_spy = io_.read_json(ctx.pq_root/'members'/'opened_files_members.json') if (ctx.pq_root/'members'/'opened_files_members.json').is_file() else None
    io_.write_once(ctx.pq_root/'opened_files.json', dict(members=members_spy, out=record, role='C22.Q1-3: every file the method side opened (the members unit and the PQ unit, each from its first file on), for the '
                                                                                                 'lead to read before the release; the subprocesses (git, systemctl, nvidia-smi) are not watched'))
    passed_pairs = error is None and sorted(documents) == sorted(ctx.shorts)
    controls = dict(m1=passed_pairs, m2=passed_pairs, m3=passed_pairs, m4=passed_pairs, m5=members_record.get('m5_passed') is True, m6=bool(record['m6'] and (members_spy or {}).get('m6', False)))
    if stage in ('m1', 'm2', 'm3', 'm4', 'm6'):
        controls[stage] = False
    out_record = dict(schema='bayesreg.pq.out_record.v1', plan=plan, contract=pins.CONTRACT, pairs=sorted(documents), checks=checks, controls=controls, passed=error is None and all(controls.values()),
                      error=error, stage=stage, code=code, gate=gate, host=host_record(), seconds=perf_counter()-tick, utc=utc())
    io_.write_once(ctx.pq_root/'out_record.json', out_record)
    if not out_record['passed']:
        log(f'out: stopped at {stage}: {error}; no sha256 list, the evaluation does not start (plan.md section 6, first row)')
        return 8
    listed = [str(p.relative_to(ctx.pq_root)) for p in sorted(ctx.pq_root.rglob('*')) if p.is_file() and not p.is_relative_to(ctx.pq_root/'eval')]
    manifest = io_.write_once(ctx.pq_root/io_.MANIFEST_FILE, text=io_.manifest_lines(ctx.pq_root, listed))
    sha = io_.sha_file(manifest)
    log(f'out: done; {len(listed)} files in {manifest}, sha256 {sha}')
    log('the lead: read opened_files.json; then commit, in ONE commit, the two files of the release (exactly these contents):')
    log(f'  runs/20261107-pq/outputs_manifest.sha256 :  {sha}  outputs_manifest.txt')
    log(f'  runs/20261107-pq/release.md              :  放行：{sha}')
    log('  (release.md: exactly that one line; 全角冒号"："（U+FF1A）, the sha256 in 64 lowercase hex, no other word; a BOM is tolerated)')
    return 0


# ---- the evaluation unit ---------------------------------------------------------------------------------------------------------------------------
class EvalStopped(BaseException):
    """SIGTERM (systemd stopping the unit) after the lock: raised in the main thread so that the evaluation record is written before the process ends (revision 1, review important 1)."""


def _stop_on_sigterm(signum, frame):
    raise EvalStopped(f'signal {signum}')


def eval_unit(ctx, *, gate=None, code=None, ev=None, read_truth=None, guard=None, log=print):
    """The evaluation unit (pq_eval's module docstring; the order of its steps 1-6).  The tests pass ev, read_truth and guard (the sealed lists and the training set of a synthetic world)."""
    import signal
    import threading
    import pq_eval as pe
    tick = perf_counter()
    plan = io_.check_plan(ctx.repo)
    lock = ctx.pq_root/pe.EVAL_LOCK_NAME
    root = ctx.pq_root/'eval'
    if lock.exists():
        raise PQRefused('once', f'{lock} exists: the truth was read for this reading already; it is read once (whatever happened), the lock is never removed')
    if root.exists():
        raise PQRefused('once', f'{root} exists: the evaluation ran (or stopped at e2); it is not rerun')
    gate = machine_gate('eval') if gate is None else gate
    code = code_state(ctx.repo, frozen_required=True) if code is None else code
    if ctx.taper is None:
        ctx.taper = frozen_taper()
    sha, release = pe.release_check(ctx)
    e2, _ = pe.e2_check(ctx, sha)
    base = dict(schema='bayesreg.pq.eval_record.v1', plan=plan, release=release, e2=e2, code=code, gate=gate)
    if not e2['passed']:
        root.mkdir(parents=True)
        io_.write_once(root/'eval_record.json', dict(base, e1=None, truth_read=False, complete=False, aborted=False, status='e2_failed', host=host_record(), utc=utc(),
                                                     note='e2 不过（plan.md 第 8 节；清单本身或清单内的文件与提交值不同）：不判，最后 1 次选择读数算用掉（第 6 节第 2 行，总负责人 11-07 对 e2 的裁定）；没有读真值'))
        log(f'eval: e2 failed ({e2}); no truth read; 不判, counted as used')
        return 8
    out_record = pe.out_record_check(ctx)
    if guard is None:
        from bayesreg.eval.truth import load_sealed_lists
        from bayesreg.io.guard import load_dev_allowlist
        from bayesreg.seg.unet.data import load_folds
        guard = dict(allowlist=load_dev_allowlist(Path(ctx.repo)/pins.ALLOWLIST_REL), sealed=load_sealed_lists(ctx.repo), training=load_folds(Path(ctx.repo)/pins.FOLDS_REL).training_subjects)
    subjects = pe.admitted_subjects(ctx, **guard)
    if ev is None:
        sys.path.insert(1, str(HERE.parent/'mc0'))
        import mc0_core
        ev = mc0_core.load_module('_pq_evaluate', 'experiments/d0/evaluate.py')
    read_truth = ev.truth_labels if read_truth is None else read_truth
    root.mkdir(parents=True)                                  # revision 1 (review minor 5): only after every check of steps 1-5
    main = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, _stop_on_sigterm) if main else None
    rows, error, stopped, locked = {}, None, None, False
    try:
        pe.take_lock(lock, dict(utc=utc(), pid=os.getpid(), plan_sha256=pins.PLAN_SHA256, release_sha256=sha, commit=code.get('commit')))
        locked = True
        for short in ctx.shorts:
            rows[short] = pe.evaluate_pair(ctx, short, ev=ev, read_truth=read_truth, out_dir=root/short)
            log(f"{short}: e1 {rows[short]['e1']}, errors on Omega_e {rows[short]['n_errors']['omega_e']}")
    except BaseException as exc:                              # revision 1 (review important 1): SystemExit, KeyboardInterrupt, EvalStopped as well; kept, the lock stays, not rerun
        error, stopped = f'{type(exc).__name__}: {exc}', exc
    finally:
        if main:
            signal.signal(signal.SIGTERM, previous)
        if locked:                                            # this process took the lock: the truth may have been read, the record is written whatever stopped it
            complete = stopped is None and sorted(rows) == sorted(ctx.shorts)
            record = dict(base, e1={s: r['e1'] for s, r in rows.items()}, pairs=sorted(rows), truth_read=True, complete=complete, aborted=stopped is not None,
                          status='complete' if complete else '真值已读、中止', subjects=subjects, out_record_sha256=io_.sha_file(ctx.pq_root/'out_record.json'),
                          eval_sha256={s: io_.sha_file(root/s/'eval.json') for s in rows}, error=error, host=host_record(), seconds=perf_counter()-tick, utc=utc())
            io_.write_once(root/'eval_record.json', record)
    log(f'eval: {len(rows)} pairs, error {error}')
    if stopped is not None and (not locked or not isinstance(stopped, Exception)):
        raise stopped                                         # a lock that appeared before ours: refused; SystemExit, KeyboardInterrupt, EvalStopped: the record is written, the stop goes on
    return 0 if error is None else 8


# ---- the reading -----------------------------------------------------------------------------------------------------------------------------------
def integrity_check(ctx, eval_record, pqs, evals):
    """Revision 1 (review minor 9): what the reading reads is what was listed and evaluated: pq_v1/outputs_manifest.txt has the committed sha256, every pq.json read has its listed sha256, every eval.json
    the sha256 the evaluation record wrote.  {'passed': bool, 'problems': [...]}."""
    import pq_eval as pe
    problems = []
    committed = Path(ctx.repo)/pe.RELEASE_SHA_REL
    found = sorted(set(pe.HEX.findall(committed.read_text(encoding='utf-8')))) if committed.is_file() else []
    listed = ctx.pq_root/io_.MANIFEST_FILE
    if len(found) != 1 or not listed.is_file() or io_.sha_file(listed) != found[0]:
        problems.append(f'{listed} is not the committed list ({found})')
        rows = {}
    else:
        rows = io_.parse_manifest(listed.read_text(encoding='utf-8'))
    for s in pqs:
        rel = f'runs/{s}/{io_.PQ_JSON}'
        if rows.get(rel) != io_.sha_file(ctx.pq_root/rel):
            problems.append(f'{rel} is not the listed file')
    written = (eval_record or {}).get('eval_sha256') or {}
    for s in evals:
        if written.get(s) != io_.sha_file(ctx.pq_root/'eval'/s/'eval.json'):
            problems.append(f'eval/{s}/eval.json is not the file the evaluation wrote')
    return dict(passed=not problems, problems=problems, rule='the list = the committed sha256; pq.json = the list; eval.json = eval_record.eval_sha256')


def truth_evidence(ctx):
    """Revision 2 (diff review minor 2): the three marks of a truth that was read -- the lock, eval_record.json's truth_read, any eval/<pair>/eval.json or e.npz -- and the state they make together:
    'not_read' (none of them), 'read' (the lock, and no mark against it: a record that is missing, unreadable or says truth_read true), 'contradiction' (any other combination, e.g. the lock deleted while
    the record says truth_read true or eval files are there, or the lock there while the record says truth_read false)."""
    import pq_eval as pe
    root = ctx.pq_root
    lock = (root/pe.EVAL_LOCK_NAME).exists()
    record, problem = io_.read_json_or_none(root/'eval'/'eval_record.json')
    record_truth = None if record is None else record.get('truth_read')
    files = sorted(f'eval/{s}/{n}' for s in ctx.shorts for n in ('eval.json', 'e.npz') if (root/'eval'/s/n).exists())
    marks = dict(lock=lock, eval_record_truth_read=record_truth, eval_record_unreadable=problem, eval_files=files[:6], n_eval_files=len(files))
    if not lock and record_truth is not True and not files and problem is None:
        state = 'not_read'
    elif lock and record_truth is not False:
        state = 'read'
    else:
        state = 'contradiction'
    return state, marks, record, problem


def decide_unit(ctx, *, log=print):
    """The reading (plan.md section 6).  Revision 1 (review important 1) and revision 2 (diff review minors 1, 2): the truth was read when the lock, the evaluation record or the evaluation files say so.
    - none of them: no reading is written (PQRefused; an e2 failure before the truth is 不判, counted as used, in eval/eval_record.json);
    - they contradict each other (the lock removed by hand, ...): refused, the contradiction named, never the first row;
    - the truth read: the reading is written once; a missing, incomplete or unreadable (truncated) evaluation record or eval.json is the second row of the table (不判, counted as used), never the first; the
      files that could not be read are named in the decision."""
    import pq_decide as decide
    root = ctx.pq_root
    analysis = root/'analysis'
    state, marks, eval_record, record_problem = truth_evidence(ctx)
    if state == 'not_read':
        what = (f"the evaluation stopped at e2 before the truth ({(eval_record or {}).get('status')}): 不判, counted as used (eval/eval_record.json)" if eval_record and (eval_record.get('e2') or {}).get('passed') is False
                else 'the evaluation has not read the truth')
        raise PQRefused('decide', f'no lock, no record of a truth read, no evaluation file: {what}; no reading is written to {analysis}')
    if state == 'contradiction':
        raise PQRefused('decide', f'the marks of a truth that was read contradict each other ({marks}): refused; the lead decides (never the first row: the truth may have been read)')
    if analysis.exists():
        raise PQRefused('once', f'{analysis} exists: the reading is written once')
    unreadable = [record_problem] if record_problem else []

    def load(path):
        document, problem = io_.read_json_or_none(path)
        if problem:
            unreadable.append(problem)
        return document
    out_record = load(root/'out_record.json')
    members = load(root/'members'/io_.MEMBERS_RECORD)
    evals = {s: d for s in ctx.shorts if (d := load(root/'eval'/s/'eval.json')) is not None}
    pqs = {s: d for s in ctx.shorts if (d := load(root/'runs'/s/io_.PQ_JSON)) is not None}
    released = bool(eval_record and eval_record.get('release')) or ((Path(ctx.repo)/'runs'/'20261107-pq'/'outputs_manifest.sha256').is_file() and (Path(ctx.repo)/'runs'/'20261107-pq'/'release.md').is_file())
    integrity = integrity_check(ctx, eval_record, pqs, evals)
    if unreadable:
        integrity = dict(integrity, passed=False, problems=list(integrity['problems'])+unreadable)
    document = decide.decide(ctx, evals=evals, pqs=pqs, out_record=out_record, members=members, eval_record=eval_record, released=released, truth_read=True, integrity=integrity)
    document.update(truth_evidence=marks, unreadable=unreadable)
    io_.write_once(analysis/'decision_pq.json', document)
    io_.write_once(analysis/'report_pq.md', text=decide.report_markdown(document))
    log(f"decide: {document['reading']['reading']}")
    return 0


# ---- the dry runs ----------------------------------------------------------------------------------------------------------------------------------
def npz_header(path):
    """{member: (dtype, shape)} from the .npy headers inside the archive; no array is read."""
    import zipfile
    from numpy.lib import format as npformat
    out = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                version = npformat.read_magic(stream)
                shape, fortran, dtype = npformat._read_array_header(stream, version)
            out[name[:-4] if name.endswith('.npy') else name] = (str(dtype), list(shape))
    return out


def dry_common(ctx, log):
    problems = []
    try:
        log(f'plan: {io_.check_plan(ctx.repo)}')
    except PQRefused as exc:
        problems.append(str(exc))
    for name, rel, want in (('registry', ctx.anchor['registry'], ctx.anchor['registry_sha256']), ('seg4 table', pins.SEG4_TABLE_REL, ctx.seg4_sha256)):
        have = io_.sha_file(Path(ctx.repo)/rel)
        log(f'{name}: {rel} sha256 {have} {"=" if have == want else "!="} pinned')
        if have != want:
            problems.append(f'{name} sha256')
    return problems


def dry_members(ctx, log):
    import pq_members as members
    problems = dry_common(ctx, log)
    from bayesreg.io.pair import load_segmenter_registry
    preload('members')
    problem = sealed_cache_problem('members --dry-run')
    if problem:
        log(problem)
        return problems+[problem]
    with fw.Spy(rules_for(ctx, 'members')) as spy:
        try:
            allowlist, record = allowlist_of(ctx)
            log(f"allowlist {record['sha256']}, sources verified against {ctx.source_root}: {sorted(record['sources'])}")
            registry = load_segmenter_registry(Path(ctx.repo)/ctx.anchor['registry'])
            available = members.member_files(Path(ctx.results)/pins.MODELS_ROOT_REL)
            log(f'members: the four pinned files hashed: {[str(p) for p in available.values()]}')
            for subject in ctx.subjects():
                plan, models, policy = members.plan_for(registry, subject, allowlist, available)
                d = ctx.pred_root/subject
                ok = io_.sha_file(d/'provenance.json') == ctx.v4[subject].provenance
                header = npz_header(d/'probs36.npz')
                if not ok or header.get('p') != ('float16', [36, 192, 224, 160]):
                    problems.append(f'{subject}: registered provenance {ok}, probs36 header {header}')
                log(f'{subject}: admitted, {policy}, members {[sha[:8] for _, sha, _ in plan]}, provenance sha256 ok {ok}, probs36 {header.get("p")}, image file present {(Path(ctx.images)/subject/"aligned_norm.nii.gz").is_file()}')
        except (PQRefused, fw.Breach, Exception) as exc:
            problems.append(f'{type(exc).__name__}: {exc}')
    record = spy.record()
    log(f"spy: {record['n_files']} files opened, {record['n_in_scope']} in the watched scope, breaches {record['breaches']}; no image opened: {not any(f['path'].endswith('aligned_norm.nii.gz') for f in record['files'])}")
    log(f"output root {ctx.pq_root/'members'} exists: {(ctx.pq_root/'members').exists()} (must be False)")
    if (ctx.pq_root/'members').exists():
        problems.append('pq_v1/members exists')
    return problems


def dry_out(ctx, log):
    problems = dry_common(ctx, log)
    try:
        import mc0_core
        mc0_core.frozen_api()
        log(f'frozen tree: jd_curl from {mc0_core.FROZEN}, preopt.py sha256 pinned: ok')
    except SystemExit as exc:
        problems.append(f'frozen tree: {exc}')
    preload('out')
    from bayesreg.io.pair import _keep_whitelisted_keys
    problem = sealed_cache_problem('out --dry-run')
    if problem:
        log(problem)
        return problems+[problem]
    with fw.Spy(rules_for(ctx, 'out')) as spy:
        try:
            allowlist, record = allowlist_of(ctx)
            log(f"allowlist {record['sha256']}, sources verified: {sorted(record['sources'])}")
            for short in ctx.shorts:
                row = ctx.p3[short]
                t88, rep = ctx.t88_dir(short), ctx.repair_dir(short)
                have = dict(run_json=io_.sha_file(t88/'run.json'), repair_json=io_.sha_file(rep/'repair.json'), repair_npz=io_.sha_file(rep/'repair.npz'), field=io_.sha_file(t88/'field_0600.npz'))
                want = dict(run_json=ctx.run_json[short], repair_json=row.repair_json, repair_npz=row.repair_npz, field=row.field_npz)
                repair, run = io_.read_json(rep/'repair.json'), io_.read_json(t88/'run.json')
                cert = repair.get('certificate') or {}
                bad = [k for k in have if have[k] != want[k]]
                bad += [] if cert.get('state_hash') == row.state_hash and not io_.certificate_problems(cert) else ['certificate']
                bad += [] if (repair.get('source') or {}).get('field_sha256') == row.field_npz and (repair.get('attempt') or {}).get('sha256') == row.repair_npz else ['repair.json source/attempt']
                bad += [] if run.get('status') == 'done' and run.get('oracle') is False and (run.get('field') or {}).get('sha256') == row.field_npz and sorted(run.get('labels_sha256') or {}) == ['fixed', 'moving'] else ['run.json']
                header = npz_header(t88/'field_0600.npz')
                stage = ctx.stage/pins.pair_name(short)
                manifest = json.loads((stage/'manifest.json').read_text(encoding='utf-8'), object_pairs_hook=_keep_whitelisted_keys)     # load_pair's whitelist: no label metric
                if header.get('u') != ('float64', [192, 224, 160, 3]) or manifest.get('spacing_xyz') != [1.0, 1.0, 1.0] or not (stage/'pair.npz').is_file():
                    bad.append(f"field header {header}, spacing {manifest.get('spacing_xyz')}, pair.npz present {(stage/'pair.npz').is_file()}")
                if bad:
                    problems.append(f'{short}: {bad}')
                log(f'{short}: run.json, repair.json, repair.npz, field_0600.npz sha256 = the plan, the certificate and the chain of hashes in the records: {"ok" if not bad else bad}')
            for subject in ctx.subjects():
                d = ctx.pred_root/subject
                ok = io_.sha_file(d/'provenance.json') == ctx.v4[subject].provenance
                prov = io_.read_json(d/'provenance.json')
                outputs = prov.get('output_sha256') or {}
                anchors = all(prov.get(k) == ctx.anchor[k] for k in ('segmenter_id', 'policy', 'registry_sha256'))
                same = outputs.get('probs36.npz') == ctx.v4[subject].probs36 and outputs.get('structure_groups.npz') == ctx.v4[subject].structure_groups
                if not (ok and anchors and same and prov.get('test_labels_seen') is False):
                    problems.append(f'{subject}: provenance {ok}, anchors {anchors}, outputs {same}')
                log(f'{subject}: provenance.json sha256 = the plan {ok}; anchors {anchors}; output_sha256 = the plan {same}; probs36 header {npz_header(d/"probs36.npz").get("p")}')
        except (PQRefused, fw.Breach, Exception) as exc:
            problems.append(f'{type(exc).__name__}: {exc}')
    record = spy.record()
    log(f"spy: {record['n_files']} files opened, {record['n_in_scope']} in the watched scope, breaches {record['breaches']}; pair.npz members read {record['pair_members']} (none in the dry run)")
    if record['breaches'] or record['pair_members']:
        problems.append('spy')
    members = ctx.pq_root/'members'/io_.MEMBERS_RECORD
    log(f"members record {members} present: {members.is_file()} (needed before the real run; the members unit comes first)")
    log(f"output {ctx.pq_root/'runs'} exists: {(ctx.pq_root/'runs').exists()} (must be False)")
    if (ctx.pq_root/'runs').exists():
        problems.append('pq_v1/runs exists')
    return problems


def dry_eval(ctx, log):
    import pq_eval as pe
    problems = dry_common(ctx, log)
    lock = ctx.pq_root/pe.EVAL_LOCK_NAME
    log(f'lock {lock} exists: {lock.exists()} (must be False)')
    if lock.exists():
        problems.append('the lock exists')
    for rel in (pe.RELEASE_SHA_REL, pe.RELEASE_NOTE_REL):
        log(f'release file {rel} present: {(Path(ctx.repo)/rel).is_file()} (expected False until the lead releases)')
    try:
        pe.release_check(ctx)
        log('release: found')
    except PQRefused as exc:
        log(f'release: not yet ({exc}) -- the real run refuses here, before any truth')
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.io.guard import load_dev_allowlist
    from bayesreg.seg.unet.data import load_folds
    try:
        subjects = pe.admitted_subjects(ctx, allowlist=load_dev_allowlist(Path(ctx.repo)/pins.ALLOWLIST_REL), sealed=load_sealed_lists(ctx.repo),
                                        training=load_folds(Path(ctx.repo)/pins.FOLDS_REL).training_subjects)
        log(f'guard (eval_seg.admit_subjects): {len(subjects)} subjects admitted, none sealed, none in training or never_for_training')
    except Exception as exc:
        problems.append(f'guard: {type(exc).__name__}: {exc}')
    for short in ctx.shorts:
        present = (ctx.repair_dir(short)/'repair_metrics.json').is_file() and (ctx.stage/pins.pair_name(short)/'pair.npz').is_file()
        if not present:
            problems.append(f'{short}: repair_metrics.json or pair.npz missing')
    log('repair_metrics.json and the original pair.npz present for the 19 pairs (not opened): ' + ('ok' if not any('missing' in p for p in problems) else 'NO'))
    try:
        import mc0_core
        mc0_core.frozen_api()
        log('frozen tree: ok')
    except SystemExit as exc:
        problems.append(f'frozen tree: {exc}')
    return problems


def dry_decide(ctx, log):
    """The reading code on a synthetic AUSE table of 19 pairs (no file of the run is read): one table that passes, one that fails on X_lex."""
    problems = dry_common(ctx, log)
    rng = np.random.default_rng(0)
    q = rng.uniform(.01, .02, len(ctx.shorts))
    table = {k: list(q*(1.2+.05*rng.standard_normal(len(ctx.shorts)))) for k in stats.JUDGED_BASELINES}
    table['q'] = list(q)
    b = stats.b_class(table, np.arange(len(ctx.shorts)))
    controls = dict(m1=True, m2=True, m3=True, m4=True, m5=True, m6=True)
    r = stats.reading(method_controls=controls, released=True, eval_controls=dict(e1='bitwise', e2=True, complete=True, integrity=True), b=b, truth_read=True)
    log(f"synthetic table (every baseline about 20 % worse): {r['reading']}; LB {[round(b['baselines'][k]['LB'], 5) for k in stats.JUDGED_BASELINES]}")
    table['X_lex'] = list(q*1.01)
    b = stats.b_class(table, np.arange(len(ctx.shorts)))
    r2 = stats.reading(method_controls=controls, released=True, eval_controls=dict(e1='bitwise', e2=True, complete=True, integrity=True), b=b, truth_read=True)
    log(f"synthetic table with X_lex 1 % worse: {r2['reading']}; failed {r2.get('failed')}")
    if r['row'] != 'b_pass' or r2['row'] != 'b_fail':
        problems.append('the synthetic readings')
    log(f"analysis {ctx.pq_root/'analysis'} exists: {(ctx.pq_root/'analysis').exists()} (must be False)")
    return problems


DRY = dict(members=dry_members, out=dry_out, eval=dry_eval, decide=dry_decide)


def dry_run(unit, ctx=None, log=print):
    ctx = production_context() if ctx is None else ctx
    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain'], capture_output=True, text=True).stdout.splitlines()
    log(f'PQ {unit} --dry-run, {utc()}; code {REPO} at {head or "?"} ({"clean" if not dirty else f"{len(dirty)} changed paths"}); no image, no truth, no GPU, nothing written')
    problems = DRY[unit](ctx, log)
    if unit in UNITS:
        log('start string (the lead): ' + ' '.join(start_string(unit)))
    log(f'estimate: {ESTIMATES[unit]}')
    log(f'problems: {problems or "none"}')
    return 0 if not problems else 8


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('unit', choices=('members', 'out', 'eval', 'decide'))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--device', default='cuda')
    o = ap.parse_args(argv)
    if o.dry_run:
        return dry_run(o.unit)
    try:
        if o.unit in UNITS and not in_own_unit(o.unit):
            raise PQRefused('unit', f'run it in its unit {UNITS[o.unit]} (the start string of --dry-run)')
        ctx = production_context()
        if o.unit == 'members':
            return members_unit(ctx, device=o.device)
        if o.unit == 'out':
            return out_unit(ctx)
        if o.unit == 'eval':
            return eval_unit(ctx)
        return decide_unit(ctx)
    except PQRefused as exc:
        print(f'refused: {exc}', file=sys.stderr)
        return 8


if __name__ == '__main__':
    sys.exit(main())
