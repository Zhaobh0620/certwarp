"""PQH: the development replay gate (D119 (a); contract C22.H8 "开发集重放门", C22.H7-1 last item; plan runs/20261109-pqh/replay_gate_plan.md).

What it checks: the PQ section of the hold-out evaluation entry (bayesreg/eval/holdout.py, the EV package: the per-pair acceptance numbers of C22.H7-1 and F4's class B) reproduces the development PQ reading
(pq_v1) bit for bit on the 19 development pairs: every pair's 10 scores x 3 masks (AUSE, AUROC, the largest tie share) as pq_v1/eval/<pair>/eval.json, its error maps e, e^(4) as pq_v1/eval/<pair>/e.npz,
and the 8 baselines' Delta, LB, R as pq_v1/analysis/decision_pq.json, with the development block_interval in place of the hold-out resampling (plan.md 5.2).  It is not a selection reading: it chooses
nothing and changes no rule; it reads the development truth (project rules, hard rules 6, 7 allow it for the evaluation of development pairs) and no hold-out file.

Inputs per pair (C22.H8): the truth through the evaluation entry's own reader in its development mode (EV_FUNCTIONS['truth']: the C23.12-10 (i) way); d = disp_voxels(v, spacing), v P3's certified final
state rebuilt by the development pq_io.bind_field (P3 has no disp.npz), its sha256 the one pq_v1's pq.json records; the scores from pq_v1/runs/<pair>/{pq.npz, baselines.npz} (their sha256 as listed in
pq_v1/outputs_manifest.txt, released 86b1e298...).  The reference files' sha256 are pinned below (REFERENCE_SHA256), so a changed file refuses the gate before any truth is read.

The interface it asks of EV (pqh 实现说明 §6; the names below are this package's proposal, to be aligned): in bayesreg.eval.holdout
  pq_dev_truth(subject) -> {'seg35': int array, 'seg4': int array, 'files': {name: sha256}}        the entry's one reader on a development subject (C23.12-10 (i))
  pq_pair_acceptance(*, truth_fixed, truth_moving, d, pq, baselines, spacing_xyz) -> {'e': bool/uint8 array, 'e4': ..., 'scores': {mask: {score: {ause, auroc, largest_tie_share, ...}}}, ...}
                                                                                                     C22.H7-1 per pair, arrays only, no file read
  pq_b_class(ause_table, *, interval) -> {'baselines': {k: {'delta': [...], 'LB': float, 'R': float, ...}}, 'passed': bool, ...}
                                                                                                     F4's class B with the resampling passed in (interval(delta) -> {'lo': ...})
Run once (REPLAY_LOCK, O_EXCL, before the first truth); the plan's sha256 pinned (REPLAY_PLAN_SHA256; None refuses); a new output directory (REPLAY_ROOT); the code tree clean at its HEAD (review
I4; pqh_units.checkout_code_identity: bayesreg.io.code_identity.gate_identity_problems from HEAD to HEAD), its commit, the sha256 of the PQ section's files (REPLAY_CODE_FILES) and the environment (REPLAY_ENVIRONMENT_KEYS) in the result and the lock line.  The reading: REPLAY_READING_PASS
('逐位相同 19/19'), REPLAY_READING_FAIL or REPLAY_READING_UNDECIDED (an EV function failed or missing after the lock).  The lead then fills pqh_pins.PQ_REPLAY_DECISION in a reviewed commit; the entry
checks it at its step (0) through replay_decision_problem() below: the decision's keys (with the environment, I4), the result file (at its one path, result_path(): B4 m2), its code commit and
environment, the environment now, and
rule two from the decision's commit (bayesreg.io.code_identity.gate_identity_problems: the code changes only by the four decision constants' assignments, judged on the AST with each constant in
its own file -- DECISION_HOMES: PQ_REPLAY_DECISION in experiments/pq/pqh_pins.py -- and by the allowlist's one allowance; contract revision 9 draft C24.3-5, -6).  The constant is filled as a
plain literal (C24.3-6 (a)): a dict literal, not MappingProxyType(...).
So any later change of code makes the gate's reading stale: the gate runs after the last code commit before the unsealing.  Not passed, or not pinned: the hold-out is not unsealed (C22.H8,
revision 7a I4); no change of code and rerun.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from types import MappingProxyType

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE.parent/'holdout') not in sys.path:
    sys.path.append(str(HERE.parent/'holdout'))

import pq_core as core  # noqa: E402
import pq_io as io_  # noqa: E402
import pq_pins as pins  # noqa: E402
import pq_stats as stats  # noqa: E402
import pqh_pins as hp  # noqa: E402

REPLAY_UNIT = hp.REPLAY_UNIT                             # when the gate runs on its own (EV may run it in the run of C23.12-10's gate instead; C22.H8); run_pqh.py launch replay-gate (m6)
EV_MODULE = 'bayesreg.eval.holdout'
EV_FUNCTIONS = MappingProxyType(dict(truth='pq_dev_truth', pair='pq_pair_acceptance', b_class='pq_b_class'))     # 〔对齐〕 EV
MASKS = ('omega_e', 'band', 'whole')
SCORES = ('q',)+stats.JUDGED_BASELINES+stats.DESCRIBED_SCORES
SCORE_FIELDS = ('ause', 'auroc', 'largest_tie_share')
CLASS_FIELDS = ('delta', 'LB', 'R')
RESULT_FILE = 'replay_result.json'
RESULT_DIR = 'result'                                    # the run's output directory under REPLAY_ROOT (main); the decision's result_json must be REPLAY_ROOT/RESULT_DIR/RESULT_FILE (B4 m2)
SCHEMA = 'bayesreg.pqh.replay.v1'
EV_DIR = 'ev' + 'al'                                     # pq_v1's evaluation directory (the name is spelled once here)
# The sha256 of the development reading's files the gate compares with (computed by the implementer on 2026-11-09 from the files, not opened; plan §2): pq_v1/outputs_manifest.txt is the released list
# (dd89633); the 19 pq.npz and baselines.npz are checked against it.
REFERENCE_SHA256 = MappingProxyType({
    'outputs_manifest.txt': '86b1e298516bf31b2b79a4288f3aa2942a19ceb4eee4266d8c7bf37a12337b06',
    'analysis/decision_pq.json': '92f68f7e43dd4cc2f21bb56275fdaab12d5681c7de0a6d1564941fafa5ff7d5b',
    'eval/eval_record.json': '8fbdcc01845eb06643879b0f02abed65a6539a53371b1bef8e528a544492faa2',
    'eval/0438-0439/eval.json': 'e22fb1545e66995bd3d81c201d6c57b851ce8af31b5974528e15a03e5792ddb5',
    'eval/0439-0440/eval.json': '4b40ef11859ad37354d0042776acccf7217ddd9b0a0a0b5b34b1fc6f59e7a714',
    'eval/0440-0441/eval.json': '3bdf581a97f32a281df9b5ad86916ba832f20d7c79495afc324ea9da5cf61492',
    'eval/0441-0442/eval.json': 'f6c12e6b6e8ec73daee0599a3f64a319cba0dc919be8596d0de8a8dd01e88f4f',
    'eval/0442-0443/eval.json': 'bbb19e3650d54295e955dfb3e26bd0d67e4e1aa124a8eddce9ad6932c77ea954',
    'eval/0443-0444/eval.json': 'e510dfeaa7786b78df2769dc9912ee1adfa8dee79d8bff5b1cdd18f80e15473f',
    'eval/0444-0445/eval.json': '4e027c46d7cb5bfa27655a2163bd1cf4b7aec3c3478b02e6b1250f0b4f4bf259',
    'eval/0445-0446/eval.json': '0ee847bca8ea5d64853e7d51bb9f13eef6ce02c8a58460022ee6ef2c846cfe66',
    'eval/0446-0447/eval.json': 'c7993f56abe72d308addd48576330d5e0b1f2ecf26e58b7f2684e2691e18c2e5',
    'eval/0447-0448/eval.json': 'dea40f32e0008581c999d9bb3e80d7af8735d36fff92142c315c18ab78d35122',
    'eval/0448-0449/eval.json': 'de719477838be775d745b724dac52f0d7f0b8ccd7b2d8fd1420b4879b8c02583',
    'eval/0449-0450/eval.json': 'cf8c274de265a853529ba96f0182f212d73a413a645dd4c24189b5d504aaeb0e',
    'eval/0450-0451/eval.json': '573a732dfc2e77bb12da51a54c1e8f73fd2d52ed9aaca1d13a2218523ddf61ac',
    'eval/0451-0452/eval.json': '67ae33c8ed2d613743548ce9d3543c494d6fdde28bc8fae1aae43c31cbc62888',
    'eval/0452-0453/eval.json': '0c1acb21bd4f386fbe9e6f16232bc0f41863381d79be271fb37a756800df770f',
    'eval/0453-0454/eval.json': 'fbb54d2f505c6c06a19e3d84ea5cab1ba7ddef824b8fab17204f74cf8dee6ffc',
    'eval/0454-0455/eval.json': 'ee9fbbd202326f2cc40e027f172298c7ad633f3c506257c7040e21232bbfe439',
    'eval/0455-0456/eval.json': '0bc7941032cf4e60c87e817d18049522b3fc32af1e6b15437d9fa9b064b5bfd0',
    'eval/0456-0457/eval.json': '2b822aa332ba56baab57d666b4f826f32dbbeeeace34e53980df147320f08acb',
    'eval/0438-0439/e.npz': '23361a9b503238ac36ee4a5c5264c4eee9438736936bc0c1e28a84943285cd16',
    'eval/0439-0440/e.npz': '48cddd0136d38b3dcedfa9203d2676b487a97c123c19ea7c64a350b8d8b5250b',
    'eval/0440-0441/e.npz': 'f49a51a0cb10c29ef86b10b674d80728323a42d8d95ce1c7a27b70a0e2dcb909',
    'eval/0441-0442/e.npz': '5c6046513ce0a8864e097ae817e94d60b5f359dfad4b4380852f1d22e37c6084',
    'eval/0442-0443/e.npz': '7d5610542f37e18bc6be25311c26d2642875d60332cfa1eeee6dea2f2f6f6bce',
    'eval/0443-0444/e.npz': '36fd973ff7780358f72e5e95cfdd51b1797db64b5c885f9e4310a2368c09d47e',
    'eval/0444-0445/e.npz': 'd23177c433cec613fe9d3f924cc23eeb5a900f778a6c593130c36076a7ee021c',
    'eval/0445-0446/e.npz': '4dfcfc7f3e309f3815b7bce316ef429cb6e51cd1476d4aca1b6e1e6255156fb3',
    'eval/0446-0447/e.npz': '00bd0ba1eb1c9493815212f69ae92d62180ba144bc45bf267ce3181df0ab54be',
    'eval/0447-0448/e.npz': '167e4a6de6b1caaa87a57fb5deaa398e10c0e819b267f0ae6ba36f2937256e70',
    'eval/0448-0449/e.npz': '87cc753158eb9e7e31097b1c5219dd9f106f14872cd457f04d7f31a593d0b692',
    'eval/0449-0450/e.npz': 'd797f03caf8fb8bc60a84e26987aec5a925ddd6329b3956f51ea44ad8c326576',
    'eval/0450-0451/e.npz': '093086beae32ec92181a79385bc5b7da66ed2b2b899fbcb88845ece04bcb7151',
    'eval/0451-0452/e.npz': '140c7b28e30639c9e2925b3eee536e645a367d96ec09643035078dcb81ef3dac',
    'eval/0452-0453/e.npz': 'a5c729b866c2de97073bc7d8ce27439100ce38077be36863c003be7e9f6a9f35',
    'eval/0453-0454/e.npz': '9174ad6d0425eafb011210cf1c6c91b997bf746721574f18ca2a5ec5168a6448',
    'eval/0454-0455/e.npz': '7d54db1769f0e77135a494c788feba0a08ce6bdbc8b9704a1a9792b2b6ba7164',
    'eval/0455-0456/e.npz': '2acd739d29983f2f6d16eec9003f1db16f8f58bc2af7f5921837a3339293a308',
    'eval/0456-0457/e.npz': 'b04aa212f361b40e31218496a2634c32876c1310b8c34c846f2b2b260a38014e',
})


class ReplayRefused(Exception):
    """Refused before the lock; nothing of the truth is read."""

    def __init__(self, stage, message):
        super().__init__(f'{stage}: {message}')
        self.stage = stage


def sha_file(path):
    return io_.sha_file(path)


def reference_rel(kind, short=None):
    return dict(manifest='outputs_manifest.txt', decision='analysis/decision_pq.json', record=f'{EV_DIR}/{EV_DIR}_record.json',
                eval_json=f'{EV_DIR}/{short}/{EV_DIR}.json', e_npz=f'{EV_DIR}/{short}/e.npz')[kind]


def result_path():
    """The one place of the gate's result: <REPLAY_ROOT>/result/replay_result.json, REPLAY_ROOT read at call time (the tests move it).  main writes there; replay_decision_problem
    requires the decision's result_json to be this path (B4 review m2, the lead's ruling 2 of 11-14: as the truth-reader gate checks its own, bayesreg/eval/holdout.py)."""
    return Path(hp.REPLAY_ROOT)/RESULT_DIR/RESULT_FILE


def check_plan(repo, pinned=None):
    pinned = hp.REPLAY_PLAN_SHA256 if pinned is None else pinned
    path = Path(repo)/hp.REPLAY_PLAN_REL
    if pinned is None:
        raise ReplayRefused('plan', 'REPLAY_PLAN_SHA256 is None: the plan is not pinned (the lead pins it after the plan\'s review)')
    digest = sha_file(path) if path.is_file() else None
    if digest != pinned:
        raise ReplayRefused('plan', f'{path}: sha256 {digest} is not the pinned {pinned}')
    return dict(path=hp.REPLAY_PLAN_REL, sha256=digest)


def ev_functions(module=None):
    """EV's three functions (EV_FUNCTIONS) from bayesreg.eval.holdout, or ReplayRefused('ev') naming the missing ones."""
    try:
        module = importlib.import_module(EV_MODULE) if module is None else module
    except Exception as exc:                        # noqa: BLE001
        raise ReplayRefused('ev', f'{EV_MODULE} cannot be imported ({type(exc).__name__}: {exc})') from None
    missing = [name for name in EV_FUNCTIONS.values() if not callable(getattr(module, name, None))]
    if missing:
        raise ReplayRefused('ev', f'{EV_MODULE} has no {missing}: the PQ section of the evaluation entry is not written yet (EV package)')
    return {k: getattr(module, v) for k, v in EV_FUNCTIONS.items()}


def check_references(pq_root, shorts, reference=None):
    """Every reference file's sha256 as pinned; pq.npz and baselines.npz as listed in the released list.  ReplayRefused('reference') otherwise."""
    reference = REFERENCE_SHA256 if reference is None else reference
    pq_root = Path(pq_root)
    problems = []
    rels = [reference_rel('manifest'), reference_rel('decision'), reference_rel('record')]
    rels += [reference_rel(k, s) for s in shorts for k in ('eval_json', 'e_npz')]
    for rel in rels:
        have = sha_file(pq_root/rel) if (pq_root/rel).is_file() else None
        if have != reference.get(rel):
            problems.append(f'{rel}: {have} is not the pinned {reference.get(rel)}')
    listed = io_.parse_manifest((pq_root/reference_rel('manifest')).read_text(encoding='utf-8')) if not problems else {}
    for s in shorts:
        for name in (io_.PQ_NPZ, io_.BASELINES_FILE, io_.PQ_JSON):
            rel = f'runs/{s}/{name}'
            if not problems and listed.get(rel) != sha_file(pq_root/rel):
                problems.append(f'{rel} is not the listed file')
    if problems:
        raise ReplayRefused('reference', str(problems[:10]))
    return dict(n_files=len(rels)+3*len(shorts), rule='pinned sha256 (REFERENCE_SHA256); the runs/ files as listed in the released outputs_manifest.txt')


def same(a, b):
    """Bit for bit for the JSON numbers (both None, or the same float: JSON keeps Python floats exactly)."""
    if a is None or b is None:
        return a is None and b is None
    return float(a) == float(b) or (np.isnan(float(a)) and np.isnan(float(b)))


def compare_pair(mine, reference, e_ref, counts=None):
    """Rows of one pair: every (mask, score, field) and the two error maps.  counts (m6): {'reference_nonnull', 'compared_nonnull'} -- the reference's non-null items and those compared equal with a
    non-null value of mine; the reading requires the two to be equal (two nulls are the same, but they are not counted as compared)."""
    rows = []
    counts = {} if counts is None else counts
    counts.setdefault('reference_nonnull', 0)
    counts.setdefault('compared_nonnull', 0)
    for mask in MASKS:
        for score in SCORES:
            m = ((mine.get('scores') or {}).get(mask) or {}).get(score) or {}
            r = ((reference.get('scores') or {}).get(mask) or {}).get(score) or {}
            for field in SCORE_FIELDS:
                ok = same(m.get(field), r.get(field))
                if r.get(field) is not None:
                    counts['reference_nonnull'] += 1
                    counts['compared_nonnull'] += int(ok and m.get(field) is not None)
                if not ok:
                    rows.append(dict(item=f'{mask}.{score}.{field}', mine=m.get(field), reference=r.get(field)))
    for key in ('e', 'e4'):
        a = np.asarray(mine.get(key)).astype(np.uint8)
        b = np.asarray(e_ref[key])
        if a.shape != b.shape or a.tobytes() != b.tobytes():
            rows.append(dict(item=key, differ=int((a != b).sum()) if a.shape == b.shape else f'shape {a.shape} vs {b.shape}'))
    return rows


def pair_equal(differ, counts):
    """m6: a pair reads 逐位相同 when nothing differs, the reference has non-null items at all, and every one of them was compared with a non-null value of mine (no empty comparison)."""
    return not differ and counts.get('reference_nonnull', 0) > 0 and counts.get('compared_nonnull') == counts.get('reference_nonnull')


def compare_class(mine, reference):
    rows = []
    for k in stats.JUDGED_BASELINES:
        m = ((mine or {}).get('baselines') or {}).get(k) or {}
        r = ((reference or {}).get('baselines') or {}).get(k) or {}
        for field in CLASS_FIELDS:
            a, b = m.get(field), r.get(field)
            ok = (isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))) if field == 'delta' else same(a, b)
            if not ok:
                rows.append(dict(item=f'{k}.{field}', mine=a, reference=b))
    return rows


def environment():
    """I4: the environment the gate runs in (C23.12-10's constant: numpy and nibabel versions; python and scipy too)."""
    import nibabel
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, nibabel=nibabel.__version__)


def code_files(repo):
    """I4: the sha256 of the PQ section's files (REPLAY_CODE_FILES) in the code tree; None for a file that is not there."""
    return {rel: (sha_file(Path(repo)/rel) if (Path(repo)/rel).is_file() else None) for rel in hp.REPLAY_CODE_FILES}


def replay(ctx, *, ev, out_root, lock, guard=None, reference=None, plan=None, log=print, git=None):
    """The gate's run on the development context (pq_io.Context: P3's pins and roots).  ev: EV's three functions (ev_functions()); guard(): the development admission of the pairs (dev subjects only);
    git: subprocess.run for the code identity (pqh_units.checkout_code_identity: the tree clean at its HEAD, I4).  Writes <out_root>/replay_result.json once; returns its document."""
    import holdout_common as hc
    import pqh_inputs as hin
    import pqh_units as hu
    tick = perf_counter()
    shorts = list(ctx.shorts)
    pq_root = ctx.pq_root
    plan = check_plan(ctx.repo) if plan is None else plan
    code, problems = hu.checkout_code_identity(ctx.repo, git)
    if problems:
        raise ReplayRefused('code', f'{ctx.repo}: {problems} (I4: the gate runs on a clean tree, its commit recorded)')
    code = dict(commit=code['commit'], head=code['head'], clean=code['clean'], repo=str(ctx.repo), files=code_files(ctx.repo))     # NM1: commit = head = the gate's HEAD
    env = environment()
    if Path(lock).exists():
        raise ReplayRefused('once', f'{lock} exists: the gate ran; it runs once (C22.H8)')
    if Path(out_root).exists():
        raise ReplayRefused('once', f'{out_root} exists')
    checked = check_references(pq_root, shorts, reference)
    admission = guard() if guard is not None else [hc.admit_dev_pair(*pins.subjects_of(s))[0] for s in shorts]
    from bayesreg.io.pair import _read_manifest
    hin.guard_write(lock)
    hin.guard_write(out_root)
    hc.take_lock(lock, hc.lock_line(tool=hp.TOOL, unit='replay_gate', plan_sha256=plan['sha256'], reference=checked, code_commit=code['commit'], clean=code['clean'], code_files=code['files'],
                                    environment=env))
    Path(out_root).mkdir(parents=True)
    rows, ause, reading, error = {}, {}, None, None
    try:
        decision = json.loads((pq_root/reference_rel('decision')).read_text(encoding='utf-8'))
        for i, short in enumerate(shorts):
            grid, _, _ = _read_manifest(ctx.stage/pins.pair_name(short)/'manifest.json')
            v, field_record = io_.bind_field(ctx, short, grid)
            spacing = tuple(float(s) for s in grid.spacing)
            d = core.disp_voxels(v, spacing)
            document = json.loads((pq_root/'runs'/short/io_.PQ_JSON).read_text(encoding='utf-8'))
            if io_.sha_array(d) != document['sampling']['d_sha256']:
                raise RuntimeError(f'{short}: d is not the one of q')
            with np.load(pq_root/'runs'/short/io_.PQ_NPZ, allow_pickle=False) as z:
                pq = {k: np.array(z[k]) for k in ('q', 'q4')}
            with np.load(pq_root/'runs'/short/io_.BASELINES_FILE, allow_pickle=False) as z:
                base = {k: np.array(z[k]) for k in core.BASELINE_KEYS}
            fixed, moving = pins.subjects_of(short)
            mine = ev['pair'](truth_fixed=ev['truth'](fixed), truth_moving=ev['truth'](moving), d=d, pq=pq, baselines=base, spacing_xyz=spacing)
            reference_eval = json.loads((pq_root/reference_rel('eval_json', short)).read_text(encoding='utf-8'))
            with np.load(pq_root/reference_rel('e_npz', short), allow_pickle=False) as z:
                e_ref = {k: np.array(z[k]) for k in ('e', 'e4')}
            counts = {}
            differ = compare_pair(mine, reference_eval, e_ref, counts)
            rows[short] = dict(equal=pair_equal(differ, counts), differences=differ[:50], n_differences=len(differ),
                               state_hash=field_record['state_hash'], d_sha256=io_.sha_array(d), **counts)
            for score in SCORES:
                ause.setdefault(score, []).append(((mine.get('scores') or {}).get('omega_e') or {}).get(score, {}).get('ause'))
            log(f"{short}: {'逐位相同' if not differ else f'{len(differ)} differences'}")
        index = np.arange(len(shorts))
        mine_class = ev['b_class'](ause, interval=lambda delta: stats.block_interval(delta, index))
        class_rows = compare_class(mine_class, decision.get('b_class'))
        reading = hp.REPLAY_READING_PASS.replace('19', str(len(shorts))) if all(r['equal'] for r in rows.values()) and len(rows) == len(shorts) and not class_rows else hp.REPLAY_READING_FAIL
    except Exception as exc:                        # noqa: BLE001 - after the lock: the reading is 不判, kept
        error, class_rows, mine_class = f'{type(exc).__name__}: {exc}', None, None
        reading = hp.REPLAY_READING_UNDECIDED
    result = dict(schema=SCHEMA, contract='C22.H8 开发集重放门 (D119 (a))', plan=plan, reading=reading, pairs=rows, b_class=dict(mine=mine_class, differences=class_rows), error=error, reference=checked,
                  admission=admission, ev=dict(module=EV_MODULE, functions=dict(EV_FUNCTIONS)), code=code, environment=env, seconds=perf_counter()-tick, utc=hc.utc(),
                  rule='every pair: 10 scores x 3 masks (AUSE, AUROC, largest tie share) and e, e4 bit for bit pq_v1, every non-null reference item compared (m6); class B: Delta, LB, R of the 8 '
                       'baselines bit for bit decision_pq.json with p1_core.block_interval')
    hc.write_json_once(hin.guard_write(Path(out_root)/RESULT_FILE), result)
    log(f'replay gate: {reading}; {Path(out_root)/RESULT_FILE} sha256 {sha_file(Path(out_root)/RESULT_FILE)}')
    return result


def replay_decision_problem(decision=None, *, repo=None, run=None):
    """For the evaluation entry's step (0) (C22.H7-1 last item): None when PQ_REPLAY_DECISION is filled with exactly REPLAY_DECISION_KEYS, its reading the pass reading, result_json the gate's own
    result_path() (B4 m2) and that file there with this sha256 and that reading, the plan the pinned one, the result's code commit (clean) and environment the decision's, the environment now the decision's, and the code tree `repo` (default this
    package's) by rule two from the decision's code_commit (bayesreg.io.code_identity.gate_identity_problems; I4, contract revision 9 draft C24.3-5); the problem otherwise (the entry then refuses
    before any truth, to BZ).  Never raises (review NM3): an exception of the check is returned as the problem.  The decision is a plain dict literal once filled (C24.3-6 (a)); a mapping proxy
    is read the same way."""
    try:
        return _replay_decision_problem(decision, repo=repo, run=run)
    except Exception as exc:                        # noqa: BLE001 - fail closed, recorded by the caller
        return f'the check of PQ_REPLAY_DECISION failed: {type(exc).__name__}: {exc}'


def _replay_decision_problem(decision, *, repo, run):
    decision = hp.PQ_REPLAY_DECISION if decision is None else decision
    if decision is None:
        return 'PQ_REPLAY_DECISION is None: the development replay gate has not been read and decided (C22.H8); the evaluation entry does not run'
    if not isinstance(decision, (dict, MappingProxyType)) or sorted(decision) != sorted(hp.REPLAY_DECISION_KEYS):
        return f'PQ_REPLAY_DECISION: exactly the keys {list(hp.REPLAY_DECISION_KEYS)} are required'
    if decision['reading'] != hp.REPLAY_READING_PASS:
        return f"PQ_REPLAY_DECISION reading {decision['reading']!r} is not {hp.REPLAY_READING_PASS!r}"
    if hp.REPLAY_PLAN_SHA256 is None or decision['plan_sha256'] != hp.REPLAY_PLAN_SHA256:
        return f"PQ_REPLAY_DECISION plan_sha256 {decision['plan_sha256']!r} is not the pinned {hp.REPLAY_PLAN_SHA256!r}"
    path = Path(str(decision['result_json']))
    if path.resolve() != result_path().resolve():      # B4 review m2 (the lead's ruling 2 of 11-14): the gate's own result, not another file with the right bytes
        return f"PQ_REPLAY_DECISION result_json {path} is not the gate's result {result_path()}"
    if not path.is_file() or sha_file(path) != decision['result_sha256']:
        return f"{path}: missing or not the sha256 {decision['result_sha256']!r}"
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return f'{path}: not JSON ({exc})'
    if result.get('reading') != decision['reading'] or (result.get('plan') or {}).get('sha256') != decision['plan_sha256']:
        return f"{path}: its reading {result.get('reading')!r} / plan {(result.get('plan') or {}).get('sha256')!r} are not the decision's"
    code = result.get('code') or {}
    if code.get('commit') != decision['code_commit'] or code.get('clean') is not True:
        return f"{path}: the gate ran on {code.get('commit')!r} (clean {code.get('clean')!r}), not the decision's code_commit {decision['code_commit']!r} on a clean tree (I4)"
    env = decision['environment']
    if not isinstance(env, (dict, MappingProxyType)) or sorted(env) != sorted(hp.REPLAY_ENVIRONMENT_KEYS) or dict(result.get('environment') or {}) != dict(env):
        return f"PQ_REPLAY_DECISION environment {env!r} is not the result's {result.get('environment')!r} with exactly the keys {list(hp.REPLAY_ENVIRONMENT_KEYS)} (I4)"
    now = environment()
    if now != dict(env):
        return f"the environment now {now} is not the gate's {dict(env)} (I4: C23.12-10's way)"
    from bayesreg.io import code_identity
    _, problems = code_identity.gate_identity_problems(decision['code_commit'], repo=HERE.parents[1] if repo is None else repo, run=run)
    if problems:
        return f"the code tree is not the gate's commit {decision['code_commit']} by rule two (the decision constants and the allowlist only): {problems[:5]} (I4)"
    return None


def production_context(taper):
    return io_.Context(repo=HERE.parents[1], source_root=pins.MAIN_REPO, results=pins.RESULTS, images=pins.DATA_IMAGES, taper=taper)


def main(argv=None):
    """python -B experiments/pq/pqh_replay.py dry-run | run   (run: inside a unit, after the plan is pinned and EV's functions are merged; PYTHONPATH=<frozen>:<code>)."""
    import argparse
    ap = argparse.ArgumentParser(description='the development replay gate of PQ (D119 (a), C22.H8)')
    ap.add_argument('command', choices=('dry-run', 'run'))
    o = ap.parse_args(argv)
    problems = []

    def lock_absent():
        if hp.REPLAY_LOCK.exists():
            raise ReplayRefused('once', f'{hp.REPLAY_LOCK} exists')
        return 'absent'
    for name, fn in (('plan', lambda: check_plan(HERE.parents[1])), ('ev', lambda: sorted(ev_functions())), ('lock', lock_absent)):
        try:
            print(f'{name}: {fn()}')
        except ReplayRefused as exc:
            problems.append(str(exc))
            print(f'{name}: PROBLEM {exc}')
    if o.command == 'dry-run':
        print(f"reference files: {len(REFERENCE_SHA256)} pinned (checked at the run, before the lock); problems: {problems or 'none'}")
        return 9 if problems else 0
    if problems:
        return 2
    import hi_core
    import holdout_common as hc
    if not hi_core.units_of_this_process(units=(REPLAY_UNIT,)):
        print(f'refused: run it in its unit {REPLAY_UNIT} (CPU; PYTHONPATH=<frozen>:<code>)')
        return 2
    try:
        hc.wait_for(lambda: hc.machine_gate(own_units=(REPLAY_UNIT,)))
    except hc.InfrastructureFailure as exc:
        print(f'refused: {exc}')
        return 20
    import mc0_core
    import pqh_units as hu
    try:
        layout = hu.hin.Layout(mode='dev', code=HERE.parents[1], pairs=(), stage=None, pq_small=None, pq_big=None, method_records=None, images=None, models_root=None, members_lock=None,
                               out_lock=None)
        problems = hu.import_problems(layout)
        if problems:
            raise ReplayRefused('code', f'm7: {problems}')
        ctx = production_context(mc0_core.frozen_api().boundary_taper)
        result = replay(ctx, ev=ev_functions(), out_root=result_path().parent, lock=hp.REPLAY_LOCK)
    except (Exception, SystemExit) as exc:          # noqa: BLE001 - m6, review NM3: every refusal or error of the run is recorded (REPLAY_ROOT/refusals/), never an exit without a record
        stage = getattr(exc, 'stage', None) or ('frozen' if isinstance(exc, SystemExit) else type(exc).__name__)
        record = hp.REPLAY_ROOT/'refusals'/f"{hc.utc().replace(':', '').replace('-', '')}_{os.getpid()}.json"
        hu.write_record(record, dict(schema=hp.SCHEMA.format('replay_refused'), stage=stage, reason=f'{type(exc).__name__}: {exc}'[:4000], lock_exists=Path(hp.REPLAY_LOCK).exists(),
                                     utc=hc.utc(), rule='the gate refused or failed outside its own after-the-lock handling (m6, NM3); lock_exists says whether it had taken its lock'))
        print(f'refused ({stage}): {exc}; {record}')
        return 2
    return 0 if result['reading'] == hp.REPLAY_READING_PASS else 8


if __name__ == '__main__':
    sys.exit(main())
