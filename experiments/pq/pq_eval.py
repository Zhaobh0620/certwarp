"""PQ, the evaluation unit (contract C22.Q4-2, C22.Q5; plan.md sections 4, 8 e1 e2): the only place of this package where the truth is read; each pair once.

Before any truth (in this order; no truth is read before step 6):
  1. the plan's sha256;
  2. the release (PQRefused 'release'): runs/20261107-pq/outputs_manifest.sha256 and runs/20261107-pq/release.md committed, unchanged, in ONE commit (the same last commit); the first holds one sha256 (the
     committed value of the list), the second is exactly one line `放行：<that sha256>` (full-line match; revision 1, review important 2) and holds no word of refusal (不、未、否、拒、not、no: any of them
     refuses).  The lead writes it after reading opened_files.json (C22.Q1-3, C22.Q5).  Both files are read as utf-8-sig (revision 2, diff review minor 3: a BOM is tolerated); the colon is the full-width
     "：", the sha256 64 lowercase hex;
  3. e2 (plan.md section 8): the list's own sha256 against the committed value AND every listed file hashed again.  Revision 1 (review important 3, the lead's ruling of 11-07 on e2, plan.md section 8 e2 and
     section 6 second row): ANY difference -- the list itself or a listed file -- is e2 not passed: eval/eval_record.json is written (truth_read false), the reading is 不判 and the last selection reading counts as
     used; the unit stops before the truth.  Revision 2 (diff review minor 4): the list's sha256 is compared before the list is parsed, and a list that cannot be parsed (清单格式坏) is e2 not passed too;
  4. the PQ unit's record: m1-m4 and m6 passed on the 19 pairs, m5 in the members' record (PQRefused 'controls');
  5. the guard of experiments/unv2/eval_seg.py admit_subjects (the 20 subjects in the allowlist source stage_L2R_val_v1, on no sealed list -- bayesreg.eval.truth.load_sealed_lists --, not
     never_for_training, no B-line training subject);
  6. eval/ is made only now (revision 1, review minor 5: a refusal of steps 4-5 leaves nothing behind); the lock pq_v1/pq_eval_truth.lock must not exist; it is taken with O_EXCL immediately before the
     first truth label and never removed: the lock is the mark "the truth was read" (run_pq.decide_unit reads it).  After the lock any stop -- an exception, SystemExit, KeyboardInterrupt, SIGTERM from systemd
     (turned into EvalStopped) -- writes eval_record.json with truth_read true, aborted true and the pairs done, then goes on (revision 1, review important 1); a kill that cannot be caught leaves the lock
     without a complete record, which the reading takes as 不判, counted as used.
Per pair: the truth through experiments/d0/evaluate.py truth_labels (sha256 against the manifest); v bound again (m1; the same d: its sha256 against pq.json); T~_M = warp_labels_l2r(T_M, d), e = 1[T_F != T~_M]
(the same iota as q); e^(4) likewise with the truth seg4; the masks (a) Omega_e = {T_F > 0} or {T~_M > 0} (D92, judged), (b) the two-sided boundary band of the truth seg35 (<= 2 mm and truth > 0 on that side),
(c) the whole grid (both described); every score's AUSE, AUROC and largest tie share on each mask; the reliability of q on Omega_e; the actual Dice with run_mc0.field_metrics' functions (dice_labels on 1..35
and the truth seg4 on 1..4, dice_l2r) and e1 against P3's repair_metrics.json after.metrics (bitwise; |difference| <= 1e-12 listed apart; more, or a missing item: e1 fails).
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
for _rel in ('unv2',):
    if str(HERE.parent/_rel) not in sys.path:
        sys.path.append(str(HERE.parent/_rel))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pq_core as core  # noqa: E402
import pq_io as io_  # noqa: E402
import pq_pins as pins  # noqa: E402
import pq_stats as stats  # noqa: E402
from pq_core import PQRefused  # noqa: E402

RELEASE_SHA_REL = 'runs/20261107-pq/outputs_manifest.sha256'
RELEASE_NOTE_REL = 'runs/20261107-pq/release.md'
RELEASE_LINE = re.compile(r'放行：([0-9a-f]{64})')               # revision 1 (review important 2): the whole release.md is this one line (full match), full-width colon
REFUSAL_WORDS = ('不', '未', '否', '拒', 'not', 'no')             # any of them anywhere in release.md refuses (English compared in lower case)
RELEASE_FORMAT_HINT = '全角冒号"："（U+FF1A），sha256 为 64 位小写十六进制，整份文件只有这一行；文件按 utf-8-sig 读，BOM 可以有'
EVAL_LOCK_NAME = 'pq_eval_truth.lock'
EVAL_SCHEMA = 'bayesreg.pq.eval.v1'
E1_TOL = 1e-12
MASKS = ('omega_e', 'band', 'whole')
BAND_MM = 2.
SCORES = ('q',)+stats.JUDGED_BASELINES+stats.DESCRIBED_SCORES
HEX = re.compile(r'[0-9a-f]{64}')


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ---- git: committed and unchanged -------------------------------------------------------------------------------------------------------------------
def _git(repo, *args):
    try:
        return subprocess.run(['git', '--literal-pathspecs', '-C', str(repo), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(args, 127, '', f'{type(exc).__name__}: {exc}')


def committed(repo, rel):
    """(last commit, problem): the file is tracked, unchanged against HEAD (not untracked, ignored, staged or modified)."""
    tracked = _git(repo, 'ls-files', '--error-unmatch', '--', rel)
    changed = _git(repo, 'status', '--porcelain', '--ignored', '--untracked-files=all', '--', rel)
    if tracked.returncode != 0 or tracked.stdout.strip() != rel or changed.returncode != 0 or changed.stdout.strip():
        return None, f'{rel}: not a committed, unchanged file (ls-files {tracked.returncode} {tracked.stdout.strip()!r}, status {changed.stdout.strip()!r})'
    last = _git(repo, 'log', '-1', '--format=%H', '--', rel).stdout.strip()
    return last or None, None if last else f'{rel}: no commit'


def release_note_problem(note, sha):
    """None when release.md is exactly the one line `放行：<sha>` (trailing newline allowed) and holds no word of refusal; the problem otherwise."""
    lines = [line for line in note.splitlines() if line.strip()]
    lowered = note.lower()
    words = [w for w in REFUSAL_WORDS if w in lowered]
    if words:
        return f'{RELEASE_NOTE_REL} holds the words of refusal {words}: not a release'
    if len(lines) != 1:
        return f'{RELEASE_NOTE_REL} must be exactly one line "放行：<the list\'s sha256>" ({RELEASE_FORMAT_HINT}), got {len(lines)} lines'
    match = RELEASE_LINE.fullmatch(lines[0].strip())
    if match is None:
        return f'{RELEASE_NOTE_REL}: {lines[0]!r} is not the line "放行：<64 hex>" ({RELEASE_FORMAT_HINT})'
    if match.group(1) != sha:
        return f'{RELEASE_NOTE_REL} releases {match.group(1)}, {RELEASE_SHA_REL} commits {sha}'
    return None


def release_check(ctx):
    """Step 2 of the module docstring: (the committed sha256 of the list, record).  PQRefused('release') otherwise.  The list itself is compared with the committed value in e2 (step 3), not here."""
    repo = Path(ctx.repo)
    problems = []
    lasts = {}
    for rel in (RELEASE_SHA_REL, RELEASE_NOTE_REL):
        if not (repo/rel).is_file():
            raise PQRefused('release', f'{rel}: missing; the lead releases after reading opened_files.json (C22.Q1-3) and commits the list\'s sha256 with the release (C22.Q5)')
        last, problem = committed(repo, rel)
        lasts[rel] = last
        if problem:
            problems.append(problem)
    if problems:
        raise PQRefused('release', str(problems))
    if lasts[RELEASE_SHA_REL] != lasts[RELEASE_NOTE_REL]:
        raise PQRefused('release', f'the list\'s sha256 and the release are not in one commit ({lasts})')
    text = (repo/RELEASE_SHA_REL).read_text(encoding='utf-8-sig')
    found = HEX.findall(text)
    if len(set(found)) != 1:
        raise PQRefused('release', f'{RELEASE_SHA_REL}: exactly one sha256 (64 lowercase hex) required, got {sorted(set(found))}')
    sha = found[0]
    problem = release_note_problem((repo/RELEASE_NOTE_REL).read_text(encoding='utf-8-sig'), sha)     # revision 2 (diff review minor 3): a BOM is tolerated
    if problem:
        raise PQRefused('release', problem)
    return sha, dict(sha256=sha, commit=lasts[RELEASE_SHA_REL], files=[RELEASE_SHA_REL, RELEASE_NOTE_REL], line=f'放行：{sha}')


def e2_check(ctx, committed_sha):
    """e2 (plan.md section 8): the list's own sha256 equal to the committed value, and every listed file hashed again (the PQ outputs and baselines must all be listed); (record, rows).  A list that is
    missing or not the committed one is e2 not passed too (revision 1, review important 3)."""
    rule = 'plan.md section 8 e2: the list\'s sha256 = the committed value, every listed file re-hashed; not passed -> 不判, counted as used (section 6, second row)'
    listed = ctx.pq_root/io_.MANIFEST_FILE
    if not listed.is_file():
        return dict(passed=False, list_sha256=None, committed_sha256=committed_sha, list_is_committed=False, differing=[], not_listed=[], n_files=0, problem=f'{listed} missing', rule=rule), {}
    list_sha = io_.sha_file(listed)                           # revision 2 (diff review minor 4): the list's sha256 first, then the parse
    same = list_sha == committed_sha
    try:
        rows = io_.parse_manifest(listed.read_text(encoding='utf-8'))
    except (UnicodeDecodeError, ValueError) as exc:
        return dict(passed=False, list_sha256=list_sha, committed_sha256=committed_sha, list_is_committed=same, differing=[], not_listed=[], n_files=0,
                    problem=f'清单格式坏 ({type(exc).__name__}: {str(exc)[:200]})', rule=rule), {}
    bad = [rel for rel, digest in rows.items() if not (ctx.pq_root/rel).is_file() or io_.sha_file(ctx.pq_root/rel) != digest]
    needed = [f'runs/{s}/{n}' for s in ctx.shorts for n in (io_.PQ_NPZ, io_.BASELINES_FILE, io_.PQ_JSON)]
    absent = [rel for rel in needed if rel not in rows]
    return dict(passed=same and not bad and not absent, list_sha256=list_sha, committed_sha256=committed_sha, list_is_committed=same, differing=bad, not_listed=absent, n_files=len(rows),
                problem=None, rule=rule), rows


def out_record_check(ctx):
    path = ctx.pq_root/'out_record.json'
    if not path.is_file():
        raise PQRefused('controls', f'{path}: the PQ unit has not finished')
    record = io_.read_json(path)
    controls = record.get('controls') or {}
    if record.get('passed') is not True or sorted(record.get('pairs') or []) != sorted(ctx.shorts) or not all(controls.get(k) is True for k in ('m1', 'm2', 'm3', 'm4', 'm5', 'm6')):
        raise PQRefused('controls', f'{path}: m1-m6 not all passed on the pairs of the plan ({controls})')
    return record


def admitted_subjects(ctx, *, allowlist, sealed, training):
    """eval_seg.admit_subjects (before any label): the 20 subjects of the allowlist source stage_L2R_val_v1, each checked against the sealed lists, never_for_training and the training set."""
    import eval_seg
    subjects = eval_seg.admit_subjects(allowlist, pins.DEV_SOURCE, sealed, training)
    if sorted(subjects) != ctx.subjects():
        raise PQRefused('guard', f'the allowlist source {pins.DEV_SOURCE} holds {sorted(subjects)}, not the plan\'s subjects')
    return subjects


def take_lock(path, line):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise PQRefused('once', f'{path} appeared before the first truth label; nothing is read') from None
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        stream.write(json.dumps(line, ensure_ascii=False)+'\n')
    return path


# ---- one pair ---------------------------------------------------------------------------------------------------------------------------------------
def actual_metrics(v, labs, spacing, ev):
    """run_mc0.field_metrics' Dice part, the same functions in the same order (experiments/mc0/run_mc0.py:197-208): seg35, seg4 (the truth seg4 map), the per-label rows, dice_official."""
    d = ev.disp_voxels(v, spacing)
    w35, w4 = ev.warp_labels_l2r(labs['moving_seg35'], d), ev.warp_labels_l2r(labs['moving_seg4'], d)
    seg35 = ev.dice_labels(labs['fixed_seg35'], w35.astype(np.float64), labels=list(range(1, 36)))
    seg4 = ev.dice_labels(labs['fixed_seg4'], w4.astype(np.float64), labels=list(range(1, 5)))
    dice, _ = ev.dice_l2r(labs['fixed_seg35'], labs['moving_seg35'], w35)
    per35, per4 = {r['label']: r['dice'] for r in seg35['per_label']}, {r['label']: r['dice'] for r in seg4['per_label']}
    return dict(seg35=seg35['macro_dice'], seg4=seg4['macro_dice'], dice_official=dice, seg35_per_label=per35, seg4_per_label=per4,
                counts35={r['label']: (r['fixed_count'], r['warped_count']) for r in seg35['per_label']}), d, w35, w4


def e1_compare(actual, recorded):
    """'bitwise', 'within_1e-12' or False, and the rows: seg35, seg4 and every label of seg35_per_label, seg4_per_label (JSON keys are strings)."""
    if not isinstance(recorded, dict):
        return False, [dict(item='after.metrics', problem='missing')]
    rows, worst, missing = [], 0., []

    def compare(name, mine, theirs):
        nonlocal worst
        if mine is None or theirs is None:
            if not (mine is None and theirs is None):
                missing.append(name)
            return
        diff = abs(float(mine)-float(theirs))
        worst = max(worst, diff)
        if diff:
            rows.append(dict(item=name, actual=mine, recorded=theirs, difference=diff))
    for key in ('seg35', 'seg4'):
        if key not in recorded:
            missing.append(key)
        else:
            compare(key, actual[key], recorded[key])
    for key, labels in (('seg35_per_label', range(1, 36)), ('seg4_per_label', range(1, 5))):
        table = recorded.get(key)
        if not isinstance(table, dict):
            missing.append(key)
            continue
        for c in labels:
            if str(c) not in table and c not in table:
                missing.append(f'{key}[{c}]')
                continue
            compare(f'{key}[{c}]', actual[key].get(c), table.get(str(c), table.get(c)))
    if missing or worst > E1_TOL:
        return False, rows+[dict(item=m, problem='missing on one side') for m in missing]
    return ('bitwise' if not rows else 'within_1e-12'), rows


def band_mask(T_F, T_tilde, spacing_xyz):
    """(b): the boundary voxels of each truth map (6 face neighbours inside the array with another label), the distance <= 2 mm and that side's truth > 0, the two sides joined."""
    from scipy.ndimage import distance_transform_edt
    out = np.zeros(np.shape(T_F), dtype=bool)
    for T in (np.asarray(T_F), np.asarray(T_tilde)):
        edge = core.boundary_voxels(T)
        if edge.any():
            out |= (distance_transform_edt(~edge, sampling=core.array_spacing(spacing_xyz)) <= BAND_MM) & (T > 0)
    return out


def score_keys(pq, base):
    """{score: the keys} (larger = more likely wrong); X_lex keeps its two keys; B_d = -delta (negation is exact)."""
    return dict(q=[pq['q']], h=[base['h']], H_F=[base['H_F']], H2=[base['H2']], V_F=[base['V_F']], V2=[base['V2']], G_plus=[base['G_plus']], X_lex=[base['h'], base['H2']], B_d=[-base['delta']],
                G_minus=[base['G_minus']])


def masked(keys, mask):
    return [k[mask] for k in keys]


def evaluate_pair(ctx, short, *, ev, read_truth, out_dir):
    """One pair, once: the evaluation of the module docstring.  Returns eval.json's document (written once to out_dir)."""
    tick = perf_counter()
    pq_dir = ctx.pq_root/'runs'/short
    document = io_.read_json(pq_dir/io_.PQ_JSON)
    from bayesreg.io.pair import _read_manifest
    grid, metadata, _ = _read_manifest(ctx.stage/pins.pair_name(short)/'manifest.json')
    v, field_record = io_.bind_field(ctx, short, grid)
    if field_record['state_hash'] != document['inputs']['field']['state_hash']:
        raise PQRefused('eval', f'{short}: v is not the one PQ used')
    labs, manifest, truth_sha = read_truth(pins.pair_name(short))
    spacing = tuple(float(s) for s in manifest['spacing_xyz'])
    if list(spacing) != document['sampling']['spacing_xyz']:
        raise PQRefused('eval', f'{short}: the spacing {spacing} is not PQ\'s {document["sampling"]["spacing_xyz"]}')
    actual, d, w35, w4 = actual_metrics(v, labs, spacing, ev)
    if io_.sha_array(d) != document['sampling']['d_sha256']:
        raise PQRefused('eval', f'{short}: d differs from the one of q (not the same iota)')
    with np.load(pq_dir/io_.PQ_NPZ, allow_pickle=False) as z:
        pq = {k: np.array(z[k]) for k in ('q', 'q4')}
    with np.load(pq_dir/io_.BASELINES_FILE, allow_pickle=False) as z:
        base = {k: np.array(z[k]) for k in core.BASELINE_KEYS}
    T_F, T_tilde = np.asarray(labs['fixed_seg35']), np.asarray(w35)
    e = T_F != T_tilde
    e4 = np.asarray(labs['fixed_seg4']) != np.asarray(w4)
    masks = dict(omega_e=(T_F > 0) | (T_tilde > 0), band=band_mask(T_F, T_tilde, spacing), whole=np.ones(T_F.shape, dtype=bool))
    keys = score_keys(pq, base)
    scores = {}
    for mask_name, mask in masks.items():
        scores[mask_name] = {}
        for name, k in keys.items():
            record = stats.score_record(masked(k, mask), e[mask])
            if mask_name != 'omega_e':
                record.pop('curve')
            scores[mask_name][name] = record
    omega = masks['omega_e']
    seg4_scale = stats.score_record([pq['q4'][omega]], e4[omega])
    reliability = dict(width=stats.reliability(pq['q'][omega], e[omega], bins='width'), decile=stats.reliability(pq['q'][omega], e[omega], bins='decile'))
    recorded = io_.read_json(ctx.repair_dir(short)/'repair_metrics.json')        # the evaluation's own record of P3 (plan.md section 4: only here)
    e1, e1_rows = e1_compare(actual, ((recorded.get('after') or {}).get('metrics')))
    out = dict(schema=EVAL_SCHEMA, pair=pins.pair_name(short), short=short, plan_sha256=pins.PLAN_SHA256, truth=dict(sha256=truth_sha, reader='experiments/d0/evaluate.py truth_labels'),
               state_hash=field_record['state_hash'], d_sha256=io_.sha_array(d), masks={k: int(m.sum()) for k, m in masks.items()}, n_errors={k: int(e[m].sum()) for k, m in masks.items()},
               n_errors_seg4=int(e4[omega].sum()), scores=scores, seg4_scale=dict(q4=seg4_scale), reliability=reliability, actual=actual, e1=e1, e1_rows=e1_rows,
               repair_metrics_sha256=io_.sha_file(ctx.repair_dir(short)/'repair_metrics.json'), seconds=perf_counter()-tick)
    out_dir = Path(out_dir)
    io_.write_once(out_dir/'eval.json', out)
    with open(out_dir/'e.npz', 'xb') as stream:
        np.savez_compressed(stream, e=e.astype(np.uint8), e4=e4.astype(np.uint8))
    return out
