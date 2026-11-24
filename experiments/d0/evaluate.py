"""D0 evaluation (docs/bayesreg/05_D0协议.md §5, §7, §12). Metrics come only from the saved displacement and the truth labels.

  evaluate.py run RUN_DIR --arm ARM --pair PAIR --run-id ID [--out RUN_DIR/metrics.json]     (ARM: A0-A3, or A4p of §12)
  evaluate.py summarize [--runs-root DIR] [--out-dir DIR]
  evaluate.py summarize-a4 [--a4-runs-root DIR] [--d0-runs-root DIR] [--out-dir DIR]
        the supplementary arm A4p (protocol §12) against the D0 batch: A4p from d0_v1_a4/runs, A0 (and, descriptive,
        A1, A2, A3) read from the metrics.json files of d0_v1/runs (read only, not re-evaluated, not re-run; A0 is not
        concurrent with A4p); writes long.csv, summary_a4.md and decision_a4.json (rules R4a / R4b of §12.7) into
        d0_v1_a4/analysis; seg35 and seg4 always both.
  evaluate.py summarize-a1p [--a1p-runs-root DIR] [--d0-runs-root DIR] [--a4-runs-root DIR] [--out-dir DIR]
        the arms A1Ph, A1Ps (runs/20261102-a1p/plan.md) against A0, A1 (d0_v1/runs) and A4p (d0_v1_a4/runs), metrics.json
        read only, not re-evaluated, not concurrent; writes long.csv, summary_a1p.md and decision_a1p.json (the plan's rule:
        mean seg35 >= .845) into d0_v1_a1p/analysis; seg35 and seg4 always both; the per-structure Dice of the cortex and the
        white matter (seg4 labels 1, 3; seg35 labels 2, 21, 1, 20, 5, 24) and the contributions to seg35.
  evaluate.py summarize-x1 [--x1-runs-root DIR] [--d0-runs-root DIR] [--pilot-root DIR] [--out-dir DIR]
        gate A X1 (runs/20261102-gate-a/plan.md): the arms X1h, X1f (the first 5 pairs) against A0 (d0_v1/runs) and P0n
        (path_pilot_v1/eval/p0n), metrics.json read only, not concurrent; writes long.csv, summary_x1.md and decision_x1.json
        (the plan's reading on seg4, revision 1: comparable runs, thresholds on the arm's own pairs) into gate_a_x1/analysis; E, F, ratio
        and step counts from the accepted rows of the runs' history.jsonl; the seed that the --x1 cleanup keeps, descriptive only.
  evaluate.py summarize-a5 [--a5-runs-root DIR] [--d0-root DIR] [--a4-root DIR] [--a1p-root DIR] [--out-dir DIR]
        the arms A5, A5P (runs/20261103-a5/plan.md, protocol §14) against A0, A1, A3 (d0_v1), A4p (d0_v1_a4) and A1Ph (d0_v1_a1p): the
        references' numbers from the original analysis/long.csv of each batch, their per-label Dice from its metrics.json (same run_id
        and Dice checked), only the same pairs compared; runs with 5 pairs or with 19; writes long.csv, summary_a5.md and decision_a5.json
        (the plan's first and second gate, by code) into d0_v1_a5/analysis; earlier summaries are moved aside, never overwritten.
  evaluate.py run-b1 --spec SPEC RUN_DIR --arm ARM --pair PAIR --run-id ID [--out RUN_DIR/metrics.json]
        as `run`, for an arm of a B1 specification (experiments/d0/b1_spec.py): the record names the arm's configuration; the definitions are the same.
  evaluate.py summarize-b1 [--b1-root DIR] [--selection-spec FILE] [--a5-root DIR] [--out-dir DIR]
        the option arms of runs/20261103-b1/plan.md (protocol §15) against A5P and A5 of d0_v1_a5, whose numbers come from its analysis/long.csv and
        its metrics.json (same run_id and Dice checked), not rerun, not concurrent: writes long.csv, summary_b1.md and decision_b1.json (the plan's
        selection on the first 5 pairs and its readings, by code) into b1_v1/analysis; earlier summaries are moved to previous_<UTC>/.

Definitions (the bl1_eval convention of paper 1, inventory B §1; every function is reused or copied with provenance):
  displacement  d = disp_voxels(u) from final_native.npz:u ((z,y,x,3) mm, xyz components) -> (3,z,y,x) voxels, pull-back
                warped(p) = moving(p + d(p)) (scripts/evaluate_l2r_style.py; Learn2Reg semantics, bitwise equal to the
                official evaluation per E83)
  labels        truth labels from the ORIGINAL data/three_d/stage_L2R_val_v1/<pair>/pair.npz (sha256 checked against its
                manifest), warped with map_coordinates(order=0) (warp_labels_l2r)
  seg35, seg4   macro Dice over labels 1..35 / 1..4 (jd_curl.three_d.nifti.dice_labels; a label absent from both images
                is skipped); both are always reported
  dice_official Learn2Reg Dice35 (dice_l2r: NaN for a label missing in the fixed or the unwarped moving image; nanmean)
  hd95          Learn2Reg HD95 over labels 1..35 (hd95_l2r, surface_distance, nanmean)
  sdlogj_official Learn2Reg SDlogJ: std of log(clip(det(I + central-difference grad d) + 3, 1e-9, 1e9)), 2-voxel crop
                (sdlogj_l2r; its folding count is always 0 and is not evidence of no folding)
  sdlogj_brain  std of log(clip(J, 1e-12)) at the cell centres whose lower-corner node has fixed seg4 > 0; J = det(I + Q1
                centre derivative) (copy of eb808be jd_curl/three_d/smoothness.py, the frozen run's smoothness.json)
  certificate   the run's own certificate of the exported state (final_native.json:certificate: status, min_jd_lower_bound,
                n_infeasible, n_uncertain); passed = status certified_feasible, min_jd_lower_bound > 0, no infeasible or
                uncertain cell, and state_hash equal to the sha256 of the exported u bytes
  self_check    seg4/seg35 against the run's labels.json and sdlogj_brain against its smoothness.json (|diff| <= 1e-12)
"""
import argparse
import csv
import glob
import hashlib
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_spec  # noqa: E402
from evaluate_l2r_style import dice_l2r, disp_voxels, hd95_l2r, sdlogj_l2r, warp_labels_l2r  # noqa: E402
from jd_curl.three_d.nifti import dice_labels  # noqa: E402

STAGE = ROOT/'data/three_d/stage_L2R_val_v1'
D0 = Path('/home/rmcfz/seq_vp_tag_results/d0_v1')
D0A4 = Path('/home/rmcfz/seq_vp_tag_results/d0_v1_a4')
D0A1P = Path('/home/rmcfz/seq_vp_tag_results/d0_v1_a1p')
D0A5 = Path('/home/rmcfz/seq_vp_tag_results/d0_v1_a5')
GAX1 =Path('/home/rmcfz/seq_vp_tag_results/gate_a_x1')
PILOT = Path('/home/rmcfz/seq_vp_tag_results/path_pilot_v1')
ARMS = ('A0', 'A1', 'A2', 'A3')
SUPPLEMENTARY = ('A4p',)                                  # protocol §12 (A4': U-Net posterior colouring)
A1P_ARMS = ('A1Ph', 'A1Ps')                               # runs/20261102-a1p/plan.md (predicted seg4 colouring)
X1_ARMS = ('X1h', 'X1f')                                  # runs/20261102-gate-a/plan.md (absolute weights .015, .03)
A5_ARMS = ('A5', 'A5P')                                   # runs/20261103-a5/plan.md (fixed colour table on truth / predicted seg35)
X1_SEED_FILE = 'level_3_level_prepared.npz'               # the native seed that run_queue.sh --x1 keeps
METHOD = 'configC_frozen_eb808be_90s'
METHOD_OF_ARM = dict(X1h='frozen_eb808be_T92_identity_s0.5_90s', X1f='frozen_eb808be_T92_identity_90s')    # not configuration C
SET, SPLIT = 'L2R_val', 'dev'
TOL = 1e-12
DEFINITIONS = 'Definitions' + __doc__.split('Definitions', 1)[1].rstrip()     # the module docstring block, verbatim


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


# ---- copied from eb808be jd_curl/three_d/smoothness.py (cell_jacobian_centres, smoothness_report; E78) -------------------
def cell_jacobian_centres(u, spacing):
    """det(I + grad u) at every cell centre; u is a (nz, ny, nx, 3) node field, components xyz."""
    u = np.asarray(u, dtype=np.float64)
    hx, hy, hz = spacing

    def d(axis, comp):
        a = u[..., comp]
        if axis == 0:
            diff = (a[:, :, 1:]-a[:, :, :-1])/hx
            return .25*(diff[:-1, :-1]+diff[1:, :-1]+diff[:-1, 1:]+diff[1:, 1:])
        if axis == 1:
            diff = (a[:, 1:, :]-a[:, :-1, :])/hy
            return .25*(diff[:-1, :, :-1]+diff[1:, :, :-1]+diff[:-1, :, 1:]+diff[1:, :, 1:])
        diff = (a[1:, :, :]-a[:-1, :, :])/hz
        return .25*(diff[:, :-1, :-1]+diff[:, 1:, :-1]+diff[:, :-1, 1:]+diff[:, 1:, 1:])
    with np.errstate(over='raise', invalid='raise'):
        F = [[(1. if r == j else 0.)+d(j, r) for j in range(3)] for r in range(3)]
        J = (F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1]) - F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
             + F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]))
    if not np.all(np.isfinite(J)):
        raise FloatingPointError('non-finite cell-centre Jacobian determinant')
    return J


def brain_smoothness(u, spacing, brain_nodes):
    J = cell_jacobian_centres(u, spacing)
    with np.errstate(divide='ignore', invalid='ignore'):
        lj = np.log(np.clip(J, 1e-12, None))
    cells = np.asarray(brain_nodes, bool)[:-1, :-1, :-1]
    return dict(sdlogj_brain=float(lj[cells].std()), pct_J_lt_half_brain=float(100*np.mean(J[cells] < .5)),
                min_J_centre=float(J.min()), cell_centre_J_nonpositive=int((J <= 0).sum()), n_brain_cells=int(cells.sum()))
# ---------------------------------------------------------------------------------------------------------------------------


def native_interval_median(task):
    """TS1b speed record (paper 1): median of the native (level 3) commit intervals 1..16, from history.jsonl."""
    path = task/'history.jsonl'
    if not path.exists():
        return None, 0
    e = [r['elapsed_seconds'] for r in map(json.loads, open(path)) if r.get('level') == 3]
    e = e[1:18]
    iv = [b-a for a, b in zip(e[:-1], e[1:])]
    return (float(statistics.median(iv)) if len(iv) >= 4 else None), len(iv)


def stage_pair(pair):
    """Contract C23.6-2 revision 6h (the F rows, SG code review I2): the pair named by the caller (--pair is free text) must be a plain name -- no '/', no
    '\\', no '..' -- registered in STAGE/selection.json, and STAGE/<pair> must not be a link nor lie outside STAGE; otherwise SystemExit before anything else
    is read.  So `--pair ../stage_H1_v1/<pair>` cannot reach another stage's labels.  Reads STAGE/selection.json only (pair names, no subject file)."""
    if not isinstance(pair, str) or not pair or any(c in pair for c in ('/', '\\', '\x00')) or '..' in pair or pair == '.':
        raise SystemExit(f'{pair!r}: not a plain pair name; nothing is read (C23.6-2 revision 6h)')
    try:
        rows = json.loads((STAGE/'selection.json').read_text())['pairs']
        names = [r['pair'] if isinstance(r, dict) else r for r in rows]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f'{STAGE}/selection.json cannot be read ({type(exc).__name__}); nothing is read (C23.6-2 revision 6h)') from None
    if pair not in names:
        raise SystemExit(f'{pair}: not a pair of {STAGE}/selection.json; nothing is read (C23.6-2 revision 6h)')
    d = STAGE/pair
    if d.is_symlink() or d.resolve().parent != STAGE.resolve():
        raise SystemExit(f'{d}: a link, or outside {STAGE}; nothing is read (C23.6-2 revision 6h)')
    return pair


def run_copy(run, pair):
    """Contract C23 revision 6i (SG follow-up review n2): the run directory's copy of the pair, <run>/inputs/<pair>/pair.npz, is read only when neither it nor
    <run>/inputs nor <run>/inputs/<pair> is a link and it lies in the run directory; SystemExit otherwise, before it is read.  Returns its path."""
    run = Path(run)
    copy = run/'inputs'/pair/'pair.npz'
    if any(p.is_symlink() for p in (run/'inputs', run/'inputs'/pair, copy)) or not copy.resolve().is_relative_to(run.resolve()):
        raise SystemExit(f'{copy}: a link, or outside the run directory {run}; not read (C23 revision 6i, review n2)')
    return copy


def truth_labels(pair):
    stage_pair(pair)                                                     # C23.6-2 revision 6h: before anything is read
    d = STAGE/pair
    man = json.loads((d/'manifest.json').read_text())
    if sha_file(d/'pair.npz') != man['pair_sha256']:
        raise SystemExit(f'{pair}: original pair.npz sha256 differs from its manifest')
    with np.load(d/'pair.npz') as z:
        labs = {k: np.array(z[k]) for k in ('fixed_seg4', 'fixed_seg35', 'moving_seg4', 'moving_seg35')}
    return labs, man, sha_file(d/'pair.npz')


def rd(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def evaluate_run(run, arm, pair, run_id, method=None):
    stage_pair(pair)                                                     # C23.6-2 revision 6h: before anything is read (the run directory's copy too)
    run = Path(run)
    proto = rd(run/'protocol.json')
    task = run/proto['task']['name']
    summary, completion, final = rd(task/'summary.json'), rd(run/'completion.json'), rd(task/'final_native.json')
    cls = rd(run/'classification.json')
    rec = dict(schema='d0.metrics.v1', run_id=run_id, set=SET, split=SPLIT, method=method or METHOD_OF_ARM.get(arm, METHOD), arm=arm, pair=pair,
               run_dir=str(run), evaluated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               classification=None if cls is None else {k: cls.get(k) for k in ('class', 'kind', 'reason')},
               status=None if summary is None else summary.get('status'),
               success=None if summary is None else summary.get('success'),
               metrics_role=None if summary is None else summary.get('metrics_role'),
               native_commits=None if summary is None else summary.get('native_commits'),
               e2e_seconds=None if summary is None else summary.get('e2e_seconds'),
               GF=None if summary is None else summary.get('GF'), GF_threshold=None if summary is None else summary.get('GF_threshold'),
               completion_seconds=None if completion is None else completion.get('seconds'),
               definitions=DEFINITIONS,
               evaluator=dict(script='experiments/d0/evaluate.py', sha256=sha_file(Path(__file__)),
                              evaluate_l2r_style_sha256=sha_file(ROOT/'scripts/evaluate_l2r_style.py'),
                              nifti_sha256=sha_file(ROOT/'jd_curl/three_d/nifti.py')))
    rec['native_interval_median_s'], rec['native_intervals_n'] = native_interval_median(task)
    if final is None or not (task/'final_native.npz').exists():
        rec.update(metrics=None, note='no exported map (method failure or incomplete run); no image metrics')
        if arm in X1_ARMS:                                # gate A plan, revision 1: the seed, whatever the native level did
            labs, man, _ = truth_labels(pair)
            rec['seed'] = x1_seed_record(task, labs, tuple(man['spacing_xyz']))
        return rec
    labs, man, truth_sha = truth_labels(pair)
    rec['truth'] = dict(path=str(STAGE/pair/'pair.npz'), sha256=truth_sha)
    run_copy(run, pair)                                                  # C23 revision 6i (review n2): the run's copy, not a link, before it is read
    with np.load(run/'inputs'/pair/'pair.npz') as z:                     # the run's own label copy == the truth
        rec['run_labels_equal_truth'] = bool(all(np.array_equal(z[k], labs[k]) for k in labs))
    npz_sha = sha_file(task/'final_native.npz')
    with np.load(task/'final_native.npz') as z:
        u = np.array(z['u'])
    cert = final.get('certificate') or {}
    state_sha = hashlib.sha256(np.ascontiguousarray(u, dtype=np.float64).view(np.uint8)).hexdigest()
    rec['certificate'] = dict(status=cert.get('status'), min_jd_lower_bound=cert.get('min_jd_lower_bound'),
                              n_cells_checked=cert.get('n_cells_checked'), n_infeasible=cert.get('n_infeasible'),
                              n_uncertain=cert.get('n_uncertain'), certificate_version=cert.get('certificate_version'),
                              state_hash=cert.get('state_hash'), state_hash_matches_u=cert.get('state_hash') == state_sha,
                              final_npz_sha256=npz_sha, final_npz_sha256_recorded=final.get('sha256'),
                              final_npz_sha256_matches=final.get('sha256') in (None, npz_sha), role=final.get('role'))
    lower = cert.get('min_jd_lower_bound')
    rec['certificate']['passed'] = bool(cert.get('status') == 'certified_feasible' and lower is not None and lower > 0
                                        and cert.get('n_infeasible') == 0 and cert.get('n_uncertain') == 0
                                        and rec['certificate']['state_hash_matches_u'])
    spacing = tuple(man['spacing_xyz'])
    d = disp_voxels(u, spacing)
    w35, w4 = warp_labels_l2r(labs['moving_seg35'], d), warp_labels_l2r(labs['moving_seg4'], d)
    seg35 = dice_labels(labs['fixed_seg35'], w35.astype(np.float64), labels=list(range(1, 36)))
    seg4 = dice_labels(labs['fixed_seg4'], w4.astype(np.float64), labels=list(range(1, 5)))
    dice, _ = dice_l2r(labs['fixed_seg35'], labs['moving_seg35'], w35)
    hd, _ = hd95_l2r(labs['fixed_seg35'], labs['moving_seg35'], w35)
    sd, folds, neg = sdlogj_l2r(d)
    sm = brain_smoothness(u, spacing, labs['fixed_seg4'] > 0)
    rec['metrics'] = dict(seg35=seg35['macro_dice'], seg4=seg4['macro_dice'], dice_official=dice, hd95=hd,
                          sdlogj_official=sd, num_foldings_official=folds, central_det_nonpositive_fraction=neg, **sm,
                          seg4_per_label={r['label']: r['dice'] for r in seg4['per_label']},
                          seg35_per_label={r['label']: r['dice'] for r in seg35['per_label']})
    lab_json, sm_json = rd(task/'labels.json'), rd(task/'smoothness.json')
    diffs = {}
    if lab_json and lab_json.get('final'):
        diffs['seg4'] = seg4['macro_dice']-lab_json['final']['seg4']['macro_dice']
        diffs['seg35'] = seg35['macro_dice']-lab_json['final']['seg35']['macro_dice']
    if sm_json and sm_json.get('final'):
        diffs['sdlogj_brain'] = sm['sdlogj_brain']-sm_json['final']['sdlogj_brain']
    rec['self_check'] = dict(diffs=diffs, tolerance=TOL, passed=bool(diffs) and all(abs(v) <= TOL for v in diffs.values()))
    if arm in X1_ARMS:                                    # gate A plan, revision 1 (descriptive): the other arms are unchanged
        rec['seed'] = x1_seed_record(task, labs, spacing)
    return rec


def x1_seed_record(task, labs, spacing):
    """The native seed of an X1 run (level_3_level_prepared.npz:u, the certified T88 handover state; run_queue.sh --x1 keeps it):
    seg35, seg4 and per-label Dice by the same functions as the final map.  Descriptive only (plan revision 1, review I3)."""
    path = Path(task)/X1_SEED_FILE
    if not path.is_file():
        return dict(present=False, file=X1_SEED_FILE)
    with np.load(path) as z:
        u = np.array(z['u'])
    d = disp_voxels(u, spacing)
    w35, w4 = warp_labels_l2r(labs['moving_seg35'], d), warp_labels_l2r(labs['moving_seg4'], d)
    seg35 = dice_labels(labs['fixed_seg35'], w35.astype(np.float64), labels=list(range(1, 36)))
    seg4 = dice_labels(labs['fixed_seg4'], w4.astype(np.float64), labels=list(range(1, 5)))
    return dict(present=True, file=X1_SEED_FILE, sha256=sha_file(path),
                state_hash=hashlib.sha256(np.ascontiguousarray(u, dtype=np.float64).view(np.uint8)).hexdigest(),
                seg35=seg35['macro_dice'], seg4=seg4['macro_dice'], seg4_per_label={r['label']: r['dice'] for r in seg4['per_label']},
                seg35_per_label={r['label']: r['dice'] for r in seg35['per_label']})


# ---- summary -------------------------------------------------------------------------------------------------------------
LONG_MEASURES = ('seg35', 'seg4', 'dice_official', 'hd95', 'sdlogj_official', 'sdlogj_brain', 'num_foldings_official',
                 'pct_J_lt_half_brain', 'cell_centre_J_nonpositive')


def chosen_runs(runs_root, arms=ARMS):
    """{(arm, pair): metrics} using, per arm and pair, the last attempt (<pair>, then <pair>__r1) that has metrics.json."""
    out = {}
    for arm in arms:
        for d in sorted(glob.glob(str(Path(runs_root)/arm/'*'))):
            d = Path(d)
            m = rd(d/'metrics.json')
            if m is None:
                continue
            pair = re.sub(r'__r[0-9]+$', '', d.name)
            out[(arm, pair)] = m
    return out


def long_rows(runs):
    rows = []
    for (arm, pair), m in sorted(runs.items()):
        base = dict(set=SET, split=SPLIT, method=METHOD_OF_ARM.get(arm, METHOD), arm=arm, pair=pair, run_id=m['run_id'])
        vals = {}
        if m.get('metrics'):
            vals.update({k: m['metrics'][k] for k in LONG_MEASURES})
        c = m.get('certificate') or {}
        vals.update(cert_passed=None if not c else int(bool(c.get('passed'))), cert_min_jd_lower_bound=c.get('min_jd_lower_bound'),
                    cert_n_infeasible=c.get('n_infeasible'), cert_n_uncertain=c.get('n_uncertain'),
                    success=None if m.get('success') is None else int(bool(m['success'])), native_commits=m.get('native_commits'),
                    e2e_seconds=m.get('e2e_seconds'), native_interval_median_s=m.get('native_interval_median_s'),
                    self_check_passed=None if not m.get('self_check') else int(bool(m['self_check']['passed'])))
        for k, v in vals.items():
            if v is not None:
                rows.append(dict(base, measure=k, value=v))
    return rows


def paired(runs, arm, measure, pairs):
    out = {}
    for p in pairs:
        a, b = runs.get((arm, p)), runs.get(('A0', p))
        if a and b and a.get('metrics') and b.get('metrics'):
            out[p] = a['metrics'][measure]-b['metrics'][measure]
    return out


def decide(runs, pairs):
    """The pre-registered rule (protocol §7): medians of per-pair seg35 differences vs A0; positives out of 19."""
    n = len(pairs)
    d1, d2 = paired(runs, 'A1', 'seg35', pairs), paired(runs, 'A2', 'seg35', pairs)
    med1 = float(np.median(list(d1.values()))) if d1 else None
    med2 = float(np.median(list(d2.values()))) if d2 else None
    pos2 = sum(v > 0 for v in d2.values())
    res = dict(n_pairs=n, A1_minus_A0=dict(median=med1, n=len(d1), positive=sum(v > 0 for v in d1.values())),
               A2_minus_A0=dict(median=med2, n=len(d2), positive=pos2))
    if len(d1) < n-2 or len(d2) < n-2:
        res.update(rows=[], conclusion='不可判：A1 或 A2 相对 A0 的有效配对少于 17 对（协议 §7）')
        return res
    rows = []
    if med2 >= .01 and pos2 >= 14:
        rows.append('R1')
    if med1 >= .01 and med2 < .005:
        rows.append('R2')
    if med1 < .005:
        rows.append('R3')
    text = {'R1': 'R1：组织通道值得做（A2−A0 逐对 seg35 中位差 ≥ .01 且 ≥ 14/19 对为正）',
            'R2': 'R2：组织信息有用，但三类后验拿不到（A1−A0 ≥ .01，A2−A0 < .005）',
            'R3': 'R3：组织层没有余量（A1−A0 < .005）'}
    if len(rows) == 1:
        concl = text[rows[0]]
    elif not rows:
        concl = '未落入预登记的任何一行（中间区域）：只描述，不下结论'
    else:
        concl = '多行同时成立（' + '、'.join(rows) + '）：交总负责人判断'
    res.update(rows=rows, conclusion=concl, row_texts={r: text[r] for r in rows})
    return res


def fmt(v, nd=4):
    return '—' if v is None else f'{v:.{nd}f}'


def summarize(runs_root, out_dir):
    runs = chosen_runs(runs_root)
    pairs = [r['pair'] for r in json.loads((STAGE/'selection.json').read_text())['pairs']]
    rows = long_rows(runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/'long.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['set', 'split', 'method', 'arm', 'pair', 'measure', 'value', 'run_id'])
        w.writeheader(); w.writerows(rows)
    dec = decide(runs, pairs)
    L = ['# D0 汇总（开发集结果，单次观测）', '',
         f'生成：{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}（UTC）；协议 docs/bayesreg/05_D0协议.md；'
         f'方法 {METHOD}；19 个 Learn2Reg 验证对；指标由 experiments/d0/evaluate.py 从终态场算出。', '',
         '## 各臂均值（seg4 与 seg35 同时报）', '',
         '| 臂 | 有终态场的对数 | seg35 | seg4 | Dice35 官方 | HD95 | SDlogJ 官方 | 脑内 SDlogJ | 证书通过 | 自检通过 | e2e 中位 s |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in ARMS:
        ms = [runs[(arm, p)] for p in pairs if (arm, p) in runs]
        ok = [m for m in ms if m.get('metrics')]
        mean = (lambda k: float(np.mean([m['metrics'][k] for m in ok])) if ok else None)
        e2e = [m['e2e_seconds'] for m in ms if m.get('e2e_seconds') is not None]
        L.append(f"| {arm} | {len(ok)}/{len(pairs)} | {fmt(mean('seg35'))} | {fmt(mean('seg4'))} | {fmt(mean('dice_official'))} | "
                 f"{fmt(mean('hd95'), 3)} | {fmt(mean('sdlogj_official'))} | {fmt(mean('sdlogj_brain'), 3)} | "
                 f"{sum(bool((m.get('certificate') or {}).get('passed')) for m in ms)}/{len(ms)} | "
                 f"{sum(bool((m.get('self_check') or {}).get('passed')) for m in ms)}/{len(ms)} | "
                 f"{fmt(float(np.median(e2e)) if e2e else None, 1)} |")
    L += ['', '## 相对同时段对照 A0 的逐对差（臂 − A0）', '',
          '| 臂 | 指标 | 中位差 | 均值差 | 为正的对数 / 有效对数 |', '|---|---|---|---|---|']
    for arm in ARMS[1:]:
        for k in ('seg35', 'seg4', 'hd95', 'sdlogj_official', 'sdlogj_brain'):
            dd = paired(runs, arm, k, pairs)
            v = list(dd.values())
            L.append(f"| {arm} | {k} | {fmt(float(np.median(v)) if v else None)} | {fmt(float(np.mean(v)) if v else None)} | "
                     f"{sum(x > 0 for x in v)} / {len(v)} |")
    L += ['', '## 逐对 seg35（seg4）', '', '| 对 | ' + ' | '.join(ARMS) + ' |', '|---|' + '---|'*len(ARMS)]
    for p in pairs:
        cells = []
        for arm in ARMS:
            m = runs.get((arm, p))
            cells.append('—' if not m or not m.get('metrics') else f"{m['metrics']['seg35']:.4f} ({m['metrics']['seg4']:.4f})")
        L.append(f"| {p[11:15]}←{p[-8:-4]} | " + ' | '.join(cells) + ' |')
    L += ['', '## 预登记判据（协议 §7）', '',
          f"- A1 − A0 seg35 逐对中位差：{fmt(dec['A1_minus_A0']['median'])}（{dec['A1_minus_A0']['positive']}/{dec['A1_minus_A0']['n']} 对为正）",
          f"- A2 − A0 seg35 逐对中位差：{fmt(dec['A2_minus_A0']['median'])}（{dec['A2_minus_A0']['positive']}/{dec['A2_minus_A0']['n']} 对为正）",
          f"- 成立的行：{', '.join(dec['rows']) if dec['rows'] else '无'}", f"- 结论：{dec['conclusion']}",
          '- A3（seg35 上色 oracle）只作描述，不进判据。', '',
          '注意：这是开发集上的单次观测；A1、A3 用了真值标签（oracle），只作上限；指标由评测代码从位移场算出。']
    (out_dir/'summary.md').write_text('\n'.join(L)+'\n')
    (out_dir/'decision.json').write_text(json.dumps(dec, indent=1, ensure_ascii=False)+'\n')
    print('\n'.join(L))
    return 0


# ---- supplementary arm A4p (protocol §12) ------------------------------------------------------------------------------
A4_ARM = SUPPLEMENTARY[0]
A4_LABEL = "A4′"
LAUNCH = re.compile(r' launch \S+; loadavg ([0-9.]+) ([0-9.]+) ([0-9.]+)')
A4_COMPARISONS = ((A4_ARM, 'A0'), (A4_ARM, 'A1'), (A4_ARM, 'A3'), ('A1', 'A0'), ('A3', 'A0'))
A4_ROW_TEXT = {'R4a': 'R4a（单通道诊断）：在只换输入图像的单通道诊断下，自训分割器的后验上色能拿回可观的上色收益，'
                      '支持做结构层通道；不直接预测 bayesreg 结构层通道的收益（协议 §12.1）'
                      '（A4′−A0 逐对 seg35 中位差 ≥ .01 且 ≥ 14/19 对为正）',
               'R4b': 'R4b：自训分割器的后验也拿不到（A4′−A0 逐对 seg35 中位差 < .005）'}


def paired_ref(runs, arm, ref, measure, pairs):
    """{pair: measure(arm) - measure(ref)} over the pairs where both runs have image metrics."""
    out = {}
    for p in pairs:
        a, b = runs.get((arm, p)), runs.get((ref, p))
        if a and b and a.get('metrics') and b.get('metrics'):
            out[p] = a['metrics'][measure]-b['metrics'][measure]
    return out


def _median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def decide_a4(runs, pairs):
    """The pre-registered rule of protocol §12.7 on Delta(p) = seg35(A4p, p) - seg35(A0, p):
    R4a median >= .01 and Delta > 0 on >= 14 of the 19 pairs; R4b median < .005; neither: the middle region (described,
    no conclusion). Missing (as §7): a pair without a final map in A4p or A0 leaves the median and counts as not positive
    (denominator 19); fewer than 17 valid pairs: not decidable. Recovery ratios (descriptive only, not in the rule):
    median(A4p - A0) / median(A1 - A0) and / median(A3 - A0), for seg35 and for seg4."""
    n = len(pairs)
    d = paired_ref(runs, A4_ARM, 'A0', 'seg35', pairs)
    med, pos = _median(d.values()), sum(v > 0 for v in d.values())
    recovery = {}
    for measure in ('seg35', 'seg4'):
        m4 = _median(paired_ref(runs, A4_ARM, 'A0', measure, pairs).values())
        row = dict(A4p_minus_A0_median=m4)
        for ref in ('A1', 'A3'):
            mr = _median(paired_ref(runs, ref, 'A0', measure, pairs).values())
            row[f'{ref}_minus_A0_median'] = mr
            row[f'recovery_vs_{ref}'] = None if m4 is None or not mr else m4/mr
        recovery[measure] = row
    res = dict(rule='protocol §12.7', n_pairs=n, A4p_minus_A0=dict(median=med, n=len(d), positive=pos),
               recovery_descriptive=recovery,
               note='A0 (and A1, A3) from the D0 batch (d0_v1), not concurrent with A4p; development set, single '
                    'observation; A1 and A3 used truth labels')
    if len(d) < n-2:
        res.update(rows=[], conclusion='不可判：A4′ 相对 A0 的有效配对少于 17 对（协议 §12.7）')
        return res
    rows = (['R4a'] if med >= .01 and pos >= 14 else [])+(['R4b'] if med < .005 else [])
    if len(rows) == 1:
        concl = A4_ROW_TEXT[rows[0]]
    elif not rows:
        concl = '未落入 R4a 或 R4b（中间区域）：只描述，不下结论'
    else:
        concl = '多行同时成立（' + '、'.join(rows) + '）：交总负责人判断'
    res.update(rows=rows, conclusion=concl, row_texts={r: A4_ROW_TEXT[r] for r in rows})
    return res


def launch_load(run_dir):
    """(1-min, 5-min, 15-min) load average at the unit launch, from the queue's log.txt line, or None."""
    path = Path(run_dir)/'log.txt'
    if not path.exists():
        return None
    for line in path.read_text(errors='replace').splitlines():
        m = LAUNCH.search(line)
        if m:
            return tuple(float(x) for x in m.groups())
    return None


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def summarize_a4(a4_root, d0_root, out_dir, stage=STAGE):
    """A4p (a4_root) with A0-A3 of the D0 batch (d0_root, metrics.json read only) -> long.csv, summary_a4.md,
    decision_a4.json in out_dir (never under d0_v1/)."""
    out_dir = Path(out_dir)
    if _inside(out_dir, D0):
        raise SystemExit(f'{out_dir}: the A4p summary is never written under {D0}')
    pairs = [r['pair'] for r in json.loads((Path(stage)/'selection.json').read_text())['pairs']]
    runs = chosen_runs(d0_root, ARMS)
    runs.update(chosen_runs(a4_root, SUPPLEMENTARY))
    arms = ARMS+SUPPLEMENTARY
    loads = {k: launch_load(m['run_dir']) for k, m in runs.items()}
    rows = long_rows(runs)
    for (arm, pair), m in sorted(runs.items()):
        if loads[(arm, pair)]:
            for k, v in zip(('launch_loadavg_1min', 'launch_loadavg_5min'), loads[(arm, pair)]):
                rows.append(dict(set=SET, split=SPLIT, method=METHOD, arm=arm, pair=pair, run_id=m['run_id'], measure=k,
                                 value=v))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/'long.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['set', 'split', 'method', 'arm', 'pair', 'measure', 'value', 'run_id'])
        w.writeheader(); w.writerows(rows)
    dec = decide_a4(runs, pairs)
    name = (lambda arm: A4_LABEL if arm == A4_ARM else arm)
    top = (lambda arm, key: [runs[(arm, p)].get(key) for p in pairs if (arm, p) in runs])
    dec['sources'] = dict(a4_runs_root=str(a4_root), d0_runs_root=str(d0_root),
                          evaluator_sha256={arm: sorted({(runs[(arm, p)].get('evaluator') or {}).get('sha256') or '-'
                                                         for p in pairs if (arm, p) in runs}) for arm in arms})
    dec['native_commits_A4p_minus_A0'] = dict(
        median=_median(runs[(A4_ARM, p)]['native_commits']-runs[('A0', p)]['native_commits'] for p in pairs
                       if (A4_ARM, p) in runs and ('A0', p) in runs and runs[(A4_ARM, p)].get('native_commits') is not None
                       and runs[('A0', p)].get('native_commits') is not None))
    dec['launch_loadavg_median'] = {arm: dict(one_min=_median(loads[(arm, p)][0] for p in pairs if loads.get((arm, p))),
                                             five_min=_median(loads[(arm, p)][1] for p in pairs if loads.get((arm, p))))
                                    for arm in arms}
    L = ['# D0 补充臂 A4′ 汇总（开发集结果，单次观测）', '',
         f'生成：{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}（UTC）；协议 docs/bayesreg/05_D0协议.md §12；'
         f'方法 {METHOD}；19 个 Learn2Reg 验证对；指标由 experiments/d0/evaluate.py 从终态场算出。',
         f'A4′ 运行：{a4_root}；A0–A3 取 D0 全批 {d0_root} 的 metrics.json（只读，不重算、不重跑）。', '',
         '**A0 不是同时段对照**：A0 来自 D0 全批（2026-10-30 5:02–10:01 am CDT，期间有后台负载），A4′ 在其后单独运行；'
         '90 s 是墙钟预算，机器状态不同会改变原生提交数。下表的原生提交数与启动时负载只作描述。', '',
         '**评测脚本版本**（A4 复核建议 7）：各臂 metrics.json 记录的 experiments/d0/evaluate.py sha256 为 '
         + '；'.join(f"{name(arm)} {', '.join(s[:12] for s in dec['sources']['evaluator_sha256'][arm]) or '—'}"
                    for arm in arms)
         + '。A0–A3 与 A4′ 的哈希不同：A4′ 的工具（f931cd1 起）在同一文件里加了 A4p 臂名与 summarize-a4，'
         '单次运行的评测函数 evaluate_run 没有改（见 runs/20261030-d0-a4/plan.md）。', '',
         '## 各臂均值（seg4 与 seg35 同时报）', '',
         '| 臂 | 有终态场的对数 | seg35 | seg4 | Dice35 官方 | HD95 | SDlogJ 官方 | 脑内 SDlogJ | 证书通过 | 自检通过 | '
         'e2e 中位 s | 原生提交中位 | 启动负载中位（1/5 分钟） |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in arms:
        ms = [runs[(arm, p)] for p in pairs if (arm, p) in runs]
        ok = [m for m in ms if m.get('metrics')]
        mean = (lambda k: float(np.mean([m['metrics'][k] for m in ok])) if ok else None)
        la = dec['launch_loadavg_median'][arm]
        L.append(f"| {name(arm)} | {len(ok)}/{len(pairs)} | {fmt(mean('seg35'))} | {fmt(mean('seg4'))} | "
                 f"{fmt(mean('dice_official'))} | {fmt(mean('hd95'), 3)} | {fmt(mean('sdlogj_official'))} | "
                 f"{fmt(mean('sdlogj_brain'), 3)} | {sum(bool((m.get('certificate') or {}).get('passed')) for m in ms)}/{len(ms)} | "
                 f"{sum(bool((m.get('self_check') or {}).get('passed')) for m in ms)}/{len(ms)} | "
                 f"{fmt(_median(top(arm, 'e2e_seconds')), 1)} | {fmt(_median(top(arm, 'native_commits')), 1)} | "
                 f"{fmt(la['one_min'], 2)} / {fmt(la['five_min'], 2)} |")
    L += ['', '## 逐对差（前者 − 后者）', '', '| 比较 | 指标 | 中位差 | 均值差 | 为正的对数 / 有效对数 |', '|---|---|---|---|---|']
    for arm, ref in A4_COMPARISONS:
        for k in ('seg35', 'seg4', 'hd95', 'sdlogj_official', 'sdlogj_brain'):
            v = list(paired_ref(runs, arm, ref, k, pairs).values())
            L.append(f"| {name(arm)} − {ref} | {k} | {fmt(_median(v))} | {fmt(float(np.mean(v)) if v else None)} | "
                     f"{sum(x > 0 for x in v)} / {len(v)} |")
    L += ['', f"A4′ − A0 原生提交数逐对中位差：{fmt(dec['native_commits_A4p_minus_A0']['median'], 1)}（描述）。", '',
          '## 逐对 seg35（seg4）', '', '| 对 | A0 | A1 | A3 | A4′ | A4′ − A0 seg35 | 原生提交 A0 / A4′ |', '|---|---|---|---|---|---|---|']
    for p in pairs:
        cells = []
        for arm in ('A0', 'A1', 'A3', A4_ARM):
            m = runs.get((arm, p))
            cells.append('—' if not m or not m.get('metrics') else f"{m['metrics']['seg35']:.4f} ({m['metrics']['seg4']:.4f})")
        dd = paired_ref(runs, A4_ARM, 'A0', 'seg35', [p]).get(p)
        nc = [(runs.get((arm, p)) or {}).get('native_commits') for arm in ('A0', A4_ARM)]
        L.append(f"| {p[11:15]}←{p[-8:-4]} | " + ' | '.join(cells) + f" | {'—' if dd is None else f'{dd:+.4f}'} | "
                 f"{'—' if nc[0] is None else nc[0]} / {'—' if nc[1] is None else nc[1]} |")
    a4 = dec['A4p_minus_A0']
    L += ['', '## 预登记判据（协议 §12.7）', '',
          f"- A4′ − A0 seg35 逐对中位差：{fmt(a4['median'])}（{a4['positive']}/{a4['n']} 对为正；分母 {dec['n_pairs']}）",
          f"- 成立的行：{', '.join(dec['rows']) if dec['rows'] else '无'}", f"- 结论：{dec['conclusion']}", '',
          '## 恢复比例（描述，不进判据）', '', '| 指标 | A4′−A0 中位差 | A1−A0 中位差 | A4′ / A1 | A3−A0 中位差 | A4′ / A3 |',
          '|---|---|---|---|---|---|']
    for k in ('seg35', 'seg4'):
        r = dec['recovery_descriptive'][k]
        L.append(f"| {k} | {fmt(r['A4p_minus_A0_median'])} | {fmt(r['A1_minus_A0_median'])} | {fmt(r['recovery_vs_A1'], 2)} | "
                 f"{fmt(r['A3_minus_A0_median'])} | {fmt(r['recovery_vs_A3'], 2)} |")
    L += ['', '注意：开发集上的单次观测；A0 非同时段（见上）；A1、A3 用了真值标签（oracle），只作参照；A4′ 与 A3 同一公式，'
          '只是 δ 的白质平均改用分割器自己的大脑白质（协议 §12.3）；指标由评测代码从位移场算出。']
    (out_dir/'summary_a4.md').write_text('\n'.join(L)+'\n')
    (out_dir/'decision_a4.json').write_text(json.dumps(dec, indent=1, ensure_ascii=False)+'\n')
    print('\n'.join(L))
    return 0


# ---- arms A1Ph, A1Ps (runs/20261102-a1p/plan.md) ----------------------------------------------------------------------------
A1P_ROUTE_THRESHOLD = .845                # plan, rule R-route: mean seg35 of an arm over the 19 pairs
A1P_MIN_VALID = 17                        # as protocol §7: fewer valid pairs: not decidable
A1P_REFERENCES = ('A0', 'A4p', 'A1')      # the plan's paired differences
A1P_NAMES = {'A4p': 'A4′'}
SEG4_STRUCTURES = {1: '皮层', 2: '皮层下灰质', 3: '白质', 4: 'CSF'}    # /mnt/d/data/seg4_labels.txt, checked (check_seg4_map.py)
SEG35_LARGE = {2: '左大脑皮层', 21: '右大脑皮层', 1: '左大脑白质', 20: '右大脑白质', 5: '左小脑白质', 24: '右小脑白质'}    # gap decomp, §2


def a1p_name(arm):
    return A1P_NAMES.get(arm, arm)


def per_label(run, measure, label):
    """Dice of one label in a metrics.json record (metrics.<measure>_per_label; None when absent), JSON keys are strings."""
    values = ((run or {}).get('metrics') or {}).get(f'{measure}_per_label') or {}
    return values.get(str(label), values.get(label))


def label_rows(runs, arm, ref, measure, labels, pairs):
    """{label: dict(arm_mean, ref_mean, mean_diff, median_diff, positive, n)} over the pairs with both values."""
    out = {}
    for label in labels:
        both = [(per_label(runs.get((arm, p)), measure, label), per_label(runs.get((ref, p)), measure, label)) for p in pairs]
        both = [(a, b) for a, b in both if a is not None and b is not None]
        diffs = [a-b for a, b in both]
        out[label] = dict(arm_mean=float(np.mean([a for a, _ in both])) if both else None,
                          ref_mean=float(np.mean([b for _, b in both])) if both else None,
                          mean_diff=float(np.mean(diffs)) if diffs else None, median_diff=_median(diffs),
                          positive=sum(d > 0 for d in diffs), n=len(diffs))
    return out


def seg35_contributions(runs, arm, ref, pairs):
    """The mean difference of macro seg35 split as in the gap decomposition (runs/20261102-gap-decomp/result.md §2): the six
    large structures and the other 29, each the sum over its labels of the mean per-label difference / 35.  A label absent
    from a pair leaves that pair out of the label's mean, so the two parts add up to the macro difference only approximately."""
    rows = label_rows(runs, arm, ref, 'seg35', range(1, 36), pairs)
    large = sum(rows[k]['mean_diff'] or 0. for k in SEG35_LARGE)/35
    rest = sum(rows[k]['mean_diff'] or 0. for k in rows if k not in SEG35_LARGE)/35
    return dict(large_six=large, other_29=rest, n_other_above=sum((rows[k]['mean_diff'] or 0.) > 0 for k in rows if k not in SEG35_LARGE))


def decide_a1p(runs, pairs):
    """The plan's rule. R-route, per arm: mean seg35 >= .845 over the pairs with a final map, at least 17 such pairs, and every
    certificate of the arm passed -> that arm opens the simple route (predicted-label colouring + frozen solver), reported in
    parallel with bayesreg; both arms: the lead chooses. Below: no route. Fewer than 17 valid pairs: not decidable. The rule
    answers only how much of the oracle gain predicted labels recover; it enters no path or objective reading (GDR M7).
    Recovery (descriptive): (arm - A0) / (A1 - A0), of the means over the pairs that all three have and of the paired medians,
    seg35 and seg4."""
    arms, recovery = {}, {}
    for arm in A1P_ARMS:
        ms = [runs[(arm, p)] for p in pairs if (arm, p) in runs]
        ok = [m for m in ms if m.get('metrics')]
        mean = float(np.mean([m['metrics']['seg35'] for m in ok])) if ok else None
        mean4 = float(np.mean([m['metrics']['seg4'] for m in ok])) if ok else None
        certified = sum(bool((m.get('certificate') or {}).get('passed')) for m in ok)
        valid = len(ok) >= A1P_MIN_VALID
        route = bool(valid and mean >= A1P_ROUTE_THRESHOLD and certified == len(ok))
        arms[arm] = dict(n_with_field=len(ok), n_certified=certified, seg35_mean=mean, seg4_mean=mean4, valid=valid, route=route)
        recovery[arm] = {}
        for measure in ('seg35', 'seg4'):
            ma, m1 = (_median(paired_ref(runs, x, 'A0', measure, pairs).values()) for x in (arm, 'A1'))
            da, d1 = (paired_ref(runs, x, 'A0', measure, pairs) for x in (arm, 'A1'))
            both = sorted(set(da) & set(d1))
            mean_a, mean_1 = (float(np.mean([d[p] for p in both])) if both else None for d in (da, d1))
            recovery[arm][measure] = dict(arm_minus_A0_median=ma, A1_minus_A0_median=m1,
                                          recovery_vs_A1=None if ma is None or not m1 else ma/m1,
                                          arm_minus_A0_mean=mean_a, A1_minus_A0_mean=mean_1,
                                          recovery_mean_vs_A1=None if mean_a is None or not mean_1 else mean_a/mean_1)
    routes = [a for a in A1P_ARMS if arms[a]['route']]
    if not all(arms[a]['valid'] for a in A1P_ARMS):
        concl = '至少一臂的有效配对少于 17 对：该臂不可判（计划 R-route）'
        if routes:
            concl += f"；{'、'.join(routes)} 达到门槛"
    elif routes:
        concl = (f"{'、'.join(routes)} 的 seg35 均值 ≥ {A1P_ROUTE_THRESHOLD}，证书全部通过：另开一条简单路线（预测标签上色 + 现成求解器），"
                 '与 bayesreg 并行报告；两臂都达到时由总负责人选')
    else:
        concl = f'两臂 seg35 均值都低于 {A1P_ROUTE_THRESHOLD}，或有证书未通过：不开这条路线（计划 R-route）'
    return dict(rule='runs/20261102-a1p/plan.md R-route', threshold=A1P_ROUTE_THRESHOLD, n_pairs=len(pairs), arms=arms,
                routes=routes, recovery_descriptive=recovery, conclusion=concl,
                note='A0, A1 from the D0 batch (d0_v1) and A4p from d0_v1_a4, not concurrent with A1P; development set, single '
                     'observation; A1 used truth labels (oracle), A1P and A4p the trained segmenter')


def summarize_a1p(a1p_root, d0_root, a4_root, out_dir, stage=STAGE):
    """A1Ph, A1Ps (a1p_root) with A0-A3 of the D0 batch and A4p (metrics.json read only) -> long.csv, summary_a1p.md,
    decision_a1p.json in out_dir (never under d0_v1/ or d0_v1_a4/)."""
    out_dir = Path(out_dir)
    if _inside(out_dir, D0) or _inside(out_dir, D0A4):
        raise SystemExit(f'{out_dir}: the A1P summary is never written under {D0} or {D0A4}')
    pairs = [r['pair'] for r in json.loads((Path(stage)/'selection.json').read_text())['pairs']]
    runs = chosen_runs(d0_root, ARMS)
    runs.update(chosen_runs(a4_root, SUPPLEMENTARY))
    runs.update(chosen_runs(a1p_root, A1P_ARMS))
    arms = ('A0', 'A1', 'A4p')+A1P_ARMS
    runs = {k: m for k, m in runs.items() if k[0] in arms}
    loads = {k: launch_load(m['run_dir']) for k, m in runs.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/'long.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['set', 'split', 'method', 'arm', 'pair', 'measure', 'value', 'run_id'])
        w.writeheader(); w.writerows(long_rows(runs))
    dec = decide_a1p(runs, pairs)
    dec['sources'] = dict(a1p_runs_root=str(a1p_root), d0_runs_root=str(d0_root), a4_runs_root=str(a4_root),
                          evaluator_sha256={arm: sorted({(runs[(arm, p)].get('evaluator') or {}).get('sha256') or '-'
                                                         for p in pairs if (arm, p) in runs}) for arm in arms})
    dec['launch_loadavg_median'] = {arm: dict(one_min=_median(loads[(arm, p)][0] for p in pairs if loads.get((arm, p))),
                                             five_min=_median(loads[(arm, p)][1] for p in pairs if loads.get((arm, p))))
                                    for arm in arms}
    L = ['# A1P 汇总（开发集结果，单次观测）', '',
         f'生成：{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}（UTC）；计划 runs/20261102-a1p/plan.md；方法 {METHOD}；'
         '19 个 Learn2Reg 验证对；指标由 experiments/d0/evaluate.py 从终态场算出。',
         f'A1P 运行：{a1p_root}；A0、A1 取 {d0_root}，A4′ 取 {a4_root} 的 metrics.json（只读，不重算、不重跑）。', '',
         '**参照不是同时段对照**：A0、A1 来自 D0 全批（2026-10-30，期间有后台负载），A4′ 在其后单独运行；90 s 是墙钟预算，机器状态不同会改变'
         '原生提交数。下表的原生提交数与启动时负载只作描述。A1 用了真值标签（oracle），只作参照；A1Ph、A1Ps、A4′ 用训练集标签训练的分割器。', '',
         '## 各臂均值（seg4 与 seg35 同时报）', '',
         '| 臂 | 有终态场的对数 | seg35 | seg4 | Dice35 官方 | HD95 | SDlogJ 官方 | 脑内 SDlogJ | 证书通过 | 自检通过 | e2e 中位 s | '
         '原生提交中位 | 启动负载中位（1/5 分钟） |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in arms:
        ms = [runs[(arm, p)] for p in pairs if (arm, p) in runs]
        ok = [m for m in ms if m.get('metrics')]
        mean = (lambda k: float(np.mean([m['metrics'][k] for m in ok])) if ok else None)
        top = (lambda key: [m.get(key) for m in ms])
        la = dec['launch_loadavg_median'][arm]
        L.append(f"| {a1p_name(arm)} | {len(ok)}/{len(pairs)} | {fmt(mean('seg35'))} | {fmt(mean('seg4'))} | "
                 f"{fmt(mean('dice_official'))} | {fmt(mean('hd95'), 3)} | {fmt(mean('sdlogj_official'))} | "
                 f"{fmt(mean('sdlogj_brain'), 3)} | {sum(bool((m.get('certificate') or {}).get('passed')) for m in ms)}/{len(ms)} | "
                 f"{sum(bool((m.get('self_check') or {}).get('passed')) for m in ms)}/{len(ms)} | "
                 f"{fmt(_median(top('e2e_seconds')), 1)} | {fmt(_median(top('native_commits')), 1)} | "
                 f"{fmt(la['one_min'], 2)} / {fmt(la['five_min'], 2)} |")
    L += ['', '## 逐对差（前者 − 后者）', '', '| 比较 | 指标 | 中位差 | 均值差 | 为正的对数 / 有效对数 |', '|---|---|---|---|---|']
    for arm in A1P_ARMS:
        for ref in A1P_REFERENCES:
            for k in ('seg35', 'seg4', 'hd95', 'sdlogj_official', 'sdlogj_brain'):
                v = list(paired_ref(runs, arm, ref, k, pairs).values())
                L.append(f"| {arm} − {a1p_name(ref)} | {k} | {fmt(_median(v))} | {fmt(float(np.mean(v)) if v else None)} | "
                         f"{sum(x > 0 for x in v)} / {len(v)} |")
    dec['per_structure'] = {}
    L += ['', '## seg4 逐结构（19 对均值）', '', '| seg4 标签 | ' + ' | '.join(a1p_name(a) for a in arms) + ' |',
          '|---|' + '---|'*len(arms)]
    table = {arm: label_rows(runs, arm, 'A0', 'seg4', SEG4_STRUCTURES, pairs) for arm in arms}
    for label, name in SEG4_STRUCTURES.items():
        L.append(f'| {label} {name} | ' + ' | '.join(fmt(table[a][label]['arm_mean']) for a in arms) + ' |')
    L += ['', '## seg35 六个大结构（19 对均值）', '', '| seg35 标签 | ' + ' | '.join(a1p_name(a) for a in arms) + ' |',
          '|---|' + '---|'*len(arms)]
    large = {arm: label_rows(runs, arm, 'A0', 'seg35', SEG35_LARGE, pairs) for arm in arms}
    for label, name in SEG35_LARGE.items():
        L.append(f'| {label} {name} | ' + ' | '.join(fmt(large[a][label]['arm_mean']) for a in arms) + ' |')
    L += ['', '## 皮层与白质相对各参照的逐对差', '', '| 比较 | 结构 | 中位差 | 均值差 | 为正的对数 / 有效对数 |', '|---|---|---|---|---|']
    for arm in A1P_ARMS:
        dec['per_structure'][arm] = {}
        for ref in A1P_REFERENCES:
            seg4_rows = label_rows(runs, arm, ref, 'seg4', (1, 3), pairs)
            seg35_rows = label_rows(runs, arm, ref, 'seg35', SEG35_LARGE, pairs)
            dec['per_structure'][arm][ref] = dict(seg4=seg4_rows, seg35=seg35_rows)
            for label, row in seg4_rows.items():
                L.append(f"| {arm} − {a1p_name(ref)} | seg4 {SEG4_STRUCTURES[label]} | {fmt(row['median_diff'])} | "
                         f"{fmt(row['mean_diff'])} | {row['positive']} / {row['n']} |")
            for label, row in seg35_rows.items():
                L.append(f"| {arm} − {a1p_name(ref)} | seg35 {SEG35_LARGE[label]} | {fmt(row['median_diff'])} | "
                         f"{fmt(row['mean_diff'])} | {row['positive']} / {row['n']} |")
    L += ['', '## seg35 均值差的拆分（相对 A0；六个大结构与其余 29 个，每项为标签均值差之和 / 35）', '',
          '| 臂 | 六个大结构 | 其余 29 个 | 其余 29 个里高于 A0 的结构数 |', '|---|---|---|---|']
    dec['seg35_contributions_vs_A0'] = {}
    for arm in ('A1', 'A4p')+A1P_ARMS:
        c = seg35_contributions(runs, arm, 'A0', pairs)
        dec['seg35_contributions_vs_A0'][arm] = c
        L.append(f"| {a1p_name(arm)} | {fmt(c['large_six'])} | {fmt(c['other_29'])} | {c['n_other_above']} |")
    L += ['', '## 逐对 seg35（seg4）', '', '| 对 | ' + ' | '.join(a1p_name(a) for a in arms) + ' |', '|---|' + '---|'*len(arms)]
    for p in pairs:
        cells = []
        for arm in arms:
            m = runs.get((arm, p))
            cells.append('—' if not m or not m.get('metrics') else f"{m['metrics']['seg35']:.4f} ({m['metrics']['seg4']:.4f})")
        L.append(f"| {p[11:15]}←{p[-8:-4]} | " + ' | '.join(cells) + ' |')
    L += ['', '## 预登记判据（计划 R-route）', '']
    for arm in A1P_ARMS:
        r = dec['arms'][arm]
        L.append(f"- {arm}：seg35 均值 {fmt(r['seg35_mean'])}（门槛 {A1P_ROUTE_THRESHOLD}），seg4 均值 {fmt(r['seg4_mean'])}，"
                 f"有终态场 {r['n_with_field']}/{len(pairs)}，证书通过 {r['n_certified']}。")
    L += [f"- 结论：{dec['conclusion']}", '- 这条判据只回答预测标签能收回多少 oracle 增益，不进路径或目标函数的判定（GDR M7）。', '',
          '## 收回的 oracle 增益（描述，不进判据）', '',
          '| 臂 | 指标 | 臂−A0 均值差 | A1−A0 均值差 | 均值比 | 臂−A0 中位差 | A1−A0 中位差 | 中位比 |', '|---|---|---|---|---|---|---|---|']
    for arm in A1P_ARMS:
        for k in ('seg35', 'seg4'):
            r = dec['recovery_descriptive'][arm][k]
            L.append(f"| {arm} | {k} | {fmt(r['arm_minus_A0_mean'])} | {fmt(r['A1_minus_A0_mean'])} | {fmt(r['recovery_mean_vs_A1'], 2)} | "
                     f"{fmt(r['arm_minus_A0_median'])} | {fmt(r['A1_minus_A0_median'])} | {fmt(r['recovery_vs_A1'], 2)} |")
    L += ['', '**评测脚本版本**：各臂 metrics.json 记录的 experiments/d0/evaluate.py sha256 为 '
          + '；'.join(f"{a1p_name(arm)} {', '.join(s[:12] for s in dec['sources']['evaluator_sha256'][arm]) or '—'}" for arm in arms) + '。',
          '', '注意：开发集上的单次观测；参照非同时段（见上）；指标由评测代码从位移场算出；预测标签来自训练集标签训练的分割器，真值标签只用于评测。']
    (out_dir/'summary_a1p.md').write_text('\n'.join(L)+'\n')
    (out_dir/'decision_a1p.json').write_text(json.dumps(dec, indent=1, ensure_ascii=False)+'\n')
    print('\n'.join(L))
    return 0


# ---- gate A, X1 (runs/20261102-gate-a/plan.md, revision 1) ---------------------------------------------------------------------
X1_N_PAIRS = 5
X1_MIN_N = 4                              # comparable pairs (of 5) for a reading; fewer: describe only
X1_PINS = dict(A0=(.8062, .8702), P0n=(.7886, .7825))     # the plan's (seg35, seg4) means over the 5 pairs, to 4 decimals
X1_MIDPOINT = .826                        # the plan: (A0 + P0n)/2 of seg4 over the 5 pairs, rounded to 3 decimals
X1_NEAR_A0 = .01                          # the plan: seg4 >= A0 - .01, rounded to 4 decimals (.8602 over the 5 pairs)
X1_NEAR_A0_5 = .8602
X1_NAMES = dict(X1h='X1h（λ、μ、ν 同乘 .5：绝对 .015）', X1f='X1f（×1：绝对 .03）')
X1_READINGS = dict(explains='这一档绝对权重（≤ 臂的权重）从 T88 种子出发 90 s，足以把 seg4 压过差距的一半：绝对权重至少能解释差距的一半',
                   elsewhere='这一档绝对权重在 90 s 内没有明显压低 seg4；不覆盖 ν 21.9 的 curl .0995（高于 X1 的上限 .03），不说明差距来自别处',
                   middle='其余：只描述', few='可比对数不足：只描述')
X1_NOTE = ('开发集前 5 对，单次观测。n = 5 的最小可检测效应约 .009–.020，只有方向信息；读数不是结论。A0（D0 全批，2026-10-30）与 P0n'
           '（PILOT，2026-10-31）都不是同时段对照。')
X1_SEMANTICS = ('X1f 是绝对 μ = ν = .03；X1h 是 λ、μ、ν 同乘 .5（.015，度量也跟着缩放）。这两个权重配合的是 E（参考 u = 0，罚 (J − 1)² 与 |curl|²），不是 F = D + B；'
                '接受只看 E 的认证下降，F 只记录。终态是从 T88 种子出发、在 E 上下降、90 s 截断的状态，不是 E 或 F 的驻点；提前结束的运行见轨迹表的收尾列（outer_no_progress、outer_stagnated）。P0n 的 curl 权 .0995 高于 X1 的上限 .03；'
                'X1h 的旁证下锚点是 H2（0438：seg4 .8383，单对，2700 s）。')


def pilot_metrics(pilot_root, pair):
    """The P0n record of one pair (bayesreg.eval metrics.json of the path pilot), None when absent; per-label Dice moved next
    to the scalars as in the D0 records."""
    short = re.sub(r'OASIS_OAS1_([0-9]+)_MR1__OASIS_OAS1_([0-9]+)_MR1', r'\1-\2', pair)
    doc = rd(Path(pilot_root)/'eval'/'p0n'/short/f'pilot_p0n_{short}'/'metrics.json')
    if doc is None or not doc.get('metrics'):
        return None
    labels = doc.get('per_label') or {}
    return dict(doc, metrics=dict(doc['metrics'], seg4_per_label=labels.get('seg4_per_label'),
                                  seg35_per_label=labels.get('seg35_per_label')))


def x1_comparable(run):
    """A run the X1 reading may use: a final map, its certificate passed, and a formal native state -- at least one native commit for a
    frozen run (a handover_only state is the seed, not the switch; review M4), field_role native_formal for a bayesreg.eval record (P0n)."""
    if not run or not run.get('metrics') or not (run.get('certificate') or {}).get('passed'):
        return False
    if 'field_role' in run:
        return run['field_role'] == 'native_formal'
    commits = run.get('native_commits')
    return isinstance(commits, int) and commits > 0


def x1_trajectory(run_dir):
    """Readouts of a frozen run from its small files.  Only the accepted rows of history.jsonl are commits (sequential.py appends a row
    before the decision; review M1).  E, F: at the seed (the first row's start) and at the last commit.  MSE ratio: at the seed (D is
    proportional to the MSE: D_start of the first row times the first commit's ratio / D), after the first commit, at the last commit
    (review M2).  F_certified: commits whose F decrease was certified too, of those that record it (T92: a commit needs only E's)."""
    run_dir = Path(run_dir)
    proto = rd(run_dir/'protocol.json') or {}
    task = run_dir/((proto.get('task') or {}).get('name') or 'missing_task_name')
    out = dict(native_reference=proto.get('native_reference'), native_penalty_scale=proto.get('native_penalty_scale'))
    if not (task/'history.jsonl').is_file():
        return out
    every = [r for r in map(json.loads, open(task/'history.jsonl')) if r.get('level') == 3]
    rows = [r for r in every if r.get('outer_accepted')]
    summary = rd(task/'summary.json') or {}
    out.update(native_status=summary.get('native_status'), status=summary.get('status'), reason=summary.get('reason'),
               energy=summary.get('energy'), commits=len(rows), rows_not_accepted=len(every)-len(rows))
    if every:
        out.update(E_first=every[0].get('E_start'), F_first=every[0].get('F_start'))
    if rows:
        first, last = rows[0], rows[-1]
        seed_ratio = None
        if every[0].get('D_start') is not None and first.get('D_candidate') and first.get('committed_mse_ratio') is not None:
            seed_ratio = every[0]['D_start']*first['committed_mse_ratio']/first['D_candidate']
        recorded = [r for r in rows if 'F_decrease_certified' in r]
        out.update(E_last=last.get('committed_E'), F_last=last.get('committed_F'), ratio_seed=seed_ratio,
                   ratio_first=first.get('committed_mse_ratio'), ratio_last=last.get('committed_mse_ratio'),
                   inner_accepted=sum(int(r.get('inner_accepted_steps') or 0) for r in rows),
                   F_certified=sum(r['F_decrease_certified'] is True for r in recorded), F_recorded=len(recorded))
    return out


def _x1_mean(runs, arm, pairs, key):
    values = [runs[(arm, p)]['metrics'][key] for p in pairs if (arm, p) in runs]
    return float(np.mean(values)) if values else None


def decide_x1(runs, pairs):
    """The plan's reading (revision 1, review I2, I4, M4), per arm on seg4 over the arm's comparable pairs (x1_comparable).  A0 and P0n are
    averaged over the SAME pairs (a pair needs all three comparable); midpoint = (A0 + P0n)/2 rounded to 3 decimals, near = A0 - .01
    rounded to 4 decimals, which over the 5 pairs are the plan's .826 and .8602.  seg4 <= midpoint: explains; seg4 >= near: elsewhere;
    otherwise described only; fewer than 4 such pairs: described only.  seg35, the seed and the structures are reported, not in the
    reading.  The pinned means of A0 and P0n and the two thresholds over the 5 pairs are re-checked; a mismatch is reported first."""
    good = {k: m for k, m in runs.items() if x1_comparable(m)}
    means = {arm: dict(n=sum((arm, p) in good for p in pairs), seg35=_x1_mean(good, arm, pairs, 'seg35'),
                       seg4=_x1_mean(good, arm, pairs, 'seg4')) for arm in ('A0', 'P0n')+X1_ARMS}
    pin_problems = [f"{arm}: 预登记 {pin}，重算 ({fmt(means[arm]['seg35'])}, {fmt(means[arm]['seg4'])})" for arm, pin in X1_PINS.items()
                    if means[arm]['seg35'] is None or abs(means[arm]['seg35']-pin[0]) > 5.01e-5 or abs(means[arm]['seg4']-pin[1]) > 5.01e-5]
    arms = {}
    for arm in X1_ARMS:
        same = [p for p in pairs if all((a, p) in good for a in (arm, 'A0', 'P0n'))]
        seg4, a0, p0 = (_x1_mean(good, a, same, 'seg4') for a in (arm, 'A0', 'P0n'))
        midpoint = None if not same else round((a0+p0)/2, 3)
        near = None if not same else round(a0-X1_NEAR_A0, 4)
        if len(same) < X1_MIN_N:
            key = 'few'
        elif seg4 <= midpoint:
            key = 'explains'
        elif seg4 >= near:
            key = 'elsewhere'
        else:
            key = 'middle'
        arms[arm] = dict(n=len(same), pairs=same, seg4=seg4, seg35=_x1_mean(good, arm, same, 'seg35'), A0_same_pairs=a0, P0n_same_pairs=p0,
                         midpoint=midpoint, near_A0=near, reading=key, reading_text=X1_READINGS[key])
    five = [p for p in pairs if ('A0', p) in good and ('P0n', p) in good]
    if len(five) == X1_N_PAIRS:
        a0, p0 = _x1_mean(good, 'A0', five, 'seg4'), _x1_mean(good, 'P0n', five, 'seg4')
        if round((a0+p0)/2, 3) != X1_MIDPOINT or round(a0-X1_NEAR_A0, 4) != X1_NEAR_A0_5:
            pin_problems.append(f"门槛：预登记 {X1_MIDPOINT}、{X1_NEAR_A0_5}，重算 {round((a0+p0)/2, 3)}、{round(a0-X1_NEAR_A0, 4)}")
    return dict(rule='runs/20261102-gate-a/plan.md X1, revision 1', midpoint_5_pairs=X1_MIDPOINT, near_A0_5_pairs=X1_NEAR_A0_5, pins=X1_PINS,
                pin_problems=pin_problems, means=means, arms=arms, note=X1_NOTE, semantics=X1_SEMANTICS)


def x1_seed_summary(runs, good, pairs, dec):
    """The seed of each arm (metrics.json seed; descriptive only, review I3): mean Dice over the pairs where it was evaluated, the distance
    from the two thresholds, and arm - seed over the pairs where the final map is comparable too."""
    out = {}
    for arm in X1_ARMS:
        seeds = {p: dict(metrics=runs[(arm, p)]['seed']) for p in pairs if (arm, p) in runs and (runs[(arm, p)].get('seed') or {}).get('present')}
        if not seeds:
            out[arm] = dict(n=0, note='no seed record')
            continue
        record = dict(n=len(seeds), per_pair={p: dict(seg35=s['metrics']['seg35'], seg4=s['metrics']['seg4']) for p, s in seeds.items()})
        for key in ('seg35', 'seg4'):
            record[key] = float(np.mean([s['metrics'][key] for s in seeds.values()]))
        for name, label in (('cortex', 1), ('white_matter', 3)):
            values = [per_label(s, 'seg4', label) for s in seeds.values() if per_label(s, 'seg4', label) is not None]
            record[name] = float(np.mean(values)) if values else None
        both = [p for p in seeds if (arm, p) in good]
        diffs = {}
        for key in ('seg35', 'seg4'):
            d = [good[(arm, p)]['metrics'][key]-seeds[p]['metrics'][key] for p in both]
            diffs[key] = dict(n=len(d), mean=float(np.mean(d)) if d else None, median=_median(d), positive=sum(x > 0 for x in d))
        for name, label in (('cortex', 1), ('white_matter', 3)):
            d = [per_label(good[(arm, p)], 'seg4', label)-per_label(seeds[p], 'seg4', label) for p in both
                 if per_label(good[(arm, p)], 'seg4', label) is not None and per_label(seeds[p], 'seg4', label) is not None]
            diffs[name] = dict(n=len(d), mean=float(np.mean(d)) if d else None, median=_median(d), positive=sum(x > 0 for x in d))
        record['arm_minus_seed'] = diffs
        r = dec['arms'][arm]
        record['seed_seg4_minus_midpoint'] = None if r['midpoint'] is None else record['seg4']-r['midpoint']
        record['near_A0_minus_seed_seg4'] = None if r['near_A0'] is None else r['near_A0']-record['seg4']
        out[arm] = record
    return out


def summarize_x1(x1_root, d0_root, pilot_root, out_dir, stage=STAGE):
    """X1h, X1f (x1_root) with A0 (d0_root) and P0n (pilot_root), metrics.json read only -> long.csv, summary_x1.md,
    decision_x1.json in out_dir (never under d0_v1/ or the pilot's root).  Means and paired differences are over comparable runs only
    (review M3); the reading's thresholds over the arm's own pairs (decide_x1)."""
    out_dir = Path(out_dir)
    if _inside(out_dir, D0) or _inside(out_dir, pilot_root):
        raise SystemExit(f'{out_dir}: the X1 summary is never written under {D0} or {pilot_root}')
    pairs = [r['pair'] for r in json.loads((Path(stage)/'selection.json').read_text())['pairs']][:X1_N_PAIRS]
    runs = {k: m for k, m in chosen_runs(d0_root, ('A0',)).items() if k[1] in pairs}
    runs.update({k: m for k, m in chosen_runs(x1_root, X1_ARMS).items() if k[1] in pairs})
    frozen = {k: m for k, m in runs.items()}
    for p in pairs:
        record = pilot_metrics(pilot_root, p)
        if record is not None:
            runs[('P0n', p)] = record
    good = {k: m for k, m in runs.items() if x1_comparable(m)}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/'long.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['set', 'split', 'method', 'arm', 'pair', 'measure', 'value', 'run_id'])
        w.writeheader(); w.writerows(long_rows(frozen))
    dec = decide_x1(runs, pairs)
    arms = ('A0', 'P0n')+X1_ARMS
    name = (lambda a: X1_NAMES.get(a, a))
    loads = {k: launch_load(m['run_dir']) for k, m in frozen.items()}
    dec['sources'] = dict(x1_runs_root=str(x1_root), d0_runs_root=str(d0_root), pilot_root=str(pilot_root), pairs=pairs)
    dec['trajectories'] = {f'{a}/{p}': x1_trajectory(frozen[(a, p)]['run_dir']) for a in X1_ARMS for p in pairs if (a, p) in frozen}
    dec['seed'] = x1_seed_summary(runs, good, pairs, dec)
    dec['launch_loadavg'] = {f'{a}/{p}': loads[(a, p)] for a, p in loads}
    dec['seg35_contributions'] = {arm: {ref: seg35_contributions(good, arm, ref, pairs) for ref in ('A0', 'P0n')} for arm in X1_ARMS}
    L = ['# 关口 A X1 汇总（开发集前 5 对，单次观测）', '',
         f'生成：{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}（UTC）；计划 runs/20261102-gate-a/plan.md（修订 1）；冻结求解器加 '
         '`--native-reference identity`，其余与配置 C 相同；指标由 experiments/d0/evaluate.py 从终态场算出（P0n 由 bayesreg.eval）。', '',
         f'**{X1_NOTE}**', '', X1_SEMANTICS, '']
    if dec['pin_problems']:
        L += ['**基线与预登记不符，读数要交总负责人**：' + '；'.join(dec['pin_problems']), '']
    L += ['## 各臂均值（seg4 与 seg35 同时报；可比的运行）', '',
          '| 臂 | 可比对数 | seg35 | seg4 | 皮层（seg4 标签 1） | 白质（seg4 标签 3） | HD95 | SDlogJ 官方 | 证书通过 | 启动负载中位（1/5 分钟） |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for arm in arms:
        ms = [good[(arm, p)] for p in pairs if (arm, p) in good]
        every = [runs[(arm, p)] for p in pairs if (arm, p) in runs]
        mean = (lambda k: float(np.mean([m['metrics'][k] for m in ms])) if ms else None)

        def lab(label):
            values = [per_label(m, 'seg4', label) for m in ms if per_label(m, 'seg4', label) is not None]
            return float(np.mean(values)) if values else None
        la = [loads[(arm, p)] for p in pairs if loads.get((arm, p))]
        L.append(f"| {name(arm)} | {len(ms)}/{len(pairs)} | {fmt(mean('seg35'))} | {fmt(mean('seg4'))} | {fmt(lab(1))} | {fmt(lab(3))} | "
                 f"{fmt(mean('hd95'), 3)} | {fmt(mean('sdlogj_official'))} | "
                 f"{sum(bool((m.get('certificate') or {}).get('passed')) for m in every)}/{len(every)} | "
                 f"{fmt(_median(x[0] for x in la), 2)} / {fmt(_median(x[1] for x in la), 2)} |")
    for arm in X1_ARMS:
        s = dec['seed'][arm]
        if s['n']:
            L.append(f"| {arm} 的种子（T88 交接态；只作描述） | {s['n']}/{len(pairs)} | {fmt(s['seg35'])} | {fmt(s['seg4'])} | {fmt(s['cortex'])} | "
                     f"{fmt(s['white_matter'])} | — | — | — | — |")
    L += ['', '## 逐对差（前者 − 后者；可比的运行）', '', '| 比较 | 指标 | 中位差 | 均值差 | 为正的对数 / 有效对数 |', '|---|---|---|---|---|']
    for arm in X1_ARMS:
        for ref in ('A0', 'P0n'):
            for k in ('seg35', 'seg4'):
                v = list(paired_ref(good, arm, ref, k, pairs).values())
                L.append(f"| {arm} − {ref} | {k} | {fmt(_median(v))} | {fmt(float(np.mean(v)) if v else None)} | {sum(x > 0 for x in v)} / {len(v)} |")
            for label, text in ((1, '皮层'), (3, '白质')):
                row = label_rows(good, arm, ref, 'seg4', (label,), pairs)[label]
                L.append(f"| {arm} − {ref} | seg4 {text} | {fmt(row['median_diff'])} | {fmt(row['mean_diff'])} | {row['positive']} / {row['n']} |")
    L += ['', '## 种子（只作描述，不进读数）', '', '| 臂 | X1 − 种子 seg35 | X1 − 种子 seg4 | 皮层 | 白质 | 种子 seg4 − 中点 | A0 − .01 − 种子 seg4 |', '|---|---|---|---|---|---|---|']
    for arm in X1_ARMS:
        s = dec['seed'][arm]
        if not s['n']:
            L.append(f"| {arm} | 没有种子记录 | — | — | — | — | — |")
            continue
        d = s['arm_minus_seed']
        cell = (lambda x: f"{fmt(x['mean'])}（{x['positive']}/{x['n']} 为正）")
        L.append(f"| {arm} | {cell(d['seg35'])} | {cell(d['seg4'])} | {cell(d['cortex'])} | {cell(d['white_matter'])} | "
                 f"{fmt(s['seed_seg4_minus_midpoint'])} | {fmt(s['near_A0_minus_seed_seg4'])} |")
    L += ['', '种子本身已经低于 A0 − .01 时，X1 留在种子附近就落进中间带；种子离中点很近时，略微下降就会过中点。读数要和 X1 − 种子一起看。']
    L += ['', '## seg35 六个大结构与其余 29 个（可比的运行）', '', '| seg35 标签 | ' + ' | '.join(name(a) for a in arms) + ' |', '|---|' + '---|'*len(arms)]
    for label, text in SEG35_LARGE.items():
        cells = []
        for arm in arms:
            values = [per_label(good[(arm, p)], 'seg35', label) for p in pairs if (arm, p) in good and per_label(good[(arm, p)], 'seg35', label) is not None]
            cells.append(fmt(float(np.mean(values)) if values else None))
        L.append(f'| {label} {text} | ' + ' | '.join(cells) + ' |')
    L += ['', '| 比较 | 六个大结构（标签均值差之和 / 35） | 其余 29 个 | 其余 29 个里为正的结构数 |', '|---|---|---|---|']
    for arm in X1_ARMS:
        for ref in ('A0', 'P0n'):
            c = dec['seg35_contributions'][arm][ref]
            L.append(f"| {arm} − {ref} | {fmt(c['large_six'])} | {fmt(c['other_29'])} | {c['n_other_above']} |")
    L += ['', '## 逐对 seg35（seg4）', '', '| 对 | ' + ' | '.join(name(a) for a in arms) + ' |', '|---|' + '---|'*len(arms)]
    for p in pairs:
        cells = []
        for arm in arms:
            m = runs.get((arm, p))
            cells.append('—' if not m or not m.get('metrics') else f"{m['metrics']['seg35']:.4f} ({m['metrics']['seg4']:.4f})")
        L.append(f"| {p[11:15]}←{p[-8:-4]} | " + ' | '.join(cells) + ' |')
    L += ['', '## 运行轨迹（history.jsonl 的已接受行；T92 下提交只要求 E 的认证下降，F 只记录）', '',
          '| 运行 | 收尾 | 原生提交 | 内层接受步 | F 也有下降证书的提交 | E 种子 → 末 | F 种子 → 末 | MSE 比 种子 → 第一次提交 → 末 |', '|---|---|---|---|---|---|---|---|']
    for key, t in dec['trajectories'].items():
        L.append(f"| {key.split('/')[0]} {key[-29:-25]}←{key[-8:-4]} | {t.get('native_status')} | {t.get('commits')} | {t.get('inner_accepted')} | "
                 f"{t.get('F_certified')}/{t.get('F_recorded')} | {fmt(t.get('E_first'), 1)} → {fmt(t.get('E_last'), 1)} | {fmt(t.get('F_first'), 1)} → {fmt(t.get('F_last'), 1)} | "
                 f"{fmt(t.get('ratio_seed'), 3)} → {fmt(t.get('ratio_first'), 3)} → {fmt(t.get('ratio_last'), 3)} |")
    L += ['', '## 读数（预登记，只看 seg4；不是结论）', '']
    for arm in X1_ARMS:
        r = dec['arms'][arm]
        L.append(f"- {name(arm)}：seg4 均值 {fmt(r['seg4'])}（同批 {r['n']} 对：A0 {fmt(r['A0_same_pairs'])}，P0n {fmt(r['P0n_same_pairs'])}，中点 {fmt(r['midpoint'], 3)}，"
                 f"A0 − .01 = {fmt(r['near_A0'])}），seg35 均值 {fmt(r['seg35'])}：**{r['reading_text']}**。")
    (out_dir/'summary_x1.md').write_text('\n'.join(L)+'\n')
    (out_dir/'decision_x1.json').write_text(json.dumps(dec, indent=1, ensure_ascii=False)+'\n')
    print('\n'.join(L))
    return 0


# ---- arms A5, A5P (runs/20261103-a5/plan.md, protocol §14) ------------------------------------------------------------------------
A5_REFERENCES = ('A0', 'A1', 'A3', 'A4p', 'A1Ph')
A5_FIRST_N = 5                            # the plan's first gate: the first 5 pairs of selection.json
A5_GATE1_PIN = .8509                      # the plan: mean seg35 of A1 over these 5 pairs, to 4 decimals
A5_MIN_VALID = 17                         # the plan's second gate: pairs with a final map, per arm
A5_SDLOGJ_MAX = .10                       # the plan's second gate: mean official SDlogJ, per arm
A5_ROUTE_THRESHOLD = .845                 # A5P: as A1P's R-route
A5_ORACLE_THRESHOLD = .885                # A5: as the brief (.86 plus the measured transfer losses A1 - A1Ph .0257, A3 - A4p .0218)
A5_THRESHOLDS = dict(A5=A5_ORACLE_THRESHOLD, A5P=A5_ROUTE_THRESHOLD)
SEG35_SMALL = {4: '左下侧脑室', 23: '右下侧脑室', 18: '左血管', 34: '右血管', 19: '左脉络丛', 35: '右脉络丛'}
A5_GROUPS = dict(small_six=tuple(sorted(SEG35_SMALL)), large_six=tuple(sorted(SEG35_LARGE)),
                 other_23=tuple(k for k in range(1, 36) if k not in SEG35_SMALL and k not in SEG35_LARGE))
A5_GROUP_NAMES = dict(small_six='小六个（下侧脑室、血管、脉络丛）', large_six='大六个（大脑皮层、大脑白质、小脑白质）', other_23='其余 23 个')
A5_LABEL_NAMES = ROOT/'configs/bayesreg/structure_groups_v1.json'
A5_READINGS = dict(
    scalar_continues='A5 ≥ .885 且 A5P ≥ .845：标量路线继续',
    segmenter_bottleneck='A5 ≥ .885 且 A5P < .845：分割器是瓶颈',
    scalar_exhausted='A5 < .885：标量到头',
    preconditions='seg35 过线的臂没有满足第二关的前提（证书全部通过、官方 SDlogJ 均值 ≤ .10）：不落入组合读数，交 BZ',
    not_decidable='有臂的终态场少于 17 对：不可判',
    not_complete='还没有跑完 19 对：第二关未判')
A5_NOTE = ('开发集结果，单次观测，读数不是结论，下一步交 BZ。A5 用了真值标签（oracle，只作上限）；A5P、A4′、A1Ph 用训练集标签训练的分割器；A1、A3 用了真值标签。'
           '参照不是同时段对照：A0、A1、A3 来自 D0 全批（2026-10-30），A4′、A1Ph 在其后单独运行；90 s 是墙钟预算，机器状态不同会改变原生提交数。')


def long_csv_runs(root, arms, pairs):
    """{(arm, pair): record} of reference arms from the original <root>/analysis/long.csv of their batch (plan: the references' numbers
    are read from there), scalars only; `rows` keeps the csv rows.  STOP when the file is missing or a run has two run ids."""
    path = Path(root)/'analysis'/'long.csv'
    if not path.is_file():
        raise SystemExit(f'{path}: missing (summarize-a5 reads each reference from the long.csv of its own batch)')
    rows = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row['arm'] in arms and row['pair'] in pairs:
                rows.setdefault((row['arm'], row['pair']), []).append(row)
    runs = {}
    for key, part in rows.items():
        ids = {r['run_id'] for r in part}
        if len(ids) != 1:
            raise SystemExit(f'{path}: {key} has run ids {sorted(ids)}')
        v = {r['measure']: float(r['value']) for r in part}
        metrics = {k: v[k] for k in LONG_MEASURES if k in v}
        runs[key] = dict(run_id=ids.pop(), metrics=metrics if 'seg35' in metrics else None, source=str(path), rows=part,
                         certificate=None if 'cert_passed' not in v else dict(passed=bool(v['cert_passed']),
                                                                              min_jd_lower_bound=v.get('cert_min_jd_lower_bound')),
                         self_check=None if 'self_check_passed' not in v else dict(passed=bool(v['self_check_passed'])),
                         native_commits=int(v['native_commits']) if 'native_commits' in v else None, e2e_seconds=v.get('e2e_seconds'))
    return runs


def attach_reference_labels(refs, root, arms):
    """Per-label Dice is not in long.csv: take it from the metrics.json of the same run (<root>/runs), only if its run_id, seg35 and seg4
    equal the long.csv row.  Returns the disagreements (reported first in the summary, and they keep the first gate shut)."""
    details = chosen_runs(Path(root)/'runs', arms)
    problems = []
    for key, rec in refs.items():
        m = details.get(key)
        if rec['metrics'] is None:
            continue
        if not m or not m.get('metrics'):
            problems.append(f'{key}: long.csv 有指标，{root}/runs 没有对应的 metrics.json')
            continue
        if m.get('run_id') != rec['run_id'] or any(m['metrics'][k] != rec['metrics'][k] for k in ('seg35', 'seg4')):
            problems.append(f"{key}: long.csv（{rec['run_id']}）与 metrics.json（{m.get('run_id')}）不一致")
            continue
        rec['metrics'] = dict(rec['metrics'], seg4_per_label=m['metrics'].get('seg4_per_label'), seg35_per_label=m['metrics'].get('seg35_per_label'))
        rec['evaluator'] = m.get('evaluator')
    return problems


def _a5_mean(runs, arm, pairs, key):
    values = [runs[(arm, p)]['metrics'][key] for p in pairs if (runs.get((arm, p)) or {}).get('metrics')]
    return float(np.mean(values)) if values else None


def _passed(run):
    return bool(((run or {}).get('certificate') or {}).get('passed'))


def decide_a5_gate1(runs, pairs, reference_problems=()):
    """The plan's first gate: the mean seg35 of A5 over the first 5 pairs > the mean seg35 of A1 over the same pairs: both arms go on to
    19 pairs; otherwise both stay at 5.  A5P is not read here.  A pair counts when A5 has a final map with a passed certificate and A1 has
    a final map; fewer than 5 such pairs, a pinned A1 mean that does not reproduce, or a reference that disagrees with its metrics.json:
    not decidable, no expansion, to the lead."""
    first = list(pairs[:A5_FIRST_N])
    same = [p for p in first if (runs.get(('A5', p)) or {}).get('metrics') and _passed(runs.get(('A5', p)))
            and (runs.get(('A1', p)) or {}).get('metrics')]
    a5, a1 = _a5_mean(runs, 'A5', same, 'seg35'), _a5_mean(runs, 'A1', same, 'seg35')
    record = dict(rule='A5 seg35 mean > A1 seg35 mean over the first 5 pairs; A5P not read', pairs=first, n=len(same), A5_seg35=a5, A1_seg35=a1,
                  A5_seg4=_a5_mean(runs, 'A5', same, 'seg4'), A1_seg4=_a5_mean(runs, 'A1', same, 'seg4'), A1_pin=A5_GATE1_PIN, expand=False)
    if len(same) < A5_FIRST_N:
        return dict(record, status='not_decidable', text=f'前 5 对里 A5 有终态场且证书通过、A1 有终态场的只有 {len(same)} 对：第一关不可判，不扩，交总负责人')
    if abs(a1-A5_GATE1_PIN) > 5.01e-5 or reference_problems:
        return dict(record, status='not_decidable', text=f'A1 同 5 对重算 {a1:.4f}（预登记 {A5_GATE1_PIN}）或参照与 metrics.json 不一致：第一关不可判，不扩，交总负责人')
    if a5 > a1:
        return dict(record, status='expand', expand=True, text=f'A5 seg35 均值 {a5:.4f} > A1 同 5 对 {a1:.4f}：两臂都扩到 19 对')
    return dict(record, status='stop', text=f'换颜色表没有超过 A1（A5 seg35 均值 {a5:.4f} ≤ A1 同 5 对 {a1:.4f}）：两臂都停在 5 对，交 BZ')


def a5_arm_gate2(runs, arm, pairs):
    """One arm of the second gate: at least 17 pairs with a final map, every certificate passed, mean official SDlogJ <= .10, mean seg35 >= the
    arm's threshold (A5 .885, A5P .845)."""
    ms = [runs[(arm, p)] for p in pairs if (arm, p) in runs]
    ok = [m for m in ms if m.get('metrics')]
    seg35, sd = (_a5_mean(runs, arm, pairs, k) for k in ('seg35', 'sdlogj_official'))
    certified = sum(_passed(m) for m in ok)
    row = dict(threshold=A5_THRESHOLDS[arm], n_ran=len(ms), n_with_field=len(ok), n_certified=certified, seg35_mean=seg35,
               seg4_mean=_a5_mean(runs, arm, pairs, 'seg4'), sdlogj_official_mean=sd, valid=len(ok) >= A5_MIN_VALID,
               certificates_all_passed=bool(ok) and certified == len(ok), sdlogj_ok=sd is not None and sd <= A5_SDLOGJ_MAX,
               seg35_at_threshold=seg35 is not None and seg35 >= A5_THRESHOLDS[arm])
    row['meets'] = row['valid'] and row['certificates_all_passed'] and row['sdlogj_ok'] and row['seg35_at_threshold']
    return row


def decide_a5_gate2(runs, pairs):
    """The plan's second gate, each arm on its own, and the combined reading; evaluated only when both arms have a metrics.json on every
    pair.  A5P meets: the route (predicted colouring + frozen solver) opens; A5 meets: the route could reach .86.  Reading: A5 below .885:
    scalar exhausted; A5 meets and A5P below .845: the segmenter is the bottleneck; both meet: the scalar route continues; an arm at its
    threshold that misses a precondition (certificates, SDlogJ): no reading, to BZ; an arm with fewer than 17 final maps: not decidable."""
    a5, a5p = a5_arm_gate2(runs, 'A5', pairs), a5_arm_gate2(runs, 'A5P', pairs)
    if a5['n_ran'] < len(pairs) or a5p['n_ran'] < len(pairs):
        key = 'not_complete'
    elif not a5['valid']:
        key = 'not_decidable'
    elif not a5['seg35_at_threshold']:
        key = 'scalar_exhausted'
    elif not a5['meets']:
        key = 'preconditions'
    elif not a5p['valid']:
        key = 'not_decidable'
    elif not a5p['seg35_at_threshold']:
        key = 'segmenter_bottleneck'
    elif not a5p['meets']:
        key = 'preconditions'
    else:
        key = 'scalar_continues'
    evaluated = key != 'not_complete'

    def line(arm, r):
        return (f"{arm} seg35 均值 {fmt(r['seg35_mean'])}（门槛 {r['threshold']}），seg4 均值 {fmt(r['seg4_mean'])}，终态场 {r['n_with_field']} 对，证书通过 {r['n_certified']}，"
                f"官方 SDlogJ 均值 {fmt(r['sdlogj_official_mean'])}：{'满足' if r['meets'] else '不满足'}")
    text = f"第二关未判：A5 {a5['n_ran']}/{len(pairs)}、A5P {a5p['n_ran']}/{len(pairs)} 次运行有 metrics.json" if not evaluated else line('A5', a5)+'；'+line('A5P', a5p)
    return dict(rule='per arm: >= 17 final maps, every certificate passed, mean official SDlogJ <= .10, mean seg35 >= threshold (A5 .885, A5P .845)',
                evaluated=evaluated, arms=dict(A5=a5, A5P=a5p), A5P_opens_route=evaluated and a5p['meets'], A5_could_reach_86=evaluated and a5['meets'],
                reading=key, reading_text=A5_READINGS[key], text=text)


def a5_group_row(runs, arm, ref, labels, pairs):
    """The paired difference of a label group's mean seg35 (per pair, over the labels both runs have), and its share of the macro difference:
    the sum over the labels of the mean per-label difference / 35."""
    diffs = []
    for p in pairs:
        both = [(per_label(runs.get((arm, p)), 'seg35', k), per_label(runs.get((ref, p)), 'seg35', k)) for k in labels]
        both = [(a, b) for a, b in both if a is not None and b is not None]
        if both:
            diffs.append(float(np.mean([a for a, _ in both])-np.mean([b for _, b in both])))
    rows = label_rows(runs, arm, ref, 'seg35', labels, pairs)
    return dict(mean_diff=float(np.mean(diffs)) if diffs else None, median_diff=_median(diffs), positive=sum(d > 0 for d in diffs), n=len(diffs),
                contribution_to_macro=sum(rows[k]['mean_diff'] or 0. for k in rows)/35)


def a5_group_mean(runs, arm, labels, pairs):
    """Mean over the pairs of the group's mean per-label seg35."""
    values = []
    for p in pairs:
        row = [per_label(runs.get((arm, p)), 'seg35', k) for k in labels]
        row = [v for v in row if v is not None]
        if row:
            values.append(float(np.mean(row)))
    return float(np.mean(values)) if values else None


def a5_anomalies(runs, pairs):
    """The runs of A5, A5P that are blocked or abnormal: no metrics.json yet, a class other than ok, no final map, a certificate or self-check
    that did not pass."""
    out = []
    for arm in A5_ARMS:
        for p in pairs:
            m = runs.get((arm, p))
            if m is None:
                out.append(dict(arm=arm, pair=p, what='没有 metrics.json（未运行、未跑完或基础设施失败）'))
                continue
            c = m.get('classification') or {}
            why = []
            if c.get('class') != 'ok':
                why.append(f"分类 {c.get('class')}（{c.get('kind')}：{c.get('reason')}）")
            if not m.get('metrics'):
                why.append('无终态场')
            else:
                if not _passed(m):
                    why.append('证书未通过')
                if not (m.get('self_check') or {}).get('passed'):
                    why.append('自检未通过')
            if why:
                out.append(dict(arm=arm, pair=p, what='；'.join(why), run_id=m.get('run_id')))
    return out


def _set_aside(out_dir, names):
    """An earlier summary is evidence (the 5-pair decision, a partial one): moved to <out_dir>/previous_<UTC>[_k]/, never overwritten."""
    present = [n for n in names if (out_dir/n).exists()]
    if not present:
        return None
    stamp, k = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'), 0
    keep = out_dir/f'previous_{stamp}'
    while keep.exists():
        k += 1
        keep = out_dir/f'previous_{stamp}_{k}'
    keep.mkdir()
    for n in present:
        (out_dir/n).rename(keep/n)
    return keep


def summarize_a5(a5_runs_root, d0_root, a4_root, a1p_root, out_dir, stage=STAGE):
    """A5, A5P (a5_runs_root, metrics.json) with the references of the plan: the numbers from the long.csv of each batch (d0_root: A0, A1, A3;
    a4_root: A4p; a1p_root: A1Ph), only the pairs of this batch -> long.csv, summary_a5.md, decision_a5.json in out_dir (never under a reference
    batch).  Works with any number of pairs; the gates decide what they can."""
    out_dir = Path(out_dir)
    for root in (D0, D0A4, D0A1P, d0_root, a4_root, a1p_root):
        if _inside(out_dir, root):
            raise SystemExit(f'{out_dir}: the A5 summary is never written under {root}')
    pairs = [r['pair'] for r in json.loads((Path(stage)/'selection.json').read_text())['pairs']]
    runs = chosen_runs(a5_runs_root, A5_ARMS)
    gate_pairs = sorted(set(pairs[:A5_FIRST_N]) | {p for _, p in runs}, key=pairs.index)      # the first gate needs A1 on its 5 pairs
    batch = [p for p in pairs if any((arm, p) in runs for arm in A5_ARMS)]
    problems = []
    for root, arms in ((d0_root, ('A0', 'A1', 'A3')), (a4_root, ('A4p',)), (a1p_root, ('A1Ph',))):
        refs = long_csv_runs(root, arms, gate_pairs)
        problems += attach_reference_labels(refs, root, arms)
        runs.update(refs)
    arms = A5_ARMS+A5_REFERENCES
    name = a1p_name
    gate1 = decide_a5_gate1(runs, pairs, problems)
    gate2 = decide_a5_gate2(runs, pairs)
    loads = {(arm, p): launch_load(runs[(arm, p)]['run_dir']) for arm in A5_ARMS for p in batch if (arm, p) in runs and runs[(arm, p)].get('run_dir')}
    out_dir.mkdir(parents=True, exist_ok=True)
    moved = _set_aside(out_dir, ('long.csv', 'summary_a5.md', 'decision_a5.json'))
    fields = ['set', 'split', 'method', 'arm', 'pair', 'measure', 'value', 'run_id']
    own = {k: m for k, m in runs.items() if k[0] in A5_ARMS}
    with open(out_dir/'long.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(long_rows(own))
        for arm in A5_REFERENCES:                             # the source rows, verbatim, for the pairs of this batch
            for p in batch:
                w.writerows(r for r in (runs.get((arm, p)) or {}).get('rows', []))
    measures = ('seg35', 'seg4', 'hd95', 'sdlogj_official', 'sdlogj_brain')
    comparisons = {}
    for arm in A5_ARMS:
        for ref in A5_REFERENCES+(('A5P',) if arm == 'A5' else ()):
            comparisons[f'{arm} - {ref}'] = {}
            for k in measures:
                v = list(paired_ref(runs, arm, ref, k, batch).values())
                comparisons[f'{arm} - {ref}'][k] = dict(mean_diff=float(np.mean(v)) if v else None, median_diff=_median(v), positive=sum(x > 0 for x in v), n=len(v))
    dec = dict(rule='runs/20261103-a5/plan.md', n_pairs_in_batch=len(batch), batch=batch, gate1=gate1, gate2=gate2, reference_problems=problems, comparisons=comparisons,
               note=A5_NOTE, previous_summary_moved_to=None if moved is None else str(moved))
    dec['sources'] = dict(a5_runs_root=str(a5_runs_root), d0_root=str(d0_root), a4_root=str(a4_root), a1p_root=str(a1p_root),
                          reference_numbers='analysis/long.csv of each batch; per-label Dice from the metrics.json of the same run_id',
                          evaluator_sha256={arm: sorted({(runs[(arm, p)].get('evaluator') or {}).get('sha256') or '-' for p in batch if (arm, p) in runs}) for arm in arms})
    dec['seg35_groups'] = {g: dict(labels=list(labels), means={arm: a5_group_mean(runs, arm, labels, batch) for arm in arms},
                                   paired={f'{arm} - {ref}': a5_group_row(runs, arm, ref, labels, batch) for arm in A5_ARMS for ref in A5_REFERENCES})
                           for g, labels in A5_GROUPS.items()}
    dec['seg4_structures'] = {str(label): dict(name=SEG4_STRUCTURES[label], paired={f'{arm} - {ref}': label_rows(runs, arm, ref, 'seg4', (label,), batch)[label]
                                                                                    for arm in A5_ARMS for ref in A5_REFERENCES}) for label in (1, 3)}
    dec['anomalies'] = a5_anomalies(runs, batch)
    dec['launch_loadavg'] = {f'{a}/{p}': v for (a, p), v in loads.items()}
    label_names = json.loads(A5_LABEL_NAMES.read_text())['label_table']['names']
    L = ['# A5、A5P 汇总（开发集结果，单次观测）', '',
         f'生成：{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}（UTC）；计划 runs/20261103-a5/plan.md；方法 {METHOD}；本批 {len(batch)} 对；'
         '指标由 experiments/d0/evaluate.py 从终态场算出，参照臂的数取各自批次的原始 analysis/long.csv，只比同一批配对。', '', A5_NOTE, '']
    if len(batch) <= A5_FIRST_N:
        L += ['**只有前 5 对时，读数只有方向信息**（X1 计划估的 n = 5 最小可检测效应约 .009–.020）。', '']
    if problems:
        L += ['**参照与预登记或 metrics.json 不符，读数要交总负责人**：' + '；'.join(problems), '']
    if moved is not None:
        L += [f'上一版汇总移到 {moved}（保留）。', '']
    L += ['## 门槛与读数（预登记，由代码判；seg4 不设门槛，同时报告）', '',
          f"- **第一关**（前 5 对，只看 A5 与 A1）：{gate1['text']}。A5 seg35 {fmt(gate1['A5_seg35'])}、seg4 {fmt(gate1['A5_seg4'])}；A1 同批 seg35 {fmt(gate1['A1_seg35'])}、seg4 {fmt(gate1['A1_seg4'])}（{gate1['n']} 对）。",
          f"- **第二关**：{gate2['text']}。", f"- A5P 打开路线（预测上色 + 现成求解器）：{gate2['A5P_opens_route']}；A5 这条路线才可能碰到 .86：{gate2['A5_could_reach_86']}。",
          f"- **组合读数**：{gate2['reading_text']}。这是读数，不是结论，下一步交 BZ。", '',
          '## 各臂均值（本批配对；seg4 与 seg35 同时报）', '',
          '| 臂 | 有终态场的对数 | seg35 | seg4 | Dice35 官方 | HD95 | SDlogJ 官方 | 脑内 SDlogJ | 证书通过 | 最小下界 | 自检通过 | e2e 中位 s | 原生提交中位 |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in arms:
        ms = [runs[(arm, p)] for p in batch if (arm, p) in runs]
        ok = [m for m in ms if m.get('metrics')]
        mean = (lambda k: _a5_mean(runs, arm, batch, k))
        lower = [m['certificate'].get('min_jd_lower_bound') for m in ms if m.get('certificate') and m['certificate'].get('min_jd_lower_bound') is not None]
        L.append(f"| {name(arm)} | {len(ok)}/{len(batch)} | {fmt(mean('seg35'))} | {fmt(mean('seg4'))} | {fmt(mean('dice_official'))} | {fmt(mean('hd95'), 3)} | "
                 f"{fmt(mean('sdlogj_official'))} | {fmt(mean('sdlogj_brain'), 3)} | {sum(_passed(m) for m in ms)}/{len(ms)} | {f'{min(lower):.3e}' if lower else '—'} | "
                 f"{sum(bool((m.get('self_check') or {}).get('passed')) for m in ms)}/{len(ms)} | {fmt(_median(m.get('e2e_seconds') for m in ms), 1)} | "
                 f"{fmt(_median(m.get('native_commits') for m in ms), 1)} |")
    L += ['', '## 逐对差（前者 − 后者，只比两臂都有终态场的配对）', '', '| 比较 | 指标 | 均值差 | 中位差 | 为正的对数 / 有效对数 |', '|---|---|---|---|---|']
    for pair_name, row in comparisons.items():
        arm, ref = pair_name.split(' - ')
        for k in measures:
            r = row[k]
            L.append(f"| {arm} − {name(ref)} | {k} | {fmt(r['mean_diff'])} | {fmt(r['median_diff'])} | {r['positive']} / {r['n']} |")
    L += ['', '## seg35 按标签组（每对先在组内平均，再对配对平均）', '', '| 组 | ' + ' | '.join(name(a) for a in arms) + ' |', '|---|' + '---|'*len(arms)]
    for g in A5_GROUPS:
        L.append(f"| {A5_GROUP_NAMES[g]} | " + ' | '.join(fmt(dec['seg35_groups'][g]['means'][a]) for a in arms) + ' |')
    L += ['', '| 比较 | 组 | 均值差 | 中位差 | 为正的对数 / 有效对数 | 对 macro 差的贡献（标签均值差之和 / 35） |', '|---|---|---|---|---|---|']
    for arm in A5_ARMS:
        for ref in A5_REFERENCES:
            for g in A5_GROUPS:
                r = dec['seg35_groups'][g]['paired'][f'{arm} - {ref}']
                L.append(f"| {arm} − {name(ref)} | {A5_GROUP_NAMES[g]} | {fmt(r['mean_diff'])} | {fmt(r['median_diff'])} | {r['positive']} / {r['n']} | {fmt(r['contribution_to_macro'])} |")
    L += ['', '## seg4 的皮层与白质', '', '| seg4 标签 | ' + ' | '.join(name(a) for a in arms) + ' |', '|---|' + '---|'*len(arms)]
    for label in (1, 3):
        L.append(f'| {label} {SEG4_STRUCTURES[label]} | ' + ' | '.join(fmt(label_rows(runs, a, a, 'seg4', (label,), batch)[label]['arm_mean']) for a in arms) + ' |')
    L += ['', '| 比较 | 结构 | 均值差 | 中位差 | 为正的对数 / 有效对数 |', '|---|---|---|---|---|']
    for arm in A5_ARMS:
        for ref in A5_REFERENCES:
            for label in (1, 3):
                r = dec['seg4_structures'][str(label)]['paired'][f'{arm} - {ref}']
                L.append(f"| {arm} − {name(ref)} | {SEG4_STRUCTURES[label]} | {fmt(r['mean_diff'])} | {fmt(r['median_diff'])} | {r['positive']} / {r['n']} |")
    L += ['', '## seg35 逐标签（本批配对均值）', '', '| 标签 | 结构 | 组 | ' + ' | '.join(name(a) for a in arms) + ' |', '|---|---|---|' + '---|'*len(arms)]
    for label in range(1, 36):
        group = next(g for g, labels in A5_GROUPS.items() if label in labels)
        L.append(f"| {label} | {label_names[str(label)]} | {group} | " + ' | '.join(fmt(label_rows(runs, a, a, 'seg35', (label,), batch)[label]['arm_mean']) for a in arms) + ' |')
    L += ['', '## 逐对 seg35（seg4）', '', '| 对 | ' + ' | '.join(name(a) for a in arms) + ' |', '|---|' + '---|'*len(arms)]
    for p in batch:
        cells = []
        for arm in arms:
            m = runs.get((arm, p))
            cells.append('—' if not m or not m.get('metrics') else f"{m['metrics']['seg35']:.4f} ({m['metrics']['seg4']:.4f})")
        L.append(f"| {p[11:15]}←{p[-8:-4]} | " + ' | '.join(cells) + ' |')
    L += ['', '## A5、A5P 的每次运行', '', '| 运行 | 分类 | 状态 | 证书 | 最小下界 | 原生提交 | e2e s | 启动负载（1/5 分钟） |', '|---|---|---|---|---|---|---|---|']
    for p in batch:
        for arm in A5_ARMS:
            m = runs.get((arm, p))
            if m is None:
                L.append(f'| {arm} {p[11:15]}←{p[-8:-4]} | 没有 metrics.json | — | — | — | — | — | — |')
                continue
            c = m.get('certificate') or {}
            load = loads.get((arm, p))
            L.append(f"| {arm} {p[11:15]}←{p[-8:-4]} | {(m.get('classification') or {}).get('class')} | {m.get('status')} | {'通过' if _passed(m) else '未通过'} | "
                     f"{'—' if c.get('min_jd_lower_bound') is None else format(c['min_jd_lower_bound'], '.3e')} | {m.get('native_commits')} | {fmt(m.get('e2e_seconds'), 1)} | "
                     f"{'—' if not load else f'{load[0]:.2f} / {load[1]:.2f}'} |")
    L += ['', '## 受阻或异常的运行', '']
    L += [f"- {a['arm']} {a['pair']}：{a['what']}" for a in dec['anomalies']] or ['- 没有。']
    L += ['', '**评测脚本版本**：各臂 metrics.json 记录的 experiments/d0/evaluate.py sha256 为 '
          + '；'.join(f"{name(arm)} {', '.join(s[:12] for s in dec['sources']['evaluator_sha256'][arm]) or '—'}" for arm in arms) + '。']
    (out_dir/'summary_a5.md').write_text('\n'.join(L)+'\n')
    (out_dir/'decision_a5.json').write_text(json.dumps(dec, indent=1, ensure_ascii=False)+'\n')
    print('\n'.join(L))
    return 0


# ---- option arms B1 (runs/20261103-b1/plan.md, protocol §15) --------------------------------------------------------------------
B1 = Path('/home/rmcfz/seq_vp_tag_results/b1_v1')
B1_SELECTION_SPEC = ROOT/'runs/20261103-b1/arms_selection.json'
B1_FIRST_N = 5                            # the plan's selection: the first 5 pairs of selection.json
B1_PINS = dict(A5P=(.8290, .8894), A5=(.8705, .9289))     # (seg35, seg4) of d0_v1_a5 over these 5 pairs, to 4 decimals (plan: baselines)
B1_PIN_TOL = 5.01e-5
B1_SDLOGJ_MAX = .095                      # qualified: mean official SDlogJ
B1_SEG4_MARGIN = .003                     # qualified: mean seg4 >= A5P on the same 5 pairs minus this
B1_CHEAPER_MARGIN = .002                  # a cheaper qualified arm closer than this to the best one wins
B1_MIN_GAIN = .005                        # winner minus A5P on the same 5 pairs, mean seg35, below this: configuration C
B1_TARGET_SEG35, B1_NEAR_SEG35, B1_TARGET_SDLOGJ = .86, .85, .10      # the readings on 19 pairs
B1_MIN_VALID = 17                         # as protocol §7: fewer valid pairs: not decidable
B1_ORACLE_RAISED = .8780                  # .885 minus the offset .0070 of A5 between its 19 pairs and its first 5
B1_REFERENCES = ('A5P', 'A5')
B1_ORACLE_ARM, B1_V2_ARM = b1_spec.ORACLE_ARM, b1_spec.V2_ARM
B1_CONFIG_C = dict(arm='configC', seconds=b1_spec.CONFIG_C['seconds'], changes={}, note='配置 C，不改任何选项')
B1_READINGS = dict(
    target='证书全过、官方 SDlogJ 均值 ≤ .10 且 seg35 ≥ .86：达到 BZ 的目标（开发集，从 5 个配置里选出）',
    near='seg35 ≥ .85：接近，要看分割器',
    not_enough='选项不够',
    no_expansion='胜出配置是配置 C，不扩展；19 对的数就是 A5P（d0_v1_a5）',
    not_decided='选配未判，不扩展',
    not_complete='胜出臂还没有跑完 19 对：读数未判',
    not_decidable='胜出臂的终态场少于 17 对，或有证书未通过的场：不可判')
B1_ORACLE_READINGS = dict(
    raised='B1W_A5 前 5 对的 seg35 ≥ .8780：标量上限比 90 s 的读数高，“标量到头”的范围要改',
    not_raised='B1W_A5 前 5 对的 seg35 未达 .8780 这条线',
    not_run='B1W_A5 还没有跑完前 5 对：未判',
    not_applicable='胜出配置是配置 C：计划不跑 B1W_A5，不判',
    not_decided='选配未判：B1W_A5 不跑，不判',
    not_decidable='B1W_A5 前 5 对的终态场不足 5 对，或证书未全过：不可判',
    config_mismatch='B1W_A5 的配置与胜出配置不同：不判，交总负责人')
B1_NOTE = ('开发集，单次观测，读数不是结论；n = 5 只有方向信息；胜出是挑出来的，带选择效应；基线（A5P、A5）不是同时段，来自 d0_v1_a5（2026-11-03）；'
           'A5 用真值，A5P 用训练集标签训练的分割器；颜色表依据 6 名开发受试者（0438–0443）；预算收尾不是收敛。'
           '增益门 .005 与更便宜门 .002 低于 n = 5 的最小可检测效应约 .009–.020。'
           '`--geometry-weight-levels` 的第 4 个值（B1M、B1TM 由 .03 改为 .01）是原生层相对上一外层的近端变化罚 μ = ν，不是 F = D + B 里的绝对先验，度量跟着变；T88 预优化器自己的 μ（.003）未改；结果读作路径或步长效应，不读作正则强度效应。')
B1_ORACLE_LABEL = '（oracle，只作上限）'
B1_MEASURES = ('seg35', 'seg4', 'hd95', 'sdlogj_official', 'sdlogj_brain', 'pct_J_lt_half_brain')


def b1_candidate_row(runs, entry, first, baseline_seg4):
    """One arm of the selection: the plan's four conditions on the first 5 pairs, each one recorded."""
    arm = entry['arm']
    ms = [runs[(arm, p)] for p in first if (arm, p) in runs]
    ok = [m for m in ms if m.get('metrics')]
    seg35, seg4, sd = (_a5_mean(runs, arm, first, k) for k in ('seg35', 'seg4', 'sdlogj_official'))
    row = dict(arm=arm, seconds=entry['seconds'], changes=entry['changes'], n_ran=len(ms), n_with_field=len(ok), n_certified=sum(_passed(m) for m in ok),
               seg35_mean=seg35, seg4_mean=seg4, sdlogj_official_mean=sd, fields_ok=len(ok) == len(first), certificates_ok=sum(_passed(m) for m in ok) == len(first),
               sdlogj_ok=sd is not None and sd <= B1_SDLOGJ_MAX, seg4_ok=seg4 is not None and baseline_seg4 is not None and seg4 >= baseline_seg4-B1_SEG4_MARGIN,
               n_handover_only=sum((m.get('classification') or {}).get('class') == 'handover_only' for m in ms))
    row['qualified'] = row['fields_ok'] and row['certificates_ok'] and row['sdlogj_ok'] and row['seg4_ok']
    return row


def decide_b1_selection(runs, pairs, candidates, reference_problems=()):
    """The plan's selection, first 5 pairs.  Qualified: 5/5 final maps, every certificate passed, mean official SDlogJ <= .095, mean seg4 >= A5P - .003.
    The best qualified arm by mean seg35 (ties: fewer seconds, then specification order); a qualified arm with fewer seconds closer than .002 takes
    its place (the fewest seconds among such arms, then the highest seg35).  Its gain over A5P below .005: configuration C.  No qualified arm:
    configuration C.  A candidate or the baseline without metrics.json on one of the 5 pairs: not complete; a baseline that disagrees with the
    pinned values or with its metrics.json: not decidable; in both the winner is None and nothing runs beyond the selection."""
    first = list(pairs[:B1_FIRST_N])
    base35, base4 = _a5_mean(runs, 'A5P', first, 'seg35'), _a5_mean(runs, 'A5P', first, 'seg4')
    base_n = sum(bool((runs.get(('A5P', p)) or {}).get('metrics')) for p in first)
    rows = [b1_candidate_row(runs, entry, first, base4) for entry in candidates]
    rec = dict(rule='runs/20261103-b1/plan.md', pairs=first, baseline=dict(arm='A5P', n=base_n, seg35_mean=base35, seg4_mean=base4, pins=B1_PINS['A5P']),
               thresholds=dict(sdlogj_official_max=B1_SDLOGJ_MAX, seg4_margin=B1_SEG4_MARGIN, cheaper_margin=B1_CHEAPER_MARGIN, min_gain=B1_MIN_GAIN), arms=rows,
               best_arm=None, best_gain=None, selected_arm=None, gain=None, winner=None, winner_spec=None)
    if base_n < len(first) or any(r['n_ran'] < len(first) for r in rows):
        return dict(rec, status='not_complete', text='基线或候选臂在前 5 对上还没有全部的 metrics.json：选配未判，不扩展')
    pinned = abs(base35-B1_PINS['A5P'][0]) <= B1_PIN_TOL and abs(base4-B1_PINS['A5P'][1]) <= B1_PIN_TOL
    if reference_problems or not pinned:
        return dict(rec, status='not_decidable', text=f'A5P 同 5 对重算 {base35:.4f} / {base4:.4f}（预登记 {B1_PINS["A5P"]}）或参照与 metrics.json 不一致：选配不可判，不扩展，交总负责人')
    entries = {e['arm']: e for e in candidates}
    good = [r for r in rows if r['qualified']]
    if not good:
        return dict(rec, status='no_qualified', winner='configC', winner_spec=dict(B1_CONFIG_C), text='没有合格臂：胜出配置 = 配置 C')
    best = max(good, key=lambda r: (r['seg35_mean'], -r['seconds']))
    near = [r for r in good if r['seconds'] < best['seconds'] and best['seg35_mean']-r['seg35_mean'] < B1_CHEAPER_MARGIN]
    if near:
        fewest = min(r['seconds'] for r in near)
        chosen = max((r for r in near if r['seconds'] == fewest), key=lambda r: r['seg35_mean'])
    else:
        chosen = best
    gain = chosen['seg35_mean']-base35
    best_gain = best['seg35_mean']-base35
    rec.update(best_arm=best['arm'], best_gain=best_gain, selected_arm=chosen['arm'], gain=gain)
    if gain < B1_MIN_GAIN:
        flipped = (f"；seg35 最高的 {best['arm']} 增益 {best_gain:+.4f} ≥ .005，按规则换成差 < .002 的更便宜臂 {chosen['arm']} 之后才不到 .005（规则不改，这里写明）"
                   if chosen is not best and best_gain >= B1_MIN_GAIN else '')
        return dict(rec, status='no_clear_gain', winner='configC', winner_spec=dict(B1_CONFIG_C),
                    text=f"选项没有明显提升：{chosen['arm']} 相对 A5P 同 5 对的 seg35 增益 {gain:+.4f} < .005，胜出配置 = 配置 C{flipped}")
    return dict(rec, status='winner', winner=chosen['arm'], winner_spec=dict(entries[chosen['arm']]),
                text=f"胜出 {chosen['arm']}：seg35 均值 {chosen['seg35_mean']:.4f}，相对 A5P 同 5 对 {gain:+.4f}；seg4 均值 {chosen['seg4_mean']:.4f}，A5P {base4:.4f}")


def decide_b1_readings(runs, pairs, selection, oracle_inputs=None):
    """The plan's readings: the winner on 19 pairs and B1W_A5 on the first 5.  Readings, not conclusions.  Rules the plan leaves open are written in
    the plan: a method failure counts as run (no final map), fewer than 17 final maps or a final map whose certificate did not pass: not decidable (revision 1, M1).
    oracle_inputs: the inputs root B1W_A5 must have used (revision 1, M10)."""
    winner = selection['winner']
    row = dict(winner=winner, n_ran=None, n_with_field=None, n_certified=None, seg35_mean=None, seg4_mean=None, sdlogj_official_mean=None)
    if winner is None:
        key = 'not_decided'
    elif winner == 'configC':
        key = 'no_expansion'
    else:
        ms = [runs[(winner, p)] for p in pairs if (winner, p) in runs]
        ok = [m for m in ms if m.get('metrics')]
        seg35, sd = (_a5_mean(runs, winner, pairs, k) for k in ('seg35', 'sdlogj_official'))
        certified = sum(_passed(m) for m in ok)
        row.update(n_ran=len(ms), n_with_field=len(ok), n_certified=certified, seg35_mean=seg35, seg4_mean=_a5_mean(runs, winner, pairs, 'seg4'), sdlogj_official_mean=sd)
        if len(ms) < len(pairs):
            key = 'not_complete'
        elif len(ok) < B1_MIN_VALID or certified < len(ok):
            key = 'not_decidable'
        elif certified == len(ms) and sd <= B1_TARGET_SDLOGJ and seg35 >= B1_TARGET_SEG35:
            key = 'target'
        elif seg35 >= B1_NEAR_SEG35:
            key = 'near'
        else:
            key = 'not_enough'
    first = list(pairs[:B1_FIRST_N])
    arm = B1_ORACLE_ARM
    ms = [runs[(arm, p)] for p in first if (arm, p) in runs]
    ok = [m for m in ms if m.get('metrics')]
    seg35 = _a5_mean(runs, arm, first, 'seg35')
    spec = selection.get('winner_spec')
    same = all((m.get('b1') or {}).get('seconds') == spec['seconds'] and (m.get('b1') or {}).get('changes') == spec['changes']
               and (oracle_inputs is None or (m.get('b1') or {}).get('inputs_root') == str(oracle_inputs)) for m in ms) if spec else None
    if winner is None:
        oracle = 'not_decided'
    elif winner == 'configC':
        oracle = 'not_applicable'
    elif len(ms) < len(first):
        oracle = 'not_run'
    elif same is not True:
        oracle = 'config_mismatch'
    elif len(ok) < len(first) or sum(_passed(m) for m in ok) < len(first):
        oracle = 'not_decidable'
    else:
        oracle = 'raised' if seg35 >= B1_ORACLE_RAISED else 'not_raised'
    text = B1_READINGS[key]
    if key in ('target', 'near', 'not_enough'):
        text += (f"（{winner} 19 对 seg35 均值 {fmt(row['seg35_mean'])}，seg4 均值 {fmt(row['seg4_mean'])}，官方 SDlogJ 均值 {fmt(row['sdlogj_official_mean'])}，"
                 f"证书通过 {row['n_certified']}/{row['n_ran']}）")
    return dict(reading=key, text=text, winner_row=row, oracle=dict(arm=arm, reading=oracle, text=B1_ORACLE_READINGS[oracle], threshold=B1_ORACLE_RAISED, n_ran=len(ms),
                                                                    n_with_field=len(ok), n_certified=sum(_passed(m) for m in ok), seg35_mean=seg35,
                                                                    seg4_mean=_a5_mean(runs, arm, first, 'seg4')))


def decide_b1_v2(runs, pairs, selection):
    """Revision 1: B1W_v2 (the inputs of the new segmenter on the winner's configuration) against A5P on the same configuration: the winner's arm, or, when the winner is
    configC, the A5P of d0_v1_a5.  Paired differences, seg35 and seg4 together; a reading, not a conclusion."""
    arm, winner = B1_V2_ARM, selection['winner']
    ref = 'A5P' if winner in (None, 'configC') else winner
    ms = [runs[(arm, p)] for p in pairs if (arm, p) in runs]
    rec = dict(arm=arm, reference=ref, n_ran=len(ms))
    if not ms:
        return dict(rec, status='not_run', text='B1W_v2 还没有跑')
    spec = selection.get('winner_spec')
    if not spec or not all((m.get('b1') or {}).get('seconds') == spec['seconds'] and (m.get('b1') or {}).get('changes') == spec['changes'] for m in ms):
        return dict(rec, status='config_mismatch', text=f'B1W_v2 的配置不是胜出配置（winner {winner}）：不读，交总负责人')
    row = {}
    for measure in ('seg35', 'seg4'):
        diffs = paired_ref(runs, arm, ref, measure, pairs)
        row[measure] = dict(mean_diff=float(np.mean(list(diffs.values()))) if diffs else None, median_diff=_median(diffs.values()), positive=sum(d > 0 for d in diffs.values()), n=len(diffs),
                            arm_mean=_a5_mean(runs, arm, list(diffs), measure), reference_mean=_a5_mean(runs, ref, list(diffs), measure))
    ok = [m for m in ms if m.get('metrics')]
    rec.update(n_with_field=len(ok), n_certified=sum(_passed(m) for m in ok), seg35=row['seg35'], seg4=row['seg4'], status='read')
    part = lambda k: f"{k} 均值差 {fmt(row[k]['mean_diff'])}、中位差 {fmt(row[k]['median_diff'])}、{row[k]['positive']}/{row[k]['n']} 为正（{fmt(row[k]['arm_mean'])} 对 {fmt(row[k]['reference_mean'])}）"
    rec['text'] = (f"B1W_v2 − {ref}（同配置，同一批配对）：{part('seg35')}；{part('seg4')}；证书通过 {rec['n_certified']}/{len(ok)}。"
                   '开发集单次观测，读数不是结论；v2 分割器用训练集标签训练，不含开发受试者；基线不是同时段')
    return rec


def b1_anomalies(runs, keys):
    """The runs that did not end as class ok with a final map, a passed certificate and a passed self-check."""
    out = []
    for arm, p in keys:
        m = runs.get((arm, p))
        if m is None:
            out.append(dict(arm=arm, pair=p, what='没有 metrics.json（未运行、未跑完或基础设施失败）'))
            continue
        c = m.get('classification') or {}
        why = []
        if c.get('class') != 'ok':
            why.append(f"分类 {c.get('class')}（{c.get('kind')}：{c.get('reason')}）")
        if not m.get('metrics'):
            why.append('无终态场')
        else:
            if not _passed(m):
                why.append('证书未通过')
            if not (m.get('self_check') or {}).get('passed'):
                why.append('自检未通过')
        if why:
            out.append(dict(arm=arm, pair=p, what='；'.join(why), run_id=m.get('run_id')))
    return out


def _b1_is_oracle(arm, config):
    return arm == 'A5' or arm == B1_ORACLE_ARM or bool((config.get(arm) or {}).get('input_oracle'))


def _b1_config_text(entry):
    if not entry:
        return '—'
    parts = [f"{k} {v}" for k, v in (entry.get('changes') or {}).items()]
    return f"{entry.get('seconds')} s" + ('；' + '；'.join(parts) if parts else '')


def _b1_counts(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0)+1
    return '、'.join(f'{k}×{n}' for k, n in counts.items()) or '—'


def summarize_b1(b1_root, selection_spec, a5_root, out_dir, stage=STAGE):
    """The B1 arms (b1_root/runs, metrics.json, whatever arms are there) with the baselines A5P, A5 of a5_root -> long.csv, summary_b1.md, decision_b1.json in
    out_dir (never under a5_root).  Works with any number of runs; the selection reads the arms of selection_spec, on the first 5 pairs only."""
    out_dir = Path(out_dir)
    for root in (D0, D0A4, D0A1P, D0A5, a5_root):
        if _inside(out_dir, root):
            raise SystemExit(f'{out_dir}: the B1 summary is never written under {root}')
    pairs = [r['pair'] for r in json.loads((Path(stage)/'selection.json').read_text())['pairs']]
    first = pairs[:B1_FIRST_N]
    candidates = b1_spec.load_spec(selection_spec)
    runs_root = Path(b1_root)/'runs'
    seen = sorted(d.name for d in runs_root.iterdir() if d.is_dir()) if runs_root.is_dir() else []
    own_arms = [e['arm'] for e in candidates]+[a for a in seen if a not in {e['arm'] for e in candidates}]
    runs = chosen_runs(runs_root, tuple(own_arms))
    refs = long_csv_runs(a5_root, B1_REFERENCES, pairs)
    problems = attach_reference_labels(refs, a5_root, B1_REFERENCES)
    for arm, (seg35, seg4) in B1_PINS.items():
        values = [refs[(arm, p)]['metrics'] for p in first if (refs.get((arm, p)) or {}).get('metrics')]
        if len(values) != len(first) or abs(np.mean([v['seg35'] for v in values])-seg35) > B1_PIN_TOL or abs(np.mean([v['seg4'] for v in values])-seg4) > B1_PIN_TOL:
            problems.append(f'{arm} 前 5 对与预登记的 {seg35} / {seg4} 不符')
    for entry in candidates:                                  # a run of a candidate name made under another configuration is not this arm
        for p in pairs:
            m = runs.get((entry['arm'], p))
            b1 = (m or {}).get('b1')
            if m is not None and (not b1 or b1_spec.configuration(b1) != b1_spec.configuration(entry) or b1.get('inputs_root') != entry['inputs_root']):
                problems.append(f"{entry['arm']} {p}: metrics.json 的配置 {None if not b1 else b1_spec.configuration(b1)}、输入 {None if not b1 else b1.get('inputs_root')} 不是选配规格的 "
                                f"{b1_spec.configuration(entry)}、{entry['inputs_root']}")
    runs.update(refs)
    selection = decide_b1_selection(runs, pairs, candidates, problems)
    readings = decide_b1_readings(runs, pairs, selection, oracle_inputs=Path(a5_root)/'inputs'/'A5')
    v2_reading = decide_b1_v2(runs, pairs, selection)
    arm_pairs = {a: [p for p in pairs if (a, p) in runs] for a in own_arms}
    batch = [p for p in pairs if any((a, p) in runs for a in own_arms)]
    config = {e['arm']: e for e in candidates}
    for arm in own_arms:
        if arm not in config:
            config[arm] = next((runs[(arm, p)].get('b1') for p in arm_pairs[arm] if runs[(arm, p)].get('b1')), None)
    out_dir.mkdir(parents=True, exist_ok=True)
    moved = _set_aside(out_dir, ('long.csv', 'summary_b1.md', 'decision_b1.json'))
    own = {k: m for k, m in runs.items() if k[0] in own_arms}
    with open(out_dir/'long.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['set', 'split', 'method', 'arm', 'pair', 'measure', 'value', 'run_id'])
        w.writeheader()
        w.writerows(dict(r, method=own[(r['arm'], r['pair'])].get('method', r['method'])) for r in long_rows(own))
        for arm in B1_REFERENCES:                             # the source rows, verbatim, for the pairs of this batch
            for p in batch:
                w.writerows((runs.get((arm, p)) or {}).get('rows', []))
    all_arms = own_arms+list(B1_REFERENCES)
    refs_of = {a: (('A5', 'A5P') if a == B1_ORACLE_ARM else ('A5P',)) for a in own_arms}
    if B1_V2_ARM in refs_of and v2_reading['reference'] != 'A5P':
        refs_of[B1_V2_ARM] = (v2_reading['reference'], 'A5P')
    sets = dict(first_5=first, all_of_arm=None)

    def pairs_of(arm, which):
        return first if which == 'first_5' else arm_pairs[arm]
    comparisons, groups, structures = {}, {}, {}
    for arm in own_arms:
        for ref in refs_of[arm]:
            for which in sets:
                sel = pairs_of(arm, which)
                key = f'{arm} - {ref} ({which})'
                comparisons[key] = {}
                for k in B1_MEASURES:
                    v = list(paired_ref(runs, arm, ref, k, sel).values())
                    comparisons[key][k] = dict(mean_diff=float(np.mean(v)) if v else None, median_diff=_median(v), positive=sum(x > 0 for x in v), n=len(v))
                groups[key] = {g: a5_group_row(runs, arm, ref, labels, sel) for g, labels in A5_GROUPS.items()}
                structures[key] = {str(label): label_rows(runs, arm, ref, 'seg4', (label,), sel)[label] for label in (1, 3)}
    dec = dict(rule='runs/20261103-b1/plan.md', winner=selection['winner'], winner_spec=selection['winner_spec'], selection=selection, readings=readings, b1w_v2=v2_reading,
               reference_problems=problems, comparisons=comparisons, seg35_groups=groups, seg4_structures=structures, note=B1_NOTE,
               seg35_group_means={g: {which: {a: a5_group_mean(runs, a, labels, first if which == 'first_5' else (arm_pairs.get(a) or batch)) for a in all_arms} for which in sets}
                                  for g, labels in A5_GROUPS.items()},
               previous_summary_moved_to=None if moved is None else str(moved))
    keys = sorted({(e['arm'], p) for e in candidates for p in first} | {k for k in own}, key=lambda k: (pairs.index(k[1]), own_arms.index(k[0])))
    dec['anomalies'] = b1_anomalies(runs, keys)
    dec['sources'] = dict(b1_root=str(b1_root), selection_spec=str(selection_spec), selection_spec_sha256=sha_file(selection_spec), a5_root=str(a5_root),
                          reference_numbers='analysis/long.csv of d0_v1_a5; per-label Dice from the metrics.json of the same run_id',
                          evaluator_sha256={a: sorted({(runs[(a, p)].get('evaluator') or {}).get('sha256') or '-' for p in batch if (a, p) in runs}) for a in all_arms})
    loads = {k: launch_load(m['run_dir']) for k, m in own.items() if m.get('run_dir')}
    dec['launch_loadavg'] = {f'{a}/{p}': v for (a, p), v in loads.items()}

    def arm_table(sel_of):
        out = ['| 臂 | 配置 | 有终态场的对数 | seg35 | seg4 | HD95 | SDlogJ 官方 | 脑内 SDlogJ | 脑内 J < .5 (%) | 证书通过 | 最小下界 | 原生提交中位 | e2e 中位 s | 收尾状态 |',
               '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for arm in all_arms:
            sel = sel_of(arm)
            ms = [runs[(arm, p)] for p in sel if (arm, p) in runs]
            if not ms and arm not in config:
                continue
            ok = [m for m in ms if m.get('metrics')]
            mean = (lambda k: _a5_mean(runs, arm, sel, k))
            lower = [m['certificate'].get('min_jd_lower_bound') for m in ms if m.get('certificate') and m['certificate'].get('min_jd_lower_bound') is not None]
            out.append(f"| {arm}{B1_ORACLE_LABEL if _b1_is_oracle(arm, config) else ''} | {_b1_config_text(config.get(arm)) if arm in config else '90 s；配置 C（d0_v1_a5，不重跑）'} | {len(ok)}/{len(sel)} | {fmt(mean('seg35'))} | {fmt(mean('seg4'))} | {fmt(mean('hd95'), 3)} | "
                       f"{fmt(mean('sdlogj_official'))} | {fmt(mean('sdlogj_brain'), 3)} | {fmt(mean('pct_J_lt_half_brain'), 2)} | {sum(_passed(m) for m in ms)}/{len(ms)} | "
                       f"{f'{min(lower):.3e}' if lower else '—'} | {fmt(_median(m.get('native_commits') for m in ms), 1)} | {fmt(_median(m.get('e2e_seconds') for m in ms), 1)} | "
                       f"{_b1_counts(m['status'] for m in ms if m.get('status'))} |")
        return out
    L = ['# B1 选项臂汇总（开发集结果，单次观测）', '',
         f'生成：{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}（UTC）；计划 runs/20261103-b1/plan.md；指标由 experiments/d0/evaluate.py 从终态场算出，基线的数取 d0_v1_a5 的原始 analysis/long.csv，只比同一批配对。', '',
         '**报告时写明**：'+B1_NOTE, '']
    if problems:
        L += ['**参照与预登记或 metrics.json 不符，读数要交总负责人**：'+'；'.join(problems), '']
    if moved is not None:
        L += [f'上一版汇总移到 {moved}（保留）。', '']
    L += ['## 判定（预登记，由代码判；seg4 与 seg35 同报）', '',
          f"- **选配（前 5 对）**：{selection['text']}。winner = `{selection['winner']}`，规格 `{json.dumps(selection['winner_spec'], ensure_ascii=False)}`。",
          f"- **读数（19 对）**：{readings['text']}。",
          f"- **B1W_A5{B1_ORACLE_LABEL}（前 5 对）**：{readings['oracle']['text']}。seg35 均值 {fmt(readings['oracle']['seg35_mean'])}、seg4 均值 {fmt(readings['oracle']['seg4_mean'])}（{readings['oracle']['n_with_field']} 对有终态场，证书通过 {readings['oracle']['n_certified']}）。",
          f"- **B1W_v2（新分割器，同配置的 {v2_reading['reference']}）**：{v2_reading['text']}。",
          '- 这些是读数，不是结论，下一步交 BZ。', '',
          '| 候选臂 | 秒 | 终态场 | handover_only | 证书 | seg35 | seg4 | SDlogJ 官方 | 合格 | 不合格的原因 |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in selection['arms']:
        why = [t for t, good in (('终态场不足 5 对', r['fields_ok']), ('证书未全过', r['certificates_ok']), ('SDlogJ > .095', r['sdlogj_ok']), ('seg4 低于 A5P − .003', r['seg4_ok'])) if not good]
        L.append(f"| {r['arm']} | {r['seconds']} | {r['n_with_field']}/{len(first)} | {r['n_handover_only']} | {r['n_certified']} | {fmt(r['seg35_mean'])} | {fmt(r['seg4_mean'])} | {fmt(r['sdlogj_official_mean'])} | {'是' if r['qualified'] else '否'} | {'、'.join(why) or '—'} |")
    L += ['', f"A5P 同 5 对：seg35 {fmt(selection['baseline']['seg35_mean'])}、seg4 {fmt(selection['baseline']['seg4_mean'])}。", '',
          f"## B1W_v2 对同配置的 {v2_reading['reference']}（读数，seg35 与 seg4 同报）", '', v2_reading['text']+'。', '',
          '## 前 5 对的各臂均值（seg4 与 seg35 同报）', '', *arm_table(lambda a: first)]
    if any(len(v) > B1_FIRST_N for v in arm_pairs.values()):
        L += ['', '## 各臂自己跑过的全部配对（A5P、A5 为 19 对）', '', *arm_table(lambda a: arm_pairs.get(a) or pairs)]
    L += ['', '## 逐对差（前者 − 后者，只比两臂都有终态场的配对；first_5 是选配用的 5 对）', '', '| 比较 | 指标 | 均值差 | 中位差 | 为正的对数 / 有效对数 |', '|---|---|---|---|---|']
    for key, row in comparisons.items():
        for k in B1_MEASURES:
            r = row[k]
            L.append(f"| {key} | {k} | {fmt(r['mean_diff'])} | {fmt(r['median_diff'])} | {r['positive']} / {r['n']} |")
    L += ['', '## seg35 按标签组（每对先在组内平均，再对配对平均；前 5 对）', '', '| 组 | ' + ' | '.join(all_arms) + ' |', '|---|' + '---|'*len(all_arms)]
    for g in A5_GROUPS:
        L.append(f"| {A5_GROUP_NAMES[g]} | " + ' | '.join(fmt(dec['seg35_group_means'][g]['first_5'][a]) for a in all_arms) + ' |')
    L += ['', '| 比较 | 组 | 均值差 | 中位差 | 为正的对数 / 有效对数 | 对 macro 差的贡献（标签均值差之和 / 35） |', '|---|---|---|---|---|---|']
    for key, row in groups.items():
        for g in A5_GROUPS:
            r = row[g]
            L.append(f"| {key} | {A5_GROUP_NAMES[g]} | {fmt(r['mean_diff'])} | {fmt(r['median_diff'])} | {r['positive']} / {r['n']} | {fmt(r['contribution_to_macro'])} |")
    L += ['', '## seg4 的皮层与白质', '', '| 比较 | 结构 | 臂均值 | 基线均值 | 均值差 | 中位差 | 为正的对数 / 有效对数 |', '|---|---|---|---|---|---|---|']
    for key, row in structures.items():
        for label in ('1', '3'):
            r = row[label]
            L.append(f"| {key} | {SEG4_STRUCTURES[int(label)]} | {fmt(r['arm_mean'])} | {fmt(r['ref_mean'])} | {fmt(r['mean_diff'])} | {fmt(r['median_diff'])} | {r['positive']} / {r['n']} |")
    L += ['', '## 逐对 seg35（seg4）', '', '| 对 | ' + ' | '.join(all_arms) + ' |', '|---|' + '---|'*len(all_arms)]
    for p in batch:
        cells = []
        for arm in all_arms:
            m = runs.get((arm, p))
            cells.append('—' if not m or not m.get('metrics') else f"{m['metrics']['seg35']:.4f} ({m['metrics']['seg4']:.4f})")
        L.append(f"| {p[11:15]}←{p[-8:-4]} | " + ' | '.join(cells) + ' |')
    L += ['', '## 每次运行', '', '| 运行 | run_id | 分类 | 收尾状态 | 证书 | 最小下界 | 原生提交 | e2e s | 启动负载（1/5 分钟） |', '|---|---|---|---|---|---|---|---|---|']
    for arm, p in keys:
        m = runs.get((arm, p))
        if m is None:
            L.append(f'| {arm} {p[11:15]}←{p[-8:-4]} | — | 没有 metrics.json | — | — | — | — | — | — |')
            continue
        c = m.get('certificate') or {}
        load = loads.get((arm, p))
        L.append(f"| {arm} {p[11:15]}←{p[-8:-4]} | {m.get('run_id')} | {(m.get('classification') or {}).get('class')} | {m.get('status')} | {'通过' if _passed(m) else '未通过'} | "
                 f"{'—' if c.get('min_jd_lower_bound') is None else format(c['min_jd_lower_bound'], '.3e')} | {m.get('native_commits')} | {fmt(m.get('e2e_seconds'), 1)} | "
                 f"{'—' if not load else f'{load[0]:.2f} / {load[1]:.2f}'} |")
    L += ['', '## 受阻或异常的运行', '']
    L += [f"- {a['arm']} {a['pair']}：{a['what']}" for a in dec['anomalies']] or ['- 没有。']
    L += ['', '**评测脚本版本**：各臂 metrics.json 记录的 experiments/d0/evaluate.py sha256 为 '
          + '；'.join(f"{arm} {', '.join(s[:12] for s in dec['sources']['evaluator_sha256'][arm]) or '—'}" for arm in all_arms) + '。']
    (out_dir/'summary_b1.md').write_text('\n'.join(L)+'\n')
    (out_dir/'decision_b1.json').write_text(json.dumps(dec, indent=1, ensure_ascii=False, default=float)+'\n')
    print('\n'.join(L))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('run', type=Path)
    r.add_argument('--arm', required=True, choices=ARMS+SUPPLEMENTARY+A1P_ARMS+X1_ARMS+A5_ARMS)
    r.add_argument('--pair', required=True); r.add_argument('--run-id', required=True); r.add_argument('--out', type=Path)
    s = sub.add_parser('summarize'); s.add_argument('--runs-root', type=Path, default=D0/'runs')
    s.add_argument('--out-dir', type=Path, default=D0/'analysis')
    s4 = sub.add_parser('summarize-a4'); s4.add_argument('--a4-runs-root', type=Path, default=D0A4/'runs')
    s4.add_argument('--d0-runs-root', type=Path, default=D0/'runs'); s4.add_argument('--out-dir', type=Path, default=D0A4/'analysis')
    s1 = sub.add_parser('summarize-a1p'); s1.add_argument('--a1p-runs-root', type=Path, default=D0A1P/'runs')
    s1.add_argument('--d0-runs-root', type=Path, default=D0/'runs'); s1.add_argument('--a4-runs-root', type=Path, default=D0A4/'runs')
    s1.add_argument('--out-dir', type=Path, default=D0A1P/'analysis')
    sx = sub.add_parser('summarize-x1'); sx.add_argument('--x1-runs-root', type=Path, default=GAX1/'runs')
    sx.add_argument('--d0-runs-root', type=Path, default=D0/'runs'); sx.add_argument('--pilot-root', type=Path, default=PILOT)
    sx.add_argument('--out-dir', type=Path, default=GAX1/'analysis')
    s5 = sub.add_parser('summarize-a5'); s5.add_argument('--a5-runs-root', type=Path, default=D0A5/'runs')
    s5.add_argument('--d0-root', type=Path, default=D0); s5.add_argument('--a4-root', type=Path, default=D0A4)
    s5.add_argument('--a1p-root', type=Path, default=D0A1P); s5.add_argument('--out-dir', type=Path, default=D0A5/'analysis')
    rb = sub.add_parser('run-b1'); rb.add_argument('run', type=Path); rb.add_argument('--spec', type=Path, required=True)
    rb.add_argument('--arm', required=True); rb.add_argument('--pair', required=True); rb.add_argument('--run-id', required=True); rb.add_argument('--out', type=Path)
    sb = sub.add_parser('summarize-b1'); sb.add_argument('--b1-root', type=Path, default=B1); sb.add_argument('--selection-spec', type=Path, default=B1_SELECTION_SPEC)
    sb.add_argument('--a5-root', type=Path, default=D0A5); sb.add_argument('--out-dir', type=Path, default=B1/'analysis')
    a = ap.parse_args(argv)
    if a.cmd == 'summarize-a4':
        return summarize_a4(a.a4_runs_root, a.d0_runs_root, a.out_dir)
    if a.cmd == 'summarize-a1p':
        return summarize_a1p(a.a1p_runs_root, a.d0_runs_root, a.a4_runs_root, a.out_dir)
    if a.cmd == 'summarize-x1':
        return summarize_x1(a.x1_runs_root, a.d0_runs_root, a.pilot_root, a.out_dir)
    if a.cmd == 'summarize-a5':
        return summarize_a5(a.a5_runs_root, a.d0_root, a.a4_root, a.a1p_root, a.out_dir)
    if a.cmd == 'summarize-b1':
        return summarize_b1(a.b1_root, a.selection_spec, a.a5_root, a.out_dir)
    if a.cmd in ('run', 'run-b1'):
        if a.cmd == 'run':
            rec = evaluate_run(a.run, a.arm, a.pair, a.run_id)
        else:
            entry = b1_spec.arm_entry(a.spec, a.arm)
            rec = evaluate_run(a.run, a.arm, a.pair, a.run_id, method=b1_spec.method_name(entry))
            rec['b1'] = dict(entry, input_oracle=b1_spec.is_oracle(entry), spec=str(a.spec), spec_sha256=sha_file(a.spec))
        out = a.out or a.run/'metrics.json'
        out.write_text(json.dumps(rec, indent=1, default=float)+'\n')
        m = rec.get('metrics') or {}
        c = rec.get('certificate') or {}
        print(f"{a.arm} {a.pair[11:15]}<-{a.pair[-8:-4]}: seg35 {fmt(m.get('seg35'))} seg4 {fmt(m.get('seg4'))} "
              f"HD95 {fmt(m.get('hd95'), 3)} SDlogJ {fmt(m.get('sdlogj_official'))} brain {fmt(m.get('sdlogj_brain'), 3)} "
              f"cert {c.get('status')} passed {c.get('passed')} self-check {(rec.get('self_check') or {}).get('passed')} "
              f"e2e {fmt(rec.get('e2e_seconds'), 1)} s")
        return 0
    return summarize(a.runs_root, a.out_dir)


if __name__ == '__main__':
    sys.exit(main())
