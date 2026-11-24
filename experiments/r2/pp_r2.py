"""R2 step 1, the pp diagnosis (runs/20261110-r2/plan.md section 2): on HS's 19 certified final fields (hs_v1, Hx24-RF05-F1200 on unet35_v4_ens's channels), the moving image's v4 predicted labels
pulled back by the field and compared, label by label, with the fixed image's v4 predicted labels.  NO truth label is read: no pair.npz member but fixed, moving, initial_u (the firewall's load_pair), no label
file, no evaluation record; every file the unit opens is recorded (opened_files.json) by PQ's spy (experiments/pq/pq_firewall.py), which stops the unit at once on anything else.

    python -B experiments/r2/pp_r2.py --dry-run
        CPU; no field, no label, no prediction read; nothing written: the committed HS reference against its pin, the 19 pairs' record files there, the start string, the estimates.
    python -B experiments/r2/pp_r2.py pp              (unit tag-r2-pp, CPU; PYTHONPATH=<frozen>:<code>, the taper of the frozen tree)
        under the spy: m1 (HS's certified state v of each pair, bound to the committed reference: field_0600.npz, repair.npz, run.json, repair.json by sha256, v by the certificate's state_hash;
        bind_hs_field: pq_io.bind_field with the segmenter checked on HS's run.json keys), m2/m3 (the v4 labels of each side through the structure layer and the method's reader, equal bit for bit to the T88 run's labels_sha256), then pp per label
        (scripts/evaluate_l2r_style.py: warp_labels_l2r, dice_l2r with the predicted labels in place of the truth).  Writes <PP_ROOT>/pp/<pair>.json, pp_r2.json, opened_files.json; once.
    python -B experiments/r2/pp_r2.py split           (the lead, CPU; after the lead has read opened_files.json)
        the split of each label's error, read with records that exist already (no truth file; numbers read before): HS's per-label Dice (repair_metrics.json, pinned by the reference) and UF4's
        segmentation-level per-label Dice (eval_uf.json, pinned).  pp_split.json, targets.json (the target rule of plan.md section 2, mechanical), pp_report.md; once.

R3b (runs/20261103-n0 (c)): 1 - D ~ sqrt(a^2 + b^2), a = 1 - pp (not aligned), b = sqrt(2) (1 - s) (wrong).  It is a quadrature model that has underestimated the error before (A5P by .0105, B1TM by .0134;
the closer the field follows the predicted labels, the more): the split is an order-of-magnitude reference, not an additive decomposition (gap analysis review, important 6).
Exit codes: 0 done, 2 usage, 8 refused or stopped.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (HERE, HERE.parent/'pq', HERE.parent/'ef', HERE.parent/'mc0', REPO/'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

import numpy as np  # noqa: E402

import r2_common as rc  # noqa: E402

UNIT = 'tag-r2-pp'
SCRIPT = 'experiments/r2/pp_r2.py'
MEMORY_MAX = '11G'
MEM_AVAILABLE_GIB = 9.
HOME_FREE_GB = 100.
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
PP_DIR, PP_FILE, OPENED_FILE = 'pp', 'pp_r2.json', 'opened_files.json'
SPLIT_FILE, TARGETS_FILE, REPORT_FILE = 'pp_split.json', 'targets.json', 'pp_report.md'
GROUPS_FILE_REL = 'configs/bayesreg/structure_groups_v1.json'
TIERS = dict(field=('SHx24_first5', 'SHx24_rest600'), repair=('RSHx24_c600', 'RSHx24_rest600'))


class PPRefused(Exception):
    def __init__(self, stage, message):
        super().__init__(f'{stage}: {message}')
        self.stage = stage


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ---- the committed HS reference: the 19 pairs' rows -------------------------------------------------------------------------------------------------
def reference_rows(repo=REPO, *, pinned=rc.HS_REFERENCE_SHA256, hs_root=rc.HS_ROOT):
    """(rows {short: P3Row}, run_json {short: sha256}, repair_metrics {short: sha256}, record) from experiments/p2/reference_unet35_v4_ens.json, the file pinned by its sha256: every pair hx_certified, the
    directories hs_v1's, the reading 采用新分割器 with decision_hs.json's pinned sha256.  PPRefused('reference') otherwise.  Only this JSON file is read."""
    import pq_pins as pins
    path = Path(repo)/rc.HS_REFERENCE_REL
    if not path.is_file():
        raise PPRefused('reference', f'{path}: no such file')
    digest = rc.sha256_file(path)
    if pinned is not None and digest != pinned:
        raise PPRefused('reference', f'{path}: sha256 {digest} is not the pinned {pinned}')
    document = json.loads(path.read_text(encoding='utf-8'))
    problems = []
    hs = document.get('hs') or {}
    if document.get('segmenter_id') != rc.V4_SEGMENTER or hs.get('reading') != '采用新分割器' or hs.get('decision_sha256') != rc.HS_DECISION_SHA256:
        problems.append(f"segmenter {document.get('segmenter_id')}, reading {hs.get('reading')}, decision {hs.get('decision_sha256')}")
    pairs = (document.get('p3') or {}).get('pairs') or {}
    if sorted(pairs) != sorted(rc.SHORTS):
        problems.append(f'the pairs are {sorted(pairs)}')
    rows, run_json, metrics = {}, {}, {}
    runs = Path(hs_root)/'runs'
    for short in rc.SHORTS:
        p = pairs.get(short) or {}
        field_dir, repair_dir = Path(p.get('field_dir') or '/'), Path(p.get('repair_dir') or '/')
        t88, rep = field_dir.parent.name, repair_dir.parent.name
        if p.get('hs_line_status') != 'hx_certified' or p.get('certified_at_1') is not True:
            problems.append(f"{short}: line status {p.get('hs_line_status')}, certified {p.get('certified_at_1')}")
        if field_dir != runs/t88/short or repair_dir != runs/rep/short or t88 not in TIERS['field'] or rep not in TIERS['repair']:
            problems.append(f'{short}: directories {field_dir}, {repair_dir} are not hs_v1/runs/<tier>/{short}')
        keys = ('attempt_sha256', 'repaired_u_sha256', 'field_sha256', 'repair_json_sha256', 'run_json_sha256', 'repair_metrics_sha256')
        if any(not (isinstance(p.get(k), str) and re.fullmatch('[0-9a-f]{64}', p[k])) for k in keys):
            problems.append(f'{short}: a sha256 is missing')
            continue
        rows[short] = pins.P3Row(t88, rep, repair_npz=p['attempt_sha256'], state_hash=p['repaired_u_sha256'], field_npz=p['field_sha256'], repair_json=p['repair_json_sha256'])
        run_json[short], metrics[short] = p['run_json_sha256'], p['repair_metrics_sha256']
    if problems:
        raise PPRefused('reference', '; '.join(problems))
    return rows, run_json, metrics, dict(path=rc.HS_REFERENCE_REL, sha256=digest, decision_sha256=hs.get('decision_sha256'))


def _context_class():
    import pq_io as io_

    @dataclass
    class PPContext(io_.Context):
        """pq_io.Context with the pp root in the place of PQ's and the T88 and repair directories under runs_root (hs_v1/runs in production)."""
        pp_dir: Path = None
        runs_root: Path = None

        @property
        def pq_root(self):
            return Path(self.pp_dir)

        def t88_dir(self, short):
            return Path(self.runs_root)/self.p3[short].t88_dir/short

        def repair_dir(self, short):
            return Path(self.runs_root)/self.p3[short].repair_dir/short
    return PPContext


def make_context(*, repo=REPO, source_root=rc.MAIN_REPO, results=rc.RESULTS, pp_root=rc.PP_ROOT, taper=None, pinned=rc.HS_REFERENCE_SHA256):
    """The context of the unit on HS's records: the rows of the committed reference, the T88 and repair directories under hs_v1/runs, PQ's pins of the v4 predictions and of the anchor (the same files)."""
    rows, run_json, _, _ = reference_rows(repo, pinned=pinned, hs_root=Path(results)/'hs_v1')
    return _context_class()(repo=Path(repo), source_root=Path(source_root), results=Path(results), p3=rows, run_json=run_json, taper=taper, pp_dir=Path(pp_root), runs_root=Path(results)/'hs_v1'/'runs')


def rules_for(ctx):
    """PQ's rules for the out unit (C22.Q1-3) on HS's directories: allowed the 19 pairs' manifest.json and pair.npz (fixed, moving, initial_u only), the v4 predictions of the 20 subjects, hs_v1/runs/<tier>/<pair>/
    {run.json, field_0600.npz} and {repair.json, repair.npz}, configs/bayesreg/, the sealed lists the guard reads, the pp root (written); forbidden: label files, metrics_*.json, repair_metrics.json, analysis/,
    decision*, summary*, the read locks and what they protect, result documents and reviews."""
    import pq_firewall as fw
    import pq_pins as pins
    return fw.Rules(repo=ctx.repo, results=ctx.results, data=Path(ctx.source_root)/'data', images=ctx.images, pairs=[pins.pair_name(s) for s in ctx.shorts], subjects=ctx.subjects(),
                    p3_dirs={s: (ctx.p3[s].t88_dir, ctx.p3[s].repair_dir) for s in ctx.shorts}, unit='out', pq_root=ctx.pq_root, pred_root=ctx.pred_root, p3_root=ctx.runs_root,
                    extra_allowed=[Path(ctx.repo)/rc.PLAN_REL])


# ---- m1 on HS's records: pq_io.bind_field with the segmenter on HS's run.json keys ------------------------------------------------------------------------
HS_RUN_SCHEMA = 'bayesreg.fl.run.v1'
SIDES = ('fixed', 'moving')


def _same_path(recorded, expected):
    return isinstance(recorded, str) and Path(recorded).resolve() == Path(expected).resolve()


def hs_run_problems(run, ctx, short):
    """m1's segmenter check on HS's run.json (hs_v1, schema bayesreg.fl.run.v1).  P3's run.json, the one pq_io.bind_field was written for, has a top-level segmenter_id; HS's has none: its
    segmenter is under channel_sources -- segmenter (segmenter_id, registry, pred_root), structure_layer.registry (path, sha256, status), sides.fixed / sides.moving (subject, segmenter_id, policy,
    registry_sha256, unet_directory, unet_provenance_sha256, unet_output_sha256 of probs36.npz and structure_groups.npz), oracle and truth_labels_read.  "The field of this pair is from an HS run of the
    v4 ensemble unet35_v4_ens": every one of these against the anchor (segmenter_id, policy, the registry's name and sha256), the v4 prediction root and PQ's pins of the 20 subjects' v4 files.  The list
    of problems; empty when it passes.  Reads nothing (the caller has read run.json against its pin)."""
    import pq_pins as pins
    anchor, problems = ctx.anchor, []
    sources = run.get('channel_sources') if isinstance(run.get('channel_sources'), dict) else {}
    segmenter = sources.get('segmenter') if isinstance(sources.get('segmenter'), dict) else {}
    registry = ((sources.get('structure_layer') or {}).get('registry') or {}) if isinstance(sources.get('structure_layer'), dict) else {}
    sides = sources.get('sides') if isinstance(sources.get('sides'), dict) else {}
    if run.get('schema') != HS_RUN_SCHEMA:
        problems.append(f"run.json schema {run.get('schema')!r} is not HS's {HS_RUN_SCHEMA}")
    if sources.get('oracle') is not False or sources.get('truth_labels_read') is not False:
        problems.append(f"run.json channel_sources oracle {sources.get('oracle')!r}, truth_labels_read {sources.get('truth_labels_read')!r}")
    if segmenter.get('segmenter_id') != anchor['segmenter_id']:
        problems.append(f"run.json channel_sources.segmenter.segmenter_id {segmenter.get('segmenter_id')!r}")
    path = segmenter.get('registry')
    if not (isinstance(path, str) and Path(path).as_posix().endswith('/'+anchor['registry'])):
        problems.append(f"run.json channel_sources.segmenter.registry {path!r} is not a {anchor['registry']}")
    if registry.get('sha256') != anchor['registry_sha256'] or registry.get('status') != 'registered' or registry.get('path') != path:
        problems.append(f"run.json channel_sources.structure_layer.registry: sha256 {registry.get('sha256')!r}, status {registry.get('status')!r}, path {registry.get('path')!r}")
    if not _same_path(segmenter.get('pred_root'), ctx.pred_root):
        problems.append(f"run.json channel_sources.segmenter.pred_root {segmenter.get('pred_root')!r} is not {ctx.pred_root}")
    for side, subject in zip(SIDES, pins.subjects_of(short)):
        s = sides.get(side) if isinstance(sides.get(side), dict) else {}
        want, outputs = ctx.v4.get(subject), s.get('unet_output_sha256') or {}
        if s.get('subject') != subject or s.get('segmenter_id') != anchor['segmenter_id'] or s.get('policy') != anchor['policy'] or s.get('registry_sha256') != anchor['registry_sha256']:
            problems.append(f"run.json channel_sources.sides.{side}: subject {s.get('subject')!r}, segmenter_id {s.get('segmenter_id')!r}, policy {s.get('policy')!r}, registry_sha256 {s.get('registry_sha256')!r}")
        if not _same_path(s.get('unet_directory'), Path(ctx.pred_root)/subject):
            problems.append(f"run.json channel_sources.sides.{side}.unet_directory {s.get('unet_directory')!r} is not {Path(ctx.pred_root)/subject}")
        if want is None or (s.get('unet_provenance_sha256'), outputs.get('probs36.npz'), outputs.get('structure_groups.npz')) != (want.provenance, want.probs36, want.structure_groups):
            problems.append(f'run.json channel_sources.sides.{side}: the sha256 of {subject}\'s provenance.json, probs36.npz, structure_groups.npz are not the pinned v4 files')
    return problems


def bind_hs_field(ctx, short, grid):
    """m1 for one pair on HS's records: pq_io.bind_field (commit cd54688) statement for statement, but for its one segmenter statement -- `need(run.get('segmenter_id') == ...)`, P3's top-level key, which
    HS's run.json does not have (pp's second start stopped there: PQRefused m1 "run.json segmenter None", 11-11 02:13:33 AM CST, decision_log) -- in whose place hs_run_problems checks the segmenter on
    HS's keys.  tests/test_bayesreg/r2/test_r2_pp.py compares the two functions' statements (the AST) after the bindings below; pq_io itself is not changed (PQH's hold-out unit uses it)."""
    import hashlib
    import pq_io as io_
    PQRefused, sha_file, read_json, sha_array, certificate_problems, pins, HX_CERTIFIED = io_.PQRefused, io_.sha_file, io_.read_json, io_.sha_array, io_.certificate_problems, io_.pins, io_.HX_CERTIFIED
    row = ctx.p3[short]
    t88, rep = ctx.t88_dir(short), ctx.repair_dir(short)
    problems = []

    def need(ok, text):
        if not ok:
            problems.append(text)
    files = dict(run_json=t88/'run.json', field=t88/'field_0600.npz', repair_json=rep/'repair.json', repair_npz=rep/'repair.npz')
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise PQRefused('m1', f'{short}: missing {missing}')
    digests = {k: sha_file(p) for k, p in files.items()}
    need(digests['run_json'] == ctx.run_json[short], f"run.json sha256 {digests['run_json']} is not the plan's {ctx.run_json[short]}")
    need(digests['repair_json'] == row.repair_json, f"repair.json sha256 {digests['repair_json']} is not the plan's {row.repair_json}")
    need(digests['field'] == row.field_npz, f"field_0600.npz sha256 {digests['field']} is not the plan's {row.field_npz}")
    need(digests['repair_npz'] == row.repair_npz, f"repair.npz sha256 {digests['repair_npz']} is not the plan's {row.repair_npz}")
    run, repair = read_json(files['run_json']), read_json(files['repair_json'])
    pair = pins.pair_name(short)
    need(run.get('pair') == pair and repair.get('pair') == pair, f"the records are of {run.get('pair')}, {repair.get('pair')}, not {pair}")
    need(run.get('status') == 'done' and run.get('oracle') is False, f"run.json status {run.get('status')!r}, oracle {run.get('oracle')!r}")
    problems.extend(hs_run_problems(run, ctx, short))
    need((run.get('field') or {}).get('file') == 'field_0600.npz' and (run.get('field') or {}).get('sha256') == digests['field'], f"run.json field {run.get('field')}")
    source = repair.get('source') or {}
    need(source.get('field_sha256') == digests['field'] and source.get('file') == 'field_0600.npz', f"repair.json source {source.get('file')} {source.get('field_sha256')}")
    need(source.get('directory') is not None and Path(source['directory']).resolve() == t88.resolve(), f"repair.json source.directory {source.get('directory')} is not {t88}")
    need(repair.get('status') == 'certified' and repair.get('passed') is True and repair.get('oracle') is False, f"repair.json status {repair.get('status')!r}, passed {repair.get('passed')!r}, oracle {repair.get('oracle')!r} (only {HX_CERTIFIED} in P3)")
    need((repair.get('attempt') or {}).get('sha256') == digests['repair_npz'] and (repair.get('attempt') or {}).get('file') == 'repair.npz', f"repair.json attempt {repair.get('attempt')}")
    cert = repair.get('certificate') or {}
    need(not certificate_problems(cert), f'the certificate: {certificate_problems(cert)}')
    need(cert.get('state_hash') == row.state_hash, f"certificate.state_hash {cert.get('state_hash')} is not the plan's {row.state_hash}")
    if problems:
        raise PQRefused('m1', f'{short}: {problems}')
    with np.load(files['field'], allow_pickle=False) as z:
        raw = np.array(z['u'], dtype=np.float64)
    if raw.shape != tuple(grid.shape)+(3,):
        raise PQRefused('m1', f'{short}: field_0600.npz u has shape {raw.shape}, the grid {tuple(grid.shape)}')
    tapered, _ = ctx.taper(raw, grid, pins.TAPER_L)
    v = np.array(tapered, dtype=np.float64, order='C', copy=True)
    tapered_sha = sha_array(v)
    if tapered_sha != source.get('tapered_u_sha256'):
        raise PQRefused('m1', f"{short}: the tapered field {tapered_sha} is not source.tapered_u_sha256 {source.get('tapered_u_sha256')}")
    with np.load(files['repair_npz'], allow_pickle=False) as z:
        nodes, values = np.array(z['nodes']), np.array(z['values'])
    if values.shape != (nodes.size, 3) or nodes.size != (repair.get('attempt') or {}).get('n_free_nodes', nodes.size):
        raise PQRefused('m1', f'{short}: repair.npz nodes {nodes.shape}, values {values.shape}')
    v.reshape(-1, 3)[nodes] = values
    state = hashlib.sha256(memoryview(v).cast('B')).hexdigest()
    if state != cert['state_hash']:
        raise PQRefused('m1', f"{short}: sha256 of v {state} is not certificate.state_hash {cert['state_hash']}")
    record = dict(line_status=HX_CERTIFIED, run_json=str(files['run_json']), run_json_sha256=digests['run_json'], field=str(files['field']), field_sha256=digests['field'],
                  repair_json=str(files['repair_json']), repair_json_sha256=digests['repair_json'], repair_npz=str(files['repair_npz']), repair_npz_sha256=digests['repair_npz'],
                  tapered_u_sha256=tapered_sha, state_hash=state, certificate={k: cert.get(k) for k in ('status', 'reason', 'min_jd_lower_bound', 'n_cells_checked', 'n_infeasible', 'n_uncertain', 'certificate_version')},
                  n_free_nodes=int(nodes.size), taper=pins.TAPER_L, labels_sha256=run.get('labels_sha256'))
    return v, record


# ---- one side's labels (m2, m3 without the posterior) ------------------------------------------------------------------------------------------------
def predicted_labels(ctx, layer, pin, side, labels_sha256):
    """(L int16, record): pq_io.read_side's checks (the structure layer's provenance and hashes, PQ's pins of the v4 files, the anchor, trained_with_labels true and test_labels_seen false) and the
    method's reader (fast_channels.raw_labels) for L, equal to the T88 run's labels_sha256 of that side; and, as read_side (review m3), the argmax of the read-back posterior
    (bayesreg.seg.channels.read_back_probabilities, ties to the smallest label) equal to L bit for bit.  The posterior is dropped after that check (pp needs the labels only)."""
    import fast_channels
    import mc0_core
    import pq_io as io_
    from bayesreg.io.pair import ProvenanceRejected
    from jd_curl.validation import InputValidationError
    subject = layer.subjects[side]
    try:
        groups, accepted, rec = layer.load(side)
    except (ProvenanceRejected, InputValidationError) as exc:
        raise PPRefused('m2', f'{side} {subject}: {type(exc).__name__}: {exc}') from None
    directory = Path(rec['directory'])
    prov = io_.read_json(directory/'provenance.json')
    want = ctx.v4.get(subject)
    problems = []
    if want is None:
        problems.append(f'{subject} is not one of the 20 subjects')
    else:
        for name, expected, have in (('provenance.json', want.provenance, rec['provenance_sha256']), ('probs36.npz', want.probs36, rec['verified_outputs'].get('probs36.npz')),
                                     ('structure_groups.npz', want.structure_groups, rec['verified_outputs'].get('structure_groups.npz'))):
            if have != expected:
                problems.append(f'{name} sha256 {have} is not the pinned {expected}')
    for key in ('segmenter_id', 'policy', 'registry_sha256'):
        if prov.get(key) != ctx.anchor[key]:
            problems.append(f'provenance {key} {prov.get(key)!r} is not the anchor\'s')
    if prov.get('trained_with_labels') is not True or prov.get('test_labels_seen') is not False:
        problems.append(f"trained_with_labels {prov.get('trained_with_labels')!r}, test_labels_seen {prov.get('test_labels_seen')!r}")
    if problems:
        raise PPRefused('m2', f'{side} {subject}: {problems}')
    g = layer.grouping
    try:
        fast = fast_channels.raw_labels(directory/'probs36.npz', (36,)+tuple(pin.grid.shape), groups, (g.background_labels,)+tuple(g.group_labels))
    except SystemExit as exc:
        raise PPRefused('m3', f'{side} {subject}: the method\'s reader refused: {exc}') from None
    del groups
    if fast is None:
        raise PPRefused('m3', f'{side} {subject}: raw_labels gives None (not the fast path)')
    L, _, _, deviation = fast
    if not deviation <= mc0_core.make_inputs().GROUP_TOL:
        raise PPRefused('m3', f'{side} {subject}: probs36 by group differs from structure_groups.npz by {deviation:.3e}')
    from bayesreg.seg.channels import read_back_probabilities
    try:
        P, _ = read_back_probabilities(directory/'probs36.npz', shape=(36,)+tuple(pin.grid.shape))
    except InputValidationError as exc:
        raise PPRefused('m3', f'{side} {subject}: {exc}') from None
    top = io_.core.argmax_smallest(np.asarray(P))
    del P
    if top.dtype != L.dtype or top.tobytes() != np.ascontiguousarray(L).tobytes():
        raise PPRefused('m3', f'{side} {subject}: argmax of the read-back posterior is not raw_labels\' L bit for bit')
    del top
    digest = io_.sha_array(L)
    if digest != (labels_sha256 or {}).get(side):
        raise PPRefused('m3', f'{side} {subject}: sha256 of L {digest} is not the T88 run.json labels_sha256')
    return L, dict(subject=subject, provenance_sha256=rec['provenance_sha256'], probs36_sha256=rec['verified_outputs'].get('probs36.npz'), labels_sha256=digest, group_max_abs_deviation=deviation,
                   accepted_model_ids=(accepted or {}).get('model_ids'))


def pp_of_labels(L_F, L_M, d):
    """(per-label pp, labels 1..35, NaN where a label is absent from L_F or from the unwarped L_M; seg35 nanmean): the evaluation's own functions with the predicted labels in place of the truth."""
    from evaluate_l2r_style import dice_l2r, warp_labels_l2r
    warped = warp_labels_l2r(np.asarray(L_M), d)
    mean, per = dice_l2r(np.asarray(L_F), np.asarray(L_M), warped)
    return [float(v) for v in per], float(mean)


def group_means(per):
    out = {}
    for name, labels in (('small_six', rc.SMALL_SIX), ('big_six', rc.BIG_SIX), ('other_23', rc.OTHER_23)):
        values = [per[k-1] for k in labels if per[k-1] is not None and math.isfinite(per[k-1])]
        out[name] = float(np.mean(values)) if values else None
    return out


# ---- the machine ---------------------------------------------------------------------------------------------------------------------------------
def in_own_unit(cgroup='/proc/self/cgroup'):
    try:
        text = Path(cgroup).read_text(encoding='utf-8')
    except OSError:
        return False
    return re.search(rf'/{re.escape(UNIT)}\.service(/|$)', text, re.MULTILINE) is not None


def machine_gate(*, run=subprocess.run, meminfo='/proc/meminfo', disk=None):
    """No other active tag-* / seq-vp-* unit, /home >= 100 GB, C: >= 40 GiB (the brief), MemAvailable >= 9 GiB.  PPRefused('machine') otherwise; the record."""
    import shutil
    disk = shutil.disk_usage if disk is None else disk
    record = dict(unit=UNIT, checked_utc=utc())
    try:
        listing = run(['systemctl', '--user', 'list-units', '--state=active,activating', '--no-legend', '--plain', 'tag-*', 'seq-vp-*'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PPRefused('machine', f'systemctl could not be queried ({exc})') from None
    if listing.returncode != 0:
        raise PPRefused('machine', f'systemctl failed ({listing.stderr.strip()})')
    others = sorted(line.split()[0] for line in listing.stdout.splitlines() if line.strip() and line.split()[0].removesuffix('.service') != UNIT)
    available = 0.
    for line in Path(meminfo).read_text(encoding='utf-8').splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1])/2**20
    home, c_drive = disk('/home').free/1e9, disk('/mnt/c').free/2**30
    record.update(other_units=others, home_free_gb=home, c_free_gib=c_drive, mem_available_gib=available)
    problems = ([f'other units are active: {others}'] if others else [])+([f'/home {home:.0f} GB < {HOME_FREE_GB:.0f}'] if home < HOME_FREE_GB else [])+([f'C: {c_drive:.1f} GiB < {rc.C_FLOOR_GIB:.0f}'] if c_drive < rc.C_FLOOR_GIB else [])+([f'MemAvailable {available:.1f} GiB < {MEM_AVAILABLE_GIB}'] if available < MEM_AVAILABLE_GIB else [])
    if problems:
        raise PPRefused('machine', '; '.join(problems))
    return dict(record, passed=True)


def code_state(repo=REPO, *, frozen_required=True):
    """The checkout's HEAD and cleanliness, bayesreg and this package from it, jd_curl from the frozen tree (mc0_core.frozen_api).  PPRefused('code') otherwise."""
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, timeout=60)
    head, status = git('rev-parse', 'HEAD'), git('status', '--porcelain')
    import bayesreg
    record = dict(root=str(repo), commit=head.stdout.strip() or None, clean=head.returncode == 0 and status.returncode == 0 and not status.stdout.strip(), bayesreg=str(Path(bayesreg.__file__).resolve().parent),
                  files={p.name: rc.sha256_file(p) for p in sorted(HERE.glob('*.py'))})
    problems = [] if record['clean'] else [f'the checkout {repo} is not clean']
    if not Path(record['bayesreg']).is_relative_to(Path(repo).resolve()):
        problems.append(f"bayesreg comes from {record['bayesreg']}")
    if frozen_required:
        import mc0_core
        try:
            mc0_core.frozen_api()
            record['frozen'] = dict(root=str(mc0_core.FROZEN), commit=mc0_core.FROZEN_COMMIT)
        except SystemExit as exc:
            problems.append(f'the frozen tree: {exc}')
    if problems:
        raise PPRefused('code', '; '.join(problems))
    return record


def preload():
    """Every module the unit calls, imported before the spy (an import that reads a file is not the unit's reading)."""
    import bayesreg.io.guard  # noqa: F401
    import bayesreg.io.pair  # noqa: F401
    import bayesreg.seg.channels  # noqa: F401
    import bayesreg.seg.structure_layer  # noqa: F401
    import jd_curl.three_d.nifti  # noqa: F401
    import jd_curl.validation  # noqa: F401
    import evaluate_l2r_style  # noqa: F401
    import fast_channels  # noqa: F401
    import mc0_core
    import pq_io  # noqa: F401
    mc0_core.make_inputs()


# ---- the unit --------------------------------------------------------------------------------------------------------------------------------------
def pp_unit(ctx, *, gate=None, code=None, log=print):
    """Once: under the spy from the first file to the last input; pp/<pair>.json per pair, then pp_r2.json and opened_files.json.  0 done, 8 stopped (the record says where)."""
    import pq_firewall as fw
    import pq_io as io_
    tick = perf_counter()
    preload()
    from bayesreg.io.sealed import sealed_cache_is_empty
    if not sealed_cache_is_empty():
        raise PPRefused('spy', 'the guard\'s cache of the sealed lists is populated before the spy: a unit runs in a process of its own')
    root = Path(ctx.pq_root)
    pairs_dir = root/PP_DIR
    if pairs_dir.exists() or (root/PP_FILE).exists() or (root/OPENED_FILE).exists():
        raise PPRefused('once', f'{root}: the pp unit ran before; nothing is overwritten or rerun')
    spy = fw.Spy(rules_for(ctx))
    rows, error, stage = {}, None, None
    with spy:
        try:
            gate = machine_gate() if gate is None else gate
            code = code_state(ctx.repo) if code is None else code
            if ctx.taper is None:
                import mc0_core
                ctx.taper = mc0_core.frozen_api().boundary_taper
            pairs_dir.mkdir(parents=True)
            from bayesreg.io.guard import load_dev_allowlist, verify_allowlist_sources
            allowlist = load_dev_allowlist(Path(ctx.repo)/'configs/bayesreg/dev_subjects.json')
            verify_allowlist_sources(allowlist, ctx.source_root)
            for short in ctx.shorts:
                started = perf_counter()
                pin = io_.load_pair_checked(ctx, short, allowlist)
                v, bound = bind_hs_field(ctx, short, pin.grid)
                layer = io_.structure_layer(ctx, pin)
                L_F, fixed = predicted_labels(ctx, layer, pin, 'fixed', bound['labels_sha256'])
                L_M, moving = predicted_labels(ctx, layer, pin, 'moving', bound['labels_sha256'])
                d = io_.displacement(v, pin)
                del v
                per, mean = pp_of_labels(L_F, L_M, d)
                identity, identity_mean = pp_of_labels(L_F, L_M, np.zeros_like(d))
                del d, L_F, L_M
                row = dict(schema='bayesreg.r2.pp.v1', pair=short, pp_seg35=mean, pp_per_label=per, pp_groups=group_means(per), identity_seg35=identity_mean, identity_per_label=identity, identity_groups=group_means(identity),
                           keys_read=list((getattr(pin, 'provenance', None) or {}).get('keys_read') or []), field=bound, sides=dict(fixed=fixed, moving=moving), seconds=perf_counter()-started, utc=utc())
                rc.write_json_once(pairs_dir/f'{short}.json', row)
                rows[short] = row
                log(f'{short}: pp seg35 {mean:.4f} (identity {identity_mean:.4f}); small six {row["pp_groups"]["small_six"]}')
        except PPRefused as exc:
            error, stage = str(exc), exc.stage
        except fw.Breach as exc:
            error, stage = str(exc), 'firewall'
        except Exception as exc:                              # kept, not rerun
            error, stage = f'{type(exc).__name__}: {exc}', 'error'
    record = spy.record()
    rc.write_json_once(root/OPENED_FILE, dict(record, role='every file the pp unit opened, for the lead to read before the split; the subprocesses (git, systemctl) are not watched'))
    done = error is None and sorted(rows) == sorted(ctx.shorts)
    document = dict(schema='bayesreg.r2.pp_run.v1', passed=done and record['m6'], error=error, stage=stage, pairs=sorted(rows), reads_truth=False, firewall_ok=record['m6'], n_files_opened=record['n_files'],
                    pair_members=record['pair_members'], gate=gate, code=code, seconds=perf_counter()-tick, utc=utc(),
                    note='pp：移动图的 v4 预测标签按 HS 的认证终态场拉回，对固定图的 v4 预测标签逐标签 Dice（评测器自己的函数）；不读真值。identity 是零场的同一个数（只描述）。')
    if done:
        document.update(per_label_mean=[_nanmean([rows[s]['pp_per_label'][k] for s in ctx.shorts]) for k in range(35)], seg35_mean=_nanmean([rows[s]['pp_seg35'] for s in ctx.shorts]),
                        groups_mean={g: _nanmean([rows[s]['pp_groups'][g] for s in ctx.shorts]) for g in ('small_six', 'big_six', 'other_23')},
                        per_pair={s: dict(pp_seg35=rows[s]['pp_seg35'], identity_seg35=rows[s]['identity_seg35'], **rows[s]['pp_groups']) for s in ctx.shorts})
    rc.write_json_once(root/PP_FILE, document)
    log(f"pp: {'done' if document['passed'] else 'STOPPED at '+str(stage)+': '+str(error)}; the lead reads {root/OPENED_FILE} before the split")
    return 0 if document['passed'] else 8


def _nanmean(values):
    ok = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(ok)) if ok else None


# ---- the split and the targets (no truth file; existing records only) -----------------------------------------------------------------------------
def label_names(repo=REPO):
    document = json.loads((Path(repo)/GROUPS_FILE_REL).read_text(encoding='utf-8'))
    return {int(k): v for k, v in document['label_table']['names'].items()}


def structures(labels, names):
    """[(name, (labels...))]: a left and a right label of the same name ('Left-X', 'Right-X') are one structure; a midline label is one on its own."""
    out, seen = [], set()
    by_name = {v: k for k, v in names.items()}
    for label in sorted(labels):
        if label in seen:
            continue
        name = names[label]
        if name.startswith(('Left-', 'Right-')):
            base = name.split('-', 1)[1]
            pair = tuple(sorted(by_name[f'{side}-{base}'] for side in ('Left', 'Right')))
            if not set(pair) <= set(labels):
                raise ValueError(f'{name}: its partner is not in the set')
            out.append((base, pair))
            seen |= set(pair)
        else:
            out.append((name, (label,)))
            seen.add(label)
    return out


def decompose(pp, hs, seg):
    """Per label l (1..35; lists of 35 means over the 19 pairs): E = 1 - D_HS, a = 1 - pp, b = sqrt(2) (1 - s), the R3b model sqrt(a^2 + b^2), the segmentation share b^2/(a^2 + b^2), the contribution E/35
    to 1 - seg35 and its segmentation part; also max(0, E - a), the error beyond the misalignment of the predicted labels (another reading, described)."""
    rows = []
    for k in range(35):
        D, p, s = hs[k], pp[k], seg[k]
        if any(v is None or not math.isfinite(v) for v in (D, p, s)):
            rows.append(dict(label=k+1, undefined=True))
            continue
        E, a, b = 1-D, 1-p, math.sqrt(2)*(1-s)
        share = b*b/(a*a+b*b) if a*a+b*b > 0 else None
        rows.append(dict(label=k+1, hs_dice=D, pp=p, seg_dice=s, error=E, a_not_aligned=a, b_wrong=b, r3b_model_error=math.sqrt(a*a+b*b), share_seg=share, contribution=E/35,
                         contribution_seg=None if share is None else share*E/35, beyond_pp=max(0., E-a), undefined=False))
    return rows


def choose_targets(rows, names, *, small=rc.SMALL_SIX, pool=rc.OTHER_23, top=rc.TARGET_TOP_STRUCTURES, share_min=rc.TARGET_SHARE_MIN):
    """The rule of plan.md section 2: T = the small six and at most `top` structures of the other 23 (a left-right pair counts as one), those whose mean segmentation share is >= share_min, ranked by
    the sum of their labels' contribution_seg (ties: the smaller first label).  Returns (T sorted, the ranking record).  A label with an undefined row makes its structure ineligible."""
    by = {r['label']: r for r in rows}
    ranking = []
    for name, labels in structures(pool, names):
        rs = [by[k] for k in labels]
        if any(r.get('undefined') or r.get('share_seg') is None for r in rs):
            ranking.append(dict(structure=name, labels=list(labels), eligible=False, why='undefined'))
            continue
        share = float(np.mean([r['share_seg'] for r in rs]))
        ranking.append(dict(structure=name, labels=list(labels), share_seg=share, contribution_seg=float(sum(r['contribution_seg'] for r in rs)), contribution=float(sum(r['contribution'] for r in rs)), eligible=share >= share_min))
    eligible = sorted((r for r in ranking if r['eligible']), key=lambda r: (-r['contribution_seg'], r['labels'][0]))
    chosen = eligible[:top]
    for r in ranking:
        r['chosen'] = r in chosen
    targets = sorted(set(small) | {k for r in chosen for k in r['labels']})
    return targets, sorted(ranking, key=lambda r: (-(r.get('contribution_seg') or -1), r['labels'][0]))


def hs_per_label(hs_root, metrics_pins):
    """{short: [35 per-label Dice]} of HS's repaired fields, repair_metrics.json after.metrics.seg35_per_label, each file against the reference's sha256 (an existing evaluation record: no truth file)."""
    out = {}
    for short, pin in metrics_pins.items():
        hits = list(Path(hs_root).glob(f'runs/RSHx24_*/{short}/repair_metrics.json'))
        if len(hits) != 1 or rc.sha256_file(hits[0]) != pin:
            raise PPRefused('split', f'{short}: repair_metrics.json missing or not the pinned file')
        table = json.loads(hits[0].read_text(encoding='utf-8'))['after']['metrics']['seg35_per_label']
        out[short] = [None if table.get(str(k)) is None else float(table[str(k)]) for k in rc.LABELS]
    return out


def uf4_per_label(path=rc.UF_EVAL, pinned=rc.UF_EVAL_SHA256):
    """{subject: [35 per-label Dice]} of UF4 (eval_uf.json rows UF4 per_label), the pinned file (an existing evaluation record)."""
    if pinned is not None and rc.sha256_file(path) != pinned:
        raise PPRefused('split', f'{path}: not the pinned eval_uf.json')
    rows = json.loads(Path(path).read_text(encoding='utf-8'))['rows']['UF4']
    return {s: [None if v is None else float(v) for v in rows[s]['per_label']] for s in rc.SUBJECTS}


def split(*, pp_root=rc.PP_ROOT, repo=REPO, hs_root=rc.HS_ROOT, uf_eval=rc.UF_EVAL, uf_pinned=rc.UF_EVAL_SHA256, ref_pinned=rc.HS_REFERENCE_SHA256, log=print):
    pp_root = Path(pp_root)
    run = rc.read_json(pp_root/PP_FILE)
    if run is None or run.get('passed') is not True or run.get('reads_truth') is not False:
        raise PPRefused('split', f'{pp_root/PP_FILE}: the pp unit did not pass')
    for name in (SPLIT_FILE, TARGETS_FILE, REPORT_FILE):
        if (pp_root/name).exists():
            raise PPRefused('once', f'{pp_root/name} exists; never overwritten')
    _, _, metrics_pins, reference = reference_rows(repo, pinned=ref_pinned, hs_root=hs_root)
    hs = hs_per_label(hs_root, metrics_pins)
    seg = uf4_per_label(uf_eval, uf_pinned)
    pp = {s: rc.read_json(pp_root/PP_DIR/f'{s}.json')['pp_per_label'] for s in rc.SHORTS}
    mean = lambda values: _nanmean(values)      # noqa: E731
    pp_mean = [mean([pp[s][k] for s in rc.SHORTS]) for k in range(35)]
    hs_mean = [mean([hs[s][k] for s in rc.SHORTS]) for k in range(35)]
    seg_mean = [mean([None if seg[a][k] is None or seg[b][k] is None else (seg[a][k]+seg[b][k])/2 for a, b in map(rc.pair_subjects, rc.SHORTS)]) for k in range(35)]
    rows = decompose(pp_mean, hs_mean, seg_mean)
    names = label_names(repo)
    targets, ranking = choose_targets(rows, names)
    rule = dict(small=list(rc.SMALL_SIX), pool='the other 23 (OTHER_23), a left-right pair one structure', top=rc.TARGET_TOP_STRUCTURES, share_min=rc.TARGET_SHARE_MIN, rank='sum of contribution_seg, ties the smaller label',
                model='R3b: share_seg = b^2/(a^2+b^2), a = 1 - pp, b = sqrt(2)(1 - s); a quadrature model, an order-of-magnitude reference (gap analysis review, important 6)')
    sources = dict(pp=str(pp_root/PP_FILE), pp_sha256=rc.sha256_file(pp_root/PP_FILE), opened_files_sha256=rc.sha256_file(pp_root/OPENED_FILE) if (pp_root/OPENED_FILE).is_file() else None, reference=reference,
                   eval_uf=dict(path=str(uf_eval), sha256=uf_pinned), hs_repair_metrics=metrics_pins)
    document = dict(schema='bayesreg.r2.pp_split.v1', rows=rows, ranking=ranking, rule=rule, sources=sources, sums=dict(error=sum(r['error'] for r in rows if not r['undefined'])/35,
                    r3b_model=sum(r['r3b_model_error'] for r in rows if not r['undefined'])/35), labels=dict(small_six=list(rc.SMALL_SIX), big_six=list(rc.BIG_SIX), other_23=list(rc.OTHER_23)), names={str(k): v for k, v in names.items()},
                    note='已有记录的数（HS 的逐标签 Dice、UF4 的分割级逐标签 Dice）与 pp 合起来；没有打开真值文件。R3b 的拆分是平方和模型，只作量级参照。', utc=utc())
    rc.write_json_once(pp_root/SPLIT_FILE, document)
    rc.write_json_once(pp_root/TARGETS_FILE, dict(schema='bayesreg.r2.targets.v1', targets=targets, rule=rule, split_sha256=rc.sha256_file(pp_root/SPLIT_FILE), utc=utc()))
    rc.write_text_once(pp_root/REPORT_FILE, report(document, targets, names))
    log(f'targets {targets}: {pp_root/TARGETS_FILE}')
    return targets


def _f(v, d=4):
    return 'n/a' if v is None else f'{v:.{d}f}'


def report(document, targets, names):
    L = ['# R2 的 pp 诊断与拆分（工具写的；开发集 19 对，单次；不读真值）', '',
         f"T = {targets}（小六个 + 至多 {document['rule']['top']} 个结构，平均分割份额 ≥ {document['rule']['share_min']}，按分割部分的贡献排序）。R3b 是平方和模型，只作量级参照，不是可加的拆分。", '',
         '| 标签 | 名称 | HS Dice | pp | 分割级 UF4 | 误差 | a=1−pp | b=√2(1−s) | 分割份额 | 贡献 | 分割部分 |', '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted((r for r in document['rows'] if not r['undefined']), key=lambda r: -r['contribution']):
        L.append(f"| {r['label']} | {names[r['label']]} | {_f(r['hs_dice'])} | {_f(r['pp'])} | {_f(r['seg_dice'])} | {_f(r['error'])} | {_f(r['a_not_aligned'])} | {_f(r['b_wrong'])} | {_f(r['share_seg'], 2)} | {_f(r['contribution'])} | {_f(r['contribution_seg'])} |")
    L += ['', '其余 23 个的结构排序：', '']
    for r in document['ranking']:
        L.append(f"- {r['structure']} {r['labels']}：份额 {_f(r.get('share_seg'), 2)}，分割部分 {_f(r.get('contribution_seg'))}，{'入选' if r.get('chosen') else ('合格' if r.get('eligible') else '不合格')}")
    return '\n'.join(L)+'\n'


# ---- the start string, the dry run -----------------------------------------------------------------------------------------------------------------
def start_string(repo=rc.MAIN_REPO, frozen=FROZEN):
    env = dict(PYTHONDONTWRITEBYTECODE=1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, CUDA_VISIBLE_DEVICES='', PYTHONPATH=f'{frozen}:{repo}')
    return ['systemd-run', '--user', f'--unit={UNIT}', '--collect', f'--working-directory={repo}', f'--property=MemoryMax={MEMORY_MAX}', '--property=RuntimeMaxSec=7200', '--nice=10',
            *[f'--setenv={k}={v}' for k, v in env.items()], rc.PY, '-B', SCRIPT, 'pp']


def run_json_shape(rows, run_json, results, *, short=rc.SHORTS[0], log=print):
    """The dry run's one real record (the coordinator's ruling after pp's second start): the run.json of one pair (only this JSON; no field, no prediction, no truth), against the reference's sha256,
    through m1's segmenter check on HS's keys (hs_run_problems) with the unit's anchor, prediction root and pins.  The list of problems."""
    import pq_io as io_
    path = Path(results)/'hs_v1'/'runs'/rows[short].t88_dir/short/'run.json'
    if not path.is_file():
        return [f'{path}: no such file']
    if rc.sha256_file(path) != run_json[short]:
        return [f'{path}: sha256 is not the reference\'s']
    ctx = io_.Context(repo=REPO, source_root=rc.MAIN_REPO, results=Path(results))
    problems = hs_run_problems(json.loads(path.read_text(encoding='utf-8')), ctx, short)
    log(f"one real run.json ({path}, pinned sha256): m1's segmenter check on HS's keys {'passes' if not problems else problems}")
    return [f'{short} run.json: {p}' for p in problems]


def dry_run(log=print, *, repo=REPO, results=rc.RESULTS):
    """Names, sizes and the committed reference only: no field, no label, no prediction is read; nothing written."""
    problems = []
    log(f'== R2 pp diagnosis, dry run ({utc()}); nothing is read but the reference JSON and one run.json, nothing written')
    log('start (the lead, after the review):\n  '+shlex.join(start_string()))
    try:
        rows, run_json, metrics, reference = reference_rows(repo, hs_root=Path(results)/'hs_v1')
        log(f"the reference {reference['path']}: sha256 {reference['sha256']} (pinned), decision_hs.json {reference['decision_sha256']} (pinned)")
    except PPRefused as exc:
        log(f'PROBLEM: {exc}')
        return ['the reference']
    runs = Path(results)/'hs_v1'/'runs'
    for short in rc.SHORTS:
        r = rows[short]
        need = [runs/r.t88_dir/short/'run.json', runs/r.t88_dir/short/'field_0600.npz', runs/r.repair_dir/short/'repair.json', runs/r.repair_dir/short/'repair.npz']
        missing = [str(p) for p in need if not p.is_file()]
        if missing:
            problems.append(f'{short}: missing {missing}')
    log(f"the 19 pairs' four record files there (existence only): {'yes' if not problems else problems}")
    problems += run_json_shape(rows, run_json, results, log=log)
    pred = Path(results)/'seg'/'unet35_v4_ens'/'pred'
    missing = [s for s in rc.SUBJECTS if not all((pred/s/n).is_file() for n in ('probs36.npz', 'structure_groups.npz', 'provenance.json'))]
    log(f'the v4 predictions of the 20 subjects there: {"yes" if not missing else missing}')
    if missing:
        problems.append(f'predictions missing {missing}')
    root = rc.PP_ROOT
    log(f"the pp root {root}: {'EXISTS (the unit refuses: once)' if root.exists() else 'not there yet (the unit makes it)'}")
    if root.exists():
        problems.append(f'{root} exists')
    log(f'frozen tree {FROZEN}: {"there" if FROZEN.is_dir() else "MISSING"}')
    if not FROZEN.is_dir():
        problems.append('the frozen tree')
    log('〔估算〕CPU about 1 min per pair (the field and the taper, two label maps by the method\'s reader, two warps) = about 20 min; host peak about 3 GiB; disk: 19 small JSON files')
    log('the unit reads, under PQ\'s spy: the 19 manifest.json and pair.npz (members fixed, moving, initial_u only), hs_v1/runs/<tier>/<pair>/{run.json, field_0600.npz, repair.json, repair.npz}, '
        'the v4 predictions (probs36.npz, structure_groups.npz, provenance.json), configs/bayesreg/, the sealed lists the guard reads; forbidden: label files, metrics_*.json, repair_metrics.json, analysis/')
    log(f"== dry run problems: {'; '.join(problems) if problems else 'none'}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('command', nargs='?', choices=('pp', 'split'))
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            return 0 if not dry_run() else 8
        if args.command == 'pp':
            if not in_own_unit():
                raise PPRefused('unit', f'the pp unit runs in {UNIT} only (the start string of --dry-run)')
            return pp_unit(make_context())
        if args.command == 'split':
            split()
            return 0
    except PPRefused as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        return 8
    parser.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
