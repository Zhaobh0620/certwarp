"""HE (runs/20261106-he/任务书.md; contract C22, revisions 1-3; C03.1; C16): the rules, records and checks of the firewall-clean entry of the C22 line ("Hx line": T88 + local repair).  The drivers are in run_hx.py.
HE2 (runs/20261107-he2/任务书.md; contract C22 revision 4, BZ 2026-11-07: C22.0-5, C22.9-4 (xv)(xvi), C22.B1, C22.B4, D80-D86): the method is Hx24-RF05-F1200-v4-C3: the structure-layer segmenter
unet35_v4_ens, checked against an anchor of its own (CONTRACT_SEGMENTER, never derived from the frozen settings), and the tissue-dependent geometry prior C (tissue_prior moment3, he_c.py) in the T88 child.
HB (runs/20261109-hb/任务书.md; contract C22 卷首修订 8 (D81 = option II), C22.H): the method is Hx24-RF05-F1200-v4, tissue_prior off (C is the ablation -C3); review_certificate_in is a frozen setting;
the fifth line status hx_implementation_error (C22.5-6).  HE2's settings are kept as a record (he2_settings).

Why here (experiments/he/) and not under bayesreg/: every part of this line is in experiments/ (the copied T88 in experiments/mc0/mc0_core.py, its channels in experiments/ef/fast_channels.py and experiments/fl/fl_channels.py,
the repair in experiments/ef/repair.py, the frozen worker's checks in experiments/d0/) and those modules are imported as flat names from their own directories; a bayesreg module that put experiments/ on sys.path
would make the package depend on scripts.  The line is not F_B (C22.0-1), which bayesreg/ holds.  A new directory touches no file that a running or reviewed batch uses (experiments/fl, experiments/uf), and
experiments/ is an existing top-level entry, so it adds no importable name to the PYTHONPATH (RP review, new minor 1).

Processes (each label-free; the only reader of truth labels is the evaluation unit, after the output is fixed):
  pair (the supervisor)  admission (load_pair's subject guard, before anything is opened); the T88 child; the repair child; the decision; the fallbacks; the output.  Its own work: the derived pair of fallback
                         one, the frozen prepare, the launch, the classification, the certificates of the fallbacks' fields.
  t88 (child)            the registered segmenter's predictions (provenance and read-back checks, the alignment QC; the anchor before and after) -> K = 35 channels; the fixed image's Atropos tissue layer
                         (C22.B1-2/3) -> three passes of T88 (pass 0 the frozen T88, passes 1, 2 with the cell weights of C); t88/run.json, t88/field_0600.npz (pass 2's field).
  repair (child)         t88/run.json and the field file checked by their own sha256 (never an evaluation record), the taper done here, local repair; its own certificate of the final v is the decision
                         certificate (C22.4-1): repair/cert.json, repair/certified.npz.
  configuration C        the frozen worker on the derived pair (firewall_c.py): fixed, moving, initial_u byte for byte, the label members zero.
  third-cert             C22.3-9 (d): a certificate of the output array in a process that reads no label; pass -> the output stands, fail -> the next fallback.
  evaluate               the review certificate first (no label read yet), then the truth through bayesreg.eval.truth and the metrics through bayesreg.eval.metrics from disp.npz; once per pair.
Children are separate processes so that a killed or out-of-memory T88 or repair still ends in a fallback (C22.3-9 (b)): the supervisor reads what they wrote.

The rules (C20 D74, D75 as BZ fixed them 11-05, contract revision 3):
  T88 without a done field (time_cap, vram_stop, cuda_oom, nonfinite, scale_mismatch, not_predicted_channels, no run.json)  -> failure entry -> fallback (D74 (2))
  alignment QC below .80                                                                                                     -> failure segmentation_qc -> fallback (D74 (2))
  provenance, firewall or admission refusal; the segmenter not the anchor's (C22.9-4 (xv))                                  -> hx_provenance_rejected, no output (D74 (4))
  C (D86): the tissue layer's provenance or files, its class names -> hx_provenance_rejected; a coarse grid -> hx_provenance_rejected, implementation_error, the queue stops (exit 9); an empty brain mask,
  a level without an Omega_B cell, estimate_failed -> failure entry -> fallback; any other exception of the tissue preparation -> failure entry -> fallback, needs_lead_rejudgment.  The lead's ruling of
  2026-11-07 (HE2 review minors 1, 2): the simplex check failing, a tissue-layer file that is not parsed (provenance.json not JSON) and load_scalar's InputValidationError are of the last kind
  (only load_tissue_layer's ProvenanceRejected is a refusal); he_c.refusals is the table
  repair not certified: round_cap or empty_mask -> repair_not_certified; the fuse and then not certified -> repair_fuse; no repair.json, a failed source check, an inconsistent certificate -> repair_error
  the fuse and then certified                                                                                               -> hx_certified (D74 (3) = (c-ii)); both readings are recorded
  fallback one: configuration C (classes ok and handover_only: final_native.npz, else the certified native seed), certified again here; fallback two: the identity u = 0 (D75).
"""
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for _dir in ('he', 'mc0', 'ef', 'fl'):
    if str(HERE.parent/_dir) not in sys.path:
        sys.path.insert(0, str(HERE.parent/_dir))

import mc0_core as core  # noqa: E402
import unit_probe  # noqa: E402

RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
SCHEMA = 'bayesreg.he.{}.v1'
# HB (runs/20261109-hb/任务书.md item 1; contract C22 卷首修订 8/8a/8b, BZ 2026-11-07 about 6:54 PM CST: D81 = option II; C22.H0-3, C22.H10-6, revision 7 adopted by BZ 2026-11-09): the method is
# Hx24-RF05-F1200-v4, tissue_prior off (C only the ablation's definition, C22.B), and the review certificate of the hold-out is written by the step-5 review unit (C22.H1-5).  The arithmetic is not
# changed: with tissue_prior off the T88 child runs mc0_core.optimize_k once, the path of HS (run_hs, hs_v1); HE2's settings are kept as a record (he2_settings, HE2_SETTINGS_SHA256).
NAME = 'Hx24-RF05-F1200-v4'                     # C22.0-5; C22 卷首修订 8 (D81 = option II)
CONTRACT = ('C22.7 (D71), C22.0-5; revision 8 (BZ 2026-11-07: D81 = option II, C only an ablation, tissue_prior off; D80 unet35_v4_ens, D82 = II-a); '
            'revision 7 (BZ 2026-11-09: C22.H, the review certificate in the step-5 review record)')
METHOD = f'A_H {NAME} (C22 revision 8; T88 + local repair, the certified final state; tissue_prior off; not F_B)'
REVIEW_CERTIFICATE_IN = ('review/review.json of the pair (the step-5 review unit of the hold-out, contract C22.H1-5; experiments/hb/hb_review.py); on a development pair run_hx.py evaluate '
                         'writes its review certificate in evaluation.json as before')        # C22.H0-3, C22.H1-7: was run_hx.py:746's literal string, now a frozen setting
# HE2's frozen settings (runs/20261107-he2/smoke_result.md line 7; decision_log 2026-11-07 about 4:43 AM CST): kept as a record only (he2_settings rebuilds them; a test compares the sha256)
HE2_NAME = 'Hx24-RF05-F1200-v4-C3'
HE2_CONTRACT = 'C22.7 (D71), C22.0-5, C22.B1; revision 4 (BZ 2026-11-07: D80 unet35_v4_ens, D81 = option I, D82 = -a, D83, D86)'
HE2_SETTINGS_SHA256 = '76a9e563ce86e9cb01e6487759949de66f479a8cd35787c55dc42d9a8d7453e5'
# C22.9-4 (xv), C22.1-2 revision 4 (D80): the segmenter's anchor, the four values of the contract written here word for word.  Never derived from the frozen settings: the settings' segmenter is checked
# against it (segmenter_start_problems), and so are the reader's records (channel_source_problems).
CONTRACT_SEGMENTER = dict(segmenter_id='unet35_v4_ens', policy='all_folds_mean', registry='configs/bayesreg/segmenters_hs_v1.json',
                          registry_sha256='b7682421bf6e8d07c606ad55cd47db7936a293c5e79d00839c82e3f8c2edefa1')     # C22.1-2 修订 4
SEGMENTER_GATE = dict(rule='C22.1-2 修订 4, D80 (BZ 2026-11-07 "1换"): the structure-layer segmenter is unet35_v4_ens', contract='C22.9-4 (xv)')     # UNetSource.gate, recorded in channel_sources.segmenter
V4_SOURCES = dict(pred_root=RESULTS/'seg'/'unet35_v4_ens'/'pred', models_root=RESULTS/'seg'/'unet35_v4_full', image_root=Path('/mnt/d/data'))     # experiments/p2/reference_unet35_v4_ens.json channels (090c985); run_hs.hs_source
# HB (the brief's item 1): the hold-out's sources.  pred_root: HI's hold-out output (hi_core.HOLDOUT_OUT_SUGGESTED/pred; xb_core.HI_PRED_ROOTS['holdout']); atropos_root: the hold-out Atropos entry's output
# (holdout_common.ATROPOS_ROOT).  Paths, not settings: they are in the context (he_holdout.holdout_paths), not in settings_sha256.  If HI runs with another --out, this value is changed by a reviewed commit.
HOLDOUT_SOURCES = dict(pred_root=RESULTS/'seg'/'unet35_v4_ens_holdout_B_v1'/'pred', atropos_root=RESULTS/'holdout_B_v1'/'atropos')
TISSUE_PRIOR = 'off'                            # C22.B0-2: tissue_prior in {off, moment3}; C22 卷首修订 8 (D81 = option II): off; moment3 is the ablation -C3 (C22.B), HE2's setting
STOP_QUEUE_EXIT = 9                             # C22.5-6 (revision 8a/8b): hx_implementation_error, line.json written (HB); for the ablation -C3 also D86 (2) (C's tissue preparation); a queue stops on it
# The units (H2, H3, the gate of c; runs/20261107-he2/任务书.md item 6).  The supervisor's unit: D0's frozen unit's memory (12G, swap 4G: in DirectLauncher's case the frozen run shares it), and
# OOMPolicy=continue: the user manager's DefaultOOMPolicy is stop, under which a child killed by the OOM killer stops the whole unit and the supervisor with it (C22.3-9 (b) would never be reached).
SUPERVISOR_UNIT = dict(memory_max='12G', memory_swap_max='4G', oom_policy='continue', quota=1200, numba=8, runtime=4800, nice=5)
# HE2 review minor 14: the evaluation unit too.  After a failed third certificate its child (the third certification, the frozen prepare) runs in it while the old output is already withdrawn; with stop, an
# OOM kill there would stop the whole unit without a record; with continue the evaluation process writes evaluation_not_written.
EVALUATION_UNIT = dict(memory_max='8G', memory_swap_max='4G', oom_policy='continue', quota=800, numba=8, runtime=1800, nice=5)
# Every value of `units` (these two, the gates, the frozen unit's) is part of the method's frozen settings and of settings_sha256: changing any of them is a new method identity and asks for a new smoke
# (C22.9-4 (xv); HE2 review minor 11).  A holdout queue builds its supervisor units from SUPERVISOR_UNIT and checks them with unit_argv_problems.
# The gate before a unit in which configuration C's frozen run can start (MemAvailable at launch, GiB): the supervisor (a nested unit or this unit) and the evaluation (the third certificate's fallback, in a
# unit of its own: H3).  A gate gives a cgroup no room; it asks the machine to be idle enough at launch.  10 GiB is the lead's value for (d) (decision_log 2026-11-06, HES review), given to every such unit
# so that (c), whose nested unit has the same 12G as (d), is not gated lower (the gate of c).  It stays below 12G: this machine has about 14 GiB (free -g), MemAvailable about 11-12 GiB when idle.
FROZEN_RUN_GATE_GIB, OTHER_GATE_GIB = 10., 7.
THIRD_FALLBACK_LAUNCHER = 'unit'                # H3: after a failed third certificate, configuration C runs in a unit of its own (D0's limits), never inside the evaluation unit
HX, FALLBACK_C, FALLBACK_ID, REJECTED = 'hx_certified', 'hx_fallback_configC', 'hx_fallback_identity', 'hx_provenance_rejected'
IMPLEMENTATION = 'hx_implementation_error'      # C22.5-4's fifth value (C22 卷首修订 8b, C22.5-6): no output, the queue stops, to BZ
LINE_STATUSES = (HX, FALLBACK_C, FALLBACK_ID, REJECTED, IMPLEMENTATION)
OUTPUT_STATUSES = (HX, FALLBACK_C, FALLBACK_ID)
FAILURE_CLASSES = ('entry', 'segmentation_qc', 'repair_not_certified', 'repair_fuse', 'repair_error')
FUSE_RULE = 'c-ii'                              # D74 (3), BZ 2026-11-05 ("第二批都按建议"; contract revision 3)
T88_DONE, T88_QC, T88_REJECTED = 'done', 'segmentation_qc', 'provenance_rejected'
FIELD_FILE = 'field_0600.npz'
OUTPUT_DIR = 'output'
CERT_PASS = 'certified_feasible'
REPO_TOP_LEVEL = frozenset(('.gitignore', 'README.md', 'archive', 'bayesreg', 'configs', 'docs', 'experiments', 'jd_curl', 'pyproject.toml', 'runs', 'scripts', 'splits', 'tests'))     # from git ls-tree --name-only ace0903 (a test checks it is a subset)
TOP_LEVEL_COMMIT = 'ace0903'
FROZEN_UNTRACKED_TOP = frozenset(('data', '.git'))     # the frozen tree's top level beyond eb808be's (RP review, new minor 1)
IGNORED_TOP = frozenset(('.git', '__pycache__'))


# ---- the frozen values (C22.7; the code constants they come from are compared by a test) ------------------------------------------------------------
def _frozen_settings():
    pre = dict(core.PRE_C)
    pre.update(iterations=[50, 50, 100, 600], sigma_field=.5, seconds=900.)          # fl_core COARSE_ITERATIONS + fd2/fx NATIVE 600; fl_core.RUN_CAP_SECONDS; fx_core.SPEC sigma_field
    pre = {k: list(v) if isinstance(v, tuple) else v for k, v in pre.items()}
    from dataclasses import fields
    import repair as rp
    defaults = {f.name: f.default for f in fields(rp.RepairConfig) if f.name != 'eps'}
    import he_c
    return dict(name=NAME, contract=CONTRACT, review_certificate_in=REVIEW_CERTIFICATE_IN,
                segmenter=dict(segmenter_id='unet35_v4_ens', policy='all_folds_mean', registry='configs/bayesreg/segmenters_hs_v1.json',
                               registry_sha256='b7682421bf6e8d07c606ad55cd47db7936a293c5e79d00839c82e3f8c2edefa1'),     # written again here (C22.9-4 (xv)): compared with CONTRACT_SEGMENTER, not taken from it
                channels=dict(k=35, channels='onehot', weight='Q3h', D_factor=24., sigma=.75, colour_table_sha256=core.COLOUR_TABLE_SHA256),
                t88=dict(pre=pre, taper=core.TAPER),
                tissue_prior=TISSUE_PRIOR, c=he_c.constants(),                                    # C22.B4-4
                repair=dict(defaults, eps=.05, wall_seconds=1200.),
                certificate=dict(delta=0., max_depth=48, max_boxes=255, tile_size=65536),
                configc=dict(opts=None, watchdog_seconds=None, stage_label=None),     # filled below from firewall_c (no import cycle)
                fuse_rule=FUSE_RULE, fallbacks=['configC', 'identity'],
                children=dict(t88_timeout_seconds=1200, repair_timeout_seconds=1800),   # the RuntimeMaxSec of RW's units: run_mc0.RUN_RUNTIME_MAX, fx_core.REPAIR_RUNTIME_SECONDS
                device_limit_gib=9,                                                     # mc0p_core.VRAM_LIMIT_BYTES
                units=dict(supervisor=dict(SUPERVISOR_UNIT), evaluation=dict(EVALUATION_UNIT), frozen_c=None, gate_frozen_run_gib=FROZEN_RUN_GATE_GIB, gate_other_gib=OTHER_GATE_GIB,
                           third_certificate_fallback=THIRD_FALLBACK_LAUNCHER))          # frozen_c filled below from firewall_c


def frozen_settings():
    import firewall_c as fc
    s = _frozen_settings()
    s['configc'] = dict(opts=list(fc.OPTS), watchdog_seconds=fc.WATCHDOG_SECONDS, stage_label=fc.STAGE_LABEL, reference_sha256=core.sha_file(fc.CONFIGC_REFERENCE))
    s['units']['frozen_c'] = dict(resources=list(fc.UNIT_RESOURCES), oom_policy=fc.UNIT_OOM_POLICY, probe=list(fc.PROBE_FILES))
    return s


def he2_settings():
    """HE2's frozen settings (method Hx24-RF05-F1200-v4-C3, tissue_prior moment3, no review_certificate_in), rebuilt for the record: HB changed only these keys (a test compares the sha256 with
    HE2_SETTINGS_SHA256).  Never a context's settings for a run of the method."""
    s = frozen_settings()
    s.update(name=HE2_NAME, contract=HE2_CONTRACT, tissue_prior='moment3')
    del s['review_certificate_in']
    return s


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=float)


def settings_sha256(settings):
    return hashlib.sha256(canonical(settings).encode()).hexdigest()


def is_frozen(settings):
    return canonical(settings) == canonical(frozen_settings())


# ---- small helpers --------------------------------------------------------------------------------------------------------------------------------
def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def read_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except ValueError:
        return None


def write_json(path, value, *, exclusive=True):
    """Written once: an existing file is never overwritten (hard rule 4)."""
    path = Path(path)
    text = json.dumps(value, indent=1, ensure_ascii=False, default=_default)+'\n'
    with path.open('x' if exclusive else 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def _default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path, tuple, set, frozenset)):
        return str(value) if isinstance(value, Path) else list(value)
    return str(value)


def array_sha256(a):
    """The certificate's state hash convention (C07-7): sha256 of the C-order bytes."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def short_name(pair):
    """0438-0439 for OASIS pairs (mc0_core.short_name), the pair's name with unsafe characters replaced otherwise."""
    try:
        return core.short_name(pair)
    except IndexError:
        return ''.join(c if c.isalnum() or c in '-_' else '_' for c in pair)[:60]


# ---- the segmenter's anchor (C22.9-4 (xv)) ----------------------------------------------------------------------------------------------------------
ANCHOR_RULE = ("C22.9-4 (xv): the frozen settings' segmenter item by item, the registry read now (its path relative to the code root, its sha256 the deciding item) and every model's segmenter_id, at the "
               "start; the reader's records (channel_sources.sides.* segmenter_id, policy, registry_sha256, registry_path; channel_sources.segmenter segmenter_id and registry) after the channels; "
               "all against CONTRACT_SEGMENTER; any difference -> provenance_rejected, no fallback (D74 (4))")


def anchor_registry(code_root=None):
    return Path(core.ROOT if code_root is None else code_root)/CONTRACT_SEGMENTER['registry']


def repo_relative(path, code_root=None):
    """The path resolved and relative to the code root, as a POSIX string; None when it is empty or not under the root (UNetSource.record() writes str(registry), an absolute path: C22.9-4 (xv))."""
    if not path:
        return None
    root = Path(core.ROOT if code_root is None else code_root).resolve()
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def segmenter_start_problems(settings, registry=None, code_root=None):
    """(problems, record) at the start of a process: the settings' segmenter is the anchor, item by item; the registry file read now has the anchor's path (relative to the code root) and sha256; each of
    its models names the anchor's segmenter_id."""
    a = CONTRACT_SEGMENTER
    path = anchor_registry(code_root) if registry is None else Path(registry)
    record = dict(anchor=dict(a), settings_segmenter=settings.get('segmenter'), registry=str(path), registry_rel=repo_relative(path, code_root), registry_sha256=None, model_segmenter_ids=None, rule=ANCHOR_RULE)
    problems = []
    if settings.get('segmenter') != a:
        problems.append(f"the frozen settings' segmenter {settings.get('segmenter')} is not the contract's anchor {a}")
    if record['registry_rel'] != a['registry']:
        problems.append(f"the registry {path} is not {a['registry']} of the code root")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return problems+[f'the registry {path}: {type(exc).__name__}: {exc}'], record
    record['registry_sha256'] = hashlib.sha256(raw).hexdigest()
    if record['registry_sha256'] != a['registry_sha256']:
        problems.append(f"the registry {path} has sha256 {record['registry_sha256']}, not the anchor's {a['registry_sha256']}")
    from bayesreg.io.pair import ProvenanceRejected, load_segmenter_registry
    try:
        models = load_segmenter_registry(path).models
    except (ProvenanceRejected, ValueError) as exc:
        return problems+[f'the registry {path}: {type(exc).__name__}: {exc}'], record
    record['model_segmenter_ids'] = {m: dict(e).get('segmenter_id') for m, e in models.items()}
    if not models or any(v != a['segmenter_id'] for v in record['model_segmenter_ids'].values()):
        problems.append(f"the registry's models {record['model_segmenter_ids']} do not all name {a['segmenter_id']}")
    return problems, record


def channel_source_problems(sources, code_root=None):
    """Problems of the reader's record (fast_channels.read_predicted's channel_sources) against the anchor; [] when every item is the anchor's."""
    a = CONTRACT_SEGMENTER
    problems = []
    sides = (sources or {}).get('sides') or {}
    if sorted(sides) != ['fixed', 'moving']:
        problems.append(f'channel_sources.sides {sorted(sides)}: fixed and moving required')
    for side in sorted(sides):
        s = sides[side] or {}
        for key, want in (('segmenter_id', a['segmenter_id']), ('policy', a['policy']), ('registry_sha256', a['registry_sha256'])):
            if s.get(key) != want:
                problems.append(f'channel_sources.sides.{side}.{key} {s.get(key)!r} is not {want!r}')
        if repo_relative(s.get('registry_path'), code_root) != a['registry']:
            problems.append(f"channel_sources.sides.{side}.registry_path {s.get('registry_path')!r} is not {a['registry']} of the code root")
    seg = (sources or {}).get('segmenter') or {}
    if seg.get('segmenter_id') != a['segmenter_id']:
        problems.append(f"channel_sources.segmenter.segmenter_id {seg.get('segmenter_id')!r} is not {a['segmenter_id']!r}")
    if repo_relative(seg.get('registry'), code_root) != a['registry']:
        problems.append(f"channel_sources.segmenter.registry {seg.get('registry')!r} is not {a['registry']} of the code root")
    return problems


# ---- the unit (H2; C22.B4-6 (d)) ---------------------------------------------------------------------------------------------------------------------
def gib_bytes(text):
    """'12G' -> 12 * 2**30 (systemd's suffixes K, M, G, T, base 1024); a plain number of bytes as it is; None otherwise."""
    value = str(text)
    scale = {'K': 2**10, 'M': 2**20, 'G': 2**30, 'T': 2**40}.get(value[-1:])
    if scale:
        try:
            return int(float(value[:-1])*scale)
        except ValueError:
            return None
    return int(value) if value.isdigit() else None


def unit_matches(probe, expected):
    """{MemoryMax, MemorySwapMax, OOMPolicy: True | False | None}: systemd's reading of the unit against the settings' unit; None where systemd was not read (读不到) or nothing is expected."""
    props = (probe or {}).get('systemd')
    want = dict(MemoryMax=gib_bytes(expected['memory_max']) if expected.get('memory_max') else None, MemorySwapMax=gib_bytes(expected['memory_swap_max']) if expected.get('memory_swap_max') else None,
                OOMPolicy=expected.get('oom_policy'))
    out = {}
    for k, w in want.items():
        got = None if props is None else props.get(k)
        # HE2 diff review new minor 1: systemd answered but this item is missing or empty: not read (None), not "differs"
        out[k] = None if w is None or got is None or str(got).strip() == '' else str(got).strip() == str(w)
    return out


def kernel_matches(probe, expected):
    """{memory.max, memory.swap.max: True | False | None}: the limits the kernel applies to the cgroup (unit_probe: max_bytes, swap_max_bytes; 'max' is no limit) against the settings' unit (HE2 review
    minor 9).  None where it was not read, nothing is expected, or the process is not in a tag- unit of its own (a session's cgroup is not a unit's)."""
    probe = probe or {}
    out = {}
    for key, field, setting in (('memory.max', 'max_bytes', 'memory_max'), ('memory.swap.max', 'swap_max_bytes', 'memory_swap_max')):
        got, want = probe.get(field), gib_bytes(expected[setting]) if expected.get(setting) else None
        out[key] = None if not probe.get('in_tag_unit') or got is None or want is None else got == want
    return out


def unit_record(start, end, expected):
    """The `unit` of line.json and evaluation.json: the probes at the start and at the end of the process (unit_probe.py), systemd's properties and the kernel's limits against the settings' unit, and the
    kernel's memory.peak and memory.events of the cgroup at the end (the whole unit: the children are in it)."""
    events = (end or {}).get('events')
    return dict(start=start, end=end, expected=dict(expected), matches_at_start=unit_matches(start, expected), kernel_matches_at_start=kernel_matches(start, expected), peak_bytes_at_end=(end or {}).get('peak_bytes'),
                events_at_end=events, oom_kills_at_end=None if events is None else events.get('oom_kill'), in_tag_unit=(start or {}).get('in_tag_unit'),
                memory_stat_at_end=(end or {}).get('memory_stat'),                   # HE2 clean: anon and file now (the peak includes the page cache); recorded only

                rule='H2, C22.B4-6 (d): what the machine reported (unit_probe.py); 读不到 where it could not be read; evidence of what the limits were, not a proof that a limit acted')


def unit_start_problems(probe, expected):
    """HE2 review minor 8: the readings of the process's own unit at its start that contradict the settings' unit (systemd's MemoryMax, MemorySwapMax, OOMPolicy; the kernel's memory.max, memory.swap.max).
    A value that was not read (None) is no problem: 读不到 does not stop a run, a reading that differs does."""
    return [f'systemd {k}: not the settings\' unit' for k, ok in unit_matches(probe, expected).items() if ok is False]+[f'kernel {k}: not the settings\' unit' for k, ok in kernel_matches(probe, expected).items() if ok is False]


def unit_argv_problems(argv, unit):
    """HE2 review minors 8 and 11: a start string (systemd-run) of a unit built from `unit` (he_core.SUPERVISOR_UNIT or EVALUATION_UNIT) carries each of its values once: MemoryMax, MemorySwapMax (the
    string's 4G is run_mc0.unit_argv's own: compared here), OOMPolicy, CPUQuota, RuntimeMaxSec, NUMBA_NUM_THREADS, nice.  [] when it does."""
    want = {'--property=MemoryMax=': unit.get('memory_max'), '--property=MemorySwapMax=': unit.get('memory_swap_max'), '--property=OOMPolicy=': unit.get('oom_policy'),
            '--property=CPUQuota=': None if unit.get('quota') is None else f"{unit['quota']}%", '--property=RuntimeMaxSec=': None if unit.get('runtime') is None else str(unit['runtime']),
            '--setenv=NUMBA_NUM_THREADS=': None if unit.get('numba') is None else str(unit['numba']), '--nice=': None if unit.get('nice') is None else str(unit['nice'])}
    problems = []
    for prefix, value in want.items():
        got = [a[len(prefix):] for a in argv if str(a).startswith(prefix)]
        if value is not None and got != [str(value)]:
            problems.append(f'{prefix}{value} expected once, the string has {got}')
    return problems


def safe_probe(when, probe=None):
    """unit_probe.probe, never raising: a failure is recorded as 读不到."""
    try:
        return (probe or unit_probe.probe)(when)
    except Exception as exc:                            # noqa: BLE001 - a probe never stops the method
        return dict(schema=unit_probe.SCHEMA, when=when, error=f'{unit_probe.UNREAD}: {type(exc).__name__}: {exc}'[:500])


# ---- certificates -----------------------------------------------------------------------------------------------------------------------------------
def certificate_config(settings):
    """Configuration C, explicitly (certify_map's own default is not it); the values of the settings, which must be run_mc0.certificate_config()'s."""
    from jd_curl.three_d.feasibility import PrescreenCertificateConfig
    c = settings['certificate']
    return PrescreenCertificateConfig(delta=float(c['delta']), max_depth=int(c['max_depth']), max_boxes=int(c['max_boxes']), tile_size=int(c['tile_size']))


def config_record(config):
    from dataclasses import asdict
    from jd_curl.three_d.feasibility import certificate_version
    record = {k: (v.hex() if k == 'delta' else v) for k, v in asdict(config).items()}
    return dict(record, config_class=type(config).__name__, certificate_version=certificate_version(config), precision='CPU FP64, outward rounding')


def passes(cert):
    """C07-3 / repair.accepted: certified_feasible, lower bound > 0, no infeasible and no uncertain cell.  `cert` a FeasibilityResult or its document."""
    get = (lambda k: cert.get(k)) if isinstance(cert, dict) else (lambda k: getattr(cert, k, None))
    lower = get('min_jd_lower_bound')
    return bool(get('status') == CERT_PASS and lower is not None and lower > 0 and get('n_infeasible') == 0 and get('n_uncertain') == 0)


def certificate_document(result, config, *, process, array_sha256_of_u=None):
    """cert.json (C16): the certificate of exactly one array, with its configuration and the process that computed it."""
    from dataclasses import fields
    doc = {f.name: getattr(result, f.name) for f in fields(result) if f.name != 'config'}
    if doc.get('witness_reference_point') is not None:
        doc['witness_reference_point'] = [float(t) for t in doc['witness_reference_point']]
    doc.update(n_cells=doc.get('n_cells_checked'), witness=dict(cell=doc.get('witness_cell'), reference_point=doc.get('witness_reference_point')), config=config_record(config), passed=passes(result),
               rule='passed = certified_feasible, min_jd_lower_bound > 0, n_infeasible = 0, n_uncertain = 0 ("zero" and "undecided" do not pass, C07-3)', process=process)
    if array_sha256_of_u is not None:
        doc['state_hash_matches_array'] = doc.get('state_hash') == array_sha256_of_u
    return doc


COMPARED = ('status', 'reason', 'min_jd_lower_bound', 'min_margin_lower_bound', 'n_cells_checked', 'n_infeasible', 'n_uncertain', 'witness_cell', 'state_hash', 'certificate_version', 'delta_hex', 'grid_id')
COMPARED_CONFIG = ('delta', 'max_depth', 'max_boxes', 'tile_size', 'config_class', 'certificate_version')     # the version is the configuration's (feasibility.certificate_version(config)); revision 1, review minor 3


def same_certificate(a, b):
    """(same, differences): two certificate documents of the same configuration and array, field by field (C22.3-9 (d), C22.9-4 (xiii))."""
    diffs = {k: (a.get(k), b.get(k)) for k in COMPARED if a.get(k) != b.get(k)}
    ca, cb = a.get('config') or {}, b.get('config') or {}
    diffs.update({f'config.{k}': (ca.get(k), cb.get(k)) for k in COMPARED_CONFIG if ca.get(k) != cb.get(k)})
    return not diffs, diffs


class RecordingCertify:
    """certify_map with every result kept: repair_field passes `certify` its arrays and returns only records, so the full result of the final v (cert.json) is taken from here.  Changes no number."""

    def __init__(self, certify):
        self.certify, self.results = certify, []

    def __call__(self, u, grid, *, config):
        result = self.certify(u, grid, config=config)
        self.results.append(result)
        return result

    def of(self, digest):
        """The last result whose state hash is `digest`, or None."""
        return next((r for r in reversed(self.results) if r.state_hash == digest), None)


# ---- the decision (pure; every branch has a test) ---------------------------------------------------------------------------------------------
def failure(cls, reason, evidence=None, *, needs_lead_rejudgment=False):
    if cls not in FAILURE_CLASSES:
        raise ValueError(f'failure class {cls}')
    out = dict(**{'class': cls}, reason=str(reason)[:2000], evidence=evidence or [])
    if needs_lead_rejudgment:                           # D86 (4): the fallback runs; the lead may judge the pair again (C22.B1-8)
        out['needs_lead_rejudgment'] = True
    return out


T88_IMPLEMENTATION, T88_TISSUE_DATA, T88_TISSUE_ERROR, T88_ESTIMATE = 'implementation_error', 'tissue_data', 'tissue_error', 'estimate_failed'     # he_c's D86 statuses (a test compares)


def t88_decision(record, returncode=None):
    """('go' | 'fallback' | 'rejected', failure or rejection): what the T88 child's record says (C22.5-1; C22.B1-8, D86)."""
    if record is None:
        return 'fallback', failure('entry', f'the T88 process wrote no run.json (exit {returncode})')
    status = record.get('status')
    if status == T88_REJECTED:
        info = dict(stage=record.get('rejection_stage') or 'channels', reason=record.get('reason'))
        if record.get('rejection_kind'):
            info['kind'] = record['rejection_kind']
        return 'rejected', info
    if status == T88_IMPLEMENTATION:                    # D86 (2): no fallback, to the lead, the queue stops
        return 'rejected', dict(stage=record.get('rejection_stage') or 'tissue', kind=T88_IMPLEMENTATION, reason=record.get('reason'), stop_queue=True)
    if status == T88_QC:
        return 'fallback', failure('segmentation_qc', record.get('reason'))
    if status == T88_TISSUE_ERROR:                      # D86 (4)
        return 'fallback', failure('entry', f"{status}: {record.get('reason')}", needs_lead_rejudgment=True)
    if status != T88_DONE or record.get('oracle') is not False or not record.get('field'):
        return 'fallback', failure('entry', f"{status}: {record.get('reason')}")      # estimate_failed and tissue_data (D86 (3)) among them
    return 'go', None


def hit_wall(record):
    """The fuse (experiments/fl/decide_fx.py hit_wall, the same rule; a test compares): reason wall, or any round's inner loop ended at the wall."""
    return record.get('reason') == 'wall' or any((r.get('inner') or {}).get('stop') == 'wall' for r in record.get('rounds') or [])


FALLBACK_NOT_RUN = 'hx_fallback_not_run'      # line_status_if_c_i of a pair that hit the fuse and then passed: under (c-i) it would have gone to a fallback, which (c-ii) does not run


def fuse_record(record, line_status):
    """C22.5-4 fuse: both readings from the record; the rule is (c-ii) (D74 (3))."""
    wall = bool(record is not None and hit_wall(record))
    certified_after = bool(wall and record.get('passed') is True)
    return dict(wall_hit=wall, certified_after_wall=certified_after, fuse_rule=FUSE_RULE, line_status_if_c_ii=line_status, line_status_if_c_i=FALLBACK_NOT_RUN if certified_after else line_status,
                note=f'{FALLBACK_NOT_RUN}: under (c-i) this pair would go to the fallbacks; the rule is (c-ii) (D74 (3)), so they were not run' if certified_after else None)


def repair_decision(record, consistent, returncode=None):
    """('hx' | 'fallback', failure): what the repair child's record says (C22.3-9).  `consistent`: the certified files agree with the record (state hash, certificate, the rule)."""
    if record is None:
        return 'fallback', failure('repair_error', f'the repair process wrote no repair.json (exit {returncode})')
    if record.get('status') == T88_REJECTED:                    # the admission's two locks, checked again in the repair process: a refusal, never a fallback
        return 'rejected', dict(stage='repair', reason=record.get('reason'))
    if record.get('error'):
        return 'fallback', failure('repair_error', record['error'])
    if record.get('passed') is True:
        if consistent is not True:
            return 'fallback', failure('repair_error', f'the certified field and its certificate do not agree with repair.json: {consistent}')
        return 'hx', None
    if record.get('reason') == 'wall':
        return 'fallback', failure('repair_fuse', 'the fuse, then the certificate did not pass')
    return 'fallback', failure('repair_not_certified', record.get('reason'))


def next_fallback(line_status):
    """C22.3-9 (d) when the third certificate fails: the fallback after this output."""
    return {HX: 'configC', FALLBACK_C: 'identity'}.get(line_status)


# ---- the output (C16) --------------------------------------------------------------------------------------------------------------------------
def u_to_d(u, grid):
    from bayesreg.eval.fields import u_to_d as convert
    return convert(u, tuple(grid.spacing))


def grid_record(grid):
    return dict(shape_zyx=list(grid.shape), spacing_xyz=list(grid.spacing), origin_xyz=[float(a[0]) for a in grid.axes], unit=grid.unit, grid_id=grid.grid_id)


def field_problem(u, grid):
    """None, or why u cannot be an output (C16): float64 (nz, ny, nx, 3) of the grid, finite, six zero faces.  The supervisor asks it of the repaired field and of every fallback candidate before they can be
    chosen (revision 1, review minor 4), write_output once more."""
    u = np.asarray(u)
    if u.dtype != np.float64 or u.shape != grid.shape+(3,):
        return f'u {u.dtype} {u.shape}, not a float64 field of the grid {grid.shape}'
    if not np.all(np.isfinite(u)):
        return 'u has non-finite values'
    if not (u[0] == 0).all() or not (u[-1] == 0).all() or not (u[:, 0] == 0).all() or not (u[:, -1] == 0).all() or not (u[:, :, 0] == 0).all() or not (u[:, :, -1] == 0).all():
        return 'u: six zero faces required'
    return None


def write_output(directory, u, grid, cert_doc, line):
    """directory (new): final.npz (u as certified, state_hash, grid, line_status), disp.npz (d, meta; the C16 bench convention, bayesreg.solve.export.write_disp's fields), cert.json, line.json.
    The certificate must be of exactly this array and pass; nothing is written otherwise."""
    directory = Path(directory)
    u = np.ascontiguousarray(np.asarray(u))
    digest = array_sha256(u)
    problem = field_problem(u, grid)
    if problem:
        raise ValueError(f'output {problem}')
    if cert_doc.get('state_hash') != digest or not passes(cert_doc):
        raise ValueError('the certificate is not a passing certificate of this array (hard rule 1)')
    directory.mkdir(parents=True, exist_ok=False)
    status = line['line_status']
    np.savez(directory/'final.npz', u=u, state_hash=np.array(digest), grid=np.array(json.dumps(grid_record(grid))), status=np.array(status))
    d = u_to_d(u, grid)
    meta = dict(convention='pull-back: warped(p) = moving(p + d(p)), p = (z, y, x) voxel index; d[k] = u[..., 2 - k] / h_(z,y,x)[k]; same semantics as Learn2Reg evaluation.py', units='voxels', dtype='float64',
                layout='(3, nz, ny, nx)', grid=grid_record(grid), u_state_hash=digest, d_sha256=array_sha256(d), exact_inverse=bool(all(h == 1. for h in grid.spacing)), line_status=status)
    np.savez(directory/'disp.npz', d=d, meta=np.array(json.dumps(meta)))
    write_json(directory/'cert.json', cert_doc)
    files = dict(final=dict(file='final.npz', sha256=core.sha_file(directory/'final.npz'), state_hash=digest), disp=dict(file='disp.npz', sha256=core.sha_file(directory/'disp.npz'), d_sha256=meta['d_sha256']),
                 cert=dict(file='cert.json', sha256=core.sha_file(directory/'cert.json')))
    write_json(directory/'line.json', dict(line, output=files, ended_utc=utc()))
    return files


def read_output(directory, grid=None):
    """(u, line, cert): the output as written, checked: u's hash the final.npz state_hash and the certificate's."""
    directory = Path(directory)
    with np.load(directory/'final.npz', allow_pickle=False) as z:
        u = np.ascontiguousarray(np.array(z['u']))
        stored = str(z['state_hash'].item())
    if array_sha256(u) != stored:
        raise ValueError(f'{directory}/final.npz: state_hash differs from u')
    line, cert = read_json(directory/'line.json'), read_json(directory/'cert.json')
    if cert is None or cert.get('state_hash') != stored:
        raise ValueError(f'{directory}/cert.json is not of final.npz')
    if grid is not None and u.shape != grid.shape+(3,):
        raise ValueError('final.npz is not of the pair grid')
    return u, line, cert


# ---- code identity (RP review, new minor 1 and 3) ---------------------------------------------------------------------------------------------
def git(args, cwd):
    done = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
    return done.returncode, done.stdout, done.stderr.strip()


def can_shadow(path):
    """Whether a top-level entry of a PYTHONPATH directory can be imported in place of a module found later: a regular package (a directory with __init__), a module or an extension module.  A directory without
    __init__ is only a namespace portion, which a regular package found later takes precedence over (PEP 420)."""
    path = Path(path)
    suffixes = tuple(importlib.machinery.all_suffixes())
    if path.is_dir():
        return any((path/('__init__'+s)).exists() for s in suffixes)
    return path.name.endswith(suffixes) and path.name.split('.')[0].isidentifier()


CLEAN_PATHS = ('bayesreg', 'experiments', 'configs', 'splits', 'jd_curl', 'scripts')     # the code and the configuration (the allowlist, the registries, the splits); docs/, runs/, the project status log are not checked (revision 1, review important 2)
MODULE_DIR = 'experiments'                      # its subdirectories are put first on sys.path (he_core, run_hx, run_fl, ...): an untracked module there is imported before the real one


def _in_pycache(path):
    return '__pycache__' in Path(path).parts


def status_entries(code_root, paths):
    """(rc, [(XY, path)], err): git status of `paths`, every untracked and every ignored file listed, renames off (one path per entry), NUL separated."""
    rc, out, err = git(['status', '--porcelain=v1', '-z', '--untracked-files=all', '--ignored=matching', '--no-renames', '--', *paths], code_root)
    entries = [(e[:2], e[3:]) for e in out.split('\0') if e]
    return rc, entries, err


def untracked_modules(code_root, directory=MODULE_DIR):
    """Untracked (or ignored) files under `directory` that Python can import as a module (an import suffix; __pycache__ aside): refused always, whatever require_clean says."""
    rc, out, err = git(['ls-files', '--others', '-z', '--', directory], code_root)      # no --exclude-standard: the ignored ones too
    suffixes = tuple(importlib.machinery.all_suffixes())
    return rc, sorted(p for p in out.split('\0') if p and p.endswith(suffixes) and not _in_pycache(p)), err


def repo_identity(code_root, *, top_level=REPO_TOP_LEVEL, require_clean=True):
    """(record, problems): the code checkout on PYTHONPATH.
    - Every tracked top-level entry (git ls-tree HEAD) is one of `top_level` (the first path segment of anything added must be an existing top-level entry); no untracked top-level entry can shadow a module.
    - No untracked or ignored importable file under experiments/ (its subdirectories are first on sys.path).
    - require_clean (every HE process started from the command line): the code and configuration paths CLEAN_PATHS are committed and clean, no changed, untracked or ignored file there but inside __pycache__
      (git status --untracked-files=all --ignored=matching); then HEAD is the code that runs, and the allowlist and the registries are the committed ones.  A test of a made-up repository passes False."""
    code_root = Path(code_root)
    problems = []
    rc, out, err = git(['ls-tree', '--name-only', '-z', 'HEAD'], code_root)
    if rc:
        return dict(code_root=str(code_root), error=err), [f'git ls-tree HEAD in {code_root}: {err}']
    tracked = sorted(x for x in out.split('\0') if x)
    added = [x for x in tracked if x not in top_level]
    if added:
        problems.append(f'top-level entries that are not the existing ones ({TOP_LEVEL_COMMIT}): {added}')
    untracked = sorted(p.name for p in code_root.iterdir() if p.name not in tracked and p.name not in IGNORED_TOP)
    shadowing = [name for name in untracked if can_shadow(code_root/name)]
    if shadowing:
        problems.append(f'untracked top-level entries that can be imported as modules: {shadowing}')
    rc_m, modules, err_m = untracked_modules(code_root)
    if rc_m:
        problems.append(f'git ls-files --others {MODULE_DIR}: {err_m}')
    elif modules:
        problems.append(f'untracked or ignored modules under {MODULE_DIR}/ (first on sys.path): {modules[:10]}')
    rc_d, entries, err_d = status_entries(code_root, CLEAN_PATHS)
    dirty = [f'{xy} {p}' for xy, p in entries if not _in_pycache(p)]
    if require_clean and (rc_d or dirty):
        problems.append(f'the code and configuration paths {list(CLEAN_PATHS)} are not committed and clean (changed, untracked or ignored files): {dirty[:10] or err_d}')
    rc_h, head, _ = git(['rev-parse', 'HEAD'], code_root)
    rc_s, status, _ = git(['status', '--porcelain', '--untracked-files=no'], code_root)
    record = dict(code_root=str(code_root), head=head.strip() if rc_h == 0 else None, tracked_clean=(status.strip() == '') if rc_s == 0 else None, clean_paths=list(CLEAN_PATHS), dirty_paths=dirty,
                  code_paths_clean=rc_d == 0 and not dirty, require_clean=require_clean, tracked_top_level=tracked, added_top_level=added, untracked_top_level=untracked, untracked_shadowing=shadowing,
                  untracked_modules=modules, rule=f'tracked top-level entries are those of {TOP_LEVEL_COMMIT}; no untracked top-level module, package or extension; no untracked module under {MODULE_DIR}/; '
                  f'with require_clean the paths {list(CLEAN_PATHS)} committed and clean', same=not problems)
    return record, problems


def frozen_identity(frozen=None, commit=None, cwd=None):
    """(record, problems): the frozen tree first on PYTHONPATH.  experiments/fl/rp_core.py frozen_identity (every tracked file of eb808be under jd_curl/ and every tracked .py at the root: the same blob; no
    untracked .py there), plus: its top-level entries beyond eb808be's only data and .git; under jd_curl/ no untracked file of any kind but inside __pycache__ (RP review, new minor 1)."""
    import rp_core
    frozen = Path(core.FROZEN if frozen is None else frozen)
    commit = core.FROZEN_COMMIT if commit is None else commit
    cwd = core.ROOT if cwd is None else cwd
    record, problems = rp_core.frozen_identity(frozen, commit, cwd=cwd)
    problems = list(problems)
    rc, out, err = git(['ls-tree', '--name-only', '-z', commit], cwd)
    rc_j, out_j, err_j = git(['ls-tree', '-r', '--name-only', '-z', commit, '--', 'jd_curl'], cwd)
    if rc or rc_j:
        return dict(record, extra_error=err or err_j), problems+[f'git ls-tree {commit}: {err or err_j}']
    tracked_top = {x for x in out.split('\0') if x}
    extra_top = sorted(p.name for p in frozen.iterdir() if p.name not in tracked_top and p.name not in FROZEN_UNTRACKED_TOP) if frozen.is_dir() else ['<the frozen tree is missing>']
    tracked_jd = {x for x in out_j.split('\0') if x}
    untracked_jd = sorted(str(p.relative_to(frozen)) for p in (frozen/'jd_curl').rglob('*') if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.relative_to(frozen).parts
                          and str(p.relative_to(frozen)) not in tracked_jd) if (frozen/'jd_curl').is_dir() else []
    if extra_top:
        problems.append(f'the frozen tree {frozen} has top-level entries beyond {commit[:7]} other than data and .git: {extra_top}')
    if untracked_jd:
        problems.append(f'the frozen tree has untracked files under jd_curl/: {untracked_jd[:10]}')
    record = dict(record, top_level_beyond_commit=extra_top, untracked_under_jd_curl=untracked_jd, allowed_untracked_top_level=sorted(FROZEN_UNTRACKED_TOP), same=not problems)
    return record, problems


def pythonpath_problems(pythonpath, code_root, frozen=None):
    """The unit's PYTHONPATH must be exactly <frozen>:<code root>, in this order: the names it can import are those two trees' (both checked)."""
    frozen = Path(core.FROZEN if frozen is None else frozen)
    entries = [e for e in (pythonpath or '').split(':') if e]
    want = [str(frozen.resolve()), str(Path(code_root).resolve())]
    got = [str(Path(e).resolve()) for e in entries]
    return [] if got == want else [f'PYTHONPATH {entries} is not {want}']


def identity(code_root=None, *, frozen=None, pythonpath=None, check_pythonpath=True, require_clean=True):
    """The start check of every HE process: PYTHONPATH, the code checkout (committed and clean in its code and configuration paths), the frozen tree.  (record, problems)."""
    code_root = Path(core.ROOT if code_root is None else code_root)
    problems = pythonpath_problems(os.environ.get('PYTHONPATH') if pythonpath is None else pythonpath, code_root, frozen) if check_pythonpath else []
    repo, repo_problems = repo_identity(code_root, require_clean=require_clean)
    frozen_record, frozen_problems = frozen_identity(frozen, cwd=code_root)
    problems = problems+repo_problems+frozen_problems
    return dict(pythonpath=os.environ.get('PYTHONPATH') if pythonpath is None else pythonpath, pythonpath_checked=check_pythonpath, repo=repo, frozen=frozen_record, same=not problems, checked_utc=utc()), problems
