"""P1 (runs/20261105-p1/任务书.md, plan.md; docs/bayesreg/notes/贝叶斯化Hx线_设计_20261105.md §5.2): where the Bayesian parts land, a CPU diagnostic on the 19 repaired
fields of RW.  No registration, no certificate, no GPU, no truth label, no metrics file, no Dice.  The numbers: p1_core.py.

    python -B experiments/p1/run_p1.py --dry-run
        the inputs of the 19 pairs and their sha256 against the records (field_0600.npz, repair.npz, pair.npz, probs36.npz of both images, the fixed image's
        Atropos posteriors), the code digest, the output root, the unit's start strings and the estimate.  Hashes files; decodes no array, computes nothing.
        `problems: none` or the list, exit 8.
    python -B experiments/p1/run_p1.py run --commit SHA --code-digest HEX
        only in the unit the dry run prints (INVOCATION_ID and the cgroup of tag-p1.service): from the main checkout at that commit, tracked files clean, the
        code digest the same and every imported module from its digested file, the frozen tree's jd_curl, no tag-rp* or seq-vp-* unit active or activating,
        a tag-uf* unit only while one UF member trains (its train step started at most 60 min ago), no GPU, one thread; any of these not so: nothing is written.
        Control (ii), then pair by pair: the records, control (i), the readers, the fields, P1a, P1b, control (iii); a control or an input check that fails
        stops the run there (nothing more is read, exit 8, the evidence is kept).  Then summary.json, summary.md, per_channel.csv; control (iii) out of its
        line on any pair holds the readings of A and C (revision 1).  Never reruns: a root with started.json is refused.

The only output root is /home/rmcfz/seq_vp_tag_results/p1_v1 (small files only, no field).  Inputs, read only: rw_v1 (repair.json, repair.npz), fd2_v1 (run.json,
field_0600.npz), experiments/fl/rw_sources_v1.json, the development stage through bayesreg.io.pair.load_pair (fixed, moving), the predictions of unet35_v1_ens
through fast_channels.read_predicted, the fixed image's Atropos posteriors through bayesreg.cli.run_pair.load_tissue_layer.
Needs PYTHONPATH=<frozen eb808be>:<code> for run (the taper of the fields is the frozen boundary_taper).  Exit codes: 0 done, 2 usage, 8 a refusal or a stop.
"""
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import traceback
from time import perf_counter

HERE = Path(__file__).resolve().parent
for i, sub in enumerate(('', '../fl', '../ef', '../mc0')):
    path = str((HERE/sub).resolve())
    if path not in sys.path:
        sys.path.insert(i, path)

import numpy as np  # noqa: E402

import fast_channels as fast  # noqa: E402
import fd2_core as fd2  # noqa: E402
import fl_core as fl  # noqa: E402
import fx_core as fx  # noqa: E402
import mc0_core as core  # noqa: E402
import mc0p_core as mc  # noqa: E402
import p1_core as p1  # noqa: E402

SCRIPT = 'experiments/p1/run_p1.py'
RESULTS = fl.RESULTS
ROOT_NAME = 'p1_v1'
DEFAULT_ROOT = RESULTS/ROOT_NAME
RW_ROOT, FD2_ROOT = 'rw_v1', 'fd2_v1'
PAIRS = fx.ALL_PAIRS                            # the 19 development pairs of RW, in their order: pair i = subjects 438 + i, 439 + i
UNIT = 'tag-p1'
MEMORY_MAX, SWAP_MAX, RUNTIME_MAX_SEC, CPU_QUOTA, NICE = '4G', '0', 7200, '100%', 19
BUSY_PATTERNS = ('tag-rp*', 'seq-vp-*')         # the timed RP repairs (runs/20261105-rp) and the other line: P1 never runs beside them
UNIT_STATES = 'active,activating'               # revision 1 (review minor 9): the states the other tools' gates count
UF_PATTERNS = ('tag-uf*',)                      # UF's chain may run beside P1, but only while a member trains (revision 1, review important 4)
UF_STATE_DIR = RESULTS/'seg'/'unet35_v4_full'/'chain_state'     # uf-tools 5015edd: uf_common.OUT, run_uf.run_step writes <step>.started and <step>.done there
UF_TRAIN_STEPS = tuple(f'train_fold{f}_s{s}' for f, s in (('A', 0), ('B', 0), ('A', 1), ('B', 1)))     # run_uf.chain_steps' names (uf_common.member_name)
UF_MAX_TRAIN_MINUTES = 60.                      # a member trains about 2.4-2.5 h (UF plan's estimate); started at most this long ago leaves P1 (about 25 min) far inside the step
SCHEMA = 'bayesreg.p1.v1'
ROLE = 'P1: CPU diagnostic on the repaired fields of RW (development set, single observation): EM channel weights and tissue moment ratios; no registration, no truth label, no Dice; NOT a method, NOT in any result table'
FIELD_FILE, REPAIR_FILE = 'field_0600.npz', 'repair.npz'
STEP = '600'
CODE_FILES = ('experiments/p1/run_p1.py', 'experiments/p1/p1_core.py', 'experiments/mc0/mc0_core.py', 'experiments/mc0/mc0p_core.py', 'experiments/ef/fast_channels.py',
              'experiments/fl/fl_core.py', 'experiments/fl/fd2_core.py', 'experiments/fl/fx_core.py', 'experiments/fl/rw_sources_v1.json', 'experiments/d0/make_inputs.py',
              'bayesreg/io/pair.py', 'bayesreg/seg/channels.py', 'bayesreg/seg/structure_layer.py', 'bayesreg/cli/run_pair.py')
SECONDS_PER_PAIR = 75.                          # 〔估算〕: the native-size synthetic timing of plan.md, rounded up
ENV_ONE_THREAD = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


class Stop(SystemExit):
    """A refusal or a failed control: the run ends here, what was written stays."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=_plain)+'\n', encoding='utf-8')
    tmp.replace(path)


def _plain(value):
    if isinstance(value, np.ndarray):
        return [None if not np.isfinite(x) else float(x) for x in value.ravel()] if value.dtype.kind == 'f' else value.tolist()
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'{type(value)} is not JSON')


def floats(array):
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(array, dtype=np.float64)]


def sha_bytes(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_record(path):
    """A JSON record of the inputs.  Never a metrics file or a table of results: the diagnostic reads no evaluation."""
    name = Path(path).name
    if 'metrics' in name or not name.endswith('.json'):
        raise Stop(f'P1 reads no evaluation and no table: {path}')
    if not Path(path).is_file():
        raise Stop(f'{path}: missing')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require_pair(pair):
    """Checked before any file of the pair is opened: only the 19 development pairs of RW."""
    if pair not in PAIRS:
        raise Stop(f'P1 runs only the 19 development pairs of RW ({core.short_name(PAIRS[0])} .. {core.short_name(PAIRS[-1])}); refused: {pair}')
    return pair


# ---- where everything is --------------------------------------------------------------------------------------------------------------------------
@dataclass
class World:
    """Every path the diagnostic reads.  None: the module's own (the real ones); a test passes its synthetic tree."""
    results: Path = None
    sources_file: Path = None
    stage: Path = None
    allowlist: Path = None
    seg_root: Path = None
    unet: object = None
    taper: object = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.results = Path(RESULTS if self.results is None else self.results)
        self.sources_file = Path(fx.RW.SOURCES_FILE if self.sources_file is None else self.sources_file)
        self.stage = Path(core.STAGE if self.stage is None else self.stage)
        self.allowlist = Path(core.ALLOWLIST if self.allowlist is None else self.allowlist)
        self.seg_root = Path(mc.SEG_ROOT if self.seg_root is None else self.seg_root)

    def unet_source(self):
        return mc.default_unet() if self.unet is None else self.unet

    def taper_function(self):
        return core.frozen_api().boundary_taper if self.taper is None else self.taper

    def rw_dir(self, pair):
        return fx.RW.run_dir(self.results/RW_ROOT, fx.RW.cell_of(pair))

    def fd2_dir(self, pair):
        return fd2.run_dir(self.results/FD2_ROOT, fx.RW.fd2_field_cell(pair))


def record_problems(pair, run, repair, pin, world):
    """What is wrong with the two JSON records of a pair and the pin of rw_sources_v1.json; [] when nothing.  No field is read here."""
    out = []

    def need(ok, text):
        if not ok:
            out.append(f'{core.short_name(pair)}: {text}')
    checkpoint = ((run.get('checkpoints') or {}).get(STEP)) or {}
    scale = run.get('scale') or {}
    sources = run.get('channel_sources') or {}
    need(run.get('pair') == pair and repair.get('pair') == pair, 'the records are of another pair')
    need(run.get('status') == 'done' and run.get('oracle') is False and sources.get('oracle') is False and sources.get('truth_labels_read') is False, f"fd2 run.json: status {run.get('status')}, oracle {run.get('oracle')}, channel sources oracle {sources.get('oracle')}, truth labels read {sources.get('truth_labels_read')}")
    need((run.get('arm'), run.get('k'), run.get('channels'), run.get('weight'), run.get('D_factor')) == (fx.SOURCE_ARM, 35, 'onehot', 'Q3h', fx.D_FACTOR), f"fd2 run.json is not Hx24: {run.get('arm')} k {run.get('k')} {run.get('channels')} {run.get('weight')} D {run.get('D_factor')}")
    need(checkpoint.get('file') == FIELD_FILE and checkpoint.get('sha256') == pin.get('field_sha256') and checkpoint.get('u_sha256') == pin.get('u_sha256'), f"fd2 checkpoint {STEP} is not the pin of {world.sources_file.name}")
    need(len(scale.get('channel_weights') or []) == 35 and len(scale.get('fixed_voxels') or []) == 35 and set(run.get('labels_sha256') or {}) == {'fixed', 'moving'}, 'fd2 run.json has no Q3h weights, voxel counts or label hashes')
    need(repair.get('status') == fx.CERTIFIED and repair.get('passed') is True and repair.get('oracle') is False, f"rw repair.json: status {repair.get('status')}, passed {repair.get('passed')}, oracle {repair.get('oracle')}")
    cert = repair.get('certificate') or {}
    need(cert.get('status') == 'certified_feasible' and cert.get('n_infeasible') == 0 and cert.get('n_uncertain') == 0 and isinstance(cert.get('state_hash'), str), f"rw certificate {cert.get('status')}, infeasible {cert.get('n_infeasible')}, uncertain {cert.get('n_uncertain')}")
    config = repair.get('certificate_config') or {}
    need((config.get('delta'), config.get('max_depth'), config.get('max_boxes')) == (0., 48, 255), f'rw certificate configuration is not C: {config}')
    source = repair.get('source') or {}
    need(source.get('field_sha256') == pin.get('field_sha256') and source.get('tapered_u_sha256') == pin.get('tapered_u_sha256') and source.get('file') == FIELD_FILE, 'rw repair.json source is not the pinned field')
    need(source.get('directory') is not None and Path(source['directory']).resolve() == world.fd2_dir(pair).resolve(), f"rw repair.json source directory {source.get('directory')} is not {world.fd2_dir(pair)}")
    attempt = repair.get('attempt') or {}
    need(attempt.get('file') == REPAIR_FILE and isinstance(attempt.get('sha256'), str), 'rw repair.json has no attempt')
    need((repair.get('data') or {}).get('pair_sha256') == (run.get('data') or {}).get('pair_sha256'), 'the two records are of different pair.npz')
    return out


def records_of(pair, world):
    """(run.json of fd2_v1, repair.json of rw_v1, the pin) after record_problems; Stop otherwise."""
    require_pair(pair)
    pins = json.loads(world.sources_file.read_text())['pairs']
    pin = pins.get(core.short_name(pair))
    if pin is None:
        raise Stop(f'{world.sources_file}: no pin for {pair}')
    run = read_record(world.fd2_dir(pair)/'run.json')
    repair = read_record(world.rw_dir(pair)/'repair.json')
    problems = record_problems(pair, run, repair, pin, world)
    if problems:
        raise Stop('; '.join(problems))
    return run, repair, pin


def file_problems(pair, run, repair, world):
    """The sha256 of the two field files against their records."""
    out = []
    field_path, repair_path = world.fd2_dir(pair)/FIELD_FILE, world.rw_dir(pair)/REPAIR_FILE
    for path, want in ((field_path, run['checkpoints'][STEP]['sha256']), (repair_path, repair['attempt']['sha256'])):
        got = core.sha_file(path) if path.is_file() else None
        if got != want:
            out.append(f'{core.short_name(pair)}: {path} sha256 {got} is not {want}')
    return out


# ---- the fields ----------------------------------------------------------------------------------------------------------------------------------------
def zero_faces(u):
    return all(not np.any(a) for a in (u[0], u[-1], u[:, 0], u[:, -1], u[:, :, 0], u[:, :, -1]))


def fields_of(pair, run, repair, grid, world):
    """(field_0600 as T88 left it, the repaired field u_hat), both (D, H, W, 3) mm, after every check: the files against their records, the raw field against
    u_sha256, the taper against the repair's source, u_hat against the certificate's state hash, zero faces.  Stop otherwise."""
    problems = file_problems(pair, run, repair, world)
    if problems:
        raise Stop('; '.join(problems))
    with np.load(world.fd2_dir(pair)/FIELD_FILE) as z:
        raw = np.array(z['u'])
    if sha_bytes(raw) != run['checkpoints'][STEP]['u_sha256']:
        raise Stop(f'{core.short_name(pair)}: field_0600 u is not its u_sha256')
    tapered, _ = world.taper_function()(raw, grid, core.TAPER)
    tapered = np.ascontiguousarray(tapered)
    if sha_bytes(tapered) != repair['source']['tapered_u_sha256']:
        raise Stop(f'{core.short_name(pair)}: the taper of field_0600 is not the field the repair started from')
    with np.load(world.rw_dir(pair)/REPAIR_FILE) as z:
        nodes, values = np.array(z['nodes']), np.array(z['values'])
    u_hat = np.array(tapered, copy=True)
    del tapered
    u_hat.reshape(-1, 3)[nodes] = values
    if sha_bytes(u_hat) != repair['certificate']['state_hash']:
        raise Stop(f'{core.short_name(pair)}: the repaired field is not the certified state (state_hash)')
    if not (zero_faces(raw) and zero_faces(u_hat)):
        raise Stop(f'{core.short_name(pair)}: a field with nonzero faces')
    return raw, u_hat, dict(field_sha256=run['checkpoints'][STEP]['sha256'], u_sha256=run['checkpoints'][STEP]['u_sha256'], tapered_u_sha256=repair['source']['tapered_u_sha256'],
                            repair_npz_sha256=repair['attempt']['sha256'], repaired_u_sha256=repair['certificate']['state_hash'], n_repaired_nodes=int(len(nodes)))


# ---- one pair ------------------------------------------------------------------------------------------------------------------------------------------
def read_images(pair, world):
    """The firewall view of pair.npz: load_pair reads fixed, moving (and the all-zero initial_u), nothing else; p1_core.check_keys_read stops on any other member."""
    require_pair(pair)
    from bayesreg.io.pair import load_pair
    pin = load_pair(world.stage/pair/'pair.npz', allowlist=world.allowlist)
    p1.check_keys_read(pin.provenance['keys_read'])
    return pin


def control_i(labels, run):
    """(i): the predicted label maps' sha256, the fixed image's voxel counts V_c and the Q3h weights, each equal to fd2_v1's run.json, bit for bit."""
    digest = {side: sha_bytes(labels[side]) for side in ('fixed', 'moving')}
    w, voxels = mc.channel_weights('Q3h', labels['fixed'])
    checks = dict(labels_sha256=digest == run['labels_sha256'], fixed_voxels=voxels.tolist() == run['scale']['fixed_voxels'], channel_weights=w.tolist() == run['scale']['channel_weights'],
                  q3h_formula=np.array_equal(w, p1.q3h_weights(voxels)))
    return dict(control='i', passed=all(checks.values()), checks=checks, labels_sha256=digest), w, voxels


def tissue_of(pin, world):
    """The fixed image's tissue layer: run_pair.load_tissue_layer (C03.1-2/3, hashes, grid), C03.2 simplex with background, C03.3 brain mask, C03.5 smooth and floor."""
    from bayesreg.cli.run_pair import load_tissue_layer
    from bayesreg.seg.channels import brain_mask, simplex_with_background, tissue_channels
    P, accepted, record = load_tissue_layer(pin, 'fixed', world.seg_root)
    pbar, simplex = simplex_with_background(P)
    del P
    brain, mask = brain_mask(pbar)
    ptilde, smoothing = tissue_channels(pbar)
    del pbar
    names = list(accepted['class_names'])
    if not set(p1.C_CLASSES) <= set(names) or names[int(accepted['wm_index'])] in p1.C_CLASSES:
        raise Stop(f"{pin.fixed_subject}: Atropos class names {names}, wm_index {accepted['wm_index']}: the reading needs {list(p1.C_CLASSES)} and white matter")
    record = dict(record, accepted=dict(accepted), simplex=simplex, brain_mask=mask, smoothing=smoothing)
    return brain, ptilde, names, int(accepted['wm_index']), record


def p1a(sums, voxels, w_q3h, n_brain):
    """Per field and domain: the weights, sigma^2, the labels left out, beta, R^2, the group ratios, d_A (the reading uses brain on u_hat; the rest is description)."""
    out = {}
    for name in p1.FIELDS:
        out[name] = {}
        for domain in p1.DOMAINS:
            w, sigma2, left_out = p1.em_weights(sums[name], voxels, n_brain, domain)
            d, n = p1.distance_to_q3h(w, w_q3h)
            out[name][domain] = dict(w=floats(w), sigma2=sigma2 if not isinstance(sigma2, np.ndarray) else floats(sigma2), left_out=left_out, fit=p1.fit_beta(w, voxels), group_ratio=p1.group_ratios(w, w_q3h),
                                     distance=dict(d=d, n=n))
    return out


def compute_pair(pair, world, log=print):
    """Everything of one pair, in this order (each check before what needs it): the records, the images, the predictions, control (i), the tissue layer, the
    fields, P1b, then the 35 channels one at a time.  Returns the pair's record; control_i.passed False: the record holds no number past it."""
    t0 = perf_counter()
    run, repair, pin_record = records_of(pair, world)
    pin = read_images(pair, world)
    if pin.provenance['pair_sha256'] != run['data']['pair_sha256']:
        raise Stop(f'{core.short_name(pair)}: pair.npz is not the one fd2_v1 ran on')
    predicted = fast.read_predicted(pin, unet=world.unet_source(), seg_root=world.seg_root, allowlist=world.allowlist)
    if predicted.record['oracle'] is not False or predicted.record['truth_labels_read'] is not False:
        raise Stop(f"{core.short_name(pair)}: the channel sources say oracle {predicted.record['oracle']}, truth labels read {predicted.record['truth_labels_read']}")
    p1.check_keys_read(predicted.record['pair_keys_read'])
    labels = predicted.labels
    first, w_q3h, voxels = control_i(labels, run)
    out = dict(schema=SCHEMA, pair=pair, short=core.short_name(pair), index=PAIRS.index(pair), role=ROLE, oracle=False, control_i=first,
               inputs=dict(fd2_run=str(world.fd2_dir(pair)/'run.json'), rw_repair=str(world.rw_dir(pair)/'repair.json'), pin=pin_record, pair_sha256=pin.provenance['pair_sha256'], keys_read=list(pin.provenance['keys_read']),
                           channel_sources=dict(oracle=predicted.record['oracle'], truth_labels_read=predicted.record['truth_labels_read'], segmenter=predicted.record['segmenter'],
                                                sides={s: {k: predicted.record['sides'][s].get(k) for k in ('subject', 'segmenter_id', 'policy', 'unet_output_sha256', 'read_path', 'alignment_qc')} for s in ('fixed', 'moving')})))
    log(f"{out['short']}: control (i) {'passed' if first['passed'] else 'NOT passed'} {first['checks']}")
    if not first['passed']:
        return out
    del predicted
    brain, ptilde, class_names, wm_index, tissue = tissue_of(pin, world)
    spacing = tuple(float(s) for s in pin.grid.spacing)
    raw, v, field_record = fields_of(pair, run, repair, pin.grid, world)
    u_hat, u_600 = p1.voxel_field(v, spacing), p1.voxel_field(raw, spacing)
    del v, raw
    config = run['config']
    j0, kappa, mu = float(config['j0']), float(config['kappa_pre']), float(config['mu'][3])
    geometry_hat = p1.geometry_sums(u_hat, spacing, tissue=ptilde, brain=brain, j0=j0)
    geometry_600 = p1.geometry_sums(u_600, spacing, tissue=ptilde, brain=brain, j0=j0)     # revision 1 (review minor 4): T88's own end state, description
    del ptilde
    coords_hat, coords_600 = p1.sample_coordinates(u_hat), p1.sample_coordinates(u_600)
    del u_hat, u_600
    L_F, L_M = labels['fixed'], labels['moving']
    boxes_f, boxes_m = p1.label_boxes(L_F), p1.label_boxes(L_M)
    sums = dict(u0=[], uhat=[], f600=[])
    for index in range(len(p1.LABELS)):
        F, M = p1.onehot_channel(L_F, index, boxes_f), p1.onehot_channel(L_M, index, boxes_m)
        sums['u0'].append(p1.channel_sums(F, p1.warp(M, None), brain))
        sums['uhat'].append(p1.channel_sums(F, p1.warp(M, coords_hat), brain))
        sums['f600'].append(p1.channel_sums(F, p1.warp(M, coords_600), brain))
    del coords_hat, coords_600
    n_brain = int(np.count_nonzero(brain))
    s, w_run = float(run['scale']['D_multiplier']), np.asarray(run['scale']['channel_weights'], dtype=np.float64)
    recomputed = dict(data=.5*float(sum(s*w_run[i]*sums['f600'][i]['R_grid'] for i in range(len(p1.LABELS)))), geometry=.5*mu*(geometry_600['sum_e']+geometry_600['sum_q']),
                      floor=kappa*geometry_600['sum_floor0'], min_det=geometry_600['min_det'])
    recorded = {k: run['checkpoints'][STEP]['terms'][k] for k in ('data', 'geometry', 'floor', 'min_det')}
    out.update(
        n_brain=n_brain, fixed_voxels=voxels.tolist(), w_q3h=w_q3h.tolist(), spacing=list(spacing), fields=field_record, tissue=tissue, sums=sums,
        p1a=p1a(sums, voxels, w_q3h, n_brain),
        p1b=dict(geometry=geometry_hat, moments=p1.tissue_moments(geometry_hat, class_names, wm_index), field='u_hat (the certified repaired field)',
                 described=dict(uhat_brain_cells=p1.tissue_moments(geometry_hat, class_names, wm_index, '_brain'), f600_all_cells=p1.tissue_moments(geometry_600, class_names, wm_index),
                                f600_brain_cells=p1.tissue_moments(geometry_600, class_names, wm_index, '_brain'))),
        control_iii=dict(control='iii', role=f'the recorded terms are of the energy before step 600 (float32, GPU), the file is the field after it; |recomputed / recorded - 1| > {p1.III_TOL} for D or G flags the pair (revision 1)',
                         recomputed=recomputed, recorded=recorded, ratio={k: (recomputed[k]/recorded[k] if recorded[k] else None) for k in recomputed}, geometry_600=geometry_600, mu=mu, kappa=kappa, j0=j0, D_multiplier=s),
        seconds=perf_counter()-t0, host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    return out


# ---- the 19 pairs: the summary and the reading ----------------------------------------------------------------------------------------------------------
def _stats(values):
    x = [v for v in values if v is not None]
    if not x:
        return dict(n=0, median=None, min=None, max=None)
    return dict(n=len(x), median=float(np.median(x)), min=float(np.min(x)), max=float(np.max(x)))


def summarize(records, control_ii_record):
    """The 19 pair records -> summary (every number) and the preregistered reading (p1_core.decide)."""
    if len(records) != len(PAIRS) or [r['pair'] for r in records] != list(PAIRS):
        raise Stop('summary: the 19 pair records in order are needed')
    index = [r['index'] for r in records]
    controls = bool(control_ii_record['passed']) and all(r['control_i']['passed'] for r in records)
    a = {}
    for name in p1.FIELDS:
        a[name] = {}
        for domain in p1.DOMAINS:
            fits = [r['p1a'][name][domain]['fit'] for r in records]
            a[name][domain] = dict(beta=_stats([f['beta'] for f in fits]), r2=_stats([f['r2'] for f in fits]), beta_per_pair=[f['beta'] for f in fits],
                                   group_ratio={g: _stats([r['p1a'][name][domain]['group_ratio'][g] for r in records]) for g in p1.GROUPS},
                                   n_left_out=sum(len(r['p1a'][name][domain]['left_out']) for r in records))
        brain, support = a[name]['brain']['beta_per_pair'], a[name]['support']['beta_per_pair']
        a[name]['beta_brain_minus_support'] = _stats([x-y if x is not None and y is not None else None for x, y in zip(brain, support)])
        for domain in p1.DOMAINS:
            d = [r['p1a'][name][domain]['distance']['d'] for r in records]
            a[name][domain]['distance'] = dict(_stats(d), per_pair=d, block_linear=p1.block_interval(d, index) if all(x is not None for x in d) else None)
    b = dict(uhat=moments_summary([r['p1b']['moments'] for r in records], index))
    for key in ('uhat_brain_cells', 'f600_all_cells', 'f600_brain_cells'):                    # description only (revision 1, review minor 3, 4)
        b[key] = moments_summary([r['p1b']['described'][key] for r in records], index)
    iii = [dict(pair=r['short'], **{f'{k}_recomputed': r['control_iii']['recomputed'][k] for k in ('data', 'geometry', 'floor', 'min_det')}, **{f'{k}_recorded': r['control_iii']['recorded'][k] for k in ('data', 'geometry', 'floor', 'min_det')},
                ratio=r['control_iii']['ratio']) for r in records]
    abnormal = p1.iii_abnormal(iii)
    reading = p1.decide(controls, a['uhat']['brain']['distance']['median'], a['uhat']['brain']['beta']['median'],
                        {k: b['uhat'][k]['block_log'] for k in p1.C_CLASSES}, {k: b['uhat'][k]['ratio_to_wm']['median'] for k in p1.C_CLASSES}, abnormal)
    return dict(schema=SCHEMA, role=ROLE, n_pairs=len(records), pairs=[r['short'] for r in records], controls=dict(i=all(r['control_i']['passed'] for r in records), ii=control_ii_record['passed'], ii_max_relative=control_ii_record['max_relative'],
                                                                                                             iii_abnormal=abnormal),
                p1a=a, p1b=b, control_iii=iii, reading=reading,
                lines=dict(a_null=p1.A_NULL, beta_description=p1.BETA_LINE, null_band=p1.NULL_BAND, substantial=p1.SUBSTANTIAL, c_classes=p1.C_CLASSES, iii_tol=p1.III_TOL,
                           interval='95 % subject-block bootstrap of the geometric mean over pairs', boot_n=p1.BOOT_N, boot_seed=p1.BOOT_SEED, revision=1))


def moments_summary(moments, index):
    """Per class: the ratio to white matter over the pairs (median, range, per pair, block intervals on the log and the linear scale); the curl / volume forms."""
    out = {}
    for cls in moments[0]['classes']:
        ratios = [m['ratio_to_wm'][cls] for m in moments]
        out[cls] = dict(ratio_to_wm=_stats(ratios), per_pair=ratios, block_log=p1.block_interval(ratios, index, log=True) if all(x and x > 0 for x in ratios) else None,
                        block_linear=p1.block_interval(ratios, index))
    for key in ('curl_to_det', 'b_per_component', 'b_helmholtz'):
        out[key] = _stats([m[key] for m in moments])
    return out


def _f(x, digits=4):
    return '—' if x is None else f'{x:.{digits}f}'


def summary_markdown(summary):
    a, b, reading = summary['p1a'], summary['p1b'], summary['reading']
    L = ['# P1 结果（自动生成；开发集 19 对单次；不是 Dice；独立复核 pending）', '', ROLE, '',
         f"控制：(i) {'通过' if summary['controls']['i'] else '不过'}；(ii) {'通过' if summary['controls']['ii'] else '不过'}（最大相对差 {summary['controls']['ii_max_relative']:.1e}）。", '']
    if reading['stopped']:
        return '\n'.join(L+[reading['text'], ''])
    A, C, bu = reading['A'], reading['C'], b['uhat']
    da = a['uhat']['brain']['distance']
    held = reading['held']
    L += ['## 读法（预登记，plan.md 修订 1）', '']
    if held:
        L += [f"**{reading['text']}**：异常的对 " + '、'.join(f"{x['pair']}（{'、'.join(x['terms'])}）" for x in reading['abnormal']) + '。下面两行是暂缓的读数，不决定进不进 P2。', '']
    beyond = '；'.join(f"{x['cls']}/WM 中位 {x['median']:.3f}，{x['direction']}" for x in C.get('beyond', []))
    L += [f"- A：d_A = RMS_c |log(w_EM,brain(û)/w_Q3h)| 的 19 对中位 {_f(da['median'])}（[{_f(da['min'])}, {_f(da['max'])}]），空操作线 ≤ log 1.25 = {p1.A_NULL:.4f}：**{A['reading']}**。"
          f"体积指数（只描述）：β 中位 {_f(a['uhat']['brain']['beta']['median'])}，{A['beta_band']}。",
          f"- C：τ̂²/τ̂²_WM 的块区间（几何均值）CSF [{_f(bu['CSF']['block_log']['lo'], 3)}, {_f(bu['CSF']['block_log']['hi'], 3)}]、GM [{_f(bu['GM']['block_log']['lo'], 3)}, {_f(bu['GM']['block_log']['hi'], 3)}]，"
          f"中位 CSF {_f(bu['CSF']['ratio_to_wm']['median'], 3)}、GM {_f(bu['GM']['ratio_to_wm']['median'], 3)}；空操作带 [{p1.NULL_BAND[0]}, {p1.NULL_BAND[1]}]（两个都整个在里面），实质线 ≥ {p1.SUBSTANTIAL} 或 ≤ 1/{p1.SUBSTANTIAL}：**{C['reading']}**" + (f"（{beyond}）" if beyond else '') + '。',
          f"- 进 P2 的候选：{'暂缓，交总负责人' if held else ('、'.join(reading['to_p2']) or '无')}（P2 要 BZ 授权）。只描述的量不改变读法。", '',
          '## P1a：EM 权（19 对；中位 [最小, 最大]）', '', '| 场 | 估计域 | d_A | β | R² | 小六个 | 大六个 | 皮层 | 白质 |', '|---|---|---|---|---|---|---|---|---|']
    for name in p1.FIELDS:
        for domain in p1.DOMAINS:
            row = a[name][domain]
            cell = lambda s, d=4: f"{_f(s['median'], d)} [{_f(s['min'], d)}, {_f(s['max'], d)}]"     # noqa: E731
            L.append(f"| {name} | {domain} | {cell(row['distance'])} | {cell(row['beta'])} | {cell(row['r2'], 3)} | " + ' | '.join(cell(row['group_ratio'][g], 3) for g in p1.GROUPS) + ' |')
    L += ['', '组比 = 组内 Σ w_EM / Σ w_Q3h（两者都在出现的标签上归一到均值 1）。support_shared 按构造 w ≡ 1，β = 0，R² 无定义。读法只用 brain、û 的 d_A；û 上的 EM 权带着 Q3h 配准的痕迹，β 要和 u = 0 的一行并读。', '',
          f"β_brain − β_support（û）中位 {_f(a['uhat']['beta_brain_minus_support']['median'])}；（u = 0）{_f(a['u0']['beta_brain_minus_support']['median'])}。", '']
    for key, title in (('uhat', 'û，全部单元（读法用这一组）'), ('uhat_brain_cells', 'û，Ω_B 内的单元（只描述）'), ('f600_all_cells', 'field_0600，全部单元（只描述）'), ('f600_brain_cells', 'field_0600，Ω_B 内的单元（只描述）')):
        m = b[key]
        L += [f'## P1b：按组织的几何矩比（{title}；19 对）', '', '| 类 | τ̂²_k/τ̂²_WM 中位 [最小, 最大] | 块区间（几何均值） | 块区间（算术均值，只描述） |', '|---|---|---|---|']
        for cls in m:
            if cls in ('curl_to_det', 'b_per_component', 'b_helmholtz'):
                continue
            row = m[cls]
            bl, bn = row['block_log'], row['block_linear']
            log_cell = '—' if bl is None else f"[{bl['lo']:.3f}, {bl['hi']:.3f}]"
            L.append(f"| {cls} | {_f(row['ratio_to_wm']['median'], 3)} [{_f(row['ratio_to_wm']['min'], 3)}, {_f(row['ratio_to_wm']['max'], 3)}] | {log_cell} | [{bn['lo']:.3f}, {bn['hi']:.3f}] |")
        L += ['', f"b̂（只描述，P1 不判）：mean|curl|²/mean(det − 1)² 中位 {_f(m['curl_to_det']['median'], 3)}；每单元 3 个分量的计数 3·mean e/mean q 中位 {_f(m['b_per_component']['median'], 3)}；Helmholtz 1:2 的计数 2·mean e/mean q 中位 {_f(m['b_helmholtz']['median'], 3)}。", '']
    L += [f'## 控制 (iii)：field_0600 上重算的 T88 各项与 run.json 记录并列（记录是第 600 步之前的能量、FP32；D 或 G 的 |重算/记录 − 1| > {p1.III_TOL} 标异常）', '',
          '| 对 | D 重算 / 记录 | G 重算 / 记录 | floor 重算 / 记录 | min det 重算 / 记录 | 异常 |', '|---|---|---|---|---|---|']
    flagged = {x['pair'] for x in summary['controls']['iii_abnormal']}
    for r in summary['control_iii']:
        L.append(f"| {r['pair']} | {r['data_recomputed']:.1f} / {r['data_recorded']:.1f} | {r['geometry_recomputed']:.1f} / {r['geometry_recorded']:.1f} | {r['floor_recomputed']:.2f} / {r['floor_recorded']:.2f} | {r['min_det_recomputed']:.3f} / {r['min_det_recorded']:.3f} | {'**是**' if r['pair'] in flagged else '否'} |")
    L += ['', '它证伪不了什么：不说明 Dice 会怎样；只是一步 EM / 矩匹配，不是不动点（不动点要重跑 T88，在 P2）。', '']
    return '\n'.join(L)


def per_channel_rows(records):
    rows = []
    for r in records:
        for name in p1.FIELDS+('f600',):
            for i, label in enumerate(p1.LABELS):
                s = r['sums'][name][i]
                row = dict(pair=r['short'], field=name, label=label, fixed_voxels=r['fixed_voxels'][i], w_q3h=r['w_q3h'][i], R_grid=s['R_grid'], R_brain=s['R_brain'], R_support=s['R_support'], N_support=s['N_support'])
                if name in p1.FIELDS:
                    row.update({f'w_{d}': r['p1a'][name][d]['w'][i] for d in p1.DOMAINS})
                rows.append(row)
    return rows


def write_csv(path, rows):
    import csv
    keys = list(rows[0])
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ---- the start: code, machine, root -----------------------------------------------------------------------------------------------------------------------
def code_files(root):
    return {rel: core.sha_file(Path(root)/rel) for rel in CODE_FILES}


def code_digest(files):
    return hashlib.sha256(''.join(f'{sha} {rel}\n' for rel, sha in sorted(files.items())).encode()).hexdigest()


def git(root, *args):
    done = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise Stop(f"git {' '.join(args)} in {root}: {done.stderr.strip()}")
    return done.stdout


def code_state(root):
    """HEAD, the tracked files' state and the code digest of a checkout."""
    head = git(root, 'rev-parse', 'HEAD').strip()
    dirty = git(root, 'status', '--porcelain', '--untracked-files=no').strip()
    files = code_files(root)
    return dict(root=str(root), head=head, tracked_clean=dirty == '', dirty=dirty.splitlines(), files=files, digest=code_digest(files))


def busy_units(patterns=BUSY_PATTERNS):
    """The active or activating units of these patterns; Stop when systemctl cannot say."""
    done = subprocess.run(['systemctl', '--user', 'list-units', f'--state={UNIT_STATES}', '--plain', '--no-legend', '--no-pager', *patterns], capture_output=True, text=True)
    if done.returncode != 0:
        raise Stop(f'systemctl --user list-units failed ({done.returncode}): {done.stderr.strip()}; the machine is not known to be free of {patterns}')
    return [line.split()[0] for line in done.stdout.splitlines() if line.strip()]


def uf_status(uf_units, state_dir=None, now=None):
    """(problems, record): P1 may run beside UF's chain only while one member trains, started at most UF_MAX_TRAIN_MINUTES ago (revision 1, the lead's ruling on
    review important 4).  No tag-uf* unit: nothing to say.  The chain state is <step>.started / <step>.done in state_dir (run_uf.run_step); the step that
    started and has not finished is where the chain is."""
    state_dir = UF_STATE_DIR if state_dir is None else state_dir
    record = dict(uf_units=list(uf_units), state_dir=str(state_dir), running=None, started_utc=None, minutes=None)
    if not uf_units:
        return [], record
    state_dir = Path(state_dir)
    if not state_dir.is_dir():
        return [f'UF units {list(uf_units)} are active but {state_dir} does not exist: where the chain is cannot be told'], record
    started = {p.name[:-len('.started')] for p in state_dir.glob('*.started')}
    done = {p.name[:-len('.done')] for p in state_dir.glob('*.done')}
    running = sorted(started-done)
    record['running'] = running
    if len(running) != 1 or running[0] not in UF_TRAIN_STEPS:
        return [f"UF's chain is at {running or 'a gate between steps'}, not a member in training: P1 would close its gate (start P1 once a member has started training, or with no tag-uf* unit)"], record
    stamp = (state_dir/f'{running[0]}.started').read_text().strip()
    try:
        since = datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    except ValueError:
        return [f'{running[0]}.started holds {stamp!r}, not a UTC time'], record
    minutes = ((datetime.now(timezone.utc) if now is None else now)-since).total_seconds()/60
    record.update(running=running[0], started_utc=stamp, minutes=minutes)
    if not 0 <= minutes <= UF_MAX_TRAIN_MINUTES:
        return [f'UF member {running[0]} started training {minutes:.0f} min ago, more than {UF_MAX_TRAIN_MINUTES:.0f}: P1 might outlast the step and close the next gate'], record
    return [], record


def in_unit(environ=None, cgroup='/proc/self/cgroup', unit=UNIT):
    """True inside the unit the dry run prints: systemd's INVOCATION_ID and the unit's own cgroup (revision 1, review minor 10)."""
    environ = os.environ if environ is None else environ
    text = Path(cgroup).read_text() if Path(cgroup).is_file() else ''
    return bool(environ.get('INVOCATION_ID')) and f'/{unit}.service' in text


# the modules the run uses and the files they must come from (revision 1, review minor 11: an untracked module of the same name would shadow one)
IMPORTED = {'p1_core': 'experiments/p1/p1_core.py', 'mc0_core': 'experiments/mc0/mc0_core.py', 'mc0p_core': 'experiments/mc0/mc0p_core.py', 'fast_channels': 'experiments/ef/fast_channels.py',
            'fl_core': 'experiments/fl/fl_core.py', 'fd2_core': 'experiments/fl/fd2_core.py', 'fx_core': 'experiments/fl/fx_core.py', '_mc0_make_inputs': 'experiments/d0/make_inputs.py',
            'bayesreg.io.pair': 'bayesreg/io/pair.py', 'bayesreg.seg.channels': 'bayesreg/seg/channels.py', 'bayesreg.seg.structure_layer': 'bayesreg/seg/structure_layer.py', 'bayesreg.cli.run_pair': 'bayesreg/cli/run_pair.py'}


def module_files():
    """Import now what the run imports later, and return {module: its file}; this script under 'run_p1'."""
    core.make_inputs()
    import bayesreg.cli.run_pair  # noqa: F401
    import bayesreg.io.pair  # noqa: F401
    import bayesreg.seg.channels  # noqa: F401
    import bayesreg.seg.structure_layer  # noqa: F401
    files = {name: str(Path(sys.modules[name].__file__).resolve()) for name in IMPORTED}
    files['run_p1'] = str(Path(__file__).resolve())
    return files


def module_problems(files, root):
    want = dict(IMPORTED, run_p1=SCRIPT)
    return [f'{name} is imported from {files.get(name)}, not {Path(root)/rel}' for name, rel in want.items() if files.get(name) != str((Path(root)/rel).resolve())]


def start_problems(o, root, state, busy, environ, *, uf=(), inside_unit=True, modules=(), frozen=()):
    """Why the unit may not run here, now; [] when nothing."""
    out = list(frozen)+list(uf)+list(modules)
    if not inside_unit:
        out.append(f'not inside the unit {UNIT}.service (INVOCATION_ID and the cgroup): start it with the string the dry run prints')
    if Path(o.root).resolve() != Path(DEFAULT_ROOT).resolve():
        out.append(f'{o.root}: P1 writes only to {DEFAULT_ROOT}')
    if Path(state['root']).resolve() != core.MAIN_REPO.resolve():
        out.append(f"the code is {state['root']}, not the main checkout {core.MAIN_REPO}")
    if state['head'] != o.commit:
        out.append(f"HEAD {state['head']} is not --commit {o.commit}")
    if not state['tracked_clean']:
        out.append(f"tracked files are not clean: {state['dirty']}")
    if state['digest'] != o.code_digest:
        out.append(f"the code digest {state['digest']} is not --code-digest {o.code_digest}")
    if busy:
        out.append(f'active units P1 may not run beside: {busy}')
    if environ.get('CUDA_VISIBLE_DEVICES') != '':
        out.append(f"CUDA_VISIBLE_DEVICES is {environ.get('CUDA_VISIBLE_DEVICES')!r}, not ''")
    for name in ENV_ONE_THREAD:
        if environ.get(name) != '1':
            out.append(f'{name} is {environ.get(name)!r}, not 1')
    if (Path(root)/'started.json').exists():
        out.append(f'{root}: started before; a run is never rerun')
    return out


def unit_command(commit, digest, root=DEFAULT_ROOT):
    """The lead's start: the root's directory first (the unit's log goes there), then one unit."""
    env = dict(**{k: '1' for k in ENV_ONE_THREAD}, NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', PYTHONPATH=f'{core.FROZEN}:{core.MAIN_REPO}')
    args = ['systemd-run', '--user', f'--unit={UNIT}', '--wait', '--collect', f'--working-directory={core.MAIN_REPO}', f'--property=MemoryMax={MEMORY_MAX}', f'--property=MemorySwapMax={SWAP_MAX}',
            f'--nice={NICE}', f'--property=CPUQuota={CPU_QUOTA}', f'--property=RuntimeMaxSec={RUNTIME_MAX_SEC}', f'--property=StandardOutput=append:{root}/log.txt', f'--property=StandardError=append:{root}/log.txt']
    args += [f'--setenv={k}={v}' for k, v in env.items()]
    args += ['/home/rmcfz/miniconda3/envs/sci310/bin/python', '-B', str(core.MAIN_REPO/SCRIPT), 'run', '--commit', commit, '--code-digest', digest]
    return [f'mkdir -p {shlex.quote(str(root))}', ' '.join(shlex.quote(a) for a in args)]


# ---- the entries ---------------------------------------------------------------------------------------------------------------------------------
def dry_run(world=None, code_root=None, out=print):
    """Lists and hashes; decodes no array of any input."""
    world = World() if world is None else world
    code_root = core.ROOT if code_root is None else Path(code_root)
    problems, notes = [], []
    from bayesreg.io.pair import _read_manifest
    unet = world.unet_source()
    frozen = core.FROZEN/'jd_curl/three_d/preopt.py'
    if not frozen.is_file() or core.sha_file(frozen) != core.FROZEN_PREOPT_SHA256:
        problems.append(f'{core.FROZEN}: preopt.py is not the pinned frozen file (the taper)')
    out(f'P1 dry run: {len(PAIRS)} pairs, output {DEFAULT_ROOT}; records rw_v1 {world.results/RW_ROOT}, fd2_v1 {world.results/FD2_ROOT}; pins {world.sources_file}')
    for pair in PAIRS:
        short = core.short_name(pair)
        try:
            run, repair, pin = records_of(pair, world)
        except Stop as stop:
            problems.append(stop.message)
            continue
        problems += file_problems(pair, run, repair, world)
        _, _, manifest = _read_manifest(world.stage/pair/'manifest.json')
        pair_sha = core.sha_file(world.stage/pair/'pair.npz')
        if not (pair_sha == manifest['pair_sha256'] == run['data']['pair_sha256']):
            problems.append(f'{short}: pair.npz sha256 {pair_sha}, manifest {manifest["pair_sha256"]}, fd2 run.json {run["data"]["pair_sha256"]}')
        probs = {}
        for side in ('fixed', 'moving'):
            subject = manifest[f'{side}_subject']
            path = Path(unet.pred_root)/subject/'probs36.npz'
            probs[side] = core.sha_file(path) if path.is_file() else None
            want = run['channel_sources']['sides'][side]['unet_output_sha256']['probs36.npz']
            if probs[side] != want:
                problems.append(f'{short}: {path} sha256 {probs[side]} is not fd2 run.json {want}')
        subject = manifest['fixed_subject']
        provenance = read_record(world.seg_root/subject/'provenance.json')
        atropos = {}
        for name in provenance.get('posterior_files', []):
            atropos[name] = core.sha_file(world.seg_root/subject/name) if (world.seg_root/subject/name).is_file() else None
            if atropos[name] != provenance['output_sha256'].get(name) or atropos[name] != run['channel_sources']['sides']['fixed']['atropos_qc']['output_sha256'].get(name):
                problems.append(f'{short}: Atropos {subject}/{name} sha256 {atropos[name]} differs from its provenance or from fd2 run.json')
        out(f"{short}: field_0600 {run['checkpoints'][STEP]['sha256'][:16]} repair.npz {repair['attempt']['sha256'][:16]} repaired state {repair['certificate']['state_hash'][:16]} pair.npz {pair_sha[:16]} "
            f"probs36 {(probs['fixed'] or '-')[:12]}/{(probs['moving'] or '-')[:12]} Atropos {subject} {len(atropos)} posteriors")
    state = None
    try:
        state = code_state(code_root)
    except Stop as stop:
        problems.append(stop.message)
    if state is not None:
        if Path(state['root']).resolve() != core.MAIN_REPO.resolve():
            notes.append(f"the code here is {state['root']}, not the main checkout: the unit runs from {core.MAIN_REPO} after the merge; run this dry run there for its commit and digest")
        if not state['tracked_clean']:
            problems.append(f"tracked files are not clean: {state['dirty']}")
        out(f"code: {state['root']} HEAD {state['head']} digest {state['digest']}")
    problems += module_problems(module_files(), code_root)
    try:                                                                # the machine now: the unit checks it again when it starts (notes, not problems)
        busy, uf_units = busy_units(), busy_units(UF_PATTERNS)
        uf, record = uf_status(uf_units)
        out(f"machine now ({UNIT_STATES}): {' '.join(BUSY_PATTERNS)}: {busy or 'none'}; {' '.join(UF_PATTERNS)}: {uf_units or 'none'}" + (f"; UF chain at {record['running']}" if uf_units else ''))
        notes += [f'P1 would not start now: active units {busy}'] if busy else []
        notes += [f'P1 would not start now: {p}' for p in uf]
    except Stop as stop:
        notes.append(f'the machine now is not known: {stop.message}')
    if (DEFAULT_ROOT/'started.json').exists():
        problems.append(f'{DEFAULT_ROOT}: started before; a run is never rerun')
    for line in unit_command('<HEAD>' if state is None else state['head'], '<digest>' if state is None else state['digest']):
        out(f'start: {line}')
    out(f'estimate〔估算〕: about {SECONDS_PER_PAIR:.0f} s a pair, {len(PAIRS)*SECONDS_PER_PAIR/60:.0f} min for {len(PAIRS)} pairs, CPU one thread, host < 2 GiB, no GPU; outputs small JSON / CSV / md, no field')
    for note in notes:
        out(f'note: {note}')
    out('problems: ' + ('none' if not problems else ''))
    for problem in problems:
        out(f'  {problem}')
    return problems


def run_all(o, world=None, root=None, log=print, state=None, busy=None, environ=None, uf=None, inside_unit=None, modules=None):
    """The unit's body.  root: the output directory (a test passes its own; the entry refuses any other).  Every check before anything is written."""
    import torch
    world = World() if world is None else world
    root = Path(DEFAULT_ROOT if root is None else root)
    frozen = []
    if world.taper is None:                                            # the frozen tree first: jd_curl must come from it before bayesreg is imported below
        try:
            core.frozen_api()
        except SystemExit as error:
            frozen = [f'the frozen tree: {error}']
    state = code_state(core.ROOT) if state is None else state
    busy = busy_units() if busy is None else busy
    uf_record = None
    if uf is None:
        uf, uf_record = uf_status(busy_units(UF_PATTERNS))
    inside_unit = in_unit(environ) if inside_unit is None else inside_unit
    files = module_files()
    modules = module_problems(files, core.ROOT) if modules is None else modules
    problems = start_problems(o, root, state, busy, os.environ if environ is None else environ, uf=uf, inside_unit=inside_unit, modules=modules, frozen=frozen)
    if problems:
        raise Stop('P1 does not start: ' + '; '.join(problems))
    root.mkdir(parents=True, exist_ok=True)
    (root/'pairs').mkdir(exist_ok=True)
    write_json(root/'started.json', dict(schema=SCHEMA, utc=utc(), argv=sys.argv, pid=os.getpid(), commit=o.commit, code=state, modules=files, machine=dict(busy=busy, uf=uf_record, inside_unit=inside_unit), role=ROLE))
    torch.set_num_threads(1)
    end = dict(schema=SCHEMA, status='stopped', reason=None)
    try:
        world.taper_function()                                         # the frozen tree, before anything is read
        second = p1.control_ii(core.pre_energy_k)
        write_json(root/'control_ii.json', second)
        log(f"control (ii): {'passed' if second['passed'] else 'NOT passed'}, max relative {second['max_relative']:.2e}")
        if not second['passed']:
            raise Stop('control (ii) not passed: nothing is read')
        records = []
        for pair in PAIRS:
            record = compute_pair(pair, world, log)
            write_json(root/'pairs'/f"{record['short']}.json", record)
            if not record['control_i']['passed']:
                raise Stop(f"{record['short']}: control (i) not passed: nothing more is read")
            log(f"{record['short']}: done in {record['seconds']:.1f} s, host peak {record['host_peak_rss_gib']:.2f} GiB")
            records.append(record)
        summary = summarize(records, second)
        write_json(root/'summary.json', summary)
        (root/'summary.md').write_text(summary_markdown(summary), encoding='utf-8')
        write_csv(root/'per_channel.csv', per_channel_rows(records))
        end.update(status='done', reading=summary['reading'])
    except SystemExit as stop:                                          # a Stop of this module or a refusal of a reader it calls
        end.update(reason=getattr(stop, 'message', None) or str(stop.code))
        log(f"STOP: {end['reason']}")
    except Exception as error:                                          # noqa: BLE001 - kept as evidence, never retried
        end.update(status='error', reason=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        log(end['traceback'])
    end.update(ended_utc=utc(), host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    write_json(root/'end.json', end)
    return 0 if end['status'] == 'done' else 8


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true')
    sub = parser.add_subparsers(dest='command')
    r = sub.add_parser('run')
    r.add_argument('--commit', required=True)
    r.add_argument('--code-digest', required=True)
    r.add_argument('--root', default=str(DEFAULT_ROOT))
    o = parser.parse_args(argv)
    if o.dry_run and o.command is None:
        return 0 if not dry_run() else 8
    if o.command == 'run':
        try:
            return run_all(o)
        except SystemExit as stop:
            print(f"STOP: {getattr(stop, 'message', None) or stop.code}", file=sys.stderr)
            return 8
    parser.print_usage()
    return 2


if __name__ == '__main__':
    sys.exit(main())
