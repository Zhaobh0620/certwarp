"""PQ (contract C22.Q1, C22.Q5; plan.md sections 1, 2, 3, 8): the files of the PQ unit -- its inputs bound to the pins (m1-m4), the outputs written once, the sha256 list.

m1 (C22.Q1-2)  the certified final state v of each pair, bound step by step: field_0600.npz's sha256 = the plan's = T88 run.json field.sha256 = repair.json source.field_sha256; the taper of the frozen tree
               (boundary_taper, L = 4) gives source.tapered_u_sha256; repair.npz's sha256 = attempt.sha256; v = the tapered field with the attempt's node values (run_ef.repaired_field); sha256 of v's C-order
               bytes = the top-level certificate.state_hash, a certificate certified_feasible with min_jd_lower_bound > 0 and n_infeasible = n_uncertain = 0; repair.json and run.json are the plan's files.
               Only line_status hx_certified exists in P3 (19/19, no fallback; plan.md 1.1): any other repair status is refused (a fallback needs the HE revision, C22.Q6).
m2 (C22.Q1-1)  the structure layer of each side through bayesreg.seg.structure_layer.StructureLayer (infer 'never': a missing prediction is refused, never made): provenance (validate_segmentation_provenance:
               layer structure, trained_with_labels true, test_labels_seen false, the registered models and members, D24 pair level), the grouping, the grid, the outputs re-hashed; then the anchor fields
               (segmenter_id, policy, registry_sha256) and the plan's sha256 of probs36.npz, structure_groups.npz and provenance.json; the registry the formal, committed and clean one with the anchor's sha256.
m3 (C22.Q1-1)  P = bayesreg.seg.channels.read_back_probabilities (float64, the raw channel sum checked before the division); L = argmax P (ties to the smallest), int16, bit for bit the L of
               experiments/ef/fast_channels.raw_labels (the method's reader) and its sha256 the T88 run.json labels_sha256 of that side; the group sums within make_inputs.GROUP_TOL.
m4 (C22.Q2)    argmax P~_M (outside: background) bit for bit warp_labels_l2r(L_M, d); load_pair's grid.spacing equal to the manifest's spacing_xyz.
m1-m4 are done for all 19 pairs before anything is computed (plan.md section 2: "在算任何东西之前做，有一项不过就停，不写输出"); then each pair is read again and computed, m4 checked again.
Outputs (pq_v1/runs/<pair>/, written once): pq.npz (q, q4; float32 after float64), baselines.npz (h uint8; H_F, H2, V_F, V2, G_plus, G_minus, delta float32), pq.json (bayesreg.pq.v1).
"""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
for _rel in ('ef', 'mc0'):
    if str(HERE.parent/_rel) not in sys.path:
        sys.path.append(str(HERE.parent/_rel))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pq_core as core  # noqa: E402
import pq_pins as pins  # noqa: E402
from pq_core import PQRefused  # noqa: E402

PQ_SCHEMA = 'bayesreg.pq.v1'
BASELINES_FILE, PQ_NPZ, PQ_JSON = 'baselines.npz', 'pq.npz', 'pq.json'
MEMBERS_RECORD, VAR_FILE = 'members_record.json', 'var.npz'
HX_CERTIFIED = 'hx_certified'


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'not JSON serializable: {type(value)}')


def no_nan(node):
    if isinstance(node, dict):
        return {str(k): no_nan(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [no_nan(v) for v in node]
    if isinstance(node, (float, np.floating)) and not np.isfinite(node):
        return None
    return node


def write_once(path, data=None, *, text=None):
    """A file that is there is never overwritten (FileExistsError).  Revision 2 (diff review minor 1): the content is serialized first, written and fsync'ed to a temporary file beside it
    (.<name>.tmp-<pid>), then put in place with os.link, which is atomic and fails if the name exists; the temporary name is removed.  A kill during the write leaves no partial file under the name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'{path} exists: never overwritten')
    payload = text if text is not None else json.dumps(no_nan(json.loads(json.dumps(data, default=json_default))), indent=1, ensure_ascii=False)+'\n'
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with open(tmp, 'x', encoding='utf-8') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, path)
    finally:
        os.unlink(tmp)
    return path


def read_json_or_none(path):
    """(document, problem): (None, None) when the file is not there; (None, 'unreadable: ...') when it cannot be read as UTF-8 JSON (a truncated or empty file); the reading takes it as missing."""
    path = Path(path)
    if not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text(encoding='utf-8')), None
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return None, f'{path}: unreadable ({type(exc).__name__}: {str(exc)[:200]})'


def save_npz_once(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'xb') as stream:
        np.savez(stream, **arrays)
    return path


@dataclass
class Context:
    """Where everything is (the tests point every root into tmp_path).  repo: the code tree (configs/, splits/, runs/ and git); source_root: the tree whose data/ holds the allowlist's stage lists and the
    stage (the main checkout: a worktree has no data/); results: the results root; images: /mnt/d/data; frozen: the taper function (u, grid, L) -> (u_tapered, w)."""
    repo: Path
    source_root: Path
    results: Path
    images: Path = pins.DATA_IMAGES
    shorts: tuple = pins.SHORTS
    p3: object = pins.P3
    run_json: object = pins.RUN_JSON
    v4: object = pins.V4
    anchor: object = pins.ANCHOR
    seg4_sha256: str = pins.SEG4_TABLE_SHA256
    taper: object = None
    extra: dict = field(default_factory=dict)

    @property
    def stage(self):
        return Path(self.source_root)/pins.STAGE_REL

    @property
    def pq_root(self):
        return Path(self.results)/pins.PQ_ROOT_REL

    @property
    def pred_root(self):
        return Path(self.results)/pins.PRED_ROOT_REL

    def subjects(self):
        return sorted({s for short in self.shorts for s in pins.subjects_of(short)})

    def t88_dir(self, short):
        return Path(self.results)/pins.P3_ROOT_REL/self.p3[short].t88_dir/short

    def repair_dir(self, short):
        return Path(self.results)/pins.P3_ROOT_REL/self.p3[short].repair_dir/short


# ---- the plan and the tables -----------------------------------------------------------------------------------------------------------------------
def check_plan(repo, pinned=pins.PLAN_SHA256):
    """runs/20261107-pq/plan.md has the pinned sha256 (PQRefused 'plan' otherwise); nothing else is read before it."""
    path = Path(repo)/pins.PLAN_REL
    if not path.is_file():
        raise PQRefused('plan', f'{path}: the pinned preregistration is missing')
    digest = sha_file(path)
    if digest != pinned:
        raise PQRefused('plan', f'{path}: sha256 {digest} is not the pinned {pinned}; the preregistration is not changed after it was pinned')
    return dict(path=pins.PLAN_REL, sha256=digest)


def load_lookup(repo, pinned=pins.SEG4_TABLE_SHA256):
    path = Path(repo)/pins.SEG4_TABLE_REL
    if sha_file(path) != pinned:
        raise PQRefused('seg4', f'{path}: not the pinned sha256 {pinned} (C22.Q2-6)')
    return core.seg4_lookup_array({int(k): int(v) for k, v in read_json(path)['seg4_of_seg35'].items()})


# ---- m1: the certified final state -----------------------------------------------------------------------------------------------------------------
def certificate_problems(cert):
    cert = cert or {}
    bound = cert.get('min_jd_lower_bound')
    problems = []
    if cert.get('status') != 'certified_feasible':
        problems.append(f"status {cert.get('status')!r}")
    if not (isinstance(bound, (int, float)) and bound > 0):
        problems.append(f'min_jd_lower_bound {bound!r}')
    if cert.get('n_infeasible') != 0 or cert.get('n_uncertain') != 0:
        problems.append(f"n_infeasible {cert.get('n_infeasible')!r}, n_uncertain {cert.get('n_uncertain')!r}")
    return problems


def bind_field(ctx, short, grid):
    """m1 for one pair: (v float64 C-contiguous (nz, ny, nx, 3) in mm, xyz components; the record).  PQRefused('m1') on any departure."""
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
    need(run.get('segmenter_id') == ctx.anchor['segmenter_id'], f"run.json segmenter {run.get('segmenter_id')!r}")
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


# ---- m2, m3: the structure layer -------------------------------------------------------------------------------------------------------------------
def structure_layer(ctx, pin):
    """StructureLayer of the pair (bayesreg.seg.structure_layer, infer 'never') on the formal registry of the anchor; the registry committed and clean with the anchor's sha256.  PQRefused('m2')."""
    from jd_curl.validation import InputValidationError
    from bayesreg.io.pair import ProvenanceRejected
    from bayesreg.seg.structure_layer import REGISTERED, StructureLayer, StructureSource
    registry = Path(ctx.repo)/ctx.anchor['registry']
    src = StructureSource(root=ctx.pred_root, registry_path=registry, allowlist_path=Path(ctx.repo)/pins.ALLOWLIST_REL, models_root=Path(ctx.results)/pins.MODELS_ROOT_REL, image_root=Path(ctx.images),
                          segmenter_id=ctx.anchor['segmenter_id'], infer='never', formal_registry=registry)
    try:
        layer = StructureLayer(pin, src)
    except (ProvenanceRejected, InputValidationError) as exc:
        raise PQRefused('m2', f'{pin.fixed_subject} <- {pin.moving_subject}: the structure layer refused ({type(exc).__name__}: {exc})') from None
    record = layer.registry_record
    git = record.get('git') or {}
    if record['status'] != REGISTERED or not (git.get('tracked') is True and git.get('clean') is True):
        raise PQRefused('m2', f"the registry {record['path']}: status {record['status']}, git tracked {git.get('tracked')}, clean {git.get('clean')}")
    if layer.registry.sha256 != ctx.anchor['registry_sha256']:
        raise PQRefused('m2', f"the registry sha256 {layer.registry.sha256} is not the anchor's {ctx.anchor['registry_sha256']}")
    return layer


def read_side(ctx, layer, pin, side, labels_sha256):
    """m2 (one side) and m3: (P float64 (36, ...), L int16, record).  The structure groups are read by layer.load (verified, the read-back), passed to raw_labels and dropped before P is read."""
    from jd_curl.validation import InputValidationError
    from bayesreg.io.pair import ProvenanceRejected
    from bayesreg.seg.channels import read_back_probabilities
    import fast_channels
    import mc0_core
    subject = layer.subjects[side]
    try:
        groups, accepted, rec = layer.load(side)
    except (ProvenanceRejected, InputValidationError) as exc:
        raise PQRefused('m2', f'{side} {subject}: {type(exc).__name__}: {exc}') from None
    directory = Path(rec['directory'])
    prov_path = directory/'provenance.json'
    prov = read_json(prov_path)
    want = ctx.v4.get(subject)
    problems = []
    if want is None:
        problems.append(f'{subject} is not one of the plan\'s 20 subjects')
    else:
        for name, expected, have in (('provenance.json', want.provenance, rec['provenance_sha256']), ('probs36.npz', want.probs36, rec['verified_outputs'].get('probs36.npz')),
                                     ('structure_groups.npz', want.structure_groups, rec['verified_outputs'].get('structure_groups.npz'))):
            if have != expected:
                problems.append(f"{name} sha256 {have} is not the plan's {expected}")
    for key in ('segmenter_id', 'policy', 'registry_sha256'):
        if prov.get(key) != ctx.anchor[key]:
            problems.append(f'provenance {key} {prov.get(key)!r} is not the anchor\'s {ctx.anchor[key]!r}')
    if prov.get('trained_with_labels') is not True or prov.get('test_labels_seen') is not False:
        problems.append(f"trained_with_labels {prov.get('trained_with_labels')!r}, test_labels_seen {prov.get('test_labels_seen')!r}")
    if problems:
        raise PQRefused('m2', f'{side} {subject}: {problems}')
    g = layer.grouping
    shape = tuple(pin.grid.shape)
    try:
        fast = fast_channels.raw_labels(directory/'probs36.npz', (core.K,)+shape, groups, (g.background_labels,)+tuple(g.group_labels))
    except SystemExit as exc:
        raise PQRefused('m3', f'{side} {subject}: the method\'s reader refused: {exc}') from None
    del groups
    if fast is None:
        raise PQRefused('m3', f'{side} {subject}: raw_labels gives None (another storage, a sign bit, NaN or Inf): the method\'s fast path did not decide; not covered by this tool')
    L_raw, _, fast_record, deviation = fast
    group_tol = mc0_core.make_inputs().GROUP_TOL
    if not deviation <= group_tol:
        raise PQRefused('m3', f'{side} {subject}: probs36 summed by group differs from structure_groups.npz by {deviation:.3e} > {group_tol:g}')
    try:
        P, read_back = read_back_probabilities(directory/'probs36.npz', shape=(core.K,)+shape)
    except InputValidationError as exc:
        raise PQRefused('m3', f'{side} {subject}: {exc}') from None
    P = np.asarray(P)
    L = core.argmax_smallest(P)
    if L.dtype != L_raw.dtype or L.tobytes() != np.ascontiguousarray(L_raw).tobytes():
        raise PQRefused('m3', f'{side} {subject}: argmax of the read-back posterior is not raw_labels\' L bit for bit ({int((L != L_raw).sum())} voxels differ)')
    digest = sha_array(L)
    if digest != (labels_sha256 or {}).get(side):
        raise PQRefused('m3', f"{side} {subject}: sha256 of L {digest} is not the T88 run.json labels_sha256 {(labels_sha256 or {}).get(side)}")
    return P, L, dict(subject=subject, directory=str(directory), provenance_sha256=rec['provenance_sha256'], probs36_sha256=rec['verified_outputs'].get('probs36.npz'),
                      structure_groups_sha256=rec['verified_outputs'].get('structure_groups.npz'), read_back=read_back, raw_labels=fast_record, group_sum_max_abs_deviation=deviation, group_tol=group_tol,
                      labels_sha256=digest, models=rec.get('models'), segmenter_id=prov.get('segmenter_id'), policy=prov.get('policy'), registry_sha256=prov.get('registry_sha256'),
                      accepted=dict(check=accepted.get('check'), model_ids=accepted.get('model_ids'), label_contact_subjects_sha256=accepted.get('label_contact_subjects_sha256')))


# ---- load_pair, m4 ----------------------------------------------------------------------------------------------------------------------------------
def load_pair_checked(ctx, short, allowlist):
    """load_pair (the only reader of pair.npz here: fixed, moving, initial_u) and the spacing check of m4: grid.spacing = the manifest's spacing_xyz."""
    from bayesreg.io.pair import _keep_whitelisted_keys, load_pair
    d = ctx.stage/pins.pair_name(short)
    pin = load_pair(d/'pair.npz', allowlist=allowlist)
    manifest = json.loads((d/'manifest.json').read_text(encoding='utf-8'), object_pairs_hook=_keep_whitelisted_keys)     # load_pair's whitelist of manifest keys: no label metric is kept
    spacing = [float(v) for v in pin.grid.spacing]
    if spacing != [float(v) for v in manifest['spacing_xyz']] or list(pin.grid.shape) != list(manifest['shape_zyx']):
        raise PQRefused('m4', f"{short}: grid spacing {spacing} / shape {list(pin.grid.shape)} differ from the manifest's {manifest['spacing_xyz']} / {manifest['shape_zyx']}")
    if (pin.fixed_subject, pin.moving_subject) != pins.subjects_of(short):
        raise PQRefused('m4', f'{short}: the pair is {pin.fixed_subject} <- {pin.moving_subject}')
    return pin


def displacement(v, pin):
    return core.disp_voxels(v, tuple(pin.grid.spacing))


def check_m4(maps_top, L_M, d):
    """argmax P~_M (outside: background) bit for bit warp_labels_l2r(L_M, d)."""
    warped = core.warp_labels_l2r(np.asarray(L_M), d)
    if warped.dtype != maps_top.dtype or warped.tobytes() != maps_top.tobytes():
        raise PQRefused('m4', f'argmax of P~_M differs from warp_labels_l2r(L_M, d) on {int((warped != maps_top).sum())} voxels')
    return warped


def nearest_top(P_M, iota):
    """argmax P~_M alone (m4 of the first pass, without the rest of pq_maps)."""
    best, top = None, np.zeros(iota.shape, dtype=np.int16)
    for c in range(core.K):
        pm = core.take(P_M[c], iota, 1. if c == 0 else 0.)
        if best is None:
            best = pm
        else:
            better = pm > best
            top[better] = c
            np.maximum(best, pm, out=best)
    return top


# ---- the members' variance maps (from the members unit) ---------------------------------------------------------------------------------------------
def members_record(ctx):
    """The members unit's record (pq_v1/members/members_record.json): m5 passed for the 20 subjects, each var.npz with its recorded sha256.  PQRefused('m5') otherwise."""
    path = ctx.pq_root/'members'/MEMBERS_RECORD
    if not path.is_file():
        raise PQRefused('m5', f'{path}: the members unit has not run (D91 (a): V_F, V2 are judged baselines)')
    record = read_json(path)
    subjects = ctx.subjects()
    rows = record.get('subjects') or {}
    if record.get('m5_passed') is not True or sorted(rows) != subjects or not all((rows[s] or {}).get('m5') is True for s in subjects):
        raise PQRefused('m5', f"{path}: m5 not passed on every subject ({record.get('m5_passed')}, {sorted(rows)})")
    return record


def read_variance(ctx, record, subject, shape):
    path = ctx.pq_root/'members'/subject/VAR_FILE
    want = ((record.get('subjects') or {}).get(subject) or {}).get('var_sha256')
    if not path.is_file() or sha_file(path) != want:
        raise PQRefused('m5', f'{path}: missing or not the recorded sha256 {want}')
    with np.load(path, allow_pickle=False) as z:
        V = np.array(z['V'])
    if V.dtype != np.float32 or V.shape != tuple(shape):
        raise PQRefused('m5', f'{path}: V {V.dtype} {V.shape}, expected float32 {tuple(shape)}')
    return V


# ---- one pair -------------------------------------------------------------------------------------------------------------------------------------
def check_pair(ctx, short, allowlist, lookup):
    """Pass 1 (m1-m4 of one pair, nothing computed, nothing written).  Returns the pair's input record."""
    pin = load_pair_checked(ctx, short, allowlist)
    v, field_record = bind_field(ctx, short, pin.grid)
    layer = structure_layer(ctx, pin)
    P_F, L_F, fixed = read_side(ctx, layer, pin, 'fixed', field_record['labels_sha256'])
    del P_F
    P_M, L_M, moving = read_side(ctx, layer, pin, 'moving', field_record['labels_sha256'])
    d = displacement(v, pin)
    iota = core.sample_index(d)
    check_m4(nearest_top(P_M, iota), L_M, d)
    return dict(pair=pins.pair_name(short), field=field_record, fixed=fixed, moving=moving, d_sha256=sha_array(d), iota_outside=int((iota < 0).sum()),
                structure_layer=dict(preflight=layer.preflight, registry=dict(path=layer.registry.path, sha256=layer.registry.sha256)),
                data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], keys_read=list(pin.provenance['keys_read'])),
                spacing_xyz=[float(s) for s in pin.grid.spacing], m=dict(m1=True, m2=True, m3=True, m4=True))


def compute_pair(ctx, short, allowlist, lookup, variance_record, out_dir):
    """Pass 2: the inputs again (every check again), PQ and the baselines, the three files.  Returns the pair's pq.json document."""
    tick = perf_counter()
    pin = load_pair_checked(ctx, short, allowlist)
    v, field_record = bind_field(ctx, short, pin.grid)
    layer = structure_layer(ctx, pin)
    P_F, L_F, fixed = read_side(ctx, layer, pin, 'fixed', field_record['labels_sha256'])
    H_F = core.entropy(P_F)
    tail_F = core.tail_masses(P_F, L_F)
    P_M, L_M, moving = read_side(ctx, layer, pin, 'moving', field_record['labels_sha256'])
    H_M = core.entropy(P_M)
    d = displacement(v, pin)
    iota = core.sample_index(d)
    maps = core.pq_maps(P_F, P_M, iota, lookup)
    L_tilde = check_m4(maps['top_tilde'], L_M, d)
    tail_M = core.tail_masses(P_M, L_tilde, iota)
    del P_F, P_M
    numbers = core.pq_numbers(maps, L_F, L_tilde, L_M, lookup)
    q = maps['q']
    omega = core.omega_pred(L_F, L_tilde)
    eps = core.eps_em(1.-q[omega])
    shape = tuple(pin.grid.shape)
    V_F = read_variance(ctx, variance_record, pin.fixed_subject, shape)
    V_M = read_variance(ctx, variance_record, pin.moving_subject, shape)
    base = core.baseline_maps(L_F=L_F, L_tilde=L_tilde, H_F=H_F, H_M=H_M, V_F=V_F, V_M=V_M, F=pin.fixed, iota=iota, spacing_xyz=tuple(pin.grid.spacing))
    q32, q4_32 = q.astype(np.float32), maps['q4'].astype(np.float32)
    out_dir = Path(out_dir)
    save_npz_once(out_dir/PQ_NPZ, q=q32, q4=q4_32)
    save_npz_once(out_dir/BASELINES_FILE, **{k: base[k] for k in core.BASELINE_KEYS})
    document = dict(schema=PQ_SCHEMA, pair=pins.pair_name(short), short=short, contract=pins.CONTRACT, plan_sha256=pins.PLAN_SHA256, role='PQ: an additional output of the method (C22.Q0); its Dice are the '
                    'method\'s own numbers and enter no Dice table; not a registration posterior, not a MAP, not a segmentation',
                    inputs=dict(field=field_record, fixed=fixed, moving=moving, seg4_table_sha256=ctx.seg4_sha256, registry_sha256=layer.registry.sha256, structure_preflight=layer.preflight,
                                data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], keys_read=list(pin.provenance['keys_read'])),
                                variance={s: (variance_record['subjects'][s] or {}).get('var_sha256') for s in (pin.fixed_subject, pin.moving_subject)}),
                    sampling=dict(rule='iota: warp_labels_l2r of the voxel numbers + 1 (scripts/evaluate_l2r_style.py), order 0, outside -> background (C22.Q2-1, 2)', d_sha256=sha_array(d),
                                  n_outside=int((iota < 0).sum()), spacing_xyz=[float(s) for s in pin.grid.spacing], m4='argmax P~_M = warp_labels_l2r(L_M, d) bit for bit'),
                    numbers=numbers, tail_mass=dict(fixed=tail_F.tolist(), moving_pulled_back=tail_M.tolist(), rule='sum_x P^c 1[L != c], c = 0..35; the moving side pulled back with L~_M'),
                    q_range=[float(q.min()), float(q.max())], eps_em=dict(eps, d89='(a) off: described only, pi not stored', omega_pred_voxels=int(omega.sum()), floor=core.EPS_FLOOR, ceiling=core.EPS_CEILING),
                    files={PQ_NPZ: dict(keys=['q', 'q4'], dtype='float32', sha256=sha_file(out_dir/PQ_NPZ)), BASELINES_FILE: dict(keys=list(core.BASELINE_KEYS), dtypes={k: str(base[k].dtype) for k in core.BASELINE_KEYS},
                                                                                                                   sha256=sha_file(out_dir/BASELINES_FILE))},
                    seconds=perf_counter()-tick)
    write_once(out_dir/PQ_JSON, document)
    return document


# ---- the sha256 list --------------------------------------------------------------------------------------------------------------------------------
MANIFEST_FILE = 'outputs_manifest.txt'


def manifest_lines(root, relative_paths):
    """sha256sum format, sorted by path, relative to pq_v1/: one line per file."""
    root = Path(root)
    return ''.join(f'{sha_file(root/rel)}  {rel}\n' for rel in sorted(relative_paths))


def parse_manifest(text):
    rows = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, rel = line.split('  ', 1)
        if len(digest) != 64 or rel in rows:
            raise ValueError(f'a malformed or repeated manifest line: {line!r}')
        rows[rel] = digest
    return rows
