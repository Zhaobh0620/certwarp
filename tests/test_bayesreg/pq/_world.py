"""A synthetic world for the PQ tools (tmp_path only): a code tree with its configs, the pinned plan.md (a copy, so that its sha256 is the pinned one), a stage of 2 pairs (3 development subjects, chained),
the registered structure-layer predictions of unet35_v4_ens (probs36, structure groups, provenance on a synthetic registry committed in the tree), the P3 records (run.json, field_0600.npz, repair.json,
repair.npz, and the evaluation records metrics_0600.json, repair_metrics.json that the method side must never open), and the members unit's outputs (var.npz, members_record.json).
No file of the real results directory, no real image, label, prediction or field is read; the subject ids are canonical spellings used as synthetic labels.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PQ = ROOT/'experiments'/'pq'
if str(PQ) not in sys.path:
    sys.path.insert(0, str(PQ))

import pq_core as core  # noqa: E402
import pq_io as io_  # noqa: E402
import pq_pins as pins  # noqa: E402

from bayesreg.io.pair import subject_list_sha256  # noqa: E402
from bayesreg.seg.structure_layer import GROUPS_FILE, PROVENANCE_FILE  # noqa: E402
from bayesreg.seg.unet.groups import group_probabilities, load_grouping  # noqa: E402

SHAPE = (10, 12, 9)                                  # (nz, ny, nx)
SUBJECTS = ('OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0439_MR1', 'OASIS_OAS1_0440_MR1')
SHORTS = ('0438-0439', '0439-0440')
FOLD0 = tuple(f'OASIS_OAS1_{n:04d}_MR1' for n in range(101, 105))
FOLD1 = tuple(f'OASIS_OAS1_{n:04d}_MR1' for n in range(105, 109))
# C23.9-13 (8), revision 6e: 0104, 0105, 0106 above are real holdout_B_v1 subjects, which SG refuses; the defaults stay, and the tests that expect the
# training draw admitted build the world with these synthetic stand-ins in the same places (build(..., synthetic=True), the world fixture's indirect parameter)
SYN_FOLD0 = tuple(f'OASIS_OAS1_{n:04d}_MR1' for n in (101, 102, 103, 994))
SYN_FOLD1 = tuple(f'OASIS_OAS1_{n:04d}_MR1' for n in (995, 996, 107, 108))
SEG_ID = 'unet35_v4_ens'
GROUPING = load_grouping()
LOOKUP = np.array([int(v) for _, v in sorted(((int(k), v) for k, v in json.loads((ROOT/pins.SEG4_TABLE_REL).read_text())['seg4_of_seg35'].items()))])


def hexof(tag):
    return hashlib.sha256(tag.encode()).hexdigest()


def number(subject):
    return int(subject.split('_')[2])


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1), encoding='utf-8')
    return path


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), '-c', 'user.name=test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null', *args],
                          check=True, capture_output=True, text=True)


def commit_all(root, message='synthetic'):
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '--allow-empty', '-m', message)


def frozen_formula_taper(u, grid, L=4):
    """boundary_taper of the frozen preopt.py (F:jd_curl/three_d/preopt.py:617), the same operations (test_pq_io compares it with the frozen one when the frozen tree is on the machine)."""
    rs = []
    for n in grid.shape:
        k = np.arange(n)
        rs.append(np.minimum(np.minimum(k, n-1-k)/float(L), 1.))
    w = rs[0][:, None, None]*rs[1][None, :, None]*rs[2][None, None, :]
    return w[..., None]*np.asarray(u, dtype=np.float64), w


def truth35(subject):
    """Voronoi cells of 14 random labels inside an ellipsoid, background outside: boundaries, small and large structures."""
    rng = np.random.default_rng(1000+number(subject))
    z, y, x = np.meshgrid(*[np.arange(n) for n in SHAPE], indexing='ij')
    centres = rng.uniform(0, 1, (14, 3))*np.array(SHAPE)
    labels = rng.choice(np.arange(1, 36), 14, replace=False)
    dist = sum((c-centres[:, k][:, None, None, None])**2 for k, c in enumerate((z, y, x)))
    out = labels[np.argmin(dist, axis=0)]
    inside = ((z-(SHAPE[0]-1)/2)/(SHAPE[0]/2.2))**2+((y-(SHAPE[1]-1)/2)/(SHAPE[1]/2.2))**2+((x-(SHAPE[2]-1)/2)/(SHAPE[2]/2.2))**2 < 1
    return np.where(inside, out, 0).astype(np.int64)


def predicted_base(subject):
    """The truth with 8 % of the voxels relabelled: the segmenter's mistakes."""
    rng = np.random.default_rng(2000+number(subject))
    L = truth35(subject).copy()
    flip = rng.uniform(size=SHAPE) < .08
    L[flip] = rng.integers(0, 36, int(flip.sum()))
    return L


def posterior(subject):
    rng = np.random.default_rng(3000+number(subject))
    noise = rng.gamma(.3, 1., (36,)+SHAPE)
    onehot = (np.arange(36)[:, None, None, None] == predicted_base(subject)[None]).astype(np.float64)
    p = .55*onehot+.45*noise/noise.sum(axis=0)
    return (p/p.sum(axis=0)).astype(np.float32)


def image(subject):
    rng = np.random.default_rng(4000+number(subject))
    return rng.uniform(.1, 1., SHAPE)+.3*(truth35(subject) % 5)


def source_sha(subject):
    return hexof(f'aligned_norm:{subject}')


def field(short):
    rng = np.random.default_rng(int(short[:4]))
    z, y, x = np.meshgrid(*[np.arange(n) for n in SHAPE], indexing='ij')
    u = np.stack([1.4*np.sin(.5*x+rng.uniform(0, 6))*np.cos(.3*y), 1.1*np.cos(.4*z+rng.uniform(0, 6)), .9*np.sin(.35*y+.2*x+rng.uniform(0, 6))], axis=-1)
    return np.ascontiguousarray(u, dtype=np.float64)


def model_entry(fold, contact):
    return dict(segmenter_id=SEG_ID, fold=fold, member_model_sha256=[hexof(f'{SEG_ID}-{fold}-{k}') for k in range(2)], label_contact_subjects=list(contact),
                label_contact_subjects_sha256=subject_list_sha256(list(contact)))


def models(fold0=FOLD0, fold1=FOLD1):
    return {f'{SEG_ID}_foldA': model_entry(0, fold0), f'{SEG_ID}_foldB': model_entry(1, fold1)}


def write_prediction(pred_root, subject, registry_sha, model_table, *, changes=None):
    d = Path(pred_root)/subject
    d.mkdir(parents=True)
    probs = posterior(subject)
    groups = group_probabilities(probs, GROUPING, dtype=np.float32)
    np.savez_compressed(d/'probs36.npz', p=probs.astype(np.float16))
    np.savez_compressed(d/GROUPS_FILE, p=groups.astype(np.float16))
    ids = sorted(model_table)
    members = [dict(model_id=m, fold='AB'[model_table[m]['fold']], seed=k, member_sha256=sha, temperature=1.2) for m in ids for k, sha in enumerate(model_table[m]['member_model_sha256'])]
    prov = dict(layer='structure', segmenter_id=SEG_ID, model_ids=ids, model_sha256={m: list(model_table[m]['member_model_sha256']) for m in ids}, folds={m: model_table[m]['fold'] for m in ids},
                trained_with_labels=True, test_labels_seen=False, subject=subject, policy='all_folds_mean', members=members, **GROUPING.record(),
                grouping_check=dict(grouping_status=GROUPING.status, grouping_status_admitted=True, allow_unverified_grouping=False), input_sha256={'aligned_norm.nii.gz': source_sha(subject)},
                output_sha256={'probs36.npz': io_.sha_file(d/'probs36.npz'), GROUPS_FILE: io_.sha_file(d/GROUPS_FILE)}, grid=dict(shape_zyx=list(SHAPE), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.]),
                registry_path='configs/bayesreg/segmenters_hs_v1.json', registry_sha256=registry_sha, environment=dict(code=dict(kind='synthetic')), runtime_seconds=0.)
    prov.update(changes or {})
    write_json(d/PROVENANCE_FILE, prov)
    return d


def labels_of_prediction(d):
    """L as the method's reader gives it: argmax of the float16 posterior divided by its channel sum (float64), ties to the smallest label, int16."""
    from bayesreg.seg.channels import read_back_probabilities
    P, _ = read_back_probabilities(Path(d)/'probs36.npz')
    return core.argmax_smallest(P)


def write_pair(stage, fixed, moving):
    d = Path(stage)/f'{fixed}__{moving}'
    d.mkdir(parents=True)
    arrays = dict(fixed=image(fixed), fixed_seg4=LOOKUP[truth35(fixed)].astype(float), fixed_seg35=truth35(fixed).astype(float), moving=image(moving), moving_seg4=LOOKUP[truth35(moving)].astype(float),
                  moving_seg35=truth35(moving).astype(float), initial_u=np.zeros(SHAPE+(3,)))
    np.savez_compressed(d/'pair.npz', **arrays)
    manifest = dict(pair=d.name, fixed_subject=fixed, moving_subject=moving, shape_zyx=list(SHAPE), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm',
                    sources=dict(fixed=dict(sha256=source_sha(fixed)), moving=dict(sha256=source_sha(moving))), pair_sha256=io_.sha_file(d/'pair.npz'),
                    baseline_label_metrics=dict(seg35=dict(macro_dice=.5)))
    write_json(d/'manifest.json', manifest)
    return d


def evaluation_module():
    spec = importlib.util.spec_from_file_location('_pq_test_evaluate', ROOT/'experiments'/'d0'/'evaluate.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['_pq_test_evaluate'] = module
    spec.loader.exec_module(module)
    return module


EV = evaluation_module()


def truth_reader(stage):
    def read_truth(pair):
        d = Path(stage)/pair
        with np.load(d/'pair.npz') as z:
            labs = {k: np.array(z[k]) for k in ('fixed_seg4', 'fixed_seg35', 'moving_seg4', 'moving_seg35')}
        return labs, json.loads((d/'manifest.json').read_text()), io_.sha_file(d/'pair.npz')
    return read_truth


def run_mc0_module():
    spec = importlib.util.spec_from_file_location('_pq_test_run_mc0', ROOT/'experiments'/'mc0'/'run_mc0.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['_pq_test_run_mc0'] = module
    spec.loader.exec_module(module)
    return module


RUN_MC0 = run_mc0_module()


def actual_metrics_of(v, labs):
    """The recorded after.metrics of P3 as P3 wrote it: experiments/mc0/run_mc0.field_metrics itself (run_fl -> run_ef.evaluate_repair -> run_mc0.field_metrics), not pq_eval's copy (revision 1, review
    minor 3: e1 is checked against an independent reference), through JSON as the record holds it."""
    return json.loads(json.dumps(RUN_MC0.field_metrics(np.asarray(v), labs, (1., 1., 1.), EV)))


def build(tmp_path, *, prediction_changes=None, synthetic=False):
    """The world; returns a namespace with ctx (pq_io.Context on it), the paths, the truth reader, the guard of the evaluation and the expected values.  synthetic: the
    synthetic training draw SYN_FOLD0, SYN_FOLD1 (C23.9-13 (8), revision 6e)."""
    tmp_path = Path(tmp_path)
    fold0, fold1 = (SYN_FOLD0, SYN_FOLD1) if synthetic else (FOLD0, FOLD1)
    repo, results, images = tmp_path/'repo', tmp_path/'results', tmp_path/'images'
    (repo/'runs'/'20261107-pq').mkdir(parents=True)
    shutil.copy(ROOT/pins.PLAN_REL, repo/pins.PLAN_REL)
    (repo/'configs'/'bayesreg').mkdir(parents=True)
    shutil.copy(ROOT/pins.SEG4_TABLE_REL, repo/pins.SEG4_TABLE_REL)
    shutil.copy(ROOT/'configs'/'bayesreg'/'structure_groups_v1.json', repo/'configs'/'bayesreg'/'structure_groups_v1.json')
    model_table = models(fold0, fold1) if synthetic else models()
    registry = write_json(repo/pins.ANCHOR['registry'], dict(schema='bayesreg.segmenters.v1', comment='synthetic', models=model_table))
    registry_sha = io_.sha_file(registry)
    stage = repo/pins.STAGE_REL
    pairs = [(SUBJECTS[i], SUBJECTS[i+1]) for i in range(len(SHORTS))]
    for f, m in pairs:
        write_pair(stage, f, m)
    write_json(stage/'selection.json', dict(pairs=[f'{f}__{m}' for f, m in pairs]))
    (repo/'splits'/'bayesreg_v1').mkdir(parents=True)
    (repo/'splits'/'bayesreg_v1'/'train_subjects.txt').write_text(''.join(f'{s}\n' for s in fold0+fold1))
    write_json(repo/pins.FOLDS_REL, dict(seed=1, rule='synthetic', subjects={**{s: dict(fold='A', role='train') for s in fold0}, **{s: dict(fold='B', role='train') for s in fold1}}))
    sources = [dict(name='stage_L2R_val_v1', dataset='OASIS', path=f'{pins.STAGE_REL}/selection.json', sha256=io_.sha_file(stage/'selection.json'), never_for_training=False, subjects=list(SUBJECTS)),
               dict(name='bayesreg_train_v1', dataset='OASIS', path='splits/bayesreg_v1/train_subjects.txt', sha256=io_.sha_file(repo/'splits'/'bayesreg_v1'/'train_subjects.txt'), never_for_training=False,
                    subjects=list(fold0+fold1))]
    write_json(repo/pins.ALLOWLIST_REL, dict(schema='bayesreg.dev_subjects.v1', subjects=dict(OASIS=list(SUBJECTS)+list(fold0+fold1)), sources=sources))
    git(repo, 'init', '-q')
    commit_all(repo, 'the synthetic code tree')
    pred_root = results/pins.PRED_ROOT_REL
    v4, labels = {}, {}
    for s in SUBJECTS:
        d = write_prediction(pred_root, s, registry_sha, model_table, changes=(prediction_changes or {}).get(s))
        v4[s] = pins.V4Row(probs36=io_.sha_file(d/'probs36.npz'), provenance=io_.sha_file(d/PROVENANCE_FILE), structure_groups=io_.sha_file(d/GROUPS_FILE))
        labels[s] = labels_of_prediction(d)
    from jd_curl.three_d.grid import make_grid_3d
    grid = make_grid_3d(nx=SHAPE[2], ny=SHAPE[1], nz=SHAPE[0], hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')
    p3, run_json, fields = {}, {}, {}
    truth = truth_reader(stage)
    for (f, m), short in zip(pairs, SHORTS):
        t88, rep = results/'p3_v1'/'runs'/'C_first5'/short, results/'p3_v1'/'runs'/'RC_c600'/short
        t88.mkdir(parents=True)
        rep.mkdir(parents=True)
        u = field(short)
        np.savez(t88/'field_0600.npz', u=u)
        tapered, _ = frozen_formula_taper(u, grid)
        tapered = np.ascontiguousarray(tapered)
        rng = np.random.default_rng(5000+int(short[:4]))
        nodes = np.sort(rng.choice(np.prod(SHAPE), 40, replace=False)).astype(np.int64)
        values = tapered.reshape(-1, 3)[nodes]*.9
        np.savez(rep/'repair.npz', nodes=nodes, values=values)
        v = tapered.copy()
        v.reshape(-1, 3)[nodes] = values
        state = io_.sha_array(v)
        fields[short] = v
        pair = f'{f}__{m}'
        run = dict(schema='bayesreg.p2.run.v1', pair=pair, status='done', oracle=False, segmenter_id=SEG_ID, field=dict(file='field_0600.npz', sha256=io_.sha_file(t88/'field_0600.npz')),
                   labels_sha256=dict(fixed=io_.sha_array(labels[f]), moving=io_.sha_array(labels[m])))
        write_json(t88/'run.json', run)
        write_json(t88/'metrics_0600.json', dict(metrics=dict(seg35=.5, seg4=.6), tapered_u_sha256=io_.sha_array(tapered)))
        repair = dict(schema='bayesreg.fl.repair.v1', pair=pair, oracle=False, status='certified', passed=True,
                      source=dict(directory=str(t88), file='field_0600.npz', field_sha256=run['field']['sha256'], tapered_u_sha256=io_.sha_array(tapered)),
                      attempt=dict(file='repair.npz', sha256=io_.sha_file(rep/'repair.npz'), n_free_nodes=int(nodes.size)),
                      certificate=dict(status='certified_feasible', reason='positive_enclosure_cover', min_jd_lower_bound=.25, n_cells_checked=int(np.prod([n-1 for n in SHAPE])), n_infeasible=0,
                                       n_uncertain=0, state_hash=state, certificate_version='synthetic'))
        write_json(rep/'repair.json', repair)
        labs, _, _ = truth(pair)
        write_json(rep/'repair_metrics.json', dict(after=dict(metrics=actual_metrics_of(v, labs)), repaired_u_sha256=state))
        p3[short] = pins.P3Row('C_first5', 'RC_c600', repair_npz=io_.sha_file(rep/'repair.npz'), state_hash=state, field_npz=run['field']['sha256'], repair_json=io_.sha_file(rep/'repair.json'))
        run_json[short] = io_.sha_file(t88/'run.json')
    members = results/pins.PQ_ROOT_REL/'members'
    rows = {}
    for s in SUBJECTS:
        V = np.random.default_rng(6000+number(s)).uniform(0., .2, SHAPE).astype(np.float32)
        io_.save_npz_once(members/s/io_.VAR_FILE, V=V)
        rows[s] = dict(m5=True, agreement=1., var_sha256=io_.sha_file(members/s/io_.VAR_FILE))
    write_json(members/'opened_files_members.json', dict(m6=True, files=[], breaches=[], n_files=0))
    write_json(members/io_.MEMBERS_RECORD, dict(schema='bayesreg.pq.members.v1', subjects=rows, m5_passed=True))
    images.mkdir()
    (images/SUBJECTS[0]).mkdir()
    (images/SUBJECTS[0]/'aligned_seg35.nii.gz').write_bytes(b'not a label file, but the name is one')
    anchor = dict(pins.ANCHOR, registry_sha256=registry_sha)
    ctx = io_.Context(repo=repo, source_root=repo, results=results, images=images, shorts=SHORTS, p3=p3, run_json=run_json, v4=v4, anchor=anchor, taper=frozen_formula_taper)
    return SimpleNamespace(ctx=ctx, repo=repo, results=results, images=images, stage=stage, registry=registry, labels=labels, fields=fields, truth=truth, pairs=pairs, grid=grid,
                           guard=None, models=model_table, folds=(fold0, fold1))


def reset_sealed_cache():
    """Exception (13) of contract C23.9-13 (revision 6f): before each in-process call of a PQ unit, the guard's cache of the sealed lists is reset
    (bayesreg.io.sealed._reset_cache, tests only), as a unit's own process starts with it empty; run_pq refuses to install its spy over a populated cache (C23.6-5)."""
    from bayesreg.io import sealed
    sealed._reset_cache()


class NoSealed:
    """The sealed lists of the synthetic world: none of its subjects is sealed."""
    subjects = {}

    def which(self, subject):
        return None


def guard_of(world):
    from bayesreg.io.guard import load_dev_allowlist
    return dict(allowlist=load_dev_allowlist(world.repo/pins.ALLOWLIST_REL), sealed=NoSealed(), training=set(world.folds[0]+world.folds[1]))


def release(world, sha, *, note=None, separate=False):
    """The lead's release in the synthetic tree: the list's sha256 and the release note (by default the one line `放行：<sha>`), committed (in one commit unless separate)."""
    repo = world.repo
    (repo/'runs'/'20261107-pq'/'outputs_manifest.sha256').write_text(f'{sha}  outputs_manifest.txt\n')
    if separate:
        commit_all(repo, 'the list')
    (repo/'runs'/'20261107-pq'/'release.md').write_text(f'放行：{sha}\n' if note is None else note)
    commit_all(repo, 'release')
