"""What R3's composition (compose_r3.py) and its development reading (eval_r3.py) share: the U-Net members and their checks, one forward pass that feeds every arm, the mixture of the U-Net
family with the nnU-Net mean, the decoding switch.  The arithmetic is UF's and R2's, not new:
  - the U-Net family's mean is bayesreg.seg.unet.infer.ensemble_probabilities' (softmax(z_m / T_m) added in the arm's order, divided by n, renormalized; experiments/r2/eval_r2.Accumulator), stored as
    float16 as infer.segment_subject stores probs36, its argmax with ties to the smallest label;
  - with nnU-Net: (1 - w) P_U + w P_N, P_U the float16 read back (float64 divided by the channel sum), P_N the stored nnU-Net mean read back (read_back_probabilities), argmax by slabs of z
    (experiments/uf/eval_uf.half_and_half with w = n_N / (n_U + n_N) in place of its fixed 1/2; at w = 1/2 the same numbers, a test compares);
  - with D: the same mixture as a renormalized float32 posterior, handed to bayesreg.seg.decode (R3b) with the switch's labels, objective and candidate floor.
"""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _p in (EXPERIMENTS/'unv2', EXPERIMENTS/'ul', EXPERIMENTS/'nn', EXPERIMENTS/'uf', EXPERIMENTS/'r2', HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import r3_common as r3c  # noqa: E402

SLAB = 8


class Refused(Exception):
    """An input is not the plan's: nothing of a truth label has been read."""


# ---- the decoding switch (run_r3.py decode-switch; the review's I1 and the lead's ruling, 11-11 ~3:43 PM CST) ------------------------------------------------
# The order: R3a and R3b merged -> R3b's calibration diagnosis -> the switch -> the smoke -> prepare -> the chain.  So R3b's final decode.py is the one of the smoke's and the training's
# snapshot; the switch pins its sha256 and the sha256 of R3b's decision record, and compose checks both again in the snapshot (the module it imports, the record on disk).
SYNTHETIC_DECISION_FORMAT = 'bayesreg.r3.synthetic_decision.v1'      # the synthetic world's stand-in for R3b's record (r3_synthetic); refused on real data


def decode_module_path():
    """The decode.py this process imports (in a unit: the snapshot's)."""
    import bayesreg.seg.decode as decode
    return Path(decode.__file__)


def load_decision(path, *, synthetic=False, expected_sha256=None):
    """R3b's decision record as a dict with its sha256 under decision_sha256.  Real data: the fixed path r3c.R3B_DECISION through R3b's own reader,
    r3b_decision.load_calibration_decision(require_pass=True): a verdict other than 'pass', another plan or decoder, a synthetic record are refused there.  The synthetic world: its stand-in
    (SYNTHETIC_DECISION_FORMAT, verdict 'pass').  Refused (Refused) otherwise, and when expected_sha256 is given and differs."""
    path = Path(path)
    if synthetic:
        document = r3c.read_json(path)
        if not isinstance(document, dict) or document.get('format') != SYNTHETIC_DECISION_FORMAT or document.get('verdict') != 'pass':
            raise Refused(f'{path}: not a synthetic decision record with the verdict pass')
        sha = r3c.sha256_file(path)
        if expected_sha256 is not None and sha != expected_sha256:
            raise Refused(f'{path}: sha256 {sha} is not the pinned {expected_sha256}')
        return dict(document, decision_sha256=sha)
    if path.resolve() != Path(r3c.R3B_DECISION).resolve():
        raise Refused(f"{path}: not R3b's decision record {r3c.R3B_DECISION}")
    try:
        import r3b_decision
    except ImportError:
        raise Refused("R3b's experiments/r3/r3b_decision.py is not in this code tree: D cannot be on") from None
    try:
        return r3b_decision.load_calibration_decision(path, require_pass=True, expected_sha256=expected_sha256)
    except (r3b_decision.DecodeNotAdmitted, ValueError, OSError) as exc:
        raise Refused(f"R3b's decision record refuses D: {type(exc).__name__}: {exc}") from None


def switch_document(decision, *, words, synthetic=False):
    """The switch with D on (run_r3.py decode-switch --on; the synthetic world): the decision's decoder_spec, which must be decoder_spec of this code tree and not the stub (but in the synthetic
    world), decode.py's sha256 (the module imported here), which must be the one the decision recorded (r3c.DECISION_DECODE_KEY), the decision's path, sha256 and verdict."""
    from bayesreg.seg.decode import decoder_spec
    spec = json.loads(json.dumps(decision.get('decoder_spec') or {}))
    try:
        again = json.loads(json.dumps(decoder_spec(tuple(spec.get('labels') or ()), objective=spec.get('objective'), candidate_floor=spec.get('candidate_floor'))))
    except Exception as exc:
        raise Refused(f"the decision's decoder spec is refused by bayesreg.seg.decode: {exc}") from None
    if again != spec:
        raise Refused(f"the decision's decoder spec {spec} is not decoder_spec's {again} in this code tree")
    if again.get('stub') and not synthetic:
        raise Refused('bayesreg.seg.decode is the stub in this code tree: D cannot be on')
    module = decode_module_path()
    sha = r3c.sha256_file(module)
    if decision.get(r3c.DECISION_DECODE_KEY) != sha:
        raise Refused(f"decode.py here ({module}, sha256 {sha}) is not the one the decision recorded ({r3c.DECISION_DECODE_KEY}: {decision.get(r3c.DECISION_DECODE_KEY)})")
    return dict(schema='bayesreg.r3.decode_switch.v2', decode=True, words=words, spec=spec, decode_py=dict(path=r3c.DECODE_PY_REL, sha256=sha),
                decision=dict(path=str(decision.get('path') or r3c.R3B_DECISION), sha256=decision['decision_sha256'], verdict=decision.get('verdict'), synthetic=bool(synthetic)))


def read_switch(path, *, synthetic=False):
    """(on, spec, record): <OUT>/decode_switch.json.  Refused when it is not there (the composition does not guess).  With D on, again in this process (in a unit: the snapshot's code): the
    decoder spec is decoder_spec of this tree and not the stub (but in the synthetic world); decode.py imported here has the pinned sha256; the decision record on disk has the pinned sha256 and
    still says 'pass' through R3b's reader, with this spec and this decode.py."""
    document = r3c.read_json(path)
    if not isinstance(document, dict) or document.get('decode') not in (True, False):
        raise Refused(f'{path}: no decoding switch (run_r3.py decode-switch --on/--off, after R3b\'s diagnosis): the composition does not start without it')
    if not document['decode']:
        return False, None, dict(document, sha256=r3c.sha256_file(path))
    pinned = document.get('decision') or {}
    if bool(pinned.get('synthetic')) != bool(synthetic):
        raise Refused(f'{path}: a {"synthetic" if pinned.get("synthetic") else "real"} switch here')
    decision = load_decision(pinned.get('path') or r3c.R3B_DECISION, synthetic=synthetic, expected_sha256=pinned.get('sha256'))
    again = switch_document(dict(decision, path=pinned.get('path')), words=document.get('words'), synthetic=synthetic)
    if again['spec'] != document.get('spec') or again['decode_py'] != document.get('decode_py') or again['decision']['sha256'] != pinned.get('sha256'):
        raise Refused(f"{path}: the switch is not what the decision and decode.py give now: spec {document.get('spec')} / {again['spec']}, decode.py {document.get('decode_py')} / {again['decode_py']}")
    return True, document['spec'], dict(document, sha256=r3c.sha256_file(path))


# ---- the U-Net members ----------------------------------------------------------------------------------------------------------------------------------
def l_expected_config(uf_root=r3c.UF_OUT, l_set=None):
    """An L member's recorded config: its UF namesakes' (the four UF records agree, the recipe v3_large as UF trained it) with only UF_CHANGES replaced by the --set values of L
    (r3c.l_train_config(); `l_set` is the synthetic world's tiny one)."""
    configs = []
    for name in r3c.UF_ORDER:
        configs.append(json.loads((Path(uf_root)/name/'train_record.json').read_text(encoding='utf-8'))['config'])
    if any(c != configs[0] for c in configs):
        raise Refused("UF's four members do not share one config")
    config = dict(configs[0])
    l_set = r3c.l_train_config() if l_set is None else l_set
    for key in r3c.UF_CHANGES:
        config[key] = l_set[key]
    if any(config.get(k) != v for k, v in l_set.items()):
        raise Refused(f'the L config differs from its --set values outside {r3c.UF_CHANGES}')
    return config


def l_record_problems(train_record, name, namesake, expected_config):
    """uf_common.member_record_problems (recipe v3_large, the config, early stop off, the schedule to its end, the last step, the fold's subjects) with L's steps (expected_config's max_iters)."""
    import uf_common
    return uf_common.member_record_problems(train_record, name, namesake, expected_config=expected_config, max_iters=expected_config['max_iters'])


def load_unet(*, uf_root, member_root, uf_evaluated, device, keys, pinned=None, l_config=None):
    """{(family, name): (model, blob, summary)} for `keys` (('uf', name) or (L_FAMILY, name)); problems.  UF's: member.pt the one eval_uf.json evaluated (and the pinned one on real data), the last step
    with a Gaussian window.  L's: the plan's member (l_record_problems against its UF namesake's subjects), member.pt re-hashed (eval_u1.load_arm), model_last.pt the recorded file, train_record.json
    the file member.json was written from."""
    import eval_r2
    import eval_u1
    namesakes = eval_r2.uf_namesakes(uf_root, pinned)
    members, problems = {}, []
    for family, name in keys:
        directory = Path(uf_root)/name if family == 'uf' else Path(member_root)/family/name
        if not (directory/'member.json').is_file():
            problems.append(f'{family}/{name}: no member.json (member missing)')
            continue
        model, blob, record = eval_u1.load_arm(directory, device)
        train_record = json.loads((directory/'train_record.json').read_text(encoding='utf-8'))
        found = []
        if r3c.sha256_file(directory/'train_record.json') != record.get('train_record_sha256'):
            found.append(f'{family}/{name}: train_record.json is not the file member.json was written from')
        if family == 'uf':
            if record['member_sha256'] != uf_evaluated.get(name):
                found.append(f'{family}/{name}: member.pt is not the one eval_uf.json evaluated')
        else:
            found += l_record_problems(train_record, name, namesakes[name[4]], l_config if l_config is not None else l_expected_config(uf_root))
            last = directory/str(train_record.get('checkpoint_file'))
            if not last.is_file() or r3c.sha256_file(last) != train_record.get('checkpoint_sha256'):
                found.append(f'{family}/{name}: {last.name} missing or its sha256 differs from train_record.json')
        if blob.get('checkpoint_choice') != 'last' or blob.get('window_gaussian_sigma') is None or not blob.get('window'):
            found.append(f'{family}/{name}: member.pt is not a last-step member with a Gaussian sliding window')
        problems += found
        members[(family, name)] = (model, blob, dict(eval_r2.member_summary(directory, blob, record, train_record), problems=found))
    return members, problems


def forward(members, order, image, *, device, arms, singles):
    """One pass over the members in `order` (a list of keys): for each ensemble arm (arms: {arm: [keys in its order]}) the float16 mean (eval_r2's Accumulator), for each single arm (singles:
    {arm: key}) the argmax of its logits.  Returns ({arm: p16}, {arm: labels})."""
    import torch
    import eval_r2
    from bayesreg.seg.unet.train import predict_logits
    accumulators = {a: eval_r2.Accumulator() for a in arms}
    tops = {}
    for key in order:
        model, blob, _ = members[key]
        z = predict_logits(model, image, device=device, amp=blob['amp'], window=tuple(blob['window']), gaussian_sigma=blob['window_gaussian_sigma'])
        for arm, k in singles.items():
            if k == key:
                tops[arm] = z.argmax(1)[0].to(torch.uint8).cpu().numpy()
        p = torch.softmax(z.float()/float(blob['temperature']), dim=1)[0]
        del z
        for arm, plan in arms.items():
            if key in plan:
                accumulators[arm].add(p)
        del p
    p16 = {}
    for arm, acc in accumulators.items():
        p16[arm] = acc.probabilities().cpu().numpy().astype(np.float16)
    return p16, tops


# ---- the mixture ---------------------------------------------------------------------------------------------------------------------------------------
def mixed_labels(p16, p_nn=None, weight=0., *, slab=SLAB):
    """The argmax (ties to the smallest label) of the U-Net family alone (p_nn None: p16's own argmax, as eval_r2) or of (1 - weight) P_U + weight P_N by slabs of z."""
    if p_nn is None:
        if weight:
            raise ValueError('a weight without nnU-Net probabilities')
        return p16.argmax(axis=0).astype(np.uint8)
    if p16.dtype != np.float16 or p16.shape != p_nn.shape or not 0 < weight < 1:
        raise ValueError(f'float16 U-Net and nnU-Net probabilities of one shape and 0 < w < 1 required: {p16.dtype} {p16.shape} {p_nn.shape} {weight}')
    out = np.empty(p16.shape[1:], dtype=np.uint8)
    for z0 in range(0, p16.shape[1], slab):
        part = p16[:, z0:z0+slab].astype(np.float64)
        part /= part.sum(axis=0)
        out[z0:z0+slab] = ((1-weight)*part+weight*p_nn[:, z0:z0+slab]).argmax(axis=0)
    return out


def mixed_posterior(p16, p_nn=None, weight=0., *, slab=SLAB):
    """The mixture whose argmax mixed_labels takes, as a float64 posterior (the same arithmetic, element by element; the review's m6): P_U = p16 / its channel sum, or (1 - w) P_U + w P_N.
    Each part sums to 1 within float64 rounding, so the sum is within 1e-6 of 1 (R3b's input rule) without a second division; values are clipped at 1 (one ulp above it cannot happen in a sum
    of probabilities but for rounding).  The input of bayesreg.seg.decode; its argmax is mixed_labels' on every voxel (a test)."""
    out = np.empty(p16.shape, dtype=np.float64)
    for z0 in range(0, p16.shape[1], slab):
        part = p16[:, z0:z0+slab].astype(np.float64)
        part /= part.sum(axis=0)
        if p_nn is not None:
            part = (1-weight)*part+weight*p_nn[:, z0:z0+slab]
        out[:, z0:z0+slab] = part
    np.minimum(out, 1., out=out)
    return out


def candidate_labels(p16, p_nn, weight, decoder=None):
    """(labels uint8, decode record or None) of one arm: the mixture's argmax, or with `decoder` (dict(labels, objective, candidate_floor)) bayesreg.seg.decode.decode_with_record of its posterior."""
    if decoder is None:
        return mixed_labels(p16, p_nn, weight), None
    from bayesreg.seg.decode import decode_with_record
    posterior = mixed_posterior(p16, p_nn, weight)
    labels, record = decode_with_record(posterior, decoder['labels'], objective=decoder['objective'], candidate_floor=decoder['candidate_floor'])
    del posterior
    return labels.astype(np.uint8), record


def read_nn_mean(directory, subject, shape, *, row):
    """(P_N float64, labels.npz as a dict): the stored nnU-Net mean of one subject; probs36.npz and labels.npz must be the files check.json recorded (`row`, its rows[subject])."""
    from bayesreg.seg.channels import read_back_probabilities
    path, labels_path = Path(directory)/subject/'probs36.npz', Path(directory)/subject/'labels.npz'
    if r3c.sha256_file(path) != row.get('probs36_sha256') or r3c.sha256_file(labels_path) != row.get('labels_sha256'):
        raise Refused(f'{path.parent}: probs36.npz or labels.npz is not the file the nnU-Net check passed')
    p_nn, _ = read_back_probabilities(path, shape=(36,)+tuple(shape))
    with np.load(labels_path, allow_pickle=False) as npz:
        labels = {k: npz[k] for k in npz.files}
    return p_nn, labels
