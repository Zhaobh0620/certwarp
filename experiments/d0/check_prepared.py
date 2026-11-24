"""D0: check one prepared configuration-C run directory before it is launched (run_queue.sh calls this after `prepare`).

Checks (any failure -> exit 1 and a reason; the queue then stops):
  1. protocol.json equals experiments/d0/configC_reference.json:reference_protocol in every key except stage, task.pair,
     input_sha256 and pair_source (90 s, v41, export reserve 1 s, weights, pre-optimizer, metric policies, ...);
     task.pair == --pair, stage == --stage, pair_dataset == stage_L2R_val_v1,
     pair_source == data/three_d/stage_L2R_val_v1/<pair>;
  2. protocol.json source_sha256 == the freeze live tree (configC_reference.json:freeze_source_py_sha256), and the
     snapshot/ copies have these hashes;
  3. inputs/<pair>/pair.npz and manifest.json have the sha256 recorded in protocol.json input_sha256, and they equal
     the expected arm input (--expect-pair-sha256 / --expect-manifest-sha256: the files of the arm stage directory);
     the manifest's pair_sha256 equals the pair.npz sha256 and names --pair.
With --native-reference (gate A X1, runs/20261102-gate-a/plan.md; the T92 switches of docs/contract_draft_D2_ablations.md): the
protocol must carry native_reference and, with --native-penalty-scale, native_penalty_scale with these values (and no
native_penalty_scale without it); its scope must be the reference scope plus only the T92 sentence; every other key as in 1-3.
With --b1-spec and --b1-arm (runs/20261103-b1/plan.md, protocol §15): an intended deviation.  The reference is first changed by exactly the
arm's entry (b1_spec.expected_protocol: --seconds, and the changes it lists with their sentence of the scope); every key is then compared as in
1-3, so what the entry does not list must still equal configuration C.  configC_reference.json itself is not changed.
Prints one JSON line with the result. Usage:
  check_prepared.py RUN_DIR --pair P --stage S --expect-pair-sha256 H --expect-manifest-sha256 H
                    [--native-reference identity [--native-penalty-scale S]] [--b1-spec FILE --b1-arm ARM]
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_spec  # noqa: E402

REF = Path(__file__).resolve().parent/'configC_reference.json'
IGNORED = ('stage', 'task', 'input_sha256', 'pair_source', 'source_sha256')
T92_KEYS = ('native_reference', 'native_penalty_scale')
T92_SENTENCE = re.compile(r'T92 \(D2 ablation, .*?never a recommended configuration; ')     # the frozen prepare's scope text


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def check(run, pair, stage, expect_pair, expect_manifest, native_reference=None, native_penalty_scale=None, b1_entry=None):
    ref = json.loads(REF.read_text())
    proto = json.loads((run/'protocol.json').read_text())
    rp = ref['reference_protocol']
    if b1_entry is not None:                                  # B1: the reference with exactly the entry's changes; all the checks below as ever
        rp = b1_spec.expected_protocol(rp, b1_entry)
    bad = []
    keys = (set(rp) | set(proto)) - set(IGNORED)
    if native_reference is not None:                          # gate A X1: the two T92 keys and the scope, nothing else
        keys -= set(T92_KEYS) | {'scope'}
        if proto.get('native_reference') != native_reference:
            bad.append(f"protocol.native_reference {proto.get('native_reference')!r} != {native_reference!r}")
        if proto.get('native_penalty_scale') != native_penalty_scale:
            bad.append(f"protocol.native_penalty_scale {proto.get('native_penalty_scale')!r} != {native_penalty_scale!r}")
        if T92_SENTENCE.sub('', str(proto.get('scope')), count=1) != rp.get('scope'):
            bad.append('protocol.scope is not the configuration C scope plus the T92 sentence')
    for k in sorted(keys):
        if proto.get(k, '<missing>') != rp.get(k, '<missing>'):
            bad.append(f'protocol.{k} differs from configuration C')
    task_ref = dict(rp['task'], pair=pair)
    if proto.get('task') != task_ref:
        bad.append(f"protocol.task {proto.get('task')} != {task_ref}")
    if proto.get('stage') != stage:
        bad.append(f"protocol.stage {proto.get('stage')!r} != {stage!r}")
    if proto.get('pair_source') != f'data/three_d/stage_L2R_val_v1/{pair}':
        bad.append(f"protocol.pair_source {proto.get('pair_source')!r}")
    if proto.get('source_sha256') != ref['freeze_source_py_sha256']:
        bad.append('protocol.source_sha256 != freeze live tree (jd_curl/, scripts/ .py)')
    else:
        wrong = [k for k, v in proto['source_sha256'].items() if sha(run/'snapshot'/k) != v]
        if wrong:
            bad.append(f'snapshot files differ: {wrong[:5]}')
    data = run/'inputs'/pair
    got = {n: sha(data/n) for n in ('pair.npz', 'manifest.json')}
    if proto.get('input_sha256') != got:
        bad.append('inputs differ from protocol.input_sha256')
    if got['pair.npz'] != expect_pair or got['manifest.json'] != expect_manifest:
        bad.append(f'inputs are not the expected arm inputs (pair.npz {got["pair.npz"][:12]} vs {expect_pair[:12]}, '
                   f'manifest {got["manifest.json"][:12]} vs {expect_manifest[:12]})')
    man = json.loads((data/'manifest.json').read_text())
    if man.get('pair_sha256') != got['pair.npz'] or man.get('pair') != pair:
        bad.append('manifest pair_sha256 / pair name inconsistent')
    result = dict(ok=not bad, problems=bad, input_sha256=got, reference=str(REF), reference_sha256=sha(REF),
                  preprocessing_d0_arm=(man.get('preprocessing_d0') or {}).get('arm', 'A0 (original)'))
    if b1_entry is not None:
        result['b1_intended_deviation'] = b1_spec.configuration(b1_entry)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--pair', required=True)
    ap.add_argument('--stage', required=True)
    ap.add_argument('--expect-pair-sha256', required=True)
    ap.add_argument('--expect-manifest-sha256', required=True)
    ap.add_argument('--native-reference', choices=('identity',), default=None)
    ap.add_argument('--native-penalty-scale', type=float, default=None)
    ap.add_argument('--b1-spec', type=Path, default=None)
    ap.add_argument('--b1-arm', default=None)
    a = ap.parse_args(argv)
    if a.native_penalty_scale is not None and a.native_reference is None:
        ap.error('--native-penalty-scale: only with --native-reference')
    if (a.b1_spec is None) != (a.b1_arm is None):
        ap.error('--b1-spec and --b1-arm go together')
    if a.b1_spec is not None and a.native_reference is not None:
        ap.error('--b1-spec: not with --native-reference')
    try:
        r = check(a.run, a.pair, a.stage, a.expect_pair_sha256, a.expect_manifest_sha256,
                  native_reference=a.native_reference, native_penalty_scale=a.native_penalty_scale,
                  b1_entry=None if a.b1_spec is None else b1_spec.arm_entry(a.b1_spec, a.b1_arm))
    except Exception as exc:                                   # a broken prepared directory is a failed check
        r = dict(ok=False, problems=[f'{type(exc).__name__}: {exc}'])
    print(json.dumps(r))
    return 0 if r['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
