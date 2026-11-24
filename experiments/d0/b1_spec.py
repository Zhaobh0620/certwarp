"""B1 (runs/20261103-b1/plan.md, docs/bayesreg/05_D0协议.md §15): the arm specification of the option arms of the frozen solver.

A specification is a JSON file, one entry per arm; everything not listed stays configuration C (experiments/d0/run_queue.sh OPTS and PRE):
  {"schema": "b1.arms.v1", "arms": [{"arm": NAME, "inputs_root": DIR, "seconds": S, "changes": {...}, "expansion": false}, ...]}
  arm          [A-Za-z][A-Za-z0-9_]*, not the name of an earlier arm (A0.., A5, A5P): unit tag-b1-<arm>-<ffff>-<mmmm>, run id b1_<arm>_*
  inputs_root  absolute; holds d0_inputs.json and stage_L2R_val_v1/ (make_inputs.py layout); read only
  seconds      a budget of the frozen parser (--seconds)
  changes      only geometry_weight_levels (four positive reals, --geometry-weight-levels) and pre_optimizer_iterations (four positive
               integers, the iterations of --pre-optimizer); a value equal to configuration C is refused: it would not be a change
  expansion    true: the arm runs only on the winner's configuration of analysis/decision_b1.json (also on the first 5 pairs).  Oracle inputs,
               and the arms B1W_A5 and B1W_v2, always count as such, whatever the specification says (review I2)
  registry     optional, relative to the repository (default configs/bayesreg/segmenters.json): the segmenter registry of non-oracle inputs; it must be
               committed, clean, the one the inputs record names, with the sha256 the record holds

The one place that turns an entry into what the frozen prepare must write: expected_protocol() (check_prepared.py --b1-spec compares with it,
key by key, and every key the entry does not list stays the configuration C reference; the reference file itself is never changed).

  b1_spec.py shell SPEC                              one tab-separated line per arm, for run_queue.sh
  b1_spec.py arm-record SPEC ARM                     the JSON record of one arm (run directory: b1_arm.json)
  b1_spec.py gate --selection-spec F SPEC DECISION ARM...   exit 1 unless the decision comes from F and every ARM has the winner's configuration
  b1_spec.py check-inputs SPEC REPO RUNS             the A5P checks, per arm, on the record that its inputs_root holds; earlier runs of the arm
  b1_spec.py expansion --decision D --out-dir DIR    the expansion specifications of the plan, from the decision (nothing if the winner is configC)
  b1_spec.py v2 --decision D --inputs-root DIR --out F [--registry REL]   the specification of B1W_v2: the new segmenter's inputs on the winner's configuration
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = 'b1.arms.v1'
BUDGETS = (30, 45, 60, 90, 120, 180, 240, 330, 600, 3600)      # eb808be scripts/run_stage_3d_e48_pg.py BUDGETS
CONFIG_C = dict(seconds=90, geometry_weight_levels=[1., 1., .1, .03], pre_optimizer_iterations=[50, 50, 100, 60])
CHANGE_KEYS = ('geometry_weight_levels', 'pre_optimizer_iterations')
ENTRY_KEYS = {'arm', 'inputs_root', 'seconds', 'changes', 'expansion', 'note', 'registry'}
REGISTRY = 'configs/bayesreg/segmenters.json'                   # the default registry of an entry
V2_REGISTRY = 'configs/bayesreg/segmenters_v2.json'             # unet35_v2_full (runs/20261103-unv2/plan.md)
EARLIER_ARMS = {'A0', 'A1', 'A2', 'A3', 'A4p', 'A1Ph', 'A1Ps', 'X1h', 'X1f', 'A5', 'A5P'}
WATCHDOG_C, SECONDS_C = 300, 90                                 # configuration C: 90 s with a 300 s watchdog (protocol §2)
ORACLE_ARM = 'B1W_A5'                                           # the plan: the winner's configuration on the oracle inputs
V2_ARM = 'B1W_v2'                                               # revision 1: the winner's configuration on the inputs of the new segmenter
A5_INPUTS = Path('/home/rmcfz/seq_vp_tag_results/d0_v1_a5/inputs/A5')


class SpecError(ValueError):
    pass


def watchdog_seconds(seconds):
    """The watchdog in proportion to the budget (300 s for 90 s, 1100 s for 330 s), never below configuration C's."""
    return max(WATCHDOG_C, math.ceil(WATCHDOG_C*seconds/SECONDS_C))


def _real(x):
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def _check_changes(arm, changes):
    if not isinstance(changes, dict):
        raise SpecError(f'{arm}: changes must be an object')
    unknown = set(changes)-set(CHANGE_KEYS)
    if unknown:
        raise SpecError(f'{arm}: changes {sorted(unknown)} are not options of this tool (only {CHANGE_KEYS})')
    out = {}
    if 'geometry_weight_levels' in changes:
        v = changes['geometry_weight_levels']
        if not isinstance(v, list) or len(v) != 4 or not all(_real(x) for x in v):
            raise SpecError(f'{arm}: geometry_weight_levels must be four positive reals')
        out['geometry_weight_levels'] = [float(x) for x in v]
    if 'pre_optimizer_iterations' in changes:
        v = changes['pre_optimizer_iterations']
        if not isinstance(v, list) or len(v) != 4 or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in v):
            raise SpecError(f'{arm}: pre_optimizer_iterations must be four positive integers')
        out['pre_optimizer_iterations'] = list(v)
    for key, value in out.items():
        if value == CONFIG_C[key]:
            raise SpecError(f'{arm}: {key} equals configuration C: not a change')
    return out


def check_entry(entry):
    if not isinstance(entry, dict):
        raise SpecError('an arm must be an object')
    arm = entry.get('arm')
    if not isinstance(arm, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', arm):
        raise SpecError(f'bad arm name {arm!r}')
    if arm in EARLIER_ARMS:
        raise SpecError(f'{arm}: the name of an earlier arm')
    missing = {'inputs_root', 'seconds', 'changes'}-set(entry)
    unknown = set(entry)-ENTRY_KEYS
    if missing or unknown:
        raise SpecError(f'{arm}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}')
    root = entry['inputs_root']
    if not isinstance(root, str) or not Path(root).is_absolute() or '\t' in root or '\n' in root:
        raise SpecError(f'{arm}: inputs_root must be an absolute path')
    seconds = entry['seconds']
    if isinstance(seconds, bool) or seconds not in BUDGETS:
        raise SpecError(f'{arm}: seconds {seconds!r} is not a budget of the frozen parser {BUDGETS}')
    expansion = entry.get('expansion', False)
    if not isinstance(expansion, bool):
        raise SpecError(f'{arm}: expansion must be true or false')
    registry = entry.get('registry', REGISTRY)
    if not isinstance(registry, str) or not registry or Path(registry).is_absolute() or '..' in Path(registry).parts:
        raise SpecError(f'{arm}: registry must be a path relative to the repository')
    return dict(arm=arm, inputs_root=root, seconds=int(seconds), changes=_check_changes(arm, entry['changes']), expansion=expansion, registry=registry)


def load_spec(path):
    """The validated entries of a specification file, in file order; SpecError for anything else."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise SpecError(f'{path}: {exc}') from None
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA or not isinstance(doc.get('arms'), list) or not doc['arms']:
        raise SpecError(f'{path}: not a {SCHEMA} specification with at least one arm')
    entries = [check_entry(e) for e in doc['arms']]
    names = [e['arm'] for e in entries]
    if len(set(names)) != len(names):
        raise SpecError(f'{path}: duplicate arms {names}')
    return entries


def arm_entry(path, arm):
    for entry in load_spec(path):
        if entry['arm'] == arm:
            return entry
    raise SpecError(f'{path}: no arm {arm}')


def configuration(entry):
    """What makes two arms the same solver configuration: the budget and the changes."""
    return dict(seconds=entry['seconds'], changes=entry['changes'])


def is_oracle(entry):
    """Truth labels coloured these inputs: the name of the oracle arm, or what the inputs record says."""
    return entry['arm'] == ORACLE_ARM or (_record_of(entry) or {}).get('oracle') is True


def is_gated(entry):
    """The arm needs the decision on every pair: not left to the specification's own flag (review I2)."""
    return entry.get('expansion', False) is True or entry['arm'] in (ORACLE_ARM, V2_ARM) or is_oracle(entry)


def method_name(entry):
    gw, it = (entry['changes'].get(k) for k in CHANGE_KEYS)
    return (f"frozen_eb808be_b1_{entry['seconds']}s"+('' if gw is None else '_gw'+'_'.join(f'{x:g}' for x in gw))
            + ('' if it is None else '_preit'+'_'.join(str(x) for x in it))+('_oracle' if is_oracle(entry) else ''))


def _swap(text, old, new):
    if text.count(old) != 1:
        raise SpecError(f'the configuration C scope does not hold {old!r} exactly once')
    return text.replace(old, new)


def expected_protocol(reference, entry):
    """The configuration C reference protocol with exactly the entry's changes: task.wall_seconds and task.name (--seconds), the changed option
    and the sentence of the scope that the frozen prepare writes for it.  Every other key is the reference's, unchanged."""
    out = json.loads(json.dumps(reference))
    task = out['task']
    task['wall_seconds'] = entry['seconds']
    task['name'] = f"{task['route']}__{task['hierarchy']}__{entry['seconds']}s"
    gw, it = (entry['changes'].get(k) for k in CHANGE_KEYS)
    if gw is not None:
        out['scope'] = _swap(out['scope'], f"mu=nu={reference['geometry_weight_levels']}", f'mu=nu={gw}')
        out['geometry_weight_levels'] = list(gw)
    if it is not None:
        out['scope'] = _swap(out['scope'], f"'iterations': {tuple(reference['pre_optimizer']['iterations'])}", f"'iterations': {tuple(it)}")
        out['pre_optimizer']['iterations'] = list(it)
    return out


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record_of(entry):
    path = Path(entry['inputs_root'])/'d0_inputs.json'
    return json.loads(path.read_text()) if path.is_file() else None


def shell_lines(path):
    """arm, inputs_root, seconds, watchdog, --geometry-weight-levels value or -, iterations value or -, gated 0/1, the input arm or -."""
    lines = []
    for e in load_spec(path):
        gw, it = (e['changes'].get(k) for k in CHANGE_KEYS)
        record = _record_of(e) or {}
        lines.append('\t'.join((e['arm'], e['inputs_root'], str(e['seconds']), str(watchdog_seconds(e['seconds'])),
                                '-' if gw is None else ','.join(f'{x:g}' for x in gw), '-' if it is None else '/'.join(str(x) for x in it),
                                str(int(is_gated(e))), str(record.get('arm', '-')))))
    return lines


def arm_record(path, arm):
    entry = arm_entry(path, arm)
    return dict(entry, spec=str(Path(path).resolve()), spec_sha256=sha_file(path), watchdog_seconds=watchdog_seconds(entry['seconds']),
                input_arm=(_record_of(entry) or {}).get('arm'), input_oracle=is_oracle(entry), method=method_name(entry))


def gate(path, decision_path, arms, selection_spec):
    """A gated item (pair index >= 5, or a gated arm) needs the plan's decision: made from selection_spec (its sha256, review M7), with a winner, and the arm has the
    winner's configuration.  When the winner is configC the plan runs no oracle arm (review M6); configC's (90 s, no change) stays open for B1W_v2."""
    decision = Path(decision_path)
    doc = json.loads(decision.read_text()) if decision.is_file() else {}
    spec = doc.get('winner_spec')
    print(f"B1 selection: {decision}: winner {doc.get('winner')}; {(doc.get('selection') or {}).get('text')}")
    if doc.get('winner') is None or not spec:
        print('no decision: nothing beyond the selection runs')
        return 1
    source = (doc.get('sources') or {}).get('selection_spec_sha256')
    want_source = sha_file(selection_spec) if Path(selection_spec).is_file() else None
    if want_source is None or source != want_source:
        print(f'the decision was made from a selection specification with sha256 {source}, the repository file {selection_spec} has {want_source}: REFUSED')
        return 1
    want = configuration(spec)
    bad = 0
    for arm in dict.fromkeys(arms):
        entry = arm_entry(path, arm)
        have = configuration(entry)
        ok = have == want and not (doc['winner'] == 'configC' and is_oracle(entry))
        bad += not ok
        print(f"{arm}: configuration {have}; winner {doc['winner']}'s {want}{'; no oracle arm when the winner is configC' if doc['winner'] == 'configC' and is_oracle(entry) else ''}: {'ok' if ok else 'REFUSED'}")
    return 1 if bad else 0


def _tracked_and_clean(repo, rel):
    tracked = subprocess.run(['git', '-C', str(repo), 'ls-files', '--error-unmatch', '--', rel], capture_output=True).returncode == 0
    status = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--', rel], capture_output=True, text=True)
    return tracked and status.returncode == 0 and not status.stdout.strip()


def _repository_file(repo, name):
    """The relative path of a file named by an inputs record, or None when it is not a file of this repository."""
    try:
        path = Path(name).resolve()
        return path, str(path.relative_to(Path(repo).resolve()))
    except (TypeError, ValueError):
        return None, None


def check_one_input(entry, repo):
    """The start checks of the A5 queue, per arm, on the record of the arm's inputs_root: not partial; the colour table it names is a committed,
    clean file of this repository with the recorded sha256; unless the inputs are oracle inputs, the same for the entry's registry, which must be the one the record
    names, status registered, tracked and clean when the inputs were made.  Returns the problems."""
    record = _record_of(entry)
    if record is None:
        return [f"no d0_inputs.json under {entry['inputs_root']}"]
    bad = []
    if record.get('partial'):
        bad.append('partial inputs record (a dry run)')
    if not isinstance(record.get('arm'), str):
        bad.append('the record names no arm')
    if not isinstance(record.get('oracle'), bool):
        bad.append('the record does not say oracle true or false')
    table = record.get('colour_table') or {}
    path, rel = _repository_file(repo, table.get('table_file'))
    if rel is None:
        bad.append(f"colour table {table.get('table_file')} is not a file of {repo}")
    elif not path.is_file() or sha_file(path) != table.get('table_sha256'):
        bad.append(f'colour table {rel}: sha256 differs from the record')
    elif not _tracked_and_clean(repo, rel):
        bad.append(f'colour table {rel}: not committed or not clean')
    if record.get('oracle') is not True:
        reg = record.get('structure_registry') or {}
        path, rel = _repository_file(repo, reg.get('path'))
        git = reg.get('git') or {}
        if rel is None:
            bad.append(f"registry {reg.get('path')} is not a file of {repo}")
        elif rel != entry['registry']:
            bad.append(f"registry: the record names {rel}, the entry {entry['registry']}")
        elif reg.get('status') != 'registered' or git.get('tracked') is not True or git.get('clean') is not True:
            bad.append(f'registry {rel}: the record says status {reg.get("status")}, git {git}')
        elif not path.is_file() or sha_file(path) != reg.get('sha256'):
            bad.append(f'registry {rel}: sha256 differs from the record')
        elif not _tracked_and_clean(repo, rel):
            bad.append(f'registry {rel}: not committed or not clean')
    return bad


def earlier_runs_problems(entry, runs_root):
    """A run directory of this arm from another configuration or other inputs would mix two arms under one name."""
    bad = []
    for path in sorted((Path(runs_root)/entry['arm']).glob('*/b1_arm.json')):
        old = json.loads(path.read_text())
        if configuration(old) != configuration(entry) or old.get('inputs_root') != entry['inputs_root']:
            bad.append(f'{path}: configuration {configuration(old)} inputs {old.get("inputs_root")}; this specification {configuration(entry)} inputs {entry["inputs_root"]}')
    return bad


def check_inputs(path, repo, runs_root):
    failed = 0
    for entry in load_spec(path):
        bad = check_one_input(entry, repo)+earlier_runs_problems(entry, runs_root)
        record = _record_of(entry) or {}
        print(f"{entry['arm']} inputs check: {entry['inputs_root']} arm {record.get('arm')} oracle {record.get('oracle')}; {'ok' if not bad else 'FAILED: '+'; '.join(bad)}")
        failed += bool(bad)
    return 1 if failed else 0


def expansion_specs(decision_path, out_dir, a5_inputs=A5_INPUTS):
    """The plan's expansion: the winner on its own inputs (run it with --pairs rest-14) and the same configuration on the oracle inputs (B1W_A5,
    run it with --pairs first-5).  A file that exists with other content is refused, never overwritten.  Returns the written paths."""
    doc = json.loads(Path(decision_path).read_text())
    winner, spec = doc.get('winner'), doc.get('winner_spec')
    if winner is None or winner == 'configC' or not spec:
        raise SpecError(f'{decision_path}: winner {winner}: the plan has no expansion')
    common = dict(seconds=spec['seconds'], changes=spec['changes'], expansion=True)
    files = {f'expand_{winner}.json': dict(arm=winner, inputs_root=spec['inputs_root'], **common),
             f'{ORACLE_ARM}.json': dict(arm=ORACLE_ARM, inputs_root=str(a5_inputs), **common)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, entry in files.items():
        check_entry(entry)
        text = json.dumps(dict(schema=SCHEMA, source=str(decision_path), arms=[entry]), indent=1)+'\n'
        path = out_dir/name
        if path.exists() and path.read_text() != text:
            raise SpecError(f'{path} exists with other content (nothing is overwritten)')
        path.write_text(text)
        written.append(path)
    return written


def v2_spec(decision_path, inputs_root, out, registry=V2_REGISTRY):
    """The specification of B1W_v2: the inputs of the new segmenter on the winner's configuration (configC: 90 s, no change).  It needs a decision with a winner.
    A file that exists with other content is refused, never overwritten."""
    path = Path(decision_path)
    doc = json.loads(path.read_text()) if path.is_file() else {}
    winner, spec = doc.get('winner'), doc.get('winner_spec')
    if winner is None or not spec:
        raise SpecError(f'{decision_path}: no decision with a winner: no specification for {V2_ARM}')
    entry = dict(arm=V2_ARM, inputs_root=str(inputs_root), seconds=spec['seconds'], changes=spec['changes'], expansion=True, registry=registry)
    check_entry(entry)
    text = json.dumps(dict(schema=SCHEMA, source=str(decision_path), winner=winner, arms=[entry]), indent=1)+'\n'
    out = Path(out)
    if out.exists() and out.read_text() != text:
        raise SpecError(f'{out} exists with other content (nothing is overwritten)')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('shell'); s.add_argument('spec', type=Path)
    r = sub.add_parser('arm-record'); r.add_argument('spec', type=Path); r.add_argument('arm')
    g = sub.add_parser('gate'); g.add_argument('--selection-spec', type=Path, required=True); g.add_argument('spec', type=Path)
    g.add_argument('decision', type=Path); g.add_argument('arms', nargs='+')
    c = sub.add_parser('check-inputs'); c.add_argument('spec', type=Path); c.add_argument('repo', type=Path); c.add_argument('runs', type=Path)
    e = sub.add_parser('expansion'); e.add_argument('--decision', type=Path, required=True); e.add_argument('--out-dir', type=Path, required=True)
    e.add_argument('--a5-inputs', type=Path, default=A5_INPUTS)
    v = sub.add_parser('v2'); v.add_argument('--decision', type=Path, required=True); v.add_argument('--inputs-root', type=Path, required=True)
    v.add_argument('--out', type=Path, required=True); v.add_argument('--registry', default=V2_REGISTRY)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'shell':
            print('\n'.join(shell_lines(a.spec)))
            return 0
        if a.cmd == 'arm-record':
            print(json.dumps(arm_record(a.spec, a.arm), indent=1))
            return 0
        if a.cmd == 'gate':
            return gate(a.spec, a.decision, a.arms, a.selection_spec)
        if a.cmd == 'check-inputs':
            return check_inputs(a.spec, a.repo, a.runs)
        if a.cmd == 'v2':
            print(v2_spec(a.decision, a.inputs_root, a.out, a.registry))
            return 0
        for path in expansion_specs(a.decision, a.out_dir, a.a5_inputs):
            print(path)
        return 0
    except SpecError as exc:
        print(f'B1 specification: {exc}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
