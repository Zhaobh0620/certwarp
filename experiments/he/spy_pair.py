"""HE: the method on one pair with the spies on (contract C22.5-2 gate (2) on a real pair: the method's process opens no member of the original pair.npz but fixed, moving, initial_u, and no evaluation record).
For the lead's GPU smoke on one development pair (runs/20261106-he/任务书.md: not part of this package); the frozen values, the default paths, the identity checks of run_hx.py pair.

    PYTHONPATH=<frozen>:<code> python -B experiments/he/spy_pair.py --stage-dir DIR --pair NAME --out DIR [--device cuda] [--launcher unit|direct]

The supervisor and its two children in this one process (run_hx.InProcessChild), so that one spy sees all of them; configuration C, if a fallback needs it, in its own process on the derived pair, whose arrays the
method's gate (1) checks against the original (fallback.firewall.gate).  Writes <out>/spy.json; exit 0 when the spies saw nothing forbidden, 8 otherwise; HE2: 9 (he_core.STOP_QUEUE_EXIT) passed through when the spies saw nothing forbidden and run_pair stopped on
an implementation error of C (D86 (2)).  The output under <out>/output is the method's own.
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_hx  # noqa: E402
import spies as spying  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--stage-dir', required=True)
    ap.add_argument('--pair', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--launcher', choices=('unit', 'direct'), default='unit')
    o = ap.parse_args(argv)
    problem = run_hx.out_problem(o.out)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    ctx = run_hx.make_context(o.stage_dir, o.pair, o.out, device=o.device, launcher=o.launcher)
    spy = spying.Spies(run_hx.pair_path(ctx)).install()
    error = None
    try:
        rc = run_hx.run_pair(ctx, child=run_hx.InProcessChild())
    except spying.Breach as exc:
        rc, error = 8, str(exc)
    finally:
        spy.remove()
    admitted = (Path(o.out)/'admission.json').is_file()               # a pair refused at admission must show no read at all; an admitted one its loaders' reads, each seen (spies.py, the positive condition)
    record = dict(spy.record(expect_reads=admitted), admitted=admitted, run_rc=rc, breach=error, role='C22.5-2 gate (2) on a real pair: the spies of experiments/he/spies.py over the supervisor and its in-process children')
    run_hx.he.write_json(Path(o.out)/'spy.json', record)
    print(f"spy: passed {record['passed']} (positive condition {record['positive_condition']}: loader calls {len(record['loader_calls'])}, reads seen {record['reads_seen']}); "
          f"forbidden {record['forbidden_access']}; evaluation records opened {record['evaluation_records_opened']}")
    if record['passed'] and error is None and rc == run_hx.he.STOP_QUEUE_EXIT:      # HE2 review minor 3: an implementation error of C (D86 (2)) keeps its own exit, as run_hx.py pair returns it
        return rc
    return 0 if record['passed'] and rc == 0 and error is None else 8


if __name__ == '__main__':
    sys.exit(main())
