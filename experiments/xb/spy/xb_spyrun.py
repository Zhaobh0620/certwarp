"""XB (runs/20261107-xb/任务书.md item 1 (e)): run a method script of the external baseline runner under the spy.  The standard library only (it runs in each baseline's conda environment).

    <env>/bin/python -I -B experiments/xb/spy/xb_spyrun.py POLICY.json LOG.jsonl SCRIPT ARG...

What it does: reads the policy (xb_policy.py), installs a Python audit hook (sys.addaudithook) that sees every open the interpreter makes -- builtins.open, io.open, os.open, the imports' open_code, and so
numpy, pathlib, shutil, nibabel, SimpleITK's and itk's Python layers as far as they open through Python -- writes every distinct path with its class to LOG.jsonl (one JSON line each, flushed), and on a
forbidden or unlisted open writes the breach line and raises Breach (a BaseException, so that a method's `except Exception` cannot swallow it): the process stops (越界就停).  Then it runs SCRIPT as
`python -I -B SCRIPT ARG...` would: sys.argv = [SCRIPT, ARG...], sys.path[0] = SCRIPT's directory, runpy.run_path(SCRIPT, run_name='__main__').  The method scripts are not changed (byte-identical copies
of BL2VAL's harness).

What it cannot see (stated in runs/20261107-xb/实现说明.md): an open made by C or C++ code without going through Python (CUDA, cuDNN, ITK's readers called from C++, TensorFlow's file system, a library's
own fopen); a file that was opened before the hook (none: the hook is the first thing).  The driver's /proc sampler (xb_spy.FdSampler) watches the process tree for those, once a second.
Exit codes: the script's own; 97 when the hook stopped the process (the driver reads LOG.jsonl).
"""
import json
import os
import runpy
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import xb_policy  # noqa: E402

BREACH_EXIT = 97


class Breach(BaseException):
    """The method opened what the policy forbids or does not list."""


class Hook:
    def __init__(self, policy, log_path):
        self.policy = policy
        self.log = open(log_path, 'a', encoding='utf-8', buffering=1)       # opened before the hook is installed
        self.seen = set()
        self.local = threading.local()
        self.breaches = 0

    def write(self, row):
        self.log.write(json.dumps(row, ensure_ascii=False)+'\n')
        self.log.flush()

    def __call__(self, event, args):
        if event != 'open' or getattr(self.local, 'busy', False):
            return
        self.local.busy = True
        try:
            path = xb_policy.norm(args[0])
            if path is None:
                return
            write = xb_policy.is_write(args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else 0)
            cls, why = self.policy.classify(path, write)
            key = (path, write)
            if key not in self.seen or cls != 'allowed':
                self.seen.add(key)
                self.write(dict(path=path, write=write, cls=cls, why=why, pid=os.getpid()))
            if cls != 'allowed':
                self.breaches += 1
                self.write(dict(event='breach', path=path, write=write, cls=cls, why=why, pid=os.getpid()))
                raise Breach(f'xb spy: {cls} open of {path} ({why})')
        finally:
            self.local.busy = False


def main(argv):
    if len(argv) < 3:
        print('usage: xb_spyrun.py POLICY.json LOG.jsonl SCRIPT ARG...', file=sys.stderr)
        return 2
    policy_path, log_path, script, *args = argv
    policy = xb_policy.Policy.load(policy_path)
    hook = Hook(policy, log_path)
    hook.write(dict(event='start', policy=policy.document.get('name'), script=os.path.realpath(script), pid=os.getpid(), python=sys.version.split()[0]))
    sys.addaudithook(hook)
    script = os.path.realpath(script)
    sys.argv = [script, *args]
    sys.path[0] = os.path.dirname(script)
    try:
        runpy.run_path(script, run_name='__main__')
    except Breach as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return BREACH_EXIT
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    finally:
        hook.write(dict(event='end', breaches=hook.breaches, pid=os.getpid()))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
