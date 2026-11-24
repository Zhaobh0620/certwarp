"""XB's controlled limit experiment (prereg item 19, revision 1c item 2; runs/20261107-xb/limit_experiment_plan.md): a synthetic load in place of the driver, the standard library only; it opens no
file and touches no subject.

    python -I -B xb_synthetic_load.py memory  --mib M [--block-mib B]   allocate and touch M MiB in B MiB blocks (the trial's lowered MemoryMax is reached first), then hold for 60 s
    python -I -B xb_synthetic_load.py runtime --seconds S               sleep S seconds (the trial's lowered RuntimeMaxSec is reached first)
Prints one line per 256 MiB or per 10 s.  Exit 0 if it was not stopped (the experiment then reads "limit not reached").
"""
import argparse
import sys
import time

PAGE = 4096


def memory(mib, block_mib):
    held, total = [], 0
    while total < mib:
        b = bytearray(block_mib*2**20)
        for i in range(0, len(b), PAGE):
            b[i] = 1
        held.append(b)
        total += block_mib
        if total % 256 < block_mib:
            print(f'held {total} MiB', flush=True)
    time.sleep(60)
    return 0


def runtime(seconds):
    t0 = time.monotonic()
    while time.monotonic()-t0 < seconds:
        time.sleep(1.)
        if int(time.monotonic()-t0) % 10 == 0:
            print(f'{time.monotonic()-t0:.0f} s', flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('memory', 'runtime'))
    ap.add_argument('--mib', type=int, default=1024)
    ap.add_argument('--block-mib', type=int, default=16)
    ap.add_argument('--seconds', type=float, default=60.)
    o = ap.parse_args(argv)
    return memory(o.mib, o.block_mib) if o.mode == 'memory' else runtime(o.seconds)


if __name__ == '__main__':
    sys.exit(main())
