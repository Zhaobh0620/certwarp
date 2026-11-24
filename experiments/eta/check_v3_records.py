"""Check that experiments/eta/sweep.py at its v3 design still builds the v3 sweep's launch records (work package ETA4;
ETA4 review "已核对" and lead's follow-up: the reviewer's method, kept in the repository). Read only on the v3 root.

    python -B experiments/eta/check_v3_records.py --scratch DIR [--v3-root DIR] [--sweep-args "ARGS"] [--json OUT]

Every probe, run and evaluation launch record of the v3 root (<v3-root>/launch/*.txt) is rebuilt through the real
Sweep.probe_phase, Sweep.one_run, Sweep.evaluate and Sweep.launch of this repository's sweep.py, with the v3 command
line (--sweep-args, default "--logvol-closed27-share 1": eta_sweep_v3 was started with only that and --root). No unit
is started: a capturing fake machine writes the minimal outputs that probe_phase, one_run and evaluate need. The code
tree, its HEAD and the registry record are taken from the v3 records themselves; the pairs from the v3 summary.json.
Records go to --scratch (a new directory outside the result roots of the sweep); then the scratch root is replaced by
the v3 root, UTC stamps are normalised, and the "configuration:" line and the systemd command line of each record are
compared byte for byte with the v3 record. The eval and probe wrappers' log names carry their own launch stamps, which
the normalisation covers. Exit 0 when every record is identical, 1 otherwise, 2 usage.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys

REPO = Path(__file__).resolve().parents[2]
STAMP = re.compile(r"[0-9]{8}T[0-9]{6}Z")
CODE_LINE = re.compile(r"code: (\S+) \(HEAD ([0-9a-f]+|\S+)\)")


def load_sweep():
    spec = importlib.util.spec_from_file_location("_check_v3_sweep", REPO/"experiments"/"eta"/"sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Capture:
    """The machine of the check: no unit is started; the outputs the sweep reads afterwards are written."""
    self_unit = ""

    def __init__(self, sweep):
        self.s = sweep

    def unit_active(self, unit):
        return False

    def other_units(self):
        return []

    def active_units(self):
        return []

    def gpu_free_mib(self):
        return 24000

    def gpu_state(self):
        return "check (no GPU read)"

    def mem_available_mib(self):
        return 15000

    def home_free_gb(self):
        return 500

    def train_state(self):
        return "check"

    def uptime(self):
        return 1.

    def loadavg(self):
        return "0 0 0"

    def dmesg_since(self, t0, pattern):
        return 0

    def sleep(self, seconds):
        raise AssertionError("the check never waits")

    def drivers_running(self):
        return []

    def run(self, cmd, record):
        unit = next(c for c in cmd if c.startswith("--unit="))[len("--unit="):]
        if "-eval-" in unit:
            Path(cmd[cmd.index("eval-wrapper")+2]).write_text("0\n")
        elif "-probe-" in unit:
            Path(cmd[cmd.index("probe-wrapper")+2]).write_text("0\n")
            out = Path(cmd[cmd.index("--out")+1])
            out.mkdir(parents=True)
            (out/"probe.json").write_text(json.dumps(dict(
                nu_star=874., alpha_star=1., u1_state_hash="0"*64, eta_tmp=dict(tau=[1., .5, .5, .5], nu=3.87),
                configuration=self.s.probe_configuration(), model_channels=14)))
        else:
            out = Path(cmd[cmd.index("--out")+1])
            out.mkdir(parents=True)
            (out/"status.json").write_text(json.dumps(dict(status="budget_exhausted", wording="预算收尾", exit_code=0)))
        return 0


def key_lines(path, root, v3_root):
    text = Path(path).read_text(encoding="utf-8").replace(str(root), str(v3_root)).splitlines()
    conf = next(x for x in text if x.startswith("configuration: "))
    cmd = text[text.index("systemd command:")+1]
    return text[0], text[1], STAMP.sub("STAMP", conf), STAMP.sub("STAMP", cmd)


def check(v3_root, scratch, sweep_args=("--logvol-closed27-share", "1")):
    """Rebuild every launch record of `v3_root` under `scratch`; returns dict(records, rebuilt, identical, different)."""
    sw = load_sweep()
    v3_root, scratch = Path(v3_root), Path(scratch)
    records = sorted((v3_root/"launch").glob("*.txt"))
    if not records:
        raise SystemExit(f"no launch records under {v3_root}/launch")
    first = records[0].read_text(encoding="utf-8").splitlines()
    match = CODE_LINE.match(first[2])
    if match is None:
        raise SystemExit(f"{records[0]}: no code line")
    stored = json.loads(next(x for x in first if x.startswith("configuration: "))[len("configuration: "):])
    opts = sw.parse_args(["--root", str(scratch), *sweep_args])
    s = sw.Sweep(opts, None)
    s.sys = Capture(s)
    s.root.mkdir(parents=True, exist_ok=False)
    s.code, s.code_sha = Path(match.group(1)), match.group(2)
    if opts.data_root is None:      # a v3 run started from the main checkout: its default data root
        s.data_root = Path("/home/rmcfz/code/python_projects/seq_vp_tag/data/three_d")
    s.allowlist = opts.allowlist or str(s.code/"configs"/"bayesreg"/"dev_subjects.json")
    s.registry_path = Path(opts.segmenter_registry) if opts.segmenter_registry else \
        s.code/"configs"/"bayesreg"/"segmenters.json"
    s.registry_record = stored.get("registry")
    summary = json.loads((v3_root/"summary.json").read_text(encoding="utf-8"))
    s.pairs = summary["pairs"]
    kinds = [r.read_text(encoding="utf-8").splitlines()[0].split(": ", 1) for r in records]
    if any(k == "probe" for k, _ in kinds):
        s.probe_phase()
    for kind, name in kinds:
        if kind != "run":
            continue
        stage, label, short = name.split("-", 2)
        pair = next(p for p in s.pairs if sw.short_name(p) == short)
        s.one_run(stage, label, pair, s.root/"eta"/f"{stage}_{label}.json")
    mine = {}
    for path in sorted((scratch/"launch").glob("*.txt")):
        lines = key_lines(path, scratch, v3_root)
        mine[lines[0]] = lines
    same, different = 0, []
    for rec in records:
        want = key_lines(rec, v3_root, v3_root)
        if mine.get(want[0]) == want:
            same += 1
        else:
            different.append(rec.name)
    return dict(v3_root=str(v3_root), scratch=str(scratch), sweep_args=list(sweep_args), records=len(records),
                rebuilt=len(mine), identical=same, different=different)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -B experiments/eta/check_v3_records.py",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--v3-root", default="/home/rmcfz/seq_vp_tag_results/eta_sweep_v3")
    parser.add_argument("--scratch", required=True, help="a new directory for the rebuilt records")
    parser.add_argument("--sweep-args", default="--logvol-closed27-share 1",
                        help="the v3 command line apart from --root (default: eta_sweep_v3's)")
    parser.add_argument("--json", default=None, help="write the report here (a new file)")
    args = parser.parse_args(argv)
    scratch = Path(args.scratch).resolve()
    results = Path("/home/rmcfz/seq_vp_tag_results")
    if any(scratch == p or p in scratch.parents for p in (results/"eta_sweep_v1", results/"eta_sweep_v2",
                                                          results/"eta_sweep_v3", results/"eta_sweep_v4")):
        parser.error(f"--scratch {scratch}: never inside a sweep root")
    if scratch.exists():
        parser.error(f"--scratch {scratch} exists")
    report = check(args.v3_root, scratch, shlex.split(args.sweep_args))
    text = json.dumps(report, indent=1, ensure_ascii=False)
    print(text)
    if args.json:
        with open(args.json, "x", encoding="utf-8") as stream:
            stream.write(text+"\n")
    return 0 if report["identical"] == report["records"] and not report["different"] else 1


if __name__ == "__main__":
    sys.exit(main())
