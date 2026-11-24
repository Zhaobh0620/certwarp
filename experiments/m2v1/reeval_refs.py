"""M2V1: re-evaluate the reference arms A0 and A4' of D0 with bayesreg.eval.cli (runs/20261031-m2v1-dev/plan.md, "统一
评测入口"; the lead, 2026-10-31). Read only on the D0 run directories and long.csv files.

    python -B experiments/m2v1/reeval_refs.py [--out DIR] [--code-commit 68721d1 | --code DIR] [--d0-root DIR]
        [--a4-root DIR] [--d0-long F] [--a4-long F] [--data-root DIR] [--stage stage_L2R_val_v1] [--allowlist PATH]
        [--arms A0,A4p] [--min-mem-available-mib 4096] [--wait-minutes 60] [--numba-threads 2]

Code: the evaluation runs in a plain export (`git archive`, not a worktree) of --code-commit (default 68721d1, the code
of the eta sweep and of the M2V1 runs) at <out>/code_<sha12>/ (bayesreg, jd_curl, configs, scripts, splits and the
sealed-list plan directory; code_commit.txt records the full sha and is checked on a restart). --code DIR uses an
existing tree instead (tests). The default allowlist is that tree's configs/bayesreg/dev_subjects.json.
Pairs: <data-root>/<stage>/selection.json, in its order (19 in the real stage). Per pair and arm the run directory is
<d0-root>/runs/A0/<pair> or <a4-root>/runs/A4p/<pair>; its run_id file must equal the run id of that pair in the
original long.csv (A0: --d0-long, A4p: --a4-long) -- the pair id check.
Routes:
  A0   configC: `python -m bayesreg.eval.cli <run dir> --pairs-stage <stage> --method configC_frozen_eb808be_90s --arm A0
       --run-id <run id> --cert auto (full for configuration C)`, as the GPU smoke re-evaluated A0 0438<-0439.
  A4p  disp_from_final_native: the A4' runs registered the recoloured pair they were given (protocol.json input_sha256
       of the recoloured pair.npz), so bayesreg.eval.cli's input check (review EV item 8) refuses them against the stage
       pair (exit 2, "nothing was read or written"; checked on 0438<-0439 before this script was written). The field is
       therefore taken from the run exactly as the configuration C path takes it -- d = bayesreg.eval.fields.u_to_d(u of
       <task>/final_native.npz, spacing of the run's manifest), computed by the code tree's own functions in a child
       process -- written to <out>/fields/A4p/<short>/disp.npz (d float64; meta: run dir, u and d sha256, the run's
       field role, metrics role and status, both pair sha256), and evaluated as a disp.npz directory with --pair <pair>
       --cert full --method configC_frozen_eb808be_90s --arm A4p --run-id <run id>. At 1 mm spacing d_to_u(d) is u bit
       for bit, so the certified field is the run's u. The derived disp.npz is deleted after a successful evaluation
       (its d sha256 stays in done.json and in the evaluator's metrics.json). field_role is then disp_only (the run's own
       configuration C role is recorded next to it).
Each evaluation is one plain child process (no systemd unit, so the gate of the running eta sweep never sees it):
`nice -n 19`, OPENBLAS / OMP / MKL / NUMEXPR threads 1, NUMBA_NUM_THREADS --numba-threads (2, the evaluator's own
default), CUDA_VISIBLE_DEVICES empty, NUMBA_CACHE_DIR <out>/cache/numba; one at a time. Before each child: MemAvailable >=
--min-mem-available-mib, polled every 30 s for at most --wait-minutes (then STOP).
Output per arm and pair: <out>/<arm>/<short>/<run_id>/metrics.json, long.csv, summary.json, eval.log, done.json (route,
exit code, seconds, peak RSS of the children so far, field sha256). Resume (hard constraint 4): done.json -> skip; an
evaluation directory without done.json -> failed earlier (recorded, not rerun).
Comparison: <out>/compare.json / compare.md -- per arm and pair the re-evaluated seg35, seg4, HD95, official and brain
SDlogJ against the original experiments/d0/evaluate.py values (long.csv), equal bit for bit or the difference, and the
certificate (status, minimum J lower bound) of both. Written once; a different recomputation goes to compare_<UTC>.*.
The differences are reported, not resolved: experiments/m2v1/summarize.py uses the re-evaluated values for the A0 / A4'
comparisons and shows the original ones as a reference column; which set to use is the lead's decision.
Exit codes: 0 every evaluation done, 8 STOP or a failed evaluation (the others still done).
"""
import argparse
from datetime import datetime, timezone
import glob
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarize = _load("_m2v1_reeval_summarize", "experiments/m2v1/summarize.py")

PY = "/home/rmcfz/miniconda3/envs/sci310/bin/python"
RESULTS = Path("/home/rmcfz/seq_vp_tag_results")
OUT = RESULTS/"m2v1_dev"/"ref_reeval"
CODE_COMMIT = "68721d1"
SNAPSHOT_PATHS = ("bayesreg", "jd_curl", "configs", "scripts", "splits",
                  "archive/seq_vp_3d/results/verification/3d_stage_HOLDOUT_v1/plan")
STAGE = "stage_L2R_val_v1"
METHOD = summarize.REF_METHOD
ARMS = dict(A0=dict(root_option="d0_root", long_option="d0_long", route="configC"),
            A4p=dict(root_option="a4_root", long_option="a4_long", route="disp_from_final_native"))
MEASURES = ("seg35", "seg4", "hd95", "sdlogj_official", "sdlogj_brain")
SCHEMA = "bayesreg.m2v1.ref_reeval.v1"
DERIVE = r'''
import json, sys
from pathlib import Path
import numpy as np
from bayesreg.eval.fields import array_sha256, describe_run, read_final_native, u_to_d
run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
info = describe_run(run_dir)
if info.kind != "configC":
    raise SystemExit(f"{run_dir}: kind {info.kind}, expected configC")
manifest = json.loads(Path(info.files["manifest.json"]).read_text())
u, rec = read_final_native(info.files["final_native.npz"], manifest["shape_zyx"])
spacing = tuple(float(h) for h in manifest["spacing_xyz"])
d = u_to_d(u, spacing)
meta = dict(source=str(run_dir), conversion="bayesreg.eval.fields.u_to_d(final_native u, run manifest spacing)",
            u_sha256=rec["u_sha256"], d_sha256=array_sha256(d), final_native_file_sha256=rec["file_sha256"],
            spacing_xyz=list(spacing), field_role=info.field_role, metrics_role=info.metrics_role,
            run_status=info.run_status, input_record=info.extra.get("inputs"))
out.mkdir(parents=True)
np.savez(out/"disp.npz", d=d, meta=np.array(json.dumps(meta)))
print(json.dumps(meta))
'''


class Stop(Exception):
    pass


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_once(path, document):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=1, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mem_available_mib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])//1024
    return 0


class Reeval:
    def __init__(self, o):
        self.o = o
        self.out = Path(o.out)
        self.code = None
        self.code_sha = None
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "experiments/m2v1"],
                               capture_output=True, text=True, check=False)
        self.script = dict(path=str(Path(__file__).resolve()), sha256=sha256_of(__file__),
                           repo_head=head.stdout.strip() or None, m2v1_uncommitted=bool(dirty.stdout.strip()))

    def log(self, text):
        line = f"{utc()} {text}"
        print(line, flush=True)
        with open(self.out/"reeval.log", "a", encoding="utf-8") as stream:
            stream.write(line+"\n")

    # ---- code ------------------------------------------------------------------------------------------------------
    def prepare_code(self):
        if self.o.code:
            self.code = Path(self.o.code).resolve()
            out = subprocess.run(["git", "-C", str(self.code), "rev-parse", "HEAD"], capture_output=True, text=True,
                                 check=False)
            self.code_sha = out.stdout.strip() or "not a git tree"
            self.log(f"code: existing tree {self.code} ({self.code_sha})")
            return
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", f"{self.o.code_commit}^{{commit}}"],
                             capture_output=True, text=True, check=False)
        if out.returncode:
            raise Stop(f"commit {self.o.code_commit} not found in {REPO}")
        sha = out.stdout.strip()
        code, record = self.out/f"code_{sha[:12]}", self.out/"code_commit.txt"
        if code.exists():
            if not record.is_file() or record.read_text().strip() != sha:
                raise Stop(f"{code} exists but {record} does not record {sha} (kept; the lead decides)")
        else:
            archive = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", sha, *SNAPSHOT_PATHS],
                                     capture_output=True, check=False)
            if archive.returncode:
                raise Stop(f"git archive {sha} failed: {archive.stderr.decode(errors='replace')[:300]}")
            code.mkdir()
            with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
                tar.extractall(code)
            if not record.exists():
                record.write_text(sha+"\n")
            self.log(f"code: git archive of {sha} ({', '.join(SNAPSHOT_PATHS)}) -> {code}")
        self.code, self.code_sha = code, sha
        for target in ("bayesreg/eval/cli.py", "bayesreg/eval/fields.py", "configs/bayesreg/dev_subjects.json"):
            if not (code/target).is_file():
                raise Stop(f"{code}: {target} missing")

    # ---- inputs ----------------------------------------------------------------------------------------------------
    def pairs(self):
        selection = read_json(Path(self.o.data_root)/self.o.stage/"selection.json")
        if selection is None:
            raise Stop(f"{self.o.data_root}/{self.o.stage}/selection.json missing")
        return [row["pair"] for row in selection["pairs"]]

    def env(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", NUMBA_NUM_THREADS=str(self.o.numba_threads),
                   PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR=str(self.out/"cache"/"numba"))
        env.pop("PYTHONPATH", None)
        return env

    def memory_gate(self, label):
        waited = 0.
        while True:
            avail = mem_available_mib()
            if avail >= self.o.min_mem_available_mib:
                return avail
            if waited >= 60*self.o.wait_minutes:
                raise Stop(f"before {label}: MemAvailable {avail} MiB < {self.o.min_mem_available_mib} MiB after "
                           f"{waited/60:.0f} min")
            if waited % 600 == 0:
                self.log(f"waiting before {label}: MemAvailable {avail} MiB < {self.o.min_mem_available_mib} MiB")
            time.sleep(30)
            waited += 30

    def child(self, command, log, label):
        avail = self.memory_gate(label)
        t0 = time.monotonic()
        with open(log, "a", encoding="utf-8") as stream:
            stream.write(f"# {utc()} cwd {self.code}; MemAvailable {avail} MiB\n# " + " ".join(map(str, command))+"\n")
            stream.flush()
            rc = subprocess.run(["nice", "-n", "19", *map(str, command)], cwd=self.code, env=self.env(), stdout=stream,
                                stderr=subprocess.STDOUT, check=False).returncode
        return rc, time.monotonic()-t0, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss//1024

    # ---- one evaluation ------------------------------------------------------------------------------------------------
    def one(self, arm, pair, original):
        short = summarize.short_name(pair)
        spec = ARMS[arm]
        run = Path(getattr(self.o, spec["root_option"]))/"runs"/arm/pair
        ev = self.out/arm/short
        if (ev/"done.json").is_file():
            self.log(f"skip {arm} {short} (done)")
            return "done"
        if ev.exists():
            self.log(f"{arm} {short}: an earlier attempt without done.json ({ev}); kept, not rerun")
            return "failed"
        rid_file = run/"run_id"
        rid = rid_file.read_text().strip() if rid_file.is_file() else None
        want = original["pairs"][pair]["run_id"]
        if rid != want:
            raise Stop(f"{run}: run_id {rid!r} differs from the original long.csv's {want!r} for {pair}")
        ev.mkdir(parents=True)
        log = ev/"eval.log"
        record = dict(arm=arm, pair=pair, run_dir=str(run), run_id=rid, route=spec["route"], code=str(self.code),
                      code_sha=self.code_sha, script=self.script)
        target, extra = run, []
        if spec["route"] == "disp_from_final_native":
            field = self.out/"fields"/arm/short
            if field.exists():
                raise Stop(f"{field} exists (an earlier derivation without an evaluation); kept, the lead decides")
            rc, seconds, peak = self.child([PY, "-B", "-c", DERIVE, run, field], log, f"derive {arm} {short}")
            if rc != 0 or not (field/"disp.npz").is_file():
                write_once(ev/"failed.json", dict(record, step="derive", exit=rc, recorded_utc=utc()))
                self.log(f"FAILED {arm} {short}: derivation exit {rc} (see {log})")
                return "failed"
            with np.load(field/"disp.npz", allow_pickle=False) as npz:
                meta = json.loads(str(npz["meta"]))
            stage_pair = read_json(Path(self.o.data_root)/self.o.stage/pair/"manifest.json") or {}
            record.update(derived=dict(meta, disp_npz=str(field/"disp.npz"), disp_file_sha256=sha256_of(field/"disp.npz"),
                                       derive_seconds=seconds),
                          input_record_pair_sha256=(meta.get("input_record") or {}).get("pair_sha256"),
                          stage_pair_sha256=stage_pair.get("pair_sha256"))
            target, extra = field, ["--pair", pair, "--cert", "full"]
        command = [PY, "-B", "-m", "bayesreg.eval.cli", target, "--pairs-stage", self.o.stage, "--out", "metrics.json",
                   "--long", ev/"long.csv", "--summary", ev/"summary.json", "--out-dir", ev, "--method", METHOD, "--arm",
                   arm, "--run-id", rid, "--data-root", self.o.data_root, "--allowlist", self.o.allowlist, *extra]
        rc, seconds, peak = self.child(command, log, f"evaluate {arm} {short}")
        metrics = sorted(glob.glob(str(ev/"*"/"metrics.json")))
        record.update(exit=rc, seconds=seconds, children_peak_rss_mib=peak, metrics=metrics, log=str(log))
        if rc != 0 or len(metrics) != 1:
            write_once(ev/"failed.json", dict(record, step="evaluate", recorded_utc=utc()))
            self.log(f"FAILED {arm} {short}: evaluation exit {rc}, {len(metrics)} metrics.json (see {log})")
            return "failed"
        doc = read_json(metrics[0]) or {}
        if spec["route"] == "disp_from_final_native":
            got = ((doc.get("field") or {}).get("disp") or {}).get("d_sha256")
            record["evaluated_d_sha256_matches_derived"] = got == record["derived"]["d_sha256"]
            (Path(record["derived"]["disp_npz"])).unlink()
            record["derived"]["deleted_after_evaluation"] = True
        record["done_utc"] = utc()
        write_once(ev/"done.json", record)
        m, c = doc.get("metrics") or {}, doc.get("certificate") or {}
        self.log(f"evaluated {arm} {short}: field_role {doc.get('field_role')}; seg35 {m.get('seg35')}, seg4 "
                 f"{m.get('seg4')}; certificate {c.get('status')}; {seconds:.0f} s; children peak RSS {peak} MiB")
        return "done"

    # ---- comparison ----------------------------------------------------------------------------------------------------
    def compare(self, pairs, originals):
        rows, counts = [], dict(compared=0, bitwise_equal=0, differ=0, missing=0)
        for arm in self.arms:
            for pair in pairs:
                short = summarize.short_name(pair)
                done = read_json(self.out/arm/short/"done.json")
                orig = originals[arm]["pairs"][pair]
                row = dict(arm=arm, pair=pair, run_id=orig["run_id"], route=ARMS[arm]["route"], done=done is not None)
                if done is None:
                    counts["missing"] += 1
                    rows.append(row)
                    continue
                doc = read_json(done["metrics"][0]) or {}
                m, c = doc.get("metrics") or {}, doc.get("certificate") or {}
                measures = {}
                for key in MEASURES:
                    a, b = m.get(key), orig.get(key)
                    equal = (a == b) if summarize.finite(a) and summarize.finite(b) else None
                    measures[key] = dict(reeval=a, original=b, equal_bitwise=equal,
                                         difference=(a-b) if summarize.finite(a) and summarize.finite(b) else None)
                    counts["compared"] += 1
                    counts["bitwise_equal" if equal else "differ"] += 1
                row.update(measures=measures, field_role=doc.get("field_role"), pair_in_record=doc.get("pair"),
                           run_id_in_record=doc.get("run_id"),
                           certificate=dict(reeval_status=c.get("status"), reeval_passed=c.get("passed"),
                                            reeval_min_jd_lower_bound=c.get("min_jd_lower_bound"),
                                            original_passed=orig.get("cert_passed"),
                                            original_min_jd_lower_bound=orig.get("cert_min_jd_lower_bound")),
                           derived=done.get("derived"), input_record_pair_sha256=done.get("input_record_pair_sha256"),
                           stage_pair_sha256=done.get("stage_pair_sha256"),
                           evaluated_d_sha256_matches_derived=done.get("evaluated_d_sha256_matches_derived"))
                rows.append(row)
        doc = dict(schema=SCHEMA, code_sha=self.code_sha, code=str(self.code), stage=self.o.stage, pairs=pairs,
                   arms=self.arms, routes={a: ARMS[a]["route"] for a in self.arms},
                   originals={a: dict(path=originals[a]["path"], sha256=originals[a]["sha256"]) for a in self.arms},
                   counts=counts, rows=rows)
        text = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False)+"\n"
        stored = read_json(self.out/"compare.json")
        if stored is not None and stored == json.loads(text):
            return self.out/"compare.json", "unchanged"
        base = "compare" if stored is None else f"compare_{stamp()}"
        with open(self.out/f"{base}.json", "x", encoding="utf-8") as stream:
            stream.write(text)
        with open(self.out/f"{base}.md", "x", encoding="utf-8") as stream:
            stream.write(markdown(doc))
        return self.out/f"{base}.json", "written"

    def main(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.arms = [a.strip() for a in self.o.arms.split(",") if a.strip()]
        try:
            unknown = sorted(set(self.arms)-set(ARMS))
            if unknown:
                raise Stop(f"--arms: unknown {unknown}")
            self.log(f"start: out {self.out}; arms {self.arms}; stage {self.o.stage}; data root {self.o.data_root}")
            self.prepare_code()
            if not self.o.allowlist:
                self.o.allowlist = str(self.code/"configs"/"bayesreg"/"dev_subjects.json")
            pairs = self.pairs()
            originals = {}
            for arm in self.arms:
                try:
                    originals[arm] = summarize.reference(getattr(self.o, ARMS[arm]["long_option"]), arm, pairs)
                except summarize.SummaryError as exc:
                    raise Stop(str(exc)) from None
            failed = []
            for arm in self.arms:
                for pair in pairs:
                    if self.one(arm, pair, originals[arm]) == "failed":
                        failed.append(f"{arm} {summarize.short_name(pair)}")
            path, state = self.compare(pairs, originals)
            self.log(f"comparison: {path} ({state}); failed: {failed or 'none'}")
            return 8 if failed else 0
        except Stop as exc:
            self.log(f"STOP: {exc}")
            return 8


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else str(value)
    return str(value)


def markdown(doc):
    c = doc["counts"]
    out = ["# A0、A4′ 的 bayesreg.eval.cli 重评与原数字（experiments/d0/evaluate.py）对照", "",
           f"代码 {doc['code_sha']}（{doc['code']}）；stage {doc['stage']}；路线 {doc['routes']}。开发集，只读重评，"
           "不是新的配准。", "",
           f"逐项比较 {c['compared']} 个：逐位相同 {c['bitwise_equal']}，不同 {c['differ']}；缺重评 {c['missing']} 对。", "",
           "| 臂 | 对 | 路线 | field_role（重评） | " + " | ".join(f"{k} 重评 / 原 / 差" for k in MEASURES)
           + " | 证书（重评 / 原） |", "|"+"---|"*(5+len(MEASURES))]
    for r in doc["rows"]:
        if not r["done"]:
            out.append(f"| {r['arm']} | {summarize.short_name(r['pair'])} | {r['route']} | 缺 |" + " — |"*(len(MEASURES)+1))
            continue
        cells = []
        for k in MEASURES:
            x = r["measures"][k]
            cells.append("逐位相同 " + fmt(x["reeval"]) if x["equal_bitwise"] else
                         f"{fmt(x['reeval'])} / {fmt(x['original'])} / {fmt(x['difference'])}")
        cert = r["certificate"]
        out.append(f"| {r['arm']} | {summarize.short_name(r['pair'])} | {r['route']} | {r['field_role']} | "
                   + " | ".join(cells) + f" | {cert['reeval_status']} {fmt(cert['reeval_min_jd_lower_bound'])} / "
                   f"{fmt(cert['original_passed'])} {fmt(cert['original_min_jd_lower_bound'])} |")
    out += ["", "说明：A4′ 走 disp_from_final_native 路线（见 experiments/m2v1/reeval_refs.py 文档串）：A4′ 运行记录的输入是"
            "上色后的 pair.npz，bayesreg.eval.cli 的输入核对按设计拒绝它；场取自运行的 final_native u（u_to_d，与配置 C "
            "路线相同），field_role 因此为 disp_only。差异只报告，用哪套由总负责人定。", ""]
    return "\n".join(out)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -B experiments/m2v1/reeval_refs.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--code-commit", default=CODE_COMMIT)
    parser.add_argument("--code", help="an existing code tree instead of the git archive (tests)")
    parser.add_argument("--d0-root", default=str(RESULTS/"d0_v1"))
    parser.add_argument("--a4-root", default=str(RESULTS/"d0_v1_a4"))
    parser.add_argument("--d0-long", default=str(summarize.D0_LONG))
    parser.add_argument("--a4-long", default=str(summarize.A4_LONG))
    parser.add_argument("--data-root", default=str(REPO/"data"/"three_d"))
    parser.add_argument("--stage", default=STAGE)
    parser.add_argument("--allowlist", default=None)
    parser.add_argument("--arms", default="A0,A4p")
    parser.add_argument("--min-mem-available-mib", type=int, default=4096)
    parser.add_argument("--wait-minutes", type=int, default=60)
    parser.add_argument("--numba-threads", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None):
    return Reeval(parse_args(argv)).main()


if __name__ == "__main__":
    sys.exit(main())
