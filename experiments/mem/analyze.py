"""Summaries of the MEM measurements (reads a measurement directory of experiments/mem/measure.sh; writes only --json).

    python -B experiments/mem/analyze.py DIR [--json OUT] [--phases]          one measurement
    python -B experiments/mem/analyze.py DIR --compare DIR2 [--prefix] [--json OUT]   bitwise comparison of two runs
(DIR may also be a run_pair directory itself, e.g. m2_gpu_smoke_v2/H2; --prefix compares the common log prefix before
the first wall-budget row of either run)

One measurement: the side-channel maxima (samples.csv: cgroup memory.peak, memory.swap.peak, VmHWM, RssAnon, GPU
used), the run's status and timing, and from memtrace.jsonl per phase (every mark outside run_pg, in order) the RSS,
the host peak since the previous mark (hwm_mb under BAYESREG_MEMTRACE_RESET_HWM=1), the cgroup memory, the device memory
and its peak since the previous mark, the malloc arena figures; for run_pg per (level, label) the largest of those.
Comparison: the log.jsonl rows of both runs with every key removed whose name contains one of TIME_KEYS -- "seconds",
"elapsed", "_unix", "time", "budget" (the wall-budget records), "started" -- at any depth, compared row by row; the
accepted state_hash sequences; the exported cert.json state_hash; status.json status.  Prints the first difference, if
any.  The verdict field is "identical" for a whole-run comparison (rows, accepted hashes and the
exported state) and "prefix_identical" with --prefix (rows and accepted hashes of the common prefix only; the exports
are reported in final_identical, not compared).  FUP1 (review MEM minor 5): the prefix verdict was called "identical"
before; mem_v1/M03/compare_H2_prefix.json was written with that name and means the prefix verdict.
"""
import argparse
import csv
import json
from pathlib import Path

TIME_KEYS = ("seconds", "elapsed", "_unix", "time", "budget", "started")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def samples_summary(path):
    path = Path(path)
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open()))
    out = dict(rows=len(rows), duration_s=number(rows[-1]["elapsed_s"]) if rows else None)
    for key in ("cg_mb", "cg_peak_mb", "cg_swap_mb", "cg_swap_peak_mb", "cg_anon_mb", "cg_file_mb", "cg_anon_thp_mb",
                "VmRSS_mb", "VmHWM_mb", "VmSwap_mb", "RssAnon_mb", "RssFile_mb", "gpu_used_mb"):
        values = [number(r.get(key)) for r in rows]
        values = [v for v in values if v is not None]
        out[f"max_{key}"] = max(values) if values else None
    if rows:
        top = max(rows, key=lambda r: number(r.get("cg_mb")) or 0.)
        out["at_max_cg"] = {k: top.get(k) for k in ("elapsed_s", "cg_mb", "cg_anon_mb", "cg_file_mb", "cg_anon_thp_mb",
                                                     "cg_swap_mb", "RssAnon_mb", "RssFile_mb")}
    return out


def locate_peaks(path, marks):
    """The samples with the largest cgroup memory, anon memory and swap, each with the last memtrace mark before it
    (UTC of the sample against the mark's time.time)."""
    import datetime
    rows = list(csv.DictReader(Path(path).open()))
    out = {}
    for key in ("cg_mb", "cg_anon_mb", "cg_swap_mb"):
        values = [(number(r.get(key)) or 0., r) for r in rows]
        if not values:
            continue
        value, row = max(values, key=lambda v: v[0])
        t = datetime.datetime.fromisoformat(row["utc"]).timestamp()
        before = [m for m in marks if m["t"] <= t]
        mark = before[-1] if before else None
        out[key] = dict(value=value, utc=row["utc"], elapsed_s=number(row["elapsed_s"]),
                        cg_mb=number(row.get("cg_mb")), cg_anon_mb=number(row.get("cg_anon_mb")),
                        cg_file_mb=number(row.get("cg_file_mb")), cg_swap_mb=number(row.get("cg_swap_mb")),
                        RssAnon_mb=number(row.get("RssAnon_mb")),
                        last_mark=None if mark is None else {k: mark.get(k) for k in ("seq", "label", "level", "round",
                                                                                     "iteration", "attempt")})
    return out


FIELDS = ("rss_mb", "hwm_mb", "rss_anon_mb", "rss_file_mb", "swap_mb", "cg_mb", "cg_peak_mb", "cg_swap_mb",
          "cg_anon_thp_mb", "malloc_inuse_mb", "malloc_free_mb", "malloc_mmap_mb", "malloc_arena_mb", "cuda_mb",
          "cuda_peak_mb", "cuda_reserved_mb")


def phases(marks):
    rows = []
    for m in marks:
        if m["label"].startswith("pg."):
            continue
        rows.append(dict(seq=m["seq"], label=m["label"], level=m.get("level"), round=m.get("round"),
                         t=round(m["mono"]-marks[0]["mono"], 2),
                         **{k: (round(m[k], 1) if isinstance(m.get(k), (int, float)) else None) for k in FIELDS}))
    return rows


def pg_summary(marks):
    groups = {}
    for m in marks:
        if not m["label"].startswith("pg."):
            continue
        key = (m.get("level"), m["label"])
        g = groups.setdefault(key, dict(level=m.get("level"), label=m["label"], count=0))
        g["count"] += 1
        for k in ("rss_mb", "hwm_mb", "cg_mb", "cg_swap_mb", "cuda_mb", "cuda_peak_mb", "malloc_free_mb"):
            v = m.get(k)
            if isinstance(v, (int, float)):
                g[f"max_{k}"] = round(max(g.get(f"max_{k}", v), v), 1)
    return sorted(groups.values(), key=lambda g: (g["level"] if g["level"] is not None else -1, g["label"]))


def summary(directory):
    directory = Path(directory)
    marks = load_jsonl(directory/"memtrace.jsonl") if (directory/"memtrace.jsonl").is_file() else []
    run = directory/"run"
    status = json.loads((run/"status.json").read_text()) if (run/"status.json").is_file() else None
    timing = json.loads((run/"timing.json").read_text()) if (run/"timing.json").is_file() else None
    out = dict(directory=str(directory), status=None if status is None else status.get("status"),
               exit_code=None if status is None else status.get("exit_code"),
               timing=None if timing is None else {k: timing.get(k) for k in ("load_pair", "channels_and_pyramid",
                                                                              "solve", "export", "total_seconds",
                                                                              "peak_rss_mb")},
               levels=None if timing is None else timing.get("solve_levels", {}).get("levels"),
               samples=samples_summary(directory/"samples.csv"), marks=len(marks))
    if marks and (directory/"samples.csv").is_file():
        out["peaks_located"] = locate_peaks(directory/"samples.csv", marks)
    if marks:
        out["max_hwm_lifetime_mb"] = max(m.get("hwm_lifetime_mb", 0.) for m in marks)
        out["max_cuda_peak_mb"] = max(m.get("cuda_peak_mb", 0.) for m in marks)
        top = max(marks, key=lambda m: m.get("hwm_mb", 0.))
        out["largest_segment_peak"] = dict(seq=top["seq"], label=top["label"], level=top.get("level"),
                                           hwm_mb=top.get("hwm_mb"))
        out["phases"] = phases(marks)
        out["pg"] = pg_summary(marks)
    return out


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if not any(t in k for t in TIME_KEYS)}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def run_dir(path):
    """A measurement directory's run/ (measure.sh), or a run_pair directory itself (e.g. the smoke's H2)."""
    path = Path(path)
    return path/"run" if (path/"run"/"log.jsonl").is_file() else path


def compare(a, b, prefix=False):
    """prefix=True: compare only the common prefix of the two logs (runs cut by a wall-clock budget at different
    points); the rows after it, the final export and the status are reported, not compared."""
    a, b = run_dir(a), run_dir(b)
    rows_a, rows_b = load_jsonl(a/"log.jsonl"), load_jsonl(b/"log.jsonl")
    if prefix:
        # the budget rows and the rows of the round that ended by the budget differ by design; stop before the first
        # row that mentions the budget in either log
        def cut(rows):
            for k, row in enumerate(rows):
                if row.get("kind") == "budget" or row.get("status") == "budget_exhausted":
                    return k
            return len(rows)
        n = min(cut(rows_a), cut(rows_b))
        rows_a, rows_b = rows_a[:n], rows_b[:n]
    out = dict(rows=[len(rows_a), len(rows_b)], prefix=prefix)
    first = None
    for k, (x, y) in enumerate(zip(rows_a, rows_b)):
        if strip(x) != strip(y):
            first = dict(row=k, a=strip(x), b=strip(y))
            break
    if first is None and len(rows_a) != len(rows_b):
        first = dict(row=min(len(rows_a), len(rows_b)), a="(end)" if len(rows_a) < len(rows_b) else "...",
                     b="(end)" if len(rows_b) < len(rows_a) else "...")
    out["first_difference"] = first
    hashes = [[r["state_hash"] for r in rows if r.get("kind") == "accept"] for rows in (rows_a, rows_b)]
    out["accepted_state_hashes"] = [len(h) for h in hashes]
    out["accepted_state_hashes_identical"] = hashes[0] == hashes[1]
    finals = []
    for run in (a, b):
        cert = run/"cert.json"
        finals.append(json.loads(cert.read_text())["state_hash"] if cert.is_file() else None)
    out["final_state_hash"] = finals
    out["final_identical"] = finals[0] is not None and finals[0] == finals[1]
    statuses = [json.loads((run/"status.json").read_text()).get("status") if (run/"status.json").is_file() else None
                for run in (a, b)]
    out["status"] = statuses
    verdict = first is None and out["accepted_state_hashes_identical"] and (prefix or out["final_identical"])
    out["prefix_identical" if prefix else "identical"] = verdict       # FUP1: the prefix verdict is named as such
    out["compared"] = ("common prefix before the first budget row of either log: rows and accepted state hashes; "
                       "exports not compared" if prefix else "whole runs: rows, accepted state hashes, exported state")
    out["removed_keys_containing"] = list(TIME_KEYS)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--prefix", action="store_true", help="--compare: the common prefix only (budget-cut runs)")
    parser.add_argument("--phases", action="store_true", help="print the phase table")
    args = parser.parse_args()
    if args.compare:
        result = compare(args.directory, args.compare, prefix=args.prefix)
        print(json.dumps({k: v for k, v in result.items() if k != "first_difference"}, indent=1))
        if result["first_difference"] is not None:
            print("first difference:", json.dumps(result["first_difference"])[:2000])
    else:
        result = summary(args.directory)
        print(json.dumps({k: v for k, v in result.items() if k not in ("phases", "pg", "samples")}, indent=1))
        print("samples:", json.dumps(result.get("samples")))
        if args.phases:
            for row in result.get("phases", []):
                print(row["seq"], row["label"], row["level"], row["round"], "t", row["t"], "rss", row["rss_mb"],
                      "hwm", row["hwm_mb"], "cg", row["cg_mb"], "swap", row["cg_swap_mb"], "thp", row["cg_anon_thp_mb"],
                      "mfree", row["malloc_free_mb"], "cuda", row["cuda_mb"], "cudapk", row["cuda_peak_mb"])
            for row in result.get("pg", []):
                print(row)
    if args.json:
        path = Path(args.json)
        if path.exists():
            raise SystemExit(f"{path}: exists (nothing is overwritten)")
        path.write_text(json.dumps(result, indent=1)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
