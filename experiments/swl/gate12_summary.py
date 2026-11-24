"""C04-6 gate 12 summary of an ABBA series of fixed_state_memory.py runs (work package SWLM; reviews
reviews/SWLM_复核_20261101.md rounds 1-3; the criteria of contract C04-6 gate 12 as adopted 11-01 (d7c7476) with the
failed-run rule of cbc133f (R3-3) and the decision order of 577b49b (round 4) -- the contract text rules).

    python experiments/swl/gate12_summary.py <root> --pairs M [--out <root>/summary.json]

Reads <root>/brain-<i>.json and <root>/support_shared-<i>.json, i = 1..M, <root>/warmup.json (the uncounted warm-up
run: not compared, but its failure counts, R3-2) and <root>/runs.tsv when present (gate12_repeat.sh: tag, arm, exit
code, new OOM lines in dmesg per run).
Per run: the host peak = the largest hwm_mb of the memtrace segments ending at swl.* marks (contract: "主机峰值只取
swl.* 各段"; the channel preparation before swl.start is not counted), the device peak = the largest cuda_peak_mb of
the same segments.  Per pair (support_shared minus brain): host and device peak differences, the allocated-heap
differences per mark (malloc_live_mb = mallinfo2 uordblks + hblkhd).
Criteria (all three must hold; only pairs whose two runs both finished, exit 0 and no new OOM line):
  (a) allocated heap, every pair: difference <= v_bytes + 8 MiB at swl.level_start, swl.state, swl.g0, swl.eval (the
      four marks the contract names); swl.prepare (before the arms differ) is reported, with a warning above 8 MiB;
  (b) device, every pair: device peak difference <= 32 MiB;
  (c) host: median(host peaks of support_shared) - median(host peaks of brain) <= 64 MiB; undecided ("未决") when the
      range of brain's host peaks exceeds 128 MiB.
Verdict, in the contract's order (577b49b): a series not run with 4 threads (cpu_threads of the JSON files) ->
"只作描述" (failed runs named in the verdict and listed); (a) or (b) fails on the complete pairs -> "不通过（持有）"; a
failed run, the warm-up included (nonzero exit, a new OOM line, no JSON, no swl.* marks) -> "未决（运行失败）", with the
failures per arm as a difference (e.g. support_shared OOM, brain not) -- never skipped silently, never rerun; no failed
run but fewer than 5 complete pairs (R3-1: also when this script is called by hand) -> "只作描述"; (c) undecided ->
"未决（c）" (it takes precedence over a failed (c)); (c) fails -> "不通过（分配器滞留）"; else "通过".  So a series that
does not meet the design can only give "不通过（持有）", "未决" or "只作描述", never "通过" or "不通过（分配器滞留）".
Descriptive (contract): every pair's differences, each arm's host peak range and maximum, the number of pairs whose host
peak difference exceeds 64 MiB, the warm-up run (its status and swl.* peaks), smaps_rollup anonymous memory per mark
(smaps_anon_mb), page_settings, failures.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ARMS = ("brain", "support_shared")
HELD = ("swl.level_start", "swl.state", "swl.g0", "swl.eval")
PREPARE = "swl.prepare"
MARKS = (PREPARE, "swl.seed_evaluated")+HELD
MIB = 1024.*1024.
ALLOC_MARGIN_MIB = 8.
DEVICE_MIB = 32.
HOST_MIB = 64.
BRAIN_RANGE_MIB = 128.
CRITERION_THREADS = 4
CRITERION_PAIRS = 5
WARMUP = "warmup"


def _marks(run):
    return {r.get("label"): r for r in run.get("memtrace", [])}


def swl_peaks(run):
    """(host peak, device peak) in MiB over the memtrace segments ending at swl.* marks; None without such a mark."""
    lines = [r for r in run.get("memtrace", []) if str(r.get("label", "")).startswith("swl.")]
    if not lines:
        return None
    return max(r.get("hwm_mb") or 0. for r in lines), max(r.get("cuda_peak_mb") or 0. for r in lines)


def _failure(run, record):
    """None, or why a run failed: runs.tsv record (nonzero exit, new OOM lines), no JSON, no swl.* marks."""
    why = []
    if record is not None:
        if record.get("exit") not in (0, "0"):
            why.append(f"exit {record.get('exit')}")
        if int(record.get("oom_new") or 0) > 0:
            why.append(f"{record.get('oom_new')} new OOM lines")
    if run is None:
        why.append("no JSON")
    elif swl_peaks(run) is None or not all(k in run.get("malloc_live_mb", {}) for k in HELD):
        why.append("swl.* marks missing")
    return "; ".join(why) or None


def summarize(runs, pairs, failures=None, warmup=None):
    """runs: {(arm, i): parsed JSON or None}; failures: {"<tag>": {"exit": .., "oom_new": ..}} (runs.tsv, tags
    "<arm>-<i>" and "warmup") or None; warmup: the warm-up run's parsed JSON, None = missing (a failure, R3-2).
    Returns the summary dict (module docstring)."""
    failures = failures or {}
    failed = {f"{arm}-{i}": why for i in range(1, pairs+1) for arm in ARMS
              if (why := _failure(runs.get((arm, i)), failures.get(f"{arm}-{i}"))) is not None}
    warmup_failure = _failure(warmup, failures.get(WARMUP))
    if warmup_failure is not None:
        failed[WARMUP] = warmup_failure
    complete = [i for i in range(1, pairs+1) if f"brain-{i}" not in failed and f"support_shared-{i}" not in failed]
    threads = sorted({run.get("cpu_threads") for run in runs.values() if run is not None}, key=str)
    rows, fail_a, fail_b = [], [], []
    for i in complete:
        brain, shared = runs["brain", i], runs["support_shared", i]
        (host_b, dev_b), (host_s, dev_s) = swl_peaks(brain), swl_peaks(shared)
        v_mib = shared["v_bytes"]/MIB
        live_b, live_s = brain["malloc_live_mb"], shared["malloc_live_mb"]
        alloc = {label: live_s[label]-live_b[label] for label in MARKS if label in live_b and label in live_s}
        ok_a = all(alloc[k] <= v_mib+ALLOC_MARGIN_MIB for k in HELD)
        ok_b = dev_s-dev_b <= DEVICE_MIB
        mb, ms = _marks(brain), _marks(shared)
        smaps = {k: (ms.get(k, {}).get("smaps_anon_mb"), mb.get(k, {}).get("smaps_anon_mb")) for k in MARKS}
        rows.append(dict(pair=i, host_peak_brain_mb=host_b, host_peak_support_shared_mb=host_s,
                         host_peak_difference_mb=host_s-host_b, device_peak_difference_mb=dev_s-dev_b, v_mib=v_mib,
                         malloc_live_difference_mb=alloc,
                         smaps_anon_difference_mb={k: None if None in s else s[0]-s[1] for k, s in smaps.items()},
                         prepare_warning=PREPARE in alloc and abs(alloc[PREPARE]) > ALLOC_MARGIN_MIB,
                         criterion_a=ok_a, criterion_b=ok_b))
        if not ok_a:
            fail_a.append(i)
        if not ok_b:
            fail_b.append(i)
    warmup_peaks = None if warmup is None else swl_peaks(warmup)
    out = dict(pairs=pairs, complete_pairs=complete, cpu_threads=threads, failed_runs=failed,
               failure_difference={arm: sorted(k for k in failed if k.rsplit("-", 1)[0] == arm) for arm in ARMS},
               warmup=dict(failed=warmup_failure, arm=None if warmup is None else warmup.get("arm"),
                           host_peak_mb=None if warmup_peaks is None else warmup_peaks[0],
                           device_peak_mb=None if warmup_peaks is None else warmup_peaks[1]),
               per_pair=rows, page_settings={f"{arm}-{i}": runs[arm, i].get("page_settings")
                                             for i in complete for arm in ARMS})
    undecided = ok_c = None
    if rows:
        peaks = {"brain": [r["host_peak_brain_mb"] for r in rows],
                 "support_shared": [r["host_peak_support_shared_mb"] for r in rows]}
        medians = {arm: statistics.median(values) for arm, values in peaks.items()}
        ranges = {arm: max(values)-min(values) for arm, values in peaks.items()}
        median_difference = medians["support_shared"]-medians["brain"]
        undecided = ranges["brain"] > BRAIN_RANGE_MIB
        ok_c = median_difference <= HOST_MIB
        out.update(criterion_a=dict(passed=not fail_a, failed_pairs=fail_a, margin_mib=ALLOC_MARGIN_MIB, marks=HELD),
                   criterion_b=dict(passed=not fail_b, failed_pairs=fail_b, limit_mib=DEVICE_MIB),
                   criterion_c=dict(passed=ok_c and not undecided, undecided=undecided, limit_mib=HOST_MIB,
                                    median_difference_mb=median_difference, medians_mb=medians, ranges_mb=ranges,
                                    brain_range_limit_mib=BRAIN_RANGE_MIB, host_peaks_mb=peaks),
                   descriptive=dict(host_peak_max_mb={arm: max(values) for arm, values in peaks.items()},
                                    pairs_host_difference_over_64_mib=sum(r["host_peak_difference_mb"] > HOST_MIB
                                                                         for r in rows)))
    if threads and threads != [CRITERION_THREADS]:            # no run at all: threads unknown, a failure below
        verdict = f"只作描述（cpu_threads {threads}；判据只用 {CRITERION_THREADS} 线程的系列）"
        if failed:
            verdict += f"；有运行失败：{', '.join(sorted(failed))}"
    elif fail_a or fail_b:
        verdict = "不通过（持有）"
    elif failed:
        verdict = "未决（运行失败）"
    elif len(complete) < CRITERION_PAIRS:
        verdict = f"只作描述（完整对 {len(complete)}；判据只用至少 {CRITERION_PAIRS} 个完整对的系列）"
    elif undecided:
        verdict = "未决（c）"
    elif not ok_c:
        verdict = "不通过（分配器滞留）"
    else:
        verdict = "通过"
    out["verdict"] = verdict
    return out


def read_failures(path):
    """runs.tsv of gate12_repeat.sh: {tag: {"arm", "exit", "oom_new"}}; {} when absent."""
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            return {row["tag"]: row for row in csv.DictReader(handle, delimiter="\t")}
    except OSError:
        return {}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root")
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--out", default=None)
    ns = parser.parse_args(argv)
    root = Path(ns.root)
    runs = {}
    for i in range(1, ns.pairs+1):
        for arm in ARMS:
            try:
                runs[arm, i] = json.loads((root/f"{arm}-{i}.json").read_text())
            except (OSError, ValueError):
                runs[arm, i] = None
    try:
        warmup = json.loads((root/f"{WARMUP}.json").read_text())
    except (OSError, ValueError):
        warmup = None
    summary = summarize(runs, ns.pairs, read_failures(root/"runs.tsv") or None, warmup)
    out = Path(ns.out) if ns.out else root/"summary.json"
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False)+"\n")
    print(f"verdict: {summary['verdict']}")
    for key, why in summary["failed_runs"].items():
        print(f"failed run {key}: {why}")
    for row in summary["per_pair"]:
        live = ", ".join(f"{k} {v:+.1f}" for k, v in row["malloc_live_difference_mb"].items())
        print(f"pair {row['pair']}: host peak {row['host_peak_difference_mb']:+.1f} MiB, device peak "
              f"{row['device_peak_difference_mb']:+.1f} MiB, allocated ({live}; v {row['v_mib']:.1f}) "
              f"a={row['criterion_a']} b={row['criterion_b']}")
    if "criterion_c" in summary:
        c, d = summary["criterion_c"], summary["descriptive"]
        print(f"(c) median difference {c['median_difference_mb']:+.1f} MiB (limit {HOST_MIB:g}); ranges "
              f"{ {k: round(v, 1) for k, v in c['ranges_mb'].items()} }; undecided {c['undecided']}; maxima "
              f"{ {k: round(v, 1) for k, v in d['host_peak_max_mb'].items()} }; pairs over 64 MiB "
              f"{d['pairs_host_difference_over_64_mib']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
