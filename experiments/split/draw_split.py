#!/usr/bin/env python
"""B-line (project 2, bayesreg) subject split: the draw required by CLAUDE.md hard constraint 6.

Authorised by BZ 2026-10-30 about 4:25 am CDT (docs/decision_log.md, item (4): "现在做 B 线抽签，种子 20261030");
pool rule = scheme (alpha) adopted 2026-10-24 about 9 pm CDT (新项目规划_bayesreg_20261024.md section 11.2).
Documentation: docs/bayesreg/06_数据划分.md.

Reads subject ID LISTS only (no image, label or result file).  Writes a new directory and refuses to write into an
existing non-empty one: the split is frozen, files under splits/ are append-only.

The draw (every rng call, in this order, on one numpy.random.default_rng(SEED)):
  1. pool  = sorted(206 canonical ids);           p1 = rng.permutation(206); shuffled = pool[p1]
     holdout_B = shuffled[:100], paired in shuffled order: (shuffled[2k], shuffled[2k+1]) = (fixed, moving), k < 50
     train     = shuffled[100:]  (106 subjects)
  2. tr    = sorted(train);                       p2 = rng.permutation(106); fold A = tr[p2[:53]], fold B = tr[p2[53:]]
  3. a     = sorted(fold A);                      p3 = rng.permutation(53);  calib A = a[p3[:8]], train-role A = a[p3[8:]]
  4. b     = sorted(fold B);                      p4 = rng.permutation(53);  calib B = b[p4[:8]], train-role B = b[p4[8:]]
  5. a (as in 3);                                 p5 = rng.permutation(53);  EB A = a[p5[:30]], pairs (2k, 2k+1), k < 15
  6. b (as in 4);                                 p6 = rng.permutation(53);  EB B = b[p6[:30]], pairs (2k, 2k+1), k < 15
EB pairs are drawn from the whole fold (both roles): a fold-A pair is segmented by the fold-B model, whose training and
calibration lists contain no fold-A subject (contract D24, pair level).

Usage (from the repository root):
  python -I -B experiments/split/draw_split.py                 # writes splits/bayesreg_v1/
  python -I -B experiments/split/draw_split.py --out DIR       # e.g. a scratch directory, to check reproducibility
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SEED = 20261030
N_POOL = 206
N_HOLDOUT = 100
N_TRAIN = 106
N_FOLD = 53
N_CALIB = 8
N_EB_PER_FOLD = 15
L2R_TRAIN_LINES = 394          # Learn2Reg OASIS training cases = lines 1-394 of subjects.txt (inventory H section 1)

REPO = Path(__file__).resolve().parents[2]
OASIS_ROOT = Path("/mnt/d/data")
ID_RE = re.compile(r"OASIS_OAS1_[0-9]{4}_MR1")   # same pattern as bayesreg/io/guard.py
HOLDOUT_PLAN = "archive/seq_vp_3d/results/verification/3d_stage_HOLDOUT_v1/plan"

INPUTS = {
    "all_subjects": OASIS_ROOT / "subjects.txt",
    "dev_L2R_val": REPO / "data/three_d/stage_L2R_val_v1/selection.json",
    "T0_tune": REPO / "data/three_d/stage_T0_tune_v1/selection.json",
    "E11_native": REPO / "data/three_d/stage_E11_native_v1/selection.json",
    "H1_pairs": REPO / HOLDOUT_PLAN / "oasis_heldout_pairs_draft.tsv",
    "H4_subjects": REPO / HOLDOUT_PLAN / "h4_subjects_20261020.txt",
    "H4_pool": REPO / HOLDOUT_PLAN / "h4_pool_volumes_20261020.tsv",
    "used265_crosscheck": REPO / HOLDOUT_PLAN / "h4_used_subjects_20261020.txt",
    "similar14_source": REPO / "docs/plan/inventory_20261024/H_训练受试者与工具可得性.md",
}

# The 14 subjects whose scan correlates >= .98 with a Learn2Reg OASIS test image.  l2r_test_scan.json is not in archive/
# (it stays in seq_vp_3d/results, which this repository does not touch), so the list is the one written in
# docs/plan/inventory_20261024/H_训练受试者与工具可得性.md section 3, "注意 1".
SIMILAR14 = ["0035", "0052", "0062", "0120", "0124", "0138", "0143", "0166", "0220", "0240", "0255", "0288", "0428",
             "0430"]

# Cross-check only: the 111 "remaining" subjects as listed in inventory H section 3.
REMAINING111_SPEC = ("0022 0030 0035 0039-0040 0043-0047 0049-0088 0090-0092 0094-0099 0101-0117 0119-0127 0129-0139 "
                     "0143 0166 0220 0240 0244 0255-0256 0262 0288 0356 0384 0396 0428 0430 0437")


def oid(num):
    return "OASIS_OAS1_%s_MR1" % num


def expand_spec(spec):
    out = []
    for tok in spec.split():
        if "-" in tok:
            lo, hi = tok.split("-")
            out.extend("%04d" % k for k in range(int(lo), int(hi) + 1))
        else:
            out.append(tok)
    return [oid(n) for n in out]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_ids(ids, name):
    bad = [s for s in ids if not ID_RE.fullmatch(s)]
    if bad:
        sys.exit("%s: %d non-canonical ids, e.g. %r" % (name, len(bad), bad[:3]))
    if len(set(ids)) != len(ids):
        sys.exit("%s: duplicate ids" % name)
    return set(ids)


def read_all_subjects(path):
    ids = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        m = ID_RE.findall(line)
        if len(m) != 1:
            sys.exit("subjects.txt: line without exactly one OASIS id: %r" % line)
        ids.append(m[0])
    return ids


def pair_subjects(sel, name):
    ids = set()
    for p in sel["pairs"]:
        f, m = os.path.basename(p["fixed"]), os.path.basename(p["moving"])
        if p["pair"] != f + "__" + m:
            sys.exit("%s: pair name %r does not match fixed/moving" % (name, p["pair"]))
        ids.update((f, m))
    return ids


def read_tsv_columns(path, cols):
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    header = lines[0].split("\t")
    idx = [header.index(c) for c in cols]
    out = []
    for l in lines[1:]:
        parts = l.split("\t")
        out.extend(parts[i] for i in idx)      # only the id columns are kept
    return out


def read_id_lines(path):
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def build_pool():
    for k, p in INPUTS.items():
        if not p.is_file():
            sys.exit("input %s missing: %s" % (k, p))
    all_list = read_all_subjects(INPUTS["all_subjects"])
    all_ids = check_ids(all_list, "all")
    line_of = {s: i + 1 for i, s in enumerate(all_list)}

    dev = pair_subjects(json.loads(INPUTS["dev_L2R_val"].read_text()), "dev")
    t0sel = json.loads(INPUTS["T0_tune"].read_text())
    t0 = check_ids(t0sel["reserved_subjects_never_for_training"], "T0")
    if t0 != pair_subjects(t0sel, "T0"):
        sys.exit("T0: reserved list differs from the pair subjects")
    e11 = pair_subjects(json.loads(INPUTS["E11_native"].read_text()), "E11")
    h1_list = read_tsv_columns(INPUTS["H1_pairs"], ["fixed", "moving"])
    h1 = check_ids(h1_list, "H1")
    h4 = check_ids(read_id_lines(INPUTS["H4_subjects"]), "H4")
    pool149 = check_ids(read_tsv_columns(INPUTS["H4_pool"], ["subject"]), "H4_pool")
    used265 = check_ids([oid(l.split("\t")[0]) for l in read_id_lines(INPUTS["used265_crosscheck"])], "used265")
    sim = check_ids([oid(n) for n in SIMILAR14], "similar14")
    rem_inv = check_ids(expand_spec(REMAINING111_SPEC), "remaining111_inventory")

    counts = {"all": len(all_ids), "dev_L2R_val": len(dev), "T0": len(t0), "E11": len(e11), "H1": len(h1),
              "H4": len(h4), "H4_pool": len(pool149), "similar14": len(sim)}
    expected = {"all": 414, "dev_L2R_val": 20, "T0": 24, "E11": 10, "H1": 100, "H4": 40, "H4_pool": 149,
                "similar14": 14}
    if counts != expected:
        sys.exit("set sizes differ from the expected ones: %r vs %r" % (counts, expected))
    named = {"dev_L2R_val": dev, "T0": t0, "E11": e11, "H1": h1, "H4_pool": pool149, "similar14": sim}
    for k, s in named.items():
        if not s <= all_ids:
            sys.exit("%s contains ids outside subjects.txt" % k)
    keys = list(named)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if named[keys[i]] & named[keys[j]]:
                sys.exit("%s and %s overlap" % (keys[i], keys[j]))
    if not h4 <= pool149:
        sys.exit("H4 is not inside the H4 pool")

    remaining111 = all_ids - dev - t0 - e11 - h1 - pool149
    released109 = pool149 - h4
    pool = all_ids - dev - t0 - e11 - h1 - h4 - sim
    checks = {
        "remaining_111": len(remaining111) == 111,
        "remaining_equals_inventory_H_list": remaining111 == rem_inv,
        "similar14_inside_remaining": sim <= remaining111,
        "released_109": len(released109) == 109,
        "pool_equals_released109_plus_remaining_minus_similar": pool == released109 | (remaining111 - sim),
        "used265_equals_all_minus_H4_pool": used265 == all_ids - pool149,
        "similar14_inside_used265": sim <= used265,
        "pool_206": len(pool) == N_POOL,
    }
    if not all(checks.values()):
        sys.exit("pool cross-checks failed: %r" % {k: v for k, v in checks.items() if not v})

    names = set(os.listdir(OASIS_ROOT))        # directory names only
    missing = sorted(s for s in pool if s not in names or not (OASIS_ROOT / s).is_dir())
    if missing:
        sys.exit("%d pool subjects have no directory under %s" % (len(missing), OASIS_ROOT))
    checks["pool_directories_present"] = True

    l2r_train = sum(line_of[s] <= L2R_TRAIN_LINES for s in pool)
    info = {"set_sizes": counts, "remaining_111": len(remaining111), "released_109": len(released109),
            "pool": len(pool), "pool_from_released_109": len(pool & released109),
            "pool_from_remaining_minus_similar": len(pool & (remaining111 - sim)),
            "pool_in_L2R_training_lines_1_394": l2r_train, "checks": checks}
    return sorted(pool), released109, info


def draw(pool_sorted):
    rng = np.random.default_rng(SEED)
    pool = np.array(pool_sorted)
    shuffled = pool[rng.permutation(N_POOL)]                                       # call 1
    holdout_order = [str(s) for s in shuffled[:N_HOLDOUT]]
    holdout_pairs = [(holdout_order[2 * k], holdout_order[2 * k + 1]) for k in range(N_HOLDOUT // 2)]
    train = sorted(str(s) for s in shuffled[N_HOLDOUT:])

    tr = np.array(train)
    p2 = rng.permutation(N_TRAIN)                                                  # call 2
    fold = {"A": sorted(str(s) for s in tr[p2[:N_FOLD]]), "B": sorted(str(s) for s in tr[p2[N_FOLD:]])}

    role = {}
    for f in ("A", "B"):                                                           # calls 3, 4
        arr = np.array(fold[f])
        p = rng.permutation(N_FOLD)
        for i, s in enumerate(arr[p]):
            role[str(s)] = "calib" if i < N_CALIB else "train"

    eb_pairs = []
    for f in ("A", "B"):                                                           # calls 5, 6
        arr = np.array(fold[f])
        pick = [str(s) for s in arr[rng.permutation(N_FOLD)][:2 * N_EB_PER_FOLD]]
        eb_pairs.extend((pick[2 * k], pick[2 * k + 1], f) for k in range(N_EB_PER_FOLD))

    # Post-conditions.
    hold = set(holdout_order)
    assert len(hold) == N_HOLDOUT and len(set(train)) == N_TRAIN and not hold & set(train)
    assert hold | set(train) == set(pool_sorted)
    assert len(fold["A"]) == len(fold["B"]) == N_FOLD and not set(fold["A"]) & set(fold["B"])
    assert set(fold["A"]) | set(fold["B"]) == set(train)
    for f in ("A", "B"):
        assert sum(role[s] == "calib" for s in fold[f]) == N_CALIB
    eb_subj = [s for p in eb_pairs for s in p[:2]]
    assert len(eb_subj) == len(set(eb_subj)) == 4 * N_EB_PER_FOLD
    assert all(p[0] in fold[p[2]] and p[1] in fold[p[2]] for p in eb_pairs)
    return holdout_pairs, train, fold, role, eb_pairs


def readme_text(info):
    return """# splits/bayesreg_v1 — B 线受试者划分（冻结）

由 `experiments/split/draw_split.py` 生成（种子 {seed}），规则、来源、核对与披露见 `docs/bayesreg/06_数据划分.md`。
本目录**只能新增，不能修改**（CLAUDE.md git 规则：冻结文件只能新增）；每个文件的 sha256 在 SHA256SUMS。

| 文件 | 内容 |
|---|---|
| pool_206.txt | B 线候选池，{pool} 人，按规范 id 排序 |
| holdout_B_subjects.txt | B 线留出集，{nh} 人，按 id 排序。**只由 BZ 解封** |
| holdout_B_pairs.tsv | B 线留出集 {np} 对，列 fixed、moving，按抽签顺序；配对名 `<fixed>__<moving>` |
| train_subjects.txt | 训练集，{nt} 人，按 id 排序 |
| folds.json | 训练集每人的折（A/B）与角色（train = 该折训练子集；calib = 该折标定子集，用于早停与温度标定） |
| eb_pairs.tsv | 群体经验贝叶斯训练对 {neb} 对，列 fixed、moving、fold；同折配对，受试者不重复 |
| draw_log.json | 种子、numpy/Python 版本、输入文件路径与 sha256、各集合人数与核对结果、脚本 sha256、UTC 时间 |

使用规则（摘要；以 06_数据划分.md 与 CLAUDE.md 硬约束 6 为准）：
- 留出集与 H1 同等对待：不用于任何开发（调参、冒烟、诊断、分割、看结果）；只由 BZ 解封；解封后每对每方法一次，全部报告。
- 训练集只用于结构层分割器训练（按折、标定子集只作早停与温度标定）与群体经验贝叶斯（eb_pairs.tsv，用另一折训练的分割器预测）。
- T0 的 24 人不在本划分里，永不用于训练任何学习部件。

复现：`python -I -B experiments/split/draw_split.py --out <新目录>`，除 draw_log.json 的 utc_time 与 SHA256SUMS 中
draw_log.json 那一行外，输出应逐字节相同（numpy {npv}；numpy 的 Generator 不保证跨版本流兼容，以本目录文件为准）。
""".format(seed=SEED, pool=info["pool"], nh=N_HOLDOUT, np=N_HOLDOUT // 2, nt=N_TRAIN, neb=2 * N_EB_PER_FOLD,
           npv=np.__version__)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "splits" / "bayesreg_v1"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit("refusing to write into non-empty %s (frozen split; files are append-only)" % out)

    pool, released109, info = build_pool()
    holdout_pairs, train, fold, role, eb_pairs = draw(pool)
    holdout = sorted(s for p in holdout_pairs for s in p)
    info["drawn"] = {
        "holdout_B_subjects": len(holdout), "holdout_B_pairs": len(holdout_pairs),
        "holdout_B_from_released_109": len(set(holdout) & released109),
        "train": len(train), "train_from_released_109": len(set(train) & released109),
        "fold_A": len(fold["A"]), "fold_B": len(fold["B"]),
        "calib_A": sum(role[s] == "calib" for s in fold["A"]), "calib_B": sum(role[s] == "calib" for s in fold["B"]),
        "eb_pairs_A": sum(p[2] == "A" for p in eb_pairs), "eb_pairs_B": sum(p[2] == "B" for p in eb_pairs),
        "eb_pairs_with_calib_subject": sum(role[p[0]] == "calib" or role[p[1]] == "calib" for p in eb_pairs),
    }

    out.mkdir(parents=True, exist_ok=True)

    def write(name, text):
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    write("pool_206.txt", "".join(s + "\n" for s in pool))
    write("holdout_B_subjects.txt", "".join(s + "\n" for s in holdout))
    write("holdout_B_pairs.tsv", "fixed\tmoving\n" + "".join("%s\t%s\n" % p for p in holdout_pairs))
    write("train_subjects.txt", "".join(s + "\n" for s in train))
    folds = {"seed": SEED, "rule": "2 folds of 53; per fold 8 calib (early stopping, temperature) + 45 train",
             "subjects": {s: {"fold": f, "role": role[s]} for f in ("A", "B") for s in fold[f]}}
    folds["subjects"] = dict(sorted(folds["subjects"].items()))
    write("folds.json", json.dumps(folds, indent=1, ensure_ascii=False) + "\n")
    write("eb_pairs.tsv", "fixed\tmoving\tfold\n" + "".join("%s\t%s\t%s\n" % p for p in eb_pairs))
    write("README.md", readme_text(info))

    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        head = None
    log = {
        "purpose": "B-line split (CLAUDE.md hard constraint 6); BZ authorisation 2026-10-30 ~4:25 am CDT",
        "seed": SEED,
        "rng": "numpy.random.default_rng(seed); calls in order: permutation(206), permutation(106), permutation(53) x4",
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "script": str(Path(__file__).resolve().relative_to(REPO)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "git_head_at_draw": head,
        "inputs": {k: {"path": str(p), "resolved": os.path.realpath(p), "sha256": sha256_file(p)}
                   for k, p in INPUTS.items()},
        "similar14_note": "l2r_test_scan.json is not in archive/; the 14 ids are those of inventory H section 3",
        "counts": info,
        "utc_time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    write("draw_log.json", json.dumps(log, indent=1, ensure_ascii=False) + "\n")

    files = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    write("SHA256SUMS", "".join("%s  %s\n" % (sha256_file(out / n), n) for n in files))
    print("wrote %d files to %s; pool %d, holdout %d (%d pairs), train %d (A %d, B %d), eb pairs %d" % (
        len(files) + 1, out, len(pool), len(holdout), len(holdout_pairs), len(train), len(fold["A"]),
        len(fold["B"]), len(eb_pairs)))


if __name__ == "__main__":
    main()
