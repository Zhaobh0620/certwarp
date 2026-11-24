"""SEGP P4: how much of the gap of 0440 the block errors can carry (runs/20261105-segprobe/result.md). Arithmetic on the v1 prediction, all 20 development subjects; nothing is changed or written back: the labels are the same files, the counterfactual lives in this process.

  thick        the wrong voxels of v1 that survive an opening by a 3x3x3 cube count as right
  gaps         the voxels the truth calls background, v1 calls cortex (cerebral or cerebellar, seg4 class 1 or the labels 6, 25) and Atropos calls GM count as right

This is not a corrected truth and not a claim about what the labels should be: a model of the size of the effect, on development subjects.

    python -B experiments/segprobe/p4_leverage.py --out DIR/leverage.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segprobe_common as common  # noqa: E402
from segprobe_common import eval_seg  # noqa: E402
from jd_curl.three_d.nifti import load_scalar  # noqa: E402
from p4_0440 import CUBE, TARGET  # noqa: E402
from p4_followup import ATROPOS, GM  # noqa: E402

CEREBELLAR_CORTEX = (6, 25)
CORTEX = (2, 21)


def counted_right(top, truth, truth4, lookup, mask):
    """eval_seg.summarize_subject with the voxels of `mask` counted as right: the truth takes the predicted label there, in memory."""
    t35, t4 = truth.copy(), truth4.copy()
    t35[mask] = top[mask]
    t4[mask] = lookup[top[mask]]
    return eval_seg.summarize_subject(top, t35, t4, lookup)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=common.OUT_ROOT/"p4_followup/leverage.json")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; results are never overwritten")
    subjects, _ = common.admit_dev_subjects()
    lookup = eval_seg.load_seg4_lookup()
    rows = {}
    for subject in subjects:
        truth, _ = eval_seg.read_truth(eval_seg.DATA_ROOT, subject, "seg35")
        truth4, _ = eval_seg.read_truth(eval_seg.DATA_ROOT, subject, "seg4")
        raw, _ = common.load_raw(common.V1_ROOT, subject, expected_segmenter=common.V1_ID)
        top = common.argmax_of_mean([raw])
        del raw
        atropos = load_scalar(ATROPOS/subject/"labels.nii.gz", labels=True).values.astype(np.uint8)
        foreground = (top > 0) | (truth > 0)
        wrong = (top != truth) & foreground
        thick = ndimage.binary_opening(wrong, structure=CUBE)
        gaps = (truth == 0) & np.isin(top, CORTEX+CEREBELLAR_CORTEX) & (atropos == GM)
        base = eval_seg.summarize_subject(top, truth, truth4, lookup)
        with_thick, with_gaps = counted_right(top, truth, truth4, lookup, thick), counted_right(top, truth, truth4, lookup, gaps)
        rows[subject] = dict(n_wrong=int(wrong.sum()), n_thick=int(thick.sum()), n_gaps=int(gaps.sum()), n_truth_foreground=int((truth > 0).sum()), gaps_cerebellar=int((gaps & np.isin(top, CEREBELLAR_CORTEX)).sum()),
                             base=dict(seg35=base["seg35"], seg4=base["seg4"]), thick=dict(seg35=with_thick["seg35"], seg4=with_thick["seg4"]), gaps=dict(seg35=with_gaps["seg35"], seg4=with_gaps["seg4"]))
        r = rows[subject]
        print(f"{subject[11:15]}: base {r['base']['seg35']:.4f} / {r['base']['seg4']:.4f}; thick counted right ({r['n_thick']}) {r['thick']['seg35']:.4f} / {r['thick']['seg4']:.4f}; gaps counted right ({r['n_gaps']}, cerebellar {r['gaps_cerebellar']}) {r['gaps']['seg35']:.4f} / {r['gaps']['seg4']:.4f}", flush=True)
    others = [s for s in subjects if s != TARGET]

    def mean(key, metric):
        return float(np.mean([rows[s][key][metric] for s in others]))

    document = dict(subjects=subjects, rows=rows, others_mean={k: {m: mean(k, m) for m in ("seg35", "seg4")} for k in ("base", "thick", "gaps")}, note="〔模型〕开发集 20 人的算术，不是修正后的真值，也不是对标签应当如何的判断；数据与标签文件都没有改。")
    common.write_json(args.out, document)
    print(json.dumps(document["others_mean"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
