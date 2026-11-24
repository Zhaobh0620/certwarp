"""SEGP P4: two small checks read from the labels and from the follow-up (runs/20261105-segprobe/result.md). Read only.

  native     the share of each seg4 class in the scanner-space seg4.nii.gz against aligned_seg4.nii.gz, all 20: is the thin cortex of the truth of 0440 there before the alignment, or did the resampling make it?
  tracking   across the other 19, how the thickness proxy of the v1 prediction follows that of the truth (line and Spearman); the line at the value of 0440 against the prediction's actual value. A prediction that only regressed to the mean
             would have a small slope across the 19.

    python -B experiments/segprobe/p4_native_check.py --followup DIR/followup.json --out DIR/native_check.json
"""
import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segprobe_common as common  # noqa: E402
from segprobe_common import eval_seg  # noqa: E402
from p4_0440 import TARGET  # noqa: E402


def shares(labels):
    counts = np.bincount(np.asarray(labels).astype(np.int64).ravel(), minlength=5).astype(np.float64)
    return {str(k): float(counts[k]/counts[1:].sum()) for k in range(1, 5)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--followup", type=Path, default=common.OUT_ROOT/"p4_followup/followup.json")
    parser.add_argument("--out", type=Path, default=common.OUT_ROOT/"p4_followup/native_check.json")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; results are never overwritten")
    subjects, _ = common.admit_dev_subjects()
    table = {}
    for subject in subjects:
        native = nib.load(str(Path(eval_seg.DATA_ROOT)/subject/"seg4.nii.gz"))
        aligned, _ = eval_seg.read_truth(eval_seg.DATA_ROOT, subject, "seg4")
        table[subject] = dict(native=shares(np.asanyarray(native.dataobj)), aligned=shares(aligned))
    others = [s for s in subjects if s != TARGET]
    follow = json.loads(args.followup.read_text())["rows"]
    truth, pred = (np.array([follow[s]["surfaces"][k] for s in others]) for k in ("thickness_truth", "thickness_pred"))
    slope, intercept = np.polyfit(truth, pred, 1)
    rho, p = scipy_stats.spearmanr(truth, pred)
    at = follow[TARGET]["surfaces"]["thickness_truth"]
    tracking = dict(slope=float(slope), intercept=float(intercept), spearman=float(rho), p=float(p), residual_sd=float(np.std(pred-(slope*truth+intercept), ddof=2)), truth_0440=at, pred_0440=follow[TARGET]["surfaces"]["thickness_pred"], line_at_0440=float(slope*at+intercept),
                    differences_pred_minus_truth=dict(others_min=float((pred-truth).min()), others_max=float((pred-truth).max()), others_median=float(np.median(pred-truth)), value_0440=follow[TARGET]["surfaces"]["thickness_pred"]-at))
    document = dict(subjects=subjects, seg4_share_of_foreground=table, tracking=tracking, note="开发集 20 人，只读。class 1 cortex，2 subcortical，3 WM，4 CSF。")
    common.write_json(args.out, document)
    for k in ("1", "3"):
        print(f"class {k}: 0440 native {table[TARGET]['native'][k]:.4f}, aligned {table[TARGET]['aligned'][k]:.4f}; others native median {np.median([table[s]['native'][k] for s in others]):.4f}, range {min(table[s]['native'][k] for s in others):.4f} to {max(table[s]['native'][k] for s in others):.4f}; "
              f"aligned median {np.median([table[s]['aligned'][k] for s in others]):.4f}; largest |native - aligned| over the 20 {max(abs(table[s]['native'][k]-table[s]['aligned'][k]) for s in subjects):.4f}")
    print(json.dumps(tracking, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
