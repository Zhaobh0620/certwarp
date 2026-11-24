"""Write splits/bayesreg_v1/folds_full_v2.json from splits/bayesreg_v1/folds.json (work package UNV2, runs/20261103-unv2/任务书.md).

Rule (bayesreg.seg.unet.data.derive_full_folds): the calib of fold X is the calib of fold X in folds.json; the train of fold X is
every other of the 106 training subjects. Only folds.json is read: no other list, no image, no label. The output is never
overwritten (splits/ is add-only); --check compares an existing file with the rule's bytes instead of writing.

    python -B experiments/unv2/export_folds.py [--folds FILE] [--out FILE] [--check]
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bayesreg.seg.unet.data import full_folds_document, load_folds, parse_full_folds, render_full_folds  # noqa: E402

SOURCE = "splits/bayesreg_v1/folds.json"
OUT = "splits/bayesreg_v1/folds_full_v2.json"


def rendered(folds_path):
    """The bytes of the output file for this folds.json, parsed once as a check of the rule's own result."""
    document = full_folds_document(load_folds(folds_path), source=SOURCE)
    parse_full_folds(document)
    return render_full_folds(document)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=Path, default=ROOT/SOURCE)
    parser.add_argument("--out", type=Path, default=ROOT/OUT)
    parser.add_argument("--check", action="store_true", help="compare the existing output with the rule; write nothing")
    args = parser.parse_args(argv)
    raw = rendered(args.folds)
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != raw:
            print(f"{args.out}: differs from the rule's output for {args.folds}")
            return 1
        print(f"{args.out}: equal to the rule's output for {args.folds}")
        return 0
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; splits/ is add-only, never overwritten")
    args.out.write_bytes(raw)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
