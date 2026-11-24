"""V1 revision 1, I4: the grid spacing h recorded in the B-line hold-out evaluation records, as float.hex values and counts.

Narrow read-only check authorised by the lead (decision_log 2026-11-22, the line after V1 review 0/4/7; ruling I4): the
hold-out is unsealed and its result is final; this read checks a reported result, it is not development.
  - Files: the evaluation records, path written from the pattern recorded in runs/20261119-hb-result/result.md line 9
    ("…/holdout_B_v1/eval/v1/pairs/*.json", "…" = /home/rmcfz/seq_vp_tag_results as on its line 8); one directory level,
    no recursion.  No image, label, field, cache or other file is opened.
  - Fields: only the spacing fields: grid.spacing_xyz; arms.method.units.spacing_xyz;
    arms.method.decision_certificate.final.grid.spacing_xyz; arms.method.field.disp.meta.grid.spacing_xyz
    (bayesreg/eval/holdout.py at cc2e06ef: evaluate_pair 2694, score_d 2474, decision_certificate 2514, fields.load_field).
  - Output: the number of records, and per field the distinct float.hex triples with their counts.  No path, subject id,
    pair id or any other value is printed; an error prints only its type.
Usage: python -B v1_oasis_spacing.py
"""
import glob
import json
import sys
from collections import Counter

PATTERN = "/home/rmcfz/seq_vp_tag_results/holdout_B_v1/eval/v1/pairs/*.json"
FIELDS = (("grid.spacing_xyz", ("grid", "spacing_xyz")),
          ("arms.method.units.spacing_xyz", ("arms", "method", "units", "spacing_xyz")),
          ("arms.method.decision_certificate.final.grid.spacing_xyz", ("arms", "method", "decision_certificate", "final", "grid", "spacing_xyz")),
          ("arms.method.field.disp.meta.grid.spacing_xyz", ("arms", "method", "field", "disp", "meta", "grid", "spacing_xyz")))


def take(doc, keys):
    for k in keys:
        if not isinstance(doc, dict) or k not in doc:
            return None
        doc = doc[k]
    return doc


def main():
    files = sorted(glob.glob(PATTERN))
    print(f"# records matched: {len(files)}")
    if not files:
        print("# no record at the recorded path: stop, hand back (no search)")
        return 2
    counts = {name: Counter() for name, _ in FIELDS}
    errors = Counter()
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as exc:                     # noqa: BLE001 - only the type is printed
            errors[type(exc).__name__] += 1
            continue
        for name, keys in FIELDS:
            v = take(doc, keys)
            if v is None:
                counts[name]["<missing>"] += 1
            elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                counts[name][tuple(float(x).hex() for x in v)] += 1
            else:
                counts[name]["<not a list of numbers>"] += 1
        del doc
    for name, _ in FIELDS:
        print(f"{name}: " + "; ".join(f"{k} x {c}" for k, c in sorted(counts[name].items(), key=str)))
    if errors:
        print("# read errors (type x count):", dict(errors))
    ok = all(set(counts[n]) == {("0x1.0000000000000p+0",)*3} for n, _ in FIELDS) and not errors
    print(f"# every record, every field: spacing exactly 1.0 on all three axes: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
