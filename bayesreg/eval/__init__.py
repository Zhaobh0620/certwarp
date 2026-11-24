"""Unified evaluation of the B line (work package EV): every method's displacement field is scored by the same code.

  fields     run directories and fields (disp.npz:d; bayesreg / configuration C final_native u), conversions, C16 checks
  metrics    official Learn2Reg metrics (bitwise as scripts/evaluate_l2r_style.py), seg4/seg35, brain log J measures
  cert_eval  the C07 certificate of a field: full or jacobian_only, cell counts over brain and background
  truth      the subject guard (sealed lists, development allowlist) and the truth labels behind it
  cli        python -m bayesreg.eval.cli RUN_DIR... --pairs-stage STAGE --out metrics.json --long long.csv --out-dir DIR
             (finished runs only; field_role native_formal / handover_only / ... on every record and long row)
  paired     paired comparisons (mean / median difference, positives, Wilcoxon + Holm, subject-block bootstrap) over
             the native_formal and disp_only rows by default; non-finite values listed and counted
Submodules are imported explicitly (this package imports nothing, so the cli can set thread variables first).
"""
