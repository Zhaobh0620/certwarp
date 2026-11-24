#!/bin/bash
# XB: the reviewed deletion step of the intermediates (prereg §5.3; the lead's ruling 2026-11-07: the driver deletes nothing, an audited step does), after runs/20261107-cleanup/delete_verified.sh.
#   bash experiments/xb/delete_verified.sh APPROVED_MANIFEST.tsv OUT.tsv
# The manifest: path<TAB>size_bytes<TAB>sha256 with a header line (the drivers' intermediates.tsv, gathered by `xb_chain.py collect-intermediates` and approved by the lead).
# Each file is deleted only if its path is canonical (no . or .. component, no empty component; `realpath -e` gives the same path back, so no component, the parent directories included, is a
# symbolic link: m5 of the XB review), lies under an XB working root, is a regular file, is not one of the kept files, and its sha256 equals the manifest's; every line is logged in OUT.tsv.
set -u
manifest="$1"; out="$2"
[ -e "$out" ] && { echo "$out exists: a deletion log is never overwritten" >&2; exit 2; }
roots="/home/rmcfz/seq_vp_tag_results/holdout_B_v1/work/ /home/rmcfz/seq_vp_tag_results/xb_dev_v1/work/ /home/rmcfz/seq_vp_tag_results/lpba40_test_v1/work/ants_s/ /home/rmcfz/seq_vp_tag_results/lpba40_test_v1/work/fireants_greedy/ /home/rmcfz/seq_vp_tag_results/lpba40_test_v1/work/fireants_syn/ /home/rmcfz/seq_vp_tag_results/lpba40_test_v1/work/convexadam_v4seg/"
# LP-F (decision_log 2026-11-20, the lead's ruling 2 (乙) and the handbook review's follow-up): only the four external arms' work directories of the LPBA40 formal root were added -- xb_core.LPBA_ARMS under hc.work_dir's <root>/work/<arm>/ (a test compares) --
# never the whole work/ (the method's and configuration C's cells there stay out of reach).  The feasibility root lpba40_feas_v1/ is not added: prereg §11 and C26.5-6 delete the external arms' large intermediates in the formal run only (§11's table is the formal run's; the feasibility pair, about 2.5 GB, keeps its files; C26.5-9 names no deletion).
keep=" disp.npz own_warped.npz own_warped_alt.npz method_out.json method_log.txt maps.npz spy_method.jsonl spy_policy.json "
printf 'path\tsize_bytes\tsha256\tresult\n' > "$out"
tail -n +2 "$manifest" | while IFS=$'\t' read -r path size sha rest; do
  case "$path/" in *"/../"*|*"/./"*|*"//"*) printf '%s\t%s\t%s\tREFUSED_not_canonical\n' "$path" "$size" "$sha" >> "$out"; continue;; esac
  inside=no
  for r in $roots; do case "$path" in "$r"*) inside=yes;; esac; done
  if [ "$inside" != yes ]; then printf '%s\t%s\t%s\tREFUSED_outside_root\n' "$path" "$size" "$sha" >> "$out"; continue; fi
  case "$keep" in *" $(basename -- "$path") "*) printf '%s\t%s\t%s\tREFUSED_kept_file\n' "$path" "$size" "$sha" >> "$out"; continue;; esac
  real=$(realpath -e -- "$path" 2>/dev/null) || { printf '%s\t%s\t%s\tMISSING_OR_LINK\n' "$path" "$size" "$sha" >> "$out"; continue; }
  if [ "$real" != "$path" ]; then printf '%s\t%s\t%s\tREFUSED_link_in_path_%s\n' "$path" "$size" "$sha" "$real" >> "$out"; continue; fi
  if [ -L "$path" ] || [ ! -f "$path" ]; then printf '%s\t%s\t%s\tMISSING_OR_LINK\n' "$path" "$size" "$sha" >> "$out"; continue; fi
  now=$(ionice -c3 nice -n 19 sha256sum -- "$path" | cut -d' ' -f1)
  if [ "$now" != "$sha" ]; then printf '%s\t%s\t%s\tSHA_MISMATCH_%s\n' "$path" "$size" "$sha" "$now" >> "$out"; continue; fi
  rm -- "$path" && printf '%s\t%s\t%s\tDELETED\n' "$path" "$size" "$sha" >> "$out"
done
