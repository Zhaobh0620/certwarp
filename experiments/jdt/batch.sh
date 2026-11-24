#!/usr/bin/env bash
# JDT L0 diagnostic: every stored CVG state used in the draft (R1-k500 was run first, alone, as the smoke check).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
while read -r RUN STATE K LABEL; do
    [ -z "$RUN" ] && continue
    bash "$HERE/run.sh" "$RUN" "$STATE" "$K" "$LABEL" | tail -n 1 | cut -c1-400
done <<'LIST'
R1-struct u_L0_k1000.npz 1000 R1-k1000
R1-struct u_L0_k2000.npz 2000 R1-k2000
R1-struct u_final_L0.npz 4716 R1-final
R2-tissue u_L0_k50.npz 50 R2-k50
R2-tissue u_L0_k200.npz 200 R2-k200
R2-tissue u_L0_k500.npz 500 R2-k500
R2-tissue u_L0_k1000.npz 1000 R2-k1000
R2-tissue u_L0_k2000.npz 2000 R2-k2000
R2-tissue u_final_L0.npz 4785 R2-final
R7-struct-0012 u_L0_k1000.npz 1000 R7-k1000
R7-struct-0012 u_final_L0.npz 1306 R7-final
LIST
