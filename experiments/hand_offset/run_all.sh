#!/usr/bin/env bash
# Run every hand-offset variant. Each call produces one overlay MP4 in
# experiments/hand_offset/outputs/<label>.mp4 (alpha=0.50).
#
# Usage:
#   conda activate 3dv
#   bash experiments/hand_offset/run_all.sh
#
# Optional environment variables:
#   ONLY=v0_baseline,v3_ee_y_neg15mm   # comma-list of labels to run; default = all
#   EXTRA_ARGS="--object none"         # appended to the kinematic_replay command
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LAUNCHER="experiments/hand_offset/launcher.py"
REFINED="data/sam_masks_aria_extrinsic.npz"
COMMON=(--refined-extrinsic "$REFINED" --record-overlay 0.50)
EXTRA=(${EXTRA_ARGS:-})

want() {
    [[ -z "${ONLY:-}" ]] && return 0
    [[ ",${ONLY}," == *",$1,"* ]]
}

run() {
    local label="$1"; shift
    if ! want "$label"; then
        echo "[run_all] skip $label"
        return
    fi
    echo
    echo "============================================================"
    echo "[run_all] $label"
    echo "============================================================"
    local start_ts; start_ts=$(date +%s)
    python "$LAUNCHER" --exp-label "$label" "$@" "${COMMON[@]}" "${EXTRA[@]}" || true
    # Belt-and-suspenders: if the launcher's atexit hook didn't catch it,
    # move any overlay MP4 produced AFTER this run started.
    local stray
    stray=$(find outputs -maxdepth 1 -name '*_overlay_a*.mp4' \
        -newermt "@$start_ts" -print 2>/dev/null | head -1 || true)
    if [[ -n "$stray" ]]; then
        mv "$stray" "experiments/hand_offset/outputs/$label.mp4"
        echo "[run_all] (fallback rename) $stray -> experiments/hand_offset/outputs/$label.mp4"
    fi
}

# v0: baseline -- the user's current observation (refined extrinsic, no tweaks).
run v0_baseline

# v1: control -- no refinement applied, so we can see the gap between
# nominal and refined and visually confirm refinement is helpful overall.
run v1_no_refine --exp-no-refine

# v2: EE wrist offset y = +0.015 m (orcahand mounting drifted along link8 +Y).
run v2_ee_y_pos15mm --exp-ee-offset 0.13,0.015,0.07

# v3: EE wrist offset y = -0.015 m (mirror of v2).
run v3_ee_y_neg15mm --exp-ee-offset 0.13,-0.015,0.07

# v4: EE wrist offset x = 0.15 m (longer mounting offset along link8 +X).
run v4_ee_x_pos20mm --exp-ee-offset 0.15,0.0,0.07

# v5: principal-point cx -= 15 px (image centre actually further left).
run v5_cx_minus15 --exp-cx-shift -15

# v6: principal-point cx += 15 px (mirror of v5).
run v6_cx_plus15  --exp-cx-shift 15

# v7: focal length scaled 1.03x (intrinsic fx/fy slightly larger than reported).
run v7_focal_scale_1p03 --exp-fx-scale 1.03 --exp-fy-scale 1.03

echo
echo "[run_all] Done. Outputs:"
ls -la experiments/hand_offset/outputs/
