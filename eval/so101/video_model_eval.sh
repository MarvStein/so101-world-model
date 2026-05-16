
#!/usr/bin/env bash
set -euo pipefail

# Usage: bash video_model_eval.sh --dit-path <path_to_checkpoint>
DIT_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dit-path) DIT_PATH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DIT_PATH" ]]; then
    echo "Error: --dit-path is required"
    exit 1
fi

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${WORKSPACE}/data/video_eval"
mkdir -p "${OUTPUT_DIR}"

# Extract iteration number from the checkpoint filename (e.g. iter_000004000 -> 4000)
ITER=$(basename "${DIT_PATH}" | grep -oP 'iter_\K[0-9]+' | sed 's/^0*//' || echo "unknown")
ITER_TAG="iter${ITER}"

bash "${WORKSPACE}/scripts/run_video2world_5cond.sh" \
    --dit-path "${DIT_PATH}" \
    --input-video "${WORKSPACE}/data/video_fine/lerobot/video/episode_021.mp4" \
    --prompt "Push the target object, in this case the white polyhedron, in a straight line to the goal position which is the smaller of the two white circles seen on the left. The target object is not allowed to leave the area bounded by the two parallel straight white lines at anytime." \
    --output-video "${OUTPUT_DIR}/task1_021_${ITER_TAG}.mp4" \
    --start-frame-index 50

bash "${WORKSPACE}/scripts/run_video2world_5cond.sh" \
    --dit-path "${DIT_PATH}" \
    --input-video "${WORKSPACE}/data/video_fine/lerobot/video/episode_340.mp4" \
    --prompt "Push the target object, in this case the white polyhedron, around the obstacle, a red cylinder laying in the big white circle, to the goal position which is the smaller of the two white circles seen on the left. The target object can leave the area bounded by the two parallel white straight lines. The target object is not allowed to touch the red cylinder at all!" \
    --output-video "${OUTPUT_DIR}/task2_340_${ITER_TAG}.mp4" \
    --start-frame-index 45

bash "${WORKSPACE}/scripts/run_video2world_5cond.sh" \
    --dit-path "${DIT_PATH}" \
    --input-video "${WORKSPACE}/data/video_fine/lerobot/video/episode_330.mp4" \
    --prompt "Push the target object, in this case the white polyhedron, around the obstacle, a red cylinder laying in the big white circle, to the goal position which is the smaller of the two white circles seen on the left. The target object can leave the area bounded by the two parallel white straight lines. The target object is not allowed to touch the red cylinder at all!" \
    --output-video "${OUTPUT_DIR}/task2_330_${ITER_TAG}.mp4" \
    --start-frame-index 100


bash "${WORKSPACE}/scripts/run_video2world_5cond.sh" \
    --dit-path "${DIT_PATH}" \
    --input-video "${WORKSPACE}/data/video_fine/lerobot/video/episode_500.mp4" \
    --prompt "Push the target object, in this case the blue cube, around the obstacle, a red cylinder laying in the big white circle, to the goal position which is the smaller of the two white circles seen on the left. The target object can leave the area bounded by the two parallel white straight lines. The target object is not allowed to touch the red cylinder at all!" \
    --output-video "${OUTPUT_DIR}/task22_500_${ITER_TAG}.mp4" \
    --start-frame-index 40

bash "${WORKSPACE}/scripts/run_video2world_5cond.sh" \
    --dit-path "${DIT_PATH}" \
    --input-video "${WORKSPACE}/data/video_fine/lerobot/video/episode_450.mp4" \
    --prompt "Push the target object, in this case the blue cube, around the obstacle, a red cylinder laying in the big white circle, to the goal position which is the smaller of the two white circles seen on the left. The target object can leave the area bounded by the two parallel white straight lines. The target object is not allowed to touch the red cylinder at all!" \
    --output-video "${OUTPUT_DIR}/task13_450_${ITER_TAG}.mp4" \
    --start-frame-index 40
    
    
