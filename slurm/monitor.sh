# Sourced by GPU jobs; sample only the GPU(s) granted by Slurm every 10 seconds.
# PyTorch worker pools preserve CUDA_VISIBLE_DEVICES. nvidia-smi uses physical indices.
monitor_file="runs/logs/gpu-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-single}.csv"
monitor_gpu="${CUDA_VISIBLE_DEVICES:-0}"
nvidia-smi --id="$monitor_gpu" \
    --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv --loop=10 > "$monitor_file" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
