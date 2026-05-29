#!/bin/bash
#SBATCH --account=project_462001444
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=02:00:00
#SBATCH --job-name=llm_phase1_final
#SBATCH --output=/pfs/lustrep1/projappl/project_462001444/parliments_migration/src/migration-experiments21/parlaments-beyond-borders/logs/%x_%j.out
#SBATCH --error=/pfs/lustrep1/projappl/project_462001444/parliments_migration/src/migration-experiments21/parlaments-beyond-borders/logs/%x_%j.err

set -euo pipefail

ROOT=/pfs/lustrep1/projappl/project_462001444/parliments_migration/src/migration-experiments21/parlaments-beyond-borders
SCRATCH=/pfs/lustrep1/scratch/project_462001444
LOG_DIR="$ROOT/logs"
RUN_NAME="llm_insights_phase1_2026_05_26_final"
OUT_DIR="$ROOT/data/processed/ALL_AVAILABLE_COUNTRIES_comparisons/$RUN_NAME"
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000

mkdir -p "$LOG_DIR" "$OUT_DIR" "$SCRATCH/tmp" "$SCRATCH/ray-tmp" "$SCRATCH/xdg-cache" "$SCRATCH/vllm-cache" "$SCRATCH/fake-home" "$SCRATCH/triton-cache" "$SCRATCH/matplotlib-cache" "$SCRATCH/torch-home" "$SCRATCH/torchinductor-cache" "$SCRATCH/singularity-cache" "$SCRATCH/xdg-config"
cd "$ROOT"

echo "[$(date)] job=$SLURM_JOB_ID node=$(hostname) root=$ROOT"
echo "[$(date)] outputs=$OUT_DIR"
echo "[$(date)] model=$MODEL"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
export HOME="$SCRATCH/fake-home"
export HF_HOME="$SCRATCH/huggingface"
export HUGGINGFACE_HUB_CACHE="$SCRATCH/huggingface/hub"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface/hub"
export VLLM_CACHE_ROOT="$SCRATCH/vllm-cache"
export XDG_CACHE_HOME="$SCRATCH/xdg-cache"
export XDG_CONFIG_HOME="$SCRATCH/xdg-config"
export TRITON_CACHE_DIR="$SCRATCH/triton-cache"
export MPLCONFIGDIR="$SCRATCH/matplotlib-cache"
export TORCH_HOME="$SCRATCH/torch-home"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/torchinductor-cache"
export SINGULARITY_CACHEDIR="$SCRATCH/singularity-cache"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export SINGULARITYENV_HOME="$SCRATCH/fake-home"
export SINGULARITYENV_HF_HOME="$SCRATCH/huggingface"
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="$SCRATCH/huggingface/hub"
export SINGULARITYENV_TRANSFORMERS_CACHE="$SCRATCH/huggingface/hub"
export SINGULARITYENV_VLLM_CACHE_ROOT="$SCRATCH/vllm-cache"
export SINGULARITYENV_XDG_CACHE_HOME="$SCRATCH/xdg-cache"
export SINGULARITYENV_XDG_CONFIG_HOME="$SCRATCH/xdg-config"
export SINGULARITYENV_TRITON_CACHE_DIR="$SCRATCH/triton-cache"
export SINGULARITYENV_MPLCONFIGDIR="$SCRATCH/matplotlib-cache"
export SINGULARITYENV_TORCH_HOME="$SCRATCH/torch-home"
export SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="$SCRATCH/torchinductor-cache"
export SINGULARITYENV_VLLM_NO_USAGE_STATS=1
export SINGULARITYENV_DO_NOT_TRACK=1
export RAY_TMPDIR="$SCRATCH/ray-tmp"
export TMPDIR="$SCRATCH/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"

MIOPEN_DIR=$(mktemp -d "$SCRATCH/tmp/miopen-${SLURM_JOB_ID}.XXXXXX")
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_DIR/cache"
export MIOPEN_USER_DB="$MIOPEN_DIR/config"

SERVER_LOG="$LOG_DIR/vllm_phase1_final_${SLURM_JOB_ID}.log"
CLIENT_LOG="$LOG_DIR/llm_phase1_final_${SLURM_JOB_ID}.log"

cleanup() {
  echo "[$(date)] cleanup"
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date)] starting vLLM server"
srun --exclusive --ntasks=1 --cpus-per-task=14 --gpus=1 singularity run "$SIF" \
  vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --disable-log-stats \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "[$(date)] waiting for vLLM on http://127.0.0.1:$PORT/v1/models"
for i in $(seq 1 180); do
  if curl -sS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[$(date)] vLLM ready after attempt $i"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[$(date)] vLLM exited early; last server log:"
    tail -120 "$SERVER_LOG" || true
    exit 1
  fi
  echo "[$(date)] waiting for vLLM... attempt $i"
  sleep 10
  if [[ "$i" == "180" ]]; then
    echo "[$(date)] vLLM did not become ready"
    tail -120 "$SERVER_LOG" || true
    exit 1
  fi
done

source "$SCRATCH/venvs/parl-llm/bin/activate"
python --version

echo "[$(date)] running prompt script task=all run=$RUN_NAME"
python scripts/llm_prompts_phase1.py \
  --task all \
  --server "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL" \
  --top-n 50 \
  --run-name "$RUN_NAME" \
  2>&1 | tee "$CLIENT_LOG"

echo "[$(date)] output files:"
find "$OUT_DIR" -maxdepth 1 -type f -printf "%f\t%s bytes\n" | sort

echo "[$(date)] done"
