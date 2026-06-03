#!/bin/bash
# MedVidBench rank-#1 reproduction — single entry point.
# Run: bash run_inference.sh
#
# Stages: install deps → download model+data → filter qa_types →
#         vLLM inference (single/multi-GPU) → local scoring → rewrite EVAL.md.

set -euo pipefail
export TZ=America/New_York

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ============================================================
# CONFIGURATION
# ============================================================
# Default is the 7B sibling: the rank-#1 4B repo (uAI-NEXUS-MedVLM-1.0b-4B-RL)
# returns HTTP 404 (no longer hosted as of 2026-05-30), so it can't be the
# default. Override MODEL_REPO=UII-AI/uAI-NEXUS-MedVLM-1.0b-4B-RL if it returns.
MODEL_REPO="${MODEL_REPO:-UII-AI/uAI-NEXUS-MedVLM-1.0a-7B-RL}"
DATA_REPO="${DATA_REPO:-UII-AI/MedVidBench}"
DATA_FILE="${DATA_FILE:-cleaned_test_data_11_04.json}"
# GPU selection — auto-detected by default, so 1/2/4/8-GPU hosts all just
# work without passing flags. Precedence:
#   explicit GPUS  >  CUDA_VISIBLE_DEVICES  >  `nvidia-smi` enumeration.
# Inference is data-parallel (one vLLM process per GPU, tensor_parallel=1),
# so the data is split into N_GPUS shards — N_GPUS therefore follows the
# resolved GPU-list length unless you override N_GPUS explicitly.
if [ -n "${GPUS:-}" ]; then
    GPUS=(${GPUS})                                   # e.g. GPUS="0 1 2 3"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null))
else
    GPUS=()
fi
[ ${#GPUS[@]} -eq 0 ] && GPUS=(0)                    # last-resort default
N_GPUS="${N_GPUS:-${#GPUS[@]}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SKIP_QA_TYPES="${SKIP_QA_TYPES:-video_summary,region_caption}"  # LLM-judge-only
# ============================================================

MODEL_PATH="${SCRIPT_DIR}/models/$(basename "${MODEL_REPO}")"
DATA_DIR="${SCRIPT_DIR}/data"
DATA_PATH="${DATA_DIR}/${DATA_FILE}"
FILTERED_PATH="${DATA_DIR}/test_data_filtered.json"
OUTPUT_DIR="${SCRIPT_DIR}/results"
ART_DIR="${SCRIPT_DIR}/.openresearch/artifacts"
LOG_DIR="${ART_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${ART_DIR}/eval" "${DATA_DIR}"

echo "=============================================="
echo "MedVidBench rank-#1 reproduction"
echo "  Model:   ${MODEL_REPO}"
echo "  Data:    ${DATA_REPO}/${DATA_FILE}"
echo "  GPUs:    ${GPUS[*]} (${N_GPUS} GPU(s))"
echo "  Batch:   ${BATCH_SIZE}"
echo "  Skip:    ${SKIP_QA_TYPES}"
echo "=============================================="

# ── Preflight: stage 4 (vLLM inference) REQUIRES a CUDA GPU ──────────
# Detect a GPU up front so we fail fast with an actionable message INSTEAD
# of burning ~6 min on the env install + a ~16 GB model download only to
# crash deep inside vLLM with the cryptic "RuntimeError: Device string must
# not be empty" (its no-CUDA symptom: current_platform.device_type == "").
# The installed vLLM is a CUDA-only wheel and cannot fall back to CPU here.
# Set ALLOW_NO_GPU=1 to bypass this and exercise the CPU-only stages
# (install / download / filter / scoring) for plumbing tests.
if ! { ls /dev/nvidia0 >/dev/null 2>&1 || command -v nvidia-smi >/dev/null 2>&1; }; then
    if [ "${ALLOW_NO_GPU:-0}" = "1" ]; then
        echo "  WARNING: no CUDA GPU detected, but ALLOW_NO_GPU=1 — continuing."
        echo "           Stage 4 (inference) will still fail; stages 1–3 are CPU-only."
    else
        echo "  ERROR: no CUDA GPU detected (no /dev/nvidia0, no nvidia-smi)." >&2
        echo "         Stage 4 (vLLM inference) requires a CUDA GPU and cannot run on" >&2
        echo "         CPU here — the pinned vLLM is a CUDA-only wheel. Run this on a" >&2
        echo "         GPU host. Expected stage-4 time: ~30–90 min over 5,374 items on" >&2
        echo "         2 GPUs (batch ${BATCH_SIZE}); scoring is seconds." >&2
        echo "         To exercise the CPU-only stages anyway, re-run with ALLOW_NO_GPU=1." >&2
        exit 1
    fi
fi

T_START=$(date +%s)

# ── 1/5 Sync environment (uv-native) ────────────────────────────────
# Reproducible install from the committed uv.lock: `uv sync --locked` asserts
# the lock is in sync with pyproject.toml and installs the exact pinned
# versions into ${SCRIPT_DIR}/.venv. Activating it makes every later `python3`
# resolve to that venv (no pip needed; this image ships only uv).
echo "[1/5] Syncing environment from uv.lock..."
if ! command -v uv >/dev/null 2>&1; then
    echo "  ERROR: uv is required — this project is uv-native (pyproject.toml + uv.lock)." >&2
    echo "         Install: https://docs.astral.sh/uv/  (or 'pip install -r requirements.txt'" >&2
    echo "         against the pinned export as a fallback)." >&2
    exit 1
fi
uv sync --locked
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/.venv/bin/activate"

# ── 2/5 Download model + data ───────────────────────────────────────
# Use the huggingface_hub Python API, NOT `huggingface-cli`: in hub >= 1.0 the
# `huggingface-cli` shim is deprecated and exits 0 WITHOUT downloading, which
# silently defeats an `if !`-style guard and breaks stage 3. The Python API is
# stable across versions and raises (non-zero exit) on real failures.
echo "[2/5] Downloading model..."
if ! python3 - "${MODEL_REPO}" "${MODEL_PATH}" <<'PY' > "${LOG_DIR}/download.log" 2>&1
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=dest)
print("ok:", dest)
PY
then
    echo "  ERROR: model download failed for ${MODEL_REPO}."
    echo "         See ${LOG_DIR}/download.log"
    echo "  Note: the default is the 7B repo (1.0a-7B-RL). The rank-#1 4B repo"
    echo "        (1.0b-4B-RL) returns HTTP 404 as of 2026-05-30 and is not a"
    echo "        usable override until it is rehosted. Check HF_TOKEN access."
    exit 1
fi

echo "[2/5] Downloading test data..."
python3 - "${DATA_REPO}" "${DATA_FILE}" "${DATA_DIR}" <<'PY' >> "${LOG_DIR}/download.log" 2>&1
import sys
from huggingface_hub import hf_hub_download
repo, fname, dest = sys.argv[1], sys.argv[2], sys.argv[3]
p = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", local_dir=dest)
print("ok:", p)
PY

# ── 3/5 Filter LLM-judge-only qa_types ──────────────────────────────
echo "[3/5] Filtering qa_types (skip: ${SKIP_QA_TYPES})..."
python3 - "${DATA_PATH}" "${FILTERED_PATH}" "${SKIP_QA_TYPES}" <<'PY'
import json, sys
from collections import Counter
src, dst, skip_csv = sys.argv[1], sys.argv[2], sys.argv[3]
skip = {s.strip() for s in skip_csv.split(",") if s.strip()}
data = json.load(open(src))
# Dataset qa_types carry judge suffixes (video_summary_gpt, region_caption_gemini,
# dense_captioning_gpt, ...), so match by PREFIX — exact-match skipped nothing.
def is_skipped(qt): return any((qt or "").startswith(s) for s in skip)
kept = [d for d in data if not is_skipped(d.get("qa_type"))]
json.dump(kept, open(dst, "w"))
print(f"  kept {len(kept)}/{len(data)};",
      "per-type:", dict(Counter(d.get('qa_type') for d in kept)))
PY

# ── 3.5/5 Download video frames + remap paths ───────────────────────
# The JSON has absolute frame paths from the original training host (/root/data/...).
# testdata.zip (~19 GB) on HuggingFace contains the 103k frames; after extracting,
# we remap each item's video[] paths so inference can actually find the files.
FRAMES_DIR="${DATA_DIR}/testdata"
INFER_DATA="${DATA_DIR}/test_data_remapped.json"
echo "[3.5/5] Setting up video frames..."
if [ ! -d "${FRAMES_DIR}" ]; then
    echo "  Downloading testdata.zip (~19 GB)..."
    python3 - "${DATA_REPO}" "testdata.zip" "${DATA_DIR}" <<'PY' >> "${LOG_DIR}/download.log" 2>&1
import sys
from huggingface_hub import hf_hub_download
repo, fname, dest_dir = sys.argv[1], sys.argv[2], sys.argv[3]
p = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", local_dir=dest_dir)
print("ok:", p)
PY
    echo "  Extracting frames..."
    python3 -m zipfile -e "${DATA_DIR}/testdata.zip" "${DATA_DIR}/"
    rm -f "${DATA_DIR}/testdata.zip"
    echo "  Extracted to ${FRAMES_DIR}"
else
    echo "  Frames already present at ${FRAMES_DIR}"
fi

# Remap /root/data/... paths to the local testdata/ directory.
python3 - "${FILTERED_PATH}" "${FRAMES_DIR}" "${INFER_DATA}" <<'PY'
import json, sys, os
src, frames_dir, dst = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.load(open(src))
PREFIX = "/root/data/"

def remap(path):
    if path and path.startswith(PREFIX):
        return os.path.join(frames_dir, path[len(PREFIX):])
    return path

for item in data:
    item["video"] = [remap(p) for p in item.get("video", [])]
    if item.get("RC_info") and item["RC_info"].get("start_frame"):
        item["RC_info"]["start_frame"] = remap(item["RC_info"]["start_frame"])

json.dump(data, open(dst, "w"))
# Quick sanity: verify first frame of first item exists
first = data[0]["video"][0] if data and data[0].get("video") else None
if first and not os.path.exists(first):
    print(f"  WARNING: first frame not found after remap: {first}")
    print(f"  Check that testdata.zip extracted correctly to {frames_dir}")
else:
    print(f"  Remapped {len(data)} items (first frame OK: {first})")
PY

# ── 4/5 Transformers inference (single or multi-GPU) ────────────────
echo "[4/5] Running inference..."
T_INFER_START=$(date +%s)

if [ "${N_GPUS}" -eq 1 ]; then
    GPU_ID=${GPUS[0]}
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 inference/vllm_infer.py \
        --model_path "${MODEL_PATH}" \
        --data_path "${INFER_DATA}" \
        --output_path "${OUTPUT_DIR}/results.json" \
        --batch_size "${BATCH_SIZE}" \
        --max_pixels_per_frame $((48*28*28)) \
        --min_pixels_per_frame $((8*28*28)) \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --gpu_memory_utilization "${GPU_MEM_UTIL}" \
        2>&1 | tee "${LOG_DIR}/gpu${GPU_ID}.log"
else
    python3 utils/split_data_balanced.py "${INFER_DATA}" "${N_GPUS}"
    declare -a PIDS
    for i in "${!GPUS[@]}"; do
        GPU_ID=${GPUS[$i]}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python3 inference/vllm_infer.py \
            --model_path "${MODEL_PATH}" \
            --data_path "${INFER_DATA%.json}_gpu${i}.json" \
            --output_path "${OUTPUT_DIR}/results_gpu${i}.json" \
            --batch_size "${BATCH_SIZE}" \
            --max_pixels_per_frame $((48*28*28)) \
            --min_pixels_per_frame $((8*28*28)) \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --gpu_memory_utilization "${GPU_MEM_UTIL}" \
            > "${LOG_DIR}/gpu${GPU_ID}.log" 2>&1 &
        PIDS[$i]=$!
        echo "  GPU ${GPU_ID}: PID ${PIDS[$i]}"
    done
    FAIL=0
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" && echo "  GPU ${GPUS[$i]} done" || { echo "  GPU ${GPUS[$i]} FAILED"; FAIL=1; }
    done
    [ "${FAIL}" -ne 0 ] && exit 1
    python3 utils/merge_results_manual.py "${OUTPUT_DIR}/results.json" "${N_GPUS}"
fi

T_INFER=$(($(date +%s) - T_INFER_START))
echo "  inference: ${T_INFER}s"

# ── 5/5 Score → EVAL.md + artifacts ─────────────────────────────────
echo "[5/5] Scoring and writing EVAL.md..."
python3 utils/score_to_eval.py \
    --results "${OUTPUT_DIR}/results.json" \
    --gt "${FILTERED_PATH}" \
    --eval_md "${SCRIPT_DIR}/EVAL.md" \
    --scores_out "${ART_DIR}/eval/scores.json" \
    --counts_out "${ART_DIR}/eval/per_qa_type_counts.json" \
    --model_repo "${MODEL_REPO}" \
    --data_repo "${DATA_REPO}/${DATA_FILE}" \
    --n_gpus "${N_GPUS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --inference_seconds "${T_INFER}"

cp -f "${OUTPUT_DIR}/results.json" "${ART_DIR}/results.json"
T_TOTAL=$(($(date +%s) - T_START))
{
    echo "model_repo=${MODEL_REPO}"
    echo "data_repo=${DATA_REPO}/${DATA_FILE}"
    echo "n_gpus=${N_GPUS}"
    echo "gpus=${GPUS[*]}"
    echo "batch_size=${BATCH_SIZE}"
    echo "gpu_memory_utilization=${GPU_MEM_UTIL}"
    echo "max_new_tokens=${MAX_NEW_TOKENS}"
    echo "skip_qa_types=${SKIP_QA_TYPES}"
    echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "total_seconds=${T_TOTAL}"
    python3 -c "import vllm, transformers, torch; print(f'vllm={vllm.__version__}\ntransformers={transformers.__version__}\ntorch={torch.__version__}\ncuda={torch.version.cuda}')" 2>/dev/null || true
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader 2>/dev/null || true
} > "${ART_DIR}/run.env"

echo "=============================================="
echo "Done in ${T_TOTAL}s. EVAL.md updated."
echo "Artifacts: ${ART_DIR}/"
echo "=============================================="
