#!/bin/bash
# Run inference + scoring on SURGVU25 Cat.2 dataset.
# Usage: bash run_surgvu25.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_REPO="${MODEL_REPO:-UII-AI/uAI-NEXUS-MedVLM-1.0a-7B-RL}"
MODEL_PATH="${SCRIPT_DIR}/models/$(basename "${MODEL_REPO}")"
DATA_JSON="${SCRIPT_DIR}/data/test_data_surgvu25.json"
RESULTS_JSON="${SCRIPT_DIR}/results/results_surgvu25.json"
SCORES_OUT="${SCRIPT_DIR}/.openresearch/artifacts/eval/scores_surgvu25.json"
COUNTS_OUT="${SCRIPT_DIR}/.openresearch/artifacts/eval/counts_surgvu25.json"
LOG_DIR="${SCRIPT_DIR}/.openresearch/artifacts/logs"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

echo "=============================================="
echo "SURGVU25 Cat.2 Inference + Scoring"
echo "  Model : ${MODEL_REPO}"
echo "  Data  : ${DATA_JSON}"
echo "  Output: ${RESULTS_JSON}"
echo "=============================================="

# ── GPU check ────────────────────────────────────────────────
if ! { ls /dev/nvidia0 >/dev/null 2>&1 || command -v nvidia-smi >/dev/null 2>&1; }; then
    echo "ERROR: no CUDA GPU detected. This script requires a GPU." >&2
    exit 1
fi
echo "[GPU] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

# ── Activate venv ────────────────────────────────────────────
echo "[1/4] Activating environment..."
if [ ! -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
    uv sync --locked
fi
source "${SCRIPT_DIR}/.venv/bin/activate"

mkdir -p "${SCRIPT_DIR}/results" "${LOG_DIR}" "$(dirname "${SCORES_OUT}")"

# ── Download model ────────────────────────────────────────────
echo "[2/4] Checking model..."
if [ ! -d "${MODEL_PATH}" ]; then
    echo "  Downloading ${MODEL_REPO} (~15 GB)..."
    python3 - "${MODEL_REPO}" "${MODEL_PATH}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
print("  Download complete:", sys.argv[2])
PY
else
    echo "  Model already present: ${MODEL_PATH}"
fi

# ── Check data ────────────────────────────────────────────────
if [ ! -f "${DATA_JSON}" ]; then
    echo "ERROR: data file not found: ${DATA_JSON}" >&2
    echo "       Run utils/convert_surgvu25_to_frames.py first." >&2
    exit 1
fi
N_SAMPLES=$(python3 -c "import json; print(len(json.load(open('${DATA_JSON}'))))")
echo "  Data: ${N_SAMPLES} samples"

# ── Inference ─────────────────────────────────────────────────
echo "[3/4] Running inference (batch=${BATCH_SIZE}, max_new_tokens=${MAX_NEW_TOKENS})..."
T_START=$(date +%s)

python3 inference/vllm_infer.py \
    --model_path "${MODEL_PATH}" \
    --data_path  "${DATA_JSON}" \
    --output_path "${RESULTS_JSON}" \
    --batch_size "${BATCH_SIZE}" \
    --max_pixels_per_frame $((48*28*28)) \
    --min_pixels_per_frame $((8*28*28)) \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${LOG_DIR}/surgvu25_inference.log"

T_INFER=$(($(date +%s) - T_START))
echo "  Inference done in ${T_INFER}s"

# ── Scoring ───────────────────────────────────────────────────
echo "[4/4] Scoring (BLEU)..."
python3 utils/score_to_eval.py \
    --results        "${RESULTS_JSON}" \
    --gt             "${DATA_JSON}" \
    --eval_md        "${SCRIPT_DIR}/EVAL.md" \
    --scores_out     "${SCORES_OUT}" \
    --counts_out     "${COUNTS_OUT}" \
    --model_repo     "${MODEL_REPO}" \
    --data_repo      "SURGVU25_cat_2_sample_set_public" \
    --n_gpus         1 \
    --batch_size     "${BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --inference_seconds "${T_INFER}"

echo "=============================================="
echo "Done. Results: ${RESULTS_JSON}"
echo "Scores: ${SCORES_OUT}"
echo "EVAL.md updated."
echo "=============================================="

# ── Per-case report ───────────────────────────────────────────
echo ""
python3 - "${DATA_JSON}" "${RESULTS_JSON}" "${SCORES_OUT}" <<'PY'
import json, sys

gt      = json.load(open(sys.argv[1]))
results = json.load(open(sys.argv[2]))
scores  = json.load(open(sys.argv[3]))

sys.path.insert(0, "utils")
from score_to_eval import score_surgical_vqa

bleu_list = []
print("=" * 70)
print(f"  Per-case results ({len(gt)} cases)")
print("=" * 70)
for i, rec in enumerate(gt):
    pred = results.get(str(i), {}).get("answer", "")
    refs = rec.get("reference_answers") or [rec["conversations"][1]["value"]]
    bleu = score_surgical_vqa(pred, refs)
    bleu_list.append(bleu)
    question = rec["conversations"][0]["value"].replace("<video>\n", "")
    print(f"[{rec['id']}]")
    print(f"  Q    : {question}")
    print(f"  Pred : {pred}")
    print(f"  Refs : {refs[0]}  (+{len(refs)-1} more)")
    print(f"  BLEU : {bleu:.4f}")
    print()

avg  = sum(bleu_list) / len(bleu_list)
best = max(bleu_list)
worst = min(bleu_list)
best_case  = gt[bleu_list.index(best)]["id"]
worst_case = gt[bleu_list.index(worst)]["id"]

print("=" * 70)
print(f"  Cases : {len(bleu_list)}")
print(f"  Avg BLEU  : {avg:.4f}")
print(f"  Max BLEU  : {best:.4f}  ({best_case})")
print(f"  Min BLEU  : {worst:.4f}  ({worst_case})")
print("=" * 70)
PY
