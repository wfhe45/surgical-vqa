# SURGVU25 Cat.2 Surgical VQA — Setup Guide

## 1. Clone the repository

```bash
git clone https://github.com/wfhe45/surgical-vqa.git
cd surgical-vqa
```

---

## 2. Install uv (package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or restart terminal
```

---

## 3. Set up Python environment

```bash
uv sync --locked
source .venv/bin/activate
```

This installs all exact dependencies from `uv.lock` (torch, transformers, vllm, etc.).

---

## 4. Download the model (~16 GB)

```bash
source .venv/bin/activate

python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="UII-AI/uAI-NEXUS-MedVLM-1.0a-7B-RL",
    local_dir="models/uAI-NEXUS-MedVLM-1.0a-7B-RL"
)
PY
```

Requires a HuggingFace account with access to the repo. Set token if needed:
```bash
export HF_TOKEN=your_token_here
```

---

## 5. Upload your MP4 videos

Place the original MP4 files back (they are not in the repo due to size):

```
SURGVU25_cat_2_sample_set_public/
    case122/case122.mp4
    case123/case123.mp4
    ...
    case132/case132.mp4
```

Extracted frames are already included in `data/testdata/SURGVU25/` (330 JPGs at 1fps).
Skip this step if you don't need to re-extract frames.

---

## 6. Run inference (requires CUDA GPU)

```bash
bash run_surgvu25.sh
```

The script will:
1. Check for GPU
2. Skip model download (already done in step 4)
3. Run inference on all 11 cases
4. Score results with BLEU and update EVAL.md
5. Print per-case results and average

Optional parameters:
```bash
BATCH_SIZE=8 MAX_NEW_TOKENS=128 bash run_surgvu25.sh
```

---

## 7. Score existing results (no GPU needed)

If you already have prediction results, score them directly:

```bash
source .venv/bin/activate

python3 utils/score_to_eval.py \
    --results  results/results_surgvu25_v5.json \
    --gt       data/test_data_surgvu25.json \
    --eval_md  EVAL.md \
    --scores_out  .openresearch/artifacts/eval/scores.json \
    --counts_out  .openresearch/artifacts/eval/counts.json \
    --model_repo  "UII-AI/uAI-NEXUS-MedVLM-1.0a-7B-RL" \
    --data_repo   "SURGVU25_cat_2_sample_set_public" \
    --n_gpus 1 --batch_size 4 --max_new_tokens 64 --inference_seconds 0
```

---

## What's included in this repo

| Path | Contents |
|------|---------|
| `inference/vllm_infer.py` | Main inference script (Qwen2.5-VL via transformers) |
| `inference/vision_process_medical.py` | Video frame processing |
| `utils/score_to_eval.py` | BLEU scoring + post-processing rewrite rules |
| `utils/convert_surgvu25_to_frames.py` | MP4 → JPG frame extractor (1fps) |
| `run_surgvu25.sh` | One-click inference + scoring pipeline |
| `SURGVU25_cat_2_sample_set_public/` | 11 cases: question + 5 reference answers (JSON) |
| `data/test_data_surgvu25.json` | MedVidBench-format dataset (11 cases) |
| `data/testdata/SURGVU25/` | Extracted frames (330 JPGs, 1fps, 11 cases) |
| `results/results_surgvu25_v*.json` | All 8 inference version outputs (v1–v8) |
| `SURGVU25_EXPERIMENT_SUMMARY.md` | Full experiment report |
| `pyproject.toml` + `uv.lock` | Locked Python dependencies |

---

## Best result achieved

**avg BLEU = 0.3850** (11.6x over no-prompt baseline)

Configuration: v5 inference (system message + simple-word guidance) + FixE post-processing rewrite.
See `SURGVU25_EXPERIMENT_SUMMARY.md` for full details.

---

## Re-extract frames from MP4 (optional)

If you add new cases or want to change the extraction FPS:

```bash
source .venv/bin/activate
uv pip install av   # required for video reading

python3 utils/convert_surgvu25_to_frames.py \
    --input_dir SURGVU25_cat_2_sample_set_public \
    --frames_dir data/testdata/SURGVU25 \
    --output_json data/test_data_surgvu25.json \
    --fps 1.0
```
