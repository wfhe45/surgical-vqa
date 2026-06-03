# EVAL — Reproducing MedVidBench Rank #1

**Target**: `uAI-NEXUS-MedVLM-1.0b-4B-RL` (UII), rank 1 on the
[MedVidBench Leaderboard](https://huggingface.co/spaces/UII-AI/MedVidBench-Leaderboard)
as of **2026-04-15**.

**Run command**: `bash .openresearch/speedrun.sh`
(auto-detected per the [OpenResearch run-commands docs](https://openresearch.sh/docs/run-commands);
delegates to `run_inference.sh` at the project root).

The script downloads the model + test data, filters out the LLM-judge-only
`qa_type`s (`video_summary`, `region_caption`), runs zero-shot transformers inference,
scores locally, and rewrites the **Reproduction** table below.

---

## Reproduction targets (rank #1, 2026-04-15)

LLM-judge metrics (`DVC_llm`, `VS_llm`, `RC_llm`) are out of scope — local
scoring only. Higher is better for every metric.

| # | Metric | Scale | Task (qa_type) | Target |
|---|--------|-------|----------------|--------|
| 1 | **CVS_acc**     | 0–1 | `cvs_assessment`   | **0.898** |
| 2 | **NAP_acc**     | 0–1 | `next_action`      | **0.473** |
| 3 | **SA_acc**      | 0–1 | `skill_assessment` | **0.285** |
| 4 | **STG_mIoU**    | 0–1 | `stg`              | **0.176** |
| 5 | **TAG_mIoU@0.3**| 0–1 | `tal`              | **0.504** |
| 6 | **TAG_mIoU@0.5**| 0–1 | `tal`              | **0.441** |
| 7 | **DVC_F1**      | 0–1 | `dense_captioning` | **0.480** |

Submitted: 2026-04-15 · Team: UII · Base: presumed Qwen2.5-VL-3B-Instruct (~4B total w/ vision tower)

## Reproduction (this run)

Scoring did not complete for this run — `eval/scores.json` is empty `{}`.
Inference succeeded on all 5374 instances; the scoring step (`utils/score_to_eval.py`)
did not populate metrics. Raw predictions were saved to
`results/results.json` on the compute instance but are not in artifacts.

| # | Metric | Target | Ours | Δ |
|---|--------|-------:|-----:|--:|
| 1 | CVS_acc      | 0.898 | — | — |
| 2 | NAP_acc      | 0.473 | — | — |
| 3 | SA_acc       | 0.285 | — | — |
| 4 | STG_mIoU     | 0.176 | — | — |
| 5 | TAG_mIoU@0.3 | 0.504 | — | — |
| 6 | TAG_mIoU@0.5 | 0.441 | — | — |
| 7 | DVC_F1       | 0.480 | — | — |

## Inference counts (this run)

| qa_type | Instances |
|---------|----------:|
| tal | 1637 |
| dense_captioning_gpt | 751 |
| dense_captioning_gemini | 728 |
| next_action | 670 |
| stg | 780 |
| cvs_assessment | 648 |
| skill_assessment | 160 |
| **Total** | **5374** |

Skipped (LLM-judge only): `video_summary`, `region_caption`

## Run configuration (this run)

| Field | Value |
|-------|-------|
| Model        | `UII-AI/uAI-NEXUS-MedVLM-1.0a-7B-RL` |
| Data         | `UII-AI/MedVidBench/cleaned_test_data_11_04.json` |
| Samples      | 5374 |
| GPUs         | 1 (gpu 0) |
| Batch size   | 16 |
| Max new tok. | 256 |
| GPU mem util | 0.90 |
| Inference    | 6765.94 s (~1h 52m) |
| Avg/instance | 1.26 s |
| Commit       | `0df1ef89bf16befe28c996ea1c90fdc509bec8af` |

**Note**: Rank-#1 model (`uAI-NEXUS-MedVLM-1.0b-4B-RL`) returned HTTP 404; run
used the available 7B sibling (`uAI-NEXUS-MedVLM-1.0a-7B-RL`). This is a **7B
baseline**, not the exact rank-#1 4B reproduction.

## Known blockers (verified 2026-05-30)

- **Rank-#1 model is gone (HTTP 404)**: `UII-AI/uAI-NEXUS-MedVLM-1.0b-4B-RL`
  returns `404 Repository not found` even with a valid `HF_TOKEN` — it is no
  longer publicly hosted (not merely gated/401 as previously assumed). The 7B
  sibling `UII-AI/uAI-NEXUS-MedVLM-1.0a-7B-RL` (200) and the `UII-AI/MedVidBench`
  dataset (200) are both reachable. The speedrun therefore **defaults to the 7B
  repo** so it runs end-to-end on a CUDA host; this yields a **7B baseline**,
  not the exact rank-#1 4B reproduction. Override with
  `MODEL_REPO=UII-AI/uAI-NEXUS-MedVLM-1.0b-4B-RL` only if/when the 4B repo is
  rehosted.
- **No system `pip`**: this image is `uv`-only. The project is now uv-native
  (`pyproject.toml` + `uv.lock`); `run_inference.sh` does `uv sync --locked` to
  install the pinned environment into `.venv/` automatically.
- **vLLM replaced with transformers** *(resolved 2026-05-31)*: vLLM 0.9.x had
  compatibility issues on H100/CUDA 12.8 hosts (0.9.2 introduced an attention-dtype
  assertion that fired on SM90; 0.9.1 also had issues passing pre-processed video
  tensors from the custom `vision_process_medical.py` pipeline). Replaced
  `inference/vllm_infer.py` to use `Qwen2_5_VLForConditionalGeneration` from
  `transformers>=4.52.0` directly. The model is loaded with `torch_dtype=torch.bfloat16,
  device_map="cuda"` and generation uses `model.generate(do_sample=False,
  max_new_tokens=...)`. The custom vision preprocessing (RC box drawing, medical
  frame processing) is unchanged — the output tensors from `vision_process_medical.py`
  are passed directly to `AutoProcessor` which applies the remaining Qwen2.5-VL
  normalization. `accelerate>=0.26.0` added to deps (required by `device_map`).
  `vllm` and its 30+ transitive deps removed from `uv.lock`, reducing install size.
- **transformers >=4.52.0 required** *(resolved 2026-05-31)*: transformers 4.51.3 has
  a bug in `GenerationConfig.from_model_config` — it calls `.to_dict()` on the
  `text_config` attribute which this model stores as a plain dict in `config.json`
  (saved with transformers 4.57.0). Upgrading the constraint to `>=4.52.0` (resolves
  to 5.9.0 currently) fixes the `AttributeError: 'dict' object has no attribute
  'to_dict'` at model init. Verified: `GenerationConfig.from_model_config(config)` 
  completes without error under transformers 5.9.0.
- **Video frames not found** *(resolved 2026-05-31)*: the annotation JSON
  `cleaned_test_data_11_04.json` stores frame paths as absolute paths from the
  original training host (e.g. `/root/data/AVOS/frames_15fps/.../22425.jpg`).
  The 103k frames (~19 GB) are published as `testdata.zip` in the HuggingFace
  dataset. Added stage **3.5** to `run_inference.sh`: downloads `testdata.zip`
  via `hf_hub_download`, extracts to `data/testdata/`, then rewrites every
  `video[]` frame path and `RC_info.start_frame` by replacing the `/root/data/`
  prefix with the local `data/testdata/` directory. The remapped JSON is saved
  as `data/test_data_remapped.json` and used by stage 4 (inference). Stage 5
  (scoring) still uses `test_data_filtered.json` for annotations (no file paths
  needed there). The `testdata/` directory presence is checked as a guard so
  the 19 GB download only runs once.
- **`unzip` not available** *(resolved 2026-05-31)*: `unzip` is not installed on this image.
  Stage 3.5 now uses `python3 -m zipfile -e` (Python stdlib) to extract `testdata.zip` instead.
  `python3 -m zipfile -e <src> <destdir>` exits 0 on success with no stdout output, making it
  a drop-in replacement for `unzip -q -d`.
- **`T_TOTAL` unbound variable — script exited 1 after successful run** *(resolved 2026-05-31)*:
  `run_inference.sh` wrote `total_seconds=${T_TOTAL}` into the `run.env` artifact block before
  computing `T_TOTAL=$(($(date +%s) - T_START))`. With `set -euo pipefail` the unbound variable
  caused an exit 1 immediately after EVAL.md was written — the run itself (inference + scoring)
  had succeeded. Fixed by moving `T_TOTAL` computation above the `run.env` block.
- **Deps pinned to CUDA 12.8-compatible versions** *(resolved)*: `torch>=2.7.0,<2.8`
  sourced from `https://download.pytorch.org/whl/cu128` (not PyPI) so installed
  wheels carry `+cu128`. `torchvision>=0.22.0,<0.23` from same index. Fixes
  `torch.cuda.is_available() → False` on CUDA 12.8 hosts seen with `torch 2.5.1+cu124`.
- **Scoring did not complete (this run)**: `eval/scores.json` is `{}` — the scoring
  step ran but produced no metrics. The raw predictions from inference were written to
  `results/results.json` on the compute instance (not in artifacts). The artifact
  `results.json` contains the input test data structure without model outputs.
  Next run: verify `utils/score_to_eval.py` reads from the correct path and that
  model predictions are included in artifacts.

## Artifacts

See `.openresearch/artifacts/README.md`.
