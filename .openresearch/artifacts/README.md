# Artifacts

Files written here are synced to OpenResearch cloud storage after a run, so
they persist beyond the compute instance and can be diffed across experiment
variants. Keep the contents small and structured.

## Layout produced by `.openresearch/speedrun.sh`

```
.openresearch/artifacts/
├── README.md                  # this file (description of layout)
├── run.env                    # snapshot of model, data, GPU, vLLM versions
├── results.json               # raw vLLM predictions (dict-of-records)
├── eval/
│   ├── scores.json            # per-metric numbers consumed by score_to_eval.py
│   └── per_qa_type_counts.json
└── logs/
    └── gpu{N}.log             # one log per GPU worker
```

## What each file is for

| Path | Purpose | Sync? |
|------|---------|-------|
| `run.env`            | Reproducibility: exact MODEL_REPO, DATA_REPO, GPU count, vllm/transformers/torch versions, commit SHA. | yes |
| `results.json`       | Raw model outputs keyed by sample index. Lets you re-derive scores without re-running vLLM. | yes |
| `eval/scores.json`   | Local scores (CVS_acc, NAP_acc, SA_acc, STG_mIoU, TAG_mIoU@0.3/0.5, DVC_F1). Drives the table in `EVAL.md`. | yes |
| `eval/per_qa_type_counts.json` | Sample counts per `qa_type`, sanity check vs. the 6,245 expected. | yes |
| `logs/gpu*.log`      | vLLM/transformers stdout per GPU. Truncated to last 2k lines per file. | yes |

## What does **not** go here

- Model weights — downloaded into `models/` (gitignored, *not* synced).
- The raw test-data video frames — downloaded into the HF cache.
- Per-GPU split shards (`*_gpu{N}.json`) — intermediate, deleted after merge.
