#!/usr/bin/env bash
# OpenResearch speedrun — auto-detected per openresearch.sh/docs/run-commands.
# Delegates to the project-root reproducer. All knobs (MODEL_REPO, N_GPUS,
# BATCH_SIZE, SKIP_QA_TYPES, ...) pass through via env vars.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT_DIR}/run_inference.sh" "$@"
