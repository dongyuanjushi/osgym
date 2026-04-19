#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────
DOMAINS=(
    "chrome"
    "gimp"
    "libreoffice_calc"
    "libreoffice_impress"
    "libreoffice_writer"
    "multi_apps"
    "os"
    "thunderbird"
    "vlc"
    "vs_code"
)

NUM_EXAMPLES=10
MAX_REF_EXAMPLES=10
MODEL="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
PROVIDER="bedrock"
ENDPOINT="http://localhost:7778"
SERVER_URL="http://localhost:20000"
NUM_WORKERS=2
MAX_STEPS=15
OUTPUT_DIR="synthetic_evaluation_examples"
VERIFY_MODE="debug"   # "run" = multi-process (production), "debug" = sequential (debugger-friendly)

# ── Phase 1: Synthesize tasks for each domain ────────────────────────────
echo "========== Phase 1: Synthesis =========="
for domain in "${DOMAINS[@]}"; do
    echo "--- Synthesizing: ${domain} ---"
    python synthesis_pipeline.py \
        --mode synthesize \
        --domains "${domain}" \
        --num-examples "${NUM_EXAMPLES}" \
        --max-ref-examples "${MAX_REF_EXAMPLES}" \
        --model "${MODEL}" \
        --provider "${PROVIDER}" \
        --endpoint "${ENDPOINT}" \
        --max-steps "${MAX_STEPS}" \
        --output-dir "${OUTPUT_DIR}"
done

# ── Phase 2: Verify all synthesized examples ─────────────────────────────
echo "========== Phase 2: Verification =========="
for domain in "${DOMAINS[@]}"; do
    echo "--- Verifying: ${domain} ---"
    python synthesis_pipeline.py \
        --mode verify \
        --domains "${domain}" \
        --model "${MODEL}" \
        --provider "${PROVIDER}" \
        --endpoint "${ENDPOINT}" \
        --server-url "${SERVER_URL}" \
        --num-workers "${NUM_WORKERS}" \
        --max-steps "${MAX_STEPS}" \
        --output-dir "${OUTPUT_DIR}" \
        --verify-mode "${VERIFY_MODE}"
done

echo "========== Done =========="
