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

TOTAL_EXAMPLES=500     # target total validated tasks per domain (resumable)
BATCH_SIZE=10          # tasks requested per LLM call; prior-batch instructions are surfaced to the LLM to avoid repeats
MAX_EMPTY_BATCHES=3    # give up on a domain after this many consecutive no-valid-yield batches
MAX_REF_EXAMPLES=2

# MODEL="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
# PROVIDER="bedrock"
# ENDPOINT="http://localhost:7778"

MODEL="Qwen/Qwen3-VL-8B-Instruct"
PROVIDER="sglang"
ENDPOINT="http://207.211.167.191:9000/v1"

# Per-model output/KB layout. Synthesized examples land in
# ${OUTPUT_ROOT}/${MODEL_TAG}/, the vector-DB lands in ${OUTPUT_ROOT}/kb/${MODEL_TAG}/,
# so multiple model variants can coexist under a single root without colliding.
# Override either with the CLI flags --output-dir / --vector-db-path.
OUTPUT_ROOT="synthetic_evaluation_examples"
MODEL_TAG="qwen3-vl-8b"

SERVER_URL="http://localhost:20000"
synthesize_workers=5     # threads for parallel synthesis (LLM calls; I/O-bound)
verification_workers=1   # processes for parallel verification (one VM per worker)
MAX_STEPS=15
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_TAG}"
SYNTHESIZE_MODE="parallel"     # "sequential" = main thread; "parallel" = thread-pool over domains
VERIFICATION_MODE="sequential"   # "sequential" = main process (debugger-friendly); "parallel" = multi-process workers

# ── Vector-DB dedup (opt-in) ─────────────────────────────────────────────
# Dedup newly-generated tasks against previously-solvable ones using a local
# ChromaDB + an OpenAI-compatible /v1/embeddings endpoint served by either
# vLLM (--task embed) or sglang (--is-embedding).
ENABLE_DEDUP=1                        # set to 1 to turn on
EMBEDDING_ENDPOINT="http://ec2-44-249-196-60.us-west-2.compute.amazonaws.com:30000/v1"                 # e.g. http://localhost:9001
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"                    # e.g. Qwen/Qwen3-Embedding-0.6B
DEDUP_THRESHOLD=0.85                   # cosine similarity cutoff (stricter dedup)
VECTOR_DB_PATH="${OUTPUT_ROOT}/kb/${MODEL_TAG}"   # leave empty to fall back to ${OUTPUT_DIR}/vector_db

dedup_args=()
if [[ "${ENABLE_DEDUP}" == "1" ]]; then
    dedup_args+=(--enable-dedup \
                 --embedding-endpoint "${EMBEDDING_ENDPOINT}" \
                 --embedding-model "${EMBEDDING_MODEL}" \
                 --dedup-threshold "${DEDUP_THRESHOLD}")
    if [[ -n "${VECTOR_DB_PATH}" ]]; then
        dedup_args+=(--vector-db-path "${VECTOR_DB_PATH}")
    fi
fi

# ── Phase 1: Synthesize tasks across every domain in a single run ────────
# Pass the full DOMAINS list to --domains (argparse nargs="*"). This lets the
# in-process synthesize_mode dispatcher (sequential vs parallel) decide how to
# fan out across domains, instead of the shell forking a fresh Python process
# per domain.
echo "--- Synthesizing: ${DOMAINS[*]} (target=${TOTAL_EXAMPLES}, batch=${BATCH_SIZE}) ---"
python -m synthesis.cli \
    --mode synthesize \
    --domains "${DOMAINS[@]}" \
    --total-examples "${TOTAL_EXAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --max-empty-batches "${MAX_EMPTY_BATCHES}" \
    --max-ref-examples "${MAX_REF_EXAMPLES}" \
    --model "${MODEL}" \
    --provider "${PROVIDER}" \
    --endpoint "${ENDPOINT}" \
    --max-steps "${MAX_STEPS}" \
    --synthesize-workers "${synthesize_workers}" \
    --verification-workers "${verification_workers}" \
    --synthesize-mode "${SYNTHESIZE_MODE}" \
    --verification-mode "${VERIFICATION_MODE}" \
    --output-dir "${OUTPUT_DIR}" \
    ${dedup_args[@]+"${dedup_args[@]}"}

echo "========== Done =========="
