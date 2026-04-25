"""CLI entrypoint for the OSWorld synthesis pipeline.

Invoke from ``envs/osgym`` with::

    python -m synthesis.cli --mode synthesize --domains chrome ...

Modes:
  synthesize  - Generate task examples + verifiers via LLM (no VM needed).
  verify      - Execute existing synthetic examples on VMs and validate.
  full        - Synthesize then verify.

The CLI constructs the two persistent stores (JSON ``SynthesisMemory`` and the
optional ChromaDB ``VectorDedupStore``) and threads them through both the
synthesis and verify stages so resumes and deduplication work across runs.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import signal
import sys
from typing import Optional

from .shared_memory import SynthesisMemory, VectorDedupStore, build_vector_store
from .task_creator import EXAMPLES_DIR, discover_domains, run_synthesize
from .verifier import run_verify, terminate_workers, verify_examples

logger = logging.getLogger("desktopenv.synthesis")

_is_terminating = False


# ═══════════════════════════════════════════════════════════════════════════
# Signal handling
# ═══════════════════════════════════════════════════════════════════════════


def _signal_handler(signum, frame):
    global _is_terminating
    if _is_terminating:
        return
    _is_terminating = True
    logger.info(f"Signal {signum} received - shutting down")
    terminate_workers()
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OSWorld synthesis pipeline")

    # Mode
    p.add_argument("--mode", choices=["synthesize", "verify", "full"], default="full",
                    help="Pipeline mode")
    p.add_argument("--list-domains", action="store_true", help="List domains and exit")

    # Synthesis args
    p.add_argument("--domains", nargs="*", default=[], help="Target domains (empty=all)")
    p.add_argument("--num-examples", type=int, default=3,
                    help="Fallback when --total-examples/--batch-size aren't set; "
                         "acts as single-batch size")
    p.add_argument("--total-examples", type=int, default=0,
                    help="Target total validated examples per domain (resumable across runs). "
                         "0 = fall back to --num-examples (single-batch mode).")
    p.add_argument("--batch-size", type=int, default=0,
                    help="Examples requested per LLM call. 0 = fall back to --num-examples.")
    p.add_argument("--max-empty-batches", type=int, default=3,
                    help="Stop a domain after this many consecutive batches yield no valid examples.")
    p.add_argument("--max-ref-examples", type=int, default=10,
                    help="Reference examples for LLM context")

    # LLM config (shared by synthesis and code-execution)
    p.add_argument("--model", type=str, required=False,
                    default="anthropic.claude-sonnet-4-20250514")
    p.add_argument("--provider", type=str, default="bedrock")
    p.add_argument("--endpoint", type=str, default="")

    # API server
    p.add_argument("--server-url", type=str, default="http://localhost:20000",
                    help="URL of the OSGym API server (main.py)")

    # Execution
    p.add_argument("--sleep-after-execution", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=15,
                    help="Max steps for code execution; also guides task complexity during synthesis")

    # Parallelism / verification mode
    p.add_argument("--num-workers", type=int, default=2,
                    help="Number of parallel worker processes for verification")
    p.add_argument("--verify-mode", choices=["run", "debug"], default="run",
                    help="'run' = multi-process workers (production), "
                         "'debug' = sequential in main process (debugger-friendly)")

    # Output
    p.add_argument("--output-dir", type=str, default="synthetic_evaluation_examples")
    p.add_argument("--log-level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Vector-DB dedup (opt-in — requires an OpenAI-compatible embeddings endpoint
    # served by vLLM `--task embed` or sglang `--is-embedding`)
    p.add_argument("--enable-dedup", action="store_true",
                    help="Enable vector-DB dedup against previously-solvable tasks")
    p.add_argument("--vector-db-path", type=str, default="",
                    help="Persistent ChromaDB path (default: <output-dir>/vector_db)")
    p.add_argument("--embedding-endpoint", type=str, default="",
                    help="OpenAI-compatible embeddings base URL, e.g. http://host:port "
                         "(the '/v1/embeddings' suffix is appended). Required with --enable-dedup.")
    p.add_argument("--embedding-model", type=str, default="",
                    help="Embedding model name to send to the endpoint. Required with --enable-dedup.")
    p.add_argument("--dedup-threshold", type=float, default=0.88,
                    help="Cosine-similarity threshold above which a new task is treated "
                         "as a duplicate (0-1). Default 0.88.")

    return p


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="\x1b[1;33m[%(asctime)s %(levelname)s %(module)s/%(lineno)d-%(processName)s]\x1b[0m %(message)s",
    )

    if args.list_domains:
        for d in discover_domains():
            n = len(glob.glob(os.path.join(EXAMPLES_DIR, d, "*.json")))
            print(f"  {d}: {n} examples")
        return

    # ---- Load synthesis memory (persists across rounds) ----
    memory = SynthesisMemory(args.output_dir).load()

    # ---- Optional: vector-DB dedup store ----
    vector_store: Optional[VectorDedupStore] = None
    if args.enable_dedup:
        if not args.embedding_endpoint or not args.embedding_model:
            logger.error(
                "--enable-dedup requires --embedding-endpoint and --embedding-model"
            )
            sys.exit(2)
        vector_db_path = args.vector_db_path or os.path.join(args.output_dir, "vector_db")
        vector_store = build_vector_store(
            persist_path=vector_db_path,
            endpoint=args.embedding_endpoint,
            model=args.embedding_model,
            similarity_threshold=args.dedup_threshold,
        )
        if vector_store is None:
            logger.warning("Continuing without dedup (vector store init failed)")

    # ---- Dispatch by mode ----
    # "full": verify each batch as soon as it's synthesized via a callback
    #         that returns the per-example verification results back to
    #         run_synthesize, which folds them into memory in one pass.
    # "synthesize": no callback — examples are persisted to disk and
    #         recorded as executable=True/solvable=None in memory.
    # "verify": run_verify loads the on-disk manifest and records outcomes
    #         to memory after verification completes.
    if args.mode == "full":
        def _verify_batch(domain: str, batch: list) -> list:
            logger.info(
                f"Interleaved verify: domain='{domain}' batch_size={len(batch)}"
            )
            return verify_examples(args, batch, vector_store)

        run_synthesize(args, memory, vector_store, on_batch_complete=_verify_batch)
    elif args.mode == "synthesize":
        run_synthesize(args, memory, vector_store)
    elif args.mode == "verify":
        run_verify(args, None, memory, vector_store)


if __name__ == "__main__":
    main()
