"""Verification: run a code-execution agent against synthesized tasks.

Workers POST to the OSGym API server (``main.py``) to allocate a VM, send a
single LLM-generated Python snippet, then ask the server to evaluate the
resulting state. Solvable examples are copied to ``solvable/`` and, when a
vector store is provided, their embeddings are added so future synthesis
rounds can dedup against them.

Memory recording happens *after* verification finishes — see
``run_verify`` (standalone) and ``run_synthesize`` (interleaved) for the
single-pass record + persist step.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import logging
import os
import re
import signal
import time
import traceback
from multiprocessing import Manager, Process, current_process
from typing import Any, Dict, List, Optional

import requests as http_requests

from mm_agents.utils.call_llm import call_llm_with_single_response
from mm_agents.utils.utils import encode_screenshot

from .prompts import CODE_EXEC_SYSTEM
from .shared_memory import SynthesisMemory, VectorDedupStore

logger = logging.getLogger("desktopenv.synthesis.verifier")

# Worker process registry — populated by _run_verify_parallel, consumed by
# cli.py's signal handler via terminate_workers().
_processes: List[Process] = []


# ═══════════════════════════════════════════════════════════════════════════
# HTTP helpers — thin wrappers around the OSGym API server (main.py)
# ═══════════════════════════════════════════════════════════════════════════


def _api_reset(server_url: str, task_config: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
    resp = http_requests.post(
        f"{server_url}/reset",
        json={"task_config": task_config, "timeout": timeout},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()  # {"screenshot": b64, "problem": str, "vm_id": int}


def _api_step(server_url: str, action: str, vm_id: int) -> Dict[str, Any]:
    resp = http_requests.post(
        f"{server_url}/step",
        json={"action": action, "vm_id": vm_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()  # {"screenshot": b64, "is_finish": bool, "reward": float}


def _api_evaluate(server_url: str, vm_id: int) -> Dict[str, Any]:
    resp = http_requests.post(
        f"{server_url}/evaluate",
        json={"vm_id": vm_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()  # {"reward": float}


def _api_shutdown(server_url: str, vm_id: int) -> None:
    try:
        http_requests.post(
            f"{server_url}/shutdown",
            json={"vm_id": vm_id},
            timeout=30,
        )
    except Exception as e:
        logger.warning(f"Failed to shutdown VM {vm_id}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Task execution via the API
# ═══════════════════════════════════════════════════════════════════════════


def _extract_code_from_response(raw: str) -> str:
    """Extract Python code from the ```python ... ``` fence in the LLM response."""
    match = re.search(r'```python\s*\n(.*?)```', raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*\n(.*?)```', raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    logger.warning("No ```python fence found in LLM response, using raw output")
    return raw.strip()


def run_code_task(
    server_url: str,
    example: Dict[str, Any],
    sleep_after_execution: float,
    llm_config: Dict[str, Any],
    result_dir: str,
) -> Dict[str, Any]:
    """Execute one example via a single LLM-generated code snippet through the API."""
    proc = current_process().name

    parts = [f"Task: {example['instruction']}"]
    eval_expr = (example.get("evaluator") or {}).get("eval", "")
    if eval_expr:
        parts.append(
            f"\nVerifier expression (this is how your result will be checked):\n"
            f"  {eval_expr}\n"
            f"Make sure your actions produce the exact state this verifier expects."
        )
    task_context = "\n".join(parts)
    

    reset_data = _api_reset(server_url, example)
    vm_id = reset_data["vm_id"]
    reward = 0

    try:
        screenshot = base64.b64decode(reset_data["screenshot"])

        messages = [
            {"role": "system", "content": CODE_EXEC_SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": encode_screenshot(screenshot)}},
                {"type": "text", "text": task_context},
            ]},
        ]
        
        breakpoint()

        raw = call_llm_with_single_response(
            messages=messages, llm_config=llm_config,
            max_tokens=8000, temperature=0.7,
        )
        code = _extract_code_from_response(raw)

        ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
        logger.info(f"[{proc}][code] generated: {code}")

        step_data = _api_step(server_url, code, vm_id)
        time.sleep(sleep_after_execution)

        screenshot_bytes = base64.b64decode(step_data["screenshot"])
        with open(os.path.join(result_dir, f"step_0_{ts}.png"), "wb") as fp:
            fp.write(screenshot_bytes)

        time.sleep(5)
        eval_data = _api_evaluate(server_url, vm_id)
        reward = eval_data["reward"]
        logger.info(f"[{proc}][code] score={reward:.2f} for {example['id']}")

        trajectory = [{"step": 0, "timestamp": ts, "code": code}]
        with open(os.path.join(result_dir, "trajectory.json"), "w") as fp:
            json.dump(trajectory, fp, indent=2)
        with open(os.path.join(result_dir, "result.txt"), "w") as fp:
            fp.write(f"{reward}\n")
        return {"id": example["id"], "mode": "code", "score": reward, "steps": 1}

    finally:
        _api_shutdown(server_url, vm_id)


# ═══════════════════════════════════════════════════════════════════════════
# Worker process
# ═══════════════════════════════════════════════════════════════════════════


def worker(task_queue, args: argparse.Namespace, shared_results: list):
    """Worker that pulls examples from the queue and runs code execution."""
    proc = current_process().name
    llm_config = {"model": args.model, "provider": args.provider, "endpoint": args.endpoint}

    while True:
        try:
            example = task_queue.get(timeout=5)
        except Exception:
            break

        try:
            result_dir = os.path.join(
                args.output_dir, example.get("_domain", "unknown"),
                "trajectories", example["id"], "code",
            )
            os.makedirs(result_dir, exist_ok=True)

            res = run_code_task(
                args.server_url, example,
                args.sleep_after_execution,
                llm_config, result_dir,
            )
            shared_results.append(res)

        except KeyboardInterrupt:
            logger.warning(f"[{proc}] KeyboardInterrupt")
            break
        except Exception as e:
            logger.error(f"[{proc}] Error on code/{example['id']}: {e}")
            logger.error(traceback.format_exc())
            shared_results.append({
                "id": example["id"], "mode": "code",
                "score": 0.0, "error": str(e),
            })

    logger.info(f"[{proc}] Worker finished")


# ═══════════════════════════════════════════════════════════════════════════
# Loading + result post-processing (no memory writes here)
# ═══════════════════════════════════════════════════════════════════════════


def _load_synthetic_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Load examples from the output dir (generated by synthesize stage)."""
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"No manifest at {manifest_path}. Run --mode synthesize first.")
    with open(manifest_path) as f:
        manifest = json.load(f)
    examples = []
    for domain, ids in manifest.items():
        for eid in ids:
            path = os.path.join(args.output_dir, domain, f"{eid}.json")
            if os.path.isfile(path):
                with open(path) as f:
                    ex = json.load(f)
                ex.setdefault("_domain", domain)
                examples.append(ex)
            else:
                logger.warning(f"Example file not found: {path}")
    logger.info(f"Loaded {len(examples)} synthetic examples from {manifest_path}")
    return examples


def _process_verify_results(
    results: List[Dict[str, Any]],
    examples: List[Dict[str, Any]],
    args: argparse.Namespace,
    vector_store: Optional[VectorDedupStore] = None,
) -> List[Dict[str, Any]]:
    """Merge raw results onto disk, copy solvables, push to vector store.

    Does NOT write to ``SynthesisMemory`` — the caller is responsible for
    that, and must do so AFTER this function returns so verification status
    is finalized before the memory record is written.
    """
    agg_path = os.path.join(args.output_dir, "verification_results.json")
    prior_results: List[Dict[str, Any]] = []
    if os.path.isfile(agg_path):
        try:
            with open(agg_path) as f:
                prior_results = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {agg_path}: {e} — starting fresh")
    agg_by_id: Dict[str, Dict[str, Any]] = {
        r.get("id"): r for r in prior_results if r.get("id")
    }
    for r in results:
        if r.get("id"):
            agg_by_id[r["id"]] = r
    with open(agg_path, "w") as f:
        json.dump(list(agg_by_id.values()), f, indent=2)

    code_scores = [r["score"] for r in results if "error" not in r]
    avg = sum(code_scores) / max(len(code_scores), 1)
    logger.info(f"Code avg={avg:.3f} ({len(code_scores)} tasks in this batch)")

    by_id = {r["id"]: r for r in results}
    solvable_ids: List[str] = []
    unsolvable_ids: List[str] = []
    errored_ids: List[str] = []
    for eid, r in by_id.items():
        if "error" in r:
            errored_ids.append(eid)
            logger.info(f"ERRORED {eid}: {r['error']}")
        elif r.get("score", 0) > 0:
            solvable_ids.append(eid)
        else:
            unsolvable_ids.append(eid)
            logger.info(f"UNSOLVABLE {eid}: code_score={r.get('score', 'missing')}")

    # Copy solvables to solvable/ and push into vector store
    solvable_dir = os.path.join(args.output_dir, "solvable")
    os.makedirs(solvable_dir, exist_ok=True)
    batch_solvable_by_domain: Dict[str, List[str]] = {}
    added_to_vector = 0
    for ex in examples:
        if ex["id"] not in solvable_ids:
            continue
        domain = ex.get("_domain", "unknown")
        domain_dir = os.path.join(solvable_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        with open(os.path.join(domain_dir, f"{ex['id']}.json"), "w") as f:
            json.dump(ex, f, indent=2)
        batch_solvable_by_domain.setdefault(domain, []).append(ex["id"])
        if vector_store is not None and vector_store.add_solvable(ex, domain):
            added_to_vector += 1
    if vector_store is not None and solvable_ids:
        logger.info(
            f"[vector-dedup] added {added_to_vector}/{len(solvable_ids)} "
            f"solvable example(s) to vector store"
        )

    # Merge solvable manifest with existing entries
    solvable_manifest_path = os.path.join(solvable_dir, "manifest.json")
    existing_manifest: Dict[str, List[str]] = {}
    if os.path.isfile(solvable_manifest_path):
        try:
            with open(solvable_manifest_path) as f:
                existing_manifest = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {solvable_manifest_path}: {e} — starting fresh")
    for domain, ids in batch_solvable_by_domain.items():
        merged = set(existing_manifest.get(domain, [])) | set(ids)
        existing_manifest[domain] = sorted(merged)
    with open(solvable_manifest_path, "w") as f:
        json.dump(existing_manifest, f, indent=2)

    logger.info(
        f"Batch verification: {len(solvable_ids)} solvable, "
        f"{len(unsolvable_ids)} unsolvable, {len(errored_ids)} errored "
        f"(will retry) out of {len(by_id)} total"
    )
    logger.info(f"Solvable examples saved to {solvable_dir}")
    logger.info(f"Aggregate results at {agg_path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Verify dispatchers
# ═══════════════════════════════════════════════════════════════════════════


def _run_verify_parallel(
    args: argparse.Namespace,
    examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Multi-process verification: workers pull examples from a queue."""
    global _processes

    total_workers = args.num_workers
    logger.info(f"Verification (parallel): {len(examples)} examples across {total_workers} workers")

    manager = Manager()
    shared_results = manager.list()
    task_queue = manager.Queue()

    for ex in examples:
        task_queue.put(ex)

    _processes = []
    for pidx in range(total_workers):
        p = Process(
            target=worker,
            args=(task_queue, args, shared_results),
            name=f"SynthWorker-{pidx}",
            daemon=True,
        )
        p.start()
        _processes.append(p)
        logger.info(f"Started {p.name} (PID {p.pid})")

    try:
        while True:
            if task_queue.empty():
                break
            if not any(p.is_alive() for p in _processes):
                logger.error("All workers died")
                break
            time.sleep(5)
        for p in _processes:
            p.join(timeout=60)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt - terminating workers")
    finally:
        terminate_workers()
        try:
            manager.shutdown()
        except Exception:
            pass

    return list(shared_results)


def _run_verify_sequential(
    args: argparse.Namespace,
    examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sequential verification in the main process (debugger-friendly)."""
    logger.info(f"Verification (sequential/debug): {len(examples)} examples")

    llm_config = {"model": args.model, "provider": args.provider, "endpoint": args.endpoint}
    results: List[Dict[str, Any]] = []

    for ex_idx, example in enumerate(examples):
        logger.info(
            f"[{ex_idx + 1}/{len(examples)}] Running code for {example['id']} "
            f"({example.get('instruction', '')})"
        )
        result_dir = os.path.join(
            args.output_dir, example.get("_domain", "unknown"),
            "trajectories", example["id"], "code",
        )
        os.makedirs(result_dir, exist_ok=True)

        try:
            res = run_code_task(
                args.server_url, example,
                args.sleep_after_execution,
                llm_config, result_dir,
            )
            results.append(res)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt — stopping verification")
            return results
        except Exception as e:
            logger.error(f"Error on code/{example['id']}: {e}")
            logger.error(traceback.format_exc())
            results.append({
                "id": example["id"], "mode": "code",
                "score": 0.0, "error": str(e),
            })

    return results


def verify_examples(
    args: argparse.Namespace,
    examples: List[Dict[str, Any]],
    vector_store: Optional[VectorDedupStore] = None,
) -> List[Dict[str, Any]]:
    """Run verification on a preloaded batch and post-process results.

    Returns the raw per-example result dicts. Memory is intentionally NOT
    touched here — callers (``run_synthesize`` and ``run_verify``) record
    the final status to memory after this call returns so verification
    completes before any memory write.
    """
    if not examples:
        return []
    verify_mode = getattr(args, "verify_mode", "run")
    if verify_mode == "debug":
        results = _run_verify_sequential(args, examples)
    else:
        results = _run_verify_parallel(args, examples)
    return _process_verify_results(results, examples, args, vector_store)


def run_verify(
    args: argparse.Namespace,
    examples: Optional[List[Dict[str, Any]]] = None,
    memory: Optional[SynthesisMemory] = None,
    vector_store: Optional[VectorDedupStore] = None,
) -> List[Dict[str, Any]]:
    """Standalone verify entrypoint: load from disk, skip already-tested,
    verify the rest, then record outcomes to memory and persist.
    """
    if examples is None:
        examples = _load_synthetic_examples(args)
    if not examples:
        logger.error("No examples to verify")
        return []

    if memory is not None:
        already_tested = {
            e["id"] for e in memory.entries
            if e.get("solvable") is not None
        }
        before = len(examples)
        examples = [ex for ex in examples if ex["id"] not in already_tested]
        if before != len(examples):
            logger.info(
                f"Skipped {before - len(examples)} already-verified examples "
                f"({len(examples)} remaining)"
            )
    if not examples:
        logger.info("All examples already verified — nothing to do")
        return []

    results = verify_examples(args, examples, vector_store)

    # Memory record + persist AFTER verification
    if memory is not None:
        results_by_id = {r["id"]: r for r in results if r.get("id")}
        for ex in examples:
            r = results_by_id.get(ex["id"])
            if r is None:
                continue
            solvable = None if "error" in r else (r.get("score", 0) > 0)
            memory.record(
                example=ex,
                domain=ex.get("_domain", "unknown"),
                code_result=r,
                executable=True,
                solvable=solvable,
            )
        memory.save()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Process cleanup
# ═══════════════════════════════════════════════════════════════════════════


def terminate_workers(timeout: int = 10) -> None:
    """Terminate any worker processes spawned by this module.

    Called from the CLI signal handler so SIGINT / SIGTERM reliably tears
    down the multiprocessing pool before the main process exits.
    """
    alive = []
    for p in _processes:
        if p is not None and p.is_alive():
            try:
                p.terminate()
                alive.append(p)
            except Exception:
                pass
    t0 = time.time()
    while alive and time.time() - t0 < timeout:
        alive = [p for p in alive if p.is_alive()]
        if not alive:
            break
        time.sleep(0.5)
    for p in alive:
        try:
            os.kill(p.pid, signal.SIGKILL)
        except Exception:
            pass
