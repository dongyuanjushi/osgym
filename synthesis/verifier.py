"""Verification: GUI-only execution of synthesized tasks, gated by an LLM
relevance check on the evaluator.

Each example is verified in two passes:

* **Pass 1 — relevance.** A small LLM call sees the instruction together with
  the evaluator's ``postconfig`` and ``eval`` expression and answers whether
  the evaluator actually inspects the substantive state the instruction
  names. The prompt (``RELEVANCE_CHECK_SYSTEM``) flags the common synthesis
  failure mode where the eval checks a superficial proxy (e.g. file
  existence / image dimensions) while the instruction asks for a content
  change (e.g. apply a filter, add a stroke). Examples ruled irrelevant
  short-circuit: no VM is allocated, no agent is launched, and the result is
  recorded as not-solvable with the rationale attached.

* **Pass 2 — execute.** For relevance-passing examples the OSGym API server
  (``main.py``) is asked to allocate a VM. ``Qwen35VLAgent.predict`` then
  drives the GUI for up to ``max_steps`` predict→/step iterations, mirroring
  ``lib_run_single.run_single_example``. After the loop ``/evaluate`` returns
  the reward.

Memory recording happens AFTER verification finishes — see ``run_verify``
(standalone) and ``run_synthesize`` (interleaved) for the single-pass record
+ persist step.
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
from typing import Any, Dict, List, Optional, Tuple

import requests as http_requests

from mm_agents.qwen35_vl import Qwen35VLAgent
from mm_agents.utils.call_llm import call_llm_with_single_response

from .prompts import RELEVANCE_CHECK_SYSTEM
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


def _build_task_context(example: Dict[str, Any]) -> str:
    """Render the task instruction + verifier expression as user-facing text."""
    parts = [f"Task: {example['instruction']}"]
    eval_expr = (example.get("evaluator") or {}).get("eval", "")
    if eval_expr:
        parts.append(
            f"\nVerifier expression (this is how your result will be checked):\n"
            f"  {eval_expr}\n"
            f"Make sure your action produces the exact state this verifier expects."
        )
    return "\n".join(parts)


# Relevance-check responses are JSON objects. Keep the parser tolerant of
# LLMs that wrap them in a ```json fence despite the prompt.
_RELEVANCE_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def _check_evaluator_relevance(
    example: Dict[str, Any],
    llm_config: Dict[str, Any],
    runtime_logger: logging.Logger,
) -> Tuple[bool, str]:
    """Pass 1: ask the LLM whether the evaluator captures the instruction's intent.

    Returns ``(relevant, reason)``. Defaults to ``relevant=True`` on parse
    failure so a flaky LLM does not silently kill an entire batch — the
    expensive VM step will still surface real bugs, and the parse failure is
    logged for follow-up.
    """
    instruction = example.get("instruction", "")
    evaluator = example.get("evaluator") or {}
    postconfig = evaluator.get("postconfig") or []
    eval_expr = evaluator.get("eval", "")

    user_text = (
        f"## Instruction\n{instruction}\n\n"
        f"## Evaluator postconfig\n"
        f"{json.dumps(postconfig, indent=2) if postconfig else '[]'}\n\n"
        f"## Evaluator eval\n{eval_expr or '(empty)'}\n\n"
        "Decide whether this evaluator inspects the substantive state the "
        "instruction names. Reply with the JSON object specified in the "
        "system prompt and nothing else."
    )

    messages = [
        {"role": "system", "content": RELEVANCE_CHECK_SYSTEM},
        {"role": "user", "content": user_text},
    ]
    raw = call_llm_with_single_response(
        messages=messages, llm_config=llm_config,
        max_tokens=400, temperature=0.0,
    )

    match = _RELEVANCE_JSON_RE.search(raw or "")
    if not match:
        runtime_logger.warning(
            f"relevance response lacks JSON object; defaulting to relevant. raw={raw!r}"
        )
        return True, "relevance parse failed (no JSON object); defaulted to relevant"

    try:
        decision = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        runtime_logger.warning(
            f"relevance JSON invalid ({e}); defaulting to relevant. raw={raw!r}"
        )
        return True, f"relevance JSON invalid ({e}); defaulted to relevant"

    relevant_raw = decision.get("relevant")
    if isinstance(relevant_raw, bool):
        relevant = relevant_raw
    elif isinstance(relevant_raw, str):
        relevant = relevant_raw.strip().lower() in {"true", "yes", "1"}
    else:
        runtime_logger.warning(
            f"relevance returned non-bool 'relevant' field {relevant_raw!r}; "
            f"defaulting to relevant"
        )
        return True, f"relevance returned 'relevant'={relevant_raw!r}; defaulted to relevant"

    reason = str(decision.get("reason", "")).strip()
    return relevant, reason


def _run_gui_action(
    server_url: str,
    vm_id: int,
    screenshot: bytes,
    task_context: str,
    sleep_after_execution: float,
    llm_config: Dict[str, Any],
    result_dir: str,
    screen_size: Tuple[int, int],
    runtime_logger: logging.Logger,
    max_steps: int,
) -> Dict[str, Any]:
    """Drive Qwen35VLAgent for up to ``max_steps`` predict→/step iterations.

    Loop ends when the env signals done, the agent emits a terminate
    sentinel (``DONE``/``FAIL``), or the step cap is reached. The pattern
    mirrors ``lib_run_single.run_single_example`` but uses the OSGym HTTP
    API instead of an in-process DesktopEnv.
    """
    proc = current_process().name
    agent = Qwen35VLAgent(
        screen_size=screen_size,
        approach="default",
        policy_model=llm_config["model"],
        policy_model_provider=llm_config["provider"],
        policy_model_endpoint=llm_config["endpoint"],
        logger=runtime_logger,
    )
    agent.reset(result_dir)

    # Save the post-reset screenshot as step_0 so the trajectory has a
    # complete visual record from t=0.
    with open(os.path.join(result_dir, "step_0_initial.png"), "wb") as fp:
        fp.write(screenshot)

    obs: Dict[str, Any] = {"screenshot": screenshot}
    trajectory: List[Dict[str, Any]] = []
    done = False
    step_idx = 0
    error: Optional[str] = None

    while not done and step_idx < max_steps:
        try:
            observation, thought, action_code = agent.predict(task_context, obs)
        except Exception as e:
            runtime_logger.error(
                f"[{proc}][gui] predict failed at step {step_idx + 1}: {e}"
            )
            runtime_logger.error(traceback.format_exc())
            error = f"predict: {e}"
            break

        ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
        runtime_logger.info(
            f"[{proc}][gui] step={step_idx + 1}/{max_steps} action={action_code!r}"
        )

        # Qwen35VLAgent.process_tool_call emits sentinel strings for the
        # terminate / wait / fail actions. /step doesn't drive these — we
        # interpret them client-side.
        try:
            step_data = _api_step(server_url, action_code, vm_id)
        except Exception as e:
            runtime_logger.error(f"[{proc}][gui] /step failed: {e}")
            error = f"step: {e}"
            break

        time.sleep(sleep_after_execution)

        done = bool(step_data.get("is_finish"))
        step_reward = float(step_data.get("reward", 0.0))
        screenshot_b64 = step_data["screenshot"]

        screenshot_path = os.path.join(result_dir, f"step_{step_idx + 1}_{ts}.png")
        with open(screenshot_path, "wb") as fp:
            fp.write(base64.b64decode(screenshot_b64))

        trajectory.append({
            "step_num": step_idx + 1,
            "action_timestamp": ts,
            "thought": thought,
            "observation": observation,
            "action": action_code,
            "reward": step_reward,
            "done": done,
            "screenshot_file": os.path.basename(screenshot_path),
        })

        # Qwen35VLAgent.process_image normalizes either bytes or a base64
        # string, so we can pass the /step screenshot through directly.
        obs = {"screenshot": screenshot_b64}
        step_idx += 1

        if done:
            break

    # Final evaluation regardless of how the loop exited (mirrors
    # lib_run_single.run_single_example).
    time.sleep(5)
    reward = 0.0
    try:
        eval_data = _api_evaluate(server_url, vm_id)
        reward = float(eval_data["reward"])
    except Exception as e:
        runtime_logger.warning(f"[{proc}][gui] /evaluate failed: {e}")
        if error is None:
            error = f"evaluate: {e}"

    runtime_logger.info(
        f"[{proc}][gui] score={reward:.2f} steps={step_idx} done={done}"
    )

    with open(os.path.join(result_dir, "trajectory.json"), "w") as fp:
        json.dump({
            "mode": "gui",
            "max_steps": max_steps,
            "steps": trajectory,
        }, fp, indent=2)
    with open(os.path.join(result_dir, "result.txt"), "w") as fp:
        fp.write(f"{reward}\n")

    result: Dict[str, Any] = {
        "mode": "gui",
        "score": reward,
        "steps": step_idx,
        "done": done,
    }
    if error is not None:
        result["error"] = error
    return result


def _persist_relevance_skip(
    result_dir: str, reason: str
) -> None:
    """Write a minimal trajectory + result so the on-disk verify dir reflects
    that this example was rejected before the VM stage.
    """
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "trajectory.json"), "w") as fp:
        json.dump({
            "mode": "relevance_skip",
            "relevance_reason": reason,
            "steps": [],
        }, fp, indent=2)
    with open(os.path.join(result_dir, "result.txt"), "w") as fp:
        fp.write("0.0\n")


def run_verify_example(
    server_url: str,
    example: Dict[str, Any],
    sleep_after_execution: float,
    llm_config: Dict[str, Any],
    result_dir: str,
    screen_size: Tuple[int, int] = (1920, 1080),
    runtime_logger: Optional[logging.Logger] = None,
    max_steps: int = 15,
) -> Dict[str, Any]:
    """Two-pass verification of a single example.

    Pass 1 (``_check_evaluator_relevance``) decides whether the evaluator
    actually inspects what the instruction asks for. An irrelevant example
    is reported as not-solvable WITHOUT allocating a VM — this is the cheap
    filter that catches the common synthesis failure where the eval is a
    superficial proxy (file size, mere existence) for a content/state change.

    Pass 2 (``_run_gui_action``) only runs when relevance passes: allocate a
    VM via ``/reset``, drive ``Qwen35VLAgent`` for up to ``max_steps``
    predict→/step iterations, then call ``/evaluate``.
    """
    proc = current_process().name
    runtime_logger = runtime_logger or logger

    relevant, relevance_reason = _check_evaluator_relevance(
        example, llm_config, runtime_logger,
    )
    runtime_logger.info(
        f"[{proc}][relevance] relevant={relevant} reason={relevance_reason!r}"
    )

    if not relevant:
        _persist_relevance_skip(result_dir, relevance_reason)
        return {
            "id": example["id"],
            "mode": "relevance_skip",
            "score": 0.0,
            "steps": 0,
            "relevant": False,
            "relevance_reason": relevance_reason,
        }

    task_context = _build_task_context(example)
    reset_data = _api_reset(server_url, example)
    vm_id = reset_data["vm_id"]

    try:
        screenshot = base64.b64decode(reset_data["screenshot"])
        res = _run_gui_action(
            server_url, vm_id, screenshot, task_context,
            sleep_after_execution, llm_config, result_dir,
            screen_size=screen_size,
            runtime_logger=runtime_logger,
            max_steps=max_steps,
        )
        res["id"] = example["id"]
        res["relevant"] = True
        res["relevance_reason"] = relevance_reason
        return res

    finally:
        _api_shutdown(server_url, vm_id)


# ═══════════════════════════════════════════════════════════════════════════
# Worker process
# ═══════════════════════════════════════════════════════════════════════════


def worker(
    task_queue,
    args: argparse.Namespace,
    shared_results: list,
    max_steps: int,
    screen_size: Tuple[int, int],
):
    """Worker that pulls examples from the queue and runs one verification.

    The ``max_steps`` and ``screen_size`` parameters are explicit (not read
    from ``args``) so the synthesize→verify callback path can pass them
    deliberately.
    """
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
                "trajectories", example["id"], "verify",
            )
            os.makedirs(result_dir, exist_ok=True)

            res = run_verify_example(
                args.server_url, example,
                args.sleep_after_execution,
                llm_config, result_dir,
                screen_size=screen_size,
                runtime_logger=logger,
                max_steps=max_steps,
            )
            shared_results.append(res)

        except KeyboardInterrupt:
            logger.warning(f"[{proc}] KeyboardInterrupt")
            break
        except Exception as e:
            logger.error(f"[{proc}] Error verifying {example['id']}: {e}")
            logger.error(traceback.format_exc())
            shared_results.append({
                "id": example["id"], "mode": "unknown",
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

    ``relevance_skip`` results are persisted alongside the rest so callers
    can see why an example was rejected without ever hitting the VM, but
    they are NOT copied into ``solvable/`` and NOT added to the vector
    store (a relevance failure means the task wasn't actually checked).
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

    scores = [r["score"] for r in results if "error" not in r]
    avg = sum(scores) / max(len(scores), 1)
    mode_counts: Dict[str, int] = {}
    for r in results:
        mode_counts[r.get("mode", "unknown")] = mode_counts.get(r.get("mode", "unknown"), 0) + 1
    mode_summary = ", ".join(f"{m}={n}" for m, n in sorted(mode_counts.items()))
    logger.info(
        f"verify avg={avg:.3f} ({len(scores)} tasks in this batch; modes: {mode_summary})"
    )

    by_id = {r["id"]: r for r in results}
    solvable_ids: List[str] = []
    unsolvable_ids: List[str] = []
    irrelevant_ids: List[str] = []
    errored_ids: List[str] = []
    for eid, r in by_id.items():
        if "error" in r:
            errored_ids.append(eid)
            logger.info(f"ERRORED {eid}: {r['error']}")
        elif r.get("mode") == "relevance_skip":
            irrelevant_ids.append(eid)
            logger.info(
                f"IRRELEVANT {eid}: evaluator does not match instruction "
                f"({r.get('relevance_reason', 'no reason')})"
            )
        elif r.get("score", 0) > 0:
            solvable_ids.append(eid)
        else:
            unsolvable_ids.append(eid)
            logger.info(f"UNSOLVABLE {eid}: score={r.get('score', 'missing')}")

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
        f"{len(unsolvable_ids)} unsolvable, "
        f"{len(irrelevant_ids)} irrelevant (skipped), "
        f"{len(errored_ids)} errored (will retry) "
        f"out of {len(by_id)} total"
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
    *,
    max_steps: int,
    screen_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """Multi-process verification: workers pull examples from a queue.

    ``max_steps`` and ``screen_size`` are passed explicitly to each worker
    process — they drive the multi-step gui loop and coordinate scaling
    respectively, and the chain from cli.py's _verify_batch callback all
    the way down should make these values visible at every layer.
    """
    global _processes

    verification_workers = getattr(args, "verification_workers", 1)
    logger.info(
        f"Verification (parallel): {len(examples)} examples across "
        f"{verification_workers} workers (max_steps={max_steps}, screen_size={screen_size})"
    )

    manager = Manager()
    shared_results = manager.list()
    task_queue = manager.Queue()

    for ex in examples:
        task_queue.put(ex)

    _processes = []
    for pidx in range(verification_workers):
        p = Process(
            target=worker,
            args=(task_queue, args, shared_results, max_steps, screen_size),
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
    *,
    max_steps: int,
    screen_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """Sequential verification in the main process (debugger-friendly).

    ``max_steps`` and ``screen_size`` are explicit so the synthesize→verify
    callback path threads them deliberately rather than reading from args.
    """
    logger.info(
        f"Verification (sequential/debug): {len(examples)} examples "
        f"(max_steps={max_steps}, screen_size={screen_size})"
    )

    llm_config = {"model": args.model, "provider": args.provider, "endpoint": args.endpoint}
    results: List[Dict[str, Any]] = []

    for ex_idx, example in enumerate(examples):
        logger.info(
            f"[{ex_idx + 1}/{len(examples)}] Verifying {example['id']} "
            f"({example.get('instruction', '')})"
        )
        result_dir = os.path.join(
            args.output_dir, example.get("_domain", "unknown"),
            "trajectories", example["id"], "verify",
        )
        os.makedirs(result_dir, exist_ok=True)

        try:
            res = run_verify_example(
                args.server_url, example,
                args.sleep_after_execution,
                llm_config, result_dir,
                screen_size=screen_size,
                runtime_logger=logger,
                max_steps=max_steps,
            )
            results.append(res)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt — stopping verification")
            return results
        except Exception as e:
            logger.error(f"Error verifying {example['id']}: {e}")
            logger.error(traceback.format_exc())
            results.append({
                "id": example["id"], "mode": "unknown",
                "score": 0.0, "error": str(e),
            })

    return results


def verify_examples(
    args: argparse.Namespace,
    examples: List[Dict[str, Any]],
    vector_store: Optional[VectorDedupStore] = None,
    *,
    max_steps: Optional[int] = None,
    screen_size: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Run verification on a preloaded batch and post-process results.

    ``max_steps`` and ``screen_size`` are required by the gui-mode loop and
    coordinate scaling. Callers (cli.py's interleaved ``_verify_batch`` and
    the standalone ``run_verify``) should pass them explicitly. If left as
    ``None`` they fall back to the matching ``args`` attributes — this
    fallback exists so the function stays usable from one-off scripts that
    just hand it ``args``.

    Returns the raw per-example result dicts. Memory is intentionally NOT
    touched here — callers (``run_synthesize`` and ``run_verify``) record
    the final status to memory after this call returns so verification
    completes before any memory write.
    """
    if not examples:
        return []
    if max_steps is None:
        max_steps = getattr(args, "max_steps", 15)
    if screen_size is None:
        screen_size = (
            getattr(args, "screen_width", 1920),
            getattr(args, "screen_height", 1080),
        )

    verification_mode = getattr(args, "verification_mode", "sequential")
    if verification_mode == "sequential":
        results = _run_verify_sequential(
            args, examples, max_steps=max_steps, screen_size=screen_size,
        )
    else:
        results = _run_verify_parallel(
            args, examples, max_steps=max_steps, screen_size=screen_size,
        )
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

    max_steps = getattr(args, "max_steps", 15)
    screen_size = (
        getattr(args, "screen_width", 1920),
        getattr(args, "screen_height", 1080),
    )
    results = verify_examples(
        args, examples, vector_store,
        max_steps=max_steps, screen_size=screen_size,
    )

    # Memory record + persist AFTER verification.
    # A relevance_skip result still has an unambiguous solvable verdict
    # (False — the example didn't even run on a VM), so we don't gate it
    # on "error".
    if memory is not None:
        results_by_id = {r["id"]: r for r in results if r.get("id")}
        for ex in examples:
            r = results_by_id.get(ex["id"])
            if r is None:
                continue
            if "error" in r:
                solvable = None
            elif r.get("mode") == "relevance_skip":
                solvable = False
            else:
                solvable = r.get("score", 0) > 0
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
