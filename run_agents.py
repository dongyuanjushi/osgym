"""Multi-process agent runner for the OSGym FastAPI env service.

Loads task configs from ``refactored_evaluation_examples/examples/<domain>/*.json``,
spawns N worker processes, and drives a full multi-step agent rollout per
task against the FastAPI server defined in ``main.py``. Each worker
allocates a VM via ``/reset``, then loops ``agent.predict`` -> ``/step``
until the task terminates or ``--max-steps`` is hit, and finally calls
``/evaluate`` for the reward and ``/shutdown`` to release the VM.

This is the HTTP/multi-process equivalent of ``lib_run_single.run_single_example``.

Usage (from ``envs/osgym``)::

    python run_agents.py \\
        --server-url http://localhost:20000 \\
        --num-workers 2 \\
        --domains libreoffice_calc,libreoffice_writer \\
        --per-domain-limit 5 \\
        --output-dir ./agent_runs \\
        --model qwen3-vl --provider vllm \\
        --endpoint http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import base64
import datetime
import glob
import json
import logging
import os
import signal
import sys
import time
import traceback
from multiprocessing import Manager, Process, current_process
from typing import Any, Dict, List, Optional

import requests as http_requests

from mm_agents.qwen35_vl import Qwen35VLAgent

logger = logging.getLogger("desktopenv.run_agents")

EXAMPLES_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "refactored_evaluation_examples",
    "examples",
)

# Worker process registry — populated in run_parallel, consumed by the
# signal handler so SIGINT / SIGTERM tears the pool down cleanly.
_processes: List[Process] = []


# ═══════════════════════════════════════════════════════════════════════════
# HTTP helpers — thin wrappers around the OSGym API server (main.py)
# ═══════════════════════════════════════════════════════════════════════════


def _api_reset(server_url: str, task_config: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    resp = http_requests.post(
        f"{server_url}/reset",
        json={"task_config": task_config, "timeout": timeout},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()  # {"screenshot": b64, "problem": str, "vm_id": int}


def _api_step(server_url: str, action: str, vm_id: int, timeout: int = 300) -> Dict[str, Any]:
    resp = http_requests.post(
        f"{server_url}/step",
        json={"action": action, "vm_id": vm_id},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()  # {"screenshot": b64, "is_finish": bool, "reward": float}


def _api_evaluate(server_url: str, vm_id: int, timeout: int = 60) -> Dict[str, Any]:
    resp = http_requests.post(
        f"{server_url}/evaluate",
        json={"vm_id": vm_id},
        timeout=timeout,
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
# Example loading
# ═══════════════════════════════════════════════════════════════════════════


def _load_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Load task configs from examples_dir/<domain>/*.json.

    Honors ``--domains`` (comma list), ``--per-domain-limit`` and
    ``--max-tasks``. Each example is tagged with ``_domain`` so the worker
    can route trajectories into per-domain output subdirs.
    """
    examples_dir = args.examples_dir
    if not os.path.isdir(examples_dir):
        raise FileNotFoundError(f"examples dir not found: {examples_dir}")

    if args.domains:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    else:
        domains = sorted(
            d for d in os.listdir(examples_dir)
            if os.path.isdir(os.path.join(examples_dir, d))
        )

    examples: List[Dict[str, Any]] = []
    for domain in domains:
        domain_dir = os.path.join(examples_dir, domain)
        if not os.path.isdir(domain_dir):
            logger.warning(f"Skipping missing domain dir: {domain_dir}")
            continue
        files = sorted(glob.glob(os.path.join(domain_dir, "*.json")))
        if args.per_domain_limit and args.per_domain_limit > 0:
            files = files[: args.per_domain_limit]
        for path in files:
            try:
                with open(path) as f:
                    ex = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")
                continue
            ex.setdefault("_domain", domain)
            ex.setdefault("id", os.path.splitext(os.path.basename(path))[0])
            examples.append(ex)

    if args.max_tasks and args.max_tasks > 0:
        examples = examples[: args.max_tasks]
    logger.info(f"Loaded {len(examples)} task(s) from {examples_dir}")
    return examples


# ═══════════════════════════════════════════════════════════════════════════
# Per-task rollout (HTTP equivalent of run_single_example)
# ═══════════════════════════════════════════════════════════════════════════


def _build_agent(args: argparse.Namespace, runtime_logger: logging.Logger) -> Qwen35VLAgent:
    return Qwen35VLAgent(
        screen_size=(args.screen_width, args.screen_height),
        approach="default",
        policy_model=args.model,
        policy_model_provider=args.provider,
        policy_model_endpoint=args.endpoint,
        logger=runtime_logger,
    )


def _save_screenshot(b64_screenshot: str, path: str) -> None:
    with open(path, "wb") as fp:
        fp.write(base64.b64decode(b64_screenshot))


def run_single_rollout(
    args: argparse.Namespace,
    example: Dict[str, Any],
    result_dir: str,
    runtime_logger: logging.Logger,
) -> Dict[str, Any]:
    """Drive one full multi-step rollout against the API server."""
    proc = current_process().name
    instruction = example.get("instruction", "")
    max_steps = args.max_steps

    agent = _build_agent(args, runtime_logger)
    agent.reset(result_dir)

    reset_data = _api_reset(args.server_url, example, timeout=args.reset_timeout)
    vm_id = reset_data["vm_id"]
    runtime_logger.info(f"[{proc}] reset vm_id={vm_id} task={example.get('id')}")

    # Initial observation. The agent expects obs["screenshot"] as bytes
    # or base64; the qwen35_vl normalizer in mm_agents handles both.
    obs: Dict[str, Any] = {"screenshot": reset_data["screenshot"]}
    _save_screenshot(
        reset_data["screenshot"],
        os.path.join(result_dir, "step_0_initial.png"),
    )

    done = False
    reward = 0.0
    step_idx = 0
    trajectory: List[Dict[str, Any]] = []
    error: Optional[str] = None

    try:
        while not done and step_idx < max_steps:
            try:
                observation, thought, action_code = agent.predict(instruction, obs)
            except Exception as e:
                runtime_logger.error(f"[{proc}] predict failed at step {step_idx}: {e}")
                runtime_logger.error(traceback.format_exc())
                error = f"predict: {e}"
                break

            ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            runtime_logger.info(f"[{proc}] step={step_idx + 1} action={action_code!r}")

            # Client-side sentinels — main.py's /step doesn't recognize these.
            if action_code in ("DONE", "FAIL"):
                trajectory.append({
                    "step_num": step_idx + 1,
                    "action_timestamp": ts,
                    "thought": thought,
                    "observation": observation,
                    "action": action_code,
                    "reward": reward,
                    "done": True,
                })
                done = True
                break
            if action_code == "WAIT":
                time.sleep(args.sleep_after_execution)
                trajectory.append({
                    "step_num": step_idx + 1,
                    "action_timestamp": ts,
                    "thought": thought,
                    "observation": observation,
                    "action": "WAIT",
                    "reward": reward,
                    "done": False,
                })
                step_idx += 1
                continue

            try:
                step_data = _api_step(args.server_url, action_code, vm_id)
            except Exception as e:
                runtime_logger.error(f"[{proc}] step failed: {e}")
                error = f"step: {e}"
                break

            time.sleep(args.sleep_after_execution)

            done = bool(step_data.get("is_finish"))
            reward = float(step_data.get("reward", 0.0))
            obs = {"screenshot": step_data["screenshot"]}

            shot_path = os.path.join(result_dir, f"step_{step_idx + 1}_{ts}.png")
            _save_screenshot(step_data["screenshot"], shot_path)

            trajectory.append({
                "step_num": step_idx + 1,
                "action_timestamp": ts,
                "thought": thought,
                "observation": observation,
                "action": action_code,
                "reward": reward,
                "done": done,
                "screenshot_file": os.path.basename(shot_path),
            })
            step_idx += 1

        # Final evaluate (best-effort — server may have already evaluated on
        # the terminating step and released the VM).
        if error is None:
            time.sleep(args.sleep_after_evaluation)
            try:
                eval_data = _api_evaluate(args.server_url, vm_id)
                reward = float(eval_data.get("reward", reward))
            except Exception as e:
                runtime_logger.warning(f"[{proc}] evaluate failed: {e}")

    finally:
        _api_shutdown(args.server_url, vm_id)

    with open(os.path.join(result_dir, "trajectory.json"), "w") as fp:
        json.dump(trajectory, fp, indent=2)
    with open(os.path.join(result_dir, "result.txt"), "w") as fp:
        fp.write(f"{reward}\n")

    result: Dict[str, Any] = {
        "id": example["id"],
        "domain": example.get("_domain", "unknown"),
        "score": reward,
        "steps": step_idx,
        "done": done,
    }
    if error is not None:
        result["error"] = error
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Worker process
# ═══════════════════════════════════════════════════════════════════════════


def _setup_worker_logger(name: str) -> logging.Logger:
    """Each worker runs in its own process — give it a per-process logger."""
    lg = logging.getLogger(f"desktopenv.run_agents.{name}")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            f"[%(asctime)s %(levelname)s {name}] %(message)s"
        ))
        lg.addHandler(h)
    return lg


def worker(task_queue, args: argparse.Namespace, shared_results: list) -> None:
    """Pull tasks from the queue and run rollouts until drained."""
    proc = current_process().name
    runtime_logger = _setup_worker_logger(proc)
    runtime_logger.info(f"[{proc}] worker started")

    while True:
        try:
            example = task_queue.get(timeout=5)
        except Exception:
            break

        eid = example.get("id", "unknown")
        domain = example.get("_domain", "unknown")
        result_dir = os.path.join(args.output_dir, domain, "trajectories", eid)
        os.makedirs(result_dir, exist_ok=True)

        try:
            res = run_single_rollout(args, example, result_dir, runtime_logger)
            shared_results.append(res)
            runtime_logger.info(
                f"[{proc}] done {domain}/{eid} score={res.get('score')} "
                f"steps={res.get('steps')} err={res.get('error')}"
            )
        except KeyboardInterrupt:
            runtime_logger.warning(f"[{proc}] KeyboardInterrupt")
            break
        except Exception as e:
            runtime_logger.error(f"[{proc}] rollout failed for {eid}: {e}")
            runtime_logger.error(traceback.format_exc())
            shared_results.append({
                "id": eid,
                "domain": domain,
                "score": 0.0,
                "steps": 0,
                "done": False,
                "error": str(e),
            })

    runtime_logger.info(f"[{proc}] worker finished")


# ═══════════════════════════════════════════════════════════════════════════
# Result aggregation + parallel/sequential dispatchers
# ═══════════════════════════════════════════════════════════════════════════


def _process_results(
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Merge per-task results into ``output_dir/results.json`` by id (upsert)."""
    agg_path = os.path.join(args.output_dir, "results.json")
    prior: List[Dict[str, Any]] = []
    if os.path.isfile(agg_path):
        try:
            with open(agg_path) as f:
                prior = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {agg_path}: {e} — starting fresh")
    by_id: Dict[str, Dict[str, Any]] = {
        r.get("id"): r for r in prior if r.get("id")
    }
    for r in results:
        if r.get("id"):
            by_id[r["id"]] = r
    with open(agg_path, "w") as f:
        json.dump(list(by_id.values()), f, indent=2)

    scores = [r["score"] for r in results if "error" not in r]
    avg = sum(scores) / max(len(scores), 1)
    n_done = sum(1 for r in results if r.get("done"))
    n_err = sum(1 for r in results if "error" in r)
    logger.info(
        f"Batch: {len(results)} tasks  avg_score={avg:.3f}  "
        f"done={n_done}  errored={n_err}"
    )
    logger.info(f"Aggregate results at {agg_path}")
    return results


def run_parallel(
    args: argparse.Namespace,
    examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Multi-process rollout: workers pull from a queue."""
    global _processes

    n = max(1, args.num_workers)
    logger.info(f"Running {len(examples)} task(s) across {n} worker(s)")

    manager = Manager()
    shared_results = manager.list()
    task_queue = manager.Queue()
    for ex in examples:
        task_queue.put(ex)

    _processes = []
    for pidx in range(n):
        p = Process(
            target=worker,
            args=(task_queue, args, shared_results),
            name=f"AgentWorker-{pidx}",
            daemon=True,
        )
        p.start()
        _processes.append(p)
        logger.info(f"Started {p.name} (PID {p.pid})")

    try:
        while True:
            if task_queue.empty() and not any(p.is_alive() for p in _processes):
                break
            if not any(p.is_alive() for p in _processes):
                logger.error("All workers died")
                break
            if task_queue.empty():
                # queue drained — wait for stragglers
                for p in _processes:
                    p.join(timeout=300)
                break
            time.sleep(5)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt - terminating workers")
    finally:
        terminate_workers()
        try:
            manager.shutdown()
        except Exception:
            pass

    return list(shared_results)


def run_sequential(
    args: argparse.Namespace,
    examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sequential rollout in the main process (debugger-friendly)."""
    logger.info(f"Running {len(examples)} task(s) sequentially")
    runtime_logger = _setup_worker_logger("seq")

    results: List[Dict[str, Any]] = []
    for i, example in enumerate(examples):
        eid = example.get("id", "unknown")
        domain = example.get("_domain", "unknown")
        logger.info(f"[{i + 1}/{len(examples)}] {domain}/{eid}")
        result_dir = os.path.join(args.output_dir, domain, "trajectories", eid)
        os.makedirs(result_dir, exist_ok=True)
        try:
            res = run_single_rollout(args, example, result_dir, runtime_logger)
            results.append(res)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt — stopping rollout")
            return results
        except Exception as e:
            logger.error(f"Rollout failed for {eid}: {e}")
            logger.error(traceback.format_exc())
            results.append({
                "id": eid, "domain": domain,
                "score": 0.0, "steps": 0, "done": False, "error": str(e),
            })
    return results


def terminate_workers(timeout: int = 10) -> None:
    """Terminate any worker processes spawned by this module."""
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


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-process agent runner against the OSGym FastAPI server",
    )
    # Server
    p.add_argument("--server-url", default="http://localhost:20000",
                   help="OSGym FastAPI base URL (main.py)")
    p.add_argument("--reset-timeout", type=int, default=600,
                   help="Reset timeout in seconds (also passed to /reset)")

    # Tasks
    p.add_argument("--examples-dir", default=EXAMPLES_DIR_DEFAULT,
                   help="Root dir containing <domain>/*.json task configs")
    p.add_argument("--domains", default="",
                   help="Comma-separated domain filter (default: all subdirs)")
    p.add_argument("--per-domain-limit", type=int, default=0,
                   help="Cap examples per domain (0 = no cap)")
    p.add_argument("--max-tasks", type=int, default=0,
                   help="Global cap on total tasks (0 = no cap)")

    # Output
    p.add_argument("--output-dir", default="./agent_runs",
                   help="Directory to write per-task trajectories and results.json")

    # Workers
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--mode", choices=["run", "debug"], default="run",
                   help="run = multi-process; debug = sequential in main proc")

    # Agent / LLM
    p.add_argument("--model", required=True)
    p.add_argument("--provider", default="vllm",
                   choices=["vllm", "sglang", "bedrock"])
    p.add_argument("--endpoint", required=True,
                   help="OpenAI-compatible /v1 base URL for the policy model")
    p.add_argument("--screen-width", type=int, default=1920)
    p.add_argument("--screen-height", type=int, default=1080)

    # Step pacing
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--sleep-after-execution", type=float, default=2.0)
    p.add_argument("--sleep-after-evaluation", type=float, default=10.0)

    return p.parse_args()


def _setup_main_logger(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, f"run_agents-{ts}.log"))
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "[%(asctime)s %(levelname)s %(processName)s] %(message)s"
    )
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(sh)


def _signal_handler(signum, frame):
    logger.warning(f"Received signal {signum} — terminating workers")
    terminate_workers()
    sys.exit(130)


def main() -> None:
    args = parse_args()
    _setup_main_logger(args)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    examples = _load_examples(args)
    if not examples:
        logger.error("No examples to run — check --examples-dir / --domains")
        return

    if args.mode == "debug":
        results = run_sequential(args, examples)
    else:
        results = run_parallel(args, examples)
    _process_results(results, args)


if __name__ == "__main__":
    main()
