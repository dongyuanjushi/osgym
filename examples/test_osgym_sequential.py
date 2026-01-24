#!/usr/bin/env python3
"""
Sequential version of test_osgym for comparison with concurrent version.
Runs tasks one by one and tracks detailed timing information.
"""

import requests
import time
import json
import argparse

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RED = '\033[91m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'


def load_examples(json_path, k=None):
    """
    Load examples from a JSON file containing a list of task configs.

    Args:
        json_path: Path to the JSON file (e.g., train_128.json)
        k: Number of examples to load. If None, load all.

    Returns:
        List of task config dictionaries
    """
    with open(json_path, 'r') as f:
        examples = json.load(f)

    if k is not None and k < len(examples):
        examples = examples[:k]

    return examples


def call_reset(server_url, task_config, timeout=1000):
    """Call the reset endpoint."""
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"{server_url}/reset",
        headers=headers,
        json={
            "task_config": task_config,
            "timeout": timeout
        }
    )
    return response.json()


def call_step(server_url, action, vm_id):
    """Call the step endpoint."""
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"{server_url}/step",
        headers=headers,
        json={"action": action, "vm_id": vm_id}
    )
    return response.json()


def run_single_task(task_idx, task_config, server_url, n_steps=0):
    """
    Run a single task sequentially.

    Args:
        task_idx: Index of the task
        task_config: Task configuration dictionary
        server_url: URL of the OSGym server
        n_steps: Number of intermediate steps before finish (default: 0)

    Returns:
        Dictionary containing timing and result information
    """
    task_id = task_config.get('id', 'unknown')
    domain = task_config.get('domain', 'unknown')

    result = {
        'task_idx': task_idx,
        'task_id': task_id,
        'domain': domain,
        'reset_time': 0,
        'step_times': [],
        'finish_time': 0,
        'total_time': 0,
        'success': False,
        'reward': 0,
        'error': None
    }

    task_start_time = time.time()

    try:
        print(f"{BLUE}[Task {task_idx}] Starting: {domain}/{task_id[:8]}...{RESET}")

        # Reset
        reset_start = time.time()
        reset_response = call_reset(server_url, task_config)
        reset_end = time.time()
        result['reset_time'] = reset_end - reset_start

        vm_id = reset_response["vm_id"]
        print(f"{YELLOW}  [Task {task_idx}] Reset complete (VM ID: {vm_id}) - {result['reset_time']:.2f}s{RESET}")

        # Intermediate steps (if any)
        for step_num in range(n_steps):
            step_start = time.time()
            # Simple click action as placeholder
            _ = call_step(server_url, "click(100,100)", vm_id)
            step_end = time.time()
            step_time = step_end - step_start
            result['step_times'].append(step_time)
            print(f"{YELLOW}  [Task {task_idx}] Step {step_num + 1}/{n_steps} - {step_time:.2f}s{RESET}")

        # Finish
        finish_start = time.time()
        finish_response = call_step(server_url, "finish()", vm_id)
        finish_end = time.time()
        result['finish_time'] = finish_end - finish_start

        finished = finish_response.get("is_finish", False)
        reward = finish_response.get("reward", 0)
        result['reward'] = reward
        result['success'] = finished

        print(f"{GREEN}  [Task {task_idx}] Finish complete - {result['finish_time']:.2f}s (reward: {reward}){RESET}")

    except Exception as e:
        result['error'] = str(e)
        print(f"{RED}  [Task {task_idx}] ERROR: {e}{RESET}")

    task_end_time = time.time()
    result['total_time'] = task_end_time - task_start_time

    return result


def print_timing_comparison(results, total_sequential_time):
    """Print detailed timing analysis."""
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}TIMING ANALYSIS{RESET}")
    print(f"{CYAN}{'='*70}{RESET}")

    # Calculate totals
    total_reset_time = sum(r['reset_time'] for r in results)
    total_step_time = sum(sum(r['step_times']) for r in results)
    total_finish_time = sum(r['finish_time'] for r in results)
    total_task_time = sum(r['total_time'] for r in results)

    num_tasks = len(results)
    num_steps = sum(len(r['step_times']) for r in results)

    print(f"\n{BOLD}Per-Task Breakdown:{RESET}")
    print(f"{'Task':<8} {'Domain':<15} {'Reset':<12} {'Steps':<12} {'Finish':<12} {'Total':<12} {'Status':<10}")
    print("-" * 85)

    for r in results:
        task_label = f"#{r['task_idx']}"
        domain = r['domain'][:14]
        reset_t = f"{r['reset_time']:.2f}s"
        steps_t = f"{sum(r['step_times']):.2f}s" if r['step_times'] else "N/A"
        finish_t = f"{r['finish_time']:.2f}s"
        total_t = f"{r['total_time']:.2f}s"
        status = f"{GREEN}OK{RESET}" if r['success'] else f"{RED}FAIL{RESET}"
        print(f"{task_label:<8} {domain:<15} {reset_t:<12} {steps_t:<12} {finish_t:<12} {total_t:<12} {status:<10}")

    print("-" * 85)

    # Summary statistics
    print(f"\n{BOLD}Summary Statistics:{RESET}")
    print(f"  Total tasks:           {num_tasks}")
    print(f"  Total steps:           {num_steps}")
    print(f"  Successful tasks:      {sum(1 for r in results if r['success'])}")
    print(f"  Failed tasks:          {sum(1 for r in results if not r['success'])}")

    print(f"\n{BOLD}Time Breakdown:{RESET}")
    print(f"  Total reset time:      {total_reset_time:.2f}s")
    if num_steps > 0:
        print(f"  Total step time:       {total_step_time:.2f}s")
    print(f"  Total finish time:     {total_finish_time:.2f}s")
    print(f"  Sum of task times:     {total_task_time:.2f}s")
    print(f"  {BOLD}Actual elapsed time:   {total_sequential_time:.2f}s{RESET}")

    print(f"\n{BOLD}Averages:{RESET}")
    if num_tasks > 0:
        print(f"  Avg reset time:        {total_reset_time / num_tasks:.2f}s")
        if num_steps > 0:
            print(f"  Avg step time:         {total_step_time / num_steps:.2f}s")
        print(f"  Avg finish time:       {total_finish_time / num_tasks:.2f}s")
        print(f"  Avg task time:         {total_task_time / num_tasks:.2f}s")

    # Theoretical concurrent time (assuming perfect parallelization)
    max_task_time = max(r['total_time'] for r in results) if results else 0
    print(f"\n{BOLD}Concurrency Comparison:{RESET}")
    print(f"  Sequential time:       {total_sequential_time:.2f}s")
    print(f"  Theoretical parallel:  {max_task_time:.2f}s (longest single task)")
    if max_task_time > 0:
        speedup = total_sequential_time / max_task_time
        print(f"  Potential speedup:     {speedup:.2f}x with {num_tasks} parallel workers")

    print(f"{CYAN}{'='*70}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Test OSGym sequentially with timing analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_osgym_sequential.py -k 2
  python test_osgym_sequential.py --examples ../refactored_evaluation_examples/train_128.json -k 5
  python test_osgym_sequential.py -k 3 --n-steps 2
        """
    )
    parser.add_argument('--server', type=str, default='http://127.0.0.1:20000',
                        help='OSGym server URL (default: http://127.0.0.1:20000)')
    parser.add_argument('--examples', type=str,
                        default='../refactored_evaluation_examples/no_gdrive_rl_train.json',
                        help='Path to JSON file containing task configs')
    parser.add_argument('-k', type=int, default=2,
                        help='Number of tasks to run (default: 2)')
    parser.add_argument('--n-steps', type=int, default=0,
                        help='Number of intermediate steps per task before finish (default: 0)')
    args = parser.parse_args()

    print(f"{BOLD}{MAGENTA}{'='*70}{RESET}")
    print(f"{BOLD}{MAGENTA}OSGym Sequential Test{RESET}")
    print(f"{MAGENTA}{'='*70}{RESET}")

    # Load examples
    print(f"\n{BLUE}Loading examples from: {args.examples}{RESET}")
    examples = load_examples(args.examples, k=args.k)
    num_tasks = len(examples)
    print(f"{BLUE}Loaded {num_tasks} examples{RESET}")

    # Print task summary
    print(f"\n{BLUE}Tasks to run sequentially:{RESET}")
    for i, task in enumerate(examples):
        print(f"  {i}: {task.get('domain', '?')}/{task.get('id', '?')[:8]}...")

    print(f"\n{BOLD}Starting sequential execution...{RESET}\n")

    # Run tasks sequentially
    results = []
    overall_start_time = time.time()

    for i, task_config in enumerate(examples):
        result = run_single_task(
            task_idx=i,
            task_config=task_config,
            server_url=args.server,
            n_steps=args.n_steps
        )
        results.append(result)
        print()  # Empty line between tasks

    overall_end_time = time.time()
    total_sequential_time = overall_end_time - overall_start_time

    # Print results
    print_timing_comparison(results, total_sequential_time)

    # Final summary
    success_count = sum(1 for r in results if r['success'])
    print(f"{BOLD}Final Result: {success_count}/{num_tasks} tasks completed successfully{RESET}")
    print(f"{BOLD}Total sequential time: {total_sequential_time:.2f}s{RESET}")

    return results, total_sequential_time


if __name__ == '__main__':
    main()
