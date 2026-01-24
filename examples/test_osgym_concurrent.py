import requests
import multiprocessing
import time
import random
import json
import argparse
import os
from osgym import OSGymEnvWorker
from filelock import Timeout, FileLock

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RED = '\033[91m'
RESET = '\033[0m'


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


def run_env_worker(worker_id, task_config, results_dict, n_steps=10, wait_time_range=(1, 3)):
    """
    Run a single environment worker with a specific task config.

    Args:
        worker_id: Unique identifier for this worker
        port: Port number for the environment server
        task_config: Task configuration dictionary from the examples JSON
        results_dict: Shared dictionary to store results
        n_steps: Number of steps to run before finishing
        wait_time_range: Tuple of (min, max) seconds to wait between steps
    """
    task_id = task_config.get('id', 'unknown')
    domain = task_config.get('domain', 'unknown')

    # print(f"{BLUE}Worker {worker_id} starting on port {port}...{RESET}")
    print(f"{BLUE}  Task ID: {task_id}, Domain: {domain}{RESET}")
    
    port = random.choice([20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007])

    server_url = f"http://127.0.0.1:{port}"
    
    def call_reset(server_url, task_config):
        headers = {"Content-Type": "application/json"}
        reset = requests.post(
            f"{server_url}/reset",
            headers=headers,
            json={
                "task_config": task_config,
                "timeout": 1000
            }
        )
        return reset.json()
    
    reset_response = call_reset(server_url, task_config)
    # print(reset_response)
    vm_id = reset_response["vm_id"]
    
    print(f"VM ID: {vm_id}")

    def call_step(server_url, action, vm_id):
        headers = {"Content-Type": "application/json"}
        step = requests.post(f"{server_url}/step",
                             headers=headers,
                             json={"action": action, "vm_id": vm_id})
        return step.json()
    
    step_response = call_step(server_url, "finish()", vm_id)
    
    finished = step_response["is_finish"]
    reward = step_response["reward"]
    print(f"Finished: {finished}, Reward: {reward}")


def main():
    parser = argparse.ArgumentParser(
        description='Test OSGym with multiple environments (concurrent)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_osgym_concurrent.py -k 2
  python test_osgym_concurrent.py --examples ../refactored_evaluation_examples/train_128.json -k 5
        """
    )
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to config.yaml with port definitions')
    parser.add_argument('--examples', type=str,
                        default='../refactored_evaluation_examples/no_gdrive_rl_train.json',
                        help='Path to JSON file containing task configs')
    parser.add_argument('-k', type=int, default=None,
                        help='Number of examples/envs to use (default: min of available ports and examples)')
    parser.add_argument('--n-steps', type=int, default=10,
                        help='Number of steps per episode (default: 10)')
    parser.add_argument('--wait-min', type=float, default=1.0,
                        help='Minimum wait time between steps in seconds (default: 1.0)')
    parser.add_argument('--wait-max', type=float, default=3.0,
                        help='Maximum wait time between steps in seconds (default: 3.0)')
    args = parser.parse_args()

    import yaml
    from multiprocessing import Manager

    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}OSGym Concurrent Test{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    # Load config for ports
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    available_ports = config['ports']
    print(f"{BLUE}Available ports: {available_ports}{RESET}")

    # Load examples
    print(f"{BLUE}Loading examples from: {args.examples}{RESET}")
    examples = load_examples(args.examples, k=args.k)
    print(f"{BLUE}Loaded {len(examples)} examples{RESET}")

    # Determine number of workers
    if args.k is not None:
        num_workers = min(args.k, len(examples))
    else:
        num_workers = 4

    print(f"{BLUE}Starting {num_workers} environment workers...{RESET}")

    # Print task summary
    print(f"\n{BLUE}Tasks to run:{RESET}")
    for i in range(num_workers):
        task = examples[i]
        print(f"  Worker {i}: {task.get('domain', '?')}/{task.get('id', '?')[:8]}...")
    print()

    # Create a Manager to share data between processes
    manager = Manager()
    results_dict = manager.dict()

    processes = []

    # Start timing
    overall_start_time = time.time()

    # Create and start processes
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=run_env_worker,
            args=(
                i,
                examples[i],
                results_dict,
                args.n_steps,
                (args.wait_min, args.wait_max)
            )
        )
        processes.append(p)
        p.start()
        # Small delay between starting workers
        time.sleep(0.5)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # End timing
    overall_end_time = time.time()
    total_concurrent_time = overall_end_time - overall_start_time

    # Aggregate results
    reset_time_total = 0
    reset_count = 0
    step_time_total = 0
    step_count = 0
    finish_time_total = 0
    finish_count = 0
    success_count = 0
    error_count = 0

    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Results Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    for worker_id in sorted(results_dict.keys()):
        result = results_dict[worker_id]
        task_id = result.get('task_id', 'unknown')[:8]
        domain = result.get('domain', 'unknown')

        if result.get('success', False):
            success_count += 1
            reset_time_total += result['reset_time']
            reset_count += result['reset_count']
            step_time_total += result['step_time']
            step_count += result['step_count']
            finish_time_total += result['finish_time']
            finish_count += result['finish_count']
            print(f"{GREEN}  Worker {worker_id} [{domain}/{task_id}]: SUCCESS{RESET}")
        else:
            error_count += 1
            error_msg = result.get('error', 'Unknown error')
            print(f"{RED}  Worker {worker_id} [{domain}/{task_id}]: FAILED - {error_msg}{RESET}")

    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{GREEN}Successful: {success_count}/{num_workers}{RESET}")
    if error_count > 0:
        print(f"{RED}Failed: {error_count}/{num_workers}{RESET}")

    if reset_count > 0:
        print(f"{BLUE}Average reset time: {reset_time_total / reset_count:.2f}s{RESET}")
    if step_count > 0:
        print(f"{BLUE}Average step time: {step_time_total / step_count:.2f}s{RESET}")
    if finish_count > 0:
        print(f"{BLUE}Average finish time: {finish_time_total / finish_count:.2f}s{RESET}")

    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{GREEN}Total concurrent execution time: {total_concurrent_time:.2f}s{RESET}")
    print(f"{BLUE}Number of parallel workers: {num_workers}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")


if __name__ == '__main__':
    main()
