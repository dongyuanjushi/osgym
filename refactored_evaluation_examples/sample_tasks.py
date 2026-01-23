#!/usr/bin/env python3
"""
Script to sample tasks from test_no_gdrive.json while maintaining
the original distribution across different domains.

Creates two JSON files:
- Training set: 128 tasks (stratified sample)
- Test set: remaining tasks

Each JSON file contains a list of full task configs.
"""

import json
import random
import sys
from pathlib import Path


def calculate_domain_counts(data):
    """Calculate the count of tasks per domain."""
    domain_counts = {}
    total_tasks = 0

    for domain, task_ids in data.items():
        count = len(task_ids)
        domain_counts[domain] = count
        total_tasks += count

    return domain_counts, total_tasks


def sample_tasks_stratified(data, target_count=128, seed=None):
    """
    Sample tasks from each domain proportionally to maintain distribution.

    Args:
        data: Dictionary mapping domain names to lists of task IDs
        target_count: Number of tasks to sample (default: 128)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (sampled_data, remaining_data) dictionaries with sampled
        and remaining tasks in the same format as input
    """
    if seed is not None:
        random.seed(seed)

    domain_counts, total_tasks = calculate_domain_counts(data)

    if target_count > total_tasks:
        print(f"Warning: Requested {target_count} tasks but only {total_tasks} available.")
        print("Returning all tasks as sampled, empty remainder.")
        return data, {domain: [] for domain in data}

    # Calculate target counts for each domain (proportional sampling)
    domain_targets = {}
    allocated = 0

    # First pass: allocate integer parts
    for domain, count in domain_counts.items():
        proportion = count / total_tasks
        target = int(proportion * target_count)
        domain_targets[domain] = target
        allocated += target

    # Second pass: allocate remaining tasks to domains with highest fractional parts
    remaining = target_count - allocated
    if remaining > 0:
        # Calculate fractional parts
        fractional_parts = []
        for domain, count in domain_counts.items():
            proportion = count / total_tasks
            fractional = (proportion * target_count) - domain_targets[domain]
            fractional_parts.append((fractional, domain))

        # Sort by fractional part (descending) and allocate remaining tasks
        fractional_parts.sort(reverse=True)
        for i in range(remaining):
            domain = fractional_parts[i][1]
            domain_targets[domain] += 1

    # Sample from each domain
    sampled_data = {}
    remaining_data = {}

    for domain, task_ids in data.items():
        target = domain_targets[domain]
        task_ids_copy = task_ids.copy()
        random.shuffle(task_ids_copy)

        if target > len(task_ids_copy):
            # If we need more than available, take all
            sampled = task_ids_copy
            leftover = []
        else:
            # Take the target number for training, rest for test
            sampled = task_ids_copy[:target]
            leftover = task_ids_copy[target:]

        sampled_data[domain] = sorted(sampled)  # Sort for consistency
        remaining_data[domain] = sorted(leftover)

    return sampled_data, remaining_data


def load_task_data(examples_dir, domain, task_id):
    """
    Load full task data from the task file.

    The refactored format uses string-based configs like:
    "_launch_setup(command=['google-chrome', '--remote-debugging-port=1337'])"

    Args:
        examples_dir: Path to the examples directory
        domain: Domain name (e.g., 'chrome', 'gimp')
        task_id: Task ID (UUID string)

    Returns:
        Dictionary containing the full task data, or None if file not found
    """
    task_file = examples_dir / domain / f"{task_id}.json"
    if not task_file.exists():
        return None

    try:
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        return task_data
    except Exception as e:
        print(f"Warning: Failed to load task {task_id} from {task_file}: {e}")
        return None


def pack_tasks_to_list(examples_dir, sampled_data):
    """
    Pack sampled tasks into a list of full task configs.

    Each task is loaded from its individual JSON file and added to a list.

    Args:
        examples_dir: Path to the examples directory containing task files
        sampled_data: Dictionary mapping domain names to lists of task IDs

    Returns:
        List of task dictionaries, each containing the full task data
    """
    dataset_tasks = []
    examples_dir = Path(examples_dir)

    for domain, task_ids in sampled_data.items():
        for task_id in task_ids:
            task_data = load_task_data(examples_dir, domain, task_id)
            if task_data is not None:
                # Ensure the task has the domain in the data
                task_data["domain"] = domain
                dataset_tasks.append(task_data)
            else:
                print(f"Warning: Skipping task {task_id} from domain {domain} (file not found)")

    return dataset_tasks


def print_distribution_comparison(original_data, train_data, test_data):
    """Print comparison of original, train, and test distributions."""
    print("\n" + "="*90)
    print("Distribution Comparison")
    print("="*90)
    print(f"{'Domain':<25} {'Original':<10} {'Train':<10} {'Test':<10} {'Orig %':<10} {'Train %':<10} {'Test %':<10}")
    print("-"*90)

    orig_counts, orig_total = calculate_domain_counts(original_data)
    train_counts, train_total = calculate_domain_counts(train_data)
    test_counts, test_total = calculate_domain_counts(test_data)

    for domain in sorted(orig_counts.keys()):
        orig_count = orig_counts[domain]
        train_count = train_counts.get(domain, 0)
        test_count = test_counts.get(domain, 0)
        orig_pct = (orig_count / orig_total) * 100 if orig_total > 0 else 0
        train_pct = (train_count / train_total) * 100 if train_total > 0 else 0
        test_pct = (test_count / test_total) * 100 if test_total > 0 else 0

        print(f"{domain:<25} {orig_count:<10} {train_count:<10} {test_count:<10} {orig_pct:>8.2f}% {train_pct:>8.2f}% {test_pct:>8.2f}%")

    print("-"*90)
    print(f"{'TOTAL':<25} {orig_total:<10} {train_total:<10} {test_total:<10} {'100.00%':<10} {'100.00%':<10} {'100.00%':<10}")
    print("="*90 + "\n")


def save_json(data, output_file):
    """
    Save data in JSON format.

    Args:
        data: Data to save (dict or list)
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    """Main function to run the sampling script."""
    # Set up paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "test_no_gdrive.json"
    train_output_file = script_dir / "train_128.json"
    test_output_file = script_dir / "test_remaining.json"
    examples_dir = script_dir / "examples"

    # Parse command line arguments
    seed = 42  # Default seed for reproducibility
    train_count = 128  # Default training set size

    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            print(f"Warning: Invalid seed value '{sys.argv[1]}', using default seed 42.")

    if len(sys.argv) > 2:
        try:
            train_count = int(sys.argv[2])
        except ValueError:
            print(f"Warning: Invalid train count '{sys.argv[2]}', using default 128.")

    # Load original data
    print(f"Loading tasks from: {input_file}")
    with open(input_file, 'r') as f:
        original_data = json.load(f)

    # Calculate and display original distribution
    orig_counts, orig_total = calculate_domain_counts(original_data)
    print(f"\nOriginal distribution:")
    print(f"  Total tasks: {orig_total}")
    for domain in sorted(orig_counts.keys()):
        count = orig_counts[domain]
        pct = (count / orig_total) * 100
        print(f"  {domain}: {count} tasks ({pct:.2f}%)")

    # Sample tasks (stratified)
    print(f"\nSampling {train_count} tasks for training set (seed={seed})...")
    train_data, test_data = sample_tasks_stratified(original_data, target_count=train_count, seed=seed)

    # Print comparison
    print_distribution_comparison(original_data, train_data, test_data)

    # Load full task data and pack into lists
    print(f"Loading full task data from: {examples_dir}")

    train_tasks = pack_tasks_to_list(examples_dir, train_data)
    test_tasks = pack_tasks_to_list(examples_dir, test_data)

    # Save as JSON files (list of task configs)
    print(f"Saving training dataset to: {train_output_file}")
    save_json(train_tasks, train_output_file)

    print(f"Saving test dataset to: {test_output_file}")
    save_json(test_tasks, test_output_file)

    # Verify total counts
    train_counts, train_total = calculate_domain_counts(train_data)
    test_counts, test_total = calculate_domain_counts(test_data)

    print(f"\n✓ Successfully created train/test split")
    print(f"✓ Training set: {train_total} tasks ({len(train_tasks)} task files loaded)")
    print(f"✓ Test set: {test_total} tasks ({len(test_tasks)} task files loaded)")
    print(f"✓ Train output: {train_output_file}")
    print(f"✓ Test output: {test_output_file}")
    print(f"\nUsage:")
    print(f"  python sample_tasks.py [seed] [train_count]")
    print(f"  Default: seed=42, train_count=128")


if __name__ == "__main__":
    main()
