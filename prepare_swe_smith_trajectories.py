#!/usr/bin/env python3
"""Download and prepare SWE-Smith trajectories dataset for training and inference."""
import json
from pathlib import Path
from datasets import load_dataset

def main():
    print("Downloading SWE-Smith trajectories dataset from HuggingFace...")
    
    # Load the dataset
    dataset = load_dataset("SWE-bench/SWE-smith-trajectories")
    
    print(f"Dataset info:")
    print(f"  Train split: {len(dataset['train'])} examples")
    print(f"  Features: {list(dataset['train'].features.keys())}")
    
    # Show first example to understand structure
    first_example = dataset['train'][0]
    print(f"\nFirst example keys: {list(first_example.keys())}")
    
    # Create output directories
    data_dir = Path("data/swe_bench")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save for inference - extract unique instances
    print("\nPreparing inference prompts...")
    instances = {}
    
    for item in dataset['train']:
        instance_id = item.get('instance_id', item.get('task_id', ''))
        if instance_id and instance_id not in instances:
            instances[instance_id] = item
    
    # Create prompts for inference
    prompts_path = data_dir / "swe_bench_prompts.jsonl"
    with open(prompts_path, 'w') as f:
        for instance_id, item in instances.items():
            prompt_item = {
                "instance_id": instance_id,
                "task_id": instance_id,
                "prompt": item.get('prompt', ''),
                "repo": item.get('repo', ''),
                "base_commit": item.get('base_commit', ''),
                "problem_statement": item.get('problem_statement', ''),
                "hints": item.get('hints', ''),
                "fail_to_pass": item.get('fail_to_pass', []),
                "pass_to_pass": item.get('pass_to_pass', [])
            }
            f.write(json.dumps(prompt_item) + '\n')
    
    print(f"Created {len(instances)} unique prompts in {prompts_path}")
    
    # Save full trajectories for training
    print("\nSaving full trajectories for training...")
    trajectories_path = data_dir / "swe_smith_trajectories.parquet"
    dataset['train'].to_parquet(trajectories_path)
    print(f"Saved {len(dataset['train'])} trajectories to {trajectories_path}")
    
    # Create a version compatible with the training config
    print("\nCreating training-compatible dataset...")
    training_data = []
    
    for item in dataset['train']:
        # Convert to training format expected by configs/training/swe_smith_tri_oracle.toml
        training_item = {
            "prompt": item.get('prompt', ''),
            "completion": item.get('completion', item.get('response', '')),
            "instance_id": item.get('instance_id', ''),
            "trajectory_id": item.get('trajectory_id', ''),
            "repo": item.get('repo', ''),
            "base_commit": item.get('base_commit', '')
        }
        training_data.append(training_item)
    
    # Save as parquet for training
    from datasets import Dataset as HFDataset
    training_dataset = HFDataset.from_list(training_data)
    training_path = Path("data/swe_smith_enhanced/swe_smith_training.parquet")
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_dataset.to_parquet(training_path)
    
    print(f"Created training dataset with {len(training_data)} examples at {training_path}")
    
    print("\n✅ Dataset preparation complete!")
    print(f"  Inference prompts: {prompts_path}")
    print(f"  Full trajectories: {trajectories_path}")
    print(f"  Training data: {training_path}")

if __name__ == "__main__":
    main()