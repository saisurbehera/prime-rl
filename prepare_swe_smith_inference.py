#!/usr/bin/env python3
"""Prepare SWE-Smith data for inference."""
import json
from pathlib import Path
from datasets import Dataset

def main():
    print("Preparing SWE-Smith data for inference...")
    
    # Load SWE-Smith dataset
    ds = Dataset.from_parquet('data/swe_smith_proper/train.parquet')
    print(f"Loaded {len(ds)} SWE-Smith instances")
    
    # Create output directory
    output_dir = Path("data/swe_bench")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to inference format
    output_path = output_dir / "swe_bench_prompts.jsonl"
    
    with open(output_path, 'w') as f:
        for item in ds:
            # Parse verification info to get the ground truth
            import ast
            verification_info = ast.literal_eval(item['verification_info'])
            
            # Create inference format
            inference_item = {
                "instance_id": item['problem_id'],
                "task_id": item['problem_id'],
                "prompt": item['prompt'] if item['prompt'] else f"Problem ID: {item['problem_id']}\nTask: Generate a patch for the given repository and commit.",
                "repo": verification_info.get('repo', ''),
                "base_commit": verification_info.get('base_commit', ''),
                "ground_truth": verification_info.get('ground_truth', ''),
                "fail_to_pass": verification_info.get('fail_to_pass', []),
                "pass_to_pass": verification_info.get('pass_to_pass', [])
            }
            
            f.write(json.dumps(inference_item) + '\n')
    
    print(f"Created {len(ds)} prompts in {output_path}")
    print("\nReady for inference!")

if __name__ == "__main__":
    main()