#!/usr/bin/env python3
"""Prepare SWE-Smith data in HuggingFace dataset format for inference."""
import json
from pathlib import Path
from datasets import Dataset, DatasetDict

def main():
    print("Preparing SWE-Smith data in HuggingFace format...")
    
    # Check if we have existing SWE-Smith data
    existing_data_path = Path("data/swe_smith_proper/train.parquet")
    if existing_data_path.exists():
        print(f"Loading existing SWE-Smith data from {existing_data_path}")
        ds = Dataset.from_parquet(str(existing_data_path))
        print(f"Loaded {len(ds)} instances")
        
        # Save in a format that can be loaded by name
        output_dir = Path("data/swe_bench_dataset")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet files
        ds.to_parquet(output_dir / "train.parquet")
        
        # Create dataset_info.json
        dataset_info = {
            "dataset_name": "swe_bench_dataset",
            "dataset_size": len(ds),
            "features": {
                "problem_id": {"dtype": "string"},
                "task_type": {"dtype": "string"},
                "prompt": {"dtype": "string"},
                "verification_info": {"dtype": "string"}
            }
        }
        
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Saved dataset to {output_dir}")
        print("Dataset is ready for inference!")
        
        # Also create the JSONL file for backup
        jsonl_path = Path("data/swe_bench/swe_bench_prompts.jsonl")
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(jsonl_path, 'w') as f:
            for item in ds:
                f.write(json.dumps(item) + '\n')
        
        print(f"Also created JSONL backup at {jsonl_path}")
        
    else:
        print("Downloading SWE-Smith trajectories from HuggingFace...")
        from datasets import load_dataset
        
        # Load the trajectories dataset
        trajectories = load_dataset("SWE-bench/SWE-smith-trajectories", split="train")
        print(f"Loaded {len(trajectories)} trajectories")
        
        # Convert to the expected format
        converted_data = []
        seen_problems = set()
        
        for item in trajectories:
            problem_id = item.get('instance_id', item.get('task_id', ''))
            if problem_id and problem_id not in seen_problems:
                seen_problems.add(problem_id)
                
                # Create verification info
                verification_info = {
                    "repo": item.get('repo', ''),
                    "base_commit": item.get('base_commit', ''),
                    "fail_to_pass": item.get('fail_to_pass', []),
                    "pass_to_pass": item.get('pass_to_pass', []),
                    "patch": item.get('patch', ''),
                    "test_patch": item.get('test_patch', '')
                }
                
                converted_item = {
                    "problem_id": problem_id,
                    "task_type": "swe_patch_generation",
                    "prompt": item.get('problem_statement', item.get('prompt', '')),
                    "verification_info": json.dumps(verification_info)
                }
                
                converted_data.append(converted_item)
        
        print(f"Converted {len(converted_data)} unique problems")
        
        # Create dataset
        dataset = Dataset.from_list(converted_data)
        
        # Save in multiple formats
        output_dir = Path("data/swe_bench_dataset")
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(output_dir / "train.parquet")
        
        # Also save to existing locations
        dataset.to_parquet("data/swe_smith_proper/train.parquet")
        
        print(f"Saved dataset to {output_dir} and data/swe_smith_proper/")
        print("Dataset is ready for inference!")

if __name__ == "__main__":
    main()