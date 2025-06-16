#!/usr/bin/env python3
"""Setup SWE-bench Verified dataset and evaluation framework."""
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import requests
from dataclasses import dataclass

@dataclass
class SWEBenchInstance:
    """SWE-bench task instance."""
    instance_id: str
    repo: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    FAIL_TO_PASS: List[str]
    PASS_TO_PASS: List[str]

def download_swe_bench_verified():
    """Download SWE-bench Verified dataset from Hugging Face."""
    print("Downloading SWE-bench Verified dataset from Hugging Face...")
    
    # Create data directory
    data_dir = Path("data/swe_bench")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    verified_path = data_dir / "swe_bench_verified.json"
    
    if not verified_path.exists():
        # Import datasets library
        from datasets import load_dataset
        
        # Load from Hugging Face
        dataset = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
        
        # Convert to list of dicts and save
        data = []
        for item in dataset:
            data.append({
                'instance_id': item['instance_id'],
                'repo': item['repo'],
                'base_commit': item['base_commit'],
                'patch': item['patch'],
                'test_patch': item['test_patch'],
                'problem_statement': item['problem_statement'],
                'hints_text': item.get('hints_text', ''),
                'created_at': item['created_at'],
                'version': item['version'],
                'FAIL_TO_PASS': item['FAIL_TO_PASS'],
                'PASS_TO_PASS': item['PASS_TO_PASS']
            })
        
        with open(verified_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Downloaded {len(data)} SWE-bench Verified instances to {verified_path}")
    else:
        print(f"SWE-bench Verified already exists at {verified_path}")
    
    return verified_path

def load_swe_bench_instances(dataset_path: Path) -> List[SWEBenchInstance]:
    """Load SWE-bench instances from JSON file."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    instances = []
    for item in data:
        instance = SWEBenchInstance(
            instance_id=item['instance_id'],
            repo=item['repo'],
            base_commit=item['base_commit'],
            patch=item['patch'],
            test_patch=item['test_patch'],
            problem_statement=item['problem_statement'],
            hints_text=item.get('hints_text', ''),
            created_at=item['created_at'],
            version=item['version'],
            FAIL_TO_PASS=item['FAIL_TO_PASS'],
            PASS_TO_PASS=item['PASS_TO_PASS']
        )
        instances.append(instance)
    
    return instances

def analyze_swe_bench_stats(instances: List[SWEBenchInstance]):
    """Analyze SWE-bench dataset statistics."""
    repos = {}
    total_instances = len(instances)
    
    for instance in instances:
        repo = instance.repo
        if repo not in repos:
            repos[repo] = {
                'count': 0,
                'fail_to_pass_tests': 0,
                'pass_to_pass_tests': 0
            }
        
        repos[repo]['count'] += 1
        repos[repo]['fail_to_pass_tests'] += len(instance.FAIL_TO_PASS)
        repos[repo]['pass_to_pass_tests'] += len(instance.PASS_TO_PASS)
    
    print(f"\n=== SWE-bench Verified Statistics ===")
    print(f"Total instances: {total_instances}")
    print(f"Unique repositories: {len(repos)}")
    print(f"\nTop repositories by instance count:")
    
    sorted_repos = sorted(repos.items(), key=lambda x: x[1]['count'], reverse=True)
    for repo, stats in sorted_repos[:10]:
        print(f"  {repo}: {stats['count']} instances")
    
    # Calculate average tests per instance
    total_fail_to_pass = sum(len(inst.FAIL_TO_PASS) for inst in instances)
    total_pass_to_pass = sum(len(inst.PASS_TO_PASS) for inst in instances)
    
    print(f"\nTest statistics:")
    print(f"  Average FAIL_TO_PASS tests per instance: {total_fail_to_pass / total_instances:.1f}")
    print(f"  Average PASS_TO_PASS tests per instance: {total_pass_to_pass / total_instances:.1f}")

def create_swe_bench_prompts(instances: List[SWEBenchInstance], output_path: Path):
    """Create prompt format for SWE-bench instances."""
    prompts = []
    
    for instance in instances:
        prompt = {
            "instance_id": instance.instance_id,
            "repo": instance.repo,
            "task_id": instance.instance_id,
            "prompt": f"""Repository: {instance.repo}
Base commit: {instance.base_commit}

Problem Statement:
{instance.problem_statement}

Hints:
{instance.hints_text}

Your task is to analyze the repository and generate a patch that fixes the issue described in the problem statement. The patch should make the failing tests pass while keeping existing tests passing.

Tests that should pass after your fix:
{instance.FAIL_TO_PASS}

Tests that should continue passing:
{instance.PASS_TO_PASS}

Generate a patch in unified diff format that solves this issue.""",
            "base_commit": instance.base_commit,
            "patch": instance.patch,
            "test_patch": instance.test_patch,
            "fail_to_pass": instance.FAIL_TO_PASS,
            "pass_to_pass": instance.PASS_TO_PASS
        }
        prompts.append(prompt)
    
    # Save prompts
    with open(output_path, 'w') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"Created {len(prompts)} prompts in {output_path}")
    return prompts

def setup_swe_agent_env():
    """Setup SWE-agent environment for evaluation."""
    print("\nSetting up SWE-agent environment...")
    
    # Check if SWE-agent is available
    swe_agent_dir = Path("swe-agent")
    if not swe_agent_dir.exists():
        print("Cloning SWE-agent repository...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/princeton-nlp/SWE-agent.git",
            "swe-agent"
        ], check=True)
    
    # Install SWE-agent requirements
    requirements_path = swe_agent_dir / "requirements.txt"
    if requirements_path.exists():
        print("Installing SWE-agent requirements...")
        subprocess.run([
            "pip", "install", "-r", str(requirements_path)
        ], check=True)
    
    print("SWE-agent environment setup complete!")

def main():
    """Main setup function."""
    print("Setting up SWE-bench Verified for Tri-Oracle evaluation...")
    
    # Download dataset
    dataset_path = download_swe_bench_verified()
    
    # Load instances
    instances = load_swe_bench_instances(dataset_path)
    
    # Analyze statistics
    analyze_swe_bench_stats(instances)
    
    # Create prompts
    prompts_path = Path("data/swe_bench/swe_bench_prompts.jsonl")
    create_swe_bench_prompts(instances, prompts_path)
    
    # Setup evaluation environment
    setup_swe_agent_env()
    
    print(f"\n✅ SWE-bench setup complete!")
    print(f"   Dataset: {dataset_path}")
    print(f"   Prompts: {prompts_path}")
    print(f"   Instances: {len(instances)}")
    print(f"\nNext steps:")
    print(f"1. Run: python swe_bench_oracle_evaluation.py")
    print(f"2. Compare results with SWE-smith's 41.6% baseline")

if __name__ == "__main__":
    main()