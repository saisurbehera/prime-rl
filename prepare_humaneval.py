#!/usr/bin/env python3
"""Prepare HumanEval dataset for Tri-Oracle training."""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import random


def load_humaneval(path: str = "HumanEval.jsonl") -> List[Dict[str, Any]]:
    """Load HumanEval dataset."""
    problems = []
    with open(path, 'r') as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def create_training_prompt(problem: Dict[str, Any]) -> str:
    """Create a training prompt from HumanEval problem."""
    # Extract the function signature and docstring
    prompt_parts = []
    
    # Add the problem statement
    prompt_parts.append(problem['prompt'])
    
    # Add instruction
    prompt_parts.append("\n# Complete the function implementation:")
    
    return ''.join(prompt_parts)


def create_training_examples(problems: List[Dict[str, Any]], 
                           split_ratio: float = 0.9) -> tuple:
    """Create training and validation examples."""
    # Shuffle problems
    random.shuffle(problems)
    
    # Split into train/val
    split_idx = int(len(problems) * split_ratio)
    train_problems = problems[:split_idx]
    val_problems = problems[split_idx:]
    
    # Create examples
    train_examples = []
    val_examples = []
    
    for problem in train_problems:
        example = {
            'task_id': problem['task_id'],
            'prompt': create_training_prompt(problem),
            'canonical_solution': problem['canonical_solution'],
            'test': problem['test'],
            'entry_point': problem['entry_point'],
            'instruction': f"Write a Python function {problem['entry_point']} to solve this problem.",
            'task_type': 'code_generation',
            'difficulty': 'medium'  # Can be refined based on problem analysis
        }
        train_examples.append(example)
    
    for problem in val_problems:
        example = {
            'task_id': problem['task_id'],
            'prompt': create_training_prompt(problem),
            'canonical_solution': problem['canonical_solution'],
            'test': problem['test'],
            'entry_point': problem['entry_point'],
            'instruction': f"Write a Python function {problem['entry_point']} to solve this problem.",
            'task_type': 'code_generation',
            'difficulty': 'medium'
        }
        val_examples.append(example)
    
    return train_examples, val_examples


def create_parquet_files(train_examples: List[Dict], 
                        val_examples: List[Dict],
                        output_dir: str = "."):
    """Save examples as Parquet files for PrimeRL."""
    output_dir = Path(output_dir)
    
    # Convert to format expected by PrimeRL
    train_data = []
    for ex in train_examples:
        # Format for PrimeRL training
        train_data.append({
            'prompt': ex['prompt'],
            'completion': ex['canonical_solution'],
            'task_id': ex['task_id'],
            'metadata': json.dumps({
                'test': ex['test'],
                'entry_point': ex['entry_point'],
                'task_type': ex['task_type'],
                'difficulty': ex['difficulty']
            })
        })
    
    val_data = []
    for ex in val_examples:
        val_data.append({
            'prompt': ex['prompt'],
            'completion': ex['canonical_solution'],
            'task_id': ex['task_id'],
            'metadata': json.dumps({
                'test': ex['test'],
                'entry_point': ex['entry_point'],
                'task_type': ex['task_type'],
                'difficulty': ex['difficulty']
            })
        })
    
    # Save as Parquet
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    train_df.to_parquet(output_dir / "humaneval_train.parquet", index=False)
    val_df.to_parquet(output_dir / "humaneval_val.parquet", index=False)
    
    print(f"Created training set with {len(train_data)} examples")
    print(f"Created validation set with {len(val_data)} examples")
    
    # Also create a JSONL file for inference
    with open(output_dir / "humaneval_prompts.jsonl", 'w') as f:
        for ex in train_examples + val_examples:
            f.write(json.dumps({
                'prompt': ex['prompt'],
                'task_id': ex['task_id'],
                'entry_point': ex['entry_point']
            }) + '\n')


def create_evaluation_script():
    """Create a script to evaluate generated solutions."""
    script = '''#!/usr/bin/env python3
"""Evaluate generated solutions on HumanEval."""

import json
import pandas as pd
from typing import Dict, List
import subprocess
import tempfile
import os


def run_test(solution: str, test: str) -> Dict[str, Any]:
    """Run test on solution."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Write solution and test
        f.write(solution)
        f.write('\\n\\n')
        f.write(test)
        f.write('\\n\\n')
        f.write('# Run tests\\n')
        f.write('check(candidate)')
        temp_file = f.name
    
    try:
        # Run the test
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        passed = result.returncode == 0
        error = result.stderr if not passed else None
        
        return {
            'passed': passed,
            'error': error,
            'stdout': result.stdout
        }
    except subprocess.TimeoutExpired:
        return {
            'passed': False,
            'error': 'Timeout',
            'stdout': ''
        }
    finally:
        os.unlink(temp_file)


def evaluate_humaneval(results_path: str, humaneval_path: str = "HumanEval.jsonl"):
    """Evaluate generated solutions."""
    # Load HumanEval problems
    problems = {}
    with open(humaneval_path, 'r') as f:
        for line in f:
            problem = json.loads(line)
            problems[problem['task_id']] = problem
    
    # Load generated solutions
    results_df = pd.read_parquet(results_path)
    
    # Evaluate each solution
    evaluation_results = []
    
    for _, row in results_df.iterrows():
        task_id = row['task_id']
        generated_code = row['generated_code']
        
        if task_id in problems:
            problem = problems[task_id]
            
            # Extract function from generated code
            # (Simple heuristic - in practice, use AST parsing)
            if 'def ' in generated_code:
                func_start = generated_code.find('def ')
                solution = generated_code[func_start:]
            else:
                solution = generated_code
            
            # Run test
            test_result = run_test(solution, problem['test'])
            
            evaluation_results.append({
                'task_id': task_id,
                'passed': test_result['passed'],
                'error': test_result['error']
            })
            
            print(f"{task_id}: {'PASS' if test_result['passed'] else 'FAIL'}")
    
    # Calculate pass rate
    pass_rate = sum(r['passed'] for r in evaluation_results) / len(evaluation_results)
    print(f"\\nOverall pass rate: {pass_rate:.2%}")
    
    return evaluation_results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        evaluate_humaneval(sys.argv[1])
    else:
        print("Usage: evaluate_humaneval.py <results_parquet_path>")
'''
    
    with open("evaluate_humaneval.py", 'w') as f:
        f.write(script)
    
    print("Created evaluation script: evaluate_humaneval.py")


def main():
    """Main function to prepare HumanEval dataset."""
    print("Preparing HumanEval dataset for Tri-Oracle training...")
    
    # Load HumanEval
    try:
        problems = load_humaneval("HumanEval.jsonl")
        print(f"Loaded {len(problems)} problems from HumanEval")
    except FileNotFoundError:
        print("Error: HumanEval.jsonl not found!")
        print("Please download it first:")
        print("wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz")
        print("gunzip HumanEval.jsonl.gz")
        return
    
    # Create training examples
    train_examples, val_examples = create_training_examples(problems)
    
    # Save as Parquet files
    create_parquet_files(train_examples, val_examples)
    
    # Create evaluation script
    create_evaluation_script()
    
    print("\nDataset preparation complete!")
    print("\nNext steps:")
    print("1. Run inference to generate solutions:")
    print("   uv run python src/zeroband/infer.py @ configs/inference/humaneval_tri_oracle.toml")
    print("2. Train with oracles:")
    print("   uv run python src/zeroband/train_with_oracles.py @ configs/training/humaneval_tri_oracle.toml")
    print("3. Evaluate results:")
    print("   python evaluate_humaneval.py output/results.parquet")


if __name__ == "__main__":
    main()