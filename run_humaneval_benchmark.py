#!/usr/bin/env python3
"""Run HumanEval benchmark with Tri-Oracle system."""

import json
import pandas as pd
from pathlib import Path
import subprocess
import argparse
from typing import Dict, List, Any
import torch
from datetime import datetime

from zeroband.training.oracle_integration import create_oracle_integration
from zeroband.training.memory_bank import MemoryBank, MemoryEntry
from zeroband.training.mcts_refinement import MCTSCodeRefiner


def load_humaneval_problems(path: str = "HumanEval.jsonl") -> Dict[str, Dict]:
    """Load HumanEval problems."""
    problems = {}
    with open(path, 'r') as f:
        for line in f:
            problem = json.loads(line)
            problems[problem['task_id']] = problem
    return problems


def extract_function(code: str, entry_point: str) -> str:
    """Extract the function with given entry point from code."""
    if f"def {entry_point}" not in code:
        return code
    
    # Find function start
    func_start = code.find(f"def {entry_point}")
    if func_start == -1:
        return code
    
    # Extract function
    lines = code[func_start:].split('\n')
    func_lines = [lines[0]]  # def line
    
    if len(lines) > 1:
        # Get indentation of first line after def
        base_indent = len(lines[1]) - len(lines[1].lstrip())
        
        for line in lines[1:]:
            if line.strip():  # Non-empty line
                current_indent = len(line) - len(line.lstrip())
                if current_indent >= base_indent:
                    func_lines.append(line)
                else:
                    break  # End of function
            else:
                func_lines.append(line)  # Keep empty lines
    
    return '\n'.join(func_lines)


def run_tests(solution: str, test_code: str, timeout: int = 5) -> Dict[str, Any]:
    """Run HumanEval tests on solution."""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Write solution and tests
        f.write(solution)
        f.write('\n\n')
        f.write(test_code)
        f.write('\n\n')
        f.write('# Run tests\n')
        f.write('check(candidate)\n')
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        passed = result.returncode == 0
        error = result.stderr if not passed else None
        
        return {
            'passed': passed,
            'error': error,
            'stdout': result.stdout,
            'returncode': result.returncode
        }
    
    except subprocess.TimeoutExpired:
        return {
            'passed': False,
            'error': 'Timeout exceeded',
            'stdout': '',
            'returncode': -1
        }
    
    finally:
        os.unlink(temp_file)


def evaluate_with_oracles(
    prompt: str,
    solution: str,
    oracle_integration,
    use_mcts: bool = False,
    mcts_refiner=None
) -> Dict[str, Any]:
    """Evaluate solution with all oracles and optionally refine with MCTS."""
    
    # Initial oracle evaluation
    feedback = oracle_integration.get_oracle_feedback_for_inference([prompt], [solution])
    
    if not feedback:
        return {}
    
    oracle_scores = feedback[0]["oracle_scores"]
    
    # Optional MCTS refinement
    if use_mcts and mcts_refiner and oracle_scores.get("execution", {}).get("score", 0) < 0.9:
        print("  Refining with MCTS...")
        refined_solution, refinement_info = mcts_refiner.refine_code(
            prompt=prompt,
            initial_code=solution,
            target_improvements=["execution", "complexity"]
        )
        
        # Re-evaluate refined solution
        refined_feedback = oracle_integration.get_oracle_feedback_for_inference(
            [prompt], [refined_solution]
        )
        
        if refined_feedback:
            refined_scores = refined_feedback[0]["oracle_scores"]
            
            # Use refined if better
            if refined_scores.get("execution", {}).get("score", 0) > oracle_scores.get("execution", {}).get("score", 0):
                solution = refined_solution
                oracle_scores = refined_scores
                oracle_scores["mcts_refined"] = True
                oracle_scores["mcts_improvement"] = refinement_info["improvement"]
    
    return {
        "solution": solution,
        "oracle_scores": oracle_scores
    }


def main():
    parser = argparse.ArgumentParser(description="Run HumanEval benchmark with Tri-Oracle")
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-base", help="Model to use")
    parser.add_argument("--input", default="humaneval_output/results.parquet", help="Generated solutions")
    parser.add_argument("--use-mcts", action="store_true", help="Use MCTS refinement")
    parser.add_argument("--use-memory", action="store_true", help="Store results in memory bank")
    parser.add_argument("--output", default="humaneval_evaluation.json", help="Output file")
    args = parser.parse_args()
    
    print("=== HumanEval Benchmark with Tri-Oracle ===\n")
    
    # Load problems
    print("Loading HumanEval problems...")
    problems = load_humaneval_problems()
    
    # Load generated solutions
    print(f"Loading generated solutions from {args.input}...")
    solutions_df = pd.read_parquet(args.input)
    
    # Initialize Oracle system
    print("Initializing Oracle system...")
    oracle_integration = create_oracle_integration(
        model_hidden_dim=1024,  # Adjust based on model
        use_execution_oracle=True,
        use_static_oracle=True,
        use_complexity_oracle=True,
        use_documentation_oracle=True,
        use_proof_oracle=True,
        use_reflective_oracle=True,
        use_meta_gating=False,  # Simpler for evaluation
        device="cpu"
    )
    
    # Initialize MCTS if requested
    mcts_refiner = None
    if args.use_mcts:
        print("Initializing MCTS refiner...")
        mcts_refiner = MCTSCodeRefiner(
            oracle_integration.oracle_manager,
            max_iterations=20,
            max_depth=3
        )
    
    # Initialize Memory Bank if requested
    memory_bank = None
    if args.use_memory:
        print("Initializing Memory Bank...")
        memory_bank = MemoryBank("humaneval_memory_bank")
    
    # Evaluate each solution
    results = []
    passed_count = 0
    
    print("\nEvaluating solutions...")
    for idx, row in solutions_df.iterrows():
        task_id = row.get('task_id', f'task_{idx}')
        generated_code = row.get('generated_code', row.get('completion', ''))
        
        if task_id not in problems:
            continue
        
        problem = problems[task_id]
        prompt = problem['prompt']
        entry_point = problem['entry_point']
        test_code = problem['test']
        
        print(f"\n{task_id}:", end=" ")
        
        # Extract function
        solution = extract_function(generated_code, entry_point)
        
        # Run actual tests
        test_result = run_tests(solution, test_code)
        passed = test_result['passed']
        passed_count += int(passed)
        
        print("PASS" if passed else "FAIL", end="")
        
        # Evaluate with oracles
        oracle_result = evaluate_with_oracles(
            prompt, solution, oracle_integration, 
            use_mcts=args.use_mcts, mcts_refiner=mcts_refiner
        )
        
        if oracle_result.get("oracle_scores", {}).get("mcts_refined"):
            print(" (MCTS refined)", end="")
            # Re-run tests on refined solution
            refined_test = run_tests(oracle_result["solution"], test_code)
            if refined_test['passed'] and not passed:
                print(" -> PASS", end="")
                passed_count += 1
                passed = True
        
        # Collect results
        result = {
            "task_id": task_id,
            "passed": passed,
            "error": test_result.get('error'),
            "oracle_scores": oracle_result.get("oracle_scores", {}),
            "solution": oracle_result.get("solution", solution)
        }
        results.append(result)
        
        # Store in memory bank
        if memory_bank and oracle_result.get("oracle_scores"):
            entry = MemoryEntry(
                id=f"{task_id}_{datetime.now().isoformat()}",
                timestamp=datetime.now().isoformat(),
                prompt=prompt,
                generated_code=solution,
                oracle_reports=oracle_result["oracle_scores"],
                joint_loss=1.0 - int(passed),  # Simple loss
                uncertainty_score=0.5
            )
            memory_bank.add_entry(entry)
        
        # Print oracle summary
        if oracle_result.get("oracle_scores"):
            scores = oracle_result["oracle_scores"]
            print(f"\n  Oracles: Exec={scores.get('execution', {}).get('score', 0):.2f}, "
                  f"Complex={scores.get('complexity', {}).get('score', 0):.2f}, "
                  f"Proof={scores.get('proof', {}).get('score', 0):.2f}")
    
    # Calculate metrics
    total = len(results)
    pass_rate = passed_count / total if total > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"Total problems: {total}")
    print(f"Passed: {passed_count}")
    print(f"Pass rate: {pass_rate:.2%}")
    
    # Analyze oracle correlations
    print(f"\nOracle Analysis:")
    oracle_pass_correlation = {}
    
    for oracle_name in ["execution", "static_analysis", "complexity", "documentation", "proof", "reflective"]:
        scores_passed = []
        scores_failed = []
        
        for result in results:
            score = result.get("oracle_scores", {}).get(oracle_name, {}).get("score")
            if score is not None:
                if result["passed"]:
                    scores_passed.append(score)
                else:
                    scores_failed.append(score)
        
        if scores_passed and scores_failed:
            avg_passed = sum(scores_passed) / len(scores_passed)
            avg_failed = sum(scores_failed) / len(scores_failed)
            print(f"  {oracle_name}: passed_avg={avg_passed:.3f}, failed_avg={avg_failed:.3f}, "
                  f"diff={avg_passed - avg_failed:.3f}")
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump({
            "summary": {
                "total": total,
                "passed": passed_count,
                "pass_rate": pass_rate,
                "model": args.model,
                "use_mcts": args.use_mcts
            },
            "results": results
        }, f, indent=2)
    
    if memory_bank:
        stats = memory_bank.get_statistics()
        print(f"\nMemory Bank stored {stats['total_entries']} interactions")
        memory_bank.close()
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()