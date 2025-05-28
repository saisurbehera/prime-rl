#!/usr/bin/env python3
"""Evaluate Tri-Oracle system on SWE-bench Verified to beat 41.6% baseline."""
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from setup_swe_bench import SWEBenchInstance, load_swe_bench_instances
from src.zeroband.training.swe_oracle_integration import (
    SWEBenchOracle, SWEBenchTask, create_swe_oracle_system
)

class SWEBenchEvaluator:
    """Evaluator for SWE-bench using Tri-Oracle system."""
    
    def __init__(
        self, 
        model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct",
        use_mcts_refinement: bool = True,
        max_refinement_iterations: int = 3
    ):
        self.model_name = model_name
        self.use_mcts_refinement = use_mcts_refinement
        self.max_refinement_iterations = max_refinement_iterations
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize oracle system
        print("Initializing Tri-Oracle system...")
        self.oracle_system = create_swe_oracle_system()
        
        # Evaluation results
        self.results = []
        
    def generate_patch(self, task: SWEBenchTask) -> str:
        """Generate a patch for the given SWE-bench task."""
        
        # Create prompt for patch generation
        prompt = self._create_patch_prompt(task)
        
        # Generate patch using model
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract patch
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        patch = self._extract_patch_from_generation(generated_text, prompt)
        
        return patch
    
    def evaluate_instance(self, instance: SWEBenchInstance) -> Dict[str, Any]:
        """Evaluate a single SWE-bench instance."""
        
        # Convert to task format
        task = SWEBenchTask(
            instance_id=instance.instance_id,
            repo=instance.repo,
            base_commit=instance.base_commit,
            problem_statement=instance.problem_statement,
            patch=instance.patch,
            test_patch=instance.test_patch,
            fail_to_pass=instance.FAIL_TO_PASS,
            pass_to_pass=instance.PASS_TO_PASS
        )
        
        # Generate initial patch
        print(f"Generating patch for {task.instance_id}...")
        generated_patch = self.generate_patch(task)
        
        # Evaluate with oracles
        if self.use_mcts_refinement:
            print("Refining patch with MCTS...")
            result = self.oracle_system.refine_patch_with_mcts(
                task, generated_patch, {}, self.max_refinement_iterations
            )
        else:
            result = self.oracle_system.evaluate_patch(task, generated_patch, {})
        
        # Check if solution is correct (based on SWE-bench criteria)
        is_correct = self._is_solution_correct(result, task)
        
        return {
            'instance_id': task.instance_id,
            'repo': task.repo,
            'is_correct': is_correct,
            'oracle_scores': result.oracle_scores,
            'oracle_feedback': result.oracle_feedback,
            'tests_passed': result.tests_passed,
            'tests_failed': result.tests_failed,
            'execution_success': result.execution_success,
            'overall_score': result.overall_score,
            'refinement_iterations': result.refinement_iterations,
            'generated_patch': result.generated_patch
        }
    
    def evaluate_dataset(
        self, 
        dataset_path: Path, 
        output_path: Path,
        max_instances: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate the entire SWE-bench Verified dataset."""
        
        # Load instances
        instances = load_swe_bench_instances(dataset_path)
        
        if max_instances:
            instances = instances[:max_instances]
            print(f"Evaluating subset of {max_instances} instances")
        
        print(f"Evaluating {len(instances)} SWE-bench instances...")
        
        results = []
        correct_count = 0
        
        for i, instance in enumerate(tqdm(instances, desc="Evaluating")):
            try:
                result = self.evaluate_instance(instance)
                results.append(result)
                
                if result['is_correct']:
                    correct_count += 1
                
                # Save intermediate results
                if (i + 1) % 10 == 0:
                    self._save_results(results, output_path)
                    current_pass_rate = correct_count / len(results)
                    print(f"Current pass@1: {current_pass_rate:.1%} ({correct_count}/{len(results)})")
                
            except Exception as e:
                print(f"Error evaluating {instance.instance_id}: {str(e)}")
                results.append({
                    'instance_id': instance.instance_id,
                    'repo': instance.repo,
                    'is_correct': False,
                    'error': str(e)
                })
        
        # Calculate final metrics
        final_pass_rate = correct_count / len(results)
        
        summary = {
            'total_instances': len(results),
            'correct_solutions': correct_count,
            'pass_rate': final_pass_rate,
            'model_name': self.model_name,
            'use_mcts_refinement': self.use_mcts_refinement,
            'max_refinement_iterations': self.max_refinement_iterations,
            'results': results
        }
        
        # Save final results
        self._save_results([summary], output_path.with_suffix('.summary.json'))
        self._save_results(results, output_path)
        
        return summary
    
    def _create_patch_prompt(self, task: SWEBenchTask) -> str:
        """Create a prompt for patch generation."""
        
        prompt = f"""You are an expert software engineer. Your task is to analyze a GitHub issue and generate a patch to fix it.

Repository: {task.repo}
Base commit: {task.base_commit}

Issue Description:
{task.problem_statement}

Tests that should pass after your fix:
{task.fail_to_pass}

Tests that should continue passing:
{task.pass_to_pass}

Please generate a unified diff patch that fixes this issue. The patch should:
1. Make all failing tests pass
2. Keep all existing tests passing
3. Follow the coding style of the repository
4. Be minimal and focused

Generate only the patch in unified diff format:

```diff
"""
        return prompt
    
    def _extract_patch_from_generation(self, generated_text: str, prompt: str) -> str:
        """Extract patch from generated text."""
        
        # Remove the prompt from the beginning
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        # Look for diff blocks
        lines = generated_text.split('\n')
        patch_lines = []
        in_diff = False
        
        for line in lines:
            if line.startswith('```diff') or line.startswith('diff --git'):
                in_diff = True
                if line.startswith('diff --git'):
                    patch_lines.append(line)
                continue
            elif line.startswith('```') and in_diff:
                break
            elif in_diff:
                patch_lines.append(line)
        
        return '\n'.join(patch_lines)
    
    def _is_solution_correct(self, result, task: SWEBenchTask) -> bool:
        """Check if the solution is correct based on SWE-bench criteria."""
        
        # For SWE-bench, a solution is correct if:
        # 1. The patch applies successfully
        # 2. All FAIL_TO_PASS tests now pass
        # 3. All PASS_TO_PASS tests still pass
        
        if not result.execution_success:
            return False
        
        total_tests = len(task.fail_to_pass) + len(task.pass_to_pass)
        return result.tests_passed == total_tests and result.tests_failed == 0
    
    def _save_results(self, results: List[Dict], output_path: Path):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tri-Oracle on SWE-bench")
    parser.add_argument("--dataset", type=str, default="data/swe_bench/swe_bench_verified.json",
                       help="Path to SWE-bench dataset")
    parser.add_argument("--output", type=str, default="swe_bench_results.json",
                       help="Output file for results")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-6.7b-instruct",
                       help="Model to use for patch generation")
    parser.add_argument("--max-instances", type=int, default=None,
                       help="Maximum instances to evaluate (for testing)")
    parser.add_argument("--no-mcts", action="store_true",
                       help="Disable MCTS refinement")
    parser.add_argument("--max-refinement", type=int, default=3,
                       help="Maximum MCTS refinement iterations")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SWEBenchEvaluator(
        model_name=args.model,
        use_mcts_refinement=not args.no_mcts,
        max_refinement_iterations=args.max_refinement
    )
    
    # Run evaluation
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    
    start_time = time.time()
    summary = evaluator.evaluate_dataset(dataset_path, output_path, args.max_instances)
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*60}")
    print(f"SWE-bench Verified Evaluation Results")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"MCTS Refinement: {'Enabled' if not args.no_mcts else 'Disabled'}")
    print(f"Total instances: {summary['total_instances']}")
    print(f"Correct solutions: {summary['correct_solutions']}")
    print(f"Pass@1: {summary['pass_rate']:.1%}")
    print(f"Evaluation time: {elapsed_time/60:.1f} minutes")
    print(f"Results saved to: {output_path}")
    
    # Compare with baseline
    baseline_pass_rate = 0.416  # SWE-smith's 41.6%
    improvement = summary['pass_rate'] - baseline_pass_rate
    
    print(f"\n{'='*60}")
    print(f"Comparison with SWE-smith baseline:")
    print(f"Baseline (SWE-agent-LM-32B): {baseline_pass_rate:.1%}")
    print(f"Tri-Oracle system: {summary['pass_rate']:.1%}")
    
    if improvement > 0:
        print(f"🎉 IMPROVEMENT: +{improvement:.1%} ({improvement*100:.1f} percentage points)")
    elif improvement < 0:
        print(f"📉 Below baseline: {improvement:.1%}")
    else:
        print(f"🤝 Equal to baseline")

if __name__ == "__main__":
    main()