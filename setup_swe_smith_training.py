#!/usr/bin/env python3
"""Setup SWE-smith datasets and create Tri-Oracle enhanced training pipeline."""
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from dataclasses import dataclass
import torch
from tqdm import tqdm

from src.zeroband.training.swe_oracle_integration import create_swe_oracle_system

@dataclass
class SWESmithInstance:
    """SWE-smith task instance."""
    repo: str
    instance_id: str
    problem_statement: str
    solution_patch: str
    test_files: List[str]
    base_commit: str

@dataclass
class SWESmithTrajectory:
    """SWE-smith solution trajectory."""
    instance_id: str
    trajectory: List[Dict[str, Any]]
    final_patch: str
    success: bool

class SWESmithDatasetProcessor:
    """Process SWE-smith datasets and enhance with Tri-Oracle feedback."""
    
    def __init__(self, output_dir: str = "data/swe_smith_enhanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize oracle system for enhancement
        print("Initializing Tri-Oracle system for data enhancement...")
        self.oracle_system = create_swe_oracle_system()
        
    def download_swe_smith_datasets(self):
        """Download SWE-smith datasets from HuggingFace."""
        print("Downloading SWE-smith datasets from HuggingFace...")
        
        # Download task instances (50k)
        print("Downloading SWE-smith task instances...")
        instances_dataset = load_dataset("SWE-bench/SWE-smith")
        instances_path = self.output_dir / "swe_smith_instances.parquet"
        instances_dataset['train'].to_parquet(instances_path)
        print(f"Saved {len(instances_dataset['train'])} instances to {instances_path}")
        
        # Download trajectories (5k)
        print("Downloading SWE-smith trajectories...")
        trajectories_dataset = load_dataset("SWE-bench/SWE-smith-trajectories")
        trajectories_path = self.output_dir / "swe_smith_trajectories.parquet"
        trajectories_dataset['train'].to_parquet(trajectories_path)
        print(f"Saved {len(trajectories_dataset['train'])} trajectories to {trajectories_path}")
        
        return instances_path, trajectories_path
    
    def analyze_swe_smith_data(self, instances_path: Path, trajectories_path: Path):
        """Analyze SWE-smith datasets to understand structure."""
        print("\nAnalyzing SWE-smith datasets...")
        
        # Load instances
        instances_df = pd.read_parquet(instances_path)
        trajectories_df = pd.read_parquet(trajectories_path)
        
        print(f"\n=== SWE-smith Dataset Analysis ===")
        print(f"Task instances: {len(instances_df):,}")
        print(f"Solution trajectories: {len(trajectories_df):,}")
        
        # Analyze instances
        print(f"\nInstance columns: {list(instances_df.columns)}")
        if 'repo' in instances_df.columns:
            repo_counts = instances_df['repo'].value_counts()
            print(f"Top repositories: {repo_counts.head().to_dict()}")
        
        # Analyze trajectories
        print(f"\nTrajectory columns: {list(trajectories_df.columns)}")
        if 'success' in trajectories_df.columns:
            success_rate = trajectories_df['success'].mean()
            print(f"Trajectory success rate: {success_rate:.1%}")
        
        return instances_df, trajectories_df
    
    def enhance_instances_with_oracles(
        self, 
        instances_df: pd.DataFrame, 
        max_instances: Optional[int] = None
    ) -> pd.DataFrame:
        """Enhance SWE-smith instances with Tri-Oracle feedback."""
        print("\nEnhancing instances with Tri-Oracle feedback...")
        
        if max_instances:
            instances_df = instances_df.head(max_instances)
            print(f"Processing subset of {max_instances} instances")
        
        enhanced_instances = []
        
        for idx, row in tqdm(instances_df.iterrows(), total=len(instances_df), desc="Enhancing instances"):
            try:
                # Create task from instance
                task = self._row_to_task(row)
                
                # Get oracle feedback on the reference solution
                oracle_feedback = self._get_oracle_feedback_for_instance(task)
                
                # Create enhanced instance
                enhanced_instance = {
                    'original_instance_id': row.get('instance_id', f'swe_smith_{idx}'),
                    'repo': row.get('repo', ''),
                    'problem_statement': row.get('problem_statement', ''),
                    'solution_patch': row.get('patch', ''),
                    'base_commit': row.get('base_commit', ''),
                    
                    # Add Tri-Oracle enhancements
                    'oracle_scores': oracle_feedback['scores'],
                    'oracle_feedback_text': oracle_feedback['feedback'],
                    'execution_success': oracle_feedback.get('execution_success', False),
                    'complexity_score': oracle_feedback['scores'].get('complexity', 0.0),
                    'documentation_score': oracle_feedback['scores'].get('documentation', 0.0),
                    'static_analysis_score': oracle_feedback['scores'].get('static_analysis', 0.0),
                    'proof_score': oracle_feedback['scores'].get('proof', 0.0),
                    'reflective_score': oracle_feedback['scores'].get('reflective', 0.0),
                    'overall_oracle_score': sum(oracle_feedback['scores'].values()) / len(oracle_feedback['scores']),
                    
                    # Create enhanced training prompt
                    'enhanced_prompt': self._create_enhanced_prompt(task, oracle_feedback),
                }
                
                enhanced_instances.append(enhanced_instance)
                
            except Exception as e:
                print(f"Error processing instance {idx}: {str(e)}")
                continue
        
        enhanced_df = pd.DataFrame(enhanced_instances)
        
        # Save enhanced instances
        enhanced_path = self.output_dir / "swe_smith_enhanced_instances.parquet"
        enhanced_df.to_parquet(enhanced_path)
        print(f"Saved {len(enhanced_df)} enhanced instances to {enhanced_path}")
        
        return enhanced_df
    
    def enhance_trajectories_with_oracles(
        self, 
        trajectories_df: pd.DataFrame,
        max_trajectories: Optional[int] = None
    ) -> pd.DataFrame:
        """Enhance SWE-smith trajectories with step-by-step oracle feedback."""
        print("\nEnhancing trajectories with step-by-step oracle feedback...")
        
        if max_trajectories:
            trajectories_df = trajectories_df.head(max_trajectories)
            print(f"Processing subset of {max_trajectories} trajectories")
        
        enhanced_trajectories = []
        
        for idx, row in tqdm(trajectories_df.iterrows(), total=len(trajectories_df), desc="Enhancing trajectories"):
            try:
                # Parse trajectory steps
                trajectory_steps = self._parse_trajectory(row)
                
                # Enhance each step with oracle feedback
                enhanced_steps = []
                for step in trajectory_steps:
                    if 'action' in step and step['action'] == 'edit':
                        # Get oracle feedback on this edit
                        oracle_feedback = self._get_oracle_feedback_for_step(step)
                        step['oracle_feedback'] = oracle_feedback
                    enhanced_steps.append(step)
                
                enhanced_trajectory = {
                    'original_instance_id': row.get('instance_id', f'traj_{idx}'),
                    'success': row.get('success', False),
                    'enhanced_trajectory': enhanced_steps,
                    'oracle_guided_solution': self._create_oracle_guided_solution(enhanced_steps),
                    'final_oracle_score': self._calculate_final_oracle_score(enhanced_steps),
                }
                
                enhanced_trajectories.append(enhanced_trajectory)
                
            except Exception as e:
                print(f"Error processing trajectory {idx}: {str(e)}")
                continue
        
        enhanced_df = pd.DataFrame(enhanced_trajectories)
        
        # Save enhanced trajectories
        enhanced_path = self.output_dir / "swe_smith_enhanced_trajectories.parquet"
        enhanced_df.to_parquet(enhanced_path)
        print(f"Saved {len(enhanced_df)} enhanced trajectories to {enhanced_path}")
        
        return enhanced_df
    
    def create_training_data_for_primerl(
        self, 
        enhanced_instances_df: pd.DataFrame,
        enhanced_trajectories_df: pd.DataFrame
    ):
        """Create training data in PrimeRL format with oracle feedback."""
        print("\nCreating PrimeRL training data with oracle feedback...")
        
        training_samples = []
        
        # Process enhanced instances
        for _, row in enhanced_instances_df.iterrows():
            # Create prompt-response pair with oracle feedback
            sample = {
                'prompt': row['enhanced_prompt'],
                'response': row['solution_patch'],
                'oracle_scores': row['oracle_scores'],
                'oracle_feedback': row['oracle_feedback_text'],
                'quality_score': row['overall_oracle_score'],
                'task_type': 'swe_patch_generation',
                'repo': row['repo'],
                'instance_id': row['original_instance_id']
            }
            training_samples.append(sample)
        
        # Process enhanced trajectories (step-by-step learning)
        for _, row in enhanced_trajectories_df.iterrows():
            if row['success']:  # Only use successful trajectories
                for step in row['enhanced_trajectory']:
                    if 'oracle_feedback' in step:
                        sample = {
                            'prompt': step.get('context', ''),
                            'response': step.get('action_result', ''),
                            'oracle_scores': step['oracle_feedback'].get('scores', {}),
                            'oracle_feedback': step['oracle_feedback'].get('feedback', ''),
                            'quality_score': step['oracle_feedback'].get('overall_score', 0.5),
                            'task_type': 'swe_step_execution',
                            'instance_id': row['original_instance_id'],
                            'step_type': step.get('action', 'unknown')
                        }
                        training_samples.append(sample)
        
        # Convert to parquet for PrimeRL
        training_df = pd.DataFrame(training_samples)
        training_path = self.output_dir / "tri_oracle_swe_training.parquet"
        training_df.to_parquet(training_path)
        
        print(f"Created {len(training_df):,} training samples")
        print(f"Saved training data to {training_path}")
        
        # Create statistics
        stats = {
            'total_samples': len(training_df),
            'patch_generation_samples': len(training_df[training_df['task_type'] == 'swe_patch_generation']) if len(training_df) > 0 and 'task_type' in training_df.columns else 0,
            'step_execution_samples': len(training_df[training_df['task_type'] == 'swe_step_execution']) if len(training_df) > 0 and 'task_type' in training_df.columns else 0,
            'average_quality_score': training_df['quality_score'].mean() if len(training_df) > 0 and 'quality_score' in training_df.columns else 0.0,
            'unique_repositories': training_df['repo'].nunique() if len(training_df) > 0 and 'repo' in training_df.columns else 0,
        }
        
        stats_path = self.output_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Training statistics saved to {stats_path}")
        return training_path, stats
    
    def _row_to_task(self, row) -> Dict[str, Any]:
        """Convert dataframe row to task format."""
        return {
            'instance_id': row.get('instance_id', ''),
            'repo': row.get('repo', ''),
            'problem_statement': row.get('problem_statement', ''),
            'patch': row.get('patch', ''),
            'base_commit': row.get('base_commit', ''),
        }
    
    def _get_oracle_feedback_for_instance(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Get oracle feedback for a task instance."""
        try:
            # Simplified oracle feedback (would need full repo context in practice)
            prompt = f"Repository: {task['repo']}\nProblem: {task['problem_statement']}"
            code = task['patch']
            
            feedback = self.oracle_system.oracle_integration.get_oracle_feedback_for_inference(
                [prompt], [code]
            )
            
            if feedback and len(feedback) > 0:
                fb = feedback[0]
                # Convert oracle_scores to expected format
                scores = {}
                feedback_text = {}
                for oracle_name, data in fb.get('oracle_scores', {}).items():
                    scores[oracle_name] = data.get('score', 0.0)
                    feedback_text[oracle_name] = str(data.get('details', ''))
                return {'scores': scores, 'feedback': feedback_text}
            else:
                return {'scores': {}, 'feedback': {}}
            
        except Exception as e:
            print(f"Error getting oracle feedback: {str(e)}")
            return {'scores': {}, 'feedback': {}, 'execution_success': False}
    
    def _get_oracle_feedback_for_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Get oracle feedback for a trajectory step."""
        # Simplified implementation - would analyze the specific action
        return {
            'scores': {'step_quality': 0.8},
            'feedback': {'step_quality': 'Good step execution'},
            'overall_score': 0.8
        }
    
    def _parse_trajectory(self, row) -> List[Dict[str, Any]]:
        """Parse trajectory from row data."""
        # This would parse the actual trajectory format from SWE-smith
        # For now, return a simplified structure
        return [
            {'action': 'edit', 'context': 'example context', 'action_result': 'example result'}
        ]
    
    def _create_enhanced_prompt(self, task: Dict[str, Any], oracle_feedback: Dict[str, Any]) -> str:
        """Create enhanced prompt with oracle guidance."""
        base_prompt = f"""Repository: {task['repo']}

Problem Statement:
{task['problem_statement']}

Oracle Guidance:
{oracle_feedback.get('feedback', {})}

Generate a high-quality patch that:
1. Fixes the issue described
2. Passes all tests (Execution Oracle score: {oracle_feedback.get('scores', {}).get('execution', 'N/A')})
3. Maintains code quality (Static Analysis score: {oracle_feedback.get('scores', {}).get('static_analysis', 'N/A')})
4. Follows best practices (Complexity score: {oracle_feedback.get('scores', {}).get('complexity', 'N/A')})

Patch:"""
        return base_prompt
    
    def _create_oracle_guided_solution(self, enhanced_steps: List[Dict[str, Any]]) -> str:
        """Create solution guided by oracle feedback."""
        # Combine steps with oracle guidance
        return "Oracle-guided solution based on trajectory feedback"
    
    def _calculate_final_oracle_score(self, enhanced_steps: List[Dict[str, Any]]) -> float:
        """Calculate final oracle score for trajectory."""
        scores = []
        for step in enhanced_steps:
            if 'oracle_feedback' in step:
                scores.append(step['oracle_feedback'].get('overall_score', 0.5))
        return sum(scores) / len(scores) if scores else 0.5

def main():
    """Main processing function."""
    print("Setting up SWE-smith enhanced training pipeline...")
    
    processor = SWESmithDatasetProcessor()
    
    # Download datasets
    instances_path, trajectories_path = processor.download_swe_smith_datasets()
    
    # Analyze data
    instances_df, trajectories_df = processor.analyze_swe_smith_data(instances_path, trajectories_path)
    
    # Enhance with oracles (start with small subset for testing)
    print("\n" + "="*60)
    print("ENHANCING DATA WITH TRI-ORACLE SYSTEM")
    print("="*60)
    
    enhanced_instances = processor.enhance_instances_with_oracles(instances_df, max_instances=100)
    enhanced_trajectories = processor.enhance_trajectories_with_oracles(trajectories_df, max_trajectories=50)
    
    # Create training data
    training_path, stats = processor.create_training_data_for_primerl(
        enhanced_instances, enhanced_trajectories
    )
    
    print(f"\n" + "="*60)
    print("TRI-ORACLE ENHANCED TRAINING DATA READY")
    print("="*60)
    print(f"Training file: {training_path}")
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Average quality: {stats['average_quality_score']:.3f}")
    print(f"\nNext: Train model with: python src/zeroband/train_with_oracles.py")

if __name__ == "__main__":
    main()