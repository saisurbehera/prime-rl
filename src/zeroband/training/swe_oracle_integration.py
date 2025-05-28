"""SWE-bench specific oracle integration for repository-level code changes."""
import os
import git
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass

from .oracle_integration import OracleIntegration, create_oracle_integration
from .execution_oracle import ExecutionOracle
from .static_analysis_oracle import StaticAnalysisOracle
from .complexity_oracle import ComplexityOracle
from .documentation_oracle import DocumentationOracle
from .proof_oracle import ProofOracle
from .reflective_oracle import ReflectiveOracle
from .memory_bank import MemoryBank, MemoryEntry
from .mcts_refinement import MCTSCodeRefiner

@dataclass
class SWEBenchTask:
    """SWE-bench task representation."""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    test_patch: str
    fail_to_pass: List[str]
    pass_to_pass: List[str]

@dataclass
class SWEOracleResult:
    """Result from SWE oracle evaluation."""
    instance_id: str
    generated_patch: str
    oracle_scores: Dict[str, float]
    oracle_feedback: Dict[str, str]
    execution_success: bool
    tests_passed: int
    tests_failed: int
    overall_score: float
    refinement_iterations: int = 0

class SWEBenchOracle:
    """Oracle system specialized for SWE-bench evaluation."""
    
    def __init__(self, oracle_integration: OracleIntegration, memory_bank: Optional[MemoryBank] = None):
        self.oracle_integration = oracle_integration
        self.memory_bank = memory_bank
        self.mcts_refiner = MCTSCodeRefiner(oracle_integration.oracle_manager)
        
    def evaluate_patch(
        self, 
        task: SWEBenchTask, 
        generated_patch: str,
        repo_context: Dict[str, str]
    ) -> SWEOracleResult:
        """Evaluate a generated patch using all oracles."""
        
        # Create temporary repository for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = self._setup_test_repo(task, temp_dir)
            
            # Apply patch and run tests
            execution_result = self._test_patch_execution(
                repo_path, generated_patch, task
            )
            
            # Get oracle feedback on the patch
            oracle_feedback = self._get_oracle_feedback(
                task, generated_patch, repo_context, execution_result
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                oracle_feedback, execution_result
            )
            
            return SWEOracleResult(
                instance_id=task.instance_id,
                generated_patch=generated_patch,
                oracle_scores=oracle_feedback['scores'],
                oracle_feedback=oracle_feedback['feedback'],
                execution_success=execution_result['success'],
                tests_passed=execution_result['passed'],
                tests_failed=execution_result['failed'],
                overall_score=overall_score
            )
    
    def refine_patch_with_mcts(
        self,
        task: SWEBenchTask,
        initial_patch: str,
        repo_context: Dict[str, str],
        max_iterations: int = 5
    ) -> SWEOracleResult:
        """Use MCTS to iteratively refine a patch."""
        
        current_patch = initial_patch
        best_result = None
        best_score = 0.0
        
        for iteration in range(max_iterations):
            # Evaluate current patch
            result = self.evaluate_patch(task, current_patch, repo_context)
            
            if result.overall_score > best_score:
                best_score = result.overall_score
                best_result = result
            
            # If perfect score, no need to continue
            if result.overall_score >= 0.95:
                break
            
            # Use MCTS to generate improvement suggestions
            improvements = self.mcts_refiner.suggest_improvements(
                current_patch,
                result.oracle_feedback,
                task.problem_statement
            )
            
            if improvements:
                # Apply best improvement
                current_patch = improvements[0]['improved_code']
            else:
                break
        
        if best_result:
            best_result.refinement_iterations = iteration + 1
        
        return best_result or result
    
    def _setup_test_repo(self, task: SWEBenchTask, temp_dir: str) -> Path:
        """Setup a temporary repository for testing."""
        repo_path = Path(temp_dir) / "repo"
        
        # Clone repository
        subprocess.run([
            "git", "clone", f"https://github.com/{task.repo}.git", str(repo_path)
        ], check=True, capture_output=True)
        
        # Checkout base commit
        repo = git.Repo(repo_path)
        repo.git.checkout(task.base_commit)
        
        return repo_path
    
    def _test_patch_execution(
        self, 
        repo_path: Path, 
        patch: str, 
        task: SWEBenchTask
    ) -> Dict[str, Any]:
        """Test patch execution by applying it and running tests."""
        try:
            # Apply patch
            patch_file = repo_path / "temp_patch.diff"
            with open(patch_file, 'w') as f:
                f.write(patch)
            
            subprocess.run([
                "git", "apply", str(patch_file)
            ], cwd=repo_path, check=True, capture_output=True)
            
            # Run tests
            test_results = self._run_specific_tests(repo_path, task)
            
            return {
                'success': True,
                'passed': test_results['passed'],
                'failed': test_results['failed'],
                'output': test_results['output']
            }
            
        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'passed': 0,
                'failed': len(task.fail_to_pass) + len(task.pass_to_pass),
                'output': str(e),
                'error': e.stderr.decode() if e.stderr else ""
            }
    
    def _run_specific_tests(self, repo_path: Path, task: SWEBenchTask) -> Dict[str, Any]:
        """Run specific tests mentioned in the task."""
        passed = 0
        failed = 0
        output = []
        
        all_tests = task.fail_to_pass + task.pass_to_pass
        
        for test in all_tests:
            try:
                # Run individual test
                result = subprocess.run([
                    "python", "-m", "pytest", test, "-v"
                ], cwd=repo_path, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    passed += 1
                    output.append(f"✅ {test}: PASSED")
                else:
                    failed += 1
                    output.append(f"❌ {test}: FAILED")
                    output.append(f"   Error: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                failed += 1
                output.append(f"⏰ {test}: TIMEOUT")
            except Exception as e:
                failed += 1
                output.append(f"💥 {test}: ERROR - {str(e)}")
        
        return {
            'passed': passed,
            'failed': failed,
            'output': '\n'.join(output)
        }
    
    def _get_oracle_feedback(
        self,
        task: SWEBenchTask,
        patch: str,
        repo_context: Dict[str, str],
        execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get feedback from all oracles on the patch."""
        
        # Extract changed files from patch
        changed_files = self._extract_changed_files_from_patch(patch)
        
        # Create prompts for oracle evaluation
        prompts = [f"Repository: {task.repo}\nProblem: {task.problem_statement}"]
        codes = [patch]
        
        # Get oracle feedback
        feedback = self.oracle_integration.get_oracle_feedback_for_inference(
            prompts, codes
        )
        
        # Add SWE-specific analysis
        swe_feedback = self._analyze_swe_specific_aspects(
            task, patch, changed_files, execution_result
        )
        
        # Combine feedback
        combined_feedback = {
            'scores': feedback[0]['scores'],
            'feedback': feedback[0]['feedback']
        }
        
        # Add SWE-specific scores
        combined_feedback['scores'].update(swe_feedback['scores'])
        combined_feedback['feedback'].update(swe_feedback['feedback'])
        
        return combined_feedback
    
    def _analyze_swe_specific_aspects(
        self,
        task: SWEBenchTask,
        patch: str,
        changed_files: List[str],
        execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze SWE-bench specific aspects of the patch."""
        
        scores = {}
        feedback = {}
        
        # Test coverage score
        total_tests = len(task.fail_to_pass) + len(task.pass_to_pass)
        if total_tests > 0:
            test_pass_rate = execution_result['passed'] / total_tests
            scores['test_coverage'] = test_pass_rate
            feedback['test_coverage'] = f"Passed {execution_result['passed']}/{total_tests} tests ({test_pass_rate:.1%})"
        
        # Patch size analysis
        lines_added = len([line for line in patch.split('\n') if line.startswith('+')])
        lines_removed = len([line for line in patch.split('\n') if line.startswith('-')])
        patch_size = lines_added + lines_removed
        
        # Prefer smaller, focused patches
        if patch_size <= 10:
            scores['patch_size'] = 1.0
            feedback['patch_size'] = f"Concise patch ({patch_size} lines changed)"
        elif patch_size <= 50:
            scores['patch_size'] = 0.8
            feedback['patch_size'] = f"Moderate patch size ({patch_size} lines changed)"
        else:
            scores['patch_size'] = 0.5
            feedback['patch_size'] = f"Large patch ({patch_size} lines changed) - consider smaller changes"
        
        # File scope analysis
        if len(changed_files) == 1:
            scores['file_scope'] = 1.0
            feedback['file_scope'] = "Single file changed - good isolation"
        elif len(changed_files) <= 3:
            scores['file_scope'] = 0.8
            feedback['file_scope'] = f"{len(changed_files)} files changed - reasonable scope"
        else:
            scores['file_scope'] = 0.6
            feedback['file_scope'] = f"{len(changed_files)} files changed - broad impact"
        
        return {'scores': scores, 'feedback': feedback}
    
    def _extract_changed_files_from_patch(self, patch: str) -> List[str]:
        """Extract list of changed files from a unified diff patch."""
        files = []
        for line in patch.split('\n'):
            if line.startswith('--- a/') or line.startswith('+++ b/'):
                file_path = line.split('/', 1)[1] if '/' in line else line[4:]
                if file_path not in files and file_path != '/dev/null':
                    files.append(file_path)
        return files
    
    def _calculate_overall_score(
        self,
        oracle_feedback: Dict[str, Any],
        execution_result: Dict[str, Any]
    ) -> float:
        """Calculate overall score combining oracle feedback and execution results."""
        
        # Weight execution results heavily for SWE-bench
        execution_weight = 0.6
        oracle_weight = 0.4
        
        # Execution score
        if execution_result['success']:
            total_tests = execution_result['passed'] + execution_result['failed']
            execution_score = execution_result['passed'] / total_tests if total_tests > 0 else 0.0
        else:
            execution_score = 0.0
        
        # Oracle scores average
        oracle_scores = list(oracle_feedback['scores'].values())
        oracle_score = sum(oracle_scores) / len(oracle_scores) if oracle_scores else 0.0
        
        # Weighted combination
        overall_score = (execution_weight * execution_score) + (oracle_weight * oracle_score)
        
        return min(1.0, max(0.0, overall_score))

def create_swe_oracle_system(config_path: str = None) -> SWEBenchOracle:
    """Create a SWE-bench oracle system."""
    
    # Create oracle integration
    oracle_integration = create_oracle_integration()
    
    # Create memory bank for SWE-bench
    memory_bank = MemoryBank("swe_bench_memory.db")
    
    return SWEBenchOracle(oracle_integration, memory_bank)