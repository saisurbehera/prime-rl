"""Execution Oracle implementation for code correctness evaluation."""

import ast
import subprocess
import tempfile
import os
import traceback
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import re
import random

from .oracle import Oracle, OracleReport


@dataclass
class ExecutionResult:
    """Result from executing a test case."""
    test_name: str
    passed: bool
    error: Optional[str] = None
    output: Optional[str] = None
    expected: Optional[str] = None


class TestSynthesizer:
    """Synthesizes test cases for code evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_tests = config.get("max_tests", 5)
        
    def synthesize_tests(self, prompt: str, code: str) -> List[Dict[str, Any]]:
        """Generate test cases based on prompt and code."""
        tests = []
        
        # Extract function signature if possible
        func_name, params = self._extract_function_info(code)
        if not func_name:
            return tests
            
        # Extract examples from prompt
        examples = self._extract_examples_from_prompt(prompt)
        
        # Generate basic test cases
        if examples:
            for i, (input_val, expected) in enumerate(examples[:self.max_tests]):
                tests.append({
                    "name": f"example_{i}",
                    "input": input_val,
                    "expected": expected,
                    "function": func_name
                })
        
        # Generate edge cases
        edge_cases = self._generate_edge_cases(func_name, params, code)
        tests.extend(edge_cases[:max(0, self.max_tests - len(tests))])
        
        return tests
    
    def _extract_function_info(self, code: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """Extract function name and parameters from code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    params = [arg.arg for arg in node.args.args]
                    return node.name, params
        except:
            # Try regex fallback
            match = re.search(r'def\s+(\w+)\s*\((.*?)\)', code)
            if match:
                func_name = match.group(1)
                params_str = match.group(2)
                params = [p.strip() for p in params_str.split(',') if p.strip()]
                return func_name, params
        return None, None
    
    def _extract_examples_from_prompt(self, prompt: str) -> List[Tuple[str, str]]:
        """Extract input-output examples from prompt."""
        examples = []
        
        # Look for common patterns like "Input: ... Output: ..."
        pattern = r'(?:Input|Example):\s*(.*?)\s*(?:Output|Expected|Result):\s*(.*?)(?:\n|$)'
        matches = re.findall(pattern, prompt, re.IGNORECASE | re.DOTALL)
        
        for input_str, output_str in matches:
            examples.append((input_str.strip(), output_str.strip()))
            
        # Also look for function call examples
        func_pattern = r'(\w+)\((.*?)\)\s*(?:=>|->|returns?)\s*(.*?)(?:\n|$)'
        func_matches = re.findall(func_pattern, prompt)
        
        for _, args, result in func_matches:
            examples.append((args.strip(), result.strip()))
            
        return examples
    
    def _generate_edge_cases(self, func_name: str, params: List[str], code: str) -> List[Dict[str, Any]]:
        """Generate edge case tests based on function analysis."""
        edge_cases = []
        
        # Analyze code for type hints or operations
        code_lower = code.lower()
        
        # Number-based edge cases
        if any(term in code_lower for term in ['int', 'float', 'num', 'count', 'sum', 'add', 'subtract']):
            edge_cases.extend([
                {"name": "edge_zero", "input": "0", "function": func_name},
                {"name": "edge_negative", "input": "-1", "function": func_name},
                {"name": "edge_large", "input": "1000000", "function": func_name},
            ])
            
        # String-based edge cases
        if any(term in code_lower for term in ['str', 'string', 'text', 'char']):
            edge_cases.extend([
                {"name": "edge_empty_string", "input": '""', "function": func_name},
                {"name": "edge_single_char", "input": '"a"', "function": func_name},
                {"name": "edge_special_chars", "input": '"!@#$%"', "function": func_name},
            ])
            
        # List/array edge cases
        if any(term in code_lower for term in ['list', 'array', 'arr', '[]']):
            edge_cases.extend([
                {"name": "edge_empty_list", "input": "[]", "function": func_name},
                {"name": "edge_single_element", "input": "[1]", "function": func_name},
                {"name": "edge_duplicates", "input": "[1, 1, 1]", "function": func_name},
            ])
            
        return edge_cases


class ExecutionOracle(Oracle):
    """Oracle that evaluates code correctness through execution."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.test_synthesizer = TestSynthesizer(config)
        self.timeout = config.get("timeout", 5)
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.3)
        
        # Differentiable test result predictor
        self.test_predictor = nn.Sequential(
            nn.Linear(768, 256),  # Assuming 768-dim hidden states
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(self.device)
        
    def should_run(self, prompt: str, candidate_code: str, uncertainty_score: float) -> bool:
        """Run execution oracle when uncertainty is above threshold."""
        return uncertainty_score > self.uncertainty_threshold
    
    def evaluate(self, prompt: str, candidate_code: str) -> OracleReport:
        """Execute code against test cases and collect results."""
        # Synthesize test cases
        test_cases = self.test_synthesizer.synthesize_tests(prompt, candidate_code)
        
        # Run tests
        execution_results = []
        for test in test_cases:
            result = self._execute_test(candidate_code, test)
            execution_results.append(result)
        
        # Calculate metrics
        num_tests = len(execution_results)
        num_passed = sum(1 for r in execution_results if r.passed)
        pass_rate = num_passed / num_tests if num_tests > 0 else 0.0
        
        # Extract error types
        error_types = {}
        for result in execution_results:
            if not result.passed and result.error:
                error_type = self._classify_error(result.error)
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        report = OracleReport(
            oracle_name="execution",
            score=pass_rate,
            details={
                "num_tests_run": num_tests,
                "num_tests_passed": num_passed,
                "execution_results": execution_results,
                "error_types": error_types,
                "test_cases": test_cases
            }
        )
        
        return report
    
    def calculate_loss(
        self, 
        prompt: str, 
        candidate_code: str, 
        hidden_states: torch.Tensor,
        report: OracleReport
    ) -> torch.Tensor:
        """Calculate differentiable loss from execution results."""
        # Get execution results
        execution_results = report.details["execution_results"]
        
        if not execution_results:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Predict test pass probabilities from hidden states
        # Use mean pooling over sequence length
        if hidden_states.dim() == 3:  # [batch, seq_len, hidden_dim]
            pooled_hidden = hidden_states.mean(dim=1)
        else:
            pooled_hidden = hidden_states
        
        # Use a simple aggregate approach for now
        # In practice, you'd want more sophisticated test-specific features
        num_tests = len(execution_results)
        if num_tests > 0:
            # Get a single prediction for the overall test pass rate
            overall_pred = self.test_predictor(pooled_hidden).squeeze()
            
            # If we have multiple samples in batch, average them
            if overall_pred.dim() > 0:
                overall_pred = overall_pred.mean()
            
            # Actual pass rate
            actual_pass_rate = sum(1 for r in execution_results if r.passed) / num_tests
            actual_pass_rate = torch.tensor(actual_pass_rate, dtype=torch.float32, device=self.device)
            
            # MSE loss on pass rate
            execution_loss = F.mse_loss(overall_pred, actual_pass_rate)
        else:
            execution_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Add coverage penalty if we have coverage info
        coverage = report.details.get("coverage", 1.0)
        coverage_loss = (1.0 - coverage) ** 2
        
        total_loss = execution_loss + 0.1 * coverage_loss
        
        return total_loss
    
    def _execute_test(self, code: str, test_case: Dict[str, Any]) -> ExecutionResult:
        """Execute a single test case."""
        try:
            # Create temporary file with code and test
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Write the code
                f.write(code)
                f.write('\n\n')
                
                # Write test execution
                func_name = test_case.get("function", "solution")
                test_input = test_case.get("input", "")
                
                # Build test code
                test_code = f"""
# Test: {test_case.get('name', 'test')}
try:
    result = {func_name}({test_input})
    print("RESULT:", result)
except Exception as e:
    print("ERROR:", type(e).__name__, str(e))
    import traceback
    traceback.print_exc()
"""
                f.write(test_code)
                temp_file = f.name
            
            # Execute the test
            try:
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                output = result.stdout
                error = result.stderr
                
                # Parse output
                if "ERROR:" in output:
                    error_match = re.search(r'ERROR:\s*(\w+)\s*(.*)', output)
                    if error_match:
                        error_type = error_match.group(1)
                        error_msg = error_match.group(2)
                        return ExecutionResult(
                            test_name=test_case.get('name', 'test'),
                            passed=False,
                            error=f"{error_type}: {error_msg}",
                            output=output
                        )
                
                if "RESULT:" in output:
                    result_match = re.search(r'RESULT:\s*(.*)', output)
                    if result_match:
                        actual_output = result_match.group(1).strip()
                        
                        # Check against expected if provided
                        expected = test_case.get("expected")
                        if expected is not None:
                            passed = str(actual_output) == str(expected)
                        else:
                            # No expected value, consider it passed if no error
                            passed = True
                            
                        return ExecutionResult(
                            test_name=test_case.get('name', 'test'),
                            passed=passed,
                            output=actual_output,
                            expected=expected
                        )
                
                # Couldn't parse output properly
                return ExecutionResult(
                    test_name=test_case.get('name', 'test'),
                    passed=False,
                    error="Failed to parse output",
                    output=output + error
                )
                
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    test_name=test_case.get('name', 'test'),
                    passed=False,
                    error="Timeout"
                )
            finally:
                # Clean up
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except Exception as e:
            return ExecutionResult(
                test_name=test_case.get('name', 'test'),
                passed=False,
                error=f"Test execution failed: {str(e)}"
            )
    
    def _classify_error(self, error: str) -> str:
        """Classify error type from error message."""
        error_lower = error.lower()
        
        if "syntax" in error_lower:
            return "SyntaxError"
        elif "name" in error_lower and "not defined" in error_lower:
            return "NameError"
        elif "type" in error_lower:
            return "TypeError"
        elif "value" in error_lower:
            return "ValueError"
        elif "index" in error_lower:
            return "IndexError"
        elif "key" in error_lower:
            return "KeyError"
        elif "attribute" in error_lower:
            return "AttributeError"
        elif "zero division" in error_lower:
            return "ZeroDivisionError"
        elif "timeout" in error_lower:
            return "TimeoutError"
        else:
            return "OtherError"