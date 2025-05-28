"""Static Analysis Oracle for code quality evaluation."""

import ast
import subprocess
import tempfile
import os
import json
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F

from .oracle import Oracle, OracleReport


class StaticAnalysisOracle(Oracle):
    """Oracle that performs static analysis on code."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.linters = config.get("linters", ["pylint", "flake8"])
        self.error_weight = config.get("error_weight", 2.0)
        self.warning_weight = config.get("warning_weight", 1.0)
        self.convention_weight = config.get("convention_weight", 0.5)
        
    def should_run(self, prompt: str, candidate_code: str, uncertainty_score: float) -> bool:
        """Always run static analysis due to low computational cost."""
        return True
    
    def evaluate(self, prompt: str, candidate_code: str) -> OracleReport:
        """Run static analysis tools on the code."""
        all_issues = []
        
        # Check for basic syntax errors first
        syntax_errors = self._check_syntax(candidate_code)
        if syntax_errors:
            all_issues.extend(syntax_errors)
        
        # Run linters
        if "pylint" in self.linters:
            pylint_issues = self._run_pylint(candidate_code)
            all_issues.extend(pylint_issues)
            
        if "flake8" in self.linters:
            flake8_issues = self._run_flake8(candidate_code)
            all_issues.extend(flake8_issues)
        
        # Categorize issues
        errors = [i for i in all_issues if i["severity"] == "error"]
        warnings = [i for i in all_issues if i["severity"] == "warning"]
        conventions = [i for i in all_issues if i["severity"] == "convention"]
        
        # Calculate score (lower is better, convert to 0-1 where 1 is best)
        raw_score = (
            len(errors) * self.error_weight +
            len(warnings) * self.warning_weight +
            len(conventions) * self.convention_weight
        )
        # Normalize score
        score = 1.0 / (1.0 + raw_score)
        
        report = OracleReport(
            oracle_name="static_analysis",
            score=score,
            details={
                "num_errors": len(errors),
                "num_warnings": len(warnings),
                "num_conventions": len(conventions),
                "total_issues": len(all_issues),
                "issues": all_issues,
                "errors": errors,
                "warnings": warnings,
                "conventions": conventions
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
        """Calculate differentiable loss from static analysis results."""
        num_errors = report.details["num_errors"]
        num_warnings = report.details["num_warnings"]
        num_conventions = report.details["num_conventions"]
        
        # Normalize counts with sigmoid to get smooth gradients
        k_error = 5.0  # Scaling factor for errors
        k_warning = 10.0  # Scaling factor for warnings
        k_convention = 20.0  # Scaling factor for conventions
        
        error_loss = torch.sigmoid(
            torch.tensor(num_errors / k_error, dtype=torch.float32, device=self.device)
        )
        warning_loss = torch.sigmoid(
            torch.tensor(num_warnings / k_warning, dtype=torch.float32, device=self.device)
        )
        convention_loss = torch.sigmoid(
            torch.tensor(num_conventions / k_convention, dtype=torch.float32, device=self.device)
        )
        
        # Weighted combination
        total_loss = (
            self.error_weight * error_loss +
            self.warning_weight * warning_loss +
            self.convention_weight * convention_loss
        ) / (self.error_weight + self.warning_weight + self.convention_weight)
        
        # Ensure gradient flow
        if hidden_states.requires_grad:
            # Add small connection to hidden states to maintain gradient flow
            hidden_penalty = 1e-6 * hidden_states.mean()
            total_loss = total_loss + hidden_penalty
        
        return total_loss
    
    def _check_syntax(self, code: str) -> List[Dict[str, Any]]:
        """Check for basic syntax errors."""
        issues = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append({
                "type": "syntax_error",
                "line": e.lineno or 0,
                "message": str(e),
                "severity": "error",
                "source": "ast"
            })
        return issues
    
    def _run_pylint(self, code: str) -> List[Dict[str, Any]]:
        """Run pylint on the code."""
        issues = []
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run pylint with JSON output
            result = subprocess.run(
                ['pylint', '--output-format=json', '--errors-only', '--disable=C,R', temp_file],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                try:
                    pylint_output = json.loads(result.stdout)
                    for issue in pylint_output:
                        severity = "error" if issue.get("type") == "error" else "warning"
                        issues.append({
                            "type": issue.get("symbol", "unknown"),
                            "line": issue.get("line", 0),
                            "column": issue.get("column", 0),
                            "message": issue.get("message", ""),
                            "severity": severity,
                            "source": "pylint"
                        })
                except json.JSONDecodeError:
                    pass
                    
        except (subprocess.SubprocessError, FileNotFoundError):
            # Pylint not available, skip
            pass
        finally:
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.unlink(temp_file)
                
        return issues
    
    def _run_flake8(self, code: str) -> List[Dict[str, Any]]:
        """Run flake8 on the code."""
        issues = []
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run flake8
            result = subprocess.run(
                ['flake8', '--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s', temp_file],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    
                    # Parse flake8 output
                    parts = line.split(':', 4)
                    if len(parts) >= 4:
                        try:
                            line_num = int(parts[1])
                            col_num = int(parts[2])
                            message = parts[3].strip() if len(parts) > 3 else ""
                            
                            # Determine severity based on error code
                            code = message.split()[0] if message else ""
                            if code.startswith('E'):
                                severity = "error"
                            elif code.startswith('W'):
                                severity = "warning"
                            else:
                                severity = "convention"
                            
                            issues.append({
                                "type": code,
                                "line": line_num,
                                "column": col_num,
                                "message": message,
                                "severity": severity,
                                "source": "flake8"
                            })
                        except (ValueError, IndexError):
                            pass
                            
        except (subprocess.SubprocessError, FileNotFoundError):
            # Flake8 not available, skip
            pass
        finally:
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.unlink(temp_file)
                
        return issues