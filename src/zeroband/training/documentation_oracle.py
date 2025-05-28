"""Documentation Oracle for evaluating code documentation quality."""

import ast
import re
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F
from difflib import SequenceMatcher

from .oracle import Oracle, OracleReport


class DocstringAnalyzer(ast.NodeVisitor):
    """Analyzes docstrings in Python code."""
    
    def __init__(self):
        self.module_docstring = None
        self.functions = []
        self.classes = []
        self.methods = []
        
    def visit_Module(self, node):
        """Extract module-level docstring."""
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant)):
            self.module_docstring = node.body[0].value.value
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        """Extract function docstrings and analyze."""
        func_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "args": [arg.arg for arg in node.args.args],
            "has_return": self._has_return(node),
            "is_method": False,
            "decorators": [d.id if isinstance(d, ast.Name) else None 
                          for d in node.decorator_list]
        }
        
        # Check if it's a method (inside a class)
        for parent in ast.walk(node):
            if isinstance(parent, ast.ClassDef):
                func_info["is_method"] = True
                self.methods.append(func_info)
                break
        else:
            self.functions.append(func_info)
            
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        """Extract class docstrings."""
        class_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "methods": []
        }
        self.classes.append(class_info)
        self.generic_visit(node)
        
    def _has_return(self, node):
        """Check if function has a return statement."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                return True
        return False


class DocumentationOracle(Oracle):
    """Oracle that evaluates documentation quality."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_docstring_length = config.get("min_docstring_length", 10)
        self.require_param_docs = config.get("require_param_docs", True)
        self.require_return_docs = config.get("require_return_docs", True)
        self.docstring_style = config.get("docstring_style", "google")  # google, numpy, sphinx
        
    def should_run(self, prompt: str, candidate_code: str, uncertainty_score: float) -> bool:
        """Always run documentation analysis due to low computational cost."""
        return True
    
    def evaluate(self, prompt: str, candidate_code: str) -> OracleReport:
        """Evaluate documentation quality of the code."""
        metrics = {
            "has_module_docstring": False,
            "functions_with_docstrings": 0,
            "total_functions": 0,
            "classes_with_docstrings": 0,
            "total_classes": 0,
            "docstring_quality_scores": [],
            "missing_param_docs": [],
            "missing_return_docs": [],
            "comment_density": 0.0,
            "avg_docstring_quality": 0.0
        }
        
        try:
            # Parse AST
            tree = ast.parse(candidate_code)
            
            # Analyze docstrings
            analyzer = DocstringAnalyzer()
            analyzer.visit(tree)
            
            # Module docstring
            if analyzer.module_docstring:
                metrics["has_module_docstring"] = True
            
            # Function docstrings
            all_functions = analyzer.functions + analyzer.methods
            metrics["total_functions"] = len(all_functions)
            
            for func in all_functions:
                if func["docstring"]:
                    metrics["functions_with_docstrings"] += 1
                    
                    # Analyze docstring quality
                    quality_score = self._analyze_docstring_quality(
                        func["docstring"],
                        func["args"],
                        func["has_return"],
                        func["name"]
                    )
                    metrics["docstring_quality_scores"].append(quality_score)
                    
                    # Check for missing parameter documentation
                    missing_params = self._check_missing_param_docs(
                        func["docstring"],
                        func["args"]
                    )
                    if missing_params:
                        metrics["missing_param_docs"].append({
                            "function": func["name"],
                            "missing": missing_params
                        })
                    
                    # Check for missing return documentation
                    if func["has_return"] and not self._has_return_doc(func["docstring"]):
                        metrics["missing_return_docs"].append(func["name"])
            
            # Class docstrings
            metrics["total_classes"] = len(analyzer.classes)
            for cls in analyzer.classes:
                if cls["docstring"]:
                    metrics["classes_with_docstrings"] += 1
            
            # Calculate comment density
            metrics["comment_density"] = self._calculate_comment_density(candidate_code)
            
            # Average docstring quality
            if metrics["docstring_quality_scores"]:
                metrics["avg_docstring_quality"] = sum(metrics["docstring_quality_scores"]) / len(metrics["docstring_quality_scores"])
            
        except SyntaxError:
            # Code has syntax errors
            pass
        
        # Calculate overall score
        score = self._calculate_overall_score(metrics)
        
        report = OracleReport(
            oracle_name="documentation",
            score=score,
            details=metrics
        )
        
        return report
    
    def calculate_loss(
        self, 
        prompt: str, 
        candidate_code: str, 
        hidden_states: torch.Tensor,
        report: OracleReport
    ) -> torch.Tensor:
        """Calculate differentiable loss from documentation metrics."""
        metrics = report.details
        
        # Calculate coverage ratios
        func_coverage = 0.0
        if metrics["total_functions"] > 0:
            func_coverage = metrics["functions_with_docstrings"] / metrics["total_functions"]
        
        class_coverage = 0.0
        if metrics["total_classes"] > 0:
            class_coverage = metrics["classes_with_docstrings"] / metrics["total_classes"]
        
        # Module docstring penalty
        module_doc_loss = 0.0 if metrics["has_module_docstring"] else 0.5
        
        # Function documentation loss
        func_doc_loss = 1.0 - func_coverage
        
        # Class documentation loss
        class_doc_loss = 1.0 - class_coverage if metrics["total_classes"] > 0 else 0.0
        
        # Quality loss
        quality_loss = 1.0 - metrics["avg_docstring_quality"]
        
        # Comment density loss (penalize too low or too high)
        comment_density = metrics["comment_density"]
        if comment_density < 0.1:
            comment_loss = (0.1 - comment_density) * 5  # Penalize low comments
        elif comment_density > 0.5:
            comment_loss = (comment_density - 0.5) * 2  # Penalize excessive comments
        else:
            comment_loss = 0.0
        
        # Combine losses
        total_loss = (
            0.2 * module_doc_loss +
            0.3 * func_doc_loss +
            0.2 * class_doc_loss +
            0.2 * quality_loss +
            0.1 * comment_loss
        )
        
        # Convert to tensor
        loss_tensor = torch.tensor(total_loss, dtype=torch.float32, device=self.device)
        
        # Ensure gradient flow
        if hidden_states.requires_grad:
            # Add small connection to hidden states
            hidden_penalty = 1e-6 * hidden_states.mean()
            loss_tensor = loss_tensor + hidden_penalty
        
        return loss_tensor
    
    def _analyze_docstring_quality(
        self, 
        docstring: str, 
        args: List[str],
        has_return: bool,
        func_name: str
    ) -> float:
        """Analyze the quality of a docstring."""
        if not docstring:
            return 0.0
        
        quality_score = 0.0
        max_score = 0.0
        
        # Length check
        max_score += 1.0
        if len(docstring) >= self.min_docstring_length:
            quality_score += 1.0
        else:
            quality_score += len(docstring) / self.min_docstring_length
        
        # Has description (first line)
        max_score += 1.0
        lines = docstring.strip().split('\n')
        if lines and len(lines[0].strip()) > 5:
            quality_score += 1.0
        
        # Parameter documentation
        if args and self.require_param_docs:
            max_score += 1.0
            documented_params = self._extract_documented_params(docstring)
            if documented_params:
                coverage = len(set(args) & set(documented_params)) / len(args)
                quality_score += coverage
        
        # Return documentation
        if has_return and self.require_return_docs:
            max_score += 1.0
            if self._has_return_doc(docstring):
                quality_score += 1.0
        
        # Examples
        max_score += 0.5
        if self._has_examples(docstring):
            quality_score += 0.5
        
        # Proper formatting
        max_score += 0.5
        if self._is_well_formatted(docstring):
            quality_score += 0.5
        
        return quality_score / max_score if max_score > 0 else 0.0
    
    def _check_missing_param_docs(self, docstring: str, args: List[str]) -> List[str]:
        """Check which parameters are missing documentation."""
        if not docstring or not args:
            return args
        
        documented_params = self._extract_documented_params(docstring)
        return [arg for arg in args if arg not in documented_params and arg != "self"]
    
    def _extract_documented_params(self, docstring: str) -> List[str]:
        """Extract parameter names from docstring."""
        params = []
        
        # Google style
        google_pattern = r'Args?:\s*\n((?:\s+\w+.*\n)*)'
        google_match = re.search(google_pattern, docstring, re.MULTILINE)
        if google_match:
            param_section = google_match.group(1)
            param_pattern = r'^\s+(\w+)'
            params.extend(re.findall(param_pattern, param_section, re.MULTILINE))
        
        # NumPy style
        numpy_pattern = r'Parameters\s*\n\s*-+\s*\n((?:\s+\w+.*\n)*)'
        numpy_match = re.search(numpy_pattern, docstring, re.MULTILINE)
        if numpy_match:
            param_section = numpy_match.group(1)
            param_pattern = r'^\s+(\w+)'
            params.extend(re.findall(param_pattern, param_section, re.MULTILINE))
        
        # Sphinx style
        sphinx_pattern = r':param\s+(\w+):'
        params.extend(re.findall(sphinx_pattern, docstring))
        
        return list(set(params))
    
    def _has_return_doc(self, docstring: str) -> bool:
        """Check if docstring documents return value."""
        if not docstring:
            return False
        
        return_patterns = [
            r'Returns?:\s*\n\s+\S',  # Google style
            r'Returns\s*\n\s*-+\s*\n\s+\S',  # NumPy style
            r':returns?:',  # Sphinx style
            r':rtype:',  # Sphinx return type
        ]
        
        for pattern in return_patterns:
            if re.search(pattern, docstring, re.MULTILINE | re.IGNORECASE):
                return True
        
        return False
    
    def _has_examples(self, docstring: str) -> bool:
        """Check if docstring contains examples."""
        if not docstring:
            return False
        
        example_patterns = [
            r'Examples?:\s*\n',
            r'>>>\s+\S',  # Doctest format
            r'```python',  # Markdown code block
        ]
        
        for pattern in example_patterns:
            if re.search(pattern, docstring, re.MULTILINE | re.IGNORECASE):
                return True
        
        return False
    
    def _is_well_formatted(self, docstring: str) -> bool:
        """Check if docstring is well-formatted."""
        if not docstring:
            return False
        
        lines = docstring.strip().split('\n')
        
        # Should have summary line
        if not lines:
            return False
        
        # Summary should be concise (one line)
        if len(lines) > 1 and lines[1].strip():
            return False
        
        # Should have proper indentation
        if len(lines) > 2:
            # Check if subsequent lines are properly indented
            base_indent = len(lines[0]) - len(lines[0].lstrip())
            for line in lines[2:]:
                if line.strip() and not line.startswith(' ' * base_indent):
                    return False
        
        return True
    
    def _calculate_comment_density(self, code: str) -> float:
        """Calculate the ratio of comment lines to total lines."""
        lines = code.split('\n')
        total_lines = len(lines)
        comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                comment_lines += 1
        
        # Also count docstrings
        in_docstring = False
        docstring_delimiters = ['"""', "'''"]
        
        for line in lines:
            for delimiter in docstring_delimiters:
                if delimiter in line:
                    if in_docstring:
                        comment_lines += 1
                        in_docstring = False
                    else:
                        in_docstring = True
            if in_docstring:
                comment_lines += 1
        
        return comment_lines / total_lines if total_lines > 0 else 0.0
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall documentation score."""
        score = 0.0
        
        # Module docstring (10%)
        if metrics["has_module_docstring"]:
            score += 0.1
        
        # Function documentation coverage (40%)
        if metrics["total_functions"] > 0:
            func_coverage = metrics["functions_with_docstrings"] / metrics["total_functions"]
            score += 0.4 * func_coverage
        else:
            score += 0.4  # No functions, no penalty
        
        # Class documentation coverage (20%)
        if metrics["total_classes"] > 0:
            class_coverage = metrics["classes_with_docstrings"] / metrics["total_classes"]
            score += 0.2 * class_coverage
        else:
            score += 0.2  # No classes, no penalty
        
        # Docstring quality (20%)
        score += 0.2 * metrics["avg_docstring_quality"]
        
        # Comment density (10%)
        comment_density = metrics["comment_density"]
        if 0.1 <= comment_density <= 0.3:
            score += 0.1
        elif comment_density < 0.1:
            score += 0.1 * (comment_density / 0.1)
        else:
            score += 0.1 * max(0, 1 - (comment_density - 0.3) / 0.3)
        
        return score