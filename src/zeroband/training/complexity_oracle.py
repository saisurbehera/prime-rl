"""Complexity Oracle for code complexity evaluation."""

import ast
from typing import Dict, Any, List, Optional
import torch
import math

from .oracle import Oracle, OracleReport


class ComplexityAnalyzer(ast.NodeVisitor):
    """AST visitor to calculate cyclomatic complexity."""
    
    def __init__(self):
        self.complexity = 1  # Base complexity
        self.functions = {}  # Function-level complexities
        self.current_function = None
        
    def visit_FunctionDef(self, node):
        """Visit function definition."""
        old_function = self.current_function
        self.current_function = node.name
        self.functions[node.name] = 1  # Base complexity for function
        
        # Visit function body
        self.generic_visit(node)
        
        self.current_function = old_function
        
    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        self.visit_FunctionDef(node)
        
    def visit_If(self, node):
        """If statement increases complexity."""
        self._increment_complexity()
        self.generic_visit(node)
        
    def visit_While(self, node):
        """While loop increases complexity."""
        self._increment_complexity()
        self.generic_visit(node)
        
    def visit_For(self, node):
        """For loop increases complexity."""
        self._increment_complexity()
        self.generic_visit(node)
        
    def visit_AsyncFor(self, node):
        """Async for loop increases complexity."""
        self._increment_complexity()
        self.generic_visit(node)
        
    def visit_ExceptHandler(self, node):
        """Exception handler increases complexity."""
        self._increment_complexity()
        self.generic_visit(node)
        
    def visit_With(self, node):
        """With statement increases complexity."""
        self._increment_complexity()
        self.generic_visit(node)
        
    def visit_AsyncWith(self, node):
        """Async with statement increases complexity."""
        self._increment_complexity()
        self.generic_visit(node)
        
    def visit_Assert(self, node):
        """Assert statement increases complexity."""
        self._increment_complexity()
        self.generic_visit(node)
        
    def visit_BoolOp(self, node):
        """Boolean operators (and/or) increase complexity."""
        # Add complexity for each additional condition
        self._increment_complexity(len(node.values) - 1)
        self.generic_visit(node)
        
    def visit_Lambda(self, node):
        """Lambda expressions increase complexity."""
        self._increment_complexity()
        self.generic_visit(node)
        
    def _increment_complexity(self, amount=1):
        """Increment complexity counter."""
        self.complexity += amount
        if self.current_function:
            self.functions[self.current_function] = \
                self.functions.get(self.current_function, 1) + amount


class ComplexityOracle(Oracle):
    """Oracle that evaluates code complexity metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_complexity = config.get("target_complexity", 10)
        self.max_acceptable_complexity = config.get("max_acceptable_complexity", 20)
        self.complexity_penalty_scale = config.get("complexity_penalty_scale", 0.1)
        
    def should_run(self, prompt: str, candidate_code: str, uncertainty_score: float) -> bool:
        """Always run complexity analysis due to low computational cost."""
        return True
    
    def evaluate(self, prompt: str, candidate_code: str) -> OracleReport:
        """Calculate various complexity metrics for the code."""
        metrics = {
            "cyclomatic_complexity": 0,
            "cognitive_complexity": 0,
            "halstead_volume": 0,
            "halstead_difficulty": 0,
            "maintainability_index": 100,
            "lines_of_code": 0,
            "function_complexities": {}
        }
        
        try:
            # Parse AST
            tree = ast.parse(candidate_code)
            
            # Calculate cyclomatic complexity
            analyzer = ComplexityAnalyzer()
            analyzer.visit(tree)
            metrics["cyclomatic_complexity"] = analyzer.complexity
            metrics["function_complexities"] = analyzer.functions
            
            # Calculate other metrics
            metrics["lines_of_code"] = len(candidate_code.split('\n'))
            metrics["cognitive_complexity"] = self._calculate_cognitive_complexity(tree)
            
            halstead_metrics = self._calculate_halstead_metrics(tree)
            metrics.update(halstead_metrics)
            
            # Calculate maintainability index
            metrics["maintainability_index"] = self._calculate_maintainability_index(
                metrics["halstead_volume"],
                metrics["cyclomatic_complexity"],
                metrics["lines_of_code"]
            )
            
        except SyntaxError:
            # Code has syntax errors, assign worst scores
            metrics["cyclomatic_complexity"] = 100
            metrics["maintainability_index"] = 0
            
        # Calculate overall score (0-1, where 1 is best)
        complexity_ratio = metrics["cyclomatic_complexity"] / self.max_acceptable_complexity
        score = 1.0 / (1.0 + complexity_ratio)
        
        report = OracleReport(
            oracle_name="complexity",
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
        """Calculate differentiable loss from complexity metrics."""
        cyclomatic = report.details["cyclomatic_complexity"]
        maintainability = report.details["maintainability_index"]
        
        # Penalize deviation from target complexity
        complexity_loss = torch.relu(
            torch.tensor(
                cyclomatic - self.target_complexity, 
                dtype=torch.float32, 
                device=self.device
            )
        ) * self.complexity_penalty_scale
        
        # Penalize low maintainability (below 50 is considered poor)
        maintainability_loss = torch.relu(
            torch.tensor(
                (50 - maintainability) / 50, 
                dtype=torch.float32, 
                device=self.device
            )
        )
        
        # Combine losses
        total_loss = (complexity_loss + maintainability_loss) / 2
        
        # Ensure gradient flow
        if hidden_states.requires_grad:
            # Add small connection to hidden states
            hidden_penalty = 1e-6 * hidden_states.mean()
            total_loss = total_loss + hidden_penalty
        
        return total_loss
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity (simplified version)."""
        class CognitiveComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.nesting_level = 0
                
            def visit_If(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_While(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_For(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_FunctionDef(self, node):
                old_nesting = self.nesting_level
                self.nesting_level = 0
                self.generic_visit(node)
                self.nesting_level = old_nesting
        
        visitor = CognitiveComplexityVisitor()
        visitor.visit(tree)
        return visitor.complexity
    
    def _calculate_halstead_metrics(self, tree: ast.AST) -> Dict[str, float]:
        """Calculate Halstead complexity metrics."""
        class HalsteadVisitor(ast.NodeVisitor):
            def __init__(self):
                self.operators = set()
                self.operands = set()
                self.total_operators = 0
                self.total_operands = 0
                
            def visit_BinOp(self, node):
                self.operators.add(type(node.op).__name__)
                self.total_operators += 1
                self.generic_visit(node)
                
            def visit_UnaryOp(self, node):
                self.operators.add(type(node.op).__name__)
                self.total_operators += 1
                self.generic_visit(node)
                
            def visit_Compare(self, node):
                for op in node.ops:
                    self.operators.add(type(op).__name__)
                    self.total_operators += 1
                self.generic_visit(node)
                
            def visit_Name(self, node):
                self.operands.add(node.id)
                self.total_operands += 1
                
            def visit_Constant(self, node):
                self.operands.add(str(node.value))
                self.total_operands += 1
                
            def visit_Call(self, node):
                self.operators.add("call")
                self.total_operators += 1
                self.generic_visit(node)
        
        visitor = HalsteadVisitor()
        visitor.visit(tree)
        
        n1 = len(visitor.operators)  # Unique operators
        n2 = len(visitor.operands)   # Unique operands
        N1 = visitor.total_operators  # Total operators
        N2 = visitor.total_operands   # Total operands
        
        # Halstead metrics
        n = n1 + n2  # Program vocabulary
        N = N1 + N2  # Program length
        
        if n > 0 and N > 0:
            volume = N * math.log2(n) if n > 0 else 0
            difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
            effort = volume * difficulty
        else:
            volume = 0
            difficulty = 0
            effort = 0
        
        return {
            "halstead_volume": volume,
            "halstead_difficulty": difficulty,
            "halstead_effort": effort,
            "halstead_vocabulary": n,
            "halstead_length": N
        }
    
    def _calculate_maintainability_index(
        self, 
        halstead_volume: float, 
        cyclomatic_complexity: int, 
        lines_of_code: int
    ) -> float:
        """Calculate maintainability index."""
        if lines_of_code == 0:
            return 0
            
        # Microsoft's maintainability index formula
        # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
        mi = (
            171 - 
            5.2 * math.log(max(halstead_volume, 1)) - 
            0.23 * cyclomatic_complexity - 
            16.2 * math.log(max(lines_of_code, 1))
        )
        
        # Normalize to 0-100 range
        return max(0, min(100, mi))