"""Monte Carlo Tree Search for code refinement at inference time."""

import math
import random
import copy
import ast
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

from .oracle import OracleManager


@dataclass
class CodeEdit:
    """Represents an edit to code."""
    edit_type: str
    location: Tuple[int, int]  # (line_start, line_end)
    original: str
    replacement: str
    description: str


@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""
    code: str
    parent: Optional['MCTSNode'] = None
    children: Dict[str, 'MCTSNode'] = field(default_factory=dict)
    visits: int = 0
    total_score: float = 0.0
    oracle_scores: Optional[Dict[str, float]] = None
    edit_applied: Optional[CodeEdit] = None
    is_terminal: bool = False
    
    @property
    def average_score(self) -> float:
        """Average score across all visits."""
        return self.total_score / self.visits if self.visits > 0 else 0.0
    
    @property
    def uct_score(self, exploration_constant: float = 1.414) -> float:
        """Upper Confidence Bound score for selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.average_score
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        
        return exploitation + exploration


class CodeEditor:
    """Generates valid code edits."""
    
    def __init__(self):
        self.edit_generators = {
            "rename_variable": self._rename_variable,
            "add_validation": self._add_validation,
            "optimize_loop": self._optimize_loop,
            "add_error_handling": self._add_error_handling,
            "refactor_condition": self._refactor_condition,
            "extract_constant": self._extract_constant,
            "add_type_hints": self._add_type_hints,
            "simplify_expression": self._simplify_expression,
        }
    
    def generate_edits(self, code: str, max_edits: int = 10) -> List[CodeEdit]:
        """Generate possible edits for the code."""
        edits = []
        
        try:
            tree = ast.parse(code)
            
            for edit_type, generator in self.edit_generators.items():
                try:
                    new_edits = generator(code, tree)
                    edits.extend(new_edits[:max(1, max_edits // len(self.edit_generators))])
                except:
                    continue
            
        except SyntaxError:
            # If code has syntax errors, only try syntax fixes
            edits.extend(self._fix_syntax_errors(code))
        
        return edits[:max_edits]
    
    def apply_edit(self, code: str, edit: CodeEdit) -> str:
        """Apply an edit to code."""
        lines = code.split('\n')
        
        # Replace the specified lines
        if 0 <= edit.location[0] <= edit.location[1] < len(lines):
            # Handle single line edit
            if edit.location[0] == edit.location[1]:
                line = lines[edit.location[0]]
                lines[edit.location[0]] = line.replace(edit.original, edit.replacement, 1)
            else:
                # Multi-line edit
                new_lines = edit.replacement.split('\n')
                lines[edit.location[0]:edit.location[1]+1] = new_lines
        
        return '\n'.join(lines)
    
    def _rename_variable(self, code: str, tree: ast.AST) -> List[CodeEdit]:
        """Generate variable renaming edits."""
        edits = []
        
        # Find poorly named variables
        class VariableFinder(ast.NodeVisitor):
            def __init__(self):
                self.variables = {}
                
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    if len(node.id) <= 2:  # Short variable names
                        self.variables[node.id] = node.lineno - 1
        
        finder = VariableFinder()
        finder.visit(tree)
        
        # Suggest better names
        better_names = {
            'i': 'index', 'j': 'inner_index', 'k': 'count',
            'n': 'number', 's': 'string', 'l': 'list_items',
            'x': 'value', 'y': 'result', 'd': 'data'
        }
        
        for var, line in finder.variables.items():
            if var in better_names:
                edits.append(CodeEdit(
                    edit_type="rename_variable",
                    location=(line, line),
                    original=var,
                    replacement=better_names[var],
                    description=f"Rename '{var}' to '{better_names[var]}' for clarity"
                ))
        
        return edits
    
    def _add_validation(self, code: str, tree: ast.AST) -> List[CodeEdit]:
        """Add input validation to functions."""
        edits = []
        lines = code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has validation
                has_validation = any(
                    isinstance(child, ast.If) or isinstance(child, ast.Assert)
                    for child in node.body[:3]  # Check first few statements
                )
                
                if not has_validation and node.args.args:
                    # Add validation after function definition
                    func_line = node.lineno - 1
                    indent = self._get_indent(lines[func_line + 1]) if func_line + 1 < len(lines) else "    "
                    
                    validations = []
                    for arg in node.args.args:
                        if arg.arg != 'self':
                            validations.append(f"{indent}if {arg.arg} is None:")
                            validations.append(f"{indent}    raise ValueError('{arg.arg} cannot be None')")
                    
                    if validations:
                        edits.append(CodeEdit(
                            edit_type="add_validation",
                            location=(func_line + 1, func_line + 1),
                            original=lines[func_line + 1] if func_line + 1 < len(lines) else "",
                            replacement='\n'.join(validations) + '\n' + (lines[func_line + 1] if func_line + 1 < len(lines) else ""),
                            description=f"Add input validation to {node.name}"
                        ))
        
        return edits
    
    def _optimize_loop(self, code: str, tree: ast.AST) -> List[CodeEdit]:
        """Optimize loop patterns."""
        edits = []
        lines = code.split('\n')
        
        class LoopOptimizer(ast.NodeVisitor):
            def __init__(self):
                self.optimizations = []
                
            def visit_For(self, node):
                # Check for range(len()) pattern
                if (isinstance(node.iter, ast.Call) and
                    isinstance(node.iter.func, ast.Name) and
                    node.iter.func.id == 'range' and
                    len(node.iter.args) == 1 and
                    isinstance(node.iter.args[0], ast.Call) and
                    isinstance(node.iter.args[0].func, ast.Name) and
                    node.iter.args[0].func.id == 'len'):
                    
                    # Suggest enumerate
                    line = node.lineno - 1
                    self.optimizations.append((
                        line,
                        "range(len())",
                        "enumerate()",
                        "Use enumerate() instead of range(len())"
                    ))
                
                self.generic_visit(node)
        
        optimizer = LoopOptimizer()
        optimizer.visit(tree)
        
        for line, old, new, desc in optimizer.optimizations:
            if line < len(lines):
                edits.append(CodeEdit(
                    edit_type="optimize_loop",
                    location=(line, line),
                    original=old,
                    replacement=new,
                    description=desc
                ))
        
        return edits
    
    def _add_error_handling(self, code: str, tree: ast.AST) -> List[CodeEdit]:
        """Add try-except blocks where appropriate."""
        edits = []
        lines = code.split('\n')
        
        # Find risky operations without error handling
        risky_patterns = [
            (r'int\(', 'ValueError', 'Type conversion'),
            (r'float\(', 'ValueError', 'Type conversion'),
            (r'\[.*\]', 'IndexError', 'List indexing'),
            (r'\.split\(', 'AttributeError', 'String operation'),
            (r'open\(', 'IOError', 'File operation'),
        ]
        
        import re
        for i, line in enumerate(lines):
            for pattern, exception, operation in risky_patterns:
                if re.search(pattern, line) and 'try:' not in lines[max(0, i-2):i]:
                    indent = self._get_indent(line)
                    
                    wrapped = [
                        f"{indent}try:",
                        f"{indent}    {line.strip()}",
                        f"{indent}except {exception}:",
                        f"{indent}    # Handle {operation} error",
                        f"{indent}    pass  # TODO: Proper error handling"
                    ]
                    
                    edits.append(CodeEdit(
                        edit_type="add_error_handling",
                        location=(i, i),
                        original=line,
                        replacement='\n'.join(wrapped),
                        description=f"Add error handling for {operation}"
                    ))
                    break
        
        return edits[:3]  # Limit to avoid too many edits
    
    def _refactor_condition(self, code: str, tree: ast.AST) -> List[CodeEdit]:
        """Refactor complex conditions."""
        edits = []
        
        class ConditionRefactorer(ast.NodeVisitor):
            def __init__(self):
                self.complex_conditions = []
                
            def visit_If(self, node):
                # Check for complex conditions
                if isinstance(node.test, ast.BoolOp) and len(node.test.values) > 3:
                    self.complex_conditions.append((node.lineno - 1, node))
                    
                # Check for negated conditions that could be simplified
                elif isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not):
                    self.complex_conditions.append((node.lineno - 1, node))
                    
                self.generic_visit(node)
        
        refactorer = ConditionRefactorer()
        refactorer.visit(tree)
        
        for line, node in refactorer.complex_conditions[:2]:
            edits.append(CodeEdit(
                edit_type="refactor_condition",
                location=(line, line),
                original="complex condition",
                replacement="simplified condition",
                description="Simplify complex conditional expression"
            ))
        
        return edits
    
    def _extract_constant(self, code: str, tree: ast.AST) -> List[CodeEdit]:
        """Extract magic numbers to constants."""
        edits = []
        lines = code.split('\n')
        
        # Find magic numbers
        magic_numbers = {}
        for i, line in enumerate(lines):
            import re
            numbers = re.findall(r'\b\d{2,}\b', line)
            for num in numbers:
                if num not in ['10', '100', '1000']:  # Common bases
                    if num not in magic_numbers:
                        magic_numbers[num] = []
                    magic_numbers[num].append(i)
        
        # Suggest constants
        for num, occurrences in magic_numbers.items():
            if len(occurrences) >= 2:  # Used multiple times
                const_name = f"MAX_VALUE_{num}" if int(num) > 100 else f"CONSTANT_{num}"
                
                # Add constant definition at top
                edits.append(CodeEdit(
                    edit_type="extract_constant",
                    location=(0, 0),
                    original=lines[0] if lines else "",
                    replacement=f"{const_name} = {num}\n" + (lines[0] if lines else ""),
                    description=f"Extract magic number {num} to constant"
                ))
                break
        
        return edits
    
    def _add_type_hints(self, code: str, tree: ast.AST) -> List[CodeEdit]:
        """Add type hints to functions."""
        edits = []
        lines = code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has type hints
                has_hints = any(arg.annotation for arg in node.args.args) or node.returns
                
                if not has_hints:
                    line = node.lineno - 1
                    if line < len(lines):
                        func_line = lines[line]
                        
                        # Simple heuristic for common patterns
                        if 'list' in node.name.lower() or 'array' in node.name.lower():
                            return_hint = " -> List[Any]"
                        elif 'count' in node.name.lower() or 'num' in node.name.lower():
                            return_hint = " -> int"
                        elif 'is_' in node.name or 'has_' in node.name:
                            return_hint = " -> bool"
                        else:
                            return_hint = " -> Any"
                        
                        # Add return type hint
                        if ')' in func_line and ':' in func_line:
                            new_line = func_line.replace('):', f'){return_hint}:')
                            
                            edits.append(CodeEdit(
                                edit_type="add_type_hints",
                                location=(line, line),
                                original=func_line,
                                replacement=new_line,
                                description=f"Add type hints to {node.name}"
                            ))
        
        return edits
    
    def _simplify_expression(self, code: str, tree: ast.AST) -> List[CodeEdit]:
        """Simplify complex expressions."""
        edits = []
        lines = code.split('\n')
        
        # Simple pattern-based simplifications
        simplifications = [
            (r'if\s+\w+\s*==\s*True:', 'if {var}:', 'Remove redundant True comparison'),
            (r'if\s+\w+\s*==\s*False:', 'if not {var}:', 'Remove redundant False comparison'),
            (r'len\(\w+\)\s*==\s*0', 'not {var}', 'Simplify empty check'),
            (r'len\(\w+\)\s*>\s*0', '{var}', 'Simplify non-empty check'),
        ]
        
        import re
        for i, line in enumerate(lines):
            for pattern, replacement, desc in simplifications:
                if re.search(pattern, line):
                    edits.append(CodeEdit(
                        edit_type="simplify_expression",
                        location=(i, i),
                        original=line,
                        replacement=line,  # Simplified in description
                        description=desc
                    ))
                    break
        
        return edits
    
    def _fix_syntax_errors(self, code: str) -> List[CodeEdit]:
        """Attempt to fix common syntax errors."""
        edits = []
        lines = code.split('\n')
        
        # Common syntax fixes
        for i, line in enumerate(lines):
            # Missing colons
            if re.match(r'^(def|if|elif|else|for|while|try|except|class)\s+.*[^:]$', line.strip()):
                edits.append(CodeEdit(
                    edit_type="fix_syntax",
                    location=(i, i),
                    original=line,
                    replacement=line + ':',
                    description="Add missing colon"
                ))
            
            # Unmatched parentheses
            if line.count('(') != line.count(')'):
                edits.append(CodeEdit(
                    edit_type="fix_syntax",
                    location=(i, i),
                    original=line,
                    replacement=line + ')' if line.count('(') > line.count(')') else '(' + line,
                    description="Fix unmatched parentheses"
                ))
        
        return edits
    
    def _get_indent(self, line: str) -> str:
        """Get the indentation of a line."""
        return line[:len(line) - len(line.lstrip())]


class MCTSCodeRefiner:
    """
    Uses MCTS to refine code by exploring edits that improve oracle scores.
    """
    
    def __init__(self,
                 oracle_manager: OracleManager,
                 exploration_constant: float = 1.414,
                 max_iterations: int = 100,
                 max_depth: int = 5):
        self.oracle_manager = oracle_manager
        self.code_editor = CodeEditor()
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        
        # Cache for oracle evaluations
        self.oracle_cache = {}
    
    def refine_code(self,
                   prompt: str,
                   initial_code: str,
                   target_improvements: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Refine code using MCTS to improve oracle scores.
        
        Args:
            prompt: The original prompt
            initial_code: The initial code to refine
            target_improvements: Specific aspects to improve (e.g., ["execution", "complexity"])
            
        Returns:
            refined_code: The improved code
            refinement_info: Information about the refinement process
        """
        # Create root node
        root = MCTSNode(code=initial_code)
        
        # Evaluate initial code
        root.oracle_scores = self._evaluate_code(prompt, initial_code)
        root.total_score = self._compute_score(root.oracle_scores, target_improvements)
        root.visits = 1
        
        # Track best code found
        best_node = root
        best_score = root.total_score
        
        # MCTS iterations
        for iteration in range(self.max_iterations):
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_terminal and len(node.children) < 5:
                child = self._expand(node, prompt)
                if child:
                    node = child
            
            # Simulation
            score = self._simulate(node, prompt, target_improvements)
            
            # Backpropagation
            self._backpropagate(node, score)
            
            # Update best
            if score > best_score:
                best_score = score
                best_node = node
        
        # Collect refinement information
        refinement_info = {
            "iterations": iteration + 1,
            "nodes_explored": self._count_nodes(root),
            "best_score": best_score,
            "initial_score": root.total_score,
            "improvement": best_score - root.total_score,
            "edits_applied": self._get_edit_path(best_node),
            "oracle_scores": {
                "initial": root.oracle_scores,
                "final": best_node.oracle_scores
            }
        }
        
        return best_node.code, refinement_info
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to explore using UCT."""
        current = node
        
        while current.children:
            # Check if we have unexplored children
            unexplored = [child for child in current.children.values() if child.visits == 0]
            
            if unexplored:
                return random.choice(unexplored)
            
            # Select child with highest UCT score
            best_uct = -float('inf')
            best_child = None
            
            for child in current.children.values():
                uct = child.uct_score
                if uct > best_uct:
                    best_uct = uct
                    best_child = child
            
            current = best_child
        
        return current
    
    def _expand(self, node: MCTSNode, prompt: str) -> Optional[MCTSNode]:
        """Expand a node by generating and applying an edit."""
        # Check depth limit
        depth = self._get_depth(node)
        if depth >= self.max_depth:
            node.is_terminal = True
            return None
        
        # Generate possible edits
        edits = self.code_editor.generate_edits(node.code)
        
        # Filter out already tried edits
        untried_edits = []
        for edit in edits:
            edit_key = f"{edit.edit_type}_{edit.location}"
            if edit_key not in node.children:
                untried_edits.append((edit_key, edit))
        
        if not untried_edits:
            node.is_terminal = True
            return None
        
        # Select a random untried edit
        edit_key, edit = random.choice(untried_edits)
        
        # Apply edit
        try:
            new_code = self.code_editor.apply_edit(node.code, edit)
            
            # Verify code is still valid
            ast.parse(new_code)
            
            # Create child node
            child = MCTSNode(
                code=new_code,
                parent=node,
                edit_applied=edit
            )
            
            node.children[edit_key] = child
            return child
            
        except:
            # Edit resulted in invalid code
            return None
    
    def _simulate(self, node: MCTSNode, prompt: str, target_improvements: Optional[List[str]]) -> float:
        """Simulate from a node to estimate its value."""
        # For efficiency, just evaluate the current node
        # In a more sophisticated implementation, we could do rollouts
        
        if node.oracle_scores is None:
            node.oracle_scores = self._evaluate_code(prompt, node.code)
        
        score = self._compute_score(node.oracle_scores, target_improvements)
        return score
    
    def _backpropagate(self, node: MCTSNode, score: float):
        """Backpropagate score up the tree."""
        current = node
        
        while current is not None:
            current.visits += 1
            current.total_score += score
            current = current.parent
    
    def _evaluate_code(self, prompt: str, code: str) -> Dict[str, float]:
        """Evaluate code with oracles (with caching)."""
        cache_key = hash(code)
        
        if cache_key in self.oracle_cache:
            return self.oracle_cache[cache_key]
        
        # Run oracles
        oracle_reports = self.oracle_manager.run_oracles(
            [prompt], [code], [0.5]  # Medium uncertainty
        )
        
        # Extract scores
        scores = {}
        for oracle_name, reports in oracle_reports.items():
            if reports[0] is not None:
                scores[oracle_name] = reports[0].score
        
        self.oracle_cache[cache_key] = scores
        return scores
    
    def _compute_score(self, 
                      oracle_scores: Dict[str, float],
                      target_improvements: Optional[List[str]] = None) -> float:
        """Compute overall score from oracle scores."""
        if not oracle_scores:
            return 0.0
        
        if target_improvements:
            # Focus on specific improvements
            relevant_scores = [
                oracle_scores.get(target, 0.0) 
                for target in target_improvements
                if target in oracle_scores
            ]
            return sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0
        else:
            # Equal weight to all oracles
            return sum(oracle_scores.values()) / len(oracle_scores)
    
    def _get_depth(self, node: MCTSNode) -> int:
        """Get the depth of a node in the tree."""
        depth = 0
        current = node
        
        while current.parent is not None:
            depth += 1
            current = current.parent
        
        return depth
    
    def _count_nodes(self, root: MCTSNode) -> int:
        """Count total nodes in the tree."""
        count = 1
        for child in root.children.values():
            count += self._count_nodes(child)
        return count
    
    def _get_edit_path(self, node: MCTSNode) -> List[Dict[str, Any]]:
        """Get the sequence of edits from root to node."""
        edits = []
        current = node
        
        while current.parent is not None:
            if current.edit_applied:
                edits.append({
                    "type": current.edit_applied.edit_type,
                    "description": current.edit_applied.description,
                    "location": current.edit_applied.location
                })
            current = current.parent
        
        return list(reversed(edits))