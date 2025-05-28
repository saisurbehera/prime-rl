"""Proof Oracle for formal verification and type checking."""

import ast
import re
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .oracle import Oracle, OracleReport


@dataclass
class ProofObligation:
    """Represents a proof obligation to be verified."""
    name: str
    property_type: str  # 'invariant', 'precondition', 'postcondition', 'type'
    expression: str
    function_name: Optional[str] = None
    line_number: Optional[int] = None


class NeuralProver(nn.Module):
    """Simple neural theorem prover for differentiable proof checking."""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # Code + property embeddings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.proof_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # proven, disproven, unknown
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, code_embedding: torch.Tensor, property_embedding: torch.Tensor):
        """
        Forward pass of neural prover.
        
        Returns:
            proof_logits: [batch_size, 3] logits for proven/disproven/unknown
            confidence: [batch_size, 1] confidence score
        """
        combined = torch.cat([code_embedding, property_embedding], dim=-1)
        features = self.encoder(combined)
        
        proof_logits = self.proof_head(features)
        confidence = self.confidence_head(features)
        
        return proof_logits, confidence


class PropertyExtractor:
    """Extracts formal properties from code and docstrings."""
    
    def extract_properties(self, code: str, prompt: str) -> List[ProofObligation]:
        """Extract proof obligations from code."""
        obligations = []
        
        # Extract from docstrings
        obligations.extend(self._extract_from_docstrings(code))
        
        # Extract from assert statements
        obligations.extend(self._extract_from_asserts(code))
        
        # Extract from type hints
        obligations.extend(self._extract_from_type_hints(code))
        
        # Extract from prompt
        obligations.extend(self._extract_from_prompt(prompt, code))
        
        # Add default properties for common patterns
        obligations.extend(self._add_default_properties(code))
        
        return obligations
    
    def _extract_from_docstrings(self, code: str) -> List[ProofObligation]:
        """Extract properties from docstring annotations."""
        obligations = []
        
        # Look for preconditions/postconditions in docstrings
        pre_pattern = r'(?:Pre|Precondition|Requires?):\s*(.+?)(?:\n|$)'
        post_pattern = r'(?:Post|Postcondition|Ensures?):\s*(.+?)(?:\n|$)'
        invariant_pattern = r'(?:Invariant|Maintains?):\s*(.+?)(?:\n|$)'
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # Extract preconditions
                        for match in re.finditer(pre_pattern, docstring, re.IGNORECASE):
                            obligations.append(ProofObligation(
                                name=f"{node.name}_precondition",
                                property_type="precondition",
                                expression=match.group(1).strip(),
                                function_name=node.name,
                                line_number=node.lineno
                            ))
                        
                        # Extract postconditions
                        for match in re.finditer(post_pattern, docstring, re.IGNORECASE):
                            obligations.append(ProofObligation(
                                name=f"{node.name}_postcondition",
                                property_type="postcondition",
                                expression=match.group(1).strip(),
                                function_name=node.name,
                                line_number=node.lineno
                            ))
                        
                        # Extract invariants
                        for match in re.finditer(invariant_pattern, docstring, re.IGNORECASE):
                            obligations.append(ProofObligation(
                                name=f"{node.name}_invariant",
                                property_type="invariant",
                                expression=match.group(1).strip(),
                                function_name=node.name,
                                line_number=node.lineno
                            ))
        except:
            pass
        
        return obligations
    
    def _extract_from_asserts(self, code: str) -> List[ProofObligation]:
        """Extract properties from assert statements."""
        obligations = []
        
        try:
            tree = ast.parse(code)
            
            class AssertVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.obligations = []
                    self.current_function = None
                
                def visit_FunctionDef(self, node):
                    old_func = self.current_function
                    self.current_function = node.name
                    self.generic_visit(node)
                    self.current_function = old_func
                
                def visit_Assert(self, node):
                    # Convert assert expression to string
                    try:
                        expr_str = ast.unparse(node.test)
                    except:
                        expr_str = "unknown_assertion"
                    
                    self.obligations.append(ProofObligation(
                        name=f"assert_{len(self.obligations)}",
                        property_type="invariant",
                        expression=expr_str,
                        function_name=self.current_function,
                        line_number=node.lineno
                    ))
            
            visitor = AssertVisitor()
            visitor.visit(tree)
            obligations = visitor.obligations
            
        except:
            pass
        
        return obligations
    
    def _extract_from_type_hints(self, code: str) -> List[ProofObligation]:
        """Extract type constraints from type hints."""
        obligations = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check return type
                    if node.returns:
                        try:
                            return_type = ast.unparse(node.returns)
                            obligations.append(ProofObligation(
                                name=f"{node.name}_return_type",
                                property_type="type",
                                expression=f"return value is {return_type}",
                                function_name=node.name,
                                line_number=node.lineno
                            ))
                        except:
                            pass
                    
                    # Check parameter types
                    for arg in node.args.args:
                        if arg.annotation:
                            try:
                                param_type = ast.unparse(arg.annotation)
                                obligations.append(ProofObligation(
                                    name=f"{node.name}_{arg.arg}_type",
                                    property_type="type",
                                    expression=f"{arg.arg} is {param_type}",
                                    function_name=node.name,
                                    line_number=node.lineno
                                ))
                            except:
                                pass
        except:
            pass
        
        return obligations
    
    def _extract_from_prompt(self, prompt: str, code: str) -> List[ProofObligation]:
        """Extract implied properties from the prompt."""
        obligations = []
        prompt_lower = prompt.lower()
        
        # Extract function name from code
        func_name = None
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    break
        except:
            pass
        
        if not func_name:
            return obligations
        
        # Common patterns in prompts that imply properties
        patterns = [
            (r'positive|non-negative', f'{func_name}(x) >= 0 for all valid x'),
            (r'sorted|ascending', f'{func_name}(arr) is sorted in ascending order'),
            (r'unique|distinct', f'all elements in {func_name}(x) are unique'),
            (r'palindrome', f'{func_name}(s) returns true iff s is palindrome'),
            (r'prime', f'{func_name}(n) returns true iff n is prime'),
            (r'factorial', f'{func_name}(n) == n! for n >= 0'),
            (r'fibonacci', f'{func_name}(n) returns nth Fibonacci number'),
            (r'gcd|greatest common divisor', f'{func_name}(a,b) == gcd(a,b)'),
            (r'reverse', f'{func_name}(x) reverses x'),
        ]
        
        for pattern, property_expr in patterns:
            if re.search(pattern, prompt_lower):
                obligations.append(ProofObligation(
                    name=f"{func_name}_implied_property",
                    property_type="postcondition",
                    expression=property_expr,
                    function_name=func_name
                ))
        
        return obligations
    
    def _add_default_properties(self, code: str) -> List[ProofObligation]:
        """Add default properties based on code patterns."""
        obligations = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for recursion
                    has_recursion = False
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                            if child.func.id == node.name:
                                has_recursion = True
                                break
                    
                    if has_recursion:
                        obligations.append(ProofObligation(
                            name=f"{node.name}_termination",
                            property_type="invariant",
                            expression=f"{node.name} terminates for all valid inputs",
                            function_name=node.name
                        ))
                    
                    # Check for loops
                    has_loop = any(isinstance(child, (ast.For, ast.While)) 
                                  for child in ast.walk(node))
                    
                    if has_loop:
                        obligations.append(ProofObligation(
                            name=f"{node.name}_loop_termination",
                            property_type="invariant",
                            expression=f"all loops in {node.name} terminate",
                            function_name=node.name
                        ))
        except:
            pass
        
        return obligations


class ProofOracle(Oracle):
    """Oracle that performs formal verification on code."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.property_extractor = PropertyExtractor()
        self.uncertainty_threshold = config.get("proof_uncertainty_threshold", 0.5)
        
        # Initialize neural prover
        self.neural_prover = NeuralProver(
            input_dim=config.get("hidden_dim", 768),
            hidden_dim=config.get("prover_hidden_dim", 512)
        ).to(self.device)
        
        # Simple embedding layers for properties
        self.property_encoder = nn.Sequential(
            nn.Linear(100, 256),  # Assuming max 100 char properties
            nn.ReLU(),
            nn.Linear(256, config.get("hidden_dim", 768))
        ).to(self.device)
    
    def should_run(self, prompt: str, candidate_code: str, uncertainty_score: float) -> bool:
        """Run proof oracle for high-uncertainty code."""
        return uncertainty_score > self.uncertainty_threshold
    
    def evaluate(self, prompt: str, candidate_code: str) -> OracleReport:
        """Extract and verify properties of the code."""
        # Extract proof obligations
        obligations = self.property_extractor.extract_properties(candidate_code, prompt)
        
        # Verify each obligation
        verification_results = {}
        proven_count = 0
        disproven_count = 0
        unknown_count = 0
        
        for obligation in obligations:
            result = self._verify_property(candidate_code, obligation)
            verification_results[obligation.name] = result
            
            if result["status"] == "proven":
                proven_count += 1
            elif result["status"] == "disproven":
                disproven_count += 1
            else:
                unknown_count += 1
        
        # Calculate overall score
        total_obligations = len(obligations)
        if total_obligations > 0:
            # Proven properties are good, disproven are bad, unknown are neutral
            score = (proven_count - disproven_count) / total_obligations
            score = (score + 1) / 2  # Normalize to 0-1
        else:
            score = 0.5  # No properties to verify
        
        report = OracleReport(
            oracle_name="proof",
            score=score,
            details={
                "obligations": [vars(o) for o in obligations],
                "verification_results": verification_results,
                "proven_count": proven_count,
                "disproven_count": disproven_count,
                "unknown_count": unknown_count,
                "total_obligations": total_obligations
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
        """Calculate differentiable loss from proof verification."""
        verification_results = report.details["verification_results"]
        
        if not verification_results:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Pool hidden states
        if hidden_states.dim() == 3:
            pooled_hidden = hidden_states.mean(dim=1)
        else:
            pooled_hidden = hidden_states
        
        # For each verification result, compute loss
        for prop_name, result in verification_results.items():
            status = result["status"]
            confidence = result["confidence"]
            
            # Create target based on status
            if status == "proven":
                target = torch.tensor([1.0, 0.0, 0.0], device=self.device)  # proven
            elif status == "disproven":
                target = torch.tensor([0.0, 1.0, 0.0], device=self.device)  # disproven
            else:
                target = torch.tensor([0.0, 0.0, 1.0], device=self.device)  # unknown
            
            # Encode property (simplified - just use random for now)
            property_embedding = torch.randn(pooled_hidden.shape[0], 768, device=self.device)
            
            # Get neural prover predictions
            proof_logits, pred_confidence = self.neural_prover(pooled_hidden, property_embedding)
            
            # Cross-entropy loss for proof status
            status_loss = F.cross_entropy(proof_logits, target.unsqueeze(0).expand(proof_logits.shape[0], -1))
            
            # MSE loss for confidence
            confidence_target = torch.tensor(confidence, device=self.device)
            confidence_loss = F.mse_loss(pred_confidence.squeeze(), confidence_target)
            
            # Combine losses
            prop_loss = status_loss + 0.1 * confidence_loss
            total_loss = total_loss + prop_loss
        
        # Average over all properties
        total_loss = total_loss / len(verification_results)
        
        return total_loss
    
    def _verify_property(self, code: str, obligation: ProofObligation) -> Dict[str, Any]:
        """Verify a single property (simplified heuristic verification)."""
        # In a real implementation, this would use Z3, Dafny, or other verifiers
        # For now, use heuristics based on property type and code analysis
        
        result = {
            "property": obligation.expression,
            "type": obligation.property_type,
            "status": "unknown",
            "confidence": 0.5,
            "explanation": ""
        }
        
        try:
            tree = ast.parse(code)
            
            # Type checking
            if obligation.property_type == "type":
                if "return value is" in obligation.expression:
                    # Check if function has return statement
                    func_node = self._find_function(tree, obligation.function_name)
                    if func_node and self._has_return(func_node):
                        result["status"] = "proven"
                        result["confidence"] = 0.8
                        result["explanation"] = "Function has return statement"
                    else:
                        result["status"] = "disproven"
                        result["confidence"] = 0.9
                        result["explanation"] = "Function missing return statement"
            
            # Termination checking
            elif "terminates" in obligation.expression:
                func_node = self._find_function(tree, obligation.function_name)
                if func_node:
                    if self._has_base_case(func_node):
                        result["status"] = "proven"
                        result["confidence"] = 0.7
                        result["explanation"] = "Function has base case for recursion"
                    else:
                        result["status"] = "unknown"
                        result["confidence"] = 0.4
                        result["explanation"] = "Cannot determine termination"
            
            # Property-specific verification
            elif "factorial" in obligation.expression:
                if self._verifies_factorial_pattern(code):
                    result["status"] = "proven"
                    result["confidence"] = 0.9
                    result["explanation"] = "Matches factorial pattern"
            
            elif "prime" in obligation.expression:
                if self._verifies_prime_pattern(code):
                    result["status"] = "proven"
                    result["confidence"] = 0.85
                    result["explanation"] = "Implements prime checking algorithm"
            
            elif "sorted" in obligation.expression:
                if self._verifies_sorting_pattern(code):
                    result["status"] = "proven"
                    result["confidence"] = 0.8
                    result["explanation"] = "Implements sorting algorithm"
            
        except:
            result["explanation"] = "Failed to parse code"
        
        return result
    
    def _find_function(self, tree: ast.AST, func_name: str) -> Optional[ast.FunctionDef]:
        """Find function node by name."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node
        return None
    
    def _has_return(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has return statement."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.value is not None:
                return True
        return False
    
    def _has_base_case(self, func_node: ast.FunctionDef) -> bool:
        """Check if recursive function has base case."""
        # Look for if statement with return before recursive call
        has_conditional_return = False
        has_recursive_call = False
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                # Check if this if has a return
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        has_conditional_return = True
                        break
            
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == func_node.name:
                    has_recursive_call = True
        
        return has_conditional_return and has_recursive_call
    
    def _verifies_factorial_pattern(self, code: str) -> bool:
        """Check if code implements factorial correctly."""
        # Simple pattern matching
        patterns = [
            r'if\s+n\s*[<=]=\s*[01].*return\s+1',  # Base case
            r'return\s+n\s*\*.*factorial\s*\(\s*n\s*-\s*1\s*\)',  # Recursive case
            r'result\s*=\s*1.*for.*range.*result\s*\*=',  # Iterative pattern
        ]
        
        matches = sum(1 for pattern in patterns if re.search(pattern, code, re.IGNORECASE))
        return matches >= 2
    
    def _verifies_prime_pattern(self, code: str) -> bool:
        """Check if code implements prime checking."""
        patterns = [
            r'if\s+n\s*<=?\s*1.*return\s+False',  # Check for n <= 1
            r'for.*range.*2.*sqrt|for.*range.*2.*n',  # Loop from 2
            r'if\s+n\s*%.*==\s*0.*return\s+False',  # Divisibility check
        ]
        
        matches = sum(1 for pattern in patterns if re.search(pattern, code, re.IGNORECASE))
        return matches >= 2
    
    def _verifies_sorting_pattern(self, code: str) -> bool:
        """Check if code implements sorting."""
        patterns = [
            r'for.*for',  # Nested loops (bubble sort, etc)
            r'pivot|partition',  # Quicksort patterns
            r'merge.*left.*right',  # Merge sort patterns
            r'sorted\(',  # Built-in sort
        ]
        
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)