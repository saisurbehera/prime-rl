"""Reflective Oracle for LLM-based self-critique and improvement suggestions."""

import re
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .oracle import Oracle, OracleReport


@dataclass
class Critique:
    """Structured critique of code."""
    aspect: str  # 'correctness', 'efficiency', 'readability', 'robustness', 'design'
    severity: str  # 'critical', 'major', 'minor', 'suggestion'
    description: str
    line_range: Optional[Tuple[int, int]] = None
    suggested_fix: Optional[str] = None
    confidence: float = 0.5


class CritiqueGenerator:
    """Generates structured critiques using predefined patterns and heuristics."""
    
    def __init__(self):
        self.critique_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize critique patterns for different aspects."""
        return {
            "correctness": [
                {
                    "pattern": r"def\s+\w+\([^)]*\):\s*\n\s*return",
                    "critique": "Function returns immediately without any logic",
                    "severity": "critical"
                },
                {
                    "pattern": r"except:\s*\n\s*pass",
                    "critique": "Bare except clause that silently ignores all errors",
                    "severity": "major"
                },
                {
                    "pattern": r"while\s+True:",
                    "critique": "Infinite loop without clear break condition",
                    "severity": "major"
                },
            ],
            "efficiency": [
                {
                    "pattern": r"for.*in.*range.*len\(",
                    "critique": "Using range(len()) is inefficient, consider enumerate()",
                    "severity": "minor"
                },
                {
                    "pattern": r"(\w+)\s*=\s*\[\].*for.*\1\.append",
                    "critique": "Building list with append in loop, consider list comprehension",
                    "severity": "minor"
                },
                {
                    "pattern": r"for.*for.*for",
                    "critique": "Triple nested loop detected, consider optimization",
                    "severity": "major"
                },
            ],
            "readability": [
                {
                    "pattern": r"[a-z]\d+",  # Like a1, b2, etc.
                    "critique": "Variable names are not descriptive",
                    "severity": "minor"
                },
                {
                    "pattern": r"if\s+\w+\s*==\s*True:|if\s+\w+\s*==\s*False:",
                    "critique": "Redundant boolean comparison",
                    "severity": "minor"
                },
                {
                    "pattern": r"def\s+\w+\((?:[^,)]+,){5,}",
                    "critique": "Function has too many parameters, consider refactoring",
                    "severity": "major"
                },
            ],
            "robustness": [
                {
                    "pattern": r"def\s+\w+\([^)]*\):[^:]*\n(?!\s*if|\s*assert|\s*try)",
                    "critique": "Function lacks input validation",
                    "severity": "minor"
                },
                {
                    "pattern": r"\[\s*\w+\s*\]",
                    "critique": "Direct indexing without bounds checking",
                    "severity": "minor"
                },
                {
                    "pattern": r"int\(|float\(",
                    "critique": "Type conversion without error handling",
                    "severity": "minor"
                },
            ],
            "design": [
                {
                    "pattern": r"global\s+\w+",
                    "critique": "Use of global variables reduces modularity",
                    "severity": "major"
                },
                {
                    "pattern": r"class\s+\w+:(?:(?!def\s+__init__)[\s\S])*$",
                    "critique": "Class without constructor",
                    "severity": "minor"
                },
                {
                    "pattern": r"def\s+\w+.*\n(?:.*\n){20,}",
                    "critique": "Function is too long, consider breaking it down",
                    "severity": "major"
                },
            ],
        }
    
    def generate_critiques(self, code: str, context: Dict[str, Any]) -> List[Critique]:
        """Generate critiques based on code analysis."""
        critiques = []
        
        # Pattern-based critiques
        for aspect, patterns in self.critique_patterns.items():
            for pattern_info in patterns:
                matches = list(re.finditer(pattern_info["pattern"], code, re.MULTILINE))
                for match in matches:
                    # Find line numbers
                    line_start = code[:match.start()].count('\n') + 1
                    line_end = code[:match.end()].count('\n') + 1
                    
                    critique = Critique(
                        aspect=aspect,
                        severity=pattern_info["severity"],
                        description=pattern_info["critique"],
                        line_range=(line_start, line_end),
                        confidence=0.8
                    )
                    critiques.append(critique)
        
        # Context-aware critiques
        if "task_type" in context:
            task_critiques = self._generate_task_specific_critiques(code, context["task_type"])
            critiques.extend(task_critiques)
        
        # Code smell detection
        smell_critiques = self._detect_code_smells(code)
        critiques.extend(smell_critiques)
        
        return critiques
    
    def _generate_task_specific_critiques(self, code: str, task_type: str) -> List[Critique]:
        """Generate critiques specific to the task type."""
        critiques = []
        
        if "sort" in task_type.lower():
            if not re.search(r"sorted|sort|bubble|quick|merge", code, re.IGNORECASE):
                critiques.append(Critique(
                    aspect="correctness",
                    severity="major",
                    description="Code doesn't appear to implement sorting",
                    confidence=0.7
                ))
        
        elif "factorial" in task_type.lower():
            if not re.search(r"factorial|fact", code, re.IGNORECASE):
                critiques.append(Critique(
                    aspect="correctness",
                    severity="minor",
                    description="Function name doesn't clearly indicate factorial",
                    confidence=0.6
                ))
        
        elif "palindrome" in task_type.lower():
            if not re.search(r"\[::-1\]|reverse", code):
                critiques.append(Critique(
                    aspect="efficiency",
                    severity="suggestion",
                    description="Consider using slicing [::-1] for palindrome check",
                    confidence=0.5
                ))
        
        return critiques
    
    def _detect_code_smells(self, code: str) -> List[Critique]:
        """Detect common code smells."""
        critiques = []
        lines = code.split('\n')
        
        # Long lines
        for i, line in enumerate(lines):
            if len(line) > 100:
                critiques.append(Critique(
                    aspect="readability",
                    severity="minor",
                    description=f"Line {i+1} is too long ({len(line)} chars)",
                    line_range=(i+1, i+1),
                    confidence=0.9
                ))
        
        # Magic numbers
        if re.search(r'(?<!\w)\d{2,}(?!\w)', code):
            critiques.append(Critique(
                aspect="readability",
                severity="minor",
                description="Magic numbers detected, consider using named constants",
                confidence=0.7
            ))
        
        # Duplicate code detection (simple)
        line_counts = {}
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 10 and not stripped.startswith('#'):
                line_counts[stripped] = line_counts.get(stripped, 0) + 1
        
        for line, count in line_counts.items():
            if count > 2:
                critiques.append(Critique(
                    aspect="design",
                    severity="major",
                    description=f"Duplicate code detected: '{line[:30]}...' appears {count} times",
                    confidence=0.8
                ))
                break
        
        return critiques


class ReflectiveOracle(Oracle):
    """Oracle that generates self-critiques and improvement suggestions."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.critique_generator = CritiqueGenerator()
        self.max_critiques = config.get("max_critiques", 5)
        
        # Neural critique model
        hidden_dim = config.get("hidden_dim", 768)
        self.critique_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),  # Code + prompt embeddings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
        ).to(self.device)
        
        # Heads for different critique aspects
        self.aspect_heads = nn.ModuleDict({
            aspect: nn.Linear(256, 4)  # 4 severity levels
            for aspect in ["correctness", "efficiency", "readability", "robustness", "design"]
        }).to(self.device)
        
        self.confidence_head = nn.Linear(256, 1).to(self.device)
    
    def should_run(self, prompt: str, candidate_code: str, uncertainty_score: float) -> bool:
        """Always run reflective oracle as it provides valuable feedback."""
        return True
    
    def evaluate(self, prompt: str, candidate_code: str) -> OracleReport:
        """Generate critiques and improvement suggestions."""
        # Extract task context from prompt
        context = self._extract_context(prompt)
        
        # Generate critiques
        critiques = self.critique_generator.generate_critiques(candidate_code, context)
        
        # Sort by severity and limit
        severity_order = {"critical": 0, "major": 1, "minor": 2, "suggestion": 3}
        critiques.sort(key=lambda c: (severity_order.get(c.severity, 4), -c.confidence))
        critiques = critiques[:self.max_critiques]
        
        # Calculate score based on critiques
        if not critiques:
            score = 1.0
        else:
            severity_weights = {"critical": 0.4, "major": 0.2, "minor": 0.1, "suggestion": 0.05}
            total_penalty = sum(
                severity_weights.get(c.severity, 0) * c.confidence 
                for c in critiques
            )
            score = max(0, 1 - total_penalty)
        
        # Generate improvement suggestions
        suggestions = self._generate_suggestions(critiques, candidate_code)
        
        report = OracleReport(
            oracle_name="reflective",
            score=score,
            details={
                "critiques": [
                    {
                        "aspect": c.aspect,
                        "severity": c.severity,
                        "description": c.description,
                        "line_range": c.line_range,
                        "confidence": c.confidence,
                        "suggested_fix": c.suggested_fix
                    }
                    for c in critiques
                ],
                "num_critiques": len(critiques),
                "suggestions": suggestions,
                "context": context
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
        """Calculate loss based on critique generation."""
        critiques = report.details["critiques"]
        
        if not critiques:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Pool hidden states
        if hidden_states.dim() == 3:
            code_embedding = hidden_states.mean(dim=1)
        else:
            code_embedding = hidden_states
        
        # Create prompt embedding (simplified - use same as code)
        prompt_embedding = code_embedding
        
        # Encode code and prompt
        combined = torch.cat([code_embedding, prompt_embedding], dim=-1)
        features = self.critique_encoder(combined)
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # For each critique, compute loss
        for critique in critiques:
            aspect = critique["aspect"]
            severity = critique["severity"]
            confidence = critique["confidence"]
            
            if aspect in self.aspect_heads:
                # Get predictions for this aspect
                severity_logits = self.aspect_heads[aspect](features)
                confidence_pred = torch.sigmoid(self.confidence_head(features))
                
                # Create target
                severity_map = {"critical": 0, "major": 1, "minor": 2, "suggestion": 3}
                severity_target = torch.tensor(
                    severity_map.get(severity, 3), 
                    device=self.device
                )
                
                # Severity classification loss
                severity_loss = F.cross_entropy(
                    severity_logits, 
                    severity_target.unsqueeze(0).expand(severity_logits.shape[0])
                )
                
                # Confidence regression loss
                confidence_target = torch.tensor(confidence, device=self.device)
                confidence_loss = F.mse_loss(confidence_pred.squeeze(), confidence_target)
                
                # Combine losses
                aspect_loss = severity_loss + 0.1 * confidence_loss
                total_loss = total_loss + aspect_loss
        
        # Average over all critiques
        total_loss = total_loss / len(critiques)
        
        # Add regularization to encourage finding critiques
        critique_penalty = torch.relu(0.3 - len(critiques) / self.max_critiques)
        total_loss = total_loss + 0.1 * critique_penalty
        
        return total_loss
    
    def _extract_context(self, prompt: str) -> Dict[str, Any]:
        """Extract context from prompt."""
        context = {}
        
        # Detect task type
        task_keywords = {
            "sort": ["sort", "arrange", "order"],
            "search": ["find", "search", "locate"],
            "factorial": ["factorial"],
            "prime": ["prime"],
            "palindrome": ["palindrome"],
            "reverse": ["reverse"],
            "fibonacci": ["fibonacci"],
        }
        
        prompt_lower = prompt.lower()
        for task, keywords in task_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                context["task_type"] = task
                break
        
        # Extract constraints
        if "efficient" in prompt_lower or "optimize" in prompt_lower:
            context["requires_efficiency"] = True
        
        if "handle" in prompt_lower or "error" in prompt_lower:
            context["requires_error_handling"] = True
        
        return context
    
    def _generate_suggestions(self, critiques: List[Critique], code: str) -> List[str]:
        """Generate improvement suggestions based on critiques."""
        suggestions = []
        
        # Group critiques by aspect
        aspect_critiques = {}
        for critique in critiques:
            aspect = critique["aspect"] if isinstance(critique, dict) else critique.aspect
            if aspect not in aspect_critiques:
                aspect_critiques[aspect] = []
            aspect_critiques[aspect].append(critique)
        
        # Generate aspect-specific suggestions
        if "correctness" in aspect_critiques:
            suggestions.append("Focus on correctness: Add input validation and handle edge cases")
        
        if "efficiency" in aspect_critiques:
            suggestions.append("Optimize performance: Consider algorithmic improvements or better data structures")
        
        if "readability" in aspect_critiques:
            suggestions.append("Improve readability: Use descriptive variable names and add comments")
        
        if "robustness" in aspect_critiques:
            suggestions.append("Enhance robustness: Add error handling and validate inputs")
        
        if "design" in aspect_critiques:
            suggestions.append("Refactor design: Break down complex functions and reduce coupling")
        
        # Add specific suggestions based on severity
        critical_count = sum(
            1 for c in critiques 
            if (c["severity"] if isinstance(c, dict) else c.severity) == "critical"
        )
        
        if critical_count > 0:
            suggestions.insert(0, f"Address {critical_count} critical issues before deployment")
        
        return suggestions[:3]  # Limit to top 3 suggestions