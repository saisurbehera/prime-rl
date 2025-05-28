"""Integration module for Tri-Oracle system with PrimeRL training pipeline."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from .oracle import (
    OracleManager, 
    MetaGatingNetwork, 
    aggregate_oracle_losses
)
from .execution_oracle import ExecutionOracle
from .static_analysis_oracle import StaticAnalysisOracle
from .complexity_oracle import ComplexityOracle
from .documentation_oracle import DocumentationOracle
from .proof_oracle import ProofOracle
from .reflective_oracle import ReflectiveOracle


@dataclass
class OracleConfig:
    """Configuration for Oracle system."""
    # Oracle activation settings
    use_execution_oracle: bool = True
    use_static_oracle: bool = True
    use_complexity_oracle: bool = True
    use_documentation_oracle: bool = True
    use_proof_oracle: bool = False  # Will implement later
    use_reflective_oracle: bool = False  # Will implement later
    
    # Uncertainty thresholds
    execution_uncertainty_threshold: float = 0.3
    proof_uncertainty_threshold: float = 0.5
    
    # Loss weights (if not using meta-gating)
    execution_weight: float = 0.3
    static_weight: float = 0.1
    complexity_weight: float = 0.1
    documentation_weight: float = 0.1
    proof_weight: float = 0.2
    reflective_weight: float = 0.2
    
    # Meta-gating settings
    use_meta_gating: bool = True
    meta_gating_hidden_dim: int = 256
    
    # Execution oracle settings
    execution_timeout: int = 5
    max_tests_per_function: int = 5
    
    # Static analysis settings
    linters: List[str] = field(default_factory=lambda: ["flake8"])
    
    # Complexity settings
    target_complexity: int = 10
    max_acceptable_complexity: int = 20
    
    # Documentation settings
    min_docstring_length: int = 10
    require_param_docs: bool = True
    require_return_docs: bool = True
    
    # Device settings
    device: str = "cuda"


class OracleIntegration:
    """Integrates Oracle system with PrimeRL training."""
    
    def __init__(self, config: OracleConfig, model_hidden_dim: int = 768):
        self.config = config
        self.model_hidden_dim = model_hidden_dim
        
        # Initialize oracle manager
        self.oracle_manager = OracleManager({"device": config.device})
        
        # Register active oracles
        self._register_oracles()
        
        # Initialize meta-gating network if enabled
        if config.use_meta_gating:
            num_oracles = len(self.oracle_manager.oracles)
            # Input: prompt_embed + code_embed + oracle_scores + uncertainty
            input_dim = model_hidden_dim * 2 + num_oracles + 1
            self.meta_gating_network = MetaGatingNetwork(
                input_dim=input_dim,
                num_oracles=num_oracles,
                hidden_dim=config.meta_gating_hidden_dim
            ).to(config.device)
        else:
            self.meta_gating_network = None
            
        # Oracle name order for consistent weight mapping
        self.oracle_names = list(self.oracle_manager.oracles.keys())
        
    def _register_oracles(self):
        """Register active oracles based on configuration."""
        oracle_configs = {
            "device": self.config.device,
            "timeout": self.config.execution_timeout,
            "max_tests": self.config.max_tests_per_function,
            "uncertainty_threshold": self.config.execution_uncertainty_threshold,
            "linters": self.config.linters,
            "target_complexity": self.config.target_complexity,
            "max_acceptable_complexity": self.config.max_acceptable_complexity,
            "min_docstring_length": self.config.min_docstring_length,
            "require_param_docs": self.config.require_param_docs,
            "require_return_docs": self.config.require_return_docs,
        }
        
        if self.config.use_execution_oracle:
            self.oracle_manager.register_oracle(
                "execution", 
                ExecutionOracle(oracle_configs)
            )
            
        if self.config.use_static_oracle:
            self.oracle_manager.register_oracle(
                "static_analysis",
                StaticAnalysisOracle(oracle_configs)
            )
            
        if self.config.use_complexity_oracle:
            self.oracle_manager.register_oracle(
                "complexity",
                ComplexityOracle(oracle_configs)
            )
            
        if self.config.use_documentation_oracle:
            self.oracle_manager.register_oracle(
                "documentation",
                DocumentationOracle(oracle_configs)
            )
            
        if self.config.use_proof_oracle:
            self.oracle_manager.register_oracle(
                "proof",
                ProofOracle(oracle_configs)
            )
            
        if self.config.use_reflective_oracle:
            self.oracle_manager.register_oracle(
                "reflective",
                ReflectiveOracle(oracle_configs)
            )
    
    def compute_oracle_losses(
        self,
        prompts: List[str],
        generated_codes: List[str],
        hidden_states: torch.Tensor,
        uncertainty_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute oracle losses for generated code.
        
        Args:
            prompts: List of prompts
            generated_codes: List of generated code strings
            hidden_states: Model hidden states [batch_size, seq_len, hidden_dim]
            uncertainty_scores: Optional uncertainty scores [batch_size]
            
        Returns:
            total_loss: Aggregated oracle loss
            metrics: Dictionary of oracle metrics
        """
        batch_size = len(prompts)
        
        # Calculate uncertainty if not provided
        if uncertainty_scores is None:
            uncertainty_scores = self._calculate_uncertainty(hidden_states)
        
        # Run oracles
        oracle_reports = self.oracle_manager.run_oracles(
            prompts, 
            generated_codes,
            uncertainty_scores.detach().cpu().numpy().tolist()
        )
        
        # Calculate individual oracle losses
        oracle_losses = self.oracle_manager.calculate_losses(
            prompts,
            generated_codes,
            hidden_states,
            oracle_reports
        )
        
        # Aggregate losses
        if self.config.use_meta_gating and self.meta_gating_network is not None:
            # Use meta-gating to weight oracle losses
            total_loss = self._aggregate_with_meta_gating(
                prompts,
                generated_codes,
                hidden_states,
                oracle_reports,
                oracle_losses,
                uncertainty_scores
            )
        else:
            # Use fixed weights
            weights = self._get_fixed_weights()
            total_loss = aggregate_oracle_losses(
                oracle_losses,
                weights,
                self.oracle_names
            )
        
        # Collect metrics
        metrics = self._collect_metrics(oracle_reports, oracle_losses)
        
        return total_loss, metrics
    
    def _calculate_uncertainty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Calculate uncertainty scores from hidden states."""
        # Simple uncertainty: entropy of attention weights or hidden state variance
        # Here we use variance of hidden states as a proxy
        if hidden_states.dim() == 3:
            # [batch_size, seq_len, hidden_dim] -> [batch_size]
            variance = hidden_states.var(dim=(1, 2))
        else:
            variance = hidden_states.var(dim=1)
            
        # Normalize to 0-1 range
        uncertainty = torch.sigmoid(variance - variance.mean())
        return uncertainty
    
    def _aggregate_with_meta_gating(
        self,
        prompts: List[str],
        generated_codes: List[str],
        hidden_states: torch.Tensor,
        oracle_reports: Dict[str, List],
        oracle_losses: Dict[str, torch.Tensor],
        uncertainty_scores: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate oracle losses using meta-gating network."""
        batch_size = len(prompts)
        
        # Get embeddings (using mean pooling)
        if hidden_states.dim() == 3:
            # Mean pool over sequence length
            prompt_embeddings = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
            code_embeddings = hidden_states.mean(dim=1)  # Using same for simplicity
        else:
            prompt_embeddings = hidden_states
            code_embeddings = hidden_states
        
        # Collect oracle scores
        oracle_scores = torch.zeros(
            batch_size, 
            len(self.oracle_names),
            device=self.config.device
        )
        
        for i, oracle_name in enumerate(self.oracle_names):
            if oracle_name in oracle_reports:
                reports = oracle_reports[oracle_name]
                for j, report in enumerate(reports):
                    if report is not None:
                        oracle_scores[j, i] = report.score
        
        # Get meta-gating weights
        weights = self.meta_gating_network(
            prompt_embeddings,
            code_embeddings,
            oracle_scores,
            uncertainty_scores.unsqueeze(1)
        )  # [batch_size, num_oracles]
        
        # Aggregate losses with learned weights
        total_loss = torch.tensor(0.0, device=self.config.device, requires_grad=True)
        
        for i, oracle_name in enumerate(self.oracle_names):
            if oracle_name in oracle_losses:
                # Weight each sample's loss by its specific weight
                weighted_loss = weights[:, i].mean() * oracle_losses[oracle_name]
                total_loss = total_loss + weighted_loss
        
        return total_loss
    
    def _get_fixed_weights(self) -> torch.Tensor:
        """Get fixed oracle weights from configuration."""
        weight_map = {
            "execution": self.config.execution_weight,
            "static_analysis": self.config.static_weight,
            "complexity": self.config.complexity_weight,
            "documentation": self.config.documentation_weight,
            "proof": self.config.proof_weight,
            "reflective": self.config.reflective_weight,
        }
        
        weights = []
        for oracle_name in self.oracle_names:
            weights.append(weight_map.get(oracle_name, 0.1))
            
        weights_tensor = torch.tensor(weights, device=self.config.device)
        return weights_tensor / weights_tensor.sum()
    
    def _collect_metrics(
        self, 
        oracle_reports: Dict[str, List],
        oracle_losses: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Collect metrics from oracle reports."""
        metrics = {}
        
        # Collect average scores and losses for each oracle
        for oracle_name, reports in oracle_reports.items():
            valid_reports = [r for r in reports if r is not None]
            
            if valid_reports:
                avg_score = np.mean([r.score for r in valid_reports])
                metrics[f"{oracle_name}_score"] = avg_score
                
                # Add specific metrics for each oracle
                if oracle_name == "execution":
                    total_tests = sum(r.details["num_tests_run"] for r in valid_reports)
                    total_passed = sum(r.details["num_tests_passed"] for r in valid_reports)
                    metrics["execution_pass_rate"] = total_passed / total_tests if total_tests > 0 else 0.0
                    
                elif oracle_name == "static_analysis":
                    total_errors = sum(r.details["num_errors"] for r in valid_reports)
                    total_warnings = sum(r.details["num_warnings"] for r in valid_reports)
                    metrics["static_errors"] = total_errors / len(valid_reports)
                    metrics["static_warnings"] = total_warnings / len(valid_reports)
                    
                elif oracle_name == "complexity":
                    avg_complexity = np.mean([r.details["cyclomatic_complexity"] for r in valid_reports])
                    avg_maintainability = np.mean([r.details["maintainability_index"] for r in valid_reports])
                    metrics["avg_complexity"] = avg_complexity
                    metrics["avg_maintainability"] = avg_maintainability
                    
                elif oracle_name == "documentation":
                    func_with_docs = sum(r.details["functions_with_docstrings"] for r in valid_reports)
                    total_funcs = sum(r.details["total_functions"] for r in valid_reports)
                    metrics["doc_coverage"] = func_with_docs / total_funcs if total_funcs > 0 else 1.0
            
            # Add losses
            if oracle_name in oracle_losses:
                metrics[f"{oracle_name}_loss"] = oracle_losses[oracle_name].item()
        
        return metrics
    
    def get_oracle_feedback_for_inference(
        self,
        prompts: List[str],
        generated_codes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get oracle feedback for inference (without computing losses).
        
        Args:
            prompts: List of prompts
            generated_codes: List of generated code strings
            
        Returns:
            List of feedback dictionaries for each sample
        """
        # Run oracles without uncertainty gating (run all oracles)
        uncertainty_scores = [0.0] * len(prompts)  # Low uncertainty to run all
        
        oracle_reports = self.oracle_manager.run_oracles(
            prompts,
            generated_codes,
            uncertainty_scores
        )
        
        # Format feedback for each sample
        feedback_list = []
        for i in range(len(prompts)):
            feedback = {
                "prompt": prompts[i],
                "generated_code": generated_codes[i],
                "oracle_scores": {}
            }
            
            for oracle_name, reports in oracle_reports.items():
                if reports[i] is not None:
                    feedback["oracle_scores"][oracle_name] = {
                        "score": reports[i].score,
                        "details": reports[i].details
                    }
            
            feedback_list.append(feedback)
        
        return feedback_list


def create_oracle_integration(
    model_hidden_dim: int = 768,
    **kwargs
) -> OracleIntegration:
    """Factory function to create OracleIntegration with config."""
    config = OracleConfig(**kwargs)
    return OracleIntegration(config, model_hidden_dim)