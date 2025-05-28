"""Base Oracle framework for Tri-Oracle LLM system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn


@dataclass
class OracleReport:
    """Base class for structured oracle output."""
    oracle_name: str
    score: float
    details: Dict[str, Any]
    loss: Optional[torch.Tensor] = None


class Oracle(ABC):
    """Base Oracle interface for all oracle implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cuda")
    
    @abstractmethod
    def should_run(self, prompt: str, candidate_code: str, uncertainty_score: float) -> bool:
        """Determines if the oracle should run based on uncertainty or other criteria."""
        pass
    
    @abstractmethod
    def evaluate(self, prompt: str, candidate_code: str) -> OracleReport:
        """Evaluates the candidate code and returns a structured report."""
        pass
    
    @abstractmethod
    def calculate_loss(
        self, 
        prompt: str, 
        candidate_code: str, 
        hidden_states: torch.Tensor,
        report: OracleReport
    ) -> torch.Tensor:
        """Calculates a differentiable loss based on the evaluation."""
        pass


class OracleManager:
    """Manages multiple oracles and their execution."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.oracles: Dict[str, Oracle] = {}
        self.device = config.get("device", "cuda")
        
    def register_oracle(self, name: str, oracle: Oracle):
        """Register an oracle with the manager."""
        self.oracles[name] = oracle
        
    def run_oracles(
        self, 
        prompts: List[str], 
        candidate_codes: List[str],
        uncertainty_scores: Optional[List[float]] = None
    ) -> Dict[str, List[OracleReport]]:
        """Run all applicable oracles on the given inputs."""
        if uncertainty_scores is None:
            uncertainty_scores = [0.5] * len(prompts)
            
        results = {oracle_name: [] for oracle_name in self.oracles}
        
        for prompt, code, uncertainty in zip(prompts, candidate_codes, uncertainty_scores):
            for oracle_name, oracle in self.oracles.items():
                if oracle.should_run(prompt, code, uncertainty):
                    report = oracle.evaluate(prompt, code)
                    results[oracle_name].append(report)
                else:
                    # Create empty report for oracles that didn't run
                    results[oracle_name].append(None)
                    
        return results
    
    def calculate_losses(
        self,
        prompts: List[str],
        candidate_codes: List[str],
        hidden_states: torch.Tensor,
        oracle_reports: Dict[str, List[OracleReport]]
    ) -> Dict[str, torch.Tensor]:
        """Calculate losses for all oracle reports."""
        losses = {}
        
        for oracle_name, reports in oracle_reports.items():
            if oracle_name not in self.oracles:
                continue
                
            oracle = self.oracles[oracle_name]
            oracle_losses = []
            
            for i, (prompt, code, report) in enumerate(zip(prompts, candidate_codes, reports)):
                if report is not None:
                    loss = oracle.calculate_loss(
                        prompt, code, hidden_states[i], report
                    )
                    oracle_losses.append(loss)
            
            if oracle_losses:
                losses[oracle_name] = torch.stack(oracle_losses).mean()
                
        return losses


class MetaGatingNetwork(nn.Module):
    """Meta-gating network to learn oracle weights dynamically."""
    
    def __init__(self, input_dim: int, num_oracles: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_oracles),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self, 
        prompt_embeddings: torch.Tensor,
        code_embeddings: torch.Tensor,
        oracle_scores: torch.Tensor,
        uncertainty_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute oracle weights based on context.
        
        Args:
            prompt_embeddings: [batch_size, embed_dim]
            code_embeddings: [batch_size, embed_dim]
            oracle_scores: [batch_size, num_oracles]
            uncertainty_scores: [batch_size, 1]
            
        Returns:
            weights: [batch_size, num_oracles]
        """
        # Concatenate all features
        features = torch.cat([
            prompt_embeddings,
            code_embeddings,
            oracle_scores,
            uncertainty_scores
        ], dim=-1)
        
        return self.network(features)


def aggregate_oracle_losses(
    losses: Dict[str, torch.Tensor],
    weights: Optional[torch.Tensor] = None,
    oracle_names: Optional[List[str]] = None
) -> torch.Tensor:
    """
    Aggregate oracle losses with optional weighting.
    
    Args:
        losses: Dictionary mapping oracle names to loss tensors
        weights: Optional weights for each oracle [num_oracles]
        oracle_names: Order of oracles for weight application
        
    Returns:
        aggregated_loss: Weighted sum of oracle losses
    """
    if not losses:
        return torch.tensor(0.0, requires_grad=True)
    
    if weights is None:
        # Equal weighting by default
        return sum(losses.values()) / len(losses)
    
    if oracle_names is None:
        oracle_names = list(losses.keys())
    
    weighted_loss = torch.tensor(0.0, requires_grad=True, device=weights.device)
    for i, oracle_name in enumerate(oracle_names):
        if oracle_name in losses:
            weighted_loss = weighted_loss + weights[i] * losses[oracle_name]
            
    return weighted_loss