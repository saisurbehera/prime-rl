"""Integration test for Tri-Oracle system with PrimeRL."""

import pytest
import torch
from typing import List

from zeroband.training.oracle_integration import create_oracle_integration
from zeroband.training.execution_oracle import ExecutionOracle
from zeroband.training.static_analysis_oracle import StaticAnalysisOracle
from zeroband.training.complexity_oracle import ComplexityOracle
from zeroband.training.documentation_oracle import DocumentationOracle


@pytest.fixture
def oracle_integration():
    """Create an oracle integration instance for testing."""
    return create_oracle_integration(
        model_hidden_dim=768,
        use_execution_oracle=True,
        use_static_oracle=True,
        use_complexity_oracle=True,
        use_documentation_oracle=True,
        use_meta_gating=True,
        device="cpu"  # Use CPU for testing
    )


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "Write a Python function to calculate factorial.",
        "Write a Python function to check if a string is palindrome."
    ]


@pytest.fixture
def sample_codes():
    """Sample generated codes for testing."""
    return [
        """def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)""",
        
        """def is_palindrome(s):
    # Check if string is palindrome
    return s == s[::-1]"""
    ]


@pytest.fixture
def sample_bad_codes():
    """Sample codes with issues for testing oracle detection."""
    return [
        """def factorial(n):
    # Missing base case - will cause infinite recursion
    return n * factorial(n - 1)""",
        
        """def is_palindrome(s)
    # Syntax error - missing colon
    return s == s[::-1]"""
    ]


def test_oracle_manager_registration(oracle_integration):
    """Test that oracles are properly registered."""
    assert len(oracle_integration.oracle_manager.oracles) == 4
    assert "execution" in oracle_integration.oracle_manager.oracles
    assert "static_analysis" in oracle_integration.oracle_manager.oracles
    assert "complexity" in oracle_integration.oracle_manager.oracles
    assert "documentation" in oracle_integration.oracle_manager.oracles


def test_oracle_evaluation(oracle_integration, sample_prompts, sample_codes):
    """Test oracle evaluation on sample code."""
    # Run oracle evaluation
    feedback = oracle_integration.get_oracle_feedback_for_inference(
        sample_prompts, sample_codes
    )
    
    assert len(feedback) == 2
    
    # Check first code (factorial)
    factorial_feedback = feedback[0]
    assert "execution" in factorial_feedback["oracle_scores"]
    assert "static_analysis" in factorial_feedback["oracle_scores"]
    assert "complexity" in factorial_feedback["oracle_scores"]
    assert "documentation" in factorial_feedback["oracle_scores"]
    
    # Factorial should have relatively good scores
    assert factorial_feedback["oracle_scores"]["complexity"]["score"] > 0.5
    
    # Check second code (palindrome)
    palindrome_feedback = feedback[1]
    assert palindrome_feedback["oracle_scores"]["documentation"]["score"] < 1.0  # Has comment but no docstring


def test_oracle_loss_computation(oracle_integration, sample_prompts, sample_codes):
    """Test oracle loss computation with gradients."""
    batch_size = len(sample_prompts)
    seq_len = 10
    hidden_dim = 768
    
    # Create dummy hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    
    # Compute oracle losses
    total_loss, metrics = oracle_integration.compute_oracle_losses(
        sample_prompts,
        sample_codes,
        hidden_states
    )
    
    # Check that loss is computed
    assert total_loss is not None
    assert total_loss.requires_grad
    
    # Check metrics
    assert "execution_score" in metrics
    assert "static_analysis_score" in metrics
    assert "complexity_score" in metrics
    assert "documentation_score" in metrics
    
    # Test backward pass
    total_loss.backward()
    assert hidden_states.grad is not None


def test_oracle_error_detection(oracle_integration, sample_prompts, sample_bad_codes):
    """Test that oracles detect errors in bad code."""
    feedback = oracle_integration.get_oracle_feedback_for_inference(
        sample_prompts, sample_bad_codes
    )
    
    # First code has infinite recursion
    infinite_recursion_feedback = feedback[0]
    execution_score = infinite_recursion_feedback["oracle_scores"]["execution"]["score"]
    assert execution_score < 0.5  # Should have low execution score
    
    # Second code has syntax error
    syntax_error_feedback = feedback[1]
    static_score = syntax_error_feedback["oracle_scores"]["static_analysis"]["score"]
    assert static_score < 0.5  # Should have low static analysis score


def test_meta_gating_network(oracle_integration):
    """Test meta-gating network functionality."""
    assert oracle_integration.meta_gating_network is not None
    
    batch_size = 2
    hidden_dim = 768
    num_oracles = 4
    
    # Create dummy inputs
    prompt_embeddings = torch.randn(batch_size, hidden_dim)
    code_embeddings = torch.randn(batch_size, hidden_dim)
    oracle_scores = torch.rand(batch_size, num_oracles)
    uncertainty_scores = torch.rand(batch_size, 1)
    
    # Get oracle weights
    weights = oracle_integration.meta_gating_network(
        prompt_embeddings,
        code_embeddings,
        oracle_scores,
        uncertainty_scores
    )
    
    # Check output shape and properties
    assert weights.shape == (batch_size, num_oracles)
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)  # Sum to 1
    assert (weights >= 0).all() and (weights <= 1).all()  # Between 0 and 1


def test_execution_oracle_test_synthesis():
    """Test the test synthesis functionality of execution oracle."""
    oracle = ExecutionOracle({"device": "cpu", "timeout": 5})
    
    prompt = "Write a function to add two numbers. Example: add(2, 3) returns 5"
    code = """def add(a, b):
    return a + b"""
    
    # Synthesize tests
    test_synthesizer = oracle.test_synthesizer
    tests = test_synthesizer.synthesize_tests(prompt, code)
    
    assert len(tests) > 0
    assert any(test["name"] == "example_0" for test in tests)
    
    # Run evaluation
    report = oracle.evaluate(prompt, code)
    assert report.score == 1.0  # Should pass all tests


def test_complexity_oracle_metrics():
    """Test complexity oracle metric calculation."""
    oracle = ComplexityOracle({"device": "cpu"})
    
    simple_code = """def add(a, b):
    return a + b"""
    
    complex_code = """def complex_function(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            if data[i] % 2 == 0:
                for j in range(data[i]):
                    if j % 3 == 0:
                        result.append(j)
            else:
                try:
                    result.append(data[i] / 2)
                except:
                    pass
    return result"""
    
    # Evaluate both codes
    simple_report = oracle.evaluate("", simple_code)
    complex_report = oracle.evaluate("", complex_code)
    
    # Simple code should have better score
    assert simple_report.score > complex_report.score
    assert simple_report.details["cyclomatic_complexity"] < complex_report.details["cyclomatic_complexity"]


def test_documentation_oracle_coverage():
    """Test documentation oracle coverage detection."""
    oracle = DocumentationOracle({"device": "cpu"})
    
    documented_code = '''def factorial(n):
    """
    Calculate the factorial of a number.
    
    Args:
        n: The input number
        
    Returns:
        The factorial of n
    """
    if n == 0:
        return 1
    return n * factorial(n - 1)'''
    
    undocumented_code = """def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)"""
    
    # Evaluate both codes
    documented_report = oracle.evaluate("", documented_code)
    undocumented_report = oracle.evaluate("", undocumented_code)
    
    # Documented code should have better score
    assert documented_report.score > undocumented_report.score
    assert documented_report.details["functions_with_docstrings"] == 1
    assert undocumented_report.details["functions_with_docstrings"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])