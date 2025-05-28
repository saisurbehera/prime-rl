#!/usr/bin/env python3
"""Basic test script for Oracle integration."""

import torch
from zeroband.training.oracle_integration import create_oracle_integration

def main():
    print("Testing Tri-Oracle integration with PrimeRL...")
    
    # Create oracle integration
    oracle_integration = create_oracle_integration(
        model_hidden_dim=768,
        use_execution_oracle=True,
        use_static_oracle=True,
        use_complexity_oracle=True,
        use_documentation_oracle=True,
        use_meta_gating=True,
        device="cpu"
    )
    
    print(f"Registered oracles: {list(oracle_integration.oracle_manager.oracles.keys())}")
    
    # Test prompts and code
    prompts = [
        "Write a Python function to calculate factorial.",
        "Write a Python function to check if a number is prime."
    ]
    
    codes = [
        """def factorial(n):
    '''Calculate factorial of n'''
    if n == 0:
        return 1
    return n * factorial(n - 1)""",
        
        """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True"""
    ]
    
    # Get oracle feedback
    print("\nRunning oracle evaluation...")
    # Note: For inference, we want to run all oracles, so we pass low uncertainty scores
    feedback = oracle_integration.get_oracle_feedback_for_inference(prompts, codes)
    
    # Display results
    for i, fb in enumerate(feedback):
        print(f"\n{'='*50}")
        print(f"Code {i+1}: {prompts[i]}")
        print(f"Oracle Scores:")
        for oracle_name, oracle_data in fb["oracle_scores"].items():
            score = oracle_data["score"]
            details = oracle_data["details"]
            print(f"  - {oracle_name}: {score:.3f}")
            
            # Show key details
            if oracle_name == "execution":
                print(f"    Tests: {details['num_tests_passed']}/{details['num_tests_run']} passed")
            elif oracle_name == "static_analysis":
                print(f"    Issues: {details['num_errors']} errors, {details['num_warnings']} warnings")
            elif oracle_name == "complexity":
                print(f"    Cyclomatic complexity: {details['cyclomatic_complexity']}")
            elif oracle_name == "documentation":
                print(f"    Doc coverage: {details['functions_with_docstrings']}/{details['total_functions']} functions")
    
    # Test loss computation
    print(f"\n{'='*50}")
    print("Testing loss computation...")
    
    hidden_states = torch.randn(len(prompts), 10, 768, requires_grad=True)
    total_loss, metrics = oracle_integration.compute_oracle_losses(
        prompts, codes, hidden_states
    )
    
    print(f"Total oracle loss: {total_loss.item():.4f}")
    print("Metrics:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")
        else:
            print(f"  - {key}: {value}")
    
    # Test backward pass
    print("\nTesting backward pass...")
    total_loss.backward()
    print(f"Gradient norm: {hidden_states.grad.norm().item():.4f}")
    
    print("\nOracle integration test completed successfully!")

if __name__ == "__main__":
    main()