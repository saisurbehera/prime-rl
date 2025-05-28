#!/usr/bin/env python3
"""Demo script showing how to use oracles during inference."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from zeroband.training.oracle_integration import create_oracle_integration

def generate_code(model, tokenizer, prompt, max_length=256):
    """Generate code using the model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("Loading model and tokenizer...")
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize Oracle system
    print("Initializing Oracle system...")
    oracle_integration = create_oracle_integration(
        model_hidden_dim=model.config.hidden_size,
        use_execution_oracle=True,
        use_static_oracle=True,
        use_complexity_oracle=True,
        use_documentation_oracle=True,
        use_meta_gating=False,  # Simpler without meta-gating for demo
        device="cpu"
    )
    
    # Code generation prompts
    prompts = [
        "Write a Python function to find the greatest common divisor (GCD) of two numbers.",
        "Write a Python function to check if a string contains only unique characters.",
        "Write a Python function to reverse words in a sentence while preserving spaces."
    ]
    
    print("\nGenerating code and evaluating with oracles...\n")
    
    for i, prompt in enumerate(prompts):
        print(f"{'='*60}")
        print(f"Prompt {i+1}: {prompt}")
        print(f"{'='*60}")
        
        # Generate code
        generated_text = generate_code(model, tokenizer, prompt)
        
        # Extract code from generation (simple heuristic)
        if "def " in generated_text:
            code_start = generated_text.find("def ")
            code = generated_text[code_start:]
            # Find the end of the function (simple heuristic)
            lines = code.split('\n')
            code_lines = []
            indent_level = None
            for line in lines:
                if line.strip():
                    if indent_level is None and line.startswith('def '):
                        code_lines.append(line)
                    elif indent_level is None and code_lines:
                        # First line after def
                        indent_level = len(line) - len(line.lstrip())
                        code_lines.append(line)
                    elif indent_level is not None:
                        current_indent = len(line) - len(line.lstrip())
                        if current_indent >= indent_level or line.strip().startswith('return'):
                            code_lines.append(line)
                        else:
                            break
            code = '\n'.join(code_lines)
        else:
            code = generated_text
        
        print("\nGenerated Code:")
        print("-" * 40)
        print(code)
        print("-" * 40)
        
        # Evaluate with oracles
        feedback = oracle_integration.get_oracle_feedback_for_inference([prompt], [code])
        
        if feedback:
            oracle_scores = feedback[0]["oracle_scores"]
            
            print("\nOracle Evaluation:")
            for oracle_name, oracle_data in oracle_scores.items():
                score = oracle_data["score"]
                details = oracle_data["details"]
                
                print(f"\n{oracle_name.upper()} (Score: {score:.2f}):")
                
                if oracle_name == "execution":
                    print(f"  - Tests passed: {details['num_tests_passed']}/{details['num_tests_run']}")
                    if details['error_types']:
                        print(f"  - Error types: {details['error_types']}")
                
                elif oracle_name == "static_analysis":
                    print(f"  - Errors: {details['num_errors']}")
                    print(f"  - Warnings: {details['num_warnings']}")
                    if details['errors']:
                        for error in details['errors'][:3]:  # Show first 3 errors
                            print(f"    * {error['type']}: {error['message']}")
                
                elif oracle_name == "complexity":
                    print(f"  - Cyclomatic complexity: {details['cyclomatic_complexity']}")
                    print(f"  - Maintainability index: {details['maintainability_index']:.1f}")
                
                elif oracle_name == "documentation":
                    print(f"  - Functions with docs: {details['functions_with_docstrings']}/{details['total_functions']}")
                    print(f"  - Has module docstring: {details['has_module_docstring']}")
        
        print()

if __name__ == "__main__":
    main()