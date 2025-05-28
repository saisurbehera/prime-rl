#!/usr/bin/env python3
"""Demo script showing advanced Tri-Oracle features: Proof, Reflective, Memory Bank, MCTS, and Adversarial Curriculum."""

import torch
import uuid
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

from zeroband.training.oracle_integration import create_oracle_integration
from zeroband.training.memory_bank import MemoryBank, MemoryEntry
from zeroband.training.adversarial_curriculum import AdversarialCurriculumAgent
from zeroband.training.mcts_refinement import MCTSCodeRefiner


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


def extract_code_from_generation(text):
    """Extract code portion from generated text."""
    if "def " in text:
        code_start = text.find("def ")
        code = text[code_start:]
        
        # Simple extraction - find the end of function
        lines = code.split('\n')
        code_lines = []
        indent_level = None
        
        for line in lines:
            if line.strip():
                if indent_level is None and line.startswith('def '):
                    code_lines.append(line)
                elif indent_level is None and code_lines:
                    indent_level = len(line) - len(line.lstrip())
                    code_lines.append(line)
                elif indent_level is not None:
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent >= indent_level:
                        code_lines.append(line)
                    else:
                        break
        
        return '\n'.join(code_lines)
    return text


def main():
    print("=== Advanced Tri-Oracle Demo ===\n")
    
    # Initialize model
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize Oracle system with ALL oracles
    print("Initializing complete Oracle system...")
    oracle_integration = create_oracle_integration(
        model_hidden_dim=model.config.hidden_size,
        use_execution_oracle=True,
        use_static_oracle=True,
        use_complexity_oracle=True,
        use_documentation_oracle=True,
        use_proof_oracle=True,  # NEW
        use_reflective_oracle=True,  # NEW
        use_meta_gating=True,
        device="cpu"
    )
    
    print(f"Active oracles: {list(oracle_integration.oracle_manager.oracles.keys())}")
    
    # Initialize Memory Bank
    print("\nInitializing Memory Bank...")
    memory_bank = MemoryBank(storage_path="demo_memory_bank", embedding_dim=model.config.hidden_size)
    
    # Initialize Adversarial Curriculum Agent
    print("Initializing Adversarial Curriculum Agent...")
    curriculum_agent = AdversarialCurriculumAgent(memory_bank, hidden_dim=256, device="cpu")
    
    # Initialize MCTS Refiner
    print("Initializing MCTS Code Refiner...")
    mcts_refiner = MCTSCodeRefiner(oracle_integration.oracle_manager, max_iterations=10)
    
    print("\n" + "="*60)
    
    # PART 1: Generate and evaluate code with all oracles
    print("\nPART 1: Complete Oracle Evaluation")
    print("-" * 40)
    
    prompt = "Write a Python function to find all prime numbers up to n. Include proper validation and documentation."
    
    print(f"Prompt: {prompt}")
    
    # Generate initial code
    generated_text = generate_code(model, tokenizer, prompt)
    initial_code = extract_code_from_generation(generated_text)
    
    print("\nInitial Generated Code:")
    print(initial_code)
    print("-" * 40)
    
    # Evaluate with all oracles
    feedback = oracle_integration.get_oracle_feedback_for_inference([prompt], [initial_code])
    
    if feedback:
        oracle_scores = feedback[0]["oracle_scores"]
        
        print("\nComplete Oracle Evaluation:")
        for oracle_name, oracle_data in oracle_scores.items():
            score = oracle_data["score"]
            details = oracle_data["details"]
            
            print(f"\n{oracle_name.upper()} Oracle (Score: {score:.2f}):")
            
            if oracle_name == "proof":
                print(f"  - Properties verified: {details['proven_count']}/{details['total_obligations']}")
                if details['verification_results']:
                    for prop, result in list(details['verification_results'].items())[:2]:
                        print(f"  - {prop}: {result['status']} (confidence: {result['confidence']:.2f})")
            
            elif oracle_name == "reflective":
                print(f"  - Critiques found: {details['num_critiques']}")
                if details['critiques']:
                    for critique in details['critiques'][:2]:
                        print(f"  - {critique['severity']}: {critique['description']}")
                if details['suggestions']:
                    print(f"  - Suggestions: {', '.join(details['suggestions'][:2])}")
    
    # Store in memory bank
    print("\nStoring interaction in Memory Bank...")
    entry = MemoryEntry(
        id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        prompt=prompt,
        generated_code=initial_code,
        oracle_reports=oracle_scores,
        joint_loss=0.5,  # Simplified
        uncertainty_score=0.6
    )
    memory_bank.add_entry(entry)
    
    print(f"Memory Bank now contains {memory_bank.get_statistics()['total_entries']} entries")
    
    print("\n" + "="*60)
    
    # PART 2: Adversarial Curriculum Generation
    print("\nPART 2: Adversarial Curriculum Generation")
    print("-" * 40)
    
    # Generate curriculum tasks
    curriculum_tasks = curriculum_agent.generate_curriculum_batch(batch_size=3)
    
    print("Generated Adversarial Tasks:")
    for i, task in enumerate(curriculum_tasks):
        print(f"\n{i+1}. Difficulty: {task.difficulty:.2f}")
        print(f"   Prompt: {task.prompt}")
        print(f"   Mutations: {', '.join(task.mutations_applied) if task.mutations_applied else 'Base task'}")
        print(f"   Expected challenges: {', '.join(task.expected_challenges[:2]) if task.expected_challenges else 'None'}")
    
    # Generate code for a challenging task
    if curriculum_tasks:
        challenging_task = curriculum_tasks[-1]  # Most difficult
        print(f"\nGenerating code for challenging task...")
        
        generated_text = generate_code(model, tokenizer, challenging_task.prompt)
        challenging_code = extract_code_from_generation(generated_text)
        
        print("Generated Code for Challenging Task:")
        print(challenging_code[:200] + "..." if len(challenging_code) > 200 else challenging_code)
        
        # Update curriculum agent with results
        challenging_feedback = oracle_integration.get_oracle_feedback_for_inference(
            [challenging_task.prompt], [challenging_code]
        )
        
        if challenging_feedback:
            curriculum_agent.update_from_results(
                challenging_task,
                challenging_feedback[0]["oracle_scores"],
                0.7  # Mock joint loss
            )
    
    print("\n" + "="*60)
    
    # PART 3: MCTS Code Refinement
    print("\nPART 3: MCTS Code Refinement")
    print("-" * 40)
    
    print("Refining initial code using MCTS...")
    print(f"Target improvements: execution, complexity")
    
    # Refine the initial code
    refined_code, refinement_info = mcts_refiner.refine_code(
        prompt=prompt,
        initial_code=initial_code,
        target_improvements=["execution", "complexity"]
    )
    
    print(f"\nRefinement Statistics:")
    print(f"  - Iterations: {refinement_info['iterations']}")
    print(f"  - Nodes explored: {refinement_info['nodes_explored']}")
    print(f"  - Initial score: {refinement_info['initial_score']:.3f}")
    print(f"  - Final score: {refinement_info['best_score']:.3f}")
    print(f"  - Improvement: {refinement_info['improvement']:.3f}")
    
    if refinement_info['edits_applied']:
        print(f"\nEdits applied:")
        for edit in refinement_info['edits_applied']:
            print(f"  - {edit['type']}: {edit['description']}")
    
    print("\nRefined Code:")
    print(refined_code[:300] + "..." if len(refined_code) > 300 else refined_code)
    
    # Compare oracle scores
    print("\nOracle Score Comparison:")
    for oracle in ["execution", "complexity"]:
        if oracle in refinement_info['oracle_scores']['initial'] and oracle in refinement_info['oracle_scores']['final']:
            initial = refinement_info['oracle_scores']['initial'][oracle]
            final = refinement_info['oracle_scores']['final'][oracle]
            print(f"  - {oracle}: {initial:.3f} -> {final:.3f} ({final-initial:+.3f})")
    
    print("\n" + "="*60)
    
    # PART 4: Memory Bank Analysis
    print("\nPART 4: Memory Bank Analysis")
    print("-" * 40)
    
    stats = memory_bank.get_statistics()
    print(f"Memory Bank Statistics:")
    print(f"  - Total entries: {stats['total_entries']}")
    print(f"  - Average joint loss: {stats['avg_joint_loss']:.3f}" if stats['avg_joint_loss'] else "  - No loss data")
    print(f"  - Loss distribution: {stats['loss_distribution']}")
    
    # Export dataset
    print("\nExporting memory bank to dataset...")
    memory_bank.export_dataset("demo_dataset.jsonl", limit=10)
    print("Dataset exported to demo_dataset.jsonl")
    
    # Close memory bank
    memory_bank.close()
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()