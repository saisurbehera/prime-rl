"""Integration tests for advanced Tri-Oracle components."""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from zeroband.training.oracle_integration import create_oracle_integration
from zeroband.training.proof_oracle import ProofOracle, PropertyExtractor
from zeroband.training.reflective_oracle import ReflectiveOracle, CritiqueGenerator
from zeroband.training.memory_bank import MemoryBank, MemoryEntry
from zeroband.training.adversarial_curriculum import AdversarialCurriculumAgent, TaskMutator
from zeroband.training.mcts_refinement import MCTSCodeRefiner, CodeEditor


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestProofOracle:
    """Test the Proof Oracle functionality."""
    
    def test_property_extraction(self):
        """Test property extraction from code."""
        extractor = PropertyExtractor()
        
        code = '''def factorial(n):
        """
        Calculate factorial of n.
        Precondition: n >= 0
        Postcondition: returns n!
        """
        assert n >= 0, "n must be non-negative"
        if n == 0:
            return 1
        return n * factorial(n - 1)'''
        
        prompt = "Write a function to calculate factorial"
        
        properties = extractor.extract_properties(code, prompt)
        
        # Check that properties were extracted
        assert len(properties) > 0
        
        # Check for specific property types
        property_types = [p.property_type for p in properties]
        assert "precondition" in property_types
        assert "postcondition" in property_types
        assert "invariant" in property_types  # From assert
    
    def test_proof_oracle_evaluation(self):
        """Test Proof Oracle evaluation."""
        oracle = ProofOracle({"device": "cpu"})
        
        code = '''def is_prime(n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True'''
        
        prompt = "Write a function to check if a number is prime"
        
        report = oracle.evaluate(prompt, code)
        
        assert report.oracle_name == "proof"
        assert 0 <= report.score <= 1
        assert "verification_results" in report.details
        assert report.details["total_obligations"] > 0


class TestReflectiveOracle:
    """Test the Reflective Oracle functionality."""
    
    def test_critique_generation(self):
        """Test critique generation."""
        generator = CritiqueGenerator()
        
        code = '''def f(x):
        y = []
        for i in range(len(x)):
            if x[i] == True:
                y.append(x[i])
        return y'''
        
        critiques = generator.generate_critiques(code, {})
        
        assert len(critiques) > 0
        
        # Check for specific critiques
        critique_descriptions = [c.description for c in critiques]
        
        # Should catch poor variable names
        assert any("not descriptive" in desc for desc in critique_descriptions)
        
        # Should catch redundant boolean comparison
        assert any("boolean comparison" in desc for desc in critique_descriptions)
        
        # Should catch range(len()) pattern
        assert any("enumerate" in desc or "range(len())" in desc for desc in critique_descriptions)
    
    def test_reflective_oracle_evaluation(self):
        """Test Reflective Oracle evaluation."""
        oracle = ReflectiveOracle({"device": "cpu"})
        
        code = '''def calculate_sum(numbers):
        total = 0
        for n in numbers:
            total = total + n
        return total'''
        
        prompt = "Write a function to calculate the sum of numbers"
        
        report = oracle.evaluate(prompt, code)
        
        assert report.oracle_name == "reflective"
        assert 0 <= report.score <= 1
        assert "critiques" in report.details
        assert "suggestions" in report.details


class TestMemoryBank:
    """Test Memory Bank functionality."""
    
    def test_memory_bank_operations(self, temp_dir):
        """Test basic memory bank operations."""
        with MemoryBank(storage_path=temp_dir, embedding_dim=128, max_entries=10) as mb:
            # Add entries
            for i in range(5):
                entry = MemoryEntry(
                    id=f"test_{i}",
                    timestamp=f"2024-01-01T00:00:{i:02d}",
                    prompt=f"Test prompt {i}",
                    generated_code=f"def test_{i}(): pass",
                    oracle_reports={"execution": {"score": 0.5 + i * 0.1}},
                    joint_loss=0.5 + i * 0.1,
                    uncertainty_score=0.3 + i * 0.1
                )
                mb.add_entry(entry)
            
            # Test retrieval
            entry = mb.get_entry("test_2")
            assert entry is not None
            assert entry.prompt == "Test prompt 2"
            
            # Test statistics
            stats = mb.get_statistics()
            assert stats["total_entries"] == 5
            
            # Test failure cases
            failures = mb.get_failure_cases(limit=2, loss_threshold=0.7)
            assert len(failures) == 2
            assert all(f.joint_loss >= 0.7 for f in failures)
    
    def test_memory_bank_similarity_search(self, temp_dir):
        """Test similarity search in memory bank."""
        with MemoryBank(storage_path=temp_dir, embedding_dim=128) as mb:
            # Add entries with embeddings
            for i in range(3):
                embedding = torch.randn(128)
                entry = MemoryEntry(
                    id=f"test_{i}",
                    timestamp=f"2024-01-01T00:00:{i:02d}",
                    prompt=f"Test prompt {i}",
                    generated_code=f"def test_{i}(): pass",
                    oracle_reports={},
                    joint_loss=0.5,
                    embeddings={"code_embedding": embedding}
                )
                mb.add_entry(entry)
            
            # Search for similar
            query_embedding = torch.randn(128)
            results = mb.search_similar(query_embedding, k=2)
            
            assert len(results) <= 2
            assert all(hasattr(r, 'id') for r in results)


class TestAdversarialCurriculum:
    """Test Adversarial Curriculum Agent."""
    
    def test_task_mutation(self):
        """Test task mutation strategies."""
        mutator = TaskMutator()
        
        original_prompt = "Write a function to sort a list."
        oracle_reports = {
            "execution": {"score": 0.9},
            "complexity": {"score": 0.95}
        }
        
        # Test specific mutation
        mutated, mutations = mutator.mutate_task(
            original_prompt, 
            oracle_reports,
            mutation_type="add_constraint"
        )
        
        assert mutated != original_prompt
        assert len(mutations) > 0
        assert "constraint" in mutations[0]
    
    def test_curriculum_generation(self, temp_dir):
        """Test curriculum task generation."""
        mb = MemoryBank(storage_path=temp_dir, embedding_dim=128)
        agent = AdversarialCurriculumAgent(mb, hidden_dim=128, device="cpu")
        
        # Generate curriculum without historical data
        tasks = agent.generate_curriculum_batch(batch_size=3)
        
        assert len(tasks) <= 3
        for task in tasks:
            assert hasattr(task, 'prompt')
            assert hasattr(task, 'difficulty')
            assert 0 <= task.difficulty <= 1


class TestMCTSRefinement:
    """Test MCTS code refinement."""
    
    def test_code_editor(self):
        """Test code edit generation."""
        editor = CodeEditor()
        
        code = '''def add(a, b):
        return a + b'''
        
        edits = editor.generate_edits(code)
        
        assert len(edits) > 0
        
        # Test applying an edit
        if edits:
            edit = edits[0]
            new_code = editor.apply_edit(code, edit)
            assert new_code != code or edit.original == edit.replacement
    
    def test_mcts_refinement(self):
        """Test MCTS code refinement process."""
        # Create a simple oracle manager
        oracle_integration = create_oracle_integration(
            model_hidden_dim=128,
            use_execution_oracle=True,
            use_complexity_oracle=True,
            device="cpu"
        )
        
        refiner = MCTSCodeRefiner(
            oracle_integration.oracle_manager,
            max_iterations=5,
            max_depth=2
        )
        
        prompt = "Write a function to add two numbers"
        initial_code = '''def add(x, y):
    return x + y'''
        
        refined_code, info = refiner.refine_code(
            prompt,
            initial_code,
            target_improvements=["complexity"]
        )
        
        assert "iterations" in info
        assert "best_score" in info
        assert "initial_score" in info
        assert info["iterations"] <= 5


class TestFullIntegration:
    """Test full integration of all components."""
    
    def test_complete_oracle_system(self):
        """Test all oracles working together."""
        oracle_integration = create_oracle_integration(
            model_hidden_dim=128,
            use_execution_oracle=True,
            use_static_oracle=True,
            use_complexity_oracle=True,
            use_documentation_oracle=True,
            use_proof_oracle=True,
            use_reflective_oracle=True,
            use_meta_gating=True,
            device="cpu"
        )
        
        # Check all oracles are registered
        assert len(oracle_integration.oracle_manager.oracles) == 6
        
        prompt = "Write a function to calculate factorial"
        code = '''def factorial(n):
    """Calculate factorial of n."""
    if n == 0:
        return 1
    return n * factorial(n - 1)'''
        
        # Test oracle evaluation
        feedback = oracle_integration.get_oracle_feedback_for_inference([prompt], [code])
        
        assert len(feedback) == 1
        assert "oracle_scores" in feedback[0]
        
        # Check all oracles provided scores
        oracle_scores = feedback[0]["oracle_scores"]
        expected_oracles = ["execution", "static_analysis", "complexity", 
                          "documentation", "proof", "reflective"]
        
        for oracle in expected_oracles:
            assert oracle in oracle_scores
            assert "score" in oracle_scores[oracle]
            assert 0 <= oracle_scores[oracle]["score"] <= 1
    
    def test_loss_computation_with_all_oracles(self):
        """Test loss computation with all oracles active."""
        oracle_integration = create_oracle_integration(
            model_hidden_dim=128,
            use_execution_oracle=True,
            use_static_oracle=True,
            use_complexity_oracle=True,
            use_documentation_oracle=True,
            use_proof_oracle=True,
            use_reflective_oracle=True,
            use_meta_gating=True,
            device="cpu"
        )
        
        prompts = ["Write a factorial function", "Write a prime checker"]
        codes = [
            "def factorial(n): return 1 if n==0 else n*factorial(n-1)",
            "def is_prime(n): return n > 1 and all(n%i for i in range(2,int(n**0.5)+1))"
        ]
        
        hidden_states = torch.randn(2, 10, 128, requires_grad=True)
        
        total_loss, metrics = oracle_integration.compute_oracle_losses(
            prompts, codes, hidden_states
        )
        
        assert total_loss is not None
        assert total_loss.requires_grad
        
        # Check metrics include all oracles
        for oracle in ["execution", "proof", "reflective"]:
            assert f"{oracle}_score" in metrics or f"{oracle}_loss" in metrics
        
        # Test backward pass
        total_loss.backward()
        assert hidden_states.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])