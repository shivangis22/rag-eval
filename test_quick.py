#!/usr/bin/env python3
"""Quick test script to verify the library works."""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from rag_eval.dataset.schema import RAGSample, RAGDataset
        from rag_eval.dataset.loader import load_dataset, save_dataset
        from rag_eval.dataset.validator import validate_dataset
        from rag_eval.metrics.retrieval import RecallAtK, PrecisionAtK
        from rag_eval.pipeline.runner import RAGEvaluator
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    try:
        from rag_eval.dataset.schema import RAGSample, RAGDataset
        from rag_eval.metrics.retrieval import RecallAtK
        from rag_eval.pipeline.runner import RAGEvaluator
        
        # Create a sample
        sample = RAGSample(
            query="What is Python?",
            retrieved_docs=["Python is a programming language.", "Python is used for AI."],
            generated_answer="Python is a high-level programming language.",
            ground_truth="Python is a programming language.",
            relevant_docs=["Python is a programming language."]
        )
        print(f"✓ Created sample: {sample.query}")
        
        # Create a dataset
        dataset = RAGDataset(
            name="test_dataset",
            samples=[sample]
        )
        print(f"✓ Created dataset with {len(dataset)} samples")
        
        # Create a metric
        metric = RecallAtK(k=2)
        print(f"✓ Created metric: {metric.name}")
        
        # Compute metric
        result = metric.compute(sample)
        print(f"✓ Computed metric score: {result.score}")
        
        # Create evaluator
        evaluator = RAGEvaluator(metrics=[metric])
        print(f"✓ Created evaluator with {len(evaluator.metrics)} metrics")
        
        # Run evaluation
        results = evaluator.evaluate(dataset)
        print(f"✓ Evaluation complete: {len(results.results)} results")
        print(f"✓ Aggregated metrics: {list(results.aggregated_metrics.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_operations():
    """Test dataset save/load."""
    print("\nTesting dataset operations...")
    try:
        from rag_eval.dataset.schema import RAGSample, RAGDataset
        from rag_eval.dataset.loader import save_dataset, load_dataset
        from rag_eval.dataset.validator import validate_dataset, get_dataset_statistics
        import tempfile
        import os
        
        # Create dataset
        samples = [
            RAGSample(
                query=f"Query {i}",
                retrieved_docs=[f"Doc {i}"],
                generated_answer=f"Answer {i}",
                ground_truth=f"Truth {i}",
                relevant_docs=[f"Doc {i}"]
            )
            for i in range(3)
        ]
        dataset = RAGDataset(name="test", samples=samples)
        print(f"✓ Created dataset with {len(dataset)} samples")
        
        # Validate
        report = validate_dataset(dataset)
        print(f"✓ Validation: {report['valid']}")
        
        # Get statistics
        stats = get_dataset_statistics(dataset)
        print(f"✓ Statistics: {stats['total_samples']} samples")
        
        # Save and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_dataset(dataset, temp_path)
            print(f"✓ Saved dataset to {temp_path}")
            
            loaded = load_dataset(temp_path)
            print(f"✓ Loaded dataset with {len(loaded)} samples")
            
            assert len(loaded) == len(dataset)
            print("✓ Dataset integrity verified")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("RAG Eval Pro - Quick Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_dataset_operations,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

