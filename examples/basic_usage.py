"""Basic usage example for RAG Eval Pro."""

from rag_eval import RAGEvaluator, load_dataset
from rag_eval.metrics import RecallAtK, PrecisionAtK, SemanticSimilarity, Faithfulness


def main():
    """Run basic RAG evaluation example."""
    
    # Load evaluation dataset
    print("Loading dataset...")
    dataset = load_dataset("datasets/sample_dataset.json")
    print(f"Loaded {len(dataset)} samples")
    
    # Initialize evaluator with metrics
    print("\nInitializing evaluator...")
    evaluator = RAGEvaluator(
        metrics=[
            RecallAtK(k=3, threshold=0.7),
            PrecisionAtK(k=3, threshold=0.7),
            SemanticSimilarity(
                backend="lexical",
                compare_to="ground_truth",
                threshold=0.75
            ),
            Faithfulness(
                provider="mock",
                model="gpt-4",
                threshold=0.8,
                mock_response="0.8"
            )
        ]
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate(dataset)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(results.summary_text())
    
    # Print detailed metrics
    print("\n" + "="*60)
    print("DETAILED METRICS")
    print("="*60)
    for metric_name, stats in results.aggregated_metrics.items():
        print(f"\n{metric_name}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.4f}")
    
    # Save results
    output_path = "evaluation_results.json"
    results.save_json(output_path)
    print(f"\nResults saved to {output_path}")
    
    # Check if evaluation passed thresholds
    print("\n" + "="*60)
    print("THRESHOLD CHECK")
    print("="*60)
    
    passed_all = True
    for metric_name, stats in results.aggregated_metrics.items():
        mean_score = stats.get("mean", 0)
        # Get threshold from metric (simplified - would need actual metric config)
        threshold = 0.7  # Default threshold
        passed = mean_score >= threshold
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{metric_name}: {mean_score:.4f} (threshold: {threshold}) {status}")
        if not passed:
            passed_all = False
    
    if passed_all:
        print("\n✓ All metrics passed thresholds!")
    else:
        print("\n✗ Some metrics failed thresholds")
    
    return results


if __name__ == "__main__":
    main()
