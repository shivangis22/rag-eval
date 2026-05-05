"""Command-line interface for RAG Eval Pro."""

import click
import json
import sys
from pathlib import Path
from typing import Dict, List
import logging

from rag_eval.dataset.loader import load_dataset
from rag_eval.pipeline.runner import RAGEvaluator
from rag_eval.metrics.retrieval import RecallAtK, PrecisionAtK, NDCG, MRR
from rag_eval.metrics.llm_metrics import (
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    CoherenceMetric
)

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """RAG Eval Pro - Production-grade RAG evaluation framework."""
    pass


@cli.command()
@click.option(
    "--dataset",
    required=True,
    type=click.Path(exists=True),
    help="Path to evaluation dataset (JSON or CSV)"
)
@click.option(
    "--metrics",
    default="recall@5,precision@5",
    help="Comma-separated list of metrics to evaluate"
)
@click.option(
    "--threshold",
    multiple=True,
    help="Metric thresholds in format metric=value (e.g., faithfulness=0.85)"
)
@click.option(
    "--output",
    default="evaluation_results.json",
    type=click.Path(),
    help="Output file path for results"
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging"
)
@click.option(
    "--fail-on-threshold",
    is_flag=True,
    help="Exit with error code if thresholds not met"
)
def run(
    dataset: str,
    metrics: str,
    threshold: tuple,
    output: str,
    config: str,
    verbose: bool,
    fail_on_threshold: bool
):
    """Run RAG evaluation on a dataset.
    
    Example:
        rag-eval run --dataset data.json --metrics faithfulness,recall@5 --threshold faithfulness=0.85

    Note:
        LLM metrics such as faithfulness/relevance/coherence require
        `pip install rag-eval-pro[llm]` plus provider credentials.
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    click.echo("🎯 RAG Eval Pro - Starting Evaluation")
    click.echo(f"📁 Dataset: {dataset}")
    
    try:
        # Load dataset
        click.echo("📊 Loading dataset...")
        eval_dataset = load_dataset(dataset)
        click.echo(f"✓ Loaded {len(eval_dataset)} samples")
        
        # Parse metrics
        click.echo(f"📏 Configuring metrics: {metrics}")
        metric_list = _parse_metrics(metrics)
        click.echo(f"✓ Initialized {len(metric_list)} metrics")
        
        # Parse thresholds
        thresholds = _parse_thresholds(threshold)
        if thresholds:
            click.echo(f"🎯 Thresholds: {thresholds}")
        
        # Create evaluator
        evaluator = RAGEvaluator(metrics=metric_list, verbose=verbose)
        
        # Run evaluation
        click.echo("\n🚀 Running evaluation...")
        if thresholds:
            results, passed = evaluator.evaluate_with_threshold(eval_dataset, thresholds)
        else:
            results = evaluator.evaluate(eval_dataset)
            passed = True
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("📊 EVALUATION RESULTS")
        click.echo("="*60)
        
        for metric_name, stats in results.aggregated_metrics.items():
            click.echo(f"\n{metric_name}:")
            for stat_name, value in stats.items():
                click.echo(f"  {stat_name}: {value:.4f}")
        
        # Check thresholds
        if thresholds:
            click.echo("\n" + "="*60)
            click.echo("🎯 THRESHOLD CHECK")
            click.echo("="*60)
            
            for metric_name, threshold_value in thresholds.items():
                if metric_name in results.aggregated_metrics:
                    mean_score = results.aggregated_metrics[metric_name].get("mean", 0.0)
                    status = "✓ PASS" if mean_score >= threshold_value else "✗ FAIL"
                    click.echo(
                        f"{metric_name}: {mean_score:.4f} "
                        f"(threshold: {threshold_value:.4f}) {status}"
                    )
        
        # Save results
        click.echo(f"\n💾 Saving results to {output}...")
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        click.echo(f"✓ Results saved to {output}")
        
        # Exit with appropriate code
        if fail_on_threshold and not passed:
            click.echo("\n❌ Evaluation failed: Thresholds not met", err=True)
            sys.exit(1)
        else:
            click.echo("\n✅ Evaluation complete!")
            sys.exit(0)
            
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("dataset", type=click.Path(exists=True))
def validate(dataset: str):
    """Validate a dataset file.
    
    Example:
        rag-eval validate data.json
    """
    click.echo(f"🔍 Validating dataset: {dataset}")
    
    try:
        from rag_eval.dataset.validator import validate_dataset, get_dataset_statistics
        
        # Load dataset
        eval_dataset = load_dataset(dataset)
        click.echo(f"✓ Loaded {len(eval_dataset)} samples")
        
        # Validate
        report = validate_dataset(eval_dataset)
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("VALIDATION REPORT")
        click.echo("="*60)
        
        if report["valid"]:
            click.echo("✅ Dataset is valid!")
        else:
            click.echo("❌ Dataset has errors:")
            for error in report["errors"]:
                click.echo(f"  - {error}")
        
        if report["warnings"]:
            click.echo("\n⚠️  Warnings:")
            for warning in report["warnings"]:
                click.echo(f"  - {warning}")
        
        # Display statistics
        stats = get_dataset_statistics(eval_dataset)
        click.echo("\n" + "="*60)
        click.echo("DATASET STATISTICS")
        click.echo("="*60)
        click.echo(f"Total samples: {stats['total_samples']}")
        click.echo(f"Samples with ground truth: {stats['samples_with_ground_truth']}")
        click.echo(f"Samples with contexts: {stats['samples_with_contexts']}")
        click.echo(f"Average query length: {stats['avg_query_length']:.1f}")
        click.echo(f"Average answer length: {stats['avg_answer_length']:.1f}")
        click.echo(f"Average contexts per sample: {stats['avg_contexts_per_sample']:.1f}")
        
        if stats["categories"]:
            click.echo("\nCategories:")
            for category, count in stats["categories"].items():
                click.echo(f"  {category}: {count}")
        
        sys.exit(0 if report["valid"] else 1)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_metrics():
    """List all available metrics."""
    click.echo("📏 Available Metrics:\n")
    
    metrics_info = [
        ("Retrieval Metrics", [
            ("recall@k", "Recall at K - fraction of relevant docs retrieved"),
            ("precision@k", "Precision at K - fraction of retrieved docs that are relevant"),
            ("ndcg", "Normalized Discounted Cumulative Gain"),
            ("mrr", "Mean Reciprocal Rank"),
        ]),
        ("Generation Metrics", [
            ("faithfulness", "Answer faithfulness to retrieved contexts"),
            ("relevance", "Answer relevance to query"),
            ("coherence", "Answer coherence and readability"),
            ("completeness", "Answer completeness"),
        ]),
    ]
    
    for category, metrics in metrics_info:
        click.echo(f"{category}:")
        for name, description in metrics:
            click.echo(f"  • {name:20} - {description}")
        click.echo()


def _parse_metrics(metrics_str: str) -> List:
    """Parse metrics string into metric objects."""
    metric_map = {
        "recall@5": lambda: RecallAtK(k=5),
        "recall@10": lambda: RecallAtK(k=10),
        "precision@5": lambda: PrecisionAtK(k=5),
        "precision@10": lambda: PrecisionAtK(k=10),
        "ndcg": lambda: NDCG(k=10),
        "mrr": lambda: MRR(),
        "faithfulness": lambda: FaithfulnessMetric(),
        "relevance": lambda: AnswerRelevanceMetric(),
        "coherence": lambda: CoherenceMetric(),
    }
    
    metric_list = []
    for metric_name in metrics_str.split(","):
        metric_name = metric_name.strip().lower()
        if metric_name in metric_map:
            metric_list.append(metric_map[metric_name]())
        else:
            logger.warning(f"Unknown metric: {metric_name}")
    
    return metric_list


def _parse_thresholds(threshold_tuple: tuple) -> Dict[str, float]:
    """Parse threshold arguments into dictionary."""
    thresholds = {}
    for threshold_str in threshold_tuple:
        try:
            metric_name, value = threshold_str.split("=")
            thresholds[metric_name.strip()] = float(value.strip())
        except ValueError:
            logger.warning(f"Invalid threshold format: {threshold_str}")
    
    return thresholds


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
