"""Streamlit dashboard for RAG evaluation results."""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_dashboard(results_path: str, port: int = 8501) -> None:
    """Create and launch Streamlit dashboard.
    
    Args:
        results_path: Path to evaluation results JSON file
        port: Port to run dashboard on
        
    Example:
        >>> create_dashboard("evaluation_results.json")
    """
    try:
        import streamlit as st
    except ImportError:
        raise ImportError(
            "streamlit required for dashboard. Install with: pip install streamlit"
        )
    
    logger.info(f"Starting dashboard on port {port}")
    logger.info(f"Loading results from: {results_path}")
    
    # Dashboard will be implemented as a separate Streamlit app
    # This is a placeholder for the dashboard launcher
    print(f"""
Dashboard Setup:

1. Install Streamlit:
   pip install streamlit plotly

2. Create dashboard app file (dashboard_app.py):
   See rag_eval/reporting/dashboard_app.py

3. Run dashboard:
   streamlit run dashboard_app.py -- --results {results_path}

4. Open browser:
   http://localhost:{port}
""")


def generate_dashboard_app() -> str:
    """Generate Streamlit dashboard app code.
    
    Returns:
        Python code for Streamlit app
    """
    return '''
"""Streamlit dashboard for RAG evaluation results."""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="RAG Eval Pro Dashboard",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 RAG Eval Pro - Evaluation Dashboard")

# Sidebar
st.sidebar.header("Configuration")
results_file = st.sidebar.file_uploader("Upload Results JSON", type=["json"])

if results_file:
    results = json.load(results_file)
    
    # Overview
    st.header("📊 Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(results.get("results", [])))
    
    with col2:
        st.metric("Metrics Evaluated", len(results.get("aggregated_metrics", {})))
    
    with col3:
        dataset_name = results.get("dataset_name", "Unknown")
        st.metric("Dataset", dataset_name)
    
    # Aggregated Metrics
    st.header("📈 Aggregated Metrics")
    
    agg_metrics = results.get("aggregated_metrics", {})
    if agg_metrics:
        metrics_df = pd.DataFrame(agg_metrics).T
        st.dataframe(metrics_df.style.highlight_max(axis=0))
        
        # Metric comparison chart
        fig = go.Figure()
        for metric_name in agg_metrics.keys():
            fig.add_trace(go.Bar(
                name=metric_name,
                x=["Mean", "Median", "P95"],
                y=[
                    agg_metrics[metric_name].get("mean", 0),
                    agg_metrics[metric_name].get("median", 0),
                    agg_metrics[metric_name].get("p95", 0)
                ]
            ))
        
        fig.update_layout(
            title="Metric Comparison",
            barmode="group",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    st.header("📊 Score Distributions")
    
    selected_metric = st.selectbox(
        "Select Metric",
        list(agg_metrics.keys()) if agg_metrics else []
    )
    
    if selected_metric:
        # Extract scores for selected metric
        scores = []
        for result in results.get("results", []):
            if selected_metric in result.get("metrics", {}):
                score = result["metrics"][selected_metric].get("score")
                if score is not None:
                    scores.append(score)
        
        if scores:
            fig = px.histogram(
                scores,
                nbins=20,
                title=f"{selected_metric} Distribution",
                labels={"value": "Score", "count": "Frequency"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot
            fig = px.box(
                y=scores,
                title=f"{selected_metric} Box Plot",
                labels={"y": "Score"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Failed samples
    st.header("❌ Failed Samples")
    
    threshold = st.slider("Failure Threshold", 0.0, 1.0, 0.7, 0.05)
    
    failed_samples = []
    for result in results.get("results", []):
        for metric_name, metric_result in result.get("metrics", {}).items():
            score = metric_result.get("score")
            if score is not None and score < threshold:
                failed_samples.append({
                    "Query": result["sample"]["query"][:100],
                    "Metric": metric_name,
                    "Score": score,
                    "Answer": result["sample"]["generated_answer"][:100]
                })
    
    if failed_samples:
        st.write(f"Found {len(failed_samples)} failures")
        st.dataframe(pd.DataFrame(failed_samples))
    else:
        st.success("No failures found!")
    
    # Sample details
    st.header("🔍 Sample Details")
    
    sample_idx = st.number_input(
        "Sample Index",
        min_value=0,
        max_value=len(results.get("results", [])) - 1,
        value=0
    )
    
    if sample_idx < len(results.get("results", [])):
        sample_result = results["results"][sample_idx]
        
        st.subheader("Query")
        st.write(sample_result["sample"]["query"])
        
        st.subheader("Generated Answer")
        st.write(sample_result["sample"]["generated_answer"])
        
        if sample_result["sample"].get("ground_truth"):
            st.subheader("Ground Truth")
            st.write(sample_result["sample"]["ground_truth"])
        
        st.subheader("Retrieved Contexts")
        for i, ctx in enumerate(sample_result["sample"].get("retrieved_docs", [])):
            with st.expander(f"Context {i+1}"):
                st.write(ctx)
        
        st.subheader("Metric Scores")
        metric_scores = {
            name: result.get("score", 0)
            for name, result in sample_result.get("metrics", {}).items()
            if result.get("score") is not None
        }
        
        if metric_scores:
            fig = px.bar(
                x=list(metric_scores.keys()),
                y=list(metric_scores.values()),
                title="Metric Scores for This Sample",
                labels={"x": "Metric", "y": "Score"}
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Upload a results JSON file to get started")
    
    st.markdown("""
    ## How to use this dashboard:
    
    1. Run your evaluation and save results to JSON
    2. Upload the JSON file using the sidebar
    3. Explore metrics, distributions, and failed samples
    4. Drill down into individual sample details
    
    ## Example:
    ```python
    from rag_eval import RAGEvaluator, load_dataset
    
    dataset = load_dataset("data.json")
    evaluator = RAGEvaluator(metrics=[...])
    results = evaluator.evaluate(dataset)
    results.save_json("results.json")
    ```
    """)
'''


# Save dashboard app to file
def save_dashboard_app(output_path: str = "dashboard_app.py") -> None:
    """Save dashboard app code to file.
    
    Args:
        output_path: Path to save dashboard app
    """
    app_code = generate_dashboard_app()
    
    with open(output_path, "w") as f:
        f.write(app_code)
    
    logger.info(f"Dashboard app saved to: {output_path}")
    print(f"Dashboard app saved to: {output_path}")
    print(f"\nTo run: streamlit run {output_path}")

