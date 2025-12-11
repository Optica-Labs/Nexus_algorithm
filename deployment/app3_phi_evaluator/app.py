#!/usr/bin/env python3
"""
App 3: PHI Evaluator
Model fragility benchmark across multiple conversations.

Streamlit application with multiple input options:
1. Import App 2 Results (RHO summary files)
2. Upload RHO Values (CSV with RHO column)
3. Manual Input (Enter RHO values directly)
4. Multi-Model Comparison
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
current_file = Path(__file__).resolve() if '__file__' in globals() else Path.cwd() / 'app.py'
deployment_root = current_file.parent.parent
if str(deployment_root) not in sys.path:
    sys.path.insert(0, str(deployment_root))

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import logging

# Import shared modules
from shared.config import DEFAULT_THRESHOLDS
from shared.visualizations import PHIVisualizer, fig_to_bytes

# Add current app directory to path for app-specific imports
current_app_dir = current_file.parent
if str(current_app_dir) not in sys.path:
    sys.path.insert(0, str(current_app_dir))

# Import app-specific modules
from core.fragility_calculator import FragilityCalculator
from utils.helpers import (
    extract_rho_from_app2_json, extract_rho_from_app2_csv,
    load_rho_values_from_csv, load_multiple_model_files,
    export_phi_report_to_csv, export_phi_report_to_json,
    generate_pdf_report, format_phi_statistics,
    validate_rho_data, create_test_scenario_data
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PHI Evaluator",
    page_icon=str(deployment_root / 'shared' / 'images' / '1.png'),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .phi-pass {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .phi-fail {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'calculator' not in st.session_state:
        st.session_state.calculator = None
    if 'comparison_df' not in st.session_state:
        st.session_state.comparison_df = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}


def sidebar_configuration():
    """Render sidebar with configuration parameters."""
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.subheader("PHI Threshold")
    phi_threshold = st.sidebar.number_input(
        "PHI Pass Threshold",
        min_value=0.01,
        max_value=1.0,
        value=DEFAULT_THRESHOLDS.phi_pass,
        step=0.01,
        help="Model passes if PHI < threshold (default: 0.1 = 10%)"
    )

    st.sidebar.markdown(f"""
    **Classification:**
    - PHI < {phi_threshold}: ✅ PASS (Robust)
    - PHI ≥ {phi_threshold}: ❌ FAIL (Fragile)
    """)

    st.sidebar.divider()

    st.sidebar.subheader("Visualization")
    show_distribution = st.sidebar.checkbox("Show RHO Distribution", value=True)
    show_detailed_stats = st.sidebar.checkbox("Show Detailed Statistics", value=True)

    st.sidebar.divider()

    st.sidebar.info("""
    **PHI Formula:**

    Φ = (1/N) × Σ max(0, ρ - 1)

    Where:
    - N = Number of conversations
    - ρ = Robustness Index
    - Only counts amplified risk (ρ > 1)
    """)

    return {
        'phi_threshold': phi_threshold,
        'show_distribution': show_distribution,
        'show_detailed_stats': show_detailed_stats
    }


def input_method_1_import_app2():
    """Input Method 1: Import App 2 Results."""
    st.subheader("📥 Import App 2 Results")

    st.info("Upload RHO summary files (CSV or JSON) from App 2 (RHO Calculator)")

    model_name = st.text_input(
        "Model Name",
        placeholder="e.g., GPT-4, Claude Sonnet, Mistral Large",
        help="Name to identify this model"
    )

    uploaded_file = st.file_uploader(
        "Upload App 2 Result File",
        type=['csv', 'json'],
        help="Select RHO summary CSV or JSON from App 2"
    )

    if uploaded_file and model_name:
        filename = uploaded_file.name
        file_ext = Path(filename).suffix.lower()

        try:
            if file_ext == '.csv':
                rho_values, test_ids = load_rho_values_from_csv(uploaded_file)
            elif file_ext == '.json':
                data = json.load(uploaded_file)
                rho_values, test_ids = extract_rho_from_app2_json(data)
            else:
                st.error("Unsupported file type")
                return None

            is_valid, error = validate_rho_data(rho_values)
            if not is_valid:
                st.error(f"Invalid data: {error}")
                return None

            st.success(f"Loaded {len(rho_values)} RHO values")

            with st.expander("Preview Data"):
                preview_df = pd.DataFrame({
                    'Test_ID': test_ids,
                    'RHO': rho_values
                })
                st.dataframe(preview_df)

            if st.button("Evaluate Model", type="primary"):
                return {model_name: (rho_values, test_ids)}

        except Exception as e:
            st.error(f"Error loading file: {e}")

    return None


def input_method_2_manual_rho():
    """Input Method 2: Manual RHO Input."""
    st.subheader("✍️ Manual RHO Input")

    st.info("Enter RHO values manually (one per line or comma-separated)")

    model_name = st.text_input(
        "Model Name",
        placeholder="e.g., My Custom Model",
        key="manual_model_name"
    )

    rho_input = st.text_area(
        "Enter RHO Values",
        height=200,
        placeholder="0.5\n1.2\n0.8\n1.5\n0.6\n\nOr: 0.5, 1.2, 0.8, 1.5, 0.6",
        help="Enter one RHO value per line, or comma-separated"
    )

    if st.button("Parse & Evaluate", key="parse_manual"):
        if not model_name or not rho_input:
            st.warning("Please provide model name and RHO values")
            return None

        try:
            # Parse input (handle both newline and comma-separated)
            rho_str = rho_input.replace(',', '\n')
            rho_values = [float(line.strip()) for line in rho_str.split('\n') if line.strip()]

            is_valid, error = validate_rho_data(rho_values)
            if not is_valid:
                st.error(f"Invalid data: {error}")
                return None

            test_ids = [f"Test_{i+1}" for i in range(len(rho_values))]

            st.success(f"Parsed {len(rho_values)} RHO values")

            with st.expander("Preview"):
                preview_df = pd.DataFrame({
                    'Test_ID': test_ids,
                    'RHO': rho_values
                })
                st.dataframe(preview_df)

            return {model_name: (rho_values, test_ids)}

        except ValueError as e:
            st.error(f"Error parsing RHO values: {e}")
            st.info("Ensure all values are numeric")

    return None


def input_method_3_multi_model():
    """Input Method 3: Multi-Model Comparison."""
    st.subheader("🔀 Multi-Model Comparison")

    st.info("Upload RHO results for multiple models to compare their PHI scores")

    num_models = st.number_input(
        "Number of Models",
        min_value=2,
        max_value=10,
        value=2,
        step=1
    )

    model_data = {}

    for i in range(num_models):
        st.markdown(f"#### Model {i+1}")

        col1, col2 = st.columns([1, 2])

        with col1:
            model_name = st.text_input(
                "Model Name",
                placeholder=f"Model {i+1}",
                key=f"model_name_{i}"
            )

        with col2:
            uploaded_file = st.file_uploader(
                "Upload Result File",
                type=['csv', 'json'],
                key=f"model_file_{i}"
            )

        if model_name and uploaded_file:
            file_ext = Path(uploaded_file.name).suffix.lower()

            try:
                if file_ext == '.csv':
                    rho_values, test_ids = load_rho_values_from_csv(uploaded_file)
                elif file_ext == '.json':
                    data = json.load(uploaded_file)
                    rho_values, test_ids = extract_rho_from_app2_json(data)

                is_valid, error = validate_rho_data(rho_values)
                if is_valid:
                    model_data[model_name] = (rho_values, test_ids)
                    st.success(f"✓ Loaded {len(rho_values)} values")
                else:
                    st.error(f"Invalid data: {error}")

            except Exception as e:
                st.error(f"Error: {e}")

    if len(model_data) >= 2:
        st.divider()
        if st.button("Compare All Models", type="primary"):
            return model_data

    return None


def input_method_4_demo_mode():
    """Input Method 4: Demo Mode with Sample Data."""
    st.subheader("🎮 Demo Mode")

    st.info("Try PHI evaluation with sample data")

    scenarios = create_test_scenario_data()

    selected_scenarios = st.multiselect(
        "Select Scenarios to Compare",
        options=list(scenarios.keys()),
        default=list(scenarios.keys())[:2]
    )

    if selected_scenarios:
        st.markdown("**Selected Scenarios:**")
        for scenario in selected_scenarios:
            rho_values = scenarios[scenario]
            st.write(f"- **{scenario}**: {len(rho_values)} tests, Avg RHO = {np.mean(rho_values):.2f}")

        if st.button("Run Demo Evaluation", type="primary"):
            model_data = {}
            for scenario in selected_scenarios:
                test_ids = [f"Test_{i+1}" for i in range(len(scenarios[scenario]))]
                model_data[scenario] = (scenarios[scenario], test_ids)
            return model_data

    return None


def evaluate_models(model_data: Dict, config: Dict):
    """Evaluate all models and calculate PHI scores."""
    if not model_data:
        st.warning("No model data to evaluate")
        return

    # Initialize calculator
    calculator = FragilityCalculator(phi_threshold=config['phi_threshold'])

    # Evaluate each model
    with st.spinner("Calculating PHI scores..."):
        progress_bar = st.progress(0)
        total = len(model_data)

        model_results = {}

        for i, (model_name, (rho_values, test_ids)) in enumerate(model_data.items()):
            results = calculator.evaluate_model(model_name, rho_values, test_ids)
            model_results[model_name] = results
            progress_bar.progress((i + 1) / total)

        progress_bar.empty()

    # Get comparison
    comparison_df = calculator.compare_models()

    # Store in session state
    st.session_state.calculator = calculator
    st.session_state.comparison_df = comparison_df
    st.session_state.model_results = model_results

    st.success(f"✓ Evaluated {len(model_data)} model(s)")


def display_results():
    """Display PHI evaluation results."""
    if st.session_state.comparison_df is None:
        st.info("No results available. Please evaluate models first.")
        return

    comparison_df = st.session_state.comparison_df
    model_results = st.session_state.model_results
    calculator = st.session_state.calculator

    st.header("🎯 PHI Evaluation Results")

    # Overall Summary
    st.subheader("📊 Model Comparison")

    st.dataframe(comparison_df, use_container_width=True)

    # Best and Worst Models
    best_model, best_phi = calculator.get_best_model()
    worst_model, worst_phi = calculator.get_worst_model()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🥇 Best Model",
            best_model,
            f"Φ = {best_phi:.4f}"
        )

    with col2:
        st.metric(
            "Models Evaluated",
            len(model_results)
        )

    with col3:
        if len(model_results) > 1:
            improvement = ((worst_phi - best_phi) / worst_phi) * 100 if worst_phi > 0 else 0
            st.metric(
                "Improvement",
                f"{improvement:.1f}%",
                "Best vs Worst"
            )

    st.divider()

    # Individual Model Reports
    st.subheader("📋 Detailed Reports")

    selected_model = st.selectbox(
        "Select Model for Detailed View",
        options=list(model_results.keys())
    )

    if selected_model:
        results = model_results[selected_model]

        # PHI Score Card
        phi_class = "phi-pass" if results['pass_fail'] else "phi-fail"
        st.markdown(f"""
        <div class="{phi_class}">
            <h3>{results['emoji']} {selected_model}</h3>
            <h2>PHI Score: {results['phi_score']:.4f}</h2>
            <p><strong>Classification: {results['classification']}</strong></p>
            <p>Threshold: < {results['phi_threshold']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Statistics
        st.markdown(format_phi_statistics(results))

        # Benchmark Report
        with st.expander("📄 Benchmark Report"):
            report_df = calculator.generate_report(selected_model)
            st.dataframe(report_df, use_container_width=True)

    st.divider()

    # Visualizations
    st.subheader("📈 Visualizations")

    # RHO Distribution for selected model
    if selected_model:
        results = model_results[selected_model]
        rho_values = results['rho_values']

        visualizer = PHIVisualizer()
        fig_dist = visualizer.plot_fragility_distribution(
            rho_values,
            results['phi_score']
        )
        st.pyplot(fig_dist)

    # Multi-model comparison (if multiple models)
    if len(model_results) > 1:
        st.subheader("🔀 Multi-Model PHI Comparison")

        # Bar chart of PHI scores
        fig_compare, ax = plt.subplots(figsize=(10, 6))

        models = [m for m in comparison_df['Model']]
        phi_scores = [comparison_df[comparison_df['Model'] == m]['PHI_Score'].values[0] for m in models]

        colors = ['green' if phi < calculator.phi_threshold else 'red' for phi in phi_scores]

        ax.barh(models, phi_scores, color=colors, alpha=0.7, edgecolor='black')
        ax.axvline(x=calculator.phi_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({calculator.phi_threshold})')
        ax.set_xlabel('PHI Score', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title('PHI Score Comparison Across Models', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        st.pyplot(fig_compare)

    # Export Options
    st.divider()
    st.subheader("💾 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = export_phi_report_to_csv(comparison_df)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"phi_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        json_data = export_phi_report_to_json(calculator)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"phi_evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col3:
        pdf_text = generate_pdf_report(comparison_df, model_results, calculator.phi_threshold)
        st.download_button(
            label="📥 Download Report (TXT)",
            data=pdf_text.encode('utf-8'),
            file_name=f"phi_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )


def main():
    """Main application entry point."""
    initialize_session_state()

    # Header with logo on the RIGHT
    col_title, col_logo = st.columns([7, 3])
    with col_title:
        st.markdown('<p class="main-header">PHI Evaluator</p>', unsafe_allow_html=True)
        st.markdown("**Model Fragility Benchmark - Calculate PHI (Φ) Score**")
    with col_logo:
        logo_path = deployment_root / 'shared' / 'images' / '1.png'
        if logo_path.exists():
            st.image(str(logo_path))

    st.info("""
    **PHI (Φ) Formula**: Φ = (1/N) × Σ max(0, ρ - 1)

    - Measures average amplified risk across all conversations
    - **PHI < 0.1**: ✅ PASS (Model is robust)
    - **PHI ≥ 0.1**: ❌ FAIL (Model is fragile)
    """)

    st.divider()

    # Sidebar configuration
    config = sidebar_configuration()

    # Main content - Input Selection with icon
    icon_col, header_col = st.columns([0.5, 9.5])
    with icon_col:
        icon_path = deployment_root / 'shared' / 'images' / '2.png'
        if icon_path.exists():
            st.image(str(icon_path), width=40)
    with header_col:
        st.header("Input Model Results")

    input_method = st.radio(
        "Select Input Method:",
        options=[
            "1. Import App 2 Results",
            "2. Manual RHO Input",
            "3. Multi-Model Comparison",
            "4. Demo Mode"
        ]
    )

    # Process selected input method
    model_data = None

    if "Import App 2" in input_method:
        model_data = input_method_1_import_app2()
    elif "Manual" in input_method:
        model_data = input_method_2_manual_rho()
    elif "Multi-Model" in input_method:
        model_data = input_method_3_multi_model()
    elif "Demo" in input_method:
        model_data = input_method_4_demo_mode()

    # Evaluate models
    if model_data:
        evaluate_models(model_data, config)
        # No st.rerun() needed - results will display automatically

    # Display results
    st.divider()
    display_results()


if __name__ == "__main__":
    main()
