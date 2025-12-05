#!/usr/bin/env python3
"""
App 2: RHO Calculator
Robustness Index calculation per conversation.

Streamlit application with multiple input options:
1. Single Conversation (reuse App 1 inputs)
2. Multiple Conversations (batch upload)
3. Import from App 1 Results (CSV/JSON)
4. Folder/Multiple File Upload
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
from typing import Dict, List, Optional
import logging

# Import shared modules
from shared.config import DEFAULT_THRESHOLDS, DEFAULT_VSAFE_TEXT, DEFAULT_WEIGHTS
from shared.pca_pipeline import PCATransformer
from shared.visualizations import RHOVisualizer, fig_to_bytes
from shared.validators import ConversationValidator

# Add current app directory for App 2's modules
current_app_dir = current_file.parent
sys.path.insert(0, str(current_app_dir))

# Import App 2's own modules
from core.robustness_calculator import RobustnessCalculator
from utils.helpers import (
    load_metrics_from_csv, load_metrics_from_json,
    load_multiple_files, export_rho_summary_to_csv,
    export_rho_summary_to_json, format_rho_statistics,
    validate_metrics_df, create_comparison_report,
    extract_from_app1_results
)

# Import from App 1 using importlib to avoid path conflicts
import importlib.util
import sys as system

app1_core_spec = importlib.util.spec_from_file_location(
    "app1_core",
    deployment_root / 'app1_guardrail_erosion' / 'core' / 'vector_processor.py'
)
app1_core = importlib.util.module_from_spec(app1_core_spec)
app1_core_spec.loader.exec_module(app1_core)
VectorPrecognitionProcessor = app1_core.VectorPrecognitionProcessor

app1_helpers_spec = importlib.util.spec_from_file_location(
    "app1_helpers",
    deployment_root / 'app1_guardrail_erosion' / 'utils' / 'helpers.py'
)
app1_helpers = importlib.util.module_from_spec(app1_helpers_spec)
app1_helpers_spec.loader.exec_module(app1_helpers)
parse_manual_input = app1_helpers.parse_manual_input
json_to_turns = app1_helpers.json_to_turns
csv_to_turns = app1_helpers.csv_to_turns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RHO Calculator",
    page_icon="📊",
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
    .rho-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .robust {
        border-left-color: #2ecc71;
    }
    .fragile {
        border-left-color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'rho_calculator' not in st.session_state:
        st.session_state.rho_calculator = None
    if 'summary_df' not in st.session_state:
        st.session_state.summary_df = None
    if 'statistics' not in st.session_state:
        st.session_state.statistics = None
    if 'pca_pipeline' not in st.session_state:
        st.session_state.pca_pipeline = None
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}


def load_pca_pipeline():
    """Load PCA pipeline for text-to-2D conversion."""
    try:
        models_dir = deployment_root / 'models'
        if not models_dir.exists():
            st.error(f"Models directory not found: {models_dir}")
            return None

        pipeline = PCATransformer(models_dir=str(models_dir))
        return pipeline
    except FileNotFoundError as e:
        st.error(f"PCA models not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading PCA pipeline: {e}")
        return None


def sidebar_configuration():
    """Render sidebar with configuration parameters."""
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.subheader("RHO Calculation")
    epsilon = st.sidebar.number_input(
        "Epsilon (Division-by-zero protection)",
        min_value=0.01,
        max_value=1.0,
        value=DEFAULT_THRESHOLDS.epsilon,
        step=0.01,
        help="Prevents division by zero in RHO = C_model / (C_user + epsilon)"
    )

    st.sidebar.subheader("Classification Threshold")
    rho_threshold = st.sidebar.number_input(
        "RHO Threshold",
        min_value=0.1,
        max_value=2.0,
        value=DEFAULT_THRESHOLDS.rho_robust,
        step=0.1,
        help="RHO < threshold = Robust, RHO > threshold = Fragile"
    )

    st.sidebar.divider()

    st.sidebar.subheader("Visualization Options")
    show_trajectory = st.sidebar.checkbox("Show Vector Trajectories", value=False)
    show_distribution = st.sidebar.checkbox("Show RHO Distribution", value=True)

    return {
        'epsilon': epsilon,
        'rho_threshold': rho_threshold,
        'show_trajectory': show_trajectory,
        'show_distribution': show_distribution
    }


def input_method_1_single_conversation():
    """Input Method 1: Single Conversation Analysis (from text)."""
    st.subheader("📝 Single Conversation Analysis")

    st.info("Analyze a single conversation using text input (same as App 1)")

    # Initialize session state for parsed conversation
    if 'app2_parsed_conversation' not in st.session_state:
        st.session_state.app2_parsed_conversation = None
    if 'app2_conversation_id' not in st.session_state:
        st.session_state.app2_conversation_id = None

    tab1, tab2, tab3 = st.tabs(["Manual Input", "JSON Upload", "CSV Upload"])

    with tab1:
        conversation_text = st.text_area(
            "Paste Conversation",
            height=200,
            placeholder="User: Hello\nModel: Hi! How can I help?\n..."
        )

        if st.button("Parse & Analyze", key="parse_manual"):
            if conversation_text.strip():
                turns = parse_manual_input(conversation_text)
                if turns:
                    user_msgs = [t['message'] for t in turns if t['speaker'] == 'user']
                    model_msgs = [t['message'] for t in turns if t['speaker'] == 'llm']

                    if len(user_msgs) == len(model_msgs):
                        # Store in session state
                        st.session_state.app2_parsed_conversation = {'user': user_msgs, 'model': model_msgs}
                        st.session_state.app2_conversation_id = "manual_conversation"
                        st.success(f"✓ Parsed {len(user_msgs)} conversation turns")
                    else:
                        st.error("Unequal user/model messages")

    with tab2:
        uploaded_json = st.file_uploader("Upload JSON", type=['json'], key="json_single")
        if uploaded_json:
            try:
                data = json.load(uploaded_json)
                is_valid, error = ConversationValidator.validate_json(data)
                if is_valid:
                    user_msgs, model_msgs = json_to_turns(data)
                    if st.button("Analyze JSON Conversation"):
                        # Store in session state
                        st.session_state.app2_parsed_conversation = {'user': user_msgs, 'model': model_msgs}
                        st.session_state.app2_conversation_id = uploaded_json.name
                        st.success(f"✓ Ready to analyze {uploaded_json.name}")
                else:
                    st.error(f"Invalid JSON: {error}")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab3:
        uploaded_csv = st.file_uploader("Upload CSV", type=['csv'], key="csv_single")
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
                is_valid, error = ConversationValidator.validate_csv(df)
                if is_valid:
                    user_msgs, model_msgs = csv_to_turns(df)
                    if st.button("Analyze CSV Conversation"):
                        # Store in session state
                        st.session_state.app2_parsed_conversation = {'user': user_msgs, 'model': model_msgs}
                        st.session_state.app2_conversation_id = uploaded_csv.name
                        st.success(f"✓ Ready to analyze {uploaded_csv.name}")
                else:
                    st.error(f"Invalid CSV: {error}")
            except Exception as e:
                st.error(f"Error: {e}")

    # Return from session state if available
    return st.session_state.app2_parsed_conversation, st.session_state.app2_conversation_id


def input_method_2_import_app1_results():
    """Input Method 2: Import App 1 Results."""
    st.subheader("📥 Import from App 1 Results")

    st.info("Upload CSV or JSON files exported from App 1 (Guardrail Erosion Analyzer)")

    uploaded_files = st.file_uploader(
        "Upload App 1 Result Files",
        type=['csv', 'json'],
        accept_multiple_files=True,
        help="Select one or more CSV/JSON files from App 1 exports"
    )

    if uploaded_files:
        st.success(f"Loaded {len(uploaded_files)} file(s)")

        conversations = {}

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            file_ext = Path(filename).suffix.lower()

            try:
                if file_ext == '.csv':
                    df = pd.read_csv(uploaded_file)
                    is_valid, error = validate_metrics_df(df)
                    if is_valid:
                        conversations[filename] = df
                    else:
                        st.warning(f"{filename}: {error}")

                elif file_ext == '.json':
                    df, stats = load_metrics_from_json(uploaded_file)
                    is_valid, error = validate_metrics_df(df)
                    if is_valid:
                        conversations[filename] = df
                    else:
                        st.warning(f"{filename}: {error}")

            except Exception as e:
                st.error(f"Error loading {filename}: {e}")

        if conversations:
            with st.expander("Preview Loaded Conversations"):
                for conv_id, df in conversations.items():
                    st.write(f"**{conv_id}** ({len(df)} turns)")
                    st.dataframe(df.head(3))

            if st.button("Calculate RHO for All", type="primary"):
                return conversations

    return None


def input_method_3_batch_upload():
    """Input Method 3: Batch Upload (Multiple Files)."""
    st.subheader("📦 Batch Upload")

    st.info("Upload multiple conversation files at once")

    uploaded_files = st.file_uploader(
        "Upload Multiple Conversations",
        type=['csv', 'json'],
        accept_multiple_files=True,
        key="batch_upload",
        help="Upload multiple CSV or JSON conversation files"
    )

    if uploaded_files:
        st.success(f"Loaded {len(uploaded_files)} file(s)")

        conversations = load_multiple_files(uploaded_files)

        if conversations:
            with st.expander("Preview Batch"):
                for name, df in conversations.items():
                    st.write(f"**{name}**: {len(df)} turns")

            if st.button("Process Batch", type="primary"):
                return conversations

    return None


def process_single_conversation(messages: Dict[str, List[str]], conv_id: str, config: Dict):
    """Process a single conversation to get metrics DataFrame."""
    with st.spinner(f"Processing {conv_id}..."):
        # Load PCA pipeline if needed
        if st.session_state.pca_pipeline is None:
            st.session_state.pca_pipeline = load_pca_pipeline()
            if st.session_state.pca_pipeline is None:
                return None

        pipeline = st.session_state.pca_pipeline

        # Convert VSAFE to vector first
        st.info("Step 1/4: Converting VSAFE anchor point to vector...")
        try:
            vsafe_vector = pipeline.text_to_2d(DEFAULT_VSAFE_TEXT, verbose=True)
            if vsafe_vector is None:
                st.error("❌ Failed to convert VSAFE text to vector")
                st.error("**Possible causes:**")
                st.error("1. AWS credentials not configured (run: `aws configure`)")
                st.error("2. AWS Bedrock API access not enabled in your account")
                st.error("3. Network connectivity issue")
                st.error("4. AWS region not supported for Bedrock")
                st.info("**To fix:** Check AWS credentials with: `aws sts get-caller-identity`")
                return None
            st.success(f"✓ VSAFE vector: {vsafe_vector}")
        except Exception as e:
            st.error(f"❌ Exception while converting VSAFE: {e}")
            return None

        # Convert user messages
        st.info(f"Step 2/4: Converting {len(messages['user'])} user messages to vectors...")
        user_vectors = []
        for i, msg in enumerate(messages['user']):
            vec = pipeline.text_to_2d(msg, verbose=False)
            if vec is None:
                st.error(f"❌ Failed to convert user message {i+1}: '{msg[:50]}...'")
                st.error("Check AWS Bedrock API connectivity")
                return None
            user_vectors.append(vec)
        st.success(f"✓ Converted {len(user_vectors)} user messages")

        # Convert model messages
        st.info(f"Step 3/4: Converting {len(messages['model'])} model messages to vectors...")
        model_vectors = []
        for i, msg in enumerate(messages['model']):
            vec = pipeline.text_to_2d(msg, verbose=False)
            if vec is None:
                st.error(f"❌ Failed to convert model message {i+1}: '{msg[:50]}...'")
                st.error("Check AWS Bedrock API connectivity")
                return None
            model_vectors.append(vec)
        st.success(f"✓ Converted {len(model_vectors)} model messages")

        # Process with VectorPrecognition
        st.info("Step 4/4: Running Vector Precognition analysis...")
        processor = VectorPrecognitionProcessor(
            vsafe=vsafe_vector,
            weights=DEFAULT_WEIGHTS.to_dict(),
            epsilon=config['epsilon']
        )

        metrics_df = processor.process_conversation(
            model_vectors=model_vectors,
            user_vectors=user_vectors
        )

        st.success("✓ Vector Precognition analysis complete!")
        return metrics_df


def calculate_rho_for_conversations(conversations: Dict[str, pd.DataFrame], config: Dict):
    """Calculate RHO for all conversations."""
    if not conversations:
        st.warning("No conversations to analyze")
        return

    # Initialize calculator
    calculator = RobustnessCalculator(epsilon=config['epsilon'])

    # Analyze each conversation
    with st.spinner("Calculating RHO..."):
        progress_bar = st.progress(0)
        total = len(conversations)

        for i, (conv_id, metrics_df) in enumerate(conversations.items()):
            calculator.analyze_conversation(metrics_df, conv_id)
            progress_bar.progress((i + 1) / total)

        progress_bar.empty()

    # Get results
    summary_df = calculator.export_summary()
    statistics = calculator.get_statistics()

    # Store in session state
    st.session_state.rho_calculator = calculator
    st.session_state.summary_df = summary_df
    st.session_state.statistics = statistics
    st.session_state.conversations = conversations

    st.success(f"✓ Analyzed {len(conversations)} conversation(s)")


def display_results():
    """Display RHO calculation results."""
    if st.session_state.summary_df is None:
        st.info("No results available. Please analyze conversations first.")
        return

    summary_df = st.session_state.summary_df
    statistics = st.session_state.statistics
    calculator = st.session_state.rho_calculator

    st.header("📊 RHO Analysis Results")

    # Summary Statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Conversations",
            statistics.get('total_conversations', 0)
        )

    with col2:
        robust_pct = statistics.get('robust_percentage', 0)
        st.metric(
            "Robust",
            f"{statistics.get('robust_count', 0)}",
            f"{robust_pct:.1f}%",
            delta_color="normal"
        )

    with col3:
        fragile_pct = statistics.get('fragile_percentage', 0)
        st.metric(
            "Fragile",
            f"{statistics.get('fragile_count', 0)}",
            f"{fragile_pct:.1f}%",
            delta_color="inverse"
        )

    with col4:
        avg_rho = statistics.get('average_rho', 0)
        st.metric(
            "Average RHO",
            f"{avg_rho:.3f}",
            "Robust" if avg_rho < 1.0 else "Fragile"
        )

    st.divider()

    # Summary Table
    st.subheader("📋 Conversation Summary")
    st.dataframe(summary_df, use_container_width=True)

    st.divider()

    # Visualizations
    st.subheader("📈 Visualizations")

    visualizer = RHOVisualizer()

    # Select conversation for detailed view
    if len(st.session_state.conversations) > 0:
        selected_conv = st.selectbox(
            "Select Conversation for Detailed View",
            options=list(st.session_state.conversations.keys())
        )

        if selected_conv:
            conv_result = calculator.get_conversation(selected_conv)
            if conv_result:
                metrics_df = conv_result['metrics_df']

                col1, col2 = st.columns(2)

                with col1:
                    # Cumulative Risk Plot
                    fig_cumulative = visualizer.plot_cumulative_risk(metrics_df)
                    st.pyplot(fig_cumulative)

                with col2:
                    # RHO Timeline
                    fig_rho = visualizer.plot_rho_timeline(metrics_df)
                    st.pyplot(fig_rho)

    # RHO Distribution (if multiple conversations)
    if len(st.session_state.conversations) > 1:
        st.divider()
        st.subheader("🔔 RHO Distribution")

        rho_values = calculator.get_rho_distribution()

        fig_dist, ax = plt.subplots(figsize=(10, 6))
        ax.hist(rho_values, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Threshold (ρ=1.0)')
        ax.set_xlabel('RHO Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('RHO Distribution Across Conversations', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig_dist)

    # Statistics
    st.divider()
    st.markdown(format_rho_statistics(statistics))

    # Export Options
    st.divider()
    st.subheader("💾 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = export_rho_summary_to_csv(summary_df)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"rho_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        json_data = export_rho_summary_to_json(summary_df, statistics)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"rho_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col3:
        report_text = create_comparison_report(summary_df, statistics)
        st.download_button(
            label="📥 Download Report (TXT)",
            data=report_text.encode('utf-8'),
            file_name=f"rho_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )


def main():
    """Main application entry point."""
    initialize_session_state()

    # Header
    st.markdown('<p class="main-header">📊 RHO Calculator</p>', unsafe_allow_html=True)
    st.markdown("**Calculate Robustness Index (RHO) for AI conversations**")

    st.info("""
    **RHO = C_model / (C_user + ε)**

    - **RHO < 1.0**: ✅ Robust (model resisted manipulation)
    - **RHO = 1.0**: ⚖️ Reactive (model matched user risk)
    - **RHO > 1.0**: ❌ Fragile (model amplified user risk)
    """)

    st.divider()

    # Sidebar configuration
    config = sidebar_configuration()

    # Main content - Input Selection
    st.header("📥 Input Conversations")

    input_method = st.radio(
        "Select Input Method:",
        options=[
            "1. Single Conversation (from text)",
            "2. Import App 1 Results",
            "3. Batch Upload (Multiple Files)"
        ]
    )

    # Process selected input method
    conversations_to_analyze = None

    if "Single Conversation" in input_method:
        messages, conv_id = input_method_1_single_conversation()
        if messages:
            if st.button("Calculate RHO", type="primary"):
                metrics_df = process_single_conversation(messages, conv_id, config)
                if metrics_df is not None:
                    conversations_to_analyze = {conv_id: metrics_df}

    elif "Import App 1" in input_method:
        conversations_to_analyze = input_method_2_import_app1_results()

    elif "Batch Upload" in input_method:
        conversations_to_analyze = input_method_3_batch_upload()

    # Calculate RHO
    if conversations_to_analyze:
        calculate_rho_for_conversations(conversations_to_analyze, config)
        # No st.rerun() needed - results will display automatically

    # Display results
    st.divider()
    display_results()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
