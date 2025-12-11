#!/usr/bin/env python3
"""
App 1: Guardrail Erosion Analyzer
Single conversation risk analysis with guardrail erosion metrics.

Streamlit application with 4 input options:
1. Manual Text Input
2. JSON Upload
3. CSV Import
4. API Integration
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
# Get the absolute path of this file
current_file = Path(__file__).resolve() if '__file__' in globals() else Path.cwd() / 'app.py'
deployment_root = current_file.parent.parent

# Add deployment root to path (so we can import shared modules)
if str(deployment_root) not in sys.path:
    sys.path.insert(0, str(deployment_root))

# Add current app directory to path
current_dir = current_file.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import numpy as np
import pandas as pd
import json
import requests
from typing import Dict, List, Optional, Tuple
import logging
import streamlit.components.v1 as components

# Import shared modules
from shared.config import (
    DEFAULT_WEIGHTS, DEFAULT_VSAFE_TEXT, DEFAULT_THRESHOLDS,
    VSAFE_PRESETS, LLM_ENDPOINTS
)
from shared.pca_pipeline import PCATransformer
from shared.validators import (
    ConversationValidator, ParameterValidator,
    parse_conversation_json, parse_conversation_csv
)
from shared.visualizations import GuardrailVisualizer, fig_to_bytes

# Import app-specific modules
from core.vector_processor import VectorPrecognitionProcessor
from utils.helpers import (
    parse_manual_input, json_to_turns, csv_to_turns,
    export_metrics_to_csv, export_metrics_to_json,
    create_export_filename, format_statistics_display
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Guardrail Erosion Analyzer",
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
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


def show_user_risk_alert_popup():
    """Show alert popup when user accumulated risk is too high."""
    # Initialize the session state flag if not exists
    if 'show_user_risk_popup' not in st.session_state:
        st.session_state.show_user_risk_popup = True

    # Check if button was clicked in previous run
    if st.session_state.get('user_risk_btn_clicked', False):
        st.session_state.show_user_risk_popup = False
        st.session_state.user_risk_btn_clicked = False
        # Clear all conversation data
        st.session_state.metrics_df = None
        st.session_state.statistics = None
        st.session_state.processor = None
        st.session_state.parsed_messages = None
        if 'api_conversation' in st.session_state:
            st.session_state.api_conversation = []
        return

    # Use Streamlit's dialog feature for truly blocking modal
    @st.dialog("⚠️ USER RISK ALERT", width="large")
    def user_risk_alert_dialog():
        # Custom CSS for beautiful styling
        st.markdown("""
        <style>
            /* Style the dialog */
            [data-testid="stDialog"] {
                background: rgba(0, 0, 0, 0.9) !important;
            }

            .warning-container {
                text-align: center;
                padding: 20px;
            }

            .warning-icon {
                font-size: 100px;
                animation: pulse 2s infinite;
                display: block;
                margin-bottom: 20px;
            }

            @keyframes pulse {
                0%, 100% {
                    transform: scale(1);
                }
                50% {
                    transform: scale(1.15);
                }
            }

            .alert-title {
                color: #e67e22;
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 20px;
                text-transform: uppercase;
                letter-spacing: 2px;
            }

            .alert-message {
                color: #2d3436;
                font-size: 18px;
                line-height: 1.8;
                margin-bottom: 25px;
                padding: 20px;
                background: #fff3cd;
                border-radius: 10px;
                border-left: 5px solid #e67e22;
            }
        </style>
        """, unsafe_allow_html=True)

        # Modal content
        st.markdown("""
        <div class="warning-container">
            <div class="warning-icon">🚨</div>
            <div class="alert-title">User Policy Violation Warning</div>
            <div class="alert-message">
                Your prompts indicate a possible policy or guardrail violation.<br><br>
                <strong>Please review your company's use policies before proceeding.</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "✓ I Understand",
                type="primary",
                use_container_width=True,
                key="user_risk_understand_btn"
            ):
                # Set flag that button was clicked
                st.session_state.user_risk_btn_clicked = True
                st.rerun()

    # Only show dialog if flag is True
    if st.session_state.show_user_risk_popup:
        user_risk_alert_dialog()


def show_erosion_alert_popup():
    """Show beautiful centered popup alert when erosion threshold is exceeded."""
    # Initialize the session state flag if not exists
    if 'show_erosion_popup' not in st.session_state:
        st.session_state.show_erosion_popup = True

    # Check if button was clicked in previous run
    if st.session_state.get('erosion_btn_clicked', False):
        st.session_state.show_erosion_popup = False
        st.session_state.erosion_btn_clicked = False
        # Clear all conversation data
        st.session_state.metrics_df = None
        st.session_state.statistics = None
        st.session_state.processor = None
        st.session_state.parsed_messages = None
        if 'api_conversation' in st.session_state:
            st.session_state.api_conversation = []
        return

    # Use Streamlit's dialog feature for truly blocking modal
    @st.dialog("⚠️ SAFETY ALERT", width="large")
    def erosion_alert_dialog():
        # Custom CSS for beautiful styling
        st.markdown("""
        <style>
            /* Style the dialog */
            [data-testid="stDialog"] {
                background: rgba(0, 0, 0, 0.9) !important;
            }

            .warning-container {
                text-align: center;
                padding: 20px;
            }

            .warning-icon {
                font-size: 100px;
                animation: pulse 2s infinite;
                display: block;
                margin-bottom: 20px;
            }

            @keyframes pulse {
                0%, 100% {
                    transform: scale(1);
                }
                50% {
                    transform: scale(1.15);
                }
            }

            .alert-title {
                color: #d63031;
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 20px;
                text-transform: uppercase;
                letter-spacing: 2px;
            }

            .alert-message {
                color: #2d3436;
                font-size: 18px;
                line-height: 1.8;
                margin-bottom: 25px;
                padding: 20px;
                background: #ffe5e5;
                border-radius: 10px;
                border-left: 5px solid #ff6b6b;
            }
        </style>
        """, unsafe_allow_html=True)

        # Modal content
        st.markdown("""
        <div class="warning-container">
            <div class="warning-icon">⚠️</div>
            <div class="alert-title">AI Guardrail Erosion Threshold Exceeded</div>
            <div class="alert-message">
                This model is approaching a break of its assigned guardrails.<br><br>
                <strong>Please close this context window and restart your conversation in a new instance.</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "✓ I Understand",
                type="primary",
                use_container_width=True,
                key="erosion_understand_btn"
            ):
                # Set flag that button was clicked
                st.session_state.erosion_btn_clicked = True
                st.rerun()

    # Only show dialog if flag is True
    if st.session_state.show_erosion_popup:
        erosion_alert_dialog()


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'metrics_df' not in st.session_state:
        st.session_state.metrics_df = None
    if 'statistics' not in st.session_state:
        st.session_state.statistics = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'pca_pipeline' not in st.session_state:
        st.session_state.pca_pipeline = None
    if 'parsed_messages' not in st.session_state:
        st.session_state.parsed_messages = None


def load_pca_pipeline():
    """Load PCA pipeline for text-to-2D conversion."""
    try:
        models_dir = deployment_root / 'models'
        if not models_dir.exists():
            st.error(f"Models directory not found: {models_dir}")
            st.info("Please copy PCA models to deployment/models/")
            return None

        pipeline = PCATransformer(models_dir=str(models_dir))
        return pipeline
    except FileNotFoundError as e:
        st.error(f"PCA models not found: {e}")
        st.info("Please train and copy PCA models to deployment/models/")
        return None
    except Exception as e:
        st.error(f"Error loading PCA pipeline: {e}")
        return None


def sidebar_configuration():
    """Render sidebar with configuration parameters."""
    st.sidebar.title("⚙️ Configuration")

    # Algorithm Weights
    st.sidebar.subheader("Algorithm Weights")
    wR = st.sidebar.slider(
        "wR (Risk Severity)",
        min_value=0.1, max_value=10.0,
        value=DEFAULT_WEIGHTS.wR,
        step=0.1,
        help="Weight for risk severity (position)"
    )
    wv = st.sidebar.slider(
        "wv (Risk Rate)",
        min_value=0.1, max_value=10.0,
        value=DEFAULT_WEIGHTS.wv,
        step=0.1,
        help="Weight for risk rate (velocity)"
    )
    wa = st.sidebar.slider(
        "wa (Erosion)",
        min_value=0.1, max_value=10.0,
        value=DEFAULT_WEIGHTS.wa,
        step=0.1,
        help="Weight for guardrail erosion (acceleration)"
    )
    b = st.sidebar.slider(
        "b (Bias)",
        min_value=-10.0, max_value=10.0,
        value=DEFAULT_WEIGHTS.b,
        step=0.1,
        help="Bias term for failure potential"
    )

    weights = {'wR': wR, 'wv': wv, 'wa': wa, 'b': b}

    # Validate weights
    is_valid, error_msg = ParameterValidator.validate_weights(wR, wv, wa, b)
    if not is_valid:
        st.sidebar.error(f"Invalid weights: {error_msg}")

    st.sidebar.divider()

    # VSAFE Configuration
    st.sidebar.subheader("VSAFE (Safe Harbor)")
    vsafe_option = st.sidebar.selectbox(
        "Select VSAFE Preset",
        options=list(VSAFE_PRESETS.keys()),
        format_func=lambda x: x.title()
    )
    vsafe_text = st.sidebar.text_area(
        "VSAFE Text",
        value=VSAFE_PRESETS[vsafe_option],
        height=100,
        help="The 'safe harbor' text defining safe AI behavior"
    )

    st.sidebar.divider()

    # Alert Thresholds Section
    st.sidebar.subheader("🚨 Alert Thresholds")

    st.sidebar.markdown("**Set sensitivity thresholds for alerts:**")

    # Likelihood Threshold
    likelihood_threshold = st.sidebar.slider(
        "Likelihood Alert Threshold",
        min_value=0.0, max_value=1.0,
        value=DEFAULT_THRESHOLDS.likelihood_alert,
        step=0.05,
        help="🔔 Alert when Likelihood L(N) exceeds this value. Higher = Less sensitive (0.8 recommended)"
    )

    # Erosion Threshold
    erosion_threshold = st.sidebar.slider(
        "Erosion Alert Threshold",
        min_value=0.0, max_value=1.0,
        value=DEFAULT_THRESHOLDS.acceleration_alert,
        step=0.01,
        help="⚠️ Alert when Guardrail Erosion a(N) exceeds this value. Lower = More sensitive (0.15 recommended)"
    )

    # User Risk Threshold (NEW)
    user_risk_threshold = st.sidebar.slider(
        "User Risk Alert Threshold",
        min_value=0.0, max_value=5.0,
        value=2.0,
        step=0.1,
        help="🚨 Alert when accumulated user risk exceeds this value. Monitors user's prompt safety (2.0 recommended)"
    )

    # Visual threshold indicator
    st.sidebar.markdown("---")
    st.sidebar.caption("**Current Alert Sensitivity:**")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        likelihood_sensitivity = "Low" if likelihood_threshold > 0.85 else "Medium" if likelihood_threshold > 0.7 else "High"
        st.sidebar.metric("Likelihood", likelihood_sensitivity, f"{likelihood_threshold:.2f}")
    with col2:
        erosion_sensitivity = "Low" if erosion_threshold > 0.2 else "Medium" if erosion_threshold > 0.1 else "High"
        st.sidebar.metric("Erosion", erosion_sensitivity, f"{erosion_threshold:.2f}")

    return {
        'weights': weights,
        'vsafe_text': vsafe_text,
        'likelihood_threshold': likelihood_threshold,
        'erosion_threshold': erosion_threshold,
        'user_risk_threshold': user_risk_threshold
    }


def input_method_1_manual():
    """Input Method 1: Manual Text Entry."""
    st.subheader("📝 Manual Text Input")

    st.info("""
    **Format:** Enter conversation turns line by line.
    - User: Your message here
    - Model: AI response here
    """)

    conversation_text = st.text_area(
        "Paste Conversation",
        height=300,
        placeholder="User: Hello\nModel: Hi! How can I help?\nUser: Tell me about...\nModel: Sure, I can help with that."
    )

    if st.button("Parse Conversation", type="primary"):
        if not conversation_text.strip():
            st.error("Please enter a conversation")
            return None

        turns = parse_manual_input(conversation_text)

        if not turns:
            st.error("Could not parse conversation. Please check format.")
            return None

        is_valid, error_msg = ConversationValidator.validate_manual_input(turns)
        if not is_valid:
            st.error(f"Validation error: {error_msg}")
            return None

        # Extract user and model messages
        user_msgs = [t['message'] for t in turns if t['speaker'] == 'user']
        model_msgs = [t['message'] for t in turns if t['speaker'] == 'llm']

        # Store in session state so it persists across reruns
        st.session_state.parsed_messages = {'user': user_msgs, 'model': model_msgs}

        st.success(f"✓ Parsed {len(user_msgs)} user messages and {len(model_msgs)} model messages")
        st.success("✓ Ready to analyze! Click '🚀 Analyze Conversation' button below.")

    # Return parsed messages from session state if available
    if hasattr(st.session_state, 'parsed_messages') and st.session_state.parsed_messages:
        return st.session_state.parsed_messages

    return None


def input_method_2_json():
    """Input Method 2: JSON File Upload."""
    st.subheader("📄 JSON Upload")

    uploaded_file = st.file_uploader(
        "Upload JSON Conversation",
        type=['json'],
        help="JSON file with 'conversation' array"
    )

    if uploaded_file is not None:
        try:
            # Read and parse JSON
            json_data = json.load(uploaded_file)

            # Validate
            is_valid, error_msg = ConversationValidator.validate_json(json_data)
            if not is_valid:
                st.error(f"Invalid JSON: {error_msg}")
                return None

            # Extract turns
            user_msgs, model_msgs = json_to_turns(json_data)

            st.success(f"Loaded {len(user_msgs)} user messages and {len(model_msgs)} model messages")

            # Show preview
            with st.expander("Preview Conversation"):
                for i in range(min(3, len(user_msgs))):
                    st.write(f"**Turn {i+1}**")
                    st.write(f"*User:* {user_msgs[i][:100]}...")
                    if i < len(model_msgs):
                        st.write(f"*Model:* {model_msgs[i][:100]}...")
                if len(user_msgs) > 3:
                    st.write(f"... and {len(user_msgs) - 3} more turns")

            return {'user': user_msgs, 'model': model_msgs}

        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON file: {e}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    return None


def input_method_3_csv():
    """Input Method 3: CSV File Upload."""
    st.subheader("📊 CSV Import")

    uploaded_file = st.file_uploader(
        "Upload CSV Conversation",
        type=['csv'],
        help="CSV with columns: turn, speaker, message"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Validate
            is_valid, error_msg = ConversationValidator.validate_csv(df)
            if not is_valid:
                st.error(f"Invalid CSV: {error_msg}")
                return None

            # Extract turns
            user_msgs, model_msgs = csv_to_turns(df)

            st.success(f"Loaded {len(user_msgs)} user messages and {len(model_msgs)} model messages")

            # Show preview
            with st.expander("Preview Data"):
                st.dataframe(df.head(10))

            return {'user': user_msgs, 'model': model_msgs}

        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    return None


def input_method_4_api():
    """Input Method 4: API Integration."""
    st.subheader("🔌 API Integration")

    st.info("Connect to a live LLM endpoint and analyze the conversation in real-time.")

    # Initialize custom endpoints in session state
    if 'custom_endpoints' not in st.session_state:
        st.session_state.custom_endpoints = {}

    # Combine predefined and custom endpoints
    all_endpoints = {**LLM_ENDPOINTS, **st.session_state.custom_endpoints}

    # Model selection
    col1, col2 = st.columns([3, 1])
    with col1:
        model_key = st.selectbox(
            "Select Model",
            options=list(all_endpoints.keys()),
            format_func=lambda x: all_endpoints[x]['name']
        )
    with col2:
        add_custom = st.button("➕ Add Custom", help="Add a new custom endpoint")

    # Add custom endpoint dialog
    if add_custom:
        st.session_state.show_custom_dialog = True

    if st.session_state.get('show_custom_dialog', False):
        st.divider()
        st.subheader("Add Custom Endpoint")

        custom_name = st.text_input("Endpoint Name", placeholder="e.g., My Custom LLM")
        custom_url = st.text_input("Endpoint URL", placeholder="https://api.example.com/chat")

        col_add, col_cancel = st.columns(2)
        with col_add:
            if st.button("Add Endpoint", type="primary"):
                if custom_name and custom_url:
                    # Generate a unique key
                    custom_key = f"custom_{len(st.session_state.custom_endpoints)}"
                    st.session_state.custom_endpoints[custom_key] = {
                        'name': custom_name,
                        'url': custom_url
                    }
                    st.session_state.show_custom_dialog = False
                    st.success(f"✓ Added: {custom_name}")
                    st.rerun()
                else:
                    st.error("Please provide both name and URL")
        with col_cancel:
            if st.button("Cancel"):
                st.session_state.show_custom_dialog = False
                st.rerun()
        st.divider()

    endpoint_url = st.text_input(
        "API Endpoint",
        value=all_endpoints[model_key]['url'],
        help="REST API endpoint for chat - you can edit this manually or add a custom endpoint above"
    )

    # Conversation builder
    st.write("**Build Conversation**")

    if 'api_conversation' not in st.session_state:
        st.session_state.api_conversation = []

    user_input = st.text_input("User Message:", key="api_user_input")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Send & Get Response"):
            if user_input.strip():
                try:
                    # Call API
                    with st.spinner("Getting response..."):
                        response = requests.post(
                            endpoint_url,
                            json={"message": user_input},
                            timeout=30
                        )
                        response.raise_for_status()
                        result = response.json()

                        model_response = result.get('response', result.get('message', 'No response'))

                        # Add to conversation
                        st.session_state.api_conversation.append({
                            'user': user_input,
                            'model': model_response
                        })

                        st.success("Response received!")
                except Exception as e:
                    st.error(f"API Error: {e}")
    with col2:
        if st.button("Clear Conversation"):
            st.session_state.api_conversation = []
            st.rerun()

    # Display conversation
    if st.session_state.api_conversation:
        st.write(f"**Conversation ({len(st.session_state.api_conversation)} turns)**")
        for i, turn in enumerate(st.session_state.api_conversation):
            with st.expander(f"Turn {i+1}"):
                st.write(f"**User:** {turn['user']}")
                st.write(f"**Model:** {turn['model']}")

        if st.button("Analyze This Conversation", type="primary"):
            user_msgs = [t['user'] for t in st.session_state.api_conversation]
            model_msgs = [t['model'] for t in st.session_state.api_conversation]

            # Store in session state so it persists across reruns
            st.session_state.parsed_messages = {'user': user_msgs, 'model': model_msgs}

            st.success(f"✓ Ready to analyze {len(user_msgs)} turns!")
            st.success("✓ Click '🚀 Analyze Conversation' button below.")

    # Return parsed messages from session state if available
    if hasattr(st.session_state, 'parsed_messages') and st.session_state.parsed_messages:
        return st.session_state.parsed_messages

    return None


def process_conversation(messages: Dict[str, List[str]], config: Dict):
    """Process conversation and generate analysis."""

    try:
        with st.spinner("Processing conversation..."):
            # Step 1: Convert text to 2D vectors
            st.info("Step 1/4: Converting text to embeddings and 2D vectors...")

            if st.session_state.pca_pipeline is None:
                st.session_state.pca_pipeline = load_pca_pipeline()
                if st.session_state.pca_pipeline is None:
                    st.error("❌ ERROR: Failed to load PCA pipeline. Check that models are in deployment/models/")
                    return

            pipeline = st.session_state.pca_pipeline

            # Convert VSAFE text to vector
            st.info(f"Converting VSAFE text to vector: '{config['vsafe_text'][:50]}...'")
            vsafe_vector = pipeline.text_to_2d(config['vsafe_text'])
            if vsafe_vector is None:
                st.error("❌ ERROR: Failed to convert VSAFE text to vector. Check AWS Bedrock credentials in .env")
                return

            st.success(f"✓ VSAFE vector created: {vsafe_vector}")

            # Convert user and model messages to vectors
            user_vectors = []
            model_vectors = []

            st.info(f"Converting {len(messages['user'])} user messages and {len(messages['model'])} model messages...")

            progress_bar = st.progress(0)
            total_messages = len(messages['user']) + len(messages['model'])
            processed = 0

            # Process user messages
            for i, msg in enumerate(messages['user']):
                st.info(f"Processing user message {i+1}/{len(messages['user'])}...")
                vec = pipeline.text_to_2d(msg, verbose=False)
                if vec is None:
                    st.error(f"❌ Failed to convert user message {i+1}: '{msg[:100]}...'")
                    return
                user_vectors.append(vec)
                processed += 1
                progress_bar.progress(processed / total_messages)

            # Process model messages
            for i, msg in enumerate(messages['model']):
                st.info(f"Processing model message {i+1}/{len(messages['model'])}...")
                vec = pipeline.text_to_2d(msg, verbose=False)
                if vec is None:
                    st.error(f"❌ Failed to convert model message {i+1}: '{msg[:100]}...'")
                    return
                model_vectors.append(vec)
                processed += 1
                progress_bar.progress(processed / total_messages)

            progress_bar.empty()

            # Check for failures (check if any vector is None)
            if any(v is None for v in user_vectors) or any(v is None for v in model_vectors):
                st.error("❌ ERROR: Failed to convert some messages to vectors. Check AWS Bedrock credentials.")
                return

            st.success(f"✓ Converted {len(model_vectors)} turns to 2D vectors")

            # Step 2: Initialize processor
            st.info("Step 2/4: Initializing Vector Precognition processor...")

            processor = VectorPrecognitionProcessor(
                vsafe=vsafe_vector,
                weights=config['weights']
            )

            # Step 3: Process conversation
            st.info("Step 3/4: Calculating risk metrics...")

            metrics_df = processor.process_conversation(
                model_vectors=model_vectors,
                user_vectors=user_vectors
            )

            statistics = processor.get_summary_statistics()

            # Store in session state
            st.session_state.metrics_df = metrics_df
            st.session_state.statistics = statistics
            st.session_state.processor = processor
            st.session_state.likelihood_threshold = config.get('likelihood_threshold', 0.8)
            st.session_state.erosion_threshold = config.get('erosion_threshold', 0.15)
            st.session_state.user_risk_threshold = config.get('user_risk_threshold', 2.0)

            st.success("✓ Analysis complete!")

    except Exception as e:
        st.error(f"❌ CRITICAL ERROR in process_conversation: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        error_trace = traceback.format_exc()
        st.code(error_trace)
        # Also log to console
        logger.error(f"Exception in process_conversation: {error_trace}")
        return


def display_results():
    """Display analysis results with visualizations and export options."""

    if st.session_state.metrics_df is None:
        st.warning("No analysis results available. Please analyze a conversation first.")
        return

    metrics_df = st.session_state.metrics_df
    statistics = st.session_state.statistics

    # Get thresholds from session state (set during process_conversation)
    likelihood_threshold = st.session_state.get('likelihood_threshold', 0.8)
    erosion_threshold = st.session_state.get('erosion_threshold', 0.15)
    user_risk_threshold = st.session_state.get('user_risk_threshold', 2.0)

    # Calculate threshold breaches
    likelihood_breaches = (metrics_df['Likelihood_L(N)'] > likelihood_threshold).sum()
    erosion_breaches = (metrics_df['GuardrailErosion_a(N)'].abs() > erosion_threshold).sum()

    # Check accumulated user risk
    cumulative_user_risk = statistics.get('cumulative_user_risk', 0.0)
    user_risk_exceeded = cumulative_user_risk > user_risk_threshold

    total_turns = len(metrics_df)

    st.header("📊 Analysis Results")

    # Show USER RISK alert first (highest priority)
    if user_risk_exceeded:
        st.error(f"🚨 **USER RISK ALERT:** Cumulative user risk ({cumulative_user_risk:.3f}) exceeds threshold ({user_risk_threshold:.1f})")
        st.warning("⚠️ Your prompts indicate possible policy violations. Review company use policies.")
        # Reset the popup flag to show it again for this analysis
        st.session_state.show_user_risk_popup = True
        show_user_risk_alert_popup()

    # Alert Banner (if any thresholds breached)
    elif likelihood_breaches > 0 or erosion_breaches > 0:
        alert_col1, alert_col2 = st.columns([1, 5])
        with alert_col1:
            st.markdown("### 🚨")
        with alert_col2:
            alert_messages = []
            if likelihood_breaches > 0:
                alert_messages.append(f"**{likelihood_breaches}** Likelihood alerts")
            if erosion_breaches > 0:
                alert_messages.append(f"**{erosion_breaches}** Erosion alerts")

            st.error(f"⚠️ **ALERTS DETECTED:** {' | '.join(alert_messages)}")

        # Show popup alert if erosion threshold is exceeded (only once per analysis)
        if erosion_breaches > 0:
            # Reset the popup flag to show it again for this analysis
            st.session_state.show_erosion_popup = True
            show_erosion_alert_popup()

    # Statistics Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Turns",
            statistics.get('total_turns', 0)
        )

    with col2:
        peak_risk = statistics.get('peak_risk_severity', 0)
        st.metric(
            "Peak Risk",
            f"{peak_risk:.3f}",
            delta=f"Turn {statistics.get('peak_risk_turn', 0)}"
        )

    with col3:
        peak_likelihood = statistics.get('peak_likelihood', 0)
        breach_pct = (likelihood_breaches / total_turns * 100) if total_turns > 0 else 0
        st.metric(
            "Peak Likelihood",
            f"{peak_likelihood:.3f}",
            delta=f"{likelihood_breaches} alerts ({breach_pct:.1f}%)",
            delta_color="inverse" if likelihood_breaches > 0 else "normal"
        )

    with col4:
        max_erosion = statistics.get('max_erosion', 0)
        erosion_breach_pct = (erosion_breaches / total_turns * 100) if total_turns > 0 else 0
        st.metric(
            "Max Erosion",
            f"{max_erosion:.3f}",
            delta=f"{erosion_breaches} alerts ({erosion_breach_pct:.1f}%)",
            delta_color="inverse" if erosion_breaches > 0 else "normal"
        )

    st.divider()

    # Visualizations
    st.subheader("📈 Visualizations")

    visualizer = GuardrailVisualizer()

    # 5-Panel Dynamics Plot with Threshold Lines
    with st.spinner("Generating dynamics plot..."):
        fig_dynamics = visualizer.plot_5panel_dynamics(
            metrics_df,
            alert_threshold=likelihood_threshold,
            erosion_threshold=erosion_threshold,
            title="Conversation Dynamics Analysis"
        )
        st.pyplot(fig_dynamics)

    st.divider()

    # Metrics Table with Threshold Highlights
    st.subheader("📋 Detailed Metrics")

    # Create styled dataframe
    def highlight_breaches(row):
        """Highlight rows that breach thresholds."""
        styles = [''] * len(row)

        # Check likelihood breach
        if 'Likelihood_L(N)' in row.index:
            if row['Likelihood_L(N)'] > likelihood_threshold:
                likelihood_idx = row.index.get_loc('Likelihood_L(N)')
                styles[likelihood_idx] = 'background-color: #ffcccc; font-weight: bold'

        # Check erosion breach
        if 'GuardrailErosion_a(N)' in row.index:
            if abs(row['GuardrailErosion_a(N)']) > erosion_threshold:
                erosion_idx = row.index.get_loc('GuardrailErosion_a(N)')
                styles[erosion_idx] = 'background-color: #ffe6cc; font-weight: bold'

        return styles

    # Display styled dataframe
    styled_df = metrics_df.style.apply(highlight_breaches, axis=1)
    st.dataframe(styled_df, use_container_width=True)

    # Legend for highlights
    st.caption("🔴 **Red highlight**: Likelihood threshold breached | 🟠 **Orange highlight**: Erosion threshold breached")

    # Statistics
    st.divider()
    st.markdown(format_statistics_display(statistics))

    # Export Options
    st.divider()
    st.subheader("💾 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = export_metrics_to_csv(metrics_df)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=create_export_filename("guardrail_metrics", "csv"),
            mime="text/csv"
        )

    with col2:
        json_data = export_metrics_to_json(metrics_df, statistics)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=create_export_filename("guardrail_analysis", "json"),
            mime="application/json"
        )

    with col3:
        png_data = fig_to_bytes(fig_dynamics, format='png', dpi=150)
        st.download_button(
            label="📥 Download Plot (PNG)",
            data=png_data,
            file_name=create_export_filename("dynamics_plot", "png"),
            mime="image/png"
        )


def main():
    """Main application entry point."""

    initialize_session_state()

    # Header with logo on the RIGHT
    col_title, col_logo = st.columns([7, 3])
    with col_title:
        st.markdown('<p class="main-header">Guardrail Erosion Analyzer</p>', unsafe_allow_html=True)
        st.markdown("**Analyze AI conversation safety through guardrail erosion and risk velocity metrics**")
    with col_logo:
        logo_path = deployment_root / 'shared' / 'images' / '1.png'
        if logo_path.exists():
            st.image(str(logo_path))

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
        st.header("Input Conversation")

    input_method = st.radio(
        "Select Input Method:",
        options=[
            "1. Manual Text Input",
            "2. JSON Upload",
            "3. CSV Import",
            "4. API Integration"
        ],
        horizontal=True
    )

    # Clear results if input method changed
    if 'last_input_method' not in st.session_state:
        st.session_state.last_input_method = input_method
    elif st.session_state.last_input_method != input_method:
        # Input method changed - clear previous results
        st.session_state.metrics_df = None
        st.session_state.statistics = None
        st.session_state.processor = None
        st.session_state.parsed_messages = None
        st.session_state.last_input_method = input_method

    # Process selected input method
    messages = None

    if "Manual" in input_method:
        messages = input_method_1_manual()
    elif "JSON" in input_method:
        messages = input_method_2_json()
    elif "CSV" in input_method:
        messages = input_method_3_csv()
    elif "API" in input_method:
        messages = input_method_4_api()

    # Analyze button
    if messages is not None:
        st.divider()
        if st.button("🚀 Analyze Conversation", type="primary", use_container_width=True):
            process_conversation(messages, config)
            # No st.rerun() needed - results will display automatically

    # Display results
    st.divider()
    display_results()


if __name__ == "__main__":
    main()
