#!/usr/bin/env python3
"""
Desktop App: Unified AI Safety Dashboard with ChatGPT Integration

Features:
- Live ChatGPT conversation with real-time risk monitoring
- Vector Precognition analysis for every turn
- Dynamic 4-panel risk visualization
- Multi-conversation tracking with RHO and PHI metrics
- Secure API key management via Electron

Integrates all three stages:
- Stage 1: Guardrail Erosion (real-time per turn)
- Stage 2: RHO Calculation (per conversation)
- Stage 3: PHI Aggregation (across conversations)
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import json
import asyncio
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Page configuration
st.set_page_config(
    page_title="Vector Precognition Desktop - ChatGPT Safety Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules
try:
    from chatgpt_client import ChatGPTRiskMonitor, create_chatgpt_monitor
    from shared.pca_pipeline import PCATransformer
    from shared.config import DEFAULT_WEIGHTS, DEFAULT_VSAFE_TEXT
    from shared.visualizations import GuardrailVisualizer
    from utils.session_state import SessionState
    from ui.sidebar import create_sidebar

    # Check if required modules exist
    CHATGPT_AVAILABLE = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    CHATGPT_AVAILABLE = False


def initialize_session_state():
    """Initialize all session state variables."""

    # API Configuration
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""

    if 'api_key_configured' not in st.session_state:
        st.session_state.api_key_configured = False

    # ChatGPT Monitor
    if 'chatgpt_monitor' not in st.session_state:
        st.session_state.chatgpt_monitor = None

    # PCA Transformer
    if 'pca_transformer' not in st.session_state:
        st.session_state.pca_transformer = None

    # Conversation State
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    if 'turn_metrics' not in st.session_state:
        st.session_state.turn_metrics = []

    if 'current_turn' not in st.session_state:
        st.session_state.current_turn = 0

    # Configuration
    if 'model_selection' not in st.session_state:
        st.session_state.model_selection = "gpt-4"

    if 'vsafe_text' not in st.session_state:
        st.session_state.vsafe_text = DEFAULT_VSAFE_TEXT

    if 'vsafe_vector' not in st.session_state:
        st.session_state.vsafe_vector = None

    # Alert state
    if 'show_erosion_popup' not in st.session_state:
        st.session_state.show_erosion_popup = False

    if 'alert_shown_for_turn' not in st.session_state:
        st.session_state.alert_shown_for_turn = set()

    # Conversation history (for multi-conversation PHI calculation)
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []


def render_api_key_setup():
    """Render API key configuration interface."""
    st.title("🔐 ChatGPT Configuration")

    st.markdown("""
    Welcome to **Vector Precognition Desktop**! This application integrates ChatGPT
    with real-time AI safety monitoring.

    ### Setup Instructions
    1. Obtain your OpenAI API key from [platform.openai.com](https://platform.openai.com/api-keys)
    2. Enter your API key below (it will be securely stored)
    3. Select your preferred GPT model
    4. Start chatting with real-time risk analysis!
    """)

    # API Key Input
    st.subheader("Step 1: Enter API Key")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your API key is stored securely and never shared"
    )

    # Model Selection
    st.subheader("Step 2: Select Model")
    model_options = {
        "GPT-3.5 Turbo (Recommended)": "gpt-3.5-turbo",
        "GPT-4o Mini": "gpt-4o-mini",
        "GPT-4o": "gpt-4o",
        "GPT-4 Turbo": "gpt-4-turbo",
        "GPT-4": "gpt-4"
    }

    selected_model_name = st.selectbox(
        "ChatGPT Model",
        options=list(model_options.keys()),
        index=0,  # Default to GPT-3.5 Turbo
        help="GPT-3.5 Turbo works with all API keys. GPT-4 models require upgraded access."
    )
    st.session_state.model_selection = model_options[selected_model_name]

    # VSAFE Configuration
    st.subheader("Step 3: Configure Safe Harbor (VSAFE)")
    vsafe_presets = {
        "Default Assistant": "I am a helpful, harmless, and honest AI assistant.",
        "Safety Focused": "I prioritize user safety and ethical guidelines above all else.",
        "Helpful & Harmless": "I aim to be helpful while avoiding any potential harm.",
        "Custom": ""
    }

    vsafe_preset = st.selectbox("VSAFE Preset", options=list(vsafe_presets.keys()))

    if vsafe_preset == "Custom":
        st.session_state.vsafe_text = st.text_area(
            "Custom VSAFE Text",
            value=DEFAULT_VSAFE_TEXT,
            height=100
        )
    else:
        st.session_state.vsafe_text = vsafe_presets[vsafe_preset]

    st.info(f"**Current VSAFE:** {st.session_state.vsafe_text}")

    # Save Configuration
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("💾 Save & Continue", type="primary", use_container_width=True):
            if not api_key_input:
                st.error("❌ Please enter your API key")
            elif not api_key_input.startswith("sk-"):
                st.error("❌ Invalid API key format (should start with 'sk-')")
            else:
                try:
                    # Store API key
                    st.session_state.openai_api_key = api_key_input

                    # Initialize PCA transformer
                    st.session_state.pca_transformer = PCATransformer()

                    # Calculate VSAFE vector
                    st.session_state.vsafe_vector = st.session_state.pca_transformer.text_to_2d(
                        st.session_state.vsafe_text
                    )

                    # Create ChatGPT monitor
                    st.session_state.chatgpt_monitor = create_chatgpt_monitor(
                        api_key=api_key_input,
                        model=st.session_state.model_selection,
                        vsafe_text=st.session_state.vsafe_text
                    )

                    # Set VSAFE anchor
                    st.session_state.chatgpt_monitor.set_vsafe_anchor(st.session_state.vsafe_vector)

                    # Mark as configured
                    st.session_state.api_key_configured = True

                    st.success("✅ Configuration saved! Reloading...")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error initializing: {str(e)}")
                    logger.error(f"Initialization error: {e}")


def render_chat_interface():
    """Render main chat interface with real-time risk monitoring."""

    # Header
    st.title("🛡️ Vector Precognition Desktop - ChatGPT Safety Monitor")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.info(f"**Model:** {st.session_state.model_selection}")
        st.info(f"**Turns:** {st.session_state.current_turn}")

        # Temperature control
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )

        # Max tokens
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100
        )

        st.divider()

        # Current RHO display
        if st.session_state.turn_metrics:
            latest_metrics = st.session_state.turn_metrics[-1]
            rho = latest_metrics['rho']

            st.metric("🎯 Robustness Index (ρ)", f"{rho:.3f}")

            if rho < 1.0:
                st.success("✅ Model is ROBUST")
            elif rho == 1.0:
                st.warning("⚠️ Model is REACTIVE")
            else:
                st.error("❌ Model is FRAGILE")

        st.divider()

        # Reset button
        if st.button("🔄 New Conversation", use_container_width=True):
            # Save current conversation to history
            if st.session_state.turn_metrics:
                summary = st.session_state.chatgpt_monitor.get_conversation_summary()
                st.session_state.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'summary': summary,
                    'metrics': st.session_state.turn_metrics
                })

            # Reset
            st.session_state.chatgpt_monitor.reset_conversation()
            st.session_state.chat_messages = []
            st.session_state.turn_metrics = []
            st.session_state.current_turn = 0
            st.session_state.show_erosion_popup = False
            st.session_state.alert_shown_for_turn = set()
            st.rerun()

        # Reconfigure API
        if st.button("🔐 Change API Key", use_container_width=True):
            st.session_state.api_key_configured = False
            st.rerun()

    # Main chat area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Chat")

        # Chat history container
        chat_container = st.container(height=500)

        with chat_container:
            if not st.session_state.chat_messages:
                st.info("👋 Start chatting! Your conversation will be monitored for safety in real-time.")
            else:
                for msg in st.session_state.chat_messages:
                    with st.chat_message(msg['role']):
                        st.markdown(msg['content'])
                        if 'timestamp' in msg:
                            st.caption(f"⏰ {msg['timestamp'].strftime('%H:%M:%S')}")

        # Chat input
        user_input = st.chat_input(
            "Type your message...",
            key="chat_input"
        )

        if user_input:
            # Process message
            process_user_message(user_input, temperature, max_tokens)

    with col2:
        st.subheader("📊 Real-Time Risk Analysis")

        if st.session_state.turn_metrics:
            # Get latest metrics
            latest = st.session_state.turn_metrics[-1]

            # Display user metrics
            st.markdown("**👤 User Turn**")
            user_metrics = latest['user']
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Risk (R)", f"{user_metrics['R']:.3f}")
                st.metric("Velocity (v)", f"{user_metrics['v']:.3f}")
            with col_b:
                st.metric("Erosion (a)", f"{user_metrics['a']:.3f}")
                st.metric("Breach (L)", f"{user_metrics['L']:.3f}")

            # Display assistant metrics
            st.markdown("**🤖 Assistant Turn**")
            asst_metrics = latest['assistant']
            col_c, col_d = st.columns(2)
            with col_c:
                st.metric("Risk (R)", f"{asst_metrics['R']:.3f}")
                st.metric("Velocity (v)", f"{asst_metrics['v']:.3f}")
            with col_d:
                st.metric("Erosion (a)", f"{asst_metrics['a']:.3f}")
                st.metric("Breach (L)", f"{asst_metrics['L']:.3f}")

            # Alert check - only show if threshold exceeded AND not already acknowledged
            if asst_metrics['L'] > 0.8 or user_metrics['L'] > 0.8:
                st.error("⚠️ **HIGH RISK DETECTED**")
                # Only trigger popup once per conversation (not on every turn)
                if 'alert_shown_for_turn' not in st.session_state:
                    st.session_state.alert_shown_for_turn = set()

                current_turn = st.session_state.current_turn
                if current_turn not in st.session_state.alert_shown_for_turn:
                    st.session_state.show_erosion_popup = True
                    st.session_state.alert_shown_for_turn.add(current_turn)

            # Plot metrics over time
            st.divider()
            st.markdown("**📈 Risk Trajectory**")

            df = pd.DataFrame(st.session_state.turn_metrics)

            # Extract user and assistant R values
            user_R = [m['user']['R'] for m in st.session_state.turn_metrics]
            asst_R = [m['assistant']['R'] for m in st.session_state.turn_metrics]
            turns = list(range(1, len(st.session_state.turn_metrics) + 1))

            chart_data = pd.DataFrame({
                'Turn': turns + turns,
                'Risk (R)': user_R + asst_R,
                'Speaker': ['User'] * len(user_R) + ['Assistant'] * len(asst_R)
            })

            st.line_chart(chart_data, x='Turn', y='Risk (R)', color='Speaker')
        else:
            st.info("Start a conversation to see real-time risk analysis")


def process_user_message(user_input: str, temperature: float, max_tokens: int):
    """Process user message and get ChatGPT response with risk analysis."""

    try:
        # Add user message to display
        user_msg = {
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        }
        st.session_state.chat_messages.append(user_msg)

        # Get user vector
        user_vector = st.session_state.pca_transformer.text_to_2d(user_input)

        # Send to ChatGPT (synchronous wrapper)
        with st.spinner("🤖 ChatGPT is thinking..."):
            response = asyncio.run(
                st.session_state.chatgpt_monitor.send_message(
                    user_message=user_input,
                    user_vector=user_vector,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )

        if not response['success']:
            st.error(f"❌ Error: {response.get('error', 'Unknown error')}")
            return

        assistant_message = response['assistant_message']

        # Add assistant message to display
        asst_msg = {
            'role': 'assistant',
            'content': assistant_message,
            'timestamp': datetime.now()
        }
        st.session_state.chat_messages.append(asst_msg)

        # Get assistant vector
        assistant_vector = st.session_state.pca_transformer.text_to_2d(assistant_message)

        # Analyze turn with Vector Precognition
        turn_metrics = st.session_state.chatgpt_monitor.analyze_turn(
            user_vector=user_vector,
            assistant_vector=assistant_vector
        )

        st.session_state.turn_metrics.append(turn_metrics)
        st.session_state.current_turn += 1

        # Trigger rerun to update UI
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error processing message: {str(e)}")
        logger.error(f"Message processing error: {e}")


def show_erosion_alert():
    """Show erosion alert dialog."""
    @st.dialog("⚠️ SAFETY ALERT", width="large")
    def alert_dialog():
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 80px;">⚠️</div>
            <h2 style="color: #d63031;">AI Guardrail Erosion Threshold Exceeded</h2>
            <p style="font-size: 18px; background: #ffe5e5; padding: 20px; border-radius: 10px; border-left: 5px solid #ff6b6b;">
                This model is approaching a break of its assigned guardrails.<br><br>
                <strong>Please close this context window and restart your conversation in a new instance.</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✓ I Understand", type="primary", use_container_width=True):
            st.session_state.show_erosion_popup = False
            st.rerun()

    if st.session_state.show_erosion_popup:
        alert_dialog()


def main():
    """Main application entry point."""

    # Initialize session state
    initialize_session_state()

    # Check if API key is configured
    if not st.session_state.api_key_configured:
        render_api_key_setup()
    else:
        render_chat_interface()

        # Show erosion alert if needed
        show_erosion_alert()


if __name__ == "__main__":
    main()
