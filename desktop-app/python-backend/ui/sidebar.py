#!/usr/bin/env python3
"""
Sidebar Component - Navigation, model selection, and configuration.

Provides:
- Model selection and configuration
- Algorithm parameter controls
- Navigation between tabs
- Export settings
"""

import streamlit as st
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Sidebar:
    """
    Sidebar component for app navigation and configuration.

    Handles model selection, algorithm parameters, and global settings.
    """

    def __init__(self):
        """Initialize sidebar."""
        pass

    def render(self) -> Dict:
        """
        Render complete sidebar.

        Returns:
            Dictionary with all sidebar selections and settings
        """
        with st.sidebar:
            st.title("⚙️ Configuration")

            # Model selection
            model_config = self._render_model_selection()

            st.divider()

            # Algorithm parameters
            algorithm_params = self._render_algorithm_parameters()

            st.divider()

            # VSAFE configuration
            vsafe_config = self._render_vsafe_configuration()

            st.divider()

            # Alert settings
            alert_settings = self._render_alert_settings()

            st.divider()

            # Export settings
            export_settings = self._render_export_settings()

            st.divider()

            # Session controls
            self._render_session_controls()

            # Combine all settings
            return {
                'model': model_config,
                'algorithm': algorithm_params,
                'vsafe': vsafe_config,
                'alerts': alert_settings,
                'export': export_settings
            }

    def _render_model_selection(self) -> Dict:
        """
        Render model selection controls.

        Returns:
            Dictionary with model configuration
        """
        st.subheader("🤖 Model Selection")

        # Import API client to get available models
        import sys
        import os
        deployment_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, os.path.join(deployment_root, 'app4_unified_dashboard', 'core'))

        from api_client import LLMAPIClient

        # Get available models
        available_models = LLMAPIClient.get_available_models()

        # Model selection dropdown
        model_key = st.selectbox(
            "Select Model",
            options=list(available_models.keys()),
            format_func=lambda x: available_models[x],
            key="sidebar_model_select"
        )

        # Check API key status
        is_configured = LLMAPIClient.is_api_key_configured(model_key)

        if not is_configured:
            st.warning(f"⚠️ API key not configured for {available_models[model_key]}")
            st.caption("Set environment variable or use mock client")

        # Mock client toggle
        use_mock = st.checkbox(
            "Use Mock Client",
            value=not is_configured,
            help="Use simulated responses for testing",
            key="sidebar_use_mock"
        )

        # Temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls response randomness",
            key="sidebar_temperature"
        )

        # Max tokens
        max_tokens = st.slider(
            "Max Tokens",
            min_value=128,
            max_value=4096,
            value=1024,
            step=128,
            help="Maximum response length",
            key="sidebar_max_tokens"
        )

        return {
            'model_key': model_key,
            'model_name': available_models[model_key],
            'use_mock': use_mock,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'api_configured': is_configured
        }

    def _render_algorithm_parameters(self) -> Dict:
        """
        Render algorithm parameter controls.

        Returns:
            Dictionary with algorithm weights
        """
        st.subheader("🔬 Algorithm Parameters")

        st.caption("Weights for risk calculation")

        # Weight for Risk Severity (position)
        wR = st.slider(
            "wR (Risk Severity)",
            min_value=0.0,
            max_value=5.0,
            value=1.5,
            step=0.1,
            help="Weight for distance from safe harbor",
            key="sidebar_wR"
        )

        # Weight for Risk Rate (velocity)
        wv = st.slider(
            "wv (Risk Rate)",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Weight for velocity of drift",
            key="sidebar_wv"
        )

        # Weight for Erosion (acceleration)
        wa = st.slider(
            "wa (Guardrail Erosion)",
            min_value=0.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Weight for acceleration (most critical)",
            key="sidebar_wa"
        )

        # Bias
        b = st.slider(
            "b (Bias)",
            min_value=-10.0,
            max_value=10.0,
            value=-2.5,
            step=0.1,
            help="Baseline threshold adjustment",
            key="sidebar_b"
        )

        # Preset buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Reset to Default", use_container_width=True):
                st.session_state.sidebar_wR = 1.5
                st.session_state.sidebar_wv = 1.0
                st.session_state.sidebar_wa = 3.0
                st.session_state.sidebar_b = -2.5
                st.rerun()

        with col2:
            if st.button("Conservative", use_container_width=True):
                st.session_state.sidebar_wR = 2.0
                st.session_state.sidebar_wv = 1.5
                st.session_state.sidebar_wa = 4.0
                st.session_state.sidebar_b = -3.0
                st.rerun()

        return {
            'wR': wR,
            'wv': wv,
            'wa': wa,
            'b': b
        }

    def _render_vsafe_configuration(self) -> Dict:
        """
        Render VSAFE (safe harbor) configuration.

        Returns:
            Dictionary with VSAFE settings
        """
        st.subheader("🛡️ Safe Harbor (VSAFE)")

        # Preset selection
        vsafe_preset = st.selectbox(
            "VSAFE Preset",
            options=[
                "Default Safety",
                "Ethical Behavior",
                "Helpful Assistant",
                "Custom"
            ],
            key="sidebar_vsafe_preset"
        )

        # Map presets to text
        preset_texts = {
            "Default Safety": "I prioritize safety, accuracy, and ethical behavior in all responses.",
            "Ethical Behavior": "I am committed to ethical AI principles and responsible assistance.",
            "Helpful Assistant": "I am a helpful, harmless, and honest AI assistant.",
            "Custom": ""
        }

        # Text input (enabled for Custom)
        vsafe_text = st.text_area(
            "VSAFE Text",
            value=preset_texts.get(vsafe_preset, preset_texts["Default Safety"]),
            height=100,
            disabled=(vsafe_preset != "Custom"),
            help="Reference text for safe behavior",
            key="sidebar_vsafe_text"
        )

        return {
            'preset': vsafe_preset,
            'text': vsafe_text
        }

    def _render_alert_settings(self) -> Dict:
        """
        Render alert threshold settings.

        Returns:
            Dictionary with alert settings
        """
        st.subheader("🚨 Alert Settings")

        # Likelihood threshold
        alert_threshold = st.slider(
            "Alert Threshold (Likelihood)",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Trigger alert when likelihood exceeds this value",
            key="sidebar_alert_threshold"
        )

        # Show visual indicator
        if alert_threshold >= 0.9:
            st.caption("🔴 Very strict (high sensitivity)")
        elif alert_threshold >= 0.7:
            st.caption("🟡 Moderate (balanced)")
        else:
            st.caption("🟢 Lenient (low sensitivity)")

        # Erosion threshold
        erosion_threshold = st.slider(
            "Erosion Alert Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
            help="Trigger alert when guardrail erosion exceeds this value",
            key="sidebar_erosion_threshold"
        )

        # Epsilon for RHO calculation
        epsilon = st.number_input(
            "Epsilon (ε)",
            min_value=1e-10,
            max_value=1e-3,
            value=1e-6,
            format="%.2e",
            help="Small constant to prevent division by zero",
            key="sidebar_epsilon"
        )

        # PHI threshold
        phi_threshold = st.slider(
            "PHI Threshold (Pass/Fail)",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Model passes if PHI < threshold",
            key="sidebar_phi_threshold"
        )

        return {
            'alert_threshold': alert_threshold,
            'erosion_threshold': erosion_threshold,
            'epsilon': epsilon,
            'phi_threshold': phi_threshold
        }

    def _render_export_settings(self) -> Dict:
        """
        Render export settings.

        Returns:
            Dictionary with export settings
        """
        st.subheader("💾 Export Settings")

        # Export directory
        export_dir = st.text_input(
            "Export Directory",
            value="exports",
            help="Directory for exported files",
            key="sidebar_export_dir"
        )

        # Export format options
        export_formats = st.multiselect(
            "Export Formats",
            options=["CSV", "JSON", "PNG", "PDF"],
            default=["CSV", "JSON", "PNG"],
            help="Formats for exported data",
            key="sidebar_export_formats"
        )

        # Include timestamps
        include_timestamps = st.checkbox(
            "Include Timestamps in Filenames",
            value=True,
            help="Add timestamp to exported filenames",
            key="sidebar_timestamps"
        )

        return {
            'directory': export_dir,
            'formats': export_formats,
            'include_timestamps': include_timestamps
        }

    def _render_session_controls(self):
        """Render session control buttons."""
        st.subheader("🔧 Session Controls")

        # Reset session
        if st.button("🔄 Reset Session", use_container_width=True):
            if st.session_state.get('confirm_reset', False):
                # Actually reset
                from utils.session_state import SessionState
                SessionState.reset()
                st.success("Session reset!")
                st.session_state.confirm_reset = False
                st.rerun()
            else:
                # Ask for confirmation
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset")

        # Export session state
        if st.button("📤 Export Session State", use_container_width=True):
            from utils.session_state import SessionState
            import json
            from datetime import datetime

            state = SessionState.export_state()
            filename = f"session_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            st.download_button(
                label="Download Session State",
                data=json.dumps(state, indent=2),
                file_name=filename,
                mime="application/json",
                use_container_width=True
            )

        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Unified AI Safety Dashboard**

            End-to-end AI safety monitoring system using
            Vector Precognition algorithm.

            **Stages:**
            1. Guardrail Erosion (per turn)
            2. RHO Calculation (per conversation)
            3. PHI Aggregation (across conversations)

            **Version:** 1.0.0
            """)


class QuickActions:
    """Quick action buttons for common tasks."""

    @staticmethod
    def render():
        """Render quick action buttons."""
        st.sidebar.subheader("⚡ Quick Actions")

        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("📊 View Stats", use_container_width=True):
                st.session_state['show_stats'] = True

        with col2:
            if st.button("📈 View Plots", use_container_width=True):
                st.session_state['show_plots'] = True


def create_sidebar() -> Sidebar:
    """
    Factory function to create Sidebar instance.

    Returns:
        Sidebar instance
    """
    return Sidebar()
