#!/usr/bin/env python3
"""
Chat View Component - Interactive chat interface with live safety monitoring.

Displays:
- Chat message history
- Real-time risk metrics
- Live guardrail monitoring visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ChatView:
    """
    Chat interface component with message display and input.

    Integrates with LLM client and pipeline orchestrator for
    real-time safety monitoring.
    """

    def __init__(self):
        """Initialize chat view."""
        self.message_count = 0

    def render_chat_history(self, messages: list):
        """
        Render chat message history.

        Args:
            messages: List of message dicts with {role, content, timestamp}
        """
        if not messages:
            st.info("No messages yet. Start the conversation below.")
            return

        # Display messages
        for msg in messages:
            role = msg['role']
            content = msg['content']
            timestamp = msg.get('timestamp', datetime.now())

            # Format timestamp
            time_str = timestamp.strftime('%H:%M:%S')

            with st.chat_message(role):
                st.markdown(content)
                st.caption(f"⏰ {time_str}")

    def render_input_area(
        self,
        disabled: bool = False,
        placeholder: str = "Type your message..."
    ) -> Optional[str]:
        """
        Render chat input area.

        Args:
            disabled: Whether input is disabled
            placeholder: Placeholder text

        Returns:
            User input text if submitted, None otherwise
        """
        # Chat input
        user_input = st.chat_input(
            placeholder=placeholder,
            disabled=disabled,
            key=f"chat_input_{self.message_count}"
        )

        return user_input

    def render_live_metrics(self, orchestrator):
        """
        Render live metrics panel showing current conversation risk.

        Args:
            orchestrator: PipelineOrchestrator instance
        """
        st.subheader("📊 Live Safety Metrics")

        status = orchestrator.get_current_status()

        if not status.get('has_active_conversation', False):
            st.info("No active conversation. Start chatting to see metrics.")
            return

        # Get Stage 1 metrics
        stage1 = status.get('stage1', {})
        turns = status.get('total_turns', 0)

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Conversation Turns",
                value=turns
            )

        with col2:
            peak_risk = stage1.get('peak_risk', 0.0)
            st.metric(
                label="Peak Risk",
                value=f"{peak_risk:.3f}",
                delta=None
            )

        with col3:
            current_risk = stage1.get('current_risk', 0.0)
            st.metric(
                label="Current Risk",
                value=f"{current_risk:.3f}",
                delta=None
            )

        with col4:
            peak_likelihood = stage1.get('peak_likelihood', 0.0)
            st.metric(
                label="Peak Likelihood",
                value=f"{peak_likelihood:.1%}",
                delta=None
            )

        # Risk status indicator
        if peak_likelihood >= 0.8:
            st.error("⚠️ HIGH RISK: Guardrail breach likely detected!")
        elif peak_likelihood >= 0.5:
            st.warning("⚡ MODERATE RISK: Elevated safety concerns")
        else:
            st.success("✅ SAFE: Conversation within safety bounds")

    def render_live_visualization(self, orchestrator, alert_threshold: float = 0.8):
        """
        Render live visualization of conversation dynamics.

        Args:
            orchestrator: PipelineOrchestrator instance
            alert_threshold: Alert threshold for likelihood
        """
        st.subheader("📈 Real-Time Guardrail Monitoring")

        if not orchestrator.current_conversation_id:
            st.info("No active conversation to visualize.")
            return

        # Get metrics
        metrics_df = orchestrator.get_stage1_metrics()

        if len(metrics_df) == 0:
            st.info("No data yet. Continue conversation to see dynamics.")
            return

        # Import visualizer
        import sys
        import os
        deployment_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, os.path.join(deployment_root, 'shared'))

        from visualizations import GuardrailVisualizer

        # Create visualization
        viz = GuardrailVisualizer()

        try:
            fig = viz.plot_5panel_dynamics(
                metrics_df,
                alert_threshold=alert_threshold,
                title=f"Conversation: {orchestrator.current_conversation_id}"
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating visualization: {e}")
            logger.error(f"Visualization error: {e}")

    def render_conversation_controls(self, orchestrator) -> Tuple[bool, bool, bool]:
        """
        Render conversation control buttons.

        Args:
            orchestrator: PipelineOrchestrator instance

        Returns:
            Tuple of (start_clicked, end_clicked, export_clicked)
        """
        col1, col2, col3 = st.columns(3)

        with col1:
            start_clicked = st.button(
                "🆕 Start New Conversation",
                disabled=orchestrator.current_conversation_id is not None,
                use_container_width=True
            )

        with col2:
            end_clicked = st.button(
                "🛑 End Conversation",
                disabled=orchestrator.current_conversation_id is None,
                use_container_width=True,
                type="primary"
            )

        with col3:
            export_clicked = st.button(
                "💾 Export Conversation",
                disabled=orchestrator.current_conversation_id is None,
                use_container_width=True
            )

        return start_clicked, end_clicked, export_clicked

    def render_model_info(self, llm_client):
        """
        Render current model information.

        Args:
            llm_client: LLMAPIClient instance
        """
        if llm_client is None:
            st.warning("No LLM client initialized")
            return

        # Display model info
        st.info(f"🤖 **Current Model:** {llm_client.config.name}")

        # Display message count
        msg_count = len(llm_client.get_conversation_history())
        st.caption(f"📝 Messages in LLM context: {msg_count}")

    def render_alert_panel(self, metrics_df: pd.DataFrame, threshold: float = 0.8):
        """
        Render alert panel for high-risk turns.

        Args:
            metrics_df: Metrics DataFrame
            threshold: Likelihood threshold for alerts
        """
        if len(metrics_df) == 0:
            return

        # Find high-risk turns
        high_risk = metrics_df[metrics_df['Likelihood_L(N)'] >= threshold]

        if len(high_risk) == 0:
            return

        st.warning(f"⚠️ **{len(high_risk)} High-Risk Turn(s) Detected**")

        with st.expander("View High-Risk Turns"):
            for idx, row in high_risk.iterrows():
                turn = row['Turn']
                likelihood = row['Likelihood_L(N)']
                risk = row['RiskSeverity_Model']

                st.markdown(
                    f"**Turn {turn}:** Risk={risk:.3f}, "
                    f"Likelihood={likelihood:.1%}"
                )

    def render_statistics_panel(self, orchestrator):
        """
        Render conversation statistics panel.

        Args:
            orchestrator: PipelineOrchestrator instance
        """
        st.subheader("📊 Conversation Statistics")

        status = orchestrator.get_current_status()

        if not status.get('has_active_conversation', False):
            st.info("No active conversation")
            return

        # Create statistics table
        stats = {
            "Total Turns": status.get('total_turns', 0),
            "Total Conversations (Session)": status.get('total_conversations', 0),
            "Completed Conversations": status.get('completed_conversations', 0),
        }

        # Add Stage 1 metrics
        stage1 = status.get('stage1', {})
        if stage1:
            stats.update({
                "Peak Risk Severity": f"{stage1.get('peak_risk', 0.0):.3f}",
                "Peak Likelihood": f"{stage1.get('peak_likelihood', 0.0):.1%}",
                "Current Risk": f"{stage1.get('current_risk', 0.0):.3f}"
            })

        # Add Stage 2 metrics if available
        stage2 = status.get('stage2', {})
        if stage2:
            stats.update({
                "RHO (Robustness Index)": f"{stage2.get('final_rho', 0.0):.3f}",
                "Classification": stage2.get('classification', 'N/A'),
                "Is Robust": "Yes" if stage2.get('is_robust', False) else "No"
            })

        # Display as DataFrame
        stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
        # Convert values to strings to avoid Arrow serialization errors
        stats_df['Value'] = stats_df['Value'].astype(str)
        st.dataframe(stats_df, hide_index=True, width='stretch')


class ChatControls:
    """Controls for chat settings and configuration."""

    @staticmethod
    def render_temperature_slider() -> float:
        """
        Render temperature control slider.

        Returns:
            Temperature value
        """
        return st.slider(
            "🌡️ Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in model responses. Lower = more focused, Higher = more creative"
        )

    @staticmethod
    def render_max_tokens_slider() -> int:
        """
        Render max tokens control slider.

        Returns:
            Max tokens value
        """
        return st.slider(
            "📝 Max Tokens",
            min_value=128,
            max_value=4096,
            value=1024,
            step=128,
            help="Maximum length of model response"
        )

    @staticmethod
    def render_system_prompt_input(default: str = "") -> str:
        """
        Render system prompt input.

        Args:
            default: Default system prompt

        Returns:
            System prompt text
        """
        return st.text_area(
            "⚙️ System Prompt",
            value=default,
            height=100,
            help="Instructions for the AI model"
        )

    @staticmethod
    def render_mock_toggle(default: bool = False) -> bool:
        """
        Render mock client toggle.

        Args:
            default: Default toggle state

        Returns:
            Whether to use mock client
        """
        return st.checkbox(
            "🧪 Use Mock Client (for testing without API keys)",
            value=default,
            help="Uses simulated responses instead of real API calls"
        )


def create_chat_view() -> ChatView:
    """
    Factory function to create ChatView instance.

    Returns:
        ChatView instance
    """
    return ChatView()
