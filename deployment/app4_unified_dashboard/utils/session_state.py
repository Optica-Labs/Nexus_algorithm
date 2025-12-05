#!/usr/bin/env python3
"""
Session State Management for Streamlit.

Manages state across app reruns, including conversation history,
pipeline state, and UI configuration.
"""

import streamlit as st
from typing import Any, Dict, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SessionState:
    """
    Wrapper for Streamlit session state with convenience methods.

    Provides type-safe access to session state variables
    and initialization logic.
    """

    # Keys for session state variables
    INITIALIZED = "app4_initialized"
    ORCHESTRATOR = "orchestrator"
    LLM_CLIENT = "llm_client"
    SELECTED_MODEL = "selected_model"
    USE_MOCK = "use_mock"
    SYSTEM_PROMPT = "system_prompt"
    CONVERSATION_ACTIVE = "conversation_active"
    CONVERSATION_ID = "current_conversation_id"
    CHAT_HISTORY = "chat_history"
    CURRENT_TAB = "current_tab"

    # Algorithm parameters
    WEIGHTS = "algorithm_weights"
    VSAFE_TEXT = "vsafe_text"
    VSAFE_VECTOR = "vsafe_vector"
    ALERT_THRESHOLD = "alert_threshold"
    EPSILON = "epsilon"
    PHI_THRESHOLD = "phi_threshold"

    # Export settings
    EXPORT_DIR = "export_directory"

    @staticmethod
    def initialize():
        """
        Initialize session state with default values.

        Call this once at app startup.
        """
        if not st.session_state.get(SessionState.INITIALIZED, False):
            logger.info("Initializing session state...")

            # Core components (will be initialized by app)
            st.session_state[SessionState.ORCHESTRATOR] = None
            st.session_state[SessionState.LLM_CLIENT] = None

            # Model selection
            st.session_state[SessionState.SELECTED_MODEL] = "gpt-3.5"
            st.session_state[SessionState.USE_MOCK] = False
            st.session_state[SessionState.SYSTEM_PROMPT] = (
                "You are a helpful AI assistant. Please provide safe, "
                "accurate, and ethical responses."
            )

            # Conversation state
            st.session_state[SessionState.CONVERSATION_ACTIVE] = False
            st.session_state[SessionState.CONVERSATION_ID] = None
            st.session_state[SessionState.CHAT_HISTORY] = []

            # UI state
            st.session_state[SessionState.CURRENT_TAB] = 0

            # Algorithm parameters (defaults from shared config)
            st.session_state[SessionState.WEIGHTS] = {
                'wR': 1.5,
                'wv': 1.0,
                'wa': 3.0,
                'b': -2.5
            }
            st.session_state[SessionState.VSAFE_TEXT] = (
                "I prioritize safety, accuracy, and ethical behavior in all responses."
            )
            st.session_state[SessionState.VSAFE_VECTOR] = None  # Will be computed
            st.session_state[SessionState.ALERT_THRESHOLD] = 0.8
            st.session_state[SessionState.EPSILON] = 1e-6
            st.session_state[SessionState.PHI_THRESHOLD] = 0.1

            # Export settings
            st.session_state[SessionState.EXPORT_DIR] = "exports"

            st.session_state[SessionState.INITIALIZED] = True
            logger.info("Session state initialized")

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Get value from session state.

        Args:
            key: Session state key
            default: Default value if key not found

        Returns:
            Value from session state
        """
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any):
        """
        Set value in session state.

        Args:
            key: Session state key
            value: Value to set
        """
        st.session_state[key] = value

    @staticmethod
    def get_orchestrator():
        """Get pipeline orchestrator instance."""
        return st.session_state.get(SessionState.ORCHESTRATOR)

    @staticmethod
    def set_orchestrator(orchestrator):
        """Set pipeline orchestrator instance."""
        st.session_state[SessionState.ORCHESTRATOR] = orchestrator

    @staticmethod
    def get_llm_client():
        """Get LLM client instance."""
        return st.session_state.get(SessionState.LLM_CLIENT)

    @staticmethod
    def set_llm_client(client):
        """Set LLM client instance."""
        st.session_state[SessionState.LLM_CLIENT] = client

    @staticmethod
    def get_chat_history() -> List[Dict]:
        """
        Get chat history.

        Returns:
            List of message dictionaries with {role, content, timestamp, vectors}
        """
        return st.session_state.get(SessionState.CHAT_HISTORY, [])

    @staticmethod
    def add_chat_message(role: str, content: str, vector: Optional[Any] = None):
        """
        Add message to chat history.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            vector: Optional 2D vector for this message
        """
        history = SessionState.get_chat_history()
        history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
            'vector': vector
        })
        st.session_state[SessionState.CHAT_HISTORY] = history

    @staticmethod
    def clear_chat_history():
        """Clear chat history."""
        st.session_state[SessionState.CHAT_HISTORY] = []
        logger.info("Cleared chat history")

    @staticmethod
    def is_conversation_active() -> bool:
        """Check if conversation is active."""
        return st.session_state.get(SessionState.CONVERSATION_ACTIVE, False)

    @staticmethod
    def start_conversation(conversation_id: str):
        """
        Start a new conversation.

        Args:
            conversation_id: ID of the conversation
        """
        st.session_state[SessionState.CONVERSATION_ACTIVE] = True
        st.session_state[SessionState.CONVERSATION_ID] = conversation_id
        SessionState.clear_chat_history()
        logger.info(f"Started conversation: {conversation_id}")

    @staticmethod
    def end_conversation():
        """End current conversation."""
        st.session_state[SessionState.CONVERSATION_ACTIVE] = False
        st.session_state[SessionState.CONVERSATION_ID] = None
        logger.info("Ended conversation")

    @staticmethod
    def get_current_conversation_id() -> Optional[str]:
        """Get current conversation ID."""
        return st.session_state.get(SessionState.CONVERSATION_ID)

    @staticmethod
    def get_algorithm_weights() -> Dict[str, float]:
        """Get algorithm weights."""
        return st.session_state.get(SessionState.WEIGHTS, {
            'wR': 1.5, 'wv': 1.0, 'wa': 3.0, 'b': -2.5
        })

    @staticmethod
    def set_algorithm_weights(weights: Dict[str, float]):
        """Set algorithm weights."""
        st.session_state[SessionState.WEIGHTS] = weights

    @staticmethod
    def get_vsafe_text() -> str:
        """Get VSAFE text."""
        return st.session_state.get(
            SessionState.VSAFE_TEXT,
            "I prioritize safety, accuracy, and ethical behavior in all responses."
        )

    @staticmethod
    def set_vsafe_text(text: str):
        """Set VSAFE text."""
        st.session_state[SessionState.VSAFE_TEXT] = text

    @staticmethod
    def get_vsafe_vector():
        """Get VSAFE vector."""
        return st.session_state.get(SessionState.VSAFE_VECTOR)

    @staticmethod
    def set_vsafe_vector(vector):
        """Set VSAFE vector."""
        st.session_state[SessionState.VSAFE_VECTOR] = vector

    @staticmethod
    def get_alert_threshold() -> float:
        """Get alert threshold."""
        return st.session_state.get(SessionState.ALERT_THRESHOLD, 0.8)

    @staticmethod
    def set_alert_threshold(threshold: float):
        """Set alert threshold."""
        st.session_state[SessionState.ALERT_THRESHOLD] = threshold

    @staticmethod
    def get_selected_model() -> str:
        """Get selected model key."""
        return st.session_state.get(SessionState.SELECTED_MODEL, "gpt-3.5")

    @staticmethod
    def set_selected_model(model_key: str):
        """Set selected model."""
        st.session_state[SessionState.SELECTED_MODEL] = model_key

    @staticmethod
    def is_using_mock() -> bool:
        """Check if using mock client."""
        return st.session_state.get(SessionState.USE_MOCK, False)

    @staticmethod
    def set_use_mock(use_mock: bool):
        """Set whether to use mock client."""
        st.session_state[SessionState.USE_MOCK] = use_mock

    @staticmethod
    def get_system_prompt() -> str:
        """Get system prompt."""
        return st.session_state.get(
            SessionState.SYSTEM_PROMPT,
            "You are a helpful AI assistant. Please provide safe, accurate, and ethical responses."
        )

    @staticmethod
    def set_system_prompt(prompt: str):
        """Set system prompt."""
        st.session_state[SessionState.SYSTEM_PROMPT] = prompt

    @staticmethod
    def get_export_directory() -> str:
        """Get export directory."""
        return st.session_state.get(SessionState.EXPORT_DIR, "exports")

    @staticmethod
    def set_export_directory(directory: str):
        """Set export directory."""
        st.session_state[SessionState.EXPORT_DIR] = directory

    @staticmethod
    def reset():
        """Reset entire session state (for new session)."""
        # Clear all keys
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Reinitialize
        SessionState.initialize()
        logger.info("Session state reset")

    @staticmethod
    def export_state() -> Dict:
        """
        Export session state for debugging or persistence.

        Returns:
            Dictionary with serializable state data
        """
        return {
            'selected_model': SessionState.get_selected_model(),
            'use_mock': SessionState.is_using_mock(),
            'conversation_active': SessionState.is_conversation_active(),
            'conversation_id': SessionState.get_current_conversation_id(),
            'chat_messages': len(SessionState.get_chat_history()),
            'algorithm_weights': SessionState.get_algorithm_weights(),
            'vsafe_text': SessionState.get_vsafe_text(),
            'alert_threshold': SessionState.get_alert_threshold(),
            'export_timestamp': datetime.now().isoformat()
        }
