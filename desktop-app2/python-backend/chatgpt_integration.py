#!/usr/bin/env python3
"""
ChatGPT Integration Client for Desktop App4

Provides ChatGPT client that is compatible with App4's LLM client interface.
This allows seamless integration with the existing App4 pipeline.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

logger = logging.getLogger(__name__)


class ChatGPTClient:
    """
    ChatGPT client compatible with App4's LLM client interface.

    This client wraps the OpenAI API to match the interface expected by App4,
    allowing seamless integration with the existing pipeline orchestrator.
    """

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize ChatGPT client.

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-3.5-turbo, gpt-4, etc.)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        if not api_key or api_key == "mock":
            raise ValueError("Valid OpenAI API key required")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.system_message = "You are a helpful AI assistant."

        logger.info(f"ChatGPT client initialized with model: {model}")

    def add_system_message(self, message: str):
        """
        Set system message for the conversation.

        Args:
            message: System prompt text
        """
        self.system_message = message
        logger.info("System message updated")

    def send_message(
        self,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Tuple[str, bool]:
        """
        Send a message to ChatGPT and get response.

        This matches the App4 LLM client interface.

        Args:
            user_message: User's input message
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (response_text, success_boolean)
        """
        try:
            # Build messages list
            messages = [{"role": "system", "content": self.system_message}]

            # Add conversation history
            messages.extend(self.conversation_history)

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            # Call OpenAI API
            logger.info(f"Sending message to {self.model}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            assistant_message = response.choices[0].message.content

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            logger.info(f"Received response ({len(assistant_message)} chars)")
            logger.info(f"Token usage: {response.usage.total_tokens} total")

            return assistant_message, True

        except Exception as e:
            logger.error(f"Error calling ChatGPT: {e}")
            return f"Error: {str(e)}", False

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def export_conversation(self, filepath: str):
        """
        Export conversation to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        export_data = {
            'metadata': {
                'model': self.model,
                'system_message': self.system_message,
                'export_time': datetime.now().isoformat()
            },
            'conversation': self.conversation_history
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Conversation exported to {filepath}")

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history.

        Returns:
            List of message dictionaries
        """
        return self.conversation_history.copy()

    def get_model_info(self) -> Dict[str, str]:
        """
        Get model information.

        Returns:
            Dictionary with model details
        """
        return {
            'model': self.model,
            'provider': 'OpenAI',
            'type': 'ChatGPT'
        }


def check_api_key(api_key: str) -> bool:
    """
    Check if OpenAI API key is valid.

    Args:
        api_key: API key to check

    Returns:
        True if valid, False otherwise
    """
    if not api_key or not api_key.startswith('sk-'):
        return False

    try:
        client = OpenAI(api_key=api_key)
        # Try a minimal API call to verify the key
        client.models.list()
        return True
    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        return False


def create_chatgpt_client(
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo"
) -> ChatGPTClient:
    """
    Create a ChatGPT client instance.

    Args:
        api_key: OpenAI API key (if None, reads from environment)
        model: Model name

    Returns:
        ChatGPTClient instance
    """
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY', '')

    if not api_key:
        raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")

    return ChatGPTClient(api_key=api_key, model=model)
