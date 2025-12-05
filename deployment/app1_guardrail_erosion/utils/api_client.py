#!/usr/bin/env python3
"""
Simple API Client for App 1 - LLM Integration

Handles communication with AWS Lambda endpoints for real-time
conversation generation and analysis.
"""

import requests
import json
import logging
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)

# AWS Lambda Endpoints
AWS_LAMBDA_ENDPOINTS = {
    "gpt-3.5": "https://kv854u79y7.execute-api.us-east-1.amazonaws.com/prod/chat",
    "gpt-4": "https://1d4qnutnqc.execute-api.us-east-1.amazonaws.com/prod/chat",
    "claude": "https://6z5nnwuyyj.execute-api.us-east-1.amazonaws.com/prod/chat",
    "mistral": "https://q1gqmwwmsf.execute-api.us-east-1.amazonaws.com/prod/chat"
}

# Model display names
MODEL_NAMES = {
    "gpt-3.5": "GPT-3.5 Turbo",
    "gpt-4": "GPT-4",
    "claude": "Claude 3 Sonnet",
    "mistral": "Mistral Large"
}


class SimpleLLMClient:
    """
    Simple LLM client for App 1 API integration.

    Maintains conversation history and handles API calls
    to AWS Lambda endpoints.
    """

    def __init__(self, model_key: str):
        """
        Initialize client.

        Args:
            model_key: Model identifier (gpt-3.5, gpt-4, claude, mistral)
        """
        if model_key not in AWS_LAMBDA_ENDPOINTS:
            raise ValueError(f"Unknown model: {model_key}")

        self.model_key = model_key
        self.model_name = MODEL_NAMES[model_key]
        self.endpoint = AWS_LAMBDA_ENDPOINTS[model_key]
        self.conversation_history: List[Dict[str, str]] = []

        logger.info(f"Initialized LLM client for {self.model_name}")

    def send_message(
        self,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Tuple[Optional[str], bool, str]:
        """
        Send a message and get response.

        Args:
            user_message: User's message
            temperature: Temperature for sampling (0.0-2.0)
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (response_text, success, error_message)
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Prepare payload
        payload = {
            "conversation": self.conversation_history.copy(),
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            logger.info(f"Sending message to {self.model_name}...")

            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )

            response.raise_for_status()
            data = response.json()

            # Extract response
            model_response = self._extract_response(data)

            if model_response:
                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": model_response
                })

                logger.info(f"Received response from {self.model_name}")
                return model_response, True, ""
            else:
                error = "Could not extract response from API"
                logger.error(error)
                return None, False, error

        except requests.exceptions.Timeout:
            error = f"Request timed out (60s)"
            logger.error(error)
            return None, False, error

        except requests.exceptions.HTTPError as e:
            error = f"HTTP error {response.status_code}: {response.text[:200]}"
            logger.error(error)
            return None, False, error

        except Exception as e:
            error = f"Error: {str(e)}"
            logger.error(error)
            return None, False, error

    def _extract_response(self, data: Dict) -> Optional[str]:
        """
        Extract response from Lambda response.

        Args:
            data: Response JSON from Lambda

        Returns:
            Extracted response text or None
        """
        # Try different response formats
        if 'response' in data:
            return data['response']
        elif 'message' in data:
            return data['message']
        elif 'body' in data:
            body = json.loads(data['body']) if isinstance(data['body'], str) else data['body']
            return body.get('response', body.get('message'))
        else:
            logger.warning(f"Unexpected response format: {list(data.keys())}")
            return str(data)

    def get_conversation_turns(self) -> List[Tuple[str, str]]:
        """
        Get conversation as list of (user, model) tuples.

        Returns:
            List of conversation turns
        """
        turns = []
        for i in range(0, len(self.conversation_history) - 1, 2):
            if i + 1 < len(self.conversation_history):
                user_msg = self.conversation_history[i]['content']
                model_msg = self.conversation_history[i + 1]['content']
                turns.append((user_msg, model_msg))
        return turns

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Cleared conversation history")

    def get_turn_count(self) -> int:
        """Get number of complete turns."""
        return len(self.conversation_history) // 2

    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """
        Get available models.

        Returns:
            Dictionary of {model_key: display_name}
        """
        return MODEL_NAMES.copy()


class MockLLMClient:
    """
    Mock LLM client for testing without API access.

    Returns canned responses based on input patterns.
    """

    def __init__(self, model_key: str):
        """Initialize mock client."""
        self.model_key = model_key
        self.model_name = MODEL_NAMES.get(model_key, "Mock Model")
        self.conversation_history: List[Dict[str, str]] = []
        self.turn_count = 0

        logger.info(f"Initialized MOCK client for {self.model_name}")

    def send_message(
        self,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Tuple[Optional[str], bool, str]:
        """Send message and get mock response."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        self.turn_count += 1

        # Generate response based on keywords
        user_lower = user_message.lower()

        if self.turn_count == 1:
            response = "Hello! I'm a helpful AI assistant. How can I help you today?"
        elif any(word in user_lower for word in ['dangerous', 'harmful', 'illegal', 'hack', 'weapon', 'malware', 'virus']):
            # Simulate safe refusal
            response = "I understand your question, but I cannot assist with activities that could be harmful, dangerous, or illegal. Is there something else I can help you with?"
        elif any(word in user_lower for word in ['bypass', 'security', 'password', 'access']):
            # Simulate another refusal
            response = "I cannot provide assistance with bypassing security measures or unauthorized access. If you need help with legitimate security concerns, I recommend consulting with cybersecurity professionals."
        elif 'how' in user_lower or '?' in user_message:
            response = f"That's an interesting question. Let me provide a helpful and safe response. [Mock response from {self.model_name}]"
        else:
            response = f"I understand. Let me assist you with that in a safe and responsible manner. [Mock response {self.turn_count} from {self.model_name}]"

        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        logger.info(f"Generated mock response #{self.turn_count}")
        return response, True, ""

    def get_conversation_turns(self) -> List[Tuple[str, str]]:
        """Get conversation turns."""
        turns = []
        for i in range(0, len(self.conversation_history) - 1, 2):
            if i + 1 < len(self.conversation_history):
                user_msg = self.conversation_history[i]['content']
                model_msg = self.conversation_history[i + 1]['content']
                turns.append((user_msg, model_msg))
        return turns

    def clear_history(self):
        """Clear history."""
        self.conversation_history = []
        self.turn_count = 0

    def get_turn_count(self) -> int:
        """Get turn count."""
        return self.turn_count

    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """Get available models."""
        return MODEL_NAMES.copy()


def create_llm_client(model_key: str, use_mock: bool = False) -> SimpleLLMClient:
    """
    Factory function to create LLM client.

    Args:
        model_key: Model identifier
        use_mock: If True, return mock client

    Returns:
        LLM client instance
    """
    if use_mock:
        return MockLLMClient(model_key)
    else:
        return SimpleLLMClient(model_key)
