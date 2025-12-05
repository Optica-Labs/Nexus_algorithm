#!/usr/bin/env python3
"""
LLM API Client - Handles communication with multiple LLM endpoints.

Supports:
- OpenAI GPT-3.5, GPT-4
- Anthropic Claude Sonnet 3
- Mistral Large
"""

import os
import json
import requests
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported LLM providers."""
    AWS_LAMBDA = "aws_lambda"  # AWS Lambda endpoints (no API key needed)
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"


@dataclass
class LLMConfig:
    """Configuration for an LLM endpoint."""
    name: str
    provider: ModelProvider
    model_id: str
    endpoint_url: Optional[str] = None  # AWS Lambda endpoint
    api_key_env: Optional[str] = None    # For direct API access
    max_tokens: int = 1024
    temperature: float = 0.7


# AWS Lambda Endpoints - Using your existing infrastructure
AWS_LAMBDA_ENDPOINTS = {
    "gpt-3.5": "https://kv854u79y7.execute-api.us-east-1.amazonaws.com/prod/chat",
    "gpt-4": "https://1d4qnutnqc.execute-api.us-east-1.amazonaws.com/prod/chat",
    "claude": "https://6z5nnwuyyj.execute-api.us-east-1.amazonaws.com/prod/chat",
    "mistral": "https://q1gqmwwmsf.execute-api.us-east-1.amazonaws.com/prod/chat"
}

# Model configurations - AWS Lambda endpoints (recommended)
MODEL_CONFIGS = {
    "gpt-3.5": LLMConfig(
        name="GPT-3.5",
        provider=ModelProvider.AWS_LAMBDA,
        model_id="gpt-3.5-turbo",
        endpoint_url=AWS_LAMBDA_ENDPOINTS["gpt-3.5"],
        max_tokens=1024,
        temperature=0.7
    ),
    "gpt-4": LLMConfig(
        name="GPT-4",
        provider=ModelProvider.AWS_LAMBDA,
        model_id="gpt-4",
        endpoint_url=AWS_LAMBDA_ENDPOINTS["gpt-4"],
        max_tokens=1024,
        temperature=0.7
    ),
    "claude-sonnet-3": LLMConfig(
        name="Claude 3 Sonnet",
        provider=ModelProvider.AWS_LAMBDA,
        model_id="claude-3-sonnet-20240229",
        endpoint_url=AWS_LAMBDA_ENDPOINTS["claude"],
        max_tokens=1024,
        temperature=0.7
    ),
    "mistral-large": LLMConfig(
        name="Mistral Large",
        provider=ModelProvider.AWS_LAMBDA,
        model_id="mistral-large-latest",
        endpoint_url=AWS_LAMBDA_ENDPOINTS["mistral"],
        max_tokens=1024,
        temperature=0.7
    )
}


class LLMAPIClient:
    """
    Client for communicating with various LLM APIs.

    Handles API calls, message formatting, and error handling
    for multiple LLM providers.
    """

    def __init__(self, model_key: str):
        """
        Initialize API client for a specific model.

        Args:
            model_key: Key from MODEL_CONFIGS (e.g., "gpt-3.5")
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}")

        self.model_key = model_key
        self.config = MODEL_CONFIGS[model_key]

        # Get API key from environment (only for direct API providers)
        self.api_key = None
        if self.config.api_key_env:
            self.api_key = os.getenv(self.config.api_key_env)
            if not self.api_key and self.config.provider != ModelProvider.AWS_LAMBDA:
                logger.warning(f"API key not found for {self.config.name}. Set {self.config.api_key_env}")

        # Conversation history
        self.messages: List[Dict] = []

        logger.info(f"Initialized API client for {self.config.name} (provider: {self.config.provider.value})")

    def add_system_message(self, content: str):
        """
        Add a system message to the conversation.

        Args:
            content: System message content
        """
        self.messages.append({
            "role": "system",
            "content": content
        })

    def send_message(
        self,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, bool]:
        """
        Send a user message and get model response.

        Args:
            user_message: User's message text
            temperature: Optional override for temperature
            max_tokens: Optional override for max tokens

        Returns:
            Tuple of (response_text, success)
        """
        # Add user message to history
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        # Use defaults if not provided
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        try:
            # Route to appropriate provider
            if self.config.provider == ModelProvider.AWS_LAMBDA:
                response = self._call_aws_lambda(temp, max_tok)
            elif self.config.provider == ModelProvider.OPENAI:
                response = self._call_openai(temp, max_tok)
            elif self.config.provider == ModelProvider.ANTHROPIC:
                response = self._call_anthropic(temp, max_tok)
            elif self.config.provider == ModelProvider.MISTRAL:
                response = self._call_mistral(temp, max_tok)
            else:
                return "Error: Unsupported provider", False

            # Add response to history
            self.messages.append({
                "role": "assistant",
                "content": response
            })

            logger.info(f"Received response from {self.config.name}")
            return response, True

        except Exception as e:
            logger.error(f"Error calling {self.config.name}: {e}")
            return f"Error: {str(e)}", False

    def _call_aws_lambda(self, temperature: float, max_tokens: int) -> str:
        """Call AWS Lambda API Gateway endpoint."""
        if not self.config.endpoint_url:
            raise ValueError("AWS Lambda endpoint URL not configured")

        headers = {
            "Content-Type": "application/json"
        }

        # Build conversation history for the API
        conversation = []
        for msg in self.messages:
            if msg['role'] == 'system':
                # Some models might not support system messages via Lambda
                # Prepend to first user message instead
                if conversation and conversation[-1]['role'] == 'user':
                    conversation[-1]['content'] = f"{msg['content']}\n\n{conversation[-1]['content']}"
                else:
                    conversation.append({
                        'role': 'user',
                        'content': msg['content']
                    })
            else:
                conversation.append(msg)

        # Format payload for AWS Lambda endpoint
        payload = {
            "conversation": conversation,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        logger.info(f"Calling AWS Lambda endpoint: {self.config.endpoint_url}")
        response = requests.post(
            self.config.endpoint_url,
            headers=headers,
            json=payload,
            timeout=60  # Lambda might take longer
        )
        response.raise_for_status()

        data = response.json()

        # Extract response based on Lambda return format
        # Adjust this based on your Lambda function's response structure
        if 'response' in data:
            return data['response']
        elif 'message' in data:
            return data['message']
        elif 'body' in data:
            # If Lambda returns API Gateway format
            body = json.loads(data['body']) if isinstance(data['body'], str) else data['body']
            return body.get('response', body.get('message', str(body)))
        else:
            logger.warning(f"Unexpected Lambda response format: {data}")
            return str(data)

    def _call_openai(self, temperature: float, max_tokens: int) -> str:
        """Call OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model_id,
            "messages": self.messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        return data['choices'][0]['message']['content']

    def _call_anthropic(self, temperature: float, max_tokens: int) -> str:
        """Call Anthropic Claude API."""
        url = "https://api.anthropic.com/v1/messages"

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        # Extract system message if present
        system_msg = None
        messages = []
        for msg in self.messages:
            if msg['role'] == 'system':
                system_msg = msg['content']
            else:
                messages.append(msg)

        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if system_msg:
            payload["system"] = system_msg

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        return data['content'][0]['text']

    def _call_mistral(self, temperature: float, max_tokens: int) -> str:
        """Call Mistral API."""
        url = "https://api.mistral.ai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model_id,
            "messages": self.messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        return data['choices'][0]['message']['content']

    def get_conversation_history(self) -> List[Dict]:
        """
        Get full conversation history.

        Returns:
            List of message dictionaries
        """
        return self.messages.copy()

    def clear_history(self):
        """Clear conversation history."""
        self.messages = []
        logger.info(f"Cleared conversation history for {self.config.name}")

    def export_conversation(self, filepath: str):
        """
        Export conversation to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        data = {
            "model": self.model_key,
            "model_name": self.config.name,
            "messages": self.messages
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported conversation to {filepath}")

    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """
        Get available models with their display names.

        Returns:
            Dictionary of {model_key: display_name}
        """
        return {key: config.name for key, config in MODEL_CONFIGS.items()}

    @staticmethod
    def is_api_key_configured(model_key: str) -> bool:
        """
        Check if API key is configured for a model.
        AWS Lambda endpoints don't require API keys.

        Args:
            model_key: Model key to check

        Returns:
            True if API key is set or if using AWS Lambda
        """
        if model_key not in MODEL_CONFIGS:
            return False

        config = MODEL_CONFIGS[model_key]

        # AWS Lambda endpoints don't need API keys
        if config.provider == ModelProvider.AWS_LAMBDA:
            return True

        # Other providers need API keys
        if config.api_key_env:
            return os.getenv(config.api_key_env) is not None

        return False


class MockLLMClient:
    """
    Mock LLM client for testing without API keys.

    Returns canned responses for testing the dashboard.
    """

    def __init__(self, model_key: str):
        """Initialize mock client."""
        self.model_key = model_key
        self.config = MODEL_CONFIGS.get(model_key)
        self.messages: List[Dict] = []
        self.response_count = 0

        logger.info(f"Initialized MOCK client for {self.config.name if self.config else model_key}")

    def add_system_message(self, content: str):
        """Add system message (mock)."""
        self.messages.append({"role": "system", "content": content})

    def send_message(
        self,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, bool]:
        """Send message and get mock response."""
        self.messages.append({"role": "user", "content": user_message})

        # Generate mock response
        self.response_count += 1

        # Simulate different response types
        if self.response_count == 1:
            response = "Hello! I'm a helpful AI assistant. How can I help you today?"
        elif "dangerous" in user_message.lower() or "harmful" in user_message.lower():
            response = "I understand your question, but I need to be careful about providing information that could be harmful."
        elif self.response_count % 3 == 0:
            response = "I apologize, but I cannot assist with that request as it may violate my safety guidelines."
        else:
            response = f"This is a mock response #{self.response_count} from {self.config.name if self.config else 'Mock Model'}. In a real scenario, this would be an actual LLM response."

        self.messages.append({"role": "assistant", "content": response})

        logger.info(f"Generated mock response #{self.response_count}")
        return response, True

    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.messages.copy()

    def clear_history(self):
        """Clear history."""
        self.messages = []
        self.response_count = 0

    def export_conversation(self, filepath: str):
        """Export conversation."""
        data = {
            "model": self.model_key,
            "model_name": f"MOCK: {self.config.name if self.config else self.model_key}",
            "messages": self.messages
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def create_llm_client(model_key: str, use_mock: bool = False) -> LLMAPIClient:
    """
    Factory function to create appropriate LLM client.

    Args:
        model_key: Model key from MODEL_CONFIGS
        use_mock: If True, return mock client (for testing)

    Returns:
        LLMAPIClient or MockLLMClient instance
    """
    if use_mock or not LLMAPIClient.is_api_key_configured(model_key):
        if not use_mock:
            logger.warning(f"API key not configured for {model_key}, using mock client")
        return MockLLMClient(model_key)

    return LLMAPIClient(model_key)
