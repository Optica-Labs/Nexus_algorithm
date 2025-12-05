#!/usr/bin/env python3
"""
Input validation utilities for deployment applications.
"""

import json
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConversationValidator:
    """Validates conversation data in various formats."""

    @staticmethod
    def validate_json(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate JSON conversation format.

        Expected format:
        {
            "conversation": [
                {"turn": 1, "speaker": "user", "message": "..."},
                {"turn": 2, "speaker": "llm", "message": "..."}
            ]
        }

        Args:
            data: Parsed JSON data

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "JSON must be a dictionary"

        if "conversation" not in data:
            return False, "JSON must contain 'conversation' key"

        conversation = data["conversation"]
        if not isinstance(conversation, list):
            return False, "'conversation' must be a list"

        if len(conversation) == 0:
            return False, "Conversation cannot be empty"

        # Validate each turn
        for i, turn in enumerate(conversation):
            if not isinstance(turn, dict):
                return False, f"Turn {i+1} must be a dictionary"

            # Check required fields
            if "speaker" not in turn:
                return False, f"Turn {i+1} missing 'speaker' field"

            if "message" not in turn:
                return False, f"Turn {i+1} missing 'message' field"

            # Validate speaker
            speaker = turn["speaker"].lower()
            if speaker not in ["user", "llm", "model", "ai", "assistant"]:
                return False, f"Turn {i+1} has invalid speaker: {turn['speaker']}"

            # Validate message
            if not isinstance(turn["message"], str):
                return False, f"Turn {i+1} message must be a string"

            if len(turn["message"].strip()) == 0:
                return False, f"Turn {i+1} has empty message"

        return True, None

    @staticmethod
    def validate_csv(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate CSV conversation format.

        Expected columns: turn, speaker, message

        Args:
            df: Pandas DataFrame

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_columns = ['turn', 'speaker', 'message']

        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"CSV missing required columns: {missing_cols}"

        # Check if DataFrame is empty
        if len(df) == 0:
            return False, "CSV file is empty"

        # Validate speaker column
        valid_speakers = ['user', 'llm', 'model', 'ai', 'assistant']
        df['speaker_lower'] = df['speaker'].str.lower()
        invalid_speakers = df[~df['speaker_lower'].isin(valid_speakers)]

        if len(invalid_speakers) > 0:
            return False, f"Invalid speakers found: {invalid_speakers['speaker'].unique().tolist()}"

        # Check for empty messages
        empty_messages = df[df['message'].isna() | (df['message'].str.strip() == '')]
        if len(empty_messages) > 0:
            return False, f"Empty messages found in rows: {empty_messages.index.tolist()}"

        return True, None

    @staticmethod
    def validate_manual_input(turns: List[Dict[str, str]]) -> Tuple[bool, Optional[str]]:
        """
        Validate manually entered conversation turns.

        Args:
            turns: List of {"speaker": str, "message": str} dicts

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(turns, list):
            return False, "Turns must be a list"

        if len(turns) == 0:
            return False, "Conversation cannot be empty"

        valid_speakers = ['user', 'llm', 'model', 'ai', 'assistant']

        for i, turn in enumerate(turns):
            if not isinstance(turn, dict):
                return False, f"Turn {i+1} must be a dictionary"

            if "speaker" not in turn or "message" not in turn:
                return False, f"Turn {i+1} missing required fields (speaker, message)"

            if turn["speaker"].lower() not in valid_speakers:
                return False, f"Turn {i+1} has invalid speaker: {turn['speaker']}"

            if not turn["message"] or len(turn["message"].strip()) == 0:
                return False, f"Turn {i+1} has empty message"

        return True, None


class ParameterValidator:
    """Validates algorithm parameters and thresholds."""

    @staticmethod
    def validate_weights(wR: float, wv: float, wa: float, b: float) -> Tuple[bool, Optional[str]]:
        """
        Validate algorithm weights.

        Args:
            wR: Weight for Risk Severity
            wv: Weight for Risk Rate
            wa: Weight for Erosion
            b: Bias term

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check types
        if not all(isinstance(x, (int, float)) for x in [wR, wv, wa, b]):
            return False, "All weights must be numeric"

        # Weights should generally be positive (bias can be negative)
        if wR <= 0:
            return False, "wR (Risk Severity weight) must be positive"
        if wv <= 0:
            return False, "wv (Risk Rate weight) must be positive"
        if wa <= 0:
            return False, "wa (Erosion weight) must be positive"

        # Reasonable range checks (optional, can be adjusted)
        if not -10 <= b <= 10:
            return False, "Bias (b) should be in range [-10, 10]"

        if not 0 < wR <= 10:
            return False, "wR should be in range (0, 10]"

        if not 0 < wv <= 10:
            return False, "wv should be in range (0, 10]"

        if not 0 < wa <= 10:
            return False, "wa should be in range (0, 10]"

        return True, None

    @staticmethod
    def validate_likelihood_threshold(threshold: float) -> Tuple[bool, Optional[str]]:
        """
        Validate likelihood alert threshold.

        Args:
            threshold: Likelihood threshold (0-1)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(threshold, (int, float)):
            return False, "Likelihood threshold must be numeric"

        if not 0 <= threshold <= 1:
            return False, "Likelihood threshold must be between 0 and 1"

        return True, None

    @staticmethod
    def validate_epsilon(epsilon: float) -> Tuple[bool, Optional[str]]:
        """
        Validate epsilon value for RHO calculation.

        Args:
            epsilon: Epsilon value for division-by-zero protection

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(epsilon, (int, float)):
            return False, "Epsilon must be numeric"

        if epsilon <= 0:
            return False, "Epsilon must be positive"

        if epsilon > 1.0:
            return False, "Epsilon should typically be <= 1.0"

        return True, None

    @staticmethod
    def validate_phi_threshold(threshold: float) -> Tuple[bool, Optional[str]]:
        """
        Validate PHI score threshold.

        Args:
            threshold: PHI threshold

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(threshold, (int, float)):
            return False, "PHI threshold must be numeric"

        if threshold < 0:
            return False, "PHI threshold must be non-negative"

        if threshold > 1.0:
            return False, "PHI threshold should typically be <= 1.0"

        return True, None


class FileValidator:
    """Validates uploaded files."""

    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate file extension.

        Args:
            filename: Name of the file
            allowed_extensions: List of allowed extensions (e.g., ['.json', '.csv'])

        Returns:
            Tuple of (is_valid, error_message)
        """
        import os

        _, ext = os.path.splitext(filename.lower())

        if ext not in allowed_extensions:
            return False, f"File type {ext} not allowed. Allowed types: {allowed_extensions}"

        return True, None

    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int = 10) -> Tuple[bool, Optional[str]]:
        """
        Validate file size.

        Args:
            file_size: Size of file in bytes
            max_size_mb: Maximum allowed size in MB

        Returns:
            Tuple of (is_valid, error_message)
        """
        max_size_bytes = max_size_mb * 1024 * 1024

        if file_size > max_size_bytes:
            return False, f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds maximum ({max_size_mb} MB)"

        return True, None


def parse_conversation_json(json_str: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Parse and validate JSON conversation string.

    Args:
        json_str: JSON string

    Returns:
        Tuple of (parsed_data, error_message)
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON format: {str(e)}"

    is_valid, error = ConversationValidator.validate_json(data)
    if not is_valid:
        return None, error

    return data, None


def parse_conversation_csv(csv_content) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parse and validate CSV conversation content.

    Args:
        csv_content: CSV file content or path

    Returns:
        Tuple of (dataframe, error_message)
    """
    try:
        if isinstance(csv_content, str):
            # It's a file path
            df = pd.read_csv(csv_content)
        else:
            # It's file content
            df = pd.read_csv(csv_content)
    except Exception as e:
        return None, f"Failed to parse CSV: {str(e)}"

    is_valid, error = ConversationValidator.validate_csv(df)
    if not is_valid:
        return None, error

    return df, None
