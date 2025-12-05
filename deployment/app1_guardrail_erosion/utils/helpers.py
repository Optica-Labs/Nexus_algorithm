#!/usr/bin/env python3
"""
Helper utilities for Guardrail Erosion Analyzer.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import csv
import io

logger = logging.getLogger(__name__)


def parse_manual_input(text: str) -> List[Dict[str, str]]:
    """
    Parse manually entered conversation text into turns.

    Expected format:
    User: message 1
    Model: response 1
    User: message 2
    Model: response 2

    Args:
        text: Raw text input

    Returns:
        List of {"speaker": str, "message": str} dictionaries
    """
    turns = []
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to split by common separators
        for separator in [':', '|', '-']:
            if separator in line:
                parts = line.split(separator, 1)
                if len(parts) == 2:
                    speaker = parts[0].strip().lower()
                    message = parts[1].strip()

                    # Normalize speaker names
                    if speaker in ['user', 'human', 'u']:
                        speaker = 'user'
                    elif speaker in ['model', 'llm', 'ai', 'assistant', 'm', 'bot']:
                        speaker = 'llm'

                    if message:
                        turns.append({"speaker": speaker, "message": message})
                    break

    return turns


def json_to_turns(json_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Extract user and model turns from JSON conversation data.

    Args:
        json_data: Parsed JSON conversation

    Returns:
        Tuple of (user_messages, model_messages)
    """
    conversation = json_data.get('conversation', [])

    user_messages = []
    model_messages = []

    for turn in conversation:
        speaker = turn.get('speaker', '').lower()
        message = turn.get('message', '')

        if speaker in ['user', 'human']:
            user_messages.append(message)
        elif speaker in ['llm', 'model', 'ai', 'assistant']:
            model_messages.append(message)

    return user_messages, model_messages


def csv_to_turns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Extract user and model turns from CSV DataFrame.

    Args:
        df: DataFrame with 'speaker' and 'message' columns

    Returns:
        Tuple of (user_messages, model_messages)
    """
    user_messages = []
    model_messages = []

    for _, row in df.iterrows():
        speaker = str(row['speaker']).lower()
        message = str(row['message'])

        if speaker in ['user', 'human']:
            user_messages.append(message)
        elif speaker in ['llm', 'model', 'ai', 'assistant']:
            model_messages.append(message)

    return user_messages, model_messages


def export_metrics_to_csv(df: pd.DataFrame) -> bytes:
    """
    Export metrics DataFrame to CSV bytes.

    Args:
        df: Metrics DataFrame

    Returns:
        CSV as bytes
    """
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue().encode('utf-8')


def export_metrics_to_json(df: pd.DataFrame, stats: Dict[str, Any]) -> bytes:
    """
    Export metrics and statistics to JSON bytes.

    Args:
        df: Metrics DataFrame
        stats: Statistics dictionary

    Returns:
        JSON as bytes
    """
    # Convert DataFrame to records and handle numpy types
    metrics_records = df.to_dict(orient='records')
    for record in metrics_records:
        for key, value in record.items():
            # Convert numpy types to native Python types
            if hasattr(value, 'item'):
                record[key] = value.item()

    # Convert statistics to JSON-serializable types
    json_safe_stats = {}
    for key, value in stats.items():
        if isinstance(value, (bool, int, float, str, type(None))):
            json_safe_stats[key] = value
        elif hasattr(value, 'item'):  # numpy types
            json_safe_stats[key] = value.item()
        elif isinstance(value, (np.bool_, np.integer, np.floating)):
            json_safe_stats[key] = value.item()
        else:
            json_safe_stats[key] = str(value)

    data = {
        'metrics': metrics_records,
        'statistics': json_safe_stats,
        'export_timestamp': pd.Timestamp.now().isoformat()
    }
    return json.dumps(data, indent=2).encode('utf-8')


def create_export_filename(base_name: str, extension: str, include_timestamp: bool = True) -> str:
    """
    Create standardized export filename.

    Args:
        base_name: Base name for file
        extension: File extension (without dot)
        include_timestamp: Whether to include timestamp

    Returns:
        Filename string
    """
    if include_timestamp:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        return f"{base_name}_{timestamp}.{extension}"
    else:
        return f"{base_name}.{extension}"


def validate_vector_coordinates(text: str) -> Optional[np.ndarray]:
    """
    Validate and parse 2D vector coordinates from text.

    Expected formats:
    - "x, y"
    - "[x, y]"
    - "x y"

    Args:
        text: Text representation of coordinates

    Returns:
        numpy array [x, y] or None if invalid
    """
    try:
        # Remove brackets and split
        text = text.strip().replace('[', '').replace(']', '')
        parts = [p.strip() for p in text.replace(',', ' ').split()]

        if len(parts) != 2:
            return None

        x = float(parts[0])
        y = float(parts[1])

        return np.array([x, y])
    except (ValueError, IndexError):
        return None


def format_statistics_display(stats: Dict[str, Any]) -> str:
    """
    Format statistics dictionary for display.

    Args:
        stats: Statistics dictionary

    Returns:
        Formatted string
    """
    lines = []
    lines.append("### Conversation Statistics")
    lines.append("")
    lines.append(f"**Total Turns:** {stats.get('total_turns', 0)}")
    lines.append(f"**Peak Risk Severity:** {stats.get('peak_risk_severity', 0):.3f} (Turn {stats.get('peak_risk_turn', 0)})")
    lines.append(f"**Peak Likelihood:** {stats.get('peak_likelihood', 0):.3f} (Turn {stats.get('peak_likelihood_turn', 0)})")
    lines.append(f"**Max Erosion:** {stats.get('max_erosion', 0):.3f}")
    lines.append(f"**Average Likelihood:** {stats.get('avg_likelihood', 0):.3f}")

    if 'final_rho' in stats:
        lines.append("")
        lines.append("### Robustness Analysis")
        rho_status = "✅ ROBUST" if stats.get('is_robust', False) else "❌ FRAGILE"
        lines.append(f"**Status:** {rho_status}")
        lines.append(f"**Final RHO:** {stats.get('final_rho', 0):.3f}")
        if 'amplified_risk' in stats and stats['amplified_risk'] > 0:
            lines.append(f"**Amplified Risk:** {stats.get('amplified_risk', 0):.3f}")

    return "\n".join(lines)


def chunk_messages_for_api(messages: List[str], chunk_size: int = 10) -> List[List[str]]:
    """
    Split messages into chunks for batch API processing.

    Args:
        messages: List of messages
        chunk_size: Maximum chunk size

    Returns:
        List of message chunks
    """
    return [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]
