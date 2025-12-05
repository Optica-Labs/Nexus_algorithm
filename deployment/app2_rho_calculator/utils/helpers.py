#!/usr/bin/env python3
"""
Helper utilities for RHO Calculator.
"""

import pandas as pd
import numpy as np
import json
import io
import zipfile
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_metrics_from_csv(file_content) -> pd.DataFrame:
    """
    Load metrics DataFrame from CSV file.

    Args:
        file_content: CSV file content or path

    Returns:
        Metrics DataFrame
    """
    try:
        if isinstance(file_content, str):
            df = pd.read_csv(file_content)
        else:
            df = pd.read_csv(file_content)
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise


def load_metrics_from_json(file_content) -> Tuple[pd.DataFrame, Dict]:
    """
    Load metrics DataFrame and statistics from JSON export.

    Expected format from App 1 export:
    {
        "metrics": [...],
        "statistics": {...}
    }

    Args:
        file_content: JSON file content

    Returns:
        Tuple of (metrics_df, statistics_dict)
    """
    try:
        if isinstance(file_content, str):
            with open(file_content, 'r') as f:
                data = json.load(f)
        else:
            data = json.load(file_content)

        metrics_df = pd.DataFrame(data.get('metrics', []))
        statistics = data.get('statistics', {})

        return metrics_df, statistics
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        raise


def load_multiple_files(uploaded_files) -> Dict[str, pd.DataFrame]:
    """
    Load multiple conversation files.

    Args:
        uploaded_files: List of uploaded file objects

    Returns:
        Dictionary of {filename: metrics_df}
    """
    conversations = {}

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        file_extension = Path(filename).suffix.lower()

        try:
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
                conversations[filename] = df
            elif file_extension == '.json':
                data = json.load(uploaded_file)
                if 'metrics' in data:
                    df = pd.DataFrame(data['metrics'])
                else:
                    df = pd.DataFrame(data)
                conversations[filename] = df
            else:
                logger.warning(f"Unsupported file type: {filename}")

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")

    return conversations


def export_rho_summary_to_csv(summary_df: pd.DataFrame) -> bytes:
    """
    Export RHO summary to CSV bytes.

    Args:
        summary_df: Summary DataFrame

    Returns:
        CSV as bytes
    """
    output = io.StringIO()
    summary_df.to_csv(output, index=False)
    return output.getvalue().encode('utf-8')


def export_rho_summary_to_json(
    summary_df: pd.DataFrame,
    statistics: Dict[str, Any]
) -> bytes:
    """
    Export RHO summary and statistics to JSON bytes.

    Args:
        summary_df: Summary DataFrame
        statistics: Statistics dictionary

    Returns:
        JSON as bytes
    """
    # Convert DataFrame to records and handle numpy types
    summary_records = summary_df.to_dict(orient='records')
    for record in summary_records:
        for key, value in record.items():
            # Convert numpy types to native Python types
            if hasattr(value, 'item'):
                record[key] = value.item()

    # Convert statistics to JSON-serializable types
    json_safe_stats = {}
    for key, value in statistics.items():
        if isinstance(value, (bool, int, float, str, type(None))):
            json_safe_stats[key] = value
        elif hasattr(value, 'item'):  # numpy types
            json_safe_stats[key] = value.item()
        elif isinstance(value, (np.bool_, np.integer, np.floating)):
            json_safe_stats[key] = value.item()
        else:
            json_safe_stats[key] = str(value)

    data = {
        'summary': summary_records,
        'statistics': json_safe_stats,
        'export_timestamp': pd.Timestamp.now().isoformat()
    }
    return json.dumps(data, indent=2).encode('utf-8')


def format_rho_statistics(stats: Dict[str, Any]) -> str:
    """
    Format RHO statistics for display.

    Args:
        stats: Statistics dictionary

    Returns:
        Formatted markdown string
    """
    lines = []
    lines.append("### Aggregate Statistics")
    lines.append("")
    lines.append(f"**Total Conversations:** {stats.get('total_conversations', 0)}")
    lines.append("")

    lines.append("**Classification Breakdown:**")
    lines.append(f"- ✅ Robust: {stats.get('robust_count', 0)} ({stats.get('robust_percentage', 0):.1f}%)")
    lines.append(f"- ⚖️ Reactive: {stats.get('reactive_count', 0)}")
    lines.append(f"- ❌ Fragile: {stats.get('fragile_count', 0)} ({stats.get('fragile_percentage', 0):.1f}%)")
    lines.append("")

    lines.append("**RHO Statistics:**")
    lines.append(f"- Average: {stats.get('average_rho', 0):.3f}")
    lines.append(f"- Median: {stats.get('median_rho', 0):.3f}")
    lines.append(f"- Min: {stats.get('min_rho', 0):.3f}")
    lines.append(f"- Max: {stats.get('max_rho', 0):.3f}")
    lines.append(f"- Std Dev: {stats.get('std_rho', 0):.3f}")

    return "\n".join(lines)


def validate_metrics_df(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate that DataFrame contains required columns for RHO calculation.

    Args:
        df: Metrics DataFrame

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_columns = ['CumulativeRisk_Model']

    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    if len(df) == 0:
        return False, "DataFrame is empty"

    return True, None


def create_comparison_report(
    summary_df: pd.DataFrame,
    statistics: Dict[str, Any]
) -> str:
    """
    Create a comprehensive comparison report.

    Args:
        summary_df: Summary DataFrame
        statistics: Statistics dictionary

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("# RHO CALCULATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"Total Conversations Analyzed: {statistics.get('total_conversations', 0)}")
    lines.append(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## Classification Results")
    lines.append("")
    lines.append(f"Robust (RHO < 1.0):   {statistics.get('robust_count', 0):3d} conversations ({statistics.get('robust_percentage', 0):5.1f}%)")
    lines.append(f"Reactive (RHO = 1.0): {statistics.get('reactive_count', 0):3d} conversations")
    lines.append(f"Fragile (RHO > 1.0):  {statistics.get('fragile_count', 0):3d} conversations ({statistics.get('fragile_percentage', 0):5.1f}%)")
    lines.append("")

    lines.append("## RHO Statistics")
    lines.append("")
    lines.append(f"Average RHO:    {statistics.get('average_rho', 0):.4f}")
    lines.append(f"Median RHO:     {statistics.get('median_rho', 0):.4f}")
    lines.append(f"Minimum RHO:    {statistics.get('min_rho', 0):.4f}")
    lines.append(f"Maximum RHO:    {statistics.get('max_rho', 0):.4f}")
    lines.append(f"Std Deviation:  {statistics.get('std_rho', 0):.4f}")
    lines.append("")

    lines.append("## Detailed Results")
    lines.append("")
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    lines.append("=" * 70)
    lines.append("End of Report")

    return "\n".join(lines)


def extract_from_app1_results(app1_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Extract necessary columns from App 1 results for RHO calculation.

    Args:
        app1_metrics: Full metrics DataFrame from App 1

    Returns:
        DataFrame with columns needed for RHO
    """
    required_cols = ['Turn', 'CumulativeRisk_Model']
    optional_cols = ['CumulativeRisk_User', 'RobustnessIndex_rho']

    cols_to_keep = required_cols.copy()
    for col in optional_cols:
        if col in app1_metrics.columns:
            cols_to_keep.append(col)

    return app1_metrics[cols_to_keep].copy()
