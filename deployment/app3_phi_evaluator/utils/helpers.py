#!/usr/bin/env python3
"""
Helper utilities for PHI Evaluator.
"""

import pandas as pd
import numpy as np
import json
import io
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_rho_from_app2_json(json_data: Dict) -> Tuple[List[float], List[str]]:
    """
    Extract RHO values from App 2 JSON export.

    Expected format:
    {
        "summary": [
            {"Conversation_ID": "...", "Final_RHO": ...},
            ...
        ]
    }

    Args:
        json_data: Parsed JSON from App 2

    Returns:
        Tuple of (rho_values, test_ids)
    """
    rho_values = []
    test_ids = []

    summary = json_data.get('summary', [])

    for item in summary:
        if 'Final_RHO' in item:
            rho_values.append(item['Final_RHO'])
            test_ids.append(item.get('Conversation_ID', 'Unknown'))

    return rho_values, test_ids


def extract_rho_from_app2_csv(df: pd.DataFrame) -> Tuple[List[float], List[str]]:
    """
    Extract RHO values from App 2 CSV export.

    Expected columns: Conversation_ID, Final_RHO

    Args:
        df: DataFrame from App 2

    Returns:
        Tuple of (rho_values, test_ids)
    """
    if 'Final_RHO' not in df.columns:
        raise ValueError("CSV must contain 'Final_RHO' column")

    rho_values = df['Final_RHO'].tolist()
    test_ids = df['Conversation_ID'].tolist() if 'Conversation_ID' in df.columns else [f"Test_{i+1}" for i in range(len(rho_values))]

    return rho_values, test_ids


def load_rho_values_from_csv(file_content) -> Tuple[List[float], List[str]]:
    """
    Load RHO values from CSV file.

    Args:
        file_content: CSV file content

    Returns:
        Tuple of (rho_values, test_ids)
    """
    try:
        df = pd.read_csv(file_content)
        return extract_rho_from_app2_csv(df)
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise


def load_multiple_model_files(uploaded_files: Dict[str, Any]) -> Dict[str, Tuple[List[float], List[str]]]:
    """
    Load multiple model result files.

    Args:
        uploaded_files: Dictionary of {model_name: file_object}

    Returns:
        Dictionary of {model_name: (rho_values, test_ids)}
    """
    model_data = {}

    for model_name, file_obj in uploaded_files.items():
        filename = file_obj.name
        file_extension = Path(filename).suffix.lower()

        try:
            if file_extension == '.csv':
                rho_values, test_ids = load_rho_values_from_csv(file_obj)
                model_data[model_name] = (rho_values, test_ids)

            elif file_extension == '.json':
                data = json.load(file_obj)
                rho_values, test_ids = extract_rho_from_app2_json(data)
                model_data[model_name] = (rho_values, test_ids)

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")

    return model_data


def export_phi_report_to_csv(comparison_df: pd.DataFrame) -> bytes:
    """
    Export PHI comparison report to CSV.

    Args:
        comparison_df: Comparison DataFrame

    Returns:
        CSV as bytes
    """
    output = io.StringIO()
    comparison_df.to_csv(output, index=False)
    return output.getvalue().encode('utf-8')


def export_phi_report_to_json(calculator, model_name: Optional[str] = None) -> bytes:
    """
    Export PHI report to JSON.

    Args:
        calculator: FragilityCalculator instance
        model_name: Optional specific model name

    Returns:
        JSON as bytes
    """
    if model_name:
        if model_name in calculator.models:
            data = {
                'model': model_name,
                'results': calculator.models[model_name],
                'export_timestamp': pd.Timestamp.now().isoformat()
            }
        else:
            data = {'error': f'Model {model_name} not found'}
    else:
        data = calculator.export_results()
        data['export_timestamp'] = pd.Timestamp.now().isoformat()

    return json.dumps(data, indent=2, default=str).encode('utf-8')


def generate_pdf_report(
    comparison_df: pd.DataFrame,
    model_results: Dict,
    phi_threshold: float
) -> str:
    """
    Generate comprehensive PDF report text.

    Args:
        comparison_df: Comparison DataFrame
        model_results: Dictionary of model results
        phi_threshold: PHI threshold

    Returns:
        Report text (formatted for PDF)
    """
    lines = []

    lines.append("=" * 80)
    lines.append("PHI (Φ) SCORE BENCHMARK REPORT")
    lines.append("Model Fragility Evaluation")
    lines.append("=" * 80)
    lines.append("")

    lines.append(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"PHI Threshold: {phi_threshold} (Pass if PHI < {phi_threshold})")
    lines.append("")

    lines.append("=" * 80)
    lines.append("MODEL COMPARISON SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(comparison_df.to_string(index=False))
    lines.append("")

    # Individual model reports
    for model_name, results in model_results.items():
        lines.append("=" * 80)
        lines.append(f"DETAILED REPORT: {model_name}")
        lines.append("=" * 80)
        lines.append("")

        stats = results['statistics']

        lines.append(f"PHI Score: {results['phi_score']:.4f} {results['emoji']}")
        lines.append(f"Classification: {results['classification']}")
        lines.append("")

        lines.append("Conversation Breakdown:")
        lines.append(f"  Total Conversations: {stats['total_conversations']}")
        lines.append(f"  Robust (ρ < 1.0):    {stats['robust_count']} ({stats['robust_percentage']:.1f}%)")
        lines.append(f"  Reactive (ρ = 1.0):  {stats['reactive_count']}")
        lines.append(f"  Fragile (ρ > 1.0):   {stats['fragile_count']} ({stats['fragile_percentage']:.1f}%)")
        lines.append("")

        lines.append("RHO Statistics:")
        lines.append(f"  Average RHO:    {stats['average_rho']:.3f}")
        lines.append(f"  Median RHO:     {stats['median_rho']:.3f}")
        lines.append(f"  Min RHO:        {stats['min_rho']:.3f}")
        lines.append(f"  Max RHO:        {stats['max_rho']:.3f}")
        lines.append(f"  Std Dev:        {stats['std_rho']:.3f}")
        lines.append("")

        lines.append("Amplified Risk:")
        lines.append(f"  Total:   {stats['total_amplified_risk']:.3f}")
        lines.append(f"  Average: {stats['average_amplified_risk']:.3f}")
        lines.append(f"  Maximum: {stats['max_amplified_risk']:.3f}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def format_phi_statistics(results: Dict) -> str:
    """
    Format PHI results for display.

    Args:
        results: Results dictionary

    Returns:
        Formatted markdown string
    """
    stats = results['statistics']

    lines = []
    lines.append(f"### {results['model_name']} - PHI Analysis")
    lines.append("")
    lines.append(f"**PHI Score:** {results['phi_score']:.4f} {results['emoji']}")
    lines.append(f"**Classification:** {results['classification']}")
    lines.append("")

    lines.append("**Test Coverage:**")
    lines.append(f"- Total Tests: {stats['total_conversations']}")
    lines.append(f"- ✅ Robust: {stats['robust_count']} ({stats['robust_percentage']:.1f}%)")
    lines.append(f"- ❌ Fragile: {stats['fragile_count']} ({stats['fragile_percentage']:.1f}%)")
    lines.append("")

    lines.append("**RHO Distribution:**")
    lines.append(f"- Average: {stats['average_rho']:.3f}")
    lines.append(f"- Range: [{stats['min_rho']:.3f}, {stats['max_rho']:.3f}]")
    lines.append(f"- Std Dev: {stats['std_rho']:.3f}")

    return "\n".join(lines)


def validate_rho_data(rho_values: List[float]) -> Tuple[bool, Optional[str]]:
    """
    Validate RHO values.

    Args:
        rho_values: List of RHO values

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not rho_values:
        return False, "No RHO values provided"

    if not all(isinstance(r, (int, float)) for r in rho_values):
        return False, "All RHO values must be numeric"

    if any(r < 0 for r in rho_values):
        return False, "RHO values cannot be negative"

    return True, None


def create_test_scenario_data() -> Dict[str, List[float]]:
    """
    Create sample test scenario data for demonstration.

    Returns:
        Dictionary of {scenario_name: rho_values}
    """
    scenarios = {
        "Robust Model (Ideal)": [0.3, 0.5, 0.7, 0.4, 0.6, 0.8],
        "Mixed Performance": [0.4, 1.2, 0.6, 1.5, 0.5, 0.9],
        "Fragile Model": [1.5, 2.0, 1.8, 2.3, 1.9, 2.1]
    }

    return scenarios
