#!/usr/bin/env python3
"""
Visualization utilities for Vector Precognition applications.
Common plotting functions used across all Streamlit apps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from datetime import datetime
import io

# Set default plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class GuardrailVisualizer:
    """Visualizations for App 1: Guardrail Erosion."""

    def __init__(self, colors: Optional[Dict[str, str]] = None):
        """
        Initialize visualizer with color scheme.

        Args:
            colors: Dictionary of color mappings
        """
        self.colors = colors or {
            'user': '#3498db',      # Blue
            'model': '#e74c3c',     # Red
            'safe': '#2ecc71',      # Green
            'alert': '#f39c12',     # Orange
            'critical': '#c0392b',  # Dark red
            'neutral': '#95a5a6'    # Gray
        }

    def plot_5panel_dynamics(
        self,
        metrics_df: pd.DataFrame,
        alert_threshold: float = 0.8,
        erosion_threshold: float = 0.15,
        figsize: Tuple[int, int] = (16, 10),
        title: str = "Conversation Dynamics"
    ) -> plt.Figure:
        """
        Create comprehensive 6-panel time-series visualization (matches demo2).

        Panels:
        1. Risk Severity (User vs Model)
        2. Risk Rate (Velocity)
        3. Guardrail Erosion (Acceleration)
        4. Likelihood of Breach
        5. Robustness Index (RHO) - Fragile vs Robust classification
        6. Statistics Summary

        Args:
            metrics_df: DataFrame with columns [Turn, RiskSeverity_User, RiskSeverity_Model, ...]
            alert_threshold: Likelihood threshold for alerts
            erosion_threshold: Erosion threshold for alerts
            figsize: Figure size (width, height)
            title: Main plot title

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

        turns = metrics_df['Turn'].values

        # Panel 1: Risk Severity (User vs Model)
        ax1 = axes[0, 0]
        if 'RiskSeverity_User' in metrics_df.columns:
            ax1.plot(turns, metrics_df['RiskSeverity_User'],
                    marker='o', color=self.colors['user'],
                    label='User Risk', linewidth=2)
        if 'RiskSeverity_Model' in metrics_df.columns:
            ax1.plot(turns, metrics_df['RiskSeverity_Model'],
                    marker='s', color=self.colors['model'],
                    label='Model Risk', linewidth=2)
        ax1.axhline(y=1.0, color=self.colors['alert'], linestyle='--',
                   label='Moderate Risk', alpha=0.7)
        ax1.set_xlabel('Turn', fontsize=11)
        ax1.set_ylabel('Risk Severity R(N)', fontsize=11)
        ax1.set_title('Risk Severity (Position)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Panel 2: Risk Rate (Velocity)
        ax2 = axes[0, 1]
        if 'RiskRate_v(N)' in metrics_df.columns:
            ax2.plot(turns, metrics_df['RiskRate_v(N)'],
                    marker='o', color=self.colors['model'], linewidth=2)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.fill_between(turns, 0, metrics_df['RiskRate_v(N)'],
                            where=(metrics_df['RiskRate_v(N)'] > 0),
                            color=self.colors['critical'], alpha=0.2, label='Increasing Risk')
            ax2.fill_between(turns, 0, metrics_df['RiskRate_v(N)'],
                            where=(metrics_df['RiskRate_v(N)'] < 0),
                            color=self.colors['safe'], alpha=0.2, label='Decreasing Risk')
        ax2.set_xlabel('Turn', fontsize=11)
        ax2.set_ylabel('Risk Rate v(N)', fontsize=11)
        ax2.set_title('Risk Rate (Velocity)', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # Panel 3: Guardrail Erosion (Acceleration)
        ax3 = axes[1, 0]
        if 'GuardrailErosion_a(N)' in metrics_df.columns:
            ax3.plot(turns, metrics_df['GuardrailErosion_a(N)'],
                    marker='o', color=self.colors['critical'], linewidth=2)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.axhline(y=erosion_threshold, color=self.colors['alert'], linestyle='--',
                       label=f'Alert Threshold ({erosion_threshold})', alpha=0.7)
            ax3.axhline(y=-erosion_threshold, color=self.colors['alert'], linestyle='--',
                       alpha=0.7)
            # Highlight dangerous acceleration (positive or negative)
            danger_mask = np.abs(metrics_df['GuardrailErosion_a(N)']) > erosion_threshold
            if danger_mask.any():
                ax3.scatter(turns[danger_mask],
                          metrics_df.loc[danger_mask, 'GuardrailErosion_a(N)'],
                          color=self.colors['critical'], s=100, marker='X',
                          label='High Erosion', zorder=5)
        ax3.set_xlabel('Turn', fontsize=11)
        ax3.set_ylabel('Erosion a(N)', fontsize=11)
        ax3.set_title('Guardrail Erosion (Acceleration)', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # Panel 4: Likelihood of Breach
        ax4 = axes[1, 1]
        if 'Likelihood_L(N)' in metrics_df.columns:
            ax4.plot(turns, metrics_df['Likelihood_L(N)'],
                    marker='o', color=self.colors['model'], linewidth=2.5)
            ax4.axhline(y=alert_threshold, color=self.colors['alert'],
                       linestyle='--', linewidth=2, label=f'Alert ({alert_threshold})')
            ax4.fill_between(turns, alert_threshold, 1.0,
                            color=self.colors['critical'], alpha=0.1,
                            label='Critical Zone')

            # Mark peak likelihood
            peak_idx = metrics_df['Likelihood_L(N)'].idxmax()
            peak_val = metrics_df.loc[peak_idx, 'Likelihood_L(N)']
            peak_turn = metrics_df.loc[peak_idx, 'Turn']
            ax4.scatter([peak_turn], [peak_val], color=self.colors['critical'],
                       s=150, marker='*', zorder=5, label=f'Peak: {peak_val:.3f}')
            ax4.annotate(f'Peak\n{peak_val:.3f}',
                        xy=(peak_turn, peak_val),
                        xytext=(peak_turn, peak_val + 0.1),
                        ha='center', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax4.set_xlabel('Turn', fontsize=11)
        ax4.set_ylabel('Likelihood L(N)', fontsize=11)
        ax4.set_title('Likelihood of Breach', fontsize=12, fontweight='bold')
        ax4.set_ylim([0, 1.05])
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        # Panel 5: Robustness Index (rho) - Matches demo2
        ax5 = axes[2, 0]
        if 'RobustnessIndex_rho' in metrics_df.columns:
            rho_values = metrics_df['RobustnessIndex_rho'].values

            # Plot rho values
            ax5.plot(turns, rho_values, marker='p', color='purple',
                    linewidth=2.5, markersize=10, label='Robustness Index (ρ)')

            # Add threshold line at rho = 1.0
            ax5.axhline(y=1.0, color='gray', linestyle='--', linewidth=2,
                       label='ρ = 1.0 (Reactive)', alpha=0.7)

            # Fill fragile zone (rho > 1.0)
            ax5.fill_between(turns, 1.0, rho_values,
                            where=(rho_values > 1.0),
                            color='red', alpha=0.2, label='Fragile Zone (ρ > 1)')

            # Fill robust zone (rho < 1.0)
            ax5.fill_between(turns, rho_values, 1.0,
                            where=(rho_values < 1.0),
                            color='green', alpha=0.2, label='Robust Zone (ρ < 1)')

            # Mark final rho value
            final_rho = rho_values[-1]
            ax5.scatter([turns[-1]], [final_rho], color='purple',
                       s=200, marker='*', zorder=5, edgecolors='black', linewidths=2)

            # Annotate final rho
            status = "ROBUST" if final_rho < 1.0 else "FRAGILE"
            color = 'green' if final_rho < 1.0 else 'red'
            ax5.annotate(f'{status}\nρ = {final_rho:.3f}',
                        xy=(turns[-1], final_rho),
                        xytext=(turns[-1], final_rho + 0.15),
                        ha='center', fontsize=10, fontweight='bold',
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        else:
            ax5.text(0.5, 0.5, 'Robustness Index not calculated',
                    ha='center', va='center', fontsize=12, transform=ax5.transAxes)

        ax5.set_xlabel('Turn', fontsize=11)
        ax5.set_ylabel('Robustness Index (ρ)', fontsize=11)
        ax5.set_title('Robustness Index (Fragile: > 1.0, Robust: < 1.0)', fontsize=12, fontweight='bold')
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3)

        # Panel 6: Statistics Summary
        ax6 = axes[2, 1]
        ax6.axis('off')

        # Calculate statistics
        stats_text = self._generate_statistics_text(metrics_df, alert_threshold)
        ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig

    def _generate_statistics_text(self, df: pd.DataFrame, threshold: float) -> str:
        """Generate statistics summary text."""
        stats = []
        stats.append("═══ CONVERSATION STATISTICS ═══\n")
        stats.append(f"Total Turns: {len(df)}\n")

        if 'RiskSeverity_Model' in df.columns:
            peak_risk = df['RiskSeverity_Model'].max()
            peak_turn = df.loc[df['RiskSeverity_Model'].idxmax(), 'Turn']
            stats.append(f"Peak Risk: {peak_risk:.3f} (Turn {peak_turn})\n")

        if 'Likelihood_L(N)' in df.columns:
            peak_likelihood = df['Likelihood_L(N)'].max()
            peak_l_turn = df.loc[df['Likelihood_L(N)'].idxmax(), 'Turn']
            stats.append(f"Peak Likelihood: {peak_likelihood:.3f} (Turn {peak_l_turn})\n")

            alerts = df[df['Likelihood_L(N)'] > threshold]
            stats.append(f"Alert Triggers: {len(alerts)} turns\n")

        if 'GuardrailErosion_a(N)' in df.columns:
            max_erosion = df['GuardrailErosion_a(N)'].max()
            stats.append(f"Max Erosion: {max_erosion:.3f}\n")

        if 'RobustnessIndex_rho' in df.columns:
            final_rho = df['RobustnessIndex_rho'].iloc[-1]
            rho_status = "ROBUST ✓" if final_rho < 1.0 else "FRAGILE ✗"
            stats.append(f"\nRobustness: {rho_status}\n")
            stats.append(f"Final RHO: {final_rho:.3f}\n")

        return ''.join(stats)

    def plot_metrics_table(self, metrics_df: pd.DataFrame) -> plt.Figure:
        """
        Create a formatted table visualization of metrics.

        Args:
            metrics_df: DataFrame with metrics

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(14, len(metrics_df) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')

        # Select columns to display
        display_cols = ['Turn']
        optional_cols = ['RiskSeverity_User', 'RiskSeverity_Model', 'RiskRate_v(N)',
                        'GuardrailErosion_a(N)', 'Likelihood_L(N)', 'RobustnessIndex_rho']

        for col in optional_cols:
            if col in metrics_df.columns:
                display_cols.append(col)

        table_data = metrics_df[display_cols].round(4)

        table = ax.table(cellText=table_data.values,
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Color header
        for i in range(len(display_cols)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color rows based on likelihood
        if 'Likelihood_L(N)' in metrics_df.columns:
            likelihood_col_idx = display_cols.index('Likelihood_L(N)')
            for i in range(1, len(table_data) + 1):
                likelihood_val = table_data.iloc[i-1]['Likelihood_L(N)']
                if likelihood_val > 0.8:
                    table[(i, likelihood_col_idx)].set_facecolor('#e74c3c')
                    table[(i, likelihood_col_idx)].set_text_props(color='white', weight='bold')
                elif likelihood_val > 0.5:
                    table[(i, likelihood_col_idx)].set_facecolor('#f39c12')

        plt.title('Conversation Metrics Table', fontsize=14, fontweight='bold', pad=20)
        return fig


class RHOVisualizer:
    """Visualizations for App 2: RHO Calculator."""

    def __init__(self, colors: Optional[Dict[str, str]] = None):
        self.colors = colors or {
            'user': '#3498db',
            'model': '#e74c3c',
            'robust': '#2ecc71',
            'fragile': '#c0392b',
            'neutral': '#95a5a6'
        }

    def plot_cumulative_risk(
        self,
        metrics_df: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot cumulative risk comparison (C_user vs C_model).

        Args:
            metrics_df: DataFrame with CumulativeRisk_User and CumulativeRisk_Model
            figsize: Figure size

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        turns = metrics_df['Turn'].values

        ax.plot(turns, metrics_df['CumulativeRisk_User'],
               marker='o', color=self.colors['user'], label='Cumulative User Risk',
               linewidth=2.5)
        ax.plot(turns, metrics_df['CumulativeRisk_Model'],
               marker='s', color=self.colors['model'], label='Cumulative Model Risk',
               linewidth=2.5)

        # Fill between to show difference
        ax.fill_between(turns,
                       metrics_df['CumulativeRisk_User'],
                       metrics_df['CumulativeRisk_Model'],
                       where=(metrics_df['CumulativeRisk_Model'] > metrics_df['CumulativeRisk_User']),
                       color=self.colors['fragile'], alpha=0.2, label='Model Amplification')

        ax.set_xlabel('Turn', fontsize=12)
        ax.set_ylabel('Cumulative Risk', fontsize=12)
        ax.set_title('Cumulative Risk Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_rho_timeline(
        self,
        metrics_df: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot RHO evolution over conversation.

        Args:
            metrics_df: DataFrame with RobustnessIndex_rho
            figsize: Figure size

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        turns = metrics_df['Turn'].values
        rho = metrics_df['RobustnessIndex_rho'].values

        # Plot RHO
        ax.plot(turns, rho, marker='o', color=self.colors['model'], linewidth=2.5)

        # Add threshold line
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2,
                  label='Threshold (ρ=1.0)', alpha=0.7)

        # Color zones
        ax.fill_between(turns, 0, 1.0, color=self.colors['robust'],
                       alpha=0.1, label='Robust Zone (ρ<1)')
        ax.fill_between(turns, 1.0, max(rho.max(), 1.5),
                       color=self.colors['fragile'], alpha=0.1, label='Fragile Zone (ρ>1)')

        # Mark final RHO
        final_rho = rho[-1]
        ax.scatter([turns[-1]], [final_rho], s=200, marker='*',
                  color='gold', edgecolors='black', linewidths=2, zorder=5,
                  label=f'Final ρ={final_rho:.3f}')

        ax.set_xlabel('Turn', fontsize=12)
        ax.set_ylabel('Robustness Index (ρ)', fontsize=12)
        ax.set_title('Robustness Index Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class PHIVisualizer:
    """Visualizations for App 3: PHI Evaluator."""

    def plot_fragility_distribution(
        self,
        rho_values: List[float],
        phi_score: float,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot fragility distribution histogram.

        Args:
            rho_values: List of RHO values from all conversations
            phi_score: Calculated PHI score
            figsize: Figure size

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(rho_values, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
                  label='Robust/Fragile Threshold')

        # Add PHI score annotation
        ax.text(0.95, 0.95, f'Φ Score: {phi_score:.4f}',
               transform=ax.transAxes, fontsize=14, fontweight='bold',
               ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        ax.set_xlabel('Robustness Index (ρ)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Model Fragility Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig


def fig_to_bytes(fig: plt.Figure, format: str = 'png', dpi: int = 100) -> bytes:
    """
    Convert matplotlib figure to bytes for download.

    Args:
        fig: Matplotlib Figure
        format: Image format ('png', 'pdf', 'svg')
        dpi: Resolution

    Returns:
        Bytes object
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()
