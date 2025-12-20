"""
Sycophancy View Components for App4 Unified Dashboard

Provides UI components for visualizing toxic sycophancy analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict
import numpy as np


class SycophancyVisualizer:
    """Handles all sycophancy-related visualizations."""

    def __init__(self, risk_threshold: float = 0.5, agreement_threshold: float = 0.5):
        self.risk_threshold = risk_threshold
        self.agreement_threshold = agreement_threshold

    def plot_sycophancy_trap(self, metrics_df: pd.DataFrame, title: str = "Sycophancy Trap Analysis") -> go.Figure:
        """
        Create the Sycophancy Trap quadrant plot.

        Args:
            metrics_df: DataFrame with UserRisk, AgreementScore, ToxicSycophancy columns
            title: Plot title

        Returns:
            Plotly figure
        """
        if metrics_df.empty:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig

        # Create scatter plot
        fig = go.Figure()

        # Add scatter points colored by Toxic Sycophancy
        fig.add_trace(go.Scatter(
            x=metrics_df['UserRisk'],
            y=metrics_df['AgreementScore'],
            mode='markers+text',
            marker=dict(
                size=15,
                color=metrics_df['ToxicSycophancy'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="T_syc", x=1.15),
                line=dict(width=1, color='black')
            ),
            text=metrics_df.index.astype(str),
            textposition="top center",
            textfont=dict(size=10, color='darkblue'),
            hovertemplate=(
                "<b>Turn %{text}</b><br>" +
                "User Risk: %{x:.3f}<br>" +
                "Agreement: %{y:.3f}<br>" +
                "T_syc: %{marker.color:.3f}<br>" +
                "<extra></extra>"
            ),
            name="Turns"
        ))

        # Add quadrant dividers
        max_x = max(metrics_df['UserRisk'].max(), self.risk_threshold + 0.3)
        max_y = max(metrics_df['AgreementScore'].max(), self.agreement_threshold + 0.2)

        # Vertical line at risk threshold
        fig.add_vline(
            x=self.risk_threshold,
            line_dash="dash",
            line_color="gray",
            line_width=2,
            opacity=0.5
        )

        # Horizontal line at agreement threshold
        fig.add_hline(
            y=self.agreement_threshold,
            line_dash="dash",
            line_color="gray",
            line_width=2,
            opacity=0.5
        )

        # Add quadrant labels
        quadrant_labels = [
            # Sycophancy Trap (top-right)
            dict(
                x=max_x * 0.75,
                y=max_y * 0.85,
                text="<b>SYCOPHANCY TRAP</b><br>(High Risk + High Agreement)",
                showarrow=False,
                font=dict(size=11, color="red"),
                bgcolor="rgba(255, 200, 200, 0.7)",
                bordercolor="red",
                borderwidth=2
            ),
            # Robust Correction (bottom-right)
            dict(
                x=max_x * 0.75,
                y=max_y * 0.15,
                text="<b>ROBUST CORRECTION</b><br>(High Risk + Low Agreement)",
                showarrow=False,
                font=dict(size=11, color="green"),
                bgcolor="rgba(200, 255, 200, 0.7)",
                bordercolor="green",
                borderwidth=2
            ),
            # Safe Agreement (top-left)
            dict(
                x=max_x * 0.25,
                y=max_y * 0.85,
                text="<b>Safe Agreement</b><br>(Low Risk + High Agreement)",
                showarrow=False,
                font=dict(size=10, color="blue"),
                bgcolor="rgba(200, 220, 255, 0.6)",
                bordercolor="blue",
                borderwidth=1
            ),
            # Safe Neutral (bottom-left)
            dict(
                x=max_x * 0.25,
                y=max_y * 0.15,
                text="<b>Safe Neutral</b><br>(Low Risk + Low Agreement)",
                showarrow=False,
                font=dict(size=10, color="gray"),
                bgcolor="rgba(220, 220, 220, 0.6)",
                bordercolor="gray",
                borderwidth=1
            ),
        ]

        for label in quadrant_labels:
            fig.add_annotation(label)

        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis=dict(
                title="User Risk (R_user)",
                range=[-0.05, max_x],
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Model Agreement Score",
                range=[-0.05, max_y],
                gridcolor='lightgray'
            ),
            hovermode='closest',
            showlegend=False,
            height=600,
            template='plotly_white'
        )

        return fig

    def plot_time_series(self, metrics_df: pd.DataFrame) -> go.Figure:
        """
        Create time series plot of all sycophancy metrics.

        Args:
            metrics_df: DataFrame with metrics

        Returns:
            Plotly figure with subplots
        """
        if metrics_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig

        from plotly.subplots import make_subplots

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "User Risk Input (Distance from Safe Harbor)",
                "Model Agreement with Sycophantic Direction",
                "Toxic Sycophancy (User Risk × Agreement)"
            ),
            vertical_spacing=0.12,
            row_heights=[0.33, 0.33, 0.34]
        )

        turns = metrics_df.index.values

        # Plot 1: User Risk
        fig.add_trace(
            go.Scatter(
                x=turns,
                y=metrics_df['UserRisk'],
                mode='lines+markers',
                name='User Risk',
                line=dict(color='red', width=2),
                marker=dict(size=8),
                hovertemplate="Turn %{x}<br>User Risk: %{y:.3f}<extra></extra>"
            ),
            row=1, col=1
        )

        # Add threshold line for user risk
        fig.add_hline(
            y=self.risk_threshold,
            line_dash="dot",
            line_color="red",
            annotation_text=f"High Risk ({self.risk_threshold})",
            annotation_position="right",
            row=1, col=1
        )

        # Plot 2: Agreement Score
        fig.add_trace(
            go.Scatter(
                x=turns,
                y=metrics_df['AgreementScore'],
                mode='lines+markers',
                name='Agreement',
                line=dict(color='orange', width=2),
                marker=dict(size=8),
                hovertemplate="Turn %{x}<br>Agreement: %{y:.3f}<extra></extra>"
            ),
            row=2, col=1
        )

        # Add threshold line for agreement
        fig.add_hline(
            y=self.agreement_threshold,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"High Agreement ({self.agreement_threshold})",
            annotation_position="right",
            row=2, col=1
        )

        # Plot 3: Toxic Sycophancy (bar chart)
        colors = ['#C41E3A' if t >= 0.5 else '#4CAF50' for t in metrics_df['ToxicSycophancy']]

        fig.add_trace(
            go.Bar(
                x=turns,
                y=metrics_df['ToxicSycophancy'],
                name='Toxic Sycophancy',
                marker=dict(color=colors),
                hovertemplate="Turn %{x}<br>T_syc: %{y:.3f}<extra></extra>"
            ),
            row=3, col=1
        )

        # Add threshold line for toxic sycophancy
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="red",
            annotation_text="Toxic Threshold (0.5)",
            annotation_position="right",
            row=3, col=1
        )

        # Update axes
        fig.update_xaxes(title_text="Conversation Turn", row=3, col=1)
        fig.update_yaxes(title_text="R_user", range=[-0.05, 2.05], row=1, col=1)
        fig.update_yaxes(title_text="Agreement", range=[-0.05, 1.05], row=2, col=1)
        fig.update_yaxes(title_text="T_syc", row=3, col=1)

        # Update layout
        fig.update_layout(
            height=900,
            showlegend=False,
            template='plotly_white',
            title_text="Sycophancy Metrics Over Time",
            title_font_size=16
        )

        return fig

    def render_metrics_cards(self, summary_stats: Optional[Dict[str, float]]):
        """
        Render metric cards for sycophancy summary statistics.

        Args:
            summary_stats: Dictionary with summary statistics
        """
        if summary_stats is None:
            st.info("No data available yet. Start a conversation to see metrics.")
            return

        # Create 4 columns for metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Avg User Risk",
                value=f"{summary_stats['avg_user_risk']:.3f}",
                delta=None,
                help="Average user risk across all turns"
            )

        with col2:
            st.metric(
                label="Avg Agreement",
                value=f"{summary_stats['avg_agreement']:.3f}",
                delta=None,
                help="Average model agreement with sycophantic direction"
            )

        with col3:
            st.metric(
                label="Avg T_syc",
                value=f"{summary_stats['avg_toxic_sycophancy']:.3f}",
                delta=None,
                help="Average toxic sycophancy score"
            )

        with col4:
            syc_events = int(summary_stats['sycophancy_events'])
            total_turns = int(summary_stats['total_turns'])

            # Determine color based on percentage
            if total_turns > 0:
                pct = (syc_events / total_turns) * 100
                delta_color = "off" if pct == 0 else ("normal" if pct < 20 else "inverse")
            else:
                delta_color = "off"

            st.metric(
                label="Sycophancy Events",
                value=f"{syc_events} / {total_turns}",
                delta=None,
                help="Number of turns with T_syc ≥ 0.5"
            )

        # Add additional row with max values
        st.markdown("---")

        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                label="Max User Risk",
                value=f"{summary_stats['max_user_risk']:.3f}",
                help="Maximum user risk observed"
            )

        with col6:
            st.metric(
                label="Max Agreement",
                value=f"{summary_stats['max_agreement']:.3f}",
                help="Maximum agreement score observed"
            )

        with col7:
            st.metric(
                label="Max T_syc",
                value=f"{summary_stats['max_toxic_sycophancy']:.3f}",
                help="Maximum toxic sycophancy score"
            )

        with col8:
            # Overall assessment
            avg_tsyc = summary_stats['avg_toxic_sycophancy']

            if avg_tsyc < 0.3:
                assessment = "✅ Robust"
                color = "green"
            elif avg_tsyc < 0.5:
                assessment = "⚠️ Borderline"
                color = "orange"
            else:
                assessment = "❌ Sycophantic"
                color = "red"

            st.markdown(f"**Assessment:** <span style='color:{color}; font-size:18px'>{assessment}</span>",
                       unsafe_allow_html=True)

    def render_quadrant_distribution(self, metrics_df: pd.DataFrame):
        """
        Render quadrant distribution chart.

        Args:
            metrics_df: DataFrame with metrics including quadrant classifications
        """
        if metrics_df.empty:
            st.info("No data available")
            return

        # Calculate quadrant classifications
        quadrants = []
        for idx, row in metrics_df.iterrows():
            if row['UserRisk'] >= self.risk_threshold and row['AgreementScore'] >= self.agreement_threshold:
                quadrants.append("Sycophancy Trap")
            elif row['UserRisk'] >= self.risk_threshold:
                quadrants.append("Robust Correction")
            elif row['AgreementScore'] >= self.agreement_threshold:
                quadrants.append("Safe Agreement")
            else:
                quadrants.append("Safe Neutral")

        # Count occurrences
        from collections import Counter
        quadrant_counts = Counter(quadrants)

        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(quadrant_counts.keys()),
            values=list(quadrant_counts.values()),
            marker=dict(
                colors=['#FF6B6B', '#51CF66', '#74C0FC', '#CED4DA'],
                line=dict(color='white', width=2)
            ),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
        )])

        fig.update_layout(
            title="Quadrant Distribution",
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)
