#!/usr/bin/env python3
"""
Vector Precognition Algorithm - Core Processor
Refactored for production deployment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class VectorPrecognitionProcessor:
    """
    Vector Precognition algorithm - Erosion Analysis Only.

    This class processes sequences of model vectors to calculate:
    - Kinematic model (R, v, a) for model responses
    - Failure potential and likelihood of guardrail breach
    - Focus on erosion metrics only (no rho or phi calculations)
    """

    def __init__(
        self,
        vsafe: np.ndarray,
        weights: Dict[str, float]
    ):
        """
        Initialize the Vector Precognition processor.

        Args:
            vsafe: The 2D "Guardrail Centroid" VSAFE vector
            weights: Weights (wR, wv, wa, b) for Failure Potential Score z(N)
        """
        if vsafe.ndim != 1 or len(vsafe) != 2:
            raise ValueError("VSAFE must be a 2-dimensional numpy array (e.g., np.array([x, y])).")

        # Normalize VSAFE to a unit vector for consistent cosine calculations
        norm = np.linalg.norm(vsafe)
        if norm == 0:
            raise ValueError("VSAFE vector cannot be a zero vector.")
        self.vsafe: np.ndarray = vsafe / norm

        self.weights: Dict[str, float] = weights

        # Initialize metric storage
        self._reset_metrics()

    def _reset_metrics(self):
        """Reset all metric lists."""
        # Vector storage
        self.v_model_vectors: List[np.ndarray] = []

        # Model Metrics (Erosion-focused only)
        self.R_model_list: List[float] = []  # Risk Severity (Position)
        self.v_model_list: List[float] = []  # Risk Rate (Velocity)
        self.a_model_list: List[float] = []  # Guardrail Erosion (Acceleration)
        self.C_model_list: List[float] = []  # Cumulative Model Risk
        self.z_n_list: List[float] = []      # Failure Potential Score
        self.L_n_list: List[float] = []      # Likelihood of Risk

    def _calculate_cosine_distance(self, v_n: np.ndarray) -> float:
        """
        Calculate Cosine Distance between turn vector V_N and VSAFE.
        This is the Risk Function R(N).

        Args:
            v_n: The 2D vector for the current conversational turn

        Returns:
            Cosine Distance, R(N)
        """
        v_n_norm = np.linalg.norm(v_n)

        # Handle zero vector case
        if v_n_norm == 0:
            return 1.0  # Neutral risk

        v_n_unit = v_n / v_n_norm

        # Cosine Similarity (vsafe is already unit vector)
        cosine_similarity = np.dot(v_n_unit, self.vsafe)

        # R(N) = 1 - Cosine Similarity
        cosine_distance = 1.0 - cosine_similarity

        # Clip to theoretical bounds [0, 2]
        return np.clip(cosine_distance, 0.0, 2.0)

    def _sigmoid(self, z: float) -> float:
        """
        Sigmoid function for likelihood calculation.

        Args:
            z: Failure potential score

        Returns:
            Likelihood value between 0 and 1
        """
        return 1.0 / (1.0 + np.exp(-z))

    def process_turn(
        self,
        v_model: np.ndarray,
        v_user: Optional[np.ndarray] = None
    ):
        """
        Process a single conversational turn.

        Calculates R(N), v(N), a(N), C(N), z(N), L(N) for model.
        User vectors are accepted but not used for rho calculations.

        Args:
            v_model: 2D vector coordinates for model's response
            v_user: 2D vector coordinates for user's prompt (ignored for erosion-only analysis)
        """
        if v_model.ndim != 1 or len(v_model) != 2:
            raise ValueError("Model vector v_model must be a 2-dimensional numpy array.")

        self.v_model_vectors.append(v_model)

        # --- Process Model Side Only (Erosion Analysis) ---

        # Step 1: Calculate R_model (Position)
        R_m = self._calculate_cosine_distance(v_model)
        self.R_model_list.append(R_m)

        # Step 2: Calculate v_model (Velocity) - First Derivative
        if len(self.R_model_list) == 1:
            v_m = 0.0  # No velocity at first turn
        else:
            v_m = self.R_model_list[-1] - self.R_model_list[-2]
        self.v_model_list.append(v_m)

        # Step 3: Calculate a_model (Acceleration) - Second Derivative (EROSION)
        if len(self.v_model_list) == 1:
            a_m = 0.0  # No acceleration at first turn
        else:
            a_m = self.v_model_list[-1] - self.v_model_list[-2]
        self.a_model_list.append(a_m)

        # Step 4: Calculate Cumulative Model Risk (Integral)
        if len(self.R_model_list) == 1:
            C_m = R_m  # First turn
        else:
            # Trapezoidal integration
            prev_C = self.C_model_list[-1]
            prev_R = self.R_model_list[-2]
            C_m = prev_C + 0.5 * (prev_R + R_m)
        self.C_model_list.append(C_m)

        # Step 5: Calculate Failure Potential Score z(N)
        wR = self.weights['wR']
        wv = self.weights['wv']
        wa = self.weights['wa']
        b = self.weights['b']

        z_n = wR * R_m + wv * v_m + wa * a_m + b
        self.z_n_list.append(z_n)

        # Step 6: Calculate Likelihood L(N)
        L_n = self._sigmoid(z_n)
        self.L_n_list.append(L_n)

    def get_metrics(self) -> pd.DataFrame:
        """
        Get all metrics as a pandas DataFrame (erosion-focused only).

        Returns:
            DataFrame with columns for erosion analysis metrics only
        """
        n_turns = len(self.R_model_list)

        data = {
            'Turn': list(range(1, n_turns + 1)),
            'RiskSeverity_Model': self.R_model_list,
            'RiskRate_v(N)': self.v_model_list,
            'GuardrailErosion_a(N)': self.a_model_list,
            'CumulativeRisk_Model': self.C_model_list,
            'FailurePotential_z(N)': self.z_n_list,
            'Likelihood_L(N)': self.L_n_list,
        }

        # Add vector coordinates if available
        if len(self.v_model_vectors) > 0:
            data['Vector_X'] = [v[0] for v in self.v_model_vectors]
            data['Vector_Y'] = [v[1] for v in self.v_model_vectors]

        return pd.DataFrame(data)

    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Get summary statistics for the conversation (erosion-focused).

        Returns:
            Dictionary with key erosion analysis statistics
        """
        if len(self.R_model_list) == 0:
            return {}

        stats = {
            'total_turns': len(self.R_model_list),
            'peak_risk_severity': max(self.R_model_list),
            'peak_risk_turn': self.R_model_list.index(max(self.R_model_list)) + 1,
            'peak_likelihood': max(self.L_n_list),
            'peak_likelihood_turn': self.L_n_list.index(max(self.L_n_list)) + 1,
            'max_erosion': max(self.a_model_list) if self.a_model_list else 0.0,
            'max_erosion_turn': self.a_model_list.index(max(self.a_model_list)) + 1 if self.a_model_list else 0,
            'final_cumulative_risk_model': self.C_model_list[-1] if self.C_model_list else 0.0,
            'avg_likelihood': np.mean(self.L_n_list) if self.L_n_list else 0.0,
            'avg_erosion': np.mean(self.a_model_list) if self.a_model_list else 0.0,
        }

        return stats

    def reset(self):
        """Reset all metrics for new conversation."""
        self._reset_metrics()

    def process_conversation(
        self,
        model_vectors: List[np.ndarray],
        user_vectors: Optional[List[np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Process an entire conversation at once.

        Args:
            model_vectors: List of 2D model response vectors
            user_vectors: List of 2D user prompt vectors (optional)

        Returns:
            DataFrame with all metrics
        """
        self.reset()

        if user_vectors is None:
            user_vectors = [None] * len(model_vectors)

        if len(model_vectors) != len(user_vectors):
            raise ValueError("Number of model and user vectors must match")

        for v_model, v_user in zip(model_vectors, user_vectors):
            self.process_turn(v_model, v_user)

        return self.get_metrics()
