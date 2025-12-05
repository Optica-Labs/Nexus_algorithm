#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vector Precognition Demo with LOCAL Embeddings (v3)

This version uses LOCAL sentence-transformers (all-MiniLM-L6-v2) instead of AWS Bedrock.

KEY DIFFERENCES FROM DEMO2:
1. Uses sentence-transformers (all-MiniLM-L6-v2) for local embedding generation
2. No AWS dependencies - runs completely offline
3. Faster embedding generation (~9ms vs ~100ms for AWS)
4. Requires new PCA model trained on local embeddings (384D instead of 1536D)
5. Zero cost - no API calls
6. Complete data privacy - embeddings never leave local machine

FEATURES:
- Symmetrical analysis of User Risk (R_user) and Model Risk (R_model)
- Cumulative Risk Exposure (C_user and C_model) using integral math
- Robustness Index (rho) = C_model / (C_user + epsilon)
- Enhanced timing measurements for full pipeline
- 5-panel time-series plot
- Three summary plot types: Likelihood, Cumulative, and Robustness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import torch
import pickle

# Set up a clean plotting style
plt.style.use('seaborn-v0_8-whitegrid')


class LocalEmbedder:
    """
    Local embedding generator using sentence-transformers.
    Replaces AWS Bedrock Titan embeddings with local all-MiniLM-L6-v2 model.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        """
        Initialize the local embedding model.
        
        Args:
            model_name: Name of sentence-transformers model (default: all-MiniLM-L6-v2)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        print(f"\n🤖 Initializing Local Embedding Model: {model_name}")
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"   Device: {device.upper()}")
        
        if device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Load model
        load_start = time.perf_counter()
        self.model = SentenceTransformer(model_name, device=device)
        load_time = (time.perf_counter() - load_start) * 1000
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"   Embedding dimension: {self.embedding_dim}D")
        print(f"   Load time: {load_time:.0f}ms")
        
        # Warm-up (first inference is slower due to JIT compilation)
        print(f"   Warming up...", end=" ", flush=True)
        _ = self.model.encode("warmup", show_progress_bar=False)
        print("✓")
        
        # Statistics
        self.total_embeddings = 0
        self.total_time_ms = 0.0
        
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        start = time.perf_counter()
        embedding = self.model.encode(text, show_progress_bar=False)
        elapsed = (time.perf_counter() - start) * 1000
        
        self.total_embeddings += 1
        self.total_time_ms += elapsed
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (more efficient than one-by-one).
        
        Args:
            texts: List of input texts
            
        Returns:
            List of numpy arrays containing embeddings
        """
        start = time.perf_counter()
        embeddings = self.model.encode(texts, show_progress_bar=False)
        elapsed = (time.perf_counter() - start) * 1000
        
        self.total_embeddings += len(texts)
        self.total_time_ms += elapsed
        
        return [emb for emb in embeddings]
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get embedding generation statistics.
        
        Returns:
            Dictionary with timing statistics
        """
        avg_time = self.total_time_ms / self.total_embeddings if self.total_embeddings > 0 else 0
        return {
            'total_embeddings': self.total_embeddings,
            'total_time_ms': self.total_time_ms,
            'avg_time_ms': avg_time
        }


def load_conversation_from_json(filepath, speaker_filter='both'):
    """
    Load a conversation from a JSON file with separate user and model messages.
    
    Expected JSON format:
    {
        "conversation": [
            {"turn": 1, "speaker": "user", "message": "Hello"},
            {"turn": 2, "speaker": "llm", "message": "Hi there!"},
            ...
        ]
    }
    
    Args:
        filepath (str): Path to the JSON file
        speaker_filter (str): Which speaker to extract - 'user', 'llm', or 'both'
        
    Returns:
        tuple: (user_messages, model_messages) if speaker_filter='both'
               list: messages if speaker_filter is 'user' or 'llm'
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Support both 'conversation' and 'turns' keys
        conversation_data = data.get('conversation') or data.get('turns')
        if not conversation_data:
            raise ValueError(f"JSON file must contain a 'conversation' or 'turns' key")
        
        user_messages = []
        model_messages = []
        
        for turn in conversation_data:
            speaker = turn.get('speaker', '').lower()
            message = turn.get('message', '')
            
            if not message:
                continue
            
            if speaker == 'user':
                user_messages.append(message)
            elif speaker in ['llm', 'ai', 'model']:
                model_messages.append(message)
        
        if speaker_filter == 'both':
            if len(user_messages) != len(model_messages):
                raise ValueError(f"Unequal user/model message counts: {len(user_messages)} vs {len(model_messages)}")
            print(f"✓ Loaded {len(user_messages)} user-model pairs from conversation")
            return user_messages, model_messages
        elif speaker_filter == 'user':
            if not user_messages:
                raise ValueError("No user messages found in conversation")
            print(f"✓ Loaded {len(user_messages)} user messages from conversation")
            return user_messages
        elif speaker_filter == 'llm':
            if not model_messages:
                raise ValueError("No LLM messages found in conversation")
            print(f"✓ Loaded {len(model_messages)} LLM messages from conversation")
            return model_messages
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {filepath}: {e}")
    except FileNotFoundError:
        raise ValueError(f"File not found: {filepath}")


class VectorPrecognitionLocal:
    """
    Enhanced Vector Precognition algorithm with LOCAL embeddings and Robustness Index.

    This class processes sequences of user and model vectors to calculate:
    - Full kinematic model (R, v, a) for the model
    - Cumulative risk streams for both user and model
    - Robustness Index (rho) to measure model resilience
    - Complete timing analysis including embedding generation
    """
    
    def __init__(self, vsafe: np.ndarray, weights: Dict[str, float], epsilon: float = 0.1):
        """
        Initializes the Enhanced Vector Precognition processor.

        Args:
            vsafe (np.ndarray): The 2D "Guardrail Centroid" VSAFE vector
            weights (Dict[str, float]): Weights (wR, wv, wa, b) for the
                                       Failure Potential Score z(N)
            epsilon (float): Small constant to prevent division by zero
                            when calculating the Robustness Index (default: 0.1)
        """
        if vsafe.ndim != 1 or len(vsafe) != 2:
            raise ValueError("VSAFE must be a 2-dimensional numpy array (e.g., np.array([x, y])).")
        
        # Normalize VSAFE to a unit vector for consistent cosine calculations
        norm = np.linalg.norm(vsafe)
        if norm == 0:
            raise ValueError("VSAFE vector cannot be a zero vector.")
        self.vsafe: np.ndarray = vsafe / norm
        
        self.weights: Dict[str, float] = weights
        self.epsilon: float = epsilon
        
        # Lists to store the history of metrics for the conversation
        self.v_n_vectors: List[np.ndarray] = []
        
        # --- Model Metrics ---
        self.R_model_list: List[float] = []  # Risk Severity (Position)
        self.v_model_list: List[float] = []  # Risk Rate (Velocity)
        self.a_model_list: List[float] = []  # Guardrail Erosion (Acceleration)
        self.C_model_list: List[float] = []  # Cumulative Model Risk (Integral)
        self.z_n_list: List[float] = []      # Failure Potential Score
        self.L_n_list: List[float] = []      # Likelihood of Risk

        # --- User Metrics ---
        self.R_user_list: List[float] = []   # Instantaneous User Risk
        self.C_user_list: List[float] = []   # Cumulative User Risk (Integral)
        
        # --- Combined Metric ---
        self.rho_list: List[float] = []      # Robustness Index (rho)
        
        # --- Timing Metrics ---
        self.timing_data: Dict[str, List[float]] = {
            'embedding_user_ms': [],      # NEW: User text embedding time
            'embedding_model_ms': [],     # NEW: Model text embedding time
            'pca_user_ms': [],            # NEW: User PCA projection time
            'pca_model_ms': [],           # NEW: Model PCA projection time
            'cosine_distance_ms': [],     # Risk calculation time
            'velocity_calc_ms': [],       # Velocity calculation time
            'acceleration_calc_ms': [],   # Acceleration calculation time
            'cumulative_calc_ms': [],     # Cumulative risk calculation time
            'likelihood_calc_ms': [],     # Likelihood calculation time
            'total_turn_ms': []           # Total time per turn
        }

    def _calculate_cosine_distance(self, v_n: np.ndarray) -> float:
        """
        Calculates the Cosine Distance between the turn vector V_N and VSAFE.
        This is the Risk Function R(N).
        
        Args:
            v_n (np.ndarray): The 2D vector for the current conversational turn.

        Returns:
            float: The Cosine Distance, R(N).
        """
        v_n_norm = np.linalg.norm(v_n)
        
        # Handle zero vector case (semantically meaningless, max risk)
        if v_n_norm == 0:
            return 1.0  # Orthogonal (neutral risk)
            
        v_n_unit = v_n / v_n_norm
        
        # Since self.vsafe is already a unit vector, dot product is Cosine Similarity
        cosine_similarity = np.dot(v_n_unit, self.vsafe)
        
        # R(N) = 1 - Cosine Similarity 
        cosine_distance = 1.0 - cosine_similarity
        
        # Clip to ensure the value is within the theoretical bounds [0, 2]
        return np.clip(cosine_distance, 0.0, 2.0)

    def process_turn(self, v_model: np.ndarray, v_user: np.ndarray):
        """
        Processes a single conversational turn with BOTH user and model vectors.
        
        Calculates R(N), v(N), a(N), C(N), z(N), L(N) for model,
        R_user, C_user for user, and rho (Robustness Index).

        Args:
            v_model (np.ndarray): The 2D vector coordinates for the model's response
            v_user (np.ndarray): The 2D vector coordinates for the user's prompt
        """
        if v_model.ndim != 1 or len(v_model) != 2:
            raise ValueError("Model vector v_model must be a 2-dimensional numpy array.")
        if v_user.ndim != 1 or len(v_user) != 2:
            raise ValueError("User vector v_user must be a 2-dimensional numpy array.")
        
        # Start timing for the entire turn
        turn_start = time.perf_counter()
        
        self.v_n_vectors.append(v_model)
        
        # --- 1. Process Model Side ---
        
        # Step 1 & 2: Calculate R_model (Position) - TIMED
        t0 = time.perf_counter()
        R_m = self._calculate_cosine_distance(v_model)
        self.R_model_list.append(R_m)
        t1 = time.perf_counter()
        
        # Also calculate R_user for later
        R_u = self._calculate_cosine_distance(v_user)
        self.R_user_list.append(R_u)
        self.timing_data['cosine_distance_ms'].append((t1 - t0) * 1000)
        
        # Step 3 (First Derivative): Calculate v_model (Velocity) - TIMED
        t0 = time.perf_counter()
        if len(self.R_model_list) == 1:
            v_m = 0.0  # No velocity at the first turn
        else:
            v_m = self.R_model_list[-1] - self.R_model_list[-2]
        self.v_model_list.append(v_m)
        t1 = time.perf_counter()
        self.timing_data['velocity_calc_ms'].append((t1 - t0) * 1000)

        # Step 3 (Second Derivative): Calculate a_model (Acceleration) - TIMED
        t0 = time.perf_counter()
        if len(self.v_model_list) <= 1:
            a_m = 0.0  # No acceleration at the first turn
        else:
            if len(self.v_model_list) == 2:
                a_m = self.v_model_list[-1] - 0.0
            else:
                a_m = self.v_model_list[-1] - self.v_model_list[-2]
        self.a_model_list.append(a_m)
        t1 = time.perf_counter()
        self.timing_data['acceleration_calc_ms'].append((t1 - t0) * 1000)
        
        # Calculate C_model and C_user (Cumulative Risk - Integral) - TIMED
        t0 = time.perf_counter()
        if len(self.C_model_list) == 0:
            C_m = R_m
        else:
            C_m = self.C_model_list[-1] + R_m
        self.C_model_list.append(C_m)
        
        # Cumulative Risk (User)
        if len(self.C_user_list) == 0:
            C_u = R_u
        else:
            C_u = self.C_user_list[-1] + R_u
        self.C_user_list.append(C_u)
        
        # Calculate Robustness Index (rho)
        rho = C_m / (C_u + self.epsilon)
        self.rho_list.append(rho)
        t1 = time.perf_counter()
        self.timing_data['cumulative_calc_ms'].append((t1 - t0) * 1000)
        
        # Likelihood Calculation - TIMED
        t0 = time.perf_counter()
        # Likelihood Step 1: Calculate z(N) (Failure Potential)
        z_n = (
            (self.weights['wR'] * R_m) +
            (self.weights['wv'] * v_m) +
            (self.weights['wa'] * a_m) +
            self.weights['b']
        )
        self.z_n_list.append(z_n)
        
        # Likelihood Step 2: Calculate L(N) (Logistic Function)
        L_n = 1.0 / (1.0 + np.exp(-z_n))
        self.L_n_list.append(L_n)
        t1 = time.perf_counter()
        self.timing_data['likelihood_calc_ms'].append((t1 - t0) * 1000)
        
        # End timing for the entire turn
        turn_end = time.perf_counter()
        self.timing_data['total_turn_ms'].append((turn_end - turn_start) * 1000)

    def print_latency_report(self, include_embeddings: bool = True):
        """
        Prints a detailed latency report for the guardrail erosion pipeline.
        Shows timing for each computational step including embeddings.
        
        Args:
            include_embeddings: Whether to include embedding times in report
        """
        if not self.timing_data['total_turn_ms']:
            print("No timing data available. Process some turns first.")
            return
        
        import statistics
        
        print("\n" + "=" * 80)
        print("GUARDRAIL EROSION PIPELINE - LATENCY REPORT (LOCAL EMBEDDINGS)")
        print("=" * 80)
        
        def print_stats(label, times_list, unit="ms"):
            if not times_list:
                return
            mean = statistics.mean(times_list)
            median = statistics.median(times_list)
            min_val = min(times_list)
            max_val = max(times_list)
            std = statistics.stdev(times_list) if len(times_list) > 1 else 0
            total = sum(times_list)
            
            print(f"\n{label}:")
            print(f"  Mean:   {mean:8.4f} {unit}")
            print(f"  Median: {median:8.4f} {unit}")
            print(f"  Min:    {min_val:8.4f} {unit}")
            print(f"  Max:    {max_val:8.4f} {unit}")
            print(f"  Std:    {std:8.4f} {unit}")
            print(f"  Total:  {total:8.2f} {unit}")
            
            return mean
        
        # Embedding times (if available)
        if include_embeddings and self.timing_data.get('embedding_user_ms'):
            print("\n📊 EMBEDDING GENERATION:")
            user_emb_mean = print_stats("  User Text Embedding", self.timing_data['embedding_user_ms'])
            model_emb_mean = print_stats("  Model Text Embedding", self.timing_data['embedding_model_ms'])
            
            if self.timing_data.get('pca_user_ms'):
                user_pca_mean = print_stats("  User PCA Projection", self.timing_data['pca_user_ms'])
                model_pca_mean = print_stats("  Model PCA Projection", self.timing_data['pca_model_ms'])
        
        # Erosion calculations
        print("\n📊 EROSION CALCULATIONS:")
        cosine_mean = print_stats("  Cosine Distance (R)", self.timing_data['cosine_distance_ms'])
        velocity_mean = print_stats("  Velocity (v)", self.timing_data['velocity_calc_ms'])
        accel_mean = print_stats("  Acceleration (a)", self.timing_data['acceleration_calc_ms'])
        cumul_mean = print_stats("  Cumulative Risk (C, ρ)", self.timing_data['cumulative_calc_ms'])
        likelihood_mean = print_stats("  Likelihood (L)", self.timing_data['likelihood_calc_ms'])
        
        print("\n📊 TOTAL PIPELINE:")
        total_mean = print_stats("  Per Turn Total", self.timing_data['total_turn_ms'])
        
        # Breakdown
        print("\n📊 TIME BREAKDOWN:")
        erosion_total = (cosine_mean + velocity_mean + accel_mean + 
                        cumul_mean + likelihood_mean)
        
        if include_embeddings and self.timing_data.get('embedding_user_ms'):
            emb_total = (statistics.mean(self.timing_data['embedding_user_ms']) +
                        statistics.mean(self.timing_data['embedding_model_ms']))
            pca_total = 0
            if self.timing_data.get('pca_user_ms'):
                pca_total = (statistics.mean(self.timing_data['pca_user_ms']) +
                           statistics.mean(self.timing_data['pca_model_ms']))
            
            print(f"  Embeddings:      {emb_total:8.4f} ms ({emb_total/total_mean*100:5.1f}%)")
            if pca_total > 0:
                print(f"  PCA Projection:  {pca_total:8.4f} ms ({pca_total/total_mean*100:5.1f}%)")
            print(f"  Erosion Math:    {erosion_total:8.4f} ms ({erosion_total/total_mean*100:5.1f}%)")
        else:
            print(f"  Erosion Math:    {erosion_total:8.4f} ms (100.0%)")
        
        # Real-time assessment
        print("\n⚡ REAL-TIME SUITABILITY:")
        if total_mean < 10:
            print(f"  ✅ EXCELLENT: {total_mean:.2f}ms avg - Perfect for real-time alerting")
        elif total_mean < 50:
            print(f"  ✅ VERY GOOD: {total_mean:.2f}ms avg - Suitable for real-time use")
        elif total_mean < 100:
            print(f"  ⚠️  GOOD: {total_mean:.2f}ms avg - Acceptable for near real-time")
        elif total_mean < 250:
            print(f"  ⚠️  MODERATE: {total_mean:.2f}ms avg - Borderline for real-time")
        else:
            print(f"  ❌ SLOW: {total_mean:.2f}ms avg - Not suitable for real-time")
        
        print("\n" + "=" * 80)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame containing all calculated metrics.
        """
        df = pd.DataFrame({
            'Turn': range(1, len(self.R_model_list) + 1),
            'R_model': self.R_model_list,
            'v_model': self.v_model_list,
            'a_model': self.a_model_list,
            'C_model': self.C_model_list,
            'z_n': self.z_n_list,
            'L_n': self.L_n_list,
            'R_user': self.R_user_list,
            'C_user': self.C_user_list,
            'rho': self.rho_list
        })
        return df

    def plot_time_series(self, title: str = "Vector Precognition Dynamics (Local Embeddings)", 
                        save_path: Optional[str] = None):
        """
        Creates an enhanced 5-panel time-series plot showing model and user dynamics.
        """
        if len(self.R_model_list) == 0:
            print("No data to plot. Process some turns first.")
            return

        turns = range(1, len(self.R_model_list) + 1)
        
        fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Panel 1: Risk Severity (R)
        ax = axes[0]
        ax.plot(turns, self.R_model_list, marker='o', linestyle='-', 
               linewidth=2, markersize=6, label='Model Risk (R_model)', color='#e74c3c')
        ax.plot(turns, self.R_user_list, marker='s', linestyle='--', 
               linewidth=2, markersize=6, label='User Risk (R_user)', color='#3498db', alpha=0.7)
        ax.set_ylabel('Risk Severity\n(Cosine Distance)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Panel 1: Instantaneous Risk (Position)', fontsize=12, loc='left', pad=10)
        
        # Panel 2: Risk Rate (v)
        ax = axes[1]
        ax.plot(turns, self.v_model_list, marker='o', linestyle='-', 
               linewidth=2, markersize=6, label='Model Velocity (v_model)', color='#f39c12')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_ylabel('Risk Rate\n(First Derivative)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Panel 2: Risk Drift (Velocity)', fontsize=12, loc='left', pad=10)
        
        # Panel 3: Guardrail Erosion (a)
        ax = axes[2]
        ax.plot(turns, self.a_model_list, marker='o', linestyle='-', 
               linewidth=2, markersize=6, label='Erosion (a_model)', color='#9b59b6')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_ylabel('Guardrail Erosion\n(Second Derivative)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Panel 3: Risk Acceleration (Erosion)', fontsize=12, loc='left', pad=10)
        
        # Panel 4: Cumulative Risk (C)
        ax = axes[3]
        ax.plot(turns, self.C_model_list, marker='o', linestyle='-', 
               linewidth=2, markersize=6, label='Model Cumulative (C_model)', color='#e74c3c')
        ax.plot(turns, self.C_user_list, marker='s', linestyle='--', 
               linewidth=2, markersize=6, label='User Cumulative (C_user)', color='#3498db', alpha=0.7)
        ax.set_ylabel('Cumulative Risk\n(Integral)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Panel 4: Total Risk Exposure (Cumulative)', fontsize=12, loc='left', pad=10)
        
        # Panel 5: Likelihood of Risk (L)
        ax = axes[4]
        ax.plot(turns, self.L_n_list, marker='o', linestyle='-', 
               linewidth=2, markersize=6, label='Likelihood (L_n)', color='#16a085')
        ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, 
                  label='Decision Threshold (0.5)', alpha=0.7)
        ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, 
                  label='High Risk Threshold (0.8)', alpha=0.7)
        ax.set_ylabel('Likelihood of Risk\nL(N)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Conversational Turn', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title('Panel 5: Risk Probability (Likelihood)', fontsize=12, loc='left', pad=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Time-series plot saved to: {save_path}")
        
        plt.show()

    def plot_summary(self, plot_type: str = "likelihood", 
                    save_path: Optional[str] = None):
        """
        Creates a summary plot showing key metrics.
        
        Args:
            plot_type: Type of summary plot ('likelihood', 'cumulative', or 'robustness')
            save_path: Optional path to save the plot
        """
        if len(self.R_model_list) == 0:
            print("No data to plot. Process some turns first.")
            return

        turns = range(1, len(self.R_model_list) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if plot_type == "likelihood":
            ax.plot(turns, self.L_n_list, marker='o', linestyle='-', 
                   linewidth=3, markersize=8, label='Likelihood L(N)', color='#16a085')
            ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, 
                      label='Decision Threshold', alpha=0.7)
            ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, 
                      label='High Risk Alert', alpha=0.7)
            ax.set_ylabel('Likelihood of Risk', fontsize=14, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.set_title('Risk Likelihood Over Conversation (Local Embeddings)', 
                        fontsize=16, fontweight='bold')
            
        elif plot_type == "cumulative":
            ax.plot(turns, self.C_model_list, marker='o', linestyle='-', 
                   linewidth=3, markersize=8, label='Model Cumulative (C_model)', color='#e74c3c')
            ax.plot(turns, self.C_user_list, marker='s', linestyle='--', 
                   linewidth=3, markersize=8, label='User Cumulative (C_user)', color='#3498db')
            ax.set_ylabel('Cumulative Risk Exposure', fontsize=14, fontweight='bold')
            ax.set_title('Cumulative Risk: Model vs User (Local Embeddings)', 
                        fontsize=16, fontweight='bold')
            
        elif plot_type == "robustness":
            ax.plot(turns, self.rho_list, marker='o', linestyle='-', 
                   linewidth=3, markersize=8, label='Robustness Index (ρ)', color='#8e44ad')
            ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=2, 
                      label='Parity (ρ=1)', alpha=0.7)
            ax.set_ylabel('Robustness Index (ρ)', fontsize=14, fontweight='bold')
            ax.set_title('Model Robustness Index Over Conversation (Local Embeddings)', 
                        fontsize=16, fontweight='bold')
            ax.text(0.02, 0.98, 
                   'ρ > 1: Model absorbs more risk than user\nρ < 1: Model is more resilient',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Conversational Turn', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Summary plot saved to: {save_path}")
        
        plt.show()


def run_local_embedding_demo(conversation_file: str, 
                             pca_model_path: str,
                             output_dir: str = "output",
                             vsafe: Optional[np.ndarray] = None,
                             weights: Optional[Dict[str, float]] = None,
                             embedding_model: str = 'all-MiniLM-L6-v2'):
    """
    Run complete demo with local embeddings (all-MiniLM-L6-v2).
    
    Args:
        conversation_file: Path to JSON conversation file
        pca_model_path: Path to PCA model (must be trained on local embeddings!)
        output_dir: Directory to save outputs
        vsafe: Optional VSAFE vector (default: [0, 1])
        weights: Optional weights dict (default: balanced weights)
        embedding_model: Name of sentence-transformers model
    """
    print("\n" + "="*80)
    print("VECTOR PRECOGNITION DEMO - LOCAL EMBEDDINGS (v3)")
    print("="*80)
    print(f"\n📁 Conversation file: {conversation_file}")
    print(f"📁 PCA model: {pca_model_path}")
    print(f"🤖 Embedding model: {embedding_model}")
    
    # Initialize local embedder
    embedder = LocalEmbedder(model_name=embedding_model)
    
    # Load PCA model
    print(f"\n📦 Loading PCA model...")
    try:
        with open(pca_model_path, 'rb') as f:
            pca = pickle.load(f)
        print(f"   ✓ PCA loaded: {pca.n_components_}D → 2D")
        print(f"   ✓ Expected input dimension: {pca.n_components_}D")
        
        if pca.n_components_ != embedder.embedding_dim:
            print(f"\n   ⚠️  WARNING: PCA expects {pca.n_components_}D but model outputs {embedder.embedding_dim}D")
            print(f"   ⚠️  You need to retrain PCA on local embeddings!")
            print(f"   ⚠️  Proceeding anyway - results may be incorrect.")
    except Exception as e:
        print(f"   ❌ Error loading PCA: {e}")
        return
    
    # Load conversation
    print(f"\n📖 Loading conversation...")
    try:
        user_messages, model_messages = load_conversation_from_json(
            conversation_file, speaker_filter='both'
        )
    except Exception as e:
        print(f"   ❌ Error loading conversation: {e}")
        return
    
    # Initialize precognition processor
    if vsafe is None:
        vsafe = np.array([0.0, 1.0])  # Default: pointing upward (safe direction)
    
    if weights is None:
        weights = {
            'wR': 3.0,   # Position weight
            'wv': 2.0,   # Velocity weight
            'wa': 1.5,   # Acceleration weight
            'b': -2.0    # Bias term
        }
    
    print(f"\n⚙️  Precognition configuration:")
    print(f"   VSAFE: {vsafe}")
    print(f"   Weights: {weights}")
    
    precog = VectorPrecognitionLocal(vsafe=vsafe, weights=weights)
    
    # Process conversation
    print(f"\n🔄 Processing {len(user_messages)} conversation turns...")
    print(f"   (Embedding → PCA → Erosion calculation)")
    
    for i, (user_msg, model_msg) in enumerate(zip(user_messages, model_messages), 1):
        print(f"\n   Turn {i}/{len(user_messages)}")
        
        # Time user embedding
        t0 = time.perf_counter()
        user_emb = embedder.embed_text(user_msg)
        t1 = time.perf_counter()
        precog.timing_data['embedding_user_ms'].append((t1 - t0) * 1000)
        
        # Time model embedding
        t0 = time.perf_counter()
        model_emb = embedder.embed_text(model_msg)
        t1 = time.perf_counter()
        precog.timing_data['embedding_model_ms'].append((t1 - t0) * 1000)
        
        # Time user PCA
        t0 = time.perf_counter()
        user_2d = pca.transform([user_emb])[0]
        t1 = time.perf_counter()
        precog.timing_data['pca_user_ms'].append((t1 - t0) * 1000)
        
        # Time model PCA
        t0 = time.perf_counter()
        model_2d = pca.transform([model_emb])[0]
        t1 = time.perf_counter()
        precog.timing_data['pca_model_ms'].append((t1 - t0) * 1000)
        
        # Process turn (erosion calculations)
        precog.process_turn(v_model=model_2d, v_user=user_2d)
        
        # Show progress
        L = precog.L_n_list[-1]
        R = precog.R_model_list[-1]
        status = "⚠️ HIGH RISK" if L > 0.8 else "⚠️ ELEVATED" if L > 0.5 else "✓ Normal"
        print(f"      R={R:.3f}, L={L:.3f} {status}")
    
    print(f"\n✅ Processing complete!")
    
    # Print timing report
    precog.print_latency_report(include_embeddings=True)
    
    # Print embedding statistics
    emb_stats = embedder.get_statistics()
    print(f"\n📊 EMBEDDING STATISTICS:")
    print(f"   Total embeddings: {emb_stats['total_embeddings']}")
    print(f"   Total time: {emb_stats['total_time_ms']:.2f}ms")
    print(f"   Average: {emb_stats['avg_time_ms']:.2f}ms per embedding")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save DataFrame
    df = precog.get_dataframe()
    csv_path = os.path.join(output_dir, "metrics_local.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Metrics saved to: {csv_path}")
    
    # Generate plots
    print(f"\n📊 Generating plots...")
    
    # Time-series plot
    ts_path = os.path.join(output_dir, "time_series_local.png")
    precog.plot_time_series(
        title=f"Vector Precognition Dynamics (Local: {embedding_model})",
        save_path=ts_path
    )
    
    # Summary plots
    for plot_type in ['likelihood', 'cumulative', 'robustness']:
        plot_path = os.path.join(output_dir, f"summary_{plot_type}_local.png")
        precog.plot_summary(plot_type=plot_type, save_path=plot_path)
    
    print(f"\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    
    return precog, embedder


if __name__ == "__main__":
    import sys
    
    # Default paths
    conversation_file = "input/unsafe_conversation_example.json"
    pca_model_path = "models/pca_model_local.pkl"  # NOTE: Must be trained on local embeddings!
    output_dir = "output/local"
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        conversation_file = sys.argv[1]
    if len(sys.argv) > 2:
        pca_model_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    # Check if files exist
    if not os.path.exists(conversation_file):
        print(f"❌ Conversation file not found: {conversation_file}")
        print(f"\nUsage: python {sys.argv[0]} <conversation_file> [pca_model] [output_dir]")
        sys.exit(1)
    
    if not os.path.exists(pca_model_path):
        print(f"❌ PCA model not found: {pca_model_path}")
        print(f"\n⚠️  You need to train a PCA model on local embeddings first!")
        print(f"   Run: python src/train_pca_local.py")
        sys.exit(1)
    
    # Run demo
    precog, embedder = run_local_embedding_demo(
        conversation_file=conversation_file,
        pca_model_path=pca_model_path,
        output_dir=output_dir,
        embedding_model='all-MiniLM-L6-v2'
    )
