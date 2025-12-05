#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script to demonstrate latency measurements in the guardrail erosion pipeline.
Uses synthetic 2D vectors to avoid needing AWS credentials.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from vector_precognition_demo2 import VectorPrecogntion

def main():
    """Test the latency measurement features with synthetic data."""
    
    print("\n" + "="*80)
    print("GUARDRAIL EROSION PIPELINE - LATENCY TEST")
    print("="*80)
    print("\nThis test uses synthetic 2D vectors to measure computational latency")
    print("for the guardrail erosion calculations (R, v, a, C, L, rho).\n")
    
    # Define VSAFE (safe harbor vector)
    VSAFE = np.array([-0.5, 0.5])
    print(f"VSAFE (safe harbor): [{VSAFE[0]:.4f}, {VSAFE[1]:.4f}]")
    
    # Define weights for likelihood calculation
    WEIGHTS = {
        'wR': 1.5,   # Weight for Risk Severity
        'wv': 1.0,   # Weight for Velocity
        'wa': 3.0,   # Weight for Acceleration/Erosion
        'b': -2.5    # Bias
    }
    
    # Initialize the VectorPrecognition processor
    processor = VectorPrecogntion(vsafe=VSAFE, weights=WEIGHTS, epsilon=0.1)
    
    # Generate synthetic conversation: safe → risky transition
    print("\nSimulating a 10-turn conversation (safe → risky drift)...\n")
    
    np.random.seed(42)  # For reproducibility
    
    # Simulate a gradual drift from safe to unsafe
    for turn in range(10):
        # User vectors: gradually becoming more provocative
        user_risk_factor = turn / 10.0
        user_x = -0.5 + (user_risk_factor * 1.5) + np.random.normal(0, 0.1)
        user_y = 0.5 - (user_risk_factor * 1.0) + np.random.normal(0, 0.1)
        user_vector = np.array([user_x, user_y])
        
        # Model vectors: following the user with some resistance
        model_risk_factor = turn / 15.0  # More resistant than user
        model_x = -0.5 + (model_risk_factor * 1.2) + np.random.normal(0, 0.08)
        model_y = 0.5 - (model_risk_factor * 0.8) + np.random.normal(0, 0.08)
        model_vector = np.array([model_x, model_y])
        
        # Process the turn
        processor.process_turn(v_model=model_vector, v_user=user_vector)
        
        print(f"Turn {turn + 1}: User [{user_x:>6.3f}, {user_y:>6.3f}] → "
              f"Model [{model_x:>6.3f}, {model_y:>6.3f}]")
    
    # Get and display metrics
    print("\n" + "="*80)
    print("RISK METRICS")
    print("="*80 + "\n")
    
    metrics = processor.get_metrics()
    print(metrics[['RiskSeverity_Model', 'RiskRate_Model', 'ErosionVelocity_Model', 
                   'Likelihood_Model', 'RobustnessIndex_rho']].to_string(float_format="%.4f"))
    
    # Print latency report
    processor.print_latency_report()
    
    # Summary
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("\n1. The latency measurements show ONLY the mathematical operations:")
    print("   - Cosine distance calculation (R)")
    print("   - First derivative (velocity v)")
    print("   - Second derivative (acceleration/erosion a)")
    print("   - Cumulative risk integration (C)")
    print("   - Likelihood calculation (L)")
    print("   - Robustness index (rho)")
    print("\n2. This does NOT include:")
    print("   - Text vectorization/embedding time (AWS Bedrock API)")
    print("   - PCA transformation time")
    print("   - Network latency")
    print("\n3. For real-world deployment, you need to add:")
    print("   - Embedding latency (~50-200ms for AWS Bedrock)")
    print("   - PCA transformation (~1-5ms)")
    print("   - Network overhead (~10-50ms)")
    print("\n4. Total expected real-time latency:")
    print(f"   - Math operations: ~{metrics.index.size * processor.timing_data['total_turn_ms'][-1] / metrics.index.size:.3f}ms per turn")
    print("   - With embeddings: ~100-300ms per turn (estimated)")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
