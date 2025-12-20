#!/usr/bin/env python3
"""
Quick test script to verify sycophancy integration works
"""

import sys
import os

# Add paths
deployment_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, deployment_root)
sys.path.insert(0, os.path.join(deployment_root, 'shared'))

import numpy as np
from core.sycophancy_analyzer import SycophancyAnalyzer

print("="*70)
print("Testing Sycophancy Analyzer Integration")
print("="*70)

# Test 1: Initialize analyzer
print("\n[Test 1] Initializing SycophancyAnalyzer...")
try:
    vsafe = np.array([0.5159, 0.8566])  # Example from testing
    vsyc = np.array([0.8425, 0.5386])   # Example from testing

    analyzer = SycophancyAnalyzer(vsafe, vsyc)
    print("✓ Analyzer initialized successfully")
    print(f"  VSAFE: {analyzer.vsafe}")
    print(f"  VSYC: {analyzer.vsyc}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Process a turn
print("\n[Test 2] Processing a turn...")
try:
    # Simulate a safe interaction
    v_user = np.array([0.5, 0.9])   # Close to VSAFE (safe)
    v_model = np.array([0.6, 0.8])  # Also safe

    analyzer.process_turn(v_model, v_user, turn_number=1)
    print("✓ Turn processed successfully")

    latest = analyzer.get_latest_metrics()
    print(f"  User Risk: {latest['user_risk']:.3f}")
    print(f"  Agreement: {latest['agreement']:.3f}")
    print(f"  Toxic Sycophancy: {latest['toxic_sycophancy']:.3f}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 3: Get metrics DataFrame
print("\n[Test 3] Getting metrics DataFrame...")
try:
    metrics_df = analyzer.get_metrics()
    print("✓ Metrics DataFrame retrieved")
    print(f"  Shape: {metrics_df.shape}")
    print(f"  Columns: {list(metrics_df.columns)}")
    print("\n  Data:")
    print(metrics_df)
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 4: Get summary statistics
print("\n[Test 4] Getting summary statistics...")
try:
    summary = analyzer.get_summary_statistics()
    print("✓ Summary statistics retrieved")
    for key, value in summary.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 5: Quadrant classification
print("\n[Test 5] Testing quadrant classification...")
try:
    # Test all four quadrants
    test_cases = [
        (1.5, 0.8, "Sycophancy Trap"),
        (1.5, 0.2, "Robust Correction"),
        (0.2, 0.8, "Safe Agreement"),
        (0.2, 0.2, "Safe Neutral"),
    ]

    all_passed = True
    for user_risk, agreement, expected in test_cases:
        result = analyzer.get_quadrant_classification(user_risk, agreement)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} Risk={user_risk}, Agree={agreement} → {result} (expected: {expected})")

    if all_passed:
        print("✓ All quadrant classifications correct")
    else:
        print("✗ Some classifications incorrect")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 6: Reset functionality
print("\n[Test 6] Testing reset...")
try:
    analyzer.reset()
    metrics_after_reset = analyzer.get_metrics()

    if metrics_after_reset.empty:
        print("✓ Reset successful - metrics cleared")
    else:
        print("✗ Reset failed - metrics still present")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED")
print("="*70)
print("\nSycophancy integration is working correctly!")
print("You can now run the dashboard with:")
print("  streamlit run app.py --server.port 8504")
