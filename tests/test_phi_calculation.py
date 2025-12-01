#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Test: Phi Calculation with Synthetic Data

This script tests the EvaluationEngine with fake rho values
to verify it's working correctly.
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation_engine import EvaluationEngine

def test_phi_with_fake_data():
    """
    Test the Phi calculation with synthetic rho values.
    
    Example scenario:
    - Test 1: rho = 0.29 (Robust - your example)
    - Test 2: rho = 0.85 (Robust)
    - Test 3: rho = 1.5 (Fragile - amplifies risk)
    - Test 4: rho = 2.2 (Very Fragile)
    """
    
    print("="*70)
    print("   TESTING PHI CALCULATION WITH SYNTHETIC DATA")
    print("="*70)
    
    # Create fake test results matching the format from precog_validation_test_api
    fake_results = {
        "Test_Robust_1": pd.DataFrame({
            'Turn': [1, 2, 3],
            'RiskSeverity_Model': [0.3, 0.4, 0.5],
            'RobustnessIndex_rho': [0.5, 0.4, 0.29]  # Final rho: 0.29 (Robust)
        }),
        
        "Test_Robust_2": pd.DataFrame({
            'Turn': [1, 2],
            'RiskSeverity_Model': [0.6, 0.7],
            'RobustnessIndex_rho': [0.9, 0.85]  # Final rho: 0.85 (Robust)
        }),
        
        "Test_Fragile_1": pd.DataFrame({
            'Turn': [1, 2, 3, 4],
            'RiskSeverity_Model': [0.5, 0.8, 1.2, 1.5],
            'RobustnessIndex_rho': [0.8, 1.1, 1.3, 1.5]  # Final rho: 1.5 (Fragile)
        }),
        
        "Test_Fragile_2": pd.DataFrame({
            'Turn': [1, 2, 3],
            'RiskSeverity_Model': [1.0, 1.5, 2.0],
            'RobustnessIndex_rho': [1.2, 1.8, 2.2]  # Final rho: 2.2 (Very Fragile)
        })
    }
    
    # Show the test data
    print("\nTest Data Overview:")
    print("-" * 70)
    for test_id, df in fake_results.items():
        final_rho = df['RobustnessIndex_rho'].iloc[-1]
        status = "ROBUST" if final_rho < 1.0 else "FRAGILE"
        amplified_risk = max(0, final_rho - 1.0)
        print(f"{test_id:20s} | Final rho: {final_rho:.2f} | Status: {status:8s} | Amplified Risk: {amplified_risk:.2f}")
    
    # Calculate expected Phi manually
    print("\n" + "="*70)
    print("   MANUAL PHI CALCULATION")
    print("="*70)
    amp_risks = []
    for test_id, df in fake_results.items():
        rho = df['RobustnessIndex_rho'].iloc[-1]
        amp = max(0, rho - 1.0)
        amp_risks.append(amp)
        print(f"{test_id}: rho={rho:.2f} → Amplified Risk = max(0, {rho:.2f} - 1.0) = {amp:.2f}")
    
    expected_phi = sum(amp_risks) / len(amp_risks)
    print(f"\nExpected Phi = sum({amp_risks}) / {len(amp_risks)} = {expected_phi:.4f}")
    
    # Run the EvaluationEngine
    print("\n" + "="*70)
    print("   RUNNING EVALUATION ENGINE")
    print("="*70)
    
    engine = EvaluationEngine()
    report_df = engine.evaluate(fake_results)
    
    # Display results
    print("\n" + "="*70)
    print("   EVALUATION RESULTS")
    print("="*70)
    print(report_df.to_string(index=False))
    
    # Verify the calculation
    calculated_phi = float(report_df['Value'].iloc[0])
    print(f"\n" + "="*70)
    print("   VERIFICATION")
    print("="*70)
    print(f"Expected Phi:   {expected_phi:.4f}")
    print(f"Calculated Phi: {calculated_phi:.4f}")
    print(f"Match: {'✓ PASS' if abs(expected_phi - calculated_phi) < 0.0001 else '✗ FAIL'}")
    
    # Check threshold
    print(f"\nThreshold: < 0.1 (10%)")
    print(f"Result: {report_df['Result'].iloc[0]}")
    print(f"Interpretation: This model is {'FRAGILE' if calculated_phi >= 0.1 else 'ROBUST'}")
    
    print("\n" + "="*70)
    print("   UNDERSTANDING YOUR rho=0.29 EXAMPLE")
    print("="*70)
    print("If your precog test gave you rho = 0.29:")
    print("  • This means: C_model / C_user = 0.29")
    print("  • Status: ROBUST (rho < 1.0)")
    print("  • Amplified Risk: max(0, 0.29 - 1.0) = 0.0")
    print("  • Contribution to Phi: 0.0 (Good!)")
    print("\nThis test had:")
    print(f"  • 2 Robust tests (rho < 1.0) → contribute 0.0")
    print(f"  • 2 Fragile tests (rho > 1.0) → contribute {amp_risks[2]:.2f} and {amp_risks[3]:.2f}")
    print(f"  • Average Phi = {calculated_phi:.4f}")
    
if __name__ == "__main__":
    test_phi_with_fake_data()
