import numpy as np
import pandas as pd
from scipy import stats
from evaluation_engine import EvaluationEngine

def run_validation_study():
    print("--- STARTING PHI SCORE VALIDATION STUDY ---")
    engine = EvaluationEngine()
    
    # --- TEST 1: H2 (Zero-Floor Condition) ---
    print("\n[TEST 1] H2: Zero-Floor Validation (Synthetic 'Saint' Data)")
    # Generate 1000 'robust' rho values (0.0 to 1.0)
    saint_rhos = np.random.uniform(0.0, 1.0, 1000)
    
    # Mock the input format expected by the engine
    saint_results = {
        f"test_{i}": pd.DataFrame({'RobustnessIndex_rho': [rho]}) 
        for i, rho in enumerate(saint_rhos)
    }
    
    phi_saint = engine.calculate_overall_fragility(saint_results)
    print(f"  Phi Score (Saint): {phi_saint:.6f}")
    
    if phi_saint == 0.0:
        print("  RESULT: PASS (Score is exactly 0.0)")
    else:
        print("  RESULT: FAIL (Score should be 0.0)")


    # --- TEST 2: H1 (Discriminatory Validity) ---
    print("\n[TEST 2] H1: Discriminatory Validity (Synthetic A/B Test)")
    
    # Model A (Robust): Mostly < 1.0, rare spikes
    # Distribution: Normal(0.5, 0.2)
    rhos_a = np.random.normal(0.5, 0.2, 500)
    # Clamp to ensure some physical realism (can't be negative)
    rhos_a = np.maximum(rhos_a, 0.0) 
    
    # Model B (Fragile): Skewed > 1.0
    # Distribution: Normal(1.5, 0.5)
    rhos_b = np.random.normal(1.5, 0.5, 500)
    rhos_b = np.maximum(rhos_b, 0.0)

    # Convert to engine format
    results_a = {f"t_{i}": pd.DataFrame({'RobustnessIndex_rho': [r]}) for i, r in enumerate(rhos_a)}
    results_b = {f"t_{i}": pd.DataFrame({'RobustnessIndex_rho': [r]}) for i, r in enumerate(rhos_b)}

    # Calculate Phis
    phi_a = engine.calculate_overall_fragility(results_a)
    phi_b = engine.calculate_overall_fragility(results_b)
    
    print(f"  Phi Model A (Robust):  {phi_a:.4f}")
    print(f"  Phi Model B (Fragile): {phi_b:.4f}")
    
    # Perform T-Test on the Amplified Risks
    # We must extract the raw R_amp values (max(0, rho-1)) for the stat test
    amp_a = [max(0, r-1) for r in rhos_a]
    amp_b = [max(0, r-1) for r in rhos_b]
    
    t_stat, p_val = stats.ttest_ind(amp_a, amp_b, equal_var=False)
    
    print(f"  Statistical Significance (T-Test): p-value = {p_val:.4e}")
    
    if p_val < 0.01 and phi_b > phi_a:
        print("  RESULT: PASS (Statistically Significant Differentiation)")
    else:
        print("  RESULT: FAIL (Indistinguishable)")

    # --- TEST 3: H4 (Edge Case Stability) ---
    print("\n[TEST 3] H4: Edge Case / Benign User (Epsilon Check)")
    # Scenario: User Risk is 0 (perfectly benign), Model Risk is 0.5 (mild drift)
    # rho = 0.5 / (0 + epsilon)
    # If epsilon is 0.1, rho = 5.0 (High Amplification).
    # This validates that the engine catches 'unprompted drift'.
    
    metrics_edge = pd.DataFrame({
        'CumulativeRisk_Model': [0.5],
        'CumulativeRisk_User': [0.0],
        'RobustnessIndex_rho': [0.5 / 0.1] # Assuming epsilon=0.1
    })
    
    edge_results = {"edge_case": metrics_edge}
    phi_edge = engine.calculate_overall_fragility(edge_results)
    
    print(f"  Phi Score (Zero User Risk): {phi_edge:.4f}")
    
    if phi_edge > 0:
         print("  RESULT: PASS (Engine successfully flagged unprompted drift)")
    else:
         print("  RESULT: FAIL (Engine ignored unprompted drift)")

if __name__ == "__main__":
    run_validation_study()
