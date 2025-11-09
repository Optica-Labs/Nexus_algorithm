#!/usr/bin/env python3
"""
Test script to verify the complete pipeline works end-to-end.
Run this after setting up AWS credentials and training PCA models.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_manual_mode():
    """Test with hardcoded 2D vectors (no AWS required)."""
    print("\n" + "="*70)
    print("TEST 1: Manual Mode (Hardcoded Vectors)")
    print("="*70)
    
    from vector_precognition_demo import VectorPrecogntion
    import numpy as np
    
    # Setup
    vsafe = np.array([-0.5, 0.5])
    weights = {'wR': 1.5, 'wv': 1.0, 'wa': 3.0, 'b': -2.5}
    processor = VectorPrecogntion(vsafe=vsafe, weights=weights)
    
    # Process some vectors
    test_vectors = [
        np.array([-0.4, 0.4]),
        np.array([0.1, 0.2]),
        np.array([0.5, 0.0]),
        np.array([-0.3, 0.4])
    ]
    
    for vec in test_vectors:
        processor.process_turn(vec)
    
    # Get results
    metrics = processor.get_metrics()
    print(f"\n✓ Processed {len(metrics)} turns")
    print(f"  Peak Risk Severity: {metrics['RiskSeverity_R(N)'].max():.3f}")
    print(f"  Peak Likelihood: {metrics['Likelihood_L(N)'].max():.3f}")
    print("\n✓ Manual mode test PASSED\n")
    return True


def test_embedding_module():
    """Test AWS Bedrock embedding generation."""
    print("\n" + "="*70)
    print("TEST 2: Embedding Module (Requires AWS)")
    print("="*70)
    
    try:
        from embeddings import get_titan_embedding
        
        test_text = "I'm here to help with appropriate requests."
        print(f"\nGenerating embedding for: '{test_text}'")
        
        embedding = get_titan_embedding(test_text)
        
        if embedding is not None:
            print(f"✓ Generated embedding with dimension: {len(embedding)}")
            print(f"✓ Embedding module test PASSED\n")
            return True
        else:
            print("✗ Failed to generate embedding")
            print("  Check AWS credentials and Bedrock access\n")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        print("  Make sure AWS credentials are configured\n")
        return False


def test_pca_pipeline():
    """Test the complete text-to-2D pipeline."""
    print("\n" + "="*70)
    print("TEST 3: Text-to-2D Pipeline (Requires AWS + Trained Models)")
    print("="*70)
    
    try:
        from text_to_2d import TextTo2DPipeline
        
        # Initialize pipeline
        print("\nInitializing pipeline...")
        pipeline = TextTo2DPipeline()
        
        # Test conversion
        test_texts = [
            "I'm designed to be helpful and harmless.",
            "I can provide some information on that topic.",
            "I cannot assist with harmful requests."
        ]
        
        print("\nConverting texts to 2D vectors:")
        all_success = True
        
        for i, text in enumerate(test_texts):
            vec = pipeline.get_2d_vector(text)
            if vec is not None:
                print(f"  {i+1}. '{text[:40]}...'")
                print(f"     → [{vec[0]:.4f}, {vec[1]:.4f}]")
            else:
                print(f"  {i+1}. FAILED: '{text[:40]}...'")
                all_success = False
        
        if all_success:
            print("\n✓ Text-to-2D pipeline test PASSED\n")
            return True
        else:
            print("\n✗ Some conversions failed\n")
            return False
            
    except FileNotFoundError as e:
        print(f"\n✗ Error: {str(e)}")
        print("  Run 'python src/pca_trainer.py' first to train models\n")
        return False
    except Exception as e:
        print(f"\n✗ Error: {str(e)}\n")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("VECTOR PRECOGNITION - PIPELINE TESTS")
    print("="*70)
    
    results = {
        'Manual Mode': test_manual_mode(),
        'Embedding Module': test_embedding_module(),
        'Text-to-2D Pipeline': test_pca_pipeline()
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests PASSED! System is fully operational.\n")
    elif results['Manual Mode']:
        print("\n⚠  Manual mode works. Set up AWS for full functionality.\n")
    else:
        print("\n✗ Some tests failed. Check the errors above.\n")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
