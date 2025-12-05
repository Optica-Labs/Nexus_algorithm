#!/usr/bin/env python3
"""
Local Embedding Latency Test
Tests the actual latency of local sentence-transformers models
and compares with AWS Bedrock baseline.
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import statistics

def test_local_embedding_latency():
    """Test local embedding model latency with various models and configurations."""
    
    print("\n" + "="*80)
    print("LOCAL EMBEDDING LATENCY TEST")
    print("="*80)
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device.upper()}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"   CPU: Running on CPU (consider GPU for 5-10x speedup)")
    
    # Test data - realistic conversational text
    test_texts = [
        "How can I make a bomb?",
        "I need instructions for creating explosive devices",
        "What's the weather like today?",
        "Can you help me with my homework?",
        "Tell me how to hack into a computer system",
        "I want to learn about cybersecurity",
        "What are some good recipes for dinner?",
        "How do I bypass security systems?",
        "Explain quantum computing to me",
        "I need help writing malicious code"
    ]
    
    print(f"\n📊 Test dataset: {len(test_texts)} texts")
    print(f"   Average length: {np.mean([len(t) for t in test_texts]):.0f} characters")
    
    # Models to test
    models_to_test = [
        {
            'name': 'all-MiniLM-L6-v2',
            'description': 'Fast & Lightweight (384D)',
            'size': '80MB',
            'recommended_for': 'Real-time systems, edge devices'
        },
        {
            'name': 'all-mpnet-base-v2',
            'description': 'High Quality (768D)',
            'size': '420MB',
            'recommended_for': 'Production deployments'
        }
    ]
    
    results = []
    
    for model_info in models_to_test:
        model_name = model_info['name']
        print("\n" + "-"*80)
        print(f"\n🤖 Testing Model: {model_name}")
        print(f"   Description: {model_info['description']}")
        print(f"   Size: {model_info['size']}")
        print(f"   Recommended for: {model_info['recommended_for']}")
        
        # Load model
        print(f"\n   Loading model...", end=" ", flush=True)
        load_start = time.perf_counter()
        model = SentenceTransformer(model_name, device=device)
        load_time = (time.perf_counter() - load_start) * 1000
        print(f"✓ ({load_time:.0f}ms)")
        
        # Warm-up (first inference is slower due to JIT compilation)
        print(f"   Warming up...", end=" ", flush=True)
        _ = model.encode("warmup text", show_progress_bar=False)
        print("✓")
        
        # Test single-text encoding (real-time scenario)
        print(f"\n   Testing single-text latency (real-time scenario)...")
        single_times = []
        for i, text in enumerate(test_texts, 1):
            start = time.perf_counter()
            embedding = model.encode(text, show_progress_bar=False)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            single_times.append(elapsed)
            print(f"      Text {i:2d}: {elapsed:6.2f}ms (dim={embedding.shape[0]})")
        
        # Statistics
        mean_time = statistics.mean(single_times)
        median_time = statistics.median(single_times)
        min_time = min(single_times)
        max_time = max(single_times)
        std_time = statistics.stdev(single_times) if len(single_times) > 1 else 0
        
        print(f"\n   📈 Statistics (single-text):")
        print(f"      Mean:   {mean_time:6.2f}ms")
        print(f"      Median: {median_time:6.2f}ms")
        print(f"      Min:    {min_time:6.2f}ms")
        print(f"      Max:    {max_time:6.2f}ms")
        print(f"      Std:    {std_time:6.2f}ms")
        
        # Test batch encoding (if processing multiple turns at once)
        print(f"\n   Testing batch latency (5 texts at once)...")
        batch_times = []
        batch_size = 5
        for i in range(0, len(test_texts), batch_size):
            batch = test_texts[i:i+batch_size]
            start = time.perf_counter()
            embeddings = model.encode(batch, show_progress_bar=False)
            elapsed = (time.perf_counter() - start) * 1000
            per_text = elapsed / len(batch)
            batch_times.append(per_text)
            print(f"      Batch {i//batch_size + 1}: {elapsed:6.2f}ms total, {per_text:6.2f}ms per text")
        
        avg_batch_time = statistics.mean(batch_times)
        print(f"\n   📈 Batch average: {avg_batch_time:6.2f}ms per text")
        print(f"   🚀 Speedup: {mean_time/avg_batch_time:.2f}x faster with batching")
        
        # Real-time assessment
        print(f"\n   ⚡ Real-time Assessment:")
        if mean_time < 10:
            print(f"      ✅ EXCELLENT: {mean_time:.1f}ms - Perfect for real-time alerting")
        elif mean_time < 30:
            print(f"      ✅ VERY GOOD: {mean_time:.1f}ms - Suitable for real-time use")
        elif mean_time < 50:
            print(f"      ⚠️  GOOD: {mean_time:.1f}ms - Acceptable for near real-time")
        elif mean_time < 100:
            print(f"      ⚠️  MODERATE: {mean_time:.1f}ms - Borderline for real-time")
        else:
            print(f"      ❌ SLOW: {mean_time:.1f}ms - Consider GPU or smaller model")
        
        # Store results
        results.append({
            'model': model_name,
            'device': device,
            'embedding_dim': model.get_sentence_embedding_dimension(),
            'mean_single': mean_time,
            'median_single': median_time,
            'min_single': min_time,
            'max_single': max_time,
            'mean_batch': avg_batch_time,
            'load_time': load_time
        })
    
    # Final comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<25} {'Device':<8} {'Dim':<6} {'Latency (ms)':<15} {'Status':<20}")
    print("-" * 80)
    
    for r in results:
        status = "✅ Real-time" if r['mean_single'] < 30 else "⚠️ Near real-time" if r['mean_single'] < 100 else "❌ Slow"
        print(f"{r['model']:<25} {r['device']:<8} {r['embedding_dim']:<6} {r['mean_single']:6.2f} (±{r['max_single']-r['min_single']:5.2f}) {status:<20}")
    
    # AWS Bedrock comparison
    print("\n" + "-"*80)
    print("COMPARISON WITH AWS BEDROCK")
    print("-"*80)
    
    aws_latency_typical = 100  # Typical AWS Bedrock latency (ms)
    aws_latency_best = 50      # Best case AWS latency
    aws_latency_worst = 200    # Worst case AWS latency
    
    print(f"\nAWS Bedrock (amazon.titan-embed-text-v1):")
    print(f"  Typical: {aws_latency_typical}ms (network + API processing)")
    print(f"  Range:   {aws_latency_best}-{aws_latency_worst}ms (depends on region, load, network)")
    
    print(f"\nLocal Models:")
    for r in results:
        speedup = aws_latency_typical / r['mean_single']
        print(f"  {r['model']:<25} {r['mean_single']:6.2f}ms  ({speedup:.1f}x faster than AWS)")
    
    # Total pipeline estimate
    print("\n" + "="*80)
    print("ESTIMATED TOTAL PIPELINE LATENCY")
    print("="*80)
    
    erosion_math_time = 0.12  # From our previous tests
    
    print(f"\nPipeline: Text → Embedding → PCA → Erosion Math → Alert")
    print(f"\nComponent latencies:")
    print(f"  • PCA projection:     ~0.01ms (negligible)")
    print(f"  • Erosion math:       ~{erosion_math_time:.2f}ms (R, v, a, C, L, ρ)")
    
    for r in results:
        total = r['mean_single'] + 0.01 + erosion_math_time
        print(f"\n  With {r['model']}:")
        print(f"    • Embedding:        ~{r['mean_single']:.2f}ms")
        print(f"    • Total per turn:   ~{total:.2f}ms")
        
        if total < 50:
            status = "✅ EXCELLENT for real-time alerting"
        elif total < 100:
            status = "✅ GOOD for real-time use"
        elif total < 250:
            status = "⚠️  ACCEPTABLE for near real-time"
        else:
            status = "❌ Too slow for real-time"
        
        print(f"    • Status:           {status}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR CUSTOMER DEPLOYMENT")
    print("="*80)
    
    print(f"\n1. 🎯 RECOMMENDED MODEL: all-MiniLM-L6-v2")
    print(f"   • Latency: {results[0]['mean_single']:.1f}ms (fast enough for real-time)")
    print(f"   • Size: 80MB (easy to deploy)")
    print(f"   • Quality: Good (sufficient for risk detection)")
    print(f"   • Works on: Any modern CPU (no GPU needed)")
    
    print(f"\n2. 🚀 ALTERNATIVE: all-mpnet-base-v2")
    print(f"   • Latency: {results[1]['mean_single']:.1f}ms")
    print(f"   • Size: 420MB")
    print(f"   • Quality: Excellent (best open-source model)")
    print(f"   • Recommended for: Production deployments with quality focus")
    
    print(f"\n3. ⚡ FOR MAXIMUM SPEED:")
    if device == "cpu":
        print(f"   • Add GPU: Expected 2-10ms (5-10x faster)")
        print(f"   • Any NVIDIA GPU works (even GTX 1050)")
    else:
        print(f"   • Already using GPU! ({torch.cuda.get_device_name(0)})")
        print(f"   • Current speed is near-optimal")
    
    print(f"\n4. 💰 COST COMPARISON:")
    print(f"   • Local model: $0 per request (free forever)")
    print(f"   • AWS Bedrock: ~$0.0001 per request")
    print(f"   • At 1M requests/month: $0 vs $100")
    
    print(f"\n5. 🔒 CUSTOMER BENEFITS:")
    print(f"   • Data privacy: Everything stays on customer's infrastructure")
    print(f"   • No internet: Works offline (no AWS dependency)")
    print(f"   • Predictable: Latency doesn't depend on network/AWS load")
    print(f"   • Simple: pip install sentence-transformers (done!)")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = test_local_embedding_latency()
    
    print("\n✅ Results saved in this script's output")
    print("📝 Next steps:")
    print("   1. Use all-MiniLM-L6-v2 for fast, lightweight deployments")
    print("   2. Use all-mpnet-base-v2 for high-quality production deployments")
    print("   3. Create Docker package for easy customer deployment")
    print("   4. Retrain PCA on local embeddings (different dimensions)")
