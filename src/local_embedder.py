#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Local Embedding Model for Customer Deployment

This module provides a drop-in replacement for AWS Bedrock embeddings
using open-source sentence-transformers models that run locally.

Advantages:
- No AWS API calls (lower latency: 10-30ms vs 50-200ms)
- Works offline
- Customer controls data (privacy)
- No per-request costs
- Can run on customer infrastructure

Usage:
    from local_embedder import LocalEmbedder
    
    embedder = LocalEmbedder('all-mpnet-base-v2')
    embedding = embedder.embed_text("Hello world")
"""

import numpy as np
import time
from typing import List, Optional
import os


class LocalEmbedder:
    """
    Local embedding model using sentence-transformers.
    
    Compatible models:
    - 'all-MiniLM-L6-v2': Fast, 384D (5-10ms)
    - 'all-mpnet-base-v2': Best quality, 768D (10-30ms) [RECOMMENDED]
    - 'all-MiniLM-L12-v2': Balanced, 384D (8-15ms)
    """
    
    def __init__(self, 
                 model_name: str = 'all-mpnet-base-v2',
                 device: str = 'cpu',
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True):
        """
        Initialize local embedding model.
        
        Args:
            model_name: HuggingFace model name
            device: 'cpu' or 'cuda' for GPU
            cache_dir: Directory to cache model files
            use_cache: Enable embedding cache for repeated texts
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        
        print(f"\n{'='*70}")
        print(f"Initializing Local Embedding Model")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"Cache: {'Enabled' if use_cache else 'Disabled'}")
        
        # Load model
        t0 = time.time()
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
        t1 = time.time()
        
        print(f"✓ Model loaded in {t1-t0:.2f}s")
        print(f"✓ Embedding dimension: {self.model.get_sentence_embedding_dimension()}D")
        print(f"{'='*70}\n")
        
        self.model_name = model_name
        self.device = device
        
        # Embedding cache
        self.cache = {} if use_cache else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Statistics
        self.total_requests = 0
        self.total_time = 0.0
    
    def embed_text(self, text: str, show_timing: bool = False) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            show_timing: Print timing information
            
        Returns:
            Numpy array with embedding vector
        """
        # Check cache
        if self.cache is not None and text in self.cache:
            self.cache_hits += 1
            self.total_requests += 1
            if show_timing:
                print(f"  ⚡ Cache hit (0.000ms)")
            return self.cache[text]
        
        # Generate embedding
        t0 = time.perf_counter()
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        t1 = time.perf_counter()
        
        elapsed_ms = (t1 - t0) * 1000
        
        # Update statistics
        self.cache_misses += 1
        self.total_requests += 1
        self.total_time += elapsed_ms
        
        # Store in cache
        if self.cache is not None:
            self.cache[text] = embedding
        
        if show_timing:
            print(f"  ⚙️  Embedding: {elapsed_ms:.2f}ms")
        
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, 
                   show_timing: bool = False) -> np.ndarray:
        """
        Generate embeddings for multiple texts (more efficient than one-by-one).
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process together
            show_timing: Print timing information
            
        Returns:
            Numpy array with embeddings (N x embedding_dim)
        """
        t0 = time.perf_counter()
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False
        )
        t1 = time.perf_counter()
        
        elapsed_ms = (t1 - t0) * 1000
        per_text_ms = elapsed_ms / len(texts)
        
        self.total_requests += len(texts)
        self.total_time += elapsed_ms
        
        if show_timing:
            print(f"  ⚙️  Batch embedding: {elapsed_ms:.2f}ms total, "
                  f"{per_text_ms:.2f}ms per text")
        
        return embeddings
    
    def get_statistics(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with stats
        """
        avg_time = self.total_time / self.total_requests if self.total_requests > 0 else 0
        
        stats = {
            'total_requests': self.total_requests,
            'total_time_ms': self.total_time,
            'avg_time_ms': avg_time,
            'model_name': self.model_name,
            'device': self.device
        }
        
        if self.cache is not None:
            total_cache_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
            stats.update({
                'cache_enabled': True,
                'cache_size': len(self.cache),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': hit_rate
            })
        else:
            stats['cache_enabled'] = False
        
        return stats
    
    def print_statistics(self):
        """Print performance statistics."""
        stats = self.get_statistics()
        
        print(f"\n{'='*70}")
        print("LOCAL EMBEDDER STATISTICS")
        print(f"{'='*70}")
        print(f"Model: {stats['model_name']}")
        print(f"Device: {stats['device']}")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Total time: {stats['total_time_ms']:.2f}ms")
        print(f"Average time: {stats['avg_time_ms']:.2f}ms per request")
        
        if stats['cache_enabled']:
            print(f"\nCache Statistics:")
            print(f"  Cache size: {stats['cache_size']} entries")
            print(f"  Cache hits: {stats['cache_hits']}")
            print(f"  Cache misses: {stats['cache_misses']}")
            print(f"  Hit rate: {stats['cache_hit_rate']:.1f}%")
        
        print(f"{'='*70}\n")
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache is not None:
            self.cache.clear()
            print("✓ Cache cleared")
        else:
            print("Cache is disabled")


def compare_models():
    """Compare different embedding models for speed and quality."""
    print("\n" + "="*70)
    print("COMPARING LOCAL EMBEDDING MODELS")
    print("="*70 + "\n")
    
    models_to_test = [
        ('all-MiniLM-L6-v2', 'Fast & Small'),
        ('all-mpnet-base-v2', 'Best Quality'),
        ('all-MiniLM-L12-v2', 'Balanced'),
    ]
    
    test_text = "This is a test message for evaluating embedding model performance."
    
    results = []
    
    for model_name, description in models_to_test:
        print(f"\nTesting: {model_name} ({description})")
        print("-" * 70)
        
        try:
            embedder = LocalEmbedder(model_name, use_cache=False)
            
            # Warmup
            _ = embedder.embed_text(test_text)
            
            # Benchmark
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                embedding = embedder.embed_text(test_text)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results.append({
                'model': model_name,
                'description': description,
                'dimension': len(embedding),
                'avg_ms': avg_time,
                'std_ms': std_time
            })
            
            print(f"✓ Dimension: {len(embedding)}D")
            print(f"✓ Average: {avg_time:.2f}ms (±{std_time:.2f}ms)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<30} {'Dim':<8} {'Avg Time':<12} {'Description'}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<30} {r['dimension']:<8} {r['avg_ms']:>8.2f}ms    {r['description']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test local embedding models')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare different models')
    parser.add_argument('--model', type=str, default='all-mpnet-base-v2',
                       help='Model to test')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device: cpu or cuda')
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        # Simple test
        print("\nTesting local embedder...")
        embedder = LocalEmbedder(args.model, device=args.device)
        
        test_texts = [
            "Hello, how are you?",
            "I need help with Python programming.",
            "What's the weather like today?",
            "Hello, how are you?",  # Duplicate to test cache
        ]
        
        print("\nGenerating embeddings:")
        for i, text in enumerate(test_texts, 1):
            print(f"\n{i}. '{text}'")
            embedding = embedder.embed_text(text, show_timing=True)
            print(f"   Shape: {embedding.shape}, First 5 values: {embedding[:5]}")
        
        embedder.print_statistics()
