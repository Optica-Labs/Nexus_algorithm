#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Latency Profiler for Guardrail Erosion Pipeline

Measures the latency for each step in the risk calculation pipeline:
1. Text input (model/user response)
2. Vectorization (embedding)
3. PCA transformation
4. Cosine distance calculation (Risk Score R(N))
5. First derivative (Velocity v(N))
6. Second derivative (Acceleration/Erosion a(N))
7. Plotting (optional - can be disabled for production)

This helps determine if the system is fast enough for real-time alerting.
"""

import time
import numpy as np
import json
import os
import sys
from typing import Dict, List, Tuple
import statistics

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class LatencyProfiler:
    """Profiles the latency of each step in the guardrail erosion pipeline."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {
            'vectorization': [],
            'pca_transform': [],
            'cosine_distance': [],
            'velocity_calc': [],
            'acceleration_calc': [],
            'total_per_turn': []
        }
        
        # Store historical data for derivative calculations
        self.R_history: List[float] = []
        self.v_history: List[float] = []
        
    def profile_single_turn(self, text: str, vsafe: np.ndarray, 
                           embedder, pca_model) -> Dict[str, float]:
        """
        Profile a single turn through the entire pipeline.
        
        Args:
            text: Input text (user or model message)
            vsafe: The safe harbor vector (2D)
            embedder: Embedding model instance
            pca_model: Trained PCA model
            
        Returns:
            Dictionary with timing for each step
        """
        turn_timings = {}
        turn_start = time.perf_counter()
        
        # Step 1: Vectorization (Embedding)
        t0 = time.perf_counter()
        embedding = embedder.embed_text(text)
        t1 = time.perf_counter()
        turn_timings['vectorization'] = (t1 - t0) * 1000  # Convert to ms
        
        # Step 2: PCA transformation
        t0 = time.perf_counter()
        embedding_2d = pca_model.transform([embedding])[0]
        t1 = time.perf_counter()
        turn_timings['pca_transform'] = (t1 - t0) * 1000
        
        # Step 3: Cosine distance (Risk Score R(N))
        t0 = time.perf_counter()
        R_n = self._calculate_cosine_distance(embedding_2d, vsafe)
        t1 = time.perf_counter()
        turn_timings['cosine_distance'] = (t1 - t0) * 1000
        
        self.R_history.append(R_n)
        
        # Step 4: First derivative (Velocity v(N))
        t0 = time.perf_counter()
        if len(self.R_history) == 1:
            v_n = 0.0
        else:
            v_n = self.R_history[-1] - self.R_history[-2]
        t1 = time.perf_counter()
        turn_timings['velocity_calc'] = (t1 - t0) * 1000
        
        self.v_history.append(v_n)
        
        # Step 5: Second derivative (Acceleration/Erosion a(N))
        t0 = time.perf_counter()
        if len(self.v_history) <= 1:
            a_n = 0.0
        else:
            if len(self.v_history) == 2:
                a_n = self.v_history[-1] - 0.0
            else:
                a_n = self.v_history[-1] - self.v_history[-2]
        t1 = time.perf_counter()
        turn_timings['acceleration_calc'] = (t1 - t0) * 1000
        
        turn_end = time.perf_counter()
        turn_timings['total_per_turn'] = (turn_end - turn_start) * 1000
        
        # Store timings
        for key, value in turn_timings.items():
            self.timings[key].append(value)
        
        return turn_timings
    
    def _calculate_cosine_distance(self, v_n: np.ndarray, vsafe: np.ndarray) -> float:
        """Calculate cosine distance between v_n and vsafe."""
        v_n_norm = np.linalg.norm(v_n)
        vsafe_norm = np.linalg.norm(vsafe)
        
        if v_n_norm == 0 or vsafe_norm == 0:
            return 1.0
        
        v_n_unit = v_n / v_n_norm
        vsafe_unit = vsafe / vsafe_norm
        
        cosine_similarity = np.dot(v_n_unit, vsafe_unit)
        cosine_distance = 1.0 - cosine_similarity
        
        return np.clip(cosine_distance, 0.0, 2.0)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for all timing measurements.
        
        Returns:
            Dictionary with mean, median, min, max, std for each step
        """
        stats = {}
        
        for step, timings in self.timings.items():
            if not timings:
                continue
                
            stats[step] = {
                'mean_ms': statistics.mean(timings),
                'median_ms': statistics.median(timings),
                'min_ms': min(timings),
                'max_ms': max(timings),
                'std_ms': statistics.stdev(timings) if len(timings) > 1 else 0.0,
                'p95_ms': self._percentile(timings, 95),
                'p99_ms': self._percentile(timings, 99)
            }
        
        return stats
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (len(sorted_data) - 1) * percentile / 100
        floor = int(index)
        ceil = floor + 1
        if ceil >= len(sorted_data):
            return sorted_data[floor]
        return sorted_data[floor] + (sorted_data[ceil] - sorted_data[floor]) * (index - floor)
    
    def print_report(self):
        """Print a detailed latency report."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("GUARDRAIL EROSION PIPELINE - LATENCY PROFILE")
        print("=" * 80)
        print(f"\nTotal turns processed: {len(self.timings['total_per_turn'])}\n")
        
        # Print header
        print(f"{'Step':<25} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'P95':<10} {'P99':<10}")
        print("-" * 80)
        
        # Order of steps to display
        step_order = [
            'vectorization',
            'pca_transform',
            'cosine_distance',
            'velocity_calc',
            'acceleration_calc',
            'total_per_turn'
        ]
        
        step_names = {
            'vectorization': '1. Vectorization',
            'pca_transform': '2. PCA Transform',
            'cosine_distance': '3. Cosine Distance (R)',
            'velocity_calc': '4. Velocity (v)',
            'acceleration_calc': '5. Erosion (a)',
            'total_per_turn': 'TOTAL PER TURN'
        }
        
        for step in step_order:
            if step in stats:
                s = stats[step]
                name = step_names.get(step, step)
                print(f"{name:<25} {s['mean_ms']:>8.3f}ms {s['median_ms']:>8.3f}ms "
                      f"{s['min_ms']:>8.3f}ms {s['max_ms']:>8.3f}ms "
                      f"{s['p95_ms']:>8.3f}ms {s['p99_ms']:>8.3f}ms")
        
        print("=" * 80)
        
        # Calculate percentage breakdown
        total_mean = stats['total_per_turn']['mean_ms']
        print(f"\n{'PERCENTAGE BREAKDOWN':<40}")
        print("-" * 80)
        for step in step_order[:-1]:  # Exclude total
            if step in stats:
                percentage = (stats[step]['mean_ms'] / total_mean) * 100
                name = step_names.get(step, step)
                print(f"{name:<25} {stats[step]['mean_ms']:>8.3f}ms ({percentage:>5.1f}%)")
        
        print("=" * 80)
        
        # Real-time suitability analysis
        print("\nREAL-TIME ALERTING SUITABILITY ANALYSIS")
        print("-" * 80)
        
        p95_total = stats['total_per_turn']['p95_ms']
        p99_total = stats['total_per_turn']['p99_ms']
        
        print(f"Average latency per turn: {total_mean:.3f}ms")
        print(f"95th percentile latency: {p95_total:.3f}ms")
        print(f"99th percentile latency: {p99_total:.3f}ms")
        print()
        
        # Define latency thresholds for different use cases
        thresholds = {
            'Real-time alerting (<100ms)': 100,
            'Near real-time (<250ms)': 250,
            'Interactive (<500ms)': 500,
            'Acceptable (<1000ms)': 1000
        }
        
        for category, threshold in thresholds.items():
            if p95_total <= threshold:
                print(f"✓ {category}: SUITABLE")
            else:
                print(f"✗ {category}: NOT SUITABLE (exceeds by {p95_total - threshold:.1f}ms)")
        
        print("\n" + "=" * 80)
        
        # Optimization recommendations
        print("\nOPTIMIZATION RECOMMENDATIONS")
        print("-" * 80)
        
        bottleneck = max(stats.items(), key=lambda x: x[1]['mean_ms'] if x[0] != 'total_per_turn' else 0)
        print(f"Bottleneck: {step_names.get(bottleneck[0], bottleneck[0])} "
              f"({bottleneck[1]['mean_ms']:.3f}ms, {(bottleneck[1]['mean_ms']/total_mean)*100:.1f}%)")
        
        if stats['vectorization']['mean_ms'] / total_mean > 0.5:
            print("• Consider caching embeddings or using a faster embedding model")
        
        if stats['pca_transform']['mean_ms'] > 10:
            print("• PCA transform is slow - ensure model is optimized")
        
        if total_mean < 100:
            print("• Latency is excellent for real-time alerting!")
        elif total_mean < 250:
            print("• Latency is good for near real-time alerting")
        else:
            print("• Consider async processing or optimization for real-time use")
        
        print("=" * 80 + "\n")


def main():
    """Main function to run latency profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Profile latency of the Guardrail Erosion Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile using a conversation JSON file
  python latency_profiler.py --conversation input/safe_conversation.json --turns 10
  
  # Profile with custom number of turns
  python latency_profiler.py --conversation input/jailbreak.json --turns 20
  
  # Profile using manual mode (skip embedding if vectors are pre-computed)
  python latency_profiler.py --conversation input/test.json --manual
        """
    )
    
    parser.add_argument('--conversation', type=str, required=True,
                        help='Path to conversation JSON file')
    parser.add_argument('--turns', type=int, default=None,
                        help='Number of turns to profile (default: all turns in conversation)')
    parser.add_argument('--manual', action='store_true',
                        help='Use manual mode (skip embedding step)')
    parser.add_argument('--vsafe-text', type=str,
                        default="I'm designed to be helpful, harmless, and honest.",
                        help='Text to define VSAFE vector')
    
    args = parser.parse_args()
    
    # Load conversation
    print(f"\nLoading conversation from: {args.conversation}")
    
    try:
        with open(args.conversation, 'r') as f:
            data = json.load(f)
        
        # Support both 'conversation' and 'turns' keys
        turns_data = data.get('conversation') or data.get('turns')
        if not turns_data:
            print("Error: JSON file must contain a 'conversation' or 'turns' key")
            return
        
        # Extract model messages (support both 'llm' and 'ai' speaker names)
        messages = []
        for turn in turns_data:
            speaker = turn.get('speaker', '').lower()
            message = turn.get('message', '')
            if speaker in ['llm', 'ai', 'model'] and message:
                messages.append(message)
        
        if not messages:
            print("Error: No LLM messages found in conversation")
            return
        
        # Limit turns if specified
        if args.turns:
            messages = messages[:args.turns]
        
        print(f"Found {len(messages)} model messages to profile\n")
        
    except Exception as e:
        print(f"Error loading conversation: {e}")
        return
    
    # Initialize profiler
    profiler = LatencyProfiler()
    
    if args.manual:
        print("Manual mode: Using synthetic 2D vectors (embedding step skipped)")
        print("Note: This doesn't accurately reflect real-world latency\n")
        
        # Use hardcoded VSAFE for manual mode
        VSAFE = np.array([-0.5, 0.5])
        
        # Simulate with random 2D vectors
        for i, message in enumerate(messages):
            print(f"Processing turn {i+1}/{len(messages)}...", end='\r')
            
            # Generate random 2D vector to simulate pre-computed embeddings
            synthetic_embedding = np.random.randn(2)
            
            # Manually time each step
            turn_start = time.perf_counter()
            
            # PCA (skip - already 2D)
            t0 = time.perf_counter()
            embedding_2d = synthetic_embedding
            t1 = time.perf_counter()
            profiler.timings['vectorization'].append(0.0)  # Not used
            profiler.timings['pca_transform'].append((t1 - t0) * 1000)
            
            # Cosine distance
            t0 = time.perf_counter()
            R_n = profiler._calculate_cosine_distance(embedding_2d, VSAFE)
            t1 = time.perf_counter()
            profiler.timings['cosine_distance'].append((t1 - t0) * 1000)
            profiler.R_history.append(R_n)
            
            # Velocity
            t0 = time.perf_counter()
            if len(profiler.R_history) == 1:
                v_n = 0.0
            else:
                v_n = profiler.R_history[-1] - profiler.R_history[-2]
            t1 = time.perf_counter()
            profiler.timings['velocity_calc'].append((t1 - t0) * 1000)
            profiler.v_history.append(v_n)
            
            # Acceleration
            t0 = time.perf_counter()
            if len(profiler.v_history) <= 1:
                a_n = 0.0
            else:
                a_n = profiler.v_history[-1] - profiler.v_history[-2]
            t1 = time.perf_counter()
            profiler.timings['acceleration_calc'].append((t1 - t0) * 1000)
            
            turn_end = time.perf_counter()
            profiler.timings['total_per_turn'].append((turn_end - turn_start) * 1000)
        
        print()
        
    else:
        # Full pipeline with embeddings
        print("Loading embedding models and PCA...")
        
        try:
            # Import the embedding module
            from embeddings import TextEmbedder
            import pickle
            
            # Initialize embedder
            embedder = TextEmbedder()
            
            # Load PCA model
            project_root = os.path.dirname(os.path.dirname(__file__))
            pca_path = os.path.join(project_root, 'models', 'pca_model.pkl')
            
            with open(pca_path, 'rb') as f:
                pca_model = pickle.load(f)
            
            print("✓ Models loaded successfully\n")
            
            # Generate VSAFE vector
            print(f"Generating VSAFE vector from text: '{args.vsafe_text}'")
            vsafe_embedding = embedder.embed_text(args.vsafe_text)
            VSAFE = pca_model.transform([vsafe_embedding])[0]
            print(f"VSAFE 2D coordinates: [{VSAFE[0]:.4f}, {VSAFE[1]:.4f}]\n")
            
            # Profile each turn
            print("Profiling conversation turns...")
            for i, message in enumerate(messages):
                print(f"Processing turn {i+1}/{len(messages)}...", end='\r')
                profiler.profile_single_turn(message, VSAFE, embedder, pca_model)
            
            print()
            
        except ImportError as e:
            print(f"Error: Could not import required modules: {e}")
            print("Make sure embeddings.py is available and dependencies are installed")
            return
        except FileNotFoundError as e:
            print(f"Error: Could not find PCA model: {e}")
            print("Run pca_trainer.py first to generate the model")
            return
        except Exception as e:
            print(f"Error during profiling: {e}")
            return
    
    # Print results
    profiler.print_report()


if __name__ == "__main__":
    main()
