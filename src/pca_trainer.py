#!/usr/bin/env python3
"""
PCA model training for dimensionality reduction of embeddings.

This module trains a scaler and PCA model on a representative set of embeddings
to enable real-time 2D projection of new text inputs.
"""

import numpy as np
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Optional
from embeddings import get_batch_embeddings


def train_pca_model(
    example_texts: List[str],
    n_components: int = 2,
    region_name: str = "us-east-1",
    save_dir: Optional[str] = None
) -> tuple:
    """
    Train PCA model on example texts and save scaler + PCA model.
    
    Args:
        example_texts: List of representative text samples (e.g., 100-5000+)
        n_components: Number of PCA components (default: 2 for 2D visualization)
        region_name: AWS region for Bedrock API
        save_dir: Directory to save models (default: models/ in project root)
    
    Returns:
        Tuple of (scaler, pca_model, explained_variance_ratio)
    """
    print(f"\n{'='*70}")
    print("Training PCA Model for Embedding Reduction")
    print(f"{'='*70}")
    print(f"Example texts: {len(example_texts)}")
    print(f"Target dimensions: {n_components}")
    print()
    
    # Step 1: Get embeddings for all example texts
    print("Step 1/4: Generating embeddings...")
    embeddings_list = get_batch_embeddings(example_texts, region_name)
    
    # Filter out None values
    valid_embeddings = [emb for emb in embeddings_list if emb is not None]
    
    if len(valid_embeddings) < 10:
        raise ValueError(f"Not enough valid embeddings ({len(valid_embeddings)}). Need at least 10.")
    
    print(f"✓ Generated {len(valid_embeddings)}/{len(example_texts)} valid embeddings")
    
    # Convert to matrix
    high_dim_matrix = np.array(valid_embeddings)
    print(f"  Embedding dimension: {high_dim_matrix.shape[1]}")
    
    # Step 2: Standardize the data (crucial for PCA)
    print("\nStep 2/4: Standardizing embeddings...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(high_dim_matrix)
    print("✓ Scaler fitted")
    
    # Step 3: Fit PCA to reduce to n_components
    print(f"\nStep 3/4: Fitting PCA (reducing to {n_components}D)...")
    pca_model = PCA(n_components=n_components)
    pca_model.fit(scaled_data)
    
    explained_variance = pca_model.explained_variance_ratio_
    total_variance = sum(explained_variance)
    print(f"✓ PCA fitted")
    print(f"  Explained variance: {explained_variance}")
    print(f"  Total variance explained: {total_variance:.2%}")
    
    # Step 4: Save models
    if save_dir is None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        save_dir = os.path.join(project_root, 'models')
    
    os.makedirs(save_dir, exist_ok=True)
    
    scaler_path = os.path.join(save_dir, 'embedding_scaler.pkl')
    pca_path = os.path.join(save_dir, 'pca_model.pkl')
    
    print(f"\nStep 4/4: Saving models...")
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca_model, pca_path)
    
    print(f"✓ Scaler saved to: {scaler_path}")
    print(f"✓ PCA model saved to: {pca_path}")
    print(f"\n{'='*70}")
    print("PCA Training Complete!")
    print(f"{'='*70}\n")
    
    return scaler, pca_model, explained_variance


def main():
    """
    Example training with diverse AI responses.
    """
    # Example texts covering a range of AI behaviors
    # In production, use 1000-5000+ examples
    example_texts = [
        # Safe/helpful responses
        "I'm here to help! What would you like to know?",
        "I cannot assist with that request as it may be harmful.",
        "Let me provide some helpful information on that topic.",
        "I'd be happy to explain that concept to you.",
        "I'm designed to be helpful, harmless, and honest.",
        "That's an interesting question. Let me break it down for you.",
        "I apologize, but I cannot provide that information.",
        "Here's a safe and constructive way to approach that.",
        "I'm committed to providing accurate and helpful responses.",
        "Let's explore that topic in a responsible way.",
        
        # Neutral responses
        "I understand your question. Here's what I know.",
        "That's a complex topic with many perspectives.",
        "I can provide some general information about that.",
        "Let me clarify what you're asking.",
        "I'm not certain about that. Let me explain what I do know.",
        "That depends on several factors.",
        "I can help with some aspects of that question.",
        "Let me provide some context for that topic.",
        "That's outside my primary area of expertise.",
        "I can offer some general guidance on that.",
        
        # Boundary-testing responses (not harmful but less aligned)
        "Well, technically you could do that, but...",
        "I suppose one approach might be...",
        "While I can't recommend that specifically...",
        "That's an unusual request. Let me think about it.",
        "I'm not supposed to discuss that in detail, however...",
        "There are ways to work around that limitation.",
        "I understand what you're asking for.",
        "That's a gray area, but here's some information.",
        "I can see why you might be interested in that.",
        "Let me provide a hypothetical example.",
        
        # More varied safe responses
        "I prioritize safety in all my responses.",
        "Let's find a constructive solution together.",
        "I'm designed to decline inappropriate requests.",
        "How can I assist you in a helpful way?",
        "I operate within clear ethical guidelines.",
        "Let me suggest a safer alternative.",
        "I'm here to provide accurate information.",
        "That request doesn't align with my guidelines.",
        "I can help you with a modified version of that.",
        "Let's focus on what I can assist with.",
        
        # Additional diversity
        "I need more context to answer that properly.",
        "That's a good question. Here's my understanding.",
        "I can provide educational information on that topic.",
        "Let me help you understand the risks involved.",
        "I'm programmed to prioritize user safety.",
        "That's something I should decline to do.",
        "Here's what I know about that subject.",
        "I can explain the principles behind that.",
        "That would not be appropriate for me to do.",
        "Let me provide a balanced perspective on that.",
    ]
    
    print(f"\n⚠  IMPORTANT: This training uses {len(example_texts)} example texts.")
    print("For production use, collect 1000-5000+ diverse examples covering:")
    print("  - Safe/aligned responses")
    print("  - Neutral/informational responses")
    print("  - Boundary cases")
    print("  - Refusals and corrections")
    print()
    
    try:
        train_pca_model(example_texts)
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        print("\nMake sure AWS credentials are configured:")
        print("  export AWS_ACCESS_KEY_ID=your_key")
        print("  export AWS_SECRET_ACCESS_KEY=your_secret")
        print("  export AWS_DEFAULT_REGION=us-east-1")


if __name__ == "__main__":
    main()
