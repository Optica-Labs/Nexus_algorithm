#!/usr/bin/env python3
"""
Text-to-2D vector conversion pipeline.

This module chains: text → embedding → scaling → PCA → 2D vector
for use in the Vector Precognition risk analysis algorithm.
"""

import numpy as np
import joblib
import os
from typing import Optional
from embeddings import get_titan_embedding


class TextTo2DPipeline:
    """
    Converts raw text to 2D vectors using pre-trained scaler and PCA models.
    """
    
    def __init__(self, models_dir: Optional[str] = None, region_name: str = "us-east-1"):
        """
        Initialize the pipeline by loading pre-trained models.
        
        Args:
            models_dir: Directory containing saved models (default: models/ in project root)
            region_name: AWS region for Bedrock API
        """
        if models_dir is None:
            project_root = os.path.dirname(os.path.dirname(__file__))
            models_dir = os.path.join(project_root, 'models')
        
        self.region_name = region_name
        
        scaler_path = os.path.join(models_dir, 'embedding_scaler.pkl')
        pca_path = os.path.join(models_dir, 'pca_model.pkl')
        
        # Check if models exist
        if not os.path.exists(scaler_path) or not os.path.exists(pca_path):
            raise FileNotFoundError(
                f"PCA models not found in {models_dir}.\n"
                "Please train the models first by running:\n"
                "  python src/pca_trainer.py"
            )
        
        # Load models
        self.scaler = joblib.load(scaler_path)
        self.pca_model = joblib.load(pca_path)
        
        print(f"✓ Loaded scaler from: {scaler_path}")
        print(f"✓ Loaded PCA model from: {pca_path}")
    
    def get_2d_vector(self, text: str) -> Optional[np.ndarray]:
        """
        End-to-end conversion: text → 2D vector.
        
        Args:
            text: Input text to convert
        
        Returns:
            2D numpy array [x, y] or None if embedding fails
        """
        # Step 1: Get high-dimensional embedding from Bedrock
        high_dim_vector = get_titan_embedding(text, self.region_name)
        
        if high_dim_vector is None:
            return None
        
        # Step 2: Reshape for scaler (expects 2D array)
        high_dim_vector_batch = high_dim_vector.reshape(1, -1)
        
        # Step 3: Apply the same scaling used during training
        scaled_vector = self.scaler.transform(high_dim_vector_batch)
        
        # Step 4: Apply the same PCA transformation
        vector_2d = self.pca_model.transform(scaled_vector)
        
        # Return as 1D array [x, y]
        return vector_2d[0]
    
    def get_batch_2d_vectors(self, texts: list) -> list:
        """
        Convert a batch of texts to 2D vectors.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of 2D numpy arrays (or None for failed conversions)
        """
        vectors = []
        for text in texts:
            vector = self.get_2d_vector(text)
            vectors.append(vector)
        return vectors


def get_2d_vector(text: str, models_dir: Optional[str] = None, region_name: str = "us-east-1") -> Optional[np.ndarray]:
    """
    Convenience function to convert text to 2D vector without managing pipeline object.
    
    Args:
        text: Input text
        models_dir: Directory containing saved models
        region_name: AWS region for Bedrock
    
    Returns:
        2D numpy array [x, y] or None if conversion fails
    """
    pipeline = TextTo2DPipeline(models_dir, region_name)
    return pipeline.get_2d_vector(text)


if __name__ == "__main__":
    """
    Example usage of the text-to-2D pipeline.
    """
    print("\n" + "="*70)
    print("Text-to-2D Vector Pipeline Demo")
    print("="*70 + "\n")
    
    # Example texts
    example_texts = [
        "I'm here to help you with that request.",
        "I cannot assist with harmful activities.",
        "Let me provide some information on that topic.",
    ]
    
    try:
        # Initialize pipeline
        pipeline = TextTo2DPipeline()
        
        print("\nConverting texts to 2D vectors:\n")
        
        for text in example_texts:
            print(f"Text: '{text[:50]}...'")
            vector = pipeline.get_2d_vector(text)
            
            if vector is not None:
                print(f"  → 2D Vector: [{vector[0]:.4f}, {vector[1]:.4f}]")
            else:
                print(f"  → Failed to generate vector")
            print()
        
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {str(e)}\n")
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}\n")
