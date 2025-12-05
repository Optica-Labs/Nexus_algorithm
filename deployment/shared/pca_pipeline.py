#!/usr/bin/env python3
"""
PCA Pipeline for Text-to-2D vector conversion.

This module chains: text → embedding → scaling → PCA → 2D vector
for use in the Vector Precognition risk analysis algorithm.
"""

import numpy as np
import joblib
import os
from typing import Optional, List
import logging

# Handle both relative and absolute imports
try:
    from .embeddings import EmbeddingGenerator
except ImportError:
    from embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class PCATransformer:
    """
    Converts raw text to 2D vectors using pre-trained scaler and PCA models.
    """

    def __init__(self, models_dir: Optional[str] = None, region_name: str = "us-east-1"):
        """
        Initialize the pipeline by loading pre-trained models.

        Args:
            models_dir: Directory containing saved models (default: deployment/models/)
            region_name: AWS region for Bedrock API
        """
        if models_dir is None:
            # Default to deployment/models directory
            current_dir = os.path.dirname(__file__)
            models_dir = os.path.join(os.path.dirname(current_dir), 'models')

        self.models_dir = models_dir
        self.region_name = region_name

        scaler_path = os.path.join(models_dir, 'embedding_scaler.pkl')
        pca_path = os.path.join(models_dir, 'pca_model.pkl')

        # Check if models exist
        if not os.path.exists(scaler_path) or not os.path.exists(pca_path):
            raise FileNotFoundError(
                f"PCA models not found in {models_dir}.\n"
                "Please train the models first by running:\n"
                "  python src/pca_trainer.py\n"
                "Then copy the models to deployment/models/"
            )

        # Load models
        self.scaler = joblib.load(scaler_path)
        self.pca_model = joblib.load(pca_path)

        logger.info(f"✓ Loaded scaler from: {scaler_path}")
        logger.info(f"✓ Loaded PCA model from: {pca_path}")

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(region_name=region_name)

    def text_to_2d(self, text: str, verbose: bool = False) -> Optional[np.ndarray]:
        """
        End-to-end conversion: text → 2D vector.

        Args:
            text: Input text to convert
            verbose: Whether to log detailed progress

        Returns:
            2D numpy array [x, y] or None if embedding fails
        """
        if verbose:
            logger.info(f"[Text-to-2D Pipeline]")
            logger.info(f"  Input text: '{text[:80]}...'")
            logger.info(f"  Region: {self.region_name}")

        # Step 1: Get high-dimensional embedding from Bedrock
        if verbose:
            logger.info(f"  Step 1/3: Getting embedding from Bedrock...")

        high_dim_vector = self.embedding_generator.generate(text)

        if high_dim_vector is None:
            logger.error(f"  [FAILED] Could not get embedding from Bedrock")
            return None

        if verbose:
            logger.info(f"  ✓ Embedding received, dimension: {len(high_dim_vector)}")

        # Step 2: Reshape for scaler (expects 2D array)
        high_dim_vector_batch = high_dim_vector.reshape(1, -1)

        # Step 3: Apply the same scaling used during training
        if verbose:
            logger.info(f"  Step 2/3: Applying StandardScaler...")
        scaled_vector = self.scaler.transform(high_dim_vector_batch)

        # Step 4: Apply the same PCA transformation
        if verbose:
            logger.info(f"  Step 3/3: Applying PCA transformation...")
        vector_2d = self.pca_model.transform(scaled_vector)

        if verbose:
            logger.info(f"  ✓ 2D vector: [{vector_2d[0][0]:.4f}, {vector_2d[0][1]:.4f}]")

        # Return as 1D array [x, y]
        return vector_2d[0]

    def batch_text_to_2d(self, texts: List[str], verbose: bool = False) -> List[Optional[np.ndarray]]:
        """
        Convert a batch of texts to 2D vectors.

        Args:
            texts: List of input texts
            verbose: Whether to log detailed progress

        Returns:
            List of 2D numpy arrays (or None for failed conversions)
        """
        vectors = []
        for i, text in enumerate(texts):
            if verbose:
                logger.info(f"Processing {i+1}/{len(texts)}")
            vector = self.text_to_2d(text, verbose=False)
            vectors.append(vector)
        return vectors

    def embedding_to_2d(self, embedding: np.ndarray) -> np.ndarray:
        """
        Convert high-dimensional embedding to 2D vector.
        Useful when embeddings are already computed.

        Args:
            embedding: High-dimensional embedding vector

        Returns:
            2D numpy array [x, y]
        """
        # Reshape for scaler
        embedding_batch = embedding.reshape(1, -1)

        # Apply scaling
        scaled = self.scaler.transform(embedding_batch)

        # Apply PCA
        vector_2d = self.pca_model.transform(scaled)

        return vector_2d[0]
