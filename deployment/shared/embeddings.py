#!/usr/bin/env python3
"""
Embedding generation using Amazon Titan Embeddings model via AWS Bedrock.
Refactored for deployment applications.
"""

import json
import boto3
from botocore.config import Config
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Handles embedding generation using AWS Bedrock Titan model.
    """

    def __init__(self, region_name: str = "us-east-1", model_id: str = "amazon.titan-embed-text-v1"):
        """
        Initialize the embedding generator.

        Args:
            region_name: AWS region where Bedrock is available
            model_id: Bedrock model ID for embeddings
        """
        self.region_name = region_name
        self.model_id = model_id
        self.bedrock_runtime = None

    def _get_client(self):
        """Lazy initialization of Bedrock client with explicit timeouts."""
        if self.bedrock_runtime is None:
            try:
                # Configure with longer timeouts and retries for WSL2/network issues
                config = Config(
                    connect_timeout=10,  # 10 seconds to establish connection
                    read_timeout=60,     # 60 seconds to read response
                    retries={
                        'max_attempts': 3,
                        'mode': 'adaptive'
                    }
                )

                self.bedrock_runtime = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=self.region_name,
                    config=config
                )
                logger.info(f"Bedrock client initialized for region: {self.region_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Bedrock client: {e}")
                raise
        return self.bedrock_runtime

    def generate(self, text: str, retry_count: int = 2) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text string.

        Args:
            text: Input text to embed
            retry_count: Number of times to retry on connection errors

        Returns:
            Numpy array containing the embedding vector, or None if error occurs
        """
        last_error = None

        for attempt in range(retry_count + 1):
            try:
                # Force fresh client on retry attempts (helps with Streamlit caching issues)
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{retry_count} - creating fresh client...")
                    self.bedrock_runtime = None

                client = self._get_client()

                # Prepare request body
                request_body = json.dumps({"inputText": text})

                # Invoke the model
                response = client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=request_body
                )

                # Parse response
                response_body = json.loads(response['body'].read())
                embedding = response_body.get('embedding', [])

                if not embedding:
                    logger.warning(f"Empty embedding returned for text: {text[:50]}...")
                    return None

                return np.array(embedding)

            except Exception as e:
                last_error = e

                # Check if this is a retryable error (connection issues)
                if "Could not connect" in str(e) or "Connection" in str(e) or "timeout" in str(e).lower():
                    if attempt < retry_count:
                        logger.warning(f"Connection error (attempt {attempt+1}/{retry_count+1}): {e}")
                        logger.info("Retrying with fresh client...")
                        continue
                    else:
                        logger.error(f"Failed after {retry_count+1} attempts")
                else:
                    # Non-retryable error - fail immediately
                    break

        # All retries exhausted
        logger.error(f"Failed to generate embedding: {last_error}")
        logger.error(f"Text (first 100 chars): '{text[:100]}...'")

        # Provide helpful error messages
        if last_error:
            if "credentials" in str(last_error).lower():
                logger.error("This appears to be a credentials issue. Check: aws sts get-caller-identity")
            elif "ValidationException" in str(last_error):
                logger.error(f"Model access issue. Possible causes:")
                logger.error(f"  1. Model not available in region: {self.region_name}")
                logger.error(f"  2. Model access not enabled in your AWS account")
            elif "Could not connect" in str(last_error):
                logger.error("Connection issue. Possible causes:")
                logger.error("  1. Network connectivity problem")
                logger.error("  2. Proxy/firewall blocking AWS API")
                logger.error("  3. Streamlit process network isolation (WSL2 issue)")
                logger.error("  Try: Run app outside venv, or check WSL2 network settings")

        return None

    def generate_batch(self, texts: List[str], show_progress: bool = True) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts
            show_progress: Whether to log progress

        Returns:
            List of numpy arrays (or None for failed embeddings)
        """
        embeddings = []

        for i, text in enumerate(texts):
            if show_progress:
                logger.info(f"Processing {i+1}/{len(texts)}: {text[:50]}...")

            embedding = self.generate(text)
            embeddings.append(embedding)

        return embeddings


def get_titan_embedding(text: str, region_name: str = "us-east-1") -> Optional[np.ndarray]:
    """
    Convenience function for backward compatibility.
    Generate embedding for a single text string using Amazon Titan Embeddings.

    Args:
        text: Input text to embed
        region_name: AWS region where Bedrock is available (default: us-east-1)

    Returns:
        Numpy array containing the embedding vector, or None if error occurs
    """
    generator = EmbeddingGenerator(region_name=region_name)
    return generator.generate(text)


def get_batch_embeddings(texts: List[str], region_name: str = "us-east-1") -> List[Optional[np.ndarray]]:
    """
    Convenience function for backward compatibility.
    Generate embeddings for a batch of texts.

    Args:
        texts: List of input texts
        region_name: AWS region where Bedrock is available

    Returns:
        List of numpy arrays (or None for failed embeddings)
    """
    generator = EmbeddingGenerator(region_name=region_name)
    return generator.generate_batch(texts)
