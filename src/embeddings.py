#!/usr/bin/env python3
"""
Embedding generation using Amazon Titan Embeddings model via AWS Bedrock.
"""

import json
import boto3
import numpy as np
from typing import List, Dict, Any, Optional


def get_titan_embedding(text: str, region_name: str = "us-east-1") -> Optional[np.ndarray]:
    """
    Generate embedding for a single text string using Amazon Titan Embeddings.
    
    Args:
        text: Input text to embed
        region_name: AWS region where Bedrock is available (default: us-east-1)
    
    Returns:
        Numpy array containing the embedding vector, or None if error occurs
    """
    try:
        # Initialize Bedrock runtime client
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        
        # Model ID for Amazon Titan Embeddings
        model_id = "amazon.titan-embed-text-v1"
        
        # Prepare request body
        request_body = json.dumps({
            "inputText": text
        })
        
        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=request_body
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        # Extract embedding vector and convert to numpy array
        embedding = response_body.get('embedding', [])
        
        if not embedding:
            print(f"Warning: Empty embedding returned for text: {text[:50]}...")
            return None
            
        return np.array(embedding)
        
    except Exception as e:
        print(f"Error generating embedding for '{text[:50]}...': {str(e)}")
        return None


def get_batch_embeddings(texts: List[str], region_name: str = "us-east-1") -> List[Optional[np.ndarray]]:
    """
    Generate embeddings for a batch of texts.
    
    Args:
        texts: List of input texts
        region_name: AWS region where Bedrock is available
    
    Returns:
        List of numpy arrays (or None for failed embeddings)
    """
    embeddings = []
    
    for i, text in enumerate(texts):
        print(f"Processing {i+1}/{len(texts)}: {text[:50]}...")
        embedding = get_titan_embedding(text, region_name)
        embeddings.append(embedding)
    
    return embeddings


def load_conversation(file_path: str) -> Dict[str, Any]:
    """
    Load conversation data from JSON file.
    
    Args:
        file_path: Path to conversation JSON file
    
    Returns:
        Dictionary containing conversation data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def get_ai_responses(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract AI responses from conversation data.
    
    Args:
        conversation: Conversation dictionary with 'turns' list
    
    Returns:
        List of dictionaries with turn_number and AI message
    """
    ai_responses = []
    for turn in conversation.get('turns', []):
        if turn.get('speaker') == 'ai':
            ai_responses.append({
                'turn_number': turn['turn_number'],
                'message': turn['message']
            })
    return ai_responses


def get_conversation_embeddings(conversation_file: str, region_name: str = "us-east-1") -> Dict[str, Any]:
    """
    Generate embeddings for all AI responses in a conversation.
    
    Args:
        conversation_file: Path to conversation JSON file
        region_name: AWS region where Bedrock is available
    
    Returns:
        Dictionary containing conversation metadata and embeddings for each AI turn
    """
    # Load conversation
    conversation = load_conversation(conversation_file)
    
    # Extract AI responses
    ai_responses = get_ai_responses(conversation)
    
    # Generate embeddings
    messages = [resp['message'] for resp in ai_responses]
    embeddings_list = get_batch_embeddings(messages, region_name)
    
    # Combine results
    result = {
        'conversation_id': conversation.get('conversation_id', 'unknown'),
        'conversation_type': conversation.get('conversation_type', 'unknown'),
        'description': conversation.get('description', ''),
        'total_turns': len(conversation.get('turns', [])),
        'ai_turns': len(ai_responses),
        'embeddings': []
    }
    
    for ai_resp, embedding in zip(ai_responses, embeddings_list):
        result['embeddings'].append({
            'turn_number': ai_resp['turn_number'],
            'message': ai_resp['message'],
            'embedding': embedding.tolist() if embedding is not None else None,
            'embedding_dimension': len(embedding) if embedding is not None else 0,
            'error': None if embedding is not None else "Failed to generate embedding"
        })
    
    return result
