#!/usr/bin/env python3
"""
Train PCA Model for Local Embeddings (all-MiniLM-L6-v2)

This script trains a PCA model to reduce local embeddings from 384D to 2D.
The trained model will be used by vector_precognition_demo3.py.

IMPORTANT:
- Local embeddings (all-MiniLM-L6-v2) are 384D
- AWS Bedrock embeddings (titan-embed-text-v1) are 1536D
- You MUST retrain PCA when switching embedding models!
"""

import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch
import os

def train_pca_for_local_embeddings(
    training_texts=None,
    model_name='all-MiniLM-L6-v2',
    n_components=2,
    output_path='models/pca_model_local.pkl'
):
    """
    Train PCA model on local embeddings.
    
    Args:
        training_texts: List of training texts (if None, uses default safe examples)
        model_name: Sentence-transformers model name
        n_components: Number of PCA components (default: 2 for visualization)
        output_path: Where to save the trained PCA model
    """
    print("\n" + "="*80)
    print("TRAINING PCA FOR LOCAL EMBEDDINGS")
    print("="*80)
    
    # Initialize local embedder
    print(f"\n🤖 Loading embedding model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device.upper()}")
    
    model = SentenceTransformer(model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"   Embedding dimension: {embedding_dim}D")
    
    # Default training data (safe/guardrail-aligned examples)
    if training_texts is None:
        print(f"\n📝 Using default training texts (safe examples)")
        training_texts = [
            # Safe, helpful responses
            "I'm here to help you with information and assistance.",
            "I can provide guidance on that topic in a responsible way.",
            "Let me explain the ethical considerations around that.",
            "I'd be happy to help you learn more about this subject.",
            "That's a great question. Here's some helpful information.",
            "I can assist you with finding legitimate resources.",
            "Let's explore the legal and ethical aspects of this.",
            "I'm designed to be helpful, harmless, and honest.",
            "I can provide educational information on this topic.",
            "Let me help you understand this in a constructive way.",
            
            # Additional safe variations
            "I appreciate your question and want to give a thoughtful response.",
            "Let me provide some context and helpful information.",
            "I'm glad to assist with your inquiry in a positive manner.",
            "Here's a balanced perspective on that topic.",
            "I can guide you toward appropriate resources.",
            "Let's discuss this in a constructive and informative way.",
            "I'm here to support your learning and understanding.",
            "That's within my capabilities to assist with responsibly.",
            "I can share helpful insights about that subject.",
            "Let me offer some guidance that aligns with best practices.",
            
            # Boundary-setting (still safe)
            "I can't help with that specific request, but I can suggest alternatives.",
            "That's outside my guidelines, but here's what I can help with.",
            "I'm not able to provide that information for safety reasons.",
            "Let me redirect you to more appropriate resources.",
            "I need to decline that request, but I can help you with...",
            "That crosses ethical boundaries, so let me offer a different approach.",
            "I can't assist with that, but I can provide information on...",
            "That's not something I can support, but here's what I can do.",
            "I'm designed to avoid that type of content, but I can help you with...",
            "That request conflicts with my guidelines, so let me suggest...",
            
            # Educational/informative
            "Here's some factual information about that topic.",
            "Let me break down the key concepts for you.",
            "I can explain the scientific principles behind this.",
            "That's an interesting area of study. Here's what research shows.",
            "Let me provide some historical context on this subject.",
            "I can share some expert perspectives on this matter.",
            "Here's a balanced overview of different viewpoints.",
            "Let me clarify some common misconceptions about this.",
            "I can help you understand the technical aspects of this.",
            "That's a complex topic. Here's a simplified explanation.",
            
            # Problem-solving/helpful
            "Let's work through this problem together step by step.",
            "I can suggest several approaches to solve this challenge.",
            "Here are some strategies that might help with your situation.",
            "Let me help you brainstorm some positive solutions.",
            "I can guide you through the decision-making process.",
            "Here's a framework for thinking about this issue.",
            "Let's explore some creative alternatives.",
            "I can offer some practical advice on this matter.",
            "Here are some tools and resources that might assist you.",
            "Let me help you develop a constructive action plan."
        ]
    
    print(f"   Training samples: {len(training_texts)}")
    
    # Generate embeddings
    print(f"\n🔄 Generating embeddings for training data...")
    embeddings = model.encode(training_texts, show_progress_bar=True)
    print(f"   ✓ Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}D")
    
    # Train PCA
    print(f"\n📊 Training PCA: {embeddings.shape[1]}D → {n_components}D")
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    
    # Show explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"\n   📈 Explained variance:")
    for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance), 1):
        print(f"      PC{i}: {ev*100:.2f}% (cumulative: {cv*100:.2f}%)")
    
    print(f"\n   Total variance captured: {cumulative_variance[-1]*100:.2f}%")
    
    if cumulative_variance[-1] < 0.5:
        print(f"   ⚠️  WARNING: Low variance captured ({cumulative_variance[-1]*100:.1f}%)")
        print(f"   ⚠️  Consider using more PCA components or more diverse training data")
    elif cumulative_variance[-1] < 0.7:
        print(f"   ⚠️  MODERATE: {cumulative_variance[-1]*100:.1f}% variance captured")
        print(f"   ✓  Acceptable for visualization, but some information is lost")
    else:
        print(f"   ✅ GOOD: {cumulative_variance[-1]*100:.1f}% variance captured")
    
    # Test projection
    print(f"\n🧪 Testing PCA projection...")
    test_text = "Hello, how can I help you today?"
    test_emb = model.encode(test_text, show_progress_bar=False)
    test_2d = pca.transform([test_emb])[0]
    print(f"   Test embedding: {test_emb.shape[0]}D → 2D")
    print(f"   Result: [{test_2d[0]:.4f}, {test_2d[1]:.4f}]")
    
    # Save PCA model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(pca, f)
    
    print(f"\n💾 PCA model saved to: {output_path}")
    
    # Save metadata
    metadata = {
        'embedding_model': model_name,
        'embedding_dim': embedding_dim,
        'pca_components': n_components,
        'explained_variance': explained_variance.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'training_samples': len(training_texts)
    }
    
    metadata_path = output_path.replace('.pkl', '_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Metadata saved to: {metadata_path}")
    
    print("\n" + "="*80)
    print("PCA TRAINING COMPLETE!")
    print("="*80)
    print(f"\n✅ You can now use this PCA model with vector_precognition_demo3.py")
    print(f"   Usage: python src/vector_precognition_demo3.py <conversation.json> {output_path}")
    
    return pca, metadata


def train_pca_from_conversation_file(
    conversation_file,
    model_name='all-MiniLM-L6-v2',
    output_path='models/pca_model_local.pkl'
):
    """
    Train PCA using texts from an existing conversation file.
    This creates a PCA space based on your actual conversation data.
    
    Args:
        conversation_file: Path to JSON conversation file
        model_name: Sentence-transformers model name
        output_path: Where to save the trained PCA model
    """
    import json
    
    print(f"\n📖 Loading training texts from: {conversation_file}")
    
    with open(conversation_file, 'r') as f:
        data = json.load(f)
    
    conversation_data = data.get('conversation') or data.get('turns')
    
    training_texts = []
    for turn in conversation_data:
        message = turn.get('message', '')
        if message:
            training_texts.append(message)
    
    print(f"   ✓ Loaded {len(training_texts)} messages")
    
    return train_pca_for_local_embeddings(
        training_texts=training_texts,
        model_name=model_name,
        output_path=output_path
    )


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("PCA TRAINER FOR LOCAL EMBEDDINGS")
    print("="*80)
    print("\nThis script trains a PCA model for use with vector_precognition_demo3.py")
    print("The PCA model reduces 384D embeddings (all-MiniLM-L6-v2) to 2D.\n")
    
    # Parse arguments
    mode = "default"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--from-conversation":
            mode = "conversation"
            if len(sys.argv) < 3:
                print("Usage: python src/train_pca_local.py --from-conversation <conversation.json>")
                sys.exit(1)
            conversation_file = sys.argv[2]
            output_path = sys.argv[3] if len(sys.argv) > 3 else "models/pca_model_local.pkl"
        else:
            output_path = sys.argv[1]
    else:
        output_path = "models/pca_model_local.pkl"
    
    # Train PCA
    if mode == "conversation":
        print(f"Mode: Training from conversation file")
        pca, metadata = train_pca_from_conversation_file(
            conversation_file=conversation_file,
            output_path=output_path
        )
    else:
        print(f"Mode: Training from default safe examples")
        pca, metadata = train_pca_for_local_embeddings(
            output_path=output_path
        )
    
    print("\n✅ Done! You can now run:")
    print(f"   python src/vector_precognition_demo3.py input/unsafe_conversation_example.json {output_path}")
