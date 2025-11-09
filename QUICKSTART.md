# Vector Precognition Implementation - Quick Start

## What I Built

I've implemented a complete **text → embedding → PCA → 2D** pipeline for the Vector Precognition risk analysis algorithm.

## Files Created/Modified

### New Modules
- **`src/embeddings.py`** - AWS Bedrock Titan embedding wrapper
- **`src/pca_trainer.py`** - PCA model training on example texts  
- **`src/text_to_2d.py`** - Complete pipeline: text → embedding → scaling → PCA → 2D
- **`src/vector_precognition_demo.py`** - Refactored demo (renamed from old file)
- **`requirements.txt`** - Python dependencies
- **`README.md`** - Comprehensive documentation

### Removed
- `src/2025.10.29 Algo Vibe Code.py` (renamed to `vector_precognition_demo.py`)
- `src/2025.10.30 PCA Vibe Code.py` (replaced by proper modules)

## Quick Start

### 1. Run Demo (Manual Mode - No AWS Required)

```bash
python src/vector_precognition_demo.py --mode manual
```

**Output**:
- Console metrics for 3 conversations (Jailbreak, Safe, Corrected)
- Plots saved to `output/visuals/`:
  - `conversation_dynamics_<timestamp>.png` (4-panel risk visualization)
  - `conversation_summary_<timestamp>.png` (multi-conversation scatter)

### 2. Run With Real Embeddings (Requires AWS Bedrock)

#### Step A: Train PCA Model

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Train PCA (generates embeddings for 50 examples)
python src/pca_trainer.py
```

This saves models to `models/embedding_scaler.pkl` and `models/pca_model.pkl`.

#### Step B: Run Demo in Text Mode

```bash
python src/vector_precognition_demo.py --mode text
```

This converts text conversations to 2D vectors via AWS Bedrock + PCA, then runs risk analysis.

## Answers to Your Questions

### 1. Are the plots hardcoded?

**Partially**:
- The **plotting functions** are generic and reusable (work with any data)
- The **demo inputs** are hardcoded for testing (3 example conversations)
- You can easily process custom conversations by:
  - Passing your own text lists (text mode)
  - Passing your own 2D vectors (manual mode)
  - Loading conversations from JSON files

### 2. What embedding model is the code using?

**Two modes**:

- **Manual Mode** (default): No embedding model. Uses hardcoded 2D vectors for testing.
- **Text Mode** (--mode text): Uses **AWS Bedrock Titan Text Embeddings v1**
  - Model ID: `amazon.titan-embed-text-v1`
  - Embedding dimension: 1536
  - API: boto3 / bedrock-runtime

The embedding function is in `src/embeddings.py`:
```python
def get_titan_embedding(text: str, region_name: str = "us-east-1") -> np.ndarray:
    # Calls AWS Bedrock to get 1536-D embedding
    # Returns numpy array
```

### 3. What are the input examples?

**Manual Mode** uses hardcoded 2D numpy arrays:
- **Jailbreak conversation** (10 turns): Starts safe [-0.4, 0.4], drifts to [1.0, -0.2], corrects back
- **Safe conversation** (5 turns): Stays near VSAFE [-0.5, 0.5]
- **Corrected conversation** (5 turns): Drifts to [0.5, 0.0], returns to [-0.3, 0.4]

**Text Mode** uses text strings:
- **Jailbreak**: "I'm happy to help..." → "While I shouldn't provide explicit details..." → "Let me provide a safer alternative"
- **Safe**: "I'm here to help with appropriate requests..." (all safe responses)
- **Corrected**: "I can help..." → "Actually, let me reconsider..." → "Here's a better way..."

## Pipeline Architecture

```
Text Input
    ↓
[AWS Bedrock Titan] → 1536-D embedding
    ↓
[StandardScaler] → Scaled vector
    ↓
[PCA Model] → 2D vector [x, y]
    ↓
[VectorPrecogntion] → Calculate R(N), v(N), a(N), z(N), L(N)
    ↓
Risk Metrics + Plots
```

## Output Files (from latest run)

```
output/visuals/
├── conversation_dynamics_20251109_125938.png
└── conversation_summary_20251109_125946.png
```

These are timestamped and saved automatically each run.

## Next Steps (Optional)

1. **Expand training data**: Collect 1000-5000+ diverse AI responses for better PCA
2. **Process real conversations**: Load from `data/safe_conversation.json` and `data/jailbreak_conversation.json`
3. **Tune weights**: Adjust wR, wv, wa, b in WEIGHTS dict for your domain
4. **Add alerting**: Implement threshold-based interventions
5. **Real-time integration**: Connect to live AI systems

## Dependencies

All listed in `requirements.txt`:
- numpy, pandas, matplotlib (data/viz)
- scikit-learn, joblib (PCA)
- boto3 (AWS Bedrock)

Install with:
```bash
pip install -r requirements.txt
```

## Complete Documentation

See `README.md` for full details, API examples, troubleshooting, and architecture diagrams.
