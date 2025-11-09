# Vector Precognition: AI Safety Risk Analysis

Implementation of the Vector Precognition algorithm for detecting AI conversational risk through guardrail erosion and risk velocity in vector space.

Based on the white paper: *"AI Safety and Accuracy Using Guardrail Erosion and Risk Velocity in Vector Space"*

## Overview

This system analyzes AI conversation safety by:
1. Converting text responses to high-dimensional embeddings (AWS Bedrock Titan)
2. Reducing embeddings to 2D vectors using PCA
3. Calculating risk metrics using vector calculus
4. Predicting likelihood of guardrail breach

### Key Metrics

- **Risk Severity R(N)**: Cosine distance from safe-harbor vector (position)
- **Risk Rate v(N)**: First derivative of R(N) (velocity)
- **Guardrail Erosion a(N)**: Second derivative of R(N) (acceleration)
- **Failure Potential z(N)**: Weighted combination of R, v, a
- **Likelihood L(N)**: Sigmoid function predicting breach probability

## Project Structure

```
algorithm_work/
├── data/                          # Conversation examples and documentation
│   ├── jailbreak_conversation.json
│   ├── safe_conversation.json
│   ├── 2025.10.29 Algo Documentation.docx
│   └── 2025.10.30 Technical White Paper.pdf
├── src/                           # Source code
│   ├── vector_precognition_demo.py   # Main demo script
│   ├── embeddings.py                 # AWS Bedrock embedding functions
│   ├── pca_trainer.py                # PCA model training
│   ├── text_to_2d.py                 # Text→2D conversion pipeline
│   └── precog_test.py                # Additional tests
├── output/                        # Generated outputs
│   └── visuals/                   # Saved plots
├── models/                        # Trained PCA models (generated)
│   ├── embedding_scaler.pkl       # StandardScaler for embeddings
│   └── pca_model.pkl              # PCA dimensionality reduction
└── requirements.txt               # Python dependencies
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS Credentials (for text mode)

```bash
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key
export AWS_DEFAULT_REGION=us-east-1
```

Or configure via `aws configure` if you have AWS CLI installed.

## Usage

### Quick Start: Manual Mode (No AWS Required)

Run the demo with hardcoded 2D vectors:

```bash
python src/vector_precognition_demo.py --mode manual
```

This will:
- Process 3 example conversations (Jailbreak, Safe, Corrected)
- Print metrics tables to console
- Generate and save visualizations to `output/visuals/`

### Advanced: Text Mode (Requires AWS Bedrock)

#### Step 1: Train PCA Model

First, train the PCA dimensionality reduction model on example texts:

```bash
python src/pca_trainer.py
```

This will:
- Generate embeddings for ~50 example AI responses
- Fit a StandardScaler and PCA model
- Save models to `models/embedding_scaler.pkl` and `models/pca_model.pkl`

**Note**: For production, use 1000-5000+ diverse examples covering safe responses, neutral responses, boundary cases, and refusals.

#### Step 2: Run Demo in Text Mode

```bash
python src/vector_precognition_demo.py --mode text
```

This will:
- Convert example text conversations to 2D vectors via AWS Bedrock + PCA
- Process conversations through the risk analysis algorithm
- Generate visualizations

#### Step 3: Customize VSAFE

Define the "safe harbor" vector from custom text:

```bash
python src/vector_precognition_demo.py --mode text \
  --vsafe-text "I prioritize safety and ethical guidelines in all responses."
```

## Pipeline Architecture

### Text → 2D Vector Pipeline

```
User Input Text
       ↓
[AWS Bedrock Titan Embeddings]
       ↓
High-D Vector (1536 dimensions)
       ↓
[StandardScaler (pre-fitted)]
       ↓
Scaled Vector
       ↓
[PCA Model (pre-fitted)]
       ↓
2D Vector [x, y]
       ↓
[Vector Precognition Algorithm]
       ↓
Risk Metrics & Likelihood
```

### Key Components

1. **`embeddings.py`**: 
   - `get_titan_embedding(text)` - Generate embedding for single text
   - `get_batch_embeddings(texts)` - Batch processing
   - `get_conversation_embeddings(file)` - Process conversation JSON

2. **`pca_trainer.py`**:
   - `train_pca_model(texts, n_components=2)` - Fit scaler + PCA
   - Saves models to `models/` directory

3. **`text_to_2d.py`**:
   - `TextTo2DPipeline` class - Encapsulates full pipeline
   - `get_2d_vector(text)` - Convenience function

4. **`vector_precognition_demo.py`**:
   - `VectorPrecogntion` class - Core risk analysis algorithm
   - `plot_conversation_dynamics()` - 4-panel visualization
   - `plot_conversation_summary()` - Multi-conversation scatter plot

## Example: Processing Custom Conversations

### From JSON Files

```python
from embeddings import get_conversation_embeddings
from text_to_2d import TextTo2DPipeline
from vector_precognition_demo import VectorPrecogntion, WEIGHTS

# Load and embed conversation
result = get_conversation_embeddings('data/safe_conversation.json')

# Initialize pipeline and processor
pipeline = TextTo2DPipeline()
vsafe = pipeline.get_2d_vector("I'm helpful, harmless, and honest.")
processor = VectorPrecogntion(vsafe=vsafe, weights=WEIGHTS)

# Process each turn
for turn_data in result['embeddings']:
    if turn_data['embedding'] is not None:
        # Convert to 2D and process
        text = turn_data['message']
        vec_2d = pipeline.get_2d_vector(text)
        if vec_2d is not None:
            processor.process_turn(vec_2d)

# View results
metrics = processor.get_metrics()
print(metrics)
processor.plot_conversation_dynamics()
```

### From Text Lists

```python
from text_to_2d import TextTo2DPipeline
from vector_precognition_demo import VectorPrecogntion

# Initialize
pipeline = TextTo2DPipeline()
vsafe = pipeline.get_2d_vector("I prioritize safety.")

weights = {'wR': 1.5, 'wv': 1.0, 'wa': 3.0, 'b': -2.5}
processor = VectorPrecogntion(vsafe=vsafe, weights=weights)

# Process conversation
conversation = [
    "Hello, how can I help?",
    "I understand your request.",
    "Let me provide some information.",
    "Actually, I should reconsider that.",
    "Here's a safer alternative."
]

for text in conversation:
    vec = pipeline.get_2d_vector(text)
    if vec is not None:
        processor.process_turn(vec)

# Analyze
metrics = processor.get_metrics()
print(f"Peak Risk: {metrics['RiskSeverity_R(N)'].max():.3f}")
print(f"Peak Likelihood: {metrics['Likelihood_L(N)'].max():.3f}")
```

## Outputs

### Console Output

- Metrics table with columns: Turn, RiskSeverity, RiskRate, ErosionVelocity, FailurePotential, Likelihood
- 2D vector coordinates for each turn (in text mode)

### Visualizations (saved to `output/visuals/`)

1. **Conversation Dynamics** (`conversation_dynamics_<timestamp>.png`):
   - 4-panel plot showing R(N), v(N), a(N), L(N) over conversation turns
   - Peak severity annotations
   - Alert threshold lines

2. **Conversation Summary** (`conversation_summary_<timestamp>.png`):
   - Scatter plot: Peak Likelihood vs. Peak Severity
   - Multi-conversation comparison
   - Risk quadrant shading

## Model Parameters

### Default Weights (in demo)

```python
WEIGHTS = {
    'wR': 1.5,   # Weight for Risk Severity (position)
    'wv': 1.0,   # Weight for Risk Rate (velocity)
    'wa': 3.0,   # Weight for Erosion (acceleration) - highest impact
    'b': -2.5    # Bias (baseline risk threshold)
}
```

Adjust these empirically based on your use case. Higher `wa` emphasizes acceleration (rapid drift).

### Alert Threshold

Default: `0.8` (80% likelihood)

Customize in plotting:
```python
processor.plot_conversation_dynamics(alert_threshold=0.9)
```

## AWS Bedrock Configuration

- **Model**: `amazon.titan-embed-text-v1`
- **Embedding Dimension**: 1536
- **Region**: us-east-1 (default, configurable)

**Cost**: ~$0.0001 per 1000 tokens for Titan embeddings.

## Troubleshooting

### "PCA models not found"

Run training first:
```bash
python src/pca_trainer.py
```

### AWS Authentication Errors

Verify credentials:
```bash
aws sts get-caller-identity
```

Or run in manual mode:
```bash
python src/vector_precognition_demo.py --mode manual
```

### matplotlib Display Issues

If running headless (no display), modify plotting to save-only:
```python
# In vector_precognition_demo.py, comment out:
# plt.show()
```

## Citation

If using this implementation, please cite the white paper:
> "AI Safety and Accuracy Using Guardrail Erosion and Risk Velocity in Vector Space"

## License

See project license file.

## Questions & Answers

**Q: Are the plots hardcoded?**  
A: The plotting functions are generic and reusable. The demo includes hardcoded example conversations for testing, but you can process any text or vectors.

**Q: What embedding model does this use?**  
A: AWS Bedrock's Titan Text Embeddings v1 (1536 dimensions). The main demo can run in manual mode (no embeddings) or text mode (with embeddings).

**Q: What are the example inputs?**  
A: The demo processes three simulated conversations:
- **Jailbreak**: Starts safe, accelerates away from VSAFE, then corrects
- **Safe**: Remains near VSAFE throughout
- **Corrected**: Drifts away then returns to safety

In manual mode, these are 2D vectors. In text mode, they're text strings converted via the pipeline.

## Next Steps

1. **Expand Training Data**: Collect 1000-5000+ diverse AI response examples
2. **Fine-tune Weights**: Empirically optimize wR, wv, wa, b for your domain
3. **Real-time Integration**: Connect to live AI systems for monitoring
4. **Alerting**: Add threshold-based alerts and intervention logic
5. **Multi-dimensional Analysis**: Explore higher-dimensional PCA projections
