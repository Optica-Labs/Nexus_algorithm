# Local Embedding Model Deployment Guide

## Problem Statement

AWS Bedrock embeddings add **50-200ms latency** due to:
- Network round-trip to AWS
- API call overhead
- Shared infrastructure variability

**Solution:** Deploy embedding models locally for **sub-10ms** embedding time.

---

## Option 1: Sentence Transformers (Recommended for Customer Deployments)

### Overview
- **Library:** `sentence-transformers` (Hugging Face)
- **Latency:** 5-50ms per text (depending on model size)
- **Requirements:** CPU (works) or GPU (faster)
- **License:** Apache 2.0 (commercial-friendly)

### Installation

```bash
pip install sentence-transformers
```

### Compatible Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 80MB | ⚡ Very Fast (5-10ms) | Good | Real-time alerting |
| `all-mpnet-base-v2` | 420MB | ⚡ Fast (10-30ms) | Better | Production (recommended) |
| `all-MiniLM-L12-v2` | 120MB | ⚡ Fast (8-15ms) | Good | Balanced |
| `paraphrase-multilingual-mpnet-base-v2` | 970MB | 🐢 Slower (30-50ms) | Best | Multi-language support |

### Implementation Example

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import time

class LocalEmbedder:
    """Local embedding using Sentence Transformers."""
    
    def __init__(self, model_name='all-mpnet-base-v2', device='cpu'):
        """
        Initialize local embedding model.
        
        Args:
            model_name: HuggingFace model name
            device: 'cpu' or 'cuda' (GPU)
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✓ Model loaded on {device}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array with embedding (384-768 dimensions)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: list) -> np.ndarray:
        """
        Generate embeddings for multiple texts (faster than one-by-one).
        
        Args:
            texts: List of input texts
            
        Returns:
            Numpy array with embeddings (N x embedding_dim)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, 
                                       batch_size=32, show_progress_bar=False)
        return embeddings


# Usage Example
if __name__ == "__main__":
    embedder = LocalEmbedder('all-mpnet-base-v2')
    
    # Single text
    text = "This is a test message for safety evaluation."
    
    t0 = time.perf_counter()
    embedding = embedder.embed_text(text)
    t1 = time.perf_counter()
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Latency: {(t1-t0)*1000:.2f}ms")
    
    # Batch processing (more efficient)
    texts = ["Message 1", "Message 2", "Message 3"]
    
    t0 = time.perf_counter()
    embeddings = embedder.embed_batch(texts)
    t1 = time.perf_counter()
    
    print(f"\nBatch embeddings shape: {embeddings.shape}")
    print(f"Total time: {(t1-t0)*1000:.2f}ms")
    print(f"Per-text: {(t1-t0)*1000/len(texts):.2f}ms")
```

---

## Option 2: ONNX Runtime (Fastest)

### Overview
- **Library:** `onnxruntime` + optimized models
- **Latency:** 2-20ms per text
- **Requirements:** CPU optimization
- **Best for:** Maximum performance on CPU

### Why ONNX?
- Models converted to optimized format
- Faster inference than PyTorch
- Smaller memory footprint
- Better CPU utilization

### Implementation

```bash
pip install onnxruntime optimum[exporters]
```

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np

class ONNXEmbedder:
    """Ultra-fast embedding using ONNX Runtime."""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_name, 
            export=True  # Auto-convert to ONNX if needed
        )
    
    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", 
                               padding=True, truncation=True)
        outputs = self.model(**inputs)
        # Mean pooling
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
        return embedding
```

---

## Option 3: TensorFlow Lite (Edge Devices)

### Overview
- **Library:** `tensorflow-lite`
- **Latency:** 5-30ms
- **Best for:** Embedded systems, mobile, IoT
- **Deployment:** Raspberry Pi, ARM devices

---

## Performance Comparison

### Latency Benchmarks (CPU: Intel i7, 1 text)

| Method | Latency | Quality | Deployment |
|--------|---------|---------|------------|
| AWS Bedrock Titan | 50-200ms | ⭐⭐⭐⭐⭐ | Cloud only |
| Sentence Transformers (mpnet) | 10-30ms | ⭐⭐⭐⭐ | ✅ Easy |
| Sentence Transformers (MiniLM) | 5-10ms | ⭐⭐⭐ | ✅ Easy |
| ONNX Runtime | 2-10ms | ⭐⭐⭐⭐ | ⚠️ Moderate |
| TensorFlow Lite | 5-20ms | ⭐⭐⭐ | ⚠️ Complex |

### With GPU (NVIDIA RTX)

| Method | Latency | Throughput |
|--------|---------|------------|
| Sentence Transformers (GPU) | 2-5ms | 100-500 texts/sec |
| ONNX + GPU | 1-3ms | 200-1000 texts/sec |

---

## Deployment Architectures

### Architecture 1: Standalone Service (Recommended)

```
Customer Application
        ↓
    HTTP/gRPC Request
        ↓
Local Embedding Service (FastAPI)
    - Runs on customer's server
    - sentence-transformers model
    - CPU or GPU
        ↓
    PCA + Erosion Detection
        ↓
    Alert/Response (0.1ms)
```

**Advantages:**
- Customer controls all data (no external API calls)
- Low latency (~10-30ms total)
- Scales with hardware
- Works offline

**Disadvantages:**
- Customer needs to host service
- Requires ~1-2GB RAM
- Updates require deployment

### Architecture 2: Embedded Library

```
Customer Application (Python)
    ↓
Import your library
    - Local embedder (sentence-transformers)
    - PCA model (pickle file)
    - Erosion detection
    ↓
Real-time monitoring (~15-40ms)
```

**Advantages:**
- Simplest integration
- No separate service needed
- Customer has full control

**Disadvantages:**
- Customer app must be Python
- Model files bundled (~500MB)

### Architecture 3: Containerized Deployment

```
Docker Container (your guardrail service)
    - sentence-transformers/all-mpnet-base-v2
    - PCA model
    - FastAPI endpoint
    - Runs on customer infrastructure
```

```dockerfile
FROM python:3.9-slim

# Install dependencies
RUN pip install sentence-transformers fastapi uvicorn numpy scikit-learn

# Copy your models
COPY models/ /app/models/
COPY src/ /app/src/

WORKDIR /app

# Run service
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Advantages:**
- Easy deployment (docker-compose)
- Isolated environment
- Version controlled
- Customer infrastructure only

---

## Model Compatibility with Your PCA

### Important: Embedding Dimensions

Your current AWS Titan model produces **1536-dimensional** embeddings.
Your PCA model was trained on these 1536D vectors.

**Options:**

### 1. Re-train PCA (Recommended)
Train new PCA on local model embeddings:

```python
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import pickle

# Load local model
embedder = SentenceTransformer('all-mpnet-base-v2')  # 768D

# Generate embeddings from your training data
texts = [...]  # Your training corpus
embeddings = embedder.encode(texts)

# Train PCA: 768D → 2D
pca = PCA(n_components=2)
pca.fit(embeddings)

# Save for deployment
with open('models/pca_model_local.pkl', 'wb') as f:
    pickle.dump(pca, f)
```

### 2. Dimension Adapter (Quick Fix)
Use a linear layer to map dimensions:

```python
# Train adapter: 768D → 1536D → PCA → 2D
adapter = nn.Linear(768, 1536)
# Train on paired data (local embedding → AWS embedding)
```

---

## Licensing and Customer Deployment

### Open Source Models (Sentence Transformers)

✅ **Commercial Use Allowed**
- Most models: Apache 2.0 or MIT license
- Can be deployed at customer sites
- No usage fees
- Check specific model license on HuggingFace

### What You Can Do:
1. ✅ Deploy model on customer infrastructure
2. ✅ Include in your commercial product
3. ✅ Modify and optimize
4. ✅ No per-request fees

### What You Must Do:
1. ⚠️ Include license notices
2. ⚠️ Credit original authors
3. ⚠️ Don't claim model as your own

---

## Recommended Deployment Strategy

### For Your Use Case (Real-time Alerting):

```python
# production_embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import time

class ProductionEmbedder:
    """Production-ready local embedder with caching."""
    
    def __init__(self, model_name='all-mpnet-base-v2', 
                 pca_model_path='models/pca_model_local.pkl',
                 use_cache=True):
        """
        Initialize production embedder.
        
        Args:
            model_name: Sentence transformer model
            pca_model_path: Path to trained PCA model
            use_cache: Enable embedding cache
        """
        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        t0 = time.time()
        self.embedder = SentenceTransformer(model_name)
        print(f"✓ Loaded in {time.time()-t0:.2f}s")
        
        # Load PCA model
        print(f"Loading PCA model: {pca_model_path}")
        with open(pca_model_path, 'rb') as f:
            self.pca = pickle.load(f)
        print(f"✓ PCA: {self.pca.n_components_}D → 2D")
        
        # Cache for repeated phrases
        self.cache = {} if use_cache else None
        self.cache_hits = 0
        self.cache_misses = 0
    
    def text_to_2d(self, text: str) -> np.ndarray:
        """
        Convert text to 2D vector.
        
        Args:
            text: Input text
            
        Returns:
            2D numpy array
        """
        # Check cache
        if self.cache is not None:
            if text in self.cache:
                self.cache_hits += 1
                return self.cache[text]
            self.cache_misses += 1
        
        # Generate embedding
        t0 = time.perf_counter()
        embedding = self.embedder.encode(text, convert_to_numpy=True)
        t1 = time.perf_counter()
        
        # Apply PCA
        vector_2d = self.pca.transform([embedding])[0]
        t2 = time.perf_counter()
        
        # Store in cache
        if self.cache is not None:
            self.cache[text] = vector_2d
        
        print(f"  Embed: {(t1-t0)*1000:.2f}ms, PCA: {(t2-t1)*1000:.2f}ms")
        
        return vector_2d
    
    def get_cache_stats(self):
        """Return cache hit rate."""
        if self.cache is None:
            return "Cache disabled"
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return "No requests yet"
        hit_rate = self.cache_hits / total * 100
        return f"Cache: {self.cache_hits}/{total} hits ({hit_rate:.1f}%)"
```

### Usage in Your Pipeline:

```python
from production_embedder import ProductionEmbedder
from vector_precognition_demo2 import VectorPrecogntion

# Initialize once at startup
embedder = ProductionEmbedder(
    model_name='all-mpnet-base-v2',
    pca_model_path='models/pca_model_local.pkl',
    use_cache=True
)

# Define VSAFE
vsafe_text = "I'm designed to be helpful, harmless, and honest."
VSAFE = embedder.text_to_2d(vsafe_text)

# Initialize detector
detector = VectorPrecogntion(vsafe=VSAFE, weights={...})

# Real-time monitoring
def monitor_conversation(user_msg, model_msg):
    """Monitor a single turn."""
    # Convert to 2D vectors
    user_vec = embedder.text_to_2d(user_msg)      # ~10-30ms
    model_vec = embedder.text_to_2d(model_msg)    # ~10-30ms
    
    # Process turn
    detector.process_turn(model_vec, user_vec)     # ~0.12ms
    
    # Check metrics
    metrics = detector.get_metrics()
    likelihood = metrics['Likelihood_Model'].iloc[-1]
    
    if likelihood > 0.8:
        return "ALERT: High risk detected!"
    return "OK"
```

---

## Hardware Requirements

### Minimum (CPU Only)
- **CPU:** 2+ cores
- **RAM:** 2GB
- **Disk:** 1GB (model + dependencies)
- **Latency:** 20-50ms per text

### Recommended (CPU)
- **CPU:** 4+ cores, modern Intel/AMD
- **RAM:** 4GB
- **Disk:** 2GB
- **Latency:** 10-30ms per text

### Optimal (GPU)
- **GPU:** NVIDIA GPU (even GTX 1050 helps)
- **VRAM:** 2GB+
- **CPU:** 4+ cores
- **RAM:** 8GB
- **Latency:** 2-10ms per text

### Cloud Deployment Options
- **AWS EC2:** t3.medium (CPU) or g4dn.xlarge (GPU)
- **GCP:** n1-standard-2 (CPU) or n1-standard-4 + T4 GPU
- **Azure:** Standard_D2s_v3 (CPU) or NC6 (GPU)

---

## Migration Path from AWS Bedrock

### Step 1: Test Local Model

```bash
# Install dependencies
pip install sentence-transformers

# Test model
python test_local_embeddings.py
```

### Step 2: Re-train PCA

```bash
# Generate embeddings with local model
python retrain_pca_local.py --model all-mpnet-base-v2

# Output: models/pca_model_local.pkl
```

### Step 3: Validate Equivalence

```bash
# Compare AWS vs Local results
python compare_embeddings.py
```

### Step 4: Deploy

```bash
# Package for customer
docker build -t guardrail-service .
docker run -p 8000:8000 guardrail-service
```

---

## Cost Comparison

### AWS Bedrock (Current)
- **Cost:** $0.0001 per 1000 input tokens
- **100k requests/month:** ~$10-50/month
- **Latency:** 50-200ms
- **Control:** Limited

### Local Deployment (Proposed)
- **Cost:** Infrastructure only (customer provides)
- **100k requests/month:** $0 (after hardware)
- **Latency:** 10-30ms (CPU) or 2-10ms (GPU)
- **Control:** Full

### ROI
- Break-even: Immediate (no per-request fees)
- Customer benefit: Lower latency + data privacy
- Your benefit: Simpler deployment + offline capability

---

## Next Steps

1. **Test local model** with your data
2. **Re-train PCA** on local embeddings
3. **Benchmark latency** on target hardware
4. **Package as Docker container**
5. **Create deployment guide** for customers

Would you like me to create the implementation files for local embedding deployment?
