# Customer Deployment: Embedding Model Options

## Your Questions Answered

### Q1: "How could I have this embedding model while working with customer?"

**Answer:** You have 3 main options:

---

## ✅ Option 1: Local Open-Source Models (RECOMMENDED)

### What is it?
Deploy **sentence-transformers** models directly on customer infrastructure.

### How it works:
```python
# Customer's server/application
from sentence_transformers import SentenceTransformer

# One-time: Download model (happens automatically)
model = SentenceTransformer('all-mpnet-base-v2')

# Runtime: Generate embeddings locally
embedding = model.encode("User message here")
# Latency: 10-30ms (no network call!)
```

###Properties:
- ✅ **Runs locally** on customer's machine
- ✅ **No internet required** after initial download
- ✅ **No API costs** - free forever
- ✅ **Customer owns their data** - never leaves their infrastructure
- ✅ **Apache 2.0 license** - commercial use allowed
- ✅ **Lower latency** - 10-30ms vs 50-200ms for AWS

### Installation for customer:
```bash
pip install sentence-transformers
```

### Model file size:
- `all-MiniLM-L6-v2`: 80MB (fastest)
- `all-mpnet-base-v2`: 420MB (best quality)

### First-time setup:
```python
# Downloads model files once, caches locally
model = SentenceTransformer('all-mpnet-base-v2')
# Model files stored in: ~/.cache/torch/sentence_transformers/
```

### Your deployment package:
```
your_guardrail_package/
├── requirements.txt (includes: sentence-transformers)
├── models/
│   └── pca_model_local.pkl (your trained PCA)
├── src/
│   ├── local_embedder.py
│   ├── vector_precognition_demo2.py
│   └── __init__.py
└── README.md (setup instructions)
```

---

## ⚠️ Option 2: Private AWS Bedrock (Customer's Account)

### What is it?
Customer uses **their own** AWS Bedrock account.

### How it works:
- Customer sets up AWS Bedrock in their account
- Your code calls **their** Bedrock endpoint
- They pay for API calls (~$0.0001 per request)

### Properties:
- ⚠️ **Customer needs AWS account**
- ⚠️ **Ongoing costs** per request
- ⚠️ **Requires internet** connection
- ⚠️ **Higher latency** (50-200ms network overhead)
- ✅ **Data stays in their AWS** (doesn't come to you)

### When to use:
- Customer already uses AWS heavily
- They prefer managed services
- Volume is low (<100k requests/month)

---

## 🚀 Option 3: Hybrid (Cache + Fallback)

### What is it?
Use local models primarily, AWS as fallback.

### How it works:
```python
class HybridEmbedder:
    def embed_text(self, text):
        # Try local first (fast)
        try:
            return self.local_model.encode(text)
        except Exception:
            # Fallback to AWS (slower but reliable)
            return aws_bedrock_embed(text)
```

### When to use:
- Maximum reliability needed
- Some customers have GPU, others don't
- Gradual migration from AWS to local

---

## 📊 Comparison Table

| Feature | Local Models | Customer's AWS | Your AWS |
|---------|-------------|----------------|----------|
| **Latency** | 10-30ms ⚡ | 50-200ms | 50-200ms |
| **Cost** | $0 ✅ | ~$10-50/month | You pay |
| **Internet needed** | No ✅ | Yes | Yes |
| **Data privacy** | Perfect ✅ | Good | Concerns |
| **Setup complexity** | Easy | Medium | Easy |
| **Customer control** | Full ✅ | Full ✅ | None |
| **Offline capable** | Yes ✅ | No | No |

---

## 🎯 Recommended Approach for Customer Deployment

### Step 1: Package Local Model
```bash
# Create deployment package
your_guardrail_service/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── models/
│   └── pca_model_local.pkl
└── src/
    ├── local_embedder.py
    ├── vector_precognition_demo2.py
    └── api.py  # FastAPI endpoint
```

### Step 2: Customer Deployment
```bash
# On customer's server
git clone your_repo
cd your_guardrail_service
docker-compose up -d

# That's it! Service runs on http://localhost:8000
```

### Step 3: Integration
```python
# Customer's application code
import requests

response = requests.post('http://localhost:8000/check-safety', json={
    'user_message': 'User input here',
    'model_message': 'Model response here'
})

if response.json()['likelihood'] > 0.8:
    print("⚠️ High risk detected!")
```

---

## 💻 Hardware Requirements (Customer Side)

### Minimum (CPU only):
- **CPU:** 2+ cores (any modern Intel/AMD)
- **RAM:** 2GB free
- **Disk:** 1GB (model + dependencies)
- **Expected latency:** 20-50ms per turn

### Recommended:
- **CPU:** 4+ cores
- **RAM:** 4GB free
- **Disk:** 2GB
- **Expected latency:** 10-30ms per turn

### Optimal (with GPU):
- **GPU:** Any NVIDIA GPU (even GTX 1050)
- **VRAM:** 2GB+
- **Expected latency:** 2-10ms per turn

### Will it work on:
- ✅ Customer's laptop: Yes (20-50ms)
- ✅ Customer's server: Yes (10-30ms)
- ✅ Cloud VM (AWS EC2, Azure, GCP): Yes (10-30ms)
- ✅ Kubernetes cluster: Yes (10-30ms)
- ⚠️ Raspberry Pi: Yes but slow (50-200ms)
- ❌ Browser JavaScript: No (use backend API)

---

## 🔐 Licensing (Can You Commercially Deploy?)

### Yes! Here's what you need:

1. **Sentence Transformers**: Apache 2.0 License
   - ✅ Commercial use allowed
   - ✅ Modify and redistribute
   - ✅ Private use
   - ⚠️ Must include license notice

2. **Your Code**: Your choice
   - You own your algorithm
   - You can license it however you want
   - Customer pays you for your IP, not the embedding model

3. **What customer gets:**
   ```
   Your Product = Your Algorithm + Open Source Embedding Model
   Customer pays you for: Your guardrail detection logic
   Customer gets for free: The embedding model
   ```

### License Requirements:
Create a `LICENSE` file in your package:
```
This product uses the following open-source components:

1. sentence-transformers (Apache 2.0 License)
   https://github.com/UKPLab/sentence-transformers
   
2. [Your proprietary code] (Your License)
   Copyright (c) 2025 Optica Labs
   All rights reserved.
```

---

## 📈 Cost Analysis (Annual)

### Local Deployment:
- **Setup cost:** $0 (open source)
- **Runtime cost:** $0 (customer's hardware)
- **Maintenance:** Update docker image annually
- **Total Year 1:** $0 ongoing costs

### AWS Bedrock (Customer's account):
- **Setup cost:** $0
- **100k requests/month:** $10-50/month
- **1M requests/month:** $100-500/month
- **Total Year 1:** $120-6000

### AWS Bedrock (Your account):
- **Setup cost:** $0
- **Cost:** You pay for all customers' usage
- **Risk:** Unbounded if customer has high volume
- **Total Year 1:** $$$$ (not recommended)

---

## 🚀 Quick Start: Deploy Local Model for Customer

### File: `local_embedder.py` (already created)
```python
from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
    
    def embed_text(self, text):
        return self.model.encode(text)
```

### File: `docker-compose.yml`
```yaml
version: '3.8'
services:
  guardrail-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_NAME=all-mpnet-base-v2
```

### File: `Dockerfile`
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run service
CMD ["python", "src/api.py"]
```

### Customer deploys:
```bash
docker-compose up -d
```

**Done!** Service runs locally, no AWS needed.

---

## ⚡ Performance: Will It Be Different on Customer Hardware?

### YES - Hardware matters for embeddings:

| Hardware | Embedding Time | Erosion Math | Total |
|----------|---------------|--------------|-------|
| **Your Dev Machine** (i7) | 10-30ms | 0.12ms | 10-30ms |
| **Customer Laptop** (i5) | 20-50ms | 0.12ms | 20-50ms |
| **Customer Server** (Xeon) | 15-40ms | 0.12ms | 15-40ms |
| **With GPU** (NVIDIA T4) | 2-10ms | 0.12ms | 2-10ms |
| **Raspberry Pi 4** | 50-200ms | 0.12ms | 50-200ms |

### Key Point:
- **Erosion math (R, v, a) is ALWAYS fast (~0.12ms)** ✅
  - Not hardware dependent
  - Already optimized (numpy)
  
- **Embeddings ARE hardware dependent** ⚠️
  - CPU speed matters
  - GPU makes 5-10x difference
  - But still fast enough (10-50ms on any modern CPU)

### Testing on Customer Hardware:
```bash
# Give customer this test script
python src/local_embedder.py --compare

# Output shows timing for their specific hardware
```

---

## 🎯 Final Recommendation

### For Customer Deployments:

**USE LOCAL MODELS** (`sentence-transformers`)

**Why:**
1. ✅ **No ongoing costs** - customer loves this
2. ✅ **Data privacy** - stays on their infrastructure
3. ✅ **Lower latency** - 10-30ms vs 50-200ms
4. ✅ **Works offline** - no internet dependency
5. ✅ **Easy deployment** - docker-compose up -d
6. ✅ **You control updates** - not dependent on AWS
7. ✅ **Legally clean** - Apache 2.0 license

**Your Package:**
- Python library or Docker container
- Includes: local_embedder.py + your PCA + your algorithm
- Customer installs on their infrastructure
- They run it, own their data, pay $0 for embeddings

**You already have:**
- ✅ `src/local_embedder.py` - Ready to use
- ✅ Documentation - Complete guide
- 🔄 Need: Retrain PCA on local embeddings (I can help with this)

Would you like me to create the Docker deployment package for you?
