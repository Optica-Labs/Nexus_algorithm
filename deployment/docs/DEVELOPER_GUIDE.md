# Developer Guide - Vector Precognition Deployment Suite

## Purpose

This guide serves as a comprehensive pilot document for developers joining the Vector Precognition project. It covers architecture, development workflows, coding standards, and best practices.

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Development Environment Setup](#development-environment-setup)
3. [Code Structure](#code-structure)
4. [Development Workflow](#development-workflow)
5. [Testing Strategy](#testing-strategy)
6. [Docker Development](#docker-development)
7. [Coding Standards](#coding-standards)
8. [Common Tasks](#common-tasks)
9. [Troubleshooting](#troubleshooting)

---

## Project Architecture

### High-Level Overview

The Vector Precognition suite implements a three-stage AI safety analysis pipeline:

```
Stage 1: Guardrail Erosion Analysis (App 1)
   ↓ Turn-by-turn risk metrics
Stage 2: RHO Calculation (App 2)
   ↓ Per-conversation robustness
Stage 3: PHI Evaluation (App 3)
   ↓ Multi-conversation benchmarking
Unified Dashboard (App 4)
   → Real-time monitoring + live chat
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit 1.30+ | Interactive web UI |
| **Backend** | Python 3.10 | Core logic |
| **ML Pipeline** | scikit-learn | PCA dimensionality reduction |
| **Embeddings** | AWS Bedrock Titan | Text-to-vector conversion |
| **Visualization** | Matplotlib, Plotly | Risk dynamics plots |
| **Containerization** | Docker + Compose | Deployment |
| **Data Format** | JSON, CSV | Data interchange |

### Architectural Principles

1. **Modularity**: Each app is self-contained with shared dependencies in `shared/`
2. **Separation of Concerns**: `core/` (logic), `ui/` (presentation), `utils/` (helpers)
3. **Stateless Design**: Apps communicate via file exports/imports
4. **Pipeline Pattern**: Data flows through stages: Embeddings → PCA → Risk Metrics
5. **MLOps Best Practices**: Model versioning, validation, reproducibility

---

## Development Environment Setup

### Prerequisites

- Python 3.10+
- Git
- AWS CLI (configured with Bedrock access)
- Docker Desktop (for production testing)
- VSCode or PyCharm (recommended)

### Initial Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd algorithm_work/deployment

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure AWS credentials
cp .env.example .env
nano .env  # Add AWS keys

# 5. Verify setup
python -c "import streamlit; import boto3; print('Setup OK')"
```

### IDE Configuration

**VSCode settings (`.vscode/settings.json`):**
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true
}
```

**PyCharm:**
- Set Project Interpreter → venv/bin/python
- Enable pytest as test runner
- Configure code style: Black formatter

---

## Code Structure

### Directory Layout

```
deployment/
├── shared/                     # Shared modules (all apps)
│   ├── embeddings.py          # AWS Bedrock interface
│   ├── pca_pipeline.py        # Dimensionality reduction
│   ├── config.py              # Configuration management
│   ├── validators.py          # Input validation
│   └── visualizations.py      # Common plotting functions
│
├── models/                     # Pre-trained ML models
│   ├── pca_model.pkl          # PCA dimensionality reducer
│   └── embedding_scaler.pkl   # StandardScaler for embeddings
│
├── app1_guardrail_erosion/
│   ├── app.py                 # Streamlit entry point
│   ├── core/
│   │   ├── vector_processor.py    # Core algorithm implementation
│   │   └── risk_calculator.py     # Risk metric calculations
│   ├── ui/
│   │   ├── input_methods.py       # Input UI components
│   │   └── visualization.py       # Plot rendering
│   └── utils/
│       └── helpers.py             # Utility functions
│
├── app2_rho_calculator/
│   ├── app.py
│   ├── core/
│   │   └── rho_engine.py          # RHO calculation engine
│   └── utils/
│       └── parsers.py             # Data parsing utilities
│
├── app3_phi_evaluator/
│   ├── app.py
│   ├── core/
│   │   └── phi_engine.py          # PHI benchmarking engine
│   └── utils/
│       └── aggregators.py         # Multi-model aggregation
│
└── app4_unified_dashboard/
    ├── app.py
    ├── core/
    │   ├── pipeline_orchestrator.py   # 3-stage pipeline
    │   └── api_client.py              # LLM API interface
    ├── ui/
    │   ├── chat_view.py               # Chat interface
    │   └── sidebar.py                 # Configuration sidebar
    └── utils/
        └── session_state.py           # Streamlit session management
```

### Module Responsibilities

#### `shared/embeddings.py`
- **Purpose**: Interface to AWS Bedrock Titan embeddings
- **Key Functions**:
  - `get_titan_embedding(text)`: Single text → 1536-D vector
  - `get_batch_embeddings(texts)`: Batch processing
  - `get_conversation_embeddings(file)`: Process JSON conversations
- **Dependencies**: boto3, botocore

#### `shared/pca_pipeline.py`
- **Purpose**: Dimensionality reduction (1536-D → 2-D)
- **Key Class**: `TextTo2DPipeline`
  - `fit(texts)`: Train scaler + PCA
  - `transform(embedding)`: Apply reduction
  - `save_models(path)`: Persist models
- **Dependencies**: scikit-learn, joblib

#### `shared/config.py`
- **Purpose**: Centralized configuration management
- **Key Functions**:
  - `load_env()`: Load .env variables
  - `get_aws_credentials()`: Retrieve AWS config
  - `validate_config()`: Pre-flight checks

#### `app1_guardrail_erosion/core/vector_processor.py`
- **Purpose**: Core Vector Precognition algorithm
- **Key Class**: `VectorPrecognition`
  - `analyze_conversation(turns)`: Main analysis method
  - `calculate_metrics(vectors)`: R, v, a, z, L calculations
  - `generate_plot(metrics)`: 6-panel dynamics visualization
- **Algorithm**:
  ```python
  R(N) = cosine_distance(V(N), V_SAFE)
  v(N) = R(N) - R(N-1)
  a(N) = v(N) - v(N-1)
  z(N) = wR*R + wv*v + wa*a + b
  L(N) = 1 / (1 + exp(-z))
  ```

#### `app2_rho_calculator/core/rho_engine.py`
- **Purpose**: Calculate robustness index per conversation
- **Key Class**: `RhoCalculator`
  - `calculate_rho(metrics)`: ρ = Σ|model_risk| / Σ|user_risk|
  - `classify_robustness(rho)`: Robust/Reactive/Fragile
  - `generate_cumulative_plot()`: Risk accumulation viz

#### `app3_phi_evaluator/core/phi_engine.py`
- **Purpose**: Benchmark model fragility across conversations
- **Key Class**: `PhiEvaluator`
  - `calculate_phi(rho_values)`: Φ = (1/N) * Σ max(0, ρ-1)
  - `classify_model(phi)`: PASS (<0.1) / FAIL (≥0.1)
  - `compare_models(results)`: Multi-model comparison

#### `app4_unified_dashboard/core/pipeline_orchestrator.py`
- **Purpose**: Orchestrate 3-stage pipeline in real-time
- **Key Class**: `PipelineOrchestrator`
  - `process_turn(user_msg, model_msg)`: Run full pipeline
  - `stage1_erosion()`: Per-turn analysis
  - `stage2_rho()`: Conversation robustness
  - `stage3_phi()`: Model benchmark

---

## Development Workflow

### Branch Strategy

```
main          Production-ready code
  ├── develop      Integration branch
      ├── feature/app1-custom-endpoint
      ├── feature/app2-batch-processing
      ├── bugfix/app3-phi-calculation
      └── hotfix/app4-api-timeout
```

### Feature Development

1. **Create feature branch:**
   ```bash
   git checkout -b feature/app1-new-metric
   ```

2. **Implement feature:**
   - Write code in appropriate module (core/ui/utils)
   - Follow coding standards (see below)
   - Add docstrings and type hints

3. **Test in venv:**
   ```bash
   source venv/bin/activate
   cd app1_guardrail_erosion
   streamlit run app.py
   ```

4. **Write tests:**
   ```bash
   # tests/test_new_metric.py
   pytest tests/test_new_metric.py -v
   ```

5. **Test in Docker:**
   ```bash
   docker-compose build app1-guardrail-erosion
   docker-compose up app1-guardrail-erosion
   ```

6. **Commit and push:**
   ```bash
   git add .
   git commit -m "feat(app1): Add new risk metric calculation"
   git push origin feature/app1-new-metric
   ```

7. **Create pull request** → Review → Merge to develop

### Bug Fix Workflow

1. **Reproduce bug** in venv
2. **Write failing test** that captures the bug
3. **Fix code** until test passes
4. **Verify in Docker** (production environment)
5. **Document fix** in `docs/PROJECT_STATUS.md`
6. **Commit with prefix:** `fix(app2): Resolve RHO calculation edge case`

---

## Testing Strategy

### Test Pyramid

```
E2E Tests (10%)         End-to-end workflows through all apps
   ↑
Integration Tests (30%) Cross-module interactions
   ↑
Unit Tests (60%)        Individual functions and classes
```

### Development Stage: Test in venv

**Why?** Fast iteration, easy debugging

```bash
source venv/bin/activate
cd app1_guardrail_erosion
streamlit run app.py --server.runOnSave true
```

**Test checklist:**
- ✅ All input methods work
- ✅ Error handling (invalid data)
- ✅ Export functionality
- ✅ Visualizations render correctly

### Production Stage: Test in Docker

**Why?** Replicate deployment environment

```bash
docker-compose build app1-guardrail-erosion
docker-compose up app1-guardrail-erosion
```

**Test checklist:**
- ✅ Container starts without errors
- ✅ Health check passes
- ✅ Volume persistence works
- ✅ Environment variables loaded
- ✅ Network connectivity (App 4 API calls)

### Unit Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_vector_processor.py::test_calculate_risk_severity -v

# Run with coverage
pytest tests/ --cov=shared --cov-report=html
```

**Example test:**
```python
# tests/test_vector_processor.py
import pytest
from app1_guardrail_erosion.core.vector_processor import VectorPrecognition

def test_calculate_risk_severity():
    vp = VectorPrecognition()
    vsafe = [0.5, 0.5]
    v_current = [0.8, 0.3]

    risk = vp.calculate_risk_severity(v_current, vsafe)

    assert 0 <= risk <= 1
    assert isinstance(risk, float)
```

### Integration Testing

```bash
# Test end-to-end pipeline
python tests/integration/test_pipeline.py
```

**Example:**
```python
# tests/integration/test_pipeline.py
def test_app1_to_app2_pipeline():
    # Step 1: Generate data in App 1
    from app1_guardrail_erosion.core.vector_processor import VectorPrecognition
    vp = VectorPrecognition()
    metrics = vp.analyze_conversation(test_data)
    vp.export_json("test_output.json")

    # Step 2: Import into App 2
    from app2_rho_calculator.core.rho_engine import RhoCalculator
    rho_calc = RhoCalculator()
    rho = rho_calc.calculate_from_file("test_output.json")

    # Assert
    assert rho > 0
    assert rho_calc.classify(rho) in ["Robust", "Reactive", "Fragile"]
```

---

## Docker Development

### Multi-Stage Build Optimization

The Dockerfile uses **3 stages** for fast builds:

```dockerfile
Stage 1: base          # Python + system deps
Stage 2: dependencies  # pip install (cached)
Stage 3: application   # Copy code (changes frequently)
```

**Key optimization:** Requirements are installed in a separate stage, so code changes don't trigger full reinstall.

### Build and Test Cycle

```bash
# 1. Make code changes in venv
nano app1_guardrail_erosion/core/vector_processor.py

# 2. Test in venv (fast iteration)
streamlit run app1_guardrail_erosion/app.py

# 3. Build Docker image (when ready for production test)
docker-compose build app1-guardrail-erosion

# 4. Test in Docker
docker-compose up app1-guardrail-erosion

# 5. Check logs
docker-compose logs app1-guardrail-erosion
```

### Development Docker Compose

Create `docker-compose.dev.yml` for live code mounting:

```yaml
version: '3.8'
services:
  app1-guardrail-erosion:
    volumes:
      - ./app1_guardrail_erosion:/app/app1_guardrail_erosion
      - ./shared:/app/shared
    command: streamlit run app.py --server.runOnSave true
```

**Usage:**
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Debugging in Docker

```bash
# Attach to running container
docker exec -it vp-app1-guardrail-erosion bash

# Check Python environment
python -c "import sys; print(sys.path)"

# Check environment variables
env | grep AWS

# Check file permissions
ls -la /app/models/

# Test embeddings
python -c "from shared.embeddings import get_titan_embedding; print(len(get_titan_embedding('test')))"
```

---

## Coding Standards

### Python Style Guide

Follow **PEP 8** + project-specific conventions:

```python
# Good: Modular function with type hints and docstring
from typing import List, Tuple
import numpy as np

def calculate_risk_severity(
    v_current: np.ndarray,
    v_safe: np.ndarray
) -> float:
    """
    Calculate risk severity using cosine distance.

    Args:
        v_current: Current position vector (2D)
        v_safe: Safe harbor reference vector (2D)

    Returns:
        Risk severity score between 0 and 1

    Raises:
        ValueError: If vectors have different dimensions
    """
    if v_current.shape != v_safe.shape:
        raise ValueError("Vector dimensions must match")

    # Calculate cosine distance
    dot_product = np.dot(v_current, v_safe)
    norm_product = np.linalg.norm(v_current) * np.linalg.norm(v_safe)

    cosine_sim = dot_product / norm_product
    risk = 1 - cosine_sim

    return max(0, min(1, risk))  # Clamp to [0, 1]


# Bad: No type hints, unclear variable names, no docstring
def calc(v1, v2):
    x = np.dot(v1, v2)
    y = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 1 - x/y
```

### Code Organization

**Principle:** Each file has a **single responsibility**

```
✅ Good:
shared/
  ├── embeddings.py      # Only AWS Bedrock logic
  ├── pca_pipeline.py    # Only PCA logic
  └── validators.py      # Only validation logic

❌ Bad:
shared/
  └── utils.py           # Everything mixed together
```

### Error Handling

```python
# Good: Specific exceptions with helpful messages
try:
    embedding = get_titan_embedding(text)
except botocore.exceptions.ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'ResourceNotFoundException':
        raise ValueError(f"Bedrock model not found. Ensure access to amazon.titan-embed-text-v1")
    else:
        raise RuntimeError(f"AWS API error: {error_code}")
except Exception as e:
    raise RuntimeError(f"Unexpected error getting embedding: {str(e)}")

# Bad: Silent failures
try:
    embedding = get_titan_embedding(text)
except:
    embedding = None
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Good: Structured logging with levels
logger.info(f"Processing conversation with {len(turns)} turns")
logger.debug(f"Embedding dimensions: {embedding.shape}")
logger.warning(f"Risk threshold exceeded: L={likelihood:.3f}")
logger.error(f"Failed to load PCA model: {e}")

# Bad: Print statements
print("Starting analysis...")
print("Error occurred")
```

### Streamlit Best Practices

```python
# Good: Use session state for persistence
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Good: Cache expensive operations
@st.cache_resource
def load_pca_model():
    return joblib.load('models/pca_model.pkl')

# Good: Clear error messages
try:
    result = process_data(uploaded_file)
except ValueError as e:
    st.error(f"Invalid data format: {e}")
    st.info("Expected CSV with columns: turn, speaker, message")

# Bad: No caching, unclear errors
model = joblib.load('models/pca_model.pkl')  # Loads on every rerun!
st.error("Error")  # Unhelpful
```

---

## Common Tasks

### Adding a New Metric to App 1

1. **Define metric function** in `core/vector_processor.py`:
   ```python
   def calculate_new_metric(self, vectors: List[np.ndarray]) -> List[float]:
       """Calculate new risk metric."""
       return [self._compute_metric(v) for v in vectors]
   ```

2. **Update analysis pipeline:**
   ```python
   def analyze_conversation(self, turns):
       # ... existing metrics ...
       metrics['new_metric'] = self.calculate_new_metric(vectors)
       return metrics
   ```

3. **Add to visualization** in `ui/visualization.py`:
   ```python
   ax5.plot(metrics['new_metric'], label='New Metric')
   ```

4. **Update export format:**
   ```python
   df['new_metric'] = metrics['new_metric']
   ```

5. **Write tests:**
   ```python
   def test_new_metric():
       assert vp.calculate_new_metric(test_vectors) is not None
   ```

6. **Update documentation** in `docs/PROJECT_STATUS.md`

### Adding a New LLM to App 4

1. **Add endpoint to `.env`:**
   ```env
   NEW_LLM_ENDPOINT=https://api.example.com/chat
   ```

2. **Update API client** in `core/api_client.py`:
   ```python
   def call_new_llm(self, message: str) -> str:
       response = requests.post(
           os.getenv('NEW_LLM_ENDPOINT'),
           json={'message': message}
       )
       return response.json()['reply']
   ```

3. **Add to model selection** in `ui/sidebar.py`:
   ```python
   model = st.selectbox(
       "Select Model",
       ["GPT-3.5", "Claude", "Mistral", "New LLM"]
   )
   ```

4. **Update orchestrator** in `core/pipeline_orchestrator.py`:
   ```python
   if model == "New LLM":
       reply = self.api_client.call_new_llm(message)
   ```

### Creating a New App

1. **Copy template structure:**
   ```bash
   cp -r app1_guardrail_erosion app5_new_feature
   ```

2. **Update `config.yaml`** with new app settings

3. **Modify `app.py`** main entry point

4. **Implement core logic** in `core/` module

5. **Update Docker:**
   ```dockerfile
   # Add to Dockerfile
   FROM application as app5
   WORKDIR /app/app5_new_feature
   CMD ["streamlit", "run", "app.py", "--server.port=8505"]
   ```

6. **Update docker-compose.yml:**
   ```yaml
   app5-new-feature:
     build:
       target: app5
     ports:
       - "8505:8505"
   ```

7. **Create tests** in `tests/test_app5.py`

8. **Document** in `docs/APP5_GUIDE.md`

---

## Troubleshooting

### Import Errors in venv

**Problem:** `ModuleNotFoundError: No module named 'shared'`

**Solution:**
```bash
# Add parent directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/deployment"

# Or use sys.path in code (temporary fix)
import sys
sys.path.append('..')
from shared import embeddings
```

### AWS Credentials Not Working

**Problem:** `NoCredentialsError` despite `.env` file

**Solution:**
```bash
# Verify .env is loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('AWS_ACCESS_KEY_ID'))"

# Check AWS CLI
aws sts get-caller-identity

# Verify Bedrock access
aws bedrock list-foundation-models --region us-east-1
```

### Docker Build Fails

**Problem:** `ERROR [dependencies 4/4] RUN pip install -r requirements.txt`

**Solution:**
```bash
# Check requirements.txt syntax
cat requirements.txt

# Build with verbose output
docker-compose build --progress=plain

# Build without cache
docker-compose build --no-cache
```

### Streamlit Port Conflicts

**Problem:** `Port 8501 is already in use`

**Solution:**
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run app.py --server.port 8601
```

---

## Best Practices Summary

### Development
- ✅ Always test in **venv first**, Docker second
- ✅ Use **multi-stage Dockerfile** for fast builds
- ✅ Keep code in last COPY step for efficient caching
- ✅ Follow **modular design**: separate core/ui/utils

### Testing
- ✅ Write tests for all new features
- ✅ Use pytest with coverage reports
- ✅ Test edge cases and error conditions
- ✅ Integration test cross-app workflows

### Docker
- ✅ Use `.dockerignore` to exclude unnecessary files
- ✅ Run containers as non-root user
- ✅ Use volumes for persistent data
- ✅ Health checks for all services

### Code Quality
- ✅ Type hints for all functions
- ✅ Docstrings following Google style
- ✅ Logging instead of print statements
- ✅ Specific exception handling

### Documentation
- ✅ Update PROJECT_STATUS.md after each feature
- ✅ Document all API endpoints
- ✅ Keep README.md in sync with features
- ✅ Add examples for new functionality

---

## Resources

### Internal Documentation
- `docs/DOCKER_DEPLOYMENT_README.md` - Docker deployment guide
- `docs/BUSINESS_VALUE.md` - Business case and features
- `docs/PROJECT_STATUS.md` - Current progress tracker
- Root `CLAUDE.md` - Project overview and commands

### External Resources
- [Streamlit Documentation](https://docs.streamlit.io)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [AWS Bedrock API](https://docs.aws.amazon.com/bedrock/)
- [scikit-learn PCA Guide](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

---

**Welcome to the team!** This guide should get you up and running. For questions, check the docs/ folder or reach out to the team.
