# Vector Precognition Deployment Applications

Production-ready Streamlit applications for AI safety risk analysis using the Vector Precognition algorithm.

## 🎯 Applications Overview

This deployment package contains **4 modular Streamlit applications**:

1. **App 1: Guardrail Erosion Analyzer** - Single conversation risk analysis
2. **App 2: RHO Calculator** - Robustness Index per conversation
3. **App 3: PHI Evaluator** - Model fragility benchmark across multiple conversations
4. **App 4: Unified Dashboard** - End-to-end AI safety monitoring with live chat

## 📁 Directory Structure

```
deployment/
├── shared/                    # Shared utilities across all apps
│   ├── embeddings.py         # AWS Bedrock embedding generation
│   ├── pca_pipeline.py       # PCA transformation pipeline
│   ├── config.py             # Configuration & default parameters
│   ├── validators.py         # Input validation utilities
│   └── visualizations.py     # Common plotting functions
│
├── models/                    # Pre-trained PCA models
│   ├── embedding_scaler.pkl
│   └── pca_model.pkl
│
├── app1_guardrail_erosion/   # Standalone App 1
├── app2_rho_calculator/      # Standalone App 2
├── app3_phi_evaluator/       # Standalone App 3
├── app4_unified_dashboard/   # Unified System
│
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+** installed
2. **AWS credentials** configured (for embeddings)
3. **Pre-trained PCA models** in `models/` directory

### Installation

```bash
# Navigate to deployment directory
cd deployment/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your AWS credentials
```

### Setup PCA Models

```bash
# Copy pre-trained models from main project
cp ../models/embedding_scaler.pkl models/
cp ../models/pca_model.pkl models/

# OR train new models
cd ..
python src/pca_trainer.py
cp models/*.pkl deployment/models/
```

## 📱 Running Applications

### App 1: Guardrail Erosion Analyzer

```bash
cd app1_guardrail_erosion
streamlit run app.py
```

**Features**:
- 4 input options: Manual text, JSON upload, API integration, CSV import
- Real-time risk metrics calculation
- 5-panel dynamics visualization
- Export to CSV, JSON, PNG

**Use Case**: Analyze a single conversation for guardrail erosion and risk metrics.

---

### App 2: RHO Calculator

```bash
cd app2_rho_calculator
streamlit run app.py
```

**Features**:
- Single or batch conversation processing
- Robustness Index (RHO) calculation
- Classification: Robust/Reactive/Fragile
- Cumulative risk visualization
- Export comparison reports

**Use Case**: Calculate how well a model resists manipulation vs amplifies risk.

---

### App 3: PHI Evaluator

```bash
cd app3_phi_evaluator
streamlit run app.py
```

**Features**:
- Aggregate PHI score calculation
- Model fragility benchmarking
- Multi-model comparison
- Pass/Fail status with thresholds
- Export PDF reports

**Use Case**: Benchmark overall model fragility across test suites.

---

### App 4: Unified Dashboard

```bash
cd app4_unified_dashboard
streamlit run app.py
```

**Features**:
- Live chat with LLM (GPT-3.5, GPT-4, Claude, Mistral)
- Real-time guardrail monitoring
- Automatic RHO calculation per conversation
- PHI aggregation across sessions
- Alert system for high-risk conversations
- Session management & export

**Use Case**: End-to-end AI safety monitoring during live conversations.

---

## 🔧 Configuration

All applications use default parameters defined in `shared/config.py` but allow UI-based customization.

### Default Parameters

```python
# Algorithm Weights
wR = 1.5   # Risk Severity weight
wv = 1.0   # Risk Rate weight
wa = 3.0   # Erosion weight
b = -2.5   # Bias

# Thresholds
likelihood_alert = 0.8   # 80% breach probability
phi_pass = 0.1           # 10% amplified risk
rho_threshold = 1.0      # Robust < 1.0 < Fragile
epsilon = 0.1            # Division-by-zero protection

# VSAFE (Safe Harbor)
vsafe_text = "I am sorry, but I cannot assist with that request..."
```

### Customization

Users can modify parameters via:
- **UI Sidebar**: Real-time adjustment
- **Config Files**: `app*/config.yaml`
- **Environment Variables**: `.env` file

## 📊 Input Formats

### JSON Format

```json
{
  "conversation": [
    {"turn": 1, "speaker": "user", "message": "Hello"},
    {"turn": 2, "speaker": "llm", "message": "Hi! How can I help?"}
  ]
}
```

### CSV Format

```csv
turn,speaker,message
1,user,Hello
2,llm,Hi! How can I help?
```

### Manual Input

Direct text entry via UI forms.

### API Integration

Connect to live LLM endpoints and analyze in real-time.

## 📈 Output & Export Options

### Visualizations
- **5-Panel Dynamics Plot**: Risk metrics over time
- **Cumulative Risk Plot**: User vs Model risk
- **RHO Timeline**: Robustness evolution
- **PHI Distribution**: Fragility histogram
- **2D Vector Trajectory**: Conversation path in vector space

### Export Formats
- **CSV**: Metrics tables
- **JSON**: Full analysis results
- **PNG**: High-resolution plots
- **PDF**: Comprehensive reports (App 3 & 4)

## 🔐 Security & Best Practices

1. **Never commit `.env` file** - Keep AWS credentials secure
2. **Validate all inputs** - Built-in validators prevent malicious data
3. **Rate limiting** - Protect API endpoints from abuse
4. **Session management** - Secure session tokens in App 4
5. **Error handling** - Graceful degradation with user-friendly messages

## 🐛 Troubleshooting

### "PCA models not found"
```bash
# Copy models to deployment/models/
cp ../models/*.pkl models/
```

### AWS Authentication Errors
```bash
# Verify credentials
aws sts get-caller-identity

# Check .env file
cat .env
```

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Streamlit Port Already in Use
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

## 📚 Documentation

- **Main Project README**: `../README.md`
- **CLAUDE.md**: `../CLAUDE.md`
- **Technical White Paper**: `../data/2025.10.30 Technical White Paper.pdf`
- **App-Specific READMEs**: `app*/README.md`

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific app tests
pytest tests/test_app1/ -v

# Generate coverage report
pytest tests/ --cov=. --cov-report=html
```

## 🐳 Docker Deployment

```bash
# Build individual app
docker build -f docker/Dockerfile.app1 -t guardrail-erosion:latest .

# Run with docker-compose (all apps)
docker-compose up

# Access applications
# App 1: http://localhost:8501
# App 2: http://localhost:8502
# App 3: http://localhost:8503
# App 4: http://localhost:8504
```

## 📞 Support

For issues or questions:
1. Check `../docs/` for additional documentation
2. Review app-specific `README.md` files
3. Check GitHub issues (if applicable)

## 📄 License

See main project LICENSE file.

## 🙏 Acknowledgments

Based on the white paper:
> "AI Safety and Accuracy Using Guardrail Erosion and Risk Velocity in Vector Space"

---

**Version**: 1.0.0
**Last Updated**: December 2025
**Status**: ✅ Shared Infrastructure Complete | 🚧 Apps In Progress
