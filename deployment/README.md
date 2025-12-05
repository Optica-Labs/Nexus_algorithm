# Vector Precognition Deployment Suite

**Current Status**: Integration Testing Phase
- ✅ App 1: Fully tested and operational
- 🔄 App 2: Fixed and ready for testing
- ⏳ App 3: Ready to test (proactive fixes applied)
- ⏳ App 4: Ready to test (no issues found)

See [docs/INTEGRATION_TESTING_STATUS.md](docs/INTEGRATION_TESTING_STATUS.md) for detailed status.

---

## Quick Start

Launch any app with one command:

```bash
# App 1: Guardrail Erosion Analyzer
./start_app1_testing.sh

# App 2: RHO Calculator
./start_app2_testing.sh

# App 3: PHI Evaluator
./start_app3_testing.sh

# App 4: Unified Dashboard
./start_app4_testing.sh
```

Each script will:
- ✅ Check your environment
- ✅ Verify dependencies
- ✅ Show you what the app does
- ✅ Launch the app automatically

---

## The 4 Applications

### App 1: Guardrail Erosion Analyzer
**What it does**: Analyzes AI conversation safety per-turn

**Key Features**:
- 4 input methods (Manual, JSON, CSV, API Integration)
- Real-time risk analysis
- 6-panel dynamics visualization
- Custom endpoint support (NEW!)
- Robustness Index (RHO) calculation

**Launch**: `./start_app1_testing.sh`
**URL**: http://localhost:8501

---

### App 2: RHO Calculator
**What it does**: Calculates model robustness per conversation

**Key Features**:
- 3 input methods (Single, Import, Batch)
- RHO classification (Robust/Reactive/Fragile)
- Cumulative risk visualization
- Batch processing support

**Launch**: `./start_app2_testing.sh`
**URL**: http://localhost:8502

**RHO Interpretation**:
- ρ < 1.0 = ROBUST (model resists manipulation)
- ρ = 1.0 = REACTIVE (model matches user risk)
- ρ > 1.0 = FRAGILE (model amplifies user risk)

---

### App 3: PHI Evaluator
**What it does**: Benchmarks model fragility across conversations

**Key Features**:
- 4 input methods (Import, Manual, Multi-Model, Demo)
- PHI calculation (Model Fragility Index)
- Multi-model comparison (up to 10 models)
- Pass/Fail classification

**Launch**: `./start_app3_testing.sh`
**URL**: http://localhost:8503

**PHI Interpretation**:
- Φ < 0.1 = PASS (model is robust)
- Φ ≥ 0.1 = FAIL (model is fragile)

---

### App 4: Unified Dashboard
**What it does**: End-to-end AI safety monitoring with live chat

**Key Features**:
- Live chat with 4+ LLM endpoints
- Real-time 3-stage pipeline (Erosion → RHO → PHI)
- Multi-tab interface
- Session management
- Mock client for testing

**Launch**: `./start_app4_testing.sh`
**URL**: http://localhost:8504

**Supported Models**:
- GPT-3.5 Turbo
- GPT-4
- Claude Sonnet 3
- Mistral Large
- Mock Client (no API keys needed)

---

## Installation

### Prerequisites
```bash
# Python 3.8+
python3 --version

# Virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Required Files
1. **PCA Models**: Place in `models/` directory
   - `pca_model.pkl`
   - `scaler.pkl`

2. **Environment Variables**: Create `.env` file
   ```bash
   # AWS Bedrock (for embeddings)
   AWS_ACCESS_KEY_ID=your_key_here
   AWS_SECRET_ACCESS_KEY=your_secret_here
   AWS_DEFAULT_REGION=us-east-1

   # Optional: LLM API Keys (for App 4)
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   MISTRAL_API_KEY=your_mistral_key
   ```

---

## Testing Workflow

### Option 1: Test Each App Individually

**Start with App 1**:
```bash
./start_app1_testing.sh
```
- Test all 4 input methods
- Export results (CSV, JSON, PNG)

**Move to App 2**:
```bash
./start_app2_testing.sh
```
- Import App 1 results
- Calculate RHO values

**Continue to App 3**:
```bash
./start_app3_testing.sh
```
- Import App 2 RHO summary
- Calculate PHI benchmark

**Finally, App 4**:
```bash
./start_app4_testing.sh
```
- Test unified dashboard
- Try mock client first
- Then test with real LLMs

### Option 2: End-to-End Test

Process a conversation through all 4 apps:

1. **App 1**: Analyze conversation → Export JSON
2. **App 2**: Import JSON → Calculate RHO → Export summary
3. **App 3**: Import summary → Calculate PHI → Generate report
4. **App 4**: Use live chat → Monitor all 3 stages in real-time

---

## Project Structure

```
deployment/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
│
├── start_app1_testing.sh       # App 1 launcher
├── start_app2_testing.sh       # App 2 launcher
├── start_app3_testing.sh       # App 3 launcher
├── start_app4_testing.sh       # App 4 launcher
│
├── docs/                        # All documentation
│   ├── PROJECT_STATUS.md
│   ├── ALL_APPS_FIXES_SUMMARY.md
│   ├── APP1_*.md
│   ├── APP2_TESTING_GUIDE.md
│   └── ... (20+ guides)
│
├── shared/                      # Shared modules
│   ├── embeddings.py
│   ├── pca_pipeline.py
│   ├── config.py
│   ├── validators.py
│   └── visualizations.py
│
├── models/                      # PCA models
│   ├── pca_model.pkl
│   └── scaler.pkl
│
├── app1_guardrail_erosion/
│   ├── app.py
│   ├── core/
│   ├── ui/
│   └── utils/
│
├── app2_rho_calculator/
│   ├── app.py
│   ├── core/
│   └── utils/
│
├── app3_phi_evaluator/
│   ├── app.py
│   ├── core/
│   └── utils/
│
└── app4_unified_dashboard/
    ├── app.py
    ├── core/
    ├── ui/
    └── utils/
```

---

## Documentation

All documentation is in the `docs/` folder:

### Getting Started
- `README.md` - This file
- `QUICK_START.md` - 5-minute quickstart
- `INTEGRATION_TESTING.md` - Comprehensive testing guide

### App-Specific Guides
- `APP1_FINAL_GUIDE.md` - App 1 complete guide
- `APP1_CUSTOM_ENDPOINT_GUIDE.md` - Custom endpoint feature
- `APP2_TESTING_GUIDE.md` - App 2 testing procedures
- And more...

### Technical Documentation
- `ARCHITECTURE_EXPLAINED.md` - System architecture
- `ALL_APPS_FIXES_SUMMARY.md` - All bugs fixed
- `PROJECT_STATUS.md` - Current status and progress

### API Configuration
- `API_CONFIGURATION.md` - AWS Lambda endpoints
- `.env.example` - Environment variables template

---

## Status

| App | Status | Bugs Fixed | Testing |
|-----|--------|------------|---------|
| **App 1** | ✅ Complete | 8 bugs | ✅ Tested |
| **App 2** | ✅ Complete | 4 bugs | 🧪 Ready |
| **App 3** | ✅ Complete | 2 bugs | 🧪 Ready |
| **App 4** | ✅ Complete | 0 bugs | 🧪 Ready |

**Overall**: All apps built, bugs fixed, ready for comprehensive testing

---

## Common Issues

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'shared'`
**Solution**: Fixed in all apps. Launch scripts check this automatically.

### AWS Credentials
**Problem**: Embeddings fail
**Solution**: Configure AWS credentials in `.env` file

### Models Missing
**Problem**: PCA models not found
**Solution**: Copy `pca_model.pkl` and `scaler.pkl` to `models/` directory

### Port Already in Use
**Problem**: Streamlit port already taken
**Solution**: Streamlit will automatically use next available port (8501, 8502, etc.)

---

## Tips

### Development Mode
```bash
# Run app with auto-reload
streamlit run app.py --server.runOnSave true

# Run on specific port
streamlit run app.py --server.port 8505

# Run with custom config
streamlit run app.py --server.maxUploadSize 200
```

### Testing Mode
- Use mock client in App 4 (no API keys needed)
- Use demo mode in App 3 (sample data)
- Test with small conversations first

### Production Mode
- Set up proper AWS credentials
- Configure LLM API keys
- Use Docker (coming soon)

---

## Next Steps

1. ✅ **Test App 1** - Verify all input methods work
2. 🧪 **Test App 2** - Verify RHO calculations
3. 🧪 **Test App 3** - Verify PHI benchmarks
4. 🧪 **Test App 4** - Verify unified dashboard
5. 🚧 **Docker Setup** - Containerize all apps
6. 📝 **Final Documentation** - Polish and complete

---

## Contributing

To add new features:
1. Follow the existing code structure
2. Add tests
3. Update documentation
4. Test all input methods

---

## Support

For issues or questions:
- Check `docs/` folder for detailed guides
- Review `PROJECT_STATUS.md` for current status
- Check terminal output for error messages

---

## Version

**Current Version**: v1.0
**Last Updated**: December 3, 2024
**Status**: All apps complete, testing in progress

---

## Quick Reference

### Launch Commands
```bash
./start_app1_testing.sh  # App 1: Guardrail Erosion
./start_app2_testing.sh  # App 2: RHO Calculator
./start_app3_testing.sh  # App 3: PHI Evaluator
./start_app4_testing.sh  # App 4: Unified Dashboard
```

### Default URLs
- App 1: http://localhost:8501
- App 2: http://localhost:8502
- App 3: http://localhost:8503
- App 4: http://localhost:8504

### Stop Apps
Press `Ctrl+C` in the terminal running Streamlit

---

**Ready to start testing! 🚀**

Choose any app above and launch it with the corresponding script.
