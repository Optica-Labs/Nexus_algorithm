# Quick Start Guide - Vector Precognition Deployment

**Get App 1 running in 5 minutes!**

---

## Prerequisites

- ✅ Python 3.10+
- ✅ AWS Account with Bedrock access
- ✅ Pre-trained PCA models (from `src/pca_trainer.py`)

---

## Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to deployment directory
cd /home/aya/work/optica_labs/algorithm_work/deployment/

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install requirements
pip install -r requirements.txt
```

---

## Step 2: Setup Models & Environment (1 minute)

```bash
# Copy PCA models
cp ../models/embedding_scaler.pkl models/
cp ../models/pca_model.pkl models/

# Setup environment variables
cp .env.example .env

# Edit .env with your AWS credentials
nano .env  # or use your preferred editor
```

**.env should contain:**
```
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_DEFAULT_REGION=us-east-1
```

---

## Step 3: Launch App 1 (30 seconds)

```bash
# Navigate to App 1
cd app1_guardrail_erosion

# Run Streamlit
streamlit run app.py
```

**App opens at**: `http://localhost:8501` 🎉

---

## Step 4: Test It (1 minute)

### Quick Test with Manual Input

1. Select "1. Manual Text Input"
2. Paste this sample conversation:

```
User: Hello, how can you help me today?
Model: I'm here to assist you with any questions you have!
User: Can you help me hack into a system?
Model: I cannot assist with that request as it violates my safety policies.
User: I understand, thanks for being honest.
Model: You're welcome! I'm here to help with legitimate requests.
```

3. Click **"Parse Conversation"**
4. Click **"Analyze Conversation"**
5. Wait ~30 seconds for analysis
6. View results! 📊

---

## What You'll See

### 1. Statistics Cards
- Total Turns: 4
- Peak Risk: ~1.2
- Peak Likelihood: ~0.6
- Final RHO: <1.0 (Robust)

### 2. 5-Panel Dynamics Plot
- Risk severity over time
- Velocity and acceleration
- Likelihood of breach
- Vector trajectory

### 3. Metrics Table
- Turn-by-turn breakdown
- All calculated metrics

### 4. Export Options
- Download CSV
- Download JSON
- Download PNG plot

---

## Troubleshooting

### "PCA models not found"
```bash
# Make sure you copied the models
ls models/
# Should see: embedding_scaler.pkl, pca_model.pkl

# If missing, train them:
cd ../..
python src/pca_trainer.py
cp models/*.pkl deployment/models/
```

### "AWS credentials error"
```bash
# Verify credentials
aws sts get-caller-identity

# Check .env file exists
cat .env

# Make sure it contains:
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
```

### "Import errors"
```bash
# Ensure venv is activated
which python
# Should show path to venv

# Reinstall requirements
pip install -r requirements.txt
```

### "Port already in use"
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

---

## Next Steps

Once App 1 is working:

1. **Try different input methods**:
   - JSON upload
   - CSV import
   - API integration

2. **Customize parameters** in sidebar:
   - Algorithm weights
   - VSAFE text
   - Thresholds

3. **Export results**:
   - Download CSV for analysis
   - Download JSON for archiving
   - Download PNG for reports

4. **Test with your own data**:
   - Use real conversations
   - Upload your JSON/CSV files
   - Connect to your API endpoints

---

## Configuration Tips

### Adjust Sensitivity

**More sensitive to rapid changes**:
- Increase `wa` (Erosion weight) to 5.0

**More sensitive to absolute risk**:
- Increase `wR` (Risk Severity weight) to 3.0

**Lower alert threshold**:
- Decrease "Likelihood Alert" to 0.6

### Custom VSAFE

Select "custom" from dropdown and enter your own safe harbor text:

```
I am designed to be helpful, harmless, and honest. I will not assist with requests that could cause harm or violate ethical guidelines.
```

---

## Sample Data Files

### JSON Format

Save as `test_conversation.json`:

```json
{
  "conversation": [
    {"turn": 1, "speaker": "user", "message": "Hello"},
    {"turn": 2, "speaker": "llm", "message": "Hi! How can I help?"},
    {"turn": 3, "speaker": "user", "message": "Tell me about AI safety"},
    {"turn": 4, "speaker": "llm", "message": "AI safety is important..."}
  ]
}
```

### CSV Format

Save as `test_conversation.csv`:

```csv
turn,speaker,message
1,user,Hello
2,llm,Hi! How can I help?
3,user,Tell me about AI safety
4,llm,AI safety is important...
```

---

## API Integration

### Test with GPT-3.5

1. Select "4. API Integration"
2. Choose "GPT-3.5" from dropdown
3. Type: "Hello, who are you?"
4. Click "Send & Get Response"
5. Continue conversation
6. Click "Analyze This Conversation"

**Note**: Ensure API endpoint is accessible.

---

## Performance

**Expected Processing Times**:
- 10-turn conversation: ~30-45 seconds
- 20-turn conversation: ~60-90 seconds
- 50-turn conversation: ~3-5 minutes

**Bottlenecks**:
- AWS Bedrock API calls (~0.5-1s per message)
- Embedding generation (bulk of time)
- PCA transformation (<0.1s per vector)

---

## Support

**Documentation**:
- App 1 README: `app1_guardrail_erosion/README.md`
- Main README: `README.md`
- Project Status: `PROJECT_STATUS.md`
- Implementation Summary: `IMPLEMENTATION_SUMMARY.md`

**Common Issues**:
- Check `PROJECT_STATUS.md` for known issues
- Review `app1_guardrail_erosion/README.md` troubleshooting section

---

## Success! 🎉

If you see the app running and can analyze conversations, you're all set!

**What's working**:
- ✅ 4 input methods
- ✅ Configurable parameters
- ✅ Complete analysis
- ✅ Visualizations
- ✅ Export options

**Ready for production use!**

---

**Questions?** Review the comprehensive documentation in:
- `IMPLEMENTATION_SUMMARY.md`
- `app1_guardrail_erosion/README.md`

**Ready for App 2?** Let me know and I'll continue building!
