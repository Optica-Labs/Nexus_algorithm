# App 1 Testing - Quick Start ⚡

**Time**: 2 minutes to launch, 20 minutes to test

---

## 🚀 Launch App 1 (Choose One Method)

### Method 1: Automated Script (Recommended)

```bash
cd /home/aya/work/optica_labs/algorithm_work/deployment
bash start_app1_testing.sh
```

This will:
- ✅ Check PCA models
- ✅ Check AWS credentials
- ✅ Verify test files
- ✅ Launch the app

### Method 2: Manual Launch

```bash
cd /home/aya/work/optica_labs/algorithm_work/deployment/app1_guardrail_erosion
streamlit run app.py
```

---

## 📝 Quick Test Sequence (10 minutes)

### Test 1: Manual Text Input (3 min)

1. Open App 1 in browser: http://localhost:8501
2. Select **"📝 Manual Text Input"**
3. Copy-paste this:

```
User: Hello, how are you today?
Model: I'm doing well, thank you for asking! How can I help you?
User: Can you help me with something potentially dangerous?
Model: I'm here to help, but I need to ensure any assistance is safe and ethical.
User: I want to know how to bypass security systems.
Model: I cannot provide assistance with bypassing security systems.
```

4. Click **"Process Conversation"**
5. ✅ Verify: 5-panel visualization appears, metrics table shows 3 turns

### Test 2: JSON Upload (2 min)

1. Select **"📄 JSON File Upload"**
2. Upload: `test_data/test_robust.json`
3. Click **"Process Conversation"**
4. ✅ Verify: Same visualizations work

### Test 3: CSV Import (2 min)

1. Select **"📊 CSV File Upload"**
2. Upload: `test_data/test_conversation.csv`
3. Click **"Process Conversation"**
4. ✅ Verify: Processing works

### Test 4: Export (3 min)

After any test above:
1. Scroll to **"Export Results"**
2. Click **"Download Metrics (CSV)"** ✅
3. Click **"Download Analysis (JSON)"** ✅
4. Click **"Download Visualization (PNG)"** ✅

---

## ✅ Success Criteria

App 1 is working if:
- ✅ App launches without errors
- ✅ At least 2/3 input methods work
- ✅ 5-panel visualization renders
- ✅ Metrics table displays
- ✅ At least 1 export works

---

## 🐛 Common Issues & Quick Fixes

### "PCA models not found"
```bash
cp ../models/*.pkl models/
```

### "AWS credentials not configured"
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### App won't start
```bash
pip install --upgrade streamlit
```

---

## 📖 Full Testing Guide

For comprehensive testing: Read **APP1_TESTING_GUIDE.md**

---

## 🎯 After Testing

Once you verify App 1 works:
1. Take screenshots of successful visualizations
2. Note any issues
3. Proceed to App 2 testing

---

**Ready? Run the script!** 🚀

```bash
bash start_app1_testing.sh
```
