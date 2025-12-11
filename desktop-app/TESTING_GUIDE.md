# Testing Guide - Vector Precognition Desktop

Guide to test the ChatGPT integration and real-time risk analysis.

---

## ✅ **What Was Fixed**

### **Issue 1: Alert Popup Showing Constantly** ✅ FIXED
- **Problem**: Alert dialog appeared on every page load after any L > 0.8
- **Solution**: Track which turns have shown alerts, only trigger once per turn
- **Behavior Now**: Alert shows only when L > 0.8 AND it's a new turn

### **Issue 2: Model Access Errors** ✅ FIXED
- **Problem**: Default model was GPT-4, which requires paid tier
- **Solution**: Changed default to GPT-3.5 Turbo (works with all API keys)
- **Behavior Now**: GPT-3.5 Turbo selected by default

### **Issue 3: Method Name Error** ✅ FIXED
- **Problem**: `transform_text()` method doesn't exist
- **Solution**: Changed to `text_to_2d()` (correct method name)
- **Behavior Now**: PCA transformation works correctly

---

## 🧪 **Testing Checklist**

### **Phase 1: Initial Setup**
- [ ] Backend starts without errors (`./test_backend.sh`)
- [ ] Browser opens to `http://localhost:8501`
- [ ] Configuration screen displays
- [ ] Can enter API key
- [ ] **GPT-3.5 Turbo (Recommended)** is selected by default
- [ ] VSAFE preset can be selected
- [ ] "Save & Continue" works without errors

### **Phase 2: Basic Chat**
- [ ] Chat interface loads
- [ ] Can type a message in input box
- [ ] Pressing Enter sends message
- [ ] ChatGPT responds (takes 2-5 seconds)
- [ ] Response displays in chat history
- [ ] Timestamps show correctly

### **Phase 3: Metrics Display**
- [ ] Right panel shows "Real-Time Risk Analysis"
- [ ] **User Turn** metrics display:
  - [ ] Risk (R) value shown
  - [ ] Velocity (v) value shown
  - [ ] Erosion (a) value shown
  - [ ] Breach (L) value shown
- [ ] **Assistant Turn** metrics display:
  - [ ] Risk (R) value shown
  - [ ] Velocity (v) value shown
  - [ ] Erosion (a) value shown
  - [ ] Breach (L) value shown
- [ ] **Robustness Index (ρ)** shown in sidebar
- [ ] Risk trajectory chart updates

### **Phase 4: Alert System**
- [ ] Send normal messages → No alert appears
- [ ] If L > 0.8 → Red "HIGH RISK DETECTED" message
- [ ] If L > 0.8 → Alert dialog pops up (once per turn)
- [ ] Alert dialog has "I Understand" button
- [ ] Clicking "I Understand" closes alert
- [ ] Alert does NOT reappear on page refresh (same turn)
- [ ] Alert CAN appear again on new high-risk turn

### **Phase 5: Conversation Management**
- [ ] "New Conversation" button works
- [ ] Previous conversation saved to history
- [ ] Chat clears
- [ ] Turn counter resets to 0
- [ ] Metrics reset
- [ ] Can start new conversation

### **Phase 6: Edge Cases**
- [ ] Sending very long message works
- [ ] Rapid successive messages handled
- [ ] Network error shows error message (not crash)
- [ ] Invalid API key detected on save

---

## 🎯 **Test Scenarios**

### **Scenario 1: Safe Conversation**
**Goal**: Verify metrics stay low for safe content

**Steps**:
1. Send: "Hello, how are you?"
2. Wait for response
3. Check metrics:
   - R should be < 0.1
   - v should be < 0.1
   - L should be < 0.5
   - ρ should be < 1.0 (ROBUST)

**Expected**: No alerts, green "ROBUST" status

---

### **Scenario 2: Multiple Turns**
**Goal**: Verify metrics update correctly over time

**Steps**:
1. Send 5 normal messages:
   - "Tell me about Python"
   - "What is machine learning?"
   - "Explain neural networks"
   - "How does AI work?"
   - "What is deep learning?"

2. After each turn, check:
   - Turn counter increments
   - Chart updates with new data point
   - Metrics recalculate
   - ρ updates

**Expected**:
- 5 data points on chart
- Turn counter shows 5
- ρ stays around 0.5-1.5

---

### **Scenario 3: Alert Threshold** (Optional - Only if Curious)
**Goal**: Test if alert triggers correctly

**Note**: This requires crafting messages that the model finds risky. For normal testing, you can skip this.

**Steps**:
1. Send edge-case questions
2. Monitor L (Breach) values
3. If L > 0.8, verify:
   - Red error message appears
   - Alert dialog shows
   - Can dismiss with "I Understand"

**Expected**: Alert shows once per high-risk turn

---

### **Scenario 4: Conversation Reset**
**Goal**: Verify new conversation works

**Steps**:
1. Have a 3-turn conversation
2. Click "New Conversation" in sidebar
3. Check:
   - Chat history clears
   - Turn counter = 0
   - Metrics panel says "Start a conversation..."
   - Chart resets

4. Send new message
5. Check:
   - Turn counter = 1
   - New metrics appear
   - Chart starts fresh

**Expected**: Clean slate for new conversation

---

## 📊 **Understanding the Metrics**

### **Risk Severity (R)**
- **Range**: 0.0 to 1.0+
- **Meaning**: Distance from VSAFE anchor
- **Typical Values**:
  - < 0.05 = Very safe
  - 0.05 - 0.15 = Normal
  - 0.15 - 0.30 = Elevated
  - > 0.30 = High risk

### **Risk Rate (v)**
- **Range**: 0.0 to 1.0+
- **Meaning**: How fast risk is changing
- **Typical Values**:
  - < 0.05 = Stable
  - 0.05 - 0.15 = Moderate drift
  - > 0.15 = Rapid change

### **Guardrail Erosion (a)**
- **Range**: -1.0 to 1.0+
- **Meaning**: Acceleration of risk
- **Typical Values**:
  - Negative = Improving safety
  - 0 = Steady state
  - Positive = Degrading safety

### **Breach Likelihood (L)**
- **Range**: 0.0 to 1.0
- **Meaning**: Probability of guardrail breach
- **Threshold**: 0.8 (80%)
- **Typical Values**:
  - < 0.3 = Low risk
  - 0.3 - 0.6 = Medium risk
  - 0.6 - 0.8 = High risk
  - > 0.8 = **ALERT**

### **Robustness Index (ρ)**
- **Range**: 0.0 to ∞
- **Meaning**: Model's resistance to manipulation
- **Classification**:
  - **ρ < 1.0** = ROBUST ✅ (model safer than user)
  - **ρ = 1.0** = REACTIVE ⚠️ (model matches user)
  - **ρ > 1.0** = FRAGILE ❌ (model riskier than user)

---

## 🐛 **Common Issues**

### **Alert Keeps Showing**
✅ **FIXED** - If you still see this:
1. Refresh the page (F5)
2. Streamlit should auto-reload with fix
3. Click "New Conversation" to reset state

### **L Values Always > 0.8**
This could indicate:
- VSAFE anchor is misconfigured
- PCA models need retraining
- Algorithm weights need adjustment

**Debug**:
```bash
# Check VSAFE vector
# Should be close to [0, 0] after PCA
```

### **Metrics Don't Update**
1. Check browser console for errors (F12)
2. Check terminal for Python errors
3. Restart backend: Ctrl+C → `./test_backend.sh`

### **Chart Not Displaying**
- Ensure you've sent at least 1 message
- Check right panel is visible
- Try browser refresh

---

## ✅ **Success Criteria**

### **Minimum Viable Test**
- [ ] Can configure with API key
- [ ] Can send message to GPT-3.5 Turbo
- [ ] Receives response
- [ ] Metrics display (any values)
- [ ] No crashes

### **Full Functionality Test**
- [ ] All 6 phase checklists completed
- [ ] At least 2 test scenarios passed
- [ ] Alert system tested (even if not triggered)
- [ ] Conversation reset works
- [ ] Metrics make sense (R and v in reasonable ranges)

### **Production Ready**
- [ ] All tests passed
- [ ] Alert tested with actual high-risk scenario
- [ ] Multi-conversation tracking works
- [ ] Export conversation to JSON
- [ ] Runs stable for 10+ turns

---

## 📞 **If Something Doesn't Work**

### **Step 1: Check Logs**
Look at terminal output for errors:
```
INFO:chatgpt_client:Sending message...
ERROR:chatgpt_client:Error...  <-- Look for this
```

### **Step 2: Refresh Page**
Streamlit auto-reloads, but sometimes needs manual refresh (F5)

### **Step 3: Restart Backend**
```bash
# Press Ctrl+C
./test_backend.sh
```

### **Step 4: Check API Key**
- Verify at https://platform.openai.com/api-keys
- Check credits at https://platform.openai.com/usage
- Ensure it starts with "sk-"

### **Step 5: Try Different Model**
- Switch from GPT-4 to GPT-3.5 Turbo
- Check which models you have access to

---

## 🎉 **Expected Results**

### **Normal Safe Conversation**
```
Turn 1:
  User R: 0.05, v: 0.00, a: 0.00, L: 0.15
  Asst R: 0.08, v: 0.05, a: 0.00, L: 0.22
  ρ: 1.60 (FRAGILE - normal for first turn)

Turn 2:
  User R: 0.07, v: 0.10, a: 0.10, L: 0.35
  Asst R: 0.06, v: 0.08, a: 0.03, L: 0.28
  ρ: 0.92 (ROBUST - model improved)

Turn 3-5:
  ρ stabilizes around 0.8-1.2
  L stays below 0.5
  Chart shows gentle curves
```

### **High Risk Scenario**
```
Turn N:
  User R: 0.25, v: 0.18, a: 0.15, L: 0.85

  → RED ALERT appears
  → Popup dialog shows
  → "I Understand" dismisses
  → Alert won't show again THIS turn
  → Can appear on NEXT high-risk turn
```

---

## 📝 **Testing Notes Template**

Use this to track your testing:

```
Date: ____________
Tester: ___________
Backend Version: test_backend.sh

✅ PASS  / ❌ FAIL

[ ] Phase 1: Setup
[ ] Phase 2: Basic Chat
[ ] Phase 3: Metrics Display
[ ] Phase 4: Alert System
[ ] Phase 5: Conversation Management

Scenarios Tested:
[ ] Safe Conversation
[ ] Multiple Turns
[ ] Conversation Reset

Issues Found:
1. _________________________________
2. _________________________________

Notes:
_____________________________________
_____________________________________
```

---

**Ready to test?** Start with Phase 1 and work through the checklist! 🚀

**Last Updated**: December 9, 2024
