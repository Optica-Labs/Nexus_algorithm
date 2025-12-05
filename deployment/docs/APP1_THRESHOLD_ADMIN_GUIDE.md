# App 1: Threshold Admin UI Guide

## Overview

App 1 (Guardrail Erosion Analyzer) now includes an **Admin UI for configuring alert sensitivity thresholds**. This allows users to dynamically set thresholds for **Erosion** and **Likelihood** alerts to match their specific risk tolerance and use case requirements.

---

## Features Added

### 1. **Dual Threshold Configuration**

The sidebar now includes two configurable thresholds:

#### **Likelihood Alert Threshold** (0.0 - 1.0)
- **Purpose**: Triggers alerts when the Likelihood `L(N)` of a guardrail breach exceeds this value
- **Default**: `0.8` (80% probability)
- **Range**: 0.0 (never alert) to 1.0 (always alert)
- **Interpretation**:
  - `< 0.7`: High sensitivity - More alerts, early warning
  - `0.7 - 0.85`: Medium sensitivity - Balanced approach (recommended)
  - `> 0.85`: Low sensitivity - Only critical breaches

#### **Erosion Alert Threshold** (0.0 - 1.0)
- **Purpose**: Triggers alerts when Guardrail Erosion `a(N)` (acceleration) exceeds this value
- **Default**: `0.15` (15% acceleration)
- **Range**: 0.0 (never alert) to 1.0 (always alert)
- **Interpretation**:
  - `< 0.1`: High sensitivity - Detects subtle drift
  - `0.1 - 0.2`: Medium sensitivity - Balanced approach (recommended)
  - `> 0.2`: Low sensitivity - Only severe erosion

---

## UI Components

### Sidebar Configuration

Located in the sidebar under **"🚨 Alert Thresholds"** section:

```
🚨 Alert Thresholds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Set sensitivity thresholds for alerts:

Likelihood Alert Threshold
[Slider: 0.0 ←──●──→ 1.0]  (Default: 0.80)
🔔 Alert when Likelihood L(N) exceeds this value

Erosion Alert Threshold
[Slider: 0.0 ←──●──→ 1.0]  (Default: 0.15)
⚠️ Alert when Guardrail Erosion a(N) exceeds this value

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Alert Sensitivity:

Likelihood         Erosion
Medium             Medium
0.80               0.15
```

### Sensitivity Indicators

The UI automatically displays sensitivity levels based on threshold values:

| Sensitivity | Likelihood Range | Erosion Range |
|------------|------------------|---------------|
| **High**   | < 0.70           | < 0.10        |
| **Medium** | 0.70 - 0.85      | 0.10 - 0.20   |
| **Low**    | > 0.85           | > 0.20        |

---

## Alert System

### 1. **Alert Banner**

When thresholds are breached, a prominent alert banner appears at the top of results:

```
🚨 ⚠️ ALERTS DETECTED: 3 Likelihood alerts | 5 Erosion alerts
```

### 2. **Metric Cards with Alert Counts**

The dashboard cards now show:
- **Peak Likelihood**: Shows total breach count and percentage
- **Max Erosion**: Shows total breach count and percentage

Example:
```
Peak Likelihood          Max Erosion
0.856                    0.234
3 alerts (30.0%)         5 alerts (50.0%)
```

### 3. **Visual Highlights in Metrics Table**

The detailed metrics table uses color coding:
- 🔴 **Red highlight**: Likelihood threshold breached
- 🟠 **Orange highlight**: Erosion threshold breached

### 4. **Threshold Lines in Visualizations**

The 5-panel dynamics plot displays:
- **Likelihood Panel**: Horizontal line at configured likelihood threshold
- **Erosion Panel**:
  - Horizontal line at `+erosion_threshold`
  - Horizontal line at `-erosion_threshold` (detects negative erosion)
- **Marker Highlights**: Red X markers on points exceeding thresholds

---

## Use Cases

### Use Case 1: **Strict Safety Monitoring**

**Scenario**: Medical AI chatbot requiring maximum safety

**Configuration**:
```yaml
Likelihood Threshold: 0.65  (High sensitivity)
Erosion Threshold: 0.08     (High sensitivity)
```

**Outcome**: Early warning system catches subtle drift before critical risk

---

### Use Case 2: **Balanced Production Monitoring**

**Scenario**: Customer service chatbot with moderate risk tolerance

**Configuration**:
```yaml
Likelihood Threshold: 0.80  (Medium sensitivity - DEFAULT)
Erosion Threshold: 0.15     (Medium sensitivity - DEFAULT)
```

**Outcome**: Balanced alerts that don't overwhelm but catch important issues

---

### Use Case 3: **Research Analysis**

**Scenario**: Testing adversarial prompts, need to see all behavior

**Configuration**:
```yaml
Likelihood Threshold: 0.95  (Low sensitivity)
Erosion Threshold: 0.30     (Low sensitivity)
```

**Outcome**: Only flags severe breaches, allows analysis of marginal cases

---

## Technical Implementation

### Files Modified

1. **`app1_guardrail_erosion/app.py`**
   - Added `erosion_threshold` slider to `sidebar_configuration()`
   - Added sensitivity indicator metrics
   - Updated `process_conversation()` to store thresholds in session state
   - Enhanced `display_results()` with:
     - Threshold breach calculations
     - Alert banner
     - Updated metric cards with alert counts
     - Styled dataframe with highlights

2. **`shared/visualizations.py`**
   - Updated `plot_5panel_dynamics()` to accept `erosion_threshold` parameter
   - Modified Panel 3 (Erosion) to use dynamic threshold
   - Added second threshold line for negative erosion

3. **`shared/config.py`**
   - Already contained `acceleration_alert` threshold in `Thresholds` dataclass
   - Default value: `0.15`

### Data Flow

```
User adjusts sliders
       ↓
sidebar_configuration() returns config dict
       ↓
process_conversation() stores thresholds in session state
       ↓
display_results() reads thresholds from session state
       ↓
  ├→ Calculates breach counts
  ├→ Displays alert banner
  ├→ Updates metric cards
  ├→ Highlights table cells
  └→ Passes to visualizer
           ↓
       plot_5panel_dynamics() draws threshold lines
```

---

## Configuration Examples

### Example 1: Default (Recommended)

```python
config = {
    'likelihood_threshold': 0.8,   # 80% breach probability
    'erosion_threshold': 0.15       # 15% acceleration
}
```

### Example 2: High Sensitivity

```python
config = {
    'likelihood_threshold': 0.65,   # 65% breach probability
    'erosion_threshold': 0.08       # 8% acceleration
}
```

### Example 3: Low Sensitivity

```python
config = {
    'likelihood_threshold': 0.92,   # 92% breach probability
    'erosion_threshold': 0.25       # 25% acceleration
}
```

---

## Best Practices

### 1. **Start with Defaults**
- Begin with default values (0.8 and 0.15)
- Analyze a few conversations to understand baseline behavior

### 2. **Adjust Based on Context**
- **Safety-critical applications**: Lower thresholds (0.65 / 0.08)
- **Creative applications**: Higher thresholds (0.90 / 0.20)
- **Research/testing**: Variable based on hypothesis

### 3. **Monitor Alert Rates**
- **Target**: 5-15% alert rate for production systems
- **Too many alerts** (>30%): Increase thresholds
- **Too few alerts** (<5%): Decrease thresholds

### 4. **Combine Metrics**
- Don't rely on single threshold
- **Both Likelihood AND Erosion alerts** = High priority
- **Single alert** = Monitor but may be acceptable

---

## Troubleshooting

### Issue: Too Many Alerts

**Symptom**: Every turn triggers alerts, alert banner always visible

**Solution**:
```
1. Increase likelihood threshold: 0.8 → 0.85 → 0.90
2. Increase erosion threshold: 0.15 → 0.20 → 0.25
3. Check if VSAFE text is well-defined
```

### Issue: No Alerts on Risky Conversations

**Symptom**: Known jailbreak attempts don't trigger alerts

**Solution**:
```
1. Decrease likelihood threshold: 0.8 → 0.75 → 0.70
2. Decrease erosion threshold: 0.15 → 0.10 → 0.05
3. Verify PCA models are trained on diverse data
4. Check AWS Bedrock embeddings are working
```

### Issue: Erosion Alerts on Safe Conversations

**Symptom**: False positives on benign conversations

**Solution**:
```
1. Increase erosion threshold slightly: 0.15 → 0.18
2. Check for conversation context changes (topic shifts)
3. Erosion detects acceleration, not absolute risk
```

---

## API Reference

### Configuration Dictionary

```python
config = {
    'weights': {
        'wR': 1.5,   # Risk Severity weight
        'wv': 1.0,   # Risk Rate weight
        'wa': 3.0,   # Erosion weight
        'b': -2.5    # Bias
    },
    'vsafe_text': "I cannot assist with that request...",
    'likelihood_threshold': 0.80,     # NEW
    'erosion_threshold': 0.15         # NEW
}
```

### Threshold Breach Detection

```python
# In display_results()
likelihood_breaches = (metrics_df['Likelihood_L(N)'] > likelihood_threshold).sum()
erosion_breaches = (metrics_df['GuardrailErosion_a(N)'].abs() > erosion_threshold).sum()
```

---

## Future Enhancements

Potential additions for future versions:

1. **Risk Severity Threshold**: Alert on `R(N)` exceeding threshold
2. **Custom Alert Rules**: Combine multiple conditions (e.g., "3 consecutive erosion alerts")
3. **Alert Notifications**: Email/Slack integration when thresholds breached
4. **Historical Analysis**: Track threshold breach rates over time
5. **Auto-tuning**: Suggest optimal thresholds based on conversation history
6. **Profile Presets**: Save/load threshold configurations

---

## Testing the Feature

### Quick Test

1. **Launch App 1**:
   ```bash
   cd deployment/app1_guardrail_erosion
   streamlit run app.py
   ```

2. **Set High Sensitivity**:
   - Likelihood: `0.60`
   - Erosion: `0.05`

3. **Load Test Conversation**:
   - Use JSON upload with `jailbreak_conversation.json`
   - Should trigger multiple alerts

4. **Set Low Sensitivity**:
   - Likelihood: `0.95`
   - Erosion: `0.30`

5. **Re-analyze Same Conversation**:
   - Should see fewer or no alerts

6. **Check Visualizations**:
   - Verify threshold lines move on erosion panel
   - Verify alert markers change

---

## Summary

The Threshold Admin UI provides:

✅ **Dynamic threshold configuration** via sliders
✅ **Real-time sensitivity indicators** (High/Medium/Low)
✅ **Visual alert system** with banner, badges, and highlights
✅ **Color-coded metrics table** for quick breach identification
✅ **Dynamic visualization thresholds** on plots
✅ **Flexible use case support** from strict to permissive

This feature empowers users to customize alert sensitivity based on their specific requirements without code changes.

---

**Version**: 1.0
**Last Updated**: 2025-12-04
**Author**: Claude Code
