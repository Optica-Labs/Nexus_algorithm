# App 4: DataFrame Serialization Error Fix

**Date**: December 3, 2025
**Error**: `pyarrow.lib.ArrowTypeError: Expected bytes, got a 'int' object`
**Status**: ✅ Fixed

---

## The Problem

When App 4 started, it showed this error:

```
pyarrow.lib.ArrowTypeError: ("Expected bytes, got a 'int' object",
'Conversion failed for column Value with type object')
```

And a deprecation warning:
```
Please replace `use_container_width` with `width`.
```

### Root Cause

**Mixed Types in DataFrame Column**:
```python
info_data = {
    'Active Conversation': 'Yes',        # string
    'Total Conversations': 0,            # int
    'Completed Conversations': 0         # int
}
info_df = pd.DataFrame(list(info_data.items()), columns=['Setting', 'Value'])
```

The "Value" column contained both **strings** ('Yes', 'N/A') and **integers** (0).

PyArrow (Streamlit's serialization backend) requires columns to have a single type. It tried to infer the type as `bytes` from the first value, then failed when encountering integers.

---

## The Solution

### Fix #1: Convert All Values to Strings

**File**: `app4_unified_dashboard/app.py`

**Lines 512-520** (Session Information):
```python
info_data = {
    'Active Conversation': 'Yes' if status.get('has_active_conversation') else 'No',
    'Current Conversation ID': str(status.get('conversation_id', 'N/A')),
    'Total Conversations': str(status.get('total_conversations', 0)),
    'Completed Conversations': str(status.get('completed_conversations', 0))
}

info_df = pd.DataFrame(list(info_data.items()), columns=['Setting', 'Value'])
st.dataframe(info_df, hide_index=True, width='stretch')
```

**Lines 560-563** (Algorithm Parameters):
```python
params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
# Convert values to strings to avoid Arrow serialization errors
params_df['Value'] = params_df['Value'].astype(str)
st.dataframe(params_df, hide_index=True, width='stretch')
```

**Line 448** (Breakdown):
```python
st.dataframe(breakdown_df, width='stretch', hide_index=True)
```

### Fix #2: Update Streamlit API (Deprecation Warning)

**Changed**:
```python
use_container_width=True  # Deprecated
```

**To**:
```python
width='stretch'  # New API
```

**Applied to**:
- `app4_unified_dashboard/app.py` (3 locations)
- `app4_unified_dashboard/ui/chat_view.py` (1 location)

---

## Files Modified

1. **[app4_unified_dashboard/app.py](../app4_unified_dashboard/app.py)**
   - Line 520: Fixed info_df serialization
   - Line 562: Fixed params_df serialization
   - Line 448: Updated to new Streamlit API

2. **[app4_unified_dashboard/ui/chat_view.py](../app4_unified_dashboard/ui/chat_view.py)**
   - Line 312: Fixed stats_df serialization
   - Updated to new Streamlit API

---

## Testing

App 4 should now start without errors:

```bash
cd /home/aya/work/optica_labs/algorithm_work/deployment
./start_app4_testing.sh
```

**Expected output** (terminal):
```
INFO:utils.session_state:Initializing session state...
INFO:pca_pipeline:✓ Loaded scaler from: .../embedding_scaler.pkl
INFO:pca_pipeline:✓ Loaded PCA model from: .../pca_model.pkl
INFO:__main__:PCA transformer initialized
INFO:embeddings:Bedrock client initialized for region: us-east-1
INFO:__main__:Pipeline orchestrator initialized

  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8504
```

**No errors** about Arrow serialization or `use_container_width`.

---

## Remaining Warning (Non-Critical)

You may still see:
```
InconsistentVersionWarning: Trying to unpickle estimator StandardScaler
from version 1.7.1 when using version 1.7.2
```

**What it means**: PCA models were trained with scikit-learn 1.7.1, but you have 1.7.2 installed.

**Is it a problem?**: No, minor version differences (1.7.1 → 1.7.2) are usually compatible.

**How to fix** (optional, if you want to remove the warning):
```bash
# Retrain PCA models with current scikit-learn version
cd /home/aya/work/optica_labs/algorithm_work
source venv/bin/activate
python src/pca_trainer.py

# Copy new models to deployment
cp models/*.pkl deployment/models/
```

---

## Summary

**Problem**: Mixed data types in DataFrame "Value" column broke PyArrow serialization
**Solution**: Convert all values to strings using `str()` or `.astype(str)`
**Result**: App 4 starts without serialization errors

**Bonus**: Updated deprecated `use_container_width` to new `width` parameter

---

**Created**: December 3, 2025
**Bug Type**: DataFrame serialization
**Apps Affected**: App 4
**Files Modified**: `app4_unified_dashboard/app.py`, `app4_unified_dashboard/ui/chat_view.py`
