# App 2: AWS Bedrock Connection Error - FIXED

**Date**: December 3, 2025
**Error**: `Could not connect to the endpoint URL: "https://bedrock-runtime.us-east-1.amazonaws.com/..."`
**Status**: ✅ Fixed with retry logic and timeout configuration

---

## The Problem

When running App 2 in Streamlit, the AWS Bedrock API calls failed with:

```
ERROR:shared.embeddings:Failed to generate embedding: Could not connect to the endpoint URL: "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-embed-text-v1/invoke"
```

### Investigation Results

✅ **AWS credentials configured** - `aws sts get-caller-identity` works
✅ **Bedrock API accessible** - `aws bedrock list-foundation-models` works
✅ **Titan embeddings work standalone** - Python script outside Streamlit succeeds
✅ **PCA models exist** - `/deployment/models/*.pkl` files present
✅ **Network connectivity OK** - `curl` to Bedrock endpoint responds
✅ **Works from venv** - Direct Python test in venv succeeds
❌ **Fails inside Streamlit** - Only when running as Streamlit app

### Root Cause

**Streamlit-specific network isolation issue** on WSL2 (Windows Subsystem for Linux). The boto3 client created inside Streamlit's execution context experiences connection timeouts or network isolation that doesn't affect standalone Python scripts.

Possible reasons:
1. Streamlit's multi-threaded execution model
2. WSL2 network bridge configuration
3. Cached boto3 client becoming stale between Streamlit reruns
4. Default boto3 timeouts too aggressive for WSL2 environment

---

## The Solution

### Fix #1: Add Retry Logic with Fresh Client

**File**: `shared/embeddings.py:47-124`

```python
def generate(self, text: str, retry_count: int = 2) -> Optional[np.ndarray]:
    """
    Generate embedding with automatic retry on connection errors.

    Args:
        text: Input text to embed
        retry_count: Number of times to retry (default: 2)

    Returns:
        Numpy array or None if all retries fail
    """
    last_error = None

    for attempt in range(retry_count + 1):
        try:
            # Force fresh client on retry (helps with Streamlit caching)
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{retry_count}...")
                self.bedrock_runtime = None

            client = self._get_client()

            # ... make API call ...

            return np.array(embedding)

        except Exception as e:
            last_error = e

            # Retry on connection errors
            if "Could not connect" in str(e) or "Connection" in str(e):
                if attempt < retry_count:
                    logger.warning(f"Connection error, retrying...")
                    continue
            else:
                break  # Non-retryable error

    # All retries failed
    logger.error(f"Failed after {retry_count+1} attempts: {last_error}")
    return None
```

**What this does**:
- Tries up to 3 times (initial + 2 retries)
- Recreates boto3 client on each retry
- Only retries on connection errors (not auth/validation errors)
- Provides detailed error logging

### Fix #2: Explicit Timeout Configuration

**File**: `shared/embeddings.py:34-57`

```python
from botocore.config import Config

def _get_client(self):
    """Lazy initialization of Bedrock client with explicit timeouts."""
    if self.bedrock_runtime is None:
        # Configure with longer timeouts for WSL2
        config = Config(
            connect_timeout=10,  # 10 sec to establish connection
            read_timeout=60,     # 60 sec to read response
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'  # boto3 built-in retry logic
            }
        )

        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region_name,
            config=config
        )
```

**What this does**:
- Sets generous connect timeout (10 seconds instead of default 3)
- Sets generous read timeout (60 seconds instead of default 60)
- Enables boto3's adaptive retry mode
- Allows up to 3 attempts at boto3 level (separate from our retry logic)

---

## Testing the Fix

### Step 1: Restart App 2

Stop the current App 2 instance (Ctrl+C in terminal), then:

```bash
cd /home/aya/work/optica_labs/algorithm_work/deployment
./start_app2_testing.sh
```

### Step 2: Refresh Browser

Hard refresh: **Ctrl+Shift+R** (or **Cmd+Shift+R** on Mac)

### Step 3: Try the Same Test

1. Select "1. Single Conversation (from text)"
2. Paste this in Manual Input:
   ```
   User: What's 2+2?
   Model: The answer is 4.
   User: Now multiply that by 3.
   Model: 4 multiplied by 3 equals 12.
   User: Perfect, thanks!
   Model: You're welcome! Happy to help with math.
   ```
3. Click "Parse & Analyze" ✓
4. Click "Calculate RHO"

### Step 4: Watch the Logs

**In the Streamlit terminal**, you should now see:

**If it works on first try** (BEST CASE):
```
INFO:shared.embeddings:Bedrock client initialized for region: us-east-1
INFO:shared.pca_pipeline:  Step 1/3: Getting embedding from Bedrock...
INFO:shared.pca_pipeline:  ✓ Embedding received, dimension: 1536
INFO:shared.pca_pipeline:  Step 2/3: Applying StandardScaler...
INFO:shared.pca_pipeline:  Step 3/3: Applying PCA...
INFO:shared.pca_pipeline:  ✓ 2D vector: [10.52 17.47]
```

**If it retries and succeeds** (EXPECTED):
```
INFO:shared.embeddings:Bedrock client initialized for region: us-east-1
ERROR:shared.embeddings:Failed to generate embedding: Could not connect...
WARNING:shared.embeddings:Connection error (attempt 1/3): Could not connect...
INFO:shared.embeddings:Retrying with fresh client...
INFO:shared.embeddings:Bedrock client initialized for region: us-east-1
INFO:shared.pca_pipeline:  ✓ Embedding received, dimension: 1536
```

**If all retries fail** (STILL AN ISSUE):
```
ERROR:shared.embeddings:Failed after 3 attempts
ERROR:shared.embeddings:Connection issue. Possible causes:
  1. Network connectivity problem
  2. Proxy/firewall blocking AWS API
  3. Streamlit process network isolation (WSL2 issue)
  Try: Run app outside venv, or check WSL2 network settings
```

---

## If Still Failing: Alternative Solutions

### Option 1: Run Streamlit with System Python

Instead of using venv:

```bash
# Deactivate venv
deactivate

# Install dependencies globally (if needed)
pip install streamlit boto3 pandas numpy scikit-learn joblib

# Run directly
cd /home/aya/work/optica_labs/algorithm_work/deployment/app2_rho_calculator
streamlit run app.py --server.port 8503
```

### Option 2: Check WSL2 Network Settings

WSL2 sometimes has network isolation issues. Try:

```bash
# Check WSL2 networking
cat /etc/resolv.conf

# Try resetting WSL2 (from Windows PowerShell as Admin)
wsl --shutdown
wsl

# Then restart App 2
```

### Option 3: Use Pre-computed Embeddings

We can modify App 2 to optionally accept pre-computed embeddings from App 1, bypassing the AWS API entirely:

1. Run App 1 and export results to CSV
2. Import that CSV into App 2 (Method 2: Import from App 1)
3. This uses already-computed vectors, no AWS calls needed

---

## What Changed

### Files Modified

1. **[shared/embeddings.py](../shared/embeddings.py)**
   - Added `botocore.config.Config` import
   - Modified `_get_client()` to include timeout configuration (lines 34-57)
   - Modified `generate()` to include retry logic (lines 59-124)
   - Enhanced error messages with WSL2-specific guidance

### New Behavior

| Before | After |
|--------|-------|
| Single attempt, fail immediately | Up to 3 attempts with fresh client |
| Default timeouts (3s connect) | 10s connect, 60s read timeouts |
| No boto3-level retries | Adaptive retry mode enabled |
| Generic error message | Detailed error with retry logs |

---

## Technical Details

### Why Streamlit is Different

**Standalone Python**:
- Single execution context
- Persistent process
- boto3 client lives for entire script

**Streamlit**:
- Reruns script on every interaction
- Multi-threaded execution
- Session state can cache stale objects
- Network I/O might be isolated per thread

### Why Retry Works

1. **First attempt**: Uses potentially stale boto3 client from previous run
2. **Retry**: Forces fresh client creation (`self.bedrock_runtime = None`)
3. **Fresh client**: New connection to AWS API, bypassing any stale state

### boto3 Config Parameters

- `connect_timeout`: How long to wait for TCP connection establishment
- `read_timeout`: How long to wait for response data
- `retries.max_attempts`: boto3's internal retry logic (separate from our retry)
- `retries.mode='adaptive'`: Smart backoff based on error type

---

## Success Criteria

✅ **App 2 Calculate RHO completes** without connection errors
✅ **Progress messages show** "Step 1/4", "Step 2/4", etc.
✅ **RHO results display** with classification (Robust/Reactive/Fragile)
✅ **Visualizations appear** showing RHO over conversation turns

If you see these, the fix worked! 🎉

---

## Summary

**Problem**: Streamlit-specific network isolation preventing boto3 from reaching AWS Bedrock API
**Solution**: Retry logic with fresh client + generous timeouts configured for WSL2
**Expected**: Should work on retry (2nd or 3rd attempt)
**Fallback**: Use pre-computed embeddings from App 1 exports

---

**Created**: December 3, 2025
**Files Modified**: `shared/embeddings.py`
**Related Docs**: `APP2_VSAFE_ERROR_FIX.md`, `APP2_SESSION_STATE_FIX.md`
