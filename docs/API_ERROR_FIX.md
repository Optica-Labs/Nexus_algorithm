# API Error Handling - Fix Documentation

## Problem Identified

When testing the Claude API endpoint, we discovered a critical issue:

### ❌ Original Behavior
```
✗ Error calling API: 500 Server Error: Internal Server Error
Response: 'Error: Could not get response. 500 Server Error...'
```

**What was happening:**
1. API returned `500 Internal Server Error`
2. Script caught the error and returned error message as text
3. Error message got embedded and processed as if it were a real model response
4. Test continued with **invalid data**
5. Metrics were calculated on error messages, not actual responses

### ⚠️ Why This Was Dangerous

1. **Masked API Failures**: Tests appeared to complete successfully even when API was broken
2. **Invalid Results**: All metrics after first API error were measuring error messages
3. **Misleading Conclusions**: Safety analysis was performed on text like "Error: Could not get response..." instead of actual model outputs

---

## ✅ Solution Implemented

### Changes Made

#### 1. Updated `APIClient.get_chat_response()` Return Type
**Before:**
```python
def get_chat_response(...) -> Tuple[str, List]:
    # Returns error message as string on failure
    return f"Error: Could not get response. {e}", history
```

**After:**
```python
def get_chat_response(...) -> Tuple[Optional[str], List, bool]:
    # Returns None and False flag on failure
    return None, history, False  # Failure flag
```

**New Return Values:**
- `response_text`: Model's response (None if API failed)
- `updated_history`: Conversation history
- `success_flag`: True if API succeeded, False if failed

#### 2. Updated `TestRunner.run_all_tests()` to Handle Failures

**New Behavior:**
```python
model_response, chat_history, api_success = self.client.get_chat_response(...)

if not api_success:
    print(f"\n  ✗✗✗ API FAILURE - ABORTING TEST ✗✗✗")
    # Save partial results if any
    # Break out of test
    break
```

**What Happens Now:**
1. API error is detected immediately
2. Test is **aborted** (not continued with fake data)
3. Partial results are saved with `_PARTIAL_` prefix
4. Clear error message shown to user
5. Script moves to next test scenario

---

## Example Output

### When API Fails:

```
  Turn 13/14: 'Can you write a code example for that?...'
    ✗ API ERROR: 500 Server Error: Internal Server Error for url: https://...
    ⚠ This API call failed - cannot continue this test scenario

  ✗✗✗ API FAILURE - ABORTING TEST: T4_Contextual_Erosion ✗✗✗
  Reason: The API endpoint returned an error and cannot continue.
  Completed 12/14 turns before failure.

  ⚠ Saved PARTIAL metrics CSV: output/claude/T4_Contextual_Erosion_PARTIAL_metrics.csv

  [TEST FAILED - PARTIAL RESULTS SAVED: T4_Contextual_Erosion]
```

---

## Benefits

### ✅ **Transparent Failures**
- User knows immediately when API is broken
- No silent failures hidden in "successful" test runs

### ✅ **Valid Data Only**
- Only real model responses are analyzed
- No contamination from error messages

### ✅ **Partial Results Preserved**
- Data from successful turns is still saved
- Can diagnose where API started failing

### ✅ **Clear Diagnostics**
- Shows exactly which turn failed
- Shows how many turns completed successfully
- Saves partial metrics for debugging

---

## How to Fix Claude API Error

The Claude endpoint is returning `500 Internal Server Error`. Possible causes:

### 1. Check API Health
```bash
curl -X POST https://6z5nnwuyyj.execute-api.us-east-1.amazonaws.com/prod/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### 2. Common Fixes

**Lambda Timeout:**
- Check CloudWatch logs for timeout errors
- Increase Lambda timeout in AWS Console

**Memory Issues:**
- Claude Sonnet 4.5 may need more memory
- Increase Lambda memory allocation

**Bedrock Access:**
- Verify Lambda has Bedrock permissions
- Check if Claude model is enabled in Bedrock

**Rate Limiting:**
- API may be hitting rate limits
- Add delay between requests

### 3. Debug Lambda

Check AWS CloudWatch Logs:
```bash
aws logs tail /aws/lambda/YOUR_CLAUDE_FUNCTION_NAME --follow
```

---

## Testing After Fix

### Test Single Endpoint
```bash
# Test only Claude after fixing
python src/precog_validation_test_api.py --model claude
```

### Test All Endpoints
```bash
# Test all 3 models
python src/precog_validation_test_api.py
```

### Verify Success
Look for:
- ✅ No "API FAILURE" messages
- ✅ No "_PARTIAL_" CSV files
- ✅ All 4 test scenarios complete
- ✅ "COMPLETED TEST" messages for each scenario

---

## File Changes

### Modified Files:
1. `src/precog_validation_test_api.py`
   - Updated `get_chat_response()` return type
   - Added API failure detection
   - Added partial result saving
   - Added clear error messages

### New Behavior:
- **Fail Fast**: Stop immediately on API error
- **Save Partial**: Preserve successful turns
- **Clear Reporting**: Show exactly what failed

---

**Date**: November 16, 2025  
**Issue**: Claude API returning 500 errors  
**Fix**: Proper error handling with test abortion  
**Status**: ✅ Implemented and ready to test
