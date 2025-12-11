# Metrics Calculation Fix - Vector Precognition

## 🐛 **Problem Identified**

### **Symptoms**
```
Turn 1: "hello, how are you?"
User:
  - Risk (R): 0.137 ✅ (normal)
  - Velocity (v): 2.409 ❌ (should be ~0 for first user message)
  - Breach (L): 0.528 (too high)

Assistant:
  - Risk (R): 0.209 ✅ (normal)
  - Velocity (v): 0.000 ❌ (should not be zero)
  - Breach (L): 0.101 ✅ (normal)

ρ (rho): 1.527 ❌ (FRAGILE - should be ROBUST for "hello")
```

### **Root Cause**
The bug was in [`chatgpt_client.py`](python-backend/chatgpt_client.py) line 179-183:

```python
# WRONG (old code):
self.vector_history.append(user_vector)         # Add vectors FIRST
self.vector_history.append(assistant_vector)
user_metrics = self._calculate_point_metrics(user_vector, ...)   # Then calculate
assistant_metrics = self._calculate_point_metrics(assistant_vector, ...)
```

**What went wrong:**
1. Vectors added to history BEFORE calculating metrics
2. When calculating velocity: `v = norm(current_vec - vector_history[-1])`
3. For user: `vector_history[-1]` was the user vector itself! ❌
4. For assistant: `vector_history[-1]` was assistant vector itself! ❌
5. Result: Incorrect velocity calculations → Wrong L values → Wrong ρ

---

## ✅ **Solution**

### **Fixed Code** ([`chatgpt_client.py`](python-backend/chatgpt_client.py))

```python
# CORRECT (new code):
# Calculate BEFORE adding to history
user_metrics = self._calculate_point_metrics(user_vector, is_user=True, prev_velocity=None)
self.vector_history.append(user_vector)    # Add AFTER calculating

# For assistant, pass user's velocity for acceleration
assistant_metrics = self._calculate_point_metrics(assistant_vector, is_user=False, prev_velocity=user_metrics['v'])
self.vector_history.append(assistant_vector)  # Add AFTER calculating
```

### **Changes Made**

1. **Reordered operations** (lines 181-192):
   - Calculate user metrics BEFORE adding user vector
   - Add user vector to history
   - Calculate assistant metrics BEFORE adding assistant vector
   - Add assistant vector to history

2. **Fixed acceleration calculation** (lines 221-253):
   - Added `prev_velocity` parameter to `_calculate_point_metrics()`
   - Assistant's acceleration uses user's velocity from same turn
   - User's acceleration uses previous assistant's velocity

3. **Improved velocity logic** (lines 236-241):
   - `v = 0` for first message (no previous vector)
   - `v = distance(current, previous)` for subsequent messages
   - Previous vector is always the LAST one in history (not current)

---

## 📊 **Expected Behavior After Fix**

### **Turn 1: "hello, how are you?"**
```
User (first message):
  - R: ~0.05-0.15 (distance from VSAFE)
  - v: 0.000 (no previous vector)
  - a: 0.000 (no previous velocity)
  - L: ~0.10-0.30 (low risk)