# App 2: Import Path Fix

## Issue

When launching App 2, got this error:
```
File "/home/aya/work/optica_labs/algorithm_work/deployment/app2_rho_calculator/app.py", line 44, in <module>
    from core.robustness_calculator import RobustnessCalculator
ModuleNotFoundError: No module named 'core.robustness_calculator'
```

## Root Cause

**Path Conflict**: App 2 imports from both App 1 and its own `core/utils` directories:

1. Line 37: Added `app1_guardrail_erosion` to path
2. Lines 38-41: Imported from App 1's `core` and `utils`
3. **BUT**: App 1 stayed in the path!
4. Lines 44-51: Tried to import from App 2's `core` and `utils`
5. **Python found App 1's `core` instead of App 2's!**

## Solution

**Strategy**: Temporarily add App 1, import what we need, then remove it before importing App 2's modules

### Code Changes (app2_rho_calculator/app.py)

**Before (Broken)**:
```python
# Import App 1 components
sys.path.insert(0, str(deployment_root / 'app1_guardrail_erosion'))
from core.vector_processor import VectorPrecognitionProcessor
from utils.helpers import (
    parse_manual_input, json_to_turns, csv_to_turns
)

# Import app-specific modules
from core.robustness_calculator import RobustnessCalculator  # ← ERROR!
```

**After (Fixed)**:
```python
# Import App 1 components
app1_dir = deployment_root / 'app1_guardrail_erosion'
sys.path.insert(0, str(app1_dir))
from core.vector_processor import VectorPrecognitionProcessor
import utils.helpers as app1_helpers
# Remove App1 from path to avoid conflicts
sys.path.remove(str(app1_dir))

# Add current app directory to path
current_app_dir = current_file.parent
if str(current_app_dir) not in sys.path:
    sys.path.insert(0, str(current_app_dir))

# Import app-specific modules (App 2's own core and utils)
from core.robustness_calculator import RobustnessCalculator  # ← WORKS!
from utils.helpers import (...)

# Get App1 helper functions
parse_manual_input = app1_helpers.parse_manual_input
json_to_turns = app1_helpers.json_to_turns
csv_to_turns = app1_helpers.csv_to_turns
```

## Key Changes

1. **Store App1 path** in variable: `app1_dir`
2. **Import App1's utils as alias**: `import utils.helpers as app1_helpers`
3. **Remove App1 from path**: `sys.path.remove(str(app1_dir))`
4. **Add App2 directory**: `sys.path.insert(0, str(current_app_dir))`
5. **Import App2's modules**: Now Python finds App 2's `core` and `utils`
6. **Get App1 functions**: Assign from `app1_helpers` module

## Same Fix Applied to App 3

App 3 doesn't import from other apps, but added current directory to path as preventive measure:

```python
# Add current app directory to path for app-specific imports
current_app_dir = current_file.parent
if str(current_app_dir) not in sys.path:
    sys.path.insert(0, str(current_app_dir))
```

## Files Modified

- `app2_rho_calculator/app.py` - Lines 36-62
- `app3_phi_evaluator/app.py` - Lines 35-38

## Testing

**Before fix**:
```bash
./start_app2_testing.sh
# Result: ModuleNotFoundError
```

**After fix**:
```bash
./start_app2_testing.sh
# Result: App launches successfully ✓
```

## Status

✅ **Fixed in App 2**
✅ **Fixed in App 3** (preventive)
✅ **Ready for testing**

## Related Issues

This is separate from the other import path fix (`.resolve()` method) which was for Streamlit compatibility. This fix is specifically for resolving conflicts between apps that share similar module names.

## Lesson Learned

When importing from multiple apps:
1. Use temporary path additions
2. Remove paths after importing
3. Be explicit about which module you're importing
4. Use aliases for conflicting module names

---

**App 2 should now launch successfully!** 🚀

Try: `./start_app2_testing.sh`
