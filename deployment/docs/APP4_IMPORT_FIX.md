# App 4: Import Error Fix

**Date**: December 3, 2025
**Error**: `ImportError: attempted relative import with no known parent package`
**Status**: ✅ Fixed

---

## The Problem

When starting App 4, you got:

```
ImportError: attempted relative import with no known parent package

File "/home/aya/work/optica_labs/algorithm_work/deployment/app4_unified_dashboard/app.py", line 53
    from pca_pipeline import PCATransformer
File "/home/aya/work/optica_labs/algorithm_work/deployment/shared/pca_pipeline.py", line 14
    from .embeddings import EmbeddingGenerator
```

### Root Cause

**shared/pca_pipeline.py** used a **relative import** (`from .embeddings`):
- This works when `shared` is imported as a package: `from shared.pca_pipeline import ...`
- This FAILS when `shared` is in sys.path and imported directly: `from pca_pipeline import ...`

App 4 adds `shared/` to sys.path and imports directly, so the relative import failed.

---

## The Solution

### Fix #1: Make pca_pipeline.py Support Both Import Styles

**File**: `shared/pca_pipeline.py:15-19`

**Before**:
```python
from .embeddings import EmbeddingGenerator
```

**After**:
```python
# Handle both relative and absolute imports
try:
    from .embeddings import EmbeddingGenerator
except ImportError:
    from embeddings import EmbeddingGenerator
```

**How it works**:
1. First tries relative import (`from .embeddings`) - works when imported as package
2. Falls back to absolute import (`from embeddings`) - works when shared/ is in sys.path
3. Now compatible with both App 1-3 (package-style) and App 4 (direct import)

### Fix #2: Improve App 4's Path Management

**File**: `app4_unified_dashboard/app.py:32-52`

**Before**:
```python
sys.path.insert(0, os.path.join(deployment_root, 'shared'))
sys.path.insert(0, os.path.join(deployment_root, 'app1_guardrail_erosion', 'core'))
# ... more inserts
```

**After**:
```python
# Add deployment root to path
deployment_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if deployment_root not in sys.path:
    sys.path.insert(0, deployment_root)

# Add shared and app-specific paths
shared_path = os.path.join(deployment_root, 'shared')
if shared_path not in sys.path:
    sys.path.insert(0, shared_path)

app1_core = os.path.join(deployment_root, 'app1_guardrail_erosion', 'core')
if app1_core not in sys.path:
    sys.path.insert(0, app1_core)

# ... same pattern for app2_core, app3_core
```

**Benefits**:
- Prevents duplicate path entries (checks `if not in sys.path`)
- More explicit and readable
- Consistent with how other apps handle paths

---

## Testing

**Verify imports work**:
```bash
cd /home/aya/work/optica_labs/algorithm_work/deployment

python3 -c "
import sys, os
deployment_root = os.getcwd()
sys.path.insert(0, os.path.join(deployment_root, 'shared'))
sys.path.insert(0, os.path.join(deployment_root, 'app1_guardrail_erosion', 'core'))
sys.path.insert(0, os.path.join(deployment_root, 'app2_rho_calculator', 'core'))
sys.path.insert(0, os.path.join(deployment_root, 'app3_phi_evaluator', 'core'))

from pca_pipeline import PCATransformer
from vector_processor import VectorPrecognitionProcessor
from robustness_calculator import RobustnessCalculator
from fragility_calculator import FragilityCalculator
print('✓ All imports successful!')
"
```

**Expected output**:
```
✓ All imports successful!
```

---

## How to Launch App 4 Now

**Option 1: Use the launch script**:
```bash
cd /home/aya/work/optica_labs/algorithm_work/deployment
./start_app4_testing.sh
```

**Option 2: Manual launch**:
```bash
cd /home/aya/work/optica_labs/algorithm_work/deployment
source venv/bin/activate
cd app4_unified_dashboard
streamlit run app.py --server.port 8504
```

App 4 should now start without import errors!

---

## Related Import Patterns

### App 1-3: Package-Style Imports
```python
# Add deployment root
sys.path.insert(0, str(deployment_root))

# Import as package
from shared.pca_pipeline import PCATransformer
from shared.config import DEFAULT_VSAFE_TEXT
```

### App 4: Direct Imports
```python
# Add shared/ to path
sys.path.insert(0, os.path.join(deployment_root, 'shared'))

# Import directly
from pca_pipeline import PCATransformer
from config import DEFAULT_VSAFE_TEXT
```

Both now work thanks to the try/except pattern in shared modules!

---

## Files Modified

1. **[shared/pca_pipeline.py:15-19](../shared/pca_pipeline.py#L15-L19)**
   - Added try/except for relative/absolute import compatibility

2. **[app4_unified_dashboard/app.py:32-52](../app4_unified_dashboard/app.py#L32-L52)**
   - Improved path management with duplicate checks
   - More explicit path additions

---

## Summary

**Problem**: Relative import in shared module broke when imported directly by App 4
**Solution**: Try relative import first, fall back to absolute import
**Result**: App 4 can now start without import errors

---

**Created**: December 3, 2025
**Bug Type**: Import compatibility
**Apps Affected**: App 4
**Files Modified**: `shared/pca_pipeline.py`, `app4_unified_dashboard/app.py`
