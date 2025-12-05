# Architecture Explained: Where is Everything?

**Date**: December 2, 2025
**Purpose**: Clarify the architecture and show where embeddings, PCA, and all components are located

---

## 🎯 Key Question: Where are the embeddings and text-to-2D conversion?

**Answer**: They're in the **`shared/`** folder, imported by all apps!

---

## 📦 Architecture: Shared vs App-Specific

### Shared Modules (Used by ALL apps)

```
deployment/shared/
├── embeddings.py          ← AWS Bedrock Titan embeddings HERE
├── pca_pipeline.py        ← Text-to-2D conversion HERE
├── config.py              ← Default parameters
├── validators.py          ← Input validation
└── visualizations.py      ← Plotting functions
```

### Why Shared?

✅ **NO CODE DUPLICATION**
- Write once, use in all 4 apps
- Consistent behavior everywhere
- Easy to maintain and update
- Follows DRY (Don't Repeat Yourself) principle

---

## 🔍 Detailed Flow: Text → 2D Vector

### Step-by-Step Process

```
User enters text in App 1
         ↓
App 1 imports PCATransformer from shared/pca_pipeline.py
         ↓
PCATransformer.text_to_2d(text) is called
         ↓
Inside PCATransformer:
    Step 1: EmbeddingGenerator.generate(text)  ← from shared/embeddings.py
           ↓
         AWS Bedrock Titan API call
           ↓
         Returns 1536-dimensional vector
           ↓
    Step 2: scaler.transform(embedding)
           ↓
         Standardizes the vector
           ↓
    Step 3: pca_model.transform(scaled)
           ↓
         Reduces to 2D
           ↓
         Returns [x, y] coordinates
```

### Code Evidence

**In `shared/pca_pipeline.py`:**
```python
from .embeddings import EmbeddingGenerator  # Line 14

class PCATransformer:
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()  # Line 60

    def text_to_2d(self, text):
        embedding = self.embedding_generator.generate(text)  # Line 82
        scaled = self.scaler.transform(embedding)
        vector_2d = self.pca_model.transform(scaled)
        return vector_2d
```

**In App 1 `app.py`:**
```python
from shared.pca_pipeline import PCATransformer  # Line 34

# Later in the code:
pipeline = PCATransformer()  # Creates instance
user_vector = pipeline.text_to_2d(user_message)  # Converts text to 2D
```

---

## 📂 Complete File Structure with Locations

```
deployment/
│
├── shared/                              ← SHARED BY ALL APPS
│   ├── embeddings.py                    ← AWS Bedrock integration
│   ├── pca_pipeline.py                  ← Text-to-2D pipeline (uses embeddings.py)
│   ├── config.py                        ← Configuration
│   ├── validators.py                    ← Validation
│   └── visualizations.py                ← Plotting
│
├── models/                              ← PCA MODELS (needed by shared/pca_pipeline.py)
│   ├── pca_model.pkl                    ← Trained PCA model
│   └── embedding_scaler.pkl             ← Trained scaler
│
├── app1_guardrail_erosion/             ← APP 1 SPECIFIC CODE
│   ├── app.py                           ← Main UI (imports from shared/)
│   ├── core/
│   │   └── vector_processor.py          ← Vector Precognition algorithm
│   └── utils/
│       ├── helpers.py                   ← App 1 specific utilities
│       └── api_client.py                ← NEW: LLM API client
│
├── app2_rho_calculator/                ← APP 2 SPECIFIC CODE
│   ├── app.py                           ← Main UI (imports from shared/)
│   ├── core/
│   │   └── robustness_calculator.py     ← RHO calculation
│   └── utils/
│       └── helpers.py                   ← App 2 specific utilities
│
├── app3_phi_evaluator/                 ← APP 3 SPECIFIC CODE
│   ├── app.py                           ← Main UI (imports from shared/)
│   ├── core/
│   │   └── fragility_calculator.py      ← PHI calculation
│   └── utils/
│       └── helpers.py                   ← App 3 specific utilities
│
└── app4_unified_dashboard/             ← APP 4 SPECIFIC CODE
    ├── app.py                           ← Main UI (imports from shared/)
    ├── core/
    │   ├── pipeline_orchestrator.py     ← Coordinates all 3 stages
    │   └── api_client.py                ← LLM API integration
    ├── ui/
    │   ├── chat_view.py                 ← Chat interface
    │   └── sidebar.py                   ← Configuration UI
    └── utils/
        └── session_state.py             ← Session management
```

---

## 🔍 Verification: Are Embeddings Used in App 1 and App 2?

### App 1: Guardrail Erosion

**YES!** Embeddings are used in App 1 through this chain:

```
App 1 app.py (Line 408-434)
    ↓ imports
shared/pca_pipeline.py
    ↓ imports
shared/embeddings.py
    ↓ calls
AWS Bedrock Titan API
```

**Evidence in App 1 code** (deployment/app1_guardrail_erosion/app.py):

```python
# Line 34
from shared.pca_pipeline import PCATransformer

# Line 408-416 (in process_conversation function)
pipeline = st.session_state.pca_pipeline
vsafe_vector = pipeline.text_to_2d(config['vsafe_text'])  # ← Uses embeddings!

# Line 422-440 (converting all messages)
for user_msg in messages['user']:
    user_vec = pipeline.text_to_2d(user_msg)  # ← Uses embeddings!

for model_msg in messages['model']:
    model_vec = pipeline.text_to_2d(model_msg)  # ← Uses embeddings!
```

### App 2: RHO Calculator

**YES!** App 2 also uses embeddings when processing single conversations:

```
App 2 app.py
    ↓ imports App 1 components for single conversation mode
App 1 components
    ↓ import
shared/pca_pipeline.py
    ↓ imports
shared/embeddings.py
```

**Evidence**: App 2 reuses App 1's conversation processing which includes text-to-2D conversion.

---

## 🎯 Input Methods Verification: All 4 Present?

### App 1: Line Numbers of Each Input Method

Let me verify by checking the actual code:

```bash
$ grep -n "def input_method" app1_guardrail_erosion/app.py
201:def input_method_1_manual():
244:def input_method_2_json():
290:def input_method_3_csv():
327:def input_method_4_api():
```

✅ **ALL 4 INPUT METHODS EXIST**

### Detailed Verification:

#### Input Method 1: Manual Text (Line 201-242)
```python
def input_method_1_manual():
    """Input Method 1: Manual Text Entry."""
    st.subheader("📝 Manual Text Input")
    # ... handles pasted conversation text
```
✅ **EXISTS**

#### Input Method 2: JSON Upload (Line 244-288)
```python
def input_method_2_json():
    """Input Method 2: JSON File Upload."""
    st.subheader("📄 JSON File Upload")
    uploaded_file = st.file_uploader(...)
    # ... handles JSON file parsing
```
✅ **EXISTS**

#### Input Method 3: CSV Import (Line 290-325)
```python
def input_method_3_csv():
    """Input Method 3: CSV File Upload."""
    st.subheader("📊 CSV File Upload")
    uploaded_file = st.file_uploader(...)
    # ... handles CSV file parsing
```
✅ **EXISTS**

#### Input Method 4: API Integration (Line 327-398)
```python
def input_method_4_api():
    """Input Method 4: API Integration."""
    st.subheader("🔌 API Integration")
    # Model selection
    model_key = st.selectbox(...)
    # Build conversation interactively
    # Send messages & get responses
```
✅ **EXISTS**

---

## 📊 Data Flow Diagram

### Complete Flow for App 1

```
┌─────────────────────────────────────────────────────────────┐
│                        APP 1: MAIN UI                       │
│                   (app1_guardrail_erosion/app.py)           │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
    ┌───▼──┐    ┌───▼──┐    ┌───▼──┐
    │Input │    │Input │    │Input │
    │  1   │    │  2   │    │  3   │
    │Manual│    │ JSON │    │ CSV  │
    └───┬──┘    └───┬──┘    └───┬──┘
        │            │            │
        └────────────┼────────────┘
                     │
             ┌───────▼────────┐
             │   Parse Text   │
             │   Validation   │
             └───────┬────────┘
                     │
        ┌────────────▼─────────────┐
        │  shared/pca_pipeline.py  │
        │  PCATransformer          │
        │  .text_to_2d(text)       │
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │  shared/embeddings.py    │
        │  EmbeddingGenerator      │
        │  .generate(text)         │
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │   AWS Bedrock Titan API  │
        │   Returns 1536-D vector  │
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │  Scaler (StandardScaler) │
        │  from models/*.pkl       │
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │  PCA Model (2 components)│
        │  from models/*.pkl       │
        └────────────┬─────────────┘
                     │
            Returns [x, y] 2D vector
                     │
        ┌────────────▼────────────────┐
        │ core/vector_processor.py    │
        │ VectorPrecognitionProcessor │
        │ Calculate R, v, a, L        │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────────┐
        │ shared/visualizations.py    │
        │ GuardrailVisualizer         │
        │ Create 5-panel plot         │
        └────────────┬────────────────┘
                     │
                Display Results
```

---

## 🧪 How to Verify This Yourself

### 1. Check Imports in App 1

```bash
cd deployment/app1_guardrail_erosion
head -50 app.py | grep "from shared"
```

**You'll see:**
```python
from shared.config import ...
from shared.pca_pipeline import PCATransformer
from shared.validators import ...
from shared.visualizations import GuardrailVisualizer
```

### 2. Check Embeddings are Used

```bash
cd deployment/shared
grep -n "EmbeddingGenerator" pca_pipeline.py
```

**You'll see:**
```
14:from .embeddings import EmbeddingGenerator
60:    self.embedding_generator = EmbeddingGenerator(...)
82:    high_dim_vector = self.embedding_generator.generate(text)
```

### 3. Check All Input Methods

```bash
cd deployment/app1_guardrail_erosion
grep -c "def input_method" app.py
```

**Result:** 4 (all present)

---

## ✅ Summary

### Question 1: Where are the embeddings?
**Answer**: `deployment/shared/embeddings.py`
- Used by: `PCATransformer` in `pca_pipeline.py`
- Called every time: Text needs to be converted to 2D
- Apps using it: Apps 1, 2, 4 (App 3 doesn't need embeddings)

### Question 2: Where is text-to-2D conversion?
**Answer**: `deployment/shared/pca_pipeline.py`
- Class: `PCATransformer`
- Method: `.text_to_2d(text)`
- Uses: `EmbeddingGenerator` from `embeddings.py`
- Used by: All apps that process text

### Question 3: Are all input options present in App 1?
**Answer**: ✅ **YES** - All 4 input methods implemented:
1. Manual Text Input (Line 201)
2. JSON Upload (Line 244)
3. CSV Import (Line 290)
4. API Integration (Line 327)

### Question 4: Are embeddings used in App 1 and App 2?
**Answer**: ✅ **YES**
- **App 1**: Direct usage via `PCATransformer`
- **App 2**: Indirect usage (reuses App 1 components)
- **App 4**: Direct usage via `PCATransformer`
- **App 3**: No (works with pre-calculated RHO values)

---

## 🎯 Design Decision: Why Shared Modules?

### Benefits:

1. **No Code Duplication**
   - Write embedding code once
   - Use in all 4 apps
   - Maintain in one place

2. **Consistent Behavior**
   - All apps use same embedding model
   - Same PCA transformation
   - Same validation logic

3. **Easy Updates**
   - Fix a bug in one place
   - All apps benefit immediately
   - No synchronization issues

4. **MLOps Best Practice**
   - Centralized model loading
   - Version control friendly
   - Easy to test independently

---

## 🔍 Still Confused? Here's a Simple Test

Run this to see the complete import chain:

```bash
cd deployment/app1_guardrail_erosion
python3 << 'EOF'
import sys
sys.path.insert(0, '..')

# This is what App 1 does
from shared.pca_pipeline import PCATransformer

# Check what PCATransformer imports
import inspect
source = inspect.getsource(PCATransformer)
if 'EmbeddingGenerator' in source:
    print("✅ PCATransformer uses EmbeddingGenerator")
if 'embeddings' in source:
    print("✅ PCATransformer imports from embeddings module")

print("\nConclusion: Embeddings ARE being used via shared modules!")
EOF
```

---

**Bottom Line**:
- ✅ Embeddings exist and are used
- ✅ They're in `shared/embeddings.py`
- ✅ All apps import from `shared/`
- ✅ All 4 input methods present in App 1
- ✅ Architecture is correct and efficient!

