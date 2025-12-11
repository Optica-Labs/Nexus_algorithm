#!/usr/bin/env python3
"""
Test if all imports work correctly before running Streamlit
"""

import sys
from pathlib import Path

# Setup paths (same as app.py)
current_dir = Path(__file__).parent
project_root = current_dir.parent
deployment_root = project_root / 'deployment'
app4_root = deployment_root / 'app4_unified_dashboard'
shared_root = deployment_root / 'shared'

print("=" * 60)
print("Testing Desktop App2 Imports")
print("=" * 60)
print()

print(f"Current dir: {current_dir}")
print(f"Project root: {project_root}")
print(f"Deployment root: {deployment_root}")
print(f"App4 root: {app4_root}")
print(f"Shared root: {shared_root}")
print()

# Check paths exist
print("Checking paths...")
print(f"  App4 exists: {app4_root.exists()}")
print(f"  Shared exists: {shared_root.exists()}")
print()

# Add core paths from other apps
app1_core = deployment_root / 'app1_guardrail_erosion' / 'core'
app2_core = deployment_root / 'app2_rho_calculator' / 'core'
app3_core = deployment_root / 'app3_phi_evaluator' / 'core'

# Add to path
for p in [str(deployment_root), str(shared_root), str(app4_root),
          str(app1_core), str(app2_core), str(app3_core)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Test imports
print("Testing imports...")
print()

# Test 1: Shared modules
try:
    from pca_pipeline import PCATransformer
    print("✅ PCATransformer")
except Exception as e:
    print(f"❌ PCATransformer: {e}")

try:
    from vector_processor import VectorPrecognitionProcessor
    print("✅ VectorPrecognitionProcessor")
except Exception as e:
    print(f"❌ VectorPrecognitionProcessor: {e}")

try:
    from robustness_calculator import RobustnessCalculator
    print("✅ RobustnessCalculator")
except Exception as e:
    print(f"❌ RobustnessCalculator: {e}")

try:
    from fragility_calculator import FragilityCalculator
    print("✅ FragilityCalculator")
except Exception as e:
    print(f"❌ FragilityCalculator: {e}")

try:
    from visualizations import GuardrailVisualizer, RHOVisualizer, PHIVisualizer
    print("✅ Visualizations")
except Exception as e:
    print(f"❌ Visualizations: {e}")

print()

# Test 2: App4 modules with importlib
import importlib.util

try:
    spec = importlib.util.spec_from_file_location("session_state", str(app4_root / 'utils' / 'session_state.py'))
    session_state_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(session_state_module)
    SessionState = session_state_module.SessionState
    print("✅ SessionState")
except Exception as e:
    print(f"❌ SessionState: {e}")

try:
    spec = importlib.util.spec_from_file_location("sidebar", str(app4_root / 'ui' / 'sidebar.py'))
    sidebar_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sidebar_module)
    create_sidebar = sidebar_module.create_sidebar
    print("✅ create_sidebar")
except Exception as e:
    print(f"❌ create_sidebar: {e}")

try:
    spec = importlib.util.spec_from_file_location("chat_view", str(app4_root / 'ui' / 'chat_view.py'))
    chat_view_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chat_view_module)
    create_chat_view = chat_view_module.create_chat_view
    print("✅ create_chat_view")
except Exception as e:
    print(f"❌ create_chat_view: {e}")

try:
    spec = importlib.util.spec_from_file_location("pipeline_orchestrator", str(app4_root / 'core' / 'pipeline_orchestrator.py'))
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    PipelineOrchestrator = pipeline_module.PipelineOrchestrator
    print("✅ PipelineOrchestrator")
except Exception as e:
    print(f"❌ PipelineOrchestrator: {e}")

print()

# Test 3: ChatGPT client
sys.path.insert(0, str(current_dir / 'python-backend'))
try:
    from chatgpt_integration import ChatGPTClient, check_api_key
    print("✅ ChatGPTClient")
except Exception as e:
    print(f"❌ ChatGPTClient: {e}")

print()
print("=" * 60)
print("Import test complete!")
print("=" * 60)
