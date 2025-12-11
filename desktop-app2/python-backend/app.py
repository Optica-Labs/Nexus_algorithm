#!/usr/bin/env python3
"""
Desktop App 2: App4 Unified Dashboard with ChatGPT Integration

This is a desktop version of App4 that integrates ChatGPT for live conversations
while maintaining all App4 features:
- 4-Tab Interface (Live Chat, RHO Analysis, PHI Benchmark, Settings)
- Real-time Vector Precognition analysis
- Multi-conversation tracking
- ChatGPT API integration via OpenAI

The app runs as App4 but uses ChatGPT as the LLM backend instead of AWS Lambda.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths for App4 and shared modules
# IMPORTANT: Use absolute paths to avoid issues when running from different directories
current_dir = Path(__file__).parent.absolute()  # Make sure it's absolute
project_root = current_dir.parent.parent  # algorithm_work directory
deployment_root = project_root / 'deployment'
app4_root = deployment_root / 'app4_unified_dashboard'
shared_root = deployment_root / 'shared'
app1_core = deployment_root / 'app1_guardrail_erosion' / 'core'
app2_core = deployment_root / 'app2_rho_calculator' / 'core'
app3_core = deployment_root / 'app3_phi_evaluator' / 'core'

# Log paths for debugging
logger.info(f"Current dir: {current_dir}")
logger.info(f"Project root: {project_root}")
logger.info(f"Deployment root: {deployment_root}")
logger.info(f"Shared root: {shared_root}")
logger.info(f"App4 root: {app4_root}")

# Add to Python path
for p in [str(deployment_root), str(shared_root), str(app4_root),
          str(app1_core), str(app2_core), str(app3_core)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Page configuration
st.set_page_config(
    page_title="Vector Precognition Desktop - App4 + ChatGPT",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import App4 components
try:
    # Verify paths exist before importing
    if not shared_root.exists():
        error_msg = f"""
        Shared directory not found!

        Looking for: {shared_root}
        Current dir: {current_dir}
        Project root: {project_root}

        Make sure you're running from: {current_dir}
        And that the deployment/ directory exists at: {deployment_root}
        """
        raise ImportError(error_msg)

    if not app4_root.exists():
        error_msg = f"""
        App4 directory not found!

        Looking for: {app4_root}

        Make sure the deployment/app4_unified_dashboard directory exists.
        """
        raise ImportError(error_msg)

    # Add all necessary paths first
    sys.path.insert(0, str(app4_root))
    sys.path.insert(0, str(app4_root / 'utils'))
    sys.path.insert(0, str(app4_root / 'ui'))
    sys.path.insert(0, str(app4_root / 'core'))

    # Import using importlib to avoid path issues
    import importlib.util

    # Import pca_pipeline
    spec = importlib.util.spec_from_file_location("pca_pipeline", str(shared_root / 'pca_pipeline.py'))
    pca_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pca_module)
    PCATransformer = pca_module.PCATransformer

    # Import vector_processor
    spec = importlib.util.spec_from_file_location("vector_processor", str(app2_core / 'vector_processor.py'))
    vp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vp_module)
    VectorPrecognitionProcessor = vp_module.VectorPrecognitionProcessor

    # Import robustness_calculator
    spec = importlib.util.spec_from_file_location("robustness_calculator", str(app2_core / 'robustness_calculator.py'))
    rc_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc_module)
    RobustnessCalculator = rc_module.RobustnessCalculator

    # Import fragility_calculator
    spec = importlib.util.spec_from_file_location("fragility_calculator", str(app3_core / 'fragility_calculator.py'))
    fc_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fc_module)
    FragilityCalculator = fc_module.FragilityCalculator

    # Import visualizations
    spec = importlib.util.spec_from_file_location("visualizations", str(shared_root / 'visualizations.py'))
    viz_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz_module)
    GuardrailVisualizer = viz_module.GuardrailVisualizer
    RHOVisualizer = viz_module.RHOVisualizer
    PHIVisualizer = viz_module.PHIVisualizer

    # Import session_state
    spec = importlib.util.spec_from_file_location("session_state", str(app4_root / 'utils' / 'session_state.py'))
    session_state_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(session_state_module)
    SessionState = session_state_module.SessionState

    # Import sidebar
    spec = importlib.util.spec_from_file_location("sidebar", str(app4_root / 'ui' / 'sidebar.py'))
    sidebar_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sidebar_module)
    create_sidebar = sidebar_module.create_sidebar

    # Import chat_view
    spec = importlib.util.spec_from_file_location("chat_view", str(app4_root / 'ui' / 'chat_view.py'))
    chat_view_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chat_view_module)
    create_chat_view = chat_view_module.create_chat_view

    # Import pipeline_orchestrator
    spec = importlib.util.spec_from_file_location("pipeline_orchestrator", str(app4_root / 'core' / 'pipeline_orchestrator.py'))
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    PipelineOrchestrator = pipeline_module.PipelineOrchestrator

    logger.info("Successfully imported App4 components")
except ImportError as e:
    st.error(f"Failed to import App4 components: {e}")
    logger.error(f"Import error: {e}", exc_info=True)
    st.stop()
except Exception as e:
    st.error(f"Error during import: {e}")
    logger.error(f"General import error: {e}", exc_info=True)
    st.stop()

# Import ChatGPT client (create this next)
try:
    from chatgpt_integration import ChatGPTClient, check_api_key
    CHATGPT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ChatGPT client not available: {e}")
    CHATGPT_AVAILABLE = False


def show_api_key_setup():
    """Show API key setup screen if not configured."""
    st.title("🔐 ChatGPT API Configuration")

    st.markdown("""
    Welcome to **Vector Precognition Desktop (App4 + ChatGPT)**!

    This application combines the full App4 Unified Dashboard with ChatGPT integration
    for real-time AI safety monitoring.

    ### Setup Instructions
    1. Obtain your OpenAI API key from [platform.openai.com](https://platform.openai.com/api-keys)
    2. Enter your API key below (securely stored by Electron)
    3. Select your preferred GPT model
    4. Start using the full App4 dashboard!
    """)

    # Check if running in Electron
    is_electron = os.environ.get('ELECTRON_MODE') == 'true'

    # API Key Input
    st.subheader("Step 1: Enter OpenAI API Key")

    # Get existing key from environment (set by Electron)
    existing_key = os.environ.get('OPENAI_API_KEY', '')

    if existing_key and existing_key.startswith('sk-'):
        st.success(f"✅ API Key configured: {existing_key[:10]}...{existing_key[-4:]}")
        st.info("Your API key is securely stored by Electron. You can change it below.")

    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=existing_key if existing_key else "",
        placeholder="sk-...",
        help="Your API key is stored securely in encrypted Electron storage"
    )

    # Model Selection
    st.subheader("Step 2: Select ChatGPT Model")
    model_options = {
        "GPT-3.5 Turbo (Recommended)": "gpt-3.5-turbo",
        "GPT-4o Mini": "gpt-4o-mini",
        "GPT-4o": "gpt-4o",
        "GPT-4 Turbo": "gpt-4-turbo",
        "GPT-4": "gpt-4"
    }

    selected_model_name = st.selectbox(
        "ChatGPT Model",
        options=list(model_options.keys()),
        index=0,
        help="GPT-3.5 Turbo is fastest and most cost-effective. GPT-4 models require upgraded API access."
    )
    selected_model = model_options[selected_model_name]

    # Save button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("💾 Save & Continue", type="primary", use_container_width=True):
            if not api_key_input:
                st.error("❌ Please enter your OpenAI API key")
            elif not api_key_input.startswith("sk-"):
                st.error("❌ Invalid API key format (should start with 'sk-')")
            else:
                # Store in session state
                st.session_state.openai_api_key = api_key_input
                st.session_state.openai_model = selected_model
                st.session_state.api_key_configured = True

                # If in Electron, the key is already passed via environment
                # Just mark as configured
                if is_electron:
                    st.info("🔄 API key will be securely stored by Electron on restart")

                st.success("✅ Configuration saved! Loading App4...")
                st.rerun()

    st.divider()

    # Testing without API key
    with st.expander("🧪 Test without API key (Mock Mode)"):
        st.markdown("""
        You can test the App4 interface without an OpenAI API key using Mock LLM mode.
        This will simulate responses without making actual API calls.
        """)
        if st.button("Use Mock Mode", use_container_width=True):
            st.session_state.openai_api_key = "mock"
            st.session_state.openai_model = "mock"
            st.session_state.api_key_configured = True
            st.session_state.use_mock = True
            st.rerun()


def initialize_app():
    """Initialize App4 with ChatGPT client."""
    # Initialize session state
    SessionState.initialize()

    # Create sidebar (this will use modified version for ChatGPT)
    sidebar = create_sidebar()
    config = sidebar.render()

    # Override model config with ChatGPT
    if not st.session_state.get('use_mock', False):
        config['model']['model_name'] = st.session_state.get('openai_model', 'gpt-3.5-turbo')
        config['model']['model_key'] = 'chatgpt'
        config['model']['use_mock'] = False
    else:
        config['model']['use_mock'] = True

    # Initialize PCA transformer
    if SessionState.get('pca_transformer') is None:
        try:
            pca = PCATransformer()
            SessionState.set('pca_transformer', pca)
            logger.info("PCA transformer initialized")
        except Exception as e:
            st.error(f"Error initializing PCA: {e}")
            logger.error(f"PCA initialization error: {e}")
            return None

    pca = SessionState.get('pca_transformer')

    # Initialize VSAFE vector
    vsafe_text = config['vsafe']['text']
    if SessionState.get_vsafe_text() != vsafe_text or SessionState.get_vsafe_vector() is None:
        vsafe_vector = pca.text_to_2d(vsafe_text)
        if vsafe_vector is not None:
            SessionState.set_vsafe_text(vsafe_text)
            SessionState.set_vsafe_vector(vsafe_vector)
            logger.info("VSAFE vector updated")

    vsafe = SessionState.get_vsafe_vector()

    # Initialize orchestrator
    if SessionState.get_orchestrator() is None:
        weights = config['algorithm']
        epsilon = config['alerts']['epsilon']
        phi_threshold = config['alerts']['phi_threshold']

        vector_processor = VectorPrecognitionProcessor(vsafe=vsafe, weights=weights)
        robustness_calculator = RobustnessCalculator(epsilon=epsilon)
        fragility_calculator = FragilityCalculator(phi_threshold=phi_threshold)

        orchestrator = PipelineOrchestrator(
            vector_processor,
            robustness_calculator,
            fragility_calculator
        )

        SessionState.set_orchestrator(orchestrator)
        logger.info("Pipeline orchestrator initialized")

    orchestrator = SessionState.get_orchestrator()

    # Initialize ChatGPT client
    if SessionState.get_llm_client() is None or SessionState.get_selected_model() != config['model']['model_name']:
        if not st.session_state.get('use_mock', False):
            api_key = st.session_state.get('openai_api_key', os.environ.get('OPENAI_API_KEY', ''))
            model = st.session_state.get('openai_model', 'gpt-3.5-turbo')

            llm_client = ChatGPTClient(api_key=api_key, model=model)
            logger.info(f"ChatGPT client initialized: {model}")
        else:
            # Mock client
            from core.api_client import create_llm_client
            llm_client = create_llm_client('mock', use_mock=True)
            logger.info("Mock LLM client initialized")

        # Set system prompt
        system_prompt = SessionState.get_system_prompt()
        llm_client.add_system_message(system_prompt)

        SessionState.set_llm_client(llm_client)
        SessionState.set_selected_model(config['model']['model_name'])

    llm_client = SessionState.get_llm_client()

    return config, orchestrator, llm_client, pca


def main():
    """Main application entry point."""

    # Check if API key is configured
    if 'api_key_configured' not in st.session_state:
        # Check environment variable (set by Electron)
        env_key = os.environ.get('OPENAI_API_KEY', '')
        if env_key and env_key.startswith('sk-'):
            st.session_state.openai_api_key = env_key
            st.session_state.openai_model = 'gpt-3.5-turbo'
            st.session_state.api_key_configured = True
        else:
            st.session_state.api_key_configured = False

    # Show API key setup if not configured
    if not st.session_state.api_key_configured:
        show_api_key_setup()
        return

    # Title
    col_title, col_logo = st.columns([7, 3])
    with col_title:
        st.title("🛡️ Vector Precognition Desktop")
        st.caption("App4 Unified Dashboard + ChatGPT Integration")
    with col_logo:
        logo_path = deployment_root / 'shared' / 'images' / '1.png'
        if logo_path.exists():
            st.image(str(logo_path))

    # Initialize app
    try:
        result = initialize_app()
        if result is None:
            st.error("Failed to initialize application")
            return

        config, orchestrator, llm_client, pca = result
    except Exception as e:
        st.error(f"Initialization error: {e}")
        logger.error(f"Initialization error: {e}", exc_info=True)
        return

    # Import App4 tab renderers using importlib to avoid circular imports
    spec = importlib.util.spec_from_file_location("app4_main", str(app4_root / 'app.py'))
    app4_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app4_main)

    render_tab1_live_chat = app4_main.render_tab1_live_chat
    render_tab2_rho_analysis = app4_main.render_tab2_rho_analysis
    render_tab3_phi_benchmark = app4_main.render_tab3_phi_benchmark
    render_tab4_settings = app4_main.render_tab4_settings

    # Create tabs (same as App4)
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Live Chat",
        "📊 RHO Analysis",
        "🎯 PHI Benchmark",
        "⚙️ Settings"
    ])

    with tab1:
        render_tab1_live_chat(config, orchestrator, llm_client, pca)

    with tab2:
        render_tab2_rho_analysis(config, orchestrator)

    with tab3:
        render_tab3_phi_benchmark(config, orchestrator)

    with tab4:
        render_tab4_settings(config, orchestrator)


if __name__ == "__main__":
    main()
