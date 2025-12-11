# Desktop App2 Architecture

**Technical design document for developers**

---

## System Overview

Desktop App2 is a hybrid Electron + Python application that combines:
- **Electron**: Native desktop wrapper, API key storage, window management
- **Streamlit**: Web UI framework running locally on port 8501
- **App4**: Full unified dashboard codebase (imported from deployment/)
- **ChatGPT**: OpenAI API client for language model integration

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────┐
│                    ELECTRON MAIN PROCESS                   │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  main.js                                            │  │
│  │  - Creates BrowserWindow                            │  │
│  │  - Spawns Python backend (Streamlit)               │  │
│  │  - Waits for http://localhost:8501                 │  │
│  │  - Loads Streamlit in window                       │  │
│  │  - IPC handlers for API keys                       │  │
│  └─────────────────────────────────────────────────────┘  │
│                           ▲                                │
│                           │ IPC Messages                   │
│                           ▼                                │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  preload.js (Security Bridge)                      │  │
│  │  - Exposes electronAPI to renderer                │  │
│  │  - Context isolation enabled                       │  │
│  └─────────────────────────────────────────────────────┘  │
│                           │                                │
│                           │ HTTP (localhost:8501)          │
│                           ▼                                │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  BrowserWindow (Renderer)                          │  │
│  │  - Loads Streamlit web UI                         │  │
│  │  - Can call electronAPI (via preload)             │  │
│  └─────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP Requests
                            ▼
┌────────────────────────────────────────────────────────────┐
│              PYTHON BACKEND (Streamlit)                    │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  app.py (Main Entry Point)                         │  │
│  │  ┌─────────────────────────────────────────────┐   │  │
│  │  │  API Key Setup Screen                       │   │  │
│  │  │  - if not st.session_state.api_key_configured│  │
│  │  │  - Prompts for OpenAI key                   │   │  │
│  │  │  - Model selection                           │   │  │
│  │  └─────────────────────────────────────────────┘   │  │
│  │  ┌─────────────────────────────────────────────┐   │  │
│  │  │  App4 Dashboard                             │   │  │
│  │  │  - if api_key_configured                    │   │  │
│  │  │  - Renders 4 tabs                           │   │  │
│  │  │  - Initializes components                   │   │  │
│  │  └─────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────┘  │
│                           │                                │
│                           │ Imports                        │
│                           ▼                                │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  chatgpt_integration.py                            │  │
│  │  ┌─────────────────────────────────────────────┐   │  │
│  │  │  class ChatGPTClient                        │   │  │
│  │  │  - __init__(api_key, model)                 │   │  │
│  │  │  - send_message(msg, temp, max_tokens)      │   │  │
│  │  │  - clear_history()                           │   │  │
│  │  │  - export_conversation(path)                 │   │  │
│  │  │  - add_system_message(prompt)                │   │  │
│  │  └─────────────────────────────────────────────┘   │  │
│  │  Compatible with App4's LLM client interface       │  │
│  └─────────────────────────────────────────────────────┘  │
│                           │                                │
│                           │ Uses                           │
│                           ▼                                │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  App4 Components (from ../../deployment/)         │  │
│  │  ┌─────────────────────────────────────────────┐   │  │
│  │  │  core/pipeline_orchestrator.py              │   │  │
│  │  │  - Orchestrates 3 stages                    │   │  │
│  │  │  - Manages conversations                    │   │  │
│  │  │  - add_turn(), calculate_stage2/3()         │   │  │
│  │  └─────────────────────────────────────────────┘   │  │
│  │  ┌─────────────────────────────────────────────┐   │  │
│  │  │  ui/chat_view.py, sidebar.py                │   │  │
│  │  │  - Renders UI components                    │   │  │
│  │  │  - Chat history, metrics, controls          │   │  │
│  │  └─────────────────────────────────────────────┘   │  │
│  │  ┌─────────────────────────────────────────────┐   │  │
│  │  │  app.py (Tab Renderers)                     │   │  │
│  │  │  - render_tab1_live_chat()                  │   │  │
│  │  │  - render_tab2_rho_analysis()               │   │  │
│  │  │  - render_tab3_phi_benchmark()              │   │  │
│  │  │  - render_tab4_settings()                   │   │  │
│  │  └─────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────┘  │
│                           │                                │
│                           │ Uses                           │
│                           ▼                                │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Shared Modules (from ../../deployment/shared/)   │  │
│  │  - pca_pipeline.py (PCATransformer)                │  │
│  │  - vector_processor.py (Stage 1)                   │  │
│  │  - robustness_calculator.py (Stage 2)              │  │
│  │  - fragility_calculator.py (Stage 3)               │  │
│  │  - visualizations.py (Plots)                       │  │
│  └─────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                            │
                            │ API Calls
                            ▼
                   ┌──────────────────┐
                   │  OpenAI API      │
                   │  (ChatGPT)       │
                   └──────────────────┘
```

---

## Component Details

### 1. Electron Layer

#### main.js

**Responsibilities**:
- Create and manage BrowserWindow
- Launch Python backend (Streamlit)
- Wait for Streamlit server to start
- Load Streamlit UI in window
- Handle IPC messages
- Manage secure storage

**Key Functions**:

```javascript
startPythonBackend()
// Spawns: python -m streamlit run app.py
// Env: OPENAI_API_KEY, ELECTRON_MODE

waitForStreamlit(maxRetries, delay)
// Polls http://localhost:8501 until ready

createWindow()
// Creates BrowserWindow with preload script
// Loads STREAMLIT_URL
```

**IPC Handlers**:

```javascript
ipcMain.handle('store-openai-key', async (event, apiKey))
// Stores API key in encrypted electron-store
// Restarts Python backend with new key

ipcMain.handle('get-openai-key', async ())
// Retrieves stored API key

ipcMain.handle('delete-openai-key', async ())
// Deletes stored API key

ipcMain.handle('store-aws-credentials', async (event, credentials))
// Stores AWS credentials for embeddings

ipcMain.handle('get-aws-credentials', async ())
// Retrieves AWS credentials
```

#### preload.js

**Responsibilities**:
- Exposes secure API to renderer process
- Context isolation bridge

**Exposed API**:

```javascript
window.electronAPI = {
  storeOpenAIKey: (key) => ipcRenderer.invoke('store-openai-key', key),
  getOpenAIKey: () => ipcRenderer.invoke('get-openai-key'),
  deleteOpenAIKey: () => ipcRenderer.invoke('delete-openai-key'),
  checkAPIKey: () => ipcRenderer.invoke('check-api-key'),
  storeAWSCredentials: (creds) => ipcRenderer.invoke('store-aws-credentials', creds),
  getAWSCredentials: () => ipcRenderer.invoke('get-aws-credentials'),
  isElectron: () => true
}
```

---

### 2. Python Backend

#### app.py

**Responsibilities**:
- Main Streamlit entry point
- API key setup screen
- App4 initialization
- Tab rendering coordination

**Flow**:

```python
def main():
    # Check if API key configured
    if not st.session_state.api_key_configured:
        # Check environment (set by Electron)
        if env_key exists:
            auto-configure
        else:
            show_api_key_setup()
            return

    # Initialize App4
    config, orchestrator, llm_client, pca = initialize_app()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([...])

    # Render tabs (using App4's renderers)
    render_tab1_live_chat(...)
    render_tab2_rho_analysis(...)
    render_tab3_phi_benchmark(...)
    render_tab4_settings(...)
```

**Key Functions**:

```python
show_api_key_setup()
# Displays API key input form
# Model selection dropdown
# Save button → stores in session_state

initialize_app()
# Imports App4 components
# Creates PipelineOrchestrator
# Initializes ChatGPTClient
# Returns (config, orchestrator, llm_client, pca)
```

#### chatgpt_integration.py

**Responsibilities**:
- Wrap OpenAI API
- Match App4's LLM client interface
- Maintain conversation history

**ChatGPTClient Class**:

```python
class ChatGPTClient:
    def __init__(self, api_key, model):
        self.client = OpenAI(api_key)
        self.model = model
        self.conversation_history = []
        self.system_message = "..."

    def send_message(self, user_message, temperature, max_tokens):
        # Build messages list (system + history + current)
        # Call OpenAI API
        # Update conversation_history
        # Return (response, success)

    def clear_history(self):
        # Reset conversation_history

    def add_system_message(self, message):
        # Update system_message

    def export_conversation(self, filepath):
        # Save to JSON
```

**Interface Compatibility**:

The ChatGPTClient matches App4's expected LLM client interface:

```python
# App4 expects:
response, success = llm_client.send_message(msg, temp, max_tokens)
llm_client.clear_history()
llm_client.add_system_message(prompt)
llm_client.export_conversation(path)

# ChatGPTClient provides exactly this!
```

---

### 3. App4 Components (Imported)

#### From deployment/app4_unified_dashboard/

**core/pipeline_orchestrator.py**:
- Orchestrates 3-stage pipeline
- Manages conversations
- `start_new_conversation()`, `end_conversation()`
- `add_turn()` → Stage 1
- `calculate_stage2_rho()` → Stage 2
- `calculate_stage3_phi()` → Stage 3

**ui/chat_view.py**:
- `create_chat_view()` → ChatView instance
- `render_chat_history()` → Shows messages
- `render_input_area()` → Chat input box
- `render_live_metrics()` → R, v, a, L cards
- `render_live_visualization()` → 4-panel plot
- `render_conversation_controls()` → Start/End/Export buttons

**ui/sidebar.py**:
- `create_sidebar()` → Sidebar instance
- `render()` → Returns config dict
- Model selection, VSAFE config, algorithm weights, etc.

**app.py Tab Renderers**:
- `render_tab1_live_chat()` → Chat + real-time metrics
- `render_tab2_rho_analysis()` → Conversation selector + RHO plots
- `render_tab3_phi_benchmark()` → Multi-conversation PHI analysis
- `render_tab4_settings()` → Session info + configuration

#### From deployment/shared/

**pca_pipeline.py**:
- `PCATransformer` class
- `text_to_2d(text)` → Converts text to 2D vector via embeddings + PCA

**vector_processor.py**:
- `VectorPrecognitionProcessor` class (Stage 1)
- Calculates R, v, a, z, L for each turn

**robustness_calculator.py**:
- `RobustnessCalculator` class (Stage 2)
- Calculates RHO from conversation metrics

**fragility_calculator.py**:
- `FragilityCalculator` class (Stage 3)
- Calculates PHI from multiple RHO values

**visualizations.py**:
- `GuardrailVisualizer`, `RHOVisualizer`, `PHIVisualizer`
- Matplotlib plotting functions

---

## Data Flow

### Conversation Turn Flow

```
1. User types message in Tab 1 chat input
   ↓
2. Streamlit detects input via st.chat_input()
   ↓
3. app.py adds message to SessionState
   ↓
4. PCATransformer.text_to_2d(user_message)
   → Calls AWS Bedrock for embedding
   → Applies PCA to get 2D vector
   ↓
5. ChatGPTClient.send_message(user_message, temp, max_tokens)
   → Calls OpenAI API
   → Returns assistant response
   ↓
6. app.py adds response to SessionState
   ↓
7. PCATransformer.text_to_2d(assistant_message)
   → Get assistant 2D vector
   ↓
8. orchestrator.add_turn(user_msg, asst_msg, user_vec, asst_vec)
   → VectorProcessor calculates R, v, a, z, L
   → Stores in conversation metrics
   ↓
9. st.rerun() → UI updates with new metrics
   ↓
10. ChatView renders updated chat + metrics
```

### RHO Calculation Flow

```
1. User clicks "End Conversation" in Tab 1
   ↓
2. orchestrator.end_conversation()
   → Marks conversation as completed
   → Stores in conversation_history
   ↓
3. User navigates to Tab 2 (RHO Analysis)
   ↓
4. render_tab2_rho_analysis() displays conversation selector
   ↓
5. User selects conversation from dropdown
   ↓
6. Check if stage2_result exists
   ↓ (if not calculated)
7. orchestrator.calculate_stage2_rho()
   → RobustnessCalculator processes metrics
   → Calculates final RHO
   → Classification (ROBUST/REACTIVE/FRAGILE)
   ↓
8. RHOVisualizer creates plots:
   - Cumulative risk (user vs model)
   - RHO timeline
   ↓
9. Display results + plots
```

### PHI Aggregation Flow

```
1. User completes multiple conversations
   ↓
2. Each conversation has RHO calculated (Tab 2)
   ↓
3. User navigates to Tab 3 (PHI Benchmark)
   ↓
4. render_tab3_phi_benchmark() filters conversations with RHO
   ↓
5. orchestrator.calculate_stage3_phi(model_name)
   → FragilityCalculator aggregates RHO values
   → PHI = (1/N) * sum(max(0, rho - 1))
   → Classification (PASS if PHI < 0.1, else FAIL)
   ↓
6. PHIVisualizer creates fragility distribution histogram
   ↓
7. Display breakdown table + PHI score + plot
```

---

## Security Model

### Electron Security

1. **Context Isolation**: Enabled in BrowserWindow
   - Renderer can't access Node.js directly
   - Only preload script has access

2. **Node Integration**: Disabled
   - Renderer runs as regular web page
   - No require() or process access

3. **Web Security**: Enabled
   - CORS, CSP enforced

4. **Preload Script**: Whitelist-only API
   - Only exposes specific IPC handlers
   - No arbitrary code execution

### API Key Storage

1. **electron-store**: AES encryption
2. **Encryption key**: Hardcoded in main.js (should be per-user in production)
3. **Storage location**:
   - Windows: `%APPDATA%/vector-precognition-app4`
   - macOS: `~/Library/Application Support/vector-precognition-app4`
   - Linux: `~/.config/vector-precognition-app4`

4. **Transmission**: Environment variables only
   - Never sent over network (except to OpenAI API)
   - Never logged

### Best Practices

- ✅ API keys encrypted at rest
- ✅ Context isolation prevents XSS
- ✅ No remote code execution
- ✅ HTTPS-only for OpenAI API
- ⚠️ Encryption key should be per-user (improvement needed)

---

## Performance Considerations

### Bottlenecks

1. **Streamlit Startup**: 10-30 seconds first launch
   - Solution: Show loading screen in Electron
   - Future: Bundle Streamlit with pyinstaller

2. **PCA Transformation**: 2-3 seconds per message
   - Calls AWS Bedrock for embedding (network latency)
   - Solution: Cache embeddings, use local models

3. **Electron Bundle Size**: ~200MB
   - Includes Chromium + Node.js
   - Solution: Use electron-builder compression

4. **Streamlit Reruns**: Full page refresh on interaction
   - Inherent to Streamlit architecture
   - Solution: Use st.experimental_fragment for partial updates

### Optimizations

1. **Lazy Imports**: Only import App4 modules when needed
2. **Session State**: Cache PCA transformer, orchestrator
3. **Matplotlib**: Use st.pyplot(fig, clear_figure=True) to prevent memory leaks
4. **IPC**: Batch multiple operations where possible

---

## Testing Strategy

### Unit Tests

- chatgpt_integration.py: Mock OpenAI API
- Pipeline components: Use App4's existing tests

### Integration Tests

1. **Electron ↔ Python**: Test IPC handlers
2. **ChatGPT ↔ App4**: Test interface compatibility
3. **End-to-end**: Full conversation flow

### Manual Testing Checklist

- [ ] API key storage/retrieval
- [ ] Conversation with ChatGPT
- [ ] Metrics calculation (R, v, a, L)
- [ ] RHO analysis
- [ ] PHI benchmark
- [ ] Export functionality
- [ ] Mock mode
- [ ] Installer build

---

## Deployment

### Development

```bash
# Python only (browser mode)
cd python-backend
streamlit run app.py

# Electron wrapper
cd electron
npm run dev  # Opens with DevTools
```

### Production

```bash
# Build installers
cd electron
npm run build:win   # Windows NSIS
npm run build:mac   # macOS DMG
npm run build:linux # AppImage + deb
```

### Distribution

Installers include:
- Electron binary
- Python backend (all .py files)
- Node modules
- Not included: Python interpreter (user must have Python)

Future: Bundle Python with pyinstaller for true single-exe distribution.

---

## Extension Points

### Adding New LLM Providers

1. Create new client in `python-backend/`:
   ```python
   class AnthropicClient:
       def send_message(self, msg, temp, max_tokens):
           # Match interface
   ```

2. Update `initialize_app()` to select client based on config

3. Add IPC handlers for new provider's API keys

### Adding New Tabs

1. Create renderer in App4 or desktop-app2:
   ```python
   def render_tab5_custom(config, orchestrator):
       st.header("Custom Tab")
       # Your code
   ```

2. Add tab in main():
   ```python
   tab5 = st.tabs([..., "🔧 Custom"])
   with tab5:
       render_tab5_custom(...)
   ```

### Customizing Visualizations

Modify `deployment/shared/visualizations.py` or create desktop-app2 specific versions.

---

## Known Limitations

1. **Python required**: User must have Python installed (can't bundle yet)
2. **Single instance**: Only one conversation at a time
3. **No offline mode**: Requires internet for embeddings + ChatGPT
4. **WSL limitations**: Needs X server for GUI

---

**Version**: 2.0.0
**Last Updated**: December 11, 2024
**Maintainer**: Optica Labs
