# Sycophancy Tab Integration Patch

This document shows the changes needed to integrate the Sycophancy tab into app.py.

## Changes Required

### 1. Add Import Statement (around line 76)

**After this line:**
```python
from visualizations import GuardrailVisualizer, RHOVisualizer, PHIVisualizer
```

**Add:**
```python
from sycophancy_tab_integration import render_tab5_sycophancy_analysis, update_sycophancy_analyzer_with_turn
```

---

### 2. Update Tab Creation (around line 745)

**Replace:**
```python
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Live Chat",
    "📊 RHO Analysis",
    "🎯 PHI Benchmark",
    "⚙️ Settings"
])
```

**With:**
```python
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Live Chat",
    "📊 RHO Analysis",
    "🎯 PHI Benchmark",
    "🎭 Sycophancy",
    "⚙️ Settings"
])
```

---

### 3. Add Tab5 Rendering (around line 762)

**After:**
```python
with tab4:
    render_tab4_settings(config, orchestrator)
```

**Add:**
```python
with tab5:
    render_tab5_sycophancy_analysis(config, orchestrator, pca)
```

---

### 4. (Optional) Update Chat Processing to Track Sycophancy

**In the chat processing logic** (around line 220-250 in render_tab1_live_chat function),
**after processing a turn with the orchestrator**, add:

```python
# After getting v_user and v_model vectors
if v_user is not None and v_model is not None:
    # Update sycophancy analyzer
    current_turn = len(orchestrator.current_processor.R_model_list)
    update_sycophancy_analyzer_with_turn(v_user, v_model, current_turn)
```

---

## Complete Integration Steps

1. **Copy the new files** to the app4 directory:
   - `core/sycophancy_analyzer.py` (already created)
   - `ui/sycophancy_view.py` (already created)
   - `sycophancy_tab_integration.py` (already created)

2. **Apply the patches** above to `app.py`

3. **Test the integration:**
   ```bash
   cd deployment/app4_unified_dashboard
   streamlit run app.py --server.port 8504
   ```

4. **Verify functionality:**
   - Start a new conversation in Live Chat tab
   - Navigate to Sycophancy tab
   - Verify that metrics are displayed
   - Check that visualizations render correctly

---

## Example: Minimal Working Version

If you want a quick integration without modifying existing chat logic, you can use the tab
in "analysis mode" where it reads data from the orchestrator after conversations are complete.

The current implementation already does this - it syncs with the orchestrator's data automatically.

---

## Troubleshooting

### Issue: "No data available" message

**Cause:** Sycophancy analyzer not receiving vector data

**Solution:**
- Ensure vectors are being stored in session state
- Add debug logging to verify v_user and v_model are being generated
- Check that `current_conversation_id` is set

### Issue: Import errors

**Cause:** Module not found

**Solution:**
- Verify all new files are in the correct directories
- Check that Python path includes the app4 directory
- Restart the Streamlit server

### Issue: VSYC vector not generating

**Cause:** PCA transformer not initialized or AWS credentials missing

**Solution:**
- Ensure PCA models are trained: `python ../../src/pca_trainer.py`
- Verify AWS credentials are configured
- Check Bedrock access permissions

---

## File Structure After Integration

```
deployment/app4_unified_dashboard/
├── app.py  (modified)
├── sycophancy_tab_integration.py  (new)
├── core/
│   ├── sycophancy_analyzer.py  (new)
│   └── ... (existing files)
└── ui/
    ├── sycophancy_view.py  (new)
    └── ... (existing files)
```

---

## Testing Checklist

- [ ] Import statements work without errors
- [ ] Tab appears in the dashboard
- [ ] VSYC vector can be configured
- [ ] Metrics cards display correctly
- [ ] Sycophancy Trap plot renders
- [ ] Time series plot renders
- [ ] Data table shows correct columns
- [ ] Quadrant distribution pie chart works
- [ ] CSV download functions correctly
- [ ] Analyzer syncs with conversation data
- [ ] No errors in console/logs
