# App 1: Custom Endpoint Feature

## New Feature Added!

You can now add custom API endpoints directly through the UI and talk to them immediately without modifying code.

---

## How to Use

### Method 1: Add Custom Endpoint via UI

1. **Select "4. API Integration"** input method

2. **Click the "➕ Add Custom" button** next to the model selector

3. **Fill in the dialog**:
   - **Endpoint Name**: Give your endpoint a friendly name (e.g., "My Custom LLM")
   - **Endpoint URL**: Enter the full API URL (e.g., `https://api.example.com/chat`)

4. **Click "Add Endpoint"**
   - Your custom endpoint will be added to the dropdown
   - It will persist for the entire session

5. **Select your custom endpoint** from the dropdown

6. **Start chatting immediately!**

---

### Method 2: Edit Endpoint Manually

You can also manually edit the "API Endpoint" text field to quickly test different URLs without saving them.

---

## Features

✅ **Add Multiple Custom Endpoints**
- Add as many custom endpoints as you need
- They all appear in the dropdown menu

✅ **Session Persistence**
- Custom endpoints persist throughout your session
- No need to re-add them each time

✅ **Edit on the Fly**
- Manually override any endpoint URL in the text field

✅ **Use Immediately**
- No code changes required
- Start chatting right away after adding

---

## Example Use Cases

### Use Case 1: Testing Your Own LLM API
```
Endpoint Name: My Local LLM
Endpoint URL: http://localhost:8000/v1/chat
```

### Use Case 2: Custom Fine-tuned Model
```
Endpoint Name: My Fine-tuned GPT
Endpoint URL: https://api.openai.com/v1/custom/my-model
```

### Use Case 3: Alternative LLM Provider
```
Endpoint Name: Cohere Command
Endpoint URL: https://api.cohere.ai/v1/chat
```

---

## API Request Format

The app sends POST requests with this JSON format:
```json
{
  "message": "User's input text"
}
```

**Expected Response Format** (one of these):
```json
{
  "response": "Model's response text"
}
```

OR

```json
{
  "message": "Model's response text"
}
```

---

## UI Screenshot Description

**Before clicking "➕ Add Custom":**
```
┌─────────────────────────────────────────┐
│ Select Model          │ ➕ Add Custom   │
│ [GPT-3.5 Turbo    ▼]  │                 │
└─────────────────────────────────────────┘
```

**After clicking "➕ Add Custom":**
```
─────────────────────────────────────────────
Add Custom Endpoint

Endpoint Name: [e.g., My Custom LLM      ]
Endpoint URL:  [https://api.example.com/chat]

[Add Endpoint]  [Cancel]
─────────────────────────────────────────────
```

**After adding custom endpoint:**
```
┌─────────────────────────────────────────┐
│ Select Model          │ ➕ Add Custom   │
│ [My Custom LLM    ▼]  │                 │
└─────────────────────────────────────────┘
  ↑
  Your custom endpoint now appears here!
```

---

## How It Works

### Session State Storage
```python
st.session_state.custom_endpoints = {
    'custom_0': {
        'name': 'My Custom LLM',
        'url': 'https://api.example.com/chat'
    },
    'custom_1': {
        'name': 'Another LLM',
        'url': 'https://another-api.com/chat'
    }
}
```

### Combined with Predefined Endpoints
The dropdown shows:
- All predefined endpoints (GPT-3.5, GPT-4, Claude, Mistral)
- All custom endpoints you've added

### Immediate Usage
After adding a custom endpoint:
1. It appears in the dropdown immediately
2. Select it
3. Send messages and get responses
4. Build a conversation
5. Analyze with Vector Precognition!

---

## Benefits

### For Development
- Test your own LLM APIs instantly
- No need to modify config files
- Quick iteration on different endpoints

### For Research
- Compare different LLM providers
- Test various API configurations
- Rapid prototyping

### For Production
- Easily switch between staging/production endpoints
- Test new endpoints before hardcoding them
- Flexible deployment

---

## Notes

⚠️ **Session Only**: Custom endpoints are stored in session state and will be cleared when you close the browser or refresh the page.

💡 **Tip**: If you find yourself using the same custom endpoint frequently, you can add it permanently to `shared/config.py` in the `LLM_ENDPOINTS` dictionary.

🔒 **Security**: Make sure to only connect to trusted endpoints. The app sends conversation data to these endpoints.

---

## Code Location

**File**: `app1_guardrail_erosion/app.py`
**Function**: `input_method_4_api()`
**Lines**: 351-405

---

## Testing

### Quick Test:
1. Go to App 1
2. Select "4. API Integration"
3. Click "➕ Add Custom"
4. Add name: "Test Endpoint"
5. Add URL: "https://httpbin.org/post"
6. Click "Add Endpoint"
7. Verify it appears in dropdown
8. Select it and try sending a message

---

## Status

✅ **Feature Complete**
✅ **Ready to Use**
🧪 **Needs Testing**

---

## Related Documentation

- [APP1_FINAL_GUIDE.md](APP1_FINAL_GUIDE.md) - Complete App 1 testing guide
- [API_CONFIGURATION.md](API_CONFIGURATION.md) - AWS Lambda endpoint configuration
- [INTEGRATION_TESTING.md](INTEGRATION_TESTING.md) - Full integration testing procedures
