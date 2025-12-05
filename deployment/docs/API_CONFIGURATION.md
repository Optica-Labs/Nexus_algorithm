# API Configuration Guide

**Last Updated**: December 2, 2025

This guide explains how to configure and use the AWS Lambda API endpoints for LLM chat functionality in App 1 and App 4.

---

## 🔗 AWS Lambda Endpoints

Your existing AWS Lambda infrastructure provides API Gateway endpoints for all 4 LLM models:

```python
AWS_LAMBDA_ENDPOINTS = {
    "gpt-3.5": "https://kv854u79y7.execute-api.us-east-1.amazonaws.com/prod/chat",
    "gpt-4": "https://1d4qnutnqc.execute-api.us-east-1.amazonaws.com/prod/chat",
    "claude": "https://6z5nnwuyyj.execute-api.us-east-1.amazonaws.com/prod/chat",
    "mistral": "https://q1gqmwwmsf.execute-api.us-east-1.amazonaws.com/prod/chat"
}
```

**Advantages of using AWS Lambda endpoints:**
- ✅ No API keys needed in the application
- ✅ Centralized billing and rate limiting through AWS
- ✅ Consistent interface across all models
- ✅ Better security (API keys stored in Lambda environment)

---

## 📦 Which Apps Use These Endpoints?

### App 1: Guardrail Erosion Analyzer
- **Input Method 4**: "API Integration"
- Allows testing real-time conversations with LLMs
- Select model from dropdown (GPT-3.5, GPT-4, Claude, Mistral)

### App 4: Unified Dashboard
- **Live Chat Interface**: Primary feature
- Real-time safety monitoring while chatting with LLMs
- Model selection in sidebar
- Mock client option for testing without endpoints

### Apps 2 & 3
- Do NOT directly use LLM endpoints
- App 2 processes existing conversation data
- App 3 aggregates RHO values (no LLM needed)

---

## 🧪 Testing Endpoints

### Quick Test Script

Test all endpoints with:

```bash
cd deployment
python test_api_endpoints.py
```

This script will:
1. Send a test message to each endpoint
2. Verify responses
3. Measure response times
4. Report success/failure for each

**Expected output:**
```
✅ gpt-3.5: SUCCESS (2.34s)
   Response: Hello! This is a test response...
✅ gpt-4: SUCCESS (3.12s)
   Response: Hello! I'd be happy to help...
✅ claude: SUCCESS (2.87s)
   Response: Hello! How can I assist you...
✅ mistral: SUCCESS (2.56s)
   Response: Hello! I'm here to help...
```

### Manual Test with curl

Test individual endpoints:

```bash
# Test GPT-3.5
curl -X POST https://kv854u79y7.execute-api.us-east-1.amazonaws.com/prod/chat \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": [
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

---

## 🔧 Configuration in Code

### App 4: api_client.py

The `LLMAPIClient` class automatically uses AWS Lambda endpoints:

```python
from core.api_client import create_llm_client

# Create client (uses AWS Lambda by default)
client = create_llm_client("gpt-3.5", use_mock=False)

# Send message
response, success = client.send_message("Hello!")
```

**Key features:**
- Automatic endpoint routing
- No API keys required
- Fallback to mock client if needed

### Request Format

All endpoints expect this JSON structure:

```json
{
  "conversation": [
    {"role": "user", "content": "Your message here"},
    {"role": "assistant", "content": "Previous response"},
    {"role": "user", "content": "Follow-up message"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024
}
```

### Response Format

Expected response structure:

```json
{
  "response": "The model's response text here"
}
```

Alternative formats (handled automatically):
```json
{
  "message": "Response text"
}
```

```json
{
  "body": {
    "response": "Response text"
  }
}
```

---

## 🔑 No API Keys Needed!

Unlike direct API access, AWS Lambda endpoints handle authentication internally:

**Direct API (old way):**
```python
# Requires API keys in environment
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
```

**AWS Lambda (new way):**
```python
# No API keys needed in the app!
# Lambda function handles authentication
```

---

## 🎯 Model Selection in UI

### App 4: Unified Dashboard

Model selection in sidebar:

1. Open sidebar
2. Under "Model Selection":
   - Select from dropdown: GPT-3.5, GPT-4, Claude Sonnet 3, Mistral Large
   - Configure temperature (0.0 - 2.0)
   - Set max tokens (128 - 4096)
3. Start conversation
4. Real-time safety monitoring begins

**Mock Client Toggle:**
- Enable for testing without hitting endpoints
- Uses simulated responses
- No network calls made

---

## 📊 Performance Expectations

### Response Times

Typical response times from Lambda endpoints:

| Model | Avg Response Time | Max Response Time |
|-------|------------------|-------------------|
| GPT-3.5 | 2-3 seconds | 5 seconds |
| GPT-4 | 3-5 seconds | 10 seconds |
| Claude | 2-4 seconds | 8 seconds |
| Mistral | 2-3 seconds | 6 seconds |

### Rate Limiting

AWS Lambda endpoints may have rate limits:
- Check AWS API Gateway settings
- Typically: 1000 requests/minute per endpoint
- Burst capacity: Varies by configuration

---

## 🐛 Troubleshooting

### Connection Timeout

**Problem**: Request times out after 60 seconds

**Solutions**:
1. Check internet connectivity
2. Verify endpoint URLs are correct
3. Test with curl first
4. Check AWS Lambda function status in AWS Console

### HTTP 403 Forbidden

**Problem**: Permission denied

**Solutions**:
1. Verify API Gateway is publicly accessible
2. Check Lambda execution role permissions
3. Ensure Lambda function is deployed

### HTTP 500 Internal Server Error

**Problem**: Lambda function error

**Solutions**:
1. Check CloudWatch logs for Lambda function
2. Verify Lambda has correct API keys configured
3. Test Lambda function in AWS Console
4. Check Lambda timeout settings (should be > 30 seconds)

### Unexpected Response Format

**Problem**: Can't parse response

**Solution**:
The code handles multiple response formats automatically. If you see warnings about "Unexpected Lambda response format", check the actual response structure:

```python
# Add logging to see raw response
logger.info(f"Raw Lambda response: {data}")
```

Then adjust parsing logic in `_call_aws_lambda()` method.

---

## 🔄 Switching Between Direct API and Lambda

If you want to use direct API instead of Lambda endpoints, modify `api_client.py`:

```python
# Change provider from AWS_LAMBDA to direct provider
MODEL_CONFIGS = {
    "gpt-3.5": LLMConfig(
        name="GPT-3.5",
        provider=ModelProvider.OPENAI,  # Changed from AWS_LAMBDA
        model_id="gpt-3.5-turbo",
        api_key_env="OPENAI_API_KEY",  # Now required
        max_tokens=1024,
        temperature=0.7
    ),
    # ... etc
}
```

Then set API keys in `.env`:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
```

---

## 📝 Summary

✅ **Current Setup**: All models use AWS Lambda endpoints

✅ **No API Keys Needed**: Lambda handles authentication

✅ **Consistent Interface**: All models use same request/response format

✅ **Easy Testing**: Use `test_api_endpoints.py` to verify

✅ **Fallback Options**: Mock client available for development

---

## 🚀 Quick Start Checklist

- [ ] Test endpoints: `python test_api_endpoints.py`
- [ ] Verify all 4 models respond correctly
- [ ] Try App 4 with mock client first
- [ ] Test live chat with real endpoints
- [ ] Monitor response times and errors
- [ ] Check CloudWatch logs if issues occur

---

For more details on Lambda function configuration, see your AWS Console:
- Service: API Gateway
- Service: Lambda
- Region: us-east-1
- CloudWatch Logs: `/aws/lambda/[function-name]`
