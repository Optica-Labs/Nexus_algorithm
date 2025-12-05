# Environment Setup Guide - Vector Precognition Suite

## Overview

This guide explains how to obtain and configure all environment variables required for the Vector Precognition deployment suite.

---

## Table of Contents

1. [Required vs Optional Variables](#required-vs-optional-variables)
2. [AWS Credentials Setup](#aws-credentials-setup)
3. [LLM API Endpoints Setup](#llm-api-endpoints-setup)
4. [Configuration Steps](#configuration-steps)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## Required vs Optional Variables

### Required (All Apps)

These variables are **required** for basic functionality:

| Variable | Used By | Purpose |
|----------|---------|---------|
| `AWS_ACCESS_KEY_ID` | All apps | AWS Bedrock authentication |
| `AWS_SECRET_ACCESS_KEY` | All apps | AWS Bedrock authentication |
| `AWS_DEFAULT_REGION` | All apps | AWS Bedrock region |

### Optional (App 4 Only)

These variables are **optional** and only needed for App 4 (Unified Dashboard) live chat:

| Variable | Used By | Purpose |
|----------|---------|---------|
| `GPT35_ENDPOINT` | App 4 | GPT-3.5 Turbo API endpoint |
| `GPT4_ENDPOINT` | App 4 | GPT-4 API endpoint |
| `CLAUDE_ENDPOINT` | App 4 | Claude Sonnet API endpoint |
| `MISTRAL_ENDPOINT` | App 4 | Mistral Large API endpoint |

### Optional (All Apps)

Application configuration variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOG_LEVEL` | INFO | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `MAX_FILE_SIZE_MB` | 10 | Maximum upload file size in MB |
| `SESSION_TIMEOUT_MINUTES` | 60 | Session timeout for App 4 |

---

## AWS Credentials Setup

### Why AWS Credentials are Required

All apps use **AWS Bedrock Titan Text Embeddings v1** to convert text into 1536-dimensional vectors. This is the foundation of the Vector Precognition algorithm.

### Option 1: Create New AWS Account (Free Tier Eligible)

**Step 1: Create AWS Account**
1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click "Create an AWS Account"
3. Follow the registration process (requires credit card, but free tier available)
4. Complete identity verification

**Step 2: Enable AWS Bedrock Access**
1. Log in to [AWS Console](https://console.aws.amazon.com)
2. Navigate to **Services** → **AWS Bedrock**
3. Click **Model access** in the left sidebar
4. Click **Manage model access**
5. Find **Titan Text Embeddings v1** and click **Request access**
6. Wait for approval (usually instant, but can take up to 24 hours)

**Step 3: Create IAM User for Programmatic Access**

1. Navigate to **Services** → **IAM** (Identity and Access Management)
2. Click **Users** → **Add users**
3. User name: `vector-precognition-user`
4. Select **Programmatic access** (Access key ID and secret)
5. Click **Next: Permissions**

**Step 4: Attach Bedrock Permissions**

Choose **Attach existing policies directly**:
- Search and select: `AmazonBedrockFullAccess`
- Or create custom policy with minimal permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1"
    }
  ]
}
```

**Step 5: Create Access Keys**

1. Click **Next: Tags** (optional, skip if not needed)
2. Click **Next: Review**
3. Click **Create user**
4. **IMPORTANT**: Copy your credentials immediately:
   - **Access key ID**: e.g., `AKIAIOSFODNN7EXAMPLE`
   - **Secret access key**: e.g., `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`
5. Download CSV file as backup
6. Store credentials securely (you won't be able to see the secret again)

**Step 6: Set Region**

AWS Bedrock Titan is available in these regions:
- `us-east-1` (N. Virginia) - **Recommended**
- `us-west-2` (Oregon)
- `eu-central-1` (Frankfurt)
- `ap-southeast-1` (Singapore)

Use `us-east-1` unless you have specific regional requirements.

### Option 2: Use Existing AWS Account

If you already have an AWS account:

1. **Check Bedrock Access**:
   ```bash
   aws bedrock list-foundation-models --region us-east-1 | grep titan-embed
   ```

2. **If not available**, follow Step 2 above to request model access

3. **Create new IAM user** or use existing credentials with Bedrock permissions

4. **Get credentials** from IAM console or AWS CLI:
   ```bash
   aws iam create-access-key --user-name your-username
   ```

### Option 3: Use AWS CLI Profile (Development Only)

For local development, you can use AWS CLI profiles instead of .env file:

1. **Install AWS CLI**:
   ```bash
   # macOS
   brew install awscli

   # Linux
   sudo apt-get install awscli

   # Or pip
   pip install awscli
   ```

2. **Configure AWS CLI**:
   ```bash
   aws configure
   ```

   Enter:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region: `us-east-1`
   - Default output: `json`

3. **Verify Configuration**:
   ```bash
   aws sts get-caller-identity
   ```

**Note**: This only works for development. Docker containers need explicit environment variables.

### Cost Estimation

**Titan Text Embeddings v1 Pricing (us-east-1):**
- $0.0001 per 1,000 tokens
- Average conversation: ~500 tokens (user + AI response)
- **Cost per conversation: ~$0.00005** (0.005 cents)

**Example costs:**
- 1,000 conversations: $0.05
- 10,000 conversations: $0.50
- 100,000 conversations: $5.00
- 1,000,000 conversations: $50.00

**Free tier:** AWS offers free tier credits for 12 months (check current offerings).

---

## LLM API Endpoints Setup

### Option 1: Use Mock Client (No API Keys Needed)

App 4 includes a **Mock Client** that simulates LLM responses without requiring any API keys.

**Usage:**
1. Start App 4: `docker-compose up app4-unified-dashboard`
2. Select **"Mock Client"** from the model dropdown
3. Chat without any API configuration

**Limitations:**
- Responses are simulated (not real AI)
- Good for testing the interface and pipeline
- Cannot evaluate real model safety

### Option 2: Deploy Your Own LLM Endpoints (Recommended)

If you have AWS Lambda functions or other API endpoints hosting LLMs:

**Requirements:**
- HTTP POST endpoint that accepts JSON: `{"message": "user input"}`
- Returns JSON: `{"reply": "AI response"}`
- HTTPS with valid SSL certificate

**Example Lambda endpoint:**
```
https://abc123xyz.execute-api.us-east-1.amazonaws.com/prod/chat
```

**Configuration:**
Add to `.env` file:
```env
GPT35_ENDPOINT=https://your-gpt35-endpoint.com/chat
CLAUDE_ENDPOINT=https://your-claude-endpoint.com/chat
```

### Option 3: Direct API Integration (Advanced)

If you want to use OpenAI, Anthropic, or Mistral APIs directly, you'll need to:

1. **Modify the API client** in `app4_unified_dashboard/core/api_client.py`
2. **Add API key environment variables**:
   ```env
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   MISTRAL_API_KEY=...
   ```
3. **Update the code** to use official SDK clients

**Example modification for OpenAI:**
```python
# In api_client.py
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

def call_gpt35(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content
```

### Option 4: Skip LLM Endpoints Entirely

Apps 1, 2, and 3 **do not require LLM endpoints**. You can use them with:

- **Manual input** (type vectors or text manually)
- **JSON/CSV upload** (analyze pre-recorded conversations)
- **Batch processing** (analyze historical data)

Only App 4 (live chat) requires LLM endpoints, and even then, you can use the mock client.

---

## Configuration Steps

### Step 1: Copy Environment Template

```bash
cd deployment
cp .env.example .env
```

### Step 2: Edit .env File

```bash
nano .env
# Or use your preferred editor: vim, code, etc.
```

### Step 3: Add AWS Credentials (Required)

```env
# AWS Bedrock Configuration (REQUIRED)
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1
```

**Replace with your actual credentials from IAM console.**

### Step 4: Add LLM Endpoints (Optional)

If you have LLM endpoints:

```env
# LLM API Endpoints (OPTIONAL - for App 4 only)
GPT35_ENDPOINT=https://your-gpt35-endpoint.execute-api.us-east-1.amazonaws.com/prod/chat
GPT4_ENDPOINT=https://your-gpt4-endpoint.execute-api.us-east-1.amazonaws.com/prod/chat
CLAUDE_ENDPOINT=https://your-claude-endpoint.execute-api.us-east-1.amazonaws.com/prod/chat
MISTRAL_ENDPOINT=https://your-mistral-endpoint.execute-api.us-east-1.amazonaws.com/prod/chat
```

If you don't have endpoints, leave these blank or commented out:

```env
# LLM API Endpoints (OPTIONAL - for App 4 only)
# GPT35_ENDPOINT=
# GPT4_ENDPOINT=
# CLAUDE_ENDPOINT=
# MISTRAL_ENDPOINT=
```

### Step 5: Configure Application Settings (Optional)

```env
# Application Settings (OPTIONAL)
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=10
SESSION_TIMEOUT_MINUTES=60
```

### Step 6: Secure Your .env File

```bash
# Set restrictive permissions
chmod 600 .env

# Verify .env is in .gitignore (it should be)
grep .env .gitignore
```

**IMPORTANT**: Never commit `.env` to version control!

---

## Verification

### Verify AWS Credentials

**Method 1: AWS CLI**
```bash
export $(cat .env | xargs)
aws sts get-caller-identity
```

**Expected output:**
```json
{
    "UserId": "AIDAI...",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/vector-precognition-user"
}
```

**Method 2: Test Bedrock Access**
```bash
aws bedrock list-foundation-models --region us-east-1 | grep titan-embed
```

**Expected output:**
```json
"modelId": "amazon.titan-embed-text-v1"
```

### Verify Docker Configuration

**Build and start containers:**
```bash
docker-compose build
docker-compose up -d
```

**Check environment variables inside container:**
```bash
docker exec -it vp-app1-guardrail-erosion env | grep AWS
```

**Expected output:**
```
AWS_ACCESS_KEY_ID=AKIAIO...
AWS_SECRET_ACCESS_KEY=wJalrX...
AWS_DEFAULT_REGION=us-east-1
```

### Test Embeddings Generation

**Quick test:**
```bash
docker exec -it vp-app1-guardrail-erosion python -c "
from shared.embeddings import get_titan_embedding
embedding = get_titan_embedding('Hello world')
print(f'Embedding dimension: {len(embedding)}')
print('AWS Bedrock connection: OK')
"
```

**Expected output:**
```
Embedding dimension: 1536
AWS Bedrock connection: OK
```

### Test Full Application

**Open each app in browser:**
1. App 1: http://localhost:8501
2. App 2: http://localhost:8502
3. App 3: http://localhost:8503
4. App 4: http://localhost:8504

**Test App 1 with manual input:**
- Click "Manual Input" tab
- Enter test vectors
- Verify no AWS errors appear
- Try text input (requires AWS credentials)

---

## Troubleshooting

### Issue: "NoCredentialsError"

**Error message:**
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Solutions:**

1. **Verify .env file exists:**
   ```bash
   ls -la deployment/.env
   ```

2. **Check .env file content:**
   ```bash
   cat deployment/.env | grep AWS
   ```

3. **Rebuild containers** (environment variables are baked in at build time):
   ```bash
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

4. **Check docker-compose.yml** environment section

### Issue: "AccessDeniedException"

**Error message:**
```
botocore.exceptions.ClientError: An error occurred (AccessDeniedException) when calling the InvokeModel operation
```

**Solutions:**

1. **Verify Bedrock model access:**
   - Log in to AWS Console
   - Navigate to AWS Bedrock
   - Check "Model access" → Ensure Titan is enabled

2. **Verify IAM permissions:**
   ```bash
   aws iam get-user-policy --user-name vector-precognition-user --policy-name BedrockAccess
   ```

3. **Check region:**
   - Titan might not be available in your region
   - Change to `us-east-1`

### Issue: "ResourceNotFoundException"

**Error message:**
```
botocore.exceptions.ClientError: An error occurred (ResourceNotFoundException)
```

**Solution:**

Wrong model ID or region. Verify:
```bash
aws bedrock list-foundation-models --region us-east-1 --query 'modelSummaries[?contains(modelId, `titan-embed`)].modelId'
```

Should return: `amazon.titan-embed-text-v1`

### Issue: LLM Endpoints Not Working (App 4)

**Error message:**
```
Connection timeout / 404 Not Found
```

**Solutions:**

1. **Use Mock Client** instead of real endpoints
2. **Verify endpoint URL** (must be HTTPS with valid SSL)
3. **Test endpoint manually:**
   ```bash
   curl -X POST https://your-endpoint.com/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello"}'
   ```

4. **Check CORS settings** if using browser-based API

### Issue: Docker Container Crashes

**Check logs:**
```bash
docker-compose logs app1-guardrail-erosion
```

**Common causes:**
- Missing models directory: Copy `models/` folder
- Invalid .env syntax: Remove spaces around `=`
- Port conflicts: Change port mapping in docker-compose.yml

---

## Security Best Practices

1. **Never commit .env to git**
   ```bash
   # Verify it's in .gitignore
   grep .env .gitignore
   ```

2. **Use IAM least privilege**
   - Only grant Bedrock permissions
   - No admin or root access

3. **Rotate credentials regularly**
   ```bash
   aws iam create-access-key --user-name vector-precognition-user
   aws iam delete-access-key --user-name vector-precognition-user --access-key-id OLD_KEY_ID
   ```

4. **Use AWS Secrets Manager for production**
   - Store credentials in Secrets Manager
   - Reference in ECS/EKS deployments
   - Automatic rotation

5. **Monitor usage**
   ```bash
   aws cloudwatch get-metric-statistics \
     --namespace AWS/Bedrock \
     --metric-name Invocations \
     --dimensions Name=ModelId,Value=amazon.titan-embed-text-v1 \
     --start-time 2024-12-01T00:00:00Z \
     --end-time 2024-12-05T23:59:59Z \
     --period 3600 \
     --statistics Sum
   ```

---

## Summary

### Minimum Required Setup (Apps 1-3)

```env
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_DEFAULT_REGION=us-east-1
```

**That's it!** All apps will work with just AWS credentials.

### Full Setup (All Apps Including Live Chat)

```env
# Required
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_DEFAULT_REGION=us-east-1

# Optional (App 4 live chat)
GPT35_ENDPOINT=https://your-endpoint.com/chat
CLAUDE_ENDPOINT=https://your-endpoint.com/chat
MISTRAL_ENDPOINT=https://your-endpoint.com/chat

# Optional (configuration)
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=10
SESSION_TIMEOUT_MINUTES=60
```

### Quick Start Commands

```bash
# 1. Create .env file
cd deployment
cp .env.example .env
nano .env  # Add AWS credentials

# 2. Build and start
docker-compose build
docker-compose up -d

# 3. Verify
curl http://localhost:8501/_stcore/health

# 4. Open in browser
open http://localhost:8501
```

---

## Getting Help

- **AWS Bedrock Documentation**: https://docs.aws.amazon.com/bedrock/
- **IAM User Guide**: https://docs.aws.amazon.com/IAM/latest/UserGuide/
- **Docker Environment Variables**: https://docs.docker.com/compose/environment-variables/

For project-specific issues, check:
- `docs/DOCKER_DEPLOYMENT_README.md` - Deployment guide
- `docs/DEVELOPER_GUIDE.md` - Development setup
- `docs/PROJECT_STATUS.md` - Current status and known issues

---

**Ready to deploy!** With your AWS credentials configured, all apps will work seamlessly.
