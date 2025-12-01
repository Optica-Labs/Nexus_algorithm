

# Test all 3 models (default)
python src/precog_validation_test_api.py

# Test only GPT-3.5
python src/precog_validation_test_api.py --model gpt-3.5

# Test only Mistral
python src/precog_validation_test_api.py --model mistral

# Test only Claude
python src/precog_validation_test_api.py --model claude

# Test specific combinations
python src/precog_validation_test_api.py --model mistral gpt-3.5
python src/precog_validation_test_api.py --model claude gpt-3.5




# Quick Command Reference - Multi-Model Testing

## Run All 3 Models
```bash
python src/precog_validation_test_api.py
# OR
python src/precog_validation_test_api.py --model all
```

## Run Specific Model

### Test Only Mistral Large
```bash
python src/precog_validation_test_api.py --model mistral
```

### Test Only Claude Sonnet 4.5
```bash
python src/precog_validation_test_api.py --model claude
```

### Test Only GPT-OSS 120B
```bash
python src/precog_validation_test_api.py --model gpt-oss
```

## Run Multiple Specific Models
```bash
# Test Mistral and Claude only
python src/precog_validation_test_api.py --model mistral claude

# Test Claude and GPT-OSS only
python src/precog_validation_test_api.py --model claude gpt-oss
```

## Get Help
```bash
python src/precog_validation_test_api.py --help
```

## Test Individual Endpoints (Manual Check)

### Mistral Large
```bash
curl -X POST https://q1gqmwwmsf.execute-api.us-east-1.amazonaws.com/prod/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

### Claude Sonnet 4.5
```bash
curl -X POST https://6z5nnwuyyj.execute-api.us-east-1.amazonaws.com/prod/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

### GPT-OSS 120B
```bash
curl -X POST https://j0ja8icjc0.execute-api.us-east-1.amazonaws.com/prod/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

## Expected Output Directories

After running the test, you'll find results in:
- `output/mistral/` - Mistral Large results
- `output/claude/` - Claude Sonnet 4.5 results
- `output/gpt-oss/` - GPT-OSS 120B results

Each directory contains:
- 4 CSV files (metrics)
- 4 PNG files (dynamics plots)
- 1 summary robustness plot

## Estimated Runtime
- **Per Model**: 5-10 minutes
- **Total**: 15-30 minutes for all 3 models
- **Total API Calls**: 90 (30 per model)

## What Gets Tested
Each model is tested with:
- T1: Jailbreak Spike (6 adversarial prompts)
- T2: Robust Deflect (5 direct attacks)
- T3: Fragile Drift (5 benign prompts)
- T4: Contextual Erosion (14 gradual manipulation prompts)
