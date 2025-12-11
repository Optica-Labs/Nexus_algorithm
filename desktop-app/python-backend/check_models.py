#!/usr/bin/env python3
"""
Quick script to check which OpenAI models your API key has access to.
"""

import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI package not installed. Run: pip install openai")
    sys.exit(1)

def check_model_access(api_key: str):
    """Check which models are available with the given API key."""

    client = OpenAI(api_key=api_key)

    print("=" * 60)
    print("OpenAI Model Access Checker")
    print("=" * 60)
    print()

    # Models to check
    models_to_check = [
        ("GPT-3.5 Turbo", "gpt-3.5-turbo"),
        ("GPT-4o Mini", "gpt-4o-mini"),
        ("GPT-4o", "gpt-4o"),
        ("GPT-4 Turbo", "gpt-4-turbo"),
        ("GPT-4", "gpt-4"),
    ]

    print("Checking model access...")
    print("-" * 60)

    available_models = []

    for name, model_id in models_to_check:
        try:
            # Try a minimal completion request
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            print(f"✅ {name:20s} ({model_id:20s}) - AVAILABLE")
            available_models.append((name, model_id))
        except Exception as e:
            error_msg = str(e)
            if "does not exist" in error_msg or "model_not_found" in error_msg:
                print(f"❌ {name:20s} ({model_id:20s}) - NO ACCESS")
            elif "rate_limit" in error_msg:
                print(f"⚠️  {name:20s} ({model_id:20s}) - RATE LIMITED (probably available)")
                available_models.append((name, model_id))
            else:
                print(f"⚠️  {name:20s} ({model_id:20s}) - ERROR: {error_msg[:50]}")

    print("-" * 60)
    print()

    if available_models:
        print(f"✅ You have access to {len(available_models)} model(s):")
        for name, model_id in available_models:
            print(f"   • {name} ({model_id})")
        print()
        print(f"📌 Recommended: Use '{available_models[0][0]}' for testing")
    else:
        print("❌ No models available. Check your API key:")
        print("   • Verify key is valid at https://platform.openai.com/api-keys")
        print("   • Check account has credits at https://platform.openai.com/usage")

    print()
    print("=" * 60)


if __name__ == "__main__":
    # Get API key from argument or environment
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Usage: python check_models.py <your-api-key>")
            print("   or: export OPENAI_API_KEY=<your-key>")
            print("       python check_models.py")
            sys.exit(1)

    if not api_key.startswith("sk-"):
        print("❌ Invalid API key format (should start with 'sk-')")
        sys.exit(1)

    check_model_access(api_key)
