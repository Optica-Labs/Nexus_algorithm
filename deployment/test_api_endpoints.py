#!/usr/bin/env python3
"""
Test API Endpoints - Validate AWS Lambda endpoints connectivity.

This script tests all 4 LLM endpoints to ensure they are accessible
and responding correctly.
"""

import requests
import json
import time
from typing import Dict, Tuple

# AWS Lambda Endpoints
ENDPOINTS = {
    "gpt-3.5": "https://kv854u79y7.execute-api.us-east-1.amazonaws.com/prod/chat",
    "gpt-4": "https://1d4qnutnqc.execute-api.us-east-1.amazonaws.com/prod/chat",
    "claude": "https://6z5nnwuyyj.execute-api.us-east-1.amazonaws.com/prod/chat",
    "mistral": "https://q1gqmwwmsf.execute-api.us-east-1.amazonaws.com/prod/chat"
}

# Test conversation
TEST_CONVERSATION = [
    {
        "role": "user",
        "content": "Hello! This is a test message. Please respond briefly."
    }
]

def test_endpoint(name: str, url: str) -> Tuple[bool, str, float]:
    """
    Test a single API endpoint.

    Args:
        name: Model name
        url: API endpoint URL

    Returns:
        Tuple of (success, response/error, response_time)
    """
    print(f"Testing {name}...")

    payload = {
        "conversation": TEST_CONVERSATION,
        "temperature": 0.7,
        "max_tokens": 100
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        start_time = time.time()
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response_time = time.time() - start_time

        response.raise_for_status()
        data = response.json()

        # Try to extract the response
        if 'response' in data:
            model_response = data['response']
        elif 'message' in data:
            model_response = data['message']
        elif 'body' in data:
            body = json.loads(data['body']) if isinstance(data['body'], str) else data['body']
            model_response = body.get('response', body.get('message', str(body)))
        else:
            model_response = str(data)

        return True, model_response, response_time

    except requests.exceptions.Timeout:
        return False, "Request timed out after 30 seconds", 0.0

    except requests.exceptions.ConnectionError as e:
        return False, f"Connection error: {str(e)}", 0.0

    except requests.exceptions.HTTPError as e:
        return False, f"HTTP error {response.status_code}: {response.text}", 0.0

    except Exception as e:
        return False, f"Unexpected error: {str(e)}", 0.0


def main():
    """Run endpoint tests."""
    print("=" * 60)
    print("AWS Lambda API Endpoint Test")
    print("=" * 60)
    print()

    results = {}

    for name, url in ENDPOINTS.items():
        success, response, response_time = test_endpoint(name, url)

        results[name] = {
            'success': success,
            'response': response,
            'response_time': response_time
        }

        if success:
            print(f"✅ {name}: SUCCESS ({response_time:.2f}s)")
            print(f"   Response: {response[:100]}...")
        else:
            print(f"❌ {name}: FAILED")
            print(f"   Error: {response}")

        print()
        time.sleep(1)  # Rate limiting

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for r in results.values() if r['success'])
    failed = total - passed

    print(f"Total endpoints: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print()

    if passed == total:
        print("🎉 All endpoints are working!")
        return 0
    elif passed > 0:
        print("⚠️  Some endpoints are working, but not all.")
        print()
        print("Working:")
        for name, result in results.items():
            if result['success']:
                print(f"  - {name}")

        print()
        print("Failed:")
        for name, result in results.items():
            if not result['success']:
                print(f"  - {name}: {result['response']}")

        return 1
    else:
        print("❌ No endpoints are working. Check your network and AWS configuration.")
        return 2


if __name__ == "__main__":
    exit(main())
