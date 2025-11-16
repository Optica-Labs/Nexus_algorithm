#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick API Connection Test

Tests basic connectivity to your API endpoint before running the full stress test.
"""

import requests
import json

# Your API endpoint
API_ENDPOINT = "https://uvnjpj1pv0.execute-api.eu-west-1.amazonaws.com/prod/chat"

def test_api_connection():
    """Test a simple API call to verify connectivity."""
    
    print("="*70)
    print(" Testing API Connection")
    print("="*70)
    print(f"\nEndpoint: {API_ENDPOINT}\n")
    
    # Test message
    test_message = "Hello, this is a test message."
    
    payload = {
        "message": test_message
    }
    
    print(f"Sending test message: '{test_message}'")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")
    
    try:
        response = requests.post(
            API_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Response Status Code: {response.status_code}")
        
        response.raise_for_status()
        
        response_data = response.json()
        
        print(f"\nResponse JSON:")
        print(json.dumps(response_data, indent=2))
        
        # Extract key fields
        if isinstance(response_data, dict):
            session_id = response_data.get('session_id', 'N/A')
            message = response_data.get('message', 'N/A')
            timestamp = response_data.get('timestamp', 'N/A')
            
            print(f"\n{'='*70}")
            print(" ✓ API CONNECTION SUCCESSFUL")
            print(f"{'='*70}")
            print(f"Session ID: {session_id}")
            print(f"Timestamp:  {timestamp}")
            print(f"Message:    {message[:100]}...")
            print(f"{'='*70}\n")
            
            return True
        else:
            print(f"\n✗ Unexpected response format: {type(response_data)}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\n{'='*70}")
        print(f" ✗ API CONNECTION FAILED")
        print(f"{'='*70}")
        print(f"Error: {e}")
        print(f"{'='*70}\n")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_api_connection()
    
    if success:
        print("✓ Your API endpoint is working correctly!")
        print("\nNext step: Run the full stress test:")
        print("  python src/api_validation_test.py")
    else:
        print("✗ API connection failed. Please check:")
        print("  1. Network connectivity")
        print("  2. API endpoint URL")
        print("  3. API service is running")
