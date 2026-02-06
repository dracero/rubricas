import requests
import time
import subprocess
import sys
import json

def test_backend_a2a():
    print("🚀 Starting Backend Server for Verification...")
    # Start server in background
    # We use uv run for consistency with package.json
    proc = subprocess.Popen(
        ["uv", "run", "uvicorn", "server:app", "--port", "8001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Wait for server startup
        time.sleep(5)
        
        base_url = "http://localhost:8001"
        
        # Test 1: Generate Intent
        print("\n🧪 Test 1: Intent 'Quiero generar una rúbrica'")
        res = requests.post(f"{base_url}/api/chat", json={"message": "Quiero generar una rúbrica para Biología"})
        print(f"Status: {res.status_code}")
        data = res.json()
        print(f"Response Type: {data.get('type')}")
        print(f"Component: {data.get('metadata', {}).get('component')}")
        
        if data.get('metadata', {}).get('component') == 'RubricGenerator':
            print("✅ PASS: Correctly identified Generator")
        else:
            print(f"❌ FAIL: Expected RubricGenerator, got {data.get('metadata', {}).get('component')}")

        # Test 2: Evaluate Intent
        print("\n🧪 Test 2: Intent 'Necesito evaluar unos apuntes'")
        res = requests.post(f"{base_url}/api/chat", json={"message": "Necesito evaluar unos apuntes de historia"})
        data = res.json()
        print(f"Response Type: {data.get('type')}")
        print(f"Component: {data.get('metadata', {}).get('component')}")
        
        if data.get('metadata', {}).get('component') == 'RubricEvaluator':
            print("✅ PASS: Correctly identified Evaluator")
        else:
            print(f"❌ FAIL: Expected RubricEvaluator, got {data.get('metadata', {}).get('component')}")

    except Exception as e:
        print(f"❌ Error during test: {e}")
    finally:
        print("\n🛑 Stopping Server...")
        proc.terminate()
        # Print server output if any error
        outs, errs = proc.communicate(timeout=2)
        if errs:
            print(f"Server Errors: {errs.decode()}")

if __name__ == "__main__":
    test_backend_a2a()
