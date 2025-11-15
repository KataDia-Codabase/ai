#!/usr/bin/env python3
"""Simple health check script for the service.

Run this with the project's venv python:
& '...ai-ml-service\venv\Scripts\python.exe' scripts\check_health.py
"""
import time
import sys
try:
    import requests
except Exception as e:
    print("requests not available - please install requests in the venv:")
    print("pip install requests")
    sys.exit(2)

URL = 'http://127.0.0.1:8000/api/v1/health'

print(f"Checking {URL} up to 15 times (1s interval)...")
for attempt in range(1, 16):
    try:
        r = requests.get(URL, timeout=3)
        print(f"Attempt {attempt}: {r.status_code}")
        print(r.text)
        if r.status_code == 200:
            print("Service healthy")
            sys.exit(0)
        else:
            time.sleep(1)
    except Exception as exc:
        print(f"Attempt {attempt} failed: {exc}")
        time.sleep(1)

print("Service did not respond successfully after retries.")
sys.exit(1)
