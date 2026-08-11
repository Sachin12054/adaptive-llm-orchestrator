import sys
import os
import json
from fastapi.testclient import TestClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from app.main import app

def main():
    print("=" * 80)
    print("FASTAPI STARTUP & SSE ROUTE TEST")
    print("=" * 80)

    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    print(f"Health Check Status: {res_health.status_code} | Payload: {res_health.json()}")

    # 2. SSE Stream Test
    payload = {"prompt": "What is Python?"}
    res_stream = client.post("/api/orchestrate/stream", json=payload)
    print(f"SSE Stream Endpoint Status: {res_stream.status_code}")
    print(f"Content-Type Header: {res_stream.headers.get('content-type')}")

    lines = res_stream.text.split("\n\n")
    valid_events = []
    has_intent_running = False
    has_intent_completed = False
    has_final_response = False
    has_font_sans_artifact = False

    for line in lines:
        if line.startswith("data: "):
            json_str = line[6:]
            if "font-sans" in line:
                has_font_sans_artifact = True
            try:
                data = json.loads(json_str)
                valid_events.append(data)
                stage = data.get("stage")
                status = data.get("status")
                if stage == "intent_analysis" and status == "running":
                    has_intent_running = True
                if stage == "intent_analysis" and status == "completed":
                    has_intent_completed = True
                if stage == "final_response" and status == "completed":
                    has_final_response = True
            except Exception as e:
                print(f"FAIL: JSON parse error on SSE line: '{line}': {e}")

    print(f"Total Valid SSE Events Delivered: {len(valid_events)}")
    print(f"Received 'intent_analysis/running':   {'YES (PASS)' if has_intent_running else 'NO (FAIL)'}")
    print(f"Received 'intent_analysis/completed': {'YES (PASS)' if has_intent_completed else 'NO (FAIL)'}")
    print(f"Received 'final_response':           {'YES (PASS)' if has_final_response else 'NO (FAIL)'}")
    print(f"Contains 'font-sans' string artifact: {'YES (FAIL)' if has_font_sans_artifact else 'NO (PASS)'}")

if __name__ == "__main__":
    main()
