import os
import sys
import json
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_palindrome():
    prompt = "Write a Python program to detect whether a string is a palindrome. Include time and space complexity."
    url = f"{BASE_URL}/api/orchestrate/stream"
    payload = {
        "prompt": prompt,
        "execution_mode": "online",
        "run_id": "test_palindrome_1"
    }
    
    print(f"Sending prompt: '{prompt}'")
    response = requests.post(url, json=payload, stream=True, timeout=60.0)
    
    buffer = ""
    events = []
    try:
        for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
            if chunk:
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    for line in block.split("\n"):
                        if line.startswith("data: "):
                            json_str = line[6:].strip()
                            if json_str:
                                try:
                                    evt = json.loads(json_str)
                                    events.append(evt)
                                    print(f"[SSE EVENT] stage={evt.get('stage')} | status={evt.get('status')} | msg={evt.get('message')}")
                                except Exception:
                                    pass
    except Exception as e:
        print("Stream ended:", e)
        
    print("\nTotal events received:", len(events))

if __name__ == "__main__":
    test_palindrome()
