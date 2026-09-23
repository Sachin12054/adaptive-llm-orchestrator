import os
import json
import requests

BASE_URL = "http://127.0.0.1:8000"

def run_test_case(test_num, prompt):
    print(f"\n==========================================")
    print(f"TEST CASE {test_num}: '{prompt}'")
    print(f"==========================================")
    
    url = f"{BASE_URL}/api/orchestrate/stream"
    payload = {
        "prompt": prompt,
        "execution_mode": "online",
        "run_id": f"test_case_{test_num}"
    }
    
    res = requests.post(url, json=payload, stream=True, timeout=90.0)
    buffer = ""
    events = []
    
    for chunk in res.iter_content(chunk_size=1, decode_unicode=True):
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
                            except Exception:
                                pass

    selected_model_decision = None
    executed_model_inference = None
    final_response_evt = None
    fallback_occurred = False

    for evt in events:
        stage = evt.get("stage")
        status = evt.get("status")
        msg = evt.get("message", "")
        meta = evt.get("metadata", {})

        if stage == "adaptive_decision" and status == "completed":
            selected_model_decision = meta.get("selected_model") or evt.get("decision", {}).get("selected_model")
            print(f" [DECISION STAGE] Baseline Policy Selected: '{selected_model_decision}'")

        if stage in ["online_inference", "local_inference"] and status == "running":
            executed_model_inference = meta.get("model")
            print(f" [INFERENCE DISPATCH] Executing Model: '{executed_model_inference}'")

        if status == "fallback":
            fallback_occurred = True
            print(f" [FALLBACK WARNING] {msg}")

        if stage in ["online_inference", "local_inference"] and status == "completed":
            completed_model = meta.get("model") or meta.get("model_id")
            provider = meta.get("provider")
            latency = meta.get("latency_ms")
            print(f" [INFERENCE RESULT] Provider: '{provider}' | Model: '{completed_model}' | Latency: {latency}ms")

        if stage == "final_response" and status == "completed":
            final_response_evt = evt

    final_model = None
    if final_response_evt:
        final_model = final_response_evt.get("metadata", {}).get("selected_model")
        print(f" [FINAL RESPONSE TELEMETRY] Reported Selected Model: '{final_model}'")

    print("\n --- VERIFICATION SUMMARY ---")
    print(f" Decision Model: {selected_model_decision}")
    print(f" Executed Model: {executed_model_inference}")
    print(f" Final Model:    {final_model}")
    print(f" Fallback Flag:  {fallback_occurred}")

    is_consistent = (selected_model_decision == executed_model_inference == final_model) or fallback_occurred
    print(f" MATCH STATUS:   {'PASSED (Consistent)' if is_consistent else 'FAILED (Mismatch)'}")

def main():
    test_prompts = [
        (1, "What is the capital of France?"),
        (2, "Write a Python program to detect whether a string is a palindrome. Include time and space complexity."),
        (3, "Explain the difference between supervised and reinforcement learning."),
        (4, "Design a Python FastAPI endpoint that accepts JSON and stores it in PostgreSQL.")
    ]
    
    for num, prompt in test_prompts:
        run_test_case(num, prompt)

if __name__ == "__main__":
    main()
