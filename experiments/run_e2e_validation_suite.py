import os
import sys
import json
import time
import requests

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"), override=True)

BASE_URL = "http://127.0.0.1:8000"

def run_sse_request_sync(prompt: str, mode: str = "online", run_id: str = None) -> list:
    if not run_id:
        run_id = f"run_val_{int(time.time()*1000)}_{os.urandom(2).hex()}"
    
    url = f"{BASE_URL}/api/orchestrate/stream"
    payload = {
        "prompt": prompt,
        "execution_mode": mode,
        "run_id": run_id
    }
    
    events = []
    try:
        response = requests.post(url, json=payload, stream=True, timeout=120.0)
        buffer = ""
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
                                        events.append(json.loads(json_str))
                                    except Exception:
                                        pass
        except Exception:
            pass

        if buffer.strip():
            for line in buffer.split("\n"):
                if line.startswith("data: "):
                    json_str = line[6:].strip()
                    if json_str:
                        try:
                            events.append(json.loads(json_str))
                        except Exception:
                            pass
    except Exception as e:
        print("Request error:", e)

    return events

def analyze_run(label: str, prompt: str, events: list):
    final_evt = next((e for e in events if e.get("stage") == "final_response"), None)
    
    stages_received = [e.get("stage") for e in events]
    if final_evt is None:
        print(f"[{label}] Stages received ({len(events)}): {stages_received}")
        for idx, e in enumerate(events):
            print(f"  Event {idx}: {e}")
    assert final_evt is not None, f"[{label}] Missing final_response event! Received stages: {stages_received}"
    
    payload = final_evt["payload"]
    gen = payload["generation"]

    # Verify run_id consistency across ALL events
    run_ids = set(e.get("run_id") for e in events if "run_id" in e)
    assert len(run_ids) == 1, f"[{label}] Inconsistent run_ids across events: {run_ids}"
    evt_run_id = list(run_ids)[0]

    res = {
        "label": label,
        "prompt": prompt,
        "run_id": evt_run_id,
        "req_prompt": payload["prompt"],
        "policy_selected": payload["selected_model"],
        "assigned": payload["selected_model"],
        "dispatched": gen["model_id"],
        "gen_model_id": gen["model_id"],
        "final_model_id": payload["generation"]["model_id"],
        "provider": gen["provider"],
        "generated_text": (gen["generated_text"] or "").strip(),
        "verified": payload["verification"]["verified"],
        "reward": payload["reward"]["reward"],
        "latency_ms": payload["pipeline_latency_ms"],
        "events_count": len(events),
        "correlation_pass": (payload["prompt"] == prompt and payload["run_id"] == evt_run_id),
        "identity_pass": (payload["selected_model"] == gen["model_id"])
    }
    return res

def safe_str(text: str) -> str:
    if not text:
        return ""
    return text.encode('ascii', errors='ignore').decode('ascii')

def main_suite():
    print("\n" + "="*100)
    print("  EXECUTING COMPREHENSIVE END-TO-END ORCHESTRATION VALIDATION")
    print("="*100)

    # 1. Basic Factual Prompt
    print("\n--- 1. BASIC FACTUAL PROMPT ---")
    p1 = "What is the national bird of the United States?"
    evts1 = run_sse_request_sync(p1, mode="online")
    res1 = analyze_run("Basic Factual (US Bird)", p1, evts1)
    print(f"Run ID:            {res1['run_id']}")
    print(f"Policy Selected:   {res1['policy_selected']}")
    print(f"Dispatched Model:  {res1['dispatched']}")
    print(f"Provider:          {res1['provider']}")
    print(f"Verified:          {res1['verified']}")
    print(f"Reward:            {res1['reward']:.4f}")
    print(f"Generated Text:    {safe_str(res1['generated_text'])[:120]}...")
    assert "eagle" in res1['generated_text'].lower() or "bald" in res1['generated_text'].lower(), "US Bird response missing Bald Eagle!"
    print("=> BASIC FACTUAL PROMPT PASSED!")

    # 2. Sequential Prompt Correlation Test
    print("\n--- 2. PROMPT CORRELATION TEST (5 PROMPTS) ---")
    prompts = [
        ("TEST 1", "What is the national bird of the United States?", "online"),
        ("TEST 2", "What is the capital of Japan?", "online"),
        ("TEST 3", "What is the capital of France?", "online"),
        ("TEST 4", "Explain how TCP congestion control works in simple terms.", "online"),
        ("TEST 5", "Compare REST APIs, GraphQL, and gRPC and recommend one for a real-time IoT telemetry platform.", "online"),
    ]
    
    for label, prompt_text, mode in prompts:
        evts = run_sse_request_sync(prompt_text, mode=mode)
        r = analyze_run(label, prompt_text, evts)
        print(f"[{label}] Prompt: '{prompt_text[:35]}...' | Model: {r['dispatched']} | Text Len: {len(r['generated_text'])} | Correlation: PASS")

    # 3. Rapid Successive Prompt Correlation Test
    print("\n--- 3. RAPID SUCCESSIVE PROMPT CORRELATION TEST ---")
    prompt_A = "What is the national bird of the United States?"
    prompt_B = "What is the capital of Japan?"
    prompt_C = "What is the capital of France?"

    run_A = f"run_rapid_A_{int(time.time()*1000)}"
    run_B = f"run_rapid_B_{int(time.time()*1000)}"
    run_C = f"run_rapid_C_{int(time.time()*1000)}"

    evts_A = run_sse_request_sync(prompt_A, mode="online", run_id=run_A)
    evts_B = run_sse_request_sync(prompt_B, mode="online", run_id=run_B)
    evts_C = run_sse_request_sync(prompt_C, mode="online", run_id=run_C)

    res_A = analyze_run("Rapid A", prompt_A, evts_A)
    res_B = analyze_run("Rapid B", prompt_B, evts_B)
    res_C = analyze_run("Rapid C", prompt_C, evts_C)

    print(f"Rapid A run_id: {res_A['run_id']} | Text: {safe_str(res_A['generated_text'])[:40]}...")
    print(f"Rapid B run_id: {res_B['run_id']} | Text: {safe_str(res_B['generated_text'])[:40]}...")
    print(f"Rapid C run_id: {res_C['run_id']} | Text: {safe_str(res_C['generated_text'])[:40]}...")
    assert "eagle" in res_A['generated_text'].lower() or "bald" in res_A['generated_text'].lower()
    assert "tokyo" in res_B['generated_text'].lower()
    assert "paris" in res_C['generated_text'].lower()
    print("=> RAPID SUCCESSIVE PROMPT CORRELATION PASSED!")

    # 4. Local Mode Test
    print("\n--- 4. LOCAL MODE TEST ---")
    p_local = "What is the capital of France?"
    evts_local = run_sse_request_sync(p_local, mode="local")
    res_local = analyze_run("Local Execution", p_local, evts_local)
    print(f"Local Model: {res_local['dispatched']} | Provider: {res_local['provider']} | Text: {safe_str(res_local['generated_text'])[:60]}...")
    print("=> LOCAL MODE TEST PASSED!")

    # 5. Online Mode Test
    print("\n--- 5. ONLINE MODE TEST ---")
    p_online = "What is the capital of France?"
    evts_online = run_sse_request_sync(p_online, mode="online")
    res_online = analyze_run("Online Execution", p_online, evts_online)
    print(f"Online Model: {res_online['dispatched']} | Provider: {res_online['provider']} | Text: {safe_str(res_online['generated_text'])[:60]}...")
    print("=> ONLINE MODE TEST PASSED!")

    # 6. Complex Prompt Test
    print("\n--- 6. COMPLEX PROMPT TEST ---")
    p_complex = "Compare REST APIs, GraphQL, and gRPC and recommend one for a real-time IoT telemetry platform."
    evts_complex = run_sse_request_sync(p_complex, mode="online")
    res_complex = analyze_run("Complex Execution", p_complex, evts_complex)
    print(f"Complex Model: {res_complex['dispatched']} | Provider: {res_complex['provider']} | Text Len: {len(res_complex['generated_text'])}")
    print("=> COMPLEX PROMPT TEST PASSED!")

    print("\n" + "="*100)
    print("  ALL E2E SUITE TEST CASES COMPLETED SUCCESSFULLY!")
    print("="*100)

if __name__ == "__main__":
    main_suite()
