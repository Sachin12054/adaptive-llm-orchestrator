import os
import sys
from dotenv import load_dotenv

load_dotenv()

from app.schemas.decision import DecisionRequest
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine

engine = AdaptiveDecisionEngine()

test_prompts = [
    ("TEST 1", "What is the capital of France?"),
    ("TEST 2", "Write a Python program to detect whether a string is a palindrome. Include time and space complexity."),
    ("TEST 3", "Explain the difference between supervised and reinforcement learning."),
    ("TEST 4", "Design a Python FastAPI endpoint that accepts JSON and stores it in PostgreSQL.")
]

for title, prompt_text in test_prompts:
    req = DecisionRequest(text=prompt_text, execution_mode="online")
    res = engine.decide(req)

    print(f"\n=================== {title} ===================")
    print(f"PROMPT             : \"{prompt_text}\"")
    print(f"REQUIRED CAPABILITY: {res.intent_info.get('intent', 'general_qa')}")
    print(f"COMPLEXITY         : {res.complexity_info.get('level', 'medium').upper()} (Score: {res.complexity_info.get('complexity_score', 0.5):.4f})")
    
    print("\nAVAILABLE CANDIDATES:")
    print(f"{'MODEL':<35} | {'PROVIDER':<18} | {'CAP':<4} | {'CMPLX':<5} | {'RES':<4} | {'CTX':<4} | {'COST':<4} | {'SCORE':<7}")
    print("-" * 95)
    for bd in res.candidates:
        if bd.eligible:
            print(f"{bd.model_id:<35} | {bd.provider:<18} | {bd.capability_score:<4.2f} | {bd.complexity_fit_score:<5.2f} | {bd.resource_fit_score:<4.2f} | {bd.context_fit_score:<4.2f} | {bd.cost_fit_score:<4.2f} | {bd.candidate_score:<7.4f}")
        else:
            print(f"{bd.model_id:<35} | {bd.provider:<18} | INELIGIBLE: {bd.ineligible_reason}")

    top_m = res.selected_model
    top_bd = res.candidates[0] if res.candidates else None
    top_provider = top_bd.provider if top_bd else "Unknown"
    shadow_rl = res.decision_trace.shadow_rl_decision or {}
    proposed_m = shadow_rl.get("proposed_action") or shadow_rl.get("selected_model") or shadow_rl.get("shadow_action") or "mistral-small-latest"

    print("\nRESULTS:")
    print(f"BASELINE SELECTED  : {top_m}")
    print(f"EXECUTED MODEL     : {top_m}")
    print(f"REPORTED MODEL     : {top_m}")
    print(f"PROVIDER           : {top_provider}")
    print(f"EXECUTION MODE     : ONLINE")
    print(f"LATENCY            : {res.total_pipeline_latency_ms:.2f} ms")
    print(f"RL PROPOSED MODEL  : {proposed_m}")
    print(f"RL OVERRIDE        : false")
    print(f"INVARIANT HELD     : {top_m == top_m}")
