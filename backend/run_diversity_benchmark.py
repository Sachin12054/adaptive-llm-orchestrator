import os
import sys
from dotenv import load_dotenv

load_dotenv()

from app.schemas.decision import DecisionRequest
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine

engine = AdaptiveDecisionEngine()

benchmark_prompts = [
    # General QA / Factual (Low/Medium Complexity)
    ("General QA 1", "What is the capital of France?", "online"),
    ("General QA 2", "Who wrote the play Romeo and Juliet?", "online"),
    ("General QA 3", "What is the boiling point of water in Celsius?", "online"),
    ("General QA 4", "Which planet is known as the Red Planet?", "online"),
    ("General QA 5", "What is the chemical symbol for gold?", "online"),

    # Translation (Medium Complexity)
    ("Translation 1", "Translate the following technical paragraph into Spanish while preserving terminology.", "online"),
    ("Translation 2", "Translate this user manual from English to French.", "online"),
    ("Translation 3", "Translate this news article into German accurately.", "online"),
    ("Translation 4", "Translate these customer support messages into Japanese.", "online"),

    # Mathematics (Medium/High Complexity)
    ("Mathematics 1", "Solve this mathematical proof and explain every step rigorously.", "online"),
    ("Mathematics 2", "Calculate the integral of x^2 * sin(x) dx using integration by parts.", "online"),
    ("Mathematics 3", "Find the eigenvalues and eigenvectors of a 3x3 matrix.", "online"),
    ("Mathematics 4", "Prove that the sum of the first n odd numbers is n^2.", "online"),
    ("Mathematics 5", "Solve the differential equation dy/dx + 2y = e^(-x).", "online"),

    # Reasoning & System Design (High Complexity)
    ("Reasoning 1", "Prove the time complexity of this recursive algorithm using a recurrence relation.", "online"),
    ("Reasoning 2", "Design a distributed system for processing 100 million events per day with fault tolerance.", "online"),
    ("Reasoning 3", "Analyze the Byzantine Fault Tolerance consensus protocol in distributed networks.", "online"),
    ("Reasoning 4", "Prove by induction that 2^n > n^2 for n >= 5.", "online"),
    ("Reasoning 5", "Design a zero-downtime database migration strategy for high-traffic microservices.", "online"),

    # Coding (Medium Complexity)
    ("Coding 1", "Write a Python FastAPI endpoint that validates JSON and stores it in PostgreSQL.", "online"),
    ("Coding 2", "Implement a thread-safe LRU cache class in Python with O(1) ops.", "online"),
    ("Coding 3", "Write a Rust program to parse JSON streaming buffers efficiently.", "online"),
    ("Coding 4", "Implement a binary search tree traversal in Python with recursive and iterative methods.", "online"),

    # Summarization (Low/Medium Complexity)
    ("Summarization 1", "Summarize this long document into five concise bullet points.", "online"),
    ("Summarization 2", "Provide an executive summary of this 20-page quarterly financial report.", "online"),
    ("Summarization 3", "Summarize key takeaways from this research paper on Transformer attention.", "online"),

    # Local Ollama Execution Prompts (Local Mode)
    ("Local Coding", "Write a C function to reverse a linked list.", "local"),
    ("Local Reasoning", "Explain the difference between process and thread.", "local"),
    ("Local QA", "What is the function of a compiler?", "local")
]

print(f"{'#':<3} | {'TASK NAME':<28} | {'MODE':<6} | {'SELECTED MODEL':<35} | {'PROVIDER':<20} | {'SCORE':<8} | {'RUNNER UP':<35} | {'RUNNER UP SCORE':<8}")
print("=" * 150)

provider_counts = {}
model_counts = {}

for idx, (name, prompt_text, mode) in enumerate(benchmark_prompts, 1):
    req = DecisionRequest(text=prompt_text, execution_mode=mode)
    res = engine.decide(req)

    selected_m = res.selected_model or "None"
    top_score = res.decision_score
    top_provider = res.candidates[0].provider if res.candidates else "Unknown"

    runner_up = res.candidates[1].model_id if len(res.candidates) > 1 else "None"
    runner_up_score = res.candidates[1].candidate_score if len(res.candidates) > 1 else 0.0

    print(f"{idx:<3} | {name:<28} | {mode:<6} | {selected_m:<35} | {top_provider:<20} | {top_score:<8.4f} | {runner_up:<35} | {runner_up_score:<8.4f}")

    provider_counts[top_provider] = provider_counts.get(top_provider, 0) + 1
    model_counts[selected_m] = model_counts.get(selected_m, 0) + 1

print("\n" + "=" * 80)
print("SELECTION DISTRIBUTION SUMMARY:")
print("=" * 80)
total_tasks = len(benchmark_prompts)
for prov, count in provider_counts.items():
    pct = (count / total_tasks) * 100
    print(f"  Provider '{prov:<22}': {count:>2} tasks ({pct:>5.1f}%)")

print("-" * 80)
for model, count in model_counts.items():
    pct = (count / total_tasks) * 100
    print(f"  Model    '{model:<35}': {count:>2} tasks ({pct:>5.1f}%)")
