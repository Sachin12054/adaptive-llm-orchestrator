import os
import re

frontend_src = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator\frontend\src"

target_terms = [
    "models[0]",
    "candidates[0]",
    "localModels[0]",
    "onlineModels[0]",
    "DEFAULT_LOCAL_MODELS",
    "DEFAULT_ONLINE_MODELS",
    "gemma-3-4b",
    "qwen-coder-3b",
    "deepseek-r1-7b",
    "gemini-2.5-flash",
    "mistral-small-latest",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-3.3-70b-instruct"
]

matches = []

for root, _, files in os.walk(frontend_src):
    for f in files:
        if f.endswith(".jsx") or f.endswith(".js"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as file_content:
                lines = file_content.readlines()
                for line_num, line in enumerate(lines, 1):
                    for term in target_terms:
                        if term in line:
                            rel_path = os.path.relpath(path, frontend_src)
                            matches.append((rel_path, line_num, term, line.strip()))

print("\n==================================================")
print("  FRONTEND HARDCODED MODEL AUDIT RESULTS")
print("==================================================")
print(f"Total occurrences found: {len(matches)}\n")
for rel_path, line_num, term, snippet in matches:
    print(f"File: {rel_path}:L{line_num} | Term: '{term}'\n  Snippet: {snippet}\n")
