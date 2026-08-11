import os

backend_dir = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator\backend"

matches = []
for root, _, files in os.walk(backend_dir):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8", errors="ignore") as fc:
                lines = fc.readlines()
                for line_num, line in enumerate(lines, 1):
                    if "qwen-coder-3b" in line or "generate_response" in line or "execute_local_model" in line or "for " in line and "candidate" in line:
                        matches.append((os.path.relpath(path, backend_dir), line_num, line.strip()))

print(f"Total occurrences found: {len(matches)}")
for rel_path, line_num, snippet in matches[:50]:
    print(f"{rel_path}:L{line_num} | {snippet}")
