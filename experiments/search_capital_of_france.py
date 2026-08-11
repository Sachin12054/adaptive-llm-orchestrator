import os

project_root = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator"

matches = []
for root, _, files in os.walk(project_root):
    if "node_modules" in root or ".git" in root or "dist" in root:
        continue
    for f in files:
        if f.endswith(".py") or f.endswith(".json") or f.endswith(".jsx") or f.endswith(".js") or f.endswith(".ts") or f.endswith(".tsx") or f.endswith(".md"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8", errors="ignore") as fc:
                for line_num, line in enumerate(fc.readlines(), 1):
                    if "france" in line.lower() or "capital of france" in line.lower():
                        matches.append((os.path.relpath(path, project_root), line_num, line.strip()))

print(f"Total occurrences of 'France' / 'capital of France': {len(matches)}")
for rel_path, line_num, snippet in matches:
    print(f"{rel_path}:L{line_num} | {snippet}")
