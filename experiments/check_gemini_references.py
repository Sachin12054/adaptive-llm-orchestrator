import os

project_root = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator"

matches = []
for root, _, files in os.walk(project_root):
    if "node_modules" in root or ".git" in root or "dist" in root:
        continue
    for f in files:
        if f.endswith(".py") or f.endswith(".json") or f.endswith(".jsx") or f.endswith(".js"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8", errors="ignore") as fc:
                for line_num, line in enumerate(fc.readlines(), 1):
                    if "gemini-2.5-flash" in line:
                        matches.append((os.path.relpath(path, project_root), line_num, line.strip()))

print(f"Total occurrences of 'gemini-2.5-flash': {len(matches)}")
for rel_path, line_num, snippet in matches:
    print(f"{rel_path}:L{line_num} | {snippet}")
