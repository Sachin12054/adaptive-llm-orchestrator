import os

frontend_src = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator\frontend\src"

matches = []
for root, _, files in os.walk(frontend_src):
    for f in files:
        if f.endswith(".jsx") or f.endswith(".js"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as fc:
                lines = fc.readlines()
                for line_num, line in enumerate(lines, 1):
                    if "isComplex" in line:
                        matches.append((os.path.relpath(path, frontend_src), line_num, line.strip()))

print(f"Total occurrences of 'isComplex': {len(matches)}")
for rel_path, line_num, snippet in matches:
    print(f"{rel_path}:L{line_num} | {snippet}")
