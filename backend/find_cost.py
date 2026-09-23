import os
import re

search_dirs = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "app")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
]

matches = []
for search_dir in search_dirs:
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith(('.py', '.json')):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_no, line in enumerate(f, 1):
                        if re.search(r'\b(cost|price|pricing|token_cost|input_cost|output_cost)\b', line, re.IGNORECASE):
                            matches.append((file, line_no, line.strip()))

print(f"Total occurrences in backend/app and datasets: {len(matches)}")
for file, line_no, line in matches:
    print(f"{file}:{line_no}: {line}")
