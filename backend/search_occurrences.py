import os

backend_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(backend_dir, ".."))

search_paths = [
    os.path.join(backend_dir, "app"),
    os.path.join(backend_dir, "tests"),
    os.path.join(project_root, "datasets"),
    os.path.join(project_root, "frontend", "src")
]

targets = ["mistral-small-latest", "gemini-3.5-flash", "default_model", "fallback_model"]

for target in targets:
    print(f"\n=================== SEARCH RESULTS FOR: '{target}' ===================")
    matches = []
    for sp in search_paths:
        if not os.path.exists(sp):
            continue
        for root, dirs, files in os.walk(sp):
            for file in files:
                if file.endswith(('.py', '.json', '.jsx', '.js', '.ts', '.tsx')):
                    path = os.path.join(root, file)
                    rel = os.path.relpath(path, project_root)
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_no, line in enumerate(f, 1):
                            if target in line:
                                matches.append((rel, line_no, line.strip()))
    for rel, line_no, line in matches:
        print(f"  {rel}:{line_no}: {line}")
