import os

frontend_src = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator\frontend\src"

hooks = ["useState", "useEffect", "useRef", "useMemo", "useCallback", "useContext", "useReducer"]

missing_imports = []

for root, _, files in os.walk(frontend_src):
    for f in files:
        if f.endswith(".jsx") or f.endswith(".js"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as fc:
                content = fc.read()
                lines = content.split("\n")
                
            first_10 = "\n".join(lines[:15])
            for hook in hooks:
                if hook in content and hook not in first_10:
                    rel_path = os.path.relpath(path, frontend_src)
                    missing_imports.append((rel_path, hook))

print(f"Total potential missing hook imports: {len(missing_imports)}")
for rel_path, hook in missing_imports:
    print(f"File: {rel_path} | Hook: '{hook}' used but not in top imports!")

