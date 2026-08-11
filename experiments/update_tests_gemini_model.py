import os

tests_dir = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator\backend\tests"

updated_files = []
for root, _, files in os.walk(tests_dir):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as fc:
                content = fc.read()
            if "gemini-2.5-flash" in content:
                new_content = content.replace("gemini-2.5-flash", "gemini-2.0-flash")
                with open(path, "w", encoding="utf-8") as fc:
                    fc.write(new_content)
                updated_files.append(f)

print(f"Updated {len(updated_files)} test files: {updated_files}")
