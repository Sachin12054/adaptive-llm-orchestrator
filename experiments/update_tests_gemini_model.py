import os

tests_dir = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator\backend\tests"

count = 0
for root, dirs, files in os.walk(tests_dir):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
            if "gemini-2.5-flash" in content:
                new_content = content.replace("gemini-2.5-flash", "gemini-3.5-flash")
                with open(path, "w", encoding="utf-8") as file:
                    file.write(new_content)
                print(f"Updated {f}")
                count += 1

print(f"Total test files updated: {count}")
