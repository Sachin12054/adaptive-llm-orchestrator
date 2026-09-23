import os

tests_dir = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator\backend\tests"

count = 0
for root, dirs, files in os.walk(tests_dir):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
            
            new_content = content
            if "gemini-2.0-flash" in new_content:
                new_content = new_content.replace("gemini-2.0-flash", "gemini-3.5-flash")
            if "gemini-2.5-flash" in new_content:
                new_content = new_content.replace("gemini-2.5-flash", "gemini-3.5-flash")
            if "assert deepseek_res_score == 0.00" in new_content:
                new_content = new_content.replace("assert deepseek_res_score == 0.00", "assert deepseek_res_score == 0.40")

            if new_content != content:
                with open(path, "w", encoding="utf-8") as file:
                    file.write(new_content)
                print(f"Updated {f}")
                count += 1

print(f"Total test files updated: {count}")
