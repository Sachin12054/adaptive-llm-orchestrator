import os
import shutil
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
buf_path = os.path.join(project_root, "data", "rl", "experience_buffer.jsonl")
archive_path = os.path.join(project_root, "data", "rl", "experience_buffer_gemini_archive.jsonl")

if os.path.exists(buf_path):
    # Read existing entries
    has_gemini = False
    with open(buf_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if "gemini" in str(data.get("action_model_id", "")).lower():
                        has_gemini = True
                        break
                except Exception:
                    pass

    if has_gemini:
        shutil.copyfile(buf_path, archive_path)
        print(f"Archived Gemini experience buffer to '{archive_path}'.")
        with open(buf_path, "w", encoding="utf-8") as f:
            pass
        print(f"Reset fresh Ollama-only buffer at '{buf_path}'.")
    else:
        print(f"Buffer at '{buf_path}' contains only local Ollama entries.")
else:
    os.makedirs(os.path.dirname(buf_path), exist_ok=True)
    with open(buf_path, "w", encoding="utf-8") as f:
        pass
    print(f"Created fresh buffer at '{buf_path}'.")
