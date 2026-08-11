import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

# Force reload settings from updated .env
from dotenv import load_dotenv
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path, override=True)

from app.core.config import settings

new_key = os.getenv("GEMINI_API_KEY")
print("==================================================")
print("  TESTING NEW GEMINI API KEY")
print("==================================================")
print("Key length:", len(new_key) if new_key else 0)
print("Key preview:", f"{new_key[:8]}...{new_key[-4:]}" if new_key else "None")

models_to_test = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-flash"
]

try:
    from google import genai
    client = genai.Client(api_key=new_key)
    
    for m in models_to_test:
        try:
            res = client.models.generate_content(model=m, contents="Say hello in one word.")
            txt = res.text if hasattr(res, "text") else None
            print(f" -> Model '{m}': SUCCESS! Response: \"{txt.strip() if txt else ''}\"")
        except Exception as e:
            print(f" -> Model '{m}': FAILED ({type(e).__name__}: {e})")

except Exception as sdk_err:
    print("Google genai SDK Error:", sdk_err)

