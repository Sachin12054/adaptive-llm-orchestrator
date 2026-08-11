import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.core.config import settings

print("GEMINI_API_KEY present:", bool(settings.GEMINI_API_KEY))

candidate_gemini_models = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-pro"
]

try:
    from google import genai
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    print("\nAttempting Google genai Client calls:")
    for m in candidate_gemini_models:
        try:
            res = client.models.generate_content(model=m, contents="Say hello")
            print(f" -> Model '{m}': SUCCESS! Response text length: {len(res.text)}")
        except Exception as e:
            print(f" -> Model '{m}': FAILED ({type(e).__name__}: {e})")
except Exception as e:
    print("Google genai import or client init error:", e)
