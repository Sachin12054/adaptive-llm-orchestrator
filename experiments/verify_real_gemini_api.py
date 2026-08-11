import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.core.config import settings

print("==================================================")
print("  REAL GOOGLE GEMINI API MODEL AVAILABILITY TEST")
print("==================================================")
print("GEMINI_API_KEY configured:", bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip()))

if not settings.GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY is not configured in settings!")
    sys.exit(1)

candidate_models = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.5-pro"
]

successful_models = []
failed_models = {}

try:
    from google import genai
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    # Also list available models from API if possible
    print("\nListing models via client.models.list()...")
    try:
        api_models = [m.name for m in client.models.list()]
        print(f"Discovered {len(api_models)} models from API list:")
        for m_name in api_models:
            if "gemini" in m_name.lower():
                print(f" - {m_name}")
    except Exception as list_err:
        print("client.models.list() error:", list_err)

    print("\nTesting generate_content across candidate models:")
    for model_id in candidate_models:
        try:
            res = client.models.generate_content(model=model_id, contents="Reply with 'OK'")
            text = res.text if hasattr(res, "text") else None
            if text and text.strip():
                print(f" [PASS] '{model_id}': SUCCESS! Text response: \"{text.strip()}\"")
                successful_models.append(model_id)
            else:
                print(f" [FAIL] '{model_id}': Empty text response.")
                failed_models[model_id] = "Empty response text"
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}"
            print(f" [FAIL] '{model_id}': {err_msg}")
            failed_models[model_id] = err_msg

except Exception as sdk_err:
    print("Gemini SDK Initialization Error:", sdk_err)

print("\n" + "="*80)
print(f"Successful Executable Models: {successful_models}")
print(f"Failed Models: {list(failed_models.keys())}")
print("="*80)
