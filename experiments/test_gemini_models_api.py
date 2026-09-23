import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
api_key = os.getenv("GEMINI_API_KEY")

from google import genai
client = genai.Client(api_key=api_key)

models_to_test = ["gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]
for m in models_to_test:
    try:
        res = client.models.generate_content(model=m, contents="Hello")
        text = res.text.strip().replace("\n", " ") if res.text else ""
        print(f"Model '{m}': SUCCESS -> {text[:40]}")
    except Exception as e:
        print(f"Model '{m}': FAILED -> {type(e).__name__}: {str(e)[:100]}")
