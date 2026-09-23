import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
api_key = os.getenv("GEMINI_API_KEY")

from google import genai
client = genai.Client(api_key=api_key)

active_candidates = [
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash"
]

for m in active_candidates:
    try:
        res = client.models.generate_content(model=m, contents="Hello")
        text = res.text.strip().replace("\n", " ") if res.text else ""
        print(f"Model '{m}': SUCCESS -> {text[:50]}")
    except Exception as e:
        print(f"Model '{m}': FAILED -> {type(e).__name__}: {str(e)[:150]}")
