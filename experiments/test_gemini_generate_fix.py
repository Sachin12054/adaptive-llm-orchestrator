import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
api_key = os.getenv("GEMINI_API_KEY")

from google import genai
client = genai.Client(api_key=api_key)

for m in ["gemini-2.5-flash", "models/gemini-2.5-flash"]:
    try:
        res = client.models.generate_content(model=m, contents="Hello")
        text = res.text.strip().replace("\n", " ") if res.text else ""
        print(f"Model '{m}': SUCCESS -> {text[:50]}")
    except Exception as e:
        print(f"Model '{m}': FAILED -> {type(e).__name__}: {str(e)[:150]}")
