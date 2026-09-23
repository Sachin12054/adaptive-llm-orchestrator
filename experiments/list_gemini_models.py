import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
api_key = os.getenv("GEMINI_API_KEY")

from google import genai
client = genai.Client(api_key=api_key)

try:
    print("Listing models via client.models.list():")
    for m in client.models.list():
        print(f" - {m.name}")
except Exception as e:
    print(f"Error listing models: {type(e).__name__}: {e}")
