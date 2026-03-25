from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

print("🔍 Listing all models...")
for model in client.models.list():
    methods = model.supported_generation_methods if hasattr(model, 'supported_generation_methods') else []
    print(f"Model: {model.name}")
    # print(f"  Methods: {methods}")
