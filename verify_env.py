from dotenv import load_dotenv
import os

try:
    load_dotenv()
    print("Successfully loaded .env file.")
    print(f"EMBEDDING_DEVICE: {os.getenv('EMBEDDING_DEVICE')}")
    print(f"LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING')}")
except Exception as e:
    print(f"Failed to load .env file: {e}")
