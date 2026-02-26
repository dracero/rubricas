import asyncio
from dotenv import load_dotenv
import os

from agents.greeter.app.agent import GreetingAgent

load_dotenv()

async def test_greeter():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env")
        return

    print("🧪 Testing Greeter Agent with LangGraph...")
    
    agent = GreetingAgent(api_key=api_key)
    
    try:
        response = agent.invoke("Hola, ¿qué haces?")
        print(f"✅ Response received:\n{response}")
    except Exception as e:
        print(f"❌ Error during invocation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_greeter())
