import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

from beeai_framework.adapters.groq import GroqChatModel
from beeai_framework.agents.react import ReActAgent
from beeai_framework.memory import UnconstrainedMemory
import logging

logging.basicConfig(level=logging.INFO)

async def test_groq():
    llm = GroqChatModel(
        model_id="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    agent = ReActAgent(
        llm=llm,
        tools=[],
        memory=UnconstrainedMemory(),
    )
    print("Running agent...")
    try:
        response = await agent.run("Hello, who are you?")
        print("Success!", response)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_groq())
