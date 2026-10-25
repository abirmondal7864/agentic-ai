import asyncio
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

load_dotenv()

# 1. Define the agent
hello_agent = Agent(
    name="hello_world_agent",
    model="gemini-3.6-flash",
    instruction="You are an agent which greets the user and helps them ans using emojis and in a funny way.",
)

async def main():
    # 2. Setup InMemoryRunner & session
    runner = InMemoryRunner(agent=hello_agent, app_name="greeting_app")
    session = await runner.session_service.create_session(
        app_name="greeting_app",
        user_id="user_1",
    )

    # 3. Format message
    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Hey there, my name is Abir")]
    )

    # 4. Stream and collect output events
    async for event in runner.run_async(
        user_id="user_1",
        session_id=session.id,
        new_message=user_message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text, end="")
    print()

if __name__ == "__main__":
    asyncio.run(main())