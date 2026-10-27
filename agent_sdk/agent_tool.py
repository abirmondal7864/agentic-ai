import asyncio
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

load_dotenv()

# 1. Specialized translation agents
spanish_agent = Agent(
    name="spanish_agent",
    model="gemini-3.6-flash",
    description="Translates input text into Spanish.",
    instruction="You translate the user's message to Spanish. Return only the translated text.",
)

french_agent = Agent(
    name="french_agent",
    model="gemini-3.6-flash",
    description="Translates input text into French.",
    instruction="You translate the user's message to French. Return only the translated text.",
)

# 2. Orchestrator using AgentTool
orchestrator_agent = Agent(
    name="orchestrator_agent",
    model="gemini-3.6-flash",
    instruction=(
        "You are a translation agent. You use the tools given to you to translate. "
        "If asked for multiple translations, you call the relevant tools."
    ),
    tools=[
        AgentTool(agent=spanish_agent),
        AgentTool(agent=french_agent),
    ],
)


async def main():
    runner = InMemoryRunner(agent=orchestrator_agent, app_name="translation_app")
    session = await runner.session_service.create_session(
        app_name="translation_app",
        user_id="user_1",
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Say 'Hello, how are you?' in Spanish and French")],
    )

    print("Final result: ", end="")
    async for event in runner.run_async(
        user_id="user_1",
        session_id=session.id,
        new_message=user_message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())