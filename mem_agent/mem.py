import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mem0 import Memory

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 1. Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# 2. Configure Mem0
config = {
    "version": "v1.1",
    "embedder": {
        "provider": "gemini",
        "config": {
            "api_key": GEMINI_API_KEY,
            "model": "gemini-embedding-001",
        },
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "api_key": GEMINI_API_KEY,
            "model": "gemini-3.6-flash",
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "uri": os.getenv("NEO_CONNECTION_URI"),
            "username": os.getenv("NEO_USERNAME"),
            "password": os.getenv("NEO_PASSWORD"),
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768,
        },
    },
}

mem_client = Memory.from_config(config)

USER_ID = "abirmondal"

while True:
    user_query = input("\n> ")
    if user_query.strip().lower() in ["exit", "quit"]:
        break

    # 3. Search memories using the filter dictionary
    search_response = mem_client.search(
        query=user_query,
        filters={"user_id": USER_ID}
    )

    raw_results = (
        search_response.get("results", [])
        if isinstance(search_response, dict)
        else search_response
    )

    memories = [
        item["memory"]
        for item in raw_results
        if isinstance(item, dict) and item.get("memory")
    ]

    print(f"Found {len(memories)} relevant memories.")

    # 4. Formulate System Instruction with contextual memory
    system_instruction = (
        f"You are a helpful assistant. Use these relevant memories about the user to personalize your answer:\n"
        f"{json.dumps(memories, indent=2)}"
        if memories
        else "You are a helpful assistant."
    )

    # 5. Generate content using Gemini
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_query)]
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction
        )
    )

    ai_response = response.text
    print("\nAI:", ai_response)

    # 6. Save interaction to Mem0
    mem_client.add(
        messages=[
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": ai_response},
        ],
        user_id=USER_ID,
    )
    print("Memory updated.")