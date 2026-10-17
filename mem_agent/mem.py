import os
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
            "model": "gemini-flash-latest",
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

# 3. Take User Input
user_query = input("> ")

# 4. Generate response using Gemini
messages = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_query)]
    )
]

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=messages
)

ai_response = response.text
print("AI:", ai_response)

# 5. Save conversation into Mem0 (Mem0 uses 'content', not 'text')
mem_client.add(
    user_id="abirmondal",
    messages=[
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": ai_response},
    ],
)

print("Memory has been saved...")