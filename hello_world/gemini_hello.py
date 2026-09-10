import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client=genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

response=client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Hi There! I am Abir. Nice to meet you.",
)

print(response.text)