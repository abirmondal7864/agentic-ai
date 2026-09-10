import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client= OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/",
)

response=client.chat.completions.create(
    model="gemini-3.6-flash",
    messages=[
        {"role":"system","content": "You are maths expert and you answer only questionsrelated to math"},
        {"role":"user","content": "Hey There! I am Abir. Whats 2+2?"}
    ]
)
print(response.choices[0].message.content)