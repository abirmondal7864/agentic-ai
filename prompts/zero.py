# Zero-shot prompting
import os 
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client= OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/",
)

SYSTEM_PROMPT="you should only and only ans the coding related questions. Do not ans anything else. Your name is CodeBaba. If user asks something other than coding, just say sorry."

response=client.chat.completions.create(
    model="gemini-3.6-flash",
    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user", "content": "Hey can u write a code to translate hello to Hindi?"}
    ]
)
print(response.choices[0].message.content)
# Zero-shot prompting: The model is given a direct question or task without prior examples